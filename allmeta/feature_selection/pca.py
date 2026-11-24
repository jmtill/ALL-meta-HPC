"""Principal components as model inputs: global PCs and per-cluster PC1 factors.

Two of the five feature representations are built here (implementation plan, Steps 2.3
and 2.4), and the machinery is shared because they differ only in which columns go into
each PCA: the whole descriptor set, or one cluster at a time.

Every function follows a fit-loadings / apply-loadings pattern rather than returning a
fitted sklearn object. The reason is the shape of this project's data. A factor must be
*defined* on the wide unlabelled space -- 732 curated liquids in subset B -- and then
*applied* to the 13 labelled ones, because a correlation structure estimated on 13
liquids is noise. Persisting the centre, scale and loading of each source feature makes
that carry-over explicit and reproducible, and lets the same factor be recomputed on
any frame months later without refitting anything.

Standardising before PCA is required, not cosmetic: PC1 of unstandardised columns is
whichever descriptor has the largest units, so a cluster holding MWt and a Hansen
parameter would simply return MWt.
"""

import numpy as np
from pandas import DataFrame, Series
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from allmeta.feature_selection.provenance import FeatureSpec
from allmeta.paths import RANDOM_STATE

# One row per (factor, source feature). Matches the schema already used by
# allmeta.screening's factor tables so the two can be concatenated and compared.
LOADING_COLUMNS: tuple[str, ...] = (
    'factor', 'feature', 'loading', 'center', 'scale', 'cluster', 'n_members',
    'explained_variance_ratio',
)

GLOBAL_PREFIX: str = 'global_PC'


def _orient(weights: Series) -> Series:
    """Fix PC1's arbitrary sign so a factor is reproducible run to run.

    sklearn may return a component or its negation depending on the LAPACK path taken.
    Orienting the largest-magnitude loading positive makes the column stable across
    runs and its sign interpretable: the factor increases with its dominant descriptor.
    """
    return -weights if weights[weights.abs().idxmax()] < 0 else weights


def _passthrough_row(factor: str, feature: str, cluster: object,
                     centre: float, scale: float) -> dict:
    """Loading row for a one-member cluster.

    A single feature has no component to extract. It is standardised and passed
    through, and recorded with ``explained_variance_ratio`` of 1.0 because the factor
    genuinely retains all of the one thing it was given. Inventing a one-column PCA
    here would obscure that rather than clarify it. The row still carries the cluster's
    name as its factor, so a caller gets one column per cluster whatever the sizes are.
    """
    return {
        'factor': factor, 'feature': feature, 'loading': 1.0,
        'center': centre, 'scale': scale, 'cluster': cluster,
        'n_members': 1, 'explained_variance_ratio': 1.0,
    }


def fit_global_pca(frame: DataFrame, features: list[str],
                   n_components: float | int = 0.95,
                   ) -> tuple[DataFrame, DataFrame, DataFrame]:
    """PCA of the full raw feature set (Step 2.3).

    Returns ``(values, loadings, variance)``. Components are named ``global_PC1`` up,
    one-based, so the names match the way scree plots and the literature index them.

    ``n_components`` as a float in (0, 1) is read by sklearn as a cumulative
    explained-variance target and the component count falls out of the data; as an int
    it is the count itself.

    The scaler and the PCA are fitted only on ``frame``. Applying the result to held-out
    rows goes through ``apply_loadings``, which uses the persisted constants and fits
    nothing -- that is what keeps the held-out rows held out.
    """
    block = frame[features].astype(float)

    scaler = StandardScaler().fit(block)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    scores = pca.fit_transform(scaler.transform(block))

    names = [f'{GLOBAL_PREFIX}{i}' for i in range(1, pca.n_components_ + 1)]
    values = DataFrame(scores, index=frame.index, columns=names)

    ratios = pca.explained_variance_ratio_
    rows: list[dict] = []
    for position, name in enumerate(names):
        weights = Series(pca.components_[position], index=features)
        for feature in features:
            rows.append({
                'factor': name, 'feature': feature,
                'loading': float(weights[feature]),
                'center': float(scaler.mean_[features.index(feature)]),
                'scale': float(scaler.scale_[features.index(feature)]),
                'cluster': name, 'n_members': len(features),
                'explained_variance_ratio': float(ratios[position]),
            })

    variance = DataFrame({
        'component': names,
        'explained_variance': pca.explained_variance_.astype(float),
        'explained_variance_ratio': ratios.astype(float),
        'cumulative_explained_variance_ratio': np.cumsum(ratios).astype(float),
        'n_source_features': len(features),
    })

    return values, DataFrame(rows, columns=list(LOADING_COLUMNS)), variance


def fit_cluster_pc1(frame: DataFrame, clusters: dict[str, list[str]],
                    ) -> tuple[DataFrame, DataFrame]:
    """One standardised first principal component per feature cluster (Step 2.4).

    ``clusters`` maps a cluster name to its member descriptors, so the same function
    serves the hand-defined grouping in ``config.MANUAL_CLUSTERS`` and, later, whatever
    partition an automatic clustering algorithm produces. Nothing here knows or cares
    which it was handed.

    Returns ``(values, loadings)``. Cluster order is the mapping's order, so a
    deliberate ordering is preserved; column names are the cluster names, which is what
    makes a coefficient table readable ('hansen' rather than 'factor_3').

    A singleton cluster is standardised and passed through -- see ``_passthrough_rows``.
    """
    values: dict[str, Series] = {}
    rows: list[dict] = []

    for position, (name, members) in enumerate(clusters.items()):
        if not members:
            raise ValueError(f'cluster {name!r} is empty')

        missing = [f for f in members if f not in frame.columns]
        if missing:
            raise ValueError(f'cluster {name!r} names columns absent from the frame: {missing}')

        block = frame[list(members)].astype(float)
        scaler = StandardScaler().fit(block)
        standardised = DataFrame(scaler.transform(block), index=frame.index,
                                 columns=list(members))

        if len(members) == 1:
            feature = members[0]
            values[name] = standardised[feature]
            rows.append(_passthrough_row(name, feature, position,
                                         float(scaler.mean_[0]),
                                         float(scaler.scale_[0])))
            continue

        pca = PCA(n_components=1, random_state=RANDOM_STATE).fit(standardised)
        weights = _orient(Series(pca.components_[0], index=list(members)))
        values[name] = (standardised * weights).sum(axis=1)

        ratio = float(pca.explained_variance_ratio_[0])
        for feature in members:
            index = list(members).index(feature)
            rows.append({
                'factor': name, 'feature': feature,
                'loading': float(weights[feature]),
                'center': float(scaler.mean_[index]),
                'scale': float(scaler.scale_[index]),
                'cluster': position, 'n_members': len(members),
                'explained_variance_ratio': ratio,
            })

    return (DataFrame(values, index=frame.index),
            DataFrame(rows, columns=list(LOADING_COLUMNS)))


def apply_loadings(frame: DataFrame, loadings: DataFrame) -> DataFrame:
    """Recompute factor scores on a new frame from persisted loadings.

    The half of the fit/apply split that keeps held-out data held out: nothing is
    fitted, the centres and scales come from the frame the factors were defined on, and
    the result therefore means the same thing on the 13 labelled liquids as it does on
    the 732 curated ones.
    """
    values: dict[str, Series] = {}

    for name, group in loadings.groupby('factor', sort=False):
        members = list(group['feature'])
        missing = [f for f in members if f not in frame.columns]
        if missing:
            raise ValueError(f'factor {name!r} needs columns absent from the frame: {missing}')

        block = frame[members].astype(float)
        centred = (block - group['center'].to_numpy()) / group['scale'].to_numpy()
        values[str(name)] = (centred * group['loading'].to_numpy()).sum(axis=1)

    return DataFrame(values, index=frame.index)


def global_pca_specs(loadings: DataFrame) -> list[FeatureSpec]:
    """Provenance specs for the components of a global PCA."""
    return _specs(loadings, feature_type='global_pca', source_method='global_pca',
                  clustering_method=None, named_clusters=False)


def cluster_pc1_specs(loadings: DataFrame, source_method: str,
                      clustering_method: str | None = None,
                      feature_type: str = 'cluster_pc1') -> list[FeatureSpec]:
    """Provenance specs for per-cluster PC1 factors.

    ``feature_type`` separates ``'manual_cluster_pc1'`` from ``'cluster_pc1'`` so a
    later comparison can ask whether the expert grouping or the algorithm's did better,
    which one shared label would make impossible.
    """
    return _specs(loadings, feature_type=feature_type, source_method=source_method,
                  clustering_method=clustering_method, named_clusters=True)


def _specs(loadings: DataFrame, feature_type: str, source_method: str,
           clustering_method: str | None, named_clusters: bool) -> list[FeatureSpec]:
    """Turn a loadings table into provenance specs, one per factor."""
    specs: list[FeatureSpec] = []

    for name, group in loadings.groupby('factor', sort=False):
        cluster = group['cluster'].iloc[0]
        specs.append(FeatureSpec(
            feature_name=str(name),
            feature_type=feature_type,  # type: ignore[arg-type]
            source_features=tuple(group['feature']),
            source_method=source_method,
            clustering_method=clustering_method,
            cluster_id=int(cluster) if isinstance(cluster, (int, np.integer)) else None,
            cluster_name=str(name) if named_clusters else None,
            explained_variance_ratio=float(group['explained_variance_ratio'].iloc[0]),
        ))

    return specs
