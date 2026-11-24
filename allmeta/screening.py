"""Unsupervised feature screening: correlation structure, clustering, factors.

Nothing here looks at the dispensing-error label, so these screens apply equally to
subsets A and B, where no label exists. Every stage emits an audit row per feature
rather than only the survivors, so a threshold can be re-argued from the written
output without rerunning.

``columns.PROTECTED`` (viscosity) survives every stage by construction.

Two changes from the original design, both made because the screens were measurably
worse than doing nothing:

**The variance screen is gone.** It min-max scaled and cut below 0.005, but variance
after scaling is exactly ``(std / denominator)**2`` -- it measures the scaler, not the
feature. Min-max divides by the range, so one extreme liquid drives the score to zero
and the screen cuts features *with outliers*; RobustScaler divides by the IQR and cuts
the opposite set (the two rankings correlate at rho = -0.755); StandardScaler divides
by the standard deviation and scores every feature identically 1.0. On subset A the
min-max version cut thirteen descriptors that each had 150-940 distinct values, among
them ``AntC``, ``EdmiW`` and ``LogS`` -- three of the features the supervised side
selects -- and ``log η``, which survived only because it is hard-coded PROTECTED.
``variance_ranking`` still runs and is still plotted, as a diagnostic.

**Clusters are aggregated, not sampled.** Keeping one representative per correlated
group throws away the averaging that made the group informative: six descriptors at
|rho| > 0.9 measure one physical factor, and their mean is a better estimate of it
than any single member. Worse, choosing representatives systematically prefers noise
-- a feature is uncorrelated with everything else precisely when it is idiosyncratic,
so the most independent candidates seed the kept list and block the coherent groups
behind them. Measured on subset B, ``spearman_0.7`` kept features whose median
relevance to the target was 0.411 while dropping features at 0.626. ``build_factors``
emits each cluster's first principal component instead; the representative-picking
sets are still produced so the comparison stays visible, but they are not the
recommendation.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import FeatureAgglomeration, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from allmeta import columns as cols  # noqa: E402
from allmeta.paths import (  # noqa: E402
    FEATURE_SELECTION_DIR,
    FIG_FEATURE_SELECTION,
    RANDOM_STATE,
)
from allmeta.subsets import load_subset  # noqa: E402

VARIANCE_THRESHOLD: float = 0.005
SPEARMAN_THRESHOLDS: tuple[float, ...] = (0.9, 0.7)
CLUSTER_METHODS: tuple[str, ...] = ('hierarchical', 'spectral', 'agglomerative')

# Short tags used to namespace factor columns, so a factor name records which
# clustering produced it without the column becoming unreadable.
METHOD_TAGS: dict[str, str] = {
    'hierarchical': 'hier', 'spectral': 'spec', 'agglomerative': 'aggl',
}
K_RANGE: tuple[int, ...] = tuple(range(4, 31))

AUDIT_COLUMNS: tuple[str, ...] = ('feature', 'kept', 'stage_dropped', 'reason')


def correlation_matrix(frame: DataFrame, method: str = 'spearman') -> DataFrame:
    """Absolute rank-correlation between features.

    Spearman rather than Pearson because several descriptors are strongly
    monotonic but not linear in each other (Tc against BPt, Parachor against MVol).
    """
    return frame.corr(method=method).abs()


def correlation_distance(rho: DataFrame) -> np.ndarray:
    """Square distance matrix ``1 - |rho|``, symmetrised with a zero diagonal.

    Floating-point asymmetry in the correlation matrix is enough to make scipy
    reject the condensed form, so both are forced here rather than at each call.
    """
    dist = (1.0 - rho).to_numpy(dtype=float)
    np.fill_diagonal(dist, 0.0)

    return (dist + dist.T) / 2.0


def pick_representative(members: list[str], rho: DataFrame,
                        protected: tuple[str, ...] = cols.PROTECTED) -> str:
    """One feature to stand for a group of mutually correlated features.

    A protected feature wins outright. Otherwise the medoid -- the member with the
    highest mean absolute correlation to the others -- because it is the member that
    best reproduces the information the discarded ones carried.
    """
    for name in protected:
        if name in members:
            return name

    if len(members) == 1:
        return members[0]

    centrality: Series = rho.loc[members, members].mean(axis=1)

    return str(centrality.idxmax())


def _audit(features: list[str], kept: list[str], stage: str,
           reasons: dict[str, str], prior: DataFrame | None = None) -> DataFrame:
    """Assemble the per-feature audit frame for one stage.

    ``prior`` carries the drops made by earlier stages so that every audit file
    accounts for all 58 descriptors, not just the ones its own stage saw. Without it
    a reader of ``selected_spearman_0.9.csv`` could not tell whether an absent
    feature was dropped here or upstream.
    """
    kept_set = set(kept)
    rows = [
        {
            'feature': feature,
            'kept': feature in kept_set,
            'stage_dropped': '' if feature in kept_set else stage,
            'reason': '' if feature in kept_set else reasons.get(feature, stage),
        }
        for feature in features
    ]
    audit = pd.DataFrame(rows, columns=list(AUDIT_COLUMNS))

    if prior is None or prior.empty:
        return audit

    carried = prior.loc[~prior['feature'].isin(features)].copy()
    carried['kept'] = False

    return pd.concat([audit, carried], ignore_index=True)


def minmax_variance(frame: DataFrame) -> Series:
    """Variance after min-max scaling, ascending. Diagnostic only.

    Retained because the ranking is worth *looking* at -- it identifies the columns
    whose distribution is dominated by one or two extreme liquids, which is useful to
    know. It is not a selection criterion: the statistic is
    ``(std / (max - min))**2``, so it ranks tail shape, and the features it scores
    lowest here are exactly the ones the supervised selection scores highest.
    """
    scaled = DataFrame(
        MinMaxScaler().fit_transform(frame),
        columns=frame.columns,
        index=frame.index,
    )

    return scaled.var().sort_values()


def variance_screen(frame: DataFrame, threshold: float = VARIANCE_THRESHOLD,
                    protected: tuple[str, ...] = cols.PROTECTED,
                    prior: DataFrame | None = None
                    ) -> tuple[list[str], DataFrame, Series]:
    """Drop features whose variance after min-max scaling falls below ``threshold``.

    Min-max first so the comparison is scale-free: raw variance would rank MWt above
    every Hansen parameter purely because of units.

    Returns the survivors, the audit frame, and the full variance ranking so the cut
    can be revisited without recomputing.

    **No longer used by** ``screen_subset``; kept so the original pipeline can be
    reproduced for comparison. See the module docstring for why it was retired.
    """
    variances: Series = minmax_variance(frame)

    kept = [f for f in frame.columns if variances[f] >= threshold or f in protected]
    reasons = {
        f: f'min-max variance {variances[f]:.6f} < {threshold}'
        for f in frame.columns if f not in kept
    }

    return kept, _audit(list(frame.columns), kept, 'variance', reasons, prior), variances


def spearman_screen(frame: DataFrame, threshold: float,
                    protected: tuple[str, ...] = cols.PROTECTED,
                    prior: DataFrame | None = None
                    ) -> tuple[list[str], DataFrame, DataFrame]:
    """Greedily drop features correlated above ``threshold`` with an already-kept one.

    Greedy rather than connected components: components chain, so with a 0.7 cut a
    long ladder of pairwise-correlated features collapses to a single representative
    even when its ends are uncorrelated. Candidates are considered most-independent
    first (lowest mean |rho|), after the protected features are seeded.

    Also returns the surviving pairs above threshold, which should be empty, and the
    dropped-to-kept mapping used in the audit.
    """
    rho = correlation_matrix(frame)
    mean_rho: Series = rho.mean(axis=1).sort_values()

    kept: list[str] = [f for f in protected if f in frame.columns]
    reasons: dict[str, str] = {}

    for feature in mean_rho.index:
        if feature in kept:
            continue
        if not kept:
            kept.append(feature)
            continue

        against: Series = rho.loc[feature, kept]
        worst = str(against.idxmax())
        if against.max() > threshold:
            reasons[feature] = f'|rho|={against.max():.3f} with kept {worst}'
        else:
            kept.append(feature)

    survivors = rho.loc[kept, kept]
    mask = np.triu(np.ones(survivors.shape), k=1).astype(bool)
    remaining_pairs = (
        survivors.where(mask).stack().reset_index()
        .set_axis(['feature_1', 'feature_2', 'corr_abs'], axis=1)
        .query('corr_abs > @threshold')
    )

    audit = _audit(list(frame.columns), kept, f'spearman_{threshold}', reasons, prior)

    return kept, audit, remaining_pairs


def _cluster_features(dist: np.ndarray, rho: DataFrame, method: str,
                      n_clusters: int) -> np.ndarray:
    """Partition features into ``n_clusters`` groups under one clustering method.

    All three consume the precomputed correlation distance. Ward linkage is
    deliberately not offered: it minimises Euclidean inertia and is undefined for a
    precomputed non-Euclidean distance, so average linkage is used instead.
    """
    if method == 'hierarchical':
        linkage_matrix = linkage(squareform(dist, checks=False), method='average')
        return fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

    if method == 'agglomerative':
        model = FeatureAgglomeration(
            n_clusters=n_clusters, metric='precomputed', linkage='average',
        )
        return model.fit(dist).labels_

    if method == 'spectral':
        model = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed',
            random_state=RANDOM_STATE, assign_labels='kmeans',
        )
        return model.fit_predict(rho.to_numpy())

    raise ValueError(f'unknown clustering method {method!r}; expected {CLUSTER_METHODS}')


def feature_clusters(frame: DataFrame, method: str,
                     k_range: tuple[int, ...] = K_RANGE
                     ) -> tuple[dict[int, list[str]], DataFrame, int]:
    """Partition features into correlated groups; return membership, sweep and ``k``.

    ``k`` is chosen by the silhouette of the feature partition itself rather than
    fixed by hand, so each method is allowed to pick the granularity its own geometry
    supports. The full sweep is returned so that choice stays visible.

    Split out from ``feature_cluster_screen`` because the membership is what
    ``build_factors`` needs -- the representative-picking that used to be the only
    consumer of these clusters is now one of two things you can do with them, and the
    weaker one.
    """
    features = list(frame.columns)
    rho = correlation_matrix(frame)
    dist = correlation_distance(rho)

    sweep_rows: list[dict] = []
    labels_by_k: dict[int, np.ndarray] = {}

    for n_clusters in k_range:
        if n_clusters >= len(features):
            continue
        labels = _cluster_features(dist, rho, method, n_clusters)
        if len(set(labels)) < 2:
            continue
        labels_by_k[n_clusters] = labels
        sweep_rows.append({
            'method': method,
            'k': n_clusters,
            'n_clusters_found': len(set(labels)),
            'silhouette': float(silhouette_score(dist, labels, metric='precomputed')),
        })

    sweep = pd.DataFrame(sweep_rows)
    if sweep.empty:
        raise ValueError(f'{method} produced no valid partition over k={k_range}')

    best_k = int(sweep.loc[sweep['silhouette'].idxmax(), 'k'])
    sweep['selected_k'] = sweep['k'] == best_k
    labels = labels_by_k[best_k]

    members: dict[int, list[str]] = {}
    for label in sorted(set(labels)):
        members[int(label)] = [f for f, lab in zip(features, labels) if lab == label]

    return members, sweep, best_k


def feature_cluster_screen(frame: DataFrame, method: str,
                           k_range: tuple[int, ...] = K_RANGE,
                           protected: tuple[str, ...] = cols.PROTECTED,
                           prior: DataFrame | None = None
                           ) -> tuple[list[str], DataFrame, DataFrame]:
    """Cluster features by correlation distance and keep one representative per cluster.

    Retained for comparison rather than as the recommendation: picking one member
    discards the noise-averaging that made a correlated group worth having. See
    ``build_factors`` for the aggregating alternative and the module docstring for
    the measurements behind that preference.
    """
    features = list(frame.columns)
    rho = correlation_matrix(frame)

    clusters, sweep, best_k = feature_clusters(frame, method, k_range)

    kept: list[str] = []
    reasons: dict[str, str] = {}
    for label in sorted(clusters):
        members = clusters[label]
        representative = pick_representative(members, rho, protected)
        kept.append(representative)
        for member in members:
            if member != representative:
                reasons[member] = f'cluster {label} represented by {representative}'

    audit = _audit(features, kept, f'{method}_k{best_k}', reasons, prior)

    return kept, audit, sweep


def factor_name(members: list[str], loadings: Series) -> str:
    """Name a factor after its highest-loading member, so the column stays readable.

    ``factor_Ant1T`` says what the column is in a way ``factor_3`` does not, and the
    name is stable as long as the cluster is.
    """
    return f'factor_{loadings.abs().idxmax()}' if len(members) > 1 else members[0]


def fit_factors(frame: DataFrame, clusters: dict[int, list[str]]
                ) -> tuple[DataFrame, DataFrame]:
    """Fit one standardised first principal component per feature cluster.

    Returns ``(values, loadings)``. ``values`` is the factor score per row, indexed
    like ``frame``; ``loadings`` records the weight of each member feature plus the
    centring and scaling constants, so the same factor can be recomputed on another
    frame without refitting -- which is the point, since the definition must come
    from the wide unlabelled space and be *applied* to the labelled liquids.

    Standardising before PCA is required, not cosmetic: PC1 of unstandardised columns
    is whichever descriptor has the largest units, so on a cluster containing MWt and
    a Hansen parameter it would simply return MWt.

    A single-member cluster passes its feature through unchanged; there is nothing to
    average, and inventing a one-column PCA would only obscure that.
    """
    values: dict[str, Series] = {}
    rows: list[dict] = []

    for label in sorted(clusters):
        members = clusters[label]
        block = frame[members]

        if len(members) == 1:
            name = members[0]
            values[name] = block[name].astype(float)
            rows.append({'factor': name, 'feature': name, 'loading': 1.0,
                         'center': 0.0, 'scale': 1.0, 'cluster': label,
                         'n_members': 1, 'explained_variance_ratio': 1.0})
            continue

        scaler = StandardScaler().fit(block)
        pca = PCA(n_components=1, random_state=RANDOM_STATE).fit(scaler.transform(block))

        weights = Series(pca.components_[0], index=members)
        # PC1's sign is arbitrary -- sklearn may return the factor or its negation
        # from run to run. Orient it so the largest-magnitude loading is positive,
        # which makes the column reproducible and its sign interpretable.
        if weights[weights.abs().idxmax()] < 0:
            weights = -weights

        name = factor_name(members, weights)
        centred = (block - scaler.mean_) / scaler.scale_
        values[name] = (centred * weights).sum(axis=1)

        ratio = float(pca.explained_variance_ratio_[0])
        for feature in members:
            rows.append({
                'factor': name, 'feature': feature,
                'loading': float(weights[feature]),
                'center': float(scaler.mean_[members.index(feature)]),
                'scale': float(scaler.scale_[members.index(feature)]),
                'cluster': label, 'n_members': len(members),
                'explained_variance_ratio': ratio,
            })

    return DataFrame(values, index=frame.index), DataFrame(rows)


def apply_factors(frame: DataFrame, loadings: DataFrame) -> DataFrame:
    """Recompute factor scores on a new frame from persisted loadings.

    Used to carry a factor defined on the wide unlabelled space (subset A, 1206
    liquids) onto the thirteen labelled ones without refitting, so the factor means
    the same thing in both places.
    """
    values: dict[str, Series] = {}
    for name, group in loadings.groupby('factor', sort=False):
        block = frame[list(group['feature'])]
        centred = (block - group['center'].to_numpy()) / group['scale'].to_numpy()
        values[name] = (centred * group['loading'].to_numpy()).sum(axis=1)

    return DataFrame(values, index=frame.index)


def plot_variance(variances: Series, threshold: float, out_path: Path) -> None:
    """Horizontal bar of min-max variance per feature with the cut drawn on."""
    fig, ax = plt.subplots(figsize=(9, max(6, len(variances) * 0.22)))
    colours = ['#4c72b0' if v >= threshold else '#c44e52' for v in variances.values]

    ax.barh(variances.index, variances.values, color=colours)
    ax.axvline(threshold, color='k', linestyle='--', linewidth=1,
               label=f'threshold = {threshold}')
    ax.set_xlabel('Variance after min-max scaling')
    ax.set_ylabel('Feature')
    ax.set_title('Feature variance')
    ax.tick_params(axis='y', labelsize=8)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='w')
    plt.close(fig)


def plot_dendrogram(frame: DataFrame, out_path: Path) -> None:
    """Average-linkage dendrogram of features in correlation-distance space."""
    rho = correlation_matrix(frame)
    dist = correlation_distance(rho)
    linkage_matrix = linkage(squareform(dist, checks=False), method='average')

    fig, ax = plt.subplots(figsize=(max(10, len(frame.columns) * 0.28), 7))
    dendrogram(linkage_matrix, labels=list(frame.columns), ax=ax,
               leaf_rotation=90, leaf_font_size=8, color_threshold=0.5)
    ax.set_ylabel('1 - |Spearman rho|')
    ax.set_title('Feature clustering by correlation distance (average linkage)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='w')
    plt.close(fig)


def screen_subset(subset: str) -> dict[str, list[str]]:
    """Run every unsupervised screen on one subset and write the results.

    Returns the named feature sets, which are what the clustering and comparison
    stages consume.
    """
    out_dir = FEATURE_SELECTION_DIR / subset
    fig_dir = FIG_FEATURE_SELECTION / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    frame = load_subset(subset)
    all_features = list(cols.NUM_COLUMNS)
    screening = cols.screening_features()

    feature_sets: dict[str, list[str]] = {
        'all': all_features,
        'no_composition': screening,
        'manual': list(cols.MANUAL_KEEP),
    }

    # Stage 0: composition columns, recorded as their own stage so the drop is
    # attributable rather than folded into the variance screen.
    comp_reasons = {f: 'composition/atom-count column' for f in cols.COMP_COLUMNS}
    audit_comp = _audit(all_features, screening, 'composition', comp_reasons)
    audit_comp.to_csv(out_dir / 'selected_no_composition.csv', index=False)

    data = frame[screening]

    # The variance ranking is still computed and plotted, but no longer gates
    # anything -- see the module docstring. Every later stage now sees all 45
    # descriptors instead of the 32-38 that survived the cut.
    variances = minmax_variance(data)
    variances.rename('minmax_variance').to_csv(out_dir / 'variance_ranking.csv')
    plot_variance(variances, VARIANCE_THRESHOLD, fig_dir / 'feature_variances.png')

    # Redundancy screens, kept so the comparison against the factor sets and against
    # no screening at all stays visible in the written output. Not recommended.
    for threshold in SPEARMAN_THRESHOLDS:
        kept, audit, remaining = spearman_screen(data, threshold, prior=audit_comp)
        tag = f'spearman_{threshold}'
        audit.to_csv(out_dir / f'selected_{tag}.csv', index=False)
        remaining.to_csv(out_dir / f'{tag}_remaining_pairs.csv', index=False)
        feature_sets[tag] = kept

    sweeps: list[DataFrame] = []
    cluster_membership: dict[str, dict[int, list[str]]] = {}
    for method in CLUSTER_METHODS:
        kept, audit, sweep = feature_cluster_screen(data, method, prior=audit_comp)
        audit.to_csv(out_dir / f'selected_fcluster_{method}.csv', index=False)
        sweeps.append(sweep)
        feature_sets[f'fcluster_{method}'] = kept
        cluster_membership[method], _, _ = feature_clusters(data, method)

    pd.concat(sweeps, ignore_index=True).to_csv(out_dir / 'fcluster_k_sweep.csv',
                                                index=False)
    plot_dendrogram(data, fig_dir / 'feature_dendrogram.png')

    # Factors: one aggregated column per cluster, replacing the representative that
    # used to stand in for it. Written per method so the sets stay comparable.
    with open(out_dir / 'feature_clusters.json', 'w') as handle:
        json.dump(cluster_membership, handle, ensure_ascii=False, indent=2)

    for method, clusters in cluster_membership.items():
        values, loadings = fit_factors(data, clusters)
        # Factor columns are namespaced by subset *and* clustering method. Both are
        # needed: A and B are clustered separately, and the three methods disagree
        # about membership, so two different linear combinations can otherwise land
        # on the same name -- hierarchical and spectral both produce a factor topped
        # by Ant1T on subset A, over fourteen and ten descriptors respectively.
        prefix = f'{subset}::{METHOD_TAGS[method]}_'
        rename = {c: f'{prefix}{c}' for c in values.columns}
        loadings['factor'] = loadings['factor'].map(lambda c: f'{prefix}{c}')
        loadings.to_csv(out_dir / f'factor_loadings_{method}.csv', index=False)

        values.columns = [rename[c] for c in values.columns]
        values.insert(0, 'Name', frame['Name'].to_numpy())
        values.to_csv(out_dir / f'factor_values_{method}.csv', index=False)
        feature_sets[f'factor_{method}'] = [c for c in values.columns if c != 'Name']

    with open(out_dir / 'feature_sets.json', 'w') as handle:
        json.dump(feature_sets, handle, ensure_ascii=False, indent=2)

    return feature_sets


def load_feature_sets(subset: str) -> dict[str, list[str]]:
    """Read the named feature sets written by ``screen_subset``."""
    with open(FEATURE_SELECTION_DIR / subset / 'feature_sets.json') as handle:
        return json.load(handle)


def load_factor_loadings(subsets: tuple[str, ...] = ('A', 'B')) -> DataFrame:
    """Every persisted factor definition, concatenated.

    Empty frame if the screens have not been run, so callers can attach factors
    unconditionally without guarding on a first-run bootstrap.
    """
    frames = [
        pd.read_csv(path)
        for subset in subsets
        for method in CLUSTER_METHODS
        for path in [FEATURE_SELECTION_DIR / subset / f'factor_loadings_{method}.csv']
        if path.is_file()
    ]
    if not frames:
        return DataFrame(columns=['factor', 'feature', 'loading', 'center', 'scale'])

    return pd.concat(frames, ignore_index=True)


def attach_derived(frame: DataFrame) -> DataFrame:
    """Add the derived physical descriptors that no single HSPiP column carries.

    Currently just ``logPvap25``. See ``columns.DERIVED_VAPOUR_PRESSURE`` for why
    the three Antoine coefficients should not be selected among individually.
    """
    needed = ('AntA', 'AntB', 'AntC')
    if not all(c in frame.columns for c in needed):
        return frame

    result = frame.copy()
    denominator = cols.REFERENCE_TEMPERATURE_C + result['AntC']
    result[cols.DERIVED_VAPOUR_PRESSURE] = (
        result['AntA'] - result['AntB'] / denominator.where(denominator != 0)
    )

    return result


def attach_factors(frame: DataFrame, subsets: tuple[str, ...] = ('A', 'B')) -> DataFrame:
    """Return ``frame`` with every persisted factor column added.

    The loadings were fitted on the wide unlabelled subsets and are *applied* here,
    so a factor evaluated on the thirteen labelled liquids means exactly what it
    means on the twelve hundred -- the definition never sees the target, and never
    sees only thirteen rows.
    """
    frame = attach_derived(frame)

    loadings = load_factor_loadings(subsets)
    if loadings.empty:
        return frame

    usable = loadings[loadings['feature'].isin(frame.columns)]
    complete = usable.groupby('factor')['feature'].size() == \
        loadings.groupby('factor')['feature'].size()
    usable = usable[usable['factor'].isin(complete[complete].index)]
    if usable.empty:
        return frame

    return pd.concat([frame, apply_factors(frame, usable)], axis=1)


if __name__ == '__main__':
    for subset_name in ('A', 'B'):
        sets = screen_subset(subset_name)
        print(f'\nsubset {subset_name}')
        for set_name, members in sets.items():
            print(f'  {set_name:24s} {len(members):3d}  {", ".join(members[:8])}'
                  + (' ...' if len(members) > 8 else ''))
