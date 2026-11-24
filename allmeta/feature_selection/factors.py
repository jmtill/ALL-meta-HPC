"""PC1 per cluster, for every automatic clustering solution (plan Phase 6).

Turns a partition of the features into model inputs: one standardised first principal
component per cluster, with the loadings, the information it retains, and the provenance
back to the original descriptors. The same construction the manual clusters already go
through in ``pca.fit_cluster_pc1`` -- this module supplies the partitions and records
the quality, it does not reimplement the PCA.

## Why this is cached

The sweep produces 430 solutions holding 7,660 clusters between them, but only **288
distinct member-sets**: a tight family like ``dHDon``/``Abr-a`` is found by almost every
hyperparameter setting of almost every algorithm. PC1 of a given set of descriptors on a
given frame does not depend on which solution asked for it, so each distinct set is
fitted once and reused. That is a ~10x saving on the multi-member clusters, and it also
guarantees that two solutions which found the same family get numerically identical
factors -- fitting twice would leave them differing in the last bits and make a later
equality check quietly unreliable.

## Why the headline summary is not the mean

5,284 of those 7,660 clusters are singletons, and a singleton's PC1 retains 1.0 of its
cluster by construction. So ``mean_pc1_explained_variance`` over all clusters rewards a
solution for declining to cluster, in exactly the way ``within_between_delta`` was found
to. A partition of 44 singletons and one pair scores ~0.99 and has summarised nothing.

Every summary is therefore reported twice, over all clusters and over the multi-member
clusters alone, and ``min_pc1_explained_variance`` is taken over the multi-member ones
only -- a minimum of 1.0 contributed by a singleton is not information. Read the
``_multi`` columns; the all-cluster ones are there to make the inflation visible rather
than to be used.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame

from allmeta.feature_selection import config, pca
from allmeta.feature_selection.algorithms import ClusterSolution, format_params
from allmeta.feature_selection.provenance import FeatureSpec

QUALITY_COLUMNS: tuple[str, ...] = (
    'method', 'hyperparameters', 'cluster_name', 'cluster_id', 'n_members',
    'features', 'explained_variance_ratio', 'is_singleton', 'flagged',
    'top_loading_feature', 'top_loading',
)


@dataclass(frozen=True)
class FactorSet:
    """The cluster-PC1 representation of one clustering solution."""

    method: str
    params: dict
    values: DataFrame
    loadings: DataFrame
    specs: tuple[FeatureSpec, ...]
    quality: DataFrame

    @property
    def n_factors(self) -> int:
        return self.values.shape[1]

    def summary(self) -> dict:
        return {'method': self.method,
                'hyperparameters': format_params(self.params),
                **solution_summary(self.quality)}


class ClusterPCACache:
    """PC1 for each distinct member-set, fitted once against one frame.

    Bound to the frame it was built for. A cache reused across frames would silently
    return factors defined on the wrong liquids, so the frame's identity is checked
    rather than trusted.
    """

    def __init__(self, frame: DataFrame):
        self._frame = frame
        self._key_of: dict[tuple[str, ...], str] = {}
        self._loadings: DataFrame | None = None

    def warm(self, member_sets) -> 'ClusterPCACache':
        """Fit PC1 for every member-set not already cached."""
        fresh = {}
        for members in member_sets:
            key = tuple(members)
            if key in self._key_of:
                continue
            name = f'set{len(self._key_of):05d}'
            self._key_of[key] = name
            fresh[name] = list(members)

        if not fresh:
            return self

        _, loadings = pca.fit_cluster_pc1(self._frame, fresh)
        self._loadings = (loadings if self._loadings is None
                          else pd.concat([self._loadings, loadings], ignore_index=True))

        return self

    def loadings_for(self, members) -> DataFrame:
        """Cached loading rows for one member-set, keyed by the canonical name."""
        key = tuple(members)
        if key not in self._key_of:
            self.warm([key])

        name = self._key_of[key]
        rows = self._loadings[self._loadings['factor'] == name]
        if rows.empty:
            raise KeyError(f'no cached PC1 for {list(members)}')

        return rows

    def __len__(self) -> int:
        return len(self._key_of)


def _canonical(members) -> tuple[str, ...]:
    """Member-set as a deterministic key.

    Sorted, so two solutions that found the same family hit the same cache entry
    whatever order their algorithm emitted the members in. PC1 does not depend on
    column order, so this cannot merge two genuinely different clusters.
    """
    return tuple(sorted(members))


def build_factors(frame: DataFrame, solution: ClusterSolution,
                  rho: DataFrame | None = None,
                  cache: ClusterPCACache | None = None) -> FactorSet:
    """One PC1 factor per cluster of ``solution``, with loadings and quality.

    ``rho`` names each factor after its cluster's medoid (``kmeans_Tc`` rather than
    ``kmeans_c03``), which is what makes a downstream coefficient table readable.
    """
    named = solution.named_clusters(rho)
    cache = ClusterPCACache(frame) if cache is None else cache
    cache.warm(_canonical(members) for members in named.values())

    blocks: list[DataFrame] = []
    for name, members in named.items():
        rows = cache.loadings_for(_canonical(members)).copy()
        rows['factor'] = name
        blocks.append(rows)

    loadings = pd.concat(blocks, ignore_index=True)
    values = pca.apply_loadings(frame, loadings)

    specs = pca.cluster_pc1_specs(
        loadings, source_method=f'{solution.method}_clusters',
        clustering_method=solution.method, feature_type='cluster_pc1')

    return FactorSet(
        method=solution.method,
        params=dict(solution.params),
        values=values[list(named)],
        loadings=loadings,
        specs=tuple(specs),
        quality=pc1_quality(solution, loadings),
    )


def pc1_quality(solution: ClusterSolution, loadings: DataFrame) -> DataFrame:
    """One row per cluster: what its PC1 retained and what it was built from."""
    label_of = {name: label for name, label in zip(
        solution.named_clusters().keys(), solution.clusters().keys())}
    ordered_names = list(loadings['factor'].drop_duplicates())
    ids = list(solution.clusters().keys())

    rows: list[dict] = []
    for position, (name, group) in enumerate(loadings.groupby('factor', sort=False)):
        members = list(group['feature'])
        ratio = float(group['explained_variance_ratio'].iloc[0])
        top = group.loc[group['loading'].abs().idxmax()]

        rows.append({
            'method': solution.method,
            'hyperparameters': format_params(solution.params),
            'cluster_name': str(name),
            'cluster_id': ids[position] if position < len(ids) else label_of.get(name),
            'n_members': len(members),
            'features': ', '.join(members),
            'explained_variance_ratio': ratio,
            'is_singleton': len(members) == 1,
            # Singletons are 1.0 by construction, so flagging one would be
            # meaningless; only a real cluster can fail to be summarised.
            'flagged': len(members) > 1 and ratio < config.PC1_RETENTION_FLOOR,
            'top_loading_feature': str(top['feature']),
            'top_loading': float(top['loading']),
        })

    assert len(ordered_names) == len(rows)

    return DataFrame(rows, columns=list(QUALITY_COLUMNS))


def solution_summary(quality: DataFrame) -> dict:
    """Solution-level PC1 retention, reported both inflated and honest.

    See the module docstring: the all-cluster figures are inflated by singletons and are
    included so that the inflation is visible, not so that they are used. The ``_multi``
    figures and ``weighted_pc1_explained_variance`` are the ones that mean something.
    """
    ratios = quality['explained_variance_ratio'].to_numpy(dtype=float)
    sizes = quality['n_members'].to_numpy(dtype=float)
    multi = quality.loc[~quality['is_singleton'], 'explained_variance_ratio']
    multi_values = multi.to_numpy(dtype=float)

    blank = {'mean_pc1_explained_variance_multi': np.nan,
             'median_pc1_explained_variance_multi': np.nan,
             'min_pc1_explained_variance': np.nan}
    if multi_values.size:
        blank = {'mean_pc1_explained_variance_multi': float(multi_values.mean()),
                 'median_pc1_explained_variance_multi': float(np.median(multi_values)),
                 'min_pc1_explained_variance': float(multi_values.min())}

    n_flagged = int(quality['flagged'].sum())

    return {
        'n_factors': int(len(quality)),
        'n_multi_member_clusters': int(multi_values.size),
        'mean_pc1_explained_variance': float(ratios.mean()),
        **blank,
        # Size-weighted over every cluster: a 13-descriptor family counts 13 times as
        # much as a singleton, which is the sense in which it carries more of the
        # descriptor set. Still not immune to a singleton-heavy partition, but far less
        # flattering to one than the plain mean.
        'weighted_pc1_explained_variance': float(np.average(ratios, weights=sizes)),
        'n_flagged_clusters': n_flagged,
        'flagged_fraction': n_flagged / multi_values.size if multi_values.size else np.nan,
    }


def build_all(frame: DataFrame, solutions, rho: DataFrame | None = None,
              with_loadings: bool = True) -> tuple[DataFrame, DataFrame, DataFrame]:
    """PC1 quality for every solution in the sweep.

    Returns ``(quality, summary, loadings)``. Factor *values* are deliberately not
    returned: 430 solutions times ~20 factors times 720 liquids is a large matrix that
    nothing at this stage reads, and Phase 8 needs values only for the candidates it
    actually scores -- which it gets from ``build_factors`` one solution at a time.
    """
    cache = ClusterPCACache(frame)
    cache.warm(_canonical(members)
               for solution in solutions
               for members in solution.named_clusters(rho).values())

    quality_frames: list[DataFrame] = []
    loading_frames: list[DataFrame] = []
    summaries: list[dict] = []

    for solution in solutions:
        named = solution.named_clusters(rho)
        blocks = []
        for name, members in named.items():
            rows = cache.loadings_for(_canonical(members)).copy()
            rows['factor'] = name
            blocks.append(rows)
        loadings = pd.concat(blocks, ignore_index=True)

        quality = pc1_quality(solution, loadings)
        quality_frames.append(quality)
        summaries.append({'method': solution.method,
                          'hyperparameters': format_params(solution.params),
                          **solution_summary(quality)})

        if with_loadings:
            loadings = loadings.copy()
            loadings['method'] = solution.method
            loadings['hyperparameters'] = format_params(solution.params)
            loading_frames.append(loadings)

    return (
        pd.concat(quality_frames, ignore_index=True),
        DataFrame(summaries),
        (pd.concat(loading_frames, ignore_index=True) if loading_frames
         else DataFrame()),
    )
