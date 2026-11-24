"""One scoring layer for every clustering solution (implementation plan, Phase 5).

Takes a partition and the representations it could be judged in, and returns a flat,
standardised metric dictionary. Every algorithm goes through this same layer, which is
the only way the Phase 9 comparison can be fair: five algorithms each scored by their
own favourite statistic would be five results, not one comparison.

Three families of metric, because each answers a different objection:

* **internal** (silhouette, Calinski-Harabasz, Davies-Bouldin) -- is the partition
  geometrically clean?
* **correlation coherence** (within- and between-cluster |r|, their difference, and
  the fraction of feature pairs the partition actually commits on) -- do the clusters
  correspond to families of descriptors that carry the same information? This is the
  domain-appropriate question, and unlike the internal metrics it is computed in the
  same space for every method whatever representation the algorithm consumed.

  ``within_between_delta`` **cannot be read on its own.** Measured on the real sweep it
  correlates at 0.80 with ``singleton_fraction``: the highest-scoring solution in 430
  is a hierarchical cut at threshold 0.05 that produces 38 clusters, 36 of them
  singletons, and so commits on 16 of the 990 feature pairs. It wins by declining to
  cluster. ``within_pair_fraction`` is reported beside it for exactly this reason -- a
  large delta over 2% of the pairs is a refusal, not a result. Neither number alone
  selects a method; that is what the predictive criterion in Phase 8 is for.
* **size diagnostics** -- how many clusters, how big, how many singletons, how much
  noise. Not quality measures, but the numbers without which a quality measure cannot
  be read: a partition of 45 singletons scores beautifully on nothing.

## Why the internal metrics are reported twice

The five algorithms consume two different geometries. Silhouette is defined on both -- a
precomputed distance or a point cloud -- but Calinski-Harabasz and Davies-Bouldin need
coordinates and have no meaning on a precomputed distance matrix.

Reporting those two as NA for hierarchical, HDBSCAN and K-Medoids would leave the
method comparison with nothing to compare on. Instead both spaces are scored for every
method and the column names say which space each number came from:

* ``silhouette_correlation`` -- in ``1 - |r|`` space, the space the domain question
  lives in, valid for all five methods.
* ``silhouette_profile``, ``calinski_harabasz_profile``, ``davies_bouldin_profile`` --
  in the standardised feature-profile space, a common Euclidean space that every
  partition can be projected into, valid for all five methods.

A method is never scored in a space where its metric is undefined, and no number is
invented. The one thing a reader must not do is compare a ``_correlation`` column
against a ``_profile`` column: they are different questions, not different estimates of
one.
"""

import numpy as np
from pandas import DataFrame
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from allmeta.feature_selection.algorithms import ClusterSolution

# Metrics where larger is better. Davies-Bouldin is the exception -- it measures
# average cluster overlap, so lower is better -- and is listed separately rather than
# sign-flipped, because a negated Davies-Bouldin in a CSV is a trap.
HIGHER_IS_BETTER: tuple[str, ...] = (
    'silhouette_correlation', 'silhouette_profile', 'calinski_harabasz_profile',
    'mean_within_abs_corr', 'within_between_delta',
)
# Deliberately in neither list. More coverage is not better -- one 45-member cluster
# covers every pair and is useless -- and less is not better either. It is the context
# without which within_between_delta cannot be read. See the module docstring.
DESCRIPTIVE: tuple[str, ...] = (
    'n_clusters', 'singleton_fraction', 'noise_fraction', 'within_pair_fraction',
)
LOWER_IS_BETTER: tuple[str, ...] = (
    'davies_bouldin_profile', 'mean_between_abs_corr',
)


def _is_degenerate(labels) -> bool:
    """A partition with one cluster, or with as many clusters as objects.

    Undefined rather than merely bad for all three internal metrics: one cluster has no
    between-cluster distance, and all-singletons has no within-cluster distance.
    """
    n_labels = len(set(labels))

    return n_labels < 2 or n_labels >= len(labels)


def internal_metrics(solution: ClusterSolution, distance: DataFrame,
                     profiles: DataFrame) -> dict[str, float]:
    """Silhouette in correlation space, and silhouette/CH/DB in profile space.

    Both representations are re-indexed onto the solution's own feature order, so a
    caller cannot silently score a partition against a differently ordered matrix --
    which would produce a plausible number computed from a shuffled partition.
    """
    features = list(solution.features)
    labels = np.asarray(solution.labels)

    blank = {
        'silhouette_correlation': np.nan,
        'silhouette_profile': np.nan,
        'calinski_harabasz_profile': np.nan,
        'davies_bouldin_profile': np.nan,
    }
    if _is_degenerate(labels):
        return blank

    scores = dict(blank)

    aligned_distance = distance.loc[features, features].to_numpy(dtype=float)
    scores['silhouette_correlation'] = float(
        silhouette_score(aligned_distance, labels, metric='precomputed'))

    aligned_profiles = profiles.loc[features].to_numpy(dtype=float)
    scores['silhouette_profile'] = float(silhouette_score(aligned_profiles, labels))
    scores['calinski_harabasz_profile'] = float(
        calinski_harabasz_score(aligned_profiles, labels))
    scores['davies_bouldin_profile'] = float(
        davies_bouldin_score(aligned_profiles, labels))

    return scores


def correlation_metrics(solution: ClusterSolution, rho: DataFrame) -> dict[str, float]:
    """Within- and between-cluster absolute correlation, from the original matrix.

    The domain metric, and the one that says whether a partition found families of
    descriptors that carry the same information. Computed on the ORIGINAL feature
    correlation matrix regardless of what the algorithm consumed, so KMeans on
    Euclidean profiles and hierarchical on correlation distance are judged on the same
    question.

    ``within_between_delta`` is mean within |r| minus mean between |r|. A large positive
    value means the partition separated correlation families; a value near zero means
    the clusters are no more internally correlated than the descriptor set as a whole,
    whatever the silhouette says.

    ``min_within_abs_corr`` is the worst-glued pair inside any cluster -- the single
    number that catches one badly assigned descriptor dragging an otherwise tight
    family down, which every mean hides.

    ``within_pair_fraction`` is what stops the delta being gamed: the share of all
    feature pairs that land inside some cluster. An all-singletons partition has zero
    within-pairs and therefore no delta at all, but a partition that is 90% singletons
    still reports a delta -- computed over a handful of pairs, and flattering. Read the
    two together or neither.
    """
    features = list(solution.features)
    values = rho.loc[features, features].to_numpy(dtype=float)
    labels = np.asarray(solution.labels)

    same = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones_like(same, dtype=bool), k=1)

    within = values[same & upper]
    between = values[~same & upper]

    def summarise(pairs: np.ndarray, prefix: str) -> dict[str, float]:
        if pairs.size == 0:
            return {f'mean_{prefix}_abs_corr': np.nan,
                    f'median_{prefix}_abs_corr': np.nan,
                    f'min_{prefix}_abs_corr': np.nan,
                    f'n_{prefix}_pairs': 0}

        return {f'mean_{prefix}_abs_corr': float(np.mean(pairs)),
                f'median_{prefix}_abs_corr': float(np.median(pairs)),
                f'min_{prefix}_abs_corr': float(np.min(pairs)),
                f'n_{prefix}_pairs': int(pairs.size)}

    scores = {**summarise(within, 'within'), **summarise(between, 'between')}
    scores['within_between_delta'] = (
        scores['mean_within_abs_corr'] - scores['mean_between_abs_corr']
        if within.size and between.size else np.nan)

    n_pairs = int(upper.sum())
    scores['within_pair_fraction'] = (within.size / n_pairs) if n_pairs else np.nan

    return scores


def evaluate(solution: ClusterSolution, rho: DataFrame, distance: DataFrame,
             profiles: DataFrame) -> dict:
    """Every metric for one solution, as one flat record."""
    record = solution.summary()
    record.update(internal_metrics(solution, distance, profiles))
    record.update(correlation_metrics(solution, rho))
    record.update({f'extra_{key}': value for key, value in solution.extra.items()})

    return record


def evaluate_all(solutions, rho: DataFrame, distance_by_correlation: dict,
                 profiles: DataFrame) -> DataFrame:
    """Score many solutions into one table.

    ``distance_by_correlation`` maps a correlation method ('pearson', 'spearman') to
    its distance matrix, because a solution found under Pearson distance must be scored
    against the Pearson silhouette; scoring it against Spearman would measure a
    partition in a space it never saw. ``rho`` is a single matrix on purpose -- the
    domain metrics are deliberately computed in one fixed space so that every method is
    compared on the same question.
    """
    records = []
    for solution in solutions:
        correlation = solution.params.get('correlation_method', 'spearman')
        distance = distance_by_correlation.get(correlation)
        if distance is None:
            raise KeyError(
                f'no {correlation!r} distance matrix supplied for a {solution.method} '
                'solution that was fitted on one')
        records.append(evaluate(solution, rho, distance, profiles))

    frame = DataFrame(records)

    # Stable, readable column order: what was run, then how big the answer is, then how
    # good it is, then the algorithm-specific extras.
    leading = ['method', 'representation', 'hyperparameters', 'n_features',
               'n_clusters', 'mean_cluster_size', 'median_cluster_size',
               'max_cluster_size', 'n_singletons', 'singleton_fraction', 'n_noise',
               'noise_fraction']
    scores = [c for c in HIGHER_IS_BETTER + LOWER_IS_BETTER if c in frame.columns]
    middle = [c for c in ('within_pair_fraction',
                          'median_within_abs_corr', 'min_within_abs_corr',
                          'median_between_abs_corr', 'min_between_abs_corr',
                          'n_within_pairs', 'n_between_pairs', 'converged', 'warnings')
              if c in frame.columns]
    rest = [c for c in frame.columns if c not in leading + scores + middle]

    return frame[leading + scores + middle + rest]


def rank(frame: DataFrame, metric: str) -> DataFrame:
    """Sort a metric table by one metric, best first, respecting its direction."""
    if metric not in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        raise ValueError(f'unknown metric {metric!r}; direction is not declared, and '
                         'sorting it would silently assume one')

    return frame.dropna(subset=[metric]).sort_values(
        metric, ascending=metric in LOWER_IS_BETTER, ignore_index=True)
