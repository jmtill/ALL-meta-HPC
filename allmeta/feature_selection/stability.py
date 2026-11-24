"""Does a clustering survive a change of liquids? (implementation plan, Phase 7)

Every metric in ``metrics.py`` scores a partition against the data it was fitted on, so
a method can score well by fitting that particular sample closely. Stability asks the
other question: refit on a subsample and see whether the same partition comes back. On a
project whose supervised half rests on 13 liquids, a partition that reshuffles when the
liquid set moves cannot support a claim about anything, however clean it looks.

## What is resampled

The **liquids**, never the features. Every subsample holds all 45 descriptors, which is
what makes this cheap and exact: the partitions being compared are always partitions of
the same 45 objects, so no index intersection is needed and every feature pair is valid
in every resample. (``allmeta.clustering.stability`` clusters liquids and therefore does
need the shared-index trick -- the two are not interchangeable.)

Every candidate sees the **same** subsamples. Stability numbers are only comparable
across methods if the methods were handed the same perturbations, and it also means the
representation for a subsample is built once rather than once per candidate.

## How partitions are compared

Cluster label numbering is arbitrary, so agreement is measured with label-invariant
statistics: adjusted Rand index, adjusted mutual information, normalised mutual
information. All three come from one shared contingency table -- see ``agreement`` --
because computing them through scikit-learn's three separate entry points rebuilds that
table three times and costs 2.3 ms per pair against 0.19 ms here. Over 430 candidates
and 435 pairs each that difference is 90 minutes against 35 seconds, which is the
difference between a sweep that gets run and one that does not.

Alongside the summary statistics, the co-clustering probability of every feature pair,

    P_ij = (times i and j landed in one cluster) / (number of fits)

which is where the summary statistics come from and where they can be checked. A method
with a high mean ARI and a mushy consensus matrix has stable cluster *counts* and
unstable cluster *membership*.
"""

import itertools
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import sparse
from sklearn.metrics.cluster._expected_mutual_info_fast import (
    expected_mutual_information)

from allmeta import columns as cols
from allmeta.feature_selection import algorithms, config, representations
from allmeta.paths import RANDOM_STATE

AGREEMENT_METRICS: tuple[str, ...] = ('ari', 'ami', 'nmi')


# ---------------------------------------------------------------------------
# Label-invariant agreement
# ---------------------------------------------------------------------------

def contingency(left, right) -> np.ndarray:
    """Co-occurrence counts between two labellings of the same objects."""
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError('labellings cover different numbers of objects')

    _, rows = np.unique(left, return_inverse=True)
    _, columns = np.unique(right, return_inverse=True)

    counts = np.zeros((rows.max() + 1, columns.max() + 1), dtype=float)
    np.add.at(counts, (rows, columns), 1)

    return counts


def agreement(left, right) -> dict[str, float]:
    """Adjusted Rand, adjusted mutual information and normalised mutual information.

    All three from one contingency table. Verified against scikit-learn to ~1e-16 in
    the tests; this exists for speed, not because the definitions differ.

    Only the AMI needs the expected-mutual-information correction, and that is the
    expensive term, so scikit-learn's compiled implementation is called for it rather
    than reimplemented -- a Python double loop over the contingency cells measured
    *slower* than scikit-learn's whole three-metric path.
    """
    counts = contingency(left, right)
    n = counts.sum()
    if n == 0:
        raise ValueError('cannot compare empty labellings')

    row_sums = counts.sum(axis=1)
    column_sums = counts.sum(axis=0)

    def pairs(values):
        return (values * (values - 1) / 2).sum()

    same = pairs(counts)
    by_row = pairs(row_sums)
    by_column = pairs(column_sums)
    expected = by_row * by_column / pairs(np.array([n]))
    maximum = (by_row + by_column) / 2
    # Two identical partitions of singletons, or two single clusters: every index
    # degenerates and the honest answer is perfect agreement, not 0/0.
    ari = 1.0 if maximum == expected else (same - expected) / (maximum - expected)

    occupied = counts > 0
    joint = counts[occupied] / n
    independent = np.outer(row_sums, column_sums)[occupied] / (n * n)
    mutual = float((joint * (np.log(joint) - np.log(independent))).sum())

    row_p = row_sums[row_sums > 0] / n
    column_p = column_sums[column_sums > 0] / n
    entropy = (float(-(row_p * np.log(row_p)).sum())
               + float(-(column_p * np.log(column_p)).sum())) / 2

    nmi = 1.0 if entropy == 0 else mutual / entropy

    expected_mutual = expected_mutual_information(sparse.csr_matrix(counts), int(n))
    denominator = entropy - expected_mutual
    ami = 1.0 if denominator == 0 else (mutual - expected_mutual) / denominator

    return {'ari': float(ari), 'ami': float(ami), 'nmi': float(nmi)}


def pairwise_agreement(labellings) -> dict[str, float]:
    """Mean and spread of each agreement metric over every pair of labellings."""
    scores = [agreement(left, right)
              for left, right in itertools.combinations(labellings, 2)]

    if not scores:
        return {f'{metric}_{stat}': np.nan
                for metric in AGREEMENT_METRICS for stat in ('mean', 'std', 'min')
                } | {'n_pairs': 0}

    summary: dict[str, float] = {'n_pairs': len(scores)}
    for metric in AGREEMENT_METRICS:
        values = np.array([score[metric] for score in scores], dtype=float)
        summary[f'{metric}_mean'] = float(values.mean())
        summary[f'{metric}_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[f'{metric}_min'] = float(values.min())

    return summary


def consensus_matrix(labellings, features) -> DataFrame:
    """Co-clustering probability for every feature pair.

    ``P_ij`` is the fraction of fits in which features i and j landed in one cluster.
    Because every subsample holds every feature, the denominator is simply the number
    of fits -- no pair is ever missing, so there is no "valid comparisons" bookkeeping
    to get wrong.
    """
    if not labellings:
        raise ValueError('no labellings to build a consensus from')

    size = len(features)
    total = np.zeros((size, size), dtype=float)
    for labels in labellings:
        labels = np.asarray(labels)
        if labels.shape[0] != size:
            raise ValueError('a labelling does not cover every feature')
        total += (labels[:, None] == labels[None, :])

    return DataFrame(total / len(labellings), index=features, columns=features)


def consensus_summary(matrix: DataFrame) -> dict[str, float]:
    """How decisive a consensus matrix is.

    A pair at 0.5 is a coin flip: the method put those two descriptors together half
    the time. ``ambiguous_fraction`` counts the pairs sitting between 0.2 and 0.8, and
    is the number that catches a method whose mean ARI looks respectable because its
    cluster *count* is stable while its cluster *membership* is not.
    """
    values = matrix.to_numpy(dtype=float)
    upper = np.triu(np.ones_like(values, dtype=bool), k=1)
    pairs = values[upper]

    return {
        'consensus_mean': float(pairs.mean()),
        'consensus_decisive_fraction': float(((pairs <= 0.2) | (pairs >= 0.8)).mean()),
        'consensus_ambiguous_fraction': float(((pairs > 0.2) & (pairs < 0.8)).mean()),
        'n_feature_pairs': int(pairs.size),
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StabilityConfig:
    """How much resampling to do. See ``config.STABILITY_PRESETS``."""

    n_resamples: int = 30
    subsample_fraction: float = 0.8
    max_candidates_per_method: int | None = None
    frame_size: int | None = None
    random_state: int = RANDOM_STATE

    @classmethod
    def preset(cls, name: str = 'full', **overrides) -> 'StabilityConfig':
        """A named preset from config, with any field overridden.

        ``StabilityConfig.preset('test')`` is the smoke-test setting;
        ``StabilityConfig.preset('test', n_resamples=10)`` adjusts one field of it.
        """
        if name not in config.STABILITY_PRESETS:
            raise ValueError(f'unknown stability preset {name!r}; '
                             f'expected one of {sorted(config.STABILITY_PRESETS)}')

        return replace(cls(**config.STABILITY_PRESETS[name]), **overrides)

    def describe(self) -> dict:
        """Flat record of the settings, to travel with the results.

        Written into every output row on purpose: a mean ARI from 5 resamples and one
        from 30 are not the same measurement, and a table that does not say which it is
        invites them to be compared.
        """
        return {
            'n_resamples': self.n_resamples,
            'subsample_fraction': self.subsample_fraction,
            'max_candidates_per_method': self.max_candidates_per_method,
            'frame_size': self.frame_size,
            'random_state': self.random_state,
        }


def thin(grid: list[dict], cap: int | None) -> list[dict]:
    """Evenly spaced subset of a hyperparameter grid.

    Evenly spaced rather than the first ``cap``: the grids are generated by nested
    loops, so the leading entries all share the first value of the outermost parameter.
    Taking a prefix of the HDBSCAN grid would test six settings that differ only in
    ``cluster_selection_method``.
    """
    if cap is None or cap >= len(grid):
        return list(grid)
    if cap <= 0:
        raise ValueError(f'max_candidates_per_method must be positive, got {cap}')

    step = len(grid) / cap

    return [grid[min(len(grid) - 1, int(i * step))] for i in range(cap)]


def resample_indices(n_rows: int, settings: StabilityConfig) -> list[np.ndarray]:
    """Row positions for each subsample, drawn once and shared by every candidate."""
    rng = np.random.default_rng(settings.random_state)
    size = max(3, int(round(n_rows * settings.subsample_fraction)))

    return [rng.choice(n_rows, size=size, replace=False)
            for _ in range(settings.n_resamples)]


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def _representations(frame: DataFrame, features: list[str]) -> dict:
    """Every geometry one subsample supports, built once for all candidates."""
    built = {}
    for method in representations.CORRELATION_METHODS:
        built[method] = representations.correlation_representation(
            frame, features, method)[1]
    built['profile'] = representations.feature_profile_representation(frame, features)[0]

    return built


def _candidate_grid(methods, n_features: int, settings: StabilityConfig
                    ) -> list[tuple[str, dict]]:
    """(method, hyperparameters) pairs to measure, after thinning."""
    return [(method, params)
            for method in methods
            for params in thin(algorithms.candidates(method, n_features),
                               settings.max_candidates_per_method)]


def run(frame: DataFrame, features: list[str] | None = None, methods=None,
        settings: StabilityConfig | None = None
        ) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Refit every candidate on every subsample and measure how much it moved.

    Returns ``(stability, consensus, assignments)``:

    * ``stability`` -- one row per candidate: mean/sd/min of ARI, AMI and NMI over the
      pairs, the consensus summary, and how often the fit failed.
    * ``consensus`` -- long-form co-clustering probability per (method, feature pair),
      pooled over that method's candidates and resamples.
    * ``assignments`` -- one row per (method, feature): how often it ends up alone, and
      which descriptor it most reliably travels with.
    """
    features = cols.screening_features() if features is None else list(features)
    methods = config.CLUSTERING_METHODS if methods is None else list(methods)
    settings = StabilityConfig() if settings is None else settings

    if settings.frame_size is not None and settings.frame_size < len(frame):
        frame = frame.sample(n=settings.frame_size,
                             random_state=settings.random_state)

    grid = _candidate_grid(methods, len(features), settings)
    samples = resample_indices(len(frame), settings)

    # candidate key -> one labelling per successful resample
    labellings: dict[tuple[str, str], list[tuple[int, ...]]] = {}
    failures: dict[tuple[str, str], int] = {}

    for indices in samples:
        built = _representations(frame.iloc[indices], features)

        for method, params in grid:
            key = (method, algorithms.format_params(params))
            kind = config.REPRESENTATION_BY_METHOD[method]
            correlation = params.get('correlation_method', 'spearman')
            representation = (built[correlation] if kind == 'correlation_distance'
                              else built['profile'])
            try:
                solution = algorithms.fit(method, representation, **params)
            except (ValueError, np.linalg.LinAlgError):
                # A candidate that fails on some subsamples is itself unstable, so the
                # count is reported rather than the row being dropped.
                failures[key] = failures.get(key, 0) + 1
                continue

            # Reindexed onto the caller's feature order: an algorithm is free to return
            # its own, and comparing two differently ordered labellings would silently
            # measure agreement between shuffled partitions.
            label_of = solution.label_of
            labellings.setdefault(key, []).append(
                tuple(label_of[feature] for feature in features))

    return _summarise(labellings, failures, grid, features, settings)


def _summarise(labellings, failures, grid, features, settings
               ) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Turn the collected labellings into the three output tables."""
    described = settings.describe()

    rows: list[dict] = []
    by_method: dict[str, list] = {}

    for method, hyperparameters in [(m, algorithms.format_params(p)) for m, p in grid]:
        key = (method, hyperparameters)
        collected = labellings.get(key, [])
        by_method.setdefault(method, []).extend(collected)

        record = {
            'method': method,
            'hyperparameters': hyperparameters,
            'n_successful_fits': len(collected),
            'n_failed_fits': failures.get(key, 0),
            **described,
        }
        record.update(pairwise_agreement(collected))

        if collected:
            matrix = consensus_matrix(collected, features)
            record.update(consensus_summary(matrix))
            counts = [len(set(labels)) for labels in collected]
            record['mean_n_clusters'] = float(np.mean(counts))
            record['std_n_clusters'] = float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0

        rows.append(record)

    stability = DataFrame(rows)

    consensus_rows: list[dict] = []
    assignment_rows: list[dict] = []
    upper = np.triu(np.ones((len(features), len(features)), dtype=bool), k=1)

    for method, collected in by_method.items():
        if not collected:
            continue
        matrix = consensus_matrix(collected, features)
        values = matrix.to_numpy()

        left, right = np.where(upper)
        consensus_rows.extend({
            'method': method, 'feature_a': features[i], 'feature_b': features[j],
            'co_cluster_probability': float(values[i, j]), 'n_fits': len(collected),
        } for i, j in zip(left, right))

        for position, feature in enumerate(features):
            partners = values[position].copy()
            partners[position] = -1.0
            best = int(partners.argmax())
            alone = float(np.mean([
                sum(1 for other in labels if other == labels[position]) == 1
                for labels in collected]))
            assignment_rows.append({
                'method': method, 'feature': feature,
                'singleton_rate': alone,
                'mean_co_cluster_probability': float(partners[partners >= 0].mean()),
                'closest_partner': features[best],
                'closest_partner_probability': float(partners[best]),
                'n_fits': len(collected),
                **described,
            })

    return stability, DataFrame(consensus_rows), DataFrame(assignment_rows)


def method_agreement(solutions) -> DataFrame:
    """Pairwise agreement between the algorithms themselves, on one fixed frame.

    A different question from stability: not "does this method repeat itself?" but "do
    the methods agree with each other?". Two methods that both repeat themselves
    perfectly and disagree with each other have found two different structures, and
    only one of them can be the descriptor families.

    Each method is represented by its median-cluster-count solution, so the comparison
    is not decided by one method having been given a finer grid than another.
    """
    representatives: dict[str, object] = {}
    for method in {solution.method for solution in solutions}:
        found = [s for s in solutions if s.method == method]
        counts = sorted(found, key=lambda s: s.n_clusters)
        representatives[method] = counts[len(counts) // 2]

    rows: list[dict] = []
    for left, right in itertools.combinations(sorted(representatives), 2):
        a, b = representatives[left], representatives[right]
        order = list(a.features)
        b_label = b.label_of
        scores = agreement([a.label_of[f] for f in order],
                           [b_label[f] for f in order])
        rows.append({
            'method_a': left, 'method_b': right,
            'n_clusters_a': a.n_clusters, 'n_clusters_b': b.n_clusters,
            **scores,
        })

    return DataFrame(rows)
