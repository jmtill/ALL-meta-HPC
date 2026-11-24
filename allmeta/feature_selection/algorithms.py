"""The five feature-clustering algorithms (implementation plan, Phase 4).

Each function partitions the **features**, takes the representation it declares in
``config.REPRESENTATION_BY_METHOD``, and returns a ``ClusterSolution``. Nothing here
scores a partition -- that is ``metrics.py`` -- and nothing here chooses
hyperparameters. This module's whole job is to turn (algorithm, hyperparameters,
representation) into labels, honestly, including when the answer is degenerate.

Three rules the implementations share:

* **No feature is ever silently dropped.** HDBSCAN can decline to assign a feature; the
  noise policy turns each such feature into its own singleton cluster rather than
  losing it. A feature set that quietly shrinks is a wrong answer that looks like a
  clean one.
* **A failed fit is reported, not swallowed.** GMM in particular can fail to converge
  on the profile representation; ``ClusterSolution.converged`` and ``warnings`` carry
  that forward so a non-converged solution cannot win a comparison by silence.
* **Labels are canonicalised.** Every algorithm's raw label numbering is arbitrary, so
  labels are renumbered by first appearance. Two runs that found the same partition
  then compare equal, which matters because ARI/AMI in Phase 7 are label-invariant but
  the written tables are not.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
from pandas import DataFrame
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.mixture import GaussianMixture

from allmeta.feature_selection import config
from allmeta.paths import RANDOM_STATE
from allmeta.screening import pick_representative

# HDBSCAN's label for a feature it declines to assign.
NOISE_LABEL: int = -1


@dataclass(frozen=True)
class ClusterSolution:
    """One partition of the features, and everything needed to read it back."""

    method: str
    params: dict
    representation: str
    features: tuple[str, ...]
    labels: tuple[int, ...]
    n_noise: int = 0
    converged: bool = True
    warnings: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.features) != len(self.labels):
            raise ValueError('features and labels have different lengths')

    @property
    def n_clusters(self) -> int:
        return len(set(self.labels))

    @property
    def label_of(self) -> dict[str, int]:
        return dict(zip(self.features, self.labels))

    def clusters(self) -> dict[int, list[str]]:
        """Cluster id to member features, ids in order of first appearance."""
        groups: dict[int, list[str]] = {}
        for feature, label in zip(self.features, self.labels):
            groups.setdefault(label, []).append(feature)

        return groups

    def sizes(self) -> list[int]:
        return [len(members) for members in self.clusters().values()]

    @property
    def n_singletons(self) -> int:
        return sum(1 for size in self.sizes() if size == 1)

    def named_clusters(self, rho: DataFrame | None = None,
                       prefix: str | None = None) -> dict[str, list[str]]:
        """Cluster name to members, ready to hand to ``pca.fit_cluster_pc1``.

        With ``rho`` each cluster is named after its medoid -- the member with the
        highest mean absolute correlation to the others, which is the member that best
        stands for what the cluster holds. ``kmeans_c03`` tells a reader nothing;
        ``kmeans_Tc`` tells them the family is the one Tc sits in. Reuses
        ``screening.pick_representative`` so the medoid rule exists in one place and
        ``columns.PROTECTED`` keeps its override.

        Without ``rho`` the clusters are numbered, which is enough when the names are
        only being compared to each other.
        """
        tag = self.method if prefix is None else prefix
        named: dict[str, list[str]] = {}

        for label, members in self.clusters().items():
            if rho is None:
                name = f'{tag}_c{label:02d}'
            else:
                name = f'{tag}_{pick_representative(members, rho)}'
            # Two clusters cannot share a medoid, but a defensive suffix costs nothing
            # and a silently merged cluster would be very hard to notice downstream.
            if name in named:
                name = f'{name}_c{label:02d}'
            named[name] = members

        return named

    def summary(self) -> dict:
        """Flat record of the solution's shape, for a results table."""
        sizes = self.sizes()

        return {
            'method': self.method,
            'representation': self.representation,
            'hyperparameters': format_params(self.params),
            'n_features': len(self.features),
            'n_clusters': self.n_clusters,
            'mean_cluster_size': float(np.mean(sizes)),
            'median_cluster_size': float(np.median(sizes)),
            'max_cluster_size': int(max(sizes)),
            'n_singletons': self.n_singletons,
            'singleton_fraction': self.n_singletons / self.n_clusters,
            'n_noise': self.n_noise,
            'noise_fraction': self.n_noise / len(self.features),
            'converged': self.converged,
            'warnings': '; '.join(self.warnings),
        }


def format_params(params: dict) -> str:
    """Hyperparameters as a stable, readable string for a CSV cell."""
    return ', '.join(f'{key}={value}' for key, value in sorted(params.items()))


def canonical_labels(labels) -> tuple[int, ...]:
    """Renumber labels by first appearance, so equal partitions compare equal."""
    mapping: dict[int, int] = {}
    canonical: list[int] = []
    for label in labels:
        key = int(label)
        if key not in mapping:
            mapping[key] = len(mapping)
        canonical.append(mapping[key])

    return tuple(canonical)


def resolve_noise(labels, policy: str = None) -> tuple[list[int], int]:
    """Give every noise-labelled feature somewhere to go.

    ``'singleton'`` (the default, and ``config.HDBSCAN_NOISE_POLICY``) puts each noise
    feature in a cluster of its own; ``'own_cluster'`` puts all of them together, which
    says "these are one family" and is usually false; ``'drop'`` is refused, because a
    representation that quietly holds fewer features than it was given produces a
    smaller feature set that looks like a cleaner one.
    """
    policy = config.HDBSCAN_NOISE_POLICY if policy is None else policy
    labels = [int(label) for label in labels]
    n_noise = sum(1 for label in labels if label == NOISE_LABEL)

    if n_noise == 0:
        return labels, 0

    if policy == 'drop':
        raise ValueError(
            "noise policy 'drop' would silently shrink the feature set; use "
            "'singleton' or 'own_cluster'")

    if policy == 'own_cluster':
        spare = max(labels) + 1
        return [spare if label == NOISE_LABEL else label for label in labels], n_noise

    if policy != 'singleton':
        raise ValueError(f'unknown noise policy {policy!r}')

    resolved: list[int] = []
    spare = max(labels) + 1
    for label in labels:
        if label == NOISE_LABEL:
            resolved.append(spare)
            spare += 1
        else:
            resolved.append(label)

    return resolved, n_noise


def _as_distance(distance: DataFrame) -> tuple[list[str], np.ndarray]:
    """Validate a precomputed feature-distance matrix and return it as an array."""
    values = distance.to_numpy(dtype=float)
    if values.shape[0] != values.shape[1]:
        raise ValueError(f'distance matrix is not square: {values.shape}')
    if not np.allclose(values, values.T, atol=1e-8):
        raise ValueError('distance matrix is not symmetric')

    # scipy's condensed form rejects a diagonal that is not exactly zero, and
    # floating-point noise in a correlation matrix is enough to trip it.
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 0.0)

    return list(distance.index), values


def _as_profiles(profiles: DataFrame) -> tuple[list[str], np.ndarray]:
    """Validate a feature-profile matrix: one ROW per feature, not per liquid."""
    if profiles.shape[0] < 2:
        raise ValueError('need at least 2 features to cluster')

    return list(profiles.index), profiles.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# 4.1 Hierarchical / agglomerative
# ---------------------------------------------------------------------------

def cluster_hierarchical(distance: DataFrame, linkage_method: str = 'average',
                         distance_threshold: float = 0.25,
                         correlation_method: str = 'spearman') -> ClusterSolution:
    """Agglomerative clustering of the correlation-distance matrix.

    Cut by DISTANCE rather than by cluster count, so the threshold means something
    chemical: at 0.25 a cluster holds descriptors that merge while correlating above
    |r| = 0.75, whatever number of clusters that turns out to be. Choosing k instead
    would fix the answer's shape before looking at the data.

    Ward linkage is not offered. It minimises Euclidean inertia and is undefined for a
    precomputed non-Euclidean distance; scipy will compute it anyway and return a
    number that means nothing.
    """
    if linkage_method == 'ward':
        raise ValueError(
            'ward linkage is undefined on a precomputed correlation distance; '
            f'use one of {config.HIERARCHICAL_CONFIG["linkages"]}')

    features, values = _as_distance(distance)
    tree = linkage(squareform(values, checks=False), method=linkage_method)
    labels = fcluster(tree, t=distance_threshold, criterion='distance')

    return ClusterSolution(
        method='hierarchical',
        # Key names must match `candidates('hierarchical', ...)` exactly. They are the
        # join key between the metric, PC1 and stability tables, and a mismatch does not
        # raise -- it produces a merge that silently drops every hierarchical row. This
        # said 'linkage' once, and cost all 42 of them their stability numbers.
        params={'linkage_method': linkage_method,
                'distance_threshold': distance_threshold,
                'correlation_method': correlation_method},
        representation='correlation_distance',
        features=tuple(features),
        labels=canonical_labels(labels),
        extra={'cophenetic_max': float(tree[:, 2].max())},
    )


# ---------------------------------------------------------------------------
# 4.2 HDBSCAN
# ---------------------------------------------------------------------------

def cluster_hdbscan(distance: DataFrame, min_cluster_size: int = 2,
                    min_samples: int | None = None,
                    cluster_selection_epsilon: float = 0.0,
                    cluster_selection_method: str = 'eom',
                    noise_policy: str | None = None,
                    correlation_method: str = 'spearman') -> ClusterSolution:
    """Density-based clustering of the correlation distance.

    ``sklearn.cluster.HDBSCAN``, native since scikit-learn 1.3, so no extra dependency.
    Density here is density in *correlation* space: a cluster is a group of descriptors
    packed closely together relative to how far they sit from everything else.

    ``min_cluster_size`` counts FEATURES, not liquids, so 2 is the smallest meaningful
    value on a 45-descriptor problem -- a good deal smaller than the default of 5,
    which would refuse to find a two-descriptor family like dHDon/Abr-a.

    Unassigned features become singletons; see ``resolve_noise``.
    """
    features, values = _as_distance(distance)

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        metric='precomputed',
        allow_single_cluster=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw = model.fit_predict(values)

    resolved, n_noise = resolve_noise(raw, noise_policy)

    persistence = getattr(model, 'cluster_persistence_', None)
    extra = {'n_hdbscan_clusters': int(len({label for label in raw
                                            if label != NOISE_LABEL}))}
    if persistence is not None and len(persistence):
        extra['mean_cluster_persistence'] = float(np.mean(persistence))
        extra['min_cluster_persistence'] = float(np.min(persistence))

    notes: list[str] = []
    if extra['n_hdbscan_clusters'] == 0:
        notes.append('HDBSCAN assigned no feature to a cluster; every feature is noise')

    return ClusterSolution(
        method='hdbscan',
        params={'min_cluster_size': min_cluster_size, 'min_samples': min_samples,
                'cluster_selection_epsilon': cluster_selection_epsilon,
                'cluster_selection_method': cluster_selection_method,
                'correlation_method': correlation_method},
        representation='correlation_distance',
        features=tuple(features),
        labels=canonical_labels(resolved),
        n_noise=n_noise,
        warnings=tuple(notes),
        extra=extra,
    )


# ---------------------------------------------------------------------------
# 4.3 KMeans
# ---------------------------------------------------------------------------

def cluster_kmeans(profiles: DataFrame, n_clusters: int = 8,
                   n_init: int = 10) -> ClusterSolution:
    """KMeans over the Euclidean feature profiles.

    Each row is one descriptor's standardised profile across the liquids, so a cluster
    is a set of descriptors that vary together across the liquid space. Unlike the
    correlation distance this is sign-SENSITIVE: two perfectly anti-correlated
    descriptors sit at opposite ends of the space and KMeans will not group them. That
    is a real difference in what the question means, not a defect, and it is one of the
    things the method comparison exists to measure.
    """
    features, values = _as_profiles(profiles)

    if n_clusters >= len(features):
        raise ValueError(
            f'n_clusters={n_clusters} with {len(features)} features would put every '
            'feature in its own cluster; the partition would carry no information')

    model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        labels = model.fit_predict(values)

    return ClusterSolution(
        method='kmeans',
        params={'n_clusters': n_clusters, 'n_init': n_init},
        representation='feature_profile',
        features=tuple(features),
        labels=canonical_labels(labels),
        extra={'inertia': float(model.inertia_), 'n_iter': int(model.n_iter_)},
    )


# ---------------------------------------------------------------------------
# 4.4 K-Medoids
# ---------------------------------------------------------------------------

def _initial_medoids(values: np.ndarray, n_clusters: int, init: str,
                     random_state: int) -> np.ndarray:
    """Starting medoids as the ``uintp`` array the kmedoids extension requires.

    Computed here rather than left to ``kmedoids._check_medoids`` because that helper
    is broken for two of its three strategies in kmedoids 0.5.5: the ``'random'`` and
    explicit-array paths cast to ``np.uintp``, but ``'first'`` returns an int64
    ``arange`` and ``'build'`` returns uint64 from ``pam_build``, and the Rust
    extension rejects both with a bare ``TypeError``. Passing the array ourselves
    sidesteps that and makes the seeding explicit, which the library's own random path
    does not.
    """
    import kmedoids

    if init == 'first':
        chosen = np.arange(n_clusters)
    elif init == 'build':
        chosen = kmedoids.pam_build(values, n_clusters).medoids
    elif init == 'random':
        chosen = np.random.RandomState(random_state).choice(
            values.shape[0], n_clusters, replace=False)
    else:
        raise ValueError(f'unknown k-medoids init {init!r}; '
                         f"expected 'random', 'first' or 'build'")

    return np.asarray(chosen).astype(np.uintp)


def cluster_kmedoids(distance: DataFrame, n_clusters: int = 8, init: str = 'build',
                     correlation_method: str = 'spearman') -> ClusterSolution:
    """K-Medoids (FasterPAM) directly on the correlation-distance matrix.

    The reason to prefer medoids over means on this problem: a medoid is an actual
    descriptor, so every cluster comes with a real member that stands for it, and no
    Euclidean embedding of the correlation distance has to be invented first. KMeans
    cannot consume a precomputed distance at all.
    """
    import kmedoids

    features, values = _as_distance(distance)

    if n_clusters >= len(features):
        raise ValueError(
            f'n_clusters={n_clusters} with {len(features)} features would put every '
            'feature in its own cluster; the partition would carry no information')

    medoids = _initial_medoids(values, n_clusters, init, RANDOM_STATE)
    with warnings.catch_warnings():
        # 'Seed will be ignored if initial medoids are given' -- expected: the seed was
        # already consumed by _initial_medoids.
        warnings.simplefilter('ignore')
        result = kmedoids.fasterpam(values, medoids)

    return ClusterSolution(
        method='kmedoids',
        params={'n_clusters': n_clusters, 'init': init,
                'correlation_method': correlation_method},
        representation='correlation_distance',
        features=tuple(features),
        labels=canonical_labels(result.labels),
        extra={
            'loss': float(result.loss),
            'n_iter': int(result.n_iter),
            'medoids': ', '.join(features[int(i)] for i in result.medoids),
        },
    )


# ---------------------------------------------------------------------------
# 4.5 Gaussian Mixture Models
# ---------------------------------------------------------------------------

def cluster_gmm(profiles: DataFrame, n_components: int = 8,
                covariance_type: str = 'diag', reg_covar: float = 1e-6,
                n_init: int = 5) -> ClusterSolution:
    """Gaussian mixture over the Euclidean feature profiles.

    Consumes the profile representation for the same reason KMeans does: a mixture
    model needs points in a space, not pairwise distances.

    ``'full'`` covariance is badly over-parameterised at this shape -- a profile has as
    many dimensions as there are liquids, so a full covariance for one component has
    O(n_liquids^2) free parameters fitted from a handful of descriptors -- which is why
    ``config.GMM_CONFIG`` starts at the constrained forms. A fit that does not converge
    is recorded rather than dropped: an unconverged mixture still returns labels, and
    silence would let it win a comparison it should not be in.
    """
    features, values = _as_profiles(profiles)

    if n_components >= len(features):
        raise ValueError(
            f'n_components={n_components} with {len(features)} features would give '
            'every feature its own component')

    model = GaussianMixture(
        n_components=n_components, covariance_type=covariance_type,
        reg_covar=reg_covar, n_init=n_init, random_state=RANDOM_STATE,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        labels = model.fit_predict(values)
        notes = tuple(sorted({str(w.message)[:120] for w in caught}))

    posteriors = model.predict_proba(values)
    # Highest posterior per feature: how confidently each descriptor was assigned. A
    # mixture that assigns everything at 0.4 has not found components, it has found a
    # smear, and n_clusters alone would not show it.
    confidence = posteriors.max(axis=1)

    return ClusterSolution(
        method='gmm',
        params={'n_components': n_components, 'covariance_type': covariance_type,
                'reg_covar': reg_covar, 'n_init': n_init},
        representation='feature_profile',
        features=tuple(features),
        labels=canonical_labels(labels),
        converged=bool(model.converged_),
        warnings=notes,
        extra={
            'bic': float(model.bic(values)),
            'aic': float(model.aic(values)),
            'mean_assignment_confidence': float(confidence.mean()),
            'min_assignment_confidence': float(confidence.min()),
            'n_iter': int(model.n_iter_),
        },
    )


# ---------------------------------------------------------------------------
# Dispatch and hyperparameter grids
# ---------------------------------------------------------------------------

ALGORITHMS = {
    'hierarchical': cluster_hierarchical,
    'hdbscan': cluster_hdbscan,
    'kmeans': cluster_kmeans,
    'kmedoids': cluster_kmedoids,
    'gmm': cluster_gmm,
}


def candidates(method: str, n_features: int) -> list[dict]:
    """Every valid hyperparameter combination for one algorithm.

    The grids live in ``config``; this expands them and drops the combinations that are
    invalid at this feature count rather than letting them raise mid-sweep. The
    correlation method is part of the grid for the distance-based algorithms, because
    Pearson and Spearman give genuinely different distance matrices and which one suits
    the data is not knowable in advance.
    """
    from itertools import product

    if method == 'hierarchical':
        grid = config.HIERARCHICAL_CONFIG
        return [
            {'correlation_method': corr, 'linkage_method': link,
             'distance_threshold': threshold}
            for corr, link, threshold in product(
                grid['correlation_methods'], grid['linkages'],
                grid['distance_thresholds'])
        ]

    if method == 'hdbscan':
        grid = config.HDBSCAN_CONFIG
        return [
            {'correlation_method': corr, 'min_cluster_size': size,
             'min_samples': samples, 'cluster_selection_epsilon': epsilon,
             'cluster_selection_method': selection}
            for corr, size, samples, epsilon, selection in product(
                config.HIERARCHICAL_CONFIG['correlation_methods'],
                grid['min_cluster_size'], grid['min_samples'],
                grid['cluster_selection_epsilon'], grid['cluster_selection_method'])
            if size <= n_features and (samples is None or samples <= size)
        ]

    if method == 'kmeans':
        grid = config.KMEANS_CONFIG
        return [{'n_clusters': k, 'n_init': n}
                for k, n in product(grid['n_clusters'], grid['n_init'])
                if k < n_features]

    if method == 'kmedoids':
        grid = config.KMEDOIDS_CONFIG
        return [{'correlation_method': corr, 'n_clusters': k, 'init': init}
                for corr, k, init in product(
                    config.HIERARCHICAL_CONFIG['correlation_methods'],
                    grid['n_clusters'], grid['init'])
                if k < n_features]

    if method == 'gmm':
        grid = config.GMM_CONFIG
        return [
            {'n_components': k, 'covariance_type': cov, 'reg_covar': reg,
             'n_init': n}
            for k, cov, reg, n in product(
                grid['n_components'], grid['covariance_type'], grid['reg_covar'],
                grid['n_init'])
            if k < n_features
        ]

    raise ValueError(f'unknown clustering method {method!r}; '
                     f'expected one of {sorted(ALGORITHMS)}')


def fit(method: str, representation: DataFrame, **params) -> ClusterSolution:
    """Run one algorithm on the representation it declares.

    ``params`` may carry ``correlation_method``, which the distance-based algorithms
    record for provenance and the profile-based ones do not accept; it is stripped for
    the latter so one grid entry can be passed to either.
    """
    if method not in ALGORITHMS:
        raise ValueError(f'unknown clustering method {method!r}; '
                         f'expected one of {sorted(ALGORITHMS)}')

    if config.REPRESENTATION_BY_METHOD[method] == 'feature_profile':
        params = {k: v for k, v in params.items() if k != 'correlation_method'}

    return ALGORITHMS[method](representation, **params)
