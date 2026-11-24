"""What each feature-clustering algorithm actually sees (implementation plan, Phase 3).

The objects being clustered are the **features**, not the liquids. That single fact
decides everything here, and it is the one most easily lost: ``allmeta.clustering``
clusters liquids, on a matrix of the same name and the same shape, and the two are not
interchangeable.

Given ``X`` of shape ``(n_liquids, n_features)``, two representations of the features
are useful, and no algorithm should be handed the one it cannot interpret:

``correlation_distance``
    An ``(n_features, n_features)`` matrix of ``d = 1 - |r|``. Two descriptors are
    close when they carry the same information, whatever the sign, and the metric is
    non-Euclidean. Consumed by hierarchical, HDBSCAN and K-Medoids, all of which accept
    a precomputed distance.

``feature_profile``
    ``X`` standardised column-wise and transposed, ``(n_features, n_liquids)``, so each
    row is one descriptor's profile across the liquids and ordinary Euclidean geometry
    applies. Consumed by KMeans and GMM, which need points in a space, not distances.
    Returned in a compacted form -- see ``compact_profiles`` -- because the raw shape is
    45 points in 720 dimensions and that breaks GMM outright.

Standardisation before transposing is not cosmetic. Untransformed, the Euclidean
distance between two descriptor profiles is dominated by whichever descriptor has the
larger units, so MWt would sit far from everything for reasons of measurement rather
than chemistry.

Neither is the compaction. The profile matrix here is 45 descriptors by 720 liquids, and
45 points can span at most 44 dimensions, so 676 of those coordinates carry exactly zero
variance. Left in, they cost a GMM fit 6.6 seconds and ask it to estimate a 720-
dimensional covariance from 45 points. ``compact_profiles`` projects them away, which
preserves every pairwise Euclidean distance to floating-point noise -- it is not an
approximation, it is the removal of coordinates that hold no information.

The optional common embedding of Step 3.3 -- MDS from the correlation distance, so that
the Euclidean algorithms could consume the correlation geometry too -- is deliberately
not implemented. The plan marks it optional and says it should not be the default, and
an MDS embedding introduces its own stress and dimensionality choices that would have
to be searched and reported. If it is added later it belongs here, beside the two it
would sit alongside, and ``config.REPRESENTATION_BY_METHOD`` is where a method would
declare it consumes it.
"""

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from allmeta.feature_selection import config
from allmeta.screening import correlation_distance, correlation_matrix

CORRELATION_METHODS: tuple[str, ...] = ('pearson', 'spearman')


def _feature_block(frame: DataFrame, features: list[str] | None) -> DataFrame:
    """The float feature block, validated."""
    columns = list(frame.columns if features is None else features)

    missing = [f for f in columns if f not in frame.columns]
    if missing:
        raise ValueError(f'frame is missing requested features: {missing}')
    if len(columns) < 2:
        raise ValueError(
            f'clustering features needs at least 2 of them, got {len(columns)}')

    block = frame[columns].astype(float)
    if block.isna().any().any():
        raise ValueError('feature block contains missing values; complete-case only')

    return block


def correlation_representation(frame: DataFrame, features: list[str] | None = None,
                               method: str = 'spearman',
                               ) -> tuple[DataFrame, DataFrame]:
    """Absolute feature correlation and the distance ``1 - |r|`` derived from it.

    Both are returned because both are read downstream and recomputing one from the
    other in two places is how they drift: the correlation matrix is what the
    correlation-coherence metrics of Phase 5 are computed from, the distance is what the
    algorithms consume.

    Delegates to ``allmeta.screening``, which already owns the symmetrisation and the
    zeroed diagonal -- floating-point asymmetry in a correlation matrix is enough to
    make scipy reject the condensed form, and that fix should exist once.

    Spearman by default: several descriptors are strongly monotonic but not linear in
    each other (Tc against BPt, Parachor against MVol), and Pearson understates those.

    Computed on whatever rows it is handed. Handing it the training rows only is the
    caller's job, and ``config.EXCLUDE_LABELLED_FROM_CLUSTER_FIT`` is the switch that
    governs it for the pipeline driver.
    """
    if method not in CORRELATION_METHODS:
        raise ValueError(f'unknown correlation method {method!r}; '
                         f'expected one of {CORRELATION_METHODS}')

    block = _feature_block(frame, features)
    rho = correlation_matrix(block, method=method)
    distance = DataFrame(correlation_distance(rho),
                         index=rho.index, columns=rho.columns)

    return rho, distance


def feature_profile_representation(frame: DataFrame, features: list[str] | None = None,
                                   compact: bool = True,
                                   ) -> tuple[DataFrame, StandardScaler]:
    """Standardise the feature columns and transpose, so each row is one feature.

    Returns ``(profiles, scaler)``. The scaler comes back because it is fitted state:
    a later frame -- another resample, a held-out fold -- must be transformed with
    *this* scaler through ``apply_feature_profile`` rather than refitted, or the two
    profile matrices are not in the same space and any comparison between them is
    meaningless.

    ``compact`` drops the zero-variance coordinates; see ``compact_profiles``. Pass
    ``False`` to get the raw ``(n_features, n_liquids)`` matrix, which is what a plot
    of one descriptor's profile across the liquids needs.
    """
    block = _feature_block(frame, features)

    scaler = StandardScaler().fit(block)
    scaled = DataFrame(scaler.transform(block), index=block.index,
                       columns=block.columns)
    profiles = scaled.T

    return (compact_profiles(profiles) if compact else profiles), scaler


def compact_profiles(profiles: DataFrame) -> DataFrame:
    """Drop the profile coordinates that carry no variance. Exactly lossless.

    ``n`` points span at most ``n - 1`` dimensions, so a profile matrix with more
    liquids than descriptors is rank-deficient by construction. Rotating onto the
    non-degenerate axes preserves all pairwise Euclidean distances -- verified at
    ~1e-13 absolute on the real data.

    What that does and does not guarantee, because the difference matters:

    * Everything **distance-based** is unchanged exactly. KMeans returns the identical
      partition, and silhouette, Davies-Bouldin and Calinski-Harabasz return identical
      values, on the real data as well as in the tests.
    * **GMM is not distance-based** and its partition CAN change, because it estimates a
      variance per dimension and the dropped dimensions were not free: each contributed
      a ``reg_covar`` floor term to the log-likelihood, so 676 of them dominated the
      objective with noise. The change is a correction, not a loss -- and a 6.6 second
      fit becomes a 0.11 second one, because the alternative was estimating a
      720-dimensional covariance from 45 points.

    The columns are named ``profile_axis_1`` up rather than ``PC1`` up, because they
    are a change of basis for computational sanity, not a dimensionality-reduction step
    a reader should interpret. Nothing is discarded and no variance target is applied.
    """
    n_objects, n_dimensions = profiles.shape
    rank = min(n_objects - 1, n_dimensions)

    if rank >= n_dimensions:
        return profiles

    # Done as a bare thin SVD rather than through sklearn's PCA, which takes 1.2
    # SECONDS on this 45x720 shape against 1.3 ms here -- a 900x difference that is
    # paid once per resample, so it decides whether the Phase 7 stability sweep is
    # minutes or hours. The two are the same computation: centring then rotating onto
    # the right singular vectors, with the scores being U * S. Nothing about PCA's
    # extra bookkeeping (components_, explained variance, sign flipping) is used here,
    # because this is a change of basis and not a component analysis.
    values = profiles.to_numpy(dtype=float)
    left, singular, _ = np.linalg.svd(values - values.mean(axis=0), full_matrices=False)
    rotated = left[:, :rank] * singular[:rank]

    return DataFrame(
        rotated, index=profiles.index,
        columns=[f'profile_axis_{i}' for i in range(1, rank + 1)])


def apply_feature_profile(frame: DataFrame, scaler: StandardScaler,
                          features: list[str]) -> DataFrame:
    """Feature profiles for a new frame, using an already-fitted scaler.

    Returned UNCOMPACTED, and deliberately so. Compaction is a rotation of the liquid
    axes, so it is only defined for the liquid set it was fitted on -- a frame with a
    different number of liquids is not even dimensionally compatible with it. That is
    not a limitation in practice: the profile space exists to cluster features within
    one frame, and what transfers to other liquids is the resulting cluster
    MEMBERSHIP, carried by ``pca.apply_loadings``, never the space itself.
    """
    block = _feature_block(frame, features)
    scaled = DataFrame(scaler.transform(block), index=block.index,
                       columns=block.columns)

    return scaled.T


def representation_for(method: str, frame: DataFrame,
                       features: list[str] | None = None,
                       correlation_method: str = 'spearman'):
    """The representation one clustering algorithm consumes, by name.

    The single dispatch point, so an algorithm cannot quietly be fed the wrong
    geometry: ``config.REPRESENTATION_BY_METHOD`` declares what each method takes, and
    a method with no declaration raises here rather than silently defaulting.
    """
    kind = config.REPRESENTATION_BY_METHOD.get(method)
    if kind is None:
        raise ValueError(
            f'{method!r} declares no representation; add it to '
            'config.REPRESENTATION_BY_METHOD before implementing it')

    if kind == 'correlation_distance':
        return correlation_representation(frame, features, method=correlation_method)[1]
    if kind == 'feature_profile':
        return feature_profile_representation(frame, features)[0]

    raise ValueError(f'{method!r} declares unknown representation {kind!r}')


def describe(distance: DataFrame) -> dict:
    """Shape and range summary of a distance matrix, for the run log.

    Cheap enough to always compute and the fastest way to notice that a distance matrix
    is not what it should be -- a non-zero diagonal or a value above 1 means the
    correlation input was wrong, not that the clustering will merely be poor.
    """
    values = distance.to_numpy(dtype=float)
    off_diagonal = values[~np.eye(len(values), dtype=bool)]

    return {
        'n_features': int(len(values)),
        'symmetric': bool(np.allclose(values, values.T)),
        'zero_diagonal': bool(np.allclose(np.diag(values), 0.0)),
        'min_distance': float(off_diagonal.min()),
        'max_distance': float(off_diagonal.max()),
        'mean_distance': float(off_diagonal.mean()),
    }
