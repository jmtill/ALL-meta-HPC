"""Cluster liquids under each candidate feature set and score the partitions.

The question is not "which clustering is best" but "which feature set produces
clusterings that hold up", so the same model grid is run against every feature set
and the scores are compared across feature sets rather than across models.

Four families of score, because each answers a different objection:

* internal (silhouette, Calinski-Harabasz, Davies-Bouldin) -- is the partition
  geometrically clean? Comparable to the earlier numbers in
  deprecated/notebooks_classes/3_clus_class_validation.ipynb.
* chemical agreement (ARI, AMI vs the RDKit Butina cluster_id) -- does it recover
  chemical families, or has it found something the structure does not support?
* stability (mean pairwise ARI across subsamples) -- would a slightly different set
  of liquids give the same answer?
* error separation (eta-squared of measured dispensing error across clusters) -- does
  it separate the property we actually care about? Only the benchmarked liquids
  carry a label,
  so this is indicative and is reported with its n.
"""

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import clone
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    HDBSCAN,
    KMeans,
    SpectralClustering,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler

from allmeta import columns as cols
from allmeta.paths import CLUSTERING_DIR, RANDOM_STATE
from allmeta.screening import attach_factors, load_feature_sets
from allmeta.subsets import load_subset

# Volume used to summarise a liquid's dispensing error. The eight protocol
# conditions split into two regimes (>=45 uL at rate 250/300, <=30 uL at 20/20), so
# averaging across all of them would blend two different pipetting behaviours.
REFERENCE_VOLUME: int = 10

# Minimum within-cluster degrees of freedom required before eta-squared is
# reported. It scales with the labelled liquid count rather than being pinned to
# it: with n labelled liquids the usable partition caps at n - 2 clusters, above
# which the statistic measures granularity, not error separation.
MIN_WITHIN_DF: int = 2

N_RESAMPLES: int = 10
RESAMPLE_FRACTION: float = 0.8

METRICS_PATH: Path = CLUSTERING_DIR / 'metrics.csv'
# Gzipped: 1140 combinations x ~1200 liquids is ~57 MB uncompressed, which is
# needless churn in a git repo. pandas reads and writes .csv.gz transparently.
LABELS_PATH: Path = CLUSTERING_DIR / 'labels.csv.gz'


def cluster_models() -> dict[str, object]:
    """The model grid, extending the one used in the earlier class project.

    Spectral clustering uses a nearest-neighbour affinity rather than the default
    RBF: on this data RBF collapses to a giant cluster plus outliers (k=5 returns 2
    non-empty clusters) and costs 120x more per fit.
    """
    models: dict[str, object] = {
        f'kmeans_k{k}': KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        for k in range(2, 11)
    }
    for k in (3, 5):
        models[f'gmm_k{k}'] = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
        models[f'agg_ward_k{k}'] = AgglomerativeClustering(n_clusters=k, linkage='ward')
    for k in (3, 5, 7):
        models[f'birch_k{k}'] = Birch(n_clusters=k)
        models[f'spectral_k{k}'] = SpectralClustering(
            n_clusters=k, affinity='nearest_neighbors', n_neighbors=15,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    for ms in (5, 10):
        # eps=1.0 is a placeholder: fit_labels() overwrites it per-embedding via
        # estimate_eps(), since a fixed eps means something different on each of
        # the raw/pca/umap representations.
        models[f'dbscan_ms{ms}'] = DBSCAN(eps=1.0, min_samples=ms)
    for mcs in (5, 15, 30):
        # min_cluster_size is a point count, not a distance, so unlike DBSCAN's
        # eps it is already scale-invariant across representations.
        models[f'hdbscan_mcs{mcs}'] = HDBSCAN(min_cluster_size=mcs)

    return models


# Projections applied after standardisation. 'raw' means standardised only.
REPRESENTATIONS: tuple[str, ...] = ('raw', 'pca', 'umap')


def embed(data: DataFrame, representation: str) -> np.ndarray:
    """Standardise and optionally project one feature matrix.

    UMAP is imported lazily: it pulls in numba and costs several seconds of import
    time that the raw and PCA paths should not pay.
    """
    scaled = RobustScaler().fit_transform(data)
    # scaled = StandardScaler().fit_transform(data)

    if representation == 'raw':
        return scaled
    if representation == 'pca':
        return PCA(n_components=0.90, random_state=RANDOM_STATE).fit_transform(scaled)
    if representation == 'umap':
        import umap
        reducer = umap.UMAP(
            n_neighbors=25, min_dist=0.3, n_components=2,
            random_state=RANDOM_STATE,
        )
        return reducer.fit_transform(scaled)

    raise ValueError(f'unknown representation {representation!r}')


def k_distance_curve(embedding: np.ndarray, min_samples: int) -> np.ndarray:
    """Sorted distance from every point to its min_samples-th nearest neighbour.

    The building block of the standard DBSCAN eps heuristic (Ester et al. 1996):
    flat where points sit inside clusters, rising steeply once a point's
    neighbourhood spills into sparser space. Exposed separately from
    estimate_eps() so a diagnostic plot can show the curve itself, not just the
    single eps value picked from it.
    """
    from sklearn.neighbors import NearestNeighbors

    neighbours = NearestNeighbors(n_neighbors=min_samples).fit(embedding)
    distances, _ = neighbours.kneighbors(embedding)

    return np.sort(distances[:, -1])


def estimate_eps(embedding: np.ndarray, min_samples: int) -> float:
    """DBSCAN eps via the k-distance elbow.

    The elbow -- the point of maximum vertical drop below the chord joining the
    curve's endpoints -- is the conventional eps estimate. Recomputing this per
    embedding (rather than fixing eps in the model grid) is what lets the same
    dbscan_ms{k} entry work across the raw/pca/umap representations, which live on
    very different scales.
    """
    k_distances = k_distance_curve(embedding, min_samples)

    n = len(k_distances)
    x = np.linspace(0, 1, n)
    chord = k_distances[0] + (k_distances[-1] - k_distances[0]) * x
    knee = int(np.argmax(chord - k_distances))

    return float(k_distances[knee])


def fit_labels(model, embedding: np.ndarray) -> np.ndarray:
    """Cluster labels from a fitted model, handling estimators without ``labels_``."""
    fitted = clone(model)
    if isinstance(fitted, DBSCAN):
        fitted.eps = estimate_eps(embedding, fitted.min_samples)
    if hasattr(fitted, 'fit_predict'):
        return np.asarray(fitted.fit_predict(embedding))

    fitted.fit(embedding)

    return np.asarray(fitted.labels_)


def internal_scores(embedding: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Silhouette, Calinski-Harabasz and Davies-Bouldin, or NaN if degenerate.

    A partition with one cluster, or with as many clusters as points, is undefined
    for all three rather than merely bad. DBSCAN/HDBSCAN noise points (label -1)
    are excluded before scoring -- "unclustered" is not a cluster, and scoring it
    as one would distort all three geometrically.
    """
    known = labels != -1
    embedding, labels = embedding[known], labels[known]

    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return {'silhouette': np.nan, 'calinski_harabasz': np.nan,
                'davies_bouldin': np.nan}

    return {
        'silhouette': float(silhouette_score(embedding, labels)),
        'calinski_harabasz': float(calinski_harabasz_score(embedding, labels)),
        'davies_bouldin': float(davies_bouldin_score(embedding, labels)),
    }


def chemical_agreement(labels: np.ndarray, chem_clusters: Series) -> dict[str, float]:
    """ARI and AMI against the RDKit Butina chemical clusters.

    Restricted to liquids that carry a chemical cluster id; a handful in subset A do
    not, and scoring them as a shared 'missing' class would invent agreement. The
    exact count is in subset_manifest.csv as n_liquids - n_with_cluster_id. Also
    excludes DBSCAN/HDBSCAN noise points (label -1), for the same reason
    internal_scores() does: "unclustered" is not a class to score agreement against.
    """
    known = chem_clusters.notna().to_numpy() & (labels != -1)
    if known.sum() < 2:
        return {'ari_chemical': np.nan, 'ami_chemical': np.nan, 'n_chemical': 0}

    truth = chem_clusters[known].to_numpy()

    return {
        'ari_chemical': float(adjusted_rand_score(truth, labels[known])),
        'ami_chemical': float(adjusted_mutual_info_score(truth, labels[known])),
        'n_chemical': int(known.sum()),
    }


def error_separation(labels: np.ndarray, errors: Series) -> dict[str, float]:
    """Fraction of dispensing-error variance lying between clusters (eta-squared).

    Only the benchmarked liquids carry a measured error, so this rests on a handful
    of points and is reported alongside its n rather than on its own. DBSCAN/HDBSCAN
    noise points (label -1) are excluded too: treating "unclustered" as a group would
    attribute structure to non-clustering.
    """
    known = errors.notna().to_numpy() & (labels != -1)
    values = errors[known].to_numpy(dtype=float)
    groups = labels[known]

    n_occupied = int(len(np.unique(groups)))
    base = {'n_labelled': int(known.sum()), 'n_labelled_clusters': n_occupied}

    if len(values) < 3 or n_occupied < 2:
        return {'error_eta_squared': np.nan, **base}

    # eta-squared rises to 1.0 mechanically as clusters approach singletons,
    # regardless of the errors: n liquids in n clusters gives exactly 1.0, and n in
    # n - 1 gives ~0.99995. Require at least MIN_WITHIN_DF within-cluster
    # degrees of freedom so the number reflects the errors rather than the
    # granularity. n_labelled_clusters stays visible to explain the blanks.
    if len(values) - n_occupied < MIN_WITHIN_DF:
        return {'error_eta_squared': np.nan, **base}

    grand_mean = values.mean()
    ss_total = float(((values - grand_mean) ** 2).sum())
    if ss_total == 0.0:
        return {'error_eta_squared': np.nan, **base}

    ss_between = float(sum(
        (groups == g).sum() * (values[groups == g].mean() - grand_mean) ** 2
        for g in np.unique(groups)
    ))

    return {'error_eta_squared': ss_between / ss_total, **base}


def resample_embeddings(data: DataFrame, representation: str,
                        n_resamples: int = N_RESAMPLES,
                        fraction: float = RESAMPLE_FRACTION,
                        random_state: int = RANDOM_STATE
                        ) -> list[tuple[pd.Index, np.ndarray]]:
    """Embed ``n_resamples`` random subsamples once, for reuse across every model.

    The representation is refit inside each resample rather than fitted on the full
    data and sliced -- otherwise the embedding has already seen every liquid and the
    stability it reports is flattering. Because the same resamples are handed to
    every model, stability numbers are comparable across the model grid, and the
    expensive fits (UMAP at several seconds each) are paid once per subsample rather
    than once per subsample per model.
    """
    rng = np.random.default_rng(random_state)
    size = max(3, int(round(len(data) * fraction)))

    embeddings: list[tuple[pd.Index, np.ndarray]] = []
    for _ in range(n_resamples):
        idx = rng.choice(len(data), size=size, replace=False)
        sample = data.iloc[idx]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                embeddings.append((sample.index, embed(sample, representation)))
            except (ValueError, np.linalg.LinAlgError):
                continue

    return embeddings


def stability(embeddings: list[tuple[pd.Index, np.ndarray]], model) -> dict[str, float]:
    """Mean pairwise ARI between clusterings of overlapping random subsamples.

    Each pair is compared only on the liquids both subsamples contain, so the score
    measures agreement on shared points rather than being diluted by the ones only
    one of them saw. Points either resample calls noise (label -1, for DBSCAN/
    HDBSCAN) are dropped from that pair's comparison: agreeing on "unclustered" is
    not evidence of a reproducible partition.
    """
    assignments: list[Series] = []
    for index, embedding in embeddings:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                assignments.append(Series(fit_labels(model, embedding), index=index))
            except (ValueError, np.linalg.LinAlgError):
                continue

    scores: list[float] = []
    for left, right in itertools.combinations(assignments, 2):
        shared = left.index.intersection(right.index)
        shared = shared[(left[shared] != -1) & (right[shared] != -1)]
        if len(shared) < 3:
            continue
        scores.append(adjusted_rand_score(left[shared], right[shared]))

    if not scores:
        return {'stability_ari': np.nan, 'n_stability_pairs': 0}

    return {'stability_ari': float(np.mean(scores)),
            'n_stability_pairs': len(scores)}


def reference_errors(names: Series) -> Series:
    """Measured dispensing error at the reference volume, aligned to ``names``."""
    labelled = load_subset('C')
    at_volume = labelled.loc[labelled['volume'] == REFERENCE_VOLUME]
    lookup = dict(zip(at_volume['Name'], at_volume[cols.TARGET]))

    return names.map(lookup)


def evaluate_subset(subset: str, feature_sets: dict[str, list[str]] | None = None,
                    n_resamples: int = N_RESAMPLES,
                    with_stability: bool = True) -> tuple[DataFrame, DataFrame]:
    """Score every (feature set x representation x model) combination for one subset.

    Returns the tidy metric table and the per-liquid cluster labels, so a partition
    can be inspected without refitting.
    """
    frame = attach_factors(load_subset(subset))
    sets = load_feature_sets(subset) if feature_sets is None else feature_sets
    models = cluster_models()

    errors = reference_errors(frame['Name'])
    chem = frame['cluster_id']

    rows: list[dict] = []
    label_frames: list[DataFrame] = []

    for set_name, features in sets.items():
        usable = [f for f in features if f in frame.columns]
        if len(usable) < 2:
            # A factor set defined on the other subset will not resolve here. Skip
            # it rather than clustering on whatever happens to be left over.
            continue
        data = frame[usable]

        for representation in REPRESENTATIONS:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                embedding = embed(data, representation)

            resamples = (
                resample_embeddings(data, representation, n_resamples=n_resamples)
                if with_stability else []
            )

            for model_name, model in models.items():
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try:
                        labels = fit_labels(model, embedding)
                    except (ValueError, np.linalg.LinAlgError) as err:
                        rows.append({
                            'subset': subset, 'feature_set': set_name,
                            'n_features': len(features),
                            'representation': representation, 'model': model_name,
                            'n_clusters': np.nan, 'error': str(err),
                        })
                        continue

                    n_noise = int((labels == -1).sum())
                    record = {
                        'subset': subset,
                        'feature_set': set_name,
                        'n_features': len(features),
                        'representation': representation,
                        'model': model_name,
                        # Excludes the DBSCAN/HDBSCAN noise label (-1), which is not
                        # a cluster; n_noise/noise_fraction report it separately so a
                        # combination that calls most of the data noise is visible
                        # rather than just showing up as a small n_clusters.
                        'n_clusters': int(len(np.unique(labels[labels != -1]))),
                        'n_noise': n_noise,
                        'noise_fraction': n_noise / len(labels),
                        'error': '',
                    }
                    record.update(internal_scores(embedding, labels))
                    record.update(chemical_agreement(labels, chem))
                    record.update(error_separation(labels, errors))

                    if with_stability:
                        record.update(stability(resamples, model))

                rows.append(record)
                label_frames.append(DataFrame({
                    'subset': subset, 'feature_set': set_name,
                    'representation': representation, 'model': model_name,
                    'Name': frame['Name'].to_numpy(), 'label': labels,
                }))

            print(f'  {subset}/{set_name}/{representation} done', flush=True)

    return pd.DataFrame(rows), pd.concat(label_frames, ignore_index=True)


def run(subsets: tuple[str, ...] = ('A', 'B'),
        n_resamples: int = N_RESAMPLES,
        with_stability: bool = True) -> DataFrame:
    """Evaluate every subset and write the tidy metric table."""
    CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)

    metric_frames: list[DataFrame] = []
    label_frames: list[DataFrame] = []
    for subset in subsets:
        metrics, labels = evaluate_subset(subset, n_resamples=n_resamples,
                                          with_stability=with_stability)
        metric_frames.append(metrics)
        label_frames.append(labels)

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(METRICS_PATH, index=False)
    pd.concat(label_frames, ignore_index=True).to_csv(LABELS_PATH, index=False)

    return metrics


def evaluate_consensus(subsets: tuple[str, ...] = ('A', 'B'),
                       n_resamples: int = N_RESAMPLES) -> DataFrame:
    """Score the supervised consensus feature set on the unlabelled subsets.

    Closes the loop: features chosen against measured dispensing error on the
    liquids are put through the same clustering evaluation as the unsupervised sets,
    so they can be compared on equal terms. Appends to the existing metric table
    rather than replacing it, and replaces any earlier consensus rows so re-running
    does not accumulate duplicates.
    """
    from allmeta.supervised import FREQUENCY_PATH, consensus_features

    if not FREQUENCY_PATH.is_file():
        raise FileNotFoundError('run allmeta.supervised first')

    features = consensus_features(pd.read_csv(FREQUENCY_PATH))
    feature_sets = {'supervised_consensus': features}

    metric_frames: list[DataFrame] = []
    label_frames: list[DataFrame] = []
    for subset in subsets:
        metrics, labels = evaluate_subset(subset, feature_sets=feature_sets,
                                          n_resamples=n_resamples)
        metric_frames.append(metrics)
        label_frames.append(labels)

    fresh = pd.concat(metric_frames, ignore_index=True)
    fresh_labels = pd.concat(label_frames, ignore_index=True)

    for path, addition in ((METRICS_PATH, fresh), (LABELS_PATH, fresh_labels)):
        if path.is_file():
            existing = pd.read_csv(path)
            existing = existing[existing['feature_set'] != 'supervised_consensus']
            addition = pd.concat([existing, addition], ignore_index=True)
        addition.to_csv(path, index=False)

    return fresh


if __name__ == '__main__':
    result = run()
    print(f'\nwrote {len(result)} rows to {METRICS_PATH}')
