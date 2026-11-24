"""How regression error scales with distance from the training set, and with how many
liquids are in it -- for each model in ``MODELS``.

Two questions, one experiment, asked on the 10 uL benchmark table (one row per liquid,
so a liquid and a sample are the same thing here and no grouping is needed):

1. **Extrapolation.** Does held-out error grow with how far the test liquid sits from
   the training liquids in descriptor space? Distance is the nearest-neighbour
   Euclidean distance in the *standardised* feature space, with the standardiser
   fitted on the training split only -- an unstandardised distance would be MWt and Tc
   and nothing else.
2. **Sample size.** How does held-out error fall as training liquids are added, from
   N-1 down to 2?

Both are answered by repeated random liquid splits rather than by a single
leave-one-out pass, because one pass gives sixteen numbers and no way to separate
"far from training" from "unlucky liquid". For every training size k the script draws
``N_DRAWS`` training subsets (or enumerates them exhaustively when there are fewer
than that), fits each model, and records the per-test-liquid squared error with that
liquid's distance to the training set beside it. Every model sees the same training
subsets, so differences between models are differences between models and not between
draws. Each model's figures go to its own subdirectory of ``OUT_DIR`` and its tables
to the matching subdirectory of ``DATA_OUT_DIR``.

THREE THINGS TO READ CAREFULLY IN THE OUTPUT

* **RMSE is always POOLED over every held-out prediction at a given size, never
  averaged over splits.** A split at k = 12 tests four liquids and a split at k = 15
  tests one, so a per-split RMSE is a four-liquid RMSE in one case and a bare absolute
  error in the other. Averaging those across k would put a downward trend into the
  figure that is an artefact of the test-set shrinking. The per-split distribution is
  written to ``split_results.csv`` for anyone who wants it, and its median and
  quartiles are carried in the summary CSV, but nothing is plotted from them.

* **The distance figure is drawn at ONE fixed training size** (``K_DISTANCE``).
  Pooling across k would confound the two questions: small training sets are
  simultaneously further from the test liquids and worse fits, so a pooled curve would
  show a distance effect that is really a sample-size effect.

* **``all_no_composition`` has 45 features and at most 15 training liquids, so OLS is
  underdetermined at every size swept.** scikit-learn returns the minimum-norm
  least-squares solution, which interpolates the training rows exactly. That is not a
  fit; it is a tie-break among infinitely many exact fits. The figures do not mark it;
  the ``determined`` column of ``rmse_vs_n_train.csv`` does.

* **ElasticNet and Ridge choose their penalty by inner cross-validation on the
  training liquids of each split**, never on the test liquids. At scikit-learn's
  default ``alpha = 1.0`` ElasticNet zeroes every coefficient on this target (sd ~0.09
  against standardised features) and would reproduce the training-mean baseline
  exactly -- a result about the default, not about the model.

The target is a dimensionless FRACTION of the target volume (see
``columns.TARGET_LABEL``): an RMSE of 0.10 is a ten-percentage-point error, not 0.10 uL.

Run:  PYTHONPATH=src python -m allmeta.distance_curves
"""

import warnings
from collections.abc import Callable
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import default_rng
from pandas import DataFrame
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from allmeta import columns as cols
from allmeta.paths import DATA_DIR, DIST_CURVES_DATA_DIR, FIG_DIST_CURVES, RANDOM_STATE
from allmeta.plotting import apply_style, save
from allmeta.subsets import load_hspip

import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# --------------------------------------------------------------------------- #
# inputs and outputs
# --------------------------------------------------------------------------- #

# The per-liquid table at the single volume the current protocol runs. One row per
# liquid, which is what makes an unnested random split honest here: there is no second
# row of the same liquid to leak across the split.
BENCHMARK_CSV: Path = DATA_DIR / 'ALL_data' / 'benchmark_10ul.csv'

OUT_DIR: Path = FIG_DIST_CURVES
# Tables mirror the figure tree under data/ rather than sitting in a data/ folder
# inside figures/.
DATA_OUT_DIR: Path = DIST_CURVES_DATA_DIR

# The leave-one-liquid-out set: every model side by side, one held-out liquid at a time,
# with the identity of that liquid's nearest training neighbour beside its errors.
LOO_DIR: Path = OUT_DIR / 'leave_one_out'

# The six features the purchase list was selected on -- build_candidates.SOBOL_FEATURES,
# restated rather than imported because that script sits outside the package and
# importing it would execute its module-level matplotlib and UMAP setup.
SIX_FEATURES: tuple[str, ...] = (
    'log η', 'SurfTen', 'EdmiW', 'Ant1T', 'Density', 'δP',
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    'six_features': SIX_FEATURES,
    'all_no_composition': tuple(cols.screening_features()),
}

SET_LABELS: dict[str, str] = {
    'six_features': f'6-feature set ({len(SIX_FEATURES)} descriptors)',
    'all_no_composition':
        f'all descriptors minus composition ({len(FEATURE_SETS["all_no_composition"])})',
}
SET_COLOURS: dict[str, str] = {'six_features': '#4c72b0',
                               'all_no_composition': '#c44e52'}

# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

# Inner-CV folds for the penalised models, capped at the training size: at k = 2 the
# inner split is two folds of one liquid each.
N_INNER_FOLDS: int = 5
RIDGE_ALPHAS: np.ndarray = np.logspace(-3, 3, 25)


def inner_folds(n_train: int) -> KFold:
    """Shuffled K-fold over the training liquids of one outer split."""
    return KFold(n_splits=min(N_INNER_FOLDS, n_train), shuffle=True,
                 random_state=RANDOM_STATE)


# Each entry builds a fresh, unfitted estimator for a training set of ``n_train``
# liquids. Directory names under OUT_DIR are these keys.
MODELS: dict[str, Callable[[int], RegressorMixin]] = {
    'LinearRegression': lambda n_train: LinearRegression(),
    # Efficient leave-one-out over the alpha grid (RidgeCV's default when cv=None).
    'Ridge': lambda n_train: RidgeCV(alphas=RIDGE_ALPHAS),
    # l1_ratio stays at 0.5; the alpha path is generated from the training data, so it
    # tracks the target's scale rather than assuming unit variance. 30 alphas rather
    # than 100 and max_iter 2000: on a spot check of 40 splits the chosen alpha and the
    # held-out RMSE were unchanged, at a seventh of the run time.
    'ElasticNet': lambda n_train: ElasticNetCV(alphas=30, cv=inner_folds(n_train),
                                               max_iter=2_000),
    'DecisionTreeRegressor':
        lambda n_train: DecisionTreeRegressor(random_state=RANDOM_STATE),
    'GradientBoostingRegressor':
        lambda n_train: GradientBoostingRegressor(random_state=RANDOM_STATE),
}

MODEL_LABELS: dict[str, str] = {
    'LinearRegression': 'LinearRegression (ordinary least squares)',
    'Ridge': 'Ridge (α by inner leave-one-out CV)',
    'ElasticNet': 'ElasticNet (α by inner 5-fold CV, l1_ratio = 0.5)',
    'DecisionTreeRegressor': 'DecisionTreeRegressor (default, fully grown)',
    'GradientBoostingRegressor': 'GradientBoostingRegressor (default, 100 trees)',
}

MODEL_COLOURS: dict[str, str] = {
    'LinearRegression': '#4c72b0',
    'Ridge': '#55a868',
    'ElasticNet': '#c44e52',
    'DecisionTreeRegressor': '#8172b3',
    'GradientBoostingRegressor': '#dd8452',
}

# --------------------------------------------------------------------------- #
# experiment settings
# --------------------------------------------------------------------------- #

# Training subsets drawn per training size. Where C(N, k) is smaller than this the
# subsets are enumerated exhaustively instead, so k = N-1 is the complete
# leave-one-liquid-out pass rather than a noisy sample of it.
N_DRAWS: int = 300

# Smallest training set swept. Not 1: with a single training liquid every feature has
# zero variance, the standardiser degenerates, and the "model" is that liquid's label
# repeated -- a number that says nothing about the feature set being tested.
K_MIN: int = 2

# Training size the distance figure is drawn at. 12 of 16 leaves four test liquids per
# draw, which spreads the distance axis while holding the sample-size effect fixed.
K_DISTANCE: int = 12

# Distance bins, as quantiles of the pooled distance rather than equal widths: the
# distance distribution is right-skewed and equal-width bins would put almost every row
# in the first bin and single rows in the tail.
N_DISTANCE_BINS: int = 10

# Cluster-bootstrap resamples for every interval in this module. Resampling is by
# LIQUID, never by row: one liquid contributes many rows across draws, so a row
# bootstrap would treat 300 redraws of anisole as 300 independent liquids and return an
# interval several times too narrow.
N_BOOTSTRAP: int = 1000
BOOTSTRAP_ALPHA: float = 0.05


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load_labelled() -> DataFrame:
    """Liquids with a measured 10 uL error and an HSPiP descriptor row.

    ``reagent-alcohol`` carries a label but no ``hspip_name`` -- it is a mixture, not a
    compound in the sheet -- so it has no descriptor vector and cannot enter a
    regression. Dropped here, and reported by the caller rather than dropped silently.
    """
    bench = pd.read_csv(BENCHMARK_CSV)
    named = bench.dropna(subset=['hspip_name'])

    frame = named.merge(
        load_hspip(), left_on='hspip_name', right_on='Name', how='inner',
        validate='many_to_one',
    )

    keep = ['liquid', 'Name', cols.TARGET] + list(cols.NUM_COLUMNS)

    return frame[keep].sort_values('liquid', ignore_index=True)


def unlabelled_liquids(frame: DataFrame) -> list[str]:
    """Benchmarked liquids the descriptor join could not keep."""
    return sorted(set(pd.read_csv(BENCHMARK_CSV)['liquid']) - set(frame['liquid']))


# --------------------------------------------------------------------------- #
# one split
# --------------------------------------------------------------------------- #

def standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z-score both matrices using the TRAINING mean and standard deviation.

    Constant-in-training columns get a scale of 1 rather than 0. At k = 2 several
    descriptors are constant across the pair by chance, and dividing by their zero
    spread would put infinities into the distance before the model ever saw them.
    """
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    scale = np.where(scale > 0, scale, 1.0)

    return (train - mean) / scale, (test - mean) / scale


def split_result(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                 test_idx: np.ndarray, model: RegressorMixin | None = None,
                 ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Fit ``model`` (OLS if None) on one split; return split-level scores and
    per-test-liquid detail.

    Distances come out of the same standardised space the model is fitted in, so "far
    from training" means far on the axes the model actually uses. That is why the
    standardiser is fitted per split and not once over all sixteen liquids: a global
    one would let each test liquid help set its own axis scale. The distances depend on
    the split only, never on the model, so every model shares one distance axis.
    """
    x_train, x_test = standardise(x[train_idx], x[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    model = LinearRegression() if model is None else model
    with warnings.catch_warnings():
        # ElasticNetCV walks its alpha path down to near-zero penalties, where an
        # underdetermined problem converges slowly; those alphas lose the inner CV
        # anyway, so their warnings are noise.
        warnings.simplefilter('ignore', ConvergenceWarning)
        predicted = model.fit(x_train, y_train).predict(x_test)
    error = predicted - y_test

    deltas = x_test[:, None, :] - x_train[None, :, :]
    nearest = np.sqrt((deltas ** 2).sum(axis=2)).min(axis=1)
    centroid = np.sqrt(((x_test - x_train.mean(axis=0)) ** 2).sum(axis=1))

    # The training mean is the honest floor: a model that cannot beat it has not used
    # the descriptors for anything.
    baseline = y_train.mean() - y_test

    summary = {
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'baseline_rmse': float(np.sqrt(np.mean(baseline ** 2))),
        'mean_nn_distance': float(nearest.mean()),
        'max_nn_distance': float(nearest.max()),
        'mean_centroid_distance': float(centroid.mean()),
    }

    return summary, error, nearest, centroid


def training_subsets(n_liquids: int, k: int, rng) -> list[np.ndarray]:
    """Training index sets of size ``k``: every one of them when there are few enough.

    C(16, 15) is 16 and C(16, 2) is 120, both below ``N_DRAWS``. Sampling those would
    report a Monte-Carlo spread where the exhaustive answer is available exactly, and
    would understate k = N-1 -- the leave-one-liquid-out pass everything else in this
    project is scored on.
    """
    if comb(n_liquids, k) <= N_DRAWS:
        return [np.array(combo) for combo in combinations(range(n_liquids), k)]

    seen: set[tuple[int, ...]] = set()
    while len(seen) < N_DRAWS:
        seen.add(tuple(sorted(rng.choice(n_liquids, size=k, replace=False).tolist())))

    return [np.array(combo) for combo in sorted(seen)]


def run_experiment(frame: DataFrame, model_name: str = 'LinearRegression',
                   ) -> tuple[DataFrame, DataFrame]:
    """Every (feature set, training size, draw) split, for one model in ``MODELS``.

    Returns the split-level table and the per-test-liquid table. Both are written out;
    every plotted number comes from the second, because pooling squared errors is the
    only aggregation that stays comparable as the test set changes size.
    """
    n_liquids = len(frame)
    y = frame[cols.TARGET].to_numpy(dtype=float)
    liquids = frame['liquid'].to_numpy()
    make_model = MODELS[model_name]

    splits: list[dict] = []
    per_liquid: list[dict] = []

    for set_name, features in FEATURE_SETS.items():
        x = frame[list(features)].to_numpy(dtype=float)
        # One generator per feature set, seeded identically, so both sets -- and every
        # model -- are compared on the SAME training subsets wherever sampling rather
        # than enumeration is used.
        rng = default_rng(RANDOM_STATE)

        for k in range(K_MIN, n_liquids):
            for draw, train_idx in enumerate(training_subsets(n_liquids, k, rng)):
                test_idx = np.setdiff1d(np.arange(n_liquids), train_idx)
                summary, error, nearest, centroid = split_result(
                    x, y, train_idx, test_idx, make_model(k))

                splits.append({'model': model_name, 'feature_set': set_name,
                               'n_features': len(features), 'draw': draw, **summary})

                for position, index in enumerate(test_idx):
                    per_liquid.append({
                        'model': model_name,
                        'feature_set': set_name,
                        'n_features': len(features),
                        'n_train': k,
                        'draw': draw,
                        'liquid': liquids[index],
                        'nn_distance': float(nearest[position]),
                        'nn_distance_sq': float(nearest[position] ** 2),
                        'centroid_distance': float(centroid[position]),
                        'actual': float(y[index]),
                        'predicted': float(y[index] + error[position]),
                        'error': float(error[position]),
                        'squared_error': float(error[position] ** 2),
                    })

    return DataFrame(splits), DataFrame(per_liquid)


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

def pooled_rmse(chunk: DataFrame) -> float:
    """RMSE over every held-out prediction in ``chunk``."""
    return float(np.sqrt(chunk['squared_error'].mean()))


def bootstrap_rmse(chunk: DataFrame) -> tuple[float, float]:
    """Percentile interval for a pooled RMSE, resampling LIQUIDS not rows.

    A bin or a training size holds hundreds of rows but often only a handful of
    distinct liquids, and the rows of one liquid are the same prediction problem seen
    again. Resampling liquids is the unit that varies.
    """
    groups = [group['squared_error'].to_numpy() for _, group in chunk.groupby('liquid')]
    if len(groups) < 2:
        return float('nan'), float('nan')

    rng = default_rng(RANDOM_STATE)
    draws = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        picked = rng.integers(0, len(groups), size=len(groups))
        draws[index] = np.sqrt(np.concatenate([groups[position] for position in picked]).mean())

    return (float(np.quantile(draws, BOOTSTRAP_ALPHA / 2)),
            float(np.quantile(draws, 1 - BOOTSTRAP_ALPHA / 2)))


def bin_by_distance(per_liquid: DataFrame, n_train: int) -> DataFrame:
    """Pooled RMSE within distance bins, at one fixed training size."""
    rows: list[dict] = []
    for set_name, group in per_liquid[per_liquid['n_train'] == n_train].groupby('feature_set'):
        edges = np.quantile(group['nn_distance'], np.linspace(0, 1, N_DISTANCE_BINS + 1))
        # Quantile edges tie when many rows share a distance; unique() collapses those
        # so np.digitize does not create empty bins between identical edges.
        edges = np.unique(edges)
        assignment = np.clip(np.digitize(group['nn_distance'], edges[1:-1]), 0, len(edges) - 2)

        for index in range(len(edges) - 1):
            chunk = group[assignment == index]
            if chunk.empty:
                continue
            lo, hi = bootstrap_rmse(chunk)
            rows.append({
                'feature_set': set_name,
                'n_train': n_train,
                'bin': index,
                'distance_lo': float(edges[index]),
                'distance_hi': float(edges[index + 1]),
                'mean_distance': float(chunk['nn_distance'].mean()),
                'median_distance': float(chunk['nn_distance'].median()),
                'mean_distance_sq': float(chunk['nn_distance_sq'].mean()),
                'n_points': int(len(chunk)),
                'n_liquids': int(chunk['liquid'].nunique()),
                'rmse': pooled_rmse(chunk),
                'rmse_ci_lo': lo,
                'rmse_ci_hi': hi,
            })

    return DataFrame(rows)


def summarise_by_size(per_liquid: DataFrame, splits: DataFrame,
                      n_liquids: int) -> DataFrame:
    """Pooled RMSE at every training size, with the per-split distribution beside it.

    ``rmse`` is the plotted number and is pooled over every held-out prediction at that
    size. ``rmse_split_*`` describe the distribution of per-split RMSEs and are carried
    only so a reader can see how much the answer moves between training subsets; they
    are not comparable across k, because the test set shrinks as k grows.
    """
    rows: list[dict] = []
    for (set_name, k), chunk in per_liquid.groupby(['feature_set', 'n_train']):
        n_features = int(chunk['n_features'].iloc[0])
        by_split = splits[(splits['feature_set'] == set_name) & (splits['n_train'] == k)]
        lo, hi = bootstrap_rmse(chunk)
        rows.append({
            'feature_set': set_name,
            'n_features': n_features,
            'n_train': int(k),
            'n_draws': int(len(by_split)),
            'n_predictions': int(len(chunk)),
            'exhaustive': bool(len(by_split) == comb(n_liquids, int(k))),
            # An OLS fit is determined only when the training rows outnumber the
            # coefficients; below that the reported error is a property of the
            # minimum-norm tie-break, not of the feature set.
            'determined': bool(k > n_features),
            'rmse': pooled_rmse(chunk),
            'rmse_ci_lo': lo,
            'rmse_ci_hi': hi,
            'baseline_rmse': float(np.sqrt(
                (by_split['baseline_rmse'] ** 2 * by_split['n_test']).sum()
                / by_split['n_test'].sum())),
            'mean_nn_distance': float(chunk['nn_distance'].mean()),
            'rmse_split_median': float(by_split['rmse'].median()),
            'rmse_split_q25': float(by_split['rmse'].quantile(0.25)),
            'rmse_split_q75': float(by_split['rmse'].quantile(0.75)),
            'rmse_split_max': float(by_split['rmse'].max()),
        })

    return DataFrame(rows).sort_values(['feature_set', 'n_train'], ignore_index=True)


def leave_one_out(per_liquid: DataFrame, n_liquids: int) -> DataFrame:
    """The k = N-1 pass on its own: one row per liquid per feature set."""
    lolo = per_liquid[per_liquid['n_train'] == n_liquids - 1]

    return lolo.drop(columns=['draw']).sort_values(
        ['feature_set', 'nn_distance'], ignore_index=True)


def loo_neighbours(frame: DataFrame) -> DataFrame:
    """Every training liquid ranked by distance to each held-out liquid, per feature set.

    One row per (feature set, held-out liquid, training liquid): the full distance
    matrix of the leave-one-liquid-out pass, with rank 1 the nearest neighbour. The
    standardiser is refitted on the other N-1 liquids for every held-out liquid, exactly
    as ``split_result`` does, so rank 1's distance is the ``nn_distance`` the models
    were scored against. That also means the matrix is not symmetric: the distance from
    A to B is measured on axes scaled without A, and from B to A on axes scaled without B.
    """
    n_liquids = len(frame)
    y = frame[cols.TARGET].to_numpy(dtype=float)
    liquids = frame['liquid'].to_numpy()

    rows: list[dict] = []
    for set_name, features in FEATURE_SETS.items():
        x = frame[list(features)].to_numpy(dtype=float)

        for held_out in range(n_liquids):
            train_idx = np.delete(np.arange(n_liquids), held_out)
            x_train, x_test = standardise(x[train_idx], x[[held_out]])
            distance = np.sqrt(((x_train - x_test) ** 2).sum(axis=1))

            # Stable sort, so a tie is broken by liquid order and reruns agree.
            for rank, position in enumerate(np.argsort(distance, kind='stable'), start=1):
                index = train_idx[position]
                rows.append({
                    'feature_set': set_name,
                    'liquid': liquids[held_out],
                    'actual': float(y[held_out]),
                    'rank': rank,
                    'neighbour': liquids[index],
                    'neighbour_actual': float(y[index]),
                    'distance': float(distance[position]),
                })

    return DataFrame(rows)


def loo_predictions(per_liquid: dict[str, DataFrame], neighbours: DataFrame,
                    n_liquids: int) -> DataFrame:
    """Every model's leave-one-liquid-out prediction, with the nearest neighbour beside it.

    Long format: one row per (model, feature set, held-out liquid). Two reference rows
    per liquid are added under pseudo-model names, both fitted on the same N-1 liquids:

    * ``nearest_neighbour``: predict the held-out liquid's label as its nearest
      neighbour's label (1-NN regression). If a model cannot beat this, it is adding
      nothing to "find the closest liquid you have measured".
    * ``training_mean``: the baseline every other figure in this module draws.
    """
    last = n_liquids - 1
    models = pd.concat([table[table['n_train'] == last] for table in per_liquid.values()],
                       ignore_index=True)
    models = models[['model', 'feature_set', 'liquid', 'actual', 'predicted',
                     'nn_distance']]

    nearest = neighbours[neighbours['rank'] == 1]
    merged = models.merge(nearest[['feature_set', 'liquid', 'neighbour',
                                   'neighbour_actual', 'distance']],
                          on=['feature_set', 'liquid'], how='left', validate='many_to_one')
    # Two independent computations of the same number; if they disagree the neighbour
    # named in the tables is not the one the models were scored against.
    if not np.allclose(merged['nn_distance'], merged['distance']):
        raise ValueError('nearest-neighbour distance disagrees with split_result')
    merged = merged.drop(columns=['nn_distance'])

    total = nearest.groupby('feature_set')['actual'].sum()
    reference = nearest[['feature_set', 'liquid', 'actual', 'neighbour',
                         'neighbour_actual', 'distance']]
    one_nn = reference.assign(model='nearest_neighbour',
                              predicted=reference['neighbour_actual'])
    mean = reference.assign(
        model='training_mean',
        predicted=(reference['feature_set'].map(total) - reference['actual']) / last)

    table = pd.concat([merged, one_nn, mean], ignore_index=True)
    table['error'] = table['predicted'] - table['actual']
    table['abs_error'] = table['error'].abs()
    table = table.rename(columns={'distance': 'nn_distance'})

    return table[['feature_set', 'liquid', 'actual', 'neighbour', 'neighbour_actual',
                  'nn_distance', 'model', 'predicted', 'error', 'abs_error']].sort_values(
        ['feature_set', 'nn_distance', 'model'], ignore_index=True)


def loo_per_liquid_summary(predictions: DataFrame, neighbours: DataFrame) -> DataFrame:
    """One row per (feature set, held-out liquid): neighbours, then every model's |error|.

    The table to read first. ``best_model`` is chosen among the fitted models only, not
    the two reference predictors, so it answers "which model" rather than "whether any".
    """
    wide = predictions.pivot_table(index=['feature_set', 'liquid'], columns='model',
                                   values='abs_error')
    wide.columns = [f'abs_error_{name}' for name in wide.columns]

    second = neighbours[neighbours['rank'] == 2].set_index(['feature_set', 'liquid'])
    first = (predictions.drop_duplicates(['feature_set', 'liquid'])
             .set_index(['feature_set', 'liquid'])
             [['actual', 'neighbour', 'neighbour_actual', 'nn_distance']])
    first['second_neighbour'] = second['neighbour']
    first['second_distance'] = second['distance']
    # How close the runner-up is: near 1 means the "nearest" neighbour is nearly a tie
    # and could swap under a small change to the feature set or the scaling.
    first['second_to_first_ratio'] = first['second_distance'] / first['nn_distance']

    fitted = [f'abs_error_{name}' for name in MODELS]
    summary = first.join(wide)
    summary['best_model'] = summary[fitted].idxmin(axis=1).str.removeprefix('abs_error_')

    return summary.reset_index().sort_values(['feature_set', 'nn_distance'],
                                             ignore_index=True)


def neighbour_agreement(neighbours: DataFrame) -> DataFrame:
    """Held-out liquids whose nearest neighbour is the same in both feature sets."""
    nearest = neighbours[neighbours['rank'] == 1].pivot(index='liquid',
                                                        columns='feature_set',
                                                        values='neighbour')
    nearest['same_neighbour'] = nearest.nunique(axis=1) == 1

    return nearest.reset_index()


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def plot_rmse_vs_distance(binned: DataFrame, lolo: DataFrame, k: int,
                          model_name: str, out_dir: Path) -> Path:
    """RMSE against distance from the training set, one panel per feature set.

    Shared y-axis, because the two sets land in the same error range and the comparison
    is the point. The x-axes are NOT shared and cannot be: a nearest-neighbour distance
    in 45 standardised dimensions is larger than one in 6 by construction, so the two
    axes measure different things.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    # Headroom above the highest band or diamond, so the legend sits over empty axis
    # rather than over a point.
    ceiling = max(float(binned['rmse_ci_hi'].max()), float(lolo['error'].abs().max()))

    for ax, (set_name, label) in zip(axes, SET_LABELS.items()):
        curve = binned[binned['feature_set'] == set_name].sort_values('mean_distance')
        colour = SET_COLOURS[set_name]

        ax.fill_between(curve['mean_distance'], curve['rmse_ci_lo'], curve['rmse_ci_hi'],
                        color=colour, alpha=0.18, lw=0, zorder=2,
                        label='95% bootstrap interval (resampled by liquid)')
        ax.plot(curve['mean_distance'], curve['rmse'], marker='o', color=colour,
                lw=2, zorder=3, label=f'pooled RMSE in distance decile (n_train = {k})')

        points = lolo[lolo['feature_set'] == set_name]
        ax.scatter(points['nn_distance'], points['error'].abs(), s=32, color='0.35',
                   alpha=0.85, zorder=4, marker='D',
                   label='|error| per liquid, leave-one-liquid-out (n_train = 15)')

        ax.set_xlabel('nearest-neighbour distance to the training set,\n'
                      '‖x − x′‖ in training-standardised descriptor space')
        ax.set_title(label)
        ax.set_ylim(bottom=0, top=ceiling * 1.35)
        ax.legend(fontsize=8, loc='upper left')

    axes[0].set_ylabel(f'RMSE\n{cols.TARGET_LABEL}')
    fig.suptitle(f'{MODEL_LABELS[model_name]}\n'
                 'Held-out error against distance from the training set', y=1.02)
    fig.text(0.5, -0.09,
             'The y-axis is shared; the x-axes are not, and must not be compared — a '
             'nearest-neighbour distance in 45 standardised dimensions is larger than one '
             'in 6 by construction. Squared distance ‖x − x′‖² is in the CSV: it is '
             'monotone in the plotted distance, so the ordering and the deciles are '
             'identical either way. The grey diamonds sit below the curve because a '
             'single liquid’s |error| is not an RMSE and because they come from a larger '
             'training set (15 rather than 12).',
             ha='center', fontsize=8.5, color='0.3', wrap=True)
    fig.tight_layout()

    return save(fig, out_dir / 'rmse_vs_distance.png')


def plot_rmse_vs_n_train(summary: DataFrame, model_name: str, out_dir: Path) -> Path:
    """Pooled RMSE against the number of training liquids, one panel per feature set."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)

    for ax, (set_name, label) in zip(axes, SET_LABELS.items()):
        group = summary[summary['feature_set'] == set_name].sort_values('n_train')

        ax.bar(group['n_train'], group['rmse'], color=SET_COLOURS[set_name],
               edgecolor='0.25', lw=0.6, zorder=3,
               yerr=[(group['rmse'] - group['rmse_ci_lo']).clip(lower=0),
                     (group['rmse_ci_hi'] - group['rmse']).clip(lower=0)],
               capsize=3, error_kw={'lw': 1.0, 'ecolor': '0.3', 'zorder': 4})
        ax.plot(group['n_train'], group['baseline_rmse'], color='k', ls='--', lw=1.4,
                zorder=5, label='predict the training mean')

        ax.set_xticks(group['n_train'])
        ax.set_xlabel('number of training liquids')
        ax.set_title(label)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(group['n_train'].min() - 0.7, group['n_train'].max() + 0.7)
        ax.set_ylim(bottom=0, top=float(summary['rmse_ci_hi'].max()) * 1.08)

    axes[0].set_ylabel(f'RMSE\n{cols.TARGET_LABEL}')
    fig.suptitle(f'{MODEL_LABELS[model_name]}\n'
                 'Held-out error against training-set size', y=1.02)
    fig.text(0.5, -0.07,
             'Each bar pools every held-out prediction made at that training size — '
             'not an average of per-split RMSEs, which would fall with k purely because '
             'the test set shrinks. Whiskers are 95% bootstrap intervals resampling '
             'liquids. The y-axis is shared between panels.',
             ha='center', fontsize=8.5, color='0.3', wrap=True)
    fig.tight_layout()

    return save(fig, out_dir / 'rmse_vs_n_train.png')


def plot_loo_per_liquid(predictions: DataFrame, out_dir: Path) -> Path:
    """|error| of every model on every held-out liquid, rows labelled by nearest neighbour.

    One panel per feature set. Rows are the held-out liquids sorted by nearest-neighbour
    distance, nearest at the top, so reading down a panel is reading outward from the
    training data. The rows are ordered separately in each panel because the distances
    differ between the sets; the x-axis (|error|) is shared.
    """
    n_liquids = predictions['liquid'].nunique()
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.42 * n_liquids + 2.2), sharex=True)

    # name -> (legend label, scatter style). Fitted models are dots; the two reference
    # predictors get shapes that cannot be mistaken for a model.
    series = {name: (name, {'color': MODEL_COLOURS[name], 'marker': 'o', 's': 30,
                            'edgecolor': 'white', 'lw': 0.4}) for name in MODELS}
    series['nearest_neighbour'] = ("1-NN: predict the nearest neighbour's label",
                                   {'color': 'k', 'marker': 'X', 's': 34})
    series['training_mean'] = ('predict the training mean',
                               {'color': '0.55', 'marker': '|', 's': 60, 'lw': 1.8})
    offsets = np.linspace(-0.3, 0.3, len(series))

    for ax, (set_name, label) in zip(axes, SET_LABELS.items()):
        group = predictions[predictions['feature_set'] == set_name]
        order = (group.drop_duplicates('liquid').sort_values('nn_distance')
                 .reset_index(drop=True))
        row_of = {liquid: position for position, liquid in enumerate(order['liquid'])}

        for band in range(0, len(order), 2):
            ax.axhspan(band - 0.5, band + 0.5, color='0.94', zorder=0, lw=0)

        for offset, (name, (legend, style)) in zip(offsets, series.items()):
            chunk = group[group['model'] == name]
            ax.scatter(chunk['abs_error'], chunk['liquid'].map(row_of) + offset,
                       zorder=3, label=legend, **style)

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f'{row.liquid}  ← {row.neighbour}  (d = {row.nn_distance:.2f})'
                            for row in order.itertuples()], fontsize=8.5)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.grid(axis='y', visible=False)
        ax.set_xlim(left=0)
        ax.set_xlabel(f'|error| on the held-out liquid\n{cols.TARGET_LABEL}')
        ax.set_title(label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle('Leave-one-liquid-out: every model on every held-out liquid\n'
                 'row label = held-out liquid ← its nearest training liquid (distance), '
                 'nearest at the top', y=1.01)
    fig.text(0.5, -0.12,
             'Each row is one fit on the other fifteen liquids. Distance is Euclidean in '
             'descriptor space standardised on those fifteen, so the two panels’ distances '
             'are not comparable (45 dimensions against 6). Rows are sorted within each '
             'panel; the same liquid sits at a different height in each.',
             ha='center', fontsize=8.5, color='0.3', wrap=True)
    fig.tight_layout()

    return save(fig, out_dir / 'per_liquid_errors.png')


def plot_loo_neighbour_map(neighbours: DataFrame, out_dir: Path) -> Path:
    """Distance from each held-out liquid to every training liquid, nearest outlined.

    Rows are held-out liquids, columns are the training liquids it could be compared
    with, both in order of the measured label so a row whose nearest neighbour sits far
    from the diagonal is a liquid whose closest descriptor match behaves differently.
    Colour is distance scaled to each panel (the two sets' distances are not comparable),
    and each row's nearest neighbour is boxed.
    """
    order = (neighbours.drop_duplicates('liquid').sort_values('actual')
             [['liquid', 'actual']].reset_index(drop=True))
    position = {liquid: index for index, liquid in enumerate(order['liquid'])}
    ticks = [f'{row.liquid} ({row.actual:.3f})' for row in order.itertuples()]
    n_liquids = len(order)

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.2))
    for ax, (set_name, label) in zip(axes, SET_LABELS.items()):
        group = neighbours[neighbours['feature_set'] == set_name]
        matrix = np.full((n_liquids, n_liquids), np.nan)
        matrix[group['liquid'].map(position), group['neighbour'].map(position)] = \
            group['distance']

        # Water and methanol sit several times further from everything than any other
        # pair does; a full-range scale spends the whole colour map on those two rows.
        # Capped at the 90th percentile instead, and the colour bar says so. Exact
        # distances are in neighbour_ranking.csv.
        image = ax.imshow(matrix, cmap='viridis_r', aspect='auto',
                          vmax=float(np.quantile(group['distance'], 0.9)))
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, extend='max',
                     label='distance ‖x − x′‖ (standardised on the other 15)')

        nearest = group[group['rank'] == 1]
        for row in nearest.itertuples():
            ax.add_patch(plt.Rectangle(
                (position[row.neighbour] - 0.5, position[row.liquid] - 0.5), 1, 1,
                fill=False, edgecolor='#e8384f', lw=2.2))

        ax.set_xticks(range(n_liquids))
        ax.set_xticklabels(ticks, rotation=90, fontsize=7.5)
        ax.set_yticks(range(n_liquids))
        ax.set_yticklabels(ticks, fontsize=7.5)
        ax.set_xlabel('training liquid (10 uL label)')
        ax.set_title(label)
        ax.grid(False)

    axes[0].set_ylabel('held-out liquid (10 uL label)')

    fig.suptitle('Leave-one-liquid-out nearest neighbours: which training liquid each '
                 'held-out liquid sits closest to\nred box = nearest neighbour; rows and '
                 'columns ordered by measured error', y=1.0)
    fig.tight_layout()

    return save(fig, out_dir / 'nearest_neighbour_map.png')


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def report_model(frame: DataFrame, model_name: str, splits: DataFrame,
                 per_liquid: DataFrame) -> list[dict]:
    """Write one model's figures to ``OUT_DIR / model_name``, tables to ``DATA_OUT_DIR / model_name``.

    Returns one comparison row per feature set for the cross-model table.
    """
    out_dir = OUT_DIR / model_name
    out_data = DATA_OUT_DIR / model_name
    out_data.mkdir(parents=True, exist_ok=True)

    summary = summarise_by_size(per_liquid, splits, len(frame))
    binned = bin_by_distance(per_liquid, K_DISTANCE)
    lolo = leave_one_out(per_liquid, len(frame))

    frame[['liquid', 'Name', cols.TARGET]].to_csv(out_data / 'liquids.csv', index=False)
    splits.to_csv(out_data / 'split_results.csv', index=False)
    per_liquid.to_csv(out_data / 'per_liquid_errors.csv', index=False)
    summary.to_csv(out_data / 'rmse_vs_n_train.csv', index=False)
    binned.to_csv(out_data / 'rmse_vs_distance.csv', index=False)
    lolo.to_csv(out_data / 'leave_one_liquid_out.csv', index=False)

    print(f'\n{model_name}: {len(splits)} splits, {len(per_liquid)} held-out predictions.')
    print(plot_rmse_vs_distance(binned, lolo, K_DISTANCE, model_name, out_dir))
    print(plot_rmse_vs_n_train(summary, model_name, out_dir))

    rows: list[dict] = []
    for set_name in FEATURE_SETS:
        loo = lolo[lolo['feature_set'] == set_name]
        spearman = loo['nn_distance'].corr(loo['squared_error'], method='spearman')
        far = binned[binned['feature_set'] == set_name].sort_values('mean_distance')
        at_size = summary[summary['feature_set'] == set_name].set_index('n_train')
        last = at_size.loc[len(frame) - 1]
        rows.append({
            'model': model_name,
            'feature_set': set_name,
            'lolo_rmse': pooled_rmse(loo),
            'lolo_rmse_ci_lo': float(last['rmse_ci_lo']),
            'lolo_rmse_ci_hi': float(last['rmse_ci_hi']),
            'lolo_baseline_rmse': float(last['baseline_rmse']),
            f'rmse_at_k{K_DISTANCE}': float(at_size.loc[K_DISTANCE, 'rmse']),
            'worst_rmse_any_k': float(at_size['rmse'].max()),
            'worst_k': int(at_size['rmse'].idxmax()),
            'rho_distance_sq_error': float(spearman),
            'nearest_decile_rmse': float(far['rmse'].iloc[0]),
            'furthest_decile_rmse': float(far['rmse'].iloc[-1]),
        })
        print(f'  {set_name:20s} LOLO RMSE {rows[-1]["lolo_rmse"]:.4f} '
              f'(mean baseline {rows[-1]["lolo_baseline_rmse"]:.4f})   '
              f'rho(distance, sq error) = {spearman:+.3f}   '
              f'nearest decile {rows[-1]["nearest_decile_rmse"]:.4f} -> '
              f'furthest {rows[-1]["furthest_decile_rmse"]:.4f}')

    return rows


def report_leave_one_out(frame: DataFrame, per_liquid: dict[str, DataFrame]) -> None:
    """Write the leave-one-liquid-out set to ``LOO_DIR`` (tables under ``DATA_OUT_DIR``).

    Reuses the k = N-1 splits every model already ran, so the predictions here are the
    same numbers as each model's ``leave_one_liquid_out.csv``; what this adds is the
    models side by side and the identity of each held-out liquid's nearest neighbour.
    """
    out_data = DATA_OUT_DIR / LOO_DIR.name
    out_data.mkdir(parents=True, exist_ok=True)

    neighbours = loo_neighbours(frame)
    predictions = loo_predictions(per_liquid, neighbours, len(frame))
    summary = loo_per_liquid_summary(predictions, neighbours)
    agreement = neighbour_agreement(neighbours)

    neighbours.to_csv(out_data / 'neighbour_ranking.csv', index=False)
    predictions.to_csv(out_data / 'predictions.csv', index=False)
    summary.to_csv(out_data / 'per_liquid_summary.csv', index=False)
    agreement.to_csv(out_data / 'neighbour_agreement.csv', index=False)

    print(f'\nleave_one_out: {len(frame)} held-out liquids x {len(FEATURE_SETS)} feature '
          f'sets x {predictions["model"].nunique()} predictors.')
    print(plot_loo_per_liquid(predictions, LOO_DIR))
    print(plot_loo_neighbour_map(neighbours, LOO_DIR))

    for set_name in FEATURE_SETS:
        chunk = predictions[predictions['feature_set'] == set_name]
        rmse = chunk.groupby('model')['error'].apply(lambda e: np.sqrt((e ** 2).mean()))
        wins = summary[summary['feature_set'] == set_name]['best_model'].value_counts()
        print(f'  {set_name:20s} ' + '  '.join(
            f'{name} {value:.4f}' for name, value in rmse.sort_values().items()))
        print(f'  {"":20s} lowest |error| per liquid: '
              + ', '.join(f'{name} {count}' for name, count in wins.items()))
    print(f'  same nearest neighbour in both feature sets for '
          f'{int(agreement["same_neighbour"].sum())} of {len(agreement)} liquids')


def main() -> None:
    apply_style()

    frame = load_labelled()
    dropped = unlabelled_liquids(frame)
    print(f'{len(frame)} liquids with both a 10 uL label and a descriptor row.')
    if dropped:
        print(f'Dropped (no HSPiP descriptor row): {", ".join(dropped)}')

    # Models are independent, so they run one per process. The figures are drawn back
    # in this process, where apply_style() has been called.
    results = Parallel(n_jobs=len(MODELS))(
        delayed(run_experiment)(frame, name) for name in MODELS)

    comparison: list[dict] = []
    for name, (splits, per_liquid) in zip(MODELS, results):
        comparison.extend(report_model(frame, name, splits, per_liquid))

    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_OUT_DIR / 'model_comparison.csv'
    DataFrame(comparison).to_csv(path, index=False)
    print(f'\n{path}')

    report_leave_one_out(frame, {name: per_liquid for name, (_, per_liquid)
                                 in zip(MODELS, results)})


if __name__ == '__main__':
    main()
