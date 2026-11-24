"""Leave-one-liquid-out regression of dispensing error, and its diagnostics.

Batch 4 chose features but never fitted a predictive model, so nothing tested
whether the chosen features actually predict anything. This module does, and it is
built to make a negative result legible rather than to hide one:

* a ``DummyRegressor`` predicting the training mean is evaluated alongside every
  real model. At this liquid count that baseline is hard to beat, and a model that
  fails to beat it has learned nothing regardless of its RMSE.
* scores are computed on predictions **pooled across the outer folds**, so
  every number answers "how well does this do on a liquid it has never seen".
  Per-fold R-squared would be meaningless -- each fold holds out one liquid, whose
  eight rows share a descriptor vector.
* unsupervised feature sets (chosen without ever seeing the target) are evaluated
  next to the supervised ones. If they predict just as well, the supervised
  selection added nothing.

The target is strongly right-skewed (0.0005 to 0.425), so RMSE is dominated by the
few high-error liquids; MAE and Spearman rank correlation are reported beside it.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas import DataFrame
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

from allmeta import columns as cols
from allmeta.paths import FIG_DIR, RANDOM_STATE, REPO_ROOT
from allmeta.plotting import SUBSET_COLOURS, apply_style, annotate_empty, save
from allmeta.scoring import (  # noqa: F401  -- re-exported for the notebooks
    BASELINE, N_BOOTSTRAP, _logo_predictions, baseline_r2, benjamini_hochberg,
    bootstrap_between_r2, bootstrap_pooled_rmse, of_evaluation, permuted_targets,
    pooled_scores, target_variance_decomposition)
from allmeta.screening import attach_derived, attach_factors
from allmeta.subsets import labelled_liquid_count, load_subset

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

REGRESSION_DIR: Path = REPO_ROOT / 'data' / 'regression'
FIG_REGRESSION: Path = FIG_DIR / 'regression'

SCORES_PATH: Path = REGRESSION_DIR / 'scores.csv'
PREDICTIONS_PATH: Path = REGRESSION_DIR / 'predictions.csv'
FOLD_SCORES_PATH: Path = REGRESSION_DIR / 'fold_scores.csv'
ALIASES_PATH: Path = REGRESSION_DIR / 'feature_set_aliases.csv'
DECOMPOSITION_PATH: Path = REGRESSION_DIR / 'target_variance_decomposition.csv'
PERMUTATION_NULL_PATH: Path = REGRESSION_DIR / 'permutation_null.csv'
PERMUTATION_PVALUES_PATH: Path = REGRESSION_DIR / 'permutation_pvalues.csv'
BOOTSTRAP_CI_PATH: Path = REGRESSION_DIR / 'bootstrap_ci.csv'
SCORES_NESTED_PATH: Path = REGRESSION_DIR / 'scores_nested.csv'
RANDOM_BASELINE_PATH: Path = REGRESSION_DIR / 'random_feature_baseline.csv'
PREDICTIONS_NESTED_PATH: Path = REGRESSION_DIR / 'predictions_nested.csv'


# Draws for the permutation null. 1000 rather than the function default of 100 so the
# empirical p can resolve below 0.01: it is bounded below by 1/(n+1), which 100 draws
# put at 0.0099 -- exactly at the threshold a reader would want to test against.
N_PERMUTATIONS: int = 1000

# Bootstrap draws for the between-liquid R-squared interval. Cheap -- each draw is a
# resample of thirteen liquid means -- so there is no reason to economise.

# Workers for the permutation loop. -1 is every core; the estimators inside are asked
# for n_jobs=1 so the parallelism happens here and not twice over.
PERMUTATION_JOBS: int = -1

# Random-feature baseline: how many five-descriptor sets to draw. 200 is enough to
# place a selected set against the distribution's percentiles without pretending to
# resolve its tails.
N_RANDOM_SETS: int = 200

# The subset a feature set was *selected* on, if that subset carries labels. Only C
# does, so only sets namespaced to C are selection-leaky -- but stating it as a
# labelled-subset lookup rather than a literal string test means adding a second
# labelled subset later cannot quietly produce a set that scores as honest.
LABELLED_SUBSETS: frozenset[str] = frozenset({'C'})


def selected_on_labels(feature_set: str) -> bool:
    """Whether this feature set's members were chosen using the target."""
    subset, separator, _ = feature_set.partition('::')

    return bool(separator) and subset in LABELLED_SUBSETS


def models(n_jobs: int = -1) -> dict[str, object]:
    """Estimators evaluated on every feature set.

    Deliberately small and low-variance: a training fold is eight volumes of each
    liquid but one, so anything with more capacity is fitting that fold, not the
    chemistry.

    ``n_jobs`` is threaded through to the two ensembles so a caller that is *itself*
    parallel can turn their internal threading off. It matters more than it looks:
    on a 96-row training fold the thread dispatch costs more than the trees do, and
    the permutation null measures 1.53 s per leave-one-liquid-out evaluation at
    ``n_jobs=1`` against 2.84 s at ``n_jobs=-1``. Leave it at -1 for a single
    evaluation; pass 1 from inside a parallel loop.
    """
    return {
        BASELINE: DummyRegressor(strategy='mean'),
        # No random_state: it is inert for Ridge's default 'auto'/cholesky
        # solver, and carrying one implies a reproducibility guarantee that
        # this line is not what provides.
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=10000),
        'elasticnet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE,
                                 max_iter=10000),
        'random_forest': RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE,
                                               n_jobs=n_jobs),
        'xgboost': XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                                random_state=RANDOM_STATE, n_jobs=n_jobs, verbosity=0),
    }


def candidate_feature_sets() -> dict[str, list[str]]:
    """Feature sets to compare, deduplicated by membership.

    Two sets with identical members would otherwise appear as two independent
    results and imply agreement that is really just the same set counted twice.
    """
    from allmeta.compare import collect_feature_sets

    collected = {
        'all': list(cols.NUM_COLUMNS),
        'manual': list(cols.MANUAL_KEEP),
        # Fixed on mechanism rather than found by search. The honest comparison for
        # every searched feature set: if a four-descriptor set written down in
        # advance matches them, the search added nothing but the opportunity to
        # overfit thirteen liquids.
        'physics': list(cols.PHYSICS_KEEP),
    }
    collected.update(collect_feature_sets())

    seen: dict[tuple[str, ...], str] = {}
    unique: dict[str, list[str]] = {}
    aliases: dict[str, list[str]] = {}
    for name, members in collected.items():
        key = tuple(sorted(members))
        if key in seen:
            kept = seen[key]
            # Dedup keeps the FIRST name seen, and 'all'/'manual'/'physics' are
            # inserted above the C:: sets -- so a supervised set with identical
            # membership would be evaluated under an honest name and escape the
            # selection_leakage flag entirely. Not reachable today (C:: sets hold 1-4
            # members, the honest sets 4-58), but the failure would be silent and the
            # number it produced would look like the headline result.
            if selected_on_labels(name) and not selected_on_labels(kept):
                raise AssertionError(
                    f'{name!r} was selected on the labels but has the same members as '
                    f'{kept!r}, which was not. Dedup would score it as leakage-free. '
                    'Keep the leaky name, or evaluate both.')
            aliases.setdefault(kept, []).append(name)
            continue
        seen[key] = name
        unique[name] = list(members)

    # Two selectors that landed on exactly the same features are one result, not
    # two, so only one is fitted. The mapping is written out because a silent drop
    # reads as "never evaluated" -- MRMR-FCQ and MRMR-RFCQ both alias here.
    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
    DataFrame(
        [{'evaluated_as': kept, 'identical_to': other,
          'n_features': len(unique[kept]),
          'features': ', '.join(unique[kept])}
         for kept, others in sorted(aliases.items()) for other in sorted(others)]
    ).to_csv(ALIASES_PATH, index=False)

    return unique


def evaluate(feature_sets: dict[str, list[str]] | None = None
             ) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Fit every model on every feature set under leave-one-liquid-out CV.

    Returns pooled scores, per-fold scores, and the predictions themselves so a
    parity plot can be drawn without refitting.
    """
    # Factor columns are attached from loadings fitted on the wide unlabelled
    # subsets, never refitted here: thirteen liquids is far too few to define a
    # principal component, and the definition must not depend on which liquids
    # happen to carry a label.
    frame = attach_factors(load_subset('C'))
    target = frame[cols.TARGET]
    groups = frame['Name']
    splitter = LeaveOneGroupOut()

    sets = candidate_feature_sets() if feature_sets is None else feature_sets

    prediction_rows: list[dict] = []
    fold_rows: list[dict] = []

    for set_name, features in sets.items():
        usable = [f for f in features if f in frame.columns]
        if not usable:
            continue

        for include_params in (True, False):
            columns = usable + list(cols.PIPETTING_PARAMS) if include_params else usable
            data = frame[columns]

            for model_name, model in models().items():
                for fold, (train_idx, test_idx) in enumerate(
                        splitter.split(data, target, groups=groups)):
                    # clone() because Pipeline does not clone its steps: one estimator
                    # instance would otherwise be re-fitted across all thirteen folds.
                    # Correct today only because scikit-learn and XGBoost both refit
                    # from scratch -- one warm-start default away from silent
                    # cross-fold contamination, and nothing here would report it.
                    pipeline = Pipeline([('scale', RobustScaler()),
                                         ('model', clone(model))])

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        pipeline.fit(data.iloc[train_idx], target.iloc[train_idx])
                        predicted = pipeline.predict(data.iloc[test_idx])

                    actual = target.iloc[test_idx].to_numpy()
                    fold_rows.append({
                        'feature_set': set_name, 'n_features': len(columns),
                        'include_params': include_params, 'model': model_name,
                        'fold': fold, 'held_out_liquid': groups.iloc[test_idx].iloc[0],
                        'rmse': float(root_mean_squared_error(actual, predicted)),
                        'mae': float(mean_absolute_error(actual, predicted)),
                    })
                    for row, y_true, y_pred in zip(
                            frame.iloc[test_idx].itertuples(), actual, predicted):
                        prediction_rows.append({
                            'feature_set': set_name, 'include_params': include_params,
                            'model': model_name, 'fold': fold,
                            'liquid': row.Name, 'volume': row.volume,
                            'y_true': float(y_true), 'y_pred': float(y_pred),
                            'residual': float(y_true - y_pred),
                        })

    predictions = DataFrame(prediction_rows)
    fold_scores = DataFrame(fold_rows)
    scores = pooled_scores(predictions, leakage=selected_on_labels)

    return scores, fold_scores, predictions


def nested_selections(include_params: bool = False
                      ) -> tuple[dict, dict, list[str]]:
    """Per-fold feature selections from disk, as ``(chosen, held_out_of, methods)``.

    Both keyed by ``(method, fold)``. Shared by ``evaluate_nested`` and the nested
    permutation null so the two cannot disagree about which features belong to which
    fold.
    """
    from allmeta.supervised import FOLD_PATH

    if not FOLD_PATH.is_file():
        raise FileNotFoundError(
            f'{FOLD_PATH} not found; run allmeta.supervised first -- the nested '
            'evaluation reuses the per-fold selections rather than redoing them.')

    folds = pd.read_csv(FOLD_PATH)
    folds = folds[folds['include_params'].astype(bool) == include_params]
    if folds.empty:
        raise ValueError(f'no fold selections with include_params={include_params}')

    return (folds.groupby(['method', 'fold'])['feature'].apply(list).to_dict(),
            folds.groupby(['method', 'fold'])['held_out_liquid'].first().to_dict(),
            sorted(folds['method'].unique()))


def permutation_null_nested(method: str, model_name: str = 'random_forest',
                            n_permutations: int = 100, include_params: bool = False,
                            random_state: int = RANDOM_STATE,
                            n_jobs: int = PERMUTATION_JOBS) -> DataFrame:
    """Permutation null for one method under the nested protocol.

    Same liquid-block permutation as ``permutation_null`` -- literally the same
    ``permuted_targets`` call -- but each fold is fitted on the features that fold's
    selector chose, so the null is comparable with ``evaluate_nested`` rather than
    with the aggregate.

    **What this null does and does not hold fixed.** The per-fold feature sets are
    read from disk, and they were chosen against the *real* target. A strictly nested
    null would re-run all thirteen selectors inside all thirteen folds for every draw:
    at the measured ~29 s per fold that is roughly 1,400 CPU-hours for 1000 draws, and
    it is not affordable here. So this tests "given these features, does the
    liquid-to-descriptor mapping matter?" and not "could this selection procedure have
    produced this score by chance?". The second question is the stronger one and this
    does not answer it. Calling this a full nested null would be the same species of
    overclaim the nested evaluation exists to remove.
    """
    chosen, held_out_of, _ = nested_selections(include_params)
    frame = attach_factors(load_subset('C'))
    target = frame[cols.TARGET]
    groups = frame['Name']
    splitter = LeaveOneGroupOut()

    plan: list[tuple[np.ndarray, np.ndarray, DataFrame]] = []
    for fold, (train_idx, test_idx) in enumerate(
            splitter.split(frame, target, groups=groups)):
        key = (method, fold)
        if key not in chosen:
            continue
        held = groups.iloc[test_idx].iloc[0]
        if held_out_of[key] != held:
            raise ValueError(
                f'fold {fold} of {method}: selections file says {held_out_of[key]!r}, '
                f'splitter says {held!r}')
        features = [f for f in chosen[key] if f in frame.columns]
        if not features:
            continue
        columns = (features + list(cols.PIPETTING_PARAMS) if include_params
                   else features)
        plan.append((train_idx, test_idx, frame[columns]))

    if not plan:
        raise ValueError(f'no usable fold selections for {method!r}')

    targets = permuted_targets(target, groups, frame['volume'],
                               n_permutations=n_permutations,
                               random_state=random_state)

    def score(draw: int, permuted: pd.Series) -> dict:
        predicted = np.empty(len(frame))
        for train_idx, test_idx, data in plan:
            pipeline = Pipeline([('scale', RobustScaler()),
                                 ('model', models(n_jobs=1)[model_name])])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pipeline.fit(data.iloc[train_idx], permuted.iloc[train_idx])
                predicted[test_idx] = pipeline.predict(data.iloc[test_idx])

        liquid = DataFrame({'t': permuted, 'p': predicted,
                            'l': groups}).groupby('l').mean()

        return {
            'draw': draw,
            'permuted': draw > 0,
            'model': model_name,
            'rmse': float(root_mean_squared_error(permuted, predicted)),
            'r2': float(r2_score(permuted, predicted)),
            'between_liquid_r2': float(r2_score(liquid['t'], liquid['p'])),
        }

    rows = Parallel(n_jobs=n_jobs)(
        delayed(score)(draw, permuted) for draw, permuted in enumerate(targets))

    result = DataFrame(rows).sort_values('draw', ignore_index=True)
    observed = result.loc[result['draw'] == 0, 'rmse'].iloc[0]
    null = result.loc[result['permuted'], 'rmse']
    result.attrs['p_value'] = float((null <= observed).sum() + 1) / (len(null) + 1)

    return result


def permutation_pvalues_nested(n_permutations: int = N_PERMUTATIONS,
                               model_name: str = 'random_forest',
                               include_params: bool = False
                               ) -> tuple[DataFrame, DataFrame]:
    """``permutation_null_nested`` over every supervised method; (draws, summary)."""
    _, _, methods = nested_selections(include_params)

    draw_frames: list[DataFrame] = []
    summary: list[dict] = []
    for method in methods:
        draws = permutation_null_nested(method, model_name=model_name,
                                        n_permutations=n_permutations,
                                        include_params=include_params)
        draws.insert(0, 'feature_set', method)
        draw_frames.append(draws)

        observed = draws[~draws['permuted']].iloc[0]
        null = draws[draws['permuted']]
        summary.append({
            'feature_set': method,
            'model': model_name,
            'n_permutations': int(len(null)),
            'observed_rmse': float(observed['rmse']),
            'null_median_rmse': float(null['rmse'].median()),
            'p_value': draws.attrs['p_value'],
            'observed_between_liquid_r2': float(observed['between_liquid_r2']),
            'null_median_between_liquid_r2': float(null['between_liquid_r2'].median()),
        })

    if not summary:
        return DataFrame(), DataFrame()

    table = DataFrame(summary)
    table['q_value'] = benjamini_hochberg(table['p_value'].to_numpy())

    return (pd.concat(draw_frames, ignore_index=True),
            table.sort_values('p_value', ignore_index=True))


def evaluate_nested(include_params: bool = False
                    ) -> tuple[DataFrame, DataFrame]:
    """Score each supervised method on the features *its own fold* chose.

    This is the number the repository did not have. ``supervised.run_selection``
    already does the hard part correctly -- it re-runs every selector inside each
    leave-one-liquid-out fold and writes the result keyed by ``held_out_liquid``. But
    ``selection_frequency`` then aggregates all thirteen folds into one feature set
    per method, and ``evaluate`` scores *that* under a fresh LOGO. So every held-out
    liquid helped choose the features it is then used to test. ``pooled_scores``
    flags the result ``selection_leakage=True``, correctly, and nothing ever
    corrected it.

    Here the selection stays inside the fold: liquid L is predicted by a model fitted
    on the other twelve, using only the features chosen when L was held out. Nothing
    about L reaches the features or the fit. Same ``LeaveOneGroupOut`` on ``Name`` and
    the same model grid as ``evaluate``, so the two are directly comparable.

    Expect these scores to be **worse** than the aggregate ones. That is the point:
    the optimism was always there, and removing it is not the same as breaking the
    model.

    Returns ``(scores, predictions)``.
    """
    chosen, held_out_of, methods = nested_selections(include_params)

    frame = attach_factors(load_subset('C'))
    target = frame[cols.TARGET]
    groups = frame['Name']
    splitter = LeaveOneGroupOut()

    prediction_rows: list[dict] = []
    cardinality: list[dict] = []

    for method in methods:
        sizes: list[int] = []
        for fold, (train_idx, test_idx) in enumerate(
                splitter.split(frame, target, groups=groups)):
            key = (method, fold)
            if key not in chosen:
                continue

            held = groups.iloc[test_idx].iloc[0]
            # The whole correctness of this function rests on the fold number in
            # fold_selections.csv meaning the same fold the splitter just produced.
            # Both come from LeaveOneGroupOut on the same frame, so they do -- but a
            # reordered or regenerated selections file would silently pair liquid L's
            # test rows with some other liquid's feature set, and the tell would be
            # nested scores coming out BETTER than the leaky ones. Assert rather than
            # trust.
            if held_out_of[key] != held:
                raise ValueError(
                    f'fold {fold} of {method}: selections file says the held-out '
                    f'liquid was {held_out_of[key]!r}, splitter says {held!r}. '
                    'The selections were written against a different fold ordering.')

            features = [f for f in chosen[key] if f in frame.columns]
            if not features:
                continue
            sizes.append(len(features))

            columns = (features + list(cols.PIPETTING_PARAMS) if include_params
                       else features)
            data = frame[columns]

            for model_name, model in models().items():
                pipeline = Pipeline([('scale', RobustScaler()),
                                     ('model', clone(model))])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    pipeline.fit(data.iloc[train_idx], target.iloc[train_idx])
                    predicted = pipeline.predict(data.iloc[test_idx])

                for row, y_true, y_pred in zip(
                        frame.iloc[test_idx].itertuples(),
                        target.iloc[test_idx].to_numpy(), predicted):
                    prediction_rows.append({
                        'feature_set': method, 'include_params': include_params,
                        'model': model_name, 'fold': fold,
                        'liquid': row.Name, 'volume': row.volume,
                        'y_true': float(y_true), 'y_pred': float(y_pred),
                        'residual': float(y_true - y_pred),
                    })

        if sizes:
            cardinality.append({
                'feature_set': method,
                'n_features_median': float(np.median(sizes)),
                'n_features_min': int(min(sizes)),
                'n_features_max': int(max(sizes)),
            })

    predictions = DataFrame(prediction_rows)
    if predictions.empty:
        return DataFrame(), predictions

    # False by construction, not by name lookup: the selection never saw the liquid
    # being predicted, whatever the feature set is called. Passed in rather than
    # computed and overwritten, so the claim is made once and in one place.
    scores = pooled_scores(predictions, leakage=False)
    # Cardinality varies by fold for wrapper_rfecv, which picks its own size, so a
    # single n_features would be a fiction. Median with the range beside it.
    scores = scores.merge(DataFrame(cardinality), on='feature_set', how='left')

    return scores, predictions


def random_feature_baseline(n_draws: int = N_RANDOM_SETS, n_features: int = 5,
                            random_state: int = RANDOM_STATE,
                            n_jobs: int = PERMUTATION_JOBS) -> DataFrame:
    """Score ``n_draws`` feature sets drawn uniformly at random from the pool.

    The question every selection method in this project implicitly claims to answer
    is "did selecting help?", and the all-features baseline answers a different one --
    it says whether 58 descriptors beat 5, which is mostly a statement about
    dimensionality. Five descriptors picked at random from the same pool the selectors
    drew from is the direct comparison: if a searched set does not clear this
    distribution, the search bought nothing that a coin could not.

    The draws come from ``columns.supervised_pool()`` -- the pool the selectors
    actually see -- and are generated up front from one seeded generator so the
    parallel loop below cannot change which sets get drawn.
    """
    frame = attach_derived(load_subset('C'))
    target = frame[cols.TARGET]
    groups = frame['Name']

    pool = [f for f in cols.supervised_pool() if f in frame.columns]
    if len(pool) < n_features:
        raise ValueError(f'pool of {len(pool)} is smaller than the requested '
                         f'{n_features} features')

    rng = np.random.default_rng(random_state)
    draws = [sorted(rng.choice(pool, size=n_features, replace=False).tolist())
             for _ in range(n_draws)]

    def score(draw: int, features: list[str]) -> list[dict]:
        data = frame[features]
        rows: list[dict] = []
        for model_name in models():
            predicted = _logo_predictions(data, target, groups, model_name, n_jobs=1)
            liquid = DataFrame({'t': target, 'p': predicted,
                                'l': groups}).groupby('l').mean()
            rows.append({
                'draw': draw,
                'model': model_name,
                'features': ', '.join(features),
                'n_features': len(features),
                'rmse': float(root_mean_squared_error(target, predicted)),
                'mae': float(mean_absolute_error(target, predicted)),
                'r2': float(r2_score(target, predicted)),
                'between_liquid_r2': float(r2_score(liquid['t'], liquid['p'])),
            })

        return rows

    batches = Parallel(n_jobs=n_jobs)(
        delayed(score)(draw, features) for draw, features in enumerate(draws))

    return DataFrame([row for batch in batches for row in batch])


def permutation_null(features: list[str], model_name: str = 'random_forest',
                     n_permutations: int = 100, include_params: bool = False,
                     random_state: int = RANDOM_STATE,
                     n_jobs: int = PERMUTATION_JOBS) -> DataFrame:
    """Re-run the whole LOGO evaluation with the liquid-to-label mapping shuffled.

    The labels are permuted **at the liquid level**: each liquid keeps its own eight
    measured errors, but that whole block is reassigned to a different liquid's
    descriptors. Shuffling rows instead would break the within-liquid structure and
    hand the null an easier problem than the real model faces, which is the usual way
    a permutation test is made to pass.

    Under this null the descriptors carry no information about which liquid they
    describe, so any configuration scoring materially better than the permuted
    distribution is finding real structure -- and one that does not is reporting
    noise. It is the cheapest decisive check available at this sample size.

    The draws run in parallel because the serial cost is prohibitive: at 1000 draws
    over the ~29 candidate feature sets this is ~18 hours on one core. Every
    permutation mapping is still drawn *sequentially* from one seeded generator
    before any fitting starts, so the sequence of shuffles -- and therefore every
    number this returns -- is identical to what the serial loop produced. The
    threading backend is used rather than processes so the 104-row frame is shared
    rather than pickled 1000 times; the estimators release the GIL where it matters
    and are asked for ``n_jobs=1`` so they do not contend with this loop.
    """
    frame = attach_factors(load_subset('C'))
    target = frame[cols.TARGET]
    groups = frame['Name']
    usable = [f for f in features if f in frame.columns]
    if not usable:
        missing = ', '.join(sorted(features)[:5])
        raise ValueError(
            f'none of the requested features are columns of subset C (e.g. {missing}). '
            'Factor columns need the screens to have been run first.')
    if include_params:
        usable = usable + list(cols.PIPETTING_PARAMS)
    data = frame[usable]

    targets = permuted_targets(target, groups, frame['volume'],
                               n_permutations=n_permutations,
                               random_state=random_state)

    def score(draw: int, permuted: pd.Series) -> dict:
        predicted = _logo_predictions(data, permuted, groups, model_name, n_jobs=1)
        liquid = DataFrame({'t': permuted, 'p': predicted, 'l': groups}).groupby('l').mean()

        return {
            'draw': draw,
            'permuted': draw > 0,
            'model': model_name,
            'rmse': float(root_mean_squared_error(permuted, predicted)),
            'r2': float(r2_score(permuted, predicted)),
            'between_liquid_r2': float(r2_score(liquid['t'], liquid['p'])),
        }

    rows = Parallel(n_jobs=n_jobs)(
        delayed(score)(draw, permuted) for draw, permuted in enumerate(targets))

    result = DataFrame(rows).sort_values('draw', ignore_index=True)
    observed = result.loc[result['draw'] == 0, 'rmse'].iloc[0]
    null = result.loc[result['permuted'], 'rmse']
    # +1 in both terms so an empirical p can never be reported as exactly zero,
    # which it is not -- it is bounded below by 1/(n_permutations + 1).
    result.attrs['p_value'] = float((null <= observed).sum() + 1) / (len(null) + 1)

    return result


def permutation_pvalues(feature_sets: dict[str, list[str]] | None = None,
                        model_name: str = 'random_forest',
                        n_permutations: int = N_PERMUTATIONS,
                        include_params: bool = False
                        ) -> tuple[DataFrame, DataFrame]:
    """Run ``permutation_null`` on every feature set; return (draws, summary).

    ``include_params`` defaults to False deliberately. With the pipetting parameters
    in, a model can address the within-liquid spread that the descriptors structurally
    cannot reach, and most of the apparent skill is the relative-error-falls-with-
    volume identity -- so a permutation test on the liquid labels would be testing
    something other than the chemistry.
    """
    sets = candidate_feature_sets() if feature_sets is None else feature_sets

    draw_frames: list[DataFrame] = []
    summary: list[dict] = []
    for name, features in sets.items():
        try:
            draws = permutation_null(features, model_name=model_name,
                                     n_permutations=n_permutations,
                                     include_params=include_params)
        except ValueError as error:
            # A factor set whose loadings are missing, for instance. Skipped rather
            # than crashing the run, but named so the gap is visible.
            print(f'  skipped {name}: {error}')
            continue

        draws.insert(0, 'feature_set', name)
        draw_frames.append(draws)

        observed = draws[~draws['permuted']].iloc[0]
        null = draws[draws['permuted']]
        summary.append({
            'feature_set': name,
            'model': model_name,
            'n_features': len(features),
            'n_permutations': int(len(null)),
            'observed_rmse': float(observed['rmse']),
            'null_median_rmse': float(null['rmse'].median()),
            'p_value': draws.attrs['p_value'],
            'observed_between_liquid_r2': float(observed['between_liquid_r2']),
            'null_median_between_liquid_r2': float(null['between_liquid_r2'].median()),
        })

    if not summary:
        return DataFrame(), DataFrame()

    table = DataFrame(summary)
    table['q_value'] = benjamini_hochberg(table['p_value'].to_numpy())

    return (pd.concat(draw_frames, ignore_index=True),
            table.sort_values('p_value', ignore_index=True))


def bootstrap_all(predictions: DataFrame, include_params: bool = False,
                  n_boot: int = N_BOOTSTRAP, label: str = 'aggregate') -> DataFrame:
    """Bootstrap interval for every (feature_set, model) in a prediction table.

    ``label`` names the evaluation these predictions came from, so the aggregate and
    the nested (leakage-free) intervals can live in one file without a reader having
    to remember which is which.
    """
    view = predictions[predictions['include_params'].astype(bool) == include_params]

    rows: list[dict] = []
    for feature_set, model in (view[['feature_set', 'model']]
                               .drop_duplicates().itertuples(index=False)):
        interval = bootstrap_between_r2(view, feature_set, model,
                                        include_params=include_params, n_boot=n_boot)
        rmse = bootstrap_pooled_rmse(view, feature_set, model,
                                     include_params=include_params, n_boot=n_boot)
        rows.append({'evaluation': label, **interval, **rmse})

    if not rows:
        return DataFrame()

    return DataFrame(rows).sort_values(['evaluation', 'feature_set', 'model'],
                                       ignore_index=True)


def run(n_permutations: int = N_PERMUTATIONS,
        n_boot: int = N_BOOTSTRAP) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Evaluate, run both validators, write every table, and return the main three.

    The permutation null and the bootstrap interval are run here rather than left to
    a caller because they were written, correct, and never once called -- and they are
    the only two instruments in this project with real statistical power at thirteen
    liquids. A scores table without them invites exactly the reading it should not:
    that the minimum RMSE over a grid of 150 configurations means something.

    ``n_permutations`` is exposed so a smoke test can ask for a cheap run; the default
    is the one the written artefacts should carry.
    """
    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)

    scores, fold_scores, predictions = evaluate()
    scores.to_csv(SCORES_PATH, index=False)
    fold_scores.to_csv(FOLD_SCORES_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    DataFrame([target_variance_decomposition()]).to_csv(DECOMPOSITION_PATH, index=False)

    intervals = [bootstrap_all(predictions, n_boot=n_boot)]

    # The leakage-free supervised number. Reported as *the* supervised result; the
    # aggregate stays available but is renamed at the comparison stage so nobody can
    # read it as honest.
    try:
        nested_scores, nested_predictions = evaluate_nested()
    except FileNotFoundError as error:
        print(f'  no nested evaluation: {error}')
        nested_scores = DataFrame()
    if not nested_scores.empty:
        nested_scores.to_csv(SCORES_NESTED_PATH, index=False)
        nested_predictions.to_csv(PREDICTIONS_NESTED_PATH, index=False)
        intervals.append(bootstrap_all(nested_predictions, n_boot=n_boot,
                                       label='nested'))

    pd.concat(intervals, ignore_index=True).to_csv(BOOTSTRAP_CI_PATH, index=False)

    random_feature_baseline().to_csv(RANDOM_BASELINE_PATH, index=False)

    draws, pvalues = permutation_pvalues(n_permutations=n_permutations)
    if not draws.empty:
        draws['evaluation'] = 'aggregate'
        pvalues['evaluation'] = 'aggregate'

    if not nested_scores.empty:
        nested_draws, nested_p = permutation_pvalues_nested(
            n_permutations=n_permutations)
        if not nested_draws.empty:
            nested_draws['evaluation'] = 'nested'
            nested_p['evaluation'] = 'nested'
            draws = pd.concat([draws, nested_draws], ignore_index=True)
            pvalues = pd.concat([pvalues, nested_p], ignore_index=True)

    if not draws.empty:
        draws.to_csv(PERMUTATION_NULL_PATH, index=False)
        pvalues.to_csv(PERMUTATION_PVALUES_PATH, index=False)

    return scores, fold_scores, predictions


def _best_config(scores: DataFrame, leakage: bool, include_params: bool = False
                 ) -> pd.Series | None:
    """Lowest-RMSE configuration within one leakage class, or None if empty.

    Still a minimum, because a parity plot has to be drawn from one configuration and
    a median is not one. But the minimum over ~75 configurations is a biased estimate
    of anything -- the deepest of many noisy traces -- so ``n_candidates`` and the
    class median travel back with it and the figures that use it say so in the title.
    """
    view = scores[(scores['selection_leakage'] == leakage)
                  & (scores['include_params'] == include_params)
                  & (scores['model'] != BASELINE)]
    if view.empty:
        return None

    best = view.nsmallest(1, 'rmse').iloc[0].copy()
    best['n_candidates'] = int(len(view))
    best['median_rmse_of_class'] = float(view['rmse'].median())
    best['median_r2_of_class'] = float(view['r2'].median())

    return best


def plot_parity(predictions: DataFrame, scores: DataFrame, path: Path) -> Path:
    """Predicted against measured error for the best honest and best leaky configs.

    Side by side because the difference between them is the point: the leaky panel
    looks better only because the held-out liquid helped choose the features.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    panels = [(False, 'no selection leakage'), (True, 'supervised set (leaky)')]

    for ax, (leakage, label) in zip(axes, panels):
        best = _best_config(scores, leakage)
        if best is None:
            annotate_empty(ax, f'no {label} configuration')
            continue

        view = predictions[
            (predictions['feature_set'] == best['feature_set'])
            & (predictions['model'] == best['model'])
            & (~predictions['include_params'].astype(bool))
        ]
        sns.scatterplot(data=view, x='y_true', y='y_pred', hue='liquid',
                        style='liquid', s=70, ax=ax, palette='tab10')

        lims = [0, max(view['y_true'].max(), view['y_pred'].max()) * 1.08]
        ax.plot(lims, lims, 'k--', linewidth=1, label='perfect')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f'measured {cols.TARGET_LABEL}')
        ax.set_ylabel(f'predicted {cols.TARGET_LABEL}')
        ax.set_title(
            f"{label}\n{best['feature_set']} / {best['model']} "
            f"(R2={best['r2']:.3f}, rho={best['spearman']:.2f})\n"
            f"best of {int(best['n_candidates'])} configurations; "
            f"class median R2 = {best['median_r2_of_class']:.3f}", fontsize=10)
        ax.legend(fontsize=7, loc='upper left')

    fig.suptitle(f'Leave-one-liquid-out predictions, pooled across the '
                 f'{labelled_liquid_count()} folds')
    fig.tight_layout()

    return save(fig, path)


def plot_residuals(predictions: DataFrame, scores: DataFrame, path: Path) -> Path:
    """Residual spread per held-out liquid and per pipetting volume.

    Shows whether the error is one difficult liquid or a systematic miss across the
    volume range, which RMSE alone cannot distinguish.
    """
    best = _best_config(scores, leakage=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    if best is None:
        for ax in axes:
            annotate_empty(ax, 'no unleaked configuration available')
        return save(fig, path)

    view = predictions[
        (predictions['feature_set'] == best['feature_set'])
        & (predictions['model'] == best['model'])
        & (~predictions['include_params'].astype(bool))
    ]

    order = view.groupby('liquid')['residual'].median().sort_values().index
    sns.boxplot(data=view, x='liquid', y='residual', order=order, ax=axes[0],
                color='#4c72b0')
    sns.stripplot(data=view, x='liquid', y='residual', order=order, ax=axes[0],
                  color='0.2', size=4)
    axes[0].axhline(0, color='k', linewidth=1)
    axes[0].set_ylabel('residual (fraction of target volume)')
    axes[0].set_xlabel('held-out liquid')
    axes[0].set_title('residual by held-out liquid')
    axes[0].tick_params(axis='x', labelrotation=30)
    for label in axes[0].get_xticklabels():
        label.set_ha('right')

    sns.lineplot(data=view, x='volume', y='residual', hue='liquid', marker='o',
                 ax=axes[1], palette='tab10')
    axes[1].axhline(0, color='k', linewidth=1)
    axes[1].set_xscale('log')
    axes[1].set_xlabel(cols.axis_label('volume', 'log scale'))
    axes[1].set_ylabel('residual (fraction of target volume)')
    axes[1].set_title('residual by volume')
    axes[1].legend(fontsize=7, ncol=2)

    fig.suptitle(f"Residuals: {best['feature_set']} / {best['model']}")
    fig.tight_layout()

    return save(fig, path)


def plot_scores(scores: DataFrame, path: Path,
                intervals: DataFrame | None = None,
                random_baseline: DataFrame | None = None) -> Path:
    """Pooled RMSE per feature set, against the mean baseline, with bootstrap CIs.

    The baseline line is the reading aid: bars to its right are configurations that
    do worse than predicting the training mean.

    The error bars are the point of the figure, not decoration. Each bar is the
    *minimum* over the five-model grid, and the intervals -- resampled over the
    thirteen liquids, which is the real sample size -- are wide enough to overlap
    almost every other bar. A reader taking the ordering of bare bars seriously would
    be reading a ranking that the data does not support.
    """
    view = scores[(~scores['include_params'].astype(bool))
                  & (scores['model'] != BASELINE)].copy()
    baseline = scores.loc[scores['model'] == BASELINE, 'rmse'].median()

    best_per_set = (view.sort_values('rmse')
                    .groupby('feature_set', as_index=False).first()
                    .sort_values('rmse'))
    best_per_set['class'] = np.where(best_per_set['selection_leakage'],
                                     'supervised (leaky)', 'no leakage')

    n_configs = int(view.groupby('feature_set').size().median())

    fig, ax = plt.subplots(figsize=(10, max(6, len(best_per_set) * 0.42)))

    # Five descriptors drawn at random from the same pool the selectors used, scored
    # the same way -- including the same minimum-over-the-model-grid, so the band
    # carries the same winner's curse the bars do and the comparison is like for
    # like. A bar inside this band is a feature set that selection did not improve on.
    if random_baseline is not None and not random_baseline.empty:
        per_draw = (random_baseline[random_baseline['model'] != BASELINE]
                    .groupby('draw')['rmse'].min())
        lo, hi = np.percentile(per_draw, [5, 95])
        ax.axvspan(lo, hi, color='0.75', alpha=0.35, zorder=0,
                   label=f'random 5 descriptors, 5-95% ({len(per_draw)} draws)')
        ax.axvline(float(per_draw.median()), color='0.45', linewidth=1.2,
                   linestyle='-.', zorder=1,
                   label=f'random 5 descriptors, median ({per_draw.median():.4f})')

    sns.barplot(data=best_per_set, y='feature_set', x='rmse', hue='class',
                dodge=False, ax=ax, palette={'no leakage': '#4c72b0',
                                             'supervised (leaky)': '#c44e52'})

    if intervals is not None and not intervals.empty:
        lookup = of_evaluation(intervals).set_index(['feature_set', 'model'])
        for i, row in enumerate(best_per_set.itertuples()):
            key = (row.feature_set, row.model)
            if key not in lookup.index:
                continue
            interval = lookup.loc[key]
            ax.plot([interval['rmse_ci_lo'], interval['rmse_ci_hi']], [i, i],
                    color='0.15', linewidth=1.2, solid_capstyle='butt', zorder=5)
            ax.plot([interval['rmse_ci_lo'], interval['rmse_ci_hi']], [i, i],
                    '|', color='0.15', markersize=6, zorder=5)

    ax.axvline(baseline, color='k', linestyle='--', linewidth=1.5,
               label=f'mean baseline ({baseline:.4f})')
    ax.set_xlabel(f'pooled leave-one-liquid-out RMSE (fraction of target volume, '
                  f'lower is better)\nbar = best of {n_configs} models; '
                  f'whisker = 95% bootstrap CI over the '
                  f'{labelled_liquid_count()} liquids')
    ax.set_ylabel('')
    ax.set_title(f'Best model per feature set, descriptors only '
                 f'(minimum over {n_configs} models each)')

    # Model names go in their own column past the widest whisker, not just past the
    # bar: at the bar they land on top of the error bars and neither is readable.
    right = max(ax.get_xlim()[1], best_per_set['rmse'].max())
    ax.set_xlim(0, right * 1.30)
    label_x = right * 1.02
    for i, row in enumerate(best_per_set.itertuples()):
        ax.text(label_x, i, row.model, va='center', fontsize=7, color='0.3')

    # Legend outside the bars entirely -- at 'lower right' it sat on top of the
    # worst-scoring feature sets, which are the ones a reader is checking.
    ax.legend(loc='lower right', fontsize=8, bbox_to_anchor=(1.0, -0.02),
              framealpha=0.95)

    fig.tight_layout()

    return save(fig, path)


def plot_r2_distribution(scores: DataFrame, path: Path) -> Path:
    """Out-of-fold R-squared by leakage class.

    Nearly every honest configuration sits at or below zero -- worse than predicting
    the mean. That is the headline this figure exists to make unmissable.
    """
    view = scores[(~scores['include_params'].astype(bool))
                  & (scores['model'] != BASELINE)].copy()
    leaky_label = f'supervised set\n(selection saw all {labelled_liquid_count()} liquids)'
    view['class'] = np.where(view['selection_leakage'], leaky_label,
                             'unsupervised set\n(no leakage)')

    baseline = baseline_r2(scores)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.boxplot(data=view, x='class', y='r2', hue='class', legend=False, ax=ax,
                width=0.5,
                palette={'unsupervised set\n(no leakage)': '#4c72b0',
                         leaky_label: '#c44e52'})
    sns.stripplot(data=view, x='class', y='r2', ax=ax, color='0.2', size=4, alpha=0.6)
    ax.axhline(baseline, color='k', linewidth=1.5,
               label=f'DummyRegressor under LOGO (R2 = {baseline:.4f})')
    ax.axhline(0, color='0.6', linewidth=1, linestyle=':',
               label='perfect-mean reference (R2 = 0, not achievable under LOGO)')
    ax.text(0.02, 0.02,
            f'the mean baseline is R2 = {baseline:.4f}, not 0: holding out one of '
            f'{labelled_liquid_count()} liquids\ninflates the mean predictor\'s '
            f'residual by n/(n-1), so R2 = 1 - (n/(n-1))^2',
            transform=ax.transAxes, fontsize=8, color='0.3')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylabel('pooled out-of-fold R2')
    ax.set_xlabel('')
    ax.set_title('Feature sets chosen with the target look better than they are')

    fig.tight_layout()

    return save(fig, path)


def plot_target(path: Path) -> Path:
    """The target itself: measured error per liquid across the eight volumes.

    Two protocol regimes are shaded because volume is confounded with rate and
    airgap; a trend across the whole axis is not a pure volume effect.
    """
    frame = load_subset('C')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.lineplot(data=frame, x='volume', y=cols.TARGET, hue='Name', marker='o',
                 ax=axes[0], palette='tab10')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].axvspan(35, 900, color='0.85', alpha=0.5, zorder=0)
    axes[0].text(120, frame[cols.TARGET].min() * 1.4,
                 'rate 250/300 µL/s, 50 µL airgaps', fontsize=8, color='0.35')
    axes[0].text(12, frame[cols.TARGET].min() * 1.4,
                 'rate 20/20 µL/s,\n2 µL airgaps', fontsize=8, color='0.35')
    axes[0].set_xlabel(cols.axis_label('volume', 'log scale'))
    axes[0].set_ylabel(f'{cols.TARGET_LABEL}, log scale')
    axes[0].set_title('Measured dispensing error by liquid and volume')
    axes[0].legend(fontsize=7, ncol=2)

    pivot = frame.pivot_table(index='Name', columns='volume', values=cols.TARGET)
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='rocket_r', ax=axes[1],
                cbar_kws={'label': cols.TARGET_LABEL})
    axes[1].set_title('Same data as a table')
    axes[1].set_xlabel(cols.axis_label('volume'))
    axes[1].set_ylabel('')

    fig.tight_layout()

    return save(fig, path)


def plot_all(scores: DataFrame | None = None,
             predictions: DataFrame | None = None) -> list[Path]:
    """Write every regression figure."""
    apply_style()
    FIG_REGRESSION.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(SCORES_PATH) if scores is None else scores
    predictions = pd.read_csv(PREDICTIONS_PATH) if predictions is None else predictions
    intervals = (pd.read_csv(BOOTSTRAP_CI_PATH) if BOOTSTRAP_CI_PATH.is_file()
                 else DataFrame())
    randoms = (pd.read_csv(RANDOM_BASELINE_PATH) if RANDOM_BASELINE_PATH.is_file()
               else DataFrame())

    return [
        plot_target(FIG_REGRESSION / 'target_by_liquid_and_volume.png'),
        plot_parity(predictions, scores, FIG_REGRESSION / 'parity.png'),
        plot_residuals(predictions, scores, FIG_REGRESSION / 'residuals.png'),
        plot_scores(scores, FIG_REGRESSION / 'rmse_by_feature_set.png',
                    intervals=intervals, random_baseline=randoms),
        plot_r2_distribution(scores, FIG_REGRESSION / 'r2_by_leakage_class.png'),
    ]


if __name__ == '__main__':
    apply_style()
    score_table, folds, preds = run()
    clean = score_table[~score_table['include_params'] & ~score_table['selection_leakage']]
    leaky = score_table[~score_table['include_params'] & score_table['selection_leakage']]

    cols_out = ['feature_set', 'model', 'rmse', 'mae', 'r2', 'spearman', 'beats_baseline']
    print('--- descriptors only, NO selection leakage (the fair comparison)')
    print(clean.nsmallest(10, 'rmse')[cols_out].to_string(index=False))
    print()
    print(f'--- descriptors only, supervised sets (selection saw all '
          f'{labelled_liquid_count()} liquids: biased)')
    print(leaky.nsmallest(5, 'rmse')[cols_out].to_string(index=False))
    print()
    print(f'{int(score_table["beats_baseline"].sum())} of {len(score_table)} '
          f'configurations beat the mean baseline')

    # Median, not maximum. The maximum over ~75 noisy configurations is biased low on
    # RMSE and high on R2 by construction -- it is the winner of a search, not an
    # estimate of anything -- so the median leads and the maximum follows, labelled.
    base = baseline_r2(score_table)
    print(f'mean baseline r2 under LOGO: {base:.4f}  (NOT zero: 1-(n/(n-1))^2)')
    print(f'honest r2  median {clean["r2"].median():+.4f}   '
          f'max {clean["r2"].max():+.4f} (best of {len(clean)} configurations)')
    print(f'leaky  r2  median {leaky["r2"].median():+.4f}   '
          f'max {leaky["r2"].max():+.4f} (best of {len(leaky)} configurations)')

    if SCORES_NESTED_PATH.is_file():
        nested = pd.read_csv(SCORES_NESTED_PATH)
        nested = nested[nested['model'] != BASELINE]
        print(f'nested r2  median {nested["r2"].median():+.4f}   '
              f'max {nested["r2"].max():+.4f} (best of {len(nested)} configurations)'
              '   <- the leakage-free supervised number')
    print()
    # plot_all() used to be defined below this block, so running the module wrote
    # the CSVs but never refreshed a single figure.
    for written in plot_all(score_table, preds):
        print('  wrote', written)
