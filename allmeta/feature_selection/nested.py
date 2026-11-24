"""Nested cross-validation: the only honest performance number (plan, Phases 14-15).

Everything before this phase is a diagnostic. The Phase 8 clustering search, the Phase 9
method comparison and the Phase 10 Elastic Net grid all chose among candidates using the
labels of all thirteen liquids, so each of their winners is a minimum over hundreds of
noisy estimates and none of their errors is an estimate of anything. This module spends
the labels once, on evaluation.

## What is fold-dependent and what is not

    fitted on subset B, no target, NOT fold-dependent
        feature clustering, cluster PC1 loadings, global PCA loadings,
        manual cluster PC1 loadings

    uses the labels, MUST be refitted inside each outer training fold
        mRMR selection, clustering-candidate selection, clustering-method comparison,
        Elastic Net alpha and l1_ratio, the Phase 13 reduction

The first list is fold-independent for a specific and checkable reason:
``config.EXCLUDE_LABELLED_FROM_CLUSTER_FIT`` drops **every labelled liquid** from the
fitting frame, not just the held-out one. That is stricter than leakage-safety requires
-- a per-fold fit could keep the other twelve -- and the strictness is what buys the
saving: one clustering sweep serves every outer fold, and there is no fold-ordering
bookkeeping to get wrong.

In numbers, the frame goes from 732 curated liquids to 720. That is **12** removed, not
13, because one of the thirteen labelled liquids was never in subset B -- the chemical
curation had already moved it to subset R. The invariant that matters is not the count
but the emptiness of the intersection, so
``test_the_fitting_frame_contains_no_labelled_liquid`` asserts that directly rather than
asserting the config flag: a flag is evidence about intent, and the intersection is
evidence about the data.

## The fold-ordering trap

``regression.evaluate_nested`` was written to fix exactly this defect in the legacy
pipeline, and it carries an assertion that the fold number in its selections file names
the same held-out liquid the splitter just produced. The reason is worth restating: if
the two disagree, liquid L's test rows get paired with some other liquid's feature set,
and **the tell is that the nested scores come out better than the leaky ones**. That is
a silent, plausible-looking failure. This module builds its selections and its
predictions in the same loop so there is no file to fall out of step, and
``_assert_fold_identity`` checks the invariant anyway.

## What this can and cannot show

The ceiling is unchanged: eight volumes of a liquid share one descriptor vector, 56.5% of
the target variance is within-liquid and structurally unreachable, and under
leave-one-out the mean predictor scores 1 - (n/(n-1))^2 = -0.1736 between liquids at
n = 13. Every number here is read against that measured baseline, never against zero.

Thirteen outer folds means thirteen predictions of thirteen liquid means. That is a small
enough evaluation that the interval matters more than the point estimate, which is why
``regression.bootstrap_between_r2`` (resampling liquids, not rows) and the liquid-level
permutation null are reported beside every score rather than as an appendix.
"""

import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from allmeta import columns as cols
from allmeta.feature_selection import (comparison, config, elastic_net, factors,
                                       mrmr, optimization, pca, refinement)
from allmeta.feature_selection.algorithms import format_params


def _assert_fold_identity(fold: int, expected: str, actual: str) -> None:
    """The invariant whose violation looks like a good result.

    Pairing liquid L's test rows with another liquid's chosen features makes the nested
    score come out *better*, not worse, so this cannot be left to be noticed.
    """
    if expected != actual:
        raise ValueError(
            f'fold {fold}: the selection was made holding out {expected!r} but the '
            f'splitter is testing on {actual!r}. The folds are out of step and every '
            'nested score is meaningless -- and would look better, not worse.')


def _predict(values: DataFrame, target: Series, train: np.ndarray, test: np.ndarray,
             model) -> np.ndarray:
    """Fit on the outer training rows and predict the outer test rows.

    The scaler is fitted here, inside the fold, on training rows only -- one of the
    transformations the plan's Rule 2 names explicitly.
    """
    pipeline = Pipeline([('scale', RobustScaler()), ('model', clone(model))])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pipeline.fit(values.iloc[train], target.iloc[train])

        return pipeline.predict(values.iloc[test])


def build_fold_families(fit_frame: DataFrame, labelled: DataFrame,
                        train_mask: np.ndarray, artefacts: dict,
                        solutions=None, cluster_metrics: DataFrame | None = None,
                        labelled_factors=None) -> tuple[dict, dict]:
    """Every feature family as this outer fold sees it.

    Returns ``(families, choices)``. ``families`` maps a name to the full labelled
    matrix -- all 104 rows -- because the outer test rows still have to be *transformed*
    even though nothing was fitted on them. ``choices`` records what the fold decided,
    for ``selected_hyperparameters_by_outer_fold.csv``.

    Two families are chosen inside the fold and are therefore genuinely fold-dependent:

    ``mrmr_raw``          refitted on the training liquids only. This is the family the
                          Phase 10-12 pass had to omit, because the only mRMR selection
                          available there was fitted on all thirteen labels and putting
                          it in a table scored on those same labels is the defect
                          ``regression.evaluate_nested`` exists to correct.
    ``auto_cluster_pc1``  the clustering candidate and method are re-selected from the
                          training liquids alone, so the held-out liquid influenced
                          neither the clusters (fitted on subset B) nor the choice
                          among them.

    The other three are applications of persisted loadings and are identical in every
    fold; they are rebuilt here anyway so the dict has one shape in all folds.
    """
    families: dict[str, DataFrame] = {}
    choices: dict[str, dict] = {}

    manual_features = [c for c in config.MANUAL_FEATURES if c in labelled.columns]
    if manual_features:
        families['manual_raw'] = labelled[manual_features].reset_index(drop=True)
        choices['manual_raw'] = {'n_inputs': len(manual_features), 'chosen': 'fixed'}

    if 'pca_loadings' in artefacts:
        families['global_pca'] = pca.apply_loadings(labelled, artefacts['pca_loadings'])
        choices['global_pca'] = {'n_inputs': families['global_pca'].shape[1],
                                 'chosen': 'fixed'}
    if 'cluster_loadings' in artefacts:
        families['manual_cluster_pc1'] = pca.apply_loadings(
            labelled, artefacts['cluster_loadings'])
        choices['manual_cluster_pc1'] = {
            'n_inputs': families['manual_cluster_pc1'].shape[1], 'chosen': 'fixed'}

    # --- mRMR, fitted on this fold's training liquids only -------------------------
    pool = [c for c in cols.screening_features() if c in labelled.columns]
    training = labelled.loc[train_mask]
    try:
        selected, _, _ = mrmr.select_mrmr(training[pool], training[cols.TARGET],
                                          n_features=config.MRMR_N_FEATURES,
                                          method=config.MRMR_METHOD)
    except (ValueError, RuntimeError) as error:
        selected = []
        choices['mrmr_raw'] = {'n_inputs': 0, 'chosen': f'failed: {str(error)[:60]}'}
    if selected:
        families['mrmr_raw'] = labelled[selected].reset_index(drop=True)
        choices['mrmr_raw'] = {'n_inputs': len(selected),
                               'chosen': ', '.join(selected)}

    # --- automatic clustering, re-selected inside the fold -------------------------
    if solutions is not None and labelled_factors is not None:
        search = optimization.search(labelled_factors, subset=Series(train_mask))
        winners, _ = comparison.select_per_method(search, context=cluster_metrics)
        gate = comparison.admission(winners, cluster_metrics)
        table = comparison.method_comparison(search, cluster_metrics,
                                             winners=winners, gate=gate)
        try:
            carried = comparison.carry_forward(table)
        except ValueError:
            carried = DataFrame()

        by_key = {(s.method, format_params(s.params)): s for s in solutions}
        cache = factors.ClusterPCACache(fit_frame)
        for record in carried.to_dict('records'):
            key = (record.get('clustering_method') or record.get('method'),
                   record['hyperparameters'])
            solution = by_key.get(key)
            if solution is None:
                continue
            factor_set = factors.build_factors(fit_frame, solution,
                                               artefacts.get('correlation'), cache)
            name = record['feature_set_name']
            families[name] = pca.apply_loadings(labelled, factor_set.loadings)
            choices[name] = {'n_inputs': families[name].shape[1],
                             'chosen': f"{key[0]} ({key[1]})"}

    return families, choices


def run_outer(fit_frame: DataFrame, labelled: DataFrame, artefacts: dict,
              solutions=None, cluster_metrics: DataFrame | None = None,
              labelled_factors=None, models: dict | None = None,
              checkpoint_dir=None, verbose: bool = False
              ) -> tuple[DataFrame, DataFrame, DataFrame]:
    """The outer loop. Returns ``(predictions, selections, diagnostics)``.

    For each held-out liquid: rebuild every family from the training liquids, tune
    Elastic Net inside them by leave-one-liquid-out over those twelve, apply the Phase
    13 treatment, refit on the whole outer training fold, and predict the held-out
    liquid. The held-out liquid enters at the last step and nowhere else.

    ``models`` defaults to Ridge alone -- the plan's common controlled benchmark
    (Phase 15). Pass ``regression.models()`` to compare the wider grid; Ridge stays in
    it as the shared reference either way.

    ``checkpoint_dir`` writes each fold's rows as that fold completes and reloads them
    on a later call instead of recomputing. This run re-runs the whole clustering search
    and a 480-candidate Elastic Net grid inside every one of thirteen folds, which is
    twenty minutes of single-threaded work -- long enough that a laptop suspending
    partway through would otherwise cost the entire run. Resuming is safe because every
    fold is independent by construction: nothing crosses between them except the
    fold-independent subset-B fits, which are recomputed identically.

    A checkpoint is keyed by fold number, and the fold number's meaning is fixed by
    ``LeaveOneGroupOut`` over the same frame. ``_assert_fold_identity`` re-checks the
    held-out liquid against every reloaded checkpoint, because a stale checkpoint from a
    differently-ordered frame is exactly the fold-ordering defect this module is built
    to prevent -- and it would make the scores look better, not worse.
    """
    from pathlib import Path as _Path

    checkpoint_dir = _Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    models = ({'ridge': optimization.ridge_model(),
               'mean_baseline': optimization.baseline_model()}
              if models is None else models)

    target = labelled[cols.TARGET].reset_index(drop=True)
    groups = labelled['Name'].reset_index(drop=True)
    volumes = (labelled['volume'].reset_index(drop=True)
               if 'volume' in labelled.columns else Series(np.nan, index=target.index))
    splitter = LeaveOneGroupOut()

    prediction_rows: list[dict] = []
    selections: list[dict] = []
    diagnostics: list[dict] = []
    fold_index: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(
            splitter.split(labelled, target, groups=groups)):
        held_out = groups.iloc[test_idx].iloc[0]
        train_mask = np.zeros(len(labelled), dtype=bool)
        train_mask[train_idx] = True
        _assert_fold_identity(fold, held_out, groups.iloc[test_idx].iloc[0])

        fold_index.append({'fold': fold, 'held_out_liquid': held_out,
                           'train_idx': train_idx, 'test_idx': test_idx})

        stored = (checkpoint_dir / f'fold_{fold:02d}.csv'
                  if checkpoint_dir is not None else None)
        stored_selections = (checkpoint_dir / f'fold_{fold:02d}_selections.csv'
                             if checkpoint_dir is not None else None)
        if stored is not None and stored.is_file() and stored_selections.is_file():
            done = pd.read_csv(stored)
            done_selections = pd.read_csv(stored_selections)
            if not done.empty:
                # A checkpoint written against a different fold ordering would silently
                # pair this liquid's rows with another liquid's features.
                _assert_fold_identity(fold, held_out, str(done['liquid'].iloc[0]))
            prediction_rows.extend(done.to_dict('records'))
            selections.extend(done_selections.to_dict('records'))
            if verbose:
                print(f'  fold {fold:2d} {held_out:24s} resumed from checkpoint',
                      flush=True)
            continue

        fold_predictions: list[dict] = []
        fold_selections: list[dict] = []

        families, choices = build_fold_families(
            fit_frame, labelled, train_mask, artefacts, solutions=solutions,
            cluster_metrics=cluster_metrics, labelled_factors=labelled_factors)

        for name, values in families.items():
            if values.empty or values.shape[1] == 0:
                continue

            inner_values = values.loc[train_mask].reset_index(drop=True)
            inner_target = target[train_mask].reset_index(drop=True)
            inner_groups = groups[train_mask].reset_index(drop=True)

            grid = elastic_net.tune(inner_values, inner_target, inner_groups,
                                    feature_set=name)
            try:
                competitive, chosen = elastic_net.select(grid)
                alpha = float(chosen['alpha'])
                l1_ratio = float(chosen['l1_ratio'])
                n_competitive = int(len(competitive))
            except ValueError:
                # Nothing beat the mean inside this fold. Recorded, and the family is
                # still evaluated at the least-penalised setting so its outer-fold row
                # exists -- an absent row would read as "not measured" rather than as
                # "measured and unimpressive".
                alpha = float(config.ELASTIC_NET_ALPHA_MIN)
                l1_ratio = float(config.ELASTIC_NET_L1_RATIOS[0])
                n_competitive = 0

            fitted = elastic_net.coefficients(inner_values, inner_target,
                                              alpha, l1_ratio)
            retained = fitted.loc[fitted['selected'], 'feature'].tolist()
            # Rule 4 in its other direction: Elastic Net may also retain nothing. That
            # is an intercept-only model, which is exactly the mean baseline, and
            # scoring it as such is more honest than falling back to every feature.
            record = refinement.refine(name, values, retained, fit_frame=fit_frame)
            final = [f for f in record['final_features'] if f in values.columns]

            fold_selections.append({
                'fold': fold, 'held_out_liquid': held_out, 'feature_set': name,
                'alpha': alpha, 'l1_ratio': l1_ratio,
                'n_inputs': int(values.shape[1]),
                'n_retained': len(retained), 'n_final': len(final),
                'retained_everything': len(retained) == values.shape[1],
                'retained_nothing': not retained,
                'n_competitive': n_competitive,
                'features': ', '.join(final),
                'treatment': record['treatment'],
                'chosen_within_fold': choices.get(name, {}).get('chosen', ''),
            })
            diagnostics.append({'fold': fold, 'held_out_liquid': held_out,
                                **{k: v for k, v in record.items()
                                   if k not in ('residual', 'membership', 'loadings',
                                                'final_features')}})

            for model_name, model in models.items():
                # A family that retained nothing still gets a row for every model.
                # Skipping it would score that family on twelve outer folds while the
                # others got thirteen, and the pooled numbers would not be comparable
                # across families -- which is the one thing this table exists to do.
                #
                # The prediction is the honest one either way: a model with no inputs
                # predicts the training mean, which is what the dummy computes. The
                # 'intercept_only' flag on the row says the feature set contributed
                # nothing, so a reader can tell this apart from a model that used
                # features and merely did badly.
                data = values[final] if final else values.iloc[:, :1]
                estimator = (optimization.baseline_model() if not final else model)
                predicted = _predict(data, target, train_idx, test_idx, estimator)

                for position, index in enumerate(test_idx):
                    fold_predictions.append({
                        'feature_set': name, 'model': model_name, 'fold': fold,
                        'liquid': held_out, 'volume': float(volumes.iloc[index]),
                        'y_true': float(target.iloc[index]),
                        'y_pred': float(predicted[position]),
                        'residual': float(target.iloc[index] - predicted[position]),
                        'n_final_features': len(final),
                        'intercept_only': not final,
                    })

        prediction_rows.extend(fold_predictions)
        selections.extend(fold_selections)
        if checkpoint_dir is not None:
            DataFrame(fold_predictions).to_csv(stored, index=False)
            DataFrame(fold_selections).to_csv(stored_selections, index=False)
        if verbose:
            print(f'  fold {fold:2d} {held_out:24s} '
                  f'{len(families)} families, {len(fold_predictions)} predictions',
                  flush=True)

    predictions = DataFrame(prediction_rows)
    predictions = _charge_unavailable_families(
        predictions, fold_index, target, groups, volumes, list(models))

    return (predictions, DataFrame(selections), DataFrame(diagnostics))


def _charge_unavailable_families(predictions: DataFrame, fold_index: list,
                                 target: Series, groups: Series, volumes: Series,
                                 model_names: list) -> DataFrame:
    """Score a family on every fold, including the folds where it produced nothing.

    A pipeline does not always yield a feature set. On this data the automatic
    clustering pipeline yields none in 3 of 13 folds -- inside those training liquids no
    clustering method both cleared the admission gate and beat the mean, so
    ``carry_forward`` has nothing to return.

    **That is a result about the pipeline, and it has to be charged to it.** Leaving the
    fold out would score the automatic clustering on 10 folds while every other family
    is scored on 13, so its mean would be taken over only the liquids where it happened
    to work -- crediting it for its successes and excusing its failures. The two numbers
    would not be comparable, which is the one thing this table exists to do.

    What it can actually do on such a fold is predict the training mean, so that is what
    it is scored at, with ``family_unavailable=True`` distinguishing this from a family
    that was built and then retained nothing (``intercept_only`` without
    ``family_unavailable``). Both predict the mean; only one of them had a pipeline that
    ran.
    """
    if predictions.empty or not fold_index:
        return predictions

    families = sorted(predictions['feature_set'].unique())
    present = set(zip(predictions['feature_set'], predictions['fold']))

    filled: list[dict] = []
    for entry in fold_index:
        fold, held_out = entry['fold'], entry['held_out_liquid']
        train_mean = float(target.iloc[entry['train_idx']].mean())
        for name in families:
            if (name, fold) in present:
                continue
            for model_name in model_names:
                for index in entry['test_idx']:
                    filled.append({
                        'feature_set': name, 'model': model_name, 'fold': fold,
                        'liquid': held_out, 'volume': float(volumes.iloc[index]),
                        'y_true': float(target.iloc[index]),
                        'y_pred': train_mean,
                        'residual': float(target.iloc[index] - train_mean),
                        'n_final_features': 0,
                        'intercept_only': True,
                        'family_unavailable': True,
                    })

    if 'family_unavailable' not in predictions.columns:
        predictions = predictions.assign(family_unavailable=False)
    if not filled:
        return predictions

    return pd.concat([predictions, DataFrame(filled)], ignore_index=True)


# ---------------------------------------------------------------------------
# Phase 15: scoring the outer folds
# ---------------------------------------------------------------------------

def score(predictions: DataFrame) -> DataFrame:
    """Pooled out-of-fold scores, one row per (feature set, model).

    Reuses ``allmeta.scoring.pooled_scores``, which computes the pooled and
    between-liquid readings, attaches the mean-baseline comparison, and carries the
    derivation of why neither R-squared has its zero at the baseline. Reimplementing it
    here would create two numbers that should agree and eventually will not.

    ``leakage=False`` is passed in **by construction, not by name lookup**: every set in
    this table was chosen inside the fold whatever it is called. Before the Phase 20
    split, ``pooled_scores`` applied the legacy ``subset::method`` name parse and this
    function corrected it afterwards -- which happened to give the right answer only
    because these family names contain no ``::``.
    """
    from allmeta.scoring import pooled_scores

    if predictions.empty:
        return DataFrame()

    frame = predictions.copy()
    if 'include_params' not in frame.columns:
        frame['include_params'] = False

    # leakage=False is passed in rather than computed and overwritten. The old route
    # let pooled_scores apply the legacy `subset::method` name parse and then corrected
    # it; that gave the right answer only because these family names contain no '::'.
    scores = pooled_scores(frame, leakage=False)
    scores['evaluation'] = 'nested_outer_fold'

    sizes = (frame.groupby('feature_set')['n_final_features']
             .agg(n_features_median='median', n_features_min='min',
                  n_features_max='max').reset_index())

    return scores.merge(sizes, on='feature_set', how='left')


def fold_scores(predictions: DataFrame) -> DataFrame:
    """Per-fold error, one row per (feature set, model, held-out liquid).

    Per-fold R-squared is deliberately absent: a fold holds one liquid whose eight rows
    share a descriptor vector, so the within-fold variance of the truth is the
    within-liquid spread and an R-squared against it measures nothing the descriptors
    can address. The liquid-mean squared error is the fold-level quantity that means
    something, and it is the same one Phases 8-12 selected on.
    """
    if predictions.empty:
        return DataFrame()

    rows = []
    for (name, model, fold), group in predictions.groupby(
            ['feature_set', 'model', 'fold']):
        actual = group['y_true'].to_numpy()
        predicted = group['y_pred'].to_numpy()
        rows.append({
            'feature_set': name, 'model': model, 'fold': int(fold),
            'held_out_liquid': group['liquid'].iloc[0],
            'liquid_squared_error': float((predicted.mean() - actual.mean()) ** 2),
            'rmse': float(np.sqrt(np.mean((actual - predicted) ** 2))),
            'mae': float(np.mean(np.abs(actual - predicted))),
            'n_final_features': int(group['n_final_features'].iloc[0]),
            'intercept_only': bool(group['intercept_only'].iloc[0]),
        })

    return DataFrame(rows).sort_values(['feature_set', 'model', 'fold'],
                                       ignore_index=True)


def intervals(predictions: DataFrame, model: str = 'ridge',
              n_boot: int | None = None) -> DataFrame:
    """Bootstrap intervals on the outer-fold scores, resampling LIQUIDS.

    Thirteen liquids is the real sample size. Resampling the 104 rows would report the
    precision of a study that does not exist, and at this n the interval is the result --
    a point estimate of between-liquid R-squared from thirteen liquids is not something
    to quote on its own.
    """
    from allmeta.scoring import bootstrap_between_r2, bootstrap_pooled_rmse

    if predictions.empty:
        return DataFrame()

    frame = predictions.copy()
    if 'include_params' not in frame.columns:
        frame['include_params'] = False

    rows = []
    for name in sorted(frame['feature_set'].unique()):
        view = frame[(frame['feature_set'] == name) & (frame['model'] == model)]
        if view.empty:
            continue
        kwargs = {} if n_boot is None else {'n_boot': n_boot}
        record = bootstrap_between_r2(frame, name, model, **kwargs)
        record.update(bootstrap_pooled_rmse(frame, name, model, **kwargs))
        record['resampling_unit'] = 'liquid'
        rows.append(record)

    return DataFrame(rows).sort_values('observed', ascending=False, ignore_index=True)


def summary(scores: DataFrame, selections: DataFrame,
            intervals_table: DataFrame | None = None,
            model: str = 'ridge') -> DataFrame:
    """One row per feature family: what it is, what it chose, and what it scored.

    The plan's ``feature_set_summary.csv``. Everything a reader needs to decide whether
    a family is worth anything, on one line, with the baseline in the same table rather
    than in a docstring.
    """
    if scores.empty:
        return DataFrame()

    view = scores[scores['model'] == model].copy()

    if not selections.empty:
        chosen = (selections.groupby('feature_set')
                  .agg(n_inputs=('n_inputs', 'max'),
                       n_final_median=('n_final', 'median'),
                       n_final_min=('n_final', 'min'),
                       n_final_max=('n_final', 'max'),
                       folds_retaining_everything=('retained_everything', 'sum'),
                       folds_retaining_nothing=('retained_nothing', 'sum'),
                       n_folds_with_a_selection=('fold', 'nunique'),
                       median_alpha=('alpha', 'median'),
                       treatment=('treatment', 'first'))
                  .reset_index())
        view = view.merge(chosen, on='feature_set', how='left')
        # Named 'n_folds_with_a_selection', not 'n_folds'. Every family is SCORED on
        # all thirteen outer folds -- _charge_unavailable_families guarantees that --
        # but a family is only SELECTED in the folds where its pipeline produced
        # something. auto_cluster_pc1 selects in 9 of 13. The two columns sitting side
        # by side under one name reads as a bug in the scoring; under two names it
        # reads as what it is.
        view['n_folds_scored'] = view['n_liquids']
        view['selection_available_in_every_fold'] = (
            view['n_folds_with_a_selection'] == view['n_folds_scored'])

        # Selections at the least-penalised end of the grid come from coordinate-descent
        # paths that did not converge (duality gaps ~1e-2 against tolerances ~5e-5), so
        # their coefficient counts are not sparsity results. Flagged rather than
        # dropped: the score is still a valid held-out score, it is the *feature count*
        # that must not be quoted.
        from allmeta.feature_selection import config as _config
        view['selected_at_alpha_floor'] = (
            view['median_alpha'] <= _config.ELASTIC_NET_ALPHA_MIN * 1.001)

    if intervals_table is not None and not intervals_table.empty:
        carried = intervals_table[['feature_set', 'ci_lo', 'ci_hi', 'frac_above_zero',
                                   'rmse_ci_lo', 'rmse_ci_hi']]
        view = view.merge(carried, on='feature_set', how='left')

    return view.sort_values('between_liquid_r2', ascending=False, ignore_index=True)
