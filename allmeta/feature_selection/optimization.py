"""Choosing a clustering candidate on predictive error (implementation plan, Phase 8).

Phases 4-7 produced 430 candidates and scored each on cluster quality, PC1 retention and
stability. None of those decides anything, and the plan is explicit about why: the
clusters exist to produce factors that predict dispensing error, not to produce tidy
partitions, so the primary criterion has to be predictive.

## What is and is not leakage-safe here

The clusters and their PC1 loadings were fitted on subset B, which carries no target, so
applying them to the labelled liquids leaks nothing. What *can* leak is the **choice
among candidates**, because that choice uses the labels.

Two searches are therefore produced and they answer different questions:

``search`` (``clustering_hyperparameter_search.csv``)
    Every candidate scored by leave-one-liquid-out CV over all 13 labelled liquids. A
    diagnostic: it says which candidates look good and how much they differ. Its winner
    is a minimum over 430 noisy estimates and its error is **not** a performance
    estimate. Every row carries ``selected_on_all_labels=True``.

``select_by_outer_fold`` (``selected_clustering_configs_by_outer_fold.csv``)
    For each held-out liquid, the whole search rerun using only the other twelve, and
    the one-standard-error rule applied to that. The held-out liquid influences neither
    the clusters nor the choice. This is the leakage-safe object, and the honest
    performance number for the pipeline as a whole still comes from the outer folds in
    Phase 14 -- not from here.

## The ceiling nothing can beat

Each liquid contributes eight rows, one per pipetting volume, and all eight share one
descriptor vector. A descriptor-only model therefore predicts one value for all eight,
and **56.5% of the target's variance is within-liquid** and structurally unreachable.
Pooled R-squared cannot exceed about 0.435 however good the features are, and pooled
RMSE has a floor well above zero. ``between_liquid_r2`` is reported beside the pooled
figures because it is the part a feature set can actually move.

Under leave-one-out the mean predictor does not score zero either: holding out liquid i,
the training mean is inflated by exactly n/(n-1), so R-squared for the mean baseline is
1 - (n/(n-1))^2, which is -0.174 at n=13. The dummy baseline is scored alongside every
candidate rather than assumed.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from allmeta import columns as cols
from allmeta.feature_selection import config, factors, pca
from allmeta.feature_selection.algorithms import format_params
from allmeta.paths import RANDOM_STATE

BASELINE_NAME: str = 'mean_baseline'


def baseline_model():
    """The mean predictor, scored on the same folds as every candidate."""
    return DummyRegressor(strategy='mean')


def ridge_model(alpha: float | None = None):
    """The common downstream model. See ``config.BASELINE_RIDGE_ALPHA``."""
    alpha = config.BASELINE_RIDGE_ALPHA if alpha is None else alpha

    # random_state is not passed: it is inert for Ridge's default solver, and passing it
    # would imply a stochastic fit that does not exist.
    return Ridge(alpha=alpha)


@dataclass(frozen=True)
class LabelledFactors:
    """Every candidate's factor scores on the labelled liquids.

    Built once and reused by both searches. The loadings come from the subset-B fit, so
    nothing here is refitted against the labels -- ``pca.apply_loadings`` only evaluates
    persisted constants.
    """

    values: dict[tuple[str, str], DataFrame]
    target: Series
    groups: Series

    def __len__(self) -> int:
        return len(self.values)

    def keys(self) -> list[tuple[str, str]]:
        return list(self.values)


def build_labelled_factors(fit_frame: DataFrame, labelled: DataFrame, solutions,
                           rho: DataFrame | None = None) -> LabelledFactors:
    """Carry every candidate's cluster-PC1 factors onto the labelled liquids."""
    if labelled[cols.TARGET].isna().any():
        raise ValueError('labelled frame has missing targets')

    cache = factors.ClusterPCACache(fit_frame)
    values: dict[tuple[str, str], DataFrame] = {}

    for solution in solutions:
        factor_set = factors.build_factors(fit_frame, solution, rho, cache)
        scores = pca.apply_loadings(labelled, factor_set.loadings)
        values[(solution.method, format_params(solution.params))] = scores

    return LabelledFactors(values=values,
                           target=labelled[cols.TARGET].reset_index(drop=True),
                           groups=labelled['Name'].reset_index(drop=True))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def between_liquid_r2(truth: Series, predicted: np.ndarray, groups: Series) -> float:
    """R-squared on the liquid means: the part descriptors can actually move.

    A descriptor-only model gives one prediction per liquid, so scoring it on all 104
    rows charges it for the 56.5% of variance that lives between volumes of the same
    liquid and that no liquid-level feature can reach.
    """
    frame = DataFrame({'y': truth.to_numpy(), 'p': predicted,
                       'g': groups.to_numpy()}).groupby('g').mean()

    return float(r2_score(frame['y'], frame['p']))


def cross_validate(values: DataFrame, target: Series, groups: Series,
                   model=None) -> dict:
    """Leave-one-liquid-out scores for one feature representation.

    Two losses are computed for every candidate, because they answer different
    questions and only one of them can discriminate here:

    ``pooled_rmse_*``  per-fold RMSE over every row. The literal reading of the plan's
                       criterion, and the one that cannot separate candidates on this
                       data -- see ``config.SELECTION_CRITERION``.
    ``liquid_mse_*``   squared error of the held-out liquid's MEAN prediction, one
                       observation per fold. Drops the within-liquid variance that no
                       liquid-level feature can reach and that enters every candidate's
                       pooled error identically.

    Each carries its own mean, standard deviation and standard error across folds; the
    SE is what the one-standard-error rule consumes.

    The estimator is cloned per fold. ``Pipeline`` does not clone its steps, so reusing
    one instance across folds is one warm-start default away from silent cross-fold
    contamination.
    """
    model = ridge_model() if model is None else model
    splitter = LeaveOneGroupOut()

    fold_rmse: list[float] = []
    fold_mae: list[float] = []
    fold_liquid_squared: list[float] = []
    fold_nonzero: list[int] = []
    predictions = np.empty(len(target), dtype=float)

    for train_idx, test_idx in splitter.split(values, target, groups=groups):
        pipeline = Pipeline([('scale', RobustScaler()), ('model', clone(model))])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipeline.fit(values.iloc[train_idx], target.iloc[train_idx])
            predicted = pipeline.predict(values.iloc[test_idx])

        actual = target.iloc[test_idx].to_numpy()
        predictions[test_idx] = predicted
        fold_rmse.append(float(root_mean_squared_error(actual, predicted)))
        fold_mae.append(float(mean_absolute_error(actual, predicted)))
        # The held-out liquid's eight rows share one descriptor vector, so the model
        # predicts one value for all of them; the mean is that value, and comparing it
        # with the liquid's mean measured error is the whole of what it can be right or
        # wrong about.
        error = float(predicted.mean() - actual.mean())
        fold_liquid_squared.append(error * error)

        # Sparsity, when the estimator has any. Recorded per fold and averaged rather
        # than taken from a single refit: Elastic Net drops a different set in each
        # fold, so one number would be a fiction -- the same reason evaluate_nested
        # reports a median and a range for wrapper_rfecv.
        estimator = pipeline.named_steps['model']
        coefficients = getattr(estimator, 'coef_', None)
        if coefficients is not None:
            fold_nonzero.append(int(np.count_nonzero(np.ravel(coefficients))))

    scores = {'n_factors': int(values.shape[1]), 'n_folds': len(fold_rmse)}
    scores.update(_spread('pooled_rmse', fold_rmse))
    scores.update(_spread('liquid_mse', fold_liquid_squared))
    scores.update({
        'liquid_rmse': float(np.sqrt(np.mean(fold_liquid_squared))),
        'mae_mean': float(np.mean(fold_mae)),
        'pooled_rmse': float(root_mean_squared_error(target, predictions)),
        'pooled_r2': float(r2_score(target, predictions)),
        'between_liquid_r2': between_liquid_r2(target, predictions, groups),
    })

    if fold_nonzero:
        counts = np.asarray(fold_nonzero, dtype=float)
        scores.update({
            'n_nonzero_mean': float(counts.mean()),
            'n_nonzero_min': int(counts.min()),
            'n_nonzero_max': int(counts.max()),
            'retained_fraction': float(counts.mean() / values.shape[1])
            if values.shape[1] else np.nan,
            # Rule 4: retaining everything is a legal, common outcome and must be
            # readable as one rather than inferred from a count.
            'retained_all': bool(counts.min() == values.shape[1]),
            'retained_none': bool(counts.max() == 0),
        })

    return scores


def _spread(name: str, losses) -> dict:
    """Mean, standard deviation and standard error of a per-fold loss."""
    values = np.asarray(losses, dtype=float)
    n = len(values)
    deviation = float(values.std(ddof=1)) if n > 1 else 0.0

    return {f'{name}_mean': float(values.mean()),
            f'{name}_std': deviation,
            f'{name}_se': deviation / np.sqrt(n) if n > 1 else 0.0}


def search(labelled_factors: LabelledFactors, subset: Series | None = None,
           model=None) -> DataFrame:
    """Score every candidate by leave-one-liquid-out CV.

    ``subset`` is a boolean mask over the labelled rows, used by the per-outer-fold
    search to restrict scoring to that fold's training liquids. None scores every
    labelled liquid, which is the diagnostic search -- see the module docstring for why
    its winner is not a performance estimate.
    """
    target = labelled_factors.target
    groups = labelled_factors.groups

    if subset is not None:
        keep = subset.to_numpy()
        target = target[keep].reset_index(drop=True)
        groups = groups[keep].reset_index(drop=True)

    rows: list[dict] = []
    for (method, hyperparameters), values in labelled_factors.values.items():
        block = values.reset_index(drop=True)
        if subset is not None:
            block = block[subset.to_numpy()].reset_index(drop=True)

        rows.append({
            'method': method,
            'hyperparameters': hyperparameters,
            **cross_validate(block, target, groups, model=model),
        })

    # The mean predictor on the same folds. Under leave-one-out it does NOT score
    # R-squared of zero -- see the module docstring -- so it is measured rather than
    # assumed, and every candidate is read against this row.
    first = next(iter(labelled_factors.values.values())).reset_index(drop=True)
    if subset is not None:
        first = first[subset.to_numpy()].reset_index(drop=True)
    rows.append({
        'method': BASELINE_NAME,
        'hyperparameters': '',
        **cross_validate(first, target, groups, model=baseline_model()),
    })

    table = DataFrame(rows)
    criterion = f'{config.SELECTION_CRITERION}_mean'
    baseline = table.loc[table['method'] == BASELINE_NAME, criterion].iloc[0]
    table['selection_criterion'] = config.SELECTION_CRITERION
    table['beats_baseline'] = table[criterion] < baseline
    table['baseline_criterion_mean'] = baseline

    return table.sort_values(criterion, ignore_index=True)


# ---------------------------------------------------------------------------
# One-standard-error selection
# ---------------------------------------------------------------------------

def one_standard_error(table: DataFrame, context: DataFrame | None = None,
                       tie_break=None, interpretability=None,
                       criterion: str | None = None,
                       require_beats_baseline: bool | None = None,
                       ) -> tuple[DataFrame, pd.Series]:
    """Competitive candidates within one SE of the best, and the one chosen.

    The rule: take the lowest mean RMSE ``E_min`` and ITS standard error ``SE_min``,
    then treat every candidate with ``E <= E_min + SE_min`` as statistically
    indistinguishable from it. Choosing the outright minimum instead would be choosing
    the luckiest of 430 noisy estimates.

    Among the competitive set the tie-break is applied in the order given by
    ``config.SELECTION_TIE_BREAK`` -- parsimony, then stability, then correlation
    coherence, then PC1 retention, then internal geometry -- and settled by
    ``config.METHOD_INTERPRETABILITY`` if everything above ties exactly. Every step is
    explicit because an implicit tie-break is a decision nobody can audit.

    ``context`` supplies the Phase 5-7 columns the tie-break reads, joined on
    ``(method, hyperparameters)``. Without it only parsimony is available, and that is
    reported rather than silently skipped.
    """
    tie_break = config.SELECTION_TIE_BREAK if tie_break is None else tie_break
    interpretability = (config.METHOD_INTERPRETABILITY if interpretability is None
                        else interpretability)
    criterion = config.SELECTION_CRITERION if criterion is None else criterion
    require_beats_baseline = (config.REQUIRE_BEATS_BASELINE
                              if require_beats_baseline is None
                              else require_beats_baseline)
    mean_column, se_column = f'{criterion}_mean', f'{criterion}_se'

    ranked = table[table['method'] != BASELINE_NAME].copy()
    if ranked.empty:
        raise ValueError('no candidates to select from')
    if mean_column not in ranked.columns:
        raise ValueError(f'table carries no {mean_column!r}; '
                         f'criterion {criterion!r} cannot be applied')

    # A rule that picks among candidates indistinguishable from the best says nothing
    # about whether the best is any good. Without this, a grid where everything loses to
    # the mean still returns a confident-looking winner.
    baseline_rows = table[table['method'] == BASELINE_NAME]
    guarded = False
    if require_beats_baseline and not baseline_rows.empty:
        baseline = float(baseline_rows[mean_column].iloc[0])
        beating = ranked[ranked[mean_column] < baseline]
        if beating.empty:
            raise ValueError(
                f'no candidate beats the mean baseline on {criterion} '
                f'({baseline:.6g}); there is nothing to select. Pass '
                'require_beats_baseline=False to select anyway.')
        ranked = beating.copy()
        guarded = True

    if context is not None:
        carried = [c for c in context.columns
                   if c not in ('method', 'hyperparameters')]
        ranked = ranked.merge(context[['method', 'hyperparameters'] + carried],
                              on=['method', 'hyperparameters'], how='left',
                              suffixes=('', '_context'))

    best = ranked.loc[ranked[mean_column].idxmin()]
    threshold = float(best[mean_column] + best[se_column])

    competitive = ranked[ranked[mean_column] <= threshold].copy()
    competitive['within_one_se'] = True
    competitive['one_se_threshold'] = threshold
    competitive['best_criterion_mean'] = float(best[mean_column])
    competitive['selection_criterion'] = criterion
    competitive['baseline_guard_applied'] = guarded

    order = []
    ascending = []
    applied = []
    for column, direction in tie_break:
        if column in competitive.columns and competitive[column].notna().any():
            order.append(column)
            ascending.append(direction == 'lower')
            applied.append(f'{column} ({direction})')

    rank_of = {method: position for position, method in enumerate(interpretability)}
    competitive['_interpretability'] = competitive['method'].map(
        lambda m: rank_of.get(m, len(rank_of)))
    order.append('_interpretability')
    ascending.append(True)
    # Final, arbitrary but DETERMINISTIC: without it the winner depends on row order,
    # which changes when the grid changes.
    order.append('hyperparameters')
    ascending.append(True)

    competitive = competitive.sort_values(order, ascending=ascending,
                                          ignore_index=True)
    competitive['tie_break_applied'] = ', '.join(applied) or 'parsimony unavailable'
    competitive = competitive.drop(columns='_interpretability')

    return competitive, competitive.iloc[0]


def random_candidate_baseline(table: DataFrame, n_draws: int | None = None,
                              random_state: int = RANDOM_STATE) -> dict:
    """What an arbitrarily chosen candidate would have scored.

    The comparison that says whether selecting helped at all. The selected candidate's
    error is a minimum over hundreds of noisy estimates and flatters itself by
    construction; if the median arbitrary candidate is nearly as good, the search bought
    ordering and not performance.
    """
    n_draws = config.N_RANDOM_CANDIDATE_DRAWS if n_draws is None else n_draws
    criterion = f'{config.SELECTION_CRITERION}_mean'
    pool = table.loc[table['method'] != BASELINE_NAME, criterion].to_numpy()
    if pool.size == 0:
        raise ValueError('no candidates to draw from')

    rng = np.random.default_rng(random_state)
    draws = rng.choice(pool, size=n_draws, replace=True)

    return {
        'n_draws': n_draws,
        'selection_criterion': config.SELECTION_CRITERION,
        'random_mean': float(draws.mean()),
        'random_median': float(np.median(draws)),
        'random_p10': float(np.percentile(draws, 10)),
        'random_p90': float(np.percentile(draws, 90)),
        'best_criterion_mean': float(pool.min()),
    }


# ---------------------------------------------------------------------------
# Leakage-safe selection, one choice per held-out liquid
# ---------------------------------------------------------------------------

def select_by_outer_fold(labelled_factors: LabelledFactors,
                         context: DataFrame | None = None,
                         model=None) -> tuple[DataFrame, DataFrame]:
    """Rerun the whole search inside each outer training fold.

    For each held-out liquid the search sees only the other twelve, so the held-out
    liquid influenced neither the clusters (fitted on subset B, no target) nor the
    choice among them. Returns ``(selections, per_fold_search)``.

    Expect the chosen configuration to move between folds. That is not a defect of the
    implementation, it is the measurement: with twelve training liquids, a search over
    430 candidates has very little to go on, and how much the winner moves is exactly
    what ``selection_summary`` reports.
    """
    groups = labelled_factors.groups
    liquids = list(dict.fromkeys(groups))

    selections: list[dict] = []
    searches: list[DataFrame] = []

    for fold, held_out in enumerate(liquids):
        mask = groups != held_out
        table = search(labelled_factors, subset=mask, model=model)
        table.insert(0, 'held_out_liquid', held_out)
        table.insert(0, 'fold', fold)
        searches.append(table)

        competitive, chosen = one_standard_error(table, context=context)
        selections.append({
            'fold': fold,
            'held_out_liquid': held_out,
            'method': chosen['method'],
            'hyperparameters': chosen['hyperparameters'],
            'n_factors': int(chosen['n_factors']),
            'selection_criterion': chosen['selection_criterion'],
            'criterion_mean': float(chosen[f"{chosen['selection_criterion']}_mean"]),
            'criterion_se': float(chosen[f"{chosen['selection_criterion']}_se"]),
            'pooled_rmse_mean': float(chosen['pooled_rmse_mean']),
            'between_liquid_r2': float(chosen['between_liquid_r2']),
            'best_criterion_mean': float(chosen['best_criterion_mean']),
            'one_se_threshold': float(chosen['one_se_threshold']),
            'n_competitive': int(len(competitive)),
            'n_candidates': int((table['method'] != BASELINE_NAME).sum()),
            'tie_break_applied': chosen['tie_break_applied'],
        })

    return DataFrame(selections), pd.concat(searches, ignore_index=True)


def selection_summary(selections: DataFrame) -> DataFrame:
    """How much the chosen configuration moves between outer folds.

    The headline honesty check on Phase 8. If a different configuration wins in most
    folds, then "the best clustering" is not a property of the data at this sample size,
    and any single configuration quoted from the diagnostic search is an artefact of
    which liquids happened to be in it.
    """
    n_folds = len(selections)
    by_config = (selections.groupby(['method', 'hyperparameters'])
                 .size().rename('n_folds_selected').reset_index()
                 .sort_values('n_folds_selected', ascending=False, ignore_index=True))
    by_config['fraction_of_folds'] = by_config['n_folds_selected'] / n_folds

    by_method = (selections.groupby('method').size()
                 .rename('n_folds_selected').reset_index())
    by_method['fraction_of_folds'] = by_method['n_folds_selected'] / n_folds
    by_method['hyperparameters'] = '(any)'

    summary = pd.concat([by_config, by_method[by_config.columns]], ignore_index=True)
    summary['scope'] = ['configuration'] * len(by_config) + ['method'] * len(by_method)
    summary['n_folds'] = n_folds
    summary['n_distinct_configurations'] = by_config.shape[0]
    summary['n_distinct_methods'] = by_method.shape[0]

    return summary
