"""Scoring shared by the legacy regression driver and the new nested pipeline.

Extracted from ``regression.py`` in Phase 20. Nothing here fits a feature selection or
chooses a model: it turns out-of-fold predictions into scores, intervals and nulls, and
both pipelines need exactly the same arithmetic.

Keeping one copy is the point. ``regression.evaluate_nested`` and
``feature_selection.nested.run_outer`` produce predictions by different routes and must
be comparable, which they can only be if the same code scores them. Two copies of
``pooled_scores`` would be two places for the leave-one-out baseline derivation below to
drift apart.

## The baseline is not zero, and this is where that is written down

Under leave-one-out the held-out liquid is excluded from the training mean, so the
residual is inflated by exactly n/(n-1) and

    R2_LOO,mean = 1 - (n / (n - 1))**2

which is -0.0756 pooled and -0.1736 between-liquid at thirteen liquids. A configuration
scoring R2 = -0.05 is therefore BETTER than predicting the mean, not worse.
``baseline_r2`` reads it from the data rather than hardcoding it, because it moves with
the liquid count.

## Resampling is by liquid, everywhere

Both bootstraps draw liquids, not rows. Thirteen liquids is the real sample size, and
resampling the 104 rows would report the precision of a study that does not exist.
``permuted_targets`` permutes at the liquid level for the same reason: each liquid keeps
its own eight measured errors and the whole block moves together, matched on volume, so
the within-liquid structure survives into the null.
"""

import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from allmeta import columns as cols
from allmeta.paths import RANDOM_STATE
from allmeta.subsets import load_subset

BASELINE: str = 'baseline_mean'
N_BOOTSTRAP: int = 4000


def target_variance_decomposition(frame: DataFrame | None = None) -> dict[str, float]:
    """Split the target's variance into within-liquid and between-liquid parts.

    A liquid's eight rows share one descriptor vector, so a descriptor-only model
    predicts a single value for all eight and cannot address the spread among them
    no matter how good it is. That within-liquid share is therefore a hard ceiling
    on pooled R-squared, and it is large here -- reporting a pooled R-squared
    without it makes the chemistry look about twice as weak as it is.
    """
    if frame is None:
        frame = load_subset('C')

    target = frame[cols.TARGET]
    liquid_mean = target.groupby(frame['Name']).transform('mean')

    ss_total = float(((target - target.mean()) ** 2).sum())
    ss_within = float(((target - liquid_mean) ** 2).sum())

    return {
        'ss_total': ss_total,
        'ss_within_liquid': ss_within,
        'ss_between_liquid': ss_total - ss_within,
        'within_liquid_fraction': ss_within / ss_total,
        'descriptor_only_r2_ceiling': 1.0 - ss_within / ss_total,
    }

def pooled_scores(predictions: DataFrame, leakage=None) -> DataFrame:
    """Score each configuration on its pooled out-of-fold predictions.

    ``r2`` here is computed against the variance of the whole target, so a negative
    value means the model does worse than predicting the overall mean -- the result
    that matters most at this sample size.

    ``between_liquid_r2`` is the same comparison made on the thirteen liquid means,
    which is the question the descriptors are actually being asked: does the model
    rank and place *liquids* correctly? Pooled ``r2`` charges a descriptor-only model
    for the within-liquid spread it structurally cannot reach (see
    ``target_variance_decomposition``), so the two differ by roughly a factor of two
    here. The ranking among descriptor-only sets is unchanged; the magnitude is not.
    Note also that this makes ``r2`` non-comparable across ``include_params``: with
    the pipetting parameters in, the model *can* address the within-liquid spread,
    and most of the apparent gain is the relative-error-falls-with-volume identity
    rather than chemistry.

    **Neither R-squared has its zero at the mean baseline.** Under leave-one-out the
    held-out liquid is excluded from the training mean, so ybar_-i = (n*ybar - y_i) /
    (n - 1), the residual is inflated by exactly n/(n - 1), and

        R2_LOO,mean = 1 - (n / (n - 1))**2

    -- at thirteen liquids, -0.0756 pooled and -0.1736 between-liquid, which is what
    the ``baseline_mean`` rows below report. A configuration scoring R2 = -0.05 is
    therefore *better* than predicting the mean, not worse. Recorded here so the next
    reader does not re-derive it, and read from the data by ``baseline_r2`` rather
    than hardcoded anywhere, because it moves with the liquid count.
    """
    rows: list[dict] = []
    for (set_name, include_params, model), group in predictions.groupby(
            ['feature_set', 'include_params', 'model']):
        actual = group['y_true'].to_numpy()
        predicted = group['y_pred'].to_numpy()

        rho = np.nan
        if len(np.unique(predicted)) > 1:
            rho = float(spearmanr(actual, predicted).statistic)

        liquid = group.groupby('liquid')[['y_true', 'y_pred']].mean()
        between_r2 = float(r2_score(liquid['y_true'], liquid['y_pred']))
        between_rho = np.nan
        if liquid['y_pred'].nunique() > 1:
            between_rho = float(spearmanr(liquid['y_true'], liquid['y_pred']).statistic)

        rows.append({
            'feature_set': set_name,
            'include_params': include_params,
            'model': model,
            'rmse': float(root_mean_squared_error(actual, predicted)),
            'mae': float(mean_absolute_error(actual, predicted)),
            'r2': float(r2_score(actual, predicted)),
            'spearman': rho,
            'between_liquid_r2': between_r2,
            'between_liquid_spearman': between_rho,
            'n_predictions': len(group),
            'n_liquids': int(liquid.shape[0]),
        })

    scores = DataFrame(rows)

    # How each model compares with the mean baseline on the same feature set: the
    # only comparison that distinguishes a useful model from a lucky one.
    baseline = (
        scores[scores['model'] == BASELINE]
        .set_index(['feature_set', 'include_params'])['rmse']
    )
    scores['baseline_rmse'] = [
        baseline.get((r.feature_set, r.include_params), np.nan) for r in scores.itertuples()
    ]
    scores['rmse_vs_baseline'] = scores['rmse'] - scores['baseline_rmse']
    scores['beats_baseline'] = scores['rmse'] < scores['baseline_rmse']

    # Whether the feature set's members were chosen using this same target on these
    # same labelled liquids. Such scores are optimistically biased and are not
    # comparable with the honest ones.
    #
    # A PARAMETER rather than a rule baked in here, because the right answer depends on
    # the caller. The legacy driver passes regression.selected_on_labels, which reads
    # the `subset::method` naming convention. feature_selection.nested passes False:
    # every set it scores was chosen inside the fold, whatever it is called.
    #
    # Hardcoding the legacy rule made the new pipeline's answer correct BY ACCIDENT --
    # its feature sets contain no '::', so the name parse returned False for all of
    # them. A future set named 'C::something' would have been silently flagged leaky.
    # None leaves the column NA, which is honest, rather than guessing.
    if leakage is None:
        scores['selection_leakage'] = pd.NA
    elif callable(leakage):
        scores['selection_leakage'] = scores['feature_set'].map(leakage)
    else:
        scores['selection_leakage'] = bool(leakage)

    return scores.sort_values(['include_params', 'rmse'], ignore_index=True)

def permuted_targets(target: pd.Series, groups: pd.Series, volumes: pd.Series,
                     n_permutations: int, random_state: int = RANDOM_STATE
                     ) -> list[pd.Series]:
    """The permutation mechanism, shared by the aggregate and nested nulls.

    Labels are permuted **at the liquid level**: each liquid keeps its own eight
    measured errors, but that whole block is reassigned to a different liquid's
    descriptors, matched on volume so the volume structure survives too. Shuffling
    rows instead would break the within-liquid structure and hand the null an easier
    problem than the real model faces, which is the usual way a permutation test is
    made to pass.

    Every mapping is drawn here, **in order, from one seeded generator**, before any
    fitting happens. That is what lets the callers fit the draws in parallel without
    changing a single number: parallelism decides when a draw is scored, never which
    draw it is. Element 0 is the unpermuted target and consumes no randomness.
    """
    rng = np.random.default_rng(random_state)
    liquids = list(dict.fromkeys(groups))
    lookup = target.groupby([groups, volumes]).mean()

    out: list[pd.Series] = [target]
    for _ in range(n_permutations):
        mapping = dict(zip(liquids, rng.permutation(liquids)))
        donor = groups.map(mapping)
        permuted = pd.Series(
            [lookup.get((d, v), np.nan) for d, v in zip(donor, volumes)],
            index=target.index)
        if permuted.isna().any():
            raise ValueError('permutation produced unmatched (liquid, volume) pairs')
        out.append(permuted)

    return out

def _logo_predictions(data: DataFrame, target: pd.Series, groups: pd.Series,
                      model_name: str, n_jobs: int = 1) -> np.ndarray:
    """Out-of-fold predictions for one target vector under leave-one-liquid-out.

    Pulled out of ``permutation_null``'s draw loop so a draw is one call, which is
    what makes the loop parallelisable without touching the permutation itself.
    """
    predicted = np.empty(len(data))
    for train_idx, test_idx in LeaveOneGroupOut().split(data, target, groups=groups):
        pipeline = Pipeline([('scale', RobustScaler()),
                             ('model', models(n_jobs=n_jobs)[model_name])])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipeline.fit(data.iloc[train_idx], target.iloc[train_idx])
            predicted[test_idx] = pipeline.predict(data.iloc[test_idx])

    return predicted

def of_evaluation(frame: DataFrame, label: str = 'aggregate') -> DataFrame:
    """Rows of an artefact belonging to one evaluation ('aggregate' or 'nested').

    Tolerates the column being absent, which it is in any file written before the
    nested evaluation existed. ``frame.get('evaluation', 'aggregate')`` looks like it
    would do this but returns a *scalar* when the column is missing, and indexing a
    frame with a bare bool raises rather than selecting everything.
    """
    if 'evaluation' not in frame.columns:
        return frame if label == 'aggregate' else frame.iloc[0:0]

    return frame[frame['evaluation'] == label]

def baseline_r2(scores: DataFrame, column: str = 'r2') -> float:
    """The mean predictor's actual out-of-fold R-squared, read from the data.

    Under leave-one-out the mean predictor does **not** score zero. Holding out liquid
    *i*, the training mean is ybar_-i = (n*ybar - y_i)/(n - 1), so every residual is
    inflated by exactly n/(n - 1) and

        R2_LOO,mean = 1 - (n / (n - 1))**2

    which at thirteen liquids is -0.0756 pooled and -0.1736 between-liquid. Drawing
    the reference at zero, as three figures and one notebook used to, credits every
    model with beating a baseline that was never achievable and makes a set of
    configurations that are merely ordinary look like they clear the bar.

    Read from the ``baseline_mean`` rows rather than computed from the identity on
    purpose: the value is a function of the liquid count and will move the moment a
    liquid is added, and a constant in the plotting code would not.
    """
    values = scores.loc[scores['model'] == BASELINE, column]
    if values.empty:
        raise ValueError(f'no {BASELINE} rows in the scores table; cannot place the '
                         'baseline reference')

    return float(values.median())

def benjamini_hochberg(p_values: "np.ndarray | list[float]") -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), in the input order.

    The permutation null is run once per feature set, so the ~29 p-values it produces
    are ~29 chances for one of them to look significant. Reporting them raw is exactly
    the multiplicity this project is otherwise careful about -- it is the same winner's
    curse that makes the minimum RMSE over five models a biased headline, just moved
    from the model grid to the feature-set grid.

    BH controls the false discovery rate rather than the family-wise error rate,
    which is the right trade here: the question is "which of these sets are worth
    following up", not "is any single one of them beyond doubt".
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p

    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downward, so a q-value can never come
    # out below one belonging to a smaller p.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]

    q = np.empty(n)
    q[order] = np.clip(ranked, 0.0, 1.0)

    return q

def bootstrap_between_r2(predictions: DataFrame, feature_set: str, model: str,
                         include_params: bool = False, n_boot: int = 2000,
                         random_state: int = RANDOM_STATE) -> dict[str, float]:
    """Percentile CI for between-liquid R-squared, resampling *liquids*.

    The resampling unit is the liquid, not the row: thirteen liquids is the real
    sample size, and bootstrapping rows would report the precision of a
    104-observation study that does not exist.
    """
    view = predictions[(predictions['feature_set'] == feature_set)
                       & (predictions['model'] == model)
                       & (predictions['include_params'] == include_params)]
    if view.empty:
        raise ValueError(f'no predictions for {feature_set!r} / {model!r}')

    liquid = view.groupby('liquid')[['y_true', 'y_pred']].mean()
    rng = np.random.default_rng(random_state)
    names = liquid.index.to_numpy()

    draws: list[float] = []
    for _ in range(n_boot):
        picked = liquid.loc[rng.choice(names, size=len(names), replace=True)]
        if picked['y_true'].nunique() < 2:
            continue
        draws.append(float(r2_score(picked['y_true'], picked['y_pred'])))

    values = np.asarray(draws)

    return {
        'feature_set': feature_set,
        'model': model,
        'observed': float(r2_score(liquid['y_true'], liquid['y_pred'])),
        'ci_lo': float(np.percentile(values, 2.5)),
        'ci_hi': float(np.percentile(values, 97.5)),
        'frac_above_zero': float((values > 0).mean()),
        'n_liquids': int(len(names)),
    }

def bootstrap_pooled_rmse(predictions: DataFrame, feature_set: str, model: str,
                          include_params: bool = False, n_boot: int = 2000,
                          random_state: int = RANDOM_STATE) -> dict[str, float]:
    """Percentile CI for pooled RMSE, resampling *liquids*.

    The companion to ``bootstrap_between_r2``, and it exists because the RMSE figure
    needs an RMSE interval: an R-squared interval drawn on an RMSE bar would be a
    different quantity wearing the same error bar.

    Same resampling unit for the same reason -- thirteen liquids is the real sample
    size. A liquid is drawn with all eight of its rows attached rather than as a
    mean, because RMSE is computed over rows and the within-liquid spread is part of
    what it is charging the model for.
    """
    view = predictions[(predictions['feature_set'] == feature_set)
                       & (predictions['model'] == model)
                       & (predictions['include_params'].astype(bool) == include_params)]
    if view.empty:
        raise ValueError(f'no predictions for {feature_set!r} / {model!r}')

    by_liquid = {name: group[['y_true', 'y_pred']].to_numpy()
                 for name, group in view.groupby('liquid')}
    names = np.array(list(by_liquid))
    rng = np.random.default_rng(random_state)

    draws: list[float] = []
    for _ in range(n_boot):
        stacked = np.concatenate([by_liquid[n] for n in rng.choice(names, size=len(names),
                                                                   replace=True)])
        draws.append(float(root_mean_squared_error(stacked[:, 0], stacked[:, 1])))

    values = np.asarray(draws)

    return {
        'rmse_observed': float(root_mean_squared_error(view['y_true'], view['y_pred'])),
        'rmse_ci_lo': float(np.percentile(values, 2.5)),
        'rmse_ci_hi': float(np.percentile(values, 97.5)),
    }
