"""Elastic Net refinement of any feature representation (plan, Phases 10-12).

The last stage before the nested-CV orchestrator. Elastic Net accepts all five feature
families -- manual raw, mRMR raw, global PCA, automatic cluster-PC1, manual cluster-PC1 --
and is asked to say which inputs earn their place. Rule 4 of the plan is the constraint
that shapes the whole module: **it is never forced to reduce.** 12 -> 12 is as legal an
outcome as 12 -> 3, and `test_elastic_net_may_retain_every_feature` exists because the
tempting bug is a well-meaning "if nothing was dropped, fall back to a sparser alpha".

## One scoring path, not two

Everything here scores through ``optimization.cross_validate``. That is deliberate: the
argument in ``config.SELECTION_CRITERION`` -- that 56.5% of the target's variance is
within-liquid, unreachable by any liquid-level feature, and enters every candidate's
pooled error identically as noise -- applies to Elastic Net exactly as it applied to the
clustering sweep. A second scoring path would be a second place for that argument to be
forgotten. The scaler is ``RobustScaler``, matching Phase 8 rather than the plan's
literal "StandardScaler", so that a number from this module and a number from the
clustering search remain comparable; both are fitted inside the fold either way.

## Resampling is by LIQUID, never by row

The single most dangerous line in Phases 11-12. Each liquid contributes eight rows, one
per pipetting volume, and all eight share one descriptor vector. Bootstrapping the 104
rows would redraw the same liquid's rows repeatedly and report an effective n of 104
where the real n is 13; selection frequencies would come out sharply peaked and mean
nothing at all. ``_resample_liquids`` draws liquid *names* and carries all their rows.

## Phase 12 is where the leak hides

Computing a selection frequency over all 13 liquids, thresholding on it, and then
scoring the retained set by leave-one-liquid-out over those same 13 means every held-out
liquid helped choose the features it is then used to test. That is precisely the defect
``regression.evaluate_nested`` was written to correct in the legacy pipeline, and it
produces the same signature -- nested scores coming out *worse* than the leaky ones.
``threshold_search`` computes both and labels them, following the ``selection_leakage``
convention ``regression.pooled_scores`` already established.

## What to expect, recorded before running it

Phase 8 found 16 of 430 clustering candidates beating the mean baseline, the best by 1.13
standard errors. There is no reason to expect Elastic Net to find more signal in the same
13 liquids. The likely result is a flat error surface -- everything from "retain all at
tiny alpha" to "collapse to the intercept at large alpha" inside one standard error, with
the tie-break doing the choosing again, exactly as Phase 8 diagnosed for pooled RMSE. If
that is what comes out, it is the finding and it gets reported as one. This paragraph is
here to make narrowing the alpha grid until a minimum appears feel like what it would be.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from allmeta.feature_selection import config, optimization
from allmeta.feature_selection.stability import StabilityConfig
from allmeta.paths import RANDOM_STATE

# Every fit is wrapped in this so a coefficient path that has not converged is visible
# rather than silently reported as a sparsity result.
MAX_ITER: int = 50_000


def alphas() -> np.ndarray:
    """The alpha grid, logarithmic.

    Logarithmic because the interesting behaviour -- the transition from "penalty does
    nothing" to "everything is zero" -- is spread over orders of magnitude, and a linear
    grid would spend nearly every point in the region where nothing is penalised.
    """
    return np.logspace(np.log10(config.ELASTIC_NET_ALPHA_MIN),
                       np.log10(config.ELASTIC_NET_ALPHA_MAX),
                       config.ELASTIC_NET_N_ALPHAS)


def model(alpha: float, l1_ratio: float, random_state: int = RANDOM_STATE):
    """One Elastic Net, configured identically everywhere in this module."""
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=MAX_ITER,
                      random_state=random_state, selection='cyclic')


# ---------------------------------------------------------------------------
# Phase 10: tuning
# ---------------------------------------------------------------------------

def tune(values: DataFrame, target: Series, groups: Series,
         feature_set: str = 'features') -> DataFrame:
    """Score every (alpha, l1_ratio) by leave-one-liquid-out CV.

    Returns one row per candidate carrying both losses with their standard errors, the
    mean number of non-zero coefficients across folds, and the retained fraction. The
    fold-to-fold range of the non-zero count is kept too, because Elastic Net drops a
    different set in each fold and a single number would misrepresent that.

    The baseline row is appended under ``optimization.BASELINE_NAME`` so the same
    "did this beat predicting the mean?" question can be asked here as in Phase 8, and
    so ``one_standard_error``'s guard has something to read.
    """
    rows: list[dict] = []

    for l1_ratio in config.ELASTIC_NET_L1_RATIOS:
        for alpha in alphas():
            scores = optimization.cross_validate(
                values, target, groups, model=model(alpha, l1_ratio))
            rows.append({
                'feature_set': feature_set,
                'method': 'elastic_net',
                'hyperparameters': f'alpha={alpha:.6g}, l1_ratio={l1_ratio:g}',
                'alpha': float(alpha),
                'l1_ratio': float(l1_ratio),
                **scores,
            })

    baseline = optimization.cross_validate(
        values, target, groups, model=optimization.baseline_model())
    rows.append({'feature_set': feature_set, 'method': optimization.BASELINE_NAME,
                 'hyperparameters': '', 'alpha': np.nan, 'l1_ratio': np.nan,
                 **baseline})

    table = DataFrame(rows)
    criterion = f'{config.SELECTION_CRITERION}_mean'
    reference = float(table.loc[table['method'] == optimization.BASELINE_NAME,
                                criterion].iloc[0])
    table['selection_criterion'] = config.SELECTION_CRITERION
    table['baseline_criterion_mean'] = reference
    table['beats_baseline'] = table[criterion] < reference

    return table.sort_values(criterion, ignore_index=True)


def select(table: DataFrame, stability: DataFrame | None = None,
           tie_break=None) -> tuple[DataFrame, Series]:
    """One-standard-error selection over the Elastic Net grid.

    The plan's Step 10.3 order: fewer non-zero coefficients, then higher selection
    stability, then more consistent coefficient signs, then stronger regularisation
    where everything else ties. The last of those is why ``alpha`` sorts *descending* --
    among genuinely equivalent candidates the more regularised one is preferred, which
    is the opposite direction from every other numeric tie-break in this project.

    Selection stability and sign consistency come from Phase 11, so ``stability`` has to
    be computed first and joined here. Without it the ordering silently degrades to
    "fewest non-zero, then strongest penalty", and ``tie_break_applied`` in the returned
    frame records which columns were actually available.
    """
    tie_break = config.ELASTIC_NET_TIE_BREAK if tie_break is None else tie_break

    context = None
    if stability is not None and not stability.empty:
        context = _aggregate_stability(stability)

    return optimization.one_standard_error(table, context=context,
                                           tie_break=tie_break,
                                           interpretability=())


def coefficients(values: DataFrame, target: Series, alpha: float,
                 l1_ratio: float) -> DataFrame:
    """Refit on everything given and report the coefficient of each input.

    Used for the final refit inside a fold, and for the coefficient-ranking figure.
    Fitting on all rows handed to it is correct *only* when the caller has already
    restricted them to a training fold, which is why this takes no groups and does no
    CV -- there is nothing here to mistake for an evaluation.
    """
    pipeline = Pipeline([('scale', RobustScaler()),
                         ('model', model(alpha, l1_ratio))])
    pipeline.fit(values, target)
    fitted = np.ravel(pipeline.named_steps['model'].coef_)

    return DataFrame({
        'feature': list(values.columns),
        'coefficient': fitted,
        'abs_coefficient': np.abs(fitted),
        'selected': fitted != 0.0,
        'alpha': alpha,
        'l1_ratio': l1_ratio,
    }).sort_values('abs_coefficient', ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# Phase 11: selection stability
# ---------------------------------------------------------------------------

@dataclass
class Resample:
    """One resampled training set, addressed by liquid rather than by row."""

    index: int
    liquids: tuple[str, ...]
    mask: np.ndarray


def _resample_liquids(groups: Series, settings: StabilityConfig) -> list[Resample]:
    """Draw subsamples of the LIQUIDS, carrying all rows of each drawn liquid.

    The eight volumes of a liquid share one descriptor vector. Resampling rows would
    draw the same vector up to eight times and report an effective n of 104 where the
    real n is 13, which would make every selection frequency look far more certain than
    it is. Sampling without replacement, so a liquid cannot appear twice and inflate its
    own weight.
    """
    liquids = np.asarray(list(dict.fromkeys(groups)))
    n_draw = max(2, int(round(len(liquids) * settings.subsample_fraction)))
    rng = np.random.default_rng(settings.random_state)

    draws: list[Resample] = []
    for index in range(settings.n_resamples):
        drawn = rng.choice(liquids, size=n_draw, replace=False)
        draws.append(Resample(index=index, liquids=tuple(drawn),
                              mask=groups.isin(drawn).to_numpy()))

    return draws


def stability(values: DataFrame, target: Series, groups: Series,
              alpha: float, l1_ratio: float,
              settings: StabilityConfig | None = None,
              feature_set: str = 'features') -> DataFrame:
    """How often each input survives, across resampled training sets.

    ``S_j = N_selected,j / N_fits``, plus the coefficient's mean, median, standard
    deviation, mean absolute value and sign consistency.

    Every row carries ``n_fits`` and the resampling settings. At 13 liquids a frequency
    is a proportion out of a few dozen fits over a pool of 13, and a table that does not
    say so invites a 0.85 from ten resamples to be read against a 0.85 from thirty.
    """
    settings = (StabilityConfig.preset('full') if settings is None else settings)
    draws = _resample_liquids(groups, settings)

    fits: list[np.ndarray] = []
    for draw in draws:
        subset = values[draw.mask]
        # A resample that lost a column's variance entirely cannot say anything about
        # it; skipping the fit is better than recording a zero coefficient as evidence
        # the feature was rejected.
        if subset.empty or subset.shape[0] < 3:
            continue
        fitted = coefficients(subset, target[draw.mask], alpha, l1_ratio)
        fits.append(fitted.set_index('feature')['coefficient']
                    .reindex(values.columns).to_numpy())

    if not fits:
        return DataFrame()

    matrix = np.vstack(fits)
    n_fits = matrix.shape[0]
    selected = matrix != 0.0

    positive = (matrix > 0).sum(axis=0)
    negative = (matrix < 0).sum(axis=0)
    non_zero = positive + negative
    with np.errstate(invalid='ignore', divide='ignore'):
        # Fraction of the NON-ZERO fits that agree on the sign. A feature selected
        # twice with opposite signs is not a stable finding, and averaging its
        # coefficient to about zero would hide that.
        consistency = np.where(non_zero > 0,
                               np.maximum(positive, negative) / non_zero, np.nan)

    frame = DataFrame({
        'feature_set': feature_set,
        'feature': list(values.columns),
        'selection_frequency': selected.mean(axis=0),
        'n_selected': selected.sum(axis=0),
        'n_fits': n_fits,
        'mean_coefficient': matrix.mean(axis=0),
        'median_coefficient': np.median(matrix, axis=0),
        'std_coefficient': matrix.std(axis=0, ddof=1) if n_fits > 1 else 0.0,
        'mean_abs_coefficient': np.abs(matrix).mean(axis=0),
        'sign_consistency': consistency,
        'alpha': alpha,
        'l1_ratio': l1_ratio,
    })

    for key, value in settings.describe().items():
        frame[key] = value
    frame['n_liquids_per_resample'] = len(draws[0].liquids) if draws else 0
    frame['resampled_by'] = 'liquid'

    return frame.sort_values('selection_frequency', ascending=False,
                             ignore_index=True)


def _aggregate_stability(frame: DataFrame) -> DataFrame:
    """Per-candidate stability summary, for the Step 10.3 tie-break."""
    if frame.empty:
        return DataFrame()

    grouped = (frame.groupby(['alpha', 'l1_ratio'], dropna=False)
               .agg(mean_selection_frequency=('selection_frequency', 'mean'),
                    mean_sign_consistency=('sign_consistency', 'mean'))
               .reset_index())
    grouped['method'] = 'elastic_net'
    grouped['hyperparameters'] = [
        f'alpha={a:.6g}, l1_ratio={r:g}'
        for a, r in zip(grouped['alpha'], grouped['l1_ratio'])]

    return grouped.drop(columns=['alpha', 'l1_ratio'])


# ---------------------------------------------------------------------------
# Phase 12: stability-threshold evaluation
# ---------------------------------------------------------------------------

def threshold_search(values: DataFrame, target: Series, groups: Series,
                     alpha: float, l1_ratio: float,
                     settings: StabilityConfig | None = None,
                     thresholds=None, feature_set: str = 'features') -> DataFrame:
    """Retain features at each stability threshold and score what is left.

    Two readings are produced for every threshold and both are written, because only one
    of them is honest and the difference between them is the measurement:

    ``nested`` (``selection_leakage=False``)
        Selection frequencies recomputed inside each outer training fold, so the
        held-out liquid contributed to neither the frequencies nor the fit. Its own
        threshold set may differ fold to fold, which is why ``n_features_min/max`` are
        reported rather than one count.

    ``diagnostic`` (``selection_leakage=True``)
        Frequencies computed once over all 13 liquids, then thresholded and scored by
        leave-one-liquid-out on those same 13. Every held-out liquid helped choose the
        features it is used to test. Kept because comparing the two is the clearest
        statement of what that optimism is worth here -- expect the nested scores to
        come out *worse*, and treat it as a bug if they do not.

    A threshold that retains nothing records NA scores rather than crashing or quietly
    falling back to a lower threshold. A threshold that retains everything is a
    legitimate outcome, not a failure (Rule 4).
    """
    settings = (StabilityConfig.preset('full') if settings is None else settings)
    thresholds = (config.ELASTIC_NET_STABILITY_THRESHOLDS if thresholds is None
                  else thresholds)

    rows: list[dict] = []
    common = {'feature_set': feature_set, 'alpha': alpha, 'l1_ratio': l1_ratio,
              'n_inputs': int(values.shape[1])}

    # --- diagnostic: frequencies from every liquid --------------------------------
    whole = stability(values, target, groups, alpha, l1_ratio, settings=settings,
                      feature_set=feature_set)
    frequency = (whole.set_index('feature')['selection_frequency']
                 if not whole.empty else Series(dtype=float))

    for threshold in thresholds:
        keep = [f for f in values.columns if frequency.get(f, 0.0) >= threshold]
        record = {**common, 'scope': 'diagnostic', 'selection_leakage': True,
                  'threshold': float(threshold), 'n_features': len(keep),
                  'features': ', '.join(keep)}
        if keep:
            record.update(optimization.cross_validate(
                values[keep], target, groups, model=optimization.ridge_model()))
        else:
            record['retained_nothing'] = True
        rows.append(record)

    # --- nested: frequencies recomputed inside each training fold -----------------
    #
    # The frequencies do NOT depend on the threshold, so they are computed once per
    # fold and reused across all six. Recomputing them inside the threshold loop would
    # cost six times as many Elastic Net fits for identical numbers.
    liquids = list(dict.fromkeys(groups))
    fold_frequency: dict[str, Series] = {}
    for held_out in liquids:
        train = (groups != held_out).to_numpy()
        inner = stability(values[train].reset_index(drop=True),
                          target[train].reset_index(drop=True),
                          groups[train].reset_index(drop=True),
                          alpha, l1_ratio, settings=settings,
                          feature_set=feature_set)
        if not inner.empty:
            fold_frequency[held_out] = inner.set_index('feature')['selection_frequency']

    for threshold in thresholds:
        predictions: list[dict] = []
        sizes: list[int] = []

        for fold, held_out in enumerate(liquids):
            frequencies = fold_frequency.get(held_out)
            if frequencies is None:
                continue
            train = (groups != held_out).to_numpy()
            keep = [f for f in values.columns
                    if frequencies.get(f, 0.0) >= threshold]
            sizes.append(len(keep))
            if not keep:
                continue

            pipeline = Pipeline([('scale', RobustScaler()),
                                 ('model', clone(optimization.ridge_model()))])
            pipeline.fit(values.loc[train, keep], target[train])
            predicted = pipeline.predict(values.loc[~train, keep])
            actual = target[~train].to_numpy()
            predictions.append({
                'fold': fold, 'held_out_liquid': held_out,
                'liquid_squared_error': float((predicted.mean() - actual.mean()) ** 2),
                'rmse': float(np.sqrt(np.mean((actual - predicted) ** 2))),
                'mae': float(np.mean(np.abs(actual - predicted))),
            })

        record = {**common, 'scope': 'nested', 'selection_leakage': False,
                  'threshold': float(threshold),
                  'n_features_median': float(np.median(sizes)) if sizes else np.nan,
                  'n_features_min': int(min(sizes)) if sizes else 0,
                  'n_features_max': int(max(sizes)) if sizes else 0,
                  'n_folds_scored': len(predictions)}
        if predictions:
            fold_frame = DataFrame(predictions)
            record.update({
                'liquid_mse_mean': float(fold_frame['liquid_squared_error'].mean()),
                'liquid_mse_se': float(fold_frame['liquid_squared_error'].sem())
                if len(fold_frame) > 1 else 0.0,
                'pooled_rmse_mean': float(fold_frame['rmse'].mean()),
                'mae_mean': float(fold_frame['mae'].mean()),
            })
        else:
            record['retained_nothing'] = True
        rows.append(record)

    frame = DataFrame(rows)
    for key, value in settings.describe().items():
        frame[key] = value

    return frame


# ---------------------------------------------------------------------------
# The five feature families, assembled on the labelled liquids
# ---------------------------------------------------------------------------

def labelled_representations(fit_frame: DataFrame, labelled: DataFrame,
                             artefacts: dict, carried=None,
                             solutions=None) -> dict[str, DataFrame]:
    """Every feature family, carried onto the labelled liquids as one dict.

    All five come back in the same shape -- name to value matrix -- so Elastic Net does
    not need to know which family it was handed. That invariant is what keeps Phase 14
    from growing a branch per feature type, and
    ``test_all_five_families_have_the_same_shape`` guards it.

    Everything here is *applied*, never fitted: the loadings come from the subset-B fit
    and ``pca.apply_loadings`` only evaluates persisted constants. The labels are spent
    on evaluation alone.

    ``mrmr_raw`` is included only when a leakage-safe selection is supplied by the
    caller. The descriptive ``run_mrmr`` fit over all 13 labelled liquids is not
    admissible here, and silently substituting it would put a selection that saw every
    label into a table of things being scored on those labels.
    """
    from allmeta.feature_selection import pca

    families: dict[str, DataFrame] = {}

    manual_features = [c for c in config.MANUAL_FEATURES if c in labelled.columns]
    if manual_features:
        families['manual_raw'] = labelled[manual_features].reset_index(drop=True)

    if 'pca_loadings' in artefacts:
        families['global_pca'] = pca.apply_loadings(labelled,
                                                    artefacts['pca_loadings'])
    if 'cluster_loadings' in artefacts:
        families['manual_cluster_pc1'] = pca.apply_loadings(
            labelled, artefacts['cluster_loadings'])

    if carried is not None and not carried.empty and solutions is not None:
        from allmeta.feature_selection import factors
        from allmeta.feature_selection.algorithms import format_params

        cache = factors.ClusterPCACache(fit_frame)
        by_key = {(s.method, format_params(s.params)): s for s in solutions}
        for record in carried.to_dict('records'):
            key = (record.get('clustering_method') or record.get('method'),
                   record['hyperparameters'])
            solution = by_key.get(key)
            if solution is None:
                continue
            factor_set = factors.build_factors(
                fit_frame, solution, artefacts.get('correlation'), cache)
            families[record['feature_set_name']] = pca.apply_loadings(
                labelled, factor_set.loadings)

    return families


def refine_all(families: dict[str, DataFrame], target: Series, groups: Series,
               settings: StabilityConfig | None = None,
               with_thresholds: bool = True
               ) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Tune, measure stability and search thresholds for every feature family.

    Returns ``(search, selected, stability_table, threshold_table)``.

    Each family is refined independently and none of them competes with another here.
    Comparing the families against each other is Phase 19's job and needs the outer-fold
    numbers of Phase 14; what this produces is one tuned Elastic Net per family, with
    the evidence behind each choice.
    """
    settings = (StabilityConfig.preset('full') if settings is None else settings)

    searches: list[DataFrame] = []
    selections: list[dict] = []
    stabilities: list[DataFrame] = []
    thresholds: list[DataFrame] = []

    for name, values in families.items():
        if values.empty or values.shape[1] == 0:
            continue

        table = tune(values, target, groups, feature_set=name)

        # Stability is computed at the *unpenalised-selection* optimum first, because
        # the Step 10.3 tie-break needs it before the final choice is made. Using the
        # provisional best rather than the final one is a deliberate ordering: the
        # alternative is circular.
        candidates = table[table['method'] != optimization.BASELINE_NAME]
        criterion = f'{config.SELECTION_CRITERION}_mean'
        provisional = candidates.loc[candidates[criterion].idxmin()]

        frame = stability(values, target, groups, float(provisional['alpha']),
                          float(provisional['l1_ratio']), settings=settings,
                          feature_set=name)
        stabilities.append(frame)

        try:
            competitive, chosen = select(table, stability=frame)
        except ValueError as error:
            # No candidate beat the mean baseline. On this data that is a real and
            # likely outcome, and recording it beats aborting every other family.
            selections.append({'feature_set': name, 'selected': False,
                               'reason': str(error).split(';')[0],
                               'n_inputs': int(values.shape[1])})
            searches.append(table)
            continue

        selections.append({
            'feature_set': name,
            'selected': True,
            'reason': '',
            'n_inputs': int(values.shape[1]),
            'alpha': float(chosen['alpha']),
            'l1_ratio': float(chosen['l1_ratio']),
            'n_nonzero_mean': float(chosen['n_nonzero_mean']),
            'retained_fraction': float(chosen['retained_fraction']),
            'retained_all': bool(chosen.get('retained_all', False)),
            'criterion_mean': float(chosen[criterion]),
            'criterion_se': float(chosen[f'{config.SELECTION_CRITERION}_se']),
            'between_liquid_r2': float(chosen['between_liquid_r2']),
            'baseline_criterion_mean': float(chosen['baseline_criterion_mean'])
            if 'baseline_criterion_mean' in chosen else np.nan,
            'n_competitive': int(len(competitive)),
            'n_candidates': int(len(candidates)),
            'tie_break_applied': chosen.get('tie_break_applied', ''),
        })
        searches.append(table)

        if with_thresholds and config.OPTIMIZE_ELASTIC_NET_STABILITY_THRESHOLD:
            thresholds.append(threshold_search(
                values, target, groups, float(chosen['alpha']),
                float(chosen['l1_ratio']), settings=settings, feature_set=name))

    return (pd.concat(searches, ignore_index=True) if searches else DataFrame(),
            DataFrame(selections),
            pd.concat(stabilities, ignore_index=True) if stabilities else DataFrame(),
            pd.concat(thresholds, ignore_index=True) if thresholds else DataFrame())
