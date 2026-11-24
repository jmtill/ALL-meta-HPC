"""Supervised feature selection against measured dispensing error (subset C).

Only the benchmarked liquids carry a label, and the eight pipetting volumes per
liquid multiply the rows without adding information: the descriptor vector is
identical across a liquid's eight rows, so the effective sample size for any
liquid-level feature is the *liquid* count, not the row count. Everything here is
built around that fact:

* splits are ``LeaveOneGroupOut`` on the liquid, never on the row, so a liquid never
  appears in both train and test;
* the deliverable is a *selection frequency* -- how often each feature is chosen
  across the folds -- rather than one feature list, because at this n a single
  fit's choice is close to arbitrary;
* selection runs twice, with and without the pipetting parameters. Volume, rate and
  airgap are the only features that vary within a liquid, so they carry almost all
  the within-group signal and will otherwise crowd out every descriptor.

The earlier notebook version of this took ``X`` from the HSPiP frame and ``y`` from
the benchmark frame with no join between them; here they arrive already joined in
subset C and the alignment is asserted.
"""

import warnings
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    RFE, RFECV, SelectFromModel, SequentialFeatureSelector, VarianceThreshold)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

from feature_engine.selection import MRMR

from allmeta import columns as cols
from allmeta.paths import FEATURE_SELECTION_DIR, RANDOM_STATE
from allmeta.screening import attach_derived
from allmeta.subsets import load_subset

# How many features each selector is asked for. Deliberately small: with only a
# handful of liquids in each training fold, a larger request would be fitting noise,
# and holding every method to the same count keeps their outputs comparable.
N_SELECT: int = 5

# How often a feature must be selected to count as chosen. A FRACTION of the folds,
# not a count of them, so it holds its stringency as liquids are added -- it started
# life as the literal 5-of-7 and stays at that ratio (~0.714), so the required fold
# count is ceil(MIN_FREQUENCY * n_folds) and moves with the data. Anything comparing
# against a selection frequency should use this rather than restating a ratio.
MIN_FREQUENCY: float = 5 / 7

# How many distinct selectors must independently choose a feature before it counts
# as consensus. Fold-frequency alone measures how deterministic one method is given
# its own inductive bias; only cross-method agreement is evidence about the data.
# Three of thirteen is deliberately lenient -- the methods are not independent (four
# share a Ridge base estimator, five share the MRMR framework) -- but it is enough
# to exclude a feature that exactly one selector likes.
MIN_METHODS: int = 3

# MRMR variants, as named in feature_engine. They differ in how relevance and
# redundancy are estimated and combined:
#   MID/MIQ  mutual information for both, combined by difference / ratio
#   FCD/FCQ  F-statistic relevance against correlation redundancy, difference / ratio
#   RFCQ     random-forest importance relevance against correlation redundancy
# All five are run because the choice of estimator, not just the framework, is what
# decides the answer -- MI-based and F-based variants disagree sharply here.
MRMR_METHODS: tuple[str, ...] = ('MID', 'MIQ', 'FCD', 'FCQ', 'RFCQ')

# Inner-CV fold count for the wrapper selectors. The splits themselves are built
# per outer fold and grouped by liquid -- see ``inner_splits``.
INNER_FOLDS: int = 3

# Methods that choose their own cardinality instead of returning ``N_SELECT``
# features. Their churn is not comparable with the fixed-size methods: a method
# free to return thirty features will touch more distinct features than one capped
# at five, whatever its stability.
FREE_CARDINALITY: frozenset[str] = frozenset({'wrapper_rfecv'})

# Draws used to simulate the churn null. Cheap -- each draw is n_folds samples of a
# few integers -- and large enough that the empirical p is stable in the third
# decimal, which matters because the interesting values sit near 0.05.
CHURN_NULL_DRAWS: int = 20_000

FREQUENCY_PATH: Path = FEATURE_SELECTION_DIR / 'C' / 'selection_frequency.csv'
FOLD_PATH: Path = FEATURE_SELECTION_DIR / 'C' / 'fold_selections.csv'
RANKING_PATH: Path = FEATURE_SELECTION_DIR / 'C' / 'model_rankings.csv'
STABILITY_PATH: Path = FEATURE_SELECTION_DIR / 'C' / 'selection_stability.csv'


def build_labeled_table(include_params: bool = True
                        ) -> tuple[DataFrame, Series, Series]:
    """Feature matrix, target and liquid grouping from subset C.

    Returns ``(X, y, groups)`` sharing one index, so a downstream split cannot
    silently misalign them.

    The candidate pool is ``columns.supervised_pool()`` -- the same descriptors the
    unsupervised screens see, with the three raw Antoine coefficients replaced by the
    ``logPvap25`` they jointly imply. It used to be all 58 of ``NUM_COLUMNS``, which
    made the two tracks incomparable in both directions at once: the atom counts were
    on offer here and nowhere else, so ``Cl#``, ``N#`` and ``O#`` cleared the
    fold-frequency threshold in descriptors that could never appear in an A:: or B::
    set, while ``AntC`` was selectable as though it meant something on its own. See
    ``columns.supervised_pool``.
    """
    frame = attach_derived(load_subset('C'))

    features = cols.supervised_pool()
    if include_params:
        features = features + list(cols.PIPETTING_PARAMS)

    data = frame[features].copy()
    target = frame[cols.TARGET].copy()
    groups = frame['Name'].copy()

    if not (data.index.equals(target.index) and data.index.equals(groups.index)):
        raise ValueError('X, y and groups are not aligned')
    if data.isna().any().any() or target.isna().any():
        raise ValueError('subset C contains missing features or targets')

    return data, target, groups


def inner_splits(data: DataFrame, target: Series, groups: Series,
                 n_splits: int = INNER_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped inner-CV splits for the wrapper selectors, as explicit index pairs.

    The wrapper selectors score candidate subsets by cross-validating *inside* the
    training fold. Their default ``cv=3`` is an ungrouped ``KFold`` over rows, and a
    liquid's eight volume rows carry an identical descriptor vector -- so the same
    liquid lands on both sides of an inner split and the score that decides which
    features get selected is optimistic.

    scikit-learn's ``cv`` accepts a precomputed iterable of ``(train, test)`` index
    pairs, which is the only route here: ``SequentialFeatureSelector`` and ``RFECV``
    call ``cross_val_score`` without forwarding a ``groups`` argument, so a bare
    ``GroupKFold`` instance would raise. Materialising the splits sidesteps that.

    Indices are positional within the training fold, which is what the selectors
    index into.
    """
    n_groups = groups.nunique()
    if n_groups < 2:
        raise ValueError(f'need at least 2 liquids for inner CV, got {n_groups}')

    return list(
        GroupKFold(n_splits=min(n_splits, n_groups)).split(data, target, groups=groups)
    )


def selectors(inner_cv: list | int = INNER_FOLDS) -> dict[str, object]:
    """Wrapper and embedded selectors, each asked for exactly ``N_SELECT`` features.

    ``Ridge`` is the wrapper base estimator: it is stable at this n-to-column ratio,
    exposes ``coef_`` for RFE to rank on, and is cheap enough that forward and
    backward selection over 63 features stays tractable. ``threshold=-inf`` forces
    the embedded selectors to return exactly ``max_features`` rather than however
    many clear their own mean-importance default, so all methods are comparable.

    ``inner_cv`` should be the grouped splits from ``inner_splits`` for the training
    fold in hand. The bare integer default is only for callers that want to inspect
    the selector line-up without a fold; it reproduces the ungrouped behaviour and
    should not be used for selection.

    ``wrapper_rfecv`` is deliberately left free to choose its own cardinality --
    that is the question it exists to answer -- so it is listed in
    ``FREE_CARDINALITY`` and its churn is reported but not compared.
    """
    # random_state omitted: inert for Ridge's default solver (see
    # regression.models), so passing one only implies a guarantee it does not give.
    ridge = Ridge(alpha=1.0)
    forest = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    boosted = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                           random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)

    return {
        'wrapper_sfs_forward': SequentialFeatureSelector(
            ridge, n_features_to_select=N_SELECT, direction='forward',
            cv=inner_cv, n_jobs=-1),
        'wrapper_sfs_backward': SequentialFeatureSelector(
            ridge, n_features_to_select=N_SELECT, direction='backward',
            cv=inner_cv, n_jobs=-1),
        'wrapper_rfe': RFE(ridge, n_features_to_select=N_SELECT),
        'wrapper_rfecv': RFECV(ridge, min_features_to_select=1, cv=inner_cv, n_jobs=-1),
        'embedded_lasso': SelectFromModel(
            Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=10000),
            max_features=N_SELECT, threshold=-np.inf),
        'embedded_elasticnet': SelectFromModel(
            ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=10000),
            max_features=N_SELECT, threshold=-np.inf),
        'embedded_random_forest': SelectFromModel(
            forest, max_features=N_SELECT, threshold=-np.inf),
        'embedded_xgboost': SelectFromModel(
            boosted, max_features=N_SELECT, threshold=-np.inf),
        **mrmr_selectors(inner_cv=inner_cv),
    }


def mrmr_selectors(max_features: int = N_SELECT,
                   inner_cv: list | int = INNER_FOLDS) -> dict[str, MRMR]:
    """The five feature_engine MRMR variants, asked for ``max_features`` features.

    MRMR sits between the two halves of this project: the unsupervised screens drop
    redundant descriptors without consulting the target, and the wrapper/embedded
    methods chase relevance without penalising redundancy. MRMR optimises both at
    once, so it is the natural third comparison.

    ``regression`` must be passed explicitly -- the class docstring claims it
    defaults to True but the signature default is False, which would silently treat
    a continuous target as class labels.
    """
    selectors_by_method: dict[str, MRMR] = {}
    for method in MRMR_METHODS:
        options: dict = {
            'method': method,
            'max_features': max_features,
            'regression': True,
            'random_state': RANDOM_STATE,
        }
        if method == 'RFCQ':
            # scoring/cv are only consulted by RFCQ, whose default 'roc_auc' is a
            # classification metric and errors on a continuous target. ``cv`` gets
            # the same grouped splits as the wrapper selectors.
            options.update(scoring='neg_root_mean_squared_error', cv=inner_cv)
        selectors_by_method[f'mrmr_{method.lower()}'] = MRMR(**options)

    return selectors_by_method


def _selected_names(pipeline: Pipeline, features: list[str]) -> list[str]:
    """Feature names a fitted pipeline kept.

    The support mask indexes the columns that reached the selector, which is fewer
    than ``features`` once the constant-feature guard has run, so the surviving
    names are read back from that step rather than assumed.
    """
    surviving = list(pipeline.named_steps['drop_constant'].get_feature_names_out())
    support = pipeline.named_steps['select'].get_support()

    return [name for name, keep in zip(surviving, support) if keep]


def run_selection(include_params: bool = True) -> DataFrame:
    """Fit every selector inside every leave-one-liquid-out fold.

    Returns one row per (method, fold, feature) that was selected.
    """
    data, target, groups = build_labeled_table(include_params=include_params)
    features = list(data.columns)
    splitter = LeaveOneGroupOut()

    rows: list[dict] = []
    # The fold loop is outermost so the grouped inner splits are built once per fold
    # and shared by every selector that needs them. Selectors are constructed inside
    # the fold for the same reason: their ``cv`` is fold-specific, and a selector
    # instance carries fitted state that must not survive into the next fold.
    for fold, (train_idx, test_idx) in enumerate(
            splitter.split(data, target, groups=groups)):
        held_out = sorted(set(groups.iloc[test_idx]))
        train_groups = set(groups.iloc[train_idx])
        if train_groups & set(held_out):
            raise ValueError(f'fold {fold} leaks {held_out} into training')

        train_data = data.iloc[train_idx]
        train_target = target.iloc[train_idx]
        train_group = groups.iloc[train_idx]
        inner_cv = inner_splits(train_data, train_target, train_group)

        for method, selector in selectors(inner_cv).items():
            # Constant features are dropped inside the fold, before anything sees
            # them. They carry no information, but MRMR's ratio variants (MIQ, FCQ,
            # RFCQ) score them *maximally*: a zero-variance column has zero
            # correlation with every other feature, so relevance/redundancy divides
            # by ~0. Without this guard, Br# -- identically zero across all the
            # labelled liquids -- is selected in every fold by three MRMR variants.
            #
            # Fitted per fold rather than once globally so the guard never consults
            # the held-out liquid.
            #
            # feature_engine selectors operate on named columns, so both steps emit
            # a DataFrame. The sklearn selectors accept one too, which keeps a
            # single code path for every method.
            pipeline = Pipeline([
                ('drop_constant', VarianceThreshold(0.0).set_output(transform='pandas')),
                ('scale', RobustScaler().set_output(transform='pandas')),
                ('select', selector),
            ])

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pipeline.fit(train_data, train_target)

            for name in _selected_names(pipeline, features):
                rows.append({
                    'method': method,
                    'fold': fold,
                    'held_out_liquid': held_out[0],
                    'feature': name,
                    'include_params': include_params,
                })

    return pd.DataFrame(rows)


def selection_frequency(folds: DataFrame, n_folds: int) -> DataFrame:
    """Fraction of folds in which each feature was selected, per method.

    The wide form is what a reader actually compares; a feature chosen in every fold
    means something, one chosen in two does not.
    """
    counts = (
        folds.groupby(['include_params', 'method', 'feature'], as_index=False)
        .size().rename(columns={'size': 'n_folds_selected'})
    )
    counts['frequency'] = counts['n_folds_selected'] / n_folds

    return counts.sort_values(
        ['include_params', 'method', 'frequency', 'feature'],
        ascending=[True, True, False, True], ignore_index=True)


def model_rankings(include_params: bool = True) -> DataFrame:
    """Raw coefficient and importance rankings, fit on all the labelled liquids.

    Complements the fold-wise selection: a feature can be ranked highly by every
    model yet fall just outside the top five in most folds, and that is worth being
    able to see.
    """
    data, target, _ = build_labeled_table(include_params=include_params)
    scaled = DataFrame(RobustScaler().fit_transform(data), columns=data.columns)

    estimators = {
        'lasso': Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=10000),
        'elasticnet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE,
                                 max_iter=10000),
        'ridge': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=200,
                                               random_state=RANDOM_STATE, n_jobs=-1),
        'xgboost': XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                                random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
    }

    rows: list[dict] = []
    for name, estimator in estimators.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            estimator.fit(scaled, target)

        importance = (
            np.abs(estimator.coef_) if hasattr(estimator, 'coef_')
            else estimator.feature_importances_
        )
        order = pd.Series(importance, index=data.columns).sort_values(ascending=False)
        for rank, (feature, value) in enumerate(order.items(), start=1):
            rows.append({
                'model': name, 'feature': feature, 'importance': float(value),
                'rank': rank, 'include_params': include_params,
            })

    return pd.DataFrame(rows)


def churn_null(per_fold_sizes: list[int], pool_size: int,
               n_draws: int = CHURN_NULL_DRAWS,
               random_state: int = RANDOM_STATE) -> np.ndarray:
    """Churn values from selecting at random, for one method's fold cardinalities.

    Each draw picks, independently per fold, as many *distinct* features as that
    method actually picked in that fold, uniformly from a pool of ``pool_size``, and
    scores the result with the same formula ``selection_stability`` uses.

    Matching the per-fold cardinalities is what makes the null comparable across
    methods that return different numbers of features. It is also what retires the
    ``FREE_CARDINALITY`` caveat in substance if not in name: ``wrapper_rfecv`` shows
    high churn partly because it picks a variable number of features, and a null that
    picks the same variable number absorbs exactly that effect.
    """
    rng = np.random.default_rng(random_state)
    typical = float(np.median(per_fold_sizes))
    span = float(sum(per_fold_sizes)) - typical
    if span <= 0:
        return np.array([])

    values = np.empty(n_draws)
    for draw in range(n_draws):
        touched: set[int] = set()
        for size in per_fold_sizes:
            touched.update(rng.choice(pool_size, size=min(size, pool_size),
                                      replace=False).tolist())
        values[draw] = (len(touched) - typical) / span

    return values


def candidate_pool_size(include_params: bool = False) -> int:
    """How many features a selector could have chosen from.

    Read from the live pool rather than written down: it was 51 non-constant
    descriptors when the supervised track took all of ``NUM_COLUMNS``, and is 43 now
    that both tracks share ``columns.supervised_pool()``. A constant here would have
    silently gone stale the moment the pool changed -- which is the same defect as
    drawing the regression baseline at zero, one module over.
    """
    pool = len(cols.supervised_pool())

    return pool + len(cols.PIPETTING_PARAMS) if include_params else pool


def selection_stability(folds: DataFrame, frequency: DataFrame, n_folds: int,
                        include_params: bool = False,
                        pool_size: int | None = None) -> DataFrame:
    """How much each method's choice changes when one liquid is held out.

    ``churn`` runs from 0.0 (every fold picked the same features) to 1.0 (every fold
    picked a disjoint set). **Neither endpoint is reachable, and the scale is not the
    one it looks like.** Churn 1.0 across thirteen folds of five picks needs 65
    distinct features and the pool holds ``candidate_pool_size()`` of them, so the
    ceiling is (pool - 5) / 60. More to the point, *random* selection does not score
    near 1.0 either: it scores around 0.5, because thirteen small samples from a
    modest pool overlap by chance. Read on the raw 0-1 scale, a method at 0.48 looks
    roughly mid-range when it is in fact indistinguishable from picking at random.

    So the columns that carry the meaning are:

    * ``churn_random`` -- the median churn of a selector that picks uniformly at
      random with this method's own per-fold cardinalities;
    * ``churn_ratio`` = churn / churn_random -- below 1 is more stable than chance;
    * ``churn_p`` -- the fraction of random draws at or below the observed churn, so
      a small value means the stability is real.

    ``fixed_cardinality`` is retained for continuity, but the null now absorbs the
    effect it was flagging: ``wrapper_rfecv`` is compared against a random selector
    that picks the same variable number of features it did.
    """
    picks = folds[folds['include_params'].astype(bool) == include_params]
    freq = frequency[frequency['include_params'].astype(bool) == include_params]
    if pool_size is None:
        pool_size = candidate_pool_size(include_params)

    rows: list[dict] = []
    for method, group in picks.groupby('method'):
        touched = group['feature'].nunique()
        method_freq = freq[freq['method'] == method]
        # Normalise against what this method actually picked, not against a
        # five-feature yardstick a free-length method never agreed to. The floor is
        # the per-fold cardinality (every fold chose the same features) and the
        # ceiling is the total number of picks (every fold chose a disjoint set), so
        # churn stays in [0, 1] even when the cardinality varies between folds.
        per_fold = group.groupby('fold')['feature'].size()
        typical = float(per_fold.median()) if len(per_fold) else float(N_SELECT)
        span = float(len(group)) - typical
        observed = (touched - typical) / span if span else np.nan

        # The null uses this method's own per-fold cardinalities, so a method that
        # picks a variable number of features is not penalised for that alone.
        null = churn_null(list(per_fold), pool_size)
        if null.size and np.isfinite(observed):
            random_churn = float(np.median(null))
            ratio = observed / random_churn if random_churn else np.nan
            # Fraction of random draws at least as stable as the observed value.
            # +1 in both terms so it is never reported as exactly zero.
            p_value = float((null <= observed).sum() + 1) / (null.size + 1)
        else:
            random_churn = ratio = p_value = np.nan

        rows.append({
            'method': method,
            'include_params': include_params,
            'fixed_cardinality': method not in FREE_CARDINALITY,
            'median_n_selected': typical,
            'n_distinct_features': touched,
            'n_unanimous': int((method_freq['frequency'] == 1.0).sum()),
            'max_frequency': float(method_freq['frequency'].max()) if len(method_freq) else np.nan,
            'churn': observed,
            'churn_random': random_churn,
            'churn_ratio': ratio,
            'churn_p': p_value,
            # Capped at the total number of picks: a method that makes 18
            # picks cannot touch 43 distinct features however unstable it is,
            # so the binding ceiling is whichever of the two is smaller.
            'churn_ceiling': ((min(pool_size, len(group)) - typical) / span
                              if span else np.nan),
            'pool_size': pool_size,
        })

    return DataFrame(rows).sort_values('churn', ignore_index=True)


def cross_method_agreement(frequency: DataFrame,
                           min_frequency: float = MIN_FREQUENCY,
                           include_params: bool = False) -> DataFrame:
    """How many distinct methods independently chose each feature.

    A descriptor picked unanimously by one method is that method's preference; one
    picked by several methods is evidence about the data.
    """
    view = frequency[
        (frequency['include_params'].astype(bool) == include_params)
        & (frequency['frequency'] >= min_frequency)
    ]
    if view.empty:
        return DataFrame(columns=['feature', 'n_methods', 'methods'])

    grouped = view.groupby('feature')['method']

    return (
        DataFrame({'n_methods': grouped.nunique(),
                   'methods': grouped.agg(lambda s: ', '.join(sorted(set(s))))})
        .reset_index()
        .sort_values(['n_methods', 'feature'], ascending=[False, True],
                     ignore_index=True)
    )


def run() -> DataFrame:
    """Run selection both with and without pipetting parameters, and write results."""
    out_dir = FEATURE_SELECTION_DIR / 'C'
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, groups = build_labeled_table()
    n_folds = groups.nunique()

    folds = pd.concat(
        [run_selection(include_params=flag) for flag in (True, False)],
        ignore_index=True)
    rankings = pd.concat(
        [model_rankings(include_params=flag) for flag in (True, False)],
        ignore_index=True)

    frequency = selection_frequency(folds, n_folds)

    stability = pd.concat(
        [selection_stability(folds, frequency, n_folds, include_params=flag)
         for flag in (True, False)],
        ignore_index=True)

    folds.to_csv(FOLD_PATH, index=False)
    frequency.to_csv(FREQUENCY_PATH, index=False)
    rankings.to_csv(RANKING_PATH, index=False)
    stability.to_csv(STABILITY_PATH, index=False)

    return frequency


def consensus_features(frequency: DataFrame, min_frequency: float = MIN_FREQUENCY,
                       min_methods: int = MIN_METHODS,
                       include_params: bool = False) -> list[str]:
    """Features that are both fold-stable *and* chosen by several methods.

    Two conditions, and both are needed:

    * ``frequency >= min_frequency`` -- the feature survives holding out a liquid,
      so it is not an artefact of one training set;
    * ``n_methods >= min_methods`` -- at least that many distinct selectors agree,
      so it is not an artefact of one method's inductive bias.

    The second condition is the one that was missing. Taking the *union* over
    methods, as this function previously did, is the opposite of a consensus: it
    admits a feature on the say-so of a single selector, and unanimity across the
    folds of one method measures that method's determinism, not the data. On the
    current run six of the fourteen unioned features rested on exactly one of the
    thirteen methods -- ``mrmr_mid`` picking ``N#`` in every fold says that
    mutual information with three neighbours likes atom counts at this n, not that
    nitrogen count predicts dispensing error.

    Defaults to the descriptor-only run: the pipetting parameters are properties of
    the protocol, not of the liquid, so they do not belong in a feature set that is
    about to be used to cluster liquids.
    """
    agreement = cross_method_agreement(
        frequency, min_frequency=min_frequency, include_params=include_params)
    if agreement.empty:
        chosen: list[str] = []
    else:
        supported = agreement[agreement['n_methods'] >= min_methods]
        chosen = sorted(set(supported['feature']) - set(cols.PIPETTING_PARAMS))

    for name in cols.PROTECTED:
        if name not in chosen:
            chosen.append(name)

    return chosen


if __name__ == '__main__':
    result = run()
    _, _, _groups = build_labeled_table()
    n_folds = _groups.nunique()
    threshold = ceil(MIN_FREQUENCY * n_folds)
    print(f'wrote {len(result)} rows to {FREQUENCY_PATH}')
    print(f'{n_folds} labelled liquids, so >={threshold}/{n_folds} folds\n')
    for flag in (True, False):
        view = result[result['include_params'] == flag]
        print(f'--- include_params={flag}: features selected in '
              f'>={threshold}/{n_folds} folds')
        top = view[view['frequency'] >= MIN_FREQUENCY]
        print(top[['method', 'feature', 'n_folds_selected', 'frequency']]
              .to_string(index=False) if len(top) else '  (none)')
        print()
    print('consensus (descriptors only):', consensus_features(result))
    print()
    print('--- selection stability, descriptors only')
    print(pd.read_csv(STABILITY_PATH).query('~include_params').to_string(index=False))
    print()
    print('--- how many methods agreed on each feature')
    print(cross_method_agreement(result).to_string(index=False))
