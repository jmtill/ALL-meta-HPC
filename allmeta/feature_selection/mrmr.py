"""Leakage-safe mRMR selection of a raw feature subset (Step 2.2).

mRMR -- Minimum Redundancy, Maximum Relevance -- is the only one of the five feature
representations that consults the target, so it is the only one that can leak. The
entire design of this module follows from that:

* ``select_mrmr`` fits on exactly the rows it is handed and nothing else. It has no
  access to a subset loader, no default frame, and no cache. Call it inside a fold with
  that fold's training rows and it cannot see the held-out liquid.
* the constant-feature guard is fitted inside the same call, not once globally, so the
  guard itself never consults held-out rows either.

The guard is not optional bookkeeping. feature_engine's ratio variants (MIQ, FCQ, RFCQ)
score relevance divided by redundancy, and a zero-variance column has zero correlation
with everything, so it divides by approximately zero and scores *maximally*. Several
atom counts are constant across the 13 labelled liquids; without the guard, ``Br#`` --
identically zero everywhere -- is selected in every fold by three variants.

The estimator itself is built by ``allmeta.supervised.mrmr_selectors`` rather than
constructed here. Two of its arguments are load-bearing and easy to lose in a
re-implementation: ``regression=True`` must be passed explicitly because
feature_engine's signature default is ``False``, which would silently treat a
continuous target as class labels, and RFCQ needs its ``scoring`` overridden away from
the default ``'roc_auc'``, which errors on a continuous target.

Two more traps, both read out of feature_engine 1.9's source and verified against it
rather than assumed:

* ``MRMR.get_feature_names_out()`` returns the selected features in **original column
  order**, not in the order mRMR admitted them -- it rebuilds the list from
  ``variables_`` after the greedy loop and the ordering is lost. On a frame where the
  dominant descriptor sits in the last column it comes back last, so reading that
  order as a ranking inverts the result. ``selection_order`` reconstructs the real
  order: rank 1 is ``relevance_.argmax()``, exactly as the algorithm's own first step
  computes it, and each later rank is the feature that appears when the budget grows
  by one.
* ``max_features=1`` returns **two** features. The fit loop runs ``max_features - 2``
  times and then appends one more unconditionally, so a budget of 1 gives -1
  iterations and still lands on two. ``select_mrmr`` truncates to the single most
  relevant feature in that case rather than silently handing back one more feature
  than it was asked for.
"""

import warnings

from pandas import DataFrame, Series
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler

from allmeta.feature_selection import config
from allmeta.feature_selection.provenance import FeatureSpec, raw_spec
from allmeta.supervised import mrmr_selectors

SOURCE_METHOD: str = 'mrmr'


def _fit(scaled: DataFrame, target: Series, method: str, budget: int):
    """Fit one feature_engine MRMR selector on an already-guarded, scaled frame."""
    selectors = mrmr_selectors(max_features=budget)
    key = f'mrmr_{method.lower()}'
    if key not in selectors:
        raise ValueError(f'unknown mRMR method {method!r}; expected one of '
                         f'{[k.removeprefix("mrmr_").upper() for k in selectors]}')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return selectors[key].fit(scaled, target)


def most_relevant(selector) -> str:
    """The feature mRMR admits first: the highest relevance, before any redundancy.

    Read from the fitted selector's own ``relevance_`` rather than inferred from a
    budget-1 fit, because a budget-1 fit returns two features (see the module
    docstring) and cannot distinguish them.
    """
    scores = relevance_scores(selector)
    if not scores:
        raise ValueError('the fitted selector exposes no usable relevance_ array')

    return max(scores, key=lambda feature: scores[feature])


def selection_order(scaled: DataFrame, target: Series, method: str,
                    budget: int, selector=None) -> list[str]:
    """The order in which mRMR admits features, reconstructed exactly.

    mRMR is greedy, so the set it returns for a budget of k always contains the set it
    returns for k-1: whichever feature appears at k and not at k-1 is the one admitted
    k-th. Rank 1 comes from ``relevance_`` instead of from a k=1 fit, which would
    return two features and leave the pair unordered.

    Costs ``budget - 1`` extra fits. Worth paying: "most relevant outright" and "added
    something beyond the first four" are different claims about a descriptor, and
    column order supports neither.
    """
    selector = _fit(scaled, target, method, budget) if selector is None else selector
    order: list[str] = [most_relevant(selector)]

    for k in range(2, budget + 1):
        chosen = list(_fit(scaled, target, method, k).get_feature_names_out())
        for feature in chosen:
            if feature not in order:
                order.append(feature)

    return order


def relevance_scores(selector, candidates: list[str] | None = None) -> dict[str, float]:
    """Per-feature relevance from a fitted selector, keyed by name.

    feature_engine exposes ``relevance_`` as a bare array aligned with ``variables_``.
    Returns an empty mapping rather than guessing if the two lengths disagree -- a
    misaligned score column would be worse than no score column.
    """
    scores = getattr(selector, 'relevance_', None)
    names = candidates if candidates is not None else getattr(selector, 'variables_', None)
    if scores is None or names is None or len(scores) != len(names):
        return {}

    return {feature: float(value) for feature, value in zip(names, scores)}


def select_mrmr(data: DataFrame, target: Series,
                n_features: int | None = None,
                method: str | None = None,
                with_rank: bool = True,
                ) -> tuple[list[str], DataFrame, list[FeatureSpec]]:
    """Select ``n_features`` raw descriptors from ``data`` against ``target``.

    Returns ``(features, results, specs)``. ``results`` is one row per *candidate*
    descriptor, not per selected one, carrying whether it was selected and -- for the
    selected ones -- the greedy order in which mRMR admitted them. Recording the
    rejected candidates matters at this sample size: a descriptor that just missed the
    cut is a different fact from one mRMR never considered because the guard dropped it.

    ``data`` and ``target`` must share an index and must be a *training* slice. Nothing
    in this function widens that slice.
    """
    n_features = config.MRMR_N_FEATURES if n_features is None else n_features
    method = config.MRMR_METHOD if method is None else method

    if not data.index.equals(target.index):
        raise ValueError('data and target are not aligned; a misaligned fit is silent')
    if data.isna().any().any() or target.isna().any():
        raise ValueError('mRMR cannot fit on missing values; complete-case only')
    if n_features < 1:
        raise ValueError(f'n_features must be >= 1, got {n_features}')

    candidates = list(data.columns)

    # Fitted here, on the training rows only. See the module docstring for why a
    # constant column is not merely uninformative but actively wins under MIQ/FCQ/RFCQ.
    guard = VarianceThreshold(0.0).set_output(transform='pandas')
    surviving = guard.fit_transform(data)
    dropped = [f for f in candidates if f not in surviving.columns]

    budget = min(n_features, surviving.shape[1])
    if budget < n_features:
        warnings.warn(
            f'asked for {n_features} features but only {surviving.shape[1]} are '
            f'non-constant in this training slice; selecting {budget}',
            stacklevel=2)

    scaled = RobustScaler().set_output(transform='pandas').fit_transform(surviving)

    selector = _fit(scaled, target, method, budget)

    if budget == 1:
        # feature_engine returns two features for a budget of one; keep the one its
        # own first step chose. See the module docstring.
        selected = [most_relevant(selector)]
    elif with_rank:
        selected = selection_order(scaled, target, method, budget, selector=selector)
    else:
        # Ranking costs budget - 1 extra fits: negligible once, not free inside a
        # 13-fold loop, so a caller that only needs the set can decline it.
        chosen = set(selector.get_feature_names_out())
        selected = [f for f in surviving.columns if f in chosen]

    chosen = set(selected)
    # `selected` is in greedy order only when the order was reconstructed, or when
    # there is a single feature and the order is trivial. Otherwise it is column order
    # and must not be reported as a ranking.
    ordered = with_rank or budget == 1
    rank = ({feature: position for position, feature in enumerate(selected, start=1)}
            if ordered else {})

    relevance = relevance_scores(selector, list(surviving.columns))

    results = DataFrame([{
        'feature': feature,
        'method': method,
        'selected': feature in chosen,
        'rank': rank.get(feature),
        'relevance': relevance.get(feature),
        'dropped_by_constant_guard': feature in dropped,
        'n_train_rows': len(data),
    } for feature in candidates])
    results = results.sort_values(['selected', 'rank', 'feature'],
                                  ascending=[False, True, True], ignore_index=True)

    specs = [raw_spec(feature, SOURCE_METHOD) for feature in selected]

    return selected, results, specs
