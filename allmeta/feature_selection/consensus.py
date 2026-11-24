"""Cross-pipeline agreement and the final comparison (plan, Phases 18-19).

Two questions, and the plan is careful to keep them apart because on this data they can
easily give opposite answers.

**Exact agreement** (Step 18, first half). Do two pipelines choose the same descriptors?
Measured by overlap, Jaccard and per-feature selection frequency across the outer folds.

**Information-family agreement** (Step 18, second half). Do they choose descriptors from
the same correlation families? Two selectors can share not one column name and still be
picking the same physics: the Phase 0 audit found mRMR returning `Ant1T, LogS, BPt, AntB,
RER` where four of the five correlate at |rho| >= 0.96, so a selector that picked `BPt`
and one that picked `AntB` disagree completely by name and not at all in substance.

``family_agreement`` measures the second by mapping every selected descriptor through a
fixed feature partition and comparing the *families* covered. The partition is the one
fitted on subset B, so it is the same reference for every pipeline and carries no target.

## Phase 19, and why it is reported rather than decided

The plan asks for a one-standard-error comparison over the final outer-CV summaries.
That is implemented, and it is also the fourth place in this project where the rule has
been applied -- the previous three all found the band too wide to discriminate at n = 13.
Phases 10-12 measured one standard error *wider than the entire gap between the best
model and predicting the mean*.

So ``final_comparison`` computes the band and reports what falls inside it, and
``one_se_is_informative`` states in the artefact itself whether the band was narrower
than the spread it was drawn over. When it is not, the correct reading of the table is
"these pipelines are not distinguishable at this sample size", and the plan's own closing
instruction applies with force: *do not declare a winner from a tiny numerical difference
alone.*
"""

from itertools import combinations

import numpy as np
import pandas as pd
from pandas import DataFrame

from allmeta.feature_selection import config


def _selected_sets(selections: DataFrame, column: str = 'features'
                   ) -> dict[tuple[str, int], set]:
    """(feature_set, fold) -> the set of inputs that fold finally used."""
    chosen: dict[tuple[str, int], set] = {}
    for record in selections.to_dict('records'):
        raw = record.get(column)
        # A fold that retained nothing writes NaN here, and float('nan') is TRUTHY --
        # so `raw or ''` keeps it and str() turns it into the literal 'nan', which then
        # enters the set as a feature name. Two families that both retained nothing
        # would agree perfectly on a descriptor called 'nan', and the phantom agreement
        # is invisible in the output because 0.077 is a plausible-looking Jaccard.
        # pd.isna covers NaN, None and NaT together.
        text = '' if raw is None or pd.isna(raw) else str(raw)
        names = [n.strip() for n in text.split(',') if n.strip()]
        chosen[(record['feature_set'], int(record['fold']))] = set(names)

    return chosen


# ---------------------------------------------------------------------------
# Phase 18a: exact feature agreement
# ---------------------------------------------------------------------------

def selection_frequency(selections: DataFrame) -> DataFrame:
    """How often each input survived, per pipeline, across the outer folds.

    The per-feature counterpart of Phase 11's stability, but measured on the *outer*
    folds and therefore on the pipeline as a whole rather than on one Elastic Net fit.
    A descriptor chosen in 13 of 13 folds is a different claim from one chosen in 7.
    """
    chosen = _selected_sets(selections)
    counts: dict[tuple[str, str], int] = {}
    folds: dict[str, set] = {}

    for (name, fold), features in chosen.items():
        folds.setdefault(name, set()).add(fold)
        for feature in features:
            counts[(name, feature)] = counts.get((name, feature), 0) + 1

    rows = [{'feature_set': name, 'feature': feature, 'n_folds_selected': count,
             'n_folds': len(folds[name]),
             'selection_frequency': count / len(folds[name])}
            for (name, feature), count in counts.items()]

    if not rows:
        return DataFrame(columns=['feature_set', 'feature', 'n_folds_selected',
                                  'n_folds', 'selection_frequency'])

    return DataFrame(rows).sort_values(
        ['feature_set', 'selection_frequency', 'feature'],
        ascending=[True, False, True], ignore_index=True)


def feature_overlap(selections: DataFrame) -> DataFrame:
    """Pairwise overlap and Jaccard between pipelines, fold by fold and pooled.

    Comparing within a fold as well as pooled matters: two pipelines can agree in every
    individual fold and still show a low pooled Jaccard if both are unstable, and the
    two readings answer different questions.
    """
    chosen = _selected_sets(selections)
    names = sorted({name for name, _ in chosen})
    if len(names) < 2:
        return DataFrame()

    all_folds = sorted({fold for _, fold in chosen})
    rows: list[dict] = []

    for left, right in combinations(names, 2):
        per_fold: list[float] = []
        for fold in all_folds:
            a, b = chosen.get((left, fold)), chosen.get((right, fold))
            if a is None or b is None or not (a | b):
                continue
            per_fold.append(len(a & b) / len(a | b))

        pooled_a = set().union(*(chosen.get((left, f), set()) for f in all_folds))
        pooled_b = set().union(*(chosen.get((right, f), set()) for f in all_folds))
        union = pooled_a | pooled_b

        rows.append({
            'feature_set_a': left, 'feature_set_b': right,
            'mean_within_fold_jaccard': float(np.mean(per_fold)) if per_fold else np.nan,
            'n_folds_compared': len(per_fold),
            'pooled_jaccard': len(pooled_a & pooled_b) / len(union) if union else np.nan,
            'n_shared': len(pooled_a & pooled_b),
            'n_only_a': len(pooled_a - pooled_b),
            'n_only_b': len(pooled_b - pooled_a),
            'shared_features': ', '.join(sorted(pooled_a & pooled_b)),
        })

    return DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 18b: information-family agreement
# ---------------------------------------------------------------------------

def family_of(membership: DataFrame) -> dict[str, str]:
    """Descriptor -> the name of the correlation family it belongs to.

    Built from a single reference partition so every pipeline is mapped through the same
    vocabulary. Using each pipeline's own clusters would compare two different questions
    and call the result agreement.
    """
    mapping: dict[str, str] = {}
    for record in membership.to_dict('records'):
        label = record.get('cluster_name') or f"cluster_{record.get('cluster_id')}"
        for feature in str(record.get('features', '')).split(','):
            feature = feature.strip()
            if feature:
                mapping[feature] = label

    return mapping


def family_agreement(selections: DataFrame, mapping: dict[str, str]) -> DataFrame:
    """Pairwise agreement measured on families rather than on column names.

    The analysis the plan asks to be able to say:

        the methods disagree on exact descriptors but agree on the same underlying
        information family

    ``family_jaccard`` above ``feature_jaccard`` is exactly that statement, and the gap
    between the two columns is the size of it. Descriptors absent from the reference
    partition are mapped to their own name, so an unclustered descriptor counts as its
    own family rather than being dropped.
    """
    chosen = _selected_sets(selections)
    names = sorted({name for name, _ in chosen})
    if len(names) < 2:
        return DataFrame()

    def families(features: set) -> set:
        return {mapping.get(f, f) for f in features}

    all_folds = sorted({fold for _, fold in chosen})
    rows: list[dict] = []

    for left, right in combinations(names, 2):
        feature_scores: list[float] = []
        family_scores: list[float] = []
        for fold in all_folds:
            a, b = chosen.get((left, fold)), chosen.get((right, fold))
            if a is None or b is None or not (a | b):
                continue
            feature_scores.append(len(a & b) / len(a | b))
            fa, fb = families(a), families(b)
            family_scores.append(len(fa & fb) / len(fa | fb) if (fa | fb) else np.nan)

        rows.append({
            'feature_set_a': left, 'feature_set_b': right,
            'feature_jaccard': float(np.mean(feature_scores))
            if feature_scores else np.nan,
            'family_jaccard': float(np.nanmean(family_scores))
            if family_scores else np.nan,
            'n_folds_compared': len(feature_scores),
        })

    frame = DataFrame(rows)
    frame['family_agreement_exceeds_feature_agreement'] = (
        frame['family_jaccard'] > frame['feature_jaccard'])

    return frame


def family_coverage(selections: DataFrame, mapping: dict[str, str]) -> DataFrame:
    """How often each information family is represented, per pipeline.

    A family selected in every fold under a different descriptor each time is a stable
    finding about the chemistry and an unstable one about the descriptor. Both columns
    are here so the two cannot be confused.
    """
    chosen = _selected_sets(selections)
    rows: list[dict] = []
    folds: dict[str, set] = {}

    counts: dict[tuple[str, str], dict] = {}
    for (name, fold), features in chosen.items():
        folds.setdefault(name, set()).add(fold)
        for family in {mapping.get(f, f) for f in features}:
            entry = counts.setdefault((name, family),
                                      {'n_folds': 0, 'descriptors': set()})
            entry['n_folds'] += 1
            entry['descriptors'] |= {f for f in features
                                     if mapping.get(f, f) == family}

    for (name, family), entry in counts.items():
        rows.append({
            'feature_set': name, 'family': family,
            'n_folds_represented': entry['n_folds'],
            'n_folds': len(folds[name]),
            'coverage': entry['n_folds'] / len(folds[name]),
            'n_distinct_descriptors_used': len(entry['descriptors']),
            'descriptors': ', '.join(sorted(entry['descriptors'])),
        })

    if not rows:
        return DataFrame()

    return DataFrame(rows).sort_values(['feature_set', 'coverage', 'family'],
                                       ascending=[True, False, True],
                                       ignore_index=True)


# ---------------------------------------------------------------------------
# Phase 19: the final one-standard-error comparison
# ---------------------------------------------------------------------------

def final_comparison(fold_scores: DataFrame, summary: DataFrame | None = None,
                     model: str = 'ridge', metric: str = 'liquid_squared_error'
                     ) -> DataFrame:
    """One-SE comparison over the outer folds, with its own informativeness reported.

    The standard error comes from the **outer folds** -- thirteen numbers, one per
    held-out liquid -- rather than from the inner search, which is the only version of
    this comparison that estimates anything.

    ``one_se_is_informative`` is the column to read before the ranking. It is False when
    the band covers the whole spread of pipelines being compared, which is what the
    previous three applications of this rule found at n = 13. When it is False the table
    says the pipelines are indistinguishable, and any ordering inside it is noise.
    """
    if fold_scores.empty:
        return DataFrame()

    view = fold_scores[fold_scores['model'] == model]
    if view.empty:
        return DataFrame()

    grouped = (view.groupby('feature_set')[metric]
               .agg(mean='mean', std='std', n='size').reset_index())
    grouped['se'] = grouped['std'] / np.sqrt(grouped['n'])

    best = grouped.loc[grouped['mean'].idxmin()]
    threshold = float(best['mean'] + best['se'])
    spread = float(grouped['mean'].max() - grouped['mean'].min())

    grouped['one_se_threshold'] = threshold
    grouped['best_pipeline'] = best['feature_set']
    grouped['within_one_se'] = grouped['mean'] <= threshold
    grouped['rank'] = grouped['mean'].rank(method='min').astype(int)
    grouped['metric'] = metric
    grouped['model'] = model
    # The band's width against the spread it is drawn over. Above 1.0 the rule cannot
    # separate anything: every pipeline is inside the band by construction.
    grouped['band_width_over_spread'] = (
        float(best['se']) / spread if spread > 0 else np.inf)
    grouped['one_se_is_informative'] = bool(
        grouped['within_one_se'].sum() < len(grouped)) if spread > 0 else False

    if summary is not None and not summary.empty:
        carried = [c for c in ('between_liquid_r2', 'rmse', 'n_features_median',
                               'ci_lo', 'ci_hi', 'frac_above_zero',
                               'folds_retaining_everything')
                   if c in summary.columns]
        grouped = grouped.merge(summary[['feature_set'] + carried],
                                on='feature_set', how='left')

    return grouped.sort_values('mean', ignore_index=True)


def pareto(summary: DataFrame, error: str = 'rmse',
           complexity: str = 'n_features_median') -> DataFrame:
    """Which pipelines are not dominated on both error and complexity.

    The plan's Step 16.9 Pareto plot as a table. A pipeline is dominated when another
    is at least as good on error *and* uses at least as few inputs; the survivors are
    the only ones where a trade-off is actually being made.
    """
    if summary.empty or error not in summary or complexity not in summary:
        return DataFrame()

    view = summary[['feature_set', error, complexity]].dropna().copy()
    dominated = []
    for row in view.itertuples():
        beaten = view[(view[error] <= getattr(row, error))
                      & (view[complexity] <= getattr(row, complexity))
                      & (view['feature_set'] != row.feature_set)]
        dominated.append(not beaten.empty)

    view['is_dominated'] = dominated
    view['on_pareto_front'] = ~view['is_dominated']

    return view.sort_values(error, ignore_index=True)
