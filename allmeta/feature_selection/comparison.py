"""Comparing the clustering methods against each other (implementation plan, Phase 9).

Phase 8 scored all 430 candidates on predictive error and applied the one-standard-error
rule **once, globally**. The plan's Step 8.3 asks for something narrower -- "*for each
clustering method*, find minimum mean RMSE" -- and the difference is not cosmetic. A
global rule never asks what the best K-Medoids candidate is on its own terms; it only
reports which single row of 430 won. Comparing methods *as methods* needs each one
selected inside its own grid first, and that is what ``select_per_method`` does.

## What the per-method split immediately showed

Of the five methods, **three have no candidate that beats the mean baseline at all**:

    method        candidates beating baseline (of its grid)
    kmedoids      10
    gmm            6
    hierarchical   0
    hdbscan        0
    kmeans         0

The 16 baseline-beating candidates the Phase 8 audit counted are entirely K-Medoids and
GMM. So `config.REQUIRE_BEATS_BASELINE` -- a guard already in the codebase, written for a
different reason -- excludes three of the five methods on its own, before any new
criterion is introduced. Hierarchical and HDBSCAN winning every cluster-quality and
stability comparison while failing to beat the mean is the sharpest form of the audit's
finding that cluster quality does not predict predictive usefulness.

A method with nothing beating the baseline is recorded ``admitted=False`` with the reason
attached. It stays in every metric, stability and search table. It is dropped only from
``carry_forward``, and a caller can see exactly why.

## The one-standard-error band does not do its job at n=13

The rule's premise is that everything inside the band is statistically indistinguishable,
so the tie-break may choose on other grounds at no cost in error. On this data the
global band holds 13 candidates whose held-out between-liquid R-squared runs from
**+0.106 to -0.140**. They are indistinguishable only in the sense that 13 liquids cannot
tell them apart -- which is a statement about the sample size, not about the candidates.

``replay_tie_breaks`` exists to measure that rather than argue about it. It re-applies
any tie-break ordering to search tables that have already been computed, so every
ordering sees exactly the same scored candidates and no model is refitted.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

from allmeta.feature_selection import config, optimization
from allmeta.feature_selection.optimization import BASELINE_NAME

# The columns the plan's Step 9.1 names for clustering_method_comparison.csv, in its
# order. Asserted against in the tests: a silently missing column is the failure mode
# here, because a reader cannot tell an absent metric from one that was never computed.
PLAN_COMPARISON_COLUMNS: tuple[str, ...] = (
    'clustering_method', 'representation', 'hyperparameters', 'n_clusters',
    'n_singletons', 'noise_fraction', 'silhouette', 'calinski_harabasz',
    'davies_bouldin', 'mean_within_abs_corr', 'mean_between_abs_corr',
    'within_between_delta', 'mean_pc1_explained_variance',
    'min_pc1_explained_variance', 'stability_ari', 'stability_ami', 'stability_nmi',
    'n_output_pc1_features', 'inner_cv_rmse_mean', 'inner_cv_rmse_std',
    'inner_cv_rmse_se', 'inner_cv_mae', 'within_one_se', 'selected',
)

# Every plan-named mean that is inflated by singletons, and the companion that has to sit
# beside it. The audit measured the inflation: within_between_delta correlates 0.796 with
# singleton_fraction, mean PC1 retention 0.61, ARI 0.61. A table that prints the plain
# mean without its companion invites precisely the misreading the audit had to correct --
# a delta of 0.60 over 1.6% of the feature pairs is a refusal to cluster, not a result.
COMPANION_COLUMNS: dict[str, tuple[str, ...]] = {
    'within_between_delta': ('within_pair_fraction', 'singleton_fraction'),
    'mean_within_abs_corr': ('within_pair_fraction',),
    'mean_pc1_explained_variance': ('mean_pc1_explained_variance_multi',
                                    'n_multi_member_clusters'),
    'stability_ari': ('singleton_fraction', 'consensus_ambiguous_fraction'),
}

# Where each plan-named column is read from in the internal tables.
_SOURCE = {
    'clustering_method': 'method',
    'silhouette': 'silhouette_correlation',
    'calinski_harabasz': 'calinski_harabasz_profile',
    'davies_bouldin': 'davies_bouldin_profile',
    'stability_ari': 'ari_mean',
    'stability_ami': 'ami_mean',
    'stability_nmi': 'nmi_mean',
    'n_output_pc1_features': 'n_factors',
    'inner_cv_rmse_mean': 'pooled_rmse_mean',
    'inner_cv_rmse_std': 'pooled_rmse_std',
    'inner_cv_rmse_se': 'pooled_rmse_se',
    'inner_cv_mae': 'mae_mean',
}


# ---------------------------------------------------------------------------
# Per-method selection -- the half of Step 8.3 Phase 8 did not implement
# ---------------------------------------------------------------------------

def select_per_method(table: DataFrame, context: DataFrame | None = None,
                      **kwargs) -> tuple[DataFrame, DataFrame]:
    """Apply the one-standard-error rule inside each method's own rows.

    Each method gets its **own** ``E_min`` and ``SE_min``. Letting a uniformly poor
    method inherit the best method's error bar would make it look competitive by
    borrowing someone else's precision, which is the opposite of what the rule is for.

    The shared baseline row is appended to every method's group so
    ``config.REQUIRE_BEATS_BASELINE`` still bites per method. A method with nothing
    beating the baseline does not raise out of the comparison -- it is recorded with
    ``admitted=False`` and its reason, because on this data that is the *typical*
    outcome and an exception would abort the whole table.

    Returns ``(winners, competitive)``: one row per method, and every competitive
    candidate of every method with a ``method_rank`` column.
    """
    baseline = table[table['method'] == BASELINE_NAME]
    methods = [m for m in table['method'].unique() if m != BASELINE_NAME]

    winners: list[dict] = []
    competitive_frames: list[DataFrame] = []

    for method in sorted(methods):
        group = table[table['method'] == method]
        # The baseline is scored on the same folds and is method-independent, so
        # appending it costs nothing and keeps the guard's semantics identical to the
        # global call.
        scoped = pd.concat([group, baseline], ignore_index=True)

        try:
            competitive, chosen = optimization.one_standard_error(
                scoped, context=context, **kwargs)
        except ValueError as error:
            winners.append({
                'method': method,
                'admitted': False,
                'admission_reason': str(error).split(';')[0],
                'n_candidates': int(len(group)),
                'n_beats_baseline': 0,
                'n_competitive': 0,
            })
            continue

        competitive = competitive.copy()
        competitive['method_rank'] = np.arange(len(competitive))
        competitive_frames.append(competitive)

        criterion = f'{config.SELECTION_CRITERION}_mean'
        beating = (int((group[criterion] < float(baseline[criterion].iloc[0])).sum())
                   if criterion in group and not baseline.empty else -1)

        record = chosen.to_dict()
        record.update({
            'admitted': True,
            'admission_reason': '',
            'n_candidates': int(len(group)),
            'n_beats_baseline': beating,
            'n_competitive': int(len(competitive)),
        })
        winners.append(record)

    frame = DataFrame(winners)
    all_competitive = (pd.concat(competitive_frames, ignore_index=True)
                       if competitive_frames else DataFrame())

    return frame, all_competitive


# ---------------------------------------------------------------------------
# The admission gate -- diagnostics only, never predictive error
# ---------------------------------------------------------------------------

def admission(winners: DataFrame, metrics: DataFrame,
              criteria: dict | None = None, scope: str | None = None) -> DataFrame:
    """Decide which methods are candidates rather than controls.

    The gate reads **fitted-model diagnostics only** -- convergence, posterior
    confidence, PC1 retention, partition stability. None of them involves the target.
    That restriction is the whole design. GMM wins the diagnostic search *and* 10 of 13
    per-fold selections while holding the worst separation, the worst retention, the
    worst stability and a diagnosable degeneracy; the two readings of that (cluster
    quality does not predict usefulness, or the signal is too weak and the winner is
    noise) cannot be separated at 16/430 and a 1.13-SE margin. A gate built from
    predictive error would be a gate built from the outcome we would prefer. This one
    cannot see the outcome at all.

    ## Why the gate is evaluated over the method's whole grid, not over its winner

    ``scope='grid'`` (the default) takes the median of each diagnostic across every
    candidate the method produced. ``scope='winner'`` reads the single selected
    candidate instead, and it is available mainly to show why it is the wrong level.

    Evaluated on the winner, this gate rejects GMM by **0.000514** -- a multi-member PC1
    retention of 0.599486 against a floor of 0.60. That is not a decision, it is a coin
    flip. Worse, it is contingent: GMM's winner is the 3-component solution only because
    the parsimony-first tie-break chose it, and the 6-component solution one row above it
    retains 0.6238 and would pass. So a winner-scoped gate hands its verdict to a
    tie-break ordering that the outer-fold replay shows is itself indistinguishable from
    noise. A gate whose answer depends on an undecidable upstream choice is not a
    criterion.

    The grid scope has no such dependence, and on this data it is not marginal either:
    GMM fails **two independent criteria** -- a median ARI of 0.679 against a 0.70 floor,
    and a median mean-posterior of exactly 1.0000 against a 0.999 ceiling. Every other
    method passes every criterion. The degeneracy argument was always a claim about GMM
    *at this data shape* -- 45 descriptors in 44 dimensions, every fit converging in 2-8
    EM iterations -- and that is a property of the grid, not of one candidate.

    Methods that produce no posterior leave the confidence column NA and pass it
    vacuously: an absent diagnostic is not a failed one.

    ``min_pc1_explained_variance_multi`` reuses ``config.PC1_RETENTION_FLOOR``, set in
    Phase 6 before any predictive number existed. It is not a threshold invented here to
    reach a conclusion.

    A failing method keeps every row it has in every other table. It is dropped from
    ``carry_forward`` and from nothing else.
    """
    criteria = (config.CLUSTERING_METHOD_ADMISSION if criteria is None else criteria)
    scope = config.ADMISSION_SCOPE if scope is None else scope
    if scope not in ('grid', 'winner'):
        raise ValueError(f"unknown admission scope {scope!r}; expected 'grid' or "
                         "'winner'")

    winner_of = {row['method']: row for row in winners.to_dict('records')}
    rows: list[dict] = []

    for method in winners['method']:
        grid = metrics[metrics['method'] == method]
        winner = winner_of[method]
        row: dict = {'method': method, 'admission_scope': scope,
                     'n_grid_candidates': int(len(grid))}

        # A method already excluded by the baseline guard keeps that reason. Re-testing
        # it on diagnostics would report a second, less relevant story about why it is
        # not carried forward.
        if not winner.get('admitted', True):
            row.update({'admitted': False,
                        'admission_reason': winner.get('admission_reason', ''),
                        'failed_criteria': 'beats_baseline'})
            rows.append(row)
            continue

        def measure(column: str) -> float:
            """The diagnostic this gate judges, at the configured scope."""
            if scope == 'winner':
                return winner.get(column, np.nan)
            if column not in grid.columns or grid[column].isna().all():
                return np.nan
            return float(grid[column].median())

        failures: list[str] = []

        if criteria.get('require_all_converged', True):
            converged = (grid['converged'] if 'converged' in grid
                         else pd.Series(dtype=bool))
            n_failed = int((~converged.fillna(True).astype(bool)).sum())
            row['n_unconverged_fits'] = n_failed
            if n_failed:
                failures.append('converged')

        for key, column, direction in (
                ('max_mean_assignment_confidence',
                 'extra_mean_assignment_confidence', 'below'),
                ('min_pc1_explained_variance_multi',
                 'mean_pc1_explained_variance_multi', 'above'),
                ('min_ari_mean', 'ari_mean', 'above')):
            limit = criteria.get(key)
            value = measure(column)
            row[column] = value
            # Carried beside the judged value so a reader can see what the other scope
            # would have said without rerunning anything.
            row[f'{column}_winner'] = winner.get(column, np.nan)
            if limit is None or pd.isna(value):
                continue
            if (direction == 'below' and value >= limit) or \
               (direction == 'above' and value < limit):
                failures.append(column)

        row.update({
            'admitted': not failures,
            'admission_reason': ('' if not failures
                                 else 'failed ' + ', '.join(failures)),
            'failed_criteria': ', '.join(failures),
        })
        rows.append(row)

    frame = DataFrame(rows)
    for key, value in criteria.items():
        frame[f'criterion_{key}'] = value

    return frame


# ---------------------------------------------------------------------------
# The comparison table
# ---------------------------------------------------------------------------

def method_comparison(search: DataFrame, metrics: DataFrame,
                      winners: DataFrame | None = None,
                      gate: DataFrame | None = None) -> DataFrame:
    """One row per clustering method, carrying every column Step 9.1 names.

    Cross-method one-SE is applied on top of the per-method winners: the best method's
    ``E_min`` and its ``SE_min`` set the band, and every method inside it is flagged
    ``within_one_se``. ``selected`` marks the single method ``CLUSTERING_COMPARISON_MODE
    = 'winner'`` would carry forward.
    """
    if winners is None:
        winners, _ = select_per_method(search, context=metrics)
    if gate is None:
        gate = admission(winners, metrics)

    frame = winners.copy()

    # Rename into the plan's vocabulary, keeping the internal name too where it is not
    # simply an alias -- a reader tracing a number back to clustering_internal_metrics
    # needs the name it has there.
    for plan_name, internal in _SOURCE.items():
        if internal in frame.columns:
            frame[plan_name] = frame[internal]
        elif plan_name not in frame.columns:
            frame[plan_name] = np.nan

    criterion = config.SELECTION_CRITERION
    mean_column, se_column = f'{criterion}_mean', f'{criterion}_se'

    # The gate is applied FIRST, and the cross-method band is then computed over
    # admitted methods only. Order matters: a rejected method that happens to hold the
    # lowest error would otherwise set E_min and SE_min for everyone else, so the band
    # would be drawn around a candidate that cannot be carried forward -- and no method
    # would come out 'selected' at all.
    if gate is not None and not gate.empty:
        overlapping = [c for c in gate.columns
                       if c != 'method' and c in frame.columns]
        frame = frame.drop(columns=overlapping).merge(gate, on='method', how='left')
    if 'admitted' not in frame.columns:
        frame['admitted'] = True
    frame['admitted'] = frame['admitted'].fillna(False).astype(bool)

    admitted = frame[frame['admitted']]
    scored = (admitted[admitted[mean_column].notna()]
              if mean_column in admitted else DataFrame())

    if not scored.empty:
        best = scored.loc[scored[mean_column].idxmin()]
        threshold = float(best[mean_column] + best[se_column])
        frame['one_se_threshold'] = threshold
        frame['best_method'] = best['method']
        frame['within_one_se'] = (frame[mean_column].notna()
                                  & (frame[mean_column] <= threshold)
                                  & frame['admitted'])
        frame['selected'] = (frame['method'] == best['method']) & frame['admitted']
    else:
        frame['one_se_threshold'] = np.nan
        frame['best_method'] = ''
        frame['within_one_se'] = False
        frame['selected'] = False

    frame['selection_criterion'] = criterion
    frame['comparison_mode'] = config.CLUSTERING_COMPARISON_MODE

    # The companion guarantee has to hold whatever the metrics table happened to carry.
    # If a companion is missing it is written as NA rather than omitted: a reader who
    # sees within_between_delta with no within_pair_fraction beside it has no way to
    # know the coverage was never measured, and that is exactly the misreading these
    # pairs exist to prevent.
    for column, companions_of in COMPANION_COLUMNS.items():
        if column not in frame.columns:
            continue
        for companion in companions_of:
            if companion not in frame.columns:
                frame[companion] = np.nan

    ordered = [c for c in PLAN_COMPARISON_COLUMNS if c in frame.columns]
    # Deduplicated: within_pair_fraction is the companion of BOTH mean_within_abs_corr
    # and within_between_delta, and singleton_fraction of both within_between_delta and
    # stability_ari. Left as-is the selection below returns a frame with repeated
    # columns, and pandas then silently drops values in .to_dict('records') -- which is
    # how a duplicated column turns into a missing feature set two phases downstream.
    companions: list[str] = []
    for column in ordered:
        for companion in COMPANION_COLUMNS.get(column, ()):
            if companion in frame.columns and companion not in companions:
                companions.append(companion)

    taken = set(ordered) | set(companions)
    rest = [c for c in frame.columns if c not in taken]

    return frame[ordered + companions + rest]


def candidate_comparison(search: DataFrame, metrics: DataFrame,
                         competitive: DataFrame | None = None) -> DataFrame:
    """Every candidate, flagged by whether it won or was competitive in its method.

    The five-row winners table hides the spread, and the audit's finding was about the
    spread: a randomly chosen candidate is roughly three times worse than the baseline,
    so selecting helps by avoiding a bad pool rather than by finding a strong
    representation. That claim is a statement about this table, not the winners one.
    """
    if competitive is None:
        _, competitive = select_per_method(search, context=metrics)

    frame = search[search['method'] != BASELINE_NAME].merge(
        metrics, on=['method', 'hyperparameters'], how='left', suffixes=('', '_metric'))

    if competitive.empty:
        frame['is_method_winner'] = False
        frame['is_method_competitive'] = False
        return frame

    key = ['method', 'hyperparameters']
    flags = competitive[key + ['method_rank']].copy()
    flags['is_method_competitive'] = True
    flags['is_method_winner'] = flags['method_rank'] == 0

    frame = frame.merge(flags, on=key, how='left')
    frame['is_method_competitive'] = frame['is_method_competitive'].fillna(False)
    frame['is_method_winner'] = frame['is_method_winner'].fillna(False)

    return frame


# ---------------------------------------------------------------------------
# What reaches Elastic Net
# ---------------------------------------------------------------------------

def carry_forward(comparison: DataFrame, mode: str | None = None) -> DataFrame:
    """The clustering solution(s) that become feature sets for Phase 10.

    ``winner``              exactly one, named ``auto_cluster_pc1``.
    ``retain_competitive``  one per admitted method inside the cross-method one-SE
                            band, named ``auto_cluster_pc1__<method>``.

    ``retain_competitive`` is the mode worth defaulting to for research use. Phase 8
    found six distinct winning configurations across thirteen outer folds; carrying a
    single method forward asserts a stability this data does not have.

    Returns the selected rows with a ``feature_set_name`` column. Resolving those into
    actual factor matrices is the caller's job -- this module does not fit anything.
    """
    mode = config.CLUSTERING_COMPARISON_MODE if mode is None else mode
    if mode not in ('winner', 'retain_competitive'):
        raise ValueError(f'unknown CLUSTERING_COMPARISON_MODE {mode!r}; '
                         "expected 'winner' or 'retain_competitive'")

    admitted = comparison[comparison.get('admitted', True).astype(bool)]
    if admitted.empty:
        raise ValueError(
            'no clustering method was admitted, so nothing can be carried forward. '
            'Every method either failed to beat the mean baseline or failed the '
            'diagnostic admission gate -- see clustering_method_admission.csv.')

    if mode == 'winner':
        chosen = admitted[admitted['selected'].astype(bool)].head(1).copy()
        # No silent fallback to "the first admitted row". If the band was drawn and
        # nothing came out selected, something upstream is wrong -- most likely the gate
        # was applied after the band instead of before it -- and quietly returning an
        # arbitrary method would hide it behind a plausible-looking answer.
        if chosen.empty:
            raise ValueError(
                'no method is marked selected although '
                f'{len(admitted)} were admitted; the cross-method one-SE band was not '
                'drawn over the admitted set. This is a bug, not a data outcome.')
        chosen['feature_set_name'] = 'auto_cluster_pc1'
    else:
        chosen = admitted[admitted['within_one_se'].astype(bool)].copy()
        if chosen.empty:
            raise ValueError(
                'no admitted method fell inside the cross-method one-SE band, which '
                'cannot happen when the band is drawn over the admitted set -- the '
                'best admitted method is always inside its own band.')
        chosen['feature_set_name'] = (
            'auto_cluster_pc1__' + chosen['method'].astype(str))

    chosen['comparison_mode'] = mode

    return chosen.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tie-break replay
# ---------------------------------------------------------------------------

def replay_tie_breaks(per_fold_search: DataFrame, context: DataFrame | None = None,
                      orderings: dict | None = None,
                      scope: str = 'global') -> tuple[DataFrame, DataFrame]:
    """Re-apply tie-break orderings to search tables that already exist.

    **No model is refitted.** Every ordering sees the identical 13 x 430 scored
    candidates, so the comparison isolates the tie-break and nothing else. That is the
    only reason this is affordable and the only reason it is interpretable.

    ``scope='global'`` reproduces the Phase 8 rule (one band over all methods);
    ``scope='per_method'`` runs the Step 8.3 reading and reports the best admitted
    method's choice.

    Returns ``(per_fold, summary)``.

    ## Read the summary as a diagnostic, not as evidence of generalisation

    The summary ranks orderings by the mean held-out between-liquid R-squared of the
    candidate each one selected. That is a quantity computed on the outer folds, so
    choosing an ordering by it is a selection made against the outer folds and its
    apparent advantage is optimistic by an unmeasured amount. It is reported because
    "which ordering would have done better here?" is worth knowing and cheap; it is
    **not** a substitute for deciding the ordering on an argument about what parsimony
    means. If Phase 14 needs a tie-break choice that costs nothing, the ordering has to
    be nested inside its outer folds there.
    """
    orderings = (config.TIE_BREAK_ORDERINGS if orderings is None else orderings)

    rows: list[dict] = []
    for name, ordering in orderings.items():
        for fold, block in per_fold_search.groupby('fold', sort=True):
            held_out = block['held_out_liquid'].iloc[0]
            try:
                competitive, chosen = optimization.one_standard_error(
                    block, context=context, tie_break=tuple(ordering))
            except ValueError as error:
                rows.append({'ordering': name, 'fold': int(fold),
                             'held_out_liquid': held_out, 'selected': False,
                             'reason': str(error).split(';')[0]})
                continue

            rows.append({
                'ordering': name,
                'fold': int(fold),
                'held_out_liquid': held_out,
                'selected': True,
                'method': chosen['method'],
                'hyperparameters': chosen['hyperparameters'],
                'n_factors': int(chosen['n_factors']),
                'criterion_mean': float(chosen[f'{config.SELECTION_CRITERION}_mean']),
                'between_liquid_r2': float(chosen['between_liquid_r2']),
                'mean_pc1_explained_variance_multi': chosen.get(
                    'mean_pc1_explained_variance_multi', np.nan),
                'ari_mean': chosen.get('ari_mean', np.nan),
                'n_competitive': int(len(competitive)),
                'scope': scope,
            })

    per_fold = DataFrame(rows)
    if per_fold.empty or 'method' not in per_fold.columns:
        return per_fold, DataFrame()

    chosen_rows = per_fold[per_fold['selected'].astype(bool)]
    summary = (chosen_rows.groupby('ordering')
               .agg(n_folds_selected=('fold', 'size'),
                    mean_between_liquid_r2=('between_liquid_r2', 'mean'),
                    median_between_liquid_r2=('between_liquid_r2', 'median'),
                    min_between_liquid_r2=('between_liquid_r2', 'min'),
                    max_between_liquid_r2=('between_liquid_r2', 'max'),
                    mean_criterion=('criterion_mean', 'mean'),
                    mean_n_factors=('n_factors', 'mean'),
                    mean_pc1_multi=('mean_pc1_explained_variance_multi', 'mean'),
                    n_distinct_configurations=(
                        'hyperparameters', lambda s: int(s.nunique())),
                    n_distinct_methods=('method', lambda s: int(s.nunique())))
               .reset_index()
               .sort_values('mean_between_liquid_r2', ascending=False,
                            ignore_index=True))

    summary['scope'] = scope
    summary['n_folds'] = int(per_fold['fold'].nunique())
    # Stated in the artefact itself, not only in this docstring: the ranking below was
    # computed on the outer folds and is therefore optimistic as a guide to a new fold.
    summary['ranking_is_diagnostic_only'] = True

    return per_fold, summary
