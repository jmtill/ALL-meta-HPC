"""Build every currently implemented feature representation and write the artefacts.

A driver, not a method: it wires ``config`` to the modules beside it and records what
came out. Everything it can decide is decided in ``config``; everything it learns from
data is fitted here and written down so a later stage can apply it without refitting.

Which frame fits what, and why:

* the unsupervised transforms -- global PCA, manual-cluster PC1, the clustering
  representations -- are fitted on ``config.CLUSTER_FIT_SUBSET`` (732 curated liquids
  in B, 1206 in A). Those subsets carry no target at all, so nothing fitted on them can
  leak the label, and a correlation structure estimated on 732 liquids is a real
  estimate where one estimated on 13 is noise.
* with ``config.EXCLUDE_LABELLED_FROM_CLUSTER_FIT`` the 13 benchmarked liquids are
  dropped from that frame first. They are members of A and B, so without this a
  transform has seen the rows -- not the labels, but the rows -- of the liquids it will
  later be scored on. Dropping them costs 13 of 732 and removes the objection instead
  of arguing about it.
* mRMR is the one selector that reads the target, so it is fitted on subset C and only
  ever on the rows it is handed. Here it is run once over all 13 labelled liquids to
  produce a *descriptive* artefact; that fit is not leakage-safe for scoring and is
  labelled as such in the output. The leakage-safe use is to call
  ``mrmr.select_mrmr`` inside a fold, which is what the nested evaluation of a later
  phase will do.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from allmeta import columns as cols
from allmeta.feature_selection import (
    algorithms, comparison, config, consensus, elastic_net, factors, manual, metrics,
    mrmr, nested, optimization, pca, plots, refinement, representations, stability)
from allmeta.feature_selection.provenance import Provenance
from allmeta.paths import FEATURE_SELECTION_DIR
from allmeta.subsets import SUBSET_C_PATH, load_subset

# A fresh directory: the existing data/feature_selection/{A,B,C}/ artefacts belong to
# the older screening pipeline and are left untouched so the two can be compared.
PIPELINE_DIR: Path = FEATURE_SELECTION_DIR / 'pipeline'

PROVENANCE_PATH: Path = PIPELINE_DIR / 'feature_provenance.csv'
MRMR_PATH: Path = PIPELINE_DIR / 'mrmr_results.csv'
LOADINGS_PATH: Path = PIPELINE_DIR / 'pca_loadings.csv'
VARIANCE_PATH: Path = PIPELINE_DIR / 'pca_explained_variance.csv'
CORRELATION_PATH: Path = PIPELINE_DIR / 'correlation_matrix.csv'
DISTANCE_PATH: Path = PIPELINE_DIR / 'correlation_distance.csv'
SUMMARY_PATH: Path = PIPELINE_DIR / 'representation_summary.csv'
CLUSTER_METRICS_PATH: Path = PIPELINE_DIR / 'clustering_internal_metrics.csv'
CLUSTER_MEMBERSHIP_PATH: Path = PIPELINE_DIR / 'cluster_membership_initial.csv'
PC1_QUALITY_PATH: Path = PIPELINE_DIR / 'clustering_pc1_quality.csv'
PC1_LOADINGS_PATH: Path = PIPELINE_DIR / 'clustering_pc1_loadings.csv'
STABILITY_PATH: Path = PIPELINE_DIR / 'clustering_stability_metrics.csv'
CONSENSUS_PATH: Path = PIPELINE_DIR / 'clustering_consensus.csv'
ASSIGNMENTS_PATH: Path = PIPELINE_DIR / 'clustering_feature_assignments.csv'
METHOD_AGREEMENT_PATH: Path = PIPELINE_DIR / 'clustering_method_agreement.csv'
SEARCH_PATH: Path = PIPELINE_DIR / 'clustering_hyperparameter_search.csv'
SELECTED_PATH: Path = (
    PIPELINE_DIR / 'selected_clustering_configs_by_outer_fold.csv')
SELECTION_SUMMARY_PATH: Path = PIPELINE_DIR / 'hyperparameter_stability_summary.csv'
METHOD_COMPARISON_PATH: Path = PIPELINE_DIR / 'clustering_method_comparison.csv'
CANDIDATE_COMPARISON_PATH: Path = (
    PIPELINE_DIR / 'clustering_method_comparison_full.csv')
ADMISSION_PATH: Path = PIPELINE_DIR / 'clustering_method_admission.csv'
CARRIED_PATH: Path = PIPELINE_DIR / 'clustering_carried_forward.csv'
PER_FOLD_SEARCH_PATH: Path = PIPELINE_DIR / 'clustering_search_by_outer_fold.csv'
TIE_BREAK_PATH: Path = PIPELINE_DIR / 'tie_break_comparison.csv'
TIE_BREAK_FOLD_PATH: Path = PIPELINE_DIR / 'tie_break_comparison_by_fold.csv'
EN_SEARCH_PATH: Path = PIPELINE_DIR / 'elastic_net_hyperparameter_search.csv'
EN_SELECTED_PATH: Path = PIPELINE_DIR / 'elastic_net_selected.csv'
EN_STABILITY_PATH: Path = PIPELINE_DIR / 'elastic_net_stability.csv'
EN_THRESHOLD_PATH: Path = PIPELINE_DIR / 'stability_threshold_search.csv'
# Phases 13-19.
POST_EN_MEMBERSHIP_PATH: Path = (
    PIPELINE_DIR / 'cluster_membership_post_elastic_net.csv')
CLUSTER_DIAGNOSTICS_PATH: Path = PIPELINE_DIR / 'cluster_diagnostics.csv'
RESIDUAL_CORRELATION_PATH: Path = PIPELINE_DIR / 'residual_factor_correlation.csv'
NESTED_PREDICTIONS_PATH: Path = PIPELINE_DIR / 'outer_fold_predictions.csv'
FOLD_PERFORMANCE_PATH: Path = PIPELINE_DIR / 'fold_performance.csv'
MODEL_PERFORMANCE_PATH: Path = PIPELINE_DIR / 'model_performance.csv'
FEATURE_SET_SUMMARY_PATH: Path = PIPELINE_DIR / 'feature_set_summary.csv'
NESTED_INTERVALS_PATH: Path = PIPELINE_DIR / 'outer_fold_intervals.csv'
NESTED_SELECTED_PATH: Path = (
    PIPELINE_DIR / 'selected_hyperparameters_by_outer_fold.csv')
FEATURE_OVERLAP_PATH: Path = PIPELINE_DIR / 'feature_overlap.csv'
CLUSTER_OVERLAP_PATH: Path = PIPELINE_DIR / 'cluster_overlap.csv'
FAMILY_COVERAGE_PATH: Path = PIPELINE_DIR / 'information_family_coverage.csv'
OUTER_SELECTION_FREQUENCY_PATH: Path = (
    PIPELINE_DIR / 'outer_fold_selection_frequency.csv')
FINAL_COMPARISON_PATH: Path = PIPELINE_DIR / 'final_one_se_comparison.csv'
PARETO_PATH: Path = PIPELINE_DIR / 'pareto_front.csv'
NOISE_DIAGNOSTICS_PATH: Path = PIPELINE_DIR / 'clustering_noise_diagnostics.csv'
CLUSTER_PREDICTIVE_PATH: Path = (
    PIPELINE_DIR / 'clustering_predictive_performance.csv')


def labelled_names() -> set[str]:
    """Names of the benchmarked liquids, or an empty set if subset C is absent.

    Empty rather than raising: subsets A and B build from tracked inputs alone, so the
    unsupervised half of this pipeline must stay runnable in a checkout that has no
    benchmarking data.
    """
    if not SUBSET_C_PATH.is_file():
        return set()

    return set(load_subset('C')['Name'])


def fitting_frame(subset: str | None = None,
                  exclude_labelled: bool | None = None) -> tuple[DataFrame, dict]:
    """The frame the unsupervised transforms are fitted on, and a note of what it is.

    The note travels with the frame because "fitted on 719 curated liquids, 13
    benchmarked ones excluded" is the claim the whole leakage argument rests on, and it
    should appear in the outputs rather than only in a docstring.
    """
    subset = config.CLUSTER_FIT_SUBSET if subset is None else subset
    exclude = (config.EXCLUDE_LABELLED_FROM_CLUSTER_FIT if exclude_labelled is None
               else exclude_labelled)

    frame = load_subset(subset)
    n_before = len(frame)

    excluded: set[str] = set()
    if exclude:
        excluded = labelled_names() & set(frame['Name'])
        frame = frame.loc[~frame['Name'].isin(excluded)].reset_index(drop=True)

    return frame, {
        'fit_subset': subset,
        'n_rows_available': n_before,
        'n_rows_used': len(frame),
        'excluded_labelled_liquids': len(excluded),
        'exclude_labelled': exclude,
    }


def build(subset: str | None = None, features: list[str] | None = None
          ) -> dict[str, object]:
    """Build every implemented representation and return the artefacts.

    ``features`` defaults to the non-composition descriptors, matching
    ``columns.screening_features()``: the atom counts are composition bookkeeping
    rather than measured liquid properties, and several are near-constant.
    """
    features = cols.screening_features() if features is None else list(features)

    frame, provenance_note = fitting_frame(subset)
    registry = Provenance()
    summary: list[dict] = []

    # 1. Manual raw subset.
    manual_values, manual_specs = manual.select_manual(frame)
    registry.extend(manual_specs)
    summary.append({'representation': 'manual_raw', 'n_inputs': manual_values.shape[1],
                    'fitted_on': provenance_note['fit_subset'], 'notes': ''})

    # 2. Direct PCA of the full raw feature set.
    pca_values, pca_loadings, pca_variance = pca.fit_global_pca(
        frame, features, n_components=config.FULL_PCA_N_COMPONENTS)
    registry.extend(pca.global_pca_specs(pca_loadings))
    summary.append({
        'representation': 'global_pca', 'n_inputs': pca_values.shape[1],
        'fitted_on': provenance_note['fit_subset'],
        'notes': f"{pca_variance['cumulative_explained_variance_ratio'].iloc[-1]:.4f} "
                 'cumulative explained variance',
    })

    # 3. Manual clusters -> PC1 per cluster.
    cluster_values, cluster_loadings = pca.fit_cluster_pc1(frame, config.MANUAL_CLUSTERS)
    registry.extend(pca.cluster_pc1_specs(
        cluster_loadings, source_method='manual_clusters',
        feature_type='manual_cluster_pc1'))
    weakest = cluster_loadings.groupby('factor')['explained_variance_ratio'].first().min()
    summary.append({
        'representation': 'manual_cluster_pc1', 'n_inputs': cluster_values.shape[1],
        'fitted_on': provenance_note['fit_subset'],
        'notes': f'lowest PC1 explained variance {weakest:.4f}',
    })

    # 4. The two clustering representations.
    rho, distance = representations.correlation_representation(frame, features)
    profiles, _ = representations.feature_profile_representation(frame, features)
    shape = representations.describe(distance)
    summary.append({
        'representation': 'correlation_distance', 'n_inputs': shape['n_features'],
        'fitted_on': provenance_note['fit_subset'],
        'notes': f"mean 1-|r| = {shape['mean_distance']:.4f}",
    })
    summary.append({
        'representation': 'feature_profile', 'n_inputs': profiles.shape[0],
        'fitted_on': provenance_note['fit_subset'],
        'notes': f'{profiles.shape[1]} liquid columns per feature',
    })

    # Concatenated for the written artefact, but kept apart in the return value: a
    # global PC's explained_variance_ratio is its share of the TOTAL variance, while a
    # cluster PC1's is its share of that cluster's. Plotting them on one axis compares
    # two different quantities, which is exactly what happened the first time.
    loadings = pd.concat([pca_loadings, cluster_loadings], ignore_index=True)

    return {
        'note': provenance_note,
        'provenance': registry,
        'manual_values': manual_values,
        'pca_values': pca_values,
        'pca_variance': pca_variance,
        'cluster_values': cluster_values,
        'loadings': loadings,
        'pca_loadings': pca_loadings,
        'cluster_loadings': cluster_loadings,
        'correlation': rho,
        'distance': distance,
        'profiles': profiles,
        'summary': DataFrame(summary),
        'distance_shape': shape,
    }


def run_mrmr() -> DataFrame | None:
    """Descriptive mRMR selection over all the labelled liquids, or None if absent.

    Fitted on every labelled liquid at once, so it is **not** a leakage-safe number and
    the ``fitted_on_all_labelled`` column says so in the file itself. It is here to show
    what mRMR does on this data, not to be scored. The scoreable version calls
    ``mrmr.select_mrmr`` with one fold's training rows.
    """
    if not SUBSET_C_PATH.is_file():
        return None

    frame = load_subset('C')
    features = cols.screening_features()
    _, results, _ = mrmr.select_mrmr(frame[features], frame[cols.TARGET])
    results['fitted_on_all_labelled'] = True
    results['n_labelled_liquids'] = frame['Name'].nunique()

    return results


def cluster_sweep(frame: DataFrame | None = None,
                  features: list[str] | None = None,
                  methods: list[str] | None = None,
                  ) -> tuple[DataFrame, DataFrame, list]:
    """Run every algorithm over its whole hyperparameter grid and score the results.

    Returns ``(metric_table, membership_table, solutions, pc1_quality, pc1_loadings)``.

    This is Phase 4 and Phase 5 only: every candidate is fitted and scored on cluster
    quality, and none is selected. Choosing among them needs the predictive criterion
    of Phase 8 -- the plan is explicit that a winner must not be picked on silhouette
    or cluster count, because the clusters exist to produce useful factors, not tidy
    partitions. What this function produces is the evidence that choice will be made
    from, plus a standalone record of how the five algorithms see the descriptor space
    with no supervised signal involved at all.

    Both correlation matrices are built because the correlation method is part of the
    hierarchical, HDBSCAN and K-Medoids grids: Pearson and Spearman give genuinely
    different distance matrices on this data, and which suits it is not knowable in
    advance. Each solution is scored against the distance it was fitted on.
    """
    features = cols.screening_features() if features is None else list(features)
    if frame is None:
        frame, _ = fitting_frame()

    methods = config.CLUSTERING_METHODS if methods is None else methods

    distances = {}
    rho_by_method = {}
    for correlation in representations.CORRELATION_METHODS:
        rho_by_method[correlation], distances[correlation] = (
            representations.correlation_representation(frame, features, correlation))
    profiles, _ = representations.feature_profile_representation(frame, features)

    # The domain metrics are computed in ONE fixed space for every solution, so that
    # "which method found the most coherent correlation families?" is a single
    # question. Spearman because several descriptors are monotonic but not linear in
    # each other, which Pearson understates.
    reference_rho = rho_by_method['spearman']

    solutions: list = []
    failures: list[dict] = []
    for method in methods:
        kind = config.REPRESENTATION_BY_METHOD[method]
        for params in algorithms.candidates(method, len(features)):
            correlation = params.get('correlation_method', 'spearman')
            representation = (distances[correlation] if kind == 'correlation_distance'
                              else profiles)
            try:
                solutions.append(algorithms.fit(method, representation, **params))
            except (ValueError, np.linalg.LinAlgError) as error:
                # Recorded, not swallowed: a grid entry that cannot be fitted is a fact
                # about the data, and a silently shorter sweep looks like a clean one.
                failures.append({'method': method,
                                 'hyperparameters': algorithms.format_params(params),
                                 'error': str(error)})

    table = metrics.evaluate_all(solutions, reference_rho, distances, profiles)

    # Phase 6 merged in here rather than run separately, because a clustering solution
    # and the factors it produces are one candidate: the Phase 8 selection needs the
    # cluster metrics and the PC1 retention side by side, and joining two tables later
    # on a hyperparameter string is a worse idea than never splitting them.
    pc1_quality, pc1_summary, pc1_loadings = factors.build_all(
        frame, solutions, reference_rho)
    table = table.merge(pc1_summary, on=['method', 'hyperparameters'], how='left')

    membership = DataFrame([
        {'method': solution.method,
         'hyperparameters': algorithms.format_params(solution.params),
         'cluster_id': label, 'cluster_name': name,
         'n_members': len(members), 'features': ', '.join(members)}
        for solution in solutions
        for (name, members), label in zip(
            solution.named_clusters(reference_rho).items(), solution.clusters())
    ])

    if failures:
        table = pd.concat([table, DataFrame(failures)], ignore_index=True)

    return table, membership, solutions, pc1_quality, pc1_loadings


def run(with_plots: bool = True,
        stability_preset: str | None = 'full',
        with_selection: bool = True,
        with_nested: bool = True) -> dict[str, object]:
    """Build everything, write the artefacts and figures, and return them.

    ``stability_preset`` selects how much Phase 7 resampling to do: ``'full'`` refits
    every candidate on 30 subsamples of the whole frame and takes a few minutes,
    ``'test'`` uses 5 subsamples of 200 liquids over a thinned grid and takes seconds,
    and ``None`` skips stability entirely. The preset's settings are written into every
    stability row, because a mean ARI over 5 resamples and one over 30 are not the same
    measurement and a table that does not say which it is invites them to be compared.
    """
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    artefacts = build()

    artefacts['provenance'].write(PROVENANCE_PATH)
    artefacts['loadings'].to_csv(LOADINGS_PATH, index=False)
    artefacts['pca_variance'].to_csv(VARIANCE_PATH, index=False)
    artefacts['correlation'].to_csv(CORRELATION_PATH)
    artefacts['distance'].to_csv(DISTANCE_PATH)
    artefacts['summary'].to_csv(SUMMARY_PATH, index=False)

    results = run_mrmr()
    if results is not None:
        results.to_csv(MRMR_PATH, index=False)
    artefacts['mrmr'] = results

    frame, _ = fitting_frame()
    cluster_metrics, membership, solutions, pc1_quality, pc1_loadings = cluster_sweep(
        frame)
    cluster_metrics.to_csv(CLUSTER_METRICS_PATH, index=False)
    membership.to_csv(CLUSTER_MEMBERSHIP_PATH, index=False)
    pc1_quality.to_csv(PC1_QUALITY_PATH, index=False)
    pc1_loadings.to_csv(PC1_LOADINGS_PATH, index=False)

    # The plan lists clustering_noise_diagnostics.csv separately. Only HDBSCAN can
    # produce noise -- every other method assigns every feature -- so this is a genuine
    # view of the sweep rather than a copy of it, and the other four methods appear with
    # a noise count of zero rather than being omitted. Omitting them would leave a
    # reader unable to tell "no noise" from "not applicable".
    noise_columns = [c for c in ('method', 'hyperparameters', 'representation',
                                 'n_features', 'n_clusters', 'n_noise',
                                 'noise_fraction', 'n_singletons',
                                 'singleton_fraction')
                     if c in cluster_metrics.columns]
    if noise_columns:
        noise = cluster_metrics[noise_columns].copy()
        noise['noise_is_applicable'] = noise['method'] == 'hdbscan'
        noise.to_csv(NOISE_DIAGNOSTICS_PATH, index=False)
        artefacts['noise_diagnostics'] = noise
    artefacts['cluster_metrics'] = cluster_metrics
    artefacts['cluster_membership'] = membership
    artefacts['solutions'] = solutions
    artefacts['pc1_quality'] = pc1_quality
    artefacts['pc1_loadings'] = pc1_loadings

    stability_table = None
    consensus_matrix = None
    if stability_preset is not None:
        settings = stability.StabilityConfig.preset(stability_preset)
        stability_table, consensus_matrix, assignments = stability.run(
            frame, settings=settings)
        agreement = stability.method_agreement(solutions)

        stability_table.to_csv(STABILITY_PATH, index=False)
        consensus_matrix.to_csv(CONSENSUS_PATH, index=False)
        assignments.to_csv(ASSIGNMENTS_PATH, index=False)
        agreement.to_csv(METHOD_AGREEMENT_PATH, index=False)

        artefacts['stability'] = stability_table
        artefacts['consensus'] = consensus_matrix
        artefacts['feature_assignments'] = assignments
        artefacts['method_agreement'] = agreement
        artefacts['stability_settings'] = settings

        # Merged so Phase 8 sees quality, retention and stability on one row per
        # candidate. A left join because a thinned stability sweep covers only part of
        # the grid, and the candidates it skipped must read as "not measured" rather
        # than silently vanish from the metric table.
        carried = [c for c in stability_table.columns
                   if c not in ('method', 'hyperparameters')]
        cluster_metrics = cluster_metrics.merge(
            stability_table[['method', 'hyperparameters'] + carried],
            on=['method', 'hyperparameters'], how='left')
        cluster_metrics.to_csv(CLUSTER_METRICS_PATH, index=False)
        artefacts['cluster_metrics'] = cluster_metrics

    if with_selection and SUBSET_C_PATH.is_file():
        labelled = load_subset('C')
        labelled_factors = optimization.build_labelled_factors(
            frame, labelled, solutions, artefacts['correlation'])

        search = optimization.search(labelled_factors)
        search['selected_on_all_labels'] = True
        competitive, chosen = optimization.one_standard_error(
            search, context=cluster_metrics)
        search = search.merge(
            competitive[['method', 'hyperparameters', 'within_one_se']],
            on=['method', 'hyperparameters'], how='left')
        search['within_one_se'] = search['within_one_se'].fillna(False)
        search.to_csv(SEARCH_PATH, index=False)

        # clustering_predictive_performance.csv: the plan asks for the predictive
        # scores of the clustering candidates as their own table, separate from the
        # hyperparameter search that produced them. Same rows, only the columns that
        # answer "how well did this clustering predict?" -- so a reader after the
        # performance does not have to pick it out of forty search columns.
        predictive_columns = [c for c in (
            'method', 'hyperparameters', 'n_factors', 'n_folds',
            'liquid_mse_mean', 'liquid_mse_std', 'liquid_mse_se', 'liquid_rmse',
            'pooled_rmse_mean', 'pooled_rmse_std', 'pooled_rmse_se', 'pooled_rmse',
            'mae_mean', 'pooled_r2', 'between_liquid_r2', 'selection_criterion',
            'beats_baseline', 'baseline_criterion_mean', 'within_one_se')
            if c in search.columns]
        search[predictive_columns].to_csv(CLUSTER_PREDICTIVE_PATH, index=False)

        selections, per_fold = optimization.select_by_outer_fold(
            labelled_factors, context=cluster_metrics)
        selections.to_csv(SELECTED_PATH, index=False)
        per_fold.to_csv(PER_FOLD_SEARCH_PATH, index=False)
        summary = optimization.selection_summary(selections)
        summary.to_csv(SELECTION_SUMMARY_PATH, index=False)

        # Phase 9. The per-method split has to happen before the methods can be
        # compared as methods: applied globally, the one-SE rule never asks what the
        # best candidate of each algorithm was on its own terms.
        winners, competitive = comparison.select_per_method(
            search, context=cluster_metrics)
        gate = comparison.admission(winners, cluster_metrics)
        method_comparison = comparison.method_comparison(
            search, cluster_metrics, winners=winners, gate=gate)
        candidates = comparison.candidate_comparison(
            search, cluster_metrics, competitive=competitive)

        gate.to_csv(ADMISSION_PATH, index=False)
        method_comparison.to_csv(METHOD_COMPARISON_PATH, index=False)
        candidates.to_csv(CANDIDATE_COMPARISON_PATH, index=False)

        carried = comparison.carry_forward(method_comparison)
        carried.to_csv(CARRIED_PATH, index=False)

        # Costs no model fits: every ordering is re-applied to the per-fold tables
        # computed just above.
        tie_break_folds, tie_break = comparison.replay_tie_breaks(
            per_fold, context=cluster_metrics)
        tie_break.to_csv(TIE_BREAK_PATH, index=False)
        tie_break_folds.to_csv(TIE_BREAK_FOLD_PATH, index=False)

        artefacts['method_winners'] = winners
        artefacts['method_admission'] = gate
        artefacts['method_comparison'] = method_comparison
        artefacts['candidate_comparison'] = candidates
        artefacts['carried_forward'] = carried
        artefacts['tie_break'] = tie_break
        artefacts['tie_break_by_fold'] = tie_break_folds

        # Phases 10-12. Every family is refined independently; comparing them against
        # each other needs the outer-fold numbers of Phase 14 and is not done here.
        families = elastic_net.labelled_representations(
            frame, labelled, artefacts, carried=carried, solutions=solutions)
        en_settings = (stability.StabilityConfig.preset(stability_preset)
                       if stability_preset is not None
                       else stability.StabilityConfig.preset('quick'))
        en_search, en_selected, en_stability, en_thresholds = elastic_net.refine_all(
            families, labelled_factors.target, labelled_factors.groups,
            settings=en_settings)

        en_search.to_csv(EN_SEARCH_PATH, index=False)
        en_selected.to_csv(EN_SELECTED_PATH, index=False)
        if not en_stability.empty:
            en_stability.to_csv(EN_STABILITY_PATH, index=False)
        if not en_thresholds.empty:
            en_thresholds.to_csv(EN_THRESHOLD_PATH, index=False)

        artefacts['feature_families'] = families
        artefacts['elastic_net_search'] = en_search
        artefacts['elastic_net_selected'] = en_selected
        artefacts['elastic_net_stability'] = en_stability
        artefacts['threshold_search'] = en_thresholds

        # Phase 13. Applied to the descriptive Elastic Net selection here; the
        # fold-local version runs inside Phase 14 and is the one that counts.
        retained = {}
        for record in en_selected.to_dict('records'):
            name = record['feature_set']
            if not record.get('selected') or name not in families:
                continue
            fitted = elastic_net.coefficients(
                families[name], labelled_factors.target,
                float(record['alpha']), float(record['l1_ratio']))
            retained[name] = fitted.loc[fitted['selected'], 'feature'].tolist()

        post_diagnostics, post_membership, residual = refinement.refine_all(
            families, retained, fit_frame=frame)
        if not post_diagnostics.empty:
            post_diagnostics.to_csv(CLUSTER_DIAGNOSTICS_PATH, index=False)
        if not post_membership.empty:
            post_membership.to_csv(POST_EN_MEMBERSHIP_PATH, index=False)
        if not residual.empty:
            residual.to_csv(RESIDUAL_CORRELATION_PATH, index=False)
        artefacts['post_elastic_net'] = post_diagnostics
        artefacts['post_elastic_net_membership'] = post_membership
        artefacts['residual_correlation'] = residual

        if with_nested:
            # Phases 14-19. The only numbers in this pipeline that estimate anything.
            # Checkpointed: this loop is ~35 minutes of single-threaded work, and an
            # interrupted run would otherwise start from zero. Delete the directory to
            # force a clean recomputation.
            predictions, nested_selections, _ = nested.run_outer(
                frame, labelled, artefacts, solutions=solutions,
                cluster_metrics=cluster_metrics, labelled_factors=labelled_factors,
                checkpoint_dir=PIPELINE_DIR / 'nested_checkpoints', verbose=True)

            performance = nested.score(predictions)
            per_fold_performance = nested.fold_scores(predictions)
            interval_table = nested.intervals(predictions)
            feature_summary = nested.summary(performance, nested_selections,
                                             interval_table)

            predictions.to_csv(NESTED_PREDICTIONS_PATH, index=False)
            per_fold_performance.to_csv(FOLD_PERFORMANCE_PATH, index=False)
            performance.to_csv(MODEL_PERFORMANCE_PATH, index=False)
            interval_table.to_csv(NESTED_INTERVALS_PATH, index=False)
            feature_summary.to_csv(FEATURE_SET_SUMMARY_PATH, index=False)
            nested_selections.to_csv(NESTED_SELECTED_PATH, index=False)

            # Phases 18-19.
            mapping = consensus.family_of(membership)
            overlap = consensus.feature_overlap(nested_selections)
            families_agreement = consensus.family_agreement(nested_selections, mapping)
            coverage = consensus.family_coverage(nested_selections, mapping)
            frequency = consensus.selection_frequency(nested_selections)
            final = consensus.final_comparison(per_fold_performance, feature_summary)
            front = consensus.pareto(feature_summary)

            for table, path in ((overlap, FEATURE_OVERLAP_PATH),
                                (families_agreement, CLUSTER_OVERLAP_PATH),
                                (coverage, FAMILY_COVERAGE_PATH),
                                (frequency, OUTER_SELECTION_FREQUENCY_PATH),
                                (final, FINAL_COMPARISON_PATH),
                                (front, PARETO_PATH)):
                if not table.empty:
                    table.to_csv(path, index=False)

            artefacts.update({
                'outer_predictions': predictions,
                'outer_selections': nested_selections,
                'model_performance': performance,
                'fold_performance': per_fold_performance,
                'outer_intervals': interval_table,
                'feature_set_summary': feature_summary,
                'feature_overlap': overlap,
                'family_agreement': families_agreement,
                'family_coverage': coverage,
                'outer_selection_frequency': frequency,
                'final_comparison': final,
                'pareto': front,
            })

        artefacts['search'] = search
        artefacts['competitive'] = competitive
        artefacts['chosen'] = chosen
        artefacts['selections'] = selections
        artefacts['per_fold_search'] = per_fold
        artefacts['selection_summary'] = summary
        artefacts['random_candidate'] = optimization.random_candidate_baseline(search)

    if with_plots:
        artefacts['figures'] = plots.plot_all(
            cluster_metrics, artefacts['correlation'], artefacts['distance'],
            artefacts['cluster_loadings'], pc1_quality=pc1_quality,
            stability=stability_table, consensus=consensus_matrix,
            search=artefacts.get('search'),
            selections=artefacts.get('selections'),
            method_comparison=artefacts.get('method_comparison'),
            candidate_comparison=artefacts.get('candidate_comparison'),
            admission=artefacts.get('method_admission'),
            tie_break=artefacts.get('tie_break'),
            tie_break_by_fold=artefacts.get('tie_break_by_fold'),
            elastic_net_search=artefacts.get('elastic_net_search'),
            elastic_net_stability=artefacts.get('elastic_net_stability'),
            threshold_search=artefacts.get('threshold_search'),
            fold_performance=artefacts.get('fold_performance'),
            final_comparison=artefacts.get('final_comparison'),
            feature_set_summary=artefacts.get('feature_set_summary'),
            pareto=artefacts.get('pareto'),
            family_agreement=artefacts.get('family_agreement'),
            family_coverage=artefacts.get('family_coverage'))

    return artefacts
