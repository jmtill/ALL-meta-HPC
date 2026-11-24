"""``python -m allmeta.feature_selection`` -- build and report every representation.

    --stability full    20 subsamples, whole grid, every liquid (~13 minutes)
    --stability quick   10 subsamples, whole grid, every liquid (~6 minutes)
    --stability test     5 subsamples, 6 candidates/method, 200 liquids (~7 seconds)
    --stability none    skip Phase 7
    --no-selection      skip the Phase 8 predictive search (~3 minutes)
    --no-nested         skip the Phase 14-19 nested cross-validation (~20 minutes).
                        Everything before it is a search diagnostic; this is the only
                        stage that produces a performance estimate.
    --no-plots          skip the figures
"""

import sys

from allmeta.feature_selection import config, metrics
from allmeta.feature_selection.pipeline import PIPELINE_DIR, run
from allmeta.feature_selection.plots import FIG_DIR


def _argument(flag: str, default: str) -> str:
    """Read ``--flag value`` from the command line."""
    if flag not in sys.argv:
        return default
    position = sys.argv.index(flag) + 1
    if position >= len(sys.argv):
        raise SystemExit(f'{flag} needs a value')

    return sys.argv[position]


if __name__ == '__main__':
    preset = _argument('--stability', 'full')
    if preset not in set(config.STABILITY_PRESETS) | {'none'}:
        raise SystemExit(f'--stability must be one of '
                         f'{sorted(set(config.STABILITY_PRESETS) | {"none"})}')

    for warning in config.validate():
        print(f'config warning: {warning}')

    artefacts = run(with_plots='--no-plots' not in sys.argv,
                    stability_preset=None if preset == 'none' else preset,
                    with_selection='--no-selection' not in sys.argv,
                    with_nested='--no-nested' not in sys.argv)
    note = artefacts['note']

    print(f"\nfitted on subset {note['fit_subset']}: {note['n_rows_used']} of "
          f"{note['n_rows_available']} liquids "
          f"({note['excluded_labelled_liquids']} benchmarked liquids excluded)\n")
    print(artefacts['summary'].to_string(index=False))
    print(f"\n{len(artefacts['provenance'])} model inputs registered")
    print(artefacts['provenance'].to_frame()
          .groupby('feature_type', as_index=False)
          .agg(n=('feature_name', 'size'))
          .to_string(index=False))

    results = artefacts['mrmr']
    if results is None:
        print('\nsubset C is absent; mRMR skipped')
    else:
        chosen = results[results['selected']].sort_values('rank')
        print(f"\nmRMR ({config.MRMR_METHOD}) on all "
              f"{int(results['n_labelled_liquids'].iloc[0])} labelled liquids "
              '-- descriptive only, not leakage-safe:')
        print('  ' + ', '.join(chosen['feature']))

    table = artefacts['cluster_metrics']
    scored = table.dropna(subset=['n_clusters'])
    failed = len(table) - len(scored)
    print(f'\n{len(scored)} clustering solutions fitted and scored'
          + (f', {failed} grid entries could not be fitted' if failed else ''))
    print(scored.groupby('method').agg(
        candidates=('n_clusters', 'size'),
        median_k=('n_clusters', 'median'),
        median_delta=('within_between_delta', 'median'),
        median_coverage=('within_pair_fraction', 'median'),
        median_singletons=('singleton_fraction', 'median'),
    ).round(3).to_string())

    # Deliberately not a winner. Selecting on cluster quality is what Phase 8 exists to
    # replace, and the solution that maximises the correlation separation here does it
    # by making almost everything a singleton.
    best = metrics.rank(scored, 'within_between_delta').iloc[0]
    print(f"\nhighest within_between_delta: {best['method']} "
          f"({best['hyperparameters']})")
    print(f"  delta {best['within_between_delta']:.3f} over only "
          f"{best['within_pair_fraction']:.1%} of feature pairs, "
          f"{best['n_singletons']}/{best['n_clusters']} clusters are singletons "
          '-- it scores well by declining to cluster, which is why the winner is '
          'chosen on predictive performance in Phase 8, not here.')

    pc1 = artefacts['pc1_quality']
    multi = pc1[~pc1['is_singleton']]
    print(f"\ncluster PC1: {len(pc1)} clusters, {len(multi)} multi-member, "
          f"{int(multi['flagged'].sum())} below the "
          f"{config.PC1_RETENTION_FLOOR:.2f} retention floor")
    print(scored.groupby('method').agg(
        median_pc1_all=('mean_pc1_explained_variance', 'median'),
        median_pc1_multi=('mean_pc1_explained_variance_multi', 'median'),
        median_min_pc1=('min_pc1_explained_variance', 'median'),
        median_flagged=('flagged_fraction', 'median'),
    ).round(3).to_string())
    print('  median_pc1_all is inflated by singletons, which retain 1.00 by '
          'construction; read median_pc1_multi.')

    if 'stability' in artefacts:
        settings = artefacts['stability_settings']
        scope = ('all liquids' if settings.frame_size is None
                 else f'{settings.frame_size} liquids')
        stable = artefacts['stability'].dropna(subset=['ari_mean'])
        print(f"\nstability over {settings.n_resamples} subsamples ({scope}, "
              f"{settings.subsample_fraction:.0%} of rows each), "
              f"{len(stable)} candidates measured:")
        print(stable.groupby('method').agg(
            mean_ari=('ari_mean', 'mean'),
            mean_ami=('ami_mean', 'mean'),
            mean_nmi=('nmi_mean', 'mean'),
            undecided_pairs=('consensus_ambiguous_fraction', 'mean'),
            sd_n_clusters=('std_n_clusters', 'mean'),
        ).round(3).to_string())
        if preset == 'test':
            print('  --stability test is a smoke test: 5 subsamples give 10 pairs and '
                  'a mean ARI with no useful precision. Use --stability full to rank.')

        print('\nagreement BETWEEN the algorithms (median-k solution of each):')
        print(artefacts['method_agreement'].round(3).to_string(index=False))

    if 'search' in artefacts:
        from allmeta.feature_selection.optimization import BASELINE_NAME

        search = artefacts['search']
        criterion = f'{config.SELECTION_CRITERION}_mean'
        candidates = search[search['method'] != BASELINE_NAME]
        baseline = search.loc[search['method'] == BASELINE_NAME, criterion].iloc[0]
        chosen = artefacts['chosen']
        drawn = artefacts['random_candidate']

        print(f'\npredictive search on {config.SELECTION_CRITERION} '
              f'({len(candidates)} candidates, leave-one-liquid-out):')
        print(f'  mean baseline                {baseline:.6f}')
        print(f'  best candidate               {candidates[criterion].min():.6f}')
        print(f'  median candidate             {candidates[criterion].median():.6f}')
        print(f"  a randomly chosen candidate  {drawn['random_median']:.6f} "
              f"(10-90%: {drawn['random_p10']:.6f}-{drawn['random_p90']:.6f})")
        print(f'  candidates beating baseline  '
              f"{int(candidates['beats_baseline'].sum())}/{len(candidates)}")

        print(f'\n  chosen by the one-SE rule: {chosen["method"]} '
              f'({chosen["hyperparameters"]})')
        print(f'    {int(chosen["n_factors"])} factors, '
              f'{criterion} {chosen[criterion]:.6f}, '
              f'between-liquid R2 {chosen["between_liquid_r2"]:.4f}')
        print(f'    competitive set: {len(artefacts["competitive"])} candidates '
              f'within one SE of the best')

        summary = artefacts['selection_summary']
        n_configs = int(summary['n_distinct_configurations'].iloc[0])
        n_folds = int(summary['n_folds'].iloc[0])
        print(f'\n  leakage-safe per-fold selection: {n_configs} distinct '
              f'configurations won across {n_folds} folds')
        by_method = summary[summary['scope'] == 'method']
        print('    ' + ', '.join(
            f"{r.method} {r.n_folds_selected}/{n_folds}"
            for r in by_method.itertuples()))
        print('    The winner above is a minimum over hundreds of noisy estimates and '
              'is NOT a\n    performance estimate; that comes from the outer folds in '
              'Phase 14.')

    comparison_table = artefacts.get('method_comparison')
    if comparison_table is not None and not comparison_table.empty:
        print('\nPhase 9 -- the methods compared as methods:')
        print(f'  {"method":13s} {"beats base":>10s} {"admitted":>9s}  reason')
        winners = artefacts['method_winners'].set_index('method')
        for record in comparison_table.to_dict('records'):
            method = record['clustering_method']
            beats = int(winners.loc[method, 'n_beats_baseline']) \
                if method in winners.index else 0
            total = int(winners.loc[method, 'n_candidates']) \
                if method in winners.index else 0
            flag = 'yes' if record.get('admitted') else 'no'
            print(f'  {method:13s} {f"{beats}/{total}":>10s} {flag:>9s}  '
                  f'{record.get("failed_criteria", "") or ""}')

        carried = artefacts.get('carried_forward')
        if carried is not None and not carried.empty:
            print(f'\n  carried forward ({config.CLUSTERING_COMPARISON_MODE}): '
                  + ', '.join(carried['feature_set_name']))

        tie_break = artefacts.get('tie_break')
        if tie_break is not None and not tie_break.empty:
            print('\n  tie-break replay (no refits; DIAGNOSTIC -- computed on the '
                  'folds it is scored on):')
            for row in tie_break.to_dict('records'):
                print(f"    {row['ordering']:16s} mean held-out between-liquid R2 "
                      f"{row['mean_between_liquid_r2']:+.4f}   "
                      f"{int(row['n_distinct_configurations'])} distinct winners")
            print('    Read the spread, not the ranking: the gap between orderings is '
                  'far smaller\n    than the fold-to-fold spread within any one of '
                  'them.')

    en_selected = artefacts.get('elastic_net_selected')
    if en_selected is not None and not en_selected.empty:
        search = artefacts['elastic_net_search']
        print('\nPhases 10-12 -- Elastic Net, one refinement per feature family:')
        for record in en_selected.to_dict('records'):
            name = record['feature_set']
            block = search[(search['feature_set'] == name)
                           & (search['method'] != 'mean_baseline')]
            beaten = int(block['beats_baseline'].sum())
            if not record.get('selected'):
                print(f'  {name:20s} {record["n_inputs"]:2d} inputs   '
                      f'{beaten}/{len(block)} beat baseline -- nothing selected')
                continue
            print(f'  {name:20s} {record["n_inputs"]:2d} inputs   '
                  f'{beaten}/{len(block)} beat baseline')
            print(f'    alpha {record["alpha"]:.5g}, L1 {record["l1_ratio"]:g} -> '
                  f'{record["n_nonzero_mean"]:.1f} of {record["n_inputs"]} retained '
                  f'({record["retained_fraction"]:.0%})')
            print(f'    {record["criterion_mean"]:.6f} +/- {record["criterion_se"]:.6f}'
                  f'   held-out between-liquid R2 {record["between_liquid_r2"]:+.4f}'
                  f'   band: {int(record["n_competitive"])} candidates')

        thresholds = artefacts.get('threshold_search')
        if thresholds is not None and not thresholds.empty:
            print('\n  stability threshold, nested vs diagnostic (higher nested error '
                  'is the leak being removed):')
            criterion = f'{config.SELECTION_CRITERION}_mean'
            for name, block in thresholds.groupby('feature_set'):
                nested = block[block['scope'] == 'nested'][criterion].min()
                leaky = block[block['scope'] == 'diagnostic'][criterion].min()
                ratio = (nested / leaky) if leaky and leaky == leaky else float('nan')
                print(f'    {name:20s} best nested {nested:.6f}  vs diagnostic '
                      f'{leaky:.6f}   ({ratio:.1f}x optimism)')

    post = artefacts.get('post_elastic_net')
    if post is not None and not post.empty:
        print('\nPhase 13 -- what each family gets after Elastic Net:')
        for record in post.to_dict('records'):
            worst = record.get('max_abs_correlation')
            worst_text = ('n/a' if worst != worst or worst is None
                          else f'{worst:.2f}')
            print(f'  {record["feature_set"]:26s} {record["treatment"]:20s} '
                  f'{record["n_retained"]:2d} retained, worst residual |r| {worst_text}')

    summary_table = artefacts.get('feature_set_summary')
    if summary_table is not None and not summary_table.empty:
        n_liquids = int(summary_table.get('n_liquids', [13])[0])
        baseline = 1 - (n_liquids / (n_liquids - 1)) ** 2
        print('\nPhases 14-15 -- NESTED CROSS-VALIDATION (the honest numbers):')
        print(f'  mean baseline between-liquid R2 at n={n_liquids}: {baseline:+.4f} '
              '(NOT zero)')
        print(f'  {"feature set":26s} {"between-R2":>10s} {"95% CI":>18s} '
              f'{"RMSE":>8s} {"feats":>6s}')
        for record in summary_table.to_dict('records'):
            lo, hi = record.get('ci_lo'), record.get('ci_hi')
            ci = ('n/a' if lo is None or lo != lo
                  else f'[{lo:+.2f}, {hi:+.2f}]')
            beats = ' *' if record['between_liquid_r2'] > baseline else '  '
            print(f'  {record["feature_set"]:26s} '
                  f'{record["between_liquid_r2"]:+10.4f}{beats}{ci:>16s} '
                  f'{record["rmse"]:8.4f} {record.get("n_features_median", 0):6.0f}')
        print('  * beats the mean baseline')

    final = artefacts.get('final_comparison')
    if final is not None and not final.empty:
        informative = bool(final['one_se_is_informative'].iloc[0])
        width = float(final['band_width_over_spread'].iloc[0])
        print('\nPhase 19 -- final one-standard-error comparison:')
        print(f'  band width / spread across pipelines: {width:.2f}')
        if informative:
            inside = final[final['within_one_se']]
            print(f'  {len(inside)} of {len(final)} pipelines inside the band: '
                  + ', '.join(inside['feature_set']))
        else:
            print('  the band contains EVERY pipeline -- they are not distinguishable')
            print('  at this sample size. Any ordering inside it is noise.')

    agreement = artefacts.get('family_agreement')
    if agreement is not None and not agreement.empty:
        print('\nPhase 18 -- do the pipelines agree on descriptors or on chemistry?')
        for record in agreement.to_dict('records'):
            marker = (' <- same families, different columns'
                      if record['family_agreement_exceeds_feature_agreement'] else '')
            print(f'  {record["feature_set_a"]:22s} vs {record["feature_set_b"]:22s} '
                  f'descriptor {record["feature_jaccard"]:.2f}  '
                  f'family {record["family_jaccard"]:.2f}{marker}')

    print(f'\nwrote artefacts to {PIPELINE_DIR}')
    print(f'wrote {len(artefacts["figures"])} figures to {FIG_DIR}')
