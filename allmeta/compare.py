"""Compare the feature sets produced by every method on every subset.

Three things a reader wants at the end: which descriptors survive regardless of how
you select them, how much the liquid curation changed the answer, and whether the
smaller feature sets actually cluster better or merely differently. This module
answers those from the artefacts the earlier stages wrote, and fits nothing itself.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from pandas import DataFrame

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from allmeta import columns as cols  # noqa: E402
from allmeta.clustering import METRICS_PATH  # noqa: E402
from allmeta.paths import COMPARISON_DIR, FEATURE_SELECTION_DIR, FIG_COMPARISON  # noqa: E402
from allmeta.plotting import save  # noqa: E402
from allmeta.screening import load_factor_loadings, load_feature_sets  # noqa: E402
from allmeta.subsets import labelled_liquid_count  # noqa: E402
from allmeta.regression import (  # noqa: E402
    ALIASES_PATH, BOOTSTRAP_CI_PATH, SCORES_NESTED_PATH, SCORES_PATH,
    baseline_r2, of_evaluation)
from allmeta.supervised import FREQUENCY_PATH, MIN_FREQUENCY  # noqa: E402

# A feature must be selected in at least this fraction of leave-one-liquid-out folds
# to count as chosen by a supervised method. Re-exported from supervised so the two
# modules cannot disagree about what "chosen" means.
SUPERVISED_MIN_FREQUENCY: float = MIN_FREQUENCY

# Metrics where a larger value is better. Davies-Bouldin is the exception and is
# handled by inverting the sort rather than the value.
HIGHER_IS_BETTER: tuple[str, ...] = (
    'silhouette', 'calinski_harabasz', 'ari_chemical', 'ami_chemical',
    'stability_ari', 'error_eta_squared',
)
LOWER_IS_BETTER: tuple[str, ...] = ('davies_bouldin',)

MATRIX_PATH: Path = COMPARISON_DIR / 'feature_survival_matrix.csv'
JACCARD_PATH: Path = COMPARISON_DIR / 'feature_set_jaccard.csv'
BEST_PATH: Path = COMPARISON_DIR / 'best_by_metric.csv'
SUMMARY_PATH: Path = COMPARISON_DIR / 'summary.md'
METHOD_COMPARISON_PATH: Path = COMPARISON_DIR / 'selection_method_comparison.csv'
FAMILY_RANKING_PATH: Path = COMPARISON_DIR / 'feature_set_ranking_by_family.csv'


def supervised_feature_sets(min_frequency: float = SUPERVISED_MIN_FREQUENCY
                            ) -> dict[str, list[str]]:
    """One feature set per supervised method, from the descriptor-only run.

    The pipetting parameters are excluded: they describe the protocol, not the
    liquid, and a feature set that includes them cannot be used to cluster liquids.
    """
    if not FREQUENCY_PATH.is_file():
        return {}

    frequency = pd.read_csv(FREQUENCY_PATH)
    frequency = frequency[~frequency['include_params'].astype(bool)]

    sets: dict[str, list[str]] = {}
    for method, group in frequency.groupby('method'):
        chosen = sorted(
            set(group.loc[group['frequency'] >= min_frequency, 'feature'])
            - set(cols.PIPETTING_PARAMS)
        )
        if chosen:
            sets[method] = chosen

    return sets


def collect_feature_sets() -> dict[str, list[str]]:
    """Every named feature set, keyed ``subset::method``."""
    collected: dict[str, list[str]] = {}

    for subset in ('A', 'B'):
        path = FEATURE_SELECTION_DIR / subset / 'feature_sets.json'
        if not path.is_file():
            continue
        for name, members in load_feature_sets(subset).items():
            collected[f'{subset}::{name}'] = list(members)

    for name, members in supervised_feature_sets().items():
        collected[f'C::{name}'] = members

    # The consensus set is *selected* on subset C but *evaluated* on A and B, so it
    # is registered once under C rather than duplicated per evaluation subset.
    from allmeta.supervised import consensus_features

    if FREQUENCY_PATH.is_file():
        consensus = consensus_features(pd.read_csv(FREQUENCY_PATH))
        if consensus:
            collected['C::supervised_consensus'] = consensus

    return collected


def factor_members(subsets: tuple[str, ...] = ('A', 'B')) -> dict[str, list[str]]:
    """Map each factor column to the descriptors it aggregates.

    Factor columns are named ``A::hier_factor_Ant1T`` -- a name that matches no entry
    in ``columns.NUM_COLUMNS``. Names are already namespaced by subset and clustering
    method, so one lookup covers A and B without collision.
    """
    loadings = load_factor_loadings(subsets)
    if loadings.empty:
        return {}

    return loadings.groupby('factor')['feature'].apply(list).to_dict()


def expand_factor_sets(feature_sets: dict[str, list[str]],
                       subsets: tuple[str, ...] = ('A', 'B')) -> dict[str, list[str]]:
    """Replace every factor column by the descriptors it is built from.

    Without this, the six factor sets -- ``{A,B}::factor_{hierarchical,spectral,
    agglomerative}`` -- contribute an all-``False`` column to the survival matrix
    while still counting toward its denominator. That is not a rounding error: it
    caps achievable survival at 25/31 = 80.6% when the reported threshold is 60%
    (19/31), and it is a large part of why ``summary.md`` reported
    "Core descriptors: (none)" for a set of screens that in fact agree on several.

    A factor built from fourteen descriptors *did* retain those fourteen descriptors;
    it just carries them as a linear combination rather than as columns. Counting it
    as having dropped all of them is the wrong reading of what the screen did.

    Lifted out of ``notebooks/2_manual_feature_filtering.ipynb``, which already had
    this logic and where it was about to become a fifth diverged copy of a column
    list -- the exact failure ``columns.py``'s own docstring was written to record.
    """
    members_of = factor_members(subsets)
    if not members_of:
        return {name: list(members) for name, members in feature_sets.items()}

    expanded: dict[str, list[str]] = {}
    for name, members in feature_sets.items():
        out: list[str] = []
        for member in members:
            for feature in members_of.get(member, [member]):
                if feature not in out:
                    out.append(feature)
        expanded[name] = out

    return expanded


def assemble_matrix(feature_sets: dict[str, list[str]]) -> DataFrame:
    """Feature x feature-set survival table, ordered by how widely each survives.

    ``n_sets_selected`` deliberately excludes the trivial sets ('all',
    'no_composition'), which would otherwise credit every descriptor twice per
    subset for surviving a screen that dropped nothing.

    Factor sets are expanded into their member descriptors before membership is
    tested. Indexed on raw column names, as this was, the six factor sets matched
    nothing at all and contributed all-``False`` columns while still counting in the
    denominator -- so no descriptor could exceed 25 of 31 and the 60% threshold could
    not be met by construction.
    """
    trivial = {'all', 'no_composition'}
    scored = {k: v for k, v in feature_sets.items()
              if k.split('::', 1)[1] not in trivial}

    # Factor columns are expanded into their member descriptors first; see
    # expand_factor_sets for why counting them as all-dropped was wrong.
    resolved = expand_factor_sets(feature_sets)

    matrix = DataFrame(
        {name: [feature in set(resolved.get(name, members))
                for feature in cols.NUM_COLUMNS]
         for name, members in feature_sets.items()},
        index=list(cols.NUM_COLUMNS),
    )
    matrix.index.name = 'feature'

    matrix.insert(0, 'n_sets_selected',
                  matrix[list(scored)].sum(axis=1) if scored else 0)
    matrix.insert(1, 'n_scored_sets', len(scored))

    return matrix.sort_values('n_sets_selected', ascending=False)


def jaccard_matrix(feature_sets: dict[str, list[str]]) -> DataFrame:
    """Pairwise Jaccard similarity between every pair of feature sets."""
    names = list(feature_sets)
    values = np.eye(len(names))

    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if j <= i:
                continue
            a, b = set(feature_sets[left]), set(feature_sets[right])
            score = len(a & b) / len(a | b) if (a | b) else np.nan
            values[i, j] = values[j, i] = score

    return DataFrame(values, index=names, columns=names)


def best_by_metric(metrics: DataFrame) -> DataFrame:
    """Best (subset, feature set, representation, model) for each metric.

    Rows that failed to produce a partition, and metrics that are undefined for a
    degenerate partition, are dropped rather than ranked as zero.
    """
    # n_labelled and n_labelled_clusters travel with the row because
    # error_eta_squared cannot be read without them: it rests on the labelled liquids
    # alone, and a handful spread over almost as many clusters scores near 1.0
    # whatever the errors do. A
    # reader taking this table on its own would otherwise see 0.997 with no n.
    keys = ['subset', 'feature_set', 'n_features', 'representation', 'model',
            'n_clusters', 'n_labelled', 'n_labelled_clusters']

    rows: list[dict] = []
    for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        if metric not in metrics.columns:
            continue
        valid = metrics.dropna(subset=[metric])
        if valid.empty:
            continue
        ascending = metric in LOWER_IS_BETTER
        best = valid.sort_values(metric, ascending=ascending).iloc[0]
        rows.append({'metric': metric, 'value': float(best[metric]),
                     **{k: best.get(k) for k in keys}})

    return DataFrame(rows)


def feature_set_ranking(metrics: DataFrame) -> DataFrame:
    """Median of each metric per (subset, feature set), across the model grid.

    Median rather than max: the best single model on a feature set says more about
    the model than about the features, whereas the median over a fixed grid is a
    fair comparison between feature sets.
    """
    available = [m for m in HIGHER_IS_BETTER + LOWER_IS_BETTER
                 if m in metrics.columns]

    return (
        metrics.groupby(['subset', 'feature_set', 'n_features'], as_index=False)[available]
        .median()
        .sort_values(['subset', 'silhouette'], ascending=[True, False],
                     ignore_index=True)
    )


# DBSCAN/HDBSCAN can refuse to assign points (noise) where every other model in the
# grid must assign every point to some cluster; kept as a separate family rather
# than folded into feature_set_ranking()'s single grid-wide median so a lucky or
# degenerate density-based result cannot quietly shift a feature set's rank.
DENSITY_PREFIXES: tuple[str, ...] = ('dbscan', 'hdbscan')


def model_family(model: str) -> str:
    """'density' for DBSCAN/HDBSCAN, 'partition' for everything else in the grid."""
    return 'density' if model.startswith(DENSITY_PREFIXES) else 'partition'


def family_ranking(metrics: DataFrame) -> DataFrame:
    """Median of each metric per (subset, feature set, model family).

    The density-vs-partition counterpart to feature_set_ranking(): reports the two
    families' verdicts side by side instead of blending them into one median, so a
    feature set that only looks good because density-based clustering called most
    of it noise is visible as a disagreement rather than hidden in an average.
    """
    available = [m for m in HIGHER_IS_BETTER + LOWER_IS_BETTER
                 if m in metrics.columns]
    view = metrics.copy()
    view['model_family'] = view['model'].map(model_family)

    return (
        view.groupby(['subset', 'feature_set', 'model_family'], as_index=False)[available]
        .median()
        .sort_values(['subset', 'model_family', 'silhouette'],
                     ascending=[True, True, False], ignore_index=True)
    )


def plot_jaccard(jaccard: DataFrame, out_path: Path) -> None:
    """Heatmap of feature-set overlap."""
    fig, ax = plt.subplots(figsize=(max(9, len(jaccard) * 0.6),
                                    max(8, len(jaccard) * 0.55)))
    sns.heatmap(jaccard, vmin=0, vmax=1, cmap='viridis', square=True,
                annot=len(jaccard) <= 24, fmt='.2f', annot_kws={'size': 7},
                cbar_kws={'label': 'Jaccard similarity'}, ax=ax)
    ax.set_title('Overlap between selected feature sets')
    ax.tick_params(axis='x', labelrotation=90, labelsize=8)
    ax.tick_params(axis='y', labelrotation=0, labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='w')
    plt.close(fig)


def plot_survival(matrix: DataFrame, out_path: Path) -> None:
    """Bar of how many feature sets retained each descriptor."""
    counts = matrix['n_sets_selected'].sort_values()
    total = int(matrix['n_scored_sets'].iloc[0]) if len(matrix) else 0

    fig, ax = plt.subplots(figsize=(9, max(8, len(counts) * 0.2)))
    ax.barh(counts.index, counts.values, color='#4c72b0')
    ax.set_xlabel(f'Feature sets retaining the descriptor (of {total})')
    ax.set_ylabel('Feature')
    ax.set_title('Descriptor survival across selection methods and subsets')
    ax.tick_params(axis='y', labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='w')
    plt.close(fig)


def plot_metric_by_feature_set(ranking: DataFrame, out_path: Path) -> None:
    """Median clustering metrics per feature set, one panel per metric."""
    panels = [m for m in ('silhouette', 'ari_chemical', 'stability_ari',
                          'davies_bouldin') if m in ranking.columns]
    if not panels:
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 4 * len(panels)))
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, panels):
        view = ranking.dropna(subset=[metric]).copy()
        view['label'] = view['subset'] + '::' + view['feature_set']
        view = view.sort_values(metric, ascending=metric in LOWER_IS_BETTER)
        sns.barplot(data=view, x='label', y=metric, hue='subset', ax=ax,
                    dodge=False)
        ax.set_title(f'median {metric} by feature set')
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelrotation=90, labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='w')
    plt.close(fig)


def plot_family_agreement(family: DataFrame, out_path: Path) -> None:
    """Median silhouette and chemical agreement per feature set, partition vs density.

    Density-based clustering is free to call weak structure noise instead of
    forcing a cluster onto it, so it can reach a different verdict than the
    always-partitions family on which feature set is best. Bars ordered by the
    partition family's value so the two families' agreement (or lack of it) reads
    directly off the picture.
    """
    panels = [m for m in ('silhouette', 'ari_chemical') if m in family.columns]
    if not panels or family.empty:
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 4.5 * len(panels)))
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, panels):
        view = family.dropna(subset=[metric]).copy()
        view['label'] = view['subset'] + '::' + view['feature_set']
        order = (view[view['model_family'] == 'partition']
                 .sort_values(metric, ascending=False)['label'])
        sns.barplot(data=view, x='label', y=metric, hue='model_family',
                    order=order if len(order) else None, ax=ax)
        ax.set_title(f'median {metric} by feature set: partition vs density')
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelrotation=90, labelsize=8)
        ax.legend(fontsize=8, title='family')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='w')
    plt.close(fig)


def _markdown_table(frame: DataFrame, max_rows: int = 20) -> str:
    """Render a frame as a pipe table.

    Hand-rolled rather than ``DataFrame.to_markdown``, which needs ``tabulate`` --
    not in environment.yml, and not worth a new dependency for one summary file.
    """
    if frame.empty:
        return '_none_'

    view = frame.head(max_rows)
    columns = [str(c) for c in view.columns]

    def cell(value: object) -> str:
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value)

    lines = ['| ' + ' | '.join(columns) + ' |',
             '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines += ['| ' + ' | '.join(cell(v) for v in row) + ' |'
              for row in view.itertuples(index=False)]

    return '\n'.join(lines)


def write_summary(matrix: DataFrame, ranking: DataFrame, best: DataFrame,
                  feature_sets: dict[str, list[str]], out_path: Path) -> None:
    """Prose summary of the core features and the best-scoring feature sets."""
    total = int(matrix['n_scored_sets'].iloc[0]) if len(matrix) else 0
    n_liquids = labelled_liquid_count()
    core = matrix[matrix['n_sets_selected'] >= max(1, int(round(total * 0.6)))]
    never = matrix[matrix['n_sets_selected'] == 0]

    lines = [
        '# Feature selection comparison',
        '',
        f'{len(feature_sets)} feature sets compared, {total} of them non-trivial.',
        '',
        '## Core descriptors',
        '',
        f'Retained by at least 60% of the {total} scored feature sets:',
        '',
    ]
    lines += [f'- `{f}` — {int(n)}/{total}'
              for f, n in core['n_sets_selected'].items()] or ['- (none)']
    lines += [
        '',
        f'Maximum achievable is {int(matrix["n_sets_selected"].max())}/{total}. Factor '
        'sets are counted through the descriptors they aggregate, not by column name:',
        'indexed on raw names the six `factor_*` sets matched nothing at all and',
        'contributed all-`False` columns while still counting in the denominator, which',
        'capped survival below the threshold and made this list read `(none)`.',
    ]
    lines += [
        '',
        '## Never retained',
        '',
        f'Dropped by every scored feature set ({len(never)} descriptors):',
        '',
        '`' + '`, `'.join(never.index) + '`' if len(never) else '- (none)',
        '',
        '## Best scoring combination per metric',
        '',
        _markdown_table(best) if len(best) else '_no metrics available_',
        '',
        '## Feature sets ranked by median silhouette',
        '',
        _markdown_table(ranking, max_rows=20),
        '',
        '## Caveats',
        '',
        f'- `error_eta_squared` rests on {n_liquids} labelled liquids; treat it as',
        '  indicative only.',
        '- Supervised feature sets come from leave-one-liquid-out selection',
        f'  frequency over {n_liquids} folds, not from a single fit.',
        f'- `{cols.PROTECTED[0]}` is retained by construction in every unsupervised',
        '  set, so its survival count carries no evidential weight.',
        '- The supervised sets draw from `columns.supervised_pool()`: no atom counts,',
        '  and `logPvap25` in place of the three raw Antoine coefficients, which are',
        '  jointly fitted to one curve and not independently selectable. The',
        '  unsupervised screens still see the raw coefficients, so the two pools are',
        '  not yet fully identical.',
    ]

    out_path.write_text('\n'.join(lines))


def run() -> dict[str, DataFrame]:
    """Assemble every comparison artefact and write it."""
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    FIG_COMPARISON.mkdir(parents=True, exist_ok=True)

    feature_sets = collect_feature_sets()
    if not feature_sets:
        raise FileNotFoundError('no feature sets found; run screening first')

    matrix = assemble_matrix(feature_sets)
    jaccard = jaccard_matrix(feature_sets)

    matrix.to_csv(MATRIX_PATH)
    jaccard.to_csv(JACCARD_PATH)
    with open(COMPARISON_DIR / 'feature_sets.json', 'w') as handle:
        json.dump(feature_sets, handle, ensure_ascii=False, indent=2)

    plot_jaccard(jaccard, FIG_COMPARISON / 'feature_set_jaccard.png')
    plot_survival(matrix, FIG_COMPARISON / 'feature_survival.png')

    ranking = DataFrame()
    best = DataFrame()
    family = DataFrame()
    if METRICS_PATH.is_file():
        metrics = pd.read_csv(METRICS_PATH)
        ranking = feature_set_ranking(metrics)
        best = best_by_metric(metrics)
        family = family_ranking(metrics)
        ranking.to_csv(COMPARISON_DIR / 'feature_set_ranking.csv', index=False)
        best.to_csv(BEST_PATH, index=False)
        family.to_csv(FAMILY_RANKING_PATH, index=False)
        plot_metric_by_feature_set(
            ranking, FIG_COMPARISON / 'clustering_metrics_by_feature_set.png')
        plot_family_agreement(
            family, FIG_COMPARISON / 'density_vs_partition_agreement.png')

    write_summary(matrix, ranking, best, feature_sets, SUMMARY_PATH)

    # Previously unreachable: the __main__ block used to sit here, above
    # selection_method_comparison()'s definition, so run() could not call it and the
    # CSV was last written by hand. Notebook 6 reads it, so it silently displayed a
    # stale seven-liquid table beside current results.
    methods = selection_method_comparison()
    methods.to_csv(METHOD_COMPARISON_PATH, index=False)
    intervals = (pd.read_csv(BOOTSTRAP_CI_PATH) if BOOTSTRAP_CI_PATH.is_file()
                 else DataFrame())
    plot_method_families(methods, FIG_COMPARISON / 'selection_method_families.png',
                         intervals=intervals)

    return {'matrix': matrix, 'jaccard': jaccard, 'ranking': ranking, 'best': best,
            'family': family, 'methods': methods}


# Which family each selection method belongs to. MRMR is broken out because the
# whole point of comparing it is that it optimises relevance and redundancy jointly,
# where the unsupervised screens handle only redundancy and the wrapper/embedded
# methods only relevance.
# ``factor`` is its own family rather than part of ``unsupervised``: it is built from
# the same clustering as ``fcluster`` but aggregates each group instead of sampling
# one member, and that difference is the whole comparison. ``physics`` is a family of
# one -- descriptors fixed on mechanism before any search -- and is the control that
# says whether searching bought anything.
METHOD_FAMILIES: dict[str, str] = {
    'variance': 'unsupervised', 'spearman': 'unsupervised', 'fcluster': 'unsupervised',
    'no_composition': 'unsupervised', 'factor': 'factor', 'physics': 'physics',
    'manual': 'manual', 'all': 'reference',
    'wrapper': 'wrapper', 'embedded': 'embedded', 'mrmr': 'mrmr',
    'supervised_consensus': 'consensus',
}


def method_family(name: str) -> str:
    """Family of a feature-set name, ignoring any ``subset::`` prefix."""
    bare = name.split('::', 1)[-1]
    for prefix, family in METHOD_FAMILIES.items():
        if bare.startswith(prefix):
            return family

    return 'other'


def selection_method_comparison() -> DataFrame:
    """One row per selection method: stability, regression skill, cluster quality.

    Brings the three previously separate verdicts into a single table so MRMR can be
    judged against the wrapper, embedded and unsupervised alternatives on the same
    axes rather than on selection frequency alone.
    """
    from allmeta.supervised import STABILITY_PATH
    from allmeta.regression import SCORES_PATH

    rows: dict[str, dict] = {}

    # Hoisted: this was called once per feature set inside the loop below, and the
    # lookup used the PREFIXED name while the rows are keyed on the bare one -- so
    # 'all', 'manual' and 'physics', which live only in regression's own set list and
    # never in collect_feature_sets(), all reported n_features = NaN in a table whose
    # whole point is comparing methods at a fixed feature count.
    from allmeta.regression import candidate_feature_sets

    known_sets = candidate_feature_sets()

    if STABILITY_PATH.is_file():
        stability = pd.read_csv(STABILITY_PATH)
        for row in stability[~stability['include_params'].astype(bool)].itertuples():
            entry = rows.setdefault(row.method, {})
            entry['churn'] = row.churn
            entry['n_unanimous'] = row.n_unanimous
            # Churn against its null. On the raw 0-1 scale a method at 0.45 reads as
            # mid-range; against a random selector from the same pool it is chance.
            for column in ('churn_random', 'churn_ratio', 'churn_p'):
                if hasattr(row, column):
                    entry[column] = getattr(row, column)

    if SCORES_PATH.is_file():
        scores = pd.read_csv(SCORES_PATH)
        view = scores[(~scores['include_params'].astype(bool))
                      & (scores['model'] != 'baseline_mean')]
        for name, group in view.groupby('feature_set'):
            bare = name.split('::', 1)[-1]
            best = group.nsmallest(1, 'rmse').iloc[0]
            rows.setdefault(bare, {}).update({
                # The minimum over five estimators, scored on the same pooled
                # out-of-fold RMSE that is then reported. The minimum of five noisy
                # numbers is biased low -- the same way the deepest of five repeat
                # profilometer traces overstates the true trench depth -- so the
                # MEDIAN is the headline and the minimum is named `_of_5` so it can
                # never be read as an estimate of anything.
                #
                # `_leaky` on every supervised row is not decoration either: these
                # feature sets were chosen using this target on these same liquids,
                # aggregated across all thirteen folds. The leakage-free number is in
                # the `nested_*` columns below.
                'best_rmse_of_5_leaky': float(best['rmse']),
                'best_r2_of_5_leaky': float(best['r2']),
                'best_model_leaky': str(best['model']),
                'median_rmse_leaky': float(group['rmse'].median()),
                'median_r2_leaky': float(group['r2'].median()),
                'best_between_liquid_r2_leaky': float(best['between_liquid_r2'])
                if 'between_liquid_r2' in group else None,
                'median_between_liquid_r2_leaky': float(
                    group['between_liquid_r2'].median())
                if 'between_liquid_r2' in group else None,
                'beats_baseline': bool(best['beats_baseline']),
                'n_beating_baseline': int(group['beats_baseline'].sum()),
                'n_models': int(len(group)),
                'n_features': int(len(known_sets.get(name, []))) or None,
            })

    # The nested (leakage-free) supervised scores. These are THE supervised result:
    # each fold is scored on the features that fold's selector chose, so the held-out
    # liquid never influenced its own feature set. They are worse than the `_leaky`
    # columns above, and that difference is the optimism the aggregate was carrying.
    if SCORES_NESTED_PATH.is_file():
        nested = pd.read_csv(SCORES_NESTED_PATH)
        nested = nested[nested['model'] != 'baseline_mean']
        for name, group in nested.groupby('feature_set'):
            best = group.nsmallest(1, 'rmse').iloc[0]
            rows.setdefault(name, {}).update({
                'nested_median_rmse': float(group['rmse'].median()),
                'nested_median_r2': float(group['r2'].median()),
                'nested_median_between_liquid_r2': float(
                    group['between_liquid_r2'].median()),
                'nested_best_rmse_of_5': float(best['rmse']),
                'nested_best_model': str(best['model']),
                'nested_n_beating_baseline': int(group['beats_baseline'].sum()),
                'nested_n_features_median': float(best.get('n_features_median',
                                                           float('nan'))),
            })

    if BOOTSTRAP_CI_PATH.is_file():
        ci = pd.read_csv(BOOTSTRAP_CI_PATH)
        ci = of_evaluation(ci, 'nested')
        for name, group in ci.groupby('feature_set'):
            if name not in rows:
                continue
            # Interval for the median model, so the CI describes the same
            # configuration the headline does rather than the winner's.
            middle = group.sort_values('observed').iloc[len(group) // 2]
            rows[name]['nested_ci_lo'] = float(middle['ci_lo'])
            rows[name]['nested_ci_hi'] = float(middle['ci_hi'])

    if METRICS_PATH.is_file():
        metrics = pd.read_csv(METRICS_PATH)
        metrics = metrics.assign(model_family=metrics['model'].map(model_family))
        for name, group in metrics.groupby('feature_set'):
            entry = rows.setdefault(name, {})
            entry['median_silhouette'] = float(group['silhouette'].median())
            # A and B are clustered separately and their feature sets differ, but
            # 'feature_set' in metrics.csv is the bare name -- so this median silently
            # blended two different partitions of two different liquid sets into one
            # number. Kept for continuity, with the per-subset values beside it so the
            # blend is visible and a disagreement between the subsets cannot hide in
            # the average.
            for subset, subset_group in group.groupby('subset'):
                entry[f'median_silhouette_{subset}'] = float(
                    subset_group['silhouette'].median())
            # Second verdict from the density family alone (DBSCAN/HDBSCAN, which
            # can call weak structure noise instead of forcing a cluster), so a
            # method's features can be checked against both without re-deriving it
            # from feature_set_ranking_by_family.csv.
            density = group.loc[group['model_family'] == 'density', 'silhouette']
            if density.notna().any():
                entry['median_silhouette_density'] = float(density.median())

    table = DataFrame(rows).T.reset_index().rename(columns={'index': 'method'})
    table['family'] = table['method'].map(method_family)

    # Every supervised method chose its features using this target on these same
    # labelled liquids, so its regression score is optimistically biased. Flagged
    # rather than dropped, because the churn column is unaffected and is the fair
    # comparison between them.
    supervised = {'wrapper', 'embedded', 'mrmr', 'consensus'}
    table['selection_leakage'] = table['family'].isin(supervised)

    # Explain the blanks: a method scores NaN either because no feature cleared the
    # selection-frequency threshold (so no feature set exists) or because another
    # method selected exactly the same features and only one of the two was fitted.
    aliased: dict[str, str] = {}
    if ALIASES_PATH.is_file():
        alias_table = pd.read_csv(ALIASES_PATH)
        aliased = {row.identical_to.split('::', 1)[-1]: row.evaluated_as
                   for row in alias_table.itertuples()}

    def note(row) -> str:
        method = row['method']
        # Self-references: a set registered under both a bare and a prefixed name
        # aliases to itself, which is not worth saying.
        alias = aliased.get(method)
        scored = pd.notna(row.get('best_r2_of_5_leaky'))
        nested_only = (pd.isna(row.get('n_features'))
                       and pd.notna(row.get('nested_median_r2')))

        if nested_only:
            # Scored nested but with no aggregate feature set at all: nesting uses the
            # per-fold picks directly and never needs a feature to clear the frequency
            # threshold. Blank n_features here reads as missing data unless said.
            return (f'no feature reached the {SUPERVISED_MIN_FREQUENCY:.0%} threshold, '
                    'so no aggregate set exists; nested score is the only one')
        if scored or pd.notna(row.get('nested_median_r2')):
            return ''
        if alias and alias != method:
            return f'identical selection to {alias}'

        return f'no feature reached the {SUPERVISED_MIN_FREQUENCY:.0%} threshold'

    table['note'] = [note(r) for _, r in table.iterrows()]

    # Nested columns lead the regression block: they are the honest number, and a
    # column order is the first thing a reader takes a ranking from.
    ordered = [c for c in ('method', 'family', 'n_features',
                           'churn', 'churn_random', 'churn_ratio', 'churn_p',
                           'n_unanimous',
                           'nested_median_rmse', 'nested_median_r2',
                           'nested_median_between_liquid_r2',
                           'nested_ci_lo', 'nested_ci_hi',
                           'nested_best_rmse_of_5', 'nested_best_model',
                           'nested_n_features_median', 'nested_n_beating_baseline',
                           'median_rmse_leaky', 'median_r2_leaky',
                           'median_between_liquid_r2_leaky',
                           'best_rmse_of_5_leaky', 'best_r2_of_5_leaky',
                           'best_between_liquid_r2_leaky', 'best_model_leaky',
                           'beats_baseline', 'n_beating_baseline', 'n_models',
                           'selection_leakage', 'median_silhouette',
                           'median_silhouette_A', 'median_silhouette_B',
                           'median_silhouette_density', 'note')
               if c in table.columns]

    return table[ordered].sort_values(['family', 'method'], ignore_index=True)


def plot_method_families(table: DataFrame, path: Path,
                         intervals: DataFrame | None = None,
                         baseline: float | None = None) -> Path:
    """Churn and regression skill by selection family.

    Churn is the only axis on which these methods separate cleanly at this n; the
    regression panel is shown beside it to make clear that a stable method is not
    thereby a predictive one.

    Both panels used to mislead in the same direction. Churn was drawn against an
    implied 0-1 scale whose upper end is unreachable and whose *middle* is what random
    selection scores, and the skill panel plotted the maximum over the model grid
    against a baseline drawn at zero. Now: churn carries its null as a line, and skill
    is the median across the grid -- nested where a leakage-free number exists --
    against the baseline the mean predictor actually achieves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    churn = table.dropna(subset=['churn']).sort_values('churn')
    sns.barplot(data=churn, y='method', x='churn', hue='family', dodge=False,
                ax=axes[0], palette='Set2')

    # Random selection does not score near 1.0 -- thirteen small samples from a
    # modest pool overlap by chance, so it lands near 0.5. Without this line a method
    # at 0.45 reads as mid-range when it is actually indistinguishable from chance.
    if 'churn_random' in churn.columns and churn['churn_random'].notna().any():
        expected = float(churn['churn_random'].median())
        axes[0].axvline(expected, color='k', linestyle='--', linewidth=1.5,
                        label=f'random selection ({expected:.3f})')
        axes[0].legend(fontsize=8, title='family')

    axes[0].set_xlabel('churn  (0 = same features every fold; dashed line = chance)')
    axes[0].set_ylabel('')
    axes[0].set_title('Selection stability under leave-one-liquid-out')
    if 'churn_random' not in churn.columns:
        axes[0].legend(fontsize=8, title='family')

    # The nested median where there is one, the leaky median otherwise -- never a
    # maximum over the model grid, which is what this panel used to plot.
    skill_column = ('nested_median_r2' if 'nested_median_r2' in table.columns
                    and table['nested_median_r2'].notna().any()
                    else 'median_r2_leaky')
    fallback = 'median_r2_leaky'
    skill = table.copy()
    if skill_column != fallback and fallback in skill.columns:
        skill['skill'] = skill[skill_column].fillna(skill[fallback])
        skill['honest'] = skill[skill_column].notna()
    else:
        skill['skill'] = skill.get(fallback)
        skill['honest'] = False
    skill = skill.dropna(subset=['skill']).sort_values('skill', ascending=False)

    sns.barplot(data=skill, y='method', x='skill', hue='family', dodge=False,
                ax=axes[1], palette='Set2')

    # Bootstrap interval over the thirteen liquids. Drawn because the bars are maxima
    # over a five-model grid and the intervals are roughly an R-squared unit wide: the
    # ordering they suggest is not one the data can support.
    if intervals is not None and not intervals.empty:
        for i, row in enumerate(skill.itertuples()):
            lo = getattr(row, 'nested_ci_lo', None)
            hi = getattr(row, 'nested_ci_hi', None)
            if lo is None or pd.isna(lo) or hi is None or pd.isna(hi):
                continue
            axes[1].plot([lo, hi], [i, i], color='0.15', linewidth=1.2, zorder=5)
            axes[1].plot([lo, hi], [i, i], '|', color='0.15', markersize=6, zorder=5)

    # The mean baseline is NOT at zero. Under leave-one-liquid-out the held-out
    # liquid is excluded from the training mean, so the mean predictor's residual is
    # inflated by n/(n-1) and it scores 1-(n/(n-1))^2 -- about -0.076 pooled at
    # thirteen liquids. Drawing the line at zero credited every method with clearing a
    # bar that does not exist. Read from the data, never hardcoded: it moves with the
    # liquid count. See regression.baseline_r2.
    if baseline is None and SCORES_PATH.is_file():
        baseline = baseline_r2(pd.read_csv(SCORES_PATH))

    if baseline is not None:
        axes[1].axvline(baseline, color='k', linewidth=1.5,
                        label=f'mean baseline (R2 = {baseline:.4f})')
        axes[1].axvline(0, color='0.6', linewidth=1, linestyle=':',
                        label='perfect-mean reference (not achievable under LOGO)')
        axes[1].set_xlabel(f'MEDIAN pooled out-of-fold R2 across the model grid  '
                           f'(mean baseline = {baseline:.4f}, not 0)')
    else:
        axes[1].set_xlabel('median pooled out-of-fold R2 across the model grid')

    axes[1].set_ylabel('')
    honest = int(skill['honest'].sum()) if 'honest' in skill else 0
    axes[1].set_title(f'Predictive skill of each feature set\n'
                      f'{honest} shown nested (leakage-free); the rest are the leaky '
                      f'aggregate', fontsize=10)
    axes[1].legend(fontsize=8, title='family')

    fig.suptitle('Feature selection methods compared: MRMR against wrapper, '
                 'embedded and unsupervised', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    return save(fig, path)


if __name__ == '__main__':
    out = run()
    print(f'{len(out["matrix"])} descriptors x {out["jaccard"].shape[0]} feature sets')
    print(out['matrix']['n_sets_selected'].head(12).to_string())
    print()
    print(f'--- selection methods compared ({labelled_liquid_count()} labelled liquids)')
    print(out['methods'].to_string(index=False))
