"""Figures for the feature-clustering sweep (implementation plan, Steps 16.1-16.4).

Written now rather than at the end, because the plan is explicit that plotting should
follow each working stage: a metric table with 430 rows is not readable, and the
questions these figures answer -- does a method agree with the chemistry, does it agree
with the other methods, is a good score just the method marking its own homework --
cannot be read off a CSV.

The one figure to read first is ``clustering_metric_comparison``, and it makes two
points that together say why Phase 8 needs a predictive criterion rather than a
cluster-quality one:

1. **Each method wins the silhouette measured in the space it optimised.** The
   distance-based methods take the correlation-space silhouette, the profile-based ones
   take the profile-space silhouette. In each case the metric restates the objective.
2. **The domain metric is not a neutral referee either.** ``within_between_delta``
   looks like the fix for (1) -- one fixed space, optimised by nobody -- but on the real
   sweep it correlates at 0.80 with ``singleton_fraction``. The best-scoring solution of
   430 makes 36 singletons out of 38 clusters and commits on 16 of 990 feature pairs.
   The ``within_pair_fraction`` panel sits next to it so that this is impossible to
   miss: a tall delta bar over a short coverage bar is a method refusing to cluster.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from allmeta.feature_selection import metrics as M
from allmeta.paths import FIG_FEATURE_SELECTION
from allmeta.plotting import annotate_empty, apply_style, save

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

FIG_DIR: Path = FIG_FEATURE_SELECTION / 'pipeline'

# Figures are filed by the FEATURE FAMILY they describe, not by the phase that drew
# them, because "what does the manual raw pipeline look like?" is the question a reader
# actually arrives with. The numeric prefixes force the reading order: the descriptor
# space first, then each of the five representations, then the comparison between them.
#
# Cross-cutting stages sit inside the family they belong to rather than in a stage
# folder of their own -- the whole clustering sweep exists to build ONE representation,
# so it lives under that representation. Only 06 is genuinely about all five at once.
FIG_SPACE: Path = FIG_DIR / '00_descriptor_space'
FIG_MANUAL_RAW: Path = FIG_DIR / '01_manual_raw'
FIG_MRMR_RAW: Path = FIG_DIR / '02_mrmr_raw'
FIG_GLOBAL_PCA: Path = FIG_DIR / '03_global_pca'
FIG_MANUAL_PC1: Path = FIG_DIR / '04_manual_cluster_pc1'
FIG_AUTO_PC1: Path = FIG_DIR / '05_auto_cluster_pc1'
FIG_AUTO_SWEEP: Path = FIG_AUTO_PC1 / 'clustering_sweep'
FIG_AUTO_SELECT: Path = FIG_AUTO_PC1 / 'method_selection'
FIG_AUTO_EN: Path = FIG_AUTO_PC1 / 'elastic_net'
FIG_COMPARISON: Path = FIG_DIR / '06_pipeline_comparison'

# Where each family's Elastic Net figures go. A family absent from this map keeps its
# name in the filename and lands in the family folder for anything unrecognised, so a
# new representation does not silently lose its figures.
FIG_BY_FAMILY: dict[str, Path] = {
    'manual_raw': FIG_MANUAL_RAW,
    'mrmr_raw': FIG_MRMR_RAW,
    'global_pca': FIG_GLOBAL_PCA,
    'manual_cluster_pc1': FIG_MANUAL_PC1,
    'auto_cluster_pc1': FIG_AUTO_EN,
}

FIG_SUBDIRS: tuple[Path, ...] = (
    FIG_SPACE, FIG_MANUAL_RAW, FIG_MRMR_RAW, FIG_GLOBAL_PCA, FIG_MANUAL_PC1,
    FIG_AUTO_SWEEP, FIG_AUTO_SELECT, FIG_AUTO_EN, FIG_COMPARISON,
)


def family_dir(name: str) -> Path:
    """Where a per-family figure belongs.

    An auto_cluster_pc1 set carried forward under ``retain_competitive`` is named
    ``auto_cluster_pc1__kmedoids``, so the match is on prefix rather than equality.
    """
    if name in FIG_BY_FAMILY:
        return FIG_BY_FAMILY[name]
    for known, path in FIG_BY_FAMILY.items():
        if name.startswith(known):
            return path

    return FIG_DIR

# The metrics worth showing side by side, and which way each reads.
PANEL_METRICS: tuple[str, ...] = (
    'silhouette_correlation', 'silhouette_profile', 'calinski_harabasz_profile',
    'davies_bouldin_profile', 'mean_within_abs_corr', 'mean_between_abs_corr',
    'within_between_delta', 'within_pair_fraction', 'singleton_fraction',
    'mean_pc1_explained_variance_multi', 'min_pc1_explained_variance',
)

# Which space each internal metric was measured in, for the axis labels. Stated on the
# figure because comparing a _correlation column against a _profile column is the one
# misreading these panels invite.
SPACE_OF: dict[str, str] = {
    'silhouette_correlation': 'correlation distance',
    'silhouette_profile': 'feature profile',
    'calinski_harabasz_profile': 'feature profile',
    'davies_bouldin_profile': 'feature profile',
}

METHOD_ORDER: tuple[str, ...] = ('hierarchical', 'hdbscan', 'kmedoids', 'kmeans', 'gmm')


def _ordered_methods(frame: DataFrame) -> list[str]:
    """Methods present, in a fixed order so colours are stable across figures."""
    present = set(frame['method'].dropna())

    return [m for m in METHOD_ORDER if m in present] + sorted(present - set(METHOD_ORDER))


def plot_feature_dendrogram(distance: DataFrame, thresholds, path: Path) -> Path:
    """The feature tree in correlation-distance space, with every cut drawn on.

    The single most informative picture of what is being clustered: the y-axis is
    ``1 - |r|``, so a merge at 0.1 joins descriptors correlating above 0.9, and the
    hierarchical grid's thresholds are exactly horizontal lines across it.
    """
    values = distance.to_numpy(dtype=float)
    tree = linkage(squareform((values + values.T) / 2.0, checks=False),
                   method='average')

    fig, ax = plt.subplots(figsize=(max(11, len(distance) * 0.3), 7))
    dendrogram(tree, labels=list(distance.index), ax=ax, leaf_rotation=90,
               leaf_font_size=8, color_threshold=0.25)

    for threshold in thresholds:
        ax.axhline(threshold, color='0.55', linestyle=':', linewidth=0.9)
    ax.axhline(0.25, color='#c44e52', linestyle='--', linewidth=1.4,
               label='0.25  (merges above |r| = 0.75)')

    ax.set_ylabel('1 - |Spearman rho|')
    ax.set_title('Feature clustering by correlation distance (average linkage)\n'
                 'dotted lines are the thresholds in the hierarchical search grid')
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()

    return save(fig, path)


def plot_ordered_correlation(rho: DataFrame, solution, path: Path) -> Path:
    """Absolute correlation with the features reordered by cluster.

    A partition that found real families shows as blocks on the diagonal. This is the
    picture ``within_between_delta`` reduces to a number, and it is worth having both:
    the number ranks methods, the blocks show whether the number means anything.
    """
    order = [feature for members in solution.clusters().values() for feature in members]
    boundaries = np.cumsum([len(m) for m in solution.clusters().values()])[:-1]

    fig, ax = plt.subplots(figsize=(11, 9.5))
    sns.heatmap(rho.loc[order, order], vmin=0, vmax=1, cmap='rocket_r', square=True,
                cbar_kws={'label': '|Spearman rho|'}, ax=ax)

    for edge in boundaries:
        ax.axhline(edge, color='#4c72b0', linewidth=1.1)
        ax.axvline(edge, color='#4c72b0', linewidth=1.1)

    ax.set_title(f'Feature correlation ordered by {solution.method} clusters '
                 f'({solution.n_clusters} clusters, '
                 f'{solution.n_singletons} singletons)\n'
                 'blocks on the diagonal are the correlation families it found')
    ax.tick_params(axis='x', labelrotation=90, labelsize=7)
    ax.tick_params(axis='y', labelrotation=0, labelsize=7)
    fig.tight_layout()

    return save(fig, path)


def plot_metric_comparison(table: DataFrame, path: Path) -> Path:
    """Every metric, every method, across the whole hyperparameter grid.

    Distributions rather than best-of, deliberately. A method's best score over 180
    candidates is a maximum over noise; the spread is what says whether the method is
    reliably good or occasionally lucky.
    """
    scored = table.dropna(subset=['n_clusters'])
    methods = _ordered_methods(scored)
    panels = [m for m in PANEL_METRICS if m in scored.columns]

    fig, axes = plt.subplots(2, 5, figsize=(23, 9))
    palette = dict(zip(methods, sns.color_palette('Set2', len(methods))))

    for ax, metric in zip(axes.ravel(), panels):
        view = scored.dropna(subset=[metric])
        if view.empty:
            annotate_empty(ax, f'{metric}\nnot computable')
            continue

        sns.boxplot(data=view, x='method', y=metric, order=methods, hue='method',
                    palette=palette, legend=False, ax=ax, width=0.6, fliersize=2)

        direction = ('higher is better' if metric in M.HIGHER_IS_BETTER
                     else 'lower is better' if metric in M.LOWER_IS_BETTER
                     else 'descriptive')
        space = SPACE_OF.get(metric)
        subtitle = f'{direction}' + (f' | measured in {space}' if space else '')
        ax.set_title(f'{metric}\n{subtitle}', fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelrotation=30, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_ha('right')

    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(
        'Clustering metrics across the full hyperparameter grid, by algorithm\n'
        'Each method wins the silhouette measured in the space it optimised -- and '
        'within_between_delta rewards refusing to cluster, so read it against '
        'within_pair_fraction', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    return save(fig, path)


def plot_metric_vs_k(table: DataFrame, path: Path) -> Path:
    """How each metric moves with the number of clusters.

    The parameter every algorithm is most sensitive to, and the one where a metric's
    preference is most likely to be an artefact: silhouette and Calinski-Harabasz both
    drift with cluster count regardless of whether the extra clusters mean anything.
    """
    scored = table.dropna(subset=['n_clusters'])
    methods = _ordered_methods(scored)
    panels = ['silhouette_correlation', 'within_between_delta',
              'within_pair_fraction', 'mean_within_abs_corr']

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
    palette = dict(zip(methods, sns.color_palette('Set2', len(methods))))

    for ax, metric in zip(axes, panels):
        view = scored.dropna(subset=[metric])
        if view.empty:
            annotate_empty(ax, f'{metric}\nnot computable')
            continue

        sns.lineplot(data=view, x='n_clusters', y=metric, hue='method',
                     hue_order=methods, palette=palette, marker='o', markersize=4,
                     errorbar=('pi', 50), ax=ax)
        space = SPACE_OF.get(metric)
        ax.set_title(f'{metric}' + (f'\n({space})' if space else ''), fontsize=10)
        ax.set_xlabel('clusters found')
        ax.set_ylabel('')
        ax.legend(fontsize=8, title=None)

    fig.suptitle('Metrics against cluster count (band = interquartile range over the '
                 'grid)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    return save(fig, path)


def plot_multi_metric_heatmap(table: DataFrame, path: Path) -> Path:
    """Methods against metrics, each column normalised for VISUAL comparison only.

    The normalisation exists so that a Calinski-Harabasz of 7 and a silhouette of 0.47
    can share a colour scale. It must not become the optimisation score: the plan is
    explicit about that, and a mean of normalised metrics would quietly weight ten
    arbitrary quantities equally and call the result a decision.

    Davies-Bouldin is inverted before normalising, so on this figure alone every column
    reads the same way -- greener is better.
    """
    scored = table.dropna(subset=['n_clusters'])
    methods = _ordered_methods(scored)
    panels = [m for m in PANEL_METRICS if m in scored.columns]

    # Median over the grid rather than best: a maximum over 180 candidates says more
    # about how many were tried than about the method.
    grid = scored.groupby('method')[panels].median().reindex(methods)

    shown = grid.copy()
    for metric in shown.columns:
        if metric in M.LOWER_IS_BETTER:
            shown[metric] = -shown[metric]

    spread = shown.max() - shown.min()
    normalised = (shown - shown.min()) / spread.replace(0, np.nan)

    # Three of these columns -- n_clusters, singleton_fraction, noise_fraction -- have
    # no better direction at all. Left unlabelled on a figure whose convention is
    # "brighter is better", a bright noise_fraction reads as praise for HDBSCAN. Each
    # column therefore says which kind it is.
    def label(metric: str) -> str:
        if metric in M.LOWER_IS_BETTER:
            return f'{metric}\n(inverted: brighter is better)'
        if metric in M.HIGHER_IS_BETTER:
            return f'{metric}\n(brighter is better)'
        return f'{metric}\n(DESCRIPTIVE: brighter is not better)'

    labels = [label(m) for m in grid.columns]

    fig, ax = plt.subplots(figsize=(14, 4.6))
    sns.heatmap(normalised.set_axis(labels, axis=1), vmin=0, vmax=1, cmap='viridis',
                annot=grid.to_numpy(), fmt='.2f', linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'normalised for colour only'}, ax=ax,
                annot_kws={'size': 8})
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelrotation=25, labelsize=8)
    ax.tick_params(axis='y', labelrotation=0)
    for label in ax.get_xticklabels():
        label.set_ha('right')

    ax.set_title('Median metric per algorithm over its whole grid\n'
                 'cells are annotated with the REAL value; colour is normalised per '
                 'column and is not an optimisation score', fontsize=12)
    ax.tick_params(axis='x', labelsize=7)
    fig.tight_layout()

    return save(fig, path)


def plot_manual_cluster_quality(loadings: DataFrame, path: Path) -> Path:
    """How much of each hand-defined cluster its PC1 actually retains.

    The manual grouping is the control the automatic methods are judged against, so its
    weaknesses need to be as visible as its strengths. A factor retaining a third of its
    cluster is not summarising that cluster, and the bar for it should be short and
    obvious rather than buried in a CSV.
    """
    # Global PCs must not appear here. Their explained_variance_ratio is a share of
    # the TOTAL variance across all 45 descriptors, while a cluster PC1's is a share of
    # its own cluster's -- so a global PC1 at 0.38 and a cluster PC1 at 0.38 mean
    # entirely different things and a shared axis silently compares them. Rejected
    # rather than filtered, because being handed the wrong table means the caller
    # believes something false.
    global_pcs = [f for f in loadings['factor'].unique()
                  if str(f).startswith('global_PC')]
    if global_pcs:
        raise ValueError(
            f'loadings contain global PCA components ({global_pcs[:3]}...): their '
            'explained variance is a share of the total, not of a cluster, and cannot '
            "share an axis with cluster PC1. Pass the cluster loadings only.")

    quality = (loadings.groupby('factor', sort=False)
               .agg(n_members=('feature', 'size'),
                    pc1=('explained_variance_ratio', 'first'))
               .sort_values('pc1'))
    multi = quality[quality['n_members'] > 1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6),
                             gridspec_kw={'width_ratios': [1.25, 1]})

    colours = ['#c44e52' if v < 0.6 else '#dd8452' if v < 0.75 else '#4c72b0'
               for v in quality['pc1']]
    axes[0].barh(quality.index, quality['pc1'], color=colours)
    axes[0].axvline(0.6, color='k', linestyle='--', linewidth=1,
                    label='0.60 -- below this the factor discards more than it keeps')
    for y, (name, row) in enumerate(quality.iterrows()):
        axes[0].text(row['pc1'] + 0.01, y, f"n={int(row['n_members'])}",
                     va='center', fontsize=8, color='0.3')
    axes[0].set_xlim(0, 1.12)
    axes[0].set_xlabel('PC1 explained variance ratio')
    axes[0].set_title('Manual cluster PC1 information retention\n'
                      'singletons are 1.00 by construction and carry no evidence')
    axes[0].legend(fontsize=8, loc='lower right')

    if multi.empty:
        annotate_empty(axes[1], 'no multi-member clusters')
    else:
        axes[1].scatter(multi['n_members'], multi['pc1'], s=90, color='#4c72b0',
                        edgecolor='k', linewidth=0.5)
        for name, row in multi.iterrows():
            axes[1].annotate(name, (row['n_members'], row['pc1']), fontsize=8,
                             xytext=(5, 4), textcoords='offset points')
        axes[1].set_xlabel('descriptors in the cluster')
        axes[1].set_ylabel('PC1 explained variance ratio')
        axes[1].set_title('Bigger clusters are not automatically worse\n'
                          'a low ratio at small n means the grouping is wrong')

    fig.tight_layout()

    return save(fig, path)


def plot_pc1_quality(quality: DataFrame, path: Path) -> Path:
    """How much of each cluster its PC1 retains, across the whole sweep (Step 16.7).

    Three panels, because the headline number is misleading on its own:

    * **left** -- retention against cluster size, multi-member clusters only. The
      question a reader actually has: does the factor still summarise the family once
      the family is large? A flat cloud says size is not the problem; a downward slope
      says the big clusters are being over-merged.
    * **middle** -- the distribution per algorithm, again multi-member only.
    * **right** -- the inflation, made explicit. Every algorithm's all-cluster mean is
      plotted against its multi-member mean, and the gap between a point and the
      diagonal IS the singleton subsidy: singletons retain 1.0 by construction, so a
      method that mostly declines to cluster reads as near-perfect on the axis a
      careless reader would quote.
    """
    multi = quality[~quality['is_singleton']]
    methods = _ordered_methods(quality)
    palette = dict(zip(methods, sns.color_palette('Set2', len(methods))))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))

    if multi.empty:
        for ax in axes:
            annotate_empty(ax, 'every cluster is a singleton')
        return save(fig, path)

    floor = config_floor()

    sns.scatterplot(data=multi, x='n_members', y='explained_variance_ratio',
                    hue='method', hue_order=methods, palette=palette, alpha=0.45,
                    s=28, edgecolor='none', ax=axes[0])
    axes[0].axhline(floor, color='#c44e52', linestyle='--', linewidth=1.2,
                    label=f'{floor:.2f} retention floor')
    axes[0].set_xlabel('descriptors in the cluster')
    axes[0].set_ylabel('PC1 explained variance ratio')
    axes[0].set_title('Retention against cluster size\n(multi-member clusters only)')
    axes[0].legend(fontsize=8)

    sns.boxplot(data=multi, x='method', y='explained_variance_ratio', order=methods,
                hue='method', palette=palette, legend=False, ax=axes[1], width=0.6,
                fliersize=2)
    axes[1].axhline(floor, color='#c44e52', linestyle='--', linewidth=1.2)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('PC1 explained variance ratio')
    axes[1].set_title('Retention by algorithm\n(multi-member clusters only)')
    axes[1].tick_params(axis='x', labelrotation=25, labelsize=9)
    for label in axes[1].get_xticklabels():
        label.set_ha('right')

    inflation = quality.groupby('method').apply(
        lambda g: pd.Series({
            'all_clusters': g['explained_variance_ratio'].mean(),
            'multi_only': g.loc[~g['is_singleton'], 'explained_variance_ratio'].mean(),
            'singleton_fraction': g['is_singleton'].mean(),
        }), include_groups=False).reindex(methods).dropna()

    axes[2].plot([0, 1], [0, 1], 'k--', linewidth=1, label='no inflation')
    # The two most singleton-heavy methods land almost on top of each other, so the
    # labels are pushed to alternating sides rather than all trailing right.
    offsets = [(8, 6), (-8, -16), (8, -14), (-8, 8), (8, 8)]
    for position, (method, row) in enumerate(inflation.iterrows()):
        axes[2].scatter(row['multi_only'], row['all_clusters'], s=150,
                        color=palette[method], edgecolor='k', linewidth=0.6)
        dx, dy = offsets[position % len(offsets)]
        axes[2].annotate(f"{method}\n{row['singleton_fraction']:.0%} singletons",
                         (row['multi_only'], row['all_clusters']), fontsize=8,
                         xytext=(dx, dy), textcoords='offset points',
                         ha='left' if dx > 0 else 'right')
    axes[2].set_xlim(0.4, 1.02)
    axes[2].set_ylim(0.4, 1.02)
    axes[2].set_xlabel('mean PC1 retention, multi-member clusters')
    axes[2].set_ylabel('mean PC1 retention, ALL clusters')
    axes[2].set_title('The singleton subsidy\n'
                      'distance above the line = credit for not clustering')
    axes[2].legend(fontsize=8, loc='lower right')

    fig.suptitle('Cluster PC1 information retention across the whole sweep', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    return save(fig, path)


def plot_consensus(consensus: DataFrame, path: Path) -> Path:
    """Co-clustering probability per feature pair, one panel per algorithm (Step 16.5).

    The evidence behind every stability summary. A block of near-1.0 off the diagonal is
    a descriptor family the method finds whatever liquids it is given; a wash of
    mid-range values is a method whose cluster COUNT is stable while its cluster
    MEMBERSHIP is not, which a mean ARI alone will not reveal.
    """
    methods = _ordered_methods(consensus)
    if not methods:
        fig, ax = plt.subplots(figsize=(6, 4))
        annotate_empty(ax, 'no consensus data')
        return save(fig, path)

    features = sorted(set(consensus['feature_a']) | set(consensus['feature_b']))

    # Ordered once, by the first method's consensus, and reused for every panel: panels
    # in different orders cannot be compared by eye, which is the only thing this
    # figure is for.
    reference = _consensus_square(consensus[consensus['method'] == methods[0]], features)
    order = _leaf_order(reference)

    fig, axes = plt.subplots(1, len(methods), figsize=(5.6 * len(methods), 6.6))
    axes = np.atleast_1d(axes)

    mesh = None
    for ax, method in zip(axes, methods):
        square = _consensus_square(consensus[consensus['method'] == method], features)
        # No per-panel colour bar: it steals width from whichever panel carries it and
        # the panels then cannot be compared by eye, which is the figure's only job.
        # One shared bar is added below instead.
        sns.heatmap(square.loc[order, order], vmin=0, vmax=1, cmap='rocket_r',
                    square=True, cbar=False, ax=ax)
        mesh = ax.collections[0]
        ambiguous = _ambiguous_fraction(square)
        ax.set_title(f'{method}\n{ambiguous:.0%} of pairs undecided (0.2-0.8)',
                     fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelrotation=90, labelsize=5)
        ax.tick_params(axis='y', labelrotation=0, labelsize=5)

    fig.suptitle('Feature co-clustering consensus across resamples '
                 '(same feature order in every panel)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.94, 0.93))
    if mesh is not None:
        fig.colorbar(mesh, cax=fig.add_axes((0.955, 0.12, 0.012, 0.7)),
                     label='co-cluster probability')

    return save(fig, path)


def _consensus_square(rows: DataFrame, features: list[str]) -> DataFrame:
    """Long-form consensus rows back into a symmetric matrix."""
    square = pd.DataFrame(np.eye(len(features)), index=features, columns=features)
    for row in rows.itertuples():
        square.loc[row.feature_a, row.feature_b] = row.co_cluster_probability
        square.loc[row.feature_b, row.feature_a] = row.co_cluster_probability

    return square


def _leaf_order(square: DataFrame) -> list[str]:
    """Feature order that puts co-clustered descriptors next to each other."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    distance = 1.0 - square.to_numpy(dtype=float)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform((distance + distance.T) / 2.0, checks=False),
                   method='average')

    return [square.index[i] for i in leaves_list(tree)]


def _ambiguous_fraction(square: DataFrame) -> float:
    values = square.to_numpy(dtype=float)
    pairs = values[np.triu(np.ones_like(values, dtype=bool), k=1)]

    return float(((pairs > 0.2) & (pairs < 0.8)).mean())


def plot_stability(stability: DataFrame, path: Path) -> Path:
    """Agreement across resamples, by algorithm (Step 16.3).

    Read the right panel against the left. In principle a method can hold a high mean
    ARI while leaving most feature pairs undecided -- the cluster count reproduces but
    the membership does not, and the label-invariant indices only see the partition as
    a whole. The upper-right of the third panel is where that would show.

    On this data it does not happen: the two agree, and the relationship is cleanly
    negative. The panel is kept because it is the check that licenses quoting the ARI
    on its own, and a check is only worth anything if it was capable of failing.
    """
    scored = stability.dropna(subset=['ari_mean'])
    methods = _ordered_methods(scored)
    palette = dict(zip(methods, sns.color_palette('Set2', len(methods))))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))

    if scored.empty:
        for ax in axes:
            annotate_empty(ax, 'no stability data')
        return save(fig, path)

    melted = scored.melt(id_vars='method',
                         value_vars=['ari_mean', 'ami_mean', 'nmi_mean'],
                         var_name='index', value_name='agreement')
    melted['index'] = melted['index'].str.removesuffix('_mean').str.upper()
    sns.boxplot(data=melted, x='index', y='agreement', hue='method', hue_order=methods,
                palette=palette, ax=axes[0], width=0.75, fliersize=2)
    axes[0].set_title('Agreement between resamples\n(higher = the partition repeats)')
    axes[0].set_xlabel('')
    axes[0].legend(fontsize=8, title=None)

    sns.boxplot(data=scored, x='method', y='consensus_ambiguous_fraction',
                order=methods, hue='method', palette=palette, legend=False, ax=axes[1],
                width=0.6, fliersize=2)
    axes[1].set_title('Feature pairs left undecided\n'
                      'co-cluster probability between 0.2 and 0.8')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('ambiguous fraction')
    axes[1].tick_params(axis='x', labelrotation=25, labelsize=9)
    for label in axes[1].get_xticklabels():
        label.set_ha('right')

    sns.scatterplot(data=scored, x='ari_mean', y='consensus_ambiguous_fraction',
                    hue='method', hue_order=methods, palette=palette, s=45,
                    alpha=0.65, edgecolor='none', ax=axes[2])
    axes[2].set_xlabel('mean ARI between resamples')
    axes[2].set_ylabel('ambiguous fraction')
    axes[2].set_title('Do the two agree?\n'
                      'upper right = repeats the partition but not the pairs')
    axes[2].legend(fontsize=8, title=None)

    n = int(scored['n_resamples'].iloc[0])
    frame_size = scored['frame_size'].iloc[0]
    scope = 'all liquids' if pd.isna(frame_size) else f'{int(frame_size)} liquids'
    fig.suptitle(f'Clustering stability over {n} subsamples ({scope}, '
                 f"{scored['subsample_fraction'].iloc[0]:.0%} of rows each)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    return save(fig, path)


def config_floor() -> float:
    """The retention floor, read from config so the figure cannot drift from the flag."""
    from allmeta.feature_selection import config

    return config.PC1_RETENTION_FLOOR


def plot_selection(search: DataFrame, selections: DataFrame, path: Path) -> Path:
    """What the predictive criterion actually separates, and how firm the choice is.

    Three panels, and the third is the one that decides how much weight the other two
    can carry:

    * **left** -- predictive error against factor count, with the mean baseline and the
      one-standard-error band drawn on. The band's width relative to the spread is the
      whole story: where it is wide, prediction is not choosing and the tie-break is.
    * **middle** -- the error distribution per algorithm, against the baseline.
    * **right** -- which configuration wins in each of the 13 leave-one-liquid-out
      folds. If the winner changes from fold to fold then "the best clustering" is a
      property of which liquids were in the fold, not of the data.
    """
    from allmeta.feature_selection import config, optimization

    criterion = f'{config.SELECTION_CRITERION}_mean'
    candidates = search[search['method'] != optimization.BASELINE_NAME]
    baseline = search.loc[search['method'] == optimization.BASELINE_NAME,
                          criterion]
    methods = _ordered_methods(candidates)
    palette = dict(zip(methods, sns.color_palette('Set2', len(methods))))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))

    sns.scatterplot(data=candidates, x='n_factors', y=criterion, hue='method',
                    hue_order=methods, palette=palette, s=34, alpha=0.6,
                    edgecolor='none', ax=axes[0])
    if not baseline.empty:
        axes[0].axhline(baseline.iloc[0], color='k', linestyle='--', linewidth=1.4,
                        label=f'mean baseline ({baseline.iloc[0]:.5f})')
    within = candidates[candidates['within_one_se'].astype(bool)]
    if not within.empty:
        edge = within[criterion].max()
        axes[0].axhspan(within[criterion].min(), edge, color='0.85', alpha=0.5,
                        zorder=0, label=f'within one SE ({len(within)} candidates)')
    axes[0].set_xlabel('factors produced')
    axes[0].set_ylabel(criterion)
    axes[0].set_title('Predictive error against parsimony')
    axes[0].legend(fontsize=8)

    sns.boxplot(data=candidates, x='method', y=criterion, order=methods,
                hue='method', palette=palette, legend=False, ax=axes[1], width=0.6,
                fliersize=2)
    if not baseline.empty:
        axes[1].axhline(baseline.iloc[0], color='k', linestyle='--', linewidth=1.4)
    axes[1].set_xlabel('')
    axes[1].set_title('Error by algorithm\n(dashed = predicting the training mean)')
    axes[1].tick_params(axis='x', labelrotation=25, labelsize=9)
    for label in axes[1].get_xticklabels():
        label.set_ha('right')

    if selections is None or selections.empty:
        annotate_empty(axes[2], 'no per-fold selections')
    else:
        counts = (selections.groupby(['method', 'hyperparameters']).size()
                  .rename('folds').reset_index().sort_values('folds'))
        labels = [f"{r.method}\n{r.hyperparameters[:38]}" for r in counts.itertuples()]
        colours = [palette.get(m, '0.6') for m in counts['method']]
        axes[2].barh(range(len(counts)), counts['folds'], color=colours)
        axes[2].set_yticks(range(len(counts)))
        axes[2].set_yticklabels(labels, fontsize=7)
        axes[2].set_xlabel('leave-one-liquid-out folds that chose it')
        axes[2].set_title(f'Which configuration wins, per fold\n'
                          f"{len(counts)} distinct winners across "
                          f'{len(selections)} folds')

    fig.suptitle('Selecting a clustering candidate on predictive error '
                 f'(criterion: {config.SELECTION_CRITERION})', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    return save(fig, path)


def plot_all(table: DataFrame, rho: DataFrame, distance: DataFrame,
             loadings: DataFrame, solution=None, thresholds=None,
             pc1_quality: DataFrame | None = None,
             stability: DataFrame | None = None,
             consensus: DataFrame | None = None,
             search: DataFrame | None = None,
             selections: DataFrame | None = None,
             method_comparison: DataFrame | None = None,
             candidate_comparison: DataFrame | None = None,
             admission: DataFrame | None = None,
             tie_break: DataFrame | None = None,
             tie_break_by_fold: DataFrame | None = None,
             elastic_net_search: DataFrame | None = None,
             elastic_net_stability: DataFrame | None = None,
             threshold_search: DataFrame | None = None,
             fold_performance: DataFrame | None = None,
             final_comparison: DataFrame | None = None,
             feature_set_summary: DataFrame | None = None,
             pareto: DataFrame | None = None,
             family_agreement: DataFrame | None = None,
             family_coverage: DataFrame | None = None) -> list[Path]:
    """Write every figure this stage supports."""
    from allmeta.feature_selection import algorithms, config

    apply_style()
    for directory in FIG_SUBDIRS:
        directory.mkdir(parents=True, exist_ok=True)

    thresholds = (config.HIERARCHICAL_CONFIG['distance_thresholds']
                  if thresholds is None else thresholds)
    if solution is None:
        solution = algorithms.fit('hierarchical', distance, linkage_method='average',
                                  distance_threshold=0.25)

    written = [
        plot_feature_dendrogram(distance, thresholds,
                                FIG_SPACE / 'feature_dendrogram.png'),
        plot_ordered_correlation(rho, solution,
                                 FIG_SPACE / 'ordered_correlation.png'),
        plot_metric_comparison(table, FIG_AUTO_SWEEP / 'metric_comparison.png'),
        plot_metric_vs_k(table, FIG_AUTO_SWEEP / 'metric_vs_k.png'),
        plot_multi_metric_heatmap(table, FIG_AUTO_SWEEP / 'multi_metric_heatmap.png'),
        plot_manual_cluster_quality(loadings, FIG_MANUAL_PC1 / 'pc1_quality.png'),
    ]

    if pc1_quality is not None and not pc1_quality.empty:
        written.append(plot_pc1_quality(pc1_quality,
                                        FIG_AUTO_SWEEP / 'pc1_quality.png'))
    if stability is not None and not stability.empty:
        written.append(plot_stability(stability, FIG_AUTO_SWEEP / 'stability.png'))
    if consensus is not None and not consensus.empty:
        written.append(plot_consensus(consensus, FIG_AUTO_SWEEP / 'consensus_matrix.png'))
    if search is not None and not search.empty:
        written.append(plot_selection(search, selections,
                                      FIG_AUTO_SELECT / 'candidate_selection.png'))

    # Phase 9 (Step 16.6).
    if candidate_comparison is not None and not candidate_comparison.empty:
        written.append(plot_predictive_comparison(
            candidate_comparison,
            method_comparison if method_comparison is not None else DataFrame(),
            FIG_AUTO_SELECT / 'predictive_comparison.png'))
        written.append(plot_error_vs_quality(
            candidate_comparison, FIG_AUTO_SELECT / 'error_vs_quality.png'))
    if admission is not None and not admission.empty:
        written.append(plot_admission(
            admission, FIG_AUTO_SELECT / 'method_admission.png'))
    if tie_break_by_fold is not None and not tie_break_by_fold.empty:
        written.append(plot_tie_break_replay(
            tie_break_by_fold,
            tie_break if tie_break is not None else DataFrame(),
            FIG_AUTO_SELECT / 'tie_break_replay.png'))

    # Phases 10-12 (Step 16.8). One set of figures per feature family, because a grid
    # heatmap averaged over families would show a surface no single model has.
    if elastic_net_search is not None and not elastic_net_search.empty:
        for name, block in elastic_net_search.groupby('feature_set'):
            written.append(plot_elastic_net_grid(
                block, family_dir(name) / 'elastic_net_grid.png'))
            written.append(plot_elastic_net_path(
                block, family_dir(name) / 'elastic_net_path.png'))
    if elastic_net_stability is not None and not elastic_net_stability.empty:
        for name, block in elastic_net_stability.groupby('feature_set'):
            written.append(plot_elastic_net_stability(
                block, family_dir(name) / 'elastic_net_stability.png'))
    if threshold_search is not None and not threshold_search.empty:
        for name, block in threshold_search.groupby('feature_set'):
            written.append(plot_threshold_search(
                block, family_dir(name) / 'stability_threshold.png'))

    # Phases 14-19 (Step 16.9). The only figures whose numbers are estimates.
    if fold_performance is not None and not fold_performance.empty:
        written.append(plot_outer_fold_performance(
            fold_performance,
            final_comparison if final_comparison is not None else DataFrame(),
            FIG_COMPARISON / 'outer_fold_performance.png'))
    if feature_set_summary is not None and not feature_set_summary.empty:
        written.append(plot_final_comparison(
            feature_set_summary, pareto if pareto is not None else DataFrame(),
            FIG_COMPARISON / 'pipeline_comparison.png'))
    if family_agreement is not None and not family_agreement.empty:
        written.append(plot_family_agreement(
            family_agreement,
            family_coverage if family_coverage is not None else DataFrame(),
            FIG_COMPARISON / 'information_family_agreement.png'))

    return written


# ---------------------------------------------------------------------------
# Phase 9: comparing the methods on predictive error (Step 16.6)
# ---------------------------------------------------------------------------

def plot_predictive_comparison(candidates: DataFrame, comparison: DataFrame,
                               path: Path, criterion: str | None = None) -> Path:
    """Each method's whole error distribution, against the mean baseline.

    The figure for the finding that decides Phase 9. Three of the five methods have no
    candidate anywhere in their grid that beats predicting the training mean -- their
    entire violin sits to the wrong side of the baseline line -- and two of those three
    are the methods that won every cluster-quality and stability comparison.

    The distribution rather than the winner, because the audit's claim is about the
    pool: a randomly chosen candidate is roughly three times worse than the baseline, so
    selecting helps by avoiding a bad pool rather than by finding a strong
    representation. That is a statement about the spread, and a bar chart of five minima
    would hide it.
    """
    from allmeta.feature_selection import config

    criterion = (f'{config.SELECTION_CRITERION}_mean' if criterion is None
                 else criterion)
    apply_style()
    figure, axes = plt.subplots(figsize=(9.5, 5.5))

    if candidates.empty or criterion not in candidates:
        annotate_empty(axes, 'no scored candidates')
        return save(figure, path)

    order = _ordered_methods(candidates)
    data = candidates[candidates[criterion].notna()]

    sns.violinplot(data=data, x='method', y=criterion, order=order, ax=axes,
                   inner=None, cut=0, density_norm='width', linewidth=0.8,
                   color='0.88')
    sns.stripplot(data=data, x='method', y=criterion, order=order, ax=axes,
                  size=2.5, alpha=0.45, jitter=0.28, color='0.35')

    baseline = data.get('baseline_criterion_mean')
    if baseline is not None and baseline.notna().any():
        level = float(baseline.dropna().iloc[0])
        axes.axhline(level, color='crimson', linestyle='--', linewidth=1.4,
                     label=f'mean baseline ({level:.5f})')

    if not comparison.empty and 'clustering_method' in comparison:
        for position, method in enumerate(order):
            row = comparison[comparison['clustering_method'] == method]
            if row.empty or criterion not in row or pd.isna(row[criterion].iloc[0]):
                continue
            record = row.iloc[0]
            admitted = bool(record.get('admitted', True))
            axes.scatter([position], [record[criterion]], s=110, zorder=5,
                         marker='o' if admitted else 'X',
                         facecolor='seagreen' if admitted else 'none',
                         edgecolor='seagreen' if admitted else 'crimson',
                         linewidth=1.6)

    # Counted from the candidate table rather than annotated by hand, so the figure
    # cannot drift from the data behind it.
    if 'beats_baseline' in data:
        beaten = data.groupby('method')['beats_baseline'].sum()
        for position, method in enumerate(order):
            axes.annotate(f'{int(beaten.get(method, 0))}/{int((data["method"] == method).sum())}'
                          '\nbeat baseline',
                          xy=(position, axes.get_ylim()[1]), xytext=(0, -4),
                          textcoords='offset points', ha='center', va='top',
                          fontsize=7.5, color='0.35')

    axes.set_yscale('log')
    axes.set_xlabel('')
    axes.set_ylabel(f'{criterion}  (log scale, lower is better)')
    axes.set_title('Predictive error by clustering method, whole grid\n'
                   'filled circle = admitted method winner, cross = rejected by the '
                   'admission gate', fontsize=10)
    axes.legend(loc='lower right', fontsize=8)

    return save(figure, path)


def plot_error_vs_quality(candidates: DataFrame, path: Path,
                          criterion: str | None = None) -> Path:
    """Predictive error against the four cluster-quality axes, coloured by method.

    The audit's headline: cluster quality does not predict predictive usefulness.
    Hierarchical and HDBSCAN sit at the good end of every quality axis and at the bad
    end of the error axis, and that pattern is the argument for the plan's Step 8.2 rule
    in one panel. Each panel carries its Spearman correlation so the reader is not
    invited to eyeball a trend that is not there.
    """
    from allmeta.feature_selection import config

    criterion = (f'{config.SELECTION_CRITERION}_mean' if criterion is None
                 else criterion)
    apply_style()

    panels = [('n_clusters', 'cluster count'),
              ('ari_mean', 'stability (mean ARI)'),
              ('mean_pc1_explained_variance_multi', 'PC1 retention (multi-member)'),
              ('within_between_delta', 'correlation separation')]

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    order = _ordered_methods(candidates) if not candidates.empty else []

    for ax, (column, label) in zip(axes.ravel(), panels):
        if candidates.empty or column not in candidates or criterion not in candidates:
            annotate_empty(ax, f'{column} not measured')
            continue

        data = candidates[[column, criterion, 'method']].dropna()
        if data.empty:
            annotate_empty(ax, f'{column} not measured')
            continue

        sns.scatterplot(data=data, x=column, y=criterion, hue='method',
                        hue_order=order, ax=ax, s=18, alpha=0.65,
                        edgecolor='none', legend=(ax is axes.ravel()[0]))

        rho = data[column].corr(data[criterion], method='spearman')
        ax.annotate(f'Spearman $\\rho$ = {rho:+.2f}', xy=(0.03, 0.03),
                    xycoords='axes fraction', fontsize=8.5, color='0.3')

        baseline = candidates.get('baseline_criterion_mean')
        if baseline is not None and baseline.notna().any():
            ax.axhline(float(baseline.dropna().iloc[0]), color='crimson',
                       linestyle='--', linewidth=1.1)

        ax.set_yscale('log')
        ax.set_xlabel(label)
        ax.set_ylabel(criterion)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        axes.ravel()[0].legend(handles, labels, fontsize=7.5, title=None,
                               loc='best', markerscale=1.6)

    figure.suptitle('Predictive error against cluster quality\n'
                    'dashed line = mean baseline; a method above it never beat '
                    'predicting the mean', fontsize=10.5)
    figure.tight_layout()

    return save(figure, path)


def plot_admission(gate: DataFrame, path: Path) -> Path:
    """The admission gate as a pass/fail matrix with the measured values on it.

    Printed as a figure because the gate is the one place Phase 9 rejects a method, and
    a rejection that lives only in a CSV column is a rejection nobody audits.
    """
    apply_style()
    figure, axes = plt.subplots(figsize=(8.5, 3.6))

    judged = [('extra_mean_assignment_confidence', 'posterior\nconfidence'),
              ('mean_pc1_explained_variance_multi', 'PC1 retention\n(multi-member)'),
              ('ari_mean', 'stability\n(mean ARI)')]
    present = [(c, label) for c, label in judged if c in gate.columns]

    if gate.empty or not present:
        annotate_empty(axes, 'admission gate not evaluated')
        return save(figure, path)

    methods = list(gate['method'])
    values = np.full((len(methods), len(present)), np.nan)
    text = np.empty_like(values, dtype=object)

    for row, method in enumerate(methods):
        record = gate[gate['method'] == method].iloc[0]
        failed = str(record.get('failed_criteria', '') or '')
        for column, (name, _) in enumerate(present):
            value = record.get(name, np.nan)
            if 'beats_baseline' in failed:
                # Excluded before the diagnostics were ever read; showing it as a
                # diagnostic failure would tell the wrong story about why.
                values[row, column] = 0.5
                text[row, column] = 'n/a'
            elif pd.isna(value):
                values[row, column] = 0.5
                text[row, column] = '--'
            else:
                values[row, column] = 0.0 if name in failed else 1.0
                text[row, column] = f'{value:.3f}'

    sns.heatmap(values, annot=text, fmt='', ax=axes, cbar=False,
                xticklabels=[label for _, label in present], yticklabels=methods,
                cmap=sns.color_palette(['#d94a4a', '#dddddd', '#4a9d6e'], as_cmap=False),
                vmin=0, vmax=1, linewidths=1.2, linecolor='white',
                annot_kws={'fontsize': 9})

    axes.set_title('Clustering-method admission gate\n'
                   'green = passes, red = fails, grey = not applicable  '
                   '(no criterion reads predictive error)', fontsize=10)
    axes.tick_params(axis='y', rotation=0)

    return save(figure, path)


def plot_tie_break_replay(per_fold: DataFrame, summary: DataFrame, path: Path) -> Path:
    """What each tie-break ordering would have selected, fold by fold.

    Read the spread, not the ranking. The orderings differ by 0.01-0.05 in mean held-out
    between-liquid R-squared while the fold-to-fold spread within any one of them is
    about 0.27, and no paired comparison across the 13 folds comes close to significance.
    The honest reading is that the tie-break is not measurable on this data, which is a
    stronger result than any ranking of the four.
    """
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                gridspec_kw={'width_ratios': [1.5, 1]})

    if per_fold.empty or 'between_liquid_r2' not in per_fold.columns:
        annotate_empty(axes[0], 'replay not run')
        annotate_empty(axes[1], 'replay not run')
        return save(figure, path)

    order = list(summary['ordering']) if not summary.empty else \
        sorted(per_fold['ordering'].unique())

    sns.stripplot(data=per_fold, x='ordering', y='between_liquid_r2', order=order,
                  ax=axes[0], size=6, alpha=0.7, jitter=0.18, hue='method',
                  dodge=False)
    for position, ordering in enumerate(order):
        mean = per_fold.loc[per_fold['ordering'] == ordering,
                            'between_liquid_r2'].mean()
        axes[0].plot([position - 0.3, position + 0.3], [mean, mean],
                     color='black', linewidth=2, zorder=5)

    axes[0].axhline(0.0, color='0.5', linewidth=1)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('held-out between-liquid $R^2$')
    axes[0].set_title('One point per outer fold; bar = mean\n'
                      'the spread within an ordering dwarfs the gap between them',
                      fontsize=9.5)
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].legend(fontsize=7.5, title=None, loc='upper left')

    if not summary.empty:
        counts = (per_fold.groupby(['ordering', 'method']).size()
                  .unstack(fill_value=0).reindex(order))
        counts.plot(kind='barh', stacked=True, ax=axes[1], width=0.7)
        axes[1].set_xlabel('outer folds')
        axes[1].set_ylabel('')
        axes[1].set_title('Which method each ordering selects\n'
                          "the tie-break, not predictive error, drives this",
                          fontsize=9.5)
        axes[1].legend(fontsize=7.5, title=None)

    figure.suptitle('Tie-break replay across the outer folds  '
                    '(diagnostic only -- computed on the folds it is scored on)',
                    fontsize=10)
    figure.tight_layout()

    return save(figure, path)


# ---------------------------------------------------------------------------
# Phases 10-12: Elastic Net (Step 16.8)
# ---------------------------------------------------------------------------

def _en_grid(table: DataFrame, value: str) -> DataFrame:
    """alpha x l1_ratio as a matrix, for the two heatmaps."""
    candidates = table[table['alpha'].notna()]

    return candidates.pivot_table(index='l1_ratio', columns='alpha', values=value,
                                  aggfunc='mean')


def plot_elastic_net_grid(table: DataFrame, path: Path,
                          criterion: str | None = None) -> Path:
    """Error and sparsity over the whole alpha x l1_ratio grid.

    Read the two panels together. The left one says where the error is low, the right
    one says how many inputs survive there, and the pairing is the only way to see
    whether a low-error region is low-error *because* it dropped something or in spite
    of it. A broad flat basin on the left with a steep gradient on the right is the
    signature of a penalty that is changing the model without changing what it predicts.
    """
    from allmeta.feature_selection import config

    criterion = (f'{config.SELECTION_CRITERION}_mean' if criterion is None
                 else criterion)
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    if table.empty or 'alpha' not in table.columns:
        annotate_empty(axes[0], 'no Elastic Net grid')
        annotate_empty(axes[1], 'no Elastic Net grid')
        return save(figure, path)

    error = _en_grid(table, criterion)
    sns.heatmap(error, ax=axes[0], cmap='viridis_r', cbar_kws={'label': criterion})
    axes[0].set_title(f'{criterion} (lower is better)', fontsize=10)

    if 'n_nonzero_mean' in table.columns:
        sparsity = _en_grid(table, 'n_nonzero_mean')
        sns.heatmap(sparsity, ax=axes[1], cmap='magma',
                    cbar_kws={'label': 'non-zero coefficients'})
        axes[1].set_title('mean non-zero coefficients across folds', fontsize=10)
    else:
        annotate_empty(axes[1], 'sparsity not recorded')

    for ax in axes:
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel('L1 ratio')
        # 60 alphas would overplot the axis completely.
        labels = [f'{float(t.get_text()):.2g}' if i % 8 == 0 else ''
                  for i, t in enumerate(ax.get_xticklabels())]
        ax.set_xticklabels(labels, rotation=0)

    figure.tight_layout()

    return save(figure, path)


def plot_elastic_net_path(table: DataFrame, path: Path,
                          criterion: str | None = None) -> Path:
    """Error and surviving-feature count against alpha, one line per L1 ratio.

    The one-SE band is drawn because it is the whole question here: if it spans most of
    the alpha axis then the penalty is not identifiable on this data and the tie-break,
    not the error, is choosing -- the same diagnosis Phase 8 reached for pooled RMSE.
    """
    from allmeta.feature_selection import config

    criterion = (f'{config.SELECTION_CRITERION}_mean' if criterion is None
                 else criterion)
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)

    candidates = table[table['alpha'].notna()] if 'alpha' in table else DataFrame()
    if candidates.empty:
        annotate_empty(axes[0], 'no Elastic Net grid')
        annotate_empty(axes[1], 'no Elastic Net grid')
        return save(figure, path)

    palette = sns.color_palette('viridis', candidates['l1_ratio'].nunique())
    for colour, (ratio, block) in zip(palette,
                                      candidates.groupby('l1_ratio', sort=True)):
        block = block.sort_values('alpha')
        axes[0].plot(block['alpha'], block[criterion], color=colour, linewidth=1.3,
                     label=f'{ratio:g}')
        if 'n_nonzero_mean' in block:
            axes[1].plot(block['alpha'], block['n_nonzero_mean'], color=colour,
                         linewidth=1.3)

    best = candidates.loc[candidates[criterion].idxmin()]
    se_column = criterion.replace('_mean', '_se')
    if se_column in candidates.columns:
        threshold = float(best[criterion] + best[se_column])
        axes[0].axhspan(float(candidates[criterion].min()), threshold,
                        color='seagreen', alpha=0.12)
        axes[0].axhline(threshold, color='seagreen', linewidth=1.1, linestyle=':',
                        label='one-SE band')
        inside = int((candidates[criterion] <= threshold).sum())
        axes[0].annotate(f'{inside} of {len(candidates)} candidates\ninside the band',
                         xy=(0.03, 0.93), xycoords='axes fraction', va='top',
                         fontsize=8.5, color='0.3')

    baseline = table.get('baseline_criterion_mean')
    if baseline is not None and baseline.notna().any():
        axes[0].axhline(float(baseline.dropna().iloc[0]), color='crimson',
                        linestyle='--', linewidth=1.2, label='mean baseline')

    axes[0].axvline(float(best['alpha']), color='0.3', linewidth=1, linestyle='-.')
    axes[1].axvline(float(best['alpha']), color='0.3', linewidth=1, linestyle='-.')

    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel(r'$\alpha$')
    axes[0].set_yscale('log')
    axes[0].set_ylabel(criterion)
    axes[1].set_ylabel('mean non-zero coefficients')
    axes[0].legend(fontsize=7.5, title='L1 ratio', ncol=2)
    axes[0].set_title('Error against penalty', fontsize=10)
    axes[1].set_title('Surviving inputs against penalty', fontsize=10)
    figure.tight_layout()

    return save(figure, path)


def plot_elastic_net_stability(stability: DataFrame, path: Path) -> Path:
    """Selection frequency and coefficient behaviour per input.

    ``n_fits`` is printed on the figure, not just in the CSV. At 13 liquids a frequency
    is a proportion out of a few dozen fits over a pool of 13, and a 0.85 from ten
    resamples read against a 0.85 from thirty is exactly the comparison this label
    exists to prevent.
    """
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    if stability.empty:
        annotate_empty(axes[0], 'stability not measured')
        annotate_empty(axes[1], 'stability not measured')
        return save(figure, path)

    frame = stability.sort_values('selection_frequency', ascending=True)
    colours = ['seagreen' if c >= 0.8 else '0.6'
               for c in frame['sign_consistency'].fillna(0.0)]
    axes[0].barh(frame['feature'], frame['selection_frequency'], color=colours)

    from allmeta.feature_selection import config
    axes[0].axvline(config.ELASTIC_NET_STABILITY_THRESHOLD, color='crimson',
                    linestyle='--', linewidth=1.2,
                    label=f'threshold {config.ELASTIC_NET_STABILITY_THRESHOLD:g}')
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel('selection frequency')
    n_fits = int(frame['n_fits'].iloc[0])
    axes[0].set_title(f'How often each input survives  ({n_fits} fits, resampled by '
                      'liquid)\ngreen = sign consistent in at least 80% of its fits',
                      fontsize=9.5)
    axes[0].legend(fontsize=8, loc='lower right')

    axes[1].errorbar(frame['mean_coefficient'], frame['feature'],
                     xerr=frame['std_coefficient'], fmt='o', markersize=4,
                     linewidth=1, capsize=2.5, color='0.25')
    axes[1].axvline(0.0, color='crimson', linewidth=1)
    axes[1].set_xlabel('coefficient (mean $\\pm$ sd across fits)')
    axes[1].set_title('Coefficient stability\na bar spanning zero is not a stable sign',
                      fontsize=9.5)

    figure.tight_layout()

    return save(figure, path)


def plot_threshold_search(table: DataFrame, path: Path,
                          criterion: str | None = None) -> Path:
    """Error and retained count against the stability threshold, nested vs diagnostic.

    The gap between the two lines is the measurement. The diagnostic scope computes
    selection frequencies over every liquid and then scores on those same liquids, so
    each held-out liquid helped choose the features it is used to test; the nested scope
    recomputes the frequencies inside each training fold. **The nested line should sit
    above the diagnostic one.** If it does not, the nesting is not doing what it claims
    and the fold bookkeeping is the first thing to check.
    """
    from allmeta.feature_selection import config

    criterion = (f'{config.SELECTION_CRITERION}_mean' if criterion is None
                 else criterion)
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharex=True)

    if table.empty or 'scope' not in table.columns:
        annotate_empty(axes[0], 'threshold search not run')
        annotate_empty(axes[1], 'threshold search not run')
        return save(figure, path)

    styles = {'nested': ('seagreen', '-', 'nested (leakage-safe)'),
              'diagnostic': ('crimson', '--', 'diagnostic (selection leaks)')}

    for scope, (colour, dash, label) in styles.items():
        block = table[table['scope'] == scope].sort_values('threshold')
        if block.empty:
            continue
        if criterion in block:
            axes[0].plot(block['threshold'], block[criterion], marker='o',
                         color=colour, linestyle=dash, label=label)
        count = ('n_features_median' if scope == 'nested' else 'n_features')
        if count in block:
            axes[1].plot(block['threshold'], block[count], marker='o', color=colour,
                         linestyle=dash, label=label)
        if scope == 'nested' and {'n_features_min', 'n_features_max'} <= set(block):
            axes[1].fill_between(block['threshold'], block['n_features_min'],
                                 block['n_features_max'], color=colour, alpha=0.15)

    baseline = table.get('baseline_criterion_mean')
    if baseline is not None and baseline.notna().any():
        axes[0].axhline(float(baseline.dropna().iloc[0]), color='0.4',
                        linestyle=':', linewidth=1.2, label='mean baseline')

    axes[0].set_ylabel(criterion)
    axes[0].set_title('Error against stability threshold', fontsize=10)
    axes[1].set_ylabel('features retained')
    axes[1].set_title('Retained inputs against threshold\n'
                      'band = fold-to-fold range under nesting', fontsize=9.5)
    for ax in axes:
        ax.set_xlabel('stability threshold')
        ax.legend(fontsize=8)
    figure.tight_layout()

    return save(figure, path)


# ---------------------------------------------------------------------------
# Phases 14-19: outer-fold performance and the final comparison (Step 16.9)
# ---------------------------------------------------------------------------

def plot_outer_fold_performance(fold_scores: DataFrame, comparison: DataFrame,
                                path: Path, model: str = 'ridge') -> Path:
    """Held-out error per pipeline, one point per outer fold.

    The only figure in this project whose numbers are performance estimates. Everything
    earlier is a search diagnostic.

    Points, not bars. Thirteen liquids means thirteen numbers per pipeline, and the
    fold-to-fold spread is large enough that a bar chart of means would imply a
    precision the study does not have. The one-SE band is shaded only when
    ``one_se_is_informative`` says it separates anything -- drawing a band that contains
    every pipeline would suggest a comparison had been made when none was possible.
    """
    apply_style()
    figure, axes = plt.subplots(figsize=(10.0, 5.2))

    if fold_scores.empty:
        annotate_empty(axes, 'nested cross-validation not run')
        return save(figure, path)

    view = fold_scores[fold_scores['model'] == model]
    if view.empty:
        annotate_empty(axes, f'no {model} predictions')
        return save(figure, path)

    order = (list(comparison['feature_set']) if not comparison.empty
             else sorted(view['feature_set'].unique()))

    sns.stripplot(data=view, x='feature_set', y='liquid_squared_error', order=order,
                  ax=axes, size=6, alpha=0.65, jitter=0.2, color='0.35')
    for position, name in enumerate(order):
        mean = view.loc[view['feature_set'] == name, 'liquid_squared_error'].mean()
        axes.plot([position - 0.3, position + 0.3], [mean, mean], color='black',
                  linewidth=2.2, zorder=5)

    if not comparison.empty:
        informative = bool(comparison['one_se_is_informative'].iloc[0])
        threshold = float(comparison['one_se_threshold'].iloc[0])
        if informative:
            axes.axhspan(float(comparison['mean'].min()), threshold,
                         color='seagreen', alpha=0.12)
            axes.axhline(threshold, color='seagreen', linestyle=':', linewidth=1.2,
                         label='one-SE band')
        else:
            # Said on the figure rather than left to the reader to infer from a band
            # that happens to cover everything.
            axes.annotate('one-SE band contains every pipeline:\n'
                          'these are not distinguishable at this sample size',
                          xy=(0.02, 0.97), xycoords='axes fraction', va='top',
                          fontsize=9, color='crimson')

    baseline = view[view['feature_set'].str.contains('baseline', case=False)]
    if not baseline.empty:
        axes.axhline(baseline['liquid_squared_error'].mean(), color='crimson',
                     linestyle='--', linewidth=1.2, label='mean baseline')

    axes.set_yscale('log')
    axes.set_xlabel('')
    axes.set_ylabel('held-out liquid squared error (log scale)')
    axes.set_title('Nested cross-validation: one point per held-out liquid\n'
                   'bar = mean across the outer folds', fontsize=10)
    axes.tick_params(axis='x', rotation=15)
    if axes.get_legend_handles_labels()[0]:
        axes.legend(fontsize=8)
    figure.tight_layout()

    return save(figure, path)


def plot_final_comparison(summary: DataFrame, pareto: DataFrame, path: Path) -> Path:
    """Performance against complexity, with intervals, and the Pareto front.

    The left panel is the plan's Step 16.9 Pareto plot. The right shows the
    between-liquid R-squared of every pipeline with its bootstrap interval and the
    measured mean-baseline line -- **not** zero, because under leave-one-out the mean
    predictor scores 1 - (n/(n-1))^2 = -0.1736 at thirteen liquids. A pipeline at
    R2 = -0.05 is better than the baseline, and drawing the reference at zero would
    invert that reading.
    """
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    if summary.empty:
        annotate_empty(axes[0], 'nested cross-validation not run')
        annotate_empty(axes[1], 'nested cross-validation not run')
        return save(figure, path)

    if not pareto.empty:
        for row in pareto.itertuples():
            on_front = bool(row.on_pareto_front)
            axes[0].scatter(row.n_features_median, row.rmse, s=110, zorder=4,
                            facecolor='seagreen' if on_front else 'none',
                            edgecolor='seagreen' if on_front else '0.5',
                            linewidth=1.6)
            axes[0].annotate(row.feature_set,
                             xy=(row.n_features_median, row.rmse),
                             xytext=(5, 4), textcoords='offset points', fontsize=8)
        front = pareto[pareto['on_pareto_front']].sort_values('n_features_median')
        if len(front) > 1:
            axes[0].plot(front['n_features_median'], front['rmse'], color='seagreen',
                         linewidth=1.1, linestyle='--', zorder=3)
    axes[0].set_xlabel('median features used (across outer folds)')
    axes[0].set_ylabel('outer-fold RMSE')
    axes[0].set_title('Performance against complexity\n'
                      'filled = on the Pareto front', fontsize=10)

    ordered = summary.sort_values('between_liquid_r2')
    positions = np.arange(len(ordered))
    if {'ci_lo', 'ci_hi'} <= set(ordered.columns):
        errors = np.vstack([
            ordered['between_liquid_r2'] - ordered['ci_lo'],
            ordered['ci_hi'] - ordered['between_liquid_r2']])
        axes[1].errorbar(ordered['between_liquid_r2'], positions, xerr=np.abs(errors),
                         fmt='o', markersize=6, capsize=3, linewidth=1.2,
                         color='0.25')
    else:
        axes[1].scatter(ordered['between_liquid_r2'], positions, s=60, color='0.25')

    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(ordered['feature_set'])
    n_liquids = int(ordered.get('n_liquids', pd.Series([13])).iloc[0])
    baseline = 1 - (n_liquids / (n_liquids - 1)) ** 2
    axes[1].axvline(baseline, color='crimson', linestyle='--', linewidth=1.3,
                    label=f'mean baseline ({baseline:.3f})')
    axes[1].axvline(0.0, color='0.6', linewidth=1, label='zero (NOT the baseline)')
    axes[1].set_xlabel('held-out between-liquid $R^2$')
    axes[1].set_title('Between-liquid $R^2$ with bootstrap interval\n'
                      'intervals resample LIQUIDS, not rows', fontsize=10)
    axes[1].legend(fontsize=8, loc='lower right')

    figure.tight_layout()

    return save(figure, path)


def plot_family_agreement(agreement: DataFrame, coverage: DataFrame,
                          path: Path) -> Path:
    """Do the pipelines pick the same descriptors, or the same information families?

    The gap between the two bars on the left is the plan's Step 18 question answered:
    where the family bar clears the feature bar, the pipelines disagree on descriptors
    and agree on the underlying chemistry.
    """
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    if agreement.empty:
        annotate_empty(axes[0], 'fewer than two pipelines to compare')
    else:
        pairs = [f"{r.feature_set_a}\nvs {r.feature_set_b}"
                 for r in agreement.itertuples()]
        positions = np.arange(len(pairs))
        width = 0.38
        axes[0].bar(positions - width / 2, agreement['feature_jaccard'], width,
                    label='exact descriptors', color='0.6')
        axes[0].bar(positions + width / 2, agreement['family_jaccard'], width,
                    label='information families', color='seagreen')
        axes[0].set_xticks(positions)
        axes[0].set_xticklabels(pairs, fontsize=7.5)
        axes[0].set_ylabel('mean within-fold Jaccard')
        axes[0].set_ylim(0, 1)
        axes[0].set_title('Agreement by descriptor and by family\n'
                          'green above grey = same chemistry, different columns',
                          fontsize=9.5)
        axes[0].legend(fontsize=8)

    if coverage.empty:
        annotate_empty(axes[1], 'no family coverage measured')
    else:
        grid = coverage.pivot_table(index='family', columns='feature_set',
                                    values='coverage', aggfunc='max').fillna(0.0)
        sns.heatmap(grid, ax=axes[1], cmap='Greens', vmin=0, vmax=1, annot=True,
                    fmt='.2f', linewidths=0.8, linecolor='white',
                    cbar_kws={'label': 'fraction of outer folds represented'})
        axes[1].set_title('How often each information family is represented',
                          fontsize=9.5)
        axes[1].set_xlabel('')
        axes[1].set_ylabel('')

    figure.tight_layout()

    return save(figure, path)
