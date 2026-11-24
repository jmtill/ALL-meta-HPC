"""Exploratory and diagnostic figures for the subsets, screens and clusterings.

Separate from the analysis modules on purpose: nothing here fits or selects
anything, it only reads the artefacts the earlier stages wrote. That keeps the
figures reproducible from disk and stops plotting code from quietly changing a
result.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from allmeta import columns as cols
from allmeta.paths import (
    CHEM_CLUSTER_CSV,
    CLUSTERING_DIR,
    FEATURE_SELECTION_DIR,
    FIG_CLUSTERING,
    FIG_DIR,
    FIG_FEATURE_SELECTION,
)
from allmeta.plotting import (
    DIVERGING,
    SEQUENTIAL,
    SUBSET_COLOURS,
    annotate_empty,
    apply_style,
    save,
)
from allmeta.screening import correlation_distance, correlation_matrix, load_feature_sets
from allmeta.subsets import load_hspip, load_subset

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

FIG_EDA: Path = FIG_DIR / 'eda'

# Subsets A and B differ only in which liquids they contain, so any descriptor whose
# distribution shifts between them is one the curation acted on.
SHIFT_TOP_N: int = 12


# --------------------------------------------------------------------------- EDA

def plot_subset_sizes(path: Path) -> Path:
    """Where the liquids go: master sheet down to each subset, and why they leave."""
    hspip = load_hspip()
    frame_a, frame_b = load_subset('A'), load_subset('B')
    dropped = pd.read_csv(CLUSTERING_DIR.parent / 'subsets' / 'dropped_liquids.csv')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    stages = ['cleaned HSPiP', 'subset A\n(complete rows)', 'subset B\n(curated)']
    counts = [len(hspip), len(frame_a), len(frame_b)]
    bars = axes[0].bar(stages, counts, color=['0.6', SUBSET_COLOURS['A'],
                                              SUBSET_COLOURS['B']])
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, count + 12, str(count),
                     ha='center', fontsize=11)
    axes[0].set_ylabel('liquids')
    axes[0].set_title('Liquids surviving each stage')
    axes[0].set_ylim(0, max(counts) * 1.15)

    reasons = (dropped[dropped['dropped_from'] == 'B']['reason']
               .value_counts().head(14).sort_values())
    axes[1].barh(reasons.index, reasons.values, color=SUBSET_COLOURS['B'])
    axes[1].set_xlabel('liquids removed')
    axes[1].set_title('Why curation removed liquids (top categories)')
    axes[1].tick_params(axis='y', labelsize=8)

    fig.tight_layout()

    return save(fig, path)


def plot_missingness(path: Path) -> Path:
    """Which descriptors are missing, and for which liquids.

    Thirty-two liquids leave subset A. This shows the loss is one block of
    co-missing descriptors rather than scattered gaps.
    """
    hspip = load_hspip()
    numeric = hspip[list(cols.NUM_COLUMNS)]
    incomplete = numeric[numeric.isna().any(axis=1)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={'width_ratios': [1, 1.4]})

    per_column = numeric.isna().sum()
    per_column = per_column[per_column > 0].sort_values()
    if per_column.empty:
        annotate_empty(axes[0], 'no missing descriptor values')
    else:
        axes[0].barh(per_column.index, per_column.values, color='#c44e52')
        axes[0].set_xlabel('liquids with the value missing')
        axes[0].set_title(f'Descriptors with gaps ({len(per_column)} of 58)')
        axes[0].tick_params(axis='y', labelsize=8)

    if incomplete.empty:
        annotate_empty(axes[1], 'every liquid is complete')
    else:
        sns.heatmap(incomplete.isna().T, cmap=['#e8e8e8', '#c44e52'], cbar=False,
                    ax=axes[1])
        axes[1].set_title(f'Missingness pattern for the {len(incomplete)} '
                          f'liquids dropped from subset A')
        axes[1].set_xlabel('liquid')
        axes[1].set_xticks([])
        axes[1].tick_params(axis='y', labelsize=6)

    fig.tight_layout()

    return save(fig, path)


def plot_curation_shift(path: Path, top_n: int = SHIFT_TOP_N) -> Path:
    """Descriptors whose distribution moved most when the curated liquids left.

    Ranked by standardised difference in means, so the panel shows what curation
    actually changed about the chemistry rather than an arbitrary selection.
    """
    frame_a, frame_b = load_subset('A'), load_subset('B')
    removed = frame_a[~frame_a['Name'].isin(set(frame_b['Name']))]

    shifts = []
    for feature in cols.NUM_COLUMNS:
        kept_values, gone_values = frame_b[feature], removed[feature]
        pooled = np.sqrt((kept_values.var() + gone_values.var()) / 2)
        if pooled > 0:
            shifts.append((feature, abs(kept_values.mean() - gone_values.mean()) / pooled))

    ranked = [f for f, _ in sorted(shifts, key=lambda x: -x[1])[:top_n]]

    long = pd.concat([
        frame_b[ranked].melt(var_name='feature', value_name='value').assign(group='kept (subset B)'),
        removed[ranked].melt(var_name='feature', value_name='value').assign(group='removed by curation'),
    ])

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, feature in zip(axes.ravel(), ranked):
        sns.kdeplot(data=long[long['feature'] == feature], x='value', hue='group',
                    fill=True, alpha=0.35, common_norm=False, ax=ax, legend=False,
                    palette={'kept (subset B)': SUBSET_COLOURS['B'],
                             'removed by curation': '0.5'})
        ax.set_title(cols.axis_label(feature), fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
    for ax in axes.ravel()[len(ranked):]:
        ax.set_visible(False)

    handles = [plt.Line2D([], [], color=SUBSET_COLOURS['B'], lw=6, alpha=0.6,
                          label='kept (subset B)'),
               plt.Line2D([], [], color='0.5', lw=6, alpha=0.6,
                          label='removed by curation')]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False)
    fig.suptitle('Descriptors whose distribution shifted most under curation',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    return save(fig, path)


def plot_liquid_space(path: Path) -> Path:
    """Subset A in two dimensions, with the curated and labelled liquids marked.

    Answers whether curation carved out a region of chemistry or thinned the whole
    space evenly, and where the labelled liquids sit relative to everything
    the model would have to generalise to.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    frame_a, frame_b, frame_c = (load_subset(k) for k in 'ABC')
    scaled = StandardScaler().fit_transform(frame_a[list(cols.NUM_COLUMNS)])
    coords = PCA(n_components=2, random_state=48).fit(scaled)
    projected = coords.transform(scaled)

    kept = frame_a['Name'].isin(set(frame_b['Name'])).to_numpy()
    labelled = frame_a['Name'].isin(set(frame_c['Name'])).to_numpy()

    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.scatter(projected[~kept, 0], projected[~kept, 1], s=14, c='0.75',
               label=f'removed by curation ({int((~kept).sum())})')
    ax.scatter(projected[kept, 0], projected[kept, 1], s=14,
               c=SUBSET_COLOURS['B'], alpha=0.7,
               label=f'kept in subset B ({int(kept.sum())})')
    ax.scatter(projected[labelled, 0], projected[labelled, 1], s=190, marker='*',
               c='#c44e52', edgecolor='k', linewidth=0.6, zorder=5,
               label=f'labelled liquids ({int(labelled.sum())})')

    # The labelled liquids sit close together, so fixed offsets would overprint.
    # Fan the annotations around a circle and draw a leader line to each point.
    names = list(frame_a.loc[labelled, 'Name'])
    for i, (name, (x, y)) in enumerate(zip(names, projected[labelled])):
        angle = 2 * np.pi * i / max(len(names), 1)
        ax.annotate(
            name, (x, y), fontsize=8,
            xytext=(34 * np.cos(angle), 34 * np.sin(angle)),
            textcoords='offset points',
            ha='center', va='center',
            bbox={'boxstyle': 'round,pad=0.25', 'fc': 'white', 'ec': '0.7', 'alpha': 0.85},
            arrowprops={'arrowstyle': '-', 'color': '0.5', 'linewidth': 0.7},
        )

    variance = coords.explained_variance_ratio_ * 100
    ax.set_xlabel(f'PC1 ({variance[0]:.1f}% of variance)')
    ax.set_ylabel(f'PC2 ({variance[1]:.1f}%)')
    ax.set_title('Liquid space of subset A')
    ax.legend(loc='best', fontsize=9)

    fig.tight_layout()

    return save(fig, path)


def plot_eda() -> list[Path]:
    """Every subset EDA figure."""
    apply_style()
    FIG_EDA.mkdir(parents=True, exist_ok=True)

    return [
        plot_subset_sizes(FIG_EDA / 'subset_sizes_and_removal_reasons.png'),
        plot_missingness(FIG_EDA / 'missingness.png'),
        plot_curation_shift(FIG_EDA / 'curation_distribution_shift.png'),
        plot_liquid_space(FIG_EDA / 'liquid_space_pca.png'),
    ]


# --------------------------------------------------------------- feature screening

def plot_spearman_heatmap(subset: str, path: Path) -> Path:
    """Full Spearman matrix, ordered by the same linkage used to cluster features.

    Ordering by the dendrogram rather than sheet order is what makes the blocks of
    redundant descriptors visible; the original notebook's sheet-ordered version hid
    them.
    """
    frame = load_subset(subset)
    features = cols.screening_features()
    rho = correlation_matrix(frame[features])
    order = [features[i] for i in leaves_list(
        linkage(squareform(correlation_distance(rho), checks=False), method='average'))]

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(rho.loc[order, order], vmin=0, vmax=1, cmap='rocket_r', square=True,
                linewidths=0.2, linecolor='white',
                cbar_kws={'label': '|Spearman rho|'}, ax=ax)
    ax.set_title(f'Subset {subset}: absolute Spearman correlation, '
                 f'clustered ordering ({len(features)} descriptors)')
    ax.tick_params(axis='x', labelrotation=90, labelsize=7)
    ax.tick_params(axis='y', labelrotation=0, labelsize=7)

    fig.tight_layout()

    return save(fig, path)


def plot_k_sweep(subset: str, path: Path) -> Path:
    """Silhouette against k for each feature-clustering method, with the pick marked.

    The chosen k is an argmax over this curve, so showing the curve is what makes
    that choice auditable -- a flat curve means the pick was close to arbitrary.
    """
    sweep = pd.read_csv(FEATURE_SELECTION_DIR / subset / 'fcluster_k_sweep.csv')

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in sweep.groupby('method'):
        group = group.sort_values('k')
        line, = ax.plot(group['k'], group['silhouette'], marker='o', markersize=4,
                        label=method)
        chosen = group[group['selected_k']]
        ax.scatter(chosen['k'], chosen['silhouette'], s=190, marker='*',
                   color=line.get_color(), edgecolor='k', zorder=5)

    ax.set_xlabel('number of feature clusters (k)')
    ax.set_ylabel('silhouette of the feature partition')
    ax.set_title(f'Subset {subset}: choosing k for feature clustering '
                 f'(stars mark the selected k)')
    ax.legend()

    fig.tight_layout()

    return save(fig, path)


def plot_method_overlap(subset: str, path: Path) -> Path:
    """Which descriptors each screening method retained, as a presence matrix.

    Rows ordered by how many methods kept them, so the consensus core and the
    method-specific picks separate visually.
    """
    sets = load_feature_sets(subset)
    ordered_sets = {k: v for k, v in sets.items() if k not in ('all', 'no_composition')}

    matrix = DataFrame(
        {name: [f in set(members) for f in cols.NUM_COLUMNS]
         for name, members in ordered_sets.items()},
        index=list(cols.NUM_COLUMNS),
    ).astype(int)
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]
    matrix = matrix[matrix.sum(axis=1) > 0]

    fig, ax = plt.subplots(figsize=(1.3 * len(ordered_sets) + 4, max(8, len(matrix) * 0.26)))
    sns.heatmap(matrix, cmap=['#f0f0f0', '#4c72b0'], cbar=False, linewidths=0.4,
                linecolor='white', ax=ax)
    ax.set_title(f'Subset {subset}: descriptors retained by each screening method')
    ax.tick_params(axis='x', labelrotation=45, labelsize=9)
    ax.tick_params(axis='y', labelrotation=0, labelsize=7)
    for label in ax.get_xticklabels():
        label.set_ha('right')

    fig.tight_layout()

    return save(fig, path)


def plot_screening() -> list[Path]:
    """Every feature-screening diagnostic, per subset."""
    apply_style()
    written: list[Path] = []

    for subset in ('A', 'B'):
        out = FIG_FEATURE_SELECTION / subset
        out.mkdir(parents=True, exist_ok=True)
        written += [
            plot_spearman_heatmap(subset, out / 'spearman_heatmap.png'),
            plot_k_sweep(subset, out / 'fcluster_k_sweep.png'),
            plot_method_overlap(subset, out / 'method_overlap.png'),
        ]

    return written


# ------------------------------------------------------------------- clustering

# Reference feature sets always shown alongside the best performer, so "best" is
# read against a fixed baseline rather than on its own.
REFERENCE_SETS: tuple[str, ...] = ('all', 'manual')
EMBED_MODELS: tuple[str, ...] = ('kmeans_k3', 'kmeans_k5')
DENSITY_PREFIXES: tuple[str, ...] = ('dbscan_', 'hdbscan_')


def _metrics() -> DataFrame:
    return pd.read_csv(CLUSTERING_DIR / 'metrics.csv')


def _labels() -> DataFrame:
    return pd.read_csv(CLUSTERING_DIR / 'labels.csv.gz')


def resolve_feature_set(subset: str, name: str) -> list[str]:
    """Members of a named feature set as it appears in the clustering metrics.

    ``supervised_consensus`` is evaluated on A and B but written by the supervised
    stage rather than the per-subset screening, so it is absent from that subset's
    feature_sets.json and has to be resolved separately.
    """
    sets = load_feature_sets(subset)
    if name in sets:
        return list(sets[name])

    if name == 'supervised_consensus':
        from allmeta.supervised import FREQUENCY_PATH, consensus_features
        return consensus_features(pd.read_csv(FREQUENCY_PATH))

    raise KeyError(f'unknown feature set {name!r} for subset {subset}')


def best_feature_set(subset: str, metric: str = 'silhouette') -> str:
    """Feature set with the best median ``metric`` across the model grid."""
    metrics = _metrics()
    view = metrics[metrics['subset'] == subset]
    ranked = view.groupby('feature_set')[metric].median().sort_values(ascending=False)

    return str(ranked.index[0])


def plot_embeddings(subset: str, path: Path) -> Path:
    """PCA and UMAP scatters for the best and reference feature sets.

    The labelled liquids are starred in every panel: whether they land in one
    cluster or scatter across several is what decides whether these partitions could
    ever stratify a dispensing-error experiment.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    frame = load_subset(subset)
    labels = _labels()
    sets = load_feature_sets(subset)
    labelled = set(load_subset('C')['Name'])

    chosen = [best_feature_set(subset)]
    chosen += [s for s in REFERENCE_SETS if s in sets and s not in chosen]
    members = {name: resolve_feature_set(subset, name) for name in chosen}

    rows = [(fs, rep) for fs in chosen for rep in ('pca', 'umap')]
    fig, axes = plt.subplots(len(rows), len(EMBED_MODELS),
                             figsize=(6.2 * len(EMBED_MODELS), 5.4 * len(rows)))
    axes = np.atleast_2d(axes)

    for row, (feature_set, representation) in enumerate(rows):
        features = members[feature_set]
        scaled = StandardScaler().fit_transform(frame[features])
        if representation == 'pca':
            coords = PCA(n_components=2, random_state=48).fit_transform(scaled)
            axis_label = 'PC'
        else:
            import umap
            coords = umap.UMAP(n_neighbors=25, min_dist=0.3, n_components=2,
                               random_state=48).fit_transform(scaled)
            axis_label = 'UMAP'

        is_labelled = frame['Name'].isin(labelled).to_numpy()

        for col, model in enumerate(EMBED_MODELS):
            ax = axes[row, col]
            found = labels[(labels['subset'] == subset)
                           & (labels['feature_set'] == feature_set)
                           & (labels['representation'] == representation)
                           & (labels['model'] == model)]
            if found.empty:
                annotate_empty(ax, f'{feature_set}/{representation}/{model} not in labels')
                continue

            assignment = found.set_index('Name')['label'].reindex(frame['Name']).to_numpy()
            ax.scatter(coords[:, 0], coords[:, 1], c=assignment, cmap='tab10', s=12,
                       alpha=0.75)
            ax.scatter(coords[is_labelled, 0], coords[is_labelled, 1], s=180,
                       marker='*', c='none', edgecolor='k', linewidth=1.2, zorder=5)
            n_occupied = len(set(assignment[is_labelled]))
            ax.set_title(f'{feature_set} ({len(features)}f) / {representation} / {model}\n'
                         f'labelled liquids occupy {n_occupied} of '
                         f'{len(set(assignment))} clusters', fontsize=10)
            ax.set_xlabel(f'{axis_label}1')
            ax.set_ylabel(f'{axis_label}2')

    fig.suptitle(f'Subset {subset}: cluster structure '
                 f'(stars = the {int(is_labelled.sum())} liquids with measured '
                 f'error)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    return save(fig, path)


def plot_elbow(subset: str, path: Path) -> Path:
    """Silhouette against k for the KMeans sweep, one line per feature set.

    Separated by representation because the three disagree sharply about how much
    structure exists, which a single averaged curve would hide.
    """
    metrics = _metrics()
    view = metrics[(metrics['subset'] == subset)
                   & metrics['model'].str.startswith('kmeans_k')].copy()
    view['k'] = view['model'].str.replace('kmeans_k', '', regex=False).astype(int)

    representations = sorted(view['representation'].unique())
    fig, axes = plt.subplots(1, len(representations),
                             figsize=(5.6 * len(representations), 5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, representation in zip(axes, representations):
        panel = view[view['representation'] == representation]
        for feature_set, group in panel.groupby('feature_set'):
            group = group.sort_values('k')
            ax.plot(group['k'], group['silhouette'], marker='o', markersize=3,
                    linewidth=1.3, label=feature_set)
        ax.set_title(representation)
        ax.set_xlabel('k')
    axes[0].set_ylabel('silhouette')
    axes[-1].legend(fontsize=7, loc='best', ncol=1)

    fig.suptitle(f'Subset {subset}: silhouette against k, by representation')
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    return save(fig, path)


def plot_cluster_sizes(subset: str, path: Path) -> Path:
    """Cluster size distribution at k=5 for each feature set.

    A high silhouette earned by one giant cluster plus a few outliers is a different
    thing from a genuinely balanced partition, and only the sizes distinguish them.
    """
    labels = _labels()
    view = labels[(labels['subset'] == subset)
                  & (labels['representation'] == 'pca')
                  & (labels['model'] == 'kmeans_k5')]

    if view.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        annotate_empty(ax, 'no k=5 PCA labels available')
        return save(fig, path)

    sizes = (view.groupby(['feature_set', 'label']).size()
             .rename('n_liquids').reset_index())
    order = sorted(sizes['feature_set'].unique())

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=sizes, x='feature_set', y='n_liquids', hue='label',
                order=order, ax=ax, palette='tab10')
    ax.set_title(f'Subset {subset}: cluster sizes at k=5 (PCA representation)')
    ax.set_xlabel('')
    ax.set_ylabel('liquids in cluster')
    ax.tick_params(axis='x', labelrotation=35, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    ax.legend(title='cluster', fontsize=8, ncol=5)

    fig.tight_layout()

    return save(fig, path)


def plot_density_sweep(subset: str, path: Path) -> Path:
    """Silhouette and noise fraction against HDBSCAN's min_cluster_size.

    The density-method analogue of plot_elbow: min_cluster_size is HDBSCAN's
    tuning knob the way k is KMeans's, but unlike KMeans it can also refuse to
    assign points, so noise fraction is shown alongside silhouette rather than
    silhouette alone -- a high silhouette earned by calling most of the data noise
    is not a good partition.
    """
    metrics = _metrics()
    view = metrics[(metrics['subset'] == subset)
                   & metrics['model'].str.startswith('hdbscan_mcs')].copy()
    if view.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        annotate_empty(ax, 'no hdbscan rows available')
        return save(fig, path)

    view['min_cluster_size'] = (view['model'].str.replace('hdbscan_mcs', '', regex=False)
                                .astype(int))

    representations = sorted(view['representation'].unique())
    fig, axes = plt.subplots(2, len(representations),
                             figsize=(5.6 * len(representations), 8.5), sharex=True)
    axes = np.atleast_2d(axes)

    for col, representation in enumerate(representations):
        panel = view[view['representation'] == representation]
        for feature_set, group in panel.groupby('feature_set'):
            group = group.sort_values('min_cluster_size')
            axes[0, col].plot(group['min_cluster_size'], group['silhouette'],
                              marker='o', markersize=3, linewidth=1.3, label=feature_set)
            axes[1, col].plot(group['min_cluster_size'], group['noise_fraction'],
                              marker='o', markersize=3, linewidth=1.3, label=feature_set)
        axes[0, col].set_title(representation)
        axes[1, col].set_xlabel('min_cluster_size')

    axes[0, 0].set_ylabel('silhouette')
    axes[1, 0].set_ylabel('noise fraction')
    axes[0, -1].legend(fontsize=7, loc='best', ncol=1)

    fig.suptitle(f'Subset {subset}: HDBSCAN against min_cluster_size, by representation')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    return save(fig, path)


def plot_density_noise(subset: str, path: Path) -> Path:
    """Mean noise fraction for every DBSCAN/HDBSCAN variant, across feature sets.

    A variant that calls most of the data noise on a representation is not a
    usable partition there regardless of what its silhouette says on the
    remainder -- this is the compact view of where that happens.
    """
    metrics = _metrics()
    view = metrics[(metrics['subset'] == subset)
                   & metrics['model'].str.startswith(DENSITY_PREFIXES)]
    if view.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        annotate_empty(ax, 'no dbscan/hdbscan rows available')
        return save(fig, path)

    pivot = (view.groupby(['model', 'representation'])['noise_fraction']
             .mean().unstack('representation'))

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(pivot) + 1.5))
    sns.heatmap(pivot, vmin=0, vmax=1, cmap=SEQUENTIAL, annot=True, fmt='.2f',
                cbar_kws={'label': 'mean noise fraction'}, ax=ax)
    ax.set_title(f'Subset {subset}: mean noise fraction by model and representation\n'
                 f'(averaged across feature sets)')
    ax.set_xlabel('')
    ax.set_ylabel('')

    fig.tight_layout()

    return save(fig, path)


def plot_eps_curve(subset: str, path: Path) -> Path:
    """The k-distance curve DBSCAN's eps is picked from, for the best feature set.

    Makes the adaptive-eps mechanism (clustering.estimate_eps) visible: without
    this, the eps each dbscan_ms{k} row actually used is invisible in metrics.csv.
    """
    from allmeta.clustering import REPRESENTATIONS, embed, estimate_eps, k_distance_curve

    frame = load_subset(subset)
    feature_set = best_feature_set(subset)
    features = resolve_feature_set(subset, feature_set)
    data = frame[features]

    fig, axes = plt.subplots(1, len(REPRESENTATIONS),
                             figsize=(5.6 * len(REPRESENTATIONS), 5))
    axes = np.atleast_1d(axes)

    for ax, representation in zip(axes, REPRESENTATIONS):
        embedding = embed(data, representation)
        for min_samples in (5, 10):
            curve = k_distance_curve(embedding, min_samples)
            eps = estimate_eps(embedding, min_samples)
            knee = int(np.searchsorted(curve, eps))
            line, = ax.plot(curve, linewidth=1.3, label=f'min_samples={min_samples}')
            ax.scatter([knee], [eps], color=line.get_color(), zorder=5, s=30)
        ax.set_title(representation)
        ax.set_xlabel('points, sorted by k-distance')

    axes[0].set_ylabel('distance to k-th nearest neighbour')
    axes[-1].legend(fontsize=8, loc='best')

    fig.suptitle(f'Subset {subset}: DBSCAN eps selection ({feature_set})')
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    return save(fig, path)


def plot_metric_agreement(path: Path) -> Path:
    """Do the four metric families rank partitions the same way?

    If silhouette and stability disagree, "best" depends entirely on which metric is
    quoted -- worth knowing before any of these numbers is used to pick a feature
    set.
    """
    metrics = _metrics()
    columns = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'ari_chemical',
               'ami_chemical', 'stability_ari', 'error_eta_squared']
    available = [c for c in columns if c in metrics.columns and metrics[c].notna().any()]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7),
                             gridspec_kw={'width_ratios': [1, 1.25]})

    corr = metrics[available].corr(method='spearman')
    sns.heatmap(corr, vmin=-1, vmax=1, cmap=DIVERGING, square=True, annot=True,
                fmt='.2f', annot_kws={'size': 8}, linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'Spearman rho'}, ax=axes[0])
    axes[0].set_title(f'Rank agreement between metrics\n(across all {len(metrics)} clusterings)')
    axes[0].tick_params(axis='x', labelrotation=45, labelsize=8)
    axes[0].tick_params(axis='y', labelsize=8)
    for label in axes[0].get_xticklabels():
        label.set_ha('right')

    sns.scatterplot(data=metrics, x='silhouette', y='stability_ari',
                    hue='representation', style='subset', s=26, alpha=0.65,
                    ax=axes[1])
    axes[1].set_title('A clean-looking partition is not always a stable one')
    axes[1].set_xlabel('silhouette (geometric quality)')
    axes[1].set_ylabel('stability ARI (reproducibility under resampling)')
    axes[1].legend(fontsize=8, ncol=2)

    fig.tight_layout()

    return save(fig, path)


def plot_chemical_agreement(subset: str, path: Path, top_n: int = 18,
                            min_family: int = 3) -> Path:
    """Where the near-zero chemical ARI comes from.

    Two things drive it, and only the second is about the features:

    1. A granularity mismatch. There are 431 Butina families over ~1193 liquids
       against 5 property clusters, and 57% of those families are singletons. ARI
       punishes that mismatch heavily no matter how good the partition is, so a
       near-zero ARI here is largely structural.
    2. Whether families that are big enough to be split actually stay together. That
       is the real question, so the right panel restricts to families of at least
       ``min_family`` liquids and compares the observed spread against a random null
       -- the spread you would see if property clusters ignored chemistry entirely.
    """
    labels = _labels()
    chem = pd.read_csv(CHEM_CLUSTER_CSV, usecols=['Name', 'cluster_id']).drop_duplicates('Name')

    view = labels[(labels['subset'] == subset)
                  & (labels['representation'] == 'pca')
                  & (labels['model'] == 'kmeans_k5')
                  & (labels['feature_set'] == best_feature_set(subset))]
    merged = view.merge(chem, on='Name', how='inner').dropna(subset=['cluster_id'])

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))

    if merged.empty:
        for ax in axes:
            annotate_empty(ax, 'no overlap between property and chemical clusters')
        return save(fig, path)

    # Left: how the largest families distribute over property clusters.
    biggest = merged['cluster_id'].value_counts().head(top_n).index
    contingency = (merged[merged['cluster_id'].isin(biggest)]
                   .groupby(['cluster_id', 'label']).size().unstack(fill_value=0))
    contingency.div(contingency.sum(axis=1), axis=0).plot(
        kind='barh', stacked=True, ax=axes[0], colormap='tab10', width=0.85)
    axes[0].set_xlabel('fraction of the chemical family')
    axes[0].set_ylabel('chemical cluster id')
    axes[0].set_title(f'The {top_n} largest chemical families,\ndistributed over property clusters')
    axes[0].legend(title='property cluster', fontsize=8, ncol=5,
                   loc='upper center', bbox_to_anchor=(0.5, -0.12))

    # Right: observed spread against a random null, for families big enough to split.
    sizes = merged['cluster_id'].value_counts()
    proportions = merged['label'].value_counts(normalize=True).to_numpy()
    eligible = sizes[sizes >= min_family].index
    subset_families = merged[merged['cluster_id'].isin(eligible)]

    observed = subset_families.groupby('cluster_id')['label'].nunique()
    # Expected distinct clusters for a family of n liquids assigned at random, in
    # proportion to the observed cluster sizes.
    expected = pd.Series(
        {family: float(sum(1 - (1 - proportions) ** sizes[family])) for family in eligible}
    )

    bins = np.arange(0.5, max(observed.max(), int(np.ceil(expected.max()))) + 1.5)
    axes[1].hist([observed, expected], bins=bins,
                 label=['observed', 'random null'],
                 color=['#4c72b0', '0.65'])
    axes[1].set_xlabel('property clusters a chemical family is spread over')
    axes[1].set_ylabel('chemical families')
    axes[1].set_title(
        f'Families of at least {min_family} liquids '
        f'({len(eligible)} families, {int(sizes[eligible].sum())} liquids)\n'
        f'mean spread {observed.mean():.2f} observed vs {expected.mean():.2f} if random')
    axes[1].legend()

    n_singleton = int((sizes == 1).sum())
    fig.suptitle(
        f'Subset {subset}: property clusters vs Butina chemical clusters '
        f'({best_feature_set(subset)}, PCA, k=5)\n'
        f'{len(sizes)} chemical families vs 5 property clusters, '
        f'{n_singleton} of them singletons -- ARI is structurally near zero',
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    return save(fig, path)


def chemical_preservation(min_family: int = 3) -> DataFrame:
    """Observed vs random-null spread of chemical families across property clusters.

    Written as a table because it corrects a reading that the ARI alone invites.
    ARI against the Butina labels is ~0.01-0.06, which looks like "property clusters
    ignore chemistry". They do not: ARI is dominated by a granularity mismatch (431
    families, 57% of them singletons, against 5 property clusters). Comparing the
    observed spread with what random assignment would produce shows families are in
    fact held together substantially better than chance.
    """
    labels = _labels()
    chem = pd.read_csv(CHEM_CLUSTER_CSV,
                       usecols=['Name', 'cluster_id']).drop_duplicates('Name')

    rows: list[dict] = []
    for subset in ('A', 'B'):
        feature_set = best_feature_set(subset)
        for model in sorted(labels['model'].unique()):
            if not model.startswith('kmeans_k'):
                continue
            view = labels[(labels['subset'] == subset)
                          & (labels['representation'] == 'pca')
                          & (labels['model'] == model)
                          & (labels['feature_set'] == feature_set)]
            merged = view.merge(chem, on='Name', how='inner').dropna(subset=['cluster_id'])
            if merged.empty:
                continue

            sizes = merged['cluster_id'].value_counts()
            proportions = merged['label'].value_counts(normalize=True).to_numpy()
            eligible = sizes[sizes >= min_family].index
            if len(eligible) == 0:
                continue

            observed = (merged[merged['cluster_id'].isin(eligible)]
                        .groupby('cluster_id')['label'].nunique())
            expected = pd.Series(
                {f: float(sum(1 - (1 - proportions) ** sizes[f])) for f in eligible})

            rows.append({
                'subset': subset,
                'feature_set': feature_set,
                'model': model,
                'n_families': len(eligible),
                'n_liquids': int(sizes[eligible].sum()),
                'observed_spread': float(observed.mean()),
                'random_spread': float(expected.mean()),
                'ratio': float(observed.mean() / expected.mean()),
            })

    return DataFrame(rows)


def plot_clustering() -> list[Path]:
    """Every clustering diagnostic."""
    apply_style()
    FIG_CLUSTERING.mkdir(parents=True, exist_ok=True)
    written: list[Path] = [plot_metric_agreement(FIG_CLUSTERING / 'metric_agreement.png')]

    preservation = chemical_preservation()
    preservation.to_csv(CLUSTERING_DIR / 'chemical_preservation.csv', index=False)

    for subset in ('A', 'B'):
        out = FIG_CLUSTERING / subset
        out.mkdir(parents=True, exist_ok=True)
        written += [
            plot_embeddings(subset, out / 'embeddings.png'),
            plot_elbow(subset, out / 'silhouette_vs_k.png'),
            plot_cluster_sizes(subset, out / 'cluster_sizes_k5.png'),
            plot_chemical_agreement(subset, out / 'chemical_agreement.png'),
            plot_density_sweep(subset, out / 'density_sweep.png'),
            plot_density_noise(subset, out / 'density_noise.png'),
            plot_eps_curve(subset, out / 'eps_curve.png'),
        ]

    return written


def plot_everything() -> list[Path]:
    """EDA, screening and clustering figures in one call."""
    return plot_eda() + plot_screening() + plot_clustering()


if __name__ == '__main__':
    # This block used to sit above plot_screening()/plot_clustering(), so running
    # the module regenerated only the EDA figures and left every screening and
    # clustering figure -- plus chemical_preservation.csv -- stale.
    for written in plot_everything():
        print('  wrote', written)
