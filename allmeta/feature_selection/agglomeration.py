"""Reproducible feature-agglomeration sweep and diagnostic figures.

Run with ``PYTHONPATH=src python -m allmeta.feature_selection.agglomeration``.
Objects are descriptors, not liquids. Metrics are descriptive, in-sample diagnostics.
Descriptors are clustered on absolute Spearman correlation distance, 1 - |rho|, not on
their Euclidean profiles: two descriptors are close when one is a monotonic function of
the other, whatever their sign or scale. Ward assumes Euclidean geometry, so it is not
among the linkages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from allmeta.paths import FEATURE_SELECTION_DIR, FIG_FEATURE_SELECTION, RANDOM_STATE

GEOMETRY = 'absolute_spearman'
LINKAGES = ('average', 'complete', 'single', 'weighted')
COLORS = ('#2878A0', '#B88A22', '#C76B35', '#78863D', '#B9658A')
# Cumulative principal components reported per cluster: PC1, PC1-2, ..., PC1-N_PCS.
N_PCS = 5
PC_COLUMNS = ('pc1_explained_variance_ratio',) + tuple(
    f'top{m}_pcs_explained_variance_ratio' for m in range(2, N_PCS + 1))


def prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Reject ambiguous/nonfinite inputs and explicitly remove constant columns."""
    if not frame.columns.is_unique or not all(isinstance(c, str) for c in frame.columns):
        raise ValueError('feature names must be unique strings')
    values = frame.to_numpy(dtype=float)
    if len(frame) < 3 or not np.isfinite(values).all():
        raise ValueError('need at least three rows and finite, complete numeric data')
    keep = np.ptp(values, axis=0) > 0
    removed = frame.columns[~keep].tolist()
    if keep.sum() < 3:
        raise ValueError('need at least three nonconstant features')
    return pd.DataFrame(StandardScaler().fit_transform(values[:, keep]),
                        index=frame.index, columns=frame.columns[keep]), removed


def spearman_distance(scaled: pd.DataFrame) -> np.ndarray:
    """1 - |Spearman rho| between descriptors: 0 for any monotonic pair, 1 if unrelated.

    Absolute, so a descriptor and its sign-reversed twin sit at distance 0 -- the same
    sign-blindness as the corr^2 variance score, under which either one reconstructs
    the other exactly.
    """
    rho = scaled.corr(method='spearman').to_numpy()
    distance = np.clip(1 - np.abs(rho), 0, 1)
    np.fill_diagonal(distance, 0)
    return distance


def representative_variance(scaled: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Score each candidate by variance reconstructed in its whole cluster.

    For standardized columns, fitting every cluster member from candidate j by OLS
    with intercept explains mean_i corr(X_i, X_j)^2 of total cluster variance.
    The candidate itself is included. PC1 is the optimal unconstrained rank-one
    reconstruction, an upper bound; its loading is NOT a feature's explained variance.
    The top-m PC columns are the rank-m bounds, cumulative, for m up to N_PCS; they
    are NaN where the cluster has fewer than m components (m > its member count).

    The spearman_ columns repeat the score with Spearman rho in place of Pearson r:
    the share of the cluster's RANK variance a linear fit on the candidate's ranks
    recovers. A monotonic but curved member counts fully here and only partly above.
    The PC columns stay on the raw standardized values and do not bound this score.
    """
    rows = []
    for cluster in np.unique(labels):
        block = scaled.iloc[:, np.flatnonzero(labels == cluster)]
        x = block.to_numpy()
        norms = np.linalg.norm(x, axis=0)
        corr = (x.T @ x) / np.outer(norms, norms)
        scores = np.clip(np.mean(corr ** 2, axis=0), 0, 1)
        rho = block.corr(method='spearman').to_numpy()
        spearman_scores = np.clip(np.mean(rho ** 2, axis=0), 0, 1)
        singular = np.linalg.svd(x, compute_uv=False)
        cumulative = np.cumsum(singular ** 2) / np.sum(singular ** 2)
        pcs = {column: float(cumulative[m]) if m < min(x.shape) else np.nan
               for m, column in enumerate(PC_COLUMNS)}
        # Stable input-order tie break for duplicate/sign-reversed descriptors.
        winner, spearman_winner = (
            int(np.flatnonzero(np.isclose(v, v.max(), rtol=0, atol=1e-12))[0])
            for v in (scores, spearman_scores))
        for j, feature in enumerate(block.columns):
            rows.append(dict(cluster=int(cluster), feature=feature, cluster_size=x.shape[1],
                             explained_variance_ratio=float(scores[j]),
                             **pcs, is_representative=j == winner,
                             spearman_explained_variance_ratio=float(spearman_scores[j]),
                             is_spearman_representative=j == spearman_winner))
    return pd.DataFrame(rows)


def sweep(frame: pd.DataFrame, ks=range(2, 16), linkages=LINKAGES) -> dict:
    """Compare identical k cuts across linkages on 1 - |Spearman rho|; return audit tables."""
    scaled, removed = prepare(frame)
    ks = sorted(set(ks))
    if not ks or any(not isinstance(k, (int, np.integer)) or not 2 <= k < scaled.shape[1] for k in ks):
        raise ValueError('k must be an integer from 2 through n_features - 1')
    if not linkages or any(method not in LINKAGES for method in linkages):
        raise ValueError(f'linkages must be drawn from {LINKAGES}')
    distances = {GEOMETRY: spearman_distance(scaled)}
    trees, memberships, variances, scores = {}, [], [], []
    for geometry, distance in distances.items():
        for method in linkages:
            tree = linkage(squareform(distance, checks=True), method=method)
            trees[(geometry, method)] = tree
            for k in ks:
                labels = cut_tree(tree, n_clusters=[k]).ravel() + 1
                key = dict(geometry=geometry, linkage=method, k=k)
                detail = representative_variance(scaled, labels).assign(**key)
                variances.append(detail)
                winners = detail[detail.is_representative]
                spearman_winners = detail[detail.is_spearman_representative]
                scores.append(dict(**key, silhouette=silhouette_score(distance, labels, metric='precomputed'),
                                   representative_variance_ratio=np.average(winners.explained_variance_ratio,
                                                                            weights=winners.cluster_size),
                                   spearman_representative_variance_ratio=np.average(
                                       spearman_winners.spearman_explained_variance_ratio,
                                       weights=spearman_winners.cluster_size),
                                   pc1_variance_ratio=np.average(winners.pc1_explained_variance_ratio,
                                                                weights=winners.cluster_size),
                                   singleton_clusters=int((winners.cluster_size == 1).sum())))
                memberships.extend(dict(**key, feature=f, cluster=int(c))
                                   for f, c in zip(scaled.columns, labels))
    metrics = pd.DataFrame(scores)
    best = (metrics.sort_values(['silhouette', 'k'], ascending=[False, True], kind='stable')
            .groupby(['geometry', 'linkage'], sort=False).head(1).reset_index(drop=True))
    return dict(scaled=scaled, removed_constants=removed, distances=distances, trees=trees,
                metrics=metrics, selected=best, membership=pd.DataFrame(memberships),
                variance=pd.concat(variances, ignore_index=True))



def plot_cluster_dendrogram(tree, names, labels, geometry, method, context, selection_label, path,
                            cut=None, cut_text=None):
    """Color exact cluster memberships, including singleton leaves and tied cuts.

    ``cut``/``cut_text`` override the drawn cut line and its legend entry with an
    explicit height, e.g. a fixed correlation threshold rather than the bracket
    midpoint around a silhouette-selected k.
    """
    import colorsys
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    clusters = sorted(np.unique(labels))
    k, n_features = len(clusters), len(names)
    palette = ('#2878A0', '#C76B35', '#78863D', '#B9658A', '#7056A3',
               '#16877A', '#B88A22', '#B53D48', '#466BB0', '#875638',
               '#9C548F', '#526B39', '#337A8C', '#8A7428', '#655B75')
    def overflow_color(i):
        r, g, b = colorsys.hsv_to_rgb((i * .61803398875) % 1, .65, .65)
        return f'#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}'

    colors = {c: palette[i] if i < len(palette) else overflow_color(i) for i, c in enumerate(clusters)}
    # An internal node belongs to a cluster only if both children do. Bridges
    # between distinct clusters stay neutral, independent of merge-height ties.
    node_clusters = dict(enumerate(labels))
    for i, (left, right, _, _) in enumerate(tree):
        a, b = node_clusters[int(left)], node_clusters[int(right)]
        node_clusters[n_features + i] = a if a == b else None
    fig, ax = plt.subplots(figsize=(max(12, n_features * .28), 7), layout='constrained')
    drawing = dendrogram(tree, labels=names, ax=ax, leaf_rotation=90, leaf_font_size=8,
                         link_color_func=lambda node: colors.get(node_clusters[node], '#777777'))
    for position, (tick, leaf) in enumerate(zip(ax.get_xticklabels(), drawing['leaves'])):
        cluster = labels[leaf]
        tick.set_color(colors[cluster])
        tick.set_fontweight('bold')
        # A singleton has no within-cluster branch; mark every leaf explicitly.
        ax.scatter(5 + 10 * position, 0, color=colors[cluster], s=25, zorder=5, clip_on=False)
    handles = [Line2D([0], [0], color=colors[c], lw=3, label=f'Cluster {c}') for c in clusters]
    if cut is None:
        low, high = tree[n_features-k-1, 2], tree[n_features-k, 2]
        if high > low:
            cut = (low + high) / 2
            cut_text = f'k={k} cut at |ρ| = {1 - cut:.2f}'
    if cut is not None:
        ax.axhline(cut, color='#444444', linestyle='--', linewidth=1)
        handles.append(Line2D([0], [0], color='#444444', linestyle='--', label=cut_text))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.24),
              ncol=min(6, len(handles)), fontsize=8, frameon=False)
    ax.set_ylim(bottom=0)
    ax.set(title=f'Feature dendrogram — {method} linkage on 1 − |Spearman ρ|, '
                 f'{selection_label}\n{context}',
           ylabel='Merge distance, 1 − |Spearman ρ|\n(0 = perfectly monotonic, 1 = unrelated)')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_representatives(winners, method, k, n_rows, path):
    """Best single feature per cluster by Pearson (bar) against its top PC bounds (ticks).

    Ticks are cumulative, so the m-th is the most any m-dimensional linear summary of
    the cluster can retain; the gap from bar to PC1 tick is what choosing one real
    descriptor costs over the ideal single axis. One hue, dark to light, because the
    ticks are ordered. A cluster with fewer than m members has no m-th tick.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pc_colors = ('#4A2308', '#7A3C12', '#A9581E', '#CF7B35', '#E8A566')
    pc_labels = ['PC1'] + [f'PC1–{m}' for m in range(2, N_PCS + 1)]
    fig, ax = _representative_bars(winners, 'explained_variance_ratio', k)
    # Last-drawn on top: where two cumulative ticks coincide the lower rank shows.
    for column, color in reversed(list(zip(PC_COLUMNS, pc_colors))):
        values = winners[column].to_numpy() * 100
        present = np.isfinite(values)
        ax.scatter(values[present], np.arange(k)[present], marker='|', s=260,
                   linewidths=2.2, color=color, zorder=3)
    handles = [Line2D([0], [0], color=COLORS[0], lw=8, label='Best single feature')]
    handles += [Line2D([0], [0], color=c, marker='|', markersize=14, markeredgewidth=2.2,
                       linestyle='', label=label) for c, label in zip(pc_colors, pc_labels)]
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles),
               fontsize=8.5, frameon=False, columnspacing=1.4)
    ax.set(xlabel='Standardized within-cluster variance explained (%) — mean Pearson r²',
           title=f'Representative features by Pearson r² — {method} linkage on 1 − |Spearman ρ|, k={k}\n'
                 f'Bar: best single feature, itself included (singleton = 100%). '
                 f'Ticks: cumulative variance of the cluster\'s top {N_PCS} PCs\n'
                 f'(PC1–m = best any m-dimensional linear summary can retain) · {n_rows} liquids')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_spearman_representatives(winners, pearson_picks, method, k, n_rows, path):
    """Best single feature per cluster by Spearman rho^2; no PC ticks.

    The PCs are fitted on raw standardized values, the same space as the pipeline's
    cluster PC1, and are not a bound on a rank score -- a monotonic but curved cluster
    puts the Spearman bar past the PC1 tick. A PCA of the ranks would bound it, but no
    step in the pipeline builds one, so it would bound nothing anyone uses. Where the
    pick differs from the Pearson pick, the Pearson pick is named beside it.
    """
    import matplotlib.pyplot as plt

    fig, ax = _representative_bars(winners, 'spearman_explained_variance_ratio', k)
    ax.set_yticklabels([f'C{r.cluster}: {r.feature} (n={r.cluster_size})'
                        + ('' if r.feature == pearson_picks[r.cluster]
                           else f'\nPearson pick: {pearson_picks[r.cluster]}')
                        for r in winners.itertuples()])
    ax.set(xlabel='Within-cluster rank variance explained (%) — mean Spearman ρ²',
           title=f'Representative features by Spearman ρ² — {method} linkage on 1 − |Spearman ρ|, k={k}\n'
                 f'Bar: best single feature, itself included (singleton = 100%) — the share of the\n'
                 f'cluster\'s rank variance a linear fit on its ranks recovers · {n_rows} liquids')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _representative_bars(winners, column, k):
    """The horizontal bars and value labels both representative plots share."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, max(4, k * .45 + 2.6)), layout='constrained')
    y = np.arange(k)
    ax.barh(y, winners[column] * 100, color=COLORS[0], height=.72)
    for row, value in zip(y, winners[column]):
        ax.text(value * 100 - 1, row, f'{value:.1%}', va='center', ha='right',
                color='white', fontsize=9)
    ax.set(yticks=y, yticklabels=[f'C{r.cluster}: {r.feature} (n={r.cluster_size})'
                                  for r in winners.itertuples()], xlim=(0, 105))
    ax.grid(axis='x', alpha=.18)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    return fig, ax


def plot_cluster_umap(embedding, names, labels, winners, method, k, geometry, context, path):
    """One facet per cluster: identical embedding and axes in every panel."""
    import matplotlib.pyplot as plt

    cols = min(3, k)
    fig, axes = plt.subplots(int(np.ceil(k / cols)), cols, squeeze=False,
                             figsize=(6 * cols, 4 * int(np.ceil(k / cols))), layout='constrained')
    for c, ax in enumerate(axes.flat, 1):
        if c > k:
            ax.set_visible(False)
            continue
        mask = labels == c
        ax.scatter(*embedding.T, c='#CCCCCC', s=18)
        ax.scatter(*embedding[mask].T, c=COLORS[0], s=45, edgecolor='#333333')
        representative = winners.loc[winners.cluster == c, 'feature'].iloc[0]
        for j in [names.index(representative)]:
            ax.annotate(names[j], embedding[j], xytext=(4, 4), textcoords='offset points', fontsize=8)
        ax.margins(.22)
        ax.set(title=f'Cluster {c} · {mask.sum()} features', xlabel='UMAP 1', ylabel='UMAP 2')
    fig.suptitle(f'Feature UMAP — {method}, {geometry}, k={k}\nRepresentative labeled; shared embedding; clusters fitted before UMAP · {context}')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_results(result: dict, directory: Path) -> pd.DataFrame:
    """Static Matplotlib exports; shared UMAP per geometry, fitted before coloring.

    Chart contract: lines compare linkage quality versus k within geometry; annotated
    feature scatters show neighborhood structure; dendrograms show exact fitted trees;
    horizontal bars compare representative reconstruction against PC1. Colors identify
    linkage only; cluster scatters use numbered facets to avoid large categorical keys.
    """
    import matplotlib.pyplot as plt
    from umap import UMAP

    for sub in ('dendrograms', 'representatives', 'umap', 'performance'):
        (directory / sub).mkdir(parents=True, exist_ok=True)
    coordinates = []
    names = result['scaled'].columns.tolist()
    n_rows, n_features = result['scaled'].shape
    context = f'{n_features} descriptors across {n_rows} liquids; descriptive fit'
    for geometry, distance in result['distances'].items():
        subset = result['metrics'].query('geometry == @geometry')
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
        for i, (method, group) in enumerate(subset.groupby('linkage', sort=False)):
            for ax, metric, title in zip(axes,
                    ['silhouette', 'representative_variance_ratio', 'pc1_variance_ratio'],
                    ['Feature silhouette (higher is better)', 'Variance retained by single features',
                     'Variance retained by cluster PC1']):
                ax.plot(group.k, group[metric], label=method, color=COLORS[i],
                        marker=['o', 's', '^', 'D', 'v'][i], markersize=4)
                ax.set(xlabel='Number of feature clusters', ylabel=metric.replace('_', ' '), title=title)
                ax.grid(alpha=.18)
                if metric != 'silhouette':
                    ax.set_ylim(0, 1.03)
        axes[0].legend(fontsize=9)
        fig.suptitle(f'Feature agglomeration performance — 1 − |Spearman ρ| distance\n{context}')
        fig.savefig(directory / 'performance' / f'{geometry}_performance.png', dpi=160)
        plt.close(fig)
        embedding = UMAP(n_components=2, metric='precomputed', n_neighbors=min(15, n_features - 1),
                         min_dist=.15, random_state=RANDOM_STATE, n_jobs=1,
                         init='random').fit_transform(distance)
        coordinates.extend(dict(geometry=geometry, feature=f, umap_1=float(x), umap_2=float(y))
                           for f, (x, y) in zip(names, embedding))
        for selected in result['selected'].query('geometry == @geometry').itertuples():
            method, k = selected.linkage, selected.k
            stem = f'{geometry}_{method}_k{k}'
            detail = result['variance'].query('geometry == @geometry and linkage == @method and k == @k')
            winners = detail[detail.is_representative].sort_values('cluster')
            tree = result['trees'][(geometry, method)]
            labels = cut_tree(tree, n_clusters=[k]).ravel() + 1
            plot_cluster_dendrogram(tree, names, labels, geometry, method, context, f'selected k={k}',
                                    directory / 'dendrograms' / f'{stem}_dendrogram.png')
            plot_cluster_umap(embedding, names, labels, winners, method, k, geometry, context,
                              directory / 'umap' / f'{stem}_umap.png')
            plot_representatives(winners, method, k, n_rows,
                                 directory / 'representatives' / f'{stem}_representatives_pearson.png')
            spearman_winners = detail[detail.is_spearman_representative].sort_values('cluster')
            plot_spearman_representatives(spearman_winners, dict(zip(winners.cluster, winners.feature)),
                                          method, k, n_rows,
                                          directory / 'representatives' / f'{stem}_representatives_spearman.png')
    return pd.DataFrame(coordinates)


def plot_threshold_suite(result: dict, embedding: np.ndarray, names: list[str], context: str, n_rows: int,
                          figures_dir: Path, data_dir: Path, geometry: str = GEOMETRY, method: str = 'complete',
                          correlation_cutoff: float = 0.8) -> dict:
    """Full suite for a fixed |Spearman rho| cutoff, alongside the silhouette-selected k's.

    Reuses the tree already built by ``sweep`` for (geometry, method) and the shared UMAP
    embedding already fitted by ``plot_results`` -- no distance/tree/embedding refit.
    """
    tree = result['trees'][(geometry, method)]
    distance_threshold = 1 - correlation_cutoff
    labels = fcluster(tree, t=distance_threshold, criterion='distance')
    k = len(np.unique(labels))
    detail = representative_variance(result['scaled'], labels).assign(
        geometry=geometry, linkage=method, correlation_cutoff=correlation_cutoff)
    winners = detail[detail.is_representative].sort_values('cluster')
    spearman_winners = detail[detail.is_spearman_representative].sort_values('cluster')
    stem = f'{geometry}_{method}_cutoff{correlation_cutoff:.2f}'
    selection_label = f'cutoff |ρ| ≥ {correlation_cutoff:.2f}'
    cut_text = f'cutoff at |ρ| = {correlation_cutoff:.2f} (k={k})'
    for sub in ('dendrograms', 'representatives', 'umap'):
        (figures_dir / sub).mkdir(parents=True, exist_ok=True)
    plot_cluster_dendrogram(tree, names, labels, geometry, method, context, selection_label,
                            figures_dir / 'dendrograms' / f'{stem}_dendrogram.png',
                            cut=distance_threshold, cut_text=cut_text)
    plot_cluster_umap(embedding, names, labels, winners, method, k, geometry, context,
                      figures_dir / 'umap' / f'{stem}_umap.png')
    plot_representatives(winners, method, k, n_rows,
                         figures_dir / 'representatives' / f'{stem}_representatives_pearson.png')
    plot_spearman_representatives(spearman_winners, dict(zip(winners.cluster, winners.feature)),
                                  method, k, n_rows,
                                  figures_dir / 'representatives' / f'{stem}_representatives_spearman.png')
    cutoff_dir = data_dir / 'cutoff'
    cutoff_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(geometry=geometry, linkage=method, feature=f, cluster=int(c))
                 for f, c in zip(names, labels)]).to_csv(cutoff_dir / 'membership.csv', index=False)
    detail.to_csv(cutoff_dir / 'variance.csv', index=False)
    return dict(geometry=geometry, linkage=method, correlation_cutoff=correlation_cutoff,
                distance_threshold=distance_threshold, k=k,
                silhouette=float(silhouette_score(result['distances'][geometry], labels, metric='precomputed')),
                singleton_clusters=int((winners.cluster_size == 1).sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset', choices=['A', 'B'], default='B')
    parser.add_argument('--max-k', type=int, default=15)
    parser.add_argument('--output', type=Path, default=FEATURE_SELECTION_DIR / 'agglomeration')
    parser.add_argument('--figures', type=Path, default=FIG_FEATURE_SELECTION / 'agglomeration')
    parser.add_argument('--cutoff-correlation', type=float, default=0.8,
                        help='Fixed |Spearman rho| threshold for the extra threshold-cut dendrogram.')
    parser.add_argument('--cutoff-linkage', choices=LINKAGES, default='complete')
    args = parser.parse_args()
    from allmeta import columns
    from allmeta.feature_selection.pipeline import fitting_frame
    frame, note = fitting_frame(args.subset)
    inputs = frame[columns.screening_features()]
    scaled, _ = prepare(inputs)
    result = sweep(inputs, range(2, min(args.max_k, scaled.shape[1] - 1) + 1))
    coords = plot_results(result, args.figures)
    names = result['scaled'].columns.tolist()
    n_rows, n_features = result['scaled'].shape
    context = f'{n_features} descriptors across {n_rows} liquids; descriptive fit'
    embedding = (coords.query('geometry == @GEOMETRY').set_index('feature')
                 .loc[names, ['umap_1', 'umap_2']].to_numpy())
    cutoff_summary = plot_threshold_suite(result, embedding, names, context, n_rows, args.figures, args.output,
                                          method=args.cutoff_linkage, correlation_cutoff=args.cutoff_correlation)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('metrics', 'selected', 'membership', 'variance'):
        result[name].to_csv(args.output / f'{name}.csv', index=False)
    coords.to_csv(args.output / 'umap_coordinates.csv', index=False)
    trees_dir = args.output / 'trees'
    trees_dir.mkdir(parents=True, exist_ok=True)
    for (geometry, method), tree in result['trees'].items():
        pd.DataFrame(tree, columns=['left', 'right', 'distance', 'count']).to_csv(
            trees_dir / f'{geometry}_{method}_tree.csv', index=False)
    import importlib.metadata
    note.update(seed=RANDOM_STATE, feature_names=result['scaled'].columns.tolist(),
                removed_constants=result['removed_constants'],
                input_sha256=hashlib.sha256(inputs.to_csv(index=True).encode()).hexdigest(),
                versions={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'umap-learn']},
                selection='Maximum in-sample silhouette per geometry/linkage; smaller k breaks exact ties.',
                variance='Mean squared Pearson correlation, including candidate; standardized total variance. '
                         'spearman_* columns: mean squared Spearman rho, i.e. rank variance.',
                distance='1 - |Spearman rho| between descriptors; average/complete/single/weighted linkage (no Ward).',
                pcs=f'Cumulative within-cluster variance of the top 1..{N_PCS} PCs; NaN where a cluster has fewer members.',
                limitations='Descriptive fit; no predictive or held-out performance.',
                umap=dict(n_neighbors=min(15, scaled.shape[1]-1), min_dist=.15, init='random'),
                max_k=args.max_k, cutoff=cutoff_summary)
    (args.output / 'manifest.json').write_text(json.dumps(note, indent=2) + '\n')
    print(result['selected'].to_string(index=False))
    print(f"Threshold cut: {args.cutoff_linkage} linkage, |ρ| >= {args.cutoff_correlation:.2f} -> "
          f"k={cutoff_summary['k']}")
    print(f'\nTables: {args.output}\nFigures: {args.figures}')


if __name__ == '__main__':
    main()
