"""MRMR feature selection (feature_engine) and the diagnostics specific to it.

MRMR -- Minimum Redundancy, Maximum Relevance -- is the natural third comparison for
this project. The unsupervised screens in ``screening`` drop redundant descriptors
without ever consulting the target; the wrapper and embedded methods in
``supervised`` chase relevance to the target with no redundancy penalty. MRMR
optimises both at once, greedily: it takes the most relevant feature first, then
repeatedly adds whichever feature maximises relevance against its average redundancy
to everything already chosen.

feature_engine exposes five variants, which differ in how the two quantities are
estimated and combined:

===== ==================== ============ ==========
name  relevance            redundancy   scheme
===== ==================== ============ ==========
MID   mutual information   mutual info  difference
MIQ   mutual information   mutual info  ratio
FCD   F-statistic          correlation  difference
FCQ   F-statistic          correlation  ratio
RFCQ  random forest        correlation  ratio
===== ==================== ============ ==========

The ratio variants have a sharp failure mode that this dataset triggers: a
zero-variance feature has no correlation with anything, so relevance/redundancy
divides by approximately zero and the feature scores *maximally*. Several of the
atom counts are constant across the labelled liquids, and before the guard in
``supervised`` they dominated the MRMR selections. See ``constant_feature_effect``.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from feature_engine.selection import MRMR
from pandas import DataFrame, Series
from sklearn.feature_selection import VarianceThreshold, f_regression, mutual_info_regression
from sklearn.preprocessing import RobustScaler

from allmeta import columns as cols
from allmeta.paths import FEATURE_SELECTION_DIR, FIG_DIR, RANDOM_STATE
from allmeta.plotting import annotate_empty, apply_style, save
from allmeta.subsets import labelled_liquid_count
from allmeta.supervised import MRMR_METHODS, N_SELECT, build_labeled_table

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

MRMR_DIR: Path = FEATURE_SELECTION_DIR / 'C' / 'mrmr'
FIG_MRMR: Path = FIG_DIR / 'mrmr'

# How far to push max_features when tracing the greedy selection order.
PATH_LENGTH: int = 15


def prepared_table(include_params: bool = False) -> tuple[DataFrame, Series, Series]:
    """Subset C with constant columns removed and features robust-scaled.

    Mirrors what the fold pipeline in ``supervised`` does, so the diagnostics here
    describe the same input the selectors actually saw. Fitted on all the liquids
    because these are descriptive summaries, not held-out estimates.
    """
    data, target, groups = build_labeled_table(include_params=include_params)

    guard = VarianceThreshold(0.0).set_output(transform='pandas')
    reduced = guard.fit_transform(data)
    scaled = RobustScaler().set_output(transform='pandas').fit_transform(reduced)

    return scaled, target, groups


def constant_feature_effect() -> DataFrame:
    """Which descriptors carry no information across the labelled liquids.

    ``n_distinct`` is the diagnostic that matters at this sample size: a descriptor
    taking two values over so few liquids is an indicator for one or two of them, not
    a chemical trend, however unanimously it is selected.
    """
    data, _, groups = build_labeled_table(include_params=False)
    per_liquid = data.groupby(groups.to_numpy()).first()

    rows = [{
        'feature': feature,
        'n_distinct': int(per_liquid[feature].nunique()),
        'variance': float(per_liquid[feature].var()),
        'constant': bool(per_liquid[feature].nunique() <= 1),
        'is_atom_count': feature in cols.COMP_COLUMNS,
    } for feature in data.columns]

    return DataFrame(rows).sort_values(['n_distinct', 'feature'], ignore_index=True)


def relevance_redundancy(include_params: bool = False) -> DataFrame:
    """Per-feature relevance to the target and redundancy with the other features.

    The two axes MRMR trades off, computed directly rather than read out of the
    fitted object, so both are on record for every feature and not just the selected
    ones. Relevance is reported two ways because MID/MIQ use mutual information
    while FCD/FCQ use the F-statistic, and on this data they rank features very
    differently.
    """
    scaled, target, _ = prepared_table(include_params=include_params)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mutual = mutual_info_regression(scaled, target, n_neighbors=3,
                                        random_state=RANDOM_STATE)
        f_stat, _ = f_regression(scaled, target)

    correlation = scaled.corr(method='pearson').abs()
    np.fill_diagonal(correlation.values, np.nan)

    return DataFrame({
        'feature': scaled.columns,
        'relevance_mi': mutual,
        'relevance_f': np.nan_to_num(f_stat, nan=0.0),
        'redundancy_corr': correlation.mean().to_numpy(),
    }).sort_values('relevance_mi', ascending=False, ignore_index=True)


def selection_path(method: str, max_features: int = PATH_LENGTH,
                   include_params: bool = False) -> DataFrame:
    """Order in which MRMR admits features as the budget grows.

    MRMR is greedy, so refitting with an increasing ``max_features`` recovers the
    ranking it would produce. A feature entering late is one MRMR considered
    redundant with what it had already taken, which is exactly the information a
    flat frequency table throws away.
    """
    scaled, target, _ = prepared_table(include_params=include_params)
    budget = min(max_features, scaled.shape[1])

    seen: list[str] = []
    rows: list[dict] = []
    for k in range(1, budget + 1):
        options: dict = {'method': method, 'max_features': k, 'regression': True,
                         'random_state': RANDOM_STATE}
        if method == 'RFCQ':
            options.update(scoring='neg_root_mean_squared_error', cv=3)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            selector = MRMR(**options).fit(scaled, target)

        chosen = list(selector.get_feature_names_out())
        for feature in chosen:
            if feature not in seen:
                seen.append(feature)
                rows.append({'method': method, 'rank': len(seen), 'feature': feature,
                             'entered_at_k': k})

    return DataFrame(rows)


def variant_agreement(max_features: int = N_SELECT,
                      include_params: bool = False) -> tuple[DataFrame, DataFrame]:
    """What each MRMR variant selects, and how much the five agree.

    Returns the per-variant selections and their pairwise Jaccard matrix.
    """
    scaled, target, _ = prepared_table(include_params=include_params)

    selections: dict[str, list[str]] = {}
    for method in MRMR_METHODS:
        options: dict = {'method': method, 'max_features': max_features,
                         'regression': True, 'random_state': RANDOM_STATE}
        if method == 'RFCQ':
            options.update(scoring='neg_root_mean_squared_error', cv=3)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            selector = MRMR(**options).fit(scaled, target)
        selections[method] = list(selector.get_feature_names_out())

    long = DataFrame([{'method': m, 'feature': f}
                      for m, features in selections.items() for f in features])

    names = list(selections)
    values = np.eye(len(names))
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if j <= i:
                continue
            a, b = set(selections[left]), set(selections[right])
            values[i, j] = values[j, i] = len(a & b) / len(a | b) if (a | b) else np.nan

    return long, DataFrame(values, index=names, columns=names)


def run(max_features: int = N_SELECT) -> dict[str, DataFrame]:
    """Compute every MRMR artefact and write it."""
    MRMR_DIR.mkdir(parents=True, exist_ok=True)

    cardinality = constant_feature_effect()
    tradeoff = relevance_redundancy()
    selections, jaccard = variant_agreement(max_features=max_features)
    paths = pd.concat([selection_path(m) for m in MRMR_METHODS], ignore_index=True)

    cardinality.to_csv(MRMR_DIR / 'feature_cardinality.csv', index=False)
    tradeoff.to_csv(MRMR_DIR / 'relevance_redundancy.csv', index=False)
    selections.to_csv(MRMR_DIR / 'variant_selections.csv', index=False)
    jaccard.to_csv(MRMR_DIR / 'variant_jaccard.csv')
    paths.to_csv(MRMR_DIR / 'selection_path.csv', index=False)

    return {'cardinality': cardinality, 'relevance_redundancy': tradeoff,
            'selections': selections, 'jaccard': jaccard, 'paths': paths}


def plot_tradeoff(path: Path, max_features: int = N_SELECT) -> Path:
    """The plane MRMR optimises in: relevance up, redundancy left.

    Two panels because MID/MIQ score relevance by mutual information and FCD/FCQ by
    the F-statistic, and the two disagree about which descriptors are relevant --
    which is why the variants select such different sets.
    """
    tradeoff = relevance_redundancy()
    selections, _ = variant_agreement(max_features=max_features)
    chosen = {m: set(g['feature']) for m, g in selections.groupby('method')}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    panels = [('relevance_mi', 'mutual information', ('MID', 'MIQ')),
              ('relevance_f', 'F-statistic', ('FCD', 'FCQ'))]

    for ax, (column, label, methods) in zip(axes, panels):
        ax.scatter(tradeoff['redundancy_corr'], tradeoff[column], s=26, c='0.72',
                   label='not selected')
        # The two variants in each panel overlap heavily where they agree, so the
        # first is drawn as a wide open ring and the second as a filled marker
        # inside it. A feature both chose shows as a ring with a dot in it.
        first, second = methods
        picked = tradeoff[tradeoff['feature'].isin(chosen.get(first, set()))]
        ax.scatter(picked['redundancy_corr'], picked[column], s=320,
                   facecolors='none', edgecolors='#4c72b0', linewidth=2.2,
                   label=first)
        picked = tradeoff[tradeoff['feature'].isin(chosen.get(second, set()))]
        ax.scatter(picked['redundancy_corr'], picked[column], s=90,
                   color='#c44e52', edgecolor='k', linewidth=0.5, label=second,
                   alpha=0.9)

        top = tradeoff.nlargest(9, column)
        for row in top.itertuples():
            ax.annotate(row.feature, (row.redundancy_corr, getattr(row, column)),
                        fontsize=8, xytext=(5, 4), textcoords='offset points')

        ax.set_xlabel('redundancy: mean |correlation| with the other descriptors')
        ax.set_ylabel(f'relevance to dispensing error ({label})')
        ax.set_title(f'{label} relevance\nMRMR prefers the upper-left')
        ax.legend(fontsize=9)

    fig.suptitle('The relevance / redundancy trade-off MRMR optimises '
                 f'(subset C, {labelled_liquid_count()} liquids)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    return save(fig, path)


def plot_selection_path(path: Path) -> Path:
    """Order in which each variant admits features.

    Rank 1 is the most relevant descriptor outright; later entries are the ones MRMR
    judged to add information beyond what it already held. Blank cells mean the
    variant never selected that descriptor within the traced budget.
    """
    paths = pd.concat([selection_path(m) for m in MRMR_METHODS], ignore_index=True)
    grid = paths.pivot_table(index='feature', columns='method', values='rank')
    grid = grid.loc[grid.min(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(9, max(7, len(grid) * 0.3)))
    sns.heatmap(grid, annot=True, fmt='.0f', cmap='viridis_r', linewidths=0.4,
                linecolor='white', cbar_kws={'label': 'selection rank (1 = first)'},
                ax=ax, annot_kws={'size': 8})
    ax.set_title('MRMR selection order by variant\n(blank = never selected within '
                 f'the first {PATH_LENGTH})')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=8, labelrotation=0)

    fig.tight_layout()

    return save(fig, path)


def plot_variant_agreement(path: Path, max_features: int = N_SELECT) -> Path:
    """How much the five MRMR variants agree with each other.

    Low off-diagonal values mean "use MRMR" is not a decision -- the estimator
    choice inside it decides the answer.
    """
    _, jaccard = variant_agreement(max_features=max_features)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(jaccard, vmin=0, vmax=1, cmap='viridis', square=True, annot=True,
                fmt='.2f', linewidths=0.4, linecolor='white',
                cbar_kws={'label': 'Jaccard similarity'}, ax=ax)
    ax.set_title(f'Agreement between MRMR variants\n(top {max_features} features, '
                 f'descriptors only)')

    fig.tight_layout()

    return save(fig, path)


def plot_cardinality(path: Path) -> Path:
    """How many distinct values each descriptor takes across the labelled liquids.

    The guard rail for reading any selection on subset C. A descriptor with two
    distinct values is an indicator for one or two liquids; selecting it unanimously
    says more about the sample size than about the chemistry.
    """
    cardinality = constant_feature_effect()
    counts = cardinality['n_distinct'].value_counts().sort_index()
    n_liquids = labelled_liquid_count()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={'width_ratios': [1, 1.3]})

    colours = ['#c44e52' if k <= 1 else '#dd8452' if k < 4 else '#4c72b0'
               for k in counts.index]
    axes[0].bar(counts.index.astype(str), counts.values, color=colours)
    for x, y in zip(counts.index.astype(str), counts.values):
        axes[0].text(x, y + 0.4, str(y), ha='center', fontsize=9)
    axes[0].set_xlabel(f'distinct values across the {n_liquids} labelled liquids')
    axes[0].set_ylabel('descriptors')
    axes[0].set_title('Red = no information; orange = indicator for 1-2 liquids')

    flagged = cardinality[cardinality['n_distinct'] < 4].copy()
    flagged['kind'] = np.where(flagged['constant'], 'constant (0 information)',
                               'indicator (2-3 values)')
    sns.barplot(data=flagged.sort_values('n_distinct'), y='feature', x='n_distinct',
                hue='kind', dodge=False, ax=axes[1],
                palette={'constant (0 information)': '#c44e52',
                         'indicator (2-3 values)': '#dd8452'})
    axes[1].set_xlabel(f'distinct values across the {n_liquids} liquids')
    axes[1].set_ylabel('')
    axes[1].set_title(f'Descriptors that cannot support a trend at n={n_liquids}')
    axes[1].legend(fontsize=9, loc='lower right')

    fig.suptitle('Why MRMR needed a constant-feature guard: a zero-variance column '
                 'has zero redundancy,\nso the ratio variants (MIQ, FCQ, RFCQ) '
                 'score it maximally', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    return save(fig, path)


def plot_all() -> list[Path]:
    """Write every MRMR figure."""
    apply_style()
    FIG_MRMR.mkdir(parents=True, exist_ok=True)

    return [
        plot_cardinality(FIG_MRMR / 'feature_cardinality.png'),
        plot_tradeoff(FIG_MRMR / 'relevance_redundancy.png'),
        plot_selection_path(FIG_MRMR / 'selection_path.png'),
        plot_variant_agreement(FIG_MRMR / 'variant_agreement.png'),
    ]


if __name__ == '__main__':
    artefacts = run()
    card = artefacts['cardinality']
    print(f"{int(card['constant'].sum())} descriptors are constant across the "
          f"{labelled_liquid_count()} labelled liquids:")
    print('  ' + ', '.join(card.loc[card['constant'], 'feature']))
    print()
    print('Descriptors with fewer than 4 distinct values (indicators, not trends):')
    low = card[(~card['constant']) & (card['n_distinct'] < 4)]
    print('  ' + ', '.join(f"{r.feature}({r.n_distinct})" for r in low.itertuples()))
    print()
    print('MRMR variant agreement (Jaccard):')
    print(artefacts['jaccard'].round(2).to_string())
    print()
    # plot_all() used to be defined below this block, so running the module wrote
    # the CSVs but never refreshed a single figure.
    for written in plot_all():
        print('  wrote', written)
