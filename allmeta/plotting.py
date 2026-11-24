"""Shared matplotlib setup and small helpers used by every plotting module.

Kept separate so the analysis modules do not each re-declare styling, and so the
Agg backend is selected exactly once before any pyplot import.
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# Diverging for correlations (zero is meaningful), sequential for magnitudes.
DIVERGING: str = 'coolwarm'
SEQUENTIAL: str = 'viridis'

SUBSET_COLOURS: dict[str, str] = {'A': '#4c72b0', 'B': '#dd8452', 'C': '#55a868'}


def apply_style() -> None:
    """House style for every figure this package writes."""
    sns.set_theme(style='whitegrid', context='notebook')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 200,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
    })


def save(fig, path: Path) -> Path:
    """Write a figure and close it.

    Closing matters: these modules produce dozens of figures per run and matplotlib
    keeps every unclosed one alive.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return path


def annotate_empty(ax, message: str) -> None:
    """Put an explanatory message on an axis that has nothing to show.

    An empty panel reads as a bug; an empty panel saying why reads as a result.
    """
    ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes,
            fontsize=10, color='0.35', wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
