"""Repository paths, anchored on this file rather than the caller's cwd.

Notebooks live at different depths, so relative paths like ``../../data`` resolve
differently depending on which notebook evaluates them. One notebook already writes
its output to ``PycharmProjects/data/feature_selection`` — outside the repository —
because of exactly that. Everything here derives from ``__file__`` so a driver can
be run from any directory and still hit the same files.
"""

from pathlib import Path

# src/allmeta/paths.py -> src/allmeta -> src -> repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = REPO_ROOT / 'data'
FIG_DIR: Path = REPO_ROOT / 'figures'

# Inputs
#
# The workbook is the only source of the full descriptor table now. It used to be
# mirrored as data/cleaned_data/cleaned_HSPiP_full_liquids.csv purely to avoid a
# 5.5 MB Excel parse; that mirror is gone, so subsets.load_master() reads the
# 'HSPip master' sheet directly and memoises it.
HSPIP_XLSX: Path = DATA_DIR / 'raw_hspip' / 'HSPiP_data.xlsx'

# The curated liquid list: liquids verified as liquid at 25 C, with tier,
# liquid_confidence and verification_route provenance columns. This replaced
# data/HSPiP_removal_candidates.csv and inverts its polarity -- it names the
# liquids to KEEP rather than the ones to drop, so subset B is an intersection
# rather than a difference.
#
# A SELECTION MASK, NOT A DATA SOURCE. The file also carries a copy of the
# descriptors, and nothing in this package may read them: every feature value comes
# from HSPIP_XLSX. subsets.curated_names() is the only door this file comes through,
# and subsets.validate_rebuild() re-reads the workbook to prove it stayed that way.
#
# The sibling data/cleaned_data/HSPiP_removed_liquids.csv is a byte-identical
# duplicate of this file, not a remove-list. Nothing reads it; the removed set is
# derived as the complement of the keep list by subsets.subset_r().
CURATED_LIQUIDS_CSV: Path = DATA_DIR / 'cleaned_data' / 'HSPiP_cleaned_liquids.csv'

CHEM_CLUSTER_CSV: Path = DATA_DIR / 'eda' / 'HSPiP_cluster_ids.csv'
BENCHMARK_DIR: Path = DATA_DIR / 'ALL_data' / 'benchmarking'

# Outputs
SUBSET_DIR: Path = DATA_DIR / 'subsets'
FEATURE_SELECTION_DIR: Path = DATA_DIR / 'feature_selection'
CLUSTERING_DIR: Path = DATA_DIR / 'clustering'
COMPARISON_DIR: Path = DATA_DIR / 'comparison'

# Study outputs. data/ and figures/ share one subfolder name per study, so the tables
# behind a figure sit at the same relative path under data/ as the figure does under
# figures/.
DIST_CURVES_DATA_DIR: Path = DATA_DIR / 'distance_learning_curves'
FIG_DIST_CURVES: Path = FIG_DIR / 'distance_learning_curves'
PURCHASE_DATA_DIR: Path = DATA_DIR / 'liquid_purchase_candidates'
FIG_PURCHASE: Path = FIG_DIR / 'liquid_purchase_candidates'
PURCHASE_DOCS_DIR: Path = REPO_ROOT / 'docs' / 'liquid_purchase_candidates'

FIG_FEATURE_SELECTION: Path = FIG_DIR / 'feature_selection'
FIG_CLUSTERING: Path = FIG_DIR / 'clustering'
FIG_COMPARISON: Path = FIG_DIR / 'comparison'

# Every analysis in this package seeds from here so reruns are comparable.
RANDOM_STATE: int = 48


def ensure_output_dirs() -> None:
    """Create every output directory this package writes to."""
    for directory in (
        SUBSET_DIR, FEATURE_SELECTION_DIR, CLUSTERING_DIR, COMPARISON_DIR,
        FIG_FEATURE_SELECTION, FIG_CLUSTERING, FIG_COMPARISON,
    ):
        directory.mkdir(parents=True, exist_ok=True)
