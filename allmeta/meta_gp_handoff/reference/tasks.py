"""Phase 17 task construction (skeleton: signatures, shapes and OPEN decisions; no bodies yet).

For each target liquid and context size k: the source data (every other eligible liquid), the target's first k
trials in time order (context) and its remaining trials (query). No target-liquid row may reach the sources.

Facts checked on 2026-11-23 against data/campaign_ingest/conditions.csv and the raw Ax snapshots:
  - campaign values: optimization (1651 rows), benchmarking (172), verification (139), condition_stability (48)
  - optimization rows: 100 per liquid (pentane 51, water 200 from two runs), volume fixed at 10 uL
  - each optimization run starts with 1 centre point + 9 Sobol points (Ax Sobol, seed 42), then BoTorch
  - condition_id looks like '2026_08_01_acetonitrile::optimization::trial_10'; string order puts trial_10
    before trial_2, so trial order must come from the parsed trial number
  - allmeta.campaign_ingest.load()['conditions'] and allmeta.campaign_ingest.eligible_liquids(conditions) exist
"""
from dataclasses import dataclass

import torch
from pandas import DataFrame

# The inputs the optimization campaigns vary.
INPUT_COLUMNS = ("asp_rate", "disp_rate", "pre_airgap", "post_airgap", "volume_offset")

# OPEN: the true bounds and log_scale flags, from the Ax experiment's search space.
# Observed ranges in the optimization rows: rates 3-800, air gaps 0-10, volume_offset -4 to 4.
SEARCH_SPACE_BOUNDS: dict[str, tuple[float, float]] = {}

# OPEN: production minimized mean_abs_error (accuracy) and std_error (precision); mean_error is signed.
OUTCOMES = ("mean_abs_error", "std_error")


@dataclass
class Task:
    """One target liquid at one context size k."""
    target: str
    k: int
    sources: dict[str, tuple[torch.Tensor, torch.Tensor]]   # liquid -> (X (n_m, d), y (n_m,)); never the target
    X_context: torch.Tensor                                  # (k, d)
    y_context: torch.Tensor                                  # (k,)
    X_query: torch.Tensor                                    # (q, d)
    y_query: torch.Tensor                                    # (q,)
    # OPEN: per-row replicate variance (std_error**2 / n_replicates) if a fixed-noise variant is added later


def trial_number(condition_id: str) -> int:
    """'...::optimization::trial_10' -> 10. Sort on this, never on condition_id."""
    ...


def optimization_rows(conditions: DataFrame) -> DataFrame:
    """campaign == 'optimization' rows of the eligible liquids, in trial order within each run.
    OPEN: water has two runs (200 rows): concatenate in run-date order, or keep one run.
    OPEN: strata (primary / legacy_space / incomplete, per the Phase 17 README): where they are recorded, and the filter."""
    ...


def to_unit_cube(frame: DataFrame) -> torch.Tensor:
    """(n, d) float64 in [0, 1] using SEARCH_SPACE_BOUNDS (fixed, identical for every liquid), never the data's
    own min/max: sources and target must share coordinates. OPEN: log transform first for any log_scale input."""
    ...


def liquid_data(rows: DataFrame, outcome: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """liquid -> (X (n, d), y (n,)) for one outcome, rows in trial order."""
    ...


def make_tasks(conditions: DataFrame, *, outcome: str, context_sizes: list[int],
               targets: list[str] | None = None) -> list[Task]:
    """One Task per (target, k). Context = the target's first k trials; query = the rest.
    Sources = all trials of every other eligible liquid. targets=None means every eligible liquid.
    OPEN: query = all remaining trials, or only the BoTorch-phase trials (trial index >= 10)."""
    ...
