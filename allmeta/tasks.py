"""Task construction for Phases 16 and 17: per-liquid trial statistics in trial order, and the source / context /
query split for one target liquid.

Decisions (defaults here; override from the sweep config):
  inputs      asp_rate, disp_rate, pre_airgap, post_airgap, volume_offset, mapped to [0, 1] with fixed BOUNDS
              shared by every liquid (sources and target must share coordinates). BOUNDS are the pipette's
              physical limits (confirmed 2026-11-24); inputs are on a linear scale.
  order       trials sorted by (run date, parsed trial number), never by condition_id (trial_10 sorts before trial_2)
  water       two optimization runs: both are source data; as a target only the first run is used (a new liquid
              is seen once), so its query set matches every other liquid's
  context     the target's first k trials in trial order (k = 0 allowed: transfer models only)
  query       the target's trials with index >= query_start (default 40 = largest k), fixed across k so every k
              is scored on the same trials; pentane (51 trials) gets 11
  outcomes    "signed"   trial mean of the signed error e; noise var / count             (TrialStats.mean_and_noise)
              "log_sd"   bias-corrected log replicate SD; noise trigamma(nu / 2) / 4   (TrialStats.log_sd)
              "log_abs"  log trial mean |e|, delta method                              (TrialStats.log_mean)
              E|e| on the raw scale is derived later from "signed" + "log_sd" (folded.py) or exp("log_abs").
  descriptors allmeta/data/descriptors.csv (the "physics" set), standardized on the sources only
  liquids     factorial_liquids(): every liquid in allmeta/data with a full descriptor row (15; the glycerol-water
              mixture has no HSPiP row). The data are the complete, uncapped, first optimization run of each liquid
              (allmeta/data/build_data.py, runs.csv), so every liquid has exactly one run.
No target trial reaches the sources; check_tasks() asserts it."""
import re
from dataclasses import dataclass, field

import pandas as pd
import torch

from pathlib import Path

from .trial_stats import TrialStats, from_replicates

INPUT_COLUMNS = ("asp_rate", "disp_rate", "pre_airgap", "post_airgap", "volume_offset")
BOUNDS = {"asp_rate": (3.0, 800.0), "disp_rate": (3.0, 800.0), "pre_airgap": (0.0, 10.0),
          "post_airgap": (0.0, 10.0), "volume_offset": (-4.0, 4.0)}   # physical pipette limits
LOG_SCALE = ()                                                          # e.g. ("asp_rate", "disp_rate") to log-map them
OUTCOMES = ("signed", "log_sd", "log_abs")
CONTEXT_SIZES = (0, 3, 5, 10, 20, 40)
DATA_DIR = Path(__file__).resolve().parent / "data"      # inside the package, so it travels with the code


def trial_number(condition_id: str) -> int:
    """'2026_08_01_acetonitrile::optimization::trial_10' -> 10."""
    return int(re.search(r"trial_(\d+)$", condition_id).group(1))


def to_unit_cube(X: torch.Tensor, bounds=None, log_scale=None) -> torch.Tensor:
    """(n, d) raw inputs in INPUT_COLUMNS order -> [0, 1]^d with fixed bounds (optional log10 first)."""
    bounds, log_scale = bounds or BOUNDS, LOG_SCALE if log_scale is None else log_scale
    out = []
    for j, name in enumerate(INPUT_COLUMNS):
        lo, hi, x = *bounds[name], X[:, j]
        if name in log_scale:
            lo, hi, x = map(torch.log10, (torch.tensor(lo), torch.tensor(hi), x))
        out.append((x - lo) / (hi - lo))
    return torch.stack(out, -1)


@dataclass
class LiquidTrials:
    """One liquid's optimization trials in trial order. run: 0 for the first run, 1 for water's second."""
    name: str
    signed: TrialStats
    absolute: TrialStats
    run: torch.Tensor

    def view(self, outcome):
        """(X, y, yvar, stats) for one outcome; stats is the TrialStats the noise models use."""
        stats = self.absolute if outcome == "log_abs" else self.signed
        y, yvar = {"signed": stats.mean_and_noise, "log_sd": stats.log_sd, "log_abs": stats.log_mean}[outcome]()
        return stats.X, y, yvar, stats

    def first_run(self):
        rows = torch.where(self.run == 0)[0]
        return LiquidTrials(self.name, self.signed.subset(rows), self.absolute.subset(rows), self.run[rows])


def factorial_liquids() -> list[str]:
    """Liquids with trials and a complete descriptor row: the targets and sources of every task."""
    D = pd.read_csv(DATA_DIR / "descriptors.csv")
    with_trials = set(pd.read_csv(DATA_DIR / "conditions.csv", usecols=["liquid"]).liquid)
    return sorted(set(D.dropna().liquid) & with_trials)


def load_liquids(transfers=None, liquids=None) -> dict[str, LiquidTrials]:
    """name -> LiquidTrials from allmeta/data/transfers.csv (optimization campaign only).
    liquids None -> factorial_liquids()."""
    T = pd.read_csv(DATA_DIR / "transfers.csv") if transfers is None else transfers
    T = T[T.campaign == "optimization"]
    out = {}
    for name in factorial_liquids() if liquids is None else liquids:
        rows = T[T.liquid == name]
        stats = {value: from_replicates(rows, INPUT_COLUMNS, value=value) for value in ("signed", "abs")}
        run_of = rows.groupby("condition_id", sort=False)["run_id"].first()
        runs = sorted(run_of.unique())
        key = [(runs.index(run_of[c]), trial_number(c)) for c in stats["signed"].ids]
        order = sorted(range(len(key)), key=key.__getitem__)
        assert stats["abs"].ids == stats["signed"].ids
        fixed = {}
        for value, s in stats.items():
            s = s.subset(order)
            fixed[value] = TrialStats(to_unit_cube(s.X.double()), s.mean.double(), s.var.double(),
                                      s.count.double(), s.ids)
        out[name] = LiquidTrials(name, fixed["signed"], fixed["abs"],
                                 torch.tensor([key[i][0] for i in order]))
    return out


def load_descriptors(path=None) -> dict[str, torch.Tensor]:
    """liquid -> raw descriptor vector, every column of descriptors.csv except the names."""
    D = pd.read_csv(DATA_DIR / "descriptors.csv" if path is None else path).dropna()
    cols = [c for c in D.columns if c not in ("liquid", "hspip_name")]
    return {row.liquid: torch.tensor(row[cols].to_numpy(dtype=float)) for _, row in D.iterrows()}


@dataclass
class Task:
    """One target liquid and outcome. Context for k = the first k target rows; query = rows >= query_start."""
    target: str
    outcome: str
    sources: dict          # liquid -> (X, y)
    source_stats: dict     # liquid -> TrialStats (noise models)
    z_sources: dict        # liquid -> standardized descriptors
    z_target: torch.Tensor
    X: torch.Tensor        # target, all rows in trial order
    y: torch.Tensor
    yvar: torch.Tensor
    stats: TrialStats
    query_start: int = 40
    meta: dict = field(default_factory=dict)

    def context(self, k):
        return self.X[:k], self.y[:k], self.stats.subset(torch.arange(k))

    @property
    def query(self):
        rows = torch.arange(self.query_start, len(self.y))
        return rows, self.X[rows], self.y[rows], self.stats.count[rows]


def make_task(liquids, descriptors, target, outcome, *, exclude=(), query_start=40) -> Task:
    """Sources = every liquid except the target and `exclude` (nested tuning removes the outer target too)."""
    names = [n for n in liquids if n != target and n not in exclude]
    sources, source_stats = {}, {}
    for n in names:
        X, y, _, stats = liquids[n].view(outcome)
        sources[n], source_stats[n] = (X, y), stats
    Z = torch.stack([descriptors[n] for n in names])
    shift, scale = Z.mean(0), Z.std(0).clamp_min(1e-12)
    X, y, yvar, stats = liquids[target].first_run().view(outcome)
    task = Task(target, outcome, sources, source_stats, {n: (descriptors[n] - shift) / scale for n in names},
                (descriptors[target] - shift) / scale, X, y, yvar, stats, query_start)
    check_task(task)
    return task


def check_task(task: Task):
    """Leakage guard: no target trial id appears in any source."""
    target_ids = set(task.stats.ids)
    for n, s in task.source_stats.items():
        assert n != task.target and not target_ids & set(s.ids), f"target rows leaked into source {n}"
    assert torch.isfinite(task.y).all(), f"{task.target}/{task.outcome}: non-finite y (zero-variance trial?)"
