"""Build the transfer-level and condition-level tables from the raw campaigns.

The raw data is four folders of per-run JSON with three different shapes:

* ``benchmarking``/``verification``/``condition_stability`` -- ``results`` is keyed
  by the condition name and each entry carries its own ``params`` block.
* ``optimization`` -- ``results`` is keyed by the Ax/Atlas trial index and carries
  only the raw weights. The dispensing parameters for that trial live in a sibling
  ``results/<run>_results.csv``, which is joined back on ``trial_index`` where the
  file has one and on row order where it does not (verified elementwise against the
  recomputed error in tests).

Every run's error is recomputed from the raw gravimetry rather than read from the
stored ``errors`` field, because the series changed its error definition twice --
grams in the two July runs, millilitres in 2026_06_30_water, a dimensionless
fraction thereafter. One definition, applied everywhere:

    e = (calc_weight / density - V_target) / V_target

which is dimensionless and signed. ``allmeta.columns.TARGET_UNIT_WARNING`` records
the same trap for the benchmarking-only ingest.

Moved here on 2026-11-23 from ``allmeta.neural_meta.ingest`` (now in ``deprecated/``) so the
response-surface build keeps a live loader. The code is unchanged; the settings it reads
from ``neural_meta.config`` are inlined below, and the tables it writes now go to
``data/campaign_ingest/`` instead of ``data/neural_meta/``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from allmeta import columns as cols
from allmeta.paths import DATA_DIR, REPO_ROOT

# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------

# The raw campaign folders sit outside this repository, in the sibling
# autonomous-liquid-learning checkout. Overridable so the ingest can be pointed at
# a copy, matching ingest_ALL_data.py's ALL_DATA_DIR convention.
ALL_DATA_DEFAULT: Path = REPO_ROOT.parent / 'autonomous-liquid-learning' / 'data'

OUTPUT_DIR: Path = DATA_DIR / 'campaign_ingest'
TRANSFER_TABLE: Path = OUTPUT_DIR / 'transfers.csv'
CONDITION_TABLE: Path = OUTPUT_DIR / 'conditions.csv'
DESCRIPTOR_TABLE: Path = OUTPUT_DIR / 'descriptors.csv'
INGEST_MANIFEST: Path = OUTPUT_DIR / 'ingest_manifest.csv'

# ---------------------------------------------------------------------------
# Ingest rules
# ---------------------------------------------------------------------------

# Campaign families, in the order the experimental hierarchy runs them.
CAMPAIGNS: tuple[str, ...] = (
    'benchmarking', 'optimization', 'condition_stability', 'verification',
)

# Dry runs, not liquids.
EXCLUDED_LIQUIDS: frozenset[str] = frozenset({'test'})

# A condition with fewer replicates than this is dropped. The campaigns that
# aborted part-way through leave conditions with 0-5 transfers, and a replicate
# standard deviation from three points is already thin; below three it is noise.
MIN_REPLICATES: int = 3

# A benchmarking or verification run is only kept if EVERY one of its conditions
# clears this. A run that died half way leaves a biased subset of volumes -- the
# ones the randomised replicate order happened to reach first -- so it is dropped
# whole rather than trimmed.
MIN_REPLICATES_FULL_RUN: int = 20

# Relative errors beyond this magnitude are gravimetric failures (a missed
# aspiration, a tip that never picked up), not dispensing error. 3.0 = 300% of the
# target volume; the largest genuine mean error in the series is about 60%.
MAX_ABS_RELATIVE_ERROR: float = 3.0

# The dispensing conditions, shared by all four campaign families. 'volume_offset'
# is the tunable over-aspiration the optimiser was allowed to search; it is 0 in
# the benchmarking protocol and in the runs that predate the parameter.
# 'pipette_volume' is hardware, 50 or 1000 uL, and is CONFOUNDED WITH CAMPAIGN:
# benchmarking ran on the 1000 uL head, everything else on the 50 uL head. It is
# kept because ignoring it would make the model attribute a hardware difference to
# the liquid; see deprecated/docs/neural_meta_comparison.md.
CONDITION_FEATURES: tuple[str, ...] = (
    'volume', 'asp_rate', 'disp_rate', 'pre_airgap', 'post_airgap',
    'volume_offset', 'pipette_volume',
)

# Liquid descriptors. PHYSICS_KEEP is the mechanism-chosen set from the
# feature-selection work -- one volatility term, one bulk-transport term, one
# surface term, one viscosity term -- and is the controlled set the neural-process
# comparison used. Four descriptors for nine liquids is already generous; see
# allmeta.columns.PHYSICS_KEEP for the reasoning.
DESCRIPTOR_SET: str = 'physics'

# Per-replicate environmental channels. Recorded in the tables but NOT model
# inputs: they are campaign-level nuisance that tracks the calendar date, and the
# calendar date identifies the liquid, so feeding them to a cross-liquid model
# leaks the task label.
ENV_COLUMNS: tuple[str, ...] = ('temperature', 'pressure', 'humidity')

# The optimisation results CSV names its parameters exactly as the condition JSON
# does, so no mapping table is needed -- but two early runs order the columns
# differently ('post_airgap' before 'pre_airgap'), so they must be read by name.
OPT_PARAM_COLUMNS: tuple[str, ...] = (
    'asp_rate', 'disp_rate', 'pre_airgap', 'post_airgap', 'volume_offset',
)


@dataclass
class RunRecord:
    """One raw run, accepted or rejected, for the ingest audit trail."""

    campaign: str
    run_id: str
    path: str
    liquid: str
    n_conditions: int = 0
    n_transfers: int = 0
    min_replicates: int | None = None
    status: str = 'rejected'
    reason: str = ''

    def as_row(self) -> dict:
        return {
            'campaign': self.campaign, 'run_id': self.run_id, 'liquid': self.liquid,
            'n_conditions': self.n_conditions, 'n_transfers': self.n_transfers,
            'min_replicates': self.min_replicates, 'status': self.status,
            'reason': self.reason, 'path': self.path,
        }


@dataclass
class Ingest:
    """Accumulates transfer rows and the run-level audit trail."""

    transfers: list[dict] = field(default_factory=list)
    runs: list[RunRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Error recomputation
# ---------------------------------------------------------------------------

def relative_errors(raw: dict, volume: float | None) -> np.ndarray:
    """Signed relative dispensing error per replicate, dimensionless.

    ``calc_weight`` is a mass in grams and ``density`` g/mL, so their ratio is a
    volume in millilitres; the target is converted from microlitres to match.
    Replicates with a missing or zero density are dropped rather than divided by,
    which is why the returned array can be shorter than the replicate list.
    """
    weights = raw.get('calc_weight') or []
    densities = raw.get('density') or []
    n = min(len(weights), len(densities))
    if not n or not volume:
        return np.zeros(0)

    target_ml = float(volume) / 1000.0
    out: list[float] = []
    for index in range(n):
        density = densities[index]
        weight = weights[index]
        if not density or weight is None or not math.isfinite(float(weight)):
            continue
        out.append((float(weight) / float(density) - target_ml) / target_ml)

    return np.asarray(out, dtype=float)


def _finite_mean(values) -> float | None:
    numbers = [float(v) for v in (values or []) if v is not None and math.isfinite(float(v))]

    return sum(numbers) / len(numbers) if numbers else None


# ---------------------------------------------------------------------------
# Condition-shaped campaigns: benchmarking, verification, condition_stability
# ---------------------------------------------------------------------------

def _condition_rows(data: dict, campaign: str, run_id: str,
                    liquid: str) -> list[list[dict]] | None:
    """Transfer rows grouped by condition, for a params-carrying campaign.

    Returns None for a legacy-protocol run. Runs before 2026-07 store no per-result
    ``params`` block; their conditions are described only by the run's
    ``experiment_params``, and those describe a DIFFERENT protocol -- 100/500 uL/s
    rates with 12 uL airgaps, plus ``touch_tip``, ``blow_out`` and ``push_out``
    settings that no later run records. Those three tip-handling flags change the
    dispensed volume and cannot be represented in a feature space the later runs
    share, so the affected runs are dropped whole rather than joined in as if they
    were the same experiment. Every eligible liquid keeps a modern benchmarking
    run, so nothing is lost but the confound.
    """
    pipette_volume = data.get('metadata', {}).get('pipette_volume')
    results = data.get('results', {})
    if results and not all('params' in entry for entry in results.values()):
        return None

    groups: list[list[dict]] = []

    for name, entry in results.items():
        params = entry.get('params') or {}
        raw = entry.get('raw data') or {}
        volume = params.get('volume')
        if volume is None:
            # The stability campaign names its conditions 'top_1_10.0'; the volume
            # is recoverable from the name when params somehow lack it.
            tail = str(name).rsplit('_', 1)[-1]
            volume = float(tail) if tail.replace('.', '', 1).isdigit() else None

        if volume is None:
            continue

        errors = relative_errors(raw, volume)
        env = {key: raw.get(key) or [] for key in ENV_COLUMNS}

        rows: list[dict] = []
        for replicate, error in enumerate(errors):
            row = {
                'liquid': liquid,
                'campaign': campaign,
                'run_id': run_id,
                'condition_name': str(name),
                'replicate_id': replicate,
                'volume': float(volume),
                'asp_rate': params.get('asp_rate'),
                'disp_rate': params.get('disp_rate'),
                'pre_airgap': params.get('pre_airgap'),
                'post_airgap': params.get('post_airgap'),
                'volume_offset': params.get('volume_offset', 0.0) or 0.0,
                'pipette_volume': pipette_volume,
                'error': float(error),
            }
            for key in ENV_COLUMNS:
                series = env[key]
                row[key] = (
                    float(series[replicate])
                    if replicate < len(series) and series[replicate] is not None
                    else None
                )
            rows.append(row)
        groups.append(rows)

    return groups


# ---------------------------------------------------------------------------
# Optimization: params live beside the JSON, not inside it
# ---------------------------------------------------------------------------

def read_optimization_params(run_dir: Path, n_trials: int) -> dict[int, dict] | None:
    """Trial index -> dispensing parameters, from the sibling results CSV.

    Ax runs write a ``trial_index`` column; the two Atlas runs and the earliest Ax
    run do not, and for those the CSV row order is the trial order. That fallback is
    not a guess -- ``tests/test_campaign_ingest.py`` re-derives each row's mean
    absolute error from the raw weights and checks it against the CSV's own error
    column, which agrees to 1e-6 for every row of every affected run.
    """
    results_dir = run_dir / 'results'
    if not results_dir.is_dir():
        return None

    candidates = sorted(results_dir.glob('*_results.csv'))
    if not candidates:
        return None

    with candidates[0].open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    params: dict[int, dict] = {}
    for position, row in enumerate(rows):
        index = int(float(row['trial_index'])) if row.get('trial_index') else position
        entry: dict = {}
        for key in OPT_PARAM_COLUMNS:
            value = row.get(key)
            entry[key] = float(value) if value not in (None, '') else 0.0
        params[index] = entry

    # Row order only stands in for the trial index when the trial count matches; a
    # CSV that lost rows would silently misalign every parameter otherwise.
    if not any(row.get('trial_index') for row in rows) and len(rows) != n_trials:
        return None

    return params


def _optimization_rows(data: dict, run_dir: Path, run_id: str,
                       liquid: str) -> list[list[dict]] | None:
    metadata = data.get('metadata', {})
    volume = metadata.get('volume')
    pipette_volume = metadata.get('pipette_volume')
    results = data.get('results', {})

    params_by_trial = read_optimization_params(run_dir, len(results))
    if params_by_trial is None:
        return None

    groups: list[list[dict]] = []
    for key, raw in results.items():
        try:
            trial = int(key)
        except (TypeError, ValueError):
            continue
        params = params_by_trial.get(trial)
        if params is None:
            continue

        errors = relative_errors(raw, volume)
        rows: list[dict] = []
        for replicate, error in enumerate(errors):
            row = {
                'liquid': liquid,
                'campaign': 'optimization',
                'run_id': run_id,
                'condition_name': f'trial_{trial}',
                'replicate_id': replicate,
                'volume': float(volume),
                'asp_rate': params['asp_rate'],
                'disp_rate': params['disp_rate'],
                'pre_airgap': params['pre_airgap'],
                'post_airgap': params['post_airgap'],
                'volume_offset': params['volume_offset'],
                'pipette_volume': pipette_volume,
                'error': float(error),
            }
            for name in ENV_COLUMNS:
                series = raw.get(name) or []
                row[name] = (
                    float(series[replicate])
                    if replicate < len(series) and series[replicate] is not None
                    else None
                )
            rows.append(row)
        groups.append(rows)

    return groups


# ---------------------------------------------------------------------------
# Run-level acceptance
# ---------------------------------------------------------------------------

def _screen(groups: list[list[dict]], campaign: str) -> tuple[list[list[dict]], str, str]:
    """Apply the replicate-count and outlier rules to one run's conditions.

    Returns the surviving condition groups, a status and a reason. The two
    thresholds differ on purpose: an optimisation campaign is a sequence of
    independent trials, so a thin trial can be dropped on its own, whereas a
    benchmarking or verification run is a designed volume sweep, and keeping only
    the volumes that finished would silently reshape the design.
    """
    kept: list[list[dict]] = []
    for rows in groups:
        clean = [r for r in rows if abs(r['error']) <= MAX_ABS_RELATIVE_ERROR]
        if len(clean) >= MIN_REPLICATES:
            kept.append(clean)

    if not kept:
        return [], 'rejected', 'no condition cleared the replicate minimum'

    if campaign in ('benchmarking', 'verification'):
        smallest = min(len(rows) for rows in groups) if groups else 0
        if smallest < MIN_REPLICATES_FULL_RUN:
            return (
                [], 'rejected',
                f'incomplete sweep: thinnest condition has {smallest} replicates',
            )

    return kept, 'accepted', 'accepted'


def _run_id(path: Path, campaign: str) -> str:
    """The dated run folder name, e.g. '2026_08_25_anisole'."""
    stem = path.stem
    suffix = f'_{campaign}'

    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def collect(all_data: Path | None = None) -> Ingest:
    """Walk every campaign folder and accumulate transfer rows plus an audit trail."""
    root = Path(all_data or ALL_DATA_DEFAULT)
    ingest = Ingest()

    for campaign in CAMPAIGNS:
        folder = root / campaign
        if not folder.is_dir():
            continue

        for path in sorted(folder.rglob(f'*_{campaign}.json')):
            try:
                with path.open() as handle:
                    data = json.load(handle)
            except (json.JSONDecodeError, OSError) as error:
                ingest.runs.append(RunRecord(
                    campaign, _run_id(path, campaign), str(path), '?',
                    reason=f'unreadable: {error}',
                ))
                continue

            liquid = str((data.get('metadata', {}).get('liquid') or {}).get('name', '')).lower()
            record = RunRecord(campaign, _run_id(path, campaign), str(path), liquid)

            if liquid in EXCLUDED_LIQUIDS or not liquid:
                record.reason = f'excluded liquid: {liquid or "unnamed"}'
                ingest.runs.append(record)
                continue

            if campaign == 'optimization':
                groups = _optimization_rows(data, path.parent, record.run_id, liquid)
                if groups is None:
                    record.reason = 'no parameter table beside the run'
                    ingest.runs.append(record)
                    continue
            else:
                groups = _condition_rows(data, campaign, record.run_id, liquid)
                if groups is None:
                    record.reason = (
                        'legacy protocol: no per-condition params, and the run-level '
                        'settings carry touch_tip/blow_out/push_out'
                    )
                    ingest.runs.append(record)
                    continue

            kept, status, reason = _screen(groups, campaign)
            record.n_conditions = len(kept)
            record.n_transfers = sum(len(rows) for rows in kept)
            record.min_replicates = min((len(rows) for rows in groups), default=None)
            record.status, record.reason = status, reason
            ingest.runs.append(record)

            if status == 'accepted':
                for rows in kept:
                    ingest.transfers.extend(rows)

    return ingest


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def transfer_table(ingest: Ingest) -> DataFrame:
    """One row per transfer, with a stable ``condition_id`` per unique condition."""
    frame = pd.DataFrame(ingest.transfers)
    if frame.empty:
        return frame

    frame['condition_id'] = (
        frame['run_id'] + '::' + frame['campaign'] + '::' + frame['condition_name']
    )
    ordered = [
        'liquid', 'campaign', 'run_id', 'condition_id', 'condition_name',
        'replicate_id', *CONDITION_FEATURES, *ENV_COLUMNS, 'error',
    ]

    return frame[ordered].sort_values(['liquid', 'campaign', 'condition_id',
                                       'replicate_id']).reset_index(drop=True)


def condition_table(transfers: DataFrame) -> DataFrame:
    """One row per unique condition, with the replicate distribution summarised.

    ``mean_error`` is the signed mean, which is what a regression target should be;
    ``mean_abs_error`` is the accuracy proxy the optimiser minimised and is carried
    alongside so the two are never confused. ``std_error`` is the population
    standard deviation, matching the upstream campaigns' own ``stdev``.
    """
    if transfers.empty:
        return transfers

    grouped = transfers.groupby(
        ['liquid', 'campaign', 'run_id', 'condition_id', *CONDITION_FEATURES],
        as_index=False, sort=False,
    )
    frame = grouped.agg(
        mean_error=('error', 'mean'),
        std_error=('error', lambda s: float(np.std(s))),
        mean_abs_error=('error', lambda s: float(np.mean(np.abs(s)))),
        median_error=('error', 'median'),
        n_replicates=('error', 'size'),
        avg_temperature=('temperature', 'mean'),
        avg_pressure=('pressure', 'mean'),
        avg_humidity=('humidity', 'mean'),
    )

    return frame.sort_values(['liquid', 'campaign', 'condition_id']).reset_index(drop=True)


def descriptor_table(liquids: list[str]) -> DataFrame:
    """HSPiP descriptors for the given run-metadata liquid names.

    Deliberately reuses ``allmeta.subsets`` and ``ingest_ALL_data`` rather than
    re-implementing name resolution: 'tetrahydrofuran' has to reach
    'Tetrahydrofuran (THF)' and 'ethyl-acetate' has to reach 'Ethyl Acetate', and
    those rules already exist and are already tested.

    A liquid with no row in 'HSPip master' falls back to the 'HSPiP 10k' sheet, the
    same two-tier lookup ``ingest_ALL_data`` and ``subsets.load_hspip`` use (o-xylene
    is only in 10k). Master wins wherever both sheets carry a liquid.
    """
    from allmeta import screening, subsets
    from ingest_ALL_data import NameIndex, resolve_hspip_name

    master = subsets.load_master()
    names = {str(name) for name in master['Name'].dropna()}
    index = NameIndex.from_sources(names)
    wanted = {liquid: resolve_hspip_name(liquid, index) for liquid in liquids}

    unresolved = [liquid for liquid, name in wanted.items() if name not in names]
    if unresolved:
        tenk_names = {
            str(name) for name in pd.read_excel(
                subsets.HSPIP_XLSX, sheet_name=subsets.TENK_SHEET, usecols=['Name'],
            )['Name'].dropna()
        }
        tenk_index = NameIndex.from_sources(tenk_names)
        recovered = {liquid: resolve_hspip_name(liquid, tenk_index) for liquid in unresolved}
        recovered = {liquid: name for liquid, name in recovered.items() if name in tenk_names}
        if recovered:
            master = pd.concat(
                [master, subsets.load_recovered(tuple(sorted(set(recovered.values()))))],
                ignore_index=True,
            )
            wanted.update(recovered)
            names |= set(recovered.values())

    missing = sorted(liquid for liquid, name in wanted.items() if name not in names)
    if missing:
        raise ValueError(f'no HSPiP descriptor row for {missing} in either sheet')

    rows = screening.attach_derived(master).set_index('Name')
    frame = pd.DataFrame({
        'liquid': list(wanted),
        'hspip_name': [wanted[liquid] for liquid in wanted],
    })
    for column in descriptor_columns():
        frame[column] = [rows.loc[wanted[liquid], column] for liquid in wanted]

    return frame.sort_values('liquid').reset_index(drop=True)


# The six descriptors the purchase-candidate work clusters on. A different
# objective -- spanning chemical space when deciding what to buy, not predicting
# dispensing error -- so it is offered as an alternative rather than the default.
# See docs/liquid_purchase_candidates/summary.md.
COVERAGE_SIX: tuple[str, ...] = ('log η', 'SurfTen', 'EdmiW', 'Ant1T', 'Density', 'δP')


def descriptor_columns(descriptor_set: str = DESCRIPTOR_SET) -> list[str]:
    """The descriptor columns for a named controlled set.

    Three are available and none is searched for here. Searching a descriptor set
    on nine liquids would be fitting the fold, which is the failure mode the
    existing feature-selection work already documented at thirteen.
    """
    if descriptor_set == 'physics':
        return list(cols.PHYSICS_KEEP)
    if descriptor_set == 'coverage6':
        return list(COVERAGE_SIX)
    if descriptor_set == 'manual':
        return [cols.DERIVED_VAPOUR_PRESSURE, *cols.MANUAL_KEEP]

    raise ValueError(f'unknown descriptor set {descriptor_set!r}')


def eligible_liquids(conditions: DataFrame) -> list[str]:
    """Liquids carrying BOTH optimization and benchmarking conditions.

    The brief for this comparison: a task is only a task if it has been both
    explored (optimisation, ~100 parameter sets at one volume) and characterised
    (benchmarking, one parameter set across eight volumes). A liquid with only one
    of the two cannot exercise the context/target split the Neural Processes are
    being asked to learn.
    """
    have = conditions.groupby('liquid')['campaign'].agg(set)
    keep = [liquid for liquid, campaigns in have.items()
            if {'optimization', 'benchmarking'} <= campaigns]

    return sorted(keep)


def build(all_data: Path | None = None, write: bool = True) -> dict[str, DataFrame]:
    """Ingest everything, restrict to eligible liquids, and optionally persist."""
    ingest = collect(all_data)
    transfers = transfer_table(ingest)
    conditions = condition_table(transfers)

    liquids = eligible_liquids(conditions)
    transfers = transfers[transfers['liquid'].isin(liquids)].reset_index(drop=True)
    conditions = conditions[conditions['liquid'].isin(liquids)].reset_index(drop=True)
    descriptors = descriptor_table(liquids)

    manifest = pd.DataFrame([record.as_row() for record in ingest.runs])
    manifest['eligible_liquid'] = manifest['liquid'].isin(liquids)

    tables = {
        'transfers': transfers, 'conditions': conditions,
        'descriptors': descriptors, 'manifest': manifest,
    }

    if write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        transfers.to_csv(TRANSFER_TABLE, index=False)
        conditions.to_csv(CONDITION_TABLE, index=False)
        descriptors.to_csv(DESCRIPTOR_TABLE, index=False)
        manifest.to_csv(INGEST_MANIFEST, index=False)

    return tables


def load() -> dict[str, DataFrame]:
    """Read the persisted tables, building them first if they are absent."""
    if not CONDITION_TABLE.exists():
        return build()

    return {
        'transfers': pd.read_csv(TRANSFER_TABLE),
        'conditions': pd.read_csv(CONDITION_TABLE),
        'descriptors': pd.read_csv(DESCRIPTOR_TABLE),
        'manifest': pd.read_csv(INGEST_MANIFEST),
    }
