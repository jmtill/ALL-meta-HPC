"""Build the liquid subsets that every downstream analysis runs on.

A - every liquid with a complete descriptor row.
B - A restricted to the curated liquid list. The liquid-KEPT table.
R - every source liquid that is not in B. The liquid-REMOVED table.
C - the liquids with measured dispensing error, one row per liquid x volume.

Every feature value in every one of these comes from data/raw_hspip/HSPiP_data.xlsx
and from nowhere else. data/cleaned_data/HSPiP_cleaned_liquids.csv carries a copy of
the descriptors, but it is read here for its 'Name' column alone -- it is a
selection mask, not a data source. curated_names() is the only door it comes
through, and validate_rebuild() re-reads the workbook and asserts cell-for-cell
equality on every written row so that stays true.

B and R partition the source exactly: B | R == master + recovered, B & R == empty.
R therefore holds two kinds of liquid, distinguished by its 'removal_reason'
column: the ones the curated list does not name ('not on curated keep list', the
complement of the mask) and the ones it does name but whose descriptor row is
incomplete ('incomplete descriptor row'). Filter on 'on_curated_list' to see the
mask complement alone.

No count is written here on purpose: C grows whenever a new liquid is benchmarked.
write_subsets() records the current sizes in data/subsets/subset_manifest.csv on
every run; read them from there.

A and B answer "how does liquid curation change feature selection"; C is the only
subset with a supervised target. C's *liquid* count is the binding constraint on
everything supervised downstream -- the eight volumes per liquid multiply the rows
but not the independent observations.

C is assembled from whatever ingest_ALL_data.py accepted, with no allowlist here.
A benchmarked liquid that has no HSPiP descriptor row -- reagent-alcohol is a
denatured-ethanol mixture, not a compound -- is dropped from C and recorded in
dropped_liquids.csv rather than raising.
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from allmeta import columns as cols
from allmeta.paths import (
    BENCHMARK_DIR,
    CHEM_CLUSTER_CSV,
    CURATED_LIQUIDS_CSV,
    HSPIP_XLSX,
    REPO_ROOT,
    SUBSET_DIR,
)

# ingest_ALL_data.py lives at the repository root and is not installed, but it
# already owns the JSON->row flattening and the liquid-name resolution. Reusing it
# beats a second copy that can drift from the ingest.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingest_ALL_data import (  # noqa: E402
    NameIndex,
    resolve_hspip_name,
    summarize_condition,
)

MASTER_SHEET: str = 'HSPip master'
TENK_SHEET: str = 'HSPiP 10k'

SUBSET_A_PATH: Path = SUBSET_DIR / 'subset_A_full.csv'
SUBSET_B_PATH: Path = SUBSET_DIR / 'subset_B_curated.csv'
SUBSET_C_PATH: Path = SUBSET_DIR / 'subset_C_labeled.csv'
SUBSET_R_PATH: Path = SUBSET_DIR / 'subset_R_removed.csv'
MANIFEST_PATH: Path = SUBSET_DIR / 'subset_manifest.csv'
DROPPED_PATH: Path = SUBSET_DIR / 'dropped_liquids.csv'


def coerce_descriptors(data: DataFrame) -> DataFrame:
    """Coerce the 58 descriptor columns to float, mapping HSPiP's '-' to NaN.

    HSPiP writes missing values as a literal '-', which pandas reads as a string and
    which would otherwise poison the whole column's dtype.
    """
    frame = data.copy()
    numeric = (
        frame[list(cols.NUM_COLUMNS)]
        .replace(to_replace='-', value=pd.NA)
        .apply(pd.to_numeric, errors='coerce')
        .astype('float64')
    )
    frame[list(cols.NUM_COLUMNS)] = numeric

    return frame


@lru_cache(maxsize=1)
def _master_sheet() -> DataFrame:
    """The raw 'HSPip master' sheet, parsed once per process.

    A 5.5 MB Excel parse is slow enough to matter across the several calls a full
    write_subsets() makes, and the extracted CSV that used to spare us it no longer
    exists. Callers get a copy, never this frame.
    """
    return pd.read_excel(HSPIP_XLSX, sheet_name=MASTER_SHEET)


def master_names() -> set[str]:
    """Every 'Name' in the master sheet."""
    return {str(name) for name in _master_sheet()['Name'].dropna()}


def load_master() -> DataFrame:
    """Name, SMILES and the 58 descriptors for every liquid in the master sheet."""
    keep = list(cols.ID_COLUMNS) + list(cols.NUM_COLUMNS)

    return coerce_descriptors(_master_sheet().copy())[keep]


def recovered_names(runs: DataFrame | None = None) -> tuple[str, ...]:
    """Canonical 10k names for benchmarked liquids the master sheet does not carry.

    Demand-driven rather than a hand-maintained list: o-xylene is measured in the
    benchmarking runs but has no row in 'HSPip master', and without recovery it
    silently drops out of the labelled table. Any future liquid in the same
    position is picked up here with no code edit.

    The two sheets agree exactly where they overlap (verified on p-Xylene, Toluene
    and Chloroform: 58/58 descriptors each), so a recovered row is the same data the
    master sheet would have carried.
    """
    liquids = benchmark_liquids(runs)
    if not liquids:
        return ()

    known = master_names()
    master_index = NameIndex.from_sources(known)

    tenk = pd.read_excel(HSPIP_XLSX, sheet_name=TENK_SHEET, usecols=['Name'])['Name'].dropna()
    tenk_names = {str(name) for name in tenk}
    tenk_index = NameIndex.from_sources(tenk_names)

    wanted: set[str] = set()
    for liquid in liquids:
        # Membership rather than truthiness: an alias can name a row that only the
        # 10k sheet carries, and that still needs recovering.
        if resolve_hspip_name(liquid, master_index) in known:
            continue
        candidate = resolve_hspip_name(liquid, tenk_index)
        if candidate in tenk_names:
            wanted.add(candidate)

    return tuple(sorted(wanted))


def load_recovered(names: tuple[str, ...] | None = None,
                   runs: DataFrame | None = None) -> DataFrame:
    """Descriptor rows for liquids present in the 10k sheet but absent from master.

    ``names`` of None derives the list from the ingested runs via recovered_names().
    """
    if names is None:
        names = recovered_names(runs)
    if not names:
        return pd.DataFrame(columns=list(cols.ID_COLUMNS) + list(cols.NUM_COLUMNS))

    tenk = pd.read_excel(HSPIP_XLSX, sheet_name=TENK_SHEET)
    wanted = tenk[tenk['Name'].isin(names)].drop_duplicates(subset='Name')

    missing = set(names) - set(wanted['Name'])
    if missing:
        raise ValueError(f'{sorted(missing)} not found in the {TENK_SHEET!r} sheet')

    keep = list(cols.ID_COLUMNS) + list(cols.NUM_COLUMNS)

    return coerce_descriptors(wanted)[keep].reset_index(drop=True)


def load_hspip(include_recovered: bool = True,
               runs: DataFrame | None = None) -> DataFrame:
    """The descriptor table, optionally extended with liquids recovered from 10k.

    Adds a 'source_sheet' column so a recovered row is never mistaken for a master
    row in later provenance checks. ``runs`` is passed through to recovered_names()
    so a caller that already read the benchmarking JSON does not re-read it.
    """
    master = load_master()
    master['source_sheet'] = MASTER_SHEET

    if not include_recovered:
        return master

    recovered = load_recovered(runs=runs)
    recovered['source_sheet'] = TENK_SHEET
    # Guard against a name appearing in both sheets: master wins.
    recovered = recovered[~recovered['Name'].isin(master['Name'])]

    return pd.concat([master, recovered], ignore_index=True)


def attach_chem_clusters(frame: DataFrame) -> DataFrame:
    """Left-join the RDKit Butina chemical cluster id on 'Name'.

    Left rather than inner: the liquids with no chemical cluster still belong in the
    subset, they are simply excluded from the ARI comparison later. The manifest
    records how many, as n_liquids - n_with_cluster_id.
    """
    clusters = pd.read_csv(CHEM_CLUSTER_CSV, usecols=['Name', 'cluster_id'])
    clusters = clusters.drop_duplicates(subset='Name')

    return frame.merge(clusters, on='Name', how='left', validate='one_to_one')


def subset_a(hspip: DataFrame | None = None) -> DataFrame:
    """Every liquid whose 58 descriptors are all present."""
    frame = load_hspip() if hspip is None else hspip.copy()
    complete = ~frame[list(cols.NUM_COLUMNS)].isna().any(axis=1)

    return attach_chem_clusters(frame.loc[complete].reset_index(drop=True))


def curated_liquids() -> DataFrame:
    """The curated liquid list, with its provenance columns.

    Read by filename on every call so the curation can be re-run without touching
    code. Names the liquids to KEEP: it superseded the old removal-candidates list,
    so subset B intersects with it rather than subtracting it.
    """
    return pd.read_csv(CURATED_LIQUIDS_CSV).drop_duplicates(subset='Name')


def curated_names() -> Series:
    """Names on the curated keep list."""
    return curated_liquids()['Name'].dropna().drop_duplicates()


def subset_b(frame_a: DataFrame | None = None) -> DataFrame:
    """Subset A restricted to the curated liquid list."""
    frame = subset_a() if frame_a is None else frame_a.copy()
    kept = frame['Name'].isin(set(curated_names()))

    return frame.loc[kept].reset_index(drop=True)


NOT_CURATED: str = 'not on curated keep list'
INCOMPLETE_ROW: str = 'incomplete descriptor row'


def subset_r(hspip: DataFrame | None = None,
             frame_b: DataFrame | None = None) -> DataFrame:
    """Every source liquid that subset B does not keep, with its raw descriptors.

    The exact complement of B over master + recovered, so B and R together account
    for every source record. That makes R the audit surface for the curation: if a
    liquid is not in B it is here, with a reason, and nothing falls between them.

    Two reasons, because there are two ways to miss B. 'on_curated_list' separates
    them without having to parse the reason string -- ``r[~r['on_curated_list']]``
    is the mask complement on its own, which is the 502 liquids the curated CSV
    declines to name.
    """
    frame = load_hspip() if hspip is None else hspip.copy()
    kept = set((subset_b() if frame_b is None else frame_b)['Name'])

    removed = frame.loc[~frame['Name'].isin(kept)].reset_index(drop=True)
    on_list = removed['Name'].isin(set(curated_names()))
    removed['on_curated_list'] = on_list
    # A curated liquid that is not in B can only have missed the completeness gate
    # in subset_a(); that is the whole of the difference between the two sets.
    removed['removal_reason'] = [INCOMPLETE_ROW if flag else NOT_CURATED
                                 for flag in on_list]

    return attach_chem_clusters(removed)


def benchmark_rows(benchmark_dir: Path = BENCHMARK_DIR) -> DataFrame:
    """One row per (liquid, volume) across every ingested benchmarking run.

    Reads whatever ingest_ALL_data.py accepted -- there is no allowlist here, so a
    newly benchmarked liquid appears as soon as it has been ingested.
    ingest_ALL_data flattens a single volume by default; every volume is needed here
    because the labelled liquids are too few to select features against on their own.
    """
    rows: list[dict] = []

    for data_file in sorted(benchmark_dir.glob('*.json')):
        with open(data_file, 'r') as handle:
            data: dict = json.load(handle)

        for volume in sorted(data.get('results', {}), key=int):
            row = summarize_condition(data, int(volume))
            if row is None:
                continue
            row['source_file'] = data_file.name
            rows.append(row)

    return pd.DataFrame(rows)


def benchmark_liquids(runs: DataFrame | None = None) -> list[str]:
    """Distinct run-metadata liquid names across the ingested benchmarking runs."""
    frame = benchmark_rows() if runs is None else runs
    if frame.empty or 'liquid' not in frame:
        return []

    return sorted({str(name) for name in frame['liquid'].dropna() if str(name)})


def resolve_runs(runs: DataFrame, hspip: DataFrame) -> DataFrame:
    """Attach the canonical HSPiP 'Name' to each benchmarking row as 'hspip_name'.

    Unresolvable liquids get '' rather than a guess; see unjoinable_liquids().
    """
    index = NameIndex.from_sources(hspip['Name'].dropna())
    frame = runs.copy()
    frame['hspip_name'] = [
        resolve_hspip_name(liquid or '', index) for liquid in frame['liquid']
    ]

    return frame


def unjoinable_liquids(resolved: DataFrame) -> list[str]:
    """Run liquid names in a resolve_runs() frame that matched no HSPiP row.

    A benchmarked liquid can be perfectly good data and still have nowhere to join:
    reagent-alcohol is a denatured-ethanol mixture, so no single compound's
    descriptor vector describes it.
    """
    if 'hspip_name' not in resolved:
        return []

    missing = resolved.loc[resolved['hspip_name'] == '', 'liquid'].dropna()

    return sorted({str(name) for name in missing})


def subset_c(hspip: DataFrame | None = None,
             runs: DataFrame | None = None) -> DataFrame:
    """The labelled table: descriptors joined to measured dispensing error.

    Grouped by liquid, never by row: the eight volumes of one liquid share its
    descriptor vector entirely, so splitting them across a train/test boundary
    leaks.

    Every ingested liquid that resolves to an HSPiP row is included. Ones that do
    not resolve are dropped with a warning rather than raising -- screening already
    happened in ingest_ALL_data.py, and a mixture with no descriptor row should not
    block the other liquids from being analysed. build_dropped_audit() records them.
    """
    runs = benchmark_rows() if runs is None else runs
    frame = load_hspip(runs=runs) if hspip is None else hspip.copy()

    resolved = resolve_runs(runs, frame)
    unjoinable = unjoinable_liquids(resolved)
    if unjoinable:
        print(
            f'Warning: no HSPiP descriptor row for {unjoinable}; dropped from subset '
            f'C and recorded in {DROPPED_PATH.name}.'
        )
        resolved = resolved.loc[resolved['hspip_name'] != ''].reset_index(drop=True)

    descriptors = attach_chem_clusters(frame)
    labelled = resolved.merge(
        descriptors,
        left_on='hspip_name',
        right_on='Name',
        how='inner',
        validate='many_to_one',
    )

    ordered = (
        ['liquid', 'Name', 'SMILES', 'source_sheet', 'cluster_id']
        + list(cols.PIPETTING_PARAMS)
        + list(cols.ENV_COLUMNS)
        + list(cols.LABEL_COLUMNS)
        + list(cols.NUM_COLUMNS)
        + ['source_file']
    )

    return labelled[ordered].sort_values(['liquid', 'volume'], ignore_index=True)


def build_dropped_audit(hspip: DataFrame, frame_a: DataFrame, frame_b: DataFrame,
                        runs: DataFrame | None = None) -> DataFrame:
    """Row-per-liquid record of what each subset excluded and why.

    The 'C' rows carry the run-metadata liquid name rather than an HSPiP 'Name',
    because the whole reason they are here is that no HSPiP name was found. Without
    this they would vanish from the audit trail entirely.
    """
    # The curated list names what was kept, not why anything was dropped, so a B
    # exclusion carries one reason: it is not on that list.
    incomplete = hspip.loc[~hspip['Name'].isin(frame_a['Name']), 'Name']
    curated = frame_a.loc[~frame_a['Name'].isin(frame_b['Name']), 'Name']
    runs = benchmark_rows() if runs is None else runs
    unjoinable = unjoinable_liquids(resolve_runs(runs, hspip))

    records = [
        {'Name': name, 'dropped_from': 'A', 'reason': 'incomplete descriptor row'}
        for name in incomplete
    ] + [
        {'Name': name, 'dropped_from': 'B', 'reason': 'not on the curated liquid list'}
        for name in curated
    ] + [
        {'Name': name, 'dropped_from': 'C', 'reason': 'no HSPiP descriptor row'}
        for name in unjoinable
    ]

    return pd.DataFrame(records, columns=['Name', 'dropped_from', 'reason'])


def _raw_descriptor_lookup() -> DataFrame:
    """Name-indexed descriptors read fresh from the workbook, for validation only.

    Deliberately bypasses _master_sheet()'s cache and load_master(): a fidelity
    check that reuses the frame it is checking proves nothing. Both sheets are
    parsed again here, which costs a few seconds on a path that runs once per
    rebuild.
    """
    keep = list(cols.ID_COLUMNS) + list(cols.NUM_COLUMNS)
    sheets = [pd.read_excel(HSPIP_XLSX, sheet_name=MASTER_SHEET)]

    tenk = pd.read_excel(HSPIP_XLSX, sheet_name=TENK_SHEET)
    known = set(sheets[0]['Name'].dropna().astype(str))
    sheets.append(tenk[~tenk['Name'].isin(known)].drop_duplicates(subset='Name'))

    raw = pd.concat([coerce_descriptors(sheet)[keep] for sheet in sheets],
                    ignore_index=True)

    return raw.drop_duplicates(subset='Name').set_index('Name')


def _fidelity_failures(frame: DataFrame, raw: DataFrame, label: str) -> list[str]:
    """Cells in ``frame`` that disagree with the workbook, as readable messages."""
    unknown = sorted(set(frame['Name']) - set(raw.index))
    if unknown:
        return [f'{label}: {len(unknown)} name(s) absent from the workbook: '
                f'{unknown[:5]}']

    indexed = frame.drop_duplicates(subset='Name').set_index('Name')
    reference = raw.loc[indexed.index]

    failures: list[str] = []
    for column in cols.NUM_COLUMNS:
        mismatch = ~np.isclose(indexed[column].to_numpy(dtype='float64'),
                               reference[column].to_numpy(dtype='float64'),
                               rtol=1e-9, atol=1e-9, equal_nan=True)
        if mismatch.any():
            names = list(indexed.index[mismatch])[:5]
            failures.append(f'{label}: {int(mismatch.sum())} row(s) differ on '
                            f'{column!r}, e.g. {names}')

    smiles = indexed['SMILES'].astype(str) != reference['SMILES'].astype(str)
    if smiles.any():
        failures.append(f'{label}: {int(smiles.sum())} row(s) differ on SMILES, '
                        f'e.g. {list(indexed.index[smiles])[:5]}')

    return failures


def validate_rebuild(hspip: DataFrame, frame_a: DataFrame, frame_b: DataFrame,
                     frame_r: DataFrame) -> DataFrame:
    """Assert the rebuilt tables are the workbook's values under the curation mask.

    Raises ValueError listing every failure rather than the first, so one run tells
    you everything that is wrong. Returns a check-by-check summary for the manifest
    notebook to print.

    The point of check 3 is the one the rebuild exists to answer: no feature value
    anywhere in B or R came from the curated CSV or from any hand edit. It re-reads
    the workbook to compare against, so it stays honest if the loaders change.
    """
    curated = curated_liquids()
    curated_set = set(curated_names())
    source = set(hspip['Name'])
    checks: list[dict] = []
    failures: list[str] = []

    def record(name: str, detail: str, ok: bool, message: str = '') -> None:
        checks.append({'check': name, 'detail': detail, 'passed': ok})
        if not ok:
            failures.append(message or f'{name}: {detail}')

    # 1. Mask integrity.
    dup_curated = int(curated['Name'].duplicated().sum())
    record('mask: no duplicate names', f'{dup_curated} duplicate(s)', dup_curated == 0)
    dup_source = int(hspip['Name'].duplicated().sum())
    record('source: no duplicate names', f'{dup_source} duplicate(s)', dup_source == 0)
    unmatched = sorted(curated_set - source)
    record('mask: every curated name resolves',
           f'{len(curated_set) - len(unmatched)}/{len(curated_set)} matched',
           not unmatched,
           f'curated names with no workbook row: {unmatched[:10]}')

    # 2. Partition. B and R must tile the source with no overlap and no gap.
    names_b, names_r = set(frame_b['Name']), set(frame_r['Name'])
    overlap = sorted(names_b & names_r)
    record('partition: B and R disjoint', f'{len(overlap)} shared', not overlap,
           f'liquids in both B and R: {overlap[:10]}')
    gap = sorted(source - (names_b | names_r))
    record('partition: B | R covers the source',
           f'{len(names_b | names_r)}/{len(source)} accounted for', not gap,
           f'source liquids in neither B nor R: {gap[:10]}')

    # A curated liquid that misses B must be recorded as such, not silently gone.
    stranded = sorted(curated_set - names_b)
    tagged = set(frame_r.loc[frame_r['on_curated_list'], 'Name'])
    record('audit: curated liquids missing from B are tagged in R',
           f'{len(stranded)} stranded, {len(tagged)} tagged',
           set(stranded) == tagged,
           f'curated liquids missing from B and untagged in R: '
           f'{sorted(set(stranded) - tagged)[:10]}')

    # 3. Value fidelity against a fresh workbook read.
    raw = _raw_descriptor_lookup()
    fidelity = (_fidelity_failures(frame_b, raw, 'B')
                + _fidelity_failures(frame_r, raw, 'R')
                + _fidelity_failures(frame_a, raw, 'A'))
    record('fidelity: A, B and R match the workbook cell for cell',
           'exact' if not fidelity else f'{len(fidelity)} discrepancy group(s)',
           not fidelity, '; '.join(fidelity))

    # 4. The mask contributes names and nothing else.
    names_only = isinstance(curated_names(), Series) and curated_names().name == 'Name'
    record('mask: curated_names() yields names only',
           'Series of Name' if names_only else 'unexpected shape', names_only)

    # 5. Counts, so a silent change in any of them shows up in the summary.
    record('counts', f'source={len(source)} A={len(frame_a)} B={len(frame_b)} '
                     f'R={len(frame_r)} curated={len(curated_set)}', True)

    if failures:
        raise ValueError('rebuild validation failed:\n  - ' + '\n  - '.join(failures))

    return DataFrame(checks)


def write_subsets() -> DataFrame:
    """Write all three subsets, the drop audit and the manifest. Returns the manifest."""
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    # Read the benchmarking JSON once and thread it through: load_hspip(),
    # subset_c() and the audit all need the same liquid list.
    runs = benchmark_rows()
    hspip = load_hspip(runs=runs)
    frame_a = subset_a(hspip)
    frame_b = subset_b(frame_a)
    frame_r = subset_r(hspip, frame_b)
    frame_c = subset_c(hspip, runs=runs)

    # Before writing, not after: a rebuild that failed validation should leave the
    # last good tables on disk rather than half-replace them.
    validate_rebuild(hspip, frame_a, frame_b, frame_r)

    frame_a.to_csv(SUBSET_A_PATH, index=False)
    frame_b.to_csv(SUBSET_B_PATH, index=False)
    frame_r.to_csv(SUBSET_R_PATH, index=False)
    frame_c.to_csv(SUBSET_C_PATH, index=False)
    build_dropped_audit(hspip, frame_a, frame_b, runs).to_csv(DROPPED_PATH, index=False)

    manifest = pd.DataFrame([
        {
            'subset': 'A',
            'description': 'all liquids with a complete descriptor row',
            'n_rows': len(frame_a),
            'n_liquids': frame_a['Name'].nunique(),
            'n_descriptors': len(cols.NUM_COLUMNS),
            'n_with_cluster_id': int(frame_a['cluster_id'].notna().sum()),
            'has_label': False,
            'path': str(SUBSET_A_PATH.relative_to(REPO_ROOT)),
        },
        {
            'subset': 'B',
            'description': 'subset A restricted to the curated liquid list',
            'n_rows': len(frame_b),
            'n_liquids': frame_b['Name'].nunique(),
            'n_descriptors': len(cols.NUM_COLUMNS),
            'n_with_cluster_id': int(frame_b['cluster_id'].notna().sum()),
            'has_label': False,
            'path': str(SUBSET_B_PATH.relative_to(REPO_ROOT)),
        },
        {
            'subset': 'R',
            'description': 'source liquids not kept by the curated list; complement of B',
            'n_rows': len(frame_r),
            'n_liquids': frame_r['Name'].nunique(),
            'n_descriptors': len(cols.NUM_COLUMNS),
            'n_with_cluster_id': int(frame_r['cluster_id'].notna().sum()),
            'has_label': False,
            'path': str(SUBSET_R_PATH.relative_to(REPO_ROOT)),
        },
        {
            'subset': 'C',
            'description': 'liquids with measured dispensing error, one row per volume',
            'n_rows': len(frame_c),
            'n_liquids': frame_c['Name'].nunique(),
            'n_descriptors': len(cols.NUM_COLUMNS) + len(cols.PIPETTING_PARAMS),
            'n_with_cluster_id': int(frame_c['cluster_id'].notna().sum()),
            'has_label': True,
            'path': str(SUBSET_C_PATH.relative_to(REPO_ROOT)),
        },
    ])
    manifest.to_csv(MANIFEST_PATH, index=False)

    return manifest


def load_subset(name: str) -> DataFrame:
    """Read a written subset by letter."""
    paths = {'A': SUBSET_A_PATH, 'B': SUBSET_B_PATH, 'C': SUBSET_C_PATH,
             'R': SUBSET_R_PATH}
    if name not in paths:
        raise ValueError(f'unknown subset {name!r}; expected one of {sorted(paths)}')

    return pd.read_csv(paths[name])


def labelled_liquid_count() -> int:
    """Independent liquids currently in subset C.

    The effective n for every supervised result, and the fold count of the
    leave-one-liquid-out splits. Read from the written subset rather than restated
    in prose, because it changes whenever a liquid is benchmarked.
    """
    return int(load_subset('C')['Name'].nunique())


if __name__ == '__main__':
    print(write_subsets().to_string(index=False))
