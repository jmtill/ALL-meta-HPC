"""Canonical column vocabularies for the HSPiP tables.

These lists existed in five near-copies across the notebooks, two of which had
diverged. This module is the single source of truth; nothing else should hardcode a
descriptor list.

Scope: the *column lists* are centralized here, not the notebooks' cleaning helpers.
``extract_numerical_data`` still exists in two diverged forms in
notebooks/data_extraction/ (one returns a frame, the other a 3-tuple). Both are
superseded by ``allmeta.subsets.coerce_descriptors``; use that in new code.

Column names carry non-ASCII characters and punctuation that do not survive
slugification -- ``log η`` has an ASCII space and U+03B7, ``Cond.`` has a trailing
period, ``#Heavy Atom`` starts with ``#``, ``ΔHv@BPt`` has an ``@``. Always index by
the exact string.
"""

# The 58 numeric HSPiP descriptors, in sheet order.
NUM_COLUMNS: tuple[str, ...] = (
    'δD', 'δP', 'δH', 'δHDon', 'δHAcc', 'MWt', 'Density', 'MVol', 'Area', 'Ovality',
    'BPt', 'MPt', 'Tc', 'Pc', 'Vc', 'Zc', 'AntA', 'AntB', 'AntC', 'Ant1T', 'LogKow',
    'LogS', 'Henry', 'LogOHR', 'RI', 'ΔHfus', 'ΔHv@BPt', 'Trouton', 'RER', 'Abr-a',
    'Abr-b', 'EdmiW', 'Parachor', 'RD', 'ΔCp', 'log η', 'Cond.', 'SurfTen',
    '#Heavy Atom', 'C#', 'H#', 'Br#', 'Cl#', 'F#', 'I#', 'N#', 'O#', 'P#', 'S#',
    'Si#', 'B#', 'MaxPc', 'MinMc', 'Sym', 'MCI', 'ΔHcomb', 'ΔHform', 'ΔGform',
)

ID_COLUMNS: tuple[str, ...] = ('Name', 'SMILES')

# Viscosity. Retained through every screening stage regardless of what the variance,
# correlation or clustering screens say, because it is mechanistically the property
# most likely to drive pipetting error.
PROTECTED: tuple[str, ...] = ('log η',)

# Atom counts and heavy-atom total. Dropped before variance screening because they
# are composition bookkeeping rather than measured liquid properties; several are
# near-constant (P# and B# have two distinct values across 1237 liquids).
COMP_COLUMNS: tuple[str, ...] = (
    '#Heavy Atom', 'C#', 'H#', 'Br#', 'Cl#', 'F#', 'I#', 'N#', 'O#', 'P#', 'S#',
    'Si#', 'B#',
)

# Hand-picked shortlist carried over verbatim from notebooks/reduced_cleaning.ipynb.
MANUAL_KEEP: tuple[str, ...] = (
    'δP', 'δH', 'MWt', 'Density', 'MVol', 'Area', 'Ovality', 'MPt', 'Tc', 'Ant1T',
    'LogKow', 'Henry', 'EdmiW', 'Parachor', 'ΔCp', 'log η', 'Cond.', 'SurfTen',
    'Sym', 'MCI',
)

# Derived descriptor: log10 of vapour pressure at 25 C, from the Antoine equation
# as HSPiP parameterises it, log10(P) = A - B / (T + C). AntA, AntB and AntC are
# fitted jointly to one vapour-pressure curve and are near-degenerate -- AntB and
# AntC correlate at |rho| = 0.81 on subset B -- so selecting among them
# individually, as every screen in this project did, is physically incoherent. The
# quantity that actually means something is the pressure they jointly imply.
REFERENCE_TEMPERATURE_C: float = 25.0
DERIVED_VAPOUR_PRESSURE: str = 'logPvap25'

# The three raw coefficients the derived quantity replaces. Named once so a screen
# cannot drop two of them and leave the third looking like an independent descriptor.
RAW_ANTOINE: tuple[str, ...] = ('AntA', 'AntB', 'AntC')

# Descriptors chosen on mechanism rather than by search, to be fixed BEFORE the next
# benchmarking run and tested on the liquids it adds. Air-displacement dispensing
# error is driven by evaporation from the tip and by how the liquid loads and drains
# it, so: one volatility term, one bulk-transport term, one surface term, one
# viscosity term. Four features for thirteen liquids is already generous; the
# thirteen-selector search in `supervised` cannot resolve more than about one
# degree of freedom at this sample size, and every strong descriptor it finds
# (Ant1T, BPt, AntB, dHv@BPt) is a rotation of the same volatility axis.
PHYSICS_KEEP: tuple[str, ...] = (
    DERIVED_VAPOUR_PRESSURE, 'MVol', 'SurfTen', 'log η',
)

# Pipetting protocol settings that vary within a benchmarking run. Features of the
# labelled table, not liquid properties: volume is confounded with rate and airgap
# (800/250/100/50/45 uL run at 250/300 with 50 uL airgaps; 30/20/10 uL at 20/20
# with 2 uL airgaps), so they cannot be interpreted independently of each other.
PIPETTING_PARAMS: tuple[str, ...] = (
    'volume', 'asp_rate', 'disp_rate', 'pre_airgap', 'post_airgap',
)

# Per-condition environmental averages available alongside the label.
ENV_COLUMNS: tuple[str, ...] = ('avg_temperature', 'avg_pressure', 'avg_humidity')

# Supervised target and its companions. 'avg_abs_error' is the accuracy proxy,
# 'error_stdev' the precision proxy.
TARGET: str = 'avg_abs_error'
LABEL_COLUMNS: tuple[str, ...] = (TARGET, 'error_stdev', 'n_replicates')

# Run-metadata liquid names that are a genuine SYNONYM of their HSPiP 'Name' --
# different words for the same compound, which no normalisation rule can bridge.
# Single source of truth: ingest_ALL_data.py imports this rather than keeping its
# own copy, which had already drifted.
#
# Do NOT add a liquid here just because its spelling differs. Handled already:
#   * punctuation and capitalisation ('ethyl-acetate' -> 'Ethyl Acetate') and
#     trailing qualifiers ('tetrahydrofuran' -> 'Tetrahydrofuran (THF)'), by
#     ingest_ALL_data.normalize_liquid_name();
#   * o-/m-/p- against 1,2-/1,3-/1,4- ('1,2-dichlorobenzene' ->
#     'o-Dichlorobenzene'), by ingest_ALL_data.name_variants();
#   * liquids absent from the master sheet ('o-xylene'), recovered from
#     'HSPiP 10k' on demand by subsets.recovered_names().
# The first and third used to be entries here; a newly benchmarked liquid should
# need no code edit at all.
HSPIP_ALIASES: dict[str, str] = {
    'n-butanol': '1-Butanol',
}


# Physical units for the columns that carry one, for axis labels. Verified against
# reference values where possible: water gives Tc=647.096 K, Pc=22.064 MPa,
# MVol=18.0 cm3/mol, BPt=100 C and dHform=-285.83 kJ/mol; dHv@BPt=40.68 vs a
# literature 40.65 kJ/mol; toluene/benzene/ethanol dHcomb match -3.91/-3.27/-1.37
# MJ/mol.
#
# Two traps this table exists to prevent:
#   * BPt and MPt are degrees Celsius while Tc is Kelvin.
#   * TARGET ('avg_abs_error') is a dimensionless FRACTION of the target volume,
#     not a volume -- 0.24 means a 24% dispensing error. See TARGET_UNIT_WARNING
#     for why it is recomputed rather than read from the runs.
#
# Columns absent from this map are either dimensionless (log-scaled quantities,
# indices, counts, ratios) or have a unit that could not be established from the
# data -- Antoine coefficients, RER, EdmiW, Parachor, 'Cond.', MaxPc/MinMc and the
# Abraham parameters. They are deliberately left blank rather than guessed.
UNITS: dict[str, str] = {
    # Hansen solubility parameters
    'δD': 'MPa^0.5', 'δP': 'MPa^0.5', 'δH': 'MPa^0.5',
    'δHDon': 'MPa^0.5', 'δHAcc': 'MPa^0.5',
    # Bulk properties
    'MWt': 'g/mol', 'Density': 'g/cm³', 'MVol': 'cm³/mol', 'Area': 'Å²',
    # Phase behaviour -- note the Celsius/Kelvin split
    'BPt': '°C', 'MPt': '°C', 'Tc': 'K', 'Pc': 'MPa', 'Vc': 'cm³/mol',
    # Thermochemistry
    'ΔHfus': 'kJ/mol', 'ΔHv@BPt': 'kJ/mol', 'ΔHcomb': 'MJ/mol',
    'ΔHform': 'kJ/mol', 'ΔGform': 'kJ/mol',
    'Trouton': 'J/(mol·K)', 'ΔCp': 'J/(mol·K)',
    # Transport and surface
    'log η': 'log₁₀ cP', 'SurfTen': 'mN/m', 'RD': 'cm³/mol',
    # Pipetting protocol and label
    'volume': 'µL', 'pre_airgap': 'µL', 'post_airgap': 'µL',
    'asp_rate': 'µL/s', 'disp_rate': 'µL/s',
    'avg_temperature': '°C', 'avg_pressure': 'kPa', 'avg_humidity': '%RH',
    # avg_abs_error / error_stdev are dimensionless fractions -- '%' would be wrong
    # by a factor of 100, so they carry no unit string. Use TARGET_LABEL.
}

# The benchmarking runs changed their error definition mid-series and the two early
# runs were never regenerated. Verified elementwise against the raw JSON, 100%
# agreement per replicate per volume:
#
#   chloroform (2026-07-04), toluene (2026-07-06)
#       'errors' = calc_weight - volume * density        -> mass error in GRAMS
#   heptane, THF, acetonitrile, n-butanol, o-xylene (2026-07-14 onward)
#       'errors' = (calc_weight/density - volume)/volume -> RELATIVE, dimensionless
#
# Left alone this made chloroform and toluene look like by far the best-pipetted
# liquids in every figure and every fit -- an artefact of the unit, not a property
# of the liquid. ingest_ALL_data.relative_error_summary() therefore ignores the
# stored 'average error' and recomputes the relative error from the raw weights for
# every run, which reproduces the five already-relative runs to 1e-6.
TARGET_UNIT_WARNING: str = (
    'avg_abs_error is recomputed from raw weights as a dimensionless relative '
    'error; the stored per-run values mixed grams and fractions. See columns.py.'
)

# Spelled out because 'avg_abs_error' on an axis tells a reader neither what was
# measured nor on what scale. 'fraction of target volume' is the whole point: 0.24
# is a 24% error, not 0.24 uL.
TARGET_LABEL: str = 'mean absolute dispensing error (fraction of target volume)'


def unit(column: str) -> str:
    """Unit for a column, or '' when it is dimensionless or not established."""
    return UNITS.get(column, '')


def axis_label(column: str, suffix: str = '') -> str:
    """``'MWt (g/mol)'`` when a unit is known, else the bare column name.

    ``suffix`` appends a qualifier inside the bracket -- ``axis_label('volume',
    'log scale')`` gives ``'volume (µL, log scale)'`` -- so a caller never has to
    hand-assemble a unit string and risk it drifting from UNITS.
    """
    parts = [part for part in (UNITS.get(column, ''), suffix) if part]

    return f'{column} ({", ".join(parts)})' if parts else column


def screening_features(columns: tuple[str, ...] = NUM_COLUMNS,
                       drop_composition: bool = True) -> list[str]:
    """Descriptors that enter unsupervised screening.

    Composition columns are removed here rather than inside the variance screen so
    the audit trail can attribute them to their own stage.
    """
    if not drop_composition:
        return list(columns)

    return [column for column in columns if column not in COMP_COLUMNS]


def supervised_pool(columns: tuple[str, ...] = NUM_COLUMNS) -> list[str]:
    """Descriptors the supervised selectors may choose from.

    The same pool the unsupervised screens see -- ``screening_features()`` -- minus
    the three raw Antoine coefficients and plus the quantity they jointly imply.

    Two defects this exists to close, both of which made the cross-method comparison
    invalid rather than merely noisy:

    * The supervised track used to take all 58 of ``NUM_COLUMNS`` while the screens
      took ``screening_features()``, so ``Cl#``, ``N#`` and ``O#`` cleared the
      fold-frequency threshold in the MRMR variants -- descriptors structurally
      unable to appear in any A:: or B:: set, yet counted in one denominator with
      them.
    * ``AntA``, ``AntB`` and ``AntC`` are fitted jointly to one vapour-pressure curve
      (see ``DERIVED_VAPOUR_PRESSURE`` above), so selecting among them individually is
      physically incoherent -- and ``AntC`` had become one of the three descriptors in
      the supervised consensus set on that basis alone.

    The caller must have run ``screening.attach_derived`` on the frame first;
    ``DERIVED_VAPOUR_PRESSURE`` is not an HSPiP column.
    """
    kept = [c for c in screening_features(columns) if c not in RAW_ANTOINE]

    return kept + [DERIVED_VAPOUR_PRESSURE]
