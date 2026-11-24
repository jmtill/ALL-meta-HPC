"""What happens to each feature family after Elastic Net (plan, Phase 13).

Rule 3 of the plan: do not force every feature type through the same refinement. The
families arrive here having been through the same Elastic Net, and they leave having
been treated differently, because what "still redundant" means depends on what the
inputs are.

    raw (manual, mRMR)   recluster diagnostics -> residual redundancy -> optional
                         final reduction under FINAL_RECLUSTER_MODE
    global PCA           stop. The components are already orthogonal by construction;
                         there is nothing a clustering pass could find.
    cluster PC1          residual factor correlation only. No second PCA, ever.

The last two are prohibitions rather than steps, and prohibitions are the easiest thing
in a codebase to lose to a well-meaning generalisation -- a later refactor that notices
four families going through `recluster` and "tidies up" the fifth would violate both
without any test failing. So both are asserted:
``test_global_pcs_are_never_reclustered`` and
``test_cluster_factors_are_never_pca_transformed_again`` fail if the dispatch ever grows
that branch.

## Why reclustering raw features is legitimate and reclustering factors is not

A manual or mRMR subset is a set of raw descriptors chosen for individual usefulness;
nothing in either procedure removed the correlation *between* the survivors, and the
Phase 0 audit measured what that looks like -- mRMR's five-descriptor selection contained
four rotations of one volatility axis at |rho| >= 0.96. Asking whether the survivors form
correlation families is asking a real question about a set that was never checked for it.

A cluster-PC1 factor is already the answer to that question. Running PCA on a set of
PC1s produces a linear combination of linear combinations whose provenance is two levels
deep and whose loadings no longer point at anything a chemist can name, and it does so to
remove a redundancy that the clustering step was specifically built to remove. That is why
the plan forbids it, and the residual correlation is reported instead: if the factors are
still correlated, that is a finding about the clustering, not a problem to be PCA'd away.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

from allmeta.feature_selection import algorithms, config, factors, representations

# Which treatment each family gets, keyed by the provenance feature type. Declared as
# data rather than as a chain of ifs so that "what happens to a global PC?" is
# answerable by reading one dict.
TREATMENT: dict[str, str] = {
    'raw': 'recluster',
    'global_pca': 'stop',
    'cluster_pc1': 'residual_correlation',
    'manual_cluster_pc1': 'residual_correlation',
}

# Families whose names do not carry their type. Anything not listed is inferred from
# its name prefix by ``feature_type_of``.
KNOWN_TYPES: dict[str, str] = {
    'manual_raw': 'raw',
    'mrmr_raw': 'raw',
    'global_pca': 'global_pca',
    'manual_cluster_pc1': 'manual_cluster_pc1',
}


def feature_type_of(name: str) -> str:
    """The provenance feature type of a family, from its name.

    Unknown names raise rather than defaulting. A family that silently fell through to
    'raw' would be reclustered, and reclustering a set of factors is the one thing
    Rule 3 forbids outright -- a default here would make that failure quiet.
    """
    if name in KNOWN_TYPES:
        return KNOWN_TYPES[name]
    if name.startswith('auto_cluster_pc1'):
        return 'cluster_pc1'

    raise ValueError(
        f'unknown feature family {name!r}: its type cannot be inferred, so the '
        'Phase 13 treatment is undefined. Add it to KNOWN_TYPES rather than letting '
        'it default -- an unrecognised factor set treated as raw would be reclustered, '
        'which Rule 3 forbids.')


def residual_correlation(values: DataFrame, method: str = 'spearman') -> DataFrame:
    """Pairwise correlation among the retained inputs, long form.

    The shared diagnostic. For factor families it is the whole of Phase 13; for raw
    families it is what the reclustering decision is read against.
    """
    if values.shape[1] < 2:
        return DataFrame(columns=['feature_a', 'feature_b', 'abs_correlation'])

    rho = values.corr(method=method)
    rows = []
    columns = list(values.columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            rows.append({'feature_a': a, 'feature_b': b,
                         'correlation': float(rho.loc[a, b]),
                         'abs_correlation': float(abs(rho.loc[a, b]))})

    return DataFrame(rows).sort_values('abs_correlation', ascending=False,
                                       ignore_index=True)


def redundancy_summary(residual: DataFrame, name: str, n_inputs: int,
                       threshold: float = 0.90) -> dict:
    """How much correlation survived, as one record.

    ``max_abs_correlation`` is the number to read first: a family whose worst surviving
    pair still correlates above 0.95 has not been de-duplicated by anything upstream,
    whatever its mean says.
    """
    if residual.empty:
        return {'feature_set': name, 'n_inputs': n_inputs, 'n_pairs': 0,
                'max_abs_correlation': np.nan, 'mean_abs_correlation': np.nan,
                'n_pairs_above_threshold': 0, 'fraction_above_threshold': np.nan,
                'redundancy_threshold': threshold}

    above = int((residual['abs_correlation'] >= threshold).sum())

    return {
        'feature_set': name,
        'n_inputs': n_inputs,
        'n_pairs': int(len(residual)),
        'max_abs_correlation': float(residual['abs_correlation'].max()),
        'mean_abs_correlation': float(residual['abs_correlation'].mean()),
        'median_abs_correlation': float(residual['abs_correlation'].median()),
        'n_pairs_above_threshold': above,
        'fraction_above_threshold': above / len(residual),
        'redundancy_threshold': threshold,
    }


def recluster(fit_frame: DataFrame, retained: list[str],
              mode: str | None = None) -> tuple[list[str], DataFrame, DataFrame | None]:
    """Cluster the surviving raw features and optionally reduce them again.

    Returns ``(final_features, membership, loadings)``. ``loadings`` is None unless the
    mode is ``cluster_pc1``.

    The clustering is fitted on ``fit_frame`` -- the unlabelled subset -- for the same
    reason everything else in this pipeline is: it costs nothing in leakage terms and
    estimates the correlation structure from hundreds of liquids rather than twelve.
    The *input* to it is the fold's retained set, so the diagnostic is fold-local even
    though the correlation estimate is not.

    ``mode``:
        ``none``            report the clusters, change nothing. The default, and the
                            right one unless the redundancy summary says otherwise.
        ``representative``  keep one member per cluster (the medoid, with
                            ``columns.PROTECTED`` winning outright).
        ``cluster_pc1``     replace each multi-member cluster with its PC1.
    """
    mode = config.FINAL_RECLUSTER_MODE if mode is None else mode
    if mode not in ('none', 'representative', 'cluster_pc1'):
        raise ValueError(f'unknown FINAL_RECLUSTER_MODE {mode!r}; expected '
                         "'none', 'representative' or 'cluster_pc1'")

    usable = [f for f in retained if f in fit_frame.columns]
    if len(usable) < 2:
        return list(retained), DataFrame(), None

    rho, distance = representations.correlation_representation(fit_frame, usable)
    solution = algorithms.fit(
        'hierarchical', distance, linkage_method='average',
        distance_threshold=config.POST_ELASTIC_NET_DISTANCE_THRESHOLD)

    clusters = solution.clusters()
    membership = DataFrame([
        {'cluster_id': label, 'n_members': len(members),
         'features': ', '.join(members), 'is_singleton': len(members) == 1}
        for label, members in clusters.items()])
    membership['recluster_mode'] = mode
    membership['distance_threshold'] = config.POST_ELASTIC_NET_DISTANCE_THRESHOLD

    if mode == 'none':
        return list(retained), membership, None

    if mode == 'representative':
        from allmeta.screening import pick_representative
        keep = [pick_representative(members, rho) if len(members) > 1 else members[0]
                for members in clusters.values()]
        # Original order preserved: the manual list is ordered meaningfully and the
        # ordering survives into every output downstream.
        return [f for f in retained if f in set(keep)], membership, None

    factor_set = factors.build_factors(fit_frame, solution, rho)

    return list(factor_set.values.columns), membership, factor_set.loadings


def refine(name: str, values: DataFrame, retained: list[str],
           fit_frame: DataFrame | None = None, mode: str | None = None) -> dict:
    """Apply the Phase 13 treatment for one family and report what was done.

    ``retained`` is what Elastic Net kept. Rule 4 means it may be everything, and this
    function must not treat that as a failure or as a reason to reduce further.

    The returned record always carries ``treatment`` and ``final_features`` so a caller
    can see which branch ran without inferring it from the family name.
    """
    feature_type = feature_type_of(name)
    treatment = TREATMENT[feature_type]
    present = [f for f in retained if f in values.columns]
    subset = values[present] if present else values.iloc[:, :0]

    residual = residual_correlation(subset)
    record = {
        'feature_set': name,
        'feature_type': feature_type,
        'treatment': treatment,
        'n_retained': len(present),
        'retained_everything': len(present) == values.shape[1],
        **redundancy_summary(residual, name, len(present)),
        'residual': residual,
        'membership': DataFrame(),
        'loadings': None,
        'final_features': list(present),
    }

    if treatment != 'recluster':
        # 'stop' and 'residual_correlation' differ in intent but not in action: neither
        # touches the feature set. Kept as separate labels because the *reasons* differ
        # and a reader tracing why a family was left alone should see which applies.
        record['recluster_mode'] = 'n/a'
        return record

    if fit_frame is None:
        raise ValueError(f'{name!r} is a raw family and needs a fit_frame to recluster '
                         'against; pass the unlabelled fitting frame')

    final, membership, loadings = recluster(fit_frame, present, mode=mode)
    record.update({'final_features': final, 'membership': membership,
                   'loadings': loadings,
                   'recluster_mode': (config.FINAL_RECLUSTER_MODE if mode is None
                                      else mode),
                   'n_final': len(final)})

    return record


def refine_all(families: dict[str, DataFrame], retained_by_family: dict[str, list[str]],
               fit_frame: DataFrame | None = None, mode: str | None = None
               ) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Phase 13 for every family. Returns ``(diagnostics, membership, residual)``."""
    diagnostics: list[dict] = []
    memberships: list[DataFrame] = []
    residuals: list[DataFrame] = []

    for name, values in families.items():
        retained = retained_by_family.get(name)
        if retained is None:
            # No Elastic Net selection was made for this family -- on this data that
            # happens whenever no candidate beat the mean baseline. Recorded with its
            # treatment and reason rather than omitted: a family missing from the
            # diagnostics table is indistinguishable from one that was never run, and
            # this is the third time in this project that silently dropping "nothing
            # happened" has produced a table that reads as complete when it is not.
            feature_type = feature_type_of(name)
            diagnostics.append({
                'feature_set': name, 'feature_type': feature_type,
                'treatment': TREATMENT[feature_type], 'n_retained': 0,
                'retained_everything': False, 'n_inputs': int(values.shape[1]),
                'no_selection_made': True,
                'note': 'no Elastic Net candidate beat the mean baseline',
            })
            continue

        record = refine(name, values, retained, fit_frame=fit_frame, mode=mode)
        record['no_selection_made'] = False
        record['note'] = ''

        residual = record.pop('residual')
        if not residual.empty:
            residual = residual.assign(feature_set=name)
            residuals.append(residual)

        membership = record.pop('membership')
        if not membership.empty:
            memberships.append(membership.assign(feature_set=name))

        record.pop('loadings', None)
        record['final_features'] = ', '.join(record['final_features'])
        diagnostics.append(record)

    return (DataFrame(diagnostics),
            pd.concat(memberships, ignore_index=True) if memberships else DataFrame(),
            pd.concat(residuals, ignore_index=True) if residuals else DataFrame())
