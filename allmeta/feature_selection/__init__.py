"""Leakage-safe feature selection: the five initial feature representations.

Phases 0-3 of ``feature_selection_implementation_plan.md``. What exists here now:

* ``config``           -- the one file a user edits; every grid and feature list.
* ``provenance``       -- which original descriptors produced each model input.
* ``manual``           -- the hand-picked raw subset.
* ``mrmr``             -- fold-safe supervised selection of a raw subset.
* ``pca``              -- global principal components, and PC1 per feature cluster.
* ``representations``  -- the two geometries feature clustering consumes.
* ``algorithms``       -- the five feature-clustering algorithms.
* ``metrics``          -- one scoring layer every solution goes through.
* ``factors``          -- PC1 per cluster, with retention and provenance.
* ``stability``        -- does a partition survive a change of liquids?
* ``plots``            -- the clustering figures.
* ``pipeline``         -- the driver that wires them together and writes the outputs.

The organising rule, and the reason this is a subpackage of ``allmeta`` rather than a
parallel package: anything learned from data is fitted on one frame and *applied* to
another through persisted constants, never refitted on the frame it is scoring. The
existing ``columns``, ``paths``, ``subsets``, ``screening`` and ``plotting`` modules are
reused rather than reimplemented.

Phases 4 to 7 fit every algorithm over its whole grid, score every solution, build its
cluster-PC1 factors, and measure how much each partition moves under resampling -- but
select nothing. That is deliberate: the plan is explicit that a winner must not be
chosen on cluster quality, because the clusters exist to produce useful factors rather
than tidy partitions, and the sweep here is the evidence that choice will be made from.

Not implemented yet (Phases 8+): hyperparameter optimisation against a predictive
criterion, method comparison and the one-standard-error rule, Elastic Net refinement
and its stability analysis, the nested-CV orchestrator, and the final consensus
reporting.
"""

from allmeta.feature_selection import (  # noqa: F401
    algorithms,
    config,
    factors,
    manual,
    metrics,
    mrmr,
    pca,
    provenance,
    representations,
    stability,
)
from allmeta.feature_selection.algorithms import ClusterSolution  # noqa: F401
from allmeta.feature_selection.provenance import (  # noqa: F401
    FeatureSpec,
    Provenance,
    raw_spec,
)

__all__ = [
    'algorithms', 'config', 'factors', 'manual', 'metrics', 'mrmr', 'pca',
    'provenance', 'representations', 'stability', 'ClusterSolution', 'FeatureSpec',
    'Provenance', 'raw_spec',
]
