"""Which original descriptor columns produced each model input.

Once a pipeline mixes raw descriptors, global principal components and per-cluster PC1
factors, a coefficient table stops being readable: ``global_PC3`` and ``factor_Tc``
name nothing a chemist can act on. Every transformation in this package therefore
emits a ``FeatureSpec`` alongside its values, and the registry here can answer, for any
model input, the only question that matters when a result is interpreted:

    which of the 58 HSPiP descriptors is this made of, and how?

The registry is deliberately dumb -- it stores what it is told and writes it out. It
does not recompute anything, so it cannot disagree with the transform that produced
the values.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pandas import DataFrame

# 'raw' passes an original column through unchanged. 'global_pca' is a component of a
# PCA over the whole descriptor set. 'cluster_pc1' is PC1 of an automatically
# discovered feature cluster; 'manual_cluster_pc1' is PC1 of a hand-defined one. The
# last two are separated because they answer different questions -- one is "what did
# the algorithm find", the other "was the expert's grouping any good" -- and a single
# label would make the comparison impossible to read back out of the tables.
FeatureType = Literal['raw', 'global_pca', 'cluster_pc1', 'manual_cluster_pc1']

PROVENANCE_COLUMNS: tuple[str, ...] = (
    'feature_name', 'feature_type', 'source_method', 'n_source_features',
    'source_features', 'clustering_method', 'cluster_id', 'cluster_name',
    'explained_variance_ratio',
)


@dataclass(frozen=True)
class FeatureSpec:
    """One model input and the original columns behind it.

    Frozen because a spec is a record of something that already happened. Mutating one
    after the fact would let the audit trail drift from the values it describes, which
    is the failure this class exists to prevent.
    """

    feature_name: str
    feature_type: FeatureType
    source_features: tuple[str, ...]
    source_method: str
    clustering_method: str | None = None
    cluster_id: int | None = None
    cluster_name: str | None = None
    # PC1's share of its cluster's variance, or a component's share of the total. None
    # for a raw passthrough, where the notion does not apply -- not 1.0, which would
    # read as "explains everything" rather than "not applicable".
    explained_variance_ratio: float | None = None

    def __post_init__(self) -> None:
        if not self.feature_name:
            raise ValueError('feature_name is empty')
        if not self.source_features:
            raise ValueError(
                f'{self.feature_name!r} lists no source features; every model input '
                'must trace back to at least one original column')

    def as_row(self) -> dict:
        """Flat record for the provenance table.

        ``source_features`` is joined rather than left as a tuple so the CSV stays one
        row per feature. A reader who needs the list back should split on ', '.
        """
        return {
            'feature_name': self.feature_name,
            'feature_type': self.feature_type,
            'source_method': self.source_method,
            'n_source_features': len(self.source_features),
            'source_features': ', '.join(self.source_features),
            'clustering_method': self.clustering_method,
            'cluster_id': self.cluster_id,
            'cluster_name': self.cluster_name,
            'explained_variance_ratio': self.explained_variance_ratio,
        }


@dataclass
class Provenance:
    """Registry of every model input a pipeline run produced."""

    specs: list[FeatureSpec] = field(default_factory=list)

    def add(self, *specs: FeatureSpec) -> 'Provenance':
        """Register one or more specs. Returns self so calls can chain."""
        for spec in specs:
            self.specs.append(spec)

        return self

    def extend(self, specs) -> 'Provenance':
        """Register an iterable of specs."""
        return self.add(*specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __contains__(self, feature_name: object) -> bool:
        return any(spec.feature_name == feature_name for spec in self.specs)

    def names(self) -> list[str]:
        """Registered feature names, in registration order."""
        return [spec.feature_name for spec in self.specs]

    def spec(self, feature_name: str) -> FeatureSpec:
        """The spec for one feature name."""
        for candidate in self.specs:
            if candidate.feature_name == feature_name:
                return candidate

        raise KeyError(f'{feature_name!r} is not registered')

    def sources_of(self, feature_name: str) -> tuple[str, ...]:
        """Original descriptor columns behind one model input.

        The question this module exists to answer. A raw feature returns itself.
        """
        return self.spec(feature_name).source_features

    def to_frame(self) -> DataFrame:
        """The registry as a table, one row per model input."""
        return DataFrame([spec.as_row() for spec in self.specs],
                         columns=list(PROVENANCE_COLUMNS))

    def write(self, path: Path) -> Path:
        """Write ``feature_provenance.csv`` and return the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)

        return path


def raw_spec(feature: str, source_method: str) -> FeatureSpec:
    """Spec for a descriptor passed through unchanged.

    Its own source, which is not a degenerate case but the point: a raw feature and a
    factor built from it must be distinguishable in the same table.
    """
    return FeatureSpec(
        feature_name=feature,
        feature_type='raw',
        source_features=(feature,),
        source_method=source_method,
    )
