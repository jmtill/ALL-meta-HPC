"""The hand-picked raw feature subset (implementation plan, Step 2.1).

The simplest of the five feature representations, and the control for the other four:
a list a person wrote down. If a search over 45 descriptors cannot beat it, the search
bought nothing but the opportunity to overfit 13 liquids.

There is no fitting here and nothing is learned from data, so this selector is trivially
leakage-safe and can be applied to any frame. All the work is validation, because the
failure mode is a typo'd descriptor name surfacing as a KeyError several stages later.
"""

from pandas import DataFrame

from allmeta import columns as cols
from allmeta.feature_selection import config
from allmeta.feature_selection.provenance import FeatureSpec, raw_spec

SOURCE_METHOD: str = 'manual'


def validate_manual(frame: DataFrame, features: list[str] | None = None) -> list[str]:
    """Check a manual feature list against a frame, and return it unchanged.

    Separate from ``select_manual`` so a caller can validate a candidate list against
    the data before committing to it -- which is what the config's own ``validate()``
    cannot do, since it only knows the descriptor vocabulary and not which columns a
    particular subset actually carries.
    """
    features = list(config.MANUAL_FEATURES if features is None else features)

    if not features:
        raise ValueError('manual feature list is empty; nothing to select')

    duplicated = sorted({f for f in features if features.count(f) > 1})
    if duplicated:
        raise ValueError(f'manual feature list repeats {duplicated}')

    if cols.TARGET in features:
        raise ValueError(
            f'manual feature list contains the target {cols.TARGET!r}; a model given '
            'its own target scores perfectly and means nothing')

    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise ValueError(
            f'manual feature list names columns absent from the frame: {missing}')

    return features


def select_manual(frame: DataFrame, features: list[str] | None = None
                  ) -> tuple[DataFrame, list[FeatureSpec]]:
    """The manual subset of ``frame``, with one provenance spec per feature.

    Column order follows the configured list rather than the frame, so a deliberate
    ordering -- most-trusted descriptor first, say -- survives into every downstream
    table and figure.
    """
    features = validate_manual(frame, features)
    specs = [raw_spec(feature, SOURCE_METHOD) for feature in features]

    return frame[features].copy(), specs
