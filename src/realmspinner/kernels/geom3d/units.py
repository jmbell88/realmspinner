"""Length units, for the numbers a person types and reads.

Storage is metres everywhere -- a document, a glTF, a scene -- and nothing here
changes that: a unit is only ever the lens a *field* is drawn through
(``to_display``/``from_display``) or the multiplier an *import* applies to
foreign numbers. One table serves both, so "cm" cannot mean 0.01 in the
Properties panel and something else in the import dialog.

``LENGTH_UNITS`` is ordered as the combos list it, metres first.
"""

from __future__ import annotations

from collections.abc import Sequence

#: ``(key, metres per unit)``. The key is also the label -- these are symbols,
#: not words. Multipliers are exact decimal fractions of a metre (an inch is
#: *defined* as 0.0254 m, a foot as 0.3048 m).
LENGTH_UNITS: tuple[tuple[str, float], ...] = (
    ("m", 1.0),
    ("cm", 0.01),
    ("mm", 0.001),
    ("in", 0.0254),
    ("ft", 0.3048),
)

DEFAULT_UNIT = "m"

_METRES_PER = dict(LENGTH_UNITS)


def metres_per(unit: str) -> float:
    """Metres in one *unit*; an unknown key reads as metres rather than raising.

    A stale key (a config written by a build with a longer table) must not take
    a panel down every frame for the sake of a display preference.
    """
    return _METRES_PER.get(unit, 1.0)


def to_display(metres: float, unit: str) -> float:
    return float(metres) / metres_per(unit)


def from_display(value: float, unit: str) -> float:
    return float(value) * metres_per(unit)


def vec_to_display(metres: Sequence[float], unit: str) -> list[float]:
    return [to_display(v, unit) for v in metres]


def scale_options() -> tuple[tuple[str, str], ...]:
    """``(f"{metres:g}", unit)`` rows, the shape the import-scale combo wants.

    Keyed on the multiplier's own ``:g`` spelling because that is what
    ``ClayState.import_scale`` is looked up by -- see ``IMPORT_SCALE_OPTIONS``.
    """
    return tuple((f"{metres:g}", key) for key, metres in LENGTH_UNITS)
