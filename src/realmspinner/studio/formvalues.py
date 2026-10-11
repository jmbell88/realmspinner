"""Coerce a value into the type a form field already holds.

Split out of ``review_mode.py`` (2026-09-18 restructure, P5): Review's
``apply_vector`` and Create's "Use ..." findings-hint buttons
(``modes/create/engine/{recipe,mesh}.py``) both need this exact cast, and
Review and Create are sibling modes -- neither may import the other's module
-- so the one function they share moves to a stdlib-only shell module
alongside :mod:`realmspinner.studio.problems`, which left ``widgets.py`` for the
identical reason.
"""

from __future__ import annotations

from typing import Any


def coerce_form_value(default: Any, value: Any) -> Any:
    """``value`` cast to the type ``default`` already is.

    Used wherever a value from outside the form -- a findings bucket key, a
    sweep vector -- is about to be written into a form field: a bucket key
    such as ``"0.6"`` in ``findings.json`` is always a string, and the form
    field it is offered against may be a float, so the write has to land in
    the type the widget actually reads.
    """
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                # The 2026-10-10 audit found a bool control's "Use False" offer
                # did nothing: findings buckets are keyed ``str(value)``, and
                # ``bool("False")`` is True, so the suggestion wrote back the
                # value the control already held.
                word = value.strip().lower()
                if word in ("true", "1", "yes", "on"):
                    return True
                if word in ("false", "0", "no", "off", ""):
                    return False
                return default
            return bool(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, int):
            return int(value)
        if isinstance(default, str):
            return str(value)
    except (TypeError, ValueError):
        return default
    return value
