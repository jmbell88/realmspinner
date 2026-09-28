"""The 2026-09-26 audit, findings shell-boot-03 and shell-boot-04: two ways a
stored settings file that does not match the shapes the app itself ever
writes could wreck Library restore on the frame thread instead of falling
back.
"""

from __future__ import annotations


def test_a_filter_with_an_unrecognized_value_falls_back_to_the_default():
    """shell-boot-03: ``filters_from_stored`` checked a restored value's
    *type* but never that it was actually one of the option table's values --
    ``{"status": "bogus"}`` is a ``str`` exactly like every real status, so it
    restored unchanged and the Library then opened with every job filtered
    out and nothing said about why."""
    from realmspinner.studio.state import Filters, filters_from_stored

    blank = Filters()
    got = filters_from_stored({"status": "bogus", "kind": "nonesuch", "sort": "phase-of-the-moon"})
    assert got.status == blank.status
    assert got.kind == blank.kind
    assert got.sort == blank.sort

    # A real option from each table still restores.
    real = filters_from_stored({"status": "error", "kind": "model", "sort": "grade"})
    assert real.status == "error"
    assert real.kind == "model"
    assert real.sort == "grade"


def test_a_non_numeric_stored_variant_does_not_crash_restore():
    """shell-boot-04: ``_restore_sheet_block`` -- unlike the coercion loop
    right beside its call site, which wraps every ``int``/``float`` cast in
    ``try/except (TypeError, ValueError)`` -- cast a stored ``variant``/
    ``variants`` with a bare ``int(...)``. A non-numeric value raised
    straight out of ``form_from_params``, on the frame thread, instead of
    being caught and defaulted the way its neighbor already is."""
    from realmspinner.studio.state import form_from_params

    params = {
        "sheet": {
            "mode": "materials",
            "materials": [{"prompt": "wood", "variant": "not-a-number"}],
            "variants": "also-not-a-number",
        }
    }

    form = form_from_params(params)  # must not raise
    assert form["tile_mode"] == "materials"
    # A variant that will not coerce is treated as variant 1 (the default),
    # so its prompt still surfaces rather than vanishing silently.
    assert form["materials"] == "wood"
    assert form["variants"] == "1"
