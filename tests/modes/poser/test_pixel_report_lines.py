"""Poser's sheet-info pane: the pixel-report line under the selected sheet.

Troupe's own ``ui/panes/sheets.py``, folded into ``ui/panes/sheet.py`` by P9
(2026-09-18). The one claim under test is narrow on purpose:
``_pixel_report_lines`` is a pure function of the worker's report dict, so the
wording is checked without a window -- ``camera_line``'s own argument, applied
to its neighbour.
"""

from __future__ import annotations

from realmspinner.studio.modes.poser.ui.panes import sheet as poser_sheet


def test_an_hd_sheet_reports_full_colour_not_a_palette():
    """The D5 HD branch's report is ``{"style": "hd", "exact_stride": bool}``
    -- no ``colors``, no ``palette_name``, no ``orphans`` -- and the pixel-art
    wording (``"? colours ()"``) read literally over it rather than saying
    what actually happened: nothing was quantised.
    """
    report = {"style": "hd", "exact_stride": True}
    lines = poser_sheet._pixel_report_lines(report)
    assert lines == ["HD -- full colour"]
    assert not any("colours" in line for line in lines)
    assert not any("?" in line for line in lines)


def test_an_hd_sheet_with_an_inexact_stride_says_so():
    report = {"style": "hd", "exact_stride": False}
    lines = poser_sheet._pixel_report_lines(report)
    assert lines[0] == "HD -- full colour"
    assert len(lines) == 2
    assert "exact stride" in lines[1]


def test_a_pixel_art_sheets_report_is_unchanged():
    """The pixel-art wording, byte for byte, before and after the HD branch
    was added beside it."""
    report = {
        "style": "pixel_art",
        "colors": 16,
        "palette_name": "nes",
        "orphans": 3,
    }
    lines = poser_sheet._pixel_report_lines(report)
    assert lines == ["16 colours (nes)", "3 stray pixels cleaned"]


def test_a_pixel_art_report_with_no_style_key_still_reports():
    """A sheet built before the D5 switch existed has no ``style`` key at
    all, and must still read as pixel art rather than as nothing."""
    report = {"colors": 32, "palette": "", "orphans": 0}
    lines = poser_sheet._pixel_report_lines(report)
    assert lines == ["32 colours ()"]


def test_an_empty_report_draws_nothing():
    assert poser_sheet._pixel_report_lines({}) == []


# --- _front_helper (ui/panes/send.py) ------------------------------------
#
# The 2026-09-20 audit, finding troupe-04: a pure, branching string builder
# with no test of its own, unlike its directly-tested neighbours -- this
# file's own docstring calls it out as ``camera_line``'s own argument,
# applied to its neighbour. Both branches, here rather than in a dedicated
# ``send`` test module: this is the pair of test files the finding named.


def test_front_helper_with_no_front_set():
    from realmspinner.studio.modes.poser.ui.panes import send as poser_send

    assert poser_send._front_helper(0.0) == (
        "This mesh has no front set; sheets are rendered from yaw 0."
    )


def test_front_helper_with_a_front_set():
    from realmspinner.studio.modes.poser.ui.panes import send as poser_send

    assert poser_send._front_helper(90.0) == (
        "This mesh's front is set to 90 degrees; sheets are rendered from it."
    )
