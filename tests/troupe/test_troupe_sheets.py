"""Troupe's right-top pane: the pixel-report line under the selected sheet.

The one claim under test is narrow on purpose: ``_pixel_report_lines`` is a
pure function of the worker's report dict, so the wording is checked without a
window -- ``camera_line``'s own argument, applied to its neighbour.
"""

from __future__ import annotations

from warlock.studio.panes import troupe_sheets


def test_an_hd_sheet_reports_full_colour_not_a_palette():
    """The D5 HD branch's report is ``{"style": "hd", "exact_stride": bool}``
    -- no ``colors``, no ``palette_name``, no ``orphans`` -- and the pixel-art
    wording (``"? colours ()"``) read literally over it rather than saying
    what actually happened: nothing was quantised.
    """
    report = {"style": "hd", "exact_stride": True}
    lines = troupe_sheets._pixel_report_lines(report)
    assert lines == ["HD -- full colour"]
    assert not any("colours" in line for line in lines)
    assert not any("?" in line for line in lines)


def test_an_hd_sheet_with_an_inexact_stride_says_so():
    report = {"style": "hd", "exact_stride": False}
    lines = troupe_sheets._pixel_report_lines(report)
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
    lines = troupe_sheets._pixel_report_lines(report)
    assert lines == ["16 colours (nes)", "3 stray pixels cleaned"]


def test_a_pixel_art_report_with_no_style_key_still_reports():
    """A sheet built before the D5 switch existed has no ``style`` key at
    all, and must still read as pixel art rather than as nothing."""
    report = {"colors": 32, "palette": "", "orphans": 0}
    lines = troupe_sheets._pixel_report_lines(report)
    assert lines == ["32 colours ()"]


def test_an_empty_report_draws_nothing():
    assert troupe_sheets._pixel_report_lines({}) == []
