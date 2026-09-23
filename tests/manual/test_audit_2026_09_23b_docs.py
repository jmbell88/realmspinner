"""Documentation findings from the 2026-09-23 (second run) docs audit,
pinned against the tree.

Each test reads its truth from the module that makes the claim true or
false rather than repeating a second hand-written copy of a label or count,
the same shape as ``test_manual_labels.py`` and the prose-drift files.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "docs" / "manual"


def _chapter(name: str) -> str:
    return (MANUAL / f"{name}.md").read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """Prose with hand-wrapped line breaks collapsed to single spaces, so a
    phrase that happens to straddle two lines in the manual's own wrapping
    still matches a plain substring check."""
    return re.sub(r"\s+", " ", text)


# --- docs-02: manual 37's sweep axis count and the missing axis -----------


def test_review_manual_sweep_axis_count_matches_kwarg_axes():
    """The 2026-09-23b audit, finding docs-02: chapter 37 said "There are
    eighteen" sweep axes and never named ``lowpoly_triangles`` (label
    "Game-ready triangles", ``recipe.py``), which ``ea7ba4b6`` added to
    ``KWARG_AXES``. The count is read off ``KWARG_AXES`` itself rather than
    hand-typed a second time, and every axis's label must appear somewhere
    in the chapter's "What you can vary" prose."""
    from realmspinner.service.sweeps import KWARG_AXES
    from realmspinner.studio.modes.create.engine.recipe import FIELD_LABELS

    assert FIELD_LABELS["lowpoly_triangles"] == "Game-ready triangles"

    words = {17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"}
    counted = words[len(KWARG_AXES)]

    text = _flat(_chapter("37-review"))
    assert f"There are {counted}" in text, (
        f"docs/manual/37-review.md disagrees with KWARG_AXES, which has "
        f"{len(KWARG_AXES)} axes"
    )
    if counted != "eighteen":
        assert "There are eighteen" not in text, (
            "docs/manual/37-review.md still says 'eighteen' sweep axes, "
            "stale against KWARG_AXES"
        )

    # One recognisable phrase per axis, as the chapter's own prose names it.
    axis_phrases = {
        "lora_weight": "Style strength",
        "profile": "Profile",
        "custom_triangles": "Custom triangles",
        "lowpoly_triangles": "Game-ready triangles",
        "negative_prompt": "Negative prompt",
        "reference_prep": "Reference prep",
        "resolution": "Resolution",
        "ip_scale": "IP-Adapter scale",
        "control_scale": "ControlNet scale",
        "control_end": "ControlNet end",
        "bg_removal": "Background removal",
        "size_m": "Size in metres",
        "trellis_band": "band width",
        "trellis_tex_res": "texture resolution",
        "trellis_gss": "guidance strengths",
        "trellis_gsh": "guidance strengths",
        "trellis_max_tokens": "token budget",
        "trellis_decim": "decimation",
        "trellis_atlas": "atlas size",
    }
    assert set(axis_phrases) == set(KWARG_AXES), (
        "this test's axis_phrases map has drifted from sweeps.KWARG_AXES -- "
        "update both together"
    )
    for axis, phrase in axis_phrases.items():
        assert phrase in text, (
            f"docs/manual/37-review.md never names axis {axis!r} (expected "
            f"the phrase {phrase!r} in the 'What you can vary' prose)"
        )


# --- docs-03: manual 28's transform field row ------------------------------


def test_inker_manual_transform_fields_match_the_canvas_toolbar_labels():
    """The 2026-09-23b audit, finding docs-03: chapter 28 said the *tool
    options panel* draws X/Y/W/H/Angle/Slant fields. The real row is drawn
    by ``canvas.py``'s ``_transform_trailing``, on the canvas toolbar: no W
    field, X/Y are the per-axis scale multipliers, H/V are the shear (not
    "Slant"), and there is a Link checkbox the old sentence never mentioned.
    """
    canvas = (
        ROOT
        / "src"
        / "realmspinner"
        / "studio"
        / "modes"
        / "inker"
        / "ui"
        / "panes"
        / "canvas.py"
    ).read_text(encoding="utf-8")
    assert 'slider_float("Angle"' in canvas
    assert 'slider_float("X##inkscalex"' in canvas
    assert 'slider_float("Y##inkscaley"' in canvas
    assert 'checkbox("Link##inkscalelink"' in canvas
    assert 'slider_float("H##inkshearx"' in canvas
    assert 'slider_float("V##inksheary"' in canvas
    assert 'slider_float("W' not in canvas, (
        "canvas.py grew a W field on the transform row -- the manual's "
        "W-less description needs revisiting"
    )

    text = _flat(_chapter("28-inker"))
    assert "canvas toolbar adds typed **Angle**, **X**, **Y**" in text
    assert "**Link** checkbox" in text
    assert "**H** and **V**" in text
    assert "typed **X**, **Y**, **W**, **H**, **Angle** and **Slant**" not in text, (
        "docs/manual/28-inker.md still describes the stale tool-options-panel "
        "X/Y/W/H/Angle/Slant row"
    )


# --- docs-04: manual 07's shade labels -------------------------------------


def test_modelling_manual_names_shade_flat_and_shade_auto_by_their_full_labels():
    """The 2026-09-23b audit, finding docs-04: chapter 7 called two of
    Clay's three shading ops "Flat" and "Auto"; ``ops.py`` registers them as
    "Shade Flat" and "Shade Auto..." (``_shading`` rows, line ~3745)."""
    ops = (
        ROOT / "src" / "realmspinner" / "studio" / "modes" / "clay" / "ops.py"
    ).read_text(encoding="utf-8")
    assert '"Shade Smooth"' in ops
    assert '"Shade Flat"' in ops
    assert '"Shade Auto' in ops

    text = _chapter("07-modelling")
    assert "**Shade Smooth**, **Shade Flat** and **Shade Auto**" in text
    assert "**Shade Smooth**, **Flat** and **Auto**" not in text


# --- docs-05: manual 28's zoom-reset op label -------------------------------


def test_inker_manual_fit_command_matches_its_op_label():
    """The 2026-09-23b audit, finding docs-05: chapter 28's zooming section
    called the fit-to-window command "Fit view" in one spot while the rest
    of the chapter (and the op registry, ``fit_view`` -> "Fit in window")
    correctly say "Fit in window"."""
    ops = (
        ROOT / "src" / "realmspinner" / "studio" / "modes" / "inker" / "ops.py"
    ).read_text(encoding="utf-8")
    assert '"fit_view"' in ops
    assert '"Fit in window"' in ops

    text = _chapter("28-inker")
    assert "come out of a **Fit in window**" in text
    assert "come out of a **Fit view**" not in text


# --- docs-06: manual 32's snap pills location -------------------------------


def test_plotter_manual_places_snap_pills_beside_the_view_popover():
    """The 2026-09-23b audit, finding docs-06: chapter 32 said "Snap objects
    to" lives "in the View block". ``tools.py``'s ``_trailing`` draws the
    snap pills (``_snap_choice``) beside the **View** popover button on the
    toolbar, not inside the popover itself -- its own docstring says the
    View block is what the bar collapses *to* when the snap pills are given
    up for space."""
    tools = (
        ROOT
        / "src"
        / "realmspinner"
        / "studio"
        / "modes"
        / "plotter"
        / "ui"
        / "panes"
        / "tools.py"
    ).read_text(encoding="utf-8")
    assert "_snap_choice(state)" in tools
    assert 'SNAP_LABELS = (("off", "Off"), ("grid", "Grid"), ("pixel", "Pixel"))' in tools

    text = _flat(_chapter("32-plotter"))
    assert "beside the **View** popover button" in text
    assert "**Snap objects to** in the View block" not in text


# --- docs-07: manual 29's mirror-run button label ---------------------------


def test_inker_animation_manual_apply_whole_run_label_names_its_direction():
    """The 2026-09-23b audit, finding docs-07: chapter 29 called the mirror
    button "Apply whole run"; ``sheet.py`` always draws it as
    ``f"Apply whole {run.direction} run"`` -- the direction is always
    named on screen."""
    sheet = (
        ROOT
        / "src"
        / "realmspinner"
        / "studio"
        / "modes"
        / "inker"
        / "ui"
        / "panes"
        / "sheet.py"
    ).read_text(encoding="utf-8")
    assert 'f"Apply whole {run.direction} run"' in sheet

    text = _chapter("29-inker-animation")
    assert "**Apply whole *{direction}* run**" in text
    assert "**Apply whole run**" not in text


# --- familiar-04: manual 20's navigable-destinations list -------------------


def test_manual_overview_lists_show_trash_among_familiars_navigable_destinations():
    """The 2026-09-23 audit, finding familiar-04: ``doors._NAV_COMMAND_KEYS``
    treats "show-trash" (palette label "Show the trash") as a place Familiar
    can send the user, but chapter 20's list of what "take me there" can
    reach never mentioned the trash."""
    doors = (
        ROOT / "src" / "realmspinner" / "studio" / "assistant" / "doors.py"
    ).read_text(encoding="utf-8")
    assert '"show-trash"' in doors

    text = _flat(_chapter("20-overview"))
    assert "the workspace layout picker or the trash, whichever you asked for" in text
