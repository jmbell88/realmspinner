"""The Map properties popup, and the one thing it forgot from its neighbours.

``plotter_tools.map_settings_popup`` sits beside the resize, offset and
tile-size forms, which all stage an edit behind an explicit Apply button for
the reason ``_resize_form``'s comment gives: a text field is re-read on every
frame, so an unstaged write pushes an undo step a frame. This popup's own
Class/Background/Parallax-origin/Skew/Hex-side fields wrote straight through
``doc.set_map_settings`` -- an unconditional ``history.push`` -- with no such
staging and no ``fold_undo`` either, so typing into any of them pushed one
step per keystroke.

No frame is driven here: like ``tests/test_undo_gesture_doors.py``'s popup-door
checks, the assertion is positional and read from source, because
``map_settings_popup`` draws only inside ``imgui.begin_popup``, which a
headless test cannot open.
"""

from __future__ import annotations

import inspect

# --- one gesture, one undo step (the 2026-09-07 audit, plotter-03) ----------


def test_map_settings_popup_field_typing_is_one_undo_step():
    """Each check is bounded to the gap between one field and the next thing
    the popup draws, rather than "a fold exists somewhere before the write" --
    which a neighbouring field's own fold would satisfy for free and prove
    nothing about the field actually being checked."""
    from warlock.studio.panes import plotter_tools

    popup = inspect.getsource(plotter_tools.map_settings_popup)

    after_class = popup.split('"##map-class"', 1)[1]
    before_background = after_class.split('"##map-background"', 1)[0]
    assert "controls.fold_undo(" in before_background, (
        "Class field is not folded before the Background field is drawn"
    )

    after_background = popup.split('"##map-background"', 1)[1]
    before_render_order = after_background.split("Render order", 1)[0]
    assert "controls.fold_undo(" in before_render_order, (
        "Background field writes before it is folded"
    )

    after_origin = popup.split('"Parallax origin"', 1)[1]
    before_skew = after_origin.split('"Skew"', 1)[0]
    assert "controls.fold_undo(" in before_skew, (
        "Parallax origin field writes before it is folded"
    )

    after_skew = popup.split('"Skew"', 1)[1]
    before_stagger = after_skew.split("Stagger axis", 1)[0]
    assert "controls.fold_undo(" in before_stagger, (
        "Skew field writes before it is folded"
    )

    after_hex_side = popup.split('"Hex side"', 1)[1]
    before_help = after_hex_side.split("widgets.help_marker", 1)[0]
    assert "controls.fold_undo(" in before_help, (
        "Hex side field writes before it is folded"
    )
