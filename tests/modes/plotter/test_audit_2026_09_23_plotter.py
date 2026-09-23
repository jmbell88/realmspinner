"""Regressions for the 2026-09-23 audit's Plotter findings.

Four independent defects, four independent tests -- see each test's docstring
for the finding it closes.
"""

from __future__ import annotations

import io
import zipfile

from realmspinner.studio.modes.plotter import mode as plotter_mode
from realmspinner.studio.modes.plotter.engine import rmap as rmaplib
from realmspinner.studio.modes.plotter.engine.tilemap import MapObject, new_uid
from realmspinner.studio.modes.plotter.ui.panes.layers import _shift_reason


def _rmap_bytes_with_deep_property(depth: int) -> bytes:
    """A ``.rmap`` whose one property nests ``depth`` levels of ``list``."""
    prefix = '{"type":"list","value":[' * depth
    core = '{"type":"int","value":1}'
    suffix = "]}" * depth
    prop = prefix + core + suffix
    text = (
        '{"version":11,"width":1,"height":1,"tile_w":16,"tile_h":16,'
        '"layers":[],"tilesets":[],'
        '"properties":{"deep":' + prop + "}}"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("map.json", text)
    return buf.getvalue()


def test_read_rmap_refuses_deeply_nested_properties_as_a_value_error():
    """finding plotter-01.

    ``props.py``'s codec recursed one Python call frame per nesting level with
    no cap, so a hand-crafted (or merely malicious) ``.rmap`` a few thousand
    levels deep raised a bare ``RecursionError`` out of ``read_rmap`` --
    unwinding the interpreter's own stack instead of being refused like every
    other malformed file. Library's "Edit in Plotter" door caught only
    ``ValueError``, so that door crashed the window rather than toasting.
    """
    data = _rmap_bytes_with_deep_property(1000)
    try:
        rmaplib.read_rmap(data)
    except RecursionError as exc:  # pragma: no cover -- the unfixed behaviour
        raise AssertionError(
            f"read_rmap let a RecursionError escape instead of refusing: {exc!r}"
        ) from None
    except ValueError:
        pass
    else:
        raise AssertionError("a 1000-deep property tree should have been refused")


def test_a_history_jump_keeps_the_selection_of_an_object_that_survives_the_jump(
    plotter_ctx,
):
    """finding plotter-02.

    ``step_history`` cleared the whole object selection outright, unlike
    ``undo``/``redo`` on the same stack, which only prune the uids a step
    actually removed -- and ``step_history``'s own docstring says all three
    match. A jump to a position where the selected object still exists should
    leave it selected, exactly as a single-step undo/redo would.
    """
    ctx, state = plotter_ctx
    tab = plotter_mode.active(ctx)
    doc = tab.doc

    objects = doc.add_object_layer()
    obj = doc.add_object(objects.uid, MapObject(uid=new_uid(), name="a", kind="point"))
    # ``step_history``'s index is a *position* (the done-count ``step_to``
    # reads), never ``history.head`` -- that property is a global monotonic
    # serial (``undo.py``'s ``_serials``, shared across every open document in
    # the process), so it only happens to equal a position in a fresh process
    # with nothing else pushed first.
    position_with_object = len(doc.history)
    doc.set_object(objects.uid, obj.uid, x=5.0)

    state.select_object(obj.uid)
    assert state.selected_objects == {obj.uid}

    moved = plotter_mode.step_history(ctx, tab, position_with_object)
    assert moved is True
    assert state.selected_objects == {obj.uid}, (
        "the object is still on the layer at this point in history; "
        "the jump should not have dropped its selection"
    )


def test_raise_lower_buttons_say_select_a_layer_first_when_none_is_selected():
    """finding plotter-03.

    With no layer selected, ``can_shift_layer`` is already ``False`` (its own
    ``uid is None`` guard), so the button was always disabled correctly -- but
    the reason string named the wrong cause: "already at the end of its
    group" describes a *selected* layer with nowhere left to move, not the
    absence of a selection.
    """
    assert _shift_reason(True, None) == "Select a layer first."
    assert _shift_reason(True, 42) == "This layer is already at the end of its group."
