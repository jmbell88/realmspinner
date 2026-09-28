"""Closes inker-mode findings from the 2026-09-26 audit, wave 3 fixer 4.

Each test names the claim it proves; see the audit's "Remaining findings"
section for the finding id cited in each fix's own comment.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest

from realmspinner.kernels import pixel as inker
from realmspinner.studio import state as state_mod
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker import ops as inker_ops
from realmspinner.studio.modes.inker import playback as inker_playback
from realmspinner.studio.modes.inker import state as inker_state
from realmspinner.studio.tasks import Done


def _session(frames=1):
    doc = inker.Document.blank(8, 8)
    if frames > 1:
        doc.ensure_animation()
        for _ in range(frames - 1):
            doc.add_frame()
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")
    state = inker_state.InkerState()
    state.add(tab)
    app = SimpleNamespace(inker=state, toasts=[])
    app.toast = MethodType(state_mod.AppState.toast, app)
    app.toast_once = MethodType(state_mod.AppState.toast_once, app)
    settings = SimpleNamespace(get=lambda key: {}, set=lambda key, value: None)
    ctx = SimpleNamespace(state=app, toast=app.toast, settings=settings)
    return ctx, state, tab


def _op(name):
    return next(op for op in inker_ops.OPS if op.name == name)


def _layers_session(count=2):
    """Like ``_session`` but with ``count`` plain layers, not frames."""

    doc = inker.Document.blank(8, 8)
    for i in range(1, count):
        doc.add_layer(f"L{i}")
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")
    state = inker_state.InkerState()
    state.add(tab)
    app = SimpleNamespace(inker=state, toasts=[])
    app.toast = MethodType(state_mod.AppState.toast, app)
    app.toast_once = MethodType(state_mod.AppState.toast_once, app)
    settings = SimpleNamespace(get=lambda key: {}, set=lambda key, value: None)
    ctx = SimpleNamespace(state=app, toast=app.toast, settings=settings)
    return ctx, state, tab


def test_first_and_last_frame_are_refused_while_playing_or_saving():
    """inker-mode-06: First/Last frame were gated on ``animated`` alone, so
    they moved the playhead mid-playback and mid-save (ops.py:1997-2018)."""

    ctx, _, tab = _session(3)
    tab.doc.set_current_frame(1)

    tab.playing = True
    assert inker_ops.run(ctx, _op("first_frame")) is False
    assert tab.doc.anim.current == 1

    tab.playing = False
    tab.saving = True
    assert inker_ops.run(ctx, _op("last_frame")) is False
    assert tab.doc.anim.current == 1


def test_play_is_refused_during_a_free_transform():
    """inker-mode-07: starting playback committed the floating transform
    buffer directly instead of through ``end_transform``, leaving
    ``state.transforming`` stuck True -- after which Enter, still claimed by
    the Transformation key context, could never reach ``toggle_play`` again
    to stop it (playback.py:28-61)."""

    ctx, state, tab = _session(2)
    state.transforming = True
    inker_playback.toggle_play(ctx, tab)
    assert tab.playing is False
    # Refused before it could touch the transform: still marked as in
    # progress, exactly as a real Free transform left it.
    assert state.transforming is True


def test_convert_to_background_refuses_a_locked_layer():
    """inker-mode-08: ``to_background`` folded the matte straight into the
    bottom layer's pixels and forced its alpha to 255 with no lock check at
    all (ops.py, the ``to_background`` registration)."""

    ctx, _, tab = _session()
    tab.doc.stack[0].locked = True
    before = tab.doc.stack[0].pixels.copy()
    assert inker_ops.run(ctx, _op("to_background")) is False
    assert (tab.doc.stack[0].pixels == before).all()
    assert tab.doc.stack[0].background is False
    assert not tab.doc.history.can_undo


def test_a_shortcut_override_with_an_infinite_priority_is_dropped():
    """inker-mode-11: ``_coerce_binding`` caught only ``TypeError``/
    ``ValueError`` around ``int(value.get("priority"))``; a hand-edited
    settings file's ``"priority": Infinity`` parses to ``float("inf")`` and
    ``int(inf)`` raises ``OverflowError`` instead (ops.py:280)."""

    assert (
        inker_ops._coerce_binding(
            {"chord": "P", "priority": float("inf")}, target_key="tool:brush"
        )
        is None
    )


def test_an_infinite_priority_override_does_not_stop_inker_building():
    """inker-mode-11, the second half: the same ``OverflowError`` used to
    escape ``ensure``'s own ``(TypeError, ValueError)`` catch too
    (mode.py:198-203), so one bad override stopped Inker opening at all."""

    from realmspinner.studio.modes.inker import mode as inker_mode

    class Settings:
        def __init__(self, block):
            self.values = {"inker": block}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

    changed = inker_ops.set_shortcuts({}, "tool", "brush", ["P"])
    changed["tool:brush"][0]["priority"] = float("inf")
    settings = Settings({"shortcuts": changed})
    ctx = SimpleNamespace(state=SimpleNamespace(inker=None), settings=settings)

    state = inker_mode.ensure(ctx)
    # Reaching this line at all is the fix: the bad entry is dropped (an
    # explicit "unbound" for that target, per ``_normalise_overrides``'s
    # docstring) rather than the whole shortcut file, or ``ensure`` itself,
    # raising.
    assert state.shortcut_overrides == {"tool:brush": []}


def test_export_slices_toasts_on_a_document_with_no_slices():
    """inker-mode-15: File > Export slices... (and Repeat last export, which
    calls this with ``repeat=True``) returned with nothing said at all on a
    document with no slices (ops.py:980-998, export.py's ``export_slices``)."""

    from realmspinner.studio.modes.inker import export as inker_export

    class _Ctx:
        def __init__(self, state):
            self.state = SimpleNamespace(inker=state)
            self.toasts: list[tuple[str, str]] = []

        def toast(self, message, kind="info", **_kw):
            self.toasts.append((message, kind))

    doc = inker.Document.blank(8, 8)
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="sprite.png")
    state = inker_state.InkerState()
    state.add(tab)
    ctx = _Ctx(state)

    inker_export.export_slices(ctx, tab)

    assert ctx.toasts == [("This document has no slices to export.", "warn")]


def test_merge_down_refuses_a_locked_layer():
    """inker-mode-16 (first of three): ``can_merge_down`` answered only the
    group question, so a locked layer on either side of the merge stayed
    enabled and would have been written through anyway (ops.py's
    ``can_merge_down``)."""

    ctx, state, tab = _layers_session(2)
    tab.doc.stack[0].locked = True  # the layer merge_down would write into
    # ``Document.merge_down`` itself already refused a locked participant, so
    # the pixels were always safe -- the bug is that the *menu row* stayed
    # enabled and clickable with nothing to show for the click: no toast, no
    # grey, no reason.
    op = _op("merge_down")
    assert op.enabled(state, tab) is False
    assert inker_ops.reason_for(op, state, tab) == (
        "One of these two layers is locked -- unlock it first."
    )
    before = tab.doc.stack[0].pixels.copy()
    assert inker_ops.run(ctx, op) is False
    assert (tab.doc.stack[0].pixels == before).all()
    assert len(tab.doc.stack) == 2


def test_merge_down_names_a_different_group_as_the_reason():
    """inker-mode-16 (second of three): a cross-group pair was greyed with
    "There is no layer under this one to merge into," which is false --
    there is one, just in another group."""

    ctx, state, tab = _layers_session(2)
    tab.doc.group_layers([1])
    op = _op("merge_down")
    assert op.enabled(state, tab) is False
    assert inker_ops.reason_for(op, state, tab) == (
        "The layer under this one is in a different group."
    )


def test_move_layer_up_is_refused_at_the_top():
    """inker-mode-16 (third of three): gated on ``many_layers`` alone, so a
    layer already at the top stayed enabled and ``move_layer`` silently
    no-opped (ops.py's ``layer_up`` registration)."""

    ctx, state, tab = _layers_session(2)
    tab.doc.set_active_layer(1)  # the top layer
    assert inker_ops.run(ctx, _op("layer_up")) is False
    assert inker_ops.reason_for(_op("layer_up"), state, tab) == "This is already the top layer."


class _BridgeCtx:
    """Enough of ``Ctx`` for the ``_done_*``/``import_tileset`` doors: the
    document lives on ``ctx.state.inker``, refusals go out through
    ``ctx.toast``, and ``submit`` is the queueing half."""

    def __init__(self, state):
        self.state = SimpleNamespace(inker=state)
        self.toasts: list[tuple[str, str]] = []
        self._busy: set[str] = set()

    def toast(self, message, level="info", **_kw):
        self.toasts.append((message, level))

    def submit(self, key, fn, *args, **kwargs):
        if key in self._busy:
            return False
        self._busy.add(key)
        return True


def test_a_picked_palette_is_not_applied_silently_while_the_tab_is_busy():
    """inker-mode-17 (1 of 3): ``palette_io.index_to`` returned False on
    ``tab.busy`` with nothing said -- so both ``_done_index`` and
    ``_done_palimg`` dropped a picked palette silently while a save or
    playback was running (palette_io.py:216-235)."""

    from realmspinner.studio.modes.inker import palette_io as inker_palette_io

    doc = inker.Document.blank(8, 8)
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")
    state = inker_state.InkerState()
    state.add(tab)
    ctx = _BridgeCtx(state)
    tab.saving = True

    assert inker_palette_io.index_to(ctx, tab, [(255, 0, 0, 255)]) is False
    assert ctx.toasts == [("Busy -- the picked colours were not applied. Try again.", "warn")]


def test_a_picked_tileset_is_not_dropped_silently_while_the_tab_is_busy():
    """inker-mode-17 (2 of 3): ``_done_tileset_import`` required
    ``not target.busy`` and fell through to nothing when it was, unlike its
    own comment's stated intent to answer (mode.py:1134-1150)."""

    doc = inker.Document.blank(8, 8)
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")
    state = inker_state.InkerState()
    state.add(tab)
    ctx = _BridgeCtx(state)
    tab.saving = True

    from realmspinner.kernels.grid2d import Tileset

    tileset = Tileset(name="picked", tile_w=8, tile_h=8, pixels=doc.stack[0].pixels)
    done = Done(key=f"inker-tileset-import:{tab.uid}", result={"tileset": tileset})
    inker_mode._done_tileset_import(ctx, state, done)

    assert len(tab.doc.tilesets) == 0
    assert ctx.toasts == [("Busy -- the picked tileset was not added. Try again.", "warn")]


def test_import_tileset_says_why_a_second_press_did_nothing():
    """inker-mode-17 (3 of 3): ``import_tileset`` discarded ``ctx.submit``'s
    refusal -- a second press while the first pick was still resolving opened
    no dialog and said nothing (mode.py's ``import_tileset``)."""

    doc = inker.Document.blank(8, 8)
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")
    state = inker_state.InkerState()
    state.add(tab)
    ctx = _BridgeCtx(state)
    ctx._busy.add(f"inker-tileset-import:{tab.uid}")

    inker_mode.import_tileset(ctx, tab)

    assert ctx.toasts == [("Already importing a tileset for this drawing.", "warn")]


def test_duplicate_document_warns_that_it_flattens():
    """inker-mode-18: Sprite > Duplicate document opens
    ``flatten(matte=False)``, unlike Aseprite's own layered *Duplicate
    Sprite* -- layers, frames and slices vanished into the copy with no
    warning anywhere in the menu (opening.py:303-317, ops.py's
    ``duplicate_sprite`` registration)."""

    op = _op("duplicate_sprite")
    assert "flatten" in op.hint.lower()
    assert "save as" in op.hint.lower()


@pytest.mark.parametrize(
    "name",
    [
        "select_layer_alpha",
        "select_colour_range",
        "feather",
        "grow",
        "shrink",
        "border",
    ],
)
def test_the_six_selection_rows_are_refused_while_saving(name):
    """inker-mode-20: six Select menu rows were gated on
    ``has_doc``/``has_selection`` alone, so all six restructured the
    selection or the document while a save was still encoding the layer
    stack (ops.py:2218-2375, before this fix)."""

    ctx, state, tab = _session()
    tab.doc.select_all()
    op = _op(name)
    tab.saving = True
    assert op.enabled(state, tab) is False
    assert inker_ops.reason_for(op, state, tab) == inker_ops.BUSY


def test_show_layer_is_refused_while_saving():
    """inker-mode-12: gated on ``not visible`` alone, so it pushed a history
    step -- ``set_layer_props`` is undoable -- while a save was still
    encoding the layer stack off-thread (ops.py, the ``show_layer``
    registration)."""

    ctx, _, tab = _session()
    tab.doc.set_layer_props(tab.doc.stack.active_index, visible=False)
    tab.doc.history.clear()
    tab.saving = True
    assert inker_ops.run(ctx, _op("show_layer")) is False
    assert tab.doc.stack.active.visible is False
    assert not tab.doc.history.can_undo
