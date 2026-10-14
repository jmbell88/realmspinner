"""The 2026-10-14 audit, fixer F31: Inker's mode door, op gates, bridge, canvas
readout, colour panel and context bar.

Findings inker-mode-01 (the history popover under a float), inker-flourish-09
(canvas-fit scales angles and brightness), inker-mode-03 / -04 / -06 (the crash
copy's settle, thread and refusal), inker-mode-10 (a transform greyed as
"busy"), inker-mode-11 (a refused OS paste falls through), inker-panes-04 (a
regeneration over a flipped canvas), -02 (the seam readout's axes), -01 (the
colour list imgui returns) and -05 (a shade direction stored as a string).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels import pixel as inker
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker import ops as inker_ops
from realmspinner.studio.modes.inker import state as inker_state

RED = (200, 20, 20, 255)


def _session():
    state = inker_state.InkerState()
    doc = inker.Document.blank(8, 8)
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="T")
    state.add(tab)
    toasts: list[tuple] = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=state, preview={}),
        toast=lambda *a, **k: toasts.append(a),
        svc=None,
    )
    ctx.toasts = toasts
    return ctx, state, tab


# --- inker-mode-01 ------------------------------------------------------------


def test_step_history_cancels_a_floating_buffer_before_it_moves_the_head():
    ctx, state, tab = _session()
    doc = tab.doc
    doc.stack.active.pixels[0:4, 0:4] = RED
    doc.invalidate_all()
    doc.select_all()
    doc.copy()
    doc.add_layer()  # step 1
    assert doc.paste()  # a float is up; nothing committed
    assert doc.floating is not None

    assert inker_mode.step_history(ctx, tab, 0) is True

    assert doc.floating is None, "the float outlived the layer it belonged to"
    doc.commit_floating()  # every save / export door calls this first; it raised KeyError


def test_step_history_lands_where_the_popover_said_when_a_lift_is_cancelled():
    """Cancelling a lift revokes the lift's own step, which moves the popover's
    done-count by one; jumping to a step *before* it must still land there."""
    ctx, state, tab = _session()
    doc = tab.doc
    doc.stack.active.pixels[0:4, 0:4] = RED
    doc.invalidate_all()
    doc.add_layer()  # step 1: something to jump back to
    doc.stack.active_index = 0
    doc.select_rect(0, 0, 4, 4) if hasattr(doc, "select_rect") else doc.select_all()
    assert doc.lift()
    assert doc.floating is not None and doc.floating.lift_edit is not None
    steps = len(doc.history._done)

    assert inker_mode.step_history(ctx, tab, 1) is True

    assert doc.floating is None
    assert len(doc.history._done) == 1, f"asked for 1 done step, held {steps} before"
    assert len(doc.stack) == 2


# --- inker-flourish-09 --------------------------------------------------------


def test_fitting_a_preset_to_a_smaller_canvas_does_not_scale_angles_or_brightness():
    from realmspinner.kernels.pixel.flourish import presets, prims

    recipe = presets.load("fireball")
    scaled = inker_mode._scaled(recipe, 0.5)

    saw_px = saw_other = False
    for old, new in zip(recipe.layers, scaled.layers, strict=True):
        for name, spec in prims.params_of(old.kind).items():
            if name not in old.params or spec.kind not in ("float", "curve"):
                continue
            if spec.label.startswith("px"):
                if old.params[name] != new.params[name]:
                    saw_px = True
                continue
            if not spec.label and name in ("radius", "thickness"):
                continue  # unlabelled pixel lengths, scaled by name
            if name in ("rise", "strength", "speed"):
                saw_other = True
                assert new.params[name] == old.params[name], (
                    f"{old.kind}.{name} ({spec.label or 'unlabelled'}) was scaled"
                )
    assert saw_px, "the pixel lengths should still scale with the canvas"
    assert saw_other, "the fireball preset no longer holds an angle or a brightness to pin"


# --- inker-mode-03 ------------------------------------------------------------


def test_the_journal_takes_no_copy_while_a_buffer_floats():
    ctx, state, tab = _session()
    doc = tab.doc
    doc.stack.active.pixels[...] = RED
    doc.invalidate_all()
    doc.add_layer()
    doc.stack.active_index = 0
    assert tab.dirty
    assert tab in inker_mode.JOURNAL.slots(ctx)

    doc.select_all()
    assert doc.begin_transform(), "transform did not start"
    assert doc.floating is not None
    assert tab not in inker_mode.JOURNAL.slots(ctx), (
        "a copy now would lack the pixels in the float (0 opaque of 64 in the probe)"
    )

    doc.commit_floating()
    assert tab in inker_mode.JOURNAL.slots(ctx)


def test_the_journal_takes_no_copy_while_a_filter_preview_is_open():
    ctx, state, tab = _session()
    doc = tab.doc
    doc.stack.active.pixels[...] = RED
    doc.invalidate_all()
    doc.add_layer()
    doc.stack.active_index = 0
    state.filter_uid = tab.uid
    assert tab not in inker_mode.JOURNAL.slots(ctx)
    state.filter_uid = ""
    assert tab in inker_mode.JOURNAL.slots(ctx)


# --- inker-mode-04 ------------------------------------------------------------


def test_the_inker_journal_encode_defers_its_zlib_work_to_the_task(monkeypatch, tmp_path):
    from realmspinner.kernels.pixel import ora

    ctx, state, tab = _session()
    doc = tab.doc
    doc.stack.active.pixels[...] = RED
    doc.invalidate_all()

    calls: list[int] = []
    real = ora.ora_bytes
    monkeypatch.setattr(ora, "ora_bytes", lambda d: calls.append(1) or real(d))

    deferred = inker_mode._journal_encode(tab)

    assert callable(deferred), "the encode returned finished bytes: it ran on the frame thread"
    assert calls == [], "ora_bytes ran inside the frame-thread half"

    # The user keeps painting between the snapshot and the task: the copy is
    # the document as it stood when the journal looked.
    doc.stack.active.pixels[...] = (1, 2, 3, 255)
    doc.invalidate_all()
    data = deferred()
    assert calls == [1]
    path = tmp_path / "copy.ora"
    path.write_bytes(data)
    back = inker.Document.load(path)
    assert tuple(back.flatten(matte=False)[0, 0]) == RED


# --- inker-mode-06 ------------------------------------------------------------


def test_saving_an_over_budget_drawing_says_which_budget_it_broke(monkeypatch, tmp_path):
    from realmspinner.kernels.pixel import ora
    from realmspinner.service.errors import ServiceError

    ctx, state, tab = _session()
    tab.doc.add_layer()
    monkeypatch.setattr(ora, "_layer_budget", lambda w, h: 1)

    with pytest.raises(ServiceError) as saved:
        inker_mode._write(tab.doc, tmp_path / "x.ora", "ora")
    assert "layers" in str(saved.value) and "could not be saved" in str(saved.value)

    # The journal's task raises the same sentence rather than a bare ValueError
    # the task layer would show as "Something went wrong".
    deferred = inker_mode._journal_encode(tab)
    with pytest.raises(ServiceError) as journaled:
        deferred()
    assert "layers" in str(journaled.value)


# --- inker-mode-10 ------------------------------------------------------------


def test_a_row_greyed_by_a_free_transform_says_so():
    from realmspinner.studio.modes.inker import export as inker_export

    ctx, state, tab = _session()
    state.transforming = True
    assert not tab.busy
    assert inker_ops.TRANSFORM_WHY == inker_export.TRANSFORM_WHY

    wrong = []
    for name in ("export_png", "save", "add_layer", "flip_h"):
        op = inker_ops.get(name)
        assert not op.enabled(state, tab), name
        said = inker_ops.reason_for(op, state, tab)
        if "free transform" not in said:
            wrong.append((name, said))
    assert wrong == [], "named a save or playback that is not running"

    # A real save still says so, and wins when both are true.
    tab.saving = True
    for name in ("export_png", "add_layer"):
        assert inker_ops.reason_for(inker_ops.get(name), state, tab) == inker_ops.BUSY


# --- inker-mode-11 ------------------------------------------------------------


def test_a_refused_os_clipboard_image_does_not_fall_through_to_the_in_app_clipboard(
    monkeypatch,
):
    from PIL import Image, ImageGrab

    from realmspinner.core.safeio import pixelguard

    ctx, state, tab = _session()
    ctx.state.inker = state
    doc = tab.doc
    doc.stack.active.pixels[0:4, 0:4] = RED
    doc.invalidate_all()
    doc.select_all()
    doc.copy()  # an earlier in-app Copy
    doc.deselect()

    monkeypatch.setattr(pixelguard, "MAX_DECODE_PIXELS", 100)
    monkeypatch.setattr(ImageGrab, "grabclipboard", lambda *a, **k: Image.new("RGBA", (64, 64)))

    assert inker_ops.run(ctx, inker_ops.get("paste")) is False
    assert ctx.toasts and "not pasted" in str(ctx.toasts[-1][0])
    assert doc.floating is None, "the stale in-app selection floated after a refusal toast"

    # No image on the OS clipboard at all is the case that still falls through.
    monkeypatch.setattr(ImageGrab, "grabclipboard", lambda *a, **k: None)
    assert inker_ops.run(ctx, inker_ops.get("paste")) is True
    assert doc.floating is not None


# --- inker-panes-04 -----------------------------------------------------------


def _inpaint_session(monkeypatch):
    from realmspinner.service import jobs as svc_jobs
    from realmspinner.studio.modes.inker.ui.panes import bridge

    tab = inker_state.InkerDoc(doc=inker.Document.blank(16, 16))
    tab.doc.stack.active.pixels[0:8, 0:8] = RED
    tab.doc.invalidate_all()
    tab.doc.select_rect(0, 0, 8, 8) if hasattr(tab.doc, "select_rect") else tab.doc.select_all()
    state = inker_state.InkerState()
    state.add(tab)
    toasts: list = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=state),
        svc=object(),
        toast=lambda *a, **k: toasts.append(a),
        submit=lambda key, fn, *a, **k: True,
    )
    ctx.toasts = toasts
    monkeypatch.setattr(svc_jobs, "create_job", lambda svc, **kw: {"id": "j"})
    assert bridge.submit_inpaint(ctx, tab, "a door", 0.5) is True
    return bridge, ctx, state, tab, state.inpaint_pending


def test_a_regeneration_does_not_land_after_the_canvas_was_flipped(monkeypatch):
    bridge, ctx, state, tab, pending = _inpaint_session(monkeypatch)
    box = tuple(pending["box"])
    pixels = np.full((box[3] - box[1], box[2] - box[0], 4), 77, dtype=np.uint8)

    tab.doc.flip("horizontal")  # while the job was queued
    before = tab.doc.stack.active.pixels.copy()

    assert bridge.land_inpaint(ctx, pending, pixels) is False
    assert (tab.doc.stack.active.pixels == before).all()
    assert "refused" in str(ctx.toasts[-1][0]) and "landed" not in str(ctx.toasts[-1][0])


def test_a_regeneration_does_not_land_after_the_canvas_was_resized(monkeypatch):
    bridge, ctx, state, tab, pending = _inpaint_session(monkeypatch)
    box = tuple(pending["box"])
    pixels = np.full((box[3] - box[1], box[2] - box[0], 4), 77, dtype=np.uint8)

    tab.doc.resize_canvas((24, 24))
    assert bridge.land_inpaint(ctx, pending, pixels) is False


def test_a_regeneration_still_lands_on_an_unmoved_canvas_and_after_an_undone_flip(
    monkeypatch,
):
    bridge, ctx, state, tab, pending = _inpaint_session(monkeypatch)
    box = tuple(pending["box"])
    pixels = np.full((box[3] - box[1], box[2] - box[0], 4), 77, dtype=np.uint8)

    tab.doc.flip("horizontal")
    assert tab.doc.undo()  # the document is back where the box was measured
    assert bridge.land_inpaint(ctx, pending, pixels) is True
    assert "landed" in str(ctx.toasts[-1][0])


# --- inker-panes-02 -----------------------------------------------------------


def test_the_seam_figure_ignores_an_axis_the_document_does_not_wrap():
    from realmspinner.kernels.pixel.tiling import seam_dominance
    from realmspinner.studio import theme
    from realmspinner.studio.modes.inker.ui.panes import canvas

    doc = inker.Document.blank(128, 128)
    ramp = np.linspace(0, 255, 128).astype(np.uint8)
    pixels = doc.stack.active.pixels
    for x in range(128):
        pixels[:, x, :3] = ramp[:, None]  # seamless sideways, a hard edge top to bottom
        pixels[:, x, 3] = 255
    doc.invalidate_all()
    across = seam_dominance(doc.flatten(matte=False))
    assert across[0] < across[1] and across[1] > canvas.SEAM_DOMINANCE_MAX

    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="strip")
    ctx = SimpleNamespace(state=SimpleNamespace(preview={}))

    tab.tiled = "x"
    colour, text = canvas.seam_text(ctx, tab)
    assert colour == theme.MUTED, f"{text} warned about the vertical edge nobody wraps"

    # The cache is keyed on the tiling mode as well as the undo head.
    tab.tiled = "y"
    colour, text = canvas.seam_text(ctx, tab)
    assert colour == theme.WARN
    tab.tiled = "x"
    assert canvas.seam_text(ctx, tab)[0] == theme.MUTED


# --- inker-panes-01 -----------------------------------------------------------


def test_colour_panel_edits_convert_the_list_imgui_returns(monkeypatch):
    from realmspinner.studio import controls
    from realmspinner.studio.modes.inker.ui.panes import colors as inker_colors

    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.begin("x")
        _changed, value = controls.color_edit4(
            "Foreground", inker_colors._vec((10, 20, 30, 255)), inker_colors.FLAGS
        )
        imgui.end()
        imgui.end_frame()

    assert inker_colors._to_rgba(value) == (10, 20, 30, 255)
    assert inker_colors._to_rgba([0.5, 0.25, 0.0, 1.0]) == (128, 64, 0, 255)
    assert inker_colors._to_rgba((1.5, -0.2, 0.0, 1.0)) == (255, 0, 0, 255)
    # A build that still returns an ImVec4 keeps working.
    assert inker_colors._to_rgba(SimpleNamespace(x=1.0, y=0.0, z=0.0, w=1.0)) == (255, 0, 0, 255)


# --- inker-panes-05 -----------------------------------------------------------


def test_the_shade_direction_the_bar_stores_survives_a_preset_round_trip(monkeypatch):
    from realmspinner.studio import controls
    from realmspinner.studio.modes.inker.ui.panes import context as inker_context

    ctx, state, tab = _session()
    state.set_tool("shade")
    ctx.state.inker = state
    field = inker_context._field(ctx, state, tab, "shade_dir")
    assert field is not None

    monkeypatch.setattr(controls, "combo", lambda *a, **k: (True, "-1"))
    field.draw(False)

    stored = state.options_for("shade")["shade_dir"]
    assert stored == -1 and type(stored) is int, repr(stored)
    assert inker_mode._coerce_preset_options({"shade_dir": stored}) == {"shade_dir": -1}

