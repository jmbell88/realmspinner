"""The 2026-10-10 audit's Mesh column findings (create-08, -10, -12, -18, -20).

Each test's name is its claim; every one fails against the unfixed code.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

from realmspinner.service.errors import Invalid
from realmspinner.studio.formvalues import coerce_form_value
from realmspinner.studio.modes.create.ui.panes import settings_3d
from realmspinner.studio.panes import pose_panel, remesh_panel, retarget_panel
from realmspinner.studio.state import DEFAULT_FORM_3D, AppState

_SOURCE = {
    "id": "ref-1",
    "name": "mossy well",
    "status": "done",
    "stage": "reference",
    "files": ["input.png"],
}


def _hosts(monkeypatch, *, blender: bool, gltfpack: bool) -> None:
    monkeypatch.setattr(remesh_panel, "blender_available", lambda _ctx: blender)
    monkeypatch.setattr(retarget_panel, "gltfpack_available", lambda _ctx: gltfpack)


def _quiet_widgets(monkeypatch, combo=None) -> None:
    """Stand in for the drawing a headless test cannot do."""
    monkeypatch.setattr(
        settings_3d.widgets,
        "labeled_combo",
        combo or (lambda label, value, options, **kw: value),
    )
    monkeypatch.setattr(settings_3d.widgets, "field_error", lambda *a, **kw: None)
    monkeypatch.setattr(settings_3d.widgets, "field_label", lambda *a, **kw: None)
    monkeypatch.setattr(settings_3d, "_hint", lambda *a, **kw: None)


def _ctx(**extra):
    state = AppState(mode="create")
    state.create.stage = "mesh"
    state.source_job = "ref-1"
    toasts: list[str] = []
    ns = SimpleNamespace(
        state=state,
        cache=SimpleNamespace(get={"ref-1": _SOURCE}.get, jobs=[_SOURCE], active=None),
        job=lambda: None,
        model_rows=[],
        submit=lambda *a, **k: True,
        busy=lambda _k: False,
        rigging_available=False,
        toast=lambda text, *a, **k: toasts.append(text),
        toasts=toasts,
    )
    for key, value in extra.items():
        setattr(ns, key, value)
    return ns


# --- create-08 ---------------------------------------------------------------------


def test_coerce_form_value_reads_the_string_False_as_false_for_a_bool_field():
    """``bool("False")`` is True, so a bool control's "Use False" suggestion
    (the findings buckets are keyed ``str(value)``) wrote True back."""
    assert coerce_form_value(True, "False") is False
    assert coerce_form_value(True, "false") is False
    assert coerce_form_value(False, "True") is True
    assert coerce_form_value(False, "true") is True
    assert coerce_form_value(False, " TRUE ") is True
    # A string that says neither leaves the field as it was.
    assert coerce_form_value(True, "maybe") is True
    # The other types are unchanged.
    assert coerce_form_value(0.5, "0.6") == 0.6
    assert coerce_form_value(3, "4") == 4
    assert coerce_form_value("a", 7) == "7"
    assert coerce_form_value(2, "x") == 2


# --- create-10 ---------------------------------------------------------------------


def test_a_draw_without_blender_does_not_overwrite_the_stored_game_ready_budget(monkeypatch):
    """``_budget`` rewrote ``profile``/``lowpoly_triangles`` on every draw of a
    host without Blender and ``form_3d`` is saved at quit, so installing the
    Rigging pack later did not bring the 5k default back."""
    _quiet_widgets(monkeypatch)
    shown: list[str] = []
    monkeypatch.setattr(
        settings_3d.widgets,
        "labeled_combo",
        lambda label, value, options, **kw: shown.append(value) or value,
    )
    for blender, gltfpack in ((False, True), (False, False)):
        _hosts(monkeypatch, blender=blender, gltfpack=gltfpack)
        form = dict(DEFAULT_FORM_3D)
        stored = (form["profile"], form["lowpoly_triangles"])
        assert stored[1] > 0, "the premise: the shipped default is a Game-ready budget"
        settings_3d._budget(_ctx(), form)
        assert (form["profile"], form["lowpoly_triangles"]) == stored, (blender, gltfpack)
    # What the combo shows is still the host's fallback, not the 5k it cannot offer.
    assert shown == ["standard"]


def test_make_3d_without_blender_sends_the_host_fallback_and_leaves_the_stored_budget(
    monkeypatch,
):
    sent: list[dict] = []
    monkeypatch.setattr(
        settings_3d.matte_preview, "open_for", lambda ctx, job_id, kw: sent.append(kw)
    )
    _hosts(monkeypatch, blender=False, gltfpack=True)
    form = dict(DEFAULT_FORM_3D)
    stored = (form["profile"], form["lowpoly_triangles"])
    settings_3d.promote(_ctx(), _SOURCE, form)
    assert sent[0]["profile"] == "standard"
    assert "lowpoly_triangles" not in sent[0]
    assert (form["profile"], form["lowpoly_triangles"]) == stored

    # With neither binary the request is Raw.
    _hosts(monkeypatch, blender=False, gltfpack=False)
    settings_3d.promote(_ctx(), _SOURCE, form)
    assert sent[1]["profile"] == "raw"
    assert "lowpoly_triangles" not in sent[1]

    # With Blender the stored Game-ready budget goes out unchanged.
    _hosts(monkeypatch, blender=True, gltfpack=True)
    settings_3d.promote(_ctx(), _SOURCE, form)
    assert sent[2]["lowpoly_triangles"] == stored[1]


# --- create-18 ---------------------------------------------------------------------

_RINGS = ("profile", "lowpoly_triangles", "mesh_finishing", "custom_triangles")


def _ringed(ctx, *names):
    for name in names:
        ctx.state.note_field_error(name, f"{name} was refused")
    return ctx


def test_picking_a_budget_clears_the_profile_and_triangle_rings(monkeypatch):
    _hosts(monkeypatch, blender=True, gltfpack=True)
    _quiet_widgets(
        monkeypatch,
        lambda label, value, options, **kw: "lowpoly:10k" if label == "Budget" else value,
    )
    ctx = _ringed(_ctx(), *_RINGS, "platform")
    settings_3d._budget(ctx, dict(DEFAULT_FORM_3D))
    for name in _RINGS:
        assert name not in ctx.state.field_errors, name
    # A different control's ring is not this pick's to clear.
    assert "platform" in ctx.state.field_errors


def test_editing_the_finishing_combo_clears_its_ring(monkeypatch):
    _hosts(monkeypatch, blender=True, gltfpack=True)
    _quiet_widgets(
        monkeypatch,
        lambda label, value, options, **kw: "repair" if label == "Finishing" else value,
    )
    ctx = _ringed(_ctx(), "mesh_finishing")
    form = dict(DEFAULT_FORM_3D)
    settings_3d._budget(ctx, form)
    assert form["mesh_finishing"] == "repair"
    assert "mesh_finishing" not in ctx.state.field_errors


def test_editing_the_custom_triangle_count_clears_its_ring(monkeypatch):
    _hosts(monkeypatch, blender=True, gltfpack=True)
    _quiet_widgets(monkeypatch)
    monkeypatch.setattr(settings_3d, "imgui", SimpleNamespace(set_next_item_width=lambda *_: None))
    monkeypatch.setattr(settings_3d.controls, "input_int", lambda *a, **kw: (True, 60_000))
    ctx = _ringed(_ctx(), "custom_triangles")
    form = dict(DEFAULT_FORM_3D)
    settings_3d._apply_budget_choice(form, "custom")
    settings_3d._budget(ctx, form)
    assert form["custom_triangles"] == 60_000
    assert "custom_triangles" not in ctx.state.field_errors


def test_the_create_3d_form_clears_a_ring_when_its_own_control_is_edited():
    source = Path(settings_3d.__file__).read_text(encoding="utf-8")
    assert "on_edit=ctx.state.clear_field_error" in source


# --- create-12 ---------------------------------------------------------------------


class _Viewer:
    def __init__(self) -> None:
        self.moved = {"hips": [0.0, 1.0, 0.0]}
        self.selected_bone = None
        self.exits = 0

    def joint_moves(self):
        return dict(self.moved)

    def corrected_bones(self):
        return [{"name": "hips"}]

    def exit_joints_mode(self) -> None:
        self.exits += 1
        self.moved.clear()


def _joints_harness(monkeypatch, *, refuses: bool):
    """A synchronous ``ctx.submit`` that behaves like ``TaskRunner``: the task's
    exception is kept for the shell's toast, not raised at the caller."""
    ran: list[str] = []

    def adjust(svc, job_id, payload):
        ran.append(job_id)
        if refuses:
            raise Invalid("the joints do not describe this rig", field="bones")
        return {"ok": True}

    monkeypatch.setattr(pose_panel.svc_rig, "adjust_joints", adjust)
    monkeypatch.setattr(pose_panel.widgets, "muted", lambda *a, **k: None)
    monkeypatch.setattr(pose_panel.controls, "button", lambda *a, **k: False)
    monkeypatch.setattr(
        pose_panel.widgets,
        "disabled_button",
        lambda label, enabled, *a, **k: bool(enabled) and label == "Apply joint positions",
    )

    def submit(key, fn, *args, **kwargs):
        with contextlib.suppress(Invalid):
            fn(*args, **kwargs)
        return True

    ctx = SimpleNamespace(busy=lambda _k: False, svc=object(), submit=submit)
    return ctx, ran


def test_a_refused_apply_joint_positions_keeps_the_dragged_markers(monkeypatch):
    ctx, ran = _joints_harness(monkeypatch, refuses=True)
    viewer = _Viewer()
    job = {"id": "rig-1"}
    pose_panel._joints(ctx, job, viewer)
    assert ran == ["rig-1"], "the press reached the door"
    # The next frames: the refusal has landed and the markers are still there.
    pose_panel._joints(ctx, job, viewer)
    pose_panel._joints(ctx, job, viewer)
    assert viewer.exits == 0
    assert viewer.moved


def test_a_landed_apply_joint_positions_leaves_joints_mode(monkeypatch):
    ctx, ran = _joints_harness(monkeypatch, refuses=False)
    viewer = _Viewer()
    job = {"id": "rig-2"}
    pose_panel._joints(ctx, job, viewer)
    assert ran == ["rig-2"]
    pose_panel._joints(ctx, job, viewer)
    assert viewer.exits == 1


# --- create-20 ---------------------------------------------------------------------


def test_a_second_drop_during_an_import_says_it_was_not_taken(tmp_path):
    ctx = _ctx(submit=lambda *a, **k: False)
    picture = tmp_path / "a.png"
    picture.write_bytes(b"x")
    settings_3d.upload(ctx, picture)
    assert ctx.toasts == [settings_3d.IMPORT_BUSY_TOAST]
    settings_3d.upload_bytes(ctx, b"x")
    assert ctx.toasts == [settings_3d.IMPORT_BUSY_TOAST] * 2
    # An accepted import says nothing.
    ok = _ctx(submit=lambda *a, **k: True)
    settings_3d.upload(ok, picture)
    settings_3d.upload_bytes(ok, b"x")
    assert ok.toasts == []
