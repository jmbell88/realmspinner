"""The "Weight with Blender" flow around Blender: the merge, the row, the queued job, the
poll, the read-back and the adoption -- everything but Blender itself.

Tranche 4 of the 2026-10-13 Clay rig plan. The Blender run is the sibling test in
``tests/pipelines/test_clay_rig_blender.py``; here the queue is fake, so each step's contract is
asserted on its own. Each claim fails on the tree before the change.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.rig import clayrig
from realmspinner.service import rig as svc_rig
from realmspinner.service.errors import Conflict, Invalid
from realmspinner.studio.modes.clay import mode as clay_mode

from .test_adopt_rig import _blender_result, _document
from .test_clay_mode import FakeCtx


@pytest.fixture(autouse=True)
def _blender_present(monkeypatch):
    monkeypatch.setattr(
        "realmspinner.doctor.blender_check", lambda *a, **k: SimpleNamespace(ok=True, detail="")
    )


class Ctx(FakeCtx):
    """``FakeCtx`` plus what this flow reads: Blender's presence and the live job cache."""

    def __init__(self, svc=None, *, rigging: bool = True) -> None:
        super().__init__(svc)
        self.rigging_available = rigging
        self.jobs: dict[str, dict] = {}

    def job(self, job_id=None):
        return self.jobs.get(job_id)


def _tab(ctx, doc=None):
    doc, ids = (doc, None) if doc is not None else _document()
    tab = clay_mode.adopt(ctx, doc, title="Hero")
    return tab, ids


# -- the sources and the skeleton ----------------------------------------------------------------


def test_blender_skins_the_visible_shapes_that_are_neither_bones_nor_weighted() -> None:
    doc, ids = _document()
    hidden = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hidden", mesh=bp.box(), visible=False))
    assert {o.name for o in clayrig.weight_sources(doc)} == {"Torso", "Hand"}
    assert hidden.uid not in {o.uid for o in clayrig.weight_sources(doc)}


def test_the_merged_source_is_one_object_in_world_space_with_the_palette_carried() -> None:
    doc, ids = _document()
    scratch = clayrig.weight_source_doc(doc)
    [merged] = scratch.objects
    assert not merged.bone and merged.skin is None
    assert np.allclose(merged.translation, 0) and np.allclose(merged.scale, 1)
    # Torso at y=1 (a unit box) and Hand at world (1.5, 1, 0) are both in it, where they stand.
    hi = merged.mesh.positions.max(axis=0)
    lo = merged.mesh.positions.min(axis=0)
    assert hi[0] == pytest.approx(2.0) and lo[0] == pytest.approx(-0.5)
    assert hi[1] == pytest.approx(1.5) and lo[1] == pytest.approx(0.5)
    assert [m.name for m in scratch.materials] == [m.name for m in doc.materials]


def test_nothing_to_weight_is_a_sentence_not_an_empty_mesh() -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True))
    with pytest.raises(el.OpError, match="nothing to weight"):
        clayrig.weight_source_doc(doc)


def test_the_skeleton_is_in_blender_axes_through_the_placement_with_a_single_root() -> None:
    doc, ids = _document()
    place = np.diag([2.0, 2.0, 2.0, 1.0])
    place[:3, 3] = [0.0, -1.0, 0.0]
    skeleton = clayrig.heat_skeleton(doc, place)
    by_name = {b["name"]: b for b in skeleton["bones"]}
    # Hip stands at (0, 1, 0) in the document: placed, (0, 1, 0); in Blender's axes, (0, 0, 1).
    assert by_name["Hip"]["head"] == pytest.approx([0.0, 0.0, 1.0])
    assert by_name["Arm"]["parent"] == "Hip"
    assert set(skeleton) == {"bones", "bounds", "mirror_pairs"}


def test_a_skeleton_with_two_roots_is_refused_by_name() -> None:
    doc, ids = _document()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Tail", mesh=bp.box(), bone=True))
    with pytest.raises(el.OpError, match="one root.*2.*Hip, Tail"):
        clayrig.heat_skeleton(doc, np.eye(4))
    with pytest.raises(el.OpError, match="no bones"):
        clayrig.heat_skeleton(bd.ClayDoc(), np.eye(4))


# -- the service door ----------------------------------------------------------------------------


def _row(svc) -> str:
    """A finished library row with a mesh on disk."""
    doc, _ = _document()
    glb = glbwrite.write_glb(bd.to_model(clayrig.weight_source_doc(doc), skinned=False))
    from realmspinner.service import jobs as svc_jobs

    return svc_jobs.import_mesh(svc, glb, name="rows")["id"]


def _write_rig(svc, source: str) -> None:
    """What the queue leaves beside the row's mesh once Blender is done."""
    rigged = glbwrite.write_glb(bd.to_model(_blender_result()))
    (svc.job_dir(source) / "rig.glb").write_bytes(rigged)


def _skeleton(svc):
    doc, _ = _document()
    return clayrig.heat_skeleton(doc, np.eye(4))


def test_rig_with_skeleton_queues_a_custom_rig_job_against_a_row_with_no_rig(svc) -> None:
    job_id = _row(svc)
    skeleton = _skeleton(svc)
    queued = svc_rig.rig_with_skeleton(
        svc, job_id, bones=skeleton["bones"], bounds=skeleton["bounds"],
        mirror_pairs=skeleton["mirror_pairs"],
    )
    row = svc.store.get(queued["id"])
    assert row["kind"] == "rig" and row["status"] == "queued"
    params = row["params"]
    assert params["source_job"] == job_id and params["skeleton"] == "custom"
    assert params["root"] == "Hip" and params["template"] == "blank" and params["adjusted"]
    assert [b["name"] for b in params["bones"]] == ["Hip", "Arm"]


def test_a_second_run_while_one_is_in_flight_is_a_conflict(svc) -> None:
    job_id = _row(svc)
    skeleton = _skeleton(svc)
    kw = {"bones": skeleton["bones"], "bounds": skeleton["bounds"]}
    svc_rig.rig_with_skeleton(svc, job_id, **kw)
    with pytest.raises(Conflict, match="already running"):
        svc_rig.rig_with_skeleton(svc, job_id, **kw)


def test_the_door_refuses_without_blender_a_rig_job_a_missing_mesh_and_a_bad_skeleton(
    svc, monkeypatch
) -> None:
    job_id = _row(svc)
    skeleton = _skeleton(svc)
    kw = {"bones": skeleton["bones"], "bounds": skeleton["bounds"]}
    with pytest.raises(Invalid, match="no finished mesh"):
        svc.store.create("image", "", {}, "abcdef012345", stage="model", status="done")
        svc_rig.rig_with_skeleton(svc, "abcdef012345", **kw)
    with pytest.raises(Invalid, match="cannot be used"):
        svc_rig.rig_with_skeleton(svc, job_id, bones=[], bounds=skeleton["bounds"])
    monkeypatch.setattr(
        "realmspinner.doctor.blender_check", lambda *a, **k: SimpleNamespace(ok=False, detail="")
    )
    with pytest.raises(Invalid, match="needs Blender"):
        svc_rig.rig_with_skeleton(svc, job_id, **kw)


# -- starting a run ------------------------------------------------------------------------------


def test_the_button_says_why_it_cannot_start() -> None:
    ctx = Ctx()
    tab, _ = _tab(ctx)
    assert clay_mode.rig_blocked_why(ctx, None) == "Open or start a document first."
    assert clay_mode.rig_blocked_why(ctx, tab) == ""
    assert "Blender" in clay_mode.rig_blocked_why(Ctx(rigging=False), tab)
    tab.saving = True
    assert clay_mode.rig_blocked_why(ctx, tab) == "Saving..."
    tab.saving = False
    tab.pending_rig = {"rig_job": "x"}
    assert "already weighting" in clay_mode.rig_blocked_why(ctx, tab)
    tab.pending_rig = None
    bare = bd.ClayDoc()
    bare.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    other, _ = _tab(ctx, bare)
    assert "Mark the joints" in clay_mode.rig_blocked_why(ctx, other)
    shapeless = bd.ClayDoc()
    shapeless.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True))
    third, _ = _tab(ctx, shapeless)
    assert "nothing to weight" in clay_mode.rig_blocked_why(ctx, third)


def test_starting_a_run_makes_a_row_queues_the_job_and_tracks_it(svc) -> None:
    ctx = Ctx(svc)
    tab, ids = _tab(ctx)
    clay_mode.rig_with_blender(ctx, tab)
    assert tab.saving and ctx.submitted == [f"clay-rigweights:{tab.uid}"]
    clay_mode.on_task_done(ctx, SimpleNamespace(key=ctx.submitted[-1], result=ctx.result))

    pending = tab.pending_rig
    assert not tab.saving and pending is not None and not pending.get("reading")
    row = svc.store.get(pending["source_job"])
    assert row["status"] == "done" and (svc.job_dir(pending["source_job"]) / "model.glb").exists()
    rig_row = svc.store.get(pending["rig_job"])
    assert rig_row["kind"] == "rig" and rig_row["params"]["source_job"] == pending["source_job"]
    assert set(pending["sources"]) == {ids["torso"], ids["hand"]}
    assert pending["names"][ids["hip"]] == "Hip"
    assert any("Weighting in Blender" in m for m, _l in ctx.toasts)


def test_a_run_that_cannot_be_queued_leaves_no_row_behind(svc, monkeypatch) -> None:
    ctx = Ctx(svc)
    tab, _ = _tab(ctx)
    monkeypatch.setattr(
        "realmspinner.doctor.blender_check", lambda *a, **k: SimpleNamespace(ok=False, detail="")
    )
    before = {j["id"] for j in svc.store.list_jobs()} if hasattr(svc.store, "list_jobs") else None
    with pytest.raises(Invalid, match="needs Blender"):
        clay_mode.rig_with_blender(ctx, tab)
    clay_mode.on_task_failed(ctx, SimpleNamespace(key=ctx.submitted[-1]))
    assert not tab.saving and tab.pending_rig is None
    if before is not None:
        assert {j["id"] for j in svc.store.list_jobs()} == before


# -- the poll and the read-back ------------------------------------------------------------------


def _pending(svc, ids, source_job="src") -> dict:
    return {
        "source_job": source_job,
        "rig_job": "rg",
        "place": np.eye(4),
        "names": {ids["hip"]: "Hip", ids["arm"]: "Arm"},
        "sources": {},
    }


def test_the_poll_leaves_a_running_job_alone_and_reads_a_finished_one_once(svc) -> None:
    ctx = Ctx(svc)
    tab, ids = _tab(ctx)
    source = _row(svc)
    _write_rig(svc, source)
    tab.pending_rig = _pending(svc, ids, source_job=source)
    clay_mode.pump_rig(ctx)
    assert ctx.submitted == [], "no job in the cache yet"
    ctx.jobs["rg"] = {"status": "running"}
    clay_mode.pump_rig(ctx)
    assert ctx.submitted == []
    ctx.jobs["rg"] = {"status": "done"}
    clay_mode.pump_rig(ctx)
    clay_mode.pump_rig(ctx)
    assert ctx.submitted == [f"clay-rigimport:{tab.uid}"], "read once, not every frame"
    assert tab.pending_rig["reading"]


def test_an_errored_or_cancelled_run_is_dropped_with_a_toast_and_changes_nothing(svc) -> None:
    ctx = Ctx(svc)
    tab, ids = _tab(ctx)
    before = [(o.uid, o.name) for o in tab.doc.objects]
    for status in ("error", "cancelled"):
        tab.pending_rig = _pending(svc, ids)
        ctx.jobs["rg"] = {"status": status}
        clay_mode.pump_rig(ctx)
        assert tab.pending_rig is None
    assert sum("did not finish" in m for m, _l in ctx.toasts) == 2
    assert [(o.uid, o.name) for o in tab.doc.objects] == before


def test_the_finished_rig_is_read_from_the_source_rows_rig_glb(svc, tmp_path) -> None:
    ctx = Ctx(svc)
    tab, ids = _tab(ctx)
    source = _row(svc)
    _write_rig(svc, source)
    tab.pending_rig = _pending(svc, ids, source_job=source)
    ctx.jobs["rg"] = {"status": "done"}
    clay_mode.pump_rig(ctx)
    result = ctx.result
    assert {o.name for o in result["rig_read"].objects if o.bone} == {"Hip", "Arm", "Hand_end"}
    assert any(o.skin is not None for o in result["rig_read"].objects)


# -- landing -------------------------------------------------------------------------------------


def _landing(ctx, tab, ids, doc):
    snapshot = {
        uid: (doc.by_uid(uid).mesh, doc.world_matrix(uid).copy())
        for uid in (ids["torso"], ids["hand"])
    }
    tab.pending_rig = {
        "source_job": "s", "rig_job": "r", "place": np.eye(4), "reading": True,
        "names": dict(clay_mode_names(doc)), "sources": snapshot,
    }


def clay_mode_names(doc):
    from realmspinner.kernels.mesh import rig

    return rig.export_names(doc.objects)


def test_a_finished_run_is_adopted_as_one_undo_step_with_a_toast_saying_what_happened(svc) -> None:
    ctx = Ctx(svc)
    doc, ids = _document()
    tab, _ = _tab(ctx, doc)
    _landing(ctx, tab, ids, doc)
    steps = len(doc.history)
    clay_mode.on_task_done(ctx, SimpleNamespace(key=f"clay-rigimport:{tab.uid}",
                                                result={"rig_read": _blender_result()}))
    assert tab.pending_rig is None and len(doc.history) == steps + 1
    assert any(o.skin is not None for o in doc.objects)
    assert not {ids["torso"], ids["hand"]} & {o.uid for o in doc.objects}
    [line] = [m for m, _l in ctx.toasts if m.startswith("Weighted")]
    assert "Weighted 2 shapes to 3 bones (1 new)" in line and "Undo puts the shapes back" in line


def test_a_shape_edited_while_blender_worked_is_not_replaced_and_the_user_is_told(svc) -> None:
    ctx = Ctx(svc)
    doc, ids = _document()
    tab, _ = _tab(ctx, doc)
    _landing(ctx, tab, ids, doc)
    doc.set_transform(ids["torso"], translation=(0.0, 3.0, 0.0))
    steps = len(doc.history)
    clay_mode.on_task_done(ctx, SimpleNamespace(key=f"clay-rigimport:{tab.uid}",
                                                result={"rig_read": _blender_result()}))
    assert len(doc.history) == steps and tab.pending_rig is None
    assert any("Torso" in m and "changed while Blender" in m for m, _l in ctx.toasts)
    assert any(o.uid == ids["torso"] for o in doc.objects)


def test_a_shape_deleted_while_blender_worked_is_noticed_too(svc) -> None:
    ctx = Ctx(svc)
    doc, ids = _document()
    tab, _ = _tab(ctx, doc)
    _landing(ctx, tab, ids, doc)
    doc.remove_object(ids["hand"])
    clay_mode.on_task_done(ctx, SimpleNamespace(key=f"clay-rigimport:{tab.uid}",
                                                result={"rig_read": _blender_result()}))
    assert any("a deleted shape" in m for m, _l in ctx.toasts)
    assert not any(o.skin is not None for o in doc.objects)


def test_a_failed_read_clears_the_tracking_and_unlocks(svc) -> None:
    ctx = Ctx(svc)
    tab, ids = _tab(ctx)
    tab.pending_rig = _pending(svc, ids)
    tab.pending_rig["reading"] = True
    clay_mode.on_task_failed(ctx, SimpleNamespace(key=f"clay-rigimport:{tab.uid}"))
    assert tab.pending_rig is None and not tab.saving


# -- the doors -----------------------------------------------------------------------------------


def test_the_palette_offers_weight_with_blender_only_when_it_could_run() -> None:
    from .test_clay_palette import _clay
    from .test_clay_palette import _ctx as palette_ctx

    ctx = palette_ctx()
    command = _clay(ctx)["clay-rig:weights"]
    assert command.label == "Weight with Blender..."
    assert not command.enabled(ctx), "no bones, and no Blender in this fake"
    ctx.rigging_available = True
    doc = ctx.state.clay.active.doc
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True))
    assert command.enabled(ctx), "a bone, a shape (the fixture's box) and Blender"


def test_the_rig_card_offers_the_button_greyed_with_its_reason_and_runs_it(monkeypatch) -> None:
    from _ui_context import imgui_context

    from realmspinner.studio import widgets

    from .test_inspector import _document as inspector_document
    from .test_inspector import _rig as inspector_rig

    calls: list[object] = []
    monkeypatch.setattr(clay_mode, "rig_with_blender", lambda ctx, tab=None: calls.append(tab))
    with imgui_context(monkeypatch) as imgui:
        ctx, tab, doc = inspector_document(1)
        doc.set_bone([doc.objects[0].uid], True)
        extra = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Skin", mesh=bp.box()))
        doc.select([extra.uid])
        widgets.request_open("clay/insp/rig")
        card = inspector_rig(imgui, ctx)
        button = card.find(contains="Weight with Blender...##rigweights")
        assert not button.enabled, "no Blender in the fake ctx"
        ctx.rigging_available = True
        widgets.request_open("clay/insp/rig")
        card = inspector_rig(imgui, ctx)
        button = card.find(contains="Weight with Blender...##rigweights")
        assert button.enabled
        card.click(contains="Weight with Blender...##rigweights")
    assert calls == [tab]
