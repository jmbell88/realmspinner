"""The 2026-10-14 audit, fixer F24: Clay's mode-level rigging, keying, export and gesture seams.

Each test is named for the claim of one finding and was run against the unfixed tree first.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.elements import OpError
from realmspinner.studio.modes.clay import mode as clay_mode

from ._rigkit import skin_for, weighted
from .test_adopt_rig import _blender_result, _document
from .test_clay_mode import FakeCtx
from .test_rig_with_blender import Ctx, _pending, _row, _tab, _write_rig

Q90 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])

SRC = Path(__file__).resolve().parents[3] / "src" / "realmspinner"


@pytest.fixture(autouse=True)
def _blender_present(monkeypatch):
    monkeypatch.setattr(
        "realmspinner.doctor.blender_check", lambda *a, **k: SimpleNamespace(ok=True, detail="")
    )
    # ``pygame.key.get_mods`` needs a video system; there is none in a test.
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


def _flat_document():
    """:func:`_document` with the hand off the arm: no shape rides a bone, so a bone can be
    changed without the shapes' own stale check being what notices."""
    from .test_adopt_rig import _add

    doc, ids = _document()
    doc.remove_object(ids["hand"])
    ids["hand"] = _add(doc, "Hand", translation=(1.5, 1, 0))
    return doc, ids


def _queued(svc, flat: bool = False):
    """A real Weight-with-Blender start: the row made, the job queued, the pins taken."""
    ctx = Ctx(svc)
    doc, ids = _flat_document() if flat else _document()
    tab, _ = _tab(ctx, doc)
    clay_mode.rig_with_blender(ctx, tab)
    clay_mode.on_task_done(ctx, SimpleNamespace(key=ctx.submitted[-1], result=ctx.result))
    assert tab.pending_rig is not None and not tab.saving
    tab.pending_rig["reading"] = True
    return ctx, tab, doc, ids


def _land(ctx, tab, result=None):
    clay_mode.on_task_done(
        ctx,
        SimpleNamespace(
            key=f"clay-rigimport:{tab.uid}", result={"rig_read": result or _blender_result()}
        ),
    )


# -- clay-08: a bone changed while Blender worked ---------------------------------------------


@pytest.mark.parametrize("change", ["moved", "unmarked", "deleted"])
def test_a_bone_moved_or_unmarked_while_blender_worked_stops_the_weights_landing(
    svc, change
) -> None:
    ctx, tab, doc, ids = _queued(svc, flat=True)
    if change == "moved":
        doc.set_transform(ids["hip"], translation=(0.0, 2.0, 0.0))
    elif change == "unmarked":
        doc.set_bone([ids["hip"]], False)
    else:
        doc.remove_object(ids["arm"])
    steps = len(doc.history)
    ctx.toasts.clear()
    _land(ctx, tab)
    assert len(doc.history) == steps, "the weights must not land on a document that moved on"
    assert tab.pending_rig is None
    assert not any(o.skin is not None for o in doc.objects)
    name = {"moved": "Hip", "unmarked": "Hip", "deleted": "Arm"}[change]
    [(message, level)] = [t for t in ctx.toasts if "changed while Blender" in t[0]]
    assert name in message and level == "error"


# -- clay-09 / clay-ops-tail-01: the I key in Pose mode ---------------------------------------


def _pose_tab(svc):
    from .test_pose_keying import _rig

    doc, hip, arm = _rig()
    ctx = FakeCtx(svc)
    tab = clay_mode.adopt(ctx, doc, title="Rig")
    doc.select([hip])
    return ctx, tab, doc, hip


def _press(ctx, key_name: str, mod: int = 0) -> bool:
    import pygame

    event = pygame.event.Event(pygame.KEYDOWN, key=getattr(pygame, f"K_{key_name}"), mod=mod)
    return clay_mode.handle_key(ctx, event)


def test_pressing_i_in_pose_mode_inserts_a_key_on_the_selected_bone(svc) -> None:
    ctx, tab, doc, hip = _pose_tab(svc)
    doc.set_live_pose(hip, "rotation", Q90)
    assert doc.animation() is None
    assert _press(ctx, "i") is True
    assert doc.animation() is not None
    assert np.allclose(doc.animation().channels[hip]["rotation"].values[0], Q90)
    assert doc.history.top.label == "Insert Key"


def test_pressing_i_in_pose_mode_with_a_bone_selected_inserts_a_key(svc) -> None:
    ctx, tab, doc, hip = _pose_tab(svc)
    steps = len(doc.history)
    assert _press(ctx, "i") is True
    assert len(doc.history) == steps + 1 and doc.animation() is not None
    # Out of Pose mode the same key keys nothing: the op's own predicate says so.
    doc.set_pose_mode(False)
    steps = len(doc.history)
    _press(ctx, "i")
    assert len(doc.history) == steps


# -- clay-27: the read-back task does not own ``saving`` ---------------------------------------


def test_a_rig_read_landing_during_an_export_does_not_clear_saving(svc) -> None:
    ctx, tab, doc, ids = _queued(svc)
    tab.saving = True  # a Save or Export is writing the document on its task thread
    _land(ctx, tab)
    assert tab.saving, "the read-back never set the lock, so it must not clear it"
    clay_mode.on_task_failed(ctx, SimpleNamespace(key=f"clay-rigimport:{tab.uid}"))
    assert tab.saving, "nor may its failure"
    assert not any(o.skin is not None for o in doc.objects), "and nothing lands under the export"


# -- clay-28: Export Posed GLB carries a pose held without a key -------------------------------


def test_export_posed_includes_a_pose_held_without_a_key(tmp_path, monkeypatch) -> None:
    from realmspinner.studio import dialogs

    from .test_pose_keying import Q0, _add, _rig

    doc, hip, arm = _rig()
    _add(doc, "Body", mesh=weighted(bp.box()), skin=skin_for(hip))
    doc.key_pose({hip: {"rotation": Q0}})
    doc.set_live_pose(hip, "rotation", Q90)  # auto-key off: held, not keyed
    ctx = FakeCtx()
    tab = clay_mode.adopt(ctx, doc, title="Rig")
    out = tmp_path / "posed.glb"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: out)
    clay_mode.export_mesh_file(ctx, tab, "posed")
    model = gltf.load(out.read_bytes())
    joint = next(n for n in model.nodes if n.name == "Hip")
    assert np.allclose(np.asarray(joint.rotation, dtype="f8"), Q90, atol=1e-5)


# -- clay-29: the weights wait for a live drag --------------------------------------------------


def test_the_weights_wait_for_a_live_drag_to_finish_before_landing(svc) -> None:
    ctx, tab, doc, ids = _queued(svc)
    ctx.clay_view = SimpleNamespace(dragging=True)
    steps = len(doc.history)
    _land(ctx, tab)
    assert len(doc.history) == steps and tab.pending_rig is not None, "held, still tracked"
    clay_mode.pump_rig(ctx)
    assert len(doc.history) == steps, "still dragging"
    ctx.clay_view.dragging = False
    clay_mode.pump_rig(ctx)
    assert len(doc.history) == steps + 1 and tab.pending_rig is None
    assert any(o.skin is not None for o in doc.objects)
    assert any(m.startswith("Weighted") for m, _l in ctx.toasts)


# -- clay-31: a weighting run whose job was deleted --------------------------------------------


def test_a_weighting_run_whose_job_was_deleted_stops_being_tracked(svc, monkeypatch) -> None:
    ctx = Ctx(svc)
    tab, ids = _tab(ctx)
    monkeypatch.setattr(clay_mode, "GONE_GRACE_S", 0.0)
    tab.pending_rig = _pending(svc, ids)  # "rg": in neither the cache nor the store
    clay_mode.pump_rig(ctx)
    assert tab.pending_rig is None
    assert any("gone from the Library" in m and k == "error" for m, k in ctx.toasts)
    # ...and Weight with Blender is no longer refused as "already weighting".
    assert "already weighting" not in clay_mode.rig_blocked_why(ctx, tab)


def test_a_weighting_job_the_cache_has_not_loaded_but_the_store_has_is_still_read(
    svc, monkeypatch
) -> None:
    ctx = Ctx(svc)
    tab, ids = _tab(ctx)
    monkeypatch.setattr(clay_mode, "GONE_GRACE_S", 0.0)
    source = _row(svc)
    _write_rig(svc, source)
    done = svc.store.create("image", "", {}, "abcdef012345", stage="rig", status="done")
    pending = _pending(svc, ids, source_job=source)
    pending["rig_job"] = done
    tab.pending_rig = pending
    clay_mode.pump_rig(ctx)
    assert tab.pending_rig is not None
    assert ctx.submitted == [f"clay-rigimport:{tab.uid}"], "the store's done row is read"


# -- clay-46: a UV island dragged with the mouse is a live gesture -----------------------------


def test_ctrl_z_is_refused_while_a_uv_island_is_being_dragged_with_the_mouse(svc) -> None:
    import pygame

    ctx = FakeCtx(svc)
    from .test_clay_mode import _tab as plain_tab

    tab = plain_tab(ctx)
    doc = tab.doc
    doc.set_props(doc.objects[0].uid, name="One")
    doc.set_props(doc.objects[0].uid, name="Two")
    tab.uv_view.drag_mode = "move"
    assert clay_mode._gesture_live(ctx, tab)
    head = doc.history.head
    clay_mode.undo(ctx, tab)
    assert doc.history.head == head
    assert _press(ctx, "z", pygame.KMOD_CTRL) is True
    assert doc.history.head == head
    tab.uv_view.drag_mode = ""
    clay_mode.undo(ctx, tab)
    assert doc.history.head != head


def test_a_uv_island_translate_for_a_vanished_object_is_a_refusal_not_a_keyerror() -> None:
    from realmspinner.studio.modes.clay.ui.panes import uv

    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    assert uv.apply_translate(doc, 999_999, frozenset({0}), (0.1, 0.0)) is False


# -- clay-72: Import Animation names an unreadable GLB -----------------------------------------


def test_import_animation_names_an_unreadable_glb_instead_of_a_generic_failure(tmp_path) -> None:
    ctx = FakeCtx()
    doc, ids = _document()
    tab = clay_mode.adopt(ctx, doc, title="Hero")
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"glTF" + b"\x02\x00\x00\x00" + b"\xff" * 40)
    with pytest.raises(OpError, match="could not be read as a GLB"):
        clay_mode.import_animation(ctx, tab, bad)


# -- clay-73: a lost animation channel is a warning --------------------------------------------


def test_a_notice_about_left_out_animation_channels_raises_the_warning_level() -> None:
    sentence = (
        "3 animation channels drove something that is not a bone (or morph weights) and were "
        "left out."
    )
    assert "were left out." in (SRC / "kernels" / "mesh" / "rigimport.py").read_text(
        encoding="utf-8"
    ).replace('"\n            "', "")
    ctx = FakeCtx()
    clay_mode.announce_migration(ctx, SimpleNamespace(notices=(sentence,)))
    [(message, level)] = ctx.toasts
    assert level == "warn" and "left out" in message


# -- clay-74: the doors say why they will not start --------------------------------------------


def test_weight_with_blender_says_why_it_will_not_start_during_a_drag() -> None:
    from realmspinner.studio.modes.clay import doc_io

    ctx = Ctx()
    tab, _ = _tab(ctx)
    ctx.clay_view = SimpleNamespace(dragging=True)
    assert clay_mode.rig_blocked_why(ctx, tab) == doc_io.GESTURE_WHY
    clay_mode.rig_with_blender(ctx, tab)
    assert (doc_io.GESTURE_WHY, "error") in ctx.toasts and ctx.submitted == []


def test_import_animation_says_why_it_will_not_start_during_a_drag(monkeypatch) -> None:
    from realmspinner.studio import menus
    from realmspinner.studio.modes.clay import doc_io

    from .test_clay_compact_layout import _clay_menu_ctx

    ctx = FakeCtx()
    doc, ids = _document()
    tab = clay_mode.adopt(ctx, doc, title="Hero")
    ctx.clay_view = SimpleNamespace(dragging=True)
    clay_mode.import_animation(ctx, tab)
    assert (doc_io.GESTURE_WHY, "error") in ctx.toasts and ctx.submitted == []

    monkeypatch.setattr(clay_mode, "recent_paths", lambda ctx: [])
    menu_ctx = _clay_menu_ctx(tab)
    menu_ctx.clay_view = SimpleNamespace(dragging=True)
    rows = {r.identity: r for r in menus._clay_file_specs(menu_ctx, [])}
    row = rows["clay:import-animation"]
    assert not row.enabled and row.disabled_reason == doc_io.GESTURE_WHY


# -- clay-75 / service-assets-04: the sidecar is written whatever the rig does -----------------


def _bones_doc():
    doc, ids = _document()
    return doc


def test_a_rig_that_cannot_be_published_still_leaves_the_authored_sidecar_beside_the_mesh(
    svc, monkeypatch
) -> None:
    from realmspinner.service import files as svc_files
    from realmspinner.service.errors import Invalid

    def refuse(*a, **k):
        raise Invalid("a rig record needs bones", field="bones")

    monkeypatch.setattr(svc_files, "save_clay_rig", refuse)
    warnings: list[str] = []
    job_id = clay_mode.build_asset(svc, _bones_doc(), title="Hero", warnings=warnings)
    assert svc_files.clay_source_path(svc, job_id).is_file()
    assert len(warnings) == 1 and "rig" in warnings[0] and "needs bones" in warnings[0]


def test_a_refused_rig_still_leaves_the_clay_source_beside_the_mesh(svc, monkeypatch) -> None:
    from realmspinner.service import files as svc_files
    from realmspinner.service.errors import Invalid

    def refuse(*a, **k):
        raise Invalid("a rig record needs bones", field="bones")

    monkeypatch.setattr(svc_files, "save_clay_rig", refuse)
    before = {j["id"] for j in svc.store.list(limit=100)}
    with pytest.raises(Invalid):  # a caller that asked for no warnings still hears about it
        clay_mode.build_asset(svc, _bones_doc(), title="Hero")
    [row] = [j for j in svc.store.list(limit=100) if j["id"] not in before]
    assert svc_files.clay_source_path(svc, row["id"]).is_file()


def test_export_asset_reports_the_rig_that_was_left_out_and_still_adopts_the_row(
    svc, monkeypatch
) -> None:
    from realmspinner.service import files as svc_files
    from realmspinner.service.errors import Invalid

    monkeypatch.setattr(
        svc_files, "save_clay_rig", lambda *a, **k: (_ for _ in ()).throw(Invalid("no", field="x"))
    )
    ctx = FakeCtx(svc)
    doc, ids = _document()
    tab = clay_mode.adopt(ctx, doc, title="Hero")
    clay_mode.export_asset(ctx, tab)
    clay_mode.on_task_done(ctx, SimpleNamespace(key=ctx.submitted[-1], result=ctx.result))
    assert tab.job_id and not tab.saving
    assert any(level == "warn" and "rig" in m for m, level in ctx.toasts)


# -- clay-76: no comment names a removed mode ---------------------------------------------------


def test_clay_mode_comments_name_no_removed_mode() -> None:
    for relative in ("studio/modes/clay/mode.py", "studio/modes/clay/state.py"):
        text = (SRC / relative).read_text(encoding="utf-8")
        assert not re.search(r"\bPoser\b", text), relative
    doc = clay_mode.import_animation.__doc__ or ""
    assert "drop handler" not in doc
    assert "Poser" not in (clay_mode.pump_rig.__doc__ or "")


# -- pipelines-children-03: Blender fell back to envelope weights ------------------------------


def test_clay_says_so_when_blender_fell_back_to_envelope_weights(svc) -> None:
    ctx, tab, doc, ids = _queued(svc)
    tab.pending_rig["reading"] = False
    tab.pending_rig["weighting"] = None
    ctx.jobs[tab.pending_rig["rig_job"]] = {
        "status": "done",
        "params": {"weighting": "envelope", "weighting_reason": "bone heat could not solve"},
    }
    source = tab.pending_rig["source_job"]
    _write_rig(svc, source)
    clay_mode.pump_rig(ctx)
    ctx.toasts.clear()
    clay_mode.on_task_done(
        ctx,
        SimpleNamespace(key=ctx.submitted[-1], result=ctx.result),
    )
    lines = [(m, k) for m, k in ctx.toasts if "envelope" in m]
    assert lines and lines[0][1] == "warn" and "bone heat could not solve" in lines[0][0]
    assert any(m.startswith("Weighted") for m, _k in ctx.toasts)


def test_an_unskinned_envelope_rig_is_refused_with_blenders_reason(svc) -> None:
    ctx, tab, doc, ids = _queued(svc)
    tab.pending_rig["weighting"] = "envelope"
    tab.pending_rig["weighting_reason"] = "bone heat could not solve"
    bare = bd.ClayDoc()
    bare.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True))
    ctx.toasts.clear()
    _land(ctx, tab, bare)
    [(message, level)] = ctx.toasts
    assert level == "error" and "bone heat could not solve" in message
