"""Findings from the 2026-09-26 audit closed in this pass (w2f8).

Each test reproduces one finding against the unfixed code and pins the fix:
service-assets-01 (export.py), service-kinds-02 (a ``sheet`` reroll),
poser-jobs-03 (the same reroll's missing cap/lock), service-assets-03 (the
same missing lock on ``adjust_joints``/``edit_skeleton``), service-gates-02
(``check_glb``'s untrusted mesh walk), service-kinds-03 (a sprite cell size
silently coerced), service-kinds-05 (``prune_jobs`` counting trashed rows
toward ``keep``) and service-queue-03 (``unverdicted_models`` counting
non-mesh and trashed rows).
"""

from __future__ import annotations

import json
import struct
import threading

import pytest

from realmspinner.kernels.geom3d.glbio import CHUNK_JSON, GLB_MAGIC
from realmspinner.kernels.rig import store as rig_store
from realmspinner.service import export as svc_export
from realmspinner.service import jobs as svc_jobs
from realmspinner.service import rig as svc_rig
from realmspinner.service import validation as svc_validation
from realmspinner.service.errors import Conflict, Invalid


@pytest.fixture
def assets(svc):
    return svc.config.data_dir


# --- service-assets-01: Keep both under a suffixed name ----------------------


def test_export_planned_to_folder_writes_a_keep_both_plan_under_the_suffixed_names(
    svc, assets, tmp_path
):
    """``export_planned_to_folder`` matched ``plan.files`` (suffixed, once a
    plan has been through ``keep_both``) against ``collect()``'s fresh,
    un-suffixed arcnames -- so a "Keep both" always found the two sets
    disjoint and refused with "the export plan is out of date", even though
    nothing about the job had actually changed since the plan was drawn."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fresh-glb")
    svc.store.set_status(job_id, "done")

    dest = tmp_path / "project" / "assets"
    (dest / job_id).mkdir(parents=True)
    (dest / job_id / "model.glb").write_bytes(b"already-here")

    job = svc_export.ExportJob(svc=svc, ids=[job_id], names_wanted=None, as_zip=False)
    plan = svc_export.plan_export(job, dest)
    kept = svc_export.keep_both(plan)

    result = svc_export.export_planned_to_folder(svc, [job_id], None, kept)

    assert result["copied"] == 1
    assert (dest / job_id / "model-2.glb").read_bytes() == b"fresh-glb"
    # The original, un-suffixed file is what "Keep both" promises not to touch.
    assert (dest / job_id / "model.glb").read_bytes() == b"already-here"


# --- service-kinds-02: a sheet reroll needs a fresh sheet_id -----------------


def _mesh_job(svc) -> str:
    mesh_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    svc.store.set_status(mesh_id, "done")
    return mesh_id


def test_a_sheet_reroll_mints_a_fresh_sheet_id(svc):
    """``rerun_job`` strips ``sheet_id`` (it is on ``DERIVED_PARAMS``) and used
    to re-seed it only for ``pixel_sheet``/``charsheet`` -- so a reroll of a
    plain ``sheet`` row minted a row with no ``sheet_id`` at all, which
    ``_q_rig.Worker._sheet`` refuses at dispatch with "sheet_id is not a
    sheet id: ''", even though ``rerollable`` says a sheet is rerollable."""
    mesh_id = _mesh_job(svc)
    old_sheet_id = rig_store.new_id()
    job_id = svc.store.create(
        "sheet", "a barrel", {"source_job": mesh_id, "sheet_id": old_sheet_id, "seed": 1}
    )
    svc.store.set_status(job_id, "done")

    new = svc_jobs.rerun_job(svc, job_id, mode="reroll")

    row = svc.store.get(new["id"])
    assert row["kind"] == "sheet"
    assert rig_store.is_valid_id(row["params"]["sheet_id"]), (
        f"a rerolled sheet must carry a fresh, valid sheet_id, got "
        f"{row['params'].get('sheet_id')!r}"
    )
    assert row["params"]["sheet_id"] != old_sheet_id


# --- poser-jobs-03: the same reroll needs the cap and the lock ---------------


def test_rerunning_a_charsheet_at_the_sheet_cap_is_refused(svc, assets, monkeypatch):
    """``rerun_job`` mints a fresh ``sheet_id`` for a ``charsheet``/``sheet``
    reroll (above) but used to insert the new row with no job-wide hold and no
    ``MAX_SHEETS`` check at all -- unlike every other door onto the same pool
    (``sheets.create_sheet``, ``troupe.create_charsheet``) -- so a reroll
    could push a mesh's sheet count straight past its cap with no refusal.

    The cap is occupied by a sheet already *published* to disk rather than by
    a second queued job: a queued sibling would instead be caught by
    ``_require_no_dependents`` (a different, earlier refusal), and the
    finding is specifically that nothing catches the published case.
    """
    monkeypatch.setattr(rig_store, "MAX_SHEETS", 1)
    mesh_id = _mesh_job(svc)
    job_dir = assets / mesh_id
    job_dir.mkdir(parents=True, exist_ok=True)
    sid = rig_store.new_id()
    rig_store.sheet_dir(job_dir).mkdir(parents=True, exist_ok=True)
    rig_store.sheet_path(job_dir, sid).write_text("{}", encoding="utf-8")
    rig_store.sheet_png_path(job_dir, sid).write_bytes(b"png")

    job_id = svc.store.create(
        "charsheet", "a barrel", {"source_job": mesh_id, "sheet_id": sid}
    )
    svc.store.set_status(job_id, "done")

    with pytest.raises(Conflict, match="at most 1 sheet"):
        svc_jobs.rerun_job(svc, job_id, mode="reroll")


def test_rerunning_a_sprite_synthesis_draft_at_the_cap_is_refused(svc, assets, monkeypatch):
    """The same incident's other pool: a ``sprite_synthesis`` reroll mints a
    fresh ``draft_id`` but used to skip ``check_sprite_draft_cap`` and its
    ``convert_lock`` entirely."""
    from realmspinner.service import sprites as svc_sprites

    monkeypatch.setattr(rig_store, "MAX_SPRITE_DRAFTS", 1)
    # The weights gate is a separate concern from the cap/lock this test is
    # about; sidestepped the way sibling doors are refused independently
    # elsewhere, by not asking this door to also prove weights are present.
    monkeypatch.setattr(svc_sprites, "_check_weights", lambda svc: None)

    ref_id = svc.store.create("text", "a knight", {}, stage="reference")
    svc.store.set_status(ref_id, "done")
    ref_dir = assets / ref_id
    ref_dir.mkdir(parents=True, exist_ok=True)
    did = rig_store.new_id()
    rig_store.sprite_dir(ref_dir).mkdir(parents=True, exist_ok=True)
    rig_store.sprite_draft_path(ref_dir, did).write_text(
        json.dumps({"candidates": ["a"]}), encoding="utf-8"
    )
    rig_store.sprite_draft_png_path(ref_dir, did, "a").write_bytes(b"png")

    job_id = svc.store.create(
        "sprite_synthesis",
        "a knight",
        {"source_job": ref_id, "draft_id": rig_store.new_id(), "seed_a": 1, "seed_b": 2},
    )
    svc.store.set_status(job_id, "done")

    with pytest.raises(Conflict, match="sprite"):
        svc_jobs.rerun_job(svc, job_id, mode="reroll")


# --- service-assets-03: adjust_joints/edit_skeleton share create_rig's lock --


def _rigged_job(svc, assets) -> tuple[str, list[dict]]:
    """A rig.json with everything either ``adjust_joints`` or
    ``edit_skeleton`` reads -- ``tests/service/test_rig_api.py``'s
    ``_rigged_job_full`` shape, duplicated locally per this brief's own rule
    (each fixer's file list is its own)."""
    from realmspinner.kernels.rig import skeleton, templates

    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    svc.store.set_status(job_id, "done")
    template = templates.get_template("humanoid")
    fitted = skeleton.fit_template(template, [-1, -1, 0], [1, 1, 2])
    rig = {
        "version": 1,
        "template": "humanoid",
        "bones": fitted,
        "root": template.root,
        "mirror_pairs": [list(p) for p in template.mirror_pairs],
        "bounds": {"min": [-1, -1, 0], "max": [1, 1, 2]},
        "skeleton": "template",
    }
    (job_dir / "rig.json").write_text(json.dumps(rig), encoding="utf-8")
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    return job_id, fitted


def test_adjust_joints_and_edit_skeleton_share_the_rig_lock_with_create_rig(
    svc, assets, monkeypatch
):
    """``create_rig`` checks ``rig_in_flight`` and inserts under
    ``svc.convert_lock(job_id, "rig")``; ``adjust_joints`` (and
    ``edit_skeleton`` the same way) did the identical check-then-insert as two
    separate, unlocked operations -- so two joint-move requests for one mesh
    landing on AgentHost's two service workers (the same shape of race
    agents-03 found for ``create_rig`` itself) could both read "no rig in
    flight" and both mint one.

    Proven deterministically by holding the lock ourselves: if
    ``adjust_joints`` takes ``svc.convert_lock(job_id, "rig")`` the way
    ``create_rig`` does, a call made while this test already holds that same
    lock must block until it is released. Pre-fix, ``adjust_joints`` never
    touches the lock at all and runs straight through -- ``doctor.blender_check``
    is pinned fast (the same move
    ``test_a_concurrent_character_rig_and_character_sheet_create_call_mint_only_one_rig_row``
    makes) so a slow *first* real probe cannot be mistaken for the lock.
    """
    from types import SimpleNamespace

    monkeypatch.setattr(
        "realmspinner.doctor.blender_check", lambda *a, **k: SimpleNamespace(ok=True, detail="")
    )
    job_id, fitted = _rigged_job(svc, assets)
    payload = {
        "bones": [{"name": b["name"], "head": b["head"], "tail": b["tail"]} for b in fitted]
    }

    lock = svc.convert_lock(job_id, "rig")
    result: dict = {}
    lock.acquire()
    try:
        thread = threading.Thread(
            target=lambda: result.__setitem__("out", svc_rig.adjust_joints(svc, job_id, payload))
        )
        thread.start()
        thread.join(timeout=0.5)
        assert thread.is_alive(), (
            "adjust_joints returned while this test still held "
            'svc.convert_lock(job_id, "rig") -- it never took the same lock '
            "create_rig does"
        )
    finally:
        lock.release()
    thread.join(5)
    assert not thread.is_alive()
    assert "out" in result and result["out"]["id"]


def test_edit_skeleton_shares_the_rig_lock_with_create_rig(svc, assets, monkeypatch):
    """``edit_skeleton``'s half of the same finding."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        "realmspinner.doctor.blender_check", lambda *a, **k: SimpleNamespace(ok=True, detail="")
    )
    job_id, fitted = _rigged_job(svc, assets)

    lock = svc.convert_lock(job_id, "rig")
    result: dict = {}
    lock.acquire()
    try:
        thread = threading.Thread(
            target=lambda: result.__setitem__(
                "out", svc_rig.edit_skeleton(svc, job_id, {"bones": fitted})
            )
        )
        thread.start()
        thread.join(timeout=0.5)
        assert thread.is_alive(), (
            "edit_skeleton returned while this test still held "
            'svc.convert_lock(job_id, "rig") -- it never took the same lock '
            "create_rig does"
        )
    finally:
        lock.release()
    thread.join(5)
    assert not thread.is_alive()
    assert "out" in result and result["out"]["id"]


# --- service-gates-02: check_glb's mesh walk is inside the try ---------------


def _glb(document: dict) -> bytes:
    """A structurally valid GLB carrying whatever JSON is handed over."""
    payload = json.dumps(document).encode()
    payload += b" " * (-len(payload) % 4)
    body = struct.pack("<II", len(payload), CHUNK_JSON) + payload
    return struct.pack("<III", GLB_MAGIC, 2, 12 + len(body)) + body


def test_check_glb_refuses_a_glb_whose_meshes_are_not_objects():
    """``check_glb`` wrapped only ``read_glb`` in its try/except -- the mesh
    walk right after it (``mesh.get("primitives")`` for every entry of
    ``doc["meshes"]``) ran unguarded, so a structurally valid GLB whose
    ``"meshes"`` list held a non-object (``[1]``) escaped this door as a raw
    ``AttributeError`` instead of the ``Invalid(field="glb")`` it promises."""
    data = _glb({"asset": {"version": "2.0"}, "meshes": [1]})

    with pytest.raises(Invalid) as excinfo:
        svc_validation.check_glb(data)
    assert excinfo.value.field == "glb"


# --- service-kinds-03: a sprite cell size is refused, not coerced -----------


def test_a_sprite_request_for_an_unbuildable_cell_size_is_refused_not_coerced(svc):
    """``create_generation_request`` coerced any ``sprite.target_cell_px``
    outside ``(32, 48, 64)`` to 64, though ``validate_request`` accepts
    ``generation.TARGET_CELL_MIN``..``_MAX`` (8-256) and the request document
    went on recording the value actually asked for -- so "make me 50px
    cells" published a 64px sheet while the request still said 50."""
    raw = {
        "generation_type": "sprite_sheet",
        "prompt": "a knight",
        "sprite": {
            "mode": "action",
            "action": "walk",
            "directions": 8,
            "target_cell_px": 50,
        },
    }
    with pytest.raises(Invalid) as excinfo:
        svc_jobs.create_generation_request(svc, raw)
    assert excinfo.value.field == "sprite.target_cell_px"
    assert "50" in str(excinfo.value)


# --- service-kinds-05: prune counts only live jobs toward keep --------------


def test_prune_counts_only_live_jobs_toward_keep(svc):
    """``store.list`` returns every job newest-first, trashed or not -- so
    ``prune_jobs``'s "seen <= keep" counter used to spend a protected "newest
    N" slot on a trashed row. Two trashed jobs newer than the one live asset
    that must survive a ``keep=1`` prune consumed that one slot between them,
    and the live asset -- the actual newest *live* job -- was archived and
    deleted."""
    live_id = svc_jobs.create_job(svc, kind="text", prompt="keep me")["id"]
    svc.store.set_status(live_id, "done")
    for i in range(2):
        jid = svc_jobs.create_job(svc, kind="text", prompt=f"trash{i}")["id"]
        svc.store.set_status(jid, "done")
        svc_jobs.trash_job(svc, jid)

    svc_jobs.prune_jobs(svc, keep=1)

    assert svc.store.get(live_id) is not None, (
        "the newest live job must survive a keep=1 prune even with two newer "
        "trashed rows ahead of it in store.list()'s newest-first order"
    )


# --- service-queue-03: unverdicted_models skips non-mesh and trashed rows ---


def test_unverdicted_models_skips_rig_rows_and_trashed_rows(svc):
    """``unverdicted_models`` filtered on ``stage = 'model'`` alone -- every
    one of ``followups.PRODUCTS``' derived kinds (rig, sheet, pixel_sheet,
    sprite_synthesis, charsheet, retexture, remesh, separate) defaults to
    that same stage (``JobStore.create``'s own default), so a finished rig
    surfaced here as an "unjudged mesh" nobody had ever asked to be judged --
    and, with no ``deleted_at`` filter at all, so did a trashed mesh."""
    mesh_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    svc.store.set_status(mesh_id, "done")

    rig_id = svc.store.create("rig", "a barrel", {"source_job": mesh_id})
    svc.store.set_status(rig_id, "done")

    trashed_id = svc_jobs.create_job(svc, kind="text", prompt="trash me")["id"]
    svc.store.set_status(trashed_id, "done")
    svc_jobs.trash_job(svc, trashed_id)

    ids = {row["id"] for row in svc.store.unverdicted_models()}
    assert mesh_id in ids
    assert rig_id not in ids, "a rig row is not a mesh anyone asked to be judged"
    assert trashed_id not in ids, "a trashed mesh is not sitting in front of anyone to judge"
