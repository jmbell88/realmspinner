"""Asset-first history and drafts must survive browsing and repeat generation."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from realmspinner.studio import candidates
from realmspinner.studio.modes.create.engine import workspace
from realmspinner.studio.modes.create.ui import preview, session
from realmspinner.studio.settings import Settings
from realmspinner.studio.state import AppState, Filters


def job(key, *, params=None, stage="reference", kind="text", **kwargs):
    return {
        "id": key,
        "kind": kind,
        "stage": stage,
        "status": "done",
        "params": params or {},
        "files": ["input.png"],
        "prompt": "a barrel",
        **kwargs,
    }


def ctx_for(jobs, tmp_path):
    by_id = {j["id"]: j for j in jobs}
    state = AppState(mode="create")
    return SimpleNamespace(
        state=state,
        cache=SimpleNamespace(jobs=jobs, get=by_id.get, _generation=1),
        settings=Settings.load(tmp_path),
    )


def test_attempts_follow_workspace_lineage_and_exclude_unrelated_assets():
    jobs = [
        job("a", params={"create_workspace": "asset"}),
        job("b", params={"create_workspace": "asset"}),
        job("mesh", kind="image", stage="model", parent_id="a"),
        job("reroll", params={"rerun_of": "b"}),
        job("rig", kind="rig", params={"source_job": "mesh"}),
        job("other"),
        job("music", kind="music"),
    ]
    index = workspace.build_index(jobs)
    assert index.by_job["mesh"] == index.by_job["reroll"] == "creation:asset"
    assert {j["id"] for j in workspace.results(index, "creation:asset")} == {
        "a",
        "b",
        "mesh",
        "reroll",
    }
    assert {j["id"] for j in workspace.results(index, "creation:asset", "mesh")} == {"mesh"}
    assert "music" not in index.by_job


def test_missing_ancestors_and_cycles_do_not_lose_sibling_attempts():
    jobs = [
        job("a", parent_id="missing"),
        job("b", parent_id="missing"),
        job("c", parent_id="d"),
        job("d", parent_id="c"),
    ]
    index = workspace.build_index(jobs)
    assert index.by_job["a"] == index.by_job["b"]
    assert index.by_job["c"] == index.by_job["d"]
    assert index.by_job["a"] != index.by_job["c"]


def test_every_candidate_is_saved_visible_and_does_not_block_the_next_batch():
    jobs = [
        job(
            str(i),
            params={"create_workspace": "asset"},
            stage="model",
            candidate_group=f"batch-{i // 3}",
        )
        for i in range(9)
    ]
    index = workspace.build_index(jobs)
    assert len(workspace.results(index, "creation:asset", "mesh")) == 9
    assert candidates.pending(jobs) is None
    assert all(Filters().matches(j) for j in jobs)
    assert candidates.pending([job("legacy", candidate_group="old")]) is not None


def test_switching_creations_restores_each_draft_and_explicit_source(tmp_path):
    a = job("a", params={"create_workspace": "a", "asset_type": "image"})
    b = job("b", params={"create_workspace": "b", "asset_type": "image"})
    ctx = ctx_for([a, b], tmp_path)
    session.resume(ctx, a)
    ctx.state.select("a")
    ctx.state.form_2d["prompt"] = "unfinished changes"
    ctx.state.source_job = "a"
    session.resume(ctx, b)
    ctx.state.select("b")
    ctx.state.form_2d["prompt"] = "a second draft"
    session.resume(ctx, a)
    assert ctx.state.form_2d["prompt"] == "unfinished changes"
    assert ctx.state.source_job == "a"


def test_workspace_draft_is_restorable_after_restart_without_reusing_a_seed(tmp_path):
    a = job("a", params={"create_workspace": "a", "asset_type": "image"})
    ctx = ctx_for([a], tmp_path)
    session.resume(ctx, a)
    ctx.state.form_2d.update(prompt="work in progress", seed=123)
    session.save_draft(ctx)
    ctx.settings.flush()
    restored = ctx_for([a], tmp_path)
    session.resume(restored, a)
    assert restored.state.form_2d["prompt"] == "work in progress"
    assert "seed" not in restored.settings.get(session.SETTINGS_KEY)["creation:a"]["form_2d"]


def test_browsing_a_result_does_not_replace_draft_or_source(tmp_path, monkeypatch):
    from realmspinner.studio import asset_open
    from realmspinner.studio.modes.create.ui import stages
    from realmspinner.studio.modes.create.ui import workspace as surface

    a, b = job("a"), job("b")
    ctx = ctx_for([a, b], tmp_path)
    ctx.state.source_job = "a"
    ctx.state.form_2d["prompt"] = "my draft"
    before = copy.deepcopy(ctx.state.form_2d)

    def go(ctx, stage, *, select=None, **kwargs):
        ctx.state.select(select)
        ctx.state.source_job = select  # Legacy selection side effect.

    monkeypatch.setattr(stages, "go", go)
    surface.inspect_result(ctx, b)
    assert ctx.state.selected == "b"
    assert ctx.state.source_job == "a"
    assert ctx.state.form_2d == before
    assert asset_open.route(b).stage == "reference"


@pytest.mark.parametrize("kind", ["image", "tileset", "sprite_sheet", "seamless_material"])
def test_non_mesh_outputs_have_a_short_journey(kind):
    assert workspace.journey(kind) == ("reference", "export")
    assert "rig" in workspace.journey(kind, has_mesh=True)


def test_playback_obeys_per_frame_durations_and_one_shot_policy():
    record = {
        "cells": [{}, {}, {}],
        "animation": {
            "frames": [
                {"cell_index": 0, "duration_ms": 50},
                {"cell_index": 1, "duration_ms": 200},
                {"cell_index": 2, "duration_ms": 100},
            ]
        },
    }
    tag = {"start": 0, "end": 2, "loop": True}
    assert preview.frame_at(record, tag, 0.049) == 0
    assert preview.frame_at(record, tag, 0.05) == 1
    assert preview.frame_at(record, tag, 0.25) == 2
    assert preview.frame_at(record, tag, 0.35) == 0
    assert preview.frame_at(record, {**tag, "loop": False}, 8) == 2


def test_stale_preview_load_cannot_land_after_selection_changes(tmp_path):
    ctx = ctx_for([job("a"), job("b")], tmp_path)
    ctx.state.select("b")
    ctx.state.preview["create_preview_requested"] = ("a", 1)
    preview.adopt(ctx, SimpleNamespace(tag=("a", 1), result=[{"name": "old"}]))
    assert "create_preview" not in ctx.state.preview


def test_opening_a_creation_restores_its_prompt_and_mesh_settings(tmp_path):
    reference = job("a", params={"create_workspace": "asset", "asset_type": "3d_model"})
    mesh = job("m", stage="model", parent_id="a", params={"mesh_seed": 19, "resolution": "1024"})
    ctx = ctx_for([reference, mesh], tmp_path)
    session.resume(ctx, mesh)
    assert ctx.state.form_2d["prompt"] == "a barrel"
    assert ctx.state.source_job == "a"
    assert ctx.state.form_3d["mesh_seed"] == 19


def test_playback_follows_exported_reverse_and_pingpong_tags():
    record = {"cells": [{}, {}, {}]}
    tag = {"start": 0, "end": 2, "direction": "reverse"}
    assert preview.frame_at(record, tag, 0) == 2
    assert preview.frame_at(record, tag, 0.2) == 0
    tag["direction"] = "pingpong"
    assert [preview.frame_at(record, tag, t) for t in (0, 0.1, 0.2, 0.3, 0.4)] == [0, 1, 2, 1, 0]


def test_recorded_sprite_request_restores_action_and_exported_cell_size(tmp_path):
    from realmspinner.generation import GenerationRequest, SpriteSettings

    request = GenerationRequest(
        generation_type="sprite_sheet",
        prompt="a mage",
        sprite=SpriteSettings(
            mode="action", action="walk", directions=4, target_cell_px=32, candidate_count=2
        ),
    )
    reference = job(
        "a", params={"generation_type": "sprite_sheet", "generation_request": request.to_dict()}
    )
    ctx = ctx_for([reference], tmp_path)
    session.resume(ctx, reference)
    assert ctx.state.form_2d["sheet_layout"] == "walk4"
    assert ctx.state.form_2d["target_cell_px"] == "32"


def test_pose_guard_defers_draft_and_selection_until_the_confirm(tmp_path):
    from realmspinner.studio import asset_open

    a, b = job("a"), job("b")
    ctx = ctx_for([a, b], tmp_path)
    session.resume(ctx, a)
    ctx.state.select("a")
    ctx.state.create.stage = "pose"
    ctx.state.form_2d["prompt"] = "unfinished draft"
    confirms = []
    ctx.confirms = SimpleNamespace(ask=confirms.append)
    ctx.viewer = SimpleNamespace(
        pose_mode=True, editor=SimpleNamespace(mode="pose", has_unsaved_edits=lambda: True)
    )
    ctx.viewer.exit_pose_mode = lambda: setattr(ctx.viewer, "pose_mode", False)
    asset_open.open_asset(ctx, b)
    assert ctx.state.selected == "a"
    assert ctx.state.form_2d["prompt"] == "unfinished draft"
    assert len(confirms) == 1
    confirms[0].on_confirm()
    assert ctx.state.selected == "b"
    assert ctx.state.create.drafts["job:a"]["form_2d"]["prompt"] == "unfinished draft"


def test_switching_creations_by_library_selection_does_not_move_the_old_drafts_source(tmp_path):
    """create-01: ``library.select`` overwrites ``source_job`` before
    ``session.sync`` calls ``resume``, whose first act filed that under the OLD
    creation -- so going back restored another creation's reference."""
    a = job("a", params={"create_workspace": "a", "asset_type": "image"})
    b = job("b", params={"create_workspace": "b", "asset_type": "image"})
    ctx = ctx_for([a, b], tmp_path)
    ctx.state.select("a")
    session.sync(ctx)  # adopts a
    ctx.state.source_job = "a"
    session.sync(ctx)

    ctx.state.select("b")  # what library.select does, before sync runs
    ctx.state.source_job = "b"
    session.sync(ctx)
    assert ctx.state.source_job == "b"

    ctx.state.select("a")
    ctx.state.source_job = "a"
    session.sync(ctx)

    assert ctx.state.source_job == "a"


@pytest.mark.parametrize("entry", ["start_2d", "start_3d"])
def test_home_new_image_starts_a_new_creation_and_leaves_the_previous_draft_alone(
    tmp_path, monkeypatch, entry
):
    """create-02: Home's New 2D/3D replaced the form but left ``workspace`` set,
    so the next save_draft overwrote the previous creation's draft with the
    blank form and the next press was filed under the old creation."""
    from realmspinner.studio.modes.create.ui import stages
    from realmspinner.studio.modes.home.ui.panes import landing

    a = job("a", params={"create_workspace": "a", "asset_type": "image"})
    ctx = ctx_for([a], tmp_path)
    session.resume(ctx, a)
    ctx.state.select("a")
    ctx.state.form_2d["prompt"] = "unsent draft"
    session.save_draft(ctx)
    monkeypatch.setattr(landing, "_create_door", lambda ctx: True)
    monkeypatch.setattr(stages, "go", lambda *args, **kwargs: None)

    getattr(landing, entry)(ctx)
    session.sync(ctx)

    assert ctx.state.create.workspace is None
    assert ctx.state.create.drafts["creation:a"]["form_2d"]["prompt"] == "unsent draft"


def test_generating_into_a_legacy_creation_keeps_its_earlier_attempts_in_the_tray(tmp_path):
    """create-05: a pre-existing job has no create_workspace, so ``metadata``
    stamped the unprefixed root while keeping the state key as
    ``creation:<root>`` -- and the new row joined a node nothing else hung from."""
    a = job("a", created_at=1)
    b = job("b", created_at=2, parent_id="a")
    ctx = ctx_for([a, b], tmp_path)
    session.resume(ctx, b)
    stamped = session.metadata(ctx)

    c = job("c", created_at=3, params=dict(stamped))
    ctx.cache.jobs.append(c)
    index = workspace.build_index(ctx.cache.jobs)

    assert index.by_job["c"] == index.by_job["a"] == index.by_job["b"]
    assert session.metadata(ctx) == stamped, "the stamp is stable for the same creation"


def test_reproduce_recorded_attempt_uses_the_seed_that_shipped_not_the_requested_one():
    """create-06: after a reference reroll ``seed`` is the refused first draw
    and ``reference_seed`` is the one that produced input.png."""
    from realmspinner.studio.modes.create.ui import workspace as surface

    redrawn = job("r", params={"seed": 1, "reference_seed": 7})
    assert surface.recorded_seed(redrawn) == 7
    mesh = job("m", stage="model", params={"seed": 1, "reference_seed": 7, "mesh_seed": 9})
    assert surface.recorded_seed(mesh) == 9
    legacy = job("l", params={"seed": 5})
    assert surface.recorded_seed(legacy) == 5
