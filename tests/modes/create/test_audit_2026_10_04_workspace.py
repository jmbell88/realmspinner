"""The 2026-10-04 audit's workspace findings: create-30, 38, 44, 45, 46, 47, 51, 60.

One file because the eight touch one surface (the creation key, the rail's
evidence, the tray) and share its fixtures. Every test name is the finding's
claim, and each was run against the unfixed code first.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from realmspinner.studio.modes.create.engine import workspace as families
from realmspinner.studio.modes.create.ui import preview, session, stages
from realmspinner.studio.modes.create.ui import workspace as generation_workspace
from realmspinner.studio.settings import Settings
from realmspinner.studio.state import AppState, default_form_2d


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


# --- create-30: a legacy creation's key does not drift ----------------------


def test_a_new_attempt_with_a_lower_sorting_id_keeps_the_legacy_creations_tray():
    """The 2026-10-04 audit found a legacy family's key was its smallest member,
    and job ids are random hex: a new attempt stamped with the family's key but
    whose own id sorts lower moved the root to itself, so the tray (which holds
    the old key) listed nothing."""
    old = [
        job("bbb", created_at=1),
        job("ccc", created_at=2, stage="model", kind="image", parent_id="bbb"),
    ]
    key = families.build_index(old).by_job["ccc"]
    assert key == "job:bbb"

    new = old + [job("aaa", created_at=3, params={families.WORKSPACE_PARAM: key})]
    index = families.build_index(new)

    assert index.by_job["aaa"] == index.by_job["ccc"] == key
    assert {j["id"] for j in families.results(index, key)} == {"aaa", "bbb", "ccc"}


def test_loading_older_history_re_derives_a_legacy_workspace_key_from_the_selection(tmp_path):
    """The audit's second route to the same drift: "Load older" brings in an
    unstamped sibling whose id sorts lower, which moves the root with no stamp
    to pin it. ``sync`` re-derives the key from the selection, and the draft
    filed under the dead key goes with it."""
    jobs = [
        job("bbb", created_at=2, parent_id="zzz"),
        job("ccc", created_at=3, stage="model", kind="image", parent_id="bbb"),
    ]
    ctx = ctx_for(jobs, tmp_path)
    ctx.state.select("ccc")
    session.sync(ctx)
    assert ctx.state.create.workspace == "job:bbb"
    ctx.state.form_2d["prompt"] = "an unsent draft"
    session.sync(ctx)
    assert ctx.state.create.drafts["job:bbb"]["form_2d"]["prompt"] == "an unsent draft"

    jobs.append(job("aaa", created_at=1, parent_id="zzz"))
    ctx.cache.jobs = jobs
    ctx.cache._generation = 2
    session.sync(ctx)

    new_key = session.index(ctx).by_job["ccc"]
    assert new_key == "job:aaa"
    assert ctx.state.create.workspace == new_key
    assert ctx.state.create.drafts[new_key]["form_2d"]["prompt"] == "an unsent draft"
    assert families.results(session.index(ctx), ctx.state.create.workspace)


# --- create-46: STAGE_ORDER is a checked copy -------------------------------


def _drift(engine_order, rail_stages):
    """What the engine's copy has that the rail lacks, and the reverse, and
    whether the order differs."""
    return (
        sorted(set(engine_order) - set(rail_stages)),
        sorted(set(rail_stages) - set(engine_order)),
        tuple(engine_order) != tuple(rail_stages),
    )


def test_engine_stage_order_matches_the_rail_stages():
    """The engine may not import ``ui``, so ``STAGE_ORDER`` is a copy; this is
    the gate that makes the copy honest. The 2026-10-04 audit found none, so a
    sixth rail stage would have been silently absent from ``journey()``."""
    assert _drift(families.STAGE_ORDER, stages.STAGES) == ([], [], False)
    assert tuple(stages.LABELS) == tuple(stages.STAGES)

    # The comparison can fail: a stage added to one side only, or two swapped.
    assert _drift(families.STAGE_ORDER, (*stages.STAGES, "bake"))[1] == ["bake"]
    assert _drift((*families.STAGE_ORDER, "bake"), stages.STAGES)[0] == ["bake"]
    swapped = list(stages.STAGES)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    assert _drift(families.STAGE_ORDER, swapped)[2] is True


# --- create-44: a two-stage rail can finish ---------------------------------


@pytest.mark.parametrize(
    ("stage", "asset_type", "files"),
    (
        ("tile", "seamless_material", ["input.png", "wrap_preview.png"]),
        ("tilesheet", "tileset", ["input.png"]),
        ("reference", "image", ["input.png"]),
        ("reference", "sprite_sheet", ["input.png"]),
    ),
)
def test_a_finished_tile_ticks_export_on_its_two_stage_rail(stage, asset_type, files):
    """The 2026-10-04 audit found ``ticked`` walked all five stages and stopped
    at Mesh, which a two-stage journey (image, material, tileset, sprite sheet)
    does not have, so Export never ticked on any 2D asset."""
    finished = job("t", stage=stage, files=files, params={"asset_type": asset_type})
    assert stages.ticked(finished) == frozenset({"reference", "export"})
    # The rail the user sees is the same two segments.
    assert families.journey(asset_type) == ("reference", "export")

    running = {**finished, "status": "running"}
    assert "export" not in stages.ticked(running)


def test_a_tile_row_with_no_recorded_asset_type_still_ticks_export():
    """A legacy tile row has no ``asset_type``; its stage alone says it has no
    Mesh (``available`` refuses one for the same reason)."""
    legacy = job("t", stage="tile", files=["input.png", "wrap_preview.png"])
    assert stages.ticked(legacy) == frozenset({"reference", "export"})


def test_a_3d_reference_still_ticks_only_reference():
    """The guard the rail's own tests keep: a reference that is the front of a
    3D journey has Mesh ahead of it, which is required."""
    three_d = job("r", params={"asset_type": "3d_model"})
    assert stages.ticked(three_d) == frozenset({"reference"})


# --- create-45: Pose is open whenever the rail ticks Rig --------------------


def test_pose_is_available_whenever_the_rail_ticks_rig(monkeypatch):
    """The 2026-10-04 audit found ``available("pose")`` asked ``_reached_rig``
    without the ``rig.json`` fallback ``ticked`` uses, so for the interval where
    ``rig.json`` has landed and the row does not yet list ``rig.glb`` the rail
    ticked Rig while Pose said "job is not rigged"."""
    mesh = job("m", stage="model", files=["model.glb", "input.png"])
    meta = {"joints": 20}
    assert "rig" in stages.ticked(mesh, meta, None)

    # Explicit evidence, the same the rail passes to ``ticked``.
    assert stages.available("pose", mesh, None, rig_meta=meta) is None
    assert stages.available("pose", mesh, None) == "job is not rigged"

    # And from the ctx alone, which is what the shell's caller has.
    from realmspinner.studio.panes import inspector

    monkeypatch.setattr(inspector, "rig_meta", lambda ctx, j: meta)
    ctx = SimpleNamespace(rigging_available=True, svc=object())
    assert stages.available("pose", mesh, ctx) is None


# --- create-47: a dropped sheet preview is asked for again ------------------


def test_a_preview_dropped_for_a_changed_selection_is_requested_again_on_return(tmp_path):
    """The 2026-10-04 audit found ``adopt`` dropped a result for a changed
    selection but left the requested key set, so returning to the job never
    resubmitted and the sheet preview stayed empty."""
    submitted: list[tuple] = []

    def submit(key, fn, *args, tag=None):
        submitted.append(tag)
        return True

    ctx = ctx_for([job("a"), job("b")], tmp_path)
    ctx.submit = submit
    ctx.job_dir = lambda job_id: tmp_path / job_id
    ctx.state.select("a")
    preview.request(ctx, job("a"))
    assert submitted == [("a", 1)]

    ctx.state.select("b")  # the user moved on before the load landed
    preview.adopt(ctx, SimpleNamespace(tag=("a", 1), result=[{"name": "old"}]))
    assert "create_preview" not in ctx.state.preview

    ctx.state.select("a")
    preview.request(ctx, job("a"))
    assert submitted == [("a", 1), ("a", 1)]


# --- create-38: the Rig card names the mesh before Blender ------------------


def test_rig_card_names_a_failed_mesh_before_missing_blender():
    """The 2026-10-04 audit found the reason was ``blocked or "A finished mesh
    is required."``, so with Blender missing a failed or running mesh said
    "needs Blender" -- true, but not what stood in the way."""
    blender = "Rigging needs Blender, which is not installed."
    failed = job("m", stage="model", status="error", error="out of memory", files=[])
    running = job("m", stage="model", status="running", files=[])
    no_glb = job("m", stage="model", status="done", files=["input.png"])
    ready = job("m", stage="model", status="done", files=["model.glb", "input.png"])

    reason = generation_workspace.rig_reason(failed, blender)
    assert "out of memory" in reason and "Blender" not in reason
    assert "not ready" in generation_workspace.rig_reason(running, blender)
    assert generation_workspace.rig_reason(no_glb, blender) == "A finished mesh is required."
    # A usable mesh is where Blender's own sentence is the honest one.
    assert generation_workspace.rig_reason(ready, blender) == blender
    assert generation_workspace.rig_reason(ready, None) is None


# --- create-60: the plan's duration follows the count -----------------------


@pytest.mark.parametrize("asset_type", ("image", "3d_model", "seamless_material"))
def test_plan_duration_scales_with_the_candidate_count(asset_type):
    """The 2026-10-04 audit found the image, 3D-reference and material arms
    hardcoded "a few seconds" whatever the Count, so eight candidates promised
    what one does. Derived from the generation count through the same phrase
    the tile and sprite arms use."""
    from realmspinner.service import sprites as svc_sprites
    from realmspinner.studio.modes.create.engine import assets as create_assets

    def plan(count):
        form = default_form_2d()
        form["asset_type"] = asset_type
        form["generation_type"] = asset_type
        form["count"] = count
        create_assets.sync_legacy_fields(form)
        return generation_workspace.plan_for(form)

    one, eight = plan(1), plan(8)
    assert one.duration == svc_sprites.generation_time_phrase(one.generations)
    assert eight.duration == svc_sprites.generation_time_phrase(
        8 * one.generations // one.candidates
    )
    assert eight.duration != one.duration
    assert "a few seconds" not in eight.duration


# --- create-51: the attempt strip draws only what is on screen --------------


def _fake_strip_modules(monkeypatch, *, scroll_x, view_w):
    imgui = MagicMock()
    imgui.begin_child.return_value = True
    imgui.is_item_clicked.return_value = False
    imgui.get_scroll_x.return_value = scroll_x
    imgui.get_window_width.return_value = view_w
    imgui.get_style.return_value.item_spacing.x = 8.0
    thumbs = MagicMock()
    controls = MagicMock()
    controls.button.return_value = False
    widgets = MagicMock()
    widgets.fit_text.side_effect = lambda text, width: text
    for name, fake in (
        ("imgui", imgui),
        ("thumbs", thumbs),
        ("controls", controls),
        ("widgets", widgets),
    ):
        monkeypatch.setattr(generation_workspace, name, fake)
    return imgui, thumbs


def _strip_ctx():
    state = SimpleNamespace(
        selected=None,
        source_job=None,
        create=SimpleNamespace(comparison_pin=None),
    )
    return SimpleNamespace(state=state)


def test_attempt_strip_draws_only_visible_attempts(monkeypatch):
    """The 2026-10-04 audit found ``_attempt_strip`` drew every attempt every
    frame and decoded each thumbnail on the frame thread. A long creation now
    pays for the cells in view (plus a margin) and fills the rest with dummies
    of the same footprint, so the scrollbar is what it would have been."""
    jobs = [job(f"j{i:04d}") for i in range(500)]
    imgui, thumbs = _fake_strip_modules(monkeypatch, scroll_x=0.0, view_w=800.0)
    generation_workspace._attempt_strip(_strip_ctx(), jobs)
    drawn = thumbs.job_thumb.call_count
    assert 0 < drawn < 40, "a 800px strip holds about nine 82px cells"
    first_ids = {c.args[1]["id"] for c in thumbs.job_thumb.call_args_list}
    cell = thumbs.job_thumb.call_args_list[0].args[2] + 8.0
    dummies = [c.args[0][0] for c in imgui.dummy.call_args_list]
    # What is drawn plus what is skipped is the whole strip, to the spacing.
    assert drawn * cell + sum(dummies) + 8.0 * len(dummies) == pytest.approx(500 * cell, abs=cell)

    # Scrolled far along, the window of drawn cells moves with it.
    imgui, thumbs = _fake_strip_modules(monkeypatch, scroll_x=100 * cell, view_w=800.0)
    generation_workspace._attempt_strip(_strip_ctx(), jobs)
    ids = {c.args[1]["id"] for c in thumbs.job_thumb.call_args_list}
    assert 0 < thumbs.job_thumb.call_count < 40
    assert "j0100" in ids and ids.isdisjoint(first_ids)


def test_a_short_attempt_strip_is_drawn_whole(monkeypatch):
    jobs = [job(f"j{i}") for i in range(generation_workspace.STRIP_CLIP_THRESHOLD - 1)]
    imgui, thumbs = _fake_strip_modules(monkeypatch, scroll_x=0.0, view_w=200.0)
    generation_workspace._attempt_strip(_strip_ctx(), jobs)
    assert thumbs.job_thumb.call_count == len(jobs)
    imgui.dummy.assert_not_called()
