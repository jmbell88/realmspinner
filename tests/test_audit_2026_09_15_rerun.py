"""Regressions from the 2026-09-15 audit's rerun/reroll table.

One file per finding group, closed by the fixer that owns the file the
finding names. This module holds the ones whose regression test does not
belong beside an existing rerun test file that this session does not own.
"""

from __future__ import annotations

from realmspinner.pipelines import tileatlas
from realmspinner.service import jobs as svc_jobs
from realmspinner.service import tilesheets

# --- service-01: a materials/terrain reroll must draw fresh per-material and
# mask seeds, not just a new top-level one nothing on this kind reads --------


def _materials_job(svc, **overrides):
    """A finished ``materials``-mode tile sheet, built through the real door
    so ``params["sheet"]`` is exactly what a live request would write."""
    kwargs = {
        "prompt": "a temperate ruin, muted palette",
        "tile_size": 32,
        "view": "top_down",
        "mode": tilesheets.MODE_MATERIALS,
        "prompt_items": ("mossy cobblestone", "cracked dry mud", "still dark water"),
    }
    kwargs.update(overrides)
    made = tilesheets.create_tile_sheet(svc, **kwargs)
    svc.store.set_status(made["id"], "done")
    return made["id"]


def _terrain_job(svc, **overrides):
    """A finished ``terrain``-mode tile sheet, same reason as above."""
    kwargs = {
        "prompt": "a temperate coastline",
        "tile_size": 32,
        "view": "top_down",
        "mode": tilesheets.MODE_TERRAIN,
        "inner_terrain": "short green grass",
        "outer_terrain": "shallow blue water",
    }
    kwargs.update(overrides)
    made = tilesheets.create_tile_sheet(svc, **kwargs)
    svc.store.set_status(made["id"], "done")
    return made["id"]


def test_reroll_of_a_materials_tile_sheet_draws_fresh_material_seeds(svc):
    """Before this fix, ``rerun_job`` rerolled only ``params["seed"]`` -- a key
    the ``materials`` worker never reads -- and copied ``params["sheet"]``
    through unchanged, so a reroll spent N full generations and republished
    byte-identical tiles (the 2026-09-15 audit, service-01)."""
    job_id = _materials_job(svc)
    source_seeds = [
        m["seed"] for m in svc.store.get(job_id)["params"]["sheet"]["materials"]
    ]

    new = svc_jobs.rerun_job(svc, job_id, mode="reroll")
    new_row = svc.store.get(new["id"])
    new_seeds = [m["seed"] for m in new_row["params"]["sheet"]["materials"]]

    assert new_seeds != source_seeds, "the nested material seeds came through unchanged"
    # Re-derived exactly as the door derives them: same function, from the
    # fresh top-level seed this reroll minted.
    assert new_seeds == list(tileatlas.material_seeds(new_row["params"]["seed"], len(new_seeds)))
    # The prompts and variants are untouched -- only the seed the worker draws
    # each material's generation from moves.
    source_row = svc.store.get(job_id)
    for old, new_entry in zip(
        source_row["params"]["sheet"]["materials"],
        new_row["params"]["sheet"]["materials"],
        strict=True,
    ):
        assert new_entry["prompt"] == old["prompt"]
        assert new_entry["variant"] == old["variant"]


def test_reroll_of_a_terrain_tile_sheet_draws_a_fresh_mask_seed(svc):
    """The terrain mode's other nested seed: the boundary field is drawn from
    ``sheet["mask"]["seed"]``, which the door writes as the sheet's own
    request seed -- so a reroll that skips it redraws the identical
    coastline over two freshly (and pointlessly) regenerated materials."""
    job_id = _terrain_job(svc)
    source_mask_seed = svc.store.get(job_id)["params"]["sheet"]["mask"]["seed"]
    source_material_seeds = [
        m["seed"] for m in svc.store.get(job_id)["params"]["sheet"]["materials"]
    ]

    new = svc_jobs.rerun_job(svc, job_id, mode="reroll")
    new_row = svc.store.get(new["id"])
    new_mask_seed = new_row["params"]["sheet"]["mask"]["seed"]
    new_material_seeds = [m["seed"] for m in new_row["params"]["sheet"]["materials"]]

    assert new_mask_seed != source_mask_seed
    assert new_mask_seed == new_row["params"]["seed"]
    assert new_material_seeds != source_material_seeds
    assert new_material_seeds == list(tileatlas.material_seeds(new_row["params"]["seed"], 2))


def test_reroll_of_a_grid_tile_sheet_still_works_with_no_sheet_materials(svc):
    """The grid mode's ``sheet`` block carries no per-material seeds at all --
    this fix must not assume every tile sheet has one."""
    made = tilesheets.create_tile_sheet(
        svc,
        prompt="mossy dungeon",
        tile_size=32,
        view="top_down",
        mode=tilesheets.MODE_GRID,
        allow_grid=True,
    )
    svc.store.set_status(made["id"], "done")

    new = svc_jobs.rerun_job(svc, made["id"], mode="reroll")
    new_row = svc.store.get(new["id"])
    assert new_row["params"]["sheet"]["materials"] == []
    assert new_row["params"]["seed"] != svc.store.get(made["id"])["params"]["seed"]


# --- create-02: the Create tray's Rerun button must agree with rerollable ----


def test_the_create_tray_rerun_button_agrees_with_rerollable():
    """Before this fix, ``_result_card`` computed its own
    ``done or status in _FAILED``, which agreed with ``svc_jobs.rerollable``
    on *when* a row is finished but not on *what kind of row* -- so a
    finished ``lora_train``, ``separate``, built or hand-made-reference row
    was offered a Rerun that pressed straight into an ``Invalid`` at dispatch
    (the 2026-09-15 audit, finding create-02).

    Source-level, like this card's sibling tests in
    ``tests/modes/create/test_generation_workspace.py``: ``_result_card`` is drawn
    straight into imgui and cannot be exercised headlessly.
    """
    import inspect

    from realmspinner.studio.modes.create.ui import workspace as gw

    source = inspect.getsource(gw._result_card)
    assert "can_rerun = svc_jobs.rerollable(job)" in source
    assert "can_rerun = done or status in _FAILED" not in source
    # And the reason beside it names the real refusal once the row is
    # finished, rather than repeating "not ready yet" for a done row that
    # simply cannot be rerolled.
    assert "rerollable_reason(job)" in source
