"""Tranche 3's new service doors: a dry-run recipe, the follow-up sheet a rig
mints with no back-link, an in-flight rig guard, a paged asset listing, one
job-status door for any link in the chain, and a cropped sheet preview that
fits one RPC frame.

Every one of these is new in this tranche, so a test failing with
``AttributeError: module 'warlock...' has no attribute '...'`` against
``git show HEAD:<path>`` *is* the red half of red-green here -- there is no
old behaviour to regress against, only a door that did not exist yet.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from warlock import rigging
from warlock.characters import family as family_mod
from warlock.service import characters as svc_characters
from warlock.service import rig as svc_rig
from warlock.service import troupe as svc_troupe
from warlock.service.errors import Conflict, Invalid

# --- fixtures the door tests share --------------------------------------------


def _mesh(svc, *, family: str = "ogre", rigged: bool = False, name: str = "") -> str:
    """A finished character mesh row, minted directly -- no Blender, no queue."""
    params = {"asset_type": "character", "family": family, "built": True}
    job_id = svc.store.create(
        "image", f"a {family}", params, stage="model", status="done"
    )
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    if name:
        svc.store.set_meta(job_id, name=name)
    if rigged:
        template = family_mod.get_family(family).template
        (job_dir / "rig.glb").write_bytes(b"fake-rig")
        (job_dir / "rig.json").write_text(
            json.dumps({"template": template}), encoding="utf-8"
        )
    return job_id


def _rig_row(
    svc, mesh_id: str, *, status: str = "queued", troupe_sheet=None, template="humanoid"
) -> str:
    params = {"source_job": mesh_id, "template": template, "auto": True}
    if troupe_sheet is not None:
        params["troupe_sheet"] = troupe_sheet
    return svc.store.create("rig", "a hooded ranger", params, stage="model", status=status)


def _charsheet_row(svc, mesh_id: str, *, sheet_id=None, extra=None, base_sheet=None) -> str:
    params = {"source_job": mesh_id, "sheet_id": sheet_id or rigging.new_id()}
    if extra:
        params.update(extra)
    if base_sheet:
        params["base_sheet"] = base_sheet
    return svc.store.create(
        "charsheet", "a hooded ranger", params, stage="model", status="queued"
    )


def _touch_created_at(svc, job_id: str, value: float) -> None:
    svc.store._conn.execute("UPDATE jobs SET created_at = ? WHERE id = ?", (value, job_id))
    svc.store._conn.commit()


def _touch_finished_at(svc, job_id: str, value: float) -> None:
    svc.store._conn.execute("UPDATE jobs SET finished_at = ? WHERE id = ?", (value, job_id))
    svc.store._conn.commit()


def _build_sheet(svc, mesh_id: str, *, sheet_id: str | None = None) -> str:
    """A minimal, hand-built sheet: one movement, two directions, four cells --
    enough to exercise :func:`sheet_preview_png`'s cropping without paying for
    the real render pipeline."""
    from PIL import Image

    sheet_id = sheet_id or rigging.new_id()
    job_dir = svc.job_dir(mesh_id)
    frame = 8
    atlas = Image.new("RGBA", (frame * 2, frame * 2), (0, 0, 0, 255))
    colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255)]
    positions = [(0, 0), (frame, 0), (0, frame), (frame, frame)]
    for color, (x, y) in zip(colors, positions, strict=True):
        atlas.paste(Image.new("RGBA", (frame, frame), color), (x, y))
    png_path = rigging.sheet_png_path(job_dir, sheet_id)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(png_path, format="PNG")

    movements = [
        {
            "key": "walk",
            "label": "Walk",
            "frames": 2,
            "loop": False,
            "duration_ms": 100,
            "directions": [
                {"key": "front", "label": "Front", "yaw": 0.0},
                {"key": "back", "label": "Back", "yaw": 180.0},
            ],
        }
    ]
    runs = [
        {"movement": "walk", "direction": "front", "yaw": 0.0, "start": 0, "end": 1},
        {"movement": "walk", "direction": "back", "yaw": 180.0, "start": 2, "end": 3},
    ]
    cells = [
        {"index": i, "x": x, "y": y, "w": frame, "h": frame}
        for i, (x, y) in enumerate(positions)
    ]
    sidecar = {
        "version": 1,
        "id": sheet_id,
        "name": "preview test",
        "source_job": mesh_id,
        "created": 0.0,
        "image": png_path.name,
        "frame_size": frame,
        "columns": 2,
        "rows": 2,
        "width": frame * 2,
        "height": frame * 2,
        "elevation": 30.0,
        "lighting": "flat",
        "yaws": [0.0, 180.0],
        "poses": [],
        "cells": cells,
        "troupe": {
            "version": 3,
            "columns": 2,
            "movements": movements,
            "runs": runs,
            "cell_count": 4,
        },
    }
    rigging.sheet_path(job_dir, sheet_id).write_text(json.dumps(sidecar), encoding="utf-8")
    return sheet_id


# --- recipe_from_prompt --------------------------------------------------------


def test_swamp_knight_resolves_to_the_knight_and_reports_the_look_it_lacks(svc):
    result = svc_characters.recipe_from_prompt(svc, "a swamp knight")
    assert result["recipe"]["family"] == "knight"
    assert result["recipe"]["theme"] != "swamp"
    dropped = [i for i in result["ignored"] if i["kind"] == "theme"]
    assert len(dropped) == 1
    assert dropped[0]["text"] == "swamp"
    assert "Knight has no 'swamp' look" in dropped[0]["reason"]
    assert "natural" in dropped[0]["reason"] and "blackened" in dropped[0]["reason"]


def test_an_explicit_theme_the_species_lacks_is_refused_on_theme(svc):
    with pytest.raises(Invalid) as excinfo:
        svc_characters.recipe_from_prompt(svc, "a knight", overrides={"theme": "swamp"})
    assert excinfo.value.field == "theme"


def test_a_prompt_naming_no_species_is_refused_in_offer_sentences_words(svc):
    from warlock.characters import resolve as resolve_mod

    expected = resolve_mod.offer_sentence(resolve_mod.resolve("a manticore"))
    assert expected is not None
    with pytest.raises(Invalid) as excinfo:
        svc_characters.recipe_from_prompt(svc, "a manticore")
    assert excinfo.value.field == "prompt"
    assert str(excinfo.value) == expected

    with pytest.raises(Invalid) as generic:
        svc_characters.recipe_from_prompt(svc, "wibble wobble")
    assert generic.value.field == "prompt"
    assert "species across four body plans" in str(generic.value)

    with pytest.raises(Invalid) as blank:
        svc_characters.recipe_from_prompt(svc, "")
    assert blank.value.field == "prompt"
    assert "pass a prompt or a family" in str(blank.value)


def test_omitted_frames_follow_the_clip_library_at_the_requested_fps(svc):
    from warlock.clips import clip_timing
    from warlock.pipelines import charsheet

    result = svc_characters.recipe_from_prompt(
        svc,
        "",
        overrides={
            "family": "ogre",
            "animations": {"walk": None},
            "fps": 30,
            "directions": 8,
        },
    )
    expected_layout = charsheet.resolve_layout(
        {"version": 3, "fps": 30, "movements": [{"key": "walk", "directions": 8}]},
        timing=clip_timing("humanoid"),
    )
    assert result["recipe"]["animations"]["walk"] == expected_layout.movements[0].frames
    assert result["recipe"]["fps"] == 30


def test_a_movement_the_skeleton_has_no_clip_for_is_refused_on_animations(svc):
    with pytest.raises(Invalid) as excinfo:
        svc_characters.recipe_from_prompt(
            svc, "", overrides={"family": "ogre", "animations": {"moonwalk": None}}
        )
    assert excinfo.value.field == "animations"
    assert "moonwalk" in str(excinfo.value)
    assert "humanoid" in str(excinfo.value)


def test_recipe_from_prompt_mints_no_row_and_starts_no_process(svc, monkeypatch):
    def _boom():
        raise AssertionError("doctor.blender_check must not run for a dry-run recipe")

    monkeypatch.setattr("warlock.doctor.blender_check", _boom)
    before = svc.store.list(1000)
    result = svc_characters.recipe_from_prompt(svc, "a fire ogre")
    after = svc.store.list(1000)
    assert before == after
    assert result["recipe"]["family"] == "ogre"
    assert result["cells"] > 0
    assert result["estimate_minutes"] > 0


# --- follow_up_sheet_job / follow_up_failure / rig_in_flight ------------------


def test_follow_up_sheet_job_is_none_before_the_rig_lands(svc):
    mesh_id = _mesh(svc)
    rig_id = _rig_row(svc, mesh_id, troupe_sheet={"template": "humanoid", "logical_size": 32})
    assert svc_troupe.follow_up_sheet_job(svc, rig_id) is None


def test_follow_up_sheet_job_finds_the_sheet_a_rig_minted(svc):
    mesh_id = _mesh(svc)
    block = {"template": "humanoid", "logical_size": 32}
    rig_id = _rig_row(svc, mesh_id, troupe_sheet=block, status="done")
    _touch_created_at(svc, rig_id, 100.0)
    _touch_finished_at(svc, rig_id, 100.0)
    sheet_id = rigging.new_id()
    charsheet_id = _charsheet_row(svc, mesh_id, sheet_id=sheet_id, extra=dict(block))
    # Inside FOLLOW_UP_WINDOW_S of the rig's finished_at -- the ordinary case,
    # where the worker mints the sheet a few awaits after the rig's own
    # terminal write in the same step.
    _touch_created_at(svc, charsheet_id, 105.0)

    assert svc_troupe.follow_up_sheet_job(svc, rig_id) == charsheet_id


def test_follow_up_sheet_job_ignores_a_rerender_of_that_sheet(svc):
    mesh_id = _mesh(svc)
    block = {"template": "humanoid", "logical_size": 32}
    rig_id = _rig_row(svc, mesh_id, troupe_sheet=block, status="done")
    _touch_created_at(svc, rig_id, 100.0)
    _touch_finished_at(svc, rig_id, 100.0)

    real_sheet = rigging.new_id()
    real = _charsheet_row(svc, mesh_id, sheet_id=real_sheet, extra=dict(block))
    _touch_created_at(svc, real, 130.0)

    rerender = _charsheet_row(
        svc, mesh_id, sheet_id=rigging.new_id(), extra=dict(block), base_sheet=real_sheet
    )
    # Even minted "sooner" than the real follow-up, a re-render must never win.
    _touch_created_at(svc, rerender, 110.0)

    assert svc_troupe.follow_up_sheet_job(svc, rig_id) == real


def test_follow_up_sheet_job_ignores_an_identical_sheet_made_long_after_the_rig(svc):
    """A later, independent ``create_charsheet`` call on the same mesh with the
    exact settings the reservation pinned used to match -- nothing else
    distinguishes it from the sheet the worker actually minted, except that
    it landed hours after the rig finished rather than moments after."""
    mesh_id = _mesh(svc)
    block = {"template": "humanoid", "logical_size": 32}
    rig_id = _rig_row(svc, mesh_id, troupe_sheet=block, status="done")
    # Explicit and well before the sheet's own created_at, so the pre-fix
    # code's cruder "no earlier than the rig" test would also have accepted
    # this candidate -- it is only the window's *upper* bound that must
    # reject it.
    _touch_created_at(svc, rig_id, 900.0)
    _touch_finished_at(svc, rig_id, 1000.0)

    later = _charsheet_row(svc, mesh_id, sheet_id=rigging.new_id(), extra=dict(block))
    _touch_created_at(svc, later, 1000.0 + svc_troupe.FOLLOW_UP_WINDOW_S + 1.0)

    assert svc_troupe.follow_up_sheet_job(svc, rig_id) is None


def test_a_failed_rig_has_no_follow_up_sheet(svc):
    """``_maybe_queue_sheet_after_rig`` only ever runs off the worker's own
    ``done`` branch -- an ``error``/``cancelled`` rig minted nothing, so a
    charsheet row that happens to match its reservation and land nearby in
    time is somebody else's, not this rig's follow-up."""
    mesh_id = _mesh(svc)
    block = {"template": "humanoid", "logical_size": 32}
    rig_id = _rig_row(svc, mesh_id, troupe_sheet=block, status="error")
    # Explicit and well before the candidate's own created_at -- the pre-fix
    # code's timing filter alone would have accepted this match; only the new
    # status check can be what rejects it.
    _touch_created_at(svc, rig_id, 90.0)
    _touch_finished_at(svc, rig_id, 100.0)

    unrelated = _charsheet_row(svc, mesh_id, sheet_id=rigging.new_id(), extra=dict(block))
    _touch_created_at(svc, unrelated, 105.0)

    assert svc_troupe.follow_up_sheet_job(svc, rig_id) is None


def test_rig_in_flight_names_the_active_rig_row(svc):
    mesh_id = _mesh(svc)
    assert svc_rig.rig_in_flight(svc, mesh_id) is None

    rig_id = _rig_row(svc, mesh_id, status="queued")
    assert svc_rig.rig_in_flight(svc, mesh_id) == rig_id

    svc.store.set_status(rig_id, "done")
    assert svc_rig.rig_in_flight(svc, mesh_id) is None


def test_sending_to_troupe_while_a_rig_is_running_queues_no_second_rig(svc, monkeypatch):
    monkeypatch.setattr(
        "warlock.doctor.blender_check", lambda: SimpleNamespace(ok=True, detail="")
    )
    mesh_id = _mesh(svc)
    rig_id = _rig_row(svc, mesh_id, troupe_sheet={"template": "humanoid"})

    with pytest.raises(Conflict):
        svc_troupe.send_to_troupe(svc, mesh_id)

    rig_rows = [
        r for r in svc.store.list(limit=1000, kind="rig")
        if (r["params"] or {}).get("source_job") == mesh_id
    ]
    assert [r["id"] for r in rig_rows] == [rig_id]


# --- list_character_assets -----------------------------------------------------


def test_list_character_assets_filters_and_pages_by_cursor(svc):
    ids = [_mesh(svc, name=f"char-{i}") for i in range(3)]
    for i, job_id in enumerate(ids):
        _touch_created_at(svc, job_id, 100.0 + i)

    job_dir = svc.job_dir(ids[-1])
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    (job_dir / "rig.json").write_text(json.dumps({"template": "humanoid"}), encoding="utf-8")

    with pytest.raises(Invalid) as excinfo:
        svc_characters.list_character_assets(svc, filter="bogus")
    assert excinfo.value.field == "filter"

    page1 = svc_characters.list_character_assets(svc, limit=1)
    assert [a["job_id"] for a in page1["assets"]] == [ids[-1]]  # newest first
    assert page1["next_cursor"]

    page2 = svc_characters.list_character_assets(svc, limit=2, cursor=page1["next_cursor"])
    seen = [a["job_id"] for a in page1["assets"]] + [a["job_id"] for a in page2["assets"]]
    assert seen == list(reversed(ids))
    assert page2["next_cursor"] is None

    rigged_only = svc_characters.list_character_assets(svc, filter="rigged")
    assert [a["job_id"] for a in rigged_only["assets"]] == [ids[-1]]

    riggable_only = svc_characters.list_character_assets(svc, filter="riggable")
    assert {a["job_id"] for a in riggable_only["assets"]} == set(ids[:-1])


# --- character_job ---------------------------------------------------------


def test_character_job_reports_a_charsheet_rows_own_sheet_id(svc):
    mesh_id = _mesh(svc)
    sheet_id = rigging.new_id()
    charsheet_id = _charsheet_row(svc, mesh_id, sheet_id=sheet_id)

    result = svc_characters.character_job(svc, charsheet_id)
    assert result["kind"] == "charsheet"
    assert result["sheet_id"] == sheet_id
    assert result["source_job"] == mesh_id
    assert result["rig_job_id"] is None


def test_character_job_on_a_mesh_names_the_sheet_its_rig_queued(svc):
    """A character's mesh is stored with ``kind == "image"``, not ``"rig"`` --
    and an agent is told to poll the mesh id, not the rig id
    ``create_character`` also hands back. Before this fix,
    ``follow_up_sheet_job`` was answered only for a rig row, so polling the
    mesh reported ``follow_up_sheet_job: None`` forever, even once the sheet
    had actually landed."""
    mesh_id = _mesh(svc)
    block = {"template": "humanoid", "logical_size": 32}
    rig_id = _rig_row(svc, mesh_id, troupe_sheet=block, status="done")
    _touch_finished_at(svc, rig_id, 100.0)
    sheet_id = rigging.new_id()
    charsheet_id = _charsheet_row(svc, mesh_id, sheet_id=sheet_id, extra=dict(block))
    _touch_created_at(svc, charsheet_id, 105.0)

    result = svc_characters.character_job(svc, mesh_id)
    assert result["kind"] == "image"
    assert result["rig_job_id"] == rig_id
    assert result["follow_up_sheet_job"] == charsheet_id

    # And the rig row itself still names its own id, today's behaviour.
    rig_result = svc_characters.character_job(svc, rig_id)
    assert rig_result["rig_job_id"] == rig_id
    assert rig_result["follow_up_sheet_job"] == charsheet_id


def test_character_job_on_an_unrigged_mesh_names_no_rig(svc):
    mesh_id = _mesh(svc)
    result = svc_characters.character_job(svc, mesh_id)
    assert result["rig_job_id"] is None
    assert result["follow_up_sheet_job"] is None


# --- sheet_preview_png -------------------------------------------------------


def test_sheet_preview_crops_one_run_and_stays_under_max_bytes(svc):
    from PIL import Image

    mesh_id = _mesh(svc)
    sheet_id = _build_sheet(svc, mesh_id)

    data, meta = svc_characters.sheet_preview_png(
        svc, mesh_id, sheet_id,
        max_side=64, movement="walk", direction="front", max_bytes=200_000,
    )
    assert meta["frames"] == 2
    assert (meta["movement"], meta["direction"]) == ("walk", "front")
    assert (meta["width"], meta["height"]) == (16, 8)
    assert meta["compass"] == charsheet_compass("front")

    out = Image.open(io.BytesIO(data)).convert("RGBA")
    assert out.size == (16, 8)
    assert out.getpixel((0, 0)) == (255, 0, 0, 255)
    assert out.getpixel((8, 0)) == (0, 255, 0, 255)

    with pytest.raises(Invalid) as too_big:
        svc_characters.sheet_preview_png(
            svc, mesh_id, sheet_id,
            max_side=64, movement="walk", direction="front", max_bytes=10,
        )
    assert too_big.value.field == "max_side"

    with pytest.raises(Invalid) as bad_movement:
        svc_characters.sheet_preview_png(
            svc, mesh_id, sheet_id, max_side=64, movement="dance", max_bytes=200_000,
        )
    assert bad_movement.value.field == "movement"

    with pytest.raises(Invalid) as bad_direction:
        svc_characters.sheet_preview_png(
            svc, mesh_id, sheet_id,
            max_side=64, movement="walk", direction="left", max_bytes=200_000,
        )
    assert bad_direction.value.field == "direction"

    with pytest.raises(Invalid) as direction_without_movement:
        svc_characters.sheet_preview_png(
            svc, mesh_id, sheet_id, max_side=64, direction="front", max_bytes=200_000,
        )
    assert direction_without_movement.value.field == "movement"


def charsheet_compass(direction_key: str) -> str:
    from warlock.pipelines import charsheet

    return charsheet.COMPASS_16[direction_key]


# --- shipped_clip_names --------------------------------------------------------


def test_shipped_clip_names_ignore_a_user_edited_library(svc):
    from warlock import poselib

    shipped = rigging.shipped_clip_names("humanoid")
    assert "walk" in shipped

    user_dir = poselib.clip_dir(svc.config)
    user_dir.mkdir(parents=True, exist_ok=True)
    edited = user_dir / "humanoid.json"
    edited.write_text(json.dumps({"version": 3, "poses": [], "clips": []}), encoding="utf-8")
    rigging.invalidate_clips()
    try:
        assert rigging.clip_library("humanoid")["clips"] == []
        assert rigging.shipped_clip_names("humanoid") == shipped
    finally:
        edited.unlink()
        rigging.invalidate_clips()


# --- _package_stem -------------------------------------------------------------


def test_a_character_named_after_a_windows_device_exports_under_its_id(svc):
    assert svc_characters._package_stem({"name": "CON"}, "abc123456789") == "abc123456789"
    assert svc_characters._package_stem({"name": "con"}, "abc123456789") == "abc123456789"
    assert svc_characters._package_stem({"name": "COM1"}, "abc123456789") == "abc123456789"
    assert svc_characters._package_stem({"name": "LPT9"}, "abc123456789") == "abc123456789"
    # A dot is already sanitised to a hyphen before the reserved-name check
    # ever runs, so "con.txt" safely becomes "con-txt" -- not a device name --
    # rather than falling back; only the bare device name is the hazard.
    assert svc_characters._package_stem({"name": "con.txt"}, "abc123456789") == "con-txt"
    # A name that merely contains a reserved word is not the reserved word.
    assert svc_characters._package_stem({"name": "COM1 the brave"}, "abc123456789") == (
        "COM1 the brave"
    )
