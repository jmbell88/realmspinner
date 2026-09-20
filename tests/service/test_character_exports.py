"""Four ways a finished character leaves the app as files on disk.

``service.characters.export_frames``/``export_godot`` are this slice's two new
doors, and ``service.export.CHARACTER_EXPORTS`` is the menu a pane draws them
from. Neither door renders anything -- both work off a sheet's own sidecar (a
Troupe render already did the work) or off ``animated.glb`` (``derive`` already
knows how to bake and refresh it), so what is asserted here is the *shape* of
what lands on disk: one folder per clip named by compass point rather than by
Troupe's internal facing key, a renamed copy beside its Godot scene that never
touches the served original, and the registry that ties both (and the two
doors from before this slice) to one uniform call.
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import pytest
from PIL import Image

from realmspinner import clips
from realmspinner.kernels import charsheet
from realmspinner.kernels import sheet as sheetlib
from realmspinner.kernels.geom3d import glbio
from realmspinner.kernels.rig import store
from realmspinner.service import characters as svc_characters
from realmspinner.service import derive as svc_derive
from realmspinner.service import export as svc_export
from realmspinner.service.errors import Invalid, NotReady

# --- fixtures: a rendered character sheet, built the way ``_q_troupe`` does --


def _new_job(svc, *, name: str | None = None) -> str:
    job_id = svc.store.create("image", "a hooded ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    svc.store.set_status(job_id, "done")
    if name:
        svc.store.set_meta(job_id, name=name)
    return job_id


def _build_sheet(
    svc,
    layout: charsheet.LayoutSpec,
    *,
    with_troupe_block: bool,
    frame_size: int = 16,
    name: str | None = None,
    character: dict | None = None,
) -> tuple[str, str, dict[int, tuple[int, int, int, int]]]:
    """A real sheet PNG + sidecar, laid out on *layout* -- no Blender, no worker.

    Each cell is painted a distinct solid colour keyed on its index, so a test
    can crop the atlas itself the way the export is supposed to and compare.
    """
    job_id = _new_job(svc, name=name)
    job_dir = svc.job_dir(job_id)

    table = charsheet.frame_table(layout)
    columns = layout.columns
    rows = (len(table) + columns - 1) // columns
    colors: dict[int, tuple[int, int, int, int]] = {}
    cells = []
    for tc in table:
        row, col = tc.index // columns, tc.index % columns
        cells.append(
            sheetlib.Cell(
                index=tc.index,
                row=row,
                column=col,
                x=col * frame_size,
                y=row * frame_size,
                pose=None,
                pose_name=tc.animation,
                yaw=tc.yaw,
                frame=tc.frame,
            )
        )
        colors[tc.index] = ((tc.index * 7) % 256, (tc.index * 53) % 256, 200, 255)
    plan = sheetlib.Plan(
        frame_size=frame_size,
        columns=columns,
        rows=rows,
        yaws=tuple(y for _n, y in layout.directions),
        elevation=30.0,
        lighting="flat",
        poses=(),
        cells=tuple(cells),
    )

    atlas = Image.new("RGBA", (plan.width, plan.height), (0, 0, 0, 255))
    for cell in cells:
        box = (cell.x, cell.y, cell.x + frame_size, cell.y + frame_size)
        atlas.paste(Image.new("RGBA", (frame_size, frame_size), colors[cell.index]), box)

    sheet_id = store.new_id()
    store.sheet_dir(job_dir).mkdir(parents=True, exist_ok=True)
    png_path = store.sheet_png_path(job_dir, sheet_id)
    atlas.save(png_path, "PNG")

    meta = sheetlib.sidecar(
        plan,
        sheet_id=sheet_id,
        source_job=job_id,
        image=png_path.name,
        created=time.time(),
        name="Test Sheet",
        animation=charsheet.animation_block(layout),
    )
    if with_troupe_block:
        meta["troupe"] = layout.as_dict()
    if character is not None:
        meta["character"] = character
    store.sheet_path(job_dir, sheet_id).write_text(json.dumps(meta), "utf-8")

    return job_id, sheet_id, colors


def _two_movement_layout() -> charsheet.LayoutSpec:
    return charsheet.resolve_layout(
        {
            "version": 3,
            "movements": [
                {"key": "idle", "loop": True, "duration_ms": 150, "frames": 2, "directions": 4},
                {"key": "patrol", "loop": False, "duration_ms": 80, "frames": 3, "directions": 4},
            ],
        }
    )


def _frame_bytes(png_path: Path, frame_size: int) -> bytes:
    with Image.open(png_path) as img:
        return img.convert("RGBA").tobytes()


def _solid_frame_bytes(color: tuple[int, int, int, int], frame_size: int) -> bytes:
    return Image.new("RGBA", (frame_size, frame_size), color).tobytes()


# --- export_frames ------------------------------------------------------------


def test_a_frame_export_lands_every_cell_under_clip_and_compass_folders(svc, tmp_path):
    layout = _two_movement_layout()
    job_id, sheet_id, colors = _build_sheet(svc, layout, with_troupe_block=True, name="Ranger")

    dest = svc_characters.export_frames(svc, job_id, sheet_id, dest_dir=tmp_path / "out")

    assert dest.is_dir()
    total_frames = 0
    for run in layout.as_dict()["runs"]:
        clip = run["movement"]
        compass = charsheet.compass_name(run["yaw"])
        folder = dest / clip / compass
        frames = sorted(folder.glob("*.png"))
        expected = run["end"] - run["start"] + 1
        assert len(frames) == expected
        for offset, path in enumerate(frames):
            assert path.name == f"{offset:03d}.png"
            cell_index = run["start"] + offset
            assert _frame_bytes(path, 16) == _solid_frame_bytes(colors[cell_index], 16)
        total_frames += expected
    assert total_frames == len(charsheet.frame_table(layout))
    assert (dest / "manifest.json").exists()


def test_a_frame_export_names_directions_by_compass_only(svc, tmp_path):
    layout = _two_movement_layout()
    job_id, sheet_id, _colors = _build_sheet(svc, layout, with_troupe_block=True)

    dest = svc_characters.export_frames(svc, job_id, sheet_id, dest_dir=tmp_path / "out")

    raw_direction_keys = {name for name, _yaw in charsheet.DIRECTIONS} | {
        name for name, _yaw in charsheet._DIRECTIONS_16  # noqa: SLF001 -- test-only peek
    }
    for clip_dir in dest.iterdir():
        if not clip_dir.is_dir():
            continue
        for compass_dir in clip_dir.iterdir():
            assert compass_dir.name not in raw_direction_keys
            assert all(ch in "NSEW" for ch in compass_dir.name)


def test_the_front_run_lands_in_the_south_folder(svc, tmp_path):
    layout = _two_movement_layout()
    job_id, sheet_id, _colors = _build_sheet(svc, layout, with_troupe_block=True)

    dest = svc_characters.export_frames(svc, job_id, sheet_id, dest_dir=tmp_path / "out")

    assert (dest / "idle" / "S").is_dir()
    assert list((dest / "idle" / "S").glob("*.png"))


def test_a_pre_troupe_block_sheet_exports_through_the_legacy_table(svc, tmp_path):
    legacy = charsheet.resolve_layout(None)
    job_id, sheet_id, colors = _build_sheet(svc, legacy, with_troupe_block=False)

    dest = svc_characters.export_frames(svc, job_id, sheet_id, dest_dir=tmp_path / "out")

    manifest = json.loads((dest / "manifest.json").read_text("utf-8"))
    assert set(manifest["clips"]) == {"idle", "walk", "run", "attack", "jump"}
    for name, frames, loop, duration_ms in charsheet.ANIMATIONS:
        entry = manifest["clips"][name]
        assert entry["frames"] == frames
        assert entry["loop"] == loop
        assert entry["duration_ms"] == duration_ms
        assert len(entry["directions"]) == 8
    # And the pixels themselves came from the atlas this fixture painted, not
    # from a re-derived legacy table that quietly disagreed about layout.
    assert (dest / "idle" / "S" / "000.png").exists()


def test_the_frame_manifest_states_fps_loop_and_frame_count_per_clip(svc, tmp_path):
    # duration_ms is ``round(1000 / 15)`` -- an fps-carrying v3 movement's own
    # duration is held to that number exactly (a snapshot's honest claim about
    # its own fps), not the multiple-of-10 authoring rule a movement with no
    # global fps is held to. See ``charsheet._validate_movement_duration_ms``.
    layout = charsheet.resolve_layout(
        {
            "version": 3,
            "fps": 15,
            "movements": [
                {"key": "sprint", "loop": True, "duration_ms": 67, "frames": 4, "directions": 4},
            ],
        }
    )
    job_id, sheet_id, _colors = _build_sheet(svc, layout, with_troupe_block=True)

    dest = svc_characters.export_frames(svc, job_id, sheet_id, dest_dir=tmp_path / "out")
    manifest = json.loads((dest / "manifest.json").read_text("utf-8"))

    entry = manifest["clips"]["sprint"]
    assert entry["loop"] is True
    assert entry["frames"] == 4
    # A global fps was named on the layout, so the manifest states *that*
    # figure exactly -- not ``1000 / duration_ms`` recomputed from the
    # rounded-to-a-scene-frame duration, which for 15 fps would answer
    # something close to but not exactly 15.
    assert entry["fps"] == 15.0

    no_fps_layout = _two_movement_layout()
    job_id2, sheet_id2, _c = _build_sheet(svc, no_fps_layout, with_troupe_block=True)
    dest2 = svc_characters.export_frames(svc, job_id2, sheet_id2, dest_dir=tmp_path / "out2")
    manifest2 = json.loads((dest2 / "manifest.json").read_text("utf-8"))
    idle = manifest2["clips"]["idle"]
    assert idle["duration_ms"] == 150
    assert idle["fps"] == pytest.approx(1000.0 / 150)


def test_the_frame_manifest_reports_an_hd_troupe_sheet_as_hd(svc, tmp_path):
    """``_q_troupe`` writes the worker's own D5 answer at the sidecar's top
    level (``"pixel_art": False``, absent for a pixel-art render). The
    manifest used to read only the *recipe*'s nested copy -- what was asked
    for, which can be a species' HD default that this particular render did
    not honour -- and so reported an HD Troupe sheet's manifest as pixel art.
    """
    layout = _two_movement_layout()
    job_id, sheet_id, _colors = _build_sheet(
        svc, layout, with_troupe_block=True, character={"recipe": {"pixel_art": True}}
    )
    job_dir = svc.job_dir(job_id)
    sidecar_path = store.sheet_path(job_dir, sheet_id)
    meta = json.loads(sidecar_path.read_text("utf-8"))
    # The worker's own record for this render: HD, even though the nested
    # recipe (what was asked for) still says pixel art.
    meta["pixel_art"] = False
    sidecar_path.write_text(json.dumps(meta), "utf-8")

    dest = svc_characters.export_frames(svc, job_id, sheet_id, dest_dir=tmp_path / "out")
    manifest = json.loads((dest / "manifest.json").read_text("utf-8"))
    assert manifest["pixel_art"] is False


def test_a_hand_made_sheet_without_a_character_layout_is_refused_not_crashed(svc, tmp_path):
    """No ``"troupe"`` block falls back to the closed legacy 256-cell table --
    but a hand-drawn sheet (or one from a format ``sheet.sidecar`` also
    writes) need not have anywhere near 256 cells on it. Before this fix,
    ``export_frames`` asked for a cell index the sidecar never recorded and
    died on a bare ``KeyError`` instead of naming the mismatch."""
    job_id = _new_job(svc)
    job_dir = svc.job_dir(job_id)
    sheet_id = store.new_id()
    store.sheet_dir(job_dir).mkdir(parents=True, exist_ok=True)
    # A real (if tiny) PNG, so the crash this regresses is the cell lookup and
    # not merely an unreadable image.
    Image.new("RGBA", (16, 16), (0, 0, 0, 255)).save(
        store.sheet_png_path(job_dir, sheet_id), "PNG"
    )
    meta = {
        "sheet_id": sheet_id,
        "source_job": job_id,
        "frame_size": 16,
        # Far short of the legacy table's 256 cells -- and no "troupe" block,
        # so the legacy table is exactly what this sheet is checked against.
        "cells": [{"index": 0, "x": 0, "y": 0, "w": 16, "h": 16}],
    }
    store.sheet_path(job_dir, sheet_id).write_text(json.dumps(meta), "utf-8")

    with pytest.raises(Invalid) as excinfo:
        svc_characters.export_frames(svc, job_id, sheet_id, dest_dir=tmp_path / "out")
    assert excinfo.value.field == "sheet_id"
    assert "export it as a package instead" in str(excinfo.value)


def test_a_frame_export_that_fails_part_way_leaves_nothing_at_the_destination(svc, tmp_path):
    layout = charsheet.resolve_layout(
        {
            "version": 3,
            "movements": [
                {"key": "idle", "loop": True, "duration_ms": 150, "frames": 1, "directions": 1},
                # Uppercase: legal as a Troupe movement name, illegal as an
                # export folder name -- refused before either clip's pixels
                # are ever cropped, so nothing lands at the destination.
                {"key": "Walk", "loop": True, "duration_ms": 100, "frames": 1, "directions": 1},
            ],
        }
    )
    job_id, sheet_id, _colors = _build_sheet(svc, layout, with_troupe_block=True, name="Ranger")
    dest_root = tmp_path / "out"

    with pytest.raises(Invalid) as excinfo:
        svc_characters.export_frames(svc, job_id, sheet_id, dest_dir=dest_root)
    assert excinfo.value.field == "sheet_id"

    assert not dest_root.exists() or list(dest_root.iterdir()) == []


def test_re_exporting_replaces_the_character_folder_whole(svc, tmp_path):
    layout = _two_movement_layout()
    job_id, sheet_id, _colors = _build_sheet(svc, layout, with_troupe_block=True, name="Ranger")
    dest_root = tmp_path / "out"

    dest = svc_characters.export_frames(svc, job_id, sheet_id, dest_dir=dest_root)
    stray = dest / "leftover-from-an-older-export.txt"
    stray.write_text("stale", "utf-8")
    assert stray.exists()

    dest2 = svc_characters.export_frames(svc, job_id, sheet_id, dest_dir=dest_root)
    assert dest2 == dest
    assert not stray.exists()
    assert (dest2 / "manifest.json").exists()


def test_a_sheet_export_without_a_sheet_is_refused_on_sheet_id(svc):
    with pytest.raises(Invalid) as excinfo:
        svc_export.run_character_export(svc, "frame_folders", store.new_id(), None)
    assert excinfo.value.field == "sheet_id"
    with pytest.raises(Invalid) as excinfo2:
        svc_export.run_character_export(svc, "sheet_package", store.new_id(), "")
    assert excinfo2.value.field == "sheet_id"


# --- an explicit ``stem`` (fix #3, 2026-09-13) ---------------------------------
#
# An agent exporting its own character used to reuse ``_package_stem``, the
# same door a human's pane calls -- named off the job's display name alone.
# Two characters sharing a name (the default, for anything built from the
# same family and never renamed) then exported to the very same folder or
# file, and the second export silently replaced the first. ``stem`` lets a
# caller name the export explicitly, and ``characters.agent_export_stem``
# is what makes an agent's own choice always unique.


def test_an_explicit_stem_names_the_export_and_is_validated(svc, tmp_path):
    layout = _two_movement_layout()
    job_id, sheet_id, _colors = _build_sheet(svc, layout, with_troupe_block=True, name="Ranger")

    dest = svc_characters.export_frames(
        svc, job_id, sheet_id, dest_dir=tmp_path / "frames", stem="my-custom-stem"
    )
    assert dest.name == "my-custom-stem"
    assert (dest / "manifest.json").exists()

    with pytest.raises(Invalid) as excinfo:
        svc_characters.export_frames(
            svc, job_id, sheet_id, dest_dir=tmp_path / "frames2", stem="../escape"
        )
    assert excinfo.value.field == "name"

    # export_package writes its pair directly rather than through
    # staged_tree, so it has to validate an explicit stem by hand.
    result = svc_characters.export_package(
        svc, job_id, sheet_id, dest_dir=tmp_path / "package", stem="pkg-stem"
    )
    assert Path(result["png"]).name == "pkg-stem.png"
    assert Path(result["json"]).name == "pkg-stem.json"

    with pytest.raises(Invalid) as excinfo2:
        svc_characters.export_package(
            svc, job_id, sheet_id, dest_dir=tmp_path / "package2", stem="a/b"
        )
    assert excinfo2.value.field == "name"


def test_agent_export_stems_never_collide_across_assets_or_sheets(svc):
    job_a = _new_job(svc, name="Knight")
    job_b = _new_job(svc, name="Knight")
    sheet1 = store.new_id()
    sheet2 = store.new_id()

    stems = [
        svc_characters.agent_export_stem(svc, job_a),
        svc_characters.agent_export_stem(svc, job_b),
        svc_characters.agent_export_stem(svc, job_a, sheet1),
        svc_characters.agent_export_stem(svc, job_a, sheet2),
        svc_characters.agent_export_stem(svc, job_b, sheet1),
    ]
    assert len(set(stems)) == len(stems)


def test_two_characters_with_the_same_name_do_not_share_an_agent_export_folder(svc, tmp_path):
    layout = _two_movement_layout()
    job_a, sheet_a, _colors_a = _build_sheet(svc, layout, with_troupe_block=True, name="Knight")
    job_b, sheet_b, _colors_b = _build_sheet(svc, layout, with_troupe_block=True, name="Knight")
    dest_root = tmp_path / "out"

    # Both share a display name, so an unqualified export of either would
    # land at the same job-name folder -- the defect this fix closes.
    assert svc.store.get(job_a)["name"] == svc.store.get(job_b)["name"] == "Knight"

    stem_a = svc_characters.agent_export_stem(svc, job_a, sheet_a)
    stem_b = svc_characters.agent_export_stem(svc, job_b, sheet_b)
    assert stem_a != stem_b

    dest_a = svc_characters.export_frames(svc, job_a, sheet_a, dest_dir=dest_root, stem=stem_a)
    dest_b = svc_characters.export_frames(svc, job_b, sheet_b, dest_dir=dest_root, stem=stem_b)

    assert dest_a != dest_b
    assert dest_a.is_dir() and dest_b.is_dir()
    assert (dest_a / "manifest.json").exists()
    assert (dest_b / "manifest.json").exists()


# --- export_godot --------------------------------------------------------------


def _fake_glb(
    animation_names: list[str], *, loops: list[str], digest: str, rig_digest: str
) -> bytes:
    header = struct.pack("<III", glbio.GLB_MAGIC, 2, 0)
    stamp = {"clips_digest": digest, "rig_digest": rig_digest, "loops": list(loops)}
    gltf = {
        "asset": {"version": "2.0"},
        "animations": [{"name": name} for name in animation_names],
        "extras": {"realmspinner_animation": stamp},
    }
    return glbio.rebuild_glb(header, gltf, b"")


def _rigged_and_animated(
    svc, *, animation_names: list[str], loops: list[str], template: str = "humanoid", name=None
) -> str:
    # Stamped exactly as a fresh bake would be -- the clip library's digest AND
    # the rig's -- so ``derive.get_file`` serves this file instead of trying to
    # rebake it from the fake ``rig.glb`` (a stamp missing ``rig_digest`` reads
    # as a bake from before a re-rig and is rebuilt).
    from realmspinner.service import derive as svc_derive

    job_id = _new_job(svc, name=name)
    job_dir = svc.job_dir(job_id)
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    (job_dir / "rig.json").write_text(json.dumps({"template": template}), "utf-8")
    digest = clips.library_digest(template)
    glb = _fake_glb(
        animation_names, loops=loops, digest=digest, rig_digest=svc_derive._rig_digest(job_dir)
    )
    (job_dir / "animated.glb").write_bytes(glb)
    return job_id


def test_a_godot_export_writes_the_renamed_glb_and_its_scene_together(svc, tmp_path):
    job_id = _rigged_and_animated(
        svc, animation_names=["idle", "walk"], loops=["idle", "walk"], name="Ranger"
    )

    dest = svc_characters.export_godot(svc, job_id, dest_dir=tmp_path / "out")

    assert dest.is_dir()
    stem = dest.name
    glb_path = dest / f"{stem}.glb"
    tscn_path = dest / f"{stem}.tscn"
    assert glb_path.exists()
    assert tscn_path.exists()
    names = glbio.animation_names(glb_path.read_bytes())
    assert set(names) == {"idle-loop", "walk-loop"}
    text = tscn_path.read_text("utf-8")
    assert f'path="{stem}.glb"' in text
    # The renamed GLB still holds the loop-suffixed names (asserted above);
    # the scene plays the names Godot's importer hands back to the running
    # game *after* it strips the suffix on the way to setting loop mode.
    assert '&"idle"' in text
    assert '&"walk"' in text


def test_a_godot_export_suffixes_only_the_looping_clips(svc, tmp_path):
    job_id = _rigged_and_animated(svc, animation_names=["idle", "attack"], loops=["idle"])

    dest = svc_characters.export_godot(svc, job_id, dest_dir=tmp_path / "out")

    glb_path = dest / f"{dest.name}.glb"
    names = glbio.animation_names(glb_path.read_bytes())
    assert "idle-loop" in names
    assert "attack" in names
    assert "attack-loop" not in names


def test_a_godot_export_leaves_the_served_animated_glb_unsuffixed(svc, tmp_path):
    job_id = _rigged_and_animated(svc, animation_names=["idle", "attack"], loops=["idle"])
    served_before = svc.job_dir(job_id).joinpath("animated.glb").read_bytes()

    svc_characters.export_godot(svc, job_id, dest_dir=tmp_path / "out")

    served_after = svc.job_dir(job_id).joinpath("animated.glb").read_bytes()
    assert served_after == served_before
    assert glbio.animation_names(served_after) == ["idle", "attack"]


def test_a_clip_name_godot_cannot_hold_is_a_refusal(svc, tmp_path):
    """``godot_clip_name`` raises ``ValueError`` for a clip name that would
    break a Godot ``&"..."`` StringName literal (a bare quote, here). The
    rename ``mapping`` used to be built *before* the door's own ``try``, so
    that raise escaped as a bare ``ValueError`` instead of this refusal."""
    job_id = _rigged_and_animated(
        svc, animation_names=["idle", 'wal"k'], loops=["idle"]
    )
    with pytest.raises(Invalid) as excinfo:
        svc_characters.export_godot(svc, job_id, dest_dir=tmp_path / "out")
    assert excinfo.value.field == "job_id"


def test_a_godot_export_of_an_unrigged_mesh_is_refused_in_the_animated_glb_words(svc, tmp_path):
    job_id = _new_job(svc)
    with pytest.raises(NotReady) as excinfo:
        svc_characters.export_godot(svc, job_id, dest_dir=tmp_path / "out")
    assert "This asset has not been rigged yet." in str(excinfo.value)


def test_an_agent_animated_glb_export_is_named_for_its_ids_and_never_replaces_another(
    svc, tmp_path
):
    """``animated_glb`` was the one export format that still ignored ``stem``
    -- ``export_to_folder`` always names its copy after the job, so two
    agent-built characters sharing a display name (the default, for anything
    built from the same family and never renamed) replaced each other's
    animated GLB even after the other three formats were fixed."""
    job_a = _rigged_and_animated(svc, animation_names=["idle"], loops=["idle"], name="Knight")
    job_b = _rigged_and_animated(svc, animation_names=["idle"], loops=["idle"], name="Knight")
    served_a_before = svc.job_dir(job_a).joinpath("animated.glb").read_bytes()
    served_b_before = svc.job_dir(job_b).joinpath("animated.glb").read_bytes()

    svc.config.export_dir = tmp_path / "out"

    stem_a = svc_characters.agent_export_stem(svc, job_a)
    stem_b = svc_characters.agent_export_stem(svc, job_b)
    assert stem_a != stem_b

    result_a = svc_export.run_character_export(svc, "animated_glb", job_a, stem=stem_a)
    result_b = svc_export.run_character_export(svc, "animated_glb", job_b, stem=stem_b)

    path_a = svc.config.export_dir / f"{stem_a}.glb"
    path_b = svc.config.export_dir / f"{stem_b}.glb"
    assert path_a.exists()
    assert path_b.exists()
    assert path_a != path_b
    assert result_a["dir"] == str(svc.config.export_dir)
    assert result_b["dir"] == str(svc.config.export_dir)
    assert result_a["copied"]
    assert result_b["copied"]

    # The served copies are untouched -- only read, never replaced.
    assert svc.job_dir(job_a).joinpath("animated.glb").read_bytes() == served_a_before
    assert svc.job_dir(job_b).joinpath("animated.glb").read_bytes() == served_b_before


def test_an_animated_glb_export_without_a_stem_keeps_the_old_job_named_path(svc, tmp_path):
    job_id = _rigged_and_animated(svc, animation_names=["idle"], loops=["idle"], name="Ranger")
    svc.config.export_dir = tmp_path / "out"

    result = svc_export.run_character_export(svc, "animated_glb", job_id)

    assert result["dir"] == str(svc.config.export_dir)
    assert "copied" in result
    assert "degraded" in result
    # export_to_folder's own arcname -- untouched by this fix.
    assert (svc.config.export_dir / job_id / "animated.glb").exists()


# --- the registry ---------------------------------------------------------------


def test_every_character_export_has_a_door_and_every_door_is_registered(svc, monkeypatch):
    assert set(svc_export.CHARACTER_EXPORTS) == {
        "animated_glb",
        "sheet_package",
        "godot_scene",
        "frame_folders",
    }
    assert svc_export.CHARACTER_EXPORTS["sheet_package"].needs_sheet is True
    assert svc_export.CHARACTER_EXPORTS["frame_folders"].needs_sheet is True
    assert svc_export.CHARACTER_EXPORTS["animated_glb"].needs_sheet is False
    assert svc_export.CHARACTER_EXPORTS["godot_scene"].needs_sheet is False
    for row in svc_export.CHARACTER_EXPORTS.values():
        assert callable(row.door)

    calls: list[str] = []
    monkeypatch.setattr(
        svc_characters, "export_package", lambda *a, **k: calls.append("sheet_package") or "P"
    )
    monkeypatch.setattr(
        svc_characters, "export_godot", lambda *a, **k: calls.append("godot_scene") or "G"
    )
    monkeypatch.setattr(
        svc_characters, "export_frames", lambda *a, **k: calls.append("frame_folders") or "F"
    )
    monkeypatch.setattr(
        svc_export, "export_to_folder", lambda *a, **k: calls.append("animated_glb") or "A"
    )
    monkeypatch.setattr(svc_derive, "get_file", lambda *a, **k: Path("animated.glb"))

    job_id = store.new_id()
    assert svc_export.run_character_export(svc, "animated_glb", job_id) == "A"
    assert svc_export.run_character_export(svc, "sheet_package", job_id, "sheet1") == "P"
    assert svc_export.run_character_export(svc, "godot_scene", job_id) == "G"
    assert svc_export.run_character_export(svc, "frame_folders", job_id, "sheet1") == "F"
    assert set(calls) == set(svc_export.CHARACTER_EXPORTS)


def test_an_unknown_export_format_is_refused_on_format(svc):
    with pytest.raises(Invalid) as excinfo:
        svc_export.run_character_export(svc, "powerpoint", store.new_id())
    assert excinfo.value.field == "format"
