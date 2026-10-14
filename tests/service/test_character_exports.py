"""Two ways a finished character leaves the app as files on disk.

``service.characters.export_godot`` and the animated GLB door, and
``service.export.CHARACTER_EXPORTS`` is the menu a pane draws them from. The
sprite-sheet pair and the frame folders left with the character sheets
(``retired.py``). Neither door renders anything -- both work off
``animated.glb`` (``derive`` already knows how to bake and refresh it), so what
is asserted here is the *shape* of what lands on disk: a renamed copy beside its
Godot scene that never touches the served original, and the registry that ties
both to one uniform call.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from realmspinner import clips
from realmspinner.kernels.geom3d import glbio
from realmspinner.kernels.rig import store
from realmspinner.service import characters as svc_characters
from realmspinner.service import derive as svc_derive
from realmspinner.service import export as svc_export
from realmspinner.service.errors import Invalid, NotReady

# --- fixtures -----------------------------------------------------------------


def _new_job(svc, *, name: str | None = None) -> str:
    job_id = svc.store.create("image", "a hooded ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    svc.store.set_status(job_id, "done")
    if name:
        svc.store.set_meta(job_id, name=name)
    return job_id


# --- export_frames ------------------------------------------------------------


def test_a_godot_re_export_takes_a_new_name_and_overwrite_replaces(svc, tmp_path):
    job_id = _rigged_and_animated(svc, animation_names=["idle"], loops=["idle"], name="Ranger")
    out = tmp_path / "out"
    first = svc_characters.export_godot(svc, job_id, dest_dir=out)
    (first / "hand_attached.gd").write_text("extends Node", "utf-8")
    second = svc_characters.export_godot(svc, job_id, dest_dir=out)
    assert second.name == "Ranger-godot-2" and first.name == "Ranger-godot"
    assert (first / "hand_attached.gd").exists()
    # the scene inside names itself after its own (new) folder
    assert (second / "Ranger-godot-2.tscn").exists() and (second / "Ranger-godot-2.glb").exists()
    third = svc_characters.export_godot(svc, job_id, dest_dir=out, overwrite=True)
    assert third == first and not (first / "hand_attached.gd").exists()


def test_a_repeat_agent_export_lands_beside_the_first_not_over_it(svc, tmp_path):
    job_id = _rigged_and_animated(svc, animation_names=["idle"], loops=["idle"], name="Knight")
    svc.config.export_dir = tmp_path / "out"
    stem = svc_characters.agent_export_stem(svc, job_id)
    svc_export.run_character_export(svc, "animated_glb", job_id, stem=stem)
    first = svc.config.export_dir / f"{stem}.glb"
    first.write_bytes(b"user edited")
    svc_export.run_character_export(svc, "animated_glb", job_id, stem=stem)
    assert first.read_bytes() == b"user edited"
    assert (svc.config.export_dir / f"{stem}-2.glb").exists()
    svc_export.run_character_export(svc, "animated_glb", job_id, stem=stem, overwrite=True)
    assert first.read_bytes() != b"user edited"
    assert not (svc.config.export_dir / f"{stem}-3.glb").exists()


def test_a_repeat_human_animated_glb_export_keeps_the_first_file(svc, tmp_path):
    job_id = _rigged_and_animated(svc, animation_names=["idle"], loops=["idle"], name="Ranger")
    svc.config.export_dir = tmp_path / "out"
    svc_export.run_character_export(svc, "animated_glb", job_id)
    first = svc.config.export_dir / job_id / "animated.glb"
    first.write_bytes(b"user edited")
    svc_export.run_character_export(svc, "animated_glb", job_id)
    assert first.read_bytes() == b"user edited"
    assert (svc.config.export_dir / job_id / "animated-2.glb").exists()
    svc_export.run_character_export(svc, "animated_glb", job_id, overwrite=True)
    assert first.read_bytes() != b"user edited"


# --- an explicit ``stem`` (fix #3, 2026-09-13) ---------------------------------
#
# An agent exporting its own character used to reuse ``_package_stem``, the
# same door a human's pane calls -- named off the job's display name alone.
# Two characters sharing a name (the default, for anything built from the
# same family and never renamed) then exported to the very same folder or
# file, and the second export silently replaced the first. ``stem`` lets a
# caller name the export explicitly, and ``characters.agent_export_stem``
# is what makes an agent's own choice always unique.


def test_agent_export_stems_never_collide_across_assets(svc):
    job_a = _new_job(svc, name="Knight")
    job_b = _new_job(svc, name="Knight")

    stems = [
        svc_characters.agent_export_stem(svc, job_a),
        svc_characters.agent_export_stem(svc, job_b),
    ]
    assert len(set(stems)) == len(stems)


def test_an_explicit_stem_names_the_godot_export_and_is_validated(svc, tmp_path):
    job_id = _rigged_and_animated(svc, animation_names=["idle"], loops=["idle"], name="Ranger")

    dest = svc_characters.export_godot(
        svc, job_id, dest_dir=tmp_path / "godot", stem="my-custom-stem"
    )
    assert dest.name == "my-custom-stem"
    assert (dest / "my-custom-stem.tscn").exists()

    with pytest.raises(Invalid) as excinfo:
        svc_characters.export_godot(svc, job_id, dest_dir=tmp_path / "godot2", stem="../escape")
    assert excinfo.value.field == "name"


def test_two_characters_with_the_same_name_do_not_share_an_agent_export_folder(svc, tmp_path):
    job_a = _rigged_and_animated(svc, animation_names=["idle"], loops=["idle"], name="Knight")
    job_b = _rigged_and_animated(svc, animation_names=["idle"], loops=["idle"], name="Knight")
    dest_root = tmp_path / "out"

    # Both share a display name, so an unqualified export of either would
    # land at the same job-name folder -- the defect the agent stem closes.
    assert svc.store.get(job_a)["name"] == svc.store.get(job_b)["name"] == "Knight"

    stem_a = svc_characters.agent_export_stem(svc, job_a)
    stem_b = svc_characters.agent_export_stem(svc, job_b)
    assert stem_a != stem_b

    dest_a = svc_characters.export_godot(svc, job_a, dest_dir=dest_root, stem=stem_a)
    dest_b = svc_characters.export_godot(svc, job_b, dest_dir=dest_root, stem=stem_b)

    assert dest_a != dest_b
    assert dest_a.is_dir() and dest_b.is_dir()


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
    # The sprite-sheet pair and the frame folders left with the character sheets
    # (``retired.py``): a menu that still offered one would hand the user a
    # button with nothing behind it.
    assert set(svc_export.CHARACTER_EXPORTS) == {"animated_glb", "godot_scene"}
    for row in svc_export.CHARACTER_EXPORTS.values():
        assert callable(row.door)

    calls: list[str] = []
    monkeypatch.setattr(
        svc_characters, "export_godot", lambda *a, **k: calls.append("godot_scene") or "G"
    )
    monkeypatch.setattr(
        svc_export, "export_to_folder", lambda *a, **k: calls.append("animated_glb") or "A"
    )
    monkeypatch.setattr(svc_derive, "get_file", lambda *a, **k: Path("animated.glb"))

    job_id = store.new_id()
    assert svc_export.run_character_export(svc, "animated_glb", job_id) == "A"
    assert svc_export.run_character_export(svc, "godot_scene", job_id) == "G"
    assert set(calls) == set(svc_export.CHARACTER_EXPORTS)


def test_an_unknown_export_format_is_refused_on_format(svc):
    with pytest.raises(Invalid) as excinfo:
        svc_export.run_character_export(svc, "powerpoint", store.new_id())
    assert excinfo.value.field == "format"
