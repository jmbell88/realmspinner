"""How a finished character leaves the app as files on disk: ``service.characters.export_godot``.

The animated GLB door, the ``CHARACTER_EXPORTS`` menu and ``run_character_export`` left with
the character agent surface (the 2026-10-14 audit, finding poser-characters-07); the
inspector's "Export for Godot..." is the one caller of what remains. The sprite-sheet pair
and the frame folders left earlier, with the character sheets (``retired.py``). The door
renders nothing -- it works off ``animated.glb`` (``derive`` already knows how to bake and
refresh it) -- so what is asserted here is the *shape* of what lands on disk: a renamed copy
beside its Godot scene that never touches the served original.
"""

from __future__ import annotations

import json
import struct

import pytest

from realmspinner import clips
from realmspinner.kernels.geom3d import glbio
from realmspinner.service import characters as svc_characters
from realmspinner.service import derive as svc_derive
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


# --- the folder name ------------------------------------------------------------


def test_a_godot_re_export_takes_a_new_name_and_never_replaces(svc, tmp_path):
    job_id = _rigged_and_animated(svc, animation_names=["idle"], loops=["idle"], name="Ranger")
    out = tmp_path / "out"
    first = svc_characters.export_godot(svc, job_id, dest_dir=out)
    (first / "hand_attached.gd").write_text("extends Node", "utf-8")
    second = svc_characters.export_godot(svc, job_id, dest_dir=out)
    assert second.name == "Ranger-godot-2" and first.name == "Ranger-godot"
    assert (first / "hand_attached.gd").exists()
    # the scene inside names itself after its own (new) folder
    assert (second / "Ranger-godot-2.tscn").exists() and (second / "Ranger-godot-2.glb").exists()


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
