"""``service.clip_import``: the door "Import clip" calls -- sampling an
external animation with Blender (faked here, as ``test_pose_library_service.py``
fakes ``blender_run.run_worker`` for the template preview) and, optionally, folding
the converted clip into a template's clip library.

What is pinned:

* :func:`analyse` never writes to the library and always cleans up its own
  scratch result file, success or failure;
* every Blender-shaped failure (missing file, wrong extension, no Blender, a
  worker that ran but said ``ok: false``, a conversion ``cliptransfer`` itself
  refuses) reaches the caller as a ``service.errors`` refusal naming a field,
  never a traceback;
* :func:`import_into_library` merges under ``service.clips``'s own lock and
  through its own ``_check_shape``/``_commit_locked``, so a name clash, a
  pose-name collision and a refused render check behave exactly as they would
  for a hand-edited save.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_cliptransfer import (
    _axis_angle,
    _baseline_source_bones,
    _entry,
    _make_sample,
    _rest_frame_bones,
)

from warlock import clips as pure_clips
from warlock import cliptransfer, doctor, poselib
from warlock.doctor import Check
from warlock.kernels.rig import cliplib
from warlock.pipelines import blender_run
from warlock.service import Conflict, Failed, Invalid, clip_import
from warlock.service import clips as svc_clips

TEMPLATE = "humanoid"


@pytest.fixture(autouse=True)
def _fresh_clip_cache():
    """The library caches are module globals filled once -- the same isolation
    ``tests/test_clip_editing.py`` and ``tests/test_clip_library_v3.py`` use."""
    cliplib.invalidate_clips()
    yield
    cliplib.invalidate_clips()


def _canned_payload() -> dict:
    """A minimal, synthetic worker result: two frames, one bone (``LeftArm``)
    swinging 30 degrees between them so the converted clip carries real
    motion -- a wholly static sample would leave every key pose's ``bones``
    empty, which ``service.clips._check_shape`` refuses on its own terms.
    Built from ``tests/test_cliptransfer.py``'s own fixtures, per the module
    docstring's claim that this whole pipeline is decidable with no Blender.
    """
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    moved = dict(rest)
    moved["LeftArm"] = _entry(_axis_angle((0, 0, 1), 30.0), source_bones["LeftArm"]["head"])
    action = {
        "name": "Motion",
        "fps": 30.0,
        "frame_start": 0,
        "frame_end": 1,
        "frames": [{"frame": 0, "bones": rest}, {"frame": 1, "bones": moved}],
    }
    sample = _make_sample(source_bones, [action])
    return {"ok": True, "armature": "Armature", **sample}


def _fake_run_worker(monkeypatch, payload: dict) -> list[dict]:
    """Stand in for ``blender_run.run_worker``: write *payload* to the spec's own
    result path (the real worker's hand-off shape) and hand it back, without
    cleaning the file up itself -- so a test that wants to see this module's
    own cleanup run is exercising it rather than the real worker's."""
    calls: list[dict] = []

    def run_worker(spec, **kwargs):
        calls.append(spec)
        result_path = Path(spec["result_path"])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return json.loads(result_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(blender_run, "run_worker", run_worker)
    return calls


def _ok_blender(monkeypatch, ok: bool = True, detail: str = "bpy 5.2.0") -> None:
    monkeypatch.setattr(
        doctor, "blender_check", lambda **kw: Check("Blender (rigging)", ok, detail, False)
    )


def _write_source(tmp_path, name: str = "clip.fbx") -> Path:
    path = tmp_path / name
    path.write_bytes(b"stub")
    return path


def _imports_dir(svc) -> Path:
    return Path(svc.config.data_dir) / "poser" / "imports"


# --- analyse: no side effects, every refusal is a service error ------------


def test_analyse_writes_nothing(svc, monkeypatch, tmp_path):
    _ok_blender(monkeypatch)
    _fake_run_worker(monkeypatch, _canned_payload())
    source = _write_source(tmp_path)
    shipped_before = (cliplib.CLIP_DIR / f"{TEMPLATE}.json").read_bytes()

    result = clip_import.analyse(svc, TEMPLATE, str(source))

    assert result["template"] == TEMPLATE
    assert len(result["clips"]) == 1
    assert svc_clips.library(svc, TEMPLATE)["edited"] is False
    assert (cliplib.CLIP_DIR / f"{TEMPLATE}.json").read_bytes() == shipped_before
    assert not poselib.clip_path(svc.config, TEMPLATE).is_file()
    imports_dir = _imports_dir(svc)
    assert not (imports_dir.is_dir() and list(imports_dir.iterdir()))


def test_a_file_that_is_not_an_animation_format_is_refused_on_source(svc, tmp_path):
    source = tmp_path / "reference.png"
    source.write_bytes(b"not an animation")
    with pytest.raises(Invalid) as caught:
        clip_import.analyse(svc, TEMPLATE, str(source))
    assert caught.value.field == "source"


def test_a_missing_file_is_refused_on_source(svc, tmp_path):
    with pytest.raises(Invalid) as caught:
        clip_import.analyse(svc, TEMPLATE, str(tmp_path / "no-such-file.fbx"))
    assert caught.value.field == "source"


def test_missing_blender_is_a_refusal_not_a_traceback(svc, monkeypatch, tmp_path):
    _ok_blender(monkeypatch, ok=False, detail="bpy import failed -- install with: uv sync")
    source = _write_source(tmp_path)
    with pytest.raises(Failed, match="install with"):
        clip_import.analyse(svc, TEMPLATE, str(source))


def test_a_worker_sentence_reaches_the_caller(svc, monkeypatch, tmp_path):
    _ok_blender(monkeypatch)
    _fake_run_worker(monkeypatch, {"ok": False, "error": "no known skeleton"})
    source = _write_source(tmp_path)
    with pytest.raises(Failed, match="no known skeleton"):
        clip_import.analyse(svc, TEMPLATE, str(source))


def test_the_result_file_is_removed_even_when_transfer_fails(svc, monkeypatch, tmp_path):
    _ok_blender(monkeypatch)
    _fake_run_worker(monkeypatch, _canned_payload())
    source = _write_source(tmp_path)

    # frames=999 is refused by cliptransfer.transfer itself (over
    # MAX_CLIP_FRAMES), well after run_worker has already handed back its
    # payload -- this is what proves the cleanup is this module's own,
    # not something the (faked) worker call did for it.
    with pytest.raises(Invalid, match="frames"):
        clip_import.analyse(svc, TEMPLATE, str(source), frames=999)

    imports_dir = _imports_dir(svc)
    assert not (imports_dir.is_dir() and list(imports_dir.iterdir()))


# --- import_into_library -----------------------------------------------------


def test_import_into_library_adds_the_clip_and_it_parses(svc, monkeypatch, tmp_path):
    _ok_blender(monkeypatch)
    _fake_run_worker(monkeypatch, _canned_payload())
    source = _write_source(tmp_path)

    result = clip_import.import_into_library(svc, TEMPLATE, str(source), clip_name="brought_in")

    assert result["added"] == ["brought_in"]
    assert result["replaced"] == []
    view = svc_clips.library(svc, TEMPLATE)
    assert view["edited"] is True
    names = {c["name"] for c in view["clips"]}
    assert "brought_in" in names
    # The renderer's own parser accepts what was written -- the whole point
    # of routing through ``service.clips``'s private commit rather than a
    # second writer.
    cliplib.invalidate_clips()
    cliplib.parse_clip_library(json.loads(poselib.clip_path(svc.config, TEMPLATE).read_text()))


def test_the_imported_clip_records_where_it_came_from(svc, monkeypatch, tmp_path):
    _ok_blender(monkeypatch)
    _fake_run_worker(monkeypatch, _canned_payload())
    source = _write_source(tmp_path, name="mocap_run.fbx")

    result = clip_import.import_into_library(svc, TEMPLATE, str(source), clip_name="brought_in")
    expected_map = result["reports"][0]["map"]

    view = svc_clips.library(svc, TEMPLATE)
    clip = next(c for c in view["clips"] if c["name"] == "brought_in")
    assert clip["source"]["file"] == "mocap_run.fbx"
    assert clip["source"]["map"] == expected_map
    assert isinstance(clip["source"]["imported"], str) and clip["source"]["imported"]


def test_import_into_library_refuses_a_name_clash_without_replace(svc, monkeypatch, tmp_path):
    _ok_blender(monkeypatch)
    _fake_run_worker(monkeypatch, _canned_payload())
    source = _write_source(tmp_path)

    with pytest.raises(Conflict) as caught:
        clip_import.import_into_library(svc, TEMPLATE, str(source), clip_name="idle")
    assert caught.value.field == "name"
    # Nothing was written on the refusal.
    assert not poselib.clip_path(svc.config, TEMPLATE).is_file()


def test_importing_into_a_node_space_library_is_refused(svc, monkeypatch, tmp_path):
    """``cliptransfer.transfer`` always emits delta-space bases (rest-relative
    deltas the target's own rig applies on top of its rest pose) -- folding
    that into a node-space library (whole-world-space quaternions) would
    silently reinterpret every imported pose as something else, the same
    hazard ``service.clips.save`` already refuses by name for a hand edit
    that tries to change a library's stored ``space``. Built the way
    ``tests/test_clip_library_v3.py`` builds a fixture library: replace the
    shipped file underneath ``cliplib.CLIP_DIR`` rather than going through
    ``svc_clips.save`` (which would itself refuse changing a library's space
    away from its current one, and this needs a *node*-space library to
    exist in the first place)."""
    _ok_blender(monkeypatch)
    _fake_run_worker(monkeypatch, _canned_payload())
    source = _write_source(tmp_path)

    raw = json.loads((cliplib.CLIP_DIR / f"{TEMPLATE}.json").read_text(encoding="utf-8"))
    raw["space"] = "node"
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir()
    (clip_dir / f"{TEMPLATE}.json").write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(cliplib, "CLIP_DIR", clip_dir)
    cliplib.invalidate_clips()
    assert svc_clips.library(svc, TEMPLATE)["space"] == "node"

    with pytest.raises(Invalid) as caught:
        clip_import.import_into_library(svc, TEMPLATE, str(source), clip_name="brought_in")
    assert caught.value.field == "template"
    assert "delta" in str(caught.value)
    # Nothing was written: no user copy was ever created for this refusal.
    assert not poselib.clip_path(svc.config, TEMPLATE).is_file()


def test_replace_swaps_the_clip_in_place(svc, monkeypatch, tmp_path):
    _ok_blender(monkeypatch)
    _fake_run_worker(monkeypatch, _canned_payload())
    source = _write_source(tmp_path, name="mocap_idle.fbx")

    result = clip_import.import_into_library(
        svc, TEMPLATE, str(source), clip_name="idle", replace=True
    )

    assert result["added"] == []
    assert result["replaced"] == ["idle"]
    view = svc_clips.library(svc, TEMPLATE)
    clips = [c for c in view["clips"] if c["name"] == "idle"]
    assert len(clips) == 1, "the shipped idle was replaced, not duplicated"
    assert clips[0]["source"]["file"] == "mocap_idle.fbx"


def test_imported_pose_names_never_overwrite_existing_poses(svc, monkeypatch, tmp_path):
    _ok_blender(monkeypatch)
    _fake_run_worker(monkeypatch, _canned_payload())
    source = _write_source(tmp_path)

    clip_import.import_into_library(svc, TEMPLATE, str(source), clip_name="mine")
    before = svc_clips.library(svc, TEMPLATE)
    original_pose = next(p for p in before["poses"] if p["name"] == "mine k00")
    original_bones = original_pose["bones"]

    # Re-importing the same clip name with replace=True drops the *clip* but,
    # by this door's documented choice, leaves its now-orphaned key poses --
    # so the second import's own "mine k00"/"mine k01" collide with them and
    # must be renamed rather than overwrite them.
    _fake_run_worker(monkeypatch, _canned_payload())
    clip_import.import_into_library(
        svc, TEMPLATE, str(source), clip_name="mine", replace=True
    )

    after = svc_clips.library(svc, TEMPLATE)
    pose_names = {p["name"] for p in after["poses"]}
    assert "mine k00" in pose_names, "the orphaned original pose was left alone"
    assert "mine k00 2" in pose_names, "the new import's colliding key was renamed"
    kept = next(p for p in after["poses"] if p["name"] == "mine k00")
    assert kept["bones"] == original_bones, "the orphaned pose was not mutated"
    new_clip = next(c for c in after["clips"] if c["name"] == "mine")
    assert "mine k00 2" in new_clip["keys"]
    assert "mine k00" not in new_clip["keys"]


def test_import_is_not_written_when_the_render_check_refuses(svc, monkeypatch, tmp_path):
    _ok_blender(monkeypatch)
    _fake_run_worker(monkeypatch, _canned_payload())
    source = _write_source(tmp_path)

    def _boom(template_key, layout=None):
        raise ValueError("the frame table refuses this")

    monkeypatch.setattr(pure_clips, "expand_clips", _boom)

    with pytest.raises(Invalid):
        clip_import.import_into_library(svc, TEMPLATE, str(source), clip_name="brought_in")

    assert not poselib.clip_path(svc.config, TEMPLATE).is_file()


def test_import_into_library_refuses_a_clip_library_the_read_door_could_never_load_back(
    svc, monkeypatch, tmp_path
):
    """The 2026-09-14 audit, finding poser-01: ``service.clips.save`` gained a
    check against ``cliplib.MAX_CLIP_LIBRARY_BYTES`` on 2026-09-13 (finding
    poser-02) because ``_check_shape`` bounds keys and segments but not bones
    per pose or the serialized whole. ``import_into_library`` merges into the
    exact same document shape through the exact same
    ``_check_shape``/``_commit_locked`` pair, but was never given the same
    size check -- so an imported animation with enough bones (a dense facial
    or cloth rig baked in by the source file) can still write a library the
    read door then refuses forever after, reverting the template to the
    shipped clips with no error saying why.

    ``cliptransfer.transfer`` is faked directly (as this module fakes
    ``blender_run.run_worker`` for the Blender step) to hand back one pose with
    50,000 bones -- comfortably over the 4 MiB cap once serialized, the same
    bone count ``test_clip_editing.py``'s sibling test for ``save`` uses.
    """
    _ok_blender(monkeypatch)
    _fake_run_worker(monkeypatch, _canned_payload())
    source = _write_source(tmp_path)

    huge_bones = {f"bone{i}": [1.0, 0.0, 0.0, 0.0] for i in range(50_000)}

    def _huge_transfer(sample, *, template, clip_name, frames, loop, root_motion):
        name = clip_name or "brought_in"
        return [
            {
                "clip": {
                    "name": name,
                    "keys": ["big", "big"],
                    "segments": [1],
                    "closed": False,
                    "easing": "linear",
                    "duration_ms": 1000,
                },
                "poses": {"big": {"bones": huge_bones}},
                "report": {"map": "auto"},
            }
        ]

    monkeypatch.setattr(cliptransfer, "transfer", _huge_transfer)

    with pytest.raises(Conflict) as excinfo:
        clip_import.import_into_library(svc, TEMPLATE, str(source), clip_name="brought_in")
    assert excinfo.value.field == "poses"
    assert not poselib.clip_path(svc.config, TEMPLATE).is_file(), (
        "a refused import must not land on disk, even staged"
    )
