"""The open clip vocabulary: five new placeholder clips shipped alongside the
legacy five, on every species' library.

Schema v3 (``rigging.parse_clip_library``, ``service.clips._check_shape``) lets
a clip library carry any name, not just the five ``pipelines.charsheet.
ANIMATIONS`` table entries a character sheet lays out. This is the first
library to use that room: ``attack_02``, ``cast``, ``fall``, ``hit`` and
``death``, each marked ``provisional`` because a human animator owns the real
versions later (TODO P8). These tests are the claim that shipping them did not
disturb the legacy five, and that every new clip is well-formed by the same
doors a hand-authored or agent-generated one would have to pass.

Named per-clip frame/timing values are pinned to the shapes chosen when these
clips were authored:

* ``attack_02``: 5 keys, ``[2, 1, 1, 1]`` -> 6 frames, 80 ms/frame, one-shot.
* ``cast``: 4 keys, ``[2, 2, 1]`` -> 6 frames, 90 ms/frame, one-shot.
* ``fall``: 2 keys, ``[2, 2]``, closed -> 4 frames, 100 ms/frame, a cycle.
* ``hit``: 3 keys, ``[1, 2]`` -> 4 frames, 90 ms/frame, one-shot.
* ``death``: 4 keys, ``[1, 2, 2]`` -> 6 frames, 120 ms/frame, one-shot.
"""

from __future__ import annotations

import pytest

from warlock import rigging
from warlock.pipelines import charsheet, spritesynth
from warlock.pipelines import sheet as sheetlib
from warlock.service import clips as svc_clips

SPECIES = ("humanoid", "quadruped", "bird", "blob")

LEGACY_NAMES = ("idle", "walk", "run", "attack", "jump")
NEW_NAMES = ("attack_02", "cast", "fall", "hit", "death")
ALL_NAMES = LEGACY_NAMES + NEW_NAMES

#: (name -> expanded frame count), for the five new clips. Independent of
#: species: every library authored the same key/segment shape for each.
NEW_FRAME_COUNTS = {"attack_02": 6, "cast": 6, "fall": 4, "hit": 4, "death": 6}


@pytest.fixture(autouse=True)
def _fresh_clip_cache():
    """Same isolation every other clip-library test file uses: the loader
    caches are module globals filled once, and a test that reads one library
    must not leak into the next."""
    rigging.invalidate_clips()
    yield
    rigging.invalidate_clips()


def _library(species: str) -> dict:
    lib = rigging.clip_library(species)
    assert lib["clips"], f"{species} shipped no clips to check"
    return lib


def _clips_by_name(species: str) -> dict:
    return {c["name"]: c for c in _library(species)["clips"]}


# --- the vocabulary itself ----------------------------------------------------


@pytest.mark.parametrize("species", SPECIES)
def test_every_shipped_library_carries_the_open_vocabulary(species):
    by_name = _clips_by_name(species)
    assert set(by_name) == set(ALL_NAMES)


# --- the legacy five are untouched ---------------------------------------------


@pytest.mark.parametrize("species", SPECIES)
def test_the_legacy_five_keep_their_frame_counts_loops_and_times(species):
    lib = _library(species)
    by_name = {c["name"]: c for c in lib["clips"]}
    for name, frames, loop, duration_ms in charsheet.ANIMATIONS:
        clip = by_name[name]
        assert clip["closed"] is loop, f"{species}.{name}: closed flag changed"
        assert clip["duration_ms"] == duration_ms, f"{species}.{name}: duration_ms changed"
        expanded = sum(clip["segments"]) + (0 if clip["closed"] else 1)
        assert expanded == frames, f"{species}.{name}: frame count changed"


# --- one-shots vs. the cycle ----------------------------------------------------


@pytest.mark.parametrize("species", SPECIES)
def test_death_hit_cast_and_attack_02_are_one_shots_and_fall_is_a_cycle(species):
    by_name = _clips_by_name(species)
    for name in ("death", "hit", "cast", "attack_02"):
        assert by_name[name]["closed"] is False, f"{species}.{name} should be a one-shot"
    assert by_name["fall"]["closed"] is True, f"{species}.fall should be a cycle"


# --- every new clip expands to its authored frame count -----------------------


@pytest.mark.parametrize("species", SPECIES)
def test_every_new_clip_expands_to_its_authored_frame_count(species):
    lib = _library(species)
    by_name = {c["name"]: c for c in lib["clips"]}
    for name, want_frames in NEW_FRAME_COUNTS.items():
        clip = by_name[name]
        keys = [lib["poses"][k] for k in clip["keys"]]
        frames = sheetlib.interpolate_clip(
            keys,
            clip["segments"],
            closed=clip["closed"],
            easing=clip["easing"],
            space=lib["space"],
            clip_id=clip["name"],
        )
        assert len(frames) == want_frames, f"{species}.{name}: expanded to {len(frames)} frames"


# --- provisional marking --------------------------------------------------------


@pytest.mark.parametrize("species", SPECIES)
def test_every_new_clip_is_marked_provisional_and_the_legacy_five_are_not(species):
    by_name = _clips_by_name(species)
    for name in NEW_NAMES:
        assert by_name[name].get("provisional") is True, f"{species}.{name} must be provisional"
    for name in LEGACY_NAMES:
        assert not by_name[name].get("provisional"), f"{species}.{name} must not be provisional"


# --- no clip names a bone its skeleton lacks -----------------------------------


@pytest.mark.parametrize("species", SPECIES)
def test_no_clip_names_a_bone_its_skeleton_lacks(species):
    template = rigging.get_template(species)
    known = {b["name"] for b in template.bones}
    lib = _library(species)
    for pose in lib["poses"].values():
        unknown = set(pose["bones"]) - known
        assert not unknown, f"{species}: pose {pose['name']!r} names unknown bone(s) {unknown}"


# --- every authored quaternion is unit length -----------------------------------


@pytest.mark.parametrize("species", SPECIES)
def test_every_key_is_a_unit_quaternion(species):
    """Against the raw shipped JSON, not the loaded library: ``clip_library``
    runs every pose through ``rigging.validate_bones``, which renormalises a
    quaternion rather than rejecting it (see its docstring -- "a browser
    accumulating gizmo [drift]"), so checking the loaded library can never
    catch an authoring tool that shipped a non-unit quaternion. This reads
    ``templates/clips/<species>.json`` with a bare ``json.loads`` -- no
    ``rigging`` parse at all -- so a badly-authored key pose would actually
    fail it.
    """
    import json

    path = rigging.CLIP_DIR / f"{species}.json"
    raw = json.loads(path.read_text("utf-8"))
    for pose in raw["poses"]:
        for bone, q in pose["bones"].items():
            norm = sum(v * v for v in q) ** 0.5
            assert abs(norm - 1.0) < 1e-3, (
                f"{species}: pose {pose['name']!r} bone {bone!r} is not unit length ({norm})"
            )


# --- cast's tempo matches what Troupe's sprite synthesis expects ---------------


@pytest.mark.parametrize("species", SPECIES)
def test_cast_plays_at_the_sprite_synthesis_tempo(species):
    """Only ``closed``/``duration_ms`` are comparable against ``ACTION_PLAYBACK``:
    its ``(loop, duration_ms)`` pair says nothing about a frame count, because
    ``spritesynth`` resamples every action to the sheet geometry's own
    ``frames_per_direction`` (``sheet.resample_clip``) rather than playing the
    clip's authored key/segment count directly -- unlike the legacy five, whose
    *frame counts* additionally happen to be pinned in ``charsheet.ANIMATIONS``.
    """
    want_loop, want_ms = spritesynth.ACTION_PLAYBACK["cast"]
    clip = _clips_by_name(species)["cast"]
    assert clip["closed"] is want_loop
    assert clip["duration_ms"] == want_ms


# --- death ends grounded ---------------------------------------------------------


@pytest.mark.parametrize("species", SPECIES)
def test_death_ends_on_the_ground(species):
    lib = _library(species)
    death = next(c for c in lib["clips"] if c["name"] == "death")
    last_pose = lib["poses"][death["keys"][-1]]
    root = last_pose.get("root_translation")
    assert root is not None, f"{species}: death's last key carries no root_translation"
    assert root[2] < 0, f"{species}: death's last key does not sink below rest ({root})"


# --- the save door ----------------------------------------------------------------


@pytest.mark.parametrize("species", SPECIES)
def test_every_shipped_library_passes_the_save_door(svc, species):
    """The whole shipped library, round-tripped through the editor's own save
    door -- ``_check_shape``, then ``rigging.parse_clip_library`` a second time
    as the renderer's own authority, and (for the Troupe template) the frame
    table check -- the same three passes a hand-authored or agent-written
    library has to clear."""
    view = svc_clips.library(svc, species)
    payload = {"space": view["space"], "poses": view["poses"], "clips": view["clips"]}
    saved = svc_clips.save(svc, species, payload)
    assert {c["name"] for c in saved["clips"]} == set(ALL_NAMES)
