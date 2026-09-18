"""Editing a template's clip library -- the Troupe programme's clip-authoring
half.

The clip *format* and its expansion shipped with Troupe; what did not was any
way to change a clip that is not editing JSON in the package tree. This is the
door that replaces that, and what is pinned here is the three properties that
make it safe to put a UI on:

* **an edit never touches the shipped library**, so an upgrade cannot silently
  overwrite the user's clips and a reverted user gets the build's improvements;
* **the editor cannot write a file the renderer refuses**, because the save runs
  the renderer's own parser *and* its own frame-count check before committing;
* **a refused save leaves the previous library exactly where it was**, since a
  half-saved clip library is discovered by the next character sheet rather than
  by the person who pressed Save.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from warlock import poselib
from warlock.kernels.rig import cliplib
from warlock.service import Conflict, Failed, Invalid, NotFound
from warlock.service import clips as svc_clips

TEMPLATE = "humanoid"


@pytest.fixture(autouse=True)
def _fresh_clip_cache():
    """The library caches are module globals filled once. A test that edits one
    must not leak into the next, and neither must the shipped read."""
    cliplib.invalidate_clips()
    yield
    cliplib.invalidate_clips()


def _shipped(svc) -> dict:
    return svc_clips.library(svc, TEMPLATE)


def _as_payload(view: dict) -> dict:
    return {
        "space": view["space"],
        "poses": view["poses"],
        "clips": view["clips"],
    }


# --- reading -----------------------------------------------------------------


def test_the_shipped_library_reads_as_unedited(svc):
    view = _shipped(svc)
    assert view["template"] == TEMPLATE
    assert view["edited"] is False
    assert {c["name"] for c in view["clips"]} >= {"idle", "walk", "run", "attack", "jump"}
    # Poses come back as an ordered *list* where the renderer keys them by name:
    # the editor shows a list a user reads top to bottom, and a dict has no
    # order to show.
    assert isinstance(view["poses"], list)
    assert all(p["name"] and p["bones"] for p in view["poses"])


def test_an_unknown_template_is_not_found(svc):
    with pytest.raises(NotFound) as caught:
        svc_clips.library(svc, "no-such-skeleton")
    assert caught.value.field == "template"


def test_an_unreadable_user_library_is_not_presented_as_edited_shipped_clips(svc):
    """Defect, fixed 2026-09-13: a user clip library that exists but fails to
    parse used to be swallowed the same way the *renderer's* loader swallows
    one -- log it, skip it, fall back to the shipped clips -- but
    ``service.clips.library()`` then presented that fallback with
    ``edited: True``, as if it were the user's own file. A user who saved
    from this door next would silently overwrite the very file that failed to
    parse. The honest answer is a refusal naming the read failure, not a
    quiet substitution."""
    path = poselib.clip_path(svc.config, TEMPLATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json at all", encoding="utf-8")
    cliplib.invalidate_clips()

    with pytest.raises(Invalid) as caught:
        svc_clips.library(svc, TEMPLATE)
    assert caught.value.field == "template"
    # And, in particular, never the shipped clips masquerading as edited.
    assert "could not be read" in caught.value.message


# --- saving ------------------------------------------------------------------


def test_a_saved_library_goes_to_the_user_copy_and_not_the_package(svc):
    """The whole storage argument in one test. An installed build replaces the
    package tree wholesale on upgrade and may not even be writable."""
    before = (cliplib.CLIP_DIR / f"{TEMPLATE}.json").read_bytes()
    view = _shipped(svc)
    payload = _as_payload(view)
    payload["clips"][0]["easing"] = "ease_out"
    saved = svc_clips.save(svc, TEMPLATE, payload)

    assert saved["edited"] is True
    assert poselib.clip_path(svc.config, TEMPLATE).is_file()
    assert (cliplib.CLIP_DIR / f"{TEMPLATE}.json").read_bytes() == before


def test_the_renderer_reads_the_edit_back_immediately(svc):
    """The caches are module globals filled once, which is right for a library
    that ships with the build and wrong the moment one can be authored. The save
    drops them; nothing polls a timestamp."""
    payload = _as_payload(_shipped(svc))
    for clip in payload["clips"]:
        if clip["name"] == "idle":
            clip["easing"] = "ease_out"
    svc_clips.save(svc, TEMPLATE, payload)

    found = cliplib.clip_library(TEMPLATE)
    idle = next(c for c in found["clips"] if c["name"] == "idle")
    assert idle["easing"] == "ease_out"


def test_reverting_deletes_the_user_copy_and_the_shipped_clips_return(svc):
    payload = _as_payload(_shipped(svc))
    payload["clips"][0]["easing"] = "ease_out"
    svc_clips.save(svc, TEMPLATE, payload)
    assert _shipped(svc)["edited"] is True

    view = svc_clips.revert(svc, TEMPLATE)
    assert view["edited"] is False
    assert not poselib.clip_path(svc.config, TEMPLATE).is_file()
    assert svc_clips.library(svc, TEMPLATE)["clips"][0]["easing"] != "ease_out"


def test_reverting_an_unedited_library_is_not_found(svc):
    with pytest.raises(NotFound):
        svc_clips.revert(svc, TEMPLATE)


def test_reverting_a_locked_clip_file_reports_as_a_failure_not_a_traceback(svc, monkeypatch):
    """poser-06, the 2026-09-11 audit: revert()'s path.unlink() was not
    wrapped in except OSError, unlike its sibling delete doors
    (service.poses.delete_library_pose, service.rig.delete_pose) -- so a
    locked file (antivirus, another program holding it open) raised a raw
    PermissionError out of the service layer instead of the same
    Failed("...locked by another program...") every other file-delete door
    gives."""
    payload = _as_payload(_shipped(svc))
    payload["clips"][0]["easing"] = "ease_out"
    svc_clips.save(svc, TEMPLATE, payload)

    path = poselib.clip_path(svc.config, TEMPLATE)
    real_unlink = Path.unlink

    def locked(self, *a, **k):
        if self == path:
            raise PermissionError("locked by another program")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", locked)
    with pytest.raises(Failed):
        svc_clips.revert(svc, TEMPLATE)

    # And the file is still there to retry against -- a failed revert must
    # not leave the library half-gone.
    assert path.is_file()


# --- the refusals ------------------------------------------------------------


def test_a_clip_naming_a_key_the_library_does_not_carry_is_refused(svc):
    """The failure the whole-file unit exists to prevent: a key renamed while a
    clip still points at the old name."""
    payload = _as_payload(_shipped(svc))
    payload["clips"][0]["keys"] = ["not a pose in this file", *payload["clips"][0]["keys"][1:]]
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert caught.value.field == "keys"
    assert "does not carry" in caught.value.message


def test_the_segment_count_must_match_the_keys_and_the_loop(svc):
    """An open clip of N keys has N-1 steps and a closed one has N, because the
    last key steps back to the first. Off-by-one here is the easiest way to
    author a clip whose frame count is wrong."""
    payload = _as_payload(_shipped(svc))
    walk = next(c for c in payload["clips"] if c["name"] == "walk")
    assert walk["closed"] is True and len(walk["keys"]) == len(walk["segments"])
    walk["segments"] = walk["segments"][:-1]
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert caught.value.field == "segments"
    # Both numbers, so the message is actionable rather than a rebuke.
    assert "needs 4 segment lengths, not 3" in caught.value.message


def test_opening_a_closed_clip_changes_how_many_segments_it_needs(svc):
    """The same rule from the other side -- and the reason it is stated as a
    rule rather than as a constant."""
    payload = _as_payload(_shipped(svc))
    walk = next(c for c in payload["clips"] if c["name"] == "walk")
    walk["closed"] = False
    with pytest.raises(Invalid, match="needs 3 segment lengths, not 4"):
        svc_clips.save(svc, TEMPLATE, payload)


@pytest.mark.parametrize("bad", [0, -1, svc_clips.MAX_SEGMENT + 1])
def test_a_segment_length_outside_the_band_is_refused(svc, bad: int):
    payload = _as_payload(_shipped(svc))
    payload["clips"][0]["segments"][0] = bad
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert caught.value.field == "segments"


def test_an_unknown_easing_is_refused_by_name(svc):
    """Offered by name rather than as a free string, so a typo is refused here
    instead of falling through to linear in the renderer."""
    payload = _as_payload(_shipped(svc))
    payload["clips"][0]["easing"] = "bouncy"
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert caught.value.field == "easing"
    assert "linear" in caught.value.message


def test_two_clips_with_one_name_are_refused(svc):
    payload = _as_payload(_shipped(svc))
    payload["clips"].append(dict(payload["clips"][0]))
    with pytest.raises(Conflict) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert caught.value.field == "clips"


def test_two_key_poses_with_one_name_are_refused(svc):
    """Names are the identity a clip references by, so two of them is not a
    duplicate, it is an ambiguity."""
    payload = _as_payload(_shipped(svc))
    payload["poses"].append(dict(payload["poses"][0]))
    with pytest.raises(Conflict) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert caught.value.field == "poses"


def test_a_one_key_clip_is_refused(svc):
    payload = _as_payload(_shipped(svc))
    clip = payload["clips"][0]
    clip["keys"] = clip["keys"][:1]
    clip["segments"] = clip["segments"][:1]
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert caught.value.field == "keys"
    assert "a pose" in caught.value.message


def test_a_library_with_an_empty_timing_segment_is_refused(svc):
    """Resampling accepts any positive timing weights, never an empty leg."""
    payload = _as_payload(_shipped(svc))
    walk = next(c for c in payload["clips"] if c["name"] == "walk")
    walk["segments"] = [0] * len(walk["segments"])
    with pytest.raises(Invalid):
        svc_clips.save(svc, TEMPLATE, payload)


def test_dropping_a_clip_troupe_needs_is_refused_by_name(svc):
    payload = _as_payload(_shipped(svc))
    payload["clips"] = [c for c in payload["clips"] if c["name"] != "walk"]
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert "walk" in caught.value.message


def test_a_non_finite_root_translation_is_refused_not_round_tripped(svc):
    """poser-05 (the 2026-09-07 audit): a key pose's ``root_translation`` was
    converted with a bare ``float(v)`` here, with neither the finite nor the
    magnitude check ``poselib.validate_record`` already applies to the
    identical field on a library pose -- so ``[nan, 1e30, 0.0]`` round-tripped
    straight into a saved clip library."""
    payload = _as_payload(_shipped(svc))
    payload["poses"][0]["root_translation"] = [float("nan"), 1e30, 0.0]
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert caught.value.field == "poses"


def test_a_malformed_root_translation_is_a_field_addressed_refusal(svc):
    """poser-06: the same door let a malformed ``root_translation`` raise a
    bare ``ValueError`` out of ``_check_shape`` instead of a field-addressed
    ``Invalid``, so the UI could only show the generic "Something went
    wrong" with no control to point at."""
    payload = _as_payload(_shipped(svc))
    payload["poses"][0]["root_translation"] = "not-a-vector"
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert caught.value.field == "poses"


# --- the property that matters most ------------------------------------------


def test_a_refused_save_leaves_the_previous_library_byte_for_byte(svc):
    """A half-saved clip library is discovered by the *next character sheet*
    rather than by the person who pressed Save. The frame-table check runs
    through the ordinary read path -- which means it runs after the write -- so
    the previous bytes have to go back when it says no."""
    good = _as_payload(_shipped(svc))
    for clip in good["clips"]:
        if clip["name"] == "idle":
            clip["easing"] = "ease_out"
    svc_clips.save(svc, TEMPLATE, good)
    path = poselib.clip_path(svc.config, TEMPLATE)
    kept = path.read_bytes()

    bad = _as_payload(svc_clips.library(svc, TEMPLATE))
    walk = next(c for c in bad["clips"] if c["name"] == "walk")
    walk["segments"] = [0] * len(walk["segments"])
    with pytest.raises(Invalid):
        svc_clips.save(svc, TEMPLATE, bad)

    assert path.read_bytes() == kept
    idle = next(c for c in cliplib.clip_library(TEMPLATE)["clips"] if c["name"] == "idle")
    assert idle["easing"] == "ease_out", "the cache must be back on the kept file too"


def test_a_first_save_that_is_refused_leaves_no_file_at_all(svc):
    """The same rule with nothing to restore: an unedited library must still be
    unedited afterwards, not edited-and-broken."""
    bad = _as_payload(_shipped(svc))
    walk = next(c for c in bad["clips"] if c["name"] == "walk")
    walk["segments"] = [0] * len(walk["segments"])
    with pytest.raises(Invalid):
        svc_clips.save(svc, TEMPLATE, bad)
    assert not poselib.clip_path(svc.config, TEMPLATE).is_file()
    assert svc_clips.library(svc, TEMPLATE)["edited"] is False


def test_the_saved_file_is_what_the_renderers_own_parser_reads(svc):
    """An editor that can write a file the renderer cannot open is the one bug
    this design exists to make impossible, so the save runs that parser rather
    than a validation of its own shape."""
    payload = _as_payload(_shipped(svc))
    svc_clips.save(svc, TEMPLATE, payload)
    raw = json.loads(poselib.clip_path(svc.config, TEMPLATE).read_text(encoding="utf-8"))
    parsed = cliplib.parse_clip_library(raw)
    assert {c["name"] for c in parsed["clips"]} == {c["name"] for c in payload["clips"]}


def test_the_two_spellings_of_the_clip_directory_agree(svc):
    """``poselib.clip_dir`` is the app asking where to write and
    ``cliplib.user_clip_dir`` is the loader asking where to read. The loader may
    not import ``config``, so they are two expressions of one path -- and this
    is what asserts they are the same one."""
    assert poselib.clip_dir(svc.config) == cliplib.user_clip_dir()


# --- the scrubber -------------------------------------------------------------


def test_a_clip_expands_to_one_pose_per_frame(svc):
    out = svc_clips.preview_frames(svc, TEMPLATE, "walk")
    assert out["clip"] == "walk"
    walk = next(c for c in _shipped(svc)["clips"] if c["name"] == "walk")
    assert len(out["frames"]) == sum(walk["segments"])
    assert all(f.get("bones") for f in out["frames"])


def test_the_scrubber_goes_through_the_renderers_own_interpolator(svc):
    """A preview with its own interpolation would be a second opinion about what
    a walk is, which is the whole thing one clip library exists to avoid. The
    check is that an easing change moves the frames the scrubber shows."""
    before = svc_clips.preview_frames(svc, TEMPLATE, "walk")["frames"]
    payload = _as_payload(_shipped(svc))
    walk = next(c for c in payload["clips"] if c["name"] == "walk")
    walk["easing"] = "ease_out"
    svc_clips.save(svc, TEMPLATE, payload)
    after = svc_clips.preview_frames(svc, TEMPLATE, "walk")["frames"]

    assert len(after) == len(before)
    assert after != before


def test_ease_and_linear_are_the_same_clip_at_the_shipped_segment_lengths(svc):
    """**A property worth knowing before authoring 22 keyframes**, and the
    reason this is a test rather than a comment.

    Easing reshapes *where inside a segment* the frames land, so it has nothing
    to do until a segment holds three. Every shipped segment is 1 or 2 frames,
    so the only sample points are t=0 and t=0.5 -- and ``ease`` is a smoothstep
    with ``smoothstep(0.5) == 0.5`` exactly. The ``"easing": "ease"`` on the
    shipped ``idle`` clip is therefore decorative: it renders identically to
    ``linear``. ``ease_in`` (0.25) and ``ease_out`` (0.75) genuinely differ,
    which is what the test above uses.

    Nothing here is wrong -- it is a consequence of short segments -- but an
    author who changes an easing and sees no change deserves to find this
    written down rather than conclude the field is ignored."""
    linear = svc_clips.preview_frames(svc, TEMPLATE, "walk")["frames"]
    payload = _as_payload(_shipped(svc))
    walk = next(c for c in payload["clips"] if c["name"] == "walk")
    assert max(walk["segments"]) <= 2, "the premise: no shipped segment reaches 3"
    walk["easing"] = "ease"
    svc_clips.save(svc, TEMPLATE, payload)
    assert svc_clips.preview_frames(svc, TEMPLATE, "walk")["frames"] == linear


def test_an_unknown_clip_is_not_found(svc):
    with pytest.raises(NotFound) as caught:
        svc_clips.preview_frames(svc, TEMPLATE, "moonwalk")
    assert caught.value.field == "clip"


# --- rotation space ---------------------------------------------------------
#
# A library's ``space`` says how to read every quaternion it holds, and nothing
# downstream re-derives it. Both of these refusals exist because the only place
# a wrong space shows up otherwise is a mangled character sheet.


def test_save_refuses_an_unknown_rotation_space(svc):
    payload = _as_payload(_shipped(svc))
    payload["space"] = "local"
    with pytest.raises(Invalid) as excinfo:
        svc_clips.save(svc, TEMPLATE, payload)
    assert excinfo.value.field == "space"
    assert "local" in str(excinfo.value)


def test_save_refuses_changing_the_rotation_space(svc):
    view = _shipped(svc)
    assert view["space"] == "delta"
    payload = _as_payload(view)
    payload["space"] = "node"
    with pytest.raises(Invalid) as excinfo:
        svc_clips.save(svc, TEMPLATE, payload)
    assert excinfo.value.field == "space"
    assert not poselib.clip_path(svc.config, TEMPLATE).is_file()


def test_save_refuses_a_clip_library_the_read_door_could_never_load_back(svc):
    """The 2026-09-13 audit, finding poser-02: ``_check_shape`` bounds keys and
    segments but not bones per pose or the serialized whole, so a save could
    write a file bigger than ``cliplib.MAX_CLIP_LIBRARY_BYTES`` -- the exact
    cap ``_load_clip_library`` enforces on read. Without this check, such a
    save lands on disk, is silently skipped by every later read, and the
    template reverts to the shipped clips with no error saying why.

    One pose with enough bones to push the serialized library past the
    4 MiB read cap; the reproduction (``poser-poses-01.py``) used 160,000
    bones for a 16 MB file, this uses fewer but still comfortably over it.
    """
    view = _shipped(svc)
    huge_bones = {f"bone{i}": [1.0, 0.0, 0.0, 0.0] for i in range(50_000)}
    payload = {
        "space": view["space"],
        "poses": [{"name": "big", "bones": huge_bones}],
        "clips": [
            {
                "name": "clip1",
                "keys": ["big", "big"],
                "segments": [1],
                "closed": False,
                "easing": "linear",
                "duration_ms": 1000,
            }
        ],
    }
    with pytest.raises(Conflict) as excinfo:
        svc_clips.save(svc, TEMPLATE, payload)
    assert excinfo.value.field == "poses"
    assert not poselib.clip_path(svc.config, TEMPLATE).is_file(), (
        "a refused save must not land on disk, even staged"
    )
