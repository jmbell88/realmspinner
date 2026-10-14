"""Tranche 3's service doors that survive: a dry-run recipe and an in-flight rig guard.

The follow-up sheet and the cropped sheet preview left with the character sheets
(``retired.py``): the chain ends at a rigged mesh. The paged asset listing, the
one-door job status and the reroll left with the character agent surface (the
2026-10-14 audit, finding poser-characters-07); ``test_audit_2026_10_14_c1_character_doors.py``
pins that they are gone.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from realmspinner.characters import family as family_mod
from realmspinner.kernels.rig import cliplib
from realmspinner.service import characters as svc_characters
from realmspinner.service import rig as svc_rig
from realmspinner.service.errors import Conflict, Invalid

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


def _rig_row(svc, mesh_id: str, *, status: str = "queued", template="humanoid") -> str:
    params = {"source_job": mesh_id, "template": template, "auto": True}
    return svc.store.create("rig", "a hooded ranger", params, stage="model", status=status)


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
    from realmspinner.characters import resolve as resolve_mod

    expected = resolve_mod.offer_sentence(resolve_mod.resolve("a manticore"))
    assert expected is not None
    with pytest.raises(Invalid) as excinfo:
        svc_characters.recipe_from_prompt(svc, "a manticore")
    assert excinfo.value.field == "prompt"
    assert str(excinfo.value) == expected

    with pytest.raises(Invalid) as generic:
        svc_characters.recipe_from_prompt(svc, "wibble wobble")
    assert generic.value.field == "prompt"
    from realmspinner.characters import family as family_mod

    # Counted off the registry, never typed (F20-7, create-17's rule): the
    # sentence keeps promising whatever the archetype registry holds.
    plans = family_mod.body_plans_phrase(len(family_mod.archetypes()))
    assert f"species across {plans}" in str(generic.value)

    with pytest.raises(Invalid) as blank:
        svc_characters.recipe_from_prompt(svc, "")
    assert blank.value.field == "prompt"
    assert "pass a prompt or a family" in str(blank.value)


def test_omitted_frames_follow_the_clip_library_at_the_requested_fps(svc):
    from realmspinner.clips import clip_timing
    from realmspinner.kernels import charsheet

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

    monkeypatch.setattr("realmspinner.doctor.blender_check", _boom)
    before = svc.store.list(1000)
    result = svc_characters.recipe_from_prompt(svc, "a fire ogre")
    after = svc.store.list(1000)
    assert before == after
    assert result["recipe"]["family"] == "ogre"
    # No sheet is laid out, so there is no cell count to report (the 2026-10-14 audit,
    # finding poser-jobs-02).
    assert "cells" not in result
    assert result["estimate_minutes"] > 0


# --- rig_in_flight ------------------------------------------------------------


def test_rig_in_flight_names_the_active_rig_row(svc):
    mesh_id = _mesh(svc)
    assert svc_rig.rig_in_flight(svc, mesh_id) is None

    rig_id = _rig_row(svc, mesh_id, status="queued")
    assert svc_rig.rig_in_flight(svc, mesh_id) == rig_id

    svc.store.set_status(rig_id, "done")
    assert svc_rig.rig_in_flight(svc, mesh_id) is None


def test_two_concurrent_create_rig_calls_for_one_mesh_mint_only_one_rig_row(
    svc, monkeypatch
):
    """The 2026-09-20 audit, finding agents-03: ``create_rig``'s
    ``rig_in_flight`` check and its ``store.create`` insert were two separate
    operations with nothing holding the gap between them, so two
    ``character_rig`` calls landing on AgentHost's two service workers could
    both read "no rig in flight" and both mint a rig row for the same mesh --
    both then finalizing into the same ``job_dir``. ``send_to_troupe``'s
    identical guard is correctly wrapped in ``convert_lock``, as this test's
    sibling above proves.

    Widened deterministically rather than raced on a sleep: ``rig_in_flight``
    is patched to pause, once, immediately after answering "nothing in
    flight" for this mesh, so a second call is proven to land in that exact
    gap before the first call's insert lands.
    """
    import threading

    monkeypatch.setattr(
        "realmspinner.doctor.blender_check", lambda: SimpleNamespace(ok=True, detail="")
    )
    mesh_id = _mesh(svc)

    real_rig_in_flight = svc_rig.rig_in_flight
    checked = threading.Event()
    release = threading.Event()
    armed = True

    def spy_rig_in_flight(svc_arg, job_id_arg):
        nonlocal armed
        result = real_rig_in_flight(svc_arg, job_id_arg)
        if armed and job_id_arg == mesh_id:
            armed = False
            checked.set()
            assert release.wait(5), "the first call's paused check never resumed"
        return result

    monkeypatch.setattr(svc_rig, "rig_in_flight", spy_rig_in_flight)

    results: list = []
    errors: list = []

    def call():
        try:
            results.append(svc_rig.create_rig(svc, mesh_id))
        except Conflict as exc:
            errors.append(exc)

    first = threading.Thread(target=call)
    first.start()
    assert checked.wait(5), "the first call's rig_in_flight check never ran"

    second = threading.Thread(target=call)
    second.start()
    # Pre-fix, nothing holds a lock at this point, so the second call runs to
    # completion here, inside the gap, and mints its own row. Post-fix, the
    # first call already holds ``convert_lock(job_id, "rig")`` before it ever
    # calls ``rig_in_flight``, so the second call blocks on that same lock
    # instead -- either way, giving it a moment to run (or to prove it
    # cannot) before releasing the first call is what makes the race
    # deterministic rather than sleep-dependent.
    second.join(timeout=1.0)

    release.set()
    first.join(5)
    second.join(5)
    assert not first.is_alive()
    assert not second.is_alive()

    rig_rows = [
        r for r in svc.store.list(limit=1000, kind="rig")
        if (r["params"] or {}).get("source_job") == mesh_id
    ]
    assert len(rig_rows) == 1, (
        f"two concurrent create_rig calls minted {len(rig_rows)} rig rows "
        f"for one mesh: {[r['id'] for r in rig_rows]}"
    )
    assert len(results) == 1
    assert len(errors) == 1
    assert errors[0].field == "job_id"


# --- shipped_clip_names --------------------------------------------------------


def test_shipped_clip_names_ignore_a_user_edited_library(svc):
    from realmspinner import poselib
    from realmspinner.service import clips as svc_clips

    shipped = cliplib.shipped_clip_names("humanoid")
    assert "walk" in shipped

    # The user clip directory is armed by the ``service.clips`` doors, not by service
    # construction (the 2026-10-14 audit, finding poser-poses-02), so a door is called first.
    svc_clips.library(svc, "humanoid")
    user_dir = poselib.clip_dir(svc.config)
    user_dir.mkdir(parents=True, exist_ok=True)
    edited = user_dir / "humanoid.json"
    edited.write_text(json.dumps({"version": 3, "poses": [], "clips": []}), encoding="utf-8")
    cliplib.invalidate_clips()
    try:
        assert cliplib.clip_library("humanoid")["clips"] == []
        assert cliplib.shipped_clip_names("humanoid") == shipped
    finally:
        edited.unlink()
        cliplib.invalidate_clips()


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


def test_the_service_refusal_counts_body_plans_from_the_registry_not_a_literal(svc, monkeypatch):
    """F20-7: the sentence said "four body plans" as typed text, so a fifth
    archetype would have shipped under a promise of four."""
    from realmspinner.characters import family as family_mod

    real = family_mod.archetypes()
    monkeypatch.setattr(
        family_mod, "archetypes", lambda: {**real, "extra_a": object(), "extra_b": object()}
    )
    with pytest.raises(Invalid) as excinfo:
        svc_characters.recipe_from_prompt(svc, "wibble wobble")
    assert len(real) == 4, "the expected wording below is spelled for four plus two"
    assert "across six body plans" in str(excinfo.value)
