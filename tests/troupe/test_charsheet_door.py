"""The character-sheet door: ``check_troupe`` and ``create_charsheet``.

Created for the 2026-09-11 audit's fallback instruction -- this door's tests
otherwise live scattered across ``test_troupe_chain.py`` (not a file this
programme owns) -- to hold the layout-refusal regression the audit asked for
by name.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from warlock.service import characters as svc_characters
from warlock.service import troupe as svc_troupe
from warlock.service.errors import Invalid


def _mesh(svc, *, rigged: bool) -> str:
    job_id = svc.store.create("image", "a hooded ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    if rigged:
        (job_dir / "rig.glb").write_bytes(b"fake-rig")
        (job_dir / "rig.json").write_text(
            json.dumps({"template": "humanoid"}), "utf-8"
        )
    svc.store.set_status(job_id, "done")
    return job_id


def test_a_layout_that_cannot_be_planned_is_refused_with_field_layout(svc, monkeypatch):
    """The 2026-09-11 audit, finding troupe-01.

    ``check_troupe``, ``create_charsheet`` and ``_charsheet_spec`` all plan a
    layout and throw the plan away, purely to refuse now what would otherwise
    fail an hour later in the worker -- but the ``except ValueError`` branch
    that catches an unrenderable plan (a frame count ``expand_clips`` cannot
    fill, or an atlas over the texture limit) called ``invalid_from`` with no
    ``field=``, even though ``panes/troupe_settings.py`` calls
    ``form_ui.note("layout")`` specifically to draw this refusal on the layout
    table (see the comment at ``troupe_settings.py:148``).

    Forced with a monkeypatch on ``expand_clips`` rather than a real
    unplannable layout, the way ``test_troupe_chain.py``'s sibling test for the
    ``KeyError`` branch already does: the shipped humanoid library fills every
    layout ``resolve_layout`` can build, so there is no live request that
    reaches this branch today. What is under test is that when planning does
    fail, the refusal is addressed to the control that asks the question.
    """

    def boom(template, layout):
        raise ValueError("the atlas is over the texture limit")

    monkeypatch.setattr(svc_troupe, "expand_clips", boom)
    monkeypatch.setattr(
        "warlock.doctor.blender_check", lambda: SimpleNamespace(ok=True, detail="")
    )

    with pytest.raises(Invalid) as at_reference:
        svc_troupe.check_troupe(svc, {})
    assert at_reference.value.field == "layout"

    rigged = _mesh(svc, rigged=True)
    with pytest.raises(Invalid) as at_direct_door:
        svc_troupe.create_charsheet(svc, rigged)
    assert at_direct_door.value.field == "layout"

    unrigged = _mesh(svc, rigged=False)
    with pytest.raises(Invalid) as at_send_door:
        svc_troupe.send_to_troupe(svc, unrigged)
    assert at_send_door.value.field == "layout"


def test_a_recipes_layout_that_cannot_be_planned_is_refused_with_field_layout(monkeypatch):
    """``characters.py``'s own copy of the same plan-and-throw-away shape --
    ``_plan``, called from ``create_character`` before a single byte of mesh is
    spent -- carried the identical fieldless ``except ValueError`` branch.

    Exercised directly against ``_plan`` rather than through the whole
    ``create_character`` door: what is under test is one branch's ``field=``,
    and going through ``create_character`` would spend a real
    ``instantiate_mod.instantiate`` call to reach it.
    """
    from warlock.characters import family as family_mod
    from warlock.characters.recipe import Recipe

    fam = family_mod.get_family("human")
    spec = Recipe.from_dict(
        {
            "family": "human",
            "theme": fam.themes[0].key,
            "animations": {"idle": 2},
            "directions": 1,
        }
    )

    def boom(clip_library, layout):
        raise ValueError("the atlas is over the texture limit")

    monkeypatch.setattr("warlock.clips.expand_clips", boom)

    with pytest.raises(Invalid) as excinfo:
        svc_characters._plan(spec, "humanoid", spec.logical_size)
    assert excinfo.value.field == "layout"
