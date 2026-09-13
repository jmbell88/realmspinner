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

from warlock.pipelines import charsheet
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


# --- the open clip vocabulary -------------------------------------------------
#
# ``docs/measurements/2026-09-12-troupe-open-clip-vocabulary.md``: a layout may
# now name any clip the rig's own library defines, not just the closed
# ``charsheet.ANIMATIONS`` five.


def test_troupe_options_offer_every_clip_the_default_skeleton_defines(svc):
    options = svc_troupe.troupe_options(svc)
    vocabulary = options["clip_vocabulary"][svc_troupe.TROUPE_TEMPLATE]
    names = {clip["name"] for clip in vocabulary}
    assert names == {
        "idle", "walk", "run", "attack", "jump",
        "attack_02", "cast", "fall", "hit", "death",
    }
    assert options["fps_choices"] == list(charsheet.FPS_CHOICES)
    # True writes no key on a request -- see ``_check_options`` -- but the
    # options block still states it as the form's starting value.
    assert options["defaults"]["pixel_art"] is True


def test_troupe_options_mark_the_provisional_clips(svc):
    vocabulary = svc_troupe.troupe_options(svc)["clip_vocabulary"][
        svc_troupe.TROUPE_TEMPLATE
    ]
    provisional = {c["name"] for c in vocabulary if c["provisional"]}
    assert provisional == {"attack_02", "cast", "fall", "hit", "death"}
    default_names = {c["name"] for c in vocabulary if c["default"]}
    assert default_names == {"idle", "walk", "run", "attack", "jump"}


def test_a_sheet_naming_hit_is_accepted_on_a_skeleton_that_has_it(svc):
    """``hit`` is not one of the closed five, but it is the humanoid rig's own
    clip, so ``check_troupe`` -- which times its layout against
    ``TROUPE_TEMPLATE``'s library -- accepts it."""
    checked = svc_troupe.check_troupe(
        svc,
        {"layout": {"version": 3, "movements": [{"key": "hit", "direction_preset": 1}]}},
    )
    assert checked["layout"]["movements"][0]["key"] == "hit"
    assert checked["layout"]["movements"][0]["frames"] == 4


def test_a_clip_missing_from_the_rigs_library_is_refused_on_a_field_the_pane_draws(
    svc, monkeypatch
):
    """``backflip`` is nobody's clip. ``check_troupe`` and ``create_charsheet``
    refuse it on ``field="layout"`` -- their pane (``troupe_settings.py``) draws
    only the layout table -- and ``send_to_troupe``'s unrigged branch (which
    plans through ``_charsheet_spec``) refuses it on ``field="template"``,
    because its pane (``troupe_send.py``) draws a Skeleton control instead."""
    bad_layout = {
        "version": 3,
        "movements": [{"key": "backflip", "direction_preset": 1}],
    }

    with pytest.raises(Invalid) as at_reference:
        svc_troupe.check_troupe(svc, {"layout": bad_layout})
    assert at_reference.value.field == "layout"

    rigged = _mesh(svc, rigged=True)
    with pytest.raises(Invalid) as at_direct_door:
        svc_troupe.create_charsheet(svc, rigged, layout=bad_layout)
    assert at_direct_door.value.field == "layout"

    monkeypatch.setattr(
        "warlock.doctor.blender_check", lambda: SimpleNamespace(ok=True, detail="")
    )
    unrigged = _mesh(svc, rigged=False)
    with pytest.raises(Invalid) as at_send_door:
        svc_troupe.send_to_troupe(svc, unrigged, layout=bad_layout)
    assert at_send_door.value.field == "template"


def test_a_256px_sheet_over_the_atlas_ceiling_is_refused_on_the_layout_field(svc):
    """256px is a real ``charsheet.SIZES`` rung, and the default five
    movements fill it exactly (256 cells, 8192px). A sixth movement pushes the
    atlas over the ceiling, and the existing ``check_atlas_size`` door -- run
    inside ``charsheet.plan`` -- is what catches it."""
    rigged = _mesh(svc, rigged=True)
    layout = {
        "version": 3,
        "movements": [
            {"key": name, "direction_preset": 8}
            for name in ("idle", "walk", "run", "attack", "jump", "hit")
        ],
    }
    with pytest.raises(Invalid) as excinfo:
        svc_troupe.create_charsheet(svc, rigged, logical_size=256, layout=layout)
    assert excinfo.value.field == "layout"
    # Pinned on the message too, not just the field: without ``timing``,
    # ``hit`` alone would already be refused (as an unknown legacy name) with
    # the same field and a different sentence -- the atlas ceiling has to be
    # the actual reason, not an accident of two refusals sharing an address.
    assert "8192" in str(excinfo.value)


def test_a_twelve_fps_sheet_records_its_fps_in_the_row(svc):
    """A layout-wide ``fps`` is a v3 property Troupe's own request may carry;
    ``check_troupe`` snapshots the resolved layout, ``fps`` included, onto the
    row it returns. Named on ``hit`` rather than a legacy movement -- proving
    the timed resolution and the ``fps`` override compose, since a legacy name
    would resolve the same way even without ``timing`` passed at all."""
    checked = svc_troupe.check_troupe(
        svc,
        {
            "layout": {
                "version": 3,
                "fps": 12,
                "movements": [{"key": "hit", "direction_preset": 1}],
            }
        },
    )
    assert checked["layout"]["fps"] == 12
    assert checked["layout"]["version"] == 3
    assert checked["layout"]["movements"][0]["key"] == "hit"


def test_an_invalid_fps_is_refused_on_the_fps_field(svc):
    with pytest.raises(Invalid) as excinfo:
        svc_troupe.check_troupe(
            svc,
            {
                "layout": {
                    "version": 3,
                    "fps": 7,
                    "movements": [{"key": "idle", "direction_preset": 1}],
                }
            },
        )
    assert excinfo.value.field == "fps"
