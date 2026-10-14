"""Regression tests for what is left of the troupe fix-brief of the 2026-09-15 audit.

The findings whose fix was a character-sheet door (troupe-03's sheet preview,
service-07's HD-mode options) left with the sheets (``retired.py``); what stays
is troupe-05 (a sprite draft is not deletable mid-publish) and troupe-08 (dead
kin under an already-populated winged archetype).
"""

from __future__ import annotations

import pytest
from PIL import Image

from realmspinner.kernels.rig import store
from realmspinner.service import sprites as svc_sprites
from realmspinner.service.errors import Conflict

# --- troupe-05: a sprite draft must not be deletable mid-publish ------------


def _reference(svc):
    job_id = svc.store.create("text", "a knight", {"seed": 1}, stage="reference")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (200, 200, 200)).save(job_dir / "input.png")
    svc.store.set_status(job_id, "done")
    return job_id


def _draft_mid_publish(svc, job_id):
    """The worker's own publish order, stopped halfway: both PNGs written,
    the sidecar (``list_sprite_drafts``' completion marker) not yet."""
    draft_id = store.new_id()
    job_dir = svc.job_dir(job_id)
    for letter in store.SPRITE_CANDIDATES:
        path = store.sprite_draft_png_path(job_dir, draft_id, letter)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(path)
    return draft_id


def test_deleting_a_sprite_draft_while_its_synthesis_is_still_publishing_leaves_no_orphan_files(
    svc,
):
    """The 2026-09-15 audit, finding troupe-05: ``delete_sprite_draft`` took
    no lock and made no in-flight check, unlike its sheet-door siblings.
    Expressed as state, the way the sheet door's own regression was: a queued/running
    ``sprite_synthesis`` row targeting this draft is exactly the state a
    delete arriving mid-publish would see, and the delete must refuse rather
    than unlink whatever the worker has written so far.
    """
    job_id = _reference(svc)
    draft_id = _draft_mid_publish(svc, job_id)
    svc.store.create(
        "sprite_synthesis",
        "a knight",
        {"source_job": job_id, "draft_id": draft_id},
        status="running",
    )

    with pytest.raises(Conflict):
        svc_sprites.delete_sprite_draft(svc, job_id, draft_id)

    job_dir = svc.job_dir(job_id)
    assert all(
        store.sprite_draft_png_path(job_dir, draft_id, c).exists()
        for c in store.SPRITE_CANDIDATES
    )


# --- troupe-08: dead kin under an already-populated winged archetype -------


def test_imp_gargoyle_angel_and_harpy_never_offer_their_declared_kin():
    """The 2026-09-15 audit, finding troupe-08: each of these four is a
    "winged" archetype creature, and the registry already ships winged
    species (dragon, wyvern, bat, bird, raven, griffin) -- so
    ``resolve._plan_for`` never falls through to humanoid for them, and
    ``resolve._offer_for``'s pool is winged-only. Each one used to declare a
    humanoid ``kin`` (goblin, ogre, human) that ``add(creature.kin)`` could
    never add to that pool: dead data no offer sentence could ever surface.
    Checked directly against the source of truth -- no ``kin`` left to be
    unreachable -- and, for the general shape of the mistake, that whatever
    ``kin`` a creature declares must actually be inside the pool
    ``_offer_for`` computes for it.
    """
    from realmspinner.characters import family as family_mod
    from realmspinner.characters.resolve import KNOWN_CREATURES, _offer_for

    registry = family_mod.families()
    for name in ("imp", "gargoyle", "angel", "harpy"):
        creature = KNOWN_CREATURES[name]
        assert creature.kin == (), f"{name} still declares dead kin {creature.kin!r}"
        offer = _offer_for(creature, registry)
        assert not (set(creature.kin) & set(offer))


# --- service-07: Troupe HD mode must drop colors/palette/dither/outline -----
