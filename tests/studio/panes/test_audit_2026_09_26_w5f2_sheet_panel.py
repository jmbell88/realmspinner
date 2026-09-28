"""Regression test for the 2026-09-26 audit, finding create-panes-07.

``sheet_panel``'s animated-clip ``From``/``To`` combos default to ``""``,
which imgui's combo widget cannot show as "nothing selected" -- it draws row
0 (the first pose) regardless, while the stored form value stays empty.
``validate`` then refuses with "A clip needs both ends." against a form that
visibly shows two ends already picked.
"""

from __future__ import annotations

from realmspinner.studio.panes import sheet_panel


def _poses():
    return [
        {"id": "pose-a", "name": "Idle"},
        {"id": "pose-b", "name": "Walk"},
    ]


def test_clip_ends_default_to_what_the_combos_display():
    form = {"clip_from": "", "clip_to": ""}
    poses = _poses()

    sheet_panel._seed_clip_ends(form, poses)

    assert form["clip_from"] == poses[0]["id"]
    assert form["clip_to"] == poses[0]["id"]


def test_a_seeded_clip_passes_validate_instead_of_the_needs_both_ends_refusal():
    job = {"status": "done", "files": ["model.glb", "rig.glb"]}
    form = {
        "clip": True,
        "clip_from": "",
        "clip_to": "",
        "poses": set(),
    }
    sheet_panel._seed_clip_ends(form, _poses())

    problems = sheet_panel.validate(job, form)

    assert "A clip needs both ends." not in problems, (
        "the combos show a pose picked at each end, so validate must not "
        "refuse with a message that contradicts what is on screen"
    )


def test_seeding_does_not_clobber_a_deliberate_choice():
    """Once either end holds a real id -- the user picked one, or a previous
    seed already ran -- a later call must not reset it back to the first
    pose."""
    form = {"clip_from": "pose-b", "clip_to": ""}
    sheet_panel._seed_clip_ends(form, _poses())

    assert form["clip_from"] == "pose-b"
    assert form["clip_to"] == "pose-a"
