"""poser-jobs-02 (2026-09-26 audit): a clip named after one of the state
machine's own state names corrupted the .tscn.

A clip literally named ``locomotion`` was not filtered out of
``non_locomotion`` (only ``idle``/``walk``/``run`` are, via
``_LOCOMOTION_NAMES``), so it landed in ``states`` a second time and wrote a
second ``states/locomotion/node`` line for the one blend-space state this
module always builds -- one .tscn, two lines claiming the same key. A clip
named ``Start`` or ``End`` collides the other way: those are Godot's own
implicit entry/exit pseudo-states on every ``AnimationNodeStateMachine``, so a
clip claiming either name writes a state indistinguishable from one Godot
itself already assumes exists.
"""

from __future__ import annotations

import pytest

from realmspinner.godotscene import scene_text


def test_scene_text_refuses_a_clip_named_after_a_state_machine_state():
    for reserved in ("locomotion", "Start", "End"):
        with pytest.raises(ValueError, match="reserved"):
            scene_text(
                root_name="Hero",
                glb_file="hero.glb",
                clips=[("idle", True), (reserved, False)],
            )
