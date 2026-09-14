"""``godotscene.py`` writes the .tscn a human sitting later opens in the real
Godot editor (no Godot binary exists in this repo or CI), so what these
tests can prove is *structural*: every resource reference resolves, every
rule the module's docstring claims is honoured, and the same clip set always
produces the same bytes. They cannot prove the file imports cleanly in
Godot 4 -- that is the open question TODO.md P50 hands to a human.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest
from _tscn import by_id, parse, parse_stringname, parse_subresource, parse_transitions, sections

from warlock import godotscene
from warlock.godotscene import animation_reference, godot_clip_name, scene_text

# The set used by most tests: one of everything the brief names.
FULL_CLIPS = [
    ("idle", True),
    ("walk", True),
    ("run", True),
    ("attack", False),
    ("attack_02", False),
    ("cast", False),
    ("jump", False),
    ("fall", True),
    ("hit", False),
    ("death", False),
]


def _scene(clips=FULL_CLIPS, root_name="Hero", glb_file="hero.glb"):
    text = scene_text(root_name=root_name, glb_file=glb_file, clips=clips)
    return text, parse(text)


def _machine_section(doc):
    for sec in sections(doc, "sub_resource"):
        if sec.header.get("type") == "AnimationNodeStateMachine":
            return sec
    raise AssertionError("no AnimationNodeStateMachine sub_resource in scene")


def _state_names(doc) -> set[str]:
    machine = _machine_section(doc)
    return {
        key.split("/")[1]
        for key in machine.body
        if key.startswith("states/") and key.endswith("/node")
    }


def _transition_pairs(doc) -> set[tuple[str, str]]:
    machine = _machine_section(doc)
    return {(frm, to) for frm, to, _sub_id in parse_transitions(machine.body["transitions"])}


def _transition_modes(doc) -> dict[tuple[str, str], tuple[int, int]]:
    machine = _machine_section(doc)
    out = {}
    for frm, to, sub_id in parse_transitions(machine.body["transitions"]):
        sec = by_id(doc, "sub_resource", sub_id)
        out[(frm, to)] = (int(sec.body["switch_mode"]), int(sec.body["advance_mode"]))
    return out


# --- shape of the document ---------------------------------------------------


def test_the_scene_is_a_godot_4_text_scene():
    text, doc = _scene()
    assert doc[0].kind == "gd_scene"
    assert doc[0].header["format"] == 3
    resource_count = len(sections(doc, "ext_resource")) + len(sections(doc, "sub_resource"))
    assert doc[0].header["load_steps"] == resource_count + 1
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_every_ext_and_sub_resource_reference_resolves():
    import re

    _text, doc = _scene()
    known_ext = {s.header["id"] for s in sections(doc, "ext_resource")}
    known_sub = {s.header["id"] for s in sections(doc, "sub_resource")}
    ref_re = re.compile(r'(ExtResource|SubResource)\("([^"]*)"\)')
    seen_any = False
    for sec in doc:
        blobs = list(sec.body.values())
        for value in sec.header.values():
            if isinstance(value, tuple):
                blobs.append(f'{value[0]}("{value[1]}")')
        for blob in blobs:
            for kind, ref_id in ref_re.findall(blob):
                seen_any = True
                if kind == "ExtResource":
                    assert ref_id in known_ext, f"unresolved ExtResource {ref_id!r}"
                else:
                    assert ref_id in known_sub, f"unresolved SubResource {ref_id!r}"
    assert seen_any, "the scene never references a resource, so this test proves nothing"


def test_the_glb_is_instanced_by_a_path_relative_to_the_scene():
    _text, doc = _scene(glb_file="hero.glb")
    ext = sections(doc, "ext_resource")
    assert len(ext) == 1
    assert ext[0].header["type"] == "PackedScene"
    assert ext[0].header["path"] == "hero.glb"
    model_nodes = [s for s in sections(doc, "node") if s.header.get("name") == "Model"]
    assert len(model_nodes) == 1
    assert model_nodes[0].header["instance"] == ("ExtResource", ext[0].header["id"])


# --- clip naming and the loop heuristic --------------------------------------


def test_every_animation_the_tree_plays_is_named_in_the_clip_list():
    _text, doc = _scene()
    anim_sections = [
        s for s in sections(doc, "sub_resource") if s.header.get("type") == "AnimationNodeAnimation"
    ]
    played = {parse_stringname(s.body["animation"]) for s in anim_sections}
    expected = {animation_reference(name, loop) for name, loop in FULL_CLIPS}
    assert played == expected
    assert len(played) == len(FULL_CLIPS)


def test_looping_clips_carry_the_godot_loop_suffix_and_one_shots_do_not():
    for name, loop in FULL_CLIPS:
        ref = godot_clip_name(name, loop)
        if loop:
            assert ref == f"{name}-loop"
        else:
            assert ref == name


def _teststr(name: str, marker: str) -> bool:
    """An independent reimplementation of Godot's ``_teststr``
    (``editor/import/3d/resource_importer_scene.cpp``, ``_pre_fix_node``,
    Godot master as of 2026-09), written straight from the verified source
    quoted in the task that produced this fix -- deliberately *not* sharing
    code with ``godotscene._godot_loop_test`` -- so these tests are a real
    cross-check on the module rather than the module checking itself.
    """
    what = name
    while what and (what[-1].isdigit() or ord(what[-1]) <= 32 or what[-1] == "_"):
        what = what[:-1]
    lowered = what.lower()
    return (
        ("$" + marker) in lowered
        or lowered.endswith("-" + marker)
        or lowered.endswith("_" + marker)
    )


def _fixstr(name: str, marker: str) -> str:
    """Independent reimplementation of Godot's ``_fixstr``; see ``_teststr``."""
    what = name
    while what and (what[-1].isdigit() or ord(what[-1]) <= 32 or what[-1] == "_"):
        what = what[:-1]
    end = name[len(what) :]
    lowered = what.lower()
    dollar = "$" + marker
    if dollar in lowered:
        return what.replace(dollar, "") + end
    if lowered.endswith("-" + marker) or lowered.endswith("_" + marker):
        return what[: len(what) - (len(marker) + 1)] + end
    return what + end


#: The order Godot's ``_pre_fix_node`` applies the markers in.
_MARKERS_IN_GODOT_ORDER = ("loop_mode", "loop", "cycle")


def _simulate_godot_import(emitted_name: str) -> tuple[str, bool]:
    """Simulates Godot's ``_pre_fix_node`` loop over one animation name using
    the independent ``_teststr``/``_fixstr`` above: returns the name the
    ``AnimationPlayer`` ends up holding, and whether any marker ever matched
    (i.e. whether Godot set the clip to loop).
    """
    animname = emitted_name
    looped = False
    for marker in _MARKERS_IN_GODOT_ORDER:
        if _teststr(animname, marker):
            animname = _fixstr(animname, marker)
            looped = True
    return animname, looped


#: Clips beyond FULL_CLIPS that exercise the adversarial suffix cases the
#: task calls out by name: a loop whose own name already ends in a marker
#: (``spin_loop``), and one-shots that do too (``fall_loop``, ``hit_cycle``,
#: a marker preceded by digits Godot strips first, ``kick_loop_2``).
_ADVERSARIAL_CASES = [
    ("spin_loop", True),
    ("fall_loop", False),
    ("hit_cycle", False),
    ("kick_loop_2", False),
]


def test_every_emitted_name_imports_as_its_reference_with_the_right_loop_mode():
    for name, loop in list(FULL_CLIPS) + _ADVERSARIAL_CASES:
        emitted = godot_clip_name(name, loop)
        final_name, looped = _simulate_godot_import(emitted)
        assert looped == loop, (name, loop, emitted, final_name, looped)
        assert final_name == animation_reference(name, loop), (
            name,
            loop,
            emitted,
            final_name,
        )


def test_godot_matches_suffixes_not_prefixes():
    # A name that only *starts* with a marker, or contains one with no
    # "-"/"_"/"$" boundary before it, is not a suffix match -- Godot's own
    # heuristic (_teststr) never tests a prefix -- so godot_clip_name must
    # leave these one-shots completely alone.
    for name in ("loop_kick", "cycle_attack", "recycle"):
        assert godot_clip_name(name, loop=False) == name
        final_name, looped = _simulate_godot_import(name)
        assert looped is False, name
        assert final_name == name, name


def test_the_tree_plays_the_names_godot_leaves_after_import():
    _text, doc = _scene()
    anim_sections = {
        s.header["id"]: parse_stringname(s.body["animation"])
        for s in sections(doc, "sub_resource")
        if s.header.get("type") == "AnimationNodeAnimation"
    }
    for name, loop in FULL_CLIPS:
        if not loop:
            continue
        played = anim_sections[f"AnimationNodeAnimation_{name}"]
        # idle/walk/run/fall carry no marker of their own, so Godot's
        # import-time rename strips the "-loop" this module added right
        # back off -- the AnimationNodeAnimation must already store that
        # post-import name, since nothing renames it a second time.
        assert played == name, (name, played)


# --- locomotion blend space ---------------------------------------------------


def test_idle_walk_and_run_share_one_blend_space_in_speed_order():
    _text, doc = _scene()
    blends = [
        s
        for s in sections(doc, "sub_resource")
        if s.header.get("type") == "AnimationNodeBlendSpace1D"
    ]
    assert len(blends) == 1
    blend = blends[0]
    for i, (name, pos) in enumerate((("idle", 0.0), ("walk", 1.0), ("run", 2.0))):
        node_id = parse_subresource(blend.body[f"blend_point_{i}/node"])
        anim = by_id(doc, "sub_resource", node_id)
        assert parse_stringname(anim.body["animation"]) == animation_reference(name, True)
        assert float(blend.body[f"blend_point_{i}/pos"]) == pos
    assert float(blend.body["min_space"]) == 0.0
    assert float(blend.body["max_space"]) == 2.0

    machine = _machine_section(doc)
    assert parse_subresource(machine.body["states/locomotion/node"]) == blend.header["id"]


# --- the hit/death net --------------------------------------------------------


def test_every_state_but_death_can_reach_hit_and_death():
    _text, doc = _scene()
    states = _state_names(doc)
    pairs = _transition_pairs(doc)
    for state in states - {"death"}:
        if state != "hit":
            assert (state, "hit") in pairs, f"{state} cannot reach hit"
        assert (state, "death") in pairs, f"{state} cannot reach death"


def test_death_is_terminal():
    _text, doc = _scene()
    pairs = _transition_pairs(doc)
    assert not any(frm == "death" for frm, _to in pairs)


def test_every_one_shot_returns_to_locomotion_at_its_end():
    _text, doc = _scene()
    modes = _transition_modes(doc)
    for name in ("attack", "attack_02", "cast", "hit"):
        assert modes[(name, "locomotion")] == (
            godotscene._SWITCH_MODE_AT_END,
            godotscene._ADVANCE_MODE_AUTO,
        )


def test_jump_hands_over_to_fall_when_the_skeleton_has_one():
    _text, doc = _scene()
    pairs = _transition_pairs(doc)
    modes = _transition_modes(doc)
    assert ("jump", "fall") in pairs
    assert ("jump", "locomotion") not in pairs
    assert modes[("jump", "fall")] == (
        godotscene._SWITCH_MODE_AT_END,
        godotscene._ADVANCE_MODE_AUTO,
    )
    assert modes[("fall", "locomotion")] == (
        godotscene._SWITCH_MODE_IMMEDIATE,
        godotscene._ADVANCE_MODE_ENABLED,
    )

    no_fall_clips = [c for c in FULL_CLIPS if c[0] != "fall"]
    _text2, doc2 = _scene(clips=no_fall_clips)
    pairs2 = _transition_pairs(doc2)
    modes2 = _transition_modes(doc2)
    assert ("jump", "locomotion") in pairs2
    assert modes2[("jump", "locomotion")] == (
        godotscene._SWITCH_MODE_AT_END,
        godotscene._ADVANCE_MODE_AUTO,
    )


# --- determinism and refusals -------------------------------------------------


def test_the_scene_text_is_byte_identical_for_the_same_clips_in_any_order():
    forward = scene_text(root_name="Hero", glb_file="hero.glb", clips=FULL_CLIPS)
    backward = scene_text(root_name="Hero", glb_file="hero.glb", clips=list(reversed(FULL_CLIPS)))
    shuffled = scene_text(
        root_name="Hero",
        glb_file="hero.glb",
        clips=[FULL_CLIPS[i] for i in (3, 0, 7, 1, 9, 2, 5, 4, 8, 6)],
    )
    assert forward == backward == shuffled


def test_a_path_in_the_glb_name_is_refused():
    """Each bad value trips a different one of ``_validate_glb_file``'s
    checks -- a bare ``pytest.raises(ValueError)`` would pass just as well if
    every case fell through to the same message, so each is matched against
    the real one it raises."""
    cases = {
        "dir/hero.glb": "must be a bare filename, not a path",
        "dir\\hero.glb": "must be a bare filename, not a path",
        # Also caught by the "/" check above, before the ".." check ever runs.
        "../hero.glb": "must be a bare filename, not a path",
        "hero..glb": "must not contain '..'",
        "hero.png": "must end with .glb or .gltf",
    }
    for bad, expected in cases.items():
        with pytest.raises(ValueError, match=re.escape(expected)):
            scene_text(root_name="Hero", glb_file=bad, clips=[("idle", True)])


def test_a_clip_name_containing_a_slash_is_refused_rather_than_corrupting_the_states_path():
    """The 2026-09-14 audit (troupe-01): nothing upstream of this module
    refuses a "/" in a clip name, and unquoted, it would split
    ``states/{name}/node`` into more property-path segments than the writer
    intended -- nesting the state under the wrong key while ``transitions``
    still names it as one atomic string. Refused here instead of silently
    writing a corrupt .tscn."""
    with pytest.raises(ValueError, match="property-path segment"):
        scene_text(
            root_name="Hero",
            glb_file="hero.glb",
            clips=[("idle", True), ("attack/special", False)],
        )
    with pytest.raises(ValueError, match="property-path segment"):
        godot_clip_name("attack/special", loop=False)


def test_a_scene_without_idle_is_refused():
    with pytest.raises(ValueError, match="a character scene needs at least idle"):
        scene_text(root_name="Hero", glb_file="hero.glb", clips=[("walk", True), ("run", True)])


def test_the_writer_imports_only_the_standard_library():
    source = Path(godotscene.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in sys.stdlib_module_names, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            root = (node.module or "").split(".")[0]
            assert root in sys.stdlib_module_names, node.module
