"""A pure, deterministic writer for the Godot 4 text scene beside an exported character.

The export action this feeds writes ``<Stem>/<Stem>.glb`` (the served,
never-renamed animated model) and ``<Stem>/<Stem>.tscn`` beside it. The scene
is generated rather than hand-authored because it has to reference a filename
this app just chose and wire up one state per clip the export happened to
carry -- both of which change per export, so the alternative is a template a
human edits by hand for every character and silently drifts from the glb.

No Godot binary is available to this repo (CI is Windows-only and headless),
so correctness here is *structural*: the parser in ``tests/_tscn.py`` proves
every resource reference resolves and every rule in this docstring's caller
is honoured, and a human opens the result in a real Godot 4 editor later to
answer what structure cannot. Because
of that, every choice this module makes that Godot itself could plausibly
answer differently across versions is pulled into one named constant with a
comment -- so the sitting's answer is a one-line fix, not an archaeology dig.

**The loop heuristic.** Verified against Godot master (2026-09),
``editor/import/3d/resource_importer_scene.cpp``, ``_pre_fix_node``: for
every animation, and for each of ``"loop_mode"``, ``"loop"``, ``"cycle"`` in
that order (``_LOOP_NAME_MARKERS`` below), the importer strips trailing
digits/whitespace/underscores off the name and then tests whether what is
left *ends* with ``"-marker"`` or ``"_marker"`` (or contains ``"$marker"``
anywhere) -- **never** a prefix test. A match sets the animation to loop and
*renames* it, removing the matched suffix. ``godot_clip_name`` produces a
name that lands on the correct side of that heuristic: a loop always gets
``-loop`` appended (which alone guarantees an ``"-loop"`` suffix match
regardless of what the name already contained); a one-shot is returned
unchanged unless it already matches one of the three suffix tests, in which
case ``-once`` is appended (verified afterwards to no longer match, since
appending can only ever fix the *end* of the string -- a prefix was never a
problem to begin with, because Godot never tests one). This also means a
one-shot whose name merely *starts* with a marker (``loop_kick``,
``cycle_attack``) or contains one without a ``-``/``_``/``$`` boundary
(``recycle``) is left completely alone -- Godot's own heuristic would not
have touched it either.

**The AnimationTree wiring.** One ``AnimationNodeStateMachine`` is the
``tree_root``. ``idle``/``walk``/``run`` (whichever are present -- ``idle``
is mandatory, see ``scene_text``) are collapsed into a single ``locomotion``
state backed by an ``AnimationNodeBlendSpace1D``, because a state machine
with three mutually-exclusive movement states and no blend would pop between
them instead of blending. Every other clip is its own state, named by its
own base name, so game code can call ``travel("attack_02")`` directly.
"""

from __future__ import annotations

import re

__all__ = ["godot_clip_name", "animation_reference", "scene_text"]

# --- Godot-version-sensitive constants, each named and commented in place. ---

#: Godot 4's glTF importer loop heuristic, in the exact order
#: ``_pre_fix_node`` (``editor/import/3d/resource_importer_scene.cpp``,
#: Godot master as of 2026-09) applies them -- each is tested, and if it
#: matches, fixed (renamed) and looped, *before* the next is tested against
#: the possibly-already-renamed name. If a future Godot version changes the
#: heuristic (order, membership, or algorithm), this tuple -- and
#: ``_godot_loop_test``/``_godot_loop_fix`` below -- are the only edits
#: needed.
_LOOP_NAME_MARKERS = ("loop_mode", "loop", "cycle")

#: ``AnimationNodeStateMachineTransition.SwitchMode`` / ``AdvanceMode``
#: integers, hardcoded rather than imported because nothing in this
#: stdlib-only module can import the Godot engine. Taken from the Godot 4.x
#: class reference; a renumbering in a later major version is the one thing
#: that would break every transition this module writes, which is exactly
#: why the four values live in one place.
_SWITCH_MODE_IMMEDIATE = 0
_SWITCH_MODE_AT_END = 2
_ADVANCE_MODE_ENABLED = 1
_ADVANCE_MODE_AUTO = 2

#: The node name Godot 4's glTF importer gives the ``AnimationPlayer`` it
#: creates inside an imported scene. Standard for Godot 4's importer at the
#: time of writing; if a future importer names it differently the
#: ``anim_player`` NodePath below is the only thing that needs to change.
_ANIM_PLAYER_NODE_NAME = "AnimationPlayer"

#: A solo-``idle`` locomotion blend space has exactly one point, at 0.0, and
#: a ``AnimationNodeBlendSpace1D`` still wants ``min_space < max_space`` to
#: describe a non-degenerate axis. This is this module's own guess at a safe
#: upper bound for that case (never sampled away from 0.0 since there is
#: nothing else on the axis) -- unverified against the real editor, hence a
#: named constant rather than a literal buried in ``_blend_space_range``.
_SOLO_LOCOMOTION_MAX_SPACE = 1.0

#: Whether to also emit ``states/Start/position`` on the state machine.
#: Godot auto-creates the implicit ``Start``/``End`` pseudo-states on load
#: when they are absent, so omitting this is expected to be harmless; kept
#: as one flag rather than scattered so the sitting can flip it in one place
#: if the editor turns out to want it written down.
_EMIT_START_POSITION = False

#: Grid step for the (cosmetic-only) editor-graph positions, so the layout
#: is deterministic without meaning anything about behaviour.
_POSITION_STEP_X = 220.0
_POSITION_ROW_Y = 200.0

# --- Fixed vocabulary the state machine is built from. -----------------------

#: The three clips that fold into one blended "locomotion" state, and the
#: fixed axis position each holds on the blend space -- both spelled out by
#: the brief, not derived.
_LOCOMOTION_POS: dict[str, float] = {"idle": 0.0, "walk": 1.0, "run": 2.0}
_LOCOMOTION_NAMES = tuple(_LOCOMOTION_POS)

_HIT = "hit"
_DEATH = "death"
_JUMP = "jump"
_FALL = "fall"

#: The order every deterministic listing (sub-resource ids, state positions)
#: uses for the clips this brief calls out by name; anything else present is
#: appended afterwards, sorted, so a custom clip never perturbs this order.
_CANONICAL_ORDER = (
    "idle",
    "walk",
    "run",
    "attack",
    "attack_02",
    "cast",
    "jump",
    "fall",
    "hit",
    "death",
)

#: Characters that break a ``&"..."`` StringName literal if left unescaped.
_UNSAFE_NAME_CHARS = re.compile(r'["\\\r\n]')

#: Godot 4 node names may not contain any of these (Godot 4 class reference,
#: ``Node.name``).
_INVALID_NODE_NAME_CHARS = frozenset('.:@/"%')


def _validate_clip_name(name: str) -> None:
    if not name:
        raise ValueError("clip name must not be empty")
    if _UNSAFE_NAME_CHARS.search(name):
        raise ValueError(f"clip name {name!r} contains a char that breaks a StringName literal")
    # The 2026-09-14 audit (troupe-01) found that a clip name containing "/"
    # -- refused nowhere upstream (service.clips._check_shape and
    # cliplib.parse_clip_library only refuse direction-suffix collisions,
    # duplicates and empty names) -- is embedded verbatim as an unquoted
    # Godot property-path segment in ``states/{name}/node`` and
    # ``states/{name}/position``. A "/" there splits the key into more path
    # segments than the writer intended, nesting the state under the wrong
    # sub-property while ``transitions`` still names it as one atomic string,
    # silently corrupting the .tscn. Reusing ``_INVALID_NODE_NAME_CHARS``
    # (Godot 4's own banned ``Node.name`` characters, which include "/")
    # refuses every character that can break a property-path segment, not
    # just the one this finding named.
    bad = sorted(_INVALID_NODE_NAME_CHARS.intersection(name))
    if bad:
        raise ValueError(
            f"clip name {name!r} contains {''.join(bad)!r}, which would break a "
            "Godot property-path segment (states/{name}/node)"
        )


def _strip_trailing_junk(text: str) -> str:
    """Mirrors the trailing-strip loop at the top of both ``_teststr`` and
    ``_fixstr`` (``editor/import/3d/resource_importer_scene.cpp``, Godot
    master as of 2026-09): trailing digits, whitespace (``<= 32``) and
    underscores are peeled off before either function looks at the name.
    """
    i = len(text)
    while i > 0 and (text[i - 1].isdigit() or ord(text[i - 1]) <= 32 or text[i - 1] == "_"):
        i -= 1
    return text[:i]


def _godot_loop_test(name: str, marker: str) -> bool:
    """Restates Godot's ``_teststr`` (same file/function as above) for one
    marker: true if the trailing-junk-stripped, lowercased name contains
    ``"$" + marker`` anywhere, or ends with ``"-" + marker`` or
    ``"_" + marker``. Never a prefix test.
    """
    stripped = _strip_trailing_junk(name)
    lowered = stripped.lower()
    return (
        ("$" + marker) in lowered
        or lowered.endswith("-" + marker)
        or lowered.endswith("_" + marker)
    )


def _godot_loop_fix(name: str, marker: str) -> str:
    """Restates Godot's ``_fixstr`` (same file/function as above) for one
    marker, to be called only once ``_godot_loop_test`` has confirmed a
    match. Removes the matched ``$marker``/``-marker``/``_marker`` and
    reattaches whatever trailing junk ``_strip_trailing_junk`` peeled off.

    Mirrors one asymmetry in the source faithfully rather than fixing it:
    the ``$marker`` branch is *detected* case-insensitively but *removed*
    with an exact-case ``replace`` -- unreachable for Realmspinner, whose clip
    names cannot contain ``$`` (see ``godot_clip_name``'s defensive check).
    """
    stripped = _strip_trailing_junk(name)
    end = name[len(stripped) :]
    lowered = stripped.lower()
    dollar = "$" + marker
    if dollar in lowered:
        return stripped.replace(dollar, "") + end
    if lowered.endswith("-" + marker) or lowered.endswith("_" + marker):
        return stripped[: len(stripped) - (len(marker) + 1)] + end
    return stripped + end


def godot_clip_name(name: str, loop: bool) -> str:
    """The animation name to give this clip so Godot 4's glTF importer sets
    its loop mode correctly (see ``_LOOP_NAME_MARKERS`` above).

    A looping clip always gets ``-loop`` appended, which is sufficient on
    its own: Godot's heuristic is suffix-only, and appending forces the
    result to end with ``"-loop"`` regardless of what it started as. A
    one-shot is returned unchanged unless it already matches one of
    Godot's own three suffix tests (``_godot_loop_test``), in which case
    ``-once`` is appended -- verified, before returning, to no longer match
    any marker. Godot never tests a *prefix*, so a one-shot name like
    ``loop_kick`` or ``recycle`` needs no fix at all and is returned as-is.
    """
    _validate_clip_name(name)
    if loop:
        return f"{name}-loop"
    if not any(_godot_loop_test(name, marker) for marker in _LOOP_NAME_MARKERS):
        return name
    result = f"{name}-once"
    if any(_godot_loop_test(result, marker) for marker in _LOOP_NAME_MARKERS):
        # Defensive: only reachable if `name` already contained a `$marker`
        # that survives the `-once` suffix (e.g. "fx$loop"), which Realmspinner
        # cannot emit -- its clip names are restricted to [a-z0-9_] and
        # never contain "$". Refusing beats silently exporting a one-shot
        # that Godot would still import as a loop.
        raise ValueError(
            f"clip name {name!r} still matches Godot's loop heuristic after "
            f"appending -once ({result!r})"
        )
    return result


def animation_reference(name: str, loop: bool) -> str:
    """The string Godot's ``AnimationPlayer`` holds -- and so the string the
    ``AnimationNodeAnimation`` this module writes must play -- once the
    importer has run ``_pre_fix_node`` on ``godot_clip_name(name, loop)``.

    This simulates that pass with ``_godot_loop_test``/``_godot_loop_fix``:
    each marker in ``_LOOP_NAME_MARKERS`` order is tested against the
    current name, and on a match the name is rewritten before the next
    marker is tested (mirroring ``animname`` being updated in-loop in the
    source). A looping ``idle`` therefore round-trips to plain ``idle``,
    and a looping ``spin_loop`` (-> ``spin_loop-loop``) also round-trips to
    ``spin_loop``. A one-shot's suffix, once appended by ``godot_clip_name``,
    is built to never match again, so one-shots are always fixed points here.

    This one-name simulation does not model Godot's ``AnimationLibrary``
    renaming a key (``E`` in the source) a second time after a first
    marker has already renamed it out from under that key -- Realmspinner never
    emits a name that matches two markers in sequence, so that case is left
    unspecified here. A human sitting in a real Godot editor still confirms
    all of this against the Godot version a user actually runs.
    """
    animname = godot_clip_name(name, loop)
    for marker in _LOOP_NAME_MARKERS:
        if _godot_loop_test(animname, marker):
            animname = _godot_loop_fix(animname, marker)
    return animname


def _sanitise_node_name(name: str) -> str:
    if not name:
        raise ValueError("root_name must not be empty")
    cleaned = "".join("_" if ch in _INVALID_NODE_NAME_CHARS else ch for ch in name)
    if not cleaned:
        raise ValueError(f"root_name {name!r} sanitises to an empty node name")
    return cleaned


def _validate_glb_file(glb_file: str) -> None:
    if not glb_file:
        raise ValueError("glb_file must not be empty")
    if "/" in glb_file or "\\" in glb_file:
        raise ValueError(f"glb_file {glb_file!r} must be a bare filename, not a path")
    if ".." in glb_file:
        raise ValueError(f"glb_file {glb_file!r} must not contain '..'")
    if not glb_file.lower().endswith((".glb", ".gltf")):
        raise ValueError(f"glb_file {glb_file!r} must end with .glb or .gltf")


def _full_order(names: set[str]) -> list[str]:
    ordered = [n for n in _CANONICAL_ORDER if n in names]
    extra = sorted(n for n in names if n not in _CANONICAL_ORDER)
    return ordered + extra


def _blend_space_range(present_positions: list[float]) -> tuple[float, float]:
    max_space = max(present_positions)
    if max_space <= 0.0:
        max_space = _SOLO_LOCOMOTION_MAX_SPACE
    return 0.0, max_space


def _quote(text: str) -> str:
    # Only used for values this module already validated against embedding
    # a quote/backslash/newline (glb_file and clip names); root_name is
    # sanitised of everything that could be a path/special char but a raw
    # quote is not among that set, so it is escaped here too, defensively.
    return text.replace("\\", "\\\\").replace('"', '\\"')


def scene_text(
    *,
    root_name: str,
    glb_file: str,
    clips: list[tuple[str, bool]] | tuple[tuple[str, bool], ...],
) -> str:
    """Render a complete Godot 4 ``.tscn`` text scene for one animated character.

    ``clips`` is an unordered ``(base clip name, loop)`` sequence; the output
    is byte-identical regardless of the order clips are given in, and
    duplicate base names are refused. At least ``idle`` is required -- there
    is no meaningful "locomotion" state, and nowhere for a one-shot to
    return to, without it.
    """
    root_name = _sanitise_node_name(root_name)
    _validate_glb_file(glb_file)

    seen: set[str] = set()
    loop_of: dict[str, bool] = {}
    for clip_name, loop in clips:
        _validate_clip_name(clip_name)
        if clip_name in seen:
            raise ValueError(f"duplicate clip name {clip_name!r}")
        seen.add(clip_name)
        loop_of[clip_name] = bool(loop)

    if "idle" not in loop_of:
        raise ValueError("a character scene needs at least idle")

    order = _full_order(seen)
    locomotion_members = [n for n in _LOCOMOTION_NAMES if n in loop_of]
    non_locomotion = [n for n in order if n not in _LOCOMOTION_NAMES]

    # --- AnimationNodeAnimation sub-resources, one per clip, canonical order.
    anim_ids = {name: f"AnimationNodeAnimation_{name}" for name in order}
    anim_blocks = [
        f'[sub_resource type="AnimationNodeAnimation" id="{anim_ids[name]}"]\n'
        f'animation = &"{animation_reference(name, loop_of[name])}"\n'
        for name in order
    ]

    # --- The locomotion blend space, always present (idle is mandatory).
    blend_id = "AnimationNodeBlendSpace1D_locomotion"
    blend_lines = [f'[sub_resource type="AnimationNodeBlendSpace1D" id="{blend_id}"]']
    positions = []
    for i, name in enumerate(locomotion_members):
        pos = _LOCOMOTION_POS[name]
        positions.append(pos)
        blend_lines.append(f'blend_point_{i}/node = SubResource("{anim_ids[name]}")')
        blend_lines.append(f"blend_point_{i}/pos = {pos}")
    min_space, max_space = _blend_space_range(positions)
    blend_lines.append(f"min_space = {min_space}")
    blend_lines.append(f"max_space = {max_space}")
    blend_block = "\n".join(blend_lines) + "\n"

    # --- Transitions, built in one fixed, deterministic pass. -------------
    transitions: dict[tuple[str, str], tuple[int, int]] = {}

    def add(frm: str, to: str, switch: int, advance: int) -> None:
        transitions.setdefault((frm, to), (switch, advance))

    add("Start", "locomotion", _SWITCH_MODE_IMMEDIATE, _ADVANCE_MODE_AUTO)

    routable = [n for n in non_locomotion if n not in (_HIT, _DEATH)]
    for name in routable:
        add("locomotion", name, _SWITCH_MODE_IMMEDIATE, _ADVANCE_MODE_ENABLED)

    for name in routable:
        if not loop_of[name]:
            if name == _JUMP and _FALL in loop_of:
                add(_JUMP, _FALL, _SWITCH_MODE_AT_END, _ADVANCE_MODE_AUTO)
            else:
                add(name, "locomotion", _SWITCH_MODE_AT_END, _ADVANCE_MODE_AUTO)
        else:
            add(name, "locomotion", _SWITCH_MODE_IMMEDIATE, _ADVANCE_MODE_ENABLED)

    if _HIT in loop_of:
        for state in ["locomotion"] + [n for n in non_locomotion if n not in (_HIT, _DEATH)]:
            add(state, _HIT, _SWITCH_MODE_IMMEDIATE, _ADVANCE_MODE_ENABLED)
        add(_HIT, "locomotion", _SWITCH_MODE_AT_END, _ADVANCE_MODE_AUTO)

    if _DEATH in loop_of:
        for state in ["locomotion"] + [n for n in non_locomotion if n != _DEATH]:
            add(state, _DEATH, _SWITCH_MODE_IMMEDIATE, _ADVANCE_MODE_ENABLED)

    transition_ids = {pair: f"Transition_{i}" for i, pair in enumerate(transitions, start=1)}
    transition_blocks = []
    for pair, (switch, advance) in transitions.items():
        tid = transition_ids[pair]
        transition_blocks.append(
            f'[sub_resource type="AnimationNodeStateMachineTransition" id="{tid}"]\n'
            f"switch_mode = {switch}\n"
            f"advance_mode = {advance}\n"
        )

    # --- The state machine itself. -----------------------------------------
    machine_id = "AnimationNodeStateMachine_root"
    states = ["locomotion"] + non_locomotion
    machine_lines = [f'[sub_resource type="AnimationNodeStateMachine" id="{machine_id}"]']
    if _EMIT_START_POSITION:
        machine_lines.append(f"states/Start/position = Vector2(0, {_POSITION_ROW_Y})")
    for i, name in enumerate(states):
        x = _POSITION_STEP_X * (i + 1)
        node_id = blend_id if name == "locomotion" else anim_ids[name]
        machine_lines.append(f'states/{name}/node = SubResource("{node_id}")')
        machine_lines.append(f"states/{name}/position = Vector2({x}, {_POSITION_ROW_Y})")
    array_items = []
    for pair in transitions:
        frm, to = pair
        tid = transition_ids[pair]
        array_items.append(f'"{frm}", "{to}", SubResource("{tid}")')
    machine_lines.append("transitions = [" + ", ".join(array_items) + "]")
    machine_block = "\n".join(machine_lines) + "\n"

    sub_resource_blocks = anim_blocks + [blend_block] + transition_blocks + [machine_block]
    load_steps = 1 + len(sub_resource_blocks) + 1  # ext_resource + sub_resources + 1

    parts = [f'[gd_scene load_steps={load_steps} format=3]\n']
    # A relative (non-``res://``) path is resolved by Godot's text-scene
    # loader against the directory the .tscn itself lives in, which is why
    # this can be a bare filename: the export action writes the glb and the
    # tscn into the same directory together.
    parts.append(f'[ext_resource type="PackedScene" path="{_quote(glb_file)}" id="1_model"]\n')
    parts.extend(sub_resource_blocks)
    parts.append(f'[node name="{_quote(root_name)}" type="Node3D"]\n')
    parts.append('[node name="Model" parent="." instance=ExtResource("1_model")]\n')
    parts.append(
        '[node name="AnimationTree" type="AnimationTree" parent="."]\n'
        f'tree_root = SubResource("{machine_id}")\n'
        f'anim_player = NodePath("../Model/{_ANIM_PLAYER_NODE_NAME}")\n'
        "active = true\n"
        "parameters/locomotion/blend_position = 0.0\n"
    )
    # Every block above already ends with its own single "\n", so joining
    # with a blank-line separator and returning as-is gives exactly one
    # trailing newline -- never a blank line at end of file.
    return "\n".join(parts)
