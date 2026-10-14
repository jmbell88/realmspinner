"""Animations from another file, laid onto the bones a document already has.

The other half of :mod:`.rigimport`'s animation reader. That one opens a rigged file *as* a
document, making a bone of every joint; this one takes a file's motion and applies it to the rig
you built, matching **by bone name**. It is how a clip authored in Blender, Mixamo or another Clay
document comes onto a character made here.

**Names are matched loosely and deterministically** (:func:`canonical`): case, separators and
the ``mixamorig:`` prefix do not matter, and the side may be spelt ``.L``, ``_L``, ``Left`` or ``L``
and sit before or after the rest, so ``LeftArm``, ``arm.L`` and ``mixamorig:LeftArm`` are one bone.
Two document bones that canonicalise alike are ambiguous and neither is matched (said by name).
Whatever matches nothing is returned by name, never guessed.

**The motion is applied as a change from each rig's own rest pose**, not as absolute numbers:
rotation ``r = rest_bone * inverse(rest_source) * source_key``, translation ``t = rest_bone +
(source_key - rest_source)``, size ``s = rest_bone * source_key / rest_source``. For two rigs built
the same way the rests agree and this is the source's number unchanged; where the rests differ (a
longer arm, a bone turned a different way at rest) the motion still lands as the same *movement*.
It assumes the two rigs' bones share an orientation convention, which names alone cannot check;
a rig built differently needs the motion retargeted in Blender first, and the result says which
bones it matched so a wrong guess is visible.

Pure: no imgui, no GL, no service. The caller adds what comes back with
:meth:`~.document.ClayDoc.import_animations`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geom3d import gltf
from ..geom3d import math3d as m3
from . import anim
from .elements import OpError
from .rigimport import _channel, _fresh_name, _guess_fps, _linear_keys

__all__ = ["Imported", "canonical", "import_by_name", "match_bones"]

_PREFIX = re.compile(r"^(mixamorig\d*[:_.\-]?|armature[:_.\-]|bip\d*[:_.\-]?)", re.IGNORECASE)
# The 2026-10-14 audit, finding clay-rig-02: the words were ASCII-only, so a non-ASCII name
# (``左腕`` and ``右腕``) canonicalised to "" and matched every other such name, and ``Hüfte``
# lost its ``ü``. A letter outside ASCII is a word of its own here, so two names differing only
# there keep different keys.
_WORDS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+|(?![A-Za-z])[^\W\d_]+")
_SIDES = {"left": "l", "right": "r", "l": "l", "r": "r"}


def canonical(name: str) -> str:
    """*name* as a comparison key: lower-case words, the side last, separators gone.

    ``mixamorig:LeftUpLeg`` -> ``uplegl``; ``upleg.L`` -> ``uplegl``; ``Spine_01`` -> ``spine01``.
    A side is a word that is exactly ``left``, ``right``, ``l`` or ``r`` at either end. A name
    with no letter or digit at all (``__``) has the empty key, which :func:`match_bones` never
    pairs.
    """
    words = [w.lower() for w in _WORDS.findall(_PREFIX.sub("", name.strip()))]
    if len(words) > 1:
        if words[0] in _SIDES:
            words = [*words[1:], _SIDES[words[0]]]
        elif words[-1] in _SIDES:
            words[-1] = _SIDES[words[-1]]
    return "".join(words)


def match_bones(bones: dict[int, str], names: dict[int, str]) -> tuple[dict[int, int], list[str]]:
    """``({source key: bone uid}, ambiguous names)``: *names* (any hashable key to a source name)
    matched to *bones* (uid to name) by :func:`canonical`.

    A source name matching two bones, or two source names matching one, is not matched: a guess
    between two is how the left arm gets the right arm's wave. Their names come back in the second
    result. A name whose key is empty is not a key (the 2026-10-14 audit, finding clay-rig-02: every
    name made only of separators or symbols would otherwise match every other): it matches nothing
    and is not called ambiguous either, so it lands in the importer's unmatched list.
    """
    by_canon: dict[str, list[int]] = {}
    for uid, name in bones.items():
        key = canonical(name)
        if key:
            by_canon.setdefault(key, []).append(uid)
    claimed: dict[int, list[Any]] = {}
    pairs: dict[Any, int] = {}
    ambiguous: list[str] = []
    for key, name in names.items():
        canon = canonical(name)
        found = by_canon.get(canon, []) if canon else []
        if len(found) == 1:
            pairs[key] = found[0]
            claimed.setdefault(found[0], []).append(key)
        elif len(found) > 1:
            ambiguous.append(name)
    for keys in claimed.values():
        if len(keys) > 1:
            for key in keys:
                pairs.pop(key, None)
                ambiguous.append(names[key])
    return pairs, sorted(set(ambiguous))


@dataclass(frozen=True)
class Imported:
    """What :func:`import_by_name` found: the animations to add, the frame rate the file
    suggests, which source names landed on which bones, what landed on nothing and what to say."""

    animations: tuple[anim.Animation, ...]
    fps: int
    matched: dict[str, str]
    unmatched: tuple[str, ...]
    notices: tuple[str, ...]


def import_by_name(doc: Any, model: gltf.Model) -> Imported:
    """The animations of *model* retargeted onto *doc*'s bones by name. See the module docstring.

    Raises :class:`~.elements.OpError` when the file has no animation, the document has no bone,
    or no animated node matched any bone (the result would be an empty clip, which is not an
    import), and for any cap Clay's animation holds.
    """
    if not model.animations:
        raise OpError("This file has no animation in it.")
    bones = {obj.uid: obj.name for obj in doc.objects if obj.bone}
    if not bones:
        raise OpError("This document has no bones to animate. Mark the joints as bones first.")
    if len(model.animations) > anim.MAX_ANIMATIONS:
        raise OpError(
            f"This file has {len(model.animations)} animations, past the {anim.MAX_ANIMATIONS} a "
            "Clay rig holds. Delete the ones you do not need first."
        )
    animated = sorted({c.node for a in model.animations for c in a.channels})
    names = {n: model.nodes[n].name for n in animated if model.nodes[n].name}
    pairs, ambiguous = match_bones(bones, names)
    if not pairs:
        raise OpError(
            "None of the animated nodes in this file is named like one of this document's bones"
            + (f" (the bones are {', '.join(sorted(bones.values())[:6])}...)." if len(bones) > 6
               else f" (the bones are {', '.join(sorted(bones.values()))}).")
        )
    fps = _guess_fps(model)
    by_uid = {obj.uid: obj for obj in doc.objects}
    taken = {a.name for a in doc.rig_state.animations}
    animations: list[anim.Animation] = []
    cubic = 0
    for source in model.animations:
        label = f"The animation {source.name!r}" if source.name else "An animation"
        grouped: dict[int, dict[str, gltf.AnimChannel]] = {}
        interpolations: set[str] = set()
        for channel in source.channels:
            if channel.node in pairs:
                grouped.setdefault(channel.node, {})[channel.path] = channel
                interpolations.add("step" if channel.interpolation == "STEP" else "linear")
        channels: dict[int, dict[str, anim.Channel]] = {}
        for node_index, props in grouped.items():
            bone = by_uid[pairs[node_index]]
            made, resampled = _retarget(model.nodes[node_index], bone, props, fps, label)
            cubic += resampled
            channels[bone.uid] = made
        if not channels:
            continue
        loop = source.extras.get("loop", True) if isinstance(source.extras, dict) else True
        try:
            animations.append(
                anim.Animation(
                    _fresh_name(source.name, taken),
                    bool(loop),
                    "step" if interpolations == {"step"} else "linear",
                    channels,
                )
            )
        except ValueError as exc:
            raise OpError(f"{label} cannot be imported: {exc}") from exc
    if not animations:
        raise OpError("None of the animations in this file moves a bone this document has.")
    matched = {model.nodes[n].name: bones[uid] for n, uid in pairs.items()}
    unmatched = tuple(
        sorted(model.nodes[n].name for n in names if n not in pairs and names[n] not in ambiguous)
    )
    notices: list[str] = []
    if ambiguous:
        notices.append(
            "Left alone because two bones answer to the same name: " + ", ".join(ambiguous) + "."
        )
    if cubic:
        notices.append(
            f"{cubic} smoothed (CUBICSPLINE) channel{'s' if cubic != 1 else ''} came in as "
            "linear keys, sampled at the frame rate: Clay keeps no tangent handles."
        )
    return Imported(tuple(animations), fps, matched, unmatched, tuple(notices))


def _retarget(
    node: gltf.Node, bone: Any, props: dict[str, gltf.AnimChannel], fps: int, label: str
) -> tuple[dict[str, anim.Channel], int]:
    """One bone's channels as a change from the source node's rest pose, from the bone's."""
    out: dict[str, anim.Channel] = {}
    resampled = 0
    for path, channel in props.items():
        # The 2026-10-14 audit, finding clay-rig-08 (the animimport half): both
        # refusals below name the animation, the bone and the property, the way
        # ``rigimport``'s own door does, instead of a bare subject.
        where = f"{label} ({bone.name!r}, {path})"
        times, values, was_cubic = _linear_keys(channel, fps, where)
        resampled += int(was_cubic)
        if len(times) > anim.MAX_KEYS_PER_CHANNEL:
            raise OpError(
                f"{label} has {len(times):,} keys on {bone.name!r}'s {path}, past the "
                f"{anim.MAX_KEYS_PER_CHANNEL:,} a Clay channel holds. Bake it to fewer keys first."
            )
        if len(times) and times[-1] > anim.MAX_ANIMATION_SECONDS:
            raise OpError(
                f"{label} runs {times[-1]:.0f} seconds, past the {anim.MAX_ANIMATION_SECONDS:.0f} "
                "a Clay clip holds."
            )
        if path == "rotation":
            rest_src = m3.quat_normalize(np.asarray(node.rotation, dtype="f8"))
            rest_bone = m3.quat_normalize(np.asarray(bone.rotation, dtype="f8"))
            undo = m3.quat_conjugate(rest_src)
            rows = np.array(
                [
                    m3.quat_normalize(
                        m3.quat_mul(rest_bone, m3.quat_mul(undo, m3.quat_normalize(row)))
                    )
                    for row in values
                ]
            )
        elif path == "translation":
            rows = np.asarray(values, dtype="f8") - np.asarray(node.translation, dtype="f8")
            rows = rows + np.asarray(bone.translation, dtype="f8")
        else:
            rest_src = np.asarray(node.scale, dtype="f8")
            if not np.all(np.abs(rest_src) > 1e-9):
                raise OpError(f"{label}'s rest size for {bone.name!r} has a zero axis.")
            rows = np.asarray(values, dtype="f8") / rest_src * np.asarray(bone.scale, dtype="f8")
        out[path] = _channel(times, rows, where)
    return out, resampled
