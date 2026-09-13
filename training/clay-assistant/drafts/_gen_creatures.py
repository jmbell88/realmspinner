"""Deterministic emitter for ``drafts/creatures.jsonl`` -- the `kind: build`
family covering ``clay_add_figure`` presets (placed, scaled, sometimes
lightly modified with an extra primitive, a boolean, a mirror/array op, a
deletion-and-replacement of one part) plus hand-built creatures assembled
from primitives, and tapered bodies built with chained ``sweep``/``tube``
segments (a neck, a snout, a coiled or spiralled body).

Run directly to (re)write ``creatures.jsonl`` beside this file:

    uv run python training/clay-assistant/drafts/_gen_creatures.py

50 hand-authored base builds (one per line of ``seeds/creatures.txt``, in
seed order: 25 easy, 20 medium, 5 hard) x 4 prompt phrasings each = 200
records, ids ``creatures-0001`` .. ``creatures-0200`` assigned in recipe
order. Every phrasing of a base build shares its call list unless a variant
explicitly perturbs a number (noted inline); the prompts vary in length,
register, stated dimensions and setting per the brief's "short/long,
casual/precise, with and without dimensions, different settings" rule.

Geometry notes shared by every recipe:

* Figures arrive already grounded (``presets.build``'s own grounding pass) --
  the six terrestrial keys need no translation to sit on y=0. ``serpent`` and
  ``fish`` are the two swimmers (:data:`presets.SWIMMERS`) and are authored
  floating a template-fixed height above the grid on purpose, which is why
  none of the fish/serpent-flavoured builds below sets ``allow_below_ground``
  -- they were never below it to begin with.
* A hand-built creature's own ground clearance is computed by this module
  from the actual radii/heights it places (see ``_lowest_y`` calls below),
  not eyeballed, so a later tweak to a magic number cannot quietly sink an
  object under the verifier's -0.02 m floor.
* Chained tapered ``sweep`` segments (a neck, a snout, a spiral tail) are
  built by :func:`chain_segments`, which keeps every segment's bend to a
  single world axis (X for a vertical curve, Y for a horizontal one) so the
  per-segment ``rotation`` is a single Euler component -- no quaternion
  composition needed to keep segments end-to-end.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

OUT_PATH = Path(__file__).with_name("creatures.jsonl")

# --- figure part names (from src/warlock/studio/clay/presets.py) ------------
# The exact ``Part.name`` strings each assembly builder emits, before a
# ``name_prefix`` is applied -- copied here as data for the same reason
# presets.py copies its own landmarks from the rig template: cross-checking
# against a live import would buy this authoring script a runtime dependency
# on ``warlock.studio`` it does not otherwise need, at the cost of a script
# that silently drifts if a preset is ever renamed (verify.check would catch
# a stale $ref immediately -- it is a real replay, not a guess).
PART_NAMES: dict[str, list[str]] = {
    "humanoid": [
        "Hips",
        "Spine",
        "Chest",
        "Neck",
        "Head",
        "Shoulder.L",
        "Upper arm.L",
        "Forearm.L",
        "Hand.L",
        "Shoulder.R",
        "Upper arm.R",
        "Forearm.R",
        "Hand.R",
        "Thigh.L",
        "Shin.L",
        "Foot.L",
        "Thigh.R",
        "Shin.R",
        "Foot.R",
    ],
    "biped_tail": [
        "Hips",
        "Spine",
        "Chest",
        "Neck",
        "Head",
        "Shoulder.L",
        "Upper arm.L",
        "Forearm.L",
        "Hand.L",
        "Shoulder.R",
        "Upper arm.R",
        "Forearm.R",
        "Hand.R",
        "Thigh.L",
        "Shin.L",
        "Foot.L",
        "Thigh.R",
        "Shin.R",
        "Foot.R",
        "Tail 01",
        "Tail 02",
        "Tail 03",
        "Tail 04",
        "Tail 05",
    ],
    "quadruped": [
        "Hips",
        "Spine",
        "Chest",
        "Neck",
        "Head",
        "Tail 01",
        "Tail 02",
        "Front upper.L",
        "Front lower.L",
        "Front foot.L",
        "Front upper.R",
        "Front lower.R",
        "Front foot.R",
        "Rear upper.L",
        "Rear lower.L",
        "Rear foot.L",
        "Rear upper.R",
        "Rear lower.R",
        "Rear foot.R",
    ],
    "bird": [
        "Hips",
        "Spine",
        "Chest",
        "Neck",
        "Head",
        "Beak",
        "Tail",
        "Tail tip",
        "Wing base.L",
        "Wing mid.L",
        "Wing tip.L",
        "Wing base.R",
        "Wing mid.R",
        "Wing tip.R",
        "Thigh.L",
        "Shin.L",
        "Foot.L",
        "Thigh.R",
        "Shin.R",
        "Foot.R",
    ],
    "serpent": [
        "Spine 01",
        "Spine 02",
        "Spine 03",
        "Spine 04",
        "Spine 05",
        "Spine 06",
        "Spine 07",
        "Tail tip",
        "Neck",
        "Head",
    ],
    "fish": [
        "Spine 01",
        "Spine 02",
        "Spine 03",
        "Spine 04",
        "Tail fin",
        "Head",
        "Jaw",
        "Dorsal fin",
        "Ventral fin",
        "Pectoral fin.L",
        "Pelvic fin.L",
        "Pectoral fin.R",
        "Pelvic fin.R",
    ],
    "insect": [
        "Thorax",
        "Head",
        "Abdomen",
        "Mandible.L",
        "Mandible.R",
        "Leg A upper.L",
        "Leg A lower.L",
        "Leg B upper.L",
        "Leg B lower.L",
        "Leg C upper.L",
        "Leg C lower.L",
        "Leg A upper.R",
        "Leg A lower.R",
        "Leg B upper.R",
        "Leg B lower.R",
        "Leg C upper.R",
        "Leg C lower.R",
    ],
    "blob": ["Base", "Core", "Crown", "Top", "Lobe front", "Lobe back", "Lobe.L", "Lobe.R"],
}


def refs(key: str, prefix: str) -> list[dict[str, str]]:
    return [{"$ref": f"{prefix}{n}"} for n in PART_NAMES[key]]


def refs_except(key: str, prefix: str, exclude: set[str]) -> list[dict[str, str]]:
    """Like :func:`refs`, minus the part names in *exclude* -- for a recipe
    that ``clay_delete``\\ s some of a preset's own parts before its
    ``clay_material`` call: a $ref to a name that no longer exists is a
    refusal ("no object named ..."), not a no-op, so a material call must
    never re-list a part its own batch already deleted."""
    return [{"$ref": f"{prefix}{n}"} for n in PART_NAMES[key] if n not in exclude]


def ref(name: str) -> dict[str, str]:
    return {"$ref": name}


# --- call builders ------------------------------------------------------------


def figure(key: str, prefix: str, *, translation=None, yaw=None, scale=None) -> dict[str, Any]:
    args: dict[str, Any] = {"key": key, "name_prefix": prefix}
    if translation is not None:
        args["translation"] = list(translation)
    if yaw is not None:
        args["yaw"] = yaw
    if scale is not None:
        args["scale"] = scale
    return {"name": "clay_add_figure", "arguments": args}


def primitive(
    generator: str,
    name: str,
    *,
    params=None,
    translation=None,
    rotation=None,
    scale=None,
) -> dict[str, Any]:
    args: dict[str, Any] = {"generator": generator, "name": name}
    if params is not None:
        args["params"] = params
    if translation is not None:
        args["translation"] = list(translation)
    if rotation is not None:
        args["rotation"] = list(rotation)
    if scale is not None:
        args["scale"] = list(scale)
    return {"name": "clay_add_primitive", "arguments": args}


def material(uids: list, color, mat_name: str, *, roughness=None, metallic=None) -> dict[str, Any]:
    args: dict[str, Any] = {"uids": uids, "color": list(color), "name": mat_name}
    if roughness is not None:
        args["roughness"] = roughness
    if metallic is not None:
        args["metallic"] = metallic
    return {"name": "clay_material", "arguments": args}


def transform(uid, *, translation=None, rotation=None, scale=None) -> dict[str, Any]:
    args: dict[str, Any] = {"uid": uid}
    if translation is not None:
        args["translation"] = list(translation)
    if rotation is not None:
        args["rotation"] = list(rotation)
    if scale is not None:
        args["scale"] = list(scale)
    return {"name": "clay_transform", "arguments": args}


def set_params(uid, params: dict) -> dict[str, Any]:
    return {"name": "clay_set_params", "arguments": {"uid": uid, "params": params}}


def delete(uids: list) -> dict[str, Any]:
    return {"name": "clay_delete", "arguments": {"uids": uids}}


def select(uids: list) -> dict[str, Any]:
    return {"name": "clay_select", "arguments": {"uids": uids}}


def op(name: str, params: dict | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {"name": name}
    if params:
        args["params"] = params
    return {"name": "clay_op", "arguments": args}


def boolean(kind: str, uids: list) -> dict[str, Any]:
    return {"name": "clay_boolean", "arguments": {"kind": kind, "uids": uids}}


def circle_outline(r: float, n: int = 8) -> list[list[float]]:
    """An ``n``-gon of circumradius ``r``, for a sweep segment's cross-section."""
    return [
        [round(r * math.cos(2 * math.pi * i / n), 5), round(r * math.sin(2 * math.pi * i / n), 5)]
        for i in range(n)
    ]


def chain_segments(
    *,
    origin,
    start_angle_deg: float,
    step_angle_deg,
    seg_len,
    r0: float,
    r_ratio,
    n: int,
    bend_axis: str,
    prefix: str,
) -> tuple[list[dict[str, Any]], tuple[float, float, float], float]:
    """A chain of *n* tapered ``sweep`` frustums, end to end, bending about one
    world axis so each segment's own ``rotation`` is a single Euler
    component -- see the module docstring's "chained tapered sweep segments"
    note. ``r_ratio`` may be one float (constant per-segment taper) or a list
    of *n* floats (a schedule, for an unevenly tapering or tightening chain).
    ``step_angle_deg`` may likewise be one float or a per-segment list.

    Returns the calls, plus the tip (far-end) world position and radius, so
    a caller can cap the chain with a head/point primitive.
    """
    ratios = r_ratio if isinstance(r_ratio, list) else [r_ratio] * n
    steps = step_angle_deg if isinstance(step_angle_deg, list) else [step_angle_deg] * n
    lengths = seg_len if isinstance(seg_len, list) else [seg_len] * n
    assert len(ratios) == n and len(steps) == n and len(lengths) == n

    calls: list[dict[str, Any]] = []
    pos = list(origin)
    angle = start_angle_deg
    radius = r0
    for i in range(n):
        length = lengths[i]
        if bend_axis == "X":
            rad = math.radians(angle)
            direction = (0.0, -math.sin(rad), math.cos(rad))
            rotation = (angle, 0.0, 0.0)
        elif bend_axis == "Y":
            rad = math.radians(angle)
            direction = (math.sin(rad), 0.0, math.cos(rad))
            rotation = (0.0, angle, 0.0)
        else:
            raise ValueError(bend_axis)

        center = tuple(pos[a] + direction[a] * length * 0.5 for a in range(3))
        far_radius = radius * ratios[i]
        calls.append(
            primitive(
                "sweep",
                f"{prefix}{i + 1}",
                params={
                    "outline": circle_outline(radius),
                    "depth": length,
                    "taper": max(ratios[i], 1e-3),
                    "twist": 0.0,
                    "sections": 1,
                },
                translation=[round(c, 5) for c in center],
                rotation=list(rotation),
            )
        )
        pos = [pos[a] + direction[a] * length for a in range(3)]
        radius = far_radius
        angle += steps[i]

    return calls, tuple(pos), radius


# --- material palette ---------------------------------------------------------
# Named so a recipe reads "SKIN_GREY" rather than a bare RGB triple; reused
# across recipes on purpose (a real session would reuse a palette index too).
SKIN_TAN = (0.80, 0.63, 0.50)
SKIN_PALE = (0.86, 0.78, 0.72)
SKIN_GREEN = (0.42, 0.55, 0.30)
SKIN_GREY_GREEN = (0.55, 0.60, 0.52)
HIDE_BROWN = (0.45, 0.32, 0.20)
HIDE_DUN = (0.62, 0.52, 0.36)
FEATHER_GREYBROWN = (0.48, 0.42, 0.36)
SCALE_BLUEGREEN = (0.20, 0.45, 0.42)
SCALE_SILVER = (0.68, 0.72, 0.76)
SLIME_GREEN = (0.35, 0.62, 0.30)
SLIME_BLUE = (0.30, 0.45, 0.65)
CHITIN_DARKBROWN = (0.22, 0.16, 0.10)
CHITIN_BLACK = (0.06, 0.06, 0.07)
METAL_GREY = (0.55, 0.56, 0.58)
CROC_GREEN = (0.30, 0.40, 0.25)
DRAGON_RED = (0.55, 0.12, 0.10)
WORM_PINK = (0.72, 0.55, 0.52)
CHAMELEON_GREEN = (0.35, 0.65, 0.35)


# =============================================================================
# Recipes: (slug, tier, notes, calls, [4 prompt phrasings], allow_below_ground)
# =============================================================================
Recipe = tuple[str, str, str, list[dict[str, Any]], list[str], bool]
recipes: list[Recipe] = []


def add(slug, tier, notes, calls, prompts, *, allow_below_ground=False):
    assert len(prompts) in (3, 4, 5, 6), f"{slug}: {len(prompts)} phrasings"
    recipes.append((slug, tier, notes, calls, prompts, allow_below_ground))


# ---- EASY (25): one seed line each, an unmodified (or lightly scaled/dressed)
# ---- clay_add_figure preset plus at most one material call -- the seed
# ---- file's own definition of "easy" for this family.

add(
    "humanoid-neutral",
    "easy",
    "A plain standing humanoid: the unmodified rig, one skin material. "
    "Nothing about 'neutral pose' needs changing -- the preset's own rest "
    "pose is the neutral one.",
    [
        figure("humanoid", "hero_"),
        material(refs("humanoid", "hero_"), SKIN_TAN, "Skin", roughness=0.6),
    ],
    [
        "a simple humanoid figure standing in a neutral pose",
        "humanoid, neutral standing pose, nothing fancy",
        "Place a plain humanoid figure in a relaxed standing pose, about 1.7 metres tall, with a plain skin material.",
        "a humanoid recruit standing at ease before drill practice",
    ],
)

add(
    "quadruped-stocky",
    "easy",
    "Unmodified quadruped preset; 'stocky' is left to the rig's own "
    "proportions rather than a non-uniform scale the tool has no argument "
    "for.",
    [
        figure("quadruped", "beast_", scale=1.05),
        material(refs("quadruped", "beast_"), HIDE_BROWN, "Hide", roughness=0.8),
    ],
    [
        "a basic quadruped creature, four-legged and stocky",
        "stocky four-legged beast, basic build",
        "A stocky quadruped creature roughly 1 metre at the shoulder, four legs, a plain hide material.",
        "a squat pack-beast pausing at a trailhead",
    ],
)

add(
    "bird-folded-wings",
    "easy",
    "Unmodified bird preset at a reduced scale for 'small'; the preset's "
    "wings are already modelled folded against a hunched perch posture.",
    [
        figure("bird", "wren_", scale=0.55),
        material(refs("bird", "wren_"), FEATHER_GREYBROWN, "Feathers", roughness=0.7),
    ],
    [
        "a small bird figure with folded wings",
        "small bird, wings folded in",
        "A small bird figure, about 30 centimetres tall, wings folded at rest, plain grey-brown feathers.",
        "a garden sparrow perched on a modern balcony rail",
    ],
)

add(
    "fish-streamlined",
    "easy",
    "Unmodified fish preset -- a swimmer, authored floating above the grid "
    "on purpose (presets.SWIMMERS), so no translation is needed to clear the "
    "ground check.",
    [
        figure("fish", "koi_"),
        material(refs("fish", "koi_"), SCALE_BLUEGREEN, "Scales", roughness=0.35),
    ],
    [
        "a simple fish figure, streamlined body",
        "plain streamlined fish",
        "A simple fish figure with a streamlined body about 40 centimetres long, blue-green scales.",
        "a koi drifting in a shrine's ornamental pond",
    ],
)

add(
    "blob-round-simple",
    "easy",
    "Unmodified blob preset, one translucent-looking material (roughness low to read as slick).",
    [figure("blob", "ooze_"), material(refs("blob", "ooze_"), SLIME_GREEN, "Ooze", roughness=0.25)],
    [
        "a blobby amorphous creature, round and simple",
        "a round blob thing",
        "A simple round amorphous blob creature, about a metre tall, glossy green.",
        "a stray ooze wobbling in a dungeon corridor",
    ],
)

add(
    "insect-six-legs",
    "easy",
    "Unmodified insect preset (the template is six-legged already), one dark chitin material.",
    [
        figure("insect", "bug_", scale=0.7),
        material(refs("insect", "bug_"), CHITIN_DARKBROWN, "Chitin", roughness=0.4),
    ],
    [
        "a basic insect figure with six legs",
        "plain six-legged insect",
        "A basic insect figure, six legs, roughly 25 centimetres long, dark brown chitin.",
        "a beetle crossing a garage floor under a work light",
    ],
)

add(
    "goblin-stout-humanoid",
    "easy",
    "Humanoid preset scaled down for a stout goblin build, sickly green "
    "skin material -- no shape edits beyond the uniform scale.",
    [
        figure("humanoid", "goblin_", scale=0.78),
        material(refs("humanoid", "goblin_"), SKIN_GREEN, "Goblin skin", roughness=0.65),
    ],
    [
        "a stout goblin-like humanoid figure",
        "short stout goblin guy",
        "A stout goblin-like humanoid figure, about 1.3 metres tall, sickly green skin.",
        "a goblin sentry grumbling at his post by the gate",
    ],
)

add(
    "biped-short-tail",
    "easy",
    "Unmodified biped_tail preset -- its five-segment tail already tapers "
    "to a point quickly, which reads as short next to a full serpent's "
    "chain.",
    [
        figure("biped_tail", "imp_"),
        material(refs("biped_tail", "imp_"), SKIN_GREY_GREEN, "Hide", roughness=0.55),
    ],
    [
        "a simple biped creature with a short tail",
        "biped with a stubby tail",
        "A simple biped creature roughly 1.6 metres tall with a short tapering tail, grey-green hide.",
        "an imp lounging near a furnace grate",
    ],
)

add(
    "slime-round-plain",
    "easy",
    "Unmodified blob preset again, distinct from ooze-round-simple by scale "
    "and a bluer, more translucent-reading material -- two different "
    "records for two different seed lines, same base assembly.",
    [
        figure("blob", "slime_", scale=0.9),
        material(refs("blob", "slime_"), SLIME_BLUE, "Slime", roughness=0.2, metallic=0.0),
    ],
    [
        "a plain round slime creature",
        "a round blue slime",
        "A plain round slime creature, about 90 centimetres tall, pale blue and glossy.",
        "a slime creature squelching across a cave floor",
    ],
)

add(
    "beetle-small-round",
    "easy",
    "Insect preset at a small scale with a shiny black material for a beetle's carapace read.",
    [
        figure("insect", "beetle_", scale=0.35),
        material(refs("insect", "beetle_"), CHITIN_BLACK, "Carapace", roughness=0.2, metallic=0.5),
    ],
    [
        "a small round beetle-like insect figure",
        "tiny round beetle",
        "A small, round, beetle-like insect figure about 12 centimetres long, glossy black carapace.",
        "a beetle basking on a warm garden stone",
    ],
)

add(
    "pack-animal-quadruped",
    "easy",
    "Quadruped preset scaled up slightly for a mule/donkey-sized pack animal, dun coat.",
    [
        figure("quadruped", "mule_", scale=1.15),
        material(refs("quadruped", "mule_"), HIDE_DUN, "Coat", roughness=0.75),
    ],
    [
        "a basic four-legged pack animal figure",
        "basic pack animal, four legs",
        "A basic four-legged pack-animal figure, about 1.15 metres at the shoulder, dun-coloured coat.",
        "a mule laden with saddlebags on a mountain trail",
    ],
)

add(
    "bird-long-neck",
    "easy",
    "Unmodified bird preset. Its neck (a single capsule bone) is already "
    "the rig's own longest bone relative to the head it carries; a genuine "
    "elongation would mean moving the head to follow a stretched neck, "
    "which is a transform edit past 'an unmodified preset' -- left for a "
    "medium-tier record instead, per the seed file's own difficulty rule.",
    [
        figure("bird", "heron_", scale=0.8),
        material(refs("bird", "heron_"), FEATHER_GREYBROWN, "Feathers", roughness=0.65),
    ],
    [
        "a simple long-necked bird figure",
        "long-necked bird, basic",
        "A simple long-necked bird figure, standing about 65 centimetres tall.",
        "a heron-like wading bird at a garden pond's edge",
    ],
)

add(
    "guard-humanoid-attention",
    "easy",
    "Unmodified humanoid preset with a grey metallic-leaning material "
    "reading as armour, standing in the rig's own rest pose ('attention' is "
    "a material/setting cue, not a pose edit).",
    [
        figure("humanoid", "guard_"),
        material(refs("humanoid", "guard_"), METAL_GREY, "Armour", roughness=0.5, metallic=0.3),
    ],
    [
        "a plain humanoid guard figure, standing at attention",
        "guard figure, standing straight",
        "A plain humanoid guard figure, about 1.75 metres tall, standing at attention, dull steel-grey armour tone.",
        "a palace guard holding position at the throne room door",
    ],
)

add(
    "mount-quadruped-horse",
    "easy",
    "Quadruped preset scaled up for horse-like proportions, brown coat.",
    [
        figure("quadruped", "steed_", scale=1.3),
        material(refs("quadruped", "steed_"), HIDE_BROWN, "Coat", roughness=0.7),
    ],
    [
        "a basic quadruped mount, horse-like proportions",
        "horse-sized mount, basic",
        "A basic quadruped mount with horse-like proportions, about 1.3 metres at the shoulder, brown coat.",
        "a cavalry mount waiting saddled outside the stables",
    ],
)

add(
    "biped-winged",
    "easy",
    "The bird preset is literally the rig's own 'winged biped' assembly; a "
    "second, distinctly scaled and coloured record from the folded-wings "
    "one above.",
    [
        figure("bird", "harpy_", scale=1.1),
        material(refs("bird", "harpy_"), FEATHER_GREYBROWN, "Feathers", roughness=0.6),
    ],
    [
        "a simple winged biped figure",
        "winged biped, plain",
        "A simple winged biped figure, roughly 1 metre tall, standing on two legs with wings folded at its sides.",
        "a harpy-like sentinel perched on a ruined parapet",
    ],
)

add(
    "jellyfish-blob",
    "easy",
    "Blob preset, pale translucent-reading material and a slightly reduced "
    "scale for a bell-shaped 'jellyfish' read off the same lobed assembly.",
    [
        figure("blob", "jelly_", scale=0.7),
        material(refs("blob", "jelly_"), (0.55, 0.55, 0.72), "Bell", roughness=0.15),
    ],
    [
        "a plain round jellyfish-like blob creature",
        "jellyfish-ish round blob",
        "A plain round jellyfish-like creature, about 70 centimetres tall, pale translucent violet.",
        "a jellyfish drifting past an aquarium viewing window",
    ],
)

add(
    "crawler-six-legged",
    "easy",
    "Insect preset, low scale and a duller olive chitin for a "
    "ground-crawling read distinct from beetle-small-round.",
    [
        figure("insect", "crawler_", scale=0.5),
        material(refs("insect", "crawler_"), (0.30, 0.34, 0.18), "Chitin", roughness=0.55),
    ],
    [
        "a basic six-legged crawling insect",
        "six-legged crawler",
        "A basic six-legged crawling insect, about 20 centimetres long, dull olive chitin.",
        "a crawling insect scuttling along a warehouse floor",
    ],
)

add(
    "fish-round-belly",
    "easy",
    "Fish preset with a bright silver material, distinct record from "
    "fish-streamlined by material and scale.",
    [
        figure("fish", "carp_", scale=1.1),
        material(refs("fish", "carp_"), SCALE_SILVER, "Scales", roughness=0.3),
    ],
    [
        "a simple fish figure with a rounded belly",
        "fish, round belly",
        "A simple fish figure roughly 45 centimetres long with a rounded belly, bright silver scales.",
        "a fat carp circling a temple koi pond",
    ],
)

add(
    "dwarf-stout-humanoid",
    "easy",
    "Humanoid preset scaled down heavily for a dwarf read, tan skin.",
    [
        figure("humanoid", "dwarf_", scale=0.68),
        material(refs("humanoid", "dwarf_"), SKIN_TAN, "Skin", roughness=0.6),
    ],
    [
        "a stout dwarf-like humanoid figure",
        "short stout dwarf",
        "A stout dwarf-like humanoid figure, about 1.15 metres tall, tan skin.",
        "a dwarven smith taking a break outside the forge",
    ],
)

add(
    "biped-stubby-tail",
    "easy",
    "biped_tail preset -- its five tapering segments already read as short "
    "next to a full snake's, distinct record from biped-short-tail by "
    "scale and material.",
    [
        figure("biped_tail", "kobold_", scale=0.85),
        material(refs("biped_tail", "kobold_"), (0.55, 0.42, 0.30), "Hide", roughness=0.6),
    ],
    [
        "a plain biped figure with a short stubby tail",
        "biped, stubby tail",
        "A plain biped figure about 1.4 metres tall with a short stubby tail, tan-brown hide.",
        "a kobold skulking at the edge of a mine tunnel",
    ],
)

add(
    "cat-sized-quadruped",
    "easy",
    "Quadruped preset scaled well down for a cat-sized, low-slung read.",
    [
        figure("quadruped", "cat_", scale=0.35),
        material(refs("quadruped", "cat_"), (0.20, 0.18, 0.16), "Fur", roughness=0.8),
    ],
    [
        "a basic quadruped, cat-sized and low to the ground",
        "cat-sized quadruped, low profile",
        "A basic quadruped about 35 centimetres at the shoulder, cat-sized and low to the ground, dark fur.",
        "a stray cat-like creature slinking along an alley wall",
    ],
)

add(
    "bird-perched-upright",
    "easy",
    "Bird preset at a modest scale; the rig's own perch posture is already upright.",
    [
        figure("bird", "finch_", scale=0.5),
        material(refs("bird", "finch_"), (0.65, 0.55, 0.30), "Feathers", roughness=0.6),
    ],
    [
        "a simple bird figure perched upright",
        "bird, perched, upright",
        "A simple bird figure about 25 centimetres tall, perched upright, warm yellow-brown feathers.",
        "a canary perched upright on a display stand",
    ],
)

add(
    "puddle-amorphous",
    "easy",
    "Not the blob preset -- a flattened, hand-built icosphere reads as a "
    "puddle where the tall lobed blob assembly does not, and the family "
    "wants hand-built primitive creatures alongside the figure presets. "
    "Scaled to sit flat: an icosphere of radius 0.5 squashed to 0.11 in Y "
    "(so half-height 0.055) sits with its lowest point at y=0 once "
    "translated up by that half-height.",
    [
        primitive(
            "icosphere",
            "puddle",
            params={"radius": 0.5, "subdivisions": 2},
            translation=[0.0, 0.055, 0.0],
            scale=[1.5, 0.11, 1.5],
        ),
        material([ref("puddle")], (0.30, 0.36, 0.20), "Ooze", roughness=0.2),
    ],
    [
        "a plain amorphous puddle-like creature",
        "flat puddle-blob thing",
        "A plain amorphous puddle-like creature, flattened and wide, about 1.5 metres across and 10 centimetres tall.",
        "a puddle-slime creature spreading across a sewer floor",
    ],
)

add(
    "humanoid-child-small",
    "easy",
    "Humanoid preset scaled well down for a child's proportions.",
    [
        figure("humanoid", "kid_", scale=0.55),
        material(refs("humanoid", "kid_"), SKIN_PALE, "Skin", roughness=0.6),
    ],
    [
        "a basic humanoid child figure, small proportions",
        "small child humanoid",
        "A basic humanoid child figure, about 95 centimetres tall, pale skin.",
        "a child-sized humanoid figure standing in a schoolyard",
    ],
)

add(
    "quadruped-long-legged",
    "easy",
    "Quadruped preset scaled up slightly; 'long-legged and slender' is left "
    "to the rig's own greyhound-ish proportions rather than a per-limb edit, "
    "which an unmodified-preset easy record does not reach for.",
    [
        figure("quadruped", "hound_", scale=1.05),
        material(refs("quadruped", "hound_"), (0.68, 0.62, 0.52), "Coat", roughness=0.6),
    ],
    [
        "a simple quadruped, long-legged and slender",
        "slender long-legged quadruped",
        "A simple quadruped about 1.05 metres at the shoulder, long-legged and slender, pale tan coat.",
        "a lean hound-like creature loping across open ground",
    ],
)

assert len([r for r in recipes if r[1] == "easy"]) == 25

# ---- MEDIUM (20): a boolean, an array/mirror op, a lathe/sweep profile, or
# ---- an element-mode selection + topology op, over a figure preset or a
# ---- small hand-built body.

add(
    "quadruped-tube-tail",
    "medium",
    "Quadruped preset, its own short two-segment tail deleted and replaced "
    "with one longer 'tube' primitive (a swept profile, per the seed's own "
    "medium-tier definition) curving up and back from the hip. Tube points "
    "are in the tail's own local frame before translation; the whole thing "
    "sits well clear of the ground since the hip attachment (y=0.70) is "
    "already mid-body height.",
    [
        figure("quadruped", "q_"),
        delete([ref("q_Tail 01"), ref("q_Tail 02")]),
        primitive(
            "tube",
            "q_Tail",
            params={
                "path": [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.10, -0.18],
                    [0.0, 0.22, -0.34],
                    [0.0, 0.28, -0.52],
                ],
                "radius": 0.045,
                "sides": 8,
            },
            translation=[0.0, 0.68, -0.35],
        ),
        material(
            refs_except("quadruped", "q_", {"Tail 01", "Tail 02"}) + [ref("q_Tail")],
            HIDE_BROWN,
            "Hide",
            roughness=0.75,
        ),
    ],
    [
        "a quadruped creature with an elongated tube-shaped tail",
        "quadruped, long tube tail",
        "A quadruped creature about a metre at the shoulder with its stub tail replaced by one elongated tube-shaped tail roughly 60 centimetres long.",
        "a beast of burden with an unusually long whip-like tail",
    ],
)

add(
    "biped-curling-tail",
    "medium",
    "biped_tail preset, then the last three tail segments stretched "
    "(clay_set_params, taller capsules) and rotated upward "
    "(clay_transform) segment by segment so the tip curls -- a parameter "
    "and transform edit over the preset, the medium-tier's own "
    "'placed and scaled, sometimes with extra work' end of the scale.",
    [
        figure("biped_tail", "b_"),
        set_params(ref("b_Tail 03"), {"height": 0.14}),
        set_params(ref("b_Tail 04"), {"height": 0.16}),
        set_params(ref("b_Tail 05"), {"height": 0.12}),
        transform(ref("b_Tail 04"), rotation=[-35.0, 0.0, 0.0]),
        transform(ref("b_Tail 05"), rotation=[-70.0, 0.0, 0.0]),
        material(refs("biped_tail", "b_"), SKIN_GREEN, "Hide", roughness=0.6),
    ],
    [
        "a biped figure with an exaggerated long tail curling upward",
        "biped, long curling tail",
        "A biped creature about 1.5 metres tall with an exaggerated, elongated tail that curls upward at the tip.",
        "an imp-like biped whose tail curls up behind it like a scorpion's",
    ],
)

add(
    "bird-outstretched-wings",
    "medium",
    "Bird preset with its own six folded-wing parts deleted; one "
    "outstretched wing hand-built from three boxes at the left shoulder, "
    "then clay_op mirror-copy (axis X, offset 0 -- the body's own "
    "centreline) makes the mirrored right wing, exercising the array/mirror "
    "family the seed's medium tier names.",
    [
        figure("bird", "w_"),
        delete(
            [
                ref("w_Wing base.L"),
                ref("w_Wing mid.L"),
                ref("w_Wing tip.L"),
                ref("w_Wing base.R"),
                ref("w_Wing mid.R"),
                ref("w_Wing tip.R"),
            ]
        ),
        primitive(
            "box",
            "w_wing_root",
            params={"size": [0.22, 0.03, 0.14]},
            translation=[0.15, 0.69, 0.06],
            rotation=[10.0, 0.0, 0.0],
        ),
        primitive(
            "box",
            "w_wing_mid",
            params={"size": [0.20, 0.02, 0.11]},
            translation=[0.33, 0.72, 0.05],
            rotation=[16.0, 0.0, 0.0],
        ),
        primitive(
            "box",
            "w_wing_tip",
            params={"size": [0.16, 0.015, 0.08]},
            translation=[0.49, 0.76, 0.03],
            rotation=[22.0, 0.0, 0.0],
        ),
        # Painted *before* mirror-copy: a duplicate carries the material slot
        # its source object already had, but has no name of its own to
        # address afterward (mirror-copy's own new object is unnamed by
        # this door), so the only way the mirrored wing ends up feathered
        # rather than default-grey is to paint the source first.
        material(
            [ref("w_wing_root"), ref("w_wing_mid"), ref("w_wing_tip")],
            FEATHER_GREYBROWN,
            "Feathers",
            roughness=0.6,
        ),
        select([ref("w_wing_root"), ref("w_wing_mid"), ref("w_wing_tip")]),
        op("mirror-copy", {"axis": 0, "offset": 0.0}),
        material(
            refs_except(
                "bird",
                "w_",
                {
                    "Wing base.L",
                    "Wing mid.L",
                    "Wing tip.L",
                    "Wing base.R",
                    "Wing mid.R",
                    "Wing tip.R",
                },
            ),
            FEATHER_GREYBROWN,
            "Feathers",
            roughness=0.6,
        ),
    ],
    [
        "a bird figure with a mirrored pair of outstretched wings",
        "bird, wings spread, mirrored",
        "A bird figure roughly a metre tall with a mirrored pair of fully outstretched wings, each wing about 65 centimetres from shoulder to tip.",
        "a bird-like creature caught mid-glide, wings spread wide",
    ],
)

add(
    "fish-tapered-dorsal",
    "medium",
    "Fish preset plus one extra dorsal fin built from a tapered sweep "
    "(a swept profile per the seed's medium definition) instead of the "
    "preset's own flat box fin, run along the fish's back.",
    [
        figure("fish", "f_"),
        primitive(
            "sweep",
            "f_dorsal",
            params={
                "outline": [[-0.10, 0.0], [0.10, 0.0], [0.0, 0.16]],
                "depth": 0.30,
                "taper": 0.35,
                "twist": 0.0,
                "sections": 1,
            },
            translation=[0.0, 0.72, 0.70],
            rotation=[90.0, 0.0, 0.0],
        ),
        material(refs("fish", "f_") + [ref("f_dorsal")], SCALE_BLUEGREEN, "Scales", roughness=0.35),
    ],
    [
        "a fish figure with a tapered tube fin running along its back",
        "fish, tapered dorsal ridge",
        "A fish figure about 50 centimetres long with a tapered dorsal ridge fin running along its back, wide at the front and narrowing toward the tail.",
        "a fish with an unusually tall, tapered dorsal fin",
    ],
)

add(
    "insect-radial-legs",
    "medium",
    "Hand-built rather than the insect preset's own bilateral rig: a squat "
    "capsule thorax at the origin, one leg built once, then clay_select + "
    "clay_op array-radial (count 6, axis Y) rings six legs evenly around "
    "it -- the spoked-hub fixture's own technique, applied to a body "
    "instead of a wheel.",
    [
        primitive(
            "capsule",
            "r_body",
            params={"radius": 0.16, "height": 0.05, "segments": 16, "rings": 4},
            translation=[0.0, 0.21, 0.0],
        ),
        primitive(
            "uv_sphere",
            "r_head",
            params={"radius": 0.09, "segments": 16, "rings": 8},
            translation=[0.0, 0.28, 0.20],
        ),
        primitive(
            "capsule",
            "r_leg",
            params={"radius": 0.022, "height": 0.18, "segments": 12, "rings": 4},
            translation=[0.30, 0.10, 0.0],
            rotation=[0.0, 0.0, 65.0],
        ),
        # Painted before the array: array-radial's five new copies carry
        # r_leg's material slot outward with them but mint no name of their
        # own, so painting after the array would leave five of six legs
        # default-grey.
        material(
            [ref("r_body"), ref("r_head"), ref("r_leg")], CHITIN_DARKBROWN, "Chitin", roughness=0.4
        ),
        select([ref("r_leg")]),
        op("array-radial", {"count": 6, "angle": 360.0, "axis": 1}),
    ],
    [
        "an insect figure with legs arranged in a radial pattern around its body",
        "insect, legs in a radial ring",
        "A squat insect-like creature about 40 centimetres across with six legs arranged in an even radial pattern around its body.",
        "a radially-legged crawling bug scuttling in a tight circle",
    ],
)

add(
    "quadruped-stretched-thin",
    "medium",
    "Quadruped preset with its three body-mass ellipsoids (Hips/Spine/"
    "Chest) re-transformed: each one's own scale (which is how presets.py "
    "stretches a uv_sphere into a body mass) is changed to pull the barrel "
    "longer and pinch it narrower than the base preset's own numbers.",
    [
        figure("quadruped", "q_"),
        transform(ref("q_Hips"), scale=[1.0, 1.75, 0.85]),
        transform(ref("q_Spine"), scale=[0.85, 2.1, 0.75]),
        transform(ref("q_Chest"), scale=[1.05, 1.9, 1.05]),
        material(refs("quadruped", "q_"), (0.55, 0.48, 0.38), "Hide", roughness=0.7),
    ],
    [
        "a quadruped whose body is stretched and thinned compared to the base preset",
        "quadruped, stretched thin body",
        "A quadruped creature with its torso stretched noticeably longer and thinner than the base rig's own proportions.",
        "a weasel-like quadruped with an elongated, narrow body",
    ],
)

add(
    "biped-bulky-tube-tail",
    "medium",
    "Humanoid preset with its torso masses scaled up for bulk, plus one "
    "short 'tube' primitive stub tail added at the hip -- the tube "
    "generator the seed's own medium tier names.",
    [
        figure("humanoid", "h_"),
        transform(ref("h_Chest"), scale=[1.75, 1.35, 1.15]),
        transform(ref("h_Spine"), scale=[1.55, 1.15, 1.05]),
        transform(ref("h_Hips"), scale=[1.55, 1.15, 1.10]),
        primitive(
            "tube",
            "h_tail",
            params={
                "path": [[0.0, 0.0, 0.0], [0.0, -0.02, -0.10], [0.0, -0.01, -0.18]],
                "radius": 0.05,
                "sides": 8,
            },
            translation=[0.0, 0.55, -0.16],
        ),
        material(refs("humanoid", "h_") + [ref("h_tail")], SKIN_GREY_GREEN, "Hide", roughness=0.6),
    ],
    [
        "a biped creature with a bulky humanoid torso and a stubby tube tail",
        "biped, bulky torso, stub tail",
        "A biped creature with a noticeably bulky, broad torso and one stubby tube-shaped tail at the base of its spine.",
        "a hulking biped brute with a short stub of a tail",
    ],
)

add(
    "blob-eye-bumps",
    "medium",
    "Blob preset with four small icosphere 'eyes' union-booleaned onto "
    "three of its lobes -- clay_boolean is the seed's own named medium "
    "technique here, fusing each bump into the body it sits on rather than "
    "leaving it a separate floating sphere.",
    [
        figure("blob", "e_"),
        primitive(
            "icosphere",
            "e_eye1",
            params={"radius": 0.05, "subdivisions": 1},
            translation=[0.10, 0.62, 0.16],
        ),
        primitive(
            "icosphere",
            "e_eye2",
            params={"radius": 0.045, "subdivisions": 1},
            translation=[-0.09, 0.58, 0.17],
        ),
        primitive(
            "icosphere",
            "e_eye3",
            params={"radius": 0.04, "subdivisions": 1},
            translation=[0.0, 0.72, 0.15],
        ),
        boolean("union", [ref("e_Crown"), ref("e_eye1")]),
        boolean("union", [ref("e_Crown"), ref("e_eye2")]),
        boolean("union", [ref("e_Top"), ref("e_eye3")]),
        material(refs("blob", "e_"), SLIME_GREEN, "Ooze", roughness=0.2),
    ],
    [
        "a blob creature with several eye-like bumps added across its surface",
        "blob with eye bumps",
        "A blob creature roughly a metre tall with several small eye-like bumps fused across its upper surface.",
        "an ooze dotted with a handful of eye-like bumps on its crown",
    ],
)

add(
    "bird-oversized-beak",
    "medium",
    "Bird preset with its Beak part resized via clay_set_params to a much "
    "larger box -- a generator-parameter edit over the preset, distinct "
    "from an unmodified easy build.",
    [
        figure("bird", "t_"),
        set_params(ref("t_Beak"), {"size": [0.10, 0.42, 0.10]}),
        material(refs("bird", "t_"), (0.75, 0.55, 0.15), "Feathers", roughness=0.55),
    ],
    [
        "a bird figure with an oversized beak extending from its head",
        "bird, huge beak",
        "A bird figure about 80 centimetres tall with a heavily oversized beak extending well out from its head, toucan-like.",
        "a toucan-like bird with an enormous beak",
    ],
)

add(
    "quadruped-armour-plate",
    "medium",
    "Quadruped preset with a box 'plate' positioned over the chest mass "
    "and boolean-unioned into it -- fused armour rather than a loose "
    "add-on, exercising clay_boolean.",
    [
        figure("quadruped", "q_"),
        # Chest is a _mass ellipsoid whose world extents (its near-horizontal
        # bone's "along" axis lands on world Z, its "depth" on world Y --
        # see presets._mass's own docstring) are roughly y in [0.55, 0.91],
        # z in [0.03, 0.42] around its centre (0, 0.73, 0.225); the plate
        # sits inside that box so the union actually overlaps its surface
        # instead of leaving a disconnected box floating near the head.
        primitive(
            "box", "q_plate", params={"size": [0.30, 0.05, 0.22]}, translation=[0.0, 0.82, 0.23]
        ),
        boolean("union", [ref("q_Chest"), ref("q_plate")]),
        material(refs("quadruped", "q_"), HIDE_DUN, "Hide", roughness=0.7),
        material([ref("q_Chest")], METAL_GREY, "Plating", roughness=0.35, metallic=0.6),
    ],
    [
        "a quadruped with a boolean-cut armour plate set into its back",
        "armoured quadruped, plate on back",
        "A quadruped creature roughly a metre at the shoulder with one metal armour plate fused into its back over the shoulders.",
        "a war-beast with a steel plate set into its back",
    ],
)

add(
    "snake-loose-coil",
    "medium",
    "Hand-built, legless: one 'tube' whose path spirals loosely inward "
    "(a slackening loop, not a tight ring) at a constant height equal to "
    "its own radius plus a small margin, so the coil rests on the ground "
    "along its whole length rather than at one point.",
    [
        primitive(
            "tube",
            "snake_body",
            params={
                "path": [
                    [0.34, 0.0, 0.0],
                    [0.24, 0.0, 0.26],
                    [0.0, 0.0, 0.34],
                    [-0.22, 0.0, 0.22],
                    [-0.28, 0.0, -0.04],
                    [-0.12, 0.0, -0.22],
                    [0.08, 0.0, -0.16],
                ],
                "radius": 0.05,
                "sides": 10,
            },
            translation=[0.0, 0.065, 0.0],
        ),
        material([ref("snake_body")], (0.30, 0.42, 0.22), "Scales", roughness=0.4),
    ],
    [
        "a coiled snake resting in a loose loop, no legs",
        "snake, loosely coiled, legless",
        "A legless snake about 30 centimetres across, its body resting coiled in one loose, open loop on the ground.",
        "a garden snake coiled loosely in a sunny patch",
    ],
)

add(
    "fish-mirrored-fins",
    "medium",
    "Fish preset with its own default pectoral/pelvic fins deleted, one "
    "fin hand-built once and clay_op mirror-copy (axis X) making the "
    "matching fin on the other side -- the array/mirror family again, this "
    "time on a fish.",
    [
        figure("fish", "m_"),
        delete(
            [
                ref("m_Pectoral fin.L"),
                ref("m_Pelvic fin.L"),
                ref("m_Pectoral fin.R"),
                ref("m_Pelvic fin.R"),
            ]
        ),
        primitive(
            "box",
            "m_pect_fin",
            params={"size": [0.14, 0.02, 0.09]},
            translation=[0.14, 0.66, 0.42],
            rotation=[0.0, 0.0, -20.0],
        ),
        # Painted before mirror-copy -- see bird-outstretched-wings' own
        # comment for why the mirrored copy would otherwise stay grey.
        material([ref("m_pect_fin")], SCALE_SILVER, "Scales", roughness=0.3),
        select([ref("m_pect_fin")]),
        op("mirror-copy", {"axis": 0, "offset": 0.0}),
        material(
            refs_except(
                "fish", "m_", {"Pectoral fin.L", "Pelvic fin.L", "Pectoral fin.R", "Pelvic fin.R"}
            ),
            SCALE_SILVER,
            "Scales",
            roughness=0.3,
        ),
    ],
    [
        "a fish figure with a set of small fins mirrored along both sides",
        "fish, small mirrored fins",
        "A fish figure about 45 centimetres long with a small pair of fins mirrored evenly on both sides of its body.",
        "a fish with a neat mirrored pair of side fins",
    ],
)

add(
    "humanoid-tube-horn",
    "medium",
    "Humanoid preset with one oversized, gently curved 'tube' horn added "
    "at the crown of the head -- a swept profile per the medium tier.",
    [
        figure("humanoid", "d_"),
        primitive(
            "tube",
            "d_horn",
            # Head is a uv_sphere centred at (0, 0.95, 0), radius 0.09 (crown
            # at y=1.04, z=0 -- the humanoid rig's whole spine runs at z=0,
            # unlike quadruped's), so the horn's root sits right at the
            # crown rather than out at some unrelated z offset.
            params={
                "path": [[0.0, 0.0, 0.0], [0.02, 0.07, 0.01], [0.03, 0.15, 0.04]],
                "radius": 0.045,
                "sides": 10,
            },
            translation=[0.0, 1.02, 0.0],
        ),
        material(refs("humanoid", "d_"), SKIN_TAN, "Skin", roughness=0.6),
        material([ref("d_horn")], (0.30, 0.26, 0.20), "Horn", roughness=0.4),
    ],
    [
        "a humanoid figure with a single oversized tube-shaped horn on its head",
        "humanoid, one big horn",
        "A humanoid figure about 1.7 metres tall with a single oversized, tube-shaped horn rising from the top of its head.",
        "a horned humanoid with one large curling horn",
    ],
)

add(
    "insect-swept-antennae",
    "medium",
    "Insect preset with two thin tapered 'sweep' antennae (a swept "
    "profile) added at the head, angled back along the body.",
    [
        figure("insect", "a_", scale=0.6),
        primitive(
            "sweep",
            "a_antenna_l",
            params={
                "outline": [[-0.008, -0.008], [0.008, -0.008], [0.008, 0.008], [-0.008, 0.008]],
                "depth": 0.22,
                "taper": 0.15,
                "twist": 0.0,
                "sections": 1,
            },
            translation=[0.03, 0.43, 0.72],
            rotation=[65.0, 0.0, 8.0],
        ),
        primitive(
            "sweep",
            "a_antenna_r",
            params={
                "outline": [[-0.008, -0.008], [0.008, -0.008], [0.008, 0.008], [-0.008, 0.008]],
                "depth": 0.22,
                "taper": 0.15,
                "twist": 0.0,
                "sections": 1,
            },
            translation=[-0.03, 0.43, 0.72],
            rotation=[65.0, 0.0, -8.0],
        ),
        material(
            refs("insect", "a_") + [ref("a_antenna_l"), ref("a_antenna_r")],
            CHITIN_DARKBROWN,
            "Chitin",
            roughness=0.4,
        ),
    ],
    [
        "an insect figure with two long antennae swept back from its head",
        "insect, long swept antennae",
        "An insect figure about 20 centimetres long with two long, tapered antennae swept back from its head.",
        "a beetle-like bug with two long antennae trailing backward",
    ],
)

add(
    "quadruped-tube-tail-stub-legs",
    "medium",
    "Quadruped preset, its two-segment tail deleted and replaced with one "
    "short tube (a swept profile), legs left as the preset's own capsules "
    "-- 'stubby' read off a smaller overall scale rather than a per-limb "
    "edit.",
    [
        figure("quadruped", "s_", scale=0.85),
        delete([ref("s_Tail 01"), ref("s_Tail 02")]),
        primitive(
            "tube",
            "s_tail",
            params={
                "path": [[0.0, 0.0, 0.0], [0.0, 0.03, -0.08], [0.0, 0.02, -0.14]],
                "radius": 0.05,
                "sides": 8,
            },
            translation=[0.0, 0.60, -0.30],
        ),
        material(
            refs_except("quadruped", "s_", {"Tail 01", "Tail 02"}) + [ref("s_tail")],
            HIDE_DUN,
            "Hide",
            roughness=0.75,
        ),
    ],
    [
        "a quadruped with a short tube tail and stubby capsule legs",
        "quadruped, short tube tail",
        "A quadruped creature, scaled down a little for a stubby build, with one short tube-shaped tail and its usual capsule legs.",
        "a squat quadruped critter with a short stub tail",
    ],
)

add(
    "biped-mirrored-back-flaps",
    "medium",
    "biped_tail preset with one wing-like flap hand-built at the shoulder "
    "blade and clay_op mirror-copy (axis X) producing the matching flap on "
    "the other side.",
    [
        figure("biped_tail", "p_"),
        primitive(
            "box",
            "p_flap",
            params={"size": [0.05, 0.34, 0.20]},
            translation=[0.13, 0.80, -0.14],
            rotation=[0.0, 0.0, -18.0],
        ),
        # Painted before mirror-copy -- see bird-outstretched-wings' own
        # comment for why the mirrored copy would otherwise stay grey.
        material([ref("p_flap")], (0.32, 0.30, 0.35), "Hide", roughness=0.5),
        select([ref("p_flap")]),
        op("mirror-copy", {"axis": 0, "offset": 0.0}),
        material(refs("biped_tail", "p_"), (0.32, 0.30, 0.35), "Hide", roughness=0.5),
    ],
    [
        "a biped figure with wing-like flaps mirrored across its back",
        "biped, mirrored back flaps",
        "A biped creature about 1.6 metres tall with a mirrored pair of wing-like flaps set across its shoulder blades.",
        "a gremlin-like biped with small mirrored flaps on its back",
    ],
)

add(
    "blob-worm-tube",
    "medium",
    "Not the blob preset -- a hand-built 'tube' body (a swept profile) "
    "along a gently curved path at a constant, wider radius, capped with a "
    "small icosphere head. Constant radius (no taper) is what keeps this "
    "one medium rather than hard, per the seed file's own line between the "
    "two.",
    [
        primitive(
            "tube",
            "worm_body",
            params={
                "path": [
                    [-0.30, 0.0, 0.0],
                    [-0.10, 0.02, 0.0],
                    [0.12, -0.01, 0.0],
                    [0.34, 0.02, 0.0],
                ],
                "radius": 0.11,
                "sides": 10,
            },
            translation=[0.0, 0.13, 0.0],
        ),
        primitive(
            "icosphere",
            "worm_head",
            params={"radius": 0.11, "subdivisions": 2},
            translation=[-0.30, 0.13, 0.0],
        ),
        material([ref("worm_body"), ref("worm_head")], WORM_PINK, "Skin", roughness=0.5),
    ],
    [
        "a blob creature stretched into a long tube-like worm shape",
        "blob, stretched into a worm",
        "A blob creature stretched out into a long tube-like worm shape, about 70 centimetres long, pale pink.",
        "an earthworm-like creature stretched long and tube-shaped",
    ],
)

add(
    "bird-one-leg-tucked",
    "medium",
    "Bird preset with its right leg (Thigh/Shin/Foot) rotated up and "
    "forward to tuck it against the body, the left leg left standing on "
    "the ground -- transform edits over the preset's own parts.",
    [
        figure("bird", "u_"),
        transform(ref("u_Thigh.R"), rotation=[95.0, 0.0, 0.0]),
        transform(ref("u_Shin.R"), rotation=[140.0, 0.0, 0.0]),
        transform(ref("u_Foot.R"), rotation=[95.0, 0.0, 0.0]),
        material(refs("bird", "u_"), FEATHER_GREYBROWN, "Feathers", roughness=0.6),
    ],
    [
        "a bird figure standing on one leg with the other tucked up",
        "bird, one leg tucked",
        "A bird figure standing balanced on one leg, its other leg folded up tight against its body.",
        "a heron-like bird resting on one leg by the water",
    ],
)

add(
    "snake-tight-ring",
    "medium",
    "Hand-built from a plain 'torus' -- a tight ring is exactly what that "
    "generator already is, so this reaches for the primitive rather than a "
    "hand-authored path. One small icosphere head sits at the ring's inner "
    "edge.",
    [
        primitive(
            "torus",
            "ring_body",
            params={"radius": 0.22, "tube": 0.045, "segments": 24, "sides": 16},
            translation=[0.0, 0.045, 0.0],
        ),
        primitive(
            "icosphere",
            "ring_head",
            params={"radius": 0.05, "subdivisions": 1},
            translation=[0.265, 0.045, 0.0],
        ),
        material([ref("ring_body"), ref("ring_head")], (0.28, 0.40, 0.24), "Scales", roughness=0.4),
    ],
    [
        "a small coiled snake figure curled into a tight ring, legless",
        "tiny snake, tight ring",
        "A small legless snake, about 55 centimetres across, curled into one tight, closed ring.",
        "a snake curled into a tight ring, dozing in the sun",
    ],
)

add(
    "quadruped-mirrored-legs",
    "medium",
    "Not the quadruped preset -- all four of its own legs are deleted and "
    "replaced by one hand-built front leg and one hand-built rear leg, "
    "each mirrored across the body's centreline with clay_op mirror-copy, "
    "for a literal 'mirrored left-right pair' rather than the preset's "
    "already-bilateral rig.",
    [
        figure("quadruped", "l_"),
        delete(
            [
                ref("l_Front upper.L"),
                ref("l_Front lower.L"),
                ref("l_Front foot.L"),
                ref("l_Front upper.R"),
                ref("l_Front lower.R"),
                ref("l_Front foot.R"),
                ref("l_Rear upper.L"),
                ref("l_Rear lower.L"),
                ref("l_Rear foot.L"),
                ref("l_Rear upper.R"),
                ref("l_Rear lower.R"),
                ref("l_Rear foot.R"),
            ]
        ),
        primitive(
            "capsule",
            "l_front_leg",
            params={"radius": 0.055, "height": 0.42, "segments": 12, "rings": 4},
            translation=[0.12, 0.32, 0.55],
        ),
        primitive(
            "capsule",
            "l_rear_leg",
            params={"radius": 0.06, "height": 0.46, "segments": 12, "rings": 4},
            translation=[0.12, 0.30, -0.55],
        ),
        # Painted before mirror-copy -- see bird-outstretched-wings' own
        # comment for why the mirrored copy would otherwise stay grey.
        material([ref("l_front_leg"), ref("l_rear_leg")], HIDE_BROWN, "Hide", roughness=0.7),
        select([ref("l_front_leg"), ref("l_rear_leg")]),
        op("mirror-copy", {"axis": 0, "offset": 0.0}),
        material(
            refs_except(
                "quadruped",
                "l_",
                {
                    "Front upper.L",
                    "Front lower.L",
                    "Front foot.L",
                    "Front upper.R",
                    "Front lower.R",
                    "Front foot.R",
                    "Rear upper.L",
                    "Rear lower.L",
                    "Rear foot.L",
                    "Rear upper.R",
                    "Rear lower.R",
                    "Rear foot.R",
                },
            ),
            HIDE_BROWN,
            "Hide",
            roughness=0.7,
        ),
    ],
    [
        "a quadruped whose legs are arranged as a mirrored left-right pair",
        "quadruped, legs mirrored left-right",
        "A quadruped creature whose front and rear legs are each built once and mirrored to form a matched left-right pair.",
        "a quadruped with cleanly matched, mirrored legs on each side",
    ],
)

assert len([r for r in recipes if r[1] == "medium"]) == 20

# ---- HARD (5): a swept/tapered form along a path -- chained tapered
# ---- ``sweep`` frustums (:func:`chain_segments`), sometimes with a ``tube``
# ---- for a constant-radius run between tapering ends.

_croc_body = figure("quadruped", "c_", scale=1.1)
_croc_snout, _croc_tip, _croc_tip_r = chain_segments(
    origin=(0.0, 0.93, 0.50),
    start_angle_deg=6.0,
    step_angle_deg=10.0,
    seg_len=0.16,
    r0=0.075,
    r_ratio=[0.75, 0.55, 0.35],
    n=3,
    bend_axis="X",
    prefix="c_snout",
)
add(
    "crocodile-tapered-snout",
    "hard",
    "Quadruped preset body/legs/tail kept as-is; the default Head/Neck "
    "stay (the snout grows out from just in front of the head). Three "
    "chained tapered sweep frustums (chain_segments, bend about world X so "
    "each segment's rotation is one Euler component) run forward and "
    "slightly down from the head, radius 0.075 tapering through 0.75/0.55/"
    "0.35 per segment to a narrow point -- the seed's own 'swept curve to a "
    "narrow point'.",
    [
        _croc_body,
        *_croc_snout,
        material(
            refs("quadruped", "c_") + [ref(f"c_snout{i + 1}") for i in range(3)],
            CROC_GREEN,
            "Scales",
            roughness=0.55,
        ),
    ],
    [
        "a crocodile-like creature whose long snout tapers along a swept curve to a narrow point",
        "croc creature, tapering snout",
        "A crocodile-like quadruped roughly 1.5 metres long, its long snout tapering along a gentle downward curve to a narrow point.",
        "a crocodilian creature with a long, tapering snout ending in a narrow tip",
    ],
)

_snake_mid, _snake_mid_tip, _snake_mid_r = chain_segments(
    origin=(0.0, 0.10, -0.35),
    start_angle_deg=-25.0,
    step_angle_deg=18.0,
    seg_len=0.22,
    r0=0.10,
    r_ratio=1.0,
    n=2,
    bend_axis="Y",
    prefix="serp_mid",
)
_snake_head, _snake_head_tip, _snake_head_r = chain_segments(
    origin=_snake_mid_tip,
    start_angle_deg=-25.0 + 2 * 18.0,
    step_angle_deg=14.0,
    seg_len=0.16,
    r0=_snake_mid_r,
    r_ratio=[0.55, 0.15],
    n=2,
    bend_axis="Y",
    prefix="serp_head",
)
_snake_tail, _snake_tail_tip, _snake_tail_r = chain_segments(
    origin=(0.0, 0.10, -0.35),
    start_angle_deg=-25.0 + 180.0,
    step_angle_deg=-16.0,
    seg_len=0.20,
    r0=0.10,
    r_ratio=[0.55, 0.18],
    n=2,
    bend_axis="Y",
    prefix="serp_tail",
)
add(
    "serpent-thick-middle-tapered-ends",
    "hard",
    "Hand-built, legless, entirely from chained sweep frustums bent about "
    "world Y (a horizontal curve, the whole body lying flat at constant "
    "height r0 + margin above the ground -- see chain_segments). Two "
    "constant-radius segments (r_ratio=1.0) form the thick middle; from "
    "each end of that middle, a second chain tapers down (ratios 0.55 then "
    "0.15/0.18) to a narrow point, one running 'forward' from the middle's "
    "far end and one running 'backward' from its near end (start_angle_deg "
    "offset by 180 degrees) so the two tapering ends curve away from each "
    "other rather than doubling back on the middle section.",
    [
        *_snake_mid,
        *_snake_head,
        *_snake_tail,
        material(
            [ref(f"serp_mid{i + 1}") for i in range(2)]
            + [ref(f"serp_head{i + 1}") for i in range(2)]
            + [ref(f"serp_tail{i + 1}") for i in range(2)],
            (0.32, 0.44, 0.20),
            "Scales",
            roughness=0.4,
        ),
    ],
    [
        "a legless serpent whose entire body sweeps in one long smooth curve, thickest in the middle and narrow at both ends",
        "legless serpent, thick middle, tapered ends",
        "A legless serpent roughly 1.3 metres long, its whole body following one smooth curve, thickest around the middle and tapering to a narrow point at each end.",
        "a snake-like creature thick through the middle, curving away to two narrow points",
    ],
)

_dragon_body = figure("quadruped", "dr_", scale=1.2)
_dragon_delete = delete([ref("dr_Neck"), ref("dr_Head")])
_dragon_neck, _dragon_tip, _dragon_tip_r = chain_segments(
    origin=(0.0, 0.80, 0.32),
    start_angle_deg=-8.0,
    step_angle_deg=-24.0,
    seg_len=0.20,
    r0=0.11,
    r_ratio=[0.85, 0.65, 0.45],
    n=3,
    bend_axis="X",
    prefix="dr_neck",
)
add(
    "dragon-tapering-neck",
    "hard",
    "Quadruped preset for the body/legs/tail; its own short Neck and Head "
    "deleted and replaced with three chained tapered sweep frustums curving "
    "from a forward lean up toward vertical (start -8 degrees, stepping "
    "-24 degrees per segment about world X) while tapering 0.85/0.65/0.45 "
    "per segment, capped with a uv_sphere head at the narrow tip.",
    [
        _dragon_body,
        _dragon_delete,
        *_dragon_neck,
        primitive(
            "uv_sphere",
            "dr_head",
            params={"radius": _dragon_tip_r * 1.15, "segments": 16, "rings": 8},
            translation=list(_dragon_tip),
        ),
        material(
            refs_except("quadruped", "dr_", {"Neck", "Head"})
            + [ref(f"dr_neck{i + 1}") for i in range(3)]
            + [ref("dr_head")],
            DRAGON_RED,
            "Scales",
            roughness=0.45,
        ),
    ],
    [
        "a dragon-like creature whose neck sweeps and tapers gracefully up toward its head",
        "dragon creature, tapering neck",
        "A dragon-like quadruped roughly 1.8 metres long, its neck sweeping upward in a graceful curve and tapering as it rises toward a small head.",
        "a dragon-like beast with a long, curving, tapering neck",
    ],
)

_worm_chain, _worm_tip, _worm_tip_r = chain_segments(
    origin=(0.0, 0.11, -0.30),
    start_angle_deg=0.0,
    step_angle_deg=-9.0,
    seg_len=0.20,
    r0=0.11,
    r_ratio=[0.82, 0.78, 0.7, 0.45],
    n=4,
    bend_axis="X",
    prefix="worm_seg",
)
add(
    "worm-even-taper-curved-sweep",
    "hard",
    "Hand-built worm: four chained tapered sweep frustums bending gently "
    "about world X (start 0 degrees, -9 degrees per segment -- a shallow "
    "curve, not the dragon neck's sharp turn upward) and tapering evenly "
    "(0.82/0.78/0.7/0.45 per segment) from a rounded uv_sphere head at the "
    "thick end down to a near-point at the tail.",
    [
        primitive(
            "uv_sphere",
            "worm_head",
            params={"radius": 0.11, "segments": 16, "rings": 8},
            translation=[0.0, 0.11, -0.30],
        ),
        *_worm_chain,
        material(
            [ref("worm_head")] + [ref(f"worm_seg{i + 1}") for i in range(4)],
            WORM_PINK,
            "Skin",
            roughness=0.55,
        ),
    ],
    [
        "a worm-like creature, long-bodied and evenly tapering, following one gentle curved sweep",
        "worm, even taper, gentle curve",
        "A worm-like creature about 90 centimetres long, its body tapering evenly along one gently curved sweep from a rounded head to a narrow tail.",
        "a segmented worm creature tapering smoothly along a shallow curve",
    ],
)

_cham_body = figure("quadruped", "ch_", scale=0.55)
_cham_delete = delete([ref("ch_Tail 01"), ref("ch_Tail 02")])
_cham_tail, _cham_tail_tip, _cham_tail_r = chain_segments(
    # Origin is the scaled (0.55x) quadruped's own hip landmark (clay
    # (0, 0.70, -0.35) at scale 1) -- the tail root the deleted Tail 01/02
    # used to occupy. start_angle_deg=160 (bend_axis Y: direction =
    # (sin, 0, cos)) points the first segment rearward and slightly
    # outward (-Z, +X) rather than back into the body, and the increasing
    # per-segment step then curls it into a tightening spiral.
    origin=(0.0, 0.385, -0.19),
    start_angle_deg=160.0,
    step_angle_deg=[35.0, 45.0, 55.0, 65.0, 75.0],
    seg_len=[0.16, 0.13, 0.10, 0.08, 0.06],
    r0=0.05,
    r_ratio=[0.85, 0.75, 0.6, 0.45, 0.2],
    n=5,
    bend_axis="Y",
    prefix="ch_tail",
)
add(
    "chameleon-spiral-tail",
    "hard",
    "Quadruped preset scaled down for a chameleon build, its own two-"
    "segment tail deleted and replaced with five chained tapered sweep "
    "frustums bending about world Y (a horizontal curl) with an "
    "increasing per-segment turn (35/45/55/65/75 degrees) and shrinking "
    "segment length (0.16 down to 0.06) so the curl tightens as it goes, "
    "while the radius tapers 0.85/0.75/0.6/0.45/0.2 per segment down to a "
    "near-point tip -- a tightening spiral rather than a constant-curvature "
    "loop.",
    [
        _cham_body,
        _cham_delete,
        *_cham_tail,
        material(
            refs_except("quadruped", "ch_", {"Tail 01", "Tail 02"})
            + [ref(f"ch_tail{i + 1}") for i in range(5)],
            CHAMELEON_GREEN,
            "Skin",
            roughness=0.5,
        ),
    ],
    [
        "a chameleon-like creature whose tail curls and tapers into a tight spiral",
        "chameleon, spiral tail",
        "A chameleon-like quadruped about 60 centimetres long whose tail curls in a tightening spiral, tapering to a fine point.",
        "a chameleon-like creature with a tail curled into a tight tapering spiral",
    ],
)

assert len([r for r in recipes if r[1] == "hard"]) == 5


def main() -> None:
    records: list[dict[str, Any]] = []
    counter = 1
    for slug, _tier, notes, calls, prompts, allow_below_ground in recipes:
        for prompt in prompts:
            rec: dict[str, Any] = {
                "id": f"creatures-{counter:04d}",
                "family": "creatures",
                "kind": "build",
                "prompt": prompt,
                "calls": calls,
                "notes": f"[{slug}] {notes}",
            }
            if allow_below_ground:
                rec["allow_below_ground"] = True
            records.append(rec)
            counter += 1

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"wrote {len(records)} records ({len(recipes)} base builds) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
