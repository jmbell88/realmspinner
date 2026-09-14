"""Deterministic emitter for ``drafts/grounding.jsonl`` -- the ``kind: build``
family that teaches the model to actually rest an object on the ground
(``bbox`` min ``y >= -0.02``), for the cases run A's own eval measured as the
programme's biggest single miss: 19 of 232 replies were built but failed the
ground check, mostly rotated cylinders/cones and profile generators (lathe/
sweep/tube re-centre their profile/outline/path on its own bounding box
before ``translation`` applies, so a naively-guessed y is wrong the moment
the object is rotated or its silhouette is not symmetric about its own
origin) placed at a guessed y, plus multi-part objects with one part hanging
below another.

Run directly to (re)write ``grounding.jsonl`` beside this file:

    uv run python training/clay-assistant/drafts/_gen_grounding.py

38 hand-authored base builds x 4 prompt phrasings each = 152 records, ids
``grounding-0001`` .. ``grounding-0152`` assigned in recipe order.

Two idioms, roughly half the base builds each (see ``IDIOM`` on every
:func:`add` call):

* ``"computed"`` -- the recipe works out the object's real lowest point
  itself and places it exactly resting, using the closed-form geometry
  below (:func:`cyl_drop_y`, :func:`cone_drop_y`, :func:`capsule_drop_y`,
  :func:`lathe_half_height`) or, for the multi-part hanging assemblies,
  plain arithmetic over each part's own half-height. Used for rotated
  cylinders/cones/capsules (any tilt about world X, optionally with an
  additional yaw about world Y -- see "Why X-tilt + Y-yaw" below), upright
  (unrotated) ``lathe`` vessels, and every hanging assembly: a part that
  must keep a fixed relationship to another cannot be handed to
  ``drop-to-ground`` (see the next paragraph), so it is placed by
  computation or not at all.
* ``"safety-net"`` -- the batch places every part at a plausible but
  possibly-wrong height (the same naive guess a model that ignored rotation
  or profile re-centring would make -- sometimes literally the unrotated
  half-height formula, reused on a rotated object on purpose), then ends
  with ``clay_select`` on the placed part(s) and ``clay_op``
  ``drop-to-ground``. Their prompts say the object "rests on" / "lies on"
  the ground or floor, so the model learns that phrase maps to the drop.

**``drop-to-ground`` is per object, not per group**
(``clay_ops._drop_to_ground``'s own docstring: "Per object, not per group:
... a box sitting three metres above the floor and one already resting on
it both land correctly in the same call"). That is exactly why every
hanging assembly (the sign-on-arm, lantern-on-hook, bell-under-frame,
planter-on-arm and wind-chime-under-crossbar bases below) uses the
``"computed"`` idiom instead: selecting a hanging part together with its
support and dropping the selection would rest *each part's own box*
independently on y=0, destroying the very relationship ("hangs below")
the record exists to teach. The safety-net idiom is used only where every
selected object's own correct resting place really is the ground itself.

**Why X-tilt + Y-yaw, never a Z-tilt, for the closed-form (computed) rotated
primitives.** ``agent_clay._quat_from_euler_xyz`` composes ``R = Rz . Ry .
Rx`` (rotate about X first, then Y, then Z, all about the fixed world axes).
For a rotation ``[rx, ry, 0]`` (no Z term), a point's world Y coordinate is
``Ry(Rx(v)).y``, and ``Ry`` (a rotation about the world Y axis) never
changes a point's Y coordinate -- only its X and Z. So the object's height
depends on ``rx`` alone; ``ry`` is free to point the object in any
horizontal direction with no effect on where the ground check's own
``bbox.min_y`` lands. A nonzero ``rz`` breaks this (``Rz`` *does* mix a
point's rotated-by-``Ry`` X back into its Y), which is exactly why the
safety-net bases below are free to use compound tilts (``rx`` and ``rz``
both nonzero) while the computed ones never do.

Every generator here keeps its default ``segments``/``sides`` (16, a
multiple of 4), which is what makes :func:`cyl_drop_y` and
:func:`cone_drop_y` exact rather than approximate: the ring vertex closest
to the rotation's own extreme direction is a lattice point (at 90 degrees,
16/4 = 4 steps around), not an interpolated guess, for any tilt angle --
the derivation is in each function's own docstring.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

OUT_PATH = Path(__file__).with_name("grounding.jsonl")
SEED_PATH = Path(__file__).resolve().parents[1] / "seeds" / "grounding.txt"


# --- call builders (same shapes as drafts/_gen_creatures.py's own) ----------


def primitive(
    generator: str,
    name: str,
    *,
    params: dict | None = None,
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


def select(uids: list) -> dict[str, Any]:
    return {"name": "clay_select", "arguments": {"uids": uids}}


def op(name: str, params: dict | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {"name": name}
    if params:
        args["params"] = params
    return {"name": "clay_op", "arguments": args}


def ref(name: str) -> dict[str, str]:
    return {"$ref": name}


def drop_to_ground(*uids: dict) -> list[dict[str, Any]]:
    """The safety-net idiom's own closing pair: select the given ``$ref``\\ s,
    then ``clay_op drop-to-ground``. Always the last two calls of a
    ``"safety-net"`` recipe."""
    return [select(list(uids)), op("drop-to-ground")]


def read_seed_lines(path: Path) -> list[tuple[str, str]]:
    """Parse ``seeds/grounding.txt``'s own ``class | prompt`` lines (blank and
    ``#`` lines skipped) into ``(class, prompt)`` pairs, in file order.

    A hand-rolled parser rather than ``scripts/campaign_props.read_corpus``
    (the parser ``gen/holdout.py`` reuses for the same shape) on purpose:
    that module imports ``warlock`` transitively (``from warlock import
    guidance`` and friends), and this recipe -- like every other
    ``drafts/_gen_*.py`` -- must not.
    """
    entries: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        cls, _, prompt = text.partition("|")
        entries.append((cls.strip().lower(), prompt.strip()))
    return entries


# --- exact ground-offset geometry (the "computed" idiom's own maths) -------


def r4(value: float) -> float:
    return round(value, 4)


def cyl_drop_y(radius: float, height: float, tilt_x_deg: float) -> float:
    """The ``translation.y`` that rests a ``cylinder`` of this ``radius``/
    ``height`` on the ground after a rotation of ``tilt_x_deg`` about world
    X (see the module docstring for why only X, never Z).

    A ring at local ``y = +-h/2`` (``h = height/2``) has vertices
    ``(r*cos(phi), y, r*sin(phi))``; rotating by ``theta`` about X sends a
    vertex's world Y to ``y*cos(theta) - r*sin(phi)*sin(theta))``, whose
    minimum over ``phi`` is ``y*cos(theta) - r*abs(sin(theta))`` -- reached
    exactly at ``phi = 90`` or ``270`` degrees, which is a real mesh vertex
    for any ``segments`` divisible by 4 (the default, 16, is). The lower of
    the two rings (``y = -h/2`` beats ``y = +h/2`` once both are negated by
    the ``abs`` below) gives the object's true lowest point; negate it for
    the translation that lifts it back to ``y = 0``.
    """
    theta = math.radians(tilt_x_deg)
    h = abs(height) / 2.0
    r = abs(radius)
    return r4(h * abs(math.cos(theta)) + r * abs(math.sin(theta)))


def cone_drop_y(radius: float, height: float, tilt_x_deg: float) -> float:
    """As :func:`cyl_drop_y`, for a ``cone``: a base ring at local
    ``y = -h/2`` (same ring maths as the cylinder) plus a single apex at
    ``(0, +h/2, 0)``, whose world Y after the same X rotation is simply
    ``h/2 * cos(theta)`` (no ``phi`` to minimise over -- it is one point on
    the rotation axis's own perpendicular-free line). The object's lowest
    point is whichever of the two is smaller.
    """
    theta = math.radians(tilt_x_deg)
    h = abs(height) / 2.0
    r = abs(radius)
    apex_y = h * math.cos(theta)
    base_min_y = -h * math.cos(theta) - r * abs(math.sin(theta))
    return r4(-min(apex_y, base_min_y))


def capsule_drop_y(radius: float, cyl_height: float, tilt_x_deg: float) -> float:
    """As :func:`cyl_drop_y`, for a ``capsule``: a capsule is the Minkowski
    sum of a ball of ``radius`` and the segment from ``(0,-L,0)`` to
    ``(0,+L,0)`` (``L = cyl_height/2``), and for a convex shape built that
    way the true (continuous) lowest point is always the segment's own
    lowest point minus ``radius`` in Y -- exact regardless of ``segments``/
    ``rings``, unlike the cylinder and cone rings above, because a sphere
    has no facets to miss. After an X rotation the segment's endpoints are
    at world Y ``+-L*cos(theta)``, so the lower one is
    ``-L*abs(cos(theta))``.
    """
    theta = math.radians(tilt_x_deg)
    length = abs(cyl_height) / 2.0
    r = abs(radius)
    return r4(length * abs(math.cos(theta)) + r)


def lathe_half_height(profile: list[list[float]]) -> float:
    """The ``translation.y`` that rests an *upright* (unrotated) ``lathe``
    object built from *profile* on the ground.

    ``primitives._clamp_profile`` re-centres a profile's own ``y`` about the
    midpoint of its min and max before the mesh is built (station radius is
    untouched -- a revolve is already centred on its axis), so the built
    mesh spans ``y`` in ``[-(max-min)/2, +(max-min)/2]`` regardless of where
    *profile*'s own stations happened to start counting from. Half that
    range is exactly the translation that puts the low end at ``y = 0``.
    """
    ys = [station[1] for station in profile]
    return r4((max(ys) - min(ys)) / 2.0)


# =============================================================================
# Recipes: (slug, idiom, notes, calls, [4 prompt phrasings])
# =============================================================================
Recipe = tuple[str, str, str, list[dict[str, Any]], list[str]]
recipes: list[Recipe] = []


def add(slug: str, idiom: str, notes: str, calls: list[dict[str, Any]], prompts: list[str]) -> None:
    assert idiom in ("computed", "safety-net"), f"{slug}: bad idiom {idiom!r}"
    assert len(prompts) == 4, f"{slug}: {len(prompts)} phrasings"
    recipes.append((slug, idiom, notes, calls, prompts))


# --- materials --------------------------------------------------------------
BARK_BROWN = (0.35, 0.24, 0.14)
WOOD_PALE = (0.72, 0.60, 0.42)
WOOD_WEATHERED = (0.55, 0.48, 0.40)
STEEL = (0.62, 0.64, 0.67)
RUBBER_BLACK = (0.05, 0.05, 0.06)
SAFETY_ORANGE = (0.88, 0.32, 0.05)
PARTY_PINK = (0.85, 0.30, 0.55)
PLASTIC_RED = (0.70, 0.10, 0.08)
CANISTER_GREY = (0.58, 0.60, 0.62)
PILL_WHITE = (0.90, 0.88, 0.84)
GLAZE_BLUE = (0.22, 0.38, 0.55)
GLAZE_GREEN = (0.28, 0.48, 0.36)
GLASS_TEAL = (0.30, 0.55, 0.52)
CLAY_TERRACOTTA = (0.72, 0.42, 0.28)
IRON_DARK = (0.20, 0.20, 0.22)
BRASS = (0.65, 0.55, 0.25)
SIGN_CREAM = (0.88, 0.84, 0.72)
HOSE_GREEN = (0.20, 0.42, 0.22)
ROPE_TAN = (0.68, 0.58, 0.38)
COPPER = (0.62, 0.35, 0.22)
LEAD_GREY = (0.40, 0.40, 0.42)
FABRIC_NAVY = (0.16, 0.22, 0.36)
CHIME_SILVER = (0.75, 0.76, 0.80)
POST_GREY = (0.45, 0.45, 0.47)


# =============================================================================
# COMPUTED idiom (19 bases)
# =============================================================================

# ---- A: rotated cylinders, X-tilt only, translation from cyl_drop_y (5) ----

_a1_y = cyl_drop_y(0.16, 1.3, 82.0)
add(
    "log-lying-on-ground",
    "computed",
    "A single cylinder, tilted 82 degrees about world X (a shallow rest "
    "angle, not a flat 90, so the translation genuinely combines the "
    "height and radius terms of cyl_drop_y rather than reducing to a bare "
    "radius) plus a 25-degree Y yaw for facing variety, which cyl_drop_y's "
    "own docstring explains does not change the answer.",
    [
        primitive(
            "cylinder",
            "log",
            params={"radius": 0.16, "height": 1.3},
            translation=[0.0, _a1_y, 0.0],
            rotation=[82.0, 25.0, 0.0],
        ),
        material([ref("log")], BARK_BROWN, "Bark", roughness=0.85),
    ],
    [
        "a fallen log lying on the ground",
        "log, fallen, lying on the ground",
        "A fallen log roughly 1.3 metres long and 32 centimetres across, lying on the forest floor.",
        "a felled tree trunk section left lying in a clearing",
    ],
)

_a2_y = cyl_drop_y(0.42, 0.07, 90.0)
add(
    "wagon-wheel-standing",
    "computed",
    "A thin wide cylinder (a wagon-wheel silhouette) rotated a full 90 "
    "degrees about X so it stands up in its rolling orientation -- the "
    "generator's own default axis (vertical) would otherwise lay the disc "
    "flat.",
    [
        primitive(
            "cylinder",
            "wheel",
            params={"radius": 0.42, "height": 0.07},
            translation=[0.0, _a2_y, 0.0],
            rotation=[90.0, 0.0, 0.0],
        ),
        material([ref("wheel")], WOOD_PALE, "Rim", roughness=0.7),
    ],
    [
        "a wagon wheel standing upright on the ground",
        "wagon wheel, standing up",
        "A large wagon wheel, about 84 centimetres across, standing upright as if leaning against a cart.",
        "an old wooden wheel propped upright outside a barn",
    ],
)

_a3_y = cyl_drop_y(0.015, 0.8, 88.0)
add(
    "sword-blade-on-ground",
    "computed",
    "A thin long cylinder standing in for a sword blade, tilted 88 degrees "
    "(not a bare 90) plus a 40-degree yaw -- the exact case the plan's own "
    "measurement named: 'a sword blade at y=0'.",
    [
        primitive(
            "cylinder",
            "blade",
            params={"radius": 0.015, "height": 0.8},
            translation=[0.0, _a3_y, 0.0],
            rotation=[88.0, 40.0, 0.0],
        ),
        material([ref("blade")], STEEL, "Steel", roughness=0.3, metallic=0.8),
    ],
    [
        "a sword blade lying on the ground",
        "sword blade, dropped, on the ground",
        "A single sword blade, about 80 centimetres long, lying flat on the ground where it fell.",
        "a lone blade left lying in the dirt after a skirmish",
    ],
)

_a4_y = cyl_drop_y(0.04, 0.38, 85.0)
add(
    "rolling-pin-on-counter",
    "computed",
    "A short, thin cylinder (a rolling pin) tilted 85 degrees with a -20-degree yaw.",
    [
        primitive(
            "cylinder",
            "pin",
            params={"radius": 0.04, "height": 0.38},
            translation=[0.0, _a4_y, 0.0],
            rotation=[85.0, -20.0, 0.0],
        ),
        material([ref("pin")], WOOD_PALE, "Wood", roughness=0.5),
    ],
    [
        "a rolling pin lying on a kitchen counter",
        "rolling pin, on its side",
        "A wooden rolling pin, about 38 centimetres long, lying on its side on a countertop.",
        "a well-used rolling pin left out after baking",
    ],
)

_a5_y = cyl_drop_y(0.055, 1.7, 87.0)
add(
    "fence-rail-on-ground",
    "computed",
    "A long thin cylinder (a fence rail) tilted 87 degrees with a "
    "10-degree yaw, as if it slipped off a stack.",
    [
        primitive(
            "cylinder",
            "rail",
            params={"radius": 0.055, "height": 1.7},
            translation=[0.0, _a5_y, 0.0],
            rotation=[87.0, 10.0, 0.0],
        ),
        material([ref("rail")], WOOD_WEATHERED, "Weathered wood", roughness=0.8),
    ],
    [
        "a fence rail lying on the ground",
        "fence rail, fallen, on the ground",
        "A single weathered fence rail, about 1.7 metres long, lying on the ground beside its post.",
        "a loose rail that slipped off a half-built fence",
    ],
)

# ---- B: rotated cones, X-tilt only, translation from cone_drop_y (3) ------

_b1_y = cone_drop_y(0.18, 0.55, 75.0)
add(
    "traffic-cone-knocked-over",
    "computed",
    "A cone tilted 75 degrees (not a bare 90) plus a 30-degree yaw, so "
    "cone_drop_y's own min(apex, base-ring) comparison genuinely has to "
    "pick the base ring over the apex.",
    [
        primitive(
            "cone",
            "cone",
            params={"radius": 0.18, "height": 0.55},
            translation=[0.0, _b1_y, 0.0],
            rotation=[75.0, 30.0, 0.0],
        ),
        material([ref("cone")], SAFETY_ORANGE, "Plastic", roughness=0.55),
    ],
    [
        "a traffic cone knocked over on the ground",
        "traffic cone, knocked over",
        "A bright orange traffic cone, about 55 centimetres tall standing, now knocked over and lying on the ground.",
        "a road cone tipped over at the edge of a car park",
    ],
)

_b2_y = cone_drop_y(0.11, 0.32, 80.0)
add(
    "party-hat-fallen",
    "computed",
    "A smaller cone (a party hat) tilted 80 degrees with a 200-degree yaw.",
    [
        primitive(
            "cone",
            "hat",
            params={"radius": 0.11, "height": 0.32},
            translation=[0.0, _b2_y, 0.0],
            rotation=[80.0, 200.0, 0.0],
        ),
        material([ref("hat")], PARTY_PINK, "Paper", roughness=0.6),
    ],
    [
        "a party hat fallen on the floor",
        "party hat, fallen over",
        "A small conical party hat, about 32 centimetres tall standing, fallen on its side on the floor.",
        "a bright pink party hat knocked off a table onto the floor",
    ],
)

_b3_y = cone_drop_y(0.20, 0.48, 70.0)
add(
    "megaphone-on-ground",
    "computed",
    "A wider cone (a megaphone) tilted 70 degrees with a -60-degree yaw.",
    [
        primitive(
            "cone",
            "megaphone",
            params={"radius": 0.20, "height": 0.48},
            translation=[0.0, _b3_y, 0.0],
            rotation=[70.0, -60.0, 0.0],
        ),
        material([ref("megaphone")], PLASTIC_RED, "Plastic", roughness=0.4),
    ],
    [
        "a megaphone lying on the ground",
        "megaphone, dropped, on the ground",
        "A cone-shaped megaphone, about 48 centimetres long, lying on the ground where it was dropped.",
        "a coach's megaphone left lying on the sideline",
    ],
)

# ---- C: capsules, X-tilt only, translation from capsule_drop_y (2) --------

_c1_y = capsule_drop_y(0.09, 0.5, 90.0)
add(
    "canister-fallen-on-side",
    "computed",
    "A capsule (a rounded canister) tilted a full 90 degrees plus a "
    "15-degree yaw -- capsule_drop_y's own docstring explains why this is "
    "exact for any angle, not just a multiple of 90.",
    [
        primitive(
            "capsule",
            "canister",
            params={"radius": 0.09, "height": 0.5},
            translation=[0.0, _c1_y, 0.0],
            rotation=[90.0, 15.0, 0.0],
        ),
        material([ref("canister")], CANISTER_GREY, "Metal", roughness=0.4, metallic=0.5),
    ],
    [
        "a canister fallen on its side",
        "canister, fallen over",
        "A rounded metal canister, about 68 centimetres long standing, fallen on its side on the ground.",
        "a supply canister that rolled off a shelf and now lies on its side",
    ],
)

_c2_y = capsule_drop_y(0.05, 0.16, 60.0)
add(
    "pill-capsule-resting",
    "computed",
    "A small capsule (a pill) leaning at a partial 60-degree tilt -- "
    "capsule_drop_y's rounded ends make a partial rest angle plausible, "
    "unlike a faceted cylinder or cone.",
    [
        primitive(
            "capsule",
            "pill",
            params={"radius": 0.05, "height": 0.16},
            translation=[0.0, _c2_y, 0.0],
            rotation=[60.0, 0.0, 0.0],
        ),
        material([ref("pill")], PILL_WHITE, "Coating", roughness=0.3),
    ],
    [
        "a pill capsule resting on a tray",
        "pill capsule, tilted, resting",
        "A single oversized pill capsule, about 26 centimetres long, resting at a slight tilt on a flat tray.",
        "a large capsule-shaped prop resting on a display tray",
    ],
)

# ---- D: upright lathe vessels, translation from lathe_half_height (4) -----

_d1_profile = [
    [0.10, 0.00],
    [0.10, 0.03],
    [0.06, 0.10],
    [0.06, 0.30],
    [0.16, 0.45],
    [0.10, 0.58],
    [0.12, 0.62],
]
_d1_y = lathe_half_height(_d1_profile)
add(
    "tall-vase-upright",
    "computed",
    "An upright (unrotated) lathe vase. lathe's own profile clamp "
    "re-centres the stations' y about the midpoint of their min and max, "
    "so the correct translation is half of the profile's own y range -- "
    "not a guessed 'height/2' from some unrelated size parameter, since "
    "this generator has none.",
    [
        primitive(
            "lathe",
            "vase",
            params={"profile": _d1_profile},
            translation=[0.0, _d1_y, 0.0],
        ),
        material([ref("vase")], GLAZE_BLUE, "Glaze", roughness=0.25),
    ],
    [
        "a tall vase standing on a table",
        "tall vase, standing",
        "A tall glazed vase, about 62 centimetres high, standing on a side table, narrow neck and a rounded belly.",
        "a blue-glazed vase standing alone on a windowsill",
    ],
)

_d2_profile = [
    [0.00, 0.00],
    [0.16, 0.05],
    [0.04, 0.18],
    [0.04, 0.28],
    [0.24, 0.40],
    [0.12, 0.50],
    [0.14, 0.54],
]
_d2_y = lathe_half_height(_d2_profile)
add(
    "stemmed-goblet-upright",
    "computed",
    "An upright lathe goblet with a pointed foot (radius 0 at the bottom "
    "station, a pole) -- the same half-range translation as tall-vase-"
    "upright, over a different profile shape.",
    [
        primitive(
            "lathe",
            "goblet",
            params={"profile": _d2_profile},
            translation=[0.0, _d2_y, 0.0],
        ),
        material([ref("goblet")], BRASS, "Brass", roughness=0.3, metallic=0.6),
    ],
    [
        "a stemmed goblet standing on a shelf",
        "goblet, standing",
        "A brass stemmed goblet, about 54 centimetres tall, standing upright on a shelf.",
        "an ornate goblet displayed upright in a trophy case",
    ],
)

_d3_profile = [
    [0.09, 0.00],
    [0.09, 0.35],
    [0.03, 0.42],
    [0.03, 0.55],
    [0.045, 0.58],
]
_d3_y = lathe_half_height(_d3_profile)
add(
    "slender-bottle-upright",
    "computed",
    "An upright lathe bottle, narrow neck flaring slightly at the rim.",
    [
        primitive(
            "lathe",
            "bottle",
            params={"profile": _d3_profile},
            translation=[0.0, _d3_y, 0.0],
        ),
        material([ref("bottle")], GLASS_TEAL, "Glass", roughness=0.1),
    ],
    [
        "a slender bottle standing on a counter",
        "bottle, slender, standing",
        "A slender teal glass bottle, about 58 centimetres tall, standing upright on a counter.",
        "a tall glass bottle left standing by a sink",
    ],
)

_d4_profile = [
    [0.05, 0.00],
    [0.20, 0.08],
    [0.24, 0.30],
    [0.14, 0.46],
    [0.10, 0.50],
    [0.00, 0.58],
]
_d4_y = lathe_half_height(_d4_profile)
add(
    "decorative-urn-upright",
    "computed",
    "An upright lathe urn with a pointed lid finial (radius 0 at the top "
    "station, a pole) as well as a narrow foot -- an asymmetric silhouette "
    "the naive 'half of some height parameter' guess has no parameter to "
    "read at all.",
    [
        primitive(
            "lathe",
            "urn",
            params={"profile": _d4_profile},
            translation=[0.0, _d4_y, 0.0],
        ),
        material([ref("urn")], CLAY_TERRACOTTA, "Terracotta", roughness=0.7),
    ],
    [
        "a decorative urn standing in a garden",
        "urn, decorative, standing",
        "A wide terracotta urn with a pointed lid, about 58 centimetres tall, standing in a garden corner.",
        "an ornamental urn standing beside a garden path",
    ],
)

# ---- E: multi-part hanging assemblies, plain arithmetic (5) ---------------
# drop-to-ground is per object (see the module docstring), so every part's
# translation here is placed by hand: a support resting normally
# (half its own height) and a hanging part positioned a small, explicit gap
# below its support's own underside -- never selected together with it for
# a drop that would separate them.

_e1_post_h = 1.8
_e1_post_y = r4(_e1_post_h / 2.0)
_e1_arm_y = 1.75
_e1_arm_half_h = 0.03
_e1_arm_bottom = r4(_e1_arm_y - _e1_arm_half_h)
_e1_gap = 0.02
_e1_sign_half_h = 0.175
_e1_sign_y = r4(_e1_arm_bottom - _e1_gap - _e1_sign_half_h)
add(
    "hanging-sign-on-arm",
    "computed",
    "A post, a horizontal arm near its top, and a sign panel hanging "
    "below the arm's outer end. The sign's own y is the arm's underside "
    "minus a small gap minus the sign's own half-height -- it must never "
    "be selected together with the post/arm for a drop-to-ground, which "
    "(per object) would instead rest the sign's own box on y=0 and destroy "
    "the 'hangs below the arm' relationship this record exists to teach.",
    [
        primitive(
            "cylinder",
            "post",
            params={"radius": 0.05, "height": _e1_post_h},
            translation=[0.0, _e1_post_y, 0.0],
        ),
        primitive(
            "box",
            "arm",
            params={"size": [0.06, 0.06, 0.5]},
            translation=[0.0, _e1_arm_y, 0.25],
        ),
        primitive(
            "box",
            "sign",
            params={"size": [0.5, 0.35, 0.03]},
            translation=[0.0, _e1_sign_y, 0.5],
        ),
        material([ref("post"), ref("arm")], WOOD_WEATHERED, "Weathered wood", roughness=0.8),
        material([ref("sign")], SIGN_CREAM, "Sign paint", roughness=0.5),
    ],
    [
        "a sign hanging from a bracket arm on a post",
        "hanging sign, post and arm",
        "A wooden post with a short horizontal arm near the top, a sign panel about 50 centimetres wide hanging below its outer end.",
        "a shop sign hanging from an arm mounted on a post",
    ],
)

_e2_post_h = 1.6
_e2_post_y = r4(_e2_post_h / 2.0)
_e2_peg_y = 1.55
_e2_peg_half_h = 0.015
_e2_peg_bottom = r4(_e2_peg_y - _e2_peg_half_h)
_e2_gap = 0.02
_e2_lantern_half_h = 0.10
_e2_lantern_y = r4(_e2_peg_bottom - _e2_gap - _e2_lantern_half_h)
add(
    "hanging-lantern-on-hook",
    "computed",
    "A post with a short peg near the top standing in for a hook, and a "
    "boxy lantern housing hanging beneath the peg's outer end -- same "
    "'gap below the support's own underside' arithmetic as "
    "hanging-sign-on-arm.",
    [
        primitive(
            "cylinder",
            "post",
            params={"radius": 0.04, "height": _e2_post_h},
            translation=[0.0, _e2_post_y, 0.0],
        ),
        primitive(
            "box",
            "peg",
            params={"size": [0.03, 0.03, 0.20]},
            translation=[0.0, _e2_peg_y, 0.10],
        ),
        primitive(
            "box",
            "lantern",
            params={"size": [0.14, 0.20, 0.14]},
            translation=[0.0, _e2_lantern_y, 0.20],
        ),
        material([ref("post"), ref("peg")], IRON_DARK, "Iron", roughness=0.5, metallic=0.4),
        material([ref("lantern")], BRASS, "Lantern housing", roughness=0.35, metallic=0.5),
    ],
    [
        "a lantern hanging from a hook on a post",
        "hanging lantern, post and hook",
        "A dark iron post with a short hook near the top, a boxy brass lantern about 20 centimetres tall hanging beneath it.",
        "a lantern hanging from a hook beside a garden path",
    ],
)

_e3_post_h = 1.2
_e3_post_y = r4(_e3_post_h / 2.0)
_e3_crossbar_y = r4(_e3_post_h - 0.025)
_e3_crossbar_bottom = r4(_e3_crossbar_y - 0.025)
_e3_gap = 0.03
_e3_bell_r = 0.12
_e3_bell_h = 0.30
_e3_bell_half_h = r4(_e3_bell_h / 2.0)
_e3_bell_y = r4(_e3_crossbar_bottom - _e3_gap - _e3_bell_half_h)
add(
    "hanging-bell-under-frame",
    "computed",
    "Two side posts and a crossbar (a simple frame), with a bell -- an "
    "unrotated cone, its narrow apex already at the top where the default "
    "generator puts it -- hanging beneath the crossbar's own underside. "
    "No rotation is needed at all: the cone's own default orientation "
    "(apex up, wide base down) already reads as a bell's attachment point "
    "and mouth.",
    [
        primitive(
            "cylinder",
            "post_l",
            params={"radius": 0.045, "height": _e3_post_h},
            translation=[-0.35, _e3_post_y, 0.0],
        ),
        primitive(
            "cylinder",
            "post_r",
            params={"radius": 0.045, "height": _e3_post_h},
            translation=[0.35, _e3_post_y, 0.0],
        ),
        primitive(
            "box",
            "crossbar",
            params={"size": [0.76, 0.05, 0.05]},
            translation=[0.0, _e3_crossbar_y, 0.0],
        ),
        primitive(
            "cone",
            "bell",
            params={"radius": _e3_bell_r, "height": _e3_bell_h},
            translation=[0.0, _e3_bell_y, 0.0],
        ),
        material(
            [ref("post_l"), ref("post_r"), ref("crossbar")], POST_GREY, "Frame", roughness=0.6
        ),
        material([ref("bell")], BRASS, "Bell bronze", roughness=0.3, metallic=0.7),
    ],
    [
        "a bell hanging under a frame",
        "hanging bell, frame",
        "A simple two-post frame about 1.2 metres tall with a crossbar, a bronze bell hanging beneath the middle of the crossbar.",
        "a bell suspended from a small frame in a courtyard",
    ],
)

_e4_post_h = 1.5
_e4_post_y = r4(_e4_post_h / 2.0)
_e4_arm_y = 1.45
_e4_arm_half_h = 0.025
_e4_arm_bottom = r4(_e4_arm_y - _e4_arm_half_h)
_e4_gap = 0.02
_e4_planter_half_h = 0.09
_e4_planter_y = r4(_e4_arm_bottom - _e4_gap - _e4_planter_half_h)
add(
    "hanging-planter-on-arm",
    "computed",
    "Post, arm, and a boxy planter hanging below the arm's outer end -- "
    "the same arithmetic as hanging-sign-on-arm again, a third time over.",
    [
        primitive(
            "cylinder",
            "post",
            params={"radius": 0.05, "height": _e4_post_h},
            translation=[0.0, _e4_post_y, 0.0],
        ),
        primitive(
            "box",
            "arm",
            params={"size": [0.05, 0.05, 0.4]},
            translation=[0.0, _e4_arm_y, 0.2],
        ),
        primitive(
            "box",
            "planter",
            params={"size": [0.24, 0.18, 0.24]},
            translation=[0.0, _e4_planter_y, 0.4],
        ),
        material([ref("post"), ref("arm")], IRON_DARK, "Iron", roughness=0.5, metallic=0.3),
        material([ref("planter")], GLAZE_GREEN, "Glazed pot", roughness=0.4),
    ],
    [
        "a planter hanging from an arm on a post",
        "hanging planter, post and arm",
        "A dark metal post with an arm near the top, a glazed planter box about 24 centimetres across hanging from its outer end.",
        "a hanging planter suspended from an arm beside a doorway",
    ],
)

_e5_post_h = 1.3
_e5_post_y = r4(_e5_post_h / 2.0)
_e5_crossbar_y = r4(_e5_post_h - 0.0225)
_e5_crossbar_bottom = r4(_e5_crossbar_y - 0.0225)
_e5_gap = 0.03
_e5_chime_half_h = 0.175
_e5_chime_y = r4(_e5_crossbar_bottom - _e5_gap - _e5_chime_half_h)
add(
    "hanging-wind-chime-under-crossbar",
    "computed",
    "Frame like hanging-bell-under-frame, but the hanging part is a "
    "vertical cylinder tube (a wind chime) rather than a cone.",
    [
        primitive(
            "cylinder",
            "post_l",
            params={"radius": 0.04, "height": _e5_post_h},
            translation=[-0.3, _e5_post_y, 0.0],
        ),
        primitive(
            "cylinder",
            "post_r",
            params={"radius": 0.04, "height": _e5_post_h},
            translation=[0.3, _e5_post_y, 0.0],
        ),
        primitive(
            "box",
            "crossbar",
            params={"size": [0.64, 0.045, 0.045]},
            translation=[0.0, _e5_crossbar_y, 0.0],
        ),
        primitive(
            "cylinder",
            "chime",
            params={"radius": 0.02, "height": 0.35},
            translation=[0.0, _e5_chime_y, 0.0],
        ),
        material(
            [ref("post_l"), ref("post_r"), ref("crossbar")], POST_GREY, "Frame", roughness=0.6
        ),
        material([ref("chime")], CHIME_SILVER, "Chime metal", roughness=0.25, metallic=0.7),
    ],
    [
        "a wind chime hanging under a crossbar",
        "hanging wind chime, crossbar",
        "A small two-post frame with a crossbar, a single silver chime tube about 35 centimetres long hanging beneath the middle.",
        "a wind chime suspended from a crossbar on a porch",
    ],
)

assert len([r for r in recipes if r[1] == "computed"]) == 19

# =============================================================================
# SAFETY-NET idiom (19 bases): every part is placed at a plausible-but-not-
# necessarily-correct height, then the batch ends with clay_select on the
# placed part(s) and clay_op drop-to-ground -- see clay_ops._drop_to_ground's
# own docstring (quoted in the module docstring above) for why this is only
# ever used on objects whose correct resting place really is the ground
# itself, never on a hanging part alongside its support.
# =============================================================================

# ---- F: single cylinders leaning at a partial, possibly compound tilt (4) --

add(
    "leaning-fence-post",
    "safety-net",
    "A fence post tilted 20 degrees about X with a 10-degree yaw, placed "
    "at the naive unrotated half-height (0.7) -- exactly the bug the plan "
    "describes: 'a post leaning at an angle... placed at a guessed y'.",
    [
        primitive(
            "cylinder",
            "post",
            params={"radius": 0.06, "height": 1.4},
            translation=[0.0, 0.7, 0.0],
            rotation=[20.0, 10.0, 0.0],
        ),
        material([ref("post")], WOOD_WEATHERED, "Weathered wood", roughness=0.8),
        *drop_to_ground(ref("post")),
    ],
    [
        "a fence post leaning at an angle, resting on the ground",
        "leaning fence post, on the ground",
        "A weathered wooden fence post, about 1.4 metres long, leaning at an angle and resting on the ground.",
        "a fence post that has come loose and now leans, its base still resting on the ground",
    ],
)

add(
    "tilted-ladder-rail",
    "safety-net",
    "A thin cylinder (a loose ladder rail) tilted 15 degrees about X with "
    "a 90-degree yaw, placed at a naive guessed height.",
    [
        primitive(
            "cylinder",
            "rail",
            params={"radius": 0.03, "height": 1.6},
            translation=[0.0, 0.8, 0.0],
            rotation=[15.0, 90.0, 0.0],
        ),
        material([ref("rail")], WOOD_PALE, "Wood", roughness=0.6),
        *drop_to_ground(ref("rail")),
    ],
    [
        "a ladder rail tilted slightly, lying on the floor",
        "tilted ladder rail, on the floor",
        "A single wooden ladder rail, about 1.6 metres long, tilted slightly and lying on the floor of a workshop.",
        "a loose ladder rail that slid off a rack and now lies tilted on the floor",
    ],
)

add(
    "pipe-resting-against-wall",
    "safety-net",
    "A cylinder (a length of pipe) tilted 35 degrees about X, placed at a naive guessed height.",
    [
        primitive(
            "cylinder",
            "pipe",
            params={"radius": 0.05, "height": 1.0},
            translation=[0.0, 0.5, 0.0],
            rotation=[35.0, 0.0, 0.0],
        ),
        material([ref("pipe")], STEEL, "Steel", roughness=0.4, metallic=0.6),
        *drop_to_ground(ref("pipe")),
    ],
    [
        "a length of pipe resting against a wall, its foot on the floor",
        "pipe, angled, resting on the floor",
        "A metal pipe about a metre long, angled up against a wall with its lower end resting on the floor.",
        "a spare length of pipe left leaning with its base on the floor",
    ],
)

add(
    "signpost-leaning",
    "safety-net",
    "A thin cylinder (a signpost) tilted 25 degrees about X with a "
    "-40-degree yaw, placed at a naive guessed height.",
    [
        primitive(
            "cylinder",
            "signpost",
            params={"radius": 0.04, "height": 1.2},
            translation=[0.0, 0.6, 0.0],
            rotation=[25.0, -40.0, 0.0],
        ),
        material([ref("signpost")], WOOD_WEATHERED, "Weathered wood", roughness=0.8),
        *drop_to_ground(ref("signpost")),
    ],
    [
        "a signpost leaning over, its base resting on the ground",
        "leaning signpost, on the ground",
        "A wooden signpost, about 1.2 metres tall standing straight, now leaning over with its base still resting on the ground.",
        "an old signpost that has begun to lean, base resting on the ground",
    ],
)

# ---- G: fallen/toppled lathe vessels (4) ----------------------------------

_g1_profile = [
    [0.09, 0.00],
    [0.09, 0.03],
    [0.05, 0.09],
    [0.05, 0.27],
    [0.15, 0.40],
    [0.09, 0.52],
    [0.11, 0.56],
]
_g1_naive_y = lathe_half_height(_g1_profile)
add(
    "fallen-vase-on-floor",
    "safety-net",
    "A lathe vase rotated 90 degrees about X with a 30-degree yaw, placed "
    "at the upright-half-height guess (lathe_half_height applied to its "
    "own profile) -- correct only if the vase were still standing, which "
    "it is not.",
    [
        primitive(
            "lathe",
            "vase",
            params={"profile": _g1_profile},
            translation=[0.0, _g1_naive_y, 0.0],
            rotation=[90.0, 30.0, 0.0],
        ),
        material([ref("vase")], GLAZE_BLUE, "Glaze", roughness=0.25),
        *drop_to_ground(ref("vase")),
    ],
    [
        "a vase fallen on its side on the floor",
        "fallen vase, on the floor",
        "A glazed vase, about 56 centimetres tall standing, fallen over and lying on its side on the floor.",
        "a vase knocked off a shelf, now lying on the floor",
    ],
)

_g2_profile = [
    [0.00, 0.00],
    [0.14, 0.06],
    [0.05, 0.16],
    [0.05, 0.30],
    [0.20, 0.38],
    [0.10, 0.46],
    [0.12, 0.49],
]
_g2_naive_y = lathe_half_height(_g2_profile)
add(
    "tipped-over-jug",
    "safety-net",
    "A lathe jug rotated 80 degrees about X, placed at the naive upright half-height guess.",
    [
        primitive(
            "lathe",
            "jug",
            params={"profile": _g2_profile},
            translation=[0.0, _g2_naive_y, 0.0],
            rotation=[80.0, 0.0, 0.0],
        ),
        material([ref("jug")], CLAY_TERRACOTTA, "Terracotta", roughness=0.6),
        *drop_to_ground(ref("jug")),
    ],
    [
        "a jug tipped over on the ground",
        "tipped jug, on the ground",
        "A terracotta jug, about 49 centimetres tall standing, tipped over and lying on the ground.",
        "a garden jug knocked over, now lying on the ground",
    ],
)

_g3_profile = [
    [0.04, 0.00],
    [0.22, 0.09],
    [0.26, 0.32],
    [0.15, 0.48],
    [0.11, 0.52],
    [0.00, 0.60],
]
_g3_naive_y = lathe_half_height(_g3_profile)
add(
    "toppled-urn",
    "safety-net",
    "A lathe urn (pointed lid finial, radius 0 at the top station) rotated "
    "90 degrees about X with a -45-degree yaw, placed at the naive "
    "upright half-height guess.",
    [
        primitive(
            "lathe",
            "urn",
            params={"profile": _g3_profile},
            translation=[0.0, _g3_naive_y, 0.0],
            rotation=[90.0, -45.0, 0.0],
        ),
        material([ref("urn")], CLAY_TERRACOTTA, "Terracotta", roughness=0.7),
        *drop_to_ground(ref("urn")),
    ],
    [
        "an urn toppled over on the ground",
        "toppled urn, on the ground",
        "A wide terracotta urn, about 60 centimetres tall standing, toppled over and lying on the ground.",
        "a garden urn blown over in a storm, now lying on the ground",
    ],
)

_g4_profile = [
    [0.08, 0.00],
    [0.08, 0.33],
    [0.025, 0.40],
    [0.025, 0.52],
    [0.04, 0.55],
]
_g4_naive_y = lathe_half_height(_g4_profile)
add(
    "fallen-bottle-on-floor",
    "safety-net",
    "A lathe bottle rotated 90 degrees about X with a 60-degree yaw, "
    "placed at the naive upright half-height guess.",
    [
        primitive(
            "lathe",
            "bottle",
            params={"profile": _g4_profile},
            translation=[0.0, _g4_naive_y, 0.0],
            rotation=[90.0, 60.0, 0.0],
        ),
        material([ref("bottle")], GLASS_TEAL, "Glass", roughness=0.1),
        *drop_to_ground(ref("bottle")),
    ],
    [
        "a bottle fallen on the floor",
        "fallen bottle, on the floor",
        "A slender glass bottle, about 55 centimetres tall standing, fallen and lying on the floor.",
        "an empty bottle rolled onto its side on the floor",
    ],
)

# ---- H: tube (bent pipe / hook / coiled hose), any path shape (4) ---------

add(
    "garden-hose-loop",
    "safety-net",
    "A tube whose path loops loosely in the XZ plane (all path points at "
    "the same y, per snake-loose-coil's own technique in "
    "_gen_creatures.py) -- placed at a plausible guess and dropped rather "
    "than reasoned about, since tube's parallel-transported frames "
    "(primitives.py's own _tube_frames) are not worth re-deriving by hand "
    "here.",
    [
        primitive(
            "tube",
            "hose",
            params={
                "path": [
                    [0.30, 0.0, 0.0],
                    [0.20, 0.0, 0.22],
                    [-0.02, 0.0, 0.30],
                    [-0.24, 0.0, 0.18],
                    [-0.28, 0.0, -0.06],
                    [-0.10, 0.0, -0.22],
                    [0.12, 0.0, -0.14],
                ],
                "radius": 0.025,
                "sides": 10,
            },
            translation=[0.0, 0.025, 0.0],
        ),
        material([ref("hose")], HOSE_GREEN, "Rubber", roughness=0.6),
        *drop_to_ground(ref("hose")),
    ],
    [
        "a garden hose lying coiled on the ground",
        "garden hose, coiled, on the ground",
        "A green garden hose, loosely coiled, lying on the ground near a tap.",
        "a length of hose left in a loose loop on the lawn",
    ],
)

add(
    "shepherds-hook-standing",
    "safety-net",
    "A tube whose path rises from the ground and curls over at the top -- "
    "a shepherd's hook (a plant hanger). A guessed mid-height translation, "
    "then dropped.",
    [
        primitive(
            "tube",
            "hook",
            params={
                "path": [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.55, 0.02],
                    [0.0, 0.85, 0.05],
                    [0.05, 1.05, 0.05],
                    [0.15, 1.15, 0.02],
                    [0.22, 1.10, -0.05],
                ],
                "radius": 0.018,
                "sides": 8,
            },
            translation=[0.0, 0.4, 0.0],
        ),
        material([ref("hook")], IRON_DARK, "Iron", roughness=0.4, metallic=0.5),
        *drop_to_ground(ref("hook")),
    ],
    [
        "a shepherd's hook standing in the garden",
        "shepherd's hook, standing",
        "A curled iron shepherd's hook, about 1.15 metres tall, standing upright in a garden bed, its foot on the ground.",
        "a plant-hanger hook standing among the flower beds",
    ],
)

add(
    "coiled-rope-on-ground",
    "safety-net",
    "A tube whose path spirals in a single flat loop in the XZ plane -- a "
    "coil of rope. A guessed translation, then dropped.",
    [
        primitive(
            "tube",
            "rope",
            params={
                "path": [
                    [0.18, 0.0, 0.0],
                    [0.13, 0.0, 0.13],
                    [0.0, 0.0, 0.18],
                    [-0.13, 0.0, 0.13],
                    [-0.18, 0.0, 0.0],
                    [-0.13, 0.0, -0.13],
                    [0.0, 0.0, -0.18],
                ],
                "radius": 0.02,
                "sides": 8,
            },
            translation=[0.0, 0.05, 0.0],
        ),
        material([ref("rope")], ROPE_TAN, "Rope fibre", roughness=0.9),
        *drop_to_ground(ref("rope")),
    ],
    [
        "a coil of rope lying on the ground",
        "coiled rope, on the ground",
        "A tan coil of rope, about 36 centimetres across, lying in a loop on the ground.",
        "a length of rope coiled up and left on the dock",
    ],
)

add(
    "bent-exhaust-pipe",
    "safety-net",
    "A tube whose path runs mostly flat then bends upward at one end -- a "
    "bent exhaust pipe segment. A guessed translation, then dropped.",
    [
        primitive(
            "tube",
            "exhaust",
            params={
                "path": [
                    [-0.30, 0.0, 0.0],
                    [-0.05, 0.0, 0.0],
                    [0.10, 0.05, 0.0],
                    [0.28, 0.14, 0.0],
                ],
                "radius": 0.035,
                "sides": 10,
            },
            translation=[0.0, 0.15, 0.0],
        ),
        material([ref("exhaust")], LEAD_GREY, "Steel", roughness=0.5, metallic=0.6),
        *drop_to_ground(ref("exhaust")),
    ],
    [
        "a bent exhaust pipe lying on the ground",
        "bent exhaust pipe, on the ground",
        "A bent length of exhaust pipe, about 60 centimetres long, lying on the workshop floor with one end curling upward.",
        "a discarded exhaust pipe segment left lying in a scrapyard",
    ],
)

# ---- I: sweep (custom outline, extruded along Z), any rotation (4) --------

_OUTLINE_STEP = [
    [-0.10, -0.10],
    [0.10, -0.10],
    [0.10, 0.02],
    [0.02, 0.02],
    [0.02, 0.14],
    [-0.10, 0.14],
]
_OUTLINE_NOTCH = [
    [-0.08, -0.14],
    [0.08, -0.14],
    [0.08, 0.0],
    [0.14, 0.0],
    [0.14, 0.10],
    [-0.08, 0.10],
]

add(
    "extruded-post-standing",
    "safety-net",
    "A sweep with a stepped (asymmetric-about-its-own-origin) outline, "
    "rotated 90 degrees about X so the extrusion axis stands vertical -- "
    "sweep's own outline re-centring (primitives._clamp_outline) makes a "
    "guessed height wrong the moment it is rotated. Guessed translation, "
    "then dropped.",
    [
        primitive(
            "sweep",
            "post",
            params={
                "outline": _OUTLINE_STEP,
                "depth": 0.9,
                "taper": 1.0,
                "twist": 0.0,
                "sections": 1,
            },
            translation=[0.0, 0.45, 0.0],
            rotation=[90.0, 0.0, 0.0],
        ),
        material([ref("post")], STEEL, "Steel", roughness=0.4, metallic=0.5),
        *drop_to_ground(ref("post")),
    ],
    [
        "a standing extruded post on the ground",
        "extruded post, standing",
        "A stepped-profile metal post, about 90 centimetres tall, standing upright on the ground.",
        "a custom extruded post standing in a workshop yard",
    ],
)

add(
    "extruded-beam-fallen",
    "safety-net",
    "The same stepped outline extruded longer and rotated 90 degrees "
    "about Z (a fallen beam) -- a compound-looking case handed straight "
    "to the safety net rather than reasoned about.",
    [
        primitive(
            "sweep",
            "beam",
            params={
                "outline": _OUTLINE_STEP,
                "depth": 0.7,
                "taper": 1.0,
                "twist": 0.0,
                "sections": 1,
            },
            translation=[0.0, 0.2, 0.0],
            rotation=[0.0, 0.0, 90.0],
        ),
        material([ref("beam")], STEEL, "Steel", roughness=0.45, metallic=0.4),
        *drop_to_ground(ref("beam")),
    ],
    [
        "a metal beam fallen on the ground",
        "fallen beam, on the ground",
        "A stepped-profile metal beam, about 70 centimetres long, fallen on its side on the ground.",
        "a length of extruded beam left lying on a construction site",
    ],
)

add(
    "extruded-moulding-standing",
    "safety-net",
    "A sweep with the notch outline, rotated 90 degrees about X to stand upright.",
    [
        primitive(
            "sweep",
            "moulding",
            params={
                "outline": _OUTLINE_NOTCH,
                "depth": 0.6,
                "taper": 1.0,
                "twist": 0.0,
                "sections": 1,
            },
            translation=[0.0, 0.3, 0.0],
            rotation=[90.0, 0.0, 0.0],
        ),
        material([ref("moulding")], WOOD_PALE, "Wood", roughness=0.5),
        *drop_to_ground(ref("moulding")),
    ],
    [
        "a standing decorative moulding on the floor",
        "decorative moulding, standing",
        "A notched wooden moulding profile, about 60 centimetres tall, standing upright on the workshop floor.",
        "a length of decorative moulding stood on end against a wall",
    ],
)

add(
    "extruded-plinth-toppled",
    "safety-net",
    "A sweep with the notch outline, rotated 90 degrees about X with a "
    "20-degree yaw, toppled over rather than standing.",
    [
        primitive(
            "sweep",
            "plinth",
            params={
                "outline": _OUTLINE_NOTCH,
                "depth": 0.5,
                "taper": 1.0,
                "twist": 0.0,
                "sections": 1,
            },
            translation=[0.0, 0.3, 0.0],
            rotation=[90.0, 20.0, 0.0],
        ),
        material([ref("plinth")], WOOD_WEATHERED, "Weathered wood", roughness=0.7),
        *drop_to_ground(ref("plinth")),
    ],
    [
        "a small plinth toppled over on the ground",
        "toppled plinth, on the ground",
        "A small notched-profile plinth, about 50 centimetres long, toppled over and lying on the ground.",
        "a display plinth knocked over, now lying on the ground",
    ],
)

# ---- J: capsules at a compound (rx and rz both nonzero) tilt (2) ----------

add(
    "fallen-potion-vial",
    "safety-net",
    "A small capsule at a compound tilt (rx and rz both nonzero, unlike "
    "every 'computed' capsule above) -- exactly the case the module "
    "docstring's 'Why X-tilt + Y-yaw' section says the closed form no "
    "longer covers, handed to the safety net instead.",
    [
        primitive(
            "capsule",
            "vial",
            params={"radius": 0.03, "height": 0.10},
            translation=[0.0, 0.08, 0.0],
            rotation=[35.0, 0.0, 50.0],
        ),
        material([ref("vial")], GLASS_TEAL, "Glass", roughness=0.15),
        *drop_to_ground(ref("vial")),
    ],
    [
        "a potion vial fallen on its side on a table",
        "fallen vial, on a table",
        "A small glass vial, about 16 centimetres long, fallen on its side on a table top.",
        "a spilled potion vial lying on its side on a workbench",
    ],
)

add(
    "dropped-pill-prop",
    "safety-net",
    "A small capsule at a different compound tilt.",
    [
        primitive(
            "capsule",
            "pill",
            params={"radius": 0.02, "height": 0.05},
            translation=[0.0, 0.05, 0.0],
            rotation=[70.0, 0.0, 25.0],
        ),
        material([ref("pill")], PILL_WHITE, "Coating", roughness=0.3),
        *drop_to_ground(ref("pill")),
    ],
    [
        "a dropped pill lying on the floor",
        "dropped pill, on the floor",
        "A single oversized pill prop, about 9 centimetres long, dropped and lying on the floor.",
        "a large pill-shaped prop that rolled onto the floor",
    ],
)

# ---- K: one more single, deliberately wrong naive guess (1) ---------------

add(
    "tipped-drum-barrel",
    "safety-net",
    "A wide cylinder (a drum/barrel) tilted 90 degrees with a 15-degree "
    "yaw, placed at a deliberately wrong naive guess (half the radius, "
    "rather than the radius itself) -- the safety net corrects it exactly "
    "regardless of how wrong the guess was.",
    [
        primitive(
            "cylinder",
            "barrel",
            params={"radius": 0.28, "height": 0.6},
            translation=[0.0, 0.14, 0.0],
            rotation=[90.0, 15.0, 0.0],
        ),
        material([ref("barrel")], IRON_DARK, "Steel", roughness=0.5, metallic=0.4),
        *drop_to_ground(ref("barrel")),
    ],
    [
        "a barrel tipped over on the ground",
        "tipped barrel, on the ground",
        "A wide steel drum, about 56 centimetres across, tipped onto its side and resting on the ground.",
        "an old barrel knocked over, now lying on the ground",
    ],
)

assert len([r for r in recipes if r[1] == "safety-net"]) == 19
assert len(recipes) == 38

# The seed list and the recipes above must never drift: seeds/grounding.txt's
# own line order is this module's recipe order, and its prompt half of every
# `class | prompt` line is that base's first phrasing (`prompts[0]`) -- a
# module-level check, so an edit to either file that breaks the pairing fails
# the moment this module is imported (by `main()` below, or by a future
# test), not only when someone happens to compare them by eye.
_seed_entries = read_seed_lines(SEED_PATH)
assert len(_seed_entries) == len(recipes), (
    f"{SEED_PATH} has {len(_seed_entries)} lines, but there are {len(recipes)} recipes"
)
for _seed_index, (
    (_seed_cls, _seed_prompt),
    (_slug, _idiom, _notes, _calls, _prompts),
) in enumerate(zip(_seed_entries, recipes, strict=True)):
    assert _seed_cls in ("easy", "medium", "hard"), (
        f"{SEED_PATH}:{_seed_index + 1}: {_slug!r} has unknown class {_seed_cls!r}"
    )
    assert _seed_prompt == _prompts[0], (
        f"{SEED_PATH}:{_seed_index + 1}: seed prompt {_seed_prompt!r} does not match "
        f"{_slug!r}'s first phrasing {_prompts[0]!r}"
    )


def main() -> None:
    records: list[dict[str, Any]] = []
    counter = 1
    for slug, idiom, notes, calls, prompts in recipes:
        for prompt in prompts:
            rec: dict[str, Any] = {
                "id": f"grounding-{counter:04d}",
                "family": "grounding",
                "kind": "build",
                "prompt": prompt,
                "calls": calls,
                "notes": f"[{slug}/{idiom}] {notes}",
            }
            records.append(rec)
            counter += 1

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"wrote {len(records)} records ({len(recipes)} base builds) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
