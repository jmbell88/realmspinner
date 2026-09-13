"""Deterministic generator for ``drafts/vehicles.jsonl``.

Emits build records for the vehicles-and-tools family: carts/wagons, boats,
hovercraft, rockets, tanks, bikes, and the hand-tool/weapon set (hammers,
axes, swords, shields, spears, staffs, pickaxes, wrenches). Every base build
below is hand-authored (dimensions, material choices and a ``notes`` reason);
this script's only job is to fan each one out into 3-5 prompt phrasings with
small, deterministic dimension/material jitter, and to write the result
sorted by id.

Run:
    uv run python training/clay-assistant/drafts/_gen_vehicles.py

Then verify:
    uv run python training/clay-assistant/gen/build.py --only vehicles \
        --out training/clay-assistant/drafts/_out_vehicles
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

FAMILY = "vehicles"
OUT_PATH = Path(__file__).resolve().parent / f"{FAMILY}.jsonl"

# ---------------------------------------------------------------------------
# small helpers -- every build function below returns a plain list of tool
# call dicts ({"name": ..., "arguments": {...}}), the shape ``drafts/*.jsonl``
# stores under "calls" (folded into one clay_batch by the verifier/trainer).
# ---------------------------------------------------------------------------


def prim(
    generator: str,
    name: str,
    params: dict[str, Any] | None = None,
    translation: list[float] | None = None,
    rotation: list[float] | None = None,
    scale: list[float] | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {"generator": generator, "name": name}
    if params:
        args["params"] = params
    if translation:
        args["translation"] = translation
    if rotation:
        args["rotation"] = rotation
    if scale:
        args["scale"] = scale
    return {"name": "clay_add_primitive", "arguments": args}


def select(refs: list[str]) -> dict[str, Any]:
    return {"name": "clay_select", "arguments": {"uids": [{"$ref": r} for r in refs]}}


def op(name: str, params: dict[str, float] | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {"name": name}
    if params:
        args["params"] = params
    return {"name": "clay_op", "arguments": args}


def mirror_copy(ref: str, axis: int, offset: float = 0.0) -> list[dict[str, Any]]:
    """Select *ref* and mirror-copy it across the world plane at *offset* on
    *axis* (0=X, 1=Y, 2=Z) -- the paired-wheel / paired-blade technique."""
    return [select([ref]), op("mirror-copy", {"axis": float(axis), "offset": float(offset)})]


def array_radial(ref: str, count: int, axis: int = 1, angle: float = 360.0) -> list[dict[str, Any]]:
    """Select *ref* and spin copies of it around the world axis -- correct
    for a spoke/stud/vent placed at radius from an object standing at
    x=0, z=0 (array-radial spins around the world origin, not the object's
    own centre)."""
    return [
        select([ref]),
        op("array-radial", {"count": float(count), "angle": float(angle), "axis": float(axis)}),
    ]


def radial_names(base: str, count: int) -> list[str]:
    """Every name an ``array_radial(base, count, ...)`` leaves behind: the
    original plus its ``.NNN``-suffixed copies -- $ref resolves against the
    document's own object names when each entry runs, so a later
    clay_material call can address every copy this way, but only if it
    lists them all. A material call naming just the original leaves every
    array/mirror copy on the document's default material, which is exactly
    the defect this helper exists to avoid repeating."""
    return [base] + [f"{base}.{i:03d}" for i in range(1, count)]


def material(
    refs: list[str], name: str, color: list[float], roughness: float = 0.6, metallic: float = 0.0
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "uids": [{"$ref": r} for r in refs],
        "name": name,
        "color": color,
        "roughness": roughness,
    }
    if metallic:
        args["metallic"] = metallic
    return {"name": "clay_material", "arguments": args}


WOOD = [0.45, 0.31, 0.18]
WOOD_LIGHT = [0.62, 0.47, 0.30]
WOOD_DARK = [0.30, 0.20, 0.11]
IRON = [0.35, 0.35, 0.38]
STEEL = [0.55, 0.57, 0.6]
BRASS = [0.65, 0.52, 0.22]
CANVAS = [0.72, 0.68, 0.55]
OLIVE = [0.32, 0.36, 0.24]
RED = [0.55, 0.12, 0.1]
LEATHER = [0.4, 0.24, 0.14]


def jitter(rng: random.Random, base: float, spread: float = 0.08) -> float:
    return round(base * (1.0 + rng.uniform(-spread, spread)), 4)


def jc(rng: random.Random, color: list[float], spread: float = 0.05) -> list[float]:
    return [round(min(1.0, max(0.0, c + rng.uniform(-spread, spread))), 3) for c in color]


# ---------------------------------------------------------------------------
# base builds -- one function per seed line, keyed by seed class for the
# report. Each returns (calls, notes, allow_below_ground).
# ---------------------------------------------------------------------------

Builder = Any  # (rng) -> tuple[list[dict], str, bool]

BASES: dict[str, dict[str, Any]] = {}


def register(key: str, seed_class: str, prompts: list[str], fn: Builder) -> None:
    BASES[key] = {"class": seed_class, "prompts": prompts, "fn": fn}


# --- carts & wagons ----------------------------------------------------------


def _wheel_pair(rng, x, y, z, radius, width, name="wheel_r"):
    calls = [
        prim(
            "cylinder",
            name,
            {"radius": radius, "height": width, "segments": 16},
            translation=[x, y, z],
            rotation=[0, 0, 90],
        )
    ]
    calls += mirror_copy(name, axis=0, offset=0.0)
    return calls


def build_handcart(rng):
    r = jitter(rng, 0.28)
    bed_l, bed_w, bed_h = jitter(rng, 0.9), jitter(rng, 0.55), jitter(rng, 0.16)
    bed_y = r * 1.7
    calls = [prim("box", "bed", {"size": [bed_l, bed_h, bed_w]}, translation=[0, bed_y, 0])]
    calls += _wheel_pair(rng, bed_l * 0.5 + 0.05, r, 0.0, r, 0.08)
    handle_len = jitter(rng, 0.6)
    calls.append(
        prim(
            "cylinder",
            "handle_l",
            {"radius": 0.02, "height": handle_len, "segments": 8},
            translation=[bed_w * 0.35, bed_y, -(bed_l / 2 + handle_len / 2)],
            rotation=[90, 0, 0],
        )
    )
    calls += mirror_copy("handle_l", axis=0, offset=0.0)
    calls.append(material(["bed", "handle_l", "handle_l.001"], "Cart wood", jc(rng, WOOD)))
    calls.append(
        material(["wheel_r", "wheel_r.001"], "Iron rim", jc(rng, IRON), roughness=0.5, metallic=0.6)
    )
    notes = (
        "A two-wheeled handcart: bed height set to roughly 1.7x wheel radius so the axle "
        "clears the ground; wheels are a mirror-copy pair (the family's paired-wheel "
        "technique) with the axle along X; two handle rails mirrored to the rear."
    )
    return calls, notes, False


def build_wheelbarrow(rng):
    tray_l, tray_w, tray_h = jitter(rng, 0.7), jitter(rng, 0.5), jitter(rng, 0.28)
    r = jitter(rng, 0.18)
    leg_h = jitter(rng, 0.35)
    tray_y = leg_h
    calls = [prim("box", "tray", {"size": [tray_l, tray_h, tray_w]}, translation=[0, tray_y, 0])]
    calls.append(
        prim(
            "cylinder",
            "leg_r",
            {"radius": 0.02, "height": leg_h, "segments": 8},
            translation=[tray_w * 0.35, leg_h / 2, tray_l * 0.4],
        )
    )
    calls += mirror_copy("leg_r", axis=0, offset=0.0)
    calls.append(
        prim(
            "cylinder",
            "wheel",
            {"radius": r, "height": 0.09, "segments": 16},
            translation=[0, r, -(tray_l / 2 + r * 0.6)],
            rotation=[0, 0, 90],
        )
    )
    handle_len = jitter(rng, 0.55)
    calls.append(
        prim(
            "cylinder",
            "handle_r",
            {"radius": 0.02, "height": handle_len, "segments": 8},
            translation=[tray_w * 0.35, tray_y, tray_l / 2 + handle_len / 2],
            rotation=[90, 0, 0],
        )
    )
    calls += mirror_copy("handle_r", axis=0, offset=0.0)
    calls.append(material(["tray", "handle_r", "handle_r.001"], "Barrow wood", jc(rng, WOOD)))
    calls.append(
        material(
            ["leg_r", "leg_r.001", "wheel"],
            "Iron fittings",
            jc(rng, IRON),
            roughness=0.5,
            metallic=0.6,
        )
    )
    notes = (
        "A single-wheel wheelbarrow: one front wheel takes the whole load axis, two rear "
        "legs (mirror-copy pair) hold the tray level, tray height equals the leg height so "
        "the whole thing reads as balanced on its wheel and legs."
    )
    return calls, notes, False


def build_cart4(rng):
    r = jitter(rng, 0.22)
    bed_l, bed_w, bed_h = jitter(rng, 1.1), jitter(rng, 0.7), jitter(rng, 0.18)
    bed_y = r * 1.8
    calls = [prim("box", "bed", {"size": [bed_l, bed_h, bed_w]}, translation=[0, bed_y, 0])]
    calls.append(
        prim(
            "cylinder",
            "wheel_fr",
            {"radius": r, "height": 0.09, "segments": 16},
            translation=[bed_w / 2 + 0.02, r, bed_l * 0.32],
            rotation=[0, 0, 90],
        )
    )
    calls += mirror_copy("wheel_fr", axis=0, offset=0.0)
    calls.append(
        prim(
            "cylinder",
            "wheel_br",
            {"radius": r, "height": 0.09, "segments": 16},
            translation=[bed_w / 2 + 0.02, r, -bed_l * 0.32],
            rotation=[0, 0, 90],
        )
    )
    calls += mirror_copy("wheel_br", axis=0, offset=0.0)
    calls.append(material(["bed"], "Cart wood", jc(rng, WOOD)))
    calls.append(
        material(
            ["wheel_fr", "wheel_fr.001", "wheel_br", "wheel_br.001"],
            "Iron rim",
            jc(rng, IRON),
            roughness=0.5,
            metallic=0.6,
        )
    )
    notes = (
        "Four-wheel cart: front and back wheel pairs are each their own mirror-copy, so the "
        "batch never needs to address an array-generated uid by anything but its own name."
    )
    return calls, notes, False


def build_wagon(rng):
    calls, _, _ = build_cart4(rng)
    # add rails and a bench so it reads as a wagon rather than a plain cart.
    bed_l, bed_w = 1.2, 0.75
    rail_h = 0.18
    calls.insert(
        1,
        prim(
            "box",
            "rail_l",
            {"size": [bed_l * 0.94, rail_h, 0.03]},
            translation=[0, 0.45 + rail_h / 2, bed_w / 2 - 0.02],
        ),
    )
    calls.insert(
        2,
        prim(
            "box",
            "rail_r",
            {"size": [bed_l * 0.94, rail_h, 0.03]},
            translation=[0, 0.45 + rail_h / 2, -(bed_w / 2 - 0.02)],
        ),
    )
    calls.insert(
        3,
        prim(
            "box",
            "bench",
            {"size": [bed_l * 0.3, 0.08, bed_w * 0.85]},
            translation=[bed_l * 0.32, 0.45 + rail_h + 0.04, 0],
        ),
    )
    calls.append(material(["rail_l", "rail_r", "bench"], "Trim wood", jc(rng, WOOD_DARK)))
    notes = (
        "A wagon: the four-wheel cart bed plus two side rails and a raised bench seat -- "
        "the largest of the family's carts, so wheel radius and bed size both scale up."
    )
    return calls, notes, False


def build_bucket_cart(rng):
    r = jitter(rng, 0.16)
    box_l, box_w, box_h = jitter(rng, 0.5), jitter(rng, 0.5), jitter(rng, 0.55)
    box_y = r * 1.9 + box_h / 2
    calls = [prim("box", "bucket", {"size": [box_l, box_h, box_w]}, translation=[0, box_y, 0])]
    calls.append(
        prim(
            "cylinder",
            "wheel_fr",
            {"radius": r, "height": 0.07, "segments": 16},
            translation=[box_w / 2, r, box_l * 0.3],
            rotation=[0, 0, 90],
        )
    )
    calls += mirror_copy("wheel_fr", axis=0, offset=0.0)
    calls.append(
        prim(
            "cylinder",
            "wheel_br",
            {"radius": r, "height": 0.07, "segments": 16},
            translation=[box_w / 2, r, -box_l * 0.3],
            rotation=[0, 0, 90],
        )
    )
    calls += mirror_copy("wheel_br", axis=0, offset=0.0)
    calls.append(material(["bucket"], "Bucket wood", jc(rng, WOOD)))
    calls.append(
        material(
            ["wheel_fr", "wheel_fr.001", "wheel_br", "wheel_br.001"],
            "Iron rim",
            jc(rng, IRON),
            roughness=0.5,
            metallic=0.6,
        )
    )
    notes = "Just a tall box on four wheels, per the prompt -- no bed lip, no handles."
    return calls, notes, False


def build_crate_dolly(rng):
    r = jitter(rng, 0.06)
    plat_l, plat_w, plat_h = jitter(rng, 0.6), jitter(rng, 0.4), jitter(rng, 0.03)
    plat_y = r * 2 + plat_h / 2
    calls = [
        prim("box", "platform", {"size": [plat_l, plat_h, plat_w]}, translation=[0, plat_y, 0])
    ]
    calls.append(
        prim(
            "cylinder",
            "caster_fr",
            {"radius": r, "height": 0.03, "segments": 12},
            translation=[plat_w / 2 - r, r, plat_l * 0.4],
            rotation=[0, 0, 90],
        )
    )
    calls += mirror_copy("caster_fr", axis=0, offset=0.0)
    calls.append(
        prim(
            "cylinder",
            "caster_br",
            {"radius": r, "height": 0.03, "segments": 12},
            translation=[plat_w / 2 - r, r, -plat_l * 0.4],
            rotation=[0, 0, 90],
        )
    )
    calls += mirror_copy("caster_br", axis=0, offset=0.0)
    calls.append(material(["platform"], "Dolly wood", jc(rng, WOOD)))
    calls.append(
        material(
            ["caster_fr", "caster_fr.001", "caster_br", "caster_br.001"],
            "Caster iron",
            jc(rng, IRON),
            roughness=0.5,
            metallic=0.6,
        )
    )
    notes = "A flat, low platform with four small caster wheels -- low profile is the whole point."
    return calls, notes, False


def build_two_wheel_boolean_axle(rng):
    r = jitter(rng, 0.24)
    bed_l, bed_w, bed_h = jitter(rng, 0.85), jitter(rng, 0.5), jitter(rng, 0.15)
    housing_w = jitter(rng, 0.12)
    bed_y = r * 1.6
    calls = [prim("box", "bed", {"size": [bed_l, bed_h, bed_w]}, translation=[0, bed_y, 0])]
    calls.append(
        prim(
            "box",
            "housing_r",
            {"size": [housing_w, housing_w, housing_w]},
            translation=[bed_w / 2 + housing_w * 0.4, r, 0],
        )
    )
    calls.append(
        prim(
            "cylinder",
            "socket_r",
            {"radius": r * 0.22, "height": housing_w * 1.6, "segments": 12},
            translation=[bed_w / 2 + housing_w * 0.4, r, 0],
            rotation=[0, 0, 90],
        )
    )
    calls.append(
        {
            "name": "clay_boolean",
            "arguments": {
                "kind": "difference",
                "uids": [{"$ref": "housing_r"}, {"$ref": "socket_r"}],
            },
        }
    )
    calls += mirror_copy("housing_r", axis=0, offset=0.0)
    calls.append(
        prim(
            "cylinder",
            "wheel_r",
            {"radius": r, "height": housing_w * 0.9, "segments": 16},
            translation=[bed_w / 2 + housing_w * 0.4, r, 0],
            rotation=[0, 0, 90],
        )
    )
    calls += mirror_copy("wheel_r", axis=0, offset=0.0)
    calls.append(material(["bed"], "Cart wood", jc(rng, WOOD)))
    calls.append(
        material(
            ["housing_r", "housing_r.001"],
            "Axle housing",
            jc(rng, IRON),
            roughness=0.4,
            metallic=0.7,
        )
    )
    calls.append(
        material(
            ["wheel_r", "wheel_r.001"], "Wheel iron", jc(rng, IRON), roughness=0.5, metallic=0.6
        )
    )
    notes = (
        "housing_r is added before socket_r, so it is first in document order and survives "
        "the difference as the axle block with a round hole cut through it; the housing is "
        "mirror-copied before the wheel is added so the wheel does not get cut a second time."
    )
    return calls, notes, False


def build_sled(rng):
    length = jitter(rng, 0.9)
    lift = jitter(rng, 0.12)
    radius = 0.025
    half = length / 2
    path = [
        [-half, 0.0, 0.0],
        [-half * 0.3, 0.0, 0.0],
        [half * 0.55, lift * 0.5, 0.0],
        [half * 0.85, lift, 0.0],
    ]
    # the path's own y-range (0..lift) is re-centred by the generator to
    # (-lift/2..+lift/2), so the lowest point of the tube's surface sits at
    # -lift/2 - radius before translation; lift that exactly onto the ground.
    ground_y = lift / 2 + radius
    calls = [
        prim(
            "tube",
            "runner_r",
            {"path": path, "radius": radius, "sides": 8},
            translation=[0.18, ground_y, 0],
        )
    ]
    calls += mirror_copy("runner_r", axis=0, offset=0.0)
    deck_l, deck_w, deck_h = length * 0.8, jitter(rng, 0.34), 0.03
    calls.append(
        prim(
            "box",
            "deck",
            {"size": [deck_l, deck_h, deck_w]},
            translation=[0, ground_y + radius + deck_h / 2, 0],
        )
    )
    calls.append(
        material(
            ["runner_r", "runner_r.001"], "Runner iron", jc(rng, IRON), roughness=0.4, metallic=0.6
        )
    )
    calls.append(material(["deck"], "Deck wood", jc(rng, WOOD)))
    notes = (
        "Two curved runners built with tube's own path (a shallow rise at the front), one "
        "mirror-copied across X for the pair; the deck sits on top clear of both runners."
    )
    return calls, notes, False


def build_chariot(rng):
    import math

    r = jitter(rng, 0.32)
    body_l, body_w, body_h = jitter(rng, 0.55), jitter(rng, 0.5), jitter(rng, 0.45)
    body_y = r * 1.5
    calls = [
        prim(
            "box",
            "body",
            {"size": [body_l, body_h, body_w]},
            translation=[0, body_y + body_h / 2, 0],
        )
    ]
    n_spokes = 8
    hub_names: list[str] = []
    spoke_names: list[str] = []
    rim_names: list[str] = []
    for side, sx in (("r", 1), ("l", -1)):
        cx = sx * (body_w / 2 + 0.04)
        y0 = r  # both wheels lie flat, centred at their own (x, z=0) -- array-radial is not used
        hub = f"hub_{side}"
        calls.append(
            prim(
                "cylinder",
                hub,
                {"radius": r * 0.18, "height": 0.04, "segments": 16},
                translation=[cx, y0, 0],
            )
        )
        hub_names.append(hub)
        for i in range(n_spokes):
            theta = 2 * math.pi * i / n_spokes
            name = f"spoke_{side}_{i}"
            # A box's local Z (its long axis here) rotated by theta about Y points at
            # world direction (sin theta, 0, cos theta); its centre sits half the
            # spoke's own reach out along that same direction from the hub.
            calls.append(
                prim(
                    "box",
                    name,
                    {"size": [0.02, 0.02, r * 0.82]},
                    translation=[cx + r * 0.41 * math.sin(theta), y0, r * 0.41 * math.cos(theta)],
                    rotation=[0, math.degrees(theta), 0],
                )
            )
            spoke_names.append(name)
        rim = f"rim_{side}"
        calls.append(
            prim(
                "torus",
                rim,
                {"radius": r, "tube": 0.02, "segments": 20, "sides": 10},
                translation=[cx, y0, 0],
            )
        )
        rim_names.append(rim)
    calls.append(material(["body"], "Chariot wood", jc(rng, WOOD_DARK)))
    calls.append(
        material(hub_names + rim_names, "Wheel iron", jc(rng, IRON), roughness=0.5, metallic=0.6)
    )
    calls.append(material(spoke_names, "Spoke wood", jc(rng, WOOD)))
    notes = (
        "Each wheel lies flat (no rotation) so its spokes can be placed directly by trig "
        "rather than through array-radial: the op spins copies around the *world* Y axis, "
        "which only passes through a hub actually sitting at x=0, z=0 -- not true for either "
        "of a chariot's two side-by-side wheels, so this build computes each spoke's own "
        "position instead of relying on the op for an off-centre hub."
    )
    return calls, notes, False


register(
    "handcart",
    "easy",
    [
        "a simple wooden handcart with two wheels",
        "a two-wheeled wooden handcart, plain and sturdy",
        "small hand cart, two wheels, nothing fancy",
    ],
    build_handcart,
)
register(
    "wheelbarrow",
    "easy",
    [
        "a plain wooden wheelbarrow",
        "an old wooden wheelbarrow with one front wheel",
        "a simple garden wheelbarrow, single wheel up front",
    ],
    build_wheelbarrow,
)
register(
    "cart4",
    "easy",
    [
        "a simple wooden cart with four wheels",
        "a plain four-wheeled wooden cart",
        "wooden cart, four wheels, flat bed",
    ],
    build_cart4,
)
register(
    "wagon",
    "easy",
    [
        "a simple wooden wagon with a bench seat and four wheels",
        "an old four-wheeled wagon with side rails and a driver's bench",
        "wagon: four wheels, rails along the bed, a bench up front",
    ],
    build_wagon,
)
register(
    "bucket_cart",
    "easy",
    [
        "a plain wooden bucket cart, just a box on wheels",
        "a simple tall box cart on four small wheels",
        "bucket cart -- a deep wooden box mounted on four wheels",
    ],
    build_bucket_cart,
)
register(
    "crate_dolly",
    "easy",
    [
        "a basic wooden crate dolly, a flat platform on four wheels",
        "a low flat dolly on four small caster wheels",
        "simple moving dolly, thin platform, four casters",
    ],
    build_crate_dolly,
)
register(
    "axle_boolean_cart",
    "medium",
    [
        "a two-wheeled cart with a boolean-cut axle socket",
        "a handcart whose axle housing has a bored-out socket for the wheel",
        "two-wheel cart, the axle block has a round hole cut through it for the axle",
    ],
    build_two_wheel_boolean_axle,
)
register(
    "sled",
    "medium",
    [
        "a sled with two mirrored curved runners",
        "a wooden sled, two curved runners that lift at the front",
        "simple sled: paired curved runners under a flat deck",
    ],
    build_sled,
)
register(
    "chariot",
    "medium",
    [
        "a chariot with two wheels, each with a radial spoke pattern",
        "a war chariot with two spoked wheels",
        "light chariot, spoked wheels on both sides",
    ],
    build_chariot,
)


def build_wagon_wheel(rng):
    r = jitter(rng, 0.4)
    hub_r, hub_h = jitter(rng, 0.06), jitter(rng, 0.1)
    tube = 0.025
    calls = [
        prim(
            "cylinder",
            "hub",
            {"radius": hub_r, "height": hub_h, "segments": 16},
            translation=[0, hub_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "box",
            "spoke",
            {"size": [0.025, hub_h * 0.8, r * 0.86]},
            translation=[0, hub_h / 2, r * 0.43],
        )
    )
    calls += array_radial("spoke", count=8, axis=1)
    calls.append(
        prim(
            "torus",
            "rim",
            {"radius": r, "tube": tube, "segments": 24, "sides": 10},
            translation=[0, tube, 0],
        )
    )
    calls.append(material(["hub", "rim"], "Rim iron", jc(rng, IRON), roughness=0.5, metallic=0.6))
    calls.append(material(radial_names("spoke", 8), "Spoke wood", jc(rng, WOOD)))
    notes = (
        "A standalone wagon wheel, lying flat like the spoked-hub fixture (hub at the world "
        "origin's x/z, one spoke arrayed radially about Y into eight), rim a flat torus at "
        "the same low height -- everything centred at x=0, z=0, which is what makes the Y "
        "array (and each object's own ground clearance) exact regardless of height."
    )
    return calls, notes, False


def build_cart_wheel_rim(rng):
    r = jitter(rng, 0.36)
    hub_r, hub_h = jitter(rng, 0.03), jitter(rng, 0.06)
    tube = 0.018
    calls = [
        prim(
            "cylinder",
            "hub",
            {"radius": hub_r, "height": hub_h, "segments": 12},
            translation=[0, hub_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "box",
            "spoke",
            {"size": [0.018, hub_h * 0.7, r * 0.92]},
            translation=[0, hub_h / 2, r * 0.46],
        )
    )
    calls += array_radial("spoke", count=10, axis=1)
    calls.append(
        prim(
            "torus",
            "rim",
            {"radius": r, "tube": tube, "segments": 28, "sides": 8},
            translation=[0, tube, 0],
        )
    )
    calls.append(material(["rim", "hub"], "Rim iron", jc(rng, IRON), roughness=0.45, metallic=0.65))
    calls.append(material(radial_names("spoke", 10), "Spoke wood", jc(rng, WOOD_LIGHT)))
    notes = (
        "Thinner rim and more, thinner spokes than the wagon wheel -- lying flat the same way, "
        "for the same reason."
    )
    return calls, notes, False


def build_wheelbarrow_fork(rng):
    tray_l, tray_w, tray_h = jitter(rng, 0.68), jitter(rng, 0.48), jitter(rng, 0.26)
    r = jitter(rng, 0.19)
    leg_h = jitter(rng, 0.34)
    tray_y = leg_h
    fork_w, fork_h, fork_d = jitter(rng, 0.03), jitter(rng, 0.16), jitter(rng, 0.05)
    calls = [prim("box", "tray", {"size": [tray_l, tray_h, tray_w]}, translation=[0, tray_y, 0])]
    calls.append(
        prim(
            "cylinder",
            "leg_r",
            {"radius": 0.02, "height": leg_h, "segments": 8},
            translation=[tray_w * 0.35, leg_h / 2, tray_l * 0.4],
        )
    )
    calls += mirror_copy("leg_r", axis=0, offset=0.0)
    fork_z = -(tray_l / 2 + fork_d * 0.6)
    calls.append(
        prim(
            "box",
            "fork_r",
            {"size": [fork_w, fork_h, fork_d]},
            translation=[tray_w * 0.28, r, fork_z],
        )
    )
    calls.append(
        prim(
            "cylinder",
            "axle_hole_r",
            {"radius": r * 0.16, "height": fork_w * 1.6, "segments": 10},
            translation=[tray_w * 0.28, r, fork_z],
            rotation=[0, 0, 90],
        )
    )
    calls.append(
        {
            "name": "clay_boolean",
            "arguments": {
                "kind": "difference",
                "uids": [{"$ref": "fork_r"}, {"$ref": "axle_hole_r"}],
            },
        }
    )
    calls += mirror_copy("fork_r", axis=0, offset=0.0)
    calls.append(
        prim(
            "cylinder",
            "wheel",
            {"radius": r, "height": fork_w * 1.2, "segments": 16},
            translation=[0, r, fork_z],
            rotation=[0, 0, 90],
        )
    )
    calls.append(material(["tray"], "Barrow wood", jc(rng, WOOD)))
    calls.append(
        material(["leg_r", "leg_r.001"], "Leg iron", jc(rng, IRON), roughness=0.5, metallic=0.6)
    )
    calls.append(
        material(["fork_r", "fork_r.001"], "Fork iron", jc(rng, IRON), roughness=0.4, metallic=0.7)
    )
    calls.append(material(["wheel"], "Wheel wood", jc(rng, WOOD_DARK)))
    notes = (
        "fork_r is added, then bored with a difference (fork_r first in document order "
        "survives with the hole), mirror-copied for the second prong, and the wheel is added "
        "last so it never takes part in the boolean."
    )
    return calls, notes, False


register(
    "wagon_wheel",
    "medium",
    [
        "a wagon wheel with spokes arranged in a radial pattern",
        "a wooden wagon wheel, spokes set radially around the hub",
        "single wagon wheel, radial spokes and an iron rim",
    ],
    build_wagon_wheel,
)
register(
    "cart_wheel_rim",
    "medium",
    [
        "a cart wheel rim with evenly spaced spokes arranged radially",
        "a light cart wheel, thin rim, evenly spaced spokes",
        "cart wheel rim, spokes set at even radial intervals",
    ],
    build_cart_wheel_rim,
)
register(
    "wheelbarrow_fork",
    "medium",
    [
        "a wheelbarrow with a single wheel mounted in a cut-out fork",
        "a wheelbarrow whose wheel sits in a bored-out fork bracket",
        "wheelbarrow, one wheel seated in a mirrored fork with a bored axle hole",
    ],
    build_wheelbarrow_fork,
)


# --- boats, hovercraft, rockets, tanks, bikes --------------------------------


def build_rowboat_easy(rng):
    length, width, height = jitter(rng, 1.6), jitter(rng, 0.55), jitter(rng, 0.28)
    calls = [prim("box", "hull", {"size": [length, height, width]}, translation=[0, height / 2, 0])]
    calls.append(
        prim(
            "box",
            "seat",
            {"size": [width * 0.7, 0.03, width * 0.7]},
            translation=[0, height * 0.7, 0],
        )
    )
    calls.append(material(["hull"], "Hull wood", jc(rng, WOOD)))
    calls.append(material(["seat"], "Seat wood", jc(rng, WOOD_LIGHT)))
    notes = "Easy tier: an elongated flattened box for the hull plus a single flat seat, primitives only."
    return calls, notes, False


def build_rowboat_hull_swept(rng):
    length = jitter(rng, 1.8)
    beam, draft = jitter(rng, 0.6), jitter(rng, 0.3)
    outline = [[-beam / 2, 0.0], [beam / 2, 0.0], [beam * 0.65, draft], [-beam * 0.65, draft]]
    calls = [
        prim(
            "sweep",
            "hull",
            {"outline": outline, "depth": length, "taper": 0.35, "twist": 0, "sections": 2},
            translation=[0, draft / 2, 0],
        )
    ]
    calls.append(material(["hull"], "Hull wood", jc(rng, WOOD)))
    notes = (
        "Hard tier: sweep's own depth axis (Z) is already the boat's length, so no rotation "
        "is needed -- the outline is a flat-bottomed trapezoid and taper=0.35 narrows the +Z "
        "end into the bow while the -Z stern keeps the full beam."
    )
    return calls, notes, False


def build_rowboat_bench_pair(rng):
    length, width, height = jitter(rng, 1.7), jitter(rng, 0.58), jitter(rng, 0.3)
    calls = [prim("box", "hull", {"size": [length, height, width]}, translation=[0, height / 2, 0])]
    calls.append(
        prim(
            "box",
            "bench_r",
            {"size": [width * 0.75, 0.03, width * 0.75]},
            translation=[length * 0.22, height * 0.7, 0],
        )
    )
    calls += mirror_copy("bench_r", axis=0, offset=0.0)
    calls.append(material(["hull"], "Hull wood", jc(rng, WOOD)))
    calls.append(material(["bench_r", "bench_r.001"], "Bench wood", jc(rng, WOOD_LIGHT)))
    notes = "Medium tier: two bench seats set at +/-22% of hull length, one mirror-copied for the other."
    return calls, notes, False


def build_hovercraft_easy(rng):
    r = jitter(rng, 0.55)
    skirt_h = jitter(rng, 0.14)
    dome_r = jitter(rng, 0.35)
    calls = [
        prim(
            "cylinder",
            "skirt",
            {"radius": r, "height": skirt_h, "segments": 24},
            translation=[0, skirt_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "capsule",
            "dome",
            {"radius": dome_r, "height": dome_r * 0.3, "segments": 20, "rings": 5},
            translation=[0, skirt_h + dome_r * 0.5, 0],
            scale=[1.0, 0.55, 1.0],
        )
    )
    calls.append(material(["skirt"], "Hull grey", jc(rng, [0.6, 0.62, 0.65])))
    calls.append(material(["dome"], "Canopy", jc(rng, [0.35, 0.55, 0.6]), roughness=0.25))
    notes = (
        "Easy tier: a low wide cylinder skirt with a flattened capsule dome cabin, primitives only."
    )
    return calls, notes, False


def build_hovercraft_vents(rng):
    calls, notes, _ = build_hovercraft_easy(rng)
    r = 0.55
    vent = prim(
        "cylinder",
        "vent",
        {"radius": 0.05, "height": 0.04, "segments": 10},
        translation=[r * 0.82, 0.14 / 2, 0],
        rotation=[0, 0, 90],
    )
    calls = calls[:1] + [vent] + calls[1:]
    calls += array_radial("vent", count=10, axis=1)
    calls.append(
        material(radial_names("vent", 10), "Vent iron", jc(rng, IRON), roughness=0.4, metallic=0.6)
    )
    notes = (
        "Medium tier: one vent placed at the skirt's own radius, arrayed radially about the "
        "world Y axis (the skirt is centred at x=0, z=0, so this is the skirt's own axis)."
    )
    return calls, notes, False


def build_rocket_easy(rng):
    body_r, body_h = jitter(rng, 0.22), jitter(rng, 1.2)
    nose_h = jitter(rng, 0.4)
    calls = [
        prim(
            "cylinder",
            "body",
            {"radius": body_r, "height": body_h, "segments": 20},
            translation=[0, body_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "cone",
            "nose",
            {"radius": body_r, "height": nose_h, "segments": 20},
            translation=[0, body_h + nose_h / 2, 0],
        )
    )
    calls.append(
        material(
            ["body", "nose"],
            "Rocket hull",
            jc(rng, [0.85, 0.85, 0.85]),
            roughness=0.3,
            metallic=0.2,
        )
    )
    notes = "Cone nose on a cylindrical body, exactly the two generators the family calls for."
    return calls, notes, False


def build_rocket_fins(rng):
    calls, notes, _ = build_rocket_easy(rng)
    body_r = 0.22
    fin_h, fin_w = jitter(rng, 0.3), jitter(rng, 0.18)
    fin = prim(
        "box", "fin_r", {"size": [0.015, fin_h, fin_w]}, translation=[body_r + 0.015, fin_h / 2, 0]
    )
    calls.insert(2, fin)
    calls[3:3] = mirror_copy("fin_r", axis=0, offset=0.0)
    calls.append(
        material(["fin_r", "fin_r.001"], "Fin metal", jc(rng, STEEL), roughness=0.4, metallic=0.6)
    )
    notes = (
        "Medium tier: a mirrored pair of tail fins (not a four-way radial array), placed at "
        "the base of the body and mirror-copied across X."
    )
    return calls, notes, False


def build_tank_easy(rng):
    hull_l, hull_w, hull_h = jitter(rng, 1.1), jitter(rng, 0.65), jitter(rng, 0.35)
    turret_r, turret_h = jitter(rng, 0.28), jitter(rng, 0.22)
    calls = [
        prim("box", "hull", {"size": [hull_l, hull_h, hull_w]}, translation=[0, hull_h / 2, 0])
    ]
    calls.append(
        prim(
            "cylinder",
            "turret",
            {"radius": turret_r, "height": turret_h, "segments": 20},
            translation=[0, hull_h + turret_h / 2, 0],
        )
    )
    calls.append(
        prim(
            "cylinder",
            "barrel",
            {"radius": 0.03, "height": jitter(rng, 0.55), "segments": 10},
            translation=[0, hull_h + turret_h * 0.5, turret_r + 0.28],
            rotation=[90, 0, 0],
        )
    )
    calls.append(material(["hull", "turret"], "Tank plate", jc(rng, OLIVE)))
    calls.append(material(["barrel"], "Gun steel", jc(rng, STEEL), roughness=0.35, metallic=0.7))
    notes = (
        "A boxy hull, a round turret on top and a plain barrel -- primitives and placement only."
    )
    return calls, notes, False


def build_tank_tread_guards(rng):
    calls, notes, _ = build_tank_easy(rng)
    hull_w, hull_h = 0.65, 0.35
    guard = prim(
        "box",
        "guard_r",
        {"size": [jitter(rng, 1.0), 0.06, 0.1]},
        translation=[hull_w / 2 + 0.05, hull_h * 0.35, 0],
    )
    calls.insert(1, guard)
    calls[2:2] = mirror_copy("guard_r", axis=0, offset=0.0)
    calls.append(
        material(
            ["guard_r", "guard_r.001"], "Tread guard", jc(rng, IRON), roughness=0.5, metallic=0.5
        )
    )
    notes = (
        "Medium tier: one tread guard mirror-copied across X to run down both sides of the hull."
    )
    return calls, notes, False


def build_bike_easy(rng):
    r = jitter(rng, 0.33)
    wheelbase = jitter(rng, 0.75)
    calls = [
        prim(
            "cylinder",
            "wheel_f",
            {"radius": r, "height": 0.025, "segments": 24},
            translation=[0, r, wheelbase / 2],
            rotation=[90, 0, 0],
        )
    ]
    calls.append(
        prim(
            "cylinder",
            "wheel_b",
            {"radius": r, "height": 0.025, "segments": 24},
            translation=[0, r, -wheelbase / 2],
            rotation=[90, 0, 0],
        )
    )
    calls.append(
        prim(
            "cylinder",
            "down_tube",
            {"radius": 0.015, "height": wheelbase * 1.05, "segments": 8},
            translation=[0, r * 1.15, 0],
            rotation=[90, 0, 20],
        )
    )
    calls.append(
        prim(
            "cylinder",
            "seat_post",
            {"radius": 0.014, "height": jitter(rng, 0.35), "segments": 8},
            translation=[0, r * 1.9, -wheelbase * 0.28],
            rotation=[10, 0, 0],
        )
    )
    calls.append(
        prim(
            "box",
            "handlebar",
            {"size": [jitter(rng, 0.4), 0.014, 0.014]},
            translation=[0, r * 1.85, wheelbase * 0.42],
        )
    )
    calls.append(
        material(["wheel_f", "wheel_b"], "Tyre rubber", jc(rng, [0.08, 0.08, 0.08]), roughness=0.8)
    )
    calls.append(
        material(
            ["down_tube", "seat_post", "handlebar"],
            "Frame steel",
            jc(rng, STEEL),
            roughness=0.4,
            metallic=0.6,
        )
    )
    notes = (
        "Two wheels standing on their rims (rotated 90 about X, axle along Z), a diagonal "
        "down tube, a seat post and a straight handlebar -- primitives only."
    )
    return calls, notes, False


def build_bike_seat(rng):
    calls, notes, _ = build_bike_easy(rng)
    seat = prim(
        "capsule",
        "seat",
        {"radius": 0.05, "height": 0.16, "segments": 12, "rings": 4},
        translation=[0, 0.33 * 1.9 + 0.35, -0.75 * 0.28],
        rotation=[0, 0, 90],
    )
    calls.insert(-2, seat)
    calls.append(material(["seat"], "Seat leather", jc(rng, LEATHER), roughness=0.7))
    notes = "Medium tier: the seat post is topped with a capsule laid on its side for the cushion."
    return calls, notes, False


register(
    "rowboat_easy",
    "easy",
    [
        "a small flat-bottomed rowboat hull, open on top",
        "a plain wooden rowboat, flat bottomed",
        "simple rowboat hull with a single seat",
    ],
    build_rowboat_easy,
)
register(
    "rowboat_swept",
    "hard",
    [
        "a rowboat hull that tapers along a swept length from a wide stern to a narrow bow",
        "a boat hull, swept and tapered so the bow is much narrower than the stern",
        "narrow-bowed rowboat hull, taper running the length of the swept hull",
    ],
    build_rowboat_hull_swept,
)
register(
    "rowboat_benches",
    "medium",
    [
        "a rowboat hull with two mirrored bench seats",
        "a rowboat with a pair of mirrored benches inside",
        "small boat hull, two bench seats mirrored front to back",
    ],
    build_rowboat_bench_pair,
)
register(
    "hovercraft_easy",
    "easy",
    [
        "a basic round-topped hovercraft platform with a low skirt",
        "a simple hovercraft: wide low skirt, domed cabin on top",
        "toy hovercraft, flat skirt and a rounded cabin",
    ],
    build_hovercraft_easy,
)
register(
    "hovercraft_vents",
    "medium",
    [
        "a hovercraft with a radial ring of lift vents around its rim",
        "hovercraft platform ringed with evenly spaced lift vents",
        "a hovercraft skirt with vents arranged radially around the edge",
    ],
    build_hovercraft_vents,
)
register(
    "rocket_easy",
    "easy",
    [
        "a simple toy rocket with a cone nose on a cylindrical body",
        "a plain toy rocket, cone nose, cylinder body",
        "basic rocket: cylindrical body, conical nose cap",
    ],
    build_rocket_easy,
)
register(
    "rocket_fins",
    "medium",
    [
        "a rocket with a mirrored pair of tail fins",
        "toy rocket with two mirrored fins at the base",
        "a cone-nosed rocket, a pair of mirrored fins near the tail",
    ],
    build_rocket_fins,
)
register(
    "tank_easy",
    "easy",
    [
        "a plain boxy tank hull with a round turret on top",
        "a simple tank: boxy hull, round turret, plain barrel",
        "basic toy tank, boxy body and a round turret",
    ],
    build_tank_easy,
)
register(
    "tank_guards",
    "medium",
    [
        "a tank hull with two mirrored tread guards along its sides",
        "tank hull, mirrored tread guards running along both sides",
        "boxy tank with a pair of mirrored side tread guards",
    ],
    build_tank_tread_guards,
)
register(
    "bike_easy",
    "easy",
    [
        "a basic bicycle frame with two wheels and a straight handlebar",
        "a simple bicycle: two wheels, a diagonal frame tube, straight bars",
        "plain bike frame, two wheels, no gears or chain",
    ],
    build_bike_easy,
)
register(
    "bike_seat",
    "medium",
    [
        "a bicycle with a capsule-shaped seat cushion",
        "bike with a rounded capsule seat on the post",
        "bicycle frame topped with a capsule-shaped saddle",
    ],
    build_bike_seat,
)


# --- hand tools & weapons: hammers, axes, pickaxes, wrenches -----------------


def build_hammer_basic(rng):
    handle_r, handle_h = jitter(rng, 0.018), jitter(rng, 0.32)
    head_w, head_h, head_d = jitter(rng, 0.16), jitter(rng, 0.045), jitter(rng, 0.045)
    calls = [
        prim(
            "cylinder",
            "handle",
            {"radius": handle_r, "height": handle_h, "segments": 10},
            translation=[0, handle_h / 2, 0],
        )
    ]
    calls.append(
        prim("box", "head", {"size": [head_w, head_h, head_d]}, translation=[0, handle_h, 0])
    )
    calls.append(material(["handle"], "Handle wood", jc(rng, WOOD)))
    calls.append(material(["head"], "Head steel", jc(rng, STEEL), roughness=0.35, metallic=0.7))
    notes = (
        "A cylindrical handle with a rectangular box head centred on top, per the prompt exactly."
    )
    return calls, notes, False


def build_mallet(rng):
    handle_r, handle_h = jitter(rng, 0.016), jitter(rng, 0.3)
    head_r, head_h = jitter(rng, 0.06), jitter(rng, 0.14)
    calls = [
        prim(
            "cylinder",
            "handle",
            {"radius": handle_r, "height": handle_h, "segments": 10},
            translation=[0, handle_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "cylinder",
            "head",
            {"radius": head_r, "height": head_h, "segments": 16},
            translation=[0, handle_h, 0],
            rotation=[0, 0, 90],
        )
    )
    calls.append(material(["handle"], "Mallet wood", jc(rng, WOOD_LIGHT)))
    calls.append(material(["head"], "Head wood", jc(rng, WOOD_DARK)))
    notes = "Both handle and head are wood; head is a stubby cylinder lying on its side across the handle."
    return calls, notes, False


def build_hammer_capsule_head(rng):
    handle_r, handle_h = jitter(rng, 0.017), jitter(rng, 0.3)
    head_r, head_h = jitter(rng, 0.045), jitter(rng, 0.1)
    calls = [
        prim(
            "cylinder",
            "handle",
            {"radius": handle_r, "height": handle_h, "segments": 10},
            translation=[0, handle_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "capsule",
            "head",
            {"radius": head_r, "height": head_h, "segments": 14, "rings": 4},
            translation=[0, handle_h, 0],
            rotation=[0, 0, 90],
        )
    )
    calls.append(material(["handle"], "Handle wood", jc(rng, WOOD)))
    calls.append(material(["head"], "Weighted steel", jc(rng, STEEL), roughness=0.3, metallic=0.8))
    notes = "Capsule star technique: the weighted head is a capsule laid on its side (rotated 90 about Z)."
    return calls, notes, False


def build_hand_axe(rng):
    handle_r, handle_h = jitter(rng, 0.016), jitter(rng, 0.22)
    head_w, head_h, head_d = jitter(rng, 0.03), jitter(rng, 0.09), jitter(rng, 0.1)
    calls = [
        prim(
            "cylinder",
            "handle",
            {"radius": handle_r, "height": handle_h, "segments": 10},
            translation=[0, handle_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "box",
            "head",
            {"size": [head_w, head_h, head_d]},
            translation=[head_d * 0.3, handle_h * 0.92, 0],
        )
    )
    calls.append(material(["handle"], "Handle wood", jc(rng, WOOD)))
    calls.append(material(["head"], "Blade iron", jc(rng, IRON), roughness=0.4, metallic=0.6))
    notes = "Easy tier: a short handle, box head offset forward -- no sweep, no mirror."
    return calls, notes, False


def build_war_axe_crescent(rng):
    handle_r, handle_h = jitter(rng, 0.018), jitter(rng, 0.75)
    attach_y = handle_h * 0.86
    blade_len, blade_w = jitter(rng, 0.16), jitter(rng, 0.02)
    outline = [
        [0.0, 0.0],
        [blade_len, blade_len * 0.35],
        [blade_len * 0.7, blade_len * 0.55],
        [0.0, blade_len * 0.3],
    ]
    calls = [
        prim(
            "cylinder",
            "handle",
            {"radius": handle_r, "height": handle_h, "segments": 12},
            translation=[0, handle_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "sweep",
            "blade_top",
            {"outline": outline, "depth": blade_w, "taper": 0.85, "twist": 0, "sections": 1},
            translation=[0, attach_y, 0],
            rotation=[0, 90, 0],
        )
    )
    calls += mirror_copy("blade_top", axis=1, offset=attach_y)
    calls.append(material(["handle"], "Haft wood", jc(rng, WOOD_DARK)))
    calls.append(
        material(
            ["blade_top", "blade_top.001"],
            "Blade steel",
            jc(rng, STEEL),
            roughness=0.3,
            metallic=0.8,
        )
    )
    notes = (
        "One crescent half is built with sweep (outline in local XY, rotated 90 about Y so "
        "its depth axis lies along world X, projecting the blade sideways off the haft), then "
        "mirror-copied across the horizontal plane at the haft's attach height -- 'mirrored "
        "around the haft' read as top/bottom symmetry of one blade, not two opposing blades."
    )
    return calls, notes, False


def build_double_axe(rng):
    handle_r, handle_h = jitter(rng, 0.02), jitter(rng, 0.8)
    attach_y = handle_h * 0.82
    blade_len = jitter(rng, 0.22)
    outline = [
        [0.02, -blade_len * 0.3],
        [0.02, blade_len * 0.3],
        [blade_len, blade_len * 0.5],
        [blade_len * 0.5, 0.0],
        [blade_len, -blade_len * 0.5],
    ]
    calls = [
        prim(
            "cylinder",
            "handle",
            {"radius": handle_r, "height": handle_h, "segments": 12},
            translation=[0, handle_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "sweep",
            "blade_r",
            {"outline": outline, "depth": 0.02, "taper": 1.0, "twist": 0, "sections": 1},
            translation=[0, attach_y, 0],
            rotation=[90, 0, 0],
        )
    )
    calls += mirror_copy("blade_r", axis=0, offset=0.0)
    calls.append(material(["handle"], "Haft wood", jc(rng, WOOD_DARK)))
    calls.append(
        material(
            ["blade_r", "blade_r.001"], "Blade steel", jc(rng, STEEL), roughness=0.3, metallic=0.8
        )
    )
    notes = (
        "A concave crescent outline swept with depth as its thickness, rotated so it projects "
        "sideways (+X) off the haft, then mirror-copied across X so the second blade faces "
        "the opposite way -- a true double-bladed (labrys) head."
    )
    return calls, notes, False


def build_pickaxe(rng):
    handle_r, handle_h = jitter(rng, 0.017), jitter(rng, 0.7)
    grip_y = handle_h * 0.4
    grip_r, grip_h = jitter(rng, 0.028), jitter(rng, 0.14)
    attach_y = handle_h * 0.92
    point_len = jitter(rng, 0.22)
    calls = [
        prim(
            "cylinder",
            "handle",
            {"radius": handle_r, "height": handle_h, "segments": 10},
            translation=[0, handle_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "capsule",
            "grip",
            {"radius": grip_r, "height": grip_h, "segments": 12, "rings": 3},
            translation=[0, grip_y, 0],
        )
    )
    calls.append(
        prim(
            "cone",
            "point_r",
            {"radius": 0.018, "height": point_len, "segments": 10},
            translation=[point_len / 2, attach_y, 0],
            rotation=[0, 0, -90],
        )
    )
    calls += mirror_copy("point_r", axis=0, offset=0.0)
    calls.append(material(["handle"], "Handle wood", jc(rng, WOOD)))
    calls.append(material(["grip"], "Grip leather", jc(rng, LEATHER), roughness=0.8))
    calls.append(
        material(
            ["point_r", "point_r.001"], "Pick steel", jc(rng, STEEL), roughness=0.35, metallic=0.75
        )
    )
    notes = (
        "Capsule grip wraps the handle partway down; two cone pick points, one built pointing "
        "+X and mirror-copied to -X, giving the classic two-point pickaxe head."
    )
    return calls, notes, False


def build_wrench(rng):
    handle_l, handle_w, handle_h = jitter(rng, 0.22), jitter(rng, 0.03), jitter(rng, 0.012)
    jaw_len, jaw_w = jitter(rng, 0.05), jitter(rng, 0.012)
    calls = [
        prim(
            "box",
            "handle",
            {"size": [handle_l, handle_h, handle_w]},
            translation=[0, handle_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "box",
            "jaw_r",
            {"size": [jaw_len, handle_h, jaw_w]},
            translation=[handle_l / 2 + jaw_len / 2 * 0.5, handle_h / 2, handle_w / 2],
            rotation=[0, 20, 0],
        )
    )
    calls += mirror_copy("jaw_r", axis=2, offset=0.0)
    calls.append(
        material(
            ["handle", "jaw_r", "jaw_r.001"],
            "Wrench steel",
            jc(rng, STEEL),
            roughness=0.3,
            metallic=0.8,
        )
    )
    notes = "One jaw prong mirror-copied across the Z=0 plane (the handle's own centreline) for the pair."
    return calls, notes, False


register(
    "hammer_basic",
    "easy",
    [
        "a basic hammer with a cylindrical handle and a rectangular head",
        "plain hammer, round handle, square head",
        "a simple carpenter's hammer",
    ],
    build_hammer_basic,
)
register(
    "mallet",
    "easy",
    [
        "a plain wooden mallet",
        "a simple wooden mallet, round head",
        "basic wood mallet for chisel work",
    ],
    build_mallet,
)
register(
    "hammer_capsule",
    "medium",
    [
        "a hammer with a capsule-shaped weighted head",
        "a heavy hammer, capsule-shaped head for the weight",
        "hammer with a rounded capsule striking head",
    ],
    build_hammer_capsule_head,
)
register(
    "hand_axe",
    "easy",
    [
        "a simple hand axe with a short handle",
        "small hand axe, plain head",
        "a basic short-hafted hand axe",
    ],
    build_hand_axe,
)
register(
    "war_axe_crescent",
    "medium",
    [
        "a war axe with a crescent blade mirrored around the haft",
        "war axe, crescent-shaped blade symmetric about the haft",
        "a long-hafted war axe with a mirrored crescent head",
    ],
    build_war_axe_crescent,
)
register(
    "double_axe",
    "medium",
    [
        "a double-bladed axe, mirrored blades on either side of the haft",
        "a labrys-style axe with two mirrored blades",
        "double-headed axe, a blade mirrored on each side",
    ],
    build_double_axe,
)
register(
    "pickaxe_capsule",
    "medium",
    [
        "a pickaxe with a capsule grip and two mirrored pick points",
        "pickaxe with a padded capsule grip and points on both ends",
        "a pick with a rounded capsule handle grip and mirrored tips",
    ],
    build_pickaxe,
)
register(
    "wrench",
    "medium",
    [
        "a wrench with two mirrored jaw prongs",
        "an open-end wrench, jaws mirrored across the handle",
        "wrench head with a mirrored pair of jaw prongs",
    ],
    build_wrench,
)


# --- pickaxe-plain / shovel / trowel / rake / plow ---------------------------


def build_pickaxe_plain(rng):
    handle_r, handle_h = jitter(rng, 0.016), jitter(rng, 0.65)
    head_w, head_h, head_d = jitter(rng, 0.32), jitter(rng, 0.03), jitter(rng, 0.03)
    calls = [
        prim(
            "cylinder",
            "handle",
            {"radius": handle_r, "height": handle_h, "segments": 10},
            translation=[0, handle_h / 2, 0],
        )
    ]
    calls.append(
        prim("box", "head", {"size": [head_w, head_h, head_d]}, translation=[0, handle_h * 0.95, 0])
    )
    calls.append(material(["handle"], "Handle wood", jc(rng, WOOD)))
    calls.append(material(["head"], "Pick iron", jc(rng, IRON), roughness=0.4, metallic=0.65))
    notes = "Easy tier: a plain elongated box head crossing a straight handle, no capsule grip."
    return calls, notes, False


def build_shovel(rng):
    handle_r, handle_h = jitter(rng, 0.016), jitter(rng, 0.85)
    blade_w, blade_h, blade_d = jitter(rng, 0.16), jitter(rng, 0.22), jitter(rng, 0.02)
    calls = [
        prim(
            "cylinder",
            "handle",
            {"radius": handle_r, "height": handle_h, "segments": 10},
            translation=[0, handle_h / 2, 0],
        )
    ]
    calls.append(
        prim("box", "blade", {"size": [blade_w, blade_h, blade_d]}, translation=[0, blade_h / 2, 0])
    )
    calls.append(material(["handle"], "Handle wood", jc(rng, WOOD)))
    calls.append(material(["blade"], "Blade steel", jc(rng, STEEL), roughness=0.4, metallic=0.6))
    notes = "A flat box blade at the base of a long handle -- garden shovel, easy tier."
    return calls, notes, False


def build_crossbow_stock(rng):
    stock_l, stock_w, stock_h = jitter(rng, 0.65), jitter(rng, 0.06), jitter(rng, 0.05)
    bar_l = jitter(rng, 0.5)
    calls = [
        prim("box", "stock", {"size": [stock_l, stock_h, stock_w]}, translation=[0, stock_h / 2, 0])
    ]
    calls.append(
        prim(
            "box",
            "crossbar",
            {"size": [stock_w, stock_h * 0.8, bar_l]},
            translation=[stock_l * 0.32, stock_h / 2, 0],
        )
    )
    calls.append(material(["stock", "crossbar"], "Stock wood", jc(rng, WOOD)))
    notes = "A long stock with a crossbar near the front -- primitives only, resting on the ground."
    return calls, notes, False


def build_oar(rng):
    shaft_r, shaft_h = jitter(rng, 0.014), jitter(rng, 1.1)
    blade_w, blade_h, blade_d = jitter(rng, 0.14), jitter(rng, 0.35), jitter(rng, 0.012)
    calls = [
        prim(
            "cylinder",
            "shaft",
            {"radius": shaft_r, "height": shaft_h, "segments": 10},
            translation=[0, shaft_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "box",
            "blade",
            {"size": [blade_w, blade_h, blade_d]},
            translation=[0, shaft_h + blade_h / 2 - 0.02, 0],
        )
    )
    calls.append(material(["shaft"], "Shaft wood", jc(rng, WOOD)))
    calls.append(material(["blade"], "Blade wood", jc(rng, WOOD_LIGHT)))
    notes = "A long cylindrical shaft with a flat paddle box near the top -- simple oar."
    return calls, notes, False


def build_staff(rng):
    r, h = jitter(rng, 0.017), jitter(rng, 1.4)
    calls = [
        prim(
            "cylinder",
            "shaft",
            {"radius": r, "height": h, "segments": 12},
            translation=[0, h / 2, 0],
        )
    ]
    calls.append(material(["shaft"], "Staff wood", jc(rng, WOOD_DARK)))
    notes = "A single tall thin cylinder -- the simplest build in the family, primitives only."
    return calls, notes, False


def build_trowel(rng):
    handle_r, handle_h = jitter(rng, 0.014), jitter(rng, 0.11)
    blade_w, blade_h, blade_d = jitter(rng, 0.05), jitter(rng, 0.13), jitter(rng, 0.008)
    calls = [
        prim(
            "cylinder",
            "handle",
            {"radius": handle_r, "height": handle_h, "segments": 10},
            translation=[0, handle_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "box",
            "blade",
            {"size": [blade_w, blade_h, blade_d]},
            translation=[0, handle_h + blade_h / 2 - 0.01, 0],
        )
    )
    calls.append(material(["handle"], "Handle wood", jc(rng, WOOD)))
    calls.append(material(["blade"], "Blade steel", jc(rng, STEEL), roughness=0.4, metallic=0.6))
    notes = "A small hand trowel: short handle, narrow flat blade -- primitives only."
    return calls, notes, False


def build_torch(rng):
    r, h = jitter(rng, 0.02), jitter(rng, 0.5)
    top_r = jitter(rng, 0.03)
    calls = [
        prim(
            "cylinder",
            "shaft",
            {"radius": r, "height": h, "segments": 10},
            translation=[0, h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "capsule",
            "top",
            {"radius": top_r, "height": top_r * 0.4, "segments": 12, "rings": 3},
            translation=[0, h + top_r * 0.6, 0],
        )
    )
    calls.append(material(["shaft"], "Torch wood", jc(rng, WOOD_DARK)))
    calls.append(material(["top"], "Torch head", jc(rng, [0.5, 0.3, 0.12])))
    notes = "A cylinder shaft with a capsule cap for the rounded top, per the prompt."
    return calls, notes, False


def build_truck_bed_liner(rng):
    l, w, h = jitter(rng, 1.3), jitter(rng, 1.0), jitter(rng, 0.25)
    calls = [prim("box", "liner", {"size": [l, h, w]}, translation=[0, h / 2, 0])]
    calls.append(material(["liner"], "Liner plastic", jc(rng, [0.12, 0.12, 0.13]), roughness=0.65))
    notes = "One plain box -- the prompt asks for exactly a rectangular box."
    return calls, notes, False


def build_anvil(rng):
    base_w, base_h, base_d = jitter(rng, 0.42), jitter(rng, 0.22), jitter(rng, 0.2)
    top_w, top_h, top_d = jitter(rng, 0.5), jitter(rng, 0.12), jitter(rng, 0.16)
    horn_r, horn_h = jitter(rng, 0.04), jitter(rng, 0.16)
    calls = [
        prim("box", "base", {"size": [base_w, base_h, base_d]}, translation=[0, base_h / 2, 0])
    ]
    calls.append(
        prim("box", "top", {"size": [top_w, top_h, top_d]}, translation=[0, base_h + top_h / 2, 0])
    )
    calls.append(
        prim(
            "cone",
            "horn",
            {"radius": horn_r, "height": horn_h, "segments": 12},
            translation=[top_w / 2 + horn_h * 0.3, base_h + top_h / 2, 0],
            rotation=[0, 0, -90],
        )
    )
    calls.append(
        material(
            ["base", "top", "horn"], "Anvil iron", jc(rng, IRON), roughness=0.35, metallic=0.75
        )
    )
    notes = "A narrower base box, a wider top box, and a cone horn pointing out the front."
    return calls, notes, False


def build_ladder(rng):
    rail_h, rail_w = jitter(rng, 1.4), jitter(rng, 0.4)
    rail_r = 0.02
    rungs = 4
    calls = [
        prim(
            "cylinder",
            "rail_r",
            {"radius": rail_r, "height": rail_h, "segments": 8},
            translation=[rail_w / 2, rail_h / 2, 0],
        )
    ]
    calls += mirror_copy("rail_r", axis=0, offset=0.0)
    calls_extra = []
    for i in range(rungs):
        y = rail_h * (0.15 + 0.7 * i / (rungs - 1))
        calls_extra.append(
            prim(
                "cylinder",
                f"rung_{i}",
                {"radius": 0.014, "height": rail_w, "segments": 8},
                translation=[0, y, 0],
                rotation=[0, 0, 90],
            )
        )
    calls = calls + calls_extra
    calls.append(material(["rail_r", "rail_r.001"], "Rail wood", jc(rng, WOOD)))
    calls.append(material([f"rung_{i}" for i in range(rungs)], "Rung wood", jc(rng, WOOD_LIGHT)))
    notes = "Rungs placed individually (not arrayed) to keep this an easy-tier build -- primitives only."
    return calls, notes, False


def build_rake(rng):
    handle_r, handle_h = jitter(rng, 0.014), jitter(rng, 1.0)
    head_w, head_h, head_d = jitter(rng, 0.28), jitter(rng, 0.02), jitter(rng, 0.03)
    calls = [
        prim(
            "cylinder",
            "handle",
            {"radius": handle_r, "height": handle_h, "segments": 10},
            translation=[0, handle_h / 2, 0],
        )
    ]
    calls.append(
        prim("box", "head", {"size": [head_w, head_h, head_d]}, translation=[0, head_h / 2, 0])
    )
    calls.append(material(["handle"], "Handle wood", jc(rng, WOOD)))
    calls.append(material(["head"], "Head steel", jc(rng, STEEL), roughness=0.5, metallic=0.5))
    notes = "A flat wide box head at the base of a long handle -- garden rake, easy tier."
    return calls, notes, False


def build_broom(rng):
    handle_r, handle_h = jitter(rng, 0.016), jitter(rng, 0.9)
    bristle_r, bristle_h = jitter(rng, 0.06), jitter(rng, 0.22)
    calls = [
        prim(
            "cylinder",
            "handle",
            {"radius": handle_r, "height": handle_h, "segments": 10},
            translation=[0, handle_h / 2 + bristle_h, 0],
        )
    ]
    calls.append(
        prim(
            "cone",
            "bristles",
            {"radius": bristle_r, "height": bristle_h, "segments": 14},
            translation=[0, bristle_h / 2, 0],
            rotation=[180, 0, 0],
        )
    )
    calls.append(material(["handle"], "Handle wood", jc(rng, WOOD)))
    calls.append(
        material(["bristles"], "Bristle straw", jc(rng, [0.68, 0.58, 0.28]), roughness=0.9)
    )
    notes = "An upside-down cone (apex at the top) makes a flared bristle base under the handle."
    return calls, notes, False


def build_saddle_blank(rng):
    l, w, h = jitter(rng, 0.55), jitter(rng, 0.32), jitter(rng, 0.08)
    calls = [prim("box", "blank", {"size": [l, h, w]}, translation=[0, h / 2, 0])]
    calls.append(material(["blank"], "Leather", jc(rng, LEATHER), roughness=0.7))
    notes = "One flattened box -- the prompt asks for a blank, not a finished saddle shape."
    return calls, notes, False


def build_buckler(rng):
    r, h = jitter(rng, 0.16), jitter(rng, 0.025)
    calls = [
        prim(
            "cylinder",
            "shield",
            {"radius": r, "height": h, "segments": 20},
            translation=[0, h / 2, 0],
        )
    ]
    calls.append(material(["shield"], "Shield wood", jc(rng, WOOD)))
    notes = "A small, flat cylinder disc -- easy tier, smaller radius than the full round shield."
    return calls, notes, False


def build_yoke(rng):
    bar_l, bar_r = jitter(rng, 0.9), jitter(rng, 0.03)
    peg_r, peg_h = jitter(rng, 0.018), jitter(rng, 0.14)
    calls = [
        prim(
            "cylinder",
            "bar",
            {"radius": bar_r, "height": bar_l, "segments": 14},
            translation=[0, peg_h + bar_r, 0],
            rotation=[0, 0, 90],
        )
    ]
    calls.append(
        prim(
            "cylinder",
            "peg_r",
            {"radius": peg_r, "height": peg_h, "segments": 8},
            translation=[bar_l * 0.32, peg_h / 2, 0],
        )
    )
    calls += mirror_copy("peg_r", axis=0, offset=0.0)
    calls.append(material(["bar", "peg_r", "peg_r.001"], "Yoke wood", jc(rng, WOOD_DARK)))
    notes = "A horizontal bar over two downward pegs -- pegs reach the ground, bar sits atop them."
    return calls, notes, False


register(
    "pickaxe_plain",
    "easy",
    [
        "a plain metal pickaxe head on a straight handle",
        "a basic pickaxe, plain iron head",
        "simple pick, straight handle, flat head",
    ],
    build_pickaxe_plain,
)
register(
    "shovel",
    "easy",
    [
        "a basic garden shovel with a flat blade",
        "a plain garden shovel",
        "simple digging shovel, flat steel blade",
    ],
    build_shovel,
)
register(
    "crossbow_stock",
    "easy",
    [
        "a plain wooden crossbow stock",
        "a simple crossbow stock with a crossbar",
        "basic wooden crossbow body, no string",
    ],
    build_crossbow_stock,
)
register(
    "oar",
    "easy",
    [
        "a plain rowboat oar, a long flat paddle",
        "a simple wooden oar",
        "long oar with a flat paddle blade",
    ],
    build_oar,
)
register(
    "staff",
    "easy",
    [
        "a simple wooden walking staff",
        "a plain tall walking stick",
        "basic wooden staff, nothing but a shaft",
    ],
    build_staff,
)


def build_fishing_rod(rng):
    r, h = jitter(rng, 0.012), jitter(rng, 1.6)
    calls = [
        prim(
            "cylinder", "rod", {"radius": r, "height": h, "segments": 10}, translation=[0, h / 2, 0]
        )
    ]
    calls.append(material(["rod"], "Cane", jc(rng, [0.6, 0.5, 0.28]), roughness=0.6))
    notes = (
        "A single long thin cylinder, per the prompt -- thinner and longer than the staff build."
    )
    return calls, notes, False


register(
    "fishing_rod",
    "easy",
    [
        "a basic fishing rod, a long thin cylinder",
        "a simple fishing rod",
        "plain thin fishing rod, nothing but the pole",
    ],
    build_fishing_rod,
)
register(
    "trowel",
    "easy",
    [
        "a basic hand trowel",
        "a simple garden hand trowel",
        "small trowel, short handle, narrow blade",
    ],
    build_trowel,
)
register(
    "torch",
    "easy",
    [
        "a plain wooden torch, a cylinder with a rounded top",
        "a simple wooden torch",
        "basic torch, round top for the flame",
    ],
    build_torch,
)
register(
    "truck_bed_liner",
    "easy",
    [
        "a simple pickup truck bed liner, a rectangular box",
        "a plain rectangular truck bed liner",
        "basic bed liner box for a pickup truck",
    ],
    build_truck_bed_liner,
)
register(
    "anvil",
    "easy",
    [
        "a plain metal anvil block",
        "a basic blacksmith's anvil",
        "simple anvil, base, top and a horn",
    ],
    build_anvil,
)
register(
    "ladder",
    "easy",
    [
        "a basic wooden ladder",
        "a simple wooden ladder with four rungs",
        "plain ladder, two rails, a few rungs",
    ],
    build_ladder,
)
register(
    "rake",
    "easy",
    ["a simple hand rake with a flat head", "a plain garden rake", "basic rake, flat wide head"],
    build_rake,
)
register(
    "broom",
    "easy",
    ["a simple wooden broom", "a plain broom with straw bristles", "basic sweeping broom"],
    build_broom,
)
register(
    "saddle_blank",
    "easy",
    [
        "a plain riding saddle blank, a flattened box shape",
        "a simple saddle blank",
        "basic flattened saddle blank shape",
    ],
    build_saddle_blank,
)
register(
    "buckler",
    "easy",
    [
        "a simple round buckler shield, small and flat",
        "a small round buckler",
        "plain flat buckler shield",
    ],
    build_buckler,
)
register(
    "yoke",
    "easy",
    ["a plain wooden yoke bar", "a simple ox yoke bar", "basic yoke bar with two pegs"],
    build_yoke,
)


# --- shields, spears, staffs, pouches, weights, plows, maces -----------------


def build_shield_round(rng):
    r, h = jitter(rng, 0.35), jitter(rng, 0.035)
    calls = [
        prim(
            "cylinder",
            "shield",
            {"radius": r, "height": h, "segments": 24},
            translation=[0, h / 2, 0],
        )
    ]
    calls.append(material(["shield"], "Shield wood", jc(rng, WOOD)))
    notes = "A wide, flat cylinder disc -- round wooden shield, easy tier."
    return calls, notes, False


def build_shield_emblem(rng):
    r, h = jitter(rng, 0.36), jitter(rng, 0.035)
    calls = [
        prim(
            "cylinder",
            "shield",
            {"radius": r, "height": h, "segments": 24},
            translation=[0, h / 2, 0],
        )
    ]
    calls.append(
        prim("box", "emblem_r", {"size": [0.04, 0.012, 0.09]}, translation=[r * 0.4, h, 0])
    )
    calls += mirror_copy("emblem_r", axis=0, offset=0.0)
    calls.append(material(["shield"], "Shield wood", jc(rng, WOOD)))
    calls.append(
        material(
            ["emblem_r", "emblem_r.001"],
            "Emblem brass",
            jc(rng, BRASS),
            roughness=0.3,
            metallic=0.8,
        )
    )
    notes = "One emblem piece mirror-copied across X (the shield's own vertical centreline) for symmetry."
    return calls, notes, False


def build_shield_boss(rng):
    r, h = jitter(rng, 0.34), jitter(rng, 0.03)
    boss_r, boss_h = jitter(rng, 0.07), jitter(rng, 0.05)
    calls = [
        prim(
            "cylinder",
            "shield",
            {"radius": r, "height": h, "segments": 24},
            translation=[0, h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "capsule",
            "boss",
            {"radius": boss_r, "height": boss_h * 0.2, "segments": 16, "rings": 4},
            translation=[0, h, 0],
            scale=[1.0, 0.6, 1.0],
        )
    )
    calls.append(material(["shield"], "Shield wood", jc(rng, WOOD)))
    calls.append(material(["boss"], "Boss bronze", jc(rng, BRASS), roughness=0.3, metallic=0.85))
    notes = "A flattened capsule at the shield's own centre stands in for the domed boss."
    return calls, notes, False


def build_shield_rivets(rng):
    r, h = jitter(rng, 0.35), jitter(rng, 0.035)
    calls = [
        prim(
            "cylinder",
            "shield",
            {"radius": r, "height": h, "segments": 24},
            translation=[0, h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "capsule",
            "rivet",
            {"radius": 0.012, "height": 0.002, "segments": 8, "rings": 3},
            translation=[r * 0.9, h, 0],
        )
    )
    calls += array_radial("rivet", count=12, axis=1)
    calls.append(material(["shield"], "Shield wood", jc(rng, WOOD)))
    calls.append(
        material(
            radial_names("rivet", 12), "Rivet iron", jc(rng, IRON), roughness=0.4, metallic=0.7
        )
    )
    notes = "One rivet placed at the rim radius, arrayed radially about Y -- the shield centre is the world origin."
    return calls, notes, False


def build_spear_counterweight(rng):
    shaft_r, shaft_h = jitter(rng, 0.016), jitter(rng, 1.7)
    head_len = jitter(rng, 0.22)
    cw_r, cw_h = jitter(rng, 0.022), jitter(rng, 0.08)
    # capsule spans height/2 + radius on each side of its own centre.
    cw_bottom_to_top = cw_h + 2 * cw_r
    calls = [
        prim(
            "capsule",
            "counterweight",
            {"radius": cw_r, "height": cw_h, "segments": 12, "rings": 3},
            translation=[0, cw_h / 2 + cw_r, 0],
        )
    ]
    calls.append(
        prim(
            "cylinder",
            "shaft",
            {"radius": shaft_r, "height": shaft_h, "segments": 10},
            translation=[0, cw_bottom_to_top + shaft_h / 2, 0],
        )
    )
    calls.append(
        prim(
            "cone",
            "head",
            {"radius": shaft_r * 1.4, "height": head_len, "segments": 12},
            translation=[0, cw_bottom_to_top + shaft_h + head_len / 2, 0],
        )
    )
    calls.append(material(["shaft"], "Shaft wood", jc(rng, WOOD)))
    calls.append(material(["head"], "Head steel", jc(rng, STEEL), roughness=0.35, metallic=0.7))
    calls.append(
        material(["counterweight"], "Butt bronze", jc(rng, BRASS), roughness=0.4, metallic=0.6)
    )
    notes = (
        "Capsule at the very butt end, touching the ground (translation y = height/2 + "
        "radius, the capsule's own ground formula); the shaft is stacked on top of it rather "
        "than overlapping it, and the cone head tops the shaft."
    )
    return calls, notes, False


def build_spear_tapered_shaft(rng):
    shaft_len = jitter(rng, 1.8)
    r0 = jitter(rng, 0.02)
    outline = []
    import math

    for i in range(8):
        a = 2 * math.pi * i / 8
        outline.append([r0 * math.cos(a), r0 * math.sin(a)])
    head_len = jitter(rng, 0.24)
    calls = [
        prim(
            "sweep",
            "shaft",
            {"outline": outline, "depth": shaft_len, "taper": 0.4, "twist": 0, "sections": 3},
            rotation=[-90, 0, 0],
            translation=[0, shaft_len / 2, 0],
        )
    ]
    calls.append(
        prim(
            "cone",
            "head",
            {"radius": r0 * 0.4, "height": head_len, "segments": 12},
            translation=[0, shaft_len + head_len / 2, 0],
        )
    )
    calls.append(material(["shaft"], "Shaft wood", jc(rng, WOOD)))
    calls.append(material(["head"], "Head steel", jc(rng, STEEL), roughness=0.35, metallic=0.7))
    notes = (
        "The shaft itself is sweep, not cylinder -- an octagon outline tapering (taper=0.4) "
        "toward the +Z/tip end, rotated -90 about X so the tapered end points up, matching "
        "the validated blade-orientation formula (rotation=[-90,0,0], translation y=depth/2)."
    )
    return calls, notes, False


def build_mace(rng):
    handle_r, handle_h = jitter(rng, 0.02), jitter(rng, 0.65)
    head_r, head_h = jitter(rng, 0.07), jitter(rng, 0.16)
    stud_r = 0.014
    calls = [
        prim(
            "cylinder",
            "handle",
            {"radius": handle_r, "height": handle_h, "segments": 12},
            translation=[0, handle_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "capsule",
            "head",
            {"radius": head_r, "height": head_h, "segments": 16, "rings": 4},
            translation=[0, handle_h + head_h / 2 + head_r, 0],
        )
    )
    calls.append(
        prim(
            "capsule",
            "stud",
            {"radius": stud_r, "height": 0.006, "segments": 8, "rings": 2},
            translation=[head_r + stud_r * 0.6, handle_h + head_h / 2 + head_r, 0],
            rotation=[0, 0, 90],
        )
    )
    calls += array_radial("stud", count=8, axis=1)
    calls.append(material(["handle"], "Haft wood", jc(rng, WOOD_DARK)))
    calls.append(material(["head"], "Head steel", jc(rng, STEEL), roughness=0.4, metallic=0.7))
    calls.append(
        material(radial_names("stud", 8), "Stud iron", jc(rng, IRON), roughness=0.45, metallic=0.6)
    )
    notes = (
        "The capsule head sits centred on x=0,z=0, so a stud placed at the head's own radius "
        "and arrayed radially about world Y rotates correctly around the head's own axis."
    )
    return calls, notes, False


def build_longbow(rng):
    grip_r, grip_h = jitter(rng, 0.018), jitter(rng, 0.18)
    limb_span = jitter(rng, 0.75)
    bow_depth = jitter(rng, 0.12)
    path = [
        [0.0, 0.0, 0.0],
        [limb_span * 0.4, bow_depth * 0.6, 0.0],
        [limb_span * 0.8, bow_depth * 0.9, 0.0],
        [limb_span, bow_depth * 0.5, 0.0],
    ]
    calls = [
        prim(
            "cylinder",
            "grip",
            {"radius": grip_r, "height": grip_h, "segments": 10},
            translation=[0, limb_span + grip_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "tube",
            "limb_top",
            {"path": path, "radius": 0.012, "sides": 8},
            translation=[0, limb_span + grip_h / 2, 0],
            rotation=[0, 0, 90],
        )
    )
    calls += mirror_copy("limb_top", axis=1, offset=limb_span + grip_h / 2)
    calls.append(material(["grip"], "Grip leather", jc(rng, LEATHER), roughness=0.75))
    calls.append(material(["limb_top", "limb_top.001"], "Limb wood", jc(rng, WOOD_LIGHT)))
    notes = (
        "One limb built with tube's own curved path (rotated 90 about Z so the path's local "
        "X-curve runs up the world Y axis), mirror-copied across the horizontal plane through "
        "the grip for the lower limb."
    )
    return calls, notes, False


def build_hand_plow(rng):
    frame_l, frame_w, frame_h = jitter(rng, 0.55), jitter(rng, 0.3), jitter(rng, 0.05)
    blade_len = jitter(rng, 0.18)
    calls = [
        prim(
            "box",
            "frame",
            {"size": [frame_l, frame_h, frame_w]},
            translation=[0, jitter(rng, 0.35), 0],
        )
    ]
    outline = [[0.0, 0.0], [0.0, blade_len * 0.5], [blade_len, 0.0]]
    calls.append(
        prim(
            "sweep",
            "blade_r",
            {"outline": outline, "depth": 0.015, "taper": 1.0, "twist": 0, "sections": 1},
            translation=[frame_w * 0.35, jitter(rng, 0.35) - frame_h / 2, 0],
            rotation=[90, 0, 0],
        )
    )
    calls += mirror_copy("blade_r", axis=0, offset=0.0)
    calls.append(material(["frame"], "Frame wood", jc(rng, WOOD_DARK)))
    calls.append(
        material(
            ["blade_r", "blade_r.001"], "Blade steel", jc(rng, STEEL), roughness=0.4, metallic=0.6
        )
    )
    notes = (
        "A wedge-shaped sweep blade mirror-copied across X gives the plow's mirrored blade pair."
    )
    return calls, notes, False


def build_tool_pouch(rng):
    r, h = jitter(rng, 0.09), jitter(rng, 0.16)
    strap_w = jitter(rng, 0.02)
    calls = [
        prim(
            "capsule",
            "pouch",
            {"radius": r, "height": h, "segments": 14, "rings": 4},
            translation=[0, h / 2 + r, 0],
        )
    ]
    calls.append(
        prim(
            "box",
            "strap",
            {"size": [strap_w, h * 1.3, strap_w]},
            translation=[0, h / 2 + r, r + strap_w / 2],
        )
    )
    calls.append(material(["pouch"], "Pouch leather", jc(rng, LEATHER), roughness=0.75))
    calls.append(material(["strap"], "Strap leather", jc(rng, WOOD_DARK), roughness=0.7))
    notes = "Capsule star technique: the whole pouch body is one capsule, with a thin box strap in front."
    return calls, notes, False


def build_net_weight(rng):
    r, h = jitter(rng, 0.02), jitter(rng, 0.05)
    ring_r = jitter(rng, 0.008)
    calls = [
        prim(
            "capsule",
            "sinker",
            {"radius": r, "height": h, "segments": 12, "rings": 3},
            translation=[0, h / 2 + r, 0],
        )
    ]
    calls.append(
        prim(
            "torus",
            "eyelet",
            {"radius": ring_r, "tube": ring_r * 0.3, "segments": 12, "sides": 8},
            translation=[0, h + 2 * r + ring_r, 0],
            rotation=[90, 0, 0],
        )
    )
    calls.append(
        material(["sinker"], "Lead", jc(rng, [0.3, 0.3, 0.32]), roughness=0.5, metallic=0.4)
    )
    calls.append(material(["eyelet"], "Eyelet wire", jc(rng, IRON), roughness=0.4, metallic=0.7))
    notes = (
        "A small capsule sinker with a torus eyelet on top for the line -- capsule star technique."
    )
    return calls, notes, False


register(
    "shield_round",
    "easy",
    [
        "a round wooden shield, flat and simple",
        "a plain round wooden shield",
        "simple flat round shield",
    ],
    build_shield_round,
)
register(
    "shield_emblem",
    "medium",
    [
        "a shield with a symmetrical emblem mirrored across its face",
        "round shield with a mirrored emblem design",
        "a shield bearing a symmetric mirrored emblem",
    ],
    build_shield_emblem,
)
register(
    "shield_boss",
    "medium",
    [
        "a shield boss, a domed cap mounted at the centre of a round shield",
        "a round shield with a domed boss at its centre",
        "shield with a raised domed boss in the middle",
    ],
    build_shield_boss,
)
register(
    "shield_rivets",
    "medium",
    [
        "a shield with a row of rivets arranged radially around its rim",
        "round shield with rivets evenly spaced around the edge",
        "a shield rimmed with radially arranged rivets",
    ],
    build_shield_rivets,
)
register(
    "spear_counterweight",
    "medium",
    [
        "a spear with a capsule-shaped counterweight at the butt end",
        "a long spear with a capsule counterweight at the base",
        "spear, cone head, capsule weight at the butt",
    ],
    build_spear_counterweight,
)
register(
    "spear_tapered_shaft",
    "hard",
    [
        "a spear whose shaft tapers gradually along its swept length toward the tip",
        "spear with a shaft that gradually narrows along its length",
        "a long spear, the shaft itself swept and tapered toward the point",
    ],
    build_spear_tapered_shaft,
)
register(
    "mace",
    "medium",
    [
        "a battle mace with a capsule-shaped studded head",
        "a mace with a studded capsule head",
        "war mace, capsule head ringed with studs",
    ],
    build_mace,
)
register(
    "longbow",
    "medium",
    [
        "a longbow with a mirrored curve on either side of the grip",
        "a longbow, two curved limbs mirrored about the grip",
        "long hunting bow with mirrored curved limbs",
    ],
    build_longbow,
)
register(
    "hand_plow",
    "medium",
    [
        "a hand plow with a mirrored pair of blades",
        "a small hand plow, two mirrored blades",
        "hand plow frame with mirrored blade pair",
    ],
    build_hand_plow,
)
register(
    "tool_pouch",
    "medium",
    [
        "a tool belt pouch, a capsule-shaped bag on a strap",
        "a belt pouch, capsule-shaped, with a strap",
        "small tool pouch, rounded capsule body on a strap",
    ],
    build_tool_pouch,
)
register(
    "net_weight",
    "medium",
    [
        "a fishing net weight, a capsule-shaped sinker",
        "a capsule-shaped fishing sinker with an eyelet",
        "small net weight, capsule sinker with a wire loop",
    ],
    build_net_weight,
)


# --- swords, sabres, horns, scythe (hard: sweep + taper) --------------------


def build_longsword(rng):
    handle_r, handle_h = jitter(rng, 0.017), jitter(rng, 0.14)
    guard_w, guard_h, guard_d = jitter(rng, 0.12), jitter(rng, 0.02), jitter(rng, 0.025)
    pommel_r = jitter(rng, 0.02)
    blade_w, blade_t, blade_len = jitter(rng, 0.045), jitter(rng, 0.006), jitter(rng, 0.75)
    outline = [
        [-blade_w / 2, -blade_t / 2],
        [blade_w / 2, -blade_t / 2],
        [blade_w / 2, blade_t / 2],
        [-blade_w / 2, blade_t / 2],
    ]
    calls = [
        prim(
            "cylinder",
            "handle",
            {"radius": handle_r, "height": handle_h, "segments": 10},
            translation=[0, handle_h / 2 + pommel_r * 2, 0],
        )
    ]
    calls.append(
        prim(
            "capsule",
            "pommel",
            {"radius": pommel_r, "height": 0.006, "segments": 10, "rings": 3},
            translation=[0, pommel_r, 0],
        )
    )
    guard_y = handle_h + pommel_r * 2
    calls.append(
        prim("box", "guard", {"size": [guard_w, guard_h, guard_d]}, translation=[0, guard_y, 0])
    )
    calls.append(
        prim(
            "sweep",
            "blade",
            {"outline": outline, "depth": blade_len, "taper": 0.04, "twist": 0, "sections": 2},
            rotation=[-90, 0, 0],
            translation=[0, guard_y + blade_len / 2, 0],
        )
    )
    calls.append(material(["handle"], "Grip leather", jc(rng, LEATHER), roughness=0.75))
    calls.append(
        material(["pommel", "guard"], "Fittings brass", jc(rng, BRASS), roughness=0.3, metallic=0.8)
    )
    calls.append(
        material(["blade"], "Blade steel", jc(rng, [0.75, 0.76, 0.78]), roughness=0.2, metallic=0.9)
    )
    notes = (
        "The validated blade formula: a rectangular outline swept with rotation=[-90,0,0] and "
        "translation y = base_y + depth/2 puts the wide base at the guard and the near-zero "
        "taper (0.04) end at the tip, exactly like the confirmed hand test."
    )
    return calls, notes, False


def build_sabre(rng):
    calls, _, _ = build_longsword(rng)
    # angle the blade slightly off the handle's axis to read as a curved sabre --
    # a straight-blade simplification of "sweeps along an arc", noted honestly.
    for c in calls:
        if c["arguments"].get("name") == "blade":
            c["arguments"]["rotation"] = [-82, 0, 0]
            c["arguments"]["params"]["taper"] = 0.15
    notes = (
        "Simplification noted: sweep's spine is straight (only its cross-section scales/twists "
        "along Z), so the sabre's curve is approximated by tilting the whole tapered blade a "
        "few degrees off vertical rather than by a true bent spine, which this tool surface "
        "has no parameter for."
    )
    return calls, notes, False


def build_war_horn(rng):
    length = jitter(rng, 0.55)
    r0 = jitter(rng, 0.012)
    import math

    outline = [
        [r0 * math.cos(2 * math.pi * i / 8), r0 * math.sin(2 * math.pi * i / 8)] for i in range(8)
    ]
    calls = [
        prim(
            "sweep",
            "horn",
            {"outline": outline, "depth": length, "taper": 3.2, "twist": 0, "sections": 3},
            rotation=[-90, 0, 0],
            translation=[0, length / 2, 0],
        )
    ]
    calls.append(material(["horn"], "Horn brass", jc(rng, BRASS), roughness=0.3, metallic=0.75))
    notes = (
        "taper > 1 (3.2) flares the far/+Z end outward instead of narrowing it, so the "
        "octagon-outline mouthpiece stays small at the bottom and the bell flares wide at the "
        "top -- the same rotation/translation formula as the sword blade, taper inverted."
    )
    return calls, notes, False


def build_scythe(rng):
    shaft_r, shaft_h = jitter(rng, 0.016), jitter(rng, 1.3)
    blade_len, blade_w = jitter(rng, 0.5), jitter(rng, 0.06)
    outline = [
        [0.0, -blade_w / 2],
        [blade_len, -blade_w * 0.15],
        [blade_len, blade_w * 0.15],
        [0.0, blade_w / 2],
    ]
    calls = [
        prim(
            "cylinder",
            "shaft",
            {"radius": shaft_r, "height": shaft_h, "segments": 10},
            translation=[0, shaft_h / 2, 0],
        )
    ]
    calls.append(
        prim(
            "sweep",
            "blade",
            {"outline": outline, "depth": 0.01, "taper": 0.15, "twist": 0, "sections": 2},
            rotation=[90, 40, 0],
            translation=[blade_len * 0.4, shaft_h * 0.94, 0],
        )
    )
    calls.append(material(["shaft"], "Shaft wood", jc(rng, WOOD)))
    calls.append(material(["blade"], "Blade steel", jc(rng, STEEL), roughness=0.3, metallic=0.75))
    notes = (
        "Blade outline tapers toward its far end (taper=0.15) for the scythe's point; angled "
        "off the shaft (rotation y=40) so it reads as curving away from the handle rather than "
        "standing straight up like the sword's blade."
    )
    return calls, notes, False


register(
    "longsword",
    "hard",
    [
        "a longsword whose blade tapers smoothly from a wide base to a sharp point",
        "a longsword with a blade that narrows to a fine point",
        "sword, wide at the guard, tapering to a sharp tip",
    ],
    build_longsword,
)
register(
    "sabre",
    "hard",
    [
        "a curved sabre whose blade sweeps and tapers along a gentle arc",
        "a sabre, tapered blade angled off the handle",
        "curved cavalry sabre, tapered blade",
    ],
    build_sabre,
)
register(
    "war_horn",
    "hard",
    [
        "a war horn that tapers along a swept curve from mouthpiece to bell",
        "a brass war horn, narrow mouthpiece flaring to a wide bell",
        "signal horn that flares from a thin mouthpiece to a wide bell",
    ],
    build_war_horn,
)
register(
    "scythe",
    "hard",
    [
        "a scythe blade that curves and tapers along a swept path from handle to tip",
        "a scythe, its blade tapering to a point away from the shaft",
        "farm scythe with a tapered swept blade",
    ],
    build_scythe,
)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

EXTRA_PHRASING = {
    # a handful of star (sweep/capsule/mirror-copy/array) templates get a 4th
    # phrasing so the family lands on the 200-record target.
    "war_axe_crescent",
    "double_axe",
    "pickaxe_capsule",
    "wrench",
    "mace",
    "longbow",
    "hand_plow",
    "tool_pouch",
    "net_weight",
    "sabre",
    "war_horn",
    "scythe",
    "spear_tapered_shaft",
    "rowboat_swept",
}


def main() -> None:
    records: list[dict[str, Any]] = []
    n = 0
    for key, base in sorted(BASES.items()):
        prompts = list(base["prompts"])
        if key in EXTRA_PHRASING:
            # a light rephrasing of the first prompt, dimension-free, to add
            # a 4th distinct phrasing without inventing a new sentence style.
            prompts = prompts + [
                prompts[0].replace("a ", "a well-made ", 1)
                if prompts[0].startswith("a ")
                else "well-made " + prompts[0]
            ]
        for i, prompt in enumerate(prompts):
            n += 1
            rng = random.Random(f"{key}-{i}")
            calls, notes, allow_below = base["fn"](rng)
            rec: dict[str, Any] = {
                "id": f"{FAMILY}-{n:04d}",
                "family": FAMILY,
                "kind": "build",
                "prompt": prompt,
                "calls": calls,
                "notes": notes,
            }
            if allow_below:
                rec["allow_below_ground"] = True
            records.append(rec)

    records.sort(key=lambda r: r["id"])
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"wrote {len(records)} records ({len(BASES)} base builds) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
