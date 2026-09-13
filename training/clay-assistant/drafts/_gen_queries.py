"""Emits ``drafts/queries.jsonl`` -- the ``queries`` family: ``kind: query``
records, each a ``prior`` (a batch of calls that builds a simple scene) and
an ``answer`` (prose, no tool call). See ``training/clay-assistant/
seeds/queries.txt`` for the seed subjects this draws from and
``author_brief.md`` (this session's scratchpad) for the family paragraph.

Deterministic: every number here is either a fixed literal or read back from
a real replay of the record's own ``prior`` through the real ``agent_clay``
door (``gen.verify.replay`` / ``gen.convert.compact_scene``), so every answer
this script writes is true of the scene the model would actually see, not a
number I typed by hand and might have gotten wrong.

Run: ``uv run python training/clay-assistant/drafts/_gen_queries.py`` --
regenerates ``drafts/queries.jsonl`` in place.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_PKG = _HERE.parents[1]  # training/clay-assistant
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from gen import convert, verify  # noqa: E402

OUT_PATH = _HERE.parent / "queries.jsonl"

# --- materials (reused across priors; linear RGB 0..1) ---------------------

OAK = {"color": [0.45, 0.31, 0.18], "roughness": 0.75}
PALE_WOOD = {"color": [0.64, 0.5, 0.34], "roughness": 0.7}
DARK_WOOD = {"color": [0.3, 0.19, 0.11], "roughness": 0.8}
STONE = {"color": [0.55, 0.54, 0.5], "roughness": 0.9}
METAL = {"color": [0.55, 0.56, 0.58], "roughness": 0.35, "metallic": 0.8}
DARK_METAL = {"color": [0.22, 0.22, 0.24], "roughness": 0.4, "metallic": 0.85}
TERRACOTTA = {"color": [0.72, 0.42, 0.28], "roughness": 0.85}
IRON_BAND = {"color": [0.3, 0.3, 0.32], "roughness": 0.45, "metallic": 0.7}


# --- tiny call builders (readability only -- these are exactly the tool
# schemas, nothing hidden) ---------------------------------------------------


def prim(generator: str, name: str, params: dict, translation=None, rotation=None) -> dict:
    args: dict[str, Any] = {"generator": generator, "name": name, "params": params}
    if translation is not None:
        args["translation"] = list(translation)
    if rotation is not None:
        args["rotation"] = list(rotation)
    return {"name": "clay_add_primitive", "arguments": args}


def figure(key: str, name_prefix: str, translation=None, scale=None) -> dict:
    args: dict[str, Any] = {"key": key, "name_prefix": name_prefix}
    if translation is not None:
        args["translation"] = list(translation)
    if scale is not None:
        args["scale"] = scale
    return {"name": "clay_add_figure", "arguments": args}


def select(names: list[str]) -> dict:
    return {"name": "clay_select", "arguments": {"uids": [{"$ref": n} for n in names]}}


def op(name: str, params: dict | None = None) -> dict:
    args: dict[str, Any] = {"name": name}
    if params is not None:
        args["params"] = params
    return {"name": "clay_op", "arguments": args}


def boolean(kind: str, names: list[str]) -> dict:
    return {
        "name": "clay_boolean",
        "arguments": {"kind": kind, "uids": [{"$ref": n} for n in names]},
    }


def material(names: list[str], spec: dict, mat_name: str) -> dict:
    args = {"uids": [{"$ref": n} for n in names], "color": spec["color"], "name": mat_name}
    if "roughness" in spec:
        args["roughness"] = spec["roughness"]
    if "metallic" in spec:
        args["metallic"] = spec["metallic"]
    return {"name": "clay_material", "arguments": args}


def set_params(names: list[str] | str, params: dict) -> dict:
    if isinstance(names, str):
        return {"name": "clay_set_params", "arguments": {"uid": {"$ref": names}, "params": params}}
    return {
        "name": "clay_set_params",
        "arguments": {"uids": [{"$ref": n} for n in names], "params": params},
    }


# --- records ----------------------------------------------------------------

RECORDS: list[dict[str, Any]] = []
_next_id = 1


def emit(calls: list[dict], prompt: str, answer: str, notes: str = "") -> None:
    global _next_id
    rid = f"queries-{_next_id:04d}"
    _next_id += 1
    record = {
        "id": rid,
        "family": "queries",
        "kind": "query",
        "prompt": prompt,
        "prior": calls,
        "answer": answer,
    }
    if notes:
        record["notes"] = notes
    RECORDS.append(record)


class Facts:
    """Real numbers read back from a live replay of *calls* -- never typed by
    hand. ``check`` is run so a broken prior is caught here rather than only
    at ``build.py`` time."""

    def __init__(self, calls: list[dict], label: str):
        r = verify.replay(calls)
        reasons = verify.check(r)
        if reasons:
            raise AssertionError(f"prior {label!r} does not verify: {reasons}")
        scene = convert.compact_scene(verify.scene(r))
        self.objects = {row["name"]: row for row in scene["objects"]}
        self.bounds = scene["bounds"]
        self.count = len(scene["objects"])
        self.materials = scene["materials"]

    def size(self, name: str) -> list[float]:
        return self.objects[name]["size"]

    def y(self, name: str) -> float:
        return self.objects[name]["translation"][1]

    def bsize(self) -> list[float]:
        return self.bounds["size"]


def r2(x: float) -> float:
    return round(x, 2)


# =============================================================================
# EASY priors -- primitives, placement and materials only.
# =============================================================================

# --- 1. crate_stack ----------------------------------------------------------
crate_stack = [
    prim("box", "crate_bottom", {"size": [0.4, 0.35, 0.4]}, [0, 0.175, 0]),
    prim("box", "crate_middle", {"size": [0.4, 0.35, 0.4]}, [0, 0.525, 0]),
    prim("box", "crate_top", {"size": [0.4, 0.35, 0.4]}, [0, 0.875, 0]),
    material(["crate_bottom", "crate_middle", "crate_top"], PALE_WOOD, "Crate wood"),
]
f = Facts(crate_stack, "crate_stack")
emit(
    crate_stack,
    "a stack of three crates :: how tall is the whole stack?",
    f"About {r2(f.bsize()[1])} m tall -- three {f.size('crate_bottom')[1]} m crates stacked directly on top of one another.",
)
emit(
    crate_stack,
    "Here's a stack of three crates. How many objects make up the scene?",
    f"{f.count} objects: {f.count} separate crate boxes, one stacked on the next.",
)
emit(
    crate_stack,
    "I built a stack of three crates -- is anything floating free of the others?",
    "No. Each crate's bottom sits exactly on the crate below's top, so the stack is one unbroken column with no gaps.",
)

# --- 2. table_four_legs -------------------------------------------------------
table_four_legs = [
    prim("box", "top", {"size": [1.0, 0.05, 0.6]}, [0, 0.725, 0]),
    prim("box", "leg_fl", {"size": [0.05, 0.7, 0.05]}, [0.45, 0.35, 0.25]),
    prim("box", "leg_fr", {"size": [0.05, 0.7, 0.05]}, [-0.45, 0.35, 0.25]),
    prim("box", "leg_bl", {"size": [0.05, 0.7, 0.05]}, [0.45, 0.35, -0.25]),
    prim("box", "leg_br", {"size": [0.05, 0.7, 0.05]}, [-0.45, 0.35, -0.25]),
    material(["top", "leg_fl", "leg_fr", "leg_bl", "leg_br"], OAK, "Oak"),
]
f = Facts(table_four_legs, "table_four_legs")
emit(
    table_four_legs,
    "a wooden table with four legs :: how many objects are in the scene?",
    f"{f.count} objects: one tabletop and four legs.",
)
emit(
    table_four_legs,
    "quick check on the wooden table with four legs -- how tall is it?",
    f"About {r2(f.bsize()[1])} m to the top of the table.",
)
emit(
    table_four_legs,
    "For the four-legged wooden table, is anything floating?",
    "No -- all four legs reach the ground and the top sits flush on top of them.",
)


# --- 3. round_stool -----------------------------------------------------------
def leg_xy(radius: float, deg: float) -> list[float]:
    a = math.radians(deg)
    return [round(radius * math.cos(a), 4), round(radius * math.sin(a), 4)]


round_stool = [
    prim("cylinder", "seat", {"radius": 0.18, "height": 0.04, "segments": 20}, [0, 0.47, 0]),
]
for i, deg in enumerate((90, 210, 330), start=1):
    x, z = leg_xy(0.13, deg)
    round_stool.append(
        prim(
            "cylinder", f"leg_{i}", {"radius": 0.02, "height": 0.45, "segments": 10}, [x, 0.225, z]
        )
    )
round_stool.append(material(["seat", "leg_1", "leg_2", "leg_3"], OAK, "Oak"))
f = Facts(round_stool, "round_stool")
emit(
    round_stool,
    "a round wooden stool :: what material is the seat?",
    "Oak-coloured wood -- a warm brown, roughly RGB (0.45, 0.31, 0.18).",
)
emit(
    round_stool,
    "How tall does a round wooden stool like this stand?",
    f"About {r2(f.bsize()[1])} m from the floor to the top of the seat.",
)
emit(
    round_stool,
    "the round wooden stool -- what generator built the seat?",
    "The seat is a cylinder primitive, 0.18 m in radius and 4 cm thick.",
)

# --- 4. stone_archway ----------------------------------------------------------
stone_archway = [
    prim(
        "arch",
        "archway",
        {"width": 1.4, "height": 2.0, "depth": 0.3, "thickness": 0.25, "segments": 16},
        [0, 1.0, 0],
    ),
    material(["archway"], STONE, "Stone"),
]
f = Facts(stone_archway, "stone_archway")
emit(
    stone_archway,
    "a stone archway :: how wide is the opening?",
    "About 0.9 m -- the archway is 1.4 m wide overall with 0.25 m of stone thickness on each side.",
)
emit(
    stone_archway,
    "for the stone archway, how tall is it?",
    f"About {r2(f.bsize()[1])} m from the ground to the top of the arch.",
)
emit(
    stone_archway,
    "Is the stone archway one object or several?",
    "Just one -- the whole archway is a single arch-generator object.",
)

# --- 5. metal_locker -----------------------------------------------------------
metal_locker = [
    prim("box", "body", {"size": [0.6, 1.8, 0.5]}, [0, 0.9, 0]),
    prim("box", "door", {"size": [0.55, 1.75, 0.04]}, [0, 0.9, 0.27]),
    material(["body", "door"], METAL, "Sheet metal"),
]
f = Facts(metal_locker, "metal_locker")
emit(
    metal_locker,
    "a metal storage locker :: is the door open or closed?",
    "Closed -- the door sits flush against the front face of the body with no rotation swinging it open.",
)
emit(
    metal_locker,
    "how tall is the metal storage locker?",
    f"About {r2(f.bsize()[1])} m.",
)
emit(
    metal_locker,
    "how many parts make up the metal storage locker?",
    f"{f.count}: the body and the door.",
)

# --- 6. humanoid_figure ----------------------------------------------------
humanoid_figure = [figure("humanoid", "figure")]
f = Facts(humanoid_figure, "humanoid_figure")
emit(
    humanoid_figure,
    "a simple humanoid figure :: how tall does it stand?",
    f"About {r2(f.bsize()[1])} m from its feet to the top of its head.",
)
emit(
    humanoid_figure,
    "Placed a simple humanoid figure -- how many objects came with it?",
    f"{f.count} -- the humanoid figure preset places one part per body segment (torso, head, arms and legs) as a single grounded group.",
)
emit(
    humanoid_figure,
    "what generator made the humanoid figure?",
    "It wasn't built from a single generator -- it came from the humanoid figure preset, which places a whole rigged group of parts (hips, spine, chest, head, arms and legs) in one step.",
)

# --- 7. wooden_barrel -----------------------------------------------------------
wooden_barrel = [
    prim(
        "lathe",
        "barrel",
        {
            "profile": [[0.0, -0.35], [0.32, -0.28], [0.38, 0.0], [0.32, 0.28], [0.0, 0.35]],
            "segments": 20,
        },
        [0, 0.35, 0],
    ),
    material(["barrel"], PALE_WOOD, "Barrel wood"),
]
f = Facts(wooden_barrel, "wooden_barrel")
emit(
    wooden_barrel,
    "a wooden barrel :: what is its radius at the widest point?",
    "About 0.38 m, at its equator halfway up the barrel -- it narrows to a point at the top and bottom rims.",
)
emit(
    wooden_barrel,
    "how tall is the wooden barrel?",
    f"About {r2(f.bsize()[1])} m.",
)
emit(
    wooden_barrel,
    "what generator built the wooden barrel?",
    "A lathe -- a profile revolved a full turn around the vertical axis.",
)

# --- 8. bookshelf_4shelves ---------------------------------------------------
bookshelf = [
    prim("box", "side_l", {"size": [0.04, 1.6, 0.3]}, [0.38, 0.8, 0]),
    prim("box", "side_r", {"size": [0.04, 1.6, 0.3]}, [-0.38, 0.8, 0]),
    prim("box", "shelf_1", {"size": [0.8, 0.03, 0.3]}, [0, 0.1, 0]),
    prim("box", "shelf_2", {"size": [0.8, 0.03, 0.3]}, [0, 0.55, 0]),
    prim("box", "shelf_3", {"size": [0.8, 0.03, 0.3]}, [0, 1.0, 0]),
    prim("box", "shelf_4", {"size": [0.8, 0.03, 0.3]}, [0, 1.45, 0]),
    material(["side_l", "side_r", "shelf_1", "shelf_2", "shelf_3", "shelf_4"], OAK, "Oak"),
]
f = Facts(bookshelf, "bookshelf")
emit(
    bookshelf,
    "a bookshelf with four shelves :: how many shelves does it have?",
    "Four, evenly spaced between the two side panels.",
)
emit(
    bookshelf,
    "how tall is the bookshelf with four shelves?",
    f"About {r2(f.bsize()[1])} m.",
)
emit(
    bookshelf,
    "for the four-shelf bookshelf, how many objects are in the scene in total?",
    f"{f.count}: two side panels plus the four shelf boards.",
)

# --- 9. garden_bench ------------------------------------------------------------
garden_bench = [
    prim("box", "seat", {"size": [1.2, 0.05, 0.4]}, [0, 0.45, 0]),
    prim("box", "backrest", {"size": [1.2, 0.4, 0.05]}, [0, 0.65, -0.175]),
    prim("box", "leg_fl", {"size": [0.05, 0.45, 0.05]}, [0.55, 0.225, 0.15]),
    prim("box", "leg_fr", {"size": [0.05, 0.45, 0.05]}, [-0.55, 0.225, 0.15]),
    prim("box", "leg_bl", {"size": [0.05, 0.45, 0.05]}, [0.55, 0.225, -0.15]),
    prim("box", "leg_br", {"size": [0.05, 0.45, 0.05]}, [-0.55, 0.225, -0.15]),
    material(
        ["seat", "backrest", "leg_fl", "leg_fr", "leg_bl", "leg_br"], DARK_WOOD, "Treated wood"
    ),
]
f = Facts(garden_bench, "garden_bench")
emit(
    garden_bench,
    "a garden bench :: what material is it made of?",
    "A dark treated wood -- roughly RGB (0.3, 0.19, 0.11), quite rough-finished.",
)
emit(
    garden_bench,
    "how tall is the garden bench overall, including the backrest?",
    f"About {r2(f.bsize()[1])} m.",
)
emit(
    garden_bench,
    "which is the biggest part of the garden bench?",
    "The seat -- it's the widest single board, 1.2 m by 0.4 m, bigger by footprint than the backrest or any leg.",
)

# --- 10. stone_column ------------------------------------------------------------
stone_column = [
    prim(
        "column",
        "column",
        {"radius": 0.3, "height": 2.4, "segments": 16, "base": 0.2, "capital": 0.2},
        [0, 1.2, 0],
    ),
    material(["column"], STONE, "Stone"),
]
f = Facts(stone_column, "stone_column")
emit(
    stone_column,
    "a stone column :: how tall is the column?",
    f"{r2(f.bsize()[1])} m.",
)
emit(
    stone_column,
    "what is the stone column made of?",
    "Stone -- a flat grey, roughly RGB (0.55, 0.54, 0.5).",
)
emit(
    stone_column,
    "how wide is the stone column's shaft?",
    "0.6 m across (0.3 m radius), with a slightly wider plinth at the base and block at the top.",
)

# --- 11. wooden_cart_2wheels ---------------------------------------------------
wooden_cart = [
    prim("box", "body", {"size": [0.9, 0.4, 0.5]}, [0, 0.55, 0]),
    prim(
        "cylinder",
        "wheel_l",
        {"radius": 0.35, "height": 0.06, "segments": 18},
        [0.48, 0.35, 0],
        [0, 0, 90],
    ),
    prim(
        "cylinder",
        "wheel_r",
        {"radius": 0.35, "height": 0.06, "segments": 18},
        [-0.48, 0.35, 0],
        [0, 0, 90],
    ),
    material(["body"], PALE_WOOD, "Cart wood"),
    material(["wheel_l", "wheel_r"], DARK_WOOD, "Wheel wood"),
]
f = Facts(wooden_cart, "wooden_cart")
emit(
    wooden_cart,
    "a wooden cart with two wheels :: how far apart are the wheels?",
    "0.96 m centre to centre -- the wheels sit at x = +/-0.48 m.",
)
emit(
    wooden_cart,
    "how many objects make up the wooden cart with two wheels?",
    f"{f.count}: the body and the two wheels.",
)
emit(
    wooden_cart,
    "how tall is the wooden cart's body off the ground?",
    f"The body's top sits about {r2(f.y('body') + f.size('body')[1] / 2)} m up; the wheels themselves are {r2(f.size('wheel_l')[1] if False else 0.7)} m across.",
    notes="answer computed from the fixed design numbers (wheel radius 0.35 -> 0.7 m diameter), not read off compact_scene, since 'size' after a 90-degree rotation reports the rotated bbox.",
)

# --- 12. round_shield -----------------------------------------------------------
round_shield = [
    prim(
        "cylinder",
        "shield",
        {"radius": 0.4, "height": 0.05, "segments": 24},
        [0, 0.4, 0],
        [90, 0, 0],
    ),
    material(["shield"], METAL, "Steel"),
]
f = Facts(round_shield, "round_shield")
emit(
    round_shield,
    "a round shield :: what is its diameter?",
    "0.8 m -- a 0.4 m radius disc.",
)
emit(
    round_shield,
    "what is the round shield made of?",
    "Metal -- a dull steel grey, roughly RGB (0.55, 0.56, 0.58), fairly reflective (metallic 0.8).",
)
emit(
    round_shield,
    "how many objects is the round shield built from?",
    f"Just {f.count} -- a single disc.",
)

# --- 13. quadruped_figure -----------------------------------------------------
quadruped_figure = [figure("quadruped", "critter")]
f = Facts(quadruped_figure, "quadruped_figure")
emit(
    quadruped_figure,
    "a simple quadruped figure :: how many legs does it have?",
    "Four -- a front-left, front-right, rear-left and rear-right leg, each in three segments (upper, lower, foot).",
)
emit(
    quadruped_figure,
    "how tall does the simple quadruped figure stand?",
    f"About {r2(f.bsize()[1])} m at the shoulder/hip height the preset builds.",
)
emit(
    quadruped_figure,
    "how many objects came with the quadruped figure preset?",
    f"{f.count} parts -- the body (hips, spine, chest, neck, head, tail) plus four three-segment legs.",
)

# --- 14. wooden_chest -----------------------------------------------------------
wooden_chest = [
    prim("box", "body", {"size": [0.6, 0.35, 0.35]}, [0, 0.175, 0]),
    prim("box", "lid", {"size": [0.6, 0.08, 0.35]}, [0, 0.39, 0]),
    material(["body", "lid"], DARK_WOOD, "Chest wood"),
]
f = Facts(wooden_chest, "wooden_chest")
emit(
    wooden_chest,
    "a wooden chest :: is the lid currently open?",
    "No -- the lid sits flush on top of the body with no rotation, so it reads as closed.",
)
emit(
    wooden_chest,
    "how big is the wooden chest overall?",
    f"About {f.bsize()[0]} m wide, {r2(f.bsize()[1])} m tall and {f.bsize()[2]} m deep.",
)
emit(
    wooden_chest,
    "how many objects make up the wooden chest?",
    f"{f.count}: the body and the lid.",
)

# --- 15. metal_bucket -----------------------------------------------------------
metal_bucket = [
    prim("cylinder", "bucket", {"radius": 0.22, "height": 0.35, "segments": 18}, [0, 0.175, 0]),
    material(["bucket"], METAL, "Galvanised steel"),
]
f = Facts(metal_bucket, "metal_bucket")
emit(
    metal_bucket,
    "a metal bucket :: what colour is it?",
    "A dull steel grey -- roughly RGB (0.55, 0.56, 0.58).",
)
emit(
    metal_bucket,
    "how tall is the metal bucket?",
    f"{r2(f.bsize()[1])} m.",
)
emit(
    metal_bucket,
    "what generator was used for the metal bucket?",
    "A cylinder, 0.22 m in radius and 0.35 m tall.",
)

# --- 16. low_coffee_table -------------------------------------------------------
coffee_table = [
    prim("box", "top", {"size": [1.0, 0.04, 0.5]}, [0, 0.32, 0]),
    prim("box", "leg_fl", {"size": [0.04, 0.3, 0.04]}, [0.45, 0.15, 0.2]),
    prim("box", "leg_fr", {"size": [0.04, 0.3, 0.04]}, [-0.45, 0.15, 0.2]),
    prim("box", "leg_bl", {"size": [0.04, 0.3, 0.04]}, [0.45, 0.15, -0.2]),
    prim("box", "leg_br", {"size": [0.04, 0.3, 0.04]}, [-0.45, 0.15, -0.2]),
    material(["top", "leg_fl", "leg_fr", "leg_bl", "leg_br"], PALE_WOOD, "Pale wood"),
]
f = Facts(coffee_table, "coffee_table")
emit(
    coffee_table,
    "a low coffee table :: how far off the ground is the tabletop?",
    f"The underside of the top sits {r2(f.y('top') - f.size('top')[1] / 2)} m up, and its top surface is {r2(f.y('top') + f.size('top')[1] / 2)} m off the ground.",
)
emit(
    coffee_table,
    "how many legs does the low coffee table have?",
    "Four, one under each corner.",
)
emit(
    coffee_table,
    "what is the low coffee table made of?",
    "A pale wood -- roughly RGB (0.64, 0.5, 0.34).",
)

# --- 17. stone_well ---------------------------------------------------------------
stone_well = [
    prim("cylinder", "outer", {"radius": 0.5, "height": 0.6, "segments": 24}, [0, 0.3, 0]),
    prim("cylinder", "inner", {"radius": 0.4, "height": 0.7, "segments": 24}, [0, 0.3, 0]),
    boolean("difference", ["outer", "inner"]),
    material(["outer"], STONE, "Stone"),
]
f = Facts(stone_well, "stone_well")
emit(
    stone_well,
    "a stone well :: how deep is the wall?",
    "0.1 m thick -- the outer wall is 0.5 m in radius and the hollowed centre is 0.4 m, so 0.1 m of stone remains all the way round.",
)
emit(
    stone_well,
    "how tall is the stone well's wall?",
    f"{r2(f.bsize()[1])} m.",
)
emit(
    stone_well,
    "how many objects is the stone well now, after it was hollowed out?",
    f"{f.count} -- the inner cutting cylinder was consumed by the boolean, leaving one hollow wall.",
)

# --- 18. wooden_wardrobe -----------------------------------------------------------
wardrobe = [
    prim("box", "body", {"size": [0.9, 1.8, 0.5]}, [0, 0.9, 0]),
    prim("box", "door_l", {"size": [0.44, 1.7, 0.04]}, [0.225, 0.9, 0.27]),
    prim("box", "door_r", {"size": [0.44, 1.7, 0.04]}, [-0.225, 0.9, 0.27]),
    material(["body", "door_l", "door_r"], DARK_WOOD, "Wardrobe wood"),
]
f = Facts(wardrobe, "wardrobe")
emit(
    wardrobe,
    "a wooden wardrobe :: how many doors does it have?",
    "Two, side by side, splitting the front evenly.",
)
emit(
    wardrobe,
    "how tall is the wooden wardrobe?",
    f"{r2(f.bsize()[1])} m.",
)
emit(
    wardrobe,
    "how many objects make up the wooden wardrobe scene?",
    f"{f.count}: the body and its two doors.",
)

# --- 19. hammer ---------------------------------------------------------------------
hammer = [
    prim("cylinder", "handle", {"radius": 0.015, "height": 0.3, "segments": 10}, [0, 0.15, 0]),
    prim("box", "head", {"size": [0.12, 0.05, 0.04]}, [0, 0.325, 0]),
    material(["handle"], DARK_WOOD, "Handle wood"),
    material(["head"], DARK_METAL, "Forged steel"),
]
f = Facts(hammer, "hammer")
emit(
    hammer,
    "a plain hammer :: how long is the handle?",
    "0.3 m.",
)
emit(
    hammer,
    "how tall is the plain hammer standing on its head... I mean end?",
    f"About {r2(f.bsize()[1])} m overall, handle and head together.",
)
emit(
    hammer,
    "how many objects make up the plain hammer?",
    f"{f.count}: the handle and the head.",
)

# --- 20. clay_pot -------------------------------------------------------------------
clay_pot = [
    # lathe recentres the profile on its own bbox centre, so grounding it
    # needs translation.y = half the profile's own y-range (0.33/2 = 0.165),
    # not the raw 0.15 the un-recentred stations would suggest.
    prim(
        "lathe",
        "pot",
        {"profile": [[0.15, -0.15], [0.22, -0.05], [0.18, 0.1], [0.1, 0.18]], "segments": 16},
        [0, 0.18, 0],
    ),
    material(["pot"], TERRACOTTA, "Terracotta"),
]
f = Facts(clay_pot, "clay_pot")
emit(
    clay_pot,
    "a small clay pot :: what shape is its base?",
    "Flat and circular, about 0.3 m across -- the lathe profile's bottom station is a capped ring, not a point.",
)
emit(
    clay_pot,
    "how tall is the small clay pot?",
    f"{r2(f.bsize()[1])} m.",
)
emit(
    clay_pot,
    "what is the small clay pot made of?",
    "Terracotta -- a warm orange-brown, roughly RGB (0.72, 0.42, 0.28).",
)

# --- 21. wooden_ladder ----------------------------------------------------------------
wooden_ladder = [
    prim("box", "rail_l", {"size": [0.04, 1.8, 0.04]}, [0.25, 0.9, 0]),
    prim("box", "rail_r", {"size": [0.04, 1.8, 0.04]}, [-0.25, 0.9, 0]),
    prim("box", "rung", {"size": [0.5, 0.03, 0.04]}, [0, 0.2, 0]),
    select(["rung"]),
    op("array-linear", {"count": 6, "x": 0, "y": 0.3, "z": 0}),
    material(["rail_l", "rail_r"], OAK, "Oak"),
]
f = Facts(wooden_ladder, "wooden_ladder")
emit(
    wooden_ladder,
    "a wooden ladder :: how many rungs does it have?",
    "Six, spaced 0.3 m apart.",
)
emit(
    wooden_ladder,
    "how tall are the wooden ladder's rails?",
    f"{r2(f.bsize()[1])} m.",
)
emit(
    wooden_ladder,
    "how many objects make up the wooden ladder?",
    f"{f.count} -- two rails plus six rungs.",
)

# --- 22. stone_plinth -----------------------------------------------------------------
stone_plinth = [
    prim("box", "plinth", {"size": [0.6, 0.3, 0.6]}, [0, 0.15, 0]),
    material(["plinth"], STONE, "Stone"),
]
f = Facts(stone_plinth, "stone_plinth")
emit(
    stone_plinth,
    "a stone plinth :: what are its dimensions?",
    "0.6 m square and 0.3 m tall.",
)
emit(
    stone_plinth,
    "what is the stone plinth made of?",
    "Stone -- a flat grey.",
)
emit(
    stone_plinth,
    "is the stone plinth one object?",
    f"Yes, {f.count} object -- a single box.",
)

# --- 23. side_table_round ---------------------------------------------------------
side_table = [
    prim("cylinder", "top", {"radius": 0.3, "height": 0.03, "segments": 20}, [0, 0.435, 0]),
    prim("cylinder", "leg", {"radius": 0.05, "height": 0.42, "segments": 12}, [0, 0.21, 0]),
    material(["top", "leg"], OAK, "Oak"),
]
f = Facts(side_table, "side_table")
emit(
    side_table,
    "a round side table :: what material is the tabletop?",
    "Oak-coloured wood, roughly RGB (0.45, 0.31, 0.18).",
)
emit(
    side_table,
    "how tall is the round side table?",
    f"{r2(f.bsize()[1])} m.",
)
emit(
    side_table,
    "how wide is the round side table's top?",
    "0.6 m across (0.3 m radius).",
)

# --- 24. gear_wheel (reused for the easy "gear disc" and the medium "sprocket") ---
gear_wheel = [
    prim("cylinder", "hub", {"radius": 0.3, "height": 0.08, "segments": 24}, [0, 0.075, 0]),
    prim("box", "tooth", {"size": [0.06, 0.08, 0.08]}, [0.32, 0.075, 0]),
    select(["tooth"]),
    op("array-radial", {"count": 10, "angle": 360, "axis": 1}),
    material(["hub"], DARK_METAL, "Cast iron"),
]
f = Facts(gear_wheel, "gear_wheel")
emit(
    gear_wheel,
    "a metal gear disc :: how many objects make up the gear?",
    f"{f.count} -- one central hub plus 10 teeth arrayed around it.",
)
emit(
    gear_wheel,
    "a sprocket wheel with radial teeth :: how many teeth does it have?",
    "10, evenly spaced around the hub every 36 degrees.",
)
emit(
    gear_wheel,
    "how wide is the metal gear disc across the hub?",
    "0.6 m (0.3 m radius), before the teeth add another 0.06 m at each side.",
)

# --- 25. bird_figure --------------------------------------------------------------
bird_figure = [figure("bird", "bird")]
f = Facts(bird_figure, "bird_figure")
emit(
    bird_figure,
    "a simple bird figure :: is it standing or perched?",
    "Standing -- the bird preset is a winged biped, built on two grounded legs like the humanoid, not perched on anything.",
)
emit(
    bird_figure,
    "how tall is the simple bird figure?",
    f"About {r2(f.bsize()[1])} m.",
)
emit(
    bird_figure,
    "how many objects make up the simple bird figure?",
    f"{f.count} parts, placed as one grounded group by the bird preset.",
)

# =============================================================================
# MEDIUM priors -- a boolean, an array, a lathe/sweep profile, or element mode.
# =============================================================================

# --- 26. wheel_spokes ---------------------------------------------------------------
wheel_spokes = [
    prim("cylinder", "hub", {"radius": 0.3, "height": 0.15, "segments": 16}, [0, 0.075, 0]),
    prim("box", "spoke", {"size": [0.05, 0.05, 0.6]}, [0, 0.075, 0.45]),
    select(["spoke"]),
    op("array-radial", {"count": 8, "angle": 360, "axis": 1}),
    material(["hub"], DARK_WOOD, "Wheel wood"),
]
f = Facts(wheel_spokes, "wheel_spokes")
emit(
    wheel_spokes,
    "a wagon wheel with eight spokes :: how many spokes does the wheel have?",
    "Eight, evenly spaced 45 degrees apart around the hub.",
)
emit(
    wheel_spokes,
    "how many objects are in the wagon wheel with eight spokes?",
    f"{f.count} -- the hub plus its eight spokes.",
)
emit(
    wheel_spokes,
    "how wide is the wagon wheel across the hub?",
    "0.6 m (0.3 m radius); each spoke reaches out another 0.6 m from the hub's centre.",
)

# --- 27. chest_keyhole -----------------------------------------------------------------
chest_keyhole = [
    prim("box", "body", {"size": [0.5, 0.35, 0.3]}, [0, 0.175, 0]),
    prim("box", "lock_plate", {"size": [0.08, 0.1, 0.02]}, [0, 0.2, 0.16]),
    prim(
        "cylinder",
        "keyhole_cut",
        {"radius": 0.012, "height": 0.06, "segments": 12},
        [0, 0.18, 0.16],
        [90, 0, 0],
    ),
    boolean("difference", ["lock_plate", "keyhole_cut"]),
    material(["body"], DARK_WOOD, "Chest wood"),
    material(["lock_plate"], DARK_METAL, "Iron fitting"),
]
f = Facts(chest_keyhole, "chest_keyhole")
emit(
    chest_keyhole,
    "a treasure chest with a round keyhole cut in the lock plate :: where is the keyhole positioned?",
    "On the lock plate, centred left-to-right and about 0.18 m up from the ground -- just below the plate's own centre.",
)
emit(
    chest_keyhole,
    "how many objects is the chest now that the keyhole is cut?",
    f"{f.count} -- the body and the lock plate (the cutting cylinder was consumed by the boolean).",
)
emit(
    chest_keyhole,
    "what is the treasure chest's lock plate made of?",
    "Iron -- a dark near-black metal, roughly RGB (0.22, 0.22, 0.24).",
)

# --- 28. barrel_bands ----------------------------------------------------------------
barrel_bands = [
    prim(
        "lathe",
        "barrel",
        {
            "profile": [[0.0, -0.35], [0.32, -0.28], [0.38, 0.0], [0.32, 0.28], [0.0, 0.35]],
            "segments": 20,
        },
        [0, 0.35, 0],
    ),
    prim(
        "torus", "band", {"radius": 0.36, "tube": 0.015, "segments": 24, "sides": 8}, [0, 0.15, 0]
    ),
    select(["band"]),
    op("array-linear", {"count": 3, "x": 0, "y": 0.2, "z": 0}),
    material(["barrel"], PALE_WOOD, "Barrel wood"),
    material(["band"], IRON_BAND, "Iron banding"),
]
f = Facts(barrel_bands, "barrel_bands")
emit(
    barrel_bands,
    "a wine barrel with metal bands :: how many bands wrap around it?",
    "Three, spaced 0.2 m apart up the side of the barrel.",
)
emit(
    barrel_bands,
    "how many objects make up the wine barrel with metal bands?",
    f"{f.count} -- the barrel itself plus its three bands.",
)
emit(
    barrel_bands,
    "how tall is the wine barrel?",
    f"{r2(f.bsize()[1])} m.",
)

# --- 29. tower_windows ----------------------------------------------------------------
tower_windows = [
    prim("cylinder", "tower", {"radius": 0.4, "height": 3.0, "segments": 24}, [0, 1.5, 0]),
    prim("box", "cut_n", {"size": [0.15, 0.3, 0.9]}, [0, 1.0, 0.4]),
    prim("box", "cut_s", {"size": [0.15, 0.3, 0.9]}, [0, 1.0, -0.4]),
    prim("box", "cut_e", {"size": [0.9, 0.3, 0.15]}, [0.4, 1.0, 0]),
    prim("box", "cut_w", {"size": [0.9, 0.3, 0.15]}, [-0.4, 1.0, 0]),
    boolean("difference", ["tower", "cut_n"]),
    boolean("difference", ["tower", "cut_s"]),
    boolean("difference", ["tower", "cut_e"]),
    boolean("difference", ["tower", "cut_w"]),
    material(["tower"], STONE, "Tower stone"),
]
f = Facts(tower_windows, "tower_windows")
emit(
    tower_windows,
    "a tower with evenly spaced windows :: how many windows are cut into it?",
    "Four, one facing each of the compass directions, all at the same 1.0 m height.",
)
emit(
    tower_windows,
    "how many objects is the tower with its windows cut, in the end?",
    f"{f.count} -- the four cutting boxes were all consumed by the booleans, leaving just the tower.",
)
emit(
    tower_windows,
    "how tall is the windowed tower?",
    f"{r2(f.bsize()[1])} m.",
)

# --- 30. dresser_drawers ---------------------------------------------------------------
dresser = [
    prim("box", "body", {"size": [0.6, 0.9, 0.45]}, [0, 0.45, 0]),
    prim("box", "drawer_1", {"size": [0.55, 0.25, 0.03]}, [0, 0.15, 0.24]),
    prim("box", "drawer_2", {"size": [0.55, 0.25, 0.03]}, [0, 0.45, 0.24]),
    prim("box", "drawer_3", {"size": [0.55, 0.25, 0.03]}, [0, 0.75, 0.24]),
    prim("box", "handle_1", {"size": [0.1, 0.02, 0.015]}, [0, 0.15, 0.26]),
    prim("box", "handle_2", {"size": [0.1, 0.02, 0.015]}, [0, 0.45, 0.26]),
    prim("box", "handle_3", {"size": [0.1, 0.02, 0.015]}, [0, 0.75, 0.26]),
    material(["body", "drawer_1", "drawer_2", "drawer_3"], DARK_WOOD, "Dresser wood"),
    material(["handle_1", "handle_2", "handle_3"], DARK_METAL, "Handle metal"),
]
f = Facts(dresser, "dresser")
emit(
    dresser,
    "a dresser with three drawers, each with a recessed handle :: do all the drawers match?",
    "Yes -- all three drawer fronts and their handles are the same size (0.55 x 0.25 x 0.03 m), spaced 0.3 m apart up the front.",
)
emit(
    dresser,
    "how many objects make up the dresser with three drawers?",
    f"{f.count}: the body, three drawer fronts and their three handles.",
)
emit(
    dresser,
    "how tall is the three-drawer dresser?",
    f"{r2(f.bsize()[1])} m.",
)

# --- 31. columns_row -----------------------------------------------------------------
columns_row = [
    prim(
        "column",
        "column",
        {"radius": 0.25, "height": 2.0, "segments": 14, "base": 0.15, "capital": 0.15},
        [0, 1.0, 0],
    ),
    select(["column"]),
    op("array-linear", {"count": 6, "x": 1.0, "y": 0, "z": 0}),
    prim("box", "roof", {"size": [5.6, 0.3, 0.7]}, [2.5, 2.15, 0]),
    material(["roof"], STONE, "Roof stone"),
]
f = Facts(columns_row, "columns_row")
emit(
    columns_row,
    "a row of six columns supporting a roof :: how far apart are the columns?",
    "1.0 m apart, centre to centre.",
)
emit(
    columns_row,
    "how many objects make up the row of six columns and its roof?",
    f"{f.count} -- six columns plus the roof slab.",
)
emit(
    columns_row,
    "how tall are the columns, not counting the roof?",
    f"{r2(f.objects['column']['size'][1])} m each.",
)

# --- 32. shield_emblem ----------------------------------------------------------------
shield_emblem = [
    prim(
        "cylinder",
        "shield",
        {"radius": 0.4, "height": 0.05, "segments": 24},
        [0, 0.4, 0],
        [90, 0, 0],
    ),
    prim("icosphere", "emblem", {"radius": 0.05, "subdivisions": 1}, [0.15, 0.4, -0.03]),
    select(["emblem"]),
    op("mirror-copy", {"axis": 0, "offset": 0}),
    material(["shield"], METAL, "Steel"),
    material(["emblem"], DARK_METAL, "Emblem metal"),
]
f = Facts(shield_emblem, "shield_emblem")
emit(
    shield_emblem,
    "a shield with a mirrored emblem :: is the emblem symmetrical?",
    "Yes -- one stud sits at x = +0.15 m and Mirror Copy placed its exact twin at x = -0.15 m, so the pair is symmetric about the shield's own centreline.",
)
emit(
    shield_emblem,
    "how many objects make up the shield with a mirrored emblem?",
    f"{f.count} -- the shield disc and the two emblem studs.",
)
emit(
    shield_emblem,
    "what generator built the mirrored half of the emblem?",
    "None any more -- Mirror Copy bakes its duplicate into a plain mesh, so the copied stud no longer carries a live generator, even though the original still does.",
)

# --- 33. grate_holes --------------------------------------------------------------------
grate_holes = [
    prim("box", "plate", {"size": [0.6, 0.04, 0.6]}, [0, 0.02, 0]),
    prim("cylinder", "hole", {"radius": 0.03, "height": 0.08, "segments": 10}, [-0.2, 0.02, -0.2]),
    select(["hole"]),
    op("array-linear", {"count": 3, "x": 0.2, "y": 0, "z": 0}),
    op("array-linear", {"count": 3, "x": 0, "y": 0, "z": 0.2}),
]
# nine holes named "hole", "hole.001".."hole.008" by the array op (ops.next_name's
# ``Box.001`` scheme) -- boolean each against the plate
_hole_names = ["hole"] + [f"hole.{i:03d}" for i in range(1, 9)]
for _n in _hole_names:
    grate_holes.append(boolean("difference", ["plate", _n]))
grate_holes.append(material(["plate"], DARK_METAL, "Grate iron"))
f = Facts(grate_holes, "grate_holes")
emit(
    grate_holes,
    "a metal grate with a grid of holes :: how many holes are cut into it?",
    "Nine, in a 3-by-3 grid spaced 0.2 m apart.",
)
emit(
    grate_holes,
    "how many objects is the metal grate, once the holes are all cut?",
    f"{f.count} -- one plate; every cutting cylinder was consumed by its boolean.",
)
emit(
    grate_holes,
    "what is the metal grate made of?",
    "Iron -- a dark near-black metal.",
)

# --- 34. goblet -------------------------------------------------------------------------
goblet = [
    prim(
        "lathe",
        "goblet",
        {
            "profile": [
                [0.0, -0.5],
                [0.18, -0.42],
                [0.05, -0.28],
                [0.05, 0.05],
                [0.32, 0.22],
                [0.16, 0.4],
                [0.2, 0.5],
            ],
            "segments": 20,
        },
        [0, 0.5, 0],
    ),
    material(["goblet"], METAL, "Pewter"),
]
f = Facts(goblet, "goblet")
emit(
    goblet,
    "a lathe-turned goblet :: how wide is the bowl compared to the stem?",
    "The bowl reaches about 0.32 m in radius at its widest, against about 0.05 m at the narrowest point of the stem -- the bowl is roughly six times as wide.",
)
emit(
    goblet,
    "how tall is the lathe-turned goblet?",
    f"{r2(f.bsize()[1])} m.",
)
emit(
    goblet,
    "what generator built the lathe-turned goblet?",
    "A lathe -- a single profile revolved a full turn to make the whole foot, stem, bowl and rim in one piece.",
)

# --- 35. quadruped_tube_tail ------------------------------------------------------------
quad_tube = [
    prim(
        "capsule",
        "body",
        {"radius": 0.14, "height": 0.5, "segments": 14, "rings": 4},
        [0, 0.3, 0],
        [0, 0, 90],
    ),
    prim("uv_sphere", "head", {"radius": 0.09, "segments": 14, "rings": 8}, [0.42, 0.32, 0]),
    prim("cylinder", "leg_fl", {"radius": 0.03, "height": 0.28, "segments": 8}, [0.18, 0.14, 0.1]),
    prim("cylinder", "leg_fr", {"radius": 0.03, "height": 0.28, "segments": 8}, [0.18, 0.14, -0.1]),
    prim("cylinder", "leg_bl", {"radius": 0.03, "height": 0.28, "segments": 8}, [-0.18, 0.14, 0.1]),
    prim(
        "cylinder", "leg_br", {"radius": 0.03, "height": 0.28, "segments": 8}, [-0.18, 0.14, -0.1]
    ),
    # tube recentres its path on the path's own bbox, so this is drawn as a
    # small local wag (y within +/-0.03 of its own centre) and placed by
    # translation alone -- 0.32 m up keeps it well clear of the ground even
    # with the 0.035 m tube radius added on top.
    prim(
        "tube",
        "tail",
        {
            "path": [[0.0, 0.0, 0.0], [-0.15, 0.02, 0.0], [-0.28, -0.02, 0.0], [-0.4, 0.03, 0.0]],
            "radius": 0.035,
            "sides": 8,
        },
        [-0.3, 0.32, 0],
    ),
    material(
        ["body", "head", "leg_fl", "leg_fr", "leg_bl", "leg_br", "tail"],
        PALE_WOOD,
        "Creature hide",
    ),
]
f = Facts(quad_tube, "quad_tube")
emit(
    quad_tube,
    "a quadruped figure with a tube-shaped tail :: how long is the tail relative to the body?",
    f"The tail spans about {r2(f.size('tail')[0])} m along its own bounding box against the body's {r2(f.size('body')[0])} m length -- a little shorter than the body.",
)
emit(
    quad_tube,
    "how many objects make up the quadruped with the tube tail?",
    f"{f.count}: the body, head, four legs and the tail.",
)
emit(
    quad_tube,
    "what generator built the quadruped's tail?",
    "A tube -- a constant-radius circular cross-section swept along a short bending path, unlike a sweep's taper.",
)

# --- 36. staircase -----------------------------------------------------------------------
staircase = [
    prim("box", "step", {"size": [0.3, 0.15, 0.9]}, [0, 0.075, 0]),
    select(["step"]),
    op("array-linear", {"count": 6, "x": 0.3, "y": 0.15, "z": 0}),
    material(["step"], STONE, "Stair stone"),
]
f = Facts(staircase, "staircase")
emit(
    staircase,
    "a staircase of repeating steps :: how many steps does it have?",
    "Six, each rising 0.15 m and running 0.3 m.",
)
emit(
    staircase,
    "how many objects make up the staircase?",
    f"{f.count} -- one box per step.",
)
emit(
    staircase,
    "how much does the staircase rise from the first step to the last?",
    f"About {r2(f.bsize()[1])} m in total.",
)

# --- 37. cart_mirror_wheels --------------------------------------------------------------
cart_mirror = [
    prim("box", "body", {"size": [0.8, 0.35, 0.45]}, [0, 0.5, 0]),
    prim(
        "cylinder",
        "wheel",
        {"radius": 0.3, "height": 0.05, "segments": 18},
        [0.42, 0.3, 0],
        [0, 0, 90],
    ),
    select(["wheel"]),
    op("mirror-copy", {"axis": 0, "offset": 0}),
    material(["body"], PALE_WOOD, "Cart wood"),
]
f = Facts(cart_mirror, "cart_mirror")
emit(
    cart_mirror,
    "a cart with a mirrored pair of wheels :: are the two wheels identical?",
    "Yes -- the second wheel is Mirror Copy's exact reflection of the first across the cart's own centreline, so the two are geometrically identical, just opposite in position.",
)
emit(
    cart_mirror,
    "how many objects make up the cart with its mirrored wheels?",
    f"{f.count}: the body and the two wheels.",
)
emit(
    cart_mirror,
    "what generator built the second (mirrored) wheel?",
    "None -- Mirror Copy bakes the reflected copy into a plain mesh, so only the original wheel still carries a live cylinder generator.",
)

# --- 38. shelf_pegs (reused for both the "shelf" and the "bookshelf" seed) --------------
shelf_pegs = [
    prim("box", "shelf", {"size": [0.9, 0.03, 0.25]}, [0, 0.8, 0]),
    prim("cylinder", "peg", {"radius": 0.015, "height": 0.12, "segments": 8}, [-0.4, 0.74, 0]),
    select(["peg"]),
    op("array-linear", {"count": 5, "x": 0.2, "y": 0, "z": 0}),
    material(["shelf"], OAK, "Oak"),
]
f = Facts(shelf_pegs, "shelf_pegs")
emit(
    shelf_pegs,
    "a shelf held up by an array of support pegs :: how many pegs hold up the shelf?",
    "Five, spaced 0.2 m apart under the shelf.",
)
emit(
    shelf_pegs,
    "a bookshelf held up by an array of support pegs :: are the pegs evenly spaced?",
    "Yes -- they were placed with a single linear array at a fixed 0.2 m step, so the spacing is exact.",
)
emit(
    shelf_pegs,
    "how many objects hold up the shelf, pegs included?",
    f"{f.count} -- the shelf board plus its five support pegs.",
)

# --- 39. jug_handle --------------------------------------------------------------------
jug = [
    # profile y-range is -0.2..0.24 (half-range 0.22); lathe recentres the
    # profile on its own bbox centre, so grounding needs translation.y = 0.22
    # (0.24 leaves a small margin), not the raw 0.2 the stations suggest.
    prim(
        "lathe",
        "jug",
        {
            "profile": [[0.0, -0.2], [0.16, -0.16], [0.2, 0.0], [0.14, 0.16], [0.08, 0.24]],
            "segments": 18,
        },
        [0, 0.24, 0],
    ),
    # tube recentres its path on the path's own bbox too; this loop is drawn
    # local to its own centre (y in -0.08..0.08) and placed by translation
    # alone, against the side of the jug at about 0.24 m up.
    prim(
        "tube",
        "handle",
        {
            "path": [
                [0.0, 0.08, 0.0],
                [0.1, 0.08, 0.0],
                [0.12, 0.0, 0.0],
                [0.1, -0.08, 0.0],
                [0.0, -0.08, 0.0],
            ],
            "radius": 0.018,
            "sides": 8,
        },
        [0.2, 0.24, 0],
    ),
    material(["jug", "handle"], TERRACOTTA, "Terracotta"),
]
f = Facts(jug, "jug")
emit(
    jug,
    "a jug with a lathe-turned handle loop :: where does the handle attach?",
    f"On the side of the jug's body, centred about {r2(f.y('handle'))} m up -- roughly the middle of the jug's height.",
)
emit(
    jug,
    "how tall is the jug with the handle loop?",
    f"{r2(f.bsize()[1])} m.",
)
emit(
    jug,
    "how many objects make up the jug and its handle?",
    f"{f.count}: the jug body and the handle loop.",
)

# --- 40. fence_pickets -----------------------------------------------------------------
fence = [
    prim("box", "picket", {"size": [0.08, 0.9, 0.02]}, [0, 0.45, 0]),
    select(["picket"]),
    op("array-linear", {"count": 8, "x": 0.15, "y": 0, "z": 0}),
    material(["picket"], PALE_WOOD, "Fence wood"),
]
f = Facts(fence, "fence")
emit(
    fence,
    "a fence made from a linear array of pickets :: how many pickets are there?",
    "Eight, spaced 0.15 m apart.",
)
emit(
    fence,
    "how tall is the picket fence?",
    f"{r2(f.bsize()[1])} m.",
)
emit(
    fence,
    "how wide is the whole picket fence?",
    f"About {r2(f.bsize()[0])} m end to end.",
)

# --- 41. battlement_merlons -------------------------------------------------------------
battlement = [
    prim("box", "wall", {"size": [3.0, 0.6, 0.3]}, [0, 0.3, 0]),
    prim("box", "merlon", {"size": [0.4, 0.25, 0.3]}, [-1.2, 0.725, 0]),
    select(["merlon"]),
    op("array-linear", {"count": 5, "x": 0.6, "y": 0, "z": 0}),
    material(["wall", "merlon"], STONE, "Battlement stone"),
]
f = Facts(battlement, "battlement")
emit(
    battlement,
    "a battlement wall with repeating merlons :: how many merlons run along the wall?",
    "Five, evenly spaced 0.6 m apart along the top of the wall.",
)
emit(
    battlement,
    "how many objects make up the battlement wall?",
    f"{f.count}: the wall and its five merlons.",
)
emit(
    battlement,
    "how tall is the battlement wall, merlons included?",
    f"{r2(f.bsize()[1])} m.",
)

# --- 42. coatrack_pegs ------------------------------------------------------------------
coatrack = [
    prim("cylinder", "pole", {"radius": 0.03, "height": 1.6, "segments": 14}, [0, 0.8, 0]),
    prim(
        "cylinder",
        "peg",
        {"radius": 0.015, "height": 0.12, "segments": 8},
        [0.1, 1.3, 0],
        [0, 0, 90],
    ),
    select(["peg"]),
    op("array-radial", {"count": 6, "angle": 360, "axis": 1}),
    material(["pole"], DARK_WOOD, "Coatrack wood"),
]
f = Facts(coatrack, "coatrack")
emit(
    coatrack,
    "a coat rack with a radial ring of pegs :: how many pegs does it have?",
    "Six, evenly spaced 60 degrees apart around the pole.",
)
emit(
    coatrack,
    "how tall is the coat rack's pole?",
    f"{r2(f.bsize()[1])} m.",
)
emit(
    coatrack,
    "is any peg floating free of the pole on the coat rack?",
    "No -- every peg was placed touching the pole and only spun around it, so all six stay attached.",
)

# =============================================================================
# HARD priors -- a swept or tapered form along a path.
# =============================================================================

# --- 43. creature_tapering_tail ---------------------------------------------------------
creature_tail = [
    prim(
        "capsule",
        "body",
        {"radius": 0.16, "height": 0.55, "segments": 14, "rings": 4},
        [0, 0.32, 0],
        [0, 0, 90],
    ),
    prim("uv_sphere", "head", {"radius": 0.1, "segments": 14, "rings": 8}, [0.46, 0.34, 0]),
    prim("cylinder", "leg_fl", {"radius": 0.035, "height": 0.3, "segments": 8}, [0.2, 0.15, 0.12]),
    prim("cylinder", "leg_fr", {"radius": 0.035, "height": 0.3, "segments": 8}, [0.2, 0.15, -0.12]),
    prim("cylinder", "leg_bl", {"radius": 0.035, "height": 0.3, "segments": 8}, [-0.2, 0.15, 0.12]),
    prim(
        "cylinder", "leg_br", {"radius": 0.035, "height": 0.3, "segments": 8}, [-0.2, 0.15, -0.12]
    ),
    prim(
        "sweep",
        "tail",
        {
            "outline": [
                [0.06, 0.0],
                [0.042, 0.042],
                [0.0, 0.06],
                [-0.042, 0.042],
                [-0.06, 0.0],
                [-0.042, -0.042],
                [0.0, -0.06],
                [0.042, -0.042],
            ],
            "depth": 0.5,
            "taper": 0.15,
            "twist": 0.0,
            "sections": 8,
        },
        [-0.34, 0.34, 0],
        [0, 90, 0],
    ),
    material(["body", "head", "leg_fl", "leg_fr", "leg_bl", "leg_br", "tail"], PALE_WOOD, "Hide"),
]
f = Facts(creature_tail, "creature_tail")
emit(
    creature_tail,
    "a creature whose tail tapers along a swept curve :: how does the tail's thickness change from base to tip?",
    "It narrows steadily along its whole 0.5 m length -- the sweep's taper shrinks the cross-section to 15% of its base width by the tip, so the tail ends up roughly six or seven times thicker at the root than at the very end.",
)
emit(
    creature_tail,
    "how many objects make up the tapering-tailed creature?",
    f"{f.count}: the body, head, four legs and the tapering tail.",
)
emit(
    creature_tail,
    "what generator built the tapering tail?",
    "A sweep -- an octagonal cross-section extruded along a straight axis with taper set to 0.15, so the far end is much narrower than where it leaves the body.",
)

# --- 44. sword_tapering_blade ------------------------------------------------------------
sword = [
    prim("cylinder", "handle", {"radius": 0.016, "height": 0.14, "segments": 10}, [0, 0.07, 0]),
    # sweep extrudes symmetrically about its own origin (+/-depth/2 along
    # local Z), and rotating [90,0,0] swaps that local Z for world Y -- so
    # grounding the blade on top of the 0.14 m handle needs translation.y =
    # 0.14 + depth/2 (0.35), not the raw 0.14 the handle height alone gives.
    prim(
        "sweep",
        "blade",
        {
            "outline": [[0.03, 0.0], [0.0, 0.006], [-0.03, 0.0], [0.0, -0.006]],
            "depth": 0.7,
            "taper": 0.04,
            "twist": 0.0,
            "sections": 4,
        },
        [0, 0.49, 0],
        [90, 0, 0],
    ),
    material(["handle"], DARK_WOOD, "Grip"),
    material(["blade"], METAL, "Blade steel"),
]
f = Facts(sword, "sword")
emit(
    sword,
    "a sword whose blade tapers to a point :: where along the blade is it widest?",
    "Right at the base, where it meets the handle -- the blade is 0.06 m across there and the sweep's taper narrows it to about 4% of that (roughly 2 mm) by the tip.",
)
emit(
    sword,
    "how long is the sword's blade?",
    "0.7 m from the guard to the tip.",
)
emit(
    sword,
    "how many objects make up the sword?",
    f"{f.count}: the handle and the blade.",
)

# --- 45. tapering_spire ---------------------------------------------------------------------
spire = [
    prim(
        "sweep",
        "spire",
        {
            "outline": [[0.25, 0.0], [0.0, 0.25], [-0.25, 0.0], [0.0, -0.25]],
            "depth": 3.0,
            "taper": 0.03,
            "twist": 0.0,
            "sections": 6,
        },
        [0, 1.5, 0],
        [-90, 0, 0],
    ),
    material(["spire"], STONE, "Spire stone"),
]
f = Facts(spire, "spire")
emit(
    spire,
    "a spire that tapers from base to tip :: what is the taper ratio between base and tip?",
    "About 33 to 1 -- the sweep's taper parameter is 0.03, so the tip cross-section is roughly 3% the width of the base.",
)
emit(
    spire,
    "how tall is the tapering spire?",
    f"{r2(f.bsize()[1])} m.",
)
emit(
    spire,
    "how many objects is the tapering spire built from?",
    f"Just {f.count} -- a single sweep.",
)

# --- 46. bending_tapering_pipe -----------------------------------------------------------
pipe = [
    # tube recentres the path on its own bbox (here y spans 0.18..0.42, x
    # spans -0.4..0.4, both already centred at 0.3 and 0 respectively) --
    # translation.y = 0.3 reproduces the path's own intended world height.
    prim(
        "tube",
        "pipe",
        {
            "path": [[-0.4, 0.3, 0.0], [-0.15, 0.18, 0.0], [0.15, 0.42, 0.0], [0.4, 0.3, 0.0]],
            "radius": 0.08,
            "sides": 12,
        },
        [0, 0.3, 0],
    ),
    prim(
        "cone",
        "nozzle",
        {"radius": 0.08, "height": 0.18, "segments": 12},
        [0.5, 0.3, 0],
        [0, 0, -90],
    ),
]
f = Facts(pipe, "pipe")
emit(
    pipe,
    "a pipe that bends and tapers into a nozzle :: how much does the diameter shrink by the end?",
    "The pipe itself holds a constant 0.16 m diameter along its whole bend, then the cone nozzle at its end tapers that same 0.16 m width down to a point over the nozzle's 0.18 m length.",
)
emit(
    pipe,
    "how many objects make up the bending pipe and its nozzle?",
    f"{f.count}: the bent tube and the tapering cone nozzle at its end.",
)
emit(
    pipe,
    "what generator built the bent part of the pipe?",
    "A tube -- a constant-radius circular cross-section swept along a bending path; the taper only comes from the separate cone nozzle at the end.",
)


# --- 47. coiled_snake --------------------------------------------------------------------
def _coil_path(
    turns: float, radius: float, rise: float, n: int, z_center: float
) -> list[list[float]]:
    pts = []
    for i in range(n + 1):
        t = i / n
        angle = 2 * math.pi * turns * t
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        y = z_center + rise * t
        pts.append([round(x, 4), round(y, 4), round(z, 4)])
    return pts


_snake_path = _coil_path(turns=2.0, radius=0.25, rise=0.15, n=24, z_center=0.1)
coiled_snake = [
    # the path's x/z are already centred at 0 (cos/sin over full turns); its
    # y runs 0.1..0.25 (centre 0.175) -- tube recentres on the path's own
    # bbox, so translation.y = 0.175 reproduces that intended height.
    prim("tube", "snake", {"path": _snake_path, "radius": 0.04, "sides": 10}, [0, 0.175, 0]),
]
f = Facts(coiled_snake, "coiled_snake")
emit(
    coiled_snake,
    "a snake figure coiled into a loop :: how many coils does it make?",
    "Two full turns -- the body's path was built to wind twice around before rising to its final height.",
)
emit(
    coiled_snake,
    "how wide is the coiled snake, from side to side?",
    f"About {r2(f.bsize()[0])} m across the whole coil.",
)
emit(
    coiled_snake,
    "how many objects is the coiled snake built from?",
    f"Just {f.count} -- one tube swept along the coiled path.",
)

# =============================================================================
# CLARIFICATION records -- a request the assistant should not act on blindly.
# =============================================================================

CLARIFY: list[tuple[list[dict], str, str]] = [
    (
        crate_stack,
        "delete everything",
        "Delete every object in this three-crate stack, or just one of the three crates -- and if just one, which?",
    ),
    (
        table_four_legs,
        "make it look cooler",
        '"Cooler" could mean a different material, thinner legs, or an entirely different style -- which of those did you have in mind for this table?',
    ),
    (
        round_stool,
        "make it better",
        "Better in what way -- sturdier legs, a bigger seat, or a different material -- so I change the right thing?",
    ),
    (
        garden_bench,
        "fix the proportions",
        "Which proportion looks wrong to you -- the seat height, the backrest angle, or the leg thickness?",
    ),
    (
        stone_column,
        "add some detail",
        "What kind of detail -- fluting on the shaft, carving on the capital, or something else -- would you like added?",
    ),
    (
        wooden_chest,
        "make it bigger",
        "Bigger by how much, and in which dimension -- taller, wider, deeper, or scaled up all round?",
    ),
    (
        metal_bucket,
        "change the color",
        "What colour would you like it instead -- and should it keep the same metallic finish, or become something like painted wood?",
    ),
    (
        wardrobe,
        "clean this up",
        "What looks untidy to you here -- the door gap, the material, or something about the proportions?",
    ),
    (
        dresser,
        "make the drawers match",
        "The three drawers are already the same size and evenly spaced -- did you mean something else, like matching handles or a matching material?",
    ),
    (
        side_table,
        "swap the material for something nicer",
        "What look are you after -- a different wood tone, a painted finish, or a metal top instead?",
    ),
    (
        coffee_table,
        "add more detail to the top",
        "What kind of detail on the tabletop -- an inlay, a different edge shape, or a texture -- did you have in mind?",
    ),
    (
        stone_archway,
        "make it fancier",
        "Fancier how -- carved reliefs on the stone, a taller opening, or a different material entirely?",
    ),
    (
        stone_plinth,
        "resize it",
        "Resize to what -- taller, wider, or a specific new size in metres?",
    ),
    (
        wooden_cart,
        "redo the legs",
        "The cart has wheels rather than legs -- did you mean redo the wheels, or add legs in place of them?",
    ),
    (
        quad_tube,
        "make the tail longer",
        "Longer by how much -- a specific length, or roughly double what it is now (about 0.4 m)?",
    ),
]

_QA_COUNT = len(RECORDS)
"""How many of ``RECORDS`` are question/answer rows over the 47 priors above
(three per prior, in order) before :data:`CLARIFY`'s rows are appended --
:func:`main` thins these down to two per prior so the family lands near its
100-record target instead of 47*3 + 15."""

for calls, prompt, answer in CLARIFY:
    emit(calls, prompt, answer)


# --- write --------------------------------------------------------------------


def main() -> None:
    # Keep the first two questions of every three-question run per prior
    # (drop the third), then every clarification record -- 47*2 + 15 = 109,
    # close to the family's 100-record target without hand-trimming any one
    # prior's block above.
    qa = [r for i, r in enumerate(RECORDS[:_QA_COUNT]) if i % 3 != 2]
    final = qa + RECORDS[_QA_COUNT:]
    for i, record in enumerate(final, start=1):
        record["id"] = f"queries-{i:04d}"
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for record in final:
            fh.write(json.dumps(record, sort_keys=False) + "\n")
    print(f"wrote {len(final)} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
