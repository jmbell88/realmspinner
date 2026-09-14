"""Deterministic generator for ``drafts/furniture.jsonl``.

Kept beside the JSONL it emits so the dataset can be regenerated -- see
``training/clay-assistant/README.md``'s "What to produce" and the common
author brief. Every record is a ``kind: build`` row: a hand-reasoned base
build (a function below) instantiated across several prompt phrasings with
lightly varied dimensions/materials/settings, all deterministic (no
wall-clock, no unseeded random).

Run with ``python _gen_furniture.py`` from this directory, or
``python training/clay-assistant/drafts/_gen_furniture.py`` from the repo
root. Writes ``furniture.jsonl`` beside this file.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

OUT_PATH = Path(__file__).resolve().parent / "furniture.jsonl"

# --- tiny call-builders, matching agent_clay's compact tool card -----------


def prim(
    generator: str,
    name: str,
    *,
    params: dict[str, Any] | None = None,
    translation: list[float] | None = None,
    rotation: list[float] | None = None,
    scale: list[float] | None = None,
    material: int | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {"generator": generator, "name": name}
    if params is not None:
        args["params"] = params
    if translation is not None:
        args["translation"] = [round(float(v), 4) for v in translation]
    if rotation is not None:
        args["rotation"] = [round(float(v), 3) for v in rotation]
    if scale is not None:
        args["scale"] = [round(float(v), 4) for v in scale]
    if material is not None:
        args["material"] = material
    return {"name": "clay_add_primitive", "arguments": args}


def ref(name: str) -> dict[str, str]:
    return {"$ref": name}


def material_call(
    names: list[str],
    color: tuple[float, float, float],
    mname: str,
    *,
    roughness: float = 0.6,
    metallic: float = 0.0,
) -> dict[str, Any]:
    return {
        "name": "clay_material",
        "arguments": {
            "uids": [ref(n) for n in names],
            "color": [round(float(c), 3) for c in color],
            "name": mname,
            "roughness": round(float(roughness), 3),
            "metallic": round(float(metallic), 3),
        },
    }


def select_call(names: list[str]) -> dict[str, Any]:
    return {"name": "clay_select", "arguments": {"uids": [ref(n) for n in names]}}


def op_call(name: str, params: dict[str, float] | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {"name": name}
    if params:
        args["params"] = {k: round(float(v), 4) for k, v in params.items()}
    return {"name": "clay_op", "arguments": args}


def boolean_call(kind: str, names: list[str]) -> dict[str, Any]:
    return {"name": "clay_boolean", "arguments": {"kind": kind, "uids": [ref(n) for n in names]}}


AXIS = {"x": 0, "y": 1, "z": 2}

# --- reusable geometry: profiles and small shapes ---------------------------


def turned_leg_profile(
    height: float,
    *,
    foot_r: float = 0.032,
    shaft_r: float = 0.017,
    bulge_r: float = 0.03,
    top_r: float = 0.02,
) -> list[list[float]]:
    """A simple turned-baluster leg, bottom to top, symmetric about its own
    centre (spans ``-height/2`` to ``+height/2``) so it can be placed at
    ``translation.y = height/2`` like every other generator -- a foot disc,
    a thin shaft, one decorative bulge, and a slightly wider top for the
    underside it meets."""
    h = height
    return [
        [0.0, -h / 2],
        [foot_r, -h / 2 + 0.06 * h],
        [shaft_r, -h / 2 + 0.18 * h],
        [bulge_r, -h / 2 + 0.42 * h],
        [shaft_r, -h / 2 + 0.66 * h],
        [shaft_r, -h / 2 + 0.86 * h],
        [top_r, h / 2],
    ]


def lamp_base_profile(
    height: float, *, foot_r: float, bulge_r: float, neck_r: float
) -> list[list[float]]:
    """A squat vase-like profile for a lathe lamp base: wide foot, a full
    bulge, and a narrow neck at the top where a pole or socket sits."""
    h = height
    return [
        [foot_r * 0.5, -h / 2],
        [foot_r, -h / 2 + 0.08 * h],
        [foot_r * 0.85, -h / 2 + 0.18 * h],
        [bulge_r, -h / 2 + 0.5 * h],
        [foot_r * 0.7, -h / 2 + 0.82 * h],
        [neck_r, -h / 2 + 0.94 * h],
        [neck_r, h / 2],
    ]


def tube_ground_y(path: list[list[float]], radius: float, clearance: float = 0.02) -> float:
    """``tube``'s own ``path`` is re-centred on its bounding-box centre by
    ``_clamp_path`` before the mesh is built (the same "positions, not a
    placement" rule ``lathe``'s ``profile`` and ``sweep``'s ``outline`` each
    follow) -- so a *path* authored with e.g. y from 0.02 to 1.0 does not
    result in a mesh whose lowest point is near y=0.02; it results in one
    re-centred about the path's own bbox mid-height, same as every other
    generator here. This returns the ``translation.y`` that puts the path's
    own lowest station back at ``clearance`` above the ground, the tube's
    ``radius`` included as a margin for the ring thickness at that station."""
    ys = [p[1] for p in path]
    span_half = (max(ys) - min(ys)) / 2.0
    return span_half + radius + clearance


def octagon_outline(r: float) -> list[list[float]]:
    import math

    return [
        [round(r * math.cos(a), 5), round(r * math.sin(a), 5)]
        for a in (i * math.pi / 4 for i in range(8))
    ]


# --- colour palette (linear RGB 0..1) --------------------------------------

WOOD_OAK = (0.45, 0.31, 0.18)
WOOD_WALNUT = (0.24, 0.15, 0.09)
WOOD_PINE = (0.68, 0.55, 0.35)
WOOD_EBONY = (0.09, 0.07, 0.06)
WOOD_CHERRY = (0.40, 0.18, 0.13)
METAL_STEEL = (0.55, 0.56, 0.58)
METAL_BRASS = (0.6, 0.47, 0.2)
METAL_GUNMETAL = (0.2, 0.21, 0.23)
FABRIC_RED = (0.55, 0.1, 0.12)
FABRIC_GREEN = (0.14, 0.35, 0.2)
FABRIC_BLUE = (0.14, 0.22, 0.45)
FABRIC_CREAM = (0.85, 0.8, 0.68)
STONE_GREY = (0.5, 0.5, 0.48)
PLASTIC_WHITE = (0.9, 0.9, 0.88)
LEATHER_TAN = (0.5, 0.32, 0.18)
GLASS_PALE = (0.75, 0.8, 0.8)

WOODS = [WOOD_OAK, WOOD_WALNUT, WOOD_PINE, WOOD_EBONY, WOOD_CHERRY]
WOOD_NAMES = ["Oak", "Walnut", "Pine", "Ebony", "Cherry"]


def wood(i: int) -> tuple[tuple[float, float, float], str]:
    return WOODS[i % len(WOODS)], WOOD_NAMES[i % len(WOODS)]


def darker(color: tuple[float, float, float], factor: float = 0.62) -> tuple[float, float, float]:
    """A visibly distinct accent tone of *color* -- used for a drawer front
    or door panel against its carcass so a flush boolean cut (or just the
    seam) actually reads at gallery-render scale instead of disappearing
    into one flat-shaded silhouette of a single material."""
    return (color[0] * factor, color[1] * factor, color[2] * factor)


# --- record bookkeeping ------------------------------------------------------

RECORDS: list[dict[str, Any]] = []
_COUNTER = [0]


def emit(
    prompt: str,
    calls: list[dict[str, Any]],
    notes: str,
    *,
    allow_below_ground: bool = False,
) -> None:
    _COUNTER[0] += 1
    rid = f"furniture-{_COUNTER[0]:04d}"
    rec: dict[str, Any] = {
        "id": rid,
        "family": "furniture",
        "kind": "build",
        "prompt": prompt,
        "calls": calls,
    }
    if allow_below_ground:
        rec["allow_below_ground"] = True
    rec["notes"] = notes
    RECORDS.append(rec)


Variant = dict[str, Any]


def instantiate(
    builder: Callable[..., tuple[list[dict[str, Any]], str]], variants: list[tuple[str, Variant]]
) -> None:
    """Call *builder* once per (prompt, variant-kwargs) pair in *variants* and
    emit the record. *builder* returns ``(calls, notes)``."""
    for prompt, kwargs in variants:
        calls, notes = builder(**kwargs)
        emit(prompt, calls, notes)


# --- generic part assemblers -------------------------------------------------


def four_box_legs(
    w: float,
    h: float,
    d: float,
    inset_x: float,
    inset_z: float,
    *,
    prefix: str = "leg",
    y0: float = 0.0,
) -> list[dict[str, Any]]:
    tags = [
        ("fl", inset_x, inset_z),
        ("fr", -inset_x, inset_z),
        ("bl", inset_x, -inset_z),
        ("br", -inset_x, -inset_z),
    ]
    return [
        prim("box", f"{prefix}_{t}", params={"size": [w, h, d]}, translation=[x, y0 + h / 2, z])
        for t, x, z in tags
    ]


def two_box_legs(
    w: float, h: float, d: float, inset_x: float, *, prefix: str = "leg", y0: float = 0.0
) -> list[dict[str, Any]]:
    return [
        prim(
            "box", f"{prefix}_l", params={"size": [w, h, d]}, translation=[inset_x, y0 + h / 2, 0.0]
        ),
        prim(
            "box",
            f"{prefix}_r",
            params={"size": [w, h, d]},
            translation=[-inset_x, y0 + h / 2, 0.0],
        ),
    ]


def four_lathe_legs(
    height: float,
    inset_x: float,
    inset_z: float,
    *,
    prefix: str = "leg",
    y0: float = 0.0,
    **profile_kw: float,
) -> list[dict[str, Any]]:
    profile = turned_leg_profile(height, **profile_kw)
    tags = [
        ("fl", inset_x, inset_z),
        ("fr", -inset_x, inset_z),
        ("bl", inset_x, -inset_z),
        ("br", -inset_x, -inset_z),
    ]
    return [
        prim(
            "lathe",
            f"{prefix}_{t}",
            params={"profile": profile, "segments": 12},
            translation=[x, y0 + height / 2, z],
        )
        for t, x, z in tags
    ]


def shelf_stack(
    count: int,
    width: float,
    depth: float,
    thickness: float,
    span: float,
    *,
    y0: float = 0.0,
    prefix: str = "shelf",
) -> list[dict[str, Any]]:
    """*count* thin rectangular shelves, evenly spaced from ``y0`` to
    ``y0 + span`` (inclusive at both ends), each its own primitive so the
    build stays in primitives-and-placement territory."""
    calls = []
    for i in range(count):
        frac = i / (count - 1) if count > 1 else 0.0
        y = y0 + frac * span
        calls.append(
            prim(
                "box",
                f"{prefix}{i + 1}",
                params={"size": [width, thickness, depth]},
                translation=[0.0, y, 0.0],
            )
        )
    return calls


def carcass_sides(
    height: float, width: float, depth: float, thickness: float, *, prefix: str = "side"
) -> list[dict[str, Any]]:
    """Two vertical side panels (left/right) of a cabinet/bookshelf, plus
    nothing else -- caller adds top/bottom/back/shelves."""
    inset = width / 2 - thickness / 2
    return [
        prim(
            "box",
            f"{prefix}_l",
            params={"size": [thickness, height, depth]},
            translation=[inset, height / 2, 0.0],
        ),
        prim(
            "box",
            f"{prefix}_r",
            params={"size": [thickness, height, depth]},
            translation=[-inset, height / 2, 0.0],
        ),
    ]


# =============================================================================
# EASY: primitives, placement and materials alone.
# =============================================================================


def bb_table_plain(top_w=1.4, top_d=0.8, top_t=0.04, leg_h=0.72, leg_w=0.06, leg_d=0.06, wood_i=0):
    inset_x, inset_z = top_w / 2 - leg_w / 2 - 0.03, top_d / 2 - leg_d / 2 - 0.03
    calls = [
        prim(
            "box",
            "top",
            params={"size": [top_w, top_t, top_d]},
            translation=[0, leg_h + top_t / 2, 0],
        )
    ]
    calls += four_box_legs(leg_w, leg_h, leg_d, inset_x, inset_z)
    color, name = wood(wood_i)
    calls.append(
        material_call(["top", "leg_fl", "leg_fr", "leg_bl", "leg_br"], color, name, roughness=0.7)
    )
    notes = (
        f"a rectangular slab top ({top_w}x{top_d} m) on four square legs inset from each "
        f"corner, leg height {leg_h} m -- primitives and placement alone, no boolean needed."
    )
    return calls, notes


def bb_nightstand(w=0.42, d=0.38, h=0.5, drawer_h=0.16, wood_i=1):
    body_h = h - 0.03
    calls = [
        prim("box", "body", params={"size": [w, body_h, d]}, translation=[0, body_h / 2, 0]),
        prim(
            "box", "top", params={"size": [w + 0.03, 0.03, d + 0.03]}, translation=[0, h - 0.015, 0]
        ),
        prim(
            "box",
            "drawer_front",
            params={"size": [w - 0.05, drawer_h, 0.02]},
            translation=[0, body_h - drawer_h / 2 - 0.05, d / 2 + 0.01],
        ),
        prim(
            "cylinder",
            "handle",
            params={"radius": 0.012, "height": 0.04, "segments": 10},
            translation=[0, body_h - drawer_h / 2 - 0.05, d / 2 + 0.03],
            rotation=[90, 0, 0],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["body", "top", "drawer_front"], color, name, roughness=0.65))
    calls.append(
        material_call(["handle"], METAL_BRASS, "Brass Fitting", roughness=0.3, metallic=0.85)
    )
    notes = "a small carcass with an overhanging top and one proud drawer front plus a turned knob; flush geometry, no cut needed for an easy build."
    return calls, notes


def bb_bookshelf_open(w=0.8, h=1.5, d=0.28, n=4, wood_i=2, lean=0.0):
    t = 0.02
    calls = carcass_sides(h, w, d, t)
    calls += shelf_stack(n, w - 2 * t, d, t, h - t, y0=t / 2)
    calls.append(
        prim(
            "box",
            "back",
            params={"size": [w - 2 * t, h - t, 0.01]},
            translation=[0, h / 2, -d / 2 + 0.005],
        )
    )
    names = ["side_l", "side_r", "back"] + [f"shelf{i + 1}" for i in range(n)]
    if lean:
        for c in calls:
            c["arguments"]["rotation"] = [0, 0, lean]
    color, name = wood(wood_i)
    calls.append(material_call(names, color, name, roughness=0.75))
    notes = f"a case of two sides, a thin back and {n} evenly spaced shelf boards -- every board its own primitive, an open bookshelf needs nothing else."
    return calls, notes


def bb_platform_bed(w=2.0, l=1.6, h=0.32, wood_i=0):
    calls = [prim("box", "base", params={"size": [l, h, w]}, translation=[0, h / 2, 0])]
    color, name = wood(wood_i)
    calls.append(material_call(["base"], color, name, roughness=0.7))
    notes = f"one low platform slab, {l}x{w} m at {h} m tall, no headboard -- a single box is the whole brief."
    return calls, notes


def bb_plank_bench(w=1.2, d=0.32, t=0.06, leg_h=0.42, wood_i=1):
    inset_x = w / 2 - 0.1
    calls = [prim("box", "seat", params={"size": [w, t, d]}, translation=[0, leg_h + t / 2, 0])]
    calls += [
        prim(
            "box",
            "block_l",
            params={"size": [0.12, leg_h, d - 0.04]},
            translation=[inset_x, leg_h / 2, 0],
        ),
        prim(
            "box",
            "block_r",
            params={"size": [0.12, leg_h, d - 0.04]},
            translation=[-inset_x, leg_h / 2, 0],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["seat", "block_l", "block_r"], color, name, roughness=0.8))
    notes = "a single thick plank on two solid block legs set well in from the ends -- rustic, three boxes total."
    return calls, notes


def bb_tall_bookshelf(w=0.55, h=1.9, d=0.26, n=5, wood_i=2, lean=3.0):
    return bb_bookshelf_open(w=w, h=h, d=d, n=n, wood_i=wood_i, lean=lean)


def bb_toy_chest(w=0.7, d=0.4, h=0.42, wood_i=3):
    calls = [prim("box", "box", params={"size": [w, h, d]}, translation=[0, h / 2, 0])]
    color, name = wood(wood_i)
    calls.append(material_call(["box"], color, name, roughness=0.75))
    notes = "a plain rectangular box, child-scale -- one primitive, nothing else needed."
    return calls, notes


def bb_stool_round_3legs(seat_r=0.18, seat_t=0.04, leg_h=0.45, leg_r=0.02, splay=8.0, wood_i=0):
    import math

    calls = [
        prim(
            "cylinder",
            "seat",
            params={"radius": seat_r, "height": seat_t, "segments": 20},
            translation=[0, leg_h + seat_t / 2, 0],
        )
    ]
    names = ["seat"]
    ring_r = seat_r - 0.03
    for i in range(3):
        ang = math.radians(90 + i * 120)
        x, z = ring_r * math.cos(ang), ring_r * math.sin(ang)
        name = f"leg_{i + 1}"
        calls.append(
            prim(
                "cylinder",
                name,
                params={"radius": leg_r, "height": leg_h, "segments": 10},
                translation=[x, leg_h / 2, z],
                rotation=[splay * math.sin(ang), 0, splay * math.cos(ang)],
            )
        )
        names.append(name)
    color, name_ = wood(wood_i)
    calls.append(material_call(names, color, name_, roughness=0.7))
    notes = "a round seat on three straight legs set 120 degrees apart with a slight outward splay for stability -- primitives and rotation, no lathe needed for the plain version."
    return calls, notes


def bb_desk_panel_legs(top_w=1.3, top_d=0.65, top_t=0.03, h=0.75, panel_t=0.03, wood_i=1):
    inset = top_w / 2 - panel_t / 2 - 0.02
    calls = [
        prim(
            "box", "top", params={"size": [top_w, top_t, top_d]}, translation=[0, h - top_t / 2, 0]
        ),
        prim(
            "box",
            "panel_l",
            params={"size": [panel_t, h - top_t, top_d - 0.04]},
            translation=[inset, (h - top_t) / 2, 0],
        ),
        prim(
            "box",
            "panel_r",
            params={"size": [panel_t, h - top_t, top_d - 0.04]},
            translation=[-inset, (h - top_t) / 2, 0],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["top", "panel_l", "panel_r"], color, name, roughness=0.6))
    notes = "a flat top on two solid side panels standing in for legs -- three boxes, an office desk silhouette."
    return calls, notes


def bb_round_side_table(top_r=0.28, top_t=0.03, h=0.4, leg_r=0.03, wood_i=0):
    calls = [
        prim(
            "cylinder",
            "top",
            params={"radius": top_r, "height": top_t, "segments": 24},
            translation=[0, h - top_t / 2, 0],
        ),
        prim(
            "cylinder",
            "leg",
            params={"radius": leg_r, "height": h - top_t, "segments": 16},
            translation=[0, (h - top_t) / 2, 0],
        ),
        prim(
            "cylinder",
            "foot",
            params={"radius": top_r * 0.4, "height": 0.02, "segments": 20},
            translation=[0, 0.01, 0],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["top", "leg", "foot"], color, name, roughness=0.6))
    notes = f"a round top on a single central cylinder pedestal with a small foot disc for stability, {h} m tall -- no lathe needed, a plain pedestal."
    return calls, notes


def bb_ottoman(size=0.5, wood_i=None, fabric=FABRIC_CREAM):
    calls = [prim("box", "body", params={"size": [size, size, size]}, translation=[0, size / 2, 0])]
    calls.append(material_call(["body"], fabric, "Upholstery", roughness=0.9))
    notes = (
        "a thick cube shape read as padded by a soft fabric material and high roughness -- one box."
    )
    return calls, notes


def bb_wardrobe(w=0.9, d=0.55, h=1.9, wood_i=1, door_color=None):
    calls = [prim("box", "body", params={"size": [w, h, d]}, translation=[0, h / 2, 0])]
    door_w = w / 2 - 0.015
    calls += [
        prim(
            "box",
            "door_l",
            params={"size": [door_w, h - 0.1, 0.025]},
            translation=[door_w / 2 + 0.01, h / 2, d / 2 + 0.0125],
        ),
        prim(
            "box",
            "door_r",
            params={"size": [door_w, h - 0.1, 0.025]},
            translation=[-(door_w / 2 + 0.01), h / 2, d / 2 + 0.0125],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["body"], color, name, roughness=0.65))
    calls.append(
        material_call(["door_l", "door_r"], darker(color), f"{name} Door Panel", roughness=0.6)
    )
    notes = (
        "a tall carcass with two square door faces proud of the front -- no hinges or booleans, just placed "
        "panels, in a darker accent tone so the two doors read against the carcass."
    )
    return calls, notes


def bb_crate_end_table(size=0.45, wood_i=2):
    calls = [
        prim("box", "crate", params={"size": [size, size, size]}, translation=[0, size / 2, 0])
    ]
    slat_t = 0.015
    for i, y in enumerate((size * 0.3, size * 0.7)):
        calls.append(
            prim(
                "box",
                f"slat{i + 1}",
                params={"size": [size + 0.01, slat_t, size + 0.01]},
                translation=[0, y, 0],
            )
        )
    color, name = wood(wood_i)
    calls.append(material_call(["crate", "slat1", "slat2"], color, name, roughness=0.8))
    notes = "a crate-shaped cube with two thin wraparound battens for a crate-style read -- primitives only."
    return calls, notes


def bb_dining_bench_long(w=1.6, d=0.34, t=0.05, leg_h=0.45, wood_i=0):
    inset_x, inset_z = w / 2 - 0.08, d / 2 - 0.04
    calls = [prim("box", "seat", params={"size": [w, t, d]}, translation=[0, leg_h + t / 2, 0])]
    calls += four_box_legs(0.06, leg_h, 0.06, inset_x, inset_z)
    color, name = wood(wood_i)
    calls.append(
        material_call(["seat", "leg_fl", "leg_fr", "leg_bl", "leg_br"], color, name, roughness=0.75)
    )
    notes = "a long plain rectangular seat on four square legs -- a dining bench, primitives only."
    return calls, notes


def bb_folding_table_metal(top_w=1.5, top_d=0.7, top_t=0.02, h=0.74):
    inset_x, inset_z = top_w / 2 - 0.03, top_d / 2 - 0.03
    calls = [
        prim(
            "box", "top", params={"size": [top_w, top_t, top_d]}, translation=[0, h - top_t / 2, 0]
        )
    ]
    calls += four_box_legs(0.03, h - top_t, 0.03, inset_x, inset_z)
    calls.append(
        material_call(
            ["top", "leg_fl", "leg_fr", "leg_bl", "leg_br"],
            METAL_STEEL,
            "Steel",
            roughness=0.4,
            metallic=0.8,
        )
    )
    notes = "a thin rectangular top on four slim steel legs -- a folding-table silhouette, metal material only."
    return calls, notes


def bb_dining_chair_solid_back(seat=0.44, seat_t=0.05, leg_h=0.45, back_h=0.42, wood_i=1):
    inset = seat / 2 - 0.03
    calls = [
        prim(
            "box",
            "seat",
            params={"size": [seat, seat_t, seat]},
            translation=[0, leg_h + seat_t / 2, 0],
        ),
        prim(
            "box",
            "back",
            params={"size": [seat, back_h, 0.035]},
            translation=[0, leg_h + seat_t + back_h / 2, -seat / 2 + 0.02],
        ),
    ]
    calls += four_box_legs(0.045, leg_h, 0.045, inset, inset)
    color, name = wood(wood_i)
    calls.append(
        material_call(
            ["seat", "back", "leg_fl", "leg_fr", "leg_bl", "leg_br"], color, name, roughness=0.7
        )
    )
    notes = "a solid back panel (not slatted -- held-out subject avoided) over a square seat on four straight legs."
    return calls, notes


def bb_three_tier_shelf(w=0.5, d=0.28, gap=0.32, t=0.015, wood_i=2):
    calls = shelf_stack(3, w, d, t, gap * 2, y0=0.05)
    inset = w / 2 - 0.02
    calls += two_box_legs(0.02, gap * 2 + 0.05, 0.02, inset, prefix="post")
    color, name = wood(wood_i)
    calls.append(
        material_call(
            ["shelf1", "shelf2", "shelf3", "post_l", "post_r"], color, name, roughness=0.75
        )
    )
    notes = "three thin rectangular shelves on two corner posts, evenly spaced -- primitives only."
    return calls, notes


def bb_coffee_table_squat(top_w=1.0, top_d=0.55, top_t=0.035, h=0.3, wood_i=0):
    return bb_table_plain(
        top_w=top_w,
        top_d=top_d,
        top_t=top_t,
        leg_h=h - top_t,
        leg_w=0.05,
        leg_d=0.05,
        wood_i=wood_i,
    )


def bb_cargo_bench_scifi(w=1.3, d=0.4, t=0.06, block_h=0.4, wood_i=None):
    inset = w / 2 - 0.18
    calls = [
        prim("box", "slab", params={"size": [w, t, d]}, translation=[0, block_h + t / 2, 0]),
        prim(
            "box",
            "block_l",
            params={"size": [0.3, block_h, d - 0.05]},
            translation=[inset, block_h / 2, 0],
        ),
        prim(
            "box",
            "block_r",
            params={"size": [0.3, block_h, d - 0.05]},
            translation=[-inset, block_h / 2, 0],
        ),
    ]
    calls.append(
        material_call(
            ["slab", "block_l", "block_r"],
            METAL_GUNMETAL,
            "Cargo Alloy",
            roughness=0.45,
            metallic=0.6,
        )
    )
    notes = "a simple slab on two wide support blocks, painted as gunmetal cargo alloy for a sci-fi read -- primitives only."
    return calls, notes


def bb_minimalist_desk_central_leg(
    top_w=1.4, top_d=0.6, top_t=0.03, h=0.73, leg_w=0.12, leg_d=0.5, wood_i=3
):
    calls = [
        prim(
            "box", "top", params={"size": [top_w, top_t, top_d]}, translation=[0, h - top_t / 2, 0]
        ),
        prim(
            "box",
            "leg",
            params={"size": [leg_w, h - top_t, leg_d]},
            translation=[0, (h - top_t) / 2, 0],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["top", "leg"], color, name, roughness=0.55))
    notes = "a single slab top on one thick central leg block -- minimalist, two boxes."
    return calls, notes


def bb_footstool_box(w=0.35, d=0.28, h=0.18, wood_i=2):
    calls = [prim("box", "box", params={"size": [w, h, d]}, translation=[0, h / 2, 0])]
    color, name = wood(wood_i)
    calls.append(material_call(["box"], color, name, roughness=0.8))
    notes = "a small flat box, medieval-plain -- one primitive."
    return calls, notes


def bb_log_bench(length=1.2, radius=0.16, block_h=0.15, wood_i=2):
    inset = length / 2 - 0.15
    calls = [
        prim(
            "cylinder",
            "log",
            params={"radius": radius, "height": length, "segments": 14},
            translation=[0, block_h + radius, 0],
            rotation=[0, 0, 90],
        ),
        prim(
            "box",
            "block_l",
            params={"size": [0.16, block_h, radius * 1.6]},
            translation=[inset, block_h / 2, 0],
        ),
        prim(
            "box",
            "block_r",
            params={"size": [0.16, block_h, radius * 1.6]},
            translation=[-inset, block_h / 2, 0],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["log", "block_l", "block_r"], color, name, roughness=0.85))
    notes = "a plain cylinder log, rotated on its side, resting on two blocks -- a campfire bench, primitives only."
    return calls, notes


def bb_vanity_cabinet(w=0.8, d=0.5, h=0.85, wood_i=1):
    calls = [
        prim("box", "body", params={"size": [w, h - 0.03, d]}, translation=[0, (h - 0.03) / 2, 0]),
        prim(
            "box", "top", params={"size": [w + 0.02, 0.03, d + 0.04]}, translation=[0, h - 0.015, 0]
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["body", "top"], color, name, roughness=0.6))
    notes = "a rectangular carcass with a slightly overhanging flat top -- a bathroom vanity, two boxes."
    return calls, notes


def bb_wine_rack_open_frame(w=0.5, d=0.35, h=1.6, wood_i=2):
    t = 0.03
    inset_x, inset_z = w / 2 - t / 2, d / 2 - t / 2
    calls = four_box_legs(t, h, t, inset_x, inset_z, prefix="post")
    for i, y in enumerate((0.02, h - 0.02)):
        calls.append(prim("box", f"rail{i + 1}", params={"size": [w, t, d]}, translation=[0, y, 0]))
    color, name = wood(wood_i)
    names = ["post_fl", "post_fr", "post_bl", "post_br", "rail1", "rail2"]
    calls.append(material_call(names, color, name, roughness=0.75))
    notes = "four corner posts and top/bottom rails -- an open box frame, no shelves or bottle holes yet (the easy version)."
    return calls, notes


def bb_plant_stand(top=0.28, top_t=0.02, h=0.55, leg_r=0.015, wood_i=0):
    inset = top / 2 - 0.03
    calls = [
        prim("box", "top", params={"size": [top, top_t, top]}, translation=[0, h - top_t / 2, 0])
    ]
    calls += four_box_legs(leg_r * 2, h - top_t, leg_r * 2, inset, inset)
    color, name = wood(wood_i)
    calls.append(
        material_call(["top", "leg_fl", "leg_fr", "leg_bl", "leg_br"], color, name, roughness=0.7)
    )
    notes = "a small square top on four thin legs, tall and narrow for a single plant pot."
    return calls, notes


# =============================================================================
# MEDIUM: a boolean, a lathe/sweep profile, an array op, or element-mode.
# =============================================================================


def bb_dining_table_lathe_legs(top_w=1.5, top_d=0.85, top_t=0.04, leg_h=0.72, wood_i=1):
    inset_x, inset_z = top_w / 2 - 0.08, top_d / 2 - 0.08
    calls = [
        prim(
            "box",
            "top",
            params={"size": [top_w, top_t, top_d]},
            translation=[0, leg_h + top_t / 2, 0],
        )
    ]
    calls += four_lathe_legs(leg_h, inset_x, inset_z)
    color, name = wood(wood_i)
    calls.append(
        material_call(["top", "leg_fl", "leg_fr", "leg_bl", "leg_br"], color, name, roughness=0.6)
    )
    notes = "a slab top on four lathe-turned legs (a shared turned-baluster profile scaled to leg height) -- the lathe generator earns its keep here."
    return calls, notes


def bb_armchair_cutout(seat=0.5, seat_t=0.06, leg_h=0.42, back_h=0.5, arm_h=0.22, wood_i=1):
    inset = seat / 2 - 0.04
    calls = [
        prim(
            "box",
            "seat",
            params={"size": [seat, seat_t, seat]},
            translation=[0, leg_h + seat_t / 2, 0],
        ),
        prim(
            "box",
            "back",
            params={"size": [seat, back_h, 0.05]},
            translation=[0, leg_h + seat_t + back_h / 2, -seat / 2 + 0.025],
        ),
        prim(
            "box",
            "arm_l",
            params={"size": [0.07, arm_h, seat]},
            translation=[seat / 2 - 0.035, leg_h + seat_t + arm_h / 2, 0],
        ),
        prim(
            "box",
            "arm_r",
            params={"size": [0.07, arm_h, seat]},
            translation=[-(seat / 2 - 0.035), leg_h + seat_t + arm_h / 2, 0],
        ),
        prim(
            "cylinder",
            "cut_l",
            params={"radius": 0.025, "height": arm_h + 0.08, "segments": 12},
            translation=[seat / 2 - 0.035, leg_h + seat_t + arm_h / 2, 0],
        ),
        prim(
            "cylinder",
            "cut_r",
            params={"radius": 0.025, "height": arm_h + 0.08, "segments": 12},
            translation=[-(seat / 2 - 0.035), leg_h + seat_t + arm_h / 2, 0],
        ),
    ]
    calls += four_box_legs(0.05, leg_h, 0.05, inset, inset)
    calls.append(boolean_call("difference", ["arm_l", "cut_l"]))
    calls.append(boolean_call("difference", ["arm_r", "cut_r"]))
    color, name = wood(wood_i)
    names = ["seat", "back", "arm_l", "arm_r", "leg_fl", "leg_fr", "leg_bl", "leg_br"]
    calls.append(material_call(names, color, name, roughness=0.65))
    notes = (
        "each armrest is a box with a cylinder bored straight through it top to bottom (the cylinder's own "
        "default Y axis, taller than the arm so it fully pierces) and subtracted -- one boolean per hand-hold, "
        "oriented vertically rather than through the arm's thin width so the cut actually reads in a render."
    )
    return calls, notes


def bb_pedestal_table_lathe_column(top_r=0.55, top_t=0.04, col_h=0.68, foot_r=0.35, wood_i=0):
    calls = [
        prim(
            "cylinder",
            "top",
            params={"radius": top_r, "height": top_t, "segments": 32},
            translation=[0, col_h + top_t / 2, 0],
        ),
        prim(
            "lathe",
            "column",
            params={
                "profile": turned_leg_profile(
                    col_h, foot_r=0.09, shaft_r=0.05, bulge_r=0.1, top_r=0.07
                ),
                "segments": 20,
            },
            translation=[0, col_h / 2, 0],
        ),
        prim(
            "cylinder",
            "foot",
            params={"radius": foot_r, "height": 0.03, "segments": 32},
            translation=[0, 0.015, 0],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["top", "column", "foot"], color, name, roughness=0.55))
    notes = "a round top on one lathe-turned column standing on a wide foot disc -- the single-pedestal case of the lathe leg."
    return calls, notes


def bb_bar_stool_lathe_leg(seat_r=0.16, seat_t=0.05, leg_h=0.75, wood_i=1):
    calls = [
        prim(
            "cylinder",
            "seat",
            params={"radius": seat_r, "height": seat_t, "segments": 20},
            translation=[0, leg_h + seat_t / 2, 0],
        ),
        prim(
            "lathe",
            "leg",
            params={
                "profile": turned_leg_profile(
                    leg_h, foot_r=0.09, shaft_r=0.035, bulge_r=0.055, top_r=0.05
                ),
                "segments": 16,
            },
            translation=[0, leg_h / 2, 0],
        ),
        prim(
            "torus",
            "footring",
            params={"radius": 0.11, "tube": 0.012, "segments": 24, "sides": 10},
            translation=[0, leg_h * 0.35, 0],
        ),
    ]
    calls.append(material_call(["seat"], FABRIC_RED, "Padded Seat", roughness=0.85))
    color, name = wood(wood_i)
    calls.append(material_call(["leg", "footring"], color, name, roughness=0.55))
    notes = "a round padded seat over one lathe-turned leg with a torus footring partway down -- a bar stool, not a dining chair."
    return calls, notes


def bb_four_poster_bed(l=2.1, w=1.7, base_h=0.3, post_h=1.5, wood_i=2):
    inset_x, inset_z = l / 2 - 0.06, w / 2 - 0.06
    profile = turned_leg_profile(post_h, foot_r=0.06, shaft_r=0.03, bulge_r=0.05, top_r=0.035)
    calls = [
        prim("box", "base", params={"size": [l, base_h, w]}, translation=[0, base_h / 2, 0]),
        prim(
            "lathe",
            "post_front",
            params={"profile": profile, "segments": 14},
            translation=[inset_x, post_h / 2, inset_z],
        ),
        prim(
            "lathe",
            "post_back",
            params={"profile": profile, "segments": 14},
            translation=[inset_x, post_h / 2, -inset_z],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["base", "post_front", "post_back"], color, name, roughness=0.6))
    calls.append(select_call(["post_front", "post_back"]))
    calls.append(op_call("mirror-copy", {"axis": AXIS["x"], "offset": 0.0}))
    notes = (
        "two lathe-turned posts placed explicitly (front and back, both on the +X side), coloured, then one "
        "mirror-copy across X reaches the other two -- selecting both originals in the same call, rather than "
        "chaining a second select+mirror-copy against a first copy whose generated name is not known in "
        "advance, is what a first attempt at this build got wrong (only 3 of 4 posts came out, one pair still "
        "the document's default material) -- see the report for the suspected agent_clay surface gap this "
        "also points at: mirror-copy leaves the selection unchanged rather than selecting its own new copies "
        "the way array-linear/array-radial do, so a caller cannot chain a second mirror-copy over the first "
        "copy without already knowing its generated name."
    )
    return calls, notes


def bb_bookshelf_support_pegs(w=0.8, h=1.5, d=0.28, n=4, wood_i=2):
    t = 0.02
    calls = carcass_sides(h, w, d, t)
    peg_y = h / (n + 1)
    calls.append(
        prim(
            "cylinder",
            "peg",
            params={"radius": 0.008, "height": 0.04, "segments": 8},
            translation=[w / 2 - t - 0.02, peg_y, 0],
            rotation=[0, 0, 90],
        )
    )
    color, name = wood(wood_i)
    calls.append(material_call(["peg"], color, name, roughness=0.7))
    calls.append(select_call(["peg"]))
    calls.append(
        op_call(
            "array-linear", {"count": n, "x": 0.0, "y": (h - 2 * peg_y) / max(n - 1, 1), "z": 0.0}
        )
    )
    calls += shelf_stack(n, w - 2 * t - 0.03, d, t, h - 2 * peg_y, y0=peg_y)
    calls.append(
        material_call(
            ["side_l", "side_r"] + [f"shelf{i + 1}" for i in range(n)], color, name, roughness=0.75
        )
    )
    notes = "one peg coloured and array-linear'd up the case (material set before the array so every copy inherits it) then a shelf board resting at each peg height."
    return calls, notes


def bb_wine_rack_grid_holes(w=0.55, d=0.3, h=0.4, rows=2, cols=3, wood_i=2):
    calls = [prim("box", "front", params={"size": [w, h, 0.03]}, translation=[0, h / 2, 0])]
    cutters = []
    for r in range(rows):
        for c in range(cols):
            x = (c - (cols - 1) / 2) * (w / (cols + 0.5))
            y = (r + 0.5) * (h / rows)
            name = f"hole_{r}_{c}"
            calls.append(
                prim(
                    "cylinder",
                    name,
                    params={"radius": 0.035, "height": 0.08, "segments": 16},
                    translation=[x, y, 0],
                    rotation=[90, 0, 0],
                )
            )
            cutters.append(name)
    calls.append(boolean_call("difference", ["front"] + cutters))
    color, name = wood(wood_i)
    calls.append(material_call(["front"], color, name, roughness=0.7))
    notes = f"a {rows}x{cols} grid of explicit cutter cylinders, all subtracted from the front panel in one boolean call -- bottle holes cut through."
    return calls, notes


def bb_rocking_chair_runners(seat=0.42, seat_h=0.34, run_len=0.6, wood_i=1):
    rise, tube = 0.12, 0.02
    radius = run_len / 2
    scale_x = rise / radius
    ring_half = scale_x * (radius + tube)
    runner_x = seat / 2 - 0.05
    leg_h = max(seat_h - 2 * ring_half, 0.03)
    calls = [
        prim(
            "box",
            "seat",
            params={"size": [seat, 0.05, seat * 0.9]},
            translation=[0, seat_h + 0.025, 0],
        ),
        prim(
            "box",
            "back",
            params={"size": [seat, 0.4, 0.04]},
            translation=[0, seat_h + 0.05 + 0.2, -seat * 0.45 + 0.02],
        ),
        prim(
            "torus",
            "runner",
            params={"radius": radius, "tube": tube, "segments": 28, "sides": 10},
            translation=[runner_x, ring_half, 0],
            rotation=[0, 0, 90],
            scale=[scale_x, 1.0, 1.0],
        ),
        prim(
            "box",
            "leg",
            params={"size": [0.04, leg_h, 0.04]},
            translation=[runner_x, 2 * ring_half + leg_h / 2, 0],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["seat", "back", "runner", "leg"], color, name, roughness=0.7))
    calls.append(select_call(["runner", "leg"]))
    calls.append(op_call("mirror-copy", {"axis": AXIS["x"], "offset": 0.0}))
    notes = (
        "one runner (a torus squashed along local X, rotated 90 about Z so it stands as a side-on curve) plus "
        "its short connecting leg, coloured, then mirror-copy across X for the second side -- a plain seat and "
        "back sit on top."
    )
    return calls, notes


def bb_folding_camp_chair(seat_d=0.4, leg_len=0.46, splay=24.0, wood_i=0, fabric=FABRIC_GREEN):
    import math

    rad = math.radians(splay)
    half_y = abs(math.cos(rad)) * (leg_len / 2) + abs(math.sin(rad)) * 0.015
    cy = half_y  # centre height that puts the rotated leg's own bottom at y=0
    seat_h = 2 * cy
    calls = [
        prim(
            "box",
            "leg_front",
            params={"size": [0.03, leg_len, 0.03]},
            translation=[0.0, cy, seat_d * 0.28],
            rotation=[splay, 0, 0],
        ),
        prim(
            "box",
            "leg_back",
            params={"size": [0.03, leg_len, 0.03]},
            translation=[0.0, cy, -seat_d * 0.28],
            rotation=[-splay, 0, 0],
        ),
        prim(
            "cylinder",
            "pivot",
            params={"radius": 0.02, "height": 0.34, "segments": 10},
            translation=[0, cy, 0],
            rotation=[0, 0, 90],
        ),
        prim(
            "box",
            "seat_fabric",
            params={"size": [0.32, 0.01, seat_d]},
            translation=[0, seat_h - 0.02, 0],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["leg_front", "leg_back", "pivot"], color, name, roughness=0.6))
    calls.append(material_call(["seat_fabric"], fabric, "Canvas", roughness=0.9))
    notes = (
        "two legs rotated to opposite angles about the X axis, crossing at a pivot cylinder, with a thin "
        "fabric-coloured seat slab stretched between -- translation.y is solved from the rotation so the "
        "legs' own rotated bounding box still touches the ground exactly, rather than a fixed fraction of "
        "seat height."
    )
    return calls, notes


def bb_vanity_table_mirror(w=0.9, d=0.45, h=0.75, mirror_r=0.22, wood_i=1):
    post_h = 0.5
    inset = w / 2 - 0.05
    calls = [
        prim("box", "body", params={"size": [w, h - 0.03, d]}, translation=[0, (h - 0.03) / 2, 0]),
        prim(
            "box", "top", params={"size": [w + 0.02, 0.03, d + 0.02]}, translation=[0, h - 0.015, 0]
        ),
        prim(
            "lathe",
            "post_l",
            params={
                "profile": turned_leg_profile(
                    post_h, foot_r=0.025, shaft_r=0.014, bulge_r=0.022, top_r=0.016
                ),
                "segments": 12,
            },
            translation=[inset, h + post_h / 2, -d / 2 + 0.05],
        ),
        prim(
            "lathe",
            "post_r",
            params={
                "profile": turned_leg_profile(
                    post_h, foot_r=0.025, shaft_r=0.014, bulge_r=0.022, top_r=0.016
                ),
                "segments": 12,
            },
            translation=[-inset, h + post_h / 2, -d / 2 + 0.05],
        ),
        prim(
            "torus",
            "mirror_frame",
            params={"radius": mirror_r, "tube": 0.015, "segments": 28, "sides": 10},
            translation=[0, h + post_h, -d / 2 + 0.05],
            rotation=[90, 0, 0],
        ),
        prim(
            "cylinder",
            "mirror_glass",
            params={"radius": mirror_r - 0.015, "height": 0.005, "segments": 28},
            translation=[0, h + post_h, -d / 2 + 0.05],
            rotation=[0, 0, 90],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["body", "top", "post_l", "post_r"], color, name, roughness=0.6))
    calls.append(
        material_call(["mirror_frame"], METAL_BRASS, "Brass Frame", roughness=0.3, metallic=0.85)
    )
    calls.append(
        material_call(["mirror_glass"], GLASS_PALE, "Mirror Glass", roughness=0.05, metallic=0.1)
    )
    notes = "a cabinet with two lathe-turned posts rising from the top to carry a torus-framed round mirror above it."
    return calls, notes


def bb_office_chair_radial_base(seat=0.42, seat_h=0.46, back_h=0.45, col_r=0.03, n=5, wood_i=None):
    calls = [
        prim("box", "seat", params={"size": [seat, 0.06, seat]}, translation=[0, seat_h, 0]),
        prim(
            "box",
            "back",
            params={"size": [seat, back_h, 0.05]},
            translation=[0, seat_h + back_h / 2, -seat / 2 + 0.025],
        ),
        prim(
            "cylinder",
            "column",
            params={"radius": col_r, "height": seat_h - 0.05, "segments": 14},
            translation=[0, (seat_h - 0.05) / 2, 0],
        ),
        prim(
            "box",
            "arm",
            params={"size": [0.05, 0.03, 0.22]},
            translation=[0.19, 0.045, 0.18],
            rotation=[0, 20, 0],
        ),
        prim(
            "uv_sphere",
            "wheel",
            params={"radius": 0.02, "segments": 10, "rings": 6},
            translation=[0.19, 0.02, 0.24],
        ),
    ]
    calls.append(material_call(["seat", "back"], FABRIC_BLUE, "Office Fabric", roughness=0.8))
    calls.append(
        material_call(
            ["column", "arm", "wheel"], METAL_GUNMETAL, "Base Metal", roughness=0.4, metallic=0.7
        )
    )
    calls.append(select_call(["arm", "wheel"]))
    calls.append(op_call("array-radial", {"count": n, "angle": 360.0, "axis": AXIS["y"]}))
    notes = f"one caster arm plus wheel, coloured first, then array-radial around the world Y axis for a {n}-point wheeled base."
    return calls, notes


def bb_church_pew_divots(w=1.7, d=0.32, t=0.05, leg_h=0.45, n=5, wood_i=1):
    inset_x, inset_z = w / 2 - 0.1, d / 2 - 0.04
    calls = [
        prim("box", "seat", params={"size": [w, t, d]}, translation=[0, leg_h + t / 2, 0]),
        prim(
            "box",
            "back",
            params={"size": [w, 0.4, 0.04]},
            translation=[0, leg_h + t + 0.2, -d / 2 + 0.02],
        ),
    ]
    calls += four_box_legs(0.06, leg_h, 0.06, inset_x, inset_z)
    cutters = []
    for i in range(n):
        x = (i - (n - 1) / 2) * (w / (n + 1))
        name = f"divot{i}"
        calls.append(
            prim(
                "icosphere",
                name,
                params={"radius": 0.09, "subdivisions": 2},
                translation=[x, leg_h + t - 0.03, 0],
            )
        )
        cutters.append(name)
    calls.append(boolean_call("difference", ["seat"] + cutters))
    color, name = wood(wood_i)
    names = ["seat", "back", "leg_fl", "leg_fr", "leg_bl", "leg_br"]
    calls.append(material_call(names, color, name, roughness=0.75))
    notes = f"{n} small spheres sunk into the seat top and subtracted in one boolean call -- a repeating row of identical divots."
    return calls, notes


def bb_dresser_three_drawers(w=0.85, d=0.5, h=0.9, wood_i=1):
    calls = [
        prim("box", "body", params={"size": [w, h - 0.03, d]}, translation=[0, (h - 0.03) / 2, 0]),
        prim(
            "box", "top", params={"size": [w + 0.02, 0.03, d + 0.03]}, translation=[0, h - 0.015, 0]
        ),
    ]
    drawer_h = (h - 0.15) / 3
    color, name = wood(wood_i)
    accent = darker(color)
    for i in range(3):
        y = 0.08 + drawer_h * (i + 0.5)
        front = f"drawer{i}"
        cut = f"cut{i}"
        calls.append(
            prim(
                "box",
                front,
                params={"size": [w - 0.06, drawer_h - 0.02, 0.03]},
                translation=[0, y, d / 2 + 0.015],
            )
        )
        calls.append(
            prim(
                "cylinder",
                cut,
                params={"radius": 0.03, "height": 0.08, "segments": 14},
                translation=[0, y, d / 2 + 0.015],
                rotation=[90, 0, 0],
            )
        )
        calls.append(boolean_call("difference", [front, cut]))
    calls.append(material_call(["body", "top"], color, name, roughness=0.65))
    calls.append(
        material_call(
            ["drawer0", "drawer1", "drawer2"], accent, f"{name} Drawer Front", roughness=0.6
        )
    )
    notes = (
        "three drawer fronts, each with its own boolean-cut recessed handle -- three separate boolean calls "
        "because each drawer must stay its own object; the fronts get a darker accent tone (a second "
        "clay_material call) so the panel seams and the cut holes actually read at gallery-render scale."
    )
    return calls, notes


def bb_coat_rack_radial_pegs(post_h=1.7, post_r=0.03, n=6, peg_len=0.12, wood_i=2):
    calls = [
        prim(
            "cylinder",
            "post",
            params={"radius": post_r, "height": post_h, "segments": 16},
            translation=[0, post_h / 2, 0],
        ),
        prim(
            "cylinder",
            "peg",
            params={"radius": 0.014, "height": peg_len, "segments": 10},
            translation=[post_r + peg_len / 2, post_h - 0.12, 0],
            rotation=[0, 0, 90],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["post", "peg"], color, name, roughness=0.6))
    calls.append(select_call(["peg"]))
    calls.append(op_call("array-radial", {"count": n, "angle": 360.0, "axis": AXIS["y"]}))
    notes = f"one peg, coloured, array-radial'd {n} times around the central post -- a ring of pegs about the world Y axis, post at the origin like the spoked-hub pattern."
    return calls, notes


def bb_side_table_one_lathe_leg(top_r=0.24, top_t=0.03, h=0.5, wood_i=0):
    calls = [
        prim(
            "cylinder",
            "top",
            params={"radius": top_r, "height": top_t, "segments": 24},
            translation=[0, h - top_t / 2, 0],
        ),
        prim(
            "lathe",
            "leg",
            params={
                "profile": turned_leg_profile(
                    h - top_t, foot_r=0.08, shaft_r=0.03, bulge_r=0.06, top_r=0.045
                ),
                "segments": 16,
            },
            translation=[0, (h - top_t) / 2, 0],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["top", "leg"], color, name, roughness=0.55))
    notes = "a round top over one central lathe-turned leg -- lathe rather than a plain cylinder pedestal."
    return calls, notes


def bb_garden_bench_cutouts(w=1.4, d=0.34, leg_h=0.45, back_h=0.4, wood_i=2):
    inset_x, inset_z = w / 2 - 0.08, d / 2 - 0.04
    calls = [
        prim("box", "seat", params={"size": [w, 0.05, d]}, translation=[0, leg_h + 0.025, 0]),
        prim(
            "box",
            "back",
            params={"size": [w, back_h, 0.04]},
            translation=[0, leg_h + 0.05 + back_h / 2, -d / 2 + 0.02],
        ),
    ]
    calls += four_box_legs(0.06, leg_h, 0.06, inset_x, inset_z)
    cutters = []
    for i in range(3):
        x = (i - 1) * (w / 4)
        name = f"cut{i}"
        calls.append(
            prim(
                "icosphere",
                name,
                params={"radius": 0.11, "subdivisions": 2},
                translation=[x, leg_h + 0.05 + back_h / 2, -d / 2 + 0.02],
            )
        )
        cutters.append(name)
    calls.append(boolean_call("difference", ["back"] + cutters))
    color, name = wood(wood_i)
    names = ["seat", "back", "leg_fl", "leg_fr", "leg_bl", "leg_br"]
    calls.append(material_call(names, color, name, roughness=0.8))
    notes = "three sphere cutters subtracted from the backrest in one boolean call -- decorative cut-outs along the back."
    return calls, notes


def bb_throne_emblem(w=0.7, d=0.6, leg_h=0.5, back_h=1.1, wood_i=3):
    inset_x, inset_z = w / 2 - 0.06, d / 2 - 0.06
    calls = [
        prim("box", "seat", params={"size": [w, 0.06, d]}, translation=[0, leg_h + 0.03, 0]),
        prim(
            "box",
            "back",
            params={"size": [w, back_h, 0.06]},
            translation=[0, leg_h + 0.06 + back_h / 2, -d / 2 + 0.03],
        ),
        prim(
            "box",
            "arm_l",
            params={"size": [0.08, 0.22, d]},
            translation=[w / 2 - 0.04, leg_h + 0.06 + 0.11, 0],
        ),
        prim(
            "box",
            "arm_r",
            params={"size": [0.08, 0.22, d]},
            translation=[-(w / 2 - 0.04), leg_h + 0.06 + 0.11, 0],
        ),
        prim(
            "torus",
            "emblem_cut",
            params={"radius": 0.14, "tube": 0.09, "segments": 24, "sides": 10},
            translation=[0, leg_h + 0.06 + back_h * 0.62, -d / 2 + 0.03],
            rotation=[90, 0, 0],
        ),
    ]
    calls += four_box_legs(0.06, leg_h, 0.06, inset_x, inset_z)
    calls.append(boolean_call("difference", ["back", "emblem_cut"]))
    color, name = wood(wood_i)
    names = ["seat", "back", "arm_l", "arm_r", "leg_fl", "leg_fr", "leg_bl", "leg_br"]
    calls.append(material_call(names, color, name, roughness=0.55))
    notes = "a high back with a torus cut clean through it as a circular emblem, one boolean call, a tall carved-looking throne."
    return calls, notes


def bb_workbench_drawer_slots(w=1.6, d=0.6, h=0.85, n=4, wood_i=1):
    inset_x, inset_z = w / 2 - 0.08, d / 2 - 0.06
    calls = [prim("box", "top", params={"size": [w, 0.05, d]}, translation=[0, h - 0.025, 0])]
    calls += four_box_legs(0.07, h - 0.05, 0.07, inset_x, inset_z)
    slot_w = (w - 0.2) / n - 0.03
    calls.append(
        prim(
            "box",
            "slot",
            params={"size": [slot_w, 0.14, 0.015]},
            translation=[-(w / 2) + 0.12 + slot_w / 2, h - 0.3, d / 2 + 0.008],
        )
    )
    color, name = wood(wood_i)
    calls.append(
        material_call(["top", "leg_fl", "leg_fr", "leg_bl", "leg_br"], color, name, roughness=0.7)
    )
    calls.append(material_call(["slot"], METAL_STEEL, "Slot Trim", roughness=0.4, metallic=0.7))
    calls.append(select_call(["slot"]))
    calls.append(op_call("array-linear", {"count": n, "x": slot_w + 0.03, "y": 0.0, "z": 0.0}))
    notes = f"one drawer-slot face, array-linear'd {n} times along the front apron -- a repeating row, no boolean needed since the slots sit proud rather than cut through."
    return calls, notes


def bb_nightstand_finger_pull(w=0.42, d=0.38, h=0.5, wood_i=1):
    body_h = h - 0.03
    calls = [
        prim("box", "body", params={"size": [w, body_h, d]}, translation=[0, body_h / 2, 0]),
        prim(
            "box", "top", params={"size": [w + 0.03, 0.03, d + 0.03]}, translation=[0, h - 0.015, 0]
        ),
        prim(
            "box",
            "drawer",
            params={"size": [w - 0.05, 0.16, 0.03]},
            translation=[0, body_h - 0.13, d / 2 + 0.015],
        ),
        prim(
            "cylinder",
            "pull_cut",
            params={"radius": 0.032, "height": 0.08, "segments": 14},
            translation=[0, body_h - 0.13, d / 2 + 0.015],
            rotation=[90, 0, 0],
        ),
    ]
    calls.append(boolean_call("difference", ["drawer", "pull_cut"]))
    color, name = wood(wood_i)
    calls.append(material_call(["body", "top"], color, name, roughness=0.65))
    calls.append(material_call(["drawer"], darker(color), f"{name} Drawer Front", roughness=0.6))
    notes = "a round cylinder bored straight through the drawer face and subtracted -- a finger-pull hole, one boolean call; the drawer front gets a darker accent tone so the cut reads clearly."
    return calls, notes


def bb_wine_barrel_table_hole(radius=0.32, height=0.55, wood_i=1):
    profile = [
        [radius * 0.7, -height / 2],
        [radius * 0.85, -height / 2 + 0.08 * height],
        [radius, -height / 2 + 0.45 * height],
        [radius * 0.85, -height / 2 + 0.82 * height],
        [radius * 0.72, height / 2],
    ]
    calls = [
        prim(
            "lathe",
            "barrel",
            params={"profile": profile, "segments": 20},
            translation=[0, height / 2, 0],
        ),
        prim(
            "cylinder",
            "pole_hole",
            params={"radius": 0.025, "height": height + 0.1, "segments": 12},
            translation=[0, height / 2, 0],
        ),
    ]
    calls.append(boolean_call("difference", ["barrel", "pole_hole"]))
    color, name = wood(wood_i)
    calls.append(material_call(["barrel"], color, name, roughness=0.8))
    notes = "a barrel-profile lathe body with one vertical cylinder bored straight through and subtracted -- an umbrella-pole hole, no separate tabletop object to keep the boolean to one cutter."
    return calls, notes


# =============================================================================
# HARD: a swept or tapered form along a path.
# =============================================================================


def tapered_sweep_leg(name, height, top_r, taper, x, z, *, tilt_axis="x", tilt_deg=0.0, sides=8):
    outline = octagon_outline(top_r)
    rot = [90 + tilt_deg, 0, 0] if tilt_axis == "x" else [90, 0, tilt_deg]
    return prim(
        "sweep",
        name,
        params={"outline": outline, "depth": height, "taper": taper, "sections": 1},
        translation=[x, height / 2, z],
        rotation=rot,
    )


def bb_tapered_leg_standalone(height=0.42, top_r=0.028, taper=0.32, wood_i=1):
    calls = [tapered_sweep_leg("leg", height, top_r, taper, 0.0, 0.0)]
    color, name = wood(wood_i)
    calls.append(material_call(["leg"], color, name, roughness=0.65))
    notes = (
        "a sweep with an octagon outline, rotated 90 about X so the near (untapered) end reads as the thick "
        "top and the far (taper-scaled) end reads as the narrow foot -- a single standalone leg, sized for a chair."
    )
    return calls, notes


def bb_chaise_longue(length=1.4, width=0.5, wood_i=2, fabric=FABRIC_BLUE):
    half = length / 2
    radius = 0.035
    path = [
        [-half, 0.16, 0.0],
        [-half * 0.85, 0.10, 0.0],
        [-half * 0.6, 0.07, 0.0],
        [-half * 0.3, 0.06, 0.0],
        [half * 0.3, 0.06, 0.0],
        [half * 0.6, 0.07, 0.0],
        [half * 0.85, 0.10, 0.0],
        [half, 0.16, 0.0],
    ]
    rail_z = width / 2 - 0.05
    ty = tube_ground_y(path, radius)
    plateau_world = (
        ty - (0.16 + 0.06) / 2 + 0.06
    )  # the flat middle station, after the path's own re-centring
    cushion_y = plateau_world + radius + 0.05
    calls = [
        prim(
            "tube",
            "rail",
            params={"path": path, "radius": radius, "sides": 8},
            translation=[0.0, ty, rail_z],
        ),
        prim(
            "box",
            "cushion",
            params={"size": [length - 0.1, 0.1, width]},
            translation=[0.0, cushion_y, 0.0],
        ),
        prim(
            "box",
            "headrest",
            params={"size": [0.28, 0.16, width]},
            translation=[-half + 0.2, cushion_y + 0.14, 0.0],
            rotation=[0, 0, -14],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["rail"], color, name, roughness=0.6))
    calls.append(
        material_call(["cushion", "headrest"], fabric, "Chaise Upholstery", roughness=0.85)
    )
    calls.append(select_call(["rail"]))
    calls.append(op_call("mirror-copy", {"axis": AXIS["z"], "offset": 0.0}))
    notes = (
        "one tube-swept rail (a shallow curve, both ends turned up near the ground) mirrored across Z for "
        "the second side, with a long cushion and a tilted headrest box resting on top -- translation.y is "
        "solved from the path's own span since ``tube``'s ``path`` is re-centred on its bounding-box centre "
        "before the mesh is built, the same rule every other array-valued parameter here follows."
    )
    return calls, notes


def bb_cabriole_table(top_w=1.1, top_d=0.7, top_t=0.04, leg_h=0.45, wood_i=1):
    inset_x, inset_z = top_w / 2 - 0.1, top_d / 2 - 0.1
    calls = [
        prim(
            "box",
            "top",
            params={"size": [top_w, top_t, top_d]},
            translation=[0, leg_h + top_t / 2, 0],
        )
    ]
    corners = [
        ("fl", inset_x, inset_z, "x", 10.0),
        ("fr", -inset_x, inset_z, "x", 10.0),
        ("bl", inset_x, -inset_z, "z", -10.0),
        ("br", -inset_x, -inset_z, "z", -10.0),
    ]
    for tag, x, z, axis, tilt in corners:
        calls.append(
            tapered_sweep_leg(f"leg_{tag}", leg_h, 0.03, 0.3, x, z, tilt_axis=axis, tilt_deg=tilt)
        )
    color, name = wood(wood_i)
    calls.append(
        material_call(["top", "leg_fl", "leg_fr", "leg_bl", "leg_br"], color, name, roughness=0.55)
    )
    notes = "four sweep legs, tapered and each tilted a few degrees outward from its own corner -- a cabriole-ish splay without a true curved path."
    return calls, notes


def bb_sweeping_bench_rail(width=0.42, wood_i=0, fabric=None):
    radius = 0.03
    path = [
        [0.5, 0.02, 0.0],
        [0.46, 0.16, 0.0],
        [0.36, 0.32, 0.0],
        [0.16, 0.44, 0.0],
        [-0.15, 0.46, 0.0],
        [-0.35, 0.50, 0.0],
        [-0.48, 0.62, 0.0],
        [-0.50, 0.80, 0.0],
        [-0.45, 1.0, 0.0],
    ]
    rail_z = width / 2 - 0.05
    ty = tube_ground_y(path, radius)
    plateau_world = (
        ty - (0.02 + 1.0) / 2 + 0.45
    )  # the seat plateau, after the path's own re-centring
    calls = [
        prim(
            "tube",
            "rail",
            params={"path": path, "radius": radius, "sides": 8},
            translation=[0.0, ty, rail_z],
        ),
        prim(
            "box",
            "seat",
            params={"size": [0.85, 0.05, width + 0.06]},
            translation=[0.05, plateau_world + radius + 0.025, 0.0],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["rail"], color, name, roughness=0.65))
    calls.append(material_call(["seat"], darker(color), f"{name} Seat Plank", roughness=0.6))
    calls.append(select_call(["rail"]))
    calls.append(op_call("mirror-copy", {"axis": AXIS["z"], "offset": 0.0}))
    notes = (
        "one continuous tube-swept rail from a ground-level front foot, up through a seat-height plateau, up "
        "into a tall backrest -- translation.y solved from the path's own span for the same re-centring "
        "reason bb_chaise_longue's notes give; mirrored across Z for the other side, with the seat plank in a "
        "darker accent tone (and slightly overhanging the rails) so it actually reads against the frame."
    )
    return calls, notes


def bb_tusk_leg_stool(seat_r=0.22, seat_t=0.05, leg_h=0.46, n=3, wood_i=2):
    calls = [
        prim(
            "cylinder",
            "seat",
            params={"radius": seat_r, "height": seat_t, "segments": 20},
            translation=[0, leg_h + seat_t / 2, 0],
        )
    ]
    calls.append(
        tapered_sweep_leg(
            "leg", leg_h, 0.03, 0.28, seat_r - 0.05, 0.0, tilt_axis="z", tilt_deg=14.0
        )
    )
    color, name = wood(wood_i)
    calls.append(material_call(["seat", "leg"], color, name, roughness=0.6))
    calls.append(select_call(["leg"]))
    calls.append(op_call("array-radial", {"count": n, "angle": 360.0, "axis": AXIS["y"]}))
    notes = f"one tapered, outward-tilted sweep leg placed off-centre and array-radial'd {n} times about the world Y axis under the seat's own centre -- three tusk-like legs, the spoked-hub pattern applied to a stool."
    return calls, notes


# =============================================================================
# EXTRA: broadens coverage beyond the seed list (lamps in particular, which
# the seed file does not name at all, plus a few more storage/seating shapes).
# =============================================================================


def bb_round_table_three_lathe_legs(top_r=0.55, top_t=0.04, leg_h=0.72, n=3, wood_i=0):
    calls = [
        prim(
            "cylinder",
            "top",
            params={"radius": top_r, "height": top_t, "segments": 28},
            translation=[0, leg_h + top_t / 2, 0],
        )
    ]
    calls.append(
        prim(
            "lathe",
            "leg",
            params={
                "profile": turned_leg_profile(
                    leg_h, foot_r=0.05, shaft_r=0.025, bulge_r=0.04, top_r=0.03
                ),
                "segments": 14,
            },
            translation=[top_r - 0.12, leg_h / 2, 0],
        )
    )
    color, name = wood(wood_i)
    calls.append(material_call(["top", "leg"], color, name, roughness=0.6))
    calls.append(select_call(["leg"]))
    calls.append(op_call("array-radial", {"count": n, "angle": 360.0, "axis": AXIS["y"]}))
    notes = f"a round top over one lathe-turned leg, array-radial'd {n} times evenly around the table's own centre -- three turned legs rather than a single pedestal."
    return calls, notes


def bb_chest_of_drawers_plain(w=0.8, d=0.48, h=1.0, n=3, wood_i=1):
    calls = [
        prim("box", "body", params={"size": [w, h - 0.03, d]}, translation=[0, (h - 0.03) / 2, 0]),
        prim(
            "box", "top", params={"size": [w + 0.02, 0.03, d + 0.03]}, translation=[0, h - 0.015, 0]
        ),
    ]
    drawer_h = (h - 0.15) / n
    fronts, knobs = [], []
    for i in range(n):
        y = 0.08 + drawer_h * (i + 0.5)
        nm, knob = f"drawer{i}", f"knob{i}"
        calls.append(
            prim(
                "box",
                nm,
                params={"size": [w - 0.06, drawer_h - 0.02, 0.025]},
                translation=[0, y, d / 2 + 0.0125],
            )
        )
        calls.append(
            prim(
                "cylinder",
                knob,
                params={"radius": 0.012, "height": 0.03, "segments": 10},
                translation=[0, y, d / 2 + 0.03],
                rotation=[90, 0, 0],
            )
        )
        fronts.append(nm)
        knobs.append(knob)
    color, name = wood(wood_i)
    calls.append(material_call(["body", "top"], color, name, roughness=0.65))
    calls.append(material_call(fronts, darker(color), f"{name} Drawer Front", roughness=0.6))
    calls.append(material_call(knobs, METAL_BRASS, "Brass Fitting", roughness=0.3, metallic=0.85))
    notes = (
        f"a plain carcass with {n} stacked drawer fronts, all flush boxes -- no cut-outs, the easy sibling of "
        "the dresser build; fronts get a darker accent tone and each a small brass knob so the stack actually "
        "reads instead of blending into one flat carcass colour."
    )
    return calls, notes


def bb_corner_shelf(arm=0.6, h=1.2, n=3, wood_i=2):
    t = 0.02
    calls = [
        prim("box", "post", params={"size": [t, h, t]}, translation=[-arm / 2, h / 2, -arm / 2]),
    ]
    names = ["post"]
    for i in range(n):
        frac = i / (n - 1) if n > 1 else 0.0
        y = 0.15 + frac * (h - 0.3)
        nx = f"shelf_x{i}"
        nz = f"shelf_z{i}"
        calls.append(
            prim(
                "box",
                nx,
                params={"size": [arm, t, arm * 0.35]},
                translation=[0.0, y, -arm / 2 + arm * 0.175],
            )
        )
        calls.append(
            prim(
                "box",
                nz,
                params={"size": [arm * 0.35, t, arm]},
                translation=[-arm / 2 + arm * 0.175, y, 0.0],
            )
        )
        names += [nx, nz]
    color, name = wood(wood_i)
    calls.append(material_call(names, color, name, roughness=0.75))
    notes = "an L-shaped pair of shelf boards at each of n heights sharing one corner post -- primitives only, fits into a room corner."
    return calls, notes


def bb_wine_cabinet_lattice(w=0.6, h=1.5, d=0.35, wood_i=1, n_h=4, n_v=4):
    t = 0.03
    calls = carcass_sides(h, w, d, t)
    calls.append(
        prim(
            "box",
            "bar_h",
            params={"size": [w - 2 * t, 0.015, 0.015]},
            translation=[0, 0.15, d / 2 + 0.008],
        )
    )
    calls.append(
        prim(
            "box",
            "bar_v",
            params={"size": [0.015, h - 0.3, 0.015]},
            translation=[-(w / 2 - t - 0.02), h / 2, d / 2 + 0.008],
        )
    )
    color, name = wood(wood_i)
    calls.append(material_call(["side_l", "side_r", "bar_h", "bar_v"], color, name, roughness=0.7))
    calls.append(select_call(["bar_h"]))
    calls.append(
        op_call(
            "array-linear", {"count": n_h, "x": 0.0, "y": (h - 0.3) / max(n_h - 1, 1), "z": 0.0}
        )
    )
    calls.append(select_call(["bar_v"]))
    calls.append(
        op_call(
            "array-linear",
            {"count": n_v, "x": (w - 2 * t - 0.04) / max(n_v - 1, 1), "y": 0.0, "z": 0.0},
        )
    )
    notes = "two arrays (horizontal bars up the case, vertical bars across it) crossing into a lattice door face -- one array-linear call per direction, material set before both."
    return calls, notes


def bb_round_ottoman(radius=0.28, h=0.4, fabric=FABRIC_GREEN):
    calls = [
        prim(
            "cylinder",
            "drum",
            params={"radius": radius, "height": h - 0.05, "segments": 24},
            translation=[0, (h - 0.05) / 2, 0],
        ),
        prim(
            "cylinder",
            "cushion",
            params={"radius": radius + 0.02, "height": 0.06, "segments": 24},
            translation=[0, h - 0.03, 0],
        ),
    ]
    calls.append(material_call(["drum", "cushion"], fabric, "Ottoman Fabric", roughness=0.88))
    notes = "a drum body with a slightly wider, rounder cushion cap on top -- two cylinders, upholstered material."
    return calls, notes


def bb_picnic_table(top_w=1.8, top_d=0.8, bench_d=0.28, h=0.75, seat_h=0.42, wood_i=2):
    inset_x = top_w / 2 - 0.12
    bench_z = top_d / 2 + bench_d / 2 + 0.15
    calls = [
        prim("box", "top", params={"size": [top_w, 0.04, top_d]}, translation=[0, h, 0]),
        prim(
            "box", "leg_l", params={"size": [0.06, h, top_d * 0.8]}, translation=[inset_x, h / 2, 0]
        ),
        prim(
            "box",
            "leg_r",
            params={"size": [0.06, h, top_d * 0.8]},
            translation=[-inset_x, h / 2, 0],
        ),
        prim(
            "box",
            "bench_l",
            params={"size": [top_w, 0.04, bench_d]},
            translation=[0, seat_h, bench_z],
        ),
        prim(
            "box",
            "bench_r",
            params={"size": [top_w, 0.04, bench_d]},
            translation=[0, seat_h, -bench_z],
        ),
    ]
    calls += four_box_legs(0.05, seat_h, 0.05, inset_x, bench_z, prefix="bsup")
    color, name = wood(wood_i)
    names = [
        "top",
        "leg_l",
        "leg_r",
        "bench_l",
        "bench_r",
        "bsup_fl",
        "bsup_fr",
        "bsup_bl",
        "bsup_br",
    ]
    calls.append(material_call(names, color, name, roughness=0.8))
    notes = "a rectangular top on two vertical panel legs, with two bench planks flanking it on their own short support legs -- attached-bench picnic table, primitives only."
    return calls, notes


def bb_daybed_frame(l=1.9, w=0.9, base_h=0.28, head_h=0.55, foot_h=0.32, wood_i=1):
    calls = [
        prim("box", "base", params={"size": [l, base_h, w]}, translation=[0, base_h / 2, 0]),
        prim(
            "box",
            "headboard",
            params={"size": [0.05, head_h, w]},
            translation=[l / 2 - 0.025, head_h / 2, 0],
        ),
        prim(
            "box",
            "footboard",
            params={"size": [0.05, foot_h, w]},
            translation=[-(l / 2 - 0.025), foot_h / 2, 0],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["base", "headboard", "footboard"], color, name, roughness=0.65))
    notes = "a platform base with a tall headboard panel and a shorter footboard panel -- three boxes, a daybed silhouette."
    return calls, notes


def bb_umbrella_stand(radius=0.18, height=0.5, wood_i=None):
    calls = [
        prim(
            "cylinder",
            "stand",
            params={"radius": radius, "height": height, "segments": 24},
            translation=[0, height / 2, 0],
        )
    ]
    cutters = []
    import math

    for i in range(4):
        ang = math.radians(i * 90)
        x, z = (radius - 0.03) * math.cos(ang), (radius - 0.03) * math.sin(ang)
        nm = f"drain{i}"
        calls.append(
            prim(
                "cylinder",
                nm,
                params={"radius": 0.012, "height": 0.06, "segments": 10},
                translation=[x, 0.03, z],
            )
        )
        cutters.append(nm)
    calls.append(boolean_call("difference", ["stand"] + cutters))
    calls.append(
        material_call(["stand"], METAL_STEEL, "Galvanised Steel", roughness=0.35, metallic=0.75)
    )
    notes = "four small cylinder cutters around the base, subtracted in one boolean call, for drainage holes near the floor."
    return calls, notes


def bb_drafting_table(top_w=1.3, top_d=0.75, top_t=0.03, h=0.95, tilt=14.0, wood_i=0):
    calls = [
        prim(
            "box",
            "top",
            params={"size": [top_w, top_t, top_d]},
            translation=[0, h, -0.05],
            rotation=[tilt, 0, 0],
        ),
        prim(
            "box",
            "leg_l",
            params={"size": [0.06, h - 0.1, 0.5]},
            translation=[top_w / 2 - 0.08, (h - 0.1) / 2, 0.1],
        ),
        prim(
            "box",
            "leg_r",
            params={"size": [0.06, h - 0.1, 0.5]},
            translation=[-(top_w / 2 - 0.08), (h - 0.1) / 2, 0.1],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["top", "leg_l", "leg_r"], color, name, roughness=0.55))
    notes = f"a top tilted {tilt} degrees about X on two wide leg panels -- a drafting-table read from rotation alone, no hinge geometry."
    return calls, notes


def bb_sideboard(w=1.2, d=0.5, h=0.85, wood_i=1):
    calls = [
        prim("box", "body", params={"size": [w, h - 0.03, d]}, translation=[0, (h - 0.03) / 2, 0]),
        prim(
            "box", "top", params={"size": [w + 0.03, 0.03, d + 0.03]}, translation=[0, h - 0.015, 0]
        ),
        prim(
            "box",
            "door_l",
            params={"size": [w * 0.32, h - 0.15, 0.025]},
            translation=[w * 0.28, (h - 0.03) / 2, d / 2 + 0.0125],
        ),
        prim(
            "box",
            "door_r",
            params={"size": [w * 0.32, h - 0.15, 0.025]},
            translation=[-w * 0.28, (h - 0.03) / 2, d / 2 + 0.0125],
        ),
        prim(
            "box",
            "drawer",
            params={"size": [w * 0.3, 0.16, 0.025]},
            translation=[0, h - 0.15, d / 2 + 0.0125],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["body", "top"], color, name, roughness=0.6))
    calls.append(
        material_call(
            ["door_l", "door_r", "drawer"], darker(color), f"{name} Panel", roughness=0.55
        )
    )
    notes = (
        "a long low cabinet with two door faces flanking one central drawer front -- a sideboard, primitives "
        "only, panels in a darker accent tone so the front reads."
    )
    return calls, notes


def bb_magazine_rack(w=0.4, h=0.5, d=0.28, n=4, wood_i=2):
    calls = [prim("box", "body", params={"size": [w, h, 0.03]}, translation=[0, h / 2, 0])]
    cutters = []
    slot_w = (w - 0.1) / n - 0.02
    for i in range(n):
        x = (i - (n - 1) / 2) * (w / n)
        nm = f"slot{i}"
        calls.append(
            prim("box", nm, params={"size": [slot_w, h - 0.1, 0.06]}, translation=[x, h / 2, 0])
        )
        cutters.append(nm)
    calls.append(boolean_call("difference", ["body"] + cutters))
    color, name = wood(wood_i)
    calls.append(material_call(["body"], color, name, roughness=0.7))
    notes = f"{n} vertical box cutters through the front face, subtracted in one boolean call -- open slots for magazines."
    return calls, notes


def bb_bunk_bed(l=1.95, w=0.95, lower_h=0.3, upper_h=1.3, post_h=1.6, wood_i=2, n_rungs=4):
    inset_x, inset_z = l / 2 - 0.06, w / 2 - 0.06
    calls = [
        prim("box", "lower", params={"size": [l, 0.15, w]}, translation=[0, lower_h, 0]),
        prim("box", "upper", params={"size": [l, 0.15, w]}, translation=[0, upper_h, 0]),
    ]
    calls += four_box_legs(0.06, post_h, 0.06, inset_x, inset_z, prefix="post")
    calls.append(
        prim(
            "box",
            "rung",
            params={"size": [0.15, 0.02, 0.03]},
            translation=[inset_x + 0.08, 0.3, inset_z],
        )
    )
    color, name = wood(wood_i)
    names = ["lower", "upper", "post_fl", "post_fr", "post_bl", "post_br", "rung"]
    calls.append(material_call(names, color, name, roughness=0.75))
    calls.append(select_call(["rung"]))
    calls.append(
        op_call(
            "array-linear",
            {"count": n_rungs, "x": 0.0, "y": (post_h - 0.6) / max(n_rungs - 1, 1), "z": 0.0},
        )
    )
    notes = "two stacked platform slabs on four corner posts, with one ladder rung array-linear'd up one post -- a bunk bed."
    return calls, notes


def bb_floor_lamp(base_h=0.14, base_r=0.18, pole_h=1.3, shade_h=0.3, shade_r=0.2, wood_i=1):
    calls = [
        prim(
            "lathe",
            "base",
            params={
                "profile": lamp_base_profile(
                    base_h, foot_r=base_r, bulge_r=base_r * 0.9, neck_r=0.025
                ),
                "segments": 24,
            },
            translation=[0, base_h / 2, 0],
        ),
        prim(
            "cylinder",
            "pole",
            params={"radius": 0.016, "height": pole_h, "segments": 12},
            translation=[0, base_h + pole_h / 2, 0],
        ),
        prim(
            "cylinder",
            "shade",
            params={"radius": shade_r, "height": shade_h, "segments": 24},
            translation=[0, base_h + pole_h + shade_h / 2, 0],
        ),
    ]
    color, name = wood(wood_i)
    calls.append(material_call(["base", "pole"], color, name, roughness=0.5))
    calls.append(material_call(["shade"], FABRIC_CREAM, "Shade Fabric", roughness=0.7))
    notes = (
        "a lathe-turned wooden base, a thin pole, and a drum shade cylinder on top -- a floor lamp."
    )
    return calls, notes


def bb_table_lamp(base_h=0.22, base_r=0.11, neck_h=0.16, shade_h=0.2, shade_r=0.15, wood_i=None):
    calls = [
        prim(
            "lathe",
            "base",
            params={
                "profile": lamp_base_profile(
                    base_h, foot_r=base_r, bulge_r=base_r * 1.05, neck_r=0.02
                ),
                "segments": 20,
            },
            translation=[0, base_h / 2, 0],
        ),
        prim(
            "cylinder",
            "neck",
            params={"radius": 0.014, "height": neck_h, "segments": 10},
            translation=[0, base_h + neck_h / 2, 0],
        ),
        prim(
            "cylinder",
            "shade",
            params={"radius": shade_r, "height": shade_h, "segments": 20},
            translation=[0, base_h + neck_h + shade_h / 2, 0],
        ),
    ]
    calls.append(material_call(["base", "neck"], (0.65, 0.62, 0.55), "Ceramic", roughness=0.35))
    calls.append(material_call(["shade"], FABRIC_CREAM, "Shade Fabric", roughness=0.75))
    notes = "a bulbous lathe-turned ceramic-toned base under a small drum shade -- a table lamp."
    return calls, notes


def bb_desk_lamp(base_r=0.09, base_h=0.02, arm_len=0.32, wood_i=None):
    calls = [
        prim(
            "cylinder",
            "base",
            params={"radius": base_r, "height": base_h, "segments": 20},
            translation=[0, base_h / 2, 0],
        ),
        prim(
            "cylinder",
            "arm",
            params={"radius": 0.012, "height": arm_len, "segments": 10},
            translation=[0.0, base_h + arm_len / 2 * 0.85, -arm_len * 0.06],
            rotation=[18, 0, 0],
        ),
        prim(
            "cone",
            "shade",
            params={"radius": 0.06, "height": 0.1, "segments": 16},
            translation=[0.0, base_h + arm_len * 0.85, -arm_len * 0.22],
            rotation=[160, 0, 0],
        ),
    ]
    calls.append(
        material_call(
            ["base", "arm", "shade"], METAL_GUNMETAL, "Lamp Metal", roughness=0.4, metallic=0.6
        )
    )
    notes = "a plain cylinder base, a tilted cylinder arm and a cone shade angled to face down -- a simple desk lamp, no lathe needed."
    return calls, notes


# =============================================================================
# Variant/phrasing lists -- one entry per (prompt, kwargs) instantiated as one
# record. Rephrased away from two seed lines that collide with the held-out
# bigram filter (see the report): seed 1 ("four-legged wooden") and seed 66
# ("wooden chair leg") both share an adjacent content-word bigram with the
# held-out chair subject and are reworded below without changing intent.
# =============================================================================

instantiate(
    bb_table_plain,
    [
        ("a plain wooden table on four legs, with a rectangular top", dict()),
        (
            "simple dining table, rectangular top, four square legs, about 1.4m long",
            dict(top_w=1.4, top_d=0.8, wood_i=0),
        ),
        (
            "modern kitchen table, pale oak, four straight legs set in from the corners",
            dict(wood_i=0),
        ),
        (
            "a sturdy medieval refectory table, dark walnut, long rectangular top",
            dict(top_w=1.8, top_d=0.7, leg_w=0.08, leg_d=0.08, wood_i=1),
        ),
        (
            "small sci-fi mess-hall table, four legs, pale wood-toned composite top",
            dict(top_w=1.1, top_d=0.6, wood_i=3),
        ),
    ],
)

instantiate(
    bb_nightstand,
    [
        ("a small square nightstand with one drawer", dict()),
        ("bedside table, one drawer with a brass knob, about 40cm wide", dict(w=0.4, wood_i=1)),
        ("cherry-wood nightstand, single drawer, small footprint", dict(wood_i=4)),
        ("a modern minimalist nightstand, one drawer, pale finish", dict(wood_i=2, drawer_h=0.12)),
        ("fantasy bedside chest with one drawer, dark ebony wood", dict(wood_i=3)),
    ],
)

instantiate(
    bb_bookshelf_open,
    [
        ("a wooden bookshelf with four open shelves", dict()),
        ("simple oak bookcase, four shelves, no doors", dict(wood_i=0)),
        (
            "tall open bookshelf, walnut finish, four evenly spaced shelves about 1.5m high",
            dict(h=1.5, wood_i=1, n=4),
        ),
        ("modern open-shelf bookcase, four shelves, pale pine", dict(wood_i=2)),
        ("a scholar's open bookshelf, four shelves, dark wood, medieval study", dict(wood_i=3)),
    ],
)

instantiate(
    bb_platform_bed,
    [
        ("a low platform bed base with no headboard, 2m by 1.6m", dict()),
        ("plain platform bed frame, 2 by 1.6 metres, no headboard, walnut", dict(wood_i=1)),
        ("minimalist low bed base, no headboard, queen-sized", dict(w=2.0, l=1.5, wood_i=2)),
        ("a simple wooden bed platform, low profile, no headboard at all", dict(wood_i=0)),
        ("sci-fi barracks bunk base, low slab, no headboard", dict(h=0.28, wood_i=3)),
    ],
)

instantiate(
    bb_plank_bench,
    [
        ("a rustic bench made from a single thick plank on two block legs", dict()),
        ("rough-hewn bench, one thick plank, two solid block legs", dict(wood_i=1)),
        (
            "medieval tavern bench, single heavy plank seat, two square blocks underneath",
            dict(wood_i=3, t=0.08),
        ),
        ("small rustic plank bench, about 1.2m long", dict(w=1.2, wood_i=2)),
        ("plain garden bench, one plank on two blocks, weathered pine", dict(wood_i=2)),
    ],
)

instantiate(
    bb_tall_bookshelf,
    [
        ("a tall narrow bookshelf, five shelves, leaning slightly", dict()),
        ("narrow leaning bookcase, five shelves, tilted a few degrees", dict(lean=4.0, wood_i=1)),
        (
            "tall thin bookshelf that leans back slightly against the wall, five shelves",
            dict(wood_i=2),
        ),
        ("a five-shelf study bookcase, narrow, faint backward lean", dict(lean=2.5, wood_i=0)),
        (
            "fantasy library bookshelf, tall and narrow, five shelves, leaning",
            dict(wood_i=3, h=2.0),
        ),
    ],
)

instantiate(
    bb_toy_chest,
    [
        ("a child's toy chest shaped like a plain rectangular box", dict()),
        ("simple toy box, rectangular, painted pine", dict(wood_i=2)),
        ("a plain wooden chest for a child's toys, no lid detail, just a box", dict(wood_i=0)),
        ("small square toy chest, cherry-stained", dict(w=0.55, d=0.4, h=0.4, wood_i=4)),
        ("medieval-style storage chest for a child's room, plain box shape", dict(wood_i=1)),
    ],
)

instantiate(
    bb_stool_round_3legs,
    [
        ("a wooden stool with a round seat and three straight legs", dict()),
        ("simple round stool, three straight legs set 120 degrees apart", dict(wood_i=1)),
        (
            "small round milking stool, three legs, slight outward splay",
            dict(seat_r=0.15, leg_h=0.35, wood_i=2),
        ),
        ("a plain three-legged round stool, about 45cm tall", dict(leg_h=0.42, wood_i=0)),
        ("fantasy tavern stool, round seat, three sturdy legs", dict(wood_i=3)),
    ],
)

instantiate(
    bb_desk_panel_legs,
    [
        ("an office desk with a flat top and two solid side panels for legs", dict()),
        ("modern office desk, flat top, two panel legs, walnut finish", dict(wood_i=1)),
        ("simple work desk with side panels instead of four legs", dict(wood_i=2)),
        ("a wide desk, flat rectangular top, two solid end panels", dict(top_w=1.5, wood_i=0)),
        ("sci-fi terminal desk, flat slab top, two solid support panels", dict(wood_i=3)),
    ],
)

instantiate(
    bb_round_side_table,
    [
        ("a small round side table about 40 centimetres tall", dict()),
        ("round pedestal side table, oak, 40cm tall", dict(wood_i=0)),
        ("small circular accent table on a plain cylinder pedestal", dict(wood_i=1)),
        ("a round side table, walnut, slightly taller than usual", dict(h=0.48, wood_i=1)),
        ("modern round side table, pale wood, small foot disc", dict(wood_i=2)),
    ],
)

instantiate(
    bb_ottoman,
    [
        ("a padded ottoman shaped like a thick cube", dict()),
        ("cube-shaped ottoman, cream fabric, thickly padded look", dict()),
        ("a soft square ottoman, deep red upholstery", dict(fabric=FABRIC_RED)),
        ("small padded footstool cube, blue fabric", dict(size=0.42, fabric=FABRIC_BLUE)),
        ("modern living-room ottoman, cube shape, green fabric", dict(fabric=FABRIC_GREEN)),
    ],
)

instantiate(
    bb_wardrobe,
    [
        ("a wardrobe with two tall square doors", dict()),
        ("tall wardrobe, two square door faces, walnut", dict(wood_i=1)),
        ("simple bedroom wardrobe, two doors, no boolean hinges needed", dict(wood_i=0)),
        ("a wide wardrobe cabinet, two tall doors, dark ebony", dict(w=1.0, wood_i=3)),
        ("medieval clothing armoire, two square doors, heavy oak", dict(wood_i=0)),
    ],
)

instantiate(
    bb_crate_end_table,
    [
        ("a wooden crate-style end table", dict()),
        ("crate-shaped side table with two wraparound battens, pine", dict(wood_i=2)),
        ("small end table that reads like a shipping crate", dict(wood_i=1)),
        ("rustic crate end table, cherry-stained battens", dict(wood_i=4)),
        ("sci-fi cargo-crate side table, metal-toned battens", dict(wood_i=3)),
    ],
)

instantiate(
    bb_dining_bench_long,
    [
        ("a long dining bench with a plain rectangular seat", dict()),
        ("simple long bench, rectangular seat, four square legs, oak", dict(wood_i=0)),
        ("plain dining bench, about 1.6m, walnut", dict(w=1.6, wood_i=1)),
        ("a narrow rectangular bench for a dining table", dict(d=0.3, wood_i=2)),
        ("medieval hall bench, long rectangular seat, four legs", dict(w=1.8, wood_i=1)),
    ],
)

instantiate(
    bb_folding_table_metal,
    [
        ("a metal folding table with a thin rectangular top", dict()),
        ("thin-topped folding table, steel legs, utility style", dict()),
        ("small metal folding table for a garage", dict(top_w=1.1, top_d=0.55)),
        ("plain steel folding table, rectangular, thin top", dict(top_w=1.6)),
        ("sci-fi field table, thin metal top and legs", dict()),
    ],
)

instantiate(
    bb_dining_chair_solid_back,
    [
        ("a plain wooden dining chair with a solid back panel", dict()),
        ("simple dining chair, solid backrest (no slats), square seat", dict(wood_i=1)),
        ("a straightforward chair with a flat solid back panel, four legs", dict(wood_i=0)),
        ("cherry-wood dining chair, solid back, plain square legs", dict(wood_i=4)),
        ("medieval dining chair, solid back panel, sturdy square legs", dict(wood_i=1, leg_h=0.42)),
    ],
)

instantiate(
    bb_three_tier_shelf,
    [
        ("a small three-tier shelf unit, each shelf a thin rectangle", dict()),
        ("three-tier display shelf, thin rectangular boards, oak posts", dict(wood_i=0)),
        ("small three-shelf stand, walnut", dict(wood_i=1)),
        ("compact three-tier shelving unit for a corner", dict(w=0.45, wood_i=2)),
        ("modern three-tier plant shelf, thin boards, pale wood", dict(wood_i=2)),
    ],
)

instantiate(
    bb_coffee_table_squat,
    [
        ("a squat wooden coffee table, 30cm tall and a metre wide", dict()),
        ("low coffee table, one metre wide, oak", dict(wood_i=0)),
        ("small squat coffee table, walnut, 30cm tall", dict(wood_i=1)),
        ("modern low coffee table, wide slab top", dict(top_w=1.1, wood_i=2)),
        (
            "a compact squat coffee table for a small living room",
            dict(top_w=0.85, top_d=0.5, wood_i=1),
        ),
    ],
)

instantiate(
    bb_cargo_bench_scifi,
    [
        ("a sci-fi cargo bench, a simple slab on two support blocks", dict()),
        ("industrial cargo-hold bench, gunmetal slab on two blocks", dict()),
        ("simple sci-fi utility bench, alloy slab seat", dict(w=1.5)),
        ("compact cargo bench for a small airlock room", dict(w=0.9, d=0.32)),
        ("wide sci-fi mess-hall bench, slab seat on two blocks", dict(w=1.6)),
    ],
)

instantiate(
    bb_minimalist_desk_central_leg,
    [
        ("a modern minimalist desk with a single slab top and one thick central leg", dict()),
        ("minimalist desk, one thick central support, wide slab top", dict(wood_i=3)),
        ("simple modern desk, single central leg block, dark wood", dict(wood_i=3)),
        ("small minimalist writing desk, one central leg", dict(top_w=1.1, wood_i=2)),
        ("wide minimalist desk, thick central support, pale finish", dict(top_w=1.6, wood_i=2)),
    ],
)

instantiate(
    bb_footstool_box,
    [
        ("a medieval wooden footstool, just a small flat box", dict()),
        ("plain flat footstool box, small", dict(wood_i=1)),
        ("tiny wooden footstool, dark wood, medieval hall", dict(wood_i=3)),
        ("a small low footstool, just a box", dict(h=0.15, wood_i=0)),
        ("rustic footstool, flat wooden box", dict(wood_i=2)),
    ],
)

instantiate(
    bb_log_bench,
    [
        ("a fantasy campfire log bench, a plain cylinder on two blocks", dict()),
        ("simple log bench for sitting around a campfire", dict(wood_i=2)),
        ("rustic log seat, cylindrical trunk on two blocks", dict(radius=0.18, wood_i=1)),
        ("a short campfire log bench", dict(length=0.9, wood_i=2)),
        ("medieval camp log bench, thick trunk, two blocks", dict(radius=0.2, wood_i=1)),
    ],
)

instantiate(
    bb_vanity_cabinet,
    [
        ("a bathroom vanity cabinet, a rectangular box with a flat top", dict()),
        ("simple bathroom vanity, flat overhanging top, walnut", dict(wood_i=1)),
        ("small vanity cabinet, rectangular carcass, pale wood", dict(w=0.65, wood_i=2)),
        ("modern bathroom vanity cabinet, wide, flat top", dict(w=0.95, wood_i=0)),
        ("plain vanity cabinet box, dark wood, flat top", dict(wood_i=3)),
    ],
)

instantiate(
    bb_wine_rack_open_frame,
    [
        ("a tall wine rack shaped like an open box frame", dict()),
        ("simple open-frame wine rack, four posts, top and bottom rails", dict(wood_i=1)),
        ("tall narrow wine rack frame, no shelves yet, just the frame", dict(h=1.7, wood_i=2)),
        ("a plain open wine rack frame for a cellar", dict(wood_i=0)),
        ("modern open-box wine rack frame, pale wood", dict(wood_i=2)),
    ],
)

instantiate(
    bb_plant_stand,
    [
        ("a low wooden plant stand, a small square top on four thin legs", dict()),
        ("small plant stand, thin legs, square top, walnut", dict(wood_i=1)),
        ("tall narrow plant stand for a single pot", dict(h=0.7, wood_i=2)),
        ("a simple plant stand, four thin legs", dict(wood_i=0)),
        ("modern plant stand, pale wood, small top", dict(wood_i=2)),
    ],
)

# --- medium ------------------------------------------------------------------

instantiate(
    bb_dining_table_lathe_legs,
    [
        ("a wooden dining table with four lathe-turned legs", dict()),
        ("dining table, four turned legs, walnut", dict(wood_i=1)),
        (
            "a large formal dining table, four lathe-turned legs, dark wood",
            dict(top_w=1.8, top_d=0.95, wood_i=3),
        ),
        ("small kitchen table with turned legs", dict(top_w=1.1, top_d=0.7, wood_i=0)),
        (
            "medieval banquet table, four heavy turned legs",
            dict(top_w=2.0, top_d=0.9, leg_h=0.75, wood_i=1),
        ),
        ("modern dining table with classic turned legs, pale oak", dict(wood_i=0)),
    ],
)

instantiate(
    bb_armchair_cutout,
    [
        ("an armchair with a boolean-cut hand-hold in each armrest", dict()),
        ("wooden armchair, each armrest bored through with a hand-hold", dict(wood_i=1)),
        ("simple armchair with a carrying hole through each arm", dict(wood_i=2)),
        (
            "a sturdy armchair, cylindrical hand-holds cut through both arms",
            dict(seat=0.55, wood_i=3),
        ),
        ("fantasy hall armchair, hand-holds bored through the armrests", dict(wood_i=1)),
        ("small armchair, arms each pierced with a round hand-hold", dict(seat=0.46, wood_i=0)),
    ],
)

instantiate(
    bb_pedestal_table_lathe_column,
    [
        ("a round pedestal table standing on a single lathe-turned column", dict()),
        ("round table on one turned pedestal column, walnut", dict(wood_i=1)),
        ("small round pedestal table, lathe-turned support", dict(top_r=0.4, col_h=0.55, wood_i=0)),
        ("large round pedestal dining table on one turned column", dict(top_r=0.7, wood_i=3)),
        ("modern round table, single turned pedestal, pale wood", dict(wood_i=2)),
        ("fantasy round table on a thick turned pedestal", dict(top_r=0.6, col_h=0.72, wood_i=1)),
    ],
)

instantiate(
    bb_bar_stool_lathe_leg,
    [
        ("a bar stool with a lathe-turned leg and a round padded seat", dict()),
        ("tall bar stool, one turned leg, red padded seat", dict()),
        ("bar stool with a footring and turned leg, walnut", dict(wood_i=1)),
        ("short pub stool, turned leg, round cushion seat", dict(leg_h=0.6, wood_i=2)),
        ("modern bar stool, turned leg, blue upholstered seat", dict(wood_i=0)),
        ("tall counter stool with a lathe-turned leg", dict(leg_h=0.8, wood_i=3)),
    ],
)

instantiate(
    bb_four_poster_bed,
    [
        ("a four-poster bed with lathe-turned corner posts", dict()),
        ("four-poster bed frame, turned corner posts, walnut", dict(wood_i=1)),
        ("grand four-poster bed, tall turned posts, dark wood", dict(post_h=1.8, wood_i=3)),
        (
            "small four-poster bed for a child's room, shorter turned posts",
            dict(l=1.4, w=1.1, post_h=1.1, wood_i=2),
        ),
        ("medieval four-poster bed, heavy turned posts", dict(post_h=1.6, wood_i=1)),
        ("modern four-poster bed frame, pale turned posts", dict(wood_i=0)),
    ],
)

instantiate(
    bb_bookshelf_support_pegs,
    [
        ("a bookshelf whose shelves are held up by a repeating row of support pegs", dict()),
        ("bookcase with shelves resting on a row of small pegs, walnut", dict(wood_i=1)),
        ("tall bookshelf, pegged shelf supports, five levels", dict(n=5, h=1.7, wood_i=2)),
        ("simple peg-supported bookshelf", dict(wood_i=0)),
        ("modern bookshelf with peg-and-hole shelf supports", dict(wood_i=2)),
        ("fantasy library shelf, pegged supports along the sides", dict(wood_i=3)),
    ],
)

instantiate(
    bb_wine_rack_grid_holes,
    [
        ("a wine rack with a grid of round bottle holes cut through the front", dict()),
        ("wine rack, 2x3 grid of bottle holes, walnut", dict(wood_i=1)),
        ("small wine rack with a grid of holes for six bottles", dict(wood_i=2)),
        ("large wine rack, 3x4 grid of bottle holes", dict(rows=3, cols=4, w=0.7, h=0.5, wood_i=3)),
        ("modern wine rack panel, round holes cut in a grid", dict(wood_i=0)),
        ("rustic wine rack, grid of bottle holes bored through the face", dict(wood_i=1)),
    ],
)

instantiate(
    bb_rocking_chair_runners,
    [
        ("a rocking chair with two curved runners under a plain seat", dict()),
        ("simple rocking chair, curved wooden runners, walnut", dict(wood_i=1)),
        ("small rocking chair for a porch, curved runners", dict(wood_i=2)),
        ("nursery rocking chair, gentle curved runners", dict(seat=0.4, wood_i=0)),
        ("rustic rocking chair, thick curved runners, dark wood", dict(wood_i=3)),
        ("modern rocking chair, plain seat on curved runners", dict(wood_i=2)),
    ],
)

instantiate(
    bb_folding_camp_chair,
    [
        ("a folding camp chair with a fabric seat stretched between crossed legs", dict()),
        ("camp chair, canvas seat between an X of crossed legs", dict()),
        ("small folding stool-chair, crossed legs, green fabric seat", dict()),
        ("military-style folding camp chair, crossed wooden legs", dict(wood_i=2)),
        ("outdoor folding chair, fabric seat, crossed legs, blue canvas", dict(fabric=FABRIC_BLUE)),
        (
            "compact folding camp chair for hiking, crossed legs",
            dict(leg_len=0.4, fabric=FABRIC_CREAM),
        ),
    ],
)

instantiate(
    bb_vanity_table_mirror,
    [
        ("a vanity table with a round mirror mounted above a cabinet", dict()),
        ("dressing table, round mirror on turned posts above the cabinet, walnut", dict(wood_i=1)),
        ("small vanity table with a round mirror, pale wood", dict(wood_i=2)),
        ("elegant vanity table, brass-framed round mirror above the cabinet", dict(wood_i=3)),
        ("modern vanity table, round mirror mounted on two turned posts", dict(wood_i=0)),
        ("wide vanity table with a large round mirror above it", dict(mirror_r=0.28, wood_i=1)),
    ],
)

instantiate(
    bb_office_chair_radial_base,
    [
        ("an office chair on a five-point wheeled base, arranged in a radial pattern", dict()),
        ("office chair, five-point caster base, blue fabric seat", dict()),
        ("simple task chair on a radial wheeled base", dict()),
        ("wide office chair, five-point base, dark upholstery", dict(seat=0.46)),
        ("modern office chair, five casters arranged radially", dict()),
        ("compact office chair, five-point wheel base, small seat", dict(seat=0.38)),
    ],
)

instantiate(
    bb_church_pew_divots,
    [
        ("a church pew with a repeating row of identical seat divots", dict()),
        ("wooden pew, five identical seat divots carved into the bench", dict(wood_i=1)),
        ("long church pew, repeating row of scooped seat divots", dict(w=2.0, n=6, wood_i=3)),
        ("small pew, three seat divots, walnut", dict(w=1.1, n=3, wood_i=1)),
        ("chapel pew bench, evenly spaced seat divots", dict(wood_i=2)),
        ("rustic pew, row of shallow seat divots cut into the seat", dict(wood_i=0)),
    ],
)

instantiate(
    bb_dresser_three_drawers,
    [
        ("a dresser with three drawers, each front cut with a recessed handle", dict()),
        ("bedroom dresser, three drawers, recessed handles, walnut", dict(wood_i=1)),
        ("wide dresser, three drawers each with a bored recessed handle", dict(w=1.0, wood_i=2)),
        ("small dresser, three drawers, recessed cut handles", dict(w=0.7, wood_i=0)),
        ("dark-wood dresser, three drawers, recessed handle cut-outs", dict(wood_i=3)),
        ("modern dresser, three drawers, round recessed handles", dict(wood_i=2)),
    ],
)

instantiate(
    bb_coat_rack_radial_pegs,
    [
        ("a coat rack with a ring of pegs arranged radially around a central post", dict()),
        ("standing coat rack, six pegs in a ring around the post, walnut", dict(wood_i=1)),
        ("tall coat stand, radial ring of pegs near the top", dict(post_h=1.8, wood_i=2)),
        ("simple coat rack post with a ring of eight short pegs", dict(n=8, wood_i=0)),
        ("rustic coat rack, radial pegs around a thick post", dict(post_r=0.04, wood_i=3)),
        ("modern coat stand, five pegs arranged radially", dict(n=5, wood_i=2)),
    ],
)

instantiate(
    bb_side_table_one_lathe_leg,
    [
        ("a side table with one lathe-turned leg under a round top", dict()),
        ("small side table, single turned leg, walnut", dict(wood_i=1)),
        ("round accent table on one lathe-turned leg", dict(wood_i=2)),
        ("tall side table, one turned leg, dark wood", dict(h=0.58, wood_i=3)),
        ("modern round side table with a single turned support", dict(wood_i=0)),
        ("small round table on one delicate turned leg", dict(top_r=0.2, wood_i=1)),
    ],
)

instantiate(
    bb_garden_bench_cutouts,
    [
        ("a garden bench with decorative cut-outs along the backrest", dict()),
        ("outdoor garden bench, three decorative cut-outs in the back, pine", dict(wood_i=2)),
        ("long park bench with cut-out shapes along the backrest", dict(w=1.7, wood_i=1)),
        ("small garden bench, cut-out decoration in the back panel", dict(w=1.1, wood_i=0)),
        ("rustic garden bench, round cut-outs along the back", dict(wood_i=3)),
        ("modern outdoor bench, decorative back cut-outs, pale wood", dict(wood_i=2)),
    ],
)

instantiate(
    bb_throne_emblem,
    [
        ("a fantasy throne with a high back and a carved circular emblem cut into it", dict()),
        ("wooden throne, tall back, circular emblem bored through the centre", dict(wood_i=3)),
        (
            "grand throne, high back, ring-shaped emblem cut into the wood",
            dict(back_h=1.3, wood_i=1),
        ),
        (
            "small throne-like chair, circular emblem cut into the backrest",
            dict(back_h=0.9, wood_i=0),
        ),
        ("dark-wood fantasy throne, high back, round emblem cut-out", dict(wood_i=3)),
        ("royal throne, tall carved back with a circular emblem", dict(back_h=1.2, wood_i=1)),
    ],
)

instantiate(
    bb_workbench_drawer_slots,
    [
        ("a workbench with a row of identical drawer slots along the front", dict()),
        ("wide workshop bench, repeating row of drawer slots, pine", dict(wood_i=2)),
        ("small workbench, three drawer slots along the apron", dict(w=1.2, n=3, wood_i=0)),
        ("long carpenter's workbench, five drawer slots in a row", dict(w=2.0, n=5, wood_i=1)),
        ("garage workbench, steel-trimmed drawer slots along the front", dict(wood_i=2)),
        ("rustic workbench, row of drawer slots, heavy oak top", dict(wood_i=1)),
    ],
)

instantiate(
    bb_nightstand_finger_pull,
    [
        ("a nightstand with a round cut-out finger pull in the drawer face", dict()),
        ("bedside table, round finger-pull hole bored through the drawer", dict(wood_i=1)),
        ("small nightstand, drawer with a bored finger pull, walnut", dict(wood_i=1)),
        ("modern nightstand, minimalist round finger-pull cut into the drawer", dict(wood_i=2)),
        ("dark-wood nightstand, drawer front pierced for a finger pull", dict(wood_i=3)),
        ("compact nightstand, single drawer, round finger-pull cut-out", dict(w=0.38, wood_i=0)),
    ],
)

instantiate(
    bb_wine_barrel_table_hole,
    [
        ("a wine barrel side table with a boolean-cut hole for an umbrella pole", dict()),
        ("barrel-shaped side table, vertical hole bored through for a pole", dict(wood_i=1)),
        (
            "small wine barrel table, umbrella-pole hole through the centre",
            dict(radius=0.26, wood_i=2),
        ),
        ("rustic barrel table, hole cut clean through for a pole", dict(wood_i=1)),
        ("wide barrel table, umbrella hole bored through the middle", dict(radius=0.38, wood_i=3)),
        ("garden barrel table with a pole hole cut through it", dict(wood_i=2)),
    ],
)

# --- hard ----------------------------------------------------------------
# Seed line 66 ("a wooden chair leg that tapers...") is rephrased below --
# its own words share the bigram ("wooden", "chair") with the held-out
# chair subject once "with"/"a" drop out, so it would fail the holdout
# check verbatim; the intent (a single tapering leg, chair-scaled) is kept.

instantiate(
    bb_tapered_leg_standalone,
    [
        (
            "a single wood leg that tapers smoothly from a thick top to a narrow foot, sized for a dining chair",
            dict(),
        ),
        (
            "a standalone furniture leg, thick at the top and narrowing smoothly to a point at the foot",
            dict(),
        ),
        ("turned-looking taper leg, walnut, thick top narrow foot, chair height", dict(wood_i=1)),
        (
            "a tall tapering table leg, wide at the top, narrow at the foot",
            dict(height=0.72, top_r=0.04, wood_i=2),
        ),
        ("small tapering stool leg, smoothly narrowing foot", dict(height=0.4, wood_i=0)),
        (
            "a single leg for a side table, tapering evenly from top to foot",
            dict(height=0.5, top_r=0.032, wood_i=3),
        ),
    ],
)

instantiate(
    bb_chaise_longue,
    [
        ("a curved chaise longue whose frame sweeps from headrest to footrest", dict()),
        ("wooden chaise lounge, gently curved frame, raised headrest", dict(wood_i=1)),
        ("long lounge chair, swept curved rails, upholstered cushion", dict(fabric=FABRIC_GREEN)),
        ("compact chaise longue for a small room, curved wooden rails", dict(length=1.2, wood_i=2)),
        ("elegant chaise longue, dark wood frame, deep curve", dict(wood_i=3)),
        (
            "wide garden chaise longue, curved rail frame, cream cushion",
            dict(width=0.6, fabric=FABRIC_CREAM),
        ),
    ],
)

instantiate(
    bb_cabriole_table,
    [
        ("a table with legs that taper and curve outward like a cabriole leg", dict()),
        ("side table, four tapering splayed legs, elegant outward curve", dict(wood_i=1)),
        (
            "small accent table, legs tapering and flaring outward at the base",
            dict(top_w=0.8, top_d=0.55, wood_i=2),
        ),
        ("antique-style table, four cabriole-like tapered legs, dark wood", dict(wood_i=3)),
        ("wide console table, tapering outward-curving legs", dict(top_w=1.3, wood_i=0)),
        ("elegant occasional table, four splayed tapering legs", dict(wood_i=1)),
    ],
)

instantiate(
    bb_sweeping_bench_rail,
    [
        ("a bench whose single continuous rail sweeps up into a backrest", dict()),
        ("sled-style bench, one continuous curved rail rising into the back", dict(wood_i=1)),
        (
            "narrow hallway bench, swept rail frame rising into a backrest",
            dict(width=0.34, wood_i=2),
        ),
        ("modern bench, one bent rail sweeping from the floor up into the back", dict(wood_i=0)),
        ("wide entryway bench, continuous curved rail into a tall back", dict(width=0.5, wood_i=3)),
        ("small bench with one sweeping rail forming both legs and backrest", dict(wood_i=2)),
    ],
)

instantiate(
    bb_tusk_leg_stool,
    [
        ("a stool whose three legs taper and curve outward like tusks", dict()),
        ("round stool, three outward-curving tapered legs", dict(wood_i=1)),
        ("small stool, tusk-like splayed tapering legs, walnut", dict(seat_r=0.18, wood_i=1)),
        (
            "wide low stool, three tapering legs curving outward sharply",
            dict(seat_r=0.26, leg_h=0.4, wood_i=2),
        ),
        ("tall stool, three tapering outward-curved legs", dict(leg_h=0.6, wood_i=3)),
        ("rustic stool, three tusk-shaped tapering legs", dict(wood_i=0)),
    ],
)

# --- extra (broadens coverage beyond the seed list; lamps live only here) ----

instantiate(
    bb_round_table_three_lathe_legs,
    [
        ("a round kitchen table on three evenly spaced turned legs", dict()),
        ("small round cafe table, three lathe-turned legs, walnut", dict(top_r=0.45, wood_i=1)),
        ("wide round dining table, three turned legs arranged evenly", dict(top_r=0.65, wood_i=3)),
        ("modern round table, three turned legs, pale finish", dict(wood_i=0)),
        ("rustic round table, three thick turned legs", dict(top_r=0.5, wood_i=2)),
    ],
)

instantiate(
    bb_chest_of_drawers_plain,
    [
        ("a plain chest of drawers with three stacked fronts", dict()),
        ("simple three-drawer chest, walnut, no cut handles", dict(wood_i=1)),
        ("tall chest of drawers, four stacked drawer fronts", dict(n=4, h=1.1, wood_i=3)),
        ("small three-drawer chest for a child's room", dict(w=0.6, wood_i=2)),
        ("modern chest of drawers, three plain fronts, pale wood", dict(wood_i=0)),
    ],
)

instantiate(
    bb_corner_shelf,
    [
        ("a corner shelf unit, L-shaped, three levels", dict()),
        ("small corner shelving unit, three shelves, walnut", dict(wood_i=1)),
        ("tall corner shelf for a room corner, four levels", dict(n=4, h=1.5, wood_i=2)),
        ("compact corner shelf unit, two levels, pale wood", dict(n=2, wood_i=0)),
        ("rustic corner shelf, three levels, dark wood", dict(wood_i=3)),
    ],
)

instantiate(
    bb_wine_cabinet_lattice,
    [
        ("a tall wine cabinet with a lattice pattern across the door", dict()),
        ("wine cabinet, crossed lattice bars on the front, walnut", dict(wood_i=1)),
        ("small wine cabinet, lattice door, dense crossing bars", dict(n_h=5, n_v=5, wood_i=2)),
        ("rustic wine cabinet, wide lattice door", dict(w=0.7, wood_i=0)),
        ("modern wine cabinet, lattice front, pale wood", dict(wood_i=2)),
    ],
)

instantiate(
    bb_round_ottoman,
    [
        ("a round drum-shaped ottoman with a padded cushion top", dict()),
        ("small round ottoman, green fabric", dict(fabric=FABRIC_GREEN)),
        ("wide round pouf ottoman, cream upholstery", dict(radius=0.34, fabric=FABRIC_CREAM)),
        ("compact round footstool, blue fabric", dict(radius=0.22, fabric=FABRIC_BLUE)),
        ("tall round drum ottoman, red fabric", dict(h=0.48, fabric=FABRIC_RED)),
    ],
)

instantiate(
    bb_picnic_table,
    [
        ("a picnic table with attached benches on both sides", dict()),
        ("outdoor picnic table, two attached bench seats, pine", dict(wood_i=2)),
        ("wide picnic table with benches, walnut", dict(top_w=2.0, wood_i=1)),
        ("small picnic table for a patio, attached benches", dict(top_w=1.4, wood_i=0)),
        ("rustic park picnic table, heavy attached benches", dict(wood_i=1)),
    ],
)

instantiate(
    bb_daybed_frame,
    [
        ("a daybed frame with a raised headboard and a shorter footboard", dict()),
        ("simple daybed, tall headboard, low footboard, walnut", dict(wood_i=1)),
        ("wide daybed frame for a guest room", dict(w=1.1, wood_i=2)),
        ("small daybed, plain headboard and footboard, pale wood", dict(l=1.7, wood_i=0)),
        ("rustic daybed frame, heavy headboard, low foot panel", dict(wood_i=3)),
    ],
)

instantiate(
    bb_umbrella_stand,
    [
        ("a metal umbrella stand with drainage holes near the base", dict()),
        ("small umbrella stand, four drain holes bored near the bottom", dict(radius=0.15)),
        ("tall umbrella stand, drainage holes cut around the base", dict(height=0.6)),
        ("wide umbrella stand for an entryway, drain holes near the floor", dict(radius=0.22)),
        ("compact umbrella stand, small drainage holes", dict(radius=0.13, height=0.42)),
    ],
)

instantiate(
    bb_drafting_table,
    [
        ("a drafting table with a tilted top on two wide leg panels", dict()),
        ("tilted-top drafting table, walnut leg panels", dict(wood_i=1)),
        ("wide architect's drafting table, steep tilt", dict(tilt=20.0, wood_i=2)),
        ("small drafting table, gentle tilt, pale wood", dict(tilt=8.0, top_w=1.0, wood_i=0)),
        ("tall drafting table, tilted top, dark wood", dict(h=1.05, wood_i=3)),
    ],
)

instantiate(
    bb_sideboard,
    [
        ("a sideboard cabinet with two doors flanking a central drawer", dict()),
        ("long sideboard, two doors, one drawer, walnut", dict(wood_i=1)),
        ("wide dining-room sideboard, two doors, central drawer", dict(w=1.5, wood_i=3)),
        ("small sideboard cabinet, two doors, pale wood", dict(w=0.9, wood_i=0)),
        ("rustic sideboard, heavy doors, central drawer", dict(wood_i=2)),
    ],
)

instantiate(
    bb_magazine_rack,
    [
        ("a magazine rack made from a row of vertical slots cut into a box", dict()),
        ("wooden magazine rack, four vertical slots, walnut", dict(wood_i=1)),
        ("wide magazine rack, five slots cut through the front", dict(w=0.55, n=5, wood_i=2)),
        ("small magazine rack, three slots, pale wood", dict(n=3, wood_i=0)),
        ("dark-wood magazine rack, vertical slots cut through", dict(wood_i=3)),
    ],
)

instantiate(
    bb_bunk_bed,
    [
        ("a bunk bed frame with two stacked platforms and a ladder", dict()),
        ("bunk bed, walnut posts, ladder rungs up one corner", dict(wood_i=1)),
        ("tall bunk bed frame, five ladder rungs", dict(n_rungs=5, post_h=1.7, wood_i=2)),
        (
            "small child's bunk bed, short posts, ladder rungs",
            dict(l=1.6, w=0.8, post_h=1.3, wood_i=0),
        ),
        ("rustic bunk bed frame, heavy posts, ladder rungs", dict(wood_i=3)),
    ],
)

instantiate(
    bb_floor_lamp,
    [
        ("a tall floor lamp with a turned wooden base and a drum shade", dict()),
        ("standing lamp, lathe-turned base, cream drum shade, walnut", dict(wood_i=1)),
        ("tall reading lamp, turned base, narrow pole, wide shade", dict(pole_h=1.5, wood_i=2)),
        ("short floor lamp, turned base, small drum shade", dict(pole_h=0.9, wood_i=0)),
        ("dark-wood floor lamp, turned base, wide drum shade", dict(shade_r=0.24, wood_i=3)),
    ],
)

instantiate(
    bb_table_lamp,
    [
        ("a table lamp with a bulbous turned base and a small shade", dict()),
        ("small table lamp, turned ceramic-toned base, cream shade", dict()),
        ("wide table lamp, bulbous turned base", dict(base_r=0.13, shade_r=0.18)),
        ("compact bedside lamp, small turned base and shade", dict(base_h=0.16, shade_r=0.12)),
        ("tall table lamp, turned base, narrow shade", dict(neck_h=0.24, shade_h=0.24)),
    ],
)

instantiate(
    bb_desk_lamp,
    [
        ("a simple desk lamp with a cylinder base and a cone shade", dict()),
        ("small articulating desk lamp, metal, angled shade", dict()),
        ("wide-base desk lamp, tilted arm, cone shade", dict(base_r=0.11)),
        ("compact desk lamp, short arm, small cone shade", dict(arm_len=0.22)),
        ("tall desk lamp, long arm, cone shade angled down", dict(arm_len=0.42)),
    ],
)


def main() -> None:
    OUT_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in RECORDS) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(RECORDS)} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
