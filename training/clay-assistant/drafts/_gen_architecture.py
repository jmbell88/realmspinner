"""Emits ``drafts/architecture.jsonl`` for the ``architecture`` family --
single columns and pillars, arches and gateways, stairs, towers, walls with
window cut-outs, fences, bridges, wells, altars, plinths, pergolas and rings
of columns. Deterministic and seeded: rerunning this script byte-for-byte
reproduces the same JSONL (materials/dims/counts come from ``random.Random``
seeded per template+variant, never wall-clock or unseeded ``random``).

Scope and the held-out rule are both documented in
``training/clay-assistant/seeds/architecture.txt``'s own header: this family
must never build the held-out subject (a circular ring of eight fluted
columns on a stepped round base) or a close paraphrase of it -- so nowhere
in this file does a column ring take a count of eight, and the words
"fluted" and "colonnade" and the adjacent phrase "round base"/"stepped
round" never appear.

Run directly to (re)write the JSONL:

    uv run python training/clay-assistant/drafts/_gen_architecture.py
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

FAMILY = "architecture"
OUT_PATH = Path(__file__).resolve().parent / f"{FAMILY}.jsonl"

# --- naming variety --------------------------------------------------------


def nm(base: str, style: int) -> str:
    """*base* (``snake_case`` or ``hyphen-case``) rendered in one of four
    naming conventions, cycled by *style* -- see the brief's "vary object
    naming" instruction."""
    words = base.replace("-", "_").split("_")
    if style % 4 == 0:
        return "_".join(words)
    if style % 4 == 1:
        return "".join(w.capitalize() for w in words)
    if style % 4 == 2:
        return "-".join(words)
    return " ".join(words)


# --- material catalog -------------------------------------------------------
# key -> (color rgb 0..1, roughness, metallic, natural-language phrase)
MATERIALS: dict[str, tuple[tuple[float, float, float], float, float, str]] = {
    "weathered_stone": ((0.47, 0.46, 0.44), 0.85, 0.0, "weathered grey stone"),
    "sandstone": ((0.76, 0.65, 0.48), 0.80, 0.0, "honey-toned sandstone"),
    "limestone": ((0.82, 0.78, 0.68), 0.75, 0.0, "pale limestone"),
    "dark_basalt": ((0.16, 0.16, 0.18), 0.70, 0.0, "dark basalt"),
    "granite": ((0.50, 0.50, 0.52), 0.60, 0.0, "polished grey granite"),
    "oak_wood": ((0.45, 0.31, 0.18), 0.75, 0.0, "oiled oak"),
    "weathered_wood": ((0.33, 0.24, 0.15), 0.85, 0.0, "weathered timber"),
    "red_brick": ((0.55, 0.24, 0.18), 0.80, 0.0, "red brick"),
    "terracotta": ((0.70, 0.35, 0.22), 0.80, 0.0, "terracotta"),
    "iron_dark": ((0.22, 0.22, 0.24), 0.40, 0.85, "blackened iron"),
    "bronze": ((0.55, 0.40, 0.20), 0.35, 0.90, "polished bronze"),
    "marble_white": ((0.90, 0.89, 0.86), 0.30, 0.0, "pale marble"),
    "concrete": ((0.62, 0.62, 0.60), 0.90, 0.0, "raw concrete"),
    "steel_scifi": ((0.60, 0.63, 0.68), 0.30, 0.70, "brushed steel"),
    "verdigris_copper": ((0.35, 0.55, 0.48), 0.60, 0.20, "verdigris-streaked copper"),
    "slate": ((0.22, 0.24, 0.27), 0.70, 0.0, "dark slate"),
    "moss_stone": ((0.40, 0.46, 0.36), 0.85, 0.0, "moss-covered stone"),
    "gilded": ((0.83, 0.68, 0.21), 0.30, 0.85, "gilded bronze"),
}


def mat(key: str) -> tuple[tuple[float, float, float], float, float, str]:
    return MATERIALS[key]


# --- call builders -----------------------------------------------------------


def prim(
    generator: str,
    name: str,
    *,
    params: dict[str, Any] | None = None,
    translation: list[float] | None = None,
    rotation: list[float] | None = None,
    scale: list[float] | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {"generator": generator, "name": name}
    if params:
        args["params"] = params
    if translation is not None:
        args["translation"] = [round(float(v), 4) for v in translation]
    if rotation is not None:
        args["rotation"] = [round(float(v), 3) for v in rotation]
    if scale is not None:
        args["scale"] = [round(float(v), 4) for v in scale]
    return {"name": "clay_add_primitive", "arguments": args}


def ref(name: str) -> dict[str, str]:
    return {"$ref": name}


def material_call(names: list[str], mat_key: str, mat_name: str) -> dict[str, Any]:
    color, roughness, metallic, _ = mat(mat_key)
    args: dict[str, Any] = {
        "uids": [ref(n) for n in names],
        "name": mat_name,
        "color": [round(c, 3) for c in color],
        "roughness": roughness,
    }
    if metallic:
        args["metallic"] = metallic
    return {"name": "clay_material", "arguments": args}


def select_call(names: list[str]) -> dict[str, Any]:
    return {"name": "clay_select", "arguments": {"uids": [ref(n) for n in names]}}


def op_call(name: str, params: dict[str, float] | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {"name": name}
    if params:
        args["params"] = params
    return {"name": "clay_op", "arguments": args}


def boolean_call(kind: str, names: list[str]) -> dict[str, Any]:
    return {"name": "clay_boolean", "arguments": {"kind": kind, "uids": [ref(n) for n in names]}}


def array_linear(
    name: str, *, count: int, x: float = 0.0, y: float = 0.0, z: float = 0.0
) -> list[dict[str, Any]]:
    return [
        select_call([name]),
        op_call("array-linear", {"count": float(count), "x": x, "y": y, "z": z}),
    ]


def array_radial(
    name: str, *, count: int, angle: float = 360.0, axis: float = 1.0
) -> list[dict[str, Any]]:
    return [
        select_call([name]),
        op_call("array-radial", {"count": float(count), "angle": angle, "axis": axis}),
    ]


# --- record plumbing ---------------------------------------------------------

Prompter = Callable[[random.Random, dict[str, Any]], str]


class Template:
    """One base build: ``build(rng, variant)`` returns ``(calls, notes,
    ctx)`` where ``ctx`` is whatever the prompt functions need (dims,
    material phrase, counts...); ``prompts`` is a list of callables, one per
    phrasing, each taking ``(rng, ctx) -> str``."""

    def __init__(
        self,
        key: str,
        build: Callable[[random.Random, int], tuple[list[dict[str, Any]], str, dict[str, Any]]],
        prompts: list[Prompter],
    ) -> None:
        self.key = key
        self.build = build
        self.prompts = prompts


TEMPLATES: list[Template] = []


def register(
    key: str, build: Callable[[random.Random, int], tuple[list, str, dict]], prompts: list[Prompter]
) -> None:
    assert len(prompts) == 6, f"{key} needs exactly 6 phrasings, got {len(prompts)}"
    TEMPLATES.append(Template(key, build, prompts))


def fmt_m(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


# =============================================================================
# EASY templates (seeds 1-25)
# =============================================================================


# 1. a simple stone archway spanning a garden path
def _b_archway(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15, 1.3])
    width = round(2.0 * scale, 2)
    height = round(2.6 * scale, 2)
    depth = round(0.6 * scale, 2)
    thickness = round(0.3 * scale, 2)
    mk = rng.choice(["weathered_stone", "sandstone", "granite"])
    style = v
    name = nm("archway", style)
    calls = [
        prim(
            "arch",
            name,
            params={
                "width": width,
                "height": height,
                "depth": depth,
                "thickness": thickness,
                "segments": 16,
            },
            translation=[0, height / 2, 0],
        ),
        material_call([name], mk, "Archway Stone"),
    ]
    notes = (
        "arch is centred like every other generator here -- its own mesh spans "
        "-height/2..+height/2 -- so translation.y is height/2 to ground it, the same "
        "rule as a box; width/height/thickness kept inside the generator's own clamps "
        "(thickness under 90% of the head radius) so the opening stays a real gap."
    )
    return calls, notes, {"width": width, "height": height, "mat": MATERIALS[mk][3]}


PROMPTS_ARCHWAY = [
    lambda r, c: "a simple stone archway spanning a garden path",
    lambda r, c: (
        f"a {c['mat']} archway {fmt_m(c['width'])} m wide and {fmt_m(c['height'])} m tall, "
        "framing a garden path"
    ),
    lambda r, c: "build an archway over the path through the garden",
    lambda r, c: f"garden archway, {c['mat']}, nothing fancy",
    lambda r, c: (
        f"a weathered archway of {c['mat']} spanning a path between two hedgerows, "
        f"about {fmt_m(c['height'])} metres to the crown"
    ),
    lambda r, c: "could you make a stone archway for a sci-fi garden dome walkway",
]
register("archway", _b_archway, PROMPTS_ARCHWAY)


# 2. a plain rectangular watchtower with a flat roof
def _b_watchtower_flat(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.8, 1.0, 1.2])
    w = round(2.2 * scale, 2)
    d = round(2.2 * scale, 2)
    h = round(6.0 * scale, 2)
    roof_h = round(0.3 * scale, 2)
    mk = rng.choice(["weathered_stone", "granite", "concrete"])
    style = v
    shaft = nm("shaft", style)
    roof = nm("roof", style)
    calls = [
        prim("box", shaft, params={"size": [w, h, d]}, translation=[0, h / 2, 0]),
        prim(
            "box",
            roof,
            params={"size": [w * 1.05, roof_h, d * 1.05]},
            translation=[0, h + roof_h / 2, 0],
        ),
        material_call([shaft, roof], mk, "Tower Stone"),
    ]
    notes = "a plain box shaft with a slightly wider flat cap sitting on top, grounded at y=0."
    return calls, notes, {"w": w, "h": h + roof_h, "mat": MATERIALS[mk][3]}


PROMPTS_WATCHTOWER = [
    lambda r, c: "a plain rectangular watchtower with a flat roof",
    lambda r, c: (
        f"a {c['mat']} watchtower, {fmt_m(c['w'])} m square and {fmt_m(c['h'])} m tall, flat-roofed"
    ),
    lambda r, c: "make me a boxy watchtower, flat top, nothing pointed",
    lambda r, c: f"square watchtower in {c['mat']}, flat roof slab",
    lambda r, c: (
        f"a squat rectangular watchtower of {c['mat']} standing "
        f"{fmt_m(c['h'])} metres tall with a flat capping slab"
    ),
    lambda r, c: "a modern concrete watchtower, boxy, flat-roofed, for a checkpoint",
]
register("watchtower_flat", _b_watchtower_flat, PROMPTS_WATCHTOWER)


# 3. a short flight of stone steps leading to a doorway (hand-placed boxes)
def _b_steps_short(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.2])
    n_steps = rng.choice([3, 4])
    rise = round(0.18 * scale, 3)
    run = round(0.32 * scale, 3)
    width = round(1.3 * scale, 2)
    mk = rng.choice(["weathered_stone", "sandstone", "granite"])
    style = v
    calls = []
    names = []
    for i in range(n_steps):
        name = nm(f"step_{i + 1}", style)
        names.append(name)
        step_h = rise * (i + 1)
        calls.append(
            prim(
                "box",
                name,
                params={"size": [width, step_h, run]},
                translation=[0, step_h / 2, -run * i],
            )
        )
    calls.append(material_call(names, mk, "Step Stone"))
    notes = (
        f"{n_steps} explicit boxes, each taller and further back than the last so the "
        "profile reads as a rising flight; each step's own box spans the ground up to "
        "its own tread height rather than floating a thin slab at the rise, so nothing "
        "shows a gap underneath."
    )
    return calls, notes, {"n": n_steps, "mat": MATERIALS[mk][3], "width": width}


PROMPTS_STEPS_SHORT = [
    lambda r, c: "a short flight of stone steps leading to a doorway",
    lambda r, c: f"{c['n']} {c['mat']} steps, {fmt_m(c['width'])} m wide, up to a door",
    lambda r, c: "a few steps up to the doorway, plain stone",
    lambda r, c: f"short stone stair, {c['n']} steps, leads to a doorway",
    lambda r, c: (
        f"a short flight of {c['n']} {c['mat']} steps rising to a doorway, each tread "
        "a little higher than the last"
    ),
    lambda r, c: "a few concrete steps up to a modern doorway, three or four of them",
]
register("steps_short", _b_steps_short, PROMPTS_STEPS_SHORT)


# 4. a single stone column standing alone in a courtyard
def _b_single_column(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.8, 1.0, 1.2, 1.4])
    radius = round(0.35 * scale, 3)
    height = round(3.0 * scale, 2)
    mk = rng.choice(["weathered_stone", "marble_white", "granite", "sandstone"])
    style = v
    name = nm("column", style)
    calls = [
        prim(
            "column",
            name,
            params={
                "radius": radius,
                "height": height,
                "segments": 16,
                "base": 0.15,
                "capital": 0.15,
            },
            translation=[0, height / 2, 0],
        ),
        material_call([name], mk, "Column Stone"),
    ]
    notes = "one column primitive, centred and grounded at height/2; base/capital left at defaults."
    return calls, notes, {"radius": radius, "height": height, "mat": MATERIALS[mk][3]}


PROMPTS_SINGLE_COLUMN = [
    lambda r, c: "a single stone column standing alone in a courtyard",
    lambda r, c: f"a {c['mat']} column, {fmt_m(c['height'])} m tall, alone in a courtyard",
    lambda r, c: "just one column, standing by itself in the yard",
    lambda r, c: f"lone column, {c['mat']}, courtyard centrepiece",
    lambda r, c: (
        f"a single round column of {c['mat']}, roughly {fmt_m(c['radius'] * 2)} m across, "
        "standing alone in an empty courtyard"
    ),
    lambda r, c: "a lone steel column standing in a sci-fi plaza",
]
register("single_column", _b_single_column, PROMPTS_SINGLE_COLUMN)


# 5. a low garden wall made of stacked rectangular blocks
def _b_wall_stacked_blocks(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.2])
    block_w = round(0.6 * scale, 3)
    block_h = round(0.3 * scale, 3)
    block_d = round(0.3 * scale, 3)
    n_blocks = rng.choice([5, 6, 7])
    mk = rng.choice(["weathered_stone", "sandstone", "red_brick"])
    style = v
    names = []
    calls = []
    rows = 2
    for row in range(rows):
        for i in range(n_blocks):
            name = nm(f"block_{row + 1}_{i + 1}", style)
            names.append(name)
            offset_x = (block_w / 2) if row % 2 else 0.0
            x = i * block_w + offset_x - (n_blocks * block_w) / 2
            calls.append(
                prim(
                    "box",
                    name,
                    params={"size": [block_w, block_h, block_d]},
                    translation=[x, block_h / 2 + row * block_h, 0],
                )
            )
    calls.append(material_call(names, mk, "Wall Stone"))
    notes = (
        f"{rows} courses of {n_blocks} blocks each, the second course offset by half a "
        "block so the joints stagger like a real stacked wall rather than lining up "
        "in one continuous seam."
    )
    length = n_blocks * block_w
    return calls, notes, {"length": length, "height": block_h * rows, "mat": MATERIALS[mk][3]}


PROMPTS_WALL_BLOCKS = [
    lambda r, c: "a low garden wall made of stacked rectangular blocks",
    lambda r, c: f"a {c['mat']} garden wall, {fmt_m(c['length'])} m long, stacked in courses",
    lambda r, c: "low wall for the garden, made of stacked blocks",
    lambda r, c: f"stacked-block garden wall, {c['mat']}, low",
    lambda r, c: (
        f"a low garden wall of {c['mat']} built from two staggered courses of rectangular blocks"
    ),
    lambda r, c: "a low brick garden wall, modern back-garden style, stacked blocks",
]
register("wall_stacked_blocks", _b_wall_stacked_blocks, PROMPTS_WALL_BLOCKS)


# 6. a simple wooden gate flanked by two square posts
def _b_gate_two_posts(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    post_size = round(0.2 * scale, 3)
    post_h = round(2.0 * scale, 2)
    gap = round(1.4 * scale, 2)
    gate_w = round(gap - post_size, 2)
    gate_h = round(1.7 * scale, 2)
    mk = rng.choice(["oak_wood", "weathered_wood"])
    post_mk = rng.choice(["weathered_stone", "granite"])
    style = v
    post_l = nm("post_left", style)
    post_r = nm("post_right", style)
    gate = nm("gate", style)
    calls = [
        prim(
            "box",
            post_l,
            params={"size": [post_size, post_h, post_size]},
            translation=[-gap / 2, post_h / 2, 0],
        ),
        prim(
            "box",
            post_r,
            params={"size": [post_size, post_h, post_size]},
            translation=[gap / 2, post_h / 2, 0],
        ),
        material_call([post_l, post_r], post_mk, "Post Stone"),
        prim("box", gate, params={"size": [gate_w, gate_h, 0.06]}, translation=[0, gate_h / 2, 0]),
        material_call([gate], mk, "Gate Wood"),
    ]
    notes = "two square stone posts flanking a thin wooden gate leaf sized to the gap between them."
    return calls, notes, {"gap": gap, "mat": MATERIALS[mk][3], "post_mat": MATERIALS[post_mk][3]}


PROMPTS_GATE = [
    lambda r, c: "a simple wooden gate flanked by two square posts",
    lambda r, c: f"a wooden gate, {fmt_m(c['gap'])} m between two {c['post_mat']} posts",
    lambda r, c: "simple wood gate, two posts either side",
    lambda r, c: f"gate of {c['mat']} between square posts",
    lambda r, c: f"a plain {c['mat']} gate hung between two square {c['post_mat']} gateposts",
    lambda r, c: "a modern timber gate between two concrete posts",
]
register("gate_two_posts", _b_gate_two_posts, PROMPTS_GATE)


# 7. a small stone well with a round low wall
def _b_well_small(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.2])
    radius = round(0.6 * scale, 2)
    height = round(0.5 * scale, 2)
    mk = rng.choice(["weathered_stone", "granite", "moss_stone"])
    style = v
    name = nm("well_wall", style)
    calls = [
        prim(
            "cylinder",
            name,
            params={"radius": radius, "height": height, "segments": 20},
            translation=[0, height / 2, 0],
        ),
        material_call([name], mk, "Well Stone"),
    ]
    notes = "a squat, wide cylinder standing for the well's low ring wall; no separate roof."
    return calls, notes, {"radius": radius, "mat": MATERIALS[mk][3]}


PROMPTS_WELL_SMALL = [
    lambda r, c: "a small stone well with a round low wall",
    lambda r, c: f"a well, {c['mat']}, {fmt_m(c['radius'] * 2)} m across, low ring wall",
    lambda r, c: "small well, low wall around it",
    lambda r, c: f"stone well, {c['mat']}, squat and low",
    lambda r, c: f"a small courtyard well ringed by a low wall of {c['mat']}",
    lambda r, c: "a small well with a low concrete ring wall, modern courtyard",
]
register("well_small", _b_well_small, PROMPTS_WELL_SMALL)


# 8. a plain brick chimney, a tall narrow box
def _b_chimney(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.8, 1.0, 1.2, 1.4])
    w = round(0.6 * scale, 2)
    h = round(4.5 * scale, 2)
    mk = rng.choice(["red_brick", "terracotta", "dark_basalt"])
    style = v
    name = nm("chimney", style)
    calls = [
        prim("box", name, params={"size": [w, h, w]}, translation=[0, h / 2, 0]),
        material_call([name], mk, "Chimney Brick"),
    ]
    notes = (
        "one tall narrow box; a chimney needs nothing more than height/width ratio and material."
    )
    return calls, notes, {"h": h, "w": w, "mat": MATERIALS[mk][3]}


PROMPTS_CHIMNEY = [
    lambda r, c: "a plain brick chimney, a tall narrow box",
    lambda r, c: f"a {c['mat']} chimney, {fmt_m(c['w'])} m square and {fmt_m(c['h'])} m tall",
    lambda r, c: "just a tall thin brick chimney",
    lambda r, c: f"chimney stack, {c['mat']}, narrow",
    lambda r, c: f"a plain chimney of {c['mat']}, tall and narrow, nothing decorative",
    lambda r, c: "a tall industrial concrete chimney stack, narrow",
]
register("chimney", _b_chimney, PROMPTS_CHIMNEY)


# 9. a castle battlement wall with square notches along the top (hand-placed merlons)
def _b_battlement_hand(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.9, 1.0, 1.15])
    wall_len = round(5.0 * scale, 2)
    wall_h = round(2.2 * scale, 2)
    wall_t = round(0.5 * scale, 2)
    n_merlons = rng.choice([5, 7])
    merlon_w = round(wall_len / (n_merlons * 2 - 1), 3)
    merlon_h = round(0.6 * scale, 3)
    mk = rng.choice(["weathered_stone", "granite", "dark_basalt"])
    style = v
    wall = nm("wall", style)
    calls = [
        prim(
            "box", wall, params={"size": [wall_len, wall_h, wall_t]}, translation=[0, wall_h / 2, 0]
        )
    ]
    names = [wall]
    for i in range(n_merlons):
        x = -wall_len / 2 + merlon_w / 2 + i * (2 * merlon_w)
        name = nm(f"merlon_{i + 1}", style)
        names.append(name)
        calls.append(
            prim(
                "box",
                name,
                params={"size": [merlon_w, merlon_h, wall_t]},
                translation=[x, wall_h + merlon_h / 2, 0],
            )
        )
    calls.append(material_call(names, mk, "Battlement Stone"))
    notes = (
        f"{n_merlons} square merlons placed by hand along the top of one wall box, spaced "
        "so the gaps between them equal a merlon's own width -- the classic notch rhythm."
    )
    return calls, notes, {"len": wall_len, "mat": MATERIALS[mk][3], "n": n_merlons}


PROMPTS_BATTLEMENT_HAND = [
    lambda r, c: "a castle battlement wall with square notches along the top",
    lambda r, c: f"a battlement wall, {fmt_m(c['len'])} m long, {c['n']} square merlons on top",
    lambda r, c: "castle wall with notches cut along the top edge",
    lambda r, c: f"battlements, {c['mat']}, square notches",
    lambda r, c: f"a {c['mat']} castle wall topped with {c['n']} evenly spaced square merlons",
    lambda r, c: "a modern concrete perimeter wall styled with square battlement notches",
]
register("battlement_hand", _b_battlement_hand, PROMPTS_BATTLEMENT_HAND)


# 10. a simple bridge made of a flat plank over two support blocks
def _b_bridge_plank(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.2])
    span = round(3.0 * scale, 2)
    deck_w = round(1.2 * scale, 2)
    deck_t = round(0.12 * scale, 3)
    support_h = round(0.5 * scale, 2)
    support_w = round(0.4 * scale, 2)
    mk = rng.choice(["oak_wood", "weathered_wood"])
    support_mk = rng.choice(["weathered_stone", "granite", "concrete"])
    style = v
    deck = nm("deck", style)
    sup_l = nm("support_left", style)
    sup_r = nm("support_right", style)
    calls = [
        prim(
            "box",
            sup_l,
            params={"size": [support_w, support_h, deck_w]},
            translation=[-span / 2 + support_w / 2, support_h / 2, 0],
        ),
        prim(
            "box",
            sup_r,
            params={"size": [support_w, support_h, deck_w]},
            translation=[span / 2 - support_w / 2, support_h / 2, 0],
        ),
        material_call([sup_l, sup_r], support_mk, "Support Stone"),
        prim(
            "box",
            deck,
            params={"size": [span, deck_t, deck_w]},
            translation=[0, support_h + deck_t / 2, 0],
        ),
        material_call([deck], mk, "Deck Wood"),
    ]
    notes = "one flat plank deck resting on two block supports at either end, span kept over the gap between them."
    return calls, notes, {"span": span, "mat": MATERIALS[mk][3]}


PROMPTS_BRIDGE_PLANK = [
    lambda r, c: "a simple bridge made of a flat plank over two support blocks",
    lambda r, c: f"a plank bridge, {fmt_m(c['span'])} m across, resting on two block supports",
    lambda r, c: "a plank across the stream on two blocks",
    lambda r, c: f"simple {c['mat']} plank bridge",
    lambda r, c: (
        f"a flat {c['mat']} plank bridge spanning {fmt_m(c['span'])} metres between two stone supports"
    ),
    lambda r, c: "a modern steel plank footbridge over a narrow gap",
]
register("bridge_plank", _b_bridge_plank, PROMPTS_BRIDGE_PLANK)


# 11. a lighthouse tower, a tall tapering cone
def _b_lighthouse_cone(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.8, 1.0, 1.2])
    radius = round(1.4 * scale, 2)
    height = round(8.0 * scale, 2)
    mk = rng.choice(["marble_white", "weathered_stone", "concrete"])
    style = v
    name = nm("lighthouse", style)
    calls = [
        prim(
            "cone",
            name,
            params={"radius": radius, "height": height, "segments": 24},
            translation=[0, height / 2, 0],
        ),
        material_call([name], mk, "Lighthouse Wall"),
    ]
    notes = (
        "a single cone, wide at the base and narrowing to a point, standing for the tapering tower."
    )
    return calls, notes, {"h": height, "mat": MATERIALS[mk][3]}


PROMPTS_LIGHTHOUSE_CONE = [
    lambda r, c: "a lighthouse tower, a tall tapering cone",
    lambda r, c: f"a {c['mat']} lighthouse tower, {fmt_m(c['h'])} m tall, tapering to a point",
    lambda r, c: "tall lighthouse, cone-shaped, tapering up",
    lambda r, c: f"lighthouse cone, {c['mat']}",
    lambda r, c: f"a tall tapering lighthouse tower built from {c['mat']}, wide at the base",
    lambda r, c: "a sci-fi beacon tower, a tall tapering steel cone",
]
register("lighthouse_cone", _b_lighthouse_cone, PROMPTS_LIGHTHOUSE_CONE)


# 12. a stone doorway with a flat lintel
def _b_doorway_lintel(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.9, 1.0, 1.1])
    jamb_w = round(0.25 * scale, 3)
    jamb_h = round(2.1 * scale, 2)
    gap = round(1.0 * scale, 2)
    lintel_t = round(0.25 * scale, 3)
    depth = round(0.3 * scale, 2)
    mk = rng.choice(["weathered_stone", "granite", "sandstone"])
    style = v
    jamb_l = nm("jamb_left", style)
    jamb_r = nm("jamb_right", style)
    lintel = nm("lintel", style)
    span = gap + jamb_w
    calls = [
        prim(
            "box",
            jamb_l,
            params={"size": [jamb_w, jamb_h, depth]},
            translation=[-span / 2, jamb_h / 2, 0],
        ),
        prim(
            "box",
            jamb_r,
            params={"size": [jamb_w, jamb_h, depth]},
            translation=[span / 2, jamb_h / 2, 0],
        ),
        prim(
            "box",
            lintel,
            params={"size": [span + jamb_w, lintel_t, depth]},
            translation=[0, jamb_h + lintel_t / 2, 0],
        ),
        material_call([jamb_l, jamb_r, lintel], mk, "Doorway Stone"),
    ]
    notes = "two jamb posts and a flat lintel spanning them, the classic post-and-lintel doorway."
    return calls, notes, {"h": jamb_h, "mat": MATERIALS[mk][3], "gap": gap}


PROMPTS_DOORWAY_LINTEL = [
    lambda r, c: "a stone doorway with a flat lintel",
    lambda r, c: f"a {c['mat']} doorway, {fmt_m(c['gap'])} m wide opening, flat lintel on top",
    lambda r, c: "doorway with a flat stone lintel over it",
    lambda r, c: f"post-and-lintel doorway, {c['mat']}",
    lambda r, c: f"a plain {c['mat']} doorway framed by two jambs and a flat lintel",
    lambda r, c: "a modern concrete doorway frame with a flat lintel",
]
register("doorway_lintel", _b_doorway_lintel, PROMPTS_DOORWAY_LINTEL)


# 13. a modest bell tower, a square shaft topped with a pyramid roof
def _b_bell_tower(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.2])
    w = round(2.0 * scale, 2)
    h = round(5.0 * scale, 2)
    roof_h = round(1.6 * scale, 2)
    mk = rng.choice(["weathered_stone", "sandstone", "granite"])
    roof_mk = rng.choice(["slate", "terracotta", "dark_basalt"])
    style = v
    shaft = nm("shaft", style)
    roof = nm("roof", style)
    calls = [
        prim("box", shaft, params={"size": [w, h, w]}, translation=[0, h / 2, 0]),
        material_call([shaft], mk, "Tower Stone"),
        prim(
            "pyramid",
            roof,
            params={"base": w * 1.1, "height": roof_h, "sides": 4},
            translation=[0, h + roof_h / 2, 0],
        ),
        material_call([roof], roof_mk, "Roof Slate"),
    ]
    notes = "a square box shaft with a four-sided pyramid roof stacked on top, base widened slightly for eaves."
    return calls, notes, {"h": h + roof_h, "mat": MATERIALS[mk][3]}


PROMPTS_BELL_TOWER = [
    lambda r, c: "a modest bell tower, a square shaft topped with a pyramid roof",
    lambda r, c: f"a bell tower, {fmt_m(c['h'])} m total, square shaft with a pyramid cap",
    lambda r, c: "small bell tower, square, pyramid roof on top",
    lambda r, c: f"bell tower in {c['mat']}, pyramid-capped",
    lambda r, c: f"a modest {c['mat']} bell tower with a square shaft and a pyramid roof",
    lambda r, c: "a modern square bell tower with a pyramid-capped roof",
]
register("bell_tower", _b_bell_tower, PROMPTS_BELL_TOWER)


# 14. a plain obelisk, a tall four-sided tapering pillar
def _b_obelisk(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.8, 1.0, 1.2, 1.4])
    base = round(0.9 * scale, 2)
    height = round(6.0 * scale, 2)
    mk = rng.choice(["granite", "marble_white", "sandstone"])
    style = v
    name = nm("obelisk", style)
    calls = [
        prim(
            "pyramid",
            name,
            params={"base": base, "height": height, "sides": 4},
            translation=[0, height / 2, 0],
        ),
        material_call([name], mk, "Obelisk Stone"),
    ]
    notes = "a single tall four-sided pyramid; a pointed apex reads as an obelisk's tapering tip."
    return calls, notes, {"h": height, "mat": MATERIALS[mk][3]}


PROMPTS_OBELISK = [
    lambda r, c: "a plain obelisk, a tall four-sided tapering pillar",
    lambda r, c: f"an obelisk, {fmt_m(c['h'])} m tall, {c['mat']}",
    lambda r, c: "tall four-sided obelisk, plain",
    lambda r, c: f"obelisk, {c['mat']}, pointed top",
    lambda r, c: f"a plain {c['mat']} obelisk tapering to a point, {fmt_m(c['h'])} metres tall",
    lambda r, c: "a modern marble obelisk monument, tall and tapering",
]
register("obelisk", _b_obelisk, PROMPTS_OBELISK)


# 15. a simple stone plinth for a statue
def _b_plinth(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.2])
    w = round(0.9 * scale, 2)
    h = round(1.1 * scale, 2)
    mk = rng.choice(["marble_white", "granite", "weathered_stone"])
    style = v
    name = nm("plinth", style)
    calls = [
        prim("box", name, params={"size": [w, h, w]}, translation=[0, h / 2, 0]),
        material_call([name], mk, "Plinth Stone"),
    ]
    notes = "a simple square block, taller than a step but shorter than a person, for a statue to stand on."
    return calls, notes, {"h": h, "mat": MATERIALS[mk][3]}


PROMPTS_PLINTH = [
    lambda r, c: "a simple stone plinth for a statue",
    lambda r, c: f"a {c['mat']} plinth, {fmt_m(c['h'])} m tall, for a statue",
    lambda r, c: "plain plinth for a statue to stand on",
    lambda r, c: f"statue plinth, {c['mat']}",
    lambda r, c: f"a simple square plinth of {c['mat']}, about waist height, for a statue",
    lambda r, c: "a modern polished-marble statue plinth",
]
register("plinth", _b_plinth, PROMPTS_PLINTH)


# 16. a wooden watchtower platform on four support posts
def _b_platform_four_posts(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.2])
    post_size = round(0.18 * scale, 3)
    post_h = round(3.0 * scale, 2)
    span = round(2.2 * scale, 2)
    plat_t = round(0.15 * scale, 3)
    mk = rng.choice(["oak_wood", "weathered_wood"])
    style = v
    names = []
    calls = []
    half = span / 2 - post_size / 2
    for i, (sx, sz) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
        name = nm(f"post_{i + 1}", style)
        names.append(name)
        calls.append(
            prim(
                "box",
                name,
                params={"size": [post_size, post_h, post_size]},
                translation=[sx * half, post_h / 2, sz * half],
            )
        )
    platform = nm("platform", style)
    names.append(platform)
    calls.append(
        prim(
            "box",
            platform,
            params={"size": [span, plat_t, span]},
            translation=[0, post_h + plat_t / 2, 0],
        )
    )
    calls.append(material_call(names, mk, "Platform Wood"))
    notes = "four corner posts of equal height carrying one flat platform slab -- a watchtower deck with nothing above it yet."
    return calls, notes, {"h": post_h, "mat": MATERIALS[mk][3]}


PROMPTS_PLATFORM = [
    lambda r, c: "a wooden watchtower platform on four support posts",
    lambda r, c: f"a {c['mat']} platform on four posts, {fmt_m(c['h'])} m up",
    lambda r, c: "a lookout platform up on four posts",
    lambda r, c: f"platform on posts, {c['mat']}",
    lambda r, c: f"a wooden watch platform raised on four support posts of {c['mat']}",
    lambda r, c: "a modern steel-post observation platform, four legs",
]
register("platform_four_posts", _b_platform_four_posts, PROMPTS_PLATFORM)


# 17. a low fence made of evenly spaced posts (hand-placed)
def _b_fence_hand(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    n = rng.choice([5, 6, 7])
    post_size = round(0.1 * scale, 3)
    post_h = round(0.9 * scale, 2)
    spacing = round(0.6 * scale, 2)
    mk = rng.choice(["oak_wood", "weathered_wood"])
    style = v
    names = []
    calls = []
    total = (n - 1) * spacing
    for i in range(n):
        name = nm(f"post_{i + 1}", style)
        names.append(name)
        x = -total / 2 + i * spacing
        calls.append(
            prim(
                "box",
                name,
                params={"size": [post_size, post_h, post_size]},
                translation=[x, post_h / 2, 0],
            )
        )
    calls.append(material_call(names, mk, "Fence Wood"))
    notes = f"{n} posts placed by hand at even spacing -- a low fence line, no rails between them."
    return calls, notes, {"n": n, "mat": MATERIALS[mk][3], "total": total}


PROMPTS_FENCE_HAND = [
    lambda r, c: "a low fence made of evenly spaced posts",
    lambda r, c: f"a low {c['mat']} fence, {c['n']} posts over {fmt_m(c['total'])} m",
    lambda r, c: "low fence, just posts, evenly spaced",
    lambda r, c: f"fence posts, {c['mat']}, low",
    lambda r, c: f"a low fence of {c['n']} evenly spaced {c['mat']} posts",
    lambda r, c: "a low modern fence line of concrete posts, evenly spaced",
]
register("fence_hand", _b_fence_hand, PROMPTS_FENCE_HAND)


# 18. a small shrine, a box with a peaked roof
def _b_shrine(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.8, 1.0, 1.2])
    w = round(1.0 * scale, 2)
    h = round(1.4 * scale, 2)
    roof_h = round(0.7 * scale, 2)
    mk = rng.choice(["weathered_stone", "oak_wood", "sandstone"])
    roof_mk = rng.choice(["slate", "terracotta", "weathered_wood"])
    style = v
    box = nm("shrine_box", style)
    roof = nm("shrine_roof", style)
    calls = [
        prim("box", box, params={"size": [w, h, w]}, translation=[0, h / 2, 0]),
        material_call([box], mk, "Shrine Wall"),
        prim(
            "pyramid",
            roof,
            params={"base": w * 1.2, "height": roof_h, "sides": 4},
            translation=[0, h + roof_h / 2, 0],
        ),
        material_call([roof], roof_mk, "Shrine Roof"),
    ]
    notes = "a small box body with a peaked pyramid roof -- a wayside shrine's usual silhouette."
    return calls, notes, {"h": h + roof_h, "mat": MATERIALS[mk][3]}


PROMPTS_SHRINE = [
    lambda r, c: "a small shrine, a box with a peaked roof",
    lambda r, c: f"a small shrine, {fmt_m(c['h'])} m tall, {c['mat']} with a peaked roof",
    lambda r, c: "tiny wayside shrine, peaked roof",
    lambda r, c: f"shrine box in {c['mat']}, peaked top",
    lambda r, c: f"a small roadside shrine of {c['mat']} with a peaked roof over the box",
    lambda r, c: "a small modern shrine box with a peaked metal roof",
]
register("shrine", _b_shrine, PROMPTS_SHRINE)


# 19. a plain stone gatepost with a pyramid cap
def _b_gatepost_cap(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.2])
    w = round(0.35 * scale, 3)
    h = round(1.6 * scale, 2)
    cap_h = round(0.35 * scale, 3)
    mk = rng.choice(["weathered_stone", "granite", "sandstone"])
    style = v
    post = nm("gatepost", style)
    cap = nm("cap", style)
    calls = [
        prim("box", post, params={"size": [w, h, w]}, translation=[0, h / 2, 0]),
        prim(
            "pyramid",
            cap,
            params={"base": w * 1.15, "height": cap_h, "sides": 4},
            translation=[0, h + cap_h / 2, 0],
        ),
        material_call([post, cap], mk, "Gatepost Stone"),
    ]
    notes = "a square post capped with a small four-sided pyramid, the whole thing one material."
    return calls, notes, {"h": h + cap_h, "mat": MATERIALS[mk][3]}


PROMPTS_GATEPOST = [
    lambda r, c: "a plain stone gatepost with a pyramid cap",
    lambda r, c: f"a {c['mat']} gatepost, {fmt_m(c['h'])} m tall, pyramid cap on top",
    lambda r, c: "gatepost with a pointed cap",
    lambda r, c: f"gatepost, {c['mat']}, pyramid top",
    lambda r, c: f"a plain square gatepost of {c['mat']} finished with a small pyramid cap",
    lambda r, c: "a modern concrete gatepost with a pyramid cap",
]
register("gatepost_cap", _b_gatepost_cap, PROMPTS_GATEPOST)


# 20. a simple aqueduct arch spanning a narrow gap
def _b_aqueduct_arch(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.8, 1.0, 1.15])
    width = round(1.4 * scale, 2)
    height = round(2.0 * scale, 2)
    depth = round(1.0 * scale, 2)
    thickness = round(0.35 * scale, 2)
    mk = rng.choice(["weathered_stone", "granite", "sandstone"])
    style = v
    name = nm("aqueduct_arch", style)
    calls = [
        prim(
            "arch",
            name,
            params={
                "width": width,
                "height": height,
                "depth": depth,
                "thickness": thickness,
                "segments": 14,
            },
            translation=[0, height / 2, 0],
        ),
        material_call([name], mk, "Aqueduct Stone"),
    ]
    notes = "a single arch, deeper than the garden archway to read as one bay of an aqueduct run."
    return calls, notes, {"width": width, "mat": MATERIALS[mk][3]}


PROMPTS_AQUEDUCT = [
    lambda r, c: "a simple aqueduct arch spanning a narrow gap",
    lambda r, c: f"an aqueduct arch, {c['mat']}, {fmt_m(c['width'])} m across a narrow gap",
    lambda r, c: "one arch of an aqueduct, over a narrow gap",
    lambda r, c: f"aqueduct arch, {c['mat']}",
    lambda r, c: f"a single {c['mat']} aqueduct arch bridging a narrow gap",
    lambda r, c: "a single modern concrete aqueduct arch over a narrow channel",
]
register("aqueduct_arch", _b_aqueduct_arch, PROMPTS_AQUEDUCT)


# 21. a courtyard fountain basin, a wide shallow cylinder
def _b_fountain_basin(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.2])
    radius = round(1.3 * scale, 2)
    height = round(0.4 * scale, 2)
    mk = rng.choice(["marble_white", "granite", "weathered_stone"])
    style = v
    name = nm("basin", style)
    calls = [
        prim(
            "cylinder",
            name,
            params={"radius": radius, "height": height, "segments": 28},
            translation=[0, height / 2, 0],
        ),
        material_call([name], mk, "Basin Stone"),
    ]
    notes = "a wide, short cylinder -- radius well over its own height, so it reads as a basin rather than a drum."
    return calls, notes, {"radius": radius, "mat": MATERIALS[mk][3]}


PROMPTS_FOUNTAIN_BASIN = [
    lambda r, c: "a courtyard fountain basin, a wide shallow cylinder",
    lambda r, c: f"a fountain basin, {c['mat']}, {fmt_m(c['radius'] * 2)} m across, shallow",
    lambda r, c: "wide shallow basin for a courtyard fountain",
    lambda r, c: f"basin, {c['mat']}, wide and shallow",
    lambda r, c: f"a wide shallow fountain basin of {c['mat']} for a courtyard",
    lambda r, c: "a modern polished-marble fountain basin, wide and shallow",
]
register("fountain_basin", _b_fountain_basin, PROMPTS_FOUNTAIN_BASIN)


# 22. a stone staircase of five even steps (hand-placed)
def _b_staircase_five(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.9, 1.0, 1.1])
    n_steps = 5
    rise = round(0.18 * scale, 3)
    run = round(0.3 * scale, 3)
    width = round(1.4 * scale, 2)
    mk = rng.choice(["weathered_stone", "granite", "sandstone"])
    style = v
    names = []
    calls = []
    for i in range(n_steps):
        name = nm(f"step_{i + 1}", style)
        names.append(name)
        step_h = rise * (i + 1)
        calls.append(
            prim(
                "box",
                name,
                params={"size": [width, step_h, run]},
                translation=[0, step_h / 2, -run * i],
            )
        )
    calls.append(material_call(names, mk, "Staircase Stone"))
    notes = "five explicit steps, each solid from the ground to its own tread -- an even, simple flight."
    return calls, notes, {"n": n_steps, "mat": MATERIALS[mk][3]}


PROMPTS_STAIRCASE_FIVE = [
    lambda r, c: "a stone staircase of five even steps",
    lambda r, c: f"a {c['mat']} staircase, five even steps, {fmt_m(0.3)} m tread depth",
    lambda r, c: "five even stone steps, nothing fancy",
    lambda r, c: f"staircase, {c['mat']}, five steps",
    lambda r, c: f"a plain {c['mat']} staircase made of exactly five even steps",
    lambda r, c: "a modern concrete staircase of five even steps",
]
register("staircase_five", _b_staircase_five, PROMPTS_STAIRCASE_FIVE)


# 23. a squat guard tower with a single window opening (real boolean cut)
def _b_guard_tower_window(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    w = round(2.4 * scale, 2)
    h = round(3.2 * scale, 2)
    win_w = round(0.5 * scale, 2)
    win_h = round(0.8 * scale, 2)
    mk = rng.choice(["weathered_stone", "granite", "concrete"])
    style = v
    tower = nm("tower", style)
    cutter = nm("window_cutter", style)
    calls = [
        prim("box", tower, params={"size": [w, h, w]}, translation=[0, h / 2, 0]),
        prim("box", cutter, params={"size": [win_w, win_h, w * 1.4]}, translation=[0, h * 0.55, 0]),
        boolean_call("difference", [tower, cutter]),
        material_call([tower], mk, "Tower Stone"),
    ]
    notes = (
        "the window is a real hole: a cutter box deeper than the tower's own wall, "
        "subtracted with clay_boolean difference so the survivor (the tower, first in "
        "document order) keeps a genuine opening through it."
    )
    return calls, notes, {"h": h, "mat": MATERIALS[mk][3]}


PROMPTS_GUARD_TOWER = [
    lambda r, c: "a squat guard tower with a single window opening",
    lambda r, c: f"a {c['mat']} guard tower, {fmt_m(c['h'])} m tall, one window cut through it",
    lambda r, c: "squat guard tower, one window",
    lambda r, c: f"guard tower, {c['mat']}, single window",
    lambda r, c: (
        f"a squat {c['mat']} guard tower with a single window opening cut through the wall"
    ),
    lambda r, c: "a modern concrete guard tower with one window opening",
]
register("guard_tower_window", _b_guard_tower_window, PROMPTS_GUARD_TOWER)


# 24. a simple flagpole base, a short cylinder with a socket
def _b_flagpole_base(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    base_r = round(0.5 * scale, 2)
    base_h = round(0.3 * scale, 2)
    pole_r = round(0.05 * scale, 3)
    pole_h = round(4.0 * scale, 2)
    mk = rng.choice(["granite", "concrete", "weathered_stone"])
    pole_mk = rng.choice(["iron_dark", "bronze", "steel_scifi"])
    style = v
    base = nm("base", style)
    pole = nm("flagpole", style)
    calls = [
        prim(
            "cylinder",
            base,
            params={"radius": base_r, "height": base_h, "segments": 20},
            translation=[0, base_h / 2, 0],
        ),
        material_call([base], mk, "Base Stone"),
        prim(
            "cylinder",
            pole,
            params={"radius": pole_r, "height": pole_h, "segments": 12},
            translation=[0, base_h + pole_h / 2, 0],
        ),
        material_call([pole], pole_mk, "Flagpole Metal"),
    ]
    notes = "a short wide cylinder base carrying a thin tall cylinder pole; the socket is implied by the base rather than modelled as a cut hole."
    return calls, notes, {"h": pole_h, "mat": MATERIALS[pole_mk][3]}


PROMPTS_FLAGPOLE = [
    lambda r, c: "a simple flagpole base, a short cylinder with a socket",
    lambda r, c: f"a flagpole, {fmt_m(c['h'])} m tall, {c['mat']}, on a stone base",
    lambda r, c: "flagpole set on a short round block",
    lambda r, c: f"flagpole base, {c['mat']} pole",
    lambda r, c: f"a simple flagpole of {c['mat']} rising from a short cylindrical stone base",
    lambda r, c: "a modern steel flagpole on a concrete base",
]
register("flagpole_base", _b_flagpole_base, PROMPTS_FLAGPOLE)


# 25. a plain garden pergola, four posts supporting a flat lattice roof
def _b_pergola(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    post_size = round(0.15 * scale, 3)
    post_h = round(2.4 * scale, 2)
    span = round(2.6 * scale, 2)
    slat_h = round(0.08 * scale, 3)
    mk = rng.choice(["oak_wood", "weathered_wood"])
    style = v
    names = []
    calls = []
    half = span / 2 - post_size / 2
    for i, (sx, sz) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
        name = nm(f"post_{i + 1}", style)
        names.append(name)
        calls.append(
            prim(
                "box",
                name,
                params={"size": [post_size, post_h, post_size]},
                translation=[sx * half, post_h / 2, sz * half],
            )
        )
    n_slats = 5
    for i in range(n_slats):
        z = -span / 2 + (span / (n_slats - 1)) * i
        name = nm(f"slat_{i + 1}", style)
        names.append(name)
        calls.append(
            prim(
                "box",
                name,
                params={"size": [span, slat_h, post_size]},
                translation=[0, post_h + slat_h / 2, z],
            )
        )
    calls.append(material_call(names, mk, "Pergola Wood"))
    notes = f"four corner posts plus {n_slats} explicit cross-slats on top, standing in for a flat lattice roof."
    return calls, notes, {"h": post_h, "mat": MATERIALS[mk][3]}


PROMPTS_PERGOLA = [
    lambda r, c: "a plain garden pergola, four posts supporting a flat lattice roof",
    lambda r, c: f"a {c['mat']} pergola, four posts, {fmt_m(c['h'])} m tall, lattice roof",
    lambda r, c: "garden pergola, four posts and a lattice top",
    lambda r, c: f"pergola in {c['mat']}",
    lambda r, c: f"a plain garden pergola of {c['mat']} on four posts with a flat lattice roof",
    lambda r, c: "a modern garden pergola, steel posts, flat slatted roof",
]
register("pergola", _b_pergola, PROMPTS_PERGOLA)


# =============================================================================
# MEDIUM templates (seeds 26-45) -- array-linear / array-radial / boolean stars
# =============================================================================


# 26. a straight row of six plain round columns holding up a portico roof
def _b_portico_row(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = rng.choice([5, 6, 7])
    scale = rng.choice([0.9, 1.0, 1.15])
    radius = round(0.28 * scale, 3)
    height = round(3.0 * scale, 2)
    spacing = round(1.6 * scale, 2)
    mk = rng.choice(["weathered_stone", "marble_white", "granite"])
    roof_mk = rng.choice(["terracotta", "slate", "weathered_stone"])
    style = v
    col = nm("column", style)
    calls = [
        prim(
            "column",
            col,
            params={
                "radius": radius,
                "height": height,
                "segments": 16,
                "base": 0.1,
                "capital": 0.1,
            },
            translation=[-((count - 1) * spacing) / 2, height / 2, 0],
        ),
        material_call([col], mk, "Column Stone"),
        *array_linear(col, count=count, x=spacing),
    ]
    roof_w = (count - 1) * spacing + spacing
    roof = nm("portico_roof", style)
    calls.append(
        prim(
            "box",
            roof,
            params={"size": [roof_w, 0.25 * scale, 1.4 * scale]},
            translation=[0, height + 0.125 * scale, 0],
        )
    )
    calls.append(material_call([roof], roof_mk, "Roof Stone"))
    notes = (
        f"one column, materialed first so the colour survives duplication, then array-linear "
        f"({count} copies, {spacing} m apart along X) -- a flat roof slab spans the whole row "
        "afterwards, sized from the same count and spacing."
    )
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3], "roof_w": roof_w}


PROMPTS_PORTICO = [
    lambda r, c: "a straight row of six plain round columns holding up a portico roof",
    lambda r, c: f"a portico, {c['count']} round {c['mat']} columns in a row, flat roof over them",
    lambda r, c: f"row of {c['count']} columns holding up a porch roof",
    lambda r, c: f"portico columns, {c['mat']}, {c['count']} of them",
    lambda r, c: (
        f"a straight row of {c['count']} plain round columns of {c['mat']} carrying a portico roof"
    ),
    lambda r, c: f"a modern row of {c['count']} concrete columns supporting a flat portico roof",
]
register("portico_row", _b_portico_row, PROMPTS_PORTICO)


# 27. a ring of five plain round columns arranged around a small courtyard
def _b_ring_columns_courtyard(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = rng.choice([5, 6, 7])
    scale = rng.choice([0.9, 1.0, 1.2])
    radius = round(0.26 * scale, 3)
    height = round(2.6 * scale, 2)
    ring_r = round(2.2 * scale, 2)
    mk = rng.choice(["weathered_stone", "marble_white", "sandstone"])
    style = v
    col = nm("column", style)
    calls = [
        prim(
            "column",
            col,
            params={
                "radius": radius,
                "height": height,
                "segments": 16,
                "base": 0.1,
                "capital": 0.1,
            },
            translation=[ring_r, height / 2, 0],
        ),
        material_call([col], mk, "Column Stone"),
        *array_radial(col, count=count, angle=360.0, axis=1.0),
    ]
    notes = (
        f"one column placed {ring_r} m out along X, materialed, then array-radial about the "
        f"world origin (axis=1, the Y axis) with count={count} and angle=360 so every copy "
        "spaces evenly around the whole circle -- the origin is the courtyard centre."
    )
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3]}


PROMPTS_RING_COURTYARD = [
    lambda r, c: f"a ring of {c['count']} plain round columns arranged around a small courtyard",
    lambda r, c: f"a ring of {c['count']} round {c['mat']} columns around a small courtyard",
    lambda r, c: f"columns in a circle, {c['count']} of them, around a courtyard",
    lambda r, c: f"ring of columns, {c['mat']}, {c['count']} total",
    lambda r, c: (
        f"{c['count']} plain round columns of {c['mat']} arranged evenly in a ring around a small courtyard"
    ),
    lambda r, c: f"a modern ring of {c['count']} steel columns around a small plaza",
]
register("ring_columns_courtyard", _b_ring_columns_courtyard, PROMPTS_RING_COURTYARD)


# 28. a staircase with a repeating pattern of identical steps (array-linear w/ offsets)
def _b_staircase_array(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = rng.choice([6, 7, 9])
    scale = rng.choice([0.9, 1.0, 1.1])
    rise = round(0.18 * scale, 3)
    run = round(0.3 * scale, 3)
    width = round(1.3 * scale, 2)
    mk = rng.choice(["weathered_stone", "granite", "concrete"])
    style = v
    step = nm("step", style)
    calls = [
        prim("box", step, params={"size": [width, rise, run]}, translation=[0, rise / 2, 0]),
        material_call([step], mk, "Step Stone"),
        *array_linear(step, count=count, y=rise, z=-run),
    ]
    notes = (
        f"one step, materialed, then array-linear with both a y step ({rise} m, the rise) and "
        f"a z step (-{run} m, the run) at once -- the brief's own 'array-linear of boxes with "
        f"offsets' -- so {count} identical steps climb together in one op rather than {count} "
        "hand-placed boxes."
    )
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3]}


PROMPTS_STAIRCASE_ARRAY = [
    lambda r, c: "a staircase with a repeating pattern of identical steps",
    lambda r, c: f"a staircase, {c['count']} identical repeating steps, {c['mat']}",
    lambda r, c: f"stairs, {c['count']} steps, all the same, one after another",
    lambda r, c: f"repeating staircase, {c['mat']}",
    lambda r, c: (
        f"a {c['mat']} staircase built from {c['count']} identical steps repeating up the run"
    ),
    lambda r, c: f"a modern concrete staircase of {c['count']} identical repeating steps",
]
register("staircase_array", _b_staircase_array, PROMPTS_STAIRCASE_ARRAY)


# 29. a castle wall with a repeated row of arrow-slit windows (boolean, explicit cutters)
def _b_wall_arrow_slits(rng: random.Random, v: int) -> tuple[list, str, dict]:
    n_slits = rng.choice([4, 5, 6])
    scale = rng.choice([0.9, 1.0, 1.15])
    wall_len = round(6.0 * scale, 2)
    wall_h = round(2.6 * scale, 2)
    wall_t = round(0.5 * scale, 2)
    slit_w = round(0.12 * scale, 3)
    slit_h = round(0.8 * scale, 2)
    mk = rng.choice(["weathered_stone", "granite", "dark_basalt"])
    style = v
    wall = nm("wall", style)
    calls = [
        prim(
            "box", wall, params={"size": [wall_len, wall_h, wall_t]}, translation=[0, wall_h / 2, 0]
        )
    ]
    cutter_names = []
    spacing = wall_len / (n_slits + 1)
    for i in range(n_slits):
        x = -wall_len / 2 + spacing * (i + 1)
        name = nm(f"slit_{i + 1}", style)
        cutter_names.append(name)
        calls.append(
            prim(
                "box",
                name,
                params={"size": [slit_w, slit_h, wall_t * 1.4]},
                translation=[x, wall_h * 0.55, 0],
            )
        )
    calls.append(boolean_call("difference", [wall, *cutter_names]))
    calls.append(material_call([wall], mk, "Wall Stone"))
    notes = (
        f"{n_slits} narrow cutter boxes placed by hand (each deeper than the wall so the cut "
        "goes all the way through), then one clay_boolean difference call against the wall "
        "and all of them at once -- the survivor keeps every slit as a real opening."
    )
    return calls, notes, {"n": n_slits, "mat": MATERIALS[mk][3], "len": wall_len}


PROMPTS_WALL_SLITS = [
    lambda r, c: "a castle wall with a repeated row of arrow-slit windows",
    lambda r, c: (
        f"a {c['mat']} castle wall, {fmt_m(c['len'])} m long, {c['n']} arrow-slits cut through it"
    ),
    lambda r, c: f"castle wall with {c['n']} arrow slits in a row",
    lambda r, c: f"wall, {c['mat']}, arrow slits repeated",
    lambda r, c: (
        f"a {c['mat']} castle wall with a repeated row of {c['n']} narrow arrow-slit windows cut through it"
    ),
    lambda r, c: f"a modern concrete wall with {c['n']} narrow slit windows cut in a row",
]
register("wall_arrow_slits", _b_wall_arrow_slits, PROMPTS_WALL_SLITS)


# 30. a temple facade with a row of matching columns and a triangular pediment
def _b_temple_facade(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = rng.choice([5, 6, 7])
    scale = rng.choice([0.9, 1.0, 1.15])
    radius = round(0.3 * scale, 3)
    height = round(3.4 * scale, 2)
    spacing = round(1.5 * scale, 2)
    mk = rng.choice(["marble_white", "weathered_stone", "sandstone"])
    facade_mk = rng.choice(["dark_basalt", "slate", "granite"])
    style = v
    col = nm("column", style)
    row_w = (count - 1) * spacing
    calls = [
        prim(
            "column",
            col,
            params={
                "radius": radius,
                "height": height,
                "segments": 16,
                "base": 0.12,
                "capital": 0.12,
            },
            translation=[-row_w / 2, height / 2, 0],
        ),
        material_call([col], mk, "Column Stone"),
        *array_linear(col, count=count, x=spacing),
    ]
    facade = nm("facade_wall", style)
    # a deliberately darker/different material from the columns -- against a
    # flat-shaded orthographic render, same-coloured touching parts merge
    # into one silhouette with no visible seam, which hid the row of columns
    # entirely against a same-toned facade wall.
    calls.append(
        prim(
            "box",
            facade,
            params={"size": [row_w + spacing, height * 0.9, 0.4 * scale]},
            translation=[0, height * 0.45, -0.8 * scale],
        )
    )
    calls.append(material_call([facade], facade_mk, "Facade Stone"))
    pediment = nm("pediment", style)
    pediment_h = round(1.2 * scale, 2)
    calls.append(
        prim(
            "pyramid",
            pediment,
            params={"base": row_w + spacing, "height": pediment_h, "sides": 4},
            translation=[0, height + pediment_h / 2, 0],
            rotation=[0, 45, 0],
        )
    )
    calls.append(material_call([pediment], mk, "Pediment Stone"))
    notes = (
        "an array-linear row of columns, a recessed facade wall behind them, and a four-sided "
        "pyramid rotated 45 degrees about Y so one of its edges (rather than a face) points "
        "forward -- reading as a triangular gable end rather than a square-fronted roof."
    )
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3]}


PROMPTS_TEMPLE_FACADE = [
    lambda r, c: "a temple facade with a row of matching columns and a triangular pediment",
    lambda r, c: f"a temple facade, {c['count']} {c['mat']} columns, triangular pediment above",
    lambda r, c: "temple front, row of columns and a triangle roof",
    lambda r, c: f"temple facade in {c['mat']}",
    lambda r, c: (
        f"a {c['mat']} temple facade with {c['count']} matching columns under a triangular pediment"
    ),
    lambda r, c: f"a modern marble temple facade, {c['count']} columns, triangular pediment",
]
register("temple_facade", _b_temple_facade, PROMPTS_TEMPLE_FACADE)


# 31. an arcade of repeating stone arches along a covered walkway
def _b_arcade_arches(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = rng.choice([4, 5, 6])
    scale = rng.choice([0.9, 1.0, 1.15])
    width = round(1.6 * scale, 2)
    height = round(2.4 * scale, 2)
    depth = round(0.5 * scale, 2)
    thickness = round(0.3 * scale, 2)
    mk = rng.choice(["weathered_stone", "sandstone", "granite"])
    style = v
    arch_name = nm("arch", style)
    calls = [
        prim(
            "arch",
            arch_name,
            params={
                "width": width,
                "height": height,
                "depth": depth,
                "thickness": thickness,
                "segments": 14,
            },
            translation=[0, height / 2, 0],
        ),
        material_call([arch_name], mk, "Arcade Stone"),
        *array_linear(arch_name, count=count, x=width),
    ]
    notes = f"one arch, materialed, array-linear along X with a step equal to its own width so {count} bays sit flush edge to edge with no gap or overlap."
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3]}


PROMPTS_ARCADE = [
    lambda r, c: "an arcade of repeating stone arches along a covered walkway",
    lambda r, c: f"an arcade, {c['count']} {c['mat']} arches in a row, covered walkway",
    lambda r, c: f"walkway with {c['count']} arches repeating along it",
    lambda r, c: f"arcade of arches, {c['mat']}",
    lambda r, c: f"a covered arcade walkway made of {c['count']} repeating {c['mat']} arches",
    lambda r, c: f"a modern arcade of {c['count']} repeating concrete arches along a walkway",
]
register("arcade_arches", _b_arcade_arches, PROMPTS_ARCADE)


# 32. a tower with a spiral staircase wrapped around its central column (hand-placed spiral)
def _b_spiral_staircase_tower(rng: random.Random, v: int) -> tuple[list, str, dict]:
    import math

    scale = rng.choice([0.9, 1.0, 1.15])
    col_r = round(0.9 * scale, 2)
    tower_h = round(6.0 * scale, 2)
    n_steps = 12
    tread_radial = round(0.4 * scale, 3)
    tread_tangential = round(0.55 * scale, 3)
    step_h = round(tower_h / (n_steps * 1.4), 3)
    mk = rng.choice(["weathered_stone", "granite", "concrete"])
    style = v
    column = nm("central_column", style)
    calls = [
        prim(
            "cylinder",
            column,
            params={"radius": col_r, "height": tower_h, "segments": 24},
            translation=[0, tower_h / 2, 0],
        ),
        material_call([column], mk, "Tower Stone"),
    ]
    step_names = []
    rise_per_step = tower_h * 0.85 / n_steps
    orbit = col_r + tread_radial / 2
    for i in range(n_steps):
        angle = i * (360.0 / n_steps)
        rad = math.radians(angle)
        x = orbit * math.cos(rad)
        z = orbit * math.sin(rad)
        y = 0.2 + rise_per_step * i
        name = nm(f"spiral_step_{i + 1}", style)
        step_names.append(name)
        calls.append(
            prim(
                "box",
                name,
                params={"size": [tread_radial, step_h, tread_tangential]},
                translation=[x, y + step_h / 2, z],
                rotation=[0, -angle, 0],
            )
        )
    calls.append(material_call(step_names, mk, "Step Stone"))
    notes = (
        f"{n_steps} explicit tread boxes, each rotated about Y by its own angle and raised a "
        "little more than the last, projecting outward from a solid central column's own "
        "surface -- an exterior spiral ramp (the pattern a real minaret's outdoor stair uses), "
        "chosen over an interior stair specifically so the treads sit outside the column's "
        "radius and stay visible in a render rather than hidden inside solid stone. Built by "
        "hand rather than by an op, since neither array op combines a rotation with a "
        "simultaneous rise."
    )
    return calls, notes, {"h": tower_h, "mat": MATERIALS[mk][3]}


PROMPTS_SPIRAL_TOWER = [
    lambda r, c: "a tower with a spiral staircase wrapped around its central column",
    lambda r, c: (
        f"a tower, {fmt_m(c['h'])} m tall, spiral stairs around a central column, {c['mat']}"
    ),
    lambda r, c: "tower with a spiral stair around the middle column",
    lambda r, c: f"spiral-stair tower, {c['mat']}",
    lambda r, c: f"a {c['mat']} tower whose spiral staircase winds around a thin central column",
    lambda r, c: "a modern concrete tower with a spiral staircase around its central column",
]
register("spiral_staircase_tower", _b_spiral_staircase_tower, PROMPTS_SPIRAL_TOWER)


# 33. a fence made from a linear array of identical pickets
def _b_fence_array(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = rng.choice([9, 10, 12])
    scale = rng.choice([0.9, 1.0, 1.15])
    picket_w = round(0.08 * scale, 3)
    picket_h = round(1.0 * scale, 2)
    spacing = round(0.25 * scale, 3)
    mk = rng.choice(["oak_wood", "weathered_wood"])
    style = v
    picket = nm("picket", style)
    calls = [
        prim(
            "box",
            picket,
            params={"size": [picket_w, picket_h, picket_w]},
            translation=[-((count - 1) * spacing) / 2, picket_h / 2, 0],
        ),
        material_call([picket], mk, "Picket Wood"),
        *array_linear(picket, count=count, x=spacing),
    ]
    notes = f"one picket, materialed, then array-linear along X with a {spacing} m step -- {count} identical pickets in one op, exactly what a linear array is for."
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3]}


PROMPTS_FENCE_ARRAY = [
    lambda r, c: "a fence made from a linear array of identical pickets",
    lambda r, c: f"a fence, {c['count']} identical {c['mat']} pickets in a line",
    lambda r, c: f"picket fence, {c['count']} pickets, all the same",
    lambda r, c: f"linear fence of pickets, {c['mat']}",
    lambda r, c: f"a fence built from a linear array of {c['count']} identical {c['mat']} pickets",
    lambda r, c: f"a modern fence of {c['count']} identical steel pickets in a straight line",
]
register("fence_array", _b_fence_array, PROMPTS_FENCE_ARRAY)


# 34. a small rotunda roof held up by five columns in a radial ring
def _b_rotunda_columns(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = rng.choice([5, 6, 7])
    scale = rng.choice([0.9, 1.0, 1.15])
    radius = round(0.24 * scale, 3)
    height = round(2.4 * scale, 2)
    ring_r = round(1.8 * scale, 2)
    mk = rng.choice(["marble_white", "weathered_stone", "sandstone"])
    roof_mk = rng.choice(["terracotta", "slate", "bronze"])
    style = v
    col = nm("column", style)
    calls = [
        prim(
            "column",
            col,
            params={
                "radius": radius,
                "height": height,
                "segments": 16,
                "base": 0.1,
                "capital": 0.1,
            },
            translation=[ring_r, height / 2, 0],
        ),
        material_call([col], mk, "Column Stone"),
        *array_radial(col, count=count, angle=360.0, axis=1.0),
    ]
    roof = nm("rotunda_roof", style)
    roof_h = round(1.0 * scale, 2)
    calls.append(
        prim(
            "cone",
            roof,
            params={"radius": ring_r * 1.15, "height": roof_h, "segments": 24},
            translation=[0, height + roof_h / 2, 0],
        )
    )
    calls.append(material_call([roof], roof_mk, "Rotunda Roof"))
    notes = f"array-radial ring of {count} columns (axis=1/Y, angle=360) with a shallow cone standing in for the domed roof over the whole ring."
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3]}


PROMPTS_ROTUNDA = [
    lambda r, c: "a small rotunda roof held up by five columns in a radial ring",
    lambda r, c: f"a rotunda, {c['count']} {c['mat']} columns in a ring, domed roof",
    lambda r, c: f"small round roof on {c['count']} columns in a circle",
    lambda r, c: f"rotunda, {c['mat']}, {c['count']} columns",
    lambda r, c: (
        f"a small rotunda roof carried on {c['count']} columns of {c['mat']} arranged in a radial ring"
    ),
    lambda r, c: f"a modern rotunda roof on {c['count']} steel columns in a ring",
]
register("rotunda_columns", _b_rotunda_columns, PROMPTS_ROTUNDA)


# 35. a bridge with a repeating row of arch supports underneath
def _b_bridge_arch_supports(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = rng.choice([3, 4, 5])
    scale = rng.choice([0.9, 1.0, 1.15])
    arch_w = round(1.4 * scale, 2)
    arch_h = round(1.6 * scale, 2)
    depth = round(1.0 * scale, 2)
    thickness = round(0.3 * scale, 2)
    deck_t = round(0.3 * scale, 2)
    mk = rng.choice(["weathered_stone", "granite", "concrete"])
    style = v
    arch_name = nm("arch_support", style)
    calls = [
        prim(
            "arch",
            arch_name,
            params={
                "width": arch_w,
                "height": arch_h,
                "depth": depth,
                "thickness": thickness,
                "segments": 14,
            },
            translation=[0, arch_h / 2, 0],
        ),
        material_call([arch_name], mk, "Support Stone"),
        *array_linear(arch_name, count=count, x=arch_w),
    ]
    deck = nm("deck", style)
    deck_len = count * arch_w
    calls.append(
        prim(
            "box",
            deck,
            params={"size": [deck_len, deck_t, depth * 1.1]},
            translation=[(count - 1) * arch_w / 2, arch_h + deck_t / 2, 0],
        )
    )
    calls.append(material_call([deck], mk, "Deck Stone"))
    notes = f"array-linear row of {count} arches, flush edge to edge, with one deck slab spanning across the top of all of them."
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3]}


PROMPTS_BRIDGE_ARCHES = [
    lambda r, c: "a bridge with a repeating row of arch supports underneath",
    lambda r, c: f"a bridge, {c['count']} {c['mat']} arch supports underneath, deck on top",
    lambda r, c: f"bridge with {c['count']} arches under the deck",
    lambda r, c: f"bridge on arches, {c['mat']}",
    lambda r, c: (
        f"a {c['mat']} bridge deck carried on a repeating row of {c['count']} arch supports"
    ),
    lambda r, c: f"a modern concrete bridge on a repeating row of {c['count']} arch supports",
]
register("bridge_arch_supports", _b_bridge_arch_supports, PROMPTS_BRIDGE_ARCHES)


# 36. a garden gazebo with six columns arranged in a circle under a domed roof
def _b_gazebo_domed(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = rng.choice([6, 7, 10])
    scale = rng.choice([0.9, 1.0, 1.15])
    radius = round(0.2 * scale, 3)
    height = round(2.2 * scale, 2)
    ring_r = round(1.9 * scale, 2)
    mk = rng.choice(["oak_wood", "weathered_stone", "marble_white"])
    roof_mk = rng.choice(["terracotta", "slate", "verdigris_copper"])
    style = v
    col = nm("column", style)
    calls = [
        prim(
            "column",
            col,
            params={
                "radius": radius,
                "height": height,
                "segments": 16,
                "base": 0.08,
                "capital": 0.08,
            },
            translation=[ring_r, height / 2, 0],
        ),
        material_call([col], mk, "Column Stone"),
        *array_radial(col, count=count, angle=360.0, axis=1.0),
    ]
    roof = nm("gazebo_roof", style)
    roof_h = round(1.1 * scale, 2)
    calls.append(
        prim(
            "cone",
            roof,
            params={"radius": ring_r * 1.2, "height": roof_h, "segments": 20},
            translation=[0, height + roof_h / 2, 0],
        )
    )
    calls.append(material_call([roof], roof_mk, "Gazebo Roof"))
    notes = f"array-radial ring of {count} columns (axis=1/Y) under a cone standing in for the gazebo's domed roof."
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3]}


PROMPTS_GAZEBO = [
    lambda r, c: "a garden gazebo with six columns arranged in a circle under a domed roof",
    lambda r, c: f"a gazebo, {c['count']} {c['mat']} columns in a circle, domed roof",
    lambda r, c: f"round gazebo, {c['count']} columns, domed top",
    lambda r, c: f"gazebo, {c['mat']}, {c['count']} columns",
    lambda r, c: (
        f"a garden gazebo with {c['count']} columns of {c['mat']} arranged in a circle under a domed roof"
    ),
    lambda r, c: f"a modern gazebo with {c['count']} steel columns in a circle under a domed roof",
]
register("gazebo_domed", _b_gazebo_domed, PROMPTS_GAZEBO)


# 37. a keep wall with evenly spaced buttresses
def _b_keep_wall_buttresses(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = rng.choice([4, 5, 7])
    scale = rng.choice([0.9, 1.0, 1.15])
    wall_len = round((count + 1) * 1.6 * scale, 2)
    wall_h = round(3.5 * scale, 2)
    wall_t = round(0.5 * scale, 2)
    but_w = round(0.5 * scale, 2)
    but_d = round(0.4 * scale, 2)
    but_h = round(wall_h * 0.85, 2)
    mk = rng.choice(["weathered_stone", "granite", "dark_basalt"])
    style = v
    wall = nm("wall", style)
    calls = [
        prim(
            "box", wall, params={"size": [wall_len, wall_h, wall_t]}, translation=[0, wall_h / 2, 0]
        ),
        material_call([wall], mk, "Wall Stone"),
    ]
    spacing = wall_len / (count + 1)
    first_x = -wall_len / 2 + spacing
    buttress = nm("buttress", style)
    calls.append(
        prim(
            "box",
            buttress,
            params={"size": [but_w, but_h, but_d]},
            translation=[first_x, but_h / 2, wall_t / 2 + but_d / 2],
        )
    )
    calls.append(material_call([buttress], mk, "Buttress Stone"))
    calls += array_linear(buttress, count=count, x=spacing)
    notes = (
        f"the wall first, then one buttress (materialed before duplication) array-linear "
        f"along X with the same spacing that lays out the count on the wall -- {count} "
        "buttresses land evenly without needing to know any of the copies' own uids."
    )
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3], "len": wall_len}


PROMPTS_KEEP_WALL = [
    lambda r, c: "a keep wall with evenly spaced buttresses",
    lambda r, c: (
        f"a keep wall, {fmt_m(c['len'])} m long, {c['count']} evenly spaced {c['mat']} buttresses"
    ),
    lambda r, c: f"keep wall with {c['count']} buttresses along it",
    lambda r, c: f"wall with buttresses, {c['mat']}",
    lambda r, c: f"a {c['mat']} keep wall reinforced with {c['count']} evenly spaced buttresses",
    lambda r, c: f"a modern concrete retaining wall with {c['count']} evenly spaced buttresses",
]
register("keep_wall_buttresses", _b_keep_wall_buttresses, PROMPTS_KEEP_WALL)


# 38. a row of plain round columns forming a covered courtyard walkway
def _b_row_columns_walkway(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = rng.choice([7, 9, 10])
    scale = rng.choice([0.9, 1.0, 1.1])
    radius = round(0.24 * scale, 3)
    height = round(2.8 * scale, 2)
    spacing = round(1.4 * scale, 2)
    mk = rng.choice(["weathered_stone", "sandstone", "granite"])
    roof_mk = rng.choice(["terracotta", "slate", "weathered_wood"])
    style = v
    col = nm("column", style)
    row_w = (count - 1) * spacing
    calls = [
        prim(
            "column",
            col,
            params={
                "radius": radius,
                "height": height,
                "segments": 16,
                "base": 0.1,
                "capital": 0.1,
            },
            translation=[-row_w / 2, height / 2, 0],
        ),
        material_call([col], mk, "Column Stone"),
        *array_linear(col, count=count, x=spacing),
    ]
    roof = nm("walkway_roof", style)
    calls.append(
        prim(
            "box",
            roof,
            params={"size": [row_w + spacing, 0.2 * scale, 1.6 * scale]},
            translation=[0, height + 0.1 * scale, 0],
        )
    )
    calls.append(material_call([roof], roof_mk, "Walkway Roof"))
    notes = f"array-linear row of {count} columns along a courtyard side, flat roof slab spanning the whole row."
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3]}


PROMPTS_ROW_WALKWAY = [
    lambda r, c: "a row of plain round columns forming a covered courtyard walkway",
    lambda r, c: f"a covered walkway, {c['count']} plain round {c['mat']} columns in a row",
    lambda r, c: f"walkway with {c['count']} columns holding the roof up",
    lambda r, c: f"covered walkway, columns in {c['mat']}",
    lambda r, c: (
        f"a covered courtyard walkway formed by a row of {c['count']} plain round columns of {c['mat']}"
    ),
    lambda r, c: f"a modern covered walkway on a row of {c['count']} concrete columns",
]
register("row_columns_walkway", _b_row_columns_walkway, PROMPTS_ROW_WALKWAY)


# 39. a windmill tower with a boolean-cut doorway near its base
def _b_windmill_doorway(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.9, 1.0, 1.15])
    radius = round(1.6 * scale, 2)
    height = round(6.0 * scale, 2)
    door_w = round(0.7 * scale, 2)
    door_h = round(1.6 * scale, 2)
    mk = rng.choice(["weathered_stone", "sandstone", "concrete"])
    style = v
    tower = nm("windmill_tower", style)
    cutter = nm("doorway_cutter", style)
    calls = [
        prim(
            "cone",
            tower,
            params={"radius": radius, "height": height, "segments": 24},
            translation=[0, height / 2, 0],
        ),
        prim(
            "box",
            cutter,
            params={"size": [door_w, door_h, radius * 2.2]},
            translation=[0, door_h / 2, 0],
        ),
        boolean_call("difference", [tower, cutter]),
        material_call([tower], mk, "Windmill Wall"),
    ]
    notes = "a tapering cone tower with a doorway-shaped cutter box subtracted near the base -- the cutter runs clean through the tower's own diameter so the cut is never a shallow dent."
    return calls, notes, {"h": height, "mat": MATERIALS[mk][3]}


PROMPTS_WINDMILL = [
    lambda r, c: "a windmill tower with a boolean-cut doorway near its base",
    lambda r, c: f"a windmill tower, {fmt_m(c['h'])} m tall, {c['mat']}, doorway cut near the base",
    lambda r, c: "windmill tower with a door cut into the bottom",
    lambda r, c: f"windmill, {c['mat']}, doorway at the base",
    lambda r, c: f"a tapering {c['mat']} windmill tower with a doorway boolean-cut near its base",
    lambda r, c: "a modern concrete windmill tower with a doorway cut through its base",
]
register("windmill_doorway", _b_windmill_doorway, PROMPTS_WINDMILL)


# 40. a stone archway with a keystone cut into the top of the curve (accent piece)
def _b_archway_keystone(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.9, 1.0, 1.15])
    width = round(2.0 * scale, 2)
    height = round(2.6 * scale, 2)
    depth = round(0.5 * scale, 2)
    thickness = round(0.3 * scale, 2)
    mk = rng.choice(["weathered_stone", "sandstone", "granite"])
    key_mk = rng.choice(["dark_basalt", "granite", "bronze"])
    style = v
    arch_name = nm("archway", style)
    key = nm("keystone", style)
    calls = [
        prim(
            "arch",
            arch_name,
            params={
                "width": width,
                "height": height,
                "depth": depth,
                "thickness": thickness,
                "segments": 16,
            },
            translation=[0, height / 2, 0],
        ),
        material_call([arch_name], mk, "Archway Stone"),
        prim(
            "box",
            key,
            params={"size": [thickness * 1.3, thickness * 1.3, depth * 1.05]},
            translation=[0, height - thickness * 0.15, 0],
        ),
        material_call([key], key_mk, "Keystone Accent"),
    ]
    notes = (
        "the keystone is a small distinct-material block placed at the crown of the arch "
        "rather than a literal cut -- an arch's own opening is a horseshoe, not a hole through "
        "material (see primitives.arch's own docstring), so there is nothing to boolean-cut a "
        "keystone shape into without breaking the arch's closed-solid topology."
    )
    return calls, notes, {"width": width, "mat": MATERIALS[mk][3]}


PROMPTS_ARCHWAY_KEYSTONE = [
    lambda r, c: "a stone archway with a keystone cut into the top of the curve",
    lambda r, c: (
        f"a {c['mat']} archway, {fmt_m(c['width'])} m wide, with a keystone accent at the crown"
    ),
    lambda r, c: "archway with a keystone block at the top of the curve",
    lambda r, c: f"archway, {c['mat']}, keystone at the top",
    lambda r, c: f"a {c['mat']} archway with a distinct keystone block set at the top of the curve",
    lambda r, c: "a modern archway with a bronze keystone accent at the crown",
]
register("archway_keystone", _b_archway_keystone, PROMPTS_ARCHWAY_KEYSTONE)


# 41. a tower with window openings cut at regular intervals up its height
def _b_tower_window_intervals(rng: random.Random, v: int) -> tuple[list, str, dict]:
    n_windows = rng.choice([3, 4, 5])
    scale = rng.choice([0.9, 1.0, 1.15])
    w = round(2.0 * scale, 2)
    h = round(6.0 * scale, 2)
    win_w = round(0.4 * scale, 2)
    win_h = round(0.6 * scale, 2)
    mk = rng.choice(["weathered_stone", "granite", "concrete"])
    style = v
    tower = nm("tower", style)
    calls = [prim("box", tower, params={"size": [w, h, w]}, translation=[0, h / 2, 0])]
    cutter_names = []
    spacing = h / (n_windows + 1)
    for i in range(n_windows):
        y = spacing * (i + 1)
        name = nm(f"window_{i + 1}", style)
        cutter_names.append(name)
        calls.append(
            prim("box", name, params={"size": [win_w, win_h, w * 1.4]}, translation=[0, y, 0])
        )
    calls.append(boolean_call("difference", [tower, *cutter_names]))
    calls.append(material_call([tower], mk, "Tower Stone"))
    notes = f"{n_windows} cutter boxes stacked evenly up the tower's own height, subtracted in one clay_boolean difference call against the tower."
    return calls, notes, {"n": n_windows, "mat": MATERIALS[mk][3], "h": h}


PROMPTS_TOWER_WINDOWS = [
    lambda r, c: "a tower with window openings cut at regular intervals up its height",
    lambda r, c: f"a {c['mat']} tower, {fmt_m(c['h'])} m tall, {c['n']} windows cut up its height",
    lambda r, c: f"tower with {c['n']} windows, evenly spaced up the height",
    lambda r, c: f"tower, {c['mat']}, windows at intervals",
    lambda r, c: (
        f"a {c['mat']} tower with {c['n']} window openings cut at regular intervals up its height"
    ),
    lambda r, c: f"a modern concrete tower with {c['n']} window openings at regular intervals",
]
register("tower_window_intervals", _b_tower_window_intervals, PROMPTS_TOWER_WINDOWS)


# 42. a set of five matching round columns holding up a temple roof
def _b_five_columns_temple_roof(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = rng.choice([5, 6, 7])
    scale = rng.choice([0.9, 1.0, 1.15])
    radius = round(0.3 * scale, 3)
    height = round(3.2 * scale, 2)
    spacing = round(1.5 * scale, 2)
    mk = rng.choice(["marble_white", "weathered_stone", "sandstone"])
    style = v
    col = nm("column", style)
    row_w = (count - 1) * spacing
    calls = [
        prim(
            "column",
            col,
            params={
                "radius": radius,
                "height": height,
                "segments": 16,
                "base": 0.12,
                "capital": 0.12,
            },
            translation=[-row_w / 2, height / 2, 0],
        ),
        material_call([col], mk, "Column Stone"),
        *array_linear(col, count=count, x=spacing),
    ]
    roof = nm("temple_roof", style)
    calls.append(
        prim(
            "box",
            roof,
            params={"size": [row_w + spacing, 0.3 * scale, 2.0 * scale]},
            translation=[0, height + 0.15 * scale, 0],
        )
    )
    calls.append(material_call([roof], mk, "Temple Roof"))
    notes = f"array-linear row of {count} matching columns with a flat slab roof spanning them, same material throughout for the classic temple-front look."
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3]}


PROMPTS_FIVE_TEMPLE = [
    lambda r, c: "a set of five matching round columns holding up a temple roof",
    lambda r, c: f"{c['count']} matching {c['mat']} columns holding up a temple roof",
    lambda r, c: f"temple roof on {c['count']} matching columns",
    lambda r, c: f"{c['count']} columns under a roof, {c['mat']}",
    lambda r, c: (
        f"a set of {c['count']} matching round columns of {c['mat']} holding up a flat temple roof"
    ),
    lambda r, c: f"a modern temple roof carried on {c['count']} matching marble columns",
]
register("five_columns_temple_roof", _b_five_columns_temple_roof, PROMPTS_FIVE_TEMPLE)


# 43. a battlement wall whose merlons repeat evenly along its length (array-linear)
def _b_battlement_array(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = rng.choice([5, 7, 9])
    scale = rng.choice([0.9, 1.0, 1.15])
    wall_len = round((count * 2) * 0.5 * scale, 2)
    wall_h = round(2.4 * scale, 2)
    wall_t = round(0.5 * scale, 2)
    merlon_w = round(wall_len / (count * 2), 3)
    merlon_h = round(0.6 * scale, 3)
    mk = rng.choice(["weathered_stone", "granite", "dark_basalt"])
    style = v
    wall = nm("wall", style)
    calls = [
        prim(
            "box", wall, params={"size": [wall_len, wall_h, wall_t]}, translation=[0, wall_h / 2, 0]
        ),
        material_call([wall], mk, "Wall Stone"),
    ]
    merlon = nm("merlon", style)
    spacing = merlon_w * 2
    first_x = -wall_len / 2 + merlon_w / 2
    calls.append(
        prim(
            "box",
            merlon,
            params={"size": [merlon_w, merlon_h, wall_t]},
            translation=[first_x, wall_h + merlon_h / 2, 0],
        )
    )
    calls.append(material_call([merlon], mk, "Merlon Stone"))
    calls += array_linear(merlon, count=count, x=spacing)
    notes = f"one merlon (materialed first), array-linear along X with a step of twice its own width so the gaps between the {count} merlons equal a merlon's own width."
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3], "len": wall_len}


PROMPTS_BATTLEMENT_ARRAY = [
    lambda r, c: "a battlement wall whose merlons repeat evenly along its length",
    lambda r, c: (
        f"a battlement wall, {fmt_m(c['len'])} m long, {c['count']} evenly repeating {c['mat']} merlons"
    ),
    lambda r, c: f"wall with {c['count']} merlons repeating along the top",
    lambda r, c: f"battlements, {c['mat']}, repeating merlons",
    lambda r, c: (
        f"a {c['mat']} battlement wall whose {c['count']} merlons repeat evenly along its whole length"
    ),
    lambda r, c: f"a modern concrete wall with {c['count']} repeating square merlons along the top",
]
register("battlement_array", _b_battlement_array, PROMPTS_BATTLEMENT_ARRAY)


# 44. a courtyard well surrounded by a ring of six support posts
def _b_well_ring_posts(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = rng.choice([5, 6, 7])
    scale = rng.choice([0.9, 1.0, 1.15])
    well_r = round(0.6 * scale, 2)
    well_h = round(0.5 * scale, 2)
    post_r = round(0.08 * scale, 3)
    post_h = round(1.8 * scale, 2)
    ring_r = round(0.9 * scale, 2)
    mk = rng.choice(["weathered_stone", "granite", "moss_stone"])
    post_mk = rng.choice(["oak_wood", "weathered_wood"])
    style = v
    well = nm("well_wall", style)
    post = nm("support_post", style)
    calls = [
        prim(
            "cylinder",
            well,
            params={"radius": well_r, "height": well_h, "segments": 20},
            translation=[0, well_h / 2, 0],
        ),
        material_call([well], mk, "Well Stone"),
        prim(
            "cylinder",
            post,
            params={"radius": post_r, "height": post_h, "segments": 12},
            translation=[ring_r, post_h / 2, 0],
        ),
        material_call([post], post_mk, "Post Wood"),
        *array_radial(post, count=count, angle=360.0, axis=1.0),
    ]
    notes = f"a squat well cylinder at the origin, then a ring of {count} support posts array-radial'd around it (axis=1/Y), the well and the posts kept as separate materials."
    return calls, notes, {"count": count, "mat": MATERIALS[post_mk][3]}


PROMPTS_WELL_RING_POSTS = [
    lambda r, c: f"a courtyard well surrounded by a ring of {c['count']} support posts",
    lambda r, c: f"a well surrounded by a ring of {c['count']} {c['mat']} support posts",
    lambda r, c: f"well with {c['count']} posts around it in a ring",
    lambda r, c: f"well, {c['count']} posts around it, {c['mat']}",
    lambda r, c: f"a courtyard well ringed by {c['count']} evenly spaced {c['mat']} support posts",
    lambda r, c: f"a modern well surrounded by a ring of {c['count']} steel support posts",
]
register("well_ring_posts", _b_well_ring_posts, PROMPTS_WELL_RING_POSTS)


# 45. a lighthouse with a ring of support struts arranged radially near its base
def _b_lighthouse_struts(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = rng.choice([5, 6, 7])
    scale = rng.choice([0.9, 1.0, 1.15])
    tower_r = round(1.2 * scale, 2)
    tower_h = round(7.0 * scale, 2)
    strut_len = round(1.4 * scale, 2)
    strut_w = round(0.12 * scale, 3)
    mk = rng.choice(["marble_white", "weathered_stone", "concrete"])
    strut_mk = rng.choice(["iron_dark", "bronze", "steel_scifi"])
    style = v
    tower = nm("lighthouse", style)
    strut = nm("strut", style)
    strut_r = tower_r + strut_len / 2
    calls = [
        prim(
            "cone",
            tower,
            params={"radius": tower_r, "height": tower_h, "segments": 24},
            translation=[0, tower_h / 2, 0],
        ),
        material_call([tower], mk, "Lighthouse Wall"),
        prim(
            "box",
            strut,
            params={"size": [strut_len, strut_w, strut_w]},
            translation=[strut_r, 0.6 * scale, 0],
            rotation=[0, 0, -20],
        ),
        material_call([strut], strut_mk, "Strut Metal"),
        *array_radial(strut, count=count, angle=360.0, axis=1.0),
    ]
    notes = f"a tapering cone tower with one angled strut box array-radial'd (axis=1/Y) into a ring of {count} near the base."
    return calls, notes, {"count": count, "mat": MATERIALS[strut_mk][3]}


PROMPTS_LIGHTHOUSE_STRUTS = [
    lambda r, c: "a lighthouse with a ring of support struts arranged radially near its base",
    lambda r, c: (
        f"a lighthouse with {c['count']} {c['mat']} support struts arranged radially near the base"
    ),
    lambda r, c: f"lighthouse with {c['count']} struts around the bottom",
    lambda r, c: f"lighthouse struts, {c['mat']}, {c['count']} of them",
    lambda r, c: (
        f"a lighthouse tower ringed near its base by {c['count']} radially arranged {c['mat']} support struts"
    ),
    lambda r, c: f"a sci-fi beacon tower with {c['count']} radial support struts near its base",
]
register("lighthouse_struts", _b_lighthouse_struts, PROMPTS_LIGHTHOUSE_STRUTS)


# =============================================================================
# HARD templates (seeds 46-50) -- sweep/tube, taper and twist along a path
# =============================================================================


# 46. a tower whose walls taper smoothly from a wide base to a narrow spire
def _b_tapering_tower(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.9, 1.0, 1.15])
    depth = round(7.0 * scale, 2)
    half = round(1.4 * scale, 2)
    taper = round(0.15, 3)
    mk = rng.choice(["weathered_stone", "granite", "concrete"])
    style = v
    name = nm("tapering_tower", style)
    outline = [[half, -half], [half, half], [-half, half], [-half, -half]]
    calls = [
        prim(
            "sweep",
            name,
            params={
                "outline": outline,
                "depth": depth,
                "taper": taper,
                "twist": 0.0,
                "sections": 12,
            },
            rotation=[-90, 0, 0],
            translation=[0, depth / 2, 0],
        ),
        material_call([name], mk, "Tower Stone"),
    ]
    notes = (
        "a square sweep outline extruded along its local Z, rotated -90 about X so that "
        "local Z becomes world Y, with taper<1 shrinking the far (+Z, now the top) end to a "
        "narrow spire; 12 sections so the taper reads as a smooth batter rather than one flat "
        "slope."
    )
    return calls, notes, {"h": depth, "mat": MATERIALS[mk][3]}


PROMPTS_TAPERING_TOWER = [
    lambda r, c: "a tower whose walls taper smoothly from a wide base to a narrow spire",
    lambda r, c: f"a {c['mat']} tower, {fmt_m(c['h'])} m tall, tapering smoothly to a narrow spire",
    lambda r, c: "tower that tapers from wide at the bottom to narrow at the top",
    lambda r, c: f"tapering tower, {c['mat']}",
    lambda r, c: (
        f"a {c['mat']} tower whose square walls taper smoothly from a wide base up to a narrow spire"
    ),
    lambda r, c: "a modern concrete tower tapering smoothly from a wide base to a narrow top",
]
register("tapering_tower", _b_tapering_tower, PROMPTS_TAPERING_TOWER)


# 47. a stone archway whose profile sweeps in a continuous curve from post to post
def _b_archway_curve_sweep(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.9, 1.0, 1.15])
    span = round(2.4 * scale, 2)
    rise = round(1.2 * scale, 2)
    r = round(0.16 * scale, 3)
    mk = rng.choice(["weathered_stone", "granite", "sandstone"])
    style = v
    name = nm("swept_archway", style)
    path = [
        [-span / 2, 0.0, 0.0],
        [-span / 2, rise * 0.55, 0.0],
        [-span * 0.2, rise, 0.0],
        [span * 0.2, rise, 0.0],
        [span / 2, rise * 0.55, 0.0],
        [span / 2, 0.0, 0.0],
    ]
    calls = [
        prim(
            "tube",
            name,
            params={"path": path, "radius": r, "sides": 12},
            translation=[0, rise / 2, 0],
        ),
        material_call([name], mk, "Archway Stone"),
    ]
    notes = (
        "a tube generator's own path is the continuous curve: six stations rising from one "
        "post, over a flat crown, and back down to the other, all in the XY plane at z=0, so "
        "the swept circular cross-section reads as one unbroken archway profile rather than a "
        "faceted arc. clay_add_primitive's own _clamp_path re-centres the path's bounding box "
        "on its own centre first, so the grounding offset is half the path's own y-extent "
        "(rise/2), not the tube's radius."
    )
    return calls, notes, {"span": span, "mat": MATERIALS[mk][3]}


PROMPTS_ARCHWAY_CURVE_SWEEP = [
    lambda r, c: "a stone archway whose profile sweeps in a continuous curve from post to post",
    lambda r, c: (
        f"a {c['mat']} archway, {fmt_m(c['span'])} m across, a continuous curved sweep from post to post"
    ),
    lambda r, c: "archway whose curve sweeps continuously from one post to the other",
    lambda r, c: f"continuously curved archway, {c['mat']}",
    lambda r, c: (
        f"a {c['mat']} archway whose profile sweeps in one continuous curve from post to post"
    ),
    lambda r, c: "a modern archway with a continuously curved swept profile from post to post",
]
register("archway_curve_sweep", _b_archway_curve_sweep, PROMPTS_ARCHWAY_CURVE_SWEEP)


# 48. a spire that tapers along a gentle twisting sweep from base to tip
def _b_twisting_spire(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.9, 1.0, 1.15])
    depth = round(6.0 * scale, 2)
    half = round(0.7 * scale, 2)
    twist = round(70.0, 1)
    mk = rng.choice(["slate", "dark_basalt", "verdigris_copper"])
    style = v
    name = nm("twisting_spire", style)
    outline = [[half, 0.0], [0.0, half], [-half, 0.0], [0.0, -half]]
    calls = [
        prim(
            "sweep",
            name,
            params={
                "outline": outline,
                "depth": depth,
                "taper": 0.08,
                "twist": twist,
                "sections": 16,
            },
            rotation=[-90, 0, 0],
            translation=[0, depth / 2, 0],
        ),
        material_call([name], mk, "Spire Metal"),
    ]
    notes = (
        "a diamond sweep outline, vertical (rotated -90 about X), tapering hard (0.08) to "
        "almost a point at the tip while twisting a gentle 70 degrees across 16 sections -- "
        "the twist is the shape itself and is not corrected back to centre, per sweep's own "
        "docstring."
    )
    return calls, notes, {"h": depth, "mat": MATERIALS[mk][3]}


PROMPTS_TWISTING_SPIRE = [
    lambda r, c: "a spire that tapers along a gentle twisting sweep from base to tip",
    lambda r, c: (
        f"a {c['mat']} spire, {fmt_m(c['h'])} m tall, tapering along a gentle twist from base to tip"
    ),
    lambda r, c: "spire that twists gently as it tapers up to the tip",
    lambda r, c: f"twisting spire, {c['mat']}",
    lambda r, c: (
        f"a {c['mat']} spire tapering along a gentle twisting sweep from its base to its tip"
    ),
    lambda r, c: "a sci-fi antenna spire, tapering along a gentle twisting sweep to its tip",
]
register("twisting_spire", _b_twisting_spire, PROMPTS_TWISTING_SPIRE)


# 49. a flying buttress whose arm sweeps outward and tapers to a point
def _b_flying_buttress(rng: random.Random, v: int) -> tuple[list, str, dict]:
    import math

    scale = rng.choice([0.9, 1.0, 1.15])
    depth = round(3.2 * scale, 2)
    half = round(0.35 * scale, 3)
    mk = rng.choice(["weathered_stone", "granite", "sandstone"])
    style = v
    pier_w = half * 2.4
    pier_h = round(2.6 * scale, 2)
    pier = nm("pier", style)
    calls = [
        prim(
            "box", pier, params={"size": [pier_w, pier_h, pier_w]}, translation=[0, pier_h / 2, 0]
        ),
        material_call([pier], mk, "Pier Stone"),
    ]
    # rotation=[0, 90, tilt_deg] carries a sweep's own local Z (its extrusion
    # axis) to the world direction (cos(tilt), sin(tilt), 0) -- verified
    # numerically against agent_clay._quat_from_euler_xyz rather than derived
    # by hand, since a Z-only rotation (the first attempt here) never moves
    # the local Z axis at all and left the arm pointing straight up with no
    # outward sweep whatsoever.
    tilt_deg = 50.0
    tilt = math.radians(tilt_deg)
    direction = (math.cos(tilt), math.sin(tilt))
    attach = (pier_w / 2, pier_h)
    arm_center = (attach[0] + direction[0] * depth / 2, attach[1] + direction[1] * depth / 2)
    arm = nm("buttress_arm", style)
    outline = [[half, -half], [half, half], [-half, half], [-half, -half]]
    calls.append(
        prim(
            "sweep",
            arm,
            params={"outline": outline, "depth": depth, "taper": 0.05, "twist": 0.0, "sections": 8},
            rotation=[0, 90, tilt_deg],
            translation=[arm_center[0], arm_center[1], 0],
        )
    )
    calls.append(material_call([arm], mk, "Buttress Stone"))
    notes = (
        f"a square sweep outline extruded along local Z, carried by rotation=[0, 90, "
        f"{tilt_deg}] to point {tilt_deg} degrees up from horizontal in the XY plane -- a "
        "rotation about Z alone never moves the local Z axis at all, so the arm needs the Y "
        "rotation to swing it out of vertical first; its near end (local z=-depth/2) is placed "
        "exactly at the pier's own top outer corner, and taper=0.05 shrinks the far end to "
        "nearly a point."
    )
    return calls, notes, {"mat": MATERIALS[mk][3]}


PROMPTS_FLYING_BUTTRESS = [
    lambda r, c: "a flying buttress whose arm sweeps outward and tapers to a point",
    lambda r, c: f"a {c['mat']} flying buttress, its arm sweeping outward and tapering to a point",
    lambda r, c: "flying buttress arm that sweeps out and comes to a point",
    lambda r, c: f"flying buttress, {c['mat']}, tapered arm",
    lambda r, c: (
        f"a {c['mat']} flying buttress whose supporting arm sweeps outward from a pier and tapers to a point"
    ),
    lambda r, c: "a modern concrete flying-buttress arm sweeping outward and tapering to a point",
]
register("flying_buttress", _b_flying_buttress, PROMPTS_FLYING_BUTTRESS)


# 50. a minaret that tapers gradually along its swept vertical profile
def _b_minaret_swept(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.9, 1.0, 1.15])
    depth = round(7.0 * scale, 2)
    half = round(0.55 * scale, 3)
    mk = rng.choice(["sandstone", "limestone", "marble_white"])
    cap_mk = rng.choice(["gilded", "verdigris_copper", "bronze"])
    style = v
    shaft = nm("minaret_shaft", style)
    n = 10
    outline = [
        [
            half * __import__("math").cos(2 * __import__("math").pi * i / n),
            half * __import__("math").sin(2 * __import__("math").pi * i / n),
        ]
        for i in range(n)
    ]
    calls = [
        prim(
            "sweep",
            shaft,
            params={
                "outline": outline,
                "depth": depth,
                "taper": 0.35,
                "twist": 0.0,
                "sections": 10,
            },
            rotation=[-90, 0, 0],
            translation=[0, depth / 2, 0],
        ),
        material_call([shaft], mk, "Minaret Stone"),
    ]
    cap = nm("minaret_cap", style)
    cap_h = round(0.9 * scale, 2)
    calls.append(
        prim(
            "cone",
            cap,
            params={"radius": half * 0.4, "height": cap_h, "segments": n},
            translation=[0, depth + cap_h / 2, 0],
        )
    )
    calls.append(material_call([cap], cap_mk, "Minaret Cap"))
    notes = (
        f"a {n}-sided near-round sweep outline, vertical (rotated -90 about X), tapering "
        "gradually (0.35) over 10 sections along its own swept profile, with a small cone cap "
        "set on top at the shaft's own full height."
    )
    return calls, notes, {"h": depth, "mat": MATERIALS[mk][3]}


PROMPTS_MINARET = [
    lambda r, c: "a minaret that tapers gradually along its swept vertical profile",
    lambda r, c: (
        f"a {c['mat']} minaret, {fmt_m(c['h'])} m tall, tapering gradually along its swept vertical profile"
    ),
    lambda r, c: "minaret that tapers gradually as it goes up",
    lambda r, c: f"minaret, {c['mat']}, gradual taper",
    lambda r, c: (
        f"a {c['mat']} minaret that tapers gradually along its own swept vertical profile, capped at the top"
    ),
    lambda r, c: "a modern minaret-like tower tapering gradually along its swept vertical profile",
]
register("minaret_swept", _b_minaret_swept, PROMPTS_MINARET)


# =============================================================================
# emit
# =============================================================================


def main() -> None:
    records = []
    idx = 1
    for template in TEMPLATES:
        for v in range(6):
            rng = random.Random(f"{FAMILY}-{template.key}-{v}")
            calls, notes, ctx = template.build(rng, v)
            prompt = template.prompts[v](rng, ctx)
            rec_id = f"{FAMILY}-{idx:04d}"
            idx += 1
            record = {
                "id": rec_id,
                "family": FAMILY,
                "kind": "build",
                "prompt": prompt,
                "calls": calls,
                "notes": notes,
            }
            records.append(record)

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
