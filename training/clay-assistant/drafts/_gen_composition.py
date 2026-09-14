"""Emits ``drafts/composition.jsonl`` for the ``composition`` family -- rings
of repeated posts/stones/lamps/candles built with ``array-radial``, tripods
and splayed-leg stands whose per-leg rotation is solved with trigonometry
rather than guessed, and stacks of things sitting on other things whose
every upper part's height is computed from the part beneath it. Deterministic
and seeded: rerunning this script byte-for-byte reproduces the same JSONL.

Why this family exists (see ``we-are-going-to-cosmic-dream.md`` and run A's
own eval writeup): the held-out "colonnade of eight columns" came back as a
slotted block -- the model placed its columns inside a base the same size as
the ring -- and the held-out telescope stand fell over because its three legs
were named inconsistently mid-build. Every ring build here puts its base or
platform *visibly past* the outer edge of the ring, and every leg in a
tripod/splayed-leg build is named ``leg_1``, ``leg_2``... consistently
through its own record.

Held-out rule (``training/clay-assistant/seeds/composition.txt``'s own
header, and ``gen/holdout.py::is_leak``): no subject here is a chair, a
mechanical bracket, a telescope, a colonnade of eight columns, or a
serpentine creature, and none of the words "fluted", "colonnade", "telescope"
or the phrases "round base"/"three-legged tripod" appears anywhere below --
nor does any ring here take a count of eight.

Run directly to (re)write the JSONL:

    uv run python training/clay-assistant/drafts/_gen_composition.py
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

FAMILY = "composition"
OUT_PATH = Path(__file__).resolve().parent / f"{FAMILY}.jsonl"
PHRASING_COUNT = 4

# --- naming variety --------------------------------------------------------


def nm(base: str, style: int) -> str:
    """*base* (``snake_case`` or ``hyphen-case``) rendered in one of four
    naming conventions, cycled by *style* -- the same four this programme's
    other families use, so a naming convention is never repeated twice in a
    row across one record's own phrasings."""
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
    "granite": ((0.50, 0.50, 0.52), 0.60, 0.0, "polished grey granite"),
    "dark_basalt": ((0.16, 0.16, 0.18), 0.70, 0.0, "dark basalt"),
    "marble_white": ((0.90, 0.89, 0.86), 0.30, 0.0, "pale marble"),
    "concrete": ((0.62, 0.62, 0.60), 0.90, 0.0, "raw concrete"),
    "moss_stone": ((0.40, 0.46, 0.36), 0.85, 0.0, "moss-covered stone"),
    "packed_earth": ((0.36, 0.28, 0.20), 0.95, 0.0, "packed earth"),
    "oak_wood": ((0.45, 0.31, 0.18), 0.75, 0.0, "oiled oak"),
    "weathered_wood": ((0.33, 0.24, 0.15), 0.85, 0.0, "weathered timber"),
    "iron_dark": ((0.22, 0.22, 0.24), 0.40, 0.85, "blackened iron"),
    "bronze": ((0.55, 0.40, 0.20), 0.35, 0.90, "polished bronze"),
    "brass": ((0.71, 0.55, 0.24), 0.35, 0.85, "polished brass"),
    "steel_scifi": ((0.60, 0.63, 0.68), 0.30, 0.70, "brushed steel"),
    "verdigris_copper": ((0.35, 0.55, 0.48), 0.60, 0.20, "verdigris-streaked copper"),
    "gilded": ((0.83, 0.68, 0.21), 0.30, 0.85, "gilded bronze"),
    "terracotta": ((0.70, 0.35, 0.22), 0.80, 0.0, "terracotta"),
    "black_plastic": ((0.05, 0.05, 0.06), 0.40, 0.0, "matte black plastic"),
    "canvas_cream": ((0.88, 0.82, 0.68), 0.90, 0.0, "cream canvas"),
    "candle_wax": ((0.93, 0.90, 0.78), 0.55, 0.0, "pale wax"),
    "flame_amber": ((0.95, 0.55, 0.10), 0.50, 0.0, "amber flame glass"),
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


def array_radial(
    name: str, *, count: int, angle: float = 360.0, axis: float = 1.0
) -> list[dict[str, Any]]:
    return [
        select_call([name]),
        op_call("array-radial", {"count": float(count), "angle": angle, "axis": axis}),
    ]


def fmt_m(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def base_radius_for(outer_edge: float) -> float:
    """A base/platform radius *visibly* past a ring's own outer edge --
    run A's held-out colonnade came back with the base sized the same as the
    ring, so this is never a small nudge: at least 30% of the outer edge
    itself, and never less than 0.3 m, whichever is larger."""
    return round(outer_edge + max(0.3, 0.30 * outer_edge), 3)


def splayed_legs(
    *,
    n: int,
    r_top: float,
    r_foot: float,
    h_top: float,
    style: int,
    name_prefix: str = "leg",
    generator: str = "cylinder",
    radius: float = 0.02,
    box_size: float = 0.04,
    segments: int = 10,
) -> tuple[list[dict[str, Any]], list[str], float]:
    """*n* legs, each solved by trigonometry rather than guessed: every leg's
    own top lands exactly on the ring of radius ``r_top`` at height ``h_top``
    (0 for a converging apex, like a camera stand) and every foot lands
    exactly on the ground ring of radius ``r_foot`` at y=0.

    A leg is a centred primitive along local Y (a cylinder or box), so it
    needs one rotation (about local X, before the azimuth spin) that tilts
    its own axis to match the line from foot to top, plus an azimuth spin
    about Y -- this document's own Euler order is intrinsic X, then Y, then
    Z (``agent_clay._quat_from_euler_xyz``), so composing exactly those two,
    with no Z term, is enough:

    * ``L = hypot(r_top - r_foot, h_top)`` is the leg's own length.
    * ``alpha = atan2(r_top - r_foot, h_top)`` is the tilt about local X that
      carries the local +Y axis to the true foot-to-top direction (solved
      once with the leg sitting at azimuth 0, along +X/+Z).
    * the azimuth itself is the object's own Y rotation, ``phi = i * 360/n``.
    * the object's translation is the midpoint between the true top point
      ``(r_top*sin phi, h_top, r_top*cos phi)`` and the true foot point
      ``(r_foot*sin phi, 0, r_foot*cos phi)``.

    Every leg is a plain ``clay_add_primitive`` call -- no ``array-radial``
    here, because each leg's own *rotation* differs (a radial array shares
    one seed's rotation), and this needs each one solved for its own
    azimuth. ``r_top=0`` converges every leg on the world Y axis at
    ``h_top`` -- a true apex, the shape a camera stand's own mount wants.
    """
    length = round(math.hypot(r_top - r_foot, h_top), 4)
    alpha = math.degrees(math.atan2(r_top - r_foot, h_top))
    calls: list[dict[str, Any]] = []
    names: list[str] = []
    for i in range(n):
        phi = i * 360.0 / n
        phi_r = math.radians(phi)
        tx = (r_top + r_foot) / 2 * math.sin(phi_r)
        tz = (r_top + r_foot) / 2 * math.cos(phi_r)
        ty = h_top / 2
        name = nm(f"{name_prefix}_{i + 1}", style)
        if generator == "box":
            params: dict[str, Any] = {"size": [box_size, length, box_size]}
        else:
            params = {"radius": radius, "height": length, "segments": segments}
        calls.append(
            prim(
                generator, name, params=params, translation=[tx, ty, tz], rotation=[alpha, phi, 0.0]
            )
        )
        names.append(name)
    return calls, names, length


def stack_y(base_top: float, height: float) -> tuple[float, float]:
    """``(translation.y, new top y)`` for a centred primitive of *height*
    resting exactly on a surface at ``base_top`` -- every "thing on top of
    thing" build in this file chains this rather than guessing a Y."""
    return round(base_top + height / 2, 4), round(base_top + height, 4)


# --- record plumbing ---------------------------------------------------------

Prompter = Callable[[random.Random, dict[str, Any]], str]


class Template:
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
    assert len(prompts) == PHRASING_COUNT, (
        f"{key} needs exactly {PHRASING_COUNT} phrasings, got {len(prompts)}"
    )
    TEMPLATES.append(Template(key, build, prompts))


# =============================================================================
# RING kind -- array-radial on one seed, base always visibly larger than the
# ring. N in 3..14, excluding 8, spread across the ten builds below.
# =============================================================================


# 1. a ring of standing stones on a wide earthen mound (N=3)
def _b_ring_standing_stones(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = 3
    scale = rng.choice([0.9, 1.0, 1.15])
    ring_r = round(1.5 * scale, 3)
    w = round(0.32 * scale, 3)
    h = round(1.5 * scale, 2)
    d = round(0.2 * scale, 3)
    mk = rng.choice(["weathered_stone", "granite", "moss_stone"])
    base_mk = "packed_earth"
    style = v
    stone = nm("stone", style)
    outer_edge = ring_r + max(w, d) / 2
    base_r = base_radius_for(outer_edge)
    base_h = round(0.25 * scale, 3)
    base = nm("mound", style)
    calls = [
        prim(
            "cylinder",
            base,
            params={"radius": base_r, "height": base_h, "segments": 28},
            translation=[0, base_h / 2, 0],
        ),
        material_call([base], base_mk, "Mound Earth"),
        prim(
            "box",
            stone,
            params={"size": [w, h, d]},
            translation=[ring_r, base_h + h / 2, 0],
        ),
        material_call([stone], mk, "Stone"),
        *array_radial(stone, count=count, angle=360.0, axis=1.0),
    ]
    notes = (
        f"one upright slab, materialed, then array-radial ({count} copies) about the world Y "
        f"axis so the stones land evenly around a circle of radius {ring_r} m -- the earthen "
        f"mound underneath is a separate cylinder of radius {base_r} m, well past the stones' "
        f"own outer edge ({round(outer_edge, 3)} m), not sized to the ring."
    )
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3], "r": ring_r}


PROMPTS_RING_STONES = [
    lambda r, c: "a ring of standing stones on a wide earthen mound",
    lambda r, c: (
        f"{c['count']} {c['mat']} standing stones set in a circle on a broad earthen mound"
    ),
    lambda r, c: "three tall stones standing in a ring atop a low mound of packed earth",
    lambda r, c: f"stone circle, {c['count']} slabs, {c['mat']}, raised mound underneath",
]
register("ring_standing_stones", _b_ring_standing_stones, PROMPTS_RING_STONES)


# 2. a ring of four square wooden posts on a round platform (N=4)
def _b_ring_square_posts(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = 4
    scale = rng.choice([0.9, 1.0, 1.2])
    ring_r = round(1.1 * scale, 3)
    w = round(0.16 * scale, 3)
    h = round(1.3 * scale, 2)
    mk = rng.choice(["oak_wood", "weathered_wood"])
    base_mk = rng.choice(["weathered_stone", "concrete"])
    style = v
    post = nm("post", style)
    outer_edge = ring_r + w / 2
    base_r = base_radius_for(outer_edge)
    base_h = round(0.12 * scale, 3)
    base = nm("platform", style)
    calls = [
        prim(
            "cylinder",
            base,
            params={"radius": base_r, "height": base_h, "segments": 28},
            translation=[0, base_h / 2, 0],
        ),
        material_call([base], base_mk, "Platform Stone"),
        prim(
            "box",
            post,
            params={"size": [w, h, w]},
            translation=[ring_r, base_h + h / 2, 0],
        ),
        material_call([post], mk, "Post Wood"),
        *array_radial(post, count=count, angle=360.0, axis=1.0),
    ]
    notes = (
        f"one square post, materialed, array-radial ({count} copies) about world Y at radius "
        f"{ring_r} m -- the platform disc underneath (radius {base_r} m) clears the posts' own "
        f"outer edge ({round(outer_edge, 3)} m) with a visible margin, not flush with them."
    )
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3], "r": ring_r}


PROMPTS_RING_POSTS = [
    lambda r, c: "a ring of four square wooden posts on a round platform",
    lambda r, c: f"{c['count']} square {c['mat']} posts set in a circle on a raised platform",
    lambda r, c: "four wooden posts standing in a ring, a wide platform beneath them",
    lambda r, c: f"post circle: {c['count']} square {c['mat']} timbers on a broad disc platform",
]
register("ring_square_posts", _b_ring_square_posts, PROMPTS_RING_POSTS)


# 3. five stone bollards in a ring on a broad plaza floor (N=5)
def _b_ring_bollards(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = 5
    scale = rng.choice([0.9, 1.0, 1.15])
    ring_r = round(1.6 * scale, 3)
    radius = round(0.11 * scale, 3)
    cyl_h = round(0.55 * scale, 3)
    mk = rng.choice(["granite", "weathered_stone", "dark_basalt"])
    base_mk = "concrete"
    style = v
    bol = nm("bollard", style)
    outer_edge = ring_r + radius
    base_r = base_radius_for(outer_edge)
    base_h = round(0.1 * scale, 3)
    base = nm("plaza_floor", style)
    calls = [
        prim(
            "cylinder",
            base,
            params={"radius": base_r, "height": base_h, "segments": 32},
            translation=[0, base_h / 2, 0],
        ),
        material_call([base], base_mk, "Plaza Concrete"),
        prim(
            "capsule",
            bol,
            params={"radius": radius, "height": cyl_h, "segments": 16, "rings": 4},
            translation=[ring_r, base_h + cyl_h / 2 + radius, 0],
        ),
        material_call([bol], mk, "Bollard Stone"),
        *array_radial(bol, count=count, angle=360.0, axis=1.0),
    ]
    notes = (
        f"a capsule (a domed-top bollard, whole extent = height + 2*radius) at radius {ring_r} "
        f"m, materialed, then array-radial ({count} copies); the plaza floor disc (radius "
        f"{base_r} m) reaches well past the bollards' own outer edge ({round(outer_edge, 3)} m)."
    )
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3], "r": ring_r}


PROMPTS_RING_BOLLARDS = [
    lambda r, c: "five stone bollards arranged in a ring on a broad plaza floor",
    lambda r, c: f"{c['count']} domed {c['mat']} bollards set in a circle on a wide plaza floor",
    lambda r, c: "a ring of squat stone bollards standing on a broad paved floor",
    lambda r, c: f"bollard ring, {c['count']} of them, {c['mat']}, plaza floor beneath",
]
register("ring_bollards", _b_ring_bollards, PROMPTS_RING_BOLLARDS)


# 4. a circular stake fence of six posts around a wide mound (N=6, two-part seed)
def _b_ring_stake_fence(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = 6
    scale = rng.choice([0.9, 1.0, 1.15])
    ring_r = round(1.4 * scale, 3)
    radius = round(0.035 * scale, 4)
    body_h = round(0.9 * scale, 3)
    tip_h = round(0.15 * scale, 3)
    mk = rng.choice(["weathered_wood", "oak_wood"])
    base_mk = "packed_earth"
    style = v
    body = nm("stake_body", style)
    tip = nm("stake_tip", style)
    outer_edge = ring_r + radius
    base_r = base_radius_for(outer_edge)
    base_h = round(0.15 * scale, 3)
    base = nm("mound", style)
    calls = [
        prim(
            "cylinder",
            base,
            params={"radius": base_r, "height": base_h, "segments": 28},
            translation=[0, base_h / 2, 0],
        ),
        material_call([base], base_mk, "Mound Earth"),
        prim(
            "cylinder",
            body,
            params={"radius": radius, "height": body_h, "segments": 10},
            translation=[ring_r, base_h + body_h / 2, 0],
        ),
        prim(
            "cone",
            tip,
            params={"radius": radius, "height": tip_h, "segments": 10},
            translation=[ring_r, base_h + body_h + tip_h / 2, 0],
        ),
        material_call([body, tip], mk, "Stake Wood"),
        *array_radial(body, count=count, angle=360.0, axis=1.0),
        *array_radial(tip, count=count, angle=360.0, axis=1.0),
    ]
    notes = (
        f"a two-part seed (a thin cylinder body plus a small cone point, same radius and same "
        f"ring position so they line up) materialed together, then array-radial run twice at "
        f"the same count/angle/axis so every duplicated tip sits on its own duplicated body -- "
        f"the mound (radius {base_r} m) is well past the stakes' own outer edge "
        f"({round(outer_edge, 3)} m)."
    )
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3], "r": ring_r}


PROMPTS_RING_STAKES = [
    lambda r, c: "a circular stake fence of six posts around a wide mound",
    lambda r, c: f"{c['count']} sharpened {c['mat']} stakes set in a circle around a raised mound",
    lambda r, c: "a ring of pointed wooden stakes fencing a low earthen mound",
    lambda r, c: f"stake circle, {c['count']} posts, {c['mat']}, mound inside the ring",
]
register("ring_stake_fence", _b_ring_stake_fence, PROMPTS_RING_STAKES)


# 5. seven garden lamp posts in a ring around a plaza (N=7, two-part seed)
def _b_ring_lamps(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = 7
    scale = rng.choice([0.9, 1.0, 1.15])
    ring_r = round(1.8 * scale, 3)
    pole_r = round(0.03 * scale, 4)
    pole_h = round(1.4 * scale, 3)
    head_r = round(0.09 * scale, 3)
    pole_mk = rng.choice(["iron_dark", "bronze"])
    base_mk = "concrete"
    style = v
    pole = nm("lamp_post", style)
    head = nm("lamp_head", style)
    outer_edge = ring_r + pole_r
    base_r = base_radius_for(outer_edge)
    base_h = round(0.1 * scale, 3)
    base = nm("plaza_floor", style)
    calls = [
        prim(
            "cylinder",
            base,
            params={"radius": base_r, "height": base_h, "segments": 32},
            translation=[0, base_h / 2, 0],
        ),
        material_call([base], base_mk, "Plaza Concrete"),
        prim(
            "cylinder",
            pole,
            params={"radius": pole_r, "height": pole_h, "segments": 12},
            translation=[ring_r, base_h + pole_h / 2, 0],
        ),
        material_call([pole], pole_mk, "Lamp Post Metal"),
        prim(
            "uv_sphere",
            head,
            params={"radius": head_r, "segments": 14, "rings": 10},
            translation=[ring_r, base_h + pole_h + head_r, 0],
        ),
        material_call([head], "flame_amber", "Lamp Glass"),
        *array_radial(pole, count=count, angle=360.0, axis=1.0),
        *array_radial(head, count=count, angle=360.0, axis=1.0),
    ]
    notes = (
        f"a two-part seed (post plus a glass-coloured sphere head, both materialed first, "
        f"both at the same ring radius) duplicated with two array-radial calls at the same "
        f"count/angle/axis, so every head lands on its own post; the plaza floor (radius "
        f"{base_r} m) sits well past the posts' own outer edge ({round(outer_edge, 3)} m)."
    )
    return calls, notes, {"count": count, "mat": MATERIALS[pole_mk][3], "r": ring_r}


PROMPTS_RING_LAMPS = [
    lambda r, c: "seven garden lamp posts arranged in a ring around a plaza",
    lambda r, c: f"{c['count']} {c['mat']} lamp posts set in a circle around a paved plaza",
    lambda r, c: "a ring of lamp posts standing around the edge of a small plaza",
    lambda r, c: f"lamp ring, {c['count']} posts, {c['mat']}, plaza floor beneath",
]
register("ring_lamps", _b_ring_lamps, PROMPTS_RING_LAMPS)


# 6. nine stone pillars in a ring, capped with a ring beam, on a broad base (N=9)
def _b_ring_pillars_capped(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = 9
    scale = rng.choice([0.9, 1.0, 1.15])
    ring_r = round(2.4 * scale, 3)
    radius = round(0.22 * scale, 3)
    height = round(2.4 * scale, 2)
    mk = rng.choice(["weathered_stone", "sandstone", "marble_white"])
    base_mk = "granite"
    style = v
    col = nm("pillar", style)
    outer_edge = ring_r + radius
    base_r = base_radius_for(outer_edge)
    base_h = round(0.3 * scale, 3)
    base = nm("plinth_floor", style)
    ring_beam = nm("ring_beam", style)
    tube = round(0.16 * scale, 3)
    calls = [
        prim(
            "cylinder",
            base,
            params={"radius": base_r, "height": base_h, "segments": 36},
            translation=[0, base_h / 2, 0],
        ),
        material_call([base], base_mk, "Base Granite"),
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
            translation=[ring_r, base_h + height / 2, 0],
        ),
        material_call([col], mk, "Pillar Stone"),
        *array_radial(col, count=count, angle=360.0, axis=1.0),
        prim(
            "torus",
            ring_beam,
            params={"radius": ring_r, "tube": tube, "segments": 40, "sides": 12},
            translation=[0, base_h + height + tube, 0],
        ),
        material_call([ring_beam], mk, "Ring Beam Stone"),
    ]
    notes = (
        f"a column ring (never eight -- {count} here), materialed and array-radial'd, sitting "
        f"on a base ({base_r} m radius) well past the pillars' own outer edge "
        f"({round(outer_edge, 3)} m). The ring beam on top is one torus, lying flat in its "
        f"own natural orientation, its major radius matched to the pillar ring ({ring_r} m) "
        f"and its own y computed from the pillars' own height: "
        f"base_h + height + tube = {round(base_h + height + tube, 4)} m."
    )
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3], "r": ring_r}


PROMPTS_RING_PILLARS_CAPPED = [
    lambda r, c: "nine stone pillars in a ring, capped with a ring beam, on a broad base",
    lambda r, c: (
        f"{c['count']} {c['mat']} pillars in a circle under a stone ring beam, on a broad base"
    ),
    lambda r, c: (
        "a ring of pillars carrying a single circular beam on top, standing on a wide base"
    ),
    lambda r, c: f"pillar ring: {c['count']} columns, ring beam above, {c['mat']}, broad base",
]
register("ring_pillars_capped", _b_ring_pillars_capped, PROMPTS_RING_PILLARS_CAPPED)


# 7. ten square pillars in a ring under a circular lintel, on a wide platform (N=10)
def _b_ring_square_pillars_capped(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = 10
    scale = rng.choice([0.9, 1.0, 1.15])
    ring_r = round(2.1 * scale, 3)
    w = round(0.3 * scale, 3)
    height = round(2.0 * scale, 2)
    mk = rng.choice(["concrete", "granite", "dark_basalt"])
    style = v
    col = nm("pillar", style)
    outer_edge = ring_r + w / 2
    base_r = base_radius_for(outer_edge)
    base_h = round(0.25 * scale, 3)
    base = nm("platform", style)
    ring_beam = nm("lintel_ring", style)
    tube = round(0.14 * scale, 3)
    calls = [
        prim(
            "cylinder",
            base,
            params={"radius": base_r, "height": base_h, "segments": 36},
            translation=[0, base_h / 2, 0],
        ),
        material_call([base], mk, "Platform Stone"),
        prim(
            "box",
            col,
            params={"size": [w, height, w]},
            translation=[ring_r, base_h + height / 2, 0],
        ),
        material_call([col], mk, "Pillar Stone"),
        *array_radial(col, count=count, angle=360.0, axis=1.0),
        prim(
            "torus",
            ring_beam,
            params={"radius": ring_r, "tube": tube, "segments": 44, "sides": 12},
            translation=[0, base_h + height + tube, 0],
        ),
        material_call([ring_beam], mk, "Lintel Ring Stone"),
    ]
    notes = (
        f"an open pavilion frame: {count} square pillars array-radial'd around a platform "
        f"(radius {base_r} m, well past the pillars' own outer edge {round(outer_edge, 3)} m), "
        f"a single torus resting on top as the circular lintel, its y computed from the "
        f"pillars' own height the same way the previous build's ring beam is."
    )
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3], "r": ring_r}


PROMPTS_RING_SQUARE_PILLARS_CAPPED = [
    lambda r, c: "ten square pillars in a ring under a circular lintel, on a wide platform",
    lambda r, c: (
        f"{c['count']} square {c['mat']} pillars in a circle, a lintel ring on top, wide platform"
    ),
    lambda r, c: (
        "an open ring of square pillars carrying one circular lintel above, on a broad platform"
    ),
    lambda r, c: f"pillar frame: {c['count']} square posts, ring lintel, {c['mat']}, wide platform",
]
register(
    "ring_square_pillars_capped", _b_ring_square_pillars_capped, PROMPTS_RING_SQUARE_PILLARS_CAPPED
)


# 8. twelve bronze bollards in a ring on a broad plaza floor (N=12)
def _b_ring_bollards_bronze(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = 12
    scale = rng.choice([0.9, 1.0, 1.15])
    ring_r = round(2.6 * scale, 3)
    radius = round(0.09 * scale, 3)
    cyl_h = round(0.45 * scale, 3)
    mk = "bronze"
    base_mk = rng.choice(["marble_white", "granite"])
    style = v
    bol = nm("bollard", style)
    outer_edge = ring_r + radius
    base_r = base_radius_for(outer_edge)
    base_h = round(0.12 * scale, 3)
    base = nm("plaza_floor", style)
    calls = [
        prim(
            "cylinder",
            base,
            params={"radius": base_r, "height": base_h, "segments": 40},
            translation=[0, base_h / 2, 0],
        ),
        material_call([base], base_mk, "Plaza Stone"),
        prim(
            "capsule",
            bol,
            params={"radius": radius, "height": cyl_h, "segments": 16, "rings": 4},
            translation=[ring_r, base_h + cyl_h / 2 + radius, 0],
        ),
        material_call([bol], mk, "Bollard Bronze"),
        *array_radial(bol, count=count, angle=360.0, axis=1.0),
    ]
    notes = (
        f"{count} ornamental bronze bollards, array-radial'd around a broad plaza floor "
        f"(radius {base_r} m, past the bollards' own outer edge {round(outer_edge, 3)} m)."
    )
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3], "r": ring_r}


PROMPTS_RING_BRONZE_BOLLARDS = [
    lambda r, c: "twelve bronze bollards arranged in a ring on a broad plaza floor",
    lambda r, c: f"{c['count']} {c['mat']} bollards set in a circle on a wide plaza floor",
    lambda r, c: "a large ring of ornamental bronze bollards standing on a broad paved floor",
    lambda r, c: f"bollard ring, {c['count']} of them, {c['mat']}, wide plaza floor",
]
register("ring_bollards_bronze", _b_ring_bollards_bronze, PROMPTS_RING_BRONZE_BOLLARDS)


# 9. thirteen slender columns in a ring under a thin ring beam, on a broad platform (N=13)
def _b_ring_slender_columns_capped(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = 13
    scale = rng.choice([0.9, 1.0, 1.15])
    ring_r = round(3.0 * scale, 3)
    radius = round(0.14 * scale, 3)
    height = round(2.8 * scale, 2)
    mk = rng.choice(["marble_white", "sandstone", "weathered_stone"])
    style = v
    col = nm("column", style)
    outer_edge = ring_r + radius
    base_r = base_radius_for(outer_edge)
    base_h = round(0.3 * scale, 3)
    base = nm("rotunda_floor", style)
    ring_beam = nm("ring_beam", style)
    tube = round(0.09 * scale, 3)
    calls = [
        prim(
            "cylinder",
            base,
            params={"radius": base_r, "height": base_h, "segments": 44},
            translation=[0, base_h / 2, 0],
        ),
        material_call([base], mk, "Rotunda Floor"),
        prim(
            "column",
            col,
            params={
                "radius": radius,
                "height": height,
                "segments": 14,
                "base": 0.08,
                "capital": 0.08,
            },
            translation=[ring_r, base_h + height / 2, 0],
        ),
        material_call([col], mk, "Column Stone"),
        *array_radial(col, count=count, angle=360.0, axis=1.0),
        prim(
            "torus",
            ring_beam,
            params={"radius": ring_r, "tube": tube, "segments": 48, "sides": 10},
            translation=[0, base_h + height + tube, 0],
        ),
        material_call([ring_beam], mk, "Ring Beam Stone"),
    ]
    notes = (
        f"an open rotunda frame: {count} slender columns array-radial'd on a broad round floor "
        f"(radius {base_r} m, past the columns' own outer edge {round(outer_edge, 3)} m), a "
        f"single thin torus resting on top as the ring beam, y computed from the columns' own "
        f"height exactly as the two earlier capped rings compute theirs."
    )
    return calls, notes, {"count": count, "mat": MATERIALS[mk][3], "r": ring_r}


PROMPTS_RING_SLENDER_CAPPED = [
    lambda r, c: "thirteen slender columns in a ring under a thin ring beam, on a broad platform",
    lambda r, c: (
        f"{c['count']} slender {c['mat']} columns in a circle, thin ring beam above, broad floor"
    ),
    lambda r, c: (
        "an open rotunda frame of slender columns carrying a thin ring beam, wide round floor"
    ),
    lambda r, c: f"rotunda frame: {c['count']} columns, ring beam, {c['mat']}, broad floor",
]
register("ring_slender_columns_capped", _b_ring_slender_columns_capped, PROMPTS_RING_SLENDER_CAPPED)


# 10. fourteen garden candles in a ring on a wide tray (N=14, two-part seed)
def _b_ring_candles(rng: random.Random, v: int) -> tuple[list, str, dict]:
    count = 14
    scale = rng.choice([0.9, 1.0, 1.15])
    ring_r = round(0.9 * scale, 3)
    radius = round(0.03 * scale, 4)
    body_h = round(0.16 * scale, 3)
    flame_r = round(0.014 * scale, 4)
    mk = "candle_wax"
    style = v
    body = nm("candle", style)
    flame = nm("flame", style)
    outer_edge = ring_r + radius
    base_r = base_radius_for(outer_edge)
    base_h = round(0.02 * scale, 3)
    base = nm("tray", style)
    calls = [
        prim(
            "cylinder",
            base,
            params={"radius": base_r, "height": base_h, "segments": 36},
            translation=[0, base_h / 2, 0],
        ),
        material_call([base], "bronze", "Tray Metal"),
        prim(
            "cylinder",
            body,
            params={"radius": radius, "height": body_h, "segments": 12},
            translation=[ring_r, base_h + body_h / 2, 0],
        ),
        material_call([body], mk, "Candle Wax"),
        prim(
            "icosphere",
            flame,
            params={"radius": flame_r, "subdivisions": 1},
            translation=[ring_r, base_h + body_h + flame_r, 0],
        ),
        material_call([flame], "flame_amber", "Candle Flame"),
        *array_radial(body, count=count, angle=360.0, axis=1.0),
        *array_radial(flame, count=count, angle=360.0, axis=1.0),
    ]
    notes = (
        f"{count} candles (never eight, spread across this family up to fourteen), a two-part "
        f"seed (wax body plus a small flame sphere, both at the same ring radius) duplicated "
        f"with two array-radial calls at the same count/angle/axis; the metal tray beneath "
        f"(radius {base_r} m) clears the candles' own outer edge ({round(outer_edge, 3)} m)."
    )
    return calls, notes, {"count": count, "r": ring_r}


PROMPTS_RING_CANDLES = [
    lambda r, c: "fourteen garden candles arranged in a ring on a wide tray",
    lambda r, c: f"{c['count']} small candles set in a circle on a broad metal tray",
    lambda r, c: "a ring of lit candles standing on a wide tray, evenly spaced",
    lambda r, c: f"candle ring, {c['count']} of them, on a broad tray",
]
register("ring_candles", _b_ring_candles, PROMPTS_RING_CANDLES)


# =============================================================================
# TRIPOD / SPLAYED-LEG kind -- per-leg rotation solved by trigonometry
# (splayed_legs, defined above), names consistent through the batch, no
# telescopes.
# =============================================================================


# 11. a slim camera stand on three splayed metal legs
def _b_camera_stand(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    h_top = round(0.85 * scale, 3)
    r_foot = round(0.32 * scale, 3)
    leg_r = round(0.012 * scale, 4)
    mk = rng.choice(["iron_dark", "steel_scifi", "brass"])
    style = v
    legs, leg_names, _ = splayed_legs(
        n=3, r_top=0.0, r_foot=r_foot, h_top=h_top, style=style, radius=leg_r, segments=10
    )
    plate = nm("mount_plate", style)
    plate_h = round(0.018 * scale, 4)
    plate_size = [round(0.11 * scale, 3), plate_h, round(0.09 * scale, 3)]
    plate_y, plate_top = stack_y(h_top, plate_h)
    body = nm("camera_body", style)
    body_size = [round(0.09 * scale, 3), round(0.06 * scale, 3), round(0.13 * scale, 3)]
    body_y, _ = stack_y(plate_top, body_size[1])
    calls = [
        *legs,
        material_call(leg_names, mk, "Stand Metal"),
        prim("box", plate, params={"size": plate_size}, translation=[0, plate_y, 0]),
        material_call([plate], mk, "Mount Plate"),
        prim("box", body, params={"size": body_size}, translation=[0, body_y, 0]),
        material_call([body], "black_plastic", "Camera Body"),
    ]
    notes = (
        f"three legs converge on the world Y axis at y={h_top} (r_top=0.0, a true apex) and "
        f"splay out to feet at radius {r_foot} m -- each leg's own tilt is solved from "
        "atan2(r_top - r_foot, h_top), not guessed. The mount plate's y is h_top plus half its "
        "own thickness, and the camera body's y is the plate's own top plus half its own "
        "height -- one computed stack, names leg_1/leg_2/leg_3 held consistent throughout."
    )
    return calls, notes, {"h": h_top, "mat": MATERIALS[mk][3]}


PROMPTS_CAMERA_STAND = [
    lambda r, c: "a slim camera stand on three splayed metal legs",
    lambda r, c: (
        f"a {c['mat']} camera stand, {fmt_m(c['h'])} m tall, three splayed legs meeting at the mount"
    ),
    lambda r, c: "a small camera mount held up by three legs that splay outward to the floor",
    lambda r, c: f"camera stand, {c['mat']}, three splayed legs, compact mount plate on top",
]
register("camera_stand", _b_camera_stand, PROMPTS_CAMERA_STAND)


# 12. a wrought-iron plant stand on three splayed legs
def _b_plant_stand(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    h_top = round(0.55 * scale, 3)
    r_top = round(0.12 * scale, 3)
    r_foot = round(0.22 * scale, 3)
    leg_r = round(0.015 * scale, 4)
    mk = rng.choice(["iron_dark", "bronze"])
    style = v
    legs, leg_names, _ = splayed_legs(
        n=3, r_top=r_top, r_foot=r_foot, h_top=h_top, style=style, radius=leg_r, segments=10
    )
    tray = nm("plant_tray", style)
    tray_r = round(0.17 * scale, 3)
    tray_h = round(0.02 * scale, 4)
    tray_y, _ = stack_y(h_top, tray_h)
    calls = [
        *legs,
        material_call(leg_names, mk, "Stand Metal"),
        prim(
            "cylinder",
            tray,
            params={"radius": tray_r, "height": tray_h, "segments": 24},
            translation=[0, tray_y, 0],
        ),
        material_call([tray], mk, "Tray Metal"),
    ]
    notes = (
        f"three legs attach in their own ring (r_top={r_top} m, not a single apex) at height "
        f"{h_top} m and splay out to feet at radius {r_foot} m; the tray's y is h_top plus half "
        "its own thickness, resting exactly on the legs' own attachment ring, wider than that "
        "ring so nothing overhangs unsupported."
    )
    return calls, notes, {"h": h_top, "mat": MATERIALS[mk][3]}


PROMPTS_PLANT_STAND = [
    lambda r, c: "a wrought-iron plant stand on three splayed legs",
    lambda r, c: (
        f"a {c['mat']} plant stand, {fmt_m(c['h'])} m tall, three splayed legs, round tray on top"
    ),
    lambda r, c: "a small plant stand whose three legs splay outward beneath a round tray",
    lambda r, c: f"plant stand, {c['mat']}, three splayed legs holding a round tray",
]
register("plant_stand", _b_plant_stand, PROMPTS_PLANT_STAND)


# 13. a campfire cooking stand on three splayed iron legs
def _b_campfire_stand(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.9, 1.0, 1.15])
    h_top = round(0.7 * scale, 3)
    r_foot = round(0.55 * scale, 3)
    leg_r = round(0.016 * scale, 4)
    mk = "iron_dark"
    style = v
    legs, leg_names, _ = splayed_legs(
        n=3, r_top=0.0, r_foot=r_foot, h_top=h_top, style=style, radius=leg_r, segments=10
    )
    ring = nm("hook_ring", style)
    tube = round(0.012 * scale, 4)
    ring_r = round(0.05 * scale, 3)
    ring_y, _ = stack_y(h_top, tube * 2)
    calls = [
        *legs,
        material_call(leg_names, mk, "Cooking Stand Iron"),
        prim(
            "torus",
            ring,
            params={"radius": ring_r, "tube": tube, "segments": 20, "sides": 8},
            translation=[0, ring_y, 0],
        ),
        material_call([ring], mk, "Hook Ring Iron"),
    ]
    notes = (
        f"three iron legs converge to a true apex (r_top=0.0) at y={h_top}, splayed to feet at "
        f"radius {r_foot} m for a wide, stable footprint over a fire; a small torus ring sits "
        "at the apex, its y computed the same way every stacked part in this family is, for "
        "hanging a cooking pot from."
    )
    return calls, notes, {"h": h_top, "mat": MATERIALS[mk][3]}


PROMPTS_CAMPFIRE_STAND = [
    lambda r, c: "a campfire cooking stand on three splayed iron legs",
    lambda r, c: (
        f"a {c['mat']} cooking stand, {fmt_m(c['h'])} m tall, three legs splayed wide over a fire"
    ),
    lambda r, c: (
        "a tripod-shaped cooking stand for a campfire, three iron legs, a small hanging ring on top"
    ),
    lambda r, c: f"campfire stand, {c['mat']}, three splayed legs, hook ring at the apex",
]
register("campfire_stand", _b_campfire_stand, PROMPTS_CAMPFIRE_STAND)


# 14. a small stool with four splayed wooden legs
def _b_stool_splayed(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    h_top = round(0.44 * scale, 3)
    r_top = round(0.13 * scale, 3)
    r_foot = round(0.19 * scale, 3)
    box_size = round(0.03 * scale, 4)
    mk = rng.choice(["oak_wood", "weathered_wood"])
    style = v
    legs, leg_names, _ = splayed_legs(
        n=4,
        r_top=r_top,
        r_foot=r_foot,
        h_top=h_top,
        style=style,
        generator="box",
        box_size=box_size,
    )
    seat = nm("seat", style)
    seat_r = round(0.16 * scale, 3)
    seat_h = round(0.035 * scale, 4)
    seat_y, _ = stack_y(h_top, seat_h)
    calls = [
        *legs,
        material_call(leg_names, mk, "Stool Legs"),
        prim(
            "cylinder",
            seat,
            params={"radius": seat_r, "height": seat_h, "segments": 24},
            translation=[0, seat_y, 0],
        ),
        material_call([seat], mk, "Stool Seat"),
    ]
    notes = (
        f"four legs attach in their own ring (r_top={r_top} m) at height {h_top} m and splay to "
        f"feet at radius {r_foot} m -- the round seat (radius {seat_r} m) sits exactly on the "
        "legs' own attachment ring, its y computed as h_top plus half its own thickness."
    )
    return calls, notes, {"h": h_top, "mat": MATERIALS[mk][3]}


PROMPTS_STOOL = [
    lambda r, c: "a small stool with four splayed wooden legs",
    lambda r, c: f"a {c['mat']} stool, seat {fmt_m(c['h'])} m up, four legs splayed outward",
    lambda r, c: "a low round stool whose four legs splay outward from the seat",
    lambda r, c: f"stool, {c['mat']}, four splayed legs, round seat",
]
register("stool_splayed", _b_stool_splayed, PROMPTS_STOOL)


# 15. a three-legged easel stand holding a canvas panel
def _b_easel(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    h_top = round(1.15 * scale, 3)
    r_foot = round(0.4 * scale, 3)
    box_size = round(0.035 * scale, 4)
    mk = rng.choice(["oak_wood", "weathered_wood"])
    style = v
    legs, leg_names, _ = splayed_legs(
        n=3, r_top=0.02, r_foot=r_foot, h_top=h_top, style=style, generator="box", box_size=box_size
    )
    ledge = nm("ledge", style)
    ledge_w = round(0.5 * scale, 3)
    ledge_h = round(0.03 * scale, 4)
    ledge_y, ledge_top = stack_y(h_top, ledge_h)
    canvas = nm("canvas", style)
    canvas_w = round(0.4 * scale, 3)
    canvas_h = round(0.5 * scale, 3)
    canvas_t = round(0.02 * scale, 4)
    canvas_y, _ = stack_y(ledge_top, canvas_h)
    calls = [
        *legs,
        material_call(leg_names, mk, "Easel Legs"),
        prim(
            "box",
            ledge,
            params={"size": [ledge_w, ledge_h, box_size * 3]},
            translation=[0, ledge_y, 0],
        ),
        material_call([ledge], mk, "Easel Ledge"),
        prim(
            "box",
            canvas,
            params={"size": [canvas_w, canvas_h, canvas_t]},
            translation=[0, canvas_y, box_size * 2],
        ),
        material_call([canvas], "canvas_cream", "Canvas"),
    ]
    notes = (
        f"three legs converge near the top (r_top=0.02 m) at y={h_top}, splayed wide to feet at "
        f"radius {r_foot} m for stability; a ledge shelf rests exactly on that near-apex, and "
        "the canvas panel's own bottom rests exactly on the ledge's own top -- two chained "
        "computed stacks, not guessed heights."
    )
    return calls, notes, {"h": h_top, "mat": MATERIALS[mk][3]}


PROMPTS_EASEL = [
    lambda r, c: "a three-legged easel stand holding a canvas panel",
    lambda r, c: (
        f"a {c['mat']} easel, {fmt_m(c['h'])} m tall, three splayed legs, canvas resting on the ledge"
    ),
    lambda r, c: (
        "an artist's easel whose three legs splay wide, a small canvas propped on its ledge"
    ),
    lambda r, c: f"easel stand, {c['mat']}, three splayed legs, canvas panel on the shelf",
]
register("easel", _b_easel, PROMPTS_EASEL)


# 16. a side table on four splayed wooden legs
def _b_side_table_splayed(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    h_top = round(0.5 * scale, 3)
    r_top = round(0.16 * scale, 3)
    r_foot = round(0.24 * scale, 3)
    box_size = round(0.035 * scale, 4)
    mk = rng.choice(["oak_wood", "weathered_wood"])
    style = v
    legs, leg_names, _ = splayed_legs(
        n=4,
        r_top=r_top,
        r_foot=r_foot,
        h_top=h_top,
        style=style,
        generator="box",
        box_size=box_size,
    )
    top = nm("tabletop", style)
    top_w = round(0.42 * scale, 3)
    top_d = round(0.3 * scale, 3)
    top_h = round(0.03 * scale, 4)
    top_y, _ = stack_y(h_top, top_h)
    calls = [
        *legs,
        material_call(leg_names, mk, "Table Legs"),
        prim(
            "box",
            top,
            params={"size": [top_w, top_h, top_d]},
            translation=[0, top_y, 0],
        ),
        material_call([top], mk, "Tabletop"),
    ]
    notes = (
        f"four legs attach in their own ring (r_top={r_top} m) at height {h_top} m and splay to "
        f"feet at radius {r_foot} m -- the rectangular top ({top_w}x{top_d} m) sits exactly on "
        "the legs' own attachment ring, its y computed the same way the stool's seat is."
    )
    return calls, notes, {"h": h_top, "mat": MATERIALS[mk][3]}


PROMPTS_SIDE_TABLE = [
    lambda r, c: "a side table on four splayed wooden legs",
    lambda r, c: f"a {c['mat']} side table, {fmt_m(c['h'])} m tall, four legs splayed outward",
    lambda r, c: "a small rectangular side table whose four legs splay outward to the floor",
    lambda r, c: f"side table, {c['mat']}, four splayed legs, small rectangular top",
]
register("side_table_splayed", _b_side_table_splayed, PROMPTS_SIDE_TABLE)


# 17. a reading lamp on three splayed metal legs
def _b_tripod_lamp(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    h_top = round(1.3 * scale, 3)
    r_foot = round(0.4 * scale, 3)
    leg_r = round(0.014 * scale, 4)
    mk = rng.choice(["iron_dark", "bronze", "brass"])
    style = v
    legs, leg_names, _ = splayed_legs(
        n=3, r_top=0.015, r_foot=r_foot, h_top=h_top, style=style, radius=leg_r, segments=10
    )
    pole = nm("pole", style)
    pole_h = round(0.35 * scale, 3)
    pole_y, pole_top = stack_y(h_top, pole_h)
    shade = nm("lampshade", style)
    shade_r = round(0.16 * scale, 3)
    shade_h = round(0.18 * scale, 3)
    shade_y, _ = stack_y(pole_top, shade_h)
    calls = [
        *legs,
        material_call(leg_names, mk, "Lamp Stand"),
        prim(
            "cylinder",
            pole,
            params={"radius": leg_r * 1.5, "height": pole_h, "segments": 10},
            translation=[0, pole_y, 0],
        ),
        material_call([pole], mk, "Lamp Pole"),
        prim(
            "cone",
            shade,
            params={"radius": shade_r, "height": shade_h, "segments": 20},
            translation=[0, shade_y, 0],
            rotation=[180.0, 0.0, 0.0],
        ),
        material_call([shade], "canvas_cream", "Lampshade"),
    ]
    notes = (
        f"three legs converge near the top (r_top=0.015 m) at y={h_top}, splayed to feet at "
        f"radius {r_foot} m; a short pole extends the apex upward by {pole_h} m, and the "
        "shade (a cone flipped 180 degrees about X so its wide mouth faces down) sits with its "
        "own bottom exactly on the pole's own top -- a three-part chained stack."
    )
    return calls, notes, {"h": h_top, "mat": MATERIALS[mk][3]}


PROMPTS_TRIPOD_LAMP = [
    lambda r, c: "a reading lamp on three splayed metal legs",
    lambda r, c: (
        f"a {c['mat']} reading lamp, {fmt_m(c['h'])} m tall, three splayed legs, shade on top"
    ),
    lambda r, c: "a floor lamp whose three legs splay outward beneath a tall pole and shade",
    lambda r, c: f"lamp stand, {c['mat']}, three splayed legs, cone shade on top",
]
register("tripod_lamp", _b_tripod_lamp, PROMPTS_TRIPOD_LAMP)


# =============================================================================
# STACK kind -- each upper part's y computed from the part beneath it.
# =============================================================================


# 18. a stone bowl resting on a pedestal
def _b_bowl_on_pedestal(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    ped_r = round(0.16 * scale, 3)
    ped_h = round(0.75 * scale, 3)
    bowl_r = round(0.32 * scale, 3)
    bowl_h = round(0.16 * scale, 3)
    mk = rng.choice(["marble_white", "granite", "weathered_stone"])
    style = v
    ped = nm("pedestal", style)
    bowl = nm("bowl", style)
    bowl_y, _ = stack_y(ped_h, bowl_h)
    calls = [
        prim(
            "cylinder",
            ped,
            params={"radius": ped_r, "height": ped_h, "segments": 24},
            translation=[0, ped_h / 2, 0],
        ),
        material_call([ped], mk, "Pedestal Stone"),
        prim(
            "cylinder",
            bowl,
            params={"radius": bowl_r, "height": bowl_h, "segments": 28},
            translation=[0, bowl_y, 0],
        ),
        material_call([bowl], mk, "Bowl Stone"),
    ]
    notes = f"a narrow pedestal ({ped_h} m) carrying a wide shallow bowl whose y is the pedestal's own height plus half the bowl's own thickness -- no gap, no overlap."
    return calls, notes, {"h": ped_h + bowl_h, "mat": MATERIALS[mk][3]}


PROMPTS_BOWL_PEDESTAL = [
    lambda r, c: "a stone bowl resting on a pedestal",
    lambda r, c: f"a wide {c['mat']} bowl on a narrow pedestal, {fmt_m(c['h'])} m total",
    lambda r, c: "a shallow stone bowl sitting on top of a plain pedestal",
    lambda r, c: f"bowl on pedestal, {c['mat']}, {fmt_m(c['h'])} m tall overall",
]
register("bowl_on_pedestal", _b_bowl_on_pedestal, PROMPTS_BOWL_PEDESTAL)


# 19. a lamp mounted on a tall post
def _b_lamp_on_post(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    post_r = round(0.045 * scale, 3)
    post_h = round(2.6 * scale, 3)
    head_r = round(0.15 * scale, 3)
    mk = rng.choice(["iron_dark", "bronze"])
    style = v
    post = nm("post", style)
    head = nm("lamp_head", style)
    head_y, _ = stack_y(post_h, 2 * head_r)
    calls = [
        prim(
            "cylinder",
            post,
            params={"radius": post_r, "height": post_h, "segments": 16},
            translation=[0, post_h / 2, 0],
        ),
        material_call([post], mk, "Post Metal"),
        prim(
            "uv_sphere",
            head,
            params={"radius": head_r, "segments": 16, "rings": 10},
            translation=[0, post_h + head_r, 0],
        ),
        material_call([head], "flame_amber", "Lamp Glass"),
    ]
    notes = f"a tall post ({post_h} m) carrying a spherical lamp head whose y is the post's own height plus its own radius -- the sphere's bottom sits exactly on the post's own top."
    return calls, notes, {"h": head_y, "mat": MATERIALS[mk][3]}


PROMPTS_LAMP_POST = [
    lambda r, c: "a lamp mounted on a tall post",
    lambda r, c: f"a {c['mat']} lamp post, {fmt_m(c['h'])} m tall, round glass head on top",
    lambda r, c: "a street lamp whose glass head sits on top of a tall metal post",
    lambda r, c: f"lamp on a post, {c['mat']}, {fmt_m(c['h'])} m overall",
]
register("lamp_on_post", _b_lamp_on_post, PROMPTS_LAMP_POST)


# 20. a birdbath with a basin on a turned stem
def _b_birdbath(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    foot_r = round(0.22 * scale, 3)
    foot_h = round(0.1 * scale, 3)
    stem_r = round(0.06 * scale, 3)
    stem_h = round(0.55 * scale, 3)
    basin_r = round(0.34 * scale, 3)
    basin_h = round(0.09 * scale, 3)
    mk = rng.choice(["weathered_stone", "marble_white", "granite"])
    style = v
    foot = nm("foot", style)
    stem = nm("stem", style)
    basin = nm("basin", style)
    stem_y, stem_top = stack_y(foot_h, stem_h)
    basin_y, _ = stack_y(stem_top, basin_h)
    calls = [
        prim(
            "cylinder",
            foot,
            params={"radius": foot_r, "height": foot_h, "segments": 24},
            translation=[0, foot_h / 2, 0],
        ),
        material_call([foot], mk, "Birdbath Foot"),
        prim(
            "cylinder",
            stem,
            params={"radius": stem_r, "height": stem_h, "segments": 20},
            translation=[0, stem_y, 0],
        ),
        material_call([stem], mk, "Birdbath Stem"),
        prim(
            "cylinder",
            basin,
            params={"radius": basin_r, "height": basin_h, "segments": 28},
            translation=[0, basin_y, 0],
        ),
        material_call([basin], mk, "Birdbath Basin"),
    ]
    notes = (
        "three parts chained: the stem's y is the foot's own height plus half the stem's, and "
        "the basin's y is the stem's own top plus half the basin's -- each computed from the "
        "part directly beneath it, never from the ground."
    )
    return calls, notes, {"h": foot_h + stem_h + basin_h, "mat": MATERIALS[mk][3]}


PROMPTS_BIRDBATH = [
    lambda r, c: "a birdbath with a basin on a turned stem",
    lambda r, c: f"a {c['mat']} birdbath, {fmt_m(c['h'])} m tall, wide basin on a narrow stem",
    lambda r, c: "a garden birdbath whose shallow basin sits on top of a narrow stem and foot",
    lambda r, c: f"birdbath, {c['mat']}, foot, stem and basin stacked, {fmt_m(c['h'])} m tall",
]
register("birdbath", _b_birdbath, PROMPTS_BIRDBATH)


# 21. a turned vase standing on a low plinth
def _b_vase_on_plinth(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    plinth_w = round(0.3 * scale, 3)
    plinth_h = round(0.18 * scale, 3)
    vase_h = round(0.42 * scale, 3)
    mk = rng.choice(["marble_white", "terracotta", "weathered_stone"])
    style = v
    plinth = nm("plinth", style)
    vase = nm("vase", style)
    vase_y, _ = stack_y(plinth_h, vase_h)
    radii = [0.02, 0.10, 0.14, 0.08, 0.10, 0.05]
    fracs = [0.0, 0.05, 0.30, 0.72, 0.86, 1.0]
    profile = [
        [round(x * scale, 4), round(f * vase_h, 4)] for x, f in zip(radii, fracs, strict=True)
    ]
    calls = [
        prim(
            "box",
            plinth,
            params={"size": [plinth_w, plinth_h, plinth_w]},
            translation=[0, plinth_h / 2, 0],
        ),
        material_call([plinth], mk, "Plinth Stone"),
        prim(
            "lathe", vase, params={"profile": profile, "segments": 20}, translation=[0, vase_y, 0]
        ),
        material_call([vase], mk, "Vase"),
    ]
    notes = (
        "a lathe-turned vase profile (re-centred on its own bounding box, per this programme's "
        "own lathe rule) whose y is the plinth's own height plus half the vase's own height -- "
        "the same computed-stack rule as every other build in this kind, just with a revolved "
        "profile instead of a plain cylinder."
    )
    return calls, notes, {"h": plinth_h + vase_h, "mat": MATERIALS[mk][3]}


PROMPTS_VASE_PLINTH = [
    lambda r, c: "a turned vase standing on a low plinth",
    lambda r, c: (
        f"a {c['mat']} turned vase, {fmt_m(c['h'])} m tall overall, standing on a low plinth"
    ),
    lambda r, c: "a bulbous turned vase resting on a low square plinth",
    lambda r, c: f"vase on plinth, {c['mat']}, {fmt_m(c['h'])} m tall",
]
register("vase_on_plinth", _b_vase_on_plinth, PROMPTS_VASE_PLINTH)


# 22. a stack of three wooden crates
def _b_crate_stack(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    sizes = [round(0.5 * scale, 3), round(0.42 * scale, 3), round(0.34 * scale, 3)]
    mk = rng.choice(["oak_wood", "weathered_wood"])
    style = v
    names = []
    calls = []
    top = 0.0
    offsets = [
        (0.0, 0.0),
        (round(0.03 * scale, 3), round(-0.02 * scale, 3)),
        (round(-0.02 * scale, 3), round(0.03 * scale, 3)),
    ]
    for i, s in enumerate(sizes):
        name = nm(f"crate_{i + 1}", style)
        names.append(name)
        y, top = stack_y(top, s)
        ox, oz = offsets[i]
        calls.append(prim("box", name, params={"size": [s, s, s]}, translation=[ox, y, oz]))
    calls.append(material_call(names, mk, "Crate Wood"))
    notes = (
        "three cubic crates, each smaller than the last, each one's y computed as the running "
        "top of the stack below it plus half its own height -- a small, deliberately bounded "
        "lateral offset per crate (never enough to lose support) rather than a perfectly "
        "centred, unrealistic tower."
    )
    return calls, notes, {"h": top, "mat": MATERIALS[mk][3]}


PROMPTS_CRATE_STACK = [
    lambda r, c: "a stack of three wooden crates",
    lambda r, c: f"three {c['mat']} crates stacked one on another, {fmt_m(c['h'])} m tall overall",
    lambda r, c: "three crates of decreasing size stacked in a small tower",
    lambda r, c: f"crate stack, {c['mat']}, three boxes, {fmt_m(c['h'])} m tall",
]
register("crate_stack", _b_crate_stack, PROMPTS_CRATE_STACK)


# 23. a globe resting on a stand
def _b_globe_on_stand(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    ped_r = round(0.12 * scale, 3)
    ped_h = round(0.7 * scale, 3)
    globe_r = round(0.22 * scale, 3)
    mk = rng.choice(["oak_wood", "weathered_wood", "iron_dark"])
    style = v
    ped = nm("stand", style)
    globe = nm("globe", style)
    globe_y, _ = stack_y(ped_h, 2 * globe_r)
    calls = [
        prim(
            "cylinder",
            ped,
            params={"radius": ped_r, "height": ped_h, "segments": 20},
            translation=[0, ped_h / 2, 0],
        ),
        material_call([ped], mk, "Stand"),
        prim(
            "uv_sphere",
            globe,
            params={"radius": globe_r, "segments": 24, "rings": 16},
            translation=[0, ped_h + globe_r, 0],
        ),
        material_call([globe], "marble_white", "Globe"),
    ]
    notes = f"a plain pedestal ({ped_h} m) carrying a sphere whose y is the pedestal's own height plus its own radius -- the sphere's bottom sits exactly on the pedestal's own top."
    return calls, notes, {"h": globe_y, "mat": MATERIALS[mk][3]}


PROMPTS_GLOBE_STAND = [
    lambda r, c: "a globe resting on a stand",
    lambda r, c: f"a large globe on a {c['mat']} stand, {fmt_m(c['h'])} m tall overall",
    lambda r, c: "a round globe sitting on top of a plain pedestal stand",
    lambda r, c: f"globe on stand, {c['mat']} pedestal, {fmt_m(c['h'])} m tall",
]
register("globe_on_stand", _b_globe_on_stand, PROMPTS_GLOBE_STAND)


# 24. a sundial mounted on a stone pedestal
def _b_sundial_on_pedestal(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    ped_w = round(0.28 * scale, 3)
    ped_h = round(0.85 * scale, 3)
    dial_r = round(0.24 * scale, 3)
    dial_h = round(0.04 * scale, 3)
    gnomon_h = round(0.16 * scale, 3)
    mk = rng.choice(["granite", "weathered_stone", "marble_white"])
    style = v
    ped = nm("pedestal", style)
    dial = nm("dial", style)
    gnomon = nm("gnomon", style)
    dial_y, dial_top = stack_y(ped_h, dial_h)
    gnomon_y, _ = stack_y(dial_top, gnomon_h)
    calls = [
        prim("box", ped, params={"size": [ped_w, ped_h, ped_w]}, translation=[0, ped_h / 2, 0]),
        material_call([ped], mk, "Pedestal Stone"),
        prim(
            "cylinder",
            dial,
            params={"radius": dial_r, "height": dial_h, "segments": 28},
            translation=[0, dial_y, 0],
        ),
        material_call([dial], mk, "Dial Stone"),
        prim(
            "pyramid",
            gnomon,
            params={"base": round(dial_r * 0.35, 3), "height": gnomon_h, "sides": 3},
            translation=[0, gnomon_y, 0],
        ),
        material_call([gnomon], "bronze", "Gnomon"),
    ]
    notes = (
        "three parts chained: the dial's y is the pedestal's own height plus half the dial's, "
        "and the gnomon's y is the dial's own top plus half the gnomon's own height -- a "
        "triangular-sided pyramid standing for the blade, kept upright rather than tilted so "
        "its own centred bounding box stays trivial to place exactly."
    )
    return calls, notes, {"h": ped_h + dial_h + gnomon_h, "mat": MATERIALS[mk][3]}


PROMPTS_SUNDIAL = [
    lambda r, c: "a sundial mounted on a stone pedestal",
    lambda r, c: f"a {c['mat']} sundial, {fmt_m(c['h'])} m tall overall, on a square pedestal",
    lambda r, c: "a round sundial face with a small blade, sitting on a stone pedestal",
    lambda r, c: f"sundial on pedestal, {c['mat']}, {fmt_m(c['h'])} m tall",
]
register("sundial_on_pedestal", _b_sundial_on_pedestal, PROMPTS_SUNDIAL)


# 25. two barrels stacked one on the other
def _b_barrel_stack(rng: random.Random, v: int) -> tuple[list, str, dict]:
    scale = rng.choice([0.85, 1.0, 1.15])
    r1 = round(0.28 * scale, 3)
    h1 = round(0.55 * scale, 3)
    r2 = round(0.24 * scale, 3)
    h2 = round(0.46 * scale, 3)
    mk = rng.choice(["oak_wood", "weathered_wood"])
    style = v
    b1 = nm("barrel_1", style)
    b2 = nm("barrel_2", style)
    y2, _ = stack_y(h1, h2)
    ox = round(0.02 * scale, 3)
    calls = [
        prim(
            "cylinder",
            b1,
            params={"radius": r1, "height": h1, "segments": 20},
            translation=[0, h1 / 2, 0],
        ),
        prim(
            "cylinder",
            b2,
            params={"radius": r2, "height": h2, "segments": 20},
            translation=[ox, y2, 0],
        ),
        material_call([b1, b2], mk, "Barrel Wood"),
    ]
    notes = (
        f"the top barrel's y is the bottom barrel's own height plus half its own height, with a "
        f"small horizontal offset ({ox} m, well inside the bottom barrel's own radius {r1} m) "
        "so it reads as casually stacked rather than perfectly centred, without overhanging."
    )
    return calls, notes, {"h": h1 + h2, "mat": MATERIALS[mk][3]}


PROMPTS_BARREL_STACK = [
    lambda r, c: "two barrels stacked one on the other",
    lambda r, c: f"two {c['mat']} barrels stacked, {fmt_m(c['h'])} m tall overall",
    lambda r, c: "a smaller barrel stacked on top of a larger one",
    lambda r, c: f"barrel stack, {c['mat']}, two barrels, {fmt_m(c['h'])} m tall",
]
register("barrel_stack", _b_barrel_stack, PROMPTS_BARREL_STACK)


# =============================================================================
# emit
# =============================================================================


def main() -> None:
    records = []
    idx = 1
    for template in TEMPLATES:
        for v in range(PHRASING_COUNT):
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
