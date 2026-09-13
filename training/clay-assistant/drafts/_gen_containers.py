"""Deterministic generator for ``drafts/containers.jsonl``.

Run with ``uv run python training/clay-assistant/drafts/_gen_containers.py``
from the repo root (or anywhere -- paths are resolved relative to this
file). Regenerates ``containers.jsonl`` beside it. Seeded (``SEED`` below):
reruns are byte-identical.

Fifty hand-designed base builds (one per line of
``training/clay-assistant/seeds/containers.txt``, in file order), each
written out as six prompt phrasings with small deterministic jitter on
dimensions/materials/counts -- 300 records total.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

FAMILY = "containers"
SEED = 20260912
OUT = Path(__file__).parent / f"{FAMILY}.jsonl"

# --------------------------------------------------------------------------
# small call-building helpers
# --------------------------------------------------------------------------


def prim(
    name: str,
    generator: str,
    params: dict | None = None,
    translation: list[float] | None = None,
    rotation: list[float] | None = None,
    scale: list[float] | None = None,
) -> dict:
    args: dict[str, Any] = {"generator": generator, "name": name}
    if params is not None:
        args["params"] = params
    if translation is not None:
        args["translation"] = [round(float(v), 4) for v in translation]
    if rotation is not None:
        args["rotation"] = [round(float(v), 3) for v in rotation]
    if scale is not None:
        args["scale"] = [round(float(v), 4) for v in scale]
    return {"name": "clay_add_primitive", "arguments": args}


def ref(name: str) -> dict:
    return {"$ref": name}


def dup_refs(base: str, count: int) -> list[dict]:
    """``$ref`` for *base* plus its ``count - 1`` array-op copies.

    ``clay_ops._array_linear``/``_array_radial`` name each copy through
    ``ops.next_name`` -- ``base``, ``base.001``, ``base.002``... -- counting
    up deterministically as long as ``base`` itself carries no numeric
    suffix and nothing else in the (small, hand-built) scene already claims
    one of those names, which holds for every array in this family. A
    material or boolean call that only names the original leaves every
    array copy on the document's default material (or, for a boolean,
    un-cut) -- this is how every array-produced group is addressed as a
    whole afterward.
    """
    return [ref(base)] + [ref(f"{base}.{k:03d}") for k in range(1, count)]


def material(
    uids: list, color: list[float], name: str | None = None, roughness=None, metallic=None
) -> dict:
    args: dict[str, Any] = {"uids": uids, "color": [round(float(c), 3) for c in color]}
    if name:
        args["name"] = name
    if roughness is not None:
        args["roughness"] = round(float(roughness), 3)
    if metallic is not None:
        args["metallic"] = round(float(metallic), 3)
    return {"name": "clay_material", "arguments": args}


def select(uids: list) -> dict:
    return {"name": "clay_select", "arguments": {"uids": uids}}


def op(name: str, params: dict | None = None) -> dict:
    args: dict[str, Any] = {"name": name}
    if params is not None:
        args["params"] = params
    return {"name": "clay_op", "arguments": args}


def boolean(kind: str, uids: list) -> dict:
    return {"name": "clay_boolean", "arguments": {"kind": kind, "uids": uids}}


def tube_endpoint_world(
    path: list[list[float]],
    point: list[float],
    translation: list[float],
    scale: list[float],
    rotation_z_deg: float = 0.0,
) -> list[float]:
    """Where *point* (one raw, pre-clamp station of *path*, usually
    ``path[-1]``) actually lands in world space once ``tube`` builds from
    *path* -- accounting for ``_clamp_path``'s own auto re-centring (see its
    docstring: every axis is shifted by the midpoint of its own range) before
    *scale*, an optional rotation about world Z, and *translation* apply,
    exactly as the generator's own pipeline does. A caller that places a
    second object (a tip, a stand) to align with the tube's own end must
    reproduce this or the two visibly separate -- see the family's own
    `recipe_48`/`recipe_50` notes for the render that first caught the
    mismatch.
    """
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    zs = [p[2] for p in path]
    cx, cy, cz = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, (min(zs) + max(zs)) / 2.0
    lx, ly, lz = (point[0] - cx) * scale[0], (point[1] - cy) * scale[1], (point[2] - cz) * scale[2]
    if rotation_z_deg:
        import math

        t = math.radians(rotation_z_deg)
        lx, ly = lx * math.cos(t) - ly * math.sin(t), lx * math.sin(t) + ly * math.cos(t)
    return [lx + translation[0], ly + translation[1], lz + translation[2]]


def profile_height(stations: list[tuple[float, float]]) -> float:
    ys = [y for _, y in stations]
    return max(ys) - min(ys)


def profile_radius(stations: list[tuple[float, float]]) -> float:
    return max(r for r, _ in stations)


WOOD = [0.42, 0.28, 0.16]
DARK_WOOD = [0.30, 0.19, 0.11]
IRON = [0.20, 0.20, 0.22]
BRASS = [0.55, 0.44, 0.18]
STEEL = [0.55, 0.56, 0.58]
CLAY_RED = [0.55, 0.30, 0.20]
TERRACOTTA = [0.62, 0.35, 0.22]
CERAMIC_WHITE = [0.82, 0.80, 0.74]
GLASS_GREEN = [0.35, 0.55, 0.42]
GLASS_CLEAR = [0.75, 0.82, 0.80]
BURLAP = [0.58, 0.48, 0.33]
CARDBOARD = [0.68, 0.55, 0.38]
PLASTIC_BLUE = [0.20, 0.35, 0.62]
SCIFI_GREY = [0.62, 0.64, 0.68]
SCIFI_ACCENT = [0.15, 0.75, 0.80]
LEATHER = [0.40, 0.26, 0.15]
GOLD = [0.72, 0.58, 0.20]
STONE = [0.55, 0.53, 0.48]
COPPER = [0.60, 0.32, 0.18]


def jitter(rng: random.Random, base: float, spread: float = 0.14) -> float:
    return base * (1.0 + rng.uniform(-spread, spread))


def jcol(rng: random.Random, c: list[float], spread: float = 0.06) -> list[float]:
    return [max(0.02, min(0.98, v + rng.uniform(-spread, spread))) for v in c]


SETTINGS = ["medieval", "sci-fi", "modern", "fantasy", "generic"]


def setting_word(rng: random.Random, base: str) -> str:
    """Pick a setting-flavoured adjective to fold into a phrasing, ignored
    for recipes that hardcode their own setting (crate #6 is always sci-fi,
    for instance)."""
    return rng.choice(["medieval", "rustic", "old", "weathered", "fantasy", "modern"])


# --------------------------------------------------------------------------
# Records collected as (id_str, prompt, calls, notes, allow_below_ground)
# --------------------------------------------------------------------------

records: list[dict] = []
_next_id = 1


def emit(prompt: str, calls: list[dict], notes: str, allow_below_ground: bool = False) -> None:
    global _next_id
    rec: dict[str, Any] = {
        "id": f"{FAMILY}-{_next_id:04d}",
        "family": FAMILY,
        "kind": "build",
        "prompt": prompt,
        "calls": calls,
        "notes": notes,
    }
    if allow_below_ground:
        rec["allow_below_ground"] = True
    records.append(rec)
    _next_id += 1


NAME_STYLES = [
    lambda base: base,
    lambda base: base.capitalize(),
    lambda base: base.replace("_", "-"),
    lambda base: base.upper() if len(base) <= 4 else base,
    lambda base: base.replace("_", " "),
    lambda base: base + "_obj",
]


def nm(rng: random.Random, base: str) -> str:
    style = rng.choice(NAME_STYLES)
    return style(base)


# ==========================================================================
# 50 base-build recipes. Each is a function(rng, phrase_idx) -> (prompt,
# calls, notes, allow_below_ground). rng is re-seeded per (recipe, phrase)
# pair by the driver below, so recipes may call rng freely.
# ==========================================================================

Recipe = Any


def recipe_01(rng, i):
    # easy: wooden shipping crate with metal corner plates
    sx = jitter(rng, 0.6)
    sy = jitter(rng, 0.5)
    sz = jitter(rng, 0.45)
    post = 0.05
    proud = 0.006  # stand the post a few mm proud of the crate face so it
    # reads as a plate bolted on rather than z-fighting with a flush face.
    body = nm(rng, "crate_body")
    calls = [prim(body, "box", {"size": [sx, sy, sz]}, [0, sy / 2, 0])]
    posts = []
    for dx, dz, tag in (
        (1, 1, "fr"),
        (-1, 1, "fl"),
        (1, -1, "br"),
        (-1, -1, "bl"),
    ):
        pname = nm(rng, f"corner_{tag}")
        posts.append(pname)
        calls.append(
            prim(
                pname,
                "box",
                {"size": [post, sy, post]},
                [dx * (sx / 2 - post / 2 + proud), sy / 2, dz * (sz / 2 - post / 2 + proud)],
            )
        )
    calls.append(material([ref(body)], jcol(rng, WOOD), "Crate Wood", roughness=0.8))
    calls.append(
        material(
            [ref(p) for p in posts], jcol(rng, IRON), "Corner Iron", roughness=0.4, metallic=0.8
        )
    )
    prompts = [
        "a sturdy wooden shipping crate with metal corner plates",
        f"a wooden crate, about {sx:.2f}m wide, reinforced with iron plates at each corner",
        "wooden shipping crate, iron corner brackets on all four vertical edges",
        "build me a plain shipping crate: wood body, metal corner plating",
        f"a {sx:.2f} by {sy:.2f} by {sz:.2f} metre wooden crate with metal corner plates",
        "an old wooden cargo crate banded with metal at every corner",
    ]
    notes = (
        "Box body plus four thin vertical box posts at the corners standing in for "
        "metal corner plates -- primitives and materials only, no boolean needed."
    )
    return prompts[i], calls, notes, False


def recipe_02(rng, i):
    # easy: plain wooden storage chest with a flat lid
    sx = jitter(rng, 0.7)
    sy = jitter(rng, 0.4)
    sz = jitter(rng, 0.4)
    lid_h = 0.05
    body = nm(rng, "chest_body")
    lid = nm(rng, "chest_lid")
    calls = [
        prim(body, "box", {"size": [sx, sy, sz]}, [0, sy / 2, 0]),
        prim(lid, "box", {"size": [sx * 1.02, lid_h, sz * 1.02]}, [0, sy + lid_h / 2, 0]),
    ]
    calls.append(
        material([ref(body), ref(lid)], jcol(rng, DARK_WOOD), "Chest Wood", roughness=0.75)
    )
    prompts = [
        "a plain wooden storage chest with a flat lid",
        f"a wooden storage chest, {sx:.2f}m long, with a flat hinged-looking lid",
        "simple wooden chest, flat top, no carvings",
        "make a plain storage chest with a flat wooden lid sitting on top",
        "a rectangular wooden chest, flat lid, unadorned",
        f"a {sx:.2f} by {sz:.2f}m wooden chest with a plain flat lid",
    ]
    notes = "Two boxes: a body and a slightly larger flat lid resting on top. Primitives only."
    return prompts[i], calls, notes, False


LATHE_CLAY_POT = (
    (0.0, 0.0),
    (0.20, 0.03),
    (0.30, 0.12),
    (0.28, 0.24),
    (0.16, 0.32),
    (0.18, 0.36),
)


def recipe_03(rng, i):
    # easy: small clay pot, round and squat
    s = jitter(rng, 0.5)
    stations = LATHE_CLAY_POT
    h = profile_height(stations)
    name = nm(rng, "pot")
    calls = [
        prim(
            name,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 20},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        ),
        material([ref(name)], jcol(rng, CLAY_RED), "Fired Clay", roughness=0.85),
    ]
    prompts = [
        "a small clay pot, round and squat",
        f"a squat round clay pot, about {s * h * 100:.0f} cm tall",
        "a little round-bellied clay pot",
        "build a small, squat terracotta pot",
        "a plain round clay pot, wider than it is tall",
        "a modest clay pot with a rounded belly and a small open mouth",
    ]
    notes = (
        "Lathe profile: flat foot, quick rounded bulge, small open mouth -- a squat "
        "silhouette. Scaled uniformly for size variety; translation is scale*height/2 "
        "to stay grounded."
    )
    return prompts[i], calls, notes, False


LATHE_BOTTLE = (
    (0.0, 0.0),
    (0.16, 0.02),
    (0.24, 0.10),
    (0.24, 0.30),
    (0.18, 0.36),
    (0.05, 0.44),
    (0.05, 0.58),
    (0.06, 0.60),
)


def recipe_04(rng, i):
    # easy: glass bottle with a narrow neck (built as two stacked primitives,
    # no lathe -- this one stays purely a primitive-placement build)
    body_r = jitter(rng, 0.14)
    body_h = jitter(rng, 0.28)
    neck_r = jitter(rng, 0.035)
    neck_h = jitter(rng, 0.16)
    body = nm(rng, "bottle_body")
    neck = nm(rng, "bottle_neck")
    calls = [
        prim(
            body,
            "cylinder",
            {"radius": body_r, "height": body_h, "segments": 20},
            [0, body_h / 2, 0],
        ),
        prim(
            neck,
            "cylinder",
            {"radius": neck_r, "height": neck_h, "segments": 16},
            [0, body_h + neck_h / 2, 0],
        ),
        material([ref(body), ref(neck)], jcol(rng, GLASS_GREEN), "Bottle Glass", roughness=0.1),
    ]
    prompts = [
        "a glass bottle with a narrow neck",
        f"a glass bottle, {body_r * 2 * 100:.0f} cm across the body, with a thin neck on top",
        "a simple glass bottle, wide body, narrow neck",
        "build a bottle: a stout cylinder body and a thin cylinder neck",
        "a green glass bottle with a slender neck",
        "a plain glass bottle, round body tapering to a narrow spout",
    ]
    notes = (
        "Two stacked cylinders (wide body, narrow neck) -- primitives and placement "
        "only, no lathe, to keep this one a genuinely easy build."
    )
    return prompts[i], calls, notes, False


def recipe_05(rng, i):
    # easy: metal bucket with a simple handle (torus, standing loop)
    r = jitter(rng, 0.22)
    h = jitter(rng, 0.30)
    body = nm(rng, "bucket_body")
    handle = nm(rng, "bucket_handle")
    handle_r = r * 0.95
    handle_tube = 0.012
    calls = [
        prim(body, "cylinder", {"radius": r, "height": h, "segments": 20}, [0, h / 2, 0]),
        prim(
            handle,
            "torus",
            {"radius": handle_r, "tube": handle_tube, "segments": 20, "sides": 8},
            [0, h + handle_r * 0.55, 0],
            rotation=[90, 0, 0],
        ),
        material([ref(body)], jcol(rng, STEEL), "Bucket Steel", roughness=0.4, metallic=0.8),
        material([ref(handle)], jcol(rng, IRON), "Handle Iron", roughness=0.3, metallic=0.9),
    ]
    prompts = [
        "a metal bucket with a simple handle",
        f"a metal bucket, about {h * 100:.0f} cm tall, with a plain bail handle",
        "a plain steel bucket with a wire loop handle arcing over the top",
        "build a simple metal pail with a handle",
        "a galvanised bucket, cylindrical, with a bail handle",
        "a metal bucket, straight sides, one loop handle",
    ]
    notes = (
        "Cylinder body plus a torus rotated 90 degrees about X so it stands as a bail "
        "handle arcing above the rim -- the torus's own bottom half sinks slightly "
        "into the rim, which is the same overlap the manual's bucket examples accept."
    )
    return prompts[i], calls, notes, False


def recipe_06(rng, i):
    # easy: sci-fi supply crate with rounded corners
    sx = jitter(rng, 0.55)
    sy = jitter(rng, 0.5)
    sz = jitter(rng, 0.45)
    body = nm(rng, "crate_hull")
    trim = nm(rng, "trim_strip")
    calls = [
        prim(body, "box", {"size": [sx, sy, sz]}, [0, sy / 2, 0]),
        prim(trim, "box", {"size": [sx * 1.01, 0.02, sz * 1.01]}, [0, sy * 0.7, 0]),
        material([ref(body)], jcol(rng, SCIFI_GREY), "Hull Plating", roughness=0.35, metallic=0.6),
        material([ref(trim)], jcol(rng, SCIFI_ACCENT), "Accent Light", roughness=0.2, metallic=0.1),
    ]
    prompts = [
        "a sci-fi supply crate with rounded corners",
        f"a futuristic supply crate, {sx:.2f}m wide, with a glowing accent stripe",
        "sci-fi cargo crate, smooth grey hull, one lit trim band",
        "build a sci-fi storage crate with a bright accent stripe near the top",
        "a space-station supply crate, grey plating and a neon stripe",
        "a compact sci-fi crate, boxy hull with a thin glowing band",
    ]
    notes = (
        "Box hull plus a thin box 'accent light' band standing in for the rounded, "
        "lit trim a sci-fi crate reads as -- primitives and materials only."
    )
    return prompts[i], calls, notes, False


def recipe_07(rng, i):
    # easy: woven wicker basket, round and open-topped
    r = jitter(rng, 0.28)
    h = jitter(rng, 0.24)
    body = nm(rng, "basket_body")
    rim = nm(rng, "basket_rim")
    calls = [
        prim(body, "cylinder", {"radius": r, "height": h, "segments": 18}, [0, h / 2, 0]),
        prim(rim, "torus", {"radius": r, "tube": 0.015, "segments": 18, "sides": 8}, [0, h, 0]),
        material([ref(body), ref(rim)], jcol(rng, [0.62, 0.46, 0.24]), "Wicker", roughness=0.9),
    ]
    prompts = [
        "a woven wicker basket, round and open-topped",
        f"a round wicker basket, {r * 2 * 100:.0f} cm across, open on top",
        "a simple round basket, no lid",
        "build an open wicker basket, cylindrical",
        "a woven straw basket, round, rim reinforced at the top",
        "a plain open-top wicker basket",
    ]
    notes = "Cylinder body with a torus rolled-rim at the opening -- primitives only."
    return prompts[i], calls, notes, False


def recipe_08(rng, i):
    # easy: ceramic vase, tall and cylindrical (kept a plain cylinder, easy tier)
    r = jitter(rng, 0.12)
    h = jitter(rng, 0.55)
    body = nm(rng, "vase")
    calls = [
        prim(body, "cylinder", {"radius": r, "height": h, "segments": 20}, [0, h / 2, 0]),
        material([ref(body)], jcol(rng, CERAMIC_WHITE), "Glazed Ceramic", roughness=0.25),
    ]
    prompts = [
        "a ceramic vase, tall and cylindrical",
        f"a tall cylindrical vase, about {h * 100:.0f} cm high",
        "a plain tall vase, straight sides",
        "build a simple cylindrical ceramic vase",
        "a slender, tall ceramic vase with no curves, just a cylinder",
        "a tall straight-sided vase in glazed ceramic",
    ]
    notes = "Deliberately a plain cylinder -- the seed calls it cylindrical, and the family has plenty of lathe-turned vases elsewhere (amphora, urn, potion bottle)."
    return prompts[i], calls, notes, False


def recipe_09(rng, i):
    # easy: simple burlap sack, rendered as a rounded box
    sx = jitter(rng, 0.35)
    sy = jitter(rng, 0.45)
    sz = jitter(rng, 0.32)
    body = nm(rng, "sack")
    calls = [
        prim(body, "box", {"size": [sx, sy, sz]}, [0, sy / 2, 0]),
        material([ref(body)], jcol(rng, BURLAP), "Burlap", roughness=0.95),
    ]
    prompts = [
        "a simple burlap sack, rendered as a rounded box",
        f"a burlap sack, roughly {sx:.2f} by {sy:.2f}m, boxy silhouette",
        "a plain sack, blocked out as a box",
        "build a simple sack shape -- just a soft-looking box",
        "a burlap sack standing upright, box-shaped",
        "a coarse cloth sack, kept as a simple box",
    ]
    notes = "A single box, as the prompt itself asks for -- primitives only."
    return prompts[i], calls, notes, False


def recipe_10(rng, i):
    # easy: treasure chest with a domed lid
    sx = jitter(rng, 0.55)
    sy = jitter(rng, 0.32)
    sz = jitter(rng, 0.35)
    dome_r = sx / 2 * 1.02
    body = nm(rng, "chest_body")
    dome = nm(rng, "chest_dome")
    calls = [
        prim(body, "box", {"size": [sx, sy, sz]}, [0, sy / 2, 0]),
        prim(
            dome,
            "uv_sphere",
            {"radius": dome_r, "segments": 20, "rings": 10},
            [0, sy, 0],
            scale=[1.0, 0.55, sz / sx * 1.0 if sx else 1.0],
        ),
        material([ref(body)], jcol(rng, DARK_WOOD), "Chest Wood", roughness=0.75),
        material([ref(dome)], jcol(rng, GOLD), "Domed Brass", roughness=0.35, metallic=0.7),
    ]
    prompts = [
        "a treasure chest with a domed lid",
        f"a treasure chest, {sx:.2f}m wide, topped with a rounded brass dome",
        "a fantasy treasure chest, arched dome lid",
        "build a chest with a domed metal lid",
        "an old treasure chest, its lid a shallow brass dome",
        "a wooden chest capped with a rounded dome",
    ]
    notes = (
        "Box body plus a flattened uv_sphere sitting on top for the dome -- the "
        "sphere's lower half is hidden inside/against the box top, which is the same "
        "'only the visible half matters' shortcut the manual's own bucket-handle "
        "reasoning uses."
    )
    return prompts[i], calls, notes, False


def recipe_11(rng, i):
    # easy: stack of three identical wooden crates
    s = jitter(rng, 0.35)
    names = [nm(rng, f"crate_{k}") for k in range(3)]
    calls = [
        prim(names[0], "box", {"size": [s, s, s]}, [0, s / 2, 0]),
        prim(names[1], "box", {"size": [s, s, s]}, [s * 1.05, s / 2, 0]),
        prim(names[2], "box", {"size": [s, s, s]}, [s * 0.5, s * 1.5, 0]),
        material([ref(n) for n in names], jcol(rng, WOOD), "Crate Wood", roughness=0.8),
    ]
    prompts = [
        "a stack of three identical wooden crates",
        f"three matching wooden crates, {s:.2f}m cubes, two side by side and one on top",
        "a small pile of three wooden crates",
        "build three identical crates stacked together",
        "a stack of wooden shipping crates, three of them",
        "three cube crates arranged in a stack",
    ]
    notes = "Three identical boxes placed by hand (two side-by-side, one on top) -- no array op needed for just three."
    return prompts[i], calls, notes, False


LATHE_BARREL = (
    (0.0, 0.0),
    (0.24, 0.02),
    (0.30, 0.10),
    (0.32, 0.26),
    (0.30, 0.42),
    (0.24, 0.50),
    (0.0, 0.52),
)


def recipe_12(rng, i):
    # easy: large storage barrel standing upright (kept straight, no bulge, easy tier)
    r = jitter(rng, 0.32)
    h = jitter(rng, 0.62)
    body = nm(rng, "barrel_body")
    hoop_top = nm(rng, "hoop_top")
    hoop_bot = nm(rng, "hoop_bottom")
    calls = [
        prim(body, "cylinder", {"radius": r, "height": h, "segments": 22}, [0, h / 2, 0]),
        prim(
            hoop_top,
            "torus",
            {"radius": r * 1.02, "tube": 0.015, "segments": 22, "sides": 8},
            [0, h * 0.85, 0],
        ),
        prim(
            hoop_bot,
            "torus",
            {"radius": r * 1.02, "tube": 0.015, "segments": 22, "sides": 8},
            [0, h * 0.15, 0],
        ),
        material([ref(body)], jcol(rng, WOOD), "Barrel Oak", roughness=0.7),
        material(
            [ref(hoop_top), ref(hoop_bot)],
            jcol(rng, IRON),
            "Iron Hoop",
            roughness=0.4,
            metallic=0.85,
        ),
    ]
    prompts = [
        "a large storage barrel standing upright",
        f"a tall storage barrel, {h * 100:.0f} cm high, with two iron hoops",
        "a plain upright barrel, straight sides, banded near top and bottom",
        "build a large wooden storage barrel",
        "a big barrel standing on end, iron-hooped",
        "a straight-sided wooden storage barrel",
    ]
    notes = "Straight cylinder plus two torus hoops -- kept unbulged for the easy tier; the bulging lathe barrel lives in the wine-barrel record."
    return prompts[i], calls, notes, False


def recipe_13(rng, i):
    # easy: small tin can, plain cylinder
    r = jitter(rng, 0.06)
    h = jitter(rng, 0.11)
    body = nm(rng, "can")
    calls = [
        prim(body, "cylinder", {"radius": r, "height": h, "segments": 16}, [0, h / 2, 0]),
        material([ref(body)], jcol(rng, STEEL), "Tin", roughness=0.3, metallic=0.9),
    ]
    prompts = [
        "a small tin can, plain cylinder",
        f"a tin can, {r * 2 * 100:.0f} cm across, {h * 100:.0f} cm tall",
        "a plain metal can, nothing on it",
        "build a small tin can",
        "a bare tin can, cylindrical",
        "a simple food tin, plain cylinder shape",
    ]
    notes = "One plain cylinder -- as simple as this family gets."
    return prompts[i], calls, notes, False


def recipe_14(rng, i):
    # easy: flowerpot, a truncated cone shape (needs a 2-station lathe since
    # the registry has no frustum primitive -- 'cone' only reaches a point)
    r_bot = jitter(rng, 0.09)
    r_top = jitter(rng, 0.16)
    h = jitter(rng, 0.18)
    body = nm(rng, "flowerpot")
    stations = [[r_bot, 0.0], [r_top, h]]
    calls = [
        prim(body, "lathe", {"profile": stations, "segments": 18}, [0, h / 2, 0]),
        material([ref(body)], jcol(rng, TERRACOTTA), "Terracotta", roughness=0.8),
    ]
    prompts = [
        "a flowerpot, a truncated cone shape",
        f"a terracotta flowerpot, {h * 100:.0f} cm tall, narrower at the base",
        "a simple flowerpot that widens toward the top",
        "build a truncated-cone flowerpot",
        "a plain flowerpot, tapered like a cut-off cone",
        "a small tapering flowerpot, narrow foot, wide mouth",
    ]
    notes = (
        "A frustum has no dedicated generator, so this is a two-station lathe "
        "(narrow bottom radius, wide top radius) -- the minimum profile the "
        "registry accepts, which still counts as a real lathe use."
    )
    return prompts[i], calls, notes, False


def recipe_15(rng, i):
    # easy: canteen, a flattened cylinder with a cap
    r = jitter(rng, 0.14)
    thickness = r * 0.7
    cap_r = jitter(rng, 0.02)
    cap_h = jitter(rng, 0.03)
    body = nm(rng, "canteen_body")
    cap = nm(rng, "canteen_cap")
    calls = [
        # Cylinder's own axis is Y; rotating 90 degrees about X swaps its
        # flat circular caps to face +/-Z, so the round profile reads
        # front-on and 'height' (now thickness) becomes the flattened depth.
        prim(
            body,
            "cylinder",
            {"radius": r, "height": thickness, "segments": 20},
            [0, r, 0],
            rotation=[90, 0, 0],
        ),
        prim(
            cap,
            "cylinder",
            {"radius": cap_r, "height": cap_h, "segments": 12},
            [0, 2 * r - 0.01 + cap_h / 2, 0],
        ),
        material([ref(body)], jcol(rng, STEEL), "Canteen Steel", roughness=0.35, metallic=0.7),
        material([ref(cap)], jcol(rng, IRON), "Cap", roughness=0.4, metallic=0.6),
    ]
    prompts = [
        "a canteen, a flattened cylinder with a cap",
        f"a metal canteen, about {r * 2 * 100:.0f} cm across, flattened, with a small cap",
        "a flask-like canteen, squashed cylinder shape with a screw cap",
        "build a canteen: a flattened cylinder and a small cap on the side",
        "a soldier's canteen, flattened round body, small cap",
        "a simple canteen, disc-shaped body, cap near the rim",
    ]
    notes = (
        "A cylinder rotated onto its side and scaled thin along its former axis to "
        "flatten it (a disc-like canteen body), plus a small cylinder cap -- "
        "primitives, rotation and scale only."
    )
    return prompts[i], calls, notes, False


LATHE_POTION = (
    (0.0, 0.0),
    (0.10, 0.02),
    (0.15, 0.10),
    (0.14, 0.20),
    (0.05, 0.26),
    (0.045, 0.34),
)


def recipe_16(rng, i):
    # easy: fantasy potion bottle, small and round-bodied
    s = jitter(rng, 0.55)
    stations = LATHE_POTION
    h = profile_height(stations)
    body = nm(rng, "potion_bottle")
    calls = [
        prim(
            body,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 16},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        ),
        material([ref(body)], jcol(rng, [0.5, 0.15, 0.55]), "Potion Glass", roughness=0.15),
    ]
    prompts = [
        "a fantasy potion bottle, small and round-bodied",
        f"a small potion bottle, round-bellied, about {s * h * 100:.0f} cm tall",
        "a tiny round potion flask",
        "build a small fantasy potion bottle with a round body",
        "a stubby round-bodied potion bottle, narrow neck",
        "a little round potion vial",
    ]
    notes = "Lathe: round bulb body narrowing to a short thin neck -- distinct from the tavern bottle profile."
    return prompts[i], calls, notes, False


def recipe_17(rng, i):
    # easy: modern plastic storage bin with a flat lid
    sx = jitter(rng, 0.5)
    sy = jitter(rng, 0.32)
    sz = jitter(rng, 0.38)
    lid_h = 0.03
    body = nm(rng, "bin_body")
    lid = nm(rng, "bin_lid")
    calls = [
        prim(body, "box", {"size": [sx, sy, sz]}, [0, sy / 2, 0]),
        prim(lid, "box", {"size": [sx * 1.03, lid_h, sz * 1.03]}, [0, sy + lid_h / 2, 0]),
        material([ref(body)], jcol(rng, PLASTIC_BLUE), "Storage Plastic", roughness=0.3),
        material([ref(lid)], jcol(rng, [0.85, 0.85, 0.85]), "Lid Plastic", roughness=0.3),
    ]
    prompts = [
        "a modern plastic storage bin with a flat lid",
        f"a plastic storage bin, {sx:.2f}m wide, snap-on flat lid",
        "a household plastic tote with a flat lid",
        "build a modern storage bin with a flat lid",
        "a blue plastic bin with a light grey flat lid",
        "a simple modern storage tote, flat-lidded",
    ]
    notes = "Same two-box shape as the wooden chest, differentiated entirely by proportion and material -- modern plastic bin."
    return prompts[i], calls, notes, False


def recipe_18(rng, i):
    # easy: rain barrel, a wide squat cylinder
    r = jitter(rng, 0.34)
    h = jitter(rng, 0.30)
    body = nm(rng, "rain_barrel")
    hoop = nm(rng, "hoop")
    calls = [
        prim(body, "cylinder", {"radius": r, "height": h, "segments": 22}, [0, h / 2, 0]),
        prim(
            hoop,
            "torus",
            {"radius": r * 1.02, "tube": 0.014, "segments": 22, "sides": 8},
            [0, h * 0.5, 0],
        ),
        material([ref(body)], jcol(rng, WOOD), "Barrel Wood", roughness=0.75),
        material([ref(hoop)], jcol(rng, IRON), "Hoop", roughness=0.45, metallic=0.8),
    ]
    prompts = [
        "a rain barrel, a wide squat cylinder",
        f"a rain barrel, {r * 2 * 100:.0f} cm across and only {h * 100:.0f} cm tall",
        "a wide, low rain barrel",
        "build a squat rain barrel with one hoop",
        "a short, wide wooden rain barrel",
        "a low, broad barrel for collecting rainwater",
    ]
    notes = "Wide, short cylinder plus a single hoop -- primitives only."
    return prompts[i], calls, notes, False


def recipe_19(rng, i):
    # easy: medieval grain sack, a soft rounded box shape (capsule instead)
    r = jitter(rng, 0.20)
    h = jitter(rng, 0.34)
    body = nm(rng, "grain_sack")
    calls = [
        prim(
            body,
            "capsule",
            {"radius": r, "height": h, "segments": 16, "rings": 4},
            [0, h / 2 + r, 0],
        ),
        material([ref(body)], jcol(rng, BURLAP), "Grain Sack Cloth", roughness=0.9),
    ]
    prompts = [
        "a medieval grain sack, a soft rounded box shape",
        f"a grain sack, round and soft, about {(h + 2 * r) * 100:.0f} cm tall",
        "a plump, rounded sack of grain",
        "build a soft, rounded grain sack",
        "a medieval sack, bulging and rounded rather than boxy",
        "a full sack of grain, rounded at both ends",
    ]
    notes = (
        "Used a capsule rather than a literal box: its rounded ends read as the "
        "bulging cloth of a full sack better than a flat-faced box would."
    )
    return prompts[i], calls, notes, False


def recipe_20(rng, i):
    # easy: metal ammo box, plain rectangular container
    sx = jitter(rng, 0.4)
    sy = jitter(rng, 0.22)
    sz = jitter(rng, 0.28)
    body = nm(rng, "ammo_box")
    calls = [
        prim(body, "box", {"size": [sx, sy, sz]}, [0, sy / 2, 0]),
        material(
            [ref(body)], jcol(rng, [0.24, 0.30, 0.20]), "Ammo Steel", roughness=0.4, metallic=0.6
        ),
    ]
    prompts = [
        "a metal ammo box, a plain rectangular container",
        f"a metal ammunition box, {sx:.2f}m long",
        "a plain olive-drab ammo crate",
        "build a simple rectangular metal ammo box",
        "an army surplus ammo box, plain steel",
        "a basic rectangular ammo tin",
    ]
    notes = "A single box, painted steel -- as plain as the prompt asks."
    return prompts[i], calls, notes, False


def recipe_21(rng, i):
    # easy: cardboard moving box, a simple cube
    s = jitter(rng, 0.42)
    body = nm(rng, "moving_box")
    calls = [
        prim(body, "box", {"size": [s, s, s]}, [0, s / 2, 0]),
        material([ref(body)], jcol(rng, CARDBOARD), "Cardboard", roughness=0.95),
    ]
    prompts = [
        "a cardboard moving box, a simple cube",
        f"a cardboard box, {s:.2f}m cube",
        "a plain cardboard moving box",
        "build a simple cube cardboard box",
        "a moving box, cube-shaped, plain cardboard",
        "a basic square cardboard carton",
    ]
    notes = "A single cube box -- primitives only."
    return prompts[i], calls, notes, False


LATHE_JUG = (
    (0.0, 0.0),
    (0.14, 0.02),
    (0.20, 0.14),
    (0.19, 0.26),
    (0.09, 0.34),
    (0.10, 0.40),
    (0.07, 0.43),
)


def recipe_22(rng, i):
    # easy: ceramic jug with a small spout (kept to a lathe body + handle,
    # no separate spout piece, per the class-22 discussion)
    s = jitter(rng, 0.62)
    stations = LATHE_JUG
    h = profile_height(stations)
    body = nm(rng, "jug_body")
    handle = nm(rng, "jug_handle")
    handle_r = 0.05 * s
    handle_tube = 0.009 * s
    calls = [
        prim(
            body,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 18},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        ),
        prim(
            handle,
            "torus",
            {"radius": handle_r, "tube": handle_tube, "segments": 16, "sides": 8},
            [0.17 * s, s * h * 0.55, 0],
            rotation=[0, 0, 90],
        ),
        material([ref(body)], jcol(rng, TERRACOTTA), "Jug Ceramic", roughness=0.5),
        material([ref(handle)], jcol(rng, TERRACOTTA), "Jug Ceramic Handle", roughness=0.5),
    ]
    prompts = [
        "a ceramic jug with a small spout",
        f"a ceramic jug, {s * h * 100:.0f} cm tall, with a side handle",
        "a round-bodied jug with a narrow spout and a loop handle",
        "build a ceramic jug with a handle on the side",
        "a simple terracotta jug, narrow-necked, one handle",
        "a household ceramic jug with a small pour spout",
    ]
    notes = (
        "Lathe jug body (the narrow neck itself reads as the spout) plus a torus "
        "handle rotated to stand vertically against the belly."
    )
    return prompts[i], calls, notes, False


def recipe_23(rng, i):
    # easy: fuel drum, a tall metal cylinder
    r = jitter(rng, 0.30)
    h = jitter(rng, 0.85)
    body = nm(rng, "fuel_drum")
    rib_top = nm(rng, "rib_top")
    rib_bot = nm(rng, "rib_bottom")
    calls = [
        prim(body, "cylinder", {"radius": r, "height": h, "segments": 22}, [0, h / 2, 0]),
        prim(
            rib_top,
            "torus",
            {"radius": r * 1.01, "tube": 0.012, "segments": 22, "sides": 6},
            [0, h * 0.82, 0],
        ),
        prim(
            rib_bot,
            "torus",
            {"radius": r * 1.01, "tube": 0.012, "segments": 22, "sides": 6},
            [0, h * 0.18, 0],
        ),
        material(
            [ref(body)], jcol(rng, [0.55, 0.15, 0.12]), "Drum Steel", roughness=0.45, metallic=0.5
        ),
        material(
            [ref(rib_top), ref(rib_bot)], jcol(rng, IRON), "Drum Ribs", roughness=0.4, metallic=0.7
        ),
    ]
    prompts = [
        "a fuel drum, a tall metal cylinder",
        f"a tall fuel drum, {h * 100:.0f} cm high, with rolled steel ribs",
        "an oil drum, tall and cylindrical",
        "build a tall metal fuel drum",
        "a red steel fuel drum with two rolled ribs",
        "a standard tall fuel drum",
    ]
    notes = "Tall cylinder plus two thin torus ribs standing in for the drum's rolled bands."
    return prompts[i], calls, notes, False


def recipe_24(rng, i):
    # easy: woven basket holding a few round apples
    r = jitter(rng, 0.24)
    h = jitter(rng, 0.18)
    basket = nm(rng, "basket")
    calls = [prim(basket, "cylinder", {"radius": r, "height": h, "segments": 18}, [0, h / 2, 0])]
    apple_r = jitter(rng, 0.045, 0.1)
    apples = []
    positions = [(0.0, 0.0), (0.09, 0.06), (-0.08, 0.07), (0.02, -0.09)]
    for k, (dx, dz) in enumerate(positions):
        aname = nm(rng, f"apple_{k}")
        apples.append(aname)
        calls.append(
            prim(
                aname,
                "uv_sphere",
                {"radius": apple_r, "segments": 14, "rings": 8},
                [dx * r * 2, h + apple_r * 0.8, dz * r * 2],
            )
        )
    calls.append(material([ref(basket)], jcol(rng, [0.6, 0.44, 0.22]), "Wicker", roughness=0.9))
    calls.append(
        material(
            [ref(a) for a in apples], jcol(rng, [0.62, 0.10, 0.10]), "Apple Skin", roughness=0.35
        )
    )
    prompts = [
        "a woven basket holding a few round apples",
        f"a wicker basket, {r * 2 * 100:.0f} cm across, with four apples piled in it",
        "a basket with apples heaped inside",
        "build a basket full of round apples",
        "a round wicker basket with a small pile of apples",
        "a basket of apples, four fruit resting on top",
    ]
    notes = "Basket cylinder plus four uv_sphere apples placed by hand near the rim, no array op needed for so few."
    return prompts[i], calls, notes, False


LATHE_CAULDRON = (
    (0.0, 0.0),
    (0.10, 0.02),
    (0.28, 0.14),
    (0.30, 0.24),
    (0.22, 0.34),
    (0.20, 0.36),
)


def recipe_25(rng, i):
    # easy: squat iron cauldron standing on three short feet
    s = jitter(rng, 0.55)
    stations = LATHE_CAULDRON
    h = profile_height(stations)
    body = nm(rng, "cauldron")
    calls = [
        prim(
            body,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 20},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        )
    ]
    foot_h = 0.05 * s
    foot_r = 0.025 * s
    import math

    feet = []
    for k in range(3):
        ang = math.radians(90 + k * 120)
        fx = math.cos(ang) * 0.18 * s
        fz = math.sin(ang) * 0.18 * s
        fname = nm(rng, f"foot_{k}")
        feet.append(fname)
        calls.append(
            prim(
                fname,
                "cone",
                {"radius": foot_r, "height": foot_h, "segments": 10},
                [fx, foot_h / 2, fz],
            )
        )
    calls.append(material([ref(body)], jcol(rng, IRON), "Cast Iron", roughness=0.55, metallic=0.65))
    calls.append(
        material(
            [ref(f) for f in feet], jcol(rng, IRON), "Cast Iron Feet", roughness=0.55, metallic=0.65
        )
    )
    # raise the whole thing so it stands on its three feet rather than its
    # own rounded base -- the feet are the ground contact, so the body is
    # lifted by the foot height.
    calls[0]["arguments"]["translation"][1] = round(s * h / 2 + foot_h, 4)
    prompts = [
        "a squat iron cauldron standing on three short feet",
        f"a squat cast-iron cauldron, {s * h * 100:.0f} cm tall, on three stubby feet",
        "an old cauldron, wide-bellied, resting on three small feet",
        "build a squat cauldron standing on three low feet",
        "a wide iron cauldron on tripod feet",
        "a squat black cauldron standing on three cone feet",
    ]
    notes = (
        "Lathe cauldron bowl (wide belly, in-turned rim) lifted by the foot height "
        "and set on three cone feet placed by hand at 120-degree intervals -- no "
        "array op for just three legs."
    )
    return prompts[i], calls, notes, False


# --------------------------------------------------------------------------
# MEDIUM (26-45)
# --------------------------------------------------------------------------

LATHE_WINE_BARREL = (
    (0.0, 0.0),
    (0.22, 0.02),
    (0.34, 0.18),
    (0.36, 0.32),
    (0.34, 0.46),
    (0.22, 0.62),
    (0.0, 0.64),
)


def recipe_26(rng, i):
    # medium: wine barrel, lathe-turned bulging sides + metal bands + rivets
    s = jitter(rng, 0.7)
    stations = LATHE_WINE_BARREL
    h = profile_height(stations)
    body = nm(rng, "barrel_body")
    hoop_mid = nm(rng, "hoop_mid")
    hoop_top = nm(rng, "hoop_top")
    hoop_bot = nm(rng, "hoop_bottom")
    rivet = nm(rng, "rivet")
    count = rng.choice([12, 16, 20])
    calls = [
        prim(
            body,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 24},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        ),
        prim(
            hoop_top,
            "torus",
            {"radius": 0.30 * s, "tube": 0.016 * s, "segments": 24, "sides": 8},
            [0, s * h * 0.86, 0],
        ),
        prim(
            hoop_mid,
            "torus",
            {"radius": 0.36 * s, "tube": 0.018 * s, "segments": 24, "sides": 8},
            [0, s * h * 0.5, 0],
        ),
        prim(
            hoop_bot,
            "torus",
            {"radius": 0.30 * s, "tube": 0.016 * s, "segments": 24, "sides": 8},
            [0, s * h * 0.14, 0],
        ),
        prim(rivet, "box", {"size": [0.02 * s, 0.02 * s, 0.015 * s]}, [0.36 * s, s * h * 0.5, 0]),
        select([ref(rivet)]),
        op("array-radial", {"count": count, "angle": 360, "axis": 1}),
    ]
    calls.append(material([ref(body)], jcol(rng, WOOD), "Barrel Oak", roughness=0.7))
    calls.append(
        material(
            [ref(hoop_top), ref(hoop_mid), ref(hoop_bot)],
            jcol(rng, IRON),
            "Barrel Hoops",
            roughness=0.4,
            metallic=0.85,
        )
    )
    calls.append(
        material(
            dup_refs(rivet, count), jcol(rng, IRON), "Hoop Rivets", roughness=0.35, metallic=0.9
        )
    )
    prompts = [
        "a wine barrel with lathe-turned bulging sides and metal bands",
        f"a wine barrel, {s * h * 100:.0f} cm tall, bulging sides, three iron hoops and a ring of rivets on the middle band",
        "a classic wine barrel: turned, bulging profile, banded with iron and studded with rivets",
        "build a bulging wine barrel with hoops and a radial ring of rivets on the centre band",
        "a wooden wine barrel, curved sides, iron hoops top and bottom, rivet studs around the middle",
        "a barrel with a real turned bulge, three metal hoops, and rivets ringing the widest band",
    ]
    notes = (
        "Lathe profile bulges out to its widest at the middle and tapers to both rims. "
        "Hoops are torus primitives (the manual's own suggestion for hoops and "
        "handles); the middle hoop's rivets are one box selected then "
        "array-radial'd around the barrel's own axis -- the array-radial 'ring of "
        "rivets' case the family brief calls out."
    )
    return prompts[i], calls, notes, False


LATHE_AMPHORA = (
    (0.0, 0.0),
    (0.03, 0.02),
    (0.16, 0.10),
    (0.24, 0.24),
    (0.22, 0.38),
    (0.10, 0.48),
    (0.11, 0.52),
    (0.13, 0.56),
)


def recipe_27(rng, i):
    # medium: ceramic amphora with a lathe-turned neck + two mirrored handles
    s = jitter(rng, 0.6)
    stations = LATHE_AMPHORA
    h = profile_height(stations)
    body = nm(rng, "amphora_body")
    handle_l = nm(rng, "handle_left")
    handle_r = nm(rng, "handle_right")
    handle_radius = 0.075 * s
    handle_tube = 0.012 * s
    attach_y = s * h * 0.62
    attach_x = 0.15 * s
    calls = [
        prim(
            body,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 20},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        ),
        prim(
            handle_l,
            "torus",
            {"radius": handle_radius, "tube": handle_tube, "segments": 16, "sides": 8},
            [attach_x, attach_y, 0],
            rotation=[0, 0, 90],
        ),
        prim(
            handle_r,
            "torus",
            {"radius": handle_radius, "tube": handle_tube, "segments": 16, "sides": 8},
            [-attach_x, attach_y, 0],
            rotation=[0, 0, 90],
        ),
        material([ref(body)], jcol(rng, TERRACOTTA), "Amphora Clay", roughness=0.55),
        material(
            [ref(handle_l), ref(handle_r)], jcol(rng, TERRACOTTA), "Amphora Handles", roughness=0.55
        ),
    ]
    prompts = [
        "a ceramic amphora with a lathe-turned neck and two mirrored side handles",
        f"an amphora, {s * h * 100:.0f} cm tall, pointed base, a long turned neck and two handles",
        "a classical amphora with a narrow turned neck and symmetric handles either side",
        "build an amphora: pointed foot, turned neck, two mirrored handles",
        "a terracotta amphora, slender neck, matching handles on both sides",
        "an ancient-style amphora with two loop handles, mirrored left and right",
    ]
    notes = (
        "Lathe profile: near-pointed foot, wide belly, a genuinely turned narrow "
        "neck. Two torus handles placed at mirrored +/-x rather than one mirrored "
        "via clay_op -- the two calls are cheap and the record stays 'build' only."
    )
    return prompts[i], calls, notes, False


def recipe_28(rng, i):
    # medium: treasure chest with a round keyhole cut through the lock plate
    sx = jitter(rng, 0.55)
    sy = jitter(rng, 0.32)
    sz = jitter(rng, 0.35)
    lid_h = 0.045
    plate_w = 0.08
    plate_h = 0.1
    plate_t = 0.015
    hole_r = jitter(rng, 0.014, 0.15)
    body = nm(rng, "chest_body")
    lid = nm(rng, "chest_lid")
    plate = nm(rng, "lock_plate")
    cutter = nm(rng, "keyhole_cutter")
    calls = [
        prim(body, "box", {"size": [sx, sy, sz]}, [0, sy / 2, 0]),
        prim(lid, "box", {"size": [sx * 1.02, lid_h, sz * 1.02]}, [0, sy + lid_h / 2, 0]),
        prim(
            plate,
            "box",
            {"size": [plate_w, plate_h, plate_t]},
            [0, sy * 0.45, sz / 2 + plate_t / 2 - 0.002],
        ),
        prim(
            cutter,
            "cylinder",
            {"radius": hole_r, "height": plate_t * 3, "segments": 16},
            [0, sy * 0.45, sz / 2 + plate_t / 2 - 0.002],
            rotation=[90, 0, 0],
        ),
        boolean("difference", [ref(plate), ref(cutter)]),
    ]
    calls.append(
        material([ref(body), ref(lid)], jcol(rng, DARK_WOOD), "Chest Wood", roughness=0.75)
    )
    calls.append(material([ref(plate)], jcol(rng, IRON), "Lock Plate", roughness=0.4, metallic=0.8))
    prompts = [
        "a treasure chest with a round keyhole cut through the lock plate",
        f"a wooden chest with an iron lock plate, a round {hole_r * 200:.1f} cm keyhole bored through it",
        "a chest whose lock plate has a round hole cut clean through",
        "build a chest with a keyhole cut through its lock plate",
        "an old chest, iron lock plate with a round keyhole drilled through",
        "a treasure chest, its front plate pierced with a round keyhole",
    ]
    notes = (
        "Lock plate added before the cutter so it survives the boolean (the "
        "survivor is whichever object comes first in document order); a cylinder, "
        "its axis rotated to run through the plate's thickness, is subtracted to "
        "punch the round keyhole."
    )
    return prompts[i], calls, notes, False


def recipe_29(rng, i):
    # medium: glass jar with a boolean-cut opening for a threaded lid
    r = jitter(rng, 0.10)
    h = jitter(rng, 0.16)
    neck_r = jitter(rng, 0.055, 0.1)
    body = nm(rng, "jar_body")
    cutter = nm(rng, "jar_opening_cutter")
    calls = [
        prim(body, "cylinder", {"radius": r, "height": h, "segments": 20}, [0, h / 2, 0]),
        prim(
            cutter,
            "cylinder",
            {"radius": neck_r, "height": h * 0.4, "segments": 20},
            [0, h - h * 0.2 + 0.001, 0],
        ),
        boolean("difference", [ref(body), ref(cutter)]),
        material([ref(body)], jcol(rng, GLASS_CLEAR), "Jar Glass", roughness=0.08),
    ]
    prompts = [
        "a glass jar with a boolean-cut opening for a threaded lid",
        f"a glass jar, {r * 2 * 100:.0f} cm across, with its mouth bored open for a lid",
        "a jar with the top cut away to leave an open threaded mouth",
        "build a glass jar with a cut opening at the top for a lid",
        "a storage jar, mouth cut open at the top",
        "a jar whose top has been hollowed out into an open neck",
    ]
    notes = (
        "A narrower cylinder subtracted from the jar's own top, positioned to just "
        "overlap the top face, carving a recessed open mouth -- the difference "
        "survivor is the jar body, added first."
    )
    return prompts[i], calls, notes, False


def recipe_30(rng, i):
    # medium: wooden barrel with a round hole cut in the lid for pouring
    s = jitter(rng, 0.65)
    stations = LATHE_BARREL
    h = profile_height(stations)
    body = nm(rng, "barrel")
    cutter = nm(rng, "bunghole_cutter")
    hole_r = jitter(rng, 0.03, 0.2)
    top_y = s * h
    calls = [
        prim(
            body,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 22},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        ),
        prim(
            cutter,
            "cylinder",
            {"radius": hole_r, "height": 0.06 * s, "segments": 16},
            [0, top_y - 0.02 * s, 0],
        ),
        boolean("difference", [ref(body), ref(cutter)]),
        material([ref(body)], jcol(rng, WOOD), "Barrel Oak", roughness=0.7),
    ]
    prompts = [
        "a wooden barrel with a round hole cut in the lid for pouring",
        f"a barrel, {s * h * 100:.0f} cm tall, with a {hole_r * 200:.1f} cm pour-hole bored into the top",
        "a barrel with its top bored out for pouring",
        "build a barrel with a round hole cut into its top for pouring",
        "a wine barrel, a round bunghole cut into the top",
        "a wooden barrel, its lid pierced with a round pouring hole",
    ]
    notes = (
        "The family brief's own worked example: a cylinder subtracted from a "
        "barrel's top, positioned to just dip into the lathe body from above so the "
        "boolean carves a real recess rather than a hole through empty air."
    )
    return prompts[i], calls, notes, False


LATHE_GOBLET_V2 = (
    (0.0, 0.0),
    (0.10, 0.02),
    (0.10, 0.04),
    (0.035, 0.10),
    (0.035, 0.32),
    (0.30, 0.42),
    (0.28, 0.46),
    (0.34, 0.50),
)


def recipe_31(rng, i):
    # medium: lathe-turned wine goblet, wide bowl + thin stem
    s = jitter(rng, 0.55)
    stations = LATHE_GOBLET_V2
    h = profile_height(stations)
    body = nm(rng, "goblet")
    calls = [
        prim(
            body,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 20},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        ),
        material([ref(body)], jcol(rng, GOLD), "Goblet Gold", roughness=0.25, metallic=0.7),
    ]
    prompts = [
        "a lathe-turned wine goblet with a wide bowl and a thin stem",
        f"a wine goblet, {s * h * 100:.0f} cm tall, wide shallow bowl on a thin stem",
        "a goblet: broad open bowl, thin turned stem, flat foot",
        "build a lathe-turned goblet with a wide bowl",
        "a golden goblet, wide bowl, slender stem",
        "a turned wine goblet, generous bowl, narrow stem",
    ]
    notes = (
        "A goblet profile of my own (distinct stations from the registry's own "
        "default goblet profile, kept for variety): flat foot, thin stem, then a "
        "wide, shallow bowl rather than the default's deep one."
    )
    return prompts[i], calls, notes, False


def recipe_32(rng, i):
    # medium: chest with a hinge cut-out along the back edge
    sx = jitter(rng, 0.55)
    sy = jitter(rng, 0.32)
    sz = jitter(rng, 0.35)
    lid_h = 0.045
    body = nm(rng, "chest_body")
    lid = nm(rng, "chest_lid")
    cutter = nm(rng, "hinge_notch_cutter")
    notch_w = jitter(rng, 0.1, 0.15)
    calls = [
        prim(body, "box", {"size": [sx, sy, sz]}, [0, sy / 2, 0]),
        prim(lid, "box", {"size": [sx * 1.02, lid_h, sz * 1.02]}, [0, sy + lid_h / 2, 0]),
        prim(cutter, "box", {"size": [notch_w, lid_h * 1.4, 0.03]}, [0, sy + lid_h / 2, -sz / 2]),
        boolean("difference", [ref(lid), ref(cutter)]),
        material([ref(body), ref(lid)], jcol(rng, DARK_WOOD), "Chest Wood", roughness=0.75),
    ]
    prompts = [
        "a chest with a hinge cut-out along the back edge",
        f"a wooden chest, its lid notched along the back edge for a {notch_w * 100:.0f} cm hinge",
        "a chest whose lid has a hinge notch cut into the back",
        "build a chest with a hinge cutout along the lid's back edge",
        "a storage chest, back edge of the lid notched out for hinges",
        "a chest lid with a cut-out channel along its rear edge",
    ]
    notes = (
        "A thin box straddling the lid's back edge, subtracted from the lid, cuts "
        "the hinge notch -- lid added before the cutter so it survives the boolean."
    )
    return prompts[i], calls, notes, False


LATHE_URN = (
    (0.0, 0.0),
    (0.10, 0.02),
    (0.24, 0.14),
    (0.26, 0.28),
    (0.16, 0.42),
    (0.13, 0.48),
    (0.22, 0.54),
)


def recipe_33(rng, i):
    # medium: decorative urn, lathe-turned with a flared lip
    s = jitter(rng, 0.6)
    stations = LATHE_URN
    h = profile_height(stations)
    body = nm(rng, "urn")
    calls = [
        prim(
            body,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 22},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        ),
        material([ref(body)], jcol(rng, STONE), "Urn Stone", roughness=0.6),
    ]
    prompts = [
        "a decorative urn, lathe-turned with a flared lip",
        f"a decorative urn, {s * h * 100:.0f} cm tall, with a lip that flares outward",
        "an ornamental urn, bulging body, wide flared rim",
        "build a decorative lathe-turned urn with a flared lip",
        "a stone urn with a strongly flared mouth",
        "an urn, its rim flaring out wider than the body below it",
    ]
    notes = "Lathe: bulging body, in-turned neck, then a final station wider again -- the flare -- for the lip."
    return prompts[i], calls, notes, False


LATHE_STOPPER = (
    (0.0, 0.0),
    (0.05, 0.01),
    (0.055, 0.05),
    (0.03, 0.075),
    (0.0, 0.09),
)


def recipe_34(rng, i):
    # medium: fantasy potion bottle with a lathe-turned stopper
    s = jitter(rng, 0.55)
    stations = LATHE_POTION
    h = profile_height(stations)
    body = nm(rng, "potion_bottle")
    stopper_stations = LATHE_STOPPER
    stopper_h = profile_height(stopper_stations)
    stopper = nm(rng, "stopper")
    neck_top = s * h
    calls = [
        prim(
            body,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 16},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        ),
        prim(
            stopper,
            "lathe",
            {"profile": [list(p) for p in stopper_stations], "segments": 12},
            [0, neck_top + s * stopper_h * 0.4, 0],
            scale=[s, s, s],
        ),
        material([ref(body)], jcol(rng, [0.15, 0.45, 0.35]), "Potion Glass", roughness=0.15),
        material([ref(stopper)], jcol(rng, DARK_WOOD), "Cork Stopper", roughness=0.85),
    ]
    prompts = [
        "a fantasy potion bottle with a lathe-turned stopper",
        f"a potion bottle, {s * h * 100:.0f} cm tall, with a turned wooden stopper in the neck",
        "a round potion bottle capped with a corked stopper",
        "build a potion bottle with a separate turned stopper",
        "a potion flask with a small lathe-turned cork",
        "a fantasy vial with a proper turned stopper plugging the neck",
    ]
    notes = "Two lathe profiles in one record: the bottle body and a small separate stopper seated in the neck."
    return prompts[i], calls, notes, False


def recipe_35(rng, i):
    # medium: crate with a round port-hole cut in one side for reaching inside
    sx = jitter(rng, 0.55)
    sy = jitter(rng, 0.45)
    sz = jitter(rng, 0.45)
    hole_r = jitter(rng, 0.09, 0.15)
    body = nm(rng, "crate")
    cutter = nm(rng, "porthole_cutter")
    calls = [
        prim(body, "box", {"size": [sx, sy, sz]}, [0, sy / 2, 0]),
        prim(
            cutter,
            "cylinder",
            {"radius": hole_r, "height": sz * 1.4, "segments": 20},
            [0, sy * 0.5, 0],
            rotation=[90, 0, 0],
        ),
        boolean("difference", [ref(body), ref(cutter)]),
        material([ref(body)], jcol(rng, WOOD), "Crate Wood", roughness=0.8),
    ]
    prompts = [
        "a crate with a round port-hole cut in one side for reaching inside",
        f"a wooden crate with a round {hole_r * 200:.1f} cm porthole cut through one side",
        "a crate whose side has a circular hole cut through it",
        "build a crate with a round porthole cut clean through one wall",
        "a storage crate with a round hand-hole cut through its side",
        "a crate, one side bored through with a round porthole",
    ]
    notes = (
        "A cylinder rotated so its axis runs straight through the crate's own depth, "
        "long enough to punch clean through both walls, subtracted from the box."
    )
    return prompts[i], calls, notes, False


LATHE_CANDLE_HOLDER = (
    (0.0, 0.0),
    (0.10, 0.02),
    (0.10, 0.05),
    (0.045, 0.09),
    (0.08, 0.13),
    (0.08, 0.15),
)


def recipe_36(rng, i):
    # medium: lathe-turned candle holder with a socket cut for the candle
    s = jitter(rng, 0.65)
    stations = LATHE_CANDLE_HOLDER
    h = profile_height(stations)
    body = nm(rng, "candle_holder")
    cutter = nm(rng, "socket_cutter")
    socket_r = jitter(rng, 0.018, 0.1)
    top_y = s * h
    calls = [
        prim(
            body,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 18},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        ),
        prim(
            cutter,
            "cylinder",
            {"radius": socket_r, "height": 0.05 * s, "segments": 14},
            [0, top_y - 0.015 * s, 0],
        ),
        boolean("difference", [ref(body), ref(cutter)]),
        material(
            [ref(body)], jcol(rng, BRASS), "Candle Holder Brass", roughness=0.3, metallic=0.75
        ),
    ]
    prompts = [
        "a lathe-turned candle holder with a socket cut for the candle",
        f"a brass candle holder, {s * h * 100:.0f} cm tall, with a socket bored for the candle",
        "a turned candlestick with a cut socket to seat a candle",
        "build a lathe-turned candle holder with a socket cut into the top",
        "a candle holder, its cup bored out to hold a candle",
        "a small brass candlestick with a recessed socket",
    ]
    notes = "Lathe candlestick pedestal plus a cylinder subtracted from the top to bore the candle socket."
    return prompts[i], calls, notes, False


LATHE_BOWL = (
    (0.0, 0.0),
    (0.06, 0.015),
    (0.05, 0.03),
    (0.14, 0.06),
    (0.30, 0.16),
    (0.32, 0.2),
)


def recipe_37(rng, i):
    # medium: ceramic bowl with a lathe-turned foot ring
    s = jitter(rng, 0.7)
    stations = LATHE_BOWL
    h = profile_height(stations)
    body = nm(rng, "bowl")
    calls = [
        prim(
            body,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 22},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        ),
        material([ref(body)], jcol(rng, CERAMIC_WHITE), "Glazed Ceramic", roughness=0.3),
    ]
    prompts = [
        "a ceramic bowl with a lathe-turned foot ring",
        f"a shallow ceramic bowl, {s * h * 100:.0f} cm tall, standing on a turned foot ring",
        "a wide bowl raised on a narrow turned foot",
        "build a ceramic bowl with a foot ring",
        "a glazed bowl, its underside turned into a small foot ring",
        "a wide, shallow bowl lifted on a lathe-turned foot",
    ]
    notes = "Lathe: a narrow foot-ring station pinched in just above the base, then flaring wide and shallow for the bowl proper."
    return prompts[i], calls, notes, False


def recipe_38(rng, i):
    # medium: metal drum with a row of rivets arranged radially around the rim
    r = jitter(rng, 0.30)
    h = jitter(rng, 0.5)
    body = nm(rng, "drum")
    rivet = nm(rng, "rim_rivet")
    count = rng.choice([10, 14, 18, 24])
    rivet_r = jitter(rng, 0.012, 0.1)
    calls = [
        prim(body, "cylinder", {"radius": r, "height": h, "segments": 24}, [0, h / 2, 0]),
        prim(rivet, "uv_sphere", {"radius": rivet_r, "segments": 10, "rings": 6}, [r, h * 0.92, 0]),
        select([ref(rivet)]),
        op("array-radial", {"count": count, "angle": 360, "axis": 1}),
        material([ref(body)], jcol(rng, STEEL), "Drum Steel", roughness=0.4, metallic=0.6),
        material(dup_refs(rivet, count), jcol(rng, IRON), "Rivets", roughness=0.35, metallic=0.85),
    ]
    prompts = [
        "a metal drum with a row of rivets arranged radially around the rim",
        f"a metal drum, {r * 2 * 100:.0f} cm across, ringed with {count} rivets near the rim",
        "a drum with a full radial ring of rivets close to the top",
        "build a metal drum with rivets arrayed around its rim",
        "a steel drum banded with a circle of rivets near the top edge",
        "a drum, its rim studded with rivets running all the way around",
    ]
    notes = (
        "One small sphere rivet, selected then array-radial'd around the drum's own "
        "Y axis for a full 360-degree ring -- the family brief's own array-radial "
        "'ring of rivets' case."
    )
    return prompts[i], calls, notes, False


def recipe_39(rng, i):
    # medium: wine rack shaped as a lathe-turned pedestal holding a bottle
    s = jitter(rng, 0.6)
    ped_stations = (
        (0.0, 0.0),
        (0.14, 0.02),
        (0.05, 0.08),
        (0.05, 0.30),
        (0.16, 0.34),
        (0.16, 0.36),
    )
    ped_h = profile_height(ped_stations)
    pedestal = nm(rng, "pedestal")
    bottle_stations = LATHE_BOTTLE
    bh = profile_height(bottle_stations)
    bottle = nm(rng, "bottle")
    bs = 0.5 * s
    calls = [
        prim(
            pedestal,
            "lathe",
            {"profile": [list(p) for p in ped_stations], "segments": 18},
            [0, s * ped_h / 2, 0],
            scale=[s, s, s],
        ),
        prim(
            bottle,
            "lathe",
            {"profile": [list(p) for p in bottle_stations], "segments": 16},
            [0.10 * s, s * ped_h + bs * bh * 0.5, 0],
            rotation=[0, 0, 78],
            scale=[bs, bs, bs],
        ),
        material([ref(pedestal)], jcol(rng, DARK_WOOD), "Pedestal Wood", roughness=0.6),
        material([ref(bottle)], jcol(rng, GLASS_GREEN), "Bottle Glass", roughness=0.15),
    ]
    prompts = [
        "a wine rack shaped as a lathe-turned pedestal holding a bottle",
        f"a turned pedestal wine rack, {s * ped_h * 100:.0f} cm tall, cradling one bottle at an angle",
        "a lathe-turned pedestal stand with a bottle resting in its cradle",
        "build a wine pedestal that holds one bottle leaning against it",
        "a turned wooden pedestal with a wine bottle tilted in its cradle",
        "a wine-rack pedestal, one bottle propped at an angle on top",
    ]
    notes = (
        "Pedestal is its own lathe profile (a slim turned baluster); the bottle "
        "reuses the family's own bottle profile, scaled down and tilted to rest "
        "against the pedestal's top."
    )
    return prompts[i], calls, notes, False


def recipe_40(rng, i):
    # medium: stack of barrels arranged in a repeating pyramid pattern
    s = jitter(rng, 0.45)
    stations = LATHE_BARREL
    h = profile_height(stations)
    names = []
    calls = []
    bottom_xs = [-1.05, 0.0, 1.05]
    for k, bx in enumerate(bottom_xs):
        n = nm(rng, f"barrel_row1_{k}")
        names.append(n)
        calls.append(
            prim(
                n,
                "lathe",
                {"profile": [list(p) for p in stations], "segments": 16},
                [bx * s * 0.32, s * h / 2, 0],
                scale=[s, s, s],
            )
        )
    mid_xs = [-0.525, 0.525]
    for k, bx in enumerate(mid_xs):
        n = nm(rng, f"barrel_row2_{k}")
        names.append(n)
        calls.append(
            prim(
                n,
                "lathe",
                {"profile": [list(p) for p in stations], "segments": 16},
                [bx * s * 0.32, s * h * 1.5, 0],
                scale=[s, s, s],
            )
        )
    n = nm(rng, "barrel_row3")
    names.append(n)
    calls.append(
        prim(
            n,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 16},
            [0, s * h * 2.5, 0],
            scale=[s, s, s],
        )
    )
    calls.append(material([ref(n) for n in names], jcol(rng, WOOD), "Barrel Wood", roughness=0.75))
    prompts = [
        "a stack of barrels arranged in a repeating pyramid pattern",
        "six barrels stacked in a pyramid, three on the bottom, two above, one on top",
        "a pyramid of stacked wooden barrels",
        "build a pyramid stack of barrels, three-two-one",
        "barrels stacked into a pyramid shape",
        "a stable pyramid arrangement of six barrels",
    ]
    notes = (
        "Six copies of the same lathe barrel, placed by hand at computed row "
        "offsets (3 - 2 - 1) rather than through an array op, since the rows shift "
        "position as well as height."
    )
    return prompts[i], calls, notes, False


def recipe_41(rng, i):
    # medium: chest whose lid has a carved emblem cut into the surface
    sx = jitter(rng, 0.55)
    sy = jitter(rng, 0.32)
    sz = jitter(rng, 0.35)
    lid_h = 0.05
    body = nm(rng, "chest_body")
    lid = nm(rng, "chest_lid")
    cutter = nm(rng, "emblem_cutter")
    emblem_r = jitter(rng, 0.06, 0.15)
    calls = [
        prim(body, "box", {"size": [sx, sy, sz]}, [0, sy / 2, 0]),
        prim(lid, "box", {"size": [sx * 1.02, lid_h, sz * 1.02]}, [0, sy + lid_h / 2, 0]),
        prim(
            cutter,
            "cylinder",
            {"radius": emblem_r, "height": 0.02, "segments": 6},
            [0, sy + lid_h - 0.006, 0],
        ),
        boolean("difference", [ref(lid), ref(cutter)]),
        material([ref(body), ref(lid)], jcol(rng, DARK_WOOD), "Chest Wood", roughness=0.75),
    ]
    prompts = [
        "a chest whose lid has a carved emblem cut into the surface",
        f"a wooden chest, its lid stamped with a shallow {emblem_r * 200:.0f} cm hexagonal emblem",
        "a chest with an emblem carved shallowly into the lid",
        "build a chest with a carved emblem recessed into the lid's top",
        "a treasure chest, its lid bearing a shallow carved crest",
        "a chest lid with a hexagonal emblem cut into its surface",
    ]
    notes = (
        "A shallow six-sided prism (a low cylinder with 6 segments) sunk just "
        "into the lid's top face and subtracted, engraving a recessed emblem "
        "without cutting through."
    )
    return prompts[i], calls, notes, False


def recipe_42(rng, i):
    # medium: jug with a handle loop cut through the side
    s = jitter(rng, 0.62)
    stations = LATHE_JUG
    h = profile_height(stations)
    body = nm(rng, "jug")
    handle_mass = nm(rng, "handle_mass")
    handle_cutter = nm(rng, "handle_hole_cutter")
    handle_y = s * h * 0.55
    handle_x = 0.20 * s
    calls = [
        prim(
            body,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 18},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        ),
        prim(
            handle_mass,
            "torus",
            {"radius": 0.055 * s, "tube": 0.028 * s, "segments": 16, "sides": 10},
            [handle_x, handle_y, 0],
            rotation=[0, 0, 90],
        ),
        prim(
            handle_cutter,
            "cylinder",
            {"radius": 0.03 * s, "height": 0.08 * s, "segments": 14},
            [handle_x, handle_y, 0],
            rotation=[0, 0, 90],
        ),
        boolean("difference", [ref(handle_mass), ref(handle_cutter)]),
        material([ref(body)], jcol(rng, TERRACOTTA), "Jug Ceramic", roughness=0.5),
        material([ref(handle_mass)], jcol(rng, TERRACOTTA), "Jug Handle", roughness=0.5),
    ]
    prompts = [
        "a jug with a handle loop cut through the side",
        f"a jug, {s * h * 100:.0f} cm tall, with a thick handle that has a grip hole cut through it",
        "a jug with a solid handle blob pierced by a grip loop",
        "build a jug whose handle has a hole cut through it to grip",
        "a ceramic jug, its side handle punched through with a loop",
        "a jug with a chunky handle, hollowed by a hole for fingers",
    ]
    notes = (
        "The handle is built as a solid torus 'mass' with a narrower cylinder "
        "subtracted straight through it, leaving a true grip loop -- a second, "
        "distinct use of difference from the jar/barrel openings."
    )
    return prompts[i], calls, notes, False


LATHE_STORAGE_JAR = (
    (0.0, 0.0),
    (0.13, 0.02),
    (0.17, 0.08),
    (0.16, 0.22),
    (0.09, 0.28),
    (0.10, 0.32),
)


def recipe_43(rng, i):
    # medium: set of five identical storage jars arranged in a row
    s = jitter(rng, 0.5)
    stations = LATHE_STORAGE_JAR
    h = profile_height(stations)
    jar = nm(rng, "storage_jar")
    step = jitter(rng, 0.22)
    calls = [
        prim(
            jar,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 16},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        ),
        select([ref(jar)]),
        op("array-linear", {"count": 5, "x": step, "y": 0, "z": 0}),
        material(dup_refs(jar, 5), jcol(rng, TERRACOTTA), "Storage Jar Clay", roughness=0.6),
    ]
    prompts = [
        "a set of five identical storage jars arranged in a row",
        f"five matching storage jars, {step * 100:.0f} cm apart, lined up in a row",
        "a row of five identical clay storage jars",
        "build five identical jars spaced evenly in a line",
        "a shelf of five matching storage jars in a row",
        "five storage jars, evenly spaced along a line",
    ]
    notes = (
        "One lathe jar, selected and array-linear'd five times along X -- a "
        "different op from the barrel/drum's array-radial, for the row layout the "
        "prompt actually asks for."
    )
    return prompts[i], calls, notes, False


def recipe_44(rng, i):
    # medium: barrel with a spigot hole cut near its base
    s = jitter(rng, 0.65)
    stations = LATHE_BARREL
    h = profile_height(stations)
    body = nm(rng, "barrel")
    cutter = nm(rng, "spigot_cutter")
    hole_r = jitter(rng, 0.022, 0.15)
    calls = [
        prim(
            body,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 22},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        ),
        prim(
            cutter,
            "cylinder",
            {"radius": hole_r, "height": 0.4 * s, "segments": 16},
            [0, s * h * 0.16, 0],
            rotation=[0, 0, 90],
        ),
        boolean("difference", [ref(body), ref(cutter)]),
        material([ref(body)], jcol(rng, WOOD), "Barrel Oak", roughness=0.7),
    ]
    prompts = [
        "a barrel with a spigot hole cut near its base",
        f"a barrel, {s * h * 100:.0f} cm tall, with a spigot hole bored near the bottom",
        "a barrel with a hole cut low on its side for a spigot",
        "build a barrel with a spigot hole near the base",
        "a wooden barrel, a round hole cut close to the bottom for a tap",
        "a barrel bored through near its base for a spigot",
    ]
    notes = "A horizontal cylinder near the barrel's base, subtracted, punches the spigot hole through the wall."
    return prompts[i], calls, notes, False


LATHE_INCENSE = (
    (0.0, 0.0),
    (0.09, 0.015),
    (0.13, 0.06),
    (0.12, 0.14),
    (0.14, 0.16),
)


def recipe_45(rng, i):
    # medium: lathe-turned incense burner with a perforated lid
    s = jitter(rng, 0.55)
    stations = LATHE_INCENSE
    h = profile_height(stations)
    body = nm(rng, "burner_body")
    lid_r = jitter(rng, 0.145, 0.1)
    lid_h = jitter(rng, 0.02, 0.1)
    lid = nm(rng, "burner_lid")
    hole = nm(rng, "lid_hole")
    count = rng.choice([6, 8, 10])
    lid_y = s * h + lid_h / 2 + 0.002
    calls = [
        prim(
            body,
            "lathe",
            {"profile": [list(p) for p in stations], "segments": 20},
            [0, s * h / 2, 0],
            scale=[s, s, s],
        ),
        prim(
            lid, "cylinder", {"radius": lid_r * s, "height": lid_h, "segments": 20}, [0, lid_y, 0]
        ),
        prim(
            hole,
            "cylinder",
            {"radius": 0.012 * s, "height": lid_h * 3, "segments": 10},
            [lid_r * s * 0.6, lid_y, 0],
        ),
        select([ref(hole)]),
        op("array-radial", {"count": count, "angle": 360, "axis": 1}),
        # array-radial leaves the whole fan selected but clay_boolean still
        # needs explicit uids -- ops.next_name counts up ``.001``, ``.002``...
        # deterministically for each copy (ops.py's own duplicate-naming
        # docstring), so every copy's $ref is predictable without a
        # clay_scene read in between.
        boolean("difference", [ref(lid)] + dup_refs(hole, count)),
        material([ref(body)], jcol(rng, BRASS), "Burner Brass", roughness=0.35, metallic=0.7),
        material([ref(lid)], jcol(rng, BRASS), "Burner Lid", roughness=0.35, metallic=0.7),
    ]
    prompts = [
        "a lathe-turned incense burner with a perforated lid",
        f"an incense burner, {s * h * 100:.0f} cm tall, its flat lid pierced with {count} small holes",
        "a turned incense burner topped with a perforated disc lid",
        "build an incense burner with a lid punched with a ring of holes",
        "a brass incense burner, lid dotted with a circle of small vents",
        "an incense holder, its lid perforated in a ring pattern",
    ]
    notes = (
        "Lathe pedestal body plus a flat cylinder lid; one small cylinder is "
        "array-radial'd into a ring of copies and the whole ring is subtracted from "
        "the lid in one clay_boolean call (uids take more than two members) -- "
        "array-radial and difference combined in a single record."
    )
    return prompts[i], calls, notes, False


# --------------------------------------------------------------------------
# HARD (46-50)
# --------------------------------------------------------------------------


def recipe_46(rng, i):
    # hard: leather waterskin, tapers and bends like a sagging pouch (tube)
    s = jitter(rng, 0.6)
    radius = 0.09 * s
    path = [
        [0.0, 0.30, 0.0],
        [0.10, 0.20, 0.02],
        [0.14, 0.08, 0.0],
        [0.08, 0.0, -0.02],
        [-0.05, -0.02, 0.0],
    ]
    miny = min(p[1] for p in path)
    maxy = max(p[1] for p in path)
    span = maxy - miny
    body = nm(rng, "waterskin")
    calls = [
        prim(
            body,
            "tube",
            {"path": path, "radius": radius, "sides": 12},
            [0, span / 2 + radius + 0.01, 0],
            scale=[1.0, 1.0, 0.7],
        ),
        material([ref(body)], jcol(rng, LEATHER), "Waterskin Leather", roughness=0.8),
    ]
    prompts = [
        "a leather waterskin, curved and swollen like a sagging, water-heavy pouch",
        f"a leather waterskin, about {span * 100:.0f} cm along its bent length, sagging like a full pouch",
        "a bent, sagging leather waterskin, narrower at the neck",
        "build a leather waterskin that bends and sags like a full pouch",
        "a soft leather waterskin, its body bowed as if heavy and full",
        "a drooping leather waterskin, curved from neck to base",
    ]
    notes = (
        "Built with tube, whose centreline path does the bending a lathe's straight "
        "Y axis cannot -- the path dips down through the middle like a laden pouch. "
        "tube's own radius is constant along the path (no per-station taper in its "
        "schema), so the tapered look comes from a non-uniform scale on the Z axis "
        "afterward, flattening the pouch slightly rather than narrowing it end to "
        "end; noted here rather than claimed as a true taper."
    )
    return prompts[i], calls, notes, True


def recipe_47(rng, i):
    # hard: woven vine basket whose handle sweeps in a smooth arc side to side
    r = jitter(rng, 0.24)
    h = jitter(rng, 0.20)
    body = nm(rng, "basket_body")
    handle = nm(rng, "basket_handle")
    handle_radius = 0.012 * (r / 0.24)
    arc_h = jitter(rng, 0.30)
    path = [
        [-r * 0.85, h, 0.0],
        [-r * 0.5, h + arc_h * 0.75, 0.0],
        [0.0, h + arc_h, 0.0],
        [r * 0.5, h + arc_h * 0.75, 0.0],
        [r * 0.85, h, 0.0],
    ]
    calls = [
        prim(body, "cylinder", {"radius": r, "height": h, "segments": 18}, [0, h / 2, 0]),
        prim(handle, "tube", {"path": path, "radius": handle_radius, "sides": 8}, [0, 0, 0]),
        material([ref(body)], jcol(rng, [0.5, 0.38, 0.20]), "Woven Vine", roughness=0.9),
        material([ref(handle)], jcol(rng, [0.45, 0.34, 0.18]), "Vine Handle", roughness=0.9),
    ]
    # the handle's own path is re-centred by the generator on its own bbox;
    # translation restores it to the world position the path was written in.
    cx = sum(p[0] for p in path) / len(path)
    cy = sum(p[1] for p in path) / len(path)
    cz = sum(p[2] for p in path) / len(path)
    calls[1]["arguments"]["translation"] = [round(cx, 4), round(cy, 4), round(cz, 4)]
    prompts = [
        "a woven vine basket whose handle sweeps in a smooth arc from side to side",
        f"a vine basket, {r * 2 * 100:.0f} cm across, with a handle arcing smoothly from one rim side to the other",
        "a basket with a tall, smoothly arced carrying handle",
        "build a basket whose handle sweeps up and over in one smooth arc",
        "a woven basket with a high, gently curved handle",
        "a vine basket topped with a single sweeping arc handle",
    ]
    notes = (
        "Handle built with tube along a five-point arced path (rim - up - apex - "
        "up - rim), which is exactly the swept-along-a-path case a lathe or sweep "
        "cannot reach on its own."
    )
    return prompts[i], calls, notes, False


def recipe_48(rng, i):
    # hard: ceremonial horn container, tapers from wide mouth to curled point
    s = jitter(rng, 0.6)
    path = [
        [0.0, 0.34, 0.0],
        [0.05, 0.24, 0.01],
        [0.10, 0.14, 0.03],
        [0.14, 0.06, 0.07],
        [0.15, 0.0, 0.12],
    ]
    radius = 0.09 * s
    tip_r = 0.012 * s
    body = nm(rng, "horn_body")
    tip = nm(rng, "horn_tip")
    miny = min(p[1] for p in path)
    maxy = max(p[1] for p in path)
    scale3 = [s, s, s]
    body_translation = [0, (maxy - miny) / 2 + radius + 0.01, 0]
    tip_pos = tube_endpoint_world(path, path[-1], body_translation, scale3)
    calls = [
        prim(
            body,
            "tube",
            {"path": path, "radius": radius, "sides": 14},
            body_translation,
            scale=scale3,
        ),
        prim(
            tip,
            "cone",
            {"radius": tip_r, "height": 0.05 * s, "segments": 10},
            tip_pos,
            rotation=[0, 0, 80],
        ),
        material([ref(body), ref(tip)], jcol(rng, [0.35, 0.28, 0.20]), "Horn", roughness=0.5),
    ]
    prompts = [
        "a ceremonial horn container that tapers from a wide mouth to a curled point",
        "a ceremonial horn, curving from a wide mouth down to a small curled tip",
        "a curved ceremonial horn, wide at the mouth and pointed at the tip",
        "build a ceremonial horn container curving to a narrow curled point",
        "a horn vessel, its curve tightening toward a small pointed tip",
        "a ceremonial drinking horn, curled and tapering to a point",
    ]
    notes = (
        "tube's bent path gives the curl; a small cone at the path's tip end "
        "reads as the tapering point, since tube itself holds one constant radius "
        "along its whole length."
    )
    return prompts[i], calls, notes, False


def recipe_49(rng, i):
    # hard: sack whose bulging profile sweeps outward then tapers to a tied neck
    s = jitter(rng, 0.55)
    outline = [
        [0.20, 0.0],
        [0.14, 0.17],
        [0.0, 0.20],
        [-0.14, 0.17],
        [-0.20, 0.0],
        [-0.14, -0.17],
        [0.0, -0.20],
        [0.14, -0.17],
    ]
    depth = jitter(rng, 0.5)
    taper = 0.12
    twist = 70.0
    body = nm(rng, "sack")
    tie = nm(rng, "neck_tie")
    calls = [
        prim(
            body,
            "sweep",
            {"outline": outline, "depth": depth, "taper": taper, "twist": twist, "sections": 10},
            [0, s * depth / 2, 0],
            rotation=[-90, 0, 0],
            scale=[s, s, s],
        ),
        prim(
            tie,
            "torus",
            {"radius": 0.045 * s, "tube": 0.012 * s, "segments": 16, "sides": 8},
            [0, s * depth * 0.92, 0],
        ),
        material([ref(body)], jcol(rng, BURLAP), "Sack Cloth", roughness=0.9),
        material([ref(tie)], jcol(rng, [0.3, 0.22, 0.12]), "Tie Cord", roughness=0.85),
    ]
    prompts = [
        "a sack whose bulging profile sweeps outward then tapers to a tied neck",
        "a cloth sack, bulging wide in the middle and gathered to a tied neck at top",
        "a sack that bulges out and narrows sharply to a knotted neck",
        "build a sack with a bulging body tapering to a tied-off neck",
        "a full sack, round in the middle, cinched at the neck",
        "a bulging sack, its neck twisted and tied shut",
    ]
    notes = (
        "sweep, rotated so its Z extrusion axis stands up as world Y: an octagon "
        "outline tapered hard toward the far end and twisted along the way for the "
        "gathered-cloth look, with a torus 'cord' marking the tie."
    )
    return prompts[i], calls, notes, True


def recipe_50(rng, i):
    # hard: curved drinking horn mounted on a stand, tapering along its length
    s = jitter(rng, 0.6)
    path = [
        [0.0, 0.30, 0.0],
        [0.04, 0.20, 0.02],
        [0.09, 0.10, 0.05],
        [0.13, 0.02, 0.09],
    ]
    radius = 0.075 * s
    tip_r = 0.01 * s
    stand_r = jitter(rng, 0.10)
    stand_h = jitter(rng, 0.12)
    horn = nm(rng, "drinking_horn")
    tip = nm(rng, "horn_tip")
    stand = nm(rng, "horn_stand")
    miny = min(p[1] for p in path)
    maxy = max(p[1] for p in path)
    horn_y = (maxy - miny) / 2 + radius + stand_h
    horn_translation = [0, horn_y, 0]
    scale3 = [s, s, s]
    tip_pos = tube_endpoint_world(path, path[-1], horn_translation, scale3, rotation_z_deg=12)
    calls = [
        prim(
            stand,
            "cylinder",
            {"radius": stand_r, "height": stand_h, "segments": 18},
            [0, stand_h / 2, 0],
        ),
        prim(
            horn,
            "tube",
            {"path": path, "radius": radius, "sides": 14},
            horn_translation,
            rotation=[0, 0, 12],
            scale=scale3,
        ),
        prim(
            tip,
            "cone",
            {"radius": tip_r, "height": 0.04 * s, "segments": 10},
            tip_pos,
            rotation=[0, 0, 78],
        ),
        material([ref(stand)], jcol(rng, DARK_WOOD), "Stand Wood", roughness=0.65),
        material(
            [ref(horn), ref(tip)], jcol(rng, [0.38, 0.30, 0.20]), "Drinking Horn", roughness=0.5
        ),
    ]
    prompts = [
        "a curved drinking horn mounted on a stand, tapering along its length",
        "a drinking horn on a wooden stand, curving and tapering to a point",
        "a horn cup resting in its own stand, curved and tapering",
        "build a drinking horn cradled in a small stand",
        "a ceremonial drinking horn propped on a stand, tapering to its tip",
        "a curved horn vessel seated in a short display stand",
    ]
    notes = (
        "tube for the horn's own curve, resting on a plain cylinder stand; a small "
        "cone at the tip end again stands in for the taper tube's constant radius "
        "cannot express on its own."
    )
    return prompts[i], calls, notes, False


RECIPES = [
    recipe_01,
    recipe_02,
    recipe_03,
    recipe_04,
    recipe_05,
    recipe_06,
    recipe_07,
    recipe_08,
    recipe_09,
    recipe_10,
    recipe_11,
    recipe_12,
    recipe_13,
    recipe_14,
    recipe_15,
    recipe_16,
    recipe_17,
    recipe_18,
    recipe_19,
    recipe_20,
    recipe_21,
    recipe_22,
    recipe_23,
    recipe_24,
    recipe_25,
    recipe_26,
    recipe_27,
    recipe_28,
    recipe_29,
    recipe_30,
    recipe_31,
    recipe_32,
    recipe_33,
    recipe_34,
    recipe_35,
    recipe_36,
    recipe_37,
    recipe_38,
    recipe_39,
    recipe_40,
    recipe_41,
    recipe_42,
    recipe_43,
    recipe_44,
    recipe_45,
    recipe_46,
    recipe_47,
    recipe_48,
    recipe_49,
    recipe_50,
]

PHRASINGS_PER_RECIPE = 6


def main() -> None:
    for recipe_idx, fn in enumerate(RECIPES):
        for phrase_idx in range(PHRASINGS_PER_RECIPE):
            rng = random.Random(SEED * 10_000 + recipe_idx * 100 + phrase_idx)
            prompt, calls, notes, allow_below = fn(rng, phrase_idx)
            emit(prompt, calls, notes, allow_below)

    with OUT.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=False) + "\n")
    print(f"wrote {len(records)} records to {OUT}")


if __name__ == "__main__":
    main()
