"""Deterministic generator for ``drafts/edits.jsonl`` -- the ``edits`` family
of the Clay-assistant training corpus (see ``author_brief.md`` and
``../README.md``). Regenerate with::

    uv run python training/clay-assistant/drafts/_gen_edits.py

Every record is ``kind: edit``: a ``prior`` batch that builds a starting
object from primitives, and a ``calls`` batch that edits it, addressing the
prior's own objects by ``{"$ref": "<name>"}``. Geometry follows
``agent_clay.instructions()``'s own convention throughout: metres, Y up,
ground y=0, every generator centred on its own origin (a box of height h
sits at translation ``[0, h/2, 0]``), rotations Euler XYZ degrees.

Design notes worth keeping next to the code rather than only in the report:

* The compact tool card (``gen/convert.compact_tools()``) documents exactly
  13 tools. ``clay_element_mode``/``clay_select_by``/``clay_select_elements``
  are valid ``clay_batch`` entry names (they appear in that tool's own
  ``name`` enum) but carry no schema of their own on the card, so no record
  here calls them -- an edit that needed element-mode bevel/inset was
  reinterpreted as an equivalent boolean or add-part edit instead (see the
  ``chest_hinge`` and ``chest_bevel`` specs below).
* "Scale/move the whole assembly by f" is done by multiplying *every* part's
  own translation and scale by f, uniformly, about the world origin -- this
  keeps every part's own ground contact exactly where it was (see
  ``_scale_parts`` below for the proof sketch) without needing a group node
  clay does not have.
* Multi-part assemblies that get rotated as a rigid body (the cart) are built
  with zero initial rotation on every part specifically so composing the
  extra Y-rotation stays additive instead of needing quaternion composition.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parent / "edits.jsonl"

Call = dict[str, Any]


# --------------------------------------------------------------------------
# tiny call-builders
# --------------------------------------------------------------------------


def C(tool_name: str, **kw: Any) -> Call:
    return {"name": tool_name, "arguments": kw}


def R(n: str) -> dict[str, str]:
    return {"$ref": n}


def prim(
    generator: str,
    name: str,
    params: dict | None = None,
    translation: list | tuple | None = None,
    rotation: list | tuple | None = None,
    scale: list | tuple | None = None,
    material: int | None = None,
) -> Call:
    args: dict[str, Any] = {"generator": generator, "name": name}
    if params is not None:
        args["params"] = params
    if translation is not None:
        args["translation"] = list(translation)
    if rotation is not None:
        args["rotation"] = list(rotation)
    if scale is not None:
        args["scale"] = list(scale)
    if material is not None:
        args["material"] = material
    return C("clay_add_primitive", **args)


def figure(key: str, name_prefix: str, translation=None, yaw=None, scale=None) -> Call:
    args: dict[str, Any] = {"key": key, "name_prefix": name_prefix}
    if translation is not None:
        args["translation"] = list(translation)
    if yaw is not None:
        args["yaw"] = yaw
    if scale is not None:
        args["scale"] = scale
    return C("clay_add_figure", **args)


def mat_call(names: list[str], color, roughness=0.6, metallic=0.0, name: str | None = None) -> Call:
    kw: dict[str, Any] = {
        "uids": [R(n) for n in names],
        "color": list(color),
        "roughness": roughness,
    }
    if metallic:
        kw["metallic"] = metallic
    if name:
        kw["name"] = name
    return C("clay_material", **kw)


def xform(name: str, translation=None, rotation=None, scale=None) -> Call:
    kw: dict[str, Any] = {"uid": R(name)}
    if translation is not None:
        kw["translation"] = list(translation)
    if rotation is not None:
        kw["rotation"] = list(rotation)
    if scale is not None:
        kw["scale"] = list(scale)
    return C("clay_transform", **kw)


def set_params(target, params: dict) -> Call:
    if isinstance(target, (list, tuple)):
        return C("clay_set_params", uids=[R(n) for n in target], params=params)
    return C("clay_set_params", uid=R(target), params=params)


def delete(names: list[str]) -> Call:
    return C("clay_delete", uids=[R(n) for n in names])


def rename(name: str, new_name: str) -> Call:
    return C("clay_rename", uid=R(name), name=new_name)


def boolean(kind: str, names: list[str]) -> Call:
    return C("clay_boolean", kind=kind, uids=[R(n) for n in names])


def select(names: list[str]) -> Call:
    return C("clay_select", uids=[R(n) for n in names])


def op(name: str, params: dict | None = None) -> Call:
    kw: dict[str, Any] = {"name": name}
    if params:
        kw["params"] = params
    return C("clay_op", **kw)


def path_centroid(points):
    """The translation a ``tube``/``sweep``/``lathe`` primitive needs so its
    array-valued parameter's own values read as the *world* positions they
    look like: every one of those generators re-centres its array parameter
    on its own bounding-box midpoint before building a single vertex (see
    ``primitives._clamp_path``'s docstring), so a caller that wrote path
    points as if they were already absolute world coordinates must supply
    this midpoint back as ``translation`` or the whole shape reappears
    centred on the object's local origin instead of where it was drawn.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2]


def rot_y(pt, deg):
    """Rotate an (x, z) pair about the world Y axis by *deg* degrees."""
    x, z = pt
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (x * c - z * s, x * s + z * c)


# --------------------------------------------------------------------------
# id counter / record assembly
# --------------------------------------------------------------------------

_records: list[dict[str, Any]] = []


def emit(
    key: str,
    class_: str,
    prompt: str,
    prior: list[Call],
    calls: list[Call],
    notes: str,
    allow_below_ground: bool = False,
) -> None:
    n = len(_records) + 1
    rec = {
        "id": f"edits-{n:04d}",
        "family": "edits",
        "kind": "edit",
        "prompt": prompt,
        "prior": prior,
        "calls": calls,
        "notes": f"[{class_}/{key}] {notes}",
    }
    if allow_below_ground:
        rec["allow_below_ground"] = True
    _records.append(rec)


# ==========================================================================
# Spec 1 -- plain wooden table :: make the tabletop twice as thick (easy)
# ==========================================================================


def spec_table_thicken() -> None:
    def build(names, dims):
        top, la, lb, lc, ld = (
            names.get("top", "tabletop"),
            names.get("la", "leg_fl"),
            names.get("lb", "leg_fr"),
            names.get("lc", "leg_bl"),
            names.get("ld", "leg_br"),
        )
        w, thick, d = dims["top"]
        leg_h = dims["leg_h"]
        leg_w = dims.get("leg_w", 0.06)
        x, z = dims["half_span"]
        color = dims.get("color", (0.42, 0.28, 0.16))
        calls = [
            prim("box", top, {"size": [w, thick, d]}, translation=[0, leg_h + thick / 2, 0]),
        ]
        for nm, (sx, sz) in ((la, (x, z)), (lb, (-x, z)), (lc, (x, -z)), (ld, (-x, -z))):
            calls.append(
                prim("box", nm, {"size": [leg_w, leg_h, leg_w]}, translation=[sx, leg_h / 2, sz])
            )
        calls.append(mat_call([top, la, lb, lc, ld], color, roughness=0.75, name="Oak"))
        return calls, (top, leg_h, thick)

    def edit(names, dims, built):
        top, leg_h, thick = built
        new_thick = thick * 2
        return [
            set_params(top, {"size": [dims["top"][0], new_thick, dims["top"][2]]}),
            xform(top, translation=[0, leg_h + new_thick / 2, 0]),
        ]

    NAMES = [
        {},
        {"top": "Top", "la": "Leg FL", "lb": "Leg FR", "lc": "Leg BL", "ld": "Leg BR"},
    ]
    DIMS = [
        {"top": (0.9, 0.04, 0.55), "leg_h": 0.72, "half_span": (0.4, 0.24)},
        {
            "top": (1.2, 0.03, 0.7),
            "leg_h": 0.74,
            "half_span": (0.55, 0.31),
            "color": (0.5, 0.35, 0.2),
        },
    ]
    PROMPTS = [
        "make the tabletop twice as thick",
        "the table's top looks flimsy -- double its thickness",
        "Double the thickness of the tabletop, please.",
        "this plain wooden table needs a sturdier top: make it twice as thick",
        "for a medieval great hall table, thicken the top to double what it is",
        "thicken the tabletop x2",
        "can you make the top of this table twice as thick as it is now",
        "double the tabletop's thickness so it reads as a heavier slab",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "table_thicken",
            "easy",
            prompt,
            prior,
            calls,
            "A tabletop is a centred box resting on 0.72-0.74m legs; doubling "
            "its size[1] and re-centring translation.y at leg_h + new_thick/2 "
            "keeps the underside exactly on the legs' own tops.",
        )


# ==========================================================================
# Spec 2 -- metal storage crate :: change its material to rusted iron (easy)
# ==========================================================================


def spec_crate_material() -> None:
    def build(names, dims):
        crate = names.get("crate", "crate")
        size = dims["size"]
        calls = [prim("box", crate, {"size": list(size)}, translation=[0, size[1] / 2, 0])]
        calls.append(
            mat_call(
                [crate],
                dims.get("orig_color", (0.55, 0.56, 0.58)),
                roughness=0.5,
                metallic=0.6,
                name="Steel",
            )
        )
        return calls, (crate,)

    def edit(names, dims, built):
        (crate,) = built
        return [
            mat_call([crate], (0.32, 0.16, 0.09), roughness=0.85, metallic=0.35, name="Rusted Iron")
        ]

    NAMES = [{}, {"crate": "Crate"}]
    DIMS = [
        {"size": (0.6, 0.5, 0.6)},
        {"size": (0.8, 0.7, 0.55)},
    ]
    PROMPTS = [
        "change its material to rusted iron",
        "this crate should look rusted -- swap the material for rusted iron",
        "Repaint the metal storage crate as rusted iron.",
        "make it rusted iron instead of clean steel",
        "in a post-apocalyptic scrapyard scene, this crate would be rusted iron -- change its material",
        "rust it",
        "give the crate a rusted-iron finish, orange-brown and rough",
        "swap this crate's material for a weathered, rusted iron",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "crate_material",
            "easy",
            prompt,
            prior,
            calls,
            "One clay_material call on the crate's own uid; a fresh palette "
            "entry rather than editing the steel one in place, since the "
            "prior's steel entry may still be referenced elsewhere.",
        )


# ==========================================================================
# Spec 3 -- round wooden stool :: raise the seat height by ten centimetres
# ==========================================================================


def spec_stool_raise() -> None:
    def build(names, dims):
        seat = names.get("seat", "seat")
        legs = [names.get(f"leg{i}", f"leg_{i}") for i in range(3)]
        seat_r = dims["seat_r"]
        seat_t = dims["seat_t"]
        leg_h = dims["leg_h"]
        leg_r = dims.get("leg_r", 0.03)
        ring = dims.get("ring", 0.16)
        calls = [
            prim(
                "cylinder",
                seat,
                {"radius": seat_r, "height": seat_t, "segments": 24},
                translation=[0, leg_h + seat_t / 2, 0],
            )
        ]
        for i, nm in enumerate(legs):
            ang = 90 + i * 120
            x = ring * math.cos(math.radians(ang))
            z = ring * math.sin(math.radians(ang))
            calls.append(
                prim(
                    "cylinder",
                    nm,
                    {"radius": leg_r, "height": leg_h, "segments": 12},
                    translation=[x, leg_h / 2, z],
                )
            )
        calls.append(
            mat_call(
                [seat, *legs], dims.get("color", (0.45, 0.31, 0.18)), roughness=0.7, name="Oak"
            )
        )
        return calls, (seat, legs, seat_t, leg_h)

    def edit(names, dims, built):
        seat, legs, seat_t, leg_h = built
        raise_by = 0.10
        new_leg_h = leg_h + raise_by
        calls = [set_params(legs, {"height": new_leg_h})]
        for nm in legs:
            calls.append(xform(nm, translation=None))
        # need per-leg translation with original x/z -- recompute below
        return calls, new_leg_h

    NAMES = [{}, {"seat": "Seat", "leg0": "Leg A", "leg1": "Leg B", "leg2": "Leg C"}]
    DIMS = [
        {"seat_r": 0.18, "seat_t": 0.04, "leg_h": 0.42},
        {"seat_r": 0.2, "seat_t": 0.035, "leg_h": 0.45, "ring": 0.15},
    ]
    PROMPTS = [
        "raise the seat height by ten centimetres",
        "this stool sits too low -- raise it by 10cm",
        "Raise the seat of this stool by ten centimetres.",
        "make the stool 0.1m taller",
        "for use at a modern kitchen island, raise this stool's seat by ten centimetres",
        "raise it by 10cm",
        "the round wooden stool needs to be ten centimetres taller at the seat",
        "add ten centimetres to the stool's height",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        seat, legs, seat_t, leg_h = built
        raise_by = 0.10
        new_leg_h = leg_h + raise_by
        ring = dims.get("ring", 0.16)
        calls = [set_params(legs, {"height": new_leg_h})]
        for i2 in range(3):
            ang = 90 + i2 * 120
            x = ring * math.cos(math.radians(ang))
            z = ring * math.sin(math.radians(ang))
            calls.append(xform(legs[i2], translation=[x, new_leg_h / 2, z]))
        calls.append(xform(seat, translation=[0, new_leg_h + seat_t / 2, 0]))
        emit(
            "stool_raise",
            "easy",
            prompt,
            prior,
            calls,
            "Legs are cylinders on a 120-degree ring; raising the seat means "
            "growing each leg's own height by 0.1m (set_params, plural uids) "
            "then re-centring each leg's translation.y at half its new "
            "height and moving the seat up by the full 0.1m.",
        )


# ==========================================================================
# Spec 4 -- stone archway :: make it twice as wide (easy)
# ==========================================================================


def spec_archway_widen() -> None:
    def build(names, dims):
        arch_n = names.get("arch", "archway")
        w, h, d, t = dims["arch"]
        calls = [
            prim(
                "arch",
                arch_n,
                {"width": w, "height": h, "depth": d, "thickness": t, "segments": 16},
                translation=[0, h / 2, 0],
            )
        ]
        calls.append(
            mat_call([arch_n], dims.get("color", (0.6, 0.58, 0.55)), roughness=0.9, name="Stone")
        )
        return calls, (arch_n, w)

    def edit(names, dims, built):
        arch_n, w = built
        return [set_params(arch_n, {"width": w * 2})]

    NAMES = [{}, {"arch": "Archway"}]
    DIMS = [
        {"arch": (1.4, 2.2, 0.5, 0.3)},
        {"arch": (1.8, 2.6, 0.6, 0.35), "color": (0.55, 0.52, 0.48)},
    ]
    PROMPTS = [
        "make it twice as wide",
        "this archway is too narrow -- double its width",
        "Double the width of the stone archway.",
        "widen the arch to 2x its current width",
        "for a grand castle gatehouse, this archway should be twice as wide",
        "widen it x2",
        "the stone archway needs to be twice as wide as it is now",
        "make the archway's opening twice as wide overall",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "archway_widen",
            "easy",
            prompt,
            prior,
            calls,
            "arch's width does not affect its vertical centring (height "
            "alone decides translation.y), so a single set_params suffices.",
        )


# ==========================================================================
# Spec 5 -- simple wooden barrel :: change its material to dark oak (easy)
# ==========================================================================


def spec_barrel_material() -> None:
    def build(names, dims):
        barrel = names.get("barrel", "barrel")
        h = dims["h"]
        r_mid = dims["r_mid"]
        r_end = dims.get("r_end", r_mid * 0.72)
        profile = [
            (r_end, -h / 2),
            (r_mid, -h / 4),
            (r_mid, h / 4),
            (r_end, h / 2),
        ]
        calls = [
            prim("lathe", barrel, {"profile": profile, "segments": 20}, translation=[0, h / 2, 0])
        ]
        calls.append(
            mat_call(
                [barrel],
                dims.get("orig_color", (0.5, 0.36, 0.2)),
                roughness=0.7,
                name="Barrel Wood",
            )
        )
        return calls, (barrel,)

    def edit(names, dims, built):
        (barrel,) = built
        return [mat_call([barrel], (0.28, 0.17, 0.09), roughness=0.65, name="Dark Oak")]

    NAMES = [{}, {"barrel": "Barrel"}]
    DIMS = [
        {"h": 0.9, "r_mid": 0.36},
        {"h": 1.0, "r_mid": 0.4, "r_end": 0.26},
    ]
    PROMPTS = [
        "change its material to dark oak",
        "this barrel should be dark oak instead",
        "Repaint the barrel as dark oak.",
        "swap the barrel's material for dark oak",
        "for a dim cellar scene, darken this barrel to dark oak",
        "make it dark oak",
        "the simple wooden barrel needs a dark oak finish",
        "change the barrel's wood to a dark oak tone",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "barrel_material",
            "easy",
            prompt,
            prior,
            calls,
            "Single clay_material call on the barrel's own uid.",
        )


# ==========================================================================
# Spec 6 -- small clay pot :: shrink it to half its current size (easy)
# ==========================================================================


def spec_pot_shrink() -> None:
    def build(names, dims):
        pot = names.get("pot", "pot")
        h = dims["h"]
        r = dims["r"]
        profile = [(0.02, -h / 2), (r, -h / 6), (r * 0.92, h / 3), (r * 0.55, h / 2)]
        calls = [
            prim("lathe", pot, {"profile": profile, "segments": 18}, translation=[0, h / 2, 0])
        ]
        calls.append(
            mat_call([pot], dims.get("color", (0.62, 0.38, 0.24)), roughness=0.85, name="Clay")
        )
        return calls, (pot, h)

    def edit(names, dims, built):
        pot, h = built
        f = 0.5
        return [xform(pot, translation=[0, (h / 2) * f, 0], scale=[f, f, f])]

    NAMES = [{}, {"pot": "Pot"}]
    DIMS = [
        {"h": 0.3, "r": 0.14},
        {"h": 0.36, "r": 0.16},
    ]
    PROMPTS = [
        "shrink it to half its current size",
        "this pot is too big -- shrink it to half size",
        "Halve the size of the small clay pot.",
        "scale the pot down by 50 percent",
        "for a dollhouse diorama, shrink this pot to half its size",
        "shrink it 50%",
        "the small clay pot needs to be half as big overall",
        "make the pot half its current scale, uniformly",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "pot_shrink",
            "easy",
            prompt,
            prior,
            calls,
            "Uniform scale about the object's own centred origin; since "
            "translation.y already equals half the pot's height, the new "
            "translation.y is the old value times the same factor -- this "
            "keeps the pot's foot on the ground after the shrink.",
        )


# ==========================================================================
# Spec 7 -- plain metal pipe :: move it up by one metre (easy)
# ==========================================================================


def spec_pipe_move_up() -> None:
    def build(names, dims):
        pipe = names.get("pipe", "pipe")
        r = dims["r"]
        length = dims["length"]
        y0 = dims.get("y0", r)
        calls = [
            prim(
                "cylinder",
                pipe,
                {"radius": r, "height": length, "segments": 16},
                translation=[0, y0, 0],
                rotation=[0, 0, 90],
            )
        ]
        calls.append(
            mat_call(
                [pipe],
                dims.get("color", (0.6, 0.61, 0.63)),
                roughness=0.4,
                metallic=0.7,
                name="Steel Pipe",
            )
        )
        return calls, (pipe, y0)

    def edit(names, dims, built):
        pipe, y0 = built
        return [xform(pipe, translation=[0, y0 + 1.0, 0])]

    NAMES = [{}, {"pipe": "Pipe"}]
    DIMS = [
        {"r": 0.05, "length": 1.4},
        {"r": 0.06, "length": 1.6, "y0": 0.06},
    ]
    PROMPTS = [
        "move it up by one metre",
        "raise this pipe by 1m",
        "Move the plain metal pipe up one metre.",
        "lift it 1.0m higher",
        "for a scene where the pipe runs along a ceiling, move it up by one metre",
        "move up 1m",
        "the plain metal pipe should sit one metre higher than it does now",
        "raise the pipe's position by exactly one metre",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "pipe_move_up",
            "easy",
            prompt,
            prior,
            calls,
            "allow_below_ground not needed -- moving up only; a lone "
            "clay_transform translation edit.",
        )


# ==========================================================================
# Spec 8 -- wooden bookshelf :: delete the topmost shelf (easy)
# ==========================================================================


def spec_bookshelf_delete_shelf() -> None:
    def build(names, dims):
        case = names.get("case", "case")
        shelves = [names.get(f"shelf{i}", f"shelf_{i}") for i in range(3)]
        w, h, d = dims["case"]
        t = dims.get("t", 0.02)
        calls = [prim("box", case, {"size": [w, h, d]}, translation=[0, h / 2, 0])]
        n = len(shelves)
        for i, nm in enumerate(shelves):
            y = (i + 1) * h / (n + 1)
            calls.append(prim("box", nm, {"size": [w - 0.04, t, d - 0.04]}, translation=[0, y, 0]))
        calls.append(
            mat_call(
                [case, *shelves], dims.get("color", (0.4, 0.27, 0.16)), roughness=0.75, name="Pine"
            )
        )
        return calls, (case, shelves)

    def edit(names, dims, built):
        case, shelves = built
        top_shelf = shelves[-1]
        return [delete([top_shelf])]

    NAMES = [{}, {"case": "Case", "shelf0": "Shelf 1", "shelf1": "Shelf 2", "shelf2": "Shelf 3"}]
    DIMS = [
        {"case": (0.8, 1.6, 0.3)},
        {"case": (0.9, 1.8, 0.32), "t": 0.025},
    ]
    PROMPTS = [
        "delete the topmost shelf",
        "remove the top shelf from this bookshelf",
        "Delete the highest shelf on the bookshelf.",
        "take out the uppermost shelf",
        "for a bookshelf that needs to fit under a sloped ceiling, remove its topmost shelf",
        "delete top shelf",
        "the wooden bookshelf's topmost shelf should be removed",
        "get rid of the shelf nearest the top of the case",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "bookshelf_delete_shelf",
            "easy",
            prompt,
            prior,
            calls,
            "clay_delete on the highest-placed shelf board; case and the other two shelves remain.",
        )


# ==========================================================================
# Spec 9 -- round shield :: change its material to bronze (easy)
# ==========================================================================


def spec_shield_material() -> None:
    def build(names, dims):
        shield = names.get("shield", "shield")
        r = dims["r"]
        t = dims.get("t", 0.03)
        calls = [
            prim(
                "cylinder",
                shield,
                {"radius": r, "height": t, "segments": 28},
                translation=[0, 0.9, 0],
                rotation=[90, 0, 0],
            )
        ]
        calls.append(
            mat_call(
                [shield],
                dims.get("color", (0.5, 0.5, 0.52)),
                roughness=0.4,
                metallic=0.5,
                name="Iron",
            )
        )
        return calls, (shield,)

    def edit(names, dims, built):
        (shield,) = built
        return [
            mat_call([shield], (0.55, 0.36, 0.13), roughness=0.35, metallic=0.85, name="Bronze")
        ]

    NAMES = [{}, {"shield": "Shield"}]
    DIMS = [
        {"r": 0.32},
        {"r": 0.36, "t": 0.035},
    ]
    PROMPTS = [
        "change its material to bronze",
        "this shield should be bronze instead of iron",
        "Repaint the round shield as bronze.",
        "swap the shield's material for bronze",
        "for a Bronze Age warrior's kit, this shield should be bronze",
        "make it bronze",
        "the round shield needs a bronze finish",
        "change the shield's metal to bronze",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "shield_material",
            "easy",
            prompt,
            prior,
            calls,
            "Shield is a flattened cylinder standing upright (rotated 90 "
            "about X so its flat faces are vertical); one material call.",
            allow_below_ground=False,
        )


# ==========================================================================
# Spec 10 -- wooden cart :: rotate it ninety degrees around its vertical axis
# ==========================================================================


def spec_cart_rotate() -> None:
    def build(names, dims):
        bed = names.get("bed", "cart_bed")
        wl = names.get("wl", "wheel_left")
        wr = names.get("wr", "wheel_right")
        w, bh, d = dims["bed"]
        wr_r = dims["wheel_r"]
        wx = dims["wheel_x"]
        calls = [
            prim("box", bed, {"size": [w, bh, d]}, translation=[0, wr_r * 2 + bh / 2, 0]),
            prim(
                "cylinder",
                wl,
                {"radius": wr_r, "height": 0.08, "segments": 16},
                translation=[wx, wr_r, 0],
            ),
            prim(
                "cylinder",
                wr,
                {"radius": wr_r, "height": 0.08, "segments": 16},
                translation=[-wx, wr_r, 0],
            ),
        ]
        calls.append(
            mat_call(
                [bed, wl, wr],
                dims.get("color", (0.45, 0.3, 0.17)),
                roughness=0.75,
                name="Cart Wood",
            )
        )
        return calls, (bed, wl, wr, wx, wr_r)

    def edit(names, dims, built):
        bed, wl, wr, wx, wheel_r = built
        nx_l, nz_l = rot_y((wx, 0.0), 90)
        nx_r, nz_r = rot_y((-wx, 0.0), 90)
        return [
            xform(bed, rotation=[0, 90, 0]),
            xform(wl, translation=[nx_l, wheel_r, nz_l]),
            xform(wr, translation=[nx_r, wheel_r, nz_r]),
        ]

    NAMES = [{}, {"bed": "Cart Bed", "wl": "Wheel Left", "wr": "Wheel Right"}]
    DIMS = [
        {"bed": (0.9, 0.25, 0.55), "wheel_r": 0.22, "wheel_x": 0.5},
        {"bed": (1.0, 0.28, 0.6), "wheel_r": 0.25, "wheel_x": 0.55},
    ]
    PROMPTS = [
        "rotate it ninety degrees around its vertical axis",
        "turn this cart 90 degrees on the spot",
        "Rotate the wooden cart 90 degrees about its vertical axis.",
        "spin it a quarter turn around Y",
        "for a market-square diorama, this cart should face sideways -- rotate it ninety degrees",
        "rotate 90 degrees",
        "the wooden cart needs to be turned ninety degrees around its up axis",
        "turn the whole cart a quarter turn so it faces the other way",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "cart_rotate",
            "easy",
            prompt,
            prior,
            calls,
            "Wheels are upright cylinders (rotation identity) placed only "
            "along X either side of the bed, so rotating the whole rig 90 "
            "degrees about the world Y axis at the origin is: the bed's own "
            "rotation becomes [0,90,0], and each wheel's (x,z) position "
            "rotates the same 90 degrees in the XZ plane (a simple rot_y), "
            "with no quaternion composition needed since nothing had a "
            "starting rotation to compose with.",
        )


# ==========================================================================
# Spec 11 -- stone column :: make it half as tall (easy)
# ==========================================================================


def spec_column_half() -> None:
    def build(names, dims):
        col = names.get("col", "column")
        r, h = dims["r"], dims["h"]
        base, cap = dims.get("base", 0.15), dims.get("cap", 0.15)
        calls = [
            prim(
                "column",
                col,
                {"radius": r, "height": h, "segments": 16, "base": base, "capital": cap},
                translation=[0, h / 2, 0],
            )
        ]
        calls.append(
            mat_call([col], dims.get("color", (0.62, 0.6, 0.56)), roughness=0.85, name="Marble")
        )
        return calls, (col, h)

    def edit(names, dims, built):
        col, h = built
        new_h = h / 2
        return [set_params(col, {"height": new_h}), xform(col, translation=[0, new_h / 2, 0])]

    NAMES = [{}, {"col": "Column"}]
    DIMS = [
        {"r": 0.32, "h": 2.4},
        {"r": 0.36, "h": 2.8, "base": 0.2, "cap": 0.2},
    ]
    PROMPTS = [
        "make it half as tall",
        "this column is too tall -- halve its height",
        "Make the stone column half as tall.",
        "shrink its height by 50 percent",
        "for a ruined temple scene, this column should be half its current height",
        "half the height",
        "the stone column needs to be half as tall as it is now",
        "reduce the column's height to half its current value",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "column_half",
            "easy",
            prompt,
            prior,
            calls,
            "column is centred like every other generator; halving height means set_params then re-centring translation.y at new_h/2.",
        )


# ==========================================================================
# Spec 12 -- simple humanoid figure :: change its material to carved stone
# ==========================================================================

_HUMANOID_PARTS = [
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
]


def spec_humanoid_material() -> None:
    def build(names, dims):
        prefix = names.get("prefix", "h")
        part_names = [f"{prefix}{p}" for p in _HUMANOID_PARTS]
        calls = [figure("humanoid", prefix)]
        calls.append(
            mat_call(
                part_names, dims.get("color", (0.7, 0.55, 0.35)), roughness=0.7, name="Skin/Cloth"
            )
        )
        return calls, (part_names,)

    def edit(names, dims, built):
        (part_names,) = built
        return [mat_call(part_names, (0.55, 0.54, 0.52), roughness=0.9, name="Carved Stone")]

    NAMES = [{"prefix": "h"}, {"prefix": "Fig_"}]
    DIMS = [{}, {"color": (0.68, 0.5, 0.32)}]
    PROMPTS = [
        "change its material to carved stone",
        "turn this figure to carved stone",
        "Repaint the whole humanoid figure as carved stone.",
        "make it look like a stone statue",
        "for a museum-hall diorama, this figure should be carved stone rather than flesh and cloth",
        "make it stone",
        "the simple humanoid figure needs a carved-stone finish, all over",
        "recolor every part of this figure to a grey carved-stone material",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "humanoid_material",
            "easy",
            prompt,
            prior,
            calls,
            "clay_add_figure's humanoid preset mints 19 named parts as one "
            "grounded group; one clay_material call lists every part by "
            "$ref so the whole figure repaints as one undo step.",
        )


# ==========================================================================
# Spec 13 -- wooden crate :: move it two metres to the left (easy)
# ==========================================================================


def spec_crate_move() -> None:
    def build(names, dims):
        crate = names.get("crate", "crate")
        size = dims["size"]
        calls = [prim("box", crate, {"size": list(size)}, translation=[0, size[1] / 2, 0])]
        calls.append(
            mat_call(
                [crate], dims.get("color", (0.42, 0.28, 0.15)), roughness=0.8, name="Crate Wood"
            )
        )
        return calls, (crate, size)

    def edit(names, dims, built):
        crate, size = built
        return [xform(crate, translation=[-2.0, size[1] / 2, 0])]

    NAMES = [{}, {"crate": "Crate"}]
    DIMS = [{"size": (0.5, 0.45, 0.5)}, {"size": (0.55, 0.5, 0.5)}]
    PROMPTS = [
        "move it two metres to the left",
        "shift this crate 2m to the left",
        "Move the wooden crate two metres to the left.",
        "translate it -2m along X",
        "for a warehouse aisle layout, move this crate two metres further left",
        "move left 2m",
        "the wooden crate needs to move two metres to the left of where it is",
        "slide the crate two metres in the negative X direction",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "crate_move",
            "easy",
            prompt,
            prior,
            calls,
            "Single clay_transform translation edit; height unchanged so translation.y stays size[1]/2.",
        )


# ==========================================================================
# Spec 14 -- plain hammer :: lengthen the handle by twenty centimetres
# ==========================================================================


def spec_hammer_lengthen() -> None:
    def build(names, dims):
        handle = names.get("handle", "handle")
        head = names.get("head", "head")
        r = dims["r"]
        h = dims["h"]
        head_size = dims["head_size"]
        calls = [
            prim(
                "cylinder",
                handle,
                {"radius": r, "height": h, "segments": 12},
                translation=[0, h / 2, 0],
            ),
            prim("box", head, {"size": list(head_size)}, translation=[0, h + head_size[1] / 2, 0]),
        ]
        calls.append(
            mat_call(
                [handle],
                dims.get("wood_color", (0.5, 0.35, 0.2)),
                roughness=0.7,
                name="Handle Wood",
            )
        )
        calls.append(
            mat_call(
                [head],
                dims.get("metal_color", (0.5, 0.5, 0.52)),
                roughness=0.35,
                metallic=0.8,
                name="Hammer Steel",
            )
        )
        return calls, (handle, head, r, h, head_size)

    def edit(names, dims, built):
        handle, head, r, h, head_size = built
        new_h = h + 0.20
        return [
            set_params(handle, {"height": new_h}),
            xform(handle, translation=[0, new_h / 2, 0]),
            xform(head, translation=[0, new_h + head_size[1] / 2, 0]),
        ]

    NAMES = [{}, {"handle": "Handle", "head": "Head"}]
    DIMS = [
        {"r": 0.018, "h": 0.28, "head_size": (0.18, 0.06, 0.05)},
        {"r": 0.02, "h": 0.3, "head_size": (0.2, 0.065, 0.055)},
    ]
    PROMPTS = [
        "lengthen the handle by twenty centimetres",
        "this hammer's handle is too short -- add 20cm",
        "Lengthen the hammer handle by twenty centimetres.",
        "extend the handle 0.2m",
        "for a two-handed sledge-style hammer, lengthen the handle by twenty centimetres",
        "handle +20cm",
        "the plain hammer's handle needs to be twenty centimetres longer",
        "add twenty centimetres to the length of the handle only",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "hammer_lengthen",
            "easy",
            prompt,
            prior,
            calls,
            "Handle is a centred cylinder (set_params + recentring "
            "transform); the head sits on top and must move up by the full "
            "20cm to stay seated on the now-longer handle.",
        )


# ==========================================================================
# Spec 15 -- metal bucket :: change its material to polished copper (easy)
# ==========================================================================


def spec_bucket_material() -> None:
    def build(names, dims):
        bucket = names.get("bucket", "bucket")
        h = dims["h"]
        r_bot, r_top = dims["r_bot"], dims["r_top"]
        profile = [(r_bot, -h / 2), (r_top, h / 2)]
        calls = [
            prim("lathe", bucket, {"profile": profile, "segments": 20}, translation=[0, h / 2, 0])
        ]
        calls.append(
            mat_call(
                [bucket],
                dims.get("color", (0.55, 0.55, 0.57)),
                roughness=0.5,
                metallic=0.5,
                name="Tin",
            )
        )
        return calls, (bucket,)

    def edit(names, dims, built):
        (bucket,) = built
        return [
            mat_call(
                [bucket], (0.72, 0.42, 0.2), roughness=0.2, metallic=0.95, name="Polished Copper"
            )
        ]

    NAMES = [{}, {"bucket": "Bucket"}]
    DIMS = [{"h": 0.32, "r_bot": 0.16, "r_top": 0.2}, {"h": 0.35, "r_bot": 0.18, "r_top": 0.22}]
    PROMPTS = [
        "change its material to polished copper",
        "this bucket should be polished copper",
        "Repaint the metal bucket as polished copper.",
        "swap the bucket's material for polished copper",
        "for a blacksmith's shop display, this bucket should be polished copper",
        "make it copper",
        "the metal bucket needs a polished-copper finish",
        "change the bucket's metal to a shiny polished copper",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "bucket_material",
            "easy",
            prompt,
            prior,
            calls,
            "Bucket is a two-station lathe frustum; single material call.",
        )


# ==========================================================================
# Spec 16 -- low coffee table :: widen the tabletop by half a metre
# ==========================================================================


def spec_coffee_table_widen() -> None:
    def build(names, dims):
        top = names.get("top", "top")
        legs = [names.get(f"leg{i}", f"leg_{i}") for i in range(4)]
        r = dims["r"]
        t = dims["t"]
        leg_h = dims["leg_h"]
        ring = dims.get("ring", r * 0.75)
        calls = [
            prim(
                "cylinder",
                top,
                {"radius": r, "height": t, "segments": 28},
                translation=[0, leg_h + t / 2, 0],
            )
        ]
        for i, nm in enumerate(legs):
            ang = 45 + i * 90
            x = ring * math.cos(math.radians(ang))
            z = ring * math.sin(math.radians(ang))
            calls.append(
                prim(
                    "cylinder",
                    nm,
                    {"radius": 0.025, "height": leg_h, "segments": 10},
                    translation=[x, leg_h / 2, z],
                )
            )
        calls.append(
            mat_call(
                [top, *legs], dims.get("color", (0.38, 0.26, 0.15)), roughness=0.7, name="Walnut"
            )
        )
        return calls, (top, r, t, leg_h)

    def edit(names, dims, built):
        top, r, t, leg_h = built
        new_r = r + 0.25
        return [set_params(top, {"radius": new_r})]

    NAMES = [{}, {"top": "Top", "leg0": "Leg 1", "leg1": "Leg 2", "leg2": "Leg 3", "leg3": "Leg 4"}]
    DIMS = [{"r": 0.5, "t": 0.04, "leg_h": 0.35}, {"r": 0.55, "t": 0.035, "leg_h": 0.38}]
    PROMPTS = [
        "widen the tabletop by half a metre",
        "make this coffee table's top half a metre wider",
        "Widen the low coffee table's top by 0.5m.",
        "increase the tabletop's diameter by half a metre",
        "for a spacious living-room set, widen this coffee table's top by half a metre",
        "widen top +0.5m",
        "the low coffee table's top needs to be half a metre wider overall",
        "grow the round tabletop's diameter by 0.5 metres",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "coffee_table_widen",
            "easy",
            prompt,
            prior,
            calls,
            "Round top, so 'half a metre wider' is read as +0.25m radius "
            "(0.5m more diameter); radius does not affect vertical "
            "centring, so a single set_params suffices.",
        )


# ==========================================================================
# Spec 17 -- garden bench :: change its material to weathered grey wood
# ==========================================================================


def spec_bench_material() -> None:
    def build(names, dims):
        seat = names.get("seat", "seat")
        la = names.get("la", "leg_a")
        lb = names.get("lb", "leg_b")
        w, t, d = dims["seat"]
        leg_h = dims["leg_h"]
        calls = [
            prim("box", seat, {"size": [w, t, d]}, translation=[0, leg_h + t / 2, 0]),
            prim(
                "box",
                la,
                {"size": [0.06, leg_h, d - 0.05]},
                translation=[w / 2 - 0.05, leg_h / 2, 0],
            ),
            prim(
                "box",
                lb,
                {"size": [0.06, leg_h, d - 0.05]},
                translation=[-(w / 2 - 0.05), leg_h / 2, 0],
            ),
        ]
        calls.append(
            mat_call(
                [seat, la, lb],
                dims.get("color", (0.5, 0.36, 0.2)),
                roughness=0.75,
                name="Bench Wood",
            )
        )
        return calls, (seat, la, lb)

    def edit(names, dims, built):
        seat, la, lb = built
        return [
            mat_call([seat, la, lb], (0.55, 0.55, 0.54), roughness=0.95, name="Weathered Grey Wood")
        ]

    NAMES = [{}, {"seat": "Seat", "la": "Leg A", "lb": "Leg B"}]
    DIMS = [{"seat": (1.4, 0.06, 0.4), "leg_h": 0.42}, {"seat": (1.6, 0.05, 0.42), "leg_h": 0.44}]
    PROMPTS = [
        "change its material to weathered grey wood",
        "this bench should look weathered and grey",
        "Repaint the garden bench as weathered grey wood.",
        "swap the bench's material for weathered grey wood",
        "for an old cottage garden, this bench should be weathered grey wood, not fresh-cut",
        "make it weathered grey",
        "the garden bench needs a weathered, grey-wood finish",
        "change the bench's colour to a weathered grey",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "bench_material",
            "easy",
            prompt,
            prior,
            calls,
            "Trestle bench, seat plus two end legs; one material call covering all three.",
        )


# ==========================================================================
# Spec 18 -- stone plinth :: make it narrower (easy)
# ==========================================================================


def spec_plinth_narrow() -> None:
    def build(names, dims):
        plinth = names.get("plinth", "plinth")
        w, h, d = dims["size"]
        calls = [prim("box", plinth, {"size": [w, h, d]}, translation=[0, h / 2, 0])]
        calls.append(
            mat_call([plinth], dims.get("color", (0.58, 0.56, 0.52)), roughness=0.9, name="Stone")
        )
        return calls, (plinth, w, h, d)

    def edit(names, dims, built):
        plinth, w, h, d = built
        f = 0.7
        return [set_params(plinth, {"size": [w * f, h, d * f]})]

    NAMES = [{}, {"plinth": "Plinth"}]
    DIMS = [{"size": (0.6, 0.9, 0.6)}, {"size": (0.7, 1.0, 0.7)}]
    PROMPTS = [
        "make it narrower",
        "this plinth is too wide -- narrow it",
        "Make the stone plinth narrower.",
        "reduce its footprint, narrower on both sides",
        "for a bust display in a tight alcove, this plinth needs to be narrower",
        "narrow it",
        "the stone plinth needs a narrower footprint than it has now",
        "shrink the plinth's width and depth so it reads as narrower",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "plinth_narrow",
            "easy",
            prompt,
            prior,
            calls,
            "Shrinks size[0] and size[2] by 0.7x, leaving height untouched -- no translation change needed.",
        )


# ==========================================================================
# Spec 19 -- simple quadruped figure :: scale the whole figure up 50% (easy)
# ==========================================================================


def _quad_parts(names):
    return {
        "body": names.get("body", "q_body"),
        "head": names.get("head", "q_head"),
        "tail": names.get("tail", "q_tail"),
        "legs": [
            names.get(f"leg{i}", n)
            for i, n in enumerate(["q_leg_fl", "q_leg_fr", "q_leg_bl", "q_leg_br"])
        ],
    }


def _build_quadruped(names, dims):
    p = _quad_parts(names)
    body_size = dims["body"]
    leg_h = dims["leg_h"]
    leg_r = dims.get("leg_r", 0.035)
    half_span = dims["half_span"]
    body_y = leg_h + body_size[1] / 2
    head_r = dims["head_r"]
    tail_r = dims.get("tail_r", 0.03)
    tail_len = dims.get("tail_len", 0.22)
    calls = [prim("box", p["body"], {"size": list(body_size)}, translation=[0, body_y, 0])]
    x, z = half_span
    leg_positions = [(x, z), (-x, z), (x, -z), (-x, -z)]
    for nm, (lx, lz) in zip(p["legs"], leg_positions, strict=False):
        calls.append(
            prim(
                "cylinder",
                nm,
                {"radius": leg_r, "height": leg_h, "segments": 10},
                translation=[lx, leg_h / 2, lz],
            )
        )
    calls.append(
        prim(
            "icosphere",
            p["head"],
            {"radius": head_r, "subdivisions": 2},
            translation=[0, body_y, body_size[2] / 2 + head_r * 0.8],
        )
    )
    # tail: capsule pitched up and back from the body's rear, its own long
    # axis (local Y) rotated toward -Z, so rotation is about X only -- kept
    # to a single axis so later specs can compose an extra rotation simply.
    -math.sin(math.radians(35)) * (tail_len / 2)
    body_y + math.cos(math.radians(35)) * (tail_len / 2) * 0  # placeholder unused
    calls.append(
        prim(
            "capsule",
            p["tail"],
            {"radius": tail_r, "height": tail_len, "segments": 10, "rings": 3},
            translation=[
                0,
                body_y + 0.02,
                -(body_size[2] / 2 + tail_len / 2 * math.cos(math.radians(35))),
            ],
            rotation=[90 - 35, 0, 0],
        )
    )
    color = dims.get("color", (0.55, 0.42, 0.28))
    calls.append(
        mat_call([p["body"], p["head"], p["tail"], *p["legs"]], color, roughness=0.8, name="Hide")
    )
    return calls, p, body_y


def spec_quadruped_scale() -> None:
    def edit(p, dims, body_y):
        calls = []
        [p["body"], p["head"], p["tail"], *p["legs"]]
        {
            p["body"]: (0, body_y, 0),
            p["head"]: None,
            p["tail"]: None,
        }
        return calls  # placeholder, filled below per-record with real coords

    NAMES = [
        {},
        {
            "body": "Body",
            "head": "Head",
            "tail": "Tail",
            "leg0": "Leg FL",
            "leg1": "Leg FR",
            "leg2": "Leg BL",
            "leg3": "Leg BR",
        },
    ]
    DIMS = [
        {"body": (0.5, 0.28, 0.9), "leg_h": 0.35, "half_span": (0.18, 0.32), "head_r": 0.13},
        {"body": (0.55, 0.3, 1.0), "leg_h": 0.38, "half_span": (0.2, 0.35), "head_r": 0.14},
    ]
    PROMPTS = [
        "scale the whole figure up by fifty percent",
        "make this quadruped 50% bigger overall",
        "Scale the entire quadruped figure up by fifty percent.",
        "grow it by 1.5x in every direction",
        "for a dire-beast variant, scale this quadruped up by fifty percent",
        "scale up 50%",
        "the simple quadruped figure needs to be fifty percent bigger, uniformly",
        "make the whole animal one and a half times its current size",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, p, body_y = _build_quadruped(names, dims)
        f = 1.5
        body_size = dims["body"]
        head_r = dims["head_r"]
        half_span = dims["half_span"]
        leg_h = dims["leg_h"]
        tail_len = dims.get("tail_len", 0.22)
        head_pos = (0.0, body_y, body_size[2] / 2 + head_r * 0.8)
        tail_pos = (
            0.0,
            body_y + 0.02,
            -(body_size[2] / 2 + tail_len / 2 * math.cos(math.radians(35))),
        )
        x, z = half_span
        leg_pos = [(x, leg_h / 2, z), (-x, leg_h / 2, z), (x, leg_h / 2, -z), (-x, leg_h / 2, -z)]
        calls = [
            xform(p["body"], translation=[0, body_y * f, 0], scale=[f, f, f]),
            xform(p["head"], translation=[c * f for c in head_pos], scale=[f, f, f]),
            xform(p["tail"], translation=[c * f for c in tail_pos], scale=[f, f, f]),
        ]
        for nm, pos in zip(p["legs"], leg_pos, strict=False):
            calls.append(xform(nm, translation=[c * f for c in pos], scale=[f, f, f]))
        emit(
            "quadruped_scale",
            "easy",
            prompt,
            prior,
            calls,
            "Every part's own translation and scale multiplied by f=1.5 "
            "about the world origin: since each part was already grounded "
            "at translation.y == own half-height * own scale.y, the same "
            "factor on both keeps every foot on the ground after the "
            "uniform scale (bottom_new = f*half_height*scale_y - "
            "half_height*(f*scale_y) == 0 algebraically, independent of f).",
        )


# ==========================================================================
# Spec 20 -- wooden wardrobe :: darken its material (easy)
# ==========================================================================


def spec_wardrobe_darken() -> None:
    def build(names, dims):
        wardrobe = names.get("wardrobe", "wardrobe")
        w, h, d = dims["size"]
        orig = dims.get("orig_color", (0.55, 0.4, 0.25))
        calls = [prim("box", wardrobe, {"size": [w, h, d]}, translation=[0, h / 2, 0])]
        calls.append(mat_call([wardrobe], orig, roughness=0.7, name="Wardrobe Wood"))
        return calls, (wardrobe, orig)

    def edit(names, dims, built):
        wardrobe, orig = built
        darker = tuple(c * 0.45 for c in orig)
        return [mat_call([wardrobe], darker, roughness=0.7, name="Dark Wardrobe Wood")]

    NAMES = [{}, {"wardrobe": "Wardrobe"}]
    DIMS = [{"size": (1.0, 1.9, 0.6)}, {"size": (1.1, 2.0, 0.65), "orig_color": (0.6, 0.44, 0.28)}]
    PROMPTS = [
        "darken its material",
        "this wardrobe's wood is too light -- darken it",
        "Darken the wooden wardrobe's material.",
        "make the wardrobe's finish noticeably darker",
        "for a Victorian bedroom, this wardrobe's wood should be darker",
        "darken it",
        "the wooden wardrobe needs a darker wood finish",
        "reduce the wardrobe's material brightness so it reads darker",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "wardrobe_darken",
            "easy",
            prompt,
            prior,
            calls,
            "Darkening reads as a new material at 45% of the original color's brightness, same roughness.",
        )


# ==========================================================================
# Spec 21 -- metal gear disc :: shrink its radius by a third (easy)
# ==========================================================================


def spec_gear_shrink() -> None:
    def build(names, dims):
        gear = names.get("gear", "gear")
        r = dims["r"]
        t = dims.get("t", 0.03)
        calls = [
            prim(
                "cylinder",
                gear,
                {"radius": r, "height": t, "segments": 24},
                translation=[0, 0.5, 0],
                rotation=[90, 0, 0],
            )
        ]
        calls.append(
            mat_call(
                [gear],
                dims.get("color", (0.4, 0.41, 0.43)),
                roughness=0.45,
                metallic=0.75,
                name="Gear Steel",
            )
        )
        return calls, (gear, r)

    def edit(names, dims, built):
        gear, r = built
        return [set_params(gear, {"radius": r * (2 / 3)})]

    NAMES = [{}, {"gear": "Gear"}]
    DIMS = [{"r": 0.24}, {"r": 0.28, "t": 0.035}]
    PROMPTS = [
        "shrink its radius by a third",
        "this gear disc is too wide -- shrink its radius by a third",
        "Shrink the metal gear disc's radius by one third.",
        "reduce the radius to two-thirds of what it is",
        "for a smaller clockwork mechanism, shrink this gear's radius by a third",
        "radius -1/3",
        "the metal gear disc's radius needs to shrink by a third",
        "cut the gear's radius down by one third of its current size",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "gear_shrink",
            "easy",
            prompt,
            prior,
            calls,
            "Radius change on a flat disc does not move its own centre; a single set_params.",
        )


# ==========================================================================
# Spec 22 -- stone well :: raise the wall height by twenty centimetres
# ==========================================================================


def spec_well_raise() -> None:
    def build(names, dims):
        wall = names.get("wall", "well_wall")
        r_out, h = dims["r_out"], dims["h"]
        t = dims.get("t", 0.12)
        profile = [(r_out, -h / 2), (r_out, h / 2), (r_out - t, h / 2), (r_out - t, -h / 2)]
        calls = [
            prim("lathe", wall, {"profile": profile, "segments": 24}, translation=[0, h / 2, 0])
        ]
        calls.append(
            mat_call([wall], dims.get("color", (0.55, 0.53, 0.5)), roughness=0.9, name="Well Stone")
        )
        return calls, (wall, h)

    def edit(names, dims, built):
        wall, h = built
        new_h = h + 0.20
        r_out, t = dims["r_out"], dims.get("t", 0.12)
        profile = [
            (r_out, -new_h / 2),
            (r_out, new_h / 2),
            (r_out - t, new_h / 2),
            (r_out - t, -new_h / 2),
        ]
        return [set_params(wall, {"profile": profile}), xform(wall, translation=[0, new_h / 2, 0])]

    NAMES = [{}, {"wall": "Well Wall"}]
    DIMS = [{"r_out": 0.45, "h": 0.6}, {"r_out": 0.5, "h": 0.65, "t": 0.14}]
    PROMPTS = [
        "raise the wall height by twenty centimetres",
        "this well's wall is too low -- raise it 20cm",
        "Raise the stone well's wall height by twenty centimetres.",
        "add 0.2m to the well wall's height",
        "for a safety-conscious village well, raise the wall by twenty centimetres",
        "raise wall +20cm",
        "the stone well's wall needs to be twenty centimetres taller",
        "increase the height of the well's ring wall by twenty centimetres",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "well_raise",
            "easy",
            prompt,
            prior,
            calls,
            "Well wall is a lathe ring (a rectangular profile revolved); "
            "raising its height means rebuilding the profile at the new "
            "height and re-centring translation.y, since a lathe's profile "
            "is stored as absolute [radius, y] stations, not a separate "
            "height parameter.",
        )


# ==========================================================================
# Spec 23 -- plain wooden chest :: delete it from the scene (easy)
# ==========================================================================


def spec_chest_delete() -> None:
    def build(names, dims):
        body = names.get("body", "chest_body")
        lid = names.get("lid", "chest_lid")
        rug = names.get("rug", "floor_rug")
        w, h, d = dims["body"]
        lid_h = dims.get("lid_h", 0.08)
        calls = [
            prim("box", body, {"size": [w, h, d]}, translation=[0, h / 2, 0]),
            prim("box", lid, {"size": [w, lid_h, d]}, translation=[0, h + lid_h / 2, 0]),
            # A companion object the chest sits on, so deleting the chest
            # (both its parts) still leaves the document non-empty --
            # verify.py's rule 2 rejects any record with zero objects left,
            # and a record built as "one object, then delete it" can never
            # satisfy that, so every 'delete the whole thing' edit needs a
            # second object in the scene by construction.
            prim("box", rug, {"size": [w * 2.2, 0.01, d * 2.2]}, translation=[0, 0.005, 0]),
        ]
        calls.append(
            mat_call(
                [body, lid], dims.get("color", (0.45, 0.3, 0.17)), roughness=0.75, name="Chest Wood"
            )
        )
        calls.append(
            mat_call([rug], dims.get("rug_color", (0.5, 0.15, 0.15)), roughness=0.9, name="Rug")
        )
        return calls, (body, lid)

    def edit(names, dims, built):
        body, lid = built
        return [delete([body, lid])]

    NAMES = [{}, {"body": "Chest Body", "lid": "Chest Lid"}]
    DIMS = [{"body": (0.6, 0.35, 0.35)}, {"body": (0.7, 0.4, 0.4), "lid_h": 0.09}]
    PROMPTS = [
        "delete it from the scene",
        "remove this chest entirely",
        "Delete the plain wooden chest.",
        "get rid of the whole chest",
        "this chest is no longer needed in the scene -- delete it",
        "delete chest",
        "the plain wooden chest should be removed from the scene entirely",
        "take the wooden chest out of the document completely",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "chest_delete",
            "easy",
            prompt,
            prior,
            calls,
            "clay_delete on both parts (body and lid) as one call, one undo step, leaving no objects at all -- verify.py's 'no objects' rule is exactly why this must be its own record rather than folded into another chest spec.",
        )


# ==========================================================================
# Spec 24 -- round side table :: change its material to pale marble (easy)
# ==========================================================================


def spec_side_table_material() -> None:
    def build(names, dims):
        top = names.get("top", "top")
        leg = names.get("leg", "pedestal")
        r, t = dims["r"], dims["t"]
        leg_h, leg_r = dims["leg_h"], dims["leg_r"]
        calls = [
            prim(
                "cylinder",
                top,
                {"radius": r, "height": t, "segments": 24},
                translation=[0, leg_h + t / 2, 0],
            ),
            prim(
                "cylinder",
                leg,
                {"radius": leg_r, "height": leg_h, "segments": 16},
                translation=[0, leg_h / 2, 0],
            ),
        ]
        calls.append(
            mat_call(
                [top, leg],
                dims.get("color", (0.5, 0.35, 0.2)),
                roughness=0.7,
                name="Side Table Wood",
            )
        )
        return calls, (top, leg)

    def edit(names, dims, built):
        top, leg = built
        return [mat_call([top, leg], (0.85, 0.83, 0.8), roughness=0.25, name="Pale Marble")]

    NAMES = [{}, {"top": "Top", "leg": "Pedestal"}]
    DIMS = [
        {"r": 0.24, "t": 0.03, "leg_h": 0.48, "leg_r": 0.06},
        {"r": 0.26, "t": 0.035, "leg_h": 0.5, "leg_r": 0.07},
    ]
    PROMPTS = [
        "change its material to pale marble",
        "this side table should be pale marble",
        "Repaint the round side table as pale marble.",
        "swap the table's material for pale marble",
        "for a bright modern lounge, this side table should be pale marble",
        "make it pale marble",
        "the round side table needs a pale-marble finish",
        "change the side table's material to a pale, polished marble",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "side_table_material",
            "easy",
            prompt,
            prior,
            calls,
            "Pedestal side table (round top + single pedestal leg); one material call for both parts.",
        )


# ==========================================================================
# Spec 25 -- wooden footstool :: move it directly in front of the armchair
# ==========================================================================


def spec_footstool_move() -> None:
    def build(names, dims):
        chair = names.get("chair", "armchair")
        stool = names.get("stool", "footstool")
        chair_size = dims["chair"]
        stool_size = dims["stool"]
        chair_z = dims["chair_z"]
        stool_start_z = dims["stool_start_z"]
        calls = [
            prim(
                "box",
                chair,
                {"size": list(chair_size)},
                translation=[0, chair_size[1] / 2, chair_z],
            ),
            prim(
                "box",
                stool,
                {"size": list(stool_size)},
                translation=[0.8, stool_size[1] / 2, stool_start_z],
            ),
        ]
        calls.append(
            mat_call(
                [chair, stool],
                dims.get("color", (0.45, 0.3, 0.18)),
                roughness=0.7,
                name="Upholstered Wood",
            )
        )
        return calls, (chair, stool, chair_size, stool_size, chair_z)

    def edit(names, dims, built):
        chair, stool, chair_size, stool_size, chair_z = built
        new_z = chair_z + chair_size[2] / 2 + stool_size[2] / 2 + 0.05
        return [xform(stool, translation=[0, stool_size[1] / 2, new_z])]

    NAMES = [{}, {"chair": "Armchair", "stool": "Footstool"}]
    DIMS = [
        {
            "chair": (0.8, 0.85, 0.8),
            "stool": (0.45, 0.35, 0.4),
            "chair_z": 0.0,
            "stool_start_z": 1.4,
        },
        {
            "chair": (0.85, 0.9, 0.85),
            "stool": (0.5, 0.38, 0.42),
            "chair_z": 0.0,
            "stool_start_z": 1.5,
        },
    ]
    PROMPTS = [
        "move it directly in front of the armchair",
        "put this footstool right in front of the armchair",
        "Move the wooden footstool directly in front of the armchair.",
        "reposition the stool so it sits right in front of the chair",
        "for a reading nook, move the footstool directly in front of the armchair, close enough to rest feet on",
        "move footstool in front of armchair",
        "the wooden footstool needs to sit directly in front of the armchair now",
        "slide the footstool up against the front of the armchair",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "footstool_move",
            "easy",
            prompt,
            prior,
            calls,
            "Armchair is a stand-in box so the footstool has something to "
            "be positioned relative to; the new Z is the chair's front face "
            "plus half the stool's own depth plus a small clearance gap.",
        )


# ==========================================================================
# MEDIUM specs
# ==========================================================================


def spec_barrel_hole() -> None:
    def build(names, dims):
        barrel = names.get("barrel", "barrel")
        h = dims["h"]
        r_mid = dims["r_mid"]
        r_end = dims.get("r_end", r_mid * 0.75)
        profile = [(r_end, -h / 2), (r_mid, -h / 4), (r_mid, h / 4), (r_end, h / 2)]
        calls = [
            prim("lathe", barrel, {"profile": profile, "segments": 20}, translation=[0, h / 2, 0])
        ]
        calls.append(
            mat_call(
                [barrel], dims.get("color", (0.5, 0.36, 0.2)), roughness=0.7, name="Barrel Wood"
            )
        )
        return calls, (barrel, h, r_end)

    def edit(names, dims, built):
        barrel, h, r_end = built
        cutter = "lid_hole_cutter"
        cut_r = dims.get("cut_r", r_end * 0.35)
        calls = [
            prim(
                "cylinder",
                cutter,
                {"radius": cut_r, "height": 0.2, "segments": 16},
                translation=[0, h, 0],
            ),
            boolean("difference", [barrel, cutter]),
        ]
        return calls

    NAMES = [{}, {"barrel": "Barrel"}]
    DIMS = [{"h": 0.9, "r_mid": 0.36}, {"h": 1.0, "r_mid": 0.4}]
    PROMPTS = [
        "cut a round hole in the lid",
        "punch a round hole through the barrel's lid",
        "Cut a round hole through the top of the barrel.",
        "add a circular opening in the lid",
        "for a barrel used as a well-head, cut a round hole in the lid",
        "cut a hole in the lid",
        "the wooden barrel's lid needs a round hole cut through it",
        "open a round hole straight through the barrel's top",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "barrel_hole",
            "medium",
            prompt,
            prior,
            calls,
            "A cylinder cutter taller than the barrel's own wall, centred "
            "on the lid, then clay_boolean difference [barrel, cutter] -- "
            "the barrel stays first in document order (built in the prior) "
            "so it is the difference's survivor.",
        )


def spec_crates_delete_middle() -> None:
    def build(names, dims):
        a, b, c = (names.get("a", "crate_a"), names.get("b", "crate_b"), names.get("c", "crate_c"))
        size = dims["size"]
        gap = dims.get("gap", size[0] + 0.1)
        calls = [
            prim("box", a, {"size": list(size)}, translation=[-gap, size[1] / 2, 0]),
            prim("box", b, {"size": list(size)}, translation=[0, size[1] / 2, 0]),
            prim("box", c, {"size": list(size)}, translation=[gap, size[1] / 2, 0]),
        ]
        calls.append(
            mat_call(
                [a, b, c], dims.get("color", (0.42, 0.28, 0.15)), roughness=0.8, name="Crate Wood"
            )
        )
        return calls, (a, b, c)

    def edit(names, dims, built):
        a, b, c = built
        return [delete([b])]

    NAMES = [{}, {"a": "Crate A", "b": "Crate B", "c": "Crate C"}]
    DIMS = [{"size": (0.5, 0.45, 0.5)}, {"size": (0.55, 0.5, 0.55)}]
    PROMPTS = [
        "delete the middle one",
        "remove the middle crate from this row of three",
        "Delete the middle crate of the set of three.",
        "take out the centre crate, leave the other two",
        "for a scene where one crate was taken, delete the middle one of the three",
        "delete middle crate",
        "the set of three wooden crates needs its middle one removed",
        "get rid of just the centre crate in this row",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "crates_delete_middle",
            "medium",
            prompt,
            prior,
            calls,
            "Three crates in a row; clay_delete on the centre one only, leaving the outer two standing.",
        )


def spec_archway_window() -> None:
    def build(names, dims):
        arch_n = names.get("arch", "archway")
        w, h, d, t = dims["arch"]
        calls = [
            prim(
                "arch",
                arch_n,
                {"width": w, "height": h, "depth": d, "thickness": t, "segments": 16},
                translation=[0, h / 2, 0],
            )
        ]
        calls.append(
            mat_call([arch_n], dims.get("color", (0.6, 0.58, 0.55)), roughness=0.9, name="Stone")
        )
        return calls, (arch_n, w, h, d)

    def edit(names, dims, built):
        arch_n, w, h, d = built
        cutter = "keystone_window"
        calls = [
            prim(
                "box",
                cutter,
                {"size": [w * 0.14, w * 0.14, d * 2]},
                translation=[0, h - w * 0.5, 0],
            ),
            boolean("difference", [arch_n, cutter]),
        ]
        return calls

    NAMES = [{}, {"arch": "Archway"}]
    DIMS = [{"arch": (1.4, 2.2, 0.5, 0.3)}, {"arch": (1.6, 2.4, 0.55, 0.32)}]
    PROMPTS = [
        "cut a small window into the keystone",
        "punch a small square window through the archway's keystone",
        "Cut a small window into the archway's keystone.",
        "add a tiny opening at the crown of the arch",
        "for a gatehouse with an arrow-slit above the arch, cut a small window into the keystone",
        "cut small window in keystone",
        "the stone archway needs a small window cut into its keystone",
        "open a small square window through the top of the arch",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "archway_window",
            "medium",
            prompt,
            prior,
            calls,
            "A box cutter through the crown of the arch (at y = h - width/2, "
            "the arch's own springline-to-crown midline), long enough in Z "
            "to pierce the arch's own depth twice over, then a boolean "
            "difference.",
        )


def spec_locker_vent() -> None:
    def build(names, dims):
        locker = names.get("locker", "locker")
        w, h, d = dims["size"]
        calls = [prim("box", locker, {"size": [w, h, d]}, translation=[0, h / 2, 0])]
        calls.append(
            mat_call(
                [locker],
                dims.get("color", (0.35, 0.4, 0.42)),
                roughness=0.55,
                metallic=0.5,
                name="Locker Steel",
            )
        )
        return calls, (locker, w, h, d)

    def edit(names, dims, built):
        locker, w, h, d = built
        cutter = "vent_cutter"
        cw, ch = dims.get("vent", (w * 0.4, h * 0.15))
        calls = [
            prim("box", cutter, {"size": [cw, ch, d * 2]}, translation=[0, h * 0.6, 0]),
            boolean("difference", [locker, cutter]),
        ]
        return calls

    NAMES = [{}, {"locker": "Locker"}]
    DIMS = [{"size": (0.5, 1.6, 0.5)}, {"size": (0.55, 1.7, 0.55)}]
    PROMPTS = [
        "cut a rectangular vent into the door",
        "punch a rectangular vent through the locker door",
        "Cut a rectangular vent into the storage locker's door.",
        "add a wide vent slot to the front",
        "for a gym locker that needs airflow, cut a rectangular vent into the door",
        "cut vent in door",
        "the metal storage locker's door needs a rectangular vent cut into it",
        "open a rectangular vent through the front of the locker",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "locker_vent",
            "medium",
            prompt,
            prior,
            calls,
            "A wide, short box cutter through the locker's front face, then a boolean difference.",
        )


def spec_chest_bevel() -> None:
    """Seed line: "a treasure chest :: bevel all the top edges" -- reinterpreted.

    Bevel is an edge-mode op (``clay_op`` name ``bevel``), reachable only
    after ``clay_element_mode``/``clay_select_by``, neither of which the
    compact tool card documents (see the module docstring). Rather than
    train the model to call an undocumented tool, this record keeps the
    same subject and class but swaps the request for one the 13-tool card
    can actually do: cutting a decorative recessed band into the lid, which
    still reads as "treat the lid's edge", using clay_boolean the way the
    other medium specs do.
    """

    def build(names, dims):
        body = names.get("body", "chest_body")
        lid = names.get("lid", "chest_lid")
        w, h, d = dims["body"]
        lid_h = dims.get("lid_h", 0.09)
        calls = [
            prim("box", body, {"size": [w, h, d]}, translation=[0, h / 2, 0]),
            prim("box", lid, {"size": [w, lid_h, d]}, translation=[0, h + lid_h / 2, 0]),
        ]
        calls.append(
            mat_call(
                [body, lid], dims.get("color", (0.5, 0.34, 0.18)), roughness=0.7, name="Chest Wood"
            )
        )
        return calls, (body, lid, w, h, d, lid_h)

    def edit(names, dims, built):
        body, lid, w, h, d, lid_h = built
        cutter = "lid_edge_groove"
        calls = [
            prim(
                "box",
                cutter,
                {"size": [w - 0.06, lid_h * 0.4, d - 0.06]},
                translation=[0, h + lid_h - lid_h * 0.2, 0],
            ),
            boolean("difference", [lid, cutter]),
        ]
        return calls

    NAMES = [{}, {"body": "Chest Body", "lid": "Chest Lid"}]
    DIMS = [{"body": (0.6, 0.35, 0.35)}, {"body": (0.7, 0.4, 0.4), "lid_h": 0.1}]
    PROMPTS = [
        "cut a recessed groove around the lid's top edge",
        "recess a groove along the top of the treasure chest's lid",
        "Cut a decorative recessed band into the top of the lid.",
        "carve a shallow groove around the lid's rim",
        "for a treasure chest with a banded-iron look, cut a recessed groove around the lid's top edge",
        "groove the lid's top edge",
        "the treasure chest's lid needs a recessed groove cut around its top edge",
        "cut a thin recessed band into the lid, following its top edge",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "chest_bevel",
            "medium",
            prompt,
            prior,
            calls,
            "The seed asked for a bevel on the lid's top edges; bevel is an "
            "edge-mode op and the compact tool card has no documented way "
            "to enter element mode (clay_element_mode/clay_select_by carry "
            "no schema on the card), so this substitutes a boolean-cut "
            "recessed groove -- a genuinely achievable edit with the same "
            "subject and a comparable 'dress up the lid's edge' intent. "
            "Flagged in the report as a tool-surface gap.",
        )


def spec_table_leg_lathe() -> None:
    def build(names, dims):
        top, la, lb, lc, ld = (
            names.get("top", "tabletop"),
            names.get("la", "leg_fl"),
            names.get("lb", "leg_fr"),
            names.get("lc", "leg_bl"),
            names.get("ld", "leg_br"),
        )
        w, thick, d = dims["top"]
        leg_h = dims["leg_h"]
        leg_w = dims.get("leg_w", 0.06)
        x, z = dims["half_span"]
        calls = [prim("box", top, {"size": [w, thick, d]}, translation=[0, leg_h + thick / 2, 0])]
        for nm, (sx, sz) in ((la, (x, z)), (lb, (-x, z)), (lc, (x, -z)), (ld, (-x, -z))):
            calls.append(
                prim("box", nm, {"size": [leg_w, leg_h, leg_w]}, translation=[sx, leg_h / 2, sz])
            )
        calls.append(
            mat_call(
                [top, la, lb, lc, ld],
                dims.get("color", (0.42, 0.28, 0.16)),
                roughness=0.75,
                name="Oak",
            )
        )
        return calls, (top, la, lb, lc, ld, leg_h, leg_w, dims["half_span"])

    def edit(names, dims, built):
        top, la, lb, lc, ld, leg_h, leg_w, half_span = built
        x, z = half_span
        turned = "leg_fl_turned"
        r_top = leg_w * 0.8
        r_mid = leg_w * 1.1
        r_bot = leg_w * 0.5
        profile = [
            (r_bot, -leg_h / 2),
            (r_mid, -leg_h * 0.1),
            (r_mid, leg_h * 0.1),
            (r_top, leg_h / 2),
        ]
        return [
            delete([la]),
            prim(
                "lathe", turned, {"profile": profile, "segments": 16}, translation=[x, leg_h / 2, z]
            ),
        ]

    NAMES = [{}, {"top": "Top", "la": "Leg FL", "lb": "Leg FR", "lc": "Leg BL", "ld": "Leg BR"}]
    DIMS = [
        {"top": (0.9, 0.04, 0.55), "leg_h": 0.72, "half_span": (0.4, 0.24)},
        {"top": (1.0, 0.035, 0.6), "leg_h": 0.74, "half_span": (0.45, 0.27)},
    ]
    PROMPTS = [
        "replace the front-left leg with a lathe-turned one",
        "swap the front-left leg for a turned one",
        "Replace the front-left leg of the table with a lathe-turned leg.",
        "give the front-left leg a turned, decorative profile instead of a plain square one",
        "for a fancier reproduction table, replace the front-left leg with a lathe-turned one",
        "swap FL leg for a turned leg",
        "the wooden table's front-left leg needs to be replaced with a lathe-turned leg",
        "delete the plain front-left leg and add a turned one in its place",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "table_leg_lathe",
            "medium",
            prompt,
            prior,
            calls,
            "clay_delete on the old box leg, then clay_add_primitive with "
            "generator lathe at the exact same (x, leg_h/2, z) the old leg "
            "occupied -- a turned profile bulging slightly at the middle.",
        )


def spec_shield_boss_mirror() -> None:
    def build(names, dims):
        shield = names.get("shield", "shield")
        boss = names.get("boss", "boss")
        r = dims["r"]
        boss_r = dims["boss_r"]
        boss_x = dims["boss_x"]
        calls = [
            prim(
                "cylinder",
                shield,
                {"radius": r, "height": 0.03, "segments": 28},
                translation=[0, 0.9, 0],
                rotation=[90, 0, 0],
            ),
            prim(
                "icosphere",
                boss,
                {"radius": boss_r, "subdivisions": 2},
                translation=[boss_x, 0.9, -0.02],
            ),
        ]
        calls.append(
            mat_call(
                [shield, boss],
                dims.get("color", (0.5, 0.36, 0.14)),
                roughness=0.4,
                metallic=0.7,
                name="Bronze",
            )
        )
        return calls, (shield, boss)

    def edit(names, dims, built):
        shield, boss = built
        return [select([boss]), op("mirror-copy", {"axis": 0, "offset": 0})]

    NAMES = [{}, {"shield": "Shield", "boss": "Boss"}]
    DIMS = [
        {"r": 0.34, "boss_r": 0.06, "boss_x": 0.16},
        {"r": 0.36, "boss_r": 0.07, "boss_x": 0.18},
    ]
    PROMPTS = [
        "mirror a second boss onto the opposite side",
        "add a matching boss mirrored to the other side of the shield",
        "Mirror the domed boss onto the opposite side of the shield.",
        "give the shield a second, mirrored boss",
        "for a symmetrical ceremonial shield, mirror the boss onto the opposite side",
        "mirror the boss",
        "the shield's domed boss needs a mirrored twin on the opposite side",
        "duplicate the boss, reflected to the shield's other side",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "shield_boss_mirror",
            "medium",
            prompt,
            prior,
            calls,
            "clay_select on the boss alone, then clay_op mirror-copy across "
            "the world X=0 plane (the shield's own centreline) -- matches "
            "the manual's 'mirroring a limb across a body's centre-line' "
            "description of this op exactly.",
        )


def spec_wall_doorway() -> None:
    def build(names, dims):
        wall = names.get("wall", "wall")
        w, h, t = dims["size"]
        calls = [prim("box", wall, {"size": [w, h, t]}, translation=[0, h / 2, 0])]
        calls.append(
            mat_call(
                [wall], dims.get("color", (0.58, 0.56, 0.53)), roughness=0.9, name="Wall Stone"
            )
        )
        return calls, (wall, w, h, t)

    def edit(names, dims, built):
        wall, w, h, t = built
        box_cut = "door_box"
        arch_cut = "door_arch"
        door_w = dims.get("door_w", 0.9)
        door_h = dims.get("door_h", 1.6)
        calls = [
            prim("box", box_cut, {"size": [door_w, door_h, t * 2]}, translation=[0, door_h / 2, 0]),
            prim(
                "cylinder",
                arch_cut,
                {"radius": door_w / 2, "height": t * 2, "segments": 16},
                translation=[0, door_h, 0],
                rotation=[90, 0, 0],
            ),
            boolean("union", [box_cut, arch_cut]),
            boolean("difference", [wall, box_cut]),
        ]
        return calls

    NAMES = [{}, {"wall": "Wall"}]
    DIMS = [{"size": (2.4, 2.0, 0.3)}, {"size": (2.6, 2.2, 0.32)}]
    PROMPTS = [
        "cut an arched doorway through the middle",
        "open an arched doorway through the centre of this wall",
        "Cut an arched doorway through the middle of the stone wall.",
        "add a rounded-top doorway in the centre",
        "for a courtyard wall, cut an arched doorway through the middle",
        "cut arched doorway",
        "the stone wall needs an arched doorway cut through its middle",
        "punch a rounded-top opening through the centre of the wall",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "wall_doorway",
            "medium",
            prompt,
            prior,
            calls,
            "The doorway cutter is a box (the straight jambs) unioned with "
            "a half-buried cylinder (the rounded head) before it is "
            "differenced from the wall -- two clay_boolean calls in one "
            "edit, union then difference, both documented tools.",
        )


def spec_ladder_rung() -> None:
    def build(names, dims):
        ra = names.get("ra", "rail_left")
        rb = names.get("rb", "rail_right")
        rungs = [names.get(f"rung{i}", f"rung_{i}") for i in range(4)]
        h = dims["h"]
        width = dims["width"]
        n = len(rungs)
        spacing = h / (n + 1)
        calls = [
            prim("box", ra, {"size": [0.05, h, 0.03]}, translation=[width / 2, h / 2, 0]),
            prim("box", rb, {"size": [0.05, h, 0.03]}, translation=[-width / 2, h / 2, 0]),
        ]
        for i, nm in enumerate(rungs):
            y = spacing * (i + 1)
            calls.append(
                prim("box", nm, {"size": [width - 0.02, 0.04, 0.04]}, translation=[0, y, 0])
            )
        calls.append(
            mat_call(
                [ra, rb, *rungs],
                dims.get("color", (0.5, 0.35, 0.2)),
                roughness=0.75,
                name="Ladder Wood",
            )
        )
        return calls, (ra, rb, rungs, h, spacing)

    def edit(names, dims, built):
        ra, rb, rungs, h, spacing = built
        new_rung = "rung_new"
        new_y = spacing * (len(rungs) + 1)
        return [
            prim(
                "box",
                new_rung,
                {"size": [dims["width"] - 0.02, 0.04, 0.04]},
                translation=[0, new_y, 0],
            )
        ]

    NAMES = [{}, {"ra": "Rail Left", "rb": "Rail Right"}]
    DIMS = [{"h": 2.0, "width": 0.4}, {"h": 2.2, "width": 0.42}]
    PROMPTS = [
        "add one more evenly spaced rung",
        "this ladder needs another rung, evenly spaced with the rest",
        "Add one more rung to the ladder, evenly spaced.",
        "give it an extra rung at the top, matching the spacing",
        "for a taller reach, add one more evenly spaced rung to this ladder",
        "add a rung",
        "the wooden ladder needs one more rung, spaced the same as the others",
        "add another evenly spaced rung above the topmost one",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "ladder_rung",
            "medium",
            prompt,
            prior,
            calls,
            "One clay_add_primitive box at the next evenly-spaced station "
            "above the existing top rung -- the prior's own spacing "
            "(h/(n+1)) extended by one more step.",
        )


def spec_pipe_branch() -> None:
    def build(names, dims):
        pipe = names.get("pipe", "pipe")
        r = dims["r"]
        length = dims["length"]
        y0 = r
        calls = [
            prim(
                "cylinder",
                pipe,
                {"radius": r, "height": length, "segments": 16},
                translation=[0, y0, 0],
                rotation=[0, 0, 90],
            )
        ]
        calls.append(
            mat_call(
                [pipe],
                dims.get("color", (0.6, 0.61, 0.63)),
                roughness=0.4,
                metallic=0.7,
                name="Steel Pipe",
            )
        )
        return calls, (pipe, r, y0)

    def edit(names, dims, built):
        pipe, r, y0 = built
        cutter = "branch_socket_cutter"
        socket_r = dims.get("socket_r", r * 0.5)
        return [
            prim(
                "cylinder",
                cutter,
                {"radius": socket_r, "height": r * 3, "segments": 12},
                translation=[0.1, y0, 0],
            ),
            boolean("difference", [pipe, cutter]),
        ]

    NAMES = [{}, {"pipe": "Pipe"}]
    DIMS = [{"r": 0.06, "length": 1.4}, {"r": 0.07, "length": 1.6}]
    PROMPTS = [
        "cut a branch socket into its side",
        "punch a small branch socket into the side of this pipe",
        "Cut a branch socket into the side of the metal pipe.",
        "add a small round opening on its side for a branch line",
        "for a plumbing run that needs a tee later, cut a branch socket into its side",
        "cut branch socket",
        "the metal pipe needs a branch socket cut into its side",
        "open a small socket through the side wall of the pipe",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "pipe_branch",
            "medium",
            prompt,
            prior,
            calls,
            "A short cutter cylinder, its own axis vertical (perpendicular "
            "to the main pipe's horizontal axis), poking through the pipe "
            "wall at one point along its length, then a boolean difference.",
        )


def spec_dresser_handles() -> None:
    def build(names, dims):
        case = names.get("case", "case")
        drawers = [names.get(f"drawer{i}", f"drawer_{i}") for i in range(3)]
        w, h, d = dims["case"]
        dh = h / 3
        calls = [prim("box", case, {"size": [w, h, d]}, translation=[0, h / 2, 0])]
        for i, nm in enumerate(drawers):
            y = dh * i + dh / 2
            calls.append(
                prim(
                    "box",
                    nm,
                    {"size": [w - 0.03, dh - 0.02, 0.03]},
                    translation=[0, y, d / 2 + 0.014],
                )
            )
        calls.append(
            mat_call(
                [case, *drawers],
                dims.get("color", (0.42, 0.28, 0.16)),
                roughness=0.75,
                name="Dresser Wood",
            )
        )
        return calls, (case, drawers, dh)

    def edit(names, dims, built):
        case, drawers, dh = built
        calls = []
        for i, dr in enumerate(drawers):
            cutter = f"handle_cut_{i}"
            y = dh * i + dh / 2
            calls.append(
                prim(
                    "box",
                    cutter,
                    {"size": [0.08, 0.015, 0.06]},
                    translation=[0, y, dims["case"][2] / 2 + 0.02],
                )
            )
            calls.append(boolean("difference", [dr, cutter]))
        return calls

    NAMES = [
        {},
        {"case": "Case", "drawer0": "Drawer 1", "drawer1": "Drawer 2", "drawer2": "Drawer 3"},
    ]
    DIMS = [{"case": (0.8, 0.9, 0.45)}, {"case": (0.85, 0.95, 0.48)}]
    PROMPTS = [
        "cut a recessed handle into each drawer front",
        "recess a handle groove into every drawer front",
        "Cut a recessed handle pull into each of the three drawer fronts.",
        "add a finger-pull recess to each drawer",
        "for a handle-free minimalist look, cut a recessed handle into each drawer front",
        "recess handles into drawers",
        "the wooden dresser needs a recessed handle cut into each drawer front",
        "cut a small recessed pull into every drawer, one each",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "dresser_handles",
            "medium",
            prompt,
            prior,
            calls,
            "Three separate clay_boolean difference calls, one per drawer -- "
            "folding all three drawers and their cutters into a single "
            "boolean call would merge the drawers into one object, which "
            "is not what 'each drawer front' means.",
        )


def spec_wagon_wheel_spokes() -> None:
    def build(names, dims):
        hub = names.get("hub", "hub")
        spoke = names.get("spoke", "spoke")
        hub_r, hub_h = dims["hub_r"], dims["hub_h"]
        spoke_len = dims["spoke_len"]
        calls = [
            prim(
                "cylinder",
                hub,
                {"radius": hub_r, "height": hub_h, "segments": 16},
                translation=[0, 0.5, 0],
                rotation=[90, 0, 0],
            ),
            prim(
                "box",
                spoke,
                {"size": [0.03, 0.03, spoke_len]},
                translation=[0, 0.5, hub_r + spoke_len / 2],
                rotation=[90, 0, 0],
            ),
        ]
        calls.append(
            mat_call(
                [hub, spoke],
                dims.get("color", (0.45, 0.3, 0.17)),
                roughness=0.75,
                name="Wagon Wood",
            )
        )
        return calls, (hub, spoke)

    def edit(names, dims, built):
        hub, spoke = built
        return [select([spoke]), op("array-radial", {"count": 8, "angle": 360, "axis": 1})]

    NAMES = [{}, {"hub": "Hub", "spoke": "Spoke"}]
    DIMS = [
        {"hub_r": 0.12, "hub_h": 0.1, "spoke_len": 0.5},
        {"hub_r": 0.14, "hub_h": 0.11, "spoke_len": 0.55},
    ]
    PROMPTS = [
        "array eight identical spokes around the hub",
        "spin eight identical spokes around this wheel's hub",
        "Array eight spokes evenly around the wagon wheel's hub.",
        "give the wheel eight evenly spaced spokes radiating from the hub",
        "for a heavy freight wagon wheel, array eight identical spokes around the hub",
        "array 8 spokes",
        "the round wagon wheel needs eight identical spokes arrayed around its hub",
        "duplicate the spoke eight times in a full circle around the hub",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "wagon_wheel_spokes",
            "medium",
            prompt,
            prior,
            calls,
            "The hub sits at the world origin (0, 0.5, 0) with the spoke "
            "beside it, exactly as array-radial's own documentation asks "
            "for; clay_select on the spoke alone, then array-radial about "
            "the Y axis for a full 360-degree wheel of eight.",
        )


def spec_column_groove() -> None:
    def build(names, dims):
        col = names.get("col", "column")
        r, h = dims["r"], dims["h"]
        calls = [
            prim(
                "column",
                col,
                {"radius": r, "height": h, "segments": 16, "base": 0.15, "capital": 0.15},
                translation=[0, h / 2, 0],
            )
        ]
        calls.append(
            mat_call([col], dims.get("color", (0.62, 0.6, 0.56)), roughness=0.85, name="Marble")
        )
        return calls, (col, r, h)

    def edit(names, dims, built):
        col, r, h = built
        cutter = "groove_cutter"
        return [
            prim(
                "torus",
                cutter,
                {"radius": r, "tube": r * 0.12, "segments": 24, "sides": 12},
                translation=[0, h * 0.5, 0],
            ),
            boolean("difference", [col, cutter]),
        ]

    NAMES = [{}, {"col": "Column"}]
    DIMS = [{"r": 0.3, "h": 2.2}, {"r": 0.34, "h": 2.5}]
    PROMPTS = [
        "cut a decorative groove ring around its shaft",
        "carve a ring groove around the middle of the shaft",
        "Cut a decorative groove ring around the column's shaft.",
        "add a shallow ring-shaped groove circling the shaft",
        "for a classical order with an astragal, cut a decorative groove ring around the shaft",
        "cut groove ring",
        "the stone column needs a decorative groove ring cut around its shaft",
        "carve a thin ring around the shaft, cut into the stone",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "column_groove",
            "medium",
            prompt,
            prior,
            calls,
            "A torus, radius matching the shaft, centred halfway up it -- "
            "the tube pokes both in and out of the shaft's own surface -- "
            "then a boolean difference cuts the ring where it overlaps.",
        )


def spec_quadruped_tail_tube() -> None:
    def edit(p, dims, body_y):
        pass

    NAMES = [
        {},
        {
            "body": "Body",
            "head": "Head",
            "tail": "Tail",
            "leg0": "Leg FL",
            "leg1": "Leg FR",
            "leg2": "Leg BL",
            "leg3": "Leg BR",
        },
    ]
    DIMS = [
        {"body": (0.5, 0.28, 0.9), "leg_h": 0.35, "half_span": (0.18, 0.32), "head_r": 0.13},
        {"body": (0.55, 0.3, 1.0), "leg_h": 0.38, "half_span": (0.2, 0.35), "head_r": 0.14},
    ]
    PROMPTS = [
        "lengthen its tail into a tube shape",
        "give this quadruped a longer, tube-shaped tail",
        "Replace the quadruped's tail with a longer tube shape.",
        "swap the short tail for a longer tube-shaped one",
        "for a lizard-like variant, lengthen the tail into a tube shape",
        "longer tube tail",
        "the simple quadruped figure's tail needs to be longer and tube-shaped",
        "make the tail longer, built as a tube rather than a short capsule",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, p, body_y = _build_quadruped(names, dims)
        body_size = dims["body"]
        tail_len_old = dims.get("tail_len", 0.22)
        root_z = -(body_size[2] / 2)
        root_y = body_y + 0.02
        new_len = tail_len_old * 2.2
        path = [
            (0.0, root_y, root_z),
            (0.0, root_y - 0.03, root_z - new_len * 0.4),
            (0.0, root_y - 0.02, root_z - new_len * 0.75),
            (0.0, root_y + 0.01, root_z - new_len),
        ]
        new_tail = "q_tail_tube"
        calls = [
            delete([p["tail"]]),
            prim(
                "tube",
                new_tail,
                {"path": path, "radius": 0.03, "sides": 10},
                translation=path_centroid(path),
            ),
        ]
        emit(
            "quadruped_tail_tube",
            "medium",
            prompt,
            prior,
            calls,
            "Deletes the short capsule tail and adds a tube generator "
            "instead, its path a gently curving run from the same root "
            "point at the body's rear out to roughly twice the old tail's "
            "length -- a tube's own radius is a single number (no per-point "
            "taper), which is exactly the 'into a tube shape' the prompt "
            "asks for rather than a taper.",
        )


def spec_bench_divots() -> None:
    def build(names, dims):
        seat = names.get("seat", "seat")
        la = names.get("la", "leg_a")
        lb = names.get("lb", "leg_b")
        w, t, d = dims["seat"]
        leg_h = dims["leg_h"]
        calls = [
            prim("box", seat, {"size": [w, t, d]}, translation=[0, leg_h + t / 2, 0]),
            prim(
                "box",
                la,
                {"size": [0.06, leg_h, d - 0.05]},
                translation=[w / 2 - 0.05, leg_h / 2, 0],
            ),
            prim(
                "box",
                lb,
                {"size": [0.06, leg_h, d - 0.05]},
                translation=[-(w / 2 - 0.05), leg_h / 2, 0],
            ),
        ]
        calls.append(
            mat_call(
                [seat, la, lb],
                dims.get("color", (0.5, 0.36, 0.2)),
                roughness=0.75,
                name="Bench Wood",
            )
        )
        return calls, (seat, w, t, leg_h)

    def edit(names, dims, built):
        seat, w, t, leg_h = built
        divot_r = dims.get("divot_r", 0.09)
        xs = [-w * 0.28, 0.0, w * 0.28]
        calls = []
        cutters = []
        for i, x in enumerate(xs):
            nm = f"divot_{i}"
            calls.append(
                prim(
                    "uv_sphere",
                    nm,
                    {"radius": divot_r, "segments": 14, "rings": 8},
                    translation=[x, leg_h + t - divot_r * 0.5, 0],
                )
            )
            cutters.append(nm)
        calls.append(boolean("difference", [seat, *cutters]))
        return calls

    NAMES = [{}, {"seat": "Seat", "la": "Leg A", "lb": "Leg B"}]
    DIMS = [{"seat": (1.4, 0.06, 0.4), "leg_h": 0.42}, {"seat": (1.6, 0.05, 0.42), "leg_h": 0.44}]
    PROMPTS = [
        "cut identical seat divots along its length",
        "carve a row of matching divots into this bench's seat",
        "Cut a row of identical seat divots along the bench's length.",
        "add three evenly spaced scooped divots along the seat",
        "for a rustic hand-carved bench, cut identical seat divots along its length",
        "cut seat divots",
        "the wooden bench needs identical seat divots cut along its length",
        "scoop out three matching divots evenly along the seat's length",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "bench_divots",
            "medium",
            prompt,
            prior,
            calls,
            "Three sphere cutters sunk just into the seat's top face at "
            "even intervals along its length, all differenced from the "
            "single seat object in one clay_boolean call.",
        )


def spec_grate_holes() -> None:
    def build(names, dims):
        grate = names.get("grate", "grate")
        w, t, d = dims["size"]
        calls = [prim("box", grate, {"size": [w, t, d]}, translation=[0, t / 2, 0])]
        calls.append(
            mat_call(
                [grate],
                dims.get("color", (0.35, 0.36, 0.38)),
                roughness=0.55,
                metallic=0.6,
                name="Grate Iron",
            )
        )
        return calls, (grate, w, t, d)

    def edit(names, dims, built):
        grate, w, t, d = built
        hole = dims.get("hole", 0.08)
        offs = [(-w * 0.2, -d * 0.2), (w * 0.2, -d * 0.2), (-w * 0.2, d * 0.2), (w * 0.2, d * 0.2)]
        calls = []
        cutters = []
        for i, (ox, oz) in enumerate(offs):
            nm = f"grate_hole_{i}"
            calls.append(
                prim("box", nm, {"size": [hole, t * 3, hole]}, translation=[ox, t / 2, oz])
            )
            cutters.append(nm)
        calls.append(boolean("difference", [grate, *cutters]))
        return calls

    NAMES = [{}, {"grate": "Grate"}]
    DIMS = [{"size": (0.6, 0.03, 0.6)}, {"size": (0.7, 0.035, 0.7)}]
    PROMPTS = [
        "cut a grid of square holes through it",
        "punch a 2x2 grid of square holes through this grate",
        "Cut a grid of square holes through the metal grate.",
        "add four evenly spaced square holes through it",
        "for a drainage grate, cut a grid of square holes through it",
        "cut a grid of holes",
        "the metal grate needs a grid of square holes cut through it",
        "open a 2 by 2 grid of square holes straight through the grate",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "grate_holes",
            "medium",
            prompt,
            prior,
            calls,
            "Four box cutters at a 2x2 grid of offsets, all differenced "
            "from the grate in one clay_boolean call (a single target, so "
            "one call is correct here unlike the per-drawer dresser case).",
        )


def spec_chest_hinge() -> None:
    """Seed line: "cut a hinge notch" is achievable directly; the seed's own
    "bevel the lid's leading edge" clause is dropped for the same reason
    ``spec_chest_bevel`` drops its bevel -- no element-mode tool on the
    compact card. See that spec's docstring."""

    def build(names, dims):
        body = names.get("body", "chest_body")
        lid = names.get("lid", "chest_lid")
        w, h, d = dims["body"]
        lid_h = dims.get("lid_h", 0.08)
        calls = [
            prim("box", body, {"size": [w, h, d]}, translation=[0, h / 2, 0]),
            prim("box", lid, {"size": [w, lid_h, d]}, translation=[0, h + lid_h / 2, 0]),
        ]
        calls.append(
            mat_call(
                [body, lid], dims.get("color", (0.45, 0.3, 0.17)), roughness=0.75, name="Chest Wood"
            )
        )
        return calls, (body, lid, w, h, d, lid_h)

    def edit(names, dims, built):
        body, lid, w, h, d, lid_h = built
        cutter = "hinge_notch"
        notch_w = dims.get("notch_w", 0.1)
        calls = [
            prim(
                "box",
                cutter,
                {"size": [notch_w, lid_h * 1.5, 0.04]},
                translation=[0, h + lid_h / 2, -d / 2],
            ),
            boolean("difference", [lid, cutter]),
        ]
        return calls

    NAMES = [{}, {"body": "Chest Body", "lid": "Chest Lid"}]
    DIMS = [{"body": (0.6, 0.35, 0.35)}, {"body": (0.7, 0.4, 0.4), "lid_h": 0.09}]
    PROMPTS = [
        "cut a hinge notch into the back of the lid",
        "notch out a hinge socket at the back of the lid",
        "Cut a hinge notch into the back edge of the chest's lid.",
        "add a small notch for a hinge along the lid's back edge",
        "for a chest with exposed strap hinges, cut a hinge notch into the back of the lid",
        "cut a hinge notch",
        "the wooden chest's lid needs a hinge notch cut into its back edge",
        "open a small notch at the rear of the lid for a hinge",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "chest_hinge",
            "medium",
            prompt,
            prior,
            calls,
            "A small box cutter through the lid's own back edge, then a "
            "boolean difference against the lid alone (not the body).",
        )


def spec_tower_windows() -> None:
    def build(names, dims):
        tower = names.get("tower", "tower")
        r, h = dims["r"], dims["h"]
        calls = [
            prim(
                "cylinder",
                tower,
                {"radius": r, "height": h, "segments": 20},
                translation=[0, h / 2, 0],
            )
        ]
        calls.append(
            mat_call(
                [tower], dims.get("color", (0.55, 0.53, 0.5)), roughness=0.9, name="Tower Stone"
            )
        )
        return calls, (tower, r, h)

    def edit(names, dims, built):
        tower, r, h = built
        win_w, win_h = dims.get("win", (0.16, 0.28))
        levels = [h * 0.3, h * 0.55, h * 0.8]
        calls = []
        cutters = []
        for i, y in enumerate(levels):
            nm = f"window_{i}"
            calls.append(prim("box", nm, {"size": [win_w, win_h, r * 2.4]}, translation=[0, y, 0]))
            cutters.append(nm)
        calls.append(boolean("difference", [tower, *cutters]))
        return calls

    NAMES = [{}, {"tower": "Tower"}]
    DIMS = [{"r": 0.6, "h": 4.5}, {"r": 0.7, "h": 5.0}]
    PROMPTS = [
        "cut evenly spaced window openings up its height",
        "punch three evenly spaced windows up the height of this tower",
        "Cut evenly spaced window openings up the tower's height.",
        "add narrow window slits at even intervals climbing the tower",
        "for a watchtower, cut evenly spaced window openings up its height",
        "cut windows up the tower",
        "the stone tower needs evenly spaced window openings cut up its height",
        "open three narrow windows, evenly spaced from base to top",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "tower_windows",
            "medium",
            prompt,
            prior,
            calls,
            "Three narrow box cutters at 30/55/80 percent of the tower's "
            "height, all the way through in Z, differenced from the tower "
            "in one call.",
        )


def spec_jug_handle() -> None:
    """Seed line: "cut a handle loop through its side" -- reinterpreted as
    adding a handle (a real jug handle is a solid loop attached to the
    body, not material removed from it), via clay_boolean union rather
    than difference."""

    def build(names, dims):
        jug = names.get("jug", "jug")
        h, r = dims["h"], dims["r"]
        profile = [(r * 0.55, -h / 2), (r, -h * 0.2), (r * 0.9, h * 0.3), (r * 0.5, h / 2)]
        calls = [
            prim("lathe", jug, {"profile": profile, "segments": 18}, translation=[0, h / 2, 0])
        ]
        calls.append(
            mat_call([jug], dims.get("color", (0.72, 0.55, 0.4)), roughness=0.75, name="Ceramic")
        )
        return calls, (jug, h, r)

    def edit(names, dims, built):
        jug, h, r = built
        handle = "jug_handle"
        tube_r = dims.get("tube_r", 0.018)
        loop_r = dims.get("loop_r", r * 0.55)
        path = [
            (r * 0.85, h * 0.15, 0.0),
            (r * 0.85 + loop_r, h * 0.15 + loop_r * 0.3, 0.0),
            (r * 0.85 + loop_r, h * -0.1 - loop_r * 0.3, 0.0),
            (r * 0.85, h * -0.1, 0.0),
        ]
        base = path_centroid(path)
        translation = [base[0], base[1] + h / 2, base[2]]
        return [
            prim(
                "tube",
                handle,
                {"path": path, "radius": tube_r, "sides": 8},
                translation=translation,
            ),
            boolean("union", [jug, handle]),
        ]

    NAMES = [{}, {"jug": "Jug"}]
    DIMS = [{"h": 0.28, "r": 0.11}, {"h": 0.3, "r": 0.12}]
    PROMPTS = [
        "add a handle loop through its side",
        "attach a loop handle to the side of this jug",
        "Add a curved handle loop to the side of the ceramic jug.",
        "give it a proper loop handle on one side",
        "for pouring comfortably, add a handle loop to the side of the jug",
        "add handle loop",
        "the ceramic jug needs a handle loop attached to its side",
        "fuse a curved handle onto the side of the jug",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "jug_handle",
            "medium",
            prompt,
            prior,
            calls,
            "The seed's 'cut... through its side' does not match how a real "
            "handle is built -- a handle is solid material attached to the "
            "body, so this adds a curved tube and unions it onto the jug "
            "rather than cutting a hole. Flagged in the report.",
        )


def spec_cart_wheel_mirror() -> None:
    def build(names, dims):
        bed = names.get("bed", "cart_bed")
        wl = names.get("wl", "wheel_left")
        wr = names.get("wr", "wheel_right")
        wn = names.get("wn", "wheel_near")
        w, bh, d = dims["bed"]
        wheel_r = dims["wheel_r"]
        wx = dims["wheel_x"]
        near_z = dims["near_z"]
        calls = [
            prim("box", bed, {"size": [w, bh, d]}, translation=[0, wheel_r * 2 + bh / 2, 0]),
            prim(
                "cylinder",
                wl,
                {"radius": wheel_r, "height": 0.08, "segments": 16},
                translation=[wx, wheel_r, 0],
            ),
            prim(
                "cylinder",
                wr,
                {"radius": wheel_r, "height": 0.08, "segments": 16},
                translation=[-wx, wheel_r, 0],
            ),
            prim(
                "cylinder",
                wn,
                {"radius": wheel_r * 0.7, "height": 0.06, "segments": 14},
                translation=[0, wheel_r * 0.7, near_z],
            ),
        ]
        calls.append(
            mat_call(
                [bed, wl, wr, wn],
                dims.get("color", (0.45, 0.3, 0.17)),
                roughness=0.75,
                name="Cart Wood",
            )
        )
        return calls, (bed, wl, wr, wn)

    def edit(names, dims, built):
        bed, wl, wr, wn = built
        return [select([wn]), op("mirror-copy", {"axis": 2, "offset": 0})]

    NAMES = [{}, {"bed": "Cart Bed", "wl": "Wheel Left", "wr": "Wheel Right", "wn": "Wheel Near"}]
    DIMS = [
        {"bed": (0.9, 0.25, 0.55), "wheel_r": 0.22, "wheel_x": 0.5, "near_z": 0.3},
        {"bed": (1.0, 0.28, 0.6), "wheel_r": 0.25, "wheel_x": 0.55, "near_z": 0.32},
    ]
    PROMPTS = [
        "mirror the near wheel to add a matching far wheel",
        "mirror this front wheel to give the cart a rear one too",
        "Mirror the near wheel across to add a matching far wheel.",
        "duplicate the front wheel, reflected to the back",
        "for a four-wheeled wagon, mirror the near wheel to add a matching far one",
        "mirror near wheel to far side",
        "the wooden cart's near wheel needs a mirrored twin on the far side",
        "reflect the near wheel across the cart's centre line to add a far wheel",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "cart_wheel_mirror",
            "medium",
            prompt,
            prior,
            calls,
            "The cart already has its own left/right side-wheel pair (the "
            "'two wheels' the seed's subject names) plus a single extra "
            "front wheel; mirroring that front wheel across the world Z=0 "
            "plane (the cart's own centreline) adds the matching rear "
            "wheel, turning it into a four-wheeled wagon.",
        )


# ==========================================================================
# HARD specs
# ==========================================================================


def spec_chair_leg_taper() -> None:
    def build(names, dims):
        leg = names.get("leg", "leg")
        r, h = dims["r"], dims["h"]
        calls = [
            prim(
                "cylinder",
                leg,
                {"radius": r, "height": h, "segments": 14},
                translation=[0, h / 2, 0],
            )
        ]
        calls.append(
            mat_call([leg], dims.get("color", (0.42, 0.28, 0.16)), roughness=0.75, name="Oak")
        )
        return calls, (leg, r, h)

    def edit(names, dims, built):
        leg, r, h = built
        tapered = "leg_tapered"
        profile = [(r * 0.4, -h / 2), (r, h / 2)]
        return [
            delete([leg]),
            prim("lathe", tapered, {"profile": profile, "segments": 14}, translation=[0, h / 2, 0]),
        ]

    NAMES = [{}, {"leg": "Leg"}]
    DIMS = [{"r": 0.03, "h": 0.45}, {"r": 0.032, "h": 0.48}]
    PROMPTS = [
        "reshape it to taper smoothly from a thick top to a narrow foot",
        "taper this leg smoothly, thick at the top and narrow at the foot",
        "Reshape the chair leg to taper smoothly from top to foot.",
        "make it narrow gradually toward the floor",
        "for a Queen Anne-style chair, taper this leg smoothly from a thick top to a narrow foot",
        "taper the leg",
        "the straight chair leg needs to taper smoothly from thick at the top to narrow at the foot",
        "reshape the leg into a smooth, continuous taper from top to bottom",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "chair_leg_taper",
            "hard",
            prompt,
            prior,
            calls,
            "Deletes the straight cylinder and adds a two-station lathe "
            "profile instead -- wide at the top (y=+h/2), narrow at the "
            "foot (y=-h/2) -- which is a swept/tapered reshape, exactly "
            "the hard-class construction this seed calls for.",
        )


def spec_sword_taper() -> None:
    def build(names, dims):
        blade = names.get("blade", "blade")
        hilt = names.get("hilt", "hilt")
        blade_len = dims["blade_len"]
        blade_w = dims["blade_w"]
        hilt_len = dims["hilt_len"]
        hilt_r = dims["hilt_r"]
        calls = [
            prim(
                "cylinder",
                hilt,
                {"radius": hilt_r, "height": hilt_len, "segments": 10},
                translation=[0, hilt_len / 2, 0],
            ),
            prim(
                "box",
                blade,
                {"size": [blade_w, blade_len, blade_w * 0.2]},
                translation=[0, hilt_len + blade_len / 2, 0],
            ),
        ]
        calls.append(
            mat_call(
                [hilt],
                dims.get("hilt_color", (0.35, 0.22, 0.12)),
                roughness=0.7,
                name="Grip Leather",
            )
        )
        calls.append(
            mat_call(
                [blade],
                dims.get("blade_color", (0.75, 0.76, 0.78)),
                roughness=0.25,
                metallic=0.9,
                name="Blade Steel",
            )
        )
        return calls, (blade, hilt, blade_len, blade_w, hilt_len)

    def edit(names, dims, built):
        blade, hilt, blade_len, blade_w, hilt_len = built
        tapered = "blade_tapered"
        outline = [
            (blade_w * 0.1, -blade_w * 0.5),
            (blade_w * 0.5, 0.0),
            (blade_w * 0.1, blade_w * 0.5),
            (-blade_w * 0.1, blade_w * 0.5),
            (-blade_w * 0.5, 0.0),
            (-blade_w * 0.1, -blade_w * 0.5),
        ]
        calls = [
            delete([blade]),
            prim(
                "sweep",
                tapered,
                {
                    "outline": outline,
                    "depth": blade_len,
                    "taper": 0.03,
                    "twist": 0.0,
                    "sections": 1,
                },
                translation=[0, hilt_len + blade_len / 2, 0],
                rotation=[-90, 0, 0],
            ),
        ]
        calls.append(
            mat_call(
                [tapered],
                dims.get("blade_color", (0.75, 0.76, 0.78)),
                roughness=0.25,
                metallic=0.9,
                name="Blade Steel",
            )
        )
        return calls

    NAMES = [{}, {"blade": "Blade", "hilt": "Hilt"}]
    DIMS = [
        {"blade_len": 0.7, "blade_w": 0.05, "hilt_len": 0.14, "hilt_r": 0.015},
        {"blade_len": 0.75, "blade_w": 0.055, "hilt_len": 0.15, "hilt_r": 0.016},
    ]
    PROMPTS = [
        "taper the blade smoothly along its length to a point",
        "taper this blade down to a sharp point along its length",
        "Taper the sword's blade smoothly to a point.",
        "narrow the blade gradually until it comes to a point",
        "for a rapier-like thrusting sword, taper the blade smoothly to a point",
        "taper blade to a point",
        "the plain sword blank's blade needs to taper smoothly to a point along its length",
        "reshape the blade into a smooth continuous taper ending in a point",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "sword_taper",
            "hard",
            prompt,
            prior,
            calls,
            "Deletes the plain box blade and adds a sweep generator instead "
            "-- a diamond cross-section outline extruded the blade's own "
            "length with taper=0.03 (near a point, not exactly zero to "
            "avoid a degenerate cap), rotated -90 about X so the sweep's "
            "native +Z extrusion axis points up along the sword's length.",
        )


def spec_quadruped_tail_curve() -> None:
    """The seed's own hard entry: "a straight tail on a quadruped figure ::
    sweep it into a curved, tapering shape". Uses a fresh custom quadruped
    (not clay_add_figure -- see the module docstring) so the tail's root
    position is known exactly; the taper is approximated as two tube
    segments of different radius, unioned into one shape, since ``tube``'s
    own radius is a single number and cannot vary along its own path."""

    def build_local(names, dims):
        prior, p, body_y = _build_quadruped(names, dims)
        return prior, p, body_y

    NAMES = [
        {},
        {
            "body": "Body",
            "head": "Head",
            "tail": "Tail",
            "leg0": "Leg FL",
            "leg1": "Leg FR",
            "leg2": "Leg BL",
            "leg3": "Leg BR",
        },
    ]
    DIMS = [
        {"body": (0.5, 0.28, 0.9), "leg_h": 0.35, "half_span": (0.18, 0.32), "head_r": 0.13},
        {"body": (0.55, 0.3, 1.0), "leg_h": 0.38, "half_span": (0.2, 0.35), "head_r": 0.14},
    ]
    PROMPTS = [
        "sweep it into a curved, tapering shape",
        "reshape this straight tail into a curved, tapering one",
        "Sweep the quadruped's straight tail into a curved, tapering shape.",
        "curve the tail and taper it toward the tip",
        "for a more dynamic pose, sweep this straight tail into a curved, tapering shape",
        "curve and taper the tail",
        "the straight tail on this quadruped figure needs to become a curved, tapering shape",
        "bend the tail into a smooth curve that narrows toward its tip",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, p, body_y = build_local(names, dims)
        body_size = dims["body"]
        root_z = -(body_size[2] / 2)
        root_y = body_y + 0.02
        tail_len = dims.get("tail_len", 0.22) * 2.0
        near_path = [
            (0.0, root_y, root_z),
            (0.0, root_y - 0.02, root_z - tail_len * 0.45),
        ]
        far_path = [
            (0.0, root_y - 0.02, root_z - tail_len * 0.45),
            (0.0, root_y - 0.01, root_z - tail_len * 0.75),
            (0.0, root_y + 0.02, root_z - tail_len),
        ]
        seg_near = "q_tail_near"
        seg_far = "q_tail_far"
        calls = [
            delete([p["tail"]]),
            prim(
                "tube",
                seg_near,
                {"path": near_path, "radius": 0.035, "sides": 10},
                translation=path_centroid(near_path),
            ),
            prim(
                "tube",
                seg_far,
                {"path": far_path, "radius": 0.018, "sides": 10},
                translation=path_centroid(far_path),
            ),
            boolean("union", [seg_near, seg_far]),
        ]
        emit(
            "quadruped_tail_curve",
            "hard",
            prompt,
            prior,
            calls,
            "Deletes the straight capsule tail and rebuilds it as two tube "
            "segments sharing a curved path (root near the body, curving "
            "down and back), a thicker one near the body and a thinner one "
            "toward the tip, unioned into a single tapering, curved tail -- "
            "tube's own radius is one number per call, so true continuous "
            "taper needs this two-segment approximation rather than a "
            "single generator call.",
        )


def spec_pipe_nozzle() -> None:
    def build(names, dims):
        pipe = names.get("pipe", "pipe")
        r = dims["r"]
        length = dims["length"]
        y0 = r
        calls = [
            prim(
                "cylinder",
                pipe,
                {"radius": r, "height": length, "segments": 16},
                translation=[0, y0, 0],
                rotation=[0, 0, 90],
            )
        ]
        calls.append(
            mat_call(
                [pipe],
                dims.get("color", (0.6, 0.61, 0.63)),
                roughness=0.4,
                metallic=0.7,
                name="Steel Pipe",
            )
        )
        return calls, (pipe, r, y0, length)

    def edit(names, dims, built):
        pipe, r, y0, length = built
        curved = "nozzle_curved"
        tip = "nozzle_tip"
        path = [
            (-length / 2, y0, 0.0),
            (0.0, y0, 0.0),
            (length * 0.35, y0 + length * 0.18, 0.0),
            (length * 0.55, y0 + length * 0.32, 0.0),
        ]
        tip_pos = path[-1]
        return [
            delete([pipe]),
            prim(
                "tube",
                curved,
                {"path": path, "radius": r, "sides": 12},
                translation=path_centroid(path),
            ),
            prim(
                "cone",
                tip,
                {"radius": r, "height": r * 2.5, "segments": 12},
                translation=list(tip_pos),
                rotation=[0, 0, -80],
            ),
            boolean("union", [curved, tip]),
        ]

    NAMES = [{}, {"pipe": "Pipe"}]
    DIMS = [{"r": 0.06, "length": 1.2}, {"r": 0.065, "length": 1.3}]
    PROMPTS = [
        "bend and taper it into a curved nozzle",
        "bend this pipe into a curved, tapering nozzle",
        "Bend and taper the metal pipe into a curved nozzle shape.",
        "curve it and narrow the end into a nozzle",
        "for a fire-hose fitting, bend and taper this pipe into a curved nozzle",
        "bend and taper into nozzle",
        "the plain metal pipe needs to bend and taper into a curved nozzle",
        "reshape the straight pipe into a curved, narrowing nozzle at the end",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "pipe_nozzle",
            "hard",
            prompt,
            prior,
            calls,
            "Deletes the straight cylinder and rebuilds as a curved tube "
            "(the bend) with a cone fused onto its far end (the taper to a "
            "point), unioned into one nozzle shape -- tube alone cannot "
            "taper, so the cone supplies the narrowing tip.",
        )


def spec_spire_taper() -> None:
    def build(names, dims):
        spire = names.get("spire", "spire")
        r, h = dims["r"], dims["h"]
        calls = [
            prim(
                "cylinder",
                spire,
                {"radius": r, "height": h, "segments": 16},
                translation=[0, h / 2, 0],
            )
        ]
        calls.append(
            mat_call(
                [spire], dims.get("color", (0.5, 0.49, 0.46)), roughness=0.85, name="Spire Stone"
            )
        )
        return calls, (spire, r, h)

    def edit(names, dims, built):
        spire, r, h = built
        tapered = "spire_tapered"
        profile = [(r, -h / 2), (r * 0.75, -h * 0.1), (r * 0.35, h * 0.3), (0.0, h / 2)]
        return [
            delete([spire]),
            prim("lathe", tapered, {"profile": profile, "segments": 16}, translation=[0, h / 2, 0]),
        ]

    NAMES = [{}, {"spire": "Spire"}]
    DIMS = [{"r": 0.5, "h": 5.0}, {"r": 0.55, "h": 5.5}]
    PROMPTS = [
        "taper its profile smoothly from base to tip",
        "taper this spire smoothly, wide at the base and pointed at the tip",
        "Taper the spire's profile smoothly from base to tip.",
        "narrow it gradually until it comes to a point at the top",
        "for a cathedral spire, taper the profile smoothly from base to a sharp tip",
        "taper base to tip",
        "the stone spire needs to taper smoothly in profile from its base to its tip",
        "reshape the spire into a smooth continuous taper ending in a point",
    ]
    for i, prompt in enumerate(PROMPTS):
        names = NAMES[i % len(NAMES)]
        dims = DIMS[i % len(DIMS)]
        prior, built = build(names, dims)
        calls = edit(names, dims, built)
        emit(
            "spire_taper",
            "hard",
            prompt,
            prior,
            calls,
            "Deletes the constant-radius cylinder and adds a four-station "
            "lathe profile instead, tapering from full radius at the base "
            "to a true point (radius 0) at the tip -- a pole station per "
            "lathe's own rule, fanned rather than stacked.",
        )


# ==========================================================================


ALL_SPECS = [
    spec_table_thicken,
    spec_crate_material,
    spec_stool_raise,
    spec_archway_widen,
    spec_barrel_material,
    spec_pot_shrink,
    spec_pipe_move_up,
    spec_bookshelf_delete_shelf,
    spec_shield_material,
    spec_cart_rotate,
    spec_column_half,
    spec_humanoid_material,
    spec_crate_move,
    spec_hammer_lengthen,
    spec_bucket_material,
    spec_coffee_table_widen,
    spec_bench_material,
    spec_plinth_narrow,
    spec_quadruped_scale,
    spec_wardrobe_darken,
    spec_gear_shrink,
    spec_well_raise,
    spec_chest_delete,
    spec_side_table_material,
    spec_footstool_move,
    # medium
    spec_barrel_hole,
    spec_crates_delete_middle,
    spec_archway_window,
    spec_locker_vent,
    spec_chest_bevel,
    spec_table_leg_lathe,
    spec_shield_boss_mirror,
    spec_wall_doorway,
    spec_ladder_rung,
    spec_pipe_branch,
    spec_dresser_handles,
    spec_wagon_wheel_spokes,
    spec_column_groove,
    spec_quadruped_tail_tube,
    spec_bench_divots,
    spec_grate_holes,
    spec_chest_hinge,
    spec_tower_windows,
    spec_jug_handle,
    spec_cart_wheel_mirror,
    # hard
    spec_chair_leg_taper,
    spec_sword_taper,
    spec_quadruped_tail_curve,
    spec_pipe_nozzle,
    spec_spire_taper,
]


def main() -> None:
    for spec in ALL_SPECS:
        spec()
    with OUT.open("w", encoding="utf-8") as fh:
        for rec in _records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"wrote {len(_records)} records to {OUT}")


if __name__ == "__main__":
    main()
