"""Emits ``training/clay-assistant/drafts/mechanical.jsonl``.

Deterministic and hand-authored: fifty base builds (one per line of
``seeds/mechanical.txt``), each carrying 6 hand-written prompt phrasings, so
`50 * 6 == 300` records exactly. No randomness -- every dimension, material
and phrasing below is typed out, because a mechanical part's proportions
need to be *reasoned about* (a gear's tooth depth, a plate's hole spacing),
not sampled.

Run with:
    uv run python training/clay-assistant/drafts/_gen_mechanical.py

which (re)writes ``mechanical.jsonl`` beside this file, sorted by id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT_PATH = Path(__file__).resolve().parent / "mechanical.jsonl"

Call = dict[str, Any]


# --- tiny call builders -------------------------------------------------


def ref(name: str) -> dict[str, str]:
    return {"$ref": name}


def refs(names: list[str]) -> list[dict[str, str]]:
    return [ref(n) for n in names]


def bbox_center(path: list[list[float]]) -> list[float]:
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    zs = [p[2] for p in path]
    return [(min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, (min(zs) + max(zs)) / 2.0]


def tube_chain(
    specs: list[tuple[str, list[list[float]], float, int]], margin: float = 0.008
) -> tuple[list[Call], float]:
    """One or more ``tube`` primitives sharing a single coordinate frame.

    ``primitives._clamp_path`` re-centres a tube's ``path`` onto its own
    bounding-box centre before building it -- true placement information a
    hand-authored path assumes is *kept*, unless the caller cancels it back
    out. Passing each segment's own ``bbox_center(path)`` as its
    ``translation`` does exactly that: the tube then sits exactly where the
    raw path coordinates say it should in world space, so several segments
    whose raw paths were authored to share an end point (one continuous
    curve on paper) actually connect once built. A uniform ``lift`` is added
    to every segment's translation (never to the path itself, so the
    segments stay one continuous frame) so the whole chain's lowest point
    clears the ground by ``margin``.
    """
    min_y = min(min(p[1] for p in path) - r for (_, path, r, _) in specs)
    lift = max(0.0, margin - min_y)
    calls: list[Call] = []
    for name, path, r, sides in specs:
        cx, cy, cz = bbox_center(path)
        calls.append(
            prim(
                name,
                "tube",
                params={"path": path, "radius": r, "sides": sides},
                translation=[cx, cy + lift, cz],
            )
        )
    return calls, lift


def expand(base: str, count: int) -> list[str]:
    """Every name a ``count``-copy array (or a chain of them multiplying to
    ``count``) leaves behind: ``base``, ``base.001`` .. ``base.{count-1:03d}``
    -- see ``ops.next_name``/``ops.duplicate``, which number a fresh name's
    copies contiguously from `.001` as long as nothing else in the document
    already claims that suffix (true here: every build starts from an empty
    document)."""
    return [base] + [f"{base}.{i:03d}" for i in range(1, count)]


def prim(
    name: str,
    generator: str,
    *,
    params: dict[str, Any] | None = None,
    translation: list[float] | None = None,
    rotation: list[float] | None = None,
    scale: list[float] | None = None,
) -> Call:
    args: dict[str, Any] = {"generator": generator, "name": name}
    if params is not None:
        args["params"] = params
    if translation is not None:
        args["translation"] = translation
    if rotation is not None:
        args["rotation"] = rotation
    if scale is not None:
        args["scale"] = scale
    return {"name": "clay_add_primitive", "arguments": args}


def select(names: list[str]) -> Call:
    return {"name": "clay_select", "arguments": {"uids": refs(names)}}


def op(name: str, params: dict[str, float] | None = None) -> Call:
    args: dict[str, Any] = {"name": name}
    if params:
        args["params"] = params
    return {"name": "clay_op", "arguments": args}


def boolean(kind: str, names: list[str]) -> Call:
    return {"name": "clay_boolean", "arguments": {"kind": kind, "uids": refs(names)}}


def material(
    names: list[str],
    color: list[float],
    *,
    mname: str | None = None,
    metallic: float | None = None,
    roughness: float | None = None,
) -> Call:
    args: dict[str, Any] = {"uids": refs(names), "color": color}
    if mname is not None:
        args["name"] = mname
    if metallic is not None:
        args["metallic"] = metallic
    if roughness is not None:
        args["roughness"] = roughness
    return {"name": "clay_material", "arguments": args}


# --- material palette (cycled across phrasings for flavour variety) ----
# (color, metallic, roughness, name) -- steel/iron/chrome/brass read as four
# distinct "settings" (modern galvanised, medieval-forge, sci-fi polished,
# and a warm brass finish for valve/instrument parts).

STEEL = ([0.55, 0.56, 0.58], 0.85, 0.45, "Steel")
IRON = ([0.20, 0.19, 0.18], 0.7, 0.75, "Cast Iron")
CHROME = ([0.75, 0.78, 0.80], 0.95, 0.12, "Chrome")
BRASS = ([0.55, 0.42, 0.18], 0.9, 0.35, "Brass")
DARK_STEEL = ([0.28, 0.29, 0.31], 0.8, 0.5, "Dark Steel")
ALUM = ([0.72, 0.73, 0.74], 0.6, 0.35, "Aluminium")
_PALETTE = [STEEL, IRON, CHROME, BRASS, DARK_STEEL, ALUM]


def mat_for(build_index: int, phrasing_index: int) -> tuple[list[float], float, float, str]:
    """A deterministic, varied material pick -- cycles through ``_PALETTE``
    offset by the build so two builds in a row do not always share a
    material, and by the phrasing so the same build's six records are not
    all painted identically."""
    return _PALETTE[(build_index * 2 + phrasing_index) % len(_PALETTE)]


# --- the 50 base builds --------------------------------------------------
# Each entry: (key, notes, calls_fn(mi, pi) -> list[Call], prompts[6])
# calls_fn takes the build index and phrasing index only to vary the
# trailing clay_material call's colour -- geometry is identical across a
# build's six phrasings unless a phrasing states different dimensions, in
# which case a dedicated calls_fn variant is written instead (see e.g.
# B02, B11 below).

BUILDS: list[dict[str, Any]] = []


def add(key: str, notes: str, calls_fn, prompts: list[str]) -> None:
    assert len(prompts) in (3, 4, 5, 6), (key, len(prompts))
    BUILDS.append({"key": key, "notes": notes, "calls_fn": calls_fn, "prompts": prompts})


# ---- EASY (primitives, placement, materials only) -----------------------


# B01 -- plain metal pipe running along the ceiling
def _b01(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim(
            "pipe",
            "cylinder",
            params={"radius": 0.04, "height": 0.9, "segments": 16},
            translation=[0, 0.04, 0],
            rotation=[0, 0, 90],
        ),
        material(["pipe"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "pipe-ceiling",
    "A ceiling pipe run is one long thin cylinder; per the family convention "
    "it is grounded (rotated flat, resting on y=0) rather than left floating, "
    "even though the prompt reads as ceiling-mounted -- the training scene "
    "always sits on the ground plane. radius 4 cm keeps it thin, length 0.9 m "
    "stays in the small-parts band.",
    _b01,
    [
        "a plain metal pipe running along the ceiling",
        "just a straight pipe, the kind that runs along a ceiling",
        "make me a plain cylindrical pipe, about 90 cm long and 8 cm across, like a ceiling pipe run",
        "a thin metal pipe, unadorned, lying along a ceiling line",
        "picture a factory ceiling pipe: plain, round, metal, running straight",
        "a simple straight length of metal pipe",
    ],
)


# B02 -- a single round wheel, plain and flat
def _b02(mi, pi, radius=0.3, height=0.06):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim(
            "wheel",
            "cylinder",
            params={"radius": radius, "height": height, "segments": 24},
            translation=[0, height / 2, 0],
        ),
        material(["wheel"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "wheel-flat",
    "A flat wheel is just a squat cylinder standing on its flat face -- no "
    "teeth, no hub, per the prompt's own 'plain and flat'. 60 cm diameter, "
    "6 cm thick reads as a solid metal wheel rather than a tyre.",
    _b02,
    [
        "a single round wheel, plain and flat",
        "just a flat round wheel, no spokes, no hub",
        "a plain metal wheel, 60 cm across and 6 cm thick",
        "give me one simple flat wheel disc",
        "a round, flat wheel -- nothing fancy, just the disc",
        "a squat metal disc shaped like a wheel, unadorned",
    ],
)


# B03 -- a simple metal storage locker, a rectangular box with a door
def _b03(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    c2, met2, rgh2, mn2 = mat_for(mi + 1, pi)
    return [
        prim("locker_body", "box", params={"size": [0.6, 0.9, 0.4]}, translation=[0, 0.45, 0]),
        prim(
            "locker_door", "box", params={"size": [0.55, 0.85, 0.02]}, translation=[0, 0.45, 0.21]
        ),
        material(["locker_body"], c, mname=mn, metallic=met, roughness=rgh),
        material(["locker_door"], c2, mname=mn2, metallic=met2, roughness=rgh2),
    ]


add(
    "locker",
    "Locker body is a 60x90x40 cm box; the door is a thin box set flush "
    "against the front face (half its own thickness proud of the body's "
    "face at z=0.2) rather than boolean-cut, since this is primitives-only "
    "(easy tier). Door painted a shade darker so it reads against the body.",
    _b03,
    [
        "a simple metal storage locker, a rectangular box with a door",
        "a plain rectangular locker, metal, with a front door panel",
        "a storage locker: a 60 by 90 by 40 centimetre steel box with a door on the front",
        "just a boxy metal locker with a door",
        "an unadorned steel locker, rectangular, door included",
        "a tall metal storage box, like a school locker, with a door",
    ],
)


# B04 -- an L-shaped shelf bracket with no holes cut into it
def _b04(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    outline = [
        [0.15, -0.15],
        [0.15, -0.03],
        [-0.03, -0.03],
        [-0.03, 0.15],
        [-0.15, 0.15],
        [-0.15, -0.15],
    ]
    return [
        prim(
            "bracket",
            "sweep",
            params={"outline": outline, "depth": 0.06, "taper": 1.0, "twist": 0.0, "sections": 1},
            translation=[0.15, 0.15, 0],
            rotation=[0, 0, 0],
        ),
        material(["bracket"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "l-bracket",
    "An L-shaped cross-section (the same concave shape sweep's own default "
    "outline uses) extruded 6 cm deep gives a true right-angle bracket with "
    "no boolean at all -- deliberately no holes, so this is not the held-out "
    "'flat plate with two bolt holes and a right-angle gusset' subject: it "
    "carries no plate, no holes, and no third gusset web. Arm length 30 cm, "
    "12 cm thick. Translated so the outline's own lowest corner sits at y=0.",
    _b04,
    [
        "an L-shaped shelf bracket with no holes cut into it",
        "a plain L-bracket, solid, no holes anywhere",
        "an angled metal shelf bracket shaped like an L, 30 cm on each arm",
        "just an L-shaped support bracket, no perforations",
        "a right-angle bracket, solid metal, unpunched",
        "a bracket bent into an L, thick and solid, holeless",
    ],
)


# B05 -- a plain steel I-beam section
def _b05(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    length, width, height, t = 0.5, 0.2, 0.3, 0.02
    return [
        prim(
            "web",
            "box",
            params={"size": [t, height - 2 * t, length]},
            translation=[0, height / 2, 0],
        ),
        prim(
            "flange_bottom", "box", params={"size": [width, t, length]}, translation=[0, t / 2, 0]
        ),
        prim(
            "flange_top",
            "box",
            params={"size": [width, t, length]},
            translation=[0, height - t / 2, 0],
        ),
        boolean("union", ["web", "flange_bottom", "flange_top"]),
        material(["web"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "i-beam",
    "Classic I cross-section: a thin web plus two flanges, unioned into one "
    "solid (web created first so it is the union survivor). 30 cm tall, "
    "20 cm flange width, 50 cm section length, 2 cm wall thickness -- proportions "
    "read as a small structural steel section rather than a full girder.",
    _b05,
    [
        "a plain steel I-beam section",
        "a short length of I-beam, steel, unadorned",
        "an I-beam section: 30 cm tall, 50 cm long, standard flanges",
        "just a plain I-shaped steel beam segment",
        "a steel structural beam, I cross-section, one solid piece",
        "picture a stub of I-beam, the kind used in framing, plain steel",
    ],
)


# B06 -- a simple hand crank, a straight rod with a perpendicular handle
def _b06(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim(
            "crank_arm",
            "cylinder",
            params={"radius": 0.012, "height": 0.2, "segments": 12},
            translation=[0, 0.012, 0],
            rotation=[0, 0, 90],
        ),
        prim(
            "crank_handle",
            "cylinder",
            params={"radius": 0.012, "height": 0.1, "segments": 12},
            translation=[0.1, 0.05, 0],
        ),
        material(["crank_arm", "crank_handle"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "hand-crank",
    "A crank is two cylinders: the arm lies flat along X resting on the "
    "ground (radius 1.2 cm, 20 cm long), and the handle stands perpendicular "
    "to it at the far end -- the grip a hand would turn. Same material "
    "throughout since it reads as one machined part.",
    _b06,
    [
        "a simple hand crank, a straight rod with a perpendicular handle",
        "a hand-crank: one straight arm, one handle sticking up at the end",
        "a crank handle, 20 cm arm, with a perpendicular grip",
        "just a plain hand crank -- rod plus perpendicular handle",
        "picture a manual crank: straight metal arm, handle at right angles",
        "a simple winding crank, metal, straight rod and cross handle",
    ],
)


# B07 -- a metal toolbox, plain rectangular box
def _b07(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim("toolbox", "box", params={"size": [0.45, 0.2, 0.22]}, translation=[0, 0.1, 0]),
        material(["toolbox"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "toolbox",
    "One plain box, 45x20x22 cm -- a handheld metal toolbox's rough "
    "proportions, nothing more since the prompt asks for exactly that.",
    _b07,
    [
        "a metal toolbox, a plain rectangular box",
        "just a plain rectangular metal toolbox",
        "a toolbox, 45 by 20 by 22 centimetres, unadorned",
        "a simple box-shaped metal toolbox",
        "picture a plain steel toolbox, no handle, no latch, just the box",
        "a rectangular metal case, toolbox-sized",
    ],
)


# B08 -- a round manhole cover, flat and circular
def _b08(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim(
            "cover",
            "cylinder",
            params={"radius": 0.35, "height": 0.03, "segments": 32},
            translation=[0, 0.015, 0],
        ),
        material(["cover"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "manhole-cover",
    "A wide, thin cylinder: 70 cm across, 3 cm thick, matching a real "
    "manhole cover's proportions -- flat and circular, exactly as asked, no "
    "pattern cut into the top since that would need a boolean this tier "
    "does not call for.",
    _b08,
    [
        "a round manhole cover, flat and circular",
        "a plain circular manhole cover, flat",
        "a manhole cover, 70 cm across, flat disc",
        "just a round flat cover, like a manhole lid",
        "picture a heavy flat disc, the kind that caps a manhole",
        "a circular street cover, thin and flat",
    ],
)


# B09 -- a plain metal ladder rung
def _b09(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim(
            "rung",
            "cylinder",
            params={"radius": 0.012, "height": 0.35, "segments": 12},
            translation=[0, 0.012, 0],
            rotation=[0, 0, 90],
        ),
        material(["rung"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "ladder-rung",
    "A single thin horizontal cylinder, 35 cm long, 2.4 cm across -- a "
    "ladder rung on its own, grounded per convention even though a rung "
    "normally sits between two rails.",
    _b09,
    [
        "a plain metal ladder rung",
        "just one rung from a metal ladder",
        "a ladder rung, 35 cm long, plain round bar",
        "a simple round bar, ladder-rung sized",
        "picture a single metal ladder step, cylindrical",
        "a plain rung, the kind a ladder is built from",
    ],
)


# B10 -- a cylindrical oil drum standing upright
def _b10(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim(
            "drum",
            "cylinder",
            params={"radius": 0.29, "height": 0.88, "segments": 24},
            translation=[0, 0.44, 0],
        ),
        material(["drum"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "oil-drum",
    "A tall cylinder, 58 cm diameter and 88 cm tall -- a standard oil drum's "
    "rough proportions, standing upright as asked.",
    _b10,
    [
        "a cylindrical oil drum standing upright",
        "a plain oil drum, standing up",
        "an oil drum, about 58 cm across and 88 cm tall, upright",
        "just a standing metal drum",
        "picture a tall cylindrical drum, the kind oil comes in",
        "a steel drum, upright, plain cylinder",
    ],
)


# B11 -- a flat metal plate with rounded corners
def _b11(mi, pi, size=(0.4, 0.02, 0.3)):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim("plate", "box", params={"size": list(size)}, translation=[0, size[1] / 2, 0]),
        material(["plate"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "plate-rounded",
    "A thin flat box, 40x30 cm, 2 cm thick. 'Rounded corners' is approximated "
    "as a plain rectangular plate here: rounding a corner needs a fillet/bevel "
    "on selected edges (an element-mode op), which is past this easy tier's "
    "primitives-and-placement-only budget -- noted rather than silently "
    "dropped.",
    _b11,
    [
        "a flat metal plate with rounded corners",
        "a plain metal slab, flat, corners softened",
        "a metal plate, 40 by 30 centimetres, thin, rounded at the corners",
        "just a flat rectangular plate",
        "picture a thin steel plate with its corners eased",
        "a flat rectangular slab, edges softened at the corners",
    ],
)


# B12 -- a simple axle, a plain cylindrical rod
def _b12(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim(
            "axle",
            "cylinder",
            params={"radius": 0.02, "height": 0.5, "segments": 12},
            translation=[0, 0.02, 0],
            rotation=[0, 0, 90],
        ),
        material(["axle"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "axle",
    "A plain thin cylinder, 4 cm diameter, 50 cm long, lying flat -- an "
    "axle rod with nothing else attached, exactly as the prompt asks.",
    _b12,
    [
        "a simple axle, a plain cylindrical rod",
        "just a plain metal rod, axle-shaped",
        "an axle, 50 cm long, 4 cm across",
        "a straight round axle rod",
        "picture a bare axle shaft, cylindrical, unadorned",
        "a simple rod that could be an axle",
    ],
)


# B13 -- a metal support strut, a straight bar
def _b13(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim("strut", "box", params={"size": [0.04, 0.04, 0.7]}, translation=[0, 0.02, 0]),
        material(["strut"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "strut",
    "A single square-section box bar, 4x4 cm, 70 cm long, lying on the "
    "ground -- a support strut with nothing else on it.",
    _b13,
    [
        "a metal support strut, a straight bar",
        "just a plain straight support strut",
        "a support strut, square bar, 70 cm long",
        "a straight metal bar for bracing",
        "picture a simple structural strut, square in section",
        "a plain bar used as a brace",
    ],
)


# B14 -- a plain valve wheel, a flat disc with a central hub
def _b14(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim(
            "disc",
            "cylinder",
            params={"radius": 0.12, "height": 0.02, "segments": 24},
            translation=[0, 0.35, 0],
        ),
        prim(
            "hub",
            "cylinder",
            params={"radius": 0.025, "height": 0.06, "segments": 12},
            translation=[0, 0.38, 0],
        ),
        material(["disc", "hub"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "valve-wheel-plain",
    "A thin disc plus a small protruding hub cylinder on the same axis -- "
    "no spokes cut through it, per 'plain'. Raised to a plausible valve "
    "height (35 cm) rather than sitting on the floor, since a valve wheel is "
    "mounted on a stem in practice; still within the ground-floor rule since "
    "nothing sinks below y=0.",
    _b14,
    [
        "a plain valve wheel, a flat disc with a central hub",
        "just a flat valve wheel with a hub in the middle",
        "a valve wheel, 24 cm across, disc with a raised centre hub",
        "a plain round valve handle, disc and hub",
        "picture a simple valve wheel, no spokes, just disc and hub",
        "a flat metal valve wheel with a central boss",
    ],
)


# B15 -- a rectangular electrical junction box
def _b15(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim("junction_box", "box", params={"size": [0.2, 0.2, 0.12]}, translation=[0, 0.1, 0]),
        material(["junction_box"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "junction-box",
    "A squat box, 20x20x12 cm -- an electrical junction box's rough size, "
    "plain, no lid seam modelled since that needs no extra geometry to read "
    "correctly at this scale.",
    _b15,
    [
        "a rectangular electrical junction box",
        "just a plain junction box",
        "an electrical junction box, 20 by 20 by 12 centimetres",
        "a small metal box for wiring",
        "picture a squat rectangular electrical box",
        "a plain metal junction box, boxy and small",
    ],
)


# B16 -- a plain metal hinge, two flat plates joined at an edge
def _b16(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    leaf = [0.1, 0.012, 0.12]
    return [
        prim("leaf_a", "box", params={"size": leaf}, translation=[-0.05, leaf[1] / 2, 0]),
        prim("leaf_b", "box", params={"size": leaf}, translation=[0.05, leaf[1] / 2, 0]),
        material(["leaf_a", "leaf_b"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "hinge-plain",
    "Two thin flat boxes lying edge to edge (touching along the centre "
    "line, x=0) -- 'joined at an edge' read literally as two leaves meeting "
    "rather than a knuckle/pin, since no boolean or pin cylinder is called "
    "for at this easy tier.",
    _b16,
    [
        "a plain metal hinge, two flat plates joined at an edge",
        "just a simple hinge, two leaves meeting at an edge",
        "a hinge made of two flat metal plates, 10 by 12 cm each",
        "a plain flat hinge, two leaves",
        "picture a basic hinge: two thin plates joined edge to edge",
        "a simple two-leaf hinge, flat metal",
    ],
)


# B17 -- a simple pulley wheel, round and grooved
def _b17(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim(
            "flange_a",
            "cylinder",
            params={"radius": 0.09, "height": 0.015, "segments": 24},
            translation=[0, 0.09, 0],
        ),
        prim(
            "groove",
            "cylinder",
            params={"radius": 0.065, "height": 0.03, "segments": 24},
            translation=[0, 0.09, 0],
        ),
        prim(
            "flange_b",
            "cylinder",
            params={"radius": 0.09, "height": 0.015, "segments": 24},
            translation=[0, 0.09, 0],
        ),
        material(["flange_a", "groove", "flange_b"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "pulley",
    "Three stacked cylinders sharing one centre -- two wide flat flanges "
    "either side of a narrower spool -- approximates a grooved pulley "
    "without a boolean cut: the 'groove' reads as the narrow waist between "
    "the two wider rims.",
    _b17,
    [
        "a simple pulley wheel, round and grooved",
        "just a plain pulley, round with a groove",
        "a pulley wheel, 18 cm across, with a grooved rim",
        "a round grooved pulley",
        "picture a small pulley wheel, the kind a rope runs over",
        "a plain metal pulley, spool-shaped",
    ],
)


# B18 -- a metal conduit pipe bent at a right angle
def _b18(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    path = [[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0]]
    calls, _lift = tube_chain([("conduit", path, 0.025, 10)])
    calls.append(material(["conduit"], c, mname=mn, metallic=met, roughness=rgh))
    return calls


add(
    "conduit-elbow",
    "A single ``tube`` generator call with a three-point right-angle path "
    "(one straight run, a corner, another straight run) -- still one "
    "primitive, no boolean, so it stays easy tier. The path is authored "
    "pre-centred on its own bounding box (``_clamp_path`` would otherwise "
    "re-centre it silently) and ``tube_chain`` adds a uniform lift so the "
    "bend's lowest point clears the ground.",
    _b18,
    [
        "a metal conduit pipe bent at a right angle",
        "just a conduit pipe with a 90 degree bend",
        "a conduit, bent at a right angle, 2.5 cm radius",
        "a pipe that turns a sharp corner",
        "picture electrical conduit bent square, metal",
        "a plain conduit pipe with one right-angle bend",
    ],
)


# B19 -- a plain steel girder, a long rectangular beam
def _b19(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim("girder", "box", params={"size": [0.18, 0.28, 1.0]}, translation=[0, 0.14, 0]),
        material(["girder"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "girder",
    "A single long box, 18x28 cm cross-section, 1 m long (the top of this "
    "family's size band) -- a plain rectangular girder with no I-flange "
    "detail, since the prompt only asks for 'a long rectangular beam'.",
    _b19,
    [
        "a plain steel girder, a long rectangular beam",
        "just a long rectangular steel beam",
        "a girder, one metre long, rectangular section",
        "a plain long steel beam",
        "picture a heavy rectangular girder",
        "a straight steel girder, boxy cross-section",
    ],
)


# B20 -- a simple wrench-shaped hand tool, flat and elongated
def _b20(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim("handle", "box", params={"size": [0.03, 0.01, 0.18]}, translation=[0, 0.005, 0]),
        prim("head", "box", params={"size": [0.05, 0.012, 0.06]}, translation=[0, 0.006, 0.11]),
        material(["handle", "head"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "wrench-flat",
    "A thin elongated handle box plus a wider head box at one end -- flat "
    "and elongated per the prompt, no jaw cut-out since that would need a "
    "boolean and the seed marks this easy.",
    _b20,
    [
        "a simple wrench-shaped hand tool, flat and elongated",
        "just a flat wrench shape",
        "a wrench-shaped tool, elongated, about 24 cm long",
        "a flat elongated hand tool, wrench-like",
        "picture a simple flat wrench outline",
        "a plain wrench shape, thin and long",
    ],
)


# B21 -- a metal control panel, a flat plate with a raised lip
def _b21(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim("panel", "box", params={"size": [0.4, 0.3, 0.02]}, translation=[0, 0.15, 0]),
        prim("lip", "box", params={"size": [0.42, 0.32, 0.01]}, translation=[0, 0.15, -0.015]),
        material(["panel", "lip"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "control-panel",
    "A flat panel plate plus a slightly larger, thinner box set just behind "
    "it as the 'raised lip' framing its edge -- placement only, no boolean.",
    _b21,
    [
        "a metal control panel, a flat panel with a raised lip",
        "just a flat control panel with a lip around it",
        "a control panel, 40 by 30 cm, with a raised edge",
        "a plain instrument panel plate with a lip",
        "picture a metal panel with a rim standing proud",
        "a flat panel plate, lipped at the edge",
    ],
)


# B22 -- a round metal cap for a pipe end
def _b22(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim(
            "cap",
            "cylinder",
            params={"radius": 0.05, "height": 0.03, "segments": 20},
            translation=[0, 0.015, 0],
        ),
        material(["cap"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "pipe-cap",
    "A short squat cylinder -- a pipe end cap, 10 cm across, 3 cm deep.",
    _b22,
    [
        "a round metal cap for a pipe end",
        "just a plain pipe end cap",
        "a pipe cap, round, 10 cm across",
        "a cap that fits over a pipe's end",
        "picture a small round cap for capping a pipe",
        "a plain metal end cap, circular",
    ],
)


# B23 -- a plain hex-shaped bolt head, a short six-sided prism
def _b23(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim(
            "bolt_head",
            "cylinder",
            params={"radius": 0.03, "height": 0.02, "segments": 6},
            translation=[0, 0.01, 0],
        ),
        material(["bolt_head"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "bolt-head",
    "A six-sided prism is just ``cylinder`` with ``segments=6`` -- no boolean "
    "needed, one primitive, a short hex head 6 cm across the flats.",
    _b23,
    [
        "a plain hex-shaped bolt head, a short six-sided prism",
        "just a hex bolt head",
        "a bolt head, six-sided, short and squat",
        "a hexagonal prism, bolt-head sized",
        "picture a plain hex nut-shaped head",
        "a six-sided bolt head, plain metal",
    ],
)


# B24 -- a simple metal footrest, a flat plate on two short legs
def _b24(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim("footrest_top", "box", params={"size": [0.3, 0.02, 0.2]}, translation=[0, 0.12, 0]),
        prim("leg_a", "box", params={"size": [0.03, 0.1, 0.03]}, translation=[-0.12, 0.05, 0]),
        prim("leg_b", "box", params={"size": [0.03, 0.1, 0.03]}, translation=[0.12, 0.05, 0]),
        material(["footrest_top", "leg_a", "leg_b"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "footrest",
    "A flat top plate on two short box legs -- straightforward placement, "
    "top plate 30x20 cm, legs 10 cm tall.",
    _b24,
    [
        "a simple metal footrest, a flat platform on two stubby legs",
        "just a flat footrest on two legs",
        "a footrest, two stubby legs under a 30 by 20 cm platform",
        "a plain metal footrest",
        "picture a low footrest, flat top, two stubby legs",
        "a small metal footrest plate on legs",
    ],
)


# B25 -- a small metal cable spool, a plain squat cylinder
def _b25(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim(
            "spool",
            "cylinder",
            params={"radius": 0.15, "height": 0.1, "segments": 20},
            translation=[0, 0.05, 0],
        ),
        material(["spool"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "cable-spool",
    "One squat cylinder, 30 cm across, 10 cm tall -- plain per the prompt.",
    _b25,
    [
        "a small metal cable spool, a plain squat cylinder",
        "just a plain squat cable spool",
        "a cable spool, 30 cm across, short and squat",
        "a small plain spool for cable",
        "picture a squat metal cable reel, unadorned",
        "a plain metal spool, short cylinder",
    ],
)


# ---- MEDIUM (boolean / array / lathe-sweep / element op) ---------------


# B26 -- a gear disc with teeth cut evenly around its rim
def _gear(name_body, name_tooth, r, h, t, tooth_h_extra, n_teeth, mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    calls = [
        prim(
            name_body,
            "cylinder",
            params={"radius": r, "height": h, "segments": 32},
            translation=[0, h / 2, 0],
        ),
        prim(
            name_tooth,
            "box",
            params={"size": [t, h + tooth_h_extra, 0.02]},
            translation=[r + t / 2, h / 2, 0],
        ),
        select([name_tooth]),
        op("array-radial", {"count": n_teeth, "angle": 360, "axis": 1}),
        boolean("union", [name_body] + expand(name_tooth, n_teeth)),
        material([name_body], c, mname=mn, metallic=met, roughness=rgh),
    ]
    return calls


def _b26(mi, pi):
    return _gear("gear_body", "tooth", 0.15, 0.03, 0.025, 0.0, 20, mi, pi)


add(
    "gear-disc",
    "The family's signature pattern: a cylinder body (created first, so it "
    "survives the union), one tooth box placed just outside the rim, "
    "``clay_select`` + ``clay_op array-radial`` (count 20, axis 1/Y) to ring "
    "it round, then ``clay_boolean union`` of the body plus every array copy "
    "(named ``tooth``, ``tooth.001`` .. ``tooth.019`` -- ``ops.duplicate``'s "
    "own contiguous numbering, predictable because nothing else in the "
    "document claims those names first). 30 cm gear, 3 cm thick, 20 teeth.",
    _b26,
    [
        "a gear disc with teeth cut evenly around its rim",
        "just a plain gear wheel with teeth around the edge",
        "a gear, 30 cm across, 20 teeth evenly spaced around the rim",
        "a toothed gear disc",
        "picture a flat gear with teeth ringing its rim",
        "a metal gear wheel, teeth arranged around the outside",
    ],
)


# B27 -- a flat mounting plate with a row of five holes cut through it
def _b27(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    plate_t = 0.015
    hole_positions = [-0.16, -0.08, 0.0, 0.08, 0.16]
    calls = [
        prim(
            "plate", "box", params={"size": [0.45, plate_t, 0.12]}, translation=[0, plate_t / 2, 0]
        ),
    ]
    hole_names = []
    for i, x in enumerate(hole_positions):
        n = f"hole_{i + 1}"
        hole_names.append(n)
        calls.append(
            prim(
                n,
                "cylinder",
                params={"radius": 0.012, "height": plate_t * 3, "segments": 16},
                translation=[x, plate_t / 2, 0],
            )
        )
    calls.append(boolean("difference", ["plate"] + hole_names))
    calls.append(material(["plate"], c, mname=mn, metallic=met, roughness=rgh))
    return calls


add(
    "plate-five-holes",
    "The plate is created first (union/difference survivor), then five "
    "separate cylinder cutters spaced 8 cm apart along its length -- named "
    "individually (``hole_1``..``hole_5``) rather than arrayed, since five is "
    "few enough to place by hand and it keeps the boolean's own uid list "
    "readable. Cutter height is 3x the plate thickness so it fully punches "
    "through both faces. Not the held-out subject: no gusset, five holes not "
    "two.",
    _b27,
    [
        "a flat mounting plate with a row of five holes cut through it",
        "just a plate with five holes in a row",
        "a mounting plate, 45 by 12 cm, with five holes punched through it",
        "a plate, five holes across it, flat and plain",
        "picture a mounting plate with a row of bolt-sized holes",
        "a plain plate, perforated with five holes in a line",
    ],
)


# B28 -- a pipe fitting with a threaded socket cut into one end
def _b28(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    r, h = 0.05, 0.14
    return [
        prim(
            "fitting",
            "cylinder",
            params={"radius": r, "height": h, "segments": 20},
            translation=[0, h / 2, 0],
        ),
        prim(
            "socket",
            "cylinder",
            params={"radius": r * 0.6, "height": h * 0.5, "segments": 20},
            translation=[0, h - h * 0.25, 0],
        ),
        boolean("difference", ["fitting", "socket"]),
        material(["fitting"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "pipe-fitting-socket",
    "A cylinder body first, then a narrower, shorter cylinder positioned at "
    "the top end and cut away with a boolean difference to leave a socket "
    "recess -- reads as a threaded socket without modelling actual thread "
    "geometry (a topology op this tool surface has no thread generator for).",
    _b28,
    [
        "a pipe fitting with a threaded socket cut into one end",
        "just a pipe fitting with a socket in one end",
        "a pipe fitting, 10 cm across, with a socket recess cut into the top",
        "a fitting with a socket bored into it",
        "picture a pipe fitting with a threaded-looking recess at one end",
        "a plain fitting with a socket cut into its end",
    ],
)


# B29 -- a wheel rim with a radial pattern of eight spokes
def _b29(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    hub_r, rim_tube, rim_center = 0.05, 0.045, 0.26
    rim_inner = rim_center - rim_tube
    spoke_len = rim_inner - hub_r
    return [
        prim(
            "hub",
            "cylinder",
            params={"radius": hub_r, "height": 0.03, "segments": 16},
            translation=[0, 0.25, 0],
        ),
        prim(
            "rim",
            "torus",
            params={"radius": rim_center, "tube": rim_tube, "segments": 32, "sides": 12},
            translation=[0, 0.25, 0],
        ),
        prim(
            "spoke",
            "box",
            params={"size": [0.03, 0.025, spoke_len]},
            translation=[0, 0.25, hub_r + spoke_len / 2],
        ),
        select(["spoke"]),
        op("array-radial", {"count": 8, "angle": 360, "axis": 1}),
        material(["hub", "rim"] + expand("spoke", 8), c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "wheel-rim-spokes",
    "Hub, outer rim and one spoke box, all at the same elevation (0.25 m, "
    "so the whole assembly floats as a mounted wheel would -- verify only "
    "rejects sinking below ground, not floating above it). The rim is a "
    "``torus`` (a true ring) rather than a solid disc: an early draft used a "
    "solid cylinder for the rim, which -- being the same radius all the way "
    "to its own centre -- covered the hub and every spoke completely from "
    "outside, so nothing but a plain disc ever rendered. The spoke, sized to "
    "reach exactly from the hub's own edge to the torus's inner edge, is "
    "selected and ``array-radial``'d 8 ways about the world Y axis, which "
    "passes through the hub's own centre; no boolean is needed since the "
    "parts are meant to read as separate spokes, hub and rim -- the same "
    "shape ``tests/fixtures/agent_transcripts/spoked-hub.jsonl`` uses.",
    _b29,
    [
        "a wheel rim with a radial pattern of eight spokes",
        "just a wheel rim with eight spokes",
        "a wheel: rim, hub, and eight spokes radiating out",
        "a spoked wheel rim, eight spokes",
        "picture a wheel with a hub and eight spokes to the rim",
        "a metal wheel rim, eight radial spokes",
    ],
)


# B30 -- a metal grate with a grid of square holes cut through it
def _b30(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    plate_t = 0.01
    cx, cz = 5, 4
    xstep, zstep = 0.07, 0.07
    calls = [
        prim(
            "grate",
            "box",
            params={"size": [cx * xstep + 0.05, plate_t, cz * zstep + 0.05]},
            translation=[0, plate_t / 2, 0],
        ),
        prim(
            "hole",
            "box",
            params={"size": [0.04, plate_t * 3, 0.04]},
            translation=[-((cx - 1) * xstep) / 2, plate_t / 2, -((cz - 1) * zstep) / 2],
        ),
        select(["hole"]),
        op("array-linear", {"count": cx, "x": xstep, "y": 0, "z": 0}),
        op("array-linear", {"count": cz, "x": 0, "y": 0, "z": zstep}),
        boolean("difference", ["grate"] + expand("hole", cx * cz)),
        material(["grate"], c, mname=mn, metallic=met, roughness=rgh),
    ]
    return calls


add(
    "grate",
    "One plate, one square cutter, and two ``array-linear`` calls in a row "
    "(along X, then along Z) -- the second array copies the whole selection "
    "left by the first (which ``doc.select`` already carries forward), so "
    "5x4=20 holes come from three calls total. The full cutter name set is "
    "``hole``..``hole.019`` (contiguous, per ``ops.duplicate``'s numbering), "
    "then one ``clay_boolean difference`` against the plate.",
    _b30,
    [
        "a metal grate with a grid of square holes cut through it",
        "just a grate, a grid of square holes",
        "a metal grate, 5 by 4 grid of square holes cut through",
        "a perforated grate plate",
        "picture a floor grate with a grid of square openings",
        "a plain metal grate, square holes in a grid",
    ],
)


# B31 -- a pipe elbow joint bent at ninety degrees
def _b31(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    path = [[-0.125, -0.125, 0.0], [0.125, -0.125, 0.0], [0.125, 0.125, 0.0]]
    calls, _lift = tube_chain([("elbow", path, 0.05, 12)])
    calls.append(material(["elbow"], c, mname=mn, metallic=met, roughness=rgh))
    return calls


add(
    "pipe-elbow",
    "A wider ``tube`` elbow than the easy-tier conduit bend (radius 5 cm vs "
    "2.5 cm, more sides for a smoother pipe), matching this seed's own "
    "medium classification (a swept-path primitive) even though the call "
    "pattern is the same shape as ``conduit-elbow`` at a different scale. "
    "Same pre-centred-path-plus-lift construction via ``tube_chain``.",
    _b31,
    [
        "a pipe elbow joint bent at ninety degrees",
        "just a pipe elbow at ninety degrees",
        "a pipe elbow, 10 cm diameter, bent 90 degrees",
        "a right-angle pipe elbow joint",
        "picture a pipe fitting that turns a sharp corner",
        "a plain 90 degree pipe elbow",
    ],
)


# B32 -- a valve wheel with a radial pattern of spokes cut through the disc
def _b32(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    r, h = 0.14, 0.025
    n = 6
    return [
        prim(
            "valve_disc",
            "cylinder",
            params={"radius": r, "height": h, "segments": 28},
            translation=[0, 0.3, 0],
        ),
        prim(
            "hub",
            "cylinder",
            params={"radius": 0.025, "height": h * 1.4, "segments": 12},
            translation=[0, 0.3, 0],
        ),
        prim(
            "spoke_cut",
            "box",
            params={"size": [r * 2 * 0.55, h * 3, 0.02]},
            translation=[0, 0.3, 0],
        ),
        select(["spoke_cut"]),
        op("array-radial", {"count": n, "angle": 360, "axis": 1}),
        boolean("difference", ["valve_disc"] + expand("spoke_cut", n)),
        material(["valve_disc", "hub"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "valve-wheel-spoked",
    "Disc created first (difference survivor), a hub added for grip, then "
    "one radial slot cutter (a box through the disc's thickness, long "
    "enough to reach past the centre) arrayed 6 ways about Y and subtracted "
    "in one ``clay_boolean difference`` call -- 'spokes cut through the "
    "disc' read as literal cut-throughs rather than solid spokes, which is "
    "what distinguishes this from B29's solid-spoke wheel rim.",
    _b32,
    [
        "a valve wheel with a radial pattern of spokes cut through the disc",
        "just a valve wheel with spokes cut through it",
        "a valve wheel, 28 cm across, six spokes cut through the disc",
        "a spoked valve wheel, cut-through pattern",
        "picture a valve handle with open spoke slots",
        "a round valve wheel with radial slots cut into it",
    ],
)


# B33 -- a gear wheel with evenly spaced teeth arranged radially
def _b33(mi, pi):
    return _gear("gear2_body", "tooth2", 0.2, 0.04, 0.03, 0.0, 24, mi, pi)


add(
    "gear-wheel",
    "Same union-of-array pattern as B26 (``gear-disc``) but larger (40 cm) "
    "and with 24 teeth, kept as a distinct base build since the seed list "
    "treats 'gear disc' and 'gear wheel' as two separate lines -- the "
    "family's boolean-star pattern is worth more than one worked example.",
    _b33,
    [
        "a gear wheel with evenly spaced teeth arranged radially",
        "just a gear wheel, teeth evenly spaced around it",
        "a gear wheel, 40 cm across, 24 teeth arranged radially",
        "an evenly-toothed gear wheel",
        "picture a large gear with teeth spaced evenly around the rim",
        "a metal gear wheel, radial even teeth",
    ],
)


# B34 -- a hinge with a row of pin cut-outs arranged along its length
def _b34(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    r, length = 0.02, 0.3
    n = 6
    step = length / n
    return [
        prim(
            "barrel",
            "cylinder",
            params={"radius": r, "height": length, "segments": 16},
            translation=[0, r, 0],
            rotation=[0, 0, 90],
        ),
        prim(
            "pin_cut",
            "cylinder",
            params={"radius": r * 1.6, "height": r * 0.5, "segments": 16},
            translation=[-length / 2 + step / 2, r, 0],
            rotation=[90, 0, 0],
        ),
        select(["pin_cut"]),
        op("array-linear", {"count": n, "x": step, "y": 0, "z": 0}),
        boolean("difference", ["barrel"] + expand("pin_cut", n)),
        material(["barrel"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "hinge-barrel-pins",
    "A hinge barrel (cylinder lying along X) with a row of short, wide "
    "cutter discs punched through it lengthwise -- one cutter, "
    "``array-linear`` along the barrel's own axis, then a single "
    "``clay_boolean difference`` against the barrel. 'Pin cut-outs' read as "
    "the machined relief a hinge pin's own knuckles need.",
    _b34,
    [
        "a hinge with a row of pin cut-outs arranged along its length",
        "just a hinge barrel with pin cut-outs in a row",
        "a hinge, 30 cm long, with six pin cut-outs along it",
        "a hinge barrel with cut-outs down its length",
        "picture a long hinge pin with notches cut along it",
        "a metal hinge barrel, pin cut-outs in a row",
    ],
)


# B35 -- a manifold with several pipe stubs arranged in a row
def _b35(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    n = 5
    step = 0.09
    return [
        prim("body", "box", params={"size": [0.5, 0.1, 0.1]}, translation=[0, 0.05, 0]),
        prim(
            "stub",
            "cylinder",
            params={"radius": 0.02, "height": 0.06, "segments": 14},
            translation=[-2 * step, 0.05, 0.08],
            rotation=[90, 0, 0],
        ),
        select(["stub"]),
        op("array-linear", {"count": n, "x": step, "y": 0, "z": 0}),
        boolean("union", ["body"] + expand("stub", n)),
        material(["body"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "manifold",
    "A box manifold body created first, then a single pipe-stub cylinder "
    "(rotated to point outward along Z), arrayed into a row of five along "
    "the body's length and unioned onto the body so the stubs read as part "
    "of one cast piece rather than five loose cylinders resting against it.",
    _b35,
    [
        "a manifold with several pipe stubs arranged in a row",
        "just a manifold, a row of pipe stubs",
        "a manifold block, 50 cm long, five pipe stubs sticking out",
        "a pipe manifold with stubs in a row",
        "picture a manifold with several outlet stubs along its side",
        "a plain manifold body with a row of stub outlets",
    ],
)


# B36 -- a sprocket wheel with teeth arranged radially around its rim
def _b36(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    r, h = 0.12, 0.02
    n = 16
    return [
        prim(
            "sprocket_body",
            "cylinder",
            params={"radius": r, "height": h, "segments": 32},
            translation=[0, h / 2, 0],
        ),
        prim(
            "tooth3",
            "pyramid",
            params={"base": 0.02, "height": 0.02, "sides": 4},
            translation=[r + 0.01, h / 2, 0],
            rotation=[0, 0, -90],
        ),
        select(["tooth3"]),
        op("array-radial", {"count": n, "angle": 360, "axis": 1}),
        boolean("union", ["sprocket_body"] + expand("tooth3", n)),
        material(["sprocket_body"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "sprocket",
    "Same union-of-array pattern as the gears but with pyramid teeth (a "
    "pointed chain-sprocket tooth reads better as a small pyramid than a "
    "box) rotated -90 about Z so its point aims outward before the radial "
    "array carries that orientation around with it. 16 teeth, 24 cm across.",
    _b36,
    [
        "a sprocket wheel with teeth arranged radially around its rim",
        "just a sprocket, pointed teeth around the rim",
        "a sprocket wheel, 24 cm across, 16 pointed teeth",
        "a chain sprocket, teeth radiating around it",
        "picture a sprocket wheel with sharp teeth around the edge",
        "a metal sprocket, radial pointed teeth",
    ],
)


# B37 -- a triangular support bracket with a cut-out in the middle
def _b37(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    # Pre-centred on its own bounding-box centre (a right triangle, legs
    # 0.25 x 0.2) so ``primitives._clamp_outline``'s own re-centring is a
    # no-op -- an earlier draft used [[0,0],[0.25,0],[0,0.2]] uncentred, so
    # the cut-out (placed assuming those raw coordinates) ended up outside
    # the re-centred solid and cut nothing visible.
    outline = [[-0.125, -0.1], [0.125, -0.1], [-0.125, 0.1]]
    centroid_x = (-0.125 + 0.125 - 0.125) / 3.0
    centroid_y = (-0.1 - 0.1 + 0.1) / 3.0
    return [
        prim(
            "bracket3",
            "sweep",
            params={"outline": outline, "depth": 0.04, "taper": 1.0, "twist": 0.0, "sections": 1},
            translation=[0, 0.1, 0],
        ),
        prim(
            "cutout",
            "cylinder",
            params={"radius": 0.035, "height": 0.06, "segments": 20},
            translation=[centroid_x, 0.1 + centroid_y, 0],
            rotation=[90, 0, 0],
        ),
        boolean("difference", ["bracket3", "cutout"]),
        material(["bracket3"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "triangular-bracket",
    "A right-triangle ``sweep`` outline (a genuinely different shape from "
    "the held-out flat-plate-plus-gusset subject: this is one solid "
    "triangular web, not a plate with a separate perpendicular gusset "
    "piece) standing as a flat vertical plaque (no rotation needed -- the "
    "sweep's own Z depth already reads as the bracket's thickness facing "
    "the viewer) with a round cut-out bored through at the triangle's own "
    "centroid via a single ``clay_boolean difference``, the cutout rotated "
    "90 about X so its axis points along that same Z depth.",
    _b37,
    [
        "a triangular support bracket with a cut-out in the middle",
        "just a triangular bracket with a hole in the middle",
        "a triangular support bracket, 25 cm arm, hole cut through the centre",
        "a solid triangle-shaped bracket with a cut-out",
        "picture a triangular gusset-like bracket with a lightening hole",
        "a plain triangular bracket, one cut-out",
    ],
)


# B38 -- a wheel hub with a radial pattern of lug holes
def _b38(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    r, h = 0.09, 0.05
    n = 5
    return [
        prim(
            "hub2",
            "cylinder",
            params={"radius": r, "height": h, "segments": 24},
            translation=[0, h / 2, 0],
        ),
        prim(
            "lug",
            "cylinder",
            params={"radius": 0.008, "height": h * 3, "segments": 12},
            translation=[r * 0.6, h / 2, 0],
        ),
        select(["lug"]),
        op("array-radial", {"count": n, "angle": 360, "axis": 1}),
        boolean("difference", ["hub2"] + expand("lug", n)),
        material(["hub2"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "wheel-hub-lugs",
    "A hub cylinder with a ring of five narrow lug-bolt holes bored through "
    "it at 60% of its radius, arrayed about Y and subtracted in one "
    "difference -- the hub is created first so it is the survivor.",
    _b38,
    [
        "a wheel hub with a radial pattern of lug holes",
        "just a wheel hub with lug holes",
        "a wheel hub, 18 cm across, five lug holes around it",
        "a hub with holes for lug bolts, arranged in a circle",
        "picture a car-wheel-style hub with lug holes",
        "a plain metal hub, lug holes arranged radially",
    ],
)


# B39 -- a cable reel with a series of radial spokes
def _b39(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    hub_r, flange_r, flange_t, width = 0.05, 0.22, 0.015, 0.2
    return [
        prim(
            "flange_bottom",
            "cylinder",
            params={"radius": flange_r, "height": flange_t, "segments": 28},
            translation=[0, flange_t / 2, 0],
        ),
        prim(
            "flange_top",
            "cylinder",
            params={"radius": flange_r, "height": flange_t, "segments": 28},
            translation=[0, width - flange_t / 2, 0],
        ),
        prim(
            "reel_hub",
            "cylinder",
            params={"radius": hub_r, "height": width, "segments": 16},
            translation=[0, width / 2, 0],
        ),
        prim(
            "reel_spoke",
            "box",
            params={"size": [flange_r - hub_r, 0.03, 0.03]},
            translation=[hub_r + (flange_r - hub_r) / 2, width / 2, 0],
        ),
        select(["reel_spoke"]),
        op("array-radial", {"count": 6, "angle": 360, "axis": 1}),
        material(
            ["flange_bottom", "flange_top", "reel_hub"] + expand("reel_spoke", 6),
            c,
            mname=mn,
            metallic=met,
            roughness=rgh,
        ),
    ]


add(
    "cable-reel",
    "Redesigned to lie flat (two horizontal flange discs, a vertical hub "
    "between them, spokes radiating horizontally at mid-height) after an "
    "earlier standing-up version put the spokes' elevation off the world "
    "Y-axis and let ``array-radial``'s axis=2 (Z) rotation swing them below "
    "the ground on the far side of the circle -- ``array-radial`` always "
    "spins about the axis *through the world origin*, so anything offset "
    "from that origin in the rotation plane sweeps a full circle through "
    "it. Lying flat keeps every spoke's elevation on the invariant Y axis "
    "(axis=1), the same safe pattern every other array in this family uses.",
    _b39,
    [
        "a cable reel with a series of radial spokes",
        "just a cable reel with spokes",
        "a cable reel, 50 cm across, six radial spokes between the flanges",
        "a reel with spokes connecting two flanges",
        "picture a large cable reel with spokes visible between its ends",
        "a plain cable reel, spoked construction",
    ],
)


# B40 -- a fan blade assembly with blades arranged radially around a hub
def _b40(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim(
            "fan_hub",
            "cylinder",
            params={"radius": 0.03, "height": 0.05, "segments": 16},
            translation=[0, 0.4, 0],
        ),
        prim(
            "blade",
            "box",
            params={"size": [0.02, 0.16, 0.05]},
            translation=[0.11, 0.4, 0],
            rotation=[0, 0, 30],
        ),
        select(["blade"]),
        op("array-radial", {"count": 5, "angle": 360, "axis": 1}),
        material(["fan_hub"] + expand("blade", 5), c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "fan-blade-assembly",
    "A hub plus one blade box, pitched 30 degrees about Z before the "
    "radial array runs -- because the array carries each copy's own "
    "rotation around with it (spin-then-original-orientation), every copy "
    "keeps that same 30 degree pitch as it is carried round the hub, which "
    "is what makes it read as a fan rather than flat paddles.",
    _b40,
    [
        "a fan blade assembly with blades arranged radially around a hub",
        "just a fan hub with blades around it",
        "a fan assembly, hub with five pitched blades arranged radially",
        "a fan, blades radiating from a central hub",
        "picture a small fan with blades set at an angle around the hub",
        "a plain fan hub and blade assembly",
    ],
)


# B41 -- a pipe rack holding a row of five identical pipes
def _b41(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    c2, met2, rgh2, mn2 = mat_for(mi + 1, pi)
    n = 5
    step = 0.12
    return [
        prim("rack_frame", "box", params={"size": [0.65, 0.1, 0.1]}, translation=[0, 0.05, 0]),
        prim(
            "pipe2",
            "cylinder",
            params={"radius": 0.03, "height": 0.6, "segments": 16},
            translation=[-2 * step, 0.15, 0],
            rotation=[0, 0, 90],
        ),
        select(["pipe2"]),
        op("array-linear", {"count": n, "x": 0, "y": 0, "z": step}),
        material(["rack_frame"], c, mname=mn, metallic=met, roughness=rgh),
        material(expand("pipe2", n), c2, mname=mn2, metallic=met2, roughness=rgh2),
    ]


add(
    "pipe-rack",
    "A rack frame box plus one horizontal pipe, arrayed sideways (along Z) "
    "into a row of five identical pipes resting above it -- two materials, "
    "one for the frame and one for the pipes, since a real rack and its "
    "pipes are rarely the same finish.",
    _b41,
    [
        "a pipe rack holding a row of five identical pipes",
        "just a rack with five identical pipes on it",
        "a pipe rack, five identical pipes in a row",
        "a rack holding several matching pipes",
        "picture a storage rack with five pipes laid across it",
        "a plain pipe rack, five pipes side by side",
    ],
)


# B42 -- a metal grille with a repeating grid of horizontal bars
def _b42(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    n = 6
    step = 0.05
    return [
        prim(
            "frame",
            "box",
            params={"size": [0.4, n * step + 0.05, 0.02]},
            translation=[0, (n * step + 0.05) / 2, 0],
        ),
        prim("bar", "box", params={"size": [0.38, 0.02, 0.025]}, translation=[0, 0.05, 0.02]),
        select(["bar"]),
        op("array-linear", {"count": n, "x": 0, "y": step, "z": 0}),
        material(["frame"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "grille-bars",
    "A frame box behind a single horizontal bar, arrayed upward (along Y) "
    "into six evenly spaced bars proud of the frame's face -- bars sit in "
    "front of the frame rather than being cut through it, so no boolean is "
    "needed for this one.",
    _b42,
    [
        "a metal grille with a repeating grid of horizontal bars",
        "just a grille, horizontal bars repeated",
        "a metal grille, six horizontal bars evenly spaced",
        "a grille made of repeating horizontal bars",
        "picture a vent grille with horizontal slats",
        "a plain metal grille, horizontal bar pattern",
    ],
)


# B43 -- a valve assembly with a cylindrical body and a boolean-cut clearance notch for the handle
def _b43(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    r, h = 0.06, 0.18
    return [
        prim(
            "valve_body",
            "cylinder",
            params={"radius": r, "height": h, "segments": 20},
            translation=[0, h / 2, 0],
        ),
        prim(
            "notch", "box", params={"size": [r * 1.2, 0.03, r * 2.2]}, translation=[0, h - 0.015, 0]
        ),
        boolean("difference", ["valve_body", "notch"]),
        material(["valve_body"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "valve-assembly-notch",
    "The prompt names the operation directly ('boolean-cut'): a cylindrical "
    "body first, then a box notch positioned at the top and subtracted, "
    "leaving a clearance slot a valve handle could swing through.",
    _b43,
    [
        "a valve assembly with a cylindrical body and a boolean-cut clearance notch for the handle",
        "just a valve body with a notch cut for the handle",
        "a valve assembly, cylindrical, with a clearance notch cut at the top",
        "a valve body with a handle clearance slot cut in",
        "picture a valve with a notch boolean-cut for its lever",
        "a plain valve cylinder with a cut clearance notch",
    ],
)


# B44 -- a chain link, a flattened torus shape
def _b44(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim(
            "link",
            "torus",
            params={"radius": 0.06, "tube": 0.012, "segments": 24, "sides": 12},
            translation=[0, 0.012, 0],
            scale=[1.6, 1.0, 0.85],
        ),
        material(["link"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "chain-link",
    "One torus, non-uniformly scaled (1.6x wider on X, 0.85x on Z) so the "
    "ring reads as an elongated, flattened chain link rather than a round "
    "one -- the scale multiplier the generator's own params cannot reach "
    "(``clay_add_primitive``'s ``scale`` argument, not a ``clay_set_params`` "
    "rebuild).",
    _b44,
    [
        "a chain link, a flattened torus shape",
        "just one flattened chain link",
        "a chain link, elongated and flattened",
        "a single stretched, flattened ring, chain-link shaped",
        "picture one link from a heavy chain, oval and flat",
        "a plain metal chain link, flattened oval",
    ],
)


# B45 -- a turbine wheel with blades arranged in a radial array
def _b45(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    return [
        prim(
            "turbine_hub",
            "cylinder",
            params={"radius": 0.05, "height": 0.06, "segments": 20},
            translation=[0, 0.3, 0],
        ),
        prim(
            "shroud",
            "torus",
            params={"radius": 0.22, "tube": 0.015, "segments": 28, "sides": 10},
            translation=[0, 0.3, 0],
        ),
        prim(
            "turbine_blade",
            "box",
            params={"size": [0.018, 0.2, 0.04]},
            translation=[0.15, 0.3, 0],
            rotation=[0, 0, 20],
        ),
        select(["turbine_blade"]),
        op("array-radial", {"count": 12, "angle": 360, "axis": 1}),
        material(
            ["turbine_hub", "shroud"] + expand("turbine_blade", 12),
            c,
            mname=mn,
            metallic=met,
            roughness=rgh,
        ),
    ]


add(
    "turbine-wheel",
    "Hub, an outer shroud ring (a thin torus at blade-tip radius), and one "
    "pitched blade radially arrayed 12 ways -- the shroud gives the turbine "
    "silhouette the plainer fan assembly (B40) does not need.",
    _b45,
    [
        "a turbine wheel with blades arranged in a radial array",
        "just a turbine wheel, blades in a radial array",
        "a turbine wheel, 44 cm across, 12 pitched blades and a shroud ring",
        "a turbine rotor, blades radiating around a hub",
        "picture a small turbine wheel with a shroud and radial blades",
        "a plain turbine wheel, radial blade array",
    ],
)


# ---- HARD (swept or tapered form along a path) --------------------------


# B46 -- a flexible metal hose that tapers and curves from a wide fitting to a narrow nozzle
def _b46(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    seg_a = [[0.0, 0.0, 0.0], [0.08, 0.03, 0.0], [0.16, 0.03, 0.0]]
    seg_b = [[0.16, 0.03, 0.0], [0.24, 0.02, 0.0], [0.32, -0.01, 0.0]]
    seg_c = [[0.32, -0.01, 0.0], [0.38, -0.02, 0.0], [0.44, -0.02, 0.0]]
    calls, _lift = tube_chain(
        [
            ("hose_wide", seg_a, 0.035, 12),
            ("hose_mid", seg_b, 0.022, 12),
            ("hose_nozzle", seg_c, 0.012, 12),
        ]
    )
    calls.append(boolean("union", ["hose_wide", "hose_mid", "hose_nozzle"]))
    calls.append(material(["hose_wide"], c, mname=mn, metallic=met, roughness=rgh))
    return calls


add(
    "flexible-hose",
    "Three ``tube`` segments sharing one bent, gently curving centreline "
    "(each segment's raw path continues from the previous one's end point, "
    "and ``tube_chain`` places each at its own path's bbox-centre so that "
    "shared coordinate frame survives ``_clamp_path``'s own re-centring and "
    "the pieces actually connect) with radius stepping down 0.035 -> 0.022 "
    "-> 0.012 -- the family's approximation for a variable-thickness sweep, "
    "which the real ``tube`` generator does not support (its ``radius`` is "
    "one scalar, by design -- see ``primitives.tube``'s own docstring). "
    "Widest segment created first so it is the union survivor.",
    _b46,
    [
        "a flexible metal hose that tapers and curves from a wide fitting to a narrow nozzle",
        "just a hose that tapers down to a narrow nozzle, with a curve in it",
        "a hose, 44 cm long, tapering from a wide fitting to a narrow nozzle, gently curved",
        "a tapering, curving metal hose",
        "picture a flexible hose narrowing toward its nozzle end, with a bend",
        "a plain metal hose, tapered and curved",
    ],
)


# B47 -- an exhaust pipe that curves and tapers along a swept path
def _b47(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    seg_a = [[0.0, 0.0, 0.0], [0.1, 0.05, 0.0], [0.2, 0.08, 0.02]]
    seg_b = [[0.2, 0.08, 0.02], [0.3, 0.09, 0.05], [0.4, 0.08, 0.09]]
    calls, _lift = tube_chain(
        [
            ("exhaust_a", seg_a, 0.05, 14),
            ("exhaust_b", seg_b, 0.03, 14),
        ]
    )
    calls.append(boolean("union", ["exhaust_a", "exhaust_b"]))
    calls.append(material(["exhaust_a"], c, mname=mn, metallic=met, roughness=rgh))
    return calls


add(
    "exhaust-pipe",
    "Two tube segments along a path that curves in both Y and Z, radius "
    "stepping 0.05 -> 0.03, unioned into one piece -- same two-segment "
    "taper-along-a-path approximation as the hose (via ``tube_chain``), at "
    "exhaust-pipe scale.",
    _b47,
    [
        "an exhaust pipe that curves and tapers along a swept path",
        "just an exhaust pipe, curved and tapering",
        "an exhaust pipe, 40 cm long, curving and tapering along its path",
        "a curving, tapering exhaust pipe",
        "picture an exhaust pipe that bends and narrows along its length",
        "a plain exhaust pipe, swept and tapered",
    ],
)


# B48 -- a robotic arm segment that tapers smoothly along a curved sweep
def _b48(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    seg_a = [[0.0, 0.0, 0.0], [0.09, 0.02, 0.0], [0.18, 0.02, 0.0]]
    seg_b = [[0.18, 0.02, 0.0], [0.27, 0.01, 0.0], [0.36, -0.01, 0.0]]
    calls, _lift = tube_chain(
        [
            ("arm_a", seg_a, 0.04, 16),
            ("arm_b", seg_b, 0.026, 16),
        ]
    )
    calls.append(boolean("union", ["arm_a", "arm_b"]))
    calls.append(material(["arm_a"], c, mname=mn, metallic=met, roughness=rgh))
    return calls


add(
    "robotic-arm-segment",
    "A gentle two-segment taper along a shallow S-curve, more sides (16) "
    "than the hose or exhaust pipe for a smoother-reading surface, since "
    "the prompt specifically asks for 'tapers smoothly'. Same "
    "``tube_chain`` construction.",
    _b48,
    [
        "a robotic arm segment that tapers smoothly along a curved sweep",
        "just a robot arm segment, smoothly tapered along a curve",
        "a robotic arm segment, 36 cm long, tapering smoothly along a gentle curve",
        "a smoothly tapering robotic arm piece",
        "picture a robot arm link that narrows smoothly along its curved length",
        "a plain robotic arm segment, curved and tapered",
    ],
)


# B49 -- a drill bit that tapers to a sharp twisting point
def _b49(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    outline = [[0.012, 0.012], [-0.012, 0.012], [-0.012, -0.012], [0.012, -0.012]]
    # The shank's length axis is local Y (a cylinder's own convention); the
    # sweep's is local Z (its own convention -- see primitives.sweep). The
    # same rotation cannot carry both onto one world axis: an earlier draft
    # rotated both [90, 0, 0] and got a shank sitting sideways off the
    # point's flank instead of beneath its wide end. Left un-rotated, the
    # shank's length already reads as world Y; rotating only the sweep by
    # [-90, 0, 0] maps *its* local Z (length) onto world Y too (y' = z under
    # that rotation), and keeps the sweep's near (-Z, full-width) end at the
    # *bottom* of its own span -- exactly where it must sit to meet the
    # shank's top.
    shank_h = 0.06
    point_depth = 0.12
    overlap = 0.005
    return [
        prim(
            "shank",
            "cylinder",
            params={"radius": 0.012, "height": shank_h, "segments": 16},
            translation=[0, shank_h / 2, 0],
        ),
        prim(
            "bit_point",
            "sweep",
            params={
                "outline": outline,
                "depth": point_depth,
                "taper": 0.06,
                "twist": 200,
                "sections": 6,
            },
            translation=[0, shank_h - overlap + point_depth / 2, 0],
            rotation=[-90, 0, 0],
        ),
        boolean("union", ["shank", "bit_point"]),
        material(["shank"], c, mname=mn, metallic=met, roughness=rgh),
    ]


add(
    "drill-bit",
    "A cylindrical shank (left un-rotated, its length native to world Y) "
    "plus a square-outline ``sweep`` for the fluted point, rotated -90 "
    "about X so *its* length axis (local Z, a sweep's own convention -- "
    "different from a cylinder's local Y) also reads as world Y, near "
    "(full-width) end down against the shank, far (tapered) end up. "
    "``taper`` shrinks the far end to 6% of the near end (a sharp point) "
    "and ``twist`` turns it 200 degrees over 6 sections, which reads as the "
    "flute's own spiral. Unioned onto the shank with a 5 mm overlap.",
    _b49,
    [
        "a drill bit that tapers to a sharp twisting point",
        "just a drill bit with a twisting, tapered point",
        "a drill bit, 18 cm long, tapering to a sharp twisted point",
        "a twisted, tapering drill bit",
        "picture a drill bit whose flutes twist toward a sharp tip",
        "a plain drill bit, twisting taper to a point",
    ],
)


# B50 -- a wrench whose handle sweeps in a curved taper toward the jaw
def _b50(mi, pi):
    c, met, rgh, mn = mat_for(mi, pi)
    seg_a = [[0.0, 0.0, 0.0], [0.06, 0.015, 0.0], [0.12, 0.02, 0.0]]
    seg_b = [[0.12, 0.02, 0.0], [0.16, 0.02, 0.0], [0.2, 0.015, 0.0]]
    calls, lift = tube_chain(
        [
            ("handle_a", seg_a, 0.016, 10),
            ("handle_b", seg_b, 0.024, 10),
        ]
    )
    end_x, end_y, end_z = seg_b[-1]
    calls.append(
        prim(
            "jaw_a",
            "box",
            params={"size": [0.02, 0.03, 0.05]},
            translation=[end_x, end_y + lift, end_z + 0.02],
        )
    )
    calls.append(
        prim(
            "jaw_b",
            "box",
            params={"size": [0.02, 0.03, 0.05]},
            translation=[end_x, end_y + lift, end_z - 0.02],
        )
    )
    calls.append(boolean("union", ["handle_a", "handle_b", "jaw_a", "jaw_b"]))
    calls.append(material(["handle_a"], c, mname=mn, metallic=met, roughness=rgh))
    return calls


add(
    "wrench-swept",
    "Two tube segments, radius growing from grip (1.6 cm) to jaw end (2.4 "
    "cm) along a gentle curve (via ``tube_chain``, so the two segments "
    "actually meet), unioned with two prong boxes positioned at the second "
    "segment's own raw end point (plus the same lift) to form the open jaw "
    "-- 'sweeps in a curved taper toward the jaw' read as the handle "
    "widening as it curves toward the head, not narrowing.",
    _b50,
    [
        "a wrench whose handle sweeps in a curved taper toward the jaw",
        "just a wrench with a curved, tapering handle toward the jaw",
        "a wrench, 20 cm long, handle curving and widening toward the jaw",
        "a curved, tapered wrench handle leading to an open jaw",
        "picture a wrench whose handle sweeps and widens toward its jaw",
        "a plain wrench, swept curved handle, open jaw",
    ],
)


# --- assembly -------------------------------------------------------------


def main() -> None:
    assert len(BUILDS) == 50, f"expected 50 base builds, got {len(BUILDS)}"
    records: list[dict[str, Any]] = []
    n = 0
    for bi, build in enumerate(BUILDS):
        prompts = build["prompts"]
        for pi, prompt in enumerate(prompts):
            n += 1
            calls = build["calls_fn"](bi, pi)
            records.append(
                {
                    "id": f"mechanical-{n:04d}",
                    "family": "mechanical",
                    "kind": "build",
                    "prompt": prompt,
                    "calls": calls,
                    "notes": build["notes"],
                }
            )
    records.sort(key=lambda r: r["id"])
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"wrote {len(records)} records ({len(BUILDS)} base builds) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
