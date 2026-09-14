"""Deterministic emitter for ``drafts/figures.jsonl`` -- the family built to
teach ``clay_add_figure`` part names, not just the tool call itself.

Run A (Q8_0) refused 15 of 232 eval replies with ``no object named '...'``;
nine of those were creatures guessing at a figure's part names (``hound_Beak``,
``t_Shank.R``, ``s_Tail 01``) rather than using the real ones. A separate
change adds every figure's part names to the tool card; this family teaches
the model to actually use them once they are there.

Run directly to (re)write ``figures.jsonl`` beside this file::

    uv run python training/clay-assistant/drafts/_gen_figures.py

~100 ``kind: build`` records (one figure, sometimes two, placed and then
reshaped/recoloured/part-swapped) and ~50 ``kind: edit`` records (a small
prior scene, then an edit addressing a real part name from that scene's own
turn). Every one of the eight ``presets.ASSEMBLIES`` keys is used, and every
part of every key is addressed by a real ``{"$ref": "<prefix><part>"}`` at
least three times across the whole family -- :func:`_check_coverage` proves
this by walking every emitted record's own calls rather than trusting the
recipe list by eye, and ``main`` refuses to write the file if it does not
hold.

``PART_NAMES`` is imported from ``_gen_creatures`` rather than copied: that
table is drift-tested there against the live ``presets.py``, and a second
hand-typed copy here would just be one more place for a renamed part to go
unnoticed.

Geometry notes worth keeping next to the code:

* A ``Part`` from ``presets.py`` carries a generator name and a params dict
  like any other object, so ``clay_set_params`` reaches a figure's own named
  part exactly the way it reaches a hand-built primitive (see
  ``_gen_creatures.py``'s own ``bird-oversized-beak``/``biped-curling-tail``
  recipes, reused verbatim below for the beak resize and the tail lengthen).
* **The absolute-scale trap.** ``clay_transform``'s own ``scale`` argument
  *replaces* an object's scale outright; it does not multiply it. Every part
  built by ``_capsule``/``_box``/``_sphere``/``_ico`` defaults to scale
  ``(1, 1, 1)``, so a "30% bigger" edit is just ``scale=[1.3, 1.3, 1.3]``
  there. A part built by ``_mass`` (a humanoid/quadruped/bird's
  Hips/Spine/Chest, a serpent/fish's spine masses, ``biped_tail``'s own
  Tail 01) already carries a non-unit ``(width, along, depth)`` scale from
  the preset itself -- replacing it with a small-looking absolute number
  (say ``[1.2, 1.15, 1.2]`` on a Chest whose own preset scale is already
  ``(1.35, 1.45, 1.35)``) would *shrink* it. Every such edit below computes
  its new absolute scale from the real preset numbers in ``presets.py``
  rather than guessing a plausible-looking triple.
* Figure part world positions used for a delete-and-replace (a helmet on a
  Head, a hammer in a Hand, a spike on a Crown) are hand-derived from the
  bone landmarks in ``presets.py`` through the same ``(x, y, z) -> (x, z,
  -y)`` axis swap :func:`presets._to_clay` applies, then the bone
  midpoint -- not read live (the recipe must not import ``warlock``). The
  whole-assembly grounding shift ``presets._grounded`` applies afterward is
  small (the 2026-09-06 audit measured six figures sinking between 0.0009
  and 0.13 before that fix) and moves every part together, so a
  freestanding replacement primitive placed at the *un-grounded* landmark
  can sit slightly proud of or into the figure it replaces -- exactly what
  the gallery review this recipe's own build step calls for is for.
* Figures are already grounded by ``clay_add_figure`` itself (``presets.build``'s
  grounding pass); only ``serpent`` and ``fish`` are swimmers
  (``presets.SWIMMERS``) and are authored floating on purpose, so nothing
  here needs ``allow_below_ground`` -- a swimmer was never below the ground
  to begin with, and a terrestrial figure's translation is never sent
  negative.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _gen_creatures import PART_NAMES  # noqa: E402

OUT_PATH = Path(__file__).with_name("figures.jsonl")

# --- coverage bookkeeping ----------------------------------------------------
# Populated by every call to figure(); checked by _check_coverage() once the
# whole record list is built, so a recipe that forgets to reference a part
# cannot silently ship under-covered.
PREFIX_KEY: dict[str, str] = {}


# --- tiny call-builders (mirrors _gen_creatures.py / _gen_edits.py) ----------


def ref(name: str) -> dict[str, str]:
    return {"$ref": name}


def refs(key: str, prefix: str) -> list[dict[str, str]]:
    return [ref(f"{prefix}{n}") for n in PART_NAMES[key]]


def refs_except(key: str, prefix: str, exclude: set[str]) -> list[dict[str, str]]:
    """Like :func:`refs`, minus the part names in *exclude* -- for a recipe
    that deletes some of a preset's own parts before its ``clay_material``
    call: a $ref to a name that no longer exists is a refusal, not a no-op."""
    return [ref(f"{prefix}{n}") for n in PART_NAMES[key] if n not in exclude]


def figure(key: str, prefix: str, *, translation=None, yaw=None, scale=None) -> dict[str, Any]:
    if key not in PART_NAMES:
        raise KeyError(f"unknown figure key {key!r}")
    existing = PREFIX_KEY.get(prefix)
    if existing is not None and existing != key:
        raise AssertionError(f"name_prefix {prefix!r} used for both {existing!r} and {key!r}")
    PREFIX_KEY[prefix] = key
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


# --- material palette ---------------------------------------------------------
SKIN_TAN = (0.80, 0.63, 0.50)
STEEL_GREY = (0.55, 0.56, 0.58)
ARMOR_BLACK = (0.12, 0.12, 0.14)
GOLD = (0.70, 0.55, 0.15)
SILVER = (0.75, 0.76, 0.78)
RED_PAINT = (0.65, 0.12, 0.10)
BLUE_PAINT = (0.15, 0.35, 0.65)
WOOD_BROWN = (0.45, 0.30, 0.16)
STONE_GREY = (0.55, 0.54, 0.50)
HIDE_BROWN = (0.45, 0.32, 0.20)
HIDE_DUN = (0.62, 0.52, 0.36)
HIDE_GREYGREEN = (0.50, 0.55, 0.42)
HIDE_DARK = (0.20, 0.22, 0.16)
FEATHER_RUST = (0.55, 0.35, 0.18)
FEATHER_GREY = (0.55, 0.55, 0.58)
FEATHER_TOUCAN = (0.75, 0.55, 0.15)
SCALE_GREEN = (0.20, 0.45, 0.30)
SCALE_DARKGREEN = (0.30, 0.42, 0.22)
SCALE_DARKER = (0.12, 0.14, 0.10)
SCALE_ORANGE = (0.75, 0.45, 0.15)
CHITIN_BLACK = (0.08, 0.08, 0.09)
CHITIN_OLIVE = (0.30, 0.34, 0.18)
SLIME_TEAL = (0.20, 0.55, 0.50)
SLIME_BLUE = (0.30, 0.45, 0.65)
GREEN_ABDOMEN = (0.30, 0.55, 0.25)
LOBE_PURPLE = (0.55, 0.35, 0.60)
SPIKE_GOLD = (0.85, 0.75, 0.20)
BRUTE_HIDE = (0.40, 0.32, 0.22)


# =============================================================================
# Builds: (slug, calls, [prompt phrasings])
# =============================================================================
BuildRecipe = tuple[str, list[dict[str, Any]], list[str]]
build_recipes: list[BuildRecipe] = []


def add_build(slug: str, calls: list[dict[str, Any]], prompts: list[str]) -> None:
    assert 2 <= len(prompts) <= 6, f"{slug}: {len(prompts)} phrasings"
    build_recipes.append((slug, calls, prompts))


# ---- A. baseline: one unmodified figure, one body-wide material (8 keys) ----

add_build(
    "humanoid-baseline",
    [
        figure("humanoid", "knight_"),
        material(refs("humanoid", "knight_"), STEEL_GREY, "Armour", roughness=0.5, metallic=0.3),
    ],
    [
        "a humanoid figure in plain steel-grey armour, standing at ease",
        "armoured humanoid, nothing fancy",
        "Place a humanoid figure about 1.8 metres tall wearing plain steel-grey armour.",
        "a lone armoured figure keeping watch on a castle wall",
    ],
)

add_build(
    "biped-tail-baseline",
    [
        figure("biped_tail", "imp_"),
        material(refs("biped_tail", "imp_"), HIDE_GREYGREEN, "Hide", roughness=0.6),
    ],
    [
        "a biped creature with a tail, plain grey-green hide",
        "tailed biped, basic build",
        "A biped figure with a tapering tail, roughly 1.5 metres tall, dull green-grey hide.",
        "an imp-like biped lurking at the edge of a torchlit hall",
    ],
)

add_build(
    "quadruped-baseline",
    [
        figure("quadruped", "hound_"),
        material(refs("quadruped", "hound_"), HIDE_BROWN, "Hide", roughness=0.75),
    ],
    [
        "a basic quadruped figure, brown hide",
        "plain four-legged animal figure",
        "A quadruped figure about a metre at the shoulder, plain brown hide, standing square.",
        "a hound-like creature waiting patiently by a hunting lodge",
    ],
)

add_build(
    "bird-baseline",
    [
        figure("bird", "hawk_"),
        material(refs("bird", "hawk_"), FEATHER_RUST, "Feathers", roughness=0.6),
    ],
    [
        "a bird figure with rust-brown feathers",
        "plain bird, rust feathers",
        "A bird figure roughly 60 centimetres tall, warm rust-brown feathers, standing upright.",
        "a hawk-like bird perched at the edge of a falconer's glove",
    ],
)

add_build(
    "serpent-baseline",
    [
        figure("serpent", "adder_"),
        material(refs("serpent", "adder_"), SCALE_GREEN, "Scales", roughness=0.4),
    ],
    [
        "a serpent figure with dark green scales",
        "plain green serpent figure",
        "A serpent figure about 70 centimetres long, dark green scales, coiled loosely.",
        "an adder-like serpent resting near a warm garden wall",
    ],
)

add_build(
    "fish-baseline",
    [
        figure("fish", "trout_"),
        material(refs("fish", "trout_"), SCALE_ORANGE, "Scales", roughness=0.3),
    ],
    [
        "a fish figure with orange scales",
        "plain orange fish figure",
        "A fish figure about 40 centimetres long, bright orange scales.",
        "a koi-like fish drifting near a garden pond's edge",
    ],
)

add_build(
    "insect-baseline",
    [
        figure("insect", "wasp_"),
        material(refs("insect", "wasp_"), CHITIN_BLACK, "Chitin", roughness=0.35),
    ],
    [
        "an insect figure with glossy black chitin",
        "plain black insect figure",
        "An insect figure about 25 centimetres long, glossy black chitin, six legs planted.",
        "a wasp-like insect resting on a sunlit windowsill",
    ],
)

add_build(
    "blob-baseline",
    [figure("blob", "ooze_"), material(refs("blob", "ooze_"), SLIME_TEAL, "Ooze", roughness=0.2)],
    [
        "a blob figure in glossy teal ooze",
        "plain teal blob figure",
        "A blob figure about a metre tall, glossy teal ooze, round and simple.",
        "a teal ooze creature squelching across a cellar floor",
    ],
)

# ---- B. styled: reshape a couple of named parts, recolour an accent --------

add_build(
    "humanoid-gilded",
    [
        figure("humanoid", "paladin_"),
        material(refs("humanoid", "paladin_"), STEEL_GREY, "Armour", roughness=0.45, metallic=0.25),
        transform(ref("paladin_Shoulder.L"), scale=[1.3, 1.3, 1.3]),
        transform(ref("paladin_Shoulder.R"), scale=[1.3, 1.3, 1.3]),
        material(
            [
                ref("paladin_Hand.L"),
                ref("paladin_Hand.R"),
                ref("paladin_Foot.L"),
                ref("paladin_Foot.R"),
            ],
            GOLD,
            "Gilt Trim",
            roughness=0.3,
            metallic=0.6,
        ),
    ],
    [
        "a broad-shouldered armoured humanoid with gilded hands and boots",
        "big-shouldered armoured figure, gold trim on hands and feet",
        "An armoured humanoid figure with noticeably broadened shoulders and gilded trim on its hands and boots.",
    ],
)

add_build(
    "biped-tail-long-curl",
    [
        figure("biped_tail", "gremlin_"),
        material(refs("biped_tail", "gremlin_"), HIDE_GREYGREEN, "Hide", roughness=0.6),
        set_params(ref("gremlin_Tail 04"), {"height": 0.16}),
        set_params(ref("gremlin_Tail 05"), {"height": 0.14}),
        # A lone absolute rotation on Tail 05 alone (clay_transform's own
        # rotation replaces a part's bone-aligned quaternion outright, it
        # does not compose with it) kinked the tail at the Tail 04/05 joint
        # in the gallery review -- _gen_creatures.py's own
        # "biped-curling-tail" rotates *both* of the last two segments by a
        # progressive amount for a smooth curl, reused here.
        transform(ref("gremlin_Tail 04"), rotation=[-35.0, 0.0, 0.0]),
        transform(ref("gremlin_Tail 05"), rotation=[-70.0, 0.0, 0.0]),
        material([ref("gremlin_Head")], HIDE_DARK, "Dark Hide", roughness=0.55),
    ],
    [
        "a biped with a tail stretched long and curling upward, darker head",
        "biped, long curling tail, dark head",
        "A biped creature whose tail is noticeably stretched and curls up at the tip, with a darker-coloured head.",
    ],
)

add_build(
    "quadruped-bulkier",
    [
        figure("quadruped", "mastiff_"),
        material(refs("quadruped", "mastiff_"), HIDE_DUN, "Coat", roughness=0.75),
        # Chest is a _mass part; its own preset scale is (width=1.35,
        # along=1.45, depth=1.35), and clay_transform's scale replaces that
        # outright rather than multiplying it -- see the module docstring's
        # "absolute-scale trap". Widening/deepening by ~25% while leaving
        # the along-body length untouched: (1.35*1.25, 1.45, 1.35*1.25).
        transform(ref("mastiff_Chest"), scale=[1.7, 1.45, 1.7]),
        material(
            [
                ref("mastiff_Front foot.L"),
                ref("mastiff_Front foot.R"),
                ref("mastiff_Rear foot.L"),
                ref("mastiff_Rear foot.R"),
            ],
            (0.85, 0.85, 0.82),
            "White Socks",
            roughness=0.7,
        ),
    ],
    [
        "a bulkier quadruped with pale feet, like it is wearing socks",
        "beefier quadruped, pale feet",
        "A quadruped figure with a noticeably bulkier chest and pale, sock-like feet.",
    ],
)

add_build(
    "bird-long-wingtips",
    [
        figure("bird", "falcon_"),
        material(refs("bird", "falcon_"), FEATHER_GREY, "Feathers", roughness=0.55),
        transform(ref("falcon_Wing tip.L"), scale=[1.4, 1.4, 1.4]),
        transform(ref("falcon_Wing tip.R"), scale=[1.4, 1.4, 1.4]),
        material([ref("falcon_Beak")], GOLD, "Beak", roughness=0.3, metallic=0.4),
    ],
    [
        "a bird with dramatically lengthened wingtips and a gilded beak",
        "bird, long wingtips, gold beak",
        "A bird figure with dramatically lengthened wingtips and a gilded beak.",
    ],
)

add_build(
    "serpent-big-head",
    [
        figure("serpent", "viper_"),
        material(refs("serpent", "viper_"), SCALE_DARKGREEN, "Scales", roughness=0.35),
        transform(ref("viper_Head"), scale=[1.25, 1.25, 1.25]),
        transform(ref("viper_Tail tip"), scale=[0.6, 0.6, 0.6]),
        material([ref("viper_Head")], SCALE_DARKER, "Dark Scales", roughness=0.3),
    ],
    [
        "a serpent with an enlarged, darker head and a narrowed tail tip",
        "serpent, big dark head, thin tail tip",
        "A serpent figure with a tail that narrows to a fine point and a noticeably enlarged, darker head.",
    ],
)

add_build(
    "fish-tall-dorsal",
    [
        figure("fish", "carp_"),
        material(refs("fish", "carp_"), SCALE_ORANGE, "Scales", roughness=0.3),
        transform(ref("carp_Dorsal fin"), scale=[1.0, 1.6, 1.3]),
        material(
            [
                ref("carp_Pectoral fin.L"),
                ref("carp_Pectoral fin.R"),
                ref("carp_Pelvic fin.L"),
                ref("carp_Pelvic fin.R"),
            ],
            SILVER,
            "Silver Fins",
            roughness=0.3,
        ),
    ],
    [
        "a fish with a tall dorsal fin and silver side fins",
        "fish, tall dorsal fin, silver fins",
        "A fish figure with a noticeably tall dorsal fin and a silver pair of side fins.",
    ],
)

add_build(
    "insect-long-forelegs",
    [
        figure("insect", "mantis_"),
        material(refs("insect", "mantis_"), CHITIN_OLIVE, "Chitin", roughness=0.4),
        transform(ref("mantis_Leg A upper.L"), scale=[1.0, 1.3, 1.0]),
        transform(ref("mantis_Leg A upper.R"), scale=[1.0, 1.3, 1.0]),
        material([ref("mantis_Abdomen")], GREEN_ABDOMEN, "Abdomen", roughness=0.4),
    ],
    [
        "an insect with lengthened forelegs and a green abdomen",
        "insect, long forelegs, green abdomen",
        "An insect figure with a pair of noticeably lengthened forelegs and a green abdomen.",
    ],
)

add_build(
    "blob-big-crown",
    [
        figure("blob", "gel_"),
        material(refs("blob", "gel_"), SLIME_BLUE, "Ooze", roughness=0.2),
        set_params(ref("gel_Crown"), {"radius": 0.30}),
        material([ref("gel_Lobe.L"), ref("gel_Lobe.R")], LOBE_PURPLE, "Lobes", roughness=0.2),
    ],
    [
        "a blob with an enlarged crown and differently coloured side lobes",
        "blob, big crown, purple side lobes",
        "A blob figure with a noticeably enlarged crown and a pair of differently coloured side lobes.",
    ],
)

add_build(
    "bird-oversized-beak",
    [
        figure("bird", "toucan_"),
        material(refs("bird", "toucan_"), FEATHER_TOUCAN, "Feathers", roughness=0.55),
        # Reuses _gen_creatures.py's own "bird-oversized-beak" params verbatim
        # -- an already-verified generator-parameter edit on this exact part.
        set_params(ref("toucan_Beak"), {"size": [0.10, 0.42, 0.10]}),
    ],
    [
        "a bird figure with an oversized beak, toucan-like",
        "bird, huge beak, toucan-like",
        "A bird figure about 80 centimetres tall with a heavily oversized, toucan-like beak.",
        "a toucan-like bird figure with an enormous beak",
    ],
)

add_build(
    "quadruped-big-head",
    [
        figure("quadruped", "brute_"),
        material(refs("quadruped", "brute_"), BRUTE_HIDE, "Hide", roughness=0.8),
        set_params(ref("brute_Head"), {"radius": 0.13}),
    ],
    [
        "a stocky quadruped figure with an oversized head",
        "quadruped, big blocky head",
        "A stocky quadruped figure with a noticeably oversized head.",
        "a brutish quadruped creature with an unusually large head",
    ],
)

# ---- C. delete one named part, add a primitive replacement -----------------

add_build(
    "humanoid-helmet",
    [
        figure("humanoid", "sentinel_"),
        delete([ref("sentinel_Head")]),
        # Head (a _sphere) sits at Clay (0, 0.95, 0), radius 0.09 -- see the
        # module docstring's landmark-derivation note.
        primitive(
            "cone",
            "sentinel_helmet",
            params={"radius": 0.10, "height": 0.18, "segments": 16},
            translation=[0.0, 0.97, 0.0],
        ),
        material(
            refs_except("humanoid", "sentinel_", {"Head"}),
            STEEL_GREY,
            "Armour",
            roughness=0.5,
            metallic=0.3,
        ),
        material([ref("sentinel_helmet")], ARMOR_BLACK, "Helmet", roughness=0.35, metallic=0.5),
    ],
    [
        "an armoured humanoid figure wearing a pointed helmet instead of a bare head",
        "armoured figure, pointed helmet instead of a head",
        "An armoured humanoid figure whose head is replaced by a pointed helmet shape.",
    ],
)

add_build(
    "quadruped-long-tail-swap",
    [
        figure("quadruped", "warhorse_"),
        delete([ref("warhorse_Tail 01"), ref("warhorse_Tail 02")]),
        # Reuses _gen_creatures.py's own "quadruped-tube-tail" geometry
        # verbatim -- an already-verified tube path and translation rooted at
        # the hip landmark the two deleted segments used to occupy.
        primitive(
            "tube",
            "warhorse_tail",
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
            refs_except("quadruped", "warhorse_", {"Tail 01", "Tail 02"}) + [ref("warhorse_tail")],
            HIDE_BROWN,
            "Coat",
            roughness=0.75,
        ),
    ],
    [
        "a quadruped figure with its stub tail replaced by a long curved tail",
        "quadruped, long curved tail instead of a stub",
        "A quadruped figure whose short stub tail is replaced by one long, curved tail.",
    ],
)

add_build(
    "blob-spike-crown",
    [
        figure("blob", "spiky_"),
        delete([ref("spiky_Crown")]),
        # Crown (an _ico) sits at Clay (0, 0.66, 0), radius 0.22.
        primitive(
            "icosphere",
            "spiky_spike",
            params={"radius": 0.18, "subdivisions": 1},
            translation=[0.0, 0.70, 0.0],
            scale=[0.8, 1.6, 0.8],
        ),
        material(refs_except("blob", "spiky_", {"Crown"}), SLIME_TEAL, "Ooze", roughness=0.2),
        material([ref("spiky_spike")], SPIKE_GOLD, "Spike", roughness=0.15),
    ],
    [
        "a blob figure with a tall spike where its crown used to be",
        "blob, tall spike instead of a crown",
        "A blob figure with a tall, narrow spike standing where its crown used to be.",
    ],
)

add_build(
    "insect-pincers",
    [
        figure("insect", "soldier_"),
        delete([ref("soldier_Mandible.L"), ref("soldier_Mandible.R")]),
        # Mandible.L/.R (capsules) sit at Clay (+/-0.07, 0.58, 0.39).
        primitive(
            "box",
            "soldier_pincer_l",
            params={"size": [0.03, 0.02, 0.10]},
            translation=[0.07, 0.58, 0.42],
            rotation=[15.0, 0.0, 0.0],
        ),
        primitive(
            "box",
            "soldier_pincer_r",
            params={"size": [0.03, 0.02, 0.10]},
            translation=[-0.07, 0.58, 0.42],
            rotation=[15.0, 0.0, 0.0],
        ),
        material(
            refs_except("insect", "soldier_", {"Mandible.L", "Mandible.R"}),
            CHITIN_OLIVE,
            "Chitin",
            roughness=0.4,
        ),
        # Painted a shade apart from the body -- the gallery review found the
        # pincers invisible against a same-colour thorax/head silhouette when
        # both used CHITIN_BLACK.
        material(
            [ref("soldier_pincer_l"), ref("soldier_pincer_r")],
            (0.35, 0.10, 0.08),
            "Pincers",
            roughness=0.3,
            metallic=0.2,
        ),
    ],
    [
        "an insect figure with a pair of blocky pincers instead of mandibles",
        "insect, blocky pincers instead of mandibles",
        "An insect figure with a pair of blocky pincers in place of its usual mandibles.",
    ],
)

# ---- D. delete a hand, hold a prop in its place -----------------------------

add_build(
    "humanoid-hammer-hand",
    [
        figure("humanoid", "smith_"),
        delete([ref("smith_Hand.R")]),
        # Hand.R (a _box) sits at Clay (-0.34, 0.495, 0).
        primitive(
            "cylinder",
            "smith_hammer_handle",
            params={"radius": 0.016, "height": 0.14, "segments": 10},
            translation=[-0.34, 0.495, 0.0],
        ),
        primitive(
            "box",
            "smith_hammer_head",
            params={"size": [0.05, 0.09, 0.05]},
            translation=[-0.34, 0.615, 0.0],
        ),
        material(refs_except("humanoid", "smith_", {"Hand.R"}), SKIN_TAN, "Skin", roughness=0.6),
        material([ref("smith_hammer_handle")], WOOD_BROWN, "Handle", roughness=0.7),
        material(
            [ref("smith_hammer_head")], STEEL_GREY, "Hammer Head", roughness=0.4, metallic=0.6
        ),
    ],
    [
        "a smith figure holding a hammer in place of one hand",
        "smith figure, hammer instead of a hand",
        "A humanoid smith figure with a hammer in place of its right hand.",
    ],
)

# ---- E. two figures placed together -----------------------------------------

add_build(
    "knight-and-steed",
    [
        figure("humanoid", "knight2_", translation=[-0.6, 0.0, 0.0]),
        material(refs("humanoid", "knight2_"), STEEL_GREY, "Armour", roughness=0.5, metallic=0.3),
        figure("quadruped", "steed2_", translation=[0.8, 0.0, 0.0]),
        material(refs("quadruped", "steed2_"), HIDE_BROWN, "Coat", roughness=0.75),
    ],
    [
        "a knight figure standing beside its steed",
        "knight and steed together",
        "A knight figure standing next to its quadruped steed, both plainly coloured.",
    ],
)

add_build(
    "goblin-and-ooze",
    [
        figure("biped_tail", "goblin2_", translation=[-0.5, 0.0, 0.0]),
        material(refs("biped_tail", "goblin2_"), HIDE_GREYGREEN, "Hide", roughness=0.6),
        figure("blob", "ooze2_", translation=[0.6, 0.0, 0.0]),
        material(refs("blob", "ooze2_"), SLIME_TEAL, "Ooze", roughness=0.2),
    ],
    [
        "a goblin figure next to a small ooze",
        "goblin and ooze, side by side",
        "A goblin-like biped figure standing beside a small ooze creature.",
    ],
)

add_build(
    "hawk-and-beetle",
    [
        figure("bird", "hawk2_", translation=[-0.4, 0.0, 0.0]),
        material(refs("bird", "hawk2_"), FEATHER_RUST, "Feathers", roughness=0.6),
        figure("insect", "beetle2_", translation=[0.5, 0.0, 0.0]),
        material(refs("insect", "beetle2_"), CHITIN_OLIVE, "Chitin", roughness=0.4),
    ],
    [
        "a hawk figure perched near a beetle figure",
        "hawk and beetle, near each other",
        "A hawk-like bird figure standing near a beetle-like insect figure.",
    ],
)

# ---- F. a figure with a small prop resting beside it ------------------------


def _stone(name: str, radius: float, x: float, z: float) -> list[dict[str, Any]]:
    """A rounded stone prop, resting exactly on the ground (translation.y ==
    radius), placed beside a figure rather than touching it."""
    return [
        primitive(
            "icosphere",
            name,
            params={"radius": radius, "subdivisions": 1},
            translation=[x, radius, z],
        ),
        material([ref(name)], STONE_GREY, "Stone", roughness=0.9),
    ]


add_build(
    "serpent-and-stone",
    [
        figure("serpent", "cobra_"),
        material(refs("serpent", "cobra_"), SCALE_GREEN, "Scales", roughness=0.4),
        *_stone("cobra_stone", 0.10, 0.45, 0.0),
    ],
    [
        "a serpent figure coiled beside a smooth stone",
        "serpent next to a smooth stone",
        "A serpent figure coiled loosely, with one smooth stone resting beside it.",
    ],
)

add_build(
    "fish-and-stone",
    [
        figure("fish", "betta_"),
        material(refs("fish", "betta_"), SCALE_ORANGE, "Scales", roughness=0.3),
        *_stone("betta_stone", 0.09, 0.4, 0.1),
    ],
    [
        "a fish figure near a rounded stone on a pond floor",
        "fish next to a stone",
        "A fish figure swimming near one rounded stone resting on the pond floor below it.",
    ],
)

add_build(
    "insect-on-leaf",
    [
        figure("insect", "ant_"),
        material(refs("insect", "ant_"), CHITIN_OLIVE, "Chitin", roughness=0.4),
        primitive(
            "box", "ant_leaf", params={"size": [0.5, 0.01, 0.3]}, translation=[0.0, 0.005, -0.3]
        ),
        material([ref("ant_leaf")], GREEN_ABDOMEN, "Leaf", roughness=0.6),
    ],
    [
        "an insect figure resting on a flat leaf",
        "insect on a leaf",
        "An insect figure standing beside one flat, wide leaf on the ground.",
    ],
)

add_build(
    "blob-and-mossy-stone",
    [
        figure("blob", "muck_"),
        material(refs("blob", "muck_"), SLIME_TEAL, "Ooze", roughness=0.2),
        *_stone("muck_stone", 0.12, -0.45, 0.0),
    ],
    [
        "a blob figure oozing up against a mossy stone",
        "blob beside a mossy stone",
        "A blob figure resting right beside one mossy-looking stone.",
    ],
)

# =============================================================================
# Edits: (slug, prior, calls, [prompt phrasings])
# =============================================================================
EditRecipe = tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[str]]
edit_recipes: list[EditRecipe] = []


def add_edit(
    slug: str, prior: list[dict[str, Any]], calls: list[dict[str, Any]], prompts: list[str]
) -> None:
    assert 2 <= len(prompts) <= 8, f"{slug}: {len(prompts)} phrasings"
    edit_recipes.append((slug, prior, calls, prompts))


add_edit(
    "humanoid-forearm-red",
    [
        figure("humanoid", "knight3_"),
        material(refs("humanoid", "knight3_"), STEEL_GREY, "Armour", roughness=0.5, metallic=0.3),
    ],
    [material([ref("knight3_Forearm.L")], RED_PAINT, "Painted Forearm", roughness=0.4)],
    [
        "paint the knight's left forearm red",
        "the knight's left arm needs a red forearm plate",
        "Paint the knight figure's left forearm red.",
        "make the left forearm red",
        "for a heraldic paint scheme, colour this knight's left forearm bright red",
        "recolour the knight's left forearm to red",
    ],
)

add_edit(
    "biped-tail-longer",
    [
        figure("biped_tail", "imp2_"),
        material(refs("biped_tail", "imp2_"), HIDE_GREYGREEN, "Hide", roughness=0.6),
    ],
    [
        set_params(ref("imp2_Tail 04"), {"height": 0.15}),
        set_params(ref("imp2_Tail 05"), {"height": 0.15}),
        # See "biped-tail-long-curl"'s own comment: both of the last two
        # segments need a progressive rotation, or the un-rotated one kinks
        # against its rotated neighbour at their shared joint.
        transform(ref("imp2_Tail 04"), rotation=[-35.0, 0.0, 0.0]),
        transform(ref("imp2_Tail 05"), rotation=[-70.0, 0.0, 0.0]),
    ],
    [
        "make the imp's tail longer",
        "this imp's tail looks stubby -- lengthen it",
        "Lengthen the imp figure's tail.",
        "give it a longer tail",
        "for a more snake-like mischief-maker read, stretch out this imp's tail",
        "the imp's tail needs to be longer than it is now",
    ],
)

add_edit(
    "quadruped-tail-longer",
    [
        figure("quadruped", "dog_"),
        material(refs("quadruped", "dog_"), HIDE_BROWN, "Coat", roughness=0.75),
    ],
    [
        set_params(ref("dog_Tail 01"), {"height": 0.14}),
        set_params(ref("dog_Tail 02"), {"height": 0.12}),
    ],
    [
        "make the dog's tail longer",
        "this dog's tail is too short -- make it longer",
        "Lengthen the dog figure's tail.",
        "give the dog a longer tail",
        "for a happier-looking mutt, lengthen its tail a bit",
        "the dog's tail needs to be longer than it currently is",
    ],
)

add_edit(
    "bird-head-cone",
    [
        figure("bird", "sparrow_"),
        material(refs("bird", "sparrow_"), FEATHER_GREY, "Feathers", roughness=0.6),
    ],
    [
        delete([ref("sparrow_Head")]),
        # Head (a _sphere) sits at Clay (0, 0.88, 0.29).
        primitive(
            "cone",
            "sparrow_cone_head",
            params={"radius": 0.09, "height": 0.16, "segments": 16},
            translation=[0.0, 0.88, 0.29],
        ),
        material([ref("sparrow_cone_head")], (0.6, 0.6, 0.62), "Cone Head", roughness=0.4),
    ],
    [
        "swap the bird's head for a cone",
        "replace this bird's head with a plain cone",
        "Swap the bird figure's head for a simple cone shape.",
        "give it a cone instead of a head",
        "for a stylised, faceless look, swap the bird's head out for a cone",
        "the bird's head should be a cone instead of the usual shape",
    ],
)

add_edit(
    "fish-dorsal-blue",
    [
        figure("fish", "koi2_"),
        material(refs("fish", "koi2_"), SCALE_ORANGE, "Scales", roughness=0.3),
    ],
    [material([ref("koi2_Dorsal fin")], BLUE_PAINT, "Dorsal Fin", roughness=0.3)],
    [
        "paint the fish's dorsal fin blue",
        "this fish's back fin should be blue",
        "Paint the fish figure's dorsal fin blue.",
        "make the dorsal fin blue",
        "for a striking two-tone look, colour this fish's dorsal fin deep blue",
        "the fish needs a blue dorsal fin instead of orange",
    ],
)

add_edit(
    "serpent-head-bigger",
    [
        figure("serpent", "cobra2_"),
        material(refs("serpent", "cobra2_"), SCALE_GREEN, "Scales", roughness=0.4),
    ],
    [transform(ref("cobra2_Head"), scale=[1.35, 1.35, 1.35])],
    [
        "give the serpent a bigger head",
        "this serpent's head looks too small -- enlarge it",
        "Enlarge the serpent figure's head.",
        "make the head noticeably bigger",
        "for a more intimidating look, enlarge this serpent's head",
        "the serpent's head needs to be bigger than it is now",
    ],
)

add_edit(
    "insect-mandible-black",
    [
        figure("insect", "beetle2_"),
        material(refs("insect", "beetle2_"), CHITIN_OLIVE, "Chitin", roughness=0.4),
    ],
    [material([ref("beetle2_Mandible.L")], CHITIN_BLACK, "Left Mandible", roughness=0.3)],
    [
        "paint the beetle's left mandible black",
        "this beetle's left pincer should be glossy black",
        "Paint the insect figure's left mandible black.",
        "make the left mandible black",
        "for a menacing look, colour this beetle's left mandible glossy black",
        "the beetle's left mandible needs to be black, not olive",
    ],
)

add_edit(
    "blob-crown-bigger",
    [figure("blob", "ooze3_"), material(refs("blob", "ooze3_"), SLIME_TEAL, "Ooze", roughness=0.2)],
    [set_params(ref("ooze3_Crown"), {"radius": 0.30})],
    [
        "make the blob's crown bigger",
        "this ooze's crown looks small -- grow it",
        "Enlarge the blob figure's crown.",
        "grow the crown a bit",
        "for a more top-heavy silhouette, enlarge this blob's crown",
        "the blob's crown needs to be bigger than it currently is",
    ],
)

add_edit(
    "quadruped-add-chest-plate",
    [
        figure("quadruped", "steed3_"),
        material(refs("quadruped", "steed3_"), HIDE_DUN, "Coat", roughness=0.75),
    ],
    [
        primitive(
            "box",
            "steed3_plate",
            params={"size": [0.30, 0.05, 0.22]},
            translation=[0.0, 0.82, 0.23],
        ),
        material([ref("steed3_plate")], STEEL_GREY, "Plate", roughness=0.35, metallic=0.6),
    ],
    [
        "add an armour plate to the horse's chest",
        "Add a metal plate over the horse figure's chest.",
    ],
)


# --- coverage assertion -------------------------------------------------------


def _collect_refs(node: Any, found: list[str]) -> None:
    if isinstance(node, dict):
        if set(node.keys()) == {"$ref"}:
            found.append(node["$ref"])
            return
        for value in node.values():
            _collect_refs(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_refs(item, found)


def _check_coverage(records: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    """Every ``(key, part)`` pair, counted by how many records' own ``prior``
    + ``calls`` reference it via a real ``{"$ref": "<prefix><part>"}`` --
    raises if any pair the family is supposed to cover (every part of every
    key in :data:`PART_NAMES`) is referenced fewer than three times."""
    ref_index: dict[str, tuple[str, str]] = {}
    for prefix, key in PREFIX_KEY.items():
        for part in PART_NAMES[key]:
            ref_index[f"{prefix}{part}"] = (key, part)

    counts: dict[tuple[str, str], int] = {}
    for rec in records:
        found: list[str] = []
        for call in [*rec.get("prior", []), *rec.get("calls", [])]:
            _collect_refs(call.get("arguments", {}), found)
        for name in found:
            hit = ref_index.get(name)
            if hit is not None:
                counts[hit] = counts.get(hit, 0) + 1

    missing = [
        f"{key}:{part} ({counts.get((key, part), 0)}x)"
        for key, parts in PART_NAMES.items()
        for part in parts
        if counts.get((key, part), 0) < 3
    ]
    assert not missing, "figures family under-covers these (key:part, count): " + ", ".join(missing)
    return counts


# --- record assembly ----------------------------------------------------------


def main() -> None:
    records: list[dict[str, Any]] = []
    counter = 1

    for slug, calls, prompts in build_recipes:
        for prompt in prompts:
            records.append(
                {
                    "id": f"figures-{counter:04d}",
                    "family": "figures",
                    "kind": "build",
                    "prompt": prompt,
                    "calls": calls,
                    "notes": f"[{slug}]",
                }
            )
            counter += 1

    build_count = len(records)

    for slug, prior, calls, prompts in edit_recipes:
        for prompt in prompts:
            records.append(
                {
                    "id": f"figures-{counter:04d}",
                    "family": "figures",
                    "kind": "edit",
                    "prompt": prompt,
                    "prior": prior,
                    "calls": calls,
                    "notes": f"[{slug}]",
                }
            )
            counter += 1

    edit_count = len(records) - build_count

    missing_keys = sorted(set(PART_NAMES) - set(PREFIX_KEY.values()))
    assert not missing_keys, f"figures family never used these keys: {missing_keys}"

    _check_coverage(records)

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    print(
        f"wrote {len(records)} records ({build_count} build, {edit_count} edit, "
        f"{len(build_recipes)} build recipes, {len(edit_recipes)} edit recipes) -> {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
