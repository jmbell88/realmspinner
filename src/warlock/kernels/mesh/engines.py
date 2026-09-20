"""Per-engine export conventions: collider naming, LOD naming, axis handling.

CLAY-PLAN.md tranche 7's other half. :mod:`.colliders` decides *what shape* a
collider is; this module decides *what a target engine has to see in the
file* to recognise it as one, plus the two other per-target facts an export
profile bundles: how far past ``readiness.PROFILES``' own budgets an export
should be judged, and whether an OBJ export needs an axis/scale conversion
Clay's own glTF-convention geometry does not.

**Every one of the four ``glb_conversion`` matrices is the identity, and that
is a fact about glTF, not a coincidence across four unrelated engines.**
Clay's own convention already *is* glTF's -- metres, Y up (stated as fact for
this tranche, and true throughout this package: ``primitives.py``'s own
module docstring states the same axes for the same reason). A glTF file
carries no engine-specific axis information for an importer to get wrong, so
every target's own glTF importer performs whatever conversion *it* needs
internally: Godot and the two WebGL engines need none at all (their native
convention already matches glTF's, see each :class:`EngineProfile`'s own
``notes``), and Unreal's and Unity's glTF importers are documented to convert
on the way in. Shipping the file in glTF's own convention and trusting the
importer is therefore not a simplification, it is the *more* correct choice
than guessing at each engine's internal representation and converting before
export -- a conversion this module would have no way to verify against the
receiving engine's actual importer version. **OBJ has no such contract.** The
format carries no axis metadata at all, so nothing on the receiving end can
correct for one convention or another, and a target whose native axes differ
from glTF's needs the conversion applied *before* the file is written. That is
the one place :attr:`EngineProfile.obj_conversion` is not the identity below,
and each non-identity case says in its own comment how confident this session
is in the exact matrix, not only that one is applied.

**Engine facts and their confidence**, so the tranche this hands off to can
tell which are citable and which are this session's best-effort judgement:

* Unreal recognises ``UCX_<MeshName>_NN`` (convex), ``UBX_`` (box), ``USP_``
  (sphere), ``UCP_`` (capsule) collision-mesh prefixes on FBX/glTF import, and
  ``_LOD0``, ``_LOD1``... suffixes to build a static mesh LOD group on
  import. High confidence -- both are long-standing, widely documented Unreal
  import conventions.
* Unreal is Z-up, centimetres. High confidence, and given directly in this
  tranche's own brief. Its glTF importer converts on the way in; this module
  does not additionally convert glTF/GLB exports (see above).
* Godot 4's scene importer recognises the node-name suffixes ``-col``
  (collision added to the visual mesh, trimesh/concave), ``-colonly``
  (trimesh-only, no visual mesh), ``-convcol``/``-convcolonly`` (the same
  pair, but a convex shape), plus ``-rigid`` and ``-navmesh``. High
  confidence -- given directly in this tranche's brief. Every collider this
  module names is convex by construction (box/sphere/capsule are already
  convex shapes; :func:`.colliders.convex_hull` and :func:`.colliders.compound`
  are convex by definition), so :func:`_godot_collider_name` only ever emits
  ``-convcolonly``: a collider-only node, convex. Godot has no naming
  convention this session could find that maps a box- or sphere-*shaped* mesh
  onto an analytic ``BoxShape3D``/``SphereShape3D`` by name; the node becomes
  a ``ConvexPolygonShape3D`` built from the mesh's own geometry regardless of
  which of the five kinds produced it.
* Godot 4 is Y-up, right-handed, one unit = one metre -- the same convention
  glTF and this package already use. High confidence (stated plainly in
  Godot's own documentation, and is why Godot's glTF import is routinely
  described as needing no axis correction at all). Godot generates its own
  LODs on import, so :attr:`EngineProfile.lod_suffix` is ``None`` -- there is
  nothing for this module to name.
* Unity ships no first-party mesh-name convention for collision at all,
  unlike Unreal and Godot -- everything Unity-side is a ``Collider``
  component added in the editor, by script, or by a project's own
  ``AssetPostprocessor``. :func:`_unity_collider_name`'s ``_collider_NN``
  infix is therefore *this project's own convention*, written for a
  custom ``AssetPostprocessor`` a Unity-side project would need to supply,
  not a Unity feature this module can claim compliance with.
* Unity is Y-up, one unit = one metre (not centimetres -- FBX's own default
  export scale is cm, which is a fact about FBX, not about Unity, and glTF
  carries no such default), but **left-handed**, where glTF is right-handed.
  High-moderate confidence: well documented as the commonly-applied
  glTF-to-Unity conversion (e.g. by UnityGLTF and similar importers) is a
  single Z-axis negation with the up axis and scale left alone -- that is
  what :attr:`EngineProfile.obj_conversion` applies for ``unity`` below.
  ``_LOD0``, ``_LOD1``... is this tranche's own brief's fact for Unity, and it
  names the *meshes*; Unity additionally needs a ``LODGroup`` component
  referencing them, which naming alone cannot create -- said in
  :attr:`EngineProfile.notes`.
* three.js and Babylon.js (this profile's ``webgl``) are both Y-up,
  right-handed -- the same convention as glTF, which both document as their
  preferred/native interchange format for exactly that reason. High
  confidence. Neither has an import-time mesh-*naming* convention for
  collision or LOD (both build a collider or an ``LOD``
  object/``addLODLevel`` call in application code instead), so this module's
  ``collider_`` node-prefix-plus-``extras`` convention
  (:func:`_webgl_collider_name`) is, like Unity's, this project's own --
  stated as such in :attr:`EngineProfile.notes` -- and :attr:`EngineProfile.lod_suffix`
  is ``None`` for a different reason than Godot's ``None``: Godot's means "the
  importer builds LODs itself, nothing to name"; WebGL's means "there is no
  naming convention to hook into at all, LODs are wired up by the receiving
  application's own code".
* Unreal's raw *OBJ* importer's own axis assumption is not something this
  session found a confident citation for -- OBJ import is a far less
  documented Unreal path than its FBX/glTF ones. :attr:`EngineProfile.obj_conversion`
  for ``unreal`` applies the well-documented parts (Z-up, centimetres) plus a
  best-effort handedness mirror this session cannot independently verify;
  said plainly in that entry's own comment, and Unreal's glTF path (needing
  no guess at all) is the one this module actually recommends.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .colliders import COLLIDER_KINDS
from .elements import OpError

__all__ = [
    "ENGINES",
    "EngineProfile",
    "collider_name",
    "is_collider_name",
    "lod_name",
]


_ALL_COLLIDER_KINDS = frozenset(COLLIDER_KINDS)


@dataclass(frozen=True)
class EngineProfile:
    """One export target's naming, budget and axis conventions.

    ``collider_naming`` is ``(kind, mesh_name, index) -> node name`` --
    called through :func:`collider_name`, which also checks *kind* against
    ``collider_kinds`` first, so a caller gets a refusal naming the engine
    and the kind rather than a name nothing on the other end recognises.
    ``lod_suffix`` is a ``str.format``-style template taking ``mesh_name``
    and ``level``, or ``None`` when there is nothing to name (see the module
    docstring for the two different reasons that can be true).
    ``obj_conversion``/``glb_conversion`` are 4x4s in
    :func:`~.mesh.transformed`'s column-vector convention (``M @ v``);
    ``glb_conversion`` is the identity for every profile below (see the
    module docstring for why that is a fact about glTF, not a per-engine
    coincidence) and is still carried as a field rather than assumed, so a
    caller never has to special-case "glTF needs no matrix" against "OBJ
    might".
    """

    key: str
    label: str
    readiness_profile: str
    collider_naming: Callable[[str, str, int], str]
    collider_kinds: frozenset[str]
    lod_suffix: str | None
    obj_conversion: np.ndarray
    glb_conversion: np.ndarray
    notes: str


# --- collider naming ----------------------------------------------------------


def _unreal_collider_name(kind: str, mesh_name: str, index: int) -> str:
    prefix = {"box": "UBX_", "sphere": "USP_", "capsule": "UCP_"}.get(kind, "UCX_")
    return f"{prefix}{mesh_name}_{index:02d}"


def _godot_collider_name(kind: str, mesh_name: str, index: int) -> str:
    # Every kind this package fits is convex; see the module docstring for
    # why that makes ``-convcolonly`` the one suffix this function ever
    # emits, regardless of *kind*. The index sits before the suffix, not
    # after, because Godot matches the suffix at the very end of the name.
    return f"{mesh_name}_{index:02d}-convcolonly"


def _unity_collider_name(kind: str, mesh_name: str, index: int) -> str:
    # Our own convention -- see the module docstring's Unity entry.
    return f"{mesh_name}_collider_{index:02d}"


def _webgl_collider_name(kind: str, mesh_name: str, index: int) -> str:
    # Our own convention -- see the module docstring's three.js/Babylon
    # entry. The exporter pairs this with glTF ``extras`` metadata naming
    # *kind* and the fit params, which the prefix alone cannot carry.
    return f"collider_{mesh_name}_{index:02d}"


_COLLIDER_PATTERNS: dict[str, Callable[[str], bool]] = {
    "unreal": lambda name: name.startswith(("UBX_", "USP_", "UCP_", "UCX_")),
    "godot4": lambda name: name.endswith(("-col", "-colonly", "-convcol", "-convcolonly")),
    "unity": lambda name: "_collider_" in name,
    "webgl": lambda name: name.startswith("collider_"),
}


# --- axis/scale conversion (OBJ only -- see the module docstring) -----------


def _identity4() -> np.ndarray:
    return np.eye(4, dtype="f8")


def _unreal_obj_conversion() -> np.ndarray:
    # Well documented: metres -> centimetres (x100), Y-up -> Z-up (swap Y/Z).
    # The handedness mirror (glTF is right-handed, Unreal left-handed) is
    # applied as the sign on the Y-slot below; *which* axis carries that sign
    # is this session's best-effort judgement, not an independently verified
    # fact about Unreal's raw OBJ importer specifically -- see the module
    # docstring's Unreal entry. Prefer the glTF path where available.
    m = _identity4()
    m[:3, :3] = np.array(
        [
            [100.0, 0.0, 0.0],
            [0.0, 0.0, -100.0],
            [0.0, 100.0, 0.0],
        ]
    )
    return m


def _unity_obj_conversion() -> np.ndarray:
    # Y-up and metres already match; only the handedness differs (glTF
    # right-handed, Unity left-handed), which a single Z negation corrects --
    # the same conversion UnityGLTF-style importers apply. See the module
    # docstring's Unity entry.
    m = _identity4()
    m[2, 2] = -1.0
    return m


# --- registry -------------------------------------------------------------


ENGINES: dict[str, EngineProfile] = {
    "godot4": EngineProfile(
        key="godot4",
        label="Godot 4",
        readiness_profile="godot-desktop",
        collider_naming=_godot_collider_name,
        collider_kinds=_ALL_COLLIDER_KINDS,
        lod_suffix=None,  # Godot generates its own LODs on import.
        obj_conversion=_identity4(),  # already Y-up, right-handed, metres.
        glb_conversion=_identity4(),
        notes=(
            "Y-up, right-handed, 1 unit = 1 m -- matches glTF exactly, no "
            "conversion needed for either export format. Colliders are all "
            "named '<mesh>_<NN>-convcolonly' (collision-only, convex): every "
            "kind this package fits is already convex, and Godot has no "
            "naming convention distinguishing a box/sphere/capsule fit from "
            "a generic convex hull. readiness_profile picks 'godot-desktop'; "
            "'godot-mobile' is available directly from readiness.PROFILES "
            "for a stricter budget."
        ),
    ),
    "unity": EngineProfile(
        key="unity",
        label="Unity",
        readiness_profile="unity",
        collider_naming=_unity_collider_name,
        collider_kinds=_ALL_COLLIDER_KINDS,
        lod_suffix="{mesh_name}_LOD{level}",
        obj_conversion=_unity_obj_conversion(),
        glb_conversion=_identity4(),
        notes=(
            "Y-up, left-handed, 1 unit = 1 m. glTF export needs no "
            "conversion (Unity's glTF importer converts handedness on the "
            "way in); OBJ export negates Z (this project's own conversion, "
            "not a spec). Unity has no first-party collider- or "
            "LOD-naming convention: '<mesh>_collider_<NN>' is our own hint "
            "for a custom AssetPostprocessor, and '<mesh>_LOD<n>' meshes "
            "still need a LODGroup component wired up by hand or by script "
            "-- the naming alone does not create one, unlike Unreal's."
        ),
    ),
    "unreal": EngineProfile(
        key="unreal",
        label="Unreal",
        readiness_profile="unreal",
        collider_naming=_unreal_collider_name,
        collider_kinds=_ALL_COLLIDER_KINDS,
        lod_suffix="{mesh_name}_LOD{level}",
        obj_conversion=_unreal_obj_conversion(),
        glb_conversion=_identity4(),
        notes=(
            "Z-up, centimetres, left-handed. glTF export needs no "
            "conversion (Unreal's glTF importer converts on the way in, "
            "authoritatively); OBJ export applies a best-effort axis swap "
            "and x100 scale -- see this module's own docstring for the "
            "confidence split. Collision prefixes ('UCX_'/'UBX_'/'USP_'/"
            "'UCP_') and '_LOD<n>' suffixes are both real, documented Unreal "
            "import conventions."
        ),
    ),
    "webgl": EngineProfile(
        key="webgl",
        label="WebGL (three.js / Babylon.js)",
        readiness_profile="webgl",
        collider_naming=_webgl_collider_name,
        collider_kinds=_ALL_COLLIDER_KINDS,
        lod_suffix=None,  # no naming convention exists to hook into (see module docstring).
        obj_conversion=_identity4(),  # already Y-up, right-handed.
        glb_conversion=_identity4(),
        notes=(
            "Y-up, right-handed, glTF-native -- both engines document glTF "
            "as their preferred interchange format for exactly that reason, "
            "so neither export format needs a conversion. Colliders are "
            "named 'collider_<mesh>_<NN>', our own convention (there is no "
            "first-party one), meant to be paired with glTF node 'extras' "
            "metadata carrying the fit kind and params; three.js/Babylon.js "
            "both expose glTF extras to application code. Neither engine has "
            "an import-time LOD *naming* convention -- LOD levels are wired "
            "up in application code (THREE.LOD / Mesh.addLODLevel), so "
            "lod_suffix is None for a different reason than Godot's: there "
            "is nothing to name, not that the importer does it for you."
        ),
    ),
}


# --- helpers ----------------------------------------------------------------


def collider_name(engine_key: str, kind: str, mesh_name: str, index: int) -> str:
    """The node name *engine_key*'s importer expects for a *kind* collider.

    Refuses (:class:`~.elements.OpError`) a *kind* the profile does not
    support, or an unknown *engine_key* -- named rather than a bare
    ``KeyError``, the same "a refusal is a sentence" contract every op in
    this package keeps.
    """
    if engine_key not in ENGINES:
        raise OpError(f"Unknown engine profile {engine_key!r}.")
    eng = ENGINES[engine_key]
    if kind not in eng.collider_kinds:
        raise OpError(f"{eng.label} does not support a {kind!r} collider.")
    return eng.collider_naming(kind, mesh_name, index)


def lod_name(engine_key: str, mesh_name: str, level: int) -> str:
    """The mesh name *engine_key* expects for LOD *level*.

    Refuses when the engine has no file-naming LOD convention to hook into
    (:attr:`EngineProfile.lod_suffix` is ``None``) -- see each profile's own
    ``notes`` for why that is true for it.
    """
    if engine_key not in ENGINES:
        raise OpError(f"Unknown engine profile {engine_key!r}.")
    eng = ENGINES[engine_key]
    if eng.lod_suffix is None:
        raise OpError(f"{eng.label} has no file-naming LOD convention; see its own notes.")
    return eng.lod_suffix.format(mesh_name=mesh_name, level=level)


def is_collider_name(engine_key: str, name: str) -> bool:
    """Whether *name* already looks like a collider node under *engine_key*'s
    own convention -- generous to the engine's whole documented vocabulary
    (e.g. Godot's ``-col``/``-convcol`` too), not only what
    :func:`collider_name` itself emits."""
    if engine_key not in ENGINES:
        raise OpError(f"Unknown engine profile {engine_key!r}.")
    return _COLLIDER_PATTERNS[engine_key](name)
