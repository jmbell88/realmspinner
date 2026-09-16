"""A GLB loader, in numpy and Pillow.

Hand-rolled rather than delegated. trimesh discards a scene root's transform --
the constraint ``normalize_glb`` writes around, and one this loader is under no
obligation to inherit -- and it has no notion of a skin at all. pygltflib parses
the JSON but still leaves accessor decoding here. What is left after those two
is small enough to own.

The node graph stays live after loading: posing a rig means setting a joint
node's local rotation and recomputing world matrices, so nothing is baked.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...glbio import read_glb
from . import math3d as m3

log = logging.getLogger(__name__)

# glTF componentType -> numpy dtype.
_COMPONENT = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}

#: Ceilings on what a hand-supplied file may declare. ``check_glb`` at the
#: import door is deliberately structural-only, so a ≤100 MB GLB reaches this
#: loader with whatever numbers its JSON chunk claims -- and this loader runs on
#: the frame thread. Both are module-level so a test can lower them rather than
#: craft a hostile asset.
#:
#: A JSON chunk declaring millions of nodes is a hang, not a scene.
MAX_NODES = 100_000
#: The 2026-09-05 audit, finding clay-05: ``load`` bounded ``nodes`` here but
#: iterated ``materials`` and ``meshes`` with no ceiling of their own, and an
#: *unreferenced* declaration -- a material array entry no mesh points at, a
#: mesh whose primitives read no accessor -- never calls ``_charge``, so
#: ``_spent`` (the byte budget below) never moves no matter how many of them
#: there are. Reproduced: 500,000 material entries loaded in 3.7s with
#: ``_spent`` staying at 0 throughout -- a hang with no bytes to refuse. Same
#: value as ``MAX_NODES``: a material or a mesh is the same order of JSON cost
#: as a node, and both are a hang before they are a scene.
MAX_MATERIALS = 100_000
MAX_MESHES = 100_000
#: The same argument one field over, for the two arrays Mason's export added
#: (cameras and KHR_lights_punctual's lights): both are tiny per entry and
#: neither is charged against the byte budget at all, because a camera or a
#: light decodes no accessor and no image -- so a file declaring a million of
#: either is finding-clay-05's hang wearing a third and fourth name. Same
#: value for the same reason: a JSON entry of this size is a hang before it
#: is a scene.
MAX_CAMERAS = 100_000
MAX_LIGHTS = 100_000
#: One field over from ``MAX_MESHES``: that bounds how many *mesh* entries a
#: file may declare, but nothing bounded how many *primitives* one mesh entry
#: may declare. The 2026-09-16 audit found a single mesh referencing the same
#: tiny, already-cached accessor from millions of primitive entries sails past
#: MAX_NODES/MAX_MATERIALS/MAX_MESHES/MAX_CAMERAS/MAX_LIGHTS and
#: ``MAX_TOTAL_BYTES`` alike -- the shared accessor means the *bytes* stay
#: small even as the *entries* do not -- while still costing seconds to build
#: one :class:`Primitive` object per entry: reproduced at 2,000,000 primitives
#: in a 66 MB GLB (comfortably under Clay's own 100 MB import-door ceiling),
#: ``load`` took 6.3 s. Same value as its siblings above for the same reason:
#: a primitive entry is the same order of JSON cost as a mesh or a node, and a
#: file declaring this many of them is a hang before it is a scene.
MAX_PRIMITIVES = 100_000
#: Mirrors ``service.validation.MAX_IMAGE_PIXELS`` without importing service
#: into the viewer (the viewer imports no business-logic layer).
MAX_TEXTURE_PIXELS = 16_000_000
#: What one accessor may decode to. Every *other* accessor path is bounded by
#: the BIN chunk it reads out of -- ``_check_span`` refuses a span past the end
#: of it, so a 400-byte GLB cannot ask for more than 400 bytes of geometry. The
#: no-``bufferView`` path is the one that allocates without touching the buffer
#: at all, so nothing downstream of it ever gets a turn: ``count:
#: 1_000_000_000, type: "MAT4"`` in a file that fits in a packet asks numpy for
#: 64 GB. 256 MiB is two orders past the largest accessor this pipeline
#: produces (a 2 M-triangle mesh's index stream is 24 MB) and four orders short
#: of what the field can express.
MAX_ACCESSOR_BYTES = 1 << 28

#: What one *document* may decode to, across every accessor and texture
#: combined. MAX_ACCESSOR_BYTES bounds a single array; nothing bounded the
#: *sum* of them, so a file with many primitives -- or one primitive's
#: accessor replayed by many nodes -- allocated without limit as long as each
#: individual array stayed under the per-accessor ceiling (H01). Measured: a
#: 4,036-byte bufferless GLB with 128 primitives produced 6,144,000 bytes of
#: geometry across 128 separate arrays, and nothing stopped scaling the
#: primitive count further. 768 MiB is comfortably above the largest asset
#: this pipeline produces (MAX_ACCESSOR_BYTES's own docstring: a 2 M-triangle
#: index stream is 24 MB; five fully populated 16-megapixel texture slots are
#: another ~320 MB) while still refusing a file that keeps asking for more.
#:
#: No `dev/measurements/` document backs the figure and none is owed: this is a
#: safety ceiling derived by arithmetic from what the pipeline can produce, not
#: a threshold the stored corpus is keyed on, so it moves when the arithmetic
#: above it moves rather than when a run says so. The 2026-09-11 audit (finding
#: docs-09) noted it sits in the checkup's example list beside corpus-keyed
#: constants like SEAM_MAX, which implied an obligation it does not have --
#: hence this sentence rather than a document.
MAX_TOTAL_BYTES = 768 * (1 << 20)


@dataclass
class Material:
    """A pbrMetallicRoughness material, with its textures already decoded.

    Images are kept as raw RGBA bytes plus a size rather than as PIL objects:
    the only consumer uploads them to the GPU, and a decoded PIL image held for
    the life of the model is 30 MB of nothing.
    """

    name: str = ""
    base_color_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0
    emissive_factor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    double_sided: bool = False
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    # (width, height, rgba_bytes) per slot, or None.
    base_color: tuple[int, int, bytes] | None = None
    metallic_roughness: tuple[int, int, bytes] | None = None
    normal: tuple[int, int, bytes] | None = None
    emissive: tuple[int, int, bytes] | None = None
    occlusion: tuple[int, int, bytes] | None = None


@dataclass
class Primitive:
    positions: np.ndarray  # (n, 3) f4
    indices: np.ndarray  # (m,) u4
    normals: np.ndarray | None = None  # (n, 3) f4
    uvs: np.ndarray | None = None  # (n, 2) f4
    # Promoted from uint8 to int32: GLSL 3.30 has no unsigned-byte vertex
    # attribute that survives as an integer, and ivec4 is what the skinning
    # shader indexes its palette with.
    joints: np.ndarray | None = None  # (n, 4) i4
    weights: np.ndarray | None = None  # (n, 4) f4
    material: Material = field(default_factory=Material)
    #: The primitive's own axis-aligned box in *its* space, computed once.
    #: ``Model.bounds`` runs per frame for the inspector and for framing, and
    #: its docstring's claim that it does not touch every vertex was only true
    #: of the transform: the ``min``/``max`` that make the box were a full pass
    #: over the positions, every frame, on 443k vertices. Positions never move
    #: -- a pose moves the *node* -- so the box is a property of the primitive.
    _box: tuple[np.ndarray, np.ndarray] | None = field(
        default=None, repr=False, compare=False
    )

    def box(self) -> tuple[np.ndarray, np.ndarray] | None:
        """``(min, max)`` over this primitive's positions, or None if empty."""

        if self._box is None:
            if len(self.positions) == 0:
                return None
            self._box = (self.positions.min(axis=0), self.positions.max(axis=0))
        return self._box


@dataclass
class Skin:
    joints: list[int]  # node indices, in palette order
    inverse_bind: np.ndarray  # (n, 4, 4)


#: KHR_lights_punctual's three kinds, verbatim. There is no fourth, and no
#: second name for any of these three: Mason's ``LightNode`` carries the same
#: strings, because the export *is* the definition and a parallel vocabulary
#: buys a conversion function and a place for the two to drift.
LIGHT_KINDS = ("directional", "point", "spot")


@dataclass
class Camera:
    """A perspective camera, in glTF's own fields and units (radians, metres).

    Added for Mason's scene export: a scene editor places camera markers and
    an engine importer reads them back, and neither half has anywhere to put
    one otherwise. Orthographic is deliberately absent -- nothing in this
    project authors one, and a field that no writer fills and no reader trusts
    is worse than its absence.

    ``zfar`` of 0 means the file carries no ``zfar`` at all, which is glTF's
    own way of saying an infinite perspective projection; ``aspect_ratio`` of
    0 likewise means "whatever the viewport is", which is what the spec says a
    missing ``aspectRatio`` means. Zero rather than ``None`` for both because
    every other optional number in this module is a float a caller can set
    without first deciding whether the field exists.
    """

    name: str = ""
    yfov: float = 1.0471975511965976  # 60 degrees
    znear: float = 0.1
    zfar: float = 0.0
    aspect_ratio: float = 0.0


@dataclass
class Light:
    """A KHR_lights_punctual light, in the extension's own fields and units.

    **Direction is not a field.** The extension defines a light's direction as
    its node's local ``-Z``, so the node's rotation is the only place that fact
    lives -- a separate direction here would be a second copy of one fact that
    can disagree with the first, which is the same argument
    ``mason/nodes.py``'s ``LightNode`` docstring makes for carrying no
    direction either.

    ``range`` of 0 means the file carries no ``range``, which the extension
    defines as an infinite one; the kinds that ignore it entirely
    (``directional``) never write it regardless. Cone angles apply to
    ``spot`` alone and are written only for it.
    """

    name: str = ""
    kind: str = "point"
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    range: float = 0.0
    inner_cone_angle: float = 0.0
    outer_cone_angle: float = 0.7853981633974483  # pi / 4


@dataclass
class Node:
    name: str = ""
    translation: np.ndarray = field(default_factory=lambda: m3.vec3())
    rotation: np.ndarray = field(default_factory=m3.quat_identity)
    scale: np.ndarray = field(default_factory=lambda: m3.vec3(1, 1, 1))
    children: list[int] = field(default_factory=list)
    mesh: int | None = None
    skin: int | None = None
    #: Indices into ``Model.cameras`` / ``Model.lights``. A node carries at
    #: most one of each, which is the spec's rule for ``camera`` and the
    #: extension's rule for its ``light``.
    camera: int | None = None
    light: int | None = None
    # Filled by Model.update_world(); never trusted before that runs.
    world: np.ndarray = field(default_factory=m3.identity)

    def local(self) -> np.ndarray:
        return m3.compose(self.translation, self.rotation, self.scale)


class Model:
    """A loaded GLB: a node graph, its meshes and its skins."""

    def __init__(
        self,
        nodes: list[Node],
        roots: list[int],
        meshes: list[list[Primitive]],
        skins: list[Skin],
        skipped_textures: int = 0,
        cameras: list[Camera] | None = None,
        lights: list[Light] | None = None,
    ) -> None:
        self.nodes = nodes
        self.roots = roots
        self.meshes = meshes
        self.skins = skins
        # Keyword, defaulted, and after ``skipped_textures``: every existing
        # caller builds a Model positionally out of the first four arguments
        # (``Model(nodes, roots, meshes, skins)``), and a document with
        # neither a camera nor a light is exactly the document every one of
        # them had before these two existed.
        self.cameras: list[Camera] = list(cameras or [])
        self.lights: list[Light] = list(lights or [])
        # How many of this file's images the loader could not use (D42). The
        # stated policy is that a texture is a cosmetic loss and never a reason
        # to refuse a file -- which is right, and left the loss reported only in
        # the log, so an untextured-looking mesh was indistinguishable from a
        # mesh that was never textured. This is the count the UI says it with.
        self.skipped_textures = skipped_textures
        # A joint node is addressed by name everywhere above this layer: a
        # pose is a bone->rotation map, and the browser never saw an index.
        self.by_name: dict[str, int] = {}
        for i, node in enumerate(nodes):
            # First wins: a duplicate name is a broken rig either way, and
            # picking the later one would silently move a different joint.
            if node.name and node.name not in self.by_name:
                self.by_name[node.name] = i
        self.rest_rotations = [n.rotation.copy() for n in nodes]
        # The mirror of rest_rotations, for the one translation posing may
        # move: the root joint, when the Poser previews a root offset as
        # ``rest + delta``. Remembered for every node because which one is the
        # root is the editor's knowledge, not the file's.
        self.rest_translations = [n.translation.copy() for n in nodes]
        self.update_world()

    # -- transforms --------------------------------------------------------

    def update_world(self) -> None:
        """Recompute every node's world matrix from the roots down.

        Called once at load and again after every pose change. Iterative
        rather than recursive: a skeleton is shallow but a mesh hierarchy from
        an exporter need not be, and a blown stack in the frame loop is not a
        failure mode worth having.
        """
        # ``seen`` and the range check are not defensiveness about our own
        # exporter: a hand-supplied GLB whose children form a cycle, or name a
        # node index that does not exist, is reachable through import (which is
        # deliberately structural-only) and would spin or raise here -- on the
        # frame thread. A malformed graph costs the malformed part of itself.
        seen: set[int] = set()
        stack = [(r, m3.identity()) for r in reversed(self.roots)]
        while stack:
            index, parent = stack.pop()
            if index in seen or not 0 <= index < len(self.nodes):
                continue
            seen.add(index)
            node = self.nodes[index]
            node.world = parent @ node.local()
            for child in reversed(node.children):
                stack.append((child, node.world))

    def mesh_instances(self) -> list[tuple[Node, list[Primitive]]]:
        return [(n, self.meshes[n.mesh]) for n in self.nodes if n.mesh is not None]

    def joint_palette(self, node: Node) -> list[np.ndarray] | None:
        """The skinning matrices for one mesh node, in palette order.

        Per the glTF spec the mesh node's own world transform is divided back
        out: the vertices are already in the skin's space, so leaving it in
        would apply the grounding transform twice.
        """
        if node.skin is None:
            return None
        skin = self.skins[node.skin]
        inv_mesh = np.linalg.inv(node.world)
        return [
            inv_mesh @ self.nodes[j].world @ skin.inverse_bind[i]
            for i, j in enumerate(skin.joints)
        ]

    # -- posing ------------------------------------------------------------

    def set_rotation(self, bone: str, quat_xyzw: Any) -> bool:
        """Set one joint node's *local* rotation. -> whether the bone exists.

        Local, not world: that is the entire pose contract. A glTF joint's
        local transform is what the Blender worker reconstructs its basis from,
        so anything else here would silently disagree with the bake.
        """
        index = self.by_name.get(bone)
        if index is None:
            return False
        self.nodes[index].rotation = m3.quat_normalize(np.asarray(quat_xyzw, dtype="f8"))
        return True

    def get_rotation(self, bone: str) -> np.ndarray | None:
        index = self.by_name.get(bone)
        return None if index is None else self.nodes[index].rotation.copy()

    def pose(self) -> dict[str, list[float]]:
        """Every joint whose rotation differs from rest, as XYZW lists."""
        out: dict[str, list[float]] = {}
        for name, index in self.by_name.items():
            node = self.nodes[index]
            if not np.allclose(node.rotation, self.rest_rotations[index], atol=1e-7):
                out[name] = [float(v) for v in node.rotation]
        return out

    # -- measurement -------------------------------------------------------

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """World-space AABB over every mesh, at the current pose.

        Computed from the eight corners of each primitive's own box rather
        than from every vertex: a 443k-vertex transform per frame would be the
        most expensive thing in the viewer, and framing does not need to be
        tighter than the box.
        """
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for node, prims in self.mesh_instances():
            for prim in prims:
                local = prim.box()
                if local is None:
                    continue
                pmin, pmax = local
                corners = np.array(
                    [
                        [x, y, z]
                        for x in (pmin[0], pmax[0])
                        for y in (pmin[1], pmax[1])
                        for z in (pmin[2], pmax[2])
                    ],
                    dtype="f8",
                )
                world = (node.world @ np.hstack([corners, np.ones((8, 1))]).T).T[:, :3]
                lo = np.minimum(lo, world.min(axis=0))
                hi = np.maximum(hi, world.max(axis=0))
        if not np.isfinite(lo).all():
            return m3.vec3(), m3.vec3()
        return lo, hi

    @property
    def triangle_count(self) -> int:
        return sum(len(p.indices) // 3 for prims in self.meshes for p in prims)

    @property
    def vertex_count(self) -> int:
        """Total vertices, computed once (B18): the primitives are immutable
        after load and the inspector asks for this every frame."""
        count = getattr(self, "_vertex_count", None)
        if count is None:
            count = sum(len(p.positions) for prims in self.meshes for p in prims)
            self._vertex_count = count
        return count


# --- loading ----------------------------------------------------------------


#: Extensions this loader implements. Anything a file lists as *required*
#: beyond these changes what its bytes mean, so it is refused rather than
#: decoded as though the extension were absent.
#: ``KHR_lights_punctual`` is here because this loader now reads it: a file
#: that *requires* it is a file whose lights this build can actually restore,
#: so refusing it would be refusing a document we write ourselves. Our own
#: writer lists it under ``extensionsUsed`` rather than
#: ``extensionsRequired`` -- a reader that drops the lights still gets the
#: geometry, which is what "used" means -- but a third-party exporter is free
#: to require it and that file is now readable rather than refused.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({"KHR_lights_punctual"})


def load(path: Path | bytes) -> Model:
    gltf, buffer = read_glb(path)
    # Refused rather than misread. KHR_mesh_quantization is the one that will
    # arrive first -- gltfpack -c writes it -- and a quantized position stream
    # decoded as though it were plain floats is geometry that looks like
    # nothing, with nothing in the data to say why.
    required = set(gltf.get("extensionsRequired") or []) - SUPPORTED_EXTENSIONS
    if required:
        raise ValueError(
            "this GLB requires glTF extensions this viewer does not implement: "
            + ", ".join(sorted(required))
        )
    declared = len(gltf.get("nodes", []))
    if declared > MAX_NODES:
        raise ValueError(f"this GLB declares {declared} nodes, more than this viewer will load")
    # The 2026-09-05 audit, finding clay-05: these two used to be iterated with
    # no ceiling of their own -- an *unreferenced* material or an
    # accessor-less mesh never reaches ``_charge``, so the byte budget never
    # trips no matter how many are declared. See ``MAX_MATERIALS`` above.
    declared_materials = len(gltf.get("materials", []))
    if declared_materials > MAX_MATERIALS:
        raise ValueError(
            f"this GLB declares {declared_materials} materials, more than this viewer will load"
        )
    declared_meshes = len(gltf.get("meshes", []))
    if declared_meshes > MAX_MESHES:
        raise ValueError(
            f"this GLB declares {declared_meshes} meshes, "
            "more than this viewer will load"
        )
    # See MAX_PRIMITIVES: MAX_MESHES bounds mesh *entries*, not how many
    # primitives one of them declares, so this is summed across every mesh
    # before the decode loop below builds one Primitive per entry.
    declared_primitives = sum(
        len(mesh.get("primitives") or []) for mesh in gltf.get("meshes", [])
    )
    if declared_primitives > MAX_PRIMITIVES:
        raise ValueError(
            f"this GLB declares {declared_primitives} primitives, "
            "more than this viewer will load"
        )
    declared_cameras = len(gltf.get("cameras", []))
    if declared_cameras > MAX_CAMERAS:
        raise ValueError(
            f"this GLB declares {declared_cameras} cameras, more than this viewer will load"
        )
    punctual = _punctual(gltf)
    if len(punctual) > MAX_LIGHTS:
        raise ValueError(
            f"this GLB declares {len(punctual)} lights, more than this viewer will load"
        )
    reader = _Reader(gltf, buffer)
    materials = [reader.material(m) for m in gltf.get("materials", [])]
    meshes = [
        [reader.primitive(p, materials) for p in mesh.get("primitives", [])]
        for mesh in gltf.get("meshes", [])
    ]
    skins = [reader.skin(s) for s in gltf.get("skins", [])]
    nodes = [reader.node(n) for n in gltf.get("nodes", [])]
    return Model(
        nodes,
        _roots(gltf, nodes),
        meshes,
        skins,
        skipped_textures=reader.skipped,
        cameras=[reader.camera(c) for c in gltf.get("cameras", [])],
        lights=[reader.light(light) for light in punctual],
    )


def _punctual(gltf: dict) -> list[Any]:
    """The ``KHR_lights_punctual`` light array, or an empty list.

    Read defensively at every level rather than indexed: a file may carry no
    ``extensions`` at all (the ordinary case), an ``extensions`` that is not a
    mapping, or the extension key holding something other than a list -- and
    each of those is a malformed *optional* section, which must cost the file
    its lights and never its geometry.
    """
    extensions = gltf.get("extensions")
    if not isinstance(extensions, dict):
        return []
    block = extensions.get("KHR_lights_punctual")
    if not isinstance(block, dict):
        return []
    lights = block.get("lights")
    return list(lights) if isinstance(lights, list) else []


def _roots(gltf: dict, nodes: list[Node]) -> list[int]:
    """The scene's root nodes, or the ones nobody parents.

    The fallback matters more than it looks. Taking *every* node as a root
    makes ``update_world`` visit each child twice: once correctly under its
    parent, and again later as a root with an identity parent, which overwrites
    the world matrix it just computed. Every node ends up with world == local,
    so anything parented renders in the wrong place and ``joint_palette``
    inverts a matrix that was never right.
    """
    scenes = gltf.get("scenes") or []
    index = gltf.get("scene", 0)
    if scenes:
        # The 2026-09-11 audit, finding create-01 (merged): range-checked
        # just below since create-01 (2026-09-05), but never type-checked --
        # a non-integer "scene" (a string, say) reached the ``<=`` comparison
        # as a bare TypeError instead of this same refusal.
        _check_int_index(index, "the document's scene index")
        if 0 <= index < len(scenes) and "nodes" in scenes[index]:
            roots = scenes[index]["nodes"]
            # Same gap, one field over: a scene's own "nodes" root list is
            # returned straight off the JSON with no check at all, so a
            # non-integer entry reached Model.update_world's bare
            # ``0 <= index < len(self.nodes)`` comparison as a TypeError.
            for root in roots:
                _check_int_index(root, "a scene's root node index")
            return list(roots)
    parented = {child for node in nodes for child in node.children}
    return [i for i in range(len(nodes)) if i not in parented]


def _check_int_index(value: Any, what: str) -> None:
    """Refuse a non-integer index before it reaches ``<=`` or list indexing.

    The 2026-09-09 audit, finding clay-04: every index-shaped field this
    loader reads out of a GLB's JSON was bounds-checked with ``0 <= x < n``
    (create-01, clay-06, clay-09, create2-01 all hardened the out-of-range
    case at these same boundaries) but never checked to *be* an integer, so a
    string, float or list value reached Python's own ``<=``/list-indexing
    operators and raised a bare, un-messaged ``TypeError`` instead of the
    named ``ValueError`` every one of these boundaries otherwise raises.
    ``bool`` is an ``int`` subclass but is never a legitimate index, so it is
    refused here too; a float that happens to be integral (``2.0``) is still
    not a valid glTF index.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{what} must be a whole number, got {value!r}")


def _number(value: Any, default: float) -> float:
    """One optional float off a camera or light entry, or ``default``.

    A camera and a light are *markers*: neither decodes a byte, neither sizes
    an allocation, and a malformed one costs a wrongly-shaped frustum or a
    differently-coloured light rather than a corrupt mesh. So unlike every
    numeric field this loader reads off a node (see :func:`_trs`), a
    non-numeric value here falls back instead of refusing the file -- the same
    trade ``Model.skipped_textures`` already names for an unreadable image,
    which is also a loss that must not take the geometry with it. A non-finite
    value falls back too: an infinite ``yfov`` reaches a projection matrix as
    a frame of NaNs, which is a viewport that draws nothing with no error
    anywhere to say why.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if np.isfinite(number) else default


def _factor(raw: Any, n: int, default: tuple[float, ...]) -> tuple[float, ...]:
    """One fixed-length material factor off a material's JSON, or the glTF default.

    create-01, the 2026-09-14 audit: ``material()`` read ``baseColorFactor``
    and ``emissiveFactor`` with a bare ``tuple(pbr.get(...))`` -- the one
    numeric field in this loader with no shape or numeric-ness check at all,
    unlike a node's TRS (:func:`_trs`, which refuses) or a camera/light field
    (:func:`_number`, which falls back). A wrong-length or non-numeric factor
    loaded the file cleanly and only crashed several frames later, in
    ``GpuMaterial.bind``/``Renderer._draw_model``, when it was written into a
    vec4/vec3 GL uniform -- taking down the whole frame loop with no refusal
    at load time to say why. A material factor is cosmetic like a
    camera/light field, not structural like a node's TRS: losing it costs a
    wrong-looking material, not a corrupt mesh, so this falls back to the
    spec default rather than refusing the file, ``_number``'s trade rather
    than ``_trs``'s.
    """
    try:
        arr = np.asarray(raw, dtype="f8")
    except (TypeError, ValueError):
        return default
    if arr.shape != (n,) or not np.all(np.isfinite(arr)):
        return default
    return tuple(float(x) for x in arr)


def _trs(
    node: dict, name: str, key: str, n: int, default: tuple[float, ...] | None = None
) -> np.ndarray:
    """One fixed-length TRS field off a node's JSON, or a refusal by name.

    The 2026-09-09 audit, finding clay-05: ``node()`` read ``translation``,
    ``rotation``, ``scale`` and ``matrix`` straight off the JSON with no shape
    check at all, unlike every other malformed field in this loader. A
    three-element "rotation" (a quaternion with its ``w`` silently dropped)
    used to reach ``Model.update_world()`` -- called from inside
    ``Model.__init__``, itself inside ``load()`` -- and fail deep in
    ``math3d.compose``/``quat_to_mat4`` with a bare unpacking error rather
    than a refusal naming the node and the field. Mirrors
    ``clay.serialize._vector``, which guards the same shape of field for the
    on-disk ``.wblk`` format.
    """
    raw = node[key] if default is None else node.get(key, default)
    value = np.asarray(raw, dtype="f8")
    if value.shape != (n,):
        raise ValueError(f"node {name!r} has a {key} of {value.size} numbers, not {n}")
    return value


class _Reader:
    def __init__(self, gltf: dict, buffer: bytes) -> None:
        self.gltf = gltf
        self.buffer = buffer
        # Decoded pixels per glTF image *source* index (D39). Several
        # materials routinely reference one atlas, and the PNG decode is the
        # dominant cost of parse_model -- so each image is decoded once and
        # the same (w, h, bytes) tuple is shared, which is also what lets the
        # GPU side de-duplicate its uploads by buffer identity.
        self._images: dict[int, tuple[int, int, bytes] | None] = {}
        # Images this file carries that could not be used (D42). Counted per
        # *source*, not per material reference, because ``_images`` memoizes:
        # one unreadable atlas shared by six primitives is one loss, and
        # counting references would report six.
        self.skipped = 0
        # Decoded accessor arrays, by accessor index (H01). Two or more
        # primitives naming the same accessor -- an instanced mesh's shared
        # POSITION stream is the ordinary case -- used to decode it once per
        # reference; caching makes a repeated reference free instead of a
        # repeated allocation, the same way ``_images`` already does for
        # textures.
        self._accessors: dict[int, np.ndarray] = {}
        # The dtype-converted form of an accessor -- ``positions.astype("f4")``
        # and friends -- shared the same way, and for the same reason (H01):
        # see ``_typed``.
        self._typed_cache: dict[Any, np.ndarray] = {}
        # ``decoded()``'s normalized-integer conversion, by accessor index --
        # shared the same way and for the same reason as ``_typed_cache``
        # (H01, clay-02, 2026-09-15): see ``decoded``.
        self._normalized_cache: dict[int, np.ndarray] = {}
        # Running total this document has allocated, against MAX_TOTAL_BYTES.
        self._spent = 0

    # -- accessors ---------------------------------------------------------

    def _charge(self, nbytes: int) -> None:
        """Add ``nbytes`` to this document's running total, refusing once the
        shared ceiling (H01) is crossed.

        Called *before* the allocation it describes is handed back, not after
        it lands in a caller's hands -- the whole point of a document-wide
        budget is that the refusal happens instead of the bytes, not
        alongside them.
        """
        self._spent += nbytes
        if self._spent > MAX_TOTAL_BYTES:
            raise ValueError(
                f"this GLB's decoded geometry and textures pass the "
                f"{MAX_TOTAL_BYTES:,} byte budget this viewer holds open at once"
            )

    def accessor(self, index: int) -> np.ndarray:
        # clay-04 (2026-09-09): checked before the dict lookup below, which
        # raises a bare ``TypeError`` for a non-hashable value (a list) and
        # otherwise leaves a string or float cached and indexed straight into
        # ``self.gltf["accessors"]`` a few lines down.
        _check_int_index(index, "an accessor reference")
        cached = self._accessors.get(index)
        if cached is not None:
            # Already charged and decoded once; a second primitive naming the
            # same accessor gets the same array rather than a second
            # allocation (H01). Safe to share: nothing downstream mutates an
            # accessor's array in place, only reads or copies it.
            return cached
        out = self._decode_accessor(index)
        self._accessors[index] = out
        return out

    def _accessor_entry(self, index: int, what: str = "an accessor reference") -> dict:
        """The raw ``accessors[index]`` entry, range-checked the same way
        every sibling index-shaped field in this loader is.

        The 2026-09-12 audit, finding clay-01: every other index-shaped field
        this loader reads (node.mesh/skin/child/camera/light, a skin's
        joints, a material's texture/image index, an image's own bufferView,
        the document's scene index) is both type- and range-checked -- but a
        primitive attribute's own accessor index (POSITION/NORMAL/
        TEXCOORD_0/JOINTS_0/WEIGHTS_0/indices, and a skin's
        inverseBindMatrices) was only type-checked (clay-04) here, never
        range-checked, so a negative index silently wrapped through Python's
        own negative-index semantics onto a different, unrelated accessor --
        no error, no log, no toast, just geometry read from the wrong stream
        -- and a positive out-of-range index reached the list indexing below
        as a bare ``IndexError`` instead of the named ``ValueError`` every
        sibling boundary raises and callers key on.
        """
        _check_int_index(index, what)
        accessors = self.gltf.get("accessors", [])
        if not 0 <= index < len(accessors):
            raise ValueError(
                f"{what} is {index}, but this GLB declares {len(accessors)} accessor(s)"
            )
        return accessors[index]

    def _decode_accessor(self, index: int) -> np.ndarray:
        acc = self._accessor_entry(index)
        if "sparse" in acc:
            # Nothing in this pipeline emits one, and silently dropping the
            # overrides would render a subtly wrong mesh rather than fail.
            raise ValueError("sparse accessors are not supported")
        # The 2026-09-07 audit, finding create-09: an unrecognised
        # ``componentType``/``type`` used to raise a bare ``KeyError`` from the
        # dict lookup below -- an internal detail leaking out of a boundary
        # every sibling refusal in this loader states as a named ``ValueError``
        # (``sparse accessors are not supported``, the byte-budget refusal two
        # lines down). A hand-supplied GLB can declare anything in its JSON
        # chunk (``check_glb`` at the import door is structural-only), so this
        # is reachable from ordinary use, not just a corrupt file.
        component_type = acc["componentType"]
        if component_type not in _COMPONENT:
            raise ValueError(f"unsupported accessor componentType {component_type!r}")
        accessor_type = acc["type"]
        if accessor_type not in _NCOMP:
            raise ValueError(f"unsupported accessor type {accessor_type!r}")
        dtype = _COMPONENT[component_type]
        ncomp = _NCOMP[accessor_type]
        count = int(acc["count"])
        # Before the branch and not inside it, so the bound is a property of
        # *reading an accessor* rather than a rule the zeros path below had to
        # remember. The interleaved path pays for it twice over -- ``_check_span``
        # will refuse it again against the real buffer -- and that is the point:
        # the branch that has no buffer to be checked against is the one that
        # was allocating from a number nothing had looked at.
        if count < 0 or count * ncomp * np.dtype(dtype).itemsize > MAX_ACCESSOR_BYTES:
            raise ValueError(
                f"an accessor in this GLB declares {count} {acc['type']} elements,"
                f" which is more than the {MAX_ACCESSOR_BYTES} bytes this viewer"
                " will allocate for one"
            )
        if "bufferView" not in acc:
            # Legal glTF: an accessor with no view reads as zeros -- and the
            # branch H01 was written for: nothing here reads the buffer, so
            # nothing about *this* accessor's cost is bounded by how small the
            # file is. Charged like every other allocation below.
            self._charge(count * ncomp * np.dtype(dtype).itemsize)
            return np.zeros((count, ncomp), dtype=dtype)
        # clay-01 (2026-09-12): an accessor's own ``bufferView`` used to index
        # straight into ``bufferViews`` with no check at all -- despite the
        # comment on the image/bufferView branch below (``_image_bytes``)
        # already, incorrectly, claiming this boundary raised the named
        # refusal. It does now, in the same message shape as that sibling.
        bv = acc["bufferView"]
        _check_int_index(bv, "an accessor's bufferView reference")
        buffer_views = self.gltf.get("bufferViews", [])
        if not 0 <= bv < len(buffer_views):
            raise ValueError(
                f"an accessor references bufferView {bv}, but this GLB "
                f"declares {len(buffer_views)} bufferView(s)"
            )
        view = buffer_views[bv]
        self._check_buffer(view)
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        item = np.dtype(dtype).itemsize * ncomp
        stride = view.get("byteStride") or item
        if stride == item:
            self._check_span(start, count * item)
            self._charge(count * item)
            flat = np.frombuffer(self.buffer, dtype=dtype, count=count * ncomp, offset=start)
            return flat.reshape(count, ncomp)
        # Interleaved: take the raw bytes and gather the rows out. Nothing this
        # pipeline writes is interleaved, but a hand-supplied GLB may be.
        #
        # The span is stride*(count-1) + item, not stride*count: the spec only
        # requires the *elements* to be present, so the padding after the last
        # one need not exist. Demanding it made an accessor sitting at the tail
        # of the BIN chunk raise "buffer is smaller than requested size" -- the
        # model simply failed to load.
        span = stride * (count - 1) + item if count else 0
        self._check_span(start, span)
        # Two real copies land here -- the fancy-indexed ``rows`` and the
        # ``ascontiguousarray`` beneath it -- on top of the raw span this reads
        # out of the buffer, which is exactly the doubling H01's budget exists
        # to account for rather than charge for the final array alone.
        self._charge(span + 2 * count * item)
        raw = np.frombuffer(self.buffer, dtype=np.uint8, count=span, offset=start)
        rows = raw[np.arange(count)[:, None] * stride + np.arange(item)[None, :]]
        return np.ascontiguousarray(rows).view(dtype).reshape(count, ncomp)

    def decoded(self, index: int) -> np.ndarray:
        """An accessor with its ``normalized`` flag honoured.

        The flag says the integers encode a float in 0..1 (or -1..1 signed), so
        reading them raw gives a UV of 65535 or a colour of 255 -- geometry and
        materials that are silently, enormously wrong rather than broken. The
        skin-weight path below already did this by hand for the one case that
        turned up in practice; this is the same rule for every attribute.
        """
        # clay-04 (2026-09-09): this indexes ``accessors`` directly, ahead of
        # ``accessor()``'s own check below, for a primitive's POSITION/NORMAL/
        # TEXCOORD_0/JOINTS_0/WEIGHTS_0 -- reproduced with a string
        # ``attributes.POSITION`` raising a bare ``TypeError`` here.
        # clay-01 (2026-09-12): the same direct index was never range-checked
        # either, so an out-of-range/negative value reached this line before
        # ``accessor()``'s own (equally unchecked, at the time) lookup ever
        # ran. ``_accessor_entry`` now carries both checks for both call sites.
        acc = self._accessor_entry(index)
        raw = self.accessor(index)
        if not acc.get("normalized") or raw.dtype.kind not in "iu":
            return raw
        # clay-02 (2026-09-15 audit): this conversion used to run unconditionally
        # on every call, ahead of ``_typed_cache`` -- so a POSITION or UV accessor
        # shared by several primitives (H01's ordinary case, an instanced mesh)
        # paid for a fresh ``.astype`` plus divide *per primitive* instead of
        # once per accessor, and neither allocation was charged against
        # MAX_TOTAL_BYTES at all, despite the module docstring's "every
        # ``.astype`` is charged". Cached here, by accessor index, the same way
        # ``accessor()`` already caches the raw read one line up -- a second
        # call for the same index is a dict lookup, not a second allocation.
        cached = self._normalized_cache.get(index)
        if cached is not None:
            return cached
        info = np.iinfo(raw.dtype)
        converted = self._astype(raw, "f4")
        if raw.dtype.kind == "u":
            out = converted / float(info.max)
        else:
            # Signed: the spec's own formula, which clamps -128 and -32768 to -1.
            out = np.maximum(converted / float(info.max), -1.0)
        # The division (and, for signed, the ``maximum`` on top of it) builds a
        # second same-size array beyond the one ``_astype`` just charged --
        # the same doubling create-05 (2026-09-13) found and charged in the
        # WEIGHTS_0 path below.
        self._charge(out.nbytes)
        self._normalized_cache[index] = out
        return out

    def _check_buffer(self, view: dict) -> None:
        """Refuse a view that points at a buffer we do not have.

        ``read_glb`` returns the GLB's single BIN chunk, which is buffer 0. A
        view naming buffer 1 (a data URI, or an external .bin) was read at the
        same offsets *into buffer 0* -- silently wrong geometry rather than an
        error, which is the worst way for this to fail.
        """
        if view.get("buffer", 0) != 0:
            raise ValueError(
                "this GLB stores geometry outside its binary chunk, which is not supported"
            )

    def _check_span(self, start: int, span: int) -> None:
        if start < 0 or start + span > len(self.buffer):
            raise ValueError("an accessor reads past the end of the binary chunk")

    def _astype(self, arr: np.ndarray, dtype: str) -> np.ndarray:
        """``arr.astype(dtype)``, charged against the document budget.

        ``astype`` copies even when the requested dtype already matches, so
        every conversion below -- positions to ``f4``, indices to ``u4``, the
        weight/joint promotions -- is a *second* array on top of whatever
        ``accessor``/``decoded`` already charged for the first one (H01).
        Charging only the final accessor size, as the per-accessor ceiling
        does, undercounted exactly this doubling.
        """
        out = arr.astype(dtype)
        self._charge(out.nbytes)
        return out

    def _typed(self, key: Any, raw: np.ndarray, dtype: str) -> np.ndarray:
        """A converted array, shared by every primitive asking for the same
        ``key`` (H01).

        Caching ``accessor()``'s *raw* output already stops a repeated
        reference from decoding twice; this closes the gap one layer up.
        Every attribute here still runs its own ``.astype`` per primitive
        even once the raw accessor is cached, because each call built a fresh
        copy of the *converted* array too -- an instanced mesh with a
        thousand nodes sharing one glTF mesh definition paid for a thousand
        ``positions.astype("f4")`` copies of an identical result. Safe to
        share: nothing downstream mutates a primitive's positions, indices,
        normals, uvs or joints in place (unlike weights, renormalised right
        below -- deliberately left out of this cache).
        """
        cached = self._typed_cache.get(key)
        if cached is not None:
            return cached
        out = self._astype(raw, dtype)
        self._typed_cache[key] = out
        return out

    # -- pieces ------------------------------------------------------------

    def primitive(self, prim: dict, materials: list[Material]) -> Primitive:
        attrs = prim.get("attributes", {})
        if "POSITION" not in attrs:
            raise ValueError("a primitive with no POSITION is not renderable")
        if prim.get("mode", 4) != 4:
            raise ValueError(f"unsupported primitive mode {prim.get('mode')}")
        positions = self._typed(
            ("POSITION", attrs["POSITION"]), self.decoded(attrs["POSITION"]), "f4"
        )
        if "indices" in prim:
            # Indices are never normalized -- they are indices -- so they take
            # the raw path deliberately.
            indices = self._typed(
                ("indices", prim["indices"]),
                self.accessor(prim["indices"]).reshape(-1),
                "u4",
            )
            # The 2026-09-08 audit, finding create-01: a TRIANGLES primitive
            # (the only mode this loader accepts) whose index count is not a
            # multiple of 3 used to load clean and only blow up later, out of
            # ``scene._face_normals``'s ``indices.reshape(-1, 3)`` -- after
            # GpuModel.__init__ had already allocated real ctx.buffer objects
            # for any earlier primitives in the model, which then leak (this
            # app sets no moderngl gc_mode). Refusing here means those buffers
            # are never allocated for this file at all.
            if len(indices) % 3 != 0:
                raise ValueError(
                    f"a TRIANGLES primitive needs an index count that is a "
                    f"multiple of 3, got {len(indices)}"
                )
        else:
            # Synthesised rather than read off the buffer, but no less real an
            # allocation -- and the one H01 names explicitly: a primitive with
            # no ``indices`` used to get this array for free regardless of how
            # large ``positions`` was. Cached on the length rather than an
            # accessor index -- there is no accessor to key on -- which still
            # dedupes the ordinary case of several instances of one unindexed
            # mesh.
            # Same guard as the indexed branch above, create-01: an unindexed
            # TRIANGLES primitive's vertex count stands in for the index
            # count, and must be a multiple of 3 for the same reason.
            if len(positions) % 3 != 0:
                raise ValueError(
                    f"a TRIANGLES primitive needs a vertex count that is a "
                    f"multiple of 3, got {len(positions)}"
                )
            key = ("arange", len(positions))
            indices = self._typed_cache.get(key)
            if indices is None:
                indices = np.arange(len(positions), dtype="u4")
                self._charge(indices.nbytes)
                self._typed_cache[key] = indices
        out = Primitive(positions=positions, indices=indices)
        if "NORMAL" in attrs:
            out.normals = self._typed(
                ("NORMAL", attrs["NORMAL"]), self.decoded(attrs["NORMAL"]), "f4"
            )
        if "TEXCOORD_0" in attrs:
            out.uvs = self._typed(
                ("TEXCOORD_0", attrs["TEXCOORD_0"]), self.decoded(attrs["TEXCOORD_0"]), "f4"
            )
        if "JOINTS_0" in attrs:
            out.joints = self._typed(
                ("JOINTS_0", attrs["JOINTS_0"]), self.accessor(attrs["JOINTS_0"]), "i4"
            )
        if "WEIGHTS_0" in attrs:
            raw = self.accessor(attrs["WEIGHTS_0"])
            # glTF allows weights as normalized ubyte/ushort as well as float.
            # Reading the integer forms as-is would give every vertex a weight
            # of 65535, which renders as an explosion rather than as a mesh.
            # The 2026-09-13 audit, finding create-05: ``/ 255.0`` and
            # ``/ 65535.0`` each build a fresh float32 array the same size as
            # the one ``_astype`` just charged, and the renormalising
            # division a few lines down builds a third -- none of them
            # charged, so a skinned mesh's peak allocation was undercounted
            # against MAX_TOTAL_BYTES by up to 2x its weights array alone.
            if raw.dtype == np.uint8:
                weights = self._astype(raw, "f4") / 255.0
                self._charge(weights.nbytes)
            elif raw.dtype == np.uint16:
                weights = self._astype(raw, "f4") / 65535.0
                self._charge(weights.nbytes)
            else:
                weights = self._astype(raw, "f4")
            # **Renormalised, and a zero-sum vertex pinned to its first joint.**
            # The shader sums ``u_joints[j] * w`` over the four influences with
            # no division, so the spec's "the weights of a vertex sum to 1" is
            # a requirement this renderer *relies* on rather than one it
            # checks. A file whose weights sum to 0 -- a stray vertex an
            # exporter left unweighted, which is common enough in rigs that
            # come back from a round trip -- collapsed every such vertex onto
            # the origin, and the mesh grew a spike to the world centre.
            # Renormalising is exact for a well-formed file (a sum of 1 divides
            # by 1) and is the only reading available for a malformed one.
            total = weights.sum(axis=1, keepdims=True)
            dead = total[:, 0] <= 0.0
            if dead.any():
                weights[dead] = 0.0
                weights[dead, 0] = 1.0
                total = weights.sum(axis=1, keepdims=True)
            out.weights = weights / total
            # create-05 again: the renormalising division itself, run for
            # every vertex regardless of whether the dead-vertex branch above
            # fired -- the second (or third) uncharged same-size copy.
            self._charge(out.weights.nbytes)
        # The 2026-09-06 audit, finding clay-06: this only checked the upper
        # bound, so ``"material": -1`` resolved through Python's own
        # negative-index wraparound to the *last* palette entry instead of
        # falling back to the default -- no error, no log, no toast. The
        # sibling reader, ``clay.document._material_at``, already got this
        # right with an explicit ``0 <= index`` check; this is the same rule.
        # clay-04 (2026-09-09): a non-integer "material" (a string, say) hit
        # the same ``<=`` operator clay-06 already covers for the negative
        # case, as a bare ``TypeError`` rather than the silent fallback an
        # out-of-range value gets. Extended, not raised: this boundary's own
        # rule (clay-06, just above) is that a bad material index is a
        # cosmetic loss -- the default material -- never a refused load, and
        # a wrong-*type* index is no different a defect than a wrong-range
        # one for that purpose.
        material_index = prim.get("material")
        if (
            isinstance(material_index, int)
            and not isinstance(material_index, bool)
            and 0 <= material_index < len(materials)
        ):
            out.material = materials[material_index]
        return out

    def material(self, mat: dict) -> Material:
        pbr = mat.get("pbrMetallicRoughness", {})
        out = Material(
            name=mat.get("name", ""),
            base_color_factor=_factor(
                pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0)), 4, (1.0, 1.0, 1.0, 1.0)
            ),
            metallic_factor=float(pbr.get("metallicFactor", 1.0)),
            roughness_factor=float(pbr.get("roughnessFactor", 1.0)),
            emissive_factor=_factor(
                mat.get("emissiveFactor", (0.0, 0.0, 0.0)), 3, (0.0, 0.0, 0.0)
            ),
            double_sided=bool(mat.get("doubleSided", False)),
            alpha_mode=mat.get("alphaMode", "OPAQUE"),
            alpha_cutoff=float(mat.get("alphaCutoff", 0.5)),
        )
        out.base_color = self.texture(pbr.get("baseColorTexture"))
        out.metallic_roughness = self.texture(pbr.get("metallicRoughnessTexture"))
        out.normal = self.texture(mat.get("normalTexture"))
        out.emissive = self.texture(mat.get("emissiveTexture"))
        out.occlusion = self.texture(mat.get("occlusionTexture"))
        return out

    def _image_bytes(self, image: dict) -> bytes | None:
        """The encoded pixels for one glTF image, or None if unreachable.

        Three cases, and the old code collapsed them into one warning that was
        false for two of them. A bufferView is the normal path. A ``data:`` URI
        is equally self-contained -- it is *inside* the file, it just is not in
        the binary chunk -- and dropping it lost a texture the asset carried
        while reporting it as "stored outside the GLB". Only a genuine external
        URI is a runtime file read of a path from inside the asset, which is
        the thing actually worth refusing.

        Every unreachable case *returns* here rather than raising, unlike the
        geometry path. A mesh read from the wrong buffer is a plausible-looking
        wrong answer and worth failing the load over; a texture that cannot be
        found is a cosmetic loss, and raising turned any third-party GLB whose
        images sit in a second buffer into "this file will not open".
        """
        if "bufferView" in image:
            # The 2026-09-06 audit, finding clay-09: indexing straight into
            # ``bufferViews`` raised a bare ``IndexError`` for an out-of-range
            # index, while every sibling boundary in this file (node.mesh,
            # node.skin, skin.joints, an accessor's own bufferView, the buffer
            # index) raises the named ``ValueError`` this module's callers key
            # on. Unlike the *reachability* check just below -- which stays a
            # cosmetic skip, because a buffer genuinely too small for a view is
            # not the same class of bug as a view that names no bufferView at
            # all -- a bad index is refused with the same message shape
            # ``node()``/``skin()`` use.
            buffer_views = self.gltf.get("bufferViews", [])
            bv = image["bufferView"]
            # The 2026-09-11 audit, finding create-01 (merged): range-checked
            # just below since clay-09, but never type-checked, the same gap
            # as the texture/image index checks a few lines below this one.
            _check_int_index(bv, "an image's bufferView reference")
            if not 0 <= bv < len(buffer_views):
                raise ValueError(
                    f"a texture references bufferView {bv}, but this GLB "
                    f"declares {len(buffer_views)} bufferView(s)"
                )
            view = buffer_views[bv]
            try:
                self._check_buffer(view)
            except Exception as exc:
                log.warning("skipping a texture in an unreachable buffer: %s", exc)
                self.skipped += 1
                return None
            start = view.get("byteOffset", 0)
            byte_length = view.get("byteLength", 0)
            # The 2026-09-13 audit, finding create-09: this used to slice
            # straight out of ``self.buffer`` with no span check, so a
            # bufferView whose declared byteLength overran the BIN chunk was
            # silently *truncated* -- a corrupt-looking image that then
            # failed to decode -- rather than refused at the boundary the
            # way an accessor's own span is (``_check_span``). Same
            # reachability class as the buffer check just above: a texture
            # is a cosmetic loss, so this stays a skip rather than a raise,
            # using the accessor path's own span logic so the two boundaries
            # cannot drift apart.
            try:
                self._check_span(start, byte_length)
            except Exception as exc:
                log.warning(
                    "skipping a texture whose bufferView reads past the end "
                    "of the binary chunk: %s",
                    exc,
                )
                self.skipped += 1
                return None
            return self.buffer[start : start + byte_length]
        uri = str(image.get("uri") or "")
        if uri.startswith("data:"):
            import base64

            _, _, payload = uri.partition(",")
            try:
                return base64.b64decode(payload)
            except Exception:
                log.warning("a texture's embedded data URI could not be decoded")
                self.skipped += 1
                return None
        log.warning("skipping a texture stored in a separate file (%s)", uri or "no uri")
        self.skipped += 1
        return None

    def texture(self, ref: dict | None) -> tuple[int, int, bytes] | None:
        if not ref:
            return None
        from PIL import Image

        # The 2026-09-06 audit, finding clay-09: both lookups below indexed
        # straight into the file's own arrays with no bounds check, so a
        # material naming an out-of-range (or negative) texture/image index
        # raised a bare IndexError instead of the named ValueError every other
        # boundary in this file raises. Refused here, before either array is
        # touched, in the same message shape ``node()``/``skin()`` use.
        textures = self.gltf.get("textures", [])
        index = ref["index"]
        # The 2026-09-09 audit, finding clay-04: checked for range just below
        # since clay-09, but never for type -- a string/float/list "index"
        # reached the ``<=`` comparison as a bare TypeError.
        _check_int_index(index, "a material's texture reference")
        if not 0 <= index < len(textures):
            raise ValueError(
                f"a material references texture {index}, but this GLB "
                f"declares {len(textures)} texture(s)"
            )
        tex = textures[index]
        # The 2026-09-08 audit's clay-04: a textures[] entry with no "source"
        # key is legal per the glTF 2.0 schema (a texture may carry only a
        # sampler), and indexing straight into it used to raise a bare
        # KeyError -- the one shape of malformed reference this file's other
        # boundaries (node.mesh, node.skin, skin.joints, this same function's
        # own texture/image bounds checks, prim["material"]) had already been
        # hardened against. Refused here in the same message shape.
        if "source" not in tex:
            raise ValueError(f"texture {index} has no source image")
        source = tex["source"]
        # The 2026-09-09 audit, finding clay-04: same gap as the texture
        # index just above -- range-checked (clay-09) but not type-checked,
        # and a non-hashable value (a list) would also fail the ``self.
        # _images`` cache lookup a few lines down before ever reaching it.
        _check_int_index(source, f"texture {index}'s source image reference")
        images = self.gltf.get("images", [])
        if not 0 <= source < len(images):
            raise ValueError(
                f"a material references image {source}, but this GLB "
                f"declares {len(images)} image(s)"
            )
        if source in self._images:
            return self._images[source]
        image = images[source]
        data = self._image_bytes(image)
        if data is None:
            self._images[source] = None
            return None
        try:
            with Image.open(io.BytesIO(data)) as im:
                # ``open`` is lazy, so the size is known before any pixel is
                # decoded: an absurd one costs a log line rather than the RAM.
                # Same policy as a corrupt map -- the texture, not the model.
                if im.width * im.height > MAX_TEXTURE_PIXELS:
                    log.warning(
                        "skipping a %dx%d texture: over this viewer's %d-pixel ceiling",
                        im.width,
                        im.height,
                        MAX_TEXTURE_PIXELS,
                    )
                    self.skipped += 1
                    self._images[source] = None
                    return None
                # Charged into the *document-wide* budget (H01) before the
                # decode it describes, and deliberately outside the
                # cosmetic-loss ``except`` below: an over-budget document is a
                # refusal, unlike one corrupt map, because this is bounding
                # the sum of every texture the file carries rather than
                # judging any one of them.
                self._charge(im.width * im.height * 4)
                rgba = im.convert("RGBA")
                decoded = rgba.width, rgba.height, rgba.tobytes()
        except ValueError:
            raise
        except Exception:
            # The stated policy for images, applied to the decode as well as to
            # the lookup: a texture that cannot be read is a cosmetic loss, and
            # raising here turned one corrupt map into "this file will not
            # open" -- for a mesh that is otherwise entirely intact.
            log.warning("skipping a texture whose image data could not be decoded")
            self.skipped += 1
            decoded = None
        self._images[source] = decoded
        return decoded

    def skin(self, skin: dict) -> Skin:
        # The 2026-09-15 audit, finding clay-03: ``skin["joints"]`` indexed
        # straight into the JSON with no check, so a skin entry missing the
        # required ``joints`` array (glTF requires it, but ``check_glb`` at
        # the import door is structural-only, same gap create-09 and clay-04
        # were found through) raised a bare ``KeyError`` instead of this
        # loader's own named refusal every sibling boundary raises.
        if "joints" not in skin:
            raise ValueError("a skin with no \"joints\" array is not supported")
        joints = list(skin["joints"])
        # Bounds-checked the same way ``node()`` checks ``node.mesh``/
        # ``node.skin`` against the file's own declared counts: the
        # 2026-09-06 audit, finding create2-01, found that a skin's *own*
        # ``joints`` array was never checked against how many nodes the file
        # declares, even though ``node.skin`` itself was in range. Such a file
        # loaded clean and only raised a bare ``IndexError`` later, out of
        # ``Model.joint_palette`` -- called from ``GpuModel.__init__``
        # (scene.py) after GPU buffers/textures for every earlier node were
        # already allocated. Refusing here, before any ``Skin`` is returned,
        # keeps that refusal on the load-time side of the same line
        # ``node()`` already draws.
        n_nodes = len(self.gltf.get("nodes", []))
        for joint in joints:
            # The 2026-09-09 audit, finding clay-04: range-checked just below
            # since create2-01, but never type-checked, so a non-integer
            # joint entry hit the ``<=`` comparison as a bare TypeError.
            _check_int_index(joint, "a skin's joint (node) index")
            if not 0 <= joint < n_nodes:
                raise ValueError(
                    f"a skin references joint (node) index {joint}, but this "
                    f"GLB declares {n_nodes} node(s)"
                )
        if "inverseBindMatrices" in skin:
            # glTF stores each matrix column-major; ours are M @ v with the
            # translation in the last column, so every one is transposed here
            # and nowhere else.
            raw = self._astype(self.accessor(skin["inverseBindMatrices"]), "f8")
            ibm = raw.reshape(-1, 4, 4).transpose(0, 2, 1)
            # Same shape of bug, one line over: ``joint_palette`` zips
            # ``skin.joints`` against ``skin.inverse_bind`` by position
            # (``enumerate(skin.joints)`` indexing ``inverse_bind[i]``), so an
            # accessor with fewer matrices than joints is the same
            # load-clean-crash-later failure as an out-of-range joint index,
            # just discovered in the same 2026-09-06 audit pass (create2-01).
            if len(ibm) != len(joints):
                raise ValueError(
                    f"a skin declares {len(joints)} joint(s) but "
                    f"{len(ibm)} inverse bind matrices"
                )
        else:
            ibm = np.tile(np.eye(4), (len(joints), 1, 1))
        return Skin(joints=joints, inverse_bind=ibm)

    # -- cameras and lights ------------------------------------------------

    def camera(self, entry: Any) -> Camera:
        """One ``cameras`` entry. Orthographic is read as a perspective camera
        carrying this module's defaults rather than refused: a camera is a
        marker, not geometry, so a kind this build does not model costs the
        file one wrongly-shaped frustum and never the scene around it.
        """
        if not isinstance(entry, dict):
            return Camera()
        persp = entry.get("perspective")
        persp = persp if isinstance(persp, dict) else {}
        return Camera(
            name=str(entry.get("name", "")),
            yfov=_number(persp.get("yfov"), Camera.yfov),
            znear=_number(persp.get("znear"), Camera.znear),
            zfar=_number(persp.get("zfar"), 0.0),
            aspect_ratio=_number(persp.get("aspectRatio"), 0.0),
        )

    def light(self, entry: Any) -> Light:
        """One ``KHR_lights_punctual`` light.

        An unrecognised ``type`` falls back to ``point`` rather than raising,
        for :meth:`camera`'s reason and one more: the extension is explicitly
        extensible, so a fourth kind added to it later must not make a file
        carrying one unopenable.
        """
        if not isinstance(entry, dict):
            return Light()
        kind = entry.get("type")
        spot = entry.get("spot")
        spot = spot if isinstance(spot, dict) else {}
        color = entry.get("color")
        if not (isinstance(color, (list, tuple)) and len(color) == 3):
            color = Light.color
        return Light(
            name=str(entry.get("name", "")),
            kind=kind if kind in LIGHT_KINDS else "point",
            color=tuple(_number(c, 1.0) for c in color),  # type: ignore[arg-type]
            intensity=_number(entry.get("intensity"), Light.intensity),
            range=_number(entry.get("range"), 0.0),
            inner_cone_angle=_number(spot.get("innerConeAngle"), Light.inner_cone_angle),
            outer_cone_angle=_number(spot.get("outerConeAngle"), Light.outer_cone_angle),
        )

    def node(self, node: dict) -> Node:
        # Bounds-checked against the file's own declared counts, not decoded
        # length -- both are read from ``self.gltf`` before any node is built,
        # so this holds regardless of load()'s decode order. The 2026-09-05
        # audit, finding create-01: an out-of-range mesh/skin index used to
        # reach ``GpuModel.__init__`` (scene.py) as a bare ``IndexError`` --
        # *after* real ``ctx.buffer``/``ctx.texture`` objects had already been
        # allocated for earlier, valid nodes in the same document, which
        # leaked them forever (this app sets no moderngl ``gc_mode``: a
        # dropped reference frees nothing). Refusing here, on the task thread
        # inside ``load()``, means no GPU resource is ever created for a file
        # that will not finish loading -- the same ceiling ``prim["material"]``
        # already gets a few lines below.
        name = node.get("name", "") or "<unnamed>"
        mesh = node.get("mesh")
        if mesh is not None:
            # The 2026-09-09 audit, finding clay-04: range-checked just below
            # since create-01, but never type-checked, so a non-integer
            # "mesh" (a string, say) hit the ``<=`` comparison as a bare
            # TypeError instead of this same refusal.
            _check_int_index(mesh, f"node {name!r}'s mesh reference")
            n_meshes = len(self.gltf.get("meshes", []))
            if not 0 <= mesh < n_meshes:
                raise ValueError(
                    f"node {name!r} references mesh "
                    f"{mesh}, but this GLB declares {n_meshes} mesh(es)"
                )
        skin = node.get("skin")
        if skin is not None:
            _check_int_index(skin, f"node {name!r}'s skin reference")
            n_skins = len(self.gltf.get("skins", []))
            if not 0 <= skin < n_skins:
                raise ValueError(
                    f"node {name!r} references skin "
                    f"{skin}, but this GLB declares {n_skins} skin(s)"
                )
        children = list(node.get("children", []))
        # The 2026-09-11 audit, finding create-01 (merged): a node's own
        # "children" array was stored here with no validation at all, so a
        # non-integer entry survived load() and reached Model.update_world's
        # bare ``0 <= index < len(self.nodes)`` comparison as a TypeError
        # instead of the named refusal every sibling index-shaped field in
        # this loader gives. Out-of-range entries stay unchecked here on
        # purpose -- forward references are legal glTF, and update_world's
        # own ``seen``/range check already skips one that names no node
        # (test_a_child_index_that_names_no_node_is_skipped_not_raised).
        for child in children:
            _check_int_index(child, f"node {name!r}'s child reference")
        camera = node.get("camera")
        if camera is not None:
            _check_int_index(camera, f"node {name!r}'s camera reference")
            n_cameras = len(self.gltf.get("cameras", []))
            if not 0 <= camera < n_cameras:
                raise ValueError(
                    f"node {name!r} references camera "
                    f"{camera}, but this GLB declares {n_cameras} camera(s)"
                )
        # The extension hangs its index off the node's own ``extensions``
        # block rather than off a top-level field, which is the one structural
        # difference between a light and a camera here.
        node_ext = node.get("extensions")
        light = None
        if isinstance(node_ext, dict):
            block = node_ext.get("KHR_lights_punctual")
            if isinstance(block, dict):
                light = block.get("light")
        if light is not None:
            _check_int_index(light, f"node {name!r}'s light reference")
            n_lights = len(_punctual(self.gltf))
            if not 0 <= light < n_lights:
                raise ValueError(
                    f"node {name!r} references light "
                    f"{light}, but this GLB declares {n_lights} light(s)"
                )
        out = Node(
            name=node.get("name", ""),
            children=children,
            mesh=mesh,
            skin=skin,
            camera=camera,
            light=light,
        )
        # The 2026-09-09 audit, finding clay-05: none of these four fields
        # was shape-checked at all -- see ``_trs`` above for what that let
        # through.
        if "matrix" in node:
            # A node gives either a matrix or TRS, never both.
            matrix = _trs(node, name, "matrix", 16)
            mat = matrix.reshape(4, 4).T
            out.translation, out.rotation, out.scale = m3.decompose(mat)
        else:
            out.translation = _trs(node, name, "translation", 3, (0.0, 0.0, 0.0))
            out.rotation = _trs(node, name, "rotation", 4, (0.0, 0.0, 0.0, 1.0))
            out.scale = _trs(node, name, "scale", 3, (1.0, 1.0, 1.0))
        return out
