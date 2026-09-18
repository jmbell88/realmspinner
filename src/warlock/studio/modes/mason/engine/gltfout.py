"""A scene as glTF: :class:`~.document.MasonDoc` to a :class:`gltf.Model`, then bytes.

**This is ``scene.walk`` accumulating structure, and it is not a second
traversal.** ``scene.py``'s own docstring says why the resolver takes a visit
callback at all: ``resolve`` flattens the tree into world matrices because a
viewport wants a draw list, and a glTF export wants the opposite -- the parents
kept, the transforms local, the hierarchy the user built arriving in the engine
as the hierarchy the user built. Both are the same walk with the same
visibility rule, the same material-override rule and the same prefab expansion,
which is the entire reason "the viewport and the export can never disagree" is
a true sentence rather than a hope. A recursive descent in this file would make
it a hope.

**Instancing is the file format's own.** Two mesh nodes that name the same
reference share one glTF mesh index -- ``viewer/glbwrite.py`` writes the
accessors once and both nodes point at them -- so five hundred placements of a
barrel are one barrel's worth of geometry in the file. Nothing in the document
says so; it falls out of keying the mesh table on ``ref_key`` exactly as the
GPU cache does, which is what makes the file and the GPU agree about what is
shared.

**The mesh key carries the material override alongside the ref**, because glTF
puts the material on the primitive rather than on the node: a retinted copy of
a shared asset is a second mesh in the file whether we like it or not. That is
the same cost the Mason programme left open for the GPU side -- a material
override buys a second upload of identical geometry, because the override is
in the cache key -- and it is stated here rather than discovered later.

**Names are made unique on the way out.** Every importer's find-by-name assumes
it, the engine manifest addresses nodes by it, and a Mason document cheerfully
holds six nodes called "Rock" because a scene is a place and places have six
rocks in them. The uniquified name is handed back on :class:`ExportedNode`
rather than recomputed by whoever wants it next, so ``manifest.py`` names the
same node this file named -- by construction, not by running the same rule
twice and hoping.

**Units and handedness: metres, Y-up, right-handed.** That is glTF's own
convention and this project now states it outright (``dev/INVARIANTS.md``);
nothing here converts anything, which is precisely what the statement buys.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .....kernels.geom3d import glbwrite, gltf
from . import scene as sc
from . import terrain as tr
from .nodes import CameraNode, GroupNode, LightNode, MeshNode, Node, PrefabNode, TerrainNode
from .refs import GeometrySource, Ref, ref_key

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import MasonDoc

__all__ = [
    "DEFAULT_NAMES",
    "ExportedNode",
    "SceneExport",
    "scene_glb",
    "scene_model",
    "unique_name",
]

#: What a node with no name of its own is called in the file. A glTF node may
#: legitimately carry no name at all, and this writes one anyway: the manifest
#: addresses nodes by name, so an unnamed node would be a node the manifest
#: cannot talk about -- and "" is not a name six of them can each have a unique
#: version of.
DEFAULT_NAMES = {
    "group": "Group",
    "mesh": "Mesh",
    "light": "Light",
    "camera": "Camera",
    "prefab": "Instance",
    "terrain": "Terrain",
}


def kind_of(node: Node) -> str:
    """The one-word kind string this module and the manifest both key on.

    Derived from the node's class rather than from a field, because
    ``nodes.py`` deliberately chose six classes over one class with a
    discriminator -- and a discriminator reinvented here would be exactly the
    ``if kind ==`` chain that decision was taken to avoid.
    """
    if isinstance(node, MeshNode):
        return "mesh"
    if isinstance(node, LightNode):
        return "light"
    if isinstance(node, CameraNode):
        return "camera"
    if isinstance(node, PrefabNode):
        return "prefab"
    if isinstance(node, TerrainNode):
        return "terrain"
    if isinstance(node, GroupNode):
        return "group"
    return "group"  # pragma: no cover - the node hierarchy is closed


def unique_name(base: str, taken: set[str]) -> str:
    """``base``, or the first ``base.001``-style variant nothing has claimed.

    Blender's suffix rather than an invented one, because it is the convention
    every DCC tool and every importer's user already reads as "the same thing
    again". The loop re-checks membership on every candidate instead of
    counting occurrences: a document that already contains a node literally
    named ``Rock.001`` alongside two called ``Rock`` would otherwise have the
    uniquifier generate the name that is already taken, which is the one thing
    it exists not to do.
    """
    if base not in taken:
        taken.add(base)
        return base
    index = 1
    while True:
        candidate = f"{base}.{index:03d}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate
        index += 1


@dataclass(frozen=True)
class ExportedNode:
    """One node as it was written, and everything the manifest needs about it.

    Frozen, and handed back rather than recomputed, for
    :class:`~.scene.Placed`'s reason one layer up: this is the record that
    makes "the manifest and the GLB name the same nodes" true by construction.
    A manifest that walked the document itself would be a second traversal
    applying the same naming rule a second time, and the day the two rules
    drifted the symptom would be an engine import script silently finding
    nothing.
    """

    #: Index into ``SceneExport.model.nodes``.
    index: int
    #: The unique name written into the file.
    name: str
    kind: str
    #: Uids root-first, exactly ``scene.Placed.path``.
    path: tuple[int, ...]
    #: The scene-tree node this belongs to -- the prefab instance, for
    #: anything inside one. ``scene.Placed.owner``.
    owner: int
    #: The Mason node itself; the *template's* object for prefab content.
    node: Node
    world: np.ndarray
    ref: Ref | None
    material: gltf.Material | None
    static: bool
    locked: bool
    #: The template this node was reached through, "" when authored directly.
    prefab: str
    dangling: bool


@dataclass(frozen=True)
class SceneExport:
    """The model, plus the per-node record the manifest is written from."""

    model: gltf.Model
    nodes: tuple[ExportedNode, ...]
    #: References that resolved to no geometry at all -- a dangling library
    #: link, or an asset whose background parse has not finished. Reported
    #: rather than silently written as an empty node, which is
    #: ``Model.skipped_textures``' precedent: a loss that is stated is not the
    #: same as a loss that is invisible.
    unresolved: tuple[tuple[int, Ref], ...]


def scene_model(
    doc: MasonDoc, source: GeometrySource, *, include_hidden: bool = False
) -> SceneExport:
    """The document as a glTF model, through one ``scene.walk``.

    One walk, two accumulators: ``visit`` for every node the walk crosses and
    ``enter`` for a prefab instance the moment it is crossed into (see
    ``scene.walk``'s own docstring for why that hook exists and what the
    alternative was). Each callback appends exactly one glTF node and hangs it
    under whatever ``path[:-1]`` already produced, so the parent of a node in
    the file is the parent of the node in the outliner -- with no recursion
    here at all.
    """
    builder = _Builder(doc, source)
    sc.walk(doc, builder.visit, include_hidden=include_hidden, enter=builder.visit)
    return builder.finish()


def scene_glb(
    doc: MasonDoc, source: GeometrySource, *, include_hidden: bool = False
) -> bytes:
    """The scene as the bytes of a self-contained binary glTF.

    Through the shared ``viewer/glbwrite.py`` rather than a writer of Mason's
    own, which is the plan's one change to a file Clay, Poser and Troupe all
    depend on: there is one GLB writer in this project and the loader beside
    it is the only thing that can test it. A second writer here would be a
    second home for the four things ``glbwrite``'s own docstring names as easy
    to get wrong -- the POSITION accessor's ``min``/``max``, four-byte view
    alignment, the index component width matching the bytes, and the two
    chunks' different padding -- with no loader to round-trip against.
    """
    return glbwrite.write_glb(scene_model(doc, source, include_hidden=include_hidden).model)


class _Builder:
    """The accumulator both walk callbacks append into.

    A class rather than two closures over a pile of locals because the two
    callbacks share seven pieces of state and the shared state *is* the
    substance here -- the mesh table, the name set and the path-to-index map
    are what make instancing, uniqueness and parenting work at all.
    """

    def __init__(self, doc: MasonDoc, source: GeometrySource) -> None:
        self.doc = doc
        self.source = source
        self.nodes: list[gltf.Node] = []
        self.roots: list[int] = []
        self.meshes: list[list[gltf.Primitive]] = []
        self.cameras: list[gltf.Camera] = []
        self.lights: list[gltf.Light] = []
        self.records: list[ExportedNode] = []
        self.unresolved: list[tuple[int, Ref]] = []
        #: ``(ref_key, id(material)) -> mesh index``. The ref half is what
        #: makes six hundred barrels one mesh; the material half is there
        #: because glTF hangs a material off the primitive, so an override is
        #: a different mesh whether or not it is different geometry.
        self._mesh_index: dict[tuple[Any, ...], int] = {}
        #: Every material an ``id()`` above is keyed on, held alive for the
        #: length of the export. An id is an address CPython reissues the
        #: moment its object is collected, and a reissued one would fold two
        #: different overrides into one mesh -- the same requirement
        #: ``glbwrite._Writer._material_keep`` makes explicit for the same
        #: reason rather than resting it on a caller's object graph.
        self._material_keep: list[gltf.Material] = []
        self._taken: set[str] = set()
        #: Path tuple -> index into ``self.nodes``. The parent lookup.
        self._by_path: dict[tuple[int, ...], int] = {}

    # -- the one callback both halves of the walk use ----------------------

    def visit(
        self,
        node: Node,
        path: tuple[int, ...],
        owner: int,
        world: np.ndarray,
        visible: bool,
        locked: bool,
        static: bool,
        ref: Ref | None,
        material: gltf.Material | None,
        prefab: str,
        dangling: bool,
    ) -> None:
        kind = kind_of(node)
        index = len(self.nodes)
        name = unique_name(node.name.strip() or DEFAULT_NAMES[kind], self._taken)
        # The node's **own** TRS, not the composed world matrix: the parent
        # chain in the file reproduces the composition, and baking it in
        # would export a scene whose every node is a root wearing its
        # ancestors' transforms -- which is what a flattening exporter does
        # and what this one exists not to do.
        out = gltf.Node(
            name=name,
            translation=np.array(node.translation, dtype="f8", copy=True),
            rotation=np.array(node.rotation, dtype="f8", copy=True),
            scale=np.array(node.scale, dtype="f8", copy=True),
        )
        self.nodes.append(out)
        self._by_path[path] = index

        parent = self._by_path.get(path[:-1])
        if parent is None:
            self.roots.append(index)
        else:
            self.nodes[parent].children.append(index)

        if kind == "mesh" and ref is not None:
            mesh = self._mesh_for(ref, material)
            if mesh is None:
                self.unresolved.append((index, ref))
            else:
                out.mesh = mesh
        elif kind == "terrain" and self.doc.terrain is not None:
            out.mesh = self._terrain_mesh(material)
        elif kind == "light":
            out.light = self._light(node, name)
        elif kind == "camera":
            out.camera = self._camera(node, name)

        self.records.append(
            ExportedNode(
                index=index,
                name=name,
                kind=kind,
                path=path,
                owner=owner,
                node=node,
                world=np.array(world, dtype="f8", copy=True),
                ref=ref,
                material=material,
                static=static,
                locked=locked,
                prefab=prefab,
                dangling=dangling,
            )
        )

    def finish(self) -> SceneExport:
        model = gltf.Model(
            self.nodes,
            self.roots,
            self.meshes,
            [],
            cameras=self.cameras,
            lights=self.lights,
        )
        return SceneExport(
            model=model,
            nodes=tuple(self.records),
            unresolved=tuple(self.unresolved),
        )

    # -- geometry ----------------------------------------------------------

    def _mesh_for(self, ref: Ref, material: gltf.Material | None) -> int | None:
        """The glTF mesh index for this reference, resolving it at most once.

        ``None`` when the reference resolved to nothing -- a library asset
        whose job is gone, or one whose background parse has not landed yet.
        The node is still written, carrying its name and its transform and no
        mesh: a scene that loses a prop must not also lose where the prop was,
        or a relink has nothing to put back.
        """
        key = (ref_key(ref), id(material))
        cached = self._mesh_index.get(key)
        if cached is not None:
            return cached
        prims = list(self.source.primitives(ref))
        if not prims:
            return None
        if material is not None:
            # ``replace`` rather than mutating what the source handed back:
            # the source caches its primitives and hands the same objects to
            # the viewport, so writing a material into one would retint the
            # asset everywhere it is drawn.
            prims = [replace(p, material=material) for p in prims]
            self._material_keep.append(material)
        index = len(self.meshes)
        self.meshes.append(prims)
        self._mesh_index[key] = index
        return index

    def _terrain_mesh(self, material: gltf.Material | None) -> int:
        """The ground, built once however many terrain nodes refer to it.

        Keyed through the same table the references use, on a tag no
        ``ref_key`` can produce: the terrain is a document singleton, so two
        :class:`~.nodes.TerrainNode`\\ s under different overrides are two
        meshes and under the same override are one -- which is the rule
        everything else here follows.
        """
        assert self.doc.terrain is not None
        key = (("mason.terrain",), id(material))
        cached = self._mesh_index.get(key)
        if cached is not None:
            return cached
        prim = tr.terrain_mesh(self.doc.terrain)
        if material is not None:
            prim = replace(prim, material=material)
            self._material_keep.append(material)
        index = len(self.meshes)
        self.meshes.append([prim])
        self._mesh_index[key] = index
        return index

    # -- markers -----------------------------------------------------------

    def _light(self, node: Node, name: str) -> int:
        """One entry per light node, never deduplicated.

        Two lights with identical settings are two lights: an engine importer
        reading the file back builds one object per node, and folding them
        would only save a few dozen bytes of JSON at the cost of making the
        node list and the light list disagree about how many there are.
        ``gltf.Light``'s fields *are* ``LightNode``'s fields, which is the
        whole reason ``nodes.py`` adopted KHR_lights_punctual's vocabulary
        rather than inventing a parallel one -- so this copies rather than
        converts, and there is nowhere for the two to drift.
        """
        assert isinstance(node, LightNode)
        self.lights.append(
            gltf.Light(
                name=name,
                kind=node.kind,
                color=tuple(float(c) for c in node.color),  # type: ignore[arg-type]
                intensity=float(node.intensity),
                range=float(node.range),
                inner_cone_angle=float(node.inner_cone_angle),
                outer_cone_angle=float(node.outer_cone_angle),
            )
        )
        return len(self.lights) - 1

    def _camera(self, node: Node, name: str) -> int:
        assert isinstance(node, CameraNode)
        self.cameras.append(
            gltf.Camera(
                name=name,
                yfov=float(node.yfov),
                znear=float(node.znear),
                zfar=float(node.zfar),
            )
        )
        return len(self.cameras) - 1
