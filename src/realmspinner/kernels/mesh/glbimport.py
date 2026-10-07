"""A GLB into a Clay document, so an imported asset is an editable one.

The loop the rest of Clay was built to close: ``jobs.import_mesh`` turns a Clay
document into an ordinary library asset, and this turns an ordinary asset back
into a Clay document. It goes through :func:`~..geom3d.gltf.load`, which already
composes every node's transform (including the grounding ``normalize_glb``
inserts under each root), already decodes textures, and already refuses sparse
accessors and non-triangle modes -- so nothing here re-implements a loader, and
an asset that will not open in the 3D viewport will not half-open here either.

**One ``Obj`` per primitive, not per node.** A glTF primitive carries exactly
one material, and Clay stores material per *face*, so the two are not the same
split -- but a node with three primitives is three materials, and merging them
into one object would need a palette lookup per face at import time to say
nothing the user asked for. Three objects the outliner shows separately, that
can be selected and moved separately, is what a modeller expects from a file
that was authored that way.

**Vertices are merged bitwise, not by tolerance.** An exporter splits a vertex
wherever a normal or a uv disagrees, and the split copies are *bit-identical* in
position -- so ``np.unique`` over the raw twelve bytes puts them back together
exactly, with no epsilon to choose and no chance of welding two features that
happen to be a hair apart. Tolerance welding is a repair tool, and Clay has one
(``ops_topo.weld``); making it the import default would silently reshape assets.

**The uv survives the merge because it is per corner.** That is precisely what
the per-corner ``Mesh.uv`` is for: the exporter's split vertices carried
different uvs at one position, and after the merge those become two corners at
one vertex, which is exactly how a seam is expressed here.

**Smoothing is a heuristic, and it is stated as one.** glTF has no smooth flag;
it has normals. A face whose corner normals all agree with its own geometric
normal was flat-shaded, and one where they do not was smooth -- measured against
a cosine of 0.999, which is about 2.5 degrees. A mesh with no normals at all is
taken as smooth, because that is what a reader that computes its own normals
will do with it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..geom3d import glbio, gltf
from ..geom3d import math3d as m3
from . import topo
from .document import ClayDoc, Obj, new_uid, reduce_material
from .elements import OpError

__all__ = ["MAX_OBJECTS", "MAX_TRIANGLES", "MAX_VERTICES", "glb_to_claydoc"]

# Above this an import is refused rather than attempted. A Clay document holds
# every mesh in memory twice per undo step, and the rebuild-per-edit cost is
# linear in the triangle count -- two million is already "usable but slow" and
# past it the app stops responding rather than becoming slower.
MAX_TRIANGLES = 2_000_000

# And above this, likewise, on the object count. Clay's outliner, undo stack
# and per-object GPU cache are all scoped for a modelling document rather than
# a scene graph: tens of thousands of instances overwhelm all three long before
# the triangle budget above notices, because a glTF that instances one small
# mesh a hundred thousand times is a few megabytes of JSON.
MAX_OBJECTS = 4_096

# The 2026-10-03 audit, finding clay-36: the two ceilings above count triangles
# and objects but never vertices, and a primitive's POSITION stream can be far
# longer than anything its indices reach -- a 588-byte GLB declaring a
# 4,000,000-vertex bufferless POSITION accessor and four instancing nodes took
# 11.4 s in ``np.unique`` and held 244 MB, with one triangle. Counted as
# vertex *references* (a primitive's POSITION count, once per node placing
# it), because every placing node gets a mesh of that many vertices. Three per
# triangle at the triangle ceiling: a triangle soup of exactly the largest
# mesh Clay edits still passes, and anything past it is vertex data nothing
# draws. A new ceiling, not a raised one.
MAX_VERTICES = 3 * MAX_TRIANGLES

# How closely a corner normal must agree with its face's own normal to count as
# flat. cos(2.5 degrees); tighter than this and float noise in an exporter's
# normals reads every flat face as smooth.
FLAT_COSINE = 0.999

# The 2026-10-07 audit's clay-19: the largest magnitude a float32 vertex holds.
_F4_MAX = float(np.finfo("f4").max)

# The 2026-10-07 audit's clay-01: the scale a zero-scaled node is lifted to
# (see ``_object_for``). One ten-thousandth: far below anything a modeller
# draws, far above where float32 or a matrix inverse starts to lose it.
_LIFTED_SCALE = 1e-4


def _placed_nodes(model: gltf.Model) -> list[gltf.Node]:
    """The nodes that name a mesh *and* that the active scene places.

    The 2026-10-03 audit, finding clay-31: this used to be every node in
    ``model.nodes`` that names a mesh, but ``Model.update_world`` only visits
    what the scene's roots reach. A node in another scene, or one nobody
    parents, was imported anyway -- at the identity, with its own translation,
    rotation and scale dropped -- so a valid multi-scene GLB came in as extra
    objects stacked at the origin that the file never places.
    """
    reached = model.reached
    return [
        node
        for index, node in enumerate(model.nodes)
        if node.mesh is not None and index in reached
    ]


def _instanced_budget(model: gltf.Model) -> tuple[int, int]:
    """Triangles and objects this GLB will actually build. -> (tris, objects)

    Counted over *nodes*, not over ``model.meshes``. The two disagree whenever
    a glTF instances -- one mesh definition referenced by a thousand nodes is a
    thousand independent ``Obj`` here, because ``_object_for`` bakes the node's
    own transform into a fresh copy rather than sharing one. Summing the mesh
    list instead counted such a file once and then allocated it a thousand
    times, which is how a few megabytes of JSON got past a two-million-triangle
    ceiling.
    """
    tris = 0
    objects = 0
    for node in _placed_nodes(model):
        for prim in model.meshes[node.mesh]:
            tris += len(prim.indices) // 3
            objects += 1
    return tris, objects


def _instanced_vertices(model: gltf.Model) -> int:
    """Vertex references this GLB will actually build: every placed node times
    every primitive's POSITION rows (clay-36, see ``MAX_VERTICES``)."""
    return sum(
        len(prim.positions) for node in _placed_nodes(model) for prim in model.meshes[node.mesh]
    )


def _declared_reachable(doc: dict, nodes: list) -> set[int] | None:
    """The node indices the active scene places, read off the JSON alone, or
    ``None`` when the structure is too odd to say (the caller then counts every
    node, the old and merely pessimistic answer).

    The 2026-10-03 audit, finding clay-31: the declared budget counted every
    node naming a mesh, so a multi-scene GLB whose *other* scenes held the
    bulk of the nodes was refused as a scene it did not place. The same walk
    ``gltf._roots`` and ``Model.update_world`` make, tolerant of a malformed
    file because this runs before ``gltf.load``'s own named refusals.
    """
    scenes = doc.get("scenes") or []
    if not isinstance(scenes, list):
        return None
    index = doc.get("scene", 0)
    roots: list | None = None
    if scenes:
        if not isinstance(index, int) or isinstance(index, bool):
            return None
        if 0 <= index < len(scenes):
            scene = scenes[index]
            if not isinstance(scene, dict):
                return None
            if "nodes" in scene:
                roots = scene["nodes"]
                if not isinstance(roots, list):
                    return None
    if roots is None:
        parented: set[int] = set()
        for node in nodes:
            kids = node.get("children") if isinstance(node, dict) else None
            if isinstance(kids, list):
                parented.update(k for k in kids if isinstance(k, int))
        roots = [i for i in range(len(nodes)) if i not in parented]
    reached: set[int] = set()
    stack = [r for r in roots if isinstance(r, int)]
    while stack:
        i = stack.pop()
        if i in reached or not 0 <= i < len(nodes):
            continue
        reached.add(i)
        kids = nodes[i].get("children") if isinstance(nodes[i], dict) else None
        if isinstance(kids, list):
            stack.extend(k for k in kids if isinstance(k, int) and k not in reached)
    return reached


def _declared_budget(data: bytes) -> tuple[int, int]:
    """Triangles and objects this GLB's JSON *claims*. See :func:`_declared_counts`."""
    tris, objects, _vertices = _declared_counts(data)
    return tris, objects


def _declared_vertices(data: bytes) -> int:
    """Vertex references this GLB's JSON *claims*. See :func:`_declared_counts`."""
    return _declared_counts(data)[2]


def _declared_counts(data: bytes) -> tuple[int, int, int]:
    """Triangles, objects and vertex references this GLB's JSON *claims*, with
    no accessor decoded. -> (tris, objects, vertices)

    H01: the real count below (``_instanced_budget``) is trustworthy but only
    answers after ``gltf.load`` has already decoded every primitive it names
    -- which is exactly the allocation this ceiling exists to refuse *before*.
    A triangle count is a declared number, not a computed one: an index
    accessor names its own ``count``, and an unindexed primitive's triangle
    count is its POSITION accessor's ``count`` -- both readable out of the
    JSON chunk alone, for the price of a JSON parse rather than a buffer
    decode. A malformed or unparsable GLB reads as ``(0, 0)`` here and is left
    to ``gltf.load``'s own, more specific refusal.
    """
    try:
        _header, doc, _rest = glbio.split_glb(data)
    except ValueError:
        return 0, 0, 0
    # The 2026-09-26 audit's clay-io-04: a node count already past what
    # ``gltf.load`` would refuse for (``gltf.MAX_NODES``) is refused here too,
    # on the JSON-only count alone, rather than walking every one of them (and
    # every primitive on every mesh they name) in pure Python first only to
    # reach the identical refusal at real decode cost a few lines later.
    nodes = doc.get("nodes") or []
    # Finding clay-io-09, the same audit: this function runs *before*
    # ``glb_to_claydoc``'s own try/except around ``gltf.load`` -- it is the
    # cheap pre-check that exists specifically to avoid paying for that load,
    # so nothing here may raise either. ``nodes``/``accessors``/``meshes``
    # each used to be trusted as a list the moment ``.get(...)`` returned a
    # truthy value; a GLB whose JSON instead declares one of them as an
    # *object* (``{"0": {...}}`` rather than ``[{...}]``) reached ``len()`` or
    # an int index into a dict below as a bare, uncaught ``TypeError``/
    # ``KeyError`` instead of the "malformed or unparsable GLB reads as
    # (0, 0)" fallback this function's own docstring promises.
    if not isinstance(nodes, list):
        nodes = []
    if len(nodes) > gltf.MAX_NODES:
        return 0, MAX_OBJECTS + 1, 0
    accessors = doc.get("accessors") or []
    if not isinstance(accessors, list):
        accessors = []
    meshes = doc.get("meshes") or []
    if not isinstance(meshes, list):
        meshes = []

    def _count(index: Any) -> int:
        if not isinstance(index, int) or not 0 <= index < len(accessors):
            return 0
        # The 2026-09-12 audit, finding clay-03: a legal JSON array can still
        # hold an illegal glTF accessor -- a bare string or list at this slot
        # -- and that used to reach ``.get`` as an uncaught AttributeError
        # rather than falling back to 0 the way an out-of-range index already
        # does a few lines up.
        if not isinstance(accessors[index], dict):
            return 0
        try:
            return max(0, int(accessors[index].get("count", 0)))
        except (TypeError, ValueError, OverflowError):
            # clay-io-09: ``count`` is a declared JSON number, and Python's
            # ``json`` module accepts a bare ``Infinity``/``-Infinity``
            # literal by default -- ``int(float("inf"))`` raises
            # ``OverflowError`` rather than either type this already caught,
            # the same gap clay-document-07 found and closed in the reader
            # for this exact "int() of a JSON number" shape elsewhere.
            return 0

    reached = _declared_reachable(doc, nodes)
    tris = 0
    objects = 0
    vertices = 0
    for position, node in enumerate(nodes):
        if reached is not None and position not in reached:
            continue
        mesh_index = node.get("mesh") if isinstance(node, dict) else None
        if not isinstance(mesh_index, int) or not 0 <= mesh_index < len(meshes):
            continue
        # Same shape of guard, same incident: a mesh entry can be present and
        # in range while still not being an object.
        if not isinstance(meshes[mesh_index], dict):
            continue
        # clay-io-09: ``primitives`` is declared as an array by the glTF
        # schema, but nothing stopped a non-list truthy value (a number, a
        # string) from reaching ``for prim in ...`` as an uncaught
        # ``TypeError`` -- a dict here already fails ``isinstance(prim,
        # dict)`` harmlessly per iteration, but a scalar is not iterable at
        # all.
        primitives = meshes[mesh_index].get("primitives")
        if not isinstance(primitives, list):
            continue
        for prim in primitives:
            if not isinstance(prim, dict):
                continue
            attrs = prim.get("attributes")
            # The 2026-10-03 audit, finding clay-87: ``or {}`` only covers an
            # absent or falsy value, so a list/string/number "attributes"
            # reached ``attrs.get`` below as a bare AttributeError -- from a
            # function that runs before ``glb_to_claydoc``'s own try/except.
            if not isinstance(attrs, dict):
                attrs = {}
            declared = (
                _count(prim["indices"])
                if "indices" in prim
                else _count(attrs.get("POSITION"))
            )
            tris += declared // 3
            objects += 1
            vertices += _count(attrs.get("POSITION"))
            # The early exit itself (see the docstring): once any running
            # total is already past what the caller refuses for, there is
            # nothing left for the rest of this file's own declared entries
            # to change about the verdict.
            if tris > MAX_TRIANGLES or objects > MAX_OBJECTS or vertices > MAX_VERTICES:
                return tris, objects, vertices
    return tris, objects, vertices


def glb_to_claydoc(data: bytes, name: str = "Imported") -> ClayDoc:
    """Parse GLB bytes into a fresh :class:`~.document.ClayDoc`.

    The document comes back with a clean history and nothing selected: it has
    just been opened, so it is not unsaved, and the objects are placed by
    construction rather than through ``add_object``, which would push a step
    apiece.
    """
    # Checked from the file's own declared numbers before a single accessor is
    # decoded (H01): the budget below re-checks the *real* counts after
    # ``gltf.load`` runs, but by then the load has already paid for whatever
    # it is about to refuse. A GLB that declares more than Clay will ever hold
    # gets the same verdict for a JSON parse instead of a full decode.
    declared_tris, declared_objects, declared_vertices = _declared_counts(data)
    if declared_tris > MAX_TRIANGLES:
        raise OpError(
            f"This mesh declares {declared_tris:,} triangles, past the "
            f"{MAX_TRIANGLES:,} Clay can edit. Retarget its triangle budget in "
            "the library first."
        )
    if declared_objects > MAX_OBJECTS:
        raise OpError(
            f"This GLB declares placing {declared_objects:,} objects, past the "
            f"{MAX_OBJECTS:,} Clay holds. It is a scene rather than a model -- "
            "import the part you mean to edit."
        )

    if declared_vertices > MAX_VERTICES:
        raise OpError(
            f"This GLB declares {declared_vertices:,} vertices across the objects "
            f"it places, past the {MAX_VERTICES:,} Clay can hold. Most of that is "
            "vertex data its triangles never reach, or one mesh instanced very "
            "widely -- import the part you mean to edit."
        )

    try:
        model = gltf.load(data)
    except Exception as exc:  # noqa: BLE001 - the loader raises several types
        raise OpError(f"This file could not be read as a GLB. {exc}") from exc

    if model.skins:
        raise OpError(
            "This GLB is rigged, and Clay has no skinning -- editing it here "
            "would drop the rig. Import it from the Library instead "
            "(Home > Import mesh..., or drop it on Home or the Library)."
        )
    total, objects_wanted = _instanced_budget(model)
    if _instanced_vertices(model) > MAX_VERTICES:
        raise OpError(
            f"This GLB places {_instanced_vertices(model):,} vertices, past the "
            f"{MAX_VERTICES:,} Clay can hold. Most of that is vertex data its "
            "triangles never reach, or one mesh instanced very widely -- import "
            "the part you mean to edit."
        )
    if total > MAX_TRIANGLES:
        raise OpError(
            f"This mesh has {total:,} triangles, past the {MAX_TRIANGLES:,} Clay "
            "can edit. Retarget its triangle budget in the library first."
        )
    if objects_wanted > MAX_OBJECTS:
        raise OpError(
            f"This GLB places {objects_wanted:,} objects, past the "
            f"{MAX_OBJECTS:,} Clay holds. It is a scene rather than a model -- "
            "import the part you mean to edit."
        )

    materials: list[gltf.Material] = []
    palette: dict[int, int] = {}
    objects: list[Obj] = []
    taken: set[str] = set()
    # The 2026-10-03 audit, finding clay-36: one merged ``Mesh`` per
    # (primitive, material slot), shared by every node that places it. The
    # merge (``np.unique`` over the whole POSITION stream) used to run once per
    # *node*, so instancing one large primitive a few thousand times paid for
    # it a few thousand times. ``Mesh`` is frozen with read-only arrays, so
    # sharing one between objects is safe -- every edit builds a new one.
    built: dict[tuple[int, int], Any] = {}
    for node in _placed_nodes(model):
        for prim in model.meshes[node.mesh]:
            obj = _object_for(prim, node, name, materials, palette, taken, built)
            if obj is not None:
                objects.append(obj)

    if not objects:
        raise OpError("This GLB has no meshes in it.")
    return ClayDoc(objects=objects, materials=materials or None)


def _object_for(
    prim: gltf.Primitive,
    node: gltf.Node,
    base: str,
    materials: list[gltf.Material],
    palette: dict[int, int],
    taken: set[str],
    built: dict[tuple[int, int], Any] | None = None,
) -> Obj | None:
    if len(prim.indices) < 3 or len(prim.positions) == 0:
        return None
    slot = _material_index(prim.material, materials, palette)
    if built is None:
        mesh = _mesh_for(prim, slot)
    else:
        mesh = built.get((id(prim), slot))
        if mesh is None:
            mesh = built[(id(prim), slot)] = _mesh_for(prim, slot)
    # The 2026-10-03 audit, finding clay-37: a node transform that is not
    # finite (a NaN the loader's own TRS check cannot see, or a chain of scales
    # that overflows when composed) would be baked into the mesh or stored on
    # the object, and the document would fail at export far from here.
    if not np.all(np.isfinite(node.world)):
        raise OpError(f"Node {node.name or base!r} has a non-finite transform.")
    # The 2026-10-07 audit's clay-19: the check above is in float64, but the
    # mesh is float32 -- a scale of 1e200 or a translation of 1e39 is a finite
    # matrix whose product with a float32 vertex is not, and the bake (or the
    # first export, which bakes the same matrix) then wrote inf/NaN positions
    # into a document that opened fine and failed far from the file. The
    # placed positions are measured here, in float64, before anything is
    # built from them; ``set_transform`` refuses what a user types, and this
    # is the same door for what a file says.
    placed = _world_positions(mesh, node.world)
    if not np.all(np.abs(placed) <= _F4_MAX):
        raise OpError(
            f"Node {node.name or base!r} places this mesh outside the range Clay can "
            "hold (its scale or translation is too large for float32 positions)."
        )
    translation, rotation, scale = m3.decompose(node.world)
    # The 2026-09-26 audit's clay-io-02: ``gltf.py``'s own loader refuses a
    # node whose *own* declared ``matrix`` has shear (the 2026-09-20 audit's
    # clay-16, right beside ``m3.decompose``'s own docstring), by this exact
    # recompose-and-compare check -- but ``node.world`` is composed through
    # the whole ancestor chain, and two individually shear-free T*R*S
    # matrices (a non-uniform-scaled parent over a rotated child) can still
    # compose into one that has shear, which that per-node check never sees.
    # ``decompose`` cannot represent it either way -- it just drops it,
    # silently distorting the imported geometry -- so a composed shear bakes
    # the real matrix straight into the mesh's own vertices instead, which
    # loses nothing: this module stores no per-vertex normal for a transform
    # to invalidate, and every reader derives one from the baked geometry.
    if not np.allclose(m3.compose(translation, rotation, scale), node.world, atol=1e-4, rtol=1e-4):
        mesh = replace(mesh, positions=np.ascontiguousarray(placed, dtype="f4"))
        translation, rotation, scale = m3.vec3(), m3.quat_identity(), m3.vec3(1.0, 1.0, 1.0)
    elif not np.any(scale):
        # The 2026-10-07 audit's clay-01: a node whose world scale is zero --
        # or so small that ``decompose``'s column norms underflow to it -- is
        # how game assets hide a part, and it used to arrive as an object with
        # scale (0, 0, 0) that ``set_transform`` refuses and ``read_rblk``
        # refuses to reopen: the document saved, autosaved and journalled, and
        # could never be opened again (crash recovery lost it too). Refusing
        # the import would refuse the asset for carrying a hidden part, and
        # baking the matrix would crush the part's geometry to a point for
        # good; instead the part comes in at a floor scale -- invisible at any
        # normal zoom, geometry intact, and one scale edit from visible.
        scale = m3.vec3(_LIFTED_SCALE, _LIFTED_SCALE, _LIFTED_SCALE)
    return Obj(
        uid=new_uid(),
        name=_unique(node.name or base, taken),
        mesh=mesh,
        translation=translation,
        rotation=rotation,
        scale=scale,
        # An imported mesh is not describable by a generator's parameters, so
        # the properties panel shows counts rather than a size field that would
        # discard the file the moment it was touched.
        generator=None,
        # The 2026-09-08 audit's clay-02: this used to fall through to the
        # dataclass default of 0 for every object, because ``slot`` was
        # computed for the mesh but never handed to ``Obj`` -- so the
        # properties panel's "default material" (and what an Extrude/Inset
        # paints new faces with) was wrong for any object past the first
        # primitive in an ordinary multi-material import.
        material=slot,
    )


def _material_index(
    material: gltf.Material | None,
    materials: list[gltf.Material],
    palette: dict[int, int],
) -> int:
    """The palette slot for a loader material, adding it the first time.

    Deduplicated by the loader's own object identity -- two primitives that
    reference one glTF material share the object, and two that reference
    different ones must stay separate even if every factor matches, because the
    user is about to edit them.

    What is stored is :func:`~.document.reduce_material`'s subset -- name,
    colour, base-colour texture, double-sided, cutout -- so a metallic, emissive
    or normal-mapped source arrives as the plain material Clay can edit, while
    the identity key stays the loader's own object.
    """
    if material is None:
        material = gltf.Material(name="imported")
    key = id(material)
    if key not in palette:
        palette[key] = len(materials)
        materials.append(reduce_material(material))
    return palette[key]


def _world_positions(mesh: Any, matrix: np.ndarray) -> np.ndarray:
    """*mesh*'s vertices placed by *matrix*, in float64 so an overflow shows.

    :func:`_object_for`'s escape hatch for a composed node transform ``Obj``'s
    own T/R/S cannot represent (see its own comment, the 2026-09-26 audit's
    clay-io-02) is to bake exactly this into the geometry once, up front: that
    keeps the imported shape identical to what the file actually places, where
    handing ``decompose`` a sheared matrix would have silently dropped the
    shear instead. The same product is what the overflow check measures.
    """
    points = np.asarray(mesh.positions, dtype="f8")
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype="f8")], axis=1)
    with np.errstate(all="ignore"):
        return (matrix @ homogeneous.T).T[:, :3]


def _mesh_for(prim: gltf.Primitive, material: int) -> Any:
    """One primitive's triangles as a CSR mesh, vertices merged bitwise."""
    positions = np.ascontiguousarray(prim.positions, dtype="f4").reshape(-1, 3)
    # The 2026-10-03 audit, finding clay-37: GLB, STL and PLY accepted a NaN or
    # infinite position that ``objimport`` already refuses by name, so the
    # import succeeded and the document could not be saved to a GLB (the bound
    # is non-finite) or round-tripped through OBJ -- the failure arriving far
    # from the file that caused it.
    if not np.all(np.isfinite(positions)):
        raise OpError("This mesh has a non-finite (NaN or infinite) vertex position.")
    # The 2026-10-07 audit's clay-70: the same refusal for the UV stream.
    # ``write_glb`` happily writes a NaN texcoord, so a GLB from any exporter
    # that did could import, and the document then wrote ``vt nan`` into every
    # OBJ it exported -- a file Clay's own OBJ reader refuses.
    if prim.uvs is not None and not np.all(np.isfinite(np.asarray(prim.uvs, dtype="f4"))):
        raise OpError("This mesh has a non-finite (NaN or infinite) UV coordinate.")
    indices = np.asarray(prim.indices, dtype="i8").reshape(-1)
    tris = indices[: (len(indices) // 3) * 3].reshape(-1, 3)

    # ``np.unique`` over the raw rows: an exporter's split copies are
    # bit-identical in position, so this is exact rather than tolerant.
    merged, inverse = np.unique(positions, axis=0, return_inverse=True)
    loops = inverse.reshape(-1)[tris].reshape(-1)

    uv = None
    if prim.uvs is not None:
        uv = np.ascontiguousarray(prim.uvs, dtype="f4").reshape(-1, 2)[tris].reshape(-1, 2)

    n_faces = len(tris)
    return topo.rebuild(
        merged,
        loops,
        topo.starts_from_counts(np.full(n_faces, 3, dtype="i8")),
        np.full(n_faces, int(material), dtype="i4"),
        _smooth_flags(prim, merged, loops, tris),
        uv=uv,
    )


def _smooth_flags(
    prim: gltf.Primitive, positions: np.ndarray, loops: np.ndarray, tris: np.ndarray
) -> np.ndarray:
    """Per-face smooth flags, from how far the corner normals bend. See the module docs."""
    n_faces = len(tris)
    if prim.normals is None or n_faces == 0:
        return np.ones(n_faces, dtype=bool)

    corners = positions[loops.reshape(-1, 3)].astype("f8")
    face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    face = _unit(face)
    supplied = _unit(
        np.asarray(prim.normals, dtype="f8").reshape(-1, 3)[tris].reshape(-1, 3)
    ).reshape(-1, 3, 3)
    agreement = np.einsum("ijk,ik->ij", supplied, face).min(axis=1)
    return agreement <= FLAT_COSINE


def _unit(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, length, out=np.zeros_like(v), where=length > 0.0)


def _unique(name: str, taken: set[str]) -> str:
    """``Box`` -> ``Box.001``, the same counting rule the outliner elsewhere uses."""
    if name not in taken:
        taken.add(name)
        return name
    for n in range(1, len(taken) + 2):
        candidate = f"{name}.{n:03d}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate
    return name  # pragma: no cover - unreachable, the loop is bounded above
