"""``.wscn`` -- the Mason document on disk, a zip in ``.wblk``'s shape.

**No geometry is stored at all.** A ``.wblk`` writes a mesh's arrays because
Clay's whole subject is the mesh; Mason's subject is an *arrangement* of
things that live somewhere else -- ``document.py``'s own module docstring
says the package never resolves a reference and never draws anything, and
this format is the disk-shaped restatement of that: a :class:`~.refs.PrimitiveRef`
is a generator name and its parameters, and a :class:`~.refs.LibraryRef` is a
job id and an artifact name. Both regenerate their triangles from the host's
``GeometrySource`` the moment the scene opens; neither is worth a byte of
vertex data in the file that names it. What *is* worth storing is exactly what
``ClayDoc.materials``'s own comment calls "a save/export convenience, not an
addressing scheme" one document up: the tree, the palette, the one terrain,
and where the camera was left.

**Version and epoch, unchanged from Clay's argument.** ``scene.json`` carries
its own ``version`` and this reader refuses one written by a newer build,
because a half-understood document is worse than a refused one -- substituting
something plausible for a field this build has never heard of would open the
file, show the user a scene that is quietly wrong, and let them save that over
their work. Every zip member -- ``scene.json``, ``terrain/heights.npy``, every
``textures/<n>.png`` -- is stamped at :data:`_EPOCH`, the same 1980-01-01 zip's
own format floor Clay stamps at, for the same reason: two saves of an unchanged
document must be byte-identical, or the file is undiffable, a content hash is
useless, and "did this actually change" is unanswerable from outside the app.

**Materials are written by index and shared by identity on read.** A
``MeshNode``/``TerrainNode``'s ``material`` field holds a ``gltf.Material``
*object*, never an index -- ``document.py``'s own comment on ``self.materials``
gives the reason, restated from ``clay/document.py``: the export is the
definition, and a parallel indexing scheme buys nothing but a second place to
disagree with the first. So the writer collects every *distinct* material by
``id()`` across ``doc.materials``, every node's override in the scene tree and
every prefab template, and the terrain's own material, in that stable order,
and writes one ``"materials"`` array with everything else pointing into it by
index. The reader builds that array exactly once and hands the *same* Python
object to every node that named its index -- both ``write_glb`` and the
viewport's ``GpuMaterial`` cache de-duplicate a material by ``id()``, so two
nodes that shared one material before a save sharing two after it would
silently double a scene's GPU uploads and its glTF material count on every
round trip. This is also why ``doc.materials`` after a load is the *whole*
array read from the file rather than some remembered subset of it: nothing in
the format records which entries were "the palette" and which were "only ever
an override", and a superset that is a strict extension of the one the
document was saved with is exactly the "convenience list", not "addressing
scheme" role ``document.py`` already gives it. It is also what keeps a second
save byte-identical to the first -- see :func:`_collect_materials`'s own note.

**The two-phase split is a finding Clay already closed, and this stage must
not reopen it.** The 2026-09-06 audit (clay-03) found ``clay_mode``'s save and
export paths calling the zip-and-encode half of ``.wblk``'s writer directly on
the frame thread, before ``ctx.submit`` ever ran -- exactly the stall
``wblk.snapshot``/``snapshot_bytes`` exists to make affordable to pay for, not
free to skip. :func:`snapshot` is what runs on the frame thread here: building
``scene.json``'s text (a walk of the tree and the palette, no large
allocation) and taking the terrain's height array and every texture's decoded
bytes *by reference*. :func:`snapshot_bytes` is the task-thread half: the zip
container, the ``.npy`` encode and every PNG encode. A caller already off the
frame thread -- a test, crash recovery, a batch conversion -- is free to call
both at once as :func:`wscn_bytes` does.

**Taking the height array by reference is sound for the reason
``terrain.Terrain``'s own docstring gives: every brush *rebinds*
``terrain.heights`` to a new array; nothing ever writes into the live one in
place.** A reference taken here keeps meaning exactly what it meant when the
snapshot was built no matter how many sculpt strokes land on the *live*
document before the task thread gets around to encoding it -- the same
argument ``WblkSnapshot`` makes for a ``Mesh``'s immutable CSR arrays, one
brush over. Texture bytes are held the same way, for the reason
``WblkSnapshot`` already gives for them: a material's image is never mutated
once it holds one, only replaced wholesale by a different tuple.

**The dangling-reference story, and why it is not a contradiction of
``.wblk``'s refuse-rather-than-substitute rule.** ``document.py``'s
``missing_refs`` docstring already makes this argument from the document's
side; restated from the file format's: a ``.wblk`` that cannot replay a build
step refuses to open because the alternative -- an empty mesh standing in for
the real one -- would show the user finished work that silently is not there,
with nothing left to recover if they saved over it. A ``.wscn`` naming a
``LibraryRef`` whose job no longer exists is a different shape of file: it is
a set of *links*, most of which still resolve, and refusing the whole scene
over one dead job id would strand every other node in it for a reason nobody
outside the app can see or fix. So this reader opens the file regardless --
the node keeps its ref exactly as saved, and ``doc.missing`` stays empty,
because this module never touches the filesystem and cannot know whether a
job id still resolves. Filling ``doc.missing`` is the host's job, once it has
tried to resolve every ref and found one it could not.

**``doc.view`` is opaque until this module, and this module is the first thing
that writes it.** ``document.py``'s own comment says so: it is a plain
``dict[str, Any]`` nothing in the engine reads a key out of. :data:`VIEW_FIELDS`
pins the exact key set this format carries, the same way Clay's own
``VIEW_FIELDS`` pins its camera's -- an unpinned dict would round-trip
whatever a caller happened to put there once, then silently drop any key the
next build did not recognise. A camera missing a key or carrying a non-finite
one is dropped whole rather than refusing the document: it is worth one
unfitted viewport on open, never a refused scene.

Restored by *construction*, never through ``doc.add_node`` -- which would push
one undo step per node and open every file already dirty -- and every restored
uid is raised through ``nodes.reserve_uid`` as it is read, so a node minted
fresh after a load can never collide with one the file just placed.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from typing import Any

import numpy as np

from ...core.safeio import npyguard, pixelguard, zipguard
from ...kernels.geom3d import gltf
from . import nodes as nd
from .document import MasonDoc
from .refs import LibraryRef, PrimitiveRef, Ref, primitive_ref
from .scene import MAX_PLACED as MAX_NODES
from .terrain import Terrain

VERSION = 1
SCENE = "scene.json"
TERRAIN_HEIGHTS = "terrain/heights.npy"
TEXTURE_DIR = "textures"

# 1980-01-01, the earliest a zip can express -- ``clay/serialize.py``'s own
# constant, restated here rather than imported: it is a property of the zip
# format, not of Clay, and importing a sibling engine for one tuple would be
# the exact chain ``test_mason_imports.py`` exists to catch.
_EPOCH = (1980, 1, 1, 0, 0, 0)

# A zip's central directory declares what each member unpacks to, and nothing
# makes that number honest -- ``zipguard.BoundedZip`` closes the gap between
# the claim and the truth, but the claim is still the cheapest place to refuse
# an archive that was never going to fit, before a single member is read. One
# gigabyte is far past any scene this editor produces (a terrain at its own
# ceiling is a bit over a megabyte; ``scene.json`` for 100,000 nodes is tens of
# megabytes of text) and far short of a machine's RAM. Read from the module
# global at call time so a test can lower it.
MAX_DECOMPRESSED_BYTES = 1 << 30

#: The texture slots a material can carry, in ``TEXTURE_FIELDS`` order --
#: mirrored from ``clay/serialize.py`` rather than imported, for the reason
#: ``_EPOCH`` is: this package may not reach for a sibling engine, and the
#: names are a *file format* here, pinned locally so a rename in the renderer
#: cannot silently change what a saved scene means.
TEXTURE_FIELDS = (
    "base_color",
    "metallic_roughness",
    "normal",
    "emissive",
    "occlusion",
)

#: The camera keys ``scene.json`` carries, and the only ones read back. Written
#: out rather than taken from whatever dict a caller hands in, because this is
#: a *file format*: an unrecognised key would round-trip once and then be
#: silently dropped by the next build that read the table instead of the dict.
#: ``target`` is not listed here -- it is three numbers, not one, and is
#: handled next to this tuple rather than inside it, exactly as Clay's own
#: ``view_json``/``read_view`` split it.
VIEW_FIELDS = ("yaw", "pitch", "distance")

#: Every node kind this format carries, and the one class each names. A dict
#: literal rather than a chain of ``isinstance`` checks on the way out, and a
#: lookup rather than a chain of ``if kind == ...`` on the way in -- the
#: module docstring's own argument against a ``kind`` discriminator, restated
#: one layer up for the six concrete classes that discriminator would have
#: replaced.
_KIND_CLASSES: dict[str, type[nd.Node]] = {
    "group": nd.GroupNode,
    "mesh": nd.MeshNode,
    "light": nd.LightNode,
    "camera": nd.CameraNode,
    "prefab": nd.PrefabNode,
    "terrain": nd.TerrainNode,
}
_CLASS_KINDS: dict[type[nd.Node], str] = {cls: kind for kind, cls in _KIND_CLASSES.items()}


# --- writing -----------------------------------------------------------------


def _material_json(material: gltf.Material, textures: dict[int, int]) -> dict[str, Any]:
    """One material's factors, plus the index of each texture slot that has one.

    Mirrors ``clay/serialize.py``'s ``_material_json`` exactly -- the shape is
    the same fact about a ``gltf.Material`` regardless of which document is
    asking -- restated locally rather than imported for the reason
    :data:`TEXTURE_FIELDS` is.
    """
    slots = {}
    for slot in TEXTURE_FIELDS:
        image = getattr(material, slot, None)
        if image is not None:
            slots[slot] = textures[id(image)]
    entry: dict[str, Any] = {
        "name": material.name,
        "base_color_factor": [float(v) for v in material.base_color_factor],
        "metallic_factor": float(material.metallic_factor),
        "roughness_factor": float(material.roughness_factor),
        "emissive_factor": [float(v) for v in material.emissive_factor],
        "double_sided": bool(material.double_sided),
        "alpha_mode": str(material.alpha_mode),
        "alpha_cutoff": float(material.alpha_cutoff),
    }
    if slots:
        entry["textures"] = slots
    return entry


def _collect_materials(doc: MasonDoc) -> tuple[list[gltf.Material], dict[int, int]]:
    """Every distinct material this document carries, and a map from identity
    to index.

    **Order is what makes two saves of an unchanged document byte-identical,
    and what makes a save-load-save round trip byte-identical too.** ``doc.materials``
    comes first, in its own order; then every override :func:`nd.walk` finds
    across the scene tree, in walk order; then every override across each
    prefab template, walked in ``sorted(doc.prefabs)`` order for the same
    reason the prefab table itself is written sorted; then the terrain's own
    material, last, since it is a document singleton rather than something a
    walk crosses. After a load, ``doc.materials`` **is** this whole array (see
    the module docstring) -- so on the *next* save, every override this walk
    would find is already present in ``doc.materials`` by identity, added
    first, in the same position it holds here. Nothing after the first pass
    can move the list, which is the whole argument.
    """
    materials: list[gltf.Material] = []
    index: dict[int, int] = {}

    def add(material: gltf.Material | None) -> None:
        if material is None or id(material) in index:
            return
        index[id(material)] = len(materials)
        materials.append(material)

    for material in doc.materials:
        add(material)
    for node, _parent, _child_index, _depth in nd.walk(doc.roots):
        if isinstance(node, (nd.MeshNode, nd.TerrainNode)):
            add(node.material)
    for name in sorted(doc.prefabs):
        for node, _parent, _child_index, _depth in nd.walk([doc.prefabs[name]]):
            if isinstance(node, (nd.MeshNode, nd.TerrainNode)):
                add(node.material)
    if doc.terrain is not None:
        add(doc.terrain.material)
    return materials, index


def _collect_textures(materials: list[gltf.Material]) -> tuple[list[Any], dict[int, int]]:
    """Every distinct texture across ``materials``, and a map from identity to
    index -- ``clay/serialize.py``'s ``_collect_textures``, taking the
    already-collected material list rather than a document, since this
    format's palette is not simply ``doc.materials`` (see
    :func:`_collect_materials`). Identity, not value: a texture is a
    multi-megabyte bytes object and comparing two of them per material per
    save is the wrong price for the same answer -- the same rule every other
    consumer already uses.
    """
    images: list[Any] = []
    index: dict[int, int] = {}
    for material in materials:
        for slot in TEXTURE_FIELDS:
            image = getattr(material, slot, None)
            if image is None or id(image) in index:
                continue
            index[id(image)] = len(images)
            images.append(image)
    return images, index


_Collected = tuple[list[gltf.Material], dict[int, int], list[Any], dict[int, int]]


def _collect(doc: MasonDoc) -> _Collected:
    materials, material_index = _collect_materials(doc)
    images, image_index = _collect_textures(materials)
    return materials, material_index, images, image_index


def _png_bytes(image: tuple[int, int, bytes]) -> bytes:
    """One ``(width, height, rgba)`` slot as PNG bytes -- ``clay/serialize.py``'s
    ``_png_bytes`` verbatim. ``optimize`` stays off: it buys a few per cent for
    several times the time, and this is meant to be cheap enough to run
    wherever :func:`snapshot_bytes` is called from.
    """
    from PIL import Image

    width, height, data = image
    out = io.BytesIO()
    Image.frombytes("RGBA", (int(width), int(height)), bytes(data)).save(out, "PNG")
    return out.getvalue()


def _ref_json(ref: Ref) -> dict[str, Any]:
    if isinstance(ref, PrimitiveRef):
        return {"kind": "primitive", "generator": ref.generator, "params": ref.as_params()}
    if isinstance(ref, LibraryRef):
        return {
            "kind": "library",
            "job_id": ref.job_id,
            "artifact": ref.artifact,
            "name": ref.name,
            "sha256": ref.sha256,
        }
    raise TypeError(f"not a Ref: {ref!r}")  # pragma: no cover - the Ref union is closed


def _node_json(node: nd.Node, material_index: dict[int, int]) -> dict[str, Any]:
    """One node entry, its ``children`` nested recursively -- sibling order is
    both export order and outliner order (``nodes.py``'s own module
    docstring), so the list is written in exactly the order it is walked in,
    never sorted.
    """
    entry: dict[str, Any] = {
        "kind": _CLASS_KINDS[type(node)],
        "uid": int(node.uid),
        "name": node.name,
        "translation": [float(v) for v in node.translation],
        "rotation": [float(v) for v in node.rotation],  # XYZW, as everywhere
        "scale": [float(v) for v in node.scale],
        "visible": bool(node.visible),
        "locked": bool(node.locked),
        "static": bool(node.static),
        "properties": node.properties,
    }
    if isinstance(node, nd.MeshNode):
        if node.ref is not None:
            entry["ref"] = _ref_json(node.ref)
        if node.material is not None:
            entry["material"] = material_index[id(node.material)]
    elif isinstance(node, nd.LightNode):
        entry["light_kind"] = node.kind
        entry["color"] = [float(v) for v in node.color]
        entry["intensity"] = float(node.intensity)
        entry["range"] = float(node.range)
        entry["inner_cone_angle"] = float(node.inner_cone_angle)
        entry["outer_cone_angle"] = float(node.outer_cone_angle)
    elif isinstance(node, nd.CameraNode):
        entry["yfov"] = float(node.yfov)
        entry["znear"] = float(node.znear)
        entry["zfar"] = float(node.zfar)
    elif isinstance(node, nd.PrefabNode):
        entry["template"] = node.template
    elif isinstance(node, nd.TerrainNode):
        if node.material is not None:
            entry["material"] = material_index[id(node.material)]
    if node.children:
        entry["children"] = [_node_json(child, material_index) for child in node.children]
    return entry


def _terrain_json(terrain: Terrain, material_index: dict[int, int]) -> dict[str, Any]:
    return {
        "size_x": float(terrain.size_x),
        "size_z": float(terrain.size_z),
        "material": material_index[id(terrain.material)],
        "file": TERRAIN_HEIGHTS,
    }


def _view_json(view: Any) -> dict[str, Any] | None:
    """``doc.view`` as ``scene.json``'s ``"view"`` entry, or ``None`` for no
    usable camera. The same function reads it back (see :func:`read_wscn`):
    both directions are "the pinned keys of a plain dict, or nothing", and a
    plain dict is exactly what ``doc.view`` already is on either side of the
    file, so there is no separate object to convert the way Clay's
    ``view_json``/``read_view`` convert a viewport's own camera type.
    """
    if not isinstance(view, dict):
        return None
    try:
        out: dict[str, Any] = {name: float(view[name]) for name in VIEW_FIELDS}
        target = [float(v) for v in view["target"]]
    except (KeyError, TypeError, ValueError):
        return None
    if len(target) != 3 or not all(np.isfinite([*out.values(), *target])):
        return None
    out["target"] = target
    return out


def scene_json(doc: MasonDoc, collected: _Collected | None = None, *, view: Any = None) -> str:
    """``scene.json``'s text: sorted keys, indented, one entry per line-worthy
    thing -- this half of the format exists to be read, by a person looking at
    why a document opens wrong and by a diff, and there is no size argument
    left for minifying it once the terrain and the textures are the parts
    that cost bytes.

    ``collected`` is :func:`_collect`'s answer, passed in by :func:`snapshot`
    because it needs the very same one to decide which PNGs to store -- the
    two must agree about which texture is index 3, and a second walk of the
    tree here would be both a wasted pass and a second place that could answer
    differently. Left ``None``, which is what a caller wanting only the text
    does, it is worked out here.

    ``view`` defaults to ``doc.view`` -- the file's camera is the document's
    own, unlike Clay's, which belongs to the tab rather than ``ClayDoc`` -- but
    stays a parameter so a caller can hand over a different (or malformed) one
    without mutating the document to test what happens.
    """
    if collected is None:
        collected = _collect(doc)
    materials, material_index, images, image_index = collected
    scene: dict[str, Any] = {
        "version": VERSION,
        "materials": [_material_json(m, image_index) for m in materials],
        "nodes": [_node_json(n, material_index) for n in doc.roots],
    }
    if doc.prefabs:
        scene["prefabs"] = {
            name: _node_json(doc.prefabs[name], material_index) for name in sorted(doc.prefabs)
        }
    if doc.terrain is not None:
        scene["terrain"] = _terrain_json(doc.terrain, material_index)
    if images:
        scene["textures"] = [
            {"file": f"{TEXTURE_DIR}/{i}.png", "width": int(w), "height": int(h)}
            for i, (w, h, _data) in enumerate(images)
        ]
    camera = _view_json(doc.view if view is None else view)
    if camera is not None:
        scene["view"] = camera
    return json.dumps(scene, sort_keys=True, indent=2)


class WscnSnapshot:
    """The frame-thread half of a save -- cheap, and safe to encode later on
    another thread.

    ``scene`` carries over from ``WblkSnapshot`` unchanged in spirit: it is
    already-built text, assembled from a walk of the tree and the palette that
    allocates nothing large, so there is no reason to defer it and every
    reason not to -- doing it here means the task thread never touches
    ``doc.roots``, ``doc.prefabs`` or ``doc.materials`` at all, none of which
    is guarded by a lock against the next edit landing on the frame thread.

    ``images`` also carries over unchanged in spirit: each entry is an
    already-built ``(width, height, bytes)`` tuple, held by reference because
    nothing mutates one once a material holds it -- only replaces it, wholesale,
    with a different tuple.

    ``heights`` is Mason's own, replacing ``WblkSnapshot.meshes`` -- there is
    no geometry to snapshot (the module docstring says why), only the one
    height field a terrain may have, taken by reference. That is sound for the
    reason ``terrain.Terrain``'s own docstring gives: every brush *rebinds*
    ``heights`` to a new array rather than writing through it, so a reference
    taken here keeps meaning exactly what it meant when the snapshot was built
    no matter how many sculpt strokes land on the live document before the
    task thread encodes it. ``None`` when the document has no terrain.
    """

    __slots__ = ("scene", "heights", "images")

    def __init__(
        self, scene: str, heights: np.ndarray | None, images: tuple[Any, ...]
    ) -> None:
        self.scene = scene
        self.heights = heights
        self.images = images


def snapshot(doc: MasonDoc) -> WscnSnapshot:
    """The frame-thread half of a save: cheap, and reads the document once."""
    collected = _collect(doc)
    _materials, _material_index, images, _image_index = collected
    scene = scene_json(doc, collected)
    heights = None if doc.terrain is None else doc.terrain.heights
    return WscnSnapshot(scene=scene, heights=heights, images=tuple(images))


def snapshot_bytes(snap: WscnSnapshot) -> bytes:
    """The task-thread half: encode a :func:`snapshot` into a ``.wscn`` archive.

    This is the whole cost this format's writer used to spend on the frame
    thread before the split existed -- the zip container, one ``.npy`` encode
    for the terrain if there is one, and one PNG encode per texture -- run here
    against a snapshot that no longer touches the live document at all. See
    the module docstring's clay-03 paragraph for why this split is not
    optional.
    """
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo(SCENE, _EPOCH), snap.scene)
        if snap.heights is not None:
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(snap.heights, dtype=np.float32))
            zf.writestr(zipfile.ZipInfo(TERRAIN_HEIGHTS, _EPOCH), member.getvalue())
        for i, image in enumerate(snap.images):
            info = zipfile.ZipInfo(f"{TEXTURE_DIR}/{i}.png", _EPOCH)
            # Stored, not deflated: a PNG is already compressed, and deflating
            # it again spends time to make it marginally bigger.
            zf.writestr(info, _png_bytes(image), zipfile.ZIP_STORED)
    return out.getvalue()


def wscn_bytes(doc: MasonDoc) -> bytes:
    """The document as the bytes of a ``.wscn`` archive -- both halves at once,
    for a caller already off the frame thread (a test, crash recovery, a batch
    conversion). A caller *on* the frame thread wants :func:`snapshot` and
    :func:`snapshot_bytes` split across its own ``ctx.submit``, the way
    ``clay_mode``'s save paths now do for ``.wblk``.
    """
    return snapshot_bytes(snapshot(doc))


# --- reading -------------------------------------------------------------------


def _vector(entry: dict[str, Any], key: str, default: tuple[float, ...]) -> np.ndarray:
    """One fixed-length numeric field off a node entry, or a refusal by name --
    ``clay/serialize.py``'s ``_vector``, mirrored: ``np.array`` is happy to
    build a ragged or wrongly-shaped array out of whatever JSON carried, and
    every one of those would reach the renderer as a transform that does
    nothing visible and raises nothing either. A three-element rotation, for
    instance, is a quaternion with its ``w`` silently dropped -- a document
    that opens and is silently wrong is worse than one that says why it did
    not.
    """
    raw = entry.get(key, default)
    try:
        value = np.asarray(raw, dtype="f8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"a node in this mason scene has a {key} that is not numbers") from exc
    if value.shape != (len(default),):
        raise ValueError(
            f"a node in this mason scene has a {key} of {value.size} numbers, "
            f"not {len(default)}"
        )
    return value


def _float_tuple(
    entry: dict[str, Any], key: str, default: tuple[float, ...]
) -> tuple[float, ...]:
    """Like :func:`_vector`, but for a field whose class stores it as a plain
    tuple rather than an owned ``ndarray`` (``LightNode.color``, which unlike
    ``Node``'s own transform fields has no ``__post_init__`` to coerce
    whatever this hands it).
    """
    raw = entry.get(key, default)
    try:
        values = [float(v) for v in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"a node in this mason scene has a {key} that is not numbers") from exc
    if len(values) != len(default):
        raise ValueError(
            f"a node in this mason scene has a {key} of {len(values)} numbers, "
            f"not {len(default)}"
        )
    return tuple(values)


def _float_field(entry: dict[str, Any], key: str, default: float) -> float:
    """One scalar field off a node entry, or a refusal by name."""
    try:
        return float(entry.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"a node in this mason scene has a {key} that is not a number") from exc


def _properties(entry: dict[str, Any]) -> dict[str, Any]:
    """A node entry's ``properties`` field, or a refusal by name --
    ``clay/serialize.py``'s ``_params`` restated: the 2026-09-11 audit's
    clay-07 finding was a bare ``dict(entry.get("params") or {})`` reaching a
    hand-edited ``"params": 5`` as an unnamed ``TypeError`` instead of this
    module's own named refusal, and this format has the identical field under
    a different name.
    """
    properties = entry.get("properties")
    if properties is not None and not isinstance(properties, dict):
        raise ValueError("a node in this mason scene has properties that is not a mapping")
    return dict(properties or {})


def _read_ref(entry: Any) -> Ref | None:
    """A mesh node's ``ref`` entry back into a :class:`Ref`, or ``None``.

    A :class:`PrimitiveRef` is rebuilt through :func:`primitive_ref` and never
    the constructor directly, so the normalization every other caller of that
    function relies on always runs here too -- a hand-edited file with an int
    where a generator wrote a float must still compare equal to the ref a
    fresh placement would build.
    """
    if entry is None:
        return None
    if not isinstance(entry, dict):
        raise ValueError("a mesh node's ref in this mason scene is not a mapping")
    kind = entry.get("kind")
    if kind == "primitive":
        generator = entry.get("generator")
        if not isinstance(generator, str):
            raise ValueError("a primitive ref in this mason scene has no usable generator")
        params = entry.get("params")
        if not isinstance(params, dict):
            raise ValueError(
                "a primitive ref in this mason scene has a params that is not a mapping"
            )
        try:
            return primitive_ref(generator, params)
        except ValueError as exc:
            raise ValueError(f"a primitive ref in this mason scene is malformed: {exc}") from exc
    if kind == "library":
        job_id = entry.get("job_id")
        if not isinstance(job_id, str):
            raise ValueError("a library ref in this mason scene has no usable job_id")
        return LibraryRef(
            job_id=job_id,
            artifact=str(entry.get("artifact", "model.glb")),
            name=str(entry.get("name", "")),
            sha256=str(entry.get("sha256", "")),
        )
    raise ValueError(f"a mesh node's ref in this mason scene has an unknown kind: {kind!r}")


def _material_at(
    entry: dict[str, Any], key: str, materials: list[gltf.Material], *, required: bool
) -> gltf.Material | None:
    """A ``material`` index off an entry, resolved into the shared object at
    that position in ``materials`` -- never a copy (see the module docstring).

    ``required`` distinguishes the terrain's own material -- ``Terrain.material``
    has no default, a ground with no material is not a ground this app can
    draw -- from a node's *override*, which is legitimately absent.
    """
    value = entry.get(key)
    if value is None:
        if required:
            raise ValueError(f"this mason scene's terrain has no usable {key}")
        return None
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"this mason scene has a {key} index that is not a number") from exc
    if not 0 <= index < len(materials):
        raise ValueError(
            f"this mason scene names material index {index}, past the "
            f"{len(materials)} it carries"
        )
    return materials[index]


def _material_from(entry: dict[str, Any], textures: list[Any]) -> gltf.Material:
    """One material off the scene, refusing a malformed one by name --
    ``clay/serialize.py``'s ``_material_from``, mirrored: everything here is a
    cast out of JSON, wrapped once because the answer is the same whichever
    field was wrong and the user can only act on "this document is malformed"
    anyway.
    """
    try:
        slots = {}
        for slot, index in (entry.get("textures") or {}).items():
            if slot not in TEXTURE_FIELDS:
                continue
            if not 0 <= int(index) < len(textures):
                raise ValueError(
                    f"a material names texture index {index!r} for its {slot} slot, "
                    f"past the {len(textures)} this file carries"
                )
            slots[slot] = textures[int(index)]
        return gltf.Material(
            **slots,
            name=str(entry.get("name", "")),
            base_color_factor=tuple(entry.get("base_color_factor", (1.0, 1.0, 1.0, 1.0))),
            metallic_factor=float(entry.get("metallic_factor", 1.0)),
            roughness_factor=float(entry.get("roughness_factor", 1.0)),
            emissive_factor=tuple(entry.get("emissive_factor", (0.0, 0.0, 0.0))),
            double_sided=bool(entry.get("double_sided", False)),
            alpha_mode=str(entry.get("alpha_mode", "OPAQUE")),
            alpha_cutoff=float(entry.get("alpha_cutoff", 0.5)),
        )
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("a material in this mason scene is malformed") from exc


def _read_textures(zf: zipfile.ZipFile, scene: dict[str, Any]) -> list[Any]:
    """Decode every ``textures/<n>.png`` the scene names, in order -- a member
    the scene names and the archive does not carry is refused rather than
    skipped, the same half-read-is-worse-than-refused rule the terrain reader
    follows.
    """
    out = []
    for entry in scene.get("textures", []):
        name = str(entry.get("file", ""))
        try:
            raw = zf.read(name)
        except KeyError as exc:
            raise ValueError(
                f"this mason scene names a texture the file does not carry ({name})"
            ) from exc
        with pixelguard.opened(io.BytesIO(raw), f"a texture in this mason scene ({name})") as im:
            image = im.convert("RGBA")
        out.append((image.width, image.height, image.tobytes()))
    return out


def _read_terrain(
    zf: zipfile.ZipFile, entry: dict[str, Any], materials: list[gltf.Material]
) -> Terrain:
    name = entry.get("file")
    if not isinstance(name, str):
        raise ValueError("this mason scene's terrain names no height file")
    try:
        raw = zf.read(name)
    except KeyError as exc:
        raise ValueError(
            f"this mason scene names a terrain whose height data is missing ({name})"
        ) from exc
    heights = npyguard.read_array(raw, f"a terrain in this mason scene ({name})")
    try:
        size_x = float(entry["size_x"])
        size_z = float(entry["size_z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("this mason scene's terrain has no usable size") from exc
    material = _material_at(entry, "material", materials, required=True)
    # ``Terrain.__post_init__`` validates shape, side and finiteness itself
    # and raises its own named ``ValueError`` -- not re-wrapped here, since its
    # message already names exactly what is wrong.
    return Terrain(heights=heights, size_x=size_x, size_z=size_z, material=material)


def _read_node(
    entry: Any, materials: list[gltf.Material], counter: list[int], depth: int
) -> nd.Node:
    """One node entry back into its concrete class, recursing into ``children``
    first -- so the node count and depth ceilings below are checked at every
    level of the tree, not only at the root.

    ``counter`` is a one-element list shared across the whole call tree
    (scene roots and every prefab template alike) rather than a return value
    threaded back up, because a return value would only let the *caller*
    refuse after every branch below it had already been built -- the module
    docstring's "count the entries as you parse, not after building objects".
    """
    if not isinstance(entry, dict):
        raise ValueError("a node in this mason scene is not a mapping")
    counter[0] += 1
    if counter[0] > MAX_NODES:
        raise ValueError(
            f"this mason scene places more than {MAX_NODES:,} nodes, past "
            "the most Mason will open"
        )
    if depth > nd.MAX_DEPTH:
        raise ValueError(
            f"this mason scene nests nodes more than {nd.MAX_DEPTH} deep, "
            "past the most Mason will read"
        )
    kind = entry.get("kind")
    cls = _KIND_CLASSES.get(kind)
    if cls is None:
        raise ValueError(f"a node in this mason scene has an unknown kind: {kind!r}")
    try:
        uid = int(entry["uid"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("a node in this mason scene has no usable uid") from exc
    # Raise the process-wide floor now, not after the rest of the document has
    # loaded: a task thread building the remaining nodes must never mint a
    # fresh one onto a uid this file is about to restore.
    nd.reserve_uid(uid)

    children_raw = entry.get("children", [])
    if not isinstance(children_raw, list):
        raise ValueError("a node's children in this mason scene is not a list")
    children = [_read_node(child, materials, counter, depth + 1) for child in children_raw]

    kwargs: dict[str, Any] = {
        "uid": uid,
        "name": str(entry.get("name", "")),
        "translation": _vector(entry, "translation", (0.0, 0.0, 0.0)),
        "rotation": _vector(entry, "rotation", (0.0, 0.0, 0.0, 1.0)),
        "scale": _vector(entry, "scale", (1.0, 1.0, 1.0)),
        "children": children,
        "visible": bool(entry.get("visible", True)),
        "locked": bool(entry.get("locked", False)),
        "static": bool(entry.get("static", False)),
        "properties": _properties(entry),
    }

    if cls is nd.MeshNode:
        kwargs["ref"] = _read_ref(entry.get("ref"))
        kwargs["material"] = _material_at(entry, "material", materials, required=False)
    elif cls is nd.LightNode:
        light_kind = entry.get("light_kind", "point")
        if light_kind not in nd.LIGHT_KINDS:
            raise ValueError(
                f"a light in this mason scene has a kind that is not one of "
                f"{nd.LIGHT_KINDS}: {light_kind!r}"
            )
        kwargs["kind"] = light_kind
        kwargs["color"] = _float_tuple(entry, "color", (1.0, 1.0, 1.0))
        kwargs["intensity"] = _float_field(entry, "intensity", 1.0)
        kwargs["range"] = _float_field(entry, "range", 0.0)
        kwargs["inner_cone_angle"] = _float_field(entry, "inner_cone_angle", 0.0)
        kwargs["outer_cone_angle"] = _float_field(entry, "outer_cone_angle", math.pi / 4)
    elif cls is nd.CameraNode:
        kwargs["yfov"] = _float_field(entry, "yfov", math.radians(60.0))
        kwargs["znear"] = _float_field(entry, "znear", 0.1)
        kwargs["zfar"] = _float_field(entry, "zfar", 1000.0)
    elif cls is nd.PrefabNode:
        template = entry.get("template", "")
        if not isinstance(template, str):
            raise ValueError(
                "a prefab instance in this mason scene has a template that is not a string"
            )
        kwargs["template"] = template
    elif cls is nd.TerrainNode:
        kwargs["material"] = _material_at(entry, "material", materials, required=False)

    return cls(**kwargs)


def read_wscn(data: bytes) -> MasonDoc:
    """A ``.wscn``'s bytes back into a :class:`MasonDoc`.

    The returned document has an empty history and reads clean: a file that
    has just been opened is not unsaved, and every node is placed directly
    rather than through ``add_node``, which would push one undo step per node.
    """
    try:
        zf = zipguard.BoundedZip(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("this is not a Warlock Mason scene") from exc

    with zf:
        # Before anything is read: the cheapest place to refuse an archive
        # claiming more than this build will hold is before the first member
        # is decoded -- ``clay/serialize.py``'s own precheck, one dimension over.
        claimed = sum(int(info.file_size) for info in zf.infolist())
        ceiling = MAX_DECOMPRESSED_BYTES
        if claimed > ceiling:
            raise ValueError(
                f"this mason scene claims {claimed} bytes unpacked, past the "
                f"{ceiling} this build will read"
            )

        try:
            scene = json.loads(zf.read(SCENE))
        except (
            zipfile.BadZipFile,
            KeyError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError("this is not a Warlock Mason scene") from exc

        version = int(scene.get("version", 0))
        if version > VERSION:
            raise ValueError(
                f"this mason scene was written by a newer version of Warlock "
                f"(format {version}, this build reads {VERSION})"
            )

        textures = _read_textures(zf, scene)

        declared_materials = scene.get("materials", [])
        if not isinstance(declared_materials, list):
            raise ValueError("this is not a Warlock Mason scene")
        materials = [_material_from(m, textures) for m in declared_materials]

        # Shared by every node and every prefab template read below, so the
        # ceiling is one number for the whole file rather than one per branch
        # -- see ``_read_node``'s own docstring.
        counter = [0]

        declared_nodes = scene.get("nodes", [])
        if not isinstance(declared_nodes, list):
            raise ValueError("this is not a Warlock Mason scene")
        roots = [_read_node(entry, materials, counter, 0) for entry in declared_nodes]

        declared_prefabs = scene.get("prefabs", {})
        if not isinstance(declared_prefabs, dict):
            raise ValueError("this is not a Warlock Mason scene")
        prefabs = {
            str(name): _read_node(template, materials, counter, 0)
            for name, template in declared_prefabs.items()
        }

        terrain: Terrain | None = None
        terrain_entry = scene.get("terrain")
        if terrain_entry is not None:
            if not isinstance(terrain_entry, dict):
                raise ValueError("this is not a Warlock Mason scene")
            terrain = _read_terrain(zf, terrain_entry, materials)

        view = _view_json(scene.get("view"))

    doc = MasonDoc(roots=roots, materials=materials)
    doc.prefabs = prefabs
    doc.terrain = terrain
    doc.view = {} if view is None else view
    return doc
