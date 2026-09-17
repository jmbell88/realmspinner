"""``.wblk`` -- the Clay document on disk, as a zip.

A zip of ``scene.json`` plus one ``meshes/<uid>.npz`` per object, mirroring the
raster editor's ``.ora`` in shape and for the same reason: the small,
human-meaningful half of the document is text that a person can read and a diff
can show, and the large numeric half is stored in a form that does not swell by
a factor of ten on the way through JSON. A subdivided mesh written as
nested float lists would be a megabyte of ``0.7071067811865476``; the same
arrays in an npz are their own bytes.

**Every timestamp in the archive is fixed.** A zip stamps each member with the
wall clock, and an npz is itself a zip, so a document that has not changed would
otherwise produce different bytes every time it was written -- which makes the
file undiffable, makes a content hash useless and makes "has this actually
changed" unanswerable outside the app. Both levels are written at the epoch the
zip format starts at, so two saves of an unchanged document are byte-identical.

**A half-read document is worse than a refused one.** Two failures are caught
rather than papered over: a file written by a newer version, and a scene naming
a mesh the archive does not carry. Substituting an empty mesh for a missing one
would open the file, show the user an object with nothing in it, and let them
save that over their work.

The reader restores a document by *construction* rather than through
``add_object``, which would push one undo step per object and open every file
already dirty; and it raises the process-wide uid floor as it goes, so a fresh
object cannot be minted onto a uid a restored one already wears.

**Version 2 adds uvs and textures**, and is written unconditionally -- there is
no "downgrade if the document happens to have neither", because a format that
sometimes claims to be v1 and sometimes v2 for the same code is a format with
two readers. A v1 file still opens: its meshes get ``uv=None`` and its
materials no textures, which is exactly what a v1 document was.

Textures are ``textures/<n>.png`` members, PNG-encoded and stored *uncompressed*
in the zip -- a PNG is already deflated, and deflating it again costs time to
make it very slightly bigger. They are deduplicated by the identity of the
``(width, height, bytes)`` tuple across all five slots of all materials, which
is the same identity rule ``GpuMaterial``, ``to_model`` and ``write_glb`` all
de-duplicate on: a document whose eight objects share one baked base-colour map
writes one PNG. A ``scene.json`` for an untextured document is v1-shaped apart
from the version number, so the readable half stays readable.

The PNG encode -- 0.2--0.5 s for a 2K map -- and the mesh zip together are what
:func:`snapshot_bytes` runs. **They no longer run on the frame thread**: the
2026-09-06 audit (clay-03) found ``clay_mode``'s own save and export paths
calling :func:`wblk_bytes` -- this whole module's cost -- directly, before
``ctx.submit`` ever ran, which is exactly the frame-thread stall this format's
byte-identity work was supposed to make affordable to pay for, not free to
skip. :func:`snapshot` is the cheap half taken on the frame thread; only its
already-copied text and already-built PNG tuples reach :func:`snapshot_bytes`,
which any caller off the frame thread -- a task closure, a test, a batch
conversion -- is free to call together as :func:`wblk_bytes` still does.

**A missing texture member is refused**, exactly as a missing mesh is, and for
the same reason: opening the file with a blank material would show the user a
model that looks finished and is not, and let them save it over their work.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np

from ...core.safeio import npyguard, pixelguard, zipguard
from ..geom3d import gltf
from . import mesh as bm
from .document import ClayDoc, Obj, reserve_uid

VERSION = 2
SCENE = "scene.json"
MESH_DIR = "meshes"
TEXTURE_DIR = "textures"

# 1980-01-01, the earliest a zip can express. Any fixed value would do; this one
# is the conventional choice for a reproducible archive and is obviously not a
# real modification time, which is the point -- nobody should read it as one.
_EPOCH = (1980, 1, 1, 0, 0, 0)

# The five arrays every mesh has. ``uv`` is deliberately *not* here: it is
# optional, so it is written only when present and defaulted to None on read,
# which is what lets a v1 file load without a migration.
_MESH_FIELDS = ("positions", "loops", "starts", "material", "smooth")

# A zip's directory declares what each member unpacks to, and nothing makes that
# number honest -- a few kilobytes of archive can claim terabytes, and the read
# that discovers this is the one that has already exhausted memory. One gigabyte
# is far past any document this editor produces (a million-face mesh is tens of
# megabytes) and far short of a machine's RAM, so the ceiling is only ever hit
# by a file that was not written by us. Read from module globals at call time so
# a test can lower it.
MAX_DECOMPRESSED_BYTES = 1 << 30

# The 2026-09-14 audit's clay-06: read_wblk bounded the objects array
# (MAX_OBJECTS, imported below) but not how many materials or textures a
# scene declared -- both are cheap in bytes, a few characters of JSON per
# entry, so a compact hand-edited or crash-recovered archive naming a huge
# count of either stalled the load in the loop that decodes it (_read_textures
# below, and the materials list comprehension in read_wblk) with nothing to
# refuse it up front the way the object count already is. Sized far past
# anything Clay itself ever writes: a document never has more materials than
# MAX_OBJECTS objects, or more textures than materials times
# ``len(TEXTURE_FIELDS)`` slots, before the by-identity dedup that usually
# shrinks it further.
MAX_DECLARED_MATERIALS = 100_000
MAX_DECLARED_TEXTURES = 100_000

# The texture slots, in the order ``scene.TEXTURE_SLOTS`` lists them. Mirrored
# rather than imported because ``clay/`` does not import the GL layer -- and the
# names are a *file format* here, so pinning them locally is what stops a
# rename in the renderer silently changing what a saved document means.
TEXTURE_FIELDS = (
    "base_color",
    "metallic_roughness",
    "normal",
    "emissive",
    "occlusion",
)


# --- writing ----------------------------------------------------------------


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """A ``.npz`` built member by member, so its timestamps are ours.

    ``np.savez`` would be one line, but it stamps each member with the wall
    clock and there is no argument that turns that off. An npz *is* a zip of
    ``.npy`` members, so writing it here costs a few lines and buys the
    byte-identity the whole format is claimed to have. ``np.load`` reads the
    result with no idea it was not written by numpy.
    """
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(array))
            zf.writestr(zipfile.ZipInfo(f"{name}.npy", _EPOCH), member.getvalue())
    return out.getvalue()


def _material_json(
    material: gltf.Material, textures: dict[int, int] | None = None
) -> dict[str, Any]:
    """The factors, plus the index of each texture slot that has one.

    Clay paints no textures, but it now *imports* them, and an imported asset
    that lost its baked maps on the way through a save would be worse than one
    that could not be imported at all. The slot map is omitted entirely when
    there are none, so an authored document's JSON is v1-shaped.
    """
    slots = {}
    if textures is not None:
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


def _object_json(obj: Obj) -> dict[str, Any]:
    return {
        "uid": int(obj.uid),
        "name": obj.name,
        "translation": [float(v) for v in obj.translation],
        "rotation": [float(v) for v in obj.rotation],  # XYZW, as everywhere
        "scale": [float(v) for v in obj.scale],
        "generator": obj.generator,
        "params": obj.params,
        "visible": bool(obj.visible),
        "material": int(obj.material),
    }


def _collect_textures(doc: ClayDoc) -> tuple[list[Any], dict[int, int]]:
    """Every distinct texture in the palette, and a map from identity to index.

    Identity rather than value: a texture is a multi-megabyte bytes object and
    comparing two of them per material per save is the wrong price for the same
    answer. It is also the rule every other consumer already uses.
    """
    images: list[Any] = []
    index: dict[int, int] = {}
    for material in doc.materials:
        for slot in TEXTURE_FIELDS:
            image = getattr(material, slot, None)
            if image is None or id(image) in index:
                continue
            index[id(image)] = len(images)
            images.append(image)
    return images, index


def _png_bytes(image: tuple[int, int, bytes]) -> bytes:
    """One ``(width, height, rgba)`` slot as PNG bytes.

    ``optimize`` is left off deliberately: it is several times slower for a few
    per cent, and this runs on the frame thread.
    """
    from PIL import Image

    width, height, data = image
    out = io.BytesIO()
    Image.frombytes("RGBA", (int(width), int(height)), bytes(data)).save(out, "PNG")
    return out.getvalue()


#: The camera keys ``scene.json`` carries, and the only ones it will read back.
#: Written out rather than taken from whatever dict a caller hands in, because
#: this is a *file format*: an unrecognised key would round-trip once and then
#: be silently dropped by the next build that read the table instead.
VIEW_FIELDS = ("yaw", "pitch", "distance")


def view_json(view: Any) -> dict[str, Any] | None:
    """One camera as the JSON ``scene["view"]`` holds, or ``None`` for no camera.

    Additive and therefore **not a version bump**: a build that has never heard
    of it reads the file exactly as it did, and one that has reads a camera. The
    same argument ``rig.json``'s ``fit`` key is added under.
    """
    if view is None:
        return None
    try:
        out: dict[str, Any] = {name: float(getattr(view, name)) for name in VIEW_FIELDS}
        out["target"] = [float(v) for v in view.target]
    except (AttributeError, TypeError, ValueError):
        return None
    return out


def read_view(data: bytes) -> dict[str, Any] | None:
    """The camera out of a ``.wblk``, or ``None`` if it has none it can trust.

    **A second function rather than a second return value from**
    :func:`read_wblk`, which is the same call ``files.unready_reason`` makes:
    that reader's job is to hand back the engine's own document type, and a
    camera is not part of one -- ``ClayDoc`` is geometry and a palette, and
    where somebody last left the viewport is a property of the *tab*. Widening
    the return would make every caller unpack a pair to ignore half of it, and
    putting the camera on the document would put it in the undo stack.

    Every way of being wrong answers ``None`` rather than raising. It is read on
    the open path beside a document that has already parsed, so a malformed
    camera is worth one unfitted viewport and never a refused file.
    """
    try:
        with zipguard.BoundedZip(io.BytesIO(data)) as zf:
            scene = json.loads(zf.read(SCENE))
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    entry = scene.get("view")
    if not isinstance(entry, dict):
        return None
    try:
        out: dict[str, Any] = {name: float(entry[name]) for name in VIEW_FIELDS}
        target = [float(v) for v in entry["target"]]
    except (KeyError, TypeError, ValueError):
        return None
    if len(target) != 3 or not all(np.isfinite([*out.values(), *target])):
        return None
    out["target"] = tuple(target)
    return out


def scene_json(
    doc: ClayDoc,
    collected: tuple[list[Any], dict[int, int]] | None = None,
    *,
    view: Any = None,
) -> str:
    """``scene.json``'s text: sorted keys, indented, one object per entry.

    Sorted and indented rather than compact because this half of the file
    exists to be *read* -- by a person looking at why a document opens wrong,
    and by a diff. The mesh arrays are the reason the format is a zip; there is
    no size argument left for minifying the small half.

    ``collected`` is :func:`_collect_textures`' answer, passed in by the writer
    because it needs the same one to decide which PNGs to store: the two must
    agree about *which* texture is index 3, and a second walk was both a wasted
    pass over the palette and a second place that could answer differently.
    Left None -- which is what a caller wanting only the JSON does -- it is
    worked out here.
    """
    images, index = _collect_textures(doc) if collected is None else collected
    scene: dict[str, Any] = {
        "version": VERSION,
        "materials": [_material_json(m, index) for m in doc.materials],
        "objects": [_object_json(o) for o in doc.objects],
    }
    if images:
        scene["textures"] = [
            {"file": f"{TEXTURE_DIR}/{i}.png", "width": int(w), "height": int(h)}
            for i, (w, h, _data) in enumerate(images)
        ]
    camera = view_json(view)
    if camera is not None:
        scene["view"] = camera
    return json.dumps(scene, sort_keys=True, indent=2)


class WblkSnapshot:
    """The frame-thread half of :func:`wblk_bytes` -- cheap, and safe to encode
    later on another thread.

    The 2026-09-06 audit (clay-03) found ``clay_mode.save_to``/``save_as``/
    ``export_asset`` calling ``wblk_bytes`` itself before ``ctx.submit`` ran --
    the zip-and-PNG encode this module's own docstring says runs on the frame
    thread, done on it every time regardless. ``wpack.Snapshot`` already drew
    this line for Packwright's atlas and this mirrors it exactly: everything
    that is *text* is finished here, and everything that is *bytes to encode*
    is deferred.

    ``scene`` is the already-built ``scene.json`` string: assembling it walks
    the document's own lists once (materials, per-object floats) and allocates
    nothing large, so there is no reason to defer it, and doing it here rather
    than in :func:`snapshot_bytes` means the task thread never reads
    ``doc.materials`` or ``doc.objects`` at all -- both are plain lists the
    document appends to and pops from on every edit, with no lock around
    either.

    ``meshes`` holds each object's mesh **by reference**, paired with the uid
    that names its archive member. That is sound under INVARIANTS 312: a
    ``Mesh`` is an immutable CSR array -- every array is copied and frozen at
    construction, and every op is ``Mesh -> Mesh`` -- so a reference taken here
    keeps meaning exactly what it meant when the snapshot was built, however
    many edits land on the *live* document before the task thread gets to it.

    ``images`` is the same reference-holding, for the same reason: each entry
    is an already-built ``(width, height, bytes)`` tuple that is never mutated
    once a material holds it -- only replaced, wholesale, by a new tuple.
    """

    __slots__ = ("scene", "meshes", "images")

    def __init__(
        self, scene: str, meshes: tuple[tuple[int, bm.Mesh], ...], images: tuple[Any, ...]
    ) -> None:
        self.scene = scene
        self.meshes = meshes
        self.images = images


def snapshot(doc: ClayDoc, *, view: Any = None) -> WblkSnapshot:
    """The frame-thread half of a save: cheap, and reads the document once.

    ``view`` is consumed here too -- ``view_json`` turns whatever camera object
    a caller hands in into a plain dict before this returns, so nothing about
    the snapshot depends on that object staying alive or unchanged.
    """
    collected = _collect_textures(doc)
    images, _index = collected
    scene = scene_json(doc, collected, view=view)
    meshes = tuple((int(obj.uid), obj.mesh) for obj in doc.objects)
    return WblkSnapshot(scene=scene, meshes=meshes, images=tuple(images))


def snapshot_bytes(snap: WblkSnapshot) -> bytes:
    """The task-thread half: encode a :func:`snapshot` into a ``.wblk`` archive.

    This is the whole cost :func:`wblk_bytes` used to spend on the frame
    thread -- the zip container, one npz build per mesh, and one PNG encode
    per texture -- run here against a snapshot that no longer touches the live
    document at all.
    """
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo(SCENE, _EPOCH), snap.scene)
        for uid, mesh in snap.meshes:
            arrays = {name: getattr(mesh, name) for name in _MESH_FIELDS}
            if mesh.uv is not None:
                arrays["uv"] = mesh.uv
            zf.writestr(
                zipfile.ZipInfo(f"{MESH_DIR}/{uid}.npz", _EPOCH),
                _npz_bytes(arrays),
            )
        for i, image in enumerate(snap.images):
            info = zipfile.ZipInfo(f"{TEXTURE_DIR}/{i}.png", _EPOCH)
            # Stored, not deflated: a PNG is already compressed, and deflating
            # it again spends time to make it marginally bigger.
            zf.writestr(info, _png_bytes(image), zipfile.ZIP_STORED)
    return out.getvalue()


def wblk_bytes(doc: ClayDoc, *, view: Any = None) -> bytes:
    """The document as the bytes of a ``.wblk`` archive.

    ``view`` is where the viewport was left, and it is optional at every call
    site: a document written without one is byte-for-byte the file this wrote
    before the key existed, which is what keeps the format additive rather than
    versioned.

    Both halves at once, for callers that are already off the frame thread --
    a test, a batch conversion, ``clay_mode``'s own crash-recovery reader. A
    caller *on* the frame thread wants :func:`snapshot` and :func:`snapshot_bytes`
    split across its own ``ctx.submit``, which is what ``clay_mode.save_to``,
    ``save_as`` and ``export_asset`` now do.
    """
    return snapshot_bytes(snapshot(doc, view=view))


# --- reading ----------------------------------------------------------------


def _read_mesh(zf: zipfile.ZipFile, uid: int) -> bm.Mesh:
    name = f"{MESH_DIR}/{uid}.npz"
    try:
        raw = zf.read(name)
    except KeyError as exc:
        raise ValueError(
            f"this clay document names an object whose mesh is missing ({name})"
        ) from exc
    # Through ``npyguard`` and not ``np.load``. Two holes, one call: ``np.load``
    # on an npz opens a **nested zip with numpy's own plain ``zipfile``**, so
    # the ``BoundedZip`` this document was opened through is not in the path at
    # all past the outer member; and each inner ``.npy`` is sized from its own
    # 128-byte header, which nothing looked at before the allocation. The
    # object-array refusal ``allow_pickle=False`` bought is still made, from
    # that header, by name.
    npz = npyguard.read_npz(raw, f"a mesh in this clay document ({name})")
    try:
        arrays = {field: npz[field] for field in _MESH_FIELDS}
    except KeyError as exc:
        raise ValueError(f"a mesh in this clay document is incomplete: {exc}") from exc
    # Optional, and absent from every v1 file: a mesh without it simply has
    # no texture coordinates, which is what a v1 mesh was.
    if "uv" in npz:
        arrays["uv"] = npz["uv"]
    mesh = bm.Mesh(**arrays)
    # Validate rather than trust: the CSR offsets are the one part of the file
    # that a truncated write or a hand edit makes *quietly* wrong -- ``edges``
    # and ``face_normals`` do not raise on a bad ``starts``, they produce
    # nonsense. Better to refuse the file than to render it.
    bm.validate(mesh)
    return mesh


def _read_textures(zf: zipfile.ZipFile, scene: dict[str, Any]) -> list[Any]:
    """Decode every ``textures/<n>.png`` the scene names, in order.

    A member the scene names and the archive does not carry is refused rather
    than skipped -- the half-read-is-worse-than-refused rule the mesh reader
    follows, and more sharply here, because a material with its map silently
    dropped looks like a deliberately untextured one.
    """
    out = []
    for entry in scene.get("textures", []):
        name = str(entry.get("file", ""))
        try:
            raw = zf.read(name)
        except KeyError as exc:
            raise ValueError(
                f"this clay document names a texture the file does not carry ({name})"
            ) from exc
        # In a ``with``: ``Image.open`` is lazy and holds the file object open
        # until it is closed, and a document with twenty textures would
        # otherwise leave twenty of them to the garbage collector. The ``with``
        # is ``pixelguard``'s now, which is also where the pixel ceiling is
        # asked -- before ``convert``, because that is the call that allocates.
        with pixelguard.opened(io.BytesIO(raw), f"a texture in this clay document ({name})") as im:
            image = im.convert("RGBA")
        out.append((image.width, image.height, image.tobytes()))
    return out


def _vector(entry: dict[str, Any], key: str, default: tuple[float, ...]) -> Any:
    """One fixed-length numeric field off an object entry, or a refusal.

    ``np.array`` is happy to build a 2x3 array, an array of strings or a ragged
    object array out of whatever JSON carried, and every one of those reaches
    the renderer as a transform that produces nothing visible and no error. The
    length is the part worth checking: a three-element rotation is a quaternion
    with its w dropped, which is a document that opens and is silently wrong.
    """
    raw = entry.get(key, default)
    try:
        value = np.asarray(raw, dtype="f8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"an object in this clay document has a {key} that is not numbers"
        ) from exc
    if value.shape != (len(default),):
        raise ValueError(
            f"an object in this clay document has a {key} of "
            f"{value.size} numbers, not {len(default)}"
        )
    return value


def _params(entry: dict[str, Any]) -> dict[str, Any]:
    """An object entry's ``params`` field, or a refusal by name.

    The 2026-09-11 audit, finding clay-07: this used to be
    ``dict(entry.get("params") or {})`` with no check at all, unlike every
    other field on this same entry (the uid, each ``_vector``, each
    material's own fields) -- this module's stated rule, restated in
    ``read_wblk``'s own docstring, is "a half-read document is worse than a
    refused one". A hand-edited or corrupted archive with ``"params": 5`` or
    ``"params": [1, 2, 3]`` reached ``dict(...)`` as a bare, un-messaged
    ``TypeError`` instead of this reader's named refusal, and this file is
    also what crash recovery reads with no user to ask first.
    """
    params = entry.get("params")
    if params is not None and not isinstance(params, dict):
        raise ValueError(
            "an object in this clay document has a params that is not a mapping"
        )
    return dict(params or {})


def _material_index(entry: dict[str, Any]) -> int:
    """An object entry's ``material`` field, or a refusal by name.

    The 2026-09-11 audit, finding clay-04: this used to be a bare
    ``int(entry.get("material", 0))``, unlike every sibling field on this
    same entry (the uid, each ``_vector``, ``params``, each material's own
    fields), which this module hardens into its own named refusal. ``None``,
    a list or a dict reached ``int()`` as an unnamed ``TypeError`` and a
    string reached it as ``int()``'s own message, both in place of this
    reader's "this is not a Warlock Clay document" sentence. Unlike a mesh's
    per-face material index, an out-of-range value here is not a fresh kind
    of corruption: ``ClayDoc`` already draws a face pointing off the end of
    the palette in ``FALLBACK_MATERIAL`` rather than refusing, so this reader
    matches that and only refuses a value that is not a number at all.
    """
    material = entry.get("material", 0)
    try:
        return int(material)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "an object in this clay document has a material that is not a number"
        ) from exc


def _generator(entry: dict[str, Any]) -> str | None:
    """An object entry's ``generator`` field, or a refusal by name.

    The 2026-09-11 audit, finding clay-06: this used to be a bare
    ``entry.get("generator")`` with no type check at all, so a non-string,
    non-``None`` value loaded cleanly and only failed later, far from this
    file, in the properties panel's ``GENERATORS[obj.generator]`` lookup.
    This module does not import ``primitives`` to check the name against
    ``GENERATORS`` -- that would be a cycle -- so it only checks the shape
    every legitimate value has ever had: ``None``, or a string.
    """
    generator = entry.get("generator")
    if generator is not None and not isinstance(generator, str):
        raise ValueError(
            "an object in this clay document has a generator that is not "
            "a string or null"
        )
    return generator


def _material_from(entry: dict[str, Any], textures: list[Any]) -> gltf.Material:
    """One material off the scene, refusing a malformed one by name.

    Everything here is a cast of a value out of JSON, so ``null`` where a number
    belongs -- or a texture index that is a string -- arrives as a ``TypeError``
    from deep inside ``float()`` with no mention of the file. The whole thing is
    wrapped once rather than field by field: the answer is the same for all of
    them, and the user can only act on "this document is malformed" anyway.
    """
    try:
        slots = {}
        for slot, index in (entry.get("textures") or {}).items():
            if slot not in TEXTURE_FIELDS:
                continue
            # The 2026-09-08 audit's clay-05: an index past the end of the
            # textures this archive actually decoded used to be silently
            # dropped -- that slot came back ``None`` -- rather than refused
            # the way a texture member missing from the zip entirely already
            # is (see ``_read_textures`` above). Both are the same shape of
            # corruption ``read_wblk``'s crash-recovery path can hand this
            # reader, so both refuse: this raise is caught by the ``except``
            # below, the same door the rest of a malformed material uses.
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
        raise ValueError("a material in this clay document is malformed") from exc


def read_wblk(data: bytes) -> ClayDoc:
    """A ``.wblk``'s bytes back into a :class:`ClayDoc`.

    The returned document has an empty history and reads clean: a file that has
    just been opened is not unsaved, and the objects are placed directly rather
    than through ``add_object``, which would push a step apiece.
    """
    try:
        zf = zipguard.BoundedZip(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("this is not a Warlock Clay document") from exc

    # Everything that can raise is inside the ``with``, the version check
    # included. It used to sit in the gap between the open and the ``with``,
    # where a refusal -- the one thing that block exists to do -- left the
    # archive open with nothing to close it but the collector.
    with zf:
        # Before anything is read: the directory says what every member unpacks
        # to, and the cheapest place to refuse an archive that claims more than
        # this build will hold is before the first ``read``.
        claimed = sum(int(info.file_size) for info in zf.infolist())
        ceiling = MAX_DECOMPRESSED_BYTES
        if claimed > ceiling:
            raise ValueError(
                f"this clay document claims {claimed} bytes unpacked, "
                f"past the {ceiling} this build will read"
            )

        try:
            scene = json.loads(zf.read(SCENE))
        except (
            zipfile.BadZipFile,
            KeyError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError("this is not a Warlock Clay document") from exc

        version = int(scene.get("version", 0))
        if version > VERSION:
            raise ValueError(
                f"this clay document was written by a newer version of Warlock "
                f"(format {version}, this build reads {VERSION})"
            )

        # The 2026-09-07 audit's clay-04: ``claimed`` above bounds the archive's
        # *decompressed bytes*, but a scene of many small objects is cheap in
        # bytes and expensive in what opening it does next -- ``glbimport`` has
        # ``MAX_OBJECTS``/``MAX_TRIANGLES`` for exactly this shape of file and
        # this door had no equivalent. 20,000 objects (4.9x ``MAX_OBJECTS``, 4%
        # of the byte ceiling) opened in 8.9s with no warning, which crash
        # recovery hits with no user to ask first. The object count is free --
        # it is the length of a list already parsed out of ``scene`` -- so it
        # is checked before a single mesh member is read; the triangle count
        # is not knowable without reading every mesh, so it is checked as the
        # loop goes, the same way ``_declared_budget`` lets a GLB's own claim
        # stand in for one where it can.
        from .glbimport import MAX_OBJECTS, MAX_TRIANGLES

        declared = scene.get("objects", [])
        # The 2026-09-08 audit's clay-05: a hand-edited or partially corrupted
        # ``.wblk`` whose "objects" field is present but not a list used to
        # reach ``len(declared)`` below and raise a bare ``TypeError`` --
        # unlike every other malformed field in this reader (the uid, each
        # vector, each material's own fields), which this module deliberately
        # catches and turns into the same named refusal. ``clay_mode._load``
        # has no wrapping try/except, so the uncaught TypeError reached the
        # user as the generic "Something went wrong" instead of this file's
        # own "this is not a Warlock Clay document" sentence.
        if not isinstance(declared, list):
            raise ValueError("this is not a Warlock Clay document")
        if len(declared) > MAX_OBJECTS:
            raise ValueError(
                f"this clay document places {len(declared):,} objects, past "
                f"the {MAX_OBJECTS:,} Clay holds"
            )

        declared_textures = scene.get("textures", [])
        # Same clay-06 gap as the objects/materials counts: a "textures" field
        # present but not a list would otherwise reach ``len`` below (or
        # ``_read_textures``'s own loop) as a bare, unnamed failure instead of
        # this reader's refusal.
        if not isinstance(declared_textures, list):
            raise ValueError("this is not a Warlock Clay document")
        if len(declared_textures) > MAX_DECLARED_TEXTURES:
            raise ValueError(
                f"this clay document names {len(declared_textures):,} textures, "
                f"past the {MAX_DECLARED_TEXTURES:,} Clay reads"
            )

        textures = _read_textures(zf, scene)
        objects = []
        triangles = 0
        seen_uids: set[int] = set()
        for entry in declared:
            # The uid is the one field with no defensible default -- it names the
            # mesh member and it is what undo addresses -- so a missing or
            # non-numeric one is a refusal rather than the bare ``KeyError`` or
            # ``TypeError`` this used to hand the mode layer.
            try:
                uid = int(entry["uid"])
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError(
                    "an object in this clay document has no usable uid"
                ) from exc
            # The 2026-09-16 audit: every other field on this entry is checked
            # for the shape a legitimate writer could never produce, but
            # uniqueness across the whole document was not -- a scene.json
            # declaring two objects with the same uid used to load cleanly,
            # both objects reading whichever archive member ``meshes/<uid>.npz``
            # happens to hold (silently losing one object's geometry), and then
            # leaving ``by_uid``/``index_of``/``set_props``/``remove_object``
            # addressing an arbitrary one of the two for the rest of the
            # session. This is exactly the "half-read document is worse than a
            # refused one" rule the rest of this reader follows, and this file
            # is also what crash recovery reads with no user to ask first. The
            # check runs before ``_read_mesh`` so the refusal names the real
            # problem rather than surfacing however that read happens to land.
            if uid in seen_uids:
                raise ValueError(
                    f"this clay document has two objects sharing uid {uid}"
                )
            seen_uids.add(uid)
            reserve_uid(uid)
            mesh = _read_mesh(zf, uid)
            # The 2026-09-08 audit's second run, clay-09: this used to add
            # ``len(mesh.starts) - 1`` -- the *face* count -- toward
            # ``MAX_TRIANGLES``. The CSR format stores n-gons untriangulated, so
            # a single 3,000,000-corner face counted as 1 here, passed the
            # ceiling in well under a second, and then cost 216s the first time
            # something fan-triangulated it (``triangulate``, and every render
            # pass). Counting ``starts[i+1] - starts[i] - 2`` per face is the
            # number of triangles `mesh.triangulate` actually produces for a
            # convex fan; a degenerate face with fewer than 3 corners
            # contributes 0 rather than a negative count.
            counts = np.diff(mesh.starts).astype("i8") - 2
            triangles += int(np.clip(counts, 0, None).sum())
            if triangles > MAX_TRIANGLES:
                raise ValueError(
                    f"this clay document has more than {MAX_TRIANGLES:,} "
                    "triangles, the most Clay can edit"
                )
            objects.append(
                Obj(
                    uid=uid,
                    name=str(entry.get("name", "")),
                    mesh=mesh,
                    translation=_vector(entry, "translation", (0.0, 0.0, 0.0)),
                    rotation=_vector(entry, "rotation", (0.0, 0.0, 0.0, 1.0)),
                    scale=_vector(entry, "scale", (1.0, 1.0, 1.0)),
                    generator=_generator(entry),
                    params=_params(entry),
                    visible=bool(entry.get("visible", True)),
                    material=_material_index(entry),
                )
            )

    # A scene with objects but no materials is a hand-edited or truncated
    # file: every face's material index would fall off the empty palette,
    # rendering and exporting magenta and crashing the properties panel.
    # ClayDoc substitutes the default palette for None, so hand it None --
    # but only when there are objects, because an empty scene legitimately
    # round-trips its empty palette.
    declared_materials = scene.get("materials", [])
    # Same clay-05 gap, the other field this reader forgot to guard: a
    # "materials" entry that is present but not a list used to fail the list
    # comprehension below with a bare TypeError instead of this reader's own
    # named refusal.
    if not isinstance(declared_materials, list):
        raise ValueError("this is not a Warlock Clay document")
    if len(declared_materials) > MAX_DECLARED_MATERIALS:
        raise ValueError(
            f"this clay document declares {len(declared_materials):,} materials, "
            f"past the {MAX_DECLARED_MATERIALS:,} Clay reads"
        )
    materials = [_material_from(m, textures) for m in declared_materials]
    if objects and not materials:
        materials = None
    return ClayDoc(objects=objects, materials=materials)
