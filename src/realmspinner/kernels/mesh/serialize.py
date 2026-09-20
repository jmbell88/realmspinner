"""``.rblk`` -- the Clay document on disk, as a zip.

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
calling :func:`rblk_bytes` -- this whole module's cost -- directly, before
``ctx.submit`` ever ran, which is exactly the frame-thread stall this format's
byte-identity work was supposed to make affordable to pay for, not free to
skip. :func:`snapshot` is the cheap half taken on the frame thread; only its
already-copied text and already-built PNG tuples reach :func:`snapshot_bytes`,
which any caller off the frame thread -- a task closure, a test, a batch
conversion -- is free to call together as :func:`rblk_bytes` still does.

**A missing texture member is refused**, exactly as a missing mesh is, and for
the same reason: opening the file with a blank material would show the user a
model that looks finished and is not, and let them save it over their work.

**Version 3 adds modifier stacks.** Each object entry gains an optional
``"modifiers"`` list -- omitted entirely for an object with none, which keeps
the common document's JSON v2-shaped apart from the version number, the same
rule the ``"textures"`` key already follows for an untextured one. Every
value is read back through :mod:`.modifiers`' own registry (:func:`~.
modifiers.make`), so a hand-edited or out-of-range number loads at the same
clamp a live edit would produce rather than as a document holding a number
nothing else in the app can. An unknown kind, a ``"modifiers"`` that is not a
list, two modifiers on one object sharing an id, or an unknown parameter name
is refused -- the half-read-is-worse-than-refused rule every other field on
this entry already follows. A v1 or v2 file still opens: an object with no
``"modifiers"`` key simply has none, exactly as one with no ``"uv"`` has none.

**Version 3 also carries scene structure, without a second version bump.**
Version 3 was unreleased at the time -- see ``dev/CLAY-PLAN.md`` -- so
``"parent"`` (an object uid or absent, meaning a root), ``"locked"`` (absent
means ``False``) and ``"tags"`` (absent means none) join it the same way
``"modifiers"`` did: each omitted at its default, which keeps an ordinary
document's JSON exactly as small as it always was. A ``"parent"`` naming a
uid this file does not carry, or a cycle anywhere in the whole document's
parenting (only detectable once every object has been read), is refused by
name -- the same half-read-is-worse-than-refused rule, checked once over the
whole set rather than per entry, because a forward reference to an object
later in the file is not thereby invalid.

**Version 3, again: tranches 6 and 7's own fields join it the same way.**
Still no fourth version -- v3 remained unreleased through both -- so
``"seams"``, ``"role"`` and ``"collider_kind"`` are each omitted at their
default (no seams, a "mesh" role, an empty kind) exactly like ``"parent"``/
``"locked"``/``"tags"`` above. A seam pair naming a vertex the *object's own*
mesh does not have (checked against that object's already-read mesh, not the
document as a whole -- a seam is local to one object), an unrecognised role,
a ``"collider_kind"`` that names nothing in :data:`~.colliders.
COLLIDER_KINDS`, or one present on an object whose role is not "collider" at
all, are each refused by name -- the same half-read-is-worse-than-refused
rule every field on this entry already follows, because a document that
opened with a role or a kind nothing else in the app can produce is worse
than one that did not open.
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
from .colliders import COLLIDER_KINDS
from .document import ClayDoc, Obj, _normalize_seams, _normalize_tags, reserve_uid

VERSION = 3
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

# The 2026-09-14 audit's clay-06: read_rblk bounded the objects array
# (MAX_OBJECTS, imported below) but not how many materials or textures a
# scene declared -- both are cheap in bytes, a few characters of JSON per
# entry, so a compact hand-edited or crash-recovered archive naming a huge
# count of either stalled the load in the loop that decodes it (_read_textures
# below, and the materials list comprehension in read_rblk) with nothing to
# refuse it up front the way the object count already is. Sized far past
# anything Clay itself ever writes: a document never has more materials than
# MAX_OBJECTS objects, or more textures than materials times
# ``len(TEXTURE_FIELDS)`` slots, before the by-identity dedup that usually
# shrinks it further.
MAX_DECLARED_MATERIALS = 100_000
MAX_DECLARED_TEXTURES = 100_000

# The 2026-09-18 audit's clay-01: MAX_DECOMPRESSED_BYTES bounds the archive's
# stored/decompressed zip bytes and MAX_DECLARED_TEXTURES bounds how many
# textures a scene may *name*, and ``pixelguard`` caps each individual
# texture's own pixel count at decode time -- but nothing bounded the *sum*
# of decoded bytes across every texture one document declares, the way
# ``gltf.MAX_TOTAL_BYTES`` bounds the same sum for a GLB (H01). A small,
# highly-compressible ``.rblk`` -- a handful of solid-colour PNGs -- decodes
# to hundreds of MB to GB from a few hundred KB on disk and passed every
# existing check, because ``_read_textures`` decoded every declared texture
# in one pass with no running total. Sized the way ``gltf.MAX_TOTAL_BYTES``
# is: comfortably above what Clay itself ever writes and well short of
# exhausting memory. Read from module globals at call time so a test can
# lower it, the same as MAX_DECOMPRESSED_BYTES above.
MAX_TOTAL_TEXTURE_BYTES = 768 * (1 << 20)

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


def _modifier_json(m: Any) -> dict[str, Any]:
    return {"id": int(m.id), "kind": str(m.kind), "enabled": bool(m.enabled), "params": m.as_dict()}


def _object_json(obj: Obj) -> dict[str, Any]:
    entry: dict[str, Any] = {
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
    if obj.modifiers:
        entry["modifiers"] = [_modifier_json(m) for m in obj.modifiers]
    # Tranche 3: scene structure. Omitted at the default -- a root, unlocked,
    # untagged object -- the same "v1/v2-shaped JSON for the common case"
    # rule ``modifiers``/``textures`` already follow.
    if obj.parent is not None:
        entry["parent"] = int(obj.parent)
    if obj.locked:
        entry["locked"] = True
    if obj.tags:
        entry["tags"] = list(obj.tags)
    # Tranches 6/7: same "omitted at the default" rule -- see this module's
    # own docstring paragraph for both.
    if obj.seams:
        entry["seams"] = [[int(a), int(b)] for a, b in obj.seams]
    if obj.role != "mesh":
        entry["role"] = obj.role
    if obj.collider_kind:
        entry["collider_kind"] = obj.collider_kind
    return entry


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
    """The camera out of a ``.rblk``, or ``None`` if it has none it can trust.

    **A second function rather than a second return value from**
    :func:`read_rblk`, which is the same call ``files.unready_reason`` makes:
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


class RblkSnapshot:
    """The frame-thread half of :func:`rblk_bytes` -- cheap, and safe to encode
    later on another thread.

    The 2026-09-06 audit (clay-03) found ``clay_mode.save_to``/``save_as``/
    ``export_asset`` calling ``rblk_bytes`` itself before ``ctx.submit`` ran --
    the zip-and-PNG encode this module's own docstring says runs on the frame
    thread, done on it every time regardless. ``rpack.Snapshot`` already drew
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


def snapshot(doc: ClayDoc, *, view: Any = None) -> RblkSnapshot:
    """The frame-thread half of a save: cheap, and reads the document once.

    ``view`` is consumed here too -- ``view_json`` turns whatever camera object
    a caller hands in into a plain dict before this returns, so nothing about
    the snapshot depends on that object staying alive or unchanged.
    """
    collected = _collect_textures(doc)
    images, _index = collected
    scene = scene_json(doc, collected, view=view)
    meshes = tuple((int(obj.uid), obj.mesh) for obj in doc.objects)
    return RblkSnapshot(scene=scene, meshes=meshes, images=tuple(images))


def snapshot_bytes(snap: RblkSnapshot) -> bytes:
    """The task-thread half: encode a :func:`snapshot` into a ``.rblk`` archive.

    This is the whole cost :func:`rblk_bytes` used to spend on the frame
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


def rblk_bytes(doc: ClayDoc, *, view: Any = None) -> bytes:
    """The document as the bytes of a ``.rblk`` archive.

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
    # The 2026-09-18 audit's clay-01: a running total against
    # MAX_TOTAL_TEXTURE_BYTES, charged from each image's already-known
    # ``width``/``height`` -- ``Image.open`` is lazy, so both are available
    # before the ``convert`` call that actually allocates the decoded bytes.
    spent = 0
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
            spent += im.width * im.height * 4
            if spent > MAX_TOTAL_TEXTURE_BYTES:
                raise ValueError(
                    f"this clay document's decoded texture bytes pass the "
                    f"{MAX_TOTAL_TEXTURE_BYTES:,} byte budget a document may spend"
                )
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
    ``read_rblk``'s own docstring, is "a half-read document is worse than a
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
    reader's "this is not a Realmspinner Clay document" sentence. Unlike a mesh's
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


def _modifiers_from(entry: dict[str, Any]) -> tuple[Any, ...]:
    """An object entry's ``modifiers`` field, or a refusal by name.

    Absent entirely -- every v1 and v2 file, and a v3 object with no stack --
    reads as ``()``, which is the field's own default and what a v2 object
    always meant. Present, each item is rebuilt through :mod:`.modifiers`'
    own :func:`~.modifiers.make`, so a value past a parameter's ``low``/
    ``high`` loads at the clamp exactly as a live edit would produce, rather
    than as a document holding a number nothing else in the app can.
    """
    from dataclasses import replace as _replace

    from . import elements as el
    from . import modifiers as mod

    raw = entry.get("modifiers")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("an object in this clay document has a modifiers that is not a list")
    seen_ids: set[int] = set()
    out = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("an object in this clay document has a malformed modifier")
        try:
            mid = int(item["id"])
            kind = str(item["kind"])
            enabled = bool(item.get("enabled", True))
            params = item.get("params") or {}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "an object in this clay document has a malformed modifier"
            ) from exc
        if not isinstance(params, dict):
            raise ValueError(
                "an object in this clay document has a modifier params that is not a mapping"
            )
        if mid in seen_ids:
            raise ValueError(
                f"this clay document has two modifiers sharing id {mid} on one object"
            )
        seen_ids.add(mid)
        try:
            built = mod.make(kind, params, id=mid)
        except el.OpError as exc:
            raise ValueError(
                f"an object in this clay document has a malformed modifier: {exc}"
            ) from exc
        out.append(built if enabled else _replace(built, enabled=False))
    return tuple(out)


def _parent_from(entry: dict[str, Any]) -> int | None:
    """An object entry's ``parent`` field, or a refusal by name.

    ``None`` -- every v1/v2 file, and a v3 root -- reads as ``None``, the
    field's own default. Existence (does this uid actually appear in the
    file) and cycle-freedom are checked once, over the *whole* document, by
    :func:`_validate_hierarchy` after every object has been read -- a
    per-entry check here could only see uids already read, and a forward
    reference to an object defined later in the file would wrongly look
    malformed.
    """
    parent = entry.get("parent")
    if parent is None:
        return None
    try:
        return int(parent)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "an object in this clay document has a parent that is not a number or null"
        ) from exc


def _locked_from(entry: dict[str, Any]) -> bool:
    return bool(entry.get("locked", False))


def _tags_from(entry: dict[str, Any]) -> tuple[str, ...]:
    """An object entry's ``tags`` field, or a refusal by name.

    Run back through :func:`~.document._normalize_tags` -- the same
    sorted/deduplicated/lower-cased shape a live tag edit already produces
    -- so a hand-edited file naming ``["Prop", "prop"]`` loads exactly as
    typing both into the tag editor would have left it, rather than as a
    document holding a duplicate nothing else in the app can produce.
    """
    raw = entry.get("tags")
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(t, str) for t in raw):
        raise ValueError(
            "an object in this clay document has a tags that is not a list of strings"
        )
    return _normalize_tags(raw)


def _seams_from(entry: dict[str, Any], mesh: bm.Mesh) -> tuple[tuple[int, int], ...]:
    """An object entry's ``seams`` field, or a refusal by name.

    Absent reads as ``()``, the field's own default -- every v1/v2 file, and
    a v3 object nothing ever marked a seam on. Present, each pair is run back
    through :func:`~.document._normalize_seams` -- the same ordered/
    deduplicated shape a live :meth:`~.document.ClayDoc.set_seams` call
    already produces, the same reason :func:`_tags_from` re-normalizes tags
    -- and then range-checked against *this object's own mesh*, already read
    by the time this runs (``read_rblk`` calls this right after
    :func:`_read_mesh`): a seam is local to one object, so it is that
    object's own vertex count this checks against, never the document's. What
    normalization cannot fix -- a vertex this mesh does not have -- is
    refused, the half-read-is-worse-than-refused rule every other field on
    this entry already follows.
    """
    raw = entry.get("seams")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("an object in this clay document has a seams that is not a list")
    try:
        pairs = [(int(item[0]), int(item[1])) for item in raw]
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise ValueError(
            "an object in this clay document has a seam that is not a pair of numbers"
        ) from exc
    normalized = _normalize_seams(pairs)
    n = len(mesh.positions)
    for a, b in normalized:
        if not (0 <= a < n and 0 <= b < n):
            raise ValueError(
                f"this clay document names a seam ({a}, {b}) past the {n} vertices "
                "its mesh carries"
            )
    return normalized


def _role_from(entry: dict[str, Any]) -> str:
    """An object entry's ``role`` field, or a refusal by name.

    Absent reads as ``"mesh"``, the field's own default -- every v1/v2 file,
    and every v3 object before tranche 7. This module does not import
    :class:`~.document.Obj` as a live source of truth for the two names it
    accepts (that would only be circular for the sake of not spelling out two
    literals) -- the same "known shape, not a live registry" choice
    :func:`_generator` already makes rather than checking against
    ``primitives.GENERATORS``.
    """
    role = entry.get("role", "mesh")
    if role not in ("mesh", "collider"):
        raise ValueError(f"an object in this clay document has an unknown role {role!r}")
    return role


def _collider_kind_from(entry: dict[str, Any], role: str) -> str:
    """An object entry's ``collider_kind`` field, or a refusal by name.

    Only meaningful when *role* (already read by :func:`_role_from`) is
    "collider": refused there if it names nothing in :data:`~.colliders.
    COLLIDER_KINDS`, the same half-read rule every other field on this entry
    follows. Refused just as firmly the other way -- a non-empty
    ``collider_kind`` on a "mesh"-role object -- because :class:`~.document.
    Obj`'s own field comment states "else empty" as part of what the field
    *means*; a live edit can never produce that combination (:meth:`~.
    document.ClayDoc.add_collider` always sets both together), so silently
    round-tripping it would let this reader load a state nothing else in the
    app can.
    """
    kind = entry.get("collider_kind", "")
    if role == "collider":
        if not isinstance(kind, str) or kind not in COLLIDER_KINDS:
            raise ValueError(
                f"this clay document names a collider kind {kind!r} that does not exist"
            )
        return kind
    if kind:
        raise ValueError(
            "an object in this clay document has a collider_kind but its role is not "
            "\"collider\""
        )
    return ""


def _validate_hierarchy(objects: list[Obj]) -> None:
    """Refuse a ``parent`` naming an absent uid, or a cycle anywhere in the
    whole document's parenting -- by name, over the whole set at once.

    Run once, after every object in the file has been read (not per entry,
    on the way through: a ``parent`` naming an object defined *later* in the
    file is a forward reference, not a defect, and this is the only point at
    which every uid the file carries is known). The half-read-is-worse-
    than-refused rule every other field on this entry already follows,
    applied to the one field whose validity depends on the rest of the file
    rather than on itself alone.
    """
    by_uid = {obj.uid: obj for obj in objects}
    for obj in objects:
        if obj.parent is not None and obj.parent not in by_uid:
            raise ValueError(
                f"this clay document names {obj.name!r}'s parent as uid {obj.parent}, "
                "which this file does not carry"
            )

    # The same white/gray/black DFS ``modifiers.would_cycle`` uses for its
    # own dependency graph -- a cycle here is unreachable through any live
    # edit (``document.ClayDoc.set_parent`` refuses one going forward), so
    # the only way one can appear is a hand-edited or corrupted file.
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(by_uid, WHITE)

    def visit(uid: int) -> bool:
        color[uid] = GRAY
        parent = by_uid[uid].parent
        if parent is not None:
            if color[parent] == GRAY:
                return True
            if color[parent] == WHITE and visit(parent):
                return True
        color[uid] = BLACK
        return False

    if any(color[uid] == WHITE and visit(uid) for uid in by_uid):
        raise ValueError("this clay document's objects form a parenting cycle")


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
            # corruption ``read_rblk``'s crash-recovery path can hand this
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


def read_rblk(data: bytes) -> ClayDoc:
    """A ``.rblk``'s bytes back into a :class:`ClayDoc`.

    The returned document has an empty history and reads clean: a file that has
    just been opened is not unsaved, and the objects are placed directly rather
    than through ``add_object``, which would push a step apiece.
    """
    try:
        zf = zipguard.BoundedZip(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("this is not a Realmspinner Clay document") from exc

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
            raise ValueError("this is not a Realmspinner Clay document") from exc

        version = int(scene.get("version", 0))
        if version > VERSION:
            raise ValueError(
                f"this clay document was written by a newer version of Realmspinner "
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
        # ``.rblk`` whose "objects" field is present but not a list used to
        # reach ``len(declared)`` below and raise a bare ``TypeError`` --
        # unlike every other malformed field in this reader (the uid, each
        # vector, each material's own fields), which this module deliberately
        # catches and turns into the same named refusal. ``clay_mode._load``
        # has no wrapping try/except, so the uncaught TypeError reached the
        # user as the generic "Something went wrong" instead of this file's
        # own "this is not a Realmspinner Clay document" sentence.
        if not isinstance(declared, list):
            raise ValueError("this is not a Realmspinner Clay document")
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
            raise ValueError("this is not a Realmspinner Clay document")
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
            # Read once, ahead of the constructor call below: collider_kind's
            # own validity depends on role, and a keyword argument list is no
            # place to guarantee that ordering or avoid asking _role_from
            # twice for the same entry.
            role = _role_from(entry)
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
                    modifiers=_modifiers_from(entry),
                    parent=_parent_from(entry),
                    locked=_locked_from(entry),
                    tags=_tags_from(entry),
                    # Tranche 6/7: seams are checked against *this* object's
                    # own mesh, already read above.
                    seams=_seams_from(entry, mesh),
                    role=role,
                    collider_kind=_collider_kind_from(entry, role),
                )
            )

        # Every uid in the file is known only once the loop above has read
        # them all -- see ``_validate_hierarchy``'s own docstring for why a
        # forward-referencing ``parent`` cannot be checked per entry.
        _validate_hierarchy(objects)

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
        raise ValueError("this is not a Realmspinner Clay document")
    if len(declared_materials) > MAX_DECLARED_MATERIALS:
        raise ValueError(
            f"this clay document declares {len(declared_materials):,} materials, "
            f"past the {MAX_DECLARED_MATERIALS:,} Clay reads"
        )
    materials = [_material_from(m, textures) for m in declared_materials]
    if objects and not materials:
        materials = None
    return ClayDoc(objects=objects, materials=materials)
