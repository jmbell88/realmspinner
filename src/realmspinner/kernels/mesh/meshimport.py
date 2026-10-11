"""STL and PLY into a Clay document, and the one door every importable format
comes in through.

Delegated to ``trimesh`` rather than hand-rolled the way :mod:`.objimport` is:
neither STL nor PLY has an n-gon or a per-corner UV to lose on the way in
(STL has no faces beyond triangles and no texture coordinates at all; a PLY
this pipeline is asked to read is, in practice, a scan or a CAD export, also
triangle soup), so the reason :mod:`.objimport` pays for its own parser does
not apply here, and ``trimesh`` is already a lazy dependency of this package
(see ``LAZY_ONLY`` in ``tests/modes/clay/test_clay_imports.py``).

**``merge_vertices()`` is not optional.** ``trimesh.load(..., process=False)``
hands back exactly the triangle soup the file declares -- a cube as thirty-six
independent vertices, no two triangles sharing an index even where they share
an edge. Clay's smoothing, its element selection (a vertex or edge is
addressed once, not once per triangle that touches it) and its export all
assume a connected mesh, so every geometry is welded before it becomes an
``Obj``.

**Smoothing is the welded mesh's own adjacent-face angle rule.** The welded
geometry goes through :func:`.shading.auto_smooth`, the per-edge rule every
Clay insertion door uses. It does not compare a face with trimesh's averaged
vertex normals: those are the angle-weighted mean of the triangles around a
vertex, so a face touching a hard edge always agrees with them to within
``FLAT_COSINE`` and a box imported smooth (the 2026-10-10 audit's clay-16).
"""

from __future__ import annotations

import io
import struct

import numpy as np

from . import mesh as bm
from . import objimport, shading, topo
from .document import ClayDoc, Obj, new_uid
from .elements import OpError
from .glbimport import MAX_OBJECTS, MAX_TRIANGLES
from .objimport import axis_matrix

__all__ = ["SUPPORTED_SUFFIXES", "import_file", "import_file_report", "mesh_file_to_claydoc"]

#: Every suffix :func:`import_file` will route somewhere. ``.glb`` goes to
#: :mod:`.glbimport`, ``.obj`` to :mod:`.objimport`, the other two here.
SUPPORTED_SUFFIXES = (".glb", ".obj", ".stl", ".ply")

_TRIMESH_TYPE = {".stl": "stl", ".ply": "ply"}


def _geometry_mesh(geom: object, material: int = 0) -> bm.Mesh | None:
    """One trimesh geometry -> one Clay :class:`~.mesh.Mesh`, or ``None`` if empty.

    The 2026-09-26 audit, finding clay-io-12: a PLY with vertices but no
    ``element face`` at all loads through ``trimesh`` as a
    :class:`trimesh.points.PointCloud`, not a :class:`trimesh.Trimesh` --
    and a ``PointCloud`` carries no ``.faces`` attribute whatsoever, so
    ``geom.faces`` raised a bare, unnamed ``AttributeError`` instead of
    reaching this function's own "empty geometry" answer. ``getattr`` with a
    default treats a faceless point cloud exactly like a geometry with zero
    faces -- ``None`` here, which :func:`mesh_file_to_claydoc`'s caller
    already turns into its named "this file has no geometry in it" refusal
    once every geometry in the file has come back empty the same way.
    """
    faces = getattr(geom, "faces", None)
    if len(geom.vertices) == 0 or faces is None or len(faces) == 0:
        return None
    positions = np.asarray(geom.vertices, dtype="f4").reshape(-1, 3)
    # The 2026-10-03 audit, finding clay-37: checked after the f4 cast and the
    # axis/scale transform, so a finite f8 that overflows float32 is caught too.
    # ``objimport`` refuses a non-finite ``v`` line by name; STL and PLY
    # imported the NaN and left the document unable to save or round-trip.
    if not np.all(np.isfinite(positions)):
        raise OpError("This mesh has a non-finite (NaN or infinite) vertex position.")
    tris = np.asarray(geom.faces, dtype="i4").reshape(-1, 3)
    loops = tris.reshape(-1)
    starts = topo.starts_from_counts(np.full(len(tris), 3, dtype="i8"))
    mat = np.full(len(tris), int(material), dtype="i4")

    # The 2026-10-10 audit's clay-16: smoothing was decided by comparing each
    # face normal with trimesh's *welded, angle-averaged* vertex normal, and a
    # face touching a hard edge always sits within the cosine of that average
    # -- a unit box as STL/PLY imported ``smooth == [True] * 12`` and a
    # hard-surface part opened as a rounded blob, the opposite of what OBJ and
    # GLB give. The welded mesh goes through the same per-edge adjacent-face
    # angle rule every other Clay insertion door uses (a box stays flat, a
    # sphere goes smooth).
    flat = topo.rebuild(positions, loops, starts, mat, np.zeros(len(tris), dtype=bool), uv=None)
    return shading.auto_smooth(flat)


def _stl_declared_triangles(data: bytes) -> int | None:
    """A binary STL's own declared triangle count, or ``None`` if this file is
    not (recognisably) binary STL.

    clay-18 (the 2026-09-19 audit): ``mesh_file_to_claydoc`` called
    ``trimesh.load`` -- the full parse and allocation -- before checking
    ``tri_total``/object count against the ceilings below, unlike its two
    siblings (:mod:`.objimport`'s own text pre-pass, :func:`.glbimport.
    _declared_budget` off a GLB's JSON chunk), both of which refuse before the
    expensive part runs. Binary STL's format hands this over for free: an
    80-byte header immediately followed by a little-endian ``uint32`` triangle
    count, before a single 50-byte facet record -- so refusing here costs
    reading 84 bytes, not parsing the file.

    The declared count is cross-checked against the file's actual length
    (``84 + count * 50`` must account for every byte) rather than trusted on
    its own: ASCII STL also starts with a variable-length ``solid`` header
    that may happen to be at least 84 bytes long, and reading 4 arbitrary
    bytes of *that* as a triangle count would refuse a legitimate ASCII file
    over a coincidence rather than a real declared count.
    """
    if len(data) < 84:
        return None
    count = struct.unpack_from("<I", data, 80)[0]
    if 84 + count * 50 != len(data):
        return None
    return count


def _ply_declared_faces(data: bytes) -> int | None:
    """A PLY's own declared face count off its ASCII header, or ``None`` if
    the header cannot be read this cheaply.

    clay-18: a PLY's header is always plain ASCII text -- even when the body
    it precedes is ``binary_little_endian`` -- terminated by an
    ``end_header`` line, and it declares ``element face N`` before a single
    byte of geometry. Read only up to that line, capped at 64 KiB so a file
    with no ``end_header`` at all (not a PLY, or one trimesh will itself
    refuse) costs a bounded read rather than a scan of the whole file. A
    header this function cannot parse returns ``None`` rather than inventing
    a count -- trimesh, not this pre-check, is the PLY parser, and refusing a
    file over a guess would be worse than not checking at all.
    """
    head = data[:65536]
    if not head.startswith(b"ply"):
        return None
    end = head.find(b"end_header")
    if end == -1:
        return None
    try:
        text = head[:end].decode("ascii")
    except UnicodeDecodeError:
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == "face" and parts[2].isdigit():
            return int(parts[2])
    return None


def mesh_file_to_claydoc(
    data: bytes,
    suffix: str,
    name: str = "Imported",
    *,
    scale: float = 1.0,
    up: str = "y",
) -> ClayDoc:
    """Parse STL or PLY bytes into a fresh :class:`~.document.ClayDoc`.

    One :class:`~.document.Obj` per geometry trimesh hands back -- a plain STL
    or PLY is one geometry and therefore one object, and a scene-shaped file
    (a handful of exporters write a multi-object PLY/STL as a ``Scene``) still
    gets an outliner row per part rather than one merged blob.
    """
    kind = _TRIMESH_TYPE.get(suffix.lower())
    if kind is None:
        raise OpError(
            f"meshimport does not read {suffix!r} files; it reads "
            + ", ".join(sorted(_TRIMESH_TYPE))
        )
    # The 2026-10-07 audit's clay-59: a zero, negative or non-finite scale is
    # refused here, before the parse, by the one function every format's scale
    # goes through (it used to be refused for GLB alone).
    matrix = axis_matrix(scale=scale, up=up)

    # clay-18: checked before trimesh.load below -- the full parse and
    # allocation -- rather than only after it, the way objimport's text
    # pre-pass and glbimport's _declared_budget both refuse an oversized file
    # before their own expensive part runs. A file whose declared count this
    # cheaply cannot be read (an ASCII STL, or a PLY header this cannot parse)
    # falls through unchecked here, same as those two: it is bounded by
    # nothing more than the 100 MiB import-door file size, which is the case
    # the audit rated Medium rather than High -- trimesh 4.12.2's own STL/PLY
    # readers already validate a declared count against the bytes remaining,
    # so today's worst case stays proportional to that door rather than
    # amplifying past it.
    declared_tris = _stl_declared_triangles(data) if kind == "stl" else _ply_declared_faces(data)
    if declared_tris is not None and declared_tris > MAX_TRIANGLES:
        raise OpError(
            f"This {suffix.upper()} declares {declared_tris:,} triangles in its "
            f"own header, past the {MAX_TRIANGLES:,} Clay can edit."
        )

    import trimesh  # lazy: see tests/modes/clay/test_clay_imports.py's LAZY_ONLY

    try:
        loaded = trimesh.load(io.BytesIO(data), file_type=kind, process=False)
    except Exception as exc:  # noqa: BLE001 - trimesh raises several types
        raise OpError(f"This file could not be read as {suffix.upper()}. {exc}") from exc

    geoms = list(loaded.geometry.items()) if hasattr(loaded, "geometry") else [(name, loaded)]

    tri_total = 0
    for _, geom in geoms:
        tri_total += len(getattr(geom, "faces", ()))
    if tri_total > MAX_TRIANGLES:
        raise OpError(
            f"This {suffix.upper()} has {tri_total:,} triangles, past the "
            f"{MAX_TRIANGLES:,} Clay can edit."
        )
    if len(geoms) > MAX_OBJECTS:
        raise OpError(
            f"This {suffix.upper()} carries {len(geoms):,} geometries, past the "
            f"{MAX_OBJECTS:,} Clay holds."
        )

    objects: list[Obj] = []
    taken: set[str] = set()
    for geom_name, geom in geoms:
        # The 2026-10-07 audit's clay-58: a face index outside the vertex list
        # reached ``merge_vertices`` as a raw numpy IndexError (a negative one
        # imported silently, wrapping to the wrong vertex). Both are a corrupt
        # file, refused by name before anything is merged.
        faces = np.asarray(getattr(geom, "faces", ()))
        vertex_count = len(getattr(geom, "vertices", ()))
        if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= vertex_count):
            raise OpError(
                f"This {suffix.upper()} has a face that names a vertex index outside "
                f"its {vertex_count:,} vertices, so it is not a valid mesh."
            )
        geom.merge_vertices()
        geom.apply_transform(matrix)
        mesh = _geometry_mesh(geom)
        if mesh is None:
            continue
        objects.append(
            Obj(uid=new_uid(), name=objimport._unique(geom_name or name, taken), mesh=mesh)
        )

    if not objects:
        raise OpError(f"This {suffix.upper()} has no geometry in it.")
    return ClayDoc(objects=objects)


def import_file(
    data: bytes,
    suffix: str,
    name: str = "Imported",
    *,
    scale: float = 1.0,
    up: str = "y",
    mtl: str | None = None,
) -> ClayDoc:
    """:func:`import_file_report` without the lost-texture count."""
    return import_file_report(data, suffix, name, scale=scale, up=up, mtl=mtl)[0]


def import_file_report(
    data: bytes,
    suffix: str,
    name: str = "Imported",
    *,
    scale: float = 1.0,
    up: str = "y",
    mtl: str | None = None,
) -> tuple[ClayDoc, int]:
    """Route *data* to the importer for *suffix*. -> ``(document, skipped_textures)``.

    ``skipped_textures`` is how many of a GLB's textures the loader could not
    use (always 0 for the other formats, whose textures are not read); show it
    with :func:`~.glbimport.skipped_textures_notice` (the 2026-10-07 audit's
    clay-07).

    The one door a drop handler or an import dialog needs: it does not have to
    know that a GLB, an OBJ and an STL are read by three different modules with
    three different libraries behind them, only that :data:`SUPPORTED_SUFFIXES`
    names what it may hand over.

    **``scale``/``up`` apply to a GLB too**, even though a glTF file is already
    in this project's own convention and ordinarily needs neither: refusing a
    non-default request would be one more thing an import dialog has to explain,
    and applying it is exactly as well-defined here as for any other format.
    Applied after the ordinary GLB import (:func:`~.glbimport.glb_to_claydoc`)
    runs, to each object's mesh positions *and* its node transform together:
    the positions go through the matrix ``M`` and the placement is conjugated
    (``M @ local @ M^-1``), so each object's world placement is ``M`` times the
    unscaled one and a node ten metres away stays in lockstep with its mesh.
    (STL and PLY have no node transform, so for them ``M`` is applied to the
    positions alone -- :func:`.objimport.axis_matrix`'s docstring.) The 2026-10-09
    second audit's clay-116: this used to say the transform was left alone,
    which is the bug the 2026-09-26 audit's clay-io-01 fixed.

    ``mtl`` is the text of the OBJ's material library, read by the caller: this
    module never opens a file, so finding the ``.mtl`` beside an ``.obj`` is
    the job of whoever knows where the ``.obj`` came from. Ignored for every
    other format.
    """
    suf = suffix.lower()
    # The scale is checked for every format up front (clay-59), so a GLB is not
    # parsed in full only to be refused at the end.
    axis_matrix(scale=scale, up=up)
    if suf == ".obj":
        # ``utf-8-sig``: the 2026-10-07 audit's clay-15 -- a BOM that survives
        # the decode hides the file's first line (the first ``v``).
        text = data.decode("utf-8-sig", errors="replace")
        return objimport.obj_to_claydoc(text, name, mtl=mtl, scale=scale, up=up), 0
    if suf in (".stl", ".ply"):
        return mesh_file_to_claydoc(data, suf, name, scale=scale, up=up), 0
    if suf == ".glb":
        from . import glbimport

        doc, skipped = glbimport.glb_to_claydoc_report(data, name)
        if scale != 1.0 or up != "y":
            from ..geom3d import math3d as m3

            matrix = axis_matrix(scale=scale, up=up)
            # The 2026-09-26 audit, finding clay-io-01: this used to rescale
            # only each object's own mesh positions and leave its
            # translation/rotation exactly as the GLB recorded them -- a
            # child GLTF node placed ten metres from the origin stayed ten
            # metres away after a scale of 0.01 shrank its mesh to fit in a
            # centimetre, so it detached from the shape it used to sit on.
            # OBJ and STL never hit this because every object they produce
            # has an identity placement (the whole reason ``axis_matrix``'s
            # own docstring can call "positions, not the node" innocent
            # there); a GLB's nodes are not identity, so the same choice here
            # silently dropped half the transform.
            #
            # Conjugating each object's own *local* matrix by the same
            # matrix (``M @ local @ M^-1``) rather than leaving it alone
            # keeps every object's *world* matrix in exact lockstep with its
            # mesh (which is transformed by ``M`` directly, as before) --
            # true independent of parent depth: for a chain of local
            # matrices L1 L2 ... Ln, conjugating each one individually gives
            # (M L1 M^-1)(M L2 M^-1)...(M Ln M^-1) = M (L1 L2 ... Ln) M^-1,
            # so the composed world matrix is conjugated exactly the same
            # way a single root object's would be.
            try:
                inverse = np.linalg.inv(matrix)
            except np.linalg.LinAlgError as error:
                raise OpError("Import scale cannot be 0.") from error
            for obj in doc.objects:
                # The overflow is the refusal below, not a warning.
                with np.errstate(over="ignore"):
                    obj.mesh = bm.transformed(obj.mesh, matrix)
                # The 2026-10-07 audit, finding clay-19: ``glb_to_claydoc``
                # refuses a node transform that overflows float32, but this
                # knob multiplies the positions afterwards, so a mesh near
                # float32's ceiling times a large scale came out as inf
                # vertices in a document that could not save or draw.
                if not np.isfinite(np.asarray(obj.mesh.positions, dtype="f4")).all():
                    raise OpError(
                        "The import scale makes this model too large for Clay's "
                        "float32 positions. Use a smaller scale."
                    )
                local = m3.compose(obj.translation, obj.rotation, obj.scale)
                # The 2026-10-09 audit's clay-26: only the positions were
                # checked above, so a node translation near float32's ceiling
                # times a large scale slipped through as an object placed at
                # inf, which OBJ export then wrote as ``v inf``. The placement
                # is refused by name the same way the positions are.
                with np.errstate(over="ignore", invalid="ignore"):
                    t, r, s = m3.decompose(matrix @ local @ inverse)
                    fits = all(
                        np.isfinite(np.asarray(part, dtype="f4")).all() for part in (t, r, s)
                    )
                if not fits:
                    raise OpError(
                        "The import scale moves a node too far for Clay's float32 "
                        "positions. Use a smaller scale."
                    )
                obj.translation, obj.rotation, obj.scale = t, r, s
        return doc, skipped
    raise OpError(
        f"Clay does not import {suffix!r} files. It reads "
        + ", ".join(SUPPORTED_SUFFIXES) + "."
    )
