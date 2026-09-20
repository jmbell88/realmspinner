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

**Smoothing tries the geometric rule first.** ``trimesh`` computes a vertex
normal for every welded vertex (the angle-weighted average of its triangles'
face normals) whether or not the source file carried one of its own -- STL has
none, PLY sometimes does -- so :mod:`.glbimport`'s own test (does every corner
of a face agree with the face's own normal to within ``FLAT_COSINE``) is
*always* reachable here, unlike glTF's optional ``NORMAL`` attribute. It still
has a fallback, :func:`.shading.auto_smooth`, for the one case trimesh itself
cannot answer: a mesh with zero faces, or one degenerate enough that a normal
comes back non-finite.
"""

from __future__ import annotations

import io
import struct

import numpy as np

from . import mesh as bm
from . import objimport, shading, topo
from .document import ClayDoc, Obj, new_uid
from .elements import OpError
from .glbimport import FLAT_COSINE, MAX_OBJECTS, MAX_TRIANGLES, _unit
from .objimport import axis_matrix

__all__ = ["SUPPORTED_SUFFIXES", "import_file", "mesh_file_to_claydoc"]

#: Every suffix :func:`import_file` will route somewhere. ``.glb`` goes to
#: :mod:`.glbimport`, ``.obj`` to :mod:`.objimport`, the other two here.
SUPPORTED_SUFFIXES = (".glb", ".obj", ".stl", ".ply")

_TRIMESH_TYPE = {".stl": "stl", ".ply": "ply"}


def _smooth_from_vertex_normals(
    positions: np.ndarray, tris: np.ndarray, vertex_normals: np.ndarray
) -> np.ndarray:
    """:func:`.glbimport._smooth_flags`'s exact test, keyed by welded vertex
    rather than by a primitive's raw (pre-merge) normal stream.

    After :meth:`trimesh.Trimesh.merge_vertices`, one normal exists per vertex
    rather than per corner, so every corner of a triangle that shares a vertex
    reads the same supplied normal -- which is what makes this the same
    question glbimport asks of a glTF primitive's ``NORMAL`` attribute, over a
    differently-shaped input.
    """
    n_faces = len(tris)
    if n_faces == 0:
        return np.zeros(0, dtype=bool)
    corners = positions[tris].astype("f8")
    face = _unit(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]))
    supplied = _unit(vertex_normals[tris].reshape(-1, 3)).reshape(-1, 3, 3)
    agreement = np.einsum("ijk,ik->ij", supplied, face).min(axis=1)
    return agreement <= FLAT_COSINE


def _geometry_mesh(geom: object, material: int = 0) -> bm.Mesh | None:
    """One trimesh geometry -> one Clay :class:`~.mesh.Mesh`, or ``None`` if empty."""
    if len(geom.vertices) == 0 or len(geom.faces) == 0:
        return None
    positions = np.asarray(geom.vertices, dtype="f4").reshape(-1, 3)
    tris = np.asarray(geom.faces, dtype="i4").reshape(-1, 3)
    loops = tris.reshape(-1)
    starts = topo.starts_from_counts(np.full(len(tris), 3, dtype="i8"))
    mat = np.full(len(tris), int(material), dtype="i4")

    smooth = None
    try:
        normals = np.asarray(geom.vertex_normals, dtype="f8")
        if normals.shape == positions.shape and np.all(np.isfinite(normals)):
            smooth = _smooth_from_vertex_normals(positions.astype("f8"), tris, normals)
    except Exception:  # noqa: BLE001 - trimesh's own normal computation can raise
        smooth = None

    if smooth is not None:
        return topo.rebuild(positions, loops, starts, mat, smooth, uv=None)
    # Not reachable (an empty or degenerate geometry never gets a usable
    # normal): fall back to the same rule every primitive generator's flat
    # default goes through before it is asked to look smooth.
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

    matrix = axis_matrix(scale=scale, up=up)

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
    """Route *data* to the importer for *suffix*. -> a fresh :class:`~.document.ClayDoc`.

    The one door a drop handler or an import dialog needs: it does not have to
    know that a GLB, an OBJ and an STL are read by three different modules with
    three different libraries behind them, only that :data:`SUPPORTED_SUFFIXES`
    names what it may hand over.

    **``scale``/``up`` apply to a GLB too**, even though a glTF file is already
    in this project's own convention and ordinarily needs neither: refusing a
    non-default request would be one more thing an import dialog has to explain,
    and applying it is exactly as well-defined here as for any other format.
    Applied to each object's own mesh positions after the ordinary GLB import
    (:func:`~.glbimport.glb_to_claydoc`) runs, rather than composed into the
    object's transform -- the same "positions, not the node" choice
    :func:`.objimport.axis_matrix`'s docstring makes for STL/PLY, so all three
    formats agree about what the knob does.

    ``mtl`` is the text of the OBJ's material library, read by the caller: this
    module never opens a file, so finding the ``.mtl`` beside an ``.obj`` is
    the job of whoever knows where the ``.obj`` came from. Ignored for every
    other format.
    """
    suf = suffix.lower()
    if suf == ".obj":
        text = data.decode("utf-8", errors="replace")
        return objimport.obj_to_claydoc(text, name, mtl=mtl, scale=scale, up=up)
    if suf in (".stl", ".ply"):
        return mesh_file_to_claydoc(data, suf, name, scale=scale, up=up)
    if suf == ".glb":
        from . import glbimport

        doc = glbimport.glb_to_claydoc(data, name)
        if scale != 1.0 or up != "y":
            matrix = axis_matrix(scale=scale, up=up)
            for obj in doc.objects:
                obj.mesh = bm.transformed(obj.mesh, matrix)
        return doc
    raise OpError(
        f"Clay does not import {suffix!r} files. It reads "
        + ", ".join(SUPPORTED_SUFFIXES) + "."
    )
