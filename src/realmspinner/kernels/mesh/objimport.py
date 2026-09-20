"""Wavefront OBJ into a Clay document.

Hand-rolled rather than delegated to ``trimesh`` -- unlike :mod:`.meshimport`'s
STL/PLY path, which has no n-gons or per-corner UVs to lose, an OBJ can carry
both, and ``trimesh.load`` triangulates on the way in and welds a UV seam's two
corners into one vertex on the way out. Clay's own :class:`~.mesh.Mesh` is a
face-corner CSR built exactly to keep both, so this parser reads the format by
hand instead of paying for the loss twice.

**Materials come from the ``mtl`` argument, not from ``mtllib``.** An OBJ's own
``mtllib`` line names an external ``.mtl`` file, and resolving that path is a
filesystem operation this pure kernel does not perform -- the caller (an import
dialog, a drop handler) already has the file bytes if there are any and passes
them straight through. ``mtllib`` itself is parsed and ignored.

**Smoothing is `` s`` groups, not compared normals.** :mod:`.glbimport` derives
a smooth flag by measuring how far a glTF file's own vertex normals disagree
with the geometric one, because glTF has no smoothing concept of its own to
read. OBJ does: the ``s`` statement is exactly "smooth from here", and reading
it is simpler and no less correct than re-deriving the same fact from ``vn``
lines that may not even be present. A mesh with several numbered smoothing
groups (``s 1``, ``s 2``, ...) folds every non-``off`` group to ``smooth=True``,
because :class:`~.mesh.Mesh` has one boolean per face and no notion of "smooth
with this face but not that one" -- the same limitation :mod:`.shading`'s
``auto_smooth`` documents for itself.

**UVs are all-or-nothing per object.** A corner with no ``vt`` in its face
token turns off ``uv`` for the whole object it belongs to, rather than writing
a zero for that one corner: a mesh half-textured by omission is a worse lie
than a mesh not textured at all, and :class:`~.mesh.Mesh` has nowhere to record
"this corner in particular has no UV".

**glTF convention throughout**: metres, Y up, Z toward the viewer.  OBJ's own
convention has *no* v-flip -- its texture origin is bottom-left, glTF's is
top-left -- so every ``vt`` read here is stored as ``1 - v``, and
:mod:`.objexport` flips back on the way out. ``up="z"`` rotates a Z-up file
(``(x, y, z) -> (x, z, -y)``) into this convention; ``scale`` multiplies
positions. See :func:`axis_matrix`, the one helper :mod:`.meshimport` also
calls so every importable format agrees about what "up" and "scale" mean.
"""

from __future__ import annotations

import math

import numpy as np

from ..geom3d import gltf
from . import topo
from .document import ClayDoc, Obj, default_material, new_uid
from .elements import OpError
from .glbimport import MAX_OBJECTS, MAX_TRIANGLES

#: Ceiling on total declared ``v``/``vt`` line count, folded into the same
#: pre-pass as ``tri_budget``/``object_budget`` below (H01, the same reasoning
#: :func:`.glbimport._declared_budget` states for a GLB's JSON chunk).
#:
#: The 2026-09-19 audit, finding clay-05: that pre-pass bounded the file's
#: declared *triangle* and *object* counts but never its raw ``v``/``vt`` line
#: count, so an OBJ with many vertex lines and almost no faces passed it and
#: was parsed into Python lists unbounded -- ``tri_budget`` never sees an
#: unreferenced vertex, and neither does any ceiling downstream of it.
#: Reproduced (and re-measured at fix time): 200,000 unreferenced ``v`` lines
#: behind one face cost 5.11 s and a 44.5 MB traced heap against 2.8 MB of
#: source (15.9x); the amplification held constant through 2,000,000 lines
#: (446.3 MB against 28 MB of source), so extrapolated to the 100 MiB
#: ``MAX_MESH_BYTES`` import-door ceiling that is ~1.55 GiB -- with nothing in
#: pass 1 to stop it. Same value as ``MAX_TRIANGLES``: a legitimate mesh
#: already declares a comparable number of ``v``/``vt`` lines when its
#: vertices are actually referenced by that many triangles, so this bounds
#: only the unreferenced-vertex case the audit found, at the measured 446 MB
#: worst case, without moving the ceiling any file this pipeline already
#: accepts relies on. Counting itself stays cheap regardless of file size --
#: measured at 2.6 s for the full pre-pass (line join plus counting, no float
#: parsing) at 7,500,000 lines, comfortably inside the 100 MiB door -- which is
#: the whole point: the expensive part (pass 2, below) never starts for a file
#: over this line.
MAX_VERTEX_LINES = MAX_TRIANGLES

__all__ = ["axis_matrix", "obj_to_claydoc", "ns_from_roughness", "roughness_from_ns"]


# --- the Wavefront <-> glTF roughness convention -----------------------------
#
# OBJ's ``Ns`` is a Phong specular exponent with no defined relationship to a
# PBR roughness; this project needs *some* invertible mapping so a round trip
# through :mod:`.objexport` does not drift, and picks one that is exact in
# both directions for roughness in [0, 1]: Ns = ((1 - r) ** 2) * 1000, so
# r = 1 - sqrt(Ns / 1000). Neither function claims to match any renderer's own
# Ns convention -- there isn't a standard one -- only each other.


def roughness_from_ns(ns: float) -> float:
    """The inverse of :func:`ns_from_roughness`, clamped to a legal roughness."""
    return max(0.0, min(1.0, 1.0 - math.sqrt(max(0.0, ns) / 1000.0)))


def ns_from_roughness(roughness: float) -> float:
    """Wavefront's ``Ns`` for a glTF roughness factor. See the module docstring."""
    r = max(0.0, min(1.0, roughness))
    return ((1.0 - r) ** 2) * 1000.0


def axis_matrix(*, scale: float, up: str) -> np.ndarray:
    """A 4x4 linear transform: ``up``'s axis convention, then uniform ``scale``.

    ``up="z"`` is the one non-identity case: ``(x, y, z) -> (x, z, -y)`` turns a
    Z-up file into glTF's Y up / Z toward viewer. That matrix's determinant is
    +1 -- a rotation, not a mirror -- so :func:`~.mesh.transformed` (what
    :mod:`.meshimport` runs this through for STL/PLY/GLB) never has to reverse a
    winding on its account; ``scale`` alone would only ever be non-negative in
    practice, but a caller who passes one is not refused here.
    """
    if up not in ("y", "z"):
        raise OpError(f"Unknown up-axis {up!r}; import expects 'y' or 'z'.")
    m = np.eye(4, dtype="f8")
    if up == "z":
        m[:3, :3] = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    m[:3, :3] *= float(scale)
    return m


# --- MTL, the small subset this parser reads ---------------------------------


def _parse_mtl(text: str) -> dict[str, dict[str, object]]:
    """``{material name: {"Kd": (r,g,b), "d": alpha, "Ns": ns}}``, best-effort.

    Every field is optional and every unrecognised statement (``map_Kd``, ``Ka``,
    an illumination model...) is skipped -- this reads exactly the three fields
    :mod:`.objexport` writes and nothing else, because nothing else has anywhere
    to go in a :class:`~gltf.Material`.
    """
    materials: dict[str, dict[str, object]] = {}
    current: dict[str, object] | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        key = parts[0]
        if key == "newmtl" and len(parts) >= 2:
            current = {}
            materials[" ".join(parts[1:])] = current
            continue
        if current is None:
            continue
        try:
            if key == "Kd" and len(parts) >= 4:
                current["Kd"] = tuple(float(v) for v in parts[1:4])
            elif key == "d" and len(parts) >= 2:
                current["d"] = float(parts[1])
            elif key == "Tr" and len(parts) >= 2 and "d" not in current:
                # ``Tr`` is the complement of ``d``; only honoured when ``d``
                # itself was not also given, since a file naming both means ``d``.
                current["d"] = 1.0 - float(parts[1])
            elif key == "Ns" and len(parts) >= 2:
                current["Ns"] = float(parts[1])
        except ValueError:
            continue  # a non-numeric value on one of these lines: skip it
    return materials


def _material_from_mtl(name: str, entry: dict[str, object] | None) -> gltf.Material:
    if entry is None:
        return default_material(name or "Material")
    kd = entry.get("Kd", (0.8, 0.8, 0.8))
    alpha = float(entry.get("d", 1.0))  # type: ignore[arg-type]
    ns = entry.get("Ns")
    roughness = 0.6 if ns is None else roughness_from_ns(float(ns))  # type: ignore[arg-type]
    return gltf.Material(
        name=name,
        base_color_factor=(*(float(c) for c in kd), alpha),  # type: ignore[misc]
        metallic_factor=0.0,
        roughness_factor=roughness,
    )


# --- parsing ------------------------------------------------------------------


def _unique(name: str, taken: set[str]) -> str:
    """``Box`` -> ``Box.001``, the same counting rule :func:`.glbimport._unique` uses."""
    if name not in taken:
        taken.add(name)
        return name
    for n in range(1, len(taken) + 2):
        candidate = f"{name}.{n:03d}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate
    return name  # pragma: no cover - unreachable, the loop is bounded above


def _joined_lines(text: str) -> list[str]:
    """Comments stripped, blank lines dropped, ``\\``-continuations joined."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    joined: list[str] = []
    buf = ""
    for raw_line in normalized.split("\n"):
        stripped = raw_line.split("#", 1)[0].rstrip()
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
            continue
        line = (buf + stripped).strip()
        buf = ""
        if line:
            joined.append(line)
    if buf.strip():
        joined.append(buf.strip())
    return joined


def _resolve_index(raw: str, count: int, what: str) -> int:
    try:
        idx = int(raw)
    except ValueError as exc:
        raise OpError(f"OBJ has a malformed {what} index {raw!r}") from exc
    if idx > 0:
        resolved = idx - 1
    elif idx < 0:
        resolved = count + idx
    else:
        raise OpError(f"OBJ has a zero {what} index, which is not valid.")
    if not 0 <= resolved < count:
        raise OpError(
            f"OBJ face references {what} {idx}, out of range for the "
            f"{count} defined by that point in the file."
        )
    return resolved


def _parse_corner(tok: str, v_count: int, vt_count: int) -> tuple[int, int | None]:
    parts = tok.split("/")
    if len(parts) not in (1, 2, 3) or not parts[0]:
        raise OpError(f"OBJ has a malformed face token {tok!r}")
    v = _resolve_index(parts[0], v_count, "vertex")
    vt = None
    if len(parts) >= 2 and parts[1]:
        vt = _resolve_index(parts[1], vt_count, "texture coordinate")
    # parts[2], the normal index, is read for shape only -- see the module
    # docstring on why smoothing comes from ``s`` groups instead.
    return v, vt


class _Building:
    """One in-progress object: local vertices, corners and per-face state."""

    __slots__ = (
        "name",
        "global_to_local",
        "positions",
        "loops",
        "corner_uv",
        "counts",
        "material",
        "smooth",
        "uv_ok",
    )

    def __init__(self, name: str) -> None:
        self.name = name
        self.global_to_local: dict[int, int] = {}
        self.positions: list[list[float]] = []
        self.loops: list[int] = []
        self.corner_uv: list[tuple[float, float]] = []
        self.counts: list[int] = []
        self.material: list[int] = []
        self.smooth: list[bool] = []
        self.uv_ok = True

    def local_index(self, global_v: int, position: list[float]) -> int:
        local = self.global_to_local.get(global_v)
        if local is None:
            local = len(self.positions)
            self.positions.append(position)
            self.global_to_local[global_v] = local
        return local


def obj_to_claydoc(
    text: str,
    name: str = "Imported",
    *,
    mtl: str | None = None,
    scale: float = 1.0,
    up: str = "y",
) -> ClayDoc:
    """Parse Wavefront OBJ text into a fresh :class:`~.document.ClayDoc`.

    ``mtl`` is that OBJ's ``.mtl`` file's *text*, if the caller has one --
    resolving ``mtllib`` is not this function's job, see the module docstring.
    ``scale``/``up`` go through :func:`axis_matrix`; applied here to each raw
    vertex as it is read, rather than to the assembled mesh afterward, because
    an OBJ vertex is a plain triple with no other transform to compose against.
    """
    if not text.strip():
        raise OpError("This OBJ is empty.")

    matrix = axis_matrix(scale=scale, up=up)[:3, :3]
    lines = _joined_lines(text)

    # --- pass 1: the declared triangle, object *and* vertex/texcoord-line
    # budget, before a single array is built (H01, the same reasoning
    # glbimport._declared_budget states for a GLB's JSON) -- clay-05 (the
    # 2026-09-19 audit) found this comment was only two-thirds true: "v"/"vt"
    # lines were counted nowhere here, so they reached pass 2 regardless of
    # how many of them the file declared.
    tri_budget = 0
    object_budget = 1  # the implicit first object, before any "o"/"g" line
    vertex_budget = 0
    for line in lines:
        head = line.split(None, 1)[0]
        if head == "f":
            corners = len(line.split()) - 1
            tri_budget += max(0, corners - 2)
        elif head in ("o", "g"):
            object_budget += 1
        elif head in ("v", "vt"):
            vertex_budget += 1
    if tri_budget > MAX_TRIANGLES:
        raise OpError(
            f"This OBJ has {tri_budget:,} triangles (n-gons counted by corners "
            f"minus two), past the {MAX_TRIANGLES:,} Clay can edit."
        )
    if object_budget > MAX_OBJECTS:
        raise OpError(
            f"This OBJ declares at least {object_budget:,} objects, past the "
            f"{MAX_OBJECTS:,} Clay holds."
        )
    if vertex_budget > MAX_VERTEX_LINES:
        raise OpError(
            f"This OBJ declares {vertex_budget:,} vertex/texture-coordinate "
            f"lines, past the {MAX_VERTEX_LINES:,} Clay will parse before it "
            "has seen how many of them a face actually uses."
        )

    # --- pass 2: the real parse.
    mtl_materials = _parse_mtl(mtl) if mtl else {}
    materials: list[gltf.Material] = []
    palette: dict[str, int] = {}

    def material_index(mat_name: str | None) -> int:
        key = mat_name or ""
        if key not in palette:
            palette[key] = len(materials)
            materials.append(_material_from_mtl(key, mtl_materials.get(key)))
        return palette[key]

    positions_all: list[list[float]] = []
    texcoords_all: list[tuple[float, float]] = []
    objects: list[Obj] = []
    taken: set[str] = set()
    current = _Building(name)
    current_material = material_index(None)
    current_smooth = False
    saw_face = False

    def flush() -> None:
        nonlocal current
        if current.loops:
            mesh = topo.rebuild(
                np.asarray(current.positions, dtype="f4").reshape(-1, 3),
                np.asarray(current.loops, dtype="i4"),
                topo.starts_from_counts(np.asarray(current.counts, dtype="i8")),
                np.asarray(current.material, dtype="i4"),
                np.asarray(current.smooth, dtype=bool),
                uv=(
                    np.asarray(current.corner_uv, dtype="f4").reshape(-1, 2)
                    if current.uv_ok
                    else None
                ),
            )
            objects.append(
                Obj(
                    uid=new_uid(),
                    name=_unique(current.name, taken),
                    mesh=mesh,
                    generator=None,
                    material=current.material[0],
                )
            )

    for line in lines:
        parts = line.split(None, 1)
        head = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

        if head == "v":
            nums = rest.split()
            if len(nums) < 3:
                raise OpError(f"OBJ has a 'v' line with fewer than 3 numbers: {line!r}")
            try:
                xyz = np.array([float(nums[0]), float(nums[1]), float(nums[2])], dtype="f8")
            except ValueError as exc:
                raise OpError(f"OBJ has a non-numeric 'v' line: {line!r}") from exc
            positions_all.append((matrix @ xyz).tolist())
        elif head == "vt":
            nums = rest.split()
            if not nums:
                raise OpError(f"OBJ has a 'vt' line with no numbers: {line!r}")
            try:
                u = float(nums[0])
                v = float(nums[1]) if len(nums) > 1 else 0.0
            except ValueError as exc:
                raise OpError(f"OBJ has a non-numeric 'vt' line: {line!r}") from exc
            texcoords_all.append((u, 1.0 - v))  # OBJ v-up -> glTF v-down
        elif head == "vn":
            continue  # read for nothing; see the module docstring
        elif head in ("o", "g"):
            flush()
            current = _Building(rest.strip() or name)
        elif head == "usemtl":
            current_material = material_index(rest.strip() or None)
        elif head == "s":
            value = rest.strip().split()[0] if rest.strip() else "off"
            current_smooth = value not in ("off", "0")
        elif head == "f":
            tokens = rest.split()
            if len(tokens) < 3:
                raise OpError(f"OBJ face has fewer than 3 vertices: {line!r}")
            corners = [
                _parse_corner(tok, len(positions_all), len(texcoords_all)) for tok in tokens
            ]
            for global_v, vt in corners:
                local = current.local_index(global_v, positions_all[global_v])
                current.loops.append(local)
                if vt is None:
                    current.uv_ok = False
                    current.corner_uv.append((0.0, 0.0))
                else:
                    current.corner_uv.append(texcoords_all[vt])
            current.counts.append(len(corners))
            current.material.append(current_material)
            current.smooth.append(current_smooth)
            saw_face = True
        # "mtllib" and anything else unrecognised: tolerated, does nothing.

    flush()

    if not objects:
        if not saw_face:
            raise OpError("This OBJ has no faces in it.")
        raise OpError("This OBJ has no geometry Clay could build an object from.")
    return ClayDoc(objects=objects, materials=materials or None)
