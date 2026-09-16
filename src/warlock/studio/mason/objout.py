"""``mason/objout.py``: the scene as a merged Wavefront OBJ, written by hand.

**OBJ and MTL are hand-written and pure inside this module, never routed
through ``pipelines.postprocess.glb_to_obj_zip``.** The plan left that choice
open and it is closed now, for four reasons.

First, byte determinism. Every other format Mason writes is byte-identical
for an unchanged document -- ``.wscn`` fixes its zip timestamps for exactly
this reason, and the GLB is deterministic by construction
(``viewer/glbwrite.py``'s own docstring is built on that claim). Routing OBJ
through trimesh would put a third party's float formatting, its material
naming and its group ordering between the document and the bytes, and none of
those are ours to pin -- a trimesh upgrade could change a decimal place or a
traversal order and this project would have no way to say the export had
silently changed.

Second, layering. This package may not import ``pipelines``
(``tests/mason/test_mason_imports.py`` enforces it, the same door it already
holds shut against ``clay`` and against ``warlock.service``), so a
trimesh-based writer would have to live outside this package entirely --
``studio/mason_io.py``, most likely -- which puts one of Mason's three
exporters untested beside its two siblings and turns "every exporter returns
a mapping of paths to bytes" into a rule with one exception in it.

Third, reporting. A light and a camera are not representable in OBJ at all,
and this codebase's own precedent for a loss like that is
``viewer/gltf.Model.skipped_textures``: a loss that is *stated* is not the
same as a loss that is invisible. trimesh has nowhere to report from -- it
would hand back a mesh with the lights and cameras simply gone, and nothing
downstream would know to say so.

Fourth, names. Writing this by hand means it owns its ``o`` groups, so a
merged file stays navigable -- a reader can find the wall they were looking
for instead of scrolling one anonymous triangle soup. The rule that makes
those names unique is imported from ``gltfout.py``
(``unique_name``/``DEFAULT_NAMES``/``kind_of``) rather than reimplemented, so
there is one answer in this package to "what is this node called and how is a
duplicate resolved".

**That is one rule, not one set of names, and the difference is worth stating
because it would otherwise read as a bug.** The OBJ's group names are *not*
the GLB's: the GLB names every node the walk crosses, including the groups and
prefab instances a merged OBJ has no room for, so the two files number their
duplicates from different populations -- a group called "Rock" holding a mesh
called "Rock" is ``Rock``/``Rock.001`` in the GLB and a single ``o Rock``
here. Nothing needs them to agree, because unlike the GLB and its manifest --
which address each other by name and are therefore written from one shared
record -- an OBJ and its MTL are a self-contained pair that refers to nothing
outside itself.

**This is ``scene.resolve``, not ``scene.walk``.** ``gltfout.py`` needs the
structural walk because a glTF export keeps the hierarchy the user built; an
OBJ has no node graph to keep one in, so the flattening resolver -- the exact
consumer ``scene.py``'s own module docstring was written for -- is the right
shape here, not a second traversal to reach past. That flattening carries a
real cost this format cannot avoid: unlike the GLB, which shares one mesh
across every placement of a reference (``gltfout.py``'s own docstring calls
this out as the reason instancing is free there), a merged OBJ has no node to
hang a shared mesh off of, so every instance of every prefab writes its own
copy of the vertices. :data:`MAX_OBJ_VERTS` is the ceiling that duplication
makes worth having.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..viewer import gltf
from . import scene as sc
from . import terrain as tr
from .gltfout import DEFAULT_NAMES, kind_of, unique_name
from .refs import GeometrySource

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import MasonDoc

__all__ = ["MAX_OBJ_VERTS", "MTL", "OBJ", "TEXTURE_DIR", "ObjExport", "obj_export"]

#: The merged mesh's own file and its material library -- one of each, since
#: nothing about this export ever produces more than one document.
OBJ = "scene.obj"
MTL = "scene.mtl"
#: Where a base-colour texture lands, mirroring ``plotter/tmx.py``'s own
#: convention of a named subdirectory rather than files loose beside the map.
TEXTURE_DIR = "textures"

#: The ceiling this writer refuses past, checked before a single byte of the
#: file is formatted -- ``gltf._Reader._charge``'s rule for a budget, restated
#: here: "the refusal happens instead of the bytes, not alongside them",
#: because the whole point of the ceiling is the allocation it prevents.
#:
#: Measured 2026-09-11 (``dev/measurements/2026-09-11-mason-obj-ceiling.md``):
#: at the ceiling, formatting a merged OBJ costs about 6.4 seconds and
#: produces roughly 211 MB of text, and turning that text into the ``bytes``
#: this module hands back peaks at roughly three times that figure -- the
#: line list, the string ``"\n".join`` builds from it, and that string's UTF-8
#: encoding are all briefly alive together. Past the ceiling the export is a
#: multi-minute wait producing a file no DCC tool this project's users touch
#: opens comfortably, for a merged mesh that -- unlike the GLB -- carries no
#: instancing to shrink it: a thousand placements of one prop write a
#: thousand copies of its vertices here, because OBJ has no node graph to hang
#: a shared mesh reference from. One million vertices is the point past which
#: that cost stops being a napkin calculation and starts being the
#: measurement above.
MAX_OBJ_VERTS = 1_000_000

#: A determinant this close to zero is a node scaled to zero (or numerically
#: indistinguishable from it) on some axis. ``np.linalg.inv`` does not
#: reliably raise on such a matrix -- it can hand back a matrix of
#: ``inf``/``nan`` instead -- so the determinant is checked before the
#: inversion is even attempted. Not a measured constant: a guard against
#: float noise around an exact zero, the same order of magnitude
#: ``clay/ops_bevel.py``'s ``_BISECTOR_EPS`` and ``clay/ops.py``'s
#: near-parallel checks already use for the identical reason.
_SINGULAR_DET_EPS = 1e-9

#: How short a transformed normal has to be before it is treated as the same
#: failure as a singular matrix. ``clay/ops.py`` already draws this line for
#: "is this vector usable as a direction at all" (``length < 1e-12``); a
#: legitimate unit normal is never this close to zero, only a degenerate one
#: (an all-zero source normal, say) is.
_ZERO_NORMAL_EPS = 1e-12

#: No timestamp, no version string, no varying value of any kind -- the whole
#: point of this line is that it is identical for an unchanged document.
_HEADER = "# Warlock Studio -- Mason scene export (metres, Y-up, right-handed)"


@dataclass(frozen=True)
class ObjExport:
    """A merged OBJ, its MTL, and whatever textures it carries -- plus the
    losses along the way.

    ``files`` is ``plotter/tmx.tmx_export``'s own shape, a path to bytes for
    whatever the io layer stages as a set: ``{"scene.obj": ..., "scene.mtl":
    ..., "textures/0.png": ...}``. ``skipped`` is complete sentences rather
    than codes -- ``Model.skipped_textures``'s rule, stated in words here
    because there is no single number for "how much of the scene did not
    make it into a format that cannot carry a light, a camera, or four of a
    material's five texture slots".
    """

    files: dict[str, bytes]
    skipped: tuple[str, ...]
    vertices: int
    triangles: int


def obj_export(
    doc: MasonDoc, source: GeometrySource, *, include_hidden: bool = False
) -> ObjExport:
    """The document as a merged OBJ, through ``scene.resolve`` -- see the
    module docstring for why the flattening resolver and not ``scene.walk``.

    Refuses before formatting a single line: :func:`_collect` resolves every
    reference and counts the vertices that would be written, and only once
    that count clears :data:`MAX_OBJ_VERTS` does :func:`_format` build any
    bytes at all.
    """
    jobs, skipped, total_vertices = _collect(doc, source, include_hidden=include_hidden)
    if total_vertices > MAX_OBJ_VERTS:
        raise ValueError(
            f"this scene resolves to {total_vertices:,} OBJ vertices, past the "
            f"{MAX_OBJ_VERTS:,} this writer will format into one file"
        )
    return _format(jobs, skipped)


# -- gathering ---------------------------------------------------------------


def _collect(
    doc: MasonDoc, source: GeometrySource, *, include_hidden: bool
) -> tuple[list[tuple[sc.Placed, str, list[gltf.Primitive]]], list[str], int]:
    """Every placed item's resolved geometry, named, with nothing formatted yet.

    **Every placed item claims a name, including the ones that write no
    geometry.** A light, a camera and a mesh whose reference did not resolve
    each take a slot from the same ordered ``taken`` set as a mesh that does,
    because each of them produces a sentence in ``skipped`` and a sentence
    that says "a light was left out" is worth much less than one that says
    *which*. Two lights both called "Sun" would otherwise report the same loss
    twice with nothing to tell them apart, which is the failure
    ``unique_name`` exists to prevent one file over.

    These names are deliberately not the GLB's -- see the module docstring's
    fourth reason for why the two files number their duplicates from different
    populations and why nothing needs them to agree.
    """
    taken: set[str] = set()
    skipped: list[str] = []
    jobs: list[tuple[sc.Placed, str, list[gltf.Primitive]]] = []
    total_vertices = 0
    for item in sc.resolve(doc, include_hidden=include_hidden):
        kind = kind_of(item.node)
        name = unique_name(item.node.name.strip() or DEFAULT_NAMES[kind], taken)
        if kind == "mesh":
            if item.ref is None:
                # Never given a source (a tool mid-placement, say): nothing to
                # place and, unlike a reference that failed to resolve,
                # nothing wrong to report either.
                continue
            prims = source.primitives(item.ref)
            if not prims:
                skipped.append(
                    f"{name!r} names a reference that did not resolve; its geometry "
                    "was left out of the OBJ"
                )
                continue
            jobs.append((item, name, prims))
            total_vertices += sum(len(p.positions) for p in prims)
        elif kind == "terrain":
            if doc.terrain is None:
                continue
            prim = tr.terrain_mesh(doc.terrain)
            jobs.append((item, name, [prim]))
            total_vertices += len(prim.positions)
        elif kind in ("light", "camera"):
            skipped.append(
                f"{name!r} is a {kind}, which OBJ has no way to carry; it was left out"
            )
        # A group or an unexpanded prefab never reaches here: scene.resolve
        # already leaves both out, the same way gltfout.scene_model's own
        # "mesh"/"terrain"/"light"/"camera" branches are the only ones that
        # produce anything.
    return jobs, skipped, total_vertices


# -- formatting ----------------------------------------------------------


def _format(
    jobs: list[tuple[sc.Placed, str, list[gltf.Primitive]]], skipped: list[str]
) -> ObjExport:
    """Every job as OBJ text and MTL text, in the document order they arrived in.

    Nothing below is ordered by ``id()`` or by iterating a ``set`` -- ``jobs``
    is already document order from :func:`_collect`, and every dict here is
    read back through a list that records the order things were first put in,
    never through the dict's own iteration -- which is what makes two exports
    of an unchanged document byte-identical.
    """
    lines = [_HEADER, f"mtllib {MTL}"]
    mat_names: dict[int, str] = {}
    mat_order: list[gltf.Material] = []
    mat_taken: set[str] = set()
    v_count = vt_count = vn_count = 0
    triangles = 0
    for item, name, prims in jobs:
        v_count, vt_count, vn_count, tri = _emit_node(
            item,
            name,
            prims,
            lines,
            v_count,
            vt_count,
            vn_count,
            mat_names=mat_names,
            mat_order=mat_order,
            mat_taken=mat_taken,
            skipped=skipped,
        )
        triangles += tri
    files: dict[str, bytes] = {}
    mtl_lines = _mtl_lines(mat_order, mat_names, files)
    files[OBJ] = ("\n".join(lines) + "\n").encode("utf-8")
    files[MTL] = ("\n".join(mtl_lines) + "\n").encode("utf-8")
    return ObjExport(files=files, skipped=tuple(skipped), vertices=v_count, triangles=triangles)


def _emit_node(
    item: sc.Placed,
    name: str,
    prims: list[gltf.Primitive],
    lines: list[str],
    v_count: int,
    vt_count: int,
    vn_count: int,
    *,
    mat_names: dict[int, str],
    mat_order: list[gltf.Material],
    mat_taken: set[str],
    skipped: list[str],
) -> tuple[int, int, int, int]:
    """One placed item's ``o`` group, its faces, and the counters moved past them.

    The normal matrix is one fact about this *node* -- its world transform --
    not about any one primitive under it, so it is computed once here and
    shared by every primitive the loop below writes; a node with several
    primitives and a singular scale reports the loss once, not once per
    primitive.
    """
    normal_matrix = _normal_matrix(item.world[:3, :3])
    if normal_matrix is None:
        skipped.append(
            f"{name!r} has a singular transform (a zero scale on some axis), so its "
            "faces were written with no normal index rather than as NaNs"
        )
    lines.append(f"o {name}")
    current_material: str | None = None
    triangles = 0
    for prim in prims:
        positions = np.asarray(prim.positions, dtype="f8")
        world_positions = positions @ item.world[:3, :3].T + item.world[:3, 3]

        normals_out: np.ndarray | None = None
        if normal_matrix is not None and prim.normals is not None:
            transformed = np.asarray(prim.normals, dtype="f8") @ normal_matrix.T
            lengths = np.linalg.norm(transformed, axis=1)
            # A degenerate (already-zero) source normal is dropped the same
            # way a singular matrix is: mixing "some vertices have a normal
            # index, some do not" within one primitive is not a shape OBJ's
            # per-vertex triple can express cleanly, so the whole primitive
            # falls back to positions-only instead.
            if lengths.size and bool((lengths > _ZERO_NORMAL_EPS).all()):
                normals_out = transformed / lengths[:, None]
        has_normals = normals_out is not None
        has_uv = prim.uvs is not None

        material = item.material if item.material is not None else prim.material
        material_name = _material_name(material, mat_names, mat_order, mat_taken, skipped)
        if material_name != current_material:
            lines.append(f"usemtl {material_name}")
            current_material = material_name

        for x, y, z in world_positions:
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
        if has_uv:
            for u, w in prim.uvs:
                lines.append(f"vt {float(u):.6f} {float(w):.6f}")
        if has_normals:
            assert normals_out is not None  # narrows for the loop below
            for x, y, z in normals_out:
                lines.append(f"vn {x:.6f} {y:.6f} {z:.6f}")

        indices = np.asarray(prim.indices).reshape(-1, 3)
        for a, b, c in indices:
            face: list[str] = []
            for vi in (int(a), int(b), int(c)):
                v_i = v_count + vi + 1
                if has_uv and has_normals:
                    face.append(f"{v_i}/{vt_count + vi + 1}/{vn_count + vi + 1}")
                elif has_normals:
                    face.append(f"{v_i}//{vn_count + vi + 1}")
                elif has_uv:
                    face.append(f"{v_i}/{vt_count + vi + 1}")
                else:
                    face.append(f"{v_i}")
            lines.append("f " + " ".join(face))
        triangles += len(indices)
        v_count += len(world_positions)
        if has_uv:
            vt_count += len(prim.uvs)
        if has_normals:
            assert normals_out is not None
            vn_count += len(normals_out)
    return v_count, vt_count, vn_count, triangles


def _normal_matrix(basis: np.ndarray) -> np.ndarray | None:
    """The inverse-transpose of a node's 3x3, or ``None`` when it has none.

    ``np.linalg.inv`` does not reliably raise on a matrix scaled to zero on
    one axis -- it can hand back a matrix of ``inf``/``nan`` instead -- so the
    determinant is checked first and the result checked again after, and
    either failure answers ``None`` rather than propagating a non-finite
    matrix to whatever multiplies by it next.
    """
    det = float(np.linalg.det(basis))
    if not np.isfinite(det) or abs(det) < _SINGULAR_DET_EPS:
        return None
    try:
        inverse = np.linalg.inv(basis)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(inverse).all():
        return None
    return inverse.T


def _material_name(
    material: gltf.Material,
    mat_names: dict[int, str],
    mat_order: list[gltf.Material],
    mat_taken: set[str],
    skipped: list[str],
) -> str:
    """This material's MTL name, minting one -- and its ``skipped`` entries --
    the first time this exact object is seen.

    Keyed on ``id(material)``, the identity ``write_glb``, ``GpuMaterial`` and
    ``clay/serialize.py`` all already deduplicate a material on, so
    ``mat_order`` holds every keyed material alive for the length of the
    call -- the same requirement ``glbwrite._Writer._material_keep``'s own
    comment states explicitly rather than resting it on a caller's object
    graph: an ``id`` is an address CPython is free to reissue the moment its
    object is collected.
    """
    key = id(material)
    name = mat_names.get(key)
    if name is not None:
        return name
    name = unique_name(material.name.strip() or "material", mat_taken)
    mat_names[key] = name
    mat_order.append(material)
    for slot in ("metallic_roughness", "normal", "emissive", "occlusion"):
        if getattr(material, slot) is not None:
            skipped.append(
                f"material {name!r} carries a {slot.replace('_', '-')} map, which MTL has "
                "no portable spelling for; it was left out"
            )
    return name


def _mtl_lines(
    mat_order: list[gltf.Material], mat_names: dict[int, str], files: dict[str, bytes]
) -> list[str]:
    """The whole ``.mtl`` file, in the order its materials were first met."""
    lines = [_HEADER]
    texture_paths: dict[int, str] = {}
    for material in mat_order:
        name = mat_names[id(material)]
        lines.append(f"newmtl {name}")
        red, green, blue, alpha = (float(c) for c in material.base_color_factor)
        lines.append(f"Kd {red:.6f} {green:.6f} {blue:.6f}")
        if alpha < 1.0:
            lines.append(f"d {alpha:.6f}")
        lines.append("Ks 0.000000 0.000000 0.000000")
        # There is no principled map from a PBR roughness to a Blinn-Phong
        # specular exponent -- the two models do not describe the same
        # surface, and no formula recovers one exactly from the other. This
        # is a stated, honest approximation rather than a derivation:
        # roughness 0..1 maps linearly onto Ns 1000..0, the classic MTL
        # range, so a mirror-smooth material (roughness 0) reads as a tight,
        # bright highlight and a fully rough one reads as none at all --
        # monotonic and bounded, nothing stronger claimed.
        roughness = min(1.0, max(0.0, float(material.roughness_factor)))
        lines.append(f"Ns {(1.0 - roughness) * 1000.0:.6f}")
        if any(material.emissive_factor):
            er, eg, eb = (float(c) for c in material.emissive_factor)
            lines.append(f"Ke {er:.6f} {eg:.6f} {eb:.6f}")
        if material.base_color is not None:
            key = id(material.base_color)
            path = texture_paths.get(key)
            if path is None:
                path = f"{TEXTURE_DIR}/{len(texture_paths)}.png"
                texture_paths[key] = path
                files[path] = _encode_png(material.base_color)
            lines.append(f"map_Kd {path}")
        lines.append("")
    return lines


def _encode_png(image: tuple[int, int, bytes]) -> bytes:
    """A decoded ``(width, height, rgba)`` slot as PNG bytes.

    Pillow is imported here and nowhere else in this module -- the same rule
    ``viewer/gltf.py``'s own ``texture()`` follows and
    ``tests/mason/test_mason_imports.py`` enforces: nothing in this package's
    import list should cost every other test in the directory a Pillow
    import it does not need.
    """
    import io

    from PIL import Image

    width, height, data = image
    out = io.BytesIO()
    Image.frombytes("RGBA", (int(width), int(height)), bytes(data)).save(out, "PNG")
    return out.getvalue()
