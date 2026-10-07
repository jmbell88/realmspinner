"""A Clay document out as Wavefront OBJ + MTL.

The counterpart of :mod:`.objimport`, and deliberately built the same way --
hand-rolled rather than handed to a library, because the thing worth keeping on
the way out is exactly what :mod:`.objimport`'s docstring says is worth keeping
on the way in: n-gons and per-corner UVs. A writer built on a triangulating
library would hand :mod:`.objimport` back a mesh with three times the faces it
started with.

**Every object is baked before it is written, to *world* space.** OBJ has no
per-object transform of its own and no node hierarchy either -- a face's
vertices are just numbers -- so an object's whole placement, ancestors
included (tranche 3: scene structure), has to be folded into its positions
first: a parented object's own local TRS alone is not where it actually sits.
:func:`~.ops.bake_transform` is the one function in this package that does
that, given the object's world matrix (``doc.world_matrix``); it is also what
:mod:`~.document`'s own ``to_model`` relies on being correct, so this does
not re-derive the math.

**Deterministic, by construction rather than by promise.** Every number is
formatted the same way every time (``%.9g`` -- the 2026-09-26 audit's
clay-io-13 widened this from ``%.6g``, which lost precision below about a
millimetre at a real-world, kilometre-scale level), objects and materials are
written in the document's own list order, and a face's corners are written in
the mesh's own order -- nothing here sorts by a dict's iteration order or a
set's. Two calls over an unchanged document produce byte-identical text.

**One material slot, one ``newmtl``, named by its palette index** --
``Material_<index>`` -- rather than by the material's own (possibly empty,
possibly duplicated) ``name``. :mod:`.objimport` resolves a ``usemtl`` by
exact string match against the ``mtl`` text it is given, so a stable,
collision-free name is what makes a round trip through both modules land a
face back on the material it started on.

**A base-colour texture is a PNG beside the OBJ**, the way picoCAD exports one:
the ``.mtl`` says ``map_Kd <name>_<index>.png`` and :func:`claydoc_textures`
hands back the PNG bytes for the caller to write next to it. The name is a bare
file name on purpose -- :mod:`.objimport` follows only a bare name in the OBJ's
own folder -- and carries the palette index so two materials never collide.
Only the base colour travels: OBJ has no standard slot for the other four, and
Clay strips them anyway.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from . import mesh as bm
from . import ops
from .document import ClayDoc
from .objimport import ns_from_roughness

__all__ = ["claydoc_textures", "claydoc_to_obj", "texture_name"]


def _num(x: float) -> str:
    # The 2026-09-26 audit, finding clay-io-13: ``%.6g`` keeps only six
    # significant digits, so a coordinate around 1000 (a real-world scale in
    # metres, a kilometre-scale outdoor level) loses precision below about a
    # millimetre, and a coordinate around 1e5 loses precision below about a
    # centimetre -- a corner that round-trips a comfortably visible amount
    # away from where it started. ``%.9g`` keeps a float32's own ~7.2
    # significant decimal digits with a full digit of headroom, so a
    # round trip through this writer loses nothing an ``f4`` position did not
    # already lack.
    return f"{float(x):.9g}"


def _line_text(name: object) -> str:
    """*name* as text that is safe on one line of an OBJ or MTL file.

    The 2026-10-03 audit, finding clay-38: an object or material name went into
    ``o {name}`` and ``# Clay material name: {name}`` verbatim, so a name with
    a newline (a GLB node's, or one an agent set through ``clay_rename``)
    injected whole statements -- a stray ``v``, an ``f``, a phantom ``o``, a
    second ``newmtl`` -- and renumbered every vertex after it. Every character
    that is not printable (control characters, and the Unicode line and
    paragraph separators ``str.splitlines`` also breaks on) becomes one space.
    """
    return "".join(ch if ch.isprintable() else " " for ch in str(name))


def _slot_name(index: int) -> str:
    return f"Material_{index}"


def texture_name(name: str, index: int) -> str:
    """The PNG file name for palette entry *index*'s base colour: a bare name.

    Passed through :func:`_line_text` so the name an ``.mtl`` line carries and
    the file written under it are the same string whatever the title held.
    """
    return _line_text(f"{name}_{index}.png")


def _write_material(lines: list[str], index: int, material: Any, name: str) -> None:
    r, g, b, a = material.base_color_factor
    lines.append(f"newmtl {_slot_name(index)}")
    if material.name:
        lines.append(f"# Clay material name: {_line_text(material.name)}")
    lines.append(f"Kd {_num(r)} {_num(g)} {_num(b)}")
    lines.append(f"d {_num(a)}")
    lines.append(f"Ns {_num(ns_from_roughness(material.roughness_factor))}")
    if material.base_color is not None:
        lines.append(f"map_Kd {texture_name(name, index)}")
    for slot in ("metallic_roughness", "normal", "emissive", "occlusion"):
        if getattr(material, slot, None) is not None:
            lines.append(
                f"# {slot} carries a texture in the source material; "
                "OBJ export does not carry image data"
            )


def claydoc_to_obj(
    doc: ClayDoc, *, name: str = "model", visible_only: bool = True
) -> tuple[str, str]:
    """*doc* as ``(obj_text, mtl_text)``.

    ``visible_only`` mirrors :func:`~.document.to_model`'s own rule -- a hidden
    object does not render, does not export and cannot be picked, and the
    default here keeps that one meaning in one place. Passing ``False`` writes
    every object regardless, for a caller that means to archive the whole
    document rather than what it currently looks like.
    """
    exported = _exported(doc, visible_only)

    obj_lines = ["# Written by Realmspinner's Clay", f"mtllib {name}.mtl"]
    used_materials: set[int] = set()
    v_offset = 0
    vt_offset = 0

    for obj in exported:
        # Baked to **world** space, not the object's own local TRS: OBJ carries
        # no node hierarchy of its own, so a parented object's siblings need
        # their real absolute positions, not positions relative to a parent
        # this format cannot express.
        baked = ops.bake_transform(obj, world=doc.world_matrix(obj.uid))
        mesh = baked.mesh
        n_faces = bm.face_count(mesh)
        if n_faces == 0:
            continue

        obj_lines.append(f"o {_line_text(obj.name)}")
        for x, y, z in mesh.positions.tolist():
            obj_lines.append(f"v {_num(x)} {_num(y)} {_num(z)}")

        has_uv = mesh.uv is not None
        if has_uv:
            assert mesh.uv is not None
            for u, v in mesh.uv.tolist():
                obj_lines.append(f"vt {_num(u)} {_num(1.0 - v)}")

        last_material: int | None = None
        last_smooth: bool | None = None
        for face in range(n_faces):
            lo, hi = int(mesh.starts[face]), int(mesh.starts[face + 1])
            material_index = int(mesh.material[face])
            smooth = bool(mesh.smooth[face])
            if material_index != last_material:
                used_materials.add(material_index)
                obj_lines.append(f"usemtl {_slot_name(material_index)}")
                last_material = material_index
            if smooth != last_smooth:
                obj_lines.append("s 1" if smooth else "s off")
                last_smooth = smooth
            corners = []
            for corner in range(lo, hi):
                v = v_offset + int(mesh.loops[corner]) + 1
                if has_uv:
                    corners.append(f"{v}/{vt_offset + corner + 1}")
                else:
                    corners.append(str(v))
            obj_lines.append("f " + " ".join(corners))

        v_offset += len(mesh.positions)
        if has_uv:
            assert mesh.uv is not None
            vt_offset += len(mesh.uv)

    mtl_lines = ["# Written by Realmspinner's Clay"]
    for index in sorted(used_materials):
        material = doc.materials[index] if 0 <= index < len(doc.materials) else None
        if material is None:
            continue
        _write_material(mtl_lines, index, material, name)

    return "\n".join(obj_lines) + "\n", "\n".join(mtl_lines) + "\n"


def _exported(doc: ClayDoc, visible_only: bool) -> list[Any]:
    return [obj for obj in doc.objects if obj.visible or not visible_only]


def claydoc_textures(doc: ClayDoc, *, visible_only: bool = True) -> dict[int, bytes]:
    """``{palette index: PNG bytes}`` of every base-colour texture the OBJ uses.

    The same set of materials :func:`claydoc_to_obj` writes ``map_Kd`` for: used
    by a face of an exported object, and carrying a base-colour texture. A
    document with no texture gives ``{}``, so the caller writes no PNG at all.
    PNG-encoding is real work -- call this off the frame thread.
    """
    from PIL import Image

    used: set[int] = set()
    for obj in _exported(doc, visible_only):
        if bm.face_count(obj.mesh):
            used.update(int(i) for i in np.unique(obj.mesh.material))
    out: dict[int, bytes] = {}
    for index in sorted(used):
        material = doc.materials[index] if 0 <= index < len(doc.materials) else None
        image = None if material is None else material.base_color
        if image is None:
            continue
        width, height, data = image
        buffer = io.BytesIO()
        Image.frombytes("RGBA", (int(width), int(height)), bytes(data)).save(buffer, "PNG")
        out[index] = buffer.getvalue()
    return out
