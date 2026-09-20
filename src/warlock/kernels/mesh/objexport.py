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
formatted the same way every time (``%.6g``), objects and materials are
written in the document's own list order, and a face's corners are written in
the mesh's own order -- nothing here sorts by a dict's iteration order or a
set's. Two calls over an unchanged document produce byte-identical text.

**One material slot, one ``newmtl``, named by its palette index** --
``Material_<index>`` -- rather than by the material's own (possibly empty,
possibly duplicated) ``name``. :mod:`.objimport` resolves a ``usemtl`` by
exact string match against the ``mtl`` text it is given, so a stable,
collision-free name is what makes a round trip through both modules land a
face back on the material it started on.

**Tranche 7 (export profiles), the OBJ half.** An OBJ export takes an
optional ``engine`` -- one of :data:`~.engines.ENGINES`'s own keys -- that
does two things purely in the *written text*, never to the document:

1. Every collider object (``Obj.role == "collider"``) is written under its
   *engine's* node name (:func:`collider_export_names`,
   :func:`~.engines.collider_name`) instead of its own ``Obj.name`` --
   ``UCX_Crate_00``, ``Crate_00-convcolonly``... -- numbered per source mesh
   in document order. Colliders are written as ordinary ``o``/geometry
   entries otherwise; nothing about this module's own loop skips them.
2. Every object's baked world matrix is composed with
   ``ENGINES[engine].obj_conversion`` **before** :func:`~.ops.bake_transform`
   folds it into positions -- see :func:`claydoc_to_obj`'s own docstring for
   why this, and only this, export path applies it.

``engine=None`` (the default) reproduces this module's pre-tranche-7
behaviour exactly: no renaming, no conversion -- what every caller before
this tranche (this module's own tests included) already gets.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from . import engines as engines_mod
from . import mesh as bm
from . import ops
from .document import ClayDoc
from .elements import OpError
from .objimport import ns_from_roughness

__all__ = ["claydoc_to_obj", "collider_export_names"]


def collider_export_names(doc: ClayDoc, engine: str) -> dict[int, str]:
    """*uid* -> the engine-convention name every collider object in *doc*
    should be written under -- shared by this module's own OBJ writer and
    ``clay_mode``'s GLB node-renaming step (:func:`~.engines.collider_name`
    called from two file formats wants one numbering, not two).

    Numbered **per source mesh, in ``doc.objects``' own order** -- not export
    or visibility order -- so a collider's exported name does not shift
    depending on which of its siblings happen to be hidden this time; a
    collider added, then later hidden, keeps the same index a re-export gives
    its still-visible neighbours. ``doc`` itself is never touched: this only
    says what a *written file* should call each node, the export doors'
    own rule that the document keeps authoring its own names regardless.
    """
    if engine not in engines_mod.ENGINES:
        raise OpError(f"Unknown engine profile {engine!r}.")
    names: dict[int, str] = {}
    counters: dict[int | None, int] = {}
    for obj in doc.objects:
        if getattr(obj, "role", "mesh") != "collider":
            continue
        source = doc.by_uid(obj.parent) if obj.parent is not None else None
        mesh_name = source.name if source is not None else obj.name
        index = counters.get(obj.parent, 0)
        counters[obj.parent] = index + 1
        names[obj.uid] = engines_mod.collider_name(engine, obj.collider_kind, mesh_name, index)
    return names


def _num(x: float) -> str:
    return f"{float(x):.6g}"


def _slot_name(index: int) -> str:
    return f"Material_{index}"


def _write_material(lines: list[str], index: int, material: Any) -> None:
    r, g, b, a = material.base_color_factor
    lines.append(f"newmtl {_slot_name(index)}")
    if material.name:
        lines.append(f"# Clay material name: {material.name}")
    lines.append(f"Kd {_num(r)} {_num(g)} {_num(b)}")
    lines.append(f"d {_num(a)}")
    lines.append(f"Ns {_num(ns_from_roughness(material.roughness_factor))}")
    for slot in ("base_color", "metallic_roughness", "normal", "emissive", "occlusion"):
        if getattr(material, slot, None) is not None:
            lines.append(
                f"# {slot} carries a texture in the source material; "
                "OBJ export does not carry image data"
            )


def claydoc_to_obj(
    doc: ClayDoc, *, name: str = "model", visible_only: bool = True, engine: str | None = None
) -> tuple[str, str]:
    """*doc* as ``(obj_text, mtl_text)``.

    ``visible_only`` mirrors :func:`~.document.to_model`'s own rule -- a hidden
    object does not render, does not export and cannot be picked, and the
    default here keeps that one meaning in one place. Passing ``False`` writes
    every object regardless, for a caller that means to archive the whole
    document rather than what it currently looks like.

    ``engine`` (a key of :data:`~.engines.ENGINES`) is this module's own
    tranche 7 half -- see the module docstring. ``None``, the default,
    reproduces every behaviour this function had before that tranche: no
    collider renaming, no axis/scale conversion. Refuses (:class:`~.elements.
    OpError`) an *engine* naming nothing in :data:`~.engines.ENGINES`, the
    same door :func:`collider_export_names` and
    :func:`~.engines.collider_name` already refuse an unknown engine through.
    """
    exported = [obj for obj in doc.objects if obj.visible or not visible_only]

    collider_names = collider_export_names(doc, engine) if engine is not None else {}
    # **OBJ only.** GLB stays in glTF's own convention -- Clay's geometry
    # already *is* that convention, every target's glTF importer converts on
    # the way in, and applying this matrix there too would be a *double*
    # conversion, the classic "worked on one engine, looked wrong on every
    # other" bug (see ``engines.py``'s own module docstring, in full). OBJ
    # carries no axis metadata at all for an importer to correct with, so the
    # conversion has to happen here, before a byte is written, or not at all.
    conversion = engines_mod.ENGINES[engine].obj_conversion if engine is not None else None

    obj_lines = ["# Written by Warlock Studio's Clay", f"mtllib {name}.mtl"]
    used_materials: set[int] = set()
    v_offset = 0
    vt_offset = 0

    for obj in exported:
        # Evaluated before baking -- the base run through the modifier stack,
        # or ``obj.mesh`` itself when there is none (:mod:`.modifiers`' fast
        # path) -- so an OBJ export writes what the viewport and the GLB
        # exporter agree the document looks like, not the pre-modifier shape.
        # Baked to **world** space, not the object's own local TRS (tranche
        # 3: scene structure) -- OBJ carries no node hierarchy of its own, so
        # a parented object's siblings need their real absolute positions,
        # not positions relative to a parent this format cannot express.
        #
        # The engine conversion (when there is one) composes *after* the
        # world matrix, in :func:`~.mesh.transformed`'s own ``M @ v`` column
        # convention -- ``conversion @ world`` first places the object in
        # Clay/glTF world space, exactly as every other export already does,
        # then re-expresses that same world-space geometry in the target
        # engine's own OBJ axes/scale. Composing the two into one matrix
        # before baking (rather than baking to world and converting the
        # result as a second step) keeps this a single :func:`~.mesh.
        # transformed` call, the same one :func:`~.ops.bake_transform` was
        # always going to make -- and gets a negative-determinant conversion
        # (none of today's do, but nothing here assumes that) the same loop
        # reversal :func:`~.mesh.transformed` already applies to a mirrored
        # object, for free.
        world = doc.world_matrix(obj.uid)
        baked = ops.bake_transform(
            replace(obj, mesh=doc.evaluated(obj.uid)),
            world=world if conversion is None else conversion @ world,
        )
        mesh = baked.mesh
        n_faces = bm.face_count(mesh)
        if n_faces == 0:
            continue

        obj_lines.append(f"o {collider_names.get(obj.uid, obj.name)}")
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

    mtl_lines = ["# Written by Warlock Studio's Clay"]
    for index in sorted(used_materials):
        material = doc.materials[index] if 0 <= index < len(doc.materials) else None
        if material is None:
            continue
        _write_material(mtl_lines, index, material)

    return "\n".join(obj_lines) + "\n", "\n".join(mtl_lines) + "\n"
