"""A Clay document's bones as the library's rig record (``rig.json``).

The adapter between two ideas of a rig. Clay's is a set of marked objects (``kernels.mesh.rig``);
the library's is a ``rig.json`` beside a ``rig.glb`` -- bones with heads and tails in **Blender's
axes** (Z up, -Y forward), a root, mirror pairs, a weighting -- which every downstream door keys
on (``files.ready("rig.glb")``, the Create stage list, the exports). This module is pure: it
reads a document and answers a dict, and ``service.files.save_clay_rig`` is what writes it.

**``kernels/rig`` imports ``kernels/mesh``, never the reverse** (the 2026-10-13 Clay rig plan's
layering decision): a rig record is built *from* a Clay document, and the mesh kernel must not
know that a library exists.

The record claims the hidden ``blank`` template (the manual-rig bootstrap) with
``skeleton: "custom"`` and ``source: "clay"``. ``blank`` is a real template with no authored
clips, so ``_rig_has_clips`` is false and ``animated.glb`` is not offered for a Clay rig the way
it is for a humanoid; the ``source`` key is what lets a later reader tell a Clay rig from a
hand-fitted one.

**Axes.** glTF is Y-up; a rig record is Z-up. ``(x, y, z)`` becomes ``(x, -z, y)``, the same
conversion Blender's importer applies, so a rig this writes and a rig Blender writes agree.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..mesh import document as bd
from ..mesh import mesh as bm
from ..mesh import ops as mesh_ops
from ..mesh import rig
from ..mesh.elements import OpError

__all__ = [
    "heat_skeleton",
    "rig_record",
    "to_blender_axes",
    "weight_source_doc",
    "weight_sources",
]

#: A leaf bone has no child to point at; it points up by this fraction of the rig's largest
#: extent, never less than :data:`MIN_TAIL_FRACTION` (Blender deletes a bone shorter than
#: ``skeleton.MIN_BONE_FRACTION`` of the mesh on leaving edit mode).
LEAF_FRACTION = 0.1
MIN_TAIL_FRACTION = 0.01


def to_blender_axes(point: Any) -> list[float]:
    """A glTF (Y-up) point as a Blender (Z-up) one: ``(x, y, z) -> (x, -z, y)``."""
    x, y, z = (float(v) for v in point)
    return [x, -z, y]


def _bounds(doc: bd.ClayDoc, by_uid: dict[int, Any], lift: np.ndarray) -> tuple[list, list]:
    """The visible geometry's box in glTF axes, or the bones' own if nothing is visible."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for obj in doc.objects:
        if obj.visible and bm.face_count(obj.mesh):
            placed = bm.transformed(obj.mesh, lift @ doc._world_in(obj.uid, by_uid)).positions
            lo, hi = np.minimum(lo, placed.min(axis=0)), np.maximum(hi, placed.max(axis=0))
    if not np.isfinite(lo).all():
        for obj in rig.bones(doc.objects):
            point = (lift @ doc._world_in(obj.uid, by_uid))[:3, 3]
            lo, hi = np.minimum(lo, point), np.maximum(hi, point)
    return lo.tolist(), hi.tolist()


def weight_sources(doc: bd.ClayDoc) -> list[bd.Obj]:
    """The shapes Blender's automatic weights would skin: visible geometry that is neither a
    bone nor already weighted. A bone's own box is a joint marker, not skin."""
    return [
        o for o in doc.objects
        if o.visible and not o.bone and o.skin is None and bm.face_count(o.mesh)
    ]


def weight_source_doc(doc: bd.ClayDoc) -> bd.ClayDoc:
    """A one-object document holding :func:`weight_sources` merged into a single mesh in world
    space: the plain, unskinned mesh a library row is made of and Blender weights.

    Merged without a weld (``eps=0``): shapes meant to stay separate shells stay so, and nothing
    is moved a hair to close a seam. The palette is carried so face colours survive the round
    trip. Raises :class:`~..mesh.elements.OpError` with nothing to weight.
    """
    sources = weight_sources(doc)
    if not sources:
        raise OpError(
            "There is nothing to weight: Blender skins visible shapes that are not bones, and "
            "this document has none."
        )
    by_uid = doc._uid_map()
    worlds = [doc._world_in(o.uid, by_uid) for o in sources]
    # ``join`` refuses fewer than two objects, so the one-shape document (a model imported
    # as a single mesh, the case the button exists for) skips it: the shape is already the
    # whole mesh, and the world transform below is all that is left to do (the 2026-10-14
    # audit, finding pipelines-children-01).
    merged = (
        sources[0].mesh if len(sources) == 1 else mesh_ops.join(sources, eps=0.0, world=worlds)
    )
    merged = bm.transformed(merged, worlds[0])
    scratch = bd.ClayDoc()
    scratch.materials = list(doc.materials)
    scratch.add_object(
        bd.Obj(uid=bd.new_uid(), name=sources[0].name, mesh=merged, material=sources[0].material)
    )
    return scratch


def heat_skeleton(doc: bd.ClayDoc, place: np.ndarray) -> dict[str, Any]:
    """The skeleton to hand Blender: ``{bones, bounds, mirror_pairs}`` in Blender's axes, with
    the geometry and the bones carried through *place* (the grounding the library applied to
    the mesh it was given). Refused by name when Blender could not take it: no bones, or more
    than one root (a skeleton has exactly one).
    """
    record = rig_record(doc, place=place)
    if record is None:
        raise OpError("There are no bones to weight against. Mark the joints as bones first.")
    roots = [bone["name"] for bone in record["bones"] if bone["parent"] is None]
    if len(roots) != 1:
        raise OpError(
            f"Blender weights against one skeleton with one root; this has {len(roots)} "
            f"({', '.join(roots)}). Parent the others under one bone first."
        )
    return {
        "bones": record["bones"],
        "bounds": record["bounds"],
        "mirror_pairs": record["mirror_pairs"],
    }


def rig_record(doc: bd.ClayDoc, *, place: np.ndarray | None = None) -> dict[str, Any] | None:
    """``rig.json``'s content for *doc*, or ``None`` when it has no bones.

    ``place`` is the same 4x4 :func:`~..mesh.document.to_model` takes (the grounding
    ``normalize_glb`` applied to ``model.glb``), so the record describes the skeleton where
    the exported mesh stands. Bone names are the legal ones (:func:`~..mesh.rig.export_names`)
    and are the names ``to_model(..., bone_names=...)`` must be given for the GLB, so the
    record and the file name the same bones.
    """
    marked = rig.bones(doc.objects)
    if not marked:
        return None
    by_uid = doc._uid_map()
    lift = np.eye(4) if place is None else np.asarray(place, dtype="f8")
    names = rig.export_names(doc.objects)
    heads = {b.uid: (lift @ doc._world_in(b.uid, by_uid))[:3, 3] for b in marked}
    parent = {b.uid: rig.bone_parent(by_uid, b.uid) for b in marked}
    children: dict[int, list[int]] = {}
    for b in marked:
        if parent[b.uid] is not None:
            children.setdefault(parent[b.uid], []).append(b.uid)

    lo, hi = _bounds(doc, by_uid, lift)
    extent = max(float(max(np.subtract(hi, lo))), 1e-6)
    tail_floor = extent * MIN_TAIL_FRACTION
    bones_out: list[dict[str, Any]] = []
    for b in marked:
        head = heads[b.uid]
        if children.get(b.uid):
            tail = heads[children[b.uid][0]]
        else:
            above = parent[b.uid]
            span = float(np.linalg.norm(head - heads[above])) * 0.5 if above is not None else 0.0
            tail = head + np.array([0.0, max(span, extent * LEAF_FRACTION, tail_floor), 0.0])
        if float(np.linalg.norm(tail - head)) < tail_floor:
            tail = head + np.array([0.0, tail_floor, 0.0])
        bones_out.append(
            {
                "name": names[b.uid],
                "parent": None if parent[b.uid] is None else names[parent[b.uid]],
                "head": to_blender_axes(head),
                "tail": to_blender_axes(tail),
            }
        )

    geometry = [o for o in doc.objects if o.visible and bm.face_count(o.mesh)]
    weighted = sum(1 for o in geometry if o.skin is not None)
    weighting = "rigid" if not weighted else "heat" if weighted == len(geometry) else "mixed"
    roots = [bone["name"] for bone in bones_out if bone["parent"] is None]
    return {
        "version": 1,
        "template": "blank",
        "label": "Clay rig",
        "root": roots[0],
        "weighting": weighting,
        "weighting_reason": None,
        "bounds": {
            "min": [lo[0], -hi[2], lo[1]],
            "max": [hi[0], -lo[2], hi[1]],
        },
        "bones": bones_out,
        "mirror_pairs": rig.mirror_pairs(bone["name"] for bone in bones_out),
        "adjusted": False,
        "skeleton": "custom",
        "fit": {"method": "clay"},
        "source": "clay",
    }
