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
from ..mesh import rig

__all__ = ["rig_record", "to_blender_axes"]

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
