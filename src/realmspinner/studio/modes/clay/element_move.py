"""Moving the selected elements by a world-space step: the one place that does it.

The viewport's G drag, its gizmo drag and the Properties pane's typed median field
all end in the same act -- take every visible object's selected vertices, shift
them in *world* space, hand the answer back to each object's local frame, and
record the lot as one undo step. Written three times they would disagree about
which objects are eligible, how a parented object converts back and what a
zero move records; written here they cannot. Logic only: no imgui, no GL, so the
panes, the view and the tests all reach it without a window.

The drag previews through :func:`moved_local` and commits through
:func:`commit_positions`; a typed value goes through :func:`move_elements`, which is
both halves in one call.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ....kernels.mesh import elements as el
from ....kernels.mesh.selection import _element_pickable


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """*points* (``(n, 3)``) through the 4x4 *matrix*."""
    homo = np.hstack([np.asarray(points, dtype="f8"), np.ones((len(points), 1))])
    return (np.asarray(matrix, dtype="f8") @ homo.T).T[:, :3]


def snap_points(points: np.ndarray, step: float) -> np.ndarray:
    """Every component onto the grid; unchanged at step zero.

    The vectorised twin of ``kernels.mesh.ops.snap_value``, with the same
    half-**away-from-zero** tie rule: ``np.round`` breaks a tie by parity, which
    would make some half-way vertices stick backwards and others forwards, and
    would leave the grid asymmetric about the origin.
    """
    step = abs(float(step))
    pts = np.asarray(points, dtype="f8")
    if step == 0.0:
        return pts.copy()
    return np.copysign(np.floor(np.abs(pts) / step + 0.5), pts) * step


def moved_local(
    local: np.ndarray,
    matrix: np.ndarray,
    inverse: np.ndarray,
    delta_world: np.ndarray,
    *,
    snap_step: float = 0.0,
) -> np.ndarray:
    """Object-space vertex positions after a world-space shift of *delta_world*.

    ``local`` are the vertices as the move began, ``matrix`` / ``inverse`` the
    object's placement. With a snap step each vertex lands on the world grid
    itself, rather than the group landing on it by its median -- a vertex that
    started off the grid ends on it, which is what a modeller snapping
    individual corners wants.
    """
    shift = np.asarray(delta_world, dtype="f8").reshape(3)
    if snap_step:
        world = apply_affine(matrix, local) + shift
        return apply_affine(inverse, snap_points(world, snap_step))
    step = np.eye(4)
    step[:3, 3] = shift
    return apply_affine(inverse @ step @ matrix, local)


def commit_positions(
    doc: Any, changes: dict[int, tuple[Any, np.ndarray]]
) -> tuple[bool, list[int], list[tuple[int, Exception]]]:
    """Record new vertex positions for several objects as **one** undo step.

    ``changes`` maps a uid to ``(before, positions)``: the mesh the move began
    on and the positions to give it. -> ``(pushed, unchanged, refused)``: whether
    a step was recorded, the uids whose positions equalled the old ones (nothing
    pushed for them -- dirty is a comparison against the history head, so an
    empty step would make a saved document ask to be saved again), and the
    ``(uid, error)`` pairs the document refused. A deleted uid is skipped.

    ``try/finally`` around the loop: a ``set_mesh`` refusal escaping it would
    skip ``collapse_since`` and leave the undo stack's gesture counter stuck
    open, which switches its eviction off for the rest of the session.
    """
    history = getattr(doc, "history", None)
    head = None if history is None else history.head
    mark = 0 if history is None else history.mark()
    unchanged: list[int] = []
    refused: list[tuple[int, Exception]] = []
    try:
        for uid, (before, final) in changes.items():
            try:
                doc.by_uid(uid)
            except KeyError:
                continue
            if np.array_equal(final, before.positions):
                unchanged.append(uid)
                continue
            try:
                doc.set_mesh(
                    uid,
                    replace(before, positions=np.asarray(final, dtype="f4")),
                    select=doc.element_sel_of(uid),
                )
            except el.OpError as error:
                refused.append((uid, error))
    finally:
        if history is not None:
            history.collapse_since(mark)
    return (history is not None and history.head != head, unchanged, refused)


def move_elements(doc: Any, delta_world: Any, *, snap_step: float = 0.0) -> bool:
    """Shift every object's selected elements by *delta_world*. -> whether anything moved.

    One undo step however many objects hold a selection. A zero shift records
    nothing and an object whose elements are all hidden is skipped, as are the
    ones the document refuses (an ``OpError`` leaves that object where it was).
    ``snap_step`` snaps each moved vertex to the world grid; zero leaves them
    exactly where the shift puts them.
    """
    shift = np.asarray(delta_world, dtype="f8").reshape(3)
    if not np.any(shift):
        return False
    changes: dict[int, tuple[Any, np.ndarray]] = {}
    for uid, sel in list(doc.element_sel.items()):
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            continue
        if not _element_pickable(obj):
            continue
        verts = el.affected_verts(obj.mesh, sel)
        if not len(verts):
            continue
        matrix = doc.world_matrix(uid)
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            continue
        positions = np.array(obj.mesh.positions, dtype="f4")
        positions[verts] = moved_local(
            obj.mesh.positions[verts].astype("f8"),
            matrix,
            inverse,
            shift,
            snap_step=snap_step,
        )
        changes[uid] = (obj.mesh, positions)
    pushed, _unchanged, _refused = commit_positions(doc, changes)
    return pushed


def element_median_world(doc: Any) -> np.ndarray | None:
    """The world mean of every selected element's vertices, or ``None``.

    The same definition as the viewport's gizmo centre (``BoundsOps.element_centre``):
    visible objects only, and each vertex counted once however the selection
    reached it. Read from the document rather than the view, so a pane with no
    viewport in hand (the typed median field) shows the number the gizmo sits on.
    """
    total = np.zeros(3)
    count = 0
    for uid, sel in doc.element_sel.items():
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            continue
        if not _element_pickable(obj):
            continue
        verts = el.affected_verts(obj.mesh, sel)
        if not len(verts):
            continue
        world = apply_affine(doc.world_matrix(uid), obj.mesh.positions[verts].astype("f8"))
        total += world.sum(axis=0)
        count += len(world)
    return None if count == 0 else total / count
