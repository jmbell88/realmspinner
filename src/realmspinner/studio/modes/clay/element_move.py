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
both halves in one call. Scale and rotate have the same two doors: the drag builds
its 4x4 with :func:`affine_about`, and the typed ops (:func:`scale_elements`,
:func:`rotate_elements`, the agent's way of tapering a box top) build theirs with
the same function, so the pointer and a typed value cannot disagree about a pivot.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ....kernels.geom3d import math3d as m3
from ....kernels.mesh import elements as el
from ....kernels.mesh.selection import _element_pickable


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """*points* (``(n, 3)``) through the 4x4 *matrix*."""
    homo = np.hstack([np.asarray(points, dtype="f8"), np.ones((len(points), 1))])
    return (np.asarray(matrix, dtype="f8") @ homo.T).T[:, :3]


#: The refusal for a negative scale factor on picked elements. The 2026-10-09 audit's
#: clay-10: a typed -1 moved the positions but not the winding, so every face came out
#: inside-out with no word said. Held here so the drag's toast and the typed op's
#: refusal are one sentence. Zero is *not* refused (the second audit's clay-81): it
#: flattens the picked elements onto a plane and inverts nothing.
NEGATIVE_SCALE_SENTENCE = (
    "Scaling selected elements by a negative factor would turn their faces "
    "inside out, so they were left as they were. Use Mirror to flip them."
)


def affine_about(pivot: Any, matrix: np.ndarray) -> np.ndarray:
    """*matrix* conjugated to act about *pivot* rather than about the origin."""
    centre = np.asarray(pivot, dtype="f8")
    to = m3.identity()
    to[:3, 3] = -centre
    back = m3.identity()
    back[:3, 3] = centre
    return back @ matrix @ to


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


def _storable(final: np.ndarray) -> np.ndarray:
    """*final* as the float32 the mesh stores, or ``OpError`` when it cannot be.

    The 2026-10-07 audit's clay-14: a 40-digit typed move reached this cast as
    1e40, which float32 turns into ``inf``, and the step went onto the undo stack
    carrying a position the document could not save until it was undone. The
    refusal is checked *after* the cast, because that is the value that would be
    stored (a float64 just under the float32 maximum can still round to ``inf``),
    and it is an ``OpError`` because that is the refusal both callers already
    handle -- the drag toasts it and restores the overlays, the typed door leaves
    the object where it was.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.asarray(final, dtype="f4")
    if not np.isfinite(out).all():
        raise el.OpError(
            "That move is too large for the model to store, so nothing moved. "
            "Use a smaller distance."
        )
    return out


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
                    replace(before, positions=_storable(final)),
                    select=doc.element_sel_of(uid),
                )
            except el.OpError as error:
                refused.append((uid, error))
    finally:
        if history is not None:
            history.collapse_since(mark)
    return (history is not None and history.head != head, unchanged, refused)


def move_elements(
    doc: Any,
    delta_world: Any,
    *,
    snap_step: float = 0.0,
    refused: list[Exception] | None = None,
) -> bool:
    """Shift every object's selected elements by *delta_world*. -> whether anything moved.

    One undo step however many objects hold a selection. A zero shift records
    nothing and an object whose elements are all hidden is skipped, as are the
    ones the document refuses (an ``OpError`` leaves that object where it was).
    ``snap_step`` snaps each moved vertex to the world grid; zero leaves them
    exactly where the shift puts them. The 2026-10-07 audit's clay-14: a caller
    with somewhere to say so passes a list as *refused* and finds each refusal's
    error in it afterwards; without one the refusal is still silent, as the G
    drag's own door (``commit_positions``) reports its refusals itself.
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
        # An over-large shift overflows this float32 store to ``inf``; that is
        # ``commit_positions``' refusal to make, not a warning to print.
        with np.errstate(over="ignore", invalid="ignore"):
            positions[verts] = moved_local(
                obj.mesh.positions[verts].astype("f8"),
                matrix,
                inverse,
                shift,
                snap_step=snap_step,
            )
        changes[uid] = (obj.mesh, positions)
    pushed, _unchanged, failed = commit_positions(doc, changes)
    if refused is not None:
        refused.extend(error for _uid, error in failed)
    return pushed


def _transform_elements(
    doc: Any, world: np.ndarray, refused: list[Exception] | None
) -> bool:
    """Put every object's selected vertices through the world-space affine *world*.

    The body :func:`move_elements` has for a shift, for any 4x4: eligibility is
    ``_element_pickable`` plus ``el.affected_verts``, the world matrix is undone
    per object so a parented one lands right, and the lot is one undo step through
    :func:`commit_positions` (which also drops an object whose positions came out
    equal, so a transform that changes nothing records nothing).
    """
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
        with np.errstate(over="ignore", invalid="ignore"):
            positions[verts] = apply_affine(
                inverse @ world @ matrix, obj.mesh.positions[verts].astype("f8")
            )
        changes[uid] = (obj.mesh, positions)
    pushed, _unchanged, failed = commit_positions(doc, changes)
    if refused is not None:
        refused.extend(error for _uid, error in failed)
    return pushed


def _pivot_world(doc: Any, pivot: Any) -> np.ndarray | None:
    """The world point to scale or turn about: *pivot*, or the selection's median."""
    if pivot is None:
        return element_median_world(doc)
    return np.asarray(pivot, dtype="f8").reshape(3)


def scale_elements(
    doc: Any,
    factors: Any,
    *,
    pivot: Any = None,
    refused: list[Exception] | None = None,
) -> bool:
    """Scale every object's selected elements by per-world-axis *factors*.

    -> whether anything moved.

    About *pivot* (a world point; ``None`` is the selection's median, the gizmo's
    centre). A negative factor is refused with :data:`NEGATIVE_SCALE_SENTENCE` --
    appended to *refused* when the caller passed a list, raised as
    ``OpError`` when not -- exactly as the S drag refuses it. All-ones records nothing.
    """
    f = np.asarray(factors, dtype="f8").reshape(3)
    if bool((f < 0.0).any()):
        error = el.OpError(NEGATIVE_SCALE_SENTENCE)
        if refused is None:
            raise error
        refused.append(error)
        return False
    if bool(np.all(f == 1.0)):
        return False
    centre = _pivot_world(doc, pivot)
    if centre is None:
        return False
    return _transform_elements(doc, affine_about(centre, np.diag(np.append(f, 1.0))), refused)


def rotate_elements(
    doc: Any,
    euler_deg: Any,
    *,
    pivot: Any = None,
    refused: list[Exception] | None = None,
) -> bool:
    """Turn every object's selected elements by Euler XYZ degrees about world axes.

    -> whether anything moved.

    The app's one Euler order (``math3d.quat_from_euler_xyz``), about *pivot* (a
    world point; ``None`` is the selection's median). All-zero records nothing.
    """
    angles = np.asarray(euler_deg, dtype="f8").reshape(3)
    if not np.any(angles):
        return False
    centre = _pivot_world(doc, pivot)
    if centre is None:
        return False
    quat = m3.quat_from_euler_xyz(angles)
    rotation = m3.compose(m3.vec3(), quat, m3.vec3(1.0, 1.0, 1.0))
    return _transform_elements(doc, affine_about(centre, rotation), refused)


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
