"""The op drag: find an operation's number with the mouse, then commit it once.

Press an op's key with an element selection and the pointer's travel sets its
first parameter (``ops.DragSpec``) -- today only Inset's thickness -- with the
result drawn live. A menu click keeps the dialog, which is
the path for an exact value; a typed number here (``0.1`` then Enter) is exact
too.

**The document is never touched until the commit.** Every frame the op's kernel
is run over the *snapshot taken at the press* -- not over the previous frame's
output, which would compound -- and the result is drawn by swapping a fresh GPU
model into the object's cache entry. The
entry keeps the key the document's own mesh has, so ``sync`` does not rebuild it
mid-drag, and release, commit and cancel all evict it, which is what makes the
document's next ``sync`` the one that rebuilds from truth. Consequently:

* Esc restores nothing, because nothing was changed -- the document is
  byte-identical (``rev`` included);
* Enter or a click runs ``clay_ops.run`` **once**, which is one undo step, and
  makes the op the recent op (``recent_op``), so the adjust card appears after.

**Throttle.** A preview that costs more than ``SLOW_PREVIEW_S`` is recomputed at
most every ``SLOW_INTERVAL_S`` (and always on the commit, which runs the real op
at the pointer's final value). The numbers come from the drag-preview incident
-- a 200k-triangle import at 368 ms a frame against 92 once the per-material
layouts were cached (``DragOps._preview_positions``) -- where a fixed per-motion
cost is what made the viewport feel stuck: past about 50 ms a frame a drag stops
following the hand, and a coarser cadence with an exact release beats it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .view import ClayView

#: A preview slower than this (seconds) switches to the coarser cadence below.
SLOW_PREVIEW_S = 0.05
#: ...recomputing at most this often (seconds).
SLOW_INTERVAL_S = 0.10

#: The clock the throttle reads; a test replaces it rather than sleeping.
_now = time.perf_counter


@dataclass
class _OpDrag:
    """One live op drag. Everything is recorded at the press."""

    op: Any
    #: Every parameter, resolved (defaults plus whatever the dialog last
    #: remembered) -- the driven one is overwritten per frame.
    params: dict[str, float]
    #: ``uid -> (mesh, selection)`` as the press found them -- the base every
    #: frame's kernel run starts from.
    bases: dict[int, tuple[Any, Any]]
    kernel: Any
    start: tuple[float, float]
    centre: np.ndarray
    #: Where the driven parameter started, for a ``fraction`` drag.
    origin: float
    value: float = 0.0
    previewed: float | None = None
    refused: str = ""
    cost: float = 0.0
    at: float = field(default=-1e9)


class OpDragOps:
    """``ClayView``'s op drag. See the module docstring."""

    def begin_op_drag(
        self: ClayView, doc: Any, op: Any, remembered: dict[str, float] | None = None
    ) -> bool:
        """Start driving ``op.drag.param`` with the pointer. -> whether one started.

        Refuses, quietly, whatever a keyboard drag refuses -- another grab owns
        the mouse, nothing is selected in an element mode. The caller falls back
        to the dialog on ``False``.
        """
        from .....kernels.mesh import drag as bdrag
        from .....kernels.mesh.selection import _element_pickable
        from .. import ops as clay_ops

        spec = getattr(op, "drag", None)
        if spec is None or self._grab is not None:
            return False
        if doc.element_mode == "object" or not doc.element_sel:
            return False
        bases: dict[int, tuple[Any, Any]] = {}
        for uid, sel in doc.element_sel.items():
            try:
                obj = doc.by_uid(uid)
            except KeyError:
                continue
            if _element_pickable(obj):
                bases[uid] = (obj.mesh, sel)
        centre = self.element_centre(doc)
        if not bases or centre is None:
            return False
        params = clay_ops.resolve_params(op, dict(remembered or {}))
        param = next(p for p in op.params if p.name == spec.param)
        self._op_drag = _OpDrag(
            op=op,
            params=params,
            bases=bases,
            kernel=clay_ops.kernel_func(spec.kernel),
            start=self._last_mouse,
            centre=np.asarray(centre, dtype="f8"),
            origin=float(params[param.name]),
        )
        self._op_drag.value = self._op_drag_value(doc)
        self.drag_input = bdrag.DragInput()
        self.drag_hud = self._op_drag_hud()
        self._grab = "opdrag"
        self._render_dirty = True
        return True

    # -- the value ----------------------------------------------------------

    def _op_drag_value(self: ClayView, doc: Any) -> float:
        """The driven parameter's value at the pointer's current place.

        A typed number replaces the pointer outright (and an unparseable half,
        like a bare ``-``, keeps the last value rather than jumping). Clamped to
        the ``Param``'s own range: ``run`` would clamp it anyway, and a preview
        that showed an unclamped value would disagree with the commit.
        """
        od = self._op_drag
        spec = od.op.drag
        param = next(p for p in od.op.params if p.name == spec.param)
        typed = self.drag_input.value()
        if self.drag_input.typed:
            value = od.value if typed is None else typed
        elif spec.kind == "distance":
            px = math.hypot(self._last_mouse[0] - od.start[0], self._last_mouse[1] - od.start[1])
            value = px * self._metres_per_pixel(od.centre)
        else:
            width = max(float(self._rect[2]), 1.0)
            span = param.high - param.low
            value = od.origin + (self._last_mouse[0] - od.start[0]) / (width * 0.5) * span * 0.5
        return min(max(float(value), param.low), param.high)

    def _metres_per_pixel(self: ClayView, point: np.ndarray) -> float:
        """World metres one screen pixel spans at *point*'s depth."""
        distance = float(np.linalg.norm(np.asarray(self.camera.position, dtype="f8") - point))
        height = max(float(self._rect[3]), 1.0)
        return 2.0 * max(distance, 1e-6) * math.tan(math.radians(self.camera.fov * 0.5)) / height

    def _op_drag_hud(self: ClayView) -> str:
        od = self._op_drag
        spec = od.op.drag
        param = next(p for p in od.op.params if p.name == spec.param)
        name = od.op.name.replace("-", " ").capitalize()
        label = param.label.split(" (")[0]
        unit = " m" if spec.kind == "distance" else ""
        text = f"{name} {label} {od.value:.3f}{unit}"
        if self.drag_input.typed:
            text += f"  = {self.drag_input.typed}"
        if od.refused:
            text += f"  ({od.refused})"
        return text

    # -- one frame ----------------------------------------------------------

    def _op_drag_update(self: ClayView, doc: Any) -> None:
        """Re-derive the value and, when it changed, redraw the preview.

        Only when the value *changed*: a pointer that jitters by a pixel at a
        fixed value would otherwise rerun the kernel for nothing. And throttled
        once a preview has proved slow (the module docstring).
        """
        od = self._op_drag
        od.value = self._op_drag_value(doc)
        self._render_dirty = True
        # The HUD is drawn *after* the preview, which is what knows whether the
        # kernel refused this value.
        if od.value != od.previewed and not (
            od.cost > SLOW_PREVIEW_S and _now() - od.at < SLOW_INTERVAL_S
        ):
            self._op_drag_preview(doc, od.value)
        self.drag_hud = self._op_drag_hud()

    def _op_drag_preview(self: ClayView, doc: Any, value: float) -> None:
        """Run the kernel over every base at *value* and draw the results."""
        from .....kernels.mesh.elements import OpError

        od = self._op_drag
        started = _now()
        params = dict(od.params)
        params[od.op.drag.param] = value
        built: dict[int, Any] = {}
        refusal = ""
        for uid, (mesh, sel) in od.bases.items():
            try:
                built[uid], _ = od.kernel(mesh, sel, **params)
            except OpError as error:
                refusal = str(error)
                break
        od.refused = refusal
        if refusal:
            # The last good picture stays; the HUD says why.
            od.previewed = value
            return
        for uid, mesh in built.items():
            self._swap_preview(doc, uid, mesh)
        od.previewed = value
        od.at = _now()
        od.cost = od.at - started

    def _swap_preview(self: ClayView, doc: Any, uid: int, mesh: Any) -> None:
        """Put *mesh* on screen for *uid* without telling the document.

        The cache entry keeps its key (so ``sync`` does not see a changed
        document and rebuild it) and has its GPU model replaced.
        """
        entry = self._cache.get(uid)
        if entry is None:
            return
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            return
        fresh = self._build(obj, doc, entry.key, mesh)
        entry.gpu.release()
        entry.gpu, entry.model = fresh.gpu, fresh.model

    def _op_drag_motion(self: ClayView, doc: Any) -> None:
        self._op_drag_update(doc)

    # -- ending -------------------------------------------------------------

    def _op_drag_evict(self: ClayView) -> None:
        """Drop every previewed entry, so the next ``sync`` rebuilds from the document."""
        od = self._op_drag
        for uid in od.bases:
            entry = self._cache.pop(uid, None)
            if entry is not None:
                entry.gpu.release()

    def _op_drag_end(self: ClayView) -> None:
        self._op_drag = None
        self._grab = None
        self._clear_drag_input()
        self._render_dirty = True

    def _op_drag_cancel(self: ClayView, doc: Any) -> None:
        """Esc: nothing was ever written, so there is nothing to put back.

        Not even a ``doc.touch()``: the document's ``rev`` is the same before the
        drag and after Esc, and the redraw comes from ``_render_dirty`` plus the
        evicted entries the next ``sync`` rebuilds.
        """
        self._op_drag_evict()
        self._op_drag_end()

    def _op_drag_commit(self: ClayView, doc: Any) -> None:
        """Enter or a click: run the op once, at the value the pointer has now.

        Recomputed from the pointer rather than read off the last preview,
        because a throttled preview may be a frame or two behind the hand. Run
        through ``clay_ops.run`` -- one undo step, a refusal toasted, the op
        recorded for the adjust card -- and the value is remembered for the
        dialog, as a dialog Apply remembers its own.
        """
        from .. import ops as clay_ops

        od = self._op_drag
        value = self._op_drag_value(doc)
        self._op_drag_evict()
        params = dict(od.params)
        params[od.op.drag.param] = value
        op = od.op
        self._op_drag_end()
        clay_ops.run(_ToastCtx(self), doc, op, **params)
        remembered = getattr(self.state, "op_params", None)
        if isinstance(remembered, dict):
            remembered.setdefault(op.name, {}).update(params)


class _ToastCtx:
    """The ctx ``clay_ops.run`` wants, over a view that may have no app ctx.

    A headless view (most of this package's own tests) has nothing to toast
    through; ``ClayView._toast`` already knows how to drop a message in that
    case, so a refusal goes through it rather than assuming a ``toast`` method.
    ``app_ctx`` is forwarded for the ops that read ``ctx.state``.
    """

    def __init__(self, view: Any) -> None:
        self._view = view

    def toast(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._view._toast(message)

    def __getattr__(self, name: str) -> Any:
        app_ctx = self._view.app_ctx
        if app_ctx is None:
            raise AttributeError(name)
        return getattr(app_ctx, name)
