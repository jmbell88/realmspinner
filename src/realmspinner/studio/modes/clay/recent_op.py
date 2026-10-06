"""The last parameterised element op, so it can be adjusted in place or repeated.

The 2026-09-07 audit's clay-10 removed ``ClayState.last_op`` because it was
written by every ``clay_ops.run`` and read by nothing. **This record has two
readers**, which is the whole of why it is back: the adjust card
(``ui/panes/adjust.py``) re-runs it at new values, and the ``repeat-last`` op
re-runs it on the current selection.

It lives on the *document* (``ClayDoc.recent_op``), not on ``ClayState``, for
``element_mode``'s reason: the registry's only question is ``enabled(doc)``, and
an agent drives its own tab's document, so "repeat" can only ever mean *that
tab's* last op.

**The card is live only while the history head is still the step the op
pushed.** Any later edit, undo or selection change hides it, because adjusting
means "undo that step and run the op again from the state it started in", and
that is only the same thing as "change the number" while nothing has happened
since. Adjusting is one undo step, not two: the old step is undone with its redo
branch kept (so a refusal can put it back) and the re-run pushes the one step
that replaces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class RecentOp:
    """One op's last run: what it was, what it started from, what it left."""

    op_name: str
    #: The *clamped* values the run used, so a repeat or an adjust starts from
    #: exactly what happened rather than from what was typed.
    params: dict[str, float]
    #: The element mode and per-object selection the op started from. Restored
    #: before an adjust re-runs it: selection is not undoable, so undoing the
    #: step puts the mesh back and leaves the selection on the op's *output*.
    element_mode: str
    selection: dict[int, Any]
    #: ``history.head`` after the op's step landed -- the card's liveness test.
    step_id: int
    #: What the op left selected. A change to any of it hides the card, and it is
    #: what is put back when an adjust is refused.
    result_mode: str = ""
    result_selection: dict[int, Any] = field(default_factory=dict)

    def live(self, doc: Any) -> bool:
        """Whether *doc* is still exactly as the op left it."""
        if doc.history.head != self.step_id or doc.element_mode != self.result_mode:
            return False
        sel = doc.element_sel
        if sel.keys() != self.result_selection.keys():
            return False
        return all(sel[uid] is kept for uid, kept in self.result_selection.items())


def snapshot(doc: Any) -> tuple[str, dict[int, Any]]:
    """The element mode and a copy of the per-object selection, before an op runs."""
    return doc.element_mode, dict(doc.element_sel)


def record(
    doc: Any, op_name: str, params: dict[str, float], before: tuple[str, dict[int, Any]]
) -> RecentOp:
    """Make *op_name*'s run the document's recent op. Call after its step landed."""
    recent = RecentOp(
        op_name=op_name,
        params=dict(params),
        element_mode=before[0],
        selection=before[1],
        step_id=doc.history.head,
        result_mode=doc.element_mode,
        result_selection=dict(doc.element_sel),
    )
    doc.recent_op = recent
    return recent


def _restore(doc: Any, mode: str, selection: dict[int, Any]) -> None:
    if doc.element_mode != mode:
        doc.set_element_mode(mode)
    doc.clear_element_sel()
    present = {obj.uid for obj in doc.objects}
    for uid, sel in selection.items():
        if uid in present:
            doc.set_element_sel(uid, sel)


@dataclass(frozen=True)
class AdjustResult:
    ok: bool
    #: Why it was refused, for the card to print; empty on success.
    message: str = ""


class _Quiet:
    """A ``ctx`` that keeps an op's refusal instead of toasting it.

    The card shows the reason itself, and a toast per keystroke while a number
    is being typed is the failure ``props._apply_transform`` already records.
    Everything but ``toast`` is the real ctx's.
    """

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self.messages: list[str] = []

    def toast(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.messages.append(message)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ctx, name)


def adjust(ctx: Any, doc: Any, **changes: float) -> AdjustResult:
    """Re-run the recent op at new values, replacing its step with one new step.

    Undo the op's step (keeping it redoable), put the starting selection back,
    run the op again, and let ``clay_ops.run`` record the new step -- so the
    result is one undo step, equal to running the op directly at the final
    values. **A refusal puts everything back**: the old step is redone and the
    old selection restored, so a width too large for the mesh leaves the last
    good result on screen and a reason in the card.
    """
    from . import ops as clay_ops

    recent = doc.recent_op
    if recent is None or not recent.live(doc):
        return AdjustResult(False, "The model has changed since; run the operation again.")
    op = clay_ops.get(recent.op_name)
    history = doc.history
    if not history.undo(doc):  # pragma: no cover - live() implies a step to undo
        return AdjustResult(False, "Nothing to adjust.")
    _restore(doc, recent.element_mode, recent.selection)
    quiet = _Quiet(ctx)
    ok = clay_ops.run(quiet, doc, op, **(recent.params | changes))
    if ok and doc.recent_op is not recent:
        return AdjustResult(True)
    # Refused, or ran and pushed nothing: either way the old result comes back.
    history.redo(doc)
    _restore(doc, recent.result_mode, recent.result_selection)
    doc.recent_op = recent
    reason = quiet.messages[-1] if quiet.messages else "That value changes nothing."
    return AdjustResult(False, reason)
