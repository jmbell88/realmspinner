"""One registry of everything Clay can *do*, and no imgui anywhere in it.

Three surfaces invoke Clay's operations -- the right-mouse context menu, the
buttons in the tools pane, and ``clay_mode.handle_key`` -- and before this
module they each had their own list. Three lists means three answers to "is
Extrude available in vertex mode", and the interesting one is always the one
nobody updated: a key that fires an op the menu greys out, or a menu row for an
op the key path never learned about.

So there is one list. :data:`OPS` is the whole of what is invocable, each entry
carrying the modes it applies to, whether it is enabled right now and the key
that fires it; how it groups in a menu is :mod:`.menutree`'s table. The menu
renders it, the pane renders a subset of it, and the key handler looks up by
key -- and none of them decides anything.

**Nothing here imports imgui**, which is what keeps the registry testable: an
``Op``'s ``enabled`` predicate is a function of a document, so "Flip Normals is
greyed out with an edge selected" is a plain assertion rather than a screenshot.
The pane layer (``studio/modes/clay/ui/panes/menu.py``) is the only thing that knows a popup
exists.

**An op that changes geometry freezes the object's generator.** A box whose
faces have been extruded is no longer describable by "box, size 1" -- the
properties panel would offer a size field that silently discards the edit the
moment it was touched. The freeze is ``Document.set_mesh``'s, not any op's:
saying "``run`` clears it in one place, for every op" was not true of ``run``
at all -- only ``run_mesh_op`` did it, while Delete and Mirror
went straight to ``set_mesh`` and kept a generator that would rebuild over
them. Putting it where the geometry actually changes is what
makes the sentence true for ops that do not exist yet.

**A refusal is a toast, not an exception.** Every op raises
:class:`~.clay.elements.OpError` with a sentence naming what it refused and what
to do instead; :func:`run` catches exactly that, shows it and records no edit.
Anything else propagates, because an ``IndexError`` out of a topology op is a
bug and swallowing it would leave a half-built mesh on screen with no clue why.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import recent_op as _recent

__all__ = [
    "OPS",
    "Op",
    "Param",
    "by_key",
    "defaults_for",
    "menu",
    "reason_for",
    "register",
    "run",
    "run_mesh_op",
    "run_object_op",
]

# The element modes an op can appear in. "object" is the fourth and is not an
# element mode; ops that name it are the object-level ones (duplicate, mirror,
# group, origin) that were previously hardcoded in the tools pane. (The 2026-10-07
# audit's clay-42: this named a bake op that Clay no longer has.)
ALL_MODES: tuple[str, ...] = ("object", "vertex", "edge", "face")
ELEMENT_MODES: tuple[str, ...] = ("vertex", "edge", "face")


@dataclass(frozen=True)
class Param:
    """One number an op takes, and everything a widget needs to offer it.

    Described rather than drawn so the pane builds every popup from one loop.
    ``warn`` is shown under the field -- a weld distance of 0 keeps the shapes
    as separate shells, and a user who finds that out from the result is a
    user who undid it to learn what the number meant.

    **``boolean`` and ``choices`` are widget kinds, not storage kinds.** The
    ops that reached for a number on 2026-09-10 because this dataclass had
    no other shape -- ``mirror-copy``'s ``axis`` ("axis (0=X, 1=Y, 2=Z)") is
    the one that is left -- are the tell: the *label* was doing the widget's
    job because the field couldn't. ``default`` stays a ``float`` and :func:`run` still
    clamps to ``low``/``high`` regardless of which of the three this is, so a
    checkbox writes 0.0/1.0 and a combo writes its index -- exactly what the
    bare int field they replaced already wrote, and why converting an op to
    either costs nothing beyond this dataclass and the pane's drawing loop.

    They are mutually exclusive -- one field is a toggle or a named set of
    options, never both -- and each pins its own range so nothing downstream
    has to ask "which kind is this and what do its bounds mean": a boolean is
    always 0..1, and a choice's ``high`` is always ``len(choices) - 1``, the
    same bound :func:`run` was already going to clamp it to. Three values are
    a choice, not a toggle, which is why this is two kinds and not one.
    """

    name: str
    label: str
    default: float
    step: float = 0.01
    low: float = 0.0
    high: float = 1e6
    integer: bool = False
    boolean: bool = False
    choices: tuple[str, ...] = ()
    warn: str = ""

    def __post_init__(self) -> None:
        if self.boolean and self.choices:
            raise ValueError(
                f"Param {self.name!r}: cannot be both boolean and a choice"
            )
        if self.boolean and (self.low, self.high) != (0.0, 1.0):
            raise ValueError(f"Param {self.name!r}: a boolean's range must be 0..1")
        if self.choices and (self.low, self.high) != (0.0, float(len(self.choices) - 1)):
            raise ValueError(
                f"Param {self.name!r}: a choice's low/high must span its "
                f"choices (0..{len(self.choices) - 1})"
            )

    @property
    def stores_int(self) -> bool:
        """Whether :func:`run` should hand the op a whole number.

        ``integer``, ``boolean`` and ``choices`` all narrow to one under the
        hood -- a checkbox writes 0/1, a combo writes its index -- so this is
        the one place that answers "is this param whole", rather than every
        caller re-deriving the same ``or`` and one of them eventually missing
        a kind.
        """
        return self.integer or self.boolean or bool(self.choices)


@dataclass(frozen=True)
class DragSpec:
    """How the mouse drives one of an op's parameters, for the viewport's op drag.

    **Metadata, not a second implementation.** ``kernel`` is the same dotted
    ``kernels.mesh.ops_*`` name the op's own ``run`` wraps (``_element`` records
    it on the callable it returns, and a test holds the two equal), so a live
    preview and the committed run are the same function at the same values.

    ``kind`` says what the pointer's travel means: ``"distance"`` is screen
    pixels turned into world metres at the selection's depth (the value starts
    at zero and grows with the distance from where the key was pressed);
    ``"fraction"`` slides the value across the parameter's own ``low..high``
    range as the pointer crosses the viewport. Either way the result is clamped
    to the :class:`Param`'s range, so the drag cannot ask for what ``run`` would
    clamp anyway. The agent schema and the dialog both ignore the field.
    """

    param: str
    kind: str
    kernel: str

    def __post_init__(self) -> None:
        if self.kind not in ("distance", "fraction"):
            raise ValueError(f"DragSpec kind {self.kind!r}: expected distance or fraction")


@dataclass(frozen=True)
class Op:
    """One invocable operation.

    ``run(ctx, doc)`` does the work and is free to push whatever steps it needs;
    ``enabled(doc)`` answers whether it can right now, and is what greys the
    menu row and disables the button. ``key`` is the shortcut *label* as well as
    the binding, so the menu and the shortcuts sheet cannot disagree about it.
    """

    name: str
    label: str
    modes: tuple[str, ...]
    run: Callable[..., Any]
    enabled: Callable[[Any], bool] = lambda doc: True
    key: str = ""
    params: tuple[Param, ...] = field(default=())
    """The numbers the pane pops a dialog for, or empty for a bare action."""
    hint: str = ""
    """One sentence under the dialog's title, about the op rather than a field.

    ``Param.warn`` is about a *number* -- what a weld distance of 0 will do to
    a merge -- and belongs under the field it qualifies. This is about the
    op: what it is for, and when the op next to it is the one you meant. Only
    the parameterised ops can show it, because only they open a dialog, which
    is the right restriction: a bare action gives no moment to read anything.
    """
    drag: DragSpec | None = None
    """Set on the ops whose first number is best found with the mouse (inset).
    The op's *key* then starts a live drag instead
    of opening the dialog; a menu click keeps the dialog, which is the path for
    an exact value. ``None`` for everything else."""
    reason: Callable[[Any], str] = lambda doc: ""
    """Why ``enabled(doc)`` is refused right now, or ``""`` when it is not.

    The 2026-09-06 audit's clay-07: every refused op greyed out in the context
    menu, the tools pane and the Delete button with nothing anywhere naming the
    gate that refused it -- ``hint`` describes what an op *does*, not why it is
    currently unavailable. Called only through :func:`reason_for`, which is the
    one place that decides *whether* to call it, so a ``reason`` never needs to
    re-check ``enabled`` itself and cannot disagree with it about that.
    """


OPS: list[Op] = []


def register(op: Op) -> Op:
    """Add an op to the registry, refusing a duplicate name."""
    if any(existing.name == op.name for existing in OPS):
        raise ValueError(f"an op named {op.name!r} is already registered")
    OPS.append(op)
    return op


def menu(mode: str) -> list[Op]:
    """Every op that applies in *mode*, in registration order."""
    return [op for op in OPS if mode in op.modes]


def by_key(mode: str, key: str) -> Op | None:
    """The op *key* fires in *mode*, or ``None``."""
    for op in menu(mode):
        if op.key and op.key == key:
            return op
    return None


def get(name: str) -> Op:
    for op in OPS:
        if op.name == name:
            return op
    raise KeyError(f"no op named {name!r}")


def reason_for(op: Op, doc: Any) -> str:
    """Why *op* is greyed for *doc* right now, or ``""`` when it is not.

    Gated on ``op.enabled`` rather than trusting ``op.reason`` to also answer
    "whether": the sentence a caller draws must never disagree with the
    predicate that actually decides the row, which is exactly the drift the
    2026-09-06 audit's clay-07 finding warns against -- a reason is only ever
    consulted once ``enabled(doc)`` has already said no.
    """
    if op.enabled(doc):
        return ""
    return op.reason(doc)


# --- running ----------------------------------------------------------------


def toast(ctx: Any, message: str) -> None:
    """A refusal, shown. ``Ctx.toast(text, level)`` is the whole API.

    It used to reach for a ``ctx.toasts`` attribute that the real ``Ctx`` has
    never had, so every refusal this module raises -- "can't delete the last
    object", every per-object one -- was raised, caught, formatted and thrown
    away in the running app, and only the test doubles ever saw one.
    """
    ctx.toast(message, "error")


def defaults_for(op: Op) -> dict[str, float]:
    """The op's parameters at their defaults, as a fresh dict."""
    return {param.name: param.default for param in op.params}


def format_for(param: Param) -> str:
    """The printf format ``input_float`` should draw this parameter with.

    imgui's own default is ``"%.3f"``, and both weld distances in this registry
    default to 1e-4 -- so the field read ``0.000``, the step arrows moved it by
    an amount no digit shown could express, and the number a user typed came
    back as a different one. A control whose value cannot be read is not a
    control.

    **Derived from the parameter rather than declared on it.** A ``format``
    field would be one more thing to remember, and the next sub-millimetre
    parameter would arrive without it and land in exactly the same hole; the
    step and the default already say what precision the number is kept at.
    Widened only downwards, so every parameter that was legible at three
    decimals still reads exactly as it did.
    """
    scale = min(abs(param.step) or 1.0, abs(param.default) or 1.0)
    decimals = 3
    while decimals < 9 and scale < 10.0**-decimals:
        decimals += 1
    return f"%.{decimals}f"


def resolve_params(op: Op, params: dict[str, Any]) -> dict[str, Any]:
    """*op*'s parameters as ``run`` will use them: defaults filled, ranges clamped.

    ``run`` is the choke point every surface funnels through, and a live drag
    previews through the kernel directly -- so the clamp is one function both
    call, and a preview can never show a value the commit would change.
    """
    from ....kernels.mesh.elements import OpError

    values = defaults_for(op) | params
    for param in op.params:
        value = float(values[param.name])
        # The 2026-10-07 audit's clay-36: ``min(max(nan, lo), hi)`` returns NaN
        # (every comparison with NaN is false), so a NaN that reached ``run``
        # sailed through the clamp -- Mirror Copy put an object at NaN and Snap
        # and Assign raised a bare ValueError. A refusal, in the same type every
        # other op refusal uses, so ``run`` toasts it and records no edit.
        if not math.isfinite(value):
            raise OpError(f"{param.label} must be a finite number.")
        value = min(max(value, param.low), param.high)
        values[param.name] = int(value) if param.stores_int else value
    return values


def kernel_func(dotted: str) -> Callable[..., Any]:
    """The ``kernels.mesh.ops_*`` function a :class:`DragSpec` names."""
    import importlib

    module, func = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(f"realmspinner.kernels.mesh.{module}"), func)


def run(ctx: Any, doc: Any, op: Op, **params: Any) -> bool:
    """Invoke an op, turning a refusal into a toast. -> whether it ran.

    Missing parameters fall back to their declared defaults, so a caller that
    has no remembered values -- the key path, a test -- gets the same result the
    popup would have produced with the fields untouched.

    **Declared parameters are clamped here.** The popup clamps its live fields
    too (``studio/modes/clay/ui/panes/menu.py``), but that is a UX affordance on one surface --
    the key path, the tools pane and every test call arrive with whatever the
    caller had remembered, and a subdivision at ``levels=99`` is not a refusal
    an op should have to write for itself. ``run`` is the choke point all three
    surfaces funnel through, so the range a ``Param`` declares is enforced once,
    where it cannot be bypassed.
    """
    from ....kernels.mesh.elements import OpError

    if not op.enabled(doc):
        return False
    try:
        values = resolve_params(op, params)
    except OpError as error:
        toast(ctx, str(error))
        return False
    head = doc.history.head
    before = _recent.snapshot(doc)
    mark = doc.history.mark()
    try:
        result = op.run(ctx, doc, **values)
    except OpError as error:
        toast(ctx, str(error))
        doc.history.collapse_since(mark)
        return False
    except BaseException:
        # The 2026-10-03 audit's clay-62: a bug (a MemoryError, a numpy or
        # manifold error) deliberately propagates, but it used to leave the
        # gesture ``mark()`` opened -- ``UndoStack._open_gestures`` stuck at 1
        # disables undo eviction for the document for the rest of the session.
        # Fold and release on the way out; the exception is still the caller's.
        doc.history.collapse_since(mark)
        raise
    # An op that refuses *per object* -- ``run_mesh_op`` toasts and carries on
    # to the next one -- says so by returning False rather than by raising, so
    # a caller still learns that nothing happened.
    if result is False:
        doc.history.collapse_since(mark)
        return False
    _one_step(doc, op, mark, head)
    # What the adjust card and Repeat Last read. Only a parameterised op that
    # started in an element mode and actually pushed a step: a bare action has
    # nothing to adjust, an object-level op is a different selection model, and
    # an op that pushed nothing (a no-op) has no step for the card to be live
    # against.
    if op.params and before[0] != "object" and doc.history.head != head:
        _recent.record(doc, op.name, values, before)
    return True


def _one_step(doc: Any, op: Op, mark: int, head: int) -> None:
    """Fold everything the op pushed into a single, named step.

    **One press, one Ctrl+Z.** An op is free to push whatever steps it needs and
    most push one, but the composed ones do not: Mirror pushed a transform and a
    mesh change, and ``run_mesh_op`` pushes one ``set_mesh`` *per object*, so
    extruding faces across three objects cost three presses to undo -- a
    gesture the user made once. ``collapse_since`` is the primitive that was
    built for exactly this and had one caller.

    Then the name. A fold reads as "compound" in the history panel, which says
    nothing about what is being undone; the op knows what it was, and
    ``Op.label`` is already the word on the button. The trailing ellipsis of a
    parameterised op's label goes -- "Inset Faces..." is an invitation to a dialog,
    and this is a record of something that happened.

    ``head`` is the stack's head **before the op ran**, and it has to be taken
    by the caller rather than read here: read at the top of this function it is
    already the post-op head, so the guard below only fired when the *collapse*
    had pushed -- which is to say a multi-object op got its name and a
    single-object one silently kept "mesh". Serials rather than a depth
    comparison, because the byte budget can evict an older step while this one
    is pushed and leave the two counts equal.
    """

    history = getattr(doc, "history", None)
    if history is None:  # pragma: no cover - every document has one
        return
    history.collapse_since(mark)
    top = history.top
    # Only when the op actually pushed something: a select-all or a frame
    # changes no document, and labelling the *previous* step with this op's
    # name would be a lie in the one place a user goes to read what happened.
    if top is not None and history.head != head:
        top.label = op.label.rstrip(".")


def run_mesh_op(
    ctx: Any, doc: Any, func: Callable[..., Any], /, **params: Any
) -> bool:
    """Apply a ``(mesh, sel) -> (mesh, sel)`` op to every object with a selection.

    The loop, the refusal handling and the generator freeze all live here, so an
    op is registered by naming its function rather than by writing this out
    again -- and a refusal on one object does not abandon the others, which is
    what a user selecting faces across two objects means by pressing the button
    once.
    """
    from ....kernels.mesh.elements import OpError

    ran = False
    for uid in list(doc.element_sel):
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            continue
        try:
            mesh, sel = func(obj.mesh, doc.element_sel_of(uid), **params)
        except OpError as error:
            toast(ctx, str(error))
            continue
        doc.set_mesh(uid, mesh, select=sel)
        ran = True
    return ran


def run_object_op(
    ctx: Any, doc: Any, func: Callable[[Any, Any], Any], /, *, uids: Iterable[int] | None = None
) -> bool:
    """Apply ``func(doc, obj)`` to every selected object. -> whether any ran.

    ``run_mesh_op``'s sibling for the object-level ops, and it exists for the
    same reason: seven ops here wrote out the same four lines -- snapshot the
    selection, look each uid up, tolerate one that has gone, toast a refusal and
    carry on -- and only two of them wrote out all four. The ones that skipped
    the ``KeyError`` guard crash on an object deleted between the snapshot and
    the loop; the ones that skipped the ``OpError`` catch abandon the rest of
    the selection when one object refuses.

    The snapshot is taken before anything runs because ``doc.selection`` is live
    and several of these ops change it.
    """
    from ....kernels.mesh.elements import OpError

    ran = False
    for uid in list(doc.selection if uids is None else uids):
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            continue
        try:
            func(doc, obj)
        except OpError as error:
            toast(ctx, str(error))
            continue
        ran = True
    return ran


# --- predicates -------------------------------------------------------------


def has_objects(doc: Any) -> bool:
    return bool(doc.selection)


def has_elements(doc: Any) -> bool:
    return bool(doc.element_sel)


def has_any_selection(doc: Any) -> bool:
    """Select None's gate, graded against the selection the mode means.

    Object mode's selection is ``doc.selection``; an element mode's is
    ``doc.element_sel``. ``has_elements`` alone would grey Select None in object
    mode for a document with three objects picked.
    """
    if doc.element_mode == "object":
        return bool(doc.selection)
    return has_elements(doc)


def has_two_visible(doc: Any) -> bool:
    """Two selected objects the user can actually see. Merge's predicate, and
    the only one that has to look past ``selection`` at what is in it."""
    return sum(1 for obj in doc.objects if obj.uid in doc.selection and obj.visible) >= 2


def has_two_or_more_selected(doc: Any) -> bool:
    """At least two -- Group Selected's and Parent to Last Selected's own
    gate. Grouping or parenting one object is the identity (there is
    nothing else to fold into the new empty, or to reparent onto the
    topmost one), the same "enabled but does nothing" trap ``has_two_visible``
    already refuses for Merge Objects.
    """
    return len(doc.selection) >= 2


def in_mode(*modes: str) -> Callable[[Any], bool]:
    def check(doc: Any) -> bool:
        return doc.element_mode in modes and bool(doc.element_sel)

    return check


# --- reasons ------------------------------------------------------------
#
# One function per predicate above, each naming the gate that predicate
# checks rather than restating the app's general "nothing selected" toast.
# clay-07 (2026-09-06 audit): none of these existed, so Merge Objects and
# every element op greyed out with nothing anywhere saying why --
# ``op.hint`` was the only sentence attached to a disabled row, and it
# describes what the op does rather than why it is refused. Each of these
# answers only the "why"; :func:`reason_for` is what decides whether to ask.


def _has_objects_reason(doc: Any) -> str:
    return "" if has_objects(doc) else "Select an object first."


def _has_elements_reason(doc: Any) -> str:
    return "" if has_elements(doc) else "Select something in the viewport first."


def _has_any_selection_reason(doc: Any) -> str:
    return "" if has_any_selection(doc) else "Select something in the viewport first."


def _shade_enabled(doc: Any) -> bool:
    """Shade Smooth/Flat's own gate, because the op is registered for two modes
    that mean different things by "the selection".

    The 2026-09-08 audit's clay-06: the row was gated on ``has_objects`` alone,
    which is an *object*-selection predicate, even though ``_shade``'s body
    reads ``doc.element_sel`` in face mode -- so an object selected with no
    faces picked drew a live, enabled button that ran an empty loop: no
    ``set_shading`` call, no history step, no toast. Grading the same thing the
    op body reads, mode for mode, is what keeps the two from disagreeing.
    """
    if doc.element_mode == "object":
        return has_objects(doc)
    return has_elements(doc)


def _shade_reason(doc: Any) -> str:
    if doc.element_mode == "object":
        return _has_objects_reason(doc)
    return _has_elements_reason(doc)


def _has_two_visible_reason(doc: Any) -> str:
    # The manual's own wording for this gate (docs/manual/30-clay.md, "Merging
    # objects"): "greys out unless two visible objects are selected".
    return "" if has_two_visible(doc) else "Select two visible objects first."


def _selection_reason(doc: Any) -> str:
    return "" if doc.selection else "Select an object first."


def _has_two_or_more_selected_reason(doc: Any) -> str:
    n = len(doc.selection)
    return "" if n >= 2 else f"Select at least two objects first -- {n} selected now."


def _in_mode_reason(*modes: str) -> Callable[[Any], str]:
    """A reason matching :func:`in_mode`'s own two gates, in the same order --
    the mode first, then the selection -- so the sentence shown can never
    disagree with the row it is explaining."""

    label = " or ".join(modes)

    def reason(doc: Any) -> str:
        if doc.element_mode not in modes:
            return f"Switch to {label} mode first."
        if not doc.element_sel:
            return "Select something first."
        return ""

    return reason


# --- the object-level ops (moved out of the tools pane) ---------------------


def _element(dotted: str) -> Callable[..., None]:
    """Register a ``kernels.mesh.ops_*`` function as an op, resolved lazily by
    name.

    Lazy so importing the registry does not drag every topology module in at
    startup, and by name so the table below reads as a list of ops rather than a
    list of imports.

    The package is named absolutely since 2026-09-17: the mesh engine used to
    be ``studio/clay/``, a relative hop from here, and is ``realmspinner.kernels.mesh``
    now that it is a kernel every layer may reach rather than one mode's
    private package.
    """
    module, func = dotted.rsplit(".", 1)

    def call(ctx: Any, doc: Any, **params: Any) -> bool:
        import importlib

        target = getattr(importlib.import_module(f"realmspinner.kernels.mesh.{module}"), func)
        return run_mesh_op(ctx, doc, target, **params)

    # What ``DragSpec.kernel`` is held equal to, so a drag preview and the
    # committed run cannot name two different functions.
    call.kernel = dotted  # type: ignore[attr-defined]
    return call


def _extrude(ctx: Any, doc: Any, **params: Any) -> bool:
    """Extrude, dispatched on the mode -- one row and one key, as Merge Faces is.

    "Extrude" means the same thing in all three modes (pull this out and wall in
    the gap it leaves) and only the implementation differs, so three rows would
    be exposing the implementation. The vertex branch is the one that is not a
    simple rename: a mesh stores no wire edges, so it extrudes the *border*
    edges the selection implies -- see ``ops_topo.extrude_verts``.
    """
    from ....kernels.mesh import ops_topo

    which = {
        "vertex": ops_topo.extrude_verts,
        "edge": ops_topo.extrude_edges,
        "face": ops_topo.extrude_faces,
    }.get(doc.element_mode)
    return which is not None and run_mesh_op(ctx, doc, which, **params)


def _duplicate(ctx: Any, doc: Any, **_: Any) -> None:
    from ....kernels.mesh import selection

    del ctx
    selection.duplicate_selected(doc)


def _join(ctx: Any, doc: Any, weld: float = 1e-4, **_: Any) -> None:
    """Merge every selected object into the topmost one in the outliner.

    Document order rather than "the one clicked last": ``doc.selection`` is a
    set and carries no order at all, so a target read out of it would be
    whichever object the hash happened to put first -- and the name, transform
    and material default that the merge keeps are the *target's*, which makes
    that an arbitrary answer to a question the user can see the answer to.

    The geometry is ``clay.ops.join``'s and the bookkeeping is
    ``ClayDoc.join_objects``'s, which is this module's usual boundary; the one
    thing that happens here is the selection afterwards, and it is deliberately
    the survivor rather than nothing -- the user has one shape now and the
    gizmo should be on it.

    **Visible objects only**, matching ``visible=False``'s documented meaning
    that an object does not render, does not export and cannot be picked. A
    merge is the one op where absorbing an unseen object is not merely odd but
    invisible in its result too: the geometry arrives inside the survivor, which
    *is* shown. ``_select_all`` no longer hands over hidden objects, so this is
    the second half -- an object hidden after it was selected.

    **Every input's own world matrix, not its local TRS (tranche 3: scene
    structure).** Two objects under different parents -- or one parented and
    one not -- are not correctly related by composing their own TRS alone;
    ``ops.join``'s own ``world`` parameter is exactly for this, and the
    result still lands in the target's local frame (``join``'s own "in the
    first object's frame" contract), so the target's transform is untouched
    either way.
    """
    from ....kernels.mesh import ops as clay_ops_geom

    uids = [obj.uid for obj in doc.objects if obj.uid in doc.selection and obj.visible]
    objs = [doc.by_uid(uid) for uid in uids]
    worlds = [doc.world_matrix(uid) for uid in uids]
    mesh = clay_ops_geom.join(objs, eps=float(weld), world=worlds)
    doc.join_objects(uids[0], mesh, uids[1:])
    doc.select([uids[0]])


def mirror(ctx: Any, doc: Any, axis: int, **_: Any) -> bool:
    from ....kernels.mesh import ops as clay_ops_geom

    mirrored: list[int] = []

    def one(doc: Any, obj: Any) -> None:
        # The 2026-10-07 audit's clay-39: a group's empty has no vertices, and
        # mirroring it built a fresh Mesh and pushed an undo step that changed
        # nothing. A vertex-less object is left alone, as Join and Unwrap's
        # own "nothing to act on" objects are.
        if len(obj.mesh.positions) == 0:
            return
        doc.set_mesh(obj.uid, clay_ops_geom.mirror(obj, axis).mesh)
        mirrored.append(obj.uid)

    return run_object_op(ctx, doc, one) and bool(mirrored)


def _mirror_copy(ctx: Any, doc: Any, axis: float = 0.0, offset: float = 0.0, **_: Any) -> bool:
    """Duplicate the selection and reflect each copy across a *world* plane.

    Where **Mirror X/Y/Z** (:func:`mirror`) replace an object with its own
    reflection about a plane through its own origin, this makes a *second*
    object, reflected about a plane the caller places anywhere in world
    space -- what mirroring a limb across a body's centre-line means, and
    something the per-object mirror cannot do at all. The manual's Mirror
    X/Y/Z paragraph says which is which.

    **The copy is frozen**, exactly as Mirror X/Y/Z's own result is (the
    module docstring's negative-scale rule, obeyed here because the mesh
    comes from ``ops.mirror_world``, which calls ``ops.mirror`` rather than
    negating a scale component). That freeze is normally ``Document.set_mesh``'s
    job, but a fresh insert through ``add_objects`` never calls it -- there is
    no prior mesh to compare identity against -- so it is done by hand here,
    on the object before it is inserted. Left un-frozen, the properties panel
    would still offer the source generator's size field, and touching it
    would rebuild a pristine, unmirrored primitive over the copy.

    **The plane is world space; the copy's TRS is not (tranche 3: scene
    structure).** ``mirror_world`` is handed the source's own world matrix,
    so a parented source is reflected about the plane it actually sits at,
    not about a plane read through its parent's frame. With ``world=``
    given, what comes back is the copy's new *world* placement (``mirror_
    world``'s own docstring), not local TRS to write straight onto an
    object sharing the source's parent -- ``doc.local_from_world`` converts
    it, the same door every other parented write in this file uses.
    """
    from ....kernels.geom3d import math3d as m3
    from ....kernels.mesh import document as bd
    from ....kernels.mesh import ops as clay_ops_geom

    del ctx
    taken = clay_ops_geom.UsedNames(obj.name for obj in doc.objects)
    # The 2026-10-03 audit's clay-61: a copied child must hang off its own
    # copied parent, not the original one, and a mirrored child is *not*
    # simply riding its parent's step -- the reflection is baked into each
    # copy's own mesh -- so every copy's world is kept and a child's local TRS
    # is taken relative to its parent copy's. Parents first, so that world is
    # already known.
    #
    # The 2026-10-07 audit's clay-35: only *visible* originals. Duplicate and
    # Delete leave a hidden selected object alone (``visible=False`` means it
    # does not render, export or pick), and Mirror Copy used to copy it into a
    # scene the user could not see the original in.
    originals = sorted(
        (u for u in doc.selection if doc.by_uid(u).visible),
        key=lambda u: (len(doc.ancestors(u)), doc.index_of(u)),
    )
    fresh = {uid: bd.new_uid() for uid in originals}
    copy_worlds: dict[int, Any] = {}
    made: list[Any] = []
    for uid in originals:
        source = doc.by_uid(uid)
        copy = clay_ops_geom.duplicate(source, fresh[uid], taken=taken)
        taken.add(copy.name)
        world = doc.world_matrix(uid)
        mirrored = clay_ops_geom.mirror_world(copy, int(axis), offset, world=world)
        new_world = m3.compose(mirrored.translation, mirrored.rotation, mirrored.scale)
        parent = source.parent
        if parent in fresh:
            try:
                inverse = np.linalg.inv(copy_worlds[parent])
            except np.linalg.LinAlgError as error:
                from ....kernels.mesh.elements import OpError

                raise OpError(
                    f"{doc.by_uid(parent).name!r} has a zero scale, so nothing can be "
                    "placed relative to its copy."
                ) from error
            t, r, s = m3.decompose(inverse @ new_world)
            parent = fresh[parent]
        else:
            t, r, s = doc.local_from_world(uid, new_world)
        copy_worlds[uid] = new_world
        made.append(
            replace(
                mirrored,
                generator=None,
                params={},
                translation=t,
                rotation=r,
                scale=s,
                parent=parent,
            )
        )
    if not made:
        return False
    doc.add_objects(made)
    # Originals *and* copies: mirroring a mirror is a normal thing to want. Leaving the
    # copies unselected made "mirror across X, then mirror the pair across Z"
    # -- four table legs from one -- silently produce three, because the
    # second press saw only the original and re-mirrored it over a leg that
    # was already there (found by the furniture author, 2026-09-12).
    doc.select(originals + [obj.uid for obj in made])
    return bool(made)


def _world_boxes(doc: Any, uids: Iterable[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{uid: world_box}`` for every *uid* whose mesh is not empty.

    Drop to Ground's input: ``mason.ops``' box arithmetic takes the same
    ``Boxes`` mapping ``scene.world_bounds`` already builds for Mason's own
    selection -- see that module's docstring for why it takes world boxes
    rather than a ``GeometrySource``. An object whose mesh has no vertices reports no box
    (``ops.world_box`` returns ``None``) and is left out rather than degrading
    every other object's math with a phantom point at the origin.

    **Measured off the object's own world matrix (tranche 3: scene
    structure)**, not the TRS composed from its own fields alone -- a
    parented object's translation/rotation/scale describe its placement
    relative to its parent, not where its box actually sits, and the caller
    (and :func:`_apply_deltas`, which writes the deltas this produces back)
    means the box the user can see.
    """
    from ....kernels.mesh import ops as clay_ops_geom

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for uid in uids:
        obj = doc.by_uid(uid)
        box = clay_ops_geom.world_box(obj, world=doc.world_matrix(uid))
        if box is not None:
            out[uid] = box
    return out


def _apply_deltas(ctx: Any, doc: Any, deltas: dict[int, np.ndarray]) -> bool:
    """Add each world-space delta to its object's own translation. -> whether
    any object actually moved.

    A refusal on one object is toasted and the loop carries on, the same
    "toast; continue" shape :func:`run_mesh_op` and :func:`run_object_op` give
    their own loops.

    One call per object rather than one ``set_transform`` per axis: ``run``'s
    own ``_one_step`` folds however many of these land into the single undo
    step its docstring promises, so a multi-object Drop to Ground is one
    Ctrl+Z whatever it moved.

    **A root's own translation, still -- but a parented object's world
    matrix (tranche 3: scene structure).** The delta is world space
    (:func:`_world_boxes`'s own contract), and adding it straight to a
    root's translation is exactly adding it in world space, since a root's
    local frame *is* the world frame -- untouched here for the common case,
    bit-identical to what this always did. A parented object's own
    ``translation`` is local to its parent, not the world the delta was
    measured in, so its *world* matrix is shifted by the delta instead and
    the result converted back through ``doc.local_from_world``, the same
    door every other parented write in this file uses.

    **Every target world is read once, before any write, and applied
    ancestor-first.** The 2026-09-20 audit's clay-06: a parented uid used to
    read ``doc.world_matrix(uid)`` live, mid-loop -- once its own ancestor's
    ``set_transform`` had already landed earlier in the same call, that read
    was no longer the pre-move box ``_world_boxes`` measured the delta
    against, but one that already carried the ancestor's own shift for free,
    so adding the descendant's delta on top moved it twice. Freezing every
    selected uid's world matrix up front, before the loop writes anything,
    fixes what each object's own *target* world is; sorting the write order
    by ancestor depth (roots, then their children, and so on) then
    guarantees that by the time a parented object's own target is converted
    back with ``doc.local_from_world`` -- which walks its *live* parent
    chain -- every ancestor of it that is also in this same batch has
    already reached its own final position, so the conversion lands on the
    frozen target exactly once, however ``deltas`` happened to iterate.
    """
    from ....kernels.mesh.elements import OpError

    ran = False
    world_before = {uid: np.array(doc.world_matrix(uid), dtype="f8", copy=True) for uid in deltas}
    order = sorted(deltas, key=lambda uid: len(doc.ancestors(uid)))
    for uid in order:
        delta = deltas[uid]
        obj = doc.by_uid(uid)
        try:
            if obj.parent is None:
                translation = np.asarray(obj.translation, dtype="f8") + delta
                if doc.set_transform(uid, translation=translation):
                    ran = True
                continue
            world = world_before[uid].copy()
            world[:3, 3] = world[:3, 3] + np.asarray(delta, dtype="f8")
            t, r, s = doc.local_from_world(uid, world)
            if doc.set_transform(uid, translation=t, rotation=r, scale=s):
                ran = True
        except OpError as error:
            toast(ctx, str(error))
            continue
    return ran


def _drop_to_ground(ctx: Any, doc: Any, **_: Any) -> bool:
    """Rest each selected object's own world-box *bottom* on ``y=0``.

    Per object, not per group: there is no shared axis to agree on, so a box
    sitting three metres above the floor and one already resting on it both
    land correctly in the same call. The flat
    ground plane at ``y=0`` is this op's whole contract -- ``mason.ops.
    drop_to_ground`` also takes a ``Terrain`` for Mason's own version, which
    this row has no use for and does not pass.
    """
    from ..mason.engine import ops as mason_ops

    boxes = _world_boxes(doc, doc.selection)
    deltas = mason_ops.drop_to_ground(boxes, ground=0.0)
    return _apply_deltas(ctx, doc, deltas)


def _snap_to_grid(ctx: Any, doc: Any, step: float = 1.0, **_: Any) -> bool:
    """Snap every selected object's *world* position onto a grid of *step*
    metres, each axis independently -- ``ops.snap_translation``'s own rounding
    (half away from zero, so the grid stays symmetric about the origin).

    **World, like Drop to Ground** (the 2026-10-03 audit's clay-115): a
    parented object's own ``translation`` is local to its parent, so snapping
    it landed the child on a grid offset by the parent's position. A root's
    local frame *is* the world frame, so a root is still snapped directly
    (bit-identical to what this always did); a parented object is snapped in
    world space and converted back through
    ``doc.local_from_world``, writing only the translation so its own
    rotation and scale are not re-derived. Ancestors go first, so a child
    snaps against the parent's final position rather than moving off the grid
    again when its parent is snapped afterwards.
    """
    from ....kernels.mesh import elements as el
    from ....kernels.mesh import ops as clay_ops_geom

    # The 2026-10-07 audit's clay-37: ``snap_translation`` treats a step of 0 as
    # "leave it alone", and the parameter's range starts at 0, so a zero step
    # used to push nothing, say nothing and report that the op ran. A grid of
    # no size is not a grid; refuse it with the sentence that names the field.
    if not step > 0.0:
        raise el.OpError("The grid step must be above zero; a step of 0 has no grid to snap to.")

    def one(doc: Any, obj: Any) -> None:
        if obj.parent is None:
            snapped = clay_ops_geom.snap_translation(obj.translation, step)
        else:
            world = np.array(doc.world_matrix(obj.uid), dtype="f8", copy=True)
            world[:3, 3] = clay_ops_geom.snap_translation(world[:3, 3], step)
            snapped = doc.local_from_world(obj.uid, world)[0]
        doc.set_transform(obj.uid, translation=snapped)

    live = [uid for uid in doc.selection if any(o.uid == uid for o in doc.objects)]
    order = sorted(live, key=lambda uid: len(doc.ancestors(uid)))
    return run_object_op(ctx, doc, one, uids=order)


# --- tranche 3: scene structure -- parenting, groups, separate, origin -------
#
# Clay tranche 3. The document-layer doors (``ClayDoc.group``/
# ``set_parent``/``remove_object``/``set_origin``/``separate``, and the pure
# ``kernels.mesh.separate`` splitters) already carry the one-step contract --
# see ``kernels/mesh/document.py``'s own module docstring. What belongs here
# is only the registry wiring: which selection each row reads, and turning its
# result into the door call.


def _is_group(doc: Any, obj: Any) -> bool:
    """Whether *obj* is a "group" in Ungroup's sense: an empty -- no mesh of
    its own, ``ClayDoc.group``'s own shape -- with at least one child to
    release. An empty with nothing under it, or an object that carries real
    geometry of its own, is not what Ungroup means to act on.
    """
    return len(obj.mesh.positions) == 0 and bool(doc.children_of(obj.uid))


def has_group_selected(doc: Any) -> bool:
    """Ungroup's own gate: at least one selected object is a group."""
    return any(_is_group(doc, obj) for obj in doc.objects if obj.uid in doc.selection)


def _has_group_selected_reason(doc: Any) -> str:
    if not doc.selection:
        return "Select a group first."
    return "" if has_group_selected(doc) else "Select an empty with children to ungroup."


def _group(ctx: Any, doc: Any, **_: Any) -> None:
    """Group the selection under one new empty, and select the empty.

    ``ClayDoc.group`` does the placement (the selection's combined world
    bounds centre) and the parenting, as one step -- see its own docstring.
    The empty is left selected rather than its new children, the same
    "leave the thing that now has a role to play selected" choice
    ``_join`` makes for its own merge target: grouping is a
    gesture aimed at moving the group as one from here on, not at any one
    member of it.
    """
    del ctx
    uids = [obj.uid for obj in doc.objects if obj.uid in doc.selection]
    empty = doc.group(uids)
    doc.select([empty.uid])


def _ungroup(ctx: Any, doc: Any, **_: Any) -> bool:
    """Release every selected group's children back to its own parent, and
    delete the empty.

    ``ClayDoc.remove_object`` already *is* "re-parent this object's children
    onto its own parent, keeping world placement, then remove it" as one
    step (its own docstring) -- exactly what Ungroup means, so this is the
    same door Delete uses, scoped to groups by :func:`_is_group`. A selected
    object that is not a group is left alone rather than refused: the row is
    enabled the moment *one* of the selection qualifies, the same
    "whichever of the selection this actually applies to" shape
    ``_shade``'s own face-mode branch already reads.
    """

    def one(doc: Any, obj: Any) -> None:
        if _is_group(doc, obj):
            doc.remove_object(obj.uid)

    return run_object_op(ctx, doc, one)


def _parent_to_last(ctx: Any, doc: Any, **_: Any) -> bool:
    """Parent every other selected object onto the topmost one in the
    outliner.

    The topmost selected object in document order is the new parent --
    ``doc.selection`` is a set with no order of its own, the same reason
    ``_join`` reads its own merge target out of ``doc.objects`` rather than
    out of the selection directly (see its docstring). "Last
    Selected" is Blender's own name for this gesture, which reads the
    *active* object; Clay has no such concept, so this reuses the identical
    document-order tiebreak a user can see and predict, rather than
    inventing a second one.

    ``ClayDoc.set_parent`` is the door, one call per child -- it refuses a
    cycle (parenting an object onto its own descendant) with its own
    sentence, toasted and skipped by ``run_object_op`` exactly like every
    other per-object refusal in this file, so one bad choice in a bigger
    selection does not abandon the rest of it.
    """
    uids = [obj.uid for obj in doc.objects if obj.uid in doc.selection]
    if len(uids) < 2:
        return False
    target, *rest = uids

    def one(doc: Any, obj: Any) -> None:
        doc.set_parent(obj.uid, target, keep_world=True)

    # The 2026-10-07 audit's clay-38: the result of ``run_object_op`` is what
    # tells ``run`` that every parenting was refused (a cycle, say). Dropping it
    # made ``run`` report True -- and an agent read ran=true with no step pushed.
    return run_object_op(ctx, doc, one, uids=rest)


def _clear_parent(ctx: Any, doc: Any, **_: Any) -> bool:
    """Every selected object becomes a root, keeping its world placement --
    ``ClayDoc.set_parent(uid, None, keep_world=True)`` per object, the same
    door :func:`_ungroup` and :func:`_parent_to_last` both use. An
    already-rootless object is a no-op (``set_parent`` pushes nothing for
    it), so a mixed selection of roots and children clears only the ones
    that had somewhere to fall from.
    """

    def one(doc: Any, obj: Any) -> None:
        doc.set_parent(obj.uid, None, keep_world=True)

    return run_object_op(ctx, doc, one)


def _separate_loose(ctx: Any, doc: Any, **_: Any) -> bool:
    """Split every selected object into one new object per loose part.

    ``kernels.mesh.separate.by_loose_parts`` computes the pieces;
    ``ClayDoc.separate`` turns them into objects, copies the source's
    transform and parent onto each one and removes the source, as one step
    (its own docstring). A single-piece object refuses with a toast
    (``by_loose_parts``'s own "nothing to separate") and is left untouched,
    the rest of the selection still separating -- and, if that was the whole
    selection, this reports it ran nothing.
    """
    from ....kernels.mesh import separate as sep

    def one(doc: Any, obj: Any) -> None:
        doc.separate(obj.uid, sep.by_loose_parts(obj.mesh))

    return run_object_op(ctx, doc, one)


def _separate_selection(ctx: Any, doc: Any, **_: Any) -> bool:
    """Split every selected object's own face selection out as a new
    object, leaving the rest of it behind.

    Face mode only, and reading ``doc.element_sel`` the way every element op
    in this registry does (the invariant the module docstring states: in an
    element mode, ``doc.selection`` already names exactly the objects with
    something picked, so ``run_object_op``'s default selection is the right
    one with no extra filtering). ``kernels.mesh.separate.by_selection``
    always answers with exactly two pieces and refuses -- a toast, not an
    abort of the rest of the selection -- a pick touching none of an
    object's faces or all of them.
    """
    from ....kernels.mesh import separate as sep

    def one(doc: Any, obj: Any) -> None:
        pieces = doc.separate(obj.uid, sep.by_selection(obj.mesh, doc.element_sel_of(obj.uid)))
        # The 2026-09-26 audit's clay-document-03: ``ClayDoc.separate`` adds
        # every new piece to ``selection`` unconditionally, but a piece just
        # split off has nothing selected inside it -- in face mode (the only
        # mode this op runs in) that broke the module docstring's own
        # invariant, "selection holds exactly the uids with a non-empty
        # element_sel". ``set_element_sel`` is the one place that invariant
        # is already kept by hand; handing it ``None`` for each new piece
        # drops it from ``selection`` again the same way an ordinary empty
        # pick would, with no direct reach into the document's own sets.
        for piece in pieces:
            doc.set_element_sel(piece.uid, None)

    return run_object_op(ctx, doc, one)


def _origin_to_bounds(ctx: Any, doc: Any, **_: Any) -> bool:
    """Move each selected object's origin to its own *world* box's centre.

    Measured off the object's world matrix (``doc.world_matrix``, tranche 3),
    the same one :func:`_world_boxes` reads for Drop to Ground -- an object
    under a parent has a placement its local TRS does not describe alone.
    ``ClayDoc.set_origin`` keeps the geometry and every child exactly where
    they were (its own docstring); only the pivot moves.
    """
    from ....kernels.mesh import ops as clay_ops_geom

    def one(doc: Any, obj: Any) -> None:
        box = clay_ops_geom.world_box(obj, world=doc.world_matrix(obj.uid))
        if box is None:
            return
        lo, hi = box
        doc.set_origin(obj.uid, (lo + hi) * 0.5)

    return run_object_op(ctx, doc, one)


def _origin_to_base(ctx: Any, doc: Any, **_: Any) -> bool:
    """Move each selected object's origin to its own world box's bottom
    centre -- :func:`_origin_to_bounds`'s identical measurement, with the
    Y (up) component read off the box's *low* corner instead of its middle,
    the same axis ``_drop_to_ground`` rests on ``y=0``.
    """
    from ....kernels.mesh import ops as clay_ops_geom

    def one(doc: Any, obj: Any) -> None:
        box = clay_ops_geom.world_box(obj, world=doc.world_matrix(obj.uid))
        if box is None:
            return
        lo, hi = box
        doc.set_origin(obj.uid, [(lo[0] + hi[0]) * 0.5, lo[1], (lo[2] + hi[2]) * 0.5])

    return run_object_op(ctx, doc, one)


def _origin_to_selection(ctx: Any, doc: Any, **_: Any) -> bool:
    """Move each selected object's origin to its own element selection's
    centroid.

    Read off the mesh in local space and carried into a world point through
    the object's own world matrix before ``ClayDoc.set_origin`` writes it
    back, since that door takes a world point (its own signature) and a
    parented object's local positions are not one.
    """
    from ....kernels.mesh import elements as el

    def one(doc: Any, obj: Any) -> None:
        verts = el.affected_verts(obj.mesh, doc.element_sel_of(obj.uid))
        if len(verts) == 0:
            return
        local = np.asarray(obj.mesh.positions, dtype="f8")[verts].mean(axis=0)
        homogeneous = np.array([local[0], local[1], local[2], 1.0], dtype="f8")
        point = (doc.world_matrix(obj.uid) @ homogeneous)[:3]
        doc.set_origin(obj.uid, point)

    return run_object_op(ctx, doc, one)


def _origin_to_world(ctx: Any, doc: Any, **_: Any) -> bool:
    """Move each selected object's origin to the world origin -- the one
    origin-* row that needs no measurement at all."""

    def one(doc: Any, obj: Any) -> None:
        doc.set_origin(obj.uid, [0.0, 0.0, 0.0])

    return run_object_op(ctx, doc, one)


def _repeat_last_reason(doc: Any) -> str:
    """Why Repeat Last cannot run, or ``""``. ``enabled`` is derived from it.

    One predicate for both so the sentence cannot drift from the gate (the
    clay-07 rule). Over MCP this is the refusal an agent reads, and because the
    record lives on the document an agent can only ever repeat its own tab's
    last op.
    """
    recent = doc.recent_op
    if recent is None:
        return "Nothing to repeat."
    op = get(recent.op_name)
    if doc.element_mode not in op.modes:
        return f"{op.label.rstrip('.')} works in {' or '.join(op.modes)} mode."
    return reason_for(op, doc) if not op.enabled(doc) else ""


def _repeat_last(ctx: Any, doc: Any, **_: Any) -> bool:
    recent = doc.recent_op
    return run(ctx, doc, get(recent.op_name), **recent.params)


def _delete(ctx: Any, doc: Any, **_: Any) -> None:
    from ....kernels.mesh import selection

    for message in selection.delete_selected(doc):
        toast(ctx, message)


def _unwrap(ctx: Any, doc: Any, **_: Any) -> bool:
    """Give every selected object a fresh box projection.

    Whole objects rather than the selected faces, and that is the decision:
    unwrapping half a mesh leaves the other half's coordinates from whenever
    they were last computed, so the two islands are at different texel
    densities and a checker map says so immediately. Per-face unwrapping is a
    real feature and it needs a face-level projection tool first.

    It does **not** freeze the generator. UVs are not geometry -- the positions,
    the topology and the parameters are all untouched -- so a box that has been
    unwrapped is still describable as "box, size 1", and re-editing the size
    correctly rebuilds it with the generator's own canonical coordinates.
    """
    from ....kernels.mesh import uv as uv_mod

    def one(doc: Any, obj: Any) -> None:
        doc.set_mesh(obj.uid, uv_mod.box_unwrap(obj.mesh), keep_generator=True)

    return run_object_op(ctx, doc, one)


def _assign_material(ctx: Any, doc: Any, index: int = 0, **_: Any) -> bool:
    """Paint the selected faces, on every object that has some, with palette slot
    *index*.

    The faces rather than the object: ``mesh.material`` is one slot per face and
    ``Obj.material`` only the slot new faces are stamped with, so this leaves the
    object's default slot alone (the Material tab's swatch does the object-level
    version in object mode). The slot is checked against the palette *here*, at
    run time, because ``Param`` can only clamp to a fixed range and the palette's
    length is a property of the document -- an out-of-range number is a refusal
    with a toast, never a silent clamp onto some other slot. ``run`` folds every
    object's ``paint_faces`` into one undo step.
    """
    from ....kernels.mesh import elements as el

    count = len(doc.materials)
    if not 0 <= index < count:
        raise el.OpError(
            f"there is no palette entry {index}; the palette has {count}"
            f" (0 to {count - 1})."
        )
    painted = 0
    for uid in list(doc.element_sel):
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            continue
        faces = el.convert(obj.mesh, doc.element_sel_of(uid), "face").faces
        if len(faces) and doc.paint_faces(uid, faces, index):
            painted += len(faces)
    name = doc.materials[index].name or f"slot {index}"
    if not painted:
        # Pushed nothing: say so rather than let a click on the slot the faces
        # already wear look dead. ``False`` keeps it out of Repeat Last.
        ctx.toast(f"Those faces already use {name}.", "info")
        return False
    ctx.toast(f"Painted {painted} face(s) with {name}.", "info")
    return True


def _recalc_normals(ctx: Any, doc: Any, **_: Any) -> bool:
    """Make every selected object's winding consistent and outward.

    **Object mode only.** A flipped face is a property of a *shell* --
    :func:`~.ops_clean.recalc_outside`'s own BFS walks whole connected
    components, majority-votes each one, then signs it by volume -- and a face
    selection is not a shell: recalculating "the selected faces" would have to
    either ignore the faces around them (silently wrong the moment the
    selection is not the whole shell) or walk past the selection into
    unselected geometry anyway (silently doing more than the button claims).
    Object mode asks the one question the algorithm actually answers.

    **Freezes the generator, deliberately, through the same door as every
    other geometry-changing op.** ``doc.set_mesh`` does the freezing (see the
    module docstring); this passes no ``keep_generator=True`` because winding
    is not a fact a generator's parameters record, so an object whose winding
    this corrected is no longer exactly what its generator would rebuild.
    In practice this is unreachable on an untouched primitive: every
    generator in this package already emits outward, consistently wound faces
    (:func:`~.ops_clean.recalc_outside` is idempotent on one), so the freeze
    only ever fires on an imported or hand-repaired mesh that actually needed
    the fix -- ``recalc_outside`` returns the identical object otherwise, and
    identity is what ``set_mesh`` reads to decide whether anything happened at
    all.
    """
    from ....kernels.mesh import ops_clean

    def one(doc: Any, obj: Any) -> None:
        mesh = ops_clean.recalc_outside(obj.mesh)
        if mesh is not obj.mesh:
            doc.set_mesh(obj.uid, mesh)

    return run_object_op(ctx, doc, one)


# --- shading, selection, frame ------------------------------------------------


def _shade(smooth: bool) -> Callable[..., bool | None]:
    """Set the shading flag on the selected faces, or on whole objects.

    Both modes, because both readings are real: in face mode a user means
    "these faces", and in object mode they mean "this whole shape". The flag is
    per face either way -- there is no object-level shading setting that would
    have to be kept in agreement with it.
    """

    def run(ctx: Any, doc: Any, **_: Any) -> bool | None:
        if doc.element_mode == "object":
            # The 2026-10-07 audit's clay-38: the result of ``run_object_op``
            # is what tells ``run`` that every object refused; dropping it
            # made ``run`` report True for a press that changed nothing.
            return run_object_op(
                ctx, doc, lambda doc, obj: doc.set_shading(obj.uid, None, smooth)
            )
        from ....kernels.mesh import elements as el

        for uid in list(doc.element_sel):
            faces = el.convert(doc.by_uid(uid).mesh, doc.element_sel_of(uid), "face")
            doc.set_shading(uid, faces.faces, smooth)

    return run


def _frame(ctx: Any, doc: Any, **_: Any) -> None:
    view = getattr(ctx, "clay_view", None)
    if view is not None:
        view.frame_selection(doc)


def _select_all(ctx: Any, doc: Any, **_: Any) -> None:
    from ....kernels.mesh import selection

    del ctx
    selection.select_all(doc)


def _select_none(ctx: Any, doc: Any, **_: Any) -> None:
    del ctx
    if doc.element_mode == "object":
        # The object-mode sense of "none", the Select menu's reason to be here:
        # ``clear_element_sel`` leaves ``doc.selection`` alone in object mode,
        # so the same op would have run and changed nothing.
        doc.select([])
        return
    doc.clear_element_sel()


def _invert(ctx: Any, doc: Any, **_: Any) -> None:
    from ....kernels.mesh import selection

    del ctx
    selection.invert(doc)


def _selection_op(verb: Any) -> Any:
    """Wrap a ``clay.select`` verb as an op that writes the element selection.

    Read what is selected on each object, ask the engine, write the answer
    back only when it changed.

    Per object, and **only the objects that already have a selection**: growing
    a selection on the object you are working on must not quietly select
    something on the one behind it.
    """

    def run(ctx: Any, doc: Any, **params: Any) -> bool:

        del ctx
        mode = doc.element_mode
        ran = False
        for uid in list(doc.element_sel):
            try:
                obj = doc.by_uid(uid)
            except KeyError:
                continue
            sel = doc.element_sel_of(uid)
            wanted = verb(obj.mesh, sel, mode, **params)
            if wanted is None or wanted.same_as(sel):
                continue
            doc.set_element_sel(uid, wanted)
            ran = True
        # ``False`` rather than a refusal: the selection is what it already was,
        # and ``run`` turns that into "nothing happened" without a toast. A verb
        # that found nothing new is not an error, it is an answer.
        return ran

    return run


def _verb_linked(mesh: Any, sel: Any, mode: str) -> Any:
    from ....kernels.mesh import select as bsel

    verts = bsel.verts_of(mesh, sel, mode)
    if not len(verts):
        return None
    return bsel.sel_from_verts(mesh, bsel.linked(mesh, verts), mode)


# --- UV ---------------------------------------------------------------------


def _pack_uv(ctx: Any, doc: Any, margin: float = 0.005, rotate: float = 0.0, **_: Any) -> bool:
    """"Pack UV Islands" -- ``uvtools.pack_islands`` over every selected
    object's own uv, keeping the generator. Refuses (the kernel's own
    sentence) an object with no uv at all -- unwrap it first.
    """
    from ....kernels.mesh import uvtools

    def one(doc: Any, obj: Any) -> None:
        packed = uvtools.pack_islands(obj.mesh, margin=float(margin), rotate=bool(rotate))
        doc.set_mesh(obj.uid, packed, keep_generator=True)

    return run_object_op(ctx, doc, one)


def _register_defaults() -> None:
    """Build the registry once, at import.

    A function rather than module-level statements so the tests can assert the
    registry is *complete* by calling it on a fresh list, and so a duplicate
    import cannot register everything twice.
    """
    if OPS:
        return

    register(
        Op(
            name="select-all",
            label="Select All",
            modes=ALL_MODES,
            run=_select_all,
            key="Ctrl+A",
        )
    )
    register(
        Op(
            name="select-none",
            label="Select None",
            modes=ALL_MODES,
            run=_select_none,
            enabled=has_any_selection,
            reason=_has_any_selection_reason,
            # No `key` here: the 2026-09-20 audit's clay-22 -- `by_key` is only
            # ever called with `pygame.key.name(...).upper()`, which yields
            # "ESCAPE", so a registered "Esc" never resolved and the popup's
            # "Select None  Esc" claimed a binding this op never had. Escape
            # is genuinely owned by `mode._escape`, hand-dispatched outside
            # the registry (and its undo fold); an alias here would just
            # relabel the same lie rather than fix it.
        )
    )
    register(
        Op(
            name="select-invert",
            label="Invert Selection",
            modes=ALL_MODES,
            run=_invert,
            key="Ctrl+Shift+I",
        )
    )
    register(
        Op(
            name="select-linked",
            label="Select Linked",
            modes=ELEMENT_MODES,
            run=_selection_op(_verb_linked),
            enabled=has_elements,
            reason=_has_elements_reason,
            key="L",
            hint="Everything joined to what is selected. Two shapes welded into "
            "one mesh are separable again by it.",
        )
    )

    register(
        Op(
            name="duplicate",
            label="Duplicate",
            modes=("object",),
            run=_duplicate,
            enabled=has_objects,
            reason=_has_objects_reason,
            key="Ctrl+J",
        )
    )
    for smooth, label in ((True, "Shade Smooth"), (False, "Shade Flat")):
        register(
            Op(
                name=f"shade-{'smooth' if smooth else 'flat'}",
                label=label,
                modes=("object", "face"),
                run=_shade(smooth),
                # clay-06 (2026-09-08 audit): ``has_objects`` graded object
                # mode's own question in face mode too, where the op body
                # reads the *element* selection -- see ``_shade_enabled``.
                enabled=_shade_enabled,
                reason=_shade_reason,
            )
        )
    register(
        Op(
            name="unwrap",
            label="Box Unwrap",
            modes=("object",),
            run=_unwrap,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Gives every selected object a fresh box projection: each face "
            "is flattened along the axis its normal points closest to. Whole "
            "objects, not just the selected faces. Pack Islands needs this first.",
        )
    )
    register(
        Op(
            name="pack-uv",
            label="Pack Islands...",
            modes=("object",),
            run=_pack_uv,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Repacks every selected object's uv islands into the unit "
            "square, preserving their relative scale. Needs a uv already -- "
            "unwrap first.",
            params=(
                Param("margin", "margin", 0.005, 0.001, low=0.0, high=0.5),
                Param(
                    "rotate", "rotate islands", 0.0, 1.0, low=0.0, high=1.0, boolean=True,
                ),
            ),
        )
    )
    register(
        Op(
            name="recalc-normals",
            label="Recalculate Normals",
            modes=("object",),
            run=_recalc_normals,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Makes every shell's winding agree with itself, then orients "
            "each shell outward. Whole objects only -- winding is a fact "
            "about a shell, not about a face selection.",
        )
    )
    register(
        Op(
            name="join",
            label="Merge Objects...",
            modes=("object",),
            run=_join,
            # Two, not one: merging a single object is the identity, and an
            # enabled button that does nothing is worse than a greyed one.
            # Counted over the *visible* selection for that same reason, since
            # that is what ``_join`` will actually merge -- a row enabled by an
            # object the merge then skips is the greyed one's problem again.
            enabled=has_two_visible,
            reason=_has_two_visible_reason,
            key="Ctrl+M",
            hint=(
                "Welds the shapes into one object and keeps every surface, "
                "including the ones buried inside an overlap."
            ),
            params=(
                Param(
                    "weld",
                    "weld distance (m)",
                    1e-4,
                    1e-4,
                    low=0.0,
                    warn="0 keeps the shapes as separate shells inside one object.",
                ),
            ),
        )
    )
    for axis, label in enumerate(("X", "Y", "Z")):
        register(
            Op(
                name=f"mirror-{label.lower()}",
                label=f"Mirror {label}",
                modes=("object",),
                run=(lambda a: lambda ctx, doc, **kw: mirror(ctx, doc, a, **kw))(axis),
                enabled=has_objects,
                reason=_has_objects_reason,
            )
        )
    register(
        Op(
            name="mirror-copy",
            label="Mirror Copy...",
            modes=("object",),
            run=_mirror_copy,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Duplicates the selection and reflects the copies across a "
            "world plane, rather than replacing the object about its own "
            "centre the way Mirror X/Y/Z does -- this is mirroring a limb "
            "across a body's centre-line.",
            params=(
                Param("axis", "axis", 0.0, 1.0, low=0.0, high=2.0, choices=("X", "Y", "Z")),
                Param("offset", "plane at (m)", 0.0, 0.1, low=-1e6),
            ),
        )
    )
    register(
        Op(
            name="drop-to-ground",
            label="Drop to Ground",
            modes=("object",),
            run=_drop_to_ground,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Rests each selected object's own world-box bottom on y=0, "
            "not its pivot -- a barrel authored with its pivot at the middle "
            "no longer floats half its height in the air.",
        )
    )
    register(
        Op(
            name="snap-to-grid",
            label="Snap to Grid...",
            modes=("object",),
            run=_snap_to_grid,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Snaps every selected object's translation onto a grid of "
            "the given step, one axis at a time.",
            params=(Param("step", "grid step (m)", 1.0, 0.1, low=0.0),),
        )
    )

    # Tranche 3: scene structure -- parenting and groups, separate, set
    # origin. What is here is only the registry rows.
    register(
        Op(
            name="group",
            label="Group Selected",
            modes=("object",),
            run=_group,
            enabled=has_two_or_more_selected,
            reason=_has_two_or_more_selected_reason,
            hint="Makes a new, mesh-less object at the selection's combined "
            "centre and parents the selection onto it -- Clay's one grouping "
            "concept (see the manual's Outliner chapter).",
        )
    )
    register(
        Op(
            name="ungroup",
            label="Ungroup",
            modes=("object",),
            run=_ungroup,
            enabled=has_group_selected,
            reason=_has_group_selected_reason,
            hint="Releases a group's children back to its own parent, "
            "keeping their world placement, and removes the empty.",
        )
    )
    register(
        Op(
            name="parent-to-last",
            # "Parent to Last Selected" is the full name the spec and the
            # manual use; shortened here because it is the longest label in
            # the object menu and does not fit even a one-column grid at the
            # narrowest tested sidebar (190 dp). (The 2026-10-07 audit's
            # clay-42: this cited ``test_no_action_button_is_narrower_than_its_own_label``,
            # which no longer exists.)
            label="Parent to Last",
            modes=("object",),
            run=_parent_to_last,
            enabled=has_two_or_more_selected,
            reason=_has_two_or_more_selected_reason,
            hint="Parents every other selected object onto the topmost one "
            "in the outliner, keeping world placement -- Clay has no "
            "'active object', so document order stands in for it, the same "
            "rule Merge Objects uses for its own target.",
        )
    )
    register(
        Op(
            name="clear-parent",
            label="Clear Parent",
            modes=("object",),
            run=_clear_parent,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Every selected object becomes a root, keeping its world "
            "placement.",
        )
    )
    register(
        Op(
            name="separate-loose",
            # Same reasoning as "Parent to Last" above: "Separate by Loose
            # Parts" is one pixel too wide for the narrowest tested sidebar.
            label="Separate Loose Parts",
            modes=("object",),
            run=_separate_loose,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Splits each selected object into one new object per "
            "connected shell, as one step: same transform and same parent, "
            "copied onto every piece.",
        )
    )
    register(
        Op(
            name="separate-selection",
            label="Separate Selection",
            modes=("face",),
            run=_separate_selection,
            enabled=in_mode("face"),
            reason=_in_mode_reason("face"),
            key="P",
            hint="Splits the selected faces out into a new object, leaving "
            "the rest behind.",
        )
    )
    register(
        Op(
            name="origin-to-bounds",
            label="Origin to Bounds",
            modes=("object",),
            run=_origin_to_bounds,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Moves each selected object's origin to its own world "
            "box's centre. Geometry and children stay exactly where they "
            "are -- only the pivot moves.",
        )
    )
    register(
        Op(
            name="origin-to-base",
            label="Origin to Base",
            modes=("object",),
            run=_origin_to_base,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Moves each selected object's origin to its own world "
            "box's bottom centre.",
        )
    )
    register(
        Op(
            name="origin-to-selection",
            label="Origin to Selection",
            modes=ELEMENT_MODES,
            run=_origin_to_selection,
            enabled=has_elements,
            reason=_has_elements_reason,
            hint="Moves each selected object's origin to its own element "
            "selection's centroid.",
        )
    )
    register(
        Op(
            name="origin-to-world",
            label="Origin to World",
            modes=("object",),
            run=_origin_to_world,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Moves each selected object's origin to the world origin.",
        )
    )

    register(
        Op(
            name="extrude",
            label="Extrude",
            modes=ELEMENT_MODES,
            run=_extrude,
            enabled=has_elements,
            reason=_has_elements_reason,
            key="E",
        )
    )
    register(
        Op(
            name="inset",
            label="Inset Faces...",
            modes=("face",),
            run=_element("ops_topo.inset_faces"),
            key="I",
            drag=DragSpec("thickness", "distance", "ops_topo.inset_faces"),
            enabled=in_mode("face"),
            reason=_in_mode_reason("face"),
            # The 2026-09-11 audit's clay-08: ``ops_topo.inset_faces`` has
            # taken a ``region`` argument -- a real, tested second mode, not a
            # variant of the default -- since before this registry existed.
            # ``_element`` forwards every param by name to the mesh op it
            # wraps, so declaring the toggle here is the whole fix: a
            # keyword-only ``bool`` parameter reads a clamped 0/1 ``int`` as
            # truthy/falsy with no cast required.
            hint="Per-face (default) insets every selected face on its own, "
            "so two touching faces get a doubled edge between them where "
            "they meet. 'Region' insets the outline of the whole selected "
            "block instead, with one shared ring and no seam down the "
            "middle -- a documented approximation on a block that is not "
            "flat, rather than a true offset.",
            params=(
                Param("thickness", "thickness (m)", 0.1, 0.01),
                Param("depth", "depth (m)", 0.0, 0.01, low=-1e6),
                Param(
                    "region",
                    "treat selection as one region",
                    0.0,
                    1.0,
                    low=0.0,
                    high=1.0,
                    boolean=True,
                ),
            ),
        )
    )
    # ``ops_dissolve.dissolve_faces`` merges a connected block of faces into
    # one n-gon; "Merge Faces" is the word a user searches for.
    #
    # Registering it (rather than adding a button) is what gets it into all
    # three surfaces at once: the Clay menu, the tools pane and the bare-letter
    # keys all read this table.
    register(
        Op(
            name="merge_faces",
            label="Merge Faces",
            modes=("face",),
            run=_element("ops_dissolve.dissolve_faces"),
            enabled=in_mode("face"),
            reason=_in_mode_reason("face"),
        )
    )
    register(
        Op(
            name="weld",
            label="Weld...",
            modes=("vertex",),
            run=_element("ops_topo.weld"),
            enabled=in_mode("vertex"),
            reason=_in_mode_reason("vertex"),
            params=(Param("eps", "distance (m)", 1e-4, 1e-4, low=1e-9),),
        )
    )
    register(
        Op(
            name="flip",
            label="Flip Normals",
            modes=("face",),
            run=_element("ops_topo.flip_normals"),
            enabled=in_mode("face"),
            reason=_in_mode_reason("face"),
        )
    )
    register(
        Op(
            name="assign-material",
            label="Assign Material...",
            modes=("face",),
            run=_assign_material,
            enabled=in_mode("face"),
            reason=_in_mode_reason("face"),
            params=(
                Param("index", "palette slot", 0.0, 1.0, low=0.0, integer=True),
            ),
            hint="Paints the selected faces with one palette entry, leaving the "
            "object's default slot alone. The slot is its number in the "
            "palette strip under the viewport, counting from 0.",
        )
    )
    register(
        Op(
            name="subdivide",
            label="Subdivide",
            modes=("face",),
            run=_element("ops_subdiv.subdivide"),
            enabled=in_mode("face"),
            reason=_in_mode_reason("face"),
        )
    )
    register(
        Op(
            name="triangulate",
            label="Triangulate Faces",
            modes=("face",),
            run=_element("ops_topo.triangulate_faces"),
            enabled=in_mode("face"),
            reason=_in_mode_reason("face"),
            key="T",
            # The 2026-09-19 audit (clay-36): the kernel's own no-selection
            # fallback ("every face") is unreachable through this row --
            # ``enabled=in_mode("face")`` requires a non-empty element
            # selection, so the button is greyed out exactly when that branch
            # would fire. The hint used to promise it anyway.
            hint="Replaces the selected faces with their own triangles.",
        )
    )
    register(
        Op(
            name="repeat-last",
            label="Repeat Last",
            modes=ELEMENT_MODES,
            run=_repeat_last,
            enabled=lambda doc: not _repeat_last_reason(doc),
            reason=_repeat_last_reason,
            key="Shift+R",
            hint="Runs the last parameterised element operation again, at the "
            "same values, on what is selected now.",
        )
    )
    register(
        Op(
            name="frame",
            label="Frame Selection",
            modes=ALL_MODES,
            run=_frame,
        )
    )
    register(
        Op(
            name="delete",
            label="Delete",
            modes=ALL_MODES,
            run=_delete,
            enabled=lambda doc: bool(doc.selection),
            reason=_selection_reason,
            key="Del",
        )
    )


_register_defaults()
