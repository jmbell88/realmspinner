"""One registry of everything Clay can *do*, and no imgui anywhere in it.

Three surfaces invoke Clay's operations -- the right-mouse context menu, the
buttons in the tools pane, and ``clay_mode.handle_key`` -- and before this
module they each had their own list. Three lists means three answers to "is
Extrude available in vertex mode", and the interesting one is always the one
nobody updated: a key that fires an op the menu greys out, or a menu row for an
op the key path never learned about.

So there is one list. :data:`OPS` is the whole of what is invocable, each entry
carrying the modes it applies to, whether it is enabled right now, the key that
fires it and how it groups in the menu. The menu renders it, the pane renders a
subset of it, and the key handler looks up by key -- and none of them decides
anything.

**Nothing here imports imgui**, which is what keeps the registry testable: an
``Op``'s ``enabled`` predicate is a function of a document, so "Fill Hole is
greyed out with a face selected" is a plain assertion rather than a screenshot.
The pane layer (``studio/modes/clay/ui/menu.py``) is the only thing that knows a popup
exists.

**An op that changes geometry freezes the object's generator.** A box whose
faces have been extruded is no longer describable by "box, size 1" -- the
properties panel would offer a size field that silently discards the edit the
moment it was touched. The freeze is ``Document.set_mesh``'s, not any op's:
saying "``run`` clears it in one place, for every op" was not true of ``run``
at all -- only ``run_mesh_op`` and Smooth did it, while Delete, Bake Transform
and Mirror went straight to ``set_mesh`` and kept a generator that would
rebuild over them. Putting it where the geometry actually changes is what
makes the sentence true for ops that do not exist yet.

**A refusal is a toast, not an exception.** Every op raises
:class:`~.clay.elements.OpError` with a sentence naming what it refused and what
to do instead; :func:`run` catches exactly that, shows it and records no edit.
Anything else propagates, because an ``IndexError`` out of a topology op is a
bug and swallowing it would leave a half-built mesh on screen with no clue why.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ....kernels.mesh import shading as _shading

__all__ = [
    "OPS",
    "Op",
    "Param",
    "bake_apply",
    "by_key",
    "decimate_apply",
    "defaults_for",
    "menu",
    "reason_for",
    "register",
    "retopo_apply",
    "run",
    "run_mesh_op",
    "run_object_op",
    "unwrap_apply",
]

# The element modes an op can appear in. "object" is the fourth and is not an
# element mode; ops that name it are the object-level ones (duplicate, bake,
# mirror) that were previously hardcoded in the tools pane.
ALL_MODES: tuple[str, ...] = ("object", "vertex", "edge", "face")
ELEMENT_MODES: tuple[str, ...] = ("vertex", "edge", "face")


@dataclass(frozen=True)
class Param:
    """One number an op takes, and everything a widget needs to offer it.

    Described rather than drawn so the pane builds every popup from one loop.
    ``warn`` is shown under the field -- Catmull-Clark at two levels multiplies
    a mesh by sixteen, and a user who finds that out by waiting is a user who
    lost their document to the undo budget.

    **``boolean`` and ``choices`` are widget kinds, not storage kinds.** The
    four ops that reached for a number on 2026-09-10 because this dataclass had
    no other shape -- ``place-between``'s ``fit`` ("fit to gap (0=off,
    1=on)") and ``array-radial``/``mirror-copy``'s ``axis`` ("axis (0=X,
    1=Y, 2=Z)") -- are the tell: the *label* was doing the widget's job because
    the field couldn't. ``default`` stays a ``float`` and :func:`run` still
    clamps to ``low``/``high`` regardless of which of the three this is, so a
    checkbox writes 0.0/1.0 and a combo writes its index -- exactly what the
    bare int field they replaced already wrote, and why converting an op to
    either costs nothing beyond this dataclass and the pane's drawing loop.

    They are mutually exclusive -- one field is a toggle or a named set of
    options, never both -- and each pins its own range so nothing downstream
    has to ask "which kind is this and what do its bounds mean": a boolean is
    always 0..1, and a choice's ``high`` is always ``len(choices) - 1``, the
    same bound :func:`run` was already going to clamp it to. An earlier
    comment on ``place-between`` called ``axis`` a precedent for the checkbox
    it wanted; that was wrong about ``axis`` -- three values are a choice, not
    a toggle -- which is why this is two kinds and not one.
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
    separator_before: bool = False
    params: tuple[Param, ...] = field(default=())
    """The numbers the pane pops a dialog for, or empty for a bare action."""
    hint: str = ""
    """One sentence under the dialog's title, about the op rather than a field.

    ``Param.warn`` is about a *number* -- what two levels of Catmull-Clark will
    do to a mesh -- and belongs under the field it qualifies. This is about the
    op: what it is for, and when the op next to it is the one you meant. Only
    the parameterised ops can show it, because only they open a dialog, which
    is the right restriction: a bare action gives no moment to read anything.
    """
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


def run(ctx: Any, doc: Any, op: Op, **params: Any) -> bool:
    """Invoke an op, turning a refusal into a toast. -> whether it ran.

    Missing parameters fall back to their declared defaults, so a caller that
    has no remembered values -- the key path, a test -- gets the same result the
    popup would have produced with the fields untouched.

    **Declared parameters are clamped here.** The popup clamps its live fields
    too (``studio/modes/clay/ui/menu.py``), but that is a UX affordance on one surface --
    the key path, the tools pane and every test call arrive with whatever the
    caller had remembered, and a subdivision at ``levels=99`` is not a refusal
    an op should have to write for itself. ``run`` is the choke point all three
    surfaces funnel through, so the range a ``Param`` declares is enforced once,
    where it cannot be bypassed.
    """
    from ....kernels.mesh.elements import OpError

    if not op.enabled(doc):
        return False
    values = defaults_for(op) | params
    for param in op.params:
        value = min(max(float(values[param.name]), param.low), param.high)
        values[param.name] = int(value) if param.stores_int else value
    head = doc.history.head
    mark = doc.history.mark()
    try:
        result = op.run(ctx, doc, **values)
    except OpError as error:
        toast(ctx, str(error))
        doc.history.collapse_since(mark)
        return False
    # An op that refuses *per object* -- ``run_mesh_op`` toasts and carries on
    # to the next one -- says so by returning False rather than by raising, so
    # a caller still learns that nothing happened.
    if result is False:
        doc.history.collapse_since(mark)
        return False
    _one_step(doc, op, mark, head)
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
    parameterised op's label goes -- "Bevel..." is an invitation to a dialog,
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


def any_object(doc: Any) -> bool:
    """Whether the *document* holds an object, selected or not.

    For the one op that means "tidy the shading" and says so: ``_shade_auto``
    falls back to every object when nothing is selected, and gating it on
    ``has_objects`` made that fallback unreachable -- a comment describing a
    branch nothing could take.
    """

    return bool(getattr(doc, "objects", ()))


def has_elements(doc: Any) -> bool:
    return bool(doc.element_sel)


def has_two_visible(doc: Any) -> bool:
    """Two selected objects the user can actually see. Merge's predicate, and
    the only one that has to look past ``selection`` at what is in it."""
    return sum(1 for obj in doc.objects if obj.uid in doc.selection and obj.visible) >= 2


def has_modifier_stack(doc: Any) -> bool:
    """Whether any selected object carries a modifier. Apply Modifiers' gate.

    Any modifier, enabled or not -- Apply drops the *whole* prefix through the
    one it is pressed for (``ClayDoc.apply_modifiers``'s own contract: a
    disabled entry in the prefix is dropped, not applied), so a stack that is
    entirely disabled still has something for the button to do.
    """
    return any(obj.modifiers for obj in doc.objects if obj.uid in doc.selection)


def has_three_selected(doc: Any) -> bool:
    """Exactly three -- Place Between's own gate.

    Not "at least three": the op reads two of the selection as anchors and
    moves the third, so a fourth selected object has no role to play, and
    silently ignoring it is worse than refusing outright and saying how many
    are selected -- the way ``_has_two_visible_reason`` already does for
    Merge/Union.
    """
    return len(doc.selection) == 3


def has_three_or_more_selected(doc: Any) -> bool:
    """Distribute's own gate -- at least three, not exactly three.

    Unlike Place Between, which reads two of the selection as fixed anchors
    and a third to move, Distribute treats every selected object as one item
    in a row: a fourth, fifth or sixtieth selected object is one more item to
    space, never a role the op runs out of. Two items have a single gap
    between them and nothing to distribute it against -- there is no "even"
    or "uneven" with one interval -- so the floor is three, but nothing above
    it is refused.
    """
    return len(doc.selection) >= 3


def has_two_or_more_selected(doc: Any) -> bool:
    """At least two -- Group Selected's and Parent to Last Selected's own
    gate. Grouping or parenting one object is the identity (there is
    nothing else to fold into the new empty, or to reparent onto the
    topmost one), the same "enabled but does nothing" trap ``has_two_visible``
    already refuses for Merge/Union.
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
# clay-07 (2026-09-06 audit): none of these existed, so Merge Objects, Bridge
# Loops and every element op greyed out with nothing anywhere saying why --
# ``op.hint`` was the only sentence attached to a disabled row, and it
# describes what the op does rather than why it is refused. Each of these
# answers only the "why"; :func:`reason_for` is what decides whether to ask.


def _has_objects_reason(doc: Any) -> str:
    return "" if has_objects(doc) else "Select an object first."


def _any_object_reason(doc: Any) -> str:
    return "" if any_object(doc) else "This document has no objects yet."


def _has_elements_reason(doc: Any) -> str:
    return "" if has_elements(doc) else "Select something in the viewport first."


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


def _has_modifier_stack_reason(doc: Any) -> str:
    if not has_objects(doc):
        return "Select an object first."
    return "" if has_modifier_stack(doc) else "The selection has no modifiers to apply."


def _has_three_selected_reason(doc: Any) -> str:
    n = len(doc.selection)
    return (
        ""
        if n == 3
        else f"Select exactly three objects first (two anchors, then the one to "
        f"place) -- {n} selected now."
    )


def _selection_reason(doc: Any) -> str:
    return "" if doc.selection else "Select an object first."


def _has_three_or_more_selected_reason(doc: Any) -> str:
    n = len(doc.selection)
    return "" if n >= 3 else f"Select at least three objects first -- {n} selected now."


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

    return call


def _dissolve(ctx: Any, doc: Any, **_: Any) -> None:
    """Dissolve, dispatched on the mode -- one menu row rather than three.

    "Dissolve" means the same thing to a user in every mode (get rid of this,
    and heal what it separated); it is only the implementation that differs, so
    splitting it into three rows would be exposing the implementation.
    """
    from ....kernels.mesh import ops_dissolve

    which = {
        "vertex": ops_dissolve.dissolve_verts,
        "edge": ops_dissolve.dissolve_edges,
        "face": ops_dissolve.dissolve_faces,
    }.get(doc.element_mode)
    if which is not None:
        run_mesh_op(ctx, doc, which)


def _extrude(ctx: Any, doc: Any, **params: Any) -> bool:
    """Extrude, dispatched on the mode -- one row and one key, as Dissolve is.

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


def _smooth(ctx: Any, doc: Any, levels: float = 1.0, **_: Any) -> None:
    """Catmull-Clark over every selected object, whatever the mode.

    It ignores the element selection deliberately -- a smoothing subdivision
    moves the original vertices, and moving only some of them tears the surface
    along the edge of the selection. See ``ops_subdiv.catmull_clark``.
    """
    from ....kernels.mesh import elements as el
    from ....kernels.mesh import ops_subdiv

    def one(doc: Any, obj: Any) -> None:
        mesh, sel = ops_subdiv.catmull_clark(obj.mesh, el.empty(), levels=int(levels))
        doc.set_mesh(obj.uid, mesh, select=sel)

    run_object_op(ctx, doc, one)


def _duplicate(ctx: Any, doc: Any, **_: Any) -> None:
    from ....kernels.mesh import selection

    del ctx
    selection.duplicate_selected(doc)


def _bake(ctx: Any, doc: Any, **_: Any) -> None:
    """Fold each selected object's transform into its geometry.

    Two ``doc`` calls -- ``set_mesh`` then ``set_transform`` -- because the mesh
    and the transform are separate edits in this document's vocabulary and each
    op should push its own kind of change rather than reach for a bake-shaped
    special case. But ``run``'s ``_one_step`` folds everything an op pushes into
    one history entry (the module docstring's "one press, one Ctrl+Z"), and
    that applies here exactly as it does to every other op: **a bake is one
    compound undo step**, and an earlier version of this docstring claimed the
    opposite -- that undoing a bake took a second press -- which was stale the
    day it was written (clay-11, 2026-09-06 audit): ``test_clay_ops.py``'s own
    ``test_every_op_that_changes_geometry_freezes_it`` already runs a bake
    through ``clay_ops.run`` and would have caught the fold landing as two
    steps rather than one.

    **Baked to world, and detached from its parent (tranche 3: scene
    structure).** ``bake_transform`` is handed the object's *world* matrix,
    not its own local TRS -- a parented object's local fields describe its
    placement relative to its parent, not what is on screen, and "fold the
    transform into the geometry" means the latter. ``bake_transform``'s own
    docstring says the object's ``parent`` field "is meaningless once its
    geometry has been baked to world space", and it means it literally: the
    reset local TRS is the identity regardless, so an object left parented
    would still inherit its parent's live transform on top of geometry that
    already contains it, moving a second time. ``set_parent(..., None,
    keep_world=False)`` runs *after* the local TRS has already been reset to
    identity, deliberately not ``keep_world=True`` -- the object's current
    world matrix at that point is only its parent's (the bake already threw
    the rest away into the mesh), and recomputing from it would reintroduce
    exactly the double transform this is closing.
    """
    from ....kernels.mesh import ops as clay_ops_geom

    def one(doc: Any, obj: Any) -> None:
        had_parent = obj.parent is not None
        baked = clay_ops_geom.bake_transform(obj, world=doc.world_matrix(obj.uid))
        doc.set_mesh(obj.uid, baked.mesh)
        doc.set_transform(
            obj.uid,
            translation=baked.translation,
            rotation=baked.rotation,
            scale=baked.scale,
        )
        if had_parent:
            doc.set_parent(obj.uid, None, keep_world=False)

    run_object_op(ctx, doc, one)


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

    **Merging applies every stack.** Every input is handed over with its
    *evaluated* mesh, not its base, so what you saw is what you get -- an
    object carrying a mirror or an array merges the shape on screen, not the
    single half or copy its base alone describes. ``join_objects``'s own
    ``clear_modifiers=True`` default drops the target's stack in the same
    step: it is now baked into the merge, and leaving it in place would apply
    it a second time the next time the target was drawn.

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
    evaluated = [replace(doc.by_uid(uid), mesh=doc.evaluated(uid)) for uid in uids]
    worlds = [doc.world_matrix(uid) for uid in uids]
    mesh = clay_ops_geom.join(evaluated, eps=float(weld), world=worlds)
    doc.join_objects(uids[0], mesh, uids[1:])
    # clay-08 (2026-09-08 audit): the absorbed objects leave ``doc.objects``
    # here, and their manifold-check cache entries would otherwise outlive
    # them -- see ``_forget_manifold``.
    _forget_manifold(ctx, uids[1:])
    doc.select([uids[0]])


def _union(ctx: Any, doc: Any, **_: Any) -> None:
    """Boolean-union every selected object into the topmost one.

    ``_join``'s shape exactly -- topmost visible as the target, the geometry
    from ``clay.ops_boolean``, the bookkeeping from ``ClayDoc.join_objects``,
    the survivor left selected -- and every one of those reasons carries over
    unchanged. What is different is only which function computes the mesh, and
    that a refusal is possible: a union is defined over closed solids, so
    ``ops_boolean.union`` raises :class:`OpError` where ``ops.join`` cannot.
    That is caught where every other element-op refusal is, in ``run``.

    In-process and synchronous, unlike a mesh pipeline: manifold is CPU and
    fast at the scale Clay authors at, and handing this to ``TaskRunner`` would
    mean a document edit landing from another thread.

    **Merging applies every stack** -- see :func:`_join`'s identical
    paragraph: every input is handed over with its evaluated mesh, so what
    you saw is what you get.

    **Every input's own world matrix, not its local TRS** -- see
    :func:`_join`'s identical paragraph: two objects under different
    parents are not correctly related by composing their own TRS alone.
    """
    from ....kernels.mesh import ops_boolean

    uids = [obj.uid for obj in doc.objects if obj.uid in doc.selection and obj.visible]
    evaluated = [replace(doc.by_uid(uid), mesh=doc.evaluated(uid)) for uid in uids]
    worlds = [doc.world_matrix(uid) for uid in uids]
    mesh = ops_boolean.union(evaluated, world=worlds)
    doc.join_objects(uids[0], mesh, uids[1:])
    # clay-08 (2026-09-08 audit): see ``_join``'s identical comment -- the
    # absorbed objects' manifold-check cache entries would otherwise outlive
    # them.
    _forget_manifold(ctx, uids[1:])
    doc.select([uids[0]])


def _difference(ctx: Any, doc: Any, **_: Any) -> None:
    """Boolean-subtract every other selected object from the topmost one.

    ``_union``'s shape exactly -- topmost visible as the target, the survivor
    left selected, ``_forget_manifold`` for whatever it absorbed -- and every
    one of those reasons carries over unchanged. What is different is only
    which ``ops_boolean`` function computes the mesh, and that here **order
    matters**: the target is also the minuend, so "select the block before the
    holes you want cut into it" is the one thing a user has to know that Union
    does not ask them to. See ``ops_boolean.KINDS`` for why.

    The 2026-09-11 audit's clay-04: ``ops_boolean`` has implemented and tested
    all three booleans since the module was written, and an MCP agent could
    already reach all three through ``studio/modes/clay/agent/dispatch.py``'s own ``clay_boolean``
    tool -- only the registry, which the menu, the tools pane and the keyboard
    all read, offered a human just this one's sibling.

    **Merging applies every stack** -- see :func:`_join`'s identical
    paragraph: every input is handed over with its evaluated mesh, so what
    you saw is what you get.

    **Every input's own world matrix, not its local TRS** -- see
    :func:`_join`'s identical paragraph.
    """
    from ....kernels.mesh import ops_boolean

    uids = [obj.uid for obj in doc.objects if obj.uid in doc.selection and obj.visible]
    evaluated = [replace(doc.by_uid(uid), mesh=doc.evaluated(uid)) for uid in uids]
    worlds = [doc.world_matrix(uid) for uid in uids]
    mesh = ops_boolean.difference(evaluated, world=worlds)
    doc.join_objects(uids[0], mesh, uids[1:])
    _forget_manifold(ctx, uids[1:])
    doc.select([uids[0]])


def _intersection(ctx: Any, doc: Any, **_: Any) -> None:
    """Boolean-intersect every selected object into the topmost one.

    ``_union``'s shape exactly, and order-free the way Union is: which object
    is selected first changes nothing about the answer, only which survives
    as the target. See ``_difference``'s docstring for the finding this and
    it both close.

    **Merging applies every stack** -- see :func:`_join`'s identical
    paragraph: every input is handed over with its evaluated mesh, so what
    you saw is what you get.

    **Every input's own world matrix, not its local TRS** -- see
    :func:`_join`'s identical paragraph.
    """
    from ....kernels.mesh import ops_boolean

    uids = [obj.uid for obj in doc.objects if obj.uid in doc.selection and obj.visible]
    evaluated = [replace(doc.by_uid(uid), mesh=doc.evaluated(uid)) for uid in uids]
    worlds = [doc.world_matrix(uid) for uid in uids]
    mesh = ops_boolean.intersection(evaluated, world=worlds)
    doc.join_objects(uids[0], mesh, uids[1:])
    _forget_manifold(ctx, uids[1:])
    doc.select([uids[0]])


def mirror(ctx: Any, doc: Any, axis: int, **_: Any) -> None:
    from ....kernels.mesh import ops as clay_ops_geom

    def one(doc: Any, obj: Any) -> None:
        doc.set_mesh(obj.uid, clay_ops_geom.mirror(obj, axis).mesh)

    run_object_op(ctx, doc, one)


MAX_ARRAY_COUNT = 200
"""The most instances one Array Linear/Array Radial call may ask for at once.

Every copy shares its source mesh (``clay.ops.duplicate``'s own property,
carried through unchanged by ``translated`` and ``rotated_about_origin``), so
this is not a mesh-memory ceiling the way ``primitives.MAX_SUBDIVISIONS`` is.
What it actually bounds is the outliner (one more row per instance) and the
document itself (one more ``Obj`` -- a uid, a name, three small transform
arrays -- recorded in the undo step and written to the ``.rblk`` on every
save). 200 sits comfortably above the module's own working example ("sixty
fence posts is one upload") with headroom for a first guess at the count
field before the outliner starts to drag. It is a **soft** guard rather than
a real bound: arraying an array multiplies rather than adds, so two presses
each near the ceiling already exceed it -- nothing here can stop that, only
the count any *one* press may ask for.
"""


def _array_linear(
    ctx: Any, doc: Any, count: float = 3.0, x: float = 1.0, y: float = 0.0, z: float = 0.0, **_: Any
) -> bool:
    """Copy the whole selection ``count - 1`` times, each further along one step.

    Whole-selection, not per-object: "array these three things five times" is
    what the words mean, and copying each object independently would
    interleave three arrays into one mess instead of moving the group as one.

    Modelled closely on ``clay.selection.duplicate_selected``: the same
    growing ``taken`` list, so many copies made in one press do not collide
    names with each other, and the same one ``add_objects`` call rather than
    one ``add_object`` per copy, for the identical reason that function
    gives -- and it applies with more force here, since one array can make
    far more than the handful ``duplicate_selected`` ever did. Each copy is
    ``ops.duplicate`` plus ``ops.translated``, and neither touches the mesh,
    so every copy shares the source's -- an array of sixty fence posts is one
    GPU upload, not sixty.
    """
    from ....kernels.mesh import document as bd
    from ....kernels.mesh import ops as clay_ops_geom

    del ctx
    n = int(count)
    if n <= 1:
        return False
    originals = list(doc.selection)
    taken = [obj.name for obj in doc.objects]
    made: list[Any] = []
    for k in range(1, n):
        offset = (x * k, y * k, z * k)
        for uid in originals:
            copy = clay_ops_geom.duplicate(doc.by_uid(uid), bd.new_uid(), taken=taken)
            taken.append(copy.name)
            made.append(clay_ops_geom.translated(copy, offset))
    doc.add_objects(made)
    # Originals and copies both, not just the newest generation: arraying an
    # array is a normal thing to want, and it only compounds if the group
    # stays whole.
    doc.select(originals + [obj.uid for obj in made])
    return bool(made)


def _closes_a_ring(angle: float) -> bool:
    """Whether *angle* degrees of sweep is a whole number of full turns, so
    the arc closes back on itself and has no far end distinct from its start.

    Compared with a tolerance rather than ``== 0``: this is read off a widget
    that stores a float, and ``360.0`` typed by a user or clamped by ``run``
    is not guaranteed to survive as bit-identical to the ``360.0`` this
    compares against.
    """
    remainder = abs(float(angle)) % 360.0
    return remainder < 1e-6 or remainder > 360.0 - 1e-6


def _array_radial(
    ctx: Any, doc: Any, count: float = 3.0, angle: float = 360.0, axis: float = 1.0, **_: Any
) -> bool:
    """Copy the whole selection ``count - 1`` times, spun about the *world*
    origin around ``axis`` and spread evenly over ``angle`` degrees.

    The world origin, not the object's own centre, because a single spoke
    rotated about an axis through itself overlaps its own copies rather than
    fanning out into a wheel -- the reviewer's own phrase for this op was
    "eight spokes around a hub". A hub that is not at the origin is reached
    by arraying at the origin and moving the whole result, not by a third
    number this op does not take: ``clay_ops.Param`` is scalar, and a centre
    is a point.

    ``t = k * angle / divisor`` for copy ``k``, and what decides *divisor* is
    whether the arc **closes**, which ``_closes_a_ring`` answers -- not
    whether ``count`` changes, since ``angle / count`` and ``angle /
    (count - 1)`` are *both* functions of ``count`` and both reposition every
    existing copy when it changes; that is not the property that tells the
    two cases apart.

    * A full turn (360, 720, a negative multiple...) has no last position
      distinct from its first, so dividing by ``count - 1`` puts copy
      ``n - 1`` exactly back on the original -- a "radial array of two" draws
      one spoke on top of another with nothing on screen to say a second one
      exists. Dividing by ``count`` instead spaces every copy, the original
      included, evenly around the whole circle: a full-turn array of four
      lands 90 degrees apart with nothing doubled.
    * An open arc has a real far end, and a user who asks for 180 degrees
      means the copies *reach* 180. Dividing by ``count`` falls short of
      that, more so as the count shrinks -- four copies over 90 degrees would
      land at 22.5, 45 and 67.5, nothing at 90 -- so an open arc divides by
      ``count - 1``, which is exactly what a straight line of evenly spaced
      points between two fixed ends means. At ``count == 2`` that divisor is
      1, so the one copy lands at ``angle`` exactly.

    Shares ``_array_linear``'s shape entirely otherwise -- the ``taken`` list,
    the one ``add_objects`` call, the whole selection left selected after.
    What is different is the per-copy step (``rotated_about_origin`` rather
    than ``translated``), and that is what keeps every copy a live parametric
    shape rather than a frozen one: a rotation about the origin is a
    transform change, and the mesh is never touched.
    """
    from ....kernels.mesh import document as bd
    from ....kernels.mesh import ops as clay_ops_geom

    del ctx
    n = int(count)
    a = int(axis)
    if n <= 1:
        return False
    divisor = n if _closes_a_ring(angle) else n - 1
    originals = list(doc.selection)
    taken = [obj.name for obj in doc.objects]
    made: list[Any] = []
    for k in range(1, n):
        degrees = k * angle / divisor
        for uid in originals:
            copy = clay_ops_geom.duplicate(doc.by_uid(uid), bd.new_uid(), taken=taken)
            taken.append(copy.name)
            made.append(clay_ops_geom.rotated_about_origin(copy, a, degrees))
    doc.add_objects(made)
    doc.select(originals + [obj.uid for obj in made])
    return bool(made)


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
    it, the same door :func:`_place_between` uses for its own world result.
    """
    from ....kernels.geom3d import math3d as m3
    from ....kernels.mesh import document as bd
    from ....kernels.mesh import ops as clay_ops_geom

    del ctx
    taken = [obj.name for obj in doc.objects]
    originals = list(doc.selection)
    made: list[Any] = []
    for uid in originals:
        source = doc.by_uid(uid)
        copy = clay_ops_geom.duplicate(source, bd.new_uid(), taken=taken)
        taken.append(copy.name)
        world = doc.world_matrix(uid)
        mirrored = clay_ops_geom.mirror_world(copy, int(axis), offset, world=world)
        new_world = m3.compose(mirrored.translation, mirrored.rotation, mirrored.scale)
        t, r, s = doc.local_from_world(uid, new_world)
        made.append(
            replace(mirrored, generator=None, params={}, translation=t, rotation=r, scale=s)
        )
    doc.add_objects(made)
    # Originals *and* copies, exactly as both arrays leave them, and for the
    # same reason: mirroring a mirror is a normal thing to want. Leaving the
    # copies unselected made "mirror across X, then mirror the pair across Z"
    # -- four table legs from one -- silently produce three, because the
    # second press saw only the original and re-mirrored it over a leg that
    # was already there (found by the furniture author, 2026-09-12).
    doc.select(originals + [obj.uid for obj in made])
    return bool(made)


def _place_between(ctx: Any, doc: Any, fit: float = 1.0, **_: Any) -> bool:
    """Move the newcomer onto the line between the other two selected objects.

    **Which two are anchors, and which one moves, both come from document
    order** -- ``doc.selection`` is a set and carries none, the same reason
    ``_join``/``_union`` read their own target out of ``doc.objects`` rather
    than out of the selection directly. The first two selected objects in
    that order are the anchors; the third is the one placed.

    **The anchors' own ``translation``, not their bounding-box centre.** That
    is the point the gizmo sits on and the number the TRS panel shows, so
    "between these two" means the same thing here as it does to a user
    looking at the panel -- a box's own centre can disagree with it the
    moment the object has been scaled or its mesh is not centred on the
    origin it was authored at. **Their *world* translation (tranche 3: scene
    structure)**, not the local field alone -- ``ops.place_between`` reads
    ``a``/``b`` as world points (its own docstring), and a parented anchor's
    ``translation`` is local to its parent, not where the gizmo actually
    sits.

    **The mover is the *last* in document order, which is the opposite of
    what ``_join``/``_union`` keep, and deliberately.** Those two keep the
    *incumbent*: a merge or a union absorbs newcomers into whichever object
    was already sitting there, so the survivor is the topmost (earliest)
    selected object. Here the newcomer is the one that has a role to play --
    you place two hubs first, and only then add a strut between them -- so
    the object this moves is whichever was selected last into the group, and
    a freshly added object sorts last in document order. The two ops read the
    same document order for the same reason (an unordered set needs a
    tiebreaker a user can see) and disagree about which end of it matters
    because they are answering different questions: "which of these survives"
    against "which of these is the newcomer".

    **``fit``'s span is read off the evaluated mesh.** ``ops.place_between``
    measures the mover's own local Y extent to stretch it across the gap
    (its own docstring), and an object under a modifier stack has a
    different footprint on screen than its base -- an array three copies
    long should fit the array's length, not one copy's. Only the mesh handed
    to the fit calculation changes; the transform this then writes back is
    unaffected, since the stack stays on top of the object exactly as it was.

    **The result converts back through the mover's own parent.** ``place_
    between`` answers in the same world space ``a``/``b`` were given in, so
    what it hands back is the mover's new *world* placement, not local TRS
    to write straight onto a parented mover -- ``doc.local_from_world`` is
    the same conversion :func:`_mirror_copy` runs its own world result
    through, for the identical reason.
    """
    from ....kernels.geom3d import math3d as m3
    from ....kernels.mesh import ops as clay_ops_geom

    del ctx
    uids = [obj.uid for obj in doc.objects if obj.uid in doc.selection]
    anchor_a, anchor_b, mover = uids[0], uids[1], uids[2]
    a = doc.world_matrix(anchor_a)[:3, 3]
    b = doc.world_matrix(anchor_b)[:3, 3]
    mover_footprint = replace(doc.by_uid(mover), mesh=doc.evaluated(mover))
    placed = clay_ops_geom.place_between(mover_footprint, a, b, fit=bool(fit))
    new_world = m3.compose(placed.translation, placed.rotation, placed.scale)
    t, r, s = doc.local_from_world(mover, new_world)
    return doc.set_transform(mover, translation=t, rotation=r, scale=s)


def _world_boxes(doc: Any, uids: Iterable[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{uid: world_box}`` for every *uid* whose mesh is not empty.

    Shared by :func:`_align`, :func:`_distribute` and :func:`_drop_to_ground`,
    which all hand ``mason.ops``' box arithmetic the same ``Boxes`` mapping
    ``scene.world_bounds`` already builds for Mason's own selection -- see
    that module's docstring for why the three take world boxes rather than a
    ``GeometrySource``. An object whose mesh has no vertices reports no box
    (``ops.world_box`` returns ``None``) and is left out rather than degrading
    every other object's math with a phantom point at the origin.

    **Measured off the evaluated mesh.** Align, Distribute and Drop to Ground
    all move objects by their visible extent, and an object under a solidify
    or an array modifier has a different one than its base -- see
    ``ops.world_box``'s own ``mesh`` override, which is what keeps this a
    one-line change: an object with no enabled modifiers evaluates to its own
    base mesh, ``is``-identical, so nothing here changes for the common case.

    **And off the object's own world matrix (tranche 3: scene structure)**,
    not the TRS composed from its own fields alone -- a parented object's
    translation/rotation/scale describe its placement relative to its
    parent, not where its box actually sits, and every caller here (and
    :func:`_apply_deltas`, which writes the deltas this produces back) means
    the box the user can see.
    """
    from ....kernels.mesh import ops as clay_ops_geom

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for uid in uids:
        obj = doc.by_uid(uid)
        box = clay_ops_geom.world_box(obj, doc.evaluated(uid), world=doc.world_matrix(uid))
        if box is not None:
            out[uid] = box
    return out


def _apply_deltas(doc: Any, deltas: dict[int, np.ndarray]) -> bool:
    """Add each world-space delta to its object's own translation. -> whether
    any object actually moved.

    One call per object rather than one ``set_transform`` per axis: ``run``'s
    own ``_one_step`` folds however many of these land into the single undo
    step its docstring promises, exactly as ``_bake``'s two-call-per-object
    fold already does, so a multi-object Align or Distribute is one Ctrl+Z
    whatever it moved.

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
    ran = False
    world_before = {uid: np.array(doc.world_matrix(uid), dtype="f8", copy=True) for uid in deltas}
    order = sorted(deltas, key=lambda uid: len(doc.ancestors(uid)))
    for uid in order:
        delta = deltas[uid]
        obj = doc.by_uid(uid)
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
    return ran


_ALIGN_MODES = ("min", "centre", "max")


def _align(ctx: Any, doc: Any, axis: float = 0.0, mode: float = 1.0, **_: Any) -> bool:
    """Line up every selected object's *world box* -- its visible edge or
    middle, not its pivot -- along one axis.

    The arithmetic is ``mason.ops.align``'s: Mason's placement math takes
    plain world boxes rather than a ``GeometrySource``, which is exactly the
    shape Clay's own ``ops.world_box`` already answers per object, so this is
    an import rather than a second copy. Mason may not import Clay (its own
    import pin says so, and for a real reason -- a scene links to a library
    asset by job id and must not resolve one itself), but nothing bars a
    plain ``studio/`` module reaching into Mason's pure package the way
    ``mason_view.py`` already does for the human-driven version of this same
    op; see that module's ``from .mason import ops as mops``.
    """
    from ..mason.engine import ops as mason_ops

    del ctx
    boxes = _world_boxes(doc, doc.selection)
    deltas = mason_ops.align(boxes, int(axis), _ALIGN_MODES[int(mode)])
    return _apply_deltas(doc, deltas)


def _distribute(ctx: Any, doc: Any, axis: float = 0.0, **_: Any) -> bool:
    """Space every selected object's world box evenly along one axis, the two
    extreme objects held fixed. See :func:`_align`'s docstring for why the
    box arithmetic is imported from ``mason.ops`` rather than duplicated.
    """
    from ..mason.engine import ops as mason_ops

    del ctx
    boxes = _world_boxes(doc, doc.selection)
    deltas = mason_ops.distribute(boxes, int(axis))
    return _apply_deltas(doc, deltas)


def _drop_to_ground(ctx: Any, doc: Any, **_: Any) -> bool:
    """Rest each selected object's own world-box *bottom* on ``y=0``.

    Per object, not per group: unlike Align and Distribute, there is no
    shared axis to agree on, so a box sitting three metres above the floor and
    one already resting on it both land correctly in the same call. The flat
    ground plane at ``y=0`` is this op's whole contract -- ``mason.ops.
    drop_to_ground`` also takes a ``Terrain`` for Mason's own version, which
    this row has no use for and does not pass.
    """
    from ..mason.engine import ops as mason_ops

    del ctx
    boxes = _world_boxes(doc, doc.selection)
    deltas = mason_ops.drop_to_ground(boxes, ground=0.0)
    return _apply_deltas(doc, deltas)


def _snap_to_grid(ctx: Any, doc: Any, step: float = 1.0, **_: Any) -> None:
    """Snap every selected object's translation onto a grid of *step* metres,
    each axis independently -- ``ops.snap_translation``'s own rounding
    (half away from zero, so the grid stays symmetric about the origin).
    """
    from ....kernels.mesh import ops as clay_ops_geom

    def one(doc: Any, obj: Any) -> None:
        snapped = clay_ops_geom.snap_translation(obj.translation, step)
        doc.set_transform(obj.uid, translation=snapped)

    run_object_op(ctx, doc, one)


# --- tranche 3: scene structure -- parenting, groups, separate, origin, lock -
#
# ``dev/CLAY-PLAN.md`` tranche 3. The document-layer doors (``ClayDoc.group``/
# ``set_parent``/``remove_object``/``set_origin``/``separate``, and the pure
# ``kernels.mesh.separate`` splitters) already carry the one-step contract and
# the locking refusals -- see ``kernels/mesh/document.py``'s own module
# docstring for both. What belongs here is only the registry wiring: which
# selection each row reads, and turning its result into the door call.


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
    ``_join``/``_union`` make for their own merge target: grouping is a
    gesture aimed at moving the group as one from here on, not at any one
    member of it.
    """
    del ctx
    uids = [obj.uid for obj in doc.objects if obj.uid in doc.selection]
    empty = doc.group(uids)
    doc.select([empty.uid])


def _ungroup(ctx: Any, doc: Any, **_: Any) -> None:
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

    run_object_op(ctx, doc, one)


def _parent_to_last(ctx: Any, doc: Any, **_: Any) -> None:
    """Parent every other selected object onto the topmost one in the
    outliner.

    The topmost selected object in document order is the new parent --
    ``doc.selection`` is a set with no order of its own, the same reason
    ``_join``/``_union`` read their own merge target out of ``doc.objects``
    rather than out of the selection directly (see either docstring). "Last
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
        return
    target, *rest = uids

    def one(doc: Any, obj: Any) -> None:
        doc.set_parent(obj.uid, target, keep_world=True)

    run_object_op(ctx, doc, one, uids=rest)


def _clear_parent(ctx: Any, doc: Any, **_: Any) -> None:
    """Every selected object becomes a root, keeping its world placement --
    ``ClayDoc.set_parent(uid, None, keep_world=True)`` per object, the same
    door :func:`_ungroup` and :func:`_parent_to_last` both use. An
    already-rootless object is a no-op (``set_parent`` pushes nothing for
    it), so a mixed selection of roots and children clears only the ones
    that had somewhere to fall from.
    """

    def one(doc: Any, obj: Any) -> None:
        doc.set_parent(obj.uid, None, keep_world=True)

    run_object_op(ctx, doc, one)


def _separate_loose(ctx: Any, doc: Any, **_: Any) -> bool:
    """Split every selected object into one new object per loose part.

    ``kernels.mesh.separate.by_loose_parts`` computes the pieces;
    ``ClayDoc.separate`` turns them into objects, copies the source's
    transform, parent and modifier stack onto each one and removes the
    source, as one step (its own docstring). A single-piece object refuses
    with a toast (``by_loose_parts``'s own "nothing to separate") and is
    left untouched, the rest of the selection still separating -- and, if
    that was the whole selection, this reports it ran nothing, the way
    :func:`_clean_mesh` does for its own identical "nothing to fix" case.
    """
    from ....kernels.mesh import separate as sep

    def one(doc: Any, obj: Any) -> None:
        doc.separate(obj.uid, sep.by_loose_parts(obj.mesh))

    return run_object_op(ctx, doc, one)


def _separate_material(ctx: Any, doc: Any, **_: Any) -> bool:
    """Split every selected object into one new object per material slot it
    uses. :func:`_separate_loose`'s identical shape, over
    ``kernels.mesh.separate.by_material`` instead."""
    from ....kernels.mesh import separate as sep

    def one(doc: Any, obj: Any) -> None:
        doc.separate(obj.uid, sep.by_material(obj.mesh))

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
        doc.separate(obj.uid, sep.by_selection(obj.mesh, doc.element_sel_of(obj.uid)))

    return run_object_op(ctx, doc, one)


def _origin_to_bounds(ctx: Any, doc: Any, **_: Any) -> None:
    """Move each selected object's origin to its own *world* box's centre.

    Measured off the evaluated mesh (``doc.evaluated``) and the object's
    world matrix (``doc.world_matrix``, tranche 3), the same pair
    :func:`_world_boxes` reads for Align/Distribute/Drop to Ground -- an
    object under a modifier stack, or under a parent, has a footprint and a
    placement its own base mesh and local TRS do not describe alone.
    ``ClayDoc.set_origin`` keeps the geometry and every child exactly where
    they were (its own docstring); only the pivot moves.
    """
    from ....kernels.mesh import ops as clay_ops_geom

    def one(doc: Any, obj: Any) -> None:
        box = clay_ops_geom.world_box(
            obj, mesh=doc.evaluated(obj.uid), world=doc.world_matrix(obj.uid)
        )
        if box is None:
            return
        lo, hi = box
        doc.set_origin(obj.uid, (lo + hi) * 0.5)

    run_object_op(ctx, doc, one)


def _origin_to_base(ctx: Any, doc: Any, **_: Any) -> None:
    """Move each selected object's origin to its own world box's bottom
    centre -- :func:`_origin_to_bounds`'s identical measurement, with the
    Y (up) component read off the box's *low* corner instead of its middle,
    the same axis ``_drop_to_ground`` rests on ``y=0``.
    """
    from ....kernels.mesh import ops as clay_ops_geom

    def one(doc: Any, obj: Any) -> None:
        box = clay_ops_geom.world_box(
            obj, mesh=doc.evaluated(obj.uid), world=doc.world_matrix(obj.uid)
        )
        if box is None:
            return
        lo, hi = box
        doc.set_origin(obj.uid, [(lo[0] + hi[0]) * 0.5, lo[1], (lo[2] + hi[2]) * 0.5])

    run_object_op(ctx, doc, one)


def _origin_to_selection(ctx: Any, doc: Any, **_: Any) -> None:
    """Move each selected object's origin to its own element selection's
    centroid.

    Read off the *base* mesh in local space -- editing always reads the
    base, never the evaluated mesh (the module docstring's rule) -- and
    carried into a world point through the object's own world matrix before
    ``ClayDoc.set_origin`` writes it back, since that door takes a world
    point (its own signature) and a parented object's local positions are
    not one.
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

    run_object_op(ctx, doc, one)


def _origin_to_world(ctx: Any, doc: Any, **_: Any) -> None:
    """Move each selected object's origin to the world origin -- the one
    origin-* row that needs no measurement at all."""

    def one(doc: Any, obj: Any) -> None:
        doc.set_origin(obj.uid, [0.0, 0.0, 0.0])

    run_object_op(ctx, doc, one)


def _lock(ctx: Any, doc: Any, **_: Any) -> None:
    """Lock every selected object -- ``ClayDoc.set_props`` per object, not a
    locking door itself (the module docstring's own exception list), so a
    selection that mixes locked and unlocked objects locks the rest without
    tripping over the ones already locked.
    """

    def one(doc: Any, obj: Any) -> None:
        doc.set_props(obj.uid, locked=True)

    run_object_op(ctx, doc, one)


def _unlock(ctx: Any, doc: Any, **_: Any) -> None:
    """Unlock every selected object. :func:`_lock`'s identical shape."""

    def one(doc: Any, obj: Any) -> None:
        doc.set_props(obj.uid, locked=False)

    run_object_op(ctx, doc, one)


def _forget_manifold(ctx: Any, uids: Iterable[int]) -> None:
    """Drop cached "mesh check" entries for objects that just left the document.

    The 2026-09-08 audit's clay-08: ``ClayState.manifold`` -- the per-object
    "last mesh check" cache the properties panel fills in (``clay_props``'s
    ``_diagnostics``) -- was only ever pruned when a tab *closed*
    (``clay_mode.close_tab``'s ``release``). Deleting, merging or unioning an
    object away mid-session left its entry keyed on the now-orphaned uid,
    pinning the whole ``Mesh`` (positions/loops/starts arrays) it measured
    alive, unreachable, for the rest of the tab's life. This is that same pop,
    at every other site an object leaves ``doc.objects``.

    Reached through ``ctx.state.clay`` with ``getattr`` at each hop, the way
    ``_frame`` above reaches ``ctx.clay_view``: this keeps the module callable
    with the bare toast-only ``ctx`` double the rest of this file's tests use,
    which has neither attribute.
    """
    state = getattr(ctx, "state", None)
    clay_state = getattr(state, "clay", None) if state is not None else None
    manifold = getattr(clay_state, "manifold", None)
    if manifold is None:
        return
    for uid in uids:
        manifold.pop(uid, None)


def _delete(ctx: Any, doc: Any, **_: Any) -> None:
    from ....kernels.mesh import selection

    before = {obj.uid for obj in doc.objects}
    for message in selection.delete_selected(doc):
        toast(ctx, message)
    _forget_manifold(ctx, before - {obj.uid for obj in doc.objects})


def _unwrap(ctx: Any, doc: Any, **_: Any) -> None:
    """Give every selected object a fresh box projection.

    Whole objects rather than the selected faces, and that is the decision:
    unwrapping half a mesh leaves the other half's coordinates from whenever
    they were last computed, so the two islands are at different texel
    densities and a checker map says so immediately. Per-face unwrapping is a
    real feature and it needs a seam tool first.

    It does **not** freeze the generator. UVs are not geometry -- the positions,
    the topology and the parameters are all untouched -- so a box that has been
    unwrapped is still describable as "box, size 1", and re-editing the size
    correctly rebuilds it with the generator's own canonical coordinates.
    """
    from ....kernels.mesh import uv as uv_mod

    def one(doc: Any, obj: Any) -> None:
        doc.set_mesh(obj.uid, uv_mod.box_unwrap(obj.mesh), keep_generator=True)

    run_object_op(ctx, doc, one)


# --- clean-mesh / recalc-normals (readiness's own FIX_OPS) ------------------
#
# ``kernels/mesh/readiness.py`` names both by these exact strings in its
# ``FIX_OPS`` -- a "Fix" button next to a readiness warning runs
# ``ops.get(check.fix)``, so the names here are load-bearing, not a label
# choice.


def _clean_mesh(
    ctx: Any, doc: Any, distance: float = 1e-5, fill_holes: float = 0.0, **_: Any
) -> bool:
    """Run :func:`~.ops_clean.clean` over every selected object, one undo step.

    Only objects :func:`~.ops_clean.clean` actually changed call ``set_mesh``
    -- its own identity contract ("every op is a no-op when there is nothing
    to fix") is what lets this tell "nothing was wrong" from "something was
    fixed" without measuring twice, the same way :func:`_shade_auto` reads its
    own ``smoothed is mesh`` check.
    """
    from ....kernels.mesh import ops_clean

    totals = dict(
        degenerate_removed=0,
        merged_vertices=0,
        duplicate_removed=0,
        loose_removed=0,
        holes_filled=0,
        faces_flipped=0,
    )
    changed = False

    def one(doc: Any, obj: Any) -> None:
        nonlocal changed
        mesh, report = ops_clean.clean(
            obj.mesh, distance=float(distance), fill_holes=bool(fill_holes)
        )
        if mesh is obj.mesh:
            return
        doc.set_mesh(obj.uid, mesh)
        changed = True
        for key in totals:
            totals[key] += getattr(report, key)

    run_object_op(ctx, doc, one)
    if not changed:
        # The 2026-09-19 audit (clay-24): the hint promises "no toast, no
        # step" for an already-clean object, but this used to toast "Nothing
        # to clean." anyway -- the one sentence a user reads before deciding
        # whether it is safe to press speculatively. Silent matches every
        # other no-op ``run()`` in this module (none of them toast either);
        # the caller already knows nothing happened from the ``False``.
        return False
    parts = [
        f"{n} {label}"
        for n, label in (
            (totals["degenerate_removed"], "degenerate"),
            (totals["merged_vertices"], "merged"),
            (totals["duplicate_removed"], "duplicate"),
            (totals["loose_removed"], "loose"),
            (totals["holes_filled"], "holes filled"),
            (totals["faces_flipped"], "faces flipped"),
        )
        if n
    ]
    ctx.toast("Cleaned: " + ", ".join(parts) + "." if parts else "Cleaned.")
    return True


def _recalc_normals(ctx: Any, doc: Any, **_: Any) -> None:
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

    run_object_op(ctx, doc, one)


def _apply_modifiers(ctx: Any, doc: Any, **_: Any) -> None:
    """Bake every selected object's whole modifier stack into its base mesh.

    ``ClayDoc.apply_modifiers`` already folds one object's bake into a single
    step of its own (a ``MeshEdit`` plus the ``ObjectPropsEdit`` that drops
    the baked prefix and, if the object still claimed one, freezes its
    generator -- see that method's own docstring); ``run_object_op`` supplies
    the per-object loop, the stale-uid tolerance and the per-object refusal
    toast every other object-level op in this file already gets, and
    ``run``'s own ``_one_step`` (the module docstring's "one press, one
    Ctrl+Z") folds however many of those land into one further step across
    the whole selection, the same two-layer fold :func:`_bake` already uses.
    An object whose stack currently refuses (a modifier in the prefix that
    errors right now) is toasted and left alone; the rest of the selection
    still bakes.
    """

    def one(doc: Any, obj: Any) -> None:
        doc.apply_modifiers(obj.uid)

    run_object_op(ctx, doc, one)


# --- decimate: Clay's first background op ------------------------------------
#
# Every op above runs synchronously on the frame thread -- cheap enough that a
# button press and its result are the same frame. gltfpack is a child process
# and is not: a triangle budget on a real mesh is real wall-clock, and running
# it inline would freeze the viewport for however long it takes. The shape
# below (``prepare`` on the frame thread, ``work`` off it, ``apply`` back on
# it) is deliberately general -- tranche 4 routes Blender retopology/UV/bake
# through the same three-function split -- and it has two dispatches:
#
# * **Interactive**: ``prepare`` snapshots what is selected, ``ctx.submit``
#   runs ``work`` on a task thread, and ``apply`` lands later from
#   ``clay_mode.on_task_done`` (routed there by the ``clay-bg:<tab uid>`` key
#   prefix). Refused with a toast, not queued, if a decimate for this tab is
#   already in flight -- ``TaskRunner.submit``'s own rule, the same one a
#   double-clicked Export already leans on.
# * **Inline**: an agent's sandboxed ``Ctx`` (``getattr(ctx, "inline", False)``)
#   has no frame/task-thread split to cross -- there is no second call the
#   agent makes later to collect a result the way a human's next frame would
#   -- so ``work`` and ``apply`` both run synchronously inside this call,
#   the same one-shot cost ``agent/tools_ops.py``'s ``_h_export`` pays
#   deliberately rather than the per-frame stall the split exists to prevent
#   everywhere else.
#
# **The GLB gltfpack sees carries no normals.** ``gltfpack`` will not simplify
# across an attribute discontinuity -- a split normal or a UV seam -- so a
# primitive built the way ``document.to_primitives`` builds one (splitting a
# vertex per corner on every flat face, to carry a *per-corner* normal) handed
# gltfpack nothing but discontinuities and it removed nothing at all (960 ->
# 960 triangles on a Clay ``uv_sphere``, measured 2026-09-19 by whoever wrote
# ``optimize.simplify_bytes``'s own docstring). Dropping the normal entirely
# is what lets a flat face's three corners share one vertex like any other
# corner would, so :func:`_decimate_primitives` builds its own primitives
# rather than reusing ``to_primitives`` -- keyed on ``(vertex, uv)`` alone,
# never on the smooth flag, so a flat-shaded face is never split for a reason
# gltfpack cannot see. ``-sa`` (the ``aggressive`` param) still restores real
# reduction on a heavily UV-seamed mesh -- see ``optimize.simplify_bytes``'s
# own docstring for that flag.
#
# **The palette survives the round trip by name, not by primitive order.**
# gltfpack is free to drop, split or reorder primitives while it simplifies,
# so a primitive's *position* in the output is not a safe way to recover which
# document material it came from. Each primitive's material is tagged with
# :func:`_material_tag` before it is written, and :func:`_decimate_mesh_from_glb`
# reads the tag back out rather than counting -- so the round trip creates no
# new palette slots for a material that already existed.


_MATERIAL_TAG_RE = re.compile(r"^__clay_decimate_material_(\d+)__$")


def _material_tag(index: int) -> str:
    return f"__clay_decimate_material_{index}__"


def _tri_count(mesh: Any) -> int:
    """Triangles a face-corner mesh renders as. ``bridge.py``'s own
    ``_triangles`` and ``readiness._tri_count`` compute the identical number
    the identical way (``corners - 2 * faces``, valid because every face has
    at least three corners); kept as its own small function here rather than
    imported from either, since both are on the far side of a layer this
    package may not reach into (a pane, and a sibling kernel module with no
    call of its own into this one)."""
    faces = max(0, len(mesh.starts) - 1)
    corners = len(mesh.loops)
    return max(0, corners - 2 * faces)


#: What :func:`_decimate_prepare` and :func:`_blender_multi_prepare` may run
#: :func:`_decimate_primitives` over in one call, summed across *every* uid
#: in the selection -- the 2026-09-19 audit's clay-40, found during this same
#: pass's own reading debt: ``_decimate`` checked each object's triangle
#: count against ``glbimport.MAX_TRIANGLES`` individually but never summed
#: the selection, and ``_blender_multi_prepare`` (retopo/unwrap/bake-detail's
#: shared prepare) had no check at all, so a selection of many legally-sized
#: objects drove ``_decimate_primitives`` -- earclip triangulation plus an
#: ``np.unique(key, axis=0)`` dedup -- once per uid with no way to refuse
#: partway through, exactly the unrefusable frame-thread stall
#: ``ops_boolean._refuse_complexity`` and ``ops.MAX_JOINED_CORNERS`` already
#: guard their own kernels against.
#:
#: Measured 2026-09-20 against ``_decimate_primitives`` itself, the
#: expensive step both callers spend their time in: linear in triangle
#: count at roughly 2.3 microseconds each (10k -> 0.033 s, 500k -> 1.14 s,
#: 2,000,000 -> 4.6 s, and four 500k-triangle objects summing to the same
#: 2,000,000 cost 4.4 s -- the same total either way, which is what "summed"
#: means here). 2,000,000 is the same order of magnitude as the reproduced
#: 4.4 s ``ops.MAX_JOINED_CORNERS`` was set against, and the same value as
#: ``ops_boolean.MAX_BOOLEAN_TRIANGLES``/``glbimport.MAX_TRIANGLES`` for the
#: same reason theirs matches: one mesh at the import ceiling is exactly
#: what a single already-legal object costs, so a selection's budget is "no
#: worse than one object at the ceiling," never a lower bar than an import
#: already clears.
MAX_PRIMITIVES_TRIANGLES = 2_000_000


def _decimate_primitives(mesh: Any, materials: Any) -> list[Any]:
    """*mesh* as one :class:`~.gltf.Primitive` per material slot it uses, with
    no normals -- see the section docstring above for why.

    Vertices are deduplicated by ``(vertex index, uv)`` -- never by the smooth
    flag or by which face a corner belongs to -- so two corners of one flat
    face that share a position and a uv share a vertex here exactly as they
    would on a smooth face. A uv seam still splits, because that is a real
    difference in the data this mesh carries, not an artefact of shading.
    """
    from ....kernels.geom3d import gltf as gltf_mod
    from ....kernels.mesh import mesh as bm_mod
    from ....kernels.mesh.earclip import corner_triangles

    if bm_mod.face_count(mesh) == 0:
        return []
    corners, tri_face = corner_triangles(
        mesh.positions, mesh.loops, mesh.starts, bm_mod.face_normals(mesh)
    )
    if len(corners) == 0:
        return []
    prims: list[Any] = []
    for material_index in np.unique(mesh.material).tolist():
        face_mask = mesh.material[tri_face] == material_index
        tri_corners = corners[face_mask]
        if len(tri_corners) == 0:
            continue
        flat = tri_corners.reshape(-1)
        vtx = mesh.loops[flat].astype("f8")
        if mesh.uv is not None:
            uv = mesh.uv[flat].astype("f8")
            key = np.concatenate([vtx[:, None], uv], axis=1)
        else:
            key = vtx[:, None]
        uniq_keys, inverse = np.unique(key, axis=0, return_inverse=True)
        indices = inverse.astype("u4").reshape(-1)
        positions = mesh.positions[uniq_keys[:, 0].astype("i4")]
        uvs = uniq_keys[:, 1:3].astype("f4") if mesh.uv is not None else None
        source = (
            materials[material_index]
            if 0 <= material_index < len(materials)
            else gltf_mod.Material()
        )
        tagged = replace(source, name=_material_tag(material_index))
        prims.append(
            gltf_mod.Primitive(
                positions=positions.astype("f4"),
                indices=indices,
                normals=None,
                uvs=uvs,
                material=tagged,
            )
        )
    return prims


def _decimate_prepare(doc: Any, uids: Iterable[int]) -> list[dict[str, Any]]:
    """Frame-thread half: one no-normals GLB per selected object, plus enough
    to detect a stale result and to fold the result back in later.

    ``doc.mesh_stamp`` is the token :meth:`_decimate_apply` checks before
    ever calling ``set_mesh`` -- see that method's own comment for why a
    result computed against a mesh the user has since edited must be
    discarded rather than silently overwriting the edit.

    Refuses up front, from :data:`MAX_PRIMITIVES_TRIANGLES`, if the
    selection's *summed* triangle count would run :func:`_decimate_primitives`
    past its budget -- see that constant's own comment (the 2026-09-19
    audit's clay-40). ``_decimate``'s own per-object check against
    ``glbimport.MAX_TRIANGLES`` catches one oversized object; it never
    summed a selection of several legal ones, which is what this catches
    instead.
    """
    from ....kernels.geom3d import glbwrite
    from ....kernels.geom3d import gltf as gltf_mod
    from ....kernels.mesh.elements import OpError

    uids = list(uids)
    total = sum(_tri_count(doc.by_uid(uid).mesh) for uid in uids)
    if total > MAX_PRIMITIVES_TRIANGLES:
        raise OpError(
            f"This decimate would need {total:,} triangles, past the "
            f"{MAX_PRIMITIVES_TRIANGLES:,} Clay can prepare in one pass. "
            "Select fewer objects, or simplify them first."
        )

    prepared: list[dict[str, Any]] = []
    for uid in uids:
        obj = doc.by_uid(uid)
        prims = _decimate_primitives(obj.mesh, doc.materials)
        if not prims:
            continue
        model = gltf_mod.Model([gltf_mod.Node(mesh=0)], [0], [prims], [])
        prepared.append(
            {
                "uid": uid,
                "name": obj.name,
                "stamp": doc.mesh_stamp(uid),
                "glb": glbwrite.write_glb(model),
                "material": int(obj.material),
                "before": _tri_count(obj.mesh),
            }
        )
    return prepared


def _decimate_work(
    prepared: list[dict[str, Any]],
    *,
    ratio: float,
    exe: Any,
    lock_border: bool,
    aggressive: bool,
) -> dict[str, Any]:
    """Off-thread half: run every prepared GLB through ``gltfpack``.

    Returns rather than raises on an :class:`~.optimize.OptimizeError` --
    ``{"error": str(error)}`` -- so the caller (whichever of the two dispatch
    paths this ran under) can toast the real gltfpack failure. A raised
    ``OptimizeError`` reaches the task layer's own generic "something went
    wrong" wording instead (``tasks.py``'s ``CARRIES_ITS_OWN_MESSAGE`` does not
    name it, and that module is outside this change's file ownership), which
    is a worse answer than this module catching its own dependency's error and
    saying so itself.
    """
    from ....pipelines import optimize as optimize_mod

    try:
        items = []
        for item in prepared:
            out = optimize_mod.simplify_bytes(
                item["glb"],
                ratio=ratio,
                exe=exe,
                lock_border=lock_border,
                aggressive=aggressive,
            )
            items.append({**item, "glb_out": out})
        return {"items": items, "ratio": float(ratio)}
    except optimize_mod.OptimizeError as error:
        return {"error": str(error)}


def _mesh_from_tagged_primitives(prims: list[Any], fallback_material: int) -> Any:
    """*prims*, each already :func:`_material_tag`-labelled the way
    :func:`_decimate_primitives` labels decimate's, merged into one Clay mesh
    on the palette slots their tags name.

    Reads each primitive's material back off its tag rather than trusting
    primitive order or count, both of which gltfpack -- and, since tranche 4,
    Blender's own glTF exporter -- are free to change. ``fallback_material``
    -- the object's own default material -- covers the one case a tag cannot
    survive: an untagged primitive (a gltfpack ``-km`` run that dropped the
    name, a hand-edited exe, an object Blender only ever saw with one
    material) still lands on a real palette slot rather than crashing.

    Extracted from :func:`_decimate_mesh_from_glb`, which is now the
    one-node-per-GLB case of this -- decimate always sends one object per
    call, so every primitive in the loaded model is that object's own. The
    Blender ops (:func:`_blender_objects_from_glb`) send several objects in
    one GLB and call this once per node instead.
    """
    from ....kernels.mesh import document as bd
    from ....kernels.mesh import glbimport
    from ....kernels.mesh import ops as clay_ops_geom
    from ....kernels.mesh import shading as shading_mod

    objs = []
    for index, prim in enumerate(prims):
        tag = prim.material.name if prim.material is not None else ""
        match = _MATERIAL_TAG_RE.match(tag or "")
        material_index = int(match.group(1)) if match else fallback_material
        mesh = glbimport._mesh_for(prim, material_index)
        objs.append(bd.Obj(uid=index, name=f"m{index}", mesh=mesh))
    merged = objs[0].mesh if len(objs) == 1 else clay_ops_geom.join(objs, eps=0.0)
    # Topology changed, so no per-face flag survives to be carried -- recomputed
    # at the shared default angle rather than left flat, which
    # ``glbimport._mesh_for`` would otherwise do for every face (no NORMAL
    # accessor went in, so none comes back, and its own smooth-flag guess reads
    # that as smooth for every triangle rather than auto-detecting hard edges).
    return shading_mod.auto_smooth(merged)


def _decimate_mesh_from_glb(data: bytes, fallback_material: int) -> Any:
    """One prepared-and-simplified GLB, merged back into a single Clay mesh.
    See :func:`_mesh_from_tagged_primitives` for the tag round trip itself --
    this is only the "one object per GLB" framing decimate sends."""
    from ....kernels.geom3d import gltf as gltf_mod
    from ....kernels.mesh.elements import OpError

    model = gltf_mod.load(data)
    prims = [
        prim
        for node in model.nodes
        if node.mesh is not None
        for prim in model.meshes[node.mesh]
    ]
    if not prims:
        raise OpError("Decimate produced a mesh with no geometry.")
    return _mesh_from_tagged_primitives(prims, fallback_material)


def _decimate_apply(ctx: Any, doc: Any, result: Any) -> None:
    """Fold a finished decimate's result into the document, one undo step.

    Called from two places: directly, inside :func:`_decimate` itself, on the
    inline agent path; and from ``clay_mode.on_task_done``, for the ``clay-bg``
    task key, once the background path's ``work`` has returned. Self-contained
    either way -- its own ``history.mark()``/``collapse_since`` fold whatever it
    pushes into one step named "Decimate", so calling it from inside
    :func:`run`'s own fold (the inline path) nests two collapses over the same
    range rather than conflicting with it.
    """
    if not isinstance(result, dict):
        return
    if "error" in result:
        ctx.toast(result["error"], "error")
        return
    items = result.get("items", [])
    ratio = float(result.get("ratio", 1.0))
    mark = doc.history.mark()
    head = doc.history.head
    total_before = 0
    total_after = 0
    applied: list[str] = []
    skipped: list[str] = []
    for item in items:
        uid = item["uid"]
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            skipped.append(item["name"])
            continue
        # The stamp taken in ``_decimate_prepare`` still has to match: an edit
        # made to this object while gltfpack ran means the result was computed
        # against geometry that no longer exists, and applying it would
        # silently discard whatever the user did in the meantime.
        if doc.mesh_stamp(uid) != item["stamp"]:
            skipped.append(obj.name)
            continue
        mesh = _decimate_mesh_from_glb(item["glb_out"], item["material"])
        doc.set_mesh(uid, mesh)
        total_before += item["before"]
        total_after += _tri_count(mesh)
        applied.append(obj.name)
    doc.history.collapse_since(mark)
    top = doc.history.top
    if top is not None and doc.history.head != head:
        top.label = "Decimate"
    _decimate_report(ctx, applied, skipped, total_before, total_after, ratio)


def _decimate_report(
    ctx: Any,
    applied: list[str],
    skipped: list[str],
    before: int,
    after: int,
    ratio: float,
) -> None:
    parts: list[str] = []
    if applied:
        parts.append(f"Decimated {before:,} -> {after:,} triangles.")
        target = before * ratio
        if target > 0 and after > target * 1.1:
            parts.append("That is more than asked for -- try Aggressive.")
    for name in skipped:
        parts.append(f"Skipped {name}: it changed while decimating.")
    if not parts:
        parts.append("Nothing to decimate.")
    ctx.toast(" ".join(parts))


def decimate_apply(ctx: Any, doc: Any, result: Any) -> None:
    """Public door for :func:`_decimate_apply`, called by
    ``clay_mode.on_task_done`` for the ``clay-bg`` task key -- the one caller
    outside this module, so it gets a name without the leading underscore
    the rest of this section's helpers keep."""
    _decimate_apply(ctx, doc, result)


def _tab_for(ctx: Any, doc: Any) -> Any:
    """The open ``ClayTab`` whose document is *doc*, or ``None``.

    Reached through ``ctx.state.clay`` with ``getattr`` at every hop, the way
    ``_forget_manifold`` above does -- this keeps the module callable with the
    bare toast-only ``ctx`` double the rest of this file's tests use, and with
    the agent's sandboxed ``Ctx``, neither of which carries a real tab list.
    """
    state = getattr(ctx, "state", None)
    clay_state = getattr(state, "clay", None) if state is not None else None
    if clay_state is None:
        return None
    for tab in getattr(clay_state, "docs", ()):
        if tab.doc is doc:
            return tab
    return None


def _decimate(
    ctx: Any,
    doc: Any,
    ratio: float = 0.5,
    keep_seams: float = 1.0,
    aggressive: float = 0.0,
    **_: Any,
) -> bool:
    from pathlib import Path

    from ....kernels.mesh import glbimport
    from ....kernels.mesh.elements import OpError

    uids = [obj.uid for obj in doc.objects if obj.uid in doc.selection]
    if not uids:
        return False
    if float(ratio) >= 1.0:
        raise OpError("Ratio is already 1.0 -- there is nothing to decimate.")

    exe = getattr(ctx, "gltfpack_exe", None) or ctx.svc.config.gltfpack_exe
    exe = Path(exe)
    if not exe.is_file():
        raise OpError(
            "gltfpack is not installed, so Decimate is unavailable. Install "
            "the base pack in Settings > Models."
        )

    for uid in uids:
        obj = doc.by_uid(uid)
        if _tri_count(obj.mesh) > glbimport.MAX_TRIANGLES:
            raise OpError(
                f"{obj.name} has more triangles than Decimate can process in one pass."
            )

    prepared = _decimate_prepare(doc, uids)
    if not prepared:
        raise OpError("Nothing to decimate.")

    work_kwargs = dict(
        ratio=float(ratio), exe=exe, lock_border=bool(keep_seams), aggressive=bool(aggressive)
    )

    if getattr(ctx, "inline", False):
        # No frame/task-thread split to cross -- see the section docstring.
        result = _decimate_work(prepared, **work_kwargs)
        _decimate_apply(ctx, doc, result)
        return True

    tab = _tab_for(ctx, doc)
    if tab is None:
        raise OpError("Decimate needs an open document tab.")
    submitted = ctx.submit(f"clay-bg:{tab.uid}", _decimate_work, prepared, **work_kwargs)
    if not submitted:
        raise OpError("A decimate is already running for this document.")
    tab.bg_busy = "Decimating..."
    return True


# --- retopo / smart-unwrap / bake-detail: Clay's first Blender ops -----------
#
# ``dev/CLAY-PLAN.md`` tranche 4. Three more background ops in decimate's own
# ``prepare``/``work``/``apply`` shape (the section above), with two real
# differences from it:
#
# * **One Blender launch per call, not one per object.** gltfpack starts in
#   milliseconds and decimate pays that cost once per selected object
#   (``_decimate_work``'s own loop); a bpy interpreter does not, so retopo,
#   unwrap and bake-detail's own high side all send *every* object they touch
#   in one combined GLB -- :func:`_blender_multi_prepare` -- and Blender's
#   own ``op_clay_retopo``/``op_clay_unwrap``/``op_clay_bake`` already handle
#   "every mesh object in this GLB, independently" (bake aside, where the
#   independence is deliberately not total -- see :func:`_bake_detail`).
# * **The GLB carries a world matrix, not just a mesh.** Decimate's objects
#   never interact with each other, so its own prepared GLB carries no
#   transform at all. A selected-to-active bake does interact -- the low
#   mesh's cage has to sit where the high meshes actually are -- so every
#   node here carries its object's *world* transform (``doc.world_matrix``,
#   decomposed), and the mesh data inside stays local, exactly like every
#   other glTF node ever written by this codebase.
#
# All three are refused up front, by name, when ``clay_blender.available()``
# says no -- :func:`_blender_enabled`/:func:`_blender_reason` (retopo,
# smart-unwrap) and :func:`_blender_bake_enabled`/:func:`_blender_bake_reason`
# (bake-detail, which also needs a second selected object). ``available()``'s
# own sentence is what ``Op.reason`` shows; nothing here writes a second one.
#
# **The three share one background task key** with decimate --
# ``clay-bg:<tab uid>`` -- because only one background Blender/gltfpack op
# makes sense running against one document at a time. What tells
# ``clay_mode.on_task_done`` which ``apply`` to run once the task lands is a
# ``"kind"`` field each of these three work functions puts in its own result
# (``"retopo"``/``"unwrap"``/``"bake"``); decimate's own result carries none,
# which is what keeps it the default case there rather than a fourth entry
# every one of these three would otherwise have needed to agree with by hand.


def _blender_node_name(uid: int) -> str:
    """The GLB node name a prepared object round-trips through Blender by.

    Not the Clay object's own ``name`` -- two objects can share one (nothing
    in this document enforces uniqueness the way a filesystem does), and a
    collision here would merge two objects' results under one node the way a
    stale stamp already has a story for but a silent name clash does not.
    """
    return f"__clay_blender_obj_{uid}__"


def _blender_multi_prepare(doc: Any, uids: Iterable[int]) -> tuple[bytes, list[dict[str, Any]]]:
    """Frame-thread half shared by retopo, unwrap and bake-detail's own two
    calls (high objects, then the low one): one combined GLB, one node per
    *uid* that still has geometry, tagged and world-placed.

    Each node's mesh is its object's *evaluated* mesh (``doc.evaluated`` --
    what a modifier stack actually built, the same rule ``_join``/``_union``
    already apply through their own evaluated-mesh copies) run through
    :func:`_decimate_primitives`, which gives it no normals and tags each
    primitive with :func:`_material_tag` -- see that function's own
    docstring for why Blender's importer, like gltfpack, is handed nothing it
    would mistake for a real attribute discontinuity, and why the tag survives
    a round trip name/order does not. Each node's transform is the object's
    *world* matrix (``doc.world_matrix``, decomposed into T/R/S) rather than
    its own local one, so several objects sent together land at their real
    relative positions -- see the section docstring for why that matters to
    bake-detail and costs nothing for retopo/unwrap, which never look past
    their own object.

    -> (glb bytes, meta), where meta has one entry per uid that actually had
    geometry to send, each carrying ``uid``, ``name`` (for a toast),
    ``node_name`` (what :func:`_blender_objects_from_glb` matches the result
    back by), ``stamp`` (:meth:`~.document.ClayDoc.mesh_stamp` at prepare
    time -- the same staleness guard decimate's own ``_decimate_prepare``
    takes) and ``material`` (the object's own default palette slot, the
    fallback :func:`_mesh_from_tagged_primitives` uses for a primitive whose
    tag did not survive). An empty *uids*, or a selection of objects with no
    geometry at all, comes back ``(b"", [])`` -- the caller's own "nothing to
    do" refusal, not this function's to raise.

    Refuses up front, from :data:`MAX_PRIMITIVES_TRIANGLES`, from the
    selection's *summed* evaluated triangle count -- the 2026-09-19 audit's
    clay-40: this function had no ceiling of any kind, unlike every sibling
    combine op (``ops_boolean._refuse_complexity``, ``ops.MAX_JOINED_CORNERS``),
    so a selection of many legally-sized objects drove
    :func:`_decimate_primitives` once per uid with no way to bail out
    partway through. Every uid is evaluated once, up front, so the meshes
    the refusal counts are exactly the ones the loop below turns into
    primitives -- summing and then evaluating a second time would double
    the modifier-stack cost this function already pays once.
    """
    from ....kernels.geom3d import glbwrite, math3d
    from ....kernels.geom3d import gltf as gltf_mod
    from ....kernels.mesh.elements import OpError

    uids = list(uids)
    evaluated = {uid: doc.evaluated(uid) for uid in uids}
    total = sum(_tri_count(mesh) for mesh in evaluated.values())
    if total > MAX_PRIMITIVES_TRIANGLES:
        raise OpError(
            f"This selection would need {total:,} triangles, past the "
            f"{MAX_PRIMITIVES_TRIANGLES:,} Clay can prepare for Blender in "
            "one pass. Select fewer objects, or simplify them first."
        )

    nodes: list[Any] = []
    meshes: list[Any] = []
    meta: list[dict[str, Any]] = []
    for uid in uids:
        obj = doc.by_uid(uid)
        prims = _decimate_primitives(evaluated[uid], doc.materials)
        if not prims:
            continue
        translation, rotation, scale = math3d.decompose(doc.world_matrix(uid))
        node_name = _blender_node_name(uid)
        nodes.append(
            gltf_mod.Node(
                name=node_name,
                translation=translation,
                rotation=rotation,
                scale=scale,
                mesh=len(meshes),
            )
        )
        meshes.append(prims)
        meta.append(
            {
                "uid": uid,
                "name": obj.name,
                "node_name": node_name,
                "stamp": doc.mesh_stamp(uid),
                "material": int(obj.material),
            }
        )
    if not meta:
        return b"", []
    model = gltf_mod.Model(nodes, list(range(len(nodes))), meshes, [])
    return glbwrite.write_glb(model), meta


def _blender_objects_from_glb(data: bytes, meta: list[dict[str, Any]]) -> dict[int, Any]:
    """*data*'s nodes, matched back to *meta* by :func:`_blender_node_name`,
    each merged into one Clay mesh via :func:`_mesh_from_tagged_primitives`.

    -> ``{uid: mesh}`` for every uid whose node came back with geometry. A uid
    in *meta* with no matching node, or a node with no primitives, is simply
    left out of the mapping -- the caller (:func:`_retopo_apply`/
    :func:`_unwrap_apply`) reports that the same way it reports a stale
    stamp, rather than this function raising over a Blender op that legally
    dropped an object (an obj_target of zero, an operator that found nothing
    to act on).
    """
    from ....kernels.geom3d import gltf as gltf_mod

    model = gltf_mod.load(data)
    by_node_name = {node.name: node for node in model.nodes if node.mesh is not None}
    out: dict[int, Any] = {}
    for item in meta:
        node = by_node_name.get(item["node_name"])
        if node is None:
            continue
        prims = model.meshes[node.mesh]
        if not prims:
            continue
        out[item["uid"]] = _mesh_from_tagged_primitives(prims, item["material"])
    return out


def _blender_timeout(ctx: Any) -> float:
    """The Blender timeout to hand ``blender_run.run_worker`` for one of
    these three ops.

    ``_q_mesh.py``'s remesh job -- the app's other quad-retopology-through-
    Blender op, ``op_remesh`` -- already passes ``self.config.rig_timeout``
    to ``blender_run.run_worker``; that is the one Settings-reachable number
    this app has for "how long is a Blender op allowed to run", so retopo,
    smart-unwrap and bake-detail reuse it rather than inventing a second,
    Clay-only config field that would answer the identical question. Reached
    through ``ctx.svc.config`` for the real interactive ``ctx``, exactly the
    way ``_decimate`` reaches ``ctx.svc.config.gltfpack_exe``; the agent's
    sandboxed ``_OpCtx`` has no ``svc`` of its own, so it carries the number
    as a plain ``blender_timeout`` attribute instead -- see that dataclass's
    own docstring, and ``clay_blender.BLENDER_TIMEOUT`` is the last resort
    when neither is reachable (a bare toast-only ``ctx`` test double).
    """
    from ....pipelines import blender_run

    timeout = getattr(ctx, "blender_timeout", None)
    if timeout is None:
        svc = getattr(ctx, "svc", None)
        timeout = getattr(getattr(svc, "config", None), "rig_timeout", None)
    return float(timeout) if timeout else blender_run.BLENDER_TIMEOUT


def _blender_available_reason(doc: Any) -> str:
    del doc
    from ....pipelines import clay_blender

    _ok, reason = clay_blender.available()
    return reason


def _blender_enabled(doc: Any) -> bool:
    """Retopologize/Smart Unwrap's own gate: Blender reachable, and a
    selection -- exactly ``decimate``'s own :data:`has_objects`, plus the one
    extra precondition every op this section adds needs."""
    from ....pipelines import clay_blender

    ok, _reason = clay_blender.available()
    return ok and has_objects(doc)


def _blender_reason(doc: Any) -> str:
    reason = _blender_available_reason(doc)
    return reason if reason else _has_objects_reason(doc)


def _blender_bake_enabled(doc: Any) -> bool:
    """Bake Detail's own gate: Blender reachable, and two *visible* selected
    objects -- the low target plus at least one high source, the same count
    :func:`has_two_visible` already checks for Merge/Union, which read their
    own target out of document order the identical way (see
    :func:`_bake_detail`'s own docstring)."""
    from ....pipelines import clay_blender

    ok, _reason = clay_blender.available()
    return ok and has_two_visible(doc)


def _blender_bake_reason(doc: Any) -> str:
    reason = _blender_available_reason(doc)
    return reason if reason else _has_two_visible_reason(doc)


def _blender_op_report(ctx: Any, verb: str, applied: list[str], skipped: list[str]) -> None:
    parts: list[str] = []
    if applied:
        parts.append(f"{verb}: {', '.join(applied)}.")
    for name in skipped:
        parts.append(f"Skipped {name}: it changed while running.")
    if not parts:
        parts.append("Nothing to do.")
    ctx.toast(" ".join(parts))


def _retopo_work(
    glb: bytes,
    meta: list[dict[str, Any]],
    *,
    target_faces: int,
    close_holes: bool,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    """Off-thread half of Retopologize. ``{"error": ...}`` on a
    :class:`~.clay_blender.ClayBlenderError`, exactly like
    :func:`_decimate_work`'s own ``OptimizeError`` catch, so the caller can
    toast the real Blender failure rather than the task layer's generic one.
    """
    from ....pipelines import clay_blender

    try:
        out_glb, report = clay_blender.retopo_bytes(
            glb, target_faces=target_faces, close_holes=close_holes, seed=seed, timeout=timeout,
        )
        return {"kind": "retopo", "glb_out": out_glb, "meta": meta, "report": report}
    except clay_blender.ClayBlenderError as error:
        return {"kind": "retopo", "error": str(error)}


def _unwrap_work(
    glb: bytes,
    meta: list[dict[str, Any]],
    *,
    angle_limit: float,
    island_margin: float,
    timeout: float,
) -> dict[str, Any]:
    """Off-thread half of Smart Unwrap. See :func:`_retopo_work`'s own
    docstring -- identical shape, the other pipeline function."""
    from ....pipelines import clay_blender

    try:
        out_glb, report = clay_blender.unwrap_bytes(
            glb, angle_limit=angle_limit, island_margin=island_margin, timeout=timeout,
        )
        return {"kind": "unwrap", "glb_out": out_glb, "meta": meta, "report": report}
    except clay_blender.ClayBlenderError as error:
        return {"kind": "unwrap", "error": str(error)}


def _bake_work(
    high_glb: bytes,
    low_glb: bytes,
    meta: list[dict[str, Any]],
    *,
    texture_size: int,
    cage_extrusion: float,
    maps: list[str],
    timeout: float,
) -> dict[str, Any]:
    """Off-thread half of Bake Detail. *meta* is the low object's own
    (:func:`_blender_multi_prepare` called with one uid) -- the high side's
    own meta is spent building ``high_glb`` and carried no further, since the
    high objects are consumed by the bake and never written back to."""
    from ....pipelines import clay_blender

    try:
        out_glb, report = clay_blender.bake_bytes(
            high_glb, low_glb, texture_size=texture_size, cage_extrusion=cage_extrusion,
            maps=maps, timeout=timeout,
        )
        return {"kind": "bake", "glb_out": out_glb, "meta": meta, "report": report}
    except clay_blender.ClayBlenderError as error:
        return {"kind": "bake", "error": str(error)}


def _retopo_apply(ctx: Any, doc: Any, result: Any) -> None:
    """Fold a finished Retopologize's result into the document, one undo
    step. :func:`_decimate_apply`'s own shape, with two differences: the
    result is one multi-object GLB read back by node name
    (:func:`_blender_objects_from_glb`), not one GLB per object, and every
    applied object's generator freezes with no ``keep_generator`` --
    retopology replaces the base mesh outright, exactly what the section of
    ``dev/CLAY-PLAN.md`` this closes asks for.
    """
    if not isinstance(result, dict):
        return
    if "error" in result:
        ctx.toast(result["error"], "error")
        return
    meta = result.get("meta") or []
    glb_out = result.get("glb_out")
    meshes = _blender_objects_from_glb(glb_out, meta) if glb_out else {}
    mark = doc.history.mark()
    head = doc.history.head
    applied: list[str] = []
    skipped: list[str] = []
    for item in meta:
        uid = item["uid"]
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            skipped.append(item["name"])
            continue
        # Exactly ``_decimate_apply``'s own guard: a result computed against
        # geometry the user has since edited is discarded rather than
        # silently overwriting the edit.
        if doc.mesh_stamp(uid) != item["stamp"]:
            skipped.append(obj.name)
            continue
        mesh = meshes.get(uid)
        if mesh is None:
            continue
        doc.set_mesh(uid, mesh)
        applied.append(obj.name)
    doc.history.collapse_since(mark)
    top = doc.history.top
    if top is not None and doc.history.head != head:
        top.label = "Retopologize"
    _blender_op_report(ctx, "Retopologized", applied, skipped)


def _unwrap_apply(ctx: Any, doc: Any, result: Any) -> None:
    """Fold a finished Smart Unwrap's result in, one undo step.
    :func:`_retopo_apply`'s shape exactly, except every ``set_mesh`` passes
    ``keep_generator=True`` -- UVs are not geometry, the same reason
    :func:`_unwrap` (Box Unwrap) never freezes either.
    """
    if not isinstance(result, dict):
        return
    if "error" in result:
        ctx.toast(result["error"], "error")
        return
    meta = result.get("meta") or []
    glb_out = result.get("glb_out")
    meshes = _blender_objects_from_glb(glb_out, meta) if glb_out else {}
    mark = doc.history.mark()
    head = doc.history.head
    applied: list[str] = []
    skipped: list[str] = []
    for item in meta:
        uid = item["uid"]
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            skipped.append(item["name"])
            continue
        if doc.mesh_stamp(uid) != item["stamp"]:
            skipped.append(obj.name)
            continue
        mesh = meshes.get(uid)
        if mesh is None:
            continue
        doc.set_mesh(uid, mesh, keep_generator=True)
        applied.append(obj.name)
    doc.history.collapse_since(mark)
    top = doc.history.top
    if top is not None and doc.history.head != head:
        top.label = "Smart Unwrap"
    _blender_op_report(ctx, "Unwrapped", applied, skipped)


def _blender_bake_material(data: bytes | None) -> Any:
    """The one baked material :func:`~.blender_worker.op_clay_bake` wrote
    into the low object's own GLB, or ``None`` if there is nothing to read."""
    if not data:
        return None
    from ....kernels.geom3d import gltf as gltf_mod

    model = gltf_mod.load(data)
    for node in model.nodes:
        if node.mesh is None:
            continue
        for prim in model.meshes[node.mesh]:
            if prim.material is not None:
                return prim.material
    return None


def _bake_apply(ctx: Any, doc: Any, result: Any) -> None:
    """Fold a finished Bake Detail's result into the low object's material,
    one undo step.

    Unlike retopo/unwrap this touches no mesh at all -- the low object's
    geometry is exactly what it was before the bake -- so what lands is a
    single :meth:`~.document.ClayDoc.set_material`, replacing the palette
    entry by identity (``replace(material, ...)``) rather than mutating one
    in place, the same rule the section this closes states. Only the maps
    Blender was actually asked to bake (``report["maps"]``) touch the
    replacement's texture fields; an unrequested one keeps whatever the
    palette entry already had. ``metallic_factor`` is the one field set
    unconditionally -- ``op_clay_bake`` always reads it off the high side's
    own materials and reports it, independent of which maps were chosen.
    """
    if not isinstance(result, dict):
        return
    if "error" in result:
        ctx.toast(result["error"], "error")
        return
    meta = result.get("meta") or []
    if not meta:
        return
    item = meta[0]
    uid = item["uid"]
    try:
        obj = doc.by_uid(uid)
    except KeyError:
        ctx.toast(f"Skipped {item['name']}: it no longer exists.", "error")
        return
    if doc.mesh_stamp(uid) != item["stamp"]:
        ctx.toast(f"Skipped {obj.name}: it changed while baking.", "error")
        return
    material = _blender_bake_material(result.get("glb_out"))
    if material is None:
        ctx.toast("Bake produced no material.", "error")
        return
    report = result.get("report") or {}
    maps = report.get("maps") or []
    fields: dict[str, Any] = {}
    if "base_color" in maps:
        fields["base_color"] = material.base_color
        fields["base_color_factor"] = material.base_color_factor
    if "roughness" in maps:
        fields["metallic_roughness"] = material.metallic_roughness
        fields["roughness_factor"] = material.roughness_factor
    if "normal" in maps:
        fields["normal"] = material.normal
    if "metallic" in report:
        fields["metallic_factor"] = float(report["metallic"])
    index = int(obj.material)
    mark = doc.history.mark()
    head = doc.history.head
    doc.set_material(index, replace(doc.materials[index], **fields))
    doc.history.collapse_since(mark)
    top = doc.history.top
    if top is not None and doc.history.head != head:
        top.label = "Bake Detail"
    ctx.toast(f"Baked onto {obj.name}.")


def retopo_apply(ctx: Any, doc: Any, result: Any) -> None:
    """Public door for :func:`_retopo_apply`, called by
    ``clay_mode.on_task_done`` for a ``clay-bg`` result tagged
    ``"kind": "retopo"``. See :func:`decimate_apply`'s own docstring for why
    this file exposes one undecorated name per background op's ``apply``."""
    _retopo_apply(ctx, doc, result)


def unwrap_apply(ctx: Any, doc: Any, result: Any) -> None:
    """Public door for :func:`_unwrap_apply`; see :func:`retopo_apply`."""
    _unwrap_apply(ctx, doc, result)


def bake_apply(ctx: Any, doc: Any, result: Any) -> None:
    """Public door for :func:`_bake_apply`; see :func:`retopo_apply`."""
    _bake_apply(ctx, doc, result)


def _retopo(
    ctx: Any,
    doc: Any,
    target_faces: float = 5000.0,
    close_holes: float = 0.0,
    seed: float = 0.0,
    **_: Any,
) -> bool:
    """Send the selection's evaluated meshes to Blender for a quad
    retopology, replacing each object's base mesh (generator frozen, modifier
    stack kept -- ``set_mesh``'s own default). Object mode, every selected
    object independently -- unlike bake-detail, retopo/unwrap have no notion
    of "the target": every selected object gets its own result back.
    """
    from ....kernels.mesh.elements import OpError
    from ....pipelines import clay_blender

    ok, reason = clay_blender.available()
    if not ok:
        raise OpError(reason)

    uids = [obj.uid for obj in doc.objects if obj.uid in doc.selection]
    if not uids:
        return False

    glb, meta = _blender_multi_prepare(doc, uids)
    if not meta:
        raise OpError("Nothing to retopologize.")

    work_kwargs = dict(
        target_faces=int(target_faces),
        close_holes=bool(close_holes),
        seed=int(seed),
        timeout=_blender_timeout(ctx),
    )

    if getattr(ctx, "inline", False):
        # No frame/task-thread split to cross -- see decimate's own identical
        # branch, the precedent this and the other two follow.
        result = _retopo_work(glb, meta, **work_kwargs)
        _retopo_apply(ctx, doc, result)
        return True

    tab = _tab_for(ctx, doc)
    if tab is None:
        raise OpError("Retopologize needs an open document tab.")
    submitted = ctx.submit(f"clay-bg:{tab.uid}", _retopo_work, glb, meta, **work_kwargs)
    if not submitted:
        raise OpError("A background Blender op is already running for this document.")
    tab.bg_busy = "Retopologizing..."
    return True


def _smart_unwrap(
    ctx: Any,
    doc: Any,
    angle_limit: float = 66.0,
    island_margin: float = 0.003,
    **_: Any,
) -> bool:
    """Smart-UV-Project the selection in Blender. UVs only -- keeps the
    generator (``set_mesh(..., keep_generator=True)``), the way Box Unwrap
    already does, since an unwrap changes no geometry."""
    from ....kernels.mesh.elements import OpError
    from ....pipelines import clay_blender

    ok, reason = clay_blender.available()
    if not ok:
        raise OpError(reason)

    uids = [obj.uid for obj in doc.objects if obj.uid in doc.selection]
    if not uids:
        return False

    glb, meta = _blender_multi_prepare(doc, uids)
    if not meta:
        raise OpError("Nothing to unwrap.")

    work_kwargs = dict(
        angle_limit=float(angle_limit),
        island_margin=float(island_margin),
        timeout=_blender_timeout(ctx),
    )

    if getattr(ctx, "inline", False):
        result = _unwrap_work(glb, meta, **work_kwargs)
        _unwrap_apply(ctx, doc, result)
        return True

    tab = _tab_for(ctx, doc)
    if tab is None:
        raise OpError("Smart Unwrap needs an open document tab.")
    submitted = ctx.submit(f"clay-bg:{tab.uid}", _unwrap_work, glb, meta, **work_kwargs)
    if not submitted:
        raise OpError("A background Blender op is already running for this document.")
    tab.bg_busy = "Unwrapping..."
    return True


def _bake_detail(
    ctx: Any,
    doc: Any,
    texture_size: float = 2.0,
    cage_extrusion: float = 0.02,
    bake_base_color: float = 1.0,
    bake_roughness: float = 1.0,
    bake_normal: float = 1.0,
    **_: Any,
) -> bool:
    """Bake every other selected, visible object onto the *topmost* selected
    one in the outliner.

    Clay has no notion of an "active object" the way Blender does, so the low
    (target) object is picked by document order -- the identical
    "topmost selected is the target" rule :func:`_join`/:func:`_union`
    already use for their own merge target, for the identical reason: a
    selection is a set with no order of its own, and document order is the
    one ordering a user can actually see (the outliner). Every other
    selected, visible object is a high source. The target must already carry
    UVs (Smart Unwrap first, or its own generator's) -- refused by
    ``clay_blender.bake_bytes`` itself, by name, rather than baking a blank
    atlas.
    """
    from ....kernels.mesh.elements import OpError
    from ....kernels.rig import blender_spec
    from ....pipelines import clay_blender

    ok, reason = clay_blender.available()
    if not ok:
        raise OpError(reason)

    uids = [obj.uid for obj in doc.objects if obj.uid in doc.selection and obj.visible]
    if len(uids) < 2:
        return False
    low_uid, high_uids = uids[0], uids[1:]

    maps = [
        name
        for name, flag in (
            ("base_color", bake_base_color),
            ("roughness", bake_roughness),
            ("normal", bake_normal),
        )
        if flag
    ]
    if not maps:
        raise OpError("Choose at least one map to bake.")

    high_glb, high_meta = _blender_multi_prepare(doc, high_uids)
    low_glb, low_meta = _blender_multi_prepare(doc, [low_uid])
    if not high_meta or not low_meta:
        raise OpError("Nothing to bake.")

    sizes = blender_spec.CLAY_TEXTURE_SIZES
    work_kwargs = dict(
        texture_size=int(sizes[int(texture_size)]),
        cage_extrusion=float(cage_extrusion),
        maps=maps,
        timeout=_blender_timeout(ctx),
    )

    if getattr(ctx, "inline", False):
        result = _bake_work(high_glb, low_glb, low_meta, **work_kwargs)
        _bake_apply(ctx, doc, result)
        return True

    tab = _tab_for(ctx, doc)
    if tab is None:
        raise OpError("Bake Detail needs an open document tab.")
    submitted = ctx.submit(
        f"clay-bg:{tab.uid}", _bake_work, high_glb, low_glb, low_meta, **work_kwargs
    )
    if not submitted:
        raise OpError("A background Blender op is already running for this document.")
    tab.bg_busy = "Baking..."
    return True


# --- tranche 5: modelling breadth --------------------------------------------
#
# ``dev/CLAY-PLAN.md`` tranche 5. ``kernels.mesh.ops_model`` and
# ``kernels.mesh.ops_spin`` are the kernel half; what belongs here is the
# registry wiring -- which selection each row reads, which of its numbers
# becomes a ``Param``, and the handful (bisect, knife, spin, screw,
# symmetrize) whose kernel signature takes a point, a vector or a choice the
# kernel reads by sign rather than by index, none of which ``_element`` can
# forward as-is. Every selection-taking op that needs nothing else still goes
# through ``_element`` exactly like Bevel/Loop Cut/Weld above.


def _bisect(
    ctx: Any, doc: Any, axis: float = 0.0, clear: float = 0.0, fill: float = 0.0, **_: Any
) -> bool:
    """Cut the selected faces with a plane through the object's own local
    origin, normal to the chosen axis.

    ``ops_model.bisect``'s own ``point``/``normal`` are not numbers a
    ``Param`` dialog can offer -- ``Param`` is scalar by design (see its own
    docstring) -- so this always cuts through the object's own local origin,
    the same "no third number, move the object instead" trade
    ``array-radial``'s own docstring documents for its world-origin hub, here
    the *local* origin because ``ops_model.bisect`` works in the object's own
    space (``_element``'s own contract, with no ``doc.world_matrix`` in
    reach). Knife is the row for an arbitrary plane, drawn as a line in the
    viewport rather than dialled in.

    ``clear``'s three choices are the kernel's own values (0 keep both, 1
    remove the negative side, 2 remove the positive), forwarded by index
    exactly as ``array-radial``'s own ``axis`` already is -- unlike
    :func:`_symmetrize`'s ``direction`` below, nothing here needs translating.
    """
    from ....kernels.mesh import ops_model

    normal = [0.0, 0.0, 0.0]
    normal[int(axis)] = 1.0
    return run_mesh_op(
        ctx,
        doc,
        ops_model.bisect,
        point=(0.0, 0.0, 0.0),
        normal=tuple(normal),
        clear=int(clear),
        fill=bool(fill),
    )


def _knife(ctx: Any, doc: Any, point: Any = None, normal: Any = None, **_: Any) -> bool:
    """Arms the viewport's click-drag knife gesture, or -- once it has drawn
    a line -- commits the cut it defines.

    ``ops_model.knife`` needs a real point and normal, which neither a bare
    context-menu row nor a tools-pane button can supply: a zero-``Param`` op
    fires as ``clay_ops.run(ctx, doc, op)``, no other keyword at all
    (``ui/panes/tools.py`` and ``ui/panes/menu.py``'s identical shape, and
    ``clay_mode._registry_key``'s for the keyboard path -- this row
    deliberately carries no ``key`` for exactly that reason: firing it bare
    has no sane default the way Extrude's "zero offset, then drag" does,
    since there is no sane default *plane*). So firing it bare **arms** the
    viewport instead: ``ctx.clay_view`` is the live ``ClayView``, reached the
    same way every other keyboard-adjacent door in this mode already does
    (``clay_mode.handle_key``'s own ``getattr(ctx, "clay_view", None)``),
    and ``ClayView.begin_knife`` (``ui/_view_drag.py``) puts it into the
    gesture. The *next* press-drag-release draws the line and calls back in
    here with a real plane, so a cut still gets ``run``'s one-step-undo fold
    like every other op -- and a caller with no ``clay_view`` at all (every
    test double in ``test_clay_ops.py``, and a headless script driving Clay
    with no window) gets the plain refusal this row has always raised,
    :class:`~.kernels.mesh.elements.OpError`, caught and toasted where every
    other refusal in this registry is.

    **``point``/``normal`` are world space, not local** -- the plane through
    the drag line and the camera's forward direction, per the tranche 5
    integration spec -- and are converted into each selected object's own
    local frame here, not by the caller. ``ops_model.knife`` (like
    ``ops_model.bisect``) works in local space, exactly the contract
    ``_bisect`` and ``_element`` already depend on, so this cannot simply
    forward one ``point``/``normal`` pair to :func:`run_mesh_op` the way
    every other selection-taking row does: two objects rarely share a local
    frame, and with parenting landed (tranche 3) they may not even share a
    parent's. The conversion uses ``doc.world_matrix(uid)`` -- never a
    hand-composed TRS, the rule every parented write in this file follows --
    but *not* the same way for both halves of the plane: ``point`` is an
    ordinary position and converts by the matrix's plain inverse, while
    ``normal`` is a covector and converts by the **transpose of the forward
    matrix's linear 3x3**. The two agree only when the object carries no
    non-uniform scale, which is why a hand-rolled ``inverse @ normal`` would
    look right on every rotated, uniformly-scaled test object and quietly
    tilt the cut the day someone stretches one axis.

    **Every object with a face selection is cut by the same world plane.**
    The honest reading of "several objects have faces selected" when only
    one line was drawn: the plane the user drew is one plane in the world
    they are looking at, not one per object, so each object's own local
    version of that single world plane is what cuts it -- matching
    ``_union``/``_join``/``_apply_deltas``'s own "every input's own world
    matrix" rule for the identical reason. The op's own hint says so.
    """
    from ....kernels.mesh import ops_model
    from ....kernels.mesh.elements import OpError

    if point is None or normal is None:
        view = getattr(ctx, "clay_view", None)
        begin = getattr(view, "begin_knife", None)
        if begin is not None and begin(doc):
            return False
        raise OpError("Draw a knife cut across the selected faces first.")

    world_point = np.asarray(point, dtype="f8")
    world_normal = np.asarray(normal, dtype="f8")
    ran = False
    for uid in list(doc.element_sel):
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            continue
        world = np.asarray(doc.world_matrix(uid), dtype="f8")
        try:
            inverse = np.linalg.inv(world)
        except np.linalg.LinAlgError:
            continue
        local_point = (inverse @ np.append(world_point, 1.0))[:3]
        # The forward matrix's own linear part, transposed -- not
        # ``inverse``'s -- per this function's own docstring on why a plane
        # normal is not a point.
        local_normal = world[:3, :3].T @ world_normal
        length = float(np.linalg.norm(local_normal))
        if length < 1e-12:
            continue
        local_normal = local_normal / length
        try:
            mesh, sel = ops_model.knife(
                obj.mesh, doc.element_sel_of(uid), point=local_point, normal=local_normal
            )
        except OpError as error:
            toast(ctx, str(error))
            continue
        doc.set_mesh(uid, mesh, select=sel)
        ran = True
    return ran


def _spin(
    ctx: Any, doc: Any, axis: float = 1.0, angle: float = 360.0, steps: float = 8.0, **_: Any
) -> bool:
    """Lathe the selected edge profile (a chain or a loop) about the object's
    own local origin.

    ``center`` is ``ops_spin.spin``'s own point and not a dial, for the
    identical reason :func:`_bisect`'s plane is not one: always the object's
    own local origin, the same "no third number" trade ``array-radial``
    documents for its own hub.
    """
    from ....kernels.mesh import ops_spin

    return run_mesh_op(
        ctx,
        doc,
        ops_spin.spin,
        axis=int(axis),
        angle=float(angle),
        steps=int(steps),
        center=(0.0, 0.0, 0.0),
    )


def _screw(
    ctx: Any,
    doc: Any,
    axis: float = 1.0,
    angle: float = 360.0,
    steps: float = 8.0,
    height: float = 1.0,
    **_: Any,
) -> bool:
    """:func:`_spin`'s identical shape, plus the per-step lift along ``axis``
    that turns a lathe into a helix -- ``ops_spin.screw``'s own docstring:
    even a whole-turn ``angle`` never closes the seam. Same local-origin
    ``center``, same reason.
    """
    from ....kernels.mesh import ops_spin

    return run_mesh_op(
        ctx,
        doc,
        ops_spin.screw,
        axis=int(axis),
        angle=float(angle),
        steps=int(steps),
        height=float(height),
        center=(0.0, 0.0, 0.0),
    )


def _symmetrize(ctx: Any, doc: Any, axis: float = 1.0, direction: float = 1.0, **_: Any) -> None:
    """Symmetrize every selected object across its own local plane, whatever
    the element mode.

    **Ignores the element selection, like Smooth** (:func:`_smooth`'s
    identical shape: whole objects, ``run_object_op``, not ``run_mesh_op``) --
    ``ops_model.symmetrize`` states its own reason: a delete-then-mirror pass
    means the whole object or it means nothing, the same way a smoothing
    subdivision cannot move only some of a surface's vertices without tearing
    it.

    **``direction`` is a choice, not the kernel's own signed float.** A
    choice ``Param``'s stored value is always ``0..len(choices) - 1`` (the
    dataclass's own constraint), so ``0`` cannot mean "the negative side" the
    way ``ops_model.symmetrize``'s own ``direction >= 0`` test reads it --
    unlike :func:`_bisect`'s ``clear``, whose three kernel values already
    start at 0 and forward unchanged, this one needs translating.
    """
    from ....kernels.mesh import elements as el
    from ....kernels.mesh import ops_model

    axis_i = int(axis)
    direction_v = -1.0 if int(direction) == 0 else 1.0

    def one(doc: Any, obj: Any) -> None:
        mesh, sel = ops_model.symmetrize(obj.mesh, el.empty(), axis=axis_i, direction=direction_v)
        doc.set_mesh(obj.uid, mesh, select=sel)

    run_object_op(ctx, doc, one)


def _shade(smooth: bool) -> Callable[..., None]:
    """Set the shading flag on the selected faces, or on whole objects.

    Both modes, because both readings are real: in face mode a user means
    "these faces", and in object mode they mean "this whole shape". The flag is
    per face either way -- there is no object-level shading setting that would
    have to be kept in agreement with it.
    """

    def run(ctx: Any, doc: Any, **_: Any) -> None:
        if doc.element_mode == "object":
            run_object_op(ctx, doc, lambda doc, obj: doc.set_shading(obj.uid, None, smooth))
            return
        from ....kernels.mesh import elements as el

        del ctx
        for uid in list(doc.element_sel):
            faces = el.convert(doc.by_uid(uid).mesh, doc.element_sel_of(uid), "face")
            doc.set_shading(uid, faces.faces, smooth)

    return run


def _shade_auto(ctx: Any, doc: Any, angle: float = _shading.DEFAULT_ANGLE, **_: Any) -> None:
    """Smooth every face whose *every* neighbour agrees with it to within *angle*.

    Delegates to :func:`clay.shading.auto_smooth`, which is the specification
    -- see its docstring for the full rule and for why a capped cylinder comes
    out entirely flat on purpose. What is left here, after the 2026-09-06
    audit's organic-shapes extraction, is only the object-selection plumbing:
    which objects to run over, and folding an unchanged one into no edit at
    all rather than a no-op history step.
    """

    def one(doc: Any, obj: Any) -> None:
        mesh = obj.mesh
        smoothed = _shading.auto_smooth(mesh, angle)
        if smoothed is mesh:
            return
        # One step per object, and only for an object this changed: going
        # through ``set_shading`` first and then writing the array would push
        # two, and a Ctrl+Z would land halfway.
        doc.set_mesh(obj.uid, smoothed, keep_generator=True)

    # The whole document when nothing is selected: this is the one op here that
    # means "tidy the shading", and a user with no selection means all of it.
    # Reachable since 2026-09-03 -- the op was gated on ``has_objects``, which
    # requires a selection, so this branch was a comment describing something
    # nothing could take.
    run_object_op(
        ctx, doc, one, uids=list(doc.selection) or [entry.uid for entry in doc.objects]
    )


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
    doc.clear_element_sel()


def _invert(ctx: Any, doc: Any, **_: Any) -> None:
    from ....kernels.mesh import selection

    del ctx
    selection.invert(doc)


def _selection_op(verb: Any) -> Any:
    """Wrap a ``clay.select`` verb as an op that writes the element selection.

    One wrapper for five verbs, because every one of them is the same three
    steps -- read what is selected on each object, ask the engine, write the
    answer back -- and five copies of that is five places for the "write it
    back only when it changed" rule to be forgotten.

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


def _verb_grow(mesh: Any, sel: Any, mode: str) -> Any:
    from ....kernels.mesh import select as bsel

    verts = bsel.verts_of(mesh, sel, mode)
    if not len(verts):
        return None
    return bsel.sel_from_verts(mesh, bsel.grow(mesh, verts), mode)


def _verb_shrink(mesh: Any, sel: Any, mode: str) -> Any:
    from ....kernels.mesh import select as bsel

    verts = bsel.verts_of(mesh, sel, mode)
    if not len(verts):
        return None
    return bsel.sel_from_verts(mesh, bsel.shrink(mesh, verts), mode)


def _verb_boundary(mesh: Any, sel: Any, mode: str) -> Any:
    from ....kernels.mesh import elements as el
    from ....kernels.mesh import select as bsel

    del sel
    pairs = bsel.boundary(mesh)
    if not len(pairs):
        return None
    # Reported in the mode's own currency: the border is a set of edges, and in
    # vertex mode what a user means by "select the boundary" is its vertices.
    if mode == "edge":
        return el.ElementSel(edges=pairs)
    if mode == "vertex":
        return el.ElementSel(verts=np.unique(pairs.reshape(-1)))
    return None


# ``_verts_of`` and ``_sel_from_verts`` used to live here, reaching into
# ``elements._face_corner_mask`` -- a private of another module -- from one
# level too high up. The 2026-09-10 groundwork pass moved both, unchanged, down
# into ``clay.select`` as the public ``verts_of``/``sel_from_verts``: see that
# module for the functions and their docstrings, and the three ``_verb_*``
# wrappers above for the only callers.


# --- tranche 6: UV (seams, unwrap-by-seams, pack, texel density) ------------
#
# ``dev/CLAY-PLAN.md`` tranche 6's integration half. ``kernels.mesh.uvtools``
# and ``kernels.mesh.uvunwrap`` are the kernel half; what belongs here is the
# same wiring tranche 5's own section states -- which selection or object set
# each row reads, which of its numbers becomes a ``Param``, and the refusal
# each row inherits verbatim from the kernel it calls rather than writing its
# own sentence.


def _seam_op(mark: bool) -> Callable[..., bool]:
    """Mark/Clear Seam -- edge mode's own doors onto ``Obj.seams``.

    Reads each selected object's own edge selection (``doc.element_sel_of``,
    already the canonical vertex pairs :meth:`~.document.ClayDoc.set_seams`
    wants -- see ``elements.ElementSel``'s own docstring) and folds it into,
    or out of, that object's seam set through ``set_seams`` -- one call per
    object, one step overall (``run``'s own ``collapse_since``). An object
    with edge mode active but nothing picked in it is never handed to ``one``
    at all: ``doc.selection`` in edge mode is exactly the uids with a
    non-empty ``element_sel`` (the document module's own invariant), which is
    what ``run_object_op`` iterates.
    """

    def run(ctx: Any, doc: Any, **_: Any) -> bool:
        def one(doc: Any, obj: Any) -> None:
            picked = {(int(a), int(b)) for a, b in doc.element_sel_of(obj.uid).edges.tolist()}
            if not picked:
                return
            current = set(obj.seams)
            wanted = current | picked if mark else current - picked
            doc.set_seams(obj.uid, wanted)

        return run_object_op(ctx, doc, one)

    return run


def _unwrap_seams(ctx: Any, doc: Any, **_: Any) -> bool:
    """"Unwrap (Seams)" -- LSCM by whatever seams each selected object
    carries, keeping the generator (a uv is not geometry, exactly the reason
    Box Unwrap and Smart Unwrap both already state for their own rows).

    Whole objects, not the element selection -- ``uvunwrap.unwrap_lscm``
    reads ``Obj.seams`` itself, a property of the *object*, not of whatever
    happens to be picked right now. Refuses with the kernel's own sentence
    ("A closed surface cannot be flattened with no seam...") through the
    ``OpError`` ``run_object_op`` already catches and toasts -- no refusal
    text of this row's own, because the kernel already names exactly what is
    wrong and what to do about it.
    """
    from ....kernels.mesh import uvunwrap

    def one(doc: Any, obj: Any) -> None:
        mesh = uvunwrap.unwrap_lscm(obj.mesh, obj.seams)
        doc.set_mesh(obj.uid, mesh, keep_generator=True)

    return run_object_op(ctx, doc, one)


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


def _texel_density(
    ctx: Any, doc: Any, target: float = 1024.0, texture_size: float = 2.0, **_: Any
) -> bool:
    """"Normalise Texel Density" -- ``uvtools.normalize_density`` scales every
    island so it reads *target* pixels per metre on a *texture_size*-square
    texture, keeping the generator. ``texture_size``'s choices are Bake
    Detail's own ``blender_spec.CLAY_TEXTURE_SIZES`` -- the one list of
    "texture sizes Clay offers", not a second one for this row to drift from.
    """
    from ....kernels.mesh import uvtools
    from ....kernels.rig import blender_spec

    def one(doc: Any, obj: Any) -> None:
        texture_px = blender_spec.CLAY_TEXTURE_SIZES[int(texture_size)]
        normalized = uvtools.normalize_density(obj.mesh, float(target), texture_px=texture_px)
        doc.set_mesh(obj.uid, normalized, keep_generator=True)

    return run_object_op(ctx, doc, one)


# --- tranche 7: colliders -----------------------------------------------------
#
# ``dev/CLAY-PLAN.md`` tranche 7's integration half. ``kernels.mesh.colliders``
# is the kernel half; one row per ``COLLIDER_KINDS`` entry, built by looping
# the registry rather than five hand-written ``register`` calls -- that
# module's own docstring says a sixth kind should need nothing here, and the
# only way that sentence stays true is if the row list is rebuilt from the
# dict every time :func:`_register_collider_ops` runs rather than snapshotted
# once and written out below.


def _collider_params(defaults: dict[str, Any]) -> tuple[Param, ...]:
    """One ``Param`` per keyword default a ``COLLIDER_KINDS`` fit function
    takes: a ``bool`` default is a checkbox, an ``int`` default is a whole
    number (``max_faces``'s own floor of 4 is the smallest hull --
    a tetrahedron -- :func:`~.colliders.convex_hull` can return). Sphere and
    capsule take neither and get ``params=()`` -- a bare-action row, the same
    "no dialog for a zero-parameter op" rule Box Unwrap already follows.
    """
    params: list[Param] = []
    for name, value in defaults.items():
        label = name.replace("_", " ")
        if isinstance(value, bool):
            params.append(
                Param(name, label, 1.0 if value else 0.0, 1.0, low=0.0, high=1.0, boolean=True)
            )
        elif isinstance(value, int):
            params.append(
                Param(name, label, float(value), 1.0, low=4.0, high=1024.0, integer=True)
            )
        else:
            params.append(Param(name, label, float(value), 0.01, low=0.0))
    return tuple(params)


def _collider_kwargs(defaults: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """*params* (``run``'s already-clamped floats) narrowed back to the type
    each default actually is, so a checkbox reads as ``bool`` and a whole
    number as ``int`` -- exactly what :func:`_collider_params` built the
    widget to promise, and what the fit functions' own signatures declare."""
    kwargs: dict[str, Any] = {}
    for name, default in defaults.items():
        if name not in params:
            continue
        value = params[name]
        if isinstance(default, bool):
            kwargs[name] = bool(value)
        elif isinstance(default, int):
            kwargs[name] = int(value)
        else:
            kwargs[name] = float(value)
    return kwargs


def _collider_op(kind: str) -> Callable[..., bool]:
    """Fit *kind* against every selected object's **evaluated** mesh and add
    one collider child per source through :meth:`~.document.ClayDoc.
    add_collider`, all as one step (``run``'s own ``collapse_since`` folds
    the per-object pushes, the same way every other multi-object row in this
    file folds its own per-object loop). Leaves the sources selected --
    neither ``doc.evaluated`` nor ``add_collider`` touches ``doc.selection``.

    *kind* is looked up in ``COLLIDER_KINDS`` fresh on every call rather than
    closed over as a fit function directly, so a row still calls the right
    fit if the dict it names was replaced (a monkeypatched test double) after
    this closure was built.
    """

    def run(ctx: Any, doc: Any, **params: Any) -> bool:
        from ....kernels.mesh import colliders as colliders_mod
        from ....kernels.mesh.elements import OpError

        _label, fit, defaults = colliders_mod.COLLIDER_KINDS[kind]
        kwargs = _collider_kwargs(defaults, params)

        ran = False
        for uid in list(doc.selection):
            try:
                doc.by_uid(uid)  # tolerate one deleted since the snapshot above
            except KeyError:
                continue
            try:
                mesh = doc.evaluated(uid)
                collider = fit(mesh, **kwargs)
            except OpError as error:
                toast(ctx, str(error))
                continue
            doc.add_collider(uid, collider)
            ran = True
        return ran

    return run


def _register_collider_ops() -> None:
    """Register one row per :data:`~.colliders.COLLIDER_KINDS` entry -- see
    this section's own header comment for why this is a loop and not five
    ``register`` calls.

    **The label is the registry's own, verbatim -- no "Collider" appended**,
    plus "..." exactly when the kind takes parameters, the same
    ``test_label_conventions.py::test_a_clay_op_that_opens_a_dialog_says_so``
    rule every other parameterised row in this file follows: a label ending
    "..." is the one signal a reader gets that pressing it opens a dialog
    rather than running immediately. Box, Convex Hull and Compound take one
    (``oriented``, ``max_faces`` -- see :func:`_collider_params`); Sphere and
    capsule take none and stay bare-action rows.

    The suffix is added **here, not on ``COLLIDER_KINDS`` itself** --
    ``colliders.COLLIDER_KINDS[kind][0]`` is also what
    :meth:`~.document.ClayDoc.add_collider` names the new child object with
    and what the agent surface's tool schema describes a kind by
    (``agent/schema.py``), and "..." means "opens a dialog" in neither of
    those places -- an object named "Barrel Box..." or a tool description
    ending in it would be the ellipsis leaking into a context it says nothing
    true about.

    "Convex Hull Collider" measures past the tools pane's own longest label
    ("Bake Transform") at the 190/240 dp widths
    ``test_clay_tools_panels.py`` pins, the exact incident that test's own
    docstring already names once; every ``COLLIDER_KINDS`` label -- with or
    without the "..." this adds -- is short enough alone, and ``Op.hint``
    (below) is where "this adds a collision proxy" actually gets said.
    """
    from ....kernels.mesh import colliders as colliders_mod

    for kind, (label, _fit, defaults) in colliders_mod.COLLIDER_KINDS.items():
        register(
            Op(
                name=f"collider-{kind}",
                label=f"{label}..." if defaults else label,
                modes=("object",),
                run=_collider_op(kind),
                enabled=has_objects,
                reason=_has_objects_reason,
                # First collider row only: the same "separator ahead of a
                # new group" convention Duplicate and Shade Smooth already
                # use, not something per-kind to get out of sync.
                separator_before=kind == next(iter(colliders_mod.COLLIDER_KINDS)),
                hint="Fits a collision proxy to the selected objects' "
                "evaluated meshes and adds it as a translucent, unshaded "
                "child of each source -- never in place of the source's own "
                "mesh.",
                params=_collider_params(defaults),
            )
        )


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
            modes=ELEMENT_MODES,
            run=_select_all,
            key="Ctrl+A",
        )
    )
    register(
        Op(
            name="select-none",
            label="Select None",
            modes=ELEMENT_MODES,
            run=_select_none,
            enabled=has_elements,
            reason=_has_elements_reason,
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
            modes=ELEMENT_MODES,
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
            name="select-more",
            label="Select More",
            modes=ELEMENT_MODES,
            run=_selection_op(_verb_grow),
            enabled=has_elements,
            reason=_has_elements_reason,
            key="Ctrl+=",
        )
    )
    register(
        Op(
            name="select-less",
            label="Select Less",
            modes=ELEMENT_MODES,
            run=_selection_op(_verb_shrink),
            enabled=has_elements,
            reason=_has_elements_reason,
            key="Ctrl+-",
            hint="Peels the border off the selection, leaving its middle.",
        )
    )
    register(
        Op(
            name="select-boundary",
            label="Select Boundary",
            # Vertex and edge only: a hole's border is a run of edges, and there
            # is no face on the open side of one to select.
            modes=("vertex", "edge"),
            run=_selection_op(_verb_boundary),
            key="",
            hint="Every open edge -- the border of every hole, which is what "
            "Fill Hole is about to close.",
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
            separator_before=True,
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
                separator_before=smooth,
            )
        )
    register(
        Op(
            name="shade-auto",
            label="Shade Auto...",
            modes=("object",),
            run=_shade_auto,
            # ``any_object``, not ``has_objects``: this op's own fallback is
            # "the whole document when nothing is selected", and the tighter
            # gate made that unreachable.
            enabled=any_object,
            reason=_any_object_reason,
            params=(
                Param(
                    "angle",
                    "sharp above (deg)",
                    _shading.DEFAULT_ANGLE,
                    5.0,
                    low=0.0,
                    high=180.0,
                    warn="0 makes everything flat; 180 makes everything smooth.",
                ),
            ),
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
        )
    )
    # Tranche 6: seams and the rows built on them -- see that section's own
    # header comment, just above ``_register_defaults``. ``blender_spec`` is
    # imported here, ahead of tranche 4's own import below, because
    # ``texel-density``'s texture-size choices reuse ``CLAY_TEXTURE_SIZES``
    # and this block registers first.
    from ....kernels.rig import blender_spec

    register(
        Op(
            name="mark-seam",
            label="Mark Seam",
            modes=("edge",),
            run=_seam_op(True),
            enabled=in_mode("edge"),
            reason=_in_mode_reason("edge"),
            hint="Adds the selected edges to this object's seam set -- where "
            "Unwrap (Seams) will cut.",
        )
    )
    register(
        Op(
            name="clear-seam",
            label="Clear Seam",
            modes=("edge",),
            run=_seam_op(False),
            enabled=in_mode("edge"),
            reason=_in_mode_reason("edge"),
            hint="Removes the selected edges from this object's seam set.",
        )
    )
    register(
        Op(
            # Not "Unwrap (Seams)": at 98px that string ties the tools
            # pane's own longest label ("Bake Transform") exactly, leaving no
            # margin at the 190/240 dp widths ``test_clay_tools_panels.py``
            # pins -- see that test's own docstring for the incident a
            # too-long label already caused here once.
            name="unwrap-seams",
            label="Unwrap Seams",
            modes=("object",),
            run=_unwrap_seams,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Flattens each selected object by its own marked seams "
            "(LSCM, angle-preserving). Keeps the generator -- a uv is not "
            "geometry. Refuses a closed surface with no seam to cut it open.",
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
            name="texel-density",
            label="Texel Density...",
            modes=("object",),
            run=_texel_density,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Scales every selected object's uv islands so each one "
            "reads the target pixels-per-metre on the chosen texture size. "
            "Needs a uv already -- unwrap first.",
            params=(
                Param("target", "target (px/m)", 1024.0, 64.0, low=1.0),
                Param(
                    "texture_size",
                    "texture size",
                    2.0,
                    1.0,
                    low=0.0,
                    high=float(len(blender_spec.CLAY_TEXTURE_SIZES) - 1),
                    choices=tuple(str(s) for s in blender_spec.CLAY_TEXTURE_SIZES),
                ),
            ),
        )
    )
    register(
        Op(
            name="bake",
            label="Bake Transform",
            modes=("object",),
            run=_bake,
            enabled=has_objects,
            reason=_has_objects_reason,
        )
    )
    register(
        Op(
            name="clean-mesh",
            label="Clean Up...",
            modes=("object",),
            run=_clean_mesh,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Removes degenerate and duplicate faces, merges coincident "
            "vertices and drops loose ones, then re-orients every shell "
            "outward. A clean object is left untouched -- no toast, no step.",
            params=(
                Param(
                    "distance", "merge distance (m)", 1e-5, 1e-5, low=1e-9,
                    warn="Vertices closer than this are merged into one.",
                ),
                Param("fill_holes", "fill holes", 0.0, 1.0, low=0.0, high=1.0, boolean=True),
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
            name="apply-modifiers",
            label="Apply Modifiers",
            modes=("object",),
            run=_apply_modifiers,
            enabled=has_modifier_stack,
            reason=_has_modifier_stack_reason,
            hint="Bakes every selected object's modifier stack into its base "
            "mesh, one step: what you saw is what you get, and the stack is "
            "gone.",
        )
    )
    register(
        Op(
            name="decimate",
            label="Decimate...",
            modes=("object",),
            run=_decimate,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Triangulates the selection and reduces its triangle count "
            "with gltfpack, keeping materials. Runs in the background -- the "
            "object stays editable while it works, and the result is dropped "
            "if the mesh changed before it landed.",
            params=(
                Param("ratio", "triangles kept", 0.5, 0.05, low=0.01, high=1.0),
                Param(
                    "keep_seams", "keep seams", 1.0, 1.0, low=0.0, high=1.0, boolean=True,
                ),
                Param("aggressive", "aggressive", 0.0, 1.0, low=0.0, high=1.0, boolean=True),
            ),
        )
    )
    # Tranche 4: the three Blender-backed ops, decimate's own background-op
    # shape aimed at ``pipelines.clay_blender`` instead of gltfpack -- see
    # that section's own docstring, just above ``_retopo``.
    from ....kernels.rig import blender_spec

    register(
        Op(
            name="retopo",
            label="Retopologize...",
            modes=("object",),
            run=_retopo,
            enabled=_blender_enabled,
            reason=_blender_reason,
            hint="Sends the selection's evaluated meshes to Blender for a "
            "quad retopology, replacing each object's base mesh and keeping "
            "its modifier stack. Spawns Blender -- seconds for a simple "
            "prop, minutes for something dense -- and is greyed out when "
            "Blender (the rig extra) is not installed.",
            params=(
                Param(
                    "target_faces",
                    "target faces",
                    5000.0,
                    100.0,
                    low=float(blender_spec.CLAY_TARGET_FACES_MIN),
                    high=float(blender_spec.CLAY_TARGET_FACES_MAX),
                    integer=True,
                ),
                Param("close_holes", "close holes", 0.0, 1.0, low=0.0, high=1.0, boolean=True),
                Param("seed", "seed", 0.0, 1.0, low=0.0, high=999_999.0, integer=True),
            ),
        )
    )
    register(
        Op(
            name="smart-unwrap",
            label="Smart Unwrap...",
            modes=("object",),
            run=_smart_unwrap,
            enabled=_blender_enabled,
            reason=_blender_reason,
            hint="Smart-UV-Projects the selection in Blender -- UVs only, so "
            "the generator is not frozen, the way Box Unwrap already works. "
            "Spawns Blender and is greyed out when it is not installed.",
            params=(
                Param("angle_limit", "angle limit (deg)", 66.0, 1.0, low=1.0, high=89.0),
                Param("island_margin", "island margin", 0.003, 0.001, low=0.0, high=0.5),
            ),
        )
    )
    register(
        Op(
            name="bake-detail",
            label="Bake Detail...",
            modes=("object",),
            run=_bake_detail,
            enabled=_blender_bake_enabled,
            reason=_blender_bake_reason,
            hint="Bakes every other selected, visible object onto the "
            "topmost one in the outliner -- Clay has no 'active object', so "
            "document order picks the low-poly target, the same rule "
            "Merge/Union Objects use for their own target. The target needs "
            "UVs already (Smart Unwrap first); the result replaces its "
            "material's textures, never in place. Spawns Blender and is "
            "greyed out when it is not installed.",
            params=(
                Param(
                    "texture_size",
                    "texture size",
                    2.0,
                    1.0,
                    low=0.0,
                    high=float(len(blender_spec.CLAY_TEXTURE_SIZES) - 1),
                    choices=tuple(str(s) for s in blender_spec.CLAY_TEXTURE_SIZES),
                ),
                Param("cage_extrusion", "cage extrusion (m)", 0.02, 0.005, low=0.0, high=1.0),
                Param(
                    "bake_base_color", "base colour", 1.0, 1.0, low=0.0, high=1.0, boolean=True,
                ),
                Param(
                    "bake_roughness", "roughness", 1.0, 1.0, low=0.0, high=1.0, boolean=True,
                ),
                Param("bake_normal", "normal", 1.0, 1.0, low=0.0, high=1.0, boolean=True),
            ),
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
            # The pointer at its counterpart, here rather than in the menu: this
            # dialog is the one moment the user has committed to "make these one
            # object" and can still choose which meaning of it they wanted, and
            # the moment the weld's cost -- the walls it is about to bury inside
            # the result -- has not been paid yet.
            hint=(
                "Welds the shapes and keeps the surfaces inside the overlap. For "
                "shapes that interpenetrate, Union Objects (Ctrl+Shift+M) cuts "
                "those away instead -- at the cost of the UVs and the n-gons."
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
    # Beside *Merge Objects...*, never instead of it. The two answer different
    # questions and the manual says which is which: a merge welds and keeps the
    # geometry inside the overlap, a union removes it -- and pays for that with
    # the UVs and the n-gons, which is exactly why the weld cannot simply be
    # retired in its favour. Same predicate, because "fewer than two visible" is
    # the identity for both.
    #
    # And the same key, shifted. Union spent its first release in the context
    # menu with no binding at all while the weld beside it held the merge key, so of
    # the pair the discoverable one was the one that leaves the interior walls
    # in -- users found *Merge*, got z-fighting inside the overlap, and had no
    # reason to suspect the other row existed. Shift is the right modifier for
    # it under the rule Ctrl+Shift+Z and Ctrl+Shift+I already follow here: the
    # same question, answered the other way.
    register(
        Op(
            name="union",
            label="Union Objects",
            modes=("object",),
            run=_union,
            enabled=has_two_visible,
            reason=_has_two_visible_reason,
            key="Ctrl+Shift+M",
        )
    )
    # Difference and Intersection: the rest of ``ops_boolean.KINDS``, closing
    # the 2026-09-11 audit's clay-04. Same predicate as Union -- "fewer than
    # two visible" refuses all three identically -- and the same shape of run
    # function, copied rather than shared, for the reason ``_union``'s own
    # docstring gives: only the ``ops_boolean`` call differs.
    #
    # **No key chord.** Every other bound op in this file fires from the
    # keyboard through one of two paths: ``clay_mode._registry_key`` reads
    # ``Op.key`` generically, but only for the *element* modes (vertex/edge/
    # face); every object-mode chord this registry owns today (Ctrl+M,
    # Ctrl+Shift+M, Ctrl+J, Ctrl+=/-) is instead hand-dispatched, one ``elif``
    # per letter, inside ``clay_mode._ctrl_key``. A ``key=`` string here would
    # only ever be display text -- the menu row and the shortcuts sheet would
    # both claim a binding this file cannot make live, which is worse than
    # having none. Wiring a real chord needs an edit to ``_ctrl_key`` itself,
    # a file this registry's own ownership slice does not extend to; a menu
    # row and a tools-pane button already reach every registered op with no
    # further wiring (``studio/modes/clay/ui/menu.py``'s ``_rows`` and
    # ``studio/modes/clay/ui/tools.py``'s ``_actions`` both iterate ``clay_ops.menu``),
    # so that is the complete fix and the one taken here -- and it is why
    # ``docs/manual/39-shortcuts.md``, gated bidirectionally against the
    # keyboard table, needs no new line for either op.
    register(
        Op(
            name="difference",
            label="Difference Objects",
            modes=("object",),
            run=_difference,
            enabled=has_two_visible,
            reason=_has_two_visible_reason,
            hint="Cuts every other selected object out of the first (topmost) "
            "one -- a countersink, a doorway punched through a wall. Order "
            "matters here, unlike Union or Intersect: select the block before "
            "the holes you mean to cut into it. Costs the same UVs and "
            "n-gons Union does.",
        )
    )
    register(
        Op(
            name="intersection",
            label="Intersect Objects",
            modes=("object",),
            run=_intersection,
            enabled=has_two_visible,
            reason=_has_two_visible_reason,
            hint="Keeps only the volume every selected object shares, and "
            "discards the rest -- carving one shape with the overlap of "
            "several others. Costs the same UVs and n-gons Union does.",
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
                separator_before=axis == 0,
            )
        )

    register(
        Op(
            name="array-linear",
            label="Array Linear...",
            modes=("object",),
            run=_array_linear,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Copies the whole selection, each further along one step. "
            "A negative step runs the array backwards along that axis.",
            params=(
                Param("count", "count", 3.0, 1.0, low=1.0, high=MAX_ARRAY_COUNT, integer=True),
                Param("x", "x step (m)", 1.0, 0.1, low=-1e6),
                Param("y", "y step (m)", 0.0, 0.1, low=-1e6),
                Param("z", "z step (m)", 0.0, 0.1, low=-1e6),
            ),
            separator_before=True,
        )
    )
    register(
        Op(
            name="array-radial",
            label="Array Radial...",
            modes=("object",),
            run=_array_radial,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Spins copies of the selection around the world origin, not "
            "the object's own centre -- put the hub at the origin and one "
            "spoke beside it, and reach a hub elsewhere by arraying here and "
            "moving the whole group.",
            params=(
                Param("count", "count", 3.0, 1.0, low=1.0, high=MAX_ARRAY_COUNT, integer=True),
                Param("angle", "sweep (deg)", 360.0, 5.0, low=-1e6),
                # One combo rather than three axis-named ops (mirror-x/y/z's
                # shape): mirror takes no other numbers, so the axis *is* the
                # whole op and three rows cost nothing; this op already has
                # two more numbers, and three near-identical dialogs is the
                # worse trade. A three-way choice, not a boolean -- see the
                # ``Param`` dataclass's own docstring for the defect this
                # once shared with ``place-between``'s ``fit``.
                Param("axis", "axis", 1.0, 1.0, low=0.0, high=2.0, choices=("X", "Y", "Z")),
            ),
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
            name="place-between",
            label="Place Between...",
            modes=("object",),
            run=_place_between,
            enabled=has_three_selected,
            reason=_has_three_selected_reason,
            hint="Select two anchors and the object to place between them. "
            "Document order decides which moves, not click order: the two "
            "anchors are the earliest-added of the three, and the newest "
            "one is carried to their midpoint and turned to face the line "
            "between them. 'Fit' also stretches it along its own Y so it "
            "spans the gap exactly.",
            params=(Param("fit", "fit to gap", 1.0, 1.0, low=0.0, high=1.0, boolean=True),),
        )
    )
    register(
        Op(
            name="align",
            label="Align...",
            modes=("object",),
            run=_align,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Lines up every selected object's world box -- its visible "
            "edge or middle, not its pivot -- along one axis. Two boxes of "
            "different sizes sharing a translation do not share a centre.",
            params=(
                Param("axis", "axis", 0.0, 1.0, low=0.0, high=2.0, choices=("X", "Y", "Z")),
                Param(
                    "mode", "align to", 1.0, 1.0, low=0.0, high=2.0,
                    choices=("Min", "Centre", "Max"),
                ),
            ),
            separator_before=True,
        )
    )
    register(
        Op(
            name="distribute",
            label="Distribute...",
            modes=("object",),
            run=_distribute,
            enabled=has_three_or_more_selected,
            reason=_has_three_or_more_selected_reason,
            hint="Spaces the selection evenly along one axis, equal gap for "
            "equal gap between neighbouring boxes -- the two extreme objects "
            "stay exactly where they were.",
            params=(Param("axis", "axis", 0.0, 1.0, low=0.0, high=2.0, choices=("X", "Y", "Z")),),
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
    # origin, lock. See the section above (just before ``_forget_manifold``)
    # for the shared door reasoning; what is here is only the registry rows.
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
            separator_before=True,
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
            # narrowest tested sidebar (190 dp) -- see
            # ``test_no_action_button_is_narrower_than_its_own_label``.
            label="Parent to Last",
            modes=("object",),
            run=_parent_to_last,
            enabled=has_two_or_more_selected,
            reason=_has_two_or_more_selected_reason,
            hint="Parents every other selected object onto the topmost one "
            "in the outliner, keeping world placement -- Clay has no "
            "'active object', so document order stands in for it, the same "
            "rule Merge/Union Objects use for their own target.",
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
            "connected shell, as one step: same transform, same parent, "
            "same modifier stack, copied onto every piece.",
            separator_before=True,
        )
    )
    register(
        Op(
            name="separate-material",
            label="Separate by Material",
            modes=("object",),
            run=_separate_material,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Splits each selected object into one new object per "
            "material slot it uses.",
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
            separator_before=True,
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
            name="lock",
            label="Lock",
            modes=("object",),
            run=_lock,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Refuses geometry, transform and delete on the selection "
            "until it is unlocked again. Renaming, visibility, tags and "
            "unlocking stay allowed, or a mistake made while locked could "
            "not be undone by anyone but the lock.",
            separator_before=True,
        )
    )
    register(
        Op(
            name="unlock",
            label="Unlock",
            modes=("object",),
            run=_unlock,
            enabled=has_objects,
            reason=_has_objects_reason,
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
            separator_before=True,
        )
    )
    register(
        Op(
            name="bridge",
            label="Bridge Loops",
            modes=("edge",),
            run=_element("ops_topo.bridge_edges"),
            enabled=in_mode("edge"),
            reason=_in_mode_reason("edge"),
        )
    )
    register(
        Op(
            name="inset",
            label="Inset Faces...",
            modes=("face",),
            run=_element("ops_topo.inset_faces"),
            enabled=in_mode("face"),
            reason=_in_mode_reason("face"),
            # The 2026-09-11 audit's clay-08: ``ops_topo.inset_faces`` has
            # taken a ``region`` argument -- a real, tested second mode, not a
            # variant of the default -- since before this registry existed,
            # and it was tested only by calling that function directly
            # (``tests/modes/clay/test_ops_topo.py``). ``_element`` forwards every
            # param by name to the mesh op it wraps, so declaring the toggle
            # here is the whole fix: no wrapper function needed, the way
            # ``_place_between`` needs one to turn its own boolean ``fit``
            # into a real ``bool`` before calling ``place_between`` -- a
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
    register(
        Op(
            name="bevel",
            label="Bevel Edges...",
            modes=("edge",),
            run=_element("ops_bevel.bevel_edges"),
            enabled=in_mode("edge"),
            reason=_in_mode_reason("edge"),
            params=(Param("width", "width (m)", 0.05, 0.01),),
        )
    )
    register(
        Op(
            name="loop-cut",
            label="Loop Cut...",
            modes=("edge",),
            run=_element("ops_bevel.loop_cut"),
            enabled=in_mode("edge"),
            reason=_in_mode_reason("edge"),
            params=(Param("t", "position", 0.5, 0.05, low=0.0, high=1.0),),
        )
    )
    register(
        Op(
            name="dissolve",
            label="Dissolve",
            modes=ELEMENT_MODES,
            run=_dissolve,
            enabled=has_elements,
            reason=_has_elements_reason,
            separator_before=True,
        )
    )
    # Deliberately a second name for ``dissolve`` in face mode rather than a
    # second implementation. ``ops_dissolve.dissolve_faces`` has always merged a
    # connected block of faces into one n-gon; what it lacked was a name anyone
    # would look for. "Dissolve" is the modelling word and stays, because it is
    # what the vertex and edge modes do too and splitting it would be exposing
    # the implementation -- but a user who wants to merge two faces searches for
    # "merge", finds nothing, and concludes the editor cannot do it.
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
            name="collapse",
            label="Collapse",
            modes=("edge", "face"),
            run=_element("ops_topo.collapse"),
            enabled=in_mode("edge", "face"),
            reason=_in_mode_reason("edge", "face"),
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
            name="fill-hole",
            label="Fill Hole",
            modes=("edge",),
            run=_element("ops_topo.fill_hole"),
            enabled=in_mode("edge"),
            reason=_in_mode_reason("edge"),
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
            separator_before=True,
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

    # Tranche 5 (dev/CLAY-PLAN.md): modelling breadth. Selection-taking rows
    # with nothing else to supply go through ``_element`` exactly like the
    # face/edge rows above; the four with their own wrapper are documented in
    # that wrapper's own docstring, just above ``_shade``.
    register(
        Op(
            name="bisect",
            label="Bisect...",
            modes=("face",),
            run=_bisect,
            enabled=in_mode("face"),
            reason=_in_mode_reason("face"),
            separator_before=True,
            hint="Cuts the selected faces with a plane through the object's "
            "own local origin, normal to the chosen axis. For an arbitrary "
            "plane, draw one with Knife instead.",
            params=(
                Param("axis", "axis", 0.0, 1.0, low=0.0, high=2.0, choices=("X", "Y", "Z")),
                Param(
                    "clear", "clear", 0.0, 1.0, low=0.0, high=2.0,
                    choices=("Keep Both", "Remove -", "Remove +"),
                ),
                Param("fill", "fill the cut", 0.0, 1.0, low=0.0, high=1.0, boolean=True),
            ),
        )
    )
    register(
        Op(
            name="knife",
            label="Knife",
            modes=("face",),
            run=_knife,
            enabled=in_mode("face"),
            reason=_in_mode_reason("face"),
            hint="Draws a line across the selected faces in the viewport; "
            "the cut follows the drag line and the view direction. With "
            "faces selected on more than one object, every one is cut by "
            "the same plane. Firing this row arms the gesture -- press, "
            "drag a line and release to cut, or Esc or a right-click to "
            "cancel.",
        )
    )
    register(
        Op(
            name="edge-slide",
            label="Edge Slide...",
            modes=("edge",),
            run=_element("ops_model.edge_slide"),
            enabled=in_mode("edge"),
            reason=_in_mode_reason("edge"),
            hint="Slides the selected edge loop along its own two rails, -1 "
            "to +1 between them; 0 leaves it where it is.",
            params=(Param("t", "position", 0.0, 0.05, low=-1.0, high=1.0),),
        )
    )
    register(
        Op(
            name="vertex-slide",
            label="Vertex Slide...",
            modes=("vertex",),
            run=_element("ops_model.vertex_slide"),
            enabled=in_mode("vertex"),
            reason=_in_mode_reason("vertex"),
            hint="Slides the selected vertices toward their nearest-index "
            "neighbour; 1 reaches it exactly, and a value outside 0..1 "
            "overshoots past it.",
            params=(Param("t", "amount", 0.0, 0.05, low=-1e6),),
        )
    )
    register(
        Op(
            name="rip",
            label="Rip",
            modes=("edge",),
            run=_element("ops_model.rip"),
            enabled=in_mode("edge"),
            reason=_in_mode_reason("edge"),
            key="V",
            hint="Splits every vertex the selected edges touch, so the "
            "faces on each side stop sharing it. Refuses a boundary edge -- "
            "there is only one face there, nothing to separate.",
        )
    )
    register(
        Op(
            name="poke",
            label="Poke Faces...",
            modes=("face",),
            run=_element("ops_model.poke"),
            enabled=in_mode("face"),
            reason=_in_mode_reason("face"),
            hint="Fans each selected face into triangles around a new "
            "centre vertex, pushed along the face normal by the offset -- "
            "zero changes topology only, the same 'extrude at zero, then "
            "drag' default Extrude uses.",
            params=(Param("offset", "offset (m)", 0.0, 0.01, low=-1e6),),
        )
    )
    register(
        Op(
            name="triangulate",
            label="Triangulate Faces",
            modes=("face",),
            run=_element("ops_model.triangulate_faces"),
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
            name="tris-to-quads",
            label="Tris to Quads...",
            modes=("face",),
            run=_element("ops_model.tris_to_quads"),
            enabled=in_mode("face"),
            reason=_in_mode_reason("face"),
            hint="Greedily joins adjacent selected triangle pairs into "
            "quads wherever the dihedral angle between them is within the "
            "limit and the merged quad stays convex.",
            params=(Param("max_angle", "max angle (deg)", 40.0, 1.0, low=0.0, high=180.0),),
        )
    )
    register(
        Op(
            name="grid-fill",
            label="Grid Fill...",
            # ``ops_model.grid_fill`` reads ``sel.edges`` -- the selected
            # boundary loop -- exactly as Fill Hole above does, not
            # ``sel.faces``; edge mode is what puts anything in that
            # selection for it to read, so this sits with edge mode's rows
            # rather than face mode's despite the "fill" in its name.
            modes=("edge",),
            run=_element("ops_model.grid_fill"),
            enabled=in_mode("edge"),
            reason=_in_mode_reason("edge"),
            hint="Fills the selected closed boundary loop with a grid of "
            "quads, span columns wide -- the loop needs an even vertex "
            "count to split evenly into two matching sides.",
            params=(Param("span", "span", 1.0, 1.0, low=1.0, high=256.0, integer=True),),
        )
    )
    register(
        Op(
            name="spin",
            label="Spin...",
            modes=("edge",),
            run=_spin,
            enabled=in_mode("edge"),
            reason=_in_mode_reason("edge"),
            hint="Lathes the selected edge profile (a chain or a loop) "
            "around the object's own local origin. A full turn closes the "
            "ring; any other sweep leaves both ends open.",
            params=(
                Param("axis", "axis", 1.0, 1.0, low=0.0, high=2.0, choices=("X", "Y", "Z")),
                Param("angle", "sweep (deg)", 360.0, 5.0, low=-1e6),
                Param("steps", "steps", 8.0, 1.0, low=1.0, high=256.0, integer=True),
            ),
        )
    )
    register(
        Op(
            name="screw",
            label="Screw...",
            modes=("edge",),
            run=_screw,
            enabled=in_mode("edge"),
            reason=_in_mode_reason("edge"),
            hint="Spin's own lathe, plus a per-step lift along the axis -- "
            "a helix, and the seam never closes even at a full turn.",
            params=(
                Param("axis", "axis", 1.0, 1.0, low=0.0, high=2.0, choices=("X", "Y", "Z")),
                Param("angle", "sweep (deg)", 360.0, 5.0, low=-1e6),
                Param("steps", "steps", 8.0, 1.0, low=1.0, high=256.0, integer=True),
                Param("height", "height (m)", 1.0, 0.05, low=-1e6),
            ),
        )
    )
    register(
        Op(
            name="symmetrize",
            label="Symmetrize...",
            modes=("object",),
            run=_symmetrize,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Deletes the chosen half of each selected object across "
            "its own local plane and mirrors what remains to rebuild the "
            "other side -- ignores the element selection, like Smooth.",
            params=(
                Param("axis", "axis", 1.0, 1.0, low=0.0, high=2.0, choices=("X", "Y", "Z")),
                Param("direction", "keep side", 1.0, 1.0, low=0.0, high=1.0, choices=("-", "+")),
            ),
        )
    )
    # Tranche 7: one row per ``colliders.COLLIDER_KINDS`` entry -- see
    # ``_register_collider_ops``'s own docstring, just above.
    _register_collider_ops()

    register(
        Op(
            name="smooth",
            # **The algorithm's name is a hint, not a label.** "Smooth
            # (Catmull-Clark)..." was the longest string in the actions grid by
            # a wide margin, and a grid sized to fit it is a grid one column
            # wide -- thirteen full-width buttons stacked down a 300 dp
            # sidebar. Sized to anything narrower, imgui drew it straight past
            # its frame and the child clipped the closing bracket off. Nobody
            # picks this op *because* it is Catmull-Clark; they pick it because
            # they want the shape rounded, and the surface that answers "which
            # smoothing is this" is the tooltip, which every op in this grid
            # already carries and this one had left empty.
            label="Smooth...",
            hint=(
                "Rounds the shape by subdividing it (Catmull-Clark), so the "
                "silhouette moves. 'Subdivide' splits the same faces without "
                "changing it, and 'Shade Smooth' changes no geometry at all."
            ),
            modes=ALL_MODES,
            run=_smooth,
            enabled=lambda doc: bool(doc.selection),
            reason=_selection_reason,
            params=(
                Param(
                    "levels",
                    "levels",
                    1.0,
                    1.0,
                    low=1.0,
                    high=4.0,
                    integer=True,
                    warn="Each level multiplies the face count by four.",
                ),
            ),
        )
    )
    register(
        Op(
            name="frame",
            label="Frame Selection",
            modes=ALL_MODES,
            run=_frame,
            separator_before=True,
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
            separator_before=True,
        )
    )


_register_defaults()
