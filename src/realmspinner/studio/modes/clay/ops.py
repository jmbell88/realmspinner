"""One registry of everything Clay can *do*, and no imgui anywhere in it.

Several surfaces invoke Clay's operations -- the right-click context menu, the
Inspector's cards, the Ctrl+K command palette, ``clay_mode.handle_key`` and the
agent's ``clay_op`` -- and before this module they each had their own list.
Several lists means several answers to "is Extrude available in vertex mode", and
the interesting one is always the one nobody updated: a key that fires an op the
menu greys out, or a menu row for an op the key path never learned about. (The
2026-10-09 second audit's clay-110: this paragraph still named a pane of buttons
that the 2026-10 Clay UI rebuild deleted.)

So there is one list. :data:`OPS` is the whole of what is invocable, each entry
carrying the modes it applies to, whether it is enabled right now and the key
that fires it; how it groups in a menu is :mod:`.menutree`'s table. The menu and
the Inspector render it, the palette lists it, and the key handler looks up by
key -- and none of them decides anything.

**Nothing here imports imgui**, which is what keeps the registry testable: an
``Op``'s ``enabled`` predicate is a function of a document, so "Flip Normals is
greyed out with an edge selected" is a plain assertion rather than a screenshot.
The panes that draw a parameter popup (``studio/modes/clay/ui/panes/menu.py``,
the Inspector's cards) and the key layer that asks for one by state
(``mode.py``'s ``open_op_popup``) are what know a popup exists.

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
    "outliner_order",
    "reason_for",
    "register",
    "run",
    "run_mesh_op",
    "run_object_op",
]

# The element modes an op can appear in. "object" is the fourth and is not an
# element mode; ops that name it are the object-level ones (duplicate, mirror,
# group, origin) that were once hardcoded in a pane's button grid. (The 2026-10-07
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
    no other shape are the tell: the *label* was doing the widget's job
    because the field couldn't. None is left -- ``mirror-copy``'s ``axis``,
    once labelled "axis (0=X, 1=Y, 2=Z)", is ``choices=("X", "Y", "Z")``
    now (the 2026-10-07 audit's clay-44 found this paragraph still calling
    it the one that was left). ``default`` stays a ``float`` and :func:`run` still
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
    refuse_below: bool = False
    """Refuse a value under ``low`` with a sentence instead of clamping it up to it.

    For a floor that is a *meaning* rather than a comfort: Snap to Grid's step of
    0 was a refusal ("a grid of no size is not a grid", the 2026-10-07 audit's
    clay-37) before the 2026-10-09 second audit's clay-108 gave the field a
    positive floor, and a clamp would have turned that refusal into a quiet snap
    onto a different grid than the one asked for.
    """

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
    menu, the Inspector and the Delete button with nothing anywhere naming the
    gate that refused it -- ``hint`` describes what an op *does*, not why it is
    currently unavailable. Called only through :func:`reason_for`, which is the
    one place that decides *whether* to call it, so a ``reason`` never needs to
    re-check ``enabled`` itself and cannot disagree with it about that.
    """
    check: Callable[[dict[str, Any]], None] | None = None
    """A refusal the *numbers* alone decide, raised as :class:`OpError` or ``None``.

    Called by :func:`run` on the resolved (defaulted, clamped) values, and by the
    viewport's op-drag preview on the value under the pointer, so the picture a
    drag shows and the press that commits it cannot disagree about whether the
    pair of numbers is legal. ``Param``'s per-field ``low``/``high`` cannot say
    "these two fields are not both zero" (the 2026-10-09 second audit's clay-64).
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


def _toast_refused(ctx: Any, uid: int, message: str) -> None:
    """:func:`toast` for a refusal that belongs to one object, naming it if asked.

    The 2026-10-11 audit's clay-19 left the agent's ``errors`` list as anonymous
    sentences: a multi-object press refused for one object and run for another
    could not say *which*. ``refused(uid, message)`` is an optional hook on
    ``ctx`` -- the agent's sandboxed proxy implements it, the app's ``Ctx`` and
    every test double do not, and a missing hook is simply the old toast.
    """
    toast(ctx, message)
    refused = getattr(ctx, "refused", None)
    if callable(refused):
        refused(uid, message)


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
        if param.refuse_below and value < param.low:
            raise OpError(f"{param.label} must be at least {param.low:g}.")
        value = min(max(value, param.low), param.high)
        values[param.name] = int(value) if param.stores_int else value
    return values


def kernel_func(dotted: str) -> Callable[..., Any]:
    """The ``kernels.mesh.ops_*`` function a :class:`DragSpec` names."""
    import importlib

    module, func = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(f"realmspinner.kernels.mesh.{module}"), func)


_GESTURE_SENTENCE = "Finish or cancel the drag first."

#: Held in a constant, not written at the ``toast(ctx, ...)`` call: the
#: toast-level scan in ``tests/test_ux_todo_fixes.py`` reads a *string literal* in
#: a ``toast``'s second argument as a level, and this module's own ``toast`` takes
#: a message there (the 2026-10-09 second audit's round-two repair).
GROUP_HAS_NO_GEOMETRY = "A group has no geometry of its own; select its shapes."


def gesture_live(ctx: Any) -> bool:
    """Whether a live gesture is writing into the document right now: a 3-D
    G/R/S, op or gizmo drag, or the UV pane's armed live rotate/scale.

    The same question ``mode._gesture_live`` answers for the key layer, asked of a
    ``ctx`` alone because ``run`` and ``recent_op.adjust`` have no tab. Duck-typed
    (this module imports no ``ui/``): the agent's sandboxed ``_OpCtx`` has no
    ``clay_view`` and a ``None`` state, so it reads as "no gesture" -- an agent
    drives its own tab, never the one under the user's pointer. ``is True`` rather
    than truthiness, so a double that answers every attribute never reads as a drag.
    """
    if getattr(getattr(ctx, "clay_view", None), "dragging", False) is True:
        return True
    clay = getattr(getattr(ctx, "state", None), "clay", None)
    uv_view = getattr(getattr(clay, "active", None), "uv_view", None)
    return getattr(uv_view, "drag_mode", "") in ("rotate", "scale")


def run(ctx: Any, doc: Any, op: Op, *, during_gesture: bool = False, **params: Any) -> bool:
    """Invoke an op, turning a refusal into a toast. -> whether it ran.

    Missing parameters fall back to their declared defaults, so a caller that
    has no remembered values -- the key path, a test -- gets the same result the
    popup would have produced with the fields untouched.

    **Declared parameters are clamped here.** The popup clamps its live fields
    too (``studio/modes/clay/ui/panes/menu.py``), but that is a UX affordance on one surface --
    the key path, the Inspector, the palette, the agent and every test call arrive
    with whatever the caller had remembered, and a subdivision at ``levels=99`` is
    not a refusal an op should have to write for itself. ``run`` is the choke point
    every surface funnels through, so the range a ``Param`` declares is enforced
    once, where it cannot be bypassed.

    **A live G/R/S, gizmo drag or armed UV rotate/scale shuts this door** (the
    2026-10-09 second audit's clay-61). The first fix gated ``mode.fire_op`` only;
    the right-click row, the parameter popup's Apply, the Inspector's Apply, the
    Last action card, starter-colour paint and the swatch Assign all reach ``run``
    directly, and an op that lands mid-gesture is reverted by the gesture's commit
    against the history head it started from -- a user step silently lost.
    ``during_gesture=True`` is the key layer's one opt-out: a Ctrl chord that is
    pinned to keep working under a drag (Ctrl+J, Ctrl+M) says so; every other
    caller gets the refusal.
    """
    from ....kernels.mesh.elements import OpError

    if not during_gesture and gesture_live(ctx):
        toast(ctx, _GESTURE_SENTENCE)
        return False
    # The 2026-10-07 audit's clay-10: only ``enabled`` was consulted, so the
    # agent's ``clay_op`` (which has no menu to hide a row) ran Join, Group or
    # Mirror Copy while the document was in an element mode -- leaving
    # ``selection`` holding objects with no ``element_sel`` behind them, the
    # state ``Select None`` then greys out of and ``Delete`` answers ``ran``
    # for with nothing deleted. ``Op.modes`` is the declared set, so it is the
    # gate, here, for every caller (menu, key, palette, Inspector, agent) at once.
    if doc.element_mode not in op.modes:
        toast(ctx, _wrong_mode_reason(op))
        return False
    if not op.enabled(doc):
        return False
    try:
        values = resolve_params(op, params)
        if op.check is not None:
            op.check(values)
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
    # An op that is another op's run under a different button (Repeat Last) has
    # already had its step named by the nested ``run``; naming it again here
    # overwrote "Inset Faces" with "Repeat Last" (the 2026-10-10 audit's
    # clay-54), so the history panel and the Ctrl+Z toast said nothing about
    # what was undone.
    _one_step(doc, op, mark, head, relabel=op.name not in _NAMES_ITS_OWN_STEP)
    # What the Last action card and Repeat Last read. Only a parameterised op that
    # started in an element mode and actually pushed a step: a bare action has
    # nothing to adjust, an object-level op is a different selection model, and
    # an op that pushed nothing (a no-op) has no step for the card to be live
    # against.
    if op.params and before[0] != "object" and doc.history.head != head:
        _recent.record(doc, op.name, values, before)
    return True


#: The stored mode names as the header's switch spells them
#: (``ui/panes/tools.py``'s ``MODE_BUTTONS``; a test holds the two equal -- this
#: module sits below ``ui/`` and cannot import it). The 2026-10-11 audit's clay-43:
#: every greyed reason said "Switch to object mode first" and the right-click
#: menu's heading printed "vertex mode", words the interface never uses -- the
#: switch reads Shape, Points, Edges, Faces.
MODE_WORDS = {"object": "Shape", "vertex": "Points", "edge": "Edges", "face": "Faces"}


def mode_word(mode: str) -> str:
    """A stored mode name in the words the header's mode switch uses."""
    return MODE_WORDS.get(mode, mode)


def _mode_list(modes: Iterable[str]) -> str:
    """``("vertex", "edge", "face")`` -> "Points, Edges or Faces"."""
    names = [mode_word(mode) for mode in modes]
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} or {names[-1]}"


def _wrong_mode_reason(op: Op) -> str:
    """The sentence for an op asked to run in a mode it does not declare."""
    return f"Switch to {_mode_list(op.modes)} mode first."


wrong_mode_reason = _wrong_mode_reason
"""The same sentence, for a surface that lists an op it cannot run in this mode
(the command palette greys it with this rather than with ``Op.reason``, which
only answers for a mode the op *does* declare)."""


#: Ops whose ``run`` is another registered op's :func:`run`, which has already
#: folded and named the step it pushed. :func:`_one_step` folds for them but keeps
#: the name the nested run gave.
_NAMES_ITS_OWN_STEP = frozenset({"repeat-last"})


def _one_step(doc: Any, op: Op, mark: int, head: int, *, relabel: bool = True) -> None:
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
    if relabel and top is not None and history.head != head:
        top.label = op.label.rstrip(".")


#: The metre-labelled distances each mesh kernel takes, by the dotted name
#: ``DragSpec.kernel`` and ``_element`` already share. The 2026-10-09 second
#: audit's clay-55: Inset, Weld and Bevel handed their field's number straight to
#: a kernel that works in the mesh's *local* units, so "0.100 m" slid a rim 0.4 m
#: on an object scaled 4x, and a 0.02 m weld welded nothing on a 1 cm box and
#: answered ``ran``. Only Merge Objects converted (``ops._local_eps``). A name in
#: this table is converted by :func:`run_mesh_op` and by the op-drag preview
#: (:func:`localise`), so the dialog, the Inspector, the agent, the drag preview
#: and the commit agree.
#:
#: **Only a param an op actually declares belongs here.** The 2026-10-10 audit's
#: clay-53: this table (and, copied from it, the manual) named an
#: "Extrude-offset" field. ``ops_topo.extrude_faces`` takes an ``offset``, but
#: the Extrude row has never declared one, so the entry was unreachable and the
#: manual described a control nobody could see.
#: ``test_every_metre_param_named_in_metre_params_is_declared_by_an_op`` holds it.
METRE_PARAMS: dict[str, tuple[str, ...]] = {
    "ops_topo.inset_faces": ("thickness", "depth"),
    "ops_topo.weld": ("eps",),
    "ops_bevel.bevel_edges": ("width",),
}


def world_scale(doc: Any, uid: int) -> float:
    """The uniform scale that stands for *uid*'s world scale: the geometric mean
    of its three axes, ``|det(world 3x3)| ** (1/3)``.

    A distance typed in metres has to become the mesh's local units, and an
    object scaled ``(8, 1, 1)`` has no single answer; the geometric mean is the
    scale that preserves volume, so a distance that is right on average is right
    exactly on a uniformly scaled object (the common case) and a fair compromise
    on a stretched one. The world matrix is used, not the object's own ``scale``,
    because a parented object's TRS is relative to its parent.

    A degenerate scale (``<= 1e-9``: an axis typed down to zero) has no inverse
    and no meaning for "metres", so it is a refusal naming the cause.
    """
    from ....kernels.mesh.elements import OpError

    matrix = np.asarray(doc.world_matrix(uid), dtype="f8")[:3, :3]
    scale = abs(float(np.linalg.det(matrix))) ** (1.0 / 3.0)
    if not math.isfinite(scale) or scale <= 1e-9:
        raise OpError(
            f"{doc.by_uid(uid).name!r} has a zero scale, so a distance in metres "
            "means nothing on it; give it a size first."
        )
    return scale


def localise(doc: Any, uid: int, kernel: str, params: dict[str, Any]) -> dict[str, Any]:
    """*params* with *kernel*'s metre-labelled distances divided by *uid*'s world
    scale -- the mesh's own units, which is what the kernel measures in.

    The one helper :func:`run_mesh_op` and the viewport's op-drag preview both
    call, so what the pointer shows is what Enter commits. A kernel not in
    :data:`METRE_PARAMS` takes its numbers as they are.
    """
    names = [name for name in METRE_PARAMS.get(kernel, ()) if name in params]
    if not names:
        return params
    scale = world_scale(doc, uid)
    return {**params, **{name: float(params[name]) / scale for name in names}}


def run_mesh_op(
    ctx: Any,
    doc: Any,
    func: Callable[..., Any],
    /,
    *,
    kernel: str = "",
    **params: Any,
) -> bool:
    """Apply a ``(mesh, sel) -> (mesh, sel)`` op to every object with a selection.

    The loop, the refusal handling and the generator freeze all live here, so an
    op is registered by naming its function rather than by writing this out
    again -- and a refusal on one object does not abandon the others, which is
    what a user selecting faces across two objects means by pressing the button
    once.

    **Two phases.** Every object's new mesh is computed first and nothing is
    written; only then is the result checked against what the document may hold,
    and only then are the ``set_mesh`` calls made. The 2026-10-09 second audit's
    clay-59: ``set_mesh`` refuses a mesh that takes the document past
    ``MAX_TRIANGLES``, the loop did not anticipate it, and a run that crossed the
    ceiling part-way left the earlier objects rewritten, toasted "refused" and
    kept an undo step. Now a refused run changes no object and pushes no step.

    **Distances are world metres** (clay-55): *kernel* names the mesh kernel
    (``"ops_topo.inset_faces"``) so :func:`localise` can divide its metre-labelled
    numbers by each object's own world scale, per object.
    """
    from ....kernels.mesh import document as bd
    from ....kernels.mesh import selection
    from ....kernels.mesh.elements import OpError

    built: list[tuple[int, Any, Any, Any]] = []
    for uid in list(doc.element_sel):
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            continue
        # The 2026-10-07 audit's clay-11: an object hidden after it was picked
        # keeps its element selection, and this loop rewrote it -- 8 vertices
        # to 32 on a box the user could not see -- while the op drag's preview
        # and every other element door (``_element_pickable``) skipped it, so
        # the preview showed one result and the commit made another.
        if not selection._element_pickable(obj):
            continue
        try:
            local = localise(doc, uid, kernel, params)
            mesh, sel = func(obj.mesh, doc.element_sel_of(uid), **local)
            # A mesh that is not a valid CSR and a position past float32 are
            # ``set_mesh``'s refusals too; taken here, as this object's own, so
            # neither can surface half way through phase two. The 2026-10-10
            # audit's clay-22: only the second was taken, so a later object's
            # invalid result raised from ``set_mesh`` after the earlier objects
            # had been rewritten -- ``run`` answered False over a document that
            # had changed and carried a step.
            doc._refuse_invalid(obj.name, mesh)
            doc._refuse_unstorable(obj.name, mesh)
        except OpError as error:
            _toast_refused(ctx, uid, str(error))
            continue
        built.append((uid, obj, mesh, sel))
    if not built:
        return False
    # Only a run that adds triangles can cross the ceiling; one that shrinks or
    # holds a document somehow already over the line stays possible, as in
    # ``ClayDoc._refuse_past_triangle_ceiling``.
    if sum(bd._triangles_of(m) for _, _, m, _ in built) > sum(
        bd._triangles_of(o.mesh) for _, o, _, _ in built
    ):
        try:
            doc._refuse_past_ceilings(
                [bd.Obj(uid=uid, name=obj.name, mesh=mesh) for uid, obj, mesh, _ in built],
                [obj for _, obj, _, _ in built],
            )
        except OpError as error:
            toast(ctx, str(error))
            return False
    # Shrinking writes first. ``set_mesh`` judges each growing write against the
    # document as it stands *at that moment*, so an object that grows before
    # another one shrinks can trip the triangle ceiling for a run whose end state
    # is inside it -- refused half way, after the earlier objects were written
    # (the 2026-10-10 audit's clay-22, the near-ceiling half). Ordered by
    # triangle delta, the running total only ever climbs on the last writes, and
    # the simulation below replays exactly the check ``set_mesh`` will make, so
    # a refusal here is a refusal of the whole run and not of its tail.
    built.sort(key=lambda item: bd._triangles_of(item[2]) - bd._triangles_of(item[1].mesh))
    from ....kernels.mesh.glbimport import MAX_TRIANGLES

    running = doc._triangle_total()
    for _, obj, mesh, _ in built:
        now, was = bd._triangles_of(mesh), bd._triangles_of(obj.mesh)
        if now > was and running - was + now > MAX_TRIANGLES:
            toast(
                ctx,
                f"That would make {running - was + now:,} triangles, past the "
                f"{MAX_TRIANGLES:,} a Clay document can hold and save.",
            )
            return False
        running += now - was
    for uid, _, mesh, sel in built:
        doc.set_mesh(uid, mesh, select=sel)
    return True


def refuse_if_no_geometry(ctx: Any, doc: Any, uids: Iterable[int]) -> bool:
    """Toast and answer ``True`` when *uids* name objects and not one has a vertex.

    The 2026-10-09 second audit's clay-62: with a group (an empty that has
    children) selected, Drop to Ground, Origin to Bounds/Base, Mirror, Box Unwrap,
    Recalculate Normals and Shade Smooth/Flat were enabled, ran, pushed no step and
    said nothing -- they measure or rewrite a mesh, and a group has none. The
    truthful sentence is cheaper than inventing a descendants-box meaning for ten
    ops. Only when *every* object lacks geometry: a mixed selection still acts on
    its shapes, and the group in it is skipped as before.
    """
    # One uid map for the batch, not a ``by_uid`` scan per uid (the 2026-10-10
    # audit's clay-23).
    by_uid = doc._uid_map()
    objs = [by_uid[uid] for uid in uids if uid in by_uid]
    if objs and not any(len(obj.mesh.positions) for obj in objs):
        toast(ctx, GROUP_HAS_NO_GEOMETRY)
        return True
    return False


def run_object_op(
    ctx: Any,
    doc: Any,
    func: Callable[[Any, Any], Any],
    /,
    *,
    uids: Iterable[int] | None = None,
    needs_geometry: bool = False,
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

    **``func`` may answer ``False`` to say it changed nothing** -- the 2026-10-07
    audit's clay-41: Clear Parent on roots, Recalculate Normals on a primitive,
    a Snap onto a grid the object already sat on and Origin to Bounds on an
    empty all returned ``True`` from ``run`` with no step pushed, and an agent
    read ``ran: true`` for a press that did nothing. Every ``ClayDoc`` door
    these closures end in already answers "did anything change", so a closure
    returns that. Any other return (``None``, ``True``) still counts as ran, so
    a closure that has no way to be a no-op needs no change.

    **``needs_geometry``** marks an op that measures or rewrites a mesh: when
    every object in the batch is a group's empty, it refuses with a sentence
    (:func:`refuse_if_no_geometry`) instead of running silently over nothing.
    """
    from ....kernels.mesh.elements import OpError

    ran = False
    batch = list(doc.selection if uids is None else uids)
    if needs_geometry and refuse_if_no_geometry(ctx, doc, batch):
        return False
    for uid in batch:
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            continue
        try:
            changed = func(doc, obj)
        except OpError as error:
            _toast_refused(ctx, uid, str(error))
            continue
        if changed is not False:
            ran = True
    return ran


def outliner_order(doc: Any) -> list[int]:
    """Every object's uid in the order the outliner draws its rows: roots first
    in document order, each followed by its whole subtree depth-first.

    The 2026-10-07 audit's clay-08: Merge Objects and Parent to Last Selected
    both say "the topmost selected object in the outliner" (the hint, the
    manual), and both read ``doc.objects`` order instead -- which differs from
    what the pane draws as soon as a group exists, because ``Group`` is added
    *last* but drawn as a root above its children. Merging A (in Group) with Q
    kept A while the outliner showed Q above it; Parent to Last on Group + A
    refused a cycle the user could not see. This reproduces
    ``ui/panes/outliner.py::_tree_rows``'s walk without importing it (this
    module draws nothing), and
    ``test_outliner_order_matches_the_outliner_pane_row_for_row`` holds the two
    equal so they cannot drift.

    An object the walk cannot reach (its parent chain never meets a root --
    only a hand-built, cyclic document) is appended in document order rather
    than dropped, so a selected one still has a rank.
    """
    children: dict[Any, list[int]] = {}
    for obj in doc.objects:
        children.setdefault(obj.parent, []).append(obj.uid)
    order: list[int] = []
    seen: set[int] = set()
    stack = list(reversed(children.get(None, [])))
    while stack:
        uid = stack.pop()
        if uid in seen:  # cycle guard, as ``ClayDoc.ancestors``
            continue
        seen.add(uid)
        order.append(uid)
        stack.extend(reversed(children.get(uid, [])))
    order.extend(obj.uid for obj in doc.objects if obj.uid not in seen)
    return order


def _in_outliner_order(doc: Any, uids: Iterable[int]) -> list[int]:
    """*uids* (any order) sorted topmost-first by :func:`outliner_order`."""
    rank = {uid: i for i, uid in enumerate(outliner_order(doc))}
    return sorted(uids, key=lambda uid: rank.get(uid, len(rank)))


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


def has_visible_selected(doc: Any) -> bool:
    """At least one selected object the user can see -- Duplicate's gate, and
    Delete's in object mode. A hidden object is left untouched by both (it is
    not rendered, exported or picked), so a selection of only those is no
    selection (the 2026-10-10 audit's clay-51)."""
    return any(obj.visible for obj in doc.objects if obj.uid in doc.selection)


def _hidden_only_sentence(doc: Any) -> str:
    return "" if has_visible_selected(doc) else "The selected objects are hidden; show one first."


def _has_visible_selected_reason(doc: Any) -> str:
    if not doc.selection:
        return "Select an object first."
    return _hidden_only_sentence(doc)


def can_delete(doc: Any) -> bool:
    """Delete's gate: a visible object in object mode, a pick in an element mode."""
    if doc.element_mode == "object":
        return has_visible_selected(doc)
    return has_elements(doc)


def _can_delete_reason(doc: Any) -> str:
    if doc.element_mode == "object" and doc.selection:
        return _hidden_only_sentence(doc)
    return _has_any_selection_reason(doc)


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


def _element_only_reason(doc: Any) -> str:
    """The reason for an op that works in vertex, edge or face mode.

    The 2026-10-07 audit's clay-42: Extrude, Select Linked and Origin to
    Selection answered "Select something in the viewport first." in object
    mode, where an object *is* selected and the viewport selection is exactly
    what the user did -- the thing missing is the mode. Mode first, then the
    selection, the order :func:`_in_mode_reason` already uses.
    """
    if doc.element_mode == "object":
        return f"Switch to {_mode_list(ELEMENT_MODES)} mode first."
    return _has_elements_reason(doc)


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


def _has_two_or_more_selected_reason(doc: Any) -> str:
    n = len(doc.selection)
    return "" if n >= 2 else f"Select at least two objects first -- {n} selected now."


def _in_mode_reason(*modes: str) -> Callable[[Any], str]:
    """A reason matching :func:`in_mode`'s own two gates, in the same order --
    the mode first, then the selection -- so the sentence shown can never
    disagree with the row it is explaining."""

    label = _mode_list(modes)

    def reason(doc: Any) -> str:
        if doc.element_mode not in modes:
            return f"Switch to {label} mode first."
        if not doc.element_sel:
            return "Select something first."
        return ""

    return reason


# --- the object-level ops (once a pane's hardcoded buttons) -----------------


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
        return run_mesh_op(ctx, doc, target, kernel=dotted, **params)

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
        "vertex": (ops_topo.extrude_verts, "ops_topo.extrude_verts"),
        "edge": (ops_topo.extrude_edges, "ops_topo.extrude_edges"),
        "face": (ops_topo.extrude_faces, "ops_topo.extrude_faces"),
    }.get(doc.element_mode)
    return which is not None and run_mesh_op(ctx, doc, which[0], kernel=which[1], **params)


#: The largest weld distance Weld and Merge Objects accept, in metres. The
#: 2026-10-07 audit's clay-43: both kept ``Param``'s 1e6 default ``high``, and a
#: typed ``1e6`` (or a stray extra digit) welded every vertex of the mesh into
#: one point and reported success. Fifty metres is far past any distance a
#: user means for a *weld* -- a model is a few metres to a few tens across and a
#: weld closes seams of millimetres -- so past it the number is a typo;
#: :func:`_collapse_sentence` guards the in-range distances that are still too
#: big for the object in front of them.
MAX_WELD_DISTANCE = 50.0


def _collapse_sentence(distance: float, what: str) -> str:
    return (
        f"A weld distance of {float(distance):g} m would collapse every face of {what}; "
        "use a smaller distance."
    )


def _weld(ctx: Any, doc: Any, **params: Any) -> bool:
    """Weld vertices closer than a distance -- ``ops_topo.weld`` per object, but
    refusing a distance that leaves an object with no faces at all.

    The refusal is per object (``run_mesh_op`` toasts it and carries on to the
    next), so one object too small for the distance does not stop the others.
    """
    from ....kernels.mesh import mesh as bm
    from ....kernels.mesh import ops_topo
    from ....kernels.mesh.elements import OpError

    distance = float(params.get("eps", 1e-4))

    def guarded(mesh: Any, sel: Any, **kw: Any) -> Any:
        out = ops_topo.weld(mesh, sel, **kw)
        if bm.face_count(mesh) and not bm.face_count(out[0]):
            raise OpError(_collapse_sentence(distance, "this object"))
        return out

    return run_mesh_op(ctx, doc, guarded, kernel="ops_topo.weld", **params)


_weld.kernel = "ops_topo.weld"  # type: ignore[attr-defined]


def _inset_check(values: dict[str, Any]) -> None:
    """Inset needs a thickness or a depth: both zero builds a rim of zero-area faces.

    The 2026-10-09 second audit's clay-64: thickness 0 and depth 0 reported
    success and left a ring of degenerate faces, and ``I`` then Enter -- the drag
    starts at 0 -- committed exactly that. A depth-only inset (thickness 0, a
    non-zero depth) is a legitimate extrusion-without-growth and stays legal.
    """
    from ....kernels.mesh.elements import OpError

    if float(values.get("thickness", 0.0)) == 0.0 and float(values.get("depth", 0.0)) == 0.0:
        raise OpError(
            "Inset needs a thickness or a depth: both are 0, which would leave "
            "zero-area faces. Set the thickness above 0, or give it a depth."
        )


def _with_descendants_in(doc: Any, uids: Iterable[int]) -> set[int]:
    """*uids* and every object under any of them, from one parent -> children map.

    The 2026-10-10 audit's clay-23 wrote this one-pass walk here because
    ``selection.with_descendants`` called ``doc.descendants`` per uid (quadratic
    at 4,000 objects); the 2026-10-11 audit's clay-21 then made that function
    the same single pass, so this is a delegate kept for its call sites' name
    rather than a second copy of the walk.
    """
    from ....kernels.mesh import selection

    return selection.with_descendants(doc, uids)


def _duplicate(ctx: Any, doc: Any, **_: Any) -> bool:
    """Copy the selection. -> whether anything was copied.

    The 2026-10-10 audit's clay-51: a selection of only hidden objects copies
    nothing (``visible=False`` means it is not duplicated), but the closure
    answered ``None``, which ``run`` reads as "ran" -- True, no step, no toast.
    ``False`` is the honest answer, and ``enabled`` no longer offers it.

    The object ceiling is taken before the first copy is built (clay-23): the
    refusal used to come from ``add_objects``, after every copy of a 4,000-object
    selection had been made.
    """
    from ....kernels.mesh import selection

    del ctx
    by_uid = doc._uid_map()
    chosen = [uid for uid in doc.selection if uid in by_uid and by_uid[uid].visible]
    if not chosen:
        return False
    doc._refuse_past_ceilings([by_uid[uid] for uid in _with_descendants_in(doc, chosen)])
    return bool(selection.duplicate_selected(doc))


def _join(ctx: Any, doc: Any, weld: float = 1e-4, **_: Any) -> None:
    """Merge every selected object into the topmost one in the outliner.

    Outliner order (:func:`outliner_order`) rather than "the one clicked last":
    ``doc.selection`` is a set and carries no order at all, so a target read out
    of it would be whichever object the hash happened to put first -- and the
    name, transform and material default that the merge keeps are the
    *target's*, which makes that an arbitrary answer to a question the user can
    see the answer to. Outliner order and not ``doc.objects`` order, which this
    docstring used to call the same thing: it is not once a group exists (the
    2026-10-07 audit's clay-08).

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
    from ....kernels.mesh import mesh as bm
    from ....kernels.mesh import ops as clay_ops_geom
    from ....kernels.mesh.elements import OpError

    visible = [obj.uid for obj in doc.objects if obj.uid in doc.selection and obj.visible]
    uids = _in_outliner_order(doc, visible)
    by_uid = doc._uid_map()
    objs = [by_uid[uid] for uid in uids]
    worlds = [doc._world_in(uid, by_uid) for uid in uids]
    mesh = clay_ops_geom.join(objs, eps=float(weld), world=worlds)
    # The 2026-10-07 audit's clay-43: the distance used to go up to 1e6 and a
    # merge at it welded every vertex into one point -- an object with no
    # faces, reported as a success. Refused with the number that did it.
    if bm.face_count(mesh) == 0 and any(bm.face_count(o.mesh) for o in objs):
        raise OpError(_collapse_sentence(weld, "these objects"))
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

    return run_object_op(ctx, doc, one, needs_geometry=True) and bool(mirrored)


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
    #
    # The 2026-10-09 second audit's clay-63: a selected group is an empty, so
    # copying "the selection" mirrored an empty and left every child behind.
    # A selected shape takes its descendants with it (``with_descendants``), so
    # the parents-first order below also puts each child's copy under its
    # parent's copy. ``chosen`` is what the user picked; only it and its copies
    # are selected afterwards, never a child riding inside a picked group.
    #
    # The 2026-10-10 audit's clay-23: one uid map for the whole press. Every
    # ``by_uid``/``ancestors``/``index_of``/``world_matrix``/``descendants`` here
    # used to rebuild or scan the document per selected object (3.8 s at 4,000
    # objects), and the object ceiling was only met by ``add_objects`` after
    # every copy had been built. The ceiling is taken first now.
    by_uid = doc._uid_map()
    position: dict[int, int] = {}
    for index, obj in enumerate(doc.objects):
        position.setdefault(obj.uid, index)
    chosen = [u for u in doc.selection if by_uid[u].visible]
    originals = sorted(
        _with_descendants_in(doc, chosen),
        key=lambda u: (len(doc._ancestors_in(u, by_uid)), position[u]),
    )
    doc._refuse_past_ceilings([by_uid[u] for u in originals])
    fresh = {uid: bd.new_uid() for uid in originals}
    copy_worlds: dict[int, Any] = {}
    made: list[Any] = []
    for uid in originals:
        source = by_uid[uid]
        copy = clay_ops_geom.duplicate(source, fresh[uid], taken=taken)
        taken.add(copy.name)
        world = doc._world_in(uid, by_uid)
        mirrored = clay_ops_geom.mirror_world(copy, int(axis), offset, world=world)
        new_world = m3.compose(mirrored.translation, mirrored.rotation, mirrored.scale)
        parent = source.parent
        if parent in fresh:
            try:
                inverse = np.linalg.inv(copy_worlds[parent])
            except np.linalg.LinAlgError as error:
                from ....kernels.mesh.elements import OpError

                raise OpError(
                    f"{by_uid[parent].name!r} has a zero scale, so nothing can be "
                    "placed relative to its copy."
                ) from error
            t, r, s = m3.decompose(inverse @ new_world)
            parent = fresh[parent]
        else:
            t, r, s = _local_from_world(doc, by_uid, source, new_world)
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
    chosen_set = set(chosen)
    doc.select(
        chosen + [obj.uid for uid, obj in zip(originals, made, strict=True) if uid in chosen_set]
    )
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

    # One uid map for the batch (the 2026-10-10 audit's clay-23): ``by_uid`` and
    # ``world_matrix`` per uid scanned, and rebuilt a map of, the whole document
    # each time.
    by_uid = doc._uid_map()
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for uid in uids:
        box = clay_ops_geom.world_box(by_uid[uid], world=doc._world_in(uid, by_uid))
        if box is not None:
            out[uid] = box
    return out


def _local_from_world(
    doc: Any, by_uid: dict[int, Any], obj: Any, world: np.ndarray
) -> tuple[Any, Any, Any]:
    """*world* as local ``(t, r, s)`` under *obj*'s own current parent --
    ``ClayDoc.local_from_world`` over a uid map the caller built once.

    The door re-scans for the object and rebuilds the map to invert its parent's
    world, per call; a loop over a whole selection (Drop to Ground, Snap to Grid,
    Mirror Copy) paid that per object (the 2026-10-10 audit's clay-23). The
    arithmetic is the door's, and so is its refusal of a zero-scale parent.
    """
    from ....kernels.geom3d import math3d as m3
    from ....kernels.mesh.elements import OpError

    world = np.asarray(world, dtype="f8")
    if obj.parent is None:
        return m3.decompose(world)
    try:
        inverse = np.linalg.inv(doc._world_in(obj.parent, by_uid))
    except np.linalg.LinAlgError as error:
        raise OpError(
            f"{by_uid[obj.parent].name!r} has a zero scale, so nothing can be "
            "placed relative to it."
        ) from error
    return m3.decompose(inverse @ world)


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
    # One uid map for the batch (the 2026-10-10 audit's clay-23). The map holds
    # the live objects, which ``set_transform`` edits in place, so a parent's
    # world is still read as it stands when its child is converted back.
    by_uid = doc._uid_map()
    world_before = {
        uid: np.array(doc._world_in(uid, by_uid), dtype="f8", copy=True) for uid in deltas
    }
    order = sorted(deltas, key=lambda uid: len(doc._ancestors_in(uid, by_uid)))
    for uid in order:
        delta = deltas[uid]
        obj = by_uid[uid]
        try:
            if obj.parent is None:
                translation = np.asarray(obj.translation, dtype="f8") + delta
                if doc.set_transform(uid, translation=translation):
                    ran = True
                continue
            world = world_before[uid].copy()
            world[:3, 3] = world[:3, 3] + np.asarray(delta, dtype="f8")
            t, r, s = _local_from_world(doc, by_uid, obj, world)
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

    # The 2026-10-09 second audit's clay-62: a selected group has no box, so the
    # op was enabled, found nothing to drop and said nothing.
    if refuse_if_no_geometry(ctx, doc, list(doc.selection)):
        return False
    boxes = _world_boxes(doc, doc.selection)
    deltas = mason_ops.drop_to_ground(boxes, ground=0.0)
    return _apply_deltas(ctx, doc, deltas)


#: The finest grid Snap to Grid offers, in metres.
SNAP_MIN_STEP = 0.001


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

    def on_grid(position: Any) -> Any:
        # The 2026-10-09 second audit's clay-108: a denormal step made
        # ``position / step`` infinite and ``snap_translation`` raised a bare
        # ``OverflowError`` out of ``run``. The ``Param`` floor keeps the dialog
        # clear of it; this is the backstop for a caller that skips the clamp.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            quotient = np.asarray(position, dtype="f8") / step
        if not np.isfinite(quotient).all():
            raise el.OpError(
                f"A grid step of {step:g} m is too small to snap to; use a step of "
                f"at least {SNAP_MIN_STEP:g} m."
            )
        return clay_ops_geom.snap_translation(position, step)

    # One uid map for the press (the 2026-10-10 audit's clay-23: 1.4 s at 4,000
    # objects, a map rebuilt per object for the depth sort, and an ``any()`` scan
    # of every object per selected uid for the liveness test).
    by_uid = doc._uid_map()

    def one(doc: Any, obj: Any) -> bool:
        if obj.parent is None:
            snapped = on_grid(obj.translation)
        else:
            world = np.array(doc._world_in(obj.uid, by_uid), dtype="f8", copy=True)
            world[:3, 3] = on_grid(world[:3, 3])
            snapped = _local_from_world(doc, by_uid, obj, world)[0]
        # The 2026-10-07 audit's clay-41: ``set_transform`` pushes nothing for a
        # position already on the grid, and says so.
        return doc.set_transform(obj.uid, translation=snapped)

    live = [uid for uid in doc.selection if uid in by_uid]
    order = sorted(live, key=lambda uid: len(doc._ancestors_in(uid, by_uid)))
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

    def one(doc: Any, obj: Any) -> bool:
        if not _is_group(doc, obj):
            return False
        doc.remove_object(obj.uid)
        return True

    return run_object_op(ctx, doc, one)


def _parent_to_last(ctx: Any, doc: Any, **_: Any) -> bool:
    """Parent every other selected object onto the topmost one in the
    outliner.

    The topmost selected object in :func:`outliner_order` is the new parent --
    ``doc.selection`` is a set with no order of its own, the same reason
    ``_join`` reads its own merge target from the outliner's order rather than
    out of the selection directly (see its docstring). "Last
    Selected" is Blender's own name for this gesture, which reads the
    *active* object; Clay has no such concept, so this reuses the identical
    tiebreak a user can see and predict, rather than inventing a second one.

    Because a parent is always drawn above its descendants, the target is never
    a descendant of anything else selected, so this op cannot form a cycle.
    Document order did not have that property -- it made the 2026-10-07
    audit's clay-08 refusal ("cannot be parented to itself or one of its own
    descendants") on a selection whose outliner order was perfectly legal.

    ``ClayDoc.set_parent`` is the door, one call per child; an OpError from it
    is toasted and skipped by ``run_object_op`` exactly like every other
    per-object refusal in this file, so one bad choice in a bigger selection
    does not abandon the rest of it.
    """
    uids = _in_outliner_order(doc, (obj.uid for obj in doc.objects if obj.uid in doc.selection))
    if len(uids) < 2:
        return False
    target, *rest = uids

    def one(doc: Any, obj: Any) -> bool:
        # Already a child of the target: nothing changed (``set_parent`` says so).
        return doc.set_parent(obj.uid, target, keep_world=True)

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

    def one(doc: Any, obj: Any) -> bool:
        return doc.set_parent(obj.uid, None, keep_world=True)

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

    # ``needs_geometry`` for the 2026-10-10 audit's clay-52: a lone group was told
    # "this object is one connected piece", which is not what is wrong with it.
    return run_object_op(ctx, doc, one, needs_geometry=True)


def _pickable_uids(doc: Any) -> list[int]:
    """The uids with something picked that an element op may act on.

    The 2026-10-09 audit's clay-07: an object hidden after it was picked keeps
    its element selection, and Origin to Selection and Separate Selection walked
    ``doc.element_sel`` unfiltered -- moving the pivot of, and splitting, a
    shape the user could not see -- while ``run_mesh_op`` and Assign Material
    skipped it through the same ``_element_pickable`` rule used here.
    """
    from ....kernels.mesh import selection

    out: list[int] = []
    for uid in list(doc.element_sel):
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            continue
        if selection._element_pickable(obj):
            out.append(uid)
    return out


def _separate_selection(ctx: Any, doc: Any, **_: Any) -> bool:
    """Split every selected object's own face selection out as a new
    object, leaving the rest of it behind.

    Face mode only, and reading ``doc.element_sel`` the way every element op
    in this registry does. It walks the objects with something *picked*, not
    ``doc.selection``: the shape being edited stays selected with nothing
    picked, and ``by_selection`` of nothing is a refusal, not a separation.
    ``kernels.mesh.separate.by_selection``
    always answers with exactly two pieces and refuses -- a toast, not an
    abort of the rest of the selection -- a pick touching none of an
    object's faces or all of them.
    """
    from ....kernels.mesh import separate as sep

    def one(doc: Any, obj: Any) -> None:
        # ``ClayDoc.separate`` swaps the source for its pieces in ``selection``
        # in every mode, and a piece just split off has nothing picked inside
        # it, which is the state the rule allows (the 2026-09-26 audit's
        # clay-document-03 had to repair it by hand while ``selection`` was
        # derived from the picks).
        doc.separate(obj.uid, sep.by_selection(obj.mesh, doc.element_sel_of(obj.uid)))

    return run_object_op(ctx, doc, one, uids=_pickable_uids(doc))


def _origin_to_bounds(ctx: Any, doc: Any, **_: Any) -> bool:
    """Move each selected object's origin to its own *world* box's centre.

    Measured off the object's world matrix (``doc.world_matrix``, tranche 3),
    the same one :func:`_world_boxes` reads for Drop to Ground -- an object
    under a parent has a placement its local TRS does not describe alone.
    ``ClayDoc.set_origin`` keeps the geometry and every child exactly where
    they were (its own docstring); only the pivot moves.
    """
    from ....kernels.mesh import ops as clay_ops_geom

    def one(doc: Any, obj: Any) -> bool:
        box = clay_ops_geom.world_box(obj, world=doc.world_matrix(obj.uid))
        if box is None:
            return False
        lo, hi = box
        return doc.set_origin(obj.uid, (lo + hi) * 0.5)

    return run_object_op(ctx, doc, one, needs_geometry=True)


def _origin_to_base(ctx: Any, doc: Any, **_: Any) -> bool:
    """Move each selected object's origin to its own world box's bottom
    centre -- :func:`_origin_to_bounds`'s identical measurement, with the
    Y (up) component read off the box's *low* corner instead of its middle,
    the same axis ``_drop_to_ground`` rests on ``y=0``.
    """
    from ....kernels.mesh import ops as clay_ops_geom

    def one(doc: Any, obj: Any) -> bool:
        box = clay_ops_geom.world_box(obj, world=doc.world_matrix(obj.uid))
        if box is None:
            return False
        lo, hi = box
        return doc.set_origin(obj.uid, [(lo[0] + hi[0]) * 0.5, lo[1], (lo[2] + hi[2]) * 0.5])

    return run_object_op(ctx, doc, one, needs_geometry=True)


def _origin_to_selection(ctx: Any, doc: Any, **_: Any) -> bool:
    """Move each selected object's origin to its own element selection's
    centroid.

    Read off the mesh in local space and carried into a world point through
    the object's own world matrix before ``ClayDoc.set_origin`` writes it
    back, since that door takes a world point (its own signature) and a
    parented object's local positions are not one.
    """
    from ....kernels.mesh import elements as el

    def one(doc: Any, obj: Any) -> bool:
        verts = el.affected_verts(obj.mesh, doc.element_sel_of(obj.uid))
        if len(verts) == 0:
            return False
        local = np.asarray(obj.mesh.positions, dtype="f8")[verts].mean(axis=0)
        homogeneous = np.array([local[0], local[1], local[2], 1.0], dtype="f8")
        point = (doc.world_matrix(obj.uid) @ homogeneous)[:3]
        return doc.set_origin(obj.uid, point)

    return run_object_op(ctx, doc, one, uids=_pickable_uids(doc))


def _origin_to_world(ctx: Any, doc: Any, **_: Any) -> bool:
    """Move each selected object's origin to the world origin -- the one
    origin-* row that needs no measurement at all."""

    def one(doc: Any, obj: Any) -> bool:
        return doc.set_origin(obj.uid, [0.0, 0.0, 0.0])

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
        return f"{op.label.rstrip('.')} works in {_mode_list(op.modes)} mode."
    return reason_for(op, doc) if not op.enabled(doc) else ""


def _repeat_last(ctx: Any, doc: Any, **_: Any) -> bool:
    recent = doc.recent_op
    return run(ctx, doc, get(recent.op_name), **recent.params)


def _delete(ctx: Any, doc: Any, **_: Any) -> bool:
    """Delete what is selected. -> whether anything was removed.

    The 2026-10-10 audit's clay-51: hidden objects are left alone by Delete, so a
    selection of only those removed nothing and the closure still answered
    ``None`` -- "ran", no step, no toast. A press whose only result is a refusal
    sentence is the same case: the sentence is shown, and the answer is False.
    """
    from ....kernels.mesh import selection

    head = doc.history.head
    for message in selection.delete_selected(doc):
        toast(ctx, message)
    return doc.history.head != head


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
    keeps the unwrapped layout: ``regen.carry_over`` carries ``uv`` across a
    rebuild whose face count is unchanged (a segment-count change, which makes
    new faces, is the one edit that re-derives them). The 2026-10-09 audit's
    clay-32 found this paragraph still promising the canonical coordinates.
    """
    from ....kernels.mesh import uv as uv_mod

    def one(doc: Any, obj: Any) -> bool:
        # ``set_mesh``'s answer, so an object with nothing to unwrap does not
        # report ``ran`` (found by the selection fuzz test, clay-41's rule).
        return doc.set_mesh(obj.uid, uv_mod.box_unwrap(obj.mesh), keep_generator=True)

    return run_object_op(ctx, doc, one, needs_geometry=True)


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
    from ....kernels.mesh import selection

    count = len(doc.materials)
    if not 0 <= index < count:
        raise el.OpError(
            f"there is no palette entry {index}; the palette has {count}"
            f" (0 to {count - 1})."
        )
    painted = 0
    hidden = 0
    for uid in list(doc.element_sel):
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            continue
        # clay-11 again: an object hidden after it was picked keeps its element
        # selection, and every other element door skips it.
        if not selection._element_pickable(obj):
            hidden += 1
            continue
        faces = el.convert(obj.mesh, doc.element_sel_of(uid), "face").faces
        if len(faces) and doc.paint_faces(uid, faces, index):
            painted += len(faces)
    name = doc.materials[index].name or f"slot {index}"
    if not painted and hidden and hidden == len(doc.element_sel):
        # The 2026-10-09 audit's clay-07: every pick was on a hidden object, and
        # "already use" told the user a falsehood about faces nothing touched.
        ctx.toast("The picked faces are on hidden objects; show them first.", "info")
        return False
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

    A flat open sheet, or a closed shell too thin to enclose any volume, has no
    meaningful outside; such a shell is left as it is.

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
    from ....kernels.mesh.elements import OpError

    # The 2026-10-09 second audit's clay-96: ``MAX_CLEAN_CORNERS`` bounded each
    # object, not the press -- thirteen maximal spheres selected together passed
    # it object by object and stalled the frame thread for about 3 s. The corners
    # are charged across the whole selection, before anything is touched. (One
    # object over the ceiling keeps the kernel's own sentence.)
    by_uid = {obj.uid: obj for obj in doc.objects}
    chosen = [by_uid[uid] for uid in doc.selection if uid in by_uid]
    total = sum(len(obj.mesh.loops) for obj in chosen)
    if len(chosen) > 1 and total > ops_clean.MAX_CLEAN_CORNERS:
        raise OpError(
            f"These {len(chosen)} objects have {total:,} corners between them, past the "
            f"{ops_clean.MAX_CLEAN_CORNERS:,} Recalculate Normals can process in one press "
            "without stalling. Select fewer objects."
        )

    def one(doc: Any, obj: Any) -> bool:
        mesh = ops_clean.recalc_outside(obj.mesh)
        # ``set_mesh`` answers False for the identical mesh -- the common case,
        # since every generator already emits outward winding.
        return mesh is not obj.mesh and doc.set_mesh(obj.uid, mesh)

    return run_object_op(ctx, doc, one, needs_geometry=True)


def _faces(count: int) -> str:
    return f"{count:,} face" + ("" if count == 1 else "s")


def _simplify(ctx: Any, doc: Any, target: int = 100, **_: Any) -> bool:
    """Reduce every selected object toward *target* faces.

    **Object mode only, and whole objects**, for :func:`_recalc_normals`'s reason:
    "the target" is a face count of a shell, and a face selection is not one.

    **Hidden objects are left alone, by the selection rather than by a check
    here**: hiding deselects and ``ClayDoc.select`` refuses a hidden uid, so a
    hidden object is never in ``doc.selection`` for this to rewrite -- the same
    reason :func:`_recalc_normals` carries no visibility test of its own.

    **Freezes the generator, through ``set_mesh``**, like every other op that
    changes geometry. The kernel answers the *identical* mesh when nothing can
    be merged (or the object is already at or under *target*), and that identity
    is what keeps a no-op from pushing an undo step.

    **The toast carries the real counts** because the result is often not the
    number the user typed: a flat merge can land far below it (a subdivided cube
    asked for 100 faces becomes 6) and the collapse stage stops above it when the
    rest would break a seam or a material border. A count that is silently not
    the one asked for reads as a bug, so the sentence names both.
    """
    from ....kernels.mesh import mesh as bm
    from ....kernels.mesh import ops_simplify
    from ....kernels.mesh.elements import OpError

    target = int(target)
    by_uid = {obj.uid: obj for obj in doc.objects}
    chosen = [by_uid[uid] for uid in doc.selection if uid in by_uid]
    # ``MAX_CLEAN_CORNERS``'s lesson (the 2026-10-09 second audit's clay-96): the
    # kernel's ceiling bounds each object, not the press, and the frame thread
    # pays for the whole selection. One object over keeps the kernel's sentence.
    #
    # Charged with the kernel's own function, over the objects it would actually
    # work on (the 2026-10-11 audit's clay-06 and clay-24): the sum used to be
    # bare corners against the bare ceiling, so two UV objects each inside the
    # kernel's halved ceiling stalled the frame for 2.6 s together, while two
    # objects already under the target -- which the kernel hands straight back --
    # were refused as a pair.
    working = [obj for obj in chosen if bm.face_count(obj.mesh) > target]
    cost = ops_simplify.selection_cost([obj.mesh for obj in working], target)
    if len(working) > 1 and cost > ops_simplify.MAX_SIMPLIFY_CORNERS:
        raise OpError(
            f"These {len(working)} objects add up to {cost:,} corners of work (a mesh "
            "with UVs counts double), past the "
            f"{ops_simplify.MAX_SIMPLIFY_CORNERS:,} Simplify can process in one press "
            "without stalling. Select fewer objects."
        )

    before = after = changed = 0
    short = refused = False

    def one(doc: Any, obj: Any) -> bool:
        nonlocal before, after, changed, short, refused
        was = bm.face_count(obj.mesh)
        try:
            mesh = ops_simplify.simplify(obj.mesh, target)
        except OpError:
            # ``run_object_op`` toasts it; the flag only keeps the "nothing to
            # do" sentence below from piling a second, untrue one on top.
            refused = True
            raise
        now = bm.face_count(mesh)
        short = short or now > target
        if mesh is obj.mesh or not doc.set_mesh(obj.uid, mesh):
            return False
        before += was
        after += now
        changed += 1
        return True

    ran = run_object_op(ctx, doc, one, needs_geometry=True)
    if ran and changed > 1:
        # The target is per object, and the sentence sums the counts: "to 12"
        # beside a typed 10 read as the target being missed (the 2026-10-11
        # audit's clay-25), so the count of objects and the "each" are said.
        reached = (
            "" if not short else "; some stopped above it rather than break seams or materials"
        )
        ctx.toast(
            f"Simplified {_faces(before)} to {after:,} across {changed} objects, "
            f"each toward {target:,}{reached}.",
            "info",
        )
    elif ran:
        reached = "" if not short else (
            f"; {target:,} was not reachable without breaking seams or materials"
        )
        ctx.toast(f"Simplified {_faces(before)} to {after:,}{reached}.", "info")
    elif not refused:
        if short:
            ctx.toast(
                "Nothing more can be merged or collapsed without breaking seams or materials.",
                "info",
            )
        else:
            ctx.toast(f"Already at or under {_faces(target)}; nothing to simplify.", "info")
    return ran


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
                ctx,
                doc,
                lambda doc, obj: doc.set_shading(obj.uid, None, smooth),
                needs_geometry=True,
            )
        from ....kernels.mesh import elements as el
        from ....kernels.mesh import selection

        # Whether any face actually changed, for the reason the object branch
        # above returns it (found by the selection fuzz test): faces already
        # shaded the way asked pushed no step and still answered ``ran``.
        ran = False
        for uid in list(doc.element_sel):
            try:
                obj = doc.by_uid(uid)
            except KeyError:
                continue
            # clay-11: a hidden object keeps its element selection; skip it.
            if not selection._element_pickable(obj):
                continue
            faces = el.convert(obj.mesh, doc.element_sel_of(uid), "face")
            ran = doc.set_shading(uid, faces.faces, smooth) or ran
        return ran

    return run


def _frame(ctx: Any, doc: Any, **_: Any) -> bool:
    """Frame the selection in the viewport. -> whether there was a viewport to
    frame in: an agent's tab has none, and ``ran: true`` for a camera that did
    not move is the 2026-10-07 audit's clay-41."""
    view = getattr(ctx, "clay_view", None)
    if view is None:
        return False
    view.frame_selection(doc)
    return True


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
        from ....kernels.mesh import selection

        del ctx
        mode = doc.element_mode
        ran = False
        for uid in list(doc.element_sel):
            try:
                obj = doc.by_uid(uid)
            except KeyError:
                continue
            # clay-11: grow/shrink/linked must not rewrite what a hidden object
            # holds -- the user cannot see the result.
            if not selection._element_pickable(obj):
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

    def one(doc: Any, obj: Any) -> bool:
        packed = uvtools.pack_islands(obj.mesh, margin=float(margin), rotate=bool(rotate))
        return doc.set_mesh(obj.uid, packed, keep_generator=True)

    # The 2026-10-10 audit's clay-52: without ``needs_geometry`` a lone group
    # answered with the kernel's own sentence ("pack_islands needs texture
    # coordinates -- unwrap this object first"), and a group cannot be unwrapped.
    return run_object_op(ctx, doc, one, needs_geometry=True)


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
            reason=_element_only_reason,
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
            enabled=has_visible_selected,
            reason=_has_visible_selected_reason,
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
            name="simplify",
            label="Simplify...",
            modes=("object",),
            run=_simplify,
            enabled=has_objects,
            reason=_has_objects_reason,
            hint="Merges flat neighbouring faces into larger ones, then, if "
            "that is not enough, collapses edges toward the target. Seams and "
            "material borders are kept, so the result can stay above it. "
            "Whole objects only.",
            params=(
                Param(
                    "target",
                    "target faces",
                    100.0,
                    10.0,
                    low=4.0,
                    integer=True,
                    refuse_below=True,
                ),
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
                    high=MAX_WELD_DISTANCE,
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
            hint="Snaps every selected object's world position onto a grid of "
            "the given step, one axis at a time.",
            # The 2026-10-09 second audit's clay-108: ``low=0.0`` is a value the op
            # itself refuses (clay-37) and a denormal one overflowed. A millimetre
            # is the finest grid worth snapping to, as Bevel's width floor is.
            params=(
                Param("step", "grid step (m)", 1.0, 0.1, low=SNAP_MIN_STEP, refuse_below=True),
            ),
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
            "'active object', so the outliner's order stands in for it, the "
            "same rule Merge Objects uses for its own target.",
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
            reason=_element_only_reason,
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
            reason=_element_only_reason,
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
            check=_inset_check,
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
    # Restored in the 2026-10 Clay UI rebuild (Landing 12): the picoCAD-level cut of
    # 112c69e2 took Bevel out with the Blender-shaped long tail, but a low-poly
    # modeller still wants the chamfered corner. The kernel (``ops_bevel``) never
    # left -- the ``.rblk`` migration replays it -- so this row is only the door.
    register(
        Op(
            name="bevel",
            label="Bevel Edges...",
            modes=("edge",),
            run=_element("ops_bevel.bevel_edges"),
            key="Ctrl+B",
            drag=DragSpec("width", "distance", "ops_bevel.bevel_edges"),
            enabled=in_mode("edge"),
            reason=_in_mode_reason("edge"),
            params=(Param("width", "width (m)", 0.05, 0.01, low=0.001),),
        )
    )
    # ``ops_dissolve.dissolve_faces`` merges a connected block of faces into
    # one n-gon; "Merge Faces" is the word a user searches for.
    #
    # Registering it (rather than adding a button) is what gets it into every
    # surface at once: the right-click menu, the Inspector, Ctrl+K and the
    # bare-letter keys all read this table.
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
            run=_weld,
            enabled=in_mode("vertex"),
            reason=_in_mode_reason("vertex"),
            params=(
                Param("eps", "distance (m)", 1e-4, 1e-4, low=1e-9, high=MAX_WELD_DISTANCE),
            ),
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
            "Colour card's palette row, counting from 0.",
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
            # ``has_any_selection``, not ``bool(doc.selection)``: in an element
            # mode the edited shape stays selected with nothing picked, and
            # Delete there deletes picks -- a gate on the object selection
            # enabled it, ran over nothing and answered ``ran`` (found by the
            # selection fuzz test).
            enabled=can_delete,
            reason=_can_delete_reason,
            key="Del",
        )
    )


_register_defaults()
