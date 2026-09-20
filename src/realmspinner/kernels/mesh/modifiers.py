"""The modifier stack: a live, re-orderable recipe layered on top of a mesh.

``Obj.mesh`` stays the **base** -- the thing every element edit, element pick,
drag, op and undo step already acts on, unchanged by anything in this module.
An object additionally carries ``modifiers: tuple[Modifier, ...]`` (default
``()``), and the **evaluated** mesh is the base run through each *enabled*
modifier in order. Display, export, measurement and object-mode picking read
the evaluated mesh (:func:`evaluate`, via ``ClayDoc.evaluated``/``.
evaluation``); editing reads the base, exactly as it always has.

**An object with no enabled modifiers evaluates to ``obj.mesh`` itself, the
same object, ``is``-identical.** :func:`evaluate` checks that before it
touches the cache or does anything else, which is what keeps every document,
test and GPU cache key that predates modifiers behaving exactly as it always
did: nothing computed, nothing cached, the base mesh handed straight back.

**A modifier that refuses is skipped, not fatal.** :func:`evaluate` catches
each kind's own :class:`~.elements.OpError` as it walks the stack, records
``(modifier id, message)`` in :attr:`Evaluated.errors` and carries on with the
*next* modifier over whatever mesh the *previous* one produced -- Blender's
own behaviour, and the one that keeps the viewport showing something rather
than blanking the object the moment one modifier's parameters stop making
sense. Any exception that is not an :class:`~.elements.OpError` propagates: a
kind's own bug is not a refusal.

**The cache lives on the document**, ``ClayDoc._evaluated: dict[uid,
_CacheEntry]``, not a module global -- a module-level cache would outlive the
document that owns the uids it is keyed on, the same reason
``adjacency``/``cached_triangulation`` key off the mesh object itself rather
than off a global. An entry is valid exactly when the base mesh is still the
same object (``is``, never ``id()`` -- ids recycle, see ``ClayDoc.
mesh_stamp``'s own docstring for why), the modifiers tuple still compares
equal, and, for every boolean dependency the stack has, the target's own
evaluated mesh is still the same object and both objects' TRS still compare
equal. A hit returns the **same** :class:`Evaluated` object, so a viewport
cache keyed on ``id(evaluated.mesh)`` -- ``document._PLANS``,
``adjacency``'s own triangulation cache -- does not re-upload geometry that
has not actually changed.

**Boolean targets are evaluated through the same :func:`_evaluate`**, which is
what makes a chain of booleans-of-booleans work with no special case. Not
through Python recursion, though, since the 2026-09-19 audit's clay-02: a
chain of 1,500 boolean targets (well inside ``glbimport.MAX_OBJECTS``) used
to raise an uncaught ``RecursionError`` here, so :func:`_evaluate` now
resolves the whole chain with its own explicit stack of frames -- see its
docstring. A missing target uid is an error on that modifier, not a crash (a
hand-edited or partially-loaded ``.rblk`` can name one); a cycle -- only
reachable the same way, since :meth:`~.document.ClayDoc.set_modifiers`
refuses one going forward -- is an error on the modifier that *closes* it,
caught by an in-progress set of uids rather than let anything grow unbounded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import elements as el
from . import ops_boolean, ops_modifiers
from .document import ClayDoc, Obj
from .mesh import Mesh

__all__ = [
    "MODIFIERS",
    "EvalContext",
    "Evaluated",
    "ModParam",
    "Modifier",
    "ModifierKind",
    "make",
    "next_id",
    "targets",
    "with_params",
    "would_cycle",
    "evaluate",
]


# --- the vocabulary ----------------------------------------------------------


@dataclass(frozen=True)
class ModParam:
    """One parameter's shape: a kernel-side description, not the UI's own.

    Kernels may not import ``studio``, so this is deliberately *not*
    ``clay.ops.Param`` -- a second, smaller type here rather than a dependency
    running the wrong way across the layering boundary. ``target=True`` marks
    a parameter whose stored value is another object's uid (only ``boolean``'s
    ``target`` has this today); :func:`targets` reads it generically rather
    than naming the one kind that currently uses it, so the next kind that
    wants an object reference needs no change here.
    """

    name: str
    label: str
    default: float
    low: float = 0.0
    high: float = 1e6
    step: float = 0.01
    integer: bool = False
    boolean: bool = False
    choices: tuple[str, ...] = ()
    target: bool = False
    warn: str = ""


@dataclass(frozen=True)
class EvalContext:
    """What a modifier's ``apply`` needs beyond the mesh and its own params.

    Only the ``boolean`` kind uses ``doc``/``obj``/``visiting`` -- reaching
    into the document for its target's own evaluated mesh and TRS, and
    detecting a cycle before it recurses into one -- but ``ModifierKind.
    apply``'s ``Callable`` shape is uniform across every kind, so the other
    nine take and ignore this rather than being an exception to it.

    ``boolean_deps`` is mutable and *not* part of that uniform contract: it is
    :func:`evaluate`'s own scratch space, appended to by ``boolean``'s
    ``apply`` as a side effect, and read back once the whole stack has run to
    build the cache entry. A kind that is not ``boolean`` never touches it.

    Each entry is ``(target_uid, target_mesh, self_world, target_world)`` --
    **world** matrices (tranche 3: scene structure), not the two objects' own
    TRS, because ``self``/``target`` can each sit under different ancestors
    once parenting exists: a pure ancestor move that never touches either
    object's own local TRS still changes where the boolean result should
    land, and a cache keyed on local TRS alone would go on serving a stale
    result. See ``_run_boolean`` and :func:`_evaluate`'s own cache-validity
    phase.
    """

    doc: ClayDoc
    obj: Obj
    visiting: frozenset[int]
    boolean_deps: list[tuple[int, Mesh, np.ndarray, np.ndarray]] = field(default_factory=list)


@dataclass(frozen=True)
class Modifier:
    """One stack entry: a kind, its coerced parameters, enabled or not.

    ``id`` is unique within its *own* object's stack only (never reused there
    -- :func:`next_id` is ``max(existing) + 1``), not across the document; a
    modifier is always addressed together with the uid that owns it. ``params``
    is a sorted tuple of ``(name, value)`` pairs rather than a dict so the
    whole object is hashable and compares by value -- both of which the
    document-held evaluation cache leans on (:mod:`.modifiers`' own
    docstring).
    """

    id: int
    kind: str
    params: tuple[tuple[str, float | int | bool], ...]
    enabled: bool = True

    def get(self, name: str, default: Any = None) -> Any:
        for key, value in self.params:
            if key == name:
                return value
        return default

    def as_dict(self) -> dict[str, float | int | bool]:
        return dict(self.params)


@dataclass(frozen=True)
class ModifierKind:
    """One entry in the :data:`MODIFIERS` registry: a kind's whole contract."""

    name: str
    label: str
    hint: str
    params: tuple[ModParam, ...]
    apply: Callable[[Mesh, dict, EvalContext], Mesh]


# --- registration -------------------------------------------------------


def _ignore_ctx(fn: Callable[[Mesh, dict], Mesh]) -> Callable[[Mesh, dict, EvalContext], Mesh]:
    """Lift a plain ``(mesh, params) -> mesh`` kernel to the registry's shape.

    Every kind but ``boolean`` is pure mesh algebra with no need of the
    document context -- see :mod:`.ops_modifiers`'s own docstring -- so this is
    the one place that difference is bridged, rather than nine kernels each
    carrying an unused ``ctx`` parameter.
    """

    def wrapped(mesh: Mesh, params: dict, ctx: EvalContext) -> Mesh:
        del ctx
        return fn(mesh, params)

    return wrapped


def _apply_boolean(mesh: Mesh, params: dict, ctx: EvalContext) -> Mesh:
    """``boolean``'s own ``apply``: the one kind that reaches into the document.

    The target is evaluated *recursively* through :func:`_evaluate`, sharing
    ``ctx.visiting`` -- so a target whose own stack has a boolean modifier is
    itself resolved fully, and a cycle anywhere in the chain is caught the
    moment it would revisit a uid already being evaluated, rather than by
    exhausting the call stack.

    **World matrices, not local TRS** (tranche 3): ``self``/``target`` may
    each sit under a different ancestor chain, so the placement that matters
    is each object's *world* transform, threaded into
    ``ops_boolean.boolean``'s own ``world=`` parameter -- see that function's
    docstring. ``mesh`` (``self``'s own evaluated mesh, in its own local
    frame) is passed through as ``ops_boolean``'s "first object", which is
    also its own convention for the frame the result lands in: since ``self``
    itself is never re-transformed (``ops_boolean.boolean`` never moves its
    first object), the result still lands in ``self``'s local frame exactly
    as it did before parenting existed, whatever the world matrices say.
    """
    target_uid = int(params.get("target", 0))
    if target_uid == 0:
        raise el.OpError("Choose a target object.")
    if target_uid in ctx.visiting:
        raise el.OpError("This modifier's boolean target would create a cycle.")
    try:
        target_obj = ctx.doc.by_uid(target_uid)
    except KeyError:
        raise el.OpError(f"Target object {target_uid} no longer exists.") from None
    # Recurses -- one Python call per link -- but only ever *one* link deep
    # from here: this function's own caller is either ``_evaluate``'s single
    # per-modifier dispatch (which, for its *own* uid's stack, resolves a
    # boolean target through the iterative path below instead of coming back
    # here -- see that function's docstring) or ``ClayDoc.apply_modifiers``'s
    # bake, which calls this once per prefix entry from its own ordinary
    # ``for`` loop. Either way, this one recursive call is the whole recursion
    # budget it ever spends; whatever chain of targets *target_uid* itself
    # depends on is resolved entirely inside that single call to
    # :func:`_evaluate`, which manages its own explicit stack rather than
    # recursing again (the 2026-09-19 audit's clay-02).
    target_eval = _evaluate(ctx.doc, target_uid, ctx.visiting)
    return _run_boolean(mesh, target_obj, params, ctx, target_eval)


def _run_boolean(
    mesh: Mesh, target_obj: Obj, params: dict, ctx: EvalContext, target_eval: Evaluated
) -> Mesh:
    """The boolean arithmetic half of :func:`_apply_boolean`, taking *target_obj*
    and *target_eval* already resolved rather than fetching them itself.

    Split out so :func:`_evaluate`'s own iterative stack can run this same
    math for a whole chain of boolean targets it has already resolved without
    Python recursion -- :func:`_apply_boolean` remains the one recursive,
    single-target entry point, used by :meth:`~.document.ClayDoc.
    apply_modifiers`'s bake (which only ever needs one target resolved, and
    lets that one call reach arbitrarily deep on its own).
    """
    target_uid = int(params.get("target", 0))
    self_world = ctx.doc.world_matrix(ctx.obj.uid)
    target_world = ctx.doc.world_matrix(target_uid)
    ctx.boolean_deps.append(
        (
            target_uid,
            target_eval.mesh,
            np.array(self_world, copy=True),
            np.array(target_world, copy=True),
        )
    )
    self_obj = replace(ctx.obj, mesh=mesh)
    target_as_obj = replace(target_obj, mesh=target_eval.mesh)
    operation = ops_boolean.KINDS[int(params.get("operation", 1))]
    return ops_boolean.boolean(
        [self_obj, target_as_obj], operation, world=[self_world, target_world]
    )


#: Registration order is menu order -- the properties pane's "Add modifier"
#: combo and the agent's ``kind`` enum both walk this dict, never a sorted
#: view of it, so the order a kind is added here is the order either surface
#: offers it in.
MODIFIERS: dict[str, ModifierKind] = {}


def _register(
    name: str, label: str, hint: str, params: tuple[ModParam, ...], apply: Callable
) -> None:
    MODIFIERS[name] = ModifierKind(name=name, label=label, hint=hint, params=params, apply=apply)


_register(
    "mirror",
    "Mirror",
    "The mesh plus its reflection across a local plane through the origin.",
    (
        ModParam(name="axis", label="Axis", default=0.0, choices=("X", "Y", "Z")),
        ModParam(name="weld", label="Weld distance", default=0.0001, low=0.0, high=1.0),
    ),
    _ignore_ctx(ops_modifiers.mirror),
)
_register(
    "array",
    "Array",
    "Copies translated one after another.",
    (
        ModParam(name="count", label="Count", default=3.0, low=2.0, high=200.0, integer=True),
        ModParam(name="offset_x", label="Offset X", default=1.0, low=-1e4, high=1e4),
        ModParam(name="offset_y", label="Offset Y", default=0.0, low=-1e4, high=1e4),
        ModParam(name="offset_z", label="Offset Z", default=0.0, low=-1e4, high=1e4),
        ModParam(name="weld", label="Weld distance", default=0.0, low=0.0, high=1.0),
    ),
    _ignore_ctx(ops_modifiers.array_linear),
)
_register(
    "radial-array",
    "Radial Array",
    "Copies spun about the local origin, spread over an angle.",
    (
        ModParam(name="count", label="Count", default=6.0, low=2.0, high=200.0, integer=True),
        ModParam(name="angle", label="Angle", default=360.0, low=-3600.0, high=3600.0),
        ModParam(name="axis", label="Axis", default=1.0, choices=("X", "Y", "Z")),
    ),
    _ignore_ctx(ops_modifiers.array_radial),
)
_register(
    "solidify",
    "Solidify",
    "A shell of the surface, with thickness.",
    (
        ModParam(name="thickness", label="Thickness", default=0.05, low=0.0, high=10.0),
        ModParam(name="offset", label="Offset", default=-1.0, low=-1.0, high=1.0),
    ),
    _ignore_ctx(ops_modifiers.solidify),
)
_register(
    "bevel",
    "Bevel",
    "Rounds every edge sharper than an angle.",
    (
        ModParam(name="width", label="Width", default=0.02, low=0.0, high=10.0),
        ModParam(name="angle", label="Angle", default=30.0, low=0.0, high=180.0),
    ),
    _ignore_ctx(ops_modifiers.bevel_by_angle),
)
_register(
    "subdivide",
    "Subdivide",
    "The Catmull-Clark limit surface, one level at a time.",
    (ModParam(name="levels", label="Levels", default=1.0, low=1.0, high=3.0, integer=True),),
    _ignore_ctx(ops_modifiers.subdivide),
)
_register(
    "weld",
    "Weld",
    "Merges vertices closer together than a distance.",
    (ModParam(name="distance", label="Distance", default=0.0001, low=0.0, high=1.0),),
    _ignore_ctx(ops_modifiers.weld),
)
_register(
    "triangulate",
    "Triangulate",
    "Every face fanned or ear-clipped to triangles.",
    (),
    _ignore_ctx(ops_modifiers.triangulate),
)
_register(
    "smooth",
    "Smooth",
    "Laplacian vertex smoothing; boundary vertices held fixed.",
    (
        ModParam(name="factor", label="Factor", default=0.5, low=0.0, high=1.0),
        ModParam(
            name="iterations", label="Iterations", default=1.0, low=1.0, high=50.0, integer=True
        ),
    ),
    _ignore_ctx(ops_modifiers.laplacian_smooth),
)
_register(
    "boolean",
    "Boolean",
    "Union, subtract or intersect against another object.",
    (
        ModParam(name="target", label="Target", default=0.0, low=0.0, high=1e9, target=True),
        ModParam(
            name="operation",
            label="Operation",
            default=float(ops_boolean.KINDS.index("difference")),
            choices=ops_boolean.KINDS,
        ),
    ),
    _apply_boolean,
)


# --- construction helpers -----------------------------------------------


def _coerce(p: ModParam, raw: Any) -> float | int | bool:
    """*raw* into the shape *p* declares: an index, a bool, or a clamped number.

    A choice given as a string that names none of *p.choices* is refused by
    name (below); a choice given as a *number* used to be silently clamped to
    the nearest legal index instead, so ``axis=5`` on a 3-choice param was
    quietly reinterpreted as ``axis=2`` -- a different, unrequested choice --
    rather than refused. The 2026-09-19 audit's clay-27: an out-of-range
    index now gets the same refusal an unrecognised string already got,
    rather than the silent stand-in a plain number happened to fall through
    to.
    """
    if p.choices:
        if isinstance(raw, str):
            try:
                idx = p.choices.index(raw)
            except ValueError:
                raise el.OpError(
                    f"{p.label} must be one of {', '.join(p.choices)}, not {raw!r}."
                ) from None
        else:
            idx = int(raw)
            if not 0 <= idx < len(p.choices):
                raise el.OpError(
                    f"{p.label} must be one of {', '.join(p.choices)}, not index {idx!r}."
                )
        return idx
    if p.boolean:
        return bool(raw)
    value = float(raw)
    value = max(p.low, min(p.high, value))
    return int(round(value)) if p.integer else value


def _kind_or_raise(kind: str) -> ModifierKind:
    kind_def = MODIFIERS.get(kind)
    if kind_def is None:
        raise el.OpError(
            f"Unknown modifier kind {kind!r}. Choose one of {', '.join(sorted(MODIFIERS))}."
        )
    return kind_def


def make(kind: str, params: dict[str, Any] | None = None, *, id: int) -> Modifier:
    """A new :class:`Modifier`: every declared parameter filled and clamped.

    Refuses an unknown ``kind`` or a ``params`` key that kind does not
    declare, naming the valid ones either way -- the same "half-read is worse
    than refused" rule :mod:`.serialize` follows, applied here to a caller's
    dict instead of a file.
    """
    kind_def = _kind_or_raise(kind)
    given = dict(params or {})
    valid_names = {p.name for p in kind_def.params}
    unknown = sorted(set(given) - valid_names)
    if unknown:
        raise el.OpError(
            f"{kind} has no parameter named {unknown[0]!r}; choose from "
            f"{', '.join(sorted(valid_names)) or '(none)'}."
        )
    out = {p.name: _coerce(p, given.get(p.name, p.default)) for p in kind_def.params}
    return Modifier(id=id, kind=kind, params=tuple(sorted(out.items())), enabled=True)


def with_params(mod: Modifier, updates: dict[str, Any]) -> Modifier:
    """*mod* with some of its parameters replaced, clamped the same way :func:`make` clamps them.

    ``enabled`` and ``id`` pass through untouched -- this only ever touches
    the parameters named in ``updates``, refusing (naming the valid ones) if
    any of them is not one of *mod*'s kind's own.
    """
    kind_def = _kind_or_raise(mod.kind)
    by_name = {p.name: p for p in kind_def.params}
    unknown = sorted(set(updates) - set(by_name))
    if unknown:
        raise el.OpError(
            f"{mod.kind} has no parameter named {unknown[0]!r}; choose from "
            f"{', '.join(sorted(by_name)) or '(none)'}."
        )
    out = mod.as_dict()
    for name, raw in updates.items():
        out[name] = _coerce(by_name[name], raw)
    return replace(mod, params=tuple(sorted(out.items())))


def next_id(stack: tuple[Modifier, ...]) -> int:
    """The next id for a new entry in *stack*: never reused within it."""
    return max((m.id for m in stack), default=0) + 1


def targets(stack: tuple[Modifier, ...]) -> set[int]:
    """Every object uid any modifier in *stack* names as a target.

    Enabled or not: a disabled boolean still names a dependency the document's
    cycle check (:func:`would_cycle`) and the properties pane's "excluded from
    this combo" rule both need to see, and re-enabling a modifier must not be
    the moment a cycle it already described becomes possible for the first
    time.
    """
    out: set[int] = set()
    for mod in stack:
        kind_def = MODIFIERS.get(mod.kind)
        if kind_def is None:
            continue
        for p in kind_def.params:
            if p.target:
                value = mod.get(p.name)
                if value:
                    out.add(int(value))
    return out


def would_cycle(doc: ClayDoc, uid: int, stack: tuple[Modifier, ...]) -> bool:
    """Whether setting *stack* on *uid* would create a boolean dependency cycle.

    Builds the whole document's target graph with *uid*'s own edges swapped
    for the proposed *stack* -- not only the edges touching *uid* -- because a
    cycle a change introduces need not pass through the object that changed
    (it only has to pass through an edge that did). A target uid absent from
    the document is not a cycle by itself; :func:`evaluate` reports that
    separately, as a missing-target error on the modifier that names it.

    The white/gray/black DFS below walks with an **explicit stack of
    resumable iterators**, not recursion: the 2026-09-19 audit's clay-02
    found a boolean-target chain of 1,500 objects (well inside
    ``glbimport.MAX_OBJECTS``) raised an uncaught ``RecursionError`` here,
    on the ordinary path of just calling :meth:`~.document.ClayDoc.
    set_modifiers`. Each stack entry is ``(node, iterator over its own
    out-edges)`` -- ``ClayDoc.ancestors``' own explicit-stack shape, widened
    from a chain to a branching graph by resuming each node's iterator where
    it left off rather than following a single ``.parent`` pointer.
    """
    graph: dict[int, set[int]] = {
        obj.uid: targets(stack if obj.uid == uid else obj.modifiers) for obj in doc.objects
    }
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(graph, WHITE)

    for start in graph:
        if color[start] != WHITE:
            continue
        color[start] = GRAY
        frames: list[tuple[int, Iterator[int]]] = [(start, iter(graph.get(start, ())))]
        while frames:
            node, neighbours = frames[-1]
            descended = False
            for v in neighbours:
                if v not in color:
                    continue
                if color[v] == GRAY:
                    return True
                if color[v] == WHITE:
                    color[v] = GRAY
                    frames.append((v, iter(graph.get(v, ()))))
                    descended = True
                    break
            if not descended:
                color[node] = BLACK
                frames.pop()
    return False


# --- evaluation -----------------------------------------------------------


@dataclass(frozen=True)
class Evaluated:
    """One object's base mesh, run through its modifier stack.

    ``errors`` is ``(modifier id, sentence)`` for each modifier that refused
    and was skipped -- see the module docstring's "skipped, not fatal" rule.
    Empty on the fast path (no enabled modifiers) and whenever every enabled
    modifier ran cleanly.
    """

    mesh: Mesh
    errors: tuple[tuple[int, str], ...] = ()


@dataclass
class _CacheEntry:
    """:mod:`.modifiers`' own note to itself about why a result is still good.

    Not exported -- ``ClayDoc._evaluated`` holds these, and nothing outside
    this module needs to know their shape, only that :func:`evaluate` honours
    them.
    """

    base: Mesh
    stack: tuple[Modifier, ...]
    boolean_deps: tuple[tuple[int, Mesh, np.ndarray, np.ndarray], ...]
    result: Evaluated


def evaluate(doc: ClayDoc, uid: int) -> Evaluated:
    """*uid*'s base mesh run through its modifier stack. See the module docstring."""
    return _evaluate(doc, uid, frozenset())


class _EvalFrame:
    """One object's evaluation, alive on :func:`_evaluate`'s own explicit
    stack in place of a Python call frame. See that function's docstring."""

    __slots__ = (
        "uid",
        "obj",
        "stack",
        "visiting",
        "ctx",
        "mesh",
        "idx",
        "errors",
        "phase",
        "cache_entry",
        "dep_idx",
    )

    def __init__(self, doc: ClayDoc, uid: int, incoming_visiting: frozenset[int]) -> None:
        obj = doc.by_uid(uid)
        self.uid = uid
        self.obj = obj
        self.stack = obj.modifiers
        # Matches the old recursive ``_evaluate``'s own ``new_visiting`` --
        # *this* uid included -- which is what a self-target or a cycle back
        # to an ancestor is checked against below.
        self.visiting = incoming_visiting | {uid}
        self.ctx = EvalContext(doc=doc, obj=obj, visiting=self.visiting)
        self.mesh = obj.mesh
        self.idx = 0
        self.errors: list[tuple[int, str]] = []
        self.cache_entry = doc._evaluated.get(uid)
        self.phase = "cache" if self.cache_entry is not None else "apply"
        self.dep_idx = 0


def _evaluate(doc: ClayDoc, uid: int, visiting: frozenset[int]) -> Evaluated:
    """*uid*'s base mesh run through its modifier stack, resolving any chain
    of boolean-modifier targets with an explicit stack of :class:`_EvalFrame`
    rather than Python recursion.

    Before this fix, a boolean modifier's target was resolved through a
    direct recursive call -- from :func:`_apply_boolean`, and again from the
    cache-validity check this function used to delegate to -- so a chain of
    1,500 objects, each targeting the next (well inside ``glbimport.
    MAX_OBJECTS``, no cycle anywhere), raised an uncaught ``RecursionError``
    on ordinary viewing, export, readiness and :meth:`~.document.ClayDoc.
    set_modifiers` itself (the 2026-09-19 audit's clay-02). Every object this
    call needs -- *uid* itself and every boolean target beneath it,
    transitively -- gets one :class:`_EvalFrame` on ``frames`` instead of one
    Python stack frame; a target not yet resolved is pushed and the current
    frame is revisited once it is done, the same "come back to this one"
    shape :meth:`~.document.ClayDoc.ancestors` already uses for a plain
    parent chain, widened here to a frame that does real work (a cache check,
    then a modifier stack) rather than only following a pointer. ``memo``
    holds this call's own results, uid to :class:`Evaluated`, so a target
    shared by two different boolean modifiers (a diamond, not a cycle) is
    resolved once.

    Each frame runs up to two phases. **"cache"**, only when
    ``doc._evaluated`` already holds an entry for this uid: valid exactly
    when the base mesh is still the same object, the modifiers tuple still
    compares equal, and -- checked here one recorded dependency at a time,
    each needing its own target *resolved*, hence a possible push -- every
    boolean target's world matrix is unchanged and its own evaluation still
    produces the identical mesh object the cache remembered (see the module
    docstring's cache paragraph). Any failure drops straight to **"apply"**:
    walk the modifier stack in order, run each enabled one, and for a boolean
    modifier resolve its target the same way -- push and revisit if it is not
    already in ``memo`` -- instead of the old recursive call. A cycle is
    still refused by name, not by exhausting the stack: :meth:`~.document.
    ClayDoc.set_modifiers` refuses one going forward, so the only way one
    reaches here is a hand-edited or partially-loaded ``.rblk``, caught the
    moment a target names a uid already in the current frame's own
    ``visiting`` set (every frame on the path down to it, itself included) --
    see ``test_a_hand_edited_cycle_is_refused_on_the_closing_modifier_not_recursion``.
    """
    memo: dict[int, Evaluated] = {}
    frames: list[_EvalFrame] = [_EvalFrame(doc, uid, visiting)]

    while frames:
        frame = frames[-1]
        obj = frame.obj

        if not any(m.enabled for m in frame.stack):
            # The fast path: no computation, no cache entry, the base mesh
            # handed back ``is``-identical. See the module docstring.
            memo[frame.uid] = Evaluated(obj.mesh, ())
            frames.pop()
            continue

        if frame.phase == "cache":
            entry = frame.cache_entry
            if entry is None or entry.base is not obj.mesh or entry.stack != frame.stack:
                frame.phase = "apply"
                continue
            deps = entry.boolean_deps
            if frame.dep_idx >= len(deps):
                # Every recorded dependency held: the cached result stands,
                # exactly as ``entry.result`` was built.
                memo[frame.uid] = entry.result
                frames.pop()
                continue
            target_uid, target_mesh, self_world, target_world = deps[frame.dep_idx]
            try:
                doc.by_uid(target_uid)  # existence only; the world check below is the value
            except KeyError:
                frame.phase = "apply"
                continue
            # World matrices (tranche 3), not local TRS: a boolean modifier's
            # placement depends on where ``self`` and its target sit in
            # *world* space, so a pure ancestor move -- neither object's own
            # local TRS changes at all -- has to invalidate this exactly as a
            # direct move would.
            if not np.array_equal(self_world, doc.world_matrix(frame.uid)) or not np.array_equal(
                target_world, doc.world_matrix(target_uid)
            ):
                frame.phase = "apply"
                continue
            if target_uid in frame.visiting:
                # A cycle turning up while validating an old cache entry is
                # unreachable through any live edit (see this function's own
                # docstring) -- only a hand-edited file could produce one --
                # and this phase has no refusal to record it under; fall
                # through to "apply", where the modifier loop below refuses
                # it by name instead of pushing a frame nothing would ever
                # pop.
                frame.phase = "apply"
                continue
            if target_uid not in memo:
                frames.append(_EvalFrame(doc, target_uid, frame.visiting))
                continue
            if memo[target_uid].mesh is not target_mesh:
                frame.phase = "apply"
                continue
            frame.dep_idx += 1
            continue

        # frame.phase == "apply"
        if frame.idx >= len(frame.stack):
            result = Evaluated(frame.mesh, tuple(frame.errors))
            doc._evaluated[frame.uid] = _CacheEntry(
                obj.mesh, frame.stack, tuple(frame.ctx.boolean_deps), result
            )
            memo[frame.uid] = result
            frames.pop()
            continue

        mod = frame.stack[frame.idx]
        if not mod.enabled:
            frame.idx += 1
            continue
        kind_def = MODIFIERS.get(mod.kind)
        if kind_def is None:
            frame.errors.append((mod.id, f"Unknown modifier kind {mod.kind!r}."))
            frame.idx += 1
            continue

        if mod.kind == "boolean":
            params = mod.as_dict()
            target_uid = int(params.get("target", 0))
            if target_uid == 0:
                frame.errors.append((mod.id, "Choose a target object."))
                frame.idx += 1
                continue
            if target_uid in frame.visiting:
                frame.errors.append(
                    (mod.id, "This modifier's boolean target would create a cycle.")
                )
                frame.idx += 1
                continue
            try:
                target_obj = doc.by_uid(target_uid)
            except KeyError:
                frame.errors.append((mod.id, f"Target object {target_uid} no longer exists."))
                frame.idx += 1
                continue
            if target_uid not in memo:
                frames.append(_EvalFrame(doc, target_uid, frame.visiting))
                continue
            target_eval = memo[target_uid]
            try:
                grown = _run_boolean(frame.mesh, target_obj, params, frame.ctx, target_eval)
                ops_modifiers._refuse_growth("This modifier", ops_modifiers._triangle_count(grown))
            except el.OpError as error:
                frame.errors.append((mod.id, str(error)))
                frame.idx += 1
                continue
            frame.mesh = grown
            frame.idx += 1
            continue

        try:
            grown = kind_def.apply(frame.mesh, mod.as_dict(), frame.ctx)
            ops_modifiers._refuse_growth("This modifier", ops_modifiers._triangle_count(grown))
        except el.OpError as error:
            frame.errors.append((mod.id, str(error)))
            frame.idx += 1
            continue
        frame.mesh = grown
        frame.idx += 1

    return memo[uid]
