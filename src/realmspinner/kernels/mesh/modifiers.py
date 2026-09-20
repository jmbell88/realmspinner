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

**Boolean targets are evaluated recursively**, through the same
:func:`evaluate`, which is what makes a chain of booleans-of-booleans work
with no special case. A missing target uid is an error on that modifier, not
a crash (a hand-edited or partially-loaded ``.rblk`` can name one); a cycle --
only reachable the same way, since :meth:`~.document.ClayDoc.set_modifiers`
refuses one going forward -- is an error on the modifier that *closes* it,
caught by an in-progress set of uids rather than let recurse until the stack
overflows.
"""

from __future__ import annotations

from collections.abc import Callable
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
    result. See ``_apply_boolean`` and ``_boolean_deps_still_valid``.
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
    target_eval = _evaluate(ctx.doc, target_uid, ctx.visiting)
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
    """*raw* into the shape *p* declares: an index, a bool, or a clamped number."""
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
        return max(0, min(len(p.choices) - 1, idx))
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
    """
    graph: dict[int, set[int]] = {
        obj.uid: targets(stack if obj.uid == uid else obj.modifiers) for obj in doc.objects
    }
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(graph, WHITE)

    def visit(u: int) -> bool:
        color[u] = GRAY
        for v in graph.get(u, ()):
            if v not in color:
                continue
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and visit(v):
                return True
        color[u] = BLACK
        return False

    return any(color[u] == WHITE and visit(u) for u in graph)


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


def _boolean_deps_still_valid(
    doc: ClayDoc,
    obj: Obj,
    boolean_deps: tuple[tuple[int, Mesh, np.ndarray, np.ndarray], ...],
    visiting: frozenset[int],
) -> bool:
    """Whether every boolean dependency this evaluation recorded still holds.

    **World matrices** (tranche 3), not local TRS: a boolean modifier's
    placement depends on where ``self`` and its target sit in *world* space,
    so a pure ancestor move -- neither object's own local TRS changes at all
    -- has to invalidate this exactly as a direct move would, or a parent
    dragged across the scene would leave every boolean beneath it stuck
    where it used to be. Comparing ``obj.trs()``/``target_obj.trs()`` (as
    this did before parenting existed) would miss precisely that case.
    """
    for target_uid, target_mesh, self_world, target_world in boolean_deps:
        try:
            doc.by_uid(target_uid)  # existence only; the world matrix below is the value
        except KeyError:
            return False
        if not np.array_equal(self_world, doc.world_matrix(obj.uid)) or not np.array_equal(
            target_world, doc.world_matrix(target_uid)
        ):
            return False
        target_eval = _evaluate(doc, target_uid, visiting | {obj.uid})
        if target_eval.mesh is not target_mesh:
            return False
    return True


def evaluate(doc: ClayDoc, uid: int) -> Evaluated:
    """*uid*'s base mesh run through its modifier stack. See the module docstring."""
    return _evaluate(doc, uid, frozenset())


def _evaluate(doc: ClayDoc, uid: int, visiting: frozenset[int]) -> Evaluated:
    obj = doc.by_uid(uid)
    stack = obj.modifiers
    if not any(m.enabled for m in stack):
        # The fast path: no computation, no cache entry, the base mesh handed
        # back ``is``-identical. See the module docstring.
        return Evaluated(obj.mesh, ())

    entry = doc._evaluated.get(uid)
    if (
        entry is not None
        and entry.base is obj.mesh
        and entry.stack == stack
        and _boolean_deps_still_valid(doc, obj, entry.boolean_deps, visiting)
    ):
        return entry.result

    new_visiting = visiting | {uid}
    ctx = EvalContext(doc=doc, obj=obj, visiting=new_visiting)
    mesh = obj.mesh
    errors: list[tuple[int, str]] = []
    for mod in stack:
        if not mod.enabled:
            continue
        kind_def = MODIFIERS.get(mod.kind)
        if kind_def is None:
            errors.append((mod.id, f"Unknown modifier kind {mod.kind!r}."))
            continue
        try:
            grown = kind_def.apply(mesh, mod.as_dict(), ctx)
            ops_modifiers._refuse_growth("This modifier", ops_modifiers._triangle_count(grown))
        except el.OpError as error:
            errors.append((mod.id, str(error)))
            continue
        mesh = grown

    result = Evaluated(mesh, tuple(errors))
    doc._evaluated[uid] = _CacheEntry(obj.mesh, stack, tuple(ctx.boolean_deps), result)
    return result
