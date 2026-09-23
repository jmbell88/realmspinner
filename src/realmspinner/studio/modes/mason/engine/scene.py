"""The one resolver: what a node actually is once its ancestors have had their say.

**One resolver, every consumer.** The viewport, the three exporters (Stage D)
and the thumbnail render must never work out inherited state twice --
``plotter/scene.py``'s argument, one dimension over. A group's visibility, a
material override, a prefab instance's copies: each is a fact about a node's
whole ancestry, and a second place computing any of them is a second place
that can compute it slightly wrong the day someone edits the first.

The five combination rules, each stated once and computed once:

* **transform composes** -- ``world = parent_world @ node.local()``;
* **visible ANDs** -- one hidden ancestor hides everything under it, and the
  leaf's own flag never moves, so unhiding it restores exactly what was
  there;
* **locked ORs**, and is *reported, never enforced* -- ``plotter/scene.py``'s
  rule, restated: a lock stops the user, not the document, or an undo could
  not put back what was there before the lock was applied;
* **static ORs**, because it reaches the engine manifest;
* **material override wins nearest-ancestor** -- a ``MeshNode`` or
  ``TerrainNode`` with its own ``material`` set replaces whatever it
  inherited, for itself and everything further under it, so a group can
  retint a whole prop set without touching the assets underneath.

**Groups are not in :func:`resolve`'s result.** A :class:`~.nodes.GroupNode`
draws nothing, and a consumer that had to remember to skip one is a consumer
that could forget to -- ``plotter/scene.py`` says exactly this about a
``GroupLayer`` and it is just as true one dimension over.

**``walk`` is the single traversal.** :func:`resolve` is ``walk`` accumulating
world matrices. ``gltfout.scene_model`` (Stage D, not this module) will be
``walk`` accumulating *structure* and local transforms instead, because a glTF
export has to keep the parents rather than flatten them away. That is why
``visit`` is handed every field :class:`Placed` carries, for *every* node the
walk crosses -- including a :class:`~.nodes.GroupNode`, which :func:`resolve`
is the one that throws away. A structural exporter that wants the tree instead
of the flattening can build one gltf node per visit call and hang it under
``path[:-1]``'s node without this module knowing or caring that it did. One
traversal, one visibility rule, one prefab expansion, so the viewport and the
export can never disagree about either.

**Prefab expansion is this module's own recursion, not ``nodes.walk``'s.**
A :class:`~.nodes.PrefabNode` names a template in ``doc.prefabs`` by string,
read through *every time* -- there is no propagation step and no "apply to
instances" button, because the propagation step is exactly what an editor
gets wrong (``document.py``'s ``unpack_instance`` docstring makes the same
argument for the escape hatch that exists instead). ``expand_prefabs=False``
yields the :class:`~.nodes.PrefabNode` itself, unexpanded -- what an outliner
wants, since the template's own nodes are not rows in *this* document's tree.
A template naming a prefab that does not exist does not raise: it yields
nothing for that branch, the same "known and repairable, not a crash" answer
``document.py`` gives for a dangling :class:`~.refs.LibraryRef`.

``nodes.walk`` only ever sees one object graph -- the scene tree, or one
template's subtree -- so its cycle guard cannot see a cycle that runs
*through prefab names*: template A placing an instance of template B placing
an instance of template A is two ordinary, acyclic object trees individually,
and only becomes a cycle when ``doc.prefabs`` is read a second time for a name
already being expanded on this path. ``define_prefab`` refuses to create that
cycle (the loud door), but a hand-edited ``.rscn`` bypasses every door in the
app the same way a hand-edited cycle in the node graph does -- so this module
carries its own backstop for its own recursion: a ``frozenset`` of template
names already being expanded on the current path (refuses a name seen twice,
quietly, the same "the corrupt branch costs itself" rule as
:func:`~.nodes.walk`'s ``path`` set) and a depth counter capped at
:data:`~.nodes.MAX_DEPTH` -- the same ceiling, not a second number to keep in
sync with it, because a chain of prefabs nested deeper than any real document
would go is exactly as suspicious as a chain of groups that deep.

**``owner`` is a deliberate deviation from the plan, and here is why.** The
plan asks for a free function ``owner_uid(path) -> int``. It cannot be
written that way: for a node inside an expanded prefab instance, ``path``
runs ``(...scene uids..., prefab_node_uid, ...template uids...)``, and
nothing in a bare tuple of uids says *where* the boundary between the two
halves is -- a template can be placed at any depth, so a function that only
sees the finished path cannot tell which prefix uid is the instance and which
suffix uids are the template's own. The walk *does* know the boundary the
moment it crosses one, so it records the answer once, right there, as
``Placed.owner`` -- which is the same "compute inherited state once, in the
resolver, not again in every consumer" rule this whole module exists to
enforce, applied to itself. The rule it implements: **picking is per ref,
selection is per owner** -- a hit on a node inside a prefab instance selects
the *instance* (the one thing the outliner actually has a row for), not its
internals; a hit on an ordinary node selects that node, not its group. Once a
walk crosses into an expanded prefab, ``owner`` is fixed at that instance's
uid for everything under it, including through a *nested* prefab boundary --
a template-of-templates has no scene-tree identity of its own to hand out.

**``dangling`` is reported, never discovered, and reported for the same
reason ``owner`` is computed here rather than downstream.** ``resolve`` takes
no :class:`~.refs.GeometrySource` -- it stays pure, per the package's own
rule that nothing here touches a filesystem or resolves a reference -- so it
cannot know whether a :class:`~.refs.LibraryRef` still resolves. What it can
do is ask ``doc.missing``, the set the *host* already filled in the one time
it tried and failed (``document.py``'s own docstring on ``missing_refs``).
``Placed.dangling`` is exactly that lookup, ``ref_key(node.ref) in
doc.missing``, so every consumer of a resolved scene agrees about which nodes
are broken without any of them touching a job id.

**Enumeration still costs the whole hidden subtree, and that is a named
trade, not an oversight.** ``plotter/scene.py``'s ``resolve`` never descends
into a hidden group at all, which is both the cheap half and the correct
half. This module cannot do that and also reuse :func:`~.nodes.walk`'s own
cycle-and-depth backstop within one tree segment (a scene's roots, or one
expanded template's roots) -- that walk is a flat, already-unrolled
traversal with no hook to stop it descending into one branch mid-generator,
and re-implementing a second cycle guard here to get that hook back is
exactly the "one backstop, not two" rule this module is told to keep. So a
hidden subtree is still walked by :func:`~.nodes.walk` (bounded the same way
every other branch is, by document size and :data:`~.nodes.MAX_DEPTH`); what
is skipped is only the work *this* module would otherwise do with it --
computing a ``Placed``, calling ``visit`` on it, recursing across a prefab
boundary underneath it. A frame's cost is the same order either way; what
changes is which lines run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .....kernels.geom3d import gltf
from .....kernels.geom3d import math3d as m3
from . import nodes as nd
from .nodes import GroupNode, MeshNode, Node, PrefabNode
from .refs import Ref, ref_key

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import MasonDoc

#: Two different numbers answer two different questions, and the first
#: measurement pass here conflated them -- see ``dev/measurements/`` for the
#: dated write-up both are read off, re-measured 2026-09-11 against
#: ``nodes.Node.local()``'s memo (which took 64% of a resolve's cost off the
#: table by not rebuilding an unmoved node's matrix every frame; the numbers
#: below are warm, best of five, against that code, not the pre-memo pass).
#:
#: :data:`MAX_PLACED` is the **absurdity bound** -- where :func:`resolve`
#: *refuses* -- and it is set for the case that is really worth guarding: a
#: corrupt ``.rscn``, or an array op run with a count someone typed an extra
#: zero into, where resolving is a hang holding an unsaved document rather
#: than a scene a person actually built. It is not a frame-rate number: past
#: it the operation is no longer an interactive application *at any* frame
#: rate, refusal or not. At 50,000 items the worst shape measured (prefab
#: instances) costs 225 ms a resolve -- under 5 fps; at 100,000 it is 469 ms --
#: under 3 fps, and the two authored shapes are within a third of that same
#: number. That is the order of magnitude where "a person placed this" stops
#: being the likely story, so the ceiling sits at the round order past it:
#: refusing a document at 100,000 placed items costs a user nothing an actual
#: level would have asked for, and a document that size gets a real sentence
#: naming the count and the ceiling instead of a multi-second hang.
MAX_PLACED = 100_000

#: :data:`PLACED_WARN_THRESHOLD` is the **soft** number -- the measured point
#: past which ``resolve()`` stops fitting a 60 Hz frame alongside everything
#: else a frame still has to do (culling, GPU submission, imgui...), on the
#: same "about a third of the frame, not all of it" criterion
#: ``terrain.MAX_TERRAIN_SIDE`` was set by (5.65 ms of 16.7 there). At 1,500
#: items the worst shape measured -- prefab instances -- costs 6.33 ms (38%
#: of a 60 Hz frame); the two authored shapes cost 4.7-5.0 ms at the same
#: count. This constant **refuses nothing**: it is what the viewport's HUD
#: and the document pane warn against in Stage E ("this scene is large enough
#: to cost you frame rate"), read from here rather than typed into a pane, so
#: the warning and the measurement it is about cannot drift apart. A scene
#: past this line is not a mistake -- it is a person's work running slower
#: than 60 Hz, which is a worse editing experience than a faster one and a
#: categorically better one than a refusal.
PLACED_WARN_THRESHOLD = 1_500

__all__ = [
    "MAX_PLACED",
    "PLACED_WARN_THRESHOLD",
    "DrawNode",
    "NodePool",
    "Placed",
    "resolve",
    "resolved_count",
    "resolved_for",
    "walk",
    "world_bounds",
]


@dataclass(frozen=True)
class Placed:
    """One node, and what its whole ancestry adds up to.

    Frozen for :class:`~.plotter._map_model.Rect`'s reason, restated here: a
    consumer holds a list of these for the length of one frame or one export,
    and a caller that could write into one would be changing what the *next*
    consumer sees, with nothing in the document to say why.
    """

    #: The leaf that draws, by reference -- for a node reached through an
    #: expanded prefab, this is the *template's* node object, shared across
    #: every instance (``document.py``'s ``define_prefab`` stores the
    #: template with ``fresh_uids=False``, precisely so re-defining it reads
    #: as an update rather than a replacement).
    node: Node
    #: Uids root-first, through every prefab boundary crossed. Distinguishes
    #: one instance's copy of a template leaf from another's, since the two
    #: share ``node`` by identity but not the path that reached it.
    path: tuple[int, ...]
    #: The uid selection and the gizmo address. See the module docstring's
    #: "owner" section for why this cannot be recovered from ``path`` alone.
    owner: int
    world: np.ndarray
    visible: bool
    locked: bool
    static: bool
    ref: Ref | None
    #: The nearest ancestor's override, or ``None`` for "as authored".
    material: gltf.Material | None
    #: "" when authored directly in the scene tree; otherwise the template
    #: this node was reached through.
    prefab: str
    dangling: bool


#: What ``visit`` is called with for every node :func:`walk` crosses -- the
#: same eleven values, in the same order, that build a :class:`Placed`. See
#: the module docstring for why one shape serves both a flattening consumer
#: (:func:`resolve`) and a structural one (``gltfout.scene_model``, later).
VisitFn = Callable[
    [Node, tuple[int, ...], int, np.ndarray, bool, bool, bool, Ref | None, Any, str, bool],
    None,
]

#: One node's fully-inherited state, threaded through a single tree segment:
#: (world, visible, locked, static, material). Not exported -- an internal
#: shape, not part of this module's contract.
_State = tuple[np.ndarray, bool, bool, bool, gltf.Material | None]

_IDENTITY_STATE: _State = (np.eye(4, dtype="f8"), True, False, False, None)


def walk(
    doc: MasonDoc,
    visit: VisitFn,
    *,
    include_hidden: bool = False,
    expand_prefabs: bool = True,
    enter: VisitFn | None = None,
    max_items: int | None = None,
    roots: Sequence[Node] | None = None,
) -> None:
    """The single traversal every other function in this module is built on.

    ``roots`` defaults to ``doc.roots`` -- every call site but
    :func:`resolved_count` wants the document's own scene tree. That function
    wants an arbitrary, possibly-unattached sequence instead (see its own
    docstring: "how much would attaching these add"), and the 2026-09-23
    audit's mason-02 is why it now asks for that through this parameter rather
    than calling :func:`_walk_segment` on its own: see this function's
    ``max_items`` paragraph below for what going around it used to cost.

    See the module docstring for the five combination rules, the prefab
    expansion rule, the ``owner`` deviation, and the hidden-subtree trade.

    ``enter`` is called for a :class:`~.nodes.PrefabNode` **the moment the
    walk crosses into it**, with the same eleven values ``visit`` takes, and
    it exists because of a gap in this module's own docstring. That docstring
    tells a structural exporter to "build one gltf node per visit call and
    hang it under ``path[:-1]``'s node" -- which is exactly right for every
    node except the one place a path has a uid in it that ``visit`` was never
    called for. An expanded instance's template roots carry
    ``(..., prefab_node_uid, template_root_uid)``, and the instance itself
    ``continue``s below without being visited, so ``path[:-1]`` names a
    node the exporter has never seen. Without this hook the only way back to
    the instance's own transform is ``world @ inv(node.local())``, an inverse
    per instance that is singular the moment anything in the scene is scaled
    to zero -- for a transform the walk is holding in its hand at the time.
    Default ``None``, so :func:`resolve`, :func:`resolved_for` and
    :func:`world_bounds` are the traversal they always were.

    It fires only when the expansion actually happens. An instance whose
    template is missing, cyclic or nested past :data:`~.nodes.MAX_DEPTH`
    yields nothing at all -- the "the corrupt branch costs itself" rule the
    module docstring states -- and an exporter that emitted a node for one
    anyway would put something in the file that the viewport does not draw,
    which is the one disagreement between them this module exists to prevent.

    ``max_items`` is the 2026-09-13 audit's mason-03: :func:`resolve` used to
    be the only place this ceiling was enforced, so ``resolved_for`` (a
    Properties panel and the gizmo, every frame) and ``gltfout.scene_model``'s
    bare walk had none -- 152 nodes of nested prefabs resolved to 127,550
    items through those two paths while ``resolve`` itself correctly refused.
    Counted here, once, so every caller of this traversal -- present or
    future -- inherits the same refusal ``resolve`` always had, rather than
    each consumer needing to remember to ask for it. :func:`resolved_count`
    used to sidestep this same door by calling :func:`_walk_segment` on its
    own, which is the 2026-09-23 audit's mason-02: the pre-flight a scene
    op's write side calls *before* attaching anything counted an exponential
    prefab expansion all the way to completion -- 45.9 s at 8.4 million items
    -- instead of refusing quickly the way every other caller of this
    traversal already does. It is routed through here now (with ``roots``
    naming what it is counting) so it inherits the same bounded refusal.
    """
    # Read off the module global at call time rather than bound as the
    # parameter default: a caller (or a test) that adjusts ``MAX_PLACED`` at
    # runtime must see it take effect on the very next walk, the same way
    # ``resolve``'s own default always has.
    ceiling = MAX_PLACED if max_items is None else max_items
    counted = _bounded(visit, ceiling)
    counted_enter = _bounded(enter, ceiling) if enter is not None else None
    _walk_segment(
        doc.roots if roots is None else roots,
        doc,
        counted,
        include_hidden=include_hidden,
        expand_prefabs=expand_prefabs,
        parent_path=(),
        owner=None,
        prefab="",
        inherited=_IDENTITY_STATE,
        prefab_chain=frozenset(),
        prefab_depth=0,
        enter=counted_enter,
    )


def _bounded(visit: VisitFn, max_items: int) -> VisitFn:
    """Wrap ``visit`` so the call past ``max_items`` refuses instead of
    running -- the single point ``walk`` enforces :data:`MAX_PLACED` from, so
    ``resolve``, ``resolved_for`` and every structural exporter share one
    ceiling rather than each needing its own copy of the count."""
    count = 0

    def counting_visit(*args: Any, **kwargs: Any) -> None:
        nonlocal count
        count += 1
        if count > max_items:
            raise ValueError(
                f"this scene resolves to more than {max_items} placed items "
                "(MAX_PLACED); refusing rather than silently drawing or "
                "exporting a truncated scene"
            )
        visit(*args, **kwargs)

    return counting_visit


def _walk_segment(
    roots: Sequence[Node],
    doc: MasonDoc,
    visit: VisitFn,
    *,
    include_hidden: bool,
    expand_prefabs: bool,
    parent_path: tuple[int, ...],
    owner: int | None,
    prefab: str,
    inherited: _State,
    prefab_chain: frozenset[str],
    prefab_depth: int,
    enter: VisitFn | None = None,
) -> None:
    """One ``nodes.walk`` over one contiguous object graph -- the scene's own
    roots, or one expanded prefab instance's template roots -- carrying the
    inherited state each node contributes to what is under it.

    ``nodes.walk`` hands back a flat, depth-first pre-order sequence with the
    parent as an object reference; because pre-order visits a parent before
    any of its children, ``state_by_id`` always has a node's parent's answer
    ready by the time the node itself is processed, so this is one pass
    rather than one pass to build the tree and a second to fold state down
    it.
    """
    state_by_id: dict[int, _State] = {}
    # The path each node was reached by, for the reason ``state_by_id``
    # exists: ``nodes.walk`` is pre-order, so a node's parent has already
    # recorded its answer by the time the node itself is processed.
    #
    # This used to be ``parent_path + (node.uid,)`` computed straight off the
    # *segment's* base, which meant a node's own ancestry inside the segment
    # was simply not in its path -- a mesh three groups down came back as
    # ``(mesh_uid,)``. Nothing in Stage C noticed, because every consumer it
    # had wanted the path only as a unique key and one uid is already unique
    # within a segment. What it broke is the thing this module's own docstring
    # tells a structural exporter to do -- "hang it under ``path[:-1]``'s
    # node" -- which lands every node at the root. Found by ``gltfout.py``
    # in Stage D, which is exactly the consumer that sentence was written for.
    path_by_id: dict[int, tuple[int, ...]] = {}
    hidden_ids: set[int] = set()
    for node, parent, _index, _depth in nd.walk(roots):
        if parent is None:
            p_world, p_visible, p_locked, p_static, p_material = inherited
            p_path = parent_path
            parent_hidden = False
        else:
            p_world, p_visible, p_locked, p_static, p_material = state_by_id[id(parent)]
            p_path = path_by_id[id(parent)]
            parent_hidden = id(parent) in hidden_ids

        path = p_path + (node.uid,)
        path_by_id[id(node)] = path
        world = p_world @ node.local()
        visible = p_visible and bool(node.visible)
        locked = p_locked or bool(node.locked)
        static = p_static or bool(node.static)
        own_material = getattr(node, "material", None)
        material = own_material if own_material is not None else p_material
        state_by_id[id(node)] = (world, visible, locked, static, material)

        if parent_hidden or (not visible and not include_hidden):
            hidden_ids.add(id(node))
            continue

        this_owner = owner if owner is not None else node.uid

        if isinstance(node, PrefabNode) and expand_prefabs:
            template = doc.prefabs.get(node.template)
            if (
                template is not None
                and node.template not in prefab_chain
                and prefab_depth < nd.MAX_DEPTH
            ):
                if enter is not None:
                    # ``prefab`` here is the template this *instance* was
                    # reached through -- "" for one placed in the scene tree,
                    # and the outer name for an instance that is itself part
                    # of another template -- not ``node.template``, which is
                    # what it expands into and is already the ``prefab`` every
                    # node under it is handed.
                    enter(
                        node, path, this_owner, world, visible, locked, static,
                        None, material, prefab, False,
                    )
                _walk_segment(
                    [template],
                    doc,
                    visit,
                    include_hidden=include_hidden,
                    expand_prefabs=expand_prefabs,
                    parent_path=path,
                    owner=this_owner,
                    prefab=node.template,
                    inherited=(world, visible, locked, static, material),
                    prefab_chain=prefab_chain | {node.template},
                    prefab_depth=prefab_depth + 1,
                    enter=enter,
                )
            # A missing template, a name already on this path (a prefab
            # cycle), or a chain nested past MAX_DEPTH each yield nothing for
            # this branch -- quietly, the same "the corrupt part costs
            # itself" rule ``nodes.walk`` already applies to an object cycle.
            continue

        ref = node.ref if isinstance(node, MeshNode) else None
        dangling = ref is not None and ref_key(ref) in doc.missing
        visit(
            node, path, this_owner, world, visible, locked, static, ref, material, prefab, dangling
        )


def resolved_count(doc: MasonDoc, roots: Sequence[Node]) -> int:
    """How many items :func:`walk` would visit over ``roots`` under ``doc``'s
    current prefab table -- groups included, since that is the exact basis
    ``walk``'s own ``max_items`` ceiling (``_bounded``) counts from, not the
    smaller, group-filtered count :func:`resolve` returns. ``roots`` need not
    already be attached to ``doc``: prefab expansion only ever reads
    ``doc.prefabs``, never ``doc.roots``, so this also answers "how much
    would attaching these add" -- see :meth:`~.document.MasonDoc.resolved_growth`.

    The 2026-09-23 audit's mason-01: a placed :class:`~.nodes.PrefabNode`
    instance costs the scene *tree* exactly one node, but it expands into its
    whole template subtree (and, transitively, whatever prefab that template
    itself places) every time the scene draws, exports or is picked. Nothing
    on the write side ever charged that expanded size, so a run of
    individually-guarded placements -- each comfortably under
    :data:`MAX_PLACED` on the tree side -- could still push what the scene
    *resolves to* past the ceiling, at which point ``resolve`` (and every
    other caller of :func:`walk`) refuses for good: the viewport draws
    nothing, export and picking stop, on a document that still saves and
    reopens clean. This function gives a write-side caller the same count
    ``walk`` would enforce, before anything is attached.

    The 2026-09-23 audit's mason-02: this used to call :func:`_walk_segment`
    directly, bypassing :func:`walk`'s own ``max_items`` refusal (``_bounded``)
    entirely -- so the very pre-flight this docstring's previous paragraph
    describes counted an exponential prefab expansion (nested branching
    references) all the way to completion instead of refusing once it was
    already well past :data:`MAX_PLACED`: 45.9 s at 8.4 million items. Routed
    through :func:`walk` now, with ``max_items=MAX_PLACED`` explicit (a
    caller-adjusted module :data:`MAX_PLACED` -- a test's ``monkeypatch``,
    say -- must still bound this the same way it bounds every other caller),
    so a runaway count raises the same refusal :func:`resolve` always has
    rather than counting to the end.
    """
    count = 0

    def visit(*_args: Any) -> None:
        nonlocal count
        count += 1

    walk(
        doc,
        visit,
        include_hidden=False,
        expand_prefabs=True,
        roots=list(roots),
        max_items=MAX_PLACED,
    )
    return count


def resolve(
    doc: MasonDoc, *, include_hidden: bool = False, max_items: int = MAX_PLACED
) -> list[Placed]:
    """Every *drawable* node, with its inherited state, groups left out.

    Groups are not in the result -- see the module docstring -- and neither is
    an expanded :class:`~.nodes.PrefabNode` itself (only what it expands to).
    This function always expands prefabs -- it calls :func:`walk` with
    ``expand_prefabs=True`` hardcoded, and takes no parameter of its own to
    change that. It is :func:`walk` itself that also answers
    ``expand_prefabs=False``, for a caller that wants a
    :class:`~.nodes.PrefabNode` instance to *be* the result for that branch --
    an outliner, say, since the template's own nodes are not rows in this
    document's tree. The 2026-09-14 audit's mason-06: this docstring used to
    describe that mode as if ``resolve`` itself accepted it, and
    ``resolve(doc, expand_prefabs=False)`` raises ``TypeError``.

    ``max_items`` refuses rather than truncates: a ``.rscn`` that resolves to
    more placed items than a frame budget allows is not a document this
    resolver silently shows part of -- the caller (a load, typically) gets a
    real sentence naming the count and the ceiling, not a scene that is
    quietly missing its last few thousand props with nothing on screen to say
    so.

    The 2026-09-23 audit's mason-02: this used to call :func:`walk` with no
    ``max_items`` of its own, so ``walk``'s own default -- the module's
    :data:`MAX_PLACED`, not whatever this function was handed -- was what
    actually bounded the traversal. A caller that raised its own ceiling
    above the module default (a batch export with a deliberately larger
    budget, say) was refused at the *lower*, module number instead, with a
    sentence naming the number it never asked for. Forwarded here so this
    function's own parameter is the one that governs.
    """
    out: list[Placed] = []

    def visit(
        node: Node,
        path: tuple[int, ...],
        owner: int,
        world: np.ndarray,
        visible: bool,
        locked: bool,
        static: bool,
        ref: Ref | None,
        material: gltf.Material | None,
        prefab: str,
        dangling: bool,
    ) -> None:
        if isinstance(node, GroupNode):
            return
        if len(out) >= max_items:
            raise ValueError(
                f"this scene resolves to more than {max_items} placed items "
                "(MAX_PLACED); refusing rather than silently drawing or "
                "exporting a truncated scene"
            )
        out.append(
            Placed(
                node, path, owner, world, visible, locked, static, ref, material, prefab, dangling
            )
        )

    walk(doc, visit, include_hidden=include_hidden, expand_prefabs=True, max_items=max_items)
    return out


class _Found(Exception):
    """Stops :func:`resolved_for`'s walk the instant its uid is seen.

    A private control-flow exception rather than a return value threaded
    through ``visit``'s signature, because that signature is shared with
    :func:`resolve` and ``gltfout.scene_model`` (see the module docstring) and
    is not this function's to narrow.
    """

    def __init__(self, placed: Placed) -> None:
        self.placed = placed


def resolved_for(doc: MasonDoc, uid: int) -> Placed | None:
    """One node's inherited state, by uid -- group or leaf, hidden or not.

    What a properties panel or a gizmo asks: the selected node may be a
    group, may be hidden, and :func:`resolve` skips both. Answers with the
    *first* match in walk order when ``uid`` names a node inside more than one
    prefab instance's expansion (the same template leaf, placed several
    times, always carries the one uid it was authored with) -- an ordinary
    scene-tree uid, a group's or a :class:`~.nodes.PrefabNode` instance's, is
    unique by construction and never hits this case.
    """

    def visit(
        node: Node,
        path: tuple[int, ...],
        owner: int,
        world: np.ndarray,
        visible: bool,
        locked: bool,
        static: bool,
        ref: Ref | None,
        material: gltf.Material | None,
        prefab: str,
        dangling: bool,
    ) -> None:
        if node.uid == uid:
            raise _Found(
                Placed(
                    node, path, owner, world, visible, locked, static,
                    ref, material, prefab, dangling,
                )
            )

    try:
        walk(doc, visit, include_hidden=True, expand_prefabs=True)
    except _Found as found:
        return found.placed
    return None


def _box_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The eight corners of an axis-aligned box, as rows of an (8, 3) array."""
    xs = (lo[0], hi[0])
    ys = (lo[1], hi[1])
    zs = (lo[2], hi[2])
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype="f8")


def world_bounds(
    doc: MasonDoc, source: Any, uids: Any = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """``(min, max)`` in world space over ``uids`` (or the whole scene).

    The one place a :class:`~.refs.GeometrySource` appears in this module,
    because a bound needs actual geometry -- everything else here is pure.
    ``uids`` filters by :attr:`Placed.owner`, not by node uid, matching
    :func:`resolved_for`'s "picking is per ref, selection is per owner" rule:
    asking for the bounds of a selected prefab instance means every leaf
    under it, addressed by the one uid the outliner actually has a row for.

    A node with no ``ref`` (a group is never reached here at all; a light, a
    camera, an unexpanded prefab instance) contributes nothing. **Terrain
    contributes nothing either, on purpose** -- its bounds come from
    ``doc.terrain``'s own extent and heights, not from ``source``, since a
    terrain's geometry is generated from the singleton, never resolved
    through a :class:`~.refs.GeometrySource`.

    ``include_hidden=True`` throughout: this is a geometry query over
    whatever set of owners the caller named, not a "what is currently drawn"
    question -- a caller framing a hidden selection wants its true extent,
    and a caller wanting only what is visible filters ``uids`` down to that
    before calling.

    Returns ``None`` when nothing in the set has resolved a box yet (an asset
    still parsing in the background, say) -- that is not an error, it is "no
    picture to frame around yet".
    """
    wanted = None if uids is None else {int(u) for u in uids}
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    for placed in resolve(doc, include_hidden=True):
        if wanted is not None and placed.owner not in wanted:
            continue
        if placed.ref is None:
            continue
        box = source.box(placed.ref)
        if box is None:
            continue
        box_lo, box_hi = box
        corners = _box_corners(box_lo, box_hi)
        world_corners = corners @ placed.world[:3, :3].T + placed.world[:3, 3]
        c_lo = world_corners.min(axis=0)
        c_hi = world_corners.max(axis=0)
        lo = c_lo if lo is None else np.minimum(lo, c_lo)
        hi = c_hi if hi is None else np.maximum(hi, c_hi)
    if lo is None or hi is None:
        return None
    return lo, hi


# --- the per-draw node proxy -------------------------------------------------
#
# The one thing Mason must do differently from Clay here: Clay's GPU cache
# is one entry per
# *object*, so ``clay_view._composite`` writing ``node.world = world`` on the
# cached entry's own ``gltf.Node`` is sound -- one object, one write, one
# reader. Mason's cache is keyed on the *ref* (item 1: five hundred instances
# of one asset are one upload), so N placements of one ref share one cached
# ``gltf.Node``, and a naive port of Clay's composite would write ``.world``
# on that single shared object N times a frame -- every instance drawing at
# wherever the *last* write left it. ``DrawNode`` and ``NodePool`` exist
# because that bug is silent: nothing raises, nothing crashes, the frame
# renders, and N-1 objects are simply in the wrong place.


class DrawNode:
    """A per-draw stand-in for one instance's placement.

    Carries exactly the two attributes ``Renderer._draw_model`` reads off a
    node in the composite path -- ``world`` (composed into ``model_matrix @
    node.world`` per draw call) and ``skin`` (looked up as ``gpu.palette(node)``
    only when a primitive is skinned; Mason places no skinned meshes, so this
    is always ``None``, and it is carried anyway so a ``DrawNode`` answers
    every attribute a real ``gltf.Node`` would be asked for in that path
    without this module importing the renderer to check). ``__slots__``
    because a scene with tens of thousands of placements pools tens of
    thousands of these a frame; a ``__dict__`` on each would be pure waste.
    """

    __slots__ = ("world", "skin")

    def __init__(self, world: np.ndarray | None = None) -> None:
        self.world: np.ndarray = world if world is not None else m3.identity()
        self.skin: int | None = None


class NodePool:
    """Pools :class:`DrawNode` objects across frames.

    ``frame()`` resets the high-water mark to zero; ``node(world)`` hands back
    the next pooled proxy, growing the backing list only the first time a
    frame asks for more than any frame before it has. A scene of a thousand
    placements then costs a thousand allocations exactly once, not once a
    frame -- the pool is what makes paying for :class:`DrawNode` at all
    affordable, and the distinctness :class:`DrawNode` gives each instance is
    what makes the fix correct; neither one alone is enough.

    **Reuse across frames is deliberately not something a caller may rely
    on.** A ``DrawNode`` handed back last frame may be the very object handed
    back again this frame, for a different placement and a different
    ``world`` -- that is what pooling *is* -- so a caller that stashed a
    reference across a frame boundary is holding a proxy this pool considers
    free to overwrite. The composite is built and consumed within the same
    frame; nothing here promises more than that.
    """

    def __init__(self) -> None:
        self._pool: list[DrawNode] = []
        self._used = 0

    def frame(self) -> None:
        self._used = 0

    def node(self, world: np.ndarray) -> DrawNode:
        if self._used < len(self._pool):
            proxy = self._pool[self._used]
            proxy.world = world
        else:
            proxy = DrawNode(world)
            self._pool.append(proxy)
        self._used += 1
        return proxy

