"""The scene graph: six node classes, not one class with a ``kind`` string.

``plotter/_map_model.py`` chose four layer classes over one class with a mode
field, and Mason makes the same choice for the same reason plus one more this
document has that a tile map does not: a light carries colour, intensity,
range and cone angles; a camera carries a vertical field of view and a near
and far plane; a mesh node carries a source reference and a material
override. A single class with a ``kind`` discriminator would make every one
of those fields optional on every node regardless of what it names, and every
consumer that cares -- the resolver, the properties panel, the exporter --
would open with an ``if kind == ...`` chain instead of the
``isinstance(node, GroupNode)`` ``plotter/scene.py`` already uses for exactly
this question.

**Children, not parent.** A node holds ``children``; nothing holds ``parent``.
The tree *is* the storage -- sibling order is meaningful, because it is both
the export order and the outliner row order -- and a ``parent`` pointer would
be a second copy of the one fact the ``children`` list already states, free to
disagree with it the moment either is updated and the other is not.
Acyclicity is not one guarantee but three, stacked:

1. **By construction.** A node object appears in exactly one ``children``
   list, because nothing in this module (or ``document.py``) ever *sets* a
   parent -- moving a node means removing it from one list and appending it
   to another, the same object, so there is no operation that could leave it
   reachable from two places at once.
2. **By refusal.** ``reparent`` (in ``document.py``, not this module) raises
   before it lets a node become its own descendant's child -- it is the
   caller of :func:`contains` this module's docstring for that function names.
3. **By backstop.** Every walk here carries an ancestor-path set and
   :data:`MAX_DEPTH` ceiling, mirroring the ``seen``/range guard
   ``gltf.Model.update_world`` already applies to a hand-supplied GLB's node
   graph. A ``.wscn`` can be hand-edited same as a GLB can, and this backstop
   runs on the frame thread every time the resolver walks the scene -- so
   unlike rule 2, it does not raise. **Refusal is loud at the door and quiet
   during the walk.** Rule 2 is the door: it is where a user asking to reparent
   a node into its own descendant gets told no, in words, before anything is
   touched. Rule 3 exists only for whatever gets past that door by being
   edited outside the app entirely, and the whole point of *this* guard is
   that such a file costs only the branch that is actually broken -- ``walk``
   skips a node already on its own ancestor path and stops descending past the
   ceiling, and lets the rest of the tree draw normally, rather than taking
   the viewport down over a corruption nothing in this frame asked about.
   :func:`copy_subtree` is the one exception, and it raises on purpose: a copy
   is an authoring operation the user asked for this frame, not a recurring
   traversal, so refusing outright is affordable and more honest than handing
   back a truncated copy of a tree the user never saw was broken.

**``LightNode`` uses KHR_lights_punctual's own field names and units;
``CameraNode`` uses glTF's.** ``clay/document.py`` makes this argument for why
a Clay material *is* a ``gltf.Material`` rather than a parallel type with a
conversion function, and it applies here without qualification: the export is
the definition, and a second type for "what a light is" buys nothing but a
place for the two to drift apart. In particular there is no ``direction``
field on ``LightNode`` to disagree with its rotation -- KHR_lights_punctual
defines a light's direction as its node's local ``-Z`` axis, so the rotation
already says everything a direction field would, and a second field stating
it again is exactly the kind of parallel fact rule 3 above exists to prevent
one level up the tree.
"""

from __future__ import annotations

import itertools
import math
import sys
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from ..viewer import gltf
from ..viewer import math3d as m3
from .refs import Ref

# A process-wide counter behind one lock, exactly the shape
# ``clay/document.py``'s ``new_uid``/``reserve_uid`` already use -- restated
# here in Mason's own terms because a scene's undo step is written against a
# node's uid in precisely the way an object's undo step is written against
# ``Obj.uid``: the uid is the *address*, not a display number, and every rule
# below follows from that.
_uids = itertools.count(1)
_uid_lock = threading.Lock()


def new_uid() -> int:
    """Mint a node uid. Never reused for the life of the process.

    A recycled uid is not a cosmetic problem: an undo step (``NodeAddEdit``,
    ``TransformEdit``, ...) names the node it changed by uid, and a reissued
    number would let a step recorded against a node the user deleted an hour
    ago land back on whatever new node happens to wear that number now. The
    counter is process-wide rather than per-document so the same guarantee
    holds across closing one scene and opening another in the same session.
    """
    with _uid_lock:
        return next(_uids)


def reserve_uid(uid: int) -> None:
    """Raise the uid floor so :func:`new_uid` never hands back ``uid`` or below.

    Called from ``serialize.read_wscn`` (a task thread) as a saved scene's
    nodes are restored, each carrying the uid it was saved with. Without this,
    a freshly-created node could be minted the same uid a restored one already
    holds, and the first edit to either would resolve against whichever
    ``walk`` happens to find first. Deliberately monotonic -- the counter only
    ever moves forward -- so opening a small scene after a large one cannot
    undo the protection the large one already raised the floor to.
    """
    global _uids
    with _uid_lock:
        _uids = itertools.count(max(int(uid) + 1, next(_uids)))


#: The ceiling every walk below refuses past. A human-authored outliner nests
#: a handful of groups deep in ordinary use -- nothing like this needs sixty
#: levels of grouping -- so this is not tuned to the deepest legitimate scene,
#: it is tuned to match this repo's other structural ceiling of the same
#: shape: ``undo.UNDO_MAX_DEPTH`` is also 64. That leaves every recursive walk
#: here two orders of magnitude under Python's default 1000-frame recursion
#: limit, so a hand-edited ``.wscn`` that nests deeper than any real outliner
#: ever would is refused by name rather than by a bare ``RecursionError`` on
#: the frame thread.
MAX_DEPTH = 64


@dataclass
class Node:
    """What every node genuinely has, regardless of what kind it is.

    ``rotation`` is XYZW, the one quaternion convention this project uses
    (``viewer.math3d``'s own docstring says so); a WXYZ default here would be
    a rotation that renders as a plausible orientation while being wrong on
    every axis, which is exactly the kind of bug a test should catch before a
    user does.

    **A caller changes a transform by rebinding, never by writing through.**
    ``node.translation = new_array`` is how a move happens; ``node.translation[:]
    = new_array`` or ``node.translation += delta`` is not, even though numpy is
    happy to do either. :meth:`local` memoizes its matrix against the identity
    of the three arrays, exactly the rule Clay's own viewport cache
    (``clay_view._world``) states for the same reason: a rebind changes which
    object ``self.translation`` names, so the memo notices; a write-through
    changes the numbers inside the object the memo already checked and moved
    on from, so it does not. Break the rule and ``local()`` keeps handing back
    the matrix from before the write -- the node draws where it used to be
    while the properties panel, reading the arrays directly, reports where it
    now is. Every mutator in this codebase that touches a transform already
    rebinds (``__post_init__`` here, ``document._apply_transform``), so this
    is a rule the document already followed; ``local()`` is what now depends on
    it being followed.
    """

    uid: int
    name: str = ""
    translation: np.ndarray = field(default_factory=m3.vec3)
    rotation: np.ndarray = field(default_factory=m3.quat_identity)
    scale: np.ndarray = field(default_factory=lambda: m3.vec3(1.0, 1.0, 1.0))
    children: list[Node] = field(default_factory=list)
    visible: bool = True
    locked: bool = False
    static: bool = False
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Own the transform arrays rather than aliasing whatever was passed
        # in: a caller that keeps mutating the array it handed over (a drag
        # gesture reusing one buffer, say) would silently rewrite a step
        # already sitting on the undo stack, and a numpy *view* would misreport
        # its own ``nbytes`` to whatever budget is costing it against. The
        # same argument applies to ``properties`` -- it is the one field here
        # that is a mutable container a caller might keep a reference to, so
        # it is copied on the way in exactly like the arrays are.
        self.translation = np.array(self.translation, dtype="f8", copy=True)
        self.rotation = np.array(self.rotation, dtype="f8", copy=True)
        self.scale = np.array(self.scale, dtype="f8", copy=True)
        self.properties = dict(self.properties)
        # Not a dataclass field on purpose: ``copy_subtree`` reconstructs a
        # node from ``dataclasses.fields()``, and a fresh copy is supposed to
        # start with no cache of its own -- this line runs again for it and
        # leaves the field this method actually reads unset until the first
        # call, exactly as a brand-new node needs.
        self._local_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    def local(self) -> np.ndarray:
        """This node's local transform matrix -- ``m3.compose(t, r, s)``,
        memoized against its own three arrays.

        Profiled resolving a 2,000-node scene (``scene.py``): rebuilding every
        node's matrix from its TRS every frame was 64% of the whole resolve
        (``N x node.local()``, 11.08 ms of 17.18 ms), the overwhelming majority
        of it in ``m3.compose``/``quat_to_mat4`` -- and the common case is an
        orbiting camera with a scene that has not moved at all, recomputing
        the same matrix every one of those frames. The memo is keyed on
        ``self.translation``/``rotation``/``scale`` being the *same objects*
        as last call (see the class docstring's rebind rule) rather than on
        their contents, so a hit costs three ``is`` checks instead of a
        quaternion-to-matrix conversion.

        **The cache entry holds the three arrays themselves, not their
        ``id()``s.** An id is an address CPython is free to hand to a
        *different* object the moment the old one is garbage collected, so a
        memo keyed on bare ids can validate a hit against the wrong array --
        the exact bug ``viewer/picking.cached_bvh`` and
        ``mason/terrain.terrain_mesh`` both already had to write down a
        guard against. Holding the arrays alongside the matrix and comparing
        with ``is`` compares against objects this cache is itself keeping
        alive, so there is no address left to recycle out from under it. This
        pins nothing beyond what the node already owns -- they are its own
        ``translation``/``rotation``/``scale``, held twice by the same node,
        not retained on anyone else's behalf.

        Returns a **copy**, not the cached array: measured at roughly 0.2 us
        for a 4x4 ``f8`` (``m.copy()``) against ``compose``'s own ~5.5 us,
        under 4% of what a miss costs and cheap insurance against a caller
        that writes into what it was handed and corrupts every future hit.
        """
        cached = self._local_cache
        if cached is not None:
            t, r, s, matrix = cached
            if t is self.translation and r is self.rotation and s is self.scale:
                return matrix.copy()
        matrix = m3.compose(self.translation, self.rotation, self.scale)
        self._local_cache = (self.translation, self.rotation, self.scale, matrix)
        return matrix.copy()

    def trs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The three live transform arrays -- **rebind them to change the
        transform, never write through them.** See the class docstring: a
        write-through leaves :meth:`local` handing back a stale matrix
        because nothing about the objects named by ``self.translation`` /
        ``rotation`` / ``scale`` changed, only the numbers inside one of them.
        """
        return self.translation, self.rotation, self.scale


@dataclass
class GroupNode(Node):
    """A node that exists only to hold others. Nothing of its own."""


@dataclass
class MeshNode(Node):
    """A placed mesh: where its geometry comes from, and its material override.

    ``ref`` is ``None`` for a mesh node that has not been given a source yet
    (freshly created by a tool that is about to ask what to place); resolving
    it into triangles is :class:`~.refs.GeometrySource`'s job, never this
    module's. ``material`` is an override the node itself sets -- if it is
    ``None``, the resolved geometry's own material applies unchanged.
    """

    ref: Ref | None = None
    material: gltf.Material | None = None


#: KHR_lights_punctual's own three kinds, verbatim -- there is no fourth, and
#: no fourth name for one of these three either, since a parallel vocabulary
#: is exactly what this module's docstring argues against.
LIGHT_KINDS = ("directional", "point", "spot")


@dataclass
class LightNode(Node):
    """A light, in KHR_lights_punctual's own fields and units.

    Direction is not a field here -- see the module docstring -- because the
    extension already defines a light's direction as its node's local ``-Z``,
    so the rotation this class inherits from :class:`Node` is the only place
    that fact lives.
    """

    kind: str = "point"
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    range: float = 0.0
    inner_cone_angle: float = 0.0
    outer_cone_angle: float = math.pi / 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind not in LIGHT_KINDS:
            raise ValueError(
                f"a light's kind must be one of {LIGHT_KINDS}, not {self.kind!r}"
            )


@dataclass
class CameraNode(Node):
    """A camera, in glTF's own perspective-camera fields and units (radians,
    metres)."""

    yfov: float = math.radians(60.0)
    znear: float = 0.1
    zfar: float = 1000.0


@dataclass
class PrefabNode(Node):
    """An instance of ``MasonDoc.prefabs[template]``.

    Carries its own uid, name, transform and flags, and deliberately no
    per-child overrides -- see the plan's "prefabs and instancing": the escape
    hatch for a placement that needs to diverge from its template is
    ``unpack_instance`` (in ``document.py``, not this module), which replaces
    the instance with a fresh, editable copy via :func:`copy_subtree`.
    """

    template: str = ""


@dataclass
class TerrainNode(Node):
    """The outliner row for ``MasonDoc.terrain``: a transform, visibility, and
    a material override for the one document-singleton height field.

    Carries no heightfield of its own -- see the plan's "terrain: a document
    singleton plus one node" for why the array lives once, off the tree,
    rather than being copied into every node that might refer to it.
    """

    material: gltf.Material | None = None


def walk(roots: Sequence[Node]) -> Iterator[tuple[Node, Node | None, int, int]]:
    """``(node, parent, index, depth)`` depth-first pre-order over ``roots``.

    Mirrors ``plotter/_map_layers._walk``'s shape, with one difference forced
    by the design this module's docstring argues for: a layer's parent is
    named by uid there because ``LayerOps`` always has the whole document to
    look one up in, but nothing here is willing to assume a document exists
    just to walk a bare list of nodes (:func:`copy_subtree` and
    :func:`subtree_bytes` both call this over a subtree with no document in
    reach at all) -- so ``parent`` is the node **object**, the one thing this
    function can hand back for free.

    Guards against the corrupt-file case rule 3 of the module docstring
    promises, and does so **quietly**: ``path`` is every node on the current
    root-to-node ancestry (by ``id()``, since two distinct nodes may
    legitimately be ``==``-indistinguishable dataclasses with identical
    fields). A node already on that path is skipped rather than descended
    into again -- it is the one branch that is actually broken, and the rest
    of the tree still walks -- and :data:`MAX_DEPTH` stops descent the same
    way once a chain runs past it without ever repeating a node at all. This
    runs every frame the resolver walks the scene, so unlike ``reparent``'s
    refusal (rule 2) or :func:`copy_subtree`'s (below), there is no exception
    to raise here and no one to raise it to.

    ``path`` is one mutable ``set`` threaded through the whole traversal
    rather than a new ``frozenset`` built at every node: this is called once a
    frame over the entire scene, and the classic path-stack shape --
    ``add`` before descending, ``discard`` after -- gets the same ancestry
    check for one allocation instead of one per node.
    """
    yield from _walk(roots, None, 0, set())


def _walk(
    nodes: Sequence[Node], parent: Node | None, depth: int, path: set[int]
) -> Iterator[tuple[Node, Node | None, int, int]]:
    if depth > MAX_DEPTH:
        # Stop descending -- not raise. Whatever is above the ceiling already
        # walked and drew normally; only the over-deep branch itself is cut.
        return
    for index, node in enumerate(nodes):
        key = id(node)
        if key in path:
            # This node is its own ancestor. Skip it (and, by never
            # recursing into it, everything under it) rather than spin, and
            # move on to its siblings -- the corrupt branch costs itself.
            continue
        yield node, parent, index, depth
        path.add(key)
        yield from _walk(node.children, node, depth + 1, path)
        path.discard(key)


def contains(node: Node, uid: int) -> bool:
    """Is ``uid`` anywhere under ``node`` -- not counting ``node`` itself?

    This is what ``reparent`` (``document.py``, not this module) calls before
    it moves a node, to refuse making a node a child of its own descendant --
    rule 2 of the module docstring's acyclicity argument. Built on
    :func:`walk` over ``node.children`` alone, so it inherits the same
    ancestor-set and depth backstop rather than a second copy of it.
    """
    return any(child.uid == uid for child, _parent, _index, _depth in walk(node.children))


def copy_subtree(node: Node, *, fresh_uids: bool) -> Node:
    """A deep copy of ``node`` and everything under it.

    Every field is copied from the dataclass's own ``fields()`` rather than
    listed by hand -- the reason ``clay/edits.mesh_bytes`` gives for doing the
    same over ``Mesh``'s fields: a node subclass that gains a field later is
    copied correctly the day it is added, rather than the day someone notices
    an instance and its "copy" quietly sharing something they should not.

    ``children`` is rebuilt recursively and ``uid`` is handled separately;
    every other field is passed through to the new node's constructor
    unchanged, and :meth:`Node.__post_init__` does the rest: the three
    transform arrays and ``properties`` come back as fresh copies on *every*
    node in the subtree, new uids or not, because that copying happens on
    construction regardless of what triggered it.

    ``fresh_uids=True`` mints a new uid (through :func:`new_uid`) for every
    node in the subtree -- what ``unpack_instance`` and a duplicate command
    both need, since either one is creating nodes that must never collide
    with the ones they were copied from. ``fresh_uids=False`` keeps every uid
    exactly as it was -- what an undo step holding a just-removed subtree
    needs, since redoing the removal has to find the very node the document
    used to hold, by the same uid every other edit addresses it by.

    **A ``MeshNode``/``TerrainNode``'s ``material`` is shared by identity, not
    copied.** It passes through the generic field loop above unchanged, which
    is deliberate: the viewport's ``GpuMaterial`` cache and the GLB writer
    both de-duplicate a material by ``id()``, so copying one that nobody has
    actually edited would silently produce a second GPU upload and a second
    glTF material for a material that is, by every visible measure, still the
    same one.

    **Unlike** :func:`walk`, **this raises rather than skips** on a cycle or a
    chain past :data:`MAX_DEPTH`. A copy is an authoring operation the user
    asked for just now -- duplicate, unpack-instance -- not a per-frame
    traversal, so refusing outright costs nothing this frame did not already
    budget for, and it is more honest than silently handing back a copy that
    is missing a branch the user never found out was broken.
    """
    return _copy_subtree(node, fresh_uids=fresh_uids, seen=frozenset(), depth=0)


def _copy_subtree(
    node: Node, *, fresh_uids: bool, seen: frozenset[int], depth: int
) -> Node:
    if depth > MAX_DEPTH:
        raise ValueError(
            f"subtree nests nodes more than {MAX_DEPTH} deep, which is deeper than "
            "a copy here will follow"
        )
    if id(node) in seen:
        raise ValueError(f"node {node.uid} is its own ancestor; there is no subtree to copy")
    branch = seen | {id(node)}
    new_children = [
        _copy_subtree(child, fresh_uids=fresh_uids, seen=branch, depth=depth + 1)
        for child in node.children
    ]
    skip = ("uid", "children")
    kwargs = {f.name: getattr(node, f.name) for f in fields(node) if f.name not in skip}
    kwargs["uid"] = new_uid() if fresh_uids else node.uid
    kwargs["children"] = new_children
    return type(node)(**kwargs)


def _props_bytes(props: dict[str, Any]) -> int:
    """A node's ``properties``, priced the way ``clay/edits._props_bytes``
    prices ``ObjectPropsEdit``'s side of a props change: shallow, and honest
    about the one shape that can actually grow (a nested dict), rather than a
    byte-exact accounting nothing here needs. Restated locally instead of
    imported, because importing anything from ``clay`` at all is the one
    import this package exists to refuse.
    """
    total = 0
    for value in props.values():
        if isinstance(value, np.ndarray):
            total += value.nbytes
        elif isinstance(value, dict):
            total += _props_bytes(value)
        else:
            total += sys.getsizeof(value)
    return total


def subtree_bytes(node: Node) -> int:
    """What an undo step holding ``node`` and everything under it costs.

    Honest rather than exhaustive: the three owned f8 arrays every node
    carries, plus a shallow accounting of ``properties`` (see
    :func:`_props_bytes`). A ``MeshNode``'s ``ref`` and ``material`` are not
    counted -- a ``Ref`` is a handful of scalars and a ``gltf.Material`` is
    shared by identity with whatever else in the document already holds it
    (see :func:`copy_subtree`), so charging a removed node for it would tax
    the same bytes twice.
    """
    total = 0
    for n, _parent, _index, _depth in walk([node]):
        total += n.translation.nbytes + n.rotation.nbytes + n.scale.nbytes
        total += _props_bytes(n.properties)
    return total
