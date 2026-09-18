"""Mason's undo steps: what changed in a scene, and how to put it back.

The engine underneath these -- ``Edit``, ``CompoundEdit``, ``UndoStack``, the
serial counter and the byte budget -- is ``studio/undo.py``, already shared by
the raster editor and by Clay. This is the third consumer, with a
``NodeAddEdit`` where Clay has an ``ObjectAddEdit`` and a ``TerrainEdit``
where the raster editor has a ``PatchEdit``. Two rules travel down from the
engine unchanged, restated here in Mason's own terms because they are as
load-bearing for a scene tree as for a layer stack or a mesh.

**Every edit addresses its node by uid, never by index.** An index stops
naming the thing it named the instant anything reorders, and the outliner can
reorder at any time -- including between an edit being recorded and an undo
being asked for. So a :class:`TransformEdit` carries ``node_uid`` and looks
the node up through the document, exactly as :class:`NodePropsEdit`,
:class:`RefEdit` and the rest do. **The two exceptions that prove the rule
are :class:`NodeAddEdit` and :class:`NodeRemoveEdit`**, and they are not a
lapse from it: a ``(parent_uid, index)`` pair on either one says only *where
to put the node back*, on the one side of the toggle that has no node to
address by uid yet (an undone add) or no longer has a position to be found at
(a done remove). Finding the node to detach it is still done by uid, on the
document's own tree, exactly like every other edit here -- the pair is a
destination, never a lookup key.

**An edit owns its data.** ``cost`` is what eviction is driven by, and a numpy
view reports its own small ``nbytes`` while pinning the whole base array
alive -- ``studio/undo.py``'s own docstring names this trap and
``mason/terrain.py``'s docstring names it again for exactly the array
:class:`TerrainEdit` carries. Every array field below is copied in
``__post_init__``, the same place Clay's ``TransformEdit`` copies its own
three.

**Undo and redo go through document hooks, never by reaching into
``MasonDoc``'s lists or a node's fields directly.** ``_attach_node``,
``_detach_node``, ``_relocate``, ``_apply_transform``, ``_apply_props``,
``_apply_ref``, ``_blit_terrain``, ``_apply_terrain``,
``_apply_terrain_config`` and ``_apply_prefab`` are ``MasonDoc``'s own methods
(``document.py``, not this module) -- the pairing ``plotter/_map_layers.py``
calls "public method plus private hook", restated here because it is the
same shape for the same reason: the public method is the only thing that
records a step, so the hook
an undo replays can never push a step of its own. And -- ``_map_layers.py``'s
own warning about ``_apply_layer_props`` applies to every hook below just as
sharply -- **every key an edit can carry has to be assigned inside the hook
it calls**. A field recorded in ``NodePropsEdit.before`` but not read back by
``_apply_props`` is a field the properties panel can set and an undo can
never take back; the trap is silent, because the edit still reports success.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from .....core.undo import Edit
from .....kernels.geom3d import gltf
from .nodes import Node, _props_bytes, subtree_bytes
from .refs import Ref
from .terrain import Rect, Terrain


def _own(value: Any) -> np.ndarray:
    """A copy of a transform array, so nothing the caller still holds can
    rewrite a step already sitting on the undo stack -- ``Node.__post_init__``'s
    own reason, restated for the edit that stores a second set of these."""
    return np.array(value, dtype="f8", copy=True)


def _own_heights(value: Any) -> np.ndarray:
    """A copy of a height-field sub-array, f4 like ``Terrain.heights`` itself.

    Not ``_own``'s f8: a :class:`TerrainEdit`'s ``before``/``after`` are
    slices of ``terrain.heights``, which is f4 (see ``terrain.Terrain``'s own
    docstring on why), and re-widening them to f8 here would silently double
    the very bytes ``cost`` exists to report honestly.

    ``copy=True`` unconditionally, **not** a bare ``np.ascontiguousarray``:
    that function hands back the very same array when it is already
    contiguous and already f4 -- which a brush's own returned sub-array
    always is -- so it would own nothing at all in the one case this
    constructor exists to guard, exactly the trap ``Terrain.__post_init__``'s
    own docstring names for "an array that already happens to be f4 and
    contiguous is not proof it is not a slice of something the caller still
    holds".
    """
    return np.array(value, dtype=np.float32, copy=True)


#: A terrain config value is one of the three fields ``Terrain`` itself
#: carries beside its heights -- see ``terrain.Terrain`` and
#: ``MasonDoc.set_terrain_config``. Named so :class:`TerrainConfigEdit` can
#: say precisely what it holds rather than falling back to ``Any``, which is
#: the reason this module reaches for ``gltf`` at all.
TerrainConfigValue = float | gltf.Material


@dataclass
class NodeAddEdit(Edit):
    """A node (and everything under it) arriving in the tree.

    ``node`` is the object itself, not a copy -- re-inserting the very
    instance the document held is what lets an edit recorded *before* this
    one (a transform drag on a child, say) still find its target by uid after
    a redo, exactly as :class:`~.clay.edits.ObjectAddEdit`'s own docstring
    argues. ``parent_uid``/``index`` name only where to reattach it;
    detaching it again is always by uid, through :meth:`MasonDoc._detach_node`.

    Costed at :func:`~.nodes.subtree_bytes` either way: while this step is
    *undone* the whole subtree is alive only because this edit holds it, and
    while it is *done* the document holds it instead -- the node is alive on
    one side of the toggle or the other, never both and never neither, so one
    cost function answers for both directions.
    """

    parent_uid: int | None
    index: int
    node: Node

    def __post_init__(self) -> None:
        self.cost = subtree_bytes(self.node)

    def undo(self, doc: Any) -> None:
        doc._detach_node(self.node.uid)

    def redo(self, doc: Any) -> None:
        doc._attach_node(self.node, self.parent_uid, self.index)


@dataclass
class NodeRemoveEdit(Edit):
    """The mirror of :class:`NodeAddEdit`: a node (and its subtree) leaving.

    Holds the same node object the tree held, for the same reason
    :class:`~.clay.edits.ObjectRemoveEdit` does -- a mesh edit or a rename
    recorded on a child before the removal must still find that exact object,
    by uid, when the removal is undone.
    """

    parent_uid: int | None
    index: int
    node: Node

    def __post_init__(self) -> None:
        self.cost = subtree_bytes(self.node)

    def undo(self, doc: Any) -> None:
        doc._attach_node(self.node, self.parent_uid, self.index)

    def redo(self, doc: Any) -> None:
        doc._detach_node(self.node.uid)


@dataclass
class NodeMoveEdit(Edit):
    """One node's place in the tree, before and after -- reparent and reorder
    as the single gesture a drag in the outliner actually is.

    Costs nothing: two ``(parent_uid, index)`` pairs, and the node never
    leaves the document -- ``ObjectMoveEdit``'s own reason, one dimension over.
    """

    node_uid: int
    before: tuple[int | None, int]
    after: tuple[int | None, int]

    def undo(self, doc: Any) -> None:
        doc._relocate(self.node_uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._relocate(self.node_uid, self.after)


@dataclass
class TransformEdit(Edit):
    """Translation, rotation and scale as one step, at near-zero cost.

    One step rather than three, for ``clay.edits.TransformEdit``'s own reason:
    a gizmo drag moves more than one of the three at once, and splitting that
    across three Ctrl+Z presses would show the user a pose the node was never
    actually in.
    """

    node_uid: int
    before: tuple[np.ndarray, np.ndarray, np.ndarray]
    after: tuple[np.ndarray, np.ndarray, np.ndarray]

    def __post_init__(self) -> None:
        self.before = tuple(_own(v) for v in self.before)  # type: ignore[assignment]
        self.after = tuple(_own(v) for v in self.after)  # type: ignore[assignment]
        self.cost = int(sum(v.nbytes for v in (*self.before, *self.after)))

    def undo(self, doc: Any) -> None:
        doc._apply_transform(self.node_uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_transform(self.node_uid, self.after)


@dataclass
class NodePropsEdit(Edit):
    """Name, visibility, lock, static, material override, user properties --
    anything about a node that is neither its place in the tree nor its
    transform.

    Costed by :func:`~.nodes._props_bytes` -- the local copy already sitting
    in ``nodes.py`` for exactly this, reused rather than written a third time
    (``clay/edits.py``'s own ``_props_bytes`` is the first). Shallow and
    approximate for the same reason it is there: a name or a visibility flag
    costs nothing worth measuring, and the one field that can actually grow
    -- a nested properties dict -- is what it is honest about. A
    ``material`` override reaching this through ``set_props`` is sized the
    same shallow way; see :class:`TerrainConfigEdit` for why that is an
    accepted, named trade rather than an oversight.
    """

    node_uid: int
    before: dict[str, Any]
    after: dict[str, Any]

    def __post_init__(self) -> None:
        self.cost = _props_bytes(self.before) + _props_bytes(self.after)

    def undo(self, doc: Any) -> None:
        doc._apply_props(self.node_uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_props(self.node_uid, self.after)


def _ref_bytes(ref: Ref | None) -> int:
    """What a reference costs a step: its own small fields, shallowly.

    Never the geometry it names -- resolving a :class:`~.refs.Ref` into
    triangles is exactly the one thing this package refuses to do (see
    ``refs.py``'s module docstring), so a step recording one cannot be
    charged for bytes it never touches, only for the job id or generator name
    and the handful of numbers the reference itself carries.
    """
    return 0 if ref is None else sys.getsizeof(ref)


@dataclass
class RefEdit(Edit):
    """A mesh node's source reference, before and after -- what Relink and
    Remove (from the missing-reference list) and re-pointing a placed
    primitive at a different generator all push."""

    node_uid: int
    before: Ref | None
    after: Ref | None

    def __post_init__(self) -> None:
        self.cost = _ref_bytes(self.before) + _ref_bytes(self.after)

    def undo(self, doc: Any) -> None:
        doc._apply_ref(self.node_uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_ref(self.node_uid, self.after)


@dataclass
class TerrainEdit(Edit):
    """One sculpt gesture: the rect it touched, and only that rect.

    **This is the edit the plan singles out.** ``before``/``after`` are the
    affected sub-arrays alone, never the whole height field -- a 256-side
    terrain is 256 KB of heights, and recording the full array on every dab
    would make eight sculpt strokes evict an hour of somebody else's work
    from a 192 MB budget for no reason but a brush radius of four cells. That
    is also why the cost below is exactly the two sub-arrays' ``nbytes`` and
    nothing else: ``cost`` drives eviction, and a step that claimed the whole
    terrain's bytes while holding a few kilobytes would make ``UNDO_BYTES``
    stop meaning anything.
    """

    rect: Rect
    before: np.ndarray
    after: np.ndarray

    def __post_init__(self) -> None:
        self.before = _own_heights(self.before)
        self.after = _own_heights(self.after)
        self.cost = int(self.before.nbytes + self.after.nbytes)

    def undo(self, doc: Any) -> None:
        doc._blit_terrain(self.rect, self.before)

    def redo(self, doc: Any) -> None:
        doc._blit_terrain(self.rect, self.after)


@dataclass
class TerrainSwapEdit(Edit):
    """The document's terrain installed, replaced, or lifted back out --
    whether a terrain exists *at all*, never its heights (:class:`TerrainEdit`)
    or its size and material (:class:`TerrainConfigEdit`).

    This one exists because it was missing: a ``set_terrain`` that mutated
    ``MasonDoc.terrain`` directly and pushed nothing meant a document's *only*
    reference to a height field was one attribute, and clearing it -- one
    ordinary click -- discarded whatever was sculpted into it with no way for
    Ctrl+Z to get it back. A node's arrival or departure gets
    :class:`NodeAddEdit`/:class:`NodeRemoveEdit` for exactly this structural
    reason; a terrain's did not, and a twenty-minute sculpt is exactly the
    kind of work ``undo.UNDO_HARD_BYTES``'s own docstring says an ordinary
    gesture must never be able to erase outright.

    **Holds the ``Terrain`` instances themselves, never a copy of either
    side.** ``Terrain.__post_init__`` already forces every instance to own
    its own ``heights`` array (``terrain.Terrain``'s own docstring), so a
    second copy here would duplicate bytes an object already owns for no
    reason but this step also existing -- :class:`NodeAddEdit`'s own argument
    for holding a node rather than copying it, one type over.

    **Costed at the whole height field, on both sides that exist, and that is
    not a contradiction of :class:`TerrainEdit`'s rect-only rule.** A reshape
    touches a region of an array that keeps existing either way, so billing
    it for only the rect it actually touched is what keeps that edit's cost
    honest. A swap is a different shape of change: the *entire* array arrives
    or leaves the document in one step, so the entire array is what a step
    reversing that arrival or departure has to be able to restore -- and
    under-billing it would let ``UNDO_BYTES`` evict other, genuinely cheaper
    work to make room for a step that was never as small as it claimed.
    """

    before: Terrain | None
    after: Terrain | None

    def __post_init__(self) -> None:
        self.cost = sum(t.heights.nbytes for t in (self.before, self.after) if t is not None)

    def undo(self, doc: Any) -> None:
        doc._apply_terrain(self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_terrain(self.after)


@dataclass
class TerrainConfigEdit(Edit):
    """A terrain's size or material, before and after -- never its heights,
    which is :class:`TerrainEdit`'s business alone.

    Costed with the same shallow accounting :class:`NodePropsEdit` uses,
    because the shape is the same: ``size_x``/``size_z`` are floats worth
    nothing to measure, and ``material`` is a shared ``gltf.Material`` object
    -- sized shallowly rather than by its textures, which is the same
    accepted trade ``clay/document.py`` already makes for a material held by
    identity rather than copied (see ``nodes.copy_subtree``'s own argument for
    why a shared material is not this package's to charge twice).
    """

    before: dict[str, TerrainConfigValue]
    after: dict[str, TerrainConfigValue]

    def __post_init__(self) -> None:
        self.cost = _props_bytes(self.before) + _props_bytes(self.after)

    def undo(self, doc: Any) -> None:
        doc._apply_terrain_config(self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_terrain_config(self.after)


@dataclass
class PrefabEdit(Edit):
    """A prefab template defined, redefined or removed.

    ``before``/``after`` are template roots -- never in the scene tree, see
    ``document.py``'s ``MasonDoc.prefabs`` -- and either side may be ``None``:
    ``before is None`` is a fresh definition, ``after is None`` is a removal.
    Costed over whichever sides actually exist, by the same
    :func:`~.nodes.subtree_bytes` a placed node's own add/remove edit uses,
    because a template is exactly that: a subtree, just one the resolver
    reaches through a name instead of a parent's ``children`` list.
    """

    name: str
    before: Node | None
    after: Node | None

    def __post_init__(self) -> None:
        self.cost = sum(subtree_bytes(n) for n in (self.before, self.after) if n is not None)

    def undo(self, doc: Any) -> None:
        doc._apply_prefab(self.name, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_prefab(self.name, self.after)
