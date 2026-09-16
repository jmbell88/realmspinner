"""``MasonDoc`` -- the scene tree, the prefab table, the terrain singleton, the
material palette, the selection, and the history over all of them.

A Mason document is a *hierarchy*, unlike a Clay one -- see ``nodes.py``'s
module docstring for why the tree is the storage and nothing holds a parent
pointer. What is here is the layer built on top of that tree: the lookups
every mutator and every consumer shares (mirroring
``plotter/_map_layers.LayerOps``, which solved the identical problem for a
layer tree first), the ten undo edits' document-side hooks (mirroring that
same module's "public method plus private hook" pairing), the sculpt session
(mirroring ``plotter/_map_paint.py``'s stroke session, one dimension over),
and the state a save, an undo panel or a properties pane reads.

**This module never resolves a reference and never draws anything.** A
``MeshNode``'s ``ref`` is a job id or a generator name plus numbers; turning
either into triangles is ``mason/scene.py``'s resolver and the host's
``GeometrySource``, neither of which this module imports or needs to.

**Dirty is a comparison against ``history.head``, never a flag.** ``rev``
counts changes for anything that caches off the document -- a viewport's GPU
upload key, an outliner row list -- and an undo *is* a change, so a rev-based
check would call an undone document unsaved forever. ``ClayDoc`` and
``PlotterDoc`` both make this argument already; a third document getting it
wrong would be the one place in the app where Ctrl+Z back to a saved scene
still nags the user to save it again.

**Selection is not undoable.** Clicking a node in the outliner is not the
kind of work a stray click should cost an undo step to recover, and an
undoable selection would push a step that moves ``history.head`` -- so
looking at a different node would make a saved document ask to be saved
again. ``select`` still bumps ``rev``, because the viewport draws a selection
outline and has to know to redraw it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from ..undo import CompoundEdit, Edit, UndoStack
from ..viewer import gltf
from . import edits as ed
from . import nodes as nd
from . import scene as sc
from .nodes import Node
from .refs import Ref, ref_key
from .terrain import Rect, Terrain

# "The parent you already have." ``None`` is a real parent -- the root list --
# so a nullable default could not tell "move to the root" apart from "leave it
# where it is", and those are two different gestures. ``plotter/_map_layers.py``
# names this sentinel for the same reason.
_KEEP = object()


class MasonDoc:
    """A scene: its tree, its prefabs, its one terrain, its palette, its
    selection and its history.

    **This attribute contract is fixed.** ``scene.py`` (the resolver, landing
    after this module) is written against the names below, and renaming or
    restructuring any of them turns a sibling's work-in-progress into a guess
    about what changed underneath it::

        roots: list[Node]
        prefabs: dict[str, Node]
        terrain: Terrain | None
        materials: list[gltf.Material]
        missing: set[tuple]
        selection: set[int]
        selection_active: int | None
        history: UndoStack
        rev: int
        saved_head: int
        view: dict[str, Any]
    """

    def __init__(
        self,
        roots: Iterable[Node] | None = None,
        materials: Iterable[gltf.Material] | None = None,
    ) -> None:
        self.roots: list[Node] = list(roots or [])
        # Template subtrees a ``PrefabNode`` instances by name. Deliberately
        # not part of ``roots`` -- see :meth:`define_prefab`: a template is
        # not a thing anyone places directly, it is what an instance refers
        # to, the same distinction ``nodes.PrefabNode``'s own docstring draws.
        self.prefabs: dict[str, Node] = {}
        self.terrain: Terrain | None = None
        # The palette a node's ``material`` field can point at. Unlike Clay's
        # ``ClayDoc.materials``, nothing here indexes into this list -- a
        # node's material override holds the ``gltf.Material`` object itself
        # (``clay/document.py``'s own argument for materials-as-objects, which the
        # Mason programme restated rather than revisited) -- so this list
        # is a save/export convenience, not an addressing scheme.
        self.materials: list[gltf.Material] = list(materials or [])
        # ``ref_key``s the *host* has told this document it could not
        # resolve. This package never resolves anything itself (see the
        # module docstring), so nothing here ever adds to or clears this set
        # on its own; it only reports through :meth:`missing_refs`.
        self.missing: set[tuple[Any, ...]] = set()
        self.selection: set[int] = set()  # node uids; not undoable
        # The uid ``select`` was last confident was the one actually clicked
        # (see :meth:`select`) -- ``mason_view.MasonView.selection_centre``'s
        # "active" pivot reads this back. The 2026-09-16 audit found no such
        # record existed at all: "Active" silently fell back to whichever
        # selected node came first in the document's own tree-walk order,
        # not "the last node clicked" its own tooltip promises. May go stale
        # (point at a uid no longer in ``selection``, or at ``None``) --
        # ``selection_centre`` falls back to the median pivot when it does.
        self.selection_active: int | None = None
        self.history = UndoStack()
        # A change counter for anything that caches off the document.
        # Deliberately not what ``dirty`` is derived from; see the module
        # docstring.
        self.rev = 0
        self.saved_head = self.history.head
        # The saved camera. Opaque to this module -- Stage D writes it, and
        # nothing here reads a key out of it.
        self.view: dict[str, Any] = {}
        # The open sculpt session, or ``None``. Private working state, not
        # part of the attribute contract above: mirrors ``PaintOps._stroke``
        # in shape (a pre-session snapshot plus a growing union rect) for the
        # reason given at :meth:`begin_sculpt`.
        self._sculpt: dict[str, Any] | None = None

    # -- lookup --------------------------------------------------------------

    def walk(self) -> Iterator[tuple[Node, int | None, int, int]]:
        """``(node, parent_uid, index, depth)`` depth-first pre-order over the
        whole tree.

        Built on :func:`nodes.walk`, whose own docstring hands back the
        parent *object* because it has no document to look a uid up in; this
        method has one, so it restates the parent as a uid instead -- the
        shape ``plotter/_map_layers.LayerOps.walk`` already settled on for a
        tree with a document behind it.
        """
        for node, parent, index, depth in nd.walk(self.roots):
            yield node, (None if parent is None else parent.uid), index, depth

    def all_nodes(self) -> list[Node]:
        """Every node, depth-first, in the same order :meth:`walk` visits them."""
        return [entry for entry, _parent, _index, _depth in self.walk()]

    def node(self, uid: int) -> Node | None:
        found = self.locate(uid)
        return None if found is None else found[0]

    def locate(self, uid: int) -> tuple[Node, int | None, int] | None:
        """``(node, parent_uid, index)``, or ``None``. Every other lookup's engine."""
        for entry, parent_uid, index, _depth in self.walk():
            if entry.uid == uid:
                return entry, parent_uid, index
        return None

    def parent_uid_of(self, uid: int) -> int | None:
        found = self.locate(uid)
        if found is None:
            raise KeyError(f"no node {uid}")
        return found[1]

    def index_of(self, uid: int) -> int:
        """Where a node sits **within its own parent's list** -- not a
        position in some flattened enumeration. An index is only ever used to
        reorder, and reordering happens among siblings, exactly the argument
        ``LayerOps.index_of`` makes for a layer tree.
        """
        found = self.locate(uid)
        if found is None:
            raise KeyError(f"no node {uid}")
        return found[2]

    def children_of(self, parent_uid: int | None) -> list[Node]:
        """The live list one node's children are stored in.

        ``None`` is the root list. **The list itself, not a copy** -- this is
        the funnel every mutator below inserts into and deletes from, the
        same rule ``LayerOps.children_of`` states for the same reason: handing
        back a copy would make every one of them silently do nothing.
        """
        if parent_uid is None:
            return self.roots
        parent = self.node(int(parent_uid))
        if parent is None:
            raise KeyError(f"no node {parent_uid}")
        return parent.children

    def _require(self, uid: int) -> Node:
        node = self.node(uid)
        if node is None:
            raise KeyError(f"no node {uid}")
        return node

    # -- structure -------------------------------------------------------------

    def _check_max_placed(self, adding: int) -> None:
        """Refuse growing past :data:`sc.MAX_PLACED` **before** anything is
        attached. The 2026-09-14 audit's mason-01 found this gap in what is
        now :meth:`add_nodes` alone; the 2026-09-15 audit's mason-01 (left
        open on two doors) found :meth:`add_node` -- every single placement,
        looped by ``mason_mode.duplicate_selected`` -- and
        :meth:`unpack_instance` -- which can attach a whole template subtree,
        and ``define_prefab`` puts no ceiling on how big that template may be
        -- still attaching unchecked. One shared check, called before every
        attach point, so a future one cannot reopen the same hole a fourth
        way. Counted with a plain structural walk (``all_nodes``), the same
        conservative, prefab-blind count :meth:`add_nodes` already used.
        """
        current = len(self.all_nodes())
        if current + adding > sc.MAX_PLACED:
            raise ValueError(
                f"adding {adding} node(s) would bring this document to "
                f"{current + adding} nodes, past the {sc.MAX_PLACED} "
                "MAX_PLACED ceiling; refusing rather than building past it"
            )

    def add_node(
        self, node: Node, *, parent_uid: int | None = None, index: int | None = None
    ) -> Node:
        """Insert one node and record the step. Returns ``node``, so a caller
        can place and keep hold of one in a single expression.

        ``node`` may itself carry a subtree (a duplicated group), so the
        ceiling counts the whole thing being attached, not just ``node``
        itself -- see :meth:`_check_max_placed`.
        """
        self._check_max_placed(len(list(nd.walk([node]))))
        siblings = self.children_of(parent_uid)
        at = len(siblings) if index is None else max(0, min(int(index), len(siblings)))
        self.history.push(ed.NodeAddEdit(parent_uid, at, node))
        self._attach_node(node, parent_uid, at)
        self.touch()
        return node

    def add_nodes(
        self, nodes: Iterable[Node], *, parent_uid: int | None = None, label: str = ""
    ) -> list[Node]:
        """Insert several nodes as **one** step.

        A prefab placement or an array op adds many nodes for one gesture, and
        N ``add_node`` calls would be N ``NodeAddEdit`` pushes -- N presses of
        Ctrl+Z through N-1 intermediate states nobody asked to see, the same
        argument ``ClayDoc.add_objects`` makes for a figure preset's sixteen
        parts. Empty is a no-op that pushes nothing, for the reason a no-op
        step always is here: a saved document must not ask to be saved again.
        """
        added = list(nodes)
        if not added:
            return []
        # The 2026-09-14 audit's mason-01: an array op with a count someone
        # typed an extra zero into (150,000 copies measured) used to build
        # and attach every copy before anything checked scene.MAX_PLACED --
        # the ceiling only ever fired downstream, in scene.resolve(), by
        # which point the document already had the extra nodes attached and
        # every future resolve()/walk() refused for good (the viewport
        # drawing empty, scene_stats reporting placed: 0). Refused here,
        # before a single node is attached, via the shared
        # :meth:`_check_max_placed` (the 2026-09-15 audit's mason-01 put
        # ``add_node`` and ``unpack_instance`` behind the same door).
        #
        # The 2026-09-16 audit's mason-engine-01: counting ``len(added)``
        # here only counted the top-level nodes handed in, not each one's
        # whole subtree -- so the Array tool duplicating a selected GroupNode
        # (``_spawn_array`` -> ``copy_subtree()`` -> here) silently attached
        # far more nodes than the ceiling check saw. Summed the same way
        # :meth:`add_node` already counts a single subtree, over every node
        # being added.
        self._check_max_placed(sum(len(list(nd.walk([node]))) for node in added))
        made: list[Edit] = []
        for node in added:
            siblings = self.children_of(parent_uid)
            at = len(siblings)
            made.append(ed.NodeAddEdit(parent_uid, at, node))
            self._attach_node(node, parent_uid, at)
        step = made[0] if len(made) == 1 else CompoundEdit(made)
        if label:
            step.label = label
        self.history.push(step)
        self.touch()
        return added

    def remove_node(self, uid: int) -> None:
        """Take a node -- and its whole subtree -- out of the tree, as one step."""
        found = self.locate(uid)
        if found is None:
            raise KeyError(f"no node {uid}")
        node, parent_uid, index = found
        self.history.push(ed.NodeRemoveEdit(parent_uid, index, node))
        self._detach_node(uid)
        self.touch()

    def move_node(self, uid: int, to_index: int, *, parent_uid: Any = _KEEP) -> None:
        """Reparent and reorder in one step -- because a drag in the outliner
        is one gesture that does both, and recording it as a reparent step
        followed by a reorder step would put a state on the undo stack the
        user never saw in between.

        **Refuses rather than clamps** when the target is the node itself or
        one of its own descendants (:func:`nodes.contains`), each with its own
        message: there is no nearby position that answers "put a group inside
        itself", so the honest response is to say no, not to silently pick
        somewhere else. This is rule 2 of ``nodes.py``'s three-part
        acyclicity argument -- the *loud* half, raised at the door before
        anything moves -- and :func:`nodes.walk`'s quiet, per-frame skip over
        a cycle it finds anyway is rule 3, the backstop for whatever gets past
        this door by being hand-edited into a ``.wscn`` outside the app
        entirely. The split exists because this call is a user asking, right
        now, to do one specific thing, and a walk is a traversal that has to
        survive a corrupt file with no one to raise to.
        """
        found = self.locate(uid)
        if found is None:
            raise KeyError(f"no node {uid}")
        node, before_parent, before_index = found
        after_parent = (
            before_parent
            if parent_uid is _KEEP
            else (None if parent_uid is None else int(parent_uid))
        )
        if after_parent is not None:
            if after_parent == uid:
                raise ValueError("a node cannot be moved inside itself")
            if nd.contains(node, after_parent):
                raise ValueError("a node cannot be moved inside its own descendant")
        siblings = self.children_of(after_parent)
        # The node is still in the tree while this clamps, so its own slot
        # counts as an available position only when it is not about to leave
        # that list -- ``LayerOps.move_layer``'s own clamp, one type over.
        limit = len(siblings) - 1 if after_parent == before_parent else len(siblings)
        after_index = max(0, min(int(to_index), max(0, limit)))
        if (after_parent, after_index) == (before_parent, before_index):
            return
        self.history.push(
            ed.NodeMoveEdit(uid, (before_parent, before_index), (after_parent, after_index))
        )
        self._relocate(uid, (after_parent, after_index))
        self.touch()

    # -- properties --------------------------------------------------------

    def set_transform(
        self,
        uid: int,
        *,
        translation: Any = None,
        rotation: Any = None,
        scale: Any = None,
        was: tuple[Any, Any, Any] | None = None,
    ) -> bool:
        """Move, rotate and scale as one step, pushing nothing for a no-op.

        ``was`` is ``clay.document.ClayDoc.set_transform``'s own parameter,
        copied verbatim in spirit: a gizmo mutates the node live so the
        viewport tracks the drag, and asks for the step only when the drag is
        released -- by which point the node already holds the *new* values,
        so reading "before" off it would compare a value against itself and
        record nothing. A caller doing that passes the values it started
        the drag with as ``was``; a caller building fresh arrays (a
        properties-panel field commit) needs none of this.
        """
        node = self._require(uid)
        for name, new, length in (
            ("translation", translation, 3),
            ("rotation", rotation, 4),
            ("scale", scale, 3),
        ):
            if new is None:
                continue
            arr = np.asarray(new, dtype="f8")
            if arr.shape != (length,) or not np.isfinite(arr).all():
                raise ValueError(f"{name} must be {length} finite numbers.")
        before = tuple(np.array(v, dtype="f8", copy=True) for v in (was or node.trs()))
        after = tuple(
            node.trs()[i] if new is None else np.array(new, dtype="f8", copy=True)
            for i, new in enumerate((translation, rotation, scale))
        )
        if all(np.array_equal(a, b) for a, b in zip(before, after, strict=True)):
            return False
        self.history.push(ed.TransformEdit(uid, before, after))  # type: ignore[arg-type]
        self._apply_transform(uid, after)  # type: ignore[arg-type]
        self.touch()
        return True

    def set_props(self, uid: int, *, was: dict[str, Any] | None = None, **props: Any) -> bool:
        """Name, visibility, lock, static, material override, user properties
        -- one step.

        ``was`` is :meth:`set_transform`'s own parameter, and the trap it
        avoids is sharper here because ``properties`` is a dict: a panel that
        edits the node's own dict in place and hands it back would make
        "before" and "after" the same object, and the change would record
        nothing at all -- ``clay.document.ClayDoc.set_props``'s own warning,
        which applies here without qualification.
        """
        node = self._require(uid)
        source = {} if was is None else was
        before = {key: source.get(key, getattr(node, key)) for key in props}
        if before == props:
            return False
        self.history.push(ed.NodePropsEdit(uid, before, dict(props)))
        self._apply_props(uid, dict(props))
        self.touch()
        return True

    def set_ref(self, uid: int, ref: Ref | None) -> bool:
        """Point a mesh node at a different source -- what Relink and a
        primitive's parameter rebuild both push."""
        node = self._require(uid)
        if not isinstance(node, nd.MeshNode):
            raise TypeError(f"node {uid} is not a mesh node and cannot carry a ref")
        if node.ref == ref:
            return False
        self.history.push(ed.RefEdit(uid, node.ref, ref))
        self._apply_ref(uid, ref)
        self.touch()
        return True

    def set_visibility(self, wanted: dict[int, bool]) -> bool:
        """Several nodes' visibility as **one** step -- isolating or
        re-showing a set is a single gesture, and hiding nine things one
        Ctrl+Z at a time is not an undo history, ``ClayDoc.set_visibility``'s
        own argument. A node already agreeing with ``wanted`` is left out
        entirely rather than recorded as a no-op.
        """
        changes: list[Edit] = []
        for uid, visible in wanted.items():
            node = self.node(uid)
            if node is None:
                continue
            if bool(node.visible) == bool(visible):
                continue
            changes.append(
                ed.NodePropsEdit(uid, {"visible": node.visible}, {"visible": bool(visible)})
            )
            self._apply_props(uid, {"visible": bool(visible)})
        if not changes:
            return False
        self.history.push(changes[0] if len(changes) == 1 else CompoundEdit(changes))
        self.touch()
        return True

    def isolate(self, uids: Iterable[int]) -> bool:
        """Show only these nodes. One step, and its own inverse is
        :meth:`show_all` rather than a second isolate."""
        keep = {int(u) for u in uids}
        return self.set_visibility({node.uid: node.uid in keep for node in self.all_nodes()})

    def show_all(self) -> bool:
        return self.set_visibility({node.uid: True for node in self.all_nodes()})

    # -- prefabs -------------------------------------------------------------

    def define_prefab(self, name: str, node: Node) -> Node:
        """Store a template subtree under ``name``, as **one** step.

        The stored template is :func:`nodes.copy_subtree` with
        ``fresh_uids=False`` -- a template is not in the scene tree (see
        ``MasonDoc.prefabs``'s own field comment), so there is no collision
        for it to have with anything a walk over ``roots`` would find, and
        keeping the uids as authored is what lets re-defining the same prefab
        from an edited version of one of its own instances read as "update",
        not "replace with something unrelated".

        **Refuses a template that would contain a ``PrefabNode`` naming
        itself, directly or through another template it places.** Recursion
        here is not a walk that could be trusted to terminate on its own --
        ``nodes.walk``'s depth ceiling is the *backstop* for a hand-edited
        file, not a licence to let an authoring gesture build a cycle on
        purpose -- so this is refused at the door, the same rule
        ``move_node`` applies to reparenting a group into itself.
        """
        if self._prefab_refers_to(node, name, frozenset()):
            raise ValueError(
                f"a prefab named {name!r} cannot contain an instance of itself, "
                "directly or through another prefab it places"
            )
        template = nd.copy_subtree(node, fresh_uids=False)
        before = self.prefabs.get(name)
        self.history.push(ed.PrefabEdit(name, before, template))
        self._apply_prefab(name, template)
        self.touch()
        return template

    def _prefab_refers_to(self, node: Node, target: str, visited: frozenset[str]) -> bool:
        """Whether ``node`` places ``target`` -- directly, or through a chain
        of prefabs it in turn places. ``visited`` stops this from re-walking a
        prefab already checked on the current path; ``nodes.walk``'s own depth
        ceiling (reached through :func:`nodes.walk` inside this walk) is what
        stops a chain that is not actually a cycle but is absurdly long.
        """
        for entry, _parent, _index, _depth in nd.walk([node]):
            if not isinstance(entry, nd.PrefabNode):
                continue
            if entry.template == target:
                return True
            if entry.template in visited:
                continue
            referenced = self.prefabs.get(entry.template)
            if referenced is not None and self._prefab_refers_to(
                referenced, target, visited | {entry.template}
            ):
                return True
        return False

    def remove_prefab(self, name: str) -> bool:
        """Drop a template. Existing instances are untouched by this call --
        they still name ``name``, and become entries in nothing until another
        :meth:`define_prefab` gives that name a template again or an
        :meth:`unpack_instance` replaces them; that resolution question
        belongs to ``scene.py``, not to this method."""
        before = self.prefabs.get(name)
        if before is None:
            return False
        self.history.push(ed.PrefabEdit(name, before, None))
        self._apply_prefab(name, None)
        self.touch()
        return True

    def unpack_instance(self, uid: int) -> Node:
        """Replace a ``PrefabNode`` with a deep, fresh-uid copy of its
        template, at the same position, keeping the instance's own name and
        transform. One compound step.

        **This is the single escape hatch the plan gives instead of per-child
        overrides, and that is a decision, not an omission.** A
        ``PrefabNode`` carries no per-child override on purpose -- deep
        overrides are what makes prefab systems hard, and the alternative to
        "unpack and edit the copy" is a propagation model that has to decide,
        forever after, which of an instance's fields still follow the
        template and which do not. Unpacking sidesteps that question outright
        by turning the one instance that needs to diverge into an ordinary,
        independent subtree, while every instance that does not call this
        keeps tracking the template with **no propagation step at all** --
        the walk simply reads through ``self.prefabs[name]`` every time, so a
        template edit reaches every remaining instance on the very next
        frame, which is the whole point of a prefab existing.
        """
        found = self.locate(uid)
        if found is None:
            raise KeyError(f"no node {uid}")
        instance, parent_uid, index = found
        if not isinstance(instance, nd.PrefabNode):
            raise TypeError(f"node {uid} is not a prefab instance")
        template = self.prefabs.get(instance.template)
        if template is None:
            raise KeyError(f"no prefab named {instance.template!r}")

        # The 2026-09-15 audit's mason-01: a template has no size ceiling of
        # its own (``define_prefab`` never counted against MAX_PLACED,
        # because a template is not part of ``roots`` -- see the module
        # docstring), so unpacking one is where an oversized template first
        # meets the scene tree it is about to be attached to. The instance
        # being replaced is still in the tree at this point (it is removed
        # below), so its own subtree size is subtracted out of the count --
        # net growth, not the copy's raw size -- which is what lets
        # replacing an instance one-for-one with a same-sized copy never
        # refuse.
        net_growth = len(list(nd.walk([template]))) - len(list(nd.walk([instance])))
        self._check_max_placed(net_growth)
        copy = nd.copy_subtree(template, fresh_uids=True)
        copy.name = instance.name
        copy.translation = np.array(instance.translation, dtype="f8", copy=True)
        copy.rotation = np.array(instance.rotation, dtype="f8", copy=True)
        copy.scale = np.array(instance.scale, dtype="f8", copy=True)
        copy.visible = instance.visible
        copy.locked = instance.locked
        copy.static = instance.static
        copy.properties = dict(instance.properties)

        # Two pushes folded into one step -- ``ClayDoc.add_material_and_assign``'s
        # own ``mark``/``collapse_since`` pattern -- because a single Ctrl+Z
        # putting the prefab instance back while the fresh copy's uid is
        # already gone (or the reverse) is exactly the half-undone state that
        # pattern exists to rule out.
        mark = self.history.mark()
        self.history.push(ed.NodeRemoveEdit(parent_uid, index, instance))
        self._detach_node(uid)
        self.history.push(ed.NodeAddEdit(parent_uid, index, copy))
        self._attach_node(copy, parent_uid, index)
        self.history.collapse_since(mark)
        self.touch()
        return copy

    # -- terrain ---------------------------------------------------------------

    def set_terrain(self, terrain: Terrain | None) -> None:
        """Install the document's one height field, or lift it back out, as
        **one** undoable :class:`~.edits.TerrainSwapEdit` step.

        This used to argue that whether the document has a terrain at all is
        a structural fact rather than a value worth an undo step -- the same
        register a document's initial object list sits in. That argument does
        not survive contact with what it actually costs: a node's arrival or
        departure gets ``NodeAddEdit``/``NodeRemoveEdit`` for the identical
        structural reason, and a terrain that could not get the same
        treatment meant its *only* reference was this one attribute -- so
        clearing it, one ordinary click, silently discarded however long
        someone had spent sculpting it, with Ctrl+Z having nothing left to
        restore. See :class:`~.edits.TerrainSwapEdit`'s own docstring for the
        rest of that argument.

        Any open sculpt session is committed first, because a session
        outlives the terrain it was opened against only by accident, never on
        purpose. Installing the very instance already in place -- by
        identity, which is the only equality ``Terrain`` defines (see its own
        ``eq=False``) -- pushes nothing, the same no-op rule every other
        setter here follows.
        """
        if self.sculpting:
            self.end_sculpt()
        if terrain is self.terrain:
            return
        self.history.push(ed.TerrainSwapEdit(self.terrain, terrain))
        self._apply_terrain(terrain)
        self.touch()

    def set_terrain_config(self, **values: Any) -> bool:
        """The terrain's ``size_x``, ``size_z`` and/or ``material``, as one step."""
        if self.terrain is None:
            raise ValueError("this document has no terrain to configure")
        unknown = set(values) - {"size_x", "size_z", "material"}
        if unknown:
            raise ValueError(f"unknown terrain config field(s): {sorted(unknown)}")
        before = {key: getattr(self.terrain, key) for key in values}
        if before == values:
            return False
        self.history.push(ed.TerrainConfigEdit(before, dict(values)))
        self._apply_terrain_config(dict(values))
        self.touch()
        return True

    @property
    def sculpting(self) -> bool:
        return self._sculpt is not None

    def begin_sculpt(self) -> None:
        """Open a sculpt session: snapshot the whole height field once.

        Re-opening closes whatever session is already open first -- the same
        rule ``PaintOps.begin_stroke`` states for a tile layer, restated
        because a session left dangling across a second ``begin_sculpt`` would
        otherwise lose the first drag's own undo step outright rather than
        merely fail to commit it.
        """
        if self.terrain is None:
            raise ValueError("this document has no terrain to sculpt")
        if self._sculpt is not None:
            self.end_sculpt()
        self._sculpt = {
            "before": np.array(self.terrain.heights, dtype=np.float32, copy=True),
            "box": None,
        }

    def sculpt(self, rect: Rect, sub: np.ndarray) -> bool:
        """Rebind the live height field with ``sub`` at ``rect``, and grow the
        session's union box. Pushes nothing -- a drag across forty frames is
        one gesture, and :meth:`end_sculpt` is where it becomes one step.
        """
        if self._sculpt is None:
            raise RuntimeError("no sculpt session is open")
        assert self.terrain is not None
        x0, y0, x1, y1 = rect
        block = np.ascontiguousarray(sub, dtype=np.float32)
        if block.shape != (y1 - y0, x1 - x0):
            raise ValueError("sub does not match rect's shape")
        if np.array_equal(self.terrain.heights[y0:y1, x0:x1], block):
            return False
        self._blit_terrain(rect, block)
        box = self._sculpt["box"]
        self._sculpt["box"] = (
            rect
            if box is None
            else (min(box[0], x0), min(box[1], y0), max(box[2], x1), max(box[3], y1))
        )
        return True

    def end_sculpt(self) -> bool:
        """Close the session and push **one** :class:`~.edits.TerrainEdit`
        over the union of everything it touched. ``False`` if nothing moved.

        **Idempotent on purpose**, the same rule ``PaintOps.end_stroke``
        states: a release can be missed -- focus loss, Esc, a save beginning
        mid-drag -- and every recovery path would otherwise have to know
        whether a session was even open.
        """
        session, self._sculpt = self._sculpt, None
        if session is None or session["box"] is None:
            return False
        assert self.terrain is not None
        x0, y0, x1, y1 = session["box"]
        before = session["before"][y0:y1, x0:x1]
        after = self.terrain.heights[y0:y1, x0:x1]
        if np.array_equal(before, after):
            return False
        self.history.push(
            ed.TerrainEdit((x0, y0, x1, y1), before, np.ascontiguousarray(after, dtype=np.float32))
        )
        self.touch()
        return True

    # -- the hooks the edits call back into ------------------------------------

    def _attach_node(self, node: Node, parent_uid: int | None, index: int) -> None:
        siblings = self.children_of(parent_uid)
        siblings.insert(max(0, min(int(index), len(siblings))), node)

    def _detach_node(self, uid: int) -> Node:
        """Remove one node from wherever it currently sits, and drop it from
        the selection.

        **The selection rule, decided and tested rather than left implicit:
        detaching a node always drops it from ``selection``, and re-attaching
        one (an undone remove, a redone add) never puts it back.** Selection
        is not undoable (the module docstring's own rule), so an edit's
        undo/redo has no business restoring it -- the alternative, leaving a
        detached uid selected, would hand the properties panel and the gizmo
        a uid that ``node()`` cannot find, which is a crash waiting for
        whichever panel reads ``selection`` first. This mirrors
        ``ClayDoc.remove_object``, whose ``ObjectAddEdit.undo`` and
        ``ObjectRemoveEdit.redo`` both discard the same way.
        """
        found = self.locate(uid)
        if found is None:
            raise KeyError(f"no node {uid}")
        node, parent_uid, index = found
        del self.children_of(parent_uid)[index]
        self.selection.discard(uid)
        return node

    def _relocate(self, uid: int, target: tuple[int | None, int]) -> None:
        parent_uid, index = target
        found = self.locate(uid)
        if found is None:
            raise KeyError(f"no node {uid}")
        node, from_parent, from_index = found
        del self.children_of(from_parent)[from_index]
        siblings = self.children_of(parent_uid)
        siblings.insert(max(0, min(int(index), len(siblings))), node)

    def _apply_transform(
        self, uid: int, trs: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        node = self._require(uid)
        node.translation, node.rotation, node.scale = (
            np.array(v, dtype="f8", copy=True) for v in trs
        )

    def _apply_props(self, uid: int, props: dict[str, Any]) -> None:
        node = self._require(uid)
        for key, value in props.items():
            setattr(node, key, value)

    def _apply_ref(self, uid: int, ref: Ref | None) -> None:
        node = self._require(uid)
        if not isinstance(node, nd.MeshNode):
            raise TypeError(f"node {uid} is not a mesh node")
        node.ref = ref

    def _apply_terrain(self, terrain: Terrain | None) -> None:
        """The hook :class:`~.edits.TerrainSwapEdit` calls back into: swap the
        whole singleton in or out. Unlike :meth:`_blit_terrain`, there is no
        existing array to rebind here -- there may be no terrain on either
        side of the swap -- so this simply assigns.
        """
        self.terrain = terrain

    def _blit_terrain(self, rect: Rect, sub: np.ndarray) -> None:
        """Rebind ``terrain.heights`` to a new array with ``sub`` pasted at
        ``rect`` -- **never write into the live array in place.**
        ``terrain.Terrain``'s own docstring names the two things that break
        silently otherwise: :func:`terrain.terrain_mesh`'s memo, keyed on
        ``cached_array is terrain.heights``, would keep describing a ground
        that no longer exists under it; and this very edit's own recorded
        ``before`` -- a slice taken *before* this call -- would be edited
        retroactively out from under the history stack, the exact trap
        ``studio/undo.py``'s module docstring names for a shared buffer.
        """
        terrain = self.terrain
        if terrain is None:
            raise ValueError("this document has no terrain")
        x0, y0, x1, y1 = rect
        full = np.array(terrain.heights, dtype=np.float32, copy=True)
        full[y0:y1, x0:x1] = sub
        terrain.heights = full

    def _apply_terrain_config(self, values: dict[str, Any]) -> None:
        terrain = self.terrain
        if terrain is None:
            raise ValueError("this document has no terrain")
        # Every key ``set_terrain_config`` can record has to be assignable
        # here -- ``_map_layers.py``'s own warning about ``_apply_layer_props``,
        # restated for this hook: a field present in a ``TerrainConfigEdit``
        # but not settable through this loop is a field an undo cannot
        # restore. ``setattr`` reaches all three of ``Terrain``'s non-height
        # fields uniformly, so there is no per-field branch here to forget.
        for key, value in values.items():
            setattr(terrain, key, value)

    def _apply_prefab(self, name: str, node: Node | None) -> None:
        if node is None:
            self.prefabs.pop(name, None)
        else:
            self.prefabs[name] = node

    # -- history -----------------------------------------------------------

    def undo(self) -> bool:
        """Reverse the newest step. Commits an open sculpt session first --
        stepping over one instead would leave the document's ground ahead of
        its own head, ``PaintOps``' own reason for closing a stroke the same
        way.
        """
        if self.sculpting:
            self.end_sculpt()
        if not self.history.undo(self):
            return False
        self.touch()
        return True

    def redo(self) -> bool:
        if self.sculpting:
            self.end_sculpt()
        if not self.history.redo(self):
            return False
        self.touch()
        return True

    def step_history(self, index: int) -> bool:
        """Jump to a position in the undo stack (the count of done steps).

        Through :meth:`undo`/:meth:`redo` and **not**
        ``self.history.step_to(self, n)`` -- ``ClayDoc.step_history`` and
        ``PlotterDoc.step_history`` both give the same reason and this is the
        one call site that keeps re-losing it: ``step_to`` walks the stack's
        own ``undo``/``redo``, neither of which commits an open sculpt
        session. Called straight, a jump made mid-drag would leave the
        session open with the document's head having moved out from under
        it -- the exact defect closing sessions on undo/redo exists to
        prevent, reintroduced by the one caller that reached past it.
        """
        total = len(self.history.history())
        wanted = max(0, min(int(index), total))
        moved = False
        while len(self.history) > wanted and self.undo():
            moved = True
        while len(self.history) < wanted and self.redo():
            moved = True
        return moved

    def push(self, edit: Edit) -> None:
        self.history.push(edit)
        self.touch()

    def compound(self, edits: list[Edit]) -> None:
        """Push several edits as one step. A no-op for an empty list."""
        if not edits:
            return
        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        self.touch()

    def mark(self) -> int:
        return self.history.mark()

    def collapse_since(self, mark: int) -> bool:
        return self.history.collapse_since(mark)

    # -- state ---------------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self.history.head != self.saved_head

    def mark_saved(self) -> None:
        self.saved_head = self.history.head

    def touch(self) -> None:
        self.rev += 1

    def select(self, uids: Iterable[int], *, active: int | None = None) -> None:
        """Replace the selection. Pushes no step -- see the module docstring
        -- but still bumps ``rev``, because the viewport draws a selection
        outline and has to know to redraw it.

        ``active`` records which uid was the one actually clicked, for
        :attr:`selection_active` (the 2026-09-16 audit's mason-... finding --
        see that attribute's own comment). Most callers replace the whole
        selection with exactly one uid -- an outliner row click, a
        context-menu click, a duplicate that reselects its own result -- and
        that is unambiguous with no ``active`` argument needed, so it is
        inferred. A caller building a multi-node selection from an
        extend/toggle/range gesture is the one case that is *not*
        unambiguous (only the caller knows which of the several uids was the
        one under the pointer) and must pass ``active`` explicitly --
        ``mason_view.MasonView._press``'s shift/ctrl branch does.
        """
        self.selection = {int(u) for u in uids}
        if active is not None:
            self.selection_active = int(active)
        elif len(self.selection) == 1:
            self.selection_active = next(iter(self.selection))
        # else: an ambiguous multi-uid replacement with no ``active`` given
        # (a Shift+range, a Ctrl+A) leaves the previous value in place --
        # stale is fine, since it is checked against the live selection
        # before use, never trusted blindly.
        self.touch()

    # -- reporting -------------------------------------------------------------

    def missing_refs(self) -> list[tuple[Node, Ref]]:
        """Every mesh node whose reference the host has told this document it
        could not resolve, in walk order.

        **This is the plan's dangling-reference story, and it is not a
        contradiction of ``.wblk``'s refuse-rather-than-substitute rule.** A
        ``.wblk`` that found a build step it could not replay refuses to open,
        because substituting an empty mesh would show the user finished work
        that silently is not there, and let them save over it with nothing
        left to recover. A Mason scene with one broken ``LibraryRef`` is a
        different shape of problem: the document is a set of *links*, most of
        which still resolve fine, and refusing to open the whole scene over
        one dead job id would strand every other node in it for a reason the
        user cannot see or fix from outside the app. So the node opens as a
        listed, repairable fact instead -- drawn as a missing-asset proxy by
        the viewport, offered **Relink…** (one :class:`~.edits.RefEdit`) and
        **Remove** by this list -- which is what "known and repairable" means
        here where ".wblk's own build step is gone" means something has
        already silently failed.
        """
        return [
            (node, node.ref)
            for node in self.all_nodes()
            if isinstance(node, nd.MeshNode)
            and node.ref is not None
            and ref_key(node.ref) in self.missing
        ]
