"""The Clay document -- objects, a material palette, and the history over them.

A Clay document is a flat list of objects, each carrying a TRS and, since
tranche 3, an optional ``parent`` uid (see the "hierarchy" section below for
what that buys and what it costs). :func:`to_model` is the one conversion
that builds real glTF hierarchy from a whole document, and the GLB writer is
its only consumer. **The viewport and the trellis render do not call it** --
each builds its per-object primitives straight from :func:`to_primitives`
(the viewport's own cache, ``studio/modes/clay/ui/_view_cache.py``, and
``render_png``/``render_ids`` in ``studio/modes/clay/ui/view.py``) rather
than through a node tree, because neither wants a flattened scene: the
viewport indexes its GPU buffers per object uid and the render needs a
colour table keyed the same way. The 2026-09-22 audit's clay-21 found the
"exactly one of it" claim this paragraph used to make false for those two --
:func:`to_primitives` is the one conversion the three consumers actually
share, and ``tests/modes/clay/test_document.py``'s
``test_the_viewports_per_object_cache_agrees_with_to_model_for_a_hidden_parent_with_a_visible_child``
pins that the two building paths -- ``to_model``'s per-object
``to_primitives`` call and the viewport cache's own -- still agree on which
objects get a node/entry and what each one draws, since nothing enforces
that agreement structurally.

**Materials are ``viewer.gltf.Material`` objects, not a new type.** They are
already pure data, they already carry ``base_color_factor`` /
``metallic_factor`` / ``roughness_factor`` / ``emissive_factor`` /
``double_sided``, and a parallel Clay-only material type would have bought
nothing but a conversion function and a place for the two to drift. It also
pays off on the GPU for free: ``GpuMaterial`` de-duplicates by
``id(material)``, and :func:`to_primitives` hands every primitive the palette
entry *itself*, so a document whose twelve objects share one material uploads
one material. The texture slots stay ``None`` throughout -- Clay paints
no textures, and a slot that is ``None`` is a slot the renderer skips.

**Rotation is XYZW**, matching ``viewer.math3d``, ``viewer.gltf``, the pose
files on disk and glTF itself. There is no other quaternion order anywhere in
this project and this is not the place to introduce one.

**``generator`` is the live-until-frozen field.** An object placed from the
primitive registry keeps the generator's name and the parameters it was built
with, so the properties panel shows "Cylinder: radius, height, segments" and a
change regenerates the mesh as one :class:`~.edits.MeshEdit`. The first
topology edit clears it -- an extruded box is not describable as "box, size 1",
and a panel still offering a size field would discard the edit the moment it
was touched. ``clay_ops`` does that in one place for every op, so no op has to
remember to.

**``modifiers`` is a second, later stage that the freeze rule does not touch.**
``Obj.mesh`` is still the *base* -- what an element edit, an element pick, a
drag and every mesh op reads and writes, exactly as before -- and an object
additionally carries ``modifiers: tuple[Modifier, ...]`` (default ``()``,
:mod:`.modifiers`). The **evaluated** mesh is the base run through each
enabled modifier in order (:func:`~.modifiers.evaluate`, ``ClayDoc.
evaluated``/``.evaluation``), and it is what display, export, measurement and
object-mode picking read; editing still reads the base. :meth:`ClayDoc.
set_mesh` freezes the generator exactly as it always did and **keeps the
stack** -- a topology edit invalidates "this is a box", not "this box also has
a mirror on it" -- while :meth:`ClayDoc.join_objects` and a boolean modifier's
own target both *consume* an evaluated mesh, and the object whose stack fed
one is cleared of it in the same step: its modifiers are now baked into
whatever adopted the result, and leaving the stack in place would apply it a
second time the next time that object was drawn.

**``locked`` means "cannot be changed", not "cannot be seen".** A locked
object still draws, still exports, and a viewport click still passes through
it -- the outliner still selects it -- so only a door that would alter what
the object *is* refuses it. :meth:`ClayDoc._refuse_if_locked` is that
refusal, called first by every door that mutates the object's own geometry,
modifier stack or transform: :meth:`set_mesh`, :meth:`set_transform` (which
also walks :meth:`ancestors`, since dragging an object *inside* a locked
group still visibly rearranges the group even though the group's own
geometry never changes), :meth:`set_generator_params`, :meth:`set_seams`,
:meth:`set_modifiers`, :meth:`apply_modifiers` and :meth:`join_objects`
(which refuses for its own target *and* for any locked object named in
``others`` -- Join and every boolean consume an evaluated mesh and delete
the objects they absorbed, and a locked object survived neither before the
2026-09-20 audit's clay-01). :meth:`remove_object` and
:meth:`separate` check ``obj.locked`` directly for the same reason without
going through the shared helper -- the object does not survive either door,
which is the least undoable change there is. Deliberately exempt:
:meth:`set_parent` and :meth:`set_origin`, because with ``keep_world``
neither moves anything on screen -- a re-framing, not a change to what the
object looks like (each says so in its own docstring); :meth:`set_props`,
because a locked object must still allow a rename, a visibility change, a
tag edit and unlocking itself, or a mistake made while locked could never be
undone by anyone but the lock; and :meth:`add_collider`, because attaching a
new child changes nothing about the parent object itself. See the
2026-09-19 audit's clay-28, which found a dozen citations of this exact
paragraph and no paragraph here for them to cite.

**Dirty is a comparison against ``history.head``, not a flag.** ``rev`` counts
changes and an undo is a change, so a rev-based check calls an undone document
unsaved forever. :attr:`ClayDoc.saved_head` records the head at save time and
:attr:`ClayDoc.dirty` compares against it, which is the same rule the raster
editor's tabs follow and for the same reason.

**Element mode is per document, not per app.** ``element_mode`` and
``element_sel`` live here rather than on ``ClayState`` because the mode is the
*interpretation key* for the selection, and the selection is a property of the
document. An app-level mode would reinterpret every open tab's selection the
moment the user switched tabs -- a face selection in one document read as a
vertex selection in another -- and neither half would have done anything wrong.
Neither the mode nor the selection is serialized: a stored element selection
would describe indices into a mesh the file might reopen with, and a stored mode
would put the user in a mode they did not choose.

In an element mode ``selection`` is **derived**: it holds exactly the uids with
a non-empty ``element_sel``, which is Wings3D's model (a body selection *is* a
selection of everything in it). That invariant is what keeps
``frame_selection``, the properties pane and ``world_bounds`` working with no
per-mode branch anywhere in them.

**Selection is not undoable.** The raster editor makes a selection undoable
because a lasso around a character's hand is minutes of work that a stray click
destroys. Clicking an object in a 3D viewport is not that, and Blender's object
mode agrees -- Ctrl+Z there reverses the last *edit*, not the last click. It
matters beyond taste: an undoable selection would push a step, the step would
move ``history.head``, and a document would ask to be saved because the user
looked at a different object.
"""

from __future__ import annotations

import itertools
import threading
import weakref
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from ...core.undo import CompoundEdit, Edit, UndoStack
from ..geom3d import gltf
from ..geom3d import math3d as m3
from . import colliders
from . import elements as el
from . import mesh as bm
from .edits import (  # noqa: F401
    MaterialEdit,
    MaterialListEdit,
    MeshEdit,
    ObjectAddEdit,
    ObjectMoveEdit,
    ObjectPropsEdit,
    ObjectRemoveEdit,
    TransformEdit,
    _boundary_uids,
    _material_holders,
    _shift_materials,
)

_uids = itertools.count(1)
#: ``reserve_uid`` runs on a task thread -- ``serialize.read_rblk`` is how an
#: open and a crash recovery both arrive -- while ``new_uid`` runs on the frame
#: thread. Swapping the counter out from under a ``next`` is the race this
#: closes; it is the one lock in the package and it is held for one line.
_uid_lock = threading.Lock()


def new_uid() -> int:
    """Mint an object uid. Never reused, for the life of the process.

    The uid is the *address* every undo step is written against, so reuse is
    not a tidiness question: a recycled uid would let a step recorded against a
    deleted object land on a different one that happens to wear its number. A
    process-wide counter rather than a per-document one, so the same rule holds
    across a document that was closed and reopened in the same session.
    """
    with _uid_lock:
        return next(_uids)


def reserve_uid(uid: int) -> None:
    """Guarantee that :func:`new_uid` never hands back ``uid`` or anything below.

    A saved document carries its uids, and a reload that reissued them would
    silently retarget every undo step recorded against one. But the counter is
    per *process*, not per document, so a file whose uids run past where this
    session happens to have got to would otherwise collide the moment the user
    adds an object -- the new object and a restored one wearing one number, and
    the first edit to either landing on whichever ``index_of`` reaches first.
    So the reader raises the floor as it restores. Deliberately monotonic: the
    counter never moves backwards, so loading a small document after a large
    one cannot undo the protection the large one bought.
    """
    global _uids
    with _uid_lock:
        _uids = itertools.count(max(int(uid) + 1, next(_uids)))


def default_material(name: str = "Material") -> gltf.Material:
    """A plain untextured dielectric -- the palette entry a new object gets.

    ``gltf.Material``'s own defaults are the glTF spec's, which are fully
    metallic and fully rough: correct as a file format default and useless as
    an editing default, because a metal with no environment behind it renders
    as a black shape. This is the mid-grey a modelling package opens on.
    """
    return gltf.Material(
        name=name,
        base_color_factor=(0.8, 0.8, 0.8, 1.0),
        metallic_factor=0.0,
        roughness_factor=0.6,
    )


# A face whose material index names no palette entry still has to draw. It gets
# this one -- a single shared object, so the identity de-duplication holds for
# it too, and a visibly wrong magenta rather than a silent grey, because a mesh
# pointing off the end of the palette is a bug somewhere upstream.
FALLBACK_MATERIAL = gltf.Material(
    name="missing",
    base_color_factor=(1.0, 0.0, 1.0, 1.0),
    metallic_factor=0.0,
    roughness_factor=1.0,
)


def _empty_mesh() -> bm.Mesh:
    """A mesh with zero vertices and zero faces -- what :meth:`ClayDoc.group`
    gives its new empty object. Draws nothing (:func:`to_primitives` already
    returns ``[]`` for a zero-face mesh) and exports as a transform-only
    node (:func:`to_model`). Built by hand rather than through
    :func:`~.mesh.from_faces` (which needs at least one face to infer
    ``starts`` from): an empty mesh is the one shape that function cannot
    describe, since it never has a face to start from."""
    return bm.Mesh(
        positions=np.zeros((0, 3), dtype="f4"),
        loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"),
        material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype=bool),
    )


def _normalize_tags(tags: Iterable[str]) -> tuple[str, ...]:
    """A tag set as the document always stores one: sorted, deduplicated,
    lower-cased -- so "Prop" and "prop" typed on two different objects are the
    same tag for the outliner's filter and ``clay_select_by``'s query, rather
    than two entries that happen to look alike in the list."""
    return tuple(sorted({str(t).strip().lower() for t in tags if str(t).strip()}))


def _normalize_seams(pairs: Iterable[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    """A seam set as the document always stores one: each pair ``(a, b)``
    with ``a < b``, deduplicated, sorted -- the same reasoning
    ``_normalize_tags`` gives for a tag typed in either case, or twice: two
    calls marking the same edge, in either vertex order or more than once,
    must describe one identical document, not two states a diff or an undo
    step could disagree about. A pair naming the same vertex twice is not an
    edge at all and is silently dropped, the same tolerant answer
    ``_normalize_tags`` gives an empty string, rather than refused -- this
    function only ever *canonicalises* a shape that is already sound. The
    one thing it cannot fix, a vertex index the mesh does not actually have,
    is :meth:`ClayDoc.set_seams`'s refusal to make, never this one's: this
    runs from :class:`Obj`'s own ``__post_init__``, with no mesh size known
    to be trustworthy yet at every call site (a scratch clone, a file mid-read
    before the archive's own bounds have been checked).
    """
    out: set[tuple[int, int]] = set()
    for pair in pairs:
        a, b = int(pair[0]), int(pair[1])
        if a == b:
            continue
        out.add((a, b) if a < b else (b, a))
    return tuple(sorted(out))


def _restrict_seams(
    seams: tuple[tuple[int, int], ...], mesh: bm.Mesh
) -> tuple[tuple[int, int], ...]:
    """*seams* with any pair naming a vertex *mesh* no longer has removed.

    The exact range check :func:`~.elements.restrict` already applies to an
    element selection, and for the exact reason that function's own
    docstring gives: an index still in range after a rebuild survives
    verbatim however little it still means, because closing that gap needs
    the *old* mesh to compare against, which this function is never given
    either -- only :meth:`ClayDoc.set_generator_params`, one of this
    function's two callers, holds both meshes at once, and it already spends
    that on the element selection alone via ``el.restrict``. Used at
    :meth:`ClayDoc.set_mesh` (reached by dozens of different mesh ops with
    no shared way to say whether *this* particular call preserved indices)
    and :meth:`ClayDoc.set_generator_params` (a full rebuild from parameters,
    the exact site ``el.restrict`` already accepts this same "narrower than
    it sounds" trade for the element selection). :meth:`ClayDoc.separate`,
    :meth:`ClayDoc.join_objects` and :meth:`ClayDoc.apply_modifiers` do not
    call this at all -- each hands its own caller a mesh that a compaction, a
    weld/concatenation or an arbitrary modifier already provably did not keep
    indexed the way the source was, so a range check there would not be
    narrow, it would be wrong; each drops every seam outright instead, and
    says so in its own docstring.
    """
    if not seams:
        return seams
    n = len(mesh.positions)
    kept = tuple(pair for pair in seams if pair[0] < n and pair[1] < n)
    return seams if kept == seams else kept


@dataclass
class Obj:
    """One object: a mesh, where it sits, and how it was made."""

    uid: int
    name: str
    mesh: bm.Mesh
    translation: Any = field(default_factory=m3.vec3)
    rotation: Any = field(default_factory=m3.quat_identity)  # XYZW
    scale: Any = field(default_factory=lambda: m3.vec3(1.0, 1.0, 1.0))
    generator: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    # The default for *new* faces only. A face's actual material lives on the
    # mesh, one index per face, because a two-toned box is one object.
    material: int = 0
    # The live recipe layered on top of ``mesh``. See the module docstring's
    # "modifiers is a second, later stage" paragraph; :mod:`.modifiers` is the
    # vocabulary and :func:`~.modifiers.evaluate` is what runs it.
    modifiers: tuple[Any, ...] = ()
    # Tranche 3: scene structure. A uid, never an index -- see ``edits``' own
    # rule for why every reference in this package is a uid. ``None`` is a
    # root, and TRS is local to whatever this names (a root's local TRS *is*
    # its world TRS, which is what keeps a document with no parenting behaving
    # exactly as it always did -- see the module docstring).
    parent: int | None = None
    # "Cannot be changed", not "cannot be seen": refused at the doors listed
    # in the module docstring's locking paragraph. Viewport clicks pass
    # through a locked object and the outliner still selects it.
    locked: bool = False
    # Free-form, sorted/deduplicated/lower-cased on the way in (see
    # ``_normalize_tags``) -- the one membership concept a group or a
    # collection would otherwise have been (see the module docstring).
    tags: tuple[str, ...] = ()
    # Tranche 6: authoring intent for an unwrap, not geometry -- a seam is
    # where the *user* means to cut before an unwrap runs, which (see the
    # module docstring's own seams paragraph) cannot be derived from the
    # mesh or the uvs the way islands or a texel-density number can.
    # Vertex-index pairs into the object's *base* mesh, each stored
    # ``(a, b)`` with ``a < b``, sorted and deduplicated (see
    # ``_normalize_seams``, run below the same way ``_normalize_tags`` is).
    seams: tuple[tuple[int, int], ...] = ()
    # Tranche 7: a collider is an ordinary object that happens to carry a
    # role -- see ``ClayDoc.add_collider`` -- not a second kind of thing the
    # rest of this module has to special-case. "mesh" (every object before
    # this tranche) or "collider".
    role: str = "mesh"
    # Which of ``colliders.COLLIDER_KINDS`` this is, when ``role`` is
    # "collider"; empty otherwise. A plain string rather than an import of
    # ``colliders.py``'s own enum-like keys, so this dataclass costs nothing
    # to construct for the overwhelming majority of objects that are not one.
    collider_kind: str = ""

    def __post_init__(self) -> None:
        # Own the transform arrays rather than aliasing whatever was passed in,
        # for the reason ``edits`` states: a caller that keeps mutating the
        # array it handed over would rewrite a recorded step behind the undo
        # stack's back, and a view would misreport its own size to eviction.
        self.translation = np.array(self.translation, dtype="f8", copy=True)
        self.rotation = np.array(self.rotation, dtype="f8", copy=True)
        self.scale = np.array(self.scale, dtype="f8", copy=True)
        self.tags = _normalize_tags(self.tags)
        self.seams = _normalize_seams(self.seams)

    def trs(self) -> tuple[Any, Any, Any]:
        return self.translation, self.rotation, self.scale


class ClayDoc:
    """Objects, the palette they share, the selection and the history."""

    def __init__(
        self,
        objects: Iterable[Obj] | None = None,
        materials: Iterable[gltf.Material] | None = None,
    ) -> None:
        self.objects: list[Obj] = list(objects or [])
        self.materials: list[gltf.Material] = (
            [default_material()] if materials is None else list(materials)
        )
        self.selection: set[int] = set()  # object uids
        # Element mode and what is selected inside each object. See the module
        # docstring: per document, derived-from rather than parallel-to
        # ``selection``, and never written to a file.
        self.element_mode: str = "object"
        self.element_sel: dict[int, el.ElementSel] = {}
        self.history = UndoStack()
        # A change counter, for anything that caches off the document -- the
        # viewport's GPU upload, the outliner's row list. Deliberately *not*
        # what "dirty" is derived from; see the module docstring.
        self.rev = 0
        self.saved_head = self.history.head
        # ``mesh_stamp``'s cache: uid -> (the mesh it last minted a number for,
        # that number). See :meth:`mesh_stamp` for why this exists at all
        # rather than an agent tool just handing over ``id(obj.mesh)``.
        self._mesh_stamps: dict[int, tuple[bm.Mesh, int]] = {}
        self._next_stamp = 0
        # The modifier-evaluation cache: uid -> the modifiers module's own
        # entry type. Kept as ``Any`` here rather than typed against
        # :mod:`.modifiers`, which imports *this* module -- see that module's
        # docstring for why the import runs one way. Never read or written
        # directly outside :meth:`evaluated`/:meth:`evaluation` and the few
        # methods that pop a stale entry; :func:`~.modifiers.evaluate` owns
        # what lives inside it.
        self._evaluated: dict[int, Any] = {}
        # Named history positions, keyed on the *serial* :attr:`history.head`
        # gives back, never a stack position -- see :meth:`set_checkpoint`'s
        # own docstring for why. Not serialized: the undo history is not
        # saved either, so a checkpoint cannot outlive the session that made
        # it.
        self.checkpoints: dict[str, int] = {}

    # -- lookup ------------------------------------------------------------

    def index_of(self, uid: int) -> int:
        for i, obj in enumerate(self.objects):
            if obj.uid == uid:
                return i
        raise KeyError(f"no object with uid {uid}")

    def by_uid(self, uid: int) -> Obj:
        return self.objects[self.index_of(uid)]

    def touch(self) -> None:
        self.rev += 1

    def mesh_stamp(self, uid: int) -> int:
        """A wire-safe revision number for one object's current mesh.

        ``Mesh`` is immutable and ``eq=False`` (see its own docstring), so
        object identity already *is* the mesh's revision -- ``clay/mesh.py``'s
        ``_RAW_CACHE`` (lines 623-641) already keys off exactly that, for a
        drag's per-frame normals cache. What identity is not is a value safe to
        hand an agent over the wire: ``id()`` is a memory address CPython
        recycles the moment the old object is garbage collected, so a stale
        token an agent held onto across a few calls could come back and
        validate against a **different** mesh that happens to have landed at
        the same address -- silently wrong, which is worse than a token that
        is merely absent.

        So this mints its own small integers instead, lazily: nothing is
        computed until asked, which is what keeps the human path -- every
        click, drag and undo that never calls this -- paying nothing for it.
        The cache holds only the *last* mesh this uid was asked about; asking
        again returns the same number for as long as ``obj.mesh`` is the same
        object, and a different object mints a fresh one.

        **One property that looks like a bug and is not: an undo can hand back
        a stamp this method already gave out.** ``MeshEdit.undo`` restores the
        exact previous ``Mesh`` object (not a rebuilt copy of it), so if
        nothing asked for a stamp while the edit was in effect, the cache still
        holds that original mesh's entry when the undo lands -- and this
        correctly reports no change at all, because from the token's point of
        view nothing *has* changed: the mesh an agent is looking at is,
        object-for-object, the one it looked at before. A naive counter that
        incremented on every ``MeshEdit`` instead of every *asked-about*
        identity change would get exactly this wrong, reporting a fresh
        revision in the one moment an agent is most likely to be recovering
        from a mistake and checking whether it actually worked. See
        ``test_an_undo_restores_the_stamp_the_mesh_had_before`` in
        ``tests/modes/clay/test_document.py``, pinned so nobody "fixes" this into a
        counter.
        """
        obj = self.by_uid(uid)
        cached = self._mesh_stamps.get(uid)
        if cached is not None and cached[0] is obj.mesh:
            return cached[1]
        self._next_stamp += 1
        self._mesh_stamps[uid] = (obj.mesh, self._next_stamp)
        return self._next_stamp

    # -- hierarchy -----------------------------------------------------------
    #
    # Tranche 3: parenting. TRS is *local to the parent* -- see the module
    # docstring -- so every world-space reader in this package funnels
    # through :meth:`world_matrix` rather than composing an object's own TRS
    # directly, and a root's world matrix is exactly its local one, which is
    # what keeps a document with no parenting behaving exactly as it always
    # did.

    def children_of(self, uid: int) -> list[int]:
        """*uid*'s direct children, in document order."""
        return [obj.uid for obj in self.objects if obj.parent == uid]

    def ancestors(self, uid: int) -> list[int]:
        """*uid*'s parent, grandparent, and so on -- nearest first.

        Stops at the first uid it has already seen rather than trusting the
        chain is acyclic: :meth:`set_parent` refuses a cycle going forward
        and :mod:`.serialize` refuses one in a loaded file, so this should
        never actually run into one, but a caller walking a hand-built
        in-memory document (a test, a script) gets a bounded answer rather
        than an infinite loop if it does.
        """
        out: list[int] = []
        seen = {uid}
        current = self.by_uid(uid).parent
        while current is not None and current not in seen:
            try:
                obj = self.by_uid(current)
            except KeyError:
                break
            out.append(current)
            seen.add(current)
            current = obj.parent
        return out

    def descendants(self, uid: int) -> list[int]:
        """Every uid under *uid*, depth-first, document order per level.

        An explicit stack, not recursion -- the same shape :meth:`ancestors`
        already uses three lines above, and for the same reason: the 2026-09-19
        audit's clay-02 found this walk raised an uncaught ``RecursionError``
        on a legal, acyclic parent chain of a few thousand objects (well
        inside ``glbimport.MAX_OBJECTS``), reachable by simply selecting any
        object with the properties panel open (``props.py`` calls this on
        every selection). Children are pushed in reverse so the stack still
        pops them in document order, one root's whole subtree finished before
        its next sibling starts -- exactly what the old recursive ``walk``
        produced.
        """
        out: list[int] = []
        stack = list(reversed(self.children_of(uid)))
        while stack:
            u = stack.pop()
            if u in out:  # cycle guard; see ancestors()'s own
                continue
            out.append(u)
            stack.extend(reversed(self.children_of(u)))
        return out

    def roots(self) -> list[int]:
        """Every parentless object, in document order."""
        return [obj.uid for obj in self.objects if obj.parent is None]

    def world_matrix(self, uid: int) -> np.ndarray:
        """*uid*'s world transform: its own local TRS, composed through every
        ancestor's -- ancestor-first, so a root's own local TRS *is* its
        world matrix (``ancestors`` gives nothing to compose through)."""
        chain = [uid, *self.ancestors(uid)]
        matrix = m3.identity()
        for u in reversed(chain):
            obj = self.by_uid(u)
            matrix = matrix @ m3.compose(obj.translation, obj.rotation, obj.scale)
        return matrix

    def _local_relative(self, world: np.ndarray, parent: int | None) -> tuple[Any, Any, Any]:
        """*world* re-expressed as local TRS relative to *parent* (or as-is
        for a root) -- ``(t, r, s)``, via :func:`~.viewer.math3d.decompose`.

        Shared by :meth:`set_parent` (the new parent), :meth:`local_from_world`
        (the object's own current parent) and :meth:`remove_object` (the
        removed object's own parent) -- three callers wanting the same "what
        would this world matrix be, named relative to that uid" question.
        """
        if parent is None:
            target = np.asarray(world, dtype="f8")
        else:
            parent_world = self.world_matrix(parent)
            try:
                inverse = np.linalg.inv(parent_world)
            except np.linalg.LinAlgError as error:
                raise el.OpError(
                    f"{self.by_uid(parent).name!r} has a zero scale, so nothing can be "
                    "placed relative to it."
                ) from error
            target = inverse @ np.asarray(world, dtype="f8")
        return m3.decompose(target)

    def local_from_world(self, uid: int, matrix: np.ndarray) -> tuple[Any, Any, Any]:
        """*matrix*, a world transform, as ``(t, r, s)`` local to *uid*'s own
        current parent -- what a gizmo writes back after dragging in world
        space on a parented object."""
        obj = self.by_uid(uid)
        return self._local_relative(matrix, obj.parent)

    def _refuse_if_locked(self, uid: int, *, check_ancestors: bool = False) -> None:
        """Raise :class:`~.elements.OpError` if *uid* is locked -- and, when
        *check_ancestors*, if any ancestor is, too.

        Every door named in the module docstring's locking paragraph calls
        this first, before it mutates anything: "refuse before the
        allocation" applies here exactly as it does to every other kernel
        refusal in this package. ``check_ancestors`` is :meth:`set_transform`'s
        alone -- moving an object *inside* a locked group still visibly
        rearranges the group even though the group's own geometry never
        changes, which is not true of the other doors (a mesh edit, a
        modifier stack, a delete) that only ever affect the object itself.
        """
        obj = self.by_uid(uid)
        if obj.locked:
            raise el.OpError(f"{obj.name!r} is locked.")
        if check_ancestors:
            for a in self.ancestors(uid):
                ancestor = self.by_uid(a)
                if ancestor.locked:
                    raise el.OpError(
                        f"{obj.name!r} is locked: its parent {ancestor.name!r} is locked."
                    )

    def set_parent(self, uid: int, parent: int | None, *, keep_world: bool = True) -> bool:
        """Reparent *uid* onto *parent* (or make it a root), as one step.

        Refuses -- :class:`~.elements.OpError`, nothing pushed -- parenting
        *uid* to itself or to one of its own descendants: a document has no
        way to compose a cycle's world matrix.

        Not a locking door **when** ``keep_world`` (see the module
        docstring's locking paragraph and its own list): reparenting with
        ``keep_world`` moves nothing on screen, so it is a re-framing rather
        than a change to what the object looks like, the same reasoning
        :meth:`set_origin` is exempted under. ``keep_world=False`` does not
        hold that reasoning -- the object's local TRS is left untouched
        while its parent changes, so it visibly jumps to wherever the new
        parent's frame puts it -- so that call *is* gated by
        :meth:`_refuse_if_locked`, the 2026-09-22 audit's clay-04: an agent
        calling ``clay_parent`` with ``keep_world=False`` moved a locked
        object with no refusal at all.

        With ``keep_world`` (the default) the object's local TRS is
        recomputed from its *current* world matrix before the parent changes,
        so nothing in the viewport moves; the transform change and the parent
        change ride in the same :class:`~..core.undo.CompoundEdit` -- one
        Ctrl+Z undoes both, or the parent would change on one press and the
        object would visibly jump on the next.
        """
        obj = self.by_uid(uid)
        if parent is not None:
            self.by_uid(parent)  # KeyError names an unknown uid, the usual way
            if parent == uid or parent in self.descendants(uid):
                raise el.OpError(
                    f"{obj.name!r} cannot be parented to itself or to one of its own "
                    "descendants."
                )
        if obj.parent == parent:
            return False
        if not keep_world:
            self._refuse_if_locked(uid)

        old_parent = obj.parent
        edits: list[Any] = []
        if keep_world:
            world = self.world_matrix(uid)
            t, r, s = self._local_relative(world, parent)
            before_trs = tuple(np.array(v, copy=True) for v in obj.trs())
            after_trs = (t, r, s)
            if not all(np.array_equal(a, b) for a, b in zip(before_trs, after_trs, strict=True)):
                obj.translation, obj.rotation, obj.scale = after_trs
                edits.append(TransformEdit(uid, before_trs, after_trs))
        obj.parent = parent
        edits.append(ObjectPropsEdit(uid, {"parent": old_parent}, {"parent": parent}))
        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        self.touch()
        return True

    def set_origin(self, uid: int, world_point: Sequence[float]) -> bool:
        """Move *uid*'s origin (its local ``(0, 0, 0)``) to a world point, as
        one step: the mesh and every child's placement stay exactly where
        they were on screen, only the pivot moves.

        The mesh is shifted by the inverse of the delta the origin moved by,
        in the object's own local frame; the translation moves by the delta;
        and each direct child's local TRS is recomputed from its own
        (unchanged) world matrix, relative to the object's *new* one -- the
        same "reparent, keeping world" arithmetic :meth:`set_parent` and
        :meth:`remove_object` use, applied to a parent whose local frame
        moved under its children rather than one that changed identity.

        Not a locking door, like :meth:`set_parent` (see its own docstring):
        nothing on screen moves.

        A mirror modifier's plane is the object's own local origin (see
        :mod:`.modifiers`), so moving the origin moves that plane too --
        intended, not a bug: the modifier reads the object's current frame
        exactly as it always did, and the frame is what just changed.

        Freezes the generator, exactly as :meth:`set_mesh` does and for the
        same reason: geometry that has been re-based to a new origin is not
        what a generator would build from its stored parameters.
        """
        obj = self.by_uid(uid)
        point = np.asarray(world_point, dtype="f8")
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("world_point must be 3 finite numbers.")

        parent_world = self.world_matrix(obj.parent) if obj.parent is not None else m3.identity()
        try:
            parent_inv = np.linalg.inv(parent_world)
        except np.linalg.LinAlgError as error:
            raise el.OpError(
                f"{obj.name}'s parent has a zero scale, so its origin cannot move."
            ) from error
        new_translation = (parent_inv @ np.array([point[0], point[1], point[2], 1.0]))[:3]
        if np.array_equal(new_translation, obj.translation):
            return False

        world_old = self.world_matrix(uid)
        world_new = parent_world @ m3.compose(new_translation, obj.rotation, obj.scale)
        try:
            shift = np.linalg.inv(world_new) @ world_old
        except np.linalg.LinAlgError as error:
            raise el.OpError(f"{obj.name} has a zero scale, so its origin cannot move.") from error

        children = self.children_of(uid)
        child_worlds_old = {c: self.world_matrix(c) for c in children}
        try:
            world_new_inv = np.linalg.inv(world_new)
        except np.linalg.LinAlgError as error:
            raise el.OpError(
                f"{obj.name} has a zero scale, so its children cannot be corrected."
            ) from error

        edits: list[Any] = []
        before_mesh, obj.mesh = obj.mesh, bm.transformed(obj.mesh, shift)
        edits.append(MeshEdit(uid, before_mesh, obj.mesh))
        if obj.generator is not None:
            was_gen = {"generator": obj.generator, "params": obj.params}
            obj.generator, obj.params = None, {}
            edits.append(ObjectPropsEdit(uid, was_gen, {"generator": None, "params": {}}))
        before_trs = tuple(np.array(v, copy=True) for v in obj.trs())
        obj.translation = new_translation
        edits.append(TransformEdit(uid, before_trs, obj.trs()))
        for c in children:
            child = self.by_uid(c)
            t, r, s = m3.decompose(world_new_inv @ child_worlds_old[c])
            before_c = tuple(np.array(v, copy=True) for v in child.trs())
            child.translation, child.rotation, child.scale = t, r, s
            edits.append(TransformEdit(c, before_c, (t, r, s)))

        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        self.touch()
        return True

    def group(self, uids: Iterable[int], name: str | None = None) -> Obj:
        """A new, mesh-less :class:`Obj` at *uids*' combined world bounds
        centre, with every uid parented onto it, as one step.

        See the module docstring's "a group is parenting to an empty object"
        decision: there is no second collection concept. The empty draws
        nothing (:func:`to_primitives` already returns ``[]`` for a
        zero-face mesh) and exports as a transform-only glTF node.

        Refuses (OpError, nothing pushed) an empty *uids*: there is no bounds
        to place the empty at and nothing to parent.
        """
        from . import ops as mesh_ops

        members = [int(u) for u in uids]
        if not members:
            raise el.OpError("Select at least one object to group.")
        for u in members:
            self.by_uid(u)  # KeyError names any bad uid the usual way

        lo = hi = None
        for u in members:
            member = self.by_uid(u)
            box = mesh_ops.world_box(member, mesh=self.evaluated(u), world=self.world_matrix(u))
            if box is None:
                continue
            b_lo, b_hi = box
            lo = b_lo if lo is None else np.minimum(lo, b_lo)
            hi = b_hi if hi is None else np.maximum(hi, b_hi)
        center = (lo + hi) * 0.5 if lo is not None else np.zeros(3, dtype="f8")

        taken = {o.name for o in self.objects}
        empty_name = name or "Group"
        if empty_name in taken:
            empty_name = mesh_ops.next_name(empty_name, taken)
        empty_obj = Obj(uid=new_uid(), name=empty_name, mesh=_empty_mesh(), translation=center)

        mark = self.history.mark()
        self.add_object(empty_obj)
        for u in members:
            self.set_parent(u, empty_obj.uid, keep_world=True)
        self.history.collapse_since(mark)
        self.touch()
        return empty_obj

    # -- saving ------------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self.history.head != self.saved_head

    def mark_saved(self) -> None:
        self.saved_head = self.history.head

    # -- history -----------------------------------------------------------

    def undo(self) -> bool:
        """Reverse the newest step, dropping any element selection it invalidates.

        The step is read *before* it is applied, because by the time ``undo``
        returns the stack has already moved it out of reach. Only mesh and
        object edits clear anything: a rename or a gizmo drag leaves the same
        vertices selected on the same mesh, and clearing there would make every
        Ctrl+Z in element mode feel like it deselected something at random.
        """
        edit = self.history.top
        if not self.history.undo(self):
            return False
        self._forget_elements(edit)
        return True

    def redo(self) -> bool:
        edit = self.history.redo_top
        if not self.history.redo(self):
            return False
        self._forget_elements(edit)
        return True

    def step_history(self, index: int) -> bool:
        """Jump to a position in the undo stack. -> whether anything moved.

        ``index`` is the count of *done* steps, which is what
        ``UndoStack.step_to`` takes and what the history panel's rows stand
        for: 0 is the document as it was opened.

        Through :meth:`undo` and :meth:`redo` rather than
        ``self.history.step_to(self, n)``, and that is the whole reason this
        method exists rather than the call site doing it: ``step_to`` walks the
        *stack's* own undo and redo, which do not run :meth:`_forget_elements`.
        Called straight, a jump made in edge mode would leave the selection
        naming edges of a mesh the jump had replaced -- the exact defect the two
        methods above exist to prevent, reintroduced by the one caller that
        reached past them. ``PlotterDoc.step_history`` is the same method for
        the same reason, ending its open sessions instead.
        """

        total = len(self.history.history())
        wanted = max(0, min(int(index), total))
        moved = False
        while len(self.history) > wanted and self.undo():
            moved = True
        while len(self.history) < wanted and self.redo():
            moved = True
        return moved

    def _forget_elements(self, edit: Edit | None) -> None:
        for uid in _geometry_uids(edit):
            gone = self.element_sel.pop(uid, None) is not None
            if gone and self.element_mode != "object":
                self.selection.discard(uid)
        self.touch()

    # -- objects -----------------------------------------------------------

    def add_object(self, obj: Obj, index: int | None = None) -> Obj:
        """Insert an object and record the step. Returns the object it was given
        so a caller can place and keep hold of one in a single expression."""
        at = len(self.objects) if index is None else index
        self.objects.insert(at, obj)
        self.history.push(ObjectAddEdit(at, obj))
        self.touch()
        return obj

    def add_objects(self, objs: Iterable[Obj], label: str = "") -> list[Obj]:
        """Insert several objects as **one** step, and select all of them.

        The figure presets build sixteen parts at once, and sixteen
        ``add_object`` calls are sixteen ``ObjectAddEdit`` pushes -- so undoing
        a humanoid you did not want is sixteen presses of Ctrl+Z, through
        fifteen intermediate states that are a dismembered figure standing in
        the viewport. One assembly is one gesture, so it is one step, for the
        reason ``set_visibility`` and :meth:`join_objects` are: an undo history
        whose entries are not the actions the user took is not a history.

        Built the way ``join_objects`` builds its own compound -- edits
        collected in the order they were applied, ``CompoundEdit`` undoing them
        in reverse, which pops the objects off the end of the list first and so
        keeps every recorded index correct on the way back out.

        Empty is a no-op that pushes nothing at all, for ``set_visibility``'s
        reason: a step that changes nothing makes a saved document ask to be
        saved again.
        """
        added = list(objs)
        if not added:
            return []
        edits: list[Any] = []
        for obj in added:
            at = len(self.objects)
            self.objects.insert(at, obj)
            edits.append(ObjectAddEdit(at, obj))
        edit = edits[0] if len(edits) == 1 else CompoundEdit(edits)
        if label:
            edit.label = label
        self.history.push(edit)
        # Object mode only (the 2026-09-13 audit's clay-03): ``select`` names
        # only the *object* selection, but in an element mode the invariant
        # (see ``set_element_mode``'s own docstring) is that ``selection`` is
        # the set of objects with something selected inside ``element_sel`` --
        # calling it unconditionally left ``selection`` naming the new object
        # while ``element_sel`` still named whatever was being edited, so the
        # Properties panel's summary line and its identity/transform/material
        # rows disagreed about which object they were for.
        if self.element_mode == "object":
            self.select([obj.uid for obj in added])
        self.touch()
        return added

    def remove_object(self, uid: int) -> bool:
        """Delete *uid*, re-parenting its children onto its own parent, as
        **one** step.

        Refuses (OpError, nothing pushed) a locked object -- one of the
        locking doors the module docstring lists.

        A child re-parented straight to ``None`` (the removed object's own
        parent) rather than left naming a uid the document no longer has:
        the alternative, an orphaned ``parent`` value, is exactly the
        dangling reference :mod:`.serialize` refuses to load and
        :meth:`world_matrix` would otherwise have to guess about. World
        placement is kept, the same "reparenting keeps world placement"
        decision :meth:`set_parent` states, computed from each child's
        current world matrix *before* the object it is relative to is gone.
        """
        obj = self.by_uid(uid)
        if obj.locked:
            raise el.OpError(f"{obj.name!r} is locked.")
        new_parent = obj.parent
        children = self.children_of(uid)
        child_worlds = {c: self.world_matrix(c) for c in children}

        edits: list[Any] = []
        for c in children:
            child = self.by_uid(c)
            t, r, s = self._local_relative(child_worlds[c], new_parent)
            before = {"parent": child.parent}
            child.parent = new_parent
            edits.append(ObjectPropsEdit(c, before, {"parent": new_parent}))
            before_trs = tuple(np.array(v, copy=True) for v in child.trs())
            after_trs = (t, r, s)
            if not all(np.array_equal(a, b) for a, b in zip(before_trs, after_trs, strict=True)):
                child.translation, child.rotation, child.scale = after_trs
                edits.append(TransformEdit(c, before_trs, after_trs))

        # Reparenting above touches only ``parent``/TRS fields, never list
        # order, so this is *uid*'s original position in ``self.objects``.
        index = self.index_of(uid)
        obj = self.objects.pop(index)
        self.selection.discard(uid)
        self.element_sel.pop(uid, None)
        # A stamp naming an object that no longer exists is worse than a
        # missing one: undo can bring the uid back with a fresh object at some
        # later address, and an old stamp entry would otherwise linger keyed to
        # a mesh that object never had.
        self._mesh_stamps.pop(uid, None)
        # Same reasoning as the stamp above: a lingering entry would be keyed
        # to a mesh and a stack this uid no longer has the moment undo puts a
        # *different* object back on it, though evaluate()'s own identity and
        # equality checks would also catch that on the next read.
        self._evaluated.pop(uid, None)
        edits.append(ObjectRemoveEdit(index, obj))
        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        self.touch()
        return True

    def move_object(self, uid: int, index: int) -> bool:
        """Put an object at *index* in the list, and record the step.

        Display order is not decoration: ``to_model`` and ``write_glb`` walk the
        list, so it is the order the nodes come out in and the order an engine
        importing the GLB will show. That is what makes this an *edit* rather
        than a view setting, and why it goes on the undo stack beside the others.

        The index is clamped rather than validated, because the caller is a drag:
        a row dropped past the end of the list means the end of the list, and
        there is nothing sensible for a refusal to show mid-gesture.
        """
        at = self.index_of(uid)
        target = max(0, min(int(index), len(self.objects) - 1))
        if target == at:
            return False
        obj = self.objects.pop(at)
        self.objects.insert(target, obj)
        self.history.push(ObjectMoveEdit(uid, at, target))
        self.touch()
        return True

    def set_visibility(self, wanted: dict[int, bool]) -> bool:
        """Several objects' visibility as **one** step.

        One step because that is what the gesture is: isolating an object is a
        single click and hiding nine things one Ctrl+Z at a time is not an undo
        history, it is a punishment. Objects whose visibility already agrees are
        left out entirely rather than recorded as no-ops, which is what keeps
        "isolate the thing that is already isolated" from making a saved
        document ask to be saved again.
        """
        edits = []
        for uid, visible in wanted.items():
            try:
                obj = self.by_uid(uid)
            except KeyError:
                continue
            if bool(obj.visible) == bool(visible):
                continue
            edits.append(ObjectPropsEdit(uid, {"visible": obj.visible}, {"visible": visible}))
            obj.visible = bool(visible)
        if not edits:
            return False
        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        self.touch()
        return True

    def isolate(self, uids: Iterable[int]) -> bool:
        """Show only these objects. One step, and its own inverse is
        :meth:`show_all` rather than a second isolate."""
        keep = set(uids)
        return self.set_visibility({obj.uid: obj.uid in keep for obj in self.objects})

    def show_all(self) -> bool:
        return self.set_visibility({obj.uid: True for obj in self.objects})

    def set_mesh(
        self,
        uid: int,
        mesh: bm.Mesh,
        *,
        select: el.ElementSel | None = None,
        keep_generator: bool = False,
    ) -> bool:
        """Replace one object's geometry as one step, and freeze its generator.

        Identity, not equality, decides whether anything happened: ``Mesh`` is
        ``eq=False`` because numpy arrays have no truthy ``==``, and every op
        is ``Mesh -> Mesh``, so the same object *is* the same geometry.

        ``select`` is the selection the op wants shown next -- extrude hands
        back its caps so the user can drag them straight away. It is applied
        *after* the push and pushes nothing of its own, because selection is
        not undoable; undoing the mesh edit then drops it, which is the right
        answer for a selection describing geometry that no longer exists.

        **The freeze lives here rather than in the op layer**, which is where
        it was and where it was only half applied: ``clay_ops.run_mesh_op`` and
        Smooth cleared ``generator``, while Delete, Bake Transform, Mirror and
        an element drag did not. An object that still claims to be "box, size
        1" keeps offering that size field, and touching it rebuilds a pristine
        box -- so the deletion, the bake, the mirror or the drag vanished with
        no warning. Geometry that is no longer what a generator would build is
        a fact about ``set_mesh``, not about which caller remembered.

        The freeze is pushed *with* the mesh edit, as one ``CompoundEdit``:
        two steps meant one Ctrl+Z restored the generator claim over the
        still-edited mesh -- the exact state the freeze exists to prevent --
        and the user had to press again to get out of it.

        ``keep_generator`` is the single exception, for the properties panel's
        own rebuild: there the new mesh *is* what the generator makes, which is
        the one case where the claim is still true.

        **Tranche 6: seams are restricted, not carried blindly.** Called by
        dozens of different mesh ops with no shared way to say whether *this*
        particular call kept old vertex indices meaning the same vertex, so
        :func:`_restrict_seams` -- the same range check :func:`~.elements.
        restrict` already applies to an element selection -- drops any seam
        pair naming a vertex the new mesh no longer has and keeps the rest,
        in the same step as the mesh replacement. See that function's own
        docstring for why a range check is the honest answer here.

        Refuses (OpError, nothing pushed) a locked object -- one of the
        locking doors the module docstring lists.
        """
        self._refuse_if_locked(uid)
        obj = self.by_uid(uid)
        if mesh is obj.mesh:
            return False
        before, obj.mesh = obj.mesh, mesh
        edits: list[Any] = [MeshEdit(uid, before, mesh)]
        if not keep_generator and obj.generator is not None:
            # One step, not two. Pushed separately, a single Ctrl+Z restored
            # the generator claim over the still-edited mesh -- the exact state
            # the freeze exists to prevent -- and only a second press undid the
            # edit it belongs to.
            was = {"generator": obj.generator, "params": obj.params}
            obj.generator, obj.params = None, {}
            edits.append(ObjectPropsEdit(uid, was, {"generator": None, "params": {}}))
        seams_before = obj.seams
        seams_after = _restrict_seams(seams_before, mesh)
        if seams_after is not seams_before:
            obj.seams = seams_after
            edits.append(ObjectPropsEdit(uid, {"seams": seams_before}, {"seams": seams_after}))
        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        if select is not None:
            self.set_element_sel(uid, select)
        self.touch()
        return True

    def join_objects(
        self,
        target_uid: int,
        mesh: bm.Mesh,
        others: Iterable[int],
        *,
        clear_modifiers: bool = True,
    ) -> bool:
        """Adopt a merged mesh and drop the objects it absorbed, as **one** step.

        One ``CompoundEdit`` and not a ``set_mesh`` followed by N
        ``remove_object`` calls, for the reason ``set_mesh``'s own freeze is one
        step: a Ctrl+Z that put back one of the absorbed objects while the
        target still carried the merged geometry would show the user a document
        in which that shape exists twice -- a state that never happened, and one
        it takes another N presses to leave.

        The removals are recorded in *descending* index order so each
        ``ObjectRemoveEdit`` names the index the object actually sat at when it
        was popped; ``CompoundEdit`` undoes in reverse, which re-inserts them
        ascending, which is the only order in which those indices are all still
        correct.

        The generator freeze applies here exactly as it does in ``set_mesh``:
        a box merged with a sphere is not a box, and leaving the claim would let
        the properties panel rebuild a pristine box over the merge.

        **Merging ops consume evaluated meshes.** Join, union, difference and
        intersection all hand this the target's *evaluated* mesh, not its
        base -- a target with a mirror on it merges the mirrored shape, not
        half of it -- and ``clear_modifiers`` (on by default) drops the
        target's stack in the very same step: its modifiers are now baked
        into what this adopted, and leaving them in place would apply them a
        second time the next time the target was drawn. A caller that has
        instead handed over the *base* mesh unchanged -- there is none today,
        but the door is real -- passes ``clear_modifiers=False`` to say so.

        **Tranche 6: the target's own seams do not survive a real merge.**
        The merged mesh is a weld/concatenation or a wholly recomputed
        boolean arrangement, and this method is never handed the
        correspondence between the target's old vertex indices and the new
        ones -- unlike ``set_mesh``, which at least gets a plain range check
        (see ``_restrict_seams``'s own docstring for why that would not be
        honest here). Every seam on the target is dropped when the mesh
        actually changes; a call that changes nothing about the target's own
        geometry (``doomed`` absorbed nothing whose mesh differed) leaves
        them alone.
        """
        # The 2026-09-20 audit's clay-01: this door never called
        # _refuse_if_locked at all, for either half of what it does -- the
        # target's mesh is overwritten and the absorbed objects are deleted,
        # and a locked object survived neither through ordinary clicks or
        # through clay_join/clay_boolean. Both refusals land before either
        # mutation, exactly as every other locking door in this module does.
        self._refuse_if_locked(target_uid)
        for other in others:
            other_uid = int(other)
            if other_uid == target_uid:
                continue
            other_obj = self.by_uid(other_uid)
            if other_obj.locked:
                raise el.OpError(f"{other_obj.name!r} is locked.")
        obj = self.by_uid(target_uid)
        doomed = sorted({int(u) for u in others} - {target_uid}, key=self.index_of, reverse=True)
        if mesh is obj.mesh and not doomed:
            return False
        before, obj.mesh = obj.mesh, mesh
        edits: list[Any] = [MeshEdit(target_uid, before, mesh)]
        props_before: dict[str, Any] = {}
        props_after: dict[str, Any] = {}
        if obj.generator is not None:
            props_before["generator"], props_before["params"] = obj.generator, obj.params
            props_after["generator"], props_after["params"] = None, {}
        if clear_modifiers and obj.modifiers:
            props_before["modifiers"] = obj.modifiers
            props_after["modifiers"] = ()
        # Tranche 6: no seam on the target survives a merge whose mesh
        # actually changed (``mesh is not before``, the same "did the
        # identity change" test the freeze above reads off ``obj.generator``).
        # ``ops.join``'s own docstring says the merged result is a weld/
        # concatenation and ``ops_boolean``'s a wholly recomputed
        # arrangement -- either way the target's old vertex index and the
        # merged mesh's are not the same question, and this method is never
        # handed the correspondence between them (unlike ``set_mesh``, which
        # at least gets a plain range check -- see ``_restrict_seams``'s own
        # docstring for why that is not honest here). A no-op merge
        # (``mesh is before``, reachable only when ``doomed`` is non-empty --
        # see the early return above) leaves the target's own geometry, and
        # so its seams, untouched.
        if mesh is not before and obj.seams:
            props_before["seams"] = obj.seams
            props_after["seams"] = ()
        if props_after:
            for key, value in props_after.items():
                setattr(obj, key, value)
            edits.append(ObjectPropsEdit(target_uid, props_before, props_after))
        for uid in doomed:
            # The 2026-09-23 audit's clay-01: this loop used to pop each
            # doomed object without re-parenting its children, unlike
            # remove_object (above) which re-parents onto the removed
            # object's own parent before popping it. A child of an absorbed
            # object then kept a ``parent`` uid the document no longer
            # carried -- it jumped in world space with no undo step, and
            # serialize._validate_hierarchy refused to reload the saved
            # file at all. Same fix as remove_object: re-parent onto the
            # doomed object's own parent, keeping world placement, computed
            # from each child's current world matrix before the object it
            # is relative to is gone, in the same compound.
            new_parent = self.by_uid(uid).parent
            for c in self.children_of(uid):
                child = self.by_uid(c)
                child_world = self.world_matrix(c)
                t, r, s = self._local_relative(child_world, new_parent)
                before = {"parent": child.parent}
                child.parent = new_parent
                edits.append(ObjectPropsEdit(c, before, {"parent": new_parent}))
                before_trs = tuple(np.array(v, copy=True) for v in child.trs())
                after_trs = (t, r, s)
                moved = zip(before_trs, after_trs, strict=True)
                if not all(np.array_equal(a, b) for a, b in moved):
                    child.translation, child.rotation, child.scale = after_trs
                    edits.append(TransformEdit(c, before_trs, after_trs))
            index = self.index_of(uid)
            gone = self.objects.pop(index)
            self.selection.discard(uid)
            self.element_sel.pop(uid, None)
            self._mesh_stamps.pop(uid, None)
            self._evaluated.pop(uid, None)
            edits.append(ObjectRemoveEdit(index, gone))
        # The target's own element selection names vertices of the mesh that
        # has just been replaced, so it describes geometry that is no longer
        # there -- the same reason ``_forget_elements`` drops one after an undo.
        self.element_sel.pop(target_uid, None)
        self.history.push(CompoundEdit(edits))
        self._evaluated.pop(target_uid, None)
        self.touch()
        return True

    def set_transform(
        self,
        uid: int,
        *,
        translation: Sequence[float] | None = None,
        rotation: Sequence[float] | None = None,
        scale: Sequence[float] | None = None,
        was: tuple[Any, Any, Any] | None = None,
    ) -> bool:
        """Move, rotate and scale as one step, pushing nothing for a no-op.

        ``was`` is for a gizmo that mutates the object live so the viewport
        follows the drag and only asks for the step when the drag is released:
        by then the object already holds the new values, so reading "before"
        off it would compare a value against itself and record nothing. What
        such a caller passes must be values the drag cannot reach: ``trs()``
        hands back the object's live arrays, so a gizmo that writes through
        them (``obj.translation[0] = x``) rather than rebinding would find its
        own ``was`` had moved with it. Rebind, as this method does. Setting
        a value to the one already there pushes no step at all, because dirty
        is a comparison against the head and a no-op step makes a saved
        document ask to be saved again.

        A per-field shape and finiteness assertion is the backstop here, not
        the message a caller sees -- ``agent_clay``'s ``_h_transform``
        validates its own arguments before ever reaching this method, but an
        unvalidated ``clay_transform`` once committed a two-element
        ``translation`` straight through this method with nothing to notice
        the wrong shape, and every later ``clay_scene`` raised trying to
        broadcast it into a 3x3 matrix (``viewer/math3d.py``'s ``compose``
        does ``m[:3, 3] = t``) -- bricking introspection for the whole
        document, with no recovery but a blind undo. This method has other
        callers than ``_h_transform`` -- the properties panel, the gizmo
        drag, ``clay_ops._bake`` -- so the assertion belongs here too,
        closing the door for every caller, present and future, rather than
        trusting each one to have validated first.

        Refuses (OpError, nothing pushed) a locked object *or one with a
        locked ancestor* -- the one locking door in the module docstring
        that checks the ancestor chain too: dragging an object inside a
        locked group still visibly rearranges the group, even though the
        group's own geometry never changes.
        """
        self._refuse_if_locked(uid, check_ancestors=True)
        obj = self.by_uid(uid)
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
        before = tuple(np.array(v, dtype="f8", copy=True) for v in (was or obj.trs()))
        after = tuple(
            obj.trs()[i] if new is None else np.array(new, dtype="f8", copy=True)
            for i, new in enumerate((translation, rotation, scale))
        )
        if all(np.array_equal(a, b) for a, b in zip(before, after, strict=True)):
            return False
        obj.translation, obj.rotation, obj.scale = after
        self.history.push(TransformEdit(uid, before, after))  # type: ignore[arg-type]
        self.touch()
        return True

    def set_props(self, uid: int, *, was: dict[str, Any] | None = None, **props: Any) -> bool:
        """Name, visibility, tags, locked, generator, params, default material
        -- one step. Deliberately **not** a locking door (see the module
        docstring's locking paragraph): a locked object still allows a
        rename, a visibility change, a tag edit and unlocking itself, or a
        mistake made while locked could not be undone by anyone but the lock.

        ``was`` is the counterpart of :meth:`set_transform`'s, and the trap it
        avoids is sharper here because ``params`` is a dict: a panel that edits
        the object's own dict in place and then passes it back would hand this
        the very object it is comparing against, so "before" and "after" would
        be the same value and the change would record nothing at all. Such a
        caller passes the values it started with as ``was``; a caller that
        builds a fresh dict -- which is what a widget reading a form does --
        needs none of this.

        ``tags`` is normalized (:func:`_normalize_tags`) before it is
        compared or stored, so a caller handing over ``["Prop", "prop"]``
        neither records a change against an object already tagged ``prop``
        nor stores the duplicate. ``parent`` is refused by name: it has its
        own door, :meth:`set_parent`, which is the only one that checks for
        a cycle -- this generic one does not, and must not be used to bypass
        it.
        """
        if "parent" in props:
            raise el.OpError(
                "Use set_parent to change an object's parent -- it is the only "
                "door that refuses a cycle."
            )
        if "tags" in props:
            props = dict(props)
            props["tags"] = _normalize_tags(props["tags"])
        obj = self.by_uid(uid)
        source = {} if was is None else was
        before = {key: source.get(key, getattr(obj, key)) for key in props}
        if before == props:
            return False
        for key, value in props.items():
            setattr(obj, key, value)
        self.history.push(ObjectPropsEdit(uid, before, dict(props)))
        self.touch()
        return True

    def set_generator_params(
        self, uid: int, params: dict[str, Any], mesh: bm.Mesh, *, was: dict[str, Any]
    ) -> bool:
        """A generator's numbers and the mesh they build, as **one** step.

        The properties panel used to call :meth:`set_props` and then
        :meth:`set_mesh` as two independent edits, which is the shape
        :meth:`set_mesh`'s own docstring argues against for the freeze: a lone
        Ctrl+Z restoring half the pair leaves the viewport showing the old mesh
        while the panel still reads the new radius. imgui's ``InputFloat`` fires
        per keystroke, so typing a multi-digit number pushed several such pairs
        for one felt edit.

        ``was`` is :meth:`set_props`' own, and mandatory here rather than
        optional: this caller edits the object's ``params`` dict in place before
        it gets here, so reading "before" off the object would compare a value
        against itself.

        ``keep_generator`` is implied. The new mesh *is* what the generator
        makes from these parameters, which is the one case where the object's
        claim to be "box, size 1" is still true.

        Refuses (OpError, nothing pushed) a locked object -- one of the
        locking doors the module docstring lists.
        """
        self._refuse_if_locked(uid)
        obj = self.by_uid(uid)
        before = {"params": was.get("params", obj.params)}
        edits: list[Any] = []
        if before != {"params": params}:
            obj.params = params
            edits.append(ObjectPropsEdit(uid, before, {"params": params}))
        if mesh is not obj.mesh:
            was_mesh, obj.mesh = obj.mesh, mesh
            edits.append(MeshEdit(uid, was_mesh, mesh))
            # A params rebuild is the one path that shrinks a mesh *outside*
            # undo: every other way an object loses faces -- Delete, a mesh
            # op, an undo/redo -- goes through ``set_mesh`` or is caught by
            # ``_forget_elements`` on the way back, but this method replaces
            # ``obj.mesh`` directly and pushes a brand-new step, not a
            # reversal. A face selection made before a segment-count edit
            # therefore used to survive verbatim into a mesh with fewer
            # faces, holding indices the overlay build and every element-mode
            # tool would index straight past the end of. ``el.restrict`` is
            # exactly the guard for this -- see its own docstring, which
            # until now had to admit no caller used it -- and going through
            # ``set_element_sel`` rather than writing ``self.element_sel``
            # directly keeps the derived-selection invariant this module's
            # own docstring states: an emptied restriction also drops the
            # uid from ``selection``, the same as every other path that
            # touches ``element_sel``. Not undoable, same as every other
            # selection change: it pushes nothing of its own.
            #
            # ``prior=was_mesh`` closes the 2026-09-19 audit's clay-17: this
            # is the one caller that already holds both the pre-edit mesh and
            # the rebuilt one, which is exactly what ``el.restrict`` needs to
            # tell a same-count topology change (every index still "in
            # range", none of it still meaning the same geometry) from an
            # honest shrink -- see that function's own docstring.
            existing = self.element_sel.get(uid)
            if existing is not None:
                self.set_element_sel(uid, el.restrict(mesh, existing, prior=was_mesh))
            # Tranche 6: the exact same range check, for the exact same
            # reason -- see ``_restrict_seams``'s own docstring, which names
            # this method as one of its two callers.
            seams_before = obj.seams
            seams_after = _restrict_seams(seams_before, mesh)
            if seams_after is not seams_before:
                obj.seams = seams_after
                edits.append(ObjectPropsEdit(uid, {"seams": seams_before}, {"seams": seams_after}))
        if not edits:
            return False
        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        self.touch()
        return True

    # -- seams (tranche 6) ----------------------------------------------------

    def set_seams(self, uid: int, seams: Iterable[Sequence[int]]) -> bool:
        """Replace one object's marked seams -- authoring intent for an
        unwrap, not geometry (see the module docstring's own seams
        paragraph) -- as **one** step.

        Normalized exactly as :class:`Obj` construction already does
        (:func:`_normalize_seams`): each pair ordered ``(a, b)`` with
        ``a < b``, deduplicated, a pair naming the same vertex twice silently
        dropped. What normalization cannot fix -- a vertex this object's
        *base* mesh does not have -- is refused (:class:`~.elements.OpError`,
        nothing pushed), naming the offending pair, the same "refuse before
        the allocation" shape every kernel refusal in this package follows.
        Contrast :func:`_restrict_seams`, which *silently* range-checks after
        a mesh replacement nobody asked this door about -- a survivor's
        problem, not a caller's mistake to report; this is the door a person
        or an agent calls directly, so a bad index is a refusal, not a quiet
        drop.

        Refuses (OpError, nothing pushed) a locked object -- one of the
        locking doors the module docstring lists: a seam is authoring intent
        about the object's own geometry, the same footing a mesh edit stands
        on.
        """
        self._refuse_if_locked(uid)
        obj = self.by_uid(uid)
        normalized = _normalize_seams(seams)
        n = len(obj.mesh.positions)
        for a, b in normalized:
            if not (0 <= a < n and 0 <= b < n):
                raise el.OpError(
                    f"Seam ({a}, {b}) names a vertex {obj.name!r}'s mesh does not "
                    f"have -- it has {n}."
                )
        if normalized == obj.seams:
            return False
        before, obj.seams = obj.seams, normalized
        self.history.push(ObjectPropsEdit(uid, {"seams": before}, {"seams": normalized}))
        self.touch()
        return True

    # -- modifiers -----------------------------------------------------------

    def evaluation(self, uid: int) -> Any:
        """*uid*'s base mesh run through its modifier stack, errors and all.

        Returns a :class:`~.modifiers.Evaluated`. Imported lazily -- see
        :mod:`.modifiers`'s own docstring for why the import runs this
        direction and not the other.
        """
        from . import modifiers as mod

        return mod.evaluate(self, uid)

    def evaluated(self, uid: int) -> bm.Mesh:
        """:meth:`evaluation`'s mesh alone -- what display, export, measurement
        and object-mode picking read; see the module docstring."""
        return self.evaluation(uid).mesh

    def set_modifiers(self, uid: int, stack: tuple[Any, ...]) -> bool:
        """Replace one object's modifier stack, as one step.

        Refuses -- :class:`~.elements.OpError`, nothing pushed -- a stack
        naming an unknown kind, or one whose boolean targets would create a
        dependency cycle anywhere in the document (:func:`~.modifiers.
        would_cycle`), a self-target included. Both checks run *before* the
        step is recorded, the same "refuse before the allocation" shape every
        kernel refusal in this package follows -- there is nothing to undo a
        refusal out of.

        Also refuses (OpError, nothing pushed) a locked object -- one of the
        locking doors the module docstring lists; ``set_props`` itself is
        not one, so this checks before delegating to it.
        """
        self._refuse_if_locked(uid)
        from . import modifiers as mod

        stack = tuple(stack)
        unknown = sorted({m.kind for m in stack} - set(mod.MODIFIERS))
        if unknown:
            raise el.OpError(
                f"Unknown modifier kind {unknown[0]!r}. Choose one of "
                f"{', '.join(sorted(mod.MODIFIERS))}."
            )
        if mod.would_cycle(self, uid, stack):
            raise el.OpError(
                "That modifier stack would create a boolean cycle -- an object "
                "cannot depend, even indirectly, on its own result."
            )
        changed = self.set_props(uid, modifiers=stack)
        if changed:
            self._evaluated.pop(uid, None)
        return changed

    def apply_modifiers(self, uid: int, through_id: int | None = None) -> bool:
        """Bake the stack's prefix through *through_id* into the base mesh.

        ``through_id=None`` bakes the whole stack. A disabled modifier inside
        the prefix is dropped without being applied -- it never contributed to
        what the user saw, so baking it in would change the shape rather than
        merely freeze it. A modifier inside the prefix that currently refuses
        refuses the *whole* apply, with its own message: never bake a
        half-result the user never saw on screen.

        One ``CompoundEdit`` -- a ``MeshEdit`` when the baked geometry differs
        from the base (a prefix of only-disabled modifiers does not), and an
        ``ObjectPropsEdit`` that always drops the baked prefix from
        ``modifiers`` and, when the object still claims a generator, freezes
        that too -- the same claim :meth:`set_mesh` freezes and for the same
        reason: geometry a modifier stack built is not what a generator would
        build.

        Refuses (OpError, nothing pushed) a locked object -- one of the
        locking doors the module docstring lists.
        """
        self._refuse_if_locked(uid)
        from . import modifiers as mod

        obj = self.by_uid(uid)
        stack = obj.modifiers
        if not stack:
            return False
        if through_id is None:
            cut = len(stack)
        else:
            ids = [m.id for m in stack]
            if through_id not in ids:
                raise el.OpError(f"This object has no modifier {through_id}.")
            cut = ids.index(through_id) + 1
        prefix, rest = stack[:cut], stack[cut:]

        mesh = obj.mesh
        ctx = mod.EvalContext(doc=self, obj=obj, visiting=frozenset({uid}))
        for m in prefix:
            if not m.enabled:
                continue
            kind_def = mod.MODIFIERS.get(m.kind)
            if kind_def is None:
                raise el.OpError(f"Unknown modifier kind {m.kind!r}.")
            mesh = kind_def.apply(mesh, m.as_dict(), ctx)

        edits: list[Any] = []
        was_mesh = obj.mesh
        if mesh is not was_mesh:
            obj.mesh = mesh
            edits.append(MeshEdit(uid, was_mesh, mesh))
        props_before: dict[str, Any] = {"modifiers": stack}
        props_after: dict[str, Any] = {"modifiers": rest}
        if obj.generator is not None:
            props_before["generator"], props_before["params"] = obj.generator, obj.params
            props_after["generator"], props_after["params"] = None, {}
        # Tranche 6: only when the bake actually changed the mesh (``mesh is
        # not was_mesh``, checked above) -- a prefix of purely disabled
        # modifiers leaves the mesh untouched, and an object's indices are
        # then, trivially, still exactly what they were, the one case in this
        # method where keeping the seams is not a guess. Once the bake *has*
        # run, ``kind_def.apply`` is an arbitrary per-kind function (mirror,
        # boolean, subdivide...) this method has no way to ask whether it
        # kept old indices meaning the same vertex -- the same "cannot know,
        # so dropped" answer :meth:`join_objects` gives its own target, for
        # the same reason -- so every seam is dropped rather than kept by
        # coincidence against an index that may no longer name the same edge.
        if mesh is not was_mesh and obj.seams:
            props_before["seams"] = obj.seams
            props_after["seams"] = ()
        for key, value in props_after.items():
            setattr(obj, key, value)
        edits.append(ObjectPropsEdit(uid, props_before, props_after))
        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        self._evaluated.pop(uid, None)
        self.touch()
        return True

    # -- separate ------------------------------------------------------------

    def separate(self, uid: int, pieces: Sequence[bm.Mesh]) -> list[Obj]:
        """*uid* replaced by one new object per *pieces*, as **one** step.

        Every piece keeps the source's parent, transform and modifier stack
        -- the stack is *copied* onto each piece (a tuple, so sharing it is
        free), unevaluated, not baked, so separating a mirrored object by
        material keeps every piece mirrored. The generator is frozen (as
        :meth:`set_mesh` freezes it): a piece of a sphere is not "sphere,
        radius 1". Names are suffixed (:func:`~.ops.next_name`) and the
        source object is removed.

        Refuses (OpError, nothing pushed) a locked source -- same reasoning
        as :meth:`remove_object`, which this is one step further than: the
        source does not survive this either -- or fewer than two pieces,
        which the kernel functions in :mod:`.separate` already refuse to
        produce; this is the same refusal for a caller that built ``pieces``
        some other way.

        **Tranche 6: no piece keeps a seam.** :mod:`.separate`'s own
        ``_piece`` compacts each piece's vertex array to only the vertices
        its own faces use and remaps ``loops`` to match (its own docstring
        says so plainly), so a piece's index 3 and the source's index 3 are,
        in general, two different vertices -- this method receives only the
        finished ``Mesh`` objects, never that remap, so there is no
        correspondence here to restrict a seam pair against, honestly or
        otherwise. Every other field a piece can meaningfully inherit
        (parent, transform, material, modifiers, tags, role, collider_kind)
        still does; seams alone start empty on every piece.
        """
        from . import ops as mesh_ops

        pieces = list(pieces)
        obj = self.by_uid(uid)
        if obj.locked:
            raise el.OpError(f"{obj.name!r} is locked.")
        if len(pieces) < 2:
            raise el.OpError("Nothing to separate: that would produce a single piece.")
        for mesh in pieces:
            bm.validate(mesh)

        taken = {o.name for o in self.objects}
        new_objs: list[Obj] = []
        for mesh in pieces:
            name = mesh_ops.next_name(obj.name, taken)
            taken.add(name)
            new_objs.append(
                Obj(
                    uid=new_uid(),
                    name=name,
                    mesh=mesh,
                    translation=np.array(obj.translation, dtype="f8", copy=True),
                    rotation=np.array(obj.rotation, dtype="f8", copy=True),
                    scale=np.array(obj.scale, dtype="f8", copy=True),
                    generator=None,
                    params={},
                    visible=obj.visible,
                    material=obj.material,
                    modifiers=obj.modifiers,
                    parent=obj.parent,
                    locked=False,
                    tags=obj.tags,
                    # seams=() (the field's own default): see this method's
                    # own docstring for why no correspondence survives a split.
                    role=obj.role,
                    collider_kind=obj.collider_kind,
                )
            )

        index = self.index_of(uid)
        edits: list[Any] = []
        # The 2026-09-23 audit's clay-02: the source used to be popped below
        # without re-parenting its children, the same orphaning join_objects
        # had through its own doomed-object loop -- a child's ``parent``
        # named a uid the document no longer carried, and serialize refused
        # to reload the saved file. Re-parent onto the source's own parent,
        # keeping world placement, before any piece is inserted or the
        # source is popped, exactly as remove_object does.
        new_parent = obj.parent
        children = self.children_of(uid)
        child_worlds = {c: self.world_matrix(c) for c in children}
        for c in children:
            child = self.by_uid(c)
            t, r, s = self._local_relative(child_worlds[c], new_parent)
            before = {"parent": child.parent}
            child.parent = new_parent
            edits.append(ObjectPropsEdit(c, before, {"parent": new_parent}))
            before_trs = tuple(np.array(v, copy=True) for v in child.trs())
            after_trs = (t, r, s)
            if not all(np.array_equal(a, b) for a, b in zip(before_trs, after_trs, strict=True)):
                child.translation, child.rotation, child.scale = after_trs
                edits.append(TransformEdit(c, before_trs, after_trs))
        for i, piece in enumerate(new_objs):
            self.objects.insert(index + i, piece)
            edits.append(ObjectAddEdit(index + i, piece))
        # The source's own new position, now pushed forward by every piece
        # inserted ahead of it -- the same re-lookup ``join_objects`` and
        # ``remove_object`` use rather than hand computing ``index + len``.
        removed_index = self.index_of(uid)
        removed = self.objects.pop(removed_index)
        self.selection.discard(uid)
        self.selection.update(o.uid for o in new_objs)
        self.element_sel.pop(uid, None)
        self._mesh_stamps.pop(uid, None)
        self._evaluated.pop(uid, None)
        edits.append(ObjectRemoveEdit(removed_index, removed))
        self.history.push(CompoundEdit(edits))
        self.touch()
        return new_objs

    # -- colliders (tranche 7) -------------------------------------------------

    def add_collider(self, source_uid: int, collider: colliders.Collider) -> Obj:
        """Add *collider* -- already fit against *source_uid*'s evaluated
        mesh by a caller through :mod:`.colliders` -- as a new child object
        of *source_uid*, as **one** step. -> the new :class:`Obj`.

        **A collider is an ordinary object with a role**, per the module
        docstring's own tranche 7 paragraph: this constructs one directly
        with ``parent=source_uid`` already set and its local TRS left at
        the identity, rather than adding it as a root and then calling
        :meth:`set_parent`. That is deliberate, not a shortcut --
        :class:`~.colliders.Collider` says plainly that its ``mesh`` is
        already expressed in the *source's own local frame* (there is no
        separate transform to compose), so the correct placement is a local
        identity under the source, not "wherever this object's world
        transform used to be, reparented" -- which is the question
        :meth:`set_parent`'s ``keep_world`` answers, and the wrong one here:
        a brand-new root object's world transform *is* the identity, and
        preserving that through a reparent would leave the collider sitting
        at the scene origin instead of on the source.

        The one thing actually validated: *collider.kind* must name an
        entry in :data:`~.colliders.COLLIDER_KINDS`. Refused
        (:class:`~.elements.OpError`, nothing pushed), naming the kind, the
        same half-read-is-worse-than-refused doctrine :mod:`.serialize`
        states for a loaded file, applied here at the door that first
        creates a collider live -- a role/kind pair readiness and an
        exporter cannot recognise is worse than one refused up front.

        Named ``"<source name> <kind label>"``, disambiguated by
        :func:`~.ops.next_name` exactly the way :meth:`group` disambiguates
        a generated empty's name -- only when that name is already taken,
        so the common case (one collider per source) is not needlessly
        suffixed. Always visible: a collider draws as a translucent fill and
        wireframe rather than shaded geometry (the 2026-09-19 audit's
        clay-09 built the UI half of that promise -- ``ClayView._composite``
        excludes ``role == "collider"`` from the opaque path and
        ``_view_overlay.OverlayOps._collider_draws`` draws it instead), so
        starting it hidden would cost an extra click just to see the thing
        that was just fit.

        **Not a locking door.** Unlike a mesh edit or a transform, adding a
        collider changes nothing about the *source* object itself -- its
        mesh, transform and every other field are untouched -- the same
        "attaching a new child changes nothing about what the parent looks
        like" reasoning :meth:`set_parent` and :meth:`group` are already
        exempted under (see :meth:`set_parent`'s own docstring). A locked
        source can still grow a collider child.
        """
        from . import ops as mesh_ops

        if collider.kind not in colliders.COLLIDER_KINDS:
            raise el.OpError(
                f"Unknown collider kind {collider.kind!r}. Choose one of "
                f"{', '.join(sorted(colliders.COLLIDER_KINDS))}."
            )
        source = self.by_uid(source_uid)  # KeyError names an unknown uid, the usual way
        label = colliders.COLLIDER_KINDS[collider.kind][0]
        taken = {o.name for o in self.objects}
        name = f"{source.name} {label}"
        if name in taken:
            name = mesh_ops.next_name(name, taken)

        new_obj = Obj(
            uid=new_uid(),
            name=name,
            mesh=collider.mesh,
            parent=source_uid,
            role="collider",
            collider_kind=collider.kind,
            visible=True,
        )
        self.add_object(new_obj)
        return new_obj

    # -- checkpoints (agent-facing, not serialized) ---------------------------

    def set_checkpoint(self, name: str) -> None:
        """Remember the current history position under *name*.

        Keyed on :attr:`history.head` -- the serial of the top done step, or
        ``0`` for a document with nothing done -- **never a stack position**:
        eviction pops from the front of the done list, and a redo replaces
        the same edits it undid, so a position (a plain integer count of done
        steps) drifts under both while a serial does not. Re-setting an
        existing name overwrites it; there is only ever one position per name.
        """
        self.checkpoints[str(name)] = self.history.head

    def _checkpoint_target(self, serial: int) -> int | None:
        """The done-count :meth:`step_history` would need to reach *serial*,
        or ``None`` if it is reachable from neither branch -- evicted out of
        the done list, or discarded from the redo list by a push that
        diverged past it. ``0`` (no edit ever has this serial: they start at
        1) always resolves to "everything undone", which is always reachable
        regardless of eviction.
        """
        if serial == 0:
            return 0
        return self.history.position_of(serial)

    def checkpoint_status(self, name: str) -> str:
        """``"current"`` | ``"reachable"`` | ``"gone"`` | ``"unknown"`` for
        *name* -- unknown for a name nothing was ever set under."""
        if name not in self.checkpoints:
            return "unknown"
        serial = self.checkpoints[name]
        if serial == self.history.head:
            return "current"
        return "gone" if self._checkpoint_target(serial) is None else "reachable"

    def restore_checkpoint(self, name: str) -> bool:
        """Move the history to *name*'s position. -> whether it moved.

        ``False`` for an unknown name, a checkpoint already current, or one
        that is gone -- the same three cases :meth:`checkpoint_status`
        reports, so a caller that only wants to know whether it worked never
        has to check status first.
        """
        if name not in self.checkpoints:
            return False
        serial = self.checkpoints[name]
        if serial == self.history.head:
            return False
        target = self._checkpoint_target(serial)
        if target is None:
            return False
        return self.step_history(target)

    # -- palette -----------------------------------------------------------

    def set_material(self, index: int, material: gltf.Material) -> bool:
        before = self.materials[index]
        if material is before:
            return False
        self.materials[index] = material
        self.history.push(MaterialEdit(index, before, material))
        self.touch()
        return True

    def add_material(self, material: gltf.Material | None = None) -> int:
        """Append a palette entry and return its index.

        Appended, never inserted: a slot is an index that every mesh's per-face
        ``material`` array names, so inserting in the middle would renumber
        those arrays in every object in the document. An append renumbers
        nothing.
        """
        index = len(self.materials)
        entry = material if material is not None else default_material(f"Material {index + 1}")
        self.materials.append(entry)
        self.history.push(MaterialListEdit(index, entry, added=True))
        self.touch()
        return index

    def material_users(self, index: int) -> int:
        """How many faces point at this slot, over everything an undo can reach.

        The document's own objects, and also every object a step on the stack is
        holding out of it -- an undone add, a done remove. Those are in no
        document, so a count over ``self.objects`` alone said zero for a slot
        that a single Ctrl+Z would put faces back onto, and the removal that
        answer permitted renumbered those faces onto whichever material had
        taken the slot's place: the silent reassignment :meth:`remove_material`
        exists to refuse, arriving later and by a different door.

        The cost is that a palette entry stays undeletable for as long as a
        deleted object that used it is still undoable. That is the safe
        direction of a bad trade -- the entry becomes deletable again once the
        step is evicted or the redo branch is dropped, whereas a face silently
        repainted is discovered three edits later with nothing to say what did
        it.

        It counts *faces*, deliberately, and not the ``Obj.material`` default:
        the properties panel's Remove button offers exactly the selected
        object's own default slot and re-points it immediately afterwards, so
        counting defaults would disable the only control that reaches this.
        """
        return sum(
            int((obj.mesh.material == index).sum()) for obj in _material_holders(self)
        )

    def remove_material(self, index: int) -> bool:
        """Drop an *unused* palette entry. -> whether it went.

        Refused while any face points at it -- including a face on an object
        only the undo stack is still holding, see :meth:`material_users` -- and
        refused for the last entry. Reassigning those faces to some other slot
        is the alternative, and it is a silent change to how part of the model
        looks, which is exactly the kind of thing a user discovers three edits
        later. Refusing lets the panel say which objects are in the way.
        """
        if not 0 <= index < len(self.materials) or len(self.materials) <= 1:
            return False
        if self.material_users(index):
            return False
        entry = self.materials[index]
        # Taken before the shift below runs -- the 2026-09-08 audit's
        # clay-06: once it has run, an object that named this slot exactly is
        # indistinguishable by number from one that already named the slot
        # below it, so the uids that named it have to be read now or the
        # information is gone. Spent by MaterialListEdit's own undo.
        boundary = _boundary_uids(self, index)
        del self.materials[index]
        _shift_materials(self, index, -1)
        self.history.push(MaterialListEdit(index, entry, added=False, boundary=boundary))
        self.touch()
        return True

    def add_material_and_assign(self, uid: int, material: gltf.Material | None = None) -> int:
        """Append a palette entry and point an object's default slot at it, as
        **one** step. -> the new entry's index.

        The 2026-09-08 audit's clay-02: the properties panel's Add button used
        to call :meth:`add_material` and then :meth:`set_props` as two
        separate pushes, so a single Ctrl+Z after the click left a stray,
        unreferenced palette entry behind instead of restoring the object's
        original slot -- "one press, one Ctrl+Z" applies here exactly as it
        does to :meth:`join_objects`' merge-and-removals.
        """
        mark = self.history.mark()
        index = self.add_material(material)
        self.set_props(uid, material=index)
        self.history.collapse_since(mark)
        return index

    def remove_material_and_reassign(self, uid: int, index: int) -> bool:
        """Drop a palette entry and repoint an object at what's left, as
        **one** step. -> whether it went.

        The counterpart of :meth:`add_material_and_assign` for the Remove
        button, same clay-02 finding: :meth:`remove_material` refuses rather
        than reassigning a used slot, so a refusal here folds to nothing
        rather than leaving a stray step -- :meth:`~.undo.UndoStack.
        collapse_since` is a no-op on an empty run.

        The two calls are exactly :meth:`remove_material` and :meth:`set_props`
        in the order the properties panel already made them, unfolded; this
        only wraps the pair. ``:func:`~.edits._shift_materials``'s own
        renumbering of ``obj.material`` runs first exactly as it did before,
        so this fold changes how many presses undo the pair, not what either
        call does -- that renumbering's own undo is what makes the pair whole
        again on the way back (the 2026-09-08 audit's clay-06: it used to be
        documented here as fixed by the very next ``set_props`` call, which
        was wrong -- that call's own "before" is read from *after* this
        renumbering already ran, so it could not recover the value the
        renumbering had lost; :meth:`remove_material` now hands the
        renumbering the uids it is about to make irreversible, so its own
        undo can put them back by name instead).
        """
        mark = self.history.mark()
        removed = self.remove_material(index)
        if removed:
            self.set_props(uid, material=min(index, len(self.materials) - 1))
        self.history.collapse_since(mark)
        return removed

    def set_shading(self, uid: int, faces: Any, smooth: bool) -> bool:
        """Set the per-face shading flag on some of one object's faces.

        Shading is not geometry -- the positions and the topology are untouched
        -- so this keeps the object's generator, exactly as an unwrap does. A
        smooth-shaded box is still a box.
        """
        import numpy as np

        obj = self.by_uid(uid)
        flags = np.array(obj.mesh.smooth, dtype=bool)
        if faces is None:
            flags[:] = smooth
        else:
            picked = np.asarray(faces, dtype="i8")
            if not len(picked):
                return False
            flags[picked] = smooth
        if np.array_equal(flags, obj.mesh.smooth):
            return False
        from dataclasses import replace as _replace

        self.set_mesh(uid, _replace(obj.mesh, smooth=flags), keep_generator=True)
        return True

    # -- selection (not undoable) ------------------------------------------

    def select(self, uids: Iterable[int]) -> None:
        """Replace the selection. Pushes no step, by design; see the module
        docstring. It still bumps ``rev``, because the viewport draws the
        selected object's outline and has to know to redraw it."""
        self.selection = {int(u) for u in uids}
        self.touch()

    # -- element mode (also not undoable) ----------------------------------

    def set_element_mode(self, mode: str) -> None:
        """Switch to object/vertex/edge/face mode, converting what is selected.

        Leaving an element mode keeps the objects selected -- the user was
        working on those objects and is still working on them. *Entering* one
        from object mode selects nothing, because there is nothing to convert
        and the invariant says the object selection in an element mode is the
        set of objects with something selected inside them.
        """
        if mode not in el.MODES:
            raise ValueError(f"unknown element mode {mode!r}")
        if mode == self.element_mode:
            return
        if mode == "object":
            self.element_sel = {}
        else:
            converted: dict[int, el.ElementSel] = {}
            for uid, sel in self.element_sel.items():
                out = el.convert(self.by_uid(uid).mesh, sel, mode)
                if not el.is_empty(out):
                    converted[uid] = out
            self.element_sel = converted
            self.selection = set(converted)
        self.element_mode = mode
        self.touch()

    def set_element_sel(self, uid: int, sel: el.ElementSel | None) -> None:
        """Replace what is selected inside one object, keeping the invariant.

        An empty selection removes the entry *and* the object from
        ``selection``: "selected with nothing selected inside it" is a state the
        invariant does not allow, and letting it exist is how a gizmo ends up
        drawn at the centroid of nothing.
        """
        if el.is_empty(sel):
            self.element_sel.pop(uid, None)
            if self.element_mode != "object":
                self.selection.discard(uid)
        else:
            assert sel is not None
            self.element_sel[uid] = sel
            if self.element_mode != "object":
                self.selection.add(uid)
        self.touch()

    def element_sel_of(self, uid: int) -> el.ElementSel:
        """What is selected inside *uid* -- an empty selection when nothing is."""
        return self.element_sel.get(uid) or el.empty()

    def clear_element_sel(self) -> None:
        self.element_sel = {}
        if self.element_mode != "object":
            self.selection = set()
        self.touch()


def _geometry_uids(edit: Edit | None) -> set[int]:
    """The uids whose *geometry* an edit changes, walking compounds.

    Deliberately narrow. A ``TransformEdit`` moves an object without changing a
    single vertex index and a rename changes nothing at all -- an element
    selection survives both, and clearing it there would make undo feel
    arbitrary. Only a mesh replacement, an add or a remove can leave a stored
    selection pointing at indices the mesh no longer has.
    """
    if edit is None:
        return set()
    if isinstance(edit, CompoundEdit):
        return {uid for child in edit.edits for uid in _geometry_uids(child)}
    if isinstance(edit, MeshEdit):
        return {edit.obj_uid}
    if isinstance(edit, ObjectAddEdit | ObjectRemoveEdit):
        return {edit.obj.uid}
    return set()


# --- the one conversion to glTF ----------------------------------------------


def _material_at(materials: Sequence[gltf.Material], index: int) -> gltf.Material:
    if 0 <= index < len(materials):
        return materials[index]
    return FALLBACK_MATERIAL


def _submesh(mesh: bm.Mesh, faces: np.ndarray) -> bm.Mesh:
    """The mesh restricted to ``faces``, with its vertex array left whole.

    Leaving the positions alone rather than compacting them is deliberate and
    free: ``render_arrays`` emits a vertex only for a corner that some face in
    *this* submesh uses, so an untouched position costs nothing downstream, and
    not re-indexing means not having a second place that could get the mapping
    wrong. The same does *not* go for ``uv``, which is indexed by corner rather
    than by vertex and so has to be gathered alongside ``loops``.

    The corner gather is arithmetic rather than a loop over faces because this
    runs once per material per object per rebuild, and an imported mesh has
    hundreds of thousands of faces: a Python-level pass over them here was the
    single worst hot spot in the rebuild.
    """
    counts = np.diff(mesh.starts).astype("i8")[faces]
    starts = np.concatenate([[0], np.cumsum(counts)]).astype("i4")
    total = int(starts[-1]) if len(starts) else 0
    if total:
        # For every output corner, the index of the corner it came from:
        # its face's start, plus how far into that face it sits.
        face_of_corner = np.repeat(np.arange(len(faces), dtype="i8"), counts)
        within = np.arange(total, dtype="i8") - starts[:-1].astype("i8")[face_of_corner]
        corners = mesh.starts[:-1].astype("i8")[faces][face_of_corner] + within
    else:
        corners = np.zeros(0, dtype="i8")
    return bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops[corners],
        starts=starts,
        material=mesh.material[faces],
        smooth=mesh.smooth[faces],
        uv=None if mesh.uv is None else mesh.uv[corners],
    )


def to_primitives(
    obj: Obj, materials: Sequence[gltf.Material], mesh: bm.Mesh | None = None
) -> list[gltf.Primitive]:
    """One :class:`~gltf.Primitive` per material the object's faces use.

    A draw call carries one material, so a two-toned box has to be two
    primitives however it is stored -- which is why the split happens here, on
    the way out, rather than in the mesh: the mesh stays one object the user
    can select faces across, and the renderer gets the grouping it needs.

    Groups come out in palette-index order, so the same document produces the
    same primitive order every time -- an exporter's output is diffable, and a
    GPU cache keyed on position does not shuffle.

    ``mesh`` defaults to ``obj.mesh`` -- the base -- for a caller with no
    document in hand to ask for the evaluated one; :func:`to_model` is the
    caller that has one, and passes ``doc.evaluated(obj.uid)`` explicitly.
    """
    if mesh is None:
        mesh = obj.mesh
    if bm.face_count(mesh) == 0:
        return []
    prims = []
    for index in np.unique(mesh.material):
        faces = np.flatnonzero(mesh.material == index)
        positions, normals, uvs, indices = bm.render_arrays(_submesh(mesh, faces))
        prims.append(
            gltf.Primitive(
                positions=positions,
                indices=indices,
                normals=normals,
                uvs=uvs,
                material=_material_at(materials, int(index)),
            )
        )
    return prims


_PLANS: weakref.WeakKeyDictionary[bm.Mesh, list[tuple[int, bm.RenderLayout]]] = (
    weakref.WeakKeyDictionary()
)
# The 2026-09-23 audit's clay-12: unsynchronized on the same false premise
# ``mesh.py``'s ``_RAW_CACHE`` carried before the 2026-09-12 audit's clay-04
# gave it a lock -- that every caller runs on the frame thread. A ``Mesh`` is
# shared, not copied, between the live document and a Familiar scratch
# preview (``kernels/mesh/scratch.py``'s ``clone``), and the scratch batch
# runs off the frame thread on ``realmspinner-task``, so a call to
# :func:`render_plan` for the same mesh from both sides can race this dict's
# get/set. Mirrors ``adjacency._CACHE_LOCK``: an uncontended acquire around a
# dict lookup, next to the numpy pass this function already does when it
# actually builds something.
_PLANS_LOCK = threading.Lock()


def render_plan(mesh: bm.Mesh) -> list[tuple[int, bm.RenderLayout]]:
    """``(material index, layout)`` per primitive, memoised against the mesh.

    The same weak-keyed shape :func:`~.adjacency.cached_triangulation` uses, and
    for the same reason one step further on: a ``Mesh`` is frozen and replaced
    whole, so everything :func:`to_primitives` computes that a *moved vertex*
    cannot change -- the material grouping, each submesh's corner gather, the
    shared/split vertex table and the index buffer -- is a pure function of one
    mesh object. An element drag holds the mesh it began on for its whole
    duration, so this is built on the first preview frame and reused by every
    frame after it.
    """
    with _PLANS_LOCK:
        got = _PLANS.get(mesh)
        if got is None:
            got = [
                (
                    int(index),
                    bm.render_layout(_submesh(mesh, np.flatnonzero(mesh.material == index))),
                )
                for index in np.unique(mesh.material)
            ]
            for _index, layout in got:
                # Read-only for ``cached_triangulation``'s reason: this is
                # handed to a different caller on every frame of a drag, and
                # the index buffer in particular goes straight out in a
                # ``Primitive``.
                for field in fields(layout):
                    value = getattr(layout, field.name)
                    if isinstance(value, np.ndarray):
                        value.setflags(write=False)
            _PLANS[mesh] = got
        return got


def preview_primitives(
    mesh: bm.Mesh,
    positions: np.ndarray,
    materials: Sequence[gltf.Material],
    *,
    moved: np.ndarray | None = None,
) -> list[gltf.Primitive]:
    """:func:`to_primitives` for *mesh* with its vertices moved to *positions*.

    The drag preview's whole reason for existing: ``positions`` is the only
    thing that changed, so only the arrays that depend on it are recomputed.
    ``positions`` must be the same length as ``mesh.positions`` -- a topology
    change is not a drag, and the layout would be answering about the wrong
    mesh.

    ``moved`` names the vertex indices this call changed, and is the *only*
    thing that makes the normals incremental: given it, a face no moved vertex
    touches reuses last frame's raw normal, which is bit-identical because a
    face's normal is the reduction over that face's own corners and nothing
    else. A caller that is not certain which vertices moved must pass nothing
    -- a ``moved`` missing a vertex leaves its faces holding stale normals with
    nothing anywhere to say so.

    Output is byte-identical to ``to_primitives`` on the moved mesh **except
    for the index buffer**, which is the layout's -- the triangulation the
    preview began with. That is what the GPU is drawing during the drag anyway:
    ``update_vertices`` leaves the IBO alone. See :class:`~.mesh.RenderLayout`.
    """
    if bm.face_count(mesh) == 0:
        return []
    prims = []
    for index, layout in render_plan(mesh):
        emitted, normals, uvs, indices = bm.render_from_layout(
            layout,
            positions,
            moved=moved,
            previous=None if moved is None else bm.raw_face_normals(layout),
        )
        prims.append(
            gltf.Primitive(
                positions=emitted,
                indices=indices,
                normals=normals,
                uvs=uvs,
                material=_material_at(materials, index),
            )
        )
    return prims


def kept_objects(doc: ClayDoc) -> list[Obj]:
    """Every object :func:`to_model` emits a node for, in ``doc.objects``
    order: visible, or with a visible descendant anywhere under it -- the
    module docstring's "hiding is per object, as in Blender" decision. See
    :func:`to_model`'s own docstring for what "kept" means and why a hidden
    parent with a visible child still gets a node.

    Split out of :func:`to_model` by the 2026-09-19 audit's clay-20:
    ``studio/modes/clay/mode.py``'s ``_rename_collider_nodes`` needs this exact
    filter -- it zips its own "which objects became nodes" list against
    ``to_model(doc).nodes`` to find each collider's node -- and used to
    re-derive it by hand because ``document.py`` was a file that tranche's
    own brief put out of reach. That constraint no longer holds (both files
    are owned together here), and "One conversion out, three consumers" is
    exactly the reason to have one function answer "which objects become
    nodes" rather than two copies that could silently drift apart and
    misalign that zip.
    """
    keep: dict[int, bool] = {obj.uid: obj.visible for obj in doc.objects}
    for obj in doc.objects:
        if obj.visible:
            for ancestor_uid in doc.ancestors(obj.uid):
                keep[ancestor_uid] = True
    return [obj for obj in doc.objects if keep.get(obj.uid, False)]


def to_model(doc: ClayDoc) -> gltf.Model:
    """The document as a :class:`~gltf.Model`: real glTF hierarchy.

    The GLB writer goes through here. **The viewport and the trellis render
    do not** -- see the module docstring's 2026-09-22 audit note, clay-21:
    they build their own per-object primitives from :func:`to_primitives`
    directly, because neither wants a flattened node tree. What the three
    consumers do share is :func:`to_primitives` itself and
    :func:`kept_objects`'s "visible, or has a visible descendant" rule,
    which the viewport cache and ``render_png``/``render_ids`` each
    re-derive rather than call -- pinned to agree by
    ``tests/modes/clay/test_document.py``'s
    ``test_the_viewports_per_object_cache_agrees_with_to_model_for_a_hidden_parent_with_a_visible_child``.

    **A node is emitted for an object that is visible, or that has a visible
    descendant anywhere under it** -- the module docstring's "hiding is per
    object, as in Blender" decision: a hidden parent still has to carry its
    visible children's frame, so its node survives with no mesh
    (``mesh=None``), while a hidden object with *no* visible descendant is
    omitted entirely, subtree and all, because nothing under it will ever be
    drawn either. ``children`` is wired from ``Obj.parent`` and ``roots`` is
    every kept object with no kept parent -- by construction that is every
    object whose own parent is ``None``, since a kept child's parent is
    always kept too (see :func:`kept_objects`'s own ancestor-marking pass);
    a dangling ``parent`` naming a uid this document does not have (never
    written by this package, but defensive against a hand-edited state)
    falls back to a root rather than raising.

    **Every visible object draws its evaluated mesh**, base run through its
    modifier stack (``doc.evaluated``), not the base alone -- this is the one
    conversion out of the document, so it is the one place a modifier stack
    has to take effect for the viewport, the exporter and the trellis render
    to agree about what the document looks like.
    """
    kept = kept_objects(doc)
    index_of_uid = {obj.uid: i for i, obj in enumerate(kept)}

    nodes: list[gltf.Node] = []
    meshes: list[list[gltf.Primitive]] = []
    for obj in kept:
        mesh_index: int | None = None
        if obj.visible:
            meshes.append(to_primitives(obj, doc.materials, doc.evaluated(obj.uid)))
            mesh_index = len(meshes) - 1
        nodes.append(
            gltf.Node(
                name=obj.name,
                translation=np.array(obj.translation, dtype="f8", copy=True),
                rotation=np.array(obj.rotation, dtype="f8", copy=True),
                scale=np.array(obj.scale, dtype="f8", copy=True),
                mesh=mesh_index,
            )
        )

    for obj in kept:
        if obj.parent is not None and obj.parent in index_of_uid:
            nodes[index_of_uid[obj.parent]].children.append(index_of_uid[obj.uid])
    roots = [
        index_of_uid[obj.uid]
        for obj in kept
        if obj.parent is None or obj.parent not in index_of_uid
    ]
    return gltf.Model(nodes, roots, meshes, [])
