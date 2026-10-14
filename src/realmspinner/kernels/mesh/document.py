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

**Materials are ``kernels.geom3d.gltf.Material`` objects, not a new type.** They are
already pure data, they already carry ``base_color_factor`` /
``double_sided`` / ``alpha_mode`` and the base-colour texture, and a parallel
Clay-only material type would have bought nothing but a conversion function
and a place for the two to drift. It also pays off on the GPU for free:
``GpuMaterial`` de-duplicates by ``id(material)``, and :func:`to_primitives`
hands every primitive the palette entry *itself*, so a document whose twelve
objects share one material uploads one material. Clay uses only the subset
:func:`reduce_material` keeps; every other slot is ``None`` or a fixed default,
and a slot that is ``None`` is a slot the renderer skips.

**Rotation is XYZW**, matching ``kernels.geom3d.math3d``, ``kernels.geom3d.gltf``, the pose
files on disk and glTF itself. There is no other quaternion order anywhere in
this project and this is not the place to introduce one.

**``generator`` is the live-until-frozen field.** An object placed from the
primitive registry keeps the generator's name and the parameters it was built
with, so the properties panel shows "Cylinder: radius, height, segments" and a
change regenerates the mesh as one :class:`~.edits.MeshEdit`. The first
topology edit clears it -- an extruded box is not describable as "box, size 1",
and a panel still offering a size field would discard the edit the moment it
was touched. :meth:`ClayDoc.set_mesh` (and :meth:`ClayDoc.join_objects` and
:meth:`ClayDoc.set_origin`) does that in one place for every op, so no op has to
remember to; the op layer used to, and the 2026-10-07 audit's clay-72 found this
paragraph still crediting it.

**A document's materials are Clay's reduced subset**: a name, a colour, a
base-colour texture, double-sided and cutout (:func:`reduce_material`). The
other glTF slots are fixed at defaults and stripped on the way in.

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

**The shape being edited stays selected in an element mode.** ``selection`` is
the set of objects the user is working on and ``element_sel`` is only the picks
inside them, so entering Points, Edges or Faces keeps the shape in the
inspector and the outliner and leaving returns to the same shape. Three rules,
all checked by :meth:`ClayDoc.selection_problems`:

* **object mode holds no element picks** (``element_sel == {}``);
* **in an element mode every pick is on a selected object** -- ``element_sel``
  keys are a subset of ``selection``, each value non-empty and in range for its
  mesh -- so ``frame_selection``, the properties pane and ``world_bounds`` still
  work with no per-mode branch;
* **``selection`` is written only by** :meth:`select` (which, in an element
  mode, drops the picks of what it deselects), :meth:`set_element_sel` (which
  adds the object and never removes it), :meth:`set_element_mode` (which only
  ever *adds*: picks that convert to something non-empty put their objects into
  it, and nothing is taken out -- the 2026-10-09 second audit's clay-105 found
  this paragraph still saying the mode switch never touches it), :meth:`separate`
  (which swaps the source for its pieces in every mode) and the sites that make
  an object stop existing or stop being visible (``remove_object``,
  ``join_objects``, ``set_visibility`` and an undo or redo that does either).
  :meth:`clear_element_sel` and the picks half of ``_forget_elements`` never
  touch it.

This replaces the Wings3D model in which the object selection was *derived*
from the picks, which meant entering an element mode deselected everything and
a shape with no picks yet was not "selected" at all.

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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np

from ...core.undo import CompoundEdit, Edit, UndoStack
from ..geom3d import gltf
from ..geom3d import math3d as m3
from . import anim, regen, rig
from . import elements as el
from . import mesh as bm
from . import uv as uv_projection
from .edits import (  # noqa: F401
    MaterialEdit,
    MaterialListEdit,
    MeshEdit,
    ObjectAddEdit,
    ObjectMoveEdit,
    ObjectPropsEdit,
    ObjectRemoveEdit,
    RigEdit,
    TransformEdit,
    _boundary_uids,
    _material_holders,
    _shift_materials,
)

#: What :meth:`ClayDoc._kept_placement` answers: the object's own ``(t, r, s)``,
#: its shear-baked mesh (or ``None``) and the ``(obj, (t, r, s), baked)`` of every
#: descendant that has to follow it.
_Placement = tuple[tuple[Any, Any, Any], Any, list[Any]]

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


#: The pixel sizes ``ClayDoc.add_texture`` offers. Small on purpose: this is a
#: picoCAD-style texel grid, and 128 is already a lot of pixels to paint by hand.
TEXTURE_SIZES: tuple[int, ...] = (32, 64, 128)


def _flat_rgba(factor: Sequence[float], size: int) -> bytes:
    """``size`` x ``size`` opaque RGBA bytes of one colour.

    *factor* is a linear glTF colour and a texture is read as sRGB, so the
    channels are sRGB-encoded here: a blank texture then reads, in Inker, as the
    colour the swatch shows.
    """
    rgb = np.clip(np.asarray(factor[:3], dtype="f8"), 0.0, 1.0)
    srgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)
    texel = bytes([*np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype("u1").tolist(), 255])
    return texel * (size * size)


def _same_geometry(a: bm.Mesh, b: bm.Mesh) -> bool:
    """Whether two meshes hold equal arrays in every field -- the comparison the
    identity test in ``set_mesh`` cannot make, for a door whose caller rebuilds
    the mesh from numbers (``set_generator_params``)."""
    for name in ("positions", "loops", "starts", "material", "smooth"):
        if not np.array_equal(getattr(a, name), getattr(b, name)):
            return False
    if (a.uv is None) != (b.uv is None) or (a.joints is None) != (b.joints is None):
        return False
    if a.joints is not None and not (
        np.array_equal(a.joints, b.joints) and np.array_equal(a.weights, b.weights)
    ):
        return False
    return a.uv is None or np.array_equal(a.uv, b.uv)


def _canonical(value: Any) -> Any:
    """*value* with every list, tuple and array read as one tuple type, at any
    depth -- for comparing generator params, never for storing them.

    The 2026-10-11 audit's clay-41: a document that went through its file holds
    ``size`` as a list and a widget hands back a tuple; Python says
    ``[1, 2] != (1, 2)``, so the same numbers read as an edit."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(v) for v in value)
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items()}
    return value


_TRIANGLES: weakref.WeakKeyDictionary[bm.Mesh, int] = weakref.WeakKeyDictionary()
_TRIANGLES_LOCK = threading.Lock()


def _triangles_of(mesh: bm.Mesh) -> int:
    """The triangles ``mesh.triangulate`` fans *mesh* into (a face of ``n``
    corners is ``n - 2``, a degenerate one 0) -- the same count
    ``serialize.triangle_count`` gives the writer and the reader, memoised on the
    mesh object because a ``Mesh`` is immutable and the ceiling check below sums
    it over every object in the document each time one is added."""
    with _TRIANGLES_LOCK:
        got = _TRIANGLES.get(mesh)
    if got is None:
        counts = np.diff(mesh.starts).astype("i8") - 2
        got = int(np.clip(counts, 0, None).sum())
        with _TRIANGLES_LOCK:
            _TRIANGLES[mesh] = got
    return got


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


def reduce_material(material: gltf.Material) -> gltf.Material:
    """*material* as Clay keeps it: name, colour, base-colour texture,
    double-sided and cutout. Everything else is fixed at the defaults.

    Idempotent, and it returns *material* itself when nothing would change, so
    the identity de-duplication ``GpuMaterial`` and the writers lean on survives
    a pass through here. A cutout is ``alpha_mode == "MASK"``; ``BLEND`` has no
    place in a picoCAD-level tool and reads as opaque. A cutout keeps its own
    ``alpha_cutoff`` (the 2026-10-07 audit's clay-56: it used to be reset to 0.5
    here, so a leaf cut at 0.2 came back from an import or a v3 migration
    cutting at 0.5 and nothing said so); an opaque material carries the spec
    default, because the number means nothing there.

    ``nearest`` (crisp texel sampling) is kept: it is Clay's own flag, set on
    every texture it makes or pulls from Inker, and stripping it would turn a
    painted pixel texture back into a blur on the next import or reload.
    """
    cutout = material.alpha_mode == "MASK"
    reduced = gltf.Material(
        name=material.name,
        base_color_factor=tuple(material.base_color_factor),
        metallic_factor=0.0,
        roughness_factor=0.6,
        double_sided=bool(material.double_sided),
        alpha_mode="MASK" if cutout else "OPAQUE",
        alpha_cutoff=float(material.alpha_cutoff) if cutout else 0.5,
        base_color=material.base_color,
        nearest=bool(material.nearest),
    )
    return material if reduced == material else reduced


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
    # Tranche 3: scene structure. A uid, never an index -- see ``edits``' own
    # rule for why every reference in this package is a uid. ``None`` is a
    # root, and TRS is local to whatever this names (a root's local TRS *is*
    # its world TRS, which is what keeps a document with no parenting behaving
    # exactly as it always did -- see the module docstring).
    parent: int | None = None
    # Marked as a bone (``rig``): the object's origin is a joint's pivot and its own
    # transform is that joint's rest pose. Explicit and never derived, so a group that
    # only organises a scene burns no joint slot.
    bone: bool = False
    # A weighted binding (``rig.Skin``) for a mesh that carries per-vertex
    # ``joints``/``weights``; ``None`` for everything else, including a shape that merely
    # sits under a bone, which is bound rigidly with nothing stored.
    skin: Any = None

    def __post_init__(self) -> None:
        # Own the transform arrays rather than aliasing whatever was passed in,
        # for the reason ``edits`` states: a caller that keeps mutating the
        # array it handed over would rewrite a recorded step behind the undo
        # stack's back, and a view would misreport its own size to eviction.
        self.translation = np.array(self.translation, dtype="f8", copy=True)
        self.rotation = np.array(self.rotation, dtype="f8", copy=True)
        self.scale = np.array(self.scale, dtype="f8", copy=True)

    def trs(self) -> tuple[Any, Any, Any]:
        return self.translation, self.rotation, self.scale


class _LiveUids:
    """``uid -> Obj`` read-only mapping over a document's validated position
    index (:meth:`ClayDoc._find`). It holds no snapshot, so it is always as
    current as ``doc.objects`` -- the property a map built once per call has only
    until the call mutates something."""

    __slots__ = ("_doc",)

    def __init__(self, doc: ClayDoc) -> None:
        self._doc = doc

    def get(self, uid: int, default: Obj | None = None) -> Obj | None:
        pos = self._doc._find(uid)
        return default if pos < 0 else self._doc.objects[pos]

    def __getitem__(self, uid: int) -> Obj:
        pos = self._doc._find(uid)
        if pos < 0:
            raise KeyError(uid)
        return self._doc.objects[pos]

    def __contains__(self, uid: object) -> bool:
        return isinstance(uid, int) and self._doc._find(uid) >= 0


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
        # The last parameterised element op that ran (``studio/modes/clay/
        # recent_op.RecentOp``), or None. Session state on the document for
        # ``element_mode``'s reason -- an op's ``enabled(doc)`` is the only
        # question the registry asks, so Repeat Last can only answer it from
        # here, and an agent's own tab then repeats only its own ops. Typed
        # ``Any`` because the record belongs to the layer above (a kernel may
        # not import it); never written to a file and never undoable.
        self.recent_op: Any = None
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
        # What opening an older file changed, as sentences for the mode to show
        # once (``legacy.migrate`` fills it; a document made here has none).
        # Session state, never written to a file.
        self.notices: tuple[str, ...] = ()
        # Named history positions, keyed on the *serial* :attr:`history.head`
        # gives back, never a stack position -- see :meth:`set_checkpoint`'s
        # own docstring for why. Not serialized: the undo history is not
        # saved either, so a checkpoint cannot outlive the session that made
        # it.
        self.checkpoints: dict[str, int] = {}
        # The rig's motion (``anim``): animations and the frame rate they are shown at,
        # restored by undo. Which animation is selected is not an edit and not in it.
        self.rig_state = anim.RigState()
        self.active_animation: int | None = None

    # -- lookup ------------------------------------------------------------

    def _refuse_past_ceilings(self, incoming: Sequence[Obj], outgoing: Sequence[Obj] = ()) -> None:
        """Refuse -- :class:`~.elements.OpError`, nothing changed -- a document
        that would pass the object or triangle ceiling the writer enforces.

        The 2026-10-07 audit's clay-13: Mirror Copy, Duplicate, Separate, Group
        and every agent creator made objects without looking at
        ``glbimport.MAX_OBJECTS`` (4,096) or ``MAX_TRIANGLES`` (2,000,000), the
        two numbers ``serialize.rblk_bytes`` and ``read_rblk`` both hold a
        document to. Thirteen Mirror Copy presses reached 8,192 objects, and the
        user learned at Ctrl+S (or the autosave) that the document they had
        built could not be saved. The ceilings are read from ``glbimport`` at
        call time -- the one place the writer and the reader already take them
        from -- by a function-scope import, because ``glbimport`` imports this
        module. *outgoing* is what the same step removes (Separate's source).
        """
        from .glbimport import MAX_OBJECTS, MAX_TRIANGLES

        marked = (
            sum(1 for o in self.objects if o.bone)
            - sum(1 for o in outgoing if o.bone)
            + sum(1 for o in incoming if o.bone)
        )
        if marked > rig.MAX_BONES:
            raise el.OpError(
                f"That would make {marked} bones, past the {rig.MAX_BONES} a Clay rig can "
                "hold. Unmark or delete some bones first."
            )
        objects = len(self.objects) - len(outgoing) + len(incoming)
        if objects > MAX_OBJECTS:
            raise el.OpError(
                f"That would make {objects:,} objects, past the {MAX_OBJECTS:,} a Clay "
                "document can hold and save. Delete or join some objects first."
            )
        added = sum(_triangles_of(o.mesh) for o in incoming)
        if added:
            triangles = (
                self._triangle_total() - sum(_triangles_of(o.mesh) for o in outgoing) + added
            )
            if triangles > MAX_TRIANGLES:
                raise el.OpError(
                    f"That would make {triangles:,} triangles, past the {MAX_TRIANGLES:,} "
                    "a Clay document can hold and save."
                )

    def _triangle_key(self) -> tuple[int, int, int]:
        """What the cached triangle total is valid for: the change counter, the
        history position and the object count. ``rev`` alone would do for every
        door in this class; the other two are the belt for a caller that drives
        ``history.undo`` directly (it moves ``head`` and replaces meshes without
        a ``touch``)."""
        return (self.rev, self.history.head, len(self.objects))

    def _triangle_total(self) -> int:
        """The triangles over every object in the document, summed once per
        change rather than once per ceiling check.

        The 2026-10-09 second audit's clay-104: every growing ``set_mesh`` and
        ``set_generator_params`` summed ``_triangles_of`` over the whole document
        (1.24 ms at 4,000 objects, so a fifteen-second op over all of them) for a
        number only the previous step could have changed. The sum is memoised
        against :meth:`_triangle_key`, and the two doors that replace a mesh
        carry it forward by their own delta (:meth:`_carry_triangle_total`)
        instead of dropping it, which is what keeps a loop of ``set_mesh`` calls
        from re-summing on every iteration. Any other writer bumps ``rev`` or
        moves ``head``, so the next call re-sums: stale in the safe direction.
        """
        key = self._triangle_key()
        cached = self.__dict__.get("_tri_cache")
        if cached is not None and cached[0] == key:
            return cached[1]
        total = sum(_triangles_of(o.mesh) for o in self.objects)
        self._tri_cache = (key, total)
        return total

    def _carry_triangle_total(self, was: tuple[int, int, int], delta: int) -> None:
        """Re-key the cached total after a door changed the document by *delta*
        triangles -- only when it was valid for the state *was* the door started
        from, so a cache that was already stale stays stale."""
        cached = self.__dict__.get("_tri_cache")
        if cached is not None and cached[0] == was:
            self._tri_cache = (self._triangle_key(), cached[1] + delta)

    @staticmethod
    def _refuse_invalid(name: str, mesh: bm.Mesh) -> None:
        """OpError when *mesh* is not a valid CSR (:func:`~.mesh.validate`).

        The 2026-10-09 second audit's clay-82: nothing on the adopt path looked,
        so a fill-hole that produced a one-corner face was adopted, saved, and
        then refused by ``read_rblk`` on reopen -- crash recovery included. The
        writer now refuses the same mesh (``serialize._refuse_unreadable``);
        this is the door that says so at the edit, naming the object, while
        Ctrl+Z is still one press away. ``validate`` is a handful of vectorised
        reductions (about 0.1 ms on a 200,000-triangle mesh, measured), so it
        costs nothing per drag frame.
        """
        try:
            bm.validate(mesh)
        except ValueError as error:
            raise el.OpError(
                f"{name!r} cannot take that change: it makes a mesh that is not valid ({error})."
            ) from error

    @staticmethod
    def _refuse_unstorable(name: str, mesh: bm.Mesh) -> None:
        """OpError when *mesh* has a position or UV past float32 (``inf``/NaN),
        which the writer refuses -- the one wording ``set_mesh`` and every other
        door that adopts a whole mesh share."""
        if not np.isfinite(mesh.positions).all() or (
            mesh.uv is not None and not np.isfinite(mesh.uv).all()
        ):
            raise el.OpError(
                f"{name!r} cannot take that change: it makes a position or a UV "
                "too large to store (past about 3e38)."
            )

    def _refuse_past_triangle_ceiling(self, obj: Obj, mesh: bm.Mesh) -> None:
        """OpError when swapping *obj*'s mesh for *mesh* takes the document past
        ``MAX_TRIANGLES`` -- but only when it adds triangles, so shrinking or
        holding a document that is somehow already over the line stays possible.

        The 2026-10-09 audit's clay-02: ``set_generator_params`` and ``set_mesh``
        swapped meshes with no look at the ceiling the add doors enforce, so
        raising segments on a few torus objects built a document Ctrl+S refused.
        """
        if _triangles_of(mesh) <= _triangles_of(obj.mesh):
            return
        probe = Obj(uid=obj.uid, name=obj.name, mesh=mesh)
        self._refuse_past_ceilings([probe], [obj])

    def _index_positions(self) -> dict[int, int]:
        """Rebuild the uid -> position index in one pass (first-seen wins, as a
        front-to-back scan always answered) and return it."""
        cache: dict[int, int] = self.__dict__.setdefault("_pos_cache", {})
        cache.clear()
        for i in range(len(self.objects) - 1, -1, -1):
            cache[self.objects[i].uid] = i
        return cache

    def _find(self, uid: int) -> int:
        """*uid*'s position in ``objects``, or ``-1``, without a scan per call.

        The 2026-10-10 audit's clay-23: ``index_of`` was a bare scan and
        ``world_matrix``/``ancestors``/``set_origin`` each built a whole uid map
        per call, so Group, Parent to Last and Origin to World over 4,000 objects
        did 4,000 full passes (5.8 s, 3.0 s, 2.6 s on the frame thread).

        The index is remembered and **checked on every use**: a remembered
        position is trusted only while ``objects[pos].uid`` is still *uid*, and
        the first miss rebuilds the lot, so an object added, removed, reordered
        or restored by an undo can never be read stale -- no mutation path has to
        know the index exists, which is what the INVARIANTS rule on every way
        Clay removes an object demands of a cache. (A uid that appears twice is
        not a legal document; if one did, the first-seen answer holds only until
        a rebuild.)"""
        cache: dict[int, int] = self.__dict__.get("_pos_cache") or {}
        pos = cache.get(uid, -1)
        objects = self.objects
        if 0 <= pos < len(objects) and objects[pos].uid == uid:
            return pos
        return self._index_positions().get(uid, -1)

    def index_of(self, uid: int) -> int:
        pos = self._find(uid)
        if pos < 0:
            raise KeyError(f"no object with uid {uid}")
        return pos

    def by_uid(self, uid: int) -> Obj:
        return self.objects[self.index_of(uid)]

    def touch(self) -> None:
        self.rev += 1

    def _is_hidden(self, uid: int) -> bool:
        """True when *uid* names a hidden object, without a linear scan per call.

        A marquee release calls :meth:`set_element_sel` once per object in a
        document of up to 4,096, and ``by_uid`` there made it quadratic (the
        2026-10-07 audit's clay-71 counts ``index_of`` calls on that path). The
        uid -> position map is rebuilt only when a remembered position stops
        holding that uid, so objects added, removed or reordered are never
        read stale; visibility is read live off the object.
        """
        return not self.by_uid(uid).visible

    def mesh_stamp(self, uid: int) -> int:
        """A wire-safe revision number for one object's current mesh.

        ``Mesh`` is immutable and ``eq=False`` (see its own docstring), so
        object identity already *is* the mesh's revision -- ``mesh.py``'s
        ``_RAW_CACHE`` already keys off exactly that, for a drag's per-frame
        normals cache (the 2026-10-07 audit's clay-47: this named a module path
        and line range that no longer existed). What identity is not is a value safe to
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

    def _uid_map(self) -> dict[int, Obj]:
        """uid -> object, built in one pass. First-seen wins on a duplicate uid,
        which is what :meth:`index_of`'s front-to-back scan has always answered.

        The 2026-10-07 audit's clay-71: :meth:`ancestors`, :meth:`descendants`
        and :meth:`world_matrix` reached every hop through ``index_of``/
        ``by_uid`` (a linear scan) or ``children_of`` (another), so one call on a
        4,000-deep chain -- legal, well inside ``glbimport.MAX_OBJECTS`` -- cost
        about half a second, and the viewport and Properties make that call per
        selected object per frame. Same shape the 2026-10-03 audit's clay-85
        gave :func:`kept_objects`: one map per call.
        """
        by_uid: dict[int, Obj] = {}
        for obj in self.objects:
            by_uid.setdefault(obj.uid, obj)
        return by_uid

    def _live(self) -> _LiveUids:
        """A ``uid -> Obj`` view that answers off :meth:`_find` on every lookup.

        What :meth:`ancestors`, :meth:`world_matrix` and the placement doors use
        in place of :meth:`_uid_map` when the caller holds no snapshot: it costs
        O(depth) per walk instead of O(objects), and cannot go stale across the
        mutations a door makes between two lookups (the 2026-10-10 audit's
        clay-23)."""
        return _LiveUids(self)

    @staticmethod
    def _ancestors_in(uid: int, by_uid: Mapping[int, Obj]) -> list[int]:
        if uid not in by_uid:
            raise KeyError(f"no object with uid {uid}")
        out: list[int] = []
        seen = {uid}
        current = by_uid[uid].parent
        while current is not None and current not in seen:
            obj = by_uid.get(current)
            if obj is None:
                break
            out.append(current)
            seen.add(current)
            current = obj.parent
        return out

    def ancestors(self, uid: int) -> list[int]:
        """*uid*'s parent, grandparent, and so on -- nearest first.

        Stops at the first uid it has already seen rather than trusting the
        chain is acyclic: :meth:`set_parent` refuses a cycle going forward
        and :mod:`.serialize` refuses one in a loaded file, so this should
        never actually run into one, but a caller walking a hand-built
        in-memory document (a test, a script) gets a bounded answer rather
        than an infinite loop if it does.
        """
        return self._ancestors_in(uid, self._live())

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

        One parent -> children map for the whole walk (the 2026-10-07 audit's
        clay-71), not a :meth:`children_of` scan per node, and a set for the
        visited test rather than a list.
        """
        kids: dict[int | None, list[int]] = {}
        for obj in self.objects:
            kids.setdefault(obj.parent, []).append(obj.uid)
        out: list[int] = []
        visited: set[int] = set()
        stack = list(reversed(kids.get(uid, ())))
        while stack:
            u = stack.pop()
            if u in visited:  # cycle guard; see ancestors()'s own
                continue
            visited.add(u)
            out.append(u)
            stack.extend(reversed(kids.get(u, ())))
        return out

    def roots(self) -> list[int]:
        """Every parentless object, in document order."""
        return [obj.uid for obj in self.objects if obj.parent is None]

    def world_matrix(self, uid: int) -> np.ndarray:
        """*uid*'s world transform: its own local TRS, composed through every
        ancestor's -- ancestor-first, so a root's own local TRS *is* its
        world matrix (``ancestors`` gives nothing to compose through)."""
        return self._world_in(uid, self._live())

    def _world_in(self, uid: int, by_uid: Mapping[int, Obj]) -> np.ndarray:
        """:meth:`world_matrix` over a map the caller built once -- for a loop
        that asks for many objects' worlds before changing anything."""
        chain = [uid, *self._ancestors_in(uid, by_uid)]
        matrix = m3.identity()
        for u in reversed(chain):
            obj = by_uid[u]
            matrix = matrix @ m3.compose(obj.translation, obj.rotation, obj.scale)
        return matrix

    def _require_invertible(
        self, uid: int, by_uid: Mapping[int, Obj] | None = None
    ) -> np.ndarray:
        """The inverse of *uid*'s world matrix, or an OpError naming it.

        *by_uid* is a map the caller built once, for a loop that would
        otherwise rebuild it per object (the 2026-10-10 audit's clay-32)."""
        by_uid = self._live() if by_uid is None else by_uid
        try:
            return np.linalg.inv(self._world_in(uid, by_uid))
        except np.linalg.LinAlgError as error:
            raise el.OpError(
                f"{by_uid[uid].name!r} has a zero scale, so nothing can be "
                "placed relative to it."
            ) from error

    def _relative_matrix(
        self, world: np.ndarray, parent: int | None, by_uid: Mapping[int, Obj] | None = None
    ) -> np.ndarray:
        """*world* as a matrix relative to *parent* (or as-is for a root)."""
        if parent is None:
            return np.asarray(world, dtype="f8")
        return self._require_invertible(parent, by_uid) @ np.asarray(world, dtype="f8")

    def _local_relative(self, world: np.ndarray, parent: int | None) -> tuple[Any, Any, Any]:
        """*world* re-expressed as local TRS relative to *parent* (or as-is
        for a root) -- ``(t, r, s)``, via :func:`~..geom3d.math3d.decompose`.

        Shared by :meth:`set_parent` (the new parent), :meth:`local_from_world`
        (the object's own current parent) and :meth:`remove_object` (the
        removed object's own parent) -- three callers wanting the same "what
        would this world matrix be, named relative to that uid" question.
        """
        return m3.decompose(self._relative_matrix(world, parent))

    def _kept_place(self, name: str, world: np.ndarray, parent: int | None) -> tuple[Any, Any, Any]:
        """:meth:`_local_relative`, refusing a result that is not finite.

        The one door every "re-express this world placement under that parent"
        caller goes through -- :meth:`set_parent`, :meth:`remove_object`,
        :meth:`join_objects`, :meth:`separate` and :meth:`set_origin`. The
        2026-10-03 audit's clay-24 put the refusal inside ``set_parent`` alone;
        the 2026-10-07 audit's clay-04 found the other four doing the same
        arithmetic without it: a parent whose scale is a denormal (``1e-320``,
        finite and nonzero, so past ``np.linalg.inv``'s singular test) inverts
        to ``inf`` and ``decompose`` hands back ``NaN`` that was then written
        onto the child -- a document that saves and can never be reopened.
        Callers compute every child's result through this *before* their first
        assignment, so a refusal leaves nothing half-changed.
        """
        return self._finite_place(name, self._local_relative(world, parent))

    def _kept_placement(
        self,
        child: Obj,
        world: np.ndarray,
        parent: int | None,
        by_uid: Mapping[int, Obj] | None = None,
    ) -> _Placement:
        """:meth:`_kept_place`, but exact when the matrix is sheared:
        ``((t, r, s), baked_mesh_or_None, carried)``.

        The 2026-10-10 audit's clay-04: a (2, 1, 1) parent over a child turned
        45 degrees has a sheared world matrix, ``decompose`` is "for a matrix
        with no shear", and every door that re-expressed such a child stored a
        quaternion of norm 0.985 -- the child moved on screen by 0.45 when its
        parent was deleted, and moved again by 0.03 when the file was reopened
        and the quaternion normalised. No translation/rotation/scale states a
        sheared matrix, so the residual shear is folded into the child's *mesh*
        (as :func:`~.ops.bake_transform` and the GLB importer do): the stored
        rotation stays a unit one and the geometry lands where it stood. Only a
        genuinely sheared matrix takes this path -- a clean orthogonal basis
        returns ``None`` for the mesh and keeps its generator.

        **The object's own children are carried** (the 2026-10-11 audit's
        clay-12 refused the move; the F20 follow-up built it). The shear is
        folded into the object's mesh, so its frame is now the unsheared one
        and a child's old local matrix ``L`` would follow that frame rather
        than the geometry. Its world is ``frame @ shear @ L`` before and
        ``frame @ L'`` after, so ``L'`` is ``shear @ L`` -- which is itself
        sheared whenever ``L`` is rotated, and the walk
        (:meth:`_carry_subtree`) then bakes *that* shear into the child and
        goes one generation further. A sheared *empty* has no mesh to bake into
        and simply drops its shear into its children the same way.

        Everything that can refuse (a non-finite placement, a baked mesh the
        writer would refuse) is raised here, subtree included, so a caller that
        computes every child's placement before its first assignment stays
        atomic.
        """
        target = self._relative_matrix(world, parent, by_uid)
        t, r, s, shear = m3.decompose_with_shear(target)
        trs = self._finite_place(child.name, (t, r, s))
        if shear is None:
            return trs, None, []
        baked = self._bake_shear(child, shear)
        return trs, baked, self._carry_subtree(child, shear)

    def _bake_shear(self, obj: Obj, shear: np.ndarray) -> bm.Mesh | None:
        """*obj*'s mesh with *shear* folded in, or ``None`` for an empty (it has
        no geometry to carry the shear)."""
        if len(obj.mesh.positions) == 0:
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            baked = bm.transformed(obj.mesh, shear)
        self._refuse_unstorable(obj.name, baked)
        return baked

    def _carry_subtree(
        self, root: Obj, shear: np.ndarray
    ) -> list[tuple[Obj, tuple[Any, Any, Any], bm.Mesh | None]]:
        """Every descendant of *root* that must change so *root*'s sheared frame
        can be replaced by an unsheared one -> ``(obj, (t, r, s), baked)`` each,
        nothing assigned. See :meth:`_kept_placement` for the arithmetic.

        An explicit stack and a visited set, like :meth:`descendants`, so a
        legal chain thousands deep cannot recurse and a hand-built cycle
        terminates. A descendant whose re-expressed matrix is *not* sheared
        keeps its frame, so its own subtree is untouched and the walk stops
        there."""
        kids: dict[int | None, list[Obj]] = {}
        for o in self.objects:
            kids.setdefault(o.parent, []).append(o)
        carried: list[tuple[Obj, tuple[Any, Any, Any], bm.Mesh | None]] = []
        seen = {root.uid}
        stack: list[tuple[Obj, np.ndarray]] = [(root, shear)]
        while stack:
            owner, owner_shear = stack.pop()
            for c in kids.get(owner.uid, ()):
                if c.uid in seen:
                    continue
                seen.add(c.uid)
                t, r, s, c_shear = m3.decompose_with_shear(
                    owner_shear @ m3.compose(*c.trs())
                )
                trs = self._finite_place(c.name, (t, r, s))
                baked = None
                if c_shear is not None:
                    baked = self._bake_shear(c, c_shear)
                    stack.append((c, c_shear))
                carried.append((c, trs, baked))
        return carried

    def _placement_edits(self, child: Obj, placement: _Placement) -> list[Any]:
        """Apply a :meth:`_kept_placement` to *child* and the descendants it
        carries -> the steps to record (each one's transform, and for a sheared
        one its baked mesh and the generator freeze that goes with it --
        geometry that is no longer what a generator would build,
        :meth:`set_mesh`'s rule)."""
        trs, baked, carried = placement
        edits = self._apply_placed(child, trs, baked)
        for obj, o_trs, o_baked in carried:
            edits.extend(self._apply_placed(obj, o_trs, o_baked))
        return edits

    def _apply_placed(
        self, obj: Obj, trs: tuple[Any, Any, Any], baked: bm.Mesh | None
    ) -> list[Any]:
        edits: list[Any] = []
        before_trs = tuple(np.array(v, copy=True) for v in obj.trs())
        if not all(np.array_equal(a, b) for a, b in zip(before_trs, trs, strict=True)):
            obj.translation, obj.rotation, obj.scale = trs
            edits.append(TransformEdit(obj.uid, before_trs, trs))
        if baked is not None:
            before_mesh, obj.mesh = obj.mesh, baked
            edits.append(MeshEdit(obj.uid, before_mesh, baked))
            if obj.generator is not None:
                was = {"generator": obj.generator, "params": obj.params}
                obj.generator, obj.params = None, {}
                edits.append(ObjectPropsEdit(obj.uid, was, {"generator": None, "params": {}}))
        return edits

    @staticmethod
    def _finite_place(name: str, trs: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
        """*trs* itself, or an OpError naming *name* when any part is not finite."""
        if not all(np.isfinite(np.asarray(v, dtype="f8")).all() for v in trs):
            raise el.OpError(
                f"{name!r} cannot keep its place under that parent: the "
                "transform between them is not finite (an extreme scale)."
            )
        return trs

    def local_from_world(self, uid: int, matrix: np.ndarray) -> tuple[Any, Any, Any]:
        """*matrix*, a world transform, as ``(t, r, s)`` local to *uid*'s own
        current parent -- what a gizmo writes back after dragging in world
        space on a parented object."""
        obj = self.by_uid(uid)
        return self._local_relative(matrix, obj.parent)

    def set_parent(self, uid: int, parent: int | None, *, keep_world: bool = True) -> bool:
        """Reparent *uid* onto *parent* (or make it a root), as one step.

        Refuses -- :class:`~.elements.OpError`, nothing pushed -- parenting
        *uid* to itself or to one of its own descendants: a document has no
        way to compose a cycle's world matrix.

        With ``keep_world`` (the default) the object's local TRS is
        recomputed from its *current* world matrix before the parent changes,
        so nothing in the viewport moves; the transform change and the parent
        change ride in the same :class:`~..core.undo.CompoundEdit` -- one
        Ctrl+Z undoes both, or the parent would change on one press and the
        object would visibly jump on the next.

        **Where the re-expressed placement is sheared** (a non-uniformly scaled
        parent over a rotated child) the residual shear is baked into the
        object's mesh and its generator freezes, and the object's own
        descendants are carried the same way so none is left behind by the
        shear -- see :meth:`_kept_placement`.
        """
        obj = self.by_uid(uid)
        if parent is not None:
            self.by_uid(parent)  # KeyError names an unknown uid, the usual way
            # The 2026-10-10 audit's clay-23: ``parent in descendants(uid)`` built
            # a whole parent -> children map per call (Parent to Last did it once
            # per selected object). "parent sits under uid" is exactly "uid is one
            # of parent's ancestors", an O(depth) walk up that needs no map.
            if parent == uid or uid in self.ancestors(parent):
                raise el.OpError(
                    f"{obj.name!r} cannot be parented to itself or to one of its own "
                    "descendants."
                )
        if obj.parent == parent:
            return False

        old_parent = obj.parent
        edits: list[Any] = []
        if keep_world:
            world = self.world_matrix(uid)
            # The 2026-10-03 audit's clay-24: a parent whose scale is a
            # denormal (``1e-320``, finite and nonzero, so past both
            # ``set_transform``'s checks and ``np.linalg.inv``'s singular
            # test) inverts to ``inf``, and ``decompose`` then handed back
            # ``NaN`` translation, rotation and scale that were written onto
            # the child without a look -- the poisoned transform
            # ``set_transform``'s own finiteness assertion exists to keep out.
            # The refusal now lives in ``_kept_place``, shared with the other
            # four doors that do this arithmetic (2026-10-07 clay-04).
            placement = self._kept_placement(obj, world, parent)
            edits.extend(self._placement_edits(obj, placement))
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
        # The 2026-10-09 audit's clay-18: only the children and the mesh were
        # checked, so a vertexless object under a denormal parent took a NaN
        # translation of its own.
        self._finite_place(obj.name, (new_translation, obj.rotation, obj.scale))
        if np.array_equal(new_translation, obj.translation):
            return False

        world_old = self.world_matrix(uid)
        world_new = parent_world @ m3.compose(new_translation, obj.rotation, obj.scale)
        try:
            shift = np.linalg.inv(world_new) @ world_old
        except np.linalg.LinAlgError as error:
            raise el.OpError(f"{obj.name} has a zero scale, so its origin cannot move.") from error

        children = self.children_of(uid)
        by_uid = self._live()  # not a per-call ``_uid_map``: the 2026-10-10 audit's clay-23
        child_worlds_old = {c: self._world_in(c, by_uid) for c in children}
        try:
            world_new_inv = np.linalg.inv(world_new)
        except np.linalg.LinAlgError as error:
            raise el.OpError(
                f"{obj.name} has a zero scale, so its children cannot be corrected."
            ) from error

        # The 2026-10-07 audit's clay-04: each child's new local TRS is worked
        # out, and checked finite, before the mesh or the translation moves --
        # a denormal-scaled origin made ``decompose`` return NaN that was
        # written onto the child after the object itself had already changed.
        child_trs = {
            c: self._finite_place(
                self.by_uid(c).name, m3.decompose(world_new_inv @ child_worlds_old[c])
            )
            for c in children
        }

        # The 2026-10-07 audit's clay-68: only the children's placements were
        # checked, and an object whose scale is tiny (``set_transform`` accepts
        # 1e-40) has a shift of ~1e40 that the float32 mesh cast to inf. Checked
        # before the first assignment, like the children above.
        with np.errstate(over="ignore", invalid="ignore"):
            shifted = bm.transformed(obj.mesh, shift)
        if not np.isfinite(shifted.positions).all():
            raise el.OpError(
                f"{obj.name} cannot move its origin there: re-basing its mesh is not "
                "finite (an extreme scale)."
            )
        edits: list[Any] = []
        before_mesh, obj.mesh = obj.mesh, shifted
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
            t, r, s = child_trs[c]
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

        Only the selection's *topmost* members are parented onto the empty (the
        2026-10-07 audit's clay-09): a member whose own ancestor is also in
        *uids* is already carried by that ancestor, and lifting it out from
        under its parent turned Select All -> Group into a flat list. Every
        parented member is checked for a finite re-expressed placement before
        anything is added (clay-10), so a refused member leaves no empty and no
        half-applied group behind. That includes a member whose re-expressed
        placement is sheared while it has children of its own, whose subtree is
        carried (see :meth:`_kept_placement`).
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
            box = mesh_ops.world_box(member, mesh=member.mesh, world=self.world_matrix(u))
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

        chosen = set(members)
        topmost: list[int] = []
        for u in dict.fromkeys(members):
            ancestor = self.by_uid(u).parent
            while ancestor is not None and ancestor not in chosen:
                ancestor = self.by_uid(ancestor).parent
            if ancestor is None:
                topmost.append(u)
        # The empty has identity rotation and scale, so its inverse world is a
        # plain translation: the same arithmetic ``set_parent`` will run, asked
        # of every member now, before the first assignment.
        empty_inverse = np.linalg.inv(m3.compose(*empty_obj.trs()))
        # Through ``_kept_placement`` (a root-relative matrix, the empty's
        # inverse already folded in) so a member whose world matrix is sheared
        # refuses here too when its baked mesh would be unstorable.
        for u in topmost:
            self._kept_placement(self.by_uid(u), empty_inverse @ self.world_matrix(u), None)

        mark = self.history.mark()
        try:
            self.add_object(empty_obj)
            for u in topmost:
                self.set_parent(u, empty_obj.uid, keep_world=True)
        finally:
            # The 2026-10-07 audit's clay-10: this gesture was opened with no
            # try/finally, so a refused member left ``_open_gestures`` at 1 for
            # the rest of the session -- undo eviction switched off for good.
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

    def undo_unrecorded(self) -> bool:
        """Reverse the newest step and forget it: no redo, element selection
        dropped exactly as :meth:`undo` drops it. -> whether there was a step.

        The door for a caller that wants a step to have *not happened* -- a
        rolled-back batch or program, a dry run. They used to call
        ``history.undo(doc, redoable=False)`` directly, which reverses the step
        and never runs :meth:`_forget_elements`: the 2026-10-07 audit's clay-05
        found a face-mode subdivide rolled back that way leaving 24 face indices
        selected on the 6-face mesh it had restored, and ``delete_faces`` then
        raised ``IndexError`` on them. Routing it here keeps the one rule (a
        selection describing geometry that is gone is dropped) in one place.
        """
        edit = self.history.top
        if not self.history.undo(self, redoable=False):
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
        # Picks only: the object stays selected, because it is the shape being
        # edited and a mesh edit does not end that.
        for uid in _geometry_uids(edit):
            self.element_sel.pop(uid, None)
        # An undo of an add (or a redo of a remove) takes the object out of the
        # document and an undo of a *show* hides it again (found by the
        # selection fuzz test): neither a deleted nor a hidden object holds a
        # selection or picks, whichever way it came to be so.
        shown = {obj.uid for obj in self.objects if obj.visible}
        self.selection &= shown
        for uid in [u for u in self.element_sel if u not in shown]:
            del self.element_sel[uid]
        self.touch()

    # -- objects -----------------------------------------------------------

    def add_object(self, obj: Obj, index: int | None = None) -> Obj:
        """Insert an object and record the step. Returns the object it was given
        so a caller can place and keep hold of one in a single expression.

        Refuses (:class:`~.elements.OpError`, nothing pushed) an object that
        would take the document past what it can save; see
        :meth:`_refuse_past_ceilings`."""
        self._refuse_past_ceilings([obj])
        at = len(self.objects) if index is None else index
        self.objects.insert(at, obj)
        self.history.push(ObjectAddEdit(at, obj))
        self.touch()
        return obj

    def add_objects(self, objs: Iterable[Obj], label: str = "") -> list[Obj]:
        """Insert several objects as **one** step, and select all of them.

        Several parts added at once (Duplicate on a multi-selection, an
        agent's batch of primitives) are N ``add_object`` calls, and N
        ``ObjectAddEdit`` pushes -- so undoing three copies you did not want is
        three presses of Ctrl+Z, through two intermediate states that were
        never a state anyone made. One gesture is one step, for the
        reason ``set_visibility`` and :meth:`join_objects` are: an undo history
        whose entries are not the actions the user took is not a history.

        Built the way ``join_objects`` builds its own compound -- edits
        collected in the order they were applied, ``CompoundEdit`` undoing them
        in reverse, which pops the objects off the end of the list first and so
        keeps every recorded index correct on the way back out.

        Empty is a no-op that pushes nothing at all, for ``set_visibility``'s
        reason: a step that changes nothing makes a saved document ask to be
        saved again.

        Refuses (:class:`~.elements.OpError`, nothing pushed) a batch that would
        take the document past what it can save; see
        :meth:`_refuse_past_ceilings`.
        """
        added = list(objs)
        if not added:
            return []
        self._refuse_past_ceilings(added)
        edits: list[Any] = []
        for obj in added:
            at = len(self.objects)
            self.objects.insert(at, obj)
            edits.append(ObjectAddEdit(at, obj))
        edit = edits[0] if len(edits) == 1 else CompoundEdit(edits)
        if label:
            edit.label = label
        self.history.push(edit)
        # Object mode only (the 2026-09-13 audit's clay-03): in an element mode
        # ``selection`` is the shape being *edited* and ``element_sel`` its
        # picks -- selecting the new object there would pull the edit away from
        # the shape the picks are on, so the Properties panel's summary line
        # and its identity/transform/material rows disagreed about which
        # object they were for.
        if self.element_mode == "object":
            self.select([obj.uid for obj in added])
        self.touch()
        return added

    def remove_object(self, uid: int) -> bool:
        """Delete *uid*, re-parenting its children onto its own parent, as
        **one** step.

        A child re-parented straight to ``None`` (the removed object's own
        parent) rather than left naming a uid the document no longer has:
        the alternative, an orphaned ``parent`` value, is exactly the
        dangling reference :mod:`.serialize` refuses to load and
        :meth:`world_matrix` would otherwise have to guess about. World
        placement is kept, the same "reparenting keeps world placement"
        decision :meth:`set_parent` states, computed from each child's
        current world matrix *before* the object it is relative to is gone.

        A child whose kept placement is sheared has the shear baked into its
        mesh (and its generator frozen), so it neither moves nor stores a
        non-unit rotation -- :meth:`_kept_placement`, which also carries that
        child's own descendants.
        """
        self._remove_batch([uid], None)
        return True

    def remove_objects(self, uids: Iterable[int]) -> list[str]:
        """Delete every one of *uids* as **one** step. -> the refusals, as
        sentences, of the ones that could not go (their children cannot keep
        their place); everything else is removed.

        Equivalent to :meth:`remove_object` per uid, highest list index first,
        folded into one step -- which is what Delete on a selection did, and
        why it was quadratic (the 2026-10-10 audit's clay-32): each call
        rebuilt the uid map, scanned for the children and popped from the
        middle of the list, 1.5 s at 4,000 objects. One map and one parent ->
        children map are kept up to date here instead, and the list is
        rebuilt once at the end.
        """
        refusals: list[str] = []
        self._remove_batch(uids, refusals)
        return refusals

    def _remove_batch(self, uids: Iterable[int], refusals: list[str] | None) -> None:
        """The one body behind :meth:`remove_object` and :meth:`remove_objects`.
        A refusal raises when *refusals* is ``None`` (the single-object door,
        which has nothing else to be atomic about) and is collected otherwise.
        """
        wanted = {int(u) for u in uids}
        by_uid = self._uid_map()
        for uid in wanted:
            if uid not in by_uid:
                raise KeyError(f"no object with uid {uid}")
        if not wanted:
            return
        wanted = self._without_skin_bones(wanted, by_uid, refusals)
        if not wanted:
            return
        pos: dict[int, int] = {}
        kids: dict[int | None, dict[int, None]] = {}
        for i, o in enumerate(self.objects):
            pos.setdefault(o.uid, i)
            kids.setdefault(o.parent, {})[o.uid] = None
        # Highest index first: each ``ObjectRemoveEdit`` then names the index
        # the object actually sat at when it was popped, and undo (reversed)
        # re-inserts ascending, the only order in which they are all correct.
        order = sorted(wanted, key=pos.__getitem__, reverse=True)

        edits: list[Any] = []
        gone: list[int] = []
        for uid in order:
            obj = by_uid[uid]
            new_parent = obj.parent
            children = sorted(kids.get(uid, ()), key=pos.__getitem__)
            # Every child's new placement is computed, and checked finite,
            # before the first assignment (the 2026-10-07 audit's clay-04): a
            # refusal part-way through used to leave the earlier children
            # already re-parented with no history step.
            try:
                kept = {
                    c: self._kept_placement(
                        by_uid[c], self._world_in(c, by_uid), new_parent, by_uid
                    )
                    for c in children
                }
            except el.OpError as error:
                if refusals is None:
                    raise
                refusals.append(str(error))
                continue
            for c in children:
                child = by_uid[c]
                child.parent = new_parent
                edits.append(ObjectPropsEdit(c, {"parent": uid}, {"parent": new_parent}))
                edits.extend(self._placement_edits(child, kept[c]))
            kids.pop(uid, None)
            siblings = kids.get(new_parent)
            if siblings is not None:
                siblings.pop(uid, None)
            if children:
                kids.setdefault(new_parent, {}).update(dict.fromkeys(children))
            del by_uid[uid]
            gone.append(uid)
            edits.append(ObjectRemoveEdit(pos[uid], obj))

        if not gone:
            return
        pruned = self._prune_animation_edit(gone)
        if pruned is not None:
            edits.append(pruned)
        # In place: nothing else may be holding a stale list.
        doomed = set(gone)
        self.objects[:] = [o for o in self.objects if o.uid not in doomed]
        self.selection.difference_update(doomed)
        for uid in gone:
            self.element_sel.pop(uid, None)
            # A stamp naming an object that no longer exists is worse than a
            # missing one: undo can bring the uid back with a fresh object at
            # some later address, and an old stamp entry would otherwise linger
            # keyed to a mesh that object never had.
            self._mesh_stamps.pop(uid, None)
        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        self.touch()

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
        # The 2026-10-07 audit's clay-06: hiding through the eye left the object
        # selected, so the gizmo, the Properties readouts and every object op
        # still addressed a thing the user could not see. Hiding deselects, as
        # in Blender; its picks go with it, so the set of objects being edited
        # may become empty while an element mode is on.
        # Selection is not undoable, so Ctrl+Z shows the object but does not
        # re-select it.
        hidden = {uid for uid, visible in wanted.items() if not visible}
        self.selection -= hidden
        for uid in hidden:
            self.element_sel.pop(uid, None)
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

        """
        obj = self.by_uid(uid)
        if mesh is obj.mesh:
            return False
        # The 2026-10-07 audit's clay-14: a typed 40-digit move (and a UV
        # transform of the same size) cast to float32 as inf and arrived here
        # with nothing to look at it, pushing a step whose positions the writer
        # refuses -- a document that cannot be saved until the step is undone.
        # The op layers clamp what they can; this is the backstop for any
        # caller, present and future, and it refuses before anything changes.
        mesh = self._with_skin_of(obj, mesh)
        self._refuse_invalid(obj.name, mesh)
        self._refuse_unstorable(obj.name, mesh)
        self._refuse_past_triangle_ceiling(obj, mesh)
        keyed = self._triangle_key()
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
        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        if select is not None:
            self.set_element_sel(uid, select)
        elif uid in self.element_sel and (
            len(mesh.positions) != len(before.positions)
            or bm.face_count(mesh) != bm.face_count(before)
        ):
            # The 2026-10-11 audit's clay-13: only a caller that handed
            # ``select`` reconciled the picks, so ``scratch.transplant`` (an
            # applied Familiar preview that deleted forty faces) left ten picks
            # standing past the end of the mesh and ``delete_faces`` then raised
            # ``IndexError``. Every caller is covered here instead of each one
            # remembering: when the counts moved, what the mesh no longer has
            # is dropped (``el.restrict``, range-only because *prior* is not
            # passed -- the counts already differ). A same-count replacement
            # (an element drag) keeps its picks exactly as before.
            self.set_element_sel(uid, el.restrict(mesh, self.element_sel[uid]))
        self.touch()
        self._carry_triangle_total(keyed, _triangles_of(mesh) - _triangles_of(before))
        return True

    def join_objects(
        self,
        target_uid: int,
        mesh: bm.Mesh,
        others: Iterable[int],
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

        """
        obj = self.by_uid(target_uid)
        self._refuse_skinned(
            [obj, *(self.by_uid(int(u)) for u in others if int(u) != target_uid)], "joined"
        )
        # One position map and one parent -> children map for the whole call
        # (the 2026-10-10 audit's clay-23 follow-up): ``children_of`` is a linear
        # scan and this door asked it twice per absorbed object, ~0.6 s of the
        # 0.7 s a 4,000-object join took. The map is kept current below as each
        # absorbed object hands its children on, as ``_remove_batch`` keeps its own.
        pos: dict[int, int] = {}
        kids: dict[int | None, dict[int, None]] = {}
        for i, o in enumerate(self.objects):
            pos.setdefault(o.uid, i)
            kids.setdefault(o.parent, {})[o.uid] = None
        wanted = {int(u) for u in others} - {target_uid}
        for u in wanted:
            if u not in pos:
                raise KeyError(f"no object with uid {u}")
        doomed = sorted(wanted, key=pos.__getitem__, reverse=True)
        if mesh is obj.mesh and not doomed:
            return False
        # The 2026-10-03 audit's clay-04: re-parenting an absorbed object's
        # children raises OpError under a zero-scale ancestor, and used to do so
        # after the target's mesh was already replaced -- a refusal with the
        # document half-changed and no history step. Every refusal is raised
        # here, before the first assignment. An ancestor's world matrix is
        # unchanged by re-parenting (world placement is kept), so checking the
        # current one is the same question the loop below asks.
        # The 2026-10-09 audit's clay-17: the merged mesh was adopted with no
        # finiteness check (a 1e-20 and a 1e20 scale joined to ``inf``), the one
        # refusal ``set_mesh`` makes that this door skipped.
        if mesh is not obj.mesh:
            self._refuse_unstorable(obj.name, mesh)
        for uid in doomed:
            new_parent = self.by_uid(uid).parent
            if new_parent is not None and kids.get(uid):
                self._require_invertible(new_parent)
            # ...and, the 2026-10-07 audit's clay-04, that the result is finite:
            # a denormal ancestor inverts without raising and decomposes to NaN.
            for c in sorted(kids.get(uid, ()), key=pos.__getitem__):
                self._kept_placement(self.by_uid(c), self.world_matrix(c), new_parent)
        before, obj.mesh = obj.mesh, mesh
        edits: list[Any] = [MeshEdit(target_uid, before, mesh)]
        props_before: dict[str, Any] = {}
        props_after: dict[str, Any] = {}
        if obj.generator is not None:
            props_before["generator"], props_before["params"] = obj.generator, obj.params
            props_after["generator"], props_after["params"] = None, {}
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
            children = sorted(kids.pop(uid, ()), key=pos.__getitem__)
            for c in children:
                child = self.by_uid(c)
                placement = self._kept_placement(child, self.world_matrix(c), new_parent)
                before = {"parent": child.parent}
                child.parent = new_parent
                edits.append(ObjectPropsEdit(c, before, {"parent": new_parent}))
                edits.extend(self._placement_edits(child, placement))
            siblings = kids.get(new_parent)
            if siblings is not None:
                siblings.pop(uid, None)
            if children:
                kids.setdefault(new_parent, {}).update(dict.fromkeys(children))
            index = self.index_of(uid)
            gone = self.objects.pop(index)
            self.selection.discard(uid)
            self.element_sel.pop(uid, None)
            self._mesh_stamps.pop(uid, None)
            edits.append(ObjectRemoveEdit(index, gone))
        # The target's own element selection names vertices of the mesh that
        # has just been replaced, so it describes geometry that is no longer
        # there -- the same reason ``_forget_elements`` drops one after an undo.
        self.element_sel.pop(target_uid, None)
        self.history.push(CompoundEdit(edits))
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
        the message a caller sees -- the agent surface's ``_h_transform``
        (``studio/modes/clay/agent/tools.py``) validates its own arguments
        before ever reaching this method, but an unvalidated ``clay_transform``
        once committed a two-element ``translation`` straight through this
        method with nothing to notice the wrong shape, and every later
        ``clay_scene`` raised trying to broadcast it into a 3x3 matrix
        (``viewer/math3d.py``'s ``compose`` does ``m[:3, 3] = t``) -- bricking
        introspection for the whole document, with no recovery but a blind
        undo. This method has other callers than ``_h_transform`` -- the
        properties panel, the gizmo drag -- so the assertion belongs here too,
        closing the door for every caller, present and future, rather than
        trusting each one to have validated first.

        """
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
            # The 2026-10-03 audit's clay-03: ``serialize._vector`` refuses an
            # all-zero scale and an all-zero quaternion on read, so a document
            # carrying either saved fine and could never be reopened (crash
            # recovery then lost the whole document). The writer and the reader
            # now agree: this door refuses what the reader would.
            if name in ("rotation", "scale") and not np.any(arr):
                raise el.OpError(
                    "A scale of zero on every axis collapses the object to a point."
                    if name == "scale"
                    else "A rotation of (0, 0, 0, 0) is not a rotation."
                )
        before = tuple(np.array(v, dtype="f8", copy=True) for v in (was or obj.trs()))
        if rotation is not None:
            # The 2026-10-07 audit's clay-69: ``[0, .5, 0, .5]`` was accepted as
            # given, ``quat_to_mat4`` does not normalise, so Clay drew a
            # non-orthonormal matrix while the exported node carried the raw
            # quaternion that every engine normalises -- two different poses.
            # Normalised here, divided by the largest component first so a
            # finite 1e200 cannot overflow the norm. A quaternion already unit
            # to within 1e-9 is left bit-for-bit alone: re-dividing it would
            # move it by an ulp and turn "set the value already there" into a
            # step. (A zero quaternion was refused above.)
            quat = np.array(rotation, dtype="f8", copy=True)
            with np.errstate(over="ignore", under="ignore"):
                if abs(float(np.linalg.norm(quat)) - 1.0) > 1e-9:
                    quat /= np.abs(quat).max()
                    quat /= np.linalg.norm(quat)
            rotation = quat
        after = tuple(
            obj.trs()[i] if new is None else np.array(new, dtype="f8", copy=True)
            for i, new in enumerate((translation, rotation, scale))
        )
        if all(np.array_equal(a, b) for a, b in zip(before, after, strict=True)):
            return False
        if obj.skin is not None:
            raise el.OpError(
                f"{obj.name!r} is weighted to a skeleton, and a weighted mesh's own transform "
                "is ignored when it is exported. Move its bones instead."
            )
        obj.translation, obj.rotation, obj.scale = after
        self.history.push(TransformEdit(uid, before, after))  # type: ignore[arg-type]
        self.touch()
        return True

    # -- animation ---------------------------------------------------------

    def _clamp_active_animation(self) -> None:
        """Keep the selected animation a real index (an undo can shrink the list)."""
        count = len(self.rig_state.animations)
        if self.active_animation is not None and not 0 <= self.active_animation < count:
            self.active_animation = count - 1 if count else None

    def animation(self, index: int | None = None) -> anim.Animation | None:
        """The animation at *index*, or the selected one, or ``None``."""
        index = self.active_animation if index is None else index
        animations = self.rig_state.animations
        if index is None or not 0 <= index < len(animations):
            return None
        return animations[index]

    def set_rig_state(self, state: anim.RigState) -> bool:
        """Replace the animations and frame rate as **one** step. -> whether anything changed.

        The one door every animation edit below goes through. Refused (``OpError``, nothing
        pushed) when a channel names an object that is not a bone, so a document can never
        hold motion the reader would refuse.
        """
        if state is self.rig_state:
            return False
        bones = {obj.uid for obj in self.objects if obj.bone}
        for animation in state.animations:
            stray = set(animation.channels) - bones
            if stray:
                raise el.OpError(
                    f"The animation {animation.name!r} animates something that is not a bone."
                )
        before, self.rig_state = self.rig_state, state
        self.history.push(RigEdit(before, state))
        self._clamp_active_animation()
        self.touch()
        return True

    def set_active_animation(self, index: int | None) -> None:
        """Select an animation. Not an edit: nothing is pushed and nothing is dirtied."""
        count = len(self.rig_state.animations)
        self.active_animation = index if index is not None and 0 <= index < count else None

    def _fresh_animation_name(self, base: str) -> str:
        taken = {a.name for a in self.rig_state.animations}
        if base not in taken:
            return base
        n = 1
        while f"{base}.{n:03d}" in taken:
            n += 1
        return f"{base}.{n:03d}"

    def add_animation(self, name: str = "Animation", *, copy_of: int | None = None) -> int:
        """A new empty animation (or a copy of *copy_of*), selected. -> its index."""
        animations = self.rig_state.animations
        if len(animations) >= anim.MAX_ANIMATIONS:
            raise el.OpError(
                f"A rig holds at most {anim.MAX_ANIMATIONS} animations. Delete one first."
            )
        label = self._fresh_animation_name(name.strip() or "Animation")
        if copy_of is not None:
            source = animations[copy_of]
            fresh = anim.Animation(label, source.loop, source.interpolation, source.channels)
        else:
            fresh = anim.Animation(label)
        self.set_rig_state(self.rig_state.with_animations((*animations, fresh)))
        self.active_animation = len(self.rig_state.animations) - 1
        return self.active_animation

    def delete_animation(self, index: int) -> bool:
        animations = list(self.rig_state.animations)
        if not 0 <= index < len(animations):
            return False
        del animations[index]
        return self.set_rig_state(self.rig_state.with_animations(animations))

    def _edit_animation(self, index: int | None, make: Any) -> bool:
        index = self.active_animation if index is None else index
        animations = list(self.rig_state.animations)
        if index is None or not 0 <= index < len(animations):
            raise el.OpError("Select or create an animation first.")
        replacement = make(animations[index])
        if replacement is animations[index]:
            return False
        animations[index] = replacement
        return self.set_rig_state(self.rig_state.with_animations(animations))

    def rename_animation(self, index: int, name: str) -> bool:
        name = name.strip()
        if not name:
            raise el.OpError("An animation needs a name.")
        if any(a.name == name for i, a in enumerate(self.rig_state.animations) if i != index):
            raise el.OpError(f"There is already an animation called {name!r}.")
        return self._edit_animation(
            index, lambda a: a if a.name == name else a.renamed(name)
        )

    def set_animation_loop(self, index: int, loop: bool) -> bool:
        return self._edit_animation(index, lambda a: a if a.loop == loop else a.with_loop(loop))

    def set_animation_interpolation(self, index: int, interpolation: str) -> bool:
        def change(a: anim.Animation) -> anim.Animation:
            return a if a.interpolation == interpolation else a.with_interpolation(interpolation)

        return self._edit_animation(index, change)

    def set_fps(self, fps: int) -> bool:
        if fps not in anim.FPS_CHOICES:
            raise el.OpError(f"The frame rate is one of {', '.join(map(str, anim.FPS_CHOICES))}.")
        if fps == self.rig_state.fps:
            return False
        return self.set_rig_state(self.rig_state.with_fps(fps))

    def _require_bone(self, uid: int) -> Obj:
        obj = self.by_uid(uid)
        if not obj.bone:
            raise el.OpError(f"{obj.name!r} is not a bone, so it cannot be animated.")
        return obj

    def set_key(
        self, uid: int, prop: str, time: float, value: Any, *, animation: int | None = None
    ) -> bool:
        """Key one property of one bone at *time*, as one step. -> whether anything changed.

        *value* is the bone's absolute local transform at that moment (a quaternion for
        ``rotation``). A value equal to the one already keyed there changes nothing.
        """
        self._require_bone(uid)
        target = self.animation(animation)
        if target is not None:
            old = target.channels.get(uid, {}).get(prop)
            if old is not None:
                near = np.flatnonzero(np.abs(old.times - time) <= anim.TIME_EPSILON)
                row = np.asarray(value, dtype="f8").reshape(-1)
                if prop == "rotation" and np.linalg.norm(row) > 0:
                    row = row / np.linalg.norm(row)
                if len(near) and np.array_equal(old.values[near[0]], row):
                    return False
        return self._edit_animation(
            animation, lambda a: a.with_key(uid, prop, float(time), value)
        )

    def delete_key(
        self, uid: int, prop: str, time: float, *, animation: int | None = None
    ) -> bool:
        return self._edit_animation(animation, lambda a: a.without_key(uid, prop, float(time)))

    def move_key(
        self, uid: int, prop: str, time: float, to: float, *, animation: int | None = None
    ) -> bool:
        return self._edit_animation(animation, lambda a: a.moved(uid, prop, float(time), float(to)))

    def clear_bone_keys(self, uid: int, *, animation: int | None = None) -> bool:
        return self._edit_animation(animation, lambda a: a.without_bones([uid]))

    def _prune_animation_edit(self, uids: Iterable[int]) -> RigEdit | None:
        """Drop the channels of bones that are going away, applied now, and answer the
        step that records it -- or ``None`` when nothing animated them. For the doors that
        delete or unmark bones to fold into their own compound step."""
        gone = {int(u) for u in uids}
        animations = self.rig_state.animations
        pruned = tuple(a.without_bones(gone) for a in animations)
        if all(new is old for new, old in zip(pruned, animations, strict=True)):
            return None
        before, self.rig_state = self.rig_state, self.rig_state.with_animations(pruned)
        return RigEdit(before, self.rig_state)

    # -- bones and skins ---------------------------------------------------

    def _with_skin_of(self, obj: Obj, mesh: bm.Mesh) -> bm.Mesh:
        """*mesh* with *obj*'s vertex weights on it -- the one place a skinned object's
        mesh is reconciled with an edit, so no op has to remember to.

        A weighted object's edits come in three kinds. A result that already carries
        weights (``dataclasses.replace`` keeps them) passes through, and so does any edit
        of an object that has none. A result with the **same vertex count and no weights**
        -- shading, UVs, flipped normals, a triangulation, an element drag -- gets the old
        weights back: those ops keep the vertices where they were, and an op that
        reordered them while keeping their number does not exist in this package. Any
        other result is refused, naming the way out, because nothing could say which
        weights a new vertex should take.
        """
        if obj.mesh.joints is None or mesh.joints is not None:
            return mesh
        if len(mesh.positions) == len(obj.mesh.positions):
            return replace(mesh, joints=obj.mesh.joints, weights=obj.mesh.weights)
        raise el.OpError(
            f"{obj.name!r} is weighted to a skeleton, and that edit would change its "
            "number of points, so nothing could say how they should move. Run Unbind Skin "
            "on it first (its weights are lost), or edit the mesh before rigging it."
        )

    @staticmethod
    def _refuse_skinned(objs: Sequence[Obj], verb: str) -> None:
        for obj in objs:
            if obj.skin is not None:
                raise el.OpError(
                    f"{obj.name!r} is weighted to a skeleton and cannot be {verb} without "
                    "losing its weights. Run Unbind Skin on it first."
                )

    def _without_skin_bones(
        self, wanted: set[int], by_uid: Mapping[int, Obj], refusals: list[str] | None
    ) -> set[int]:
        """*wanted* less the bones some surviving skin still names -- refused by name.

        Deleting a bone out from under a weighted mesh would leave a skin pointing at a
        uid the document no longer has, which the reader refuses; so the bone stays, with
        a sentence saying which mesh needs its skin unbound first. A single-object delete
        (``refusals is None``) raises, as its other refusals do.
        """
        users: dict[int, str] = {}
        for obj in by_uid.values():
            if obj.skin is not None and obj.uid not in wanted:
                for joint in obj.skin.joints:
                    users.setdefault(joint, obj.name)
        kept = set(wanted)
        for uid in sorted(wanted & users.keys()):
            message = (
                f"{by_uid[uid].name!r} is a bone that {users[uid]!r} is weighted to. "
                f"Run Unbind Skin on {users[uid]!r} first."
            )
            if refusals is None:
                raise el.OpError(message)
            refusals.append(message)
            kept.discard(uid)
        return kept

    def set_bone(self, uids: Iterable[int], flag: bool) -> bool:
        """Mark objects as bones, or clear the mark, as **one** step. -> whether anything changed.

        Refused (:class:`~.elements.OpError`, nothing pushed) when it would take the
        document past :data:`~.rig.MAX_BONES`, or clear a bone some weighted mesh's skin
        still names. An object already in the wanted state is left alone and costs no step.
        """
        by_uid = self._uid_map()
        wanted = [by_uid[int(u)] for u in dict.fromkeys(int(u) for u in uids) if int(u) in by_uid]
        flag = bool(flag)
        changing = [obj for obj in wanted if obj.bone != flag]
        if not changing:
            return False
        if flag:
            marked = sum(1 for o in self.objects if o.bone) + len(changing)
            if marked > rig.MAX_BONES:
                raise el.OpError(
                    f"That would make {marked} bones, past the {rig.MAX_BONES} a Clay rig can "
                    "hold. Unmark some bones first."
                )
        else:
            self._without_skin_bones({o.uid for o in changing}, by_uid, None)
        edits: list[Any] = []
        for obj in changing:
            obj.bone = flag
            edits.append(ObjectPropsEdit(obj.uid, {"bone": not flag}, {"bone": flag}))
        if not flag:
            pruned = self._prune_animation_edit(obj.uid for obj in changing)
            if pruned is not None:
                edits.append(pruned)
        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        self.touch()
        return True

    def unbind_skin(self, uid: int) -> bool:
        """Drop an object's weights and its skin as **one** step. -> whether it had any.

        The mesh keeps its shape; it is rigidly bound again if it sits under a bone and
        unbound if it does not. The way out of the refusals that guard a weighted mesh's
        point count, its joining and its separating, and the bones it names.
        """
        obj = self.by_uid(uid)
        if obj.skin is None and obj.mesh.joints is None:
            return False
        before = obj.mesh
        bare = replace(before, joints=None, weights=None)
        was_skin = obj.skin
        obj.mesh, obj.skin = bare, None
        self.history.push(
            CompoundEdit(
                [
                    MeshEdit(uid, before, bare),
                    ObjectPropsEdit(uid, {"skin": was_skin}, {"skin": None}),
                ]
            )
        )
        self.touch()
        return True

    def rig_problems(self) -> list[str]:
        """Every way bones and skins disagree in this document -- empty is healthy."""
        return rig.skin_problems(self.objects)

    def set_props(self, uid: int, *, was: dict[str, Any] | None = None, **props: Any) -> bool:
        """Name, visibility, generator, params, default material -- one step.

        ``was`` is the counterpart of :meth:`set_transform`'s, and the trap it
        avoids is sharper here because ``params`` is a dict: a panel that edits
        the object's own dict in place and then passes it back would hand this
        the very object it is comparing against, so "before" and "after" would
        be the same value and the change would record nothing at all. Such a
        caller passes the values it started with as ``was``; a caller that
        builds a fresh dict -- which is what a widget reading a form does --
        needs none of this.

        ``parent`` is refused by name: it has its own door, :meth:`set_parent`,
        which is the only one that checks for a cycle -- this generic one does
        not, and must not be used to bypass it.
        """
        if "parent" in props:
            raise el.OpError(
                "Use set_parent to change an object's parent -- it is the only "
                "door that refuses a cycle."
            )
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

        """
        obj = self.by_uid(uid)
        before = {"params": was.get("params", obj.params)}
        # The 2026-10-11 audit's clay-41: a reopened box's ``size`` is a list
        # (the file is JSON) and the widget hands back a tuple, so the same
        # three numbers compared unequal and retyping them pushed a step. The
        # comparison reads both through one container type; what is stored is
        # untouched.
        same_params = _canonical(before["params"]) == _canonical(params)
        # The 2026-10-09 second audit's clay-102: a rebuilt mesh is always a new
        # object, so ``mesh is not obj.mesh`` was the only no-op test and typing
        # the radius a field already held pushed a step and turned the document
        # dirty. Equal numbers that build array-equal geometry change nothing,
        # exactly as ``set_transform`` and ``set_props`` already answer.
        if same_params and (mesh is obj.mesh or _same_geometry(mesh, obj.mesh)):
            return False
        # The 2026-10-09 audit's clay-01 and clay-02: this door assigned the
        # rebuilt mesh with neither of ``set_mesh``'s refusals, so a torus of
        # radius 3e38 stored ``inf`` (every save then refused until undo) and a
        # segment bump could take the document past the triangle ceiling.
        # Checked before the first assignment, so a refusal changes nothing.
        if mesh is not obj.mesh:
            self._refuse_invalid(obj.name, mesh)
            self._refuse_unstorable(obj.name, mesh)
            self._refuse_past_triangle_ceiling(obj, mesh)
        keyed = self._triangle_key()
        old_mesh = obj.mesh
        edits: list[Any] = []
        if not same_params:
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
                # clay-05 (2026-10-03): one changed key at the same face count
                # moves positions only (the same rule ``regen.carry_over``
                # trusts), so the same faces are still selected; ``prior`` is
                # withheld there. Several keys at the same count keep the
                # selection only when the faces are provably the same faces
                # (``regen.same_faces``, the 2026-10-07 audit's clay-08: the
                # key count alone dropped it on ``{"radius", "height"}``);
                # anything else keeps the drop-on-same-count policy.
                old_params = before["params"]
                changed = [
                    k
                    for k in set(old_params) | set(params)
                    if _canonical(old_params.get(k)) != _canonical(params.get(k))
                ]
                same = len(mesh.starts) == len(was_mesh.starts) and (
                    len(changed) <= 1 or regen.same_faces(was_mesh, mesh)
                )
                self.set_element_sel(
                    uid,
                    el.restrict(mesh, existing, prior=None if same else was_mesh),
                )
        if not edits:
            return False
        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        self.touch()
        self._carry_triangle_total(keyed, _triangles_of(obj.mesh) - _triangles_of(old_mesh))
        return True

    # -- separate ------------------------------------------------------------

    def separate(self, uid: int, pieces: Sequence[bm.Mesh]) -> list[Obj]:
        """*uid* replaced by one new object per *pieces*, as **one** step.

        Every piece keeps the source's parent and transform. The generator is frozen (as
        :meth:`set_mesh` freezes it): a piece of a sphere is not "sphere,
        radius 1". Names are suffixed (:func:`~.ops.next_name`) and the
        source object is removed.

        Refuses (OpError, nothing pushed) fewer than two pieces, which the
        kernel functions in :mod:`.separate` already refuse to produce; this is
        the same refusal for a caller that built ``pieces`` some other way.
        """
        from . import ops as mesh_ops

        pieces = list(pieces)
        obj = self.by_uid(uid)
        self._refuse_skinned([obj], "separated")
        if len(pieces) < 2:
            raise el.OpError("Nothing to separate: that would produce a single piece.")
        for mesh in pieces:
            bm.validate(mesh)

        # ``UsedNames``, not a plain set: N pieces of one name probed from
        # ``.001`` each time otherwise (the 2026-10-03 audit's naming follow-up).
        taken = mesh_ops.UsedNames(o.name for o in self.objects)
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
                    parent=obj.parent,
                )
            )
        # The 2026-10-07 audit's clay-13: a split that takes the document past
        # the object ceiling builds a file the writer refuses (the pieces
        # replace the source, hence ``outgoing``). Checked before the first
        # assignment, so a refusal changes nothing.
        self._refuse_past_ceilings(new_objs, [obj])

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
        by_uid = self._uid_map()
        child_worlds = {c: self._world_in(c, by_uid) for c in children}
        # Computed, and checked finite, before the first assignment (the
        # 2026-10-07 audit's clay-04).
        kept = {
            c: self._kept_placement(self.by_uid(c), child_worlds[c], new_parent)
            for c in children
        }
        for c in children:
            child = self.by_uid(c)
            before = {"parent": child.parent}
            child.parent = new_parent
            edits.append(ObjectPropsEdit(c, before, {"parent": new_parent}))
            edits.extend(self._placement_edits(child, kept[c]))
        for i, piece in enumerate(new_objs):
            self.objects.insert(index + i, piece)
            edits.append(ObjectAddEdit(index + i, piece))
        # The source's own new position, now pushed forward by every piece
        # inserted ahead of it -- the same re-lookup ``join_objects`` and
        # ``remove_object`` use rather than hand computing ``index + len``.
        removed_index = self.index_of(uid)
        removed = self.objects.pop(removed_index)
        self.selection.discard(uid)
        # The pieces replace the source in the selection in every mode (the
        # 2026-10-03 audit's clay-30 kept them out of an element mode's
        # ``selection``, which was derived then): in an element mode the pieces
        # become the shapes being edited, with nothing picked inside them yet.
        # Visible ones only -- a hidden source's pieces are hidden too, and a
        # hidden object holds no selection (``selection_problems``).
        self.selection.update(o.uid for o in new_objs if o.visible)
        self.element_sel.pop(uid, None)
        self._mesh_stamps.pop(uid, None)
        edits.append(ObjectRemoveEdit(removed_index, removed))
        self.history.push(CompoundEdit(edits))
        self.touch()
        return new_objs

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
        diverged past it. The undo stack's :attr:`~..core.undo.UndoStack.floor`
        (``0``, which no edit ever has as its serial, until something is
        evicted; then the last evicted step's) resolves to "everything undone".
        """
        # The evicted floor (0 while nothing has been evicted) is the position
        # under the oldest step still held, so undoing everything reaches it.
        # Serial 0 used to be that unconditionally -- the 2026-10-10 audit's
        # clay-06: a checkpoint named on an empty history, after the stack had
        # evicted past it, "restored" to the bottom of what was left (16 of 80
        # objects) and reported success.
        if serial == self.history.floor:
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
        the properties panel's Remove button acts on the slot the Colours page
        is editing -- the selected object's own default, or one Ctrl+clicked in
        the palette -- and :meth:`remove_material_and_reassign` re-points a
        default that named the removed slot immediately afterwards, so counting
        defaults would disable the only control that reaches this.
        """
        return sum(
            int((obj.mesh.material == index).sum())
            for obj in _material_holders(self, redoable=True)
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

    def repaint_object(self, uid: int, index: int) -> bool:
        """Point *uid*'s default slot **and every face** at palette entry *index*,
        as one step. -> whether anything changed.

        The 2026-10-03 audit's clay-17: ``Obj.material`` is only the slot new
        faces are stamped with; what renders and exports is the per-face
        ``mesh.material`` array, so a lone ``set_props(material=...)`` left an
        existing object looking exactly as before.
        """
        if not 0 <= index < len(self.materials):
            raise el.OpError(f"there is no palette entry {index}.")
        obj = self.by_uid(uid)
        head = self.history.head
        mark = self.history.mark()
        try:
            if obj.mesh.material.size and not np.all(obj.mesh.material == index):
                painted = replace(
                    obj.mesh, material=np.full(len(obj.mesh.material), index, dtype="i4")
                )
                self.set_mesh(uid, self._with_uvs_for(painted, index), keep_generator=True)
            self.set_props(uid, material=index)
        finally:
            self.history.collapse_since(mark)
        return self.history.head != head

    def paint_faces(self, uid: int, faces: Sequence[int], index: int) -> bool:
        """Point the given faces of one object at palette entry *index*, as one
        step. -> whether anything changed.

        The face-level sibling of :meth:`repaint_object`: ``mesh.material`` is
        one index per face and nothing else assigned a slot to a *selection*.
        An empty *faces*, or faces that already name *index*, is ``False`` with
        nothing pushed. The generator claim is kept -- a face's material is not
        geometry, so a painted box is still a box -- and undo is by *uid*, like
        every other edit, so it survives the object being re-ordered.

        Raises :class:`~.elements.OpError` for a slot outside the palette or a
        face outside the mesh, before anything changes.
        """
        if not 0 <= index < len(self.materials):
            raise el.OpError(f"there is no palette entry {index}.")
        obj = self.by_uid(uid)
        picked = np.unique(np.asarray(faces, dtype="i8").reshape(-1))
        if not len(picked):
            return False
        count = len(obj.mesh.material)
        if picked[0] < 0 or picked[-1] >= count:
            bad = int(picked[0] if picked[0] < 0 else picked[-1])
            raise el.OpError(f"{obj.name} has no face {bad} ({count} faces).")
        if np.all(obj.mesh.material[picked] == index):
            return False
        painted = np.array(obj.mesh.material, dtype="i4", copy=True)
        painted[picked] = index
        return self.set_mesh(
            uid,
            self._with_uvs_for(replace(obj.mesh, material=painted), index),
            keep_generator=True,
        )

    def _with_uvs_for(self, mesh: bm.Mesh, index: int) -> bm.Mesh:
        """*mesh*, box-unwrapped when it has no UVs and palette entry *index*
        carries a texture; otherwise *mesh* itself.

        The 2026-10-07 audit's clay-40: painting a textured slot onto an
        object with no UVs put the texture on faces with no coordinates -- the
        manual's "one texel everywhere" -- with no unwrap and no warning.
        :meth:`add_texture` already unwraps every UV-less object that wears
        the slot it textures; this is the same rule for the other direction
        (the slot already textured, the object arriving), and it rides inside
        the caller's one ``set_mesh`` so the paint and the unwrap are one undo
        step. An object that already has UVs keeps them: an author's layout is
        not ours to redo.
        """
        if mesh.uv is None and self.materials[index].base_color is not None:
            return uv_projection.box_unwrap(mesh)
        return mesh

    def add_texture(self, index: int, size: int = 64) -> bool:
        """Give palette entry *index* a blank ``size`` x ``size`` base-colour
        texture, and box-unwrap what needs it, as **one** step. -> whether
        anything changed.

        The image is opaque RGBA filled with the slot's colour (sRGB-encoded,
        since a texture is read as sRGB and ``base_color_factor`` is linear),
        flagged ``nearest`` so it renders crisp. The slot's colour factor goes to
        white (alpha kept): the shader multiplies factor by texel, so leaving
        the colour in both places would render it squared, darker than the
        swatch the user just clicked. Every object that has a face on
        this slot and no UVs at all is box-unwrapped with it -- a texture on a
        mesh with no coordinates would sample one texel everywhere. Objects that
        already have UVs are left alone: an author's layout is not ours to redo.
        The unwrap is a UV change, not a topology change, so generator claims
        stay.

        Refuses (:class:`~.elements.OpError`, nothing pushed) a *size* other
        than 32, 64 or 128, a missing slot, and a slot that already has a
        texture: replacing one is an Inker round trip's job, and a refused click
        beats a silent loss of someone's painting.
        """
        if size not in TEXTURE_SIZES:
            sizes = ", ".join(str(s) for s in TEXTURE_SIZES)
            raise el.OpError(f"texture size must be {sizes}, not {size}.")
        if not 0 <= index < len(self.materials):
            raise el.OpError(f"there is no palette entry {index}.")
        material = self.materials[index]
        if material.base_color is not None:
            raise el.OpError(f"{material.name or 'this material'} already has a texture.")
        mark = self.history.mark()
        try:
            self.set_material(
                index,
                replace(
                    material,
                    base_color=(size, size, _flat_rgba(material.base_color_factor, size)),
                    base_color_factor=(1.0, 1.0, 1.0, float(material.base_color_factor[3])),
                    nearest=True,
                ),
            )
            for obj in list(self.objects):
                if obj.mesh.uv is None and np.any(obj.mesh.material == index):
                    self.set_mesh(obj.uid, uv_projection.box_unwrap(obj.mesh), keep_generator=True)
        finally:
            self.history.collapse_since(mark)
        return True

    def add_material_and_assign(
        self, uid: int, material: gltf.Material | None = None, *, repaint: bool = False
    ) -> int:
        """Append a palette entry and point an object's default slot at it, as
        **one** step. -> the new entry's index.

        The 2026-09-08 audit's clay-02: the properties panel's Add button used
        to call :meth:`add_material` and then :meth:`set_props` as two
        separate pushes, so a single Ctrl+Z after the click left a stray,
        unreferenced palette entry behind instead of restoring the object's
        original slot -- "one press, one Ctrl+Z" applies here exactly as it
        does to :meth:`join_objects`' merge-and-removals.

        The 2026-09-26 audit's clay-document-06: ``set_props`` used to run
        outside a ``try`` here, so a bad ``uid`` (an object deleted out from
        under a stale panel reference) raised past ``collapse_since`` and left
        the gesture open forever -- ``UndoStack._open_gestures`` never
        dropped back to zero, so eviction stayed deferred for the rest of the
        session. The ``finally`` closes the gesture on every path, including
        this one, whether or not there was a run to fold.
        """
        mark = self.history.mark()
        try:
            index = self.add_material(material)
            if repaint:
                self.repaint_object(uid, index)
            else:
                self.set_props(uid, material=index)
        finally:
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

        The 2026-09-26 audit's clay-document-06: the ``set_props`` call ran
        outside a ``try`` here too, so the same bad-``uid`` failure that
        :meth:`add_material_and_assign` could hit left this gesture open as
        well. Wrapped in ``finally`` for the same reason.

        **The default is repointed only when it named the removed slot.** The
        2026-10-09 second audit's clay-57: the Remove button now acts on the
        slot the Colours page edits, which a Ctrl+click can make a slot the
        object does not wear at all, and this used to rewrite ``obj.material``
        to ``min(index, last)`` regardless -- removing an unused slot 0 from a
        shape wearing slot 2 left its faces renumbered to 1 and its default at
        0, an unrelated neighbour. A default that did *not* name *index* is
        already renumbered correctly by :func:`~.edits._shift_materials`, so it
        is left to that. (An unknown *uid* is still a ``KeyError``, raised
        before anything is removed.)
        """
        mark = self.history.mark()
        try:
            wore_it = self.by_uid(uid).material == index
            removed = self.remove_material(index)
            if removed and wore_it:
                self.set_props(uid, material=min(index, len(self.materials) - 1))
        finally:
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
        """Replace the selection -- in an element mode, the set of objects being
        edited. Pushes no step, by design; see the module docstring. It still
        bumps ``rev``, because the viewport draws the selected object's outline
        and has to know to redraw it."""
        # The 2026-10-09 audit's clay-08: a hidden uid used to enter the
        # selection (the agent's ``clay_select``), which ``selection_problems``
        # then reported -- hiding deselects, so selecting a hidden object is
        # skipped the same way. A uid the document does not carry is skipped too.
        live = {o.uid for o in self.objects if o.visible}
        self.selection = {int(u) for u in uids if int(u) in live}
        # In an element mode the picks follow the objects being edited: what is
        # no longer selected keeps no picks (I2). Object mode has none to drop.
        if self.element_sel:
            self.element_sel = {
                uid: sel for uid, sel in self.element_sel.items() if uid in self.selection
            }
        self.touch()

    # -- element mode (also not undoable) ----------------------------------

    def set_element_mode(self, mode: str) -> None:
        """Switch to object/vertex/edge/face mode, converting what is picked.

        **Removes nothing from ``selection``, and only ever adds.** The shapes
        the user is working on stay selected on the way in and on the way out:
        entering an element mode from object mode picks nothing (there is
        nothing to convert) and the first pick adds its own object if it is not
        already being edited; picks that convert to a non-empty result put their
        objects into ``selection``. Leaving drops the picks, which are
        meaningless in object mode.
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
            # Only a caller that set picks in object mode (no path does) can
            # reach this with a pick on an unselected object; keep I2 anyway.
            self.selection |= set(converted)
        self.element_mode = mode
        self.touch()

    def set_element_sel(self, uid: int, sel: el.ElementSel | None) -> None:
        """Replace what is picked inside one object, keeping the invariant.

        A non-empty pick makes the object one of those being edited (adds it to
        ``selection``) and an empty one drops only the entry: the object stays
        selected, because it is the shape being edited and "nothing picked yet"
        is an ordinary state of that. Consumers that need the picks (the gizmo,
        a mesh op) read ``element_sel``, which is why they cannot be drawn at
        the centroid of nothing.
        """
        if el.is_empty(sel):
            self.element_sel.pop(uid, None)
        elif self._is_hidden(uid):
            # The 2026-10-09 audit's clay-08: a hidden object holds no picks,
            # as ``set_visibility`` leaves it (``selection_problems``).
            self.element_sel.pop(uid, None)
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
        """Drop every pick. The objects being edited stay selected."""
        self.element_sel = {}
        self.touch()

    def selection_problems(self) -> list[str]:
        """Every way ``selection`` and ``element_sel`` disagree with the rules
        the module docstring states -- ``[]`` when they hold. Pure: reads, never
        repairs, so a test (or a debug assertion) can say *which* rule broke.

        * ``selection`` names only live objects, and no hidden one holds a
          selection or a pick (hiding deselects, and an undo that hides again
          must too);
        * object mode holds no element picks at all;
        * in an element mode every picked object is one of the objects being
          edited (``element_sel`` keys are a subset of ``selection``), and each
          pick is non-empty and in range for that object's mesh.
        """
        problems: list[str] = []
        live = {obj.uid: obj for obj in self.objects}
        for uid in sorted(self.selection):
            if uid not in live:
                problems.append(f"selection names uid {uid}, which is not in the document")
            elif not live[uid].visible:
                problems.append(f"selection holds hidden object {uid}")
        if self.element_mode == "object":
            if self.element_sel:
                problems.append(f"object mode holds element picks on {sorted(self.element_sel)}")
            return problems
        for uid, sel in sorted(self.element_sel.items()):
            if uid not in live:
                problems.append(f"element_sel names uid {uid}, which is not in the document")
                continue
            if uid not in self.selection:
                problems.append(f"element_sel holds picks on {uid}, which is not selected")
            if not live[uid].visible:
                problems.append(f"element_sel holds picks on hidden object {uid}")
            if el.is_empty(sel):
                problems.append(f"element_sel holds an empty pick on {uid}")
                continue
            mesh = live[uid].mesh
            n_verts, n_faces = len(mesh.positions), bm.face_count(mesh)
            if len(sel.verts) and int(sel.verts.max()) >= n_verts:
                problems.append(f"{uid}: a picked vertex is past the mesh's {n_verts}")
            if len(sel.edges) and int(sel.edges.max()) >= n_verts:
                problems.append(f"{uid}: a picked edge ends past the mesh's {n_verts} vertices")
            if len(sel.faces) and int(sel.faces.max()) >= n_faces:
                problems.append(f"{uid}: a picked face is past the mesh's {n_faces}")
        return problems


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
    obj: Obj,
    materials: Sequence[gltf.Material],
    mesh: bm.Mesh | None = None,
    *,
    skin_data: bool = False,
) -> list[gltf.Primitive]:
    """One :class:`~gltf.Primitive` per material the object's faces use.

    A draw call carries one material, so a two-toned box has to be two
    primitives however it is stored -- which is why the split happens here, on
    the way out, rather than in the mesh: the mesh stays one object the user
    can select faces across, and the renderer gets the grouping it needs.

    Groups come out in palette-index order, so the same document produces the
    same primitive order every time -- an exporter's output is diffable, and a
    GPU cache keyed on position does not shuffle.

    ``mesh`` defaults to ``obj.mesh``; a caller drawing a drag preview or a
    scratch copy passes its own.

    ``skin_data`` carries a weighted mesh's per-vertex ``joints``/``weights`` onto each
    primitive, through the same split-and-shared vertex table the positions went through
    (:func:`~.mesh.render_sources`). Off by default and only :func:`to_model` asks: the
    viewport draws a weighted object at rest like any other, and a primitive that carried
    joints it had no skin for would only invite the GPU to skin it by nothing.
    """
    if mesh is None:
        mesh = obj.mesh
    if bm.face_count(mesh) == 0:
        return []
    prims = []
    for index in np.unique(mesh.material):
        faces = np.flatnonzero(mesh.material == index)
        sub = _submesh(mesh, faces)
        joints = weights = None
        if skin_data and mesh.joints is not None:
            layout = bm.render_layout(sub)
            positions, normals, uvs, indices = bm.render_from_layout(layout, sub.positions)
            sources = bm.render_sources(layout)
            joints, weights = mesh.joints[sources], mesh.weights[sources]
        else:
            positions, normals, uvs, indices = bm.render_arrays(sub)
        prims.append(
            gltf.Primitive(
                positions=positions,
                indices=indices,
                normals=normals,
                uvs=uvs,
                joints=joints,
                weights=weights,
                material=_material_at(materials, int(index)),
            )
        )
    return prims


_PLANS: weakref.WeakKeyDictionary[bm.Mesh, list[tuple[int, bm.RenderLayout]]] = (
    weakref.WeakKeyDictionary()
)
# The 2026-09-23 audit's clay-12: unsynchronized on the same false premise
# ``mesh.py``'s ``_RAW_CACHE`` carried before the 2026-09-12 audit's clay-04
# gave it a lock -- that every caller runs on the frame thread. The character
# pipeline's generators run on the MCP service lane's own ``TaskRunner``
# (``studio/agent_host.py``'s ``SERVICE_WORKERS`` pool), and a Familiar
# scratch preview (``kernels/mesh/scratch.py``'s ``clone``) shares ``Mesh``
# objects with the live document and runs its batch on ``realmspinner-task``;
# both are off the frame thread, and a call to :func:`render_plan` for a mesh
# shared with the live document can race this dict's get/set from both
# sides. Mirrors
# ``adjacency._CACHE_LOCK``: an uncontended acquire around a dict lookup, next
# to the numpy pass this function already does when it actually builds
# something.
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
            # The stash is only valid for the moved set it was computed under:
            # the 2026-10-03 audit's clay-39 found a second drag on this same
            # unchanged mesh reading the first drag's last-frame normals.
            previous=None if moved is None else bm.raw_face_normals(layout, moved),
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

    Split out of :func:`to_model` so a caller that needs to line its own list
    of objects up against ``to_model(doc).nodes`` asks one function which objects
    become nodes, rather than keeping a copy that could silently drift apart and
    misalign the zip.
    """
    keep: dict[int, bool] = {obj.uid: obj.visible for obj in doc.objects}
    # One uid -> parent map and one walk per chain, not ``doc.ancestors`` per
    # visible object: each hop there is a linear ``index_of`` scan, so the
    # cost was objects x depth x objects -- a 1,000-deep chain took 4.9 s and
    # a 4,096-deep one minutes (the 2026-10-03 audit's clay-85), stalling an
    # export and the viewport's per-rebuild visibility rule. Same answer as
    # ``ancestors``: first-seen uid wins on a duplicate, a dangling parent
    # ends the walk, and a cycle ends it at the first revisit.
    parent_of: dict[int, int | None] = {}
    for obj in doc.objects:
        parent_of.setdefault(obj.uid, obj.parent)
    walked: set[int] = set()
    for obj in doc.objects:
        if not obj.visible or obj.uid in walked:
            continue
        seen = {obj.uid}
        current = parent_of[obj.uid]
        while current is not None and current not in seen and current in parent_of:
            keep[current] = True
            if current in walked:
                break  # an earlier walk already marked everything above it
            walked.add(current)
            seen.add(current)
            current = parent_of[current]
    return [obj for obj in doc.objects if keep.get(obj.uid, False)]


def to_model(
    doc: ClayDoc,
    *,
    skinned: bool = True,
    place: np.ndarray | None = None,
    bone_names: Mapping[int, str] | None = None,
    at_time: float | None = None,
) -> gltf.Model:
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

    **A document with bones exports one skin** (Tranche 2 of the Clay rig plan), in the
    shape Blender writes its own: every bone is an ordinary node, hidden ones and their
    ancestors included (a joint is not optional), and the skin lists them in document
    order with ``inverse_bind = inv(world_rest(bone))``. A shape that moves with a bone --
    weighted to it, or simply under it -- cannot keep its node, because a skinned node's
    own transform is ignored by the spec; its geometry goes to a **scene-root node with an
    identity transform**, vertices baked into world rest space, ``JOINTS_0`` and
    ``WEIGHTS_0`` on every vertex. A shape bound rigidly gets the single joint it sits
    under at weight 1; a weighted one has its skin's joint indices remapped onto the
    document skin. The original node stays, without its mesh, so the hierarchy and every
    other node's world matrix are exactly what they were; the new nodes come after the
    ``kept_objects`` ones, so a caller that lines the two up still can. A document with no
    bones takes none of these branches and its bytes do not change.

    ``skinned=False`` is the static model: every shape on its own node at its own place,
    bones as plain nodes, no skin -- what the library's ``model.glb`` is, because rigging,
    retargeting and every mesh export are pure functions of a mesh that is not skinned.

    ``place`` is a 4x4 applied to the whole scene from outside (uniform scale and a
    translation: ``postprocess.normalize_glb``'s grounding, so a rig written beside a
    normalised ``model.glb`` stands where that mesh stands). It is folded into each root
    node's transform and into the baked vertices and inverse bind matrices of the skinned
    meshes, because nothing above a skinned mesh's joints could carry it for them.
    ``bone_names`` renames bones (and the geometry bound to them) for a record that wants
    legal names; both default to the document as it is.

    **Animation.** A rigged document writes its animations: one glTF animation each, a channel
    per animated property of a bone, ``STEP`` or ``LINEAR`` as the animation says, with a
    non-looping one marked ``{"loop": false}`` in its extras (glTF has no field for it). A root
    bone's translation and scale keys carry ``place`` as its node does. ``at_time`` instead
    writes the pose of the selected animation at that moment into the nodes and *no* animations
    -- "Export Posed": the meshes stay in the rest pose they were baked in and the joint nodes
    carry the pose, which is how a skinned file shows one.
    """
    kept = kept_objects(doc)
    by_uid = doc._uid_map()
    marked = [obj for obj in doc.objects if obj.bone] if skinned else []
    problems = doc.rig_problems() if skinned else []
    if problems:
        raise ValueError("this document's rig is not consistent: " + "; ".join(problems))
    names = dict(bone_names or {})
    place_t, place_s = np.zeros(3), 1.0
    if place is not None:
        place = np.asarray(place, dtype="f8")
        place_t, place_q, place_scale = m3.decompose(place)
        if not gltf.recomposes_without_shear(m3.compose(place_t, place_q, place_scale), place):
            raise ValueError("to_model's place must be a scale and a translation, with no shear")
        place_s = float(place_scale[0])
    posed = anim.pose(doc.animation(), at_time) if marked and at_time is not None else {}
    if marked:
        wanted = {obj.uid for obj in kept}
        for bone in marked:
            cursor: int | None = bone.uid
            while cursor is not None and cursor not in wanted and cursor in by_uid:
                wanted.add(cursor)
                cursor = by_uid[cursor].parent
        kept = [obj for obj in doc.objects if obj.uid in wanted]
    index_of_uid = {obj.uid: i for i, obj in enumerate(kept)}
    joint_of_bone = {bone.uid: j for j, bone in enumerate(marked)}
    skinned_meshes: list[tuple[Obj, bm.Mesh]] = []

    nodes: list[gltf.Node] = []
    meshes: list[list[gltf.Primitive]] = []
    for obj in kept:
        mesh_index: int | None = None
        if obj.visible:
            home = rig.nearest_bone(by_uid, obj.uid) if marked else None
            if (obj.skin is not None or home is not None) and bm.face_count(obj.mesh):
                skinned_meshes.append(
                    (obj, _skin_mesh(doc, by_uid, obj, home, joint_of_bone, place))
                )
            else:
                meshes.append(to_primitives(obj, doc.materials))
                mesh_index = len(meshes) - 1
        translation = np.array(obj.translation, dtype="f8", copy=True)
        rotation = np.array(obj.rotation, dtype="f8", copy=True)
        scale = np.array(obj.scale, dtype="f8", copy=True)
        override = posed.get(obj.uid, {})
        if "translation" in override:
            translation = np.array(override["translation"], dtype="f8")
        if "rotation" in override:
            rotation = np.array(override["rotation"], dtype="f8")
        if "scale" in override:
            scale = np.array(override["scale"], dtype="f8")
        if place is not None and (obj.parent is None or obj.parent not in index_of_uid):
            translation, rotation, scale = m3.decompose(
                place @ m3.compose(translation, rotation, scale)
            )
        nodes.append(
            gltf.Node(
                name=names.get(obj.uid, obj.name),
                translation=translation,
                rotation=rotation,
                scale=scale,
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
    skins: list[gltf.Skin] = []
    if marked:
        lift = np.eye(4) if place is None else place
        inverse = np.array([np.linalg.inv(lift @ doc._world_in(b.uid, by_uid)) for b in marked])
        skins.append(
            gltf.Skin(joints=[index_of_uid[bone.uid] for bone in marked], inverse_bind=inverse)
        )
        for obj, mesh in skinned_meshes:
            meshes.append(to_primitives(obj, doc.materials, mesh, skin_data=True))
            nodes.append(
                gltf.Node(name=names.get(obj.uid, obj.name), mesh=len(meshes) - 1, skin=0)
            )
            roots.append(len(nodes) - 1)
    animations: list[gltf.Animation] = []
    if marked and at_time is None:
        topmost = {obj.uid for obj in kept if obj.parent is None or obj.parent not in index_of_uid}
        for source in doc.rig_state.animations:
            channels: list[gltf.AnimChannel] = []
            for uid, props in source.channels.items():
                for prop, channel in props.items():
                    values = np.array(channel.values, dtype="f8", copy=True)
                    if place is not None and uid in topmost:
                        if prop == "translation":
                            values = values * place_s + place_t
                        elif prop == "scale":
                            values = values * place_s
                    channels.append(
                        gltf.AnimChannel(
                            index_of_uid[uid],
                            prop,
                            channel.times,
                            values,
                            "STEP" if source.interpolation == "step" else "LINEAR",
                        )
                    )
            animations.append(
                gltf.Animation(
                    source.name, channels, {} if source.loop else {"loop": False}
                )
            )
    return gltf.Model(nodes, roots, meshes, skins, animations=animations)


def _skin_mesh(
    doc: ClayDoc,
    by_uid: Mapping[int, Obj],
    obj: Obj,
    home: int | None,
    joint_of_bone: Mapping[int, int],
    place: np.ndarray | None = None,
) -> bm.Mesh:
    """*obj*'s mesh in world rest space with every vertex's joints and weights on it.

    The one place the two kinds of binding meet the one document skin: a weighted object's
    own joint indices are looked up through its skin's bone uids, a rigidly bound one gets
    ``(home, 0, 0, 0)`` at ``(1, 0, 0, 0)``.
    """
    matrix = doc._world_in(obj.uid, by_uid)
    world = bm.transformed(obj.mesh, matrix if place is None else place @ matrix)
    n = len(world.positions)
    if obj.skin is not None:
        table = np.array([joint_of_bone[uid] for uid in obj.skin.joints], dtype="i4")
        return replace(world, joints=table[world.joints], weights=world.weights)
    joints = np.zeros((n, 4), dtype="i4")
    weights = np.zeros((n, 4), dtype="f4")
    joints[:, 0] = joint_of_bone[home]
    weights[:, 0] = 1.0
    return replace(world, joints=joints, weights=weights)
