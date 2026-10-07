"""The Clay viewport's GPU cache: one upload per object, keyed on identity.

Split out of :mod:`~realmspinner.studio.modes.clay.ui.view` as pure code motion. The soundness
argument is the viewport's own and is stated in its module docstring: an entry
is keyed on ``(id(mesh), materials, obj.material)``, and that identity works
precisely because ``Mesh`` is frozen and every op on it is ``Mesh -> Mesh``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .....kernels.geom3d import gltf
from .....kernels.mesh import document as bd
from ....viewer import scene as scenelib

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .view import ClayView


class _Entry:
    """One object's GPU state, and the key that says whether it is still valid."""

    __slots__ = ("key", "gpu", "model", "mesh")

    def __init__(self, key: Any, gpu: Any, model: Any, mesh: Any) -> None:
        self.key = key
        self.gpu = gpu
        self.model = model
        # The Mesh whose id() the key carries, held so that id cannot be
        # recycled while the entry lives. Nothing in ``model`` keeps the Mesh
        # *instance* alive -- ``_submesh`` builds fresh arrays on the way to
        # the GPU -- so a freed mesh's address coming back on different
        # geometry would make ``entry.key == key`` true of a stale upload:
        # the viewport drawing the old shape forever, the exact failure the
        # identity key's soundness argument forbids. The reference dies with
        # the entry at the next key mismatch, so nothing leaks.
        self.mesh = mesh


def _materials_key(doc: Any) -> tuple[int, ...]:
    """Identity of every palette entry, in order.

    Identity rather than value, matching ``to_model`` and ``GpuMaterial``:
    ``ClayDoc.set_material`` replaces the entry object, so an edit changes the
    key, and a palette that has not been touched keeps it. Comparing values
    would mean hashing five floats per entry per frame to learn the same thing.
    """
    return tuple(id(m) for m in doc.materials)


def _object_key(mesh: Any, materials: tuple[int, ...], material: int) -> tuple[Any, ...]:
    # The transform is deliberately *not* in the key: it is a uniform, not a
    # buffer, so moving an object must not rebuild it.
    return (id(mesh), materials, material)


class CacheOps:
    """``ClayView``'s GPU cache. See the module docstring."""

    # -- the GPU cache -----------------------------------------------------

    def sync(self: ClayView, doc: Any) -> None:
        """Bring the GPU up to date with the document, rebuilding only what changed."""
        materials = _materials_key(doc)
        live: set[int] = set()
        for obj in doc.objects:
            if not obj.visible:
                continue
            live.add(obj.uid)
            mesh = obj.mesh
            key = _object_key(mesh, materials, obj.material)
            entry = self._cache.get(obj.uid)
            if entry is not None and entry.key == key:
                continue
            if entry is not None:
                entry.gpu.release()
                # The hovered element is an index into the *old* mesh's
                # vertices, edges or faces. A keyboard op that shrank any of
                # those would have next frame's overlay read
                # ``edge_verts[hover]`` past the end -- an IndexError out of
                # ``draw()`` on the frame loop -- or send a stale vertex index
                # to the GPU. This is the one place every mesh *replacement*
                # passes through before anything draws.
                #
                # Only a replacement, which is what ``entry is not None`` says.
                # Unconditionally, it also cleared the hover on an object's
                # *first* build -- where there is no old mesh for the index to
                # be stale against -- and ``sync`` runs immediately before
                # ``_element_overlays`` in ``draw``, so the hovered element was
                # never drawn on the frame an object was built. That is every
                # first frame, and every frame after a mesh edit: the highlight
                # dropped out until the cursor moved again.
                if self.hover_element is not None and self.hover_element[0] == obj.uid:
                    self.hover_element = None
            self._cache[obj.uid] = self._build(obj, doc, key, mesh)
            self.rebuilds += 1
        for uid in [u for u in self._cache if u not in live]:
            # A hidden object's buffers go too: ``visible=False`` means it does
            # not render, does not export and is not picked, and holding its
            # upload would make it the one of the three that is only half true.
            self._cache.pop(uid).gpu.release()
        # The projection cache rides the same lifetime: a deleted object's
        # entry pins its mesh and its projected arrays, and nothing else would
        # ever evict it.
        for uid in [u for u in self._screens if u not in live]:
            self._screens.pop(uid)

    def _build(self: ClayView, obj: Any, doc: Any, key: Any, mesh: Any) -> _Entry:
        prims = bd.to_primitives(obj, doc.materials)
        node = gltf.Node(name=obj.name, mesh=0)
        model = gltf.Model([node], [0], [prims], [])
        # *mesh* is the entry's pin: see ``_Entry.mesh``'s own comment for why
        # an id in the key must be held alive for as long as the entry claims
        # to match it.
        # The 2026-10-07 audit's clay-28: every object's model shares the view's
        # one texture cache, so N objects over one palette picture are one GL
        # texture and a preview frame's fresh model finds it already uploaded.
        # ``getattr`` because a stand-in ``self`` (the wiring test's) carries no
        # cache, and then the model keeps its own, as it always did.
        textures = getattr(self, "_textures", None)
        if textures is None:
            gpu = scenelib.GpuModel(self.ctx, model)
        else:
            gpu = scenelib.GpuModel(self.ctx, model, texture_cache=textures)
        return _Entry(key, gpu, model, mesh)

    def clear(self: ClayView) -> None:
        for entry in self._cache.values():
            entry.gpu.release()
        self._cache.clear()
        self._screens.clear()
        self.hover_object = None
        self._hover_pick_at = None
