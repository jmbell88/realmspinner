"""The host's :class:`~.mason.refs.GeometrySource`: what a :class:`~.mason.refs.Ref`
resolves to on screen.

``mason/refs.py`` explains why the resolving has to happen here rather than in
the pure package: a :class:`~.mason.refs.PrimitiveRef` names a generator in
``clay.primitives`` (a sibling engine ``mason/`` may not import) and a
:class:`~.mason.refs.LibraryRef` names a job id (a path only the service layer
is allowed to turn into one), and this module is free to import both.

**Two ways a ref answers, one cache.** A primitive resolves synchronously --
running a generator function is cheap and touches no disk -- so
:meth:`AssetSource.primitives` can build and return it in the same call. A
library asset means opening a GLB, which is neither cheap nor safe on the
frame thread, so the first ``primitives()`` call for an unseen
:class:`~.mason.refs.LibraryRef` returns ``[]`` and *starts* a parse through
``ctx.submit``; the result is adopted later, off :func:`on_task_done`, on the
frame that the task lands. Both kinds end up in the same ``ref_key``-addressed
cache, because once resolved there is nothing left that distinguishes how a
flat ``list[gltf.Primitive]`` got built.
"""

from __future__ import annotations

import logging
import time
import weakref
from typing import Any

import numpy as np

from ....core.safeio import sizeguard
from ....kernels.geom3d import gltf
from .engine import refs as mason_refs

log = logging.getLogger(__name__)

#: The task key prefix every library-asset parse is submitted under, keyed
#: further by job id and artifact so two different assets loading at once do
#: not collide in ``TaskRunner``'s one-in-flight-per-key rule.
TASK_PREFIX = "mason-asset:"

#: **Not a measured constant.** ``dev/measurements/`` is where a number a
#: stored document is keyed on gets a dated write-up before it is fixed --
#: ``trellis_band``, ``SEAM_MAX``, the grade scale. This is not that: nothing
#: in a ``.rscn`` is keyed on ``CACHE_BYTES``, it only decides when an already-
#: placed asset's decoded geometry gets evicted and has to be re-parsed on
#: next use, which costs a frame and never correctness (see
#: ``AssetSource._evict``). That is exactly why it may be picked here rather
#: than measured first. 512 MB is chosen as roughly what a textured library
#: asset's decoded geometry, times sixty of them, costs -- sixty being about
#: the scene size a mode built to place props, lights and a terrain is for.
CACHE_BYTES = 512 * 1024 * 1024

#: How often :meth:`AssetSource.revalidate` actually stats the files. Not a
#: measured constant: a rebuilt asset appearing within a second is instant to a
#: person, and a stat per distinct ref per second is far below a frame's noise.
REVALIDATE_SECONDS = 1.0

# Copied from ``mason/objout.py``'s ``_normal_matrix`` rather than imported:
# that module is part of the pure ``mason`` package and this one is not, and
# the singular-matrix trap it guards against (``np.linalg.inv`` on a matrix
# scaled to zero on one axis can hand back inf/nan instead of raising) applies
# here identically -- baking a node's world transform into a library asset's
# normals is the same operation OBJ export does to a placed node's normals.
_SINGULAR_DET_EPS = 1e-9


def _normal_matrix(basis: np.ndarray) -> np.ndarray | None:
    """The inverse-transpose of a node's 3x3, or ``None`` when it has none.

    See ``mason/objout.py``'s ``_normal_matrix`` -- this is that function,
    copied rather than imported because this module sits outside the pure
    package it lives in.
    """
    det = float(np.linalg.det(basis))
    if not np.isfinite(det) or abs(det) < _SINGULAR_DET_EPS:
        return None
    try:
        inverse = np.linalg.inv(basis)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(inverse).all():
        return None
    return inverse.T


def _bake_model(model: gltf.Model) -> list[gltf.Primitive]:
    """Flatten a parsed GLB's node tree into one primitive list, baked to world.

    A library asset is one *thing* in a Mason scene -- its internal hierarchy
    does not belong in the outliner (see the module docstring's "two ways a
    ref answers") -- so every node's mesh is transformed into the asset's own
    root space once, here, rather than carried as a tree Mason would have to
    re-walk on every frame. ``POSITION`` takes the node's full world matrix;
    ``NORMAL`` takes the inverse-transpose of its 3x3, following
    ``_normal_matrix``'s guard against a singular one -- a node scaled to zero
    on some axis loses its normals rather than gaining NaNs.
    """
    baked: list[gltf.Primitive] = []
    for node in model.nodes:
        if node.mesh is None:
            continue
        basis = node.world[:3, :3]
        translation = node.world[:3, 3]
        normal_matrix = _normal_matrix(basis)
        for prim in model.meshes[node.mesh]:
            positions = np.asarray(prim.positions, dtype="f8") @ basis.T + translation
            normals = None
            if prim.normals is not None and normal_matrix is not None:
                transformed = np.asarray(prim.normals, dtype="f8") @ normal_matrix.T
                lengths = np.linalg.norm(transformed, axis=1)
                if lengths.size and bool((lengths > 1e-12).all()):
                    normals = (transformed / lengths[:, None]).astype("f4")
            baked.append(
                gltf.Primitive(
                    positions=positions.astype("f4"),
                    indices=prim.indices,
                    normals=normals,
                    uvs=prim.uvs,
                    joints=prim.joints,
                    weights=prim.weights,
                    material=prim.material,
                )
            )
    return baked


def _box_of(prims: list[gltf.Primitive]) -> tuple[np.ndarray, np.ndarray] | None:
    """The AABB of every primitive's own box, merged once. See
    :meth:`AssetSource.box`'s docstring for why this must not walk vertices."""
    boxes = [p.box() for p in prims]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    los = np.stack([b[0] for b in boxes])
    his = np.stack([b[1] for b in boxes])
    return los.min(axis=0), his.max(axis=0)


def _nbytes(prims: list[gltf.Primitive]) -> int:
    """Bytes actually held by ``prims`` -- the cache's own accounting unit."""
    total = 0
    for prim in prims:
        for arr in (
            prim.positions,
            prim.indices,
            prim.normals,
            prim.uvs,
            prim.joints,
            prim.weights,
        ):
            if arr is not None:
                total += np.asarray(arr).nbytes
    return total


class _Entry:
    """One resolved ref's geometry, its precomputed box, and its byte cost."""

    __slots__ = ("prims", "box", "nbytes")

    def __init__(
        self,
        prims: list[gltf.Primitive],
        box: tuple[np.ndarray, np.ndarray] | None,
        nbytes: int,
    ) -> None:
        self.prims = prims
        self.box = box
        self.nbytes = nbytes


class AssetSource:
    """The one :class:`~.mason.refs.GeometrySource` a Mason session holds.

    Built through :func:`ensure`, never directly -- see its docstring.
    """

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self._cache: dict[tuple[Any, ...], _Entry] = {}
        # LRU order, oldest first, touched on every resolved lookup -- not on
        # a lookup that is still loading, since there is nothing yet to say
        # was "used".
        self._order: list[tuple[Any, ...]] = []
        self._total_bytes = 0
        # Refs that resolved to nothing *permanently*: an unknown generator, a
        # builder that raised, a job dir with no such artifact, a parse that
        # raised. Deliberately excludes a LibraryRef that is merely still
        # loading -- see the module docstring's "two ways a ref answers" and
        # ``primitives()`` below.
        self.missing: set[tuple[Any, ...]] = set()
        # Keys with a parse already in flight, so a second primitives() call
        # for the same still-loading ref does not try to resubmit -- though
        # ``ctx.submit``'s own key dedup would refuse the second submission
        # anyway, this avoids paying for the job_dir lookup and the closure
        # build on every frame a slow parse is pending.
        self._pending: set[tuple[Any, ...]] = set()
        self._rev = 0
        # Keys the open document places, exempt from :meth:`_evict` -- see
        # :meth:`pin_document`. ``_pin_for`` is (weak doc, rev) it was last
        # computed for, so an unchanged document costs nothing per frame.
        self._pinned: set[tuple[Any, ...]] = set()
        self._pin_for: tuple[weakref.ref[Any], int] | None = None
        # The 2026-10-03 audit's mason-28: a library ref's file as it was when
        # its parse started (``(mtime_ns, size)``, or ``None`` for "no such
        # file"), so :meth:`revalidate` can tell an artifact rebuilt in place
        # (``optimize_job`` rewrites ``model.glb``) or put back from one that
        # is unchanged. Recorded at the *start* of a parse, so a file replaced
        # mid-parse still differs afterwards and is parsed again.
        self._sigs: dict[tuple[Any, ...], tuple[int, int] | None] = {}
        self._paths: dict[tuple[Any, ...], Any] = {}
        self._last_revalidate = 0.0
        # Keys :meth:`revalidate` dropped, for the viewport to drop its GPU
        # upload of too (it holds the geometry by key, not through this cache).
        self._invalidated: set[tuple[Any, ...]] = set()

    @property
    def rev(self) -> int:
        return self._rev

    # -- the GeometrySource contract -----------------------------------

    def primitives(self, ref: mason_refs.Ref) -> list[gltf.Primitive]:
        key = mason_refs.ref_key(ref)
        entry = self._cache.get(key)
        if entry is not None:
            self._touch(key)
            return entry.prims
        if key in self.missing or key in self._pending:
            return []
        if isinstance(ref, mason_refs.PrimitiveRef):
            self._resolve_primitive(ref, key)
            entry = self._cache.get(key)
            return entry.prims if entry is not None else []
        if isinstance(ref, mason_refs.LibraryRef):
            self._start_library(ref, key)
            return []
        return []  # pragma: no cover - the Ref union is closed

    def box(self, ref: mason_refs.Ref) -> tuple[np.ndarray, np.ndarray] | None:
        entry = self._cache.get(mason_refs.ref_key(ref))
        return None if entry is None else entry.box

    # -- primitives: synchronous ----------------------------------------

    def _resolve_primitive(self, ref: mason_refs.PrimitiveRef, key: tuple[Any, ...]) -> None:
        """Build a primitive ref's mesh now, on whatever thread called us.

        Never raises: an unknown generator key or a builder that rejects its
        own params (a negative radius, a segment count clamped to nonsense)
        records ``ref`` as missing instead of propagating into a frame -- the
        one thing every task-thread and frame-thread caller of this module
        must be able to rely on.
        """
        from ....kernels.mesh import document as bd
        from ....kernels.mesh.document import Obj
        from ....kernels.mesh.primitives import GENERATORS

        entry = GENERATORS.get(ref.generator)
        if entry is None:
            self.missing.add(key)
            return
        defaults, make = entry
        try:
            mesh = make(**{**defaults, **ref.as_params()})
            # ``to_primitives`` groups by material index and reads
            # ``obj.material`` only as the *default* for faces with no
            # explicit one -- a bare generator mesh has none, so 0 (the
            # fallback material) is fine. Reusing ``to_primitives`` rather
            # than writing a second Mesh -> Primitive conversion is the point
            # (see the module docstring and ``studio/modes/clay/ui/_view_cache.py``'s ``_build``).
            obj = Obj(uid=0, name="", mesh=mesh)
            prims = bd.to_primitives(obj, ())
        except Exception:
            log.exception("mason: primitive generator %r rejected its params", ref.generator)
            self.missing.add(key)
            return
        self._store(key, prims)

    # -- library assets: asynchronous ------------------------------------

    def _start_library(self, ref: mason_refs.LibraryRef, key: tuple[Any, ...]) -> None:
        job_id, artifact = ref.job_id, ref.artifact
        # The 2026-10-03 audit's mason-19: both strings come verbatim from a
        # ``.rscn``, and ``job_dir(job_id) / artifact`` is a bare join -- a
        # shared scene naming ``job_id="..\\..\\outside"`` made the app read and
        # bake any GLB-shaped file on disk. ``check_job_id`` is the guard every
        # path built from a caller-supplied id needs (``mason_source_path``
        # already calls it); an ``artifact`` may only be a bare file name. A ref
        # that fails either is permanently missing, never an exception on the
        # frame that asked (this runs from ``primitives()`` every frame).
        from ....service.errors import ServiceError
        from ....service.validation import check_job_id

        try:
            check_job_id(job_id)
            if (
                not artifact
                or artifact in (".", "..")
                or any(c in artifact for c in ("/", "\\", ":"))
            ):
                raise ValueError(artifact)
        except (ServiceError, ValueError):
            self.missing.add(key)
            self._rev += 1
            return
        task_key = f"{TASK_PREFIX}{job_id}:{artifact}"
        path = self.ctx.svc.config.job_dir(job_id) / artifact
        self._paths[key] = path
        self._sigs[key] = _signature(path)

        def run() -> list[gltf.Primitive]:
            # The 2026-09-15 audit's mason-02: the GLB decode already ran here,
            # on the task thread, but ``_bake_model`` -- a per-vertex matrix
            # and normal transform, up to a 100 MB GLB -- used to run back in
            # ``on_task``, on the frame thread that adopts the result. Baked
            # here instead, so a big asset's bake costs a task-pool thread,
            # never a dropped frame; a bake that raises is caught the same
            # way a bad decode always was, by the task runner turning it into
            # ``error`` for :meth:`on_task` rather than a frame-thread crash.
            from ....service.validation import MAX_MESH_BYTES

            # shell-07 (the 2026-09-18 audit): the bounded read closes the
            # stat/read_bytes race the split call used to leave open.
            data = sizeguard.read_bytes_within_ceiling(path, MAX_MESH_BYTES)
            model = gltf.load(data)
            return _bake_model(model)

        if self.ctx.submit(task_key, run, tag=key):
            self._pending.add(key)
        # If submit refused (another primitives() call already started the
        # same key this session), the ref is already pending -- nothing new
        # to do; the earlier caller's parse will land for both.

    def on_task(self, key: str, tag: Any, result: Any, error: BaseException | None) -> bool:
        """Handle one ``TaskRunner`` completion. -> whether it was ours.

        See :func:`on_task_done`, the module-level entry point that unpacks a
        ``tasks.Done`` and calls this.
        """
        if not key.startswith(TASK_PREFIX):
            return False
        self._pending.discard(tag)
        if error is not None or result is None:
            self.missing.add(tag)
            # The picture changed even though nothing resolved: a ref that was
            # a loading placeholder a moment ago is now a permanently missing
            # one, and the viewport's redraw key must see that -- see the
            # module docstring's ``rev`` paragraph.
            self._rev += 1
            return True
        # ``result`` is already baked -- see the 2026-09-15 audit's mason-02
        # comment on ``_start_library.run``: the bake happened on the task
        # thread, not here, so there is nothing left to do on this one but
        # adopt it.
        self._store(tag, result)
        self._rev += 1
        return True

    # -- staleness ---------------------------------------------------------

    def revalidate(self, force: bool = False) -> None:
        """Drop every library ref whose file changed on disk since it was read.

        The 2026-10-03 audit's mason-28: the cache was keyed on ``(job_id,
        artifact)`` alone, so a model re-optimised or remeshed in place kept
        drawing and exporting the old geometry, and a ref that had failed once
        stayed in ``missing`` -- so neither a re-run nor putting the asset back
        took effect until restart, against the chapter's "re-running an asset
        updates every scene using it". Polled by the viewport every frame but
        throttled to once a second: one ``stat`` per distinct resolved or
        missing ref, never per placement.
        """
        now = time.monotonic()
        if not force and now - self._last_revalidate < REVALIDATE_SECONDS:
            return
        self._last_revalidate = now
        for key, path in list(self._paths.items()):
            if key in self._pending:
                continue  # in flight: compared once it lands, at the next poll
            if key not in self._cache and key not in self.missing:
                continue  # evicted: re-read, and re-recorded, on the next request
            if _signature(path) == self._sigs.get(key):
                continue
            self._drop(key)
            self.missing.discard(key)
            self._sigs.pop(key, None)
            self._paths.pop(key, None)
            self._invalidated.add(key)
            self._rev += 1

    def take_invalidated(self) -> set[tuple[Any, ...]]:
        """The keys :meth:`revalidate` dropped since the last call."""
        taken, self._invalidated = self._invalidated, set()
        return taken

    def _drop(self, key: tuple[Any, ...]) -> None:
        entry = self._cache.pop(key, None)
        if entry is not None:
            self._total_bytes -= entry.nbytes
        if key in self._order:
            self._order.remove(key)

    # -- cache bookkeeping -------------------------------------------------

    def _store(self, key: tuple[Any, ...], prims: list[gltf.Primitive]) -> None:
        self.missing.discard(key)
        old = self._cache.get(key)
        if old is not None:
            self._total_bytes -= old.nbytes
        nbytes = _nbytes(prims)
        self._cache[key] = _Entry(prims, _box_of(prims), nbytes)
        self._total_bytes += nbytes
        self._touch(key)
        self._evict()

    def _touch(self, key: tuple[Any, ...]) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def pin_document(self, doc: Any) -> None:
        """Exempt every ref ``doc`` places from eviction (and evict what it no
        longer places).

        The 2026-10-03 audit's mason-23: eviction was purely LRU and pressure,
        yet exports, picking and framing read *through* this cache -- so when
        the placed assets' decoded geometry together exceeded
        :data:`CACHE_BYTES` there was no state in which all of them were
        resident, and ``export.unresolved`` stayed non-empty for ever ("not
        finished loading") as each retry re-parsed the evicted refs and evicted
        others. A scene that large is a person's legitimate work, so what it
        places stays resident even past the budget; the budget bounds the
        *unplaced* remainder. Called by the viewport every frame and by the
        three exporters (cheap: it recomputes only when the document's ``rev``
        moves).
        """
        from .engine import scene as mason_scene

        stamp = self._pin_for
        if stamp is not None and stamp[0]() is doc and stamp[1] == doc.rev:
            return
        try:
            placed = mason_scene.resolve(doc, include_hidden=True)
        except ValueError:
            placed = []
        self._pinned = {mason_refs.ref_key(p.ref) for p in placed if p.ref is not None}
        self._pin_for = (weakref.ref(doc), doc.rev)
        self._evict()

    def _evict(self) -> None:
        """Drop the least-recently-used entries above :data:`CACHE_BYTES`,
        except the refs :meth:`pin_document` says the document places.

        An unpinned ref evicted while still on screen (nothing pinned the
        document yet) simply loses its cache entry: the next ``primitives()``
        call for it re-resolves (a re-parse for a library asset, a rebuild for
        a primitive), which costs a frame and nothing else.
        """
        while self._total_bytes > CACHE_BYTES and len(self._order) > 1:
            # The newest entry is never the victim (the one just stored).
            victim = next((k for k in self._order[:-1] if k not in self._pinned), None)
            if victim is None:
                break
            self._order.remove(victim)
            entry = self._cache.pop(victim, None)
            if entry is not None:
                self._total_bytes -= entry.nbytes

    # -- teardown -----------------------------------------------------------

    def release(self) -> None:
        """Drop every cached array. Memory only -- nothing here owns a GL
        object (the GPU cache is the viewport's own), so there is no texture
        or buffer to release. If a later change gives this cache anything
        GPU-side, this docstring is the reminder that this method does not
        cover it yet."""
        self._cache.clear()
        self._order.clear()
        self._pending.clear()
        self._pinned.clear()
        self._pin_for = None
        self._total_bytes = 0
        self._sigs.clear()
        self._paths.clear()
        self._invalidated.clear()


def _signature(path: Any) -> tuple[int, int] | None:
    """A file's ``(mtime_ns, size)``, or ``None`` when it cannot be statted."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def pin(source: Any, doc: Any) -> None:
    """:meth:`AssetSource.pin_document`, for a source that has one -- the
    stand-in a test installs on ``ctx.mason_assets`` does not, and needs none."""
    pin_document = getattr(source, "pin_document", None)
    if pin_document is not None:
        pin_document(doc)


def ensure(ctx: Any) -> AssetSource:
    """The session's one :class:`AssetSource`, built on first use.

    ``clay_mode.ensure``'s lazy shape, for its reason: a session that never
    opens Mason should not pay for this cache or its parse machinery.
    """
    source = getattr(ctx, "mason_assets", None)
    if source is None:
        source = AssetSource(ctx)
        ctx.mason_assets = source
    return source


def on_task_done(ctx: Any, done: Any) -> bool:
    """Route one finished ``TaskRunner`` task here if it is ours. -> claimed?

    Called from the frame thread's task-collection loop the way every other
    ``*_mode``'s task handler is (see ``main.py``'s ``_collect_tasks``). Only
    ever true for a ``mason-asset:`` key; every other key is left untouched.
    """
    if not str(done.key).startswith(TASK_PREFIX):
        return False
    source = ensure(ctx)
    return source.on_task(done.key, done.tag, done.result, done.error)
