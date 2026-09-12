"""Mason's controller: opening, saving, exporting, guarding and keys.

Stage E gives Mason a document, so this module is what ``clay_mode.py`` is to
Clay -- everything here is *about* scenes rather than geometry. The engine
under ``mason/`` has no idea a job or a task thread exists; this is the layer
that knows about both, and the panes draw against it.

The rule that shapes this file is Clay's own: **no file dialog and no encode
ever runs on the frame thread.** A native picker is modal to the OS and blocks
until dismissed, and a ``.wscn`` of any size is a zip to build. Both go through
``ctx.submit`` and come back through :func:`on_task_done`, which is why saving
is a *state* (``MasonTab.saving``) rather than a function call that returns.

``mason_io.py`` already states the split for this format's own reasons:
``serialize.snapshot`` is the cheap frame-thread half (it reads ``doc.view``
itself -- there is no ``view=`` kwarg, unlike Clay's ``.wblk`` snapshot, since
a ``.wscn``'s camera lives inside ``scene.json`` rather than a second small
member); ``snapshot_bytes`` is the expensive half and runs inside ``run()``.
:func:`camera_of` therefore writes the live camera onto ``doc.view`` *before*
either save path takes its snapshot -- there is nowhere else for the camera to
land, since ``snapshot`` never takes one as an argument.

**A failed save must clear that state**, Clay's own rule restated: ``saving``
gates every control that changes the document, so without
:func:`on_task_failed` one failed write leaves the tab permanently read-only.

Every task key carries the ``mason-`` prefix, because the app claims results
by prefix: a key without one is a result delivered nowhere. ``mason-asset:``
keys are :mod:`.mason_assets`'s alone -- :func:`on_task_done` gives that module
first refusal before doing anything of its own with a ``mason-`` key, since a
library asset's background parse is not a document task at all.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from . import dialogs, docmodes, journal, mason_assets, mason_io, mason_state, sizeguard
from ._view_frame import AXIS_VIEW_KEYS, axis_view_key
from .mason_state import MasonState, MasonTab

log = logging.getLogger(__name__)


def _path_key(path: Path) -> str:
    """A short, stable id for a path -- ``clay_mode``'s own function,
    restated: ``hash(str(path))`` is salted per process and can collide within
    one session; sha1 never does."""
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def ensure(ctx: Any) -> MasonState:
    """The mode's state, built on first use -- ``clay_mode.ensure``'s reason:
    a session that never opens Mason should not pay for it."""
    state = ctx.state.mason
    if state is None:
        state = MasonState()
        ctx.state.mason = state
    return state


# The three recents wrappers every document mode carries.
remember_path, forget_path, recent_paths = docmodes.recents_for("mason")


def persist(ctx: Any) -> None:
    """A no-op -- ``clay_mode.persist``'s reason: the recent list lives in
    :mod:`.recents`, which persists itself on every write. Kept because it is
    called from a dozen sites after every open and save, and turning each of
    those into "call this only if the mode still has settings" is how one of
    them comes to skip a write that mattered later."""


def active(ctx: Any) -> MasonTab | None:
    state = ctx.state.mason
    return state.active if state is not None else None


def _enter_mason(ctx: Any) -> None:
    """Adoption switches modes through ``state.set_mode``, never by
    assignment -- ``clay_mode._enter_clay``'s reason, restated here."""
    from .state import set_mode

    set_mode(ctx.state, "mason")


# --- opening ------------------------------------------------------------------


def adopt(
    ctx: Any,
    doc: Any,
    *,
    path: Path | None = None,
    title: str | None = None,
    view: dict[str, Any] | None = None,
) -> MasonTab:
    state = ensure(ctx)
    tab = MasonTab(
        doc=doc,
        title=title or mason_state.title_for(path),
        path=path,
        saved_head=doc.history.head,
    )
    if view:
        tab.view.yaw = view["yaw"]
        tab.view.pitch = view["pitch"]
        tab.view.distance = view["distance"]
        tab.view.target = view["target"]
        # Already framed: an auto-fit over the top would throw away the
        # answer just read off disk, ``clay_mode.adopt``'s own reason.
        tab.view.fitted = True
    state.add(tab)
    remember_path(ctx, path)
    persist(ctx)
    return tab


def new_document(ctx: Any) -> MasonTab:
    from .mason import document as md

    return adopt(ctx, md.MasonDoc(), title="Untitled")


def _within_ceiling(path: Path) -> Path:
    return sizeguard.within_ceiling(Path(path), mason_io.MAX_WSCN_BYTES)


def ask_open(ctx: Any) -> None:
    """The picker, on a task thread, then the decode on the same one."""
    ensure(ctx)

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Open Mason scene", mason_io.WSCN_FILTER)
        return None if path is None else mason_io.load(Path(path))

    ctx.submit("mason-open", run)


def open_path(ctx: Any, path: Path) -> None:
    state = ensure(ctx)
    path = Path(path)
    existing = state.find_path(path)
    if existing is not None:
        # Focus rather than fork: two tabs over one path would race on save.
        state.activate(existing.uid)
        return
    ctx.submit(f"mason-open:{_path_key(path)}", mason_io.load, path)


# --- placing ------------------------------------------------------------------


def place_ref(ctx: Any, ref: Any, *, name: str = "") -> int | None:
    """Place a resolved reference (a primitive or a library asset) as a new
    :class:`~.mason.nodes.MeshNode`, selected."""
    tab = active(ctx)
    if tab is None or tab.saving:
        return None
    from .mason import nodes as nd

    node = nd.MeshNode(uid=nd.new_uid(), name=name or "Mesh", ref=ref)
    tab.doc.add_node(node)
    tab.doc.select([node.uid])
    return node.uid


def place_primitive(ctx: Any, generator: str) -> int | None:
    from .mason import refs as mason_refs

    ref = mason_refs.primitive_ref(generator, {})
    return place_ref(ctx, ref, name=generator.replace("_", " ").title())


def place_light(ctx: Any, kind: str) -> int | None:
    tab = active(ctx)
    if tab is None or tab.saving:
        return None
    from .mason import nodes as nd

    node = nd.LightNode(uid=nd.new_uid(), name=kind.title(), kind=kind)
    tab.doc.add_node(node)
    tab.doc.select([node.uid])
    return node.uid


def place_camera(ctx: Any) -> int | None:
    tab = active(ctx)
    if tab is None or tab.saving:
        return None
    from .mason import nodes as nd

    node = nd.CameraNode(uid=nd.new_uid(), name="Camera")
    tab.doc.add_node(node)
    tab.doc.select([node.uid])
    return node.uid


def import_glb_path(ctx: Any, path: Path) -> None:
    """A bare ``.glb`` dropped on Mason: import it into the library, then
    place the resulting job as a mesh node.

    A :class:`~.mason.refs.LibraryRef` names a job id, never a path -- see
    that module's own docstring -- so a file with no job behind it has to get
    one first. The import (a task-thread read plus ``svc_jobs.import_mesh``)
    and the placement both go through :func:`on_task_done`'s
    ``mason-import`` arm, mirroring ``clay_mode.import_glb_path``'s split.
    """
    path = Path(path)

    def run() -> dict[str, Any]:
        from ..service import jobs as svc_jobs
        from ..service.validation import MAX_MESH_BYTES

        data = sizeguard.within_ceiling(path, MAX_MESH_BYTES).read_bytes()
        result = svc_jobs.import_mesh(ctx.svc, data, name=path.stem, prompt=path.stem)
        return {"job_id": result["id"], "name": path.stem}

    ctx.submit(f"mason-import:{_path_key(path)}", run)


def place_job(ctx: Any, job: Any) -> int | None:
    """A library mesh row -> a placed :class:`~.mason.nodes.MeshNode`."""
    from .mason import refs as mason_refs

    job_id = job["id"] if isinstance(job, dict) else str(job)
    name = (job.get("name") if isinstance(job, dict) else "") or ""
    ref = mason_refs.LibraryRef(job_id=job_id, name=name)
    return place_ref(ctx, ref, name=name or "Asset")


# --- editing --------------------------------------------------------------------


def group_selected(ctx: Any) -> None:
    """Wrap the current selection in one new :class:`~.mason.nodes.GroupNode`,
    as a single undo step.

    Every selected node keeps its own parent's *transform space* by being
    reparented as a sibling of itself first -- ``move_node`` only reorders and
    reparents, it never re-expresses a local transform relative to its new
    parent -- so a selection under different parents ends up moved to wherever
    the group node landed, exactly as a modelling package's "group" gesture
    reads on screen: the arrangement is preserved by construction because the
    new group starts at the identity, not by any transform math here.
    """
    tab = active(ctx)
    if tab is None or tab.saving:
        return
    doc = tab.doc
    uids = sorted(doc.selection)
    if not uids:
        return
    from .mason import nodes as nd

    mark = doc.mark()
    group = nd.GroupNode(uid=nd.new_uid(), name="Group")
    # The first selected node's own parent, so the group lands beside what it
    # is about to hold rather than always at the root.
    parent_uid = doc.parent_uid_of(uids[0])
    doc.add_node(group, parent_uid=parent_uid)
    for uid in uids:
        doc.move_node(uid, len(group.children), parent_uid=group.uid)
    doc.collapse_since(mark)
    doc.select([group.uid])


def duplicate_selected(ctx: Any) -> None:
    """Deep-copy every selected node, as siblings of the originals, as one
    step."""
    tab = active(ctx)
    if tab is None or tab.saving:
        return
    doc = tab.doc
    uids = sorted(doc.selection)
    if not uids:
        return
    from .mason import nodes as nd

    copies: list[Any] = []
    parents: dict[int, int | None] = {}
    for uid in uids:
        node = doc.node(uid)
        if node is None:
            continue
        parents[uid] = doc.parent_uid_of(uid)
        copies.append((parents[uid], nd.copy_subtree(node, fresh_uids=True)))
    if not copies:
        return
    mark = doc.mark()
    new_uids: list[int] = []
    for parent_uid, copy in copies:
        doc.add_node(copy, parent_uid=parent_uid)
        new_uids.append(copy.uid)
    doc.collapse_since(mark)
    doc.select(new_uids)


def delete_selected(ctx: Any) -> None:
    """Remove every selected node -- and its subtree -- as one step."""
    tab = active(ctx)
    if tab is None or tab.saving:
        return
    doc = tab.doc
    uids = sorted(doc.selection)
    if not uids:
        return
    mark = doc.mark()
    for uid in uids:
        if doc.node(uid) is not None:
            doc.remove_node(uid)
    doc.collapse_since(mark)
    doc.select([])


# --- camera -------------------------------------------------------------------


def camera_of(ctx: Any, tab: MasonTab) -> Any:
    """The tab's stored camera, refreshed from the live viewport first, and
    written onto ``doc.view`` -- the one place a ``.wscn``'s camera lives.

    ``serialize.snapshot`` takes no ``view=`` argument (see the module
    docstring): it reads ``doc.view`` itself, so this is the write every save
    path must make before taking a snapshot.
    """
    view = getattr(ctx, "mason_view", None)
    if view is not None and getattr(view, "camera", None) is not None:
        tab.view.read_from(view.camera)
    from .mason import serialize

    tab.doc.view = {
        **{name: getattr(tab.view, name) for name in serialize.VIEW_FIELDS},
        "target": tuple(tab.view.target),
    }
    return tab.view


def remember_camera(ctx: Any, tab: MasonTab | None) -> None:
    """Snapshot the live camera onto a tab that is being switched away from."""
    if tab is not None:
        camera_of(ctx, tab)


def apply_camera(ctx: Any, tab: MasonTab) -> None:
    """Put a tab's camera back on the viewport, or frame it if it has none."""
    view = getattr(ctx, "mason_view", None)
    if view is None or getattr(view, "camera", None) is None:
        return
    if tab.view.fitted:
        tab.view.write_to(view.camera)
        return
    source = mason_assets.ensure(ctx)
    view.frame_selection(tab.doc, source)
    tab.view.read_from(view.camera)
    tab.view.fitted = True


# --- saving ---------------------------------------------------------------------

_start = docmodes.start_save


def save_to(ctx: Any, tab: MasonTab, path: Path) -> None:
    """Write the document to a known path. See :func:`camera_of` and the
    module docstring for the read/encode ordering this follows."""
    from .mason import serialize

    path = Path(path)
    doc = tab.doc
    rev = doc.history.head
    camera_of(ctx, tab)
    snap = serialize.snapshot(doc)

    def run() -> dict[str, Any]:
        mason_io.write_wscn(path, serialize.snapshot_bytes(snap))
        return {"rev": rev, "path": str(path), "retitle": True}

    _start(ctx, tab, f"mason-save:{tab.uid}", run)


def save(ctx: Any, tab: MasonTab | None = None) -> None:
    tab = tab or active(ctx)
    docmodes.save(
        tab, save_as=lambda: save_as(ctx, tab), save_to=lambda: save_to(ctx, tab, tab.path)
    )


def save_as(ctx: Any, tab: MasonTab | None = None) -> None:
    """The picker and the encode on one task thread -- ``clay_mode.save_as``'s
    reason: the *snapshot* is cheap enough to take before an unbounded modal
    dialog, and the encode must not run before or during it."""
    from .mason import serialize

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, title = tab.doc, tab.title
    rev = doc.history.head
    camera_of(ctx, tab)
    snap = serialize.snapshot(doc)

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file(
            "Save Mason scene",
            f"{title}{mason_state.WSCN_SUFFIX}",
            mason_io.WSCN_FILTER,
        )
        if path is None:
            return None
        path = Path(path).with_suffix(mason_state.WSCN_SUFFIX)
        mason_io.write_wscn(path, serialize.snapshot_bytes(snap))
        return {"rev": rev, "path": str(path), "retitle": True}

    _start(ctx, tab, f"mason-saveas:{tab.uid}", run)


# --- export -----------------------------------------------------------------------


def export_glb(ctx: Any, tab: MasonTab | None = None) -> None:
    """Write ``scene.glb`` plus its ``scene.json`` sidecar, an engine can read
    directly."""
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, title = tab.doc, tab.title

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file(
            "Export Mason scene as GLB", f"{title}.glb", mason_io.GLB_FILTER
        )
        if path is None:
            return None
        path = Path(path).with_suffix(".glb")
        source = mason_assets.ensure(ctx)
        files = mason_io.glb_bundle(doc, source)
        mason_io.write_files(files, path, primary="scene.glb")
        return {"exported": True, "path": str(path)}

    _start(ctx, tab, f"mason-exportglb:{tab.uid}", run)


def export_obj(ctx: Any, tab: MasonTab | None = None) -> None:
    """Write a merged ``scene.obj``, surfacing ``skipped`` -- a loss that is
    stated is not the same as a loss that is invisible."""
    from .mason import objout

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, title = tab.doc, tab.title

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file("Export Mason scene as OBJ", f"{title}.obj", mason_io.OBJ_FILTER)
        if path is None:
            return None
        path = Path(path).with_suffix(".obj")
        source = mason_assets.ensure(ctx)
        files, skipped, vertices, triangles = mason_io.obj_bundle(doc, source)
        mason_io.write_files(files, path, primary=objout.OBJ)
        return {
            "exported": True,
            "path": str(path),
            "skipped": skipped,
            "vertices": vertices,
            "triangles": triangles,
        }

    _start(ctx, tab, f"mason-exportobj:{tab.uid}", run)


# --- scene stats ------------------------------------------------------------------


def scene_stats(ctx: Any, tab: Any) -> dict[str, Any]:
    """How big a scene is, against the engine's own constant -- never a number
    typed in a pane."""
    from .mason import scene as msc

    if tab is None:
        return {"placed": 0, "warn": False, "threshold": msc.PLACED_WARN_THRESHOLD, "missing": 0}
    doc = tab.doc
    try:
        placed = len(msc.resolve(doc, include_hidden=True))
    except ValueError:
        placed = 0
    return {
        "placed": placed,
        "warn": placed >= msc.PLACED_WARN_THRESHOLD,
        "threshold": msc.PLACED_WARN_THRESHOLD,
        "missing": len(doc.missing_refs()),
    }


# --- task results -----------------------------------------------------------------


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from the app for every ``mason-`` key.

    ``mason_assets`` gets first refusal: a ``mason-asset:`` key is its
    background parse, never a document task, and it claims that prefix
    itself (see its own ``on_task_done``).
    """
    if mason_assets.on_task_done(ctx, done):
        return

    state = ensure(ctx)
    key, result = done.key, done.result
    name = key.split(":", 1)[0]

    if name == "mason-open":
        if isinstance(result, dict):
            adopt(
                ctx,
                result["doc"],
                path=Path(result["path"]),
                title=result.get("title"),
                view=result.get("view"),
            )
            _enter_mason(ctx)
        return

    if name == "mason-import":
        if isinstance(result, dict):
            place_job(ctx, {"id": result["job_id"], "name": result.get("name") or ""})
            ctx.cache.invalidate()
        return

    if name == "mason-recover":
        if result is None:
            journal.adopt_failed(ctx, "scene")
            return
        if isinstance(result, dict):
            tab = adopt(ctx, result["doc"], path=None, title=result.get("title"))
            docmodes.mark_recovered(tab, result["autosave"])
            _enter_mason(ctx)
        return

    tab = state.get(key.split(":", 1)[1]) if ":" in key else None
    if tab is None:
        return
    tab.saving = False
    if not isinstance(result, dict):
        return  # a cancelled dialog

    if result.get("exported"):
        skipped = result.get("skipped") or []
        if skipped:
            ctx.toast("Exported, with " + "; ".join(skipped), "warn")
        else:
            ctx.toast("Exported.")
        return

    tab.mark_saved(result.get("rev"))
    journal.drop(ctx, tab)
    if result.get("retitle") and result.get("path"):
        tab.path = Path(result["path"])
        tab.title = mason_state.title_for(tab.path)
        remember_path(ctx, tab.path)
        persist(ctx)
    ctx.toast("Saved.")


def on_task_failed(ctx: Any, done: Any) -> None:
    """A failed save must not leave the document locked -- ``clay_mode``'s
    own reason."""
    state = ctx.state.mason
    if state is None or ":" not in done.key:
        return
    tab = state.get(done.key.split(":", 1)[1])
    if tab is not None:
        tab.saving = False


# --- the guard ------------------------------------------------------------------


def guard(ctx: Any, verb: str, proceed: Any) -> bool:
    return docmodes.guard(ctx, "mason", "scene", "scenes", verb, proceed)


def close_tab(ctx: Any, uid: str) -> None:
    """Close one scene, asking first if it has unsaved work.

    What a Mason document owns in the single GL context: the active tab's
    cache in ``ctx.mason_view``, cleared after cancelling a live drag.
    """
    state = ensure(ctx)

    def release(tab: MasonTab) -> None:
        view = getattr(ctx, "mason_view", None)
        if view is not None and tab.uid == state.active_uid:
            if getattr(view, "dragging", False):
                view.cancel_drag(tab.doc)
            view.clear()

    docmodes.close_tab(ctx, state, uid, release)


# --- keys -------------------------------------------------------------------------

# Built from ``mason_state.TOOLS`` so the keyboard binding and the tool list
# can never name two different tools for the same letter.
TOOL_KEYS = {shortcut.lower(): key for key, _label, shortcut in mason_state.TOOLS}

_MUTATING_CTRL = docmodes.WRITE_CHORDS | frozenset({"a", "i", "j", "m"})
_DRAG_BLOCKED_CTRL = frozenset({"z", "y", "n", "o", "tab", "w", "s"})


def undo(ctx: Any, tab: Any) -> None:
    tab.doc.undo()


def redo(ctx: Any, tab: Any) -> None:
    tab.doc.redo()


def step_history(ctx: Any, tab: Any, index: int) -> bool:
    del ctx
    return tab.doc.step_history(index)


def handle_key(ctx: Any, event: Any) -> bool:
    """Mason's shortcuts. -> whether the key was consumed.

    **False with nothing open**, which the caller's fall-through depends on:
    Mason owns a viewport, and with no document the viewport's own shortcuts
    must still work.
    """
    import pygame

    if event.type == pygame.KEYDOWN and pygame.key.name(event.key) == "f" and not (
        event.mod & (pygame.KMOD_CTRL | pygame.KMOD_ALT)
    ):
        ensure(ctx).frame_pending = True
        return True

    state = ctx.state.mason
    if state is None or not state.docs:
        return False
    tab = state.active
    if tab is None:
        return False
    doc = tab.doc

    if event.type != pygame.KEYDOWN:
        return True

    mods = event.mod
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    name = pygame.key.name(event.key)

    if ctrl:
        return _ctrl_key(ctx, state, tab, doc, name, shift=shift)

    if name in TOOL_KEYS and not shift:
        state.tool = TOOL_KEYS[name]
    elif name == "g" and not tab.saving:
        group_selected(ctx)
    elif event.key == pygame.K_DELETE:
        if not tab.saving:
            delete_selected(ctx)
    elif event.key == pygame.K_ESCAPE:
        doc.select([])
    return True


def _ctrl_key(
    ctx: Any, state: MasonState, tab: MasonTab, doc: Any, name: str, *, shift: bool
) -> bool:
    if docmodes.blocked_while_writing(tab, name, _MUTATING_CTRL):
        return True
    view = getattr(ctx, "mason_view", None)
    if getattr(view, "dragging", False) and name in _DRAG_BLOCKED_CTRL:
        return True
    if name in AXIS_VIEW_KEYS or name == "5":
        if view is not None:
            axis_view_key(view.camera, name, shift)
    elif name == "s":
        save_as(ctx, tab) if shift else save(ctx, tab)
    elif name == "o":
        ask_open(ctx)
    elif name == "n":
        new_document(ctx)
    elif name == "w":
        close_tab(ctx, tab.uid)
    elif name == "e" and not shift:
        export_glb(ctx, tab)
    elif name == "z":
        redo(ctx, tab) if shift else undo(ctx, tab)
    elif name == "y":
        redo(ctx, tab)
    elif name == "a":
        doc.select([n.uid for n in doc.all_nodes()])
    elif name == "j" and not tab.saving:
        duplicate_selected(ctx)
    elif name == "d" and not tab.saving:
        doc.select([])
    elif name == "tab":
        state.cycle(-1 if shift else 1)
    return True


# --- crash recovery (UX-05) ---------------------------------------------------


def _journal_slots(ctx: Any) -> list[Any]:
    """Dirty tabs that are not mid-write -- ``clay_mode``'s reason: an edit
    landing mid-encode would produce an archive whose parts disagree about
    what is in the document."""
    state = getattr(ctx.state, "mason", None)
    if state is None:
        return []
    return [tab for tab in state.docs if tab.dirty and not tab.saving]


def _journal_encode(tab: Any) -> bytes:
    from .mason import serialize

    tab.doc.view = {
        **{name: getattr(tab.view, name) for name in serialize.VIEW_FIELDS},
        "target": tuple(tab.view.target),
    }
    return serialize.wscn_bytes(tab.doc)


def _journal_adopt(ctx: Any, path: Path, meta: dict[str, Any]) -> bool:
    """Reopen one recovered ``.wscn`` as an *untitled, dirty* document --
    ``clay_mode._journal_adopt``'s reason: the file it was copied from may
    still be on disk with its own contents, and adopting the path would arm
    Ctrl+S to overwrite something the user has not looked at."""
    ensure(ctx)
    ctx.submit(f"mason-recover:{_path_key(path)}", _load_recovery, Path(path), dict(meta))
    return True


def _load_recovery(path: Path, meta: dict[str, Any]) -> dict[str, Any] | None:
    from .mason import serialize

    try:
        doc = serialize.read_wscn(_within_ceiling(path).read_bytes())
    except Exception:
        log.exception("could not reopen the recovered scene at %s", path)
        return None
    return {
        "doc": doc,
        "title": f"{meta.get('title') or path.stem} (recovered)",
        "autosave": str(path),
    }


JOURNAL = journal.register(
    journal.Provider(
        kind="mason",
        ext=".wscn",
        label="scene",
        slots=_journal_slots,
        uid_of=lambda tab: tab.uid,
        title_of=lambda tab: tab.title,
        head_of=lambda tab: tab.doc.history.head,
        encode=_journal_encode,
        adopt=_journal_adopt,
    )
)


def release_all(ctx: Any) -> None:
    """Teardown door: release the viewport's GL state. Every mode's own
    tabs stay in memory -- this is the GL half, the part that must not
    outlive the context."""
    view = getattr(ctx, "mason_view", None)
    if view is not None:
        view.release()
    source = getattr(ctx, "mason_assets", None)
    if source is not None:
        source.release()
