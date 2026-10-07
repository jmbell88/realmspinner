"""Clay's controller: opening, saving, exporting, guarding and keys.

Everything here is *about* documents rather than geometry -- the engine under
``clay/`` has no idea a job or a task thread exists, and this is the layer that
knows about both. The panes draw; this decides.

The one rule that shapes the whole file is the raster editor's: **no file dialog
and no encode ever runs on the frame thread.** A native picker is modal to the
OS and blocks until dismissed, and a document of any size is a zip to build.
Both go through ``ctx.submit`` and come back through :func:`on_task_done`, which
is why saving is a *state* (``ClayTab.saving``) rather than a function call
that returns.

**The encode half of that rule was not actually followed here until the
2026-09-06 audit (clay-03).** ``save_to``, ``save_as`` and ``export_asset``
each called the zip-and-PNG build (``serialize.rblk_bytes``, and
``export_asset``'s ``glbwrite.write_glb``) on the calling thread, before
``ctx.submit`` ever ran -- only the disk write was inside the closure.
``save_to`` and ``save_as`` now take a cheap snapshot on the frame thread
(``serialize.snapshot``) and encode it inside ``run()``, the shape
``packwright_io.save_to`` already used. ``export_asset``'s document reads and
encodes both moved into ``build_asset`` (2026-09-09), called from ``run()``,
so the same one chain also serves the agent's synchronous export -- see its
docstring.

Two consequences follow, and both were bugs in the raster editor before they
were rules here.

**A failed save must clear that state.** ``saving`` gates every control that
changes the document, so without :func:`on_task_failed` one failed write leaves
the tab permanently read-only with no way back short of closing it. The same
applies to a submit the runner *refuses* -- a second save while one is in flight
-- which is why :func:`_start` unsets the flag on a False return.

**The head a save records is read after the document settles**, at exactly one
place. A head captured before whatever the save itself pushes saves the document
against a head one behind it, and dirty -- being a comparison against that head
-- then stays true however many times the user saves.

Every task key carries the ``clay-`` prefix, because the app claims results by
prefix: a key without one is a result delivered nowhere.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any

from ....core.safeio import atomic, sizeguard
from ... import dialogs, docmodes, journal
from ..._view_frame import AXIS_VIEW_KEYS, axis_view_key
from . import state as clay_state
from .state import ClayState, ClayTab

log = logging.getLogger(__name__)

RBLK_FILTER = ["Realmspinner Clay document (*.rblk)", "*.rblk"]


def _path_key(path: Path) -> str:
    """A short, stable id for a path, safe to fold into a task key.

    ``hash(str(path))`` is salted per process (``PYTHONHASHSEED``) and, at 64
    bits, two different paths in the same session can still land on the same
    ``abs()`` value -- a collision silently drops the second open rather than
    submitting it. sha1 has neither problem: same input, same digest, always.
    """
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def ensure(ctx: Any) -> ClayState:
    """The mode's state, built on first use.

    Lazy because a session that never opens Clay should not pay for it,
    and because ``AppState`` deliberately knows nothing about it.

    The view block -- the grid, its size and the god light -- is read back
    from settings on this first build only, the same point ``inker_mode.
    ensure`` restores its canvas furniture from. Validated rather than
    trusted (``_restore_view``): the file is hand-editable.
    """
    state = ctx.state.clay
    if state is None:
        state = ClayState()
        # The agent host and the headless test contexts build Clay with no
        # settings store at all; a missing store means defaults, not a crash
        # that fails every agent tool call.
        settings = getattr(ctx, "settings", None)
        stored = settings.get("clay") if callable(getattr(settings, "get", None)) else None
        _restore_view(state, stored.get("view") if isinstance(stored, dict) else None)
        ctx.state.clay = state
    if state.settle_drag is None:
        # Wired here rather than at construction: a fresh ``ClayState`` has
        # no ``ctx`` to close over above, and the closure only ever captures
        # ``ctx`` itself (not ``ctx.clay_view``), so it resolves the view at
        # call time -- sound whether or not one exists yet the first time
        # this runs. See ``ClayState.settle_drag``'s own docstring (clay-01,
        # 2026-09-15 audit).
        state.settle_drag = lambda old_uid: _settle_drag_on_tab_switch(ctx, state, old_uid)
    return state


def _settle_drag_on_tab_switch(ctx: Any, state: ClayState, old_uid: str) -> None:
    """``ClayState.settle_drag``'s real body: commit a live transform drag
    against the tab being left, *before* ``activate`` moves ``active_uid``.

    Mirrors ``close_tab``'s ``release``, which already calls
    ``view.cancel_drag`` for the tab-closing case; this is deliberately a
    *commit* rather than a cancel, because switching tabs does not discard
    the document the way closing one does -- the drag stays on the tab it
    was made on and the user can Ctrl+Z it like anything else when they
    switch back, rather than losing it silently.
    """
    tab = state.get(old_uid)
    if tab is None:
        return
    # The 2026-10-07 audit's clay-12: an armed UV live rotate/scale is a drag the
    # viewport knows nothing about, and the pane that would commit it draws only
    # the *active* tab. Left armed, the leaving tab kept its undo gesture open
    # (which switches the stack's eviction off) and its preview applied, and
    # swallowed every bare key the next time it was active. Committed, like the
    # 3-D drag below, for the same reason: the preview is already on the object.
    uv_view = getattr(tab, "uv_view", None)
    if uv_view is not None and uv_view.drag_mode in ("rotate", "scale"):
        _settle_uv_gesture(tab.doc, uv_view, commit=True)
    view = getattr(ctx, "clay_view", None)
    if view is None or not getattr(view, "dragging", False):
        return
    view.settle_drag(tab.doc)


def _restore_view(state: ClayState, stored: Any) -> None:
    """The grid/god-light preferences back off disk, clamped rather than
    trusted -- ``inker_mode._restore_canvas``'s doctrine for the same reason:
    a hand-edited ``settings.json`` can hold anything JSON allows."""
    if not isinstance(stored, dict):
        return
    state.grid = bool(stored.get("grid", state.grid))
    size = stored.get("grid_size")
    # The 2026-10-03 audit's clay-58: ``json.loads`` accepts NaN, Infinity and
    # 1e999 (which parses to infinity), and ``round`` raises on all three, out of
    # ``ensure`` before ``ctx.state.clay`` is assigned -- so one hand-edited value
    # made Clay unopenable on every launch. A non-finite size keeps the default.
    if (
        isinstance(size, int | float)
        and not isinstance(size, bool)
        and math.isfinite(float(size))
    ):
        state.grid_size = max(1.0, min(1000.0, round(float(size))))
    state.god_light = bool(stored.get("god_light", state.god_light))


# The three recents wrappers every document mode carries, over the one
# list Home's Resume rows are built from (``docmodes.recents_for``).
remember_path, forget_path, recent_paths = docmodes.recents_for("clay")


def persist(ctx: Any) -> None:
    """The view block: the grid, its size and the god light.

    The recent list moved to :mod:`.recents`, which persists itself on every
    write, so this used to be a no-op -- kept callable from a dozen places
    after every open and save on purpose, so a mode with nothing to persist
    today can gain a setting tomorrow without a second door being wired in.
    Task A's ``grid_size`` and Task C's ``god_light`` are that setting:
    properties of the person rather than of any document, ``inker_mode.
    persist``'s own distinction, so they live beside the swatches and the
    canvas furniture rather than in a ``.rblk``.

    Merged into whatever is already stored, ``inker_mode.persist``'s reason:
    a block this function does not know about yet (there is none today, but
    the shape is worth keeping) must survive a write that only touches the
    view.
    """
    state = ctx.state.clay
    settings = getattr(ctx, "settings", None)
    if state is None or not callable(getattr(settings, "set", None)):
        return
    stored = settings.get("clay")
    block = dict(stored) if isinstance(stored, dict) else {}
    block["view"] = {
        "grid": bool(state.grid),
        "grid_size": float(state.grid_size),
        "god_light": bool(state.god_light),
    }
    ctx.settings.set("clay", block)


def active(ctx: Any) -> ClayTab | None:
    state = ctx.state.clay
    return state.active if state is not None else None


def _enter_clay(ctx: Any) -> None:
    """Adoption switches modes through ``state.set_mode``, never by assignment.

    A bare assignment to ``state.mode`` skips the ``previous_mode`` /
    ``mode_observed`` pair that function maintains -- the drift its own
    docstring names -- so Esc out of the next pass-through mode would go back
    to wherever a *keypress* last was rather than to Clay. (Worded to stay out
    of ``tests/test_mode_writes.py``'s line scan, which cannot tell prose from
    code.)
    """
    from ...state import set_mode

    set_mode(ctx.state, "clay")


# --- opening ----------------------------------------------------------------


def announce_migration(ctx: Any, doc: Any) -> None:
    """One toast for what reading an older ``.rblk`` changed, when it changed
    anything.

    ``read_rblk`` records each change as a sentence on ``doc.notices`` (empty
    for a current file and for any document built in the app). Called from
    :func:`adopt`, the one landing every opener -- Open, Open in Clay on a
    library asset, crash recovery -- ends in, so the person hears it exactly
    once and no opener has to remember to ask.

    **A warning when anything was lost, and a bounded one.** The 2026-10-07
    audit's clay-23: this was the one announcement of an irreversible v3 -> v4
    change and it went out as an "info" toast (four seconds, no input) whose
    per-object sentences were unbounded -- 532 characters for the repo's own
    fixture -- and ``doc.notices`` is shown nowhere else. A notice that says
    something was dropped, removed or gone raises the sticky "warn" level
    (``state.TOAST_STICKY``: it takes the mouse, so the person can read it and
    close it), lists those sentences first, and stops at
    :data:`MAX_MIGRATION_NOTICES` with "and N more"; the whole list goes to the
    log. The lead sentence says what happened rather than that the file was
    older: a current-format file can carry notices too (an unknown generator
    frozen to a plain mesh), and "older" was untrue of it.
    """
    notices = tuple(getattr(doc, "notices", ()) or ())
    if not notices:
        return
    lost = [n for n in notices if any(word in n for word in _LOSS_WORDS)]
    ordered = lost + [n for n in notices if n not in lost]
    shown = ordered[:MAX_MIGRATION_NOTICES]
    message = "Clay changed this file as it opened it.\n" + " ".join(shown)
    if len(ordered) > len(shown):
        message += f" ...and {len(ordered) - len(shown)} more (see the log)."
        log.info("Clay changed a file as it opened it: %s", " ".join(notices))
    ctx.toast(message, "warn" if lost else "info")


#: The most conversion sentences the open toast spells out.
MAX_MIGRATION_NOTICES = 3
#: What ``kernels/mesh/legacy.py``'s notices say when they report something lost.
_LOSS_WORDS = ("dropped", "removed", " gone")


def adopt(
    ctx: Any,
    doc: Any,
    *,
    path: Path | None = None,
    title: str | None = None,
    view: dict[str, Any] | None = None,
) -> ClayTab:
    state = ensure(ctx)
    tab = ClayTab(
        doc=doc,
        title=title or clay_state.title_for(path),
        path=path,
        saved_head=doc.history.head,
    )
    if view:
        tab.view.yaw = view["yaw"]
        tab.view.pitch = view["pitch"]
        tab.view.distance = view["distance"]
        tab.view.target = view["target"]
        # Already framed, which is the whole point of having stored one: an
        # auto-fit over the top would throw away the answer just read off disk.
        tab.view.fitted = True
    state.add(tab)
    remember_path(ctx, path)
    persist(ctx)
    announce_migration(ctx, doc)
    return tab


def new_document(ctx: Any) -> ClayTab:
    from ....kernels.mesh import document as bd

    return adopt(ctx, bd.ClayDoc(), title="Untitled")


def _within_ceiling(path: Path) -> bytes:
    """Refuse and read a document too big to open, in one bounded call.

    Clay had **no size ceiling anywhere**, though
    ``service.files.MAX_CLAY_SOURCE_BYTES`` has existed since the format did --
    applied at exactly one place, the *upload*, and at neither of the two doors
    a user reaches. The ceiling is that same number rather than a second one
    invented here, for ``plotter_io``'s reason: "how big may a clay document
    be" has one answer, and two would drift the first time either moved.

    Reads through :func:`sizeguard.read_bytes_within_ceiling` rather than
    ``sizeguard.within_ceiling(path, N).read_bytes()`` -- the separate
    ``stat()`` and ``read_bytes()`` that shape used left a window for a file
    that grows in between to sail past the ceiling it was meant to bound
    (shell-07, the 2026-09-18 audit).
    """
    from ....service.files import MAX_CLAY_SOURCE_BYTES

    return sizeguard.read_bytes_within_ceiling(path, MAX_CLAY_SOURCE_BYTES)


def _refuse_oversized_save(data: bytes) -> None:
    """Refuse to write a ``.rblk`` that ``_load`` would then refuse to reopen.

    The 2026-09-23 audit (clay-03): ``save_to`` and ``save_as`` wrote past
    ``MAX_CLAY_SOURCE_BYTES`` with no check at all -- Clay had no ceiling on
    *editing* a document, so a user could grow one past the size this same
    module already refuses to open (:func:`_within_ceiling`), and only find
    out the next time they tried. Checked here, inside ``run()``, against the
    exact bytes about to be written -- the same "after the document settles"
    reasoning :func:`save_to`'s own docstring gives for reading the head where
    it does.
    """
    from ....service.errors import TooLarge
    from ....service.files import MAX_CLAY_SOURCE_BYTES

    if len(data) > MAX_CLAY_SOURCE_BYTES:
        raise TooLarge(
            f"This document is past the {MAX_CLAY_SOURCE_BYTES:,} bytes Clay will reopen.",
            field="save",
        )


def _encode_or_refuse(snap: Any) -> bytes:
    """``serialize.snapshot_bytes`` with its refusal made a message for a person.

    The 2026-10-03 audit's follow-up to clay-27: the encoder refuses a document
    past the object or triangle ceilings :func:`~.kernels.mesh.serialize.read_rblk`
    would refuse to reopen, with a plain ``ValueError`` because a kernel cannot
    import ``service.errors``. The task runner treats a bare ``ValueError`` as a
    bug -- "Something went wrong; see the log" -- when the message already names
    the cause ("this clay document places 4,097 objects, past the 4,096 Clay
    holds"). Re-raised here as the same ``TooLarge(field="save")``
    :func:`_refuse_oversized_save` raises, so both reach the toast alike. It runs
    before ``atomic.write_bytes``, so nothing is written and the document is
    as it was.
    """
    from ....kernels.mesh import serialize
    from ....service.errors import TooLarge

    try:
        data = serialize.snapshot_bytes(snap)
    except ValueError as error:
        raise TooLarge(str(error), field="save") from error
    _refuse_oversized_save(data)
    return data


def _within_mesh_ceiling(path: Path) -> bytes:
    """The same question about a GLB, which is a different number.

    ``MAX_MESH_BYTES`` and not the document ceiling: an imported mesh is what
    the service already accepts as an *uploaded* mesh, and a hundred-thousand
    triangle ``model.glb`` is the ordinary case rather than the extreme one.

    Reads through :func:`sizeguard.read_bytes_within_ceiling`, same reason as
    :func:`_within_ceiling` above (shell-07, the 2026-09-18 audit).
    """
    from ....service.validation import MAX_MESH_BYTES

    return sizeguard.read_bytes_within_ceiling(path, MAX_MESH_BYTES)


def _load(path: Path) -> dict[str, Any]:
    """Blocking; task thread only. Raises rather than returning a broken tab."""
    from ....kernels.mesh import serialize

    data = _within_ceiling(Path(path))
    doc = serialize.read_rblk(data)
    return {
        "doc": doc,
        "path": str(path),
        "title": clay_state.title_for(Path(path)),
        # A second read of the same bytes, deliberately: see ``read_view``'s own
        # docstring for why the camera is not a second return value from
        # ``read_rblk``. It parses one small JSON member of an in-memory zip.
        "view": serialize.read_view(data),
    }


def ask_open(ctx: Any) -> None:
    """The picker, on a task thread, then the decode on the same one."""
    ensure(ctx)

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Open Clay document", RBLK_FILTER)
        return None if path is None else _load(path)

    ctx.submit("clay-open", run)


def open_path(ctx: Any, path: Path) -> None:
    state = ensure(ctx)
    path = Path(path)
    existing = state.find_path(path)
    if existing is not None:
        # Focus rather than fork: two tabs over one path would race on save.
        state.activate(existing.uid)
        return
    ctx.submit(f"clay-open:{_path_key(path)}", _load, path)


# --- importing --------------------------------------------------------------

# Above this an import gets a confirm dialog first. Not a refusal -- Clay can
# edit it, and a user who has just asked to edit their own asset should be
# allowed to -- but a rebuild-per-edit at this scale is a visible pause, and
# finding that out by pressing Extrude is worse than being told.
SLOW_TRIANGLES = 200_000


#: A material library is a few lines per material; anything this large is
#: not one, and reading it would only feed the parser noise.
MAX_MTL_BYTES = 4 * 1024 * 1024


def _sibling_mtl(path: Path, data: bytes) -> str | None:
    """The text of the ``.mtl`` an OBJ's first ``mtllib`` line names, if it
    sits beside the OBJ. Blocking; task thread only.

    Without this an OBJ that Clay itself exported came back grey: Export OBJ
    writes the colours into the ``.mtl``, and the import never looked. Only a
    bare file name in the same folder is followed -- a path with a directory
    part is refused rather than resolved, so an OBJ cannot make the importer
    read a file somewhere else on the disk. A missing, oversized or unreadable
    library is not an error: the OBJ still imports, just without its colours.
    """
    # ``utf-8-sig`` on both reads (the 2026-10-07 audit's clay-15): a BOM at the
    # head of the OBJ hid a first-line ``mtllib`` from the ``startswith`` below, and
    # one at the head of the ``.mtl`` made its first ``newmtl`` unrecognisable --
    # Windows tools write them routinely, and the colours silently went grey.
    for raw in data.decode("utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line.startswith("mtllib"):
            continue
        name = line[len("mtllib"):].strip()
        if not name or Path(name).name != name:
            return None
        candidate = path.with_name(name)
        try:
            if not candidate.is_file() or candidate.stat().st_size > MAX_MTL_BYTES:
                return None
            return candidate.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            return None
    return None


def import_mesh_path(
    ctx: Any, path: Path, *, scale: float | None = None, up: str | None = None
) -> None:
    """Parse any of :data:`~.meshimport.SUPPORTED_SUFFIXES` on a task thread
    and adopt it as a document.

    The generalisation of what used to be ``import_glb_path`` (now a thin
    alias below): the drop handler (``studio/shell/events.py``'s Clay branch
    of ``_on_drop``) and the bridge pane's "Import Mesh..." button both call
    this rather than each knowing GLB is one format among four.

    The parse is O(triangles) and a ``model.glb`` is routinely a hundred
    thousand of them, so it never runs on the frame thread -- the same rule
    every dialog and every encode in this module follows.

    ``scale``/``up`` default to whatever the tab last remembered
    (``ClayState.import_scale``/``import_up``), so a session importing a batch
    of same-convention files sets them once. An explicit value updates that
    memory for the *next* import too, which is what makes the bridge's own
    combo -- read the state, not passed a value -- and a caller that always
    wants exactly 1.0/"y" (there is none today) behave the same way.
    """
    state = ensure(ctx)
    path = Path(path)
    if scale is not None:
        state.import_scale = float(scale)
    if up is not None:
        state.import_up = up
    use_scale = state.import_scale
    use_up = state.import_up

    def run() -> dict[str, Any]:
        from ....kernels.mesh import meshimport

        data = _within_mesh_ceiling(path)
        mtl = _sibling_mtl(path, data) if path.suffix.lower() == ".obj" else None
        doc = meshimport.import_file(
            data, path.suffix, path.stem, scale=use_scale, up=use_up, mtl=mtl
        )
        triangles = sum(max(len(obj.mesh.starts) - 1, 0) for obj in doc.objects)
        return {"doc": doc, "title": path.stem, "triangles": triangles}

    ctx.submit(f"clay-import:{_path_key(path)}", run)


def import_glb_path(ctx: Any, path: Path) -> None:
    """Thin alias of :func:`import_mesh_path`, kept for existing callers (this
    module's own tests, and anything reaching for it by its old, GLB-only
    name) rather than a signature change rippling out for no behaviour
    change -- a bare GLB import is exactly ``scale=1.0, up="y"``, this
    function's own defaults."""
    import_mesh_path(ctx, path)


def ask_import_mesh(ctx: Any) -> None:
    """The picker for the bridge's "Import Mesh..." button -- ``ask_open``'s
    own shape (the picker and the parse, both on one task thread), pointed at
    :data:`~.meshimport.SUPPORTED_SUFFIXES` instead of ``.rblk``.

    The bare ``clay-import`` key, not a path-keyed one: a picker's result is
    not known until it returns, so there is nothing to key on yet, the same
    reason ``ask_open`` submits under bare ``clay-open``.
    """
    state = ensure(ctx)
    use_scale = state.import_scale
    use_up = state.import_up

    def run() -> dict[str, Any] | None:
        from ....kernels.mesh import meshimport

        path = dialogs.open_file("Import mesh", IMPORT_MESH_FILTER)
        if path is None:
            return None
        data = _within_mesh_ceiling(path)
        # The 2026-09-19 audit (clay-07): this picker never read the sibling
        # ``.mtl``, so an OBJ Clay itself exported came back grey through this
        # button while drag-and-drop (``import_mesh_path``, which has always
        # called ``_sibling_mtl``) got the colours right. Same file, same
        # lookup, so the two routes agree.
        mtl = _sibling_mtl(path, data) if path.suffix.lower() == ".obj" else None
        doc = meshimport.import_file(
            data, path.suffix, path.stem, scale=use_scale, up=use_up, mtl=mtl
        )
        triangles = sum(max(len(obj.mesh.starts) - 1, 0) for obj in doc.objects)
        return {"doc": doc, "title": path.stem, "triangles": triangles}

    ctx.submit("clay-import", run)


def edit_asset_in_clay(ctx: Any, job: Any) -> None:
    """Open a library asset in Clay: its authored document if it has one.

    The ``build.rblk`` sidecar is preferred whenever it is there, and that is
    the point of the whole feature -- it is the document the user actually
    authored, with its objects, its names and its generator parameters intact,
    and until now it was a file written and never read back. Failing that, the
    *optimized* ``model.glb`` is imported: it is the mesh that is served,
    grounded and exported, and ``source.glb`` is the raw reconstruction that
    nothing downstream uses.
    """
    ensure(ctx)
    job_id = job["id"] if isinstance(job, dict) else str(job)
    name = (job.get("name") if isinstance(job, dict) else "") or "Asset"

    def run() -> dict[str, Any]:
        from ....service import files as svc_files

        # 2026-09-26 audit, finding clay-mode-06: ``.exists()`` is true of a
        # directory too, so a sidecar name that happened to be a directory (or
        # a ``model.glb`` that was one) used to pass this gate and then fail
        # inside ``_load``/``_parse_glb`` with whatever raw OS error reading a
        # directory as a file raises, instead of falling back the way a
        # genuinely missing sidecar already does. ``.is_file()`` makes both
        # checks agree with what they actually need -- a file to open -- and
        # a sidecar that is not one now falls through to ``model.glb`` exactly
        # like a sidecar that never existed.
        sidecar = svc_files.clay_source_path(ctx.svc, job_id)
        if sidecar.is_file():
            return _load(sidecar)
        mesh = ctx.svc.config.job_dir(job_id) / "model.glb"
        if not mesh.is_file():
            raise FileNotFoundError(f"{job_id} has no mesh to edit")
        return _parse_glb(_within_mesh_ceiling(mesh), name)

    ctx.submit(f"clay-import:{job_id}", run)


def _parse_glb(data: bytes, name: str) -> dict[str, Any]:
    """Blocking; task thread only. Raises rather than returning a broken tab."""
    from ....kernels.mesh import glbimport

    doc = glbimport.glb_to_claydoc(data, name=name)
    triangles = sum(
        max(len(obj.mesh.starts) - 1, 0) for obj in doc.objects
    )
    return {"doc": doc, "title": name, "triangles": triangles}


def _adopt_import(ctx: Any, result: dict[str, Any]) -> None:
    """Adopt a parsed import, asking first when it is big enough to be slow."""
    doc, title = result["doc"], result.get("title") or "Imported"
    # A ``.rblk`` sidecar carries a camera; a GLB does not, and ``None`` is
    # simply "frame it", which is what an import has always done.
    view = result.get("view")
    if int(result.get("triangles", 0)) <= SLOW_TRIANGLES:
        adopt(ctx, doc, title=title, view=view)
        _enter_clay(ctx)
        return
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Edit this mesh in Clay?",
            message=(
                f"{int(result['triangles']):,} faces. Editing will be slow -- "
                "every edit rebuilds the whole mesh, and the undo history holds "
                "two copies per step."
            ),
            confirm_label="Edit anyway",
            cancel_label="Cancel",
            on_confirm=lambda: _adopt_now(ctx, doc, title, view),
        )
    )


def _adopt_now(ctx: Any, doc: Any, title: str, view: dict[str, Any] | None = None) -> None:
    adopt(ctx, doc, title=title, view=view)
    _enter_clay(ctx)


# --- saving -----------------------------------------------------------------


def camera_of(ctx: Any, tab: ClayTab) -> Any:
    """The tab's stored camera, refreshed from the live viewport first.

    Called on every path that writes the document, because ``tab.view`` is only
    brought up to date when the tab is switched away from -- so saving the tab
    you are looking at would otherwise store wherever the camera was when you
    last left it, which is the one case where the answer is visibly wrong.
    """
    view = getattr(ctx, "clay_view", None)
    if view is not None and getattr(view, "camera", None) is not None:
        tab.view.read_from(view.camera)
    return tab.view


def sync_active_camera(ctx: Any) -> None:
    """Keep the drawn tab's stored camera live, not just at save and switch.

    ``tab.view`` is what a journal autosave writes, and it used to be refreshed
    only on an explicit save or the viewport's tab-switch handoff, so minutes of
    free orbiting could be lost to a crash. The viewport calls this every frame
    *before* its handoff: the live camera belongs to the tab it last drew
    (``state.camera_tab``), not to whatever ``active_uid`` says now -- an
    event-layer switch (Ctrl+Tab, closing a tab) moves ``active_uid`` first --
    so syncing from ``active_uid`` would write the outgoing tab's orientation
    onto the incoming tab's stored view.
    """
    state = ctx.state.clay
    if state is None:
        return
    drawn = getattr(state, "camera_tab", None)
    if not drawn:
        # The 2026-10-07 audit's clay-02: until the viewport has drawn a tab the
        # live camera belongs to *nothing* -- it is the fresh viewport's default.
        # Falling back to ``active_uid`` here copied that default onto the tab a
        # session's first Open/Resume/recovery had just adopted, before
        # ``apply_camera`` could read the stored view off it, and marked it framed.
        return
    tab = state.get(drawn)
    if tab is not None:
        camera_of(ctx, tab)


def remember_camera(ctx: Any, tab: ClayTab | None) -> None:
    """Snapshot the live camera onto a tab that is being switched away from."""
    if tab is not None:
        camera_of(ctx, tab)


def apply_camera(ctx: Any, tab: ClayTab) -> None:
    """Put a tab's camera back on the viewport, or frame it if it has none.

    Framing here rather than in ``ClayView`` because this is the layer that
    knows a *tab* exists: the viewport has one camera and no idea that the thing
    it is drawing changed identity.
    """
    view = getattr(ctx, "clay_view", None)
    if view is None or getattr(view, "camera", None) is None:
        return
    if tab.view.fitted:
        tab.view.write_to(view.camera)
        return
    view.frame_selection(tab.doc)
    tab.view.read_from(view.camera)
    tab.view.fitted = True


# The submit-or-unlock helper is one rule for all four document modes and lives
# in :mod:`.docmodes`; bound here as an assignment rather than wrapped, because
# every call site in this file reaches for one object.
_start = docmodes.start_save


def save_to(ctx: Any, tab: ClayTab, path: Path) -> None:
    """Write the document to a known path.

    The head is read *here*, before the submit and after the document is in
    whatever state the save will encode -- one place, so the two halves of the
    dirty comparison cannot drift apart.

    ``serialize.snapshot`` is the cheap frame-thread half; the zip-and-PNG
    encode (``snapshot_bytes``) now runs inside ``run()``, on the task thread.
    The 2026-09-06 audit (clay-03) found this calling ``rblk_bytes`` -- the
    encode itself -- right here instead, which is the exact stall this
    module's stated rule (see the module docstring) exists to forbid.
    """
    from ....kernels.mesh import serialize

    path = Path(path)
    doc = tab.doc
    rev = doc.history.head
    snap = serialize.snapshot(doc, view=camera_of(ctx, tab))

    def run() -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _encode_or_refuse(snap)
        atomic.write_bytes(path, data)
        return {"rev": rev, "path": str(path), "retitle": True}

    _start(ctx, tab, f"clay-save:{tab.uid}", run)


def save(ctx: Any, tab: ClayTab | None = None) -> None:
    tab = tab or active(ctx)
    docmodes.save(
        tab, save_as=lambda: save_as(ctx, tab), save_to=lambda: save_to(ctx, tab, tab.path)
    )


def _with_extension(path: Path, suffix: str) -> Path:
    """*path* ending in *suffix*: kept when it already does (any case), appended
    otherwise.

    The 2026-10-03 audit, finding clay-111: ``with_suffix`` *replaces* a typed
    extension, so a typed ``barrel.v2`` became ``barrel.rblk`` -- a name the OS
    overwrite prompt never asked about -- and silently overwrote an unrelated
    file of that name.
    """
    if path.suffix.lower() == suffix.lower():
        return path.with_suffix(suffix)
    return path.with_name(path.name + suffix)


def save_as(ctx: Any, tab: ClayTab | None = None) -> None:
    """The picker and the encode on one task thread.

    The *snapshot* is taken on the frame thread and the picker -- and now the
    encode too -- run on the task thread, which is the opposite of what it
    looks like it should be: reading the live document after an unbounded
    modal dialog would snapshot whatever the user did while it was open.

    ``serialize.snapshot`` is cheap enough to take before the picker for
    exactly that reason; the zip-and-PNG encode it used to do inline
    (``rblk_bytes``, before ``ctx.submit`` ran at all) is ``snapshot_bytes``
    now, moved inside ``run()`` by the 2026-09-06 audit (clay-03) alongside
    ``save_to``'s.
    """
    from ....kernels.mesh import serialize

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, title = tab.doc, tab.title
    rev = doc.history.head
    snap = serialize.snapshot(doc, view=camera_of(ctx, tab))

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file(
            "Save Clay document", f"{title}{clay_state.RBLK_SUFFIX}", RBLK_FILTER
        )
        if path is None:
            return None
        path = _with_extension(path, clay_state.RBLK_SUFFIX)
        # Staged, as ``save_to`` is: a picker aimed at an existing document is
        # the ordinary way to overwrite one, and a write that dies partway
        # through would leave that file truncated with no copy of it anywhere.
        # No mkdir -- the picker returns a directory that exists.
        data = _encode_or_refuse(snap)
        atomic.write_bytes(path, data)
        return {"rev": rev, "path": str(path), "retitle": True}

    _start(ctx, tab, f"clay-saveas:{tab.uid}", run)


# --- export -----------------------------------------------------------------


def build_asset(
    svc: Any,
    doc: Any,
    *,
    title: str,
    prompt: str | None = None,
    view: Any = None,
) -> str:
    """The one document -> finished ``model`` row chain.

    **Both the interactive export (``export_asset``) and the agent's
    (``agent_clay._h_export``) go through this rather than each keeping a hand
    copy of it** -- the two had drifted into exactly that before the
    2026-09-09 audit found them, and a step added to one (a new sidecar, a
    thumbnail, a different normalisation) would silently not happen for the
    other. Landing it here means it happens for both.

    ``to_model`` and ``serialize.snapshot`` are the two reads of the live
    ``doc``, taken first and immediately followed by the encodes -- so nothing
    downstream ever sees a document that changed halfway through this call.
    ``to_model`` copies every transform and rebuilds every primitive's arrays
    fresh, and a snapshot holds only references INVARIANTS 312 says are safe
    to keep. The mesh (``import_mesh``) is written before the ``.rblk``
    source sidecar (``save_clay_source``) for the reason ``export_asset``
    always kept: a crash between them leaves the sidecar absent rather than
    lying about a mesh it did not produce.

    Takes a ``RealmspinnerService`` rather than a ``ctx`` -- the house convention
    (see ``src/realmspinner/service/``) -- so this is callable from anywhere a
    document exists, headlessly included, with no tab, toast or task runner
    anywhere in it.
    """
    from ....kernels.geom3d import glbwrite
    from ....kernels.mesh import document as bd
    from ....kernels.mesh import serialize
    from ....service import files as svc_files
    from ....service import jobs as svc_jobs

    model = bd.to_model(doc)
    snap = serialize.snapshot(doc, view=view)
    glb = glbwrite.write_glb(model)
    result = svc_jobs.import_mesh(svc, glb, name=title, prompt=prompt or title)
    job_id = result["id"]
    svc_files.save_clay_source(svc, job_id, serialize.snapshot_bytes(snap))
    return job_id


def export_asset(ctx: Any, tab: ClayTab | None = None) -> None:
    """Mint an ordinary asset from the document: the point of Clay.

    What comes out is a ``done`` model row, so rigging, posing, sprite sheets,
    the triangle retarget and every mesh export work on it with none of those
    paths learning that Clay exists.

    The document -> model-row chain itself is :func:`build_asset` now; see
    its docstring for the read/encode ordering it owns. What stays here is
    what only the interactive path needs: the "nothing visible" refusal,
    the camera pulled from the live viewport (``camera_of``, GL-thread-bound,
    so it is read before the task runs rather than inside it), and
    the task-thread split -- ``build_asset``'s reads and encodes both now run
    inside ``run()``, off the frame thread entirely.
    """
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, title = tab.doc, tab.title
    if not any(obj.visible for obj in doc.objects):
        # Refused here rather than at the service door, so no job directory is
        # ever created for it: check_glb would refuse the same bytes, but only
        # after this had told the user a build was under way.
        ctx.toast("There is nothing visible to export.", "error")
        return

    view = camera_of(ctx, tab)

    def run() -> dict[str, Any]:
        job_id = build_asset(ctx.svc, doc, title=title, view=view)
        return {"job_id": job_id, "exported": True}

    _start(ctx, tab, f"clay-export:{tab.uid}", run)


OBJ_FILTER = ["Wavefront OBJ (*.obj)", "*.obj"]
#: Side of the square a screenshot is drawn at.
SCREENSHOT_SIZE = 2048
IMPORT_MESH_FILTER = [
    "Mesh files (*.glb *.obj *.stl *.ply)",
    "*.glb *.obj *.stl *.ply",
]


#: The first line :func:`~.kernels.mesh.objexport.claydoc_to_obj` writes into
#: both of its files: what makes an existing ``.mtl`` recognisably one of ours.
_EXPORT_HEADER = b"# Written by Realmspinner's Clay"


def _refuse_foreign_sidecars(targets: dict[Path, bytes], primary: Path) -> None:
    """Refuse, before anything is written, to replace a file the user did not
    pick and that is not a previous Clay export.

    The 2026-10-07 audit's clay-07, Mason's mason-34 over again: the native
    dialog confirms an overwrite of the one name typed (``crate.obj``), while
    the export also writes ``crate.mtl`` and each ``crate_<n>.png`` beside it,
    so a user's own ``crate_0.png`` was replaced with no prompt. An ``.mtl``
    is ours when it opens with the export header; a PNG carries no marker, so
    it counts as ours only when an ``.mtl`` of ours is among the existing files
    being replaced alongside it. Raises ``Conflict`` (``field="export"``), which
    reaches the person as a toast naming the file.
    """
    from ....service.errors import Conflict

    def ours(target: Path) -> bool:
        try:
            with target.open("rb") as handle:
                return handle.read(len(_EXPORT_HEADER)) == _EXPORT_HEADER
        except OSError:
            return False

    existing = [t for t in targets if t != primary and t.is_file()]
    owns_textures = any(t.suffix == ".mtl" and ours(t) for t in existing)
    for target in existing:
        if not (ours(target) if target.suffix == ".mtl" else owns_textures):
            raise Conflict(
                f"{target.name} already exists beside {primary.name} and is not a "
                "Clay export -- exporting would replace it. Pick another file name "
                "or move it first.",
                field="export",
            )


def export_mesh_file(ctx: Any, tab: ClayTab | None, kind: str) -> None:
    """Save the document as a plain mesh file on disk -- GLB or OBJ+MTL --
    beside :func:`export_asset`'s library export.

    The two are genuinely different destinations: this writes a file the
    library never sees, for a user handing a mesh straight to another tool.

    **Nothing is built on the frame thread.** The 2026-10-03 audit's clay-59:
    the GLB (``bd.to_model`` then ``glbwrite.write_glb``, which PNG-encodes every
    texture) and the OBJ text were both built here before ``ctx.submit`` -- 0.68 s
    for one 2048x2048 texture, seconds for larger maps -- the exact stall the
    2026-09-06 audit's clay-03 moved off the frame thread for ``export_asset``
    and the saves. The read of the live document and the encode both run inside
    ``run()`` now, *before* the picker opens (so the dialog's open time is never
    the moment a read straddles), the way ``build_asset`` reads on its task:
    ``_start`` has the tab ``saving`` for the whole task, which locks every
    control that edits the document, so nothing writes it between the read and
    the encode.
    """
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, title = tab.doc, tab.title
    if not any(obj.visible for obj in doc.objects):
        ctx.toast("There is nothing visible to export.", "error")
        return

    if kind == "glb":

        def run() -> dict[str, Any] | None:
            from ....kernels.geom3d import glbwrite
            from ....kernels.mesh import document as bd

            model = bd.to_model(doc)
            data = glbwrite.write_glb(model)
            path = dialogs.save_file("Export mesh", f"{title}.glb", dialogs.GLB_FILTER)
            if path is None:
                return None
            path = _with_extension(path, ".glb")
            atomic.write_bytes(path, data)
            return {"path": str(path), "exported_file": True}

    elif kind == "obj":

        def run() -> dict[str, Any] | None:
            from ....kernels.mesh import objexport

            obj_text, mtl_text = objexport.claydoc_to_obj(doc, name=title)
            # PNG-encoded here, on the task thread, before the picker opens:
            # the same read-then-ask order as the text above.
            pngs = objexport.claydoc_textures(doc)
            path = dialogs.save_file("Export mesh", f"{title}.obj", OBJ_FILTER)
            if path is None:
                return None
            path = _with_extension(path, ".obj")
            stem = path.stem
            # ``claydoc_to_obj`` wrote ``mtllib {title}.mtl`` against the tab's
            # own title, which the save dialog is free to have renamed --
            # kept in sync here rather than re-reading the document with the
            # chosen name, so the obj always names the mtl actually beside it.
            # Both names go through ``safe_name``, the form the OBJ text itself
            # carries (a ``#`` or a trailing backslash is neutralised there), and
            # the ``.mtl`` and PNGs are written under that same form: a name the
            # OBJ line holds and no file answers to is a grey import.
            safe_title, safe_stem = objexport.safe_name(title), objexport.safe_name(stem)
            text = obj_text.replace(f"mtllib {safe_title}.mtl", f"mtllib {safe_stem}.mtl", 1)
            # The same rename for every ``map_Kd``: a texture is written as
            # ``<stem>_<index>.png`` so the .mtl names the files actually beside it.
            for index in pngs:
                mtl_text = mtl_text.replace(
                    f"map_Kd {objexport.texture_name(title, index)}",
                    f"map_Kd {objexport.texture_name(stem, index)}",
                    1,
                )
            # One set, staged whole (the 2026-10-07 audit's clay-07). Four
            # separate ``write_bytes`` calls put the ``.obj`` on disk first, so a
            # PNG that then failed left an ``.obj``/``.mtl`` naming a texture that
            # never arrived. Written PNGs first, ``.mtl`` next, ``.obj`` last, so
            # the window between the replaces never shows an OBJ whose library is
            # missing; and a sidecar that is not a previous Clay export is refused
            # before anything is written -- the dialog confirmed only the ``.obj``.
            targets: dict[Path, bytes] = {
                path.with_name(objexport.texture_name(stem, index)): png
                for index, png in pngs.items()
            }
            targets[path.with_name(f"{safe_stem}.mtl")] = mtl_text.encode("utf-8")
            targets[path] = text.encode("utf-8")
            _refuse_foreign_sidecars(targets, path)
            atomic.staged_set(targets)
            return {"path": str(path), "exported_file": True}

    else:
        raise ValueError(f"unknown mesh export kind {kind!r}")

    _start(ctx, tab, f"clay-exportfile:{tab.uid}", run)


def save_screenshot(ctx: Any, tab: ClayTab | None = None) -> None:
    """Save a still of the document, as the user has it posed and framed.

    Drawn offscreen through the live camera (``frame=False``: the picture is
    the angle the user chose, not one this call picks), lit, with no gizmos,
    overlays or grid -- a picture of the subject, not of the editor. The GL
    draw and the PNG encode happen here on the frame thread; only the dialog
    and the write go to a task.
    """
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    if not any(obj.visible for obj in tab.doc.objects):
        ctx.toast("There is nothing visible to photograph.", "error")
        return
    view = getattr(ctx, "clay_view", None)
    if view is None:
        ctx.toast("This window has no viewport to take a screenshot from.", "error")
        return
    try:
        png = view.render_png(tab.doc, size=SCREENSHOT_SIZE, frame=False, shading="lit")
    except Exception:
        log.exception("could not render a Clay screenshot")
        ctx.toast("That document could not be rendered.", "error", "log")
        return
    title = tab.title

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file("Save screenshot", f"{title}.png", dialogs.PNG_FILTER)
        if path is None:
            return None
        path = _with_extension(path, ".png")
        atomic.write_bytes(path, png)
        return {"path": str(path), "exported_file": True}

    _start(ctx, tab, f"clay-exportfile:{tab.uid}", run)


# --- task results -----------------------------------------------------------


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from the app for every ``clay-`` key."""
    state = ensure(ctx)
    key, result = done.key, done.result
    name = key.split(":", 1)[0]

    if name == "clay-open":
        if isinstance(result, dict):
            # ``open_path`` submits ``clay-open:<path key>`` and the picker
            # (``ask_open``) submits bare ``clay-open``, so both land here --
            # this is the one place that can dedupe both. Before the
            # 2026-09-12 audit (finding clay-02), only ``open_path`` checked
            # ``find_path``; the picker had no check anywhere in its path, so
            # opening a document already open in another tab forked a second,
            # independent tab on the same file instead of focusing the first.
            # Two tabs over one path race on save, and whichever wrote last
            # silently overwrote the other's edits. The decode has already
            # happened on the task thread by the time we get here -- that
            # freshly-built ``doc`` is simply dropped in favour of the tab
            # that is already open, which is also whatever the user has
            # unsaved in it, so nothing the user typed is discarded either.
            existing = state.find_path(Path(result["path"]))
            if existing is not None:
                state.activate(existing.uid)
            else:
                adopt(
                    ctx,
                    result["doc"],
                    path=Path(result["path"]),
                    title=result.get("title"),
                    view=result.get("view"),
                )
            _enter_clay(ctx)
        return

    if name == "clay-recover":
        if result is None:
            journal.adopt_failed(ctx, "model")
        if isinstance(result, dict):
            tab = adopt(
                ctx,
                result["doc"],
                path=None,
                title=result.get("title"),
                view=result.get("view"),
            )
            docmodes.mark_recovered(tab, result["autosave"])
            _enter_clay(ctx)
        return

    if name == "clay-import":
        # No path: an imported document has no file of its own, so Ctrl+S asks
        # where to put it rather than overwriting the asset it came from.
        if isinstance(result, dict):
            _adopt_import(ctx, result)
        return

    tab = state.get(key.split(":", 1)[1]) if ":" in key else None
    if tab is None:
        ctx.cache.invalidate()
        return
    tab.saving = False
    if not isinstance(result, dict):
        return  # a cancelled dialog

    if result.get("exported"):
        tab.job_id = result["job_id"]
        ctx.cache.invalidate()
        ctx.toast("Exported as an asset.")
        return

    if result.get("exported_file"):
        # A plain mesh file on disk, not the ``.rblk`` that makes the tab
        # clean -- unlike ``exported`` (the library asset) and the save
        # branch below, this touches neither ``mark_saved`` nor the journal:
        # the document itself is exactly as saved or dirty as it was before.
        ctx.toast(f"Exported to {result['path']}.")
        return

    tab.mark_saved(result.get("rev"))
    # See ``inker_mode``: saved is the moment the crash copy stops
    # describing anything at risk (UX-05).
    journal.drop(ctx, tab)
    if result.get("retitle") and result.get("path"):
        tab.path = Path(result["path"])
        tab.title = clay_state.title_for(tab.path)
        remember_path(ctx, tab.path)
        persist(ctx)
    ctx.toast("Saved.")


def on_task_failed(ctx: Any, done: Any) -> None:
    """A failed save must not leave the document locked.

    ``saving`` disables every editing control, so without this a single failed
    write makes the tab permanently read-only with no way back short of
    closing it.
    """
    state = ctx.state.clay
    if state is None or ":" not in done.key:
        return
    name = done.key.split(":", 1)[0]
    tab = state.get(done.key.split(":", 1)[1])
    if tab is None:
        return
    if name in ("clay-save", "clay-saveas", "clay-export", "clay-exportfile"):
        tab.saving = False


# --- the guard --------------------------------------------------------------


def guard(ctx: Any, verb: str, proceed: Any) -> bool:
    """Ask before losing unsaved work. -> whether it went ahead now.

    One question for all of them: ``ConfirmQueue`` holds a single pending
    question, so asking per dirty document would silently drop all but the
    first. Only quitting and closing a tab are destructive -- switching modes
    is not, because Clay is a mode rather than a takeover and its tabs are
    still there when you come back.
    """
    return docmodes.guard(ctx, "clay", "document", "documents", verb, proceed)


def close_tab(ctx: Any, uid: str) -> None:
    """Close one document, asking first if it has unsaved work.

    ``ClayState.close`` has been here since the multi-document work and had no
    caller at all: Clay could open documents and never shut one, which also
    meant ``guard``'s "3 documents have unsaved changes" named documents the
    user had no way to reach -- Ctrl+Tab cycled them with nothing on screen
    saying so.

    ``docmodes.close_tab`` asks the question and refuses a tab mid-save; what
    is Clay's is the release below.

    What a Clay document owns in the single GL context is the part worth
    stating. The per-document camera is plain data on the tab
    (``clay_state.CameraView``); the GPU buffers live in ``ctx.clay_view``, whose
    ``_cache`` is keyed on *object* uid and holds whichever document was last
    synced. Dropping the document those buffers were built from without
    releasing them leaves the renderer one frame away from drawing a mesh that
    no longer belongs to anything -- so the cache is cleared, which costs one
    frame of rebuild and is what ``sync`` does on every tab switch anyway.
    """
    state = ensure(ctx)

    def release(tab: ClayTab) -> None:
        # The UV pane's texture is keyed on the tab uid and would outlive the
        # document in the one GL context; the Inker links point into a document
        # that is about to stop existing. Function-scope: ``ui`` imports this
        # module.
        from .ui import _uv_texture

        _uv_texture.release_doc(ctx, tab.uid)
        tab.inker_links.clear()
        view = getattr(ctx, "clay_view", None)
        if view is not None:
            # Not gated on "active tab" the way the rest of this function is:
            # a Familiar ghost (``_preview``/``_preview_scratch``/``_ghost_cache``
            # on ``ClayView``) has no tab identity of its own, so closing *any*
            # tab is a potential staleness event for it -- the 2026-09-14 audit
            # found a background-tab close left a stale preview's GL overlays
            # drawn over whichever tab became active. ``clear_preview()`` is
            # idempotent (a no-op when no preview is showing), so calling it on
            # every close costs nothing.
            view.clear_preview()
        if view is not None and tab.uid == state.active_uid:
            if getattr(view, "dragging", False):
                # Before the clear, and against the document it was started on:
                # a drag holds the mesh it is moving.
                view.cancel_drag(tab.doc)
            view.clear()

    docmodes.close_tab(ctx, state, uid, release)


def release_all(ctx: Any) -> None:
    """Quit's sweep of the GL textures Clay's panes hold, ``plotter_mode.release_all``'s
    twin.

    The 2026-10-07 audit's clay-51: the UV pane's one-texture-per-tab cache has
    a ``release_all`` that nothing called, so quitting released Inker's,
    Plotter's and Packwright's tab textures and left Clay's registered with the
    imgui backend when the viewer's context went away. Function-scope import:
    ``ui`` imports this module.
    """
    from .ui import _uv_texture

    _uv_texture.release_all(ctx)


# --- keys -------------------------------------------------------------------

# Q, then Blender's own G/R/S, which is where a modeller's left hand already
# sits. Held here rather than in the pane so the mapping is testable. **E is
# not a tool key**: it is Extrude's, and a letter doing two jobs by element mode
# was how pressing E in object mode used to flip the tool under the user.
TOOL_KEYS = {
    "q": "select",
    "g": "move",
    "r": "rotate",
    "s": "scale",
}

# The element modes, on the number row. **Not Tab**, which imgui's keyboard
# navigation owns and which would move focus out of the viewport as well as
# changing the mode; and **not b**, because the hand that is about to press
# Ctrl+Z is already on the number row. 4 is object mode rather than a separate
# key, so the four modes are one contiguous run under four fingers.
ELEMENT_KEYS = {"1": "vertex", "2": "edge", "3": "face", "4": "object"}

# Ctrl+digit axis views, on the numbers a modeller's hand already knows from
# Blender's numpad. **Bound here rather than in ``App._shortcut``**: a global
# binding is checked above the workspace modes and takes its key from them
# permanently, which is the whole reason the mode switch moved to Alt. The
# *binding* belongs to Clay; the table and the function do not, and live in
# ``_view_frame`` -- Poser already called this one function rather than
# restating it, and they are imported above under the name both call sites
# already use.
# Ctrl-shortcuts that change the document. Serialising reads the live document
# on a task thread, so anything that restructures it or moves the history head
# the save captured waits for the save, exactly as a gizmo drag does.
_MUTATING_CTRL = docmodes.WRITE_CHORDS | frozenset({"a", "i", "j", "m"})
#: The chords a live drag swallows -- history and the tab; see ``_ctrl_key``.
#: Deliberately *not* the whole of ``_MUTATING_CTRL``: Ctrl+J under a drag is a
#: pinned behaviour (``test_ctrl_chords_still_reach_their_ops_during_a_drag``).
_DRAG_BLOCKED_CTRL = frozenset({"z", "y", "n", "o", "tab", "w", "s"})


# --- history ------------------------------------------------------------------
#
# One call per direction, rather than two lines under the key handler, because
# the header and the Properties pane draw the same Undo/Redo pair Inker's does.
# Clay, Plotter and Packwright each had a full undo stack and no on-screen
# control at all, so the feature existed only for a user who already knew the
# chord -- and every
# side effect a step has (nothing, here) belongs to *undoing*, not to the
# keyboard.


def undo(ctx: Any, tab: Any) -> None:
    """One step back, whichever surface asked for it."""
    tab.doc.undo()



def redo(ctx: Any, tab: Any) -> None:
    """One step forward. :func:`undo`'s twin, and its reasoning."""
    tab.doc.redo()


def step_history(ctx: Any, tab: Any, index: int) -> bool:
    """Jump the document to a position in its undo stack. -> whether it moved.

    The history panel's door, and the *third* surface onto the same stack --
    which is why it is here beside the other two rather than in the pane, and
    why the pane will not call ``doc.step_history`` itself. ``plotter_mode``
    has the same three, for the same reason written out there.

    ``ctx`` is taken and dropped: the parameter stays so the sibling editors'
    ``step_history(ctx, tab, index)`` and this one's one caller
    (the Properties pane's Document tab) share a signature. A jump needs
    no bookkeeping for the recent op (``ClayDoc.recent_op``): the adjust card
    is live only while the history head is the step the op pushed, so moving
    the head hides it by itself.
    """

    del ctx
    return tab.doc.step_history(index)



def _settle_uv_gesture(doc: Any, view_state: Any, *, commit: bool) -> None:
    """End the UV pane's armed live rotate/scale from the key layer --
    ``ui/panes/uv.py``'s ``commit_live_transform``/``cancel_live_transform``
    restated on the same ``UvPaneState`` fields, because this module cannot
    import the pane."""
    mark, kind = view_state.drag_mark, view_state.drag_mode
    view_state.drag_mode = ""
    history = doc.history
    if history.head != mark:
        history.collapse_since(mark)
        if commit:
            top = history.top
            if top is not None:
                top.label = kind.capitalize()
        else:
            history.undo(doc, redoable=False)
    view_state.drag_base = None
    view_state.drag_islands = frozenset()


def _uv_canvas_owns_keys(state: ClayState, uv_view: Any) -> bool:
    """Whether the UV canvas was hovered, islands boxed, on the current frame or
    the one before it (``ClayState.frame_serial``).

    The press is handled in the event layer *before* the frame whose pane would
    arm the gesture, so the freshest stamp there is can be the previous frame's --
    and a pane that stopped drawing (hidden, docked away, the tab switched) lets
    its claim lapse a frame later instead of swallowing the keys. A count of
    frames rather than a reading of the clock: a frame that stalls for a second
    must not let the same press through to the 3-D layer.
    """
    stamp = getattr(uv_view, "key_hover_at", None)
    return stamp is not None and state.frame_serial - stamp <= 1


def _keypad(name: str) -> str | None:
    """The character a keypad key stands for, or ``None`` for any other key.

    ``pygame.key.name`` spells the keypad ``"[1]"``, ``"[.]"``, ``"[-]"``: the
    same characters as the number row and the same meanings in a drag's typed
    value, but a different string, so nothing keyed on ``"1"`` ever saw them.
    """
    if len(name) == 3 and name[0] == "[" and name[2] == "]":
        return name[1]
    return None


def _hide_keys(tab: ClayTab, doc: Any, *, shift: bool, alt: bool) -> None:
    """``H`` hides the selected objects, ``Shift+H`` isolates them, ``Alt+H`` shows all.

    Blender's three, on the same chords. Hiding clears the selection afterwards:
    a hidden object that stays selected keeps its gizmo, its outline and its
    place in the next drag's set while being invisible, which reads as the
    editor acting on things that are not there. Refused while the tab is saving,
    like every control that changes the document.
    """
    if tab.saving:
        return
    if alt:
        doc.show_all()
        return
    chosen = [uid for uid in sorted(doc.selection)]
    if not chosen:
        return
    if shift:
        doc.isolate(chosen)
    else:
        doc.set_visibility({uid: False for uid in chosen})
        if doc.element_mode == "object":
            doc.select([])
        else:
            doc.clear_element_sel()


def handle_key(ctx: Any, event: Any) -> bool:
    """Clay's shortcuts. -> whether the key was consumed.

    **False with nothing open**, which the caller's fall-through depends on:
    Clay owns a viewport, and with no document the viewport's own
    shortcuts must still work. With a document open the key is consumed
    unconditionally, because falling through would let it act on a pane Clay
    mode has replaced.
    """
    import pygame

    if event.type == pygame.KEYDOWN and pygame.key.name(event.key) == "f" and not (
        event.mod & (pygame.KMOD_CTRL | pygame.KMOD_ALT)
    ):
        # Recorded, not done: see ``ClayState.frame_pending``.
        ensure(ctx).frame_pending = True
        return True

    state = ctx.state.clay
    if state is None or not state.docs:
        return False
    tab = state.active
    if tab is None:
        return False
    doc = tab.doc

    if event.type != pygame.KEYDOWN:
        return True

    # Off ``event.mod``, never ``pygame.key.get_mods()`` -- ``main._shortcut``'s
    # rule (UX-12): ``mod`` is the modifier state at the instant this key was
    # pressed, where ``get_mods()`` is the state *now*, after the event batch
    # drained, so a Ctrl released between the two made a fast chord fall
    # through as the bare letter.
    mods = event.mod
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    alt = bool(mods & pygame.KMOD_ALT)
    name = pygame.key.name(event.key)

    # A live gizmo drag owns the bare keys, and it has to be asked *first*: the
    # number row is bound to the element modes, so a "1" typed into a drag would
    # otherwise jump into vertex mode halfway through moving something. Esc
    # cancels the drag rather than falling through to the staged clear, and
    # Enter commits it -- neither means anything else while one is under way.
    view = getattr(ctx, "clay_view", None)
    if not ctrl and view is not None and getattr(view, "dragging", False):
        if event.key == pygame.K_ESCAPE:
            view.cancel_drag(doc)
            return True
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            view._release_drag(doc)
            return True
        # The keypad types into a drag as the row does: ``"[5]"`` is a 5.
        view.drag_key(doc, _keypad(name) or name)
        # Consumed whether or not the drag wanted it. Falling through here put
        # every unclaimed bare key into the op registry below, so ``E`` typed
        # mid-``G`` ran Extrude against the mesh the drag was still moving --
        # and the drag's own commit then measured from ``_drag_start``, the
        # pre-drag baseline, and reverted it. The rule the comment above states
        # is only a rule if it holds for the keys the drag does *not* know.
        return True

    # The 2026-10-03 audit's clay-18: an armed UV live rotate/scale is a drag
    # too, and its own Esc check lives in a pane that returns early with
    # nothing selected -- so Esc here ran the staged clear (deselecting the
    # object) and the gesture stayed armed with its undo gesture open; E/R
    # flipped the 3-D tool and E ran Extrude. Duck-typed on ``tab.uv_view`` (the
    # mode root may not import ``ui/``): Esc cancels, Enter commits, and every
    # other bare key is consumed, exactly as for a 3-D drag above.
    uv_view = getattr(tab, "uv_view", None)
    if not ctrl and uv_view is not None and uv_view.drag_mode in ("rotate", "scale"):
        if event.key == pygame.K_ESCAPE:
            _settle_uv_gesture(doc, uv_view, commit=False)
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            _settle_uv_gesture(doc, uv_view, commit=True)
        return True

    # The 2026-10-07 audit's clay-03: the shell routes every KEYDOWN here without
    # asking which pane the pointer is over, and the UV canvas arms its own live
    # rotate/scale off the same R/S press -- so one press started the 3-D drag
    # (and switched the 3-D tool) *and* armed the UV gesture. The pane records each
    # frame it is hovered with islands boxed (``key_hover_at``; this module may not
    # import ``ui/``), and a bare R/S on that frame or the next is the pane's alone.
    if (
        uv_view is not None
        and name in ("r", "s")
        and not (ctrl or alt or shift)
        and _uv_canvas_owns_keys(state, uv_view)
    ):
        return True

    pad = _keypad(name)
    if pad is not None and not alt:
        # Blender's numpad views: 1 front, 3 right, 7 top (Shift the opposite
        # side), 5 orthographic, "." frame -- with or without Ctrl, since the
        # keypad has no digit-row meaning for Ctrl to disambiguate. Anything
        # else on the pad is consumed and does nothing, as every bare key is.
        if pad == ".":
            state.frame_pending = True
        elif view is not None and (pad in AXIS_VIEW_KEYS or pad == "5"):
            axis_view_key(view.camera, pad, shift)
        return True

    if ctrl:
        return _ctrl_key(ctx, state, tab, doc, name, shift=shift)

    # Before the digit/registry/tool chain below, so the chord is its own and a
    # future bare binding on H cannot shadow it.
    if name == "h":
        _hide_keys(tab, doc, shift=shift, alt=alt)
        return True

    if alt and name == "z":
        # The 2026-09-07 audit's clay-08: the X-ray button's own tooltip
        # (``studio/modes/clay/ui/panes/header.py``) has named "(Alt+Z)" since it was added,
        # but nothing bound the chord and this handler simply consumed the
        # press with no effect. Wired rather than the tooltip's claim
        # dropped: the toggle it names already exists (``state.xray``, the
        # same field the button flips) and is not gated on ``saving`` there
        # either -- it is a viewport setting, not a document edit.
        state.xray = not state.xray
        return True

    if name in ELEMENT_KEYS and not shift:
        if not tab.saving:
            doc.set_element_mode(ELEMENT_KEYS[name])
    elif shift and not alt and _registry_key(ctx, tab, doc, name, shift=True):
        # Shift+letter registry bindings (Repeat Last is Shift+R). A separate
        # branch because the bare-letter one below is ``not shift`` -- Shift+R
        # must not also reach the Scale tool's letter -- and it fires through
        # the same ``by_key`` the menu prints, so the binding shown and the
        # binding that fires stay one value.
        pass
    elif not shift and (
        # The registry first, and the ``or`` short-circuits, so a letter an
        # element mode has claimed still fires its op rather than starting a
        # drag -- which is the ordering ``DRAG_KEYS`` records the reason for.
        _registry_key(ctx, tab, doc, name)
        or _keyboard_drag(ctx, view, tab, doc, name)
    ):
        pass
    elif name in TOOL_KEYS and not shift:
        state.tool = TOOL_KEYS[name]
    elif event.key == pygame.K_DELETE:
        if not tab.saving:
            # The 2026-09-07 audit's clay-02: this used to call
            # ``selection.delete_selected`` directly, so a face selection
            # spanning two objects pushed one ``set_mesh`` per object and one
            # Ctrl+Z only undid the last one. ``clay_ops.run`` folds whatever
            # an op pushes into a single step for every caller -- the menu's
            # "Delete" row already went through it, so the keyboard path is
            # the one being brought in line rather than a special case.
            from . import ops as clay_ops

            clay_ops.run(ctx, doc, clay_ops.get("delete"))
    elif event.key == pygame.K_ESCAPE:
        _escape(state, tab, doc)
    return True



#: The three letters that start a transform with no handle grabbed, and what
#: each starts. They are also the Move, Rotate and Scale tools' letters
#: (``TOOL_KEYS``): pressing one selects the tool *and*, with something selected,
#: begins the drag, so the gizmo and the gesture always agree about which
#: transform is running.
#:
#: Checked *after* the op registry, so a letter an element mode has claimed
#: still fires its op: none of G, R or S is in the registry bare today (Repeat
#: Last is Shift+R), and the ordering is what keeps that from being a thing to
#: remember if one ever is.
DRAG_KEYS = {"g": "move", "r": "rotate", "s": "scale"}


def _keyboard_drag(ctx: Any, view: Any, tab: ClayTab, doc: Any, name: str) -> bool:
    """Start a keyboard transform. -> whether the key was one.

    Refused while the tab is saving, like every control that changes the
    document -- and refused with nothing selected, where it would be a drag with
    nothing to drag and would swallow a keystroke that means nothing else.
    """

    kind = DRAG_KEYS.get(name)
    if kind is None or view is None or tab.saving:
        return False
    # The tool first, so the gizmo the drag begins over is the one its key names;
    # a drag that is refused (nothing selected) still leaves the tool changed,
    # which is the same thing the tool branch in ``handle_key`` does for it.
    state = getattr(ctx.state, "clay", None)
    if state is not None:
        state.tool = TOOL_KEYS[name]
    return bool(view.begin_keyboard_drag(doc, kind))


def _registry_key(
    ctx: Any, tab: ClayTab, doc: Any, name: str, *, shift: bool = False
) -> bool:
    """Fire the registry op bound to a bare letter, if there is one.

    Checked *before* the tool keys so an element mode can claim a letter the
    transform tools also use -- none does today (E is Extrude's alone), but the
    ordering keeps a future bare letter from being shadowed by a tool -- and
    checked through ``clay_ops.menu`` so the binding shown in the context menu
    and the binding that fires are one value.
    """
    from . import ops as clay_ops

    if tab.saving or doc.element_mode == "object":
        return False
    op = clay_ops.by_key(doc.element_mode, ("Shift+" if shift else "") + name.upper())
    if op is None or not op.enabled(doc):
        return False
    return _fire_op(ctx, doc, op, interactive=True)


def _fire_op(ctx: Any, doc: Any, op: Any, *, interactive: bool = False) -> bool:
    """Run a registry op from the event layer, popping its dialog if it has one.

    Shared by the bare-letter path and the Ctrl-shortcut path so a
    parameterised op bound to either kind of key behaves the same way.

    ``interactive`` is the *keyboard's* door and nothing else's: an op with a
    ``drag`` spec starts the viewport's live op drag instead of the dialog, and
    Extrude hands straight over to a move along the face normal. The menu strip
    reaches this function too (``fire_op``) and passes nothing, so a menu click
    keeps the dialog -- the path for an exact value.
    """
    from . import ops as clay_ops

    state = ensure(ctx)
    if interactive and op.name == "extrude":
        return _extrude_and_drag(ctx, doc, op)
    if interactive and op.drag is not None:
        view = getattr(ctx, "clay_view", None)
        if view is not None and view.begin_op_drag(doc, op, state.op_params.get(op.name)):
            return True
    if op.params:
        state.pending_op = op.name
        state.op_params.setdefault(op.name, clay_ops.defaults_for(op))
        # Asked for rather than opened: ``imgui.open_popup`` only takes effect
        # inside the window whose id stack is current, and this runs in the
        # event layer.
        state.open_op_popup = True
        return True
    return clay_ops.run(ctx, doc, op)


def _extrude_and_drag(ctx: Any, doc: Any, op: Any) -> bool:
    """``E``: extrude, then at once drag what it made, as one undo step.

    The mark is taken **before** the extrude so commit can fold the two into
    one step and Esc can undo both (``DragOps.begin_extrude_drag``). With no
    viewport, or a grab already live, or nothing for the drag to move, it is
    the plain extrude it always was -- the fold is closed on every path, since an
    open gesture switches the undo budget off for the document.
    """
    from . import ops as clay_ops
    from . import recent_op

    view = getattr(ctx, "clay_view", None)
    if view is None or view.grabbing:
        return clay_ops.run(ctx, doc, op)
    history = doc.history
    before = recent_op.snapshot(doc)
    mark = history.mark()
    head = history.head
    ran = clay_ops.run(ctx, doc, op)
    if ran and history.head != head and view.begin_extrude_drag(doc, mark, before):
        return True
    history.collapse_since(mark)
    return ran


fire_op = _fire_op
"""The same door, named for the menu strip: a submenu row asks for its dialog by
state (``open_op_popup``) exactly as a key does, because an ``imgui.open_popup``
made inside a submenu names the popup in the submenu's own id stack."""


def _escape(state: ClayState, tab: ClayTab, doc: Any) -> None:
    """Esc, staged: the elements, then the mode, then the objects.

    One key that undoes the last thing the user got into, in the order they got
    into it. It **never leaves Clay mode**: Esc means "drop what I am doing",
    and losing a workspace full of tabs to a stray keypress is not that.
    """
    if not tab.saving:
        if doc.element_mode != "object":
            if doc.element_sel:
                doc.clear_element_sel()
            else:
                doc.set_element_mode("object")
        else:
            doc.select([])
    # Always: abandoning a half-finished drag touches the pane's own state and
    # never the document, so it is safe mid-save.
    state.clear_drag()


def _toast(ctx: Any, message: str) -> None:
    """``docmodes.refuse``: the one refusal door the non-Inker modes share."""
    docmodes.refuse(ctx, message)


def _ctrl_key(
    ctx: Any, state: ClayState, tab: ClayTab, doc: Any, name: str, *, shift: bool
) -> bool:
    if docmodes.blocked_while_writing(tab, name, _MUTATING_CTRL):
        return True
    view = getattr(ctx, "clay_view", None)
    if getattr(view, "dragging", False) and name in _DRAG_BLOCKED_CTRL:
        # A live gizmo drag mutates the geometry in place and commits on
        # release. Ctrl+Z under it undid a step the release then re-applied;
        # Ctrl+N/Tab/O/W swapped the tab out from under the grab and left the
        # moved vertices with no edit and ``dirty`` false. The chords that
        # change history or the tab wait for the release; the view chords and
        # the object ops go on working.
        return True
    if name in AXIS_VIEW_KEYS or name == "5":
        view = getattr(ctx, "clay_view", None)
        if view is not None:
            axis_view_key(view.camera, name, shift)
    elif name == "s":
        save_as(ctx, tab) if shift else save(ctx, tab)
    elif name == "o":
        ask_open(ctx)
    elif name == "n":
        new_document(ctx)
    elif name == "w":
        # Beside its siblings, and the same key Inker, Plotter and Packwright
        # already close a document with.
        close_tab(ctx, tab.uid)
    elif name == "e" and not shift:
        # Not the shifted half: Ctrl+Shift+E was a silent alias of Ctrl+E here,
        # which is the one thing a printed binding exists to prevent
        # (``inker_keys``'s rule). Clay's file export *is* its library export,
        # so the chord every other mode gives the file export has nothing
        # distinct to do here and does nothing.
        export_asset(ctx, tab)
    elif name == "z":
        redo(ctx, tab) if shift else undo(ctx, tab)
    elif name == "y":
        redo(ctx, tab)
    elif name == "a":
        _select_all(doc)
    elif name == "i" and shift:
        _invert(doc)
    elif name == "m" and doc.element_mode == "object":
        # **Merge is Ctrl+M, and Duplicate is Ctrl+J.** Clay used to be the
        # editor that disagreed with the other three about both keys: Ctrl+D
        # duplicated here and deselects in Inker and Plotter, and Ctrl+J merged
        # here and duplicates in Plotter (whose comment names the raster
        # editor's "copy this to its own layer" as the same idea). Two chords
        # meaning two different things in two workspaces of one app is a user
        # pressing the one they learned and getting the other verb.
        #
        # Object mode only, for the reason Ctrl+J is: a merge is about whole
        # objects, and there is no element-mode reading of it to fall back on.
        from . import ops as clay_ops

        op = clay_ops.get("join")
        if op.enabled(doc):
            _fire_op(ctx, doc, op)
    elif name == "j" and doc.element_mode == "object":
        # Object mode only: duplicating a *face* selection is a different
        # operation with a different name, and doing the object one instead
        # would silently double a mesh the user is mid-edit on.
        _duplicate_selection(ctx, state, doc)
    elif name == "d" and not tab.saving:
        # Deselect, which is what it does in Inker and in Plotter. Not gated on
        # the element mode: clearing a selection means something in all four.
        # Unlike Esc it is *only* the deselect -- no mode step, no drag cancel
        # -- because a chord a user reaches for deliberately should do one
        # thing, and the staged key already exists for the other reading.
        if doc.element_mode != "object":
            doc.clear_element_sel()
        else:
            doc.select([])
    elif name == "tab":
        state.cycle(-1 if shift else 1)
    elif not tab.saving and not getattr(view, "dragging", False):
        # A registry op bound to a Ctrl chord the cases above do not own. Last,
        # so an existing chord can never be taken by a later registration, and
        # not under a live drag, which swallows the keys it does not know.
        from . import ops as clay_ops

        label = ("Ctrl+Shift+" if shift else "Ctrl+") + name.upper()
        op = clay_ops.by_key(doc.element_mode, label)
        if op is not None and op.enabled(doc):
            _fire_op(ctx, doc, op, interactive=True)
    return True


# The two below are thin wrappers on ``clay.selection``, kept at their old
# names and signatures because the panes and the tests call them by those
# names. The behaviour and the reasoning are in that module. ``_duplicate_
# selection`` below them used to be a third, but the 2026-09-07 audit's
# clay-07 routed it through the op registry instead -- see its own docstring.


def _select_all(doc: Any) -> None:
    """``clay.selection.select_all``: everything visible, in the current mode."""
    from ....kernels.mesh import selection

    selection.select_all(doc)


def _invert(doc: Any) -> None:
    """``clay.selection.invert``: Ctrl+Shift+I, in the current mode."""
    from ....kernels.mesh import selection

    selection.invert(doc)


def _duplicate_selection(ctx: Any, state: ClayState, doc: Any) -> None:
    """Duplicate, run through the op registry rather than ``clay.selection``
    directly.

    Both the Ctrl+J handler above and the outliner's context-menu "Duplicate"
    row (``studio/modes/clay/ui/panes/outliner.py``) call this by name. Before the 2026-09-07
    audit's clay-07 it called ``selection.duplicate_selected`` straight, so
    the edit landed in history under ``add_objects``'s generic "object add"
    label instead of "Duplicate". (A bare action like Duplicate is not the
    recent op -- only a parameterised element op is, see ``recent_op``.)
    Routing both callers through this one function is what lets fixing it
    here fix the outliner's row too without touching that pane.

    ``state`` is taken and dropped: it was never read, and the callers pass
    it, so the parameter stays rather than becoming a rename.
    """
    from . import ops as clay_ops

    del state
    op = clay_ops.get("duplicate")
    if op.enabled(doc):
        _fire_op(ctx, doc, op)


# --- crash recovery (UX-05) ---------------------------------------------------
#
# The mechanism is :mod:`studio.journal`'s; these are the four answers that are
# about *models*. See that module for the loop, the debounce, the head gate and
# the completion gate, and for the rule they all serve: a journal entry is never
# a save.


def _journal_snapshot(tab: Any) -> Any:
    """The journal provider's encoder: the cheap half now, the archive later.

    **Frame thread: ``serialize.snapshot`` only.** The 2026-10-07 audit's
    clay-24: ``rblk_bytes`` (the zip, one npz per mesh, one PNG per texture) ran
    here on every autosave -- 377 ms for two 100k-triangle objects against
    0.1 ms for the snapshot -- though :attr:`journal.Provider.encode` may return
    a zero-argument callable that the write's task runs (Packwright's does). The
    snapshot holds references to immutable meshes and a camera already turned
    into a dict, so the closure never touches the live document. An oversize
    document's ``ValueError`` (``snapshot_bytes``'s object and triangle ceilings)
    now surfaces when the task calls it, which the shell reports as the failed
    autosave it is; the debounce has already moved, so it retries in
    ``JOURNAL_SECONDS`` rather than every frame.

    The camera goes in for the same reason a save carries it: a recovered model
    that framed itself somewhere else is a recovered model the user has to find
    their way back around. But **only a camera that has been framed**
    (``clay-60``): a tab nothing has drawn yet still holds ``CameraView``'s
    defaults, and writing those made recovery adopt them as a stored view and
    mark the tab fitted, so the recovered model was never auto-framed.
    """
    from ....kernels.mesh import serialize

    snap = serialize.snapshot(tab.doc, view=tab.view if tab.view.fitted else None)
    return lambda: serialize.snapshot_bytes(snap)


def _journal_encode(tab: Any) -> bytes:
    """:func:`_journal_snapshot` and its archive in one call -- for a caller
    already off the frame thread (a test, a batch tool)."""
    return _journal_snapshot(tab)()


def _journal_adopt(ctx: Any, path: Path, meta: dict[str, Any]) -> bool:
    """Reopen one recovered ``.rblk`` as an *untitled, dirty* document.

    Untitled for Inker's reason: the file it was copied from may still be on
    disk with its own contents, and adopting the path would arm Ctrl+S to
    overwrite something the user has not looked at.
    """
    ensure(ctx)
    # Read and parsed on a task, ``inker_mode``'s ``inker-recover`` shape: a
    # recovered model is read on the first frame of the session, which is
    # the frame the Home pane is meant to appear on, and a ``.rblk`` is as
    # large as the model it holds. True means "submitted", the same answer the
    # Inker provider gives; ``on_task_done`` does the adopting.
    ctx.submit(f"clay-recover:{_path_key(path)}", _load_recovery, Path(path), dict(meta))
    return True


def _load_recovery(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """The task-thread half of a crash recovery: bytes to document."""
    from ....kernels.mesh import serialize

    try:
        data = _within_ceiling(path)
        doc = serialize.read_rblk(data)
    except Exception:
        # ``None`` rather than a raise: the landing turns it into the one
        # sentence every provider says (``journal.adopt_failed``), where a
        # raise arrived as an *error* toast that no other mode's copy raised.
        log.exception("could not reopen the recovered model at %s", path)
        return None
    return {
        "doc": doc,
        "title": f"{meta.get('title') or path.stem} (recovered)",
        "autosave": str(path),
        # The 2026-10-03 audit, finding clay-110: ``_journal_encode`` writes the
        # camera "for the same reason a save carries it", but this half never
        # read it back, so a recovered model was auto-framed anyway. Same
        # second read of the same in-memory bytes as ``_load``.
        "view": serialize.read_view(data),
    }


JOURNAL = journal.tab_provider(
    "clay", ".rblk", "model", encode=_journal_snapshot, adopt=_journal_adopt
)
