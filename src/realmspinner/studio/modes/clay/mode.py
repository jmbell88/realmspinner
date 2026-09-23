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
    view = getattr(ctx, "clay_view", None)
    if view is None or not getattr(view, "dragging", False):
        return
    tab = state.get(old_uid)
    if tab is not None:
        view.settle_drag(tab.doc)


def _restore_view(state: ClayState, stored: Any) -> None:
    """The grid/god-light preferences back off disk, clamped rather than
    trusted -- ``inker_mode._restore_canvas``'s doctrine for the same reason:
    a hand-edited ``settings.json`` can hold anything JSON allows."""
    if not isinstance(stored, dict):
        return
    state.grid = bool(stored.get("grid", state.grid))
    size = stored.get("grid_size")
    if isinstance(size, int | float) and not isinstance(size, bool):
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


# --- export engine (tranche 7: export profiles) ------------------------------
#
# **Lives in app settings, not on ``ClayTab``/``ClayState``.** The brief for
# this tranche asked to check that first, since ``state.py`` was concurrently
# owned by another agent while this was built: a per-tab field would have
# meant either touching that file anyway or grafting the setting onto
# ``ClayTab`` from outside it, and there is nothing document-specific about
# "which engine do my exports target" to begin with -- it is a preference
# about the *person's* project, the same kind of thing ``grid``/``grid_size``/
# ``god_light`` already are (see :func:`persist`'s own docstring), and
# unlike those three it has to be readable with no ``ClayState`` built yet
# (:func:`export_engine` is called from :func:`export_asset`/
# ``export_mesh_file`` before ``ensure`` runs). Stored under the same
# ``"clay"`` settings block those three use, as a sibling key to ``"view"``
# rather than a second top-level settings entry -- one block for "Clay's own
# app-level preferences" rather than an ever-growing set of bare keys.


def _default_export_engine() -> str:
    """The engine :func:`export_engine` answers with when nothing has been
    chosen yet -- whichever :data:`~.engines.ENGINES` entry targets
    ``readiness.DEFAULT_PROFILE``, read off that one mapping rather than a
    second hard-coded default this tranche's brief explicitly warns against
    ("the engine choice must not become a second list -- ``ENGINES[key].
    readiness_profile`` is the one mapping, gated both ways";
    ``test_engines.py``'s own ``test_every_engine_profile_points_at_a_real_
    readiness_profile`` is that gate's other direction). Today that is
    ``"godot4"`` (``readiness.DEFAULT_PROFILE == "godot-desktop"``), but this
    reads it rather than states it, so the two can never quietly disagree.
    """
    from ....kernels.mesh import engines as engines_mod
    from ....kernels.mesh import readiness

    for key, eng in engines_mod.ENGINES.items():
        if eng.readiness_profile == readiness.DEFAULT_PROFILE:
            return key
    # Unreachable while ``test_every_engine_profile_points_at_a_real_
    # readiness_profile`` holds *and* some engine targets the default
    # profile -- both true today -- but a caller still gets a real engine
    # key rather than ``None`` if that ever stops being so.
    return next(iter(engines_mod.ENGINES))


def export_engine(ctx: Any) -> str:
    """The persisted export engine choice, read fresh from settings rather
    than cached anywhere -- a value Settings' own combo writes (once built;
    this tranche is the door, not that pane -- see the module docstring's
    file list) must be visible here with no second copy to fall out of sync.

    Always a real :data:`~.engines.ENGINES` key: an unrecognised or missing
    stored value (a hand-edited ``settings.json``, or nothing chosen yet)
    falls back to :func:`_default_export_engine` rather than ``None`` --
    export always targets *some* engine's conventions, the same "there is
    always a profile" contract ``ClayTab.readiness_profile`` keeps by falling
    back to ``readiness.DEFAULT_PROFILE``.
    """
    from ....kernels.mesh import engines as engines_mod

    settings = getattr(ctx, "settings", None)
    if callable(getattr(settings, "get", None)):
        stored = settings.get("clay")
        key = stored.get("export_engine") if isinstance(stored, dict) else None
        if isinstance(key, str) and key in engines_mod.ENGINES:
            return key
    return _default_export_engine()


def set_export_engine(ctx: Any, key: str) -> None:
    """Persist *key* as the export engine choice. Refuses an unknown engine
    by name rather than silently storing a value :func:`export_engine` would
    then have to fall back past."""
    from ....kernels.mesh import engines as engines_mod

    if key not in engines_mod.ENGINES:
        choices = ", ".join(sorted(engines_mod.ENGINES))
        raise ValueError(f"Unknown engine profile {key!r}. Choose one of {choices}.")
    settings = getattr(ctx, "settings", None)
    if not callable(getattr(settings, "set", None)):
        return
    stored = settings.get("clay") if callable(getattr(settings, "get", None)) else None
    block = dict(stored) if isinstance(stored, dict) else {}
    block["export_engine"] = key
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


def adopt(
    ctx: Any,
    doc: Any,
    *,
    path: Path | None = None,
    title: str | None = None,
    view: dict[str, Any] | None = None,
    rblk_bytes: int = 0,
) -> ClayTab:
    state = ensure(ctx)
    tab = ClayTab(
        doc=doc,
        title=title or clay_state.title_for(path),
        path=path,
        saved_head=doc.history.head,
        rblk_bytes=rblk_bytes,
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
        # Free here -- the file is already fully in memory -- and it is what
        # lets ``.generate``'s own too-big-to-reopen check answer for a
        # document that has never been saved *this session* without an encode
        # of its own.
        "rblk_bytes": len(data),
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
    for raw in data.decode("utf-8", errors="replace").splitlines():
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
            return candidate.read_text(encoding="utf-8", errors="replace")
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

        sidecar = svc_files.clay_source_path(ctx.svc, job_id)
        if sidecar.exists():
            return _load(sidecar)
        mesh = ctx.svc.config.job_dir(job_id) / "model.glb"
        if not mesh.exists():
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
        data = serialize.snapshot_bytes(snap)
        _refuse_oversized_save(data)
        atomic.write_bytes(path, data)
        return {"rev": rev, "path": str(path), "retitle": True, "rblk_bytes": len(data)}

    _start(ctx, tab, f"clay-save:{tab.uid}", run)


def save(ctx: Any, tab: ClayTab | None = None) -> None:
    tab = tab or active(ctx)
    docmodes.save(
        tab, save_as=lambda: save_as(ctx, tab), save_to=lambda: save_to(ctx, tab, tab.path)
    )


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
        path = path.with_suffix(clay_state.RBLK_SUFFIX)
        # Staged, as ``save_to`` is: a picker aimed at an existing document is
        # the ordinary way to overwrite one, and a write that dies partway
        # through would leave that file truncated with no copy of it anywhere.
        # No mkdir -- the picker returns a directory that exists.
        data = serialize.snapshot_bytes(snap)
        _refuse_oversized_save(data)
        atomic.write_bytes(path, data)
        return {"rev": rev, "path": str(path), "retitle": True, "rblk_bytes": len(data)}

    _start(ctx, tab, f"clay-saveas:{tab.uid}", run)


# --- export -----------------------------------------------------------------


def _rename_collider_nodes(doc: Any, model: Any, engine: str) -> None:
    """Rewrite *model*'s collider node names to *engine*'s own convention, in
    place -- the GLB half of tranche 7's export-profiles rule 2, mirroring
    what :func:`~.objexport.claydoc_to_obj` already does for OBJ text.

    ``document.to_model`` builds one ``gltf.Node`` per *kept* object -- every
    visible object, plus every hidden object with a visible descendant
    (its own docstring's "hiding is per object, as in Blender" rule) -- in
    that same order, so zipping that "kept" list against ``model.nodes`` is
    exactly how to tell which node came from which :class:`~.document.Obj`.
    That filter is :func:`~.document.kept_objects` -- called here, not
    re-derived, since the 2026-09-19 audit's clay-20: this function used to
    hand-copy ``to_model``'s own five-line filter because ``document.py`` was,
    at the time, a file a concurrent agent owned and this tranche's brief
    said not to touch. Two copies of "which objects become nodes" is exactly
    what "One conversion out, three consumers" exists to prevent -- a future
    edit to one copy and not the other would silently misalign this zip and
    rename a collider node belonging to the wrong object -- so now that both
    files are owned together, this calls the one the exporter and the
    viewport already agree on rather than keeping a second copy in sync by
    hand.

    Only a node whose object is a collider is touched
    (:func:`~.objexport.collider_export_names` only ever returns collider
    uids) -- an ordinary mesh node's name is exactly what ``to_model`` gave
    it, ``obj.name``, untouched. The *document*'s own ``Obj.name`` is never
    written to: only ``model.nodes[i].name``, on the in-memory
    :class:`~.geom3d.gltf.Model` this call's caller is about to hand to
    :func:`~.geom3d.glbwrite.write_glb` and throw away.
    """
    from ....kernels.mesh import document as bd
    from ....kernels.mesh import objexport

    names = objexport.collider_export_names(doc, engine)
    if not names:
        return
    kept = bd.kept_objects(doc)
    for obj, node in zip(kept, model.nodes, strict=True):
        new_name = names.get(obj.uid)
        if new_name is not None:
            node.name = new_name


def build_asset(
    svc: Any,
    doc: Any,
    *,
    title: str,
    prompt: str | None = None,
    view: Any = None,
    engine: str | None = None,
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

    ``engine``, tranche 7: when given, every collider node in the written GLB
    is renamed to that engine's own convention (:func:`_rename_collider_nodes`)
    before it is encoded -- the document's own object names are untouched, as
    always. ``None`` (the default, and what the agent's own call above still
    passes) skips that step entirely, reproducing this function's
    pre-tranche-7 behaviour: the agent surface for an engine choice on export
    is a later tranche's own door (see this tranche's brief, item 5), not
    this one.

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
    if engine is not None:
        _rename_collider_nodes(doc, model, engine)
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
    so it is read before the task runs rather than inside it), the persisted
    export engine (:func:`export_engine`, read on the frame thread beside the
    camera -- settings, like the camera, do not belong inside ``run()``), and
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
    engine = export_engine(ctx)

    def run() -> dict[str, Any]:
        job_id = build_asset(ctx.svc, doc, title=title, view=view, engine=engine)
        return {"job_id": job_id, "exported": True}

    _start(ctx, tab, f"clay-export:{tab.uid}", run)


OBJ_FILTER = ["Wavefront OBJ (*.obj)", "*.obj"]
IMPORT_MESH_FILTER = [
    "Mesh files (*.glb *.obj *.stl *.ply)",
    "*.glb *.obj *.stl *.ply",
]


def export_mesh_file(ctx: Any, tab: ClayTab | None, kind: str) -> None:
    """Save the document as a plain mesh file on disk -- GLB or OBJ+MTL --
    beside :func:`export_asset`'s library export.

    The two are genuinely different destinations, the bridge pane's own
    reasoning for keeping "Export to Library" and "Make 3D" apart applies
    here too: this writes a file the library never sees, for a user handing a
    mesh straight to another tool.

    The document read (``bd.to_model``/``objexport.claydoc_to_obj``, both
    reads of the live ``doc``) happens *before* the picker opens, on the frame
    thread, the same ordering ``save_as``'s own docstring explains: an
    unbounded modal dialog is exactly the moment a read must not straddle. The
    encode was already done by the time the read happened -- ``to_model``
    and ``claydoc_to_obj`` both return plain data, not bytes, so this is the
    same "read the document, encode what it read" split every task-thread
    closure in this module keeps, just landing after the picker for the OBJ
    case's own reason (below) instead of before it.

    Tranche 7: both branches export under the persisted :func:`export_engine`
    -- a collider is renamed to that engine's own convention either way
    (:func:`_rename_collider_nodes` for GLB, ``objexport.claydoc_to_obj``'s
    own ``engine`` argument for OBJ), and the OBJ branch alone also applies
    that engine's axis/scale conversion (``claydoc_to_obj``'s own docstring
    says why GLB does not).
    """
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, title = tab.doc, tab.title
    if not any(obj.visible for obj in doc.objects):
        ctx.toast("There is nothing visible to export.", "error")
        return

    engine = export_engine(ctx)

    if kind == "glb":
        from ....kernels.geom3d import glbwrite
        from ....kernels.mesh import document as bd

        model = bd.to_model(doc)
        _rename_collider_nodes(doc, model, engine)
        data = glbwrite.write_glb(model)

        def run() -> dict[str, Any] | None:
            path = dialogs.save_file("Export mesh", f"{title}.glb", dialogs.GLB_FILTER)
            if path is None:
                return None
            path = path.with_suffix(".glb")
            atomic.write_bytes(path, data)
            return {"path": str(path), "exported_file": True}

    elif kind == "obj":
        from ....kernels.mesh import objexport

        obj_text, mtl_text = objexport.claydoc_to_obj(doc, name=title, engine=engine)

        def run() -> dict[str, Any] | None:
            path = dialogs.save_file("Export mesh", f"{title}.obj", OBJ_FILTER)
            if path is None:
                return None
            path = path.with_suffix(".obj")
            stem = path.stem
            # ``claydoc_to_obj`` wrote ``mtllib {title}.mtl`` against the tab's
            # own title, which the save dialog is free to have renamed --
            # kept in sync here rather than re-reading the document with the
            # chosen name, so the obj always names the mtl actually beside it.
            text = obj_text.replace(f"mtllib {title}.mtl", f"mtllib {stem}.mtl", 1)
            atomic.write_bytes(path, text.encode("utf-8"))
            atomic.write_bytes(path.with_name(f"{stem}.mtl"), mtl_text.encode("utf-8"))
            return {"path": str(path), "exported_file": True}

    else:
        raise ValueError(f"unknown mesh export kind {kind!r}")

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
                    rblk_bytes=int(result.get("rblk_bytes") or 0),
                )
            _enter_clay(ctx)
        return

    if name == "clay-recover":
        if result is None:
            journal.adopt_failed(ctx, "model")
        if isinstance(result, dict):
            tab = adopt(ctx, result["doc"], path=None, title=result.get("title"))
            docmodes.mark_recovered(tab, result["autosave"])
            _enter_clay(ctx)
        return

    if name == "clay-import":
        # No path: an imported document has no file of its own, so Ctrl+S asks
        # where to put it rather than overwriting the asset it came from.
        if isinstance(result, dict):
            _adopt_import(ctx, result)
        return

    if name == "clay-readiness":
        # The 2026-09-20 audit's clay-05: check_readiness now backgrounds
        # readiness.validate rather than calling it on the frame thread, so
        # the result lands here instead of directly on the tab. A closed tab
        # (state.get returns None) has nowhere to put it -- the same "the tab
        # may be gone by the time this comes back" reading every other
        # per-tab task key in this function already has to handle.
        tab = state.get(key.split(":", 1)[1]) if ":" in key else None
        if tab is not None:
            from ....kernels.mesh import readiness

            if isinstance(result, readiness.Report):
                tab.readiness_report = result
                # ``done.tag`` is the head check_readiness captured *before*
                # submitting -- see that function's own comment on why this
                # is read from the tag rather than from tab.doc.history.head
                # now that the call is asynchronous.
                tab.readiness_head = done.tag
        return

    from . import generate as clay_generate

    if name in clay_generate.TASK_KEYS:
        # "Generate into the current tab" (``.generate``): three task keys
        # that never touch ``saving`` -- a generate in flight leaves the
        # document editable, the ``clay-bg`` precedent just below. Routed
        # here, before the generic tail below, for the same reason ``clay-bg``
        # is: that tail unconditionally clears ``tab.saving`` for *any*
        # ``clay-*:<uid>`` key, which would unlock a tab mid-save the moment a
        # refused or deferred landing happened to land on one that was
        # genuinely saving.
        clay_generate.on_task_done(ctx, done)
        return

    if name == "clay-bg":
        # Clay's background ops -- Decimate (tranche 1) and, since tranche 4,
        # Retopologize/Smart Unwrap/Bake Detail -- all land here; see
        # ``clay_ops``'s own section docstrings for the ``prepare``/``work``/
        # ``apply`` shape they share. Not folded into the generic
        # ``tab.saving`` tail below: this key's result shapes (``{"items":
        # ...}``/``{"glb_out": ...}``/``{"error": ...}``) are nothing like a
        # save's, and nothing here should touch ``saving`` at all -- a
        # background op in flight leaves the document editable, which is the
        # feature.
        #
        # Which of the four ``apply`` functions to run is read off the
        # result's own ``"kind"`` -- decimate's own result carries none
        # (tranche 1 predates the other three), so it stays the default.
        tab = state.get(key.split(":", 1)[1]) if ":" in key else None
        if tab is not None:
            tab.bg_busy = ""
            from . import ops as clay_ops

            kind = result.get("kind") if isinstance(result, dict) else None
            apply = {
                "retopo": clay_ops.retopo_apply,
                "unwrap": clay_ops.unwrap_apply,
                "bake": clay_ops.bake_apply,
            }.get(kind, clay_ops.decimate_apply)
            apply(ctx, tab.doc, result)
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
    if "rblk_bytes" in result:
        tab.rblk_bytes = int(result["rblk_bytes"])
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

    A failed ``clay-bg`` task clears ``bg_busy`` the same way, as a safety
    net: ``clay_ops._decimate_work`` catches its own ``OptimizeError`` and
    returns it as an ordinary (non-failed) result specifically so
    :func:`on_task_done` can toast the real gltfpack message rather than the
    generic one this path shows, so this branch exists for whatever an
    unexpected exception past that catch would otherwise leave stuck.

    A failed ``clay-gen*`` task is routed to :mod:`.generate` instead of
    falling into the generic body below, and for a stronger reason than
    either of those two: this function's own unconditional ``tab.saving =
    False`` would unlock a tab that is genuinely mid-save the moment a
    generate task happens to fail while that save is in flight on the same
    tab -- a refused or interrupted generate must never be what makes a save
    that is still running look finished.
    """
    state = ctx.state.clay
    if state is None or ":" not in done.key:
        return
    name = done.key.split(":", 1)[0]
    from . import generate as clay_generate

    if name in clay_generate.TASK_KEYS:
        clay_generate.on_task_failed(ctx, done)
        return
    tab = state.get(done.key.split(":", 1)[1])
    if tab is not None:
        tab.saving = False
        tab.bg_busy = ""


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
        # Keyed on object uid, so only this document's entries go: the uid
        # counter is process-wide and never rewinds, so nothing can collide
        # with a stale entry, and clearing the whole table made every other
        # open tab redo its (adjacency-building) checks on the next draw.
        for obj in tab.doc.objects:
            state.manifold.pop(obj.uid, None)

    docmodes.close_tab(ctx, state, uid, release)


# --- keys -------------------------------------------------------------------

# Q/W/E/R, which is where a user coming from Blender or Unity puts their left
# hand. Held here rather than in the pane so the mapping is testable.
TOOL_KEYS = {
    "q": "select",
    "w": "move",
    "e": "rotate",
    "r": "scale",
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
# the bridge panel draws the same Undo/Redo pair Inker's does. Clay, Plotter and
# Packwright each had a full undo stack and no on-screen control at all, so the
# feature existed only for a user who already knew the chord -- and every
# side effect a step has (nothing, here) belongs to *undoing*, not to the
# keyboard.


def undo(ctx: Any, tab: Any) -> None:
    """One step back, whichever surface asked for it."""
    tab.doc.undo()



def redo(ctx: Any, tab: Any) -> None:
    """One step forward. :func:`undo`'s twin, and its reasoning."""
    tab.doc.redo()


def check_readiness(ctx: Any, tab: ClayTab, profile: str) -> None:
    """Run ``readiness.validate`` against *tab*'s document, on a task thread.

    O(corners) -- a BFS per visible object, see that module's own cost section
    -- so this runs once per press of the bridge's "Check" button, never per
    frame. Submitted through ``ctx.submit`` rather than called here directly:
    the 2026-09-20 audit's clay-05 measured ``readiness.validate`` at 522 ms
    for 50 visible objects, 3.7 s for 800, 15.1 s for 3,200, all of it on the
    frame thread before this fix -- the same "nothing blocking runs on the
    main thread" rule every other document-changing call in this module
    already follows (this module's own docstring). The result lands on the
    tab in :func:`on_task_done` once it comes back, keyed
    ``clay-readiness:<tab uid>`` like ``clay-bg``'s own per-tab keys, so a
    tab closed while the check is still running has nowhere to land it.

    ``readiness.validate`` can also refuse outright now
    (``readiness.MAX_VALIDATE_OBJECTS``, the same audit's other half): raised
    inside ``run()``, on the task thread, it reaches the user as an ordinary
    task-failure toast (``tasks.CARRIES_ITS_OWN_MESSAGE`` already lists
    ``OpError``) rather than needing its own handling here.
    """
    tab.readiness_profile = profile
    doc = tab.doc
    # Captured *before* the task ever runs, not after it returns: once this
    # is backgrounded, an edit landing while the check is still on the pool
    # would otherwise get the *new* head attached to a report computed
    # against the *old* one -- the exact "stale but marked fresh" reading
    # ``readiness_head`` exists to prevent (see ``ClayTab``'s own comment),
    # just moved to a different point in time than the old, synchronous call
    # had to worry about.
    head = doc.history.head

    def run() -> Any:
        from ....kernels.mesh import readiness

        return readiness.validate(doc, profile)

    ctx.submit(f"clay-readiness:{tab.uid}", run, tag=head)


def step_history(ctx: Any, tab: Any, index: int) -> bool:
    """Jump the document to a position in its undo stack. -> whether it moved.

    The history panel's door, and the *third* surface onto the same stack --
    which is why it is here beside the other two rather than in the pane, and
    why the pane will not call ``doc.step_history`` itself. ``plotter_mode``
    has the same three, for the same reason written out there.

    ``ctx`` is taken and dropped: a jump used to also clear ``ClayState.
    last_op``, the record an "adjust last operation" card would have re-run
    from, but the 2026-09-07 audit's clay-10 removed that bookkeeping -- no
    pane ever read it -- and the parameter stays so the sibling editors'
    ``step_history(ctx, tab, index)`` and this one's one caller
    (``studio/modes/clay/ui/bridge.py``) do not need a signature change over it.
    """

    del ctx
    return tab.doc.step_history(index)



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
        view.drag_key(doc, name)
        # Consumed whether or not the drag wanted it. Falling through here put
        # every unclaimed bare key into the op registry below, so ``E`` typed
        # mid-``G`` ran Extrude against the mesh the drag was still moving --
        # and the drag's own commit then measured from ``_drag_start``, the
        # pre-drag baseline, and reverted it. The rule the comment above states
        # is only a rule if it holds for the keys the drag does *not* know.
        return True

    if ctrl:
        return _ctrl_key(ctx, state, tab, doc, name, shift=shift)

    if alt and name == "z":
        # The 2026-09-07 audit's clay-08: the X-ray button's own tooltip
        # (``studio/modes/clay/ui/header.py``) has named "(Alt+Z)" since it was added,
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
        _escape(state, tab, doc, view)
    return True



#: The two letters that start a transform with no handle grabbed, and what each
#: starts. **G and S only** -- not R, and the omission is the one interesting
#: thing about the table.
#:
#: ``R`` is the Scale *tool*'s letter and ``E`` is Rotate's, both taken long
#: before this and both in ``clay_state.TOOLS``; taking either back for a drag
#: would move a binding a user already has. What is free is ``G``, which every
#: modelling package uses for grab, and ``S``, which every one of them uses for
#: scale. Rotate is reached mid-drag instead -- ``G`` then ``R`` -- which is a
#: gesture Blender has anyway and which costs nothing here, because switching
#: transforms mid-drag had to work regardless.
#:
#: Checked *after* the op registry, so a letter an element mode has claimed
#: still fires its op: ``S`` is nothing in the registry today, and the ordering
#: is what keeps that from being a thing to remember if it ever is.
DRAG_KEYS = {"g": "move", "s": "scale"}


def _keyboard_drag(ctx: Any, view: Any, tab: ClayTab, doc: Any, name: str) -> bool:
    """Start a keyboard transform. -> whether the key was one.

    Refused while the tab is saving, like every control that changes the
    document -- and refused with nothing selected, where it would be a drag with
    nothing to drag and would swallow a keystroke that means nothing else.
    """

    kind = DRAG_KEYS.get(name)
    if kind is None or view is None or tab.saving:
        return False
    return bool(view.begin_keyboard_drag(doc, kind))


def _registry_key(ctx: Any, tab: ClayTab, doc: Any, name: str) -> bool:
    """Fire the registry op bound to a bare letter, if there is one.

    Checked *before* the tool keys so an element mode can claim a letter the
    transform tools also use -- E is Extrude with faces selected and Rotate
    without -- and checked through ``clay_ops.menu`` so the binding shown in the
    context menu and the binding that fires are one value.
    """
    from . import ops as clay_ops

    if tab.saving or doc.element_mode == "object":
        return False
    op = clay_ops.by_key(doc.element_mode, name.upper())
    if op is None or not op.enabled(doc):
        return False
    return _fire_op(ctx, doc, op)


def _fire_op(ctx: Any, doc: Any, op: Any) -> bool:
    """Run a registry op from the event layer, popping its dialog if it has one.

    Shared by the bare-letter path and the Ctrl-shortcut path so a
    parameterised op bound to either kind of key behaves the same way.
    """
    from . import ops as clay_ops

    state = ensure(ctx)
    if op.params:
        state.pending_op = op.name
        state.op_params.setdefault(op.name, clay_ops.defaults_for(op))
        # Asked for rather than opened: ``imgui.open_popup`` only takes effect
        # inside the window whose id stack is current, and this runs in the
        # event layer.
        state.open_op_popup = True
        return True
    return clay_ops.run(ctx, doc, op)


def _escape(state: ClayState, tab: ClayTab, doc: Any, view: Any = None) -> None:
    """Esc, staged: the armed knife, then the elements, then the mode, then
    the objects.

    One key that undoes the last thing the user got into, in the order they got
    into it. It **never leaves Clay mode**: Esc means "drop what I am doing",
    and losing a workspace full of tabs to a stray keypress is not that.

    **The knife check comes first and returns early.** ``ClayView.begin_knife``
    arms the gesture before any press lands -- waiting for the drag that draws
    the cut line -- and that armed-but-undragged state sets no ``_grab`` of
    its own (``begin_knife``'s own docstring), so ``handle_key``'s live-drag
    branch above (which checks ``view.dragging``) never sees it and an armed
    knife reached this function with nothing here that cancelled it: Esc
    silently did nothing while the knife sat waiting for its first click.
    ``view.cancel_drag`` already knows how to disarm exactly this state (its
    own docstring: "checked *ahead* of the ``dragging`` guard... since an
    *armed* knife... sets no ``_grab`` at all") -- this only has to ask it
    before running the staged clearing below, which is otherwise correct but
    has nothing to do with a knife that has not touched the document yet.
    """
    if view is not None and getattr(view, "_knife_armed", False):
        view.cancel_drag(doc)
        return
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
        # Shift picks the union rather than the weld -- one predicate gates
        # both, so the shift never changes whether the key does anything, only
        # which of the two answers it gives.
        from . import ops as clay_ops

        op = clay_ops.get("union" if shift else "join")
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
    elif name in GROW_KEYS:
        # Ctrl+plus and Ctrl+minus, on both the number row and the keypad.
        # Four names for two verbs, because the two rows report different key
        # names for the same glyph and a user pressing the one under their hand
        # should not have to know which.
        from . import ops as clay_ops

        op = clay_ops.get(GROW_KEYS[name])
        if op.enabled(doc) and doc.element_mode in op.modes:
            _fire_op(ctx, doc, op)
    elif name == "tab":
        state.cycle(-1 if shift else 1)
    return True


#: The two selection-size chords, by the key names pygame reports. ``=`` is the
#: unshifted key that carries ``+``, which is what a user presses; ``[+]`` and
#: ``[-]`` are the keypad's own names.
GROW_KEYS = {
    "=": "select-more",
    "+": "select-more",
    "[+]": "select-more",
    "-": "select-less",
    "[-]": "select-less",
}


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
    row (``studio/modes/clay/ui/outliner.py``) call this by name. Before the 2026-09-07
    audit's clay-07 it called ``selection.duplicate_selected`` straight, so
    the edit landed in history under ``add_objects``'s generic "object add"
    label instead of "Duplicate", and ``ClayState.last_op`` -- read by nothing
    today, but written by every other op -- never learned Duplicate had run.
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


def _journal_encode(tab: Any) -> bytes:
    from ....kernels.mesh import serialize

    # The camera goes in for the same reason a save carries it: a recovered
    # model that framed itself somewhere else is a recovered model the user has
    # to find their way back around.
    return serialize.rblk_bytes(tab.doc, view=tab.view)


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
        doc = serialize.read_rblk(_within_ceiling(path))
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
    }


JOURNAL = journal.tab_provider(
    "clay", ".rblk", "model", encode=_journal_encode, adopt=_journal_adopt
)
