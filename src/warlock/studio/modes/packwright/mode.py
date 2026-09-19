"""Packwright's controller: sources, packing, saving, exporting and keys.

The layer that knows about jobs and task threads; the engine under
``packwright/`` knows about neither. The panes draw, this decides.

**Packing is a task, and re-arming it is a flag.** MaxRects over several hundred
sprites plus a full-atlas numpy composite is not frame-thread work, so a repack
goes through ``ctx.submit`` and the result is adopted in :func:`on_task_done`,
where the texture upload belongs. What re-arms it is ``PackTab.pack_dirty``,
pumped from the preview pane's draw and cleared *only when a submit is
accepted* -- the ``findings_dirty`` lesson verbatim: ``TaskRunner.submit``
refuses a key already in flight and nothing re-arms it, so a burst of setting
changes would otherwise pack the state as it stood at the first one and drop
every edit after it.

**Composing reads only frozen data.** A ``Sprite``'s pixels are read-only and a
``Layout`` is frozen, so the pack task can safely run against them while the
frame thread draws -- which is why the snapshot taken at submit time is a list
of sprites and a settings object rather than the document.

Every task key carries the ``packwright-`` prefix, because the app claims
results by prefix: a key without one is a result delivered nowhere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ... import dialogs, docmodes, journal
from ...state import set_mode
from . import fileio as packwright_io
from . import state as packwright_state

# ``ensure`` and ``active`` live in :mod:`.packwright_state` -- they touch
# nothing but ``ctx.state.packwright`` -- and the file layer lives in
# :mod:`.packwright_io`. Both are re-exported here, as **plain imports rather
# than wrappers**, because every pane, every key binding and every test says
# ``packwright_mode.save(ctx)``: a wrapper would be a second object where the
# callers reach for one, and a wiring test parametrizes over ``IMAGE_FILTER``
# by identity besides.
from .fileio import (  # noqa: F401
    IMAGE_FILTER,
    PNG_FILTER,
    WPACK_FILTER,
    _decode,
    _load,
    _start,
    ask_open,
    edit_asset_in_packwright,
    export_files,
    export_library,
    open_path,
    save,
    save_as,
    save_to,
)
from .state import (  # noqa: F401
    PackTab,
    PackwrightState,
    active,
    ensure,
)

log = logging.getLogger(__name__)


# The three recents wrappers every document mode carries, over the one
# list Home's Resume rows are built from (``docmodes.recents_for``).
remember_path, forget_path, recent_paths = docmodes.recents_for("packwright")


def persist(ctx: Any) -> None:
    """Nothing to write any more: the recent list moved to :mod:`.recents`,
    which persists itself on every write. Kept as a no-op because it is called
    from a dozen places after every open and save, and turning each of those
    into "call this only if the mode still has settings" is how one of them
    comes to skip a write that mattered later."""


# --- documents ----------------------------------------------------------------


def adopt(ctx: Any, doc: Any, *, path: Path | None = None, title: str | None = None) -> PackTab:
    state = ensure(ctx)
    tab = PackTab(
        doc=doc,
        title=title or packwright_state.title_for(path),
        path=path,
        saved_head=doc.history.head,
    )
    state.add(tab)
    remember_path(ctx, path)
    persist(ctx)
    return tab


def new_document(ctx: Any) -> PackTab:
    from .engine.document import PackDoc

    return adopt(ctx, PackDoc(), title="Untitled")


# --- sources ------------------------------------------------------------------


def ask_add_sources(ctx: Any) -> None:
    """The picker and the decode on one task thread."""
    tab = active(ctx)
    if tab is None:
        docmodes.refuse(ctx, "Start or open an atlas first.")
        return
    uid = tab.uid

    from .engine.sources import file_key

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Add an image", IMAGE_FILTER)
        if path is None:
            return None
        return {"sprites": [(file_key(path), path.stem, _decode(path))], "uid": uid}

    ctx.submit(f"packwright-add:{uid}", run)


def ask_add_tileset(ctx: Any) -> None:
    """The picker and decode for an already-made tile sheet.

    The slicing happens later, in the sources pane's popup, because a sheet
    cannot say its own tile size -- the user provides it there, with a live
    count of what each answer would keep and drop.
    """
    tab = active(ctx)
    if tab is None:
        docmodes.refuse(ctx, "Start or open an atlas first.")
        return
    uid = tab.uid

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Add a tile set", IMAGE_FILTER)
        if path is None:
            return None
        return {"tileset": (str(path), path.stem, _decode(path)), "uid": uid}

    ctx.submit(f"packwright-tileset:{uid}", run)


def add_rendered_sheet(ctx: Any, job_id: str, sheet_id: str, *, pixel: bool = False) -> None:
    """A rendered 8-direction sheet onto the open atlas, through the usual door.

    The house pattern is confirm-at-the-door, so this parks the sheet on the
    existing ``tileset_import``/``tileset_import_open``/``tileset_cell`` trio
    rather than importing anything: the popup opens with the cell size the
    sidecar recorded already filled in, shows what that answer keeps and drops,
    and the existing occupancy/import path does the rest. Nothing new is
    imported and nothing new is refused.
    """
    tab = active(ctx)
    if tab is None:
        # ``add_job_source``'s rule: this is offered from *outside* Packwright,
        # so refusing is an offer taken back -- and an atlas has no numbers
        # that cannot be taken back later, so one is simply made.
        tab = new_document(ctx)
        set_mode(ctx.state, "packwright")
    uid = tab.uid

    def run() -> dict[str, Any] | None:
        import numpy as np
        from PIL import Image

        from ....service import sheets as svc_sheets
        from ..inker.mode import sheet_grid

        if pixel:
            record = svc_sheets.get_pixel_sheet(ctx.svc, job_id, sheet_id)
            png = svc_sheets.sheet_pixel_png(ctx.svc, job_id, sheet_id)
        else:
            record = svc_sheets.get_sheet(ctx.svc, job_id, sheet_id)
            png = svc_sheets.sheet_png(ctx.svc, job_id, sheet_id)
        cell, _count = sheet_grid(record)
        with Image.open(png) as opened:
            opened.load()
            pixels = np.asarray(opened.convert("RGBA"), dtype=np.uint8).copy()
        name = str(record.get("name") or sheet_id)
        return {"tileset": (str(png), name, pixels), "cell": cell, "uid": uid}

    # Keyed on the request's own identity, not the bare tab uid
    # ``ask_add_tileset`` submits under. The 2026-09-07 audit's packwright-02:
    # both call sites shared ``packwright-tileset:{uid}`` and neither inspects
    # ``ctx.submit``'s return, so a character-sheet handoff (Poser's own
    # since P9, 2026-09-18; Troupe's before it) landing while the manual
    # picker's OS dialog was still open -- the same tab, same key, already in
    # flight -- was refused with nothing to say so: manual submit ``True``,
    # handoff ``False``, no toast. ``job_id``/``sheet_id`` make this handoff
    # its own request, the ``add_job_source``/``add_source_paths`` shape.
    ctx.submit(f"packwright-tileset:{uid}:{job_id}:{sheet_id}", run)


def import_tileset(ctx: Any) -> bool:
    """Slice the pending sheet with the popup's cell size and add the tiles.

    The key carries the tile size as well as the path, so re-importing the same
    sheet cut differently adds a different set rather than being skipped as
    duplicates of the first cut.

    **Targets the tab that asked, not whichever is active.** The 2026-09-11
    audit's packwright-02: this used to resolve its target through
    ``active(ctx)``, so a sheet requested from tab A that landed (or was
    confirmed) while tab B was focused -- ordinary, since a background decode
    never blocks the frame loop and switching tabs while one is in flight is
    nothing the user is warned against -- was added to B's document instead,
    with no error or mismatched-document warning. ``on_task_done``'s own
    "packwright-add" handler already resolves its target from the task key
    rather than from whatever is active; this now does the same, from
    ``PackwrightState.tileset_import_uid``.
    """
    from .engine.sources import dedup_tiles, sprites_from_tileset

    state = ensure(ctx)
    if state.tileset_import is None:
        return False
    tab = state.get(state.tileset_import_uid)
    if tab is None:
        # The requesting tab closed in the window between the sheet landing
        # and the Import press -- the on_task_done landing already declines
        # (silently) a sheet whose tab closed before *it* lands; this is the
        # same closure one step later, but Import is a button the user just
        # pressed, so it gets a word rather than doing nothing.
        state.tileset_import = None
        state.tileset_preview_key = None
        state.tileset_import_open = False
        docmodes.refuse(ctx, "That atlas was closed before the tile set was imported.")
        return False
    path, stem, pixels = state.tileset_import
    tile = state.tileset_cell
    try:
        sprites = sprites_from_tileset(
            pixels, tile=tile, prefix=f"{path}@{tile[0]}x{tile[1]}", name=stem
        )
    except ValueError as exc:
        docmodes.refuse(ctx, f"That tile set was not imported: {exc}")
        return False
    if state.tileset_dedup:
        sprites, _dropped = dedup_tiles(
            sprites, orientations=state.tileset_dedup_flips
        )
    state.tileset_import = None
    state.tileset_preview_key = None
    state.tileset_import_open = False
    ctx.toast(_added_sentence(*_add_sprites(ctx, tab, sprites), noun="tile"))
    return True


def tileset_preview_key(
    pixels: Any, cell: tuple[int, int], dedup: bool, dedup_flips: bool
) -> tuple[Any, ...]:
    """What the tile-set import popup's promised counts are a function of.

    ``id(pixels)`` rather than the array's bytes: the decode is already behind
    a task and this only has to change when a *different* sheet lands, not
    hash a few megabytes on every frame the popup is open.
    """
    return (id(pixels), tuple(cell), bool(dedup), bool(dedup_flips))


def request_tileset_preview(
    ctx: Any, state: PackwrightState, path: str, stem: str, pixels: Any
) -> None:
    """Recompute the tile-set import popup's promised counts, off the frame
    thread.

    The 2026-09-07 audit's packwright-05: ``tileset_occupancy`` and, with
    dedup on, ``dedup_tiles``/``sprites_from_tileset`` re-slicing the whole
    sheet into fresh ``Sprite`` copies and hashing all eight dihedral variants
    of each ran **synchronously on the frame thread** every time the popup
    redrew with a changed key -- which is to say, once per typed digit in
    either tile-size field. The module's own comment already measured this at
    some hundreds of milliseconds on a full sheet; this submits it instead.

    Safe to call every frame, the ``request_pack`` shape: when the key has not
    moved this is one tuple comparison and nothing submitted, and while a
    computation for the *current* key is already in flight the key-derived
    task key makes the resubmit a no-op the runner refuses.
    """
    from .engine.layout import MAX_SPRITES
    from .engine.sources import dedup_tiles, sprites_from_tileset, tileset_occupancy

    key = tileset_preview_key(
        pixels, state.tileset_cell, state.tileset_dedup, state.tileset_dedup_flips
    )
    if state.tileset_preview_key == key:
        return

    tile = tuple(int(v) for v in state.tileset_cell)
    dedup = bool(state.tileset_dedup)
    dedup_flips = bool(state.tileset_dedup_flips)

    def run() -> dict[str, Any]:
        occupied = tileset_occupancy(pixels, tile=tile)
        rows, columns = occupied.shape
        kept = int(occupied.sum())
        dropped = rows * columns - kept
        duplicates = 0
        if dedup and kept and kept <= MAX_SPRITES:
            # The same call the import runs, not a second count of its own:
            # the popup-promise contract ``tileset_occupancy`` already
            # enforces for emptiness, applied to the dedup it also promises.
            _kept, duplicates = dedup_tiles(
                sprites_from_tileset(
                    pixels, tile=tile, prefix=f"{path}@{tile[0]}x{tile[1]}", name=stem
                ),
                orientations=dedup_flips,
            )
            kept -= duplicates
        return {"key": key, "preview": (rows, columns, kept, dropped, duplicates)}

    # Keyed on the inputs rather than the tab: this popup is not per-tab
    # state (``PackwrightState.tileset_import``), and the key doubles as the
    # runner's own in-flight dedupe, so a cell size typed and then un-typed
    # inside one frame's worth of keystrokes submits at most once per distinct
    # answer.
    ctx.submit(f"packwright-tileset-preview:{abs(hash(key))}", run)


def add_source_paths(ctx: Any, paths: list[Path]) -> None:
    """Files dropped on the window.

    **Keyed on the batch, not on the tab.** pygame raises one ``DROPFILE`` per
    file, so a drag of twenty PNGs reaches this twenty times in one pump; under
    a per-tab key the runner's dedupe refused nineteen of them and the refusal
    was thrown away, so a multi-file drop silently added the first file only
    (the 2026-09-02 review, section 7). ``inker-open``'s shape: the paths are
    in the key, so every distinct drop runs and a *repeated* drop of the same
    files while one is still decoding still dedupes, which is the half of the
    key worth keeping.
    """
    tab = active(ctx)
    if tab is None:
        # A drop is the same offer from outside: minted rather than refused,
        # ``add_job_source``'s rule.
        tab = new_document(ctx)
        set_mode(ctx.state, "packwright")
    from .engine.sources import file_key

    wanted = [Path(p) for p in paths]
    uid = tab.uid

    def run() -> dict[str, Any]:
        return {
            "sprites": [(file_key(p), p.stem, _decode(p)) for p in wanted],
            "uid": uid,
        }

    token = abs(hash(tuple(str(p) for p in wanted)))
    ctx.submit(f"packwright-add:{uid}:{token}", run)


def add_job_source(ctx: Any, job: Any) -> None:
    """A library asset's reference image as one sprite.

    No ceiling in front of the decode, unlike the ``.wpack`` path: a job's
    ``input.png`` was bounded by ``service.files`` when it was written, and a
    second door on the way back out would be a second answer to a question that
    already has one.
    """
    tab = active(ctx)
    if tab is None:
        # Start the atlas rather than refuse. The Library offers this for any
        # asset with an ``input.png`` and cannot know whether an atlas is open,
        # so the toast was an offer taken back. ``landing.start_packwright``'s
        # move, and unlike a map an atlas has no numbers that cannot be taken
        # back later, so there is nothing to ask about first.
        tab = new_document(ctx)
        # Only when one was minted here: joining an atlas the user already had
        # open is a background addition and does not move them, but a brand new
        # empty atlas they cannot see is the same dead end by another route.
        set_mode(ctx.state, "packwright")
    job_id = job["id"] if isinstance(job, dict) else str(job)
    name = (job.get("name") or job_id) if isinstance(job, dict) else job_id
    uid = tab.uid

    def run() -> dict[str, Any]:
        from ....service import files as svc_files

        path = svc_files.job_dir_file(ctx.svc, job_id, "input.png")
        return {"sprites": [(f"job:{job_id}", str(name), _decode(Path(path)))], "uid": uid}

    # Keyed on the job for ``add_source_paths``' reason: two assets sent from
    # the library in one gesture are two adds, and a shared key would drop the
    # second without a word.
    ctx.submit(f"packwright-add:{uid}:{job_id}", run)


def add_inker_document(ctx: Any, inker_tab: Any) -> None:
    """Every frame (or layer) of an open Inker document, as sprites.

    Enumerated on the frame thread deliberately: ``frame_flat`` fills and
    evicts the document's own flatten cache and ``layers_for`` copies track
    properties down onto cels, which is exactly what the onion-skin draw is
    doing to the same dicts. That is the ``inker.sheetout`` split, and this is
    the same boundary in a different mode.
    """
    from .engine.sources import sprites_from_document

    tab = active(ctx)
    if tab is None:
        # ``add_job_source``'s rule: this is offered from *outside* Packwright
        # (Inker's bridge, and Packwright's own sources pane before an atlas
        # exists), so refusing here is an offer taken back. An atlas has no
        # numbers that cannot be taken back later, so one is simply made.
        tab = new_document(ctx)
        if ctx.state.mode != "packwright":
            set_mode(ctx.state, "packwright")
    prefix = Path(inker_tab.title).stem or inker_tab.uid
    try:
        sprites = sprites_from_document(inker_tab.doc, prefix=prefix)
    except ValueError as exc:
        docmodes.refuse(ctx, f"Those frames were not added: {exc}")
        return
    ctx.toast(_added_sentence(*_add_sprites(ctx, tab, sprites)))


def set_pivot(
    ctx: Any, tab: PackTab, uid: int, pivot: tuple[float, float] | None
) -> None:
    """Move one sprite's anchor, and re-arm the pack.

    Through the mode rather than onto the document for ``rename_source``'s
    reason: the pivot rides in the layout and out into the exported sidecar, so
    a pivot that never reaches a pack is a pivot the sidecar does not carry.
    """

    if tab is None or tab.saving:
        return
    before = tab.doc.history.head
    tab.doc.set_pivot(int(uid), pivot)
    if tab.doc.history.head != before:
        tab.pack_dirty = True


def _add_sprites(ctx: Any, tab: PackTab, sprites: list[Any]) -> tuple[int, int]:
    """Add what is not there and refresh what is. -> ``(added, replaced)``.

    A duplicate key is *skipped* rather than refused, unlike ``PackDoc``'s own
    rule: dropping twenty files of which one is already in the atlas should add
    nineteen, not fail. The document's refusal stays the authority on what may
    coexist; this is the caller deciding what to ask for.

    **A key already here whose pixels have changed is a replacement**, not a
    skip. ``wpack``'s own docstring says what a source is -- "what the document
    records is what was packed; re-adding the source is how you pick up a
    change" -- and until 2026-09-03 re-adding an edited PNG was silently
    nothing at all, which made that sentence false and left the only way to
    pick up an edit "delete the sprite first" (the 2026-09-02 review, section
    7). Unchanged pixels are still a skip, so re-dropping a folder is not
    twenty undo steps.
    """
    added = replaced = 0
    for sprite in sprites:
        existing = next(
            (one for one in tab.doc.sources if one.key == sprite.key), None
        )
        try:
            if existing is None:
                tab.doc.add_source(sprite)
                added += 1
                continue
            before = tab.doc.history.head
            tab.doc.replace_source(existing.uid, sprite)
            if tab.doc.history.head != before:
                replaced += 1
        except ValueError as exc:
            # A ceiling tripped partway through a multi-file batch used to
            # propagate straight out of this loop, past on_task_done, into
            # main.py's generic task-landing handler, which toasted "That did
            # not finish landing: packwright-add:..." -- a fact about the
            # frame loop, not the pack -- and left whatever sprite *did* land
            # before it with pack_dirty unset (the 2026-09-14 audit,
            # packwright-01). Caught here instead: what landed stays landed
            # and dirty, and a second toast (the tsx-skip idiom in
            # on_task_done, below) names the ceiling and how far the batch got.
            if added or replaced:
                tab.pack_dirty = True
            ctx.toast(
                f"Stopped after {added + replaced} of {len(sprites)}: {exc}",
                "warn",
            )
            return added, replaced
    if added or replaced:
        tab.pack_dirty = True
    return added, replaced


def _added_sentence(added: int, replaced: int, *, noun: str = "sprite") -> str:
    """What ``_add_sprites`` just did, in one sentence naming both halves.

    Both counts or neither: "Added 19" hides the twentieth file being the
    edited one the user actually dropped this folder for, and "Updated 1" hides
    the nineteen.
    """
    parts = []
    if added:
        parts.append(f"Added {added} {noun}(s)")
    if replaced:
        parts.append(f"updated {replaced}" if parts else f"Updated {replaced} {noun}(s)")
    if not parts:
        return f"Those {noun}s are already in this atlas, unchanged."
    return ", ".join(parts) + "."


def remove_source(ctx: Any, uid: int, tab: PackTab | None = None) -> None:
    tab = tab or active(ctx)
    if tab is None or tab.busy:
        return
    if tab.doc.source(uid) is None:
        # A uid goes stale for ordinary reasons -- an undone add is the one that
        # bit -- and Delete against one is a no-op rather than a refusal: there
        # is nothing to tell the user that they did not already see.
        return
    tab.doc.remove_source(uid)
    tab.pack_dirty = True
    state = ensure(ctx)
    if state.selected == uid:
        state.selected = None


def rename_source(ctx: Any, tab: PackTab | None, uid: int, name: str) -> None:
    """Rename one source. **The pack is re-armed**, which is the whole reason
    this exists: the pane used to call ``tab.doc.rename_source`` directly, so
    the name changed in the list and the *layout* -- which is what the
    TexturePacker sidecar's ``filename`` is written from -- kept the old one
    until something else happened to dirty the pack. An export in between
    carried a name nothing on screen still showed."""
    tab = tab or active(ctx)
    if tab is None or tab.busy:
        return
    try:
        tab.doc.rename_source(uid, name)
    except ValueError as exc:
        docmodes.refuse(ctx, f"That name was not applied: {exc}")
        return
    tab.pack_dirty = True


def set_settings(ctx: Any, tab: PackTab | None = None, **values: Any) -> None:
    """Every settings edit goes through here, so ``pack_dirty`` cannot be
    forgotten at one of six call sites."""
    tab = tab or active(ctx)
    if tab is None or tab.busy:
        return
    try:
        tab.doc.set_settings(**values)
    except (ValueError, TypeError) as exc:
        # ``TypeError`` as well: ``dataclasses.replace`` raises it for a field
        # that does not exist, which is what a stale keyword from a pane would
        # be -- and an uncaught one here is a crash rather than a refusal.
        # Framed rather than forwarded, the house rule: a bare ``str(exc)``
        # toast is library text with no subject in front of it.
        docmodes.refuse(ctx, f"That setting was not applied: {exc}")
        return
    tab.pack_dirty = True


def source_index(tab: PackTab) -> dict[str, int]:
    """``source.key -> uid`` for every source, cached on the tab.

    The 2026-09-07 audit's packwright-07: the preview and items panes each
    rebuilt this dict comprehension from scratch on every single frame they
    drew, though every caller only ever maps a *packed* frame's key back to
    its source -- and a packed frame's key set cannot change except when
    ``pack_generation`` does, since that is the one counter ``adopt_pack``
    bumps and the one place ``tab.layout`` (what a frame's key comes from)
    is replaced. Stashed as a plain attribute on the tab -- a bare
    ``@dataclass``, not slotted -- rather than growing ``PackTab`` a field
    for it, the same reach ``packwright_textures`` makes into
    ``ctx.state.preview`` for a texture cache keyed on the same counter.
    """
    cached = getattr(tab, "_pw_source_index", None)
    if cached is not None and cached[0] == tab.pack_generation:
        return cached[1]
    index = {source.key: source.uid for source in tab.doc.sources}
    tab._pw_source_index = (tab.pack_generation, index)
    return index


# --- packing ------------------------------------------------------------------


def request_repack(ctx: Any, tab: PackTab | None = None) -> None:
    """Mark the atlas dirty so the next pump repacks it.

    What ``R`` does, given a name so the settings pane's Repack button is the
    same verb rather than a second one that happens to agree. It does not pack
    here: packing is a worker job the centre pane's pump owns, and a pane that
    started one would be doing frame-thread work on a full-atlas composite.
    """
    tab = tab or active(ctx)
    if tab is None:
        return
    tab.pack_dirty = True


def request_pack(ctx: Any, tab: PackTab | None = None) -> None:
    """Ask for a repack. Safe to call every frame -- that is the point."""
    tab = tab or active(ctx)
    if tab is None or not tab.pack_dirty:
        return
    if not tab.doc.sources:
        tab.layout, tab.atlas, tab.pack_dirty, tab.pack_error = None, None, False, ""
        return

    from .engine import compose as composelib
    from .engine import layout as laylib

    # The snapshot: frozen sprites and a frozen settings object, so the task
    # reads nothing the frame thread can be writing.
    sprites = tab.doc.sprites()
    settings = tab.doc.settings
    uid = tab.uid

    def run() -> dict[str, Any]:
        from ....service.errors import invalid_from

        try:
            result = laylib.layout(sprites, settings)
        except ValueError as exc:
            # Framed, because only a ``ServiceError``'s text survives the task
            # classifier: the engine's *remedy* sentence -- raise the max size,
            # trim them, or split the pack -- is what ``pack_error`` is for, and
            # a bare ValueError put "see the log for details" there instead.
            raise invalid_from(exc, "That pack did not work") from exc
        return {"layout": result, "atlas": composelib.compose(sprites, result), "uid": uid}

    tab.packing = True
    if ctx.submit(f"packwright-pack:{uid}", run):
        # Cleared *only* on an accepted submit. The runner refuses a key already
        # in flight, and clearing regardless would drop the edit that arrived
        # while the previous pack was running.
        tab.pack_dirty = False
    else:
        tab.packing = False


def pump(ctx: Any) -> None:
    """Called from the preview pane's draw, which is the only thing that runs
    every frame in this mode -- the ``motion.py`` idiom."""
    request_pack(ctx)


# --- task results -------------------------------------------------------------


def on_task_done(ctx: Any, done: Any) -> None:
    state = ensure(ctx)
    key, result = done.key, done.result
    name = key.split(":", 1)[0]

    if name == "packwright-open":
        if isinstance(result, dict):
            # The 2026-09-18 audit (second run, finding packwright-01) found
            # this arm adopting unconditionally while ``fileio.open_path``
            # already guards -- the gap Clay closed on 2026-09-12 (clay-02).
            path = Path(result["path"]) if result.get("path") else None
            existing = state.find_path(path) if path is not None else None
            if existing is not None:
                state.activate(existing.uid)
            else:
                adopt(
                    ctx,
                    result["doc"],
                    path=path,
                    title=result.get("title"),
                )
            set_mode(ctx.state, "packwright")
        return

    if name == "packwright-recover":
        # No tab uid in this key -- there is no tab yet -- so this is handled
        # before the generic lookup below, the ``packwright-open`` shape.
        if result is None:
            journal.adopt_failed(ctx, "atlas")
        elif isinstance(result, dict):
            tab = adopt(ctx, result["doc"], path=None, title=result.get("title"))
            docmodes.mark_recovered(tab, Path(result["path"]), result["doc"])
        return

    if name == "packwright-tileset-preview":
        # Not tab-scoped -- the popup's counts live on ``PackwrightState``,
        # not on a ``PackTab`` -- so this is handled before the generic
        # tab lookup below, which a task key with no tab uid in it cannot
        # satisfy. Adopted only if the result still describes the *current*
        # inputs: the cell size may have changed again while this task was
        # in flight, and a second, fresher submit could already be running
        # under its own key -- landing first or last, whichever result no
        # longer matches ``tileset_preview_key`` must lose.
        if isinstance(result, dict) and state.tileset_import is not None:
            pixels = state.tileset_import[2]
            current = tileset_preview_key(
                pixels, state.tileset_cell, state.tileset_dedup, state.tileset_dedup_flips
            )
            if result.get("key") == current:
                state.tileset_preview_key = current
                state.tileset_preview = result["preview"]
        return

    # ``split(":")[1]``, not ``split(":", 1)[1]``: an add carries a third
    # segment (the batch token) so that several drops can be in flight at once,
    # and a tab uid never contains a colon.
    tab = state.get(key.split(":")[1]) if ":" in key else None
    if tab is None:
        return

    if name == "packwright-pack":
        if isinstance(result, dict):
            tab.adopt_pack(result["layout"], result["atlas"])
        else:
            tab.packing = False
        return

    if name == "packwright-tileset":
        # The decode landing: the sheet parks on the state until the popup's
        # tile-size answer turns it into sprites, or a cancel drops it.
        if isinstance(result, dict):
            if state.tileset_import is not None:
                # **Refused rather than adopted.** The 2026-09-08 audit
                # (finding packwright-01): a second tile-sheet landing while
                # an earlier one's popup was already open used to overwrite
                # ``tileset_import`` unconditionally and reset
                # ``tileset_import_open`` to False -- which the pane's own
                # "open once a new import lands" check (above, in
                # ``modes/packwright/ui/panes/sources.py``) then read as a fresh
                # import and reopened the popup over, silently, completely
                # different pixels: no toast, no confirm, no visible sign
                # anything had changed underneath a user mid-typing a tile
                # size. The user is already answering a question about one
                # sheet; a second sheet does not get to jump the queue and
                # answer it for them -- they finish or cancel the open popup
                # and press Add a tile set again.
                #
                # Guarded on ``tileset_import is not None`` alone, not also
                # ``and tileset_import_open`` (the 2026-09-15 audit,
                # packwright-02): ``tileset_import_open`` is set only by the
                # pane's own draw, so two landings inside one poll batch --
                # both processed before a frame is ever drawn -- both saw it
                # still ``False`` and the second one swapped the first's
                # parked pixels out from under it with neither popup ever
                # having been on screen.
                ctx.toast(
                    "Another tile set is already waiting on the tile-size "
                    "popup -- confirm or cancel it first."
                )
                return
            state.tileset_import = result["tileset"]
            # Recorded, not just checked: ``tab`` here is the *requesting*
            # tab, resolved above (from the task key) only to confirm it is
            # still open -- packwright-02 was that nothing carried its
            # identity any further than that check.
            state.tileset_import_uid = tab.uid
            state.tileset_import_open = False
            cell = result.get("cell")
            if cell is not None:
                # Only when the door knew the answer. A picked file does not,
                # and overwriting the last typed cell size for it would throw
                # away the number the user is cutting a folder of sheets with.
                state.tileset_cell = (int(cell[0]), int(cell[1]))
        return

    if name == "packwright-add":
        # ``packing`` is deliberately not touched: it belongs to the pack task
        # (set by ``request_pack``, cleared where the pack lands or fails), and
        # an add landing while a pack was in flight used to clear it here --
        # the preview then read "not packing" about a pack still running.
        if isinstance(result, dict):
            from .engine.sources import sprite_from_image

            sprites = [
                sprite_from_image(pixels, key=key_, name=display)
                for key_, display, pixels in result.get("sprites", [])
            ]
            ctx.toast(_added_sentence(*_add_sprites(ctx, tab, sprites)))
        return

    tab.saving = False
    if not isinstance(result, dict):
        return  # a cancelled dialog

    if result.get("exported_asset"):
        ctx.cache.invalidate()
        ctx.toast("Exported to the library.")
        return
    if result.get("exported"):
        ctx.toast(f"Exported {result.get('files', 2)} file(s) to {result['exported']}")
        if result.get("tsx_skipped"):
            # A second toast rather than folded into the first: "Exported 2
            # file(s)" is a success sentence, and burying "the .tsx was not
            # one of them, because ..." inside it reads as one long message
            # nobody finishes. The PNG and JSON still exported -- only the
            # tileset that would have sliced wrong did not.
            ctx.toast(f"No .tsx written: {result['tsx_skipped']}", "warn")
        return

    tab.mark_saved(result.get("head"))
    # See ``inker_mode``: saved is the moment the crash copy stops
    # describing anything at risk (UX-05).
    journal.drop(ctx, tab)
    if result.get("retitle") and result.get("path"):
        tab.path = Path(result["path"])
        tab.title = packwright_state.title_for(tab.path)
        remember_path(ctx, tab.path)
        persist(ctx)
    ctx.toast("Saved.")


def on_task_failed(ctx: Any, done: Any) -> None:
    """A failed save must not leave the document locked, and a failed *pack*
    must clear ``packing`` and record why -- an empty items list that looks
    like success is the worst outcome of a pack that could not fit."""
    if done.key.startswith(packwright_io.OPEN_PREFIX):
        # Before the tab lookup, because an open that failed has no tab: what it
        # has is a path that does not open, and a Resume list that keeps
        # offering one is worse than a short one. The key carries the path
        # rather than a hash of it precisely so this can be done. An
        # ``edit_asset`` key carries a job id instead, and forgetting one of
        # those is a lookup that matches nothing.
        forget_path(ctx, done.key.split(":", 1)[1])
        return
    state = ctx.state.packwright
    if state is None or ":" not in done.key:
        return
    tab = state.get(done.key.split(":", 1)[1])
    if tab is None:
        return
    tab.saving = False
    if done.key.startswith("packwright-pack"):
        tab.packing = False
        tab.pack_error = done.message or "That pack did not work."


# --- guard and keys -----------------------------------------------------------


def guard(ctx: Any, verb: str, proceed: Any) -> bool:
    """Ask before losing unsaved work. -> whether it went ahead now.

    One question for all of them, the ``clay_mode.guard`` shape. Only quitting
    and closing a tab are destructive: switching modes is not, because
    Packwright is a mode rather than a takeover and its tabs are still there on
    the way back.
    """
    return docmodes.guard(ctx, "packwright", "atlas", "atlases", verb, proceed)


def close_tab(ctx: Any, uid: str) -> None:
    """``docmodes.close_tab``; what is Packwright's is the release."""
    state = ensure(ctx)

    def release(_tab: PackTab) -> None:
        from .ui.panes import settings as packwright_settings
        from .ui.panes import sources as packwright_sources
        from .ui.panes import textures as packwright_textures

        packwright_textures.release_doc(ctx, uid)
        # The 2026-09-08 audit (finding packwright-03): ``_last_columns`` is a
        # module-level ``dict[tab.uid, int]`` remembering each tab's last
        # explicit column count, and it is the one per-tab-uid cache in this
        # segment with no matching release -- ``packwright_textures`` above
        # already pops its own cache keyed the same way, every closed tab
        # ever given an explicit column count left one entry behind for the
        # life of the process.
        packwright_settings._last_columns.pop(uid, None)
        # The 2026-09-16 audit: a tile-set import waits on its own popup
        # (``PackwrightState.tileset_import``), named to the tab that asked by
        # ``tileset_import_uid`` -- closing *that* tab (Ctrl+W, reachable with
        # no confirm on a still-clean new atlas) used to leave the popup on
        # screen describing a document that no longer exists, holding its
        # texture and grid cache alive, until a human noticed and cancelled it
        # by hand. ``import_tileset`` already refuses an Import press once the
        # tab is gone; this is the same closure, paid at close time instead of
        # waiting for that press.
        if state.tileset_import is not None and state.tileset_import_uid == uid:
            packwright_sources.clear_tileset_import(ctx, state)

    docmodes.close_tab(ctx, state, uid, release)


def release_all(ctx: Any) -> None:
    from .ui.panes import textures as packwright_textures

    packwright_textures.release_all(ctx)


_MUTATING_CTRL = docmodes.WRITE_CHORDS


# --- history ------------------------------------------------------------------
#
# One call per direction, rather than two lines under the key handler, because
# the bridge panel draws the same Undo/Redo pair Inker's does. Clay, Plotter and
# Packwright each had a full undo stack and no on-screen control at all, so the
# feature existed only for a user who already knew the chord -- and every
# side effect a step has (a repack, and a selection that may name a sprite
# the step removed) belongs to *undoing*, not to the keyboard.


def undo(ctx: Any, tab: Any) -> None:
    """One step back, whichever surface asked for it."""
    tab.doc.undo()
    tab.pack_dirty = True
    _drop_stale_selection(ensure(ctx), tab)



def redo(ctx: Any, tab: Any) -> None:
    """One step forward. :func:`undo`'s twin, and its reasoning."""
    tab.doc.redo()
    tab.pack_dirty = True
    _drop_stale_selection(ensure(ctx), tab)


def step_history(ctx: Any, tab: Any, index: int) -> bool:
    """Jump to a position in the undo stack -- the history popover's door,
    carrying :func:`undo`'s side effects for :func:`undo`'s reason."""
    moved = tab.doc.history.step_to(tab.doc, index)
    tab.pack_dirty = True
    _drop_stale_selection(ensure(ctx), tab)
    return moved



def handle_key(ctx: Any, event: Any) -> bool:
    """Packwright's keyboard. Returns whether the key was consumed; the app
    returns afterwards either way, as it does for every workspace mode."""
    import pygame

    if event.type != pygame.KEYDOWN:
        return False
    state = ensure(ctx)
    tab = state.active
    # Off ``event.mod``, never ``pygame.key.get_mods()`` -- ``main._shortcut``'s
    # rule (UX-12): ``mod`` is the state when this key was pressed, and
    # ``get_mods()`` is the state now, after the event batch drained.
    mods = event.mod
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    name = pygame.key.name(event.key).lower()

    if ctrl:
        if tab is not None and docmodes.blocked_while_writing(tab, name, _MUTATING_CTRL):
            return True
        return _ctrl_key(ctx, state, tab, name, shift=shift)

    if tab is None:
        return False
    if name == "r":
        request_repack(ctx, tab)
        return True
    if event.key == pygame.K_DELETE and state.selected is not None:
        remove_source(ctx, state.selected, tab)
        return True
    if event.key == pygame.K_ESCAPE:
        state.selected = None
        return True
    return False


def _drop_stale_selection(state: PackwrightState, tab: PackTab) -> None:
    """Clear the selection if the step just undone detached what it names.

    Narrower than ``plotter_mode``'s unconditional clear, and deliberately:
    Plotter's selected *object* is a position in a layer that any step can move,
    while a source's uid survives every step but the one that removes it -- so
    keeping the row selected through an undo of a rename is the right answer,
    and only a detached uid has to go. Leaving it was a crash: Delete then
    addressed a uid the document no longer holds.
    """
    if state.selected is not None and tab.doc.source(state.selected) is None:
        state.selected = None


def _ctrl_key(
    ctx: Any, state: PackwrightState, tab: PackTab | None, name: str, *, shift: bool
) -> bool:
    if name == "n":
        new_document(ctx)
        return True
    if name == "o":
        ask_open(ctx)
        return True
    if tab is None:
        return False
    if name == "w":
        close_tab(ctx, tab.uid)
        return True
    if name == "s":
        save_as(ctx, tab) if shift else save(ctx, tab)
        return True
    if name == "e":
        export_files(ctx, tab) if shift else export_library(ctx, tab)
        return True
    if name == "z":
        # Ctrl+Shift+Z redoes as well, which is what Inker, Clay and Plotter
        # accept and what a user arriving from any of them already has in their
        # hand. Ctrl+Y keeps working: this adds a spelling rather than
        # replacing one.
        redo(ctx, tab) if shift else undo(ctx, tab)
        return True
    if name == "y":
        redo(ctx, tab)
        return True
    if name == "tab":
        state.cycle(-1 if shift else 1)
        return True
    if name == "0":
        tab.view.fitted = False
        return True
    if name == "1":
        tab.view.pending_zoom = 1.0
        return True
    return False


# --- crash recovery (UX-05) ---------------------------------------------------
#
# ``clay_mode``'s four answers, for atlases. See :mod:`studio.journal`.
#
# The *layout* is deliberately not journalled, for the reason it is not saved:
# it is derived, and a re-export of an unchanged document is byte-identical
# with nothing to invalidate. A recovered atlas repacks itself, which is
# seconds and is the only answer that cannot be stale.


def _journal_encode(tab: Any) -> bytes:
    from .engine import wpack

    return wpack.wpack_bytes(tab.doc)


def _journal_adopt(ctx: Any, path: Path, meta: dict[str, Any]) -> bool:
    """Reopen one recovered ``.wpack`` as an *untitled, dirty* tab.

    Read and decoded on a task, ``clay_mode``'s and ``inker_mode``'s
    ``*-recover`` shape (the 2026-09-11 audit's packwright-01): this used to
    read the file and decompress every sprite PNG synchronously, right here --
    and the Recover button in ``panes/landing.py`` calls ``journal.take`` ->
    ``journal.adopt`` -> ``provider.adopt`` with no ``ctx.submit`` anywhere in
    that chain, so "here" was the frame thread. Measured at 137.7 ms for a
    300-sprite/128px, 17.4 MB ``.wpack`` -- an ~8-frame freeze on one click, on
    an atlas size the manual calls ordinary, and ``MAX_PACK_SOURCE_BYTES``
    allows up to ~1 GB. True means "submitted"; ``on_task_done`` does the
    adopting.
    """
    ensure(ctx)
    ctx.submit(
        f"packwright-recover:{abs(hash(str(path)))}", _load_recovery, Path(path), dict(meta)
    )
    return True


def _load_recovery(path: Path, meta: dict[str, Any]) -> dict[str, Any] | None:
    """The task-thread half of a crash recovery: bytes to document.

    ``None`` rather than a raise on a bad file: the landing turns it into the
    one sentence every provider says (``journal.adopt_failed``), where a raise
    here would arrive as an *error* toast no other mode's copy raises.
    """
    from .engine import wpack

    try:
        doc = wpack.read_wpack(packwright_io._within_ceiling(Path(path)).read_bytes())
    except Exception:
        log.exception("could not reopen the recovered atlas at %s", path)
        return None
    title = f"{meta.get('title') or Path(path).stem} (recovered)"
    return {"doc": doc, "title": title, "path": str(path)}


JOURNAL = journal.tab_provider(
    "packwright", ".wpack", "atlas", encode=_journal_encode, adopt=_journal_adopt
)
