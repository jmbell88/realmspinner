"""Packwright's left-top pane: what is going into the atlas.

Three ways in, and the third is the reason the mode exists beside Inker: an
**open Inker document** contributes one sprite per frame of an animated clip, or
one per layer of a still one. That enumeration runs on the frame thread on
purpose -- see ``packwright_mode.add_inker_document`` for why -- and the two
loose-file paths run on a task thread because they decode PNGs.

A duplicate is skipped rather than refused when a *batch* is added, which is the
caller's decision and not the document's: dropping twenty files of which one is
already present should add nineteen.
"""

from __future__ import annotations

from typing import Any

from ..... import controls, docmodes, icons, tokens, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import mode as packwright_mode


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = packwright_mode.ensure(ctx)
    tab = state.active
    widgets.section("Sources")
    manual_render.help_button(ctx, "packwright-sources")

    if tab is None:
        # The heading and nothing else. One voice for one empty state:
        # the canvas's ``nothing_open`` is it, and four panels each
        # repeating it reads as four separate problems.
        return

    editable = not tab.busy
    # reason=: the 2026-09-18 audit's packwright-02 -- these two greyed out
    # while a save was in flight with no ``reason``, so the tooltip that
    # explains every other greyed control in this app said nothing here.
    if widgets.disabled_button(
        f"{icons.PLUS} Add an image...",
        editable,
        (-1, 0),
        reason=widgets.DOCUMENT_SAVING_WHY,
    ):
        packwright_mode.ask_add_sources(ctx)

    if widgets.disabled_button(
        f"{icons.GRID} Add a tile set...",
        editable,
        (-1, 0),
        reason=widgets.DOCUMENT_SAVING_WHY,
        tooltip="Loads an already-made tile sheet, slices it on a grid you "
        "set, drops the empty cells, and packs what is left -- so a sparse "
        "sheet comes back as a smaller one.",
    ):
        packwright_mode.ask_add_tileset(ctx)
    if state.tileset_import is not None and not state.tileset_import_open:
        state.tileset_import_open = True
        imgui.open_popup(TILESET_POPUP)
    _tileset_popup(ctx, state)

    _from_inker(ctx, tab, editable)

    imgui.dummy((0, sp(tokens.SP_2)))
    sources = tab.doc.sources
    if not sources:
        widgets.muted_wrapped(
            "Drop images on the window, add one above, or pull the frames out of "
            "a document open in Inker."
        )
        return

    widgets.muted(f"{len(sources)} sprite(s)")
    # **The one list in this app that grows without bound** (B-list, wave 4).
    # Packwright's items pane is a *result* list and Troupe's cast is below
    # ``list_filter``'s own self-hiding threshold; an atlas's sources are
    # whatever the user has dropped on it, which is hundreds by the end of a
    # sheet.
    needle = widgets.list_filter(ctx, "packwright-sources", len(sources))
    imgui.dummy((0, 2))
    matched = [
        source
        for source in sources
        if not needle or needle in (getattr(source, "name", "") or "").lower()
    ]
    # **Clipped.** This is the one list in this app that grows without bound
    # (see the comment above), and every row of it was submitted whether or not
    # the pane could show it -- at a thousand sources that is a thousand
    # selectables, a thousand context-menu registrations and a thousand hover
    # tests per frame, for the twenty a 300 dp column holds. ``ListClipper``
    # submits the visible span and seeks the cursor past the rest.
    #
    # The *selected* row is taller than the others (it grows a rename field and
    # a Remove button), which the clipper handles by re-measuring: the cost is
    # a scroll estimate that is one row out while that row is off screen, and
    # the alternative -- drawing every row to keep the heights uniform -- is
    # what this exists to stop.
    clipper = imgui.ListClipper()
    clipper.begin(len(matched))
    while clipper.step():
        for index in range(clipper.display_start, clipper.display_end):
            _row(ctx, state, tab, matched[index], editable)
    clipper.end()
    widgets.no_matches(needle, len(matched))


TILESET_POPUP = "packwright-tileset-import"

#: The parked sheet's own GL texture, keyed on ``id(pixels)``. One import is
#: parked at a time (``PackwrightState.tileset_import`` is a single slot), so
#: a bare prefix sweep is enough to forget it -- ``packwright_textures``'s
#: rule, shrunk to one entry.
_SLICE_TEX_PREFIX = "packwright_tileset_slice:"

#: The last occupancy grid the slice preview drew, and the ``(id(pixels),
#: tile)`` it was drawn for. Module-level rather than on ``PackwrightState``:
#: this is the preview's own cache, nothing else reads it. Caching at all is
#: the same reason ``request_tileset_preview`` moved its own count off the
#: frame thread (the 2026-09-07 audit's packwright-05) -- this grid is one
#: vectorised pass over the sheet's alpha channel rather than a re-slice with
#: dihedral hashing, so it is cheap by comparison, but the popup redraws
#: every frame it is open and there is still no reason to pay for the same
#: answer sixty times a second.
_slice_grid_cache: tuple[tuple[int, tuple[int, int]], Any] | None = None


def _occupancy_for(pixels: Any, tile: tuple[int, int]) -> Any:
    """The occupancy grid for one sheet at one cell size -- ``tileset_occupancy``'s
    own output, cached until either input moves. Returns ``None`` for a cell
    size ``tileset_occupancy`` refuses (below 1 x 1)."""
    global _slice_grid_cache
    from ...engine.sources import tileset_occupancy

    cell = (int(tile[0]), int(tile[1]))
    key = (id(pixels), cell)
    if _slice_grid_cache is not None and _slice_grid_cache[0] == key:
        return _slice_grid_cache[1]
    try:
        grid = tileset_occupancy(pixels, tile=cell)
    except ValueError:
        grid = None
    _slice_grid_cache = (key, grid)
    return grid


def _slice_texture(ctx: Any, pixels: Any) -> Any:
    """The parked sheet's own pixels as a GL texture, or ``None`` with no GL.

    ``packwright_textures.atlas_texture``'s shape, shrunk to one entry: the
    sheet is frozen for as long as it is parked (the decode task hands the
    popup one array and never mutates it), so identity alone is the staleness
    stamp.
    """
    if ctx.viewer is None:
        return None
    key = f"{_SLICE_TEX_PREFIX}{id(pixels)}"
    texture = ctx.state.preview.get(key)
    if texture is None:
        texture = ctx.viewer.ctx.texture(
            (int(pixels.shape[1]), int(pixels.shape[0])), 4, pixels.tobytes()
        )
        # Nearest: the same reason ``packwright_textures.atlas_texture`` picks
        # it -- this is inspected at whole-pixel zooms and a linear filter
        # would blur the very cell edges this preview exists to mark.
        nearest = ctx.viewer.ctx.NEAREST
        texture.filter = (nearest, nearest)
        ctx.state.preview[key] = texture
    return texture


def _forget_slice_preview(ctx: Any) -> None:
    """Drop the parked sheet's texture and cached grid.

    Called wherever the popup stops showing one -- cancelled, imported, or
    dismissed by a click outside -- so a closed popup does not go on holding a
    megapixel texture for the rest of the session. Idempotent: the texture
    prefix may already be empty and the grid may already be ``None``."""
    global _slice_grid_cache
    docmodes.release_prefix(ctx, _SLICE_TEX_PREFIX)
    _slice_grid_cache = None


def clear_tileset_import(ctx: Any, state: Any) -> None:
    """Drop a parked tile-set import outright: the state quartet, and the
    parked sheet's own texture and cached grid.

    Pulled out of ``_tileset_popup``'s Cancel button and its click-outside
    branch, which used to do this inline in two slightly different orders, so
    ``packwright_mode.close_tab`` can reach the same clear. The 2026-09-16
    audit: closing the tab a pending import named (Ctrl+W, reachable with no
    confirm on a still-clean new atlas) used to leave this popup on screen
    describing a document that no longer existed -- and its texture and grid
    cache alive -- until a human noticed and cancelled it by hand.
    """
    state.tileset_import = None
    state.tileset_import_uid = ""
    state.tileset_import_open = False
    state.tileset_preview_key = None
    _forget_slice_preview(ctx)


def tileset_popup_open(ctx: Any) -> bool:
    """Whether the tile-set import popup owns the keyboard. Tolerant of a
    partial ``ctx`` (``getattr``, not attribute access) -- ``matte_preview.
    is_open``'s reason: ``dialogs.modal_open`` asks this on every key press
    (I77), for a caller that has never built Packwright state.

    The 2026-09-16 audit found this popup missing from ``modal_open``'s
    answers: it is a real popup with the user's attention (the tile-size
    fields, the Import/Cancel pair), but every global shortcut still reached
    the app while it was up -- the same UX-08 shape ``modal_open``'s own
    docstring names the matte preview for, reproduced in a door that fix
    never reached.
    """
    state = getattr(getattr(ctx, "state", None), "packwright", None)
    return bool(state is not None and state.tileset_import_open)


def _hatch(
    draw: Any, lo: tuple[float, float], hi: tuple[float, float], colour: int, spacing: float
) -> None:
    """Diagonal lines filling one rect -- the remainder strip's mark.

    Clipped rather than measured: cutting a line off at the rect's own edges
    by hand needs the same trig at every call site, and the draw list already
    knows how to do it once.
    """
    if hi[0] <= lo[0] or hi[1] <= lo[1]:
        return
    draw.push_clip_rect(lo, hi, True)
    span = hi[1] - lo[1]
    x = lo[0] - span
    while x < hi[0]:
        draw.add_line((x, hi[1]), (x + span, lo[1]), colour, 1.0)
        x += spacing
    draw.pop_clip_rect()


def _slice_preview(ctx: Any, pixels: Any, tile: tuple[int, int]) -> None:
    """The sheet with its occupancy grid over it: what Import is about to keep.

    Kept cells (opaque somewhere in them) are **outlined**, so the sheet still
    shows through; dropped cells (fully transparent) are **dimmed**; the
    remainder strip -- the sliver along the right or bottom edge too narrow to
    make a whole tile, which ``tileset_occupancy`` already leaves out of its
    grid rather than this function deciding it a second time -- is
    **hatched**. Three different marks for three different fates, because the
    sentence above this already says the counts and a fourth repetition of the
    same three numbers would not tell anyone *which* cells.
    """
    from imgui_bundle import imgui

    from ..... import theme
    from .....tokens import sp

    grid = _occupancy_for(pixels, tile)
    if grid is None or grid.shape[0] == 0 or grid.shape[1] == 0:
        return
    height, width = pixels.shape[:2]
    tile_w, tile_h = int(tile[0]), int(tile[1])
    avail = max(imgui.get_content_region_avail().x, sp(80))
    zoom = min(avail / width, 1.0) if width > 0 else 1.0
    max_h = sp(220)
    if height > 0 and height * zoom > max_h:
        zoom = max_h / height
    draw_w, draw_h = width * zoom, height * zoom

    origin = imgui.get_cursor_screen_pos()
    texture = _slice_texture(ctx, pixels)
    if texture is None:
        # No GL context: the headless smoke suite and every state-only test.
        widgets.thumb_placeholder(draw_w, icons.GRID, draw_h)
    else:
        imgui.image(widgets.texture_ref(texture), (draw_w, draw_h))

    draw = imgui.get_window_draw_list()
    step_w, step_h = tile_w * zoom, tile_h * zoom
    rows, columns = grid.shape
    kept_colour = imgui.get_color_u32(theme.rgba(theme.OK, 0.9))
    dropped_colour = imgui.get_color_u32((0.0, 0.0, 0.0, 0.55))
    hatch_colour = imgui.get_color_u32(theme.rgba(theme.WARN, 0.7))
    for row in range(rows):
        for column in range(columns):
            lo = (origin.x + column * step_w, origin.y + row * step_h)
            hi = (lo[0] + step_w, lo[1] + step_h)
            if grid[row, column]:
                draw.add_rect(lo, hi, kept_colour, 0.0, max(sp(1.5), 1.0))
            else:
                draw.add_rect_filled(lo, hi, dropped_colour)

    # The remainder: whatever the grid above does not reach because the sheet's
    # own size leaves less than one tile on the right, the bottom, or both.
    # The bottom strip stops at the grid's own width so the corner -- covered
    # by the right strip's full height -- is not hatched twice.
    grid_w, grid_h = columns * step_w, rows * step_h
    if grid_w < draw_w:
        _hatch(
            draw,
            (origin.x + grid_w, origin.y),
            (origin.x + draw_w, origin.y + draw_h),
            hatch_colour,
            sp(6),
        )
    if grid_h < draw_h:
        _hatch(
            draw,
            (origin.x, origin.y + grid_h),
            (origin.x + grid_w, origin.y + draw_h),
            hatch_colour,
            sp(6),
        )


def _cell_pair(value: tuple[int, int]) -> tuple[int, int]:
    """Two small integer fields on one row, name above -- ``inker_bridge._pair``'s
    shape (2026-09-08 consistency pass: the name used to sit as muted text
    below-and-right of the pair, after both boxes on the same line; moved
    above them, matching the fix landing on ``inker_bridge._pair`` the same
    day so the two stay in agreement)."""
    from imgui_bundle import imgui

    from .....tokens import sp

    widgets.field_label("tile size")
    imgui.set_next_item_width(sp(70))
    _changed_x, x = controls.input_int("##tilew", int(value[0]), 1, 8)
    imgui.same_line()
    imgui.set_next_item_width(sp(70))
    _changed_y, y = controls.input_int("##tileh", int(value[1]), 1, 8)
    return (max(1, int(x)), max(1, int(y)))


def _tileset_popup(ctx: Any, state: Any) -> None:
    from imgui_bundle import imgui

    from ..... import theme
    from .....tokens import sp
    from ...engine.layout import MAX_SPRITES

    if not imgui.begin_popup(TILESET_POPUP):
        # imgui closes a popup on a click outside, and the sheet is a megabyte
        # or two: dropping it here is what keeps a cancelled import from
        # pinning the pixels for the rest of the session.
        #
        # Guarded on something actually being parked -- the 2026-09-18
        # audit's packwright-01: this ran unconditionally, so every frame the
        # popup was closed (which is most of them, for most of a session)
        # called ``clear_tileset_import`` -> ``release_prefix``, a scan of
        # the whole app-wide ``ctx.state.preview`` dict, to forget a texture
        # that was never parked in the first place.
        if state.tileset_import is not None or state.tileset_import_open:
            clear_tileset_import(ctx, state)
        return
    widgets.popup_chrome(_imgui=imgui)
    if state.tileset_import is None:
        _forget_slice_preview(ctx)
        imgui.end_popup()
        return
    path, stem, pixels = state.tileset_import
    height, width = pixels.shape[:2]
    widgets.muted(f"{stem} - {width} x {height}")

    state.tileset_cell = _cell_pair(state.tileset_cell)
    state.tileset_dedup = controls.checkbox(
        "Drop duplicate tiles", state.tileset_dedup
    )[1]
    if state.tileset_dedup:
        imgui.indent()
        state.tileset_dedup_flips = controls.checkbox(
            "match flipped / rotated", state.tileset_dedup_flips
        )[1]
        imgui.unindent()

    # The counts the numbers above actually produce, computed from the same
    # occupancy grid the import slices by -- so what the popup promises and
    # what the import does cannot disagree.
    #
    # **Off the frame thread.** The 2026-09-07 audit's packwright-05: the
    # dedup branch re-slices the sheet into up to ``MAX_SPRITES`` fresh
    # ``Sprite`` copies and hashes all eight dihedral variants of each --
    # measured at some hundreds of milliseconds on a full 8192-square sheet --
    # and this used to run inline, on the frame thread, once per typed digit
    # in either tile-size field. ``request_tileset_preview`` submits it and
    # ``on_task_done`` adopts the answer into ``state.tileset_preview`` when it
    # lands; until then this shows the last one it has, marked as pending
    # rather than silently stale.
    computing = state.tileset_preview_key != packwright_mode.tileset_preview_key(
        pixels, state.tileset_cell, state.tileset_dedup, state.tileset_dedup_flips
    )
    packwright_mode.request_tileset_preview(ctx, state, path, stem, pixels)
    rows, columns, kept, dropped, duplicates = state.tileset_preview
    problem = ""
    if not computing:
        if kept == 0:
            problem = "No occupied cells at that tile size."
        elif kept > MAX_SPRITES:
            problem = f"{kept} tiles; the packer's ceiling is {MAX_SPRITES}."
    if computing:
        widgets.muted("Counting the tiles this would keep...")
    elif problem:
        widgets.text_colored(theme.WARN, problem)
    elif duplicates:
        widgets.muted(
            f"{kept} unique of {kept + duplicates} tiles "
            f"({duplicates} duplicate(s) dropped), {dropped} empty dropped"
        )
    else:
        widgets.muted(f"{columns} x {rows} cells - {kept} tile(s), {dropped} empty dropped")

    imgui.dummy((0, sp(tokens.SP_1)))
    _slice_preview(ctx, pixels, state.tileset_cell)

    imgui.dummy((0, sp(tokens.SP_1)))
    imgui.begin_disabled(bool(problem) or computing)
    if controls.button("Import", (sp(90), 0)) and packwright_mode.import_tileset(ctx):
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.same_line()
    if controls.button("Cancel##tileset", (sp(90), 0)):
        clear_tileset_import(ctx, state)
        imgui.close_current_popup()
    imgui.end_popup()


def _from_inker(ctx: Any, tab: Any, editable: bool) -> None:
    """One button per open Inker document, or nothing at all.

    Nothing rather than a disabled button: with no document open the control
    would be explaining a mode the user may not have visited, and the sources
    list already says what can be added.
    """
    from imgui_bundle import imgui

    inker = ctx.state.inker
    if inker is None or not inker.docs:
        return
    imgui.dummy((0, sp(tokens.SP_1)))
    widgets.muted("From Inker")
    for doc in inker.docs:
        frames = len(doc.doc.anim.frames) if doc.doc.anim is not None else len(doc.doc.stack)
        what = "frame" if doc.doc.anim is not None else "layer"
        label = f"{doc.title} ({frames} {what}{'s' if frames != 1 else ''})"
        if widgets.disabled_button(f"{icons.FILM} {label}##ink-{doc.uid}", editable, (-1, 0)):
            packwright_mode.add_inker_document(ctx, doc)


def _pivot_row(ctx: Any, tab: Any, source: Any) -> None:
    """Where this sprite's anchor sits, in its own untrimmed pixels.

    The pivot was modelled end to end -- ``SpriteMeta.pivot``, the layout frame,
    the exported sidecar -- and could only be *set* by importing an Inker
    document that already carried one. A loose PNG had no way to say where its
    feet were, which is the one thing an atlas is asked for after "where is the
    rectangle".

    A checkbox and two fields rather than a always-on pair, because clearing it
    is a real answer and is not the same as the centre: an absent pivot takes
    the format's documented 0.5/0.5 default and one pinned at the centre says
    so. The preview draws a cross wherever this puts it.
    """
    from imgui_bundle import imgui

    from .....tokens import sp

    sprite = source.sprite
    pivot = sprite.meta.pivot
    changed, wanted = controls.checkbox("Anchor", pivot is not None)
    if changed:
        # Defaulting to the centre of the untrimmed picture, which is where the
        # format would have put it anyway -- so ticking the box changes nothing
        # until the numbers are moved, and the *file* starts saying it.
        packwright_mode.set_pivot(
            ctx,
            tab,
            source.uid,
            (sprite.width / 2.0, sprite.height / 2.0) if wanted else None,
        )
        return
    if pivot is None:
        return
    imgui.set_next_item_width(sp(64))
    moved_x, x = controls.drag_float(
        "##pivotx", float(pivot[0]), 0.5, 0.0, float(sprite.width)
    )
    # One gesture, one step: a drag reports on every frame the pointer moves,
    # and ``set_pivot`` pushes a step per report without this.
    controls.fold_undo(tab.doc.history)
    imgui.same_line()
    imgui.set_next_item_width(sp(64))
    moved_y, y = controls.drag_float(
        "##pivoty", float(pivot[1]), 0.5, 0.0, float(sprite.height)
    )
    controls.fold_undo(tab.doc.history)
    if moved_x or moved_y:
        packwright_mode.set_pivot(ctx, tab, source.uid, (x, y))
    widgets.muted("px from this sprite's own top-left, before any trim")


def _row(ctx: Any, state: Any, tab: Any, source: Any, editable: bool) -> None:
    from imgui_bundle import imgui

    imgui.push_id(str(source.uid))
    selected = state.selected == source.uid
    if state.renaming == source.uid and editable:
        # **Double-click renames**, which is what Plotter's layers and Clay's
        # outliner do; this list grew an inline field under the selected row
        # instead, so the same gesture renamed in two lists and did nothing in
        # the third (2026-09-05).
        imgui.set_next_item_width(-1.0)
        # commit=True: the 2026-09-07 audit's packwright-03 found this field
        # reporting a change on every keystroke, unlike the identical widget
        # in ``clay_outliner.py`` -- so ``rename_source`` (an unconditional
        # ``history.push``) fired once per letter typed, and typing "lead"
        # then one Ctrl+Z left "lea" rather than undoing the rename.
        name = widgets.input_text("##rename", source.name, max_length=64, commit=True)
        if name != source.name:
            # Through the mode, not onto the document: the mode is what re-arms
            # the pack, and a name that never reaches the layout is a name the
            # exported sidecar does not carry.
            packwright_mode.rename_source(ctx, tab, source.uid, name)
        if imgui.is_item_deactivated():
            state.renaming = None
    else:
        if controls.selectable(f"{source.name}##src", selected)[0]:
            state.selected = None if selected else source.uid
        if imgui.is_item_hovered():
            sprite = source.sprite
            imgui.set_tooltip(f"{sprite.width} x {sprite.height}\n{sprite.key}")
            if imgui.is_mouse_double_clicked(0) and editable:
                state.renaming = source.uid
        if imgui.begin_popup_context_item("src-menu"):
            widgets.popup_chrome(_imgui=imgui)
            # ``enabled=editable``, not ``and editable`` on the click: the
            # 2026-09-23 audit's packwright-02 found these two menu items drawn
            # fully enabled while a save was writing -- so a click during a
            # save looked like it worked (the item highlighted, the popup
            # closed) and silently did nothing, the exact "greyed with a
            # reason" contract ``widgets.disabled_button`` already gives the
            # Add buttons above. ``menu_item_simple`` folds ``enabled`` into
            # its own return, so nothing else here has to gate the effect.
            if controls.menu_item_simple(
                f"{icons.PENCIL} Rename", enabled=editable, reason=widgets.DOCUMENT_SAVING_WHY
            ):
                state.renaming = source.uid
            if controls.menu_item_simple(
                "Remove", enabled=editable, reason=widgets.DOCUMENT_SAVING_WHY
            ):
                packwright_mode.remove_source(ctx, source.uid, tab)
            imgui.end_popup()
    if selected and editable:
        _pivot_row(ctx, tab, source)
        # Remove, not Delete: the source leaves this atlas and the file it
        # came from is untouched. ``MINUS`` is the detach glyph; ``TRASH`` is
        # for what destroys the thing (2026-09-05).
        if controls.button(f"{icons.MINUS} Remove", (-1, 0)):
            packwright_mode.remove_source(ctx, source.uid, tab)
    imgui.pop_id()
