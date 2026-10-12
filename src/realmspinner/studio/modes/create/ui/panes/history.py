"""Create's **History** pane: every attempt that was made or judged, deleted ones
included, newest first.

It replaces the "Creations" list, which grouped live assets by lineage and so
could say nothing about the work that was thrown away. This one reads
``service.generation_log.history`` -- the verdict and observation tables joined
to the jobs that still exist -- so a hard-deleted attempt keeps its prompt, its
labels, its notes and the settings line of what ran, and **Load settings**
works on it from the snapshot the verdict stored.

**Nothing here runs on the frame thread but drawing.** The join is a task
(``KEY``), landed by ``shell/tasks.py`` into ``CreateState.history_rows``; a
reload is asked for by a flag (``history_dirty``) raised when a verdict is filed
(``review_mode.refresh_findings``), when a job finishes or is deleted (the
jobs-cache signature, :func:`pump`) and when a filter or the page changes. The
flag is cleared only when the submit is *accepted* -- the ``findings_dirty``
rule, because ``TaskRunner.submit`` refuses a key already in flight.

Thumbnails are the cache's own background decodes (``ctx.textures.get(...,
background=True)``), registered with and forgotten by the imgui backend like
every other texture, and requested only for the rows the clipper shows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from ......bench import findings as findings_lib
from ......service import generation_log
from ..... import asset_open, controls, generation_log_export, theme, tokens, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ...engine import history as engine

#: The task key a load runs under, and the one ``shell/tasks.py`` lands.
KEY = "create-history"

#: The thumbnail's side in design px, and the longest side it is decoded at.
_THUMB_DP = 56.0
_THUMB_TEXTURE_SIDE = 128

#: Lines of text beside the thumbnail: the prompt, the marks and tags, and the
#: settings -- and under them, on a line of its own, **Load settings**.
_TEXT_LINES = 3


# --- loading -------------------------------------------------------------------------


def load(
    svc: Any, filters: dict[str, Any], limit: int, want_options: bool
) -> dict[str, Any]:
    """The task body: one page of rows (one extra, to know whether there is
    more) and, when asked, the options derived from the whole log. **Off the
    frame thread.**

    The options come from the whole unfiltered log because a filtered page
    cannot name the models it filtered out; they are asked for only when the log
    itself changed (a verdict, a finished or deleted job), never for a filter or
    a page. They ride the one pass that reads the page: the 2026-10-11 audit,
    finding create-20, found a second ``history(limit=1_000_000)`` call here, so
    every finished generation cost two whole-library joins.
    """
    rows, option_rows = generation_log.history_page(
        svc,
        **engine.service_filters(filters),
        limit=limit + 1,
        offset=0,
        want_options=want_options,
    )
    options = engine.options_from(option_rows) if want_options else None
    return {"rows": rows, "options": options}


def pump(ctx: Any) -> None:
    """Notice what changed and submit a reload if one is due. Every frame, cheap."""
    state = ctx.state.create
    cache = ctx.cache
    generation = getattr(cache, "_generation", None)
    if generation is not None and (state.history_seen or (None,))[0] != generation:
        signature = engine.jobs_signature(list(cache.jobs))
        previous = state.history_seen[1] if state.history_seen else None
        state.history_seen = (generation, signature)
        if previous is not None and signature != previous:
            state.history_dirty = True
            state.history_options_stale = True
    if not state.history_dirty:
        return
    tag = {"filters": dict(state.history_filters), "limit": state.history_limit}
    if ctx.submit(
        KEY,
        load,
        ctx.svc,
        dict(state.history_filters),
        state.history_limit,
        state.history_options_stale or not state.history_options,
        tag=tag,
    ):
        state.history_dirty = False
        state.history_options_stale = False


def land(ctx: Any, done: Any) -> None:
    """A finished load. Options always merge; rows only when the request they
    answer is still the one on screen (a filter changed mid-flight would
    otherwise flash the old filter's rows)."""
    state = ctx.state.create
    result = done.result if isinstance(done.result, dict) else {}
    if result.get("options"):
        state.history_options = engine.merge_options(state.history_options, result["options"])
    tag = done.tag if isinstance(done.tag, dict) else {}
    if tag.get("filters") != state.history_filters or tag.get("limit") != state.history_limit:
        return
    rows = list(result.get("rows") or [])
    state.history_more = len(rows) > state.history_limit
    state.history_rows = rows[: state.history_limit]
    state.history_error = ""


def on_task_failed(ctx: Any, done: Any) -> None:
    ctx.state.create.history_error = done.message or "History could not be read."


def refresh(ctx: Any) -> None:
    """Ask for a reload (a filter, a page, or an outside change)."""
    ctx.state.create.history_dirty = True


# --- Load settings ---------------------------------------------------------------------


def load_settings(ctx: Any, row: dict[str, Any]) -> bool:
    """Write a row's recorded settings into the recipe column. -> whether it did.

    A live attempt goes through ``workspace._vary`` -- the very path the tray's
    **Load settings** takes, with the full recipe the job kept. A deleted one
    has only its snapshot: it is dressed as the minimal job ``copy_settings``
    reads (:func:`engine.snapshot_job`), so the same restore runs on it, and the
    base model, which the snapshot names exactly, becomes an Advanced pick
    unless Automatic would have chosen it anyway. The mesh settings it carries
    fill the Mesh form.

    Refused with the reason in a toast when the snapshot names a model or style
    that was removed -- loading it would put a key in the form every door
    refuses, and substituting another would record a settings line that ran
    something else.
    """
    from .....state import DEFAULT_FORM_3D
    from ....library.ui.panes import library
    from .. import workspace
    from . import settings_2d

    reason = engine.load_block(row)
    if reason:
        ctx.toast(reason, "error")
        return False
    job = ctx.cache.get(row["job_id"]) if row.get("exists") and not row.get("trashed") else None
    if job is not None:
        workspace._vary(ctx, job)
        return True
    snapshot = row.get("snapshot") or {}
    library.copy_settings(ctx, engine.snapshot_job(row), announce=False, follow=False)
    form = ctx.state.form_2d
    base = str(snapshot.get("base_model") or "")
    if base:
        _choose_base(ctx, form, base, settings_2d)
    mesh = engine.mesh_updates(snapshot, DEFAULT_FORM_3D)
    if mesh:
        ctx.state.form_3d.update(mesh)
        if "mesh_seed" in mesh:
            ctx.state.form_3d["mesh_seed_locked"] = True
    ctx.toast(
        "Loaded these settings from History"
        + (" and locked their seed" if form.get("seed_locked") else "")
        + ". Change one thing, then generate a variation."
    )
    return True


def _choose_base(ctx: Any, form: dict[str, Any], base: str, settings_2d: Any) -> None:
    """Advanced with ``base`` unless Automatic would already load it."""
    from ...engine import recipe as create_recipe

    form["model_mode"] = "auto"
    form["model_override"] = ""
    try:
        resolved = create_recipe.resolved_recipe(ctx, form)
        automatic = str(getattr(resolved, "base_model", "") or "")
    except Exception:  # noqa: BLE001 - a form that will not resolve is simply not automatic
        automatic = ""
    if automatic != base:
        settings_2d._pick_base_model(ctx, form, base)
    else:
        form["base_model"] = base


# --- drawing ---------------------------------------------------------------------------


def draw(ctx: Any) -> None:
    """The right column's History slot: one card (title, Export, (?), the corpus
    line and the four filters), then a card per attempt."""
    state = ctx.state.create
    pump(ctx)
    with widgets.group(
        "History",
        id="history",
        actions=(("export", "Export...", lambda: generation_log_export.request(ctx)),),
    ):
        # First in the body, so it lands on the card's header row beside Export.
        manual_render.help_button(ctx, "create-history")
        summary = findings_lib.corpus_line(
            findings_lib.load(Path(ctx.svc.config.bench_dir) / "findings.json")
        )
        # One line, trimmed to the card's width: the corpus the judgements below
        # feed, said by Review's own reader.
        widgets.muted(
            widgets.fit_text(
                summary or "Nothing judged yet.", max(1.0, imgui.get_content_region_avail().x)
            )
        )
        _filters(ctx)
    rows = state.history_rows
    if rows is None:
        widgets.muted(state.history_error or "Loading history...")
        return
    if state.history_error:
        widgets.muted_wrapped(state.history_error)
    if not rows:
        widgets.muted_wrapped(
            "Nothing here yet. Every attempt is kept, and so is what you say about it."
            if not _filtering(state)
            else "Nothing matches these filters."
        )
        return
    _list(ctx, rows)
    if state.history_more and controls.button("More##history-more", (-1, 0)):
        state.history_limit += engine.PAGE
        refresh(ctx)


def _filtering(state: Any) -> bool:
    return any(v is not None for v in state.history_filters.values())


def _filters(ctx: Any) -> None:
    """Four compact combos in a 2 x 2 grid: models | styles over types | judged.
    Options come from the rows, retired keys included, so the filter that finds
    old work outlives the model; the chosen key is always offered even if a later
    load no longer lists it.

    Two across, not "full, full, half, half" as it was: the first two were the
    only combos at a different width from the other two, which read as a
    hierarchy among four peers (2026-10-11 spacing review)."""
    state = ctx.state.create
    filters = state.history_filters
    options = state.history_options
    changed = False
    width = widgets.grid_width(2)

    def pick(field: str, everything: str) -> None:
        nonlocal changed
        entries = list(options.get(field, ()))
        current = filters.get(field)
        if current is not None and current not in {k for k, _ in entries}:
            entries.append((str(current), str(current)))
        shown = widgets.combo(
            f"##history-{field}",
            "" if current is None else str(current),
            [("", everything), *entries],
            width,
        )
        if (shown or None) != current:
            filters[field] = shown or None
            changed = True

    pick("base_model", "All models")
    imgui.same_line()
    pick("style_lora", "All styles")
    pick("generation_type", "All types")
    imgui.same_line()
    shown = widgets.combo(
        "##history-judged",
        engine.judged_key(filters.get("judged")),
        list(engine.JUDGED_OPTIONS),
        width,
    )
    if engine.judged_value(shown) is not filters.get("judged"):
        filters["judged"] = engine.judged_value(shown)
        changed = True
    if changed:
        state.history_limit = engine.PAGE
        refresh(ctx)


class RowMetrics:
    """One History row's geometry, in screen px. **Derived, never guessed.**

    The ListClipper multiplies one number by the row count, so every row must be
    exactly this tall. The number is the sum of what a row holds -- a 12 px inset
    top and bottom, three text lines 4 px apart, 8 px, then the 26 px **Load
    settings** line -- so a font or UI-scale change moves it instead of clipping the
    button (the old row was a hand-set ``56 + 2 * 6`` with the button laid over its
    last text line).
    """

    __slots__ = ("pad", "thumb", "line", "line_gap", "gap", "button", "text_h", "row_h")

    def __init__(self) -> None:
        self.pad = sp(tokens.SP_3)
        self.thumb = sp(_THUMB_DP)
        self.line = float(imgui.get_text_line_height())
        self.line_gap = sp(tokens.SP_1)
        self.gap = sp(tokens.SP_2)
        self.button = sp(tokens.CONTROL_HEIGHT_COMPACT)
        self.text_h = (
            _TEXT_LINES * self.line + (_TEXT_LINES - 1) * self.line_gap + self.gap + self.button
        )
        self.row_h = 2 * self.pad + max(self.text_h, self.thumb)

    def line_y(self, top: float, index: int) -> float:
        """The y of text line ``index`` (0 = prompt, 1 = marks, 2 = settings)."""
        return top + self.pad + index * (self.line + self.line_gap)

    def button_y(self, top: float) -> float:
        """The y of the Load settings line, ``gap`` under the last text line."""
        return self.line_y(top, _TEXT_LINES - 1) + self.line + self.gap


def _list(ctx: Any, rows: list[dict[str, Any]]) -> None:
    """The rows, clipped: every row is the same height, so the clipper's
    measurement is exact and only the rows on screen request a texture. A row is
    drawn by hand as a card surface (a child window per row would make the height
    imgui's to decide, and the clipper's whole premise is that it is ours)."""
    spacing = imgui.get_style().item_spacing.y
    metrics = RowMetrics()
    clipper = imgui.ListClipper()
    clipper.begin(len(rows), metrics.row_h + spacing)
    while clipper.step():
        for index in range(clipper.display_start, clipper.display_end):
            _row(ctx, rows[index], metrics)
    clipper.end()


def _row(ctx: Any, row: dict[str, Any], metrics: RowMetrics) -> None:
    job_id = str(row["job_id"])
    height = metrics.row_h
    imgui.push_id(job_id)
    width = max(1.0, imgui.get_content_region_avail().x)
    low = imgui.get_cursor_screen_pos()
    clicked = imgui.invisible_button("##row", (width, height))
    hovered = imgui.is_item_hovered()
    draw = imgui.get_window_draw_list()
    high = imgui.ImVec2(low.x + width, low.y + height)
    # The card surface every group wears (ELEV_2 at 0.5, RADIUS_M); pointing at
    # it firms the fill up rather than swapping to a different surface.
    draw.add_rect_filled(
        low,
        high,
        imgui.get_color_u32(theme.rgba(theme.ELEV_2, 0.9 if hovered else 0.5)),
        sp(tokens.RADIUS_M),
    )
    pad = metrics.pad
    _thumb(ctx, row, low.x + pad, low.y + pad, metrics.thumb)

    left = low.x + pad + metrics.thumb + pad
    room = max(1.0, high.x - left - pad)
    gone = engine.gone_text(row)
    muted = imgui.get_color_u32(theme.rgba(theme.MUTED))
    text = imgui.get_color_u32(theme.rgba(theme.TEXT))

    prompt = " ".join(str(row.get("prompt") or "").split()) or "(no prompt kept)"
    flag_w = imgui.calc_text_size(gone).x + pad if gone else 0.0
    draw.add_text((left, metrics.line_y(low.y, 0)), text, widgets.fit_text(prompt, room - flag_w))
    if gone:
        flag_x = high.x - pad - imgui.calc_text_size(gone).x
        draw.add_text((flag_x, metrics.line_y(low.y, 0)), muted, gone)

    marks = _marks(row)
    mark_y = metrics.line_y(low.y, 1)
    mark_x = left
    for words, colour in marks:
        draw.add_text((mark_x, mark_y), imgui.get_color_u32(theme.rgba(colour)), words)
        mark_x += imgui.calc_text_size(words).x + sp(tokens.SP_2)
    tags = engine.row_tags(row)
    notes = engine.row_notes(row)
    tail = ", ".join(tags) + (" · note" if notes else "") if tags or notes else ""
    if tail:
        used = mark_x - left
        draw.add_text(
            (mark_x, mark_y), muted, widgets.fit_text(tail, max(1.0, room - used))
        )
    settings = str(row.get("settings_line") or "")
    draw.add_text(
        (left, metrics.line_y(low.y, 2)),
        muted,
        widgets.fit_text(settings or "no settings kept", room),
    )

    # **Load settings has a line of its own** under the settings text, at the text
    # column. It used to sit on the last line's right edge -- over the settings
    # line itself, whose tail it covered (2026-10-11 spacing review, finding 20).
    # Items drawn later take the click over the row's invisible button.
    unavailable = engine.no_settings_reason(row)
    imgui.set_cursor_screen_pos((left, metrics.button_y(low.y)))
    controls.push_elevation()  # the row is a card: ghost hover must read on its fill
    try:
        pressed = controls.small_button(
            "Load settings##history-load",
            enabled=not unavailable,
            reason=unavailable,
            tooltip="Copy this attempt's recipe back into the form, to change one thing.",
        )
    finally:
        controls.pop_elevation()
    if pressed:
        load_settings(ctx, row)
    over_button = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value)

    if hovered and not over_button:
        imgui.set_tooltip(_tooltip(row, tags, notes))
    openable = bool(row.get("exists")) and not row.get("trashed")
    if clicked and openable and not over_button:
        found = ctx.cache.get(job_id)
        asset_open.open_asset(ctx, found if found is not None else job_id)
    imgui.set_cursor_screen_pos((low.x, high.y))
    imgui.dummy((0, 0))
    imgui.pop_id()


def _marks(row: dict[str, Any]) -> list[tuple[str, tuple[float, ...]]]:
    """The image label and the mesh grade, each with its colour. **Pure.**"""
    out: list[tuple[str, tuple[float, ...]]] = []
    image = engine.image_mark(row.get("reference_label"))
    if image:
        out.append((f"Image {image}", theme.OK if image == "Good" else theme.ERR))
    grade = engine.grade_text(row.get("mesh"))
    if grade:
        label = row.get("mesh") or {}
        out.append((f"Mesh {grade}", theme.OK if label.get("verdict") == "accept" else theme.ERR))
    return out


def _thumb(ctx: Any, row: dict[str, Any], x: float, y: float, side: float) -> None:
    draw = imgui.get_window_draw_list()
    low, high = imgui.ImVec2(x, y), imgui.ImVec2(x + side, y + side)
    texture = None
    if row.get("thumbnail") and ctx.textures is not None:
        texture = ctx.textures.get(
            f"create-history:{row['job_id']}",
            Path(str(row["thumbnail"])),
            max_side=_THUMB_TEXTURE_SIDE,
            background=True,
        )
    if texture is None:
        draw.add_rect_filled(
            low, high, imgui.get_color_u32(theme.rgba(theme.BG)), sp(tokens.RADIUS_S)
        )
        return
    factor = min(side / texture.size[0], side / texture.size[1])
    w, h = texture.size[0] * factor, texture.size[1] * factor
    x0, y0 = x + (side - w) / 2, y + (side - h) / 2
    draw.add_image(widgets.texture_ref(texture), (x0, y0), (x0 + w, y0 + h))


def _tooltip(row: dict[str, Any], tags: list[str], notes: list[str]) -> str:
    lines = [str(row.get("prompt") or "(no prompt kept)")]
    if row.get("settings_line"):
        lines.append(str(row["settings_line"]))
    if tags:
        lines.append("Tags: " + ", ".join(tags))
    for note in notes:
        lines.append(f"Note: {note}")
    gone = engine.gone_text(row)
    if gone:
        lines.append(f"This attempt is {gone}; its labels and settings are kept.")
    return "\n".join(lines)

