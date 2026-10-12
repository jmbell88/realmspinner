"""What Create's canvas draws besides the viewer: the contact sheet, the empty
states and the canvas's own Make 3D.

``shell.frame._viewport_pane`` decides *which* of these (or the viewer) fills
the canvas; this module draws them. Split from ``workspace.py`` (the tray, the
plan footer and the inspector's actions) because those are about attempts as
rows and these are about the picture in the middle of the screen.

**The contact sheet** is the answer to "Count 8 gave me eight 82 px thumbnails
in a strip": when a press with two or more images has finished, and one of
those images is the selection, the canvas shows all of them at once, scaled to
fit (:func:`families.grid_shape`). Click chooses, the arrow keys move, Enter is
Make 3D and Esc returns to one image at a time. :func:`grid_cell` is the one
function that draws a cell, so the labelling -- Good or Bad from the keyboard,
tags, a note -- has one place to add its marks and :func:`handle_keys` its keys.

**Labelling** is the image half of what Review's labelling pass does, done where
the images are: G files ``accept`` and B ``reject`` against the image under the
choice (``stage="reference"``, ``source="human"``, through
``inspector.record_label``), N opens a small popup for tags and a note. The
latest label per image is read once and kept in ``CreateState.label_memo``, so
a cell never costs a query per frame; filing one raises the findings and History
refresh. The single-image view carries the same buttons and keys.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .....service import verdicts as svc_verdicts
from .... import controls, dialogs, theme, tokens, widgets
from ....panes import model_gate, thumbs
from ....tokens import sp
from ..engine import assets as create_assets
from ..engine import mesh as create_mesh
from ..engine import recipe as create_recipe
from ..engine import workspace as families
from . import session

#: The two answers a reference image takes, and the key that files each. Plain
#: letters, read only under :func:`keys_enabled`; neither is bound anywhere else
#: in Create (F, W and S are the viewer's, Alt+1-5 the rail's).
LABEL_KEYS = (("accept", imgui.Key.g), ("reject", imgui.Key.b))
LABEL_NOTE_KEY = imgui.Key.n
LABEL_WORDS = {"accept": "Good", "reject": "Bad"}
_LABEL_POPUP = "create-label-popup"

#: The longest side a cell's texture is decoded at. The sheet shows at most
#: eight images on a canvas a fraction of a window wide, so this is generous.
GRID_TEXTURE_SIDE = 768

#: The gap between cells and the inset of a picture inside its cell, design px.
_GAP_DP = 8.0
_INSET_DP = 6.0

#: The reference kinds the canvas's Make 3D is offered for -- the same set the
#: inspector's old "Use as source for 3D" used, so the two cannot disagree.
MAKE_3D_KINDS = ("", "image", "3d_model")


# --- which batch the sheet is about -------------------------------------------


def grid_batch(ctx: Any, *, ignore_dismissed: bool = False) -> list[dict[str, Any]] | None:
    """The batch the contact sheet should show right now, or ``None``.

    All of these must hold, and each is a thing the review asked for by name:
    the Reference stage; the newest press of this creation made two or more
    images; every one of them has settled (a sheet that fills in while you
    watch is the strip again); at least two of them produced a picture; the
    selection is one of them (so choosing something else, anywhere, ends it);
    and nobody closed this batch's sheet with Esc or the toggle
    (``ignore_dismissed`` drops that last clause, for the button that reopens it).
    """
    state = ctx.state
    if state.create.stage != "reference" or state.mode != "create":
        return None
    key = state.create.workspace
    if key is None:
        return None
    batch = families.reference_batch(families.results(session.index(ctx), key, "reference"))
    if len(batch) < 2 or not families.batch_settled(batch):
        return None
    if sum(1 for j in batch if _has_picture(j)) < 2:
        return None
    if str(state.selected) not in {str(j["id"]) for j in batch}:
        return None
    if not ignore_dismissed and str(batch[0]["id"]) in state.create.grid_dismissed:
        return None
    return batch


def _has_picture(job: dict[str, Any]) -> bool:
    return job.get("status") == "done" and "input.png" in (job.get("files") or [])


# --- the sheet -----------------------------------------------------------------


def draw_grid(ctx: Any, width: float, height: float) -> bool:
    """Draw the contact sheet if there is one to draw. -> whether it did."""
    batch = grid_batch(ctx)
    if batch is None:
        return False
    ids = [str(j["id"]) for j in batch]
    index = ids.index(str(ctx.state.selected))
    selected = batch[index]
    prime_labels(ctx, ids)

    widgets.muted(f"{len(batch)} images")
    imgui.same_line()
    can_3d = _has_picture(selected)
    # Right-aligned pair on the header row. ``calc_text_size`` rather than a
    # constant so the buttons follow the UI scale and the font.
    labels = ("Make 3D##grid-make-3d", "Single view##grid-single")
    widths = [widgets.button_width(label.split("##")[0]) for label in labels]
    room = imgui.get_content_region_avail().x
    need = sum(widths) + imgui.get_style().item_spacing.x
    if room > need:
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + room - need)
    else:
        imgui.new_line()
    if widgets.primary_button(
        labels[0],
        enabled=can_3d,
        reason="This image did not finish; choose one that did.",
        tooltip="Check the cutout, then reconstruct the chosen image as a 3D model. "
        "Enter does the same.",
    ):
        from . import workspace

        workspace.make_3d(ctx, selected)
    imgui.same_line()
    if controls.button(labels[1], role=controls.ButtonRole.GHOST):
        dismiss(ctx, batch)
    widgets.muted_wrapped(
        "Click to choose, arrow keys to move, G good, B bad, N note, "
        "Enter to make 3D, Esc for one image."
    )

    avail = imgui.get_content_region_avail()
    rows, cols = families.grid_shape(len(batch))
    gap = sp(_GAP_DP)
    cell_w = max(1.0, (avail.x - gap * (cols - 1)) / cols)
    cell_h = max(1.0, (avail.y - gap * (rows - 1) - sp(4)) / rows)
    origin = imgui.get_cursor_screen_pos()
    for i, job in enumerate(batch):
        row, col = divmod(i, cols)
        imgui.set_cursor_screen_pos(
            (origin.x + col * (cell_w + gap), origin.y + row * (cell_h + gap))
        )
        if grid_cell(ctx, job, (cell_w, cell_h), selected=i == index, index=i):
            choose(ctx, job)
    # One item to account for the whole sheet, drawn last so it sits under
    # nothing: the cells were placed by cursor and imgui must still be told how
    # far this window now reaches.
    imgui.set_cursor_screen_pos(origin)
    imgui.dummy((avail.x, avail.y))

    if keys_enabled(ctx):
        handle_keys(ctx, batch, index)
    label_popup(ctx)
    return True


def grid_cell(
    ctx: Any,
    job: dict[str, Any],
    size: tuple[float, float],
    *,
    selected: bool,
    index: int,
) -> bool:
    """Draw one cell of the contact sheet. -> whether it was clicked.

    **The single place a cell is drawn**, on purpose: the labelling work that
    follows (Good/Bad keys, tags, a note) adds its marks and its tooltip lines
    here and its keys to :func:`handle_keys`, and nothing else in the sheet
    needs to know. ``size`` is the cell's full rectangle in pixels; the picture
    is scaled to fit inside it and centred, never stretched, and a member that
    did not produce one says why instead of leaving a hole.
    """
    job_id = str(job["id"])
    imgui.push_id(job_id)
    low = imgui.get_cursor_screen_pos()
    clicked = imgui.invisible_button("##cell", size)
    hovered = imgui.is_item_hovered()
    high = imgui.ImVec2(low.x + size[0], low.y + size[1])
    draw = imgui.get_window_draw_list()
    draw.add_rect_filled(low, high, imgui.get_color_u32(theme.rgba(theme.ELEV_1)), sp(4))
    inset = sp(_INSET_DP)
    texture = None
    if _has_picture(job) and ctx.textures is not None:
        texture = ctx.textures.get(
            f"create-grid:{job_id}",
            ctx.job_dir(job_id) / "input.png",
            max_side=GRID_TEXTURE_SIDE,
            background=True,
        )
    if texture is not None:
        room_w, room_h = max(1.0, size[0] - 2 * inset), max(1.0, size[1] - 2 * inset)
        factor = min(room_w / texture.size[0], room_h / texture.size[1])
        w, h = texture.size[0] * factor, texture.size[1] * factor
        x0, y0 = low.x + (size[0] - w) / 2, low.y + (size[1] - h) / 2
        draw.add_image(widgets.texture_ref(texture), (x0, y0), (x0 + w, y0 + h))
    else:
        status = str(job.get("status") or "")
        words = (
            "Loading..."
            if status == "done"
            else ("This one failed" if status == "error" else status.capitalize())
        )
        extent = imgui.calc_text_size(words)
        draw.add_text(
            (low.x + (size[0] - extent.x) / 2, low.y + (size[1] - extent.y) / 2),
            imgui.get_color_u32(theme.rgba(theme.MUTED)),
            words,
        )
    # The number is the order the press drew them in, which is also the order
    # the arrow keys walk -- so a reader can say "the third one".
    draw.add_text(
        (low.x + sp(8), low.y + sp(6)),
        imgui.get_color_u32(theme.rgba(theme.MUTED)),
        str(index + 1),
    )
    label = label_of(ctx, job_id)
    if label:
        _label_marks(draw, low, high, size, label)
    if selected:
        draw.add_rect(
            low, high, imgui.get_color_u32(theme.rgba(theme.ACCENT)), sp(4), thickness=sp(2.5)
        )
    elif hovered:
        draw.add_rect(
            low, high, imgui.get_color_u32(theme.rgba(theme.MUTED, 0.6)), sp(4), thickness=sp(1.5)
        )
    if hovered:
        seed = (job.get("params") or {}).get(
            "reference_seed", (job.get("params") or {}).get("seed")
        )
        lines = [f"Image {index + 1}" + (f" - seed {seed}" if seed is not None else "")]
        lines += _label_lines(label)
        imgui.set_tooltip("\n".join(lines))
    imgui.pop_id()
    return clicked


def _label_marks(draw: Any, low: Any, high: Any, size: tuple[float, float], label: dict) -> None:
    """A cell's label: a coloured edge (green Good, red Bad), the word at the top
    right, and the tags along the bottom. Drawn under the selection ring, which
    stays the louder of the two."""
    good = label.get("verdict") == "accept"
    colour = theme.OK if good else theme.ERR
    draw.add_rect(
        (low.x + sp(1), low.y + sp(1)),
        (high.x - sp(1), high.y - sp(1)),
        imgui.get_color_u32(theme.rgba(colour, 0.9)),
        sp(4),
        thickness=sp(2),
    )
    word = LABEL_WORDS.get(str(label.get("verdict")), "") + (" · note" if label.get("note") else "")
    extent = imgui.calc_text_size(word)
    draw.add_text(
        (high.x - extent.x - sp(8), low.y + sp(6)), imgui.get_color_u32(theme.rgba(colour)), word
    )
    tags = ", ".join(label.get("reasons") or ())
    if tags:
        draw.add_text(
            (low.x + sp(8), high.y - imgui.get_text_line_height() - sp(6)),
            imgui.get_color_u32(theme.rgba(theme.MUTED)),
            widgets.fit_text(tags, max(1.0, size[0] - sp(16))),
        )


def _label_lines(label: dict | None) -> list[str]:
    """The tooltip lines a label adds. **Pure.**"""
    if not label:
        return ["Not labelled. G for good, B for bad."]
    lines = [LABEL_WORDS.get(str(label.get("verdict")), "Labelled")]
    if label.get("reasons"):
        lines.append("Tags: " + ", ".join(label["reasons"]))
    if label.get("note"):
        lines.append("Note: " + str(label["note"]))
    return lines


# --- labelling a reference ------------------------------------------------------------


def labelable(job: dict[str, Any] | None) -> bool:
    """Whether a Good/Bad label applies: a reference-stage row that has a picture.

    Not gated on ``done`` -- ``record_verdict``'s image stages are not, because a
    composition-gate refusal is errored with its reference on disk and is the
    most informative negative there is -- but the cell and the single view only
    show the picture of a finished one, so this asks for the same.
    """
    return bool(job) and job.get("stage") == "reference" and _has_picture(job)


def prime_labels(ctx: Any, ids: list[str]) -> None:
    """Read the latest human label of every id not yet known, in one query.

    A failing read answers *unlabelled* and is remembered, so a broken store is
    not asked again sixty times a second (``inspector.is_graded``'s rule).
    """
    memo = ctx.state.create.label_memo
    missing = [i for i in ids if i not in memo]
    if not missing:
        return
    try:
        found = ctx.svc.store.verdicts_for(missing, source="human", stage="reference")
    except Exception:  # noqa: BLE001 - see the docstring; never fail a frame
        found = {}
    for job_id in missing:
        memo[job_id] = found.get((job_id, "human"))


def label_of(ctx: Any, job_id: str) -> dict[str, Any] | None:
    """The latest human label on a reference, or ``None``. Memoised."""
    prime_labels(ctx, [job_id])
    return ctx.state.create.label_memo.get(job_id)


def set_label(
    ctx: Any,
    job_id: str,
    verdict: str | None = None,
    *,
    tags: list[str] | None = None,
    note: str | None = None,
) -> bool:
    """Label a reference Good or Bad. -> whether the label now stands.

    ``None`` for ``tags``/``note``/``verdict`` keeps what is on file, because the
    newest row wins and a row that said only "Bad" would silently drop the tags
    and the note filed before it. Pressing the key for the label an image
    already has, with nothing else changed, writes nothing: the log is
    append-only and a held key must not fill it.
    """
    from ....panes import inspector

    current = label_of(ctx, job_id) or {}
    verdict = verdict or current.get("verdict")
    if verdict not in LABEL_WORDS:
        ctx.toast("Choose Good or Bad first.", "error")
        return False
    old_tags = list(current.get("reasons") or [])
    new_tags = old_tags if tags is None else list(tags)
    new_note = current.get("note") if note is None else note
    if (
        current
        and verdict == current.get("verdict")
        and new_tags == old_tags
        and (new_note or None) == (current.get("note") or None)
    ):
        return True
    if not inspector.record_label(ctx, job_id, verdict, tuple(new_tags), new_note or None):
        return False
    ctx.state.create.label_memo[job_id] = {
        "verdict": verdict,
        "reasons": new_tags,
        "note": (new_note or "").strip() or None,
        "grade": None,
        "source": "human",
    }
    return True


def handle_label_keys(ctx: Any, job: dict[str, Any]) -> bool:
    """G files Good, B files Bad, N opens the tags-and-note popup. -> whether one fired.

    Bare letters only: a chord with Ctrl, Alt or Shift belongs to the app's own
    shortcuts and must not label a picture.
    """
    io = imgui.get_io()
    if io.key_ctrl or io.key_alt or io.key_shift or io.key_super or not labelable(job):
        return False
    for verdict, key in LABEL_KEYS:
        if imgui.is_key_pressed(key, False):
            set_label(ctx, str(job["id"]), verdict)
            return True
    if imgui.is_key_pressed(LABEL_NOTE_KEY, False):
        open_label_popup(ctx, job)
        return True
    return False


def open_label_popup(ctx: Any, job: dict[str, Any]) -> None:
    """Ask for the tags-and-note popup on this image, seeded from its label.

    Only a request: ``open_popup`` has to be called at the id scope
    ``begin_popup`` is, which is :func:`label_popup`'s.
    """
    state = ctx.state.create
    current = label_of(ctx, str(job["id"])) or {}
    state.label_popup_job = str(job["id"])
    state.label_popup_ask = True
    state.label_verdict = str(current.get("verdict") or "")
    state.label_tags = list(current.get("reasons") or [])
    state.label_note = str(current.get("note") or "")


def label_popup(ctx: Any) -> None:
    """The popup: Good or Bad, the tag vocabulary, a note, and Save. Nothing is
    written until Save, so closing it loses nothing but what was being typed.

    The tag rows are the mesh grade's own (``widgets.tag_toggles``): the two
    vocabularies are one namespace shared by every stage, and the service accepts
    them on image labels as on meshes.
    """
    state = ctx.state.create
    job_id = state.label_popup_job
    if job_id is None:
        return
    if state.label_popup_ask:
        imgui.open_popup(_LABEL_POPUP)
        state.label_popup_ask = False
    if not imgui.begin_popup(_LABEL_POPUP):
        state.label_popup_job = None
        return
    widgets.secondary("Label this image")
    for verdict, _key in LABEL_KEYS:
        if verdict != "accept":
            imgui.same_line()
        if controls.button(
            f"{LABEL_WORDS[verdict]}##label-{verdict}",
            role=(
                controls.ButtonRole.PRIMARY
                if state.label_verdict == verdict
                else controls.ButtonRole.SECONDARY
            ),
        ):
            state.label_verdict = verdict
    tag = widgets.tag_toggles(f"create-label-{job_id}", state.label_tags, True)
    if tag is not None:
        if tag in state.label_tags:
            state.label_tags.remove(tag)
        elif tag in svc_verdicts.TAGS:
            state.label_tags.append(tag)
    widgets.field_label("Note")
    state.label_note = widgets.multiline(
        "##create-label-note", state.label_note, sp(90), svc_verdicts.NOTE_MAX
    )
    if widgets.disabled_button(
        "Save##create-label-save",
        state.label_verdict in LABEL_WORDS,
        reason="Choose Good or Bad first.",
    ):
        set_label(
            ctx, job_id, state.label_verdict, tags=list(state.label_tags), note=state.label_note
        )
        state.label_popup_job = None
        imgui.close_current_popup()
    imgui.same_line()
    if controls.button("Cancel##create-label-cancel"):
        state.label_popup_job = None
        imgui.close_current_popup()
    imgui.end_popup()


def label_row(ctx: Any, job: dict[str, Any]) -> None:
    """The single-image view's Good / Bad / tags-and-note row, at the canvas's
    top left. The current label shows as the selected button."""
    job_id = str(job["id"])
    label = label_of(ctx, job_id) or {}
    for verdict, _key in LABEL_KEYS:
        if verdict != "accept":
            imgui.same_line()
        word = LABEL_WORDS[verdict]
        if controls.button(
            f"{word}##canvas-label-{verdict}",
            role=(
                controls.ButtonRole.PRIMARY
                if label.get("verdict") == verdict
                else controls.ButtonRole.SECONDARY
            ),
            tooltip=f"Mark this image as {word.lower()} ({word[0]}).",
        ):
            set_label(ctx, job_id, verdict)
    imgui.same_line()
    if controls.button(
        "Tags and note...##canvas-label-more",
        role=controls.ButtonRole.GHOST,
        tooltip="Tag this image and write a note (N).",
    ):
        open_label_popup(ctx, job)
    for line in _label_lines(label or None)[1:]:
        widgets.muted(line)


def choose(ctx: Any, job: dict[str, Any]) -> None:
    """Select one image of the sheet, the way the attempts strip does."""
    from . import workspace

    workspace.inspect_result(ctx, job)


def dismiss(ctx: Any, batch: list[dict[str, Any]]) -> None:
    """Close this batch's sheet; the selected image fills the canvas alone."""
    ctx.state.create.grid_dismissed.add(str(batch[0]["id"]))


def reopen(ctx: Any, batch: list[dict[str, Any]]) -> None:
    ctx.state.create.grid_dismissed.discard(str(batch[0]["id"]))


# --- the keyboard --------------------------------------------------------------


def keys_enabled(ctx: Any) -> bool:
    """Whether the sheet may read the keyboard this frame.

    The canvas must own the focus (a click on a cell gives it; a click in the
    prompt or the inspector takes it away), no text field may be taking
    keystrokes, and no palette or modal may be up. This is the gate the other
    Create keys use, stated for the canvas: ``brief._count`` reads its arrows
    only while its own control is the focused one, and ``shell.events`` lets a
    plain key through only when ``_passes_text_field`` says typing is not
    happening.
    """
    return bool(
        imgui.is_window_focused()
        and not imgui.get_io().want_text_input
        and not ctx.state.palette_open
        and not dialogs.modal_open(ctx)
    )


_ARROWS = (
    ("left", imgui.Key.left_arrow),
    ("right", imgui.Key.right_arrow),
    ("up", imgui.Key.up_arrow),
    ("down", imgui.Key.down_arrow),
)


def handle_keys(ctx: Any, batch: list[dict[str, Any]], index: int) -> None:
    """Arrows move the selection, G and B label it, N opens its note, Enter makes
    3D, Esc returns to one image."""
    from . import workspace

    if handle_label_keys(ctx, batch[index]):
        return
    for name, key in _ARROWS:
        if imgui.is_key_pressed(key):
            target = families.grid_step(index, name, len(batch))
            if target != index:
                choose(ctx, batch[target])
            return
    if imgui.is_key_pressed(imgui.Key.enter) or imgui.is_key_pressed(imgui.Key.keypad_enter):
        if _has_picture(batch[index]):
            workspace.make_3d(ctx, batch[index])
        return
    if imgui.is_key_pressed(imgui.Key.escape):
        dismiss(ctx, batch)


# --- the single view's overlay ---------------------------------------------------

#: Where the overlay's controls sit inside the picture, in design pixels, every
#: one a step of the spacing scale. The top inset clears the 38 dp toolbar row
#: (an 8 dp pane inset over a 30 dp row) on the stages that have no tray, where
#: the window these draw into *is* the viewport pane; it was a literal 40 and the
#: bottom one a literal 18, neither on the scale.
_OVERLAY_SIDE_DP = tokens.SP_4
_OVERLAY_TOP_DP = tokens.SP_6 + tokens.SP_4
_OVERLAY_BOTTOM_DP = tokens.SP_4

#: Make 3D's size: a regular control tall (it was 38, a one-off), wide enough for
#: its label with room to spare.
MAKE_3D_SIZE_DP = (168.0, tokens.CONTROL_HEIGHT_REGULAR)


def make_3d_job(ctx: Any) -> dict[str, Any] | None:
    """The selected reference when it is a finished image Make 3D applies to."""
    job = ctx.job()
    if job is None or job.get("stage") != "reference" or not _has_picture(job):
        return None
    kind = create_assets.asset_type_from_params(job.get("params") or {}, stage="reference")
    return job if kind in MAKE_3D_KINDS else None


def draw_overlay(ctx: Any) -> None:
    """Over the picture: **Make 3D** at the bottom, and **Compare all** at the top
    when the picture is one of a finished batch.

    Drawn into the window the picture was drawn into, by cursor position, after
    it -- imgui items drawn later are on top and take the click. The primary
    button is the same ``workspace.make_3d`` the tray card and the inspector
    call, cutout check included.
    """
    if ctx.state.create.stage != "reference":
        return
    pos, size = imgui.get_window_pos(), imgui.get_window_size()
    batch = grid_batch(ctx, ignore_dismissed=True)
    if batch is not None:
        label = f"Compare all {len(batch)}##canvas-grid"
        reach = widgets.button_width(f"Compare all {len(batch)}")
        imgui.set_cursor_screen_pos(
            (pos.x + size.x - reach - sp(_OVERLAY_SIDE_DP), pos.y + sp(_OVERLAY_TOP_DP))
        )
        if controls.button(
            label, role=controls.ButtonRole.GHOST, tooltip="Show every image of this press at once."
        ):
            reopen(ctx, batch)
    selected = ctx.job()
    if labelable(selected):
        imgui.set_cursor_screen_pos((pos.x + sp(_OVERLAY_SIDE_DP), pos.y + sp(_OVERLAY_TOP_DP)))
        label_row(ctx, selected)
        if keys_enabled(ctx):
            handle_label_keys(ctx, selected)
        label_popup(ctx)
    job = make_3d_job(ctx)
    if job is None:
        return
    button = (sp(MAKE_3D_SIZE_DP[0]), sp(MAKE_3D_SIZE_DP[1]))
    imgui.set_cursor_screen_pos(
        (
            pos.x + (size.x - button[0]) / 2,
            pos.y + size.y - button[1] - sp(_OVERLAY_BOTTOM_DP),
        )
    )
    if widgets.primary_button(
        "Make 3D##canvas-make-3d",
        button,
        tooltip="Check the cutout, then reconstruct this image as a 3D model.",
    ):
        from . import workspace

        workspace.make_3d(ctx, job)


# --- the empty canvas --------------------------------------------------------------


def image_weights_missing(ctx: Any) -> bool:
    """Whether this host has **no image model at all**. Quiet until the answers land.

    Judged from ``ctx.model_rows`` for ``model_gate``'s reason: an empty list is
    "nothing to say", never "everything is missing". Any present ``base:`` row
    means a press can run.
    """
    rows = [
        r
        for r in (getattr(ctx, "model_rows", None) or [])
        if str(r.get("row_key", "")).startswith("base:")
    ]
    return bool(rows) and not any(r.get("present") for r in rows)


def empty(ctx: Any, stage: str) -> None:
    """The canvas with nothing on it, for whichever stage is standing here."""
    if stage == "reference":
        _empty_reference(ctx)
    elif stage == "mesh":
        _empty_mesh(ctx)
    else:
        # Empty-state hero: the only thing on the screen, not one group among several, so it
        # keeps the pane heading rather than becoming a card.
        widgets.pane_header("Nothing here yet")
        widgets.muted_wrapped("Choose a finished creation on the Mesh stage to continue.")


def _empty_reference(ctx: Any) -> None:
    form = ctx.state.form_2d
    if image_weights_missing(ctx) and not create_recipe.is_character(form):
        setup_card(ctx)
        return
    if str(form.get("prompt") or "").strip():
        # Not pristine: the brief has words in it, so the type cards (which
        # duplicate the column's "What are you making?") have nothing left to
        # offer. Say what pressing Generate will do and stop.
        from . import workspace

        # Empty-state hero: the only thing on the screen, not one group among several, so it
        # keeps the pane heading rather than becoming a card.
        widgets.pane_header("Ready when you are")
        widgets.muted_wrapped(workspace.plan_for(form).stages)
        return
    from . import workspace

    workspace.type_cards(ctx)


def setup_card(ctx: Any) -> None:
    """The first-run card: no image model on this machine, nothing loaded.

    Two downloads and one way around them. The Character is real and needs no
    GPU, but it renders with Blender, so a host without the Rigging pack is told
    so up front rather than on the press.
    """
    # Empty-state hero: the only thing on the screen, not one group among several, so it
    # keeps the pane heading rather than becoming a card.
    widgets.pane_header("Set up Realmspinner")
    widgets.muted_wrapped(
        "Everything runs on this computer, so it needs its models first. "
        "Get the image model to start drawing."
    )
    rows = _image_rows(ctx)
    if rows and controls.button(f"Get the image model{_size_suffix(ctx, rows)}##setup-image"):
        model_gate.request_install(ctx, rows)
    engine = tuple(r for r in create_mesh.ENGINE_ROWS if _row_missing(ctx, r))
    if engine and controls.button(f"Get the 3D engine{_size_suffix(ctx, engine)}##setup-engine"):
        model_gate.request_install(ctx, engine)
    imgui.spacing()
    widgets.muted("or make a Character (no GPU needed)")
    if controls.button("Make a Character##setup-character"):
        form = ctx.state.form_2d
        form["asset_type"] = form["generation_type"] = "character"
        create_assets.sync_legacy_fields(form)
        ctx.state.clear_field_errors()
    if not getattr(ctx, "rigging_available", False):
        widgets.muted_wrapped("A Character is rigged with Blender, so it needs the Rigging pack.")


def _image_rows(ctx: Any) -> tuple[str, ...]:
    offer = create_recipe.automatic_install(ctx, ctx.state.form_2d)
    if offer is not None:
        return offer[0]
    return tuple(
        str(r["row_key"])
        for r in (getattr(ctx, "model_rows", None) or [])
        if str(r.get("row_key", "")).startswith("base:") and not r.get("present")
    )[:1]


def _row_missing(ctx: Any, key: str) -> bool:
    return any(
        str(r.get("row_key")) == key and not r.get("present")
        for r in (getattr(ctx, "model_rows", None) or [])
    )


def _size_suffix(ctx: Any, keys: tuple[str, ...]) -> str:
    by_key = {str(r.get("row_key")): r for r in (getattr(ctx, "model_rows", None) or [])}
    gib = sum(float(by_key.get(k, {}).get("size_gib") or 0.0) for k in keys)
    return f" ({gib:.1f} GiB)" if gib > 0.0 else ""


def _empty_mesh(ctx: Any) -> None:
    """The Mesh stage with no reference chosen: a drop zone and this creation's
    finished references to pick from.

    The picks are what the Source chip takes from a dragged library card --
    ``state.source_job`` -- and the file button is the chip's own upload, so
    there is one way to name a reference however it arrives.
    """
    from ...library.ui.panes import library

    state = ctx.state
    # Empty-state hero: the only thing on the screen, not one group among several, so it
    # keeps the pane heading rather than becoming a card.
    widgets.pane_header("Choose a reference to build from")
    widgets.muted_wrapped("Drop an image here, or pick a reference.")
    if controls.button(
        "Choose an image...##mesh-empty-upload",
        enabled=not ctx.busy("upload"),
        reason="A file picker is already open.",
    ):
        ctx.submit("upload", dialogs.open_file, "Choose a reference image", dialogs.IMAGE_FILTER)
    key = state.create.workspace
    refs = [
        j
        for j in (families.results(session.index(ctx), key, "reference") if key else [])
        if library.can_drag(j)
    ]
    if refs:
        imgui.spacing()
        widgets.secondary("This creation's references")
        side = sp(96)
        for i, job in enumerate(refs[:12]):
            if i:
                imgui.same_line()
            imgui.push_id(str(job["id"]))
            imgui.begin_group()
            thumbs.job_thumb(ctx, job, side)
            if imgui.is_item_clicked():
                state.source_job = str(job["id"])
            if imgui.is_item_hovered():
                imgui.set_tooltip(str(job.get("name") or job.get("prompt") or job["id"]))
            imgui.end_group()
            imgui.pop_id()
    # A drop target for a library card dragged onto the canvas: the zone the
    # sentence above names. The chip's own handling, minus its highlight.
    avail = imgui.get_content_region_avail()
    imgui.invisible_button("##mesh-drop-zone", (max(1.0, avail.x), max(1.0, avail.y)))
    if imgui.begin_drag_drop_target():
        payload = imgui.accept_drag_drop_payload_py_id(library.DRAG_JOB)
        if payload is not None and state.dragging_job:
            dropped = ctx.cache.get(state.dragging_job)
            if dropped is not None and library.can_drag(dropped):
                state.source_job = state.dragging_job
            state.dragging_job = None
        imgui.end_drag_drop_target()
