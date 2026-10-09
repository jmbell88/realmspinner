"""The Inspector: one panel whose body follows what you have selected.

This is the Clay right column's one slot (``clay-inspector``), and it replaces
Properties' four tabs (``props.py``) with a switch on the situation, so the answer to
"what can I do to this?" is always the thing in front of you. The body below scrolls
in a child of its own; **docked at the pane's foot** is "All shapes (N)", the
outliner (``outliner.draw_embedded``): one header row while closed, a pinned height
while open, so its rows never move when the cards above them change.

* **Nothing selected** -- "What do you want to make?": four big shape tiles, every
  other shape one disclosure away, and the Model card (counts, New / Open / Save,
  the ways out, a History row). The first screen a new model has, and the one the
  tour's Save and Export steps point at -- Esc lets go of a shape to get back to it.
* **One shape** -- its name and colour, the **Size** card (the one place a size is
  typed -- ``inspector_size.py``), its own settings when it has any, the **Colour**
  card (eight named starter colours, ``starter_colours.py``), Duplicate and
  Delete, then *closed* disclosures: Position & turn (scale behind its own
  Advanced), Group & parent, and More actions (every object-level op, generated
  from the registry -- ``inspector_actions.py``).
* **Several shapes** -- Colour, Duplicate and Delete, and the multi-object actions;
  no Size card ("Select one shape to change its size").
* **An element mode** (Points, Edges, Faces) -- "Editing faces" and a Done button that
  goes back to Shape mode, then the element state (``inspector_elements.py``): the
  Select row, what is picked and its median, **Shape it** (the mode's mesh ops from
  the registry, the rest behind closed folds) and, while an op's numbers are still
  live, the **Last action** card. Colours & textures and Texture mapping stay one
  click below, because painting a face and laying out an island are what an element
  mode is for.

**Pages push, and Esc or Back pops.** Colours & textures, Unwrap and Model &
history are whole pages (``ClayState.inspector_page``) that replace the body under
a "Back" row, because each needs the full column (the UV canvas wants all of the
height; the texture rows are a form of their own) and none belongs in a card. They
reuse Properties' bodies -- ``props._material``, ``uv.draw_embedded``,
``props._document`` -- so a page cannot drift from the tab it came from. A page is
reachable from every mode (Colours and Unwrap from an element mode's links as well),
because the Material and UV tabs it replaces were.

**Cheap by construction.** The panel runs every frame at any document size, so the
body reads ``doc.selection`` and nothing heavier; anything that walks the document
(the Model card's counts, More actions' gates) is memoised on ``doc.rev`` and the
disclosures that hold it are not run while closed
(``tests/modes/clay/test_inspector.py`` measures it at 1, 100 and 4096 shapes).

One ``section_blocks`` scope per window, never nested: every section is a tinted
block, and the page bodies open sections of their own inside the same scope. With the
dock the body is a child window, which has a draw list of its own, so the scope is
opened inside that child; the dock has no headings of its own and opens none.
"""

from __future__ import annotations

import weakref
from typing import Any

from imgui_bundle import imgui

from ......kernels.mesh import primitives as bp
from ..... import anchors, controls, fonts, icons, theme, tokens, tool_palette, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import doc_io, starter_colours
from ... import mode as clay_mode
from ... import ops as clay_ops
from .. import inspector_actions, inspector_elements, inspector_size
from . import outliner as clay_outliner
from . import props, rail, swatches
from . import tools as clay_tools
from . import uv as clay_uv

#: What this pane refuses to shrink past, in design pixels: the title, the first
#: card and a few rows under it, plus the dock's header. The right column's only
#: slot, so this is the column's floor (a fill with no floor takes whatever the share
#: above left).
INSPECTOR_FLOOR = 280.0

#: The "All shapes" dock: its persisted open state, the share of the pane its open
#: region takes, the fewest rows that region may show, and the height the cards
#: above keep (design px) before the list is allowed to take more. See
#: :func:`dock_region_height`.
#:
#: Landing 9 retuned ``DOCK_SHARE`` (0.40 -> 0.28) and ``BODY_FLOOR`` (180 -> 420), measured
#: at 1600x950 with the element state up: that state's Select row, element line and
#: Shape it card are about 580 px, and at 40% the open dock left the body 470 px, so
#: Shape it was cut across its second row and the folds below it were a scroll away. At
#: 28% the body is about 600 (Delete a few pixels short of whole) and the list shows
#: five rows; the floor is what stops a shorter window from taking more of the cards.
DOCK_KEY = "clay/insp/all-shapes"
DOCK_SHARE = 0.28
DOCK_MIN_ROWS = 5
BODY_FLOOR = 420.0

#: The pages a click can push, in the order they are documented. ``""`` is the
#: main view and is not a page.
PAGES = ("colours", "unwrap", "model")

PAGE_TITLES = {
    "colours": "Colours & textures",
    "unwrap": "Unwrap",
    "model": "Model & history",
}

#: Generator parameters that only say how big a shape is. A shape whose every
#: parameter is one of these (box, plane) has nothing the Size card does not
#: already say, and a second W/H/D under it -- one that rebuilds the mesh where the
#: card scales it -- is exactly the two sizing controls the card exists to replace.
SIZE_PARAMS = frozenset({"size"})

#: A shape tile's height in design pixels: a big glyph over the shape's word.
TILE_H = 84.0

_SAVING = "This document is being written; the controls come back when it lands."
_NO_DOC = "Open or start a document to build in."


def dock_region_height(
    avail: float, header: float, bars: float, row: float, body_floor: float, *, compact: bool
) -> float:
    """The pinned height of the open "All shapes" region, header excluded. -> px.

    A function of the pane's own height and of constants, never of the selection or
    of what the cards above happen to hold: that is what keeps the rows where they
    were between a press and its release. About :data:`DOCK_SHARE` of the pane, no
    fewer than :data:`DOCK_MIN_ROWS` rows under the Find and Solo bars, and never so
    much that the cards above fall under *body_floor* -- except that two rows is the
    least a list is worth, and when the pane is too short for both the cards scroll
    (they are in a child that does) rather than the list vanishing.

    *compact* is the empty document: one line saying so, with nothing to pin.
    """
    if compact:
        return row
    least = bars + 2.0 * row
    wanted = max(avail * DOCK_SHARE, bars + DOCK_MIN_ROWS * row)
    ceiling = avail - header - body_floor
    return max(least, min(wanted, ceiling))


def draw(ctx: Any) -> None:
    """The body, then (with a document, outside a page) the docked "All shapes" list.

    The column has one slot, so this pane is the whole of it: the body scrolls in
    a child of its own and the dock sits under that child, at the foot, not
    scrolling. With the dock open the child is the pane less a pinned height (see
    :func:`dock_region_height`); closed, the child is the pane less one header row.
    A page (Colours, Unwrap, Model) takes the whole pane -- the UV canvas fits
    itself to all of the height -- and with no document there is nothing to list.
    """
    state = clay_mode.ensure(ctx)
    tab = state.active
    if tab is None or state.inspector_page in PAGES:
        with widgets.section_blocks():
            _body(ctx)
        return
    _docked(ctx, state, tab)


def _docked(ctx: Any, state: Any, tab: Any) -> None:
    doc = tab.doc
    style = imgui.get_style()
    gap = style.item_spacing.y
    # The header's line is taller than the header: the (?) that right-aligns onto it
    # draws its glyph with zero frame padding, so imgui adds the difference between
    # the two baselines (one ``frame_padding.y``) to the line. Measured on screen as
    # 29 px of header in a 36 px line, and a pane that forgot it grew a scrollbar.
    header = imgui.get_frame_height() + style.frame_padding.y
    row = sp(clay_outliner.ROW_HEIGHT) + gap
    bars = 2.0 * (imgui.get_frame_height() + gap)
    empty = not doc.objects
    # Open when the dock has nothing to compete with: no shape picked, or no shape
    # at all. Only the first draw reads this -- after it the stored choice wins.
    default_open = empty or not doc.selection
    assumed = state.dock_open
    if assumed is None or widgets.open_requested(DOCK_KEY) or widgets.FORCE_SECTIONS_OPEN:
        # A one-shot open request (and the smoke test's force-open) is taken by the
        # header on this very draw, so the layout has to expect it now.
        assumed = widgets.persisted_open(DOCK_KEY, default_open)
    region = 0.0
    if assumed:
        region = dock_region_height(
            imgui.get_content_region_avail().y,
            header + gap,
            bars,
            row,
            sp(BODY_FLOOR),
            compact=empty,
        )
    # ``- 1``: a pixel of slack so rounding can never push the pane's content past
    # its own bottom and give it a scrollbar of its own.
    body = imgui.get_content_region_avail().y - header - gap - (region + gap if region else 0.0) - 1
    _body_child(ctx, max(body, 0.0))

    opened = widgets.header(
        f"All shapes ({len(doc.objects)})##clay-all-shapes",
        default_open=default_open,
        persist_key=DOCK_KEY,
    )
    anchors.mark("clay/outliner")
    manual_render.help_button(ctx, "clay-outliner")
    state.dock_open = bool(opened)
    # Skipped on the one frame the header disagrees with the height this frame was
    # laid out for (the click that toggled it): the next frame has it right.
    if opened and assumed:
        clay_outliner.draw_embedded(ctx, region)


def _body_child(ctx: Any, height: float) -> None:
    """The pane's body in a scrolling child that bleeds to the pane's edges.

    Bled, so the sections' tinted blocks (which overhang their content by
    ``SP_2``) land exactly where they did when the body was the pane itself, and
    the scrollbar sits on the pane's edge rather than inside its padding. The
    child's own padding is the host's, so every heading and field keeps its x.
    """
    cursor = imgui.get_cursor_pos()
    top = max(cursor.y - sp(tokens.SP_2), 0.0)
    imgui.set_cursor_pos((0.0, top))
    imgui.push_style_var(imgui.StyleVar_.window_padding.value, (cursor.x, cursor.y - top))
    imgui.push_style_color(imgui.Col_.child_bg.value, imgui.ImVec4(0.0, 0.0, 0.0, 0.0))
    inside = imgui.begin_child(
        "##insp-body",
        (imgui.get_window_size().x, height + (cursor.y - top)),
        imgui.ChildFlags_.always_use_window_padding.value,
        0,
    )
    imgui.pop_style_color()
    imgui.pop_style_var()
    if inside:
        with widgets.section_blocks():
            _body(ctx)
    imgui.end_child()


def _title(ctx: Any, text: str, *, help_button: bool = True) -> None:
    """The pane heading. The tour's ``clay/inspector`` ring is on it."""
    widgets.section(text)
    anchors.mark("clay/inspector")
    if help_button:
        manual_render.help_button(ctx, "clay-inspector")


def _body(ctx: Any) -> None:
    state = clay_mode.ensure(ctx)
    tab = state.active
    if tab is None:
        # No document, so no page: a page left open would be there when the
        # next document opened, over a body that has not been drawn yet.
        state.inspector_page = ""
        _title(ctx, "Inspector")
        return
    doc = tab.doc
    page = state.inspector_page if state.inspector_page in PAGES else ""
    state.inspector_page = page
    _settle_uv(tab, doc, page)
    if page or doc.element_mode == "object":
        # Anything but the element state: the Last action card is not on screen, so the
        # next time it appears it is new again and scrolls itself into view.
        state.last_action_open = False
    if page:
        _page(ctx, state, tab, doc, page)
        return
    if doc.element_mode != "object":
        _elements(ctx, state, tab, doc)
        return
    obj = swatches.single_selected(doc)
    if obj is not None:
        _one(ctx, state, tab, doc, obj)
    elif len(doc.selection) > 1:
        _many(ctx, state, tab, doc)
    else:
        _nothing(ctx, state, tab, doc)


def _settle_uv(tab: Any, doc: Any, page: str) -> None:
    """Commit a live UV rotate/scale when the Unwrap page is not what is showing.

    ``props._body``'s own rule, moved with the tab it guarded: a gesture armed on
    the canvas whose canvas is no longer drawn would stay armed with its undo
    gesture open (switching the stack's eviction off for the document's life) and
    swallow the 3-D viewport's R and S.
    """
    uv_view = getattr(tab, "uv_view", None)
    if page != "unwrap" and uv_view is not None and uv_view.drag_mode in ("rotate", "scale"):
        clay_uv.commit_live_transform(doc, uv_view)


# --- nothing selected -----------------------------------------------------------


def _nothing(ctx: Any, state: Any, tab: Any, doc: Any) -> None:
    _title(ctx, "Inspector")
    with fonts.heading(imgui):
        imgui.text_wrapped("What do you want to make?")
    _tiles(ctx, state, tab, doc)
    if widgets.header("More shapes", default_open=False, persist_key="clay/insp/more-shapes"):
        clay_tools.draw_add_menu(ctx, state, tab, heading=False)
    _model_card(ctx, tab, doc)


def _tiles(ctx: Any, state: Any, tab: Any, doc: Any) -> None:
    """The quick shapes as big labelled cards, two across.

    ImGui has no thumbnails to show, so each card is the shape's own glyph drawn
    large over its word -- the glyph and the word both come from the tables the
    rail and the flyout read (``tool_palette.PRIMITIVE_ICONS``,
    ``rail.SHAPE_WORDS``), so a card cannot drift from its rail button. A click is
    the human Add door (``tools.place_primitive``): on the ground, beside what is
    there, in the colour last picked, framed.
    """
    width = widgets.grid_width(2)
    for index, name in enumerate(clay_tools.QUICK_SHAPES):
        if index % 2:
            imgui.same_line()
        word = rail.SHAPE_WORDS.get(name, clay_tools.display_name(name))
        glyph = tool_palette.PRIMITIVE_ICONS.get(name, icons.BOX)
        clicked = controls.button(
            f"##insp-tile/{name}",
            (width, sp(TILE_H)),
            enabled=not tab.saving,
            reason=_SAVING,
            tooltip=f"Add a {clay_tools.display_name(name)}",
        )
        _tile_face(glyph, word, enabled=not tab.saving)
        if clicked and clay_tools.place_primitive(ctx, doc, name) is not None:
            state.generator = name


def _tile_face(glyph: str, word: str, *, enabled: bool) -> None:
    """Draw a tile's glyph and word over the button just submitted.

    The draw list, not the button's label, because a button's label is one run of
    one font and the two are different sizes (the rail's cell does the same).
    """
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    centre = (low.x + high.x) * 0.5
    alpha = 1.0 if enabled else 0.5
    draw_list = imgui.get_window_draw_list()
    with fonts.push(imgui, fonts.MEDIUM, 30.0):
        pos = fonts.centred_glyph_pos(imgui, glyph, centre, low.y + (high.y - low.y) * 0.4)
        draw_list.add_text(pos, imgui.get_color_u32(theme.rgba(theme.TEXT, alpha)), glyph)
    with fonts.push(imgui, fonts.MEDIUM, tokens.TEXT_BODY):
        size = imgui.calc_text_size(word)
        draw_list.add_text(
            (centre - size.x * 0.5, high.y - size.y - sp(8.0)),
            imgui.get_color_u32(theme.rgba(theme.MUTED, alpha)),
            word,
        )


#: ``doc -> (rev, text)`` for the Model card's counts line, which walks every
#: shape. Weak, so a closed document takes its entry with it.
_COUNTS: weakref.WeakKeyDictionary[Any, tuple[Any, str]] = weakref.WeakKeyDictionary()


def counts_line(doc: Any) -> str:
    """:func:`doc_io.model_counts_line`, memoised on ``doc.rev``.

    The line sums every shape's vertices, faces and triangles; at 4,096 shapes
    that is a few milliseconds a frame for a number that changes when the model
    does. ``rev`` covers every edit, selection and undo, so nothing can change the
    answer without moving it first.
    """
    rev = getattr(doc, "rev", None)
    cached = _COUNTS.get(doc)
    if rev is not None and cached is not None and cached[0] == rev:
        return cached[1]
    text = doc_io.model_counts_line(doc)
    if rev is not None:
        _COUNTS[doc] = (rev, text)
    return text


def _model_card(ctx: Any, tab: Any, doc: Any) -> None:
    """The Model card: the counts, the four file verbs, the ways out, History.

    The file verbs are ``widgets.document_header`` -- the one widget every
    document mode draws -- over the same ``clay_mode`` doors as the File menu and
    the chords; the ways out are Properties' own ``_outputs`` (the library, GLB,
    OBJ and a screenshot, with the refusal reasons and the "Last exported" lines),
    so the card and the Model & history page cannot disagree about either.
    """
    state = clay_mode.ensure(ctx)
    widgets.section("Model")
    widgets.muted_wrapped(counts_line(doc))
    widgets.document_header(
        tab,
        new=lambda: clay_mode.new_document(ctx),
        open_=lambda: clay_mode.ask_open(ctx),
        save=lambda: clay_mode.save(ctx, tab),
        save_as=lambda: clay_mode.save_as(ctx, tab),
    )
    anchors.mark("clay/props/save")
    props._outputs(ctx, tab)
    widgets.end_section()
    imgui.dummy((0, sp(tokens.SP_2)))
    if controls.button(
        f"{icons.LIST} History {icons.CHEVRON_RIGHT}##insp-history",
        (imgui.get_content_region_avail().x, 0),
        tooltip="Every step you have taken. Click one to go back to it. The import "
        "settings are there too.",
    ):
        state.inspector_page = "model"


# --- one shape ------------------------------------------------------------------


def _one(ctx: Any, state: Any, tab: Any, doc: Any, obj: Any) -> None:
    _title(ctx, "Selected shape")
    imgui.begin_disabled(bool(tab.saving))
    _header_row(doc, obj)
    inspector_size.draw(ctx, state, doc, obj)
    _settings(ctx, state, doc, obj)
    _colour_card(ctx, state, tab, doc, [obj.uid])
    imgui.end_disabled()
    widgets.end_section()
    imgui.dummy((0, sp(tokens.SP_2)))
    _action_row(ctx, tab, doc)
    imgui.dummy((0, sp(tokens.SP_2)))
    imgui.begin_disabled(bool(tab.saving))
    if widgets.header("Position & turn", default_open=False, persist_key="clay/insp/position"):
        props._transform(
            doc, obj, ctx=ctx, state=state, show_unit=False, show_scale=False, show_size=False
        )
        if widgets.header("Advanced", default_open=False, persist_key="clay/insp/advanced"):
            widgets.field_label("scale")
            props.scale_fields(doc, obj, ctx=ctx)
            widgets.muted_wrapped(
                "Scale compares the shape with the mesh as first made; Size above is "
                "the number to use."
            )
    if widgets.header("Group & parent", default_open=False, persist_key="clay/insp/group"):
        props._relations(ctx, doc, obj)
    imgui.end_disabled()
    _more_actions(ctx, state, tab, doc, persist_key="clay/insp/more", default_open=False)


def _header_row(doc: Any, obj: Any) -> None:
    """The colour dot and the name, which is a field (rename in place).

    ``commit=True`` for the reason ``props._identity`` states: a rename is an undo
    step, and a step per letter is not a rename.
    """
    slot = min(max(int(obj.material), 0), len(doc.materials) - 1) if doc.materials else -1
    colour = swatches._swatch_colour(doc.materials[slot]) if slot >= 0 else (0.5, 0.5, 0.5, 1.0)
    radius = sp(7.0)
    height = imgui.get_frame_height()
    origin = imgui.get_cursor_screen_pos()
    imgui.get_window_draw_list().add_circle_filled(
        (origin.x + radius, origin.y + height * 0.5),
        radius,
        imgui.get_color_u32(imgui.ImVec4(*colour)),
    )
    imgui.dummy((radius * 2.0, height))
    imgui.same_line()
    imgui.set_next_item_width(-1.0)
    name = widgets.input_text("##insp-name", obj.name, max_length=120, commit=True)
    if name and name != obj.name:
        doc.set_props(obj.uid, name=name)


def _settings(ctx: Any, state: Any, doc: Any, obj: Any) -> None:
    """The shape's own numbers -- only when it has any beyond its size -- or the
    frozen note.

    ``props._generator`` is the whole of it: one widget per generator parameter
    (so a new generator needs nothing here), the "frozen" note for an edited or
    imported mesh, and ``apply_generator_params`` for the one-step rebuild.
    """
    if obj.generator is None:
        props._generator(doc, obj, ctx=ctx, state=state)
        return
    entry = bp.GENERATORS.get(obj.generator)
    if entry is None:
        props._generator(doc, obj, ctx=ctx, state=state)
        return
    if not set(entry[0]) - SIZE_PARAMS:
        return
    widgets.section("Shape settings")
    props._generator(doc, obj, ctx=ctx, state=state)


def _colour_card(ctx: Any, state: Any, tab: Any, doc: Any, uids: list[int]) -> None:
    """The eight starter colours, and a way to the full editor.

    A click paints every shape in *uids* in one undo step, finding or adding the
    document's slot for it (``starter_colours.paint``). The swatch the shapes
    already wear is outlined; a colour that is none of the eight (an edited slot, a
    textured one) outlines a ninth, "custom" swatch instead, so the card never
    shows no answer to "what colour is it now?".
    """
    widgets.section("Colour")
    current = _shared_slot(doc, uids)
    match = starter_colours.matching(doc, current)
    side = sp(swatches.SWATCH)
    for index, colour in enumerate(starter_colours.STARTER_COLOURS):
        if index:
            widgets.same_line_or_wrap(side)
        hit = swatches._swatch(
            f"##inspsw/{colour.name}",
            colour.factor,
            side,
            selected=colour is match,
            textured=False,
            tooltip=colour.name,
        )
        if hit and not tab.saving:
            starter_colours.paint(ctx, doc, uids, colour)
    if match is None and current is not None:
        material = doc.materials[current]
        widgets.same_line_or_wrap(side)
        swatches._swatch(
            "##inspsw/custom",
            swatches._swatch_colour(material),
            side,
            selected=True,
            textured=material.base_color is not None,
            tooltip=f"Custom colour: {material.name or f'slot {current}'}. Edit colour "
            "below changes it.",
        )
    label = f"{icons.PALETTE} Edit colour {icons.CHEVRON_RIGHT}##insp-edit-colour"
    if controls.button(
        label,
        (imgui.get_content_region_avail().x, 0),
        tooltip="Rename it, change the colour, add a picture. Colours & textures opens "
        "as a page; Esc comes back.",
    ):
        state.inspector_page = "colours"


def _shared_slot(doc: Any, uids: list[int]) -> int | None:
    """The one palette slot every shape in *uids* wears as its default, else ``None``.

    One pass over the document with the uids as a set. ``ClayDoc.by_uid`` is a
    linear scan, so asking it once per selected shape made this quadratic: a frame
    with 4,096 shapes selected cost a quarter of a second.
    """
    if not doc.materials:
        return None
    wanted = set(uids)
    top = len(doc.materials) - 1
    slots = {min(max(int(obj.material), 0), top) for obj in doc.objects if obj.uid in wanted}
    return next(iter(slots)) if len(slots) == 1 else None


def _action_row(ctx: Any, tab: Any, doc: Any) -> None:
    """Duplicate and Delete, side by side, each greyed with its own sentence.

    The registry's two ops through ``clay_mode.fire_op`` -- the menus' and the
    keys' door -- so a click here and Ctrl+J / Del cannot differ.
    """
    width = widgets.grid_width(2)
    duplicate, delete = clay_ops.get("duplicate"), clay_ops.get("delete")
    why_dup = _SAVING if tab.saving else clay_ops.reason_for(duplicate, doc)
    why_del = _SAVING if tab.saving else clay_ops.reason_for(delete, doc)
    if widgets.disabled_button(
        "Duplicate##insp-duplicate",
        not why_dup,
        (width, 0),
        reason=why_dup,
        tooltip=inspector_actions.tooltip_for(duplicate),
    ):
        clay_mode.fire_op(ctx, doc, duplicate)
    imgui.same_line()
    if widgets.destructive_button(
        "Delete##insp-delete",
        (width, 0),
        enabled=not why_del,
        reason=why_del,
        tooltip=inspector_actions.tooltip_for(delete),
    ):
        clay_mode.fire_op(ctx, doc, delete)


def _more_actions(
    ctx: Any, state: Any, tab: Any, doc: Any, *, persist_key: str, default_open: bool
) -> None:
    """The More actions disclosure. Its body -- and every ``op.enabled`` under it --
    runs only while it is open."""
    if widgets.header("More actions", default_open=default_open, persist_key=persist_key):
        inspector_actions.draw(ctx, state, tab, doc)


# --- several shapes -------------------------------------------------------------


def _many(ctx: Any, state: Any, tab: Any, doc: Any) -> None:
    count = len(doc.selection)
    _title(ctx, f"{count} shapes selected")
    widgets.muted_wrapped("Select one shape to change its size.")
    imgui.begin_disabled(bool(tab.saving))
    _colour_card(ctx, state, tab, doc, sorted(doc.selection))
    imgui.end_disabled()
    widgets.end_section()
    imgui.dummy((0, sp(tokens.SP_2)))
    _action_row(ctx, tab, doc)
    imgui.dummy((0, sp(tokens.SP_2)))
    # Open by default: with several shapes picked, the things worth doing are
    # Group, Merge Objects, Parent to Last and Mirror, and they are all in here.
    _more_actions(ctx, state, tab, doc, persist_key="clay/insp/more-many", default_open=True)


# --- element modes -------------------------------------------------------------


_NOUN = {"vertex": "points", "edge": "edges", "face": "faces"}


def _elements(ctx: Any, state: Any, tab: Any, doc: Any) -> None:
    """Points, Edges and Faces: the title, a Done button, then the element state."""
    _title(ctx, f"Editing {_NOUN.get(doc.element_mode, 'parts')}")
    if widgets.primary_button(
        f"{icons.CHECK} Done##insp-done",
        enabled=not tab.saving,
        reason=widgets.DOCUMENT_SAVING_WHY,
        tooltip="Finish editing and go back to the whole shape (Esc).",
    ):
        doc.set_element_mode("object")
    imgui.dummy((0, sp(tokens.SP_2)))
    inspector_elements.draw(ctx, state, tab, doc)
    # The Material and UV tabs this slot used to carry are pages now; an element
    # mode is where both are wanted (paint a face, lay out an island), so they are
    # one click from here as well.
    for page, label in (("colours", "Colours & textures"), ("unwrap", "Texture mapping")):
        if controls.button(
            f"{label} {icons.CHEVRON_RIGHT}##insp-page/{page}",
            (imgui.get_content_region_avail().x, 0),
        ):
            state.inspector_page = page


# --- pages ----------------------------------------------------------------------


def _page_object(doc: Any) -> Any:
    """The shape a page edits: the one selected, else the first of several."""
    obj = swatches.single_selected(doc)
    if obj is not None or not doc.selection:
        return obj
    for uid in sorted(doc.selection):
        try:
            return doc.by_uid(uid)
        except KeyError:
            continue
    return None


def _page(ctx: Any, state: Any, tab: Any, doc: Any, page: str) -> None:
    """A pushed page: its title, a Back row, then its body."""
    # The UV body draws its own (?) -- on the line above it, which is the Back
    # row's -- so this page must not draw a second one.
    _title(ctx, PAGE_TITLES[page], help_button=page != "unwrap")
    if controls.button(
        f"{icons.ARROW_LEFT} Back##insp-back", tooltip="Back to the inspector (Esc)"
    ):
        state.inspector_page = ""
    if page == "unwrap":
        # The canvas wants the pane's whole height, so no child of its own and no
        # disabled wrapper: the UV tools gate themselves on ``tab.saving``. No
        # spacer after the Back row either -- the body's (?) lands on that line.
        clay_uv.draw_embedded(ctx)
        return
    imgui.dummy((0, sp(tokens.SP_2)))
    imgui.begin_disabled(bool(tab.saving))
    if page == "model":
        props._document(ctx, tab, doc)
    else:
        obj = _page_object(doc)
        if obj is None:
            widgets.muted_wrapped("Select a shape to edit its colour.")
        else:
            props._material(ctx, tab, doc, obj)
    imgui.end_disabled()
