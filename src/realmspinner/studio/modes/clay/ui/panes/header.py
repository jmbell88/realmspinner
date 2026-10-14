"""The row across the top of Clay's viewport: what you edit, how it draws, the way out.

Clay had no viewport toolbar at all. Which element mode you were in, which
transform tool you held, whether snapping was on and what the viewport was
showing were four separate blocks down a 300 px sidebar -- and the sidebar is
on the far side of the window from the model. Every one of them is a setting
changed *between* clicks in the viewport, which is the same argument that moved
Inker's tool options onto a context bar and Plotter's tools onto one.

**One row, left to right, in four groups by what the controls are for:** *history*
(Undo, Redo and a chevron that opens the list); *mode* (**Shape / Points / Edges /
Faces**, what a click edits); *display* (the shading switch **Flat / Lit / Wire**
and X-ray, how the surface is drawn); *aids* (**View**, the overlays and the
camera in one popover, and Snap). A thin rule separates the groups
(``toolbar.Item.group``), and it is counted in the row's width arithmetic like any
other control. At the far end the help marker and **Export**, a split button whose main
click is the library export and whose chevron lists the other ways out. The row
degrades rather than clips (``toolbar``): words become glyphs, the shading
switch and the three view buttons fold into the overflow menu before the mode
switch does, and the mode switch, Undo and Redo are pinned -- a mode picker in a
menu is a mode picker nobody can see the state of, which is the one thing a mode
picker is for.

**The labels are the plain words; the state is not.** "Shape" is the document's
``object`` mode and "Flat / Lit / Wire" are ``state.shading``'s ``solid /
material / wireframe``: stored names are what a saved layout, the agent surface
and the key handlers already speak, and a rename there would be a migration for
the sake of a word. Keys 1-4 are unchanged; every tooltip carries its key.

**The tool is not here.** Select / Move / Turn / Resize are the top of the tool
rail beside the viewport (``rail.py``); a second pill for the same four buttons
was two controls for one state.

**Modes are lettered when narrow, not drawn.** ``icons.py`` is a transcription of
lucide-static 0.525.0 and its docstring forbids guessing a codepoint; the
vendored subset has nothing for a vertex, an edge or a face. ``inker_context``
made the same call for its four symmetry mirrors and said so.

What is *not* here yet, and deliberately: the pivot and orientation menus, and
X-mirror. Each is a control whose behaviour is a later wave, and a switch that
changes a setting nothing reads is worse than no switch -- it is the UI stating
what is not true, which is the defect this codebase has paid for before. They
arrive with the code that reads them.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ..... import controls, fonts, icons, tokens, toolbar, verbs, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import doc_io
from ... import mode as clay_mode
from . import tools as clay_tools

#: This bar's imgui id and the prefix every one of its controls is keyed from.
BAR = "clay-header"

#: How tall the header is, in design pixels, for the viewport's reservation.
#: A frame height plus the separator under it plus the row's own spacing --
#: measured rather than guessed would be better and is not possible: the
#: viewport has to subtract this *before* it draws, and the bar is drawn after.
HEADER_H = 34.0

#: The row's four groups, by what the controls are for. The names are only compared
#: for equality by ``toolbar`` (a rule is drawn where they change); they are not shown.
HISTORY_GROUP = "history"
MODE_GROUP = "mode"
DISPLAY_GROUP = "display"
AIDS_GROUP = "aids"

SNAP_POPUP = "clay-snap"
VIEW_POPUP = "clay-view"
HISTORY_POPUP = "clay-header-history"
EXPORT_POPUP = "clay-export"

#: How the surface is drawn, in the order Blender lists them: the flat albedo,
#: the lit render, the edges alone. The first column is ``state.shading``'s own
#: value and does not change with the word on the button. Glyph-free -- lucide
#: has nothing for a shading mode and ``icons.py`` forbids inventing a codepoint
#: -- so the compact tier is the first letter, which is unambiguous across three.
SHADING: tuple[tuple[str, str, str, str], ...] = (
    ("solid", "Flat", "F", "Flat colours, no lighting"),
    ("material", "Lit", "L", "Lit surface"),
    ("wireframe", "Wire", "W", "Edges only"),
)

#: The element modes, compact. ``clay_tools.MODE_BUTTONS`` carries the full
#: labels and the keys; this is only what a narrow window shows instead.
MODE_SHORT = {"object": "S", "vertex": "P", "edge": "E", "face": "F", "pose": "O"}

_SAVING = "This document is being written; the controls come back when it lands."
#: The 2026-10-10 audit's clay-58: ``clay_mode.undo``/``redo``/``step_history`` return
#: early under a live G/R/S or UV rotate/scale, so the header's three history items
#: and the Model page's block stayed lit and then did nothing. The same sentence
#: ``ops._GESTURE_SENTENCE`` gives the op rows.
_GESTURE = "Finish or cancel the drag first."
_UNDO_WHY = "Nothing to undo yet."
_REDO_WHY = "Nothing to redo: this is the newest step."
_HISTORY_WHY = "No steps yet."
_NO_VIEW = "The viewport is still starting up."

#: The two sentences the command palette's Clay rows grey with, so a row there and
#: its control here never give two accounts of one state (``studio/palette.py``).
SAVING_WHY = _SAVING
GESTURE_WHY = _GESTURE
NO_VIEW_WHY = _NO_VIEW

#: The Export button's words. The long form is the library door's own name
#: (``verbs.EXPORT_TO_LIBRARY``); the button is the short form because its
#: tooltip and the menu under its chevron spell the destination out.
EXPORT_LABEL = "Export"


def draw(ctx: Any, view: Any = None) -> None:
    """The bar. Called by the viewport, between the tab bar and the image."""

    state = clay_mode.ensure(ctx)
    tab = state.active
    if tab is None:
        return
    # The bar is the first row above the render: the Select | Add | Object | Mesh | UV
    # menu strip that used to sit over it is gone (every op in it is in the
    # Inspector, the right-click menu and Ctrl+K), so the viewport, which sizes
    # itself from what is left below this call, is that row taller.
    items, fields, trailing = _row(ctx, state, tab, view)
    hit = toolbar.toolbar(BAR, items, fields=fields, trailing=trailing)
    if hit == "undo":
        # ``clay_mode.undo``/``redo`` rather than ``tab.doc.undo()``: the button
        # and the chord carry the same side effects (see that module's history
        # block).
        clay_mode.undo(ctx, tab)
    elif hit == "redo":
        clay_mode.redo(ctx, tab)
    elif hit == "history":
        imgui.open_popup(HISTORY_POPUP)
    _history_popup(ctx, tab)
    widgets.divider()


def history_why(*, busy: bool, gesture: bool, idle: str) -> str:
    """Why an Undo/Redo/History control is greyed, or *idle* when the stack alone
    is the reason (``""`` for a control that is lit).

    Pure and beside the draw call, the shape ``clay_ops.reason_for`` takes. A save
    outranks a gesture: it is the longer of the two and the one a click cannot
    wait out. Shared with the Model page's history block (``props._history``), so
    the header and the page never give two accounts of one state.
    """
    if busy:
        return _SAVING
    if gesture:
        return _GESTURE
    return idle


def _row(ctx: Any, state: Any, tab: Any, view: Any) -> tuple[list[Any], list[Any], Any]:
    """The three things ``toolbar.toolbar`` is handed, built once so a test can
    plan the very row :func:`draw` draws."""
    return (
        _items(state, tab, clay_mode._gesture_live(ctx, tab)),
        _fields(ctx, state, tab, view),
        _trailing(ctx, tab),
    )


def row_plan(ctx: Any, state: Any, tab: Any, avail: float, view: Any = None) -> Any:
    """The tier every entry of the row would be drawn at in ``avail`` physical px.

    Needs an imgui context for its measurements and draws nothing. For
    ``tests/studio/test_studio_smoke.py``, which asks the one question a header
    can silently get wrong: does it fit on one line at the window the app opens
    at and at its floor.
    """
    items, fields, trailing = _row(ctx, state, tab, view)
    return toolbar.plan_row(items, fields, trailing, avail)


def _items(state: Any, tab: Any, gesture: bool = False) -> list[Any]:
    """Undo, Redo and the history chevron.

    **Undo and Redo read the document's own stack** (``tab.doc.history``): the
    reason a greyed one gives is the history's, not a guess, and both grey while
    a save is in flight for ``widgets.history_block``'s reason -- the chord is
    already refused then (``docmodes.blocked_while_writing``), and a click that
    could mutate the stack under a running encode must not be the way round it.
    Clicking runs ``clay_mode.undo`` and ``redo`` (see :func:`draw`). Under a live
    gesture (*gesture*, ``clay_mode._gesture_live``) all three grey too, with
    :data:`_GESTURE` for the reason -- the mode's doors refuse then, and a lit
    control that does nothing is the 2026-10-10 audit's clay-58.

    All three are glyphs at every width (``icon_only``): the history is three
    squares, never three words, and the mode switch beside them is what the row
    spends its words on. Undo and Redo are pinned; the chevron is not, because
    the same list is on the Inspector's Model & history page.
    """

    del state
    history = tab.doc.history
    busy = bool(getattr(tab, "busy", getattr(tab, "saving", False)))
    return [
        toolbar.Item(
            "undo",
            "Undo",
            icons.UNDO,
            tooltip="Undo the last step (Ctrl+Z)",
            enabled=bool(history.can_undo) and not busy and not gesture,
            reason=history_why(busy=busy, gesture=gesture, idle=_UNDO_WHY),
            icon_only=True,
            pinned=True,
            group=HISTORY_GROUP,
        ),
        toolbar.Item(
            "redo",
            "Redo",
            icons.REDO,
            tooltip="Redo the step you undid (Ctrl+Y)",
            enabled=bool(history.can_redo) and not busy and not gesture,
            reason=history_why(busy=busy, gesture=gesture, idle=_REDO_WHY),
            icon_only=True,
            pinned=True,
            group=HISTORY_GROUP,
        ),
        toolbar.Item(
            "history",
            "History",
            icons.CHEVRON_DOWN,
            tooltip="Every step, with the newest marked. Click one to go there.",
            enabled=bool(history.total) and not busy and not gesture,
            reason=history_why(busy=busy, gesture=gesture, idle=_HISTORY_WHY),
            icon_only=True,
            priority=1,
            group=HISTORY_GROUP,
        ),
    ]


def _history_popup(ctx: Any, tab: Any) -> None:
    """The undo stack as a popover: a click on a step is ``clay_mode.step_history``.

    The mode's door, not ``doc.step_history``: it refuses under a live gesture
    (a jump mid-drag would undo the thing being dragged), which the pane must not
    be able to bypass. Keyed apart from the Model & history page (``clay``), which
    draws the same list in the same window set -- two rows with one id would be
    one row that answers both.
    """
    widgets.history_popup(
        HISTORY_POPUP,
        tab,
        lambda index: clay_mode.step_history(ctx, tab, index),
        key=BAR,
        opened="(the document as opened)",
    )


# --- measuring ---------------------------------------------------------------------


def _design(px: float) -> float:
    """Physical px -> design px, for ``toolbar.Field``, which scales its own widths."""
    return px / tokens.SCALE


def _segments(labels: list[str]) -> float:
    """What ``controls.segmented_choice`` draws *labels* at, in physical px."""
    style = imgui.get_style()
    pad = style.frame_padding.x * 2.0
    gap = style.item_spacing.x
    with fonts.label(imgui):
        text = sum(imgui.calc_text_size(label).x + pad for label in labels)
    return text + gap * max(len(labels) - 1, 0)


def _word(label: str) -> float:
    """A text button's width, in physical px."""
    with fonts.label(imgui):
        return widgets.button_width(label)


# --- the mode and the shading ------------------------------------------------------


def _mode_field(tab: Any) -> Any:
    """Shape / Points / Edges / Faces.

    The mode lives on the **document** rather than on ``ClayState``: it is the
    interpretation key for a selection, and an app-level mode would reinterpret
    every other tab's selection on a tab switch. So this reads ``tab.doc`` and
    the state is not involved.

    A field, not an item, and **pinned**: ``toolbar`` collapses items into the
    overflow menu before it collapses anything else, and a mode picker in a menu
    is a mode picker nobody can see the state of.
    """

    def draw_it(compact: bool) -> None:
        doc = tab.doc
        options = [
            (mode, MODE_SHORT[mode] if compact else label)
            for mode, label, _key in clay_tools.MODE_BUTTONS
        ]
        tips = {
            mode: f"{label} mode  ({key})" for mode, label, key in clay_tools.MODE_BUTTONS
        }
        changed, picked = controls.segmented_choice(
            "clay-mode",
            options,
            "pose" if doc.pose_mode else doc.element_mode,
            tooltips=tips,
            compact=True,
            enabled=not tab.saving,
            reason=_SAVING,
        )
        if changed:
            # Pose is a layer over Shape mode, not an element mode; any other choice leaves it.
            if picked == "pose":
                doc.set_pose_mode(True)
            else:
                doc.set_element_mode(picked)

    full = _segments([label for _mode, label, _key in clay_tools.MODE_BUTTONS])
    short = _segments([MODE_SHORT[mode] for mode, _label, _key in clay_tools.MODE_BUTTONS])
    return toolbar.Field(
        "mode",
        "Mode",
        draw_it,
        width=_design(full),
        compact=_design(short),
        pinned=True,
        group=MODE_GROUP,
    )


def _shading_field(state: Any) -> Any:
    """Flat / Lit / Wire: how the surface is drawn.

    First to give up its words after the three view buttons, and the first
    *segment* to fold into the menu after them: it changes what the render is
    rather than what you are editing, and the same three are in the overflow menu
    as a row each when the bar has no room.
    """

    def draw_it(compact: bool) -> None:
        changed, picked = controls.segmented_choice(
            "clay-shading",
            [(key, short if compact else label) for key, label, short, _tip in SHADING],
            state.shading,
            tooltips={key: tip for key, _label, _short, tip in SHADING},
            compact=True,
        )
        if changed:
            state.shading = picked

    full = _segments([label for _key, label, _short, _tip in SHADING])
    short = _segments([short for _key, _label, short, _tip in SHADING])
    return toolbar.Field(
        "shading",
        "Shading",
        draw_it,
        width=_design(full),
        compact=_design(short),
        priority=1,
        group=DISPLAY_GROUP,
    )


# --- X-ray, View, Snap -------------------------------------------------------------


def _glyph_word(glyph: str, word: str) -> tuple[str, float, float]:
    """``("<glyph> word", full px, glyph-only px)`` for a button with both tiers."""
    text = f"{glyph} {word}"
    with fonts.label(imgui):
        solo = widgets.button_width(glyph)
    return text, _word(text), solo


def _tiered_button(
    key: str,
    glyph: str,
    word: str,
    compact: bool,
    *,
    tooltip: str,
    selected: bool = False,
) -> bool:
    """The three view buttons' one drawing: a word and its glyph, or the glyph
    alone with the word in the tooltip (a glyph with no name is the accessible-name
    failure ``icon_button`` forbids)."""
    label = glyph if compact else f"{glyph} {word}"
    return controls.button(
        f"{label}##{BAR}/{key}",
        role=controls.ButtonRole.GHOST,
        control_size=controls.ControlSize.COMPACT,
        selected=selected,
        tooltip=f"{word} -- {tooltip}" if compact else tooltip,
    )


def _xray_field(state: Any) -> Any:
    def draw_it(compact: bool) -> None:
        if _tiered_button(
            "xray",
            icons.BLEND,
            "X-ray",
            compact,
            tooltip="See through the surface, so an element behind it can be picked (Alt+Z)",
            selected=bool(state.xray),
        ):
            state.xray = not state.xray

    _text, full, solo = _glyph_word(icons.BLEND, "X-ray")
    return toolbar.Field(
        "xray",
        "X-ray",
        draw_it,
        width=_design(full),
        compact=_design(solo),
        priority=2,
        group=DISPLAY_GROUP,
    )


def _view_field(ctx: Any, state: Any, view: Any) -> Any:
    def draw_it(compact: bool) -> None:
        if _tiered_button(
            "view",
            icons.EYE,
            "View",
            compact,
            tooltip="What the viewport draws over the model, and where the camera looks from",
            selected=any(
                overlay_value(state, key) != OVERLAY_DEFAULTS[key]
                for key, _label, _tip in OVERLAY_ROWS
            ),
        ):
            imgui.open_popup(VIEW_POPUP)
        _view_popup(ctx, state, view)

    _text, full, solo = _glyph_word(icons.EYE, "View")
    return toolbar.Field(
        "view",
        "View",
        draw_it,
        width=_design(full),
        compact=_design(solo),
        priority=2,
        group=AIDS_GROUP,
    )


def _snap_field(state: Any) -> Any:
    def draw_it(compact: bool) -> None:
        if _tiered_button(
            "snap",
            icons.MAGNET,
            "Snap",
            compact,
            tooltip="Snap a drag to the grid or to an angle",
            selected=bool(state.snap),
        ):
            imgui.open_popup(SNAP_POPUP)
        _snap_popup(state)

    _text, full, solo = _glyph_word(icons.MAGNET, "Snap")
    return toolbar.Field(
        "snap",
        "Snap",
        draw_it,
        width=_design(full),
        compact=_design(solo),
        priority=2,
        group=AIDS_GROUP,
    )


def _fields(ctx: Any, state: Any, tab: Any, view: Any) -> list[Any]:
    """The row's controls after Undo / Redo, in the order they are read."""
    return [
        _mode_field(tab),
        _shading_field(state),
        _xray_field(state),
        _view_field(ctx, state, view),
        _snap_field(state),
    ]


def _snap_popup(state: Any) -> None:
    """The grid and the angle switch. Lifted whole from the pane.

    Live while a save runs, on purpose: none of it touches the document.
    """

    with controls.menu_popup(SNAP_POPUP) as opened:
        if not opened:
            return
        changed, value = widgets.toggle(f"{icons.MAGNET} Snap", state.snap)
        if changed:
            state.snap = value
        imgui.begin_disabled(not state.snap)
        # "%.4f", because the grid steps by 1/16 m and imgui's default "%.3f"
        # drew that as 0.063 -- a field that disagrees with its own step button.
        # Label above, unit kept in it (2026-09-08 consistency pass); ids
        # kept stable, "grid (m)##snapt" -> "##grid (m)##snapt".
        widgets.field_label("grid (m)")
        _, state.snap_translate = controls.input_float(
            "##grid (m)##snapt", state.snap_translate, 0.0625, 0.0, "%.4f"
        )
        widgets.field_label("angle (deg)")
        _, state.snap_rotate = controls.input_float(
            "##angle (deg)##snapr", state.snap_rotate, 5.0, 0.0
        )
        imgui.end_disabled()
        # Clamped rather than validated: zero is the off switch every snap
        # function already treats as the identity, and a negative grid is
        # meaningless.
        state.snap_translate = max(0.0, float(state.snap_translate))
        state.snap_rotate = max(0.0, float(state.snap_rotate))


#: What the viewport draws over the model, as a table. One row per flag so the
#: popover is a loop rather than a column of hand-written toggles, and so a
#: sixth overlay is one line here.
#:
#: ``grid`` is a field on ``ClayState`` and the rest live in ``state.overlays``,
#: which is not an inconsistency to tidy: the grid is wired straight to
#: ``ClayView.show_grid`` and has been since the viewport existed, and giving it
#: a second home in the dict would be two places that can disagree about one
#: switch. The table says which is which and the popover does not care.
OVERLAY_ROWS: tuple[tuple[str, str, str], ...] = (
    ("grid", "Grid", "The ground plane, in 1 m cells -- its size is set below"),
    ("wire", "Wireframe over surface", "Every edge, over the shaded surface"),
    (
        "stats",
        "Statistics",
        "Objects, vertices, edges, faces and triangles -- and how many are selected.",
    ),
    (
        "god_light",
        "God light",
        "Light straight down from 100 m overhead onto a ground plane under "
        "the grid",
    ),
)

#: What each row is when nothing has been changed, so the View button can
#: say "something in here is not the default" without a second hand-written
#: list. It had one -- ``(("grid", True), ("wire", False))`` -- which left
#: ``stats`` out, so turning Statistics on and nothing else drew the button
#: unselected: the one overlay with no other sign it is on was the one the
#: button would not report.
OVERLAY_DEFAULTS: dict[str, bool] = {
    "grid": True, "wire": False, "stats": False, "god_light": False,
}


def overlay_value(state: Any, key: str) -> bool:
    if key == "grid":
        return bool(state.grid)
    if key == "god_light":
        return bool(state.god_light)
    return bool(state.overlays.get(key, False))


def set_overlay(state: Any, key: str, value: bool) -> None:
    if key == "grid":
        state.grid = bool(value)
    elif key == "god_light":
        state.god_light = bool(value)
    else:
        state.overlays[key] = bool(value)


def _grid_size_field(ctx: Any, state: Any) -> None:
    """The grid's size, in metres -- Task A. Indented under the Grid row it
    belongs to.

    ``commit=True``: this is a settings write on every keystroke otherwise,
    which is undo-stack spam's sibling for a preference rather than a
    document edit -- typing "37" would persist "3" first. Clamped and rounded
    to a whole metre on commit, the doctrine every clamped-not-validated field
    in this popup already follows (``_snap_popup``'s grid step): a hand-typed 0
    or a negative size is meaningless for a ground plane, and a fractional one
    breaks the "1 m cells" promise the field makes.
    """

    imgui.indent()
    widgets.field_label("size (m)")
    imgui.set_next_item_width(sp(90))
    settled, value = controls.input_float(
        f"##{BAR}/gridsize", float(state.grid_size), 1.0, 10.0, "%.0f", commit=True
    )
    if settled:
        state.grid_size = max(1.0, min(1000.0, round(float(value))))
        clay_mode.persist(ctx)
    imgui.unindent()


#: The six views and the chord that reaches each, in the order the popover lists
#: them. The three that had buttons were labelled ``F``/``R``/``T`` in a sidebar
#: and their backs were reachable only by holding Shift, which nothing said.
AXIS_ROWS: tuple[tuple[str, str, str], ...] = (
    ("front", "Front", "Ctrl+1"),
    ("back", "Back", "Ctrl+Shift+1"),
    ("left", "Left", "Ctrl+Shift+3"),
    ("right", "Right", "Ctrl+3"),
    ("top", "Top", "Ctrl+7"),
    ("bottom", "Bottom", "Ctrl+Shift+7"),
)


def look_along(view: Any, name: str) -> None:
    """One of :data:`AXIS_ROWS`' views: the camera looks along that axis.

    The View popover's row and the command palette's are this one call, so a
    seventh view is a line in the table and not a line in two places.
    """
    view.camera.look_along(name)


def toggle_orthographic(view: Any) -> None:
    """Perspective <-> orthographic; shared by the popover and the palette."""
    view.camera.orthographic = not view.camera.orthographic


def request_frame(state: Any) -> None:
    """Frame the selection. A flag rather than a call, the house pattern: framing
    needs the viewport, which is a thing ``main`` owns."""
    state.frame_pending = True


def _view_popup(ctx: Any, state: Any, view: Any) -> None:
    """What the viewport draws and where the camera looks from, in one popover.

    The old **Overlays** and **View** buttons were two popovers a reader could not
    tell apart ("Overlays" held the grid, "View" held the front/back views), and a
    third of the row's width. One list, in two captioned groups, then the two rows
    that are neither (orthographic and framing).

    ``view`` is the live viewport and is ``None`` before it exists -- the first
    frame, and exactly when a pane must not raise. The rows that need it are
    disabled rather than absent, which is the house pattern and also the honest
    answer: they will work in a moment.
    """

    with controls.menu_popup(VIEW_POPUP) as opened:
        if not opened:
            return
        widgets.field_label("Overlays")
        for key, label, tip in OVERLAY_ROWS:
            hit = controls.menu_item(
                f"{label}##{BAR}/overlay/{key}",
                "",
                overlay_value(state, key),
                tooltip=tip,
            )
            if bool(hit[0] if isinstance(hit, tuple) else hit):
                set_overlay(state, key, not overlay_value(state, key))
                clay_mode.persist(ctx)
            if key == "grid":
                _grid_size_field(ctx, state)
        controls.menu_separator()
        widgets.field_label("Views")
        camera = None if view is None else view.camera
        for name, label, chord in AXIS_ROWS:
            hit = controls.menu_item(
                f"{label}##{BAR}/axis/{name}",
                chord,
                False,
                camera is not None,
                reason=_NO_VIEW,
            )
            if bool(hit[0] if isinstance(hit, tuple) else hit):
                look_along(view, name)
        controls.menu_separator()
        ortho = bool(camera is not None and camera.orthographic)
        hit = controls.menu_item(
            f"Orthographic##{BAR}/ortho",
            "Ctrl+5",
            ortho,
            camera is not None,
            reason=_NO_VIEW,
            tooltip="No perspective, so parallel edges stay parallel -- which "
            "is what makes two parts line up by eye.",
        )
        if bool(hit[0] if isinstance(hit, tuple) else hit):
            toggle_orthographic(view)
        hit = controls.menu_item(
            f"Frame the selection##{BAR}/frame", "F", False, True
        )
        if bool(hit[0] if isinstance(hit, tuple) else hit):
            request_frame(state)


# --- the way out -------------------------------------------------------------------


def _export_menu(ctx: Any, tab: Any) -> None:
    """The ways out under the Export chevron: the library, a mesh file, a picture.

    The same four doors the Model & history page draws (``props._outputs``), each
    greyed with the one sentence ``doc_io.outputs_why`` gives for every output
    button -- a user reading two explanations of one state would look for two
    different problems.
    """

    why = doc_io.outputs_why(tab.doc, bool(tab.saving), clay_mode._gesture_live(ctx, tab))
    ready = not why
    with controls.menu_popup(EXPORT_POPUP) as opened:
        if not opened:
            return
        rows = (
            (
                f"{icons.DOWNLOAD} {verbs.EXPORT_TO_LIBRARY}",
                "library",
                doc_io.EXPORT_LIBRARY_TOOLTIP,
                lambda: clay_mode.export_asset(ctx, tab),
            ),
            (
                f"{icons.DOWNLOAD} Export GLB...",
                "glb",
                doc_io.EXPORT_GLB_TOOLTIP,
                lambda: clay_mode.export_mesh_file(ctx, tab, "glb"),
            ),
            (
                "Export OBJ...",
                "obj",
                doc_io.EXPORT_OBJ_TOOLTIP,
                lambda: clay_mode.export_mesh_file(ctx, tab, "obj"),
            ),
            *(
                (
                    (
                        "Export Posed GLB...",
                        "posed",
                        doc_io.EXPORT_POSED_TOOLTIP,
                        lambda: clay_mode.export_mesh_file(ctx, tab, "posed"),
                    ),
                )
                if tab.doc.animation() is not None and any(o.bone for o in tab.doc.objects)
                else ()
            ),
            (
                f"{icons.CAMERA} Save screenshot...",
                "screenshot",
                doc_io.SCREENSHOT_TOOLTIP,
                lambda: clay_mode.save_screenshot(ctx, tab),
            ),
        )
        for label, key, tip, run in rows:
            hit = controls.menu_item(
                f"{label}##{BAR}/export/{key}",
                "",
                False,
                ready,
                reason=why,
                tooltip=tip,
            )
            if bool(hit[0] if isinstance(hit, tuple) else hit):
                run()


def _trailing(ctx: Any, tab: Any) -> Any:
    """The help marker and the Export split button, at the end of the row.

    A trailing block rather than two more entries, for ``inker_context``'s
    reason: it never folds into the overflow menu (a way out that moves house when
    the window is resized is not a way out), and what it gives up first is its
    *word*. It draws itself against the right edge when the row has room to spare,
    which is the one place in the bar where the position is the point.

    **The main click is the library export, the same door as Properties' button,
    and it refuses in place.** Greyed with the reason ``doc_io.outputs_why``
    gives -- an empty or fully hidden scene has nothing to send -- rather than
    toasting after the click. The chevron lists every way out.
    """

    style = imgui.get_style()
    gap = style.item_spacing.x
    text, word_w, glyph_w = _glyph_word(icons.DOWNLOAD, EXPORT_LABEL)
    square = imgui.get_frame_height()
    help_w = square
    # The split's two halves touch: a button and its chevron are one control.
    seam = sp(2)

    def block(main_w: float) -> float:
        return help_w + gap + main_w + seam + square

    full, small = block(word_w), block(glyph_w)

    def draw_it(tight: bool) -> None:
        width = small if tight else full
        # Pushed against the right edge when there is room: what the row has
        # left over goes *between* the settings and the way out, not after it.
        spare = imgui.get_content_region_avail().x - width
        if spare > 0.0:
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + spare)
        manual_render.help_button_inline(ctx, "clay-header")
        imgui.same_line()
        why = doc_io.outputs_why(tab.doc, bool(tab.saving), clay_mode._gesture_live(ctx, tab))
        if widgets.primary_button(
            (f"{icons.DOWNLOAD}" if tight else text) + f"##{BAR}/export",
            enabled=not why,
            reason=why,
            tooltip=f"{verbs.EXPORT_TO_LIBRARY}: " + doc_io.EXPORT_LIBRARY_TOOLTIP,
        ):
            clay_mode.export_asset(ctx, tab)
        imgui.same_line(0.0, seam)
        if widgets.icon_button(
            f"{icons.CHEVRON_DOWN}##{BAR}/export-more",
            "Other ways out: a GLB or OBJ file, or a screenshot",
        ):
            imgui.open_popup(EXPORT_POPUP)
        _export_menu(ctx, tab)

    # Physical px, as ``toolbar`` subtracts them from the row's physical width.
    return toolbar.Trailing(full, small, draw_it)
