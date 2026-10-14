"""Clay's workspace: the viewport, its tabs, its marquee and its drag HUD.

A **mixin on** :class:`~.main.App`, which is this repository's idiom for a body
of drawing that belongs to the shell -- ``ClayView`` is assembled the same way
from the ``_view_*`` ops mixins and ``Document`` from its own. So ``self`` here is the App and
every method's body is unchanged.

Lifted out of ``studio/main`` on 2026-09-04 (T7 of the 2026-09-02 review), after
the behavioural findings that touch it were closed, so the move is code motion
over tested behaviour.

The viewport itself is ``ClayView``'s; what is here is the *pane* around it --
the layout skeleton, the invisible button that takes the mouse, the tab bar,
the empty state and the two overlays drawn on top.

``_ensure_build_view``, ``_frame_clay_selection`` and
``_capture_clay_thumbnail`` live here too: they are read only from this class's
own methods, so they came here rather than to a shell module.
``_capture_thumbnail_from`` -- the general
framebuffer-read-and-queue -- did not follow them: ``shell/tasks.py``'s
``_on_task_done`` calls it directly for Mason's own export as well as for
Clay's, so it stayed shared shell plumbing; see that module's own docstring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class ClayViewport:
    """Clay's pane drawing, mixed into :class:`~.main.App`.

    The shell names it reaches are imported *inside* the methods that use them:
    ``main`` imports :class:`~.shell.app.App` (which assembles this mixin) to
    build the class, so a module-scope import back would be a cycle. Same
    shape as ``review_panes``.
    """

    def _clay_workspace(self) -> None:
        """The same column / centre / column skeleton every other mode uses, with
        nothing in the left column:

                          the header      [ clay-inspector, with
            [ rail ]      the viewport      All shapes docked at its foot ]
                          the hint

        The right column is ``skeletons.clay``, which is where the argument for
        that arrangement is written down. The left column is declared and empty:
        its Add pane became the tool rail, which is drawn *inside* the centre
        (``_clay_viewport``) so the model gets the width the sidebar took.
        """
        from imgui_bundle import imgui

        from .... import layout as layout_mod
        from .... import skeletons, widgets
        from ....shell.frame import _column_boundary
        from .. import mode as clay_mode

        ctx = self.app_ctx
        lay = self.layout
        left_w = layout_mod.sidebar_width("left")
        right_w = layout_mod.sidebar_width("right")
        columns = skeletons.for_mode(ctx, "clay")

        left = skeletons.ordered(ctx, self.layouts, "clay", columns["left"])
        # Skipped outright when empty, boundary included: ``_column_boundary`` is
        # a ``same_line``, and one with no item before it in the content child
        # indents the centre by an item spacing for nothing.
        if left:
            layout_mod.column(ctx, lay, left, width=left_w, handle_length=left_w)
            _column_boundary(self.layouts, "clay", "left")
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "clay-centre",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                self._clay_viewport(ctx, clay_mode, widgets)

        _column_boundary(self.layouts, "clay", "right")
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "clay", columns["right"]),
            width=right_w,
            handle_length=right_w,
        )

    def _clay_viewport(self, ctx: Any, clay_mode: Any, widgets: Any) -> None:
        from imgui_bundle import imgui

        from .... import icons, tokens
        from ....main import TARGET_FPS
        from ....panes import overlay
        from .. import texture_link
        from .. import timeline as clay_timeline
        from .panes import header as clay_header
        from .panes import hud as clay_hud
        from .panes import menu as clay_menu
        from .panes import rail as clay_rail
        from .panes import timeline as clay_timeline_pane

        # One tick per drawn frame, before anything below can return: the UV pane
        # stamps its hover against this count and ``clay_mode.handle_key`` reads it
        # back (``ClayState.frame_serial``).
        clay_mode.ensure(ctx).frame_serial += 1
        self._clay_tabs(ctx, clay_mode)
        tab = clay_mode.active(ctx)
        if tab is None:
            self._clay_empty(ctx, clay_mode)
            return
        # The header, between the tabs and the image. Before the content region
        # is read, exactly as Inker's context bar and Plotter's toolbar are: the
        # viewport sizes itself from what is left, so a strip drawn after the
        # measurement is a strip drawn over the render.
        clay_header.draw(ctx, getattr(ctx, "clay_view", None))
        avail = imgui.get_content_region_avail()
        # And the hint line's own row, reserved rather than drawn over: a line
        # the viewport has already claimed the height for is a line clipped away
        # at the bottom of the pane, which is where every status row in this app
        # has gone wrong at least once. (The palette strip that used to sit above
        # it is the Inspector's Colour card now, so the render owns that height.)
        hint_h = float(tokens.sp(clay_hud.HINT_H))
        # The timeline strip's height, which is nothing outside Pose mode: reserved the way the
        # hint line is, so the render is sized from what the strip leaves.
        timeline_h = clay_timeline_pane.height(tab.doc)
        body_h = max(avail.y - hint_h - timeline_h, 1.0)
        origin = imgui.get_cursor_screen_pos()
        # The tool rail, to the left of the render and as tall as it: a child of
        # this pane rather than a docked one, so it costs the model its width and
        # nothing else. Drawn before the rect is read -- the render is sized from
        # what is left of the row.
        clay_rail.draw(ctx, body_h)
        imgui.same_line()
        pos = imgui.get_cursor_screen_pos()
        rect = (
            pos.x,
            pos.y,
            max(avail.x - (pos.x - origin.x), 1.0),
            body_h,
        )
        state = clay_mode.ensure(ctx)
        # Textures painted in Inker land here, before anything below reads the
        # document: every tab with a link, not only the drawn one, and one undo
        # step per return (``texture_link``'s docstring says why the trigger is
        # a history head).
        texture_link.pull_all(ctx)
        view = self._ensure_build_view()
        # One viewport, many tabs: the camera belongs to the *document*, so it
        # is snapshotted off the live one on the way out of a tab and put back
        # on the way in. Done here rather than in ``ClayState.activate`` because
        # this is the only place that has the viewport -- and it is keyed on
        # what is being drawn rather than on the switch, so a tab restored from
        # a ``.rblk`` or closed out from under the pointer lands correctly too.
        clay_mode.sync_active_camera(ctx)
        if self._clay_camera_tab != tab.uid:
            clay_mode.remember_camera(ctx, state.get(self._clay_camera_tab))
            clay_mode.apply_camera(ctx, tab)
            self._clay_camera_tab = tab.uid
        state.camera_tab = tab.uid
        if state.frame_pending:
            # The other half of ``F``: the mode recorded the intent and this is
            # the only place that has the viewport to act on it (B6).
            #
            # After the hand-off above, never before it (the 2026-10-11 audit's
            # clay-46): on the frame a tab switches the live camera still
            # belongs to the *outgoing* tab until ``sync_active_camera`` and
            # the hand-off have stored it there, and framing first put the
            # incoming document's framing on it -- which the hand-off then
            # stored as the outgoing tab's own camera. Framed here, the camera
            # is the incoming tab's, ``camera_tab`` already names it, and the
            # view exists (the old position consumed a frame request, and
            # dropped it, on the very frame the view was first built).
            state.frame_pending = False
            whole, state.frame_whole = state.frame_whole, False
            if whole:
                self._frame_clay_whole()
            else:
                self._frame_clay_selection()
        # The shading mode decides two of the renderer's three switches and the
        # overlay decides the third: *Wire* is the surface replaced by its
        # edges, *Solid* is the surface drawn unlit, and the Wireframe overlay
        # is edges drawn over whichever of those is showing.
        view.wireframe = state.shading == "wireframe"
        view.flat = state.shading == "solid"
        view.wire_overlay = bool(state.overlays.get("wire", False))
        view.xray = bool(state.xray)
        view.show_grid = state.grid
        view.grid_size = float(state.grid_size)
        view.god_light = bool(state.god_light)
        # Playback advances the playhead by the time since the last frame, before the view
        # draws it. ``delta_time`` for the reason Inker's playback uses it: a stalled frame
        # stalls the clip by the same amount it stalls everything else on screen.
        clay_timeline.tick(tab.doc, imgui.get_io().delta_time)
        texture = view.draw(tab.doc, rect, 1.0 / TARGET_FPS)
        imgui.image(widgets.texture_ref(texture), (rect[2], rect[3]), (0, 1), (1, 0))
        self._build_hovered = imgui.is_item_hovered()
        if not tab.doc.objects:
            # The tab is open and the render is live -- an empty grid, not a
            # missing document -- so the "start a document" screen
            # (``_clay_empty``) is the wrong one here. What is missing is the
            # *first shape*, and nothing on screen said so before this: the
            # viewport was just an empty grid until Tools was noticed at the
            # side (W1.5).
            imgui.set_cursor_screen_pos((rect[0], rect[1]))
            overlay.centred_empty(
                icons.BOX,
                "Add a shape",
                "Pick one from the tool rail.",
                action=overlay.action_for(ctx, "clay"),
            )
        self._clay_marquee(imgui, view, rect)
        self._clay_drag_hud(imgui, widgets, view, rect)
        # Over the render and inside the same clip: the widget is a control you
        # reach for without looking away from what you are turning, which is the
        # whole of why it is in the corner rather than in a pane.
        # Clears ``_build_hovered``: the flag was recorded off the render image
        # above, which cannot know a control has since been drawn over it, and
        # ``_clay_event`` routes the pygame press on it -- so a click on a ball
        # turned the camera and picked the mesh behind it in one gesture.
        if clay_hud.axis_widget(ctx, view, rect):
            self._build_hovered = False
        # Opposite corner from the widget: two readouts in one corner is one of
        # them unreadable.
        clay_hud.stats_overlay(ctx, rect)
        clay_menu.draw(ctx, view)
        if not self._build_hovered:
            # clay-83: the pointer is on a panel, a card or a control, and no
            # mouse move will reach the view to say so.
            view.clear_hover(tab.doc)
        # Last, under the render and the rail both: read when you are stuck, and a
        # line over the model covers the thing you are stuck on.
        if timeline_h:
            imgui.set_cursor_screen_pos((origin.x, rect[1] + rect[3]))
            clay_timeline_pane.draw(ctx, tab, avail.x, timeline_h)
        imgui.set_cursor_screen_pos((origin.x, rect[1] + rect[3] + timeline_h))
        clay_hud.hint_line(ctx)

    def _clay_tabs(self, ctx: Any, clay_mode: Any) -> None:
        """Clay's open documents, which nothing has ever drawn.

        The document model was all there -- ``docs``, ``active``, ``activate``,
        ``cycle``, ``close`` -- and the only thing missing was the bar: Ctrl+Tab
        switched between documents with nothing on screen to say there was more
        than one, and ``close`` had no caller at all.

        Drawn above ``_clay_empty`` as well as above the viewport, because the
        last tab closing is exactly when the bar disappears and the empty state
        has to be what is underneath it.

        ``unsaved_document`` rather than a ``"* "`` prefix, which is Inker's
        rule and the right one: the title is half of the tab's identity.
        """

        from .... import docmodes

        state = clay_mode.ensure(ctx)
        docmodes.tab_bar(ctx, state, "clay-tabs", lambda tab: clay_mode.close_tab(ctx, tab.uid))

    def _clay_empty(self, ctx: Any, clay_mode: Any) -> None:
        """What Clay shows with nothing open, mirroring the raster editor's.

        Buttons rather than a sentence: ``new_document`` was reachable only
        through Ctrl+N, so the empty state told the user to "start a document"
        and offered no way to.
        """

        from .... import anchors, widgets

        # The tour rings the whole empty pane: ``nothing_open`` draws its own
        # buttons and recents, so the last item after it is not "New model".
        anchors.mark_window("clay/empty")
        # This was written as a copy of ``inker_canvas._empty`` and the copy
        # dropped the ``sp()`` scaling, so at 150 % the raster editor's empty
        # state grew with the text while Clay's kept 240-*physical*-pixel
        # buttons under 1.5x labels -- which is where a label stops fitting its
        # button. Both are one function now (the UI redesign, wave 2), which is the
        # only fix that also holds for the next copy.
        widgets.nothing_open(
            "Start with a box, open a document, or drop a .rblk on the window.",
            [
                # The primary, because an empty Clay used to need "New model" and
                # then a hunt for the rail before the user had anything to look
                # at. "New model" stays beside it for the person who wants a
                # blank page (the torch tour's ``new-model`` step names it).
                ("Start with a box", lambda: self._clay_start_with_box(ctx, clay_mode)),
                ("New model", lambda: clay_mode.new_document(ctx)),
                ("Open a file...", lambda: clay_mode.ask_open(ctx)),
            ],
            # On the empty canvas *as well as* on the bridge (B5 put it there
            # alone): Inker and Sirens offered recents under the two buttons
            # and this mode did not, so the same screen answered "how do I get
            # back to yesterday's work" in two document modes and not the other three.
            recent_paths=clay_mode.recent_paths(ctx),
            on_open=lambda path: clay_mode.open_path(ctx, Path(path)),
        )

    def _clay_start_with_box(self, ctx: Any, clay_mode: Any) -> None:
        """A new model with one box in it, placed the way a click on the rail places
        it: on the ground, selected and framed.

        The shape is read off ``primitives.CLAY_GENERATORS`` (the pane never names
        a generator) and placed through ``place_primitive``, the human door, so
        this is one tab and one undo step for the add -- the same object the
        empty viewport's own "Add a primitive" button makes.
        """
        from .....kernels.mesh import primitives as bp
        from .panes import tools as clay_tools

        tab = clay_mode.new_document(ctx)
        clay_tools.place_primitive(ctx, tab.doc, bp.CLAY_GENERATORS[0][1][0])

    def _clay_marquee(self, imgui: Any, view: Any, rect: Any) -> None:
        """The selection rectangle, drawn in imgui rather than in GL.

        It is a two-dimensional screen decoration with no depth and no place in
        the scene, so putting it through the renderer would mean a vertex
        buffer rebuilt every mouse-move for four corners. The draw list is
        already there and already clipped to this window.
        """
        box = getattr(view, "marquee", None)
        if box is None:
            return
        draw = imgui.get_window_draw_list()
        x0, y0 = rect[0] + min(box[0], box[2]), rect[1] + min(box[1], box[3])
        x1, y1 = rect[0] + max(box[0], box[2]), rect[1] + max(box[1], box[3])
        draw.add_rect_filled((x0, y0), (x1, y1), imgui.get_color_u32((1, 1, 1, 0.08)))
        draw.add_rect((x0, y0), (x1, y1), imgui.get_color_u32((1, 1, 1, 0.55)))

    def _clay_drag_hud(self, imgui: Any, widgets: Any, view: Any, rect: Any) -> None:
        """What the live drag currently amounts to, above the cursor.

        In the draw list for ``_clay_marquee``'s reason, and *near the cursor*
        rather than in a corner: the number answers a question the user is
        asking with their hand, and a readout they have to look away to find is
        one they stop looking at. It draws only while a drag is live, so an idle
        viewport is unchanged.
        """
        text = getattr(view, "drag_hud", "")
        if not text or not getattr(view, "dragging", False):
            return
        from .... import theme
        from ....tokens import sp

        mouse = imgui.get_mouse_pos()
        x, y = mouse.x + sp(18), mouse.y - sp(28)
        draw = imgui.get_window_draw_list()
        size = imgui.calc_text_size(text)
        pad = sp(6)
        draw.add_rect_filled(
            (x - pad, y - pad),
            (x + size.x + pad, y + size.y + pad),
            imgui.get_color_u32(theme.rgba(theme.ELEV_2, 0.92)),
            sp(4),
        )
        draw.add_text((x, y), imgui.get_color_u32(theme.rgba(theme.TEXT)), text)

    def _ensure_build_view(self) -> Any:
        """Clay's viewport, built on first use -- and mirrored onto the ctx,
        ``_ensure_poser_viewer``'s way: clay_mode's drag keyboard, the axis
        views and ``camera_of`` all read ``ctx.clay_view``, and without the
        mirror every one of them found None forever."""
        from .view import ClayView

        if self.clay_view is None:
            self.clay_view = ClayView(self.ctx, self.app_ctx)
            self.app_ctx.clay_view = self.clay_view
        return self.clay_view

    def _frame_clay_selection(self) -> None:
        """F, in Clay. Frames the selection, or the whole document."""
        from .. import mode as clay_mode

        tab = clay_mode.active(self.app_ctx)
        if tab is not None and self.clay_view is not None:
            self.clay_view.frame_selection(tab.doc)
            self._clay_framed(tab)

    def _frame_clay_whole(self) -> None:
        """The framing a shape placed off-screen asks for: every visible object."""
        from .. import mode as clay_mode

        tab = clay_mode.active(self.app_ctx)
        if tab is not None and self.clay_view is not None:
            self.clay_view.frame_whole(tab.doc)
            self._clay_framed(tab)

    def _clay_framed(self, tab: Any) -> None:
        """Mark the tab's camera as one the user has seen framed.

        A tab whose first shape has just been framed is *fitted*: without the
        mark the next tab hand-off found it unfitted and framed it again,
        throwing away wherever the user had orbited to since.
        """
        if tab.doc.objects:
            tab.view.read_from(self.clay_view.camera)
            tab.view.fitted = True

    def _capture_clay_thumbnail(self, job_id: str) -> None:
        """The library card's picture, from Clay's viewport.

        The general capture -- reading the framebuffer and queuing the PNG
        encode -- is ``shell.tasks.TasksMixin._capture_thumbnail_from``, since
        ``_on_task_done`` calls it directly for Mason's own export too. This is
        the one-argument convenience the Clay-only call sites use.
        """
        self._capture_thumbnail_from(job_id, self.clay_view)
