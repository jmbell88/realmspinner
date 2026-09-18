"""Mason's workspace: the viewport, its tabs, and the empty state.

A **mixin on** :class:`~.main.App`, ``clay_viewport.ClayViewport``'s own idiom
restated: ``self`` here is the App and every method's body is unchanged.

The viewport itself is ``MasonView``'s; what is here is the *pane* around it --
the layout skeleton, the tab bar, the empty state, and the frame-thread half
of the per-tab camera restore.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MasonViewport:
    """Mason's pane drawing, mixed into :class:`~.main.App`."""

    def _mason_workspace(self) -> None:
        """The sidebar / centre / sidebar skeleton every other mode uses,
        through ``skeletons.for_mode(ctx, "mason")`` -- the same shape
        ``_clay_workspace`` builds for Clay.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from . import mason_mode, skeletons, widgets
        from .shell.frame import _column_boundary

        ctx = self.app_ctx
        lay = self.layout
        left_w = layout_mod.sidebar_width("left")
        right_w = layout_mod.sidebar_width("right")
        columns = skeletons.for_mode(ctx, "mason")

        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "mason", columns["left"]),
            width=left_w,
            handle_length=left_w,
        )

        _column_boundary(self.layouts, "mason", "left")
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "mason-centre",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                self._mason_viewport(ctx, mason_mode, widgets)

        _column_boundary(self.layouts, "mason", "right")
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "mason", columns["right"]),
            width=right_w,
            handle_length=right_w,
        )

    def _mason_viewport(self, ctx: Any, mason_mode: Any, widgets: Any) -> None:
        from imgui_bundle import imgui

        from . import icons, mason_assets, tokens
        from .main import TARGET_FPS
        from .panes import mason_header, mason_hud, mason_menu, overlay

        self._mason_tabs(ctx, mason_mode)
        tab = mason_mode.active(ctx)
        if tab is None:
            self._mason_empty(ctx, mason_mode)
            return

        mason_header.draw(ctx, getattr(ctx, "mason_view", None))
        avail = imgui.get_content_region_avail()
        hint_h = float(tokens.sp(mason_hud.HINT_H))
        rect = (
            imgui.get_cursor_screen_pos().x,
            imgui.get_cursor_screen_pos().y,
            max(avail.x, 1.0),
            max(avail.y - hint_h, 1.0),
        )
        state = mason_mode.ensure(ctx)
        if state.frame_pending:
            state.frame_pending = False
            self._frame_mason_selection()
        view = self._ensure_mason_view()
        source = mason_assets.ensure(ctx)
        # One viewport, many tabs: keyed on what is being *drawn* rather than
        # on the switch, ``clay_viewport``'s own reason -- a tab restored from
        # a ``.wscn`` or closed out from under the pointer lands correctly too.
        if self._mason_camera_tab != tab.uid:
            mason_mode.remember_camera(ctx, state.get(self._mason_camera_tab))
            mason_mode.apply_camera(ctx, tab)
            self._mason_camera_tab = tab.uid

        view.wireframe = state.shading == "wireframe"
        view.flat = state.shading == "solid"
        view.wire_overlay = bool(state.overlays.get("wire", False))
        view.xray = bool(state.xray)
        view.show_grid = state.grid
        texture = view.draw(tab.doc, source, rect, 1.0 / TARGET_FPS)
        imgui.image(widgets.texture_ref(texture), (rect[2], rect[3]), (0, 1), (1, 0))
        self._mason_hovered = imgui.is_item_hovered()
        if not tab.doc.roots:
            imgui.set_cursor_screen_pos((rect[0], rect[1]))
            overlay.centred_empty(
                icons.BLOCKS,
                "Add something",
                "Pick one from the Assets panel.",
                action=overlay.action_for(ctx, "mason"),
            )
        # The armed placement the viewport's own click asked for, drained here
        # because what a placement *means* is ``mason_mode``'s and the view does
        # not import the controller -- ``menu_request`` just above is drained the
        # same way for the same reason.
        if view.place_request is not None:
            point, view.place_request = view.place_request, None
            mason_mode.place_armed(ctx, point)
        mason_hud.stats_overlay(ctx, view, rect)
        mason_menu.draw(ctx, view)
        imgui.set_cursor_screen_pos((rect[0], rect[1] + rect[3]))
        mason_hud.hint_line(ctx)

    def _mason_tabs(self, ctx: Any, mason_mode: Any) -> None:
        from . import docmodes

        state = mason_mode.ensure(ctx)
        docmodes.tab_bar(
            ctx, state, "mason-tabs", lambda tab: mason_mode.close_tab(ctx, tab.uid)
        )

    def _mason_empty(self, ctx: Any, mason_mode: Any) -> None:
        from . import widgets

        widgets.nothing_open(
            "Start a scene, open a document, or drop a .wscn on the window.",
            [
                ("New scene", lambda: mason_mode.new_document(ctx)),
                ("Open a file...", lambda: mason_mode.ask_open(ctx)),
            ],
            recent_paths=mason_mode.recent_paths(ctx),
            on_open=lambda path: mason_mode.open_path(ctx, Path(path)),
        )

    def _ensure_mason_view(self) -> Any:
        """Built on first use, ``_ensure_build_view``'s reason: Mason's own
        keyboard bindings, axis views and ``camera_of`` all read
        ``ctx.mason_view``, and without this every one of them would find
        nothing there until the pane happened to run first."""
        from .mason_view import MasonView

        if self.mason_view is None:
            self.mason_view = MasonView(self.ctx, self.app_ctx)
            self.app_ctx.mason_view = self.mason_view
        return self.mason_view

    def _frame_mason_selection(self) -> None:
        from . import mason_assets, mason_mode

        tab = mason_mode.active(self.app_ctx)
        if tab is not None and self.mason_view is not None:
            source = mason_assets.ensure(self.app_ctx)
            self.mason_view.frame_selection(tab.doc, source)
