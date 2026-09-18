"""Plotter's workspace: the sidebar / centre / sidebar skeleton, Tiled's way.

A **mixin on** :class:`~.shell.app.App`, ``clay_viewport.ClayViewport``'s idiom
restated -- ``self`` here is the App and the method's body is unchanged from
the line it stood on in ``studio/main.py`` before the P4 restructure moved
each of the six inline ``_*_workspace`` methods to a module of its own.

The shell names this module reaches are imported *inside* the method that
uses them, the same rule ``studio/modes/clay/ui/viewport.py`` states: ``main`` imports
:class:`~.shell.app.App` (which assembles this mixin) to build the class, so a
module-scope import back would be a cycle.
"""

from __future__ import annotations


class PlotterWorkspace:
    """Plotter's pane drawing, mixed into :class:`~.shell.app.App`."""

    def _plotter_workspace(self) -> None:
        """The same sidebar / centre / sidebar skeleton every other mode uses,
        arranged the way Tiled arranges its own:

            [ plotter-properties ]  the toolbar  [ plotter-layers  ]
            [ plotter-stamps     ]  the map      [ plotter-tileset ]
                                                 [ plotter-bridge  ]

        Both sidebars are ``skeletons.plotter``, which is where the argument
        for that arrangement is written down. The toolbar is not a slot: it is
        a strip inside the centre pane, drawn by ``plotter_canvas`` between the
        tab bar and the map, exactly as Inker's context bar is.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from . import skeletons
        from .panes import plotter_canvas, plotter_tileset_editor
        from .shell.frame import _column_boundary

        ctx = self.app_ctx
        lay = self.layout
        left_w = layout_mod.sidebar_width("left")
        right_w = layout_mod.sidebar_width("right")
        # Both sidebars through ``layout.column`` over ``skeletons.plotter``
        # (wave 5), so the arrangement is data a saved layout can permute.
        columns = skeletons.for_mode(ctx, "plotter")
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "plotter", columns["left"]),
            width=left_w,
            handle_length=left_w,
        )

        _column_boundary(self.layouts, "plotter", "left")
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        # The tileset editor is a **sheet over the centre pane**, drawn instead
        # of the map: the branch ``_review_workspace`` already takes, with the
        # role that already exists for it. Not a mode (a 21-place checklist,
        # including prose asserting the mode count) and not a document kind
        # (which would teach nine places a second shape).
        sheet = plotter_tileset_editor.active(ctx)
        with layout_mod.pane(
            "plotter-centre",
            (width, 0),
            layout_mod.PaneRole.SHEET if sheet else layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                if sheet:
                    plotter_tileset_editor.draw(ctx)
                else:
                    plotter_canvas.draw(ctx)

        _column_boundary(self.layouts, "plotter", "right")
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "plotter", columns["right"]),
            width=right_w,
            handle_length=right_w,
        )
