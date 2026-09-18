"""Troupe's workspace: cast and settings on the left, the sprite in the
middle, sheets and the bridge on the right.

A **mixin on** :class:`~.shell.app.App`, ``clay_viewport.ClayViewport``'s idiom
restated -- ``self`` here is the App and the method's body is unchanged from
the line it stood on in ``studio/main.py`` before the P4 restructure moved
each of the six inline ``_*_workspace`` methods to a module of its own.

The shell names this module reaches are imported *inside* the method that
uses them, the same rule ``studio/modes/clay/ui/viewport.py`` states: ``main`` imports
:class:`~.shell.app.App` (which assembles this mixin) to build the class, so a
module-scope import back would be a cycle. Both ``_split_column`` and
``_column_boundary`` are module-level functions in ``shell/frame.py`` now,
not methods -- this workspace was already reaching across a module boundary
for them before the split, just one that was invisible while both lived in
``main.py``.
"""

from __future__ import annotations


class TroupeWorkspace:
    """Troupe's pane drawing, mixed into :class:`~.shell.app.App`."""

    def _troupe_workspace(self) -> None:
        """The same skeleton the other five use:

            [ troupe-cast     ]                  [ troupe-sheets ]
            [ troupe-settings ]   the sprite     [ troupe-bridge ]

        The centre pane is also the mode's heartbeat -- there is no per-mode
        update hook, so the pane that draws is what pumps the preview clock.
        """
        from imgui_bundle import imgui

        from .... import layout as layout_mod
        from ....shell.frame import _column_boundary, _split_column
        from .panes import bridge as troupe_bridge
        from .panes import characters as troupe_characters
        from .panes import preview as troupe_preview
        from .panes import settings as troupe_settings
        from .panes import sheets as troupe_sheets

        ctx = self.app_ctx
        lay = self.layout
        left_w = layout_mod.sidebar_width("left")
        right_w = layout_mod.sidebar_width("right")

        _split_column(
            ctx,
            lay,
            split_id="troupe-cast",
            handle_length=left_w,
            width=left_w,
            edge=layout_mod.PaneEdge.RIGHT,
            top=("troupe-cast", layout_mod.PaneRole.SIDEBAR, troupe_characters.draw),
            bottom=("troupe-settings", layout_mod.PaneRole.SIDEBAR, troupe_settings.draw),
        )

        _column_boundary(self.layouts, "troupe", "left")
        width = layout_mod.centre_width()
        # No scroll-with-mouse for the reason Plotter's centre has none: the
        # wheel belongs to the picture. It said so from the day the mode was
        # built and it was not true until W0.3 -- no Troupe pane read the
        # wheel, so the flag took it away from the pane's scrollbar and gave it
        # to nothing. ``troupe_preview`` zooms with it now, over the sprite.
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "troupe-centre",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                troupe_preview.draw(ctx)

        _column_boundary(self.layouts, "troupe", "right")
        _split_column(
            ctx,
            lay,
            split_id="troupe-sheets",
            handle_length=right_w,
            width=right_w,
            edge=layout_mod.PaneEdge.LEFT,
            top=("troupe-sheets", layout_mod.PaneRole.INSPECTOR, troupe_sheets.draw),
            bottom=("troupe-bridge", layout_mod.PaneRole.INSPECTOR, troupe_bridge.draw),
        )
