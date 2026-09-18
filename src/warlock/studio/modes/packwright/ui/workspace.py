"""Packwright's workspace: sources and settings on the left, the atlas in the
middle, items and the bridge on the right.

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


class PackwrightWorkspace:
    """Packwright's pane drawing, mixed into :class:`~.shell.app.App`."""

    def _packwright_workspace(self) -> None:
        """The same skeleton again:

            [ packwright-sources  ]              [ packwright-items  ]
            [ packwright-settings ]  the atlas   [ packwright-bridge ]

        The centre pane is also the mode's heartbeat -- there is no per-mode
        update hook, so the pane that draws is what pumps the repack request.
        """
        from imgui_bundle import imgui

        from .... import layout as layout_mod
        from ....shell.frame import _column_boundary, _split_column
        from .panes import bridge as packwright_bridge
        from .panes import items as packwright_items
        from .panes import preview as packwright_preview
        from .panes import settings as packwright_settings
        from .panes import sources as packwright_sources

        ctx = self.app_ctx
        lay = self.layout
        left_w = layout_mod.sidebar_width("left")
        right_w = layout_mod.sidebar_width("right")

        _split_column(
            ctx,
            lay,
            split_id="packwright-sources",
            handle_length=left_w,
            width=left_w,
            edge=layout_mod.PaneEdge.RIGHT,
            top=("packwright-sources", layout_mod.PaneRole.SIDEBAR, packwright_sources.draw),
            bottom=("packwright-settings", layout_mod.PaneRole.SIDEBAR, packwright_settings.draw),
        )

        _column_boundary(self.layouts, "packwright", "left")
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "packwright-centre",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                packwright_preview.draw(ctx)

        _column_boundary(self.layouts, "packwright", "right")
        _split_column(
            ctx,
            lay,
            split_id="packwright-items",
            handle_length=right_w,
            width=right_w,
            edge=layout_mod.PaneEdge.LEFT,
            top=("packwright-items", layout_mod.PaneRole.INSPECTOR, packwright_items.draw),
            bottom=("packwright-bridge", layout_mod.PaneRole.INSPECTOR, packwright_bridge.draw),
        )
