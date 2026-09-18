"""Sirens' workspace: the sidebar / centre / sidebar skeleton, over a tracker
grid.

A **mixin on** :class:`~.shell.app.App`, ``clay_viewport.ClayViewport``'s idiom
restated -- ``self`` here is the App and the method's body is unchanged from
the line it stood on in ``studio/main.py`` before the P4 restructure moved
each of the six inline ``_*_workspace`` methods to a module of its own.

The shell names this module reaches are imported *inside* the method that
uses them, the same rule ``clay_viewport.py`` states: ``main`` imports
:class:`~.shell.app.App` (which assembles this mixin) to build the class, so a
module-scope import back would be a cycle.
"""

from __future__ import annotations


class SirensWorkspace:
    """Sirens' pane drawing, mixed into :class:`~.shell.app.App`."""

    def _sirens_workspace(self) -> None:
        """The same sidebar / centre / sidebar skeleton every other mode uses:

            [ sirens-transport ]                 [ sirens-instruments ]
            [ sirens-orders    ]  the grid       [ sirens-bridge      ]

        Both sidebars through ``layout.column`` over ``skeletons.sirens``,
        which is the direction of travel: Packwright still composes its columns
        by hand here, and a table is what a saved layout can permute.

        The centre column is one pane: the tab bar, the caret strip and the
        grid, in that order, all of which ``sirens_patterns.draw`` composes --
        the grid sizes its row count from what is left of the content region,
        so the strip has to be drawn before it rather than beside it here.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from . import skeletons
        from .panes import sirens_patterns
        from .shell.frame import _column_boundary

        ctx = self.app_ctx
        lay = self.layout
        left_w = layout_mod.sidebar_width("left")
        right_w = layout_mod.sidebar_width("right")
        columns = skeletons.for_mode(ctx, "sirens")
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "sirens", columns["left"]),
            width=left_w,
            handle_length=left_w,
        )

        _column_boundary(self.layouts, "sirens", "left")
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "sirens-centre",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                sirens_patterns.draw(ctx)

        _column_boundary(self.layouts, "sirens", "right")
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "sirens", columns["right"]),
            width=right_w,
            handle_length=right_w,
        )
