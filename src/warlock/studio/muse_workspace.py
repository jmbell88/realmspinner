"""Muse's workspace: the brief across the top, the takes in the middle, the
recipe beside, the player along the bottom once a take has been auditioned.

A **mixin on** :class:`~.shell.app.App`, ``clay_viewport.ClayViewport``'s idiom
restated -- ``self`` here is the App and the method's body is unchanged from
the line it stood on in ``studio/main.py`` before the P4 restructure moved
each of the six inline ``_*_workspace`` methods to a module of its own.

**One defect the move surfaced, not one it made**: the method read the
module-level ``tokens`` binding ``main.py`` imports at the top of the file
for its own reason (``VIEWER_KEY = viewer_embed.LOAD_KEY`` and friends) with
no import of its own -- every one of the other five ``_*_workspace`` methods
names ``tokens`` in its own local import line, and this was the one that
silently rode on the enclosing module's namespace instead. It still resolved
in ``main.py``, because everything lived in one module; moved here alone it
would have raised a ``NameError`` on the first frame Muse ever drew. Fixed by
adding the import this method always needed, not by changing what it does.

The shell names this module reaches are imported *inside* the method that
uses them, the same rule ``clay_viewport.py`` states: ``main`` imports
:class:`~.shell.app.App` (which assembles this mixin) to build the class, so a
module-scope import back would be a cycle.
"""

from __future__ import annotations


class MuseWorkspace:
    """Muse's pane drawing, mixed into :class:`~.shell.app.App`."""

    def _muse_workspace(self) -> None:
        """The brief across the top, the takes in the middle, the recipe beside:

            [ muse-brief                                        ]
            [ the takes                       ]  [ muse-recipe  ]

        Composed by hand rather than through ``skeletons``, and **one** sidebar
        rather than the pair that table is built for. Packwright is the
        precedent that hand composition here is current rather than legacy; the
        reason it applies is that Muse has nothing to put in a second column.
        Two columns of which one is empty is a worse answer than one column.

        The bar is a full-width pane above both, which is ``create_brief``'s
        arrangement -- except that it is unconditional, because Muse has no
        stages for it to be absent on.

        The player is a fourth pane along the bottom, full width:

            [ muse-brief                                        ]
            [ the takes                       ]  [ muse-recipe  ]
            [ muse-player                                       ]

        Full width for the reason its own docstring gives -- 240 seconds across
        a 260 dp sidebar is a second per pixel, and a loop marker dragged at
        that scale is a guess. Drawn only once a take has been auditioned, so
        the two columns get the whole height until there is something to put
        under them.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from . import muse_brief, tokens
        from .panes import muse_player, muse_recipe, muse_results
        from .shell.frame import _column_boundary

        ctx = self.app_ctx
        right_w = layout_mod.sidebar_width("right")

        with layout_mod.pane(
            "muse-brief",
            (0, tokens.sp(muse_brief.BAR_H)),
            layout_mod.PaneRole.CONTENT,
            edge=layout_mod.PaneEdge.BOTTOM,
            title="The brief bar",
        ) as visible:
            if visible:
                muse_brief.draw(ctx)

        # **The row's height is one number, and everything in the row gets it.**
        # The strip is conditional, so what is left over has to be measured;
        # what cannot be left to chance is that the two columns *and the
        # boundary handle between them* are all told the same figure. They were
        # not: the splitter defaults to ``get_content_region_avail().y``, so it
        # claimed the whole remainder while the columns were shortened by the
        # strip's height, and the handle is what the row then sized itself to.
        # ``muse_player.draw`` ran with -8 px left (measured, 2560x1369),
        # ``begin_child`` returned false, and the mode reserved 148 dp for a
        # band that drew nothing at all -- no waveform, no playhead, no
        # transport, no loop markers -- in every build and every screenshot
        # this repo has ever taken of Muse.
        strip = muse_player.should_draw(ctx)
        gap = imgui.get_style().item_spacing.y
        body = imgui.get_content_region_avail().y
        body_h = max(body - (tokens.sp(muse_player.STRIP_H) + gap), 1.0) if strip else body

        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "muse-centre",
            # ``centre_width()`` alone. It already answers "what is left once
            # the right sidebar is reserved", measured from a cursor with no
            # left sidebar drawn before it -- so adding the left sidebar's
            # width **counted the space Muse does not use twice**, and the
            # centre came out 2443 px wide inside a 2466 px row. That left 0 px
            # for the recipe column: it was clipped out of every frame, which
            # is why the right-hand third of every Muse screenshot in this repo
            # is empty. Muse has no left column to give the space back from.
            (layout_mod.centre_width(), body_h),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                muse_results.draw(ctx)

        _column_boundary(self.layouts, "muse", "right", length=body_h)
        with layout_mod.pane(
            "muse-recipe",
            (right_w, body_h),
            layout_mod.PaneRole.SIDEBAR,
            edge=layout_mod.PaneEdge.LEFT,
        ) as visible:
            if visible:
                muse_recipe.draw(ctx)

        if strip:
            muse_player.draw(ctx)
