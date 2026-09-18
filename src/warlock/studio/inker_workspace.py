"""Inker's workspace: colour / canvas / tools, with the timeline along the
bottom.

A **mixin on** :class:`~.shell.app.App`, ``clay_viewport.ClayViewport``'s idiom
restated -- ``self`` here is the App and the method's body is unchanged from
the line it stood on in ``studio/main.py`` before the P4 restructure moved
each of the six inline ``_*_workspace`` methods to a module of its own.

The shell names this module reaches are imported *inside* the method that
uses them, the same rule ``studio/modes/clay/ui/viewport.py`` states: ``main`` imports
:class:`~.shell.app.App` (which assembles this mixin) to build the class, so a
module-scope import back would be a cycle. ``_column_boundary`` is the one
that was not a bare name to begin with, even in ``main.py``: it is a
module-level function there (now in ``shell/frame.py``), and this method
already reached across a module boundary for it, just one that used to be
invisible because both lived in the same file.
"""

from __future__ import annotations


class InkerWorkspace:
    """Inker's pane drawing, mixed into :class:`~.shell.app.App`."""

    def _inker_workspace(self) -> None:
        """Colour / canvas / tools, with the timeline along the bottom.

        Aseprite's default arrangement, which is what ``skeletons.inker`` now
        declares -- the previous shape (a 90 px tool rail on the left, the
        palette on the right, no bottom region at all) was its *Mirrored
        Default* preset plus a bug.

        Deliberately not a takeover of the whole window: the progress card
        floats over every mode, so a trellis run started before switching here
        is still visible while painting.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from . import skeletons, tokens
        from .panes import inker_canvas, inker_timeline
        from .shell.frame import _column_boundary
        from .tokens import sp

        ctx = self.app_ctx
        lay = self.layout
        left_w = layout_mod.sidebar_width("left")
        right_w = layout_mod.sidebar_width("right")
        state = ctx.state.inker
        tab = None if state is None else state.active
        # The tile panel appears with the tilesets, the way the preview appears
        # with the frames: a drawing that has never seen a tilemap layer is
        # byte-for-byte the workspace it always was, and a fixed-height palette
        # taken out of every Inker session for a feature most of them do not
        # use is a cost with no matching benefit. The *verbs* that make the
        # first tileset are not hidden with it -- they are menu rows, which are
        # always drawn.
        # (The tile panel's own ``when`` predicate answers that in the
        # skeleton table, which is where the shape of a workspace lives now.)
        #
        # Both sidebars go through ``layout.column`` over ``skeletons.inker``
        # (P5.1): one renderer, one height arithmetic, and a saved layout has
        # something to be a permutation of.
        columns = skeletons.for_mode(ctx, "inker")
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "inker", columns["left"]),
            width=left_w,
            handle_length=left_w,
            on_hidden=lambda _slot: None,
        )

        _column_boundary(self.layouts, "inker", "left")
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        imgui.begin_group()
        # A *positive* height, never a bottom offset: with little room left a
        # negative height collapses the canvas child to nothing and the canvas
        # -- and its texture uploads -- silently stops being drawn. Same rule
        # the status bar inside ``inker_canvas`` already follows.
        #
        # **Unconditional, because the strip is unconditional.** It was gated
        # on ``doc.anim is not None`` from 2a56df6 until 2026-08-23, which is
        # what hid the layer list from every still document; it was then gated
        # on ``state.timeline_open``, which is what a ``Tab`` key could hide.
        # Both gates are gone: the strip holds the layers, so a hidden strip is
        # a document with no visible layer list, and the height drag is the
        # thing hiding it was really being used for.
        available_h = imgui.get_content_region_avail().y
        centre_h = 0.0
        strip_key = "inker-timeline"
        if tab is not None:
            # A high UI scale can make the timeline's preferred design-pixel
            # height consume the whole physical window. The canvas remains the
            # anchor in that case and the timeline becomes the compressed,
            # scrollable pane. The ratio is frame-local, just like horizontal
            # side-panel compression; it never rewrites a saved share.
            strip = max(
                sp(inker_timeline.STRIP_H),
                available_h * lay.share("inker-timeline"),
            )
            centre_h = max(available_h - strip, available_h * 0.62)
        with layout_mod.pane(
            "inker-centre",
            (width, centre_h),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                inker_canvas.draw(ctx)
            elif tab is not None:
                # A keyboard zoom rung banked by ``handle_key`` is consumed
                # inside the canvas child; on a frame the centre pane does not
                # draw at all it would survive and fire later, unprompted.
                # Dropped here for the same reason the canvas's own invisible
                # branch drops it.
                tab.view.pending_zoom_rung = 0
        if tab is not None:
            # The handle between the canvas and the strip. Dragging it *down*
            # is a smaller strip, which is why the delta is subtracted: the
            # share names the timeline's portion, not the canvas's.
            drag = layout_mod.splitter(f"{strip_key}-share", vertical=False, length=width)
            if drag and available_h > 0:
                before = lay.share(strip_key)
                lay.set_share(strip_key, before - drag * tokens.SCALE / available_h)
                if lay.share(strip_key) != before:
                    lay.save()
            with layout_mod.pane(
                "inker-timeline",
                (width, 0),
                layout_mod.PaneRole.SHEET,
                edge=layout_mod.PaneEdge.TOP,
            ) as visible:
                if visible:
                    inker_timeline.draw(ctx)
        imgui.end_group()

        _column_boundary(self.layouts, "inker", "right")
        # **Preview / Tools / Tiles / Generation**, top to bottom, with a
        # handle between each adjacent shareable pair. The toolbox moved here
        # from the left column in this wave: the note it used to carry said
        # that putting it on the right "would put the toolbox on the far side
        # of the canvas from the hand", and what answers that is Aseprite's own
        # default, which is the program these users already have open.
        #
        # Which panes are here, and in what order, is now the active saved
        # layout's answer (wave 5) -- reconciled against this table every read
        # and never written back.
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "inker", columns["right"]),
            width=right_w,
            handle_length=right_w,
            on_hidden=lambda _slot: None,
        )
