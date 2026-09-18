"""Multi-document state for Packwright, without imgui.

The Inker mode's ``state.py`` / ``plotter_state`` split, third instance, and the
view type is imported for the same reason it is there: a zoomable, pannable 2D
viewport is not about pixels, and a second copy would be a second set of
clamping rules and a second Ctrl+0. It lives at ``shell.paintview`` rather than
in either mode, since dev/RESTRUCTURE.md's P5 -- Inker was never its owner
either, only its first caller.

**The atlas is a task result, not a document field.** ``PackDoc`` holds sources
and settings and derives its layout on demand -- that is what makes a re-export
reproducible -- but *composing* the atlas is numpy work over every sprite, which
is not frame-thread work at a hundred of them. So the composed pixels and the
layout they came from live here, on the tab, adopted when the task lands.

**``pack_generation`` is what the texture cache keys on.** A repack replaces the
pixels wholesale, and an upload gated on anything else -- a rev, a hash, the
settings -- either re-uploads a megapixel every frame or misses a change. A
counter bumped in exactly one place is the only version of this that is both.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ... import docmodes
from ...shell.paintview import PaintView

WPACK_SUFFIX = ".wpack"

_uids = itertools.count(1)


@dataclass
class PackTab(docmodes.DocTab):
    """One tab (``docmodes.DocTab`` holds the shared fields)."""

    uid: str = field(default_factory=lambda: f"pw{next(_uids)}")
    view: PaintView = field(default_factory=PaintView)

    # The last successful pack. ``layout`` is what the items pane lists and the
    # preview outlines; ``atlas`` is what it draws.
    layout: Any = None
    atlas: np.ndarray | None = None
    # Bumped once, when a pack is adopted. The texture cache keys on it.
    pack_generation: int = 0
    # A pack is in flight. Separate from ``saving`` because a repack does not
    # stop the user editing -- it only means the preview is a frame behind.
    packing: bool = False
    # Something changed that the current layout does not describe. A *flag*
    # pumped from the preview pane's draw rather than a direct submit, for the
    # reason ``findings_dirty`` is one: ``TaskRunner.submit`` refuses a key
    # already in flight and nothing re-arms it, so a burst of edits would
    # otherwise leave the last one unpacked for good.
    pack_dirty: bool = True
    # What the last pack could not do, if anything. Kept so the items pane can
    # say so rather than showing an empty list that looks like success.
    pack_error: str = ""

    @property
    def pack_stale_why(self) -> str:
        """Why the atlas on screen is not the one this document describes.

        Empty when it is. A *failed* pack clears ``packing`` and records
        ``pack_error``, but leaves ``layout``/``atlas`` at the last pack that
        worked -- deliberately, because a picture the user can still look at
        beats a blank pane. What was missing is that nothing said so: the
        preview drew the old atlas with no mark on it and both exports wrote
        it, so a pack that could not fit produced a file describing sprites the
        document no longer holds (the 2026-09-02 review, section 7).
        """
        if not self.pack_error or self.atlas is None:
            return ""
        return (
            "The last pack failed, so this is the atlas from before it: "
            f"{self.pack_error}"
        )

    def adopt_pack(self, layout: Any, atlas: np.ndarray) -> None:
        """Take a finished pack. The one place ``pack_generation`` moves.

        **``pack_dirty`` is deliberately not touched here.** An edit made while
        a pack was in flight set it, and that edit is not in the layout landing
        now -- clearing it would drop the edit for good, because ``request_pack``
        clears the flag only on an accepted submit and nothing else re-arms it.
        The flag is cleared at the submit, never at the adoption.
        """
        # Whether the atlas changed *shape*. Only then is the view's framing
        # about something that is no longer there: a repack of the same size --
        # which is most of them, since the size search lands on the same answer
        # until the sprite set grows past it -- used to throw away the zoom and
        # the pan on every rename, every trim toggle, every padding nudge, so
        # anybody working at 400% on one corner was flung back to "fit" a
        # dozen times an hour.
        before = None if self.layout is None else (self.layout.width, self.layout.height)
        self.layout = layout
        self.atlas = atlas
        self.pack_generation += 1
        self.packing = False
        self.pack_error = ""
        if before != (layout.width, layout.height):
            self.view.fitted = False


@dataclass
class PackwrightState(docmodes.DocTabs[PackTab]):

    # Which source row is selected, by uid. View state, shared between the
    # sources list, the items list and the preview's highlight -- one answer to
    # "which sprite are we talking about" rather than three.
    selected: int | None = None
    # Which source row is being renamed, by uid -- a double-click opens the
    # field, Plotter's and Clay's rule; the selected row used to grow one.
    renaming: int | None = None

    # A decoded tile sheet waiting for its cell size: ``(path, display name,
    # pixels)``, set by the decode task and consumed (or dropped, on cancel)
    # by the sources pane's popup. Inker's ``sheet_import`` trio, verbatim,
    # because a sheet cannot say its own tile size and the user answers in a
    # popup either way.
    tileset_import: tuple[str, str, np.ndarray] | None = None
    # Which tab asked. The 2026-09-11 audit's packwright-02: this used to be
    # absent, so ``on_task_done`` only checked the requesting tab was still
    # *open* before parking the sheet on these state-wide fields -- nothing
    # recorded *which* tab that was -- and ``import_tileset`` read
    # ``active(ctx)`` to find a target, which is whichever tab the user has
    # since switched to, not the one that asked. Set alongside
    # ``tileset_import`` in exactly one place (``packwright_mode.on_task_done``)
    # and read in exactly one (``import_tileset``); the popup panes that clear
    # ``tileset_import`` on cancel never read this, so a stale leftover value
    # is harmless -- ``import_tileset`` never consults it while
    # ``tileset_import`` is ``None``.
    tileset_import_uid: str = ""
    tileset_import_open: bool = False
    tileset_cell: tuple[int, int] = (32, 32)
    # Whether to drop tiles whose content another tile already carries, and
    # whether a flipped or turned copy counts as the same content.
    #
    # **Default off, both.** A repack is a faithful repack unless somebody asks
    # otherwise: an atlas quietly missing the four tiles that happened to be
    # rotations of each other is a bug report nobody can reproduce.
    tileset_dedup: bool = False
    tileset_dedup_flips: bool = False
    # The popup's counts, and the inputs they were computed from. Cached
    # because the dedup pass behind them is measured in hundreds of
    # milliseconds on a full sheet and the popup redraws every frame; see
    # ``panes.packwright_sources._tileset_popup``. Cleared wherever
    # ``tileset_import`` is, so a stale preview cannot outlive its sheet.
    tileset_preview_key: tuple[Any, ...] | None = None
    tileset_preview: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)

    def _switched(self, previous: str) -> None:
        # The selection names a source of the *previous* document.
        self.selected = None

    def _closed(self, was_active: bool) -> None:
        self.selected = None



def ensure(ctx: Any) -> PackwrightState:
    """The mode's state, built on first use.

    Here rather than in ``packwright_mode`` because this and :func:`active`
    touch exactly one thing -- ``ctx.state.packwright`` -- which is this
    module's whole charter, and neither knows a job or a task thread exists.
    The ``plotter_state`` shape.
    """
    state = ctx.state.packwright
    if state is None:
        state = PackwrightState()
        ctx.state.packwright = state
    return state


def active(ctx: Any) -> PackTab | None:
    """The focused tab, or ``None``. Deliberately *not* through :func:`ensure`:
    asking which atlas is open must not create the state that says none is."""
    state = ctx.state.packwright
    return state.active if state is not None else None


# The same answer in three of the four modes; Clay's is on ``stem`` on purpose.
title_for = docmodes.title_for
