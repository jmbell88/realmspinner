"""Troupe's right-bottom pane: the ways out.

Every one of them is an existing bridge rather than a new writer. A character
sheet is an ordinary sheet plus an ``animation`` block, so Inker's grid slicer,
Packwright's sheet adder and the sheet exporters all already read it -- and a
second path would be a second dialect of one format, which is the mistake
``sheet.sidecar``'s docstring spends a paragraph refusing.

Four of them now, and the last two are the ones that produce *files*: Inker
and Packwright hand the sheet to another mode. Export package copies the PNG
and its JSON out together for an engine -- together, because either one alone
is an asset nothing can interpret -- and ``service.characters.export_package``
puts that promise on ``export.staged_copy_all``. Export frames cuts the same
atlas into one PNG per frame, in clip and compass-direction folders, for an
engine that wants frame folders instead of an atlas-plus-sidecar pair; the
same promise, on ``export.staged_tree``.
"""

from __future__ import annotations

from typing import Any

from .. import tokens, troupe_mode, verbs, widgets
from ..manual import render as manual_render
from ..tokens import sp


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = troupe_mode.ensure(ctx)
    widgets.section("Take it somewhere")
    manual_render.help_button(ctx, "troupe-bridge")

    ready = bool(state.job_id and state.sheet_id)
    if widgets.disabled_button(
        verbs.open_in("inker"),
        ready,
        (-1, 0),
        reason="Pick a character sheet first.",
        tooltip="Opens the sheet sliced on its own grid, with one tag per "
        "animation and direction. It opens unlinked: the first Ctrl+S is a "
        "Save As, so cleaning up frames cannot overwrite the render they came "
        "from.",
    ):
        troupe_mode.open_in_inker(ctx)
    imgui.dummy((0, sp(tokens.SP_1)))
    if widgets.disabled_button(
        verbs.add_to("packwright"),
        ready,
        (-1, 0),
        reason="Pick a character sheet first.",
        tooltip="One sprite per cell, packed beside everything else in the "
        "atlas.",
    ):
        troupe_mode.add_to_packwright(ctx)
    imgui.dummy((0, sp(tokens.SP_1)))
    # **The third way out, and the first of two that produce files.** The two
    # above hand the sheet to another mode; this one is for the engine, and it
    # copies the *pair* -- the PNG is the atlas and the JSON is what says which
    # cell is ``walk`` facing south-east, so a folder with one and not the
    # other holds an asset nothing can interpret.
    busy = ctx.busy(troupe_mode.export_key(state.job_id, state.sheet_id))
    if widgets.disabled_button(
        "Export package...",
        ready and not busy,
        (-1, 0),
        reason=(
            "That sheet is already being exported."
            if busy
            else "Pick a character sheet first."
        ),
        tooltip="Copies the PNG and its JSON sidecar together -- the pair an "
        "engine imports. Asks where to put them unless an export folder is "
        "configured.",
    ):
        troupe_mode.export_package(ctx)

    imgui.dummy((0, sp(tokens.SP_1)))
    # A fourth way out, beside the package export: an engine that wants
    # ``AnimatedSprite2D``-style frame folders rather than an atlas-plus-
    # sidecar pair gets one PNG per frame instead. Same enable/disable rule
    # and the same per-sheet busy key one door over -- a second press while
    # one is in flight is refused rather than raced.
    frames_busy = ctx.busy(troupe_mode.frames_key(state.job_id, state.sheet_id))
    if widgets.disabled_button(
        "Export frames...",
        ready and not frames_busy,
        (-1, 0),
        reason=(
            "That sheet is already being exported."
            if frames_busy
            else "Pick a character sheet first."
        ),
        tooltip="Writes one PNG per frame into clip and compass-direction "
        "folders (N, NE, E...) with a manifest.json of frame rates and "
        "loops. Asks where to put them unless an export folder is "
        "configured.",
    ):
        troupe_mode.export_frames(ctx)

    imgui.dummy((0, sp(tokens.SP_2)))
    widgets.muted_wrapped(
        "The sheet and its sidecar are already on disk beside the mesh. The "
        "Library's export list is where the files themselves are."
    )
