"""Context help: which manual chapter a pane's (?) opens.

Pure data, importable headlessly -- the docs integrity test validates every
entry against the real chapters and anchors.
"""

from __future__ import annotations

HELP_TARGETS: dict[str, tuple[str, str | None]] = {
    "settings-2d": ("22-generating-references", None),
    # The Sheet output's own block. Its own entry rather than sharing
    # ``settings-2d``: the chapter is long and the two settings it explains --
    # why the grid is not a control, and why the sprite arm makes two rows --
    # are the questions a user has while looking at that section.
    "settings-sheet": ("22-generating-references", "sheets"),
    # The Character type's own column. Its own entry rather than sharing
    # ``settings-2d``: nothing in that section applies -- there is no model, no
    # LoRA and no seed-per-candidate -- and the questions asked in front of this
    # block (why the species picker is empty, what "no GPU needed" means, what
    # happens to a creature Warlock does not model) are all in one place.
    "settings-character": ("22-generating-references", "characters"),
    "settings-3d": ("23-generating-meshes", None),
    # The Rig stage's own column (the UI redesign, wave 5). Rigging was three
    # buttons in three places and no pane of its own, so it had no (?) either.
    "settings-rig": ("25-rigging-and-posing", "rigging-a-mesh"),
    "library": ("36-library-and-jobs", None),
    "inspector": ("23-generating-meshes", "exports"),
    "retarget": ("23-generating-meshes", "triangle-budget"),
    "retexture": ("23-generating-meshes", "surface-texture"),
    "remesh": ("23-generating-meshes", "game-ready-remesh"),
    "loras": ("42-app-settings", "your-style-loras"),
    "pose": ("25-rigging-and-posing", "posing"),
    "poser-library": ("26-poser", "the-pose-library"),
    "poser-controls": ("26-poser", "posing-a-skeleton"),
    "poser-clips": ("26-poser", "editing-clips"),
    # The skeleton editor's own section (P6/P8, 2026-09-13): reachable only
    # from an open asset session, and its questions -- what a pivot vs. a
    # limb removal does, what a re-rig does to poses and clips, why choosing
    # another template discards it -- are answered nowhere else.
    "poser-skeleton": ("26-poser", "editing-the-skeleton"),
    "sheet": ("27-sprite-sheets", None),
    "sprites": ("27-sprite-sheets", "from-a-single-drawing"),
    "inker-timeline": ("29-inker-animation", "the-timeline"),
    # The sheet-correction strip under the transport, which exists only on a
    # Poser character sheet and is the phase-6 cleanup loop.
    "inker-sheet": ("29-inker-animation", "sheet-corrections"),
    "inker-flourish": ("29-inker-animation", "effects"),
    # Found by the O118 coverage sweep: two panes a user reads and neither had
    # a way into the chapter that describes it.
    "inker-colors": ("28-inker", "colour"),
    # The tile panel (Wave 3). Its own anchor rather than sharing
    # ``inker-tools``: a tilemap layer is a different *kind* of layer, and the
    # questions asked in front of this panel -- what Manual/Auto/Stack do, what
    # a flag bit is, why an old build drops the tiles -- are all in that one
    # section.
    "inker-tiles": ("28-inker", "tilemap-layers"),
    "inker-preview": ("29-inker-animation", "preview"),
    # The walk-cycle setup panel, present only while a session is open. Its own
    # anchor rather than sharing ``inker-timeline``: the questions asked in front
    # of it -- what a near limb is, why the stride slider stops where it does,
    # what happens to the drawing it was cut out of -- are all in one section.
    "inker-walk": ("29-inker-animation", "a-walk-cycle-from-a-still-drawing"),
    # The toolbox. It had no entry for as long as it was a 90 px rail with no
    # room for a heading to hang a (?) beside; it is a sidebar pane now, so it
    # points at the section that lists every tool and its letter.
    "inker-tools": ("28-inker", "tools"),
    # The slider surface under the palette. Its own anchor rather than sharing
    # ``inker-colors``: the questions asked in front of it -- which of the two
    # colours am I editing, why do the sliders move a palette entry in an
    # indexed document, what does the hex field accept -- are all in that one
    # section.
    "inker-picker": ("28-inker", "the-colour-picker"),
    # The four ways a drawing leaves the Inker. The bridges section is where
    # the *directions* are explained, which is what somebody standing in front
    # of a greyed "Revert to original" is asking about.
    "inker-generate": ("28-inker", "pipeline-bridges"),
    "candidates": ("23-generating-meshes", "candidates"),
    # The viewport toolbar. The ~5k-LOC subsystem in the middle of the window
    # was chrome as far as this map was concerned -- exempted in
    # ``tests/manual/test_coverage.py`` for having "no titled section" -- while
    # being the one thing on screen at every stage of Create and in Review.
    "overlay": ("24-the-3d-viewport", "the-toolbar"),
    "clay-header": ("30-clay", "the-viewport-header"),
    "clay-tools": ("30-clay", "adding-a-primitive"),
    "clay-props": ("30-clay", "materials"),
    "clay-outliner": ("30-clay", "adding-a-primitive"),
    # Tranche 6 ("UV and materials"). Its own anchor rather than sharing
    # ``clay-props``: the questions asked in front of this panel -- what the
    # overlap/stretch tint means, what dragging a box does, what Pack Islands
    # and Texel Density actually set -- are answered in one section, "The UV
    # view" under "Texture coordinates".
    "clay-uv": ("30-clay", "the-uv-view"),
    "clay-bridge": ("30-clay", "the-two-ways-out"),
    # Mason's seven, pointed at its own chapter. They were interim from Stage E
    # until ``31-mason.md`` landed: Part II was full at 20-38, so giving Mason a
    # slot was a fifteen-file renumbering, and until that was taken these rows
    # named the nearest existing anchor that genuinely covered the material --
    # Clay's modelling chapter for the shared ideas and the 3D viewport
    # chapter's toolbar section for the header strip. ``mason-prefabs`` was the
    # one whose real subject nothing covered at all, and it now has a section of
    # its own. Every row here names a heading the chapter actually has, and
    # ``tests/manual/test_docs.py`` checks that in both directions.
    "mason-header": ("31-mason", "the-viewport-header"),
    "mason-assets": ("31-mason", "the-assets-panel"),
    "mason-tools": ("31-mason", "transforming"),
    "mason-outliner": ("31-mason", "the-outliner"),
    "mason-props": ("31-mason", "properties"),
    "mason-prefabs": ("31-mason", "prefabs-and-instances"),
    "mason-bridge": ("31-mason", "the-three-ways-out"),
    "plotter-tools": ("32-plotter", "tools"),
    # The sheet over the centre pane: three titled tabs a user interacts with,
    # so an exemption would be false.
    "plotter-tileset-editor": ("32-plotter", "tilesets"),
    "plotter-tileset": ("32-plotter", "tilesets"),
    "inker-image-size": ("28-inker", "image-size"),
    "inker-canvas-size": ("28-inker", "canvas-size"),
    "plotter-layers": ("32-plotter", "layers"),
    "plotter-objects": ("32-plotter", "objects"),
    "plotter-stamps": ("32-plotter", "tile-stamps"),
    "plotter-properties": ("32-plotter", "layer-and-map-properties"),
    "plotter-bridge": ("32-plotter", "files"),
    "packwright-sources": ("33-packwright", "sources"),
    "packwright-settings": ("33-packwright", "settings"),
    "packwright-items": ("33-packwright", "when-it-does-not-fit"),
    "packwright-bridge": ("33-packwright", "exporting"),
    # Sirens' six panes, pointed at the chapter that now exists. Every one of
    # them sat on ``20-overview#the-modes`` between phases 2 and 5 -- a
    # placeholder rather than a missing button, because a (?) that appears
    # later is one users learn to look for later. Six entries rather than one
    # chapter-wide target for Troupe's reason: the question differs per pane.
    # What a render is and why Play is greyed is the transport's; what an order
    # list is *for* is the order panel's; which number a cell holds is the
    # instrument list's; what a release point splits is the envelope editor's;
    # why an effect keeps its own tempo is the effects panel's; and what a
    # folder of WAVs contains is the bridge's.
    "sirens-transport": ("34-sirens", "playing-it"),
    "sirens-orders": ("34-sirens", "patterns-and-the-order"),
    "sirens-instruments": ("34-sirens", "instruments"),
    "sirens-envelopes": ("34-sirens", "the-envelope-editor"),
    "sirens-effects": ("34-sirens", "sound-effects"),
    "sirens-bridge": ("34-sirens", "exporting-the-audio"),
    # Muse's one target. The recipe column carries it, because that is the pane
    # whose controls a reader has a question about; the brief is a bar and
    # carries none, exactly as ``create_brief`` does -- a (?) in a one-row
    # command bar competes with the button the bar exists for. The results tray
    # is exempt rather than targeted: it is the surface the mode is *about*, the
    # way the pattern grid and the two canvases are.
    "muse-recipe": ("35-muse", "the-window"),
    "muse-player": ("35-muse", "the-player"),
    # Poser's character-sheet section (Troupe's own four panes, folded in by
    # P9, 2026-09-18, and 26-poser.md rewritten to cover them the same day).
    # Per-pane again now that the chapter has its own anchors for each: the
    # questions differ per pane (what a sheet *is* and how to start one, what
    # the preview and heatmap mean, what a stray-pixel count or Needs repair
    # means, where a finished sheet goes).
    "poser-sheets": ("26-poser", "rendering-a-character-sheet"),
    "poser-new-character": ("26-poser", "starting-a-new-character"),
    "poser-sheet-preview": ("26-poser", "watching-the-sheet"),
    "poser-sheet-info": ("26-poser", "the-sheet-panel"),
    "poser-sheet-bridge": ("26-poser", "taking-a-sheet-somewhere"),
    "review": ("37-review", None),
    "app-settings": ("42-app-settings", None),
    # The chooser the app opens on (F56/O118): the one pane a first run
    # certainly sees, and the only one that had no way into the manual at all.
    "home": ("21-home", None),
}

# Where "something is wrong and I do not know what" goes.
#
# Deliberately *not* a HELP_TARGETS entry, though it is the same shape. That
# dict is the pane-(?)-button map and is asserted against the call sites in both
# directions (``test_help_button_call_sites_match_help_targets``) precisely so a
# dead button or dead data fails a test -- and the three surfaces that lead here
# are a red banner, a popup and a Home row, none of which is a pane with a (?).
# Named once rather than spelled at each of the three, so a chapter that moved
# does not have to be found in three places (F57).
TROUBLESHOOTING: tuple[str, str | None] = ("43-troubleshooting", None)
