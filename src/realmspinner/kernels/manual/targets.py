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
    # happens to a creature Realmspinner does not model) are all in one place.
    "settings-character": ("22-generating-references", "characters"),
    # Create's History pane: every attempt made or judged, deleted ones included,
    # with Load settings and the generation-log export.
    "create-history": ("22-generating-references", "history"),
    "settings-3d": ("23-generating-meshes", None),
    # The Rig stage's own column (the UI redesign, wave 5). Rigging was three
    # buttons in three places and no pane of its own, so it had no (?) either.
    "settings-rig": ("23-generating-meshes", "rigging-a-mesh"),
    "library": ("33-library-and-jobs", None),
    "inspector": ("23-generating-meshes", "exports"),
    "retarget": ("23-generating-meshes", "triangle-budget"),
    "reorient": ("23-generating-meshes", "fixing-a-finished-asset"),
    "retexture": ("23-generating-meshes", "surface-texture"),
    "remesh": ("23-generating-meshes", "game-ready-remesh"),
    # The Earlier meshes panel (2026-09-22): what a retarget, a remesh
    # or a re-texture replaces is kept rather than discarded, and the
    # questions asked in front of this panel -- how many are kept, what
    # restoring one does to the rig -- are answered in its own section.
    "mesh-history": ("23-generating-meshes", "earlier-meshes"),
    "loras": ("42-app-settings", "your-style-loras"),
    "pose": ("23-generating-meshes", "posing"),
    "sprites": ("22-generating-references", "from-a-single-drawing"),
    "inker-timeline": ("26-inker-animation", "the-timeline"),
    "inker-flourish": ("26-inker-animation", "effects"),
    # Found by the O118 coverage sweep: two panes a user reads and neither had
    # a way into the chapter that describes it.
    "inker-colors": ("25-inker", "colour"),
    # The tile panel (Wave 3). Its own anchor rather than sharing
    # ``inker-tools``: a tilemap layer is a different *kind* of layer, and the
    # questions asked in front of this panel -- what Manual/Auto/Stack do, what
    # a flag bit is, why an old build drops the tiles -- are all in that one
    # section.
    "inker-tiles": ("25-inker", "tilemap-layers"),
    "inker-preview": ("26-inker-animation", "preview"),
    # The walk-cycle setup panel, present only while a session is open. Its own
    # anchor rather than sharing ``inker-timeline``: the questions asked in front
    # of it -- what a near limb is, why the stride slider stops where it does,
    # what happens to the drawing it was cut out of -- are all in one section.
    "inker-walk": ("26-inker-animation", "a-walk-cycle-from-a-still-drawing"),
    # The toolbox. It had no entry for as long as it was a 90 px rail with no
    # room for a heading to hang a (?) beside; it is a sidebar pane now, so it
    # points at the section that lists every tool and its letter.
    "inker-tools": ("25-inker", "tools"),
    # The slider surface under the palette. Its own anchor rather than sharing
    # ``inker-colors``: the questions asked in front of it -- which of the two
    # colours am I editing, why do the sliders move a palette entry in an
    # indexed document, what does the hex field accept -- are all in that one
    # section.
    "inker-picker": ("25-inker", "the-colour-picker"),
    # The four ways a drawing leaves the Inker. The bridges section is where
    # the *directions* are explained, which is what somebody standing in front
    # of a greyed "Revert to original" is asking about.
    "inker-generate": ("25-inker", "pipeline-bridges"),
    "candidates": ("23-generating-meshes", "attempts"),
    # The viewport toolbar. The ~5k-LOC subsystem in the middle of the window
    # was chrome as far as this map was concerned -- exempted in
    # ``tests/manual/test_coverage.py`` for having "no titled section" -- while
    # being the one thing on screen at every stage of Create and in Review.
    "overlay": ("24-the-3d-viewport", "the-toolbar"),
    "clay-header": ("27-clay", "the-viewport-header"),
    "clay-tools": ("27-clay", "adding-a-primitive"),
    # The Inspector that took the ``clay-props`` slot (that id is retired and never
    # reused: a saved layout may have hidden it, and ``studio/skeletons.py`` says so).
    # Its own anchor, pointing at the section that describes the whole panel; Colours
    # & textures and Unwrap are pages of it, and the (?) on Unwrap is ``clay-uv``'s.
    "clay-inspector": ("27-clay", "the-inspector"),
    # The (?) on the Inspector's docked "All shapes" header. The outliner is a
    # section of that pane now, not a pane of its own, but the questions asked in
    # front of it -- what a drag onto a row does, how a range is picked -- are the
    # outliner section's, not the Inspector's.
    "clay-outliner": ("27-clay", "the-outliner"),
    # Tranche 6 ("UV and materials"). Its own anchor rather than sharing
    # the Inspector's: the questions asked in front of this panel -- what the
    # overlap/stretch tint means, what dragging a box does, what Pack Islands
    # and Texel Density actually set -- are answered in one section, "The UV
    # view" under "Texture coordinates".
    "clay-uv": ("27-clay", "the-uv-view"),
    # The timeline strip under the viewport in Pose mode.
    "clay-timeline": ("27-clay", "the-timeline"),
    # Mason's seven, pointed at its own chapter. They were interim from Stage E
    # until ``28-mason.md`` landed: Part II was full at 20-38, so giving Mason a
    # slot was a fifteen-file renumbering, and until that was taken these rows
    # named the nearest existing anchor that genuinely covered the material --
    # Clay's modelling chapter for the shared ideas and the 3D viewport
    # chapter's toolbar section for the header strip. ``mason-prefabs`` was the
    # one whose real subject nothing covered at all, and it now has a section of
    # its own. Every row here names a heading the chapter actually has, and
    # ``tests/manual/test_docs.py`` checks that in both directions.
    "mason-header": ("28-mason", "the-viewport-header"),
    "mason-assets": ("28-mason", "the-assets-panel"),
    "mason-tools": ("28-mason", "transforming"),
    "mason-outliner": ("28-mason", "the-outliner"),
    "mason-props": ("28-mason", "properties"),
    "mason-prefabs": ("28-mason", "prefabs-and-instances"),
    "mason-bridge": ("28-mason", "the-three-ways-out"),
    "plotter-tools": ("29-plotter", "tools"),
    # The sheet over the centre pane: three titled tabs a user interacts with,
    # so an exemption would be false.
    "plotter-tileset-editor": ("29-plotter", "tilesets"),
    "plotter-tileset": ("29-plotter", "tilesets"),
    "inker-image-size": ("25-inker", "image-size"),
    "inker-canvas-size": ("25-inker", "canvas-size"),
    "plotter-layers": ("29-plotter", "layers"),
    "plotter-objects": ("29-plotter", "objects"),
    "plotter-stamps": ("29-plotter", "tile-stamps"),
    "plotter-properties": ("29-plotter", "layer-and-map-properties"),
    "plotter-bridge": ("29-plotter", "files"),
    "packwright-sources": ("30-packwright", "sources"),
    "packwright-settings": ("30-packwright", "settings"),
    "packwright-items": ("30-packwright", "when-it-does-not-fit"),
    "packwright-bridge": ("30-packwright", "exporting"),
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
    "sirens-transport": ("31-sirens", "playing-it"),
    "sirens-orders": ("31-sirens", "patterns-and-the-order"),
    "sirens-instruments": ("31-sirens", "instruments"),
    "sirens-envelopes": ("31-sirens", "the-envelope-editor"),
    "sirens-effects": ("31-sirens", "sound-effects"),
    "sirens-bridge": ("31-sirens", "exporting-the-audio"),
    # Muse's two targets. The recipe column carries the first, because that is
    # the pane whose controls a reader has a question about; the brief is a bar
    # and carries none, exactly as ``create_brief`` does -- a (?) in a one-row
    # command bar competes with the button the bar exists for. The player is the
    # second: a pane with its own ``help_button`` (``ui/panes/player.py``), whose
    # question (how to audition and keep a render) is not the recipe's. The
    # results tray is exempt rather than targeted: it is the surface the mode is
    # *about*, the way the pattern grid and the two canvases are.
    "muse-recipe": ("32-muse", "the-window"),
    "muse-player": ("32-muse", "the-player"),
    "review": ("34-review", None),
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
