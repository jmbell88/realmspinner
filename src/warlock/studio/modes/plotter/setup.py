"""What a new map is, asked before the map exists.

Plotter used to answer "New map" by making a 32x32 grid of 32px cells,
orthogonal, and saying so nowhere. Two of those four numbers cannot be taken
back by any amount of editing afterwards -- **projection** is fixed the moment
anything is painted (a set drawn for one lattice paints the wrong shape into
every cell drawn for the other), and the **tile size** is what a plain image is
sliced at when it is added, so a tileset added to a 32px map is a 32px tileset
forever. A default nobody was asked about is a poor place to put a decision that
permanent, which is what this module exists to move.

Pure, and deliberately: no imgui, no ``ctx``. It owns *what the numbers mean* --
the presets, the caps, and what Create does with the answer -- so the popup that
collects them is a body that draws fields, and the tests do not need a window.
The pane half lives in :mod:`.panes.plotter_canvas`, registered per pane for
the reason ``inker_canvas.new_canvas_popup`` documents: a popup belongs to the
window that begins it.
"""

from __future__ import annotations

from typing import Any

from .engine import project

#: What a hexagonal preset's flat run is, when the form has not been told
#: otherwise. Half the tile is the regular hexagon, which is the shape somebody
#: choosing "hexagonal" almost always means. Defined before ``PRESETS`` because
#: the Hexagonal row below is written in terms of it.
DEFAULT_HEX_SIDE = 16

#: The starting points, as ``(label, width, height, tile_w, tile_h, projection,
#: hex_side)``. Sizes in tiles, tile sizes in pixels. Three rather than a page
#: of them: a preset list is a way of *not* answering the question, and the
#: useful answers are "the two common cell sizes" and "the other lattice".
PRESETS: tuple[tuple[str, int, int, int, int, str, int], ...] = (
    ("Small, 16 px tiles", 40, 30, 16, 16, project.ORTHOGONAL, 0),
    ("Standard, 32 px tiles", 32, 32, 32, 32, project.ORTHOGONAL, 0),
    # 2:1 is the isometric convention, and the generator already warns when a
    # map departs from it -- so the preset that exists to be correct is 2:1.
    ("Isometric, 64 x 32", 32, 32, 64, 32, project.ISOMETRIC, 0),
    # The offset lattices, one each. A staggered map is the hexagonal one with
    # no flat run, so the two presets differ by exactly the hex side -- which is
    # the clearest way to say what the relationship between them is. Staggered
    # is hex_side=0, deliberately: project.py's own Lattice docstring says a
    # zero flat run is what makes a hexagonal lattice read as staggered, so
    # this is the one preset that must stay at it.
    ("Staggered, 32 px", 32, 32, 32, 32, project.STAGGERED, 0),
    # The 2026-09-08 audit, plotter-02: this row used to leave hex_side at the
    # ``MapDoc`` default of 0, which is the *staggered* lattice by the same
    # docstring -- so "Hexagonal" drew exactly like "Staggered" until a user
    # separately found Map > Map properties > Hex side and set it by hand.
    ("Hexagonal, 32 px", 32, 32, 32, 32, project.HEXAGONAL, DEFAULT_HEX_SIDE),
)

#: The preset Create starts on.
DEFAULT = PRESETS[1]

#: What a *typed* number may do. The fields accept free text, so one stray digit
#: turns 32 into 320 -- and unlike Inker's canvas, a map multiplies: 512 tiles
#: square at 32 px is the 268-megapixel composite ``plotter_mode.export_library``
#: already warns about. The tile cap matches, for the same arithmetic read the
#: other way. Both are limits on a slip of the keyboard, not on the engine.
MAX_TILES = 512
MAX_TILE_PX = 512

#: What to do about a tileset once the map exists. The map is unpaintable until
#: it has one, so the dialog offers the two doors rather than leaving the user
#: to find them -- these are the keys, and ``panes.plotter_canvas`` routes them.
NEXT_EMPTY = "empty"
NEXT_FILE = "file"


def blank_form() -> dict[str, Any]:
    """The dialog's state, on :attr:`DEFAULT`."""
    label, width, height, tile_w, tile_h, projection, hex_side = DEFAULT
    return {
        "width": width,
        "height": height,
        "tile_w": tile_w,
        "tile_h": tile_h,
        "projection": projection,
        "hex_side": hex_side,
        "next": NEXT_FILE,
        "preset": label,
        # Off by default: a fixed rectangle is what most maps want, it is what
        # every preset above describes, and an infinite map's size fields say
        # something different from what they say here.
        "infinite": False,
    }


def apply_preset(form: dict[str, Any], label: str) -> dict[str, Any]:
    """``form`` moved onto the named preset, in place. Unknown labels are kept
    as a custom entry rather than refused: the label is only a note about where
    the numbers came from, and the numbers are the answer."""
    for name, width, height, tile_w, tile_h, projection, hex_side in PRESETS:
        if name == label:
            form.update(
                width=width,
                height=height,
                tile_w=tile_w,
                tile_h=tile_h,
                projection=projection,
                hex_side=hex_side,
                preset=name,
            )
            break
    return form


def clamp(form: dict[str, Any]) -> dict[str, Any]:
    """Every number brought inside its cap, in place.

    Clamped on the way *into* the fields as well as on the way out, the rule
    ``inker_canvas``'s new-canvas popup states: a box that goes on showing a
    number the Create button will not honour is a box that lies.
    """
    form["width"] = max(1, min(int(form.get("width", 1) or 1), MAX_TILES))
    form["height"] = max(1, min(int(form.get("height", 1) or 1), MAX_TILES))
    form["tile_w"] = max(1, min(int(form.get("tile_w", 1) or 1), MAX_TILE_PX))
    form["tile_h"] = max(1, min(int(form.get("tile_h", 1) or 1), MAX_TILE_PX))
    if form.get("projection") not in project.PROJECTIONS:
        form["projection"] = project.ORTHOGONAL
    form["infinite"] = bool(form.get("infinite", False))
    # Same cap as the tile sizes: hex_side is a pixel run inserted into the
    # same cell, so anything ``tile_w``/``tile_h`` may not exceed neither may
    # it.
    form["hex_side"] = max(0, min(int(form.get("hex_side", 0) or 0), MAX_TILE_PX))
    return form


def size_of(form: dict[str, Any]) -> tuple[int, int, int, int]:
    """The ``new_document`` size tuple this form describes."""
    clamp(form)
    return (int(form["width"]), int(form["height"]), int(form["tile_w"]), int(form["tile_h"]))


def summary(form: dict[str, Any]) -> str:
    """One line saying what Create will make, in both units.

    Both, because the two numbers that matter are in different ones: a map is
    authored in tiles and *exported* in pixels, and 512 square at 32 px is the
    difference between a sentence and a surprise.
    """
    width, height, tile_w, tile_h = size_of(form)
    if form.get("infinite"):
        # The numbers still mean something -- they are the starting window --
        # but "overall" does not, so the sentence says what they are instead of
        # multiplying out a total the map does not have.
        return (
            f"Infinite, starting at {width} x {height} tiles of "
            f"{tile_w} x {tile_h} px. It grows as you paint past the edge."
        )
    return (
        f"{width} x {height} tiles at {tile_w} x {tile_h} px "
        f"-- {width * tile_w} x {height * tile_h} px overall"
    )


def isometric_warning(form: dict[str, Any]) -> str:
    """The 2:1 note, or "" when there is nothing to say.

    The generator's own wording and the same rule; repeated here because this
    dialog is now the first place a projection is chosen, and it would be the
    one place that let the mistake through silently.
    """
    clamp(form)
    if form["projection"] != project.ISOMETRIC:
        return ""
    if form["tile_w"] == form["tile_h"] * 2:
        return ""
    return (
        f"An isometric cell is conventionally 2:1 -- {form['tile_h'] * 2} x {form['tile_h']} "
        f"rather than {form['tile_w']} x {form['tile_h']}. It will still be created."
    )
