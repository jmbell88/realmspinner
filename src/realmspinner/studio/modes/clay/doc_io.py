"""What the Document tab says about a document, and what it may send out.

The facts, the import choices and the export sentences that the old bridge pane
held when the right column had a third pane for them. The pane is gone (the
column is the outliner over the Inspector now, and these are drawn by the
Inspector's Document tab); the **data and the pure functions** came here, to the
mode root, because they are not drawing -- they are claims a test can check
without a frame (a count is a number, an export refusal is a sentence), and the
Document tab, the File menu and the tests all read them from one place.

No imgui, no GL: ``tests/test_layering.py`` keeps the mode root free of both.
"""

from __future__ import annotations

from typing import Any

from ....kernels.geom3d import units

#: Scale choices for "Import Mesh...", key is the multiplier ``import_file``
#: takes -- a unit a modeller actually authors in, never a bare number a user
#: would have to already know the conversion for.
#: Keys are ``f"{value:g}"`` of the multiplier itself -- what the Document tab
#: formats ``state.import_scale`` as to look the option up, so the two can
#: never drift into two different spellings of the same number.
IMPORT_SCALE_OPTIONS = units.scale_options()
IMPORT_UP_OPTIONS = (("y", "Y up"), ("z", "Z up"))

#: The Export OBJ tooltip. The 2026-10-07 audit's clay-77: it named the ``.mtl``
#: and not the PNG a textured material writes beside it (``<name>_<slot>.png``), so
#: a user who exported a textured model found files the button never mentioned.
EXPORT_OBJ_TOOLTIP = (
    "Saves the document as a plain mesh file on disk, for handing straight "
    "to another tool. The library never sees it. OBJ writes a .mtl of the same "
    "name beside it, and a PNG next to that for each textured material."
)

EXPORT_GLB_TOOLTIP = (
    "Saves the document as a plain mesh file on disk, for handing straight "
    "to another tool. The library never sees it."
)

EXPORT_LIBRARY_TOOLTIP = (
    "The exact geometry, as an ordinary asset. It picks up rigging, posing, "
    "sprite sheets, the triangle retarget and every mesh export, because all "
    "of those are functions of model.glb."
)

SCREENSHOT_TOOLTIP = (
    "Saves a lit PNG of the document from the angle you are looking at it, "
    "without the grid or gizmos. Position objects first, then frame the view."
)


def triangles(mesh: Any) -> int:
    """Counted the way the renderer fans them, so the number matches the
    exported file rather than the face count the outliner would give.

    Off array *lengths* alone, the way ``viewport_hints.stats()`` derives its
    own triangle count -- ``corners - 2*faces`` is exactly
    ``sum(max(n-2, 0))`` over every face's corner count ``n``, because
    ``validate`` guarantees every face has at least three corners, so the
    ``max(n-2, 0)`` floor never bites. The 2026-09-18 audit's clay-02 found
    this scanning the whole face array with ``np.diff``/``np.maximum``/``.sum``
    on every draw of the Document panel -- ~8 ms per call for a 1,000,000-face
    object, paid every imgui frame the panel is visible.
    """
    faces = max(0, len(mesh.starts) - 1)
    corners = len(mesh.loops)
    return max(0, corners - 2 * faces)


def objects_line(doc: Any) -> str:
    """``"3 of 4 objects visible  -  2 materials"``.

    The counts are of what leaves the document -- visible objects only, the same
    set the exporters write.
    """
    visible = sum(1 for obj in doc.objects if obj.visible)
    return f"{visible} of {len(doc.objects)} objects visible  -  {len(doc.materials)} materials"


def geometry_line(doc: Any) -> str:
    """``"1,024 vertices  -  1,022 faces  -  2,040 triangles"``, of the visible objects."""
    visible = [obj for obj in doc.objects if obj.visible]
    return (
        f"{sum(len(o.mesh.positions) for o in visible):,} vertices  -  "
        f"{sum(max(0, len(o.mesh.starts) - 1) for o in visible):,} faces  -  "
        f"{sum(triangles(o.mesh) for o in visible):,} triangles"
    )


def outputs_why(doc: Any, saving: bool) -> str:
    """Why the output buttons are refused right now, or ``""`` when they are not.

    One sentence for every output button, because they are refused for the
    same two reasons and a user reading two different explanations of one state
    would look for two different problems. A plain function of a document and a
    bool so it can be asserted without imgui -- panes cannot be driven
    headlessly, but the sentence a button greys with can still be a pure answer.
    """
    if saving:
        return "Saving..."
    if not any(obj.visible for obj in doc.objects):
        return "Nothing visible to send -- every object is hidden."
    return ""
