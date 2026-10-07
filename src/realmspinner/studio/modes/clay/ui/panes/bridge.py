"""What the document is, and the ways out of it.

The counts and the save state on top, the export buttons underneath -- the
same shape the raster editor's bridge takes, and for the same reason: a panel
that offers to send something somewhere should first say what it is going to
send. Export puts the *exact* geometry in the library as an ordinary asset, or
writes it to a file on disk.

Every button is disabled while a save is in flight, for the reason the tool
panel states.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ......kernels.geom3d import units
from ..... import icons, tokens, verbs, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import mode as clay_mode

#: What this pane refuses to shrink past, in design pixels: the path line, the
#: undo pair, the step count and the two ways out.
#:
#: It had none while the right column was a hand-composed pair. The declarative
#: layout stacks the outliner and the properties above it, and
#: ``layout_skeleton.heights`` gives each share its proportion of the room
#: before the fill sees any -- so two at the default 0.5 leave this pane exactly
#: zero pixels and the Document panel is not on screen at all. Plotter's map
#: file panel went the same way on the same day and for the same reason.
BRIDGE_FLOOR = 170.0


def draw(ctx: Any) -> None:
    state = clay_mode.ensure(ctx)
    tab = state.active
    # "Model file", the shape of every other bridge's heading ("Drawing file",
    # "Map file", "Song file", "Atlas file"); this one said "Document".
    widgets.section("Model file")
    manual_render.help_button(ctx, "clay-bridge")
    if tab is None:
        # The recent list and nothing else -- Plotter's bridge exactly (B5).
        # New/Open are on the empty canvas two columns to the left, and drawing
        # them here as well was one pair of buttons in two places; the *list*
        # is the opposite case, because the moment it matters most is the
        # moment there is nothing open.
        _recent(ctx)
        return
    _files(ctx, tab)

    _facts(tab)
    imgui.dummy((0, sp(tokens.SP_2)))
    _history(ctx, tab)
    imgui.dummy((0, sp(tokens.SP_2)))
    _outputs(ctx, tab)
    _recent(ctx)


def _history(ctx: Any, tab: Any) -> None:
    """Undo and Redo, on screen.

    This mode had a full undo stack and no visible control for it, so the
    feature existed only for a user who already knew Ctrl+Z -- while Inker drew
    the same pair twice. ``clay_mode.undo``/``redo`` rather than
    ``tab.doc.undo()`` here, so the button and the chord carry the same side
    effects (see the history block in that module).
    """
    widgets.history_block(
        ctx,
        tab,
        key="clay",
        undo=lambda: clay_mode.undo(ctx, tab),
        redo=lambda: clay_mode.redo(ctx, tab),
        step=lambda index: clay_mode.step_history(ctx, tab, index),
    )


def _facts(tab: Any) -> None:
    doc = tab.doc
    visible = [obj for obj in doc.objects if obj.visible]
    triangles = sum(_triangles(obj.mesh) for obj in visible)
    widgets.muted(
        f"{len(visible)} of {len(doc.objects)} objects visible  -  "
        f"{triangles:,} triangles  -  {len(doc.materials)} materials"
    )


def _triangles(mesh: Any) -> int:
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


def _files(ctx: Any, tab: Any) -> None:
    """The file row for an *open* document -- the shared header, so it is the
    same four buttons and the same status ladder every other workspace has.
    ``draw`` returns before this when there is none, because New and Open also
    belong to the empty canvas."""
    widgets.document_header(
        tab,
        new=lambda: clay_mode.new_document(ctx),
        open_=lambda: clay_mode.ask_open(ctx),
        save=lambda: clay_mode.save(ctx, tab),
        save_as=lambda: clay_mode.save_as(ctx, tab),
    )


#: Scale choices for "Import Mesh...", key is the multiplier ``import_file``
#: takes -- a unit a modeller actually authors in, never a bare number a user
#: would have to already know the conversion for.
#: Keys are ``f"{value:g}"`` of the multiplier itself -- what :func:`_import_mesh`
#: formats ``state.import_scale`` as to look the option up, so the two can
#: never drift into two different spellings of the same number.
IMPORT_SCALE_OPTIONS = units.scale_options()
IMPORT_UP_OPTIONS = (("y", "Y up"), ("z", "Z up"))


def import_settings(ctx: Any) -> None:
    """The units and up-axis the next mesh import uses, beside nothing.

    ``ClayState.import_scale``/``import_up`` are what the Add menu's "Import
    Mesh..." row and a file dropped on the viewport both read
    (``clay_mode.import_mesh_path``'s own defaults) -- set here, in the
    Properties pane's Document tab, and remembered for the next import in either
    form. The row that *runs* the import moved to the menu strip with the rest
    of the verbs; these two are settings, and the Document tab is where the
    document-wide settings are.
    """
    state = clay_mode.ensure(ctx)
    widgets.field_label("import units")
    scale_key = f"{state.import_scale:g}"
    picked = widgets.combo("##clay-import-scale", scale_key, IMPORT_SCALE_OPTIONS, sp(90))
    if picked != scale_key:
        state.import_scale = float(picked)
    widgets.field_label("import up axis")
    state.import_up = widgets.combo("##clay-import-up", state.import_up, IMPORT_UP_OPTIONS, sp(90))


#: The Export OBJ tooltip. The 2026-10-07 audit's clay-77: it named the ``.mtl``
#: and not the PNG a textured material writes beside it (``<name>_<slot>.png``), so
#: a user who exported a textured model found files the button never mentioned.
EXPORT_OBJ_TOOLTIP = (
    "Saves the document as a plain mesh file on disk, for handing straight "
    "to another tool. The library never sees it. OBJ writes a .mtl of the same "
    "name beside it, and a PNG next to that for each textured material."
)


def _outputs_why(doc: Any, saving: bool) -> str:
    """Why the output buttons below are refused right now, or ``""`` when
    they are not.

    One sentence for every output button, because they are refused for the
    same two reasons and a user reading two different explanations of one state
    would look for two different problems. Pulled out as its own function so it
    can be asserted without imgui -- panes cannot be driven headlessly, but the
    sentence a button greys with can still be a plain function of a document
    and a bool.
    """
    if saving:
        return "Saving..."
    if not any(obj.visible for obj in doc.objects):
        return "Nothing visible to send -- every object is hidden."
    return ""


def _outputs(ctx: Any, tab: Any) -> None:
    # The one heading every mode's exits are under. See ``inker_bridge``'s
    # ``_pipeline`` for why the five of them agree on a name.
    widgets.section("Take it somewhere")
    doc = tab.doc
    why = _outputs_why(doc, tab.saving)
    ready = not why

    if widgets.primary_button(
        f"{icons.DOWNLOAD} {verbs.EXPORT_TO_LIBRARY}", enabled=ready, reason=why
    ):
        clay_mode.export_asset(ctx, tab)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "The exact geometry, as an ordinary asset. It picks up rigging, posing, "
            "sprite sheets, the triangle retarget and every mesh export, because all "
            "of those are functions of model.glb."
        )

    # Two labelled buttons rather than "Export File..." plus a bare "OBJ": the
    # first spelling wrote GLB without saying so, and a format is exactly the
    # thing a user reading the row needs to see before pressing.
    tip = (
        "Saves the document as a plain mesh file on disk, for handing straight "
        "to another tool. The library never sees it."
    )
    if widgets.disabled_button(f"{icons.DOWNLOAD} Export GLB...", ready, reason=why):
        clay_mode.export_mesh_file(ctx, tab, "glb")
    if imgui.is_item_hovered():
        imgui.set_tooltip(tip)
    imgui.same_line()
    if widgets.disabled_button("Export OBJ...", ready, reason=why):
        clay_mode.export_mesh_file(ctx, tab, "obj")
    if imgui.is_item_hovered():
        imgui.set_tooltip(EXPORT_OBJ_TOOLTIP)

    imgui.same_line()
    if widgets.disabled_button(f"{icons.CAMERA} Save screenshot...", ready, reason=why):
        clay_mode.save_screenshot(ctx, tab)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Saves a lit PNG of the document from the angle you are looking at it, "
            "without the grid or gizmos. Position objects first, then frame the view."
        )

    if tab.job_id:
        widgets.muted(f"Last exported as {tab.job_id}")


def _recent(ctx: Any) -> None:
    """The recent list, on the bridge, on **both** branches.

    Plotter's bridge already draws its list whether or not a document is open,
    and that is the answer: a recent list is how you get *back* to work, so the
    one moment it matters most is the moment there is nothing open. Clay's was
    on the empty canvas instead, which is the one screen it disappears from as
    soon as it becomes useful again.
    """
    from pathlib import Path

    widgets.recent_files(
        clay_mode.recent_paths(ctx),
        lambda path: clay_mode.open_path(ctx, Path(path)),
    )
