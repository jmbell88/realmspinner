"""Clay's menu strip: Select, Add, Object (or Mesh and its mode), UV.

Blender's header, on ``inker_menu``'s and ``plotter_menu``'s shape: a row of
menu names above the bar, drawn from the centre window because an imgui popup
only renders in the id stack of the window that opened it, and through
:mod:`~realmspinner.studio.toolbar` so a strip too wide for the pane collapses
into an overflow with the names back rather than clipping.

**Every row is generated.** The op rows come from
:func:`~realmspinner.studio.modes.clay.menutree.resolve` -- the same table the
right-click menu reads -- so a label, a key, a greyed reason or a parameter
dialog cannot differ between the two. The Add menu's primitive and figure rows
come from ``primitives.CATEGORIES`` and ``presets.ASSEMBLIES``, the registries
the left Add palette is already generated from, so a new shape appears in all
three on its own.

**The strip follows the element mode**, as Blender's swaps Object for Mesh /
Vertex / Edge / Face: a menu whose groups hold nothing for the current mode is
not drawn (:func:`~realmspinner.studio.modes.clay.menutree.resolve` drops it),
which is why vertex mode shows no UV menu and object mode shows no Vertex one.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ......kernels.mesh import presets
from ..... import controls, toolbar
from ... import menutree as clay_menutree
from ... import mode as clay_mode
from ... import ops as clay_ops
from . import bridge as clay_bridge
from . import tools as clay_tools

BAR = "clay-menu"

_SAVING = "This document is being written; the rows come back when it lands."


def draw(ctx: Any, state: Any, tab: Any) -> None:
    """The strip and the popups its names open. Called from the header."""

    menus = clay_menutree.resolve(tab.doc.element_mode)
    items = [
        toolbar.Item(menu.title, menu.title, role=controls.ButtonRole.GHOST)
        for menu in menus
    ]
    clicked = toolbar.toolbar(BAR, items)
    if clicked:
        imgui.open_popup(controls.menu_bar_id(BAR, clicked))
    for menu in menus:
        with controls.menu_popup(controls.menu_bar_id(BAR, menu.title)) as opened:
            if opened:
                _menu_rows(ctx, state, tab, menu)
    # The Generate popup is hosted here rather than in the Document pane: the
    # strip is drawn whenever a document is open, where a pane can be hidden by
    # the layout, and the row that asks for it is in this strip's Add menu.
    if state.generate_open_pending:
        state.generate_open_pending = False
        imgui.open_popup(clay_bridge.GENERATE_POPUP)
    clay_bridge.generate_popup(ctx, tab, state)


def _menu_rows(ctx: Any, state: Any, tab: Any, menu: Any) -> None:
    """One menu's sections: submenus as submenus, inline groups between rules."""
    previous_inline = False
    for index, section in enumerate(menu.sections):
        # An Add source draws submenus of its own, so it is not an inline run.
        inline = not section.submenu and not section.source
        # A rule between an inline run and whatever follows it, so Object's
        # Duplicate does not touch the Transform submenu header under it and
        # Delete stays a row of its own at the bottom.
        if index and (inline or previous_inline):
            controls.menu_separator()
        previous_inline = inline
        if section.source:
            _add_source(ctx, state, tab, section)
        elif section.submenu:
            with controls.menu(section.label) as opened:
                if opened:
                    for op in section.ops:
                        _op_row(ctx, tab, op)
        else:
            for op in section.ops:
                _op_row(ctx, tab, op)


def _op_row(ctx: Any, tab: Any, op: Any) -> None:
    """One op: greyed with its reason, run, or its dialog asked for."""
    doc = tab.doc
    enabled = op.enabled(doc) and not tab.saving
    reason = _SAVING if tab.saving else clay_ops.reason_for(op, doc)
    clicked, _ = controls.menu_item(
        f"{op.label}##{BAR}/{op.name}",
        op.key,
        False,
        enabled,
        reason=reason,
        tooltip=op.hint,
    )
    if clicked:
        # ``fire_op`` rather than ``imgui.open_popup`` here: a submenu is a
        # popup window of its own, so a popup opened from inside one is named in
        # *its* id stack and the viewport's ``params_popup`` would never find
        # it. The event layer's door asks for the popup by state instead, which
        # is exactly the case ``state.open_op_popup`` was written for.
        clay_mode.fire_op(ctx, doc, op)


def _add_source(ctx: Any, state: Any, tab: Any, section: Any) -> None:
    """The Add menu's rows that are not ops: shapes, figures.

    ``section.source`` names which: shapes, figures, or the two rows that bring
    something in from outside (a file, or a generated mesh).
    """
    doc = tab.doc
    if section.source == "primitives":
        for label, names in clay_tools.sections():
            with controls.menu(label.title(), enabled=not tab.saving, reason=_SAVING) as opened:
                if not opened:
                    continue
                for name in names:
                    if controls.menu_item(
                        f"{name.replace('_', ' ').title()}##{BAR}/add/{name}"
                    )[0]:
                        clay_tools.add_primitive(ctx, doc, name)
                        state.generator = name
    elif section.source == "import":
        controls.menu_separator()
        why = _SAVING if tab.saving else ""
        if controls.menu_item(
            f"Import Mesh...##{BAR}/import",
            "",
            False,
            not tab.saving,
            reason=why,
            tooltip="Open a .glb, .obj, .stl or .ply into this document. Units and "
            "up axis are in Properties > Scene.",
        )[0]:
            clay_mode.ask_import_mesh(ctx)
        if controls.menu_item(
            f"Generate...##{BAR}/generate",
            "",
            False,
            not tab.saving,
            reason=why,
            tooltip="Build a mesh from a prompt or an image and land it here.",
        )[0]:
            state.generate_open_pending = True
    elif section.source == "figures" and presets.ASSEMBLIES:
        with controls.menu("Figures", enabled=not tab.saving, reason=_SAVING) as opened:
            if opened:
                for key, (label, _build) in presets.ASSEMBLIES.items():
                    if controls.menu_item(f"{label}##{BAR}/figure/{key}")[0]:
                        clay_tools.add_assembly(ctx, doc, key)
                        state.generator = key
