"""Clay's menu strip: Select, Add, Object (or Mesh and its mode), UV.

Blender's header, on ``inker_menu``'s and ``plotter_menu``'s shape: a row of
menu names above the bar, drawn from the centre window because an imgui popup
only renders in the id stack of the window that opened it, and through
:mod:`~realmspinner.studio.toolbar` so a strip too wide for the pane collapses
into an overflow with the names back rather than clipping.

**Every row is generated.** The op rows come from
:func:`~realmspinner.studio.modes.clay.menutree.resolve` -- the same table the
right-click menu reads -- so a label, a key, a greyed reason or a parameter
dialog cannot differ between the two. The Add menu's primitive rows
come from ``primitives.CLAY_GENERATORS``, the registry the tool rail's shape
flyout is generated from, so a new shape appears in all three on its own.

**The strip follows the element mode**, as Blender's swaps Object for Mesh /
Vertex / Edge / Face: a menu whose groups hold nothing for the current mode is
not drawn (:func:`~realmspinner.studio.modes.clay.menutree.resolve` drops it),
which is why vertex mode shows no UV menu and object mode shows no Vertex one.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ..... import controls, toolbar
from ... import menutree as clay_menutree
from ... import mode as clay_mode
from ... import ops as clay_ops
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
    """The Add menu's rows that are not ops: shapes, and the import row.

    ``section.source`` names which.
    """
    doc = tab.doc
    if section.source == "primitives":
        for label, names in clay_tools.sections():
            with controls.menu(label.title(), enabled=not tab.saving, reason=_SAVING) as opened:
                if not opened:
                    continue
                for name in names:
                    if controls.menu_item(f"{clay_tools.display_name(name)}##{BAR}/add/{name}")[0]:
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
            "up axis are in Properties > Document.",
        )[0]:
            clay_mode.ask_import_mesh(ctx)
