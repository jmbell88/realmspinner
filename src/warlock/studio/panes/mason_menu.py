"""The right-mouse context menu over the Mason viewport.

``clay_menu``'s own idea, one dimension over: the operations that apply to
the current selection, under the cursor, at the moment the user wants them --
Group, Duplicate, Delete and drop-to-ground, the placement verbs a scene
editor reaches for most often. This is the only layer here that knows imgui
exists; what a click actually does is ``mason_mode``'s.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imgui_bundle import imgui

from .. import controls, icons, mason_assets, mason_mode, widgets
from ..mason import ops as mops
from ..mason import scene as mscene

POPUP = "mason-context"


def draw(ctx: Any, view: Any) -> None:
    """Open and render the menu. Called from the viewport pane, after the image."""
    tab = mason_mode.active(ctx)
    if tab is None:
        return
    if view is not None and view.menu_request is not None:
        view.menu_request = None
        imgui.open_popup(POPUP)

    if imgui.begin_popup(POPUP):
        widgets.popup_chrome(_imgui=imgui)
        _rows(ctx, tab)
        imgui.end_popup()


def _rows(ctx: Any, tab: Any) -> None:
    doc = tab.doc
    selected = bool(doc.selection)
    if tab.saving:
        widgets.secondary("Saving...")
        controls.menu_separator()
        return
    if controls.menu_item(f"{icons.COPY} Duplicate", "Ctrl+J", False, selected)[0]:
        mason_mode.duplicate_selected(ctx)
    if controls.menu_item("Group", "Ctrl+G", False, selected)[0]:
        mason_mode.group_selected(ctx)
    if controls.menu_item("Ungroup", "Ctrl+Shift+G", False, selected)[0]:
        mason_mode.ungroup_selected(ctx)
    controls.menu_separator()
    # Where the *first* prefab is made, and it has to be somewhere that exists
    # before one does: the Prefabs pane is a conditional slot that only appears
    # once the document has a template, so it cannot be the place a template is
    # authored. See ``panes/mason_prefabs``'s own docstring.
    if controls.menu_item(
        f"{icons.COPY} Make prefab", "", False, len(doc.selection) == 1
    )[0]:
        mason_mode.define_prefab_from_selection(ctx)
    if controls.menu_item(f"{icons.UNLINK} Unpack instance", "", False, _any_instance(doc))[0]:
        mason_mode.unpack_selected(ctx)
    controls.menu_separator()
    if controls.menu_item("Drop to ground", "", False, selected)[0]:
        _drop_to_ground(ctx, tab)
    controls.menu_separator()
    if controls.menu_item(f"{icons.TRASH} Delete", "Del", False, selected)[0]:
        mason_mode.delete_selected(ctx)


def _any_instance(doc: Any) -> bool:
    from ..mason import nodes as nd

    return any(isinstance(doc.node(uid), nd.PrefabNode) for uid in doc.selection)


def _drop_to_ground(ctx: Any, tab: Any) -> None:
    """Rest every selected node's box on the terrain (or the ground plane).

    Shares its arithmetic with ``mason_tools``'s own "Drop selection to
    ground" button -- both read ``mason.ops.drop_to_ground`` off the same
    world boxes, because a menu row and a sidebar button that computed this
    two different ways would be free to disagree about where "the ground" is.
    """
    doc = tab.doc
    uids = list(doc.selection)
    if not uids:
        return
    source = mason_assets.ensure(ctx)
    boxes = {}
    for uid in uids:
        box = mscene.world_bounds(doc, source, uids=[uid])
        if box is not None:
            boxes[uid] = box
    deltas = mops.drop_to_ground(boxes, terrain=doc.terrain)
    for uid, delta in deltas.items():
        node = doc.node(uid)
        if node is None:
            continue
        was = node.trs()
        translation = np.asarray(node.translation, dtype="f8") + np.asarray(delta, dtype="f8")
        doc.set_transform(uid, translation=translation, was=was)
