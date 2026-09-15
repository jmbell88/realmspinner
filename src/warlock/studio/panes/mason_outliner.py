"""The scene tree: what is in the document, and what is selected.

``clay_outliner``'s own argument carries over unchanged, one dimension over:
selection here is the same selection the viewport shows, and it is
deliberately **not undoable** -- clicking a node is not laborious to redo the
way a lasso is, and an undoable selection would move ``history.head``, which
would make a saved document ask to be saved again just because the user
looked at a different node.

**Ctrl toggles and Shift extends**, the convention every outliner in this app
shares. The range anchor is ``state.outliner_anchor``, a node *uid* rather
than a row index -- every address in ``mason/`` is one, for the reason
``nodes.py``'s module docstring gives for children-not-parent: a list that
reorders makes an index anchor point at whatever now sits where the anchor
used to, silently measuring a different range than the one the user shift-
clicked to make.

Depth is drawn as indentation rather than a second tree widget: ``doc.walk()``
already hands back ``(node, parent_uid, index, depth)`` in the one order that
matters (export order, matching the document's own child lists), and imgui's
own tree nodes cost a second identity to key selection off of that this pane
does not need.
"""

from __future__ import annotations

import contextlib
from typing import Any

from imgui_bundle import imgui

from .. import controls, icons, mason_mode, theme, widgets
from ..manual import render as manual_render
from ..mason import gltfout
from ..tokens import sp

ROW_HEIGHT = 24.0
INDENT = 16.0

#: One glyph per ``gltfout.kind_of`` answer -- the one naming rule this
#: package asks callers to follow, so a row's icon and the manifest's own
#: notion of "what kind is this" cannot drift apart the way two independent
#: ``isinstance`` chains eventually would.
KIND_ICONS = {
    "mesh": icons.BOX,
    "light": icons.SPARKLES,
    "camera": icons.CAMERA,
    "prefab": icons.COPY,
    "terrain": icons.GRID,
    "group": icons.FOLDER_OPEN,
}

_DRAG_NODE = "mason-node"


def draw(ctx: Any) -> None:
    with widgets.section_blocks():
        _body(ctx)


def _body(ctx: Any) -> None:
    state = mason_mode.ensure(ctx)
    tab = mason_mode.active(ctx)
    widgets.section("Outliner")
    manual_render.help_button(ctx, "mason-outliner")
    if tab is None:
        return
    doc = tab.doc
    rows = list(doc.walk())
    if not rows:
        widgets.empty_state(icons.LIST, "Empty scene", "Place something from Assets to start.")
        return

    imgui.begin_disabled(tab.saving)
    needle = widgets.list_filter(ctx, "mason-outliner", len(rows))
    _visibility_row(doc)
    shown = 0
    for node, parent_uid, index, depth in rows:
        if needle and needle not in (node.name or "").lower():
            continue
        shown += 1
        _row(
            ctx, state, doc, node, parent_uid, index, depth,
            filtered=bool(needle), saving=bool(tab.saving),
        )
    widgets.no_matches(needle, shown)
    imgui.end_disabled()


def _visibility_row(doc: Any) -> None:
    hidden = sum(1 for node in doc.all_nodes() if not node.visible)
    # Both carry a ``reason`` as well as a tooltip, which is what
    # ``disabled_button`` exists to make possible: a greyed control that cannot
    # say why is what ``exercise_mode`` reports as ``disabled-no-reason``, and it
    # reported both of these the moment the driver's seed had a node for the
    # outliner to draw a row for.
    if widgets.disabled_button(
        f"{icons.EYE} Solo##masonsolo",
        bool(doc.selection),
        tooltip="Show only the selected nodes",
        reason="Select something to show on its own.",
    ):
        doc.isolate(doc.selection)
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.EYE} Show all##masonshowall",
        hidden > 0,
        tooltip="Unhide every hidden node",
        reason="Nothing is hidden.",
    ):
        doc.show_all()
    if hidden:
        widgets.muted(f"{hidden} hidden")


def _click(state: Any, doc: Any, node: Any) -> None:
    """Apply one row click, honouring Ctrl (toggle) and Shift (range) --
    ``clay_outliner._click``, verbatim, against node uids instead of object
    uids."""
    io = imgui.get_io()
    if io.key_shift and state.outliner_anchor:
        range_uids = _range(doc, state.outliner_anchor, node.uid)
        doc.select(range_uids)
        return
    if io.key_ctrl:
        chosen_uids = set(doc.selection) ^ {node.uid}
        doc.select(chosen_uids)
    else:
        doc.select([node.uid])
    state.outliner_anchor = node.uid


def _range(doc: Any, anchor: int, uid: int) -> list[int]:
    """Every node between two uids in ``doc.walk()``'s own order, inclusive.

    Measured against ``doc.walk()`` fresh rather than a row list the caller
    might be holding -- the whole point of a uid anchor is that it survives a
    reorder, so this must re-walk the *current* tree, not one captured before
    it.
    """
    order = [node.uid for node, _p, _i, _d in doc.walk()]
    try:
        lo, hi = sorted((order.index(anchor), order.index(uid)))
    except ValueError:
        return [uid]
    return order[lo : hi + 1]


def _reorder(ctx: Any, doc: Any, node: Any, *, filtered: bool, saving: bool) -> None:
    """Drag one row onto another to **reparent** it -- the dropped node becomes
    the target's last child.

    This used to insert the dropped node at the target's own index under the
    target's own *parent*, which is a reorder dressed as a tree drag: dropping a
    prop onto a group put it beside the group rather than into it, so the one
    thing a scene editor's outliner drag is for could not be done at all.
    Reparenting is what a drop onto a row means everywhere else, and
    ``move_node``'s own refusal (a target that is the node or one of its
    descendants) is what makes it safe to offer over every row rather than only
    over groups -- caught below, so an impossible drop is a no-op rather than an
    exception on the frame thread.

    **Sibling order is still addressable, and has to be**: it is export order and
    outliner order both. It moved to the row menu's Move up / Move down, which
    are an *index* computed from the node's uid at the moment of the press --
    never a row position held across a frame, the rule every address in
    ``mason/`` follows. A drop-between-rows target would be the other way to
    offer it and was not taken: it needs an invisible gap widget per row, which
    is a control the ``controls.py`` census cannot see.

    Disabled while filtered or saving for the identical reasons ``clay_outliner``
    states: a filtered list is not the tree, and a restructure mid-encode would
    write a file describing a document that never existed.
    """
    if filtered or saving:
        return
    if imgui.begin_drag_drop_source(imgui.DragDropFlags_.source_no_hold_to_open_others.value):
        imgui.set_drag_drop_payload_py_id(_DRAG_NODE, node.uid)
        imgui.text(node.name or "node")
        imgui.end_drag_drop_source()
    if imgui.begin_drag_drop_target():
        payload = imgui.accept_drag_drop_payload_py_id(_DRAG_NODE)
        if payload is not None:
            with contextlib.suppress(KeyError, ValueError):
                dropped_uid = int(payload.data_id)
                if dropped_uid != node.uid:
                    doc.move_node(dropped_uid, len(node.children), parent_uid=node.uid)
    del ctx


def _context_menu(ctx: Any, state: Any, doc: Any, node: Any) -> None:
    if not imgui.begin_popup_context_item(f"masonrow{node.uid}"):
        return
    widgets.popup_chrome(_imgui=imgui)
    if node.uid not in doc.selection:
        doc.select([node.uid])
    if controls.menu_item(f"{icons.PENCIL} Rename", "", False)[0]:
        state.renaming = node.uid
    if controls.menu_item(f"{icons.COPY} Duplicate", "Ctrl+J", False)[0]:
        mason_mode.duplicate_selected(ctx)
    if controls.menu_item(f"{icons.EYE} Solo", "", False)[0]:
        doc.isolate([node.uid])
    widgets.divider()
    # Sibling order, by uid: ``index_of`` is read here, at the press, rather
    # than passed in from the row loop -- a number captured a frame ago would
    # address whatever has since moved into that position.
    index = doc.index_of(node.uid)
    siblings = doc.children_of(doc.parent_uid_of(node.uid))
    if controls.menu_item(f"{icons.ARROW_UP} Move up", "", False, index > 0)[0]:
        doc.move_node(node.uid, index - 1)
    if controls.menu_item(
        f"{icons.ARROW_DOWN} Move down", "", False, index < len(siblings) - 1
    )[0]:
        doc.move_node(node.uid, index + 1)
    if controls.menu_item(
        f"{icons.ARROW_LEFT} Move to root", "", False, doc.parent_uid_of(node.uid) is not None
    )[0]:
        doc.move_node(node.uid, len(doc.roots), parent_uid=None)
    widgets.divider()
    if controls.menu_item("Group", "G", False)[0]:
        mason_mode.group_selected(ctx)
    if controls.menu_item("Ungroup", "Shift+G", False, groupish(doc))[0]:
        mason_mode.ungroup_selected(ctx)
    if controls.menu_item(f"{icons.COPY} Make prefab", "", False, len(doc.selection) == 1)[0]:
        # The 2026-09-12 audit's docs-03: see ``mason_menu.py``'s identical row.
        mason_mode.prompt_define_prefab_from_selection(ctx)
    if controls.menu_item(f"{icons.UNLINK} Unpack instance", "", False, _instanceish(doc))[0]:
        mason_mode.unpack_selected(ctx)
    widgets.divider()
    if controls.menu_item(f"{icons.TRASH} Delete", "Del", False)[0]:
        mason_mode.delete_selected(ctx)
    imgui.end_popup()


def groupish(doc: Any) -> bool:
    """Whether anything selected is a group with children -- what Ungroup can
    act on. Asked of the *document* rather than of the clicked row, because
    Ungroup (like Group) works on the selection and the context menu has already
    made the clicked row part of it.

    **Not prefixed private** -- the 2026-09-15 audit's mason-03 found the
    viewport context menu (``mason_menu.py``) enabling its own "Ungroup" row
    for any selection at all, gated on nothing, while this row was already
    gated on this exact predicate; the handler both rows call
    (``mason_mode.ungroup_selected``) just returns silently for a selection
    with no group in it. One predicate, imported rather than a second copy
    that could drift from this one the same way the menu row already had.
    """
    from ..mason import nodes as nd

    return any(
        isinstance(doc.node(uid), nd.GroupNode) and doc.node(uid).children
        for uid in doc.selection
    )


def _instanceish(doc: Any) -> bool:
    from ..mason import nodes as nd

    return any(isinstance(doc.node(uid), nd.PrefabNode) for uid in doc.selection)


def _row(
    ctx: Any,
    state: Any,
    doc: Any,
    node: Any,
    parent_uid: int | None,
    index: int,
    depth: int,
    *,
    filtered: bool,
    saving: bool,
) -> None:
    selected = node.uid in doc.selection
    imgui.push_id(str(node.uid))

    eye = icons.EYE if node.visible else icons.EYE_OFF
    if controls.button(f"{eye}##vis", (sp(28), sp(ROW_HEIGHT))):
        doc.set_props(node.uid, visible=not node.visible)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Hidden nodes do not render, export or pick.")
    imgui.same_line()

    if not filtered and depth:
        imgui.dummy((sp(INDENT) * depth, 1))
        imgui.same_line()

    kind = gltfout.kind_of(node)
    imgui.text(KIND_ICONS.get(kind, icons.BOX))
    imgui.same_line()

    width = imgui.get_content_region_avail().x - sp(28) - imgui.get_style().item_spacing.x
    if state.renaming == node.uid:
        imgui.set_next_item_width(width)
        name = widgets.input_text("##masonrename", node.name, max_length=120, commit=True)
        if name != node.name:
            doc.set_props(node.uid, name=name)
        if imgui.is_item_deactivated():
            state.renaming = 0
    else:
        label = node.name or f"{kind} {node.uid}"
        hidden = not node.visible
        if hidden:
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.MUTED)))
        if controls.selectable(
            f"{label}##row", selected, imgui.SelectableFlags_.none, (width, 0)
        )[0]:
            _click(state, doc, node)
        if hidden:
            imgui.pop_style_color()
        _reorder(ctx, doc, node, filtered=filtered, saving=saving)
        _context_menu(ctx, state, doc, node)
        if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
            state.renaming = node.uid

    imgui.same_line()
    if controls.button(f"{icons.TRASH}##del", (sp(28), sp(ROW_HEIGHT))):
        doc.select([node.uid])
        mason_mode.delete_selected(ctx)
    imgui.pop_id()
    del parent_uid, index
