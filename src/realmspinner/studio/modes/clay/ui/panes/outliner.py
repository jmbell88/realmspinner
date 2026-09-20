"""The object list: what is in the document, and what is selected.

**Tranche 3: scene structure -- the list is now a tree.** Depth is drawn as
indentation and an expander sits beside any row with children, the same
"indentation, not a second widget" call Mason's own outliner already makes
(that module's docstring gives the reason: ``doc.walk()``'s order already is
the one that matters, and imgui's own tree nodes would be a second identity to
key selection off of that this pane does not need). Clay's document has no
``walk()`` of its own -- :func:`_tree_rows` is this pane's, built from
``ClayDoc.roots``/``children_of``, both already in document order.

A row's own document-list position (what used to decide screen order outright)
now decides only *sibling* order: the roots come first, in document order,
then each root's children, in document order, recursively. That is a change
from before parenting existed -- the top row used to be the *last* object
added, matching a layers panel -- and it is the one the tree forces: a child
has to draw under its parent, and "newest first" and "grouped by parent" are
two different orderings that cannot both hold at once. Document order is the
one every other reader of the object list already uses (:func:`~.document.
to_model`, the glb writer, ``_range`` below), so this is the pane joining
everyone else rather than a new invention.

Selection here is the same selection the viewport shows, and it is deliberately
**not undoable**: clicking an object is not laborious to redo the way a lasso
is, an undoable selection would move ``history.head``, and a document would
then ask to be saved because the user looked at a different object.

**Ctrl toggles and Shift extends**, the convention every file list and every
outliner shares. The range anchor is a *uid* on ``ClayState`` rather than a row
index, for the reason every address in this package is one: the list reorders,
and an index anchor would quietly measure the range from a different object.
An anchor that has been deleted falls back to the clicked row, which is the
same thing a plain click does.

**A drop onto a row reparents; a drop between two rows reorders.** The two
gestures share one drag payload and are told apart by *where inside the row's
own rect* the drop lands (:func:`_drop_zone`): the top and bottom quarters
are "between" -- ``move_object``, exactly what every drop in this pane did
before parenting existed -- and the middle half is "onto" --
``ClayDoc.set_parent`` with ``keep_world=True``, so a dropped object does not
visibly jump the moment it changes parent. A cycle (dropping a group onto its
own descendant) is refused by ``set_parent`` itself and shown as a toast
through this pane's own door onto :func:`~.clay.ops.toast`.
"""

from __future__ import annotations

import contextlib
from typing import Any

from imgui_bundle import imgui

from ..... import controls, icons, theme, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import mode as clay_mode

# Design pixels: every use goes through sp(), or the row keeps its 1.0x height
# while the glyph inside it grows with the UI scale.
ROW_HEIGHT = 24.0
#: How far one level of the tree indents, in design pixels -- Mason's own
#: outliner's figure, so the two scene trees in this app read the same depth
#: the same width.
INDENT = 16.0

# The drag-and-drop payload name. One constant, because the source and the
# target have to agree and two spellings of it fail silently -- the row simply
# refuses every drop, with nothing to say why.
_DRAG_OBJECT = "clay-object"

#: The middle band of a row's own rect that a drop reads as "reparent onto
#: this row" rather than "reorder beside it" -- the top and bottom quarters
#: outside it are "between". Half the row, not all of it: a drop has to be
#: unambiguous, and a band that started at the very top edge would never let
#: a user land "between" two rows that are visually touching.
_DROP_ONTO_BAND = (0.25, 0.75)


def draw(ctx: Any) -> None:
    """This pane's headings, on tinted blocks.

    The blocks are opened *here* rather than in :func:`layout.pane`, which is
    flat: a pane on a wide canvas wants no tint, and this is one of the four
    narrow sidebars the grouping was written for (see
    ``tests/test_section_blocks.py`` for the report it came from). Wrapping
    ``_body`` rather than inlining the ``with`` keeps every early return inside
    the scope, and the scope closes its last block on the way out.
    """
    with widgets.section_blocks():
        _body(ctx)


def _tree_rows(doc: Any) -> list[tuple[Any, int, bool]]:
    """Every object, depth-first, in document order within each parent, as
    ``(obj, depth, has_children)``.

    ``ClayDoc.roots``/``children_of`` are already document order (both
    methods' own docstrings), so this is the plain depth-first walk over
    them -- roots first, then each root's own children before its next
    sibling. Ignores collapse entirely: what a row's expander hides is a
    *drawing* decision (``_body``'s own depth-skip loop), never a fact this
    walk itself forgets, because a tag or name filter has to be able to find
    a match inside a collapsed group.

    An explicit stack, not recursion, the same shape ``ClayDoc.ancestors``
    uses for its own parent walk: the 2026-09-19 audit's clay-02 found this
    walk raised an uncaught ``RecursionError`` on a legal, acyclic parent
    chain of a few thousand objects (well inside ``glbimport.MAX_OBJECTS``),
    crashing the app the moment the outliner opened. Each uid is pushed with
    its depth; children are pushed in reverse so the stack still pops them
    in document order, one root's whole subtree finished before its next
    sibling starts -- exactly what the old recursive ``walk`` produced.
    """
    rows: list[tuple[Any, int, bool]] = []
    stack: list[tuple[int, int]] = [(root, 0) for root in reversed(doc.roots())]
    while stack:
        uid, depth = stack.pop()
        try:
            obj = doc.by_uid(uid)
        except KeyError:  # pragma: no cover - defensive; no caller builds this
            continue
        children = doc.children_of(uid)
        rows.append((obj, depth, bool(children)))
        stack.extend((child, depth + 1) for child in reversed(children))
    return rows


def _tag_filter(state: Any) -> str:
    """A one-line box narrowing rows by tag. -> the lowered query.

    Always shown, unlike the name filter's ``list_filter``: that one hides
    below eight rows because a short list needs no search box at all, but a
    tag filter earns its place the moment two objects share one tag, which a
    four-object document can already do.
    """
    imgui.set_next_item_width(-1)
    state.outliner_tag_filter = widgets.input_text(
        "##clay-outliner-tag", state.outliner_tag_filter, max_length=60, hint="Filter by tag..."
    )
    return state.outliner_tag_filter.strip().lower()


def _body(ctx: Any) -> None:
    state = clay_mode.ensure(ctx)
    tab = state.active
    widgets.section("Outliner")
    manual_render.help_button(ctx, "clay-outliner")
    if tab is None:
        # One voice for one empty state -- the canvas's ``nothing_open`` is it.
        # This heading plus three more each saying "Nothing open." was Clay
        # telling the user four times, which reads as four problems.
        return
    doc = tab.doc
    if not doc.objects:
        widgets.empty_state(icons.LIST, "Empty document", "Add a primitive to start.")
        return

    # The 2026-09-18 audit's clay-03: visibility is one history step per
    # toggle, same as every other ``set_props`` caller in this pane (rename)
    # -- there is no exempt control. Only *selection* sits outside history,
    # for the reason this module's docstring gives. The whole panel is
    # disabled regardless of which controls would touch history: a save is
    # encoding the document on a task thread, and a list where some controls
    # respond and others do not is more confusing than one that does not.
    imgui.begin_disabled(tab.saving)
    # J86. Above the rows and inside the disable, because a search that worked
    # while a save was running would let the user narrow the list to one object
    # and then find every control on it refusing the click.
    needle = widgets.list_filter(ctx, "clay-outliner", len(doc.objects))
    tag_needle = _tag_filter(state)
    _visibility_row(doc)
    filtered = bool(needle) or bool(tag_needle)
    shown = 0
    # Collapsed subtrees are skipped by depth: once a collapsed row is drawn,
    # every following row deeper than it belongs to its own (hidden) subtree,
    # until a row at or above its depth -- a sibling, or an uncle -- ends it.
    # Suspended outright while filtered, the same "flat list of matches" rule
    # Mason's outliner states for its own indent: a filtered list is not the
    # tree, and a match nested three deep under a collapsed group must still
    # be found.
    skip_below: int | None = None
    for obj, depth, has_children in _tree_rows(doc):
        if not filtered and skip_below is not None:
            if depth > skip_below:
                continue
            skip_below = None
        if needle and needle not in (obj.name or "").lower():
            continue
        if tag_needle and not any(tag_needle in t for t in obj.tags):
            continue
        shown += 1
        _row(
            ctx, state, doc, obj, depth, has_children,
            filtered=filtered, saving=bool(tab.saving),
        )
        if not filtered and has_children and obj.uid in state.outliner_collapsed:
            skip_below = depth
    widgets.no_matches(needle or tag_needle, shown)
    imgui.end_disabled()


def _visibility_row(doc: Any) -> None:
    """Solo and Show all, above the list.

    Isolating is one click where nine eye toggles were, and it is the pair of
    them that makes it usable: a solo you cannot undo in one action is a solo
    you have to remember the previous state of. Both are single undo steps, so
    Ctrl+Z is the third way back.
    """
    hidden = sum(1 for obj in doc.objects if not obj.visible)
    if widgets.disabled_button(f"{icons.EYE} Solo##claysolo", bool(doc.selection)):
        doc.isolate(doc.selection)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Show only the selected objects")
    imgui.same_line()
    if widgets.disabled_button(f"{icons.EYE} Show all##clayshowall", hidden > 0):
        doc.show_all()
    if hidden:
        widgets.muted(f"{hidden} hidden")


def _click(state: Any, doc: Any, obj: Any) -> None:
    """Apply one row click, honouring Ctrl (toggle) and Shift (range)."""
    io = imgui.get_io()
    if io.key_shift and state.outliner_anchor:
        range_uids = _range(doc, state.outliner_anchor, obj.uid)
        doc.select(range_uids)
        return
    if io.key_ctrl:
        chosen_uids = set(doc.selection) ^ {obj.uid}
        doc.select(chosen_uids)
    else:
        doc.select([obj.uid])
    state.outliner_anchor = obj.uid


def _range(doc: Any, anchor: int, uid: int) -> list[int]:
    """Every uid visually between two rows, inclusive, in the order the tree
    actually draws them.

    Built from :func:`_tree_rows`, not ``doc.objects`` -- the 2026-09-20
    audit's clay-08: ``doc.objects`` is flat insertion order, and
    ``set_parent`` never reorders it, so once anything has been reparented,
    the tree's depth-first walk and the flat list disagree about what sits
    "between" two rows. A range built from the flat list silently omitted or
    included rows the user never saw between the two they Shift-clicked.
    """
    order = [o.uid for o, _depth, _has_children in _tree_rows(doc)]
    try:
        lo, hi = sorted((order.index(anchor), order.index(uid)))
    except ValueError:
        return [uid]
    return order[lo : hi + 1]


def _drop_zone(mouse_y: float, row_min_y: float, row_max_y: float) -> str:
    """Which of a row's own three horizontal bands *mouse_y* falls in --
    ``"before"``, ``"onto"`` or ``"after"``.

    A pure function beside the imgui plumbing that calls it, the shape every
    decidable rule in this pane's tests already takes (``_range``): the top
    and bottom quarters (:data:`_DROP_ONTO_BAND`) read as "between", the
    middle half as "onto". A degenerate (zero-height) row has no band to
    measure and reads as "onto" -- reparenting is the safer of the two to
    fall back to, since a between-drop with nothing to measure against would
    have to guess a position instead.
    """
    height = row_max_y - row_min_y
    if height <= 0:
        return "onto"
    frac = (mouse_y - row_min_y) / height
    lo, hi = _DROP_ONTO_BAND
    if frac < lo:
        return "before"
    if frac > hi:
        return "after"
    return "onto"


def _apply_drop(ctx: Any, doc: Any, dropped_uid: int, target: Any, index: int, zone: str) -> None:
    """One drop's worth of document edit -- a reparent or a reorder.

    ``zone == "onto"`` reparents through ``set_parent(..., keep_world=True)``,
    so the dropped object does not visibly jump the moment it changes parent;
    a cycle (dropping a group onto its own descendant, or an object onto
    itself) is ``set_parent``'s own refusal and is toasted here rather than
    left to raise on the frame thread, through :func:`~.clay.ops.toast` --
    the one door every other refusal in this mode already goes through.

    ``"before"``/``"after"`` reorder exactly as every drop in this pane did
    before parenting existed: ``move_object`` to the target row's own index
    (or the index just after it), touching no object's ``parent`` at all.
    """
    if dropped_uid == target.uid:
        return
    if zone == "onto":
        from ......kernels.mesh.elements import OpError

        try:
            doc.set_parent(dropped_uid, target.uid, keep_world=True)
        except OpError as error:
            from ... import ops as clay_ops

            clay_ops.toast(ctx, str(error))
        except KeyError:
            pass  # the dropped object was deleted between the pick-up and the drop
        return
    with contextlib.suppress(KeyError):
        doc.move_object(dropped_uid, index if zone == "before" else index + 1)


def _reorder(
    ctx: Any, doc: Any, obj: Any, *, filtered: bool, saving: bool
) -> None:
    """Drag one row onto another to reparent it, or between two rows to
    reorder it -- see :func:`_apply_drop` for which is which and why.

    **Disabled while the list is filtered**, which is the one rule here that is
    not obvious. The rows on screen are then a subset, so a between-drop names
    a position in the *filtered* list and there is no honest answer for where
    that is in the real one -- a drop that silently landed somewhere else
    would be a reorder the user cannot see. An onto-drop has no such
    ambiguity (it names the target row, not a position), but the filtered
    list is still every row that is going to disappear the moment the filter
    is cleared, and a drag source that behaves differently depending on
    whether the *target* happens to be filtered is worse than one switch that
    covers both.

    **Refused outright while the tab is saving**, rather than relying on
    ``_body``'s ``begin_disabled(tab.saving)`` -- ``inker_timeline._reorder``
    says why in the same words: *begin_disabled does not stop a drag-drop
    source from registering*. Every other control in this panel is genuinely
    covered by that disable, so a drag was the one write that could still land
    on ``doc.objects`` while ``write_glb`` was walking it on a task thread.
    """
    if filtered or saving:
        return
    # The payload carries the *uid* rather than a row index, for the reason
    # every address in this package is one: the drop reads the list again, so a
    # reorder in between cannot make it move a different object.
    if imgui.begin_drag_drop_source(imgui.DragDropFlags_.source_no_hold_to_open_others.value):
        imgui.set_drag_drop_payload_py_id(_DRAG_OBJECT, obj.uid)
        imgui.text(obj.name or "object")
        imgui.end_drag_drop_source()
    if imgui.begin_drag_drop_target():
        payload = imgui.accept_drag_drop_payload_py_id(_DRAG_OBJECT)
        if payload is not None:
            # Suppressed rather than checked: the object can be deleted between
            # the pick-up and the drop, and there is nothing for a half-finished
            # drag to report.
            with contextlib.suppress(KeyError):
                rect_min = imgui.get_item_rect_min()
                rect_max = imgui.get_item_rect_max()
                zone = _drop_zone(imgui.get_mouse_pos().y, rect_min.y, rect_max.y)
                _apply_drop(ctx, doc, int(payload.data_id), obj, doc.index_of(obj.uid), zone)
        imgui.end_drag_drop_target()


def _remove_object(ctx: Any, doc: Any, obj: Any) -> None:
    """Remove exactly this one row's object, not the selection.

    The 2026-09-09 audit's clay-03: this pane's trash button and its
    context-menu "Delete" are the only two Clay callers of
    ``doc.remove_object`` -- every other deletion goes through
    ``clay_ops.run(delete)``, which calls ``_forget_manifold`` after the
    removal so the properties panel's per-object mesh-check cache
    (``ClayState.manifold``) does not keep the removed object's ``Mesh``
    (positions/loops/starts arrays) alive under an orphaned uid, per the
    2026-09-08 audit's clay-08. These two sites called ``remove_object``
    straight, so they leaked exactly the cache entry clay-08 had already
    fixed everywhere else. Deliberately *not* routed through the
    selection-wide delete op instead -- that would change what the button
    does, which this finding does not ask for.

    Tranche 3: scene structure. ``remove_object`` is one of the locking
    doors (``document.py``'s own list) and now genuinely can refuse -- a
    locked row's own trash button existed before locking did and never had
    anything to catch. Toasted rather than left to raise on the frame
    thread, through the same door every other refusal in this mode goes
    through.
    """
    from ......kernels.mesh.elements import OpError
    from ... import ops as clay_ops

    try:
        doc.remove_object(obj.uid)
    except OpError as error:
        clay_ops.toast(ctx, str(error))
        return
    clay_ops._forget_manifold(ctx, [obj.uid])


def _context_menu(ctx: Any, state: Any, doc: Any, obj: Any) -> None:
    """Rename, duplicate and delete on the row itself.

    Duplicate is here rather than only on Ctrl+J because the keyboard version
    acts on the *selection*, and the thing a user right-clicks is one row --
    which is why it selects the row first: a menu item that operated on
    something other than what was clicked would be the sharpest possible
    misreading of a right-click.
    """
    if not imgui.begin_popup_context_item(f"clayrow{obj.uid}"):
        return
    widgets.popup_chrome(_imgui=imgui)
    if obj.uid not in doc.selection:
        doc.select([obj.uid])
    if controls.menu_item(f"{icons.PENCIL} Rename", "", False)[0]:
        state.renaming = obj.uid
    if controls.menu_item(f"{icons.COPY} Duplicate", "Ctrl+J", False)[0]:
        from ... import mode as clay_mode

        clay_mode._duplicate_selection(ctx, state, doc)
    if controls.menu_item(f"{icons.EYE} Solo", "", False)[0]:
        doc.isolate([obj.uid])
    widgets.divider()
    if controls.menu_item(f"{icons.TRASH} Delete", "Del", False)[0]:
        _remove_object(ctx, doc, obj)
    imgui.end_popup()


def row_label(obj: Any) -> str:
    """This row's display text: the object's name (or a placeholder for an
    unnamed one), with a distinct icon prefix for a collider.

    clay-25 (2026-09-19 audit): the outliner drew a collider as an ordinary
    row with ordinary icons -- the eye and the lock are the only two any row
    ever gets -- so the auto-generated name (``document.add_collider``'s own
    ``"<source> <kind label>"``) was the only thing anywhere in the tree
    saying an object was one, and a double-click rename could erase that with
    nothing left to say so. ``SQUARE_DASHED`` reads as "a boundary, not the
    real geometry" -- the same reason a dashed outline means a proxy or a
    guide everywhere else this app draws one.
    """
    label = obj.name or f"object {obj.uid}"
    if obj.role == "collider":
        return f"{icons.SQUARE_DASHED} {label}"
    return label


#: The expander's own width -- narrower than the eye/lock buttons, which are
#: real targets a thumb aims at; the expander is a tree decoration most users
#: never touch, so it earns less of the row. A leaf row reserves the same
#: width with a blank ``imgui.dummy`` (see ``_row``), which is what keeps
#: every row's name starting in the same column regardless of whether that
#: row happens to have children.
_EXPANDER_W = ROW_HEIGHT * 0.7


def _row(
    ctx: Any, state: Any, doc: Any, obj: Any, depth: int, has_children: bool,
    *, filtered: bool, saving: bool,
) -> None:
    selected = obj.uid in doc.selection
    imgui.push_id(str(obj.uid))

    # Indent and the expander are both suppressed while filtered: the rows on
    # screen are a flat list of matches at that point (``_body``'s own
    # comment), and an indent measured against a depth whose ancestors are
    # not being drawn reads as random left margin rather than as structure.
    if not filtered and depth:
        imgui.dummy((sp(INDENT) * depth, 1))
        imgui.same_line()
    if not filtered and has_children:
        collapsed = obj.uid in state.outliner_collapsed
        glyph = icons.CHEVRON_RIGHT if collapsed else icons.CHEVRON_DOWN
        if controls.button(f"{glyph}##expand", (sp(_EXPANDER_W), sp(ROW_HEIGHT))):
            if collapsed:
                state.outliner_collapsed.discard(obj.uid)
            else:
                state.outliner_collapsed.add(obj.uid)
        imgui.same_line()
    elif not filtered:
        imgui.dummy((sp(_EXPANDER_W), 1))
        imgui.same_line()

    eye = icons.EYE if obj.visible else icons.EYE_OFF
    if controls.button(f"{eye}##vis", (sp(28), sp(ROW_HEIGHT))):
        doc.set_props(obj.uid, visible=not obj.visible)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Hidden objects do not render, export or pick.")
    imgui.same_line()

    # Tranche 3: scene structure. ``locked`` is not a locking door itself
    # (``document.py``'s own paragraph on the point) -- toggling it, like
    # toggling visibility, is always allowed, or a mistake made while locked
    # could never be undone by anyone but the lock.
    lock_icon = icons.LOCK if obj.locked else icons.LOCK_OPEN
    if controls.button(f"{lock_icon}##lock", (sp(28), sp(ROW_HEIGHT))):
        doc.set_props(obj.uid, locked=not obj.locked)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Locked: geometry, transform and delete are refused until unlocked."
            if obj.locked
            else "Lock: refuses geometry, transform and delete on this object."
        )
    imgui.same_line()

    # What is still to come on this line: the delete button and the gap before
    # it. Measured rather than written out -- the literal 32 this replaces was
    # already a pixel short of sp(28) + spacing at scale 1.0, and at 1.6 it
    # reserved 32 for 58 and clipped the trash button off the pane.
    width = imgui.get_content_region_avail().x - sp(28) - imgui.get_style().item_spacing.x
    if state.renaming == obj.uid:
        imgui.set_next_item_width(width)
        # commit=True: the 2026-09-06 audit's clay-02 found this field
        # reporting a change on every keystroke, so ``set_props`` -- an
        # unconditional ``history.push`` -- fired once per letter typed and a
        # lone Ctrl+Z after a rename undid one character instead of the name.
        name = widgets.input_text("##rename", obj.name, max_length=120, commit=True)
        if name != obj.name:
            doc.set_props(obj.uid, name=name)
        if imgui.is_item_deactivated():
            state.renaming = 0
    else:
        label = row_label(obj)
        # A hidden object's name is drawn muted. It used to be a
        # ``text_colored(theme.MUTED, "")`` above the selectable, which coloured
        # nothing -- and, being an item rather than a style push, put the name
        # on the *next* line, so hiding an object silently doubled the height of
        # its row. The colour has to be pushed around the selectable, which is
        # what draws the text.
        hidden = not obj.visible
        if hidden:
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.MUTED)))
        if controls.selectable(
            f"{label}##row", selected, imgui.SelectableFlags_.none, (width, 0)
        )[0]:
            _click(state, doc, obj)
        if hidden:
            imgui.pop_style_color()
        # Both hang off the selectable, which is the item the user aims at --
        # the eye and the trash button are their own targets and must keep
        # doing only what they say.
        _reorder(ctx, doc, obj, filtered=filtered, saving=saving)
        _context_menu(ctx, state, doc, obj)
        if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
            state.renaming = obj.uid

    imgui.same_line()
    if controls.button(f"{icons.TRASH}##del", (sp(28), sp(ROW_HEIGHT))):
        _remove_object(ctx, doc, obj)
    imgui.pop_id()
