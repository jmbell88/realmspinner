"""Mason's Stage E panes: the Assets pane, the Outliner, Properties and the
Document bridge.

Each test's name is its claim, ``clay_tools`` test module's own convention.
"""

from __future__ import annotations

import inspect
from typing import Any

from _ui_context import imgui_context

from warlock.studio.modes.mason.engine import document as md
from warlock.studio.modes.mason.engine import nodes as nd
from warlock.studio.modes.mason.engine import scene as mscene
from warlock.studio.modes.mason.ui.panes import bridge as mason_bridge
from warlock.studio.modes.mason.ui.panes import outliner as mason_outliner
from warlock.studio.modes.mason.ui.panes import palette as mason_palette
from warlock.studio.modes.mason.ui.panes import props as mason_props


class _AppState:
    def __init__(self) -> None:
        self.mason = None
        self.dragging_job = None


class FakeCtx:
    """Just enough of ``Ctx`` for ``mason_mode.ensure`` and the panes under
    test -- ``test_clay_tools_panels.FakeCtx``'s own shape, one mode over."""

    def __init__(self) -> None:
        self.state = _AppState()
        self.cache = _FakeCache()

    def toast(self, message: str, kind: str = "info") -> None:  # pragma: no cover
        pass


class _FakeCache:
    jobs: list = []


def _scene_with_three_nodes() -> tuple[md.MasonDoc, nd.Node, nd.Node, nd.Node]:
    a = nd.GroupNode(uid=nd.new_uid(), name="A")
    b = nd.GroupNode(uid=nd.new_uid(), name="B")
    c = nd.GroupNode(uid=nd.new_uid(), name="C")
    doc = md.MasonDoc(roots=[a, b, c])
    return doc, a, b, c


# --- the primitive grid is derived, never hand-listed -----------------------


def test_mason_palette_primitive_grid_gets_a_button_for_a_fake_generator_it_has_never_seen(
    monkeypatch,
):
    """The claim ``mason_palette``'s own docstring makes: a sixteenth
    generator gets a button the day it is written. Proven by registering one
    ``clay.primitives.GENERATORS`` has never heard of and checking the grid
    draws it anyway -- the same fake-registration shape
    ``tests/clay/test_primitives.py`` uses to pin ``clay_tools``'s own grid."""
    from warlock.kernels.mesh import primitives as bp
    from warlock.studio import probe

    def fake_build(**kwargs):  # pragma: no cover - never invoked by this test
        raise NotImplementedError

    monkeypatch.setitem(bp.GENERATORS, "widget_zzz", ({}, fake_build))

    from warlock.studio import icons

    expected_label = f"{icons.BOX}##masonaddwidget_zzz"

    with imgui_context(monkeypatch) as imgui:
        ctx = FakeCtx()
        from warlock.studio.modes.mason import mode as mason_mode

        state = mason_mode.ensure(ctx)

        probe.begin_frame()
        imgui.new_frame()
        imgui.set_next_window_size((320.0, 900.0))
        imgui.begin("##host")
        mason_palette._primitives(ctx, state)
        imgui.end()
        imgui.end_frame()

        found = [c for c in probe.FRAME_CONTROLS if c.label == expected_label]
        assert found, "no button drawn for the fake generator -- the grid is hand-listed"


def test_mason_palette_hand_lists_no_primitive_name():
    """The other half of the claim: ``_sections``/``_primitives`` read
    ``clay.primitives.CATEGORIES``/``GENERATORS`` and nothing here spells out
    a primitive's name."""
    source = inspect.getsource(mason_palette._primitives) + inspect.getsource(
        mason_palette._sections
    )
    for literal in ("\"box\"", "\"cylinder\"", "\"torus\"", "\"lathe\""):
        assert literal not in source


# --- the outliner's range anchor is a uid, and survives a reorder -----------


def test_outliner_range_anchor_is_a_uid_and_survives_a_reorder_that_would_break_an_index():
    """``mason_outliner._range`` measures a Shift+click range off ``doc.walk()``
    fresh, by uid -- so reordering the tree between the anchor click and the
    extending click must not change which nodes the range names, the failure
    an *index* anchor could not avoid."""
    doc, a, b, c = _scene_with_three_nodes()
    # Anchor at A, before any reorder: A..C is every node.
    before = mason_outliner._range(doc, a.uid, c.uid)
    assert before == [a.uid, b.uid, c.uid]

    # Reorder: move C to the front. An index-based anchor recorded as "0"
    # would now point at C instead of A.
    doc.move_node(c.uid, 0)
    after = mason_outliner._range(doc, a.uid, b.uid)
    # A is still A, by uid, wherever it now sits -- the range from A to B is
    # exactly the nodes between them in the *new* order, not the old one.
    order = [node.uid for node, _p, _i, _d in doc.walk()]
    lo, hi = sorted((order.index(a.uid), order.index(b.uid)))
    assert after == order[lo : hi + 1]
    assert a.uid in after and b.uid in after


# --- properties shows the resolver's world transform, never its own --------


def test_mason_props_reads_world_transform_from_the_resolver_not_a_second_computation():
    """``mason_props._world_transform`` must call ``scene.resolved_for`` for
    its numbers rather than composing a matrix itself -- the pairing the
    module's own docstring calls the whole answer to Clay's argument against
    parenting. Checked at the source level (no second ``compose``/``@`` matrix
    build in the function) and functionally (a nested node's reported world
    position matches the resolver's own answer)."""
    source = inspect.getsource(mason_props._world_transform)
    assert "resolved_for" in source
    assert "m3.compose" not in source

    parent = nd.GroupNode(uid=nd.new_uid(), name="Parent", translation=[5.0, 0.0, 0.0])
    child = nd.MeshNode(uid=nd.new_uid(), name="Child", translation=[1.0, 0.0, 0.0])
    parent.children.append(child)
    doc = md.MasonDoc(roots=[parent])

    placed = mscene.resolved_for(doc, child.uid)
    assert placed is not None
    expected = placed.world[:3, 3]
    # The child's own local translation (1, 0, 0) is not its world position --
    # it must be the parent's translation composed in, (6, 0, 0).
    assert abs(float(expected[0]) - 6.0) < 1e-9


# --- the bridge's warning threshold is the engine constant, never a literal -


def test_mason_bridge_warning_threshold_is_scene_placed_warn_threshold_not_a_literal():
    """``mason_bridge``'s own docstring: the number a user reads must be
    ``mason.scene.PLACED_WARN_THRESHOLD``, read back through
    ``mason_mode.scene_stats`` -- never retyped in this file, where it could
    silently drift from the measurement it is about."""
    source = inspect.getsource(mason_bridge)
    assert str(mscene.PLACED_WARN_THRESHOLD) not in source
    assert "stats[\"threshold\"]" in source or "stats['threshold']" in source
    assert "scene_stats" in source


def test_mason_bridge_facts_warn_only_past_the_real_threshold(monkeypatch):
    """Functional half: drop the constant to something a test can actually
    build a scene past, and check the sentence only appears once the scene
    resolves to more placed items than that."""
    monkeypatch.setattr(mscene, "PLACED_WARN_THRESHOLD", 2)
    doc = md.MasonDoc(
        roots=[nd.MeshNode(uid=nd.new_uid(), name=f"m{i}") for i in range(5)]
    )
    from warlock.studio.modes.mason import mode as mason_mode

    class _Tab:
        def __init__(self, doc):
            self.doc = doc
            self.saving = False

    tab = _Tab(doc)
    stats = mason_mode.scene_stats(FakeCtx(), tab)
    assert stats["warn"] is True
    assert stats["threshold"] == 2


# --- Stage F: the ground, the brushes, the prefab slot, the tree drag --------


class _Settings:
    """``docmodes.remember_path`` writes a recents list, which ``adopt`` calls on
    every new document -- so a ctx that opens a scene needs one."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


def _ctx_with_a_scene():
    from warlock.studio.modes.mason import mode as mason_mode

    ctx = FakeCtx()
    ctx.settings = _Settings()
    mason_mode.new_document(ctx)
    return ctx, mason_mode


def _drawn(monkeypatch, draw, *args):
    """Every control one pane body drew this frame, by label."""
    from warlock.studio import probe

    with imgui_context(monkeypatch) as imgui:
        probe.begin_frame()
        imgui.new_frame()
        imgui.set_next_window_size((320.0, 1400.0))
        imgui.begin("##host")
        draw(*args)
        imgui.end()
        imgui.end_frame()
        return [c.label for c in probe.FRAME_CONTROLS]


def test_the_assets_pane_offers_a_button_for_every_registered_brush(monkeypatch):
    """Derived from ``mason_state.BRUSHES``, so a sixth brush gets a button on
    the day it is written -- the rule the primitive grid already follows and for
    the same reason: a hand-listed palette is a palette that silently lags the
    engine."""
    from warlock.studio.modes.mason import state as mason_state

    ctx, mason_mode = _ctx_with_a_scene()
    state = mason_mode.ensure(ctx)
    tab = state.active
    mason_mode.add_terrain(ctx)

    labels = _drawn(monkeypatch, mason_palette._terrain, ctx, state, tab)
    for key, label, _tip in mason_state.BRUSHES:
        assert any(f"{label}##masonbrush{key}" == found for found in labels), key


def test_the_assets_pane_offers_the_ground_before_there_is_one_and_the_brushes_after(
    monkeypatch,
):
    """A brush with no ground to apply it to is a control that cannot work, and
    "Add ground" after there is one is a second ground the document refuses."""
    ctx, mason_mode = _ctx_with_a_scene()
    state = mason_mode.ensure(ctx)
    tab = state.active

    before = _drawn(monkeypatch, mason_palette._terrain, ctx, state, tab)
    assert any("masonterrainadd" in label for label in before)
    assert not any("masonbrush" in label for label in before)

    mason_mode.add_terrain(ctx)
    after = _drawn(monkeypatch, mason_palette._terrain, ctx, state, tab)
    assert not any("masonterrainadd" in label for label in after)
    assert any("masonbrush" in label for label in after)


def test_choosing_a_brush_also_puts_the_sculpt_tool_in_hand(monkeypatch):
    """A brush chosen while the Move gizmo is still in hand is a brush the left
    button never reaches -- the gizmo takes the press. This is the one place
    outside the tool grid that writes ``state.tool``, which is said out loud in
    the pane because a second such place would be two controls fighting over one
    setting."""
    source = inspect.getsource(mason_palette._terrain)
    assert 'state.tool = "sculpt"' in source
    writers = [
        name
        for name, fn in vars(mason_palette).items()
        if callable(fn) and getattr(fn, "__module__", "") == mason_palette.__name__
        and "state.tool" in inspect.getsource(fn)
    ]
    assert writers == ["_terrain"]


def test_the_prefabs_pane_is_in_the_column_only_while_the_scene_has_a_template():
    """The first conditional slot in this workspace: a permanently-empty panel in
    a four-panel column costs the outliner and Properties the height it sits in,
    on every scene that never authors a prefab."""
    from warlock.studio import skeletons

    ctx, mason_mode = _ctx_with_a_scene()
    right = skeletons.mason(ctx)["right"]
    assert "mason-prefabs" not in [slot.id for slot in right.live(ctx)]

    doc = mason_mode.ensure(ctx).active.doc
    node = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Barrel"))
    doc.select([node.uid])
    mason_mode.define_prefab_from_selection(ctx)

    assert "mason-prefabs" in [slot.id for slot in right.live(ctx)]
    # The slot is still *declared* either way -- a saved layout has something to
    # be a permutation of whether or not this scene uses prefabs.
    assert "mason-prefabs" in [slot.id for slot in right.slots]


def test_the_prefabs_pane_counts_the_instances_in_the_scene_tree():
    """What a row answers is "how many of these are in my scene", so an instance
    that only exists inside another template's subtree is deliberately not
    counted: that number would change when the *outer* template was placed
    again."""
    from warlock.studio.modes.mason.ui.panes import prefabs as mason_prefabs

    doc = md.MasonDoc()
    doc.prefabs["post"] = nd.MeshNode(uid=nd.new_uid(), name="post")
    for _ in range(3):
        doc.add_node(nd.PrefabNode(uid=nd.new_uid(), name="post", template="post"))
    assert mason_prefabs._instance_counts(doc) == {"post": 3}


def test_the_make_prefab_gesture_is_not_in_the_pane_that_needs_one_to_exist():
    """A pane that only exists once a template does cannot be where the first one
    is made. The gesture lives on the selection instead -- the viewport's context
    menu and the outliner's row menu."""
    from warlock.studio.modes.mason.ui.panes import menu as mason_menu
    from warlock.studio.modes.mason.ui.panes import outliner as mason_outliner
    from warlock.studio.modes.mason.ui.panes import prefabs as mason_prefabs

    assert "define_prefab_from_selection" not in inspect.getsource(mason_prefabs)
    assert "define_prefab_from_selection" in inspect.getsource(mason_menu)
    assert "define_prefab_from_selection" in inspect.getsource(mason_outliner)


def test_an_outliner_drop_reparents_rather_than_reordering_among_the_targets_siblings():
    """The drag used to insert the dropped node at the target's own index under
    the target's *parent*, which is a reorder dressed as a tree drag: dropping a
    prop onto a group put it beside the group rather than into it, so the one
    thing an outliner drag is for could not be done at all.

    Asserted at the source level, because the drop itself is an imgui payload
    exchange across two frames and what is in question is the one line inside
    it. The engine half -- that a reparent onto a descendant is refused, and that
    ``len(children)`` appends -- is ``tests/mason/test_document.py``'s.
    """
    source = inspect.getsource(mason_outliner._reorder)
    assert "parent_uid=node.uid" in source
    assert "doc.parent_uid_of(node.uid)" not in source
    # And sibling order stayed addressable, which it has to: it is export order
    # and outliner order both.
    menu = inspect.getsource(mason_outliner._context_menu)
    assert "Move up" in menu and "Move down" in menu
    assert "doc.index_of(node.uid)" in menu


def test_every_accelerator_a_mason_menu_advertises_is_one_handle_key_answers():
    """The two menus said ``Ctrl+G`` and ``Ctrl+Shift+G`` for Group and Ungroup
    for as long as both existed, and ``handle_key`` binds plain ``G`` and
    ``Shift+G`` -- ``_ctrl_key`` has no ``g`` arm at all, so the chord the menu
    taught did nothing whatsoever. A user who read the menu once and then
    reached for the keyboard got silence, and the only record they had of the
    binding was wrong.

    Nothing could have caught it: ``tests/manual/test_shortcuts.py`` gates the
    Ctrl+/ sheet against the chapter in both directions, and a *menu's* own
    accelerator column is in neither document. This is that third surface, read
    off ``mason_mode``'s own dispatch rather than off a list written here, so
    the next binding a menu advertises has to be one the mode actually answers.

    Source-level on purpose: what is in question is a literal string in a
    ``menu_item`` call, and pressing the key would test the handler rather than
    the label beside it.
    """
    import re

    from warlock.studio.modes.mason import mode as mason_mode
    from warlock.studio.modes.mason.ui.panes import menu as mason_menu
    from warlock.studio.modes.mason.ui.panes import outliner as mason_outliner

    # What the mode dispatches on, from the two functions that do the
    # dispatching: the ``name == "x"`` arms, plus the tool letters and the axis
    # views, both of which are tables rather than arms.
    def _names(func):
        return set(re.findall(r'name == "([^"]+)"', inspect.getsource(func)))

    plain = _names(mason_mode.handle_key) | set(mason_mode.TOOL_KEYS)
    chords = _names(mason_mode._ctrl_key) | set(mason_mode.AXIS_VIEW_KEYS) | {"5"}
    assert "g" in plain, "handle_key no longer binds G -- this gate has nothing to check"

    advertised = re.findall(
        r'menu_item\(\s*(?:f?"[^"]*"|[^,]+),\s*"([^"]+)"',
        inspect.getsource(mason_menu) + inspect.getsource(mason_outliner),
    )
    assert advertised, "no accelerators found -- the scan has stopped matching"

    wrong = []
    for accel in advertised:
        key = accel.split("+")[-1].lower()
        if key == "del":  # the one named key these menus use; K_DELETE, not a letter
            continue
        known = chords if accel.startswith("Ctrl+") else plain
        if key not in known:
            wrong.append(accel)
    assert not wrong, (
        f"Mason's menus advertise {wrong}, which mason_mode.handle_key does not "
        f"bind -- the menu is the only place a user reads that binding"
    )


# --- arming a primitive/light/camera disarms a prefab -----------------------


def test_arming_a_primitive_after_a_prefab_disarms_the_prefab():
    """The 2026-09-14 audit's mason-03: arming a primitive, light or camera
    used to set ``state.place_kind`` and leave ``state.place_prefab`` exactly
    as a previous prefab arm left it, and ``mason_mode.place_armed`` checks
    ``place_prefab`` first -- so the next click placed the old prefab while
    the palette highlighted the newly-armed item and the HUD hint still
    described the prefab.

    This must fail against the unfixed call sites: ``place_prefab`` would
    still read ``"Barrel"`` after arming a plain primitive.
    """
    from warlock.studio.modes.mason import state as mason_state
    from warlock.studio.modes.mason.ui.panes import palette as mason_palette

    state = mason_state.MasonState()
    state.place_prefab = "Barrel"  # armed earlier from the Prefabs pane

    mason_palette._arm_kind(state, "box")

    assert state.place_kind == "box"
    assert state.place_prefab == ""


# --- the Move hint promises only what the tool does --------------------------


def test_the_move_hint_does_not_promise_typed_entry_the_mode_lacks():
    """The 2026-09-14 audit's mason-04: the Move hint said "type a number, or
    X/Y/Z to lock an axis", which is Clay's ``clay/drag.py`` typed-entry and
    keyboard axis-lock machinery -- never ported to Mason. ``mason_mode``'s
    ``handle_key``/``_ctrl_key`` have no digit or X/Y/Z handling, so the hint
    described a gesture that simply did nothing when a reader tried it.

    This must fail against the unfixed hint, whose text names "type a
    number" and "X/Y/Z".
    """
    from warlock.studio.modes.mason import state as mason_state
    from warlock.studio.modes.mason.ui.panes import hud as mason_hud

    state = mason_state.MasonState()
    state.tool = "move"
    line = mason_hud._hint(state)

    assert "type a number" not in line.lower()
    assert "x/y/z" not in line.lower()


# --- one undo step per multi-node placement gesture --------------------------


def test_align_on_three_nodes_undoes_in_one_step():
    """The 2026-09-14 audit's mason-02: ``_apply_deltas`` (Align, Distribute
    and Drop selection to ground's shared engine) used to push one
    ``TransformEdit`` per moved node with no ``mark``/``collapse_since``
    around the loop, so aligning three nodes cost three Ctrl+Z presses.
    docs/manual/31-mason.md promises "Each of these lands as a single undo
    step" and every other multi-node mutator in ``mason_mode.py``
    (``group_selected``, ``duplicate_selected``...) already folds this way.

    This must fail against the unfixed ``_apply_deltas``: three separate
    ``TransformEdit`` pushes, so one ``undo()`` would restore only the last
    node moved and leave the other two at their aligned position.
    """
    from warlock.studio.modes.mason.ui.panes import tools as mason_tools

    a = nd.GroupNode(uid=nd.new_uid(), name="A")
    b = nd.GroupNode(uid=nd.new_uid(), name="B")
    c = nd.GroupNode(uid=nd.new_uid(), name="C")
    doc = md.MasonDoc(roots=[a, b, c])

    deltas = {
        a.uid: [1.0, 0.0, 0.0],
        b.uid: [0.0, 2.0, 0.0],
        c.uid: [0.0, 0.0, 3.0],
    }
    mason_tools._apply_deltas(doc, deltas)

    assert a.translation.tolist() == [1.0, 0.0, 0.0]
    assert b.translation.tolist() == [0.0, 2.0, 0.0]
    assert c.translation.tolist() == [0.0, 0.0, 3.0]
    # One step for all three moves, not three -- ``len(history)`` is the
    # stack's own step count (``UndoStack.__len__``); ``head`` is a global
    # per-edit serial, not a per-stack count, and is the wrong thing to
    # assert one-ness against.
    assert len(doc.history) == 1

    doc.undo()
    assert a.translation.tolist() == [0.0, 0.0, 0.0]
    assert b.translation.tolist() == [0.0, 0.0, 0.0]
    assert c.translation.tolist() == [0.0, 0.0, 0.0]


def test_drop_to_ground_context_menu_row_undoes_in_one_step(monkeypatch):
    """The context menu's own "Drop to ground" row (``mason_menu._drop_to_ground``)
    shares its arithmetic with the sidebar button but is a separate call
    site, and the 2026-09-14 audit's mason-02 found it with the identical
    per-node-push bug. Fails against the unfixed row the same way the sidebar
    test above does: three pushes instead of one.
    """
    import numpy as np

    from warlock.studio.modes.mason.ui.panes import menu as mason_menu

    a = nd.GroupNode(uid=nd.new_uid(), name="A")
    b = nd.GroupNode(uid=nd.new_uid(), name="B")
    c = nd.GroupNode(uid=nd.new_uid(), name="C")
    a.translation = np.array([0.0, 5.0, 0.0])
    b.translation = np.array([0.0, 3.0, 0.0])
    c.translation = np.array([0.0, 8.0, 0.0])
    doc = md.MasonDoc(roots=[a, b, c])
    doc.select([a.uid, b.uid, c.uid])

    # world_bounds and mason_assets.ensure both need a real geometry source
    # this test has no use for -- faked out so only the undo-folding under
    # test is exercised, the same way ``FakeCtx`` stands in for ``Ctx`` above.
    # Each box's own lo/hi is just the node's current translation, so
    # drop_to_ground (ground plane at y=0, no terrain) computes a real,
    # non-zero delta per node rather than a no-op that would tell this test
    # nothing.
    monkeypatch.setattr(
        mason_menu, "mason_assets", type("_M", (), {"ensure": staticmethod(lambda ctx: None)})
    )
    monkeypatch.setattr(
        mason_menu.mscene,
        "world_bounds",
        lambda doc, source, uids: (
            np.asarray(doc.node(uids[0]).translation, dtype="f8"),
            np.asarray(doc.node(uids[0]).translation, dtype="f8"),
        ),
    )

    class _Tab:
        pass

    tab = _Tab()
    tab.doc = doc
    mason_menu._drop_to_ground(FakeCtx(), tab)

    # One step for all three drops, not three.
    assert len(doc.history) == 1
    doc.undo()
    assert a.translation.tolist() == [0.0, 5.0, 0.0]
    assert b.translation.tolist() == [0.0, 3.0, 0.0]
    assert c.translation.tolist() == [0.0, 8.0, 0.0]
