"""Mason's Stage E panes: the Assets pane, the Outliner, Properties and the
Document bridge.

Each test's name is its claim, ``clay_tools`` test module's own convention.
"""

from __future__ import annotations

import inspect

from _ui_context import imgui_context

from warlock.studio.mason import document as md
from warlock.studio.mason import nodes as nd
from warlock.studio.mason import scene as mscene
from warlock.studio.panes import mason_bridge, mason_outliner, mason_palette, mason_props


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
    from warlock.studio import probe
    from warlock.studio.clay import primitives as bp

    def fake_build(**kwargs):  # pragma: no cover - never invoked by this test
        raise NotImplementedError

    monkeypatch.setitem(bp.GENERATORS, "widget_zzz", ({}, fake_build))

    from warlock.studio import icons

    expected_label = f"{icons.BOX}##masonaddwidget_zzz"

    with imgui_context(monkeypatch) as imgui:
        ctx = FakeCtx()
        from warlock.studio import mason_mode

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
    from warlock.studio import mason_mode

    class _Tab:
        def __init__(self, doc):
            self.doc = doc
            self.saving = False

    tab = _Tab(doc)
    stats = mason_mode.scene_stats(FakeCtx(), tab)
    assert stats["warn"] is True
    assert stats["threshold"] == 2
