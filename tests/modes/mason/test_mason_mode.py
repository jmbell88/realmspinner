"""Mason's controller: the rules Clay had to learn first, inherited on purpose.

Nothing here is about the viewport. It is about the document layer: a save
that is a *state* rather than a call that returns, a failed save that must
clear it, an encode that never runs on the calling thread, and a scene-size
warning read off the engine's own constant rather than a number in a pane.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from warlock.studio.modes.mason import mode as mason_mode
from warlock.studio.modes.mason import state as mason_state
from warlock.studio.modes.mason.engine import document as md
from warlock.studio.modes.mason.engine import nodes as nd
from warlock.studio.modes.mason.engine import scene as msc


class FakeCtx:
    """Runs a submitted callable inline, so the test sees what the task thread
    would have done without needing one."""

    def __init__(self, *, accept: bool = True) -> None:
        self.svc = None
        self.state = _AppState()
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.accept = accept
        self.result: Any = None

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        if not self.accept:
            return False
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info", *_args: Any, **_kwargs: Any) -> None:
        self.toasts.append((message, kind))


class _AppState:
    def __init__(self) -> None:
        self.mason = None
        self.mode = "home"


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _Done:
    def __init__(self, key: str, result: Any = None) -> None:
        self.key = key
        self.result = result


def _tab(ctx: FakeCtx, *, dirty: bool = False) -> mason_state.MasonTab:
    """One open scene. ``adopt`` records the head it is given, so a tab is
    clean the moment it is adopted -- dirtying it means editing it afterwards."""
    doc = md.MasonDoc()
    doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="Group"))
    tab = mason_mode.adopt(ctx, doc, title="Scene")
    if dirty:
        doc.set_props(doc.roots[0].uid, name="Edited")
    return tab


def _save(ctx: FakeCtx, tab: mason_state.MasonTab, path: Path) -> None:
    """A whole save: the submit, and the result coming back. ``save_to`` only
    does the first half; a test that skipped ``on_task_done`` would be
    asserting against a half-finished save."""
    mason_mode.save_to(ctx, tab, path)
    mason_mode.on_task_done(ctx, _Done(f"mason-save:{tab.uid}", ctx.result))


def test_ask_open_focuses_an_already_open_scene_instead_of_forking_a_second_tab(
    tmp_path,
):
    """The 2026-09-18 audit (second run, finding mason-02): the ``mason-open``
    arm called ``adopt`` unconditionally, while ``open_path`` (drag-drop and
    recents) already checked ``state.find_path`` -- Clay closed the identical
    gap in its own dialog arm on 2026-09-12 (clay-02). Two tabs over one path
    race on save; whichever writes last silently discards the other's edits."""
    ctx = FakeCtx()
    path = tmp_path / "scene.wscn"
    path.write_bytes(b"")
    existing = _tab(ctx)
    existing.path = path

    doc = md.MasonDoc()
    doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="Group"))
    mason_mode.on_task_done(
        ctx, _Done("mason-open", {"doc": doc, "path": str(path), "title": "Scene"})
    )

    state = mason_mode.ensure(ctx)
    assert len(state.docs) == 1
    assert state.active is existing


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


# --- opening ------------------------------------------------------------------


def test_a_new_document_adopts_and_is_not_dirty() -> None:
    ctx = FakeCtx()
    tab = mason_mode.new_document(ctx)
    assert tab.dirty is False
    assert mason_mode.active(ctx) is tab


def test_an_edit_makes_it_dirty_and_undo_makes_it_clean_again() -> None:
    ctx = FakeCtx()
    tab = mason_mode.new_document(ctx)
    node = nd.GroupNode(uid=nd.new_uid(), name="Group")
    tab.doc.add_node(node)
    assert tab.dirty is True
    mason_mode.undo(ctx, tab)
    assert tab.dirty is False


# --- saving is a state --------------------------------------------------------


def test_a_failed_save_clears_the_saving_state() -> None:
    """``saving`` disables every control that changes the document, so
    without this one failed write makes the tab read-only forever with no
    way back short of closing it."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True

    mason_mode.on_task_failed(ctx, _Done(f"mason-save:{tab.uid}"))
    assert tab.saving is False


def test_a_submit_that_is_refused_clears_the_saving_state() -> None:
    ctx = FakeCtx(accept=False)
    tab = _tab(ctx)
    mason_mode.save_as(ctx, tab)
    assert tab.saving is False


def test_a_completed_save_clears_it_and_marks_the_tab_saved(tmp_path) -> None:
    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)
    assert tab.dirty is True

    _save(ctx, tab, tmp_path / "scene.wscn")
    assert tab.saving is False
    assert tab.dirty is False
    assert (tmp_path / "scene.wscn").exists()


def test_save_submits_under_a_mason_key_and_never_encodes_on_the_calling_thread(
    tmp_path,
) -> None:
    """The submitted key carries the ``mason-`` prefix the app claims results
    by, and the file appears only once the (inline, in this fake) task runs --
    never before ``ctx.submit`` is called."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    path = tmp_path / "scene.wscn"

    mason_mode.save_to(ctx, tab, path)
    assert ctx.submitted == [f"mason-save:{tab.uid}"]
    assert path.exists()  # the fake ran the task inline; a real one would not have yet


# --- scene stats ---------------------------------------------------------------


def test_scene_stats_reads_the_threshold_from_the_engine_constant() -> None:
    ctx = FakeCtx()
    tab = _tab(ctx)
    stats = mason_mode.scene_stats(ctx, tab)
    assert stats["threshold"] == msc.PLACED_WARN_THRESHOLD
    assert stats["warn"] is False


def test_scene_stats_warn_flag_flips_at_the_threshold() -> None:
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    # One group already exists from ``_tab``; groups draw nothing, so use
    # mesh nodes with no ref -- ``resolve`` still counts an unresolved node as
    # placed (it draws nothing, but it is there).
    for _ in range(msc.PLACED_WARN_THRESHOLD):
        doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="m"))
    stats = mason_mode.scene_stats(ctx, tab)
    assert stats["placed"] >= msc.PLACED_WARN_THRESHOLD
    assert stats["warn"] is True


def test_scene_stats_resolves_once_per_revision(monkeypatch) -> None:
    """The 2026-09-14 audit's mason-05: ``mason_bridge._facts`` and
    ``mason_hud.stats_overlay`` each call ``scene_stats`` every frame with no
    memo of its own, so two panes reading it in the same frame ran
    ``scene.resolve(doc, include_hidden=True)`` twice over -- about triple the
    resolve cost at ``PLACED_WARN_THRESHOLD`` once ``MasonView.resolved()``'s
    own memo is counted in.

    This must fail against the unfixed ``scene_stats``: calling it twice with
    no edit between calls would run ``resolve`` twice, not once.
    """
    ctx = FakeCtx()
    tab = _tab(ctx)

    calls = []
    real_resolve = msc.resolve

    def counting_resolve(doc, **kwargs):
        calls.append(1)
        return real_resolve(doc, **kwargs)

    monkeypatch.setattr(msc, "resolve", counting_resolve)

    mason_mode.scene_stats(ctx, tab)
    mason_mode.scene_stats(ctx, tab)
    assert len(calls) == 1, "two same-revision calls should share one resolve"

    # An edit bumps doc.rev, and the next call is entitled to a fresh resolve.
    tab.doc.add_node(nd.GroupNode(uid=nd.new_uid()))
    mason_mode.scene_stats(ctx, tab)
    assert len(calls) == 2


# --- keys ----------------------------------------------------------------------


def test_handle_key_answers_false_with_nothing_open() -> None:
    ctx = FakeCtx()
    import pygame

    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q, mod=0)
    assert mason_mode.handle_key(ctx, event) is False


def test_mutating_ctrl_keys_are_all_actually_dispatched() -> None:
    """The 2026-09-14 audit's mason-07: ``_MUTATING_CTRL`` used to also list
    "i" and "m", which no arm of ``_ctrl_key`` dispatches and which
    shortcuts.py's Mason table never advertises either -- gating a key
    against writing-in-flight that nothing could ever actually press.

    Source-level, ``test_every_accelerator_a_mason_menu_advertises_is_one_handle_key_answers``'s
    own reason: what is in question is a literal set of key names, not a
    press, and this must fail against the unfixed table -- "i" and "m" are
    members with no matching ``elif name == "i"``/``elif name == "m"`` arm.
    """
    import inspect
    import re

    source = inspect.getsource(mason_mode._ctrl_key)
    dispatched = set(re.findall(r'elif name == "([^"]+)"', source))
    dispatched |= set(mason_mode.AXIS_VIEW_KEYS) | {"5"}

    undispatched = mason_mode._MUTATING_CTRL - dispatched
    assert not undispatched, (
        f"_MUTATING_CTRL lists {sorted(undispatched)}, which _ctrl_key "
        "dispatches nothing for"
    )


# --- placing --------------------------------------------------------------------


def test_placing_a_primitive_selects_it() -> None:
    ctx = FakeCtx()
    tab = _tab(ctx)
    uid = mason_mode.place_primitive(ctx, "box")
    assert uid is not None
    assert tab.doc.selection == {uid}


# --- the journal ---------------------------------------------------------------


def test_the_journal_provider_round_trips_a_document() -> None:
    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)

    encoded = mason_mode.JOURNAL.encode(tab)
    assert isinstance(encoded, bytes) and encoded

    from warlock.studio.modes.mason.engine import serialize

    doc = serialize.read_wscn(encoded)
    assert len(doc.roots) == len(tab.doc.roots)


# --- Stage F: the armed placement a click actually performs -------------------


def _armed_ctx(tab: Any = None) -> Any:
    """A ctx with a scene open and the mode state reachable -- Stage F's own
    tests place into a document rather than merely arming one."""
    ctx = FakeCtx()
    mason_mode.new_document(ctx)
    return ctx


def test_arming_a_primitive_and_clicking_places_it_where_the_click_was() -> None:
    """**The gap Stage E left and this closes.** ``state.place_kind`` was
    written by the Assets pane and read by nothing but the hint line, so arming
    a primitive, a light or a camera and clicking in the viewport placed nothing
    at all -- the pane's whole lights/camera/primitive half was inert.
    """
    ctx = _armed_ctx()
    state = mason_mode.ensure(ctx)
    doc = state.active.doc
    state.place_kind = "box"

    uid = mason_mode.place_armed(ctx, (2.0, 0.0, -5.0))

    assert uid is not None
    node = doc.node(uid)
    assert isinstance(node, nd.MeshNode)
    assert node.ref is not None
    assert list(node.translation) == [2.0, 0.0, -5.0]


def test_arming_a_light_or_a_camera_places_that_kind_and_not_a_primitive() -> None:
    """The dispatch is a table keyed on the pane's own arming prefixes, so the
    three kinds cannot silently collapse into one -- ``place_primitive`` would
    have been handed ``"light:spot"`` as a generator name."""
    ctx = _armed_ctx()
    state = mason_mode.ensure(ctx)
    doc = state.active.doc

    state.place_kind = "light:spot"
    light = doc.node(mason_mode.place_armed(ctx, (0.0, 2.0, 0.0)))
    state.place_kind = "camera"
    camera = doc.node(mason_mode.place_armed(ctx, (0.0, 2.0, 0.0)))

    assert isinstance(light, nd.LightNode) and light.kind == "spot"
    assert isinstance(camera, nd.CameraNode)


def test_placing_at_a_point_is_one_undo_step_and_not_an_add_then_a_move() -> None:
    """A first Ctrl+Z must take the new node away, not leave it sitting at the
    origin. ``collapse_since``'s mark is taken *before* the add for exactly
    this, and a mark taken between the two would pass every other assertion
    here."""
    ctx = _armed_ctx()
    state = mason_mode.ensure(ctx)
    doc = state.active.doc
    state.place_kind = "box"
    before = len(doc.history)

    uid = mason_mode.place_armed(ctx, (7.0, 0.0, 0.0))
    # ``len`` and not ``head``: head is a per-edit *serial*, which a two-push
    # gesture also advances by one.
    assert len(doc.history) == before + 1

    doc.undo()
    assert doc.node(uid) is None


def test_nothing_armed_places_nothing() -> None:
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    assert mason_mode.place_armed(ctx, (1.0, 1.0, 1.0)) is None
    assert doc.roots == []


def test_an_armed_placement_survives_the_click_so_a_row_of_props_is_one_arming() -> None:
    """Placing six fence posts is one arming and six clicks. A palette that
    disarmed itself after the first would make the other five a trip back to
    the sidebar each; Esc is what clears it, and the hint line says so."""
    ctx = _armed_ctx()
    state = mason_mode.ensure(ctx)
    state.place_kind = "box"
    for x in range(3):
        mason_mode.place_armed(ctx, (float(x), 0.0, 0.0))
    assert state.place_kind == "box"
    assert len(state.active.doc.roots) == 3


# --- the ground: both halves, one step ---------------------------------------


def test_adding_a_ground_gives_the_document_the_field_and_its_node_together() -> None:
    """Either half alone is a state nothing in the app can act on: a ``terrain``
    with no node has no outliner row, no transform and no export, and a node
    with no ``terrain`` draws and picks nothing."""
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    steps = len(doc.history)

    uid = mason_mode.add_terrain(ctx)

    assert doc.terrain is not None
    assert isinstance(doc.node(uid), nd.TerrainNode)
    # One step, so one Ctrl+Z takes the ground away whole rather than leaving
    # a node with no field under it.
    assert len(doc.history) == steps + 1
    doc.undo()
    assert doc.terrain is None
    assert doc.node(uid) is None


def test_a_second_ground_is_refused_because_two_grounds_is_two_ground_planes() -> None:
    ctx = _armed_ctx()
    mason_mode.add_terrain(ctx)
    assert mason_mode.add_terrain(ctx) is None


def test_removing_the_ground_is_undoable_with_its_heights_intact() -> None:
    """The incident ``TerrainSwapEdit`` exists for: clearing a terrain used to
    discard however long someone had spent sculpting it, with Ctrl+Z having
    nothing left to restore."""
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    mason_mode.add_terrain(ctx)
    doc.begin_sculpt()
    doc.sculpt((0, 0, 2, 2), np.full((2, 2), 3.5, dtype="f4"))
    doc.end_sculpt()
    sculpted = doc.terrain.heights.copy()

    assert mason_mode.remove_terrain(ctx) is True
    assert doc.terrain is None
    doc.undo()
    assert doc.terrain is not None
    assert np.array_equal(doc.terrain.heights, sculpted)


# --- prefabs: authored, and the instance is what the user keeps ---------------


def test_making_a_prefab_leaves_the_selection_as_an_instance_of_it() -> None:
    """The authoring half, and the reason it is not just ``define_prefab``:
    without replacing the scene node, "make prefab" would leave the thing the
    user is looking at untracked by the template it was just made from, and
    editing that template would visibly change every instance *except* that
    one.
    """
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    node = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Barrel"))
    doc.select([node.uid])

    name = mason_mode.define_prefab_from_selection(ctx)

    assert name == "Barrel"
    assert "Barrel" in doc.prefabs
    assert doc.node(node.uid) is None
    instances = [n for n in doc.all_nodes() if isinstance(n, nd.PrefabNode)]
    assert len(instances) == 1
    assert instances[0].template == "Barrel"
    assert doc.selection == {instances[0].uid}


def test_making_a_prefab_of_an_instance_or_of_the_ground_is_refused() -> None:
    """An instance of an instance is the recursion ``define_prefab`` refuses at
    the door, and the ground's large array must not be copied into a template at
    all -- both refused here so they read as a disabled gesture rather than as an
    exception on the frame thread."""
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    node = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Barrel"))
    doc.select([node.uid])
    mason_mode.define_prefab_from_selection(ctx)
    assert mason_mode.define_prefab_from_selection(ctx) == ""

    mason_mode.add_terrain(ctx)
    assert mason_mode.define_prefab_from_selection(ctx) == ""


def test_placing_an_instance_of_a_template_that_does_not_exist_is_refused() -> None:
    """A ``PrefabNode`` naming nothing is legal in a *loaded* document and
    resolves as dangling; minting one on purpose would be authoring that case."""
    ctx = _armed_ctx()
    assert mason_mode.place_prefab(ctx, "nothing-by-that-name") is None


def test_three_instances_of_one_template_track_one_edit_to_it() -> None:
    """The whole point of a prefab: there is no propagation step and no "apply
    to instances" button, because the walk reads through ``doc.prefabs`` every
    time."""
    ctx = _armed_ctx()
    state = mason_mode.ensure(ctx)
    doc = state.active.doc
    node = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Post"))
    doc.select([node.uid])
    mason_mode.define_prefab_from_selection(ctx)
    for x in (1.0, 2.0):
        state.place_prefab = "Post"
        mason_mode.place_armed(ctx, (x, 0.0, 0.0))

    placed = msc.resolve(doc)
    meshes = [p for p in placed if isinstance(p.node, nd.MeshNode)]
    assert len(meshes) == 3

    # Edit the template itself: every instance follows on the next resolve.
    doc.prefabs["Post"].scale = np.array([2.0, 2.0, 2.0], dtype="f8")
    doc.touch()
    for item in msc.resolve(doc):
        if isinstance(item.node, nd.MeshNode):
            assert item.world[0, 0] == pytest.approx(2.0)


def test_unpacking_an_instance_replaces_it_with_an_independent_copy() -> None:
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    node = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Post"))
    doc.select([node.uid])
    mason_mode.define_prefab_from_selection(ctx)
    instance_uid = next(iter(doc.selection))

    mason_mode.unpack_selected(ctx)

    assert doc.node(instance_uid) is None
    fresh = doc.node(next(iter(doc.selection)))
    assert isinstance(fresh, nd.MeshNode)
    # The template is untouched -- unpacking one instance is not removing the
    # prefab.
    assert "Post" in doc.prefabs


# --- hierarchy ----------------------------------------------------------------


def test_ungroup_lifts_children_into_the_group_s_own_place_as_one_step() -> None:
    """Sibling order is export order and outliner order both, so the children
    land where the group sat rather than at the end of their new parent's list."""
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    first = doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="first"))
    a = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="a"))
    b = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="b"))
    doc.select([a.uid, b.uid])
    mason_mode.group_selected(ctx)
    group_uid = next(iter(doc.selection))
    steps = len(doc.history)

    mason_mode.ungroup_selected(ctx)

    assert doc.node(group_uid) is None
    assert [n.uid for n in doc.roots] == [first.uid, a.uid, b.uid]
    assert len(doc.history) == steps + 1
    doc.undo()
    assert doc.node(group_uid) is not None


def test_ungroup_does_nothing_to_a_selection_with_no_group_in_it() -> None:
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    node = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="a"))
    doc.select([node.uid])
    steps = len(doc.history)
    mason_mode.ungroup_selected(ctx)
    assert len(doc.history) == steps


# --- Esc means the nearer of its two jobs ------------------------------------


# --- MAX_PLACED: every attach door refuses rather than crashing --------------


def test_place_primitive_refuses_with_a_toast_rather_than_crashing_past_max_placed(
    monkeypatch,
) -> None:
    """The 2026-09-18 audit's mason-01: ``place_ref`` (reached here through
    ``place_primitive``) called ``doc.add_node`` with no ``MAX_PLACED``
    pre-check, so the ``ValueError`` ``MasonDoc._check_max_placed`` raises at
    the ceiling escaped uncaught -- past ``App.run()``'s own catch-all
    (shell-04) -- ending the session with every open document's unsaved
    work. Against the unfixed function this call raises instead of toasting.
    """
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    monkeypatch.setattr(msc, "MAX_PLACED", 0)
    steps = len(doc.history)

    uid = mason_mode.place_primitive(ctx, "box")

    assert uid is None
    assert len(doc.all_nodes()) == 0
    assert len(doc.history) == steps
    assert any(kind == "error" for _msg, kind in ctx.toasts)


def test_place_light_refuses_with_a_toast_rather_than_crashing_past_max_placed(
    monkeypatch,
) -> None:
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    monkeypatch.setattr(msc, "MAX_PLACED", 0)

    uid = mason_mode.place_light(ctx, "point")

    assert uid is None
    assert len(doc.all_nodes()) == 0
    assert any(kind == "error" for _msg, kind in ctx.toasts)


def test_place_camera_refuses_with_a_toast_rather_than_crashing_past_max_placed(
    monkeypatch,
) -> None:
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    monkeypatch.setattr(msc, "MAX_PLACED", 0)

    uid = mason_mode.place_camera(ctx)

    assert uid is None
    assert len(doc.all_nodes()) == 0
    assert any(kind == "error" for _msg, kind in ctx.toasts)


def test_place_prefab_refuses_with_a_toast_rather_than_crashing_past_max_placed(
    monkeypatch,
) -> None:
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    node = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Barrel"))
    doc.select([node.uid])
    mason_mode.define_prefab_from_selection(ctx)
    before = len(doc.all_nodes())
    monkeypatch.setattr(msc, "MAX_PLACED", before)

    uid = mason_mode.place_prefab(ctx, "Barrel")

    assert uid is None
    assert len(doc.all_nodes()) == before
    assert any(kind == "error" for _msg, kind in ctx.toasts)


def test_group_selected_refuses_with_a_toast_rather_than_crashing_past_max_placed(
    monkeypatch,
) -> None:
    """``group_selected`` still adds one new (empty) ``GroupNode`` even
    though the selection it wraps is not itself re-added -- a scene already
    sitting at the ceiling must refuse that one node too."""
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    node = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="a"))
    doc.select([node.uid])
    monkeypatch.setattr(msc, "MAX_PLACED", len(doc.all_nodes()))
    steps = len(doc.history)

    mason_mode.group_selected(ctx)

    assert len(doc.history) == steps
    assert not any(isinstance(n, nd.GroupNode) for n in doc.all_nodes())
    assert any(kind == "error" for _msg, kind in ctx.toasts)


def test_add_terrain_refuses_with_a_toast_rather_than_crashing_past_max_placed(
    monkeypatch,
) -> None:
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    monkeypatch.setattr(msc, "MAX_PLACED", 0)

    uid = mason_mode.add_terrain(ctx)

    assert uid is None
    assert doc.terrain is None
    assert any(kind == "error" for _msg, kind in ctx.toasts)


def test_duplicate_selected_refuses_with_a_toast_rather_than_crashing_past_max_placed(
    monkeypatch,
) -> None:
    """The 2026-09-18 audit's mason-01: ``duplicate_selected`` called
    ``doc.add_node`` in a loop with no ``MAX_PLACED`` pre-check, so a
    duplication that crossed the ceiling raised uncaught -- and left
    whatever copies had already been attached before the exception with the
    undo mark still open, uncollapsed. Reproduced directly against the real
    (unmodified) engine layer in this fix's scratch script
    (``prefix_repro.py``): calling ``doc.add_node`` in the same loop, with no
    guard, raises ``ValueError`` uncaught. Against the unfixed
    ``duplicate_selected`` this call raises instead of toasting and leaving
    the document untouched.
    """
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    node = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="a"))
    doc.select([node.uid])
    monkeypatch.setattr(msc, "MAX_PLACED", len(doc.all_nodes()))
    steps = len(doc.history)

    mason_mode.duplicate_selected(ctx)

    # Refused before ``mark()`` was even taken: no half-attached copy, no
    # uncollapsed undo step left behind.
    assert len(doc.all_nodes()) == 1
    assert len(doc.history) == steps
    assert doc.selection == {node.uid}
    assert any(kind == "error" for _msg, kind in ctx.toasts)


def test_unpack_selected_refuses_with_a_toast_rather_than_crashing_past_max_placed(
    monkeypatch,
) -> None:
    """The 2026-09-18 audit's mason-02: ``unpack_selected`` caught
    ``(KeyError, TypeError)`` but not the ``ValueError`` ``unpack_instance``'s
    own ``MAX_PLACED`` refusal raises (added by the 2026-09-15 audit's
    mason-01 for this exact attach point), so it escaped the same way
    mason-01's call sites did. Against the unfixed function this call raises
    instead of skipping the instance and toasting how many were skipped.

    The template must be *bigger* than the instance it replaces --
    ``unpack_instance`` refuses on net growth, and a one-node template
    unpacking a one-node instance is always net-zero (see
    ``test_unpack_instance_replacing_an_instance_with_a_same_sized_copy_never_refuses``
    in ``test_audit_2026_09_15_mason.py``), so a template with children is
    what makes the ceiling reachable at all.
    """
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    template = nd.GroupNode(uid=nd.new_uid(), name="big")
    template.children.append(nd.GroupNode(uid=nd.new_uid()))
    template.children.append(nd.GroupNode(uid=nd.new_uid()))
    doc.define_prefab("Big", template)
    instance = doc.add_node(nd.PrefabNode(uid=nd.new_uid(), name="Big", template="Big"))
    doc.select([instance.uid])
    monkeypatch.setattr(msc, "MAX_PLACED", len(doc.all_nodes()))  # unpack would grow it by 2
    steps = len(doc.history)

    mason_mode.unpack_selected(ctx)

    # Refused: the instance is untouched, nothing attached, nothing undoable.
    assert doc.node(instance.uid) is not None
    assert isinstance(doc.node(instance.uid), nd.PrefabNode)
    assert len(doc.history) == steps
    assert any(kind == "error" for _msg, kind in ctx.toasts)


def test_unpack_selected_unpacks_what_it_can_and_reports_only_what_it_skipped(
    monkeypatch,
) -> None:
    """A mixed selection -- one instance that fits under the ceiling, one
    that would cross it -- unpacks the first and only refuses the second,
    rather than the whole gesture failing (or crashing) over one instance."""
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    small = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Post"))
    doc.select([small.uid])
    mason_mode.define_prefab_from_selection(ctx)
    small_instance = next(iter(doc.selection))

    big_template = nd.GroupNode(uid=nd.new_uid(), name="big")
    big_template.children.append(nd.GroupNode(uid=nd.new_uid()))
    big_template.children.append(nd.GroupNode(uid=nd.new_uid()))
    doc.define_prefab("Big", big_template)
    big_instance = doc.add_node(nd.PrefabNode(uid=nd.new_uid(), name="Big", template="Big"))

    doc.select([small_instance, big_instance.uid])
    # One node of headroom: enough for the 1-for-1 small unpack, not enough
    # for the big template's net growth of +2.
    monkeypatch.setattr(msc, "MAX_PLACED", len(doc.all_nodes()) + 1)

    mason_mode.unpack_selected(ctx)

    assert doc.node(small_instance) is None  # unpacked
    assert isinstance(doc.node(big_instance.uid), nd.PrefabNode)  # skipped, still there
    assert any(kind == "error" for _msg, kind in ctx.toasts)


def test_escape_disarms_a_placement_before_it_clears_the_selection() -> None:
    """One key, two jobs, in the order the user means them: Esc after arming a
    light is "not that after all", and must not also throw away the selection
    they were about to place it beside."""
    import pygame

    ctx = _armed_ctx()
    state = mason_mode.ensure(ctx)
    doc = state.active.doc
    node = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="a"))
    doc.select([node.uid])
    state.place_kind = "light:point"

    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0)
    mason_mode.handle_key(ctx, event)
    assert state.place_kind == ""
    assert doc.selection == {node.uid}

    mason_mode.handle_key(ctx, event)
    assert doc.selection == set()
