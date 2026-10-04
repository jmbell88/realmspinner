"""Regressions for the 2026-10-03 audit's mason-14 .. mason-21 and mason-23
(the Medium findings in Mason's scene mode, panes, asset source and viewport).

Each test's name is the claim it makes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.studio import dialogs
from realmspinner.studio.modes.mason import assets as mason_assets
from realmspinner.studio.modes.mason import mode as mason_mode
from realmspinner.studio.modes.mason.engine import document as md
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import refs as mrefs
from realmspinner.studio.modes.mason.engine import scene as msc
from realmspinner.studio.modes.mason.engine import terrain as mterrain
from realmspinner.studio.modes.mason.ui import view as mason_view
from realmspinner.studio.modes.mason.ui.panes import menu as mason_menu
from realmspinner.studio.modes.mason.ui.panes import tools as mason_tools

from .test_mason_view import RECT, _Ctx, _Source


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class FakeCtx:
    def __init__(self) -> None:
        self.svc = None
        self.state = type("S", (), {"mason": None, "mode": "home"})()
        self.settings = _Settings()
        self.prompts = dialogs.PromptQueue()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, kind: str = "info", *_a: Any, **_k: Any) -> None:
        self.toasts.append((message, kind))


def _armed() -> tuple[FakeCtx, md.MasonDoc]:
    ctx = FakeCtx()
    mason_mode.new_document(ctx)
    return ctx, mason_mode.ensure(ctx).active.doc


def _mesh(name: str = "M", **kw: Any) -> nd.MeshNode:
    return nd.MeshNode(uid=nd.new_uid(), name=name, ref=mrefs.primitive_ref("box", {}), **kw)


class _BoxSource:
    """Every ref is a unit cube centred on its own origin."""

    rev = 0

    def box(self, ref: Any) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5])

    def primitives(self, ref: Any) -> list[gltf.Primitive]:
        return []


# --- mason-14 ----------------------------------------------------------------


def test_duplicating_a_selection_that_holds_a_group_and_its_child_copies_the_child_once():
    ctx, doc = _armed()
    group = doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="G"))
    doc.add_node(_mesh("a"), parent_uid=group.uid)
    doc.add_node(_mesh("b"), parent_uid=group.uid)
    doc.select([n.uid for n in doc.all_nodes()])
    assert len(doc.all_nodes()) == 3

    mason_mode.duplicate_selected(ctx)

    assert len(doc.all_nodes()) == 6
    # The original group still holds exactly its own two props.
    assert [c.name for c in doc.node(group.uid).children] == ["a", "b"]


# --- mason-15 ----------------------------------------------------------------


def _terrain_doc() -> tuple[FakeCtx, md.MasonDoc]:
    ctx, doc = _armed()
    mason_mode.add_terrain(ctx)
    return ctx, doc


def test_a_terrain_node_grouped_with_other_nodes_cannot_be_duplicated_or_orphaned():
    ctx, doc = _terrain_doc()
    doc.add_node(_mesh("prop"))
    doc.select([n.uid for n in doc.all_nodes()])

    mason_mode.group_selected(ctx)
    group_uid = next(iter(doc.selection))
    mason_mode.duplicate_selected(ctx)

    terrains = [n for n in doc.all_nodes() if isinstance(n, nd.TerrainNode)]
    assert len(terrains) == 1, "the ground must not export twice"

    doc.select([group_uid])
    mason_mode.delete_selected(ctx)
    terrains = [n for n in doc.all_nodes() if isinstance(n, nd.TerrainNode)]
    # Node and height field leave (or stay) together: never a node-less field
    # that makes Add ground refuse with no toast.
    assert (doc.terrain is None) == (not terrains)
    if doc.terrain is None:
        assert mason_mode.add_terrain(ctx) is not None


def test_a_terrain_node_cannot_be_moved_under_a_parent():
    ctx, doc = _terrain_doc()
    terrain = next(n for n in doc.all_nodes() if isinstance(n, nd.TerrainNode))
    group = doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="G"))

    with pytest.raises(ValueError):
        doc.move_node(terrain.uid, 0, parent_uid=group.uid)


def test_deleting_the_last_nested_terrain_node_clears_the_height_field():
    # A scene file can still carry a nested ground; deleting its ancestor must
    # not strand ``doc.terrain``.
    ctx, doc = _armed()
    group = doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="G"))
    doc.set_terrain(
        mterrain.Terrain(
            heights=np.zeros((5, 5), dtype="f4"),
            size_x=8.0,
            size_z=8.0,
            material=gltf.Material(name="g"),
        )
    )
    doc.add_node(nd.TerrainNode(uid=nd.new_uid(), name="T"), parent_uid=group.uid)
    doc.select([group.uid])

    mason_mode.delete_selected(ctx)

    assert doc.terrain is None


# --- mason-16 / mason-17 -------------------------------------------------------


def _prop_with_child() -> tuple[md.MasonDoc, nd.Node, nd.Node]:
    doc = md.MasonDoc()
    parent = doc.add_node(_mesh("P", translation=np.array([0.0, 5.0, 0.0])))
    child = doc.add_node(_mesh("C"), parent_uid=parent.uid)
    return doc, parent, child


def test_drop_to_ground_does_not_move_a_selected_child_twice_when_its_selected_parent_moves():
    doc, parent, child = _prop_with_child()
    doc.select([parent.uid, child.uid])
    ctx = FakeCtx()
    ctx.mason_assets = _BoxSource()

    mason_menu._drop_to_ground(ctx, type("T", (), {"doc": doc})())

    assert msc.resolved_for(doc, parent.uid).world[1, 3] == pytest.approx(0.5)
    assert msc.resolved_for(doc, child.uid).world[1, 3] == pytest.approx(0.5)


def test_world_boxes_resolves_the_scene_once_however_many_nodes_are_selected(monkeypatch):
    doc = md.MasonDoc()
    uids = [doc.add_node(_mesh(f"n{i}")).uid for i in range(12)]
    ctx = FakeCtx()
    ctx.mason_assets = _BoxSource()
    calls = {"n": 0}
    real = msc.resolve

    def counting(*a: Any, **k: Any):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(msc, "resolve", counting)

    boxes = mason_tools._world_boxes(ctx, doc, uids)

    assert len(boxes) == 12
    assert calls["n"] == 1


# --- mason-18 ----------------------------------------------------------------


def test_make_prefab_does_not_silently_replace_another_template_of_the_same_name():
    ctx, doc = _armed()
    first = doc.add_node(_mesh("Barrel"))
    doc.select([first.uid])
    assert mason_mode.define_prefab_from_selection(ctx) == "Barrel"
    template = doc.prefabs["Barrel"]
    second = doc.add_node(
        nd.MeshNode(uid=nd.new_uid(), name="Barrel", ref=mrefs.primitive_ref("sphere", {}))
    )
    doc.select([second.uid])

    # The prompt offers a name that does not collide ...
    mason_mode.prompt_define_prefab_from_selection(ctx)
    assert ctx.prompts.pending is not None
    assert ctx.prompts.pending.value not in doc.prefabs
    # ... and typing the colliding one anyway is refused, not applied.
    assert mason_mode.define_prefab_from_selection(ctx, "Barrel") == ""

    assert doc.prefabs["Barrel"] is template
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert doc.node(second.uid) is not None, "the refused node stays a plain node"


# --- mason-19 ----------------------------------------------------------------


class _AssetCtx:
    def __init__(self, root: Path) -> None:
        self.svc = type(
            "Svc", (), {"config": type("C", (), {"job_dir": lambda s, j: root / j})()}
        )()
        self.submitted: list[str] = []

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kw: Any) -> bool:
        self.submitted.append(key)
        return True


def test_a_library_ref_with_a_path_traversing_job_id_resolves_to_missing_not_to_a_file_outside_the_data_dir(  # noqa: E501
    tmp_path,
):
    data = tmp_path / "data"
    data.mkdir()
    ctx = _AssetCtx(data)
    source = mason_assets.ensure(ctx)
    ref = mrefs.LibraryRef(job_id="..\\..\\outside", artifact="model.glb")

    assert source.primitives(ref) == []

    assert ctx.submitted == [], "no parse may be started for a traversing job id"
    assert mrefs.ref_key(ref) in source.missing
    # An artifact naming a path is refused the same way, behind a valid id.
    bad = mrefs.LibraryRef(job_id="0123456789ab", artifact="..\\..\\x.glb")
    assert source.primitives(bad) == []
    assert ctx.submitted == []
    assert mrefs.ref_key(bad) in source.missing


# --- mason-20 ----------------------------------------------------------------


def test_scene_stats_missing_count_updates_when_a_parse_fails_without_the_document_changing():
    ctx, doc = _armed()
    ref = mrefs.LibraryRef(job_id="0123456789ab", artifact="model.glb")
    doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Lib", ref=ref))
    tab = mason_mode.ensure(ctx).active

    assert mason_mode.scene_stats(ctx, tab)["missing"] == 0
    rev = doc.rev
    # What ``MasonView.sync`` does when the background parse fails: it writes
    # ``doc.missing`` and bumps the source's rev, never the document's.
    doc.missing = {mrefs.ref_key(ref)}
    assert doc.rev == rev

    assert mason_mode.scene_stats(ctx, tab)["missing"] == 1


# --- mason-21 ----------------------------------------------------------------


@pytest.fixture
def view(gl):
    v = mason_view.MasonView(gl, _Ctx())
    yield v
    v.release()


def _ground() -> md.MasonDoc:
    doc = md.MasonDoc()
    doc.set_terrain(
        mterrain.Terrain(
            heights=np.zeros((17, 17), dtype="f4"),
            size_x=16.0,
            size_z=16.0,
            material=gltf.Material(name="ground"),
        )
    )
    doc.add_node(nd.TerrainNode(uid=nd.new_uid(), name="Terrain"))
    return doc


def test_a_wheel_tick_or_middle_press_mid_stroke_neither_ends_nor_strands_the_grab(view):
    doc = _ground()
    view._rect = RECT
    view.state.tool = "sculpt"
    view.state.brush = "raise"
    view.state.brush_radius = 4.0
    centre = (RECT[2] / 2.0, RECT[3] / 2.0)
    steps = len(doc.history)

    assert view._press(doc, _Source(), 1, centre) is True
    assert view._grab == "sculpt"

    # A middle press must not replace the stroke with a pan (which would leave
    # the session open with nothing to close it) ...
    view._press(doc, _Source(), 2, centre)
    assert view._grab == "sculpt"
    # ... and neither a middle release nor a wheel tick may end it early.
    view._release(doc, 2)
    view._release(doc, 5)
    assert view._grab == "sculpt"
    assert doc.sculpting is True

    view._release(doc, 1)
    assert view._grab is None
    assert doc.sculpting is False
    assert len(doc.history) == steps + 1


def test_a_wheel_tick_mid_gizmo_drag_does_not_commit_the_drag_early(view):
    doc = md.MasonDoc()
    view._grab = "gizmo"

    view._release(doc, 4)

    assert view._grab == "gizmo"


# --- mason-23 ----------------------------------------------------------------


def test_a_scene_larger_than_the_cache_budget_can_still_resolve_every_ref_for_export(
    monkeypatch,
):
    monkeypatch.setattr(mason_assets, "CACHE_BYTES", 1000)
    ctx = _AssetCtx(Path("."))
    source = mason_assets.ensure(ctx)
    doc = md.MasonDoc()
    refs = [
        mrefs.LibraryRef(job_id=f"{i:012x}", artifact="model.glb") for i in range(4)
    ]
    for ref in refs:
        doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="L", ref=ref))

    def prims() -> list[gltf.Primitive]:
        # 600 bytes of positions + 12 of indices: two of these exceed the budget.
        return [
            gltf.Primitive(
                positions=np.zeros((50, 3), dtype="f4"),
                indices=np.array([0, 1, 2], dtype="u4"),
            )
        ]

    source.pin_document(doc)
    for ref in refs:
        source._store(mrefs.ref_key(ref), prims())

    assert all(source.box(ref) is not None for ref in refs)
    assert all(source.primitives(ref) for ref in refs)
