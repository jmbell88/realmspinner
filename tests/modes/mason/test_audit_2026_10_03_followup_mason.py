"""The 2026-10-03 audit's mason-13 and mason-22 callers.

The engine fixes landed first (``TakenNames`` in ``gltfout``, ``terrain_world``
on ``ops.drop_to_ground`` and ``MasonDoc``); these are the three call sites that
still used the old shape: Export OBJ's naming set, the sidebar's "Drop selection
to ground" button and the drag-time ground snap.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.studio.modes.mason.engine import document as md
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import objout
from realmspinner.studio.modes.mason.engine import refs as mrefs
from realmspinner.studio.modes.mason.engine import scene as msc
from realmspinner.studio.modes.mason.engine import terrain as mterrain
from realmspinner.studio.modes.mason.ui import view as mason_view
from realmspinner.studio.modes.mason.ui.panes import tools as mason_tools


def _mesh(name: str = "M", **kw: Any) -> nd.MeshNode:
    return nd.MeshNode(uid=nd.new_uid(), name=name, ref=mrefs.primitive_ref("box", {}), **kw)


class _Source:
    """A one-triangle source that also answers the unit box ``world_bounds`` reads."""

    rev = 0

    def primitives(self, ref: Any) -> list[gltf.Primitive]:
        return [
            gltf.Primitive(
                positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"),
                indices=np.array([0, 1, 2], dtype="u4"),
                material=gltf.Material(name=""),
            )
        ]

    def box(self, ref: Any) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5])


# --- mason-13: Export OBJ names every node from a TakenNames -------------------


def _obj_naming_seconds(count: int) -> float:
    doc = md.MasonDoc(roots=[_mesh("Rock") for _ in range(count)])
    best = float("inf")
    for _ in range(2):
        start = time.perf_counter()
        jobs = objout.collect_jobs(doc, _Source())
        best = min(best, time.perf_counter() - start)
    names = [name for _item, name, _prims in jobs.jobs]
    assert len(set(names)) == len(names) == count
    return best


def test_obj_export_naming_cost_does_not_grow_quadratically_with_identical_names() -> None:
    small, large = 1500, 6000
    ratio = _obj_naming_seconds(large) / max(_obj_naming_seconds(small), 1e-6)
    # 4x the nodes: linear is ~4x, the old probe-from-.001 loop is ~16x.
    assert ratio < 9.0, f"OBJ-naming {large} identical nodes cost {ratio:.1f}x naming {small}"


def test_obj_export_names_a_few_thousand_identical_nodes_uniquely() -> None:
    doc = md.MasonDoc(roots=[_mesh("Rock") for _ in range(3000)])
    export = objout.obj_export(doc, _Source())
    lines = [ln for ln in export.files["scene.obj"].decode().splitlines() if ln.startswith("o ")]
    assert len(lines) == len(set(lines)) == 3000


# --- mason-22: the sidebar button and the drag snap follow a moved ground ----


def _doc_with_moved_ground() -> tuple[md.MasonDoc, nd.MeshNode]:
    doc = md.MasonDoc()
    ground = doc.add_node(nd.TerrainNode(uid=nd.new_uid(), name="Ground"))
    doc.set_terrain(
        mterrain.Terrain(
            heights=np.zeros((5, 5), dtype="f4"),
            size_x=8.0,
            size_z=8.0,
            material=gltf.Material(name="g"),
        )
    )
    doc.set_transform(ground.uid, translation=np.array([0.0, 10.0, 0.0]))
    prop = doc.add_node(_mesh("P", translation=np.array([0.0, 20.0, 0.0])))
    return doc, prop


def test_the_sidebar_drop_to_ground_button_lands_on_a_moved_ground() -> None:
    doc, prop = _doc_with_moved_ground()
    doc.select([prop.uid])

    # Press only the drop button; every other widget the pane draws is inert.
    original = {n: getattr(mason_tools, n) for n in ("imgui", "widgets", "controls", "_array")}
    widgets = SimpleNamespace(
        field_label=lambda *a, **k: None,
        grid_width=lambda *a, **k: 100,
        disabled_button=lambda label, *a, **k: "masondrop" in label,
    )
    controls = SimpleNamespace(segmented_choice=lambda key, keys, cur: (False, cur))
    imgui = SimpleNamespace(same_line=lambda *a, **k: None, dummy=lambda *a, **k: None)
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(mason_tools, "widgets", widgets)
        mp.setattr(mason_tools, "controls", controls)
        mp.setattr(mason_tools, "imgui", imgui)
        mp.setattr(mason_tools, "_array", lambda *a, **k: None)
        mp.setattr(mason_tools, "sp", lambda v: v)
        ctx = SimpleNamespace(mason_assets=_Source())
        mason_tools._placement(ctx, SimpleNamespace(), SimpleNamespace(doc=doc))
    finally:
        mp.undo()
    assert {n: getattr(mason_tools, n) for n in original} == original

    # The unit cube's bottom rests on the ground node's world height (10).
    assert msc.resolved_for(doc, prop.uid).world[1, 3] == pytest.approx(10.5)


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(
            mason=SimpleNamespace(
                tool="move",
                pivot="median",
                snap=False,
                snap_translate=0.0,
                snap_rotate=0.0,
                snap_ground=True,
            )
        )


def test_the_drag_time_ground_snap_lands_on_a_moved_ground() -> None:
    doc, prop = _doc_with_moved_ground()
    view = mason_view.MasonView.__new__(mason_view.MasonView)
    view.app_ctx = _Ctx()
    view._drag_start = {prop.uid: tuple(np.array(v, copy=True) for v in prop.trs())}
    view._drag_pivot = np.zeros(3)
    source = _Source()

    view._apply_drag(doc, source, delta=np.array([0.0, 5.0, 0.0]))

    box = msc.world_bounds(doc, source, uids=[prop.uid])
    assert box is not None
    assert box[0][1] == pytest.approx(10.0)
