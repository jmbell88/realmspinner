"""The 2026-10-07 Clay audit's mesh-core findings: clay-14, 53, 54, 58, 59."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np

from realmspinner.kernels import mesh as clay_pkg
from realmspinner.kernels.mesh import curves, measure, scratch
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp

# --- clay-14 ------------------------------------------------------------------


def _with_face_flipped(mesh: bm.Mesh, face: int) -> bm.Mesh:
    loops = mesh.loops.copy()
    lo, hi = int(mesh.starts[face]), int(mesh.starts[face + 1])
    loops[lo:hi] = loops[lo:hi][::-1]
    return bm.Mesh(mesh.positions, loops, mesh.starts, mesh.material, mesh.smooth)


def test_volume_if_closed_refuses_a_mesh_with_a_flipped_face() -> None:
    box = bp.box((1.0, 1.0, 1.0))
    assert abs(measure.volume_if_closed(box) - 1.0) < 1e-6
    bad = _with_face_flipped(box, 0)
    # The cube is still watertight edge-wise; only one face disagrees about
    # which way is out, so the divergence sum is a confident wrong number.
    assert measure.volume_if_closed(bad) is None


# --- clay-53 ------------------------------------------------------------------


def test_transplant_adds_new_objects_in_the_order_the_preview_showed() -> None:
    base = bd.ClayDoc()
    base.add_object(bd.Obj(uid=bd.new_uid(), name="base", mesh=bp.box()))
    sc = scratch.clone(base)
    # Sequential uids that straddle a multiple of a small set's table size, so
    # set iteration order is not insertion order.
    for uid in (125, 126, 127, 128, 129):
        sc.add_object(bd.Obj(uid=uid, name=f"new{uid}", mesh=bp.box()))
    d = scratch.diff(base, sc)

    scratch.transplant(base, sc, d)

    assert [o.name for o in base.objects] == [o.name for o in sc.objects]


# --- clay-54 ------------------------------------------------------------------


def test_a_finite_but_enormous_handle_flattens_to_the_station_cap_instead_of_overflowing() -> None:
    pts = [[0.0, 0.0], [1.0, 1.0]]
    handles = [[[0.0, 0.0], [1e308, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
    assert curves.normalise_handles(handles, 2, 2)  # finite, so it is accepted
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flat = curves.flatten(pts, handles)
    assert 2 < len(flat) <= curves.MAX_STATIONS
    assert np.isfinite(np.asarray(flat)).all()

    # Opposite huge handles on one segment: inf - inf would be nan.
    handles = [[[0.0, 0.0], [1e308, 0.0]], [[-1e308, 0.0], [0.0, 0.0]]]
    flat = curves.flatten(pts, handles)
    assert len(flat) <= curves.MAX_STATIONS
    assert np.isfinite(np.asarray(flat)).all()


# --- clay-58 ------------------------------------------------------------------


def _layouts(count: int) -> list[bm.RenderLayout]:
    layouts = []
    for i in range(count):
        mesh = bp.box((1.0 + i, 1.0, 1.0))
        layouts.append(bm.render_layout(mesh))
    return layouts


def test_a_drag_over_sixteen_materials_reuses_every_layouts_raw_normals() -> None:
    layouts = _layouts(16)
    for layout in layouts:
        bm.render_from_layout(layout, np.asarray(layout_positions(layout)))
    missing = [i for i, layout in enumerate(layouts) if bm.raw_face_normals(layout) is None]
    assert missing == []


def layout_positions(layout: bm.RenderLayout) -> np.ndarray:
    # Positions are only read through the layout's own loops, so any array of
    # the right length will do for a stash test.
    return np.zeros((layout.n_verts, 3), dtype="f8")


def test_the_raw_normal_stash_is_bounded_by_bytes_not_only_by_layout_lifetime(monkeypatch) -> None:
    layouts = _layouts(16)
    one = 6 * 3 * 8  # a box's raw normals: six faces of three f8
    # Room for exactly four stashes: the newest four survive, the rest go.
    monkeypatch.setattr(bm, "_RAW_CACHE_MAX_BYTES", 4 * one + 4 * 8 * 3)
    with bm._RAW_CACHE_LOCK:
        bm._RAW_CACHE.clear()
    for layout in layouts:
        bm.render_from_layout(layout, layout_positions(layout), moved=np.array([0]))
    kept = [bm.raw_face_normals(layout) is not None for layout in layouts]
    assert sum(kept) <= 4
    assert kept[-1], "the stash just written must always be kept"
    with bm._RAW_CACHE_LOCK:
        total = sum(
            raw.nbytes + (0 if stamp is None else stamp.nbytes)
            for _ref, raw, stamp in bm._RAW_CACHE.values()
        )
    assert total <= bm._RAW_CACHE_MAX_BYTES


# --- clay-59 ------------------------------------------------------------------


def _pin_module():
    path = Path(__file__).with_name("test_clay_imports.py")
    spec = importlib.util.spec_from_file_location("_clay_pin_for_docstring_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_mesh_package_docstring_names_every_import_the_pin_test_allows() -> None:
    pin = _pin_module()
    doc = clay_pkg.__doc__ or ""
    wanted = {module for _file, module in pin.OUTWARD_IMPORTS}
    wanted |= {f"{m}" for m in pin.VIEWER_MODULES}
    wanted |= set(pin.LAZY_ONLY)
    missing = sorted(name for name in wanted if name not in doc)
    assert missing == [], f"the package docstring does not mention {missing}"
    # A deleted module is not a thing the package may reach for.
    assert "viewer" not in doc
