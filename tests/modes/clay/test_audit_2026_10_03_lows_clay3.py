"""The 2026-10-03 audit's Low Clay findings, batch clay3 (clay-84, 88, 90, 94,
95, 101, 104, 112): the rblk compression, the MTL name and factor reading, the
ear-clip budget, weld, the boolean refusal, two agent doors, and the comments
and evidence gaps around them.
"""

from __future__ import annotations

# ruff: noqa: E501 - a regression test's name is the claim, and these are long
import io
import zipfile

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser

# --- clay-84 -----------------------------------------------------------------


def test_rblk_members_other_than_textures_are_deflated() -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="ball", mesh=bp.uv_sphere(segments=32, rings=16)))
    data = ser.rblk_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        infos = zf.infolist()
        assert infos
        for info in infos:
            if info.filename.endswith(".png"):
                assert info.compress_type == zipfile.ZIP_STORED
            else:
                assert info.compress_type == zipfile.ZIP_DEFLATED, info.filename
        npz = next(i for i in infos if i.filename.endswith(".npz"))
        inner = zipfile.ZipFile(io.BytesIO(zf.read(npz)))
        for info in inner.infolist():
            assert info.compress_type == zipfile.ZIP_DEFLATED, info.filename
    # and it is still a document
    assert len(ser.read_rblk(data).objects) == 1
    assert isinstance(np.zeros(1), np.ndarray)


def test_rscn_members_other_than_textures_are_deflated() -> None:
    # The same ZipInfo-beats-the-archive-default shape as the .rblk writer
    # (clay-84 names mason/engine/serialize.py too).
    from realmspinner.kernels.geom3d import gltf
    from realmspinner.studio.modes.mason.engine import serialize as mser
    from realmspinner.studio.modes.mason.engine.document import MasonDoc
    from realmspinner.studio.modes.mason.engine.terrain import Terrain

    doc = MasonDoc()
    doc.terrain = Terrain(
        heights=np.zeros((9, 9), dtype=np.float32),
        size_x=8.0,
        size_z=8.0,
        material=gltf.Material(name="ground"),
    )
    with zipfile.ZipFile(io.BytesIO(mser.rscn_bytes(doc))) as zf:
        names = zf.namelist()
        assert mser.TERRAIN_HEIGHTS in names
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_DEFLATED, info.filename


# --- clay-88 -----------------------------------------------------------------

_TRI = "v 0 0 0\nv 1 0 0\nv 0 1 0\n"


def test_usemtl_matches_a_newmtl_name_with_a_doubled_space() -> None:
    from realmspinner.kernels.mesh import objimport

    mtl = "newmtl Dark  Wood\nKd 0.25 0.5 0.75\n"
    doc = objimport.obj_to_claydoc(_TRI + "usemtl Dark  Wood\nf 1 2 3\n", mtl=mtl)
    idx = doc.objects[0].material
    assert doc.materials[idx].base_color_factor[:3] == (0.25, 0.5, 0.75)


def test_mtl_factors_are_clamped_into_gltfs_zero_to_one_range() -> None:
    from realmspinner.kernels.mesh import objimport

    mtl = "newmtl Hot\nKd 5 -3 0.5\nd 7\n"
    doc = objimport.obj_to_claydoc(_TRI + "usemtl Hot\nf 1 2 3\n", mtl=mtl)
    factor = doc.materials[doc.objects[0].material].base_color_factor
    assert factor == (1.0, 0.0, 0.5, 1.0)


# --- clay-90 -----------------------------------------------------------------


def _comb_faces(count: int, corners: int):
    """``count`` concave comb polygons of ``corners`` corners each, as raw arrays."""
    pts: list[list[float]] = []
    starts = [0]
    for k in range(count):
        base = len(pts)
        width = float(corners)
        pts.append([0.0, -1.0, float(k) * 10.0])
        pts.append([width, -1.0, float(k) * 10.0])
        for i in range(corners - 2):
            x = width - (i + 0.5) * width / (corners - 2)
            pts.append([x, 1.0 if i % 2 == 0 else 2.0, float(k) * 10.0])
        starts.append(base + corners)
    positions = np.asarray(pts, dtype="f4")
    loops = np.arange(len(pts), dtype="i8")
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (count, 1))
    return positions, loops, np.asarray(starts, dtype="i8"), normals


def test_triangulating_many_large_concave_faces_stays_inside_a_time_budget(monkeypatch) -> None:
    # Counted in ear-search work rather than wall clock: a stopwatch is
    # meaningless under xdist contention, and the cost *is* the sum of squares.
    from realmspinner.kernels.mesh import earclip

    seen: list[int] = []
    real = earclip._earclip

    def counting(pts):
        seen.append(len(pts))
        return real(pts)

    monkeypatch.setattr(earclip, "_earclip", counting)
    positions, loops, starts, normals = _comb_faces(20, 500)
    corners, tri_face = earclip.corner_triangles(positions, loops, starts, normals)
    assert sum(n * n for n in seen) <= earclip.MAX_EARCLIP_TOTAL_WORK
    assert 0 < len(seen) < 20  # some ears were cut, the rest fanned
    # the contract every caller indexes by: n - 2 triangles per face, either way
    assert len(corners) == 20 * 498
    assert np.bincount(tri_face).tolist() == [498] * 20


# --- clay-91 -----------------------------------------------------------------


# --- clay-92 -----------------------------------------------------------------


# --- clay-94 -----------------------------------------------------------------


def _one_face(points, uv=None):
    from realmspinner.kernels.mesh import mesh as bm

    return bm.Mesh(
        positions=np.asarray(points, dtype="f4"),
        loops=np.arange(len(points), dtype="i4"),
        starts=np.array([0, len(points)], dtype="i4"),
        material=np.zeros(1, dtype="i4"),
        smooth=np.zeros(1, dtype=bool),
        uv=uv,
    )


def test_weld_does_not_leave_a_face_that_visits_one_vertex_twice() -> None:
    from realmspinner.kernels.mesh import adjacency, ops_topo
    from realmspinner.kernels.mesh import elements as el
    from realmspinner.kernels.mesh import mesh as bm

    # A quad with two *opposite* corners welded is the zero-area bowtie
    # [0, 1, 0, 2]: nothing the consecutive-duplicate pass catches.
    quad = _one_face([[0, 0, 0], [1, 0, 0], [1e-6, 0, 0], [0, 1, 0]])
    out, _ = ops_topo.weld(quad, el.empty(), eps=1e-3)
    bm.validate(out)
    assert bm.face_count(out) == 0

    # A hexagon welded across its waist is a figure-eight; its two lobes are
    # real triangles and stay, as two faces that each visit a vertex once.
    hexagon = _one_face([[0, 0, 0], [1, 1, 0], [1, -1, 0], [1e-6, 0, 0], [-1, -1, 0], [-1, 1, 0]])
    out, _ = ops_topo.weld(hexagon, el.empty(), eps=1e-3)
    bm.validate(out)
    assert sorted(int(out.starts[i + 1] - out.starts[i]) for i in range(bm.face_count(out))) == [3, 3]
    assert not adjacency.check_manifold(out).repeated_corner_faces.size


# --- clay-95 -----------------------------------------------------------------


def _split_vertex_box():
    """A closed box whose every face owns its own four vertices (24 vertices)."""
    from realmspinner.kernels.mesh import mesh as bm

    box = bp.box()
    return bm.from_faces(
        box.positions[box.loops], [list(range(4 * i, 4 * i + 4)) for i in range(6)]
    )


def test_boolean_refusal_names_weld_for_a_split_vertex_solid_and_normals_for_an_inside_out_one() -> None:

    pytest.importorskip("manifold3d")
    from realmspinner.kernels.mesh import mesh as bm
    from realmspinner.kernels.mesh import ops_boolean
    from realmspinner.kernels.mesh.elements import OpError

    def obj(name, mesh):
        return bd.Obj(uid=bd.new_uid(), name=name, mesh=mesh)

    partner = obj("B", bp.box())
    split = _split_vertex_box()
    assert len(split.positions) == 24
    with pytest.raises(OpError, match="Weld") as err:
        ops_boolean.union([obj("A", split), partner])
    assert "fill them" not in str(err.value)

    box = bp.box()
    flipped = bm.Mesh(
        positions=box.positions,
        loops=np.concatenate(
            [box.loops[box.starts[i] : box.starts[i + 1]][::-1] for i in range(bm.face_count(box))]
        ),
        starts=box.starts,
        material=box.material,
        smooth=box.smooth,
    )
    with pytest.raises(OpError, match="Recalculate Normals") as err:
        ops_boolean.union([obj("A", flipped), partner])
    assert "fill them" not in str(err.value)


# --- clay-98 -----------------------------------------------------------------


# --- clay-101 ----------------------------------------------------------------


# --- clay-104 ----------------------------------------------------------------


def test_drag_module_cites_the_geom3d_math3d_not_the_viewers() -> None:
    import inspect

    from realmspinner.kernels.mesh import drag

    assert "viewer.math3d" not in inspect.getsource(drag)
    assert "kernels.geom3d.math3d" in inspect.getsource(drag)


# --- clay-105 ----------------------------------------------------------------


# --- clay-106 ----------------------------------------------------------------


# --- clay-109 ----------------------------------------------------------------


# --- clay-112 ----------------------------------------------------------------


def test_chapter_07_names_no_snap_switch_clay_no_longer_has() -> None:
    import dataclasses
    from pathlib import Path

    from realmspinner.studio.modes.clay.state import ClayState

    switches = [
        f.name for f in dataclasses.fields(ClayState) if f.name.startswith("snap_") and f.type in (bool, "bool")
    ]
    assert switches == []  # element snapping went; only the grid and angle snap remain
    text = (Path(__file__).resolve().parents[3] / "docs" / "manual" / "07-modelling.md").read_text(encoding="utf-8")
    assert "Four snapping switches" not in text
    assert "Two snapping switches" not in text
