"""The 2026-10-03 audit's Low Clay findings, batch clay3 (clay-84, 88, 90, 91,
92, 94, 95, 98, 101, 104, 105, 106, 109, 112, 124): the rblk compression, the
MTL name and factor reading, the ear-clip budget, bisect, the bevel modifier's
threshold, weld, the boolean refusal, two agent doors, and the comments and
evidence gaps around them.
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
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

from .test_agent_clay import _Ctx

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


def _u_shape():
    from realmspinner.kernels.mesh import mesh as bm

    pts = [(0, 0), (3, 0), (3, 2), (2, 2), (2, 1), (1, 1), (1, 2), (0, 2)]
    return bm.from_faces([[x, y, 0.0] for x, y in pts], [list(range(8))])


def test_bisect_splits_a_u_shaped_face_crossed_four_times_into_two_faces() -> None:
    from realmspinner.kernels.mesh import elements as el
    from realmspinner.kernels.mesh import mesh as bm
    from realmspinner.kernels.mesh import ops_model as om

    mesh = _u_shape()
    out, sel = om.bisect(
        mesh, el.ElementSel(faces=np.array([0])), point=(0, 1.5, 0), normal=(0, 1, 0), clear=0
    )
    bm.validate(out)
    sizes = sorted(int(out.starts[i + 1] - out.starts[i]) for i in range(bm.face_count(out)))
    # the two arm tops are quads; the U's lower body stays one 8-gon
    assert sizes == [4, 4, 8]
    # both stretches of the cut line inside the face are named, not one of two
    assert len(sel.edges) == 2
    # no face may visit a vertex twice (the old single 8-gon touched itself)
    for i in range(bm.face_count(out)):
        face = out.loops[out.starts[i] : out.starts[i + 1]]
        assert len(set(face.tolist())) == len(face)
    # and the area is conserved: 3*2 minus the 1x1 notch
    total = 0.0
    for i in range(bm.face_count(out)):
        ring = out.positions[out.loops[out.starts[i] : out.starts[i + 1]]].astype("f8")
        total += 0.5 * abs(np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1]))
    assert abs(total - 5.0) < 1e-6


def test_bisect_of_a_convex_face_is_unchanged_by_the_multi_crossing_split() -> None:
    from realmspinner.kernels.mesh import elements as el
    from realmspinner.kernels.mesh import mesh as bm
    from realmspinner.kernels.mesh import ops_model as om

    box = bp.box()
    out, sel = om.bisect(
        box,
        el.ElementSel(faces=np.arange(bm.face_count(box))),
        point=(0, 0, 0),
        normal=(1, 0, 0),
        clear=0,
    )
    assert bm.face_count(out) == 10  # four faces cut in two, two caps untouched
    assert len(sel.edges) == 4


# --- clay-92 -----------------------------------------------------------------


def test_bevel_modifier_at_angle_zero_leaves_coplanar_edges_alone() -> None:
    from realmspinner.kernels.mesh import mesh as bm
    from realmspinner.kernels.mesh import ops_modifiers

    flat = bp.grid(size=(1.0, 1.0), divisions=4)
    c, s = np.cos(0.3), np.sin(0.3)
    tilt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype="f4")
    tilted = bm.Mesh(
        positions=(flat.positions @ tilt.T).astype("f4"),
        loops=flat.loops,
        starts=flat.starts,
        material=flat.material,
        smooth=flat.smooth,
        uv=flat.uv,
    )
    out = ops_modifiers.bevel_by_angle(tilted, {"width": 0.02, "angle": 0.0})
    assert bm.face_count(out) == bm.face_count(tilted)


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
    import pytest

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


def _two_agent_boxes():
    ctx = _Ctx()
    session = agent_clay.Session()
    a = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    b = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    uid_a = a["structuredContent"]["uid"]
    uid_b = b["structuredContent"]["uid"]
    return ctx, session, uid_a, uid_b


def test_clay_boolean_refuses_an_unknown_or_hidden_uid_instead_of_dropping_it() -> None:
    ctx, session, uid_a, uid_b = _two_agent_boxes()
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    before = len(tab.doc.objects)

    unknown = agent_clay.call(
        ctx, session, "clay_boolean", {"kind": "union", "uids": [uid_a, uid_b, 999999]}
    )
    assert unknown["isError"] is True
    assert "999999" in unknown["content"][0]["text"]
    assert len(tab.doc.objects) == before  # nothing ran on the two it did know

    tab.doc.by_uid(uid_b).visible = False
    hidden = agent_clay.call(
        ctx, session, "clay_boolean", {"kind": "union", "uids": [uid_a, uid_b]}
    )
    assert hidden["isError"] is True
    assert "hidden" in hidden["content"][0]["text"].lower()
    assert str(uid_b) in hidden["content"][0]["text"]
    assert len(tab.doc.objects) == before


# --- clay-101 ----------------------------------------------------------------


def _with_manifold(ctx, *uids):
    manifold = clay_mode.ensure(ctx).manifold
    for u in uids:
        manifold[u] = object()
    return manifold


def test_agent_separate_forgets_the_manifold_entry_of_the_removed_source() -> None:
    from .test_agent_clay_structure import _TWO_TETRA_ARGS

    ctx = _Ctx()
    session = agent_clay.Session()
    added = agent_clay.call(ctx, session, "clay_add_mesh", _TWO_TETRA_ARGS)
    import json

    uid = json.loads(added["content"][0]["text"])["uid"]
    manifold = _with_manifold(ctx, uid)

    result = agent_clay.call(ctx, session, "clay_separate", {"uid": uid, "by": "loose_parts"})
    assert result["isError"] is False, result
    assert uid not in manifold  # the source left doc.objects; its pinned Mesh goes too


def test_agent_ungroup_forgets_the_manifold_entry_of_the_removed_group() -> None:
    import json

    ctx = _Ctx()
    session = agent_clay.Session()
    a = json.loads(agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})["content"][0]["text"])
    b = json.loads(agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})["content"][0]["text"])
    grouped = agent_clay.call(ctx, session, "clay_group", {"uids": [a["uid"], b["uid"]]})
    group_uid = json.loads(grouped["content"][0]["text"])["uid"]
    manifold = _with_manifold(ctx, group_uid)

    result = agent_clay.call(ctx, session, "clay_ungroup", {"uid": group_uid})
    assert result["isError"] is False, result
    assert group_uid not in manifold


# --- clay-104 ----------------------------------------------------------------


def test_queries_comment_count_matches_the_registry() -> None:
    import inspect

    from realmspinner.kernels.mesh import drag, select

    words = {6: "six", 12: "twelve", 13: "thirteen", 19: "nineteen"}
    src = inspect.getsource(select)
    assert "Deliberately six entries" not in src
    assert f"**{words[len(select.QUERIES)].capitalize()} entries" in src
    assert "viewer.math3d" not in inspect.getsource(drag)
    assert "kernels.geom3d.math3d" in inspect.getsource(drag)


# --- clay-105 ----------------------------------------------------------------


def test_findings_runs_the_shell_walk_once_when_a_shell_is_inside_out(monkeypatch) -> None:
    from realmspinner.kernels.mesh import diagnose, ops_clean
    from realmspinner.kernels.mesh import mesh as bm

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
    walks = []
    real = ops_clean._shell_and_flip

    def counting(mesh, a):
        walks.append(1)
        return real(mesh, a)

    monkeypatch.setattr(ops_clean, "_shell_and_flip", counting)
    rows = diagnose.findings(flipped)
    inside = [r for r in rows if r.kind == "inside_out"]
    assert len(inside) == 1 and inside[0].count == 1
    assert len(walks) == 1


# --- clay-106 ----------------------------------------------------------------


def test_merge_into_closes_its_gesture_when_grouping_raises(monkeypatch) -> None:
    import pytest

    from realmspinner.kernels.mesh import merge
    from realmspinner.kernels.mesh.elements import OpError

    doc = bd.ClayDoc()
    incoming = bd.ClayDoc()
    for name in ("a", "b"):
        incoming.add_object(bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box()))

    def refuse(*args, **kwargs):
        raise OpError("locked")

    monkeypatch.setattr(doc, "group", refuse)
    with pytest.raises(OpError):
        merge.merge_into(doc, incoming, offset=(0.0, 0.0, 0.0))
    assert doc.history._open_gestures == 0


# --- clay-109 ----------------------------------------------------------------


def test_uv_module_docstring_does_not_deny_the_lscm_solver_exists() -> None:
    from realmspinner.kernels.mesh import uv

    doc = uv.__doc__ or ""
    assert "wrong shape for this codebase" not in doc
    assert "that is the moment to write it" not in doc
    assert "uvunwrap" in doc


# --- clay-112 ----------------------------------------------------------------


def test_chapter_07_names_every_clay_snap_switch() -> None:
    import dataclasses
    from pathlib import Path

    from realmspinner.studio.modes.clay.state import ClayState

    switches = [
        f.name for f in dataclasses.fields(ClayState) if f.name.startswith("snap_") and f.type in (bool, "bool")
    ]
    assert sorted(switches) == ["snap_edge", "snap_face", "snap_vertex"]  # plus plain `snap`, the grid
    text = (Path(__file__).resolve().parents[3] / "docs" / "manual" / "07-modelling.md").read_text(encoding="utf-8")
    assert "Two snapping switches" not in text
    text = text.replace("\r\n", "\n")
    para = text.split("Four snapping switches", 1)[1].split("\n\n", 1)[0]
    for word in ("grid", "vertex", "edge", "face"):
        assert word in para


# --- clay-124 ----------------------------------------------------------------


def test_every_part_matches_its_bones_direction_and_length() -> None:
    """test_presets' landmark check compares translation only (midpoint within
    0.03), so a bone whose direction or length changed with its midpoint inside
    that tolerance left a part with the wrong rotation or capsule height and no
    red test. This compares what ``presets._placed`` derives from the real
    template bone: the rotation (as a rotation, so q and -q agree) and the
    length a capsule's or box's section is built from."""
    from realmspinner.kernels.mesh import presets
    from realmspinner.kernels.rig.templates import templates

    checked_rotation = checked_length = 0
    identity = (0.0, 0.0, 0.0, 1.0)
    for key, (_label, build) in presets.ASSEMBLIES.items():
        bones = {b["name"]: b for b in templates()[key].bones}
        for part in build():
            if part.bone is None:
                continue
            bone = bones[part.bone]
            _mid, rotation, length = presets._placed(bone["head"], bone["tail"])
            if part.generator in ("capsule", "box") or tuple(part.rotation) != identity:
                dot = abs(float(np.dot(np.asarray(rotation, dtype="f8"), np.asarray(part.rotation, dtype="f8"))))
                assert dot == pytest.approx(1.0, abs=1e-6), f"{key}/{part.name}: rotation is not {part.bone}'s"
                checked_rotation += 1
            if part.generator == "capsule":
                expected = max(length - 2.0 * part.params["radius"], presets.MIN_CAPSULE_SECTION)
                assert part.params["height"] == pytest.approx(expected, abs=1e-6), (
                    f"{key}/{part.name}: capsule height is not {part.bone}'s length"
                )
                checked_length += 1
            elif part.generator == "box":
                expected = max(length, presets.MIN_CAPSULE_SECTION)
                assert part.params["size"][1] == pytest.approx(expected, abs=1e-6), (
                    f"{key}/{part.name}: box length is not {part.bone}'s length"
                )
                checked_length += 1
    # the gate must actually have looked at something
    assert checked_rotation >= 90 and checked_length >= 90
