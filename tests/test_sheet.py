"""Sprite sheets: the pure layout/packing module, and the framing the Blender
worker's ``op_sheet`` does (which the rig deformation QA still renders with).

The sheet *job kinds* and their doors left with Poser (``retired.py``), and with
them the service surface and the queue tests that lived here. Blender is faked
for all of what is left except the last section, which is behind the gpu
marker: everything up to there is arithmetic and plumbing, and only the final
tests need bpy to put actual pixels in a frame.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from realmspinner.kernels import sheet as sheetlib
from realmspinner.kernels.rig import blender_spec
from realmspinner.kernels.rig import store as rig_store

IDENTITY = [0.0, 0.0, 0.0, 1.0]


# --- layout -----------------------------------------------------------------


def test_grid_is_poses_down_and_yaws_across():
    layout = sheetlib.plan([{"id": "a" * 12, "name": "idle"}, {"id": "b" * 12, "name": "run"}])
    assert (layout.columns, layout.rows) == (8, 2)
    assert (layout.width, layout.height) == (8 * 128, 2 * 128)
    assert len(layout.cells) == 16


def test_cells_are_indexed_row_major_and_placed_on_the_grid():
    layout = sheetlib.plan([{"id": "a" * 12, "name": "idle"}], frame_size=64)
    for i, cell in enumerate(layout.cells):
        assert cell.index == i
        assert (cell.x, cell.y) == (cell.column * 64, cell.row * 64)
    assert [c.yaw for c in layout.cells] == [0, 45, 90, 135, 180, 225, 270, 315]


def test_no_poses_means_one_rest_row_not_an_empty_sheet():
    """An unrigged prop still wants a turnaround."""
    layout = sheetlib.plan([])
    assert layout.rows == 1
    assert {c.pose for c in layout.cells} == {None}
    assert {c.pose_name for c in layout.cells} == {sheetlib.REST_POSE_NAME}


def test_every_cell_carries_a_frame_number():
    """The seam animated clips arrive on: more cells, same format."""
    layout = sheetlib.plan([])
    assert {c.frame for c in layout.cells} == {0}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_size": 100},
        {"lighting": "toon"},
        {"elevation": 90.0},
        {"elevation": -95.0},
        {"yaws": 0},
    ],
)
def test_unrenderable_requests_are_rejected_at_plan_time(kwargs):
    with pytest.raises(ValueError):
        sheetlib.plan([], **kwargs)


def test_an_atlas_over_the_texture_limit_is_refused():
    poses = [{"id": f"{i:012x}", "name": str(i)} for i in range(20)]
    with pytest.raises(ValueError, match="8192"):
        sheetlib.plan(poses, frame_size=512)


def test_yaw_angles_start_at_the_front_view():
    assert sheetlib.yaw_angles(4) == (0.0, 90.0, 180.0, 270.0)


# --- animation ----------------------------------------------------------------


def test_slerp_endpoints_and_midpoint():
    a = [0.0, 0.0, 0.0, 1.0]
    half = math.radians(90) / 2
    b = [0.0, 0.0, math.sin(half), math.cos(half)]
    assert sheetlib.slerp(a, b, 0.0) == pytest.approx(a)
    assert sheetlib.slerp(a, b, 1.0) == pytest.approx(b)
    quarter = math.radians(45) / 2
    assert sheetlib.slerp(a, b, 0.5) == pytest.approx(
        [0.0, 0.0, math.sin(quarter), math.cos(quarter)], abs=1e-6
    )


def test_slerp_takes_the_short_way_round():
    """Negating a quaternion is the same rotation; without the sign fix the
    interpolation spins the long way and the sprite counter-rotates."""
    a = [0.0, 0.0, 0.0, 1.0]
    b = [0.0, 0.0, 0.0, -1.0]
    mid = sheetlib.slerp(a, b, 0.5)
    assert abs(mid[3]) == pytest.approx(1.0, abs=1e-6)


def _between(a, b, frames):
    """``frames`` records stepping from A toward B and stopping short of it.

    The two-pose door ``sheet.interpolate`` was (deleted: the 2026-10-14 audit,
    finding inker-80, no production caller), spelled through the multi-key one
    that is live.
    """
    return sheetlib.interpolate_clip([a, b], [frames, frames], closed=True)[:frames]


def test_interpolate_produces_numbered_frames_between_two_poses():
    a = {"id": "a" * 12, "name": "contact A", "bones": {"thigh.L": [0.0, 0.0, 0.0, 1.0]}}
    half = math.radians(60) / 2
    b = {
        "id": "b" * 12,
        "name": "contact B",
        "bones": {"thigh.L": [math.sin(half), 0.0, 0.0, math.cos(half)]},
    }

    frames = _between(a, b, 4)

    assert len(frames) == 4
    assert [f["frame"] for f in frames] == [0, 1, 2, 3]
    assert frames[0]["bones"]["thigh.L"] == pytest.approx(a["bones"]["thigh.L"])
    # The last frame stops short of B, so a looping clip does not hold a
    # duplicate frame at the seam.
    assert frames[-1]["bones"]["thigh.L"] != pytest.approx(b["bones"]["thigh.L"])


def test_a_bone_posed_in_only_one_end_is_held_rather_than_swung():
    """**In node space, identity is not rest** -- it is parent alignment.

    This used to assert the identity quaternion, on the reading that an
    omitted bone means rest (which is what the worker's ``_reset_pose``
    means, and which is true in ``delta`` space). A *node* rotation is
    absolute against the parent joint, so blending the missing end from
    identity swung the bone out through the limb lying along its parent's
    axis, mid-segment, on every clip built from library poses -- a pose
    nobody authored, rendered into the middle of the clip.

    The honest reading of a key that says nothing about a bone is that it
    says nothing, so the other end is held and the bone does not move.
    Blending from the bone's own *rest* rotation would be better still and
    is not available here: ``_blend`` is handed two poses and no rig.
    """
    a = {"id": "a" * 12, "name": "A", "bones": {}}
    turned = [0.0, 0.0, 0.7071, 0.7071]
    b = {"id": "b" * 12, "name": "B", "bones": {"head": turned}}

    frames = _between(a, b, 2)

    # ``slerp`` normalises, so the held value comes back as the unit
    # quaternion the input rounds to rather than bit-for-bit.
    assert frames[0]["bones"]["head"] == pytest.approx(turned, abs=1e-5)
    assert frames[1]["bones"]["head"] == pytest.approx(turned, abs=1e-5)


def test_delta_space_still_blends_an_omitted_bone_from_rest():
    """There identity *is* rest -- a delta of nothing -- which is what
    ``_reset_pose`` means by an omitted bone."""
    a = {"name": "A", "bones": {}}
    b = {"name": "B", "bones": {"head": [0.0, 0.0, 0.7071, 0.7071]}}

    frames = sheetlib.interpolate_clip([a, b], [2], space="delta", closed=False)

    assert frames[0]["bones"]["head"] == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_interpolate_clip_rejects_a_clip_length_it_cannot_render():
    a = {"id": "a" * 12, "name": "A", "bones": {}}
    with pytest.raises(ValueError):
        sheetlib.interpolate_clip([a, a], [0, 0], closed=True)
    with pytest.raises(ValueError):
        sheetlib.interpolate_clip([a, a], [sheetlib.MAX_CLIP_FRAMES + 1] * 2, closed=True)


def test_plan_honours_the_yaw_count():
    layout = sheetlib.plan([], yaws=4)
    assert layout.columns == 4
    assert len(layout.cells) == 4


def test_plan_carries_each_records_own_frame_index():
    """A clip's rows differ only by frame; dropping it renders row 0 six times."""
    a = {"id": "a" * 12, "name": "A", "bones": {}}
    b = {"id": "b" * 12, "name": "B", "bones": {}}
    layout = sheetlib.plan(_between(a, b, 3), yaws=2)
    assert [c.frame for c in layout.cells] == [0, 0, 1, 1, 2, 2]


# --- packing ----------------------------------------------------------------


def _frames(tmp_path, layout, colours=None):
    """One solid-colour PNG per cell, so a paste can be located by pixel."""
    out = {}
    for cell in layout.cells:
        colour = (colours or {}).get(cell.index, (cell.index * 8 % 256, 64, 128, 255))
        path = tmp_path / f"{cell.index:04d}.png"
        Image.new("RGBA", (layout.frame_size, layout.frame_size), colour).save(path)
        out[cell.index] = path
    return out


def test_pack_places_each_frame_at_its_own_cell(tmp_path):
    layout = sheetlib.plan([{"id": "a" * 12, "name": "idle"}], frame_size=64)
    out = tmp_path / "sheet.png"
    sheetlib.pack(layout, _frames(tmp_path, layout), out)
    with Image.open(out) as atlas:
        assert atlas.size == (layout.width, layout.height)
        for cell in layout.cells:
            probe = atlas.getpixel((cell.x + 32, cell.y + 32))
            assert probe == (cell.index * 8 % 256, 64, 128, 255)


def test_pack_resizes_a_frame_that_came_back_the_wrong_size(tmp_path):
    layout = sheetlib.plan([], frame_size=64)
    frames = _frames(tmp_path, layout)
    Image.new("RGBA", (32, 32), (10, 20, 30, 255)).save(frames[0])
    out = tmp_path / "sheet.png"
    sheetlib.pack(layout, frames, out)
    with Image.open(out) as atlas:
        assert atlas.size == (layout.width, layout.height)
        assert atlas.getpixel((32, 32)) == (10, 20, 30, 255)


def test_pack_refuses_to_leave_a_hole(tmp_path):
    """A gap in a sheet reads as a modelling bug and sends the user hunting in
    the wrong place."""
    layout = sheetlib.plan([], frame_size=64)
    frames = _frames(tmp_path, layout)
    frames[3].unlink()
    with pytest.raises(ValueError, match="cell 3"):
        sheetlib.pack(layout, frames, tmp_path / "sheet.png")


def test_pack_refuses_a_frame_path_that_is_a_directory(tmp_path):
    """inker-11 (2026-09-07 audit): the refusal gated on ``path.exists()``,
    which is also true of a directory -- a stale or malformed frames entry
    pointing at one slipped past "no rendered frame" only to fail later
    inside ``Image.open`` with no mention of which cell. ``is_file()`` is
    what the docstring's refusal actually means.
    """
    layout = sheetlib.plan([], frame_size=64)
    frames = _frames(tmp_path, layout)
    frames[3].unlink()
    frames[3].mkdir()
    with pytest.raises(ValueError, match="cell 3"):
        sheetlib.pack(layout, frames, tmp_path / "sheet.png")


def test_untouched_cells_stay_transparent(tmp_path):
    layout = sheetlib.plan([], frame_size=64)
    frames = _frames(tmp_path, layout, colours={i: (0, 0, 0, 0) for i in range(8)})
    out = tmp_path / "sheet.png"
    sheetlib.pack(layout, frames, out)
    with Image.open(out) as atlas:
        assert atlas.getpixel((10, 10))[3] == 0


# --- sidecar ----------------------------------------------------------------


def test_sidecar_describes_the_grid_and_every_cell():
    layout = sheetlib.plan([{"id": "a" * 12, "name": "idle"}], frame_size=64, lighting="lit")
    meta = sheetlib.sidecar(
        layout, sheet_id="c" * 12, source_job="d" * 12, image="c.png", created=1.0
    )
    assert meta["version"] == sheetlib.SHEET_VERSION
    assert (meta["columns"], meta["rows"]) == (8, 1)
    assert (meta["width"], meta["height"]) == (512, 64)
    assert meta["lighting"] == "lit"
    assert meta["poses"] == [{"id": "a" * 12, "name": "idle"}]
    assert len(meta["cells"]) == 8
    first = meta["cells"][0]
    assert first == {
        "index": 0, "row": 0, "column": 0, "x": 0, "y": 0, "w": 64, "h": 64,
        "pose": "a" * 12, "pose_name": "idle", "yaw": 0.0, "frame": 0,
        # Additive: an importer that ignores these reads the sheet exactly as
        # it did before, which is why the version does not move.
        "pivot_x": 32.0, "pivot_y": 64.0, "trim": None,
    }


def test_sidecar_carries_a_pivot_per_cell():
    layout = sheetlib.plan([], frame_size=128, yaws=4)
    meta = sheetlib.sidecar(
        layout, sheet_id="a" * 12, source_job="b" * 12, image="s.png",
        created=1.0, pivot=(64.0, 118.0),
    )
    assert all(c["pivot_x"] == 64.0 and c["pivot_y"] == 118.0 for c in meta["cells"])


def test_pivot_defaults_to_the_cell_centre_bottom_when_unmeasured():
    layout = sheetlib.plan([], frame_size=128, yaws=4)
    meta = sheetlib.sidecar(
        layout, sheet_id="a" * 12, source_job="b" * 12, image="s.png", created=1.0
    )
    assert meta["cells"][0]["pivot_x"] == 64.0
    assert meta["cells"][0]["pivot_y"] == 128.0


def test_sidecar_carries_the_trim_it_was_measured(tmp_path):
    layout = sheetlib.plan([], frame_size=64, yaws=2)
    meta = sheetlib.sidecar(
        layout, sheet_id="a" * 12, source_job="b" * 12, image="s.png", created=1.0,
        trims={0: {"x": 1, "y": 2, "w": 3, "h": 4}},
    )
    assert meta["cells"][0]["trim"] == {"x": 1, "y": 2, "w": 3, "h": 4}
    # A cell nothing was measured for says so, rather than claiming full bleed.
    assert meta["cells"][1]["trim"] is None


def test_trim_measures_the_alpha_bounding_box():
    frame = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    frame.paste((255, 0, 0, 255), (10, 20, 30, 50))
    assert sheetlib.measure_trim(frame) == {"x": 10, "y": 20, "w": 20, "h": 30}


def test_trim_of_an_empty_frame_is_none():
    assert sheetlib.measure_trim(Image.new("RGBA", (64, 64), (0, 0, 0, 0))) is None


def test_pack_returns_a_trim_for_every_cell(tmp_path):
    """Measured while packing because every frame is already open and decoded
    there; a second pass over the atlas would be the same pixels twice."""
    layout = sheetlib.plan([], frame_size=64, yaws=2)
    frames = {}
    for cell in layout.cells:
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        if cell.index == 0:
            img.paste((0, 255, 0, 255), (8, 16, 24, 40))
        path = tmp_path / f"{cell.index}.png"
        img.save(path)
        frames[cell.index] = path
    trims = sheetlib.pack(layout, frames, tmp_path / "sheet.png")
    assert trims[0] == {"x": 8, "y": 16, "w": 16, "h": 24}
    assert trims[1] is None


def test_sidecar_is_plain_json():
    layout = sheetlib.plan([])
    meta = sheetlib.sidecar(
        layout, sheet_id="c" * 12, source_job="d" * 12, image="c.png", created=1.0
    )
    assert json.loads(json.dumps(meta)) == meta


# --- padding geometry ---------------------------------------------------------
#
# ``Plan.padding`` is additive and defaults to 0, which is what keeps every 3D
# sheet -- none of whose callers ever set it -- computing the exact width and
# height it always did: the formula below collapses to the old ``columns *
# cell_w`` at ``padding == 0``.


def test_zero_padding_is_the_old_width_and_height_formula():
    plan = sheetlib.plan([], frame_size=64, yaws=4)
    assert plan.padding == 0
    assert plan.width == plan.columns * plan.cell_w
    assert plan.height == plan.rows * plan.cell_h


def test_padding_adds_a_border_and_a_gutter_per_cell():
    """The packwright grid contract, restated: ``padding + columns * (cell +
    padding)`` -- one border round the outside, one gutter between cells."""
    from dataclasses import replace

    plan = sheetlib.plan([], frame_size=64, yaws=4)
    padded = replace(plan, padding=5)
    assert padded.width == 5 + padded.columns * (padded.cell_w + 5)
    assert padded.height == 5 + padded.rows * (padded.cell_h + 5)
    assert padded.width == 5 + 4 * 69
    assert padded.height == 5 + 1 * 69


# --- extrude_edges --------------------------------------------------------------
#
# Ported from ``studio.packwright.compose._extrude``; asserted the same way
# that module's own tests assert it, pixel by pixel, because the ordering
# (sides, then top/bottom across the widened span) is what makes the corners
# come free rather than being a separate case -- and is exactly the kind of
# thing that only shows wrong on a filtered GPU sample.

RED = (255, 0, 0, 255)


def _solid_atlas(size: int, x: int, y: int, w: int, h: int, colour=RED) -> np.ndarray:
    atlas = np.zeros((size, size, 4), dtype=np.uint8)
    atlas[y : y + h, x : x + w] = colour
    return atlas


def test_extrude_edges_replicates_the_border_including_the_corners():
    atlas = _solid_atlas(12, 4, 4, 4, 4)
    sheetlib.extrude_edges(atlas, 4, 4, 4, 4, 2)
    grown = atlas[2:10, 2:10]
    assert grown.shape == (8, 8, 4)
    assert (grown == np.array(RED, np.uint8)).all()


def test_extrude_edges_stays_inside_the_gutter_a_neighbour_left():
    """Two 4x4 sprites with a 4px gutter (``padding == extrude * 2``) must not
    bleed into each other -- the whole reason that guarantee exists -- and
    each one has exactly ``extrude`` room against the atlas edge too."""
    atlas = np.zeros((8, 16, 4), dtype=np.uint8)
    BLUE = (0, 0, 255, 255)
    atlas[2:6, 2:6] = RED
    atlas[2:6, 10:14] = BLUE
    sheetlib.extrude_edges(atlas, 2, 2, 4, 4, 2)  # room to the atlas edge too
    sheetlib.extrude_edges(atlas, 10, 2, 4, 4, 2)
    assert (atlas[2:6, 0:8] == np.array(RED, np.uint8)).all()
    assert (atlas[2:6, 8:16] == np.array(BLUE, np.uint8)).all()


def test_extrude_edges_zero_margin_is_a_no_op():
    atlas = _solid_atlas(8, 2, 2, 2, 2)
    before = atlas.copy()
    sheetlib.extrude_edges(atlas, 2, 2, 2, 2, 0)
    assert np.array_equal(atlas, before)


# --- framing, against a fake Blender ----------------------------------------
#
# op_sheet decides the ortho window before it renders anything, and that
# decision is the whole of this section: which corners it unions, where the
# orbit axis sits. None of it needs
# pixels, and putting it behind the gpu marker would mean the framing rules
# were only ever checked on a machine with a card in it.
#
# What is faked is Blender's *scene* -- objects with bounding boxes, an
# armature whose evaluated box depends on the pose applied to it -- and the two
# calls that need a real ``bpy`` context, ``_aim_camera`` and ``_project``.
# ``_ortho_pixel`` below restates exactly the arithmetic those two perform for
# an orthographic camera, so a framing that moved the orbit axis would move the
# pixel here for the same reason it would move it in Blender.

_IDENTITY4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _box_corners(lo, hi):
    return [
        (x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])
    ]


class _FakeEvaluated:
    def __init__(self, box):
        self.matrix_world = _IDENTITY4
        self.bound_box = _box_corners(*box)


class _FakeMesh:
    """A render mesh whose *evaluated* box follows the pose, and whose own
    ``bound_box`` does not -- which is precisely Blender's behaviour for a
    skinned mesh, and the distinction the pre-pass exists for."""

    type = "MESH"
    hide_render = False

    def __init__(self, rig, rest, posed=None):
        self._rig, self._rest, self._posed = rig, rest, dict(posed or {})
        self.matrix_world = _IDENTITY4

    @property
    def bound_box(self):
        return _box_corners(*self._rest)

    def evaluated_get(self, _depsgraph):
        return _FakeEvaluated(self._posed.get(self._rig.pose_key, self._rest))


class _FakeArmature:
    type = "ARMATURE"
    hide_render = True

    def __init__(self):
        self.pose_key = None
        self.matrix_world = _IDENTITY4


class _FakeCamera:
    def __init__(self, extent, distance):
        self.extent, self.distance = extent, distance
        self.aims = []


def _ortho_pixel(cam, point, size):
    """Blender's orthographic projection, restated.

    ``_aim_camera`` builds the camera basis as ``Rz(yaw) @ Rx(90 - elevation)``
    and ``_setup_camera`` sets ``ortho_scale`` to the extent, so a world point's
    pixel is its offset from the camera's own axes over that extent. Written
    out here rather than imported because ``bpy_extras`` needs a real scene --
    and because a projection this section can compute is what makes "the pivot
    lands on the same pixel at every yaw" a claim a test can make.
    """
    centre, yaw_deg, elevation_deg, _distance = cam.aims[-1]
    yaw, elevation = math.radians(yaw_deg), math.radians(elevation_deg)
    right = (math.cos(yaw), math.sin(yaw), 0.0)
    up = (
        -math.sin(yaw) * math.sin(elevation),
        math.cos(yaw) * math.sin(elevation),
        math.cos(elevation),
    )
    d = [float(point[i]) - float(centre[i]) for i in range(3)]
    dx = sum(d[i] * right[i] for i in range(3))
    dy = sum(d[i] * up[i] for i in range(3))
    return ((0.5 + dx / cam.extent) * size, (0.5 - dy / cam.extent) * size)


class _FakeBpySheet:
    """Just enough ``bpy`` for ``op_sheet`` to walk its cells."""

    def __init__(self, objects):
        renders = type("R", (), {"filepath": ""})()
        scene = type("S", (), {"objects": list(objects), "render": renders})()
        self.rendered = []
        outer = self

        class _Render:
            @staticmethod
            def render(write_still=False):
                outer.rendered.append(scene.render.filepath)

        class _Import:
            @staticmethod
            def gltf(**_kwargs):
                pass

        self.ops = type("O", (), {"render": _Render, "import_scene": _Import})()
        self.context = type(
            "C",
            (),
            {
                "scene": scene,
                "view_layer": type("V", (), {"update": staticmethod(lambda: None)})(),
                "evaluated_depsgraph_get": staticmethod(lambda: "depsgraph"),
            },
        )()


def _fake_sheet_render(monkeypatch, objects):
    """Point ``op_sheet`` at a fake scene. -> ``(bpy, cameras)``."""
    from realmspinner.pipelines import blender_worker

    bpy = _FakeBpySheet(objects)
    cameras = []
    rig = next((o for o in objects if o.type == "ARMATURE"), None)

    def fake_setup_camera(_bpy, extent, distance):
        cameras.append(_FakeCamera(extent, distance))
        return cameras[-1]

    def fake_aim(cam, centre, yaw, elevation, distance):
        cam.aims.append((list(centre), float(yaw), float(elevation), float(distance)))

    def fake_apply_pose(_arm, bones, _space="node"):
        # The fake's pose is a name, not rotations: ``_FakeMesh`` looks its
        # evaluated box up by it. The first bone name in the cell is the pose.
        rig.pose_key = next(iter(bones), None)
        return len(bones), []

    monkeypatch.setattr(blender_worker, "_reset_scene", lambda _bpy: None)
    monkeypatch.setattr(blender_worker, "_purge_import_helpers", lambda _bpy: None)
    monkeypatch.setattr(blender_worker, "_setup_render", lambda *a, **k: None)
    monkeypatch.setattr(blender_worker, "_make_flat", lambda _bpy: None)
    monkeypatch.setattr(blender_worker, "_make_lit", lambda *a, **k: None)
    monkeypatch.setattr(blender_worker, "_world", lambda *a, **k: None)
    monkeypatch.setattr(blender_worker, "_setup_camera", fake_setup_camera)
    monkeypatch.setattr(blender_worker, "_aim_camera", fake_aim)
    monkeypatch.setattr(
        blender_worker, "_project", lambda _bpy, cam, point, size: _ortho_pixel(cam, point, size)
    )
    monkeypatch.setattr(
        blender_worker, "_reset_pose", lambda _arm: setattr(rig, "pose_key", None)
    )
    monkeypatch.setattr(blender_worker, "_apply_pose", fake_apply_pose)
    return bpy, cameras


def _sheet_spec_for(tmp_path, cells):
    glb = tmp_path / "model.glb"
    glb.write_bytes(b"fake-glb")
    frames = tmp_path / "frames"
    return blender_spec.sheet_spec(
        glb, frames, cells, frame_size=64, elevation=0.0, lighting="flat"
    )


REST_BOX = ((-0.3, -0.2, 0.0), (0.3, 0.2, 2.0))


def test_union_framing_never_clips_the_attack_apex(tmp_path, monkeypatch):
    """The defect: the camera was framed once from the *rest* bounding box, so
    a pose whose apex leaves it -- an overhead attack wind, a jump -- was
    clipped on every cell of that run, with nothing in the sheet to say so.

    Three claims in one, because they are one decision: the window grows to
    hold the posed apex, the apex is then inside the frame, and the orbit axis
    does *not* move -- the pivot lands on the same pixel at every yaw, which is
    the property an engine placing a sprite by it depends on and the reason the
    union is not simply re-centred on itself.
    """
    from realmspinner.pipelines import blender_worker

    apex = (-0.3, -0.2, 0.0), (0.3, 0.2, 3.2)
    rig = _FakeArmature()
    mesh = _FakeMesh(rig, REST_BOX, {"jump": apex})
    cells = [
        {"index": 0, "yaw": 0.0, "pose": None, "frame": 0, "bones": {}},
        {"index": 1, "yaw": 90.0, "pose": None, "frame": 0, "bones": {}},
        {"index": 2, "yaw": 0.0, "pose": "p", "frame": 0, "bones": {"jump": IDENTITY}},
        {"index": 3, "yaw": 90.0, "pose": "p", "frame": 0, "bones": {"jump": IDENTITY}},
    ]

    bpy, cameras = _fake_sheet_render(monkeypatch, [mesh, rig])
    result = blender_worker.op_sheet(bpy, _sheet_spec_for(tmp_path, cells))
    widened = cameras[0]

    bpy2, rest_only = _fake_sheet_render(monkeypatch, [_FakeMesh(_FakeArmature(), REST_BOX), rig])
    blender_worker.op_sheet(bpy2, _sheet_spec_for(tmp_path, cells[:2]))
    assert widened.extent > rest_only[0].extent

    # The apex is inside the frame it is rendered in. Against the rest-framed
    # window it is not: 3.2 is 2.2 above a centre of 1.0 in a 2.24 window.
    px, py = _ortho_pixel(widened, (0.0, 0.0, 3.2), 64)
    assert 0.0 <= px <= 64.0 and 0.0 <= py <= 64.0

    # ...and the pivot does not move as the subject turns.
    pivot_point = (widened.aims[0][0][0], widened.aims[0][0][1], REST_BOX[0][2])
    seen = set()
    for i in range(1, len(widened.aims)):
        cam = _FakeCamera(widened.extent, widened.distance)
        cam.aims = [widened.aims[i]]
        seen.add(tuple(round(v, 9) for v in _ortho_pixel(cam, pivot_point, 64)))
    assert len(seen) == 1
    assert tuple(round(v, 9) for v in result["pivot"]) in seen
    assert result["framing"]["union_bounds"]["max"][2] == pytest.approx(3.2)


def test_a_rest_only_sheet_frames_exactly_as_before(tmp_path, monkeypatch):
    """The union is universal, not gated on which animations were asked for --
    which is only safe while a sheet with nothing to union comes out exactly as
    it did. This is that arithmetic, spelled the way op_sheet used to spell it.
    """
    from realmspinner.pipelines import blender_worker

    rig = _FakeArmature()
    mesh = _FakeMesh(rig, REST_BOX)
    cells = [{"index": 0, "yaw": 0.0, "pose": None, "frame": 0, "bones": {}}]
    bpy, cameras = _fake_sheet_render(monkeypatch, [mesh, rig])
    result = blender_worker.op_sheet(bpy, _sheet_spec_for(tmp_path, cells))

    span = [hi - lo for lo, hi in zip(*REST_BOX, strict=True)]
    old = max(math.hypot(span[0], span[1]), span[2], 1e-6) * sheetlib.FRAME_MARGIN
    assert cameras[0].extent == old
    assert cameras[0].aims[0][0] == [0.0, 0.0, 1.0]
    assert result["bounds"] == {"min": [-0.3, -0.2, 0.0], "max": [0.3, 0.2, 2.0]}


# --- the real renderer ------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("lighting", ["flat", "lit"])
def test_a_rendered_sheet_actually_contains_eight_distinct_views(tmp_path, lighting):
    """The half only Blender can do: pixels in the frames, transparency around
    them, and a different silhouette in every column."""
    pytest.importorskip("bpy")
    import bpy

    from realmspinner.pipelines import blender_worker

    # Deliberately lopsided, so a camera that failed to turn would show up as
    # eight identical cells rather than eight plausible ones.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    bpy.context.object.scale = (0.3, 0.2, 1.0)
    bpy.ops.mesh.primitive_cone_add(radius1=0.35, depth=0.6, location=(0, -0.5, 0.9))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(tmp_path / "model.glb"), export_format="GLB")

    layout = sheetlib.plan([], frame_size=64, elevation=25.0, lighting=lighting)
    frames_dir = tmp_path / "frames"
    spec = blender_spec.sheet_spec(
        tmp_path / "model.glb",
        frames_dir,
        [{"index": c.index, "yaw": c.yaw, "pose": None, "bones": {}} for c in layout.cells],
        frame_size=64,
        elevation=25.0,
        lighting=lighting,
    )
    result = blender_worker.op_sheet(bpy, spec)
    assert result["frames"] == [c.index for c in layout.cells]

    frames = {c.index: frames_dir / f"{c.index:04d}.png" for c in layout.cells}
    out = tmp_path / "sheet.png"
    sheetlib.pack(layout, frames, out)
    with Image.open(out) as atlas:
        assert atlas.size == (512, 64)
        cells = [atlas.crop((c.x, c.y, c.x + 64, c.y + 64)) for c in layout.cells]
        # Every column shows something...
        for i, cell in enumerate(cells):
            opaque = sum(1 for v in cell.getchannel("A").get_flattened_data() if v > 8)
            assert 0.02 < opaque / (64 * 64) < 0.9, f"column {i} covers {opaque} px"
        # ...the background is not part of it...
        assert atlas.getpixel((1, 1))[3] == 0
        # ...and no two columns are the same picture.
        assert len({c.tobytes() for c in cells}) == len(cells)


@pytest.mark.gpu
def test_a_rigged_subject_is_framed_by_its_own_size(tmp_path):
    """Regression: Blender's glTF importer adds bone-shape widgets in a
    glTF_not_exported collection. They never render, but they were still in
    scene.objects -- a unit icosphere among them trebled the measured bounds
    and framed every rigged sheet's subject at a third of its proper size."""
    pytest.importorskip("bpy")
    import bpy

    from realmspinner.pipelines import blender_worker

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    bpy.context.object.scale = (0.3, 0.2, 1.0)   # 0.6 x 0.4 x 2.0
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(tmp_path / "model.glb"), export_format="GLB")
    blender_worker.op_rig(bpy, blender_spec.rig_spec(tmp_path, "humanoid"))
    rig_store.finalize_rig(tmp_path)

    layout = sheetlib.plan([], frame_size=128, elevation=0.0)
    frames_dir = tmp_path / "frames"
    blender_worker.op_sheet(
        bpy,
        blender_spec.sheet_spec(
            tmp_path / "rig.glb",
            frames_dir,
            [{"index": c.index, "yaw": c.yaw, "pose": None, "bones": {}} for c in layout.cells],
            frame_size=128,
            elevation=0.0,
            lighting="flat",
        ),
    )
    with Image.open(frames_dir / "0000.png") as frame:
        mask = frame.convert("RGBA").getchannel("A").point(lambda v: 255 if v > 8 else 0)
        box = mask.getbbox()
    height = box[3] - box[1]
    # 2.0 tall inside a 2.24 ortho window is ~89% of the frame; anything near a
    # third of that means something else got into the bounds again.
    assert height / 128 > 0.75, f"subject only fills {height}/128 px"


@pytest.mark.gpu
def test_the_reported_pivot_sits_at_the_subjects_feet_in_every_direction(tmp_path):
    """The sidecar's pivot is only worth anything if an engine can place a
    sprite by it without the subject drifting as it turns.

    Rendered at elevation 0 with a lopsided subject, so a pivot computed from
    the wrong point or projected with the wrong sign would land somewhere the
    silhouette is not. Checked against each cell's own alpha bbox, which is the
    same rectangle a Pygame ``subsurface`` blit would honour.
    """
    pytest.importorskip("bpy")
    import bpy

    from realmspinner.pipelines import blender_worker

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    bpy.context.object.scale = (0.3, 0.2, 1.0)
    bpy.ops.mesh.primitive_cone_add(radius1=0.35, depth=0.6, location=(0, -0.5, 0.9))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(tmp_path / "model.glb"), export_format="GLB")

    layout = sheetlib.plan([], frame_size=128, elevation=0.0)
    frames_dir = tmp_path / "frames"
    spec = blender_spec.sheet_spec(
        tmp_path / "model.glb",
        frames_dir,
        [{"index": c.index, "yaw": c.yaw, "pose": None, "bones": {}} for c in layout.cells],
        frame_size=128,
        elevation=0.0,
        lighting="flat",
    )
    result = blender_worker.op_sheet(bpy, spec)
    pivot = result["pivot"]

    frames = {c.index: frames_dir / f"{c.index:04d}.png" for c in layout.cells}
    trims = sheetlib.pack(layout, frames, tmp_path / "sheet.png")
    meta = sheetlib.sidecar(
        layout, sheet_id="a" * 12, source_job="b" * 12, image="s.png", created=1.0,
        pivot=(pivot[0], pivot[1]), trims=trims,
    )

    px, py = meta["cells"][0]["pivot_x"], meta["cells"][0]["pivot_y"]
    # Horizontally centred, and on the floor: at elevation 0 the bottom of the
    # silhouette *is* the ground plane.
    assert px == pytest.approx(64.0, abs=1.5)
    for cell in meta["cells"]:
        trim = cell["trim"]
        assert trim is not None, f"column {cell['column']} rendered nothing"
        # Every cell carries the same pivot -- that stability is the point.
        assert (cell["pivot_x"], cell["pivot_y"]) == (px, py)
        bottom = trim["y"] + trim["h"]
        assert abs(py - bottom) <= 2, f"column {cell['column']}: pivot {py} vs floor {bottom}"
        assert trim["x"] <= px <= trim["x"] + trim["w"]


# --- the non-square seam ------------------------------------------------------
#
# ``Plan`` grew ``frame_w``/``frame_h`` so an Inker animation -- whose frame is a
# canvas the user chose, not a camera -- can be packed by the same code. Every
# one of those changes is behaviour-preserving by construction (``frame_w or
# frame_size``), and the pin below is what says so from outside the module.

#: ``json.dumps(sidecar(...))`` captured from HEAD before the non-square fields
#: were added. One string equality covers the values, the key *order* and --
#: the part a field-by-field check would miss -- the absence of any new key.
SQUARE_SIDECAR = (
    '{"version": 1, "id": "cccccccccccc", "name": "n", "source_job": "dddddddddddd",'
    ' "created": 1.5, "image": "c.png", "frame_size": 64, "columns": 3, "rows": 1,'
    ' "width": 192, "height": 64, "elevation": 25.0, "lighting": "lit",'
    ' "yaws": [0.0, 120.0, 240.0], "poses": [{"id": "aaaaaaaaaaaa", "name": "idle"}],'
    ' "cells": [{"index": 0, "row": 0, "column": 0, "x": 0, "y": 0, "w": 64, "h": 64,'
    ' "pose": "aaaaaaaaaaaa", "pose_name": "idle", "yaw": 0.0, "frame": 0,'
    ' "pivot_x": 32.0, "pivot_y": 64.0, "trim": {"x": 1, "y": 2, "w": 3, "h": 4}},'
    ' {"index": 1, "row": 0, "column": 1, "x": 64, "y": 0, "w": 64, "h": 64,'
    ' "pose": "aaaaaaaaaaaa", "pose_name": "idle", "yaw": 120.0, "frame": 0,'
    ' "pivot_x": 32.0, "pivot_y": 64.0, "trim": null},'
    ' {"index": 2, "row": 0, "column": 2, "x": 128, "y": 0, "w": 64, "h": 64,'
    ' "pose": "aaaaaaaaaaaa", "pose_name": "idle", "yaw": 240.0, "frame": 0,'
    ' "pivot_x": 32.0, "pivot_y": 64.0, "trim": null}]}'
)


def _square_sidecar() -> dict:
    layout = sheetlib.plan(
        [{"id": "a" * 12, "name": "idle"}],
        frame_size=64,
        elevation=25.0,
        lighting="lit",
        yaws=3,
    )
    return sheetlib.sidecar(
        layout,
        sheet_id="c" * 12,
        source_job="d" * 12,
        image="c.png",
        created=1.5,
        name="n",
        trims={0: {"x": 1, "y": 2, "w": 3, "h": 4}},
    )


def test_a_square_sidecar_is_byte_for_byte_what_it_always_was():
    assert json.dumps(_square_sidecar()) == SQUARE_SIDECAR


def test_plan_never_produces_a_non_square_grid():
    """Which is what makes every ``cell_w``/``cell_h`` substitution above a
    no-op for the 3D path: ``frame_w or frame_size`` is ``frame_size``."""
    layout = sheetlib.plan([], frame_size=128, yaws=4)
    assert (layout.frame_w, layout.frame_h) == (0, 0)
    assert (layout.cell_w, layout.cell_h) == (128, 128)
    assert (layout.width, layout.height) == (512, 128)


def _oblong(**kwargs) -> sheetlib.Plan:
    cells = tuple(
        sheetlib.Cell(
            index=i, row=i // 3, column=i % 3, x=(i % 3) * 50, y=(i // 3) * 80,
            pose=None, pose_name="rest", yaw=0.0, frame=i,
        )
        for i in range(6)
    )
    return sheetlib.Plan(
        frame_size=0, columns=3, rows=2, yaws=(0.0,), elevation=0.0,
        lighting="flat", poses=({"id": None, "name": "rest"},), cells=cells,
        frame_w=50, frame_h=80, **kwargs,
    )


def test_a_non_square_plan_sizes_its_atlas_from_the_cell():
    layout = _oblong()
    assert (layout.cell_w, layout.cell_h) == (50, 80)
    assert (layout.width, layout.height) == (150, 160)


def test_a_non_square_sidecar_says_frame_size_zero_and_carries_the_real_one():
    """Zero is a loud wrong answer. Putting the width in ``frame_size`` would
    let an importer that reads only that key slice the atlas correctly across
    and wrongly down, which is worse than failing."""
    meta = sheetlib.sidecar(
        _oblong(), sheet_id="s", source_job="j", image="s.png", created=0.0
    )
    assert meta["frame_size"] == 0
    assert (meta["frame_w"], meta["frame_h"]) == (50, 80)
    assert all(cell["w"] == 50 and cell["h"] == 80 for cell in meta["cells"])
    # The pivot follows the cell, not the (absent) square size.
    assert (meta["cells"][0]["pivot_x"], meta["cells"][0]["pivot_y"]) == (25.0, 80.0)


def test_a_square_sidecar_carries_no_frame_w_at_all():
    assert "frame_w" not in _square_sidecar()


def test_the_animation_block_is_absent_unless_it_is_given():
    assert "animation" not in _square_sidecar()
    meta = sheetlib.sidecar(
        _oblong(),
        sheet_id="s", source_job="j", image="s.png", created=0.0,
        animation={"frames": [], "tags": []},
    )
    assert meta["animation"] == {"frames": [], "tags": []}
    assert list(meta)[-1] == "animation"


def test_pack_composites_non_square_cells(tmp_path: Path):
    layout = _oblong()
    frames = {}
    for cell in layout.cells:
        path = tmp_path / f"{cell.index}.png"
        Image.new("RGBA", (50, 80), (cell.index * 40, 0, 0, 255)).save(path)
        frames[cell.index] = path

    sheetlib.pack(layout, frames, tmp_path / "atlas.png")

    with Image.open(tmp_path / "atlas.png") as atlas:
        assert atlas.size == (150, 160)
        assert atlas.getpixel((0, 0)) == (0, 0, 0, 255)
        # Column 100//50 = 2, row 100//80 = 1, so cell 1*3+2 = 5.
        assert atlas.getpixel((100, 100)) == (200, 0, 0, 255)


def test_the_extracted_atlas_guard_is_the_one_plan_uses():
    with pytest.raises(ValueError, match="8192"):
        sheetlib.check_atlas_size(sheetlib.MAX_ATLAS_PX + 1, 10)
    sheetlib.check_atlas_size(sheetlib.MAX_ATLAS_PX, sheetlib.MAX_ATLAS_PX)
