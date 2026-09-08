"""Which way a mesh faces, and what carries that answer to the renderer.

Every directional sheet rendered before 2026-09-08 assumed the subject's
forward was -Y. That is true of the shipped skeleton templates and true of
nothing a reconstruction returns, and it could not be fixed by measuring: the
2026-08-05 view-calibration sweep over 37 finished jobs
(``docs/measurements/2026-08-04-view-calibration.md``) found each mesh's
best-matching view scattered by 330 degrees -- effectively uniform, with two
metrics agreeing with each other no better than chance. So the front is a
human's press, stored as ``params["front_yaw"]`` on the model row.

The claims here are the ones that make that press safe: the number means the
same thing at both ends, a sheet that has none renders exactly what it always
did, the value is snapshotted at the door rather than read live, and a cell's
own ``yaw`` keeps meaning *facing* rather than *camera angle*.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from warlock import _q_troupe
from warlock.pipelines import charsheet, sheetcheck
from warlock.pipelines import sheet as sheetlib
from warlock.service import jobs as svc_jobs
from warlock.service import sheets as svc_sheets
from warlock.service import troupe as svc_troupe
from warlock.service.errors import Invalid


def _mesh(svc, *, rigged: bool = True, template: str = "humanoid") -> str:
    job_id = svc.store.create("image", "a ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    if rigged:
        (job_dir / "rig.glb").write_bytes(b"fake-glb")
        (job_dir / "rig.json").write_text(json.dumps({"template": template}), "utf-8")
    svc.store.set_status(job_id, "done")
    return job_id


# -- the angle means one thing at both ends -----------------------------------


def test_the_viewport_camera_and_the_sheet_camera_are_the_same_angle():
    """The whole feature rests on this and nothing asserted it.

    A user orbits in Poser or the 3D viewport, and the number that press
    records is handed to a *sheet* camera in another module written years
    apart: ``viewer/camera.py`` is spherical about +Y with a polar angle, and
    ``viewer/sheet.py`` is yaw-and-elevation about the same pole. Both
    docstrings claim yaw 0 sits on +Z, but a claim in prose is what a
    quarter-turn regression looks like right up until somebody renders a
    sheet. With ``phi = pi/2 - elevation`` the two are the same function, so
    ``front_yaw`` needs no sign flip and no origin shift -- and here that
    stops being an assertion and becomes a measurement.
    """
    from warlock.studio.viewer import camera as viewer_camera
    from warlock.studio.viewer import sheet as viewer_sheet

    centre = np.zeros(3)
    for degrees in (0.0, 45.0, 137.5, 270.0, 359.0):
        for elevation in (0.0, math.radians(30.0), math.radians(60.0)):
            cam = viewer_camera.Camera()
            cam.target = centre.copy()
            cam.distance = 3.0
            cam.theta = math.radians(degrees)
            cam.phi = math.pi / 2 - elevation
            expected = viewer_sheet.camera_position(centre, degrees, elevation, 3.0)
            assert np.allclose(cam.position, expected), (
                f"the two cameras disagree at yaw {degrees}, elevation {elevation}"
            )


# -- the write door -----------------------------------------------------------


def test_a_front_is_normalised_into_one_turn(svc):
    """A viewport hands over whatever its damped goal happens to hold, which
    for a camera the user has spun a few times is well outside [0, 360).
    Normalising once at the door rather than at each reader is what lets every
    site downstream treat the value as a plain angle."""
    job_id = _mesh(svc)
    assert svc_jobs.set_front_yaw(svc, job_id, 405.0)["front_yaw"] == 45.0
    assert svc.store.get(job_id)["params"]["front_yaw"] == 45.0
    assert svc_jobs.set_front_yaw(svc, job_id, -90.0)["front_yaw"] == 270.0
    assert svc.store.get(job_id)["params"]["front_yaw"] == 270.0


def test_setting_the_front_back_to_zero_removes_the_key(svc):
    """Absent and zero have to be the same row, because every reader
    downstream is a bare ``params.get("front_yaw")`` guarded on truthiness.
    Storing 0.0 would leave a key meaning exactly what no key means, and the
    first reader to test ``"front_yaw" in params`` instead would then be right
    about a distinction that does not exist."""
    job_id = _mesh(svc)
    svc_jobs.set_front_yaw(svc, job_id, 137.0)
    assert "front_yaw" in svc.store.get(job_id)["params"]
    svc_jobs.set_front_yaw(svc, job_id, 0.0)
    assert "front_yaw" not in svc.store.get(job_id)["params"]


def test_an_unrigged_prop_can_carry_a_front(svc):
    """Checked against a finished mesh, never against a rig. Poser can only
    bind a rigged ``rig.glb``, which is why the shared 3D viewport carries the
    same control -- and a prop reaching the plain ``sheet`` door needs no rig
    at all, so refusing one here would put a front out of reach of exactly the
    assets the second door exists for."""
    job_id = _mesh(svc, rigged=False)
    assert svc_jobs.set_front_yaw(svc, job_id, 90.0)["front_yaw"] == 90.0


def test_a_front_needs_a_finished_mesh(svc):
    job_id = svc.store.create("image", "a ranger", {}, stage="model")
    with pytest.raises(Invalid):
        svc_jobs.set_front_yaw(svc, job_id, 90.0)


# -- snapshotted at the door, never read live ---------------------------------


def test_a_character_sheet_snapshots_the_meshs_front(svc):
    job_id = _mesh(svc)
    svc_jobs.set_front_yaw(svc, job_id, 137.0)
    made = svc_troupe.create_charsheet(svc, job_id, logical_size=32)
    assert svc.store.get(made["id"])["params"]["front_yaw"] == 137.0


def test_a_mesh_with_no_front_mints_the_row_it_always_did(svc):
    """The byte-identity claim, at the door. A sheet of a mesh nobody has
    oriented must carry no new key at all, or a stored row from before
    2026-09-08 reads as a different request from an identical one made
    today."""
    job_id = _mesh(svc)
    made = svc_troupe.create_charsheet(svc, job_id, logical_size=32)
    assert "front_yaw" not in svc.store.get(made["id"])["params"]


def test_a_plain_sprite_sheet_snapshots_it_too(svc):
    """The front is a fact about the asset, not a Troupe setting. The plain
    ``sheet`` kind spins the same camera around the same mesh, so a front
    honoured by one renderer and ignored by the other would not be an asset
    property at all."""
    job_id = _mesh(svc)
    svc_jobs.set_front_yaw(svc, job_id, 42.0)
    made = svc_sheets.create_sheet(svc, job_id, frame_size=64, yaws=8)
    assert svc.store.get(made["id"])["params"]["front_yaw"] == 42.0


def test_a_later_press_does_not_reach_a_sheet_already_requested(svc):
    """Why the door snapshots instead of letting the worker read the source
    row live. A subset re-render composites its cells back onto the sheet it
    was built from and must therefore be shot from the same camera -- the
    argument ``frame_margin``'s sidecar round-trip already makes, arriving by
    the other road. Read live, a user who re-pressed the viewport button
    between the base sheet and the re-render would get twelve cells from one
    front composited onto 244 from another."""
    job_id = _mesh(svc)
    svc_jobs.set_front_yaw(svc, job_id, 137.0)
    base = svc_troupe.create_charsheet(svc, job_id, logical_size=32)
    svc_jobs.set_front_yaw(svc, job_id, 12.0)
    assert svc.store.get(base["id"])["params"]["front_yaw"] == 137.0


def test_a_reroll_drops_a_front_measured_against_the_mesh_it_replaces(svc):
    """A reroll reconstructs *different* geometry, and the calibration sweep
    that motivates this whole feature is the evidence that the new mesh's
    front will not be the old one's: 37 jobs, 330-degree scatter, no
    relationship worth carrying. Keeping it would silently orient the next
    reconstruction by a number nobody measured against it."""
    job_id = _mesh(svc)
    (svc.job_dir(job_id) / "input.png").write_bytes(b"fake-png")
    svc_jobs.set_front_yaw(svc, job_id, 137.0)
    made = svc_jobs.rerun_job(svc, job_id)
    assert "front_yaw" not in svc.store.get(made["id"])["params"]


# -- the sidecar keeps the two numbers apart ----------------------------------


def test_the_camera_block_omits_the_front_when_there_is_none():
    assert "front_yaw" not in _q_troupe._camera_meta(30.0, pixel_size=32, margin=1.12)


def test_the_camera_block_records_the_front_beside_the_elevation():
    meta = _q_troupe._camera_meta(30.0, pixel_size=32, margin=1.12, front_yaw=137.0)
    assert meta["front_yaw"] == 137.0
    assert meta["elevation"] == 30.0


def test_a_sidecar_without_a_front_grows_no_key():
    plan = sheetlib.plan([], frame_size=64, yaws=8)
    fixed = {"sheet_id": "s1", "source_job": "j1", "image": "s1.png", "created": 0.0}
    assert "front_yaw" not in sheetlib.sidecar(plan, **fixed)
    assert sheetlib.sidecar(plan, **fixed, front_yaw=42.0)["front_yaw"] == 42.0


def test_a_cell_yaw_still_means_facing_and_the_check_says_so():
    """The split the whole design rests on: a cell's ``yaw`` is the direction
    the sprite faces, and ``camera.front_yaw`` alone is what was added to
    reach it. Nothing in the app reads a cell yaw as a camera angle, so
    nothing would visibly break if a future edit "fixed" the cells to match
    the camera -- it would simply publish two contradictory front directions
    for one sheet, ``meta["troupe"].runs[].yaw`` saying one and
    ``cells[].yaw`` the other. This is the tripwire for that."""
    meta = {
        "troupe": {
            "runs": [
                {"movement": "idle", "direction": "front", "yaw": 0.0, "start": 0, "end": 1}
            ]
        },
        "camera": {"front_yaw": 137.0},
        "cells": [{"index": 0, "yaw": 0.0}, {"index": 1, "yaw": 0.0}],
    }
    assert not any("front_yaw" in f for f in sheetcheck.metadata_findings(meta))

    drifted = {**meta, "cells": [{"index": 0, "yaw": 137.0}, {"index": 1, "yaw": 0.0}]}
    findings = sheetcheck.metadata_findings(drifted)
    assert any("front_yaw" in f for f in findings), findings


def test_the_direction_table_itself_never_rotates():
    """``charsheet.resolve_layout`` refuses any direction list that is not
    literally a ``DIRECTION_PRESETS`` value, so a rotated layout cannot exist
    -- which is precisely why the offset rides the cell dict the worker sends
    to Blender rather than the plan the sidecar publishes."""
    assert charsheet.DIRECTIONS[0] == ("front", 0.0)
    rotated = tuple((name, (yaw + 137.0) % 360.0) for name, yaw in charsheet.DIRECTIONS)
    assert rotated not in charsheet.DIRECTION_PRESETS.values()
