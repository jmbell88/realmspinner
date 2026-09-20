"""service-03, the 2026-09-14 audit: ``service.rig.create_rig`` -- the door
Library's "Rig this mesh" and Poser's "Re-rig" both call -- never checked
``rig_in_flight`` before minting a fresh rig row, unlike ``troupe.send_to_troupe``
and the agent's own ``character_rig`` wrapper. Because Library and Poser submit
under different ``ctx.submit`` keys (``rig:<id>`` vs ``poser-asset-rerig:<id>``),
the client-side busy guard never caught a press from one surface while the
other's rig was still queued or running, so two rig jobs could be minted for
one mesh -- both finalizing into the same ``job_dir``, leaving history with two
"done" rigs for one served result.
"""

from __future__ import annotations

import pytest

from realmspinner import doctor
from realmspinner.service import Conflict
from realmspinner.service import jobs as svc_jobs
from realmspinner.service import rig as svc_rig


def _finished_mesh_job(svc, assets) -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    svc.store.set_status(job_id, "done")
    return job_id


def test_create_rig_refuses_a_second_rig_while_one_for_the_mesh_is_already_in_flight(
    svc, tmp_path, monkeypatch
):
    assets = svc.config.data_dir
    job_id = _finished_mesh_job(svc, assets)
    first = svc_rig.create_rig(svc, job_id, template="humanoid")
    assert svc_rig.rig_in_flight(svc, job_id) == first["id"]

    with pytest.raises(Conflict) as caught:
        svc_rig.create_rig(svc, job_id, template="quadruped")
    assert caught.value.field == "job_id"
    # Refused at the door, before a second row was ever written.
    assert sorted(row["kind"] for row in svc.store.list()) == ["rig", "text"]


def test_a_second_rig_is_accepted_once_the_first_leaves_the_active_set(svc, tmp_path):
    """Not a blanket "one rig ever" rule -- only while a rig for this mesh is
    still queued or running. Once the first finishes (or errors, or is
    cancelled), a fresh rig request is exactly what "Rig this mesh again"
    means and must still work."""
    assets = svc.config.data_dir
    job_id = _finished_mesh_job(svc, assets)
    first = svc_rig.create_rig(svc, job_id, template="humanoid")["id"]
    svc.store.set_status(first, "done")
    assert svc_rig.rig_in_flight(svc, job_id) is None

    second = svc_rig.create_rig(svc, job_id, template="quadruped")
    assert second["id"] != first


def test_the_door_refuses_before_blender_is_even_checked(svc, monkeypatch):
    """The in-flight refusal is cheaper than the Blender probe below it, and
    ordering it first means an already-queued rig never has to reach that
    check to be refused."""
    assets = svc.config.data_dir
    job_id = _finished_mesh_job(svc, assets)
    svc_rig.create_rig(svc, job_id, template="humanoid")

    probed = []
    monkeypatch.setattr(
        doctor,
        "blender_check",
        lambda **_kw: probed.append(True) or doctor.Check(
            name="Blender (rigging)", ok=False, detail="bpy is not installed", fatal=False,
        ),
    )
    with pytest.raises(Conflict):
        svc_rig.create_rig(svc, job_id, template="quadruped")
    assert not probed
