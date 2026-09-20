"""The Export stage's engine-readiness checklist (:mod:`realmspinner.studio.readiness`).

Pure: every row is a function of the job's already-recorded params, so the
claims here are assertable without a GL context and without ever building a
GLB. The 2026-09-07 Create review, item 3.3.
"""

from __future__ import annotations

from realmspinner.studio import readiness


def _report(**over):
    report = {
        "triangles": 40_000,
        "welded_watertight": True,
        "has_uvs": True,
        "textures": {"base_color": True, "metallic_roughness": True, "normal": False},
        "grounded": True,
        "achieved_size_m": 1.0,
    }
    report.update(over)
    return report


def _job(**over):
    job = {
        "id": "abc123abc123",
        "stage": "model",
        "status": "done",
        "files": [],
        "params": {},
    }
    job.update(over)
    return job


# --- the evidence gate -------------------------------------------------------


def test_a_reference_with_no_mesh_report_gets_no_readiness_checklist():
    """A bare reference at Export is legal and carries no ``mesh_report``, no
    ``degraded`` note and no rig -- there is nothing to have measured, so the
    checklist must not appear at all, empty or otherwise."""
    job = _job(stage="reference", files=["input.png"], params={})
    assert readiness.rows_for(job) == []


def test_a_job_with_no_params_at_all_gets_no_readiness_checklist():
    assert readiness.rows_for(_job(params={})) == []


def test_a_non_dict_job_is_answered_with_nothing_rather_than_a_crash():
    assert readiness.rows_for(None) == []
    assert readiness.rows_for("not a job") == []


# --- triangle budget ----------------------------------------------------------


def test_the_export_stage_lists_an_over_budget_mesh_with_its_remesh_repair():
    job = _job(
        params={"mesh_report": _report(triangles=200_000, achieved_size_m=None), "size_m": 1.0},
    )
    rows = {row.label: row for row in readiness.rows_for(job)}
    assert "Triangles" in rows
    row = rows["Triangles"]
    assert row.state == "attention"
    assert "200,000" in row.detail
    assert "150,000" in row.detail
    assert row.repair == readiness.REPAIR_GO_TO_RIG


def test_an_in_budget_mesh_reports_its_triangles_as_ok_with_no_repair():
    job = _job(params={"mesh_report": _report(triangles=40_000), "size_m": 1.0})
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Triangles"]
    assert row.state == "ok"
    assert row.repair is None


# --- watertight / UVs / textures / pivot --------------------------------------


def test_a_leaky_mesh_is_flagged_not_watertight_with_no_repair():
    job = _job(
        params={
            "mesh_report": _report(
                welded_watertight=False, welded_boundary_edges=6, welded_components=2
            ),
            "size_m": 1.0,
        }
    )
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Watertight"]
    assert row.state == "attention"
    assert "6" in row.detail and "2" in row.detail
    assert row.repair is None


def test_missing_uvs_and_missing_textures_are_each_their_own_row():
    job = _job(
        params={
            "mesh_report": _report(
                has_uvs=False,
                textures={"base_color": False, "metallic_roughness": False, "normal": False},
            ),
            "size_m": 1.0,
        }
    )
    rows = {row.label: row for row in readiness.rows_for(job)}
    assert rows["UV coordinates"].state == "attention"
    assert rows["Textures"].state == "attention"
    assert "base colour" in rows["Textures"].detail
    assert "metallic/roughness" in rows["Textures"].detail
    assert rows["UV coordinates"].repair is None
    assert rows["Textures"].repair is None


def test_an_ungrounded_pivot_is_flagged_with_no_repair():
    job = _job(params={"mesh_report": _report(grounded=False), "size_m": 1.0})
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Pivot on floor"]
    assert row.state == "attention"
    assert row.repair is None


# --- size ----------------------------------------------------------------------


def test_no_target_size_states_the_guidance_suggestion_and_offers_no_press():
    """The hint is worth saying; applying it to a finished mesh is not a press.

    ``size_m`` is a recorded input in ``vectors.VECTOR_PARAMS`` and the scale
    was composed into the geometry by ``normalize_glb`` when the mesh was
    made, so a button here could only re-key the job's stored evidence and
    claim a size the mesh does not have. The first draft of this row wired one
    to ``jobs.update_job``, which accepts name/tags/favorite and silently
    ignores everything else -- a repair that did nothing at all.
    """
    # size_hint takes the *last* noun it recognises (the head of an English
    # noun phrase), so the prompt names only the one word it should match.
    job = _job(
        prompt="a rusty sword",
        params={"mesh_report": _report(achieved_size_m=0.4)},
    )
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Size"]
    assert row.state == "attention"
    assert "no target size" in row.detail
    assert "1 m" in row.detail  # guidance.SIZE_HINTS_M["sword"], stated
    assert row.repair is None


def test_no_target_size_and_no_recognised_noun_has_no_repair():
    job = _job(prompt="a strange glowing thing", params={"mesh_report": _report()})
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Size"]
    assert row.state == "attention"
    assert row.repair is None


def test_a_size_close_to_the_target_is_ok():
    job = _job(params={"mesh_report": _report(achieved_size_m=1.0), "size_m": 1.0})
    rows = {row.label: row for row in readiness.rows_for(job)}
    assert rows["Size"].state == "ok"


def test_a_size_far_from_the_target_is_flagged_with_no_repair():
    job = _job(params={"mesh_report": _report(achieved_size_m=1.5), "size_m": 1.0})
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Size"]
    assert row.state == "attention"
    assert "1.500" in row.detail and "1.000" in row.detail
    assert row.repair is None


# --- degraded processing --------------------------------------------------------


def test_a_degraded_normalize_offers_the_rebuild_door():
    job = _job(
        params={
            "degraded": {"normalize": "trimesh could not parse the mesh"},
        }
    )
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Processing"]
    assert row.state == "attention"
    assert "normalize" in row.detail
    assert row.repair == readiness.REPAIR_GO_TO_RIG


def test_a_degraded_report_step_has_no_repair():
    """Nothing re-measures a mesh on its own, so a failed *report* step (as
    opposed to a failed *normalize*) is stated with no button."""
    job = _job(params={"degraded": {"report": "the mesh could not be measured"}})
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Processing"]
    assert row.state == "attention"
    assert row.repair is None


def test_an_empty_degraded_dict_produces_no_processing_row():
    # A stale ``{}`` left after a clean rerun must not read as a live failure.
    job = _job(params={"degraded": {}})
    rows = {row.label: row for row in readiness.rows_for(job)}
    assert "Processing" not in rows


# --- stale rig -------------------------------------------------------------------


def test_a_retargeted_mesh_with_an_unrebuilt_rig_is_flagged_stale():
    """``optimize_job`` drops ``mesh_report`` when it rewrites ``model.glb`` and
    nothing recomputes it afterwards -- a rig beside a report-less, retargeted
    mesh is a rig nothing has re-verified since."""
    job = _job(
        files=["rig.glb", "model.glb"],
        params={"optimize": {"faces": 40_000}},
    )
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Rig"]
    assert row.state == "attention"
    assert row.repair == readiness.REPAIR_GO_TO_RIG


def test_a_rig_beside_a_freshly_measured_mesh_is_not_flagged_stale():
    job = _job(
        files=["rig.glb", "model.glb"],
        params={"optimize": {"faces": 40_000}, "mesh_report": _report()},
    )
    rows = {row.label: row for row in readiness.rows_for(job)}
    assert "Rig" not in rows


def test_no_rig_at_all_is_never_flagged_stale():
    job = _job(files=["model.glb"], params={"optimize": {"faces": 40_000}})
    rows = {row.label: row for row in readiness.rows_for(job)}
    assert "Rig" not in rows


def test_a_rig_that_was_never_retargeted_is_not_flagged_stale():
    # rig.glb with no mesh_report and no recorded optimize -- e.g. mesh_report
    # simply was never built for this row -- is not evidence of a retarget.
    job = _job(files=["rig.glb", "model.glb"], params={})
    rows = {row.label: row for row in readiness.rows_for(job)}
    assert "Rig" not in rows


# --- ordering ------------------------------------------------------------------


def test_the_export_stage_draws_readiness_first():
    """:mod:`realmspinner.studio.panes.inspector`'s own stage-section table."""
    from realmspinner.studio.panes import inspector

    assert inspector._STAGE_SECTIONS["export"][0] == "_readiness"


def test_the_export_stage_body_knows_how_to_draw_readiness():
    import inspect

    from realmspinner.studio.panes import inspector

    source = inspect.getsource(inspector._stage_body)
    assert '"_readiness": lambda: _readiness(ctx, job)' in source


# --- the size row -----------------------------------------------------------


def test_size_row_survives_a_non_numeric_size_m():
    """The 2026-09-18 audit, finding create-02: ``params["size_m"]`` is a
    recorded input (``vectors.VECTOR_PARAMS``), not a validated one, and
    ``float(size_m)`` used to run unguarded -- so a hand-edited or
    otherwise-mangled row raised a ``ValueError`` straight out of a function
    ``panes/inspector.py`` calls on the frame thread, on every Export-stage
    frame.
    """
    row = readiness._size_row(_report(), {"size_m": "not-a-number"}, None)
    assert row.state == "attention"

    row = readiness._size_row(_report(), {"size_m": object()}, None)
    assert row.state == "attention"

    row = readiness._size_row(_report(), {"size_m": None}, None)
    assert row.state == "attention"  # the "no target size" branch, unchanged


def test_size_row_still_reads_a_numeric_size_m():
    row = readiness._size_row(_report(achieved_size_m=1.0), {"size_m": 1.0}, None)
    assert row.state == "ok"
    row = readiness._size_row(_report(achieved_size_m=1.0), {"size_m": "1.0"}, None)
    assert row.state == "ok"
