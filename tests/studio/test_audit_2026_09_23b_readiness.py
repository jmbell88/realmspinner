"""The 2026-09-23 (second run) audit, finding pipelines-01.

:func:`realmspinner.studio.readiness._effective_triangle_budget` read a
remesh record's ``target_faces`` as though it were the triangle budget the
mesh was built to. ``target_faces`` is Blender's own quad count --
``pipelines.remesh.target_faces`` halves the triangle budget before handing
it to quadriflow -- so a remesh that landed on its 5,000-triangle default
(reported as ~4,996 actual triangles) was flagged "above the 2,500 budget"
forever, with the row's own repair button pointing back at the panel that
had already built exactly what it asked for.
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


def test_a_panel_remesh_that_lands_on_its_budget_is_not_flagged_over_budget():
    # ``_q_mesh.py`` records both fields on a modern (target_triangles) remesh
    # request: ``target_faces`` (2,500 -- Blender's quad count, triangles//2)
    # and ``target_triangles`` (5,000 -- the actual budget the panel offered
    # and the user picked). The mesh Blender actually produced (4,996) is
    # under the 5,000 budget but over the 2,500 quad count the old code
    # mistook for it.
    job = _job(
        params={
            "mesh_report": _report(triangles=4_996, achieved_size_m=None),
            "size_m": 1.0,
            "remesh": {"target_faces": 2_500, "target_triangles": 5_000},
        },
    )
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Triangles"]
    assert row.state == "ok", row.detail
    assert row.repair is None
    assert "5,000" in row.detail
    assert "2,500" not in row.detail


def test_a_legacy_remesh_record_with_only_target_faces_is_doubled_into_triangles():
    # A row queued before ``target_triangles`` existed carries only
    # ``target_faces`` -- itself a quad count, per ``_q_mesh.py``'s own
    # comment ("a legacy row's number is already Blender's own quad count").
    # Comparing triangles against it directly (the pre-fix behaviour) halves
    # the real budget; doubling it back is the honest reading.
    job = _job(
        params={
            "mesh_report": _report(triangles=4_996, achieved_size_m=None),
            "size_m": 1.0,
            "remesh": {"target_faces": 2_500},
        },
    )
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Triangles"]
    assert row.state == "ok", row.detail
    assert "5,000" in row.detail


def test_a_remesh_that_genuinely_exceeds_its_recorded_budget_is_still_flagged():
    job = _job(
        params={
            "mesh_report": _report(triangles=9_000, achieved_size_m=None),
            "size_m": 1.0,
            "remesh": {"target_faces": 2_500, "target_triangles": 5_000},
        },
    )
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Triangles"]
    assert row.state == "attention"
    assert row.repair == readiness.REPAIR_GO_TO_RIG
    assert "5,000" in row.detail
