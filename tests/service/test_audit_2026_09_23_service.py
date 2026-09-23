"""Regression tests for the service-misc fix-brief of the 2026-09-23 audit.

service-02, service-04 and service-05 share one shape: a table that is either
read straight off a hand-edited sidecar (``characters.py``) or shared live
with another thread (``fetch.py``, ``guidance.py``), indexed or iterated bare
instead of validated or snapshotted first. pipelines-01 is a different shape
(two independent budget constants that disagree) but lands here too, per the
fixer brief's instruction to put every new regression test in its own file
so no two fixers edit one test module.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from realmspinner.kernels.rig import store
from realmspinner.service import characters as svc_characters
from realmspinner.service.errors import Invalid
from realmspinner.studio import readiness

# --- service-02: export_frames / sheet_preview_png must refuse, not KeyError ---


def _mesh_with_sheet(svc, *, frame_size=16, columns=2, rows=1):
    """A finished character mesh plus a hand-built character sheet -- the
    same minimal shape ``tests/test_audit_2026_09_15_troupe.py``'s own
    ``_mesh_with_sheet`` uses, with a real ``"troupe"`` block (movements,
    runs, cells) a hand-edit can then corrupt.
    """
    from realmspinner.service import jobs as svc_jobs

    mesh_id = svc_jobs.create_job(svc, kind="text", prompt="a knight")["id"]
    svc.store.set_status(mesh_id, "done")
    job_dir = svc.job_dir(mesh_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"glb")

    sheet_id = store.new_id()
    atlas = Image.new("RGBA", (frame_size * columns, frame_size * rows), (0, 0, 0, 255))
    png_path = store.sheet_png_path(job_dir, sheet_id)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(png_path, format="PNG")

    movements = [
        {
            "key": "walk", "label": "Walk", "frames": 2, "loop": False,
            "duration_ms": 100,
            "directions": [{"key": "front", "label": "Front", "yaw": 0.0}],
        }
    ]
    runs = [{"movement": "walk", "direction": "front", "yaw": 0.0, "start": 0, "end": 1}]
    cells = [
        {"index": i, "x": (i % columns) * frame_size, "y": 0, "w": frame_size, "h": frame_size}
        for i in range(columns * rows)
    ]
    sidecar = {
        "version": 1, "id": sheet_id, "name": "service-02 test", "source_job": mesh_id,
        "created": 0.0, "image": png_path.name, "frame_size": frame_size,
        "columns": columns, "rows": rows, "width": frame_size * columns,
        "height": frame_size * rows, "yaws": [0.0], "poses": [], "cells": cells,
        "troupe": {"version": 3, "columns": columns, "movements": movements, "runs": runs,
                   "cell_count": len(cells)},
    }
    store.sheet_path(job_dir, sheet_id).write_text(json.dumps(sidecar), encoding="utf-8")
    return mesh_id, sheet_id


def test_export_frames_refuses_a_sidecar_whose_movement_is_missing_its_key(svc, tmp_path):
    """service-02 (the 2026-09-23 audit): ``export_frames`` built its
    ``movements`` table with ``{m["key"]: m for m in ...}`` -- a bare
    subscript. A hand-edited sidecar whose movement entry lost its ``"key"``
    field used to raise ``KeyError`` right there, before the door's own
    "refused, not crashed" cell-index guard ever ran. Must raise
    ``service.errors.Invalid`` with ``field="sheet_id"`` instead.
    """
    mesh_id, sheet_id = _mesh_with_sheet(svc)
    job_dir = svc.job_dir(mesh_id)
    sidecar_path = store.sheet_path(job_dir, sheet_id)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    del sidecar["troupe"]["movements"][0]["key"]
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(Invalid) as excinfo:
        svc_characters.export_frames(svc, mesh_id, sheet_id, dest_dir=tmp_path)
    assert excinfo.value.field == "sheet_id"


def test_sheet_preview_png_refuses_a_sidecar_whose_cells_are_missing_index_rather_than_keyerror(
    svc,
):
    """service-02 (the 2026-09-23 audit): ``sheet_preview_png`` built its
    ``cell_by_index`` table with ``{int(c["index"]): c for c in ...}`` -- a
    bare subscript, exercised before any of this door's own guards. A
    hand-edited sidecar whose cell entry lost its ``"index"`` field used to
    raise ``KeyError`` instead of a service refusal.
    """
    mesh_id, sheet_id = _mesh_with_sheet(svc)
    job_dir = svc.job_dir(mesh_id)
    sidecar_path = store.sheet_path(job_dir, sheet_id)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    del sidecar["cells"][0]["index"]
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(Invalid) as excinfo:
        svc_characters.sheet_preview_png(
            svc, mesh_id, sheet_id, max_side=64, max_bytes=200_000,
        )
    assert excinfo.value.field == "sheet_id"


# --- service-04 / service-05: STYLE_LORAS must be snapshotted, not iterated live ---


def test_fetch_entries_does_not_iterate_the_live_style_loras_table(monkeypatch):
    """service-04 (the 2026-09-23 audit): ``fetch.entries`` iterated the live
    ``models.STYLE_LORAS`` mapping (``kind.table.items()``, where
    ``fetch.KINDS`` binds ``Kind("lora", models.STYLE_LORAS, ...)`` to that
    very object) while a background import thread mutates it -- exactly the
    "dict changed size during iteration" service-04 from the 2026-09-15
    audit fixed for ``doctor``'s own copy of this loop, but not for this
    door. Reproduced with a real dict and a genuine mid-loop mutation (not a
    simulated one): a ``label`` property that adds a row to the *live* table
    the first time it is read, standing in for a concurrent LoRA import
    landing between two iterations of the loop body.
    """
    from realmspinner import fetch as svc_fetch
    from realmspinner import models as svc_models

    class FakeSpec:
        mutated = False

        def __init__(self, table):
            self._table = table

        @property
        def label(self):
            # Fires once, the moment the loop body first reads a spec's
            # label -- mutating the *same* dict object the loop is
            # iterating, the way a concurrent import would.
            if not FakeSpec.mutated:
                FakeSpec.mutated = True
                self._table[f"concurrent-{len(self._table)}"] = FakeSpec(self._table)
            return "x"

    live: dict[str, FakeSpec] = {}
    live["a"] = FakeSpec(live)
    live["b"] = FakeSpec(live)
    monkeypatch.setattr(svc_models, "STYLE_LORAS", live)
    lora_kind = svc_fetch.Kind("lora", live, "style LoRA: ", "Style LoRAs")
    new_kinds = tuple(lora_kind if k.key == "lora" else k for k in svc_fetch.KINDS)
    monkeypatch.setattr(svc_fetch, "KINDS", new_kinds)

    # Before the fix: RuntimeError("dictionary changed size during
    # iteration") from the live ``kind.table.items()`` loop. After: the
    # snapshot absorbs the mutation and this just returns normally.
    list(svc_fetch.entries())


def test_guidance_lookup_does_not_iterate_the_live_style_loras_table_on_refusal(monkeypatch):
    """service-05 (the 2026-09-23 audit): ``guidance._lookup``'s unknown-value
    refusal path does ``sorted(table)`` straight over the live
    ``models.STYLE_LORAS`` mapping to build its "expected one of" list, so
    the same concurrent-import mutation ``fetch.entries`` was vulnerable to
    (service-04) can also crash a plain guidance lookup that merely mistypes
    a style name, raising ``RuntimeError: dict changed size during
    iteration`` instead of the intended ``GuidanceError``.
    """
    from realmspinner import guidance as svc_guidance
    from realmspinner import models as svc_models
    from realmspinner.guidance import GuidanceError

    class MutatingLoras(dict):
        """Mutates itself the moment it is iterated by ``sorted()`` --
        standing in for a second thread's LoRA import landing mid-call."""

        def __iter__(self):
            it = list(super().keys())
            for i, key in enumerate(it):
                if i == 0:
                    super().__setitem__(f"concurrent-{len(self)}", {"path": "x"})
                yield key

    live = MutatingLoras({"anime": object(), "noir": object()})
    monkeypatch.setattr(svc_models, "STYLE_LORAS", live)
    monkeypatch.setitem(svc_guidance._TABLES, "style_lora", live)

    # Before the fix: RuntimeError("dict changed size during iteration").
    # After: the intended refusal, naming the field.
    with pytest.raises(GuidanceError) as excinfo:
        svc_guidance._lookup("style_lora", "not-a-real-style")
    assert excinfo.value.field == "style_lora"


# --- pipelines-01: a mesh built to its own custom budget must not be flagged ---


def _report(**over):
    report = {
        "triangles": 40_000,
        "welded_watertight": True,
        "has_uvs": True,
        "textures": {"base_color": True, "metallic_roughness": True},
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


def test_a_mesh_optimized_to_its_own_custom_budget_is_not_flagged_over_budget():
    """pipelines-01 (the 2026-09-23 audit): ``meshreport.TRIANGLE_BUDGET``
    (150,000) is below the custom budgets the retarget panel accepts
    (``pipelines.optimize.CUSTOM_MAX`` is 250,000), so a mesh a user
    explicitly retargeted to 200,000 triangles -- and which
    ``optimize.resolve`` accepted with no complaint -- used to be shown as
    "attention"/over budget forever, with a repair button pointing back at
    the very retarget panel that had already done exactly what was asked.
    The accepted budget is recorded on the job as
    ``params["optimize"]["requested"]`` (``pipelines.optimize.run``'s own
    return value); the readiness row must read it instead of the fixed
    module default.
    """
    job = _job(
        params={
            "mesh_report": _report(triangles=200_000),
            "optimize": {"requested": 200_000, "achieved": 200_000, "source_triangles": 290_000},
            "size_m": 1.0,
        },
    )
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Triangles"]
    assert row.state == "ok", row.detail
    assert row.repair is None
    assert "200,000" in row.detail


def test_a_mesh_remeshed_to_its_own_custom_face_target_is_not_flagged_over_budget():
    """Same finding, the remesh panel's own field name
    (``remesh.FACES_MAX`` is 200,000, also above ``TRIANGLE_BUDGET``):
    ``params["remesh"]["target_faces"]`` is the accepted budget for a
    remeshed job, recorded by ``_q_mesh.py``'s ``_remesh`` stage.
    """
    job = _job(
        params={
            "mesh_report": _report(triangles=180_000),
            "remesh": {"target_faces": 200_000, "faces": 180_000, "method": "quadric"},
            "size_m": 1.0,
        },
    )
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Triangles"]
    assert row.state == "ok", row.detail
    assert row.repair is None


def test_a_mesh_still_over_its_own_accepted_custom_budget_is_flagged():
    """The fix must not simply widen the ceiling for everyone -- a mesh that
    overshoots the budget it was itself retargeted to is still "attention".
    """
    job = _job(
        params={
            "mesh_report": _report(triangles=260_000),
            "optimize": {"requested": 200_000, "achieved": 260_000, "source_triangles": 290_000},
            "size_m": 1.0,
        },
    )
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Triangles"]
    assert row.state == "attention"
    assert row.repair == readiness.REPAIR_GO_TO_RIG
    assert "200,000" in row.detail


def test_a_raw_unretargeted_mesh_still_uses_the_module_default_budget():
    """No ``optimize``/``remesh`` record at all (the raw reconstruction, or a
    report predating this fix) must fall back to
    ``meshreport.TRIANGLE_BUDGET`` exactly as before -- this is the existing
    behaviour ``tests/studio/test_export_readiness.py`` already covers; kept
    here too so this module's own claim about the fallback is self-contained.
    """
    job = _job(params={"mesh_report": _report(triangles=200_000), "size_m": 1.0})
    rows = {row.label: row for row in readiness.rows_for(job)}
    row = rows["Triangles"]
    assert row.state == "attention"
    assert "150,000" in row.detail
