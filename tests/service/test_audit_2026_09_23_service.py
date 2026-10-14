"""Regression tests for the service-misc fix-brief of the 2026-09-23 audit.

service-04 and service-05 share one shape: a table shared live with another
thread (``fetch.py``, ``guidance.py``), iterated bare instead of snapshotted
first. (service-02's sidecar-reading doors, ``export_frames`` and
``sheet_preview_png``, left with the character sheets, ``retired.py``.)
pipelines-01 is a different shape (two independent budget constants that
disagree) but lands here too, per the
fixer brief's instruction to put every new regression test in its own file
so no two fixers edit one test module.
"""

from __future__ import annotations

import pytest

from realmspinner.studio import readiness

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
    (``remesh.TRIANGLES_MAX`` is 200,000, also above ``TRIANGLE_BUDGET``):
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
    assert row.repair == readiness.REPAIR_GO_TO_MESH
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
