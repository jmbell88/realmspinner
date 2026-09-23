"""Regression tests for the 2026-09-23 (second run) audit, finding
pipelines-01 (dev/TODO.md P65 item 3): ``_q_mesh.py``'s two
``meshreport.build`` call sites recorded every mesh's report against the
module's 150k ``TRIANGLE_BUDGET`` default, even a mesh retargeted or
remeshed to a larger, explicitly accepted budget. Beside ``tests/test_remesh.py``,
which already exercises ``_remesh`` end to end with a faked Blender but never
touches ``meshreport``/``meshaudit`` at all (both fail silently against the
fake bytes that fixture writes as ``model.glb``, so this file drives the two
mesh-report call sites directly instead -- the same "call the mixin method
with a minimal fake ``self``" shape
``tests/service/test_audit_2026_09_23b_service.py``'s own ``_FakeWorker``
already established for ``_publish_model_version``).
"""

from __future__ import annotations

import pytest

from realmspinner import _q_mesh, meshaudit, meshreport
from realmspinner.db import JobStore


class _FakeProgress:
    def update(self, *args, **kwargs) -> None:
        pass


class _FakeWorker:
    """Just enough of ``Worker`` for ``_audit_published``/``_audit_mesh`` to run."""

    def __init__(self, store: JobStore) -> None:
        self.store = store
        self.progress = _FakeProgress()
        self._cancel = None


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.sqlite")
    yield s
    s.close()


def _fake_audit(*args, **kwargs):
    return {"worst": 0.0, "mean": 0.0, "faces": 1, "resolution": 1}


def _capturing_build(captured):
    def fake_build(glb_path, *, target_size_m=None, silhouette=None, triangle_budget=None):
        captured["triangle_budget"] = triangle_budget
        return {
            "status": "ready",
            "reasons": [],
            "triangles": 1,
            "triangle_budget": triangle_budget if triangle_budget is not None else meshreport.TRIANGLE_BUDGET,  # noqa: E501
        }

    return fake_build


# --- the pure helper ---------------------------------------------------------


def test_effective_triangle_budget_prefers_lowpoly_then_optimize_then_the_module_default():
    """``_lowpoly`` runs after ``_optimize`` in ``_q_generate``'s pipeline and
    is the stricter, final remesh -- its own ``"requested"`` wins when both
    ran. Neither present means a raw or optimize-only mesh: ``None`` so
    ``meshreport.build`` falls back to its own module default, not this
    helper silently picking one."""
    assert _q_mesh._effective_triangle_budget({}) is None
    assert _q_mesh._effective_triangle_budget({"optimize": {"requested": 120_000}}) == 120_000
    assert (
        _q_mesh._effective_triangle_budget(
            {"optimize": {"requested": 120_000}, "lowpoly": {"requested": 8_000}}
        )
        == 8_000
    )
    # A record that is not a dict, or one recording no positive "requested",
    # is ignored rather than trusted blind -- the params blob is not this
    # function's own write.
    assert _q_mesh._effective_triangle_budget({"optimize": "not a dict"}) is None
    assert _q_mesh._effective_triangle_budget({"optimize": {"requested": 0}}) is None
    assert _q_mesh._effective_triangle_budget({"optimize": {"requested": None}}) is None


# --- _audit_published (the remesh path, _remesh's own call site) -----------


async def test_a_mesh_remeshed_to_a_custom_triangle_budget_is_not_recorded_over_budget_in_its_mesh_report(  # noqa: E501
    monkeypatch, store, tmp_path
):
    """``_remesh`` calls ``_audit_published`` with ``triangle_budget=triangles``
    -- this mesh's own requested triangle count, not Blender's halved
    ``target_faces`` and not nothing at all. Before this fix the call carried
    no budget, so a mesh remeshed to, say, 180k triangles (comfortably inside
    ``remesh.TRIANGLES_MAX`` = 200k) was recorded against the 150k module
    default -- "over budget" for a budget the user had already been granted.
    """
    monkeypatch.setattr(meshaudit, "hole_fraction", _fake_audit)
    captured: dict[str, object] = {}
    monkeypatch.setattr(meshreport, "build", _capturing_build(captured))

    source_id = store.create("text", "a crate", {"seed": 1})
    glb_path = tmp_path / "model.glb"
    glb_path.write_bytes(b"glb")

    fake = _FakeWorker(store)
    await _q_mesh.MeshPostOps._audit_published(
        fake, source_id, glb_path, None, triangle_budget=180_000
    )

    assert captured["triangle_budget"] == 180_000
    report = store.get(source_id)["params"]["mesh_report"]
    assert report["triangle_budget"] == 180_000


async def test_audit_published_with_no_budget_given_still_falls_back_to_the_module_default(
    monkeypatch, store, tmp_path
):
    """A raw reconstruction that was never retargeted or remeshed has no
    budget of its own -- ``triangle_budget=None`` reaches ``meshreport.build``
    unchanged, and that module's own default (150k) is still what judges it."""
    monkeypatch.setattr(meshaudit, "hole_fraction", _fake_audit)
    captured: dict[str, object] = {}
    monkeypatch.setattr(meshreport, "build", _capturing_build(captured))

    source_id = store.create("text", "a crate", {"seed": 1})
    glb_path = tmp_path / "model.glb"
    glb_path.write_bytes(b"glb")

    fake = _FakeWorker(store)
    await _q_mesh.MeshPostOps._audit_published(fake, source_id, glb_path, None)

    assert captured["triangle_budget"] is None
    report = store.get(source_id)["params"]["mesh_report"]
    assert report["triangle_budget"] == meshreport.TRIANGLE_BUDGET


# --- _audit_mesh (the job's own generate-pipeline report) -------------------


async def test_a_mesh_optimized_to_a_custom_triangle_budget_is_not_recorded_over_budget_in_its_mesh_report(  # noqa: E501
    monkeypatch, store, tmp_path
):
    """``_audit_mesh`` runs at the end of ``_q_generate``'s own pipeline, after
    ``_optimize`` (and ``_lowpoly``, when asked for) have already recorded
    their own requested budgets on the same job's ``params``. Before this fix
    it called ``meshreport.build`` with none of that, so a mesh optimized to
    a custom 220k budget (inside ``optimize.CUSTOM_MAX`` = 250k) was recorded
    against the 150k module default."""
    monkeypatch.setattr(meshaudit, "hole_fraction", _fake_audit)
    captured: dict[str, object] = {}
    monkeypatch.setattr(meshreport, "build", _capturing_build(captured))

    job_id = store.create("text", "a crate", {"seed": 1, "optimize": {"requested": 220_000}})
    glb_path = tmp_path / "model.glb"
    glb_path.write_bytes(b"glb")
    params = store.get(job_id)["params"]

    fake = _FakeWorker(store)
    await _q_mesh.MeshPostOps._audit_mesh(fake, job_id, glb_path, params)

    assert captured["triangle_budget"] == 220_000
    report = store.get(job_id)["params"]["mesh_report"]
    assert report["triangle_budget"] == 220_000
