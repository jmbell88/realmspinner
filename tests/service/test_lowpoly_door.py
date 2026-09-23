"""The game-ready remesh's door rules: ``service._jobs_create.resolve_lowpoly``.

Sits beside ``resolve_profile`` and shares its shape -- called with the
request's own ``profile``/``lowpoly_triangles`` (never the resolved ones),
recording both keys on every row so a legacy row with neither is
unambiguous. dev/measurements/2026-09-23-default-mesh-budget.md is the
design document; the four cases below are its own summary, each pinned by
name so a future change to the order of the checks cannot silently swap
"defaulted" and "explicit" without a test noticing.
"""

from __future__ import annotations

import pytest

from realmspinner.pipelines import remesh
from realmspinner.service import jobs as svc_jobs
from realmspinner.service.errors import Invalid


@pytest.fixture
def blender_present(monkeypatch):
    from realmspinner import doctor

    class _Ok:
        ok = True

    monkeypatch.setattr(doctor, "blender_check", lambda *a, **k: _Ok())


@pytest.fixture
def blender_absent(monkeypatch):
    from realmspinner import doctor

    class _No:
        ok = False

    monkeypatch.setattr(doctor, "blender_check", lambda *a, **k: _No())


def _create(svc, **kwargs):
    out = svc_jobs.create_job(svc, kind="text", prompt="a barrel", output="model", **kwargs)
    return svc.store.get(out["id"])["params"]


def test_nothing_named_with_blender_defaults_to_the_configured_lowpoly_budget(
    svc, blender_present
):
    """Neither ``profile`` nor ``lowpoly_triangles`` named, Blender available:
    defaults to ``Config.lowpoly_triangles`` (5000) and forces ``profile`` to
    "raw" -- the remesh bakes from the full-detail reconstruction, so a
    gltfpack pass in front of it would simplify twice for nothing."""
    assert svc.config.lowpoly_triangles == 5000
    params = _create(svc)
    assert params["lowpoly_triangles"] == 5000
    assert params["profile"] == "raw"


def test_nothing_named_without_blender_records_no_lowpoly(svc, blender_absent):
    """Without Blender there is nothing to remesh with, so no lowpoly runs --
    ``profile`` is left exactly as ``resolve_profile`` decided it on its own
    (here, downgraded to "raw" too, because the svc fixture also pins
    gltfpack absent -- a defaulted tier downgrades rather than refusing)."""
    params = _create(svc)
    assert params["lowpoly_triangles"] is None
    assert "profile" in params  # resolve_profile always records one


def test_an_explicit_lowpoly_without_blender_refuses(svc, blender_absent):
    with pytest.raises(Invalid) as info:
        _create(svc, lowpoly_triangles=5000)
    assert info.value.field == "lowpoly_triangles"


def test_an_explicit_profile_with_no_lowpoly_named_runs_no_lowpoly(svc, blender_present):
    """The user picked a gltfpack tier (or Raw) by name; a defaulted remesh
    must not silently override that choice the moment Blender happens to be
    on this machine."""
    params = _create(svc, profile="raw")
    assert params["lowpoly_triangles"] is None
    assert params["profile"] == "raw"


def test_an_explicit_lowpoly_forces_profile_to_raw_even_if_named_otherwise(svc, blender_present):
    """``lowpoly_triangles`` named explicitly always wins: it bakes from the
    full-detail reconstruction, so a gltfpack profile named alongside it
    would either be silently discarded or (worse) silently applied first --
    the door removes the ambiguity by forcing "raw"."""
    params = _create(svc, lowpoly_triangles=10_000)
    assert params["lowpoly_triangles"] == 10_000
    assert params["profile"] == "raw"


def test_an_explicit_lowpoly_out_of_range_is_refused_by_name(svc, blender_present):
    with pytest.raises(Invalid) as info:
        _create(svc, lowpoly_triangles=remesh.TRIANGLES_MAX + 1)
    assert info.value.field == "lowpoly_triangles"
    with pytest.raises(Invalid) as info:
        _create(svc, lowpoly_triangles=remesh.TRIANGLES_MIN - 1)
    assert info.value.field == "lowpoly_triangles"


def test_a_legacy_row_shape_is_unambiguous(svc, blender_present):
    """Every row records both keys, ``lowpoly_triangles`` included -- a
    ``None`` on the row means "no lowpoly ran", never simply an absent key a
    reader has to guess the meaning of."""
    params = _create(svc, profile="raw")
    assert "lowpoly_triangles" in params
