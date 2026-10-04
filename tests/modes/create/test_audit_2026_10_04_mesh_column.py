"""The 2026-10-04 audit's Mesh column findings (create-26, create-36)."""

from __future__ import annotations

from types import SimpleNamespace

from realmspinner.pipelines import optimize
from realmspinner.studio.modes.create.engine import mesh as create_mesh
from realmspinner.studio.modes.create.ui.panes import settings_3d
from realmspinner.studio.state import DEFAULT_FORM_3D, AppState

_SOURCE = {
    "id": "ref-1",
    "name": "mossy well",
    "status": "done",
    "stage": "reference",
    "files": ["input.png"],
}


def _ctx(*, rigging_available=True, form=None):
    state = AppState(mode="create")
    state.create.stage = "mesh"
    state.source_job = "ref-1"
    if form is not None:
        state.form_3d = form
    toasts: list[str] = []
    return SimpleNamespace(
        state=state,
        cache=SimpleNamespace(get={"ref-1": _SOURCE}.get, jobs=[_SOURCE], active=None),
        job=lambda: None,
        model_rows=[],
        submit=lambda *a, **k: True,
        busy=lambda _k: False,
        rigging_available=rigging_available,
        toast=lambda text, *a, **k: toasts.append(text),
        toasts=toasts,
    )


def _custom_form(count):
    form = dict(DEFAULT_FORM_3D)
    settings_3d._apply_budget_choice(form, "custom")
    form["custom_triangles"] = count
    return form


def test_custom_budget_without_a_valid_count_is_refused_before_the_cutout_check(monkeypatch):
    """The 2026-10-04 audit, finding create-26: Custom left ``custom_triangles``
    at 0, ``problems`` said ready, the cutout check ran and Accept was the first
    place ``optimize.resolve`` refused -- as a toast, panel already closed."""
    opened: list[object] = []
    monkeypatch.setattr(settings_3d.matte_preview, "open_for", lambda *a, **k: opened.append(a))
    for bad in (0, optimize.CUSTOM_MIN - 1, optimize.CUSTOM_MAX + 1):
        ctx = _ctx()
        form = _custom_form(bad)
        found = settings_3d.problems(ctx, _SOURCE, form)
        assert [p.field for p in found] == ["custom_triangles"], bad
        assert f"{optimize.CUSTOM_MIN:,}" in str(found[0])
        # The bar's button and Ctrl+Enter read the live form when handed none.
        ctx.state.form_3d = form
        assert [p.field for p in settings_3d.problems(ctx, _SOURCE)] == ["custom_triangles"]
        settings_3d.promote(ctx, _SOURCE, form)
        assert opened == [], bad
    ok = _custom_form(optimize.CUSTOM_MIN)
    assert settings_3d.problems(_ctx(), _SOURCE, ok) == []
    # A Game-ready rung forces profile to raw, so a stale custom count is moot.
    lowpoly = dict(DEFAULT_FORM_3D)
    settings_3d._apply_budget_choice(lowpoly, "lowpoly:5k")
    lowpoly["custom_triangles"] = 0
    assert settings_3d.problems(_ctx(), _SOURCE, lowpoly) == []


def test_picking_custom_seeds_a_count_the_door_accepts():
    form = dict(DEFAULT_FORM_3D)
    form["custom_triangles"] = 0
    settings_3d._apply_budget_choice(form, "custom")
    assert optimize.CUSTOM_MIN <= form["custom_triangles"] <= optimize.CUSTOM_MAX
    assert settings_3d.problems(_ctx(), _SOURCE, form) == []
    # A count the user already typed survives a re-pick.
    form["custom_triangles"] = 12_345
    settings_3d._apply_budget_choice(form, "custom")
    assert form["custom_triangles"] == 12_345


def test_make_3d_does_not_request_a_rig_the_checkbox_shows_off(monkeypatch):
    """The 2026-10-04 audit, finding create-36: without Blender the Rig box
    draws unchecked, but a persisted ``rig = True`` was still sent."""
    sent: list[dict] = []
    monkeypatch.setattr(
        settings_3d.matte_preview, "open_for", lambda ctx, job_id, kw: sent.append(kw)
    )
    form = dict(DEFAULT_FORM_3D)
    form["rig"] = True
    form["rig_template"] = "humanoid"
    settings_3d.promote(_ctx(rigging_available=False), _SOURCE, form)
    assert sent[0]["rig"] is False
    assert "rig_template" not in sent[0]
    # With Blender present the request is unchanged.
    settings_3d.promote(_ctx(rigging_available=True), _SOURCE, form)
    assert sent[1]["rig"] is True
    assert sent[1]["rig_template"] == "humanoid"


def test_the_engine_kwargs_take_rig_availability_as_a_plain_flag():
    form = dict(DEFAULT_FORM_3D)
    form["rig"] = True
    form["rig_template"] = "humanoid"
    assert create_mesh.promote_kwargs(form, rig_available=False)["rig"] is False
    assert "rig" not in create_mesh.upload_kwargs(form, rig_available=False)
    assert create_mesh.upload_kwargs(form)["rig"] is True
