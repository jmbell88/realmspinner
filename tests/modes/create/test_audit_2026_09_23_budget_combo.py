"""Regression test for the 2026-09-23 audit, finding create-07.

``settings_3d._budget`` used to call ``_apply_budget_choice(form, picked)``
on every draw with no gesture gate. ``widgets.combo`` returns the value it
was passed, unchanged, on an untouched draw -- so a stored
``lowpoly_triangles`` that matches no ``remesh.TRIANGLE_PROFILES`` rung (a
form saved before a ladder edit, or any other writer) read back as "raw"
through ``_budget_current`` and was zeroed the first frame the Mesh pane was
merely drawn, with no click at all.
"""

from __future__ import annotations

from realmspinner.pipelines import remesh
from realmspinner.studio.modes.create.ui.panes import settings_3d
from realmspinner.studio.panes import remesh_panel, retarget_panel


class _State:
    def __init__(self) -> None:
        self.field_errors: dict[str, str] = {}

    def clear_field_error(self, *_a, **_kw) -> None:
        pass


class _Ctx:
    def __init__(self) -> None:
        self.state = _State()


def _untouched_combo(monkeypatch) -> None:
    """Stand in for ``widgets.labeled_combo`` on a frame where nothing was
    clicked: ``widgets.combo`` (studio/widgets.py ~1030-1040) returns exactly
    the value it was handed when the user made no gesture."""
    monkeypatch.setattr(
        settings_3d.widgets,
        "labeled_combo",
        lambda label, value, options, **kw: value,
    )
    monkeypatch.setattr(settings_3d.widgets, "field_error", lambda *a, **kw: None)
    monkeypatch.setattr(settings_3d, "_hint", lambda *a, **kw: None)


def test_the_budget_combo_does_not_zero_a_stored_lowpoly_budget_merely_by_being_drawn(
    monkeypatch,
):
    """The orchestrator's repro: ``form = {"profile": "raw",
    "lowpoly_triangles": 7500}``. 7500 matches no rung, so
    ``_budget_current`` falls through to "raw", and the unfixed ``_budget``
    called ``_apply_budget_choice(form, "raw")`` unconditionally -- zeroing
    ``lowpoly_triangles`` on a draw with no gesture."""
    monkeypatch.setattr(remesh_panel, "blender_available", lambda _ctx: True)
    monkeypatch.setattr(retarget_panel, "gltfpack_available", lambda _ctx: False)
    _untouched_combo(monkeypatch)

    assert 7500 not in remesh.TRIANGLE_PROFILES.values(), "the repro's premise"
    form = {"profile": "raw", "lowpoly_triangles": 7500, "custom_triangles": 0}
    ctx = _Ctx()

    settings_3d._budget(ctx, form)

    assert form["lowpoly_triangles"] == 7500
    assert form["profile"] == "raw"


def test_the_budget_combo_still_leaves_a_matched_default_form_untouched(monkeypatch):
    """The case the audit already verified as fine, kept fine: a default
    form (profile="raw", lowpoly_triangles=5000, which *does* match the "5k"
    rung) must not be rewritten by an untouched draw either."""
    monkeypatch.setattr(remesh_panel, "blender_available", lambda _ctx: True)
    monkeypatch.setattr(retarget_panel, "gltfpack_available", lambda _ctx: False)
    _untouched_combo(monkeypatch)

    form = {"profile": "raw", "lowpoly_triangles": 5000, "custom_triangles": 0}
    ctx = _Ctx()

    settings_3d._budget(ctx, form)

    assert form["lowpoly_triangles"] == 5000
    assert form["profile"] == "raw"


def test_an_actual_pick_still_applies(monkeypatch):
    """The gate must only block a no-op re-read -- an actual gesture (the
    combo returning a different key than the form's current one) still has
    to write through ``_apply_budget_choice``."""
    monkeypatch.setattr(remesh_panel, "blender_available", lambda _ctx: True)
    monkeypatch.setattr(retarget_panel, "gltfpack_available", lambda _ctx: False)
    monkeypatch.setattr(settings_3d.widgets, "labeled_combo", lambda *a, **kw: "lowpoly:10k")
    monkeypatch.setattr(settings_3d.widgets, "field_error", lambda *a, **kw: None)
    monkeypatch.setattr(settings_3d, "_hint", lambda *a, **kw: None)

    form = {"profile": "raw", "lowpoly_triangles": 5000, "custom_triangles": 0}
    ctx = _Ctx()

    settings_3d._budget(ctx, form)

    assert form["lowpoly_triangles"] == remesh.TRIANGLE_PROFILES["10k"]
    assert form["profile"] == "raw"
