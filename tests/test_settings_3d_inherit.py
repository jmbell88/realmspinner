"""W1.1: the Mesh stage's "keep the reference's" options name the value they
would actually inherit.

Before this fix ``_platform_options``/``_bg_options`` always offered the
generic "keep the reference's" wording for the unset entry, even when the
selected reference job had a recorded ``platform``/``bg_removal`` value sitting
right there in ``ctx.cache`` -- a user staring at "keep the reference's" had
no way to find out what that was without leaving the Mesh stage.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from warlock.studio.modes.create.engine import mesh as create_mesh
from warlock.studio.modes.create.ui import settings_3d
from warlock.studio.state import DEFAULT_FORM_3D, AppState


class _Ctx:
    """Enough of ``Ctx`` for ``_platform_options``/``_bg_options``: a source
    job cache and the guidance catalog they read field labels from.

    ``svc.config.bench_dir`` points at a directory that is never created --
    every ``_hint`` call now also asks ``bench.findings.best_value`` for the
    same doc, and a real ``Ctx`` always carries a ``svc``, so this stands in
    for one rather than making every test below also build a real service.
    ``findings.load`` treats a missing file as a cached miss, never a raise.
    """

    def __init__(self, jobs: dict, guidance: dict | None = None) -> None:
        self.state = AppState()
        self.state.form_3d = dict(DEFAULT_FORM_3D)
        self._jobs = dict(jobs)
        self.cache = SimpleNamespace(get=lambda job_id: self._jobs.get(job_id))
        self.svc = SimpleNamespace(
            config=SimpleNamespace(
                bench_dir=Path(tempfile.gettempdir()) / "warlock-test-settings-3d-no-bench"
            )
        )
        self.guidance = guidance or {
            "fields": {
                "platform": [
                    {"key": "low", "label": "Low detail"},
                    {"key": "high", "label": "High detail"},
                ]
            },
            "bg_removal": ["birefnet", "flood"],
        }


def test_inherited_mesh_option_names_the_references_value():
    reference = {
        "id": "ref1",
        "status": "done",
        "params": {"platform": "high", "bg_removal": "birefnet"},
    }
    ctx = _Ctx({"ref1": reference})
    ctx.state.source_job = "ref1"

    platform_options = settings_3d._platform_options(ctx)
    bg_options = settings_3d._bg_options(ctx)

    assert platform_options[0] == ("", "From reference: High detail")
    assert bg_options[0] == ("", "From reference: birefnet")


def test_no_selected_source_falls_back_to_the_generic_wording():
    ctx = _Ctx({})
    ctx.state.source_job = None

    assert settings_3d._platform_options(ctx)[0] == ("", "keep the reference's")
    assert settings_3d._bg_options(ctx)[0] == ("", "keep the reference's")


def test_a_reference_with_no_recorded_value_also_falls_back():
    """An older reference job, or one whose params never recorded this key,
    must not be misreported as inheriting a value it doesn't have."""
    reference = {"id": "ref1", "status": "done", "params": {}}
    ctx = _Ctx({"ref1": reference})
    ctx.state.source_job = "ref1"

    assert settings_3d._platform_options(ctx)[0] == ("", "keep the reference's")
    assert settings_3d._bg_options(ctx)[0] == ("", "keep the reference's")


def test_inherited_mesh_option_names_the_selected_meshs_reference_value():
    """The 2026-09-08 audit, finding create-05: with no explicit
    ``source_job`` but a finished mesh selected in the library, ``_submit``
    (via ``_effective_source``) inherits from that mesh's *parent* reference,
    and its own muted line names that reference correctly. The unset-option
    label above it must name the same job -- not fall back to the generic
    wording, which is what happened while ``_source_param`` read
    ``ctx.state.source_job`` on its own instead of resolving through
    ``_effective_source`` the same way.
    """
    reference = {
        "id": "ref1",
        "status": "done",
        "params": {"platform": "high", "bg_removal": "birefnet"},
    }
    mesh = {
        "id": "mesh1",
        "stage": "model",
        "status": "done",
        "files": ["model.glb"],
        "parent_id": "ref1",
    }
    ctx = _Ctx({"ref1": reference})
    ctx.state.source_job = None
    ctx.job = lambda: mesh

    platform_options = settings_3d._platform_options(ctx)
    bg_options = settings_3d._bg_options(ctx)

    assert platform_options[0] == ("", "From reference: High detail")
    assert bg_options[0] == ("", "From reference: birefnet")


# --- the engine disclosure ----------------------------------------------------
#
# The seven trellis_* launch flags, drawn under a collapsed "Engine
# (advanced)" header (item 3 of the plan). Real ``imgui`` layout with no GL,
# ``test_settings_character.py``'s precedent for the same reason: a bug in the
# order these controls draw in, or in how many there are, is invisible to a
# pure test that never calls the drawing function.


@pytest.fixture
def ui(monkeypatch):
    from _ui_context import imgui_context

    with imgui_context(monkeypatch) as imgui:
        yield imgui


def test_the_engine_disclosure_hints_each_axis_it_draws(monkeypatch, ui):
    """Every control the header draws is followed by a findings hint lookup,
    ``settings_3d._hint``'s own contract for every other control on this
    pane -- and the header opens (``controls.collapsing_header`` forced true
    here, the way a click would leave it) rather than staying collapsed, or
    nothing below it would ever run."""
    from warlock.studio import forms, probe

    monkeypatch.setattr(settings_3d.controls, "collapsing_header", lambda *a, **k: True)
    seen: list[str] = []
    monkeypatch.setattr(
        create_mesh,
        "findings_hint",
        lambda ctx, param, value: seen.append(param) or None,
    )
    ctx = _Ctx({})

    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    try:
        with forms.Form("engine-test") as form_ui:
            settings_3d._engine(ctx, dict(DEFAULT_FORM_3D), form_ui)
    finally:
        ui.end()
        ui.end_frame()

    assert seen == [
        "trellis_band",
        "trellis_tex_res",
        "trellis_gss",
        "trellis_gsh",
        "trellis_max_tokens",
        "trellis_decim",
        "trellis_atlas",
    ]


def test_the_engine_disclosure_draws_nothing_while_collapsed(monkeypatch, ui):
    """Collapsed by default (module docstring: a restart nobody asked for),
    so an unopened header must hint nothing and touch no form field."""
    from warlock.studio import forms, probe

    monkeypatch.setattr(settings_3d.controls, "collapsing_header", lambda *a, **k: False)
    calls: list[str] = []
    monkeypatch.setattr(
        create_mesh,
        "findings_hint",
        lambda ctx, param, value: calls.append(param) or None,
    )
    ctx = _Ctx({})
    form = dict(DEFAULT_FORM_3D)

    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    try:
        with forms.Form("engine-test") as form_ui:
            settings_3d._engine(ctx, form, form_ui)
    finally:
        ui.end()
        ui.end_frame()

    assert calls == []
    assert form == DEFAULT_FORM_3D
