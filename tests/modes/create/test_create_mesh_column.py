"""Step 3 of the Create redesign, the Mesh column's half: settings only (the
press moved to the command bar), an unset seed that reads as unset, and a
missing engine that reaches the footer as a problem with its repair.

The bar's own pins are in ``test_create_brief.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from realmspinner.studio import forms
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
_MISSING = [
    {"row_key": "engine:trellis_runtime", "present": False, "label": "TRELLIS.2 runtime"},
    {"row_key": "engine:trellis_gguf", "present": True, "label": "TRELLIS.2 weights"},
]


@pytest.fixture
def frames():
    from realmspinner.studio import theme

    previous = imgui.get_current_context()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)

    def draw(build) -> None:
        imgui.new_frame()
        imgui.set_next_window_size((400.0, 900.0))
        imgui.begin("smoke")
        try:
            build()
        finally:
            imgui.end()
            imgui.end_frame()
            imgui.render()

    yield draw
    imgui.destroy_context(ctx)
    if previous is not None:
        imgui.set_current_context(previous)


def _ctx(model_rows=None):
    state = AppState(mode="create")
    state.create.stage = "mesh"
    state.source_job = "ref-1"
    return SimpleNamespace(
        state=state,
        cache=SimpleNamespace(get={"ref-1": _SOURCE}.get, jobs=[_SOURCE], active=None),
        job=lambda: None,
        model_rows=model_rows or [],
        submit=lambda *a, **k: True,
        busy=lambda _k: False,
    )


# --- (a) an unset seed reads as unset ---------------------------------------


def _draw_seed(frames, monkeypatch, form):
    """-> (the numbers ``Form.number`` was asked to show, the muted lines, the
    switches drawn) for one frame of the seed block."""
    numbers: list[tuple[str, int]] = []
    muted: list[str] = []
    switches: list[str] = []
    real_number, real_switch = forms.Form.number, forms.Form.switch

    def number(self, key, label, value, *a, **k):
        numbers.append((key, value))
        return real_number(self, key, label, value, *a, **k)

    def switch(self, key, *a, **k):
        switches.append(key)
        return real_switch(self, key, *a, **k)

    monkeypatch.setattr(forms.Form, "number", number)
    monkeypatch.setattr(forms.Form, "switch", switch)
    real_muted = settings_3d.widgets.muted
    monkeypatch.setattr(
        settings_3d.widgets, "muted", lambda text: muted.append(text) or real_muted(text)
    )
    ctx = _ctx()

    def build() -> None:
        with forms.Form("t", errors={}) as form_ui:
            settings_3d._seed(ctx, form, form_ui)

    frames(build)
    return numbers, muted, switches


def test_an_unset_mesh_seed_reads_random_and_shows_no_number(frames, monkeypatch):
    """It used to display ``int(None or 0)`` -- a 0 nobody chose, identical to a
    real seed 0 -- and there was nothing to tell them apart."""
    form = dict(DEFAULT_FORM_3D)
    assert form["mesh_seed"] is None
    numbers, muted, switches = _draw_seed(frames, monkeypatch, form)
    assert numbers == [], "an unset seed is not a number"
    assert any("random" in line for line in muted)
    # The 2D Lock seed pattern: there is nothing to lock until there is a seed.
    assert switches == []
    assert form["mesh_seed"] is None, "drawing must not write a seed"


def test_a_real_seed_of_zero_shows_as_zero_with_its_lock(frames, monkeypatch):
    form = {**DEFAULT_FORM_3D, "mesh_seed": 0}
    numbers, muted, switches = _draw_seed(frames, monkeypatch, form)
    assert numbers == [("mesh_seed", 0)]
    assert not any("random" in line for line in muted)
    assert switches == ["mesh_seed_locked"]
    assert form["mesh_seed"] == 0


# --- (b) a missing engine is a problem, and its repair draws ----------------


def test_engine_problem_names_the_missing_rows_in_the_repairs_own_words():
    problem = create_mesh.engine_problem(_MISSING)
    assert problem is not None
    assert "not downloaded" in str(problem)
    assert "TRELLIS.2 runtime" in str(problem)
    assert "TRELLIS.2 weights" not in str(problem), "a present row is not a problem"


def test_engine_problem_says_nothing_without_a_snapshot():
    """``model_gate``'s doctrine: an empty snapshot must not lock a host that
    has everything installed, and a row it has never heard of is skipped."""
    assert create_mesh.engine_problem([]) is None
    assert create_mesh.engine_problem(None) is None
    assert create_mesh.engine_problem([{"row_key": "base:sdxl_cfg", "present": False}]) is None
    assert create_mesh.engine_problem([{**_MISSING[0], "present": True}]) is None


def test_the_missing_engine_is_a_problem_of_the_press_after_the_reference():
    ctx = _ctx(_MISSING)
    both = settings_3d.problems(ctx, None)
    assert [str(p) for p in both][0] == "Choose a reference first."
    assert "not downloaded" in str(both[-1])
    assert len(settings_3d.problems(_ctx(_MISSING), _SOURCE)) == 1
    assert settings_3d.problems(_ctx(), _SOURCE) == []


def test_ctrl_enter_refuses_a_missing_engine_before_the_cutout(monkeypatch):
    """The keyboard door judges the same list the button does."""
    ctx = _ctx(_MISSING)
    ctx.toast = lambda text, *a, **k: toasts.append(text)
    toasts: list[str] = []
    opened: list[object] = []
    monkeypatch.setattr(settings_3d.matte_preview, "open_for", lambda *a, **k: opened.append(a))
    settings_3d.promote(ctx, _SOURCE, dict(DEFAULT_FORM_3D))
    assert opened == []
    assert "not downloaded" in toasts[0]


def test_the_mesh_footer_draws_open_model_setup_for_a_missing_engine(frames, monkeypatch):
    """The repair used to wait for a problem no validator produced. Drawn for
    real: with the engine absent the footer shows the button, and pressing it
    ticks the engine's own rows and goes to Settings."""
    drawn: list[str] = []
    picked: list[tuple[str, ...]] = []
    real = settings_3d.controls.button

    def spy(label, *a, **k):
        drawn.append(label)
        real(label, *a, **k)
        return label.startswith("Open model setup")

    monkeypatch.setattr(settings_3d.controls, "button", spy)
    monkeypatch.setattr(
        settings_3d.model_gate, "request_install", lambda ctx, keys: picked.append(tuple(keys))
    )
    ctx = _ctx(_MISSING)
    frames(lambda: settings_3d._footer(ctx, dict(DEFAULT_FORM_3D), _SOURCE))
    assert "Open model setup##preflight-models" in drawn
    assert picked == [create_mesh.ENGINE_ROWS]

    drawn.clear()
    ctx = _ctx()
    frames(lambda: settings_3d._footer(ctx, dict(DEFAULT_FORM_3D), _SOURCE))
    assert "Open model setup##preflight-models" not in drawn


def test_the_engine_rows_are_the_two_the_gate_lists_for_create():
    from realmspinner.studio.modes import NEEDS_ROWS

    assert set(create_mesh.ENGINE_ROWS) <= set(NEEDS_ROWS["create"])
