"""``sirens_instruments``: the Kind combo, and the frame it changes on.

Not beside ``sirens_instruments.py`` (``tests/test_sirens_panes_smoke.py`` is
that pane's usual home) because the 2026-09-16 audit's fix pass was scoped to
run only the files under ``tests/sirens/``; this reuses
``test_sirens_mode``'s ``FakeCtx``/``_tab`` the way that file does, across the
directory boundary, since ``tests/conftest.py`` puts ``tests/`` on
``sys.path`` regardless of which subset is collected.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_sirens_mode import FakeCtx, _tab

from warlock.studio.modes.sirens import mode as sirens_mode
from warlock.studio.modes.sirens.ui.panes import instruments as sirens_instruments


@pytest.fixture
def frame():
    """A bare imgui context, built and destroyed around one test.

    ``test_sirens_panes_smoke.py``'s ``frames`` fixture, trimmed to the one
    shape this file needs: no GL, no sound card, just enough of a frame that
    every widget call in ``draw()`` has somewhere to write.
    """
    from imgui_bundle import imgui

    from warlock.studio import theme

    previous = imgui.get_current_context()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)

    def draw(build: Any) -> None:
        imgui.new_frame()
        imgui.set_next_window_size((760.0, 900.0))
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


@pytest.fixture(autouse=True)
def _no_device(monkeypatch):
    from warlock.studio.modes.sirens import audio as sirens_audio

    monkeypatch.setattr(sirens_audio, "available", lambda: False)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)


def test_kind_combo_change_shows_the_sample_panel_on_the_same_frame(frame, monkeypatch):
    """The 2026-09-16 audit: ``selected`` is fetched once at the top of
    ``draw()``, and ``update_instrument`` installs a *new* object through
    ``dataclasses.replace`` rather than mutating in place, so the
    ``if selected.kind == "sample":`` check used to read the pre-update kind
    and show the sample panel one frame late. Reproduced by simulating the
    Kind combo picking "sample" and checking whether ``_sample`` ran on that
    same call to ``draw()``, rather than needing a second one.
    """
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    uid = doc.instruments[0].uid
    assert doc.instrument(uid).kind != "sample"
    sirens_mode.ensure(ctx).instrument = uid

    real_combo = sirens_instruments.controls.combo

    def fake_combo(control_id, current, options, **kwargs):
        if control_id == "##sirens-inst-kind":
            return True, "sample"
        return real_combo(control_id, current, options, **kwargs)

    monkeypatch.setattr(sirens_instruments.controls, "combo", fake_combo)

    calls: list[str] = []

    def fake_sample(ctx: Any, tab: Any, selected: Any) -> None:
        calls.append(selected.kind)

    monkeypatch.setattr(sirens_instruments, "_sample", fake_sample)

    frame(lambda: sirens_instruments.draw(ctx))

    assert doc.instrument(uid).kind == "sample"
    assert calls == ["sample"]
