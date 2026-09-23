"""The 2026-09-23 (second run) audit, muse-01/02/03.

``muse-01``: opening "Make it loop" on a short take and pressing "Queue it"
immediately -- without touching the slider -- sent the door the popup's
untouched default window, 8.0 seconds, even when the take was too short for
that to be legal (``derive_music_job`` refuses unless
``MIN_WINDOW <= span <= parent_duration / 2.0``). The extend branch of
``_derive_field`` clamps its opening value with ``value = min(value, high)``
before handing it to the slider; the loop branch computed the same ``high``
but never clamped ``value`` against it, so the untouched slider carried
8.0 back into ``derive_form`` on the very first frame regardless of how
short ``high`` was.

``muse-03``: ``_play_from`` (``mode.py``) quantises the seek point into a
sample offset into the crossfaded loop body (``cut``), but named the
*unquantised* ``seconds`` as ``loop_anchor`` -- the point ``position()``
treats as where the rotated buffer began. When ``cut`` rounds up to
``len(body)`` the rotation is a no-op (``body[len(body):]`` is empty, so the
buffer is unrotated and starts at ``loop_start``), but the anchor still named
the raw seek point near the *end* of the region -- so the very first
``position()`` read after such a seek reported a point close to the region's
far edge instead of its start.
"""

from __future__ import annotations

from typing import Any

import pytest

from realmspinner.studio.modes.muse import mode as muse_mode
from realmspinner.studio.modes.muse.ui.panes import results as muse_results

from .test_muse_mode import FakeCtx
from .test_muse_player import _Device, _loaded, _settle


@pytest.fixture
def frames():
    from imgui_bundle import imgui

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

    def draw(build: Any, size: tuple[float, float] = (900.0, 700.0)) -> None:
        imgui.new_frame()
        imgui.set_next_window_size(size)
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
    from realmspinner.studio.modes.sirens import audio as sirens_audio

    monkeypatch.setattr(sirens_audio, "available", lambda: False)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)
    monkeypatch.setattr(sirens_audio, "tag", lambda: "")


class _Cache:
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self.jobs = jobs


def _ctx(tmp_path, jobs: list[dict[str, Any]] | None = None) -> FakeCtx:
    from realmspinner.studio.state import AppState

    ctx = FakeCtx(tmp_path)
    ctx.state = AppState()
    ctx.cache = _Cache(jobs or [])
    return ctx


def _take(job_id: str, duration: float, status: str = "done") -> dict[str, Any]:
    return {
        "id": job_id,
        "kind": "music",
        "stage": "music",
        "status": status,
        "prompt": "dark ambient, dungeon, low strings, slow",
        "params": {"duration": duration, "actual_duration": duration},
    }


def test_make_it_loop_with_the_untouched_default_window_is_accepted_for_a_short_parent_take(
    frames, tmp_path, monkeypatch
):
    """A 10 s take: half is 5 s, so the popup's untouched default (8.0 s)
    must be clamped to 5.0 s or less before it ever reaches the door --
    not left at 8.0 s, which ``derive_music_job`` refuses outright.

    Fails against the unfixed code: with a 10 s take, ``derive_form
    ["repaint_end"]`` is still 8.0 after one draw, the same span the door
    (``service/_jobs_music.py:557``) refuses because
    ``8.0 > parent_duration / 2.0 == 5.0``.
    """
    short_take = _take("a", 10.0)
    ctx = _ctx(tmp_path, [short_take])
    muse_mode.open_derive(ctx, "a", "loop")
    assert muse_mode.ensure(ctx).derive_form["repaint_end"] == 8.0

    seen: dict[str, tuple[float, float, float]] = {}
    from realmspinner.studio import widgets as widgets_module

    real_slider = widgets_module.labeled_slider_float

    def spy_slider(title, value, low, high, **kwargs):
        seen[title] = (value, low, high)
        return real_slider(title, value, low, high, **kwargs)

    monkeypatch.setattr(muse_results.widgets, "labeled_slider_float", spy_slider)

    def build() -> None:
        muse_results.draw(ctx)

    frames(build)

    value, low, high = seen["Joint to rewrite"]
    assert high == 5.0
    # The claim: the value handed to the slider -- and so the value the very
    # next "Queue it" press will submit unless the user first drags the
    # slider -- is clamped to the parent's half-length, not left at the
    # popup's untouched 8.0 s default.
    assert value <= high

    # And the state the popup actually queues from is clamped too, not just
    # what the slider happened to be drawn with this frame.
    span = muse_mode.ensure(ctx).derive_form["repaint_end"] - 0.0
    assert span <= 5.0


def test_play_reason_is_a_pure_testable_function():
    """muse-02 (2026-09-23 audit, second run). ``_play_reason`` -- unlike its
    siblings ``_ready_reason``/``_stems_reason``/``_extend_reason``, each with
    its own direct test in this suite -- had none of its own. An evidence
    gap, not a behaviour bug: this passes against both the fixed and the
    unfixed code, and is here so the next change to ``_play_reason`` has
    something guarding it the way its siblings already do.
    """
    assert muse_results._play_reason(False) == "this take has not finished yet"
    # Ready, and the device check is exercised through the module's real
    # ``sirens_audio.unavailable_reason`` rather than stubbed to "" --
    # module-level import, called fresh each time, so this asserts the two
    # reasons compose rather than asserting a stub's own answer.
    from realmspinner.studio.modes.sirens import audio as sirens_audio

    assert muse_results._play_reason(True) == sirens_audio.unavailable_reason()


def test_position_matches_the_audible_buffer_when_a_seek_rounds_up_to_the_full_loop_body(
    tmp_path, monkeypatch
):
    """muse-03 (2026-09-23 audit, second run).

    A 2.0-8.0 s region at 44100 Hz is a 264600-sample loop body. Seeking to
    a point whose quantised phase rounds up to the full 264600 samples --
    here, 7.999999 s, one microsecond shy of the region's own end -- makes
    ``cut == len(body)``, so ``np.concatenate([body[cut:], body[:cut]])`` is
    ``body`` itself: unrotated, starting at ``loop_start`` (2.0 s), not at the
    raw seek point. The unfixed code named ``loop_anchor`` after that raw
    seek point anyway, so the very first ``position()`` read after the seek
    -- before the mixer's own clock has advanced at all -- reported 7.999999
    instead of 2.0: the audible buffer had just started at the region's
    beginning, but ``position()`` placed the playhead six seconds away, at
    its far edge.

    Fails against the unfixed code: ``muse_mode.position(ctx)`` comes back
    ``pytest.approx(7.999999)`` instead of ``pytest.approx(2.0)``.
    """
    ctx = FakeCtx(tmp_path)
    ctx.cache = type("_Cache", (), {"jobs": [{"id": "a"}]})()
    device = _Device()
    monkeypatch.setattr(muse_mode, "sirens_audio", device)

    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 8.0)
    _settle(ctx, one)
    device.busy, device.tag_value = True, "a"

    seek_point = 8.0 - 1e-6  # phase quantises (rounds) up to len(body) exactly
    muse_mode.seek(ctx, seek_point)

    device.pos = 0.0  # nothing of the mixer's own clock has elapsed yet
    assert muse_mode.position(ctx) == pytest.approx(2.0, abs=1e-4)
