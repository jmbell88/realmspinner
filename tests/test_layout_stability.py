"""A permanent regression gate for the scrollbar-feedback oscillation loop.

``widgets.stable_width``'s docstring has the incident in full: ``layout.pane``
opens every pane as a scrolling child with no ``no_scrollbar`` flag, and Dear
ImGui decides a child's scrollbar from the *previous* frame's content size --
so a widget whose drawn height grows with the pane's width feeds its own
height back into next frame's scrollbar decision and the pane oscillates
forever, with nothing in the log. Three sites were found and fixed only
because a user happened to say "the right side flickers"
(``panes/inspector.py``, CHANGELOG.md:781); a fourth was reported the same
way, against the Rig stage's deformation-review thumbnail (already guarded)
-- and until this file, nothing in the suite could have caught any of them:
no test drew two consecutive frames and compared their geometry.

``layout.pane``'s oscillation detector (module constant ``layout.
TRACE_ENABLED``, gated on ``WARLOCK_LAYOUT_TRACE=1`` in the running app) is
what makes this assertable. The sweep below drives Create's real inspector
through a real ``layout.pane`` across the four variables that decide whether a
site lands on the scroll threshold -- pane width, UI scale, pane height, and
which Create stage is open -- for a rigged mesh, which is the asset the report
named.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from imgui_bundle import imgui
from PIL import Image

from warlock.service import matte as svc_matte
from warlock.studio import layout, theme, tokens
from warlock.studio import textures as textures_mod
from warlock.studio.app_ctx import Ctx
from warlock.studio.modes.create.ui import stages as create_stages
from warlock.studio.modes.create.ui.panes import settings_3d
from warlock.studio.panes import inspector
from warlock.studio.state import AppState

# --- harness ------------------------------------------------------------
#
# Shape copied from ``tests/test_create_brief.py``'s ``frames`` fixture: a
# bare imgui context (no GL, no renderer -- ``renderer_has_textures`` is what
# lets a frame finish without a backend claiming the font atlas), rebuilt and
# torn down around this file because at most one imgui context may exist at a
# time.


@pytest.fixture
def frames():
    previous = imgui.get_current_context()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)

    def draw(build: Any, size: tuple[float, float] = (1200.0, 900.0)) -> None:
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


class _FakeTexture:
    """Enough of a moderngl texture for ``widgets.texture_ref`` and the
    thumbnail cache's own accounting. ``.size`` is what ``inspector.
    _deform_qa`` sizes its image from; ``.glo`` is the GL name ``texture_ref``
    hands to ``imgui.ImTextureRef``, which -- unlike ``tests/test_thumbnail_
    cache.py``'s otherwise-identical fake -- this test actually calls, so it
    has to be a small int rather than ``id(self)``: nanobind's binding refuses
    a raw object address as out of range for ``tex_id``.
    """

    _next_glo = 1

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.filter = None
        self.repeat_x = True
        self.repeat_y = True
        self.glo = _FakeTexture._next_glo
        _FakeTexture._next_glo += 1

    def release(self) -> None:
        pass


class _FakeGL:
    NEAREST = "nearest"
    LINEAR = "linear"

    def texture(self, size: tuple[int, int], components: int, data: bytes) -> _FakeTexture:
        return _FakeTexture(size)


class _FakeTasks:
    """Records a submit rather than running it. Nothing in this sweep needs a
    job to actually queue, and a real ``TaskRunner`` wants a worker thread
    this test has no business starting."""

    def is_busy(self, key: str) -> bool:
        return False

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        return True

    def progress(self, key: str) -> Any:
        return None


def _rigged_mesh_job(svc: Any) -> dict[str, Any]:
    """A finished, rigged mesh with a real deformation-QA sheet on disk --
    the exact asset the report names: "selecting a RIGGED MESH in the
    library list, the inspector above it flickers". Real PNGs rather than
    stub bytes for the two the inspector actually decodes, because
    ``ThumbnailCache`` reads them through PIL and a fake would report a
    ``.size`` the code under test never computed.
    """
    job_id = svc.store.create("image", "a rigged chest", {}, stage="model", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (512, 512), (180, 70, 70, 255)).save(job_dir / "input.png")
    Image.new("RGBA", (256, 256), (70, 140, 200, 255)).save(job_dir / "rig_qa.png")
    (job_dir / "rig_qa.json").write_text("{}", encoding="utf-8")
    for name in ("model.glb", "source.glb", "rig.glb"):
        (job_dir / name).write_bytes(b"stub")
    job = svc.store.get(job_id)
    job["files"] = [
        "model.glb",
        "source.glb",
        "rig.glb",
        "input.png",
        "rig_qa.png",
        "rig_qa.json",
    ]
    return job


def _ctx(svc: Any, job: dict[str, Any]) -> Ctx:
    state = AppState(mode="create", selected=job["id"])
    state.create.stage = "rig"
    cache = SimpleNamespace(
        get=lambda job_id: job if job_id == job["id"] else None, jobs=[job]
    )
    # Deliberately *not* pre-graded: an ungraded mesh is what opens
    # ``_verdict``'s "Was this any good?" section by default, and that section
    # -- specifically ``widgets.tag_toggles``'s grid -- was the second real
    # oscillation site this sweep found (widgets.py:2359's comment has the
    # incident). Leaving the job ungraded is what exercises it: the common
    # case, and the one a person is looking at right after a rig lands.
    return Ctx(
        svc=svc,
        runtime=SimpleNamespace(config=svc.config),
        state=state,
        cache=cache,
        tasks=_FakeTasks(),
        settings=SimpleNamespace(get=lambda *_a, **_kw: None, set=lambda *_a, **_kw: None),
        textures=textures_mod.ThumbnailCache(_FakeGL()),
        rig_templates=[],
        rig_default="",
        rigging_available=False,
        sheet_options={},
    )


# --- the sweep ------------------------------------------------------------
#
# Widths bracket the 220-480 dp saved-panel range (``layout.PANEL_MIN``/
# ``PANEL_MAX``); the dense cluster at 220-300 is deliberate, not padding --
# that is where a pane's content width (after ``layout.PANE_PADDING`` and a
# maybe-scrollbar are taken off) crosses ``inspector.THUMB_SIZE * 2`` (192 dp),
# which is the exact threshold ``inspector._deform_qa``'s own comment names.
WIDTHS_DESIGN = (layout.PANEL_MIN, 226.0, 232.0, 240.0, 260.0, 300.0, layout.PANEL_MAX)
SCALES = (1.0, 1.25, 1.6)

#: ``theme.apply`` sets ``style.scrollbar_size = sp(10)`` -- the design-space
#: width a pane's content region loses the instant its scrollbar appears,
#: whatever the scale. Named here because :func:`_settled_height` below has to
#: reproduce exactly that shrink to measure the *other* side of the same
#: threshold ``layout.pane`` itself decides between one frame later.
SCROLLBAR_DESIGN = 10.0


def test_the_create_inspector_settles_within_four_frames(frames, svc, monkeypatch):
    """Frames 3 and 4 must draw identical geometry, for every (stage, width,
    scale, height) combination the sweep covers -- and the detector built for
    exactly this must never fire.

    **Heights are derived, not guessed.** A fixed handful of pane heights
    almost never lands on the few pixels where a real oscillation lives: this
    pane's content is several hundred px tall and only the last handful of
    those px move with the scrollbar at all (``_deform_qa``'s image, plus
    whatever text nearby wraps a line differently at the same width). An
    early version of this sweep used two arbitrary heights and stayed green
    even with the guarded call reverted to a raw ``get_content_region_avail``
    -- not because nothing was wrong, but because neither height ever put the
    pane within reach of its own scrollbar threshold. So for every (stage,
    width, scale) this measures the two heights ``layout.pane`` itself is
    actually choosing between one frame apart -- this content's height with a
    generous avail (scrollbar down) and with exactly ``style.scrollbar_size``
    taken off (scrollbar up, :data:`SCROLLBAR_DESIGN`) -- and sweeps pane
    heights that bracket *that* pair. A site that has gone back to reading the
    live avail draws a different height in each state and this lands the pane
    exactly where the two disagree; a fixed pair of guesses does not.

    Failures are collected rather than raised on the first one, so a run
    reports every combination that broke rather than only the first
    alphabetic stage -- which matters here because the defect is
    threshold-dependent and the whole point of the sweep is to find where.
    """
    job = _rigged_mesh_job(svc)
    ctx = _ctx(svc, job)
    monkeypatch.setattr(layout, "TRACE_ENABLED", True)

    def build_pane(pane_id: str, width: float, height: float) -> None:
        with layout.pane(pane_id, (width, height), layout.PaneRole.INSPECTOR) as visible:
            if visible:
                inspector.draw(ctx)

    def settled_height(pane_id: str, width: float) -> float | None:
        """This pane's content height against a pane so tall its own
        scrollbar never enters the decision -- i.e. the same "how tall will
        this draw" question a real frame's scrollbar choice is one frame
        behind on. Two frames because the first is imgui establishing this
        (brand new) child's content size at all, exactly as every call site
        here needs two settling frames before frame 3 and 4 can be compared.
        """
        probe_height = 6000.0
        layout.reset_trace()
        for _ in range(2):
            frames(
                lambda: build_pane(pane_id, width, probe_height),
                size=(width + 80.0, probe_height + 260.0),
            )
        samples = layout.trace_samples(pane_id)
        return samples[-1][0] if samples else None

    failures: list[str] = []
    for stage in create_stages.STAGES:
        ctx.state.create.stage = stage
        for scale in SCALES:
            monkeypatch.setattr(tokens, "SCALE", scale)
            # Idempotent and reads tokens.SCALE at call time (see
            # panes/app_settings.py:554) -- the same re-application a live
            # UI-scale change triggers (main.py), and without it every style
            # metric ``stable_content_width`` reads (the scrollbar width, the
            # pane padding) would stay pinned at whatever scale the context
            # was created under.
            theme.apply(imgui)
            scrollbar_px = tokens.sp(SCROLLBAR_DESIGN)
            for width_design in WIDTHS_DESIGN:
                width = tokens.sp(width_design)
                wide = settled_height(f"probe/{stage}/{width_design:g}/{scale:g}/wide", width)
                narrow = settled_height(
                    f"probe/{stage}/{width_design:g}/{scale:g}/narrow", width - scrollbar_px
                )
                if wide is None or narrow is None:
                    failures.append(
                        f"stage={stage} width={width_design:g} scale={scale:g}: "
                        "the probe pane never became visible"
                    )
                    continue
                lo, hi = sorted((wide, narrow))
                span = max(hi - lo, 2.0 * scrollbar_px)
                heights = sorted(
                    {
                        max(lo - span, 50.0),
                        lo,
                        (lo + hi) / 2.0,
                        hi,
                        hi + span,
                    }
                )
                for height in heights:
                    pane_id = f"sweep/{stage}/{width_design:g}/{scale:g}/{height:g}"
                    layout.reset_trace()
                    for _ in range(4):
                        frames(
                            lambda pane_id=pane_id, width=width, height=height: build_pane(
                                pane_id, width, height
                            ),
                            size=(width + 80.0, height + 260.0),
                        )

                    samples = layout.trace_samples(pane_id)
                    combo = (
                        f"stage={stage} width={width_design:g} scale={scale:g} "
                        f"height={height:g}"
                    )
                    if len(samples) < 4:
                        failures.append(f"{combo}: pane never became visible")
                        continue
                    if samples[-1] != samples[-2]:
                        failures.append(
                            f"{combo}: frame 3 {samples[-2]!r} != frame 4 {samples[-1]!r}"
                        )
                    reported = layout.oscillations()
                    if reported:
                        failures.append(f"{combo}: layout.oscillations() reported {reported!r}")

    assert not failures, "\n".join(failures)


# --- A3.1: the Mesh-stage matte preview --------------------------------------


def _matte_preview(width: int = 2000, height: int = 2000) -> svc_matte.Preview:
    # Large and square: ``_matte_image``'s scale is ``min(1.0, avail / width,
    # limit / height)``, and ``avail`` (a few hundred px, at most) has to stay
    # the binding term for the box to move at all -- a preview close to the
    # modal's own size would instead be bound by ``limit / height``, constant
    # regardless of ``avail``, which would make this test insensitive to the
    # very thing it exists to catch. Square rather than merely large: at 1:1
    # a change in ``avail`` lands on the drawn height at (close to) the same
    # number of px, which is what makes the few-px oscillation band below
    # findable at all -- a shallower aspect spreads the same avail delta over
    # a smaller height delta until no candidate height lands inside it.
    return svc_matte.Preview(
        job_id="0123456789ab",
        stamp=1,
        width=width,
        height=height,
        rgb=bytes(width * height * 3),
        source="birefnet",
        approved=False,
        coverage=0.42,
    )


def test_the_matte_preview_image_is_stable_across_the_scrollbar(frames, monkeypatch):
    """``settings_3d._matte_image`` (the modal that shows what Make 3D is
    about to cut out) used to size its aspect-preserving image off a raw
    ``imgui.get_content_region_avail().x`` -- the same shape as the three
    sites ``widgets.stable_width``'s docstring names, reached through
    ``widgets.modal_body``'s own scrolling child (see the comment now on
    ``_matte_image`` itself). This drives the real function twice, at the two
    avail widths one frame of that child actually alternates between -- the
    full width with no scrollbar and exactly ``style.scrollbar_size`` less
    with one -- and asserts the drawn box is identical either way, which is
    only true once the width comes from ``widgets.stable_content_width()``.
    """
    preview = _matte_preview()
    drawn: list[tuple[float, float]] = []
    real_image = imgui.image

    def capture(ref: Any, size: Any, *args: Any, **kwargs: Any) -> Any:
        drawn.append((float(size[0]), float(size[1])))
        return real_image(ref, size, *args, **kwargs)

    monkeypatch.setattr(imgui, "image", capture)
    monkeypatch.setattr(layout, "TRACE_ENABLED", True)
    ctx = SimpleNamespace(textures=textures_mod.ThumbnailCache(_FakeGL()))

    def build(pane_id: str, width: float, height: float) -> None:
        with layout.pane(pane_id, (width, height), layout.PaneRole.OVERLAY) as visible:
            if visible:
                settings_3d._matte_image(ctx, preview)

    def settled_height(pane_id: str, width: float) -> float:
        """This pane's total content extent (image plus the padding around
        it) at ``width``, against a pane so tall it never needs a scrollbar
        of its own -- the same technique and the same reason the main sweep
        above uses it: an outer pane height picked at random is very unlikely
        to land within the handful of px this image's height actually moves
        between "scrollbar down" and "scrollbar up", and this measures both
        directly rather than guessing.
        """
        layout.reset_trace()
        frames(
            lambda: build(pane_id, width, 3000.0), size=(width + 80.0, 3260.0)
        )
        return layout.trace_samples(pane_id)[-1][0]

    width = 400.0
    scrollbar_px = tokens.sp(SCROLLBAR_DESIGN)
    wide = settled_height("matte-probe-wide", width)
    narrow = settled_height("matte-probe-narrow", width - scrollbar_px)
    drawn.clear()

    # The wide (no-scrollbar) content height itself, not some point between
    # the two -- confirmed, while building this test, against the sweep
    # above's own known-real repro (rig stage, width 220): imgui's scrollbar
    # decision has a few px of one-sided hysteresis just above the narrow
    # value that *settles* rather than alternates, and the genuine infinite
    # A/B loop only starts once the height reaches the wide value. ``hi`` is
    # exactly that height, sorted-order making no assumption about which of
    # the two is larger.
    _lo, hi = sorted((wide, narrow))
    height = hi
    for _ in range(4):
        frames(lambda: build("matte-test", width, height), size=(width + 80.0, height + 260.0))

    assert len(drawn) == 4
    assert drawn[-1] == drawn[-2], (
        f"the matte preview drew {drawn[-2]} one frame and {drawn[-1]} the "
        "next, with nothing else changed -- the two must agree"
    )


# --- A3.2: the library footer reservation ------------------------------------


def test_the_library_footer_reservation_only_grows_within_one_regime(frames, monkeypatch):
    """``library._footer_reserve``/``_measure_footer`` (see the comment on
    ``library._footer_px`` for the feedback chain in full) latch the
    reservation so it only ever grows within one (scale, stable pane width)
    regime. This alternates the *measured* footer height between a small and
    a large value at a fixed width and scale -- exactly what the scrollbar
    loop would do to it -- and asserts the reservation converges to the
    larger value and stays there, rather than tracking the smaller one back
    down and feeding the loop again.
    """
    from warlock.studio.modes.library.ui.panes import library

    monkeypatch.setattr(library, "_footer_px", [36.0])
    monkeypatch.setattr(library, "_footer_regime", [None])

    def measure(height_px: float) -> None:
        imgui.begin_child("footer-regime-test", (400.0, 300.0))
        top = imgui.get_cursor_pos_y()
        imgui.dummy((0.0, height_px))
        library._measure_footer(top)
        imgui.end_child()

    frames(lambda: measure(40.0))
    grown = library._footer_px[0]
    assert grown >= 40.0

    frames(lambda: measure(18.0))
    # The smaller measurement, same regime: must not undercut what was
    # already reserved -- that give-back is exactly what lets the scrollbar
    # loop feed on itself.
    assert library._footer_px[0] == grown

    frames(lambda: measure(70.0))
    assert library._footer_px[0] > grown
    bigger = library._footer_px[0]

    frames(lambda: measure(10.0))
    assert library._footer_px[0] == bigger
