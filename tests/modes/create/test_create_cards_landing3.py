"""Create's band and centre column, as concept 03 draws them (Landing 3 of the
cards redesign, 2026-10-11).

The band is 46 dp (8 + 30 + 8) with the stage rail, New and Inspector on one
centre line; the canvas and the tray are separated by the 8 px gutter every card
is; the tray is one ``widgets.group`` card whose strip is a measured height, not
``(0, 0)``. Each test names the claim and fails against the code it replaced.
"""

from __future__ import annotations

import ast
import inspect
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from imgui_bundle import imgui

from realmspinner.studio import layout, probe, theme, tokens, widgets
from realmspinner.studio.modes.create.ui import brief as create_brief
from realmspinner.studio.modes.create.ui import canvas as create_canvas
from realmspinner.studio.modes.create.ui import rail as create_rail
from realmspinner.studio.modes.create.ui import stages as create_stages
from realmspinner.studio.modes.create.ui import workspace as gw
from realmspinner.studio.state import AppState
from realmspinner.studio.tokens import sp

SRC = Path(layout.__file__).resolve().parent


@pytest.fixture
def frames():
    """A bare imgui context; ``draw(build, size, repeat)`` runs ``build`` for
    ``repeat`` frames (a ``widgets.group`` card is one frame late, so geometry is
    read on the last of a few). No GL and no renderer, the way
    ``test_create_brief.py``'s fixture of the same name works."""
    previous = imgui.get_current_context()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)

    def draw(build: Any, size: tuple[float, float] = (1200.0, 900.0), repeat: int = 3) -> None:
        for _ in range(repeat):
            imgui.new_frame()
            imgui.set_next_window_pos((0.0, 0.0))
            imgui.set_next_window_size(size)
            imgui.begin("smoke", flags=imgui.WindowFlags_.no_scrollbar.value)
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


# --- the band ----------------------------------------------------------------


def _rail(ctx: Any, *, max_width: float | None = None, row_height: float | None = None) -> None:
    """``App._stage_rail`` without a job: five unblocked stages."""
    items = [
        (stage, create_stages.LABELS[stage], create_stages.ICONS[stage], None)
        for stage in create_stages.STAGES
    ]
    create_rail.stage_rail(
        "create-stages", items, ctx.state.create.stage, max_width=max_width, row_height=row_height
    )


def _band_ctx() -> Any:
    state = AppState(mode="create")
    state.create.stage = "reference"
    return SimpleNamespace(
        state=state,
        svc=SimpleNamespace(config=None),
        busy=lambda _key: False,
        confirms=SimpleNamespace(ask=lambda *_a: None),
        cache=SimpleNamespace(get=lambda _id: None, jobs=[], active=None),
        job=lambda: None,
        textures=None,
        model_rows=[],
        submit=lambda *a, **k: True,
        toast=lambda *a, **k: None,
    )


def test_the_band_height_is_derived_from_the_control_row_and_the_inset_not_measured():
    """46 = 8 + 30 + 8. It was a measured 72: a guess that happened to fit, which
    left the one 30 dp row floating in a band 26 dp taller than itself."""
    assert tokens.CONTROL_HEIGHT_REGULAR + 2 * tokens.SP_2 == 46.0
    assert create_brief.RAIL_ONLY_H == tokens.CONTROL_HEIGHT_REGULAR + 2 * tokens.SP_2
    assert create_brief.bar_height(_band_ctx()) == sp(46.0)
    # Spelled from tokens in source, so a retuned control height moves the band.
    source = inspect.getsource(create_brief)
    assert "RAIL_ONLY_H = tokens.CONTROL_HEIGHT_REGULAR + 2 * tokens.SP_2" in source


def _draw_band(frames: Any, monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], dict[str, float]]:
    monkeypatch.setattr(probe, "ENABLED", True)
    ctx = _band_ctx()
    seen: dict[str, float] = {}

    def build() -> None:
        layout.begin_frame()
        probe.begin_frame()
        seen["origin"] = float(imgui.get_cursor_screen_pos().y)
        with layout.pane(
            "brief",
            (1200.0, create_brief.bar_height(ctx)),
            layout.PaneRole.CONTENT,
            padding=tokens.SP_2,
        ) as visible:
            if visible:
                create_brief.draw(ctx, _rail)
                seen["scroll_max"] = float(imgui.get_scroll_max_y())

    frames(build)
    return [c for c in probe.census() if c.where == "brief"], seen


def test_the_rail_new_and_inspector_share_one_centre_line(frames, monkeypatch):
    """The rail capsule is ~25 dp and the buttons 30; left to ``row_height=None``
    the capsule sat at the row's top, 2.5 px above the centre the buttons share."""
    controls_in_band, _ = _draw_band(frames, monkeypatch)
    rail = [c for c in controls_in_band if c.text in create_stages.LABELS.values()]
    buttons = [c for c in controls_in_band if c.text in ("New...", "Inspector")]
    assert len(rail) == len(create_stages.STAGES) and len(buttons) == 2
    centres = [c.rect[1] + c.rect[3] / 2 for c in rail + buttons]
    assert max(centres) - min(centres) <= 1.0, centres


def test_the_bands_row_fills_the_pane_exactly_with_the_pane_inset(frames, monkeypatch):
    """8 above the 30 dp row, 8 below it, and nothing to scroll: the declared
    height and what is drawn are the same number."""
    controls_in_band, seen = _draw_band(frames, monkeypatch)
    buttons = [c for c in controls_in_band if c.text in ("New...", "Inspector")]
    top = min(c.rect[1] for c in buttons)
    bottom = max(c.rect[1] + c.rect[3] for c in buttons)
    pane_top = seen["origin"]
    assert top - pane_top == pytest.approx(sp(tokens.SP_2), abs=1.0)
    assert (pane_top + create_brief.bar_height(_band_ctx())) - bottom == pytest.approx(
        sp(tokens.SP_2), abs=1.0
    )
    assert seen["scroll_max"] == pytest.approx(0.0, abs=0.5)


def test_the_new_and_inspector_reserve_is_the_two_item_gaps_not_a_literal():
    tree = ast.parse(inspect.getsource(create_brief))
    draw = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "draw")
    literals = [
        node.args[0].value
        for node in ast.walk(draw)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sp"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    assert 24 not in literals, "the reserve is back to a hand-typed slack"
    body = ast.get_source_segment(inspect.getsource(create_brief), draw)
    assert "item_spacing.x" in body
    assert "row_height=" in body


# --- the tray ----------------------------------------------------------------


def _jobs(count: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"job{index:04d}",
            "name": f"attempt number {index}",
            "stage": "reference",
            "status": "done",
            "favorite": index == 0,
            "files": ["input.png"],
        }
        for index in range(count)
    ]


def _tray_ctx() -> Any:
    state = AppState(mode="create")
    state.create.stage = "reference"
    return SimpleNamespace(
        state=state,
        cache=SimpleNamespace(active=None, jobs=[], get=lambda _id: None),
    )


class _TraySpies:
    """What a drawn tray did, read off wrapped real calls."""

    def __init__(self) -> None:
        self.titles: list[str] = []
        self.pane_headers: list[str] = []
        self.thumb_sides: list[float] = []
        self.name_heights: list[float] = []
        self.pill_bottom = 0.0
        self.note_bottom = 0.0
        self.card: tuple[float, float] = (0.0, 0.0)
        self.outer_scroll_max = 0.0


def _install(monkeypatch: pytest.MonkeyPatch, jobs: list[dict[str, Any]]) -> _TraySpies:
    spies = _TraySpies()
    monkeypatch.setattr(gw.families, "results", lambda *a, **k: jobs)
    monkeypatch.setattr(gw.session, "index", lambda _ctx: None)
    monkeypatch.setattr(gw.candidates_mod, "pending_cached", lambda _cache: None)

    def thumb(_ctx: Any, _job: Any, side: float) -> None:
        spies.thumb_sides.append(side)
        imgui.dummy((side, side))

    monkeypatch.setattr(gw.thumbs, "job_thumb", thumb)

    real_group = widgets.group

    @contextmanager
    def group(*args: Any, **kwargs: Any):
        spies.titles.append(str(args[0]) if args else "")
        with real_group(*args, **kwargs) as opened:
            yield opened
        spies.card = (float(imgui.get_item_rect_min().y), float(imgui.get_item_rect_max().y))
        spies.outer_scroll_max = float(imgui.get_scroll_max_y())

    monkeypatch.setattr(widgets, "group", group)

    real_header = widgets.pane_header

    def pane_header(text: str, *args: Any, **kwargs: Any) -> None:
        spies.pane_headers.append(text)
        real_header(text, *args, **kwargs)

    monkeypatch.setattr(widgets, "pane_header", pane_header)

    real_button = gw.controls.button

    def button(label: str, *args: Any, **kwargs: Any) -> bool:
        pressed = real_button(label, *args, **kwargs)
        if label.endswith("##select"):
            spies.name_heights.append(float(imgui.get_item_rect_size().y))
        return pressed

    monkeypatch.setattr(gw.controls, "button", button)

    real_pill = widgets.status_pill

    def pill(status: str) -> None:
        real_pill(status)
        spies.pill_bottom = max(spies.pill_bottom, float(imgui.get_item_rect_max().y))

    monkeypatch.setattr(widgets, "status_pill", pill)

    real_muted = widgets.muted

    def muted(text: str) -> None:
        real_muted(text)
        spies.note_bottom = max(spies.note_bottom, float(imgui.get_item_rect_max().y))

    monkeypatch.setattr(widgets, "muted", muted)
    return spies


def test_the_tray_is_one_card_titled_attempts_and_not_a_pane_header(frames, monkeypatch):
    spies = _install(monkeypatch, _jobs(3))
    ctx = _tray_ctx()
    frames(lambda: gw.draw(ctx, sp(300), "reference"))
    assert spies.titles == ["Attempts"] * 3  # one card a frame
    assert spies.pane_headers == []
    source = inspect.getsource(gw.draw)
    assert "pane_header" not in source.replace("``pane_header``", "")
    assert 'widgets.group("Attempts"' in source


def test_attempt_thumbnails_are_80_and_their_name_buttons_are_26_tall(frames, monkeypatch):
    spies = _install(monkeypatch, _jobs(3))
    ctx = _tray_ctx()
    frames(lambda: gw.draw(ctx, sp(300), "reference"))
    assert set(spies.thumb_sides) == {sp(80)}
    assert gw.ATTEMPT_THUMB_DP == 80.0
    assert spies.name_heights and set(spies.name_heights) == {sp(26)}


@pytest.mark.parametrize("window", [(1600, 950), (1100, 700)])
def test_the_strip_is_not_clipped_and_the_tray_stays_within_48_percent(
    frames, monkeypatch, window
):
    """The floor was a guessed ``sp(186)`` with a pane header over it. The card's
    own chrome is 58 dp, so a strip of 82 + button + pill + note ran past the
    tray's bottom at every window size."""
    jobs = _jobs(12)
    jobs[1]["id"] = "chosen"
    spies = _install(monkeypatch, jobs)
    ctx = _tray_ctx()
    ctx.state.source_job = "chosen"
    # What the viewport pane has left under the menu, band and toolbar.
    pane_h = float(window[1]) - 46 - 36 - 16 - 30 - 8
    sizes: dict[str, float] = {}

    def build() -> None:
        sizes["tray"] = gw.tray_height(ctx, pane_h)
        sizes["strip"] = gw.strip_height()
        gw.draw(ctx, sizes["tray"], "reference")

    frames(build, (float(window[0]), 900.0))
    tray_h = sizes["tray"]
    assert gw.TRAY_MAX_FRACTION == 0.48
    assert tray_h <= pane_h * 0.48 + 0.01
    assert tray_h == pytest.approx(sp(gw.TRAY_CHROME_DP) + sizes["strip"], abs=0.5), (
        "with nothing running the whole card fits under the cap"
    )
    card_top, card_bottom = spies.card
    assert spies.note_bottom > 0 and spies.pill_bottom > 0
    chrome_bottom = card_bottom - sp(tokens.SP_3)
    assert spies.note_bottom <= chrome_bottom + 0.5, "last cell line must clear the card padding"
    assert spies.pill_bottom <= spies.note_bottom
    assert card_bottom - card_top <= tray_h + 0.5
    assert spies.outer_scroll_max <= 0.5, "the tray scrolls: something is taller than the tray"


def test_a_running_job_grows_the_tray_but_never_past_48_percent(frames):
    ctx = _tray_ctx()
    got: dict[str, float] = {}

    def build() -> None:
        ctx.cache.active = None
        got["idle"] = gw.tray_height(ctx, 2000.0)
        ctx.cache.active = {"id": "x"}
        got["busy"] = gw.tray_height(ctx, 2000.0)
        got["capped"] = gw.tray_height(ctx, 400.0)

    frames(build, repeat=1)
    assert got["busy"] - got["idle"] == pytest.approx(sp(gw._PROGRESS_DP))
    assert got["capped"] == pytest.approx(400.0 * 0.48)


def test_the_strip_child_has_an_explicit_height():
    """Inside an ``auto_resize_y`` card a ``(0, 0)`` child is "as tall as I
    already am"; it must be the measured strip height."""
    source = inspect.getsource(gw._attempt_strip)
    assert '"create-attempt-strip", (0, 0)' not in source
    assert '"create-attempt-strip", (0, strip_height())' in source


# --- the centre column -------------------------------------------------------


def _bare_calls(path: Path, name: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "imgui"
    ]


@pytest.mark.parametrize(
    "path",
    [SRC / "shell" / "frame.py", SRC / "modes" / "create" / "ui" / "workspace.py"],
    ids=lambda p: p.name,
)
def test_no_raw_imgui_separator_in_the_shell_frame_or_the_tray(path):
    """The panes' guard scans ``panes/*.py`` only; these two files drew the rule
    between the canvas and the tray, in the stage column and in More by hand."""
    assert _bare_calls(path, "separator") == []


def test_the_canvas_and_the_tray_are_exactly_one_gutter_apart(frames, monkeypatch):
    """The separator between them added an item spacing either side of a 1 px
    rule that ``canvas_height`` never subtracted, so the tray sat about 17 px
    lower than the arithmetic said and the pane overflowed by that much."""
    from realmspinner.studio.shell import frame

    measured: dict[str, float] = {}
    ctx = SimpleNamespace(
        state=SimpleNamespace(
            mode="create",
            create=SimpleNamespace(stage="reference", inspector_open=True),
        ),
        viewer=None,
        cache=SimpleNamespace(active=None),
    )

    def canvas(*_args: Any, **_kwargs: Any) -> None:
        # Drawn inside the canvas child: its window is the canvas' rect.
        measured["canvas_bottom"] = float(imgui.get_window_pos().y + imgui.get_window_size().y)

    shell = SimpleNamespace(app_ctx=ctx, _canvas_body=canvas)

    def tray(_ctx: Any, height: float, _stage: str) -> None:
        # Called with the cursor where the tray's top edge will be.
        measured["tray_top"] = float(imgui.get_cursor_screen_pos().y)
        measured["height"] = height

    monkeypatch.setattr(gw, "should_draw", lambda *a, **k: True)
    monkeypatch.setattr(gw, "draw", tray)

    def build() -> None:
        layout.begin_frame()
        frame.FrameMixin._viewport_pane(shell)

    frames(build, (1200.0, 900.0), repeat=2)
    assert measured["tray_top"] - measured["canvas_bottom"] == pytest.approx(
        sp(tokens.SP_2), abs=0.5
    )
    assert measured["height"] > 0


def test_the_centre_column_floor_and_inset_are_scaled_and_on_the_gutter():
    from realmspinner.studio.shell import frame

    source = inspect.getsource(frame.FrameMixin._viewport_pane)
    assert "max(avail.y, 64)" not in source
    assert "max(avail.y, sp(64))" in source
    assert "padding=tokens_mod.SP_2" in source


# --- the Make 3D overlay -----------------------------------------------------


def test_the_make_3d_overlay_is_a_regular_control_tall_with_insets_from_the_scale(
    frames, monkeypatch
):
    """It was 38 tall (one off the 30/26/40 heights), 40 from the top and 18 from
    the bottom."""
    seen: dict[str, Any] = {}

    def primary(label: str, size: tuple[float, float] = (0, 0), **kwargs: Any) -> bool:
        seen["size"] = size
        seen["pos"] = imgui.get_cursor_screen_pos()
        imgui.dummy(size)
        return False

    monkeypatch.setattr(widgets, "primary_button", primary)
    monkeypatch.setattr(create_canvas, "grid_batch", lambda *a, **k: None)
    monkeypatch.setattr(create_canvas, "labelable", lambda _job: False)
    monkeypatch.setattr(create_canvas, "make_3d_job", lambda _ctx: {"id": "j"})
    ctx = SimpleNamespace(
        state=SimpleNamespace(create=SimpleNamespace(stage="reference")), job=lambda: None
    )

    def build() -> None:
        create_canvas.draw_overlay(ctx)
        seen["window"] = (imgui.get_window_pos(), imgui.get_window_size())

    frames(build, repeat=1)
    assert seen["size"][1] == sp(30)
    pos, size = seen["window"]
    below = pos.y + size.y - (seen["pos"].y + seen["size"][1])
    assert below == pytest.approx(sp(tokens.SP_4))

    scale = {tokens.SP_1, tokens.SP_2, tokens.SP_3, tokens.SP_4, tokens.SP_6}
    assert create_canvas._OVERLAY_SIDE_DP in scale
    assert create_canvas._OVERLAY_BOTTOM_DP in scale
    assert create_canvas._OVERLAY_TOP_DP == tokens.SP_6 + tokens.SP_4
    source = inspect.getsource(create_canvas.draw_overlay)
    for literal in ("sp(38)", "sp(18)", "sp(40)", "sp(16)"):
        assert literal not in source, literal
