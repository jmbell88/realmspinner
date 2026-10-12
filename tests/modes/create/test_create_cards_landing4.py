"""Create's right column, as concept 03 draws it (Landing 4 of the cards
redesign, 2026-10-11).

The Inspector is a stack of cards in Create -- an identity card, a Next step
card, a collapsible card per section -- and the Library's full-window inspector,
which draws the very same function, keeps its flat look. History is a card of
title, filters and the corpus line, then a card per attempt whose **Load
settings** has a line of its own. The two panes are 8 px apart and 8 px inside.

Each test names the claim and fails against the code it replaced.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from imgui_bundle import imgui
from PIL import Image

from realmspinner.studio import controls, layout, theme, tokens, widgets
from realmspinner.studio import settings as studio_settings
from realmspinner.studio import textures as textures_mod
from realmspinner.studio.app_ctx import Ctx
from realmspinner.studio.modes.create.ui.panes import history
from realmspinner.studio.panes import inspector
from realmspinner.studio.shell import frame as frame_mod
from realmspinner.studio.state import AppState
from realmspinner.studio.tokens import sp

SRC = Path(layout.__file__).resolve().parent


@pytest.fixture
def frames():
    """A bare imgui context; ``draw(build, size, repeat)`` runs ``build`` for
    ``repeat`` frames in one host window (a ``widgets.group`` card is one frame
    late, so geometry is read on the last of a few)."""
    previous = imgui.get_current_context()
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)

    def draw(build: Any, size: tuple[float, float] = (400.0, 1400.0), repeat: int = 3) -> None:
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
    imgui.destroy_context(context)
    if previous is not None:
        imgui.set_current_context(previous)


# --- the inspector ----------------------------------------------------------------------


class _FakeTexture:
    _next = 1

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.filter = None
        self.repeat_x = True
        self.repeat_y = True
        self.glo = _FakeTexture._next
        _FakeTexture._next += 1

    def release(self) -> None:
        pass


class _FakeGL:
    NEAREST = "nearest"
    LINEAR = "linear"

    def texture(self, size: tuple[int, int], components: int, data: bytes) -> _FakeTexture:
        return _FakeTexture(size)


class _FakeTasks:
    def is_busy(self, key: str) -> bool:
        return False

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        return True

    def progress(self, key: str) -> Any:
        return None


def _reference_job(svc: Any, **extra: Any) -> dict[str, Any]:
    job_id = svc.store.create(
        "text", "a hooded adventurer", {"seed": 5, "base_model": "sdxl"},
        stage="reference", status="done",
    )
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (256, 256), (180, 70, 70, 255)).save(job_dir / "input.png")
    job = svc.store.get(job_id)
    job["files"] = ["input.png"]
    job.update(extra)
    return job


def _ctx(svc: Any, job: dict[str, Any], *, mode: str, stage: str = "reference") -> Ctx:
    state = AppState(mode=mode, selected=job["id"])
    state.create.stage = stage
    parent = {"id": "p1", "name": "the first draft", "status": "done", "stage": "reference",
              "files": ["input.png"]}
    cache = SimpleNamespace(
        get=lambda job_id: job if job_id == job["id"] else (parent if job_id == "p1" else None),
        jobs=[job],
    )
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


class _Spy:
    """Records every card, sub-label and sized item the inspector draws."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.cards: list[tuple[str | None, str | None, bool]] = []
        self.sub_labels: list[str] = []
        self.sizes: dict[str, tuple[float, float, float, float]] = {}
        self.owners: dict[str, str] = {}
        real_group = widgets.group
        real_sub = widgets.sub_label
        real_input = widgets.input_text
        real_small = controls.small_button
        real_primary = widgets.primary_button
        real_button = controls.button
        real_summary = widgets.asset_summary

        def group(*args: Any, **kwargs: Any):
            title = args[0] if args else kwargs.get("title")
            self.cards.append((title, kwargs.get("id"), bool(kwargs.get("collapsible"))))
            return real_group(*args, **kwargs)

        def sub_label(label: str) -> None:
            self.sub_labels.append(label)
            real_sub(label)

        def note(name: str) -> None:
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            self.sizes[name] = (lo.x, lo.y, hi.x, hi.y)
            self.owners[name] = widgets._window_owner()

        def input_text(label: str, *args: Any, **kwargs: Any) -> str:
            out = real_input(label, *args, **kwargs)
            note(label)
            return out

        def small_button(label: str, *args: Any, **kwargs: Any) -> bool:
            out = real_small(label, *args, **kwargs)
            note(label)
            return out

        def primary_button(label: str, *args: Any, **kwargs: Any) -> bool:
            out = real_primary(label, *args, **kwargs)
            note("primary")
            return out

        def button(label: str, *args: Any, **kwargs: Any) -> bool:
            out = real_button(label, *args, **kwargs)
            note(label)
            return out

        def asset_summary(job: Any, **kwargs: Any) -> None:
            self.owners["summary"] = widgets._window_owner()
            real_summary(job, **kwargs)

        monkeypatch.setattr(widgets, "group", group)
        monkeypatch.setattr(widgets, "sub_label", sub_label)
        monkeypatch.setattr(widgets, "input_text", input_text)
        monkeypatch.setattr(controls, "small_button", small_button)
        monkeypatch.setattr(widgets, "primary_button", primary_button)
        monkeypatch.setattr(controls, "button", button)
        monkeypatch.setattr(widgets, "asset_summary", asset_summary)

    def titles(self) -> list[str | None]:
        return [title for title, _id, _fold in self.cards]

    def height(self, name: str) -> float:
        lo_x, lo_y, hi_x, hi_y = self.sizes[name]
        return hi_y - lo_y


def _draw_inspector(frames: Any, ctx: Any) -> None:
    frames(lambda: inspector.draw(ctx))


def test_the_library_inspector_draws_no_cards(frames, svc, monkeypatch):
    """The full-window inspector in the Library draws the same function, with no
    card scope open: it keeps its flat headings and its tab bar. Cards belong to
    Create's column alone."""
    spy = _Spy(monkeypatch)
    job = _reference_job(svc)
    ctx = _ctx(svc, job, mode="library")
    _draw_inspector(frames, ctx)
    assert spy.cards == [], "a card leaked into the Library's inspector"
    assert spy.sub_labels == []


def test_create_draws_an_identity_card_a_next_step_card_and_a_card_per_section(
    frames, svc, monkeypatch
):
    monkeypatch.setattr(widgets, "FORCE_SECTIONS_OPEN", True)
    spy = _Spy(monkeypatch)
    job = _reference_job(svc)
    ctx = _ctx(svc, job, mode="create", stage="reference")
    _draw_inspector(frames, ctx)
    ids = [card_id for _title, card_id, _fold in spy.cards]
    assert ids[0] == "identity", "the identity card is first, and header-less"
    assert spy.cards[0][0] is None
    assert "Next step" in spy.titles()
    assert spy.titles().index("Next step") == 1
    # Every section the Reference stage draws is its own card, the destinations fold
    # and the stage's own folds (collapsible) included.
    folds = {title for title, _id, fold in spy.cards if fold}
    assert {"Open in...", "Generation settings", "Pixel art"} <= folds, spy.cards


def test_the_identity_name_is_a_30_px_field_in_create_and_heading_sized_in_the_library(
    frames, svc, monkeypatch
):
    """The heading face made the name a 36 px field above the 30 px tags field
    (finding 6). In Create the identity card is the header, so the name is an
    ordinary field -- 30 px, the same as the tags field under it; the Library has
    no card, so it keeps the heading."""
    spy = _Spy(monkeypatch)
    job = _reference_job(svc)
    _draw_inspector(frames, _ctx(svc, job, mode="create"))
    # The regular control height is the frame height (30 in the app; the headless
    # default font is a pixel shorter, which is why this reads it rather than 30).
    assert spy.height("##name") == pytest.approx(imgui.get_frame_height(), abs=0.5)
    assert spy.height("##tags") == pytest.approx(spy.height("##name"), abs=0.5)
    assert spy.height("Favourite") == pytest.approx(sp(tokens.CONTROL_HEIGHT_COMPACT), abs=0.5)

    # Which face the name is drawn in is what differs (the headless atlas has one
    # font, so heights cannot show it): the heading in the Library, never in Create.
    from realmspinner.studio import fonts

    used: list[str] = []
    real_heading = fonts.heading

    def heading(gui: Any):
        used.append("heading")
        return real_heading(gui)

    monkeypatch.setattr(fonts, "heading", heading)
    _draw_inspector(frames, _ctx(svc, job, mode="create"))
    assert used == [], "Create's name field must not use the heading face"
    _draw_inspector(frames, _ctx(svc, job, mode="library"))
    assert used, "the Library keeps its heading-sized name"


def test_take_it_somewhere_inside_open_in_is_a_sub_label_not_a_card_in_a_card(
    frames, svc, monkeypatch
):
    monkeypatch.setattr(widgets, "FORCE_SECTIONS_OPEN", True)
    spy = _Spy(monkeypatch)
    job = _reference_job(svc)
    _draw_inspector(frames, _ctx(svc, job, mode="create", stage="reference"))
    assert "Take it somewhere" in spy.sub_labels
    assert "Take it somewhere" not in spy.titles()


def test_a_section_after_an_open_fold_is_a_sibling_card(frames, svc, monkeypatch):
    """"Open in..." is open and the Reference stage's first section (Lineage)
    follows it. Left to the scope's rule that reads as part of the fold -- a
    sub-label inside "Open in..." -- so ``_stage_body`` closes the card between
    sections."""
    monkeypatch.setattr(widgets, "FORCE_SECTIONS_OPEN", True)
    spy = _Spy(monkeypatch)
    job = _reference_job(svc, parent_id="p1")
    _draw_inspector(frames, _ctx(svc, job, mode="create", stage="reference"))
    assert "Lineage" in spy.titles(), "Lineage was swallowed into the fold before it"
    assert "Lineage" not in spy.sub_labels
    body = ast.get_source_segment(
        inspect.getsource(inspector), _function(inspector, "_stage_body")
    )
    assert "widgets.end_section()" in body


def _function(module: Any, name: str) -> ast.FunctionDef:
    tree = ast.parse(inspect.getsource(module))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_next_step_puts_the_primary_and_more_on_one_row_with_the_summary_in_the_card(
    frames, svc, monkeypatch
):
    spy = _Spy(monkeypatch)
    job = _reference_job(svc)
    ctx = _ctx(svc, job, mode="create", stage="reference")
    _draw_inspector(frames, ctx)
    primary, more = spy.sizes["primary"], spy.sizes["More##next-step-more"]
    assert primary[1] == pytest.approx(more[1], abs=0.5), "one row"
    assert primary[2] < more[0], "the primary sits left of More"
    # Flush: the row spends the card's whole inner width (the gap between the two
    # is the item spacing, so the primary is derived rather than guessed).
    assert more[0] - primary[2] == pytest.approx(imgui.get_style().item_spacing.x, abs=0.5)
    assert spy.owners["primary"] == spy.owners["More##next-step-more"] == spy.owners["summary"]
    assert "Next step" in spy.owners["summary"], "the summary is inside the Next step card"


def test_a_failure_is_its_own_card_not_buried_in_next_step(frames, svc, monkeypatch):
    spy = _Spy(monkeypatch)
    job = _reference_job(svc)
    job["status"] = "error"
    job["error"] = "TRELLIS ran out of memory."
    _draw_inspector(frames, _ctx(svc, job, mode="create", stage="reference"))
    ids = [card_id for _t, card_id, _f in spy.cards]
    assert "problems" in ids
    assert ids.index("problems") > spy.titles().index("Next step")


def test_stable_content_width_inside_a_card_is_measured_against_the_panes_scrollbar(frames):
    """A card is a child with no scrollbar, so asked from inside it ``stable_content_width``
    answered "no bar, here is my live avail" -- the pane's width, which the pane's own
    scrollbar moves a frame late. An image sized from it inside a card then fed its
    height back into the scrollbar decision and the Create inspector stopped settling
    (``test_layout_stability`` at 1.6x, Mesh stage). The card hands down the width as
    if its pane's bar were showing, less its own 12 px padding each side."""
    seen: dict[str, float] = {}

    def build() -> None:
        imgui.dummy((10.0, 5000.0))  # taller than the host: the pane scrolls
        outer = widgets.stable_content_width()
        assert imgui.get_scroll_max_y() > 0 or imgui.get_frame_count() < 2
        with widgets.group(id="stable"):
            seen["inside"] = widgets.stable_content_width()
            seen["live"] = imgui.get_content_region_avail().x
        seen["outer"] = outer

    frames(build)
    assert seen["inside"] == pytest.approx(seen["outer"] - 2 * sp(tokens.SP_3), abs=0.01)
    assert seen["inside"] >= seen["live"] - 0.01, "the pane's bar is already off the live width"


# --- the right column's two panes -------------------------------------------------------


def _right_column_geometry(frames: Any, tmp_path: Path) -> dict[str, float]:
    settings = studio_settings.Settings(tmp_path / studio_settings.FILENAME)
    lay = layout.Layout(settings)
    seen: dict[str, float] = {}

    def probe_pane(name: str):
        def draw(_ctx: Any) -> None:
            win, size = imgui.get_window_pos(), imgui.get_window_size()
            seen[f"{name}/pad_x"] = imgui.get_cursor_screen_pos().x - win.x
            seen[f"{name}/pad_y"] = imgui.get_cursor_screen_pos().y - win.y
            seen[f"{name}/top"] = win.y
            seen[f"{name}/bottom"] = win.y + size.y

        return draw

    def build() -> None:
        layout.begin_frame()
        frame_mod._right_column(
            object(), lay, 300.0, inspector_draw=probe_pane("inspector"),
            library_draw=probe_pane("history"),
        )

    frames(build, size=(340.0, 900.0))
    return seen


def test_both_right_column_panes_pad_8_and_sit_8_apart(frames, tmp_path):
    """Inspector and History are columns of cards, so they take the 8 px inset
    the left column and the viewport have; and the gap between them is the same
    8 px every card gutter is, with the 7 px drag handle living inside it (it was
    grip + two item spacings, 23 px at 1x)."""
    seen = _right_column_geometry(frames, tmp_path)
    for pane in ("inspector", "history"):
        assert seen[f"{pane}/pad_x"] == pytest.approx(sp(tokens.SP_2), abs=0.5), pane
        assert seen[f"{pane}/pad_y"] == pytest.approx(sp(tokens.SP_2), abs=0.5), pane
    gap = seen["history/top"] - seen["inspector/bottom"]
    assert gap == pytest.approx(sp(tokens.SP_2), abs=0.5)
    assert sp(layout.GRIP) <= gap, "the grip has to fit inside the gap"


def test_the_gutter_option_is_opt_in_so_other_split_columns_keep_their_stack():
    """Packwright and Review share ``_split_column``: ``gutter=None`` is today's
    item-spacing + grip + item-spacing, and ``padding=None`` today's pane padding."""
    params = inspect.signature(frame_mod._split_column).parameters
    assert params["gutter"].default is None
    assert params["padding"].default is None


# --- History -----------------------------------------------------------------------------


class _Recording:
    """A window draw list that remembers what was drawn and forwards it."""

    def __init__(self, real: Any, log: list[tuple[str, tuple[Any, ...]]]) -> None:
        self._real = real
        self._log = log

    def add_rect_filled(self, *args: Any, **kwargs: Any) -> Any:
        self._log.append(("rect", args))
        return self._real.add_rect_filled(*args, **kwargs)

    def add_text(self, *args: Any, **kwargs: Any) -> Any:
        self._log.append(("text", args))
        return self._real.add_text(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _ImguiProxy:
    """``imgui`` with ``get_window_draw_list`` swapped for a recording one."""

    def __init__(self, log: list[tuple[str, tuple[Any, ...]]]) -> None:
        self._log = log

    def get_window_draw_list(self) -> _Recording:
        return _Recording(imgui.get_window_draw_list(), self._log)

    def __getattr__(self, name: str) -> Any:
        return getattr(imgui, name)


def _row(job_id: str, **over: Any) -> dict[str, Any]:
    row = {
        "job_id": job_id, "exists": True, "trashed": False, "snapshot": {"v": 1},
        "prompt": "a hooded adventurer, standing, long cloak",
        "settings_line": "SDXL 1.0 - Render 3D 0.8 - 30 steps - seed 1204285292 - tail tail tail",
        "thumbnail": None, "reference_label": None, "mesh": None, "tags": [], "notes": [],
    }
    row.update(over)
    return row


def _history_ctx(rows: list[dict[str, Any]]) -> Any:
    state = AppState(mode="create")
    state.create.history_rows = rows
    state.create.history_dirty = False
    state.create.history_options = {
        "base_model": [("sdxl", "SDXL")], "style_lora": [], "generation_type": [],
    }
    return SimpleNamespace(
        svc=SimpleNamespace(config=SimpleNamespace(bench_dir="nowhere")),
        state=state,
        cache=SimpleNamespace(get=lambda _id: None, jobs=[], _generation=1),
        textures=None,
        submit=lambda *a, **k: True,
        toast=lambda *a, **k: None,
    )


def _draw_history(
    frames: Any, monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]], width: float
) -> dict[str, Any]:
    """Draw History in a ``width`` dp pane and collect what it put where."""
    log: list[tuple[str, tuple[Any, ...]]] = []
    items: dict[str, list[tuple[float, float, float, float]]] = {}
    real_small = controls.small_button
    real_combo = widgets.combo
    real_icon = widgets.icon_button

    def note(name: str) -> None:
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        items.setdefault(name, []).append((lo.x, lo.y, hi.x, hi.y))

    def small_button(label: str, *a: Any, **k: Any) -> bool:
        out = real_small(label, *a, **k)
        note(label.split("##")[0] if "history-load" not in label else "load")
        return out

    def combo(label: str, *a: Any, **k: Any) -> Any:
        out = real_combo(label, *a, **k)
        note(label)
        return out

    def icon_button(label: str, *a: Any, **k: Any) -> bool:
        out = real_icon(label, *a, **k)
        note("help" if "help-create-history" in label else label)
        return out

    monkeypatch.setattr(controls, "small_button", small_button)
    monkeypatch.setattr(widgets, "combo", combo)
    monkeypatch.setattr(widgets, "icon_button", icon_button)
    monkeypatch.setattr(history, "imgui", _ImguiProxy(log))
    ctx = _history_ctx(rows)
    pane: dict[str, float] = {}
    metrics: list[Any] = []

    def build() -> None:
        items.clear()
        log.clear()
        metrics[:] = [history.RowMetrics()]
        with layout.pane("history", (sp(width), 700.0), padding=tokens.SP_2) as visible:
            if visible:
                win = imgui.get_window_pos()
                pane["x"], pane["y"] = win.x, win.y
                history.draw(ctx)

    frames(build, size=(sp(width) + 40.0, 800.0), repeat=4)
    return {"items": items, "log": log, "pane": pane, "metrics": metrics[0]}


def test_history_title_export_and_help_are_one_header_row(frames, monkeypatch):
    out = _draw_history(frames, monkeypatch, [_row("a1")], 300.0)
    export = out["items"]["Export..."][0]
    (help_rect,) = out["items"]["help"]
    top = out["pane"]["y"] + sp(tokens.SP_2) + sp(tokens.SP_3)  # pane padding + card padding
    header_h = sp(tokens.CONTROL_HEIGHT_COMPACT)
    for rect in (export, help_rect):
        assert rect[1] == pytest.approx(top, abs=1.0), "on the card's header row"
        assert rect[3] - rect[1] == pytest.approx(header_h, abs=1.0)
    assert help_rect[2] <= export[0] + 0.5, "the (?) sits left of Export, never over it"


def test_the_four_history_filters_are_a_two_by_two_grid(frames, monkeypatch):
    out = _draw_history(frames, monkeypatch, [_row("a1")], 300.0)
    models, styles, types, judged = (
        out["items"][f"##history-{name}"][0]
        for name in ("base_model", "style_lora", "generation_type", "judged")
    )
    widths = [r[2] - r[0] for r in (models, styles, types, judged)]
    assert max(widths) - min(widths) <= 1.0, widths
    assert models[1] == pytest.approx(styles[1], abs=0.5)
    assert types[1] == pytest.approx(judged[1], abs=0.5)
    assert types[1] > models[3], "the second pair is the second row"
    assert styles[0] > models[2] - 0.5 and judged[0] > types[2] - 0.5


@pytest.mark.parametrize("width", [300.0, 230.0], ids=["sidebar", "squeezed-1100x700"])
def test_a_history_row_is_a_card_surface_and_load_settings_is_clear_of_its_text(
    frames, monkeypatch, width
):
    rows = [_row("a1"), _row("b2", prompt="a short prompt", settings_line="")]
    out = _draw_history(frames, monkeypatch, rows, width)
    log = out["log"]
    wash = imgui.get_color_u32(theme.rgba(theme.ELEV_2, 0.5))
    surfaces = [a for kind, a in log if kind == "rect" and a[2] == wash]
    assert len(surfaces) >= 2, "rows are drawn as ELEV_2 at 0.5 surfaces"
    for _lo, _hi, _colour, rounding in surfaces:
        assert rounding == pytest.approx(sp(tokens.RADIUS_M), abs=0.01)
    loads = out["items"]["load"]
    assert len(loads) == len(rows)
    # Each row's settings text is the last text drawn at that row's left column.
    texts = [a for kind, a in log if kind == "text"]
    for (lo, hi, *_rest), load in zip(surfaces, loads, strict=False):
        inside = [t for t in texts if lo.y <= t[0][1] <= hi.y and t[0][0] > lo.x + sp(60)]
        assert inside, "the row drew its text"
        for pos, _colour, text in inside:
            size = imgui.calc_text_size(text)
            text_rect = (pos[0], pos[1], pos[0] + size.x, pos[1] + size.y)
            overlap = not (
                load[2] <= text_rect[0]
                or text_rect[2] <= load[0]
                or load[3] <= text_rect[1]
                or text_rect[3] <= load[1]
            )
            assert not overlap, f"Load settings {load} covers the text {text!r} {text_rect}"
        assert load[3] == pytest.approx(hi.y - sp(tokens.SP_3), abs=1.0), "12 px under the button"
        assert load[3] - load[1] == pytest.approx(sp(tokens.CONTROL_HEIGHT_COMPACT), abs=1.0)


def test_history_rows_are_exactly_the_derived_height_and_8_apart(frames, monkeypatch):
    rows = [_row(f"r{i}") for i in range(3)]
    out = _draw_history(frames, monkeypatch, rows, 300.0)
    loads = out["items"]["load"]
    metrics = out["metrics"]
    pitch = loads[1][1] - loads[0][1]
    assert pitch == pytest.approx(metrics.row_h + imgui.get_style().item_spacing.y, abs=0.5)
    assert imgui.get_style().item_spacing.y == pytest.approx(sp(tokens.SP_2), abs=0.01)
    expected = (
        2 * sp(tokens.SP_3)
        + 3 * metrics.line
        + 2 * sp(tokens.SP_1)
        + sp(tokens.SP_2)
        + sp(tokens.CONTROL_HEIGHT_COMPACT)
    )
    assert metrics.row_h == pytest.approx(expected, abs=0.01)
    assert history.RowMetrics.__doc__ and "derived" in history.RowMetrics.__doc__.lower()


# --- the files this landing touched stay clean --------------------------------------------


@pytest.mark.parametrize(
    "relative",
    ["studio/modes/create/ui/panes/history.py", "studio/panes/inspector.py"],
)
def test_no_raw_separator_in_the_right_column_files(relative):
    text = (SRC / relative.replace("studio/", "", 1)).read_text(encoding="utf-8")
    assert "imgui.separator(" not in text, relative


def test_history_hand_spells_no_button_role():
    """Compact ghost buttons come from ``controls.small_button``."""
    text = (SRC / "modes/create/ui/panes/history.py").read_text(encoding="utf-8")
    assert "role=" not in text
    assert "widgets.pane_header" not in text, "History's title is its card's header row"
    assert "_ROW_PAD_DP" not in text
