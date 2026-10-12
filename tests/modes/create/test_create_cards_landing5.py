"""Rig, Pose and Export are cards too, and the closing sweep (Landing 5 of the
cards redesign, 2026-10-11).

Reference, Mesh, the band, the tray, the Inspector and History were cards by
Landing 4; the three remaining stage columns are drawn by panels the Library
shares (``stage_rig``, ``pose_panel``, ``sprite_panel``, ``inspector.downloads``),
so ``shell.frame._stage_pane`` opens the card scope around them and nothing in
those files was rewritten. "Working now" outside the tray is a header-less card
instead of bare lines over a divider. The settings pane is 8 px inside on every
stage. And what is *not* a card -- the one-per-screen empty-state heroes -- is
pinned as the exception it is.

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

from realmspinner.studio import layout, theme, tokens, widgets
from realmspinner.studio import textures as textures_mod
from realmspinner.studio.app_ctx import Ctx
from realmspinner.studio.modes.create.ui import workspace
from realmspinner.studio.panes import inspector, pose_panel, stage_rig
from realmspinner.studio.shell import frame as frame_mod
from realmspinner.studio.state import AppState
from realmspinner.studio.tokens import sp

SRC = Path(layout.__file__).resolve().parent


@pytest.fixture
def frames():
    """A bare imgui context; ``draw(build, size, repeat)`` runs ``build`` for
    ``repeat`` frames in one host window (a card is one frame late, so geometry
    is read on the last of a few)."""
    previous = imgui.get_current_context()
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)

    def draw(build: Any, size: tuple[float, float] = (400.0, 1400.0), repeat: int = 4) -> None:
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


class _FakeTexture:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.filter = None
        self.repeat_x = True
        self.repeat_y = True
        self.glo = 1

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


def _mesh_job(svc: Any, files: list[str], **params: Any) -> dict[str, Any]:
    job_id = svc.store.create(
        "text", "a hooded adventurer", {"seed": 5, **params}, stage="model", status="done"
    )
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    job = svc.store.get(job_id)
    job["files"] = files
    return job


def _ctx(svc: Any, job: dict[str, Any], *, stage: str, mode: str = "create", active: Any = None):
    state = AppState(mode=mode, selected=job["id"])
    state.create.stage = stage
    cache = SimpleNamespace(
        get=lambda job_id: job if job_id == job["id"] else None, jobs=[job], active=active
    )
    runtime = SimpleNamespace(config=svc.config, progress=lambda _job_id: None)
    return Ctx(
        svc=svc,
        runtime=runtime,
        state=state,
        cache=cache,
        tasks=_FakeTasks(),
        settings=SimpleNamespace(get=lambda *_a, **_kw: None, set=lambda *_a, **_kw: None),
        textures=textures_mod.ThumbnailCache(_FakeGL()),
        rig_templates=[],
        rig_default="",
        rigging_available=True,
        sheet_options={},
    )


class _Cards:
    """Every card drawn, with its title, id and (last-frame) rect."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.opened: list[tuple[str | None, str | None]] = []
        self.rects: dict[str, tuple[float, float, float, float]] = {}
        self.dividers = 0
        real_group = widgets.group
        real_divider = widgets.divider
        outer = self

        def group(*args: Any, **kwargs: Any):
            title = args[0] if args else kwargs.get("title")
            ident = kwargs.get("id") or title or "card"
            outer.opened.append((title, kwargs.get("id")))
            real = real_group(*args, **kwargs)

            class _Wrapped:
                def __enter__(self_inner) -> Any:
                    opened = real.__enter__()
                    pos, size = imgui.get_window_pos(), imgui.get_window_size()
                    outer.rects[str(ident)] = (pos.x, pos.y, pos.x + size.x, pos.y + size.y)
                    return opened

                def __exit__(self_inner, *exc: Any) -> Any:
                    return real.__exit__(*exc)

            return _Wrapped()

        def divider(*args: Any, **kwargs: Any) -> Any:
            outer.dividers += 1
            return real_divider(*args, **kwargs)

        monkeypatch.setattr(widgets, "group", group)
        monkeypatch.setattr(widgets, "divider", divider)

    def titles(self) -> list[str | None]:
        return [title for title, _ident in self.opened]

    def reset(self) -> None:
        self.opened.clear()
        self.rects.clear()
        self.dividers = 0


def _draw_stage(frames: Any, ctx: Any, cards: _Cards) -> None:
    def build() -> None:
        cards.reset()
        frame_mod._stage_pane(ctx)

    frames(build)


# --- the three stage columns are cards ---------------------------------------------------


def test_the_rig_stage_draws_its_skeleton_section_as_a_card(frames, svc, monkeypatch):
    """"Skeleton" was a bare heading; under the scope ``_stage_pane`` opens it is the
    header of a card, with no edit to ``stage_rig``."""
    cards = _Cards(monkeypatch)
    job = _mesh_job(svc, ["model.glb"])
    _draw_stage(frames, _ctx(svc, job, stage="rig"), cards)
    assert "Skeleton" in cards.titles(), cards.opened


def test_the_pose_stage_draws_pose_and_the_saved_list_as_cards(frames, svc, monkeypatch):
    cards = _Cards(monkeypatch)
    job = _mesh_job(svc, ["model.glb", "rig.glb"])
    _draw_stage(frames, _ctx(svc, job, stage="pose"), cards)
    assert cards.titles()[:2] == ["Pose", "Saved poses"], cards.opened


def test_the_export_stage_draws_downloads_as_a_card(frames, svc, monkeypatch):
    cards = _Cards(monkeypatch)
    job = _mesh_job(svc, ["model.glb"])
    _draw_stage(frames, _ctx(svc, job, stage="export"), cards)
    assert "Downloads" in cards.titles(), cards.opened


def test_a_sprite_sheet_export_draws_drafts_then_downloads_as_two_cards(
    frames, svc, monkeypatch
):
    cards = _Cards(monkeypatch)
    job = _mesh_job(svc, ["model.glb"], sprite_sheet=True)
    _draw_stage(frames, _ctx(svc, job, stage="export"), cards)
    assert cards.titles() == ["Drafts", "Downloads"], cards.opened


def test_the_empty_states_stay_bare(frames, svc, monkeypatch):
    """Pose and Export with nothing selected are one empty state, not a card."""
    cards = _Cards(monkeypatch)
    job = _mesh_job(svc, ["model.glb"])
    for stage in ("pose", "export"):
        ctx = _ctx(svc, job, stage=stage)
        ctx.state.selected = None
        ctx.cache.get = lambda _job_id: None
        _draw_stage(frames, ctx, cards)
        assert cards.opened == [], (stage, cards.opened)


def test_the_cards_of_a_stage_column_are_exactly_8_px_apart(frames, svc, monkeypatch):
    """``group_stack``'s gutter reaches the three stages: the progress card, the
    Pose card and the Saved poses card are 8 px apart."""
    cards = _Cards(monkeypatch)
    job = _mesh_job(svc, ["model.glb", "rig.glb"])
    active = {"id": "run-1", "status": "running", "name": "a goblin"}
    _draw_stage(frames, _ctx(svc, job, stage="pose", active=active), cards)
    ordered = [cards.rects[i] for i in ("progress", "Pose", "Saved poses")]
    gutter = sp(tokens.SP_2)
    for above, below in zip(ordered, ordered[1:], strict=False):
        assert below[1] - above[3] == pytest.approx(gutter, abs=1.0), ordered


# --- "Working now" outside the tray -------------------------------------------------------


def test_working_now_outside_the_tray_is_a_headerless_card_and_no_divider(
    frames, svc, monkeypatch
):
    """It was bare lines and a divider on Rig, Pose and Export: the one block in
    those columns that was not a card."""
    cards = _Cards(monkeypatch)
    job = _mesh_job(svc, ["model.glb"])
    active = {"id": "run-1", "status": "running", "name": "a goblin"}
    _draw_stage(frames, _ctx(svc, job, stage="rig", active=active), cards)
    assert (None, "progress") in cards.opened, cards.opened
    assert cards.dividers == 0
    assert "widgets.divider" not in inspect.getsource(frame_mod._stage_pane)


def test_working_now_inside_the_attempts_card_draws_no_card_of_its_own(
    frames, svc, monkeypatch
):
    cards = _Cards(monkeypatch)
    job = _mesh_job(svc, ["model.glb"])
    ctx = _ctx(svc, job, stage="mesh", active={"id": "run-1", "status": "queued", "name": "x"})

    def build() -> None:
        cards.reset()
        with widgets.group("Attempts", id="attempts"):
            assert workspace.progress_row(ctx) is True

    frames(build)
    assert [ident for _t, ident in cards.opened] == ["attempts"], cards.opened


def test_an_idle_stage_opens_no_progress_card(frames, svc, monkeypatch):
    cards = _Cards(monkeypatch)
    job = _mesh_job(svc, ["model.glb"])
    _draw_stage(frames, _ctx(svc, job, stage="rig"), cards)
    assert (None, "progress") not in cards.opened


# --- the pane, and the Library ------------------------------------------------------------


def _settings_pane_padding() -> ast.expr:
    tree = ast.parse(inspect.getsource(frame_mod))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "pane"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "settings"
        ):
            return next(k.value for k in node.keywords if k.arg == "padding")
    raise AssertionError("the settings pane is gone")


def test_the_settings_pane_is_8_px_inside_on_all_five_stages():
    """Reference/Mesh were SP_2 and the other three passed None (the default);
    one expression, with no stage in it, is SP_2 on every stage."""
    padding = _settings_pane_padding()
    assert ast.unparse(padding) == "tokens_mod.SP_2"
    assert tokens.SP_2 == 8


def test_the_librarys_rig_and_pose_panels_open_no_scope_and_draw_no_cards(
    frames, svc, monkeypatch
):
    """The same panels in the Library (its Inspector's Rig & Pose and Export
    tabs) keep their flat headings: only ``_stage_pane`` opens the scope."""
    cards = _Cards(monkeypatch)
    job = _mesh_job(svc, ["model.glb", "rig.glb"])
    ctx = _ctx(svc, job, stage="rig", mode="library")

    def build() -> None:
        cards.reset()
        stage_rig.draw(ctx)
        pose_panel.draw(ctx, job)
        inspector.downloads(ctx, job)

    frames(build)
    assert cards.opened == [], "a card leaked into the Library's Rig & Pose / Export"


# --- the sweep ---------------------------------------------------------------------------

#: One-per-screen empty states: the only thing on the canvas, not one group among
#: several. They keep the 20 px pane heading on purpose.
EMPTY_STATE_HEROES = {
    "Nothing here yet",
    "Ready when you are",
    "Set up Realmspinner",
    "Choose a reference to build from",
    "What would you like to make?",
}

CREATE_FILES = sorted((SRC / "modes/create/ui").rglob("*.py")) + [SRC / "shell/frame.py"]


def _pane_header_calls() -> list[tuple[Path, str, int]]:
    found = []
    for path in CREATE_FILES:
        text = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(text)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pane_header"
            ):
                arg = node.args[0] if node.args else None
                title = arg.value if isinstance(arg, ast.Constant) else ast.unparse(node)
                found.append((path, str(title), node.lineno))
    return found


def test_the_only_pane_headers_left_in_create_are_the_empty_state_heroes():
    titles = {title for _path, title, _line in _pane_header_calls()}
    assert titles == EMPTY_STATE_HEROES


def test_each_empty_state_hero_says_why_it_is_not_a_card():
    for path, title, line in _pane_header_calls():
        above = path.read_text(encoding="utf-8").splitlines()[line - 3 : line - 1]
        assert any("Empty-state hero" in text for text in above), (path.name, title)


@pytest.mark.parametrize("path", CREATE_FILES, ids=lambda p: p.name)
def test_no_old_grouping_idiom_is_left_in_create(path):
    text = path.read_text(encoding="utf-8")
    for idiom in ("section_blocks", "controls.collapsing_header", "imgui.separator("):
        assert idiom not in text, f"{path.name} still uses {idiom}"
