"""The 2026-10-14 audit, fixer F56: the shell findings that live in ``studio/``.

shell-home-library-02 (palette trash), shell-chrome-04 (rail vs dock),
shell-documents-08 (recents), shell-widgets-04 (settings log), shell-chrome-11
(status bar), create-viewer-02 (click vs drag), shell-widgets-01/02/06
(widgets).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from _ui_context import imgui_context

from realmspinner.studio import layout as layout_mod
from realmspinner.studio import palette, rail, recents, status_bar, tokens, widgets
from realmspinner.studio import settings as settings_mod
from realmspinner.studio.state import AppState


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


# --- shell-home-library-02 ---------------------------------------------------------


def test_showing_the_trash_from_the_palette_clears_the_tick_set():
    """Both Library toggles clear ``state.checked`` because a bulk Delete armed in the
    library would mean something else in the trash. The palette's own door did not."""
    state = AppState()
    state.checked.update({"live-1", "live-2"})
    ctx = SimpleNamespace(state=state, cache=SimpleNamespace(jobs=[], get=lambda _i: None))
    command = next(c for c in palette.commands(ctx) if c.key == "show-trash")

    command.run(ctx)

    assert state.filters.trash is True
    assert not state.checked

    state.checked.add("trashed")
    command.run(ctx)  # and back out again: the same rule both ways
    assert state.filters.trash is False
    assert not state.checked


# --- shell-chrome-04 ---------------------------------------------------------------


class _Style:
    def __init__(self) -> None:
        self.item_spacing = SimpleNamespace(x=8.0)
        self.window_padding = SimpleNamespace(x=8.0)


class _Imgui:
    def __init__(self, width: float) -> None:
        self._width = width

    def get_current_context(self) -> object:
        return object()

    def get_style(self) -> _Style:
        return _Style()

    def get_main_viewport(self) -> SimpleNamespace:
        return SimpleNamespace(work_size=(self._width, 0.0))


def test_the_labelled_rail_collapses_when_an_open_dock_would_leave_the_centre_under_its_floor(
    monkeypatch,
):
    """At the 1100 px minimum window with the Familiar dock open the labelled rail was
    judged to fit and left a ~200 px centre; the icon rail leaves ``CENTRE_MIN``."""
    monkeypatch.setattr(rail, "imgui", _Imgui(1100.0))

    monkeypatch.setattr(layout_mod, "FAMILIAR_OPEN_T", 0.0)
    assert rail.expanded_fits() is True, "the closed dock still fits the labels"

    monkeypatch.setattr(layout_mod, "FAMILIAR_OPEN_T", 1.0)
    assert rail.expanded_fits() is False

    # And the rail is not collapsed for nothing: a roomier window keeps the labels.
    monkeypatch.setattr(rail, "imgui", _Imgui(1400.0))
    assert rail.expanded_fits() is True

    # The decision is the one ``proportions`` makes: with the wide rail the centre
    # keeps its floor exactly when the labels are kept.
    for width in (1100.0, 1200.0, 1300.0):
        monkeypatch.setattr(rail, "imgui", _Imgui(width))
        _r, _l, centre, _rt, _d = layout_mod.proportions(
            width - 16.0, 1.0, 8.0, rail=tokens.sp(rail.RAIL_EXPANDED_W)
        )
        assert rail.expanded_fits() is (centre >= tokens.sp(layout_mod.CENTRE_MIN) - 0.5)


# --- shell-documents-08 ------------------------------------------------------------


@pytest.mark.parametrize(
    "when", [float("-inf"), float("inf"), float("nan"), True, False, 10**400, "5", None]
)
def test_a_non_finite_recent_timestamp_reads_as_unstamped(when):
    rows = [{"kind": "inker", "path": "a.rpx", "when": when}]
    (entry,) = recents._decode(rows)
    assert entry.when is None
    # What the Resume row then does with it must not raise.
    assert widgets.ago(entry.when) == ""


def test_a_finite_recent_timestamp_is_kept():
    (entry,) = recents._decode([{"kind": "inker", "path": "a.rpx", "when": 12}])
    assert entry.when == 12.0


# --- shell-widgets-04 --------------------------------------------------------------


def test_a_persistent_save_failure_logs_one_traceback_not_one_per_tick(
    tmp_path, monkeypatch, caplog
):
    store = settings_mod.Settings(tmp_path / "studio_settings.json")
    store.set("probe", 1)

    def refuse(*_args, **_kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(settings_mod.atomic, "staged", refuse)
    with caplog.at_level(logging.DEBUG, logger=settings_mod.log.name):
        for _ in range(5):
            assert store.flush() is False
    tracebacks = [r for r in caplog.records if r.exc_info]
    assert len(tracebacks) == 1
    assert store.take_notice() is not None

    # Recovered, then failing again is a new streak and is logged in full again.
    monkeypatch.undo()
    assert store.flush() is True
    monkeypatch.setattr(settings_mod.atomic, "staged", refuse)
    store.set("probe", 2)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=settings_mod.log.name):
        store.flush()
    assert len([r for r in caplog.records if r.exc_info]) == 1


# --- shell-chrome-11 ---------------------------------------------------------------


def test_a_mode_whose_tool_label_raises_is_logged_not_silently_dropped_from_the_status_group(
    monkeypatch, caplog
):
    from realmspinner.studio import mode_manifest

    module = SimpleNamespace(
        active=lambda _ctx: SimpleNamespace(title="Doc", dirty=False, view=None),
        tool_label=lambda _ctx: (_ for _ in ()).throw(AttributeError("broken tool_label")),
    )
    monkeypatch.setattr(mode_manifest, "module_of", lambda _entry: module)
    monkeypatch.setattr(status_bar, "_FAILED", set())
    ctx = SimpleNamespace(state=SimpleNamespace(mode="inker"))

    with caplog.at_level(logging.WARNING, logger=status_bar.log.name):
        status_bar.items(ctx)
        status_bar.items(ctx)  # a frame later: said once, not every frame

    records = [r for r in caplog.records if r.exc_info]
    assert len(records) == 1
    assert "broken tool_label" in str(records[0].exc_info[1])


def test_a_mode_with_no_active_probe_is_not_a_failure(monkeypatch, caplog):
    from realmspinner.studio import mode_manifest

    monkeypatch.setattr(mode_manifest, "module_of", lambda _entry: SimpleNamespace())
    monkeypatch.setattr(status_bar, "_FAILED", set())
    ctx = SimpleNamespace(state=SimpleNamespace(mode="inker"))
    with caplog.at_level(logging.WARNING, logger=status_bar.log.name):
        out = status_bar.items(ctx)
    assert [i.key for i in out][:1] == ["workspace"]
    assert not caplog.records


# --- create-viewer-02 --------------------------------------------------------------


def test_a_slow_orbit_drag_of_many_small_motions_does_not_deselect_on_release():
    """Click versus drag was decided per motion event, so a drag whose every event moved
    under 2 px never left the click state and cleared the selected joint on release."""
    from realmspinner.studio import viewer_embed

    viewer = viewer_embed.Viewer.__new__(viewer_embed.Viewer)
    viewer.pose_mode = False
    viewer._grab = "orbit"
    viewer._deselect_on_click = True
    viewer._render_dirty = False
    viewer._rect = (0.0, 0.0, 100.0, 100.0)
    viewer.camera = SimpleNamespace(orbit=lambda *_a: None)
    viewer._press_at = None
    viewer._last_mouse = (50.0, 50.0)
    viewer._press_at = (50.0, 50.0)

    # Forty events of 0.5 px: each far under the slop, twenty px from the press by the end.
    x = 50.0
    for _ in range(40):
        x += 0.5
        viewer._motion((x, 50.0))

    assert viewer._deselect_on_click is False

    # A tap that wobbles within the slop is still a click.
    viewer._deselect_on_click = True
    viewer._press_at = (50.0, 50.0)
    viewer._last_mouse = (50.0, 50.0)
    for step in (0.4, -0.3, 0.5, -0.2):
        viewer._motion((50.0 + step, 50.0))
    assert viewer._deselect_on_click is True


# --- shell-widgets-01 --------------------------------------------------------------


def _frame(imgui, label, *, persist_key, default_open=True):
    imgui.new_frame()
    imgui.set_next_window_pos((0, 0))
    imgui.set_next_window_size((600, 300))
    imgui.begin("host", None, imgui.WindowFlags_.no_title_bar.value)
    opened = widgets.header(label, default_open=default_open, persist_key=persist_key)
    imgui.end()
    imgui.render()
    return opened


def test_reset_collapsed_sections_reopens_a_section_the_user_collapsed_this_session(
    ui, tmp_path, monkeypatch
):
    store = settings_mod.Settings(tmp_path / "studio_settings.json")
    monkeypatch.setattr(widgets, "_SETTINGS", store)
    monkeypatch.setattr(widgets, "_RESET_SEEN", {})
    monkeypatch.setattr(widgets, "_RESET_GENERATION", 0)
    store.set("panels_open", {"probe/section": False})

    for _ in range(3):
        assert _frame(ui, "Section", persist_key="probe/section") is False

    # What the Settings button does.
    widgets.reset_sections(store)

    opened = [_frame(ui, "Section", persist_key="probe/section") for _ in range(3)]
    assert opened == [True, True, True]
    assert store.get("panels_open") == {"probe/section": True}

    # And it is one-shot: the generation is taken, so nothing forces the next draw.
    assert widgets._RESET_SEEN["probe/section"] == widgets._RESET_GENERATION


# --- shell-widgets-02 --------------------------------------------------------------


def test_centred_text_default_colour_follows_the_theme_in_force(ui, monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(widgets, "text_colored", lambda colour, _text: seen.append(colour))
    monkeypatch.setattr(widgets, "_centre_cursor", lambda _w: None)

    marker = 0xFF123456
    monkeypatch.setattr(widgets.theme, "MUTED", marker, raising=False)
    # ``theme`` resolves colours live through ``__getattr__``; a patched module attribute
    # is what a theme switch looks like to a reader.
    ui.new_frame()
    ui.begin("host")
    widgets.centred_text("hint")
    assert seen == [marker]

    widgets.centred_text("explicit", 0xFF000001)
    assert seen[-1] == 0xFF000001
    ui.end()
    ui.end_frame()


# --- shell-widgets-06 --------------------------------------------------------------


def test_recent_files_gap_scales_with_the_ui_scale(ui, monkeypatch):
    gaps: list[float] = []
    real_dummy = widgets.imgui.dummy

    def record(size):
        gaps.append(size[1])
        return real_dummy(size)

    monkeypatch.setattr(widgets.imgui, "dummy", record)
    monkeypatch.setattr(widgets, "section", lambda _label: None)
    monkeypatch.setattr(tokens, "SCALE", 1.5)

    ui.new_frame()
    ui.begin("host")
    widgets.recent_files(["a.rpx"], lambda _p: None)
    ui.end()
    ui.end_frame()

    assert gaps and gaps[0] == pytest.approx(tokens.SP_2 * 1.5)


# --- create-viewer-03 --------------------------------------------------------------
#
# After Poser's removal a large part of the Viewer's pose and strip surface had no
# production caller. The deletion landed in the same audit's cleanup round (C3), so
# this is now a strict guard: a method of the Viewer, its pose mixin or its pose
# editor with no production reference fails it, and the way to pass is to delete the
# method with its tests rather than to list it anywhere.

_VIEWER_UNITS = {
    "studio/viewer_embed.py": "Viewer",
    "studio/_viewer_pose.py": "PoseOps",
    "studio/viewer/pose.py": "PoseEditor",
}


def _methods_without_a_production_caller() -> set[str]:
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src" / "realmspinner"
    # Callers are all in the shell; the rest of the package imports nothing from it.
    defined: dict[tuple[str, str], str] = {}  # (file, method) -> "Class.method"
    scopes: dict[str, dict[str, set[str]]] = {}  # file -> enclosing "Class.method" -> names read

    def _read(path: Path):
        return ast.parse(path.read_text(encoding="utf-8"))

    for path in (root / "studio").rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        tree = _read(path)
        owner = _VIEWER_UNITS.get(rel)
        by_scope: dict[str, set[str]] = {}

        def walk(node, scope, cls=None, rel=rel, owner=owner, by_scope=by_scope):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    walk(child, scope, child.name)
                elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    here = scope
                    if cls == owner and owner is not None and scope == "":
                        here = f"{cls}.{child.name}"
                        defined[(rel, child.name)] = here
                    walk(child, here, cls)
                else:
                    if isinstance(child, ast.Attribute):
                        by_scope.setdefault(scope, set()).add(child.attr)
                    elif isinstance(child, ast.Name):
                        by_scope.setdefault(scope, set()).add(child.id)
                    walk(child, scope, cls)

        walk(tree, "")
        scopes[rel] = by_scope

    units = set(defined.values())
    by_method: dict[str, list[str]] = {}
    for unit in units:
        by_method.setdefault(unit.split(".", 1)[1], []).append(unit)
    live: set[str] = set()
    pending: list[str] = []

    def reach(names: set[str]) -> None:
        for name in names:
            for unit in by_method.get(name, ()):
                if unit not in live:
                    live.add(unit)
                    pending.append(unit)

    for by_scope in scopes.values():
        for scope, names in by_scope.items():
            if scope not in units:
                reach(names)
    while pending:
        unit = pending.pop()
        for by_scope in scopes.values():
            if unit in by_scope:
                reach(by_scope[unit])
    return {u for u in units if u not in live and not u.split(".", 1)[1].startswith("__")}


def test_every_public_viewer_method_has_a_production_caller():
    dead = _methods_without_a_production_caller()
    assert not dead, (
        "These Viewer / PoseOps / PoseEditor methods have no production caller; delete "
        f"them with their tests: {sorted(dead)}"
    )
