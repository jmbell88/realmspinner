"""The inspector's "Was this any good?" control, slice C of the 2026-10-11 Create
review: three buttons by default, the -5..+5 row behind a persisted switch.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.studio import controls as real_controls
from realmspinner.studio import settings as studio_settings
from realmspinner.studio.panes import inspector
from realmspinner.studio.state import AppState


class _Spy:
    """Records calls by name; answers from ``answers`` (a value or a callable)."""

    def __init__(self, answers: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.answers = answers or {}

    def __getattr__(self, name: str):
        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            answer = self.answers.get(name)
            return answer(*args, **kwargs) if callable(answer) else answer

        return call

    def args(self, name: str) -> list[tuple[Any, ...]]:
        return [a for n, a, _ in self.calls if n == name]


_JOB = {"id": "m1", "stage": "model", "status": "done", "files": ["model.glb"]}


def _drive(
    monkeypatch,
    tmp_path,
    *,
    full: bool | None = None,
    press: str | None = None,
    checkbox: tuple[bool, bool] | None = None,
    graded: bool = False,
):
    settings = studio_settings.Settings(tmp_path / studio_settings.FILENAME)
    if full is not None:
        settings.set(inspector.FULL_SCALE_SETTING, full)
    state = AppState(mode="create")
    state.inspector_graded["m1"] = graded
    ctx = SimpleNamespace(state=state, settings=settings)
    widgets = _Spy(
        {
            "header": True,
            "disabled_button": lambda label, *a, **k: bool(press) and label.startswith(press),
            "grade_buttons": None,
            "tag_toggles": None,
        }
    )
    widgets.grid_width = lambda n, **k: 80.0
    controls = _Spy({"checkbox": lambda label, value, **k: checkbox or (False, value)})
    gui = _Spy()
    filed: list[tuple[int, tuple[str, ...]]] = []
    controls.ButtonRole = real_controls.ButtonRole
    widgets.input_text = lambda label, value, **k: value
    monkeypatch.setattr(inspector, "widgets", widgets)
    monkeypatch.setattr(inspector, "controls", controls)
    monkeypatch.setattr(inspector, "imgui", gui)
    monkeypatch.setattr(
        inspector,
        "record_verdict",
        lambda c, jid, grade, tags=(), note=None: filed.append((grade, tags)),
    )
    inspector._verdict(ctx, _JOB)
    return ctx, widgets, controls, filed


def test_the_default_grade_control_is_three_buttons(monkeypatch, tmp_path):
    _ctx, widgets, _controls, _filed = _drive(monkeypatch, tmp_path)
    labels = [a[0].split("##")[0] for a in widgets.args("disabled_button")]
    assert labels == ["Ship it", "Usable", "Unusable"]
    assert widgets.args("grade_buttons") == [], "the eleven-button row is behind the switch"


@pytest.mark.parametrize(("label", "grade"), [("Ship it", 5), ("Usable", 3), ("Unusable", -5)])
def test_each_quick_button_files_the_grade_its_full_scale_twin_would(
    monkeypatch, tmp_path, label, grade
):
    _ctx, _widgets, _controls, filed = _drive(monkeypatch, tmp_path, press=label)
    assert filed == [(grade, ())]


def test_the_quick_grades_are_the_scale_values_the_review_documents():
    from realmspinner.studio import widgets as real_widgets

    assert [g for g, _l, _t in inspector.QUICK_GRADES] == [5, 3, -5]
    assert all(g in real_widgets.GRADES for g, _l, _t in inspector.QUICK_GRADES)


def test_the_full_scale_switch_shows_the_whole_row_and_carries_the_legend(monkeypatch, tmp_path):
    _ctx, widgets, _controls, _filed = _drive(monkeypatch, tmp_path, full=True)
    (call,) = [(a, k) for n, a, k in widgets.calls if n == "grade_buttons"]
    assert call[1]["anchors"] == {5: "Ships as it is.", 3: "Usable.", -5: "Unusable."}
    assert widgets.args("disabled_button") == []


def test_the_full_scale_switch_is_persisted_through_the_apps_settings(monkeypatch, tmp_path):
    ctx, _widgets, controls, _filed = _drive(monkeypatch, tmp_path, checkbox=(True, True))
    assert ctx.settings.get(inspector.FULL_SCALE_SETTING) is True
    label = controls.args("checkbox")[0][0]
    assert label.startswith("Full scale")
    # And it survives the file: a second Settings over the same directory reads it.
    ctx.settings.flush()
    again = studio_settings.Settings.load(tmp_path)
    assert again.get(inspector.FULL_SCALE_SETTING) is True


def test_the_header_opens_itself_for_three_buttons_but_not_for_eleven(monkeypatch, tmp_path):
    def default_open(**kw: Any) -> bool:
        _ctx, widgets, _c, _f = _drive(monkeypatch, tmp_path, **kw)
        return next(k for n, _a, k in widgets.calls if n == "header")["default_open"]

    assert default_open() is True, "an ungraded mesh opens onto three buttons"
    assert default_open(full=True) is False, "but not onto eleven"
    assert default_open(graded=True) is False, "and not once there is a verdict"


def test_the_verdict_section_speaks_no_sweep_or_findings_jargon():
    source = "\n".join(
        textwrap.dedent(inspect.getsource(fn))
        for fn in (inspector.quick_grade_buttons, inspector._verdict)
    )
    strings = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    # Docstrings are for maintainers; the strings a user reads are the rest.
    prose = [s for s in strings if "\n" not in s]
    joined = " ".join(prose).lower()
    for word in ("sweep", "findings"):
        assert word not in joined, word
    assert "Feeds the same findings a sweep does." not in source


def test_grade_buttons_put_a_legend_anchor_on_the_buttons_it_describes(monkeypatch):
    from realmspinner.studio import widgets

    tips: dict[str, str] = {}
    monkeypatch.setattr(
        widgets,
        "disabled_button",
        lambda label, enabled, size=(0, 0), **k: (
            tips.__setitem__(label, k.get("tooltip", "")) or False
        ),
    )
    monkeypatch.setattr(widgets, "grid_width", lambda *a, **k: 40.0)
    monkeypatch.setattr(widgets.imgui, "same_line", lambda *a, **k: None)
    widgets.grade_buttons("t", True, anchors={5: "Ships as it is."})
    five = next(v for k, v in tips.items() if k.startswith("+5"))
    assert five.startswith("Ships as it is.")
    assert next(v for k, v in tips.items() if k.startswith("+4")) == widgets.grade_key_hint(4)
