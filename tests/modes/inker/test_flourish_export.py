"""Exporting an effect: one sheet per phase through the per-tag export, and
the engine snippet that describes one of those files."""

from __future__ import annotations

import ast
import dataclasses
from types import SimpleNamespace

import pytest
from _ui_context import imgui_context

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.pixel import sheetout
from realmspinner.kernels.pixel.flourish import bake as B
from realmspinner.kernels.pixel.flourish import engines, presets
from realmspinner.studio import probe
from realmspinner.studio.modes.inker import flourish as inker_flourish
from realmspinner.studio.modes.inker import ops as inker_ops
from realmspinner.studio.modes.inker import state as inker_state
from realmspinner.studio.modes.inker.ui.panes import flourish as pane


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(inker=inker_state.InkerState(), manual=None)
        self.toasts: list = []
        self.legs: list = []
        self.tasks = SimpleNamespace(set_progress=lambda *a, **k: None)

    def toast(self, text, level="info", **_):
        self.toasts.append((text, level))

    def busy(self, key):
        return False

    def progress(self, key):
        return None

    def submit(self, key, fn, *a, **k):
        return True


def _scene(with_effect: bool = True):
    ctx = _Ctx()
    tab = inker_state.InkerDoc(doc=inker.Document.blank(32, 32), title="spell.ora")
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    if with_effect:
        rec = dataclasses.replace(presets.load("sword_impact"), width=32, height=32, supersample=2)
        tab.doc.insert_flourish(B.bake(rec))
    return ctx, tab


def test_export_is_greyed_without_an_effect_and_offered_with_one():
    ctx, tab = _scene(with_effect=False)
    op = inker_ops.get("flourish_export")
    assert not op.enabled(ctx.state.inker, tab)
    assert inker_ops.reason_for(op, ctx.state.inker, tab) == inker_flourish.NO_EFFECT
    ctx, tab = _scene()
    assert op.enabled(ctx.state.inker, tab)
    assert inker_ops.get("flourish_snippet").enabled(ctx.state.inker, tab)


def test_export_runs_the_per_tag_export_once(monkeypatch):
    ctx, tab = _scene()
    calls: list = []
    # Patched on ``inker_export``, which is where it lives since 2026-09-04
    # (T7): ``inker_mode`` serves the name through ``__getattr__``, and a
    # module-level ``setattr`` on a name that module does not define would be
    # shadowing rather than replacing what the caller reaches.
    from realmspinner.studio.modes.inker import export as inker_export

    monkeypatch.setattr(
        inker_export, "export_per_tag", lambda c, t, kind: calls.append((t, kind))
    )
    assert inker_ops.run(ctx, inker_ops.get("flourish_export"))
    assert calls == [(tab, "sheet")]


def test_the_snippet_describes_the_file_the_per_tag_export_writes():
    ctx, tab = _scene()
    anim = tab.doc.anim
    info = inker_flourish.snippet_info(tab, "sparks")
    tag = next(t for t in anim.tags if t.name == "sparks")
    first, last = sheetout.tag_span(anim, tag)
    stem = sheetout.filename_for(sheetout.DEFAULT_TAG_TEMPLATE, title="spell", tag="sparks")
    assert info["image"] == f"{stem}.png"
    assert info["frames"] == last - first + 1
    assert info["frame_width"] == 32 and info["frame_height"] == 32
    assert info["origin"] == [16, 16]
    assert info["loop"] is False
    assert info["fps"] == round(1000 / anim.frames[first].duration_ms)
    assert inker_flourish.snippet_info(tab, "no-such-tag") is None
    assert inker_flourish.snippet_text(tab, "no-such-tag", "godot") == ""


def test_the_snippet_text_is_the_engine_module_output():
    ctx, tab = _scene()
    for engine in engines.ENGINES:
        text = inker_flourish.snippet_text(tab, "hit", engine)
        assert text == engines.snippet(engine, inker_flourish.snippet_info(tab, "hit"))
    ast.parse(inker_flourish.snippet_text(tab, "hit", "pygame-ce"))


def test_a_quote_in_the_effect_name_does_not_break_the_godot_snippet():
    """The 2026-09-14 audit (inker-10): ``engines._godot`` spliced the effect
    name straight into GDScript double-quoted string literals with no
    escaping. Nothing stops a user naming an effect with a quote or a
    backslash in it, and either one used to close a literal early and leave
    the rest of the line as bare, unparseable GDScript."""
    info = engines.describe(
        name='Bob\'s "big" swing\\',
        image="sheet.png",
        frame_width=32,
        frame_height=32,
        frames=4,
        fps=12,
        loop=False,
        origin=(16, 16),
    )
    text = engines.snippet("godot", info)
    # Every animation name the script hands to Godot must be one closed,
    # backslash-escaped string -- not the raw name breaking the literal.
    assert '.add_animation("Bob\'s \\"big\\" swing\\\\")' in text
    assert '.set_animation_speed("Bob\'s \\"big\\" swing\\\\", 12)' in text
    assert '.play("Bob\'s \\"big\\" swing\\\\")' in text


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def test_the_snippet_popup_draws_its_controls(ui):
    ctx, tab = _scene()

    def draw():
        pane.open_snippet_popup(ctx, tab)
        pane.snippet_popup(ctx, tab)

    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    try:
        draw()
    finally:
        ui.end()
        ui.end_frame()
    labels = {c.label for c in probe.census()}
    assert any("Copy" in label for label in labels)
    assert any("Close" in label for label in labels)
    assert ctx.state.inker.flourish_snippet_tag == "hit"
