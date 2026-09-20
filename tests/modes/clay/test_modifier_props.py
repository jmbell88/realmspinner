"""The Properties panel's Modifiers section (Clay tranche 2).

Three things this pins, mirroring how the rest of this file's siblings pin
the generator section it sits beside:

* the bidirectional registry gate -- the "Add modifier" combo lists exactly
  ``kernels.mesh.modifiers.MODIFIERS``, in its own order, and a kind
  registered there reaches the combo with no edit here at all
  (``test_clay_props_widget.py``'s own generator-registry gate, one layer
  over);
* per-keystroke param edits fold into one undo step through the same
  ``controls.fold_undo`` mechanism the generator loop already uses, checked
  by source inspection the way ``test_clay_props_undo.py`` checks the
  generator and transform fields, for the reason that module's own docstring
  states (one active item per headless frame, so a live frame cannot tell one
  field's activation from another's);
* a refusal from ``ClayDoc.set_modifiers`` (a boolean cycle) becomes a toast
  through the same ``clay_ops.toast`` path every other refusal in this mode
  already uses, not an unhandled exception on the frame thread.

The row buttons (enabled, reorder, apply, remove) and the evaluated-mesh
reads (dimensions, the frozen line, the manifold label) get one live,
headless-imgui test each, in the same shape ``test_clay_props_widget.py``
and ``test_clay_props_regen.py`` already use for the generator section: a
real (GL-less) imgui frame with one control's own function monkeypatched to
report a definite answer, so the test presses the control without asking
the harness to reproduce a mouse.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _ui_context import imgui_context

from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import modifiers as mods
from warlock.kernels.mesh import primitives as bp
from warlock.studio import widgets as widgets_mod
from warlock.studio.modes.clay.ui.panes import props as clay_props


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` and this package's other
    props tests for why this is not a conftest fixture."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui


class _Toasts:
    """``test_clay_ops.py``'s own double, verbatim: only what ``Ctx`` really
    offers (``ctx.toast``), not the ``ctx.toasts.error`` shape a stale double
    once carried while every refusal it saw was raised, caught and thrown
    away in the real app."""

    def __init__(self) -> None:
        self.errors: list[str] = []


class _Ctx:
    def __init__(self) -> None:
        self.toasts = _Toasts()

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.toasts.errors.append(message)


def _doc_with_one_modifier() -> tuple[bd.ClayDoc, bd.Obj]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    doc.set_modifiers(obj.uid, (mods.make("mirror", id=1),))
    return doc, doc.by_uid(obj.uid)


def _doc_with_two_modifiers() -> tuple[bd.ClayDoc, bd.Obj]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    stack = (mods.make("mirror", id=1), mods.make("weld", id=2))
    doc.set_modifiers(obj.uid, stack)
    return doc, doc.by_uid(obj.uid)


# --- the bidirectional registry gate -----------------------------------------


def test_the_add_modifier_combo_lists_every_registered_kind_in_order() -> None:
    expected = [(kind, kind_def.label) for kind, kind_def in mods.MODIFIERS.items()]
    assert clay_props.modifier_kind_options() == expected


def test_a_new_modifier_kind_reaches_the_add_combo_with_no_edit_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claim that makes the gate above a gate rather than a snapshot --
    ``test_agent_clay.py``'s own generator-registry test, the same shape,
    one layer over. ``monkeypatch.setitem`` on the real registry, restored
    automatically."""
    fake = mods.ModifierKind(
        name="fake_kind",
        label="Fake Kind",
        hint="",
        params=(),
        apply=lambda mesh, params, ctx: mesh,
    )
    monkeypatch.setitem(mods.MODIFIERS, "fake_kind", fake)
    assert ("fake_kind", "Fake Kind") in clay_props.modifier_kind_options()


def test_the_add_combo_offers_nothing_the_registry_does_not() -> None:
    """The other half of "bidirectional": every option the combo would draw
    names a real registered kind."""
    registered = set(mods.MODIFIERS)
    for kind, _label in clay_props.modifier_kind_options():
        assert kind in registered


# --- per-keystroke folding ----------------------------------------------------


def test_editing_a_modifier_param_by_keystroke_is_one_undo_step() -> None:
    """The 2026-09-07 audit's clay-01 fold, reused rather than re-invented:
    a modifier param field fires on every keystroke/drag frame exactly like a
    generator field does, and ``controls.fold_undo`` has to run between the
    widget draw and the write that would otherwise push once per frame.
    """
    import inspect

    source = inspect.getsource(clay_props._modifier_row)
    after = source.split("_mod_param_widget(p,", 1)[1]
    assert "controls.fold_undo(" in after, "no fold_undo(...) call after the param widget"
    fold_at = after.index("controls.fold_undo(")
    write_at = after.index("_set_modifier_stack(")
    assert fold_at < write_at, "_set_modifier_stack runs before the fold"


# --- refusal -> toast, not a crash -------------------------------------------


def test_a_cycle_refusal_from_set_modifiers_becomes_a_toast_not_a_crash() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    b = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    doc.set_modifiers(b.uid, (mods.make("boolean", {"target": a.uid}, id=1),))
    ctx = _Ctx()
    depth = len(doc.history)

    cyclical = (mods.make("boolean", {"target": b.uid}, id=1),)
    clay_props._set_modifier_stack(ctx, doc, a, cyclical)

    assert len(doc.history) == depth, "a refusal pushes nothing"
    assert ctx.toasts.errors, "the refusal must reach the same toast path as every other op"
    assert "cycle" in ctx.toasts.errors[0].lower()
    assert doc.by_uid(a.uid).modifiers == (), "the refused stack was never written"


# --- the row's buttons, each one press --------------------------------------


def test_toggling_the_enabled_checkbox_is_one_undo_step(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc, obj = _doc_with_one_modifier()
    monkeypatch.setattr(
        clay_props.controls, "checkbox", lambda label, value, **kw: (True, not value)
    )
    depth = len(doc.history)

    ui.new_frame()
    ui.begin("##host")
    clay_props._modifier_row(_Ctx(), doc, obj, obj.modifiers[0], 0, 1, None)
    ui.end()
    ui.end_frame()

    updated = doc.by_uid(obj.uid)
    assert updated.modifiers[0].enabled is False
    assert len(doc.history) == depth + 1, "one click, one undo step"


def test_moving_a_modifier_down_reorders_the_stack_as_one_step(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc, obj = _doc_with_two_modifiers()
    first_kind, second_kind = (m.kind for m in obj.modifiers)
    monkeypatch.setattr(
        clay_props.controls, "small_button", lambda label, **kw: "##down" in label
    )
    depth = len(doc.history)

    ui.new_frame()
    ui.begin("##host")
    clay_props._modifier_row(_Ctx(), doc, obj, obj.modifiers[0], 0, 2, None)
    ui.end()
    ui.end_frame()

    updated = doc.by_uid(obj.uid)
    assert [m.kind for m in updated.modifiers] == [second_kind, first_kind]
    assert len(doc.history) == depth + 1


def test_removing_a_modifier_drops_it_as_one_step(monkeypatch: pytest.MonkeyPatch, ui) -> None:
    doc, obj = _doc_with_two_modifiers()
    kept_kind = obj.modifiers[1].kind
    monkeypatch.setattr(
        clay_props.controls, "small_button", lambda label, **kw: "##remove" in label
    )
    depth = len(doc.history)

    ui.new_frame()
    ui.begin("##host")
    clay_props._modifier_row(_Ctx(), doc, obj, obj.modifiers[0], 0, 2, None)
    ui.end()
    ui.end_frame()

    updated = doc.by_uid(obj.uid)
    assert [m.kind for m in updated.modifiers] == [kept_kind]
    assert len(doc.history) == depth + 1


def test_the_apply_button_bakes_the_prefix_through_that_modifier(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc, obj = _doc_with_two_modifiers()
    first, second = obj.modifiers
    monkeypatch.setattr(
        clay_props.controls, "small_button", lambda label, **kw: "##apply" in label
    )
    depth = len(doc.history)

    ui.new_frame()
    ui.begin("##host")
    clay_props._modifier_row(_Ctx(), doc, obj, first, 0, 2, None)
    ui.end()
    ui.end_frame()

    updated = doc.by_uid(obj.uid)
    assert [m.kind for m in updated.modifiers] == [second.kind], "the first modifier was baked away"
    assert len(doc.history) == depth + 1, "the bake is one step, not two"


def test_the_add_modifier_combo_appends_a_new_modifier_with_default_params(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    monkeypatch.setattr(clay_props.widgets, "labeled_combo", lambda *a, **kw: "mirror")

    ui.new_frame()
    ui.begin("##host")
    clay_props._add_modifier_row(_Ctx(), doc, doc.by_uid(obj.uid))
    ui.end()
    ui.end_frame()

    updated = doc.by_uid(obj.uid)
    assert [m.kind for m in updated.modifiers] == ["mirror"]
    assert updated.modifiers[0].enabled is True


# --- evaluated-mesh reads ----------------------------------------------------


def test_the_dimensions_row_measures_off_the_evaluated_mesh(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    from warlock.kernels.mesh import ops as bops

    doc, obj = _doc_with_one_modifier()
    captured: list[object] = []
    real = bops.world_box

    def spy(o: object, mesh: object = None, world: object = None) -> object:
        captured.append(mesh)
        return real(o, mesh, world=world)

    monkeypatch.setattr(bops, "world_box", spy)

    ui.new_frame()
    ui.begin("##host")
    clay_props._dimensions(doc, doc.by_uid(obj.uid))
    ui.end()
    ui.end_frame()

    assert captured, "world_box was never called"
    assert captured[0] is doc.evaluated(obj.uid), "must measure the evaluated mesh, not the base"


def test_the_frozen_line_reports_base_and_evaluated_counts_when_a_stack_exists(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc, obj = _doc_with_one_modifier()
    lines: list[str] = []
    monkeypatch.setattr(widgets_mod, "muted", lambda text: lines.append(text))

    ui.new_frame()
    ui.begin("##host")
    clay_props._generator(doc, doc.by_uid(obj.uid))
    ui.end()
    ui.end_frame()

    assert lines, "the frozen line was not drawn"
    line = lines[0]
    assert "base" in line
    assert "evaluated" in line
    evaluated = doc.evaluated(obj.uid)
    assert str(len(evaluated.positions)) in line


def test_the_manifold_check_label_says_base_mesh_when_a_stack_exists(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc, obj = _doc_with_one_modifier()
    state = SimpleNamespace(manifold={})
    labels: list[str] = []
    monkeypatch.setattr(
        widgets_mod, "field_label", lambda label, help_text=None: labels.append(label)
    )

    ui.new_frame()
    ui.begin("##host")
    clay_props._diagnostics(state, doc, doc.by_uid(obj.uid))
    ui.end()
    ui.end_frame()

    assert labels and "base mesh" in labels[0]


def test_the_manifold_check_label_stays_plain_with_no_stack(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    state = SimpleNamespace(manifold={})
    labels: list[str] = []
    monkeypatch.setattr(
        widgets_mod, "field_label", lambda label, help_text=None: labels.append(label)
    )

    ui.new_frame()
    ui.begin("##host")
    clay_props._diagnostics(state, doc, obj)
    ui.end()
    ui.end_frame()

    assert labels == ["mesh check"]
