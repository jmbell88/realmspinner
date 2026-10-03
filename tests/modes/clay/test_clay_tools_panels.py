"""Clay's Tools panel: one grid, one selection, one options block.

The 2026-09-08 panel-grammar pass replaced three affordances stacked in one
sidebar -- primitives as an unlabelled icon grid, figures as a column of
full-width text buttons, and the ops as a ragged two-column grid with a
hand-rolled Delete -- with one selection field every add-tool writes
(``state.generator``) and one options block that reads it. The ops grid itself
left for the header's menu strip on 2026-10-02. Each test's name is the claim it
makes about the *redesigned* panel, and each is checked below to fail against
the code as it stood before this pass (never with git -- a scratch copy with
the fix reverted by hand).
"""

from __future__ import annotations

import inspect

from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.studio import icons, probe
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools


class FakeCtx:
    """Just enough of ``Ctx`` for ``clay_mode.ensure`` and a button's own
    click handler (``add_primitive``/``add_assembly`` only touch ``doc``)."""

    def __init__(self) -> None:
        self.state = _AppState()

    def toast(self, message: str, kind: str = "info") -> None:  # pragma: no cover
        pass


class _AppState:
    def __init__(self) -> None:
        self.clay = None


# --- the options block is pure, and follows the selection -------------------


def test_options_for_names_the_selected_primitive():
    """Before this pass there was no ``_options_for`` at all: the grid was one
    click and done, with nothing in the panel describing what it had just
    placed."""
    heading, rows, note = clay_tools._options_for("box")
    assert heading == "Box"
    assert ("size", "1.00, 1.00, 1.00") in rows
    assert note


def test_options_change_when_the_selected_tool_changes():
    """The redesign's whole claim: switch which tool is selected and the
    options block describes a different tool, not the same one restated."""
    box_heading, box_rows, _note = clay_tools._options_for("box")
    sphere_heading, sphere_rows, _note = clay_tools._options_for("uv_sphere")
    assert box_heading != sphere_heading
    assert box_rows != sphere_rows
    # And a generator's own fields, not a generic placeholder -- the sphere's
    # options name its own parameters, not the box's.
    sphere_labels = {label for label, _value in sphere_rows}
    assert "segments" in sphere_labels
    assert "size" not in sphere_labels


def test_options_for_a_figure_name_the_figure_rather_than_a_primitive():
    """A figure has no per-field defaults of its own -- ``presets.build``
    computes a whole rig template's worth of parts -- so the heading is the
    template's own label and the rows are empty rather than borrowing a
    primitive's."""
    heading, rows, note = clay_tools._options_for("humanoid")
    assert heading == "Humanoid (biped)"
    assert rows == ()
    assert "preset arrangement" in note


def test_options_for_an_unknown_tool_is_none():
    """A document opened from an older save whose remembered ``state.generator``
    names a generator since retired from the registry must not draw a stale
    heading with nothing behind it."""
    assert clay_tools._options_for("not-a-real-tool-any-more") is None


def test_a_fresh_clay_state_lights_no_add_tool_and_shows_no_options():
    """clay-12 (2026-09-08 audit, second run): ``ClayState.generator`` used to default to
    ``"box"``, so a brand-new session showed Box lit in the add grid and its
    defaults printed in the options block below it before the user had ever
    pressed an add-tool -- state that reads as a prior click nobody made. The
    redesign's own framing ("clicking one... marks it the tool in hand") only
    holds if a fresh state marks nothing."""
    state = clay_mode.ensure(FakeCtx())
    assert state.generator == "", "a fresh session must not arrive with a tool already in hand"
    # And the options block agrees: the empty sentinel names neither a
    # generator nor a figure, so it draws nothing rather than a stale Box.
    assert clay_tools._options_for(state.generator) is None


# --- primitives and figures share one affordance, not two -------------------


def test_a_figure_button_writes_the_same_field_a_primitive_button_does():
    """Before this pass ``_add`` took ``state`` only to immediately discard it
    (``del state``), and ``_figures`` never touched ``state`` at all -- so a
    figure and a primitive were two disconnected mechanisms with no shared
    idea of "the tool in hand". Checked at the source because the claim is
    that both write one field, which nothing on screen shows by itself."""
    add_source = inspect.getsource(clay_tools._add)
    figures_source = inspect.getsource(clay_tools._figures)
    assert "del state" not in add_source
    assert "state.generator = clicked" in add_source
    assert "state.generator = key" in figures_source


def test_clicking_the_box_button_selects_it_and_places_it(monkeypatch):
    """Driven for real through a synthetic click, the way
    ``test_context_controls`` presses controls -- a control wired to nothing
    passes every test that calls the setter directly."""
    with imgui_context(monkeypatch) as imgui:
        ctx = FakeCtx()
        state = clay_mode.ensure(ctx)
        state.generator = "cylinder"
        doc = bd.ClayDoc()
        box_label = f"{icons.BOX}##addbox"

        def frame(pos=(-100.0, -100.0), down=False):
            io = imgui.get_io()
            io.add_mouse_pos_event(pos[0], pos[1])
            io.add_mouse_button_event(0, down)
            probe.begin_frame()
            imgui.new_frame()
            imgui.set_next_window_size((320.0, 900.0))
            imgui.begin("##host")
            clay_tools._add(ctx, state, doc)
            imgui.end()
            imgui.end_frame()
            return list(probe.FRAME_CONTROLS)

        found = [c for c in frame() if c.label == box_label]
        assert found, "no Box button drawn -- the grid changed shape"
        cx, cy = found[0].centre

        # imgui's default button fires on release-while-hovered, and hover
        # itself is only current once a frame has already put the mouse over
        # the item -- so the down/up pair needs a frame of hover in front of
        # it, the same three-frame shape ``test_context_controls`` presses a
        # real button with.
        frame((cx, cy), down=False)
        frame((cx, cy), down=True)
        frame((cx, cy), down=False)

        assert state.generator == "box"
        assert len(doc.objects) == 1
        assert doc.objects[0].generator == "box"


# --- the operations left this pane for the menu strip ------------------------


def test_the_add_palette_draws_no_operation_buttons(monkeypatch):
    """The ~50-button op grid is gone from the left pane: every operation is a
    row in the header's menu strip and the right-click menu now, grouped by
    ``menutree``. A pane that drew them again would be a second list of what
    Clay can do -- the thing the registry exists to prevent."""
    with imgui_context(monkeypatch) as imgui:
        ctx = FakeCtx()
        state = clay_mode.ensure(ctx)
        doc = bd.ClayDoc()

        def frame():
            probe.begin_frame()
            imgui.new_frame()
            imgui.set_next_window_size((450.0, 900.0))
            imgui.begin("##host")
            clay_tools._add(ctx, state, doc)
            imgui.end()
            imgui.end_frame()
            return list(probe.FRAME_CONTROLS)

        controls_ = frame()
    labels = {c.text for c in controls_}
    assert not any("##clayop" in c.label for c in controls_)
    assert not labels & {op.label.rstrip(".") for op in clay_ops.OPS}, (
        "an op is drawn in the Add palette"
    )


def test_the_tools_pane_no_longer_carries_the_action_grid_or_its_popup_call():
    source = inspect.getsource(clay_tools)
    assert "_actions" not in source
    assert "params_popup" not in source, (
        "the viewport owns the op dialog; a second caller steals open_op_popup"
    )
    assert 'widgets.section("Add")' in source


# --- the op-params popup's Apply button greys with a reason, like its siblings


def test_the_op_params_apply_button_names_why_it_is_greyed_while_saving():
    """clay-13 (2026-09-08 audit, second run): every other saving-gated control in Clay
    (the tools-pane actions, the header's mode field, the context-menu rows)
    passes a ``reason=`` to ``disabled_button`` so a user who hovers a greyed
    control mid-save is told why; the parameterised-op popup's Apply button
    did not, breaking the pattern for the one dialog most likely to be open
    when a save starts (Bevel/Inset/Weld all park a value there)."""
    from realmspinner.studio.modes.clay.ui.panes import menu as clay_menu

    source = inspect.getsource(clay_menu.params_popup)
    start = source.index('f"Apply')
    # Bounded by the click handler the button guards, so the assertion cannot
    # pass by finding ``reason=`` somewhere later in the function instead of
    # in this call.
    end = source.index("clay_ops.run(ctx, tab.doc, op, **values)", start)
    call_region = source[start:end]
    assert "reason=" in call_region, "Apply greys with no reason while the tab is saving"
