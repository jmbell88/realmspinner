"""Clay's Tools panel: one grid, one selection, one options block.

The 2026-09-08 panel-grammar pass replaced three affordances stacked in one
sidebar -- primitives as an unlabelled icon grid, figures as a column of
full-width text buttons, and the ops as a ragged two-column grid with a
hand-rolled Delete -- with one selection field every add-tool writes
(``state.generator``), one options block that reads it, and one width rule
(``widgets.grid_width``) for the ops grid. Each test's name is the claim it
makes about the *redesigned* panel, and each is checked below to fail against
the code as it stood before this pass (never with git -- a scratch copy with
the fix reverted by hand).
"""

from __future__ import annotations

import inspect

import pytest
from _ui_context import imgui_context

from warlock.kernels.mesh import document as bd
from warlock.studio import icons, probe, theme, tokens, widgets
from warlock.studio.modes.clay import mode as clay_mode
from warlock.studio.modes.clay import ops as clay_ops
from warlock.studio.modes.clay.ui.panes import tools as clay_tools


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


# --- the ops grid keeps one width, even at UI scale 1.5 ---------------------


def test_the_action_grid_uses_one_width_at_ui_scale_one_point_five(monkeypatch):
    """The measured failure class ``widgets.grid_width`` documents: an
    unscaled gap literal is right at 1.0x and short by 4.8px per gap at 1.5x.
    Before this pass every action button auto-sized to its own label
    (``widgets.disabled_button(..., enabled, reason=...)`` with no ``size``),
    so "Duplicate" and "Smooth" were never the same width to begin with --
    ragged at *any* scale, and asserted here at the one the incident is about.
    """
    with imgui_context(monkeypatch) as imgui:
        tokens.set_scale(1.5)
        theme.apply(imgui)
        try:
            ctx = FakeCtx()
            state = clay_mode.ensure(ctx)
            doc = bd.ClayDoc()
            expected = None

            def frame():
                nonlocal expected
                probe.begin_frame()
                imgui.new_frame()
                # A 300 dp sidebar *at this scale*, which is 450 physical px.
                # The window used to be 300 px wide regardless, i.e. a 200 dp
                # column at 1.5x -- narrower than any sidebar the app offers,
                # so the test was asserting about a layout nobody can produce.
                imgui.set_next_window_size((450.0, 900.0))
                imgui.begin("##host")
                # Measured *before* the ops draw, at the cursor they start
                # from: ``grid_width`` reads the remaining content region, so
                # asking afterwards asks about the space they did not use.
                labels = [
                    op.label.rstrip(".")
                    for op in clay_ops.menu(doc.element_mode)
                    if not op.name.startswith("select-") and op.name != "delete"
                ]
                expected = widgets.grid_width(widgets.grid_columns_for(labels, maximum=2))
                clay_tools._actions(ctx, state, doc)
                imgui.end()
                imgui.end_frame()
                return list(probe.FRAME_CONTROLS)

            controls_ = frame()
        finally:
            tokens.set_scale(1.0)
            theme.apply(imgui)

    action_buttons = [
        c for c in controls_ if c.kind == "button" and "##clayop" in c.label
    ]
    assert action_buttons, "no ops offered in object mode on an empty document"
    widths = {round(c.rect[2], 3) for c in action_buttons}
    assert len(widths) == 1, f"the action grid is ragged again: {widths}"
    assert abs(widths.pop() - expected) < 0.01


# --- the local destructive-button reimplementation is gone -------------------


def test_no_local_destructive_button_reimplementation_remains():
    """``widgets.destructive_button`` grew the ``reason`` keyword that forced
    ``clay_tools._destructive_button`` to exist (the 2026-09-08
    button-vocabulary pass); this file's own copy is dead weight once it does,
    and two implementations of "a destructive button with a disabled-reason
    tooltip" is exactly the kind of drift this codebase's own style rejects."""
    source = inspect.getsource(clay_tools)
    assert "_destructive_button" not in source
    assert "widgets.destructive_button(" in source


def test_delete_is_drawn_through_the_shared_destructive_button(monkeypatch):
    """Not just absent from the source -- actually reached, with a reason, on
    an empty document where Delete is refused."""
    with imgui_context(monkeypatch) as imgui:
        ctx = FakeCtx()
        state = clay_mode.ensure(ctx)
        doc = bd.ClayDoc()

        probe.begin_frame()
        imgui.new_frame()
        imgui.set_next_window_size((300.0, 900.0))
        imgui.begin("##host")
        clay_tools._actions(ctx, state, doc)
        imgui.end()
        imgui.end_frame()
        controls_ = list(probe.FRAME_CONTROLS)

    delete_row = next(c for c in controls_ if icons.TRASH in c.label)
    assert delete_row.enabled is False
    assert delete_row.reason


@pytest.mark.parametrize("window_w", (300.0, 240.0, 190.0))
def test_no_action_button_is_narrower_than_its_own_label(monkeypatch, window_w):
    """An even grid that is one character too narrow loses the end of a word.

    Making this grid even (``grid_width(2)``) fixed the raggedness and bought a
    clipped label with it: "Smooth (Catmull-Clark)..." was the longest string in
    the actions list by a wide margin, imgui drew it straight past its frame,
    and the child cut the closing bracket off. Two things answer that together
    and this pins both -- the column count comes from
    ``widgets.grid_columns_for`` so the grid fits what it holds, and the
    algorithm's name moved out of the label into the op's ``hint``, because the
    reader picks this op to round a shape rather than because it is
    Catmull-Clark.

    Parametrised over three widths, and the narrow ones are the point. Once the
    label was shortened, two columns fit a 300 dp sidebar again -- so a test at
    that width alone passes against a hard-coded ``grid_width(2)`` and proves
    nothing about the derivation. At 240 and 190 two columns cannot hold
    "Bake Transform", and the grid has to drop to one rather than clip; those
    are the widths a narrow sidebar and a 1.5x scale actually produce.
    """
    with imgui_context(monkeypatch) as imgui:
        ctx = FakeCtx()
        state = clay_mode.ensure(ctx)
        doc = bd.ClayDoc()
        probe.begin_frame()
        imgui.new_frame()
        imgui.set_next_window_size((window_w, 900.0))
        imgui.begin("##host")
        clay_tools._actions(ctx, state, doc)
        rows = [c for c in probe.FRAME_CONTROLS if c.kind == "button" and "##clayop" in c.label]
        # The census keeps the imgui id; only the part before ``##`` is drawn.
        seen = [(row.label.split("##", 1)[0], row.rect[2]) for row in rows]
        measured = {label: (frame, imgui.calc_text_size(label).x) for label, frame in seen}
        imgui.end()
        imgui.end_frame()

    assert measured, "no ops offered in object mode on an empty document"
    for label, (frame, text) in sorted(measured.items()):
        assert frame >= text, f"{label!r} is drawn {frame:.0f} px wide for {text:.0f} px of label"


def test_delete_never_shares_a_row_with_an_ordinary_action():
    """The one arrangement this grid has always refused.

    ``_actions``' own comment says a destructive button beside an ordinary one
    invites the wrong click of the two, and Delete is drawn after the loop for
    that reason. It is not enough on its own: the op count is odd, so the last
    op leaves its row open and Delete lands in the gap unless the loop declines
    to ``same_line`` after its final item. That is what the stride's second
    condition is for, and it is the half a reader would delete as redundant.
    """
    source = inspect.getsource(clay_tools._actions)
    stride = source.split("if (index + 1) % columns", 1)[1].split(":", 1)[0]
    assert "index + 1 < len(ops_here)" in stride, "Delete can land beside the last op"


# --- the op-params popup's Apply button greys with a reason, like its siblings


def test_the_op_params_apply_button_names_why_it_is_greyed_while_saving():
    """clay-13 (2026-09-08 audit, second run): every other saving-gated control in Clay
    (the tools-pane actions, the header's mode field, the context-menu rows)
    passes a ``reason=`` to ``disabled_button`` so a user who hovers a greyed
    control mid-save is told why; the parameterised-op popup's Apply button
    did not, breaking the pattern for the one dialog most likely to be open
    when a save starts (Bevel/Inset/Weld all park a value there)."""
    from warlock.studio.modes.clay.ui.panes import menu as clay_menu

    source = inspect.getsource(clay_menu.params_popup)
    start = source.index('f"Apply')
    # Bounded by the click handler the button guards, so the assertion cannot
    # pass by finding ``reason=`` somewhere later in the function instead of
    # in this call.
    end = source.index("clay_ops.run(ctx, tab.doc, op, **values)", start)
    call_region = source[start:end]
    assert "reason=" in call_region, "Apply greys with no reason while the tab is saving"
