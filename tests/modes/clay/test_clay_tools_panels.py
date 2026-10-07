"""Clay's shapes: the registry, the names, and the flyout the rail's ``+`` opens.

The 2026-09-08 panel-grammar pass gave every add-tool one selection field
(``state.generator``). The ops grid left for the header's menu strip on
2026-10-02, and the sidebar's icon grid left for the tool rail's flyout on
2026-10-07: the same entries, listed by name instead of as a grid of
near-identical silhouettes. Each test's name is the claim it makes, and each is
checked to fail against the code as it stood before the change (never with git
-- a scratch copy with the fix reverted by hand).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import icons, probe, tool_palette
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


def test_options_for_an_unknown_tool_is_none():
    """A document opened from an older save whose remembered ``state.generator``
    names a generator since retired from the registry (or a figure, which Clay no longer
    has) must not draw a stale heading with nothing behind it."""
    assert clay_tools._options_for("not-a-real-tool-any-more") is None
    assert clay_tools._options_for("humanoid") is None


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


# --- the names and the glyphs -------------------------------------------------


def test_display_name_spells_a_shape_the_way_a_person_reads_it():
    """``uv_sphere`` is a UV sphere; ``"Uv Sphere"`` read as a typo, and the Add
    menu, the rail and the flyout each spelled it their own way."""
    assert clay_tools.display_name("uv_sphere") == "UV Sphere"
    assert clay_tools.display_name("rounded_box") == "Rounded Box"
    assert clay_tools.display_name("icosphere") == "Icosphere"
    assert clay_tools.display_name("box") == "Box"


def test_every_clay_shape_has_a_name_and_no_two_share_a_glyph():
    """The rail names each shape beside its glyph, and a glyph two shapes wear is
    two shapes the eye cannot tell apart: cone and wedge shared triangle-alert,
    and uv_sphere and torus were both a plain circle."""
    names = [name for _label, group in bp.CLAY_GENERATORS for name in group]
    assert len(names) == 15
    glyphs = [tool_palette.PRIMITIVE_ICONS[name] for name in names]
    assert len(set(glyphs)) == len(glyphs), {
        glyph: [n for n in names if tool_palette.PRIMITIVE_ICONS[n] == glyph]
        for glyph in glyphs
        if glyphs.count(glyph) > 1
    }
    assert len({clay_tools.display_name(name) for name in names}) == len(names)


def test_the_mode_pill_is_ordered_by_its_keys():
    assert [key for _mode, _label, key in clay_tools.MODE_BUTTONS] == ["1", "2", "3", "4"]


def test_the_quick_shapes_are_real_clay_shapes():
    assert set(clay_tools.QUICK_SHAPES) <= set(bp.CLAY_GENERATOR_NAMES)
    assert 1 <= len(clay_tools.QUICK_SHAPES) <= 6, "the rail has room for a few, not all fifteen"


# --- the flyout ---------------------------------------------------------------


def _tab(doc):
    return SimpleNamespace(doc=doc, saving=False, uid="t1")


def test_the_flyout_lists_every_shape_with_its_name_and_glyph(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        ctx = FakeCtx()
        state = clay_mode.ensure(ctx)
        tab = _tab(bd.ClayDoc())

        probe.begin_frame()
        imgui.new_frame()
        imgui.set_next_window_size((320.0, 900.0))
        imgui.begin("##host")
        clay_tools.draw_add_menu(ctx, state, tab)
        imgui.end()
        imgui.end_frame()
        drawn = {c.label for c in probe.FRAME_CONTROLS}
    for _section, names in clay_tools.sections():
        for name in names:
            glyph = tool_palette.PRIMITIVE_ICONS[name]
            assert f"{glyph} {clay_tools.display_name(name)}##clay-add/{name}" in drawn, name


def test_clicking_the_box_entry_selects_it_and_places_it(monkeypatch):
    """Driven for real through a synthetic click, the way
    ``test_context_controls`` presses controls -- a control wired to nothing
    passes every test that calls the setter directly."""
    with imgui_context(monkeypatch) as imgui:
        ctx = FakeCtx()
        state = clay_mode.ensure(ctx)
        state.generator = "cylinder"
        doc = bd.ClayDoc()
        tab = _tab(doc)
        box_label = f"{icons.BOX} Box##clay-add/box"

        def frame(pos=(-100.0, -100.0), down=False):
            io = imgui.get_io()
            io.add_mouse_pos_event(pos[0], pos[1])
            io.add_mouse_button_event(0, down)
            probe.begin_frame()
            imgui.new_frame()
            imgui.set_next_window_size((320.0, 900.0))
            imgui.begin("##host")
            clay_tools.draw_add_menu(ctx, state, tab)
            imgui.end()
            imgui.end_frame()
            return list(probe.FRAME_CONTROLS)

        found = [c for c in frame() if c.label == box_label]
        assert found, "no Box entry drawn -- the flyout changed shape"
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


def test_the_add_flyout_draws_no_operation_buttons(monkeypatch):
    """The ~50-button op grid is gone from the sidebar: every operation is a
    row in the header's menu strip and the right-click menu now, grouped by
    ``menutree``. A flyout that drew them again would be a second list of what
    Clay can do -- the thing the registry exists to prevent."""
    with imgui_context(monkeypatch) as imgui:
        ctx = FakeCtx()
        state = clay_mode.ensure(ctx)
        tab = _tab(bd.ClayDoc())

        def frame():
            probe.begin_frame()
            imgui.new_frame()
            imgui.set_next_window_size((450.0, 900.0))
            imgui.begin("##host")
            clay_tools.draw_add_menu(ctx, state, tab)
            imgui.end()
            imgui.end_frame()
            return list(probe.FRAME_CONTROLS)

        controls_ = frame()
    labels = {c.text for c in controls_}
    assert not any("##clayop" in c.label for c in controls_)
    assert not labels & {op.label.rstrip(".") for op in clay_ops.OPS}, (
        "an op is drawn in the Add palette"
    )


def test_the_shapes_module_no_longer_carries_the_action_grid_or_its_popup_call():
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
