"""Clay has no menu strip, and nothing the strip offered was lost with it.

The Select | Add | Object | Mesh | UV strip drew ``menutree.BARS`` above the toolbar. Its
ops now live in the Inspector (More actions in Shape mode, Select row and Shape it in an
element mode), the right-click menu (``menutree.flat``) and Ctrl+K (``clay-op:<name>``);
its Add menu's shapes in the rail's ``+`` and the Inspector's tiles and "More shapes";
its Import Mesh... row on the Inspector's Model & history page, in File and in Ctrl+K.

Each test's name is the claim it makes. Against the code as it stood before the change:
the "no strip pane" test fails on ``ui/panes/strip.py`` existing, the Import Mesh tests
fail on there being no such button, File row or palette command, and the safety net
below is a claim the old code satisfied only *through* the strip -- it exists so the
next deletion cannot lose an op quietly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import icons, palette, widgets
from realmspinner.studio.modes.clay import menutree
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.ui import inspector_actions
from realmspinner.studio.modes.clay.ui import inspector_elements as clay_elements
from realmspinner.studio.modes.clay.ui.panes import header as clay_header
from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools

from .test_clay_palette import _clay, _ctx, _doc
from .test_inspector import _clean, _document, _rig, _state, imgui_ui  # noqa: F401

# -- no strip --------------------------------------------------------------------------


def test_no_menu_strip_pane_exists_and_the_header_does_not_draw_one():
    assert importlib.util.find_spec("realmspinner.studio.modes.clay.ui.panes.strip") is None
    assert not hasattr(clay_header, "clay_strip")
    source = Path(clay_header.__file__).read_text(encoding="utf-8")
    assert "strip.draw" not in source


# -- the safety net --------------------------------------------------------------------


def _bar_ops(mode: str) -> list[clay_ops.Op]:
    """Every op the strip listed for *mode*: the bar's groups, filtered to the mode."""
    seen: dict[str, clay_ops.Op] = {}
    for menu in menutree.BARS[mode]:
        for entry in menu.entries:
            for op in menutree.grouped_ops(entry.name, mode):
                seen.setdefault(op.name, op)
    return list(seen.values())


def _inspector_ops(mode: str) -> set[str]:
    """The ops the Inspector can run in *mode*, by the same functions it draws from."""
    names = {op.name for _key, _label, ops in inspector_actions.action_groups(mode) for op in ops}
    if mode == "object":
        names |= set(inspector_actions.PRIMARY)  # Duplicate and Delete have rows of their own
    else:
        names |= {op.name for op in clay_elements.select_ops(mode)}  # the Select row
    return names


@pytest.mark.parametrize("mode", clay_ops.ALL_MODES)
def test_every_op_the_old_strip_listed_is_reachable_without_it(mode: str):
    ctx = _ctx()
    _doc(ctx).set_element_mode(mode)
    palette_keys = set(_clay(ctx))
    right_click = {op.name for run in menutree.flat(mode) for op in run}
    inspector = _inspector_ops(mode)

    bar = _bar_ops(mode)
    assert bar, f"the {mode} bar lists no ops at all"
    for op in bar:
        homes = {
            "right-click menu": op.name in right_click,
            "Ctrl+K": f"clay-op:{op.name}" in palette_keys,
            "Inspector": op.name in inspector,
        }
        assert homes["right-click menu"] and homes["Ctrl+K"], (mode, op.name, homes)
        # Everything but the object-mode Select group (Ctrl+A, Ctrl+Shift+I and Esc own
        # those) is also a button in the Inspector.
        if not (mode == "object" and op.name in menutree.GROUPS["select"].ops):
            assert homes["Inspector"], (mode, op.name, homes)


def test_the_strips_add_menu_shapes_are_all_in_the_shape_list_the_rail_and_inspector_draw(
    imgui_ui,  # noqa: F811
):
    """Every shape ``CLAY_GENERATORS`` offers is a row of ``tools.draw_add_menu``, which
    the rail's ``+`` flyout and the Inspector's "More shapes" both host."""
    names = {name for _label, group in bp.CLAY_GENERATORS for name in group}
    assert names == {name for _label, group in clay_tools.sections() for name in group}

    ctx, _tab, doc = _document(1)
    doc.select([])
    widgets.request_open("clay/insp/more-shapes")
    rig = _rig(imgui_ui, ctx)
    drawn = {label.rsplit("##clay-add/", 1)[1] for label in rig.labels if "##clay-add/" in label}
    assert drawn == names


# -- Import Mesh... --------------------------------------------------------------------


def _model_page(imgui_ui) -> tuple[Any, Any, Any]:  # noqa: F811
    ctx, tab, doc = _document(1)
    doc.select([])
    rig = _rig(imgui_ui, ctx)
    _state(ctx).inspector_page = "model"
    rig.settle()
    return ctx, tab, rig


def test_import_mesh_is_a_button_on_the_model_and_history_page_next_to_its_settings(
    imgui_ui,  # noqa: F811
    monkeypatch,
):
    ctx, _tab, rig = _model_page(imgui_ui)
    labels = rig.labels
    button = next(label for label in labels if "Import Mesh..." in label)
    assert button.startswith(icons.FOLDER_OPEN)
    assert labels.index(button) < labels.index("##clay-import-scale"), "the settings follow it"
    assert "##clay-import-up" in labels

    asked: list[Any] = []
    monkeypatch.setattr(clay_mode, "ask_import_mesh", lambda c: asked.append(c))
    rig.click(contains="Import Mesh...")
    assert asked == [ctx]
    assert rig.driver.disabled_presses == []


def test_the_import_mesh_button_opens_a_picker_and_parses_the_file_it_is_given(
    imgui_ui,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    """The whole door, not the stub: button -> ``ask_import_mesh`` -> picker -> parse."""
    from realmspinner.kernels.geom3d import glbwrite
    from realmspinner.kernels.mesh import document as bd
    from realmspinner.studio import dialogs

    source = bd.ClayDoc()
    source.objects.append(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    path = tmp_path / "thing.glb"
    path.write_bytes(glbwrite.write_glb(bd.to_model(source)))
    monkeypatch.setattr(dialogs, "open_file", lambda *a, **k: path)

    ctx, _tab, rig = _model_page(imgui_ui)
    submitted: list[Any] = []

    def submit(key: str, run: Any, *args: Any) -> bool:
        submitted.append((key, run(*args)))
        return True

    ctx.submit = submit
    rig.click(contains="Import Mesh...")
    assert [key for key, _result in submitted] == ["clay-import"]
    result = submitted[0][1]
    assert result["title"] == "thing" and len(result["doc"].objects) == 1


def test_the_import_mesh_button_is_greyed_with_a_reason_while_the_document_saves(
    imgui_ui,  # noqa: F811
):
    ctx, tab, rig = _model_page(imgui_ui)
    tab.saving = True
    rig.settle()
    button = rig.find(contains="Import Mesh...")
    assert not button.enabled and button.reason == widgets.DOCUMENT_SAVING_WHY


def test_import_mesh_is_a_ctrl_k_command_and_a_file_menu_row(monkeypatch: pytest.MonkeyPatch):
    from realmspinner.studio import menus

    ctx = _ctx()
    asked: list[Any] = []
    monkeypatch.setattr(clay_mode, "ask_import_mesh", lambda c: asked.append(c))

    command = next(c for c in palette.commands(ctx) if c.key == "clay-file:import-mesh")
    assert command.label == "Import Mesh..." and command.enabled(ctx)
    command.run(ctx)

    rows = {row.identity: row for row in menus._clay_file_specs(ctx, [])}
    row = rows["clay:import-mesh"]
    assert row.path == ("File",) and row.label == "Import Mesh..." and row.enabled
    row.callback()
    assert asked == [ctx, ctx]
