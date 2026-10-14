"""Clay's commands in the global Ctrl+K palette.

The menu strip is gone, so every tool that was on it has to stay one search from the
user; these tests pin that the palette offers them, only in Clay, only with a
document, as doors onto what the mode already does, and that building ~65 rows
per frame is not done per frame.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import palette
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.ui.panes import header as clay_header

from .test_clay_mode import FakeCtx


class _Camera:
    def __init__(self) -> None:
        self.orthographic = False
        self.looked: list[str] = []

    def look_along(self, name: str) -> None:
        self.looked.append(name)


def _ctx(*, objects: int = 2, with_view: bool = False) -> Any:
    """A Clay workspace with ``objects`` boxes, the first one selected, wired
    with just enough of the app's state for ``palette.commands`` to build."""
    from realmspinner.studio.state import ManualState

    ctx = FakeCtx()
    ctx.state.mode = "clay"
    ctx.state.previous_mode = "clay"
    ctx.state.mode_observed = "clay"
    ctx.state.create = SimpleNamespace(stage="mesh")
    ctx.state.selected = None
    ctx.state.source_job = None
    ctx.state.wireframe = False
    ctx.state.turntable = False
    ctx.state.show_fps = False
    ctx.state.manual = ManualState()
    ctx.cache.jobs = []
    ctx.cache.get = lambda _job_id: None
    ctx.viewer = None
    ctx.svc = SimpleNamespace(store=SimpleNamespace(trashed=lambda: []))
    doc = bd.ClayDoc()
    for index in range(objects):
        doc.add_object(bd.Obj(uid=bd.new_uid(), name=f"Box {index}", mesh=bp.box()))
    doc.select([doc.objects[0].uid])
    clay_mode.adopt(ctx, doc, title="Scene")
    if with_view:
        ctx.clay_view = SimpleNamespace(camera=_Camera())
    return ctx


def _clay(ctx: Any) -> dict[str, palette.Command]:
    return {c.key: c for c in palette.commands(ctx) if c.group == "Clay"}


def _doc(ctx: Any) -> Any:
    return ctx.state.clay.active.doc


# --- where they appear ----------------------------------------------------------


def test_clay_commands_appear_only_in_the_clay_workspace_with_a_document():
    ctx = _ctx()
    assert _clay(ctx)

    ctx.state.mode = "inker"
    assert _clay(ctx) == {}, "another workspace must not list Clay's commands"

    ctx.state.mode = "clay"
    ctx.state.clay.close(ctx.state.clay.active.uid)
    assert ctx.state.clay.active is None
    assert _clay(ctx) == {}, "no document, nothing to act on"


def test_clay_commands_are_absent_before_clay_has_any_state():
    ctx = _ctx()
    ctx.state.clay = None
    assert _clay(ctx) == {}


# --- the registry's ops -----------------------------------------------------------


def test_every_op_for_the_current_mode_is_listed_with_its_key_text_in_the_clay_group():
    ctx = _ctx()
    found = _clay(ctx)
    assert _doc(ctx).element_mode == "object"
    for op in clay_ops.menu("object"):
        if op.name == "frame":  # the view row below owns it
            continue
        command = found[f"clay-op:{op.name}"]
        assert command.label == op.label
        assert command.hint == op.key
        assert command.group == "Clay"


def test_an_op_of_another_mode_is_listed_greyed_and_names_the_mode_it_lives_in():
    ctx = _ctx()
    extrude = _clay(ctx)["clay-op:extrude"]
    assert not extrude.enabled(ctx)
    assert extrude.why == "Switch to Points, Edges or Faces mode first."

    _doc(ctx).set_element_mode("face")
    found = _clay(ctx)
    assert found["clay-op:extrude"].why != "Switch to Points, Edges or Faces mode first."
    mirror = found["clay-op:mirror-x"]
    assert not mirror.enabled(ctx)
    assert mirror.why == "Switch to Shape mode first."


def test_every_greyed_clay_command_says_why():
    ctx = _ctx()
    doc = _doc(ctx)
    for element_mode in ("object", "vertex", "edge", "face"):
        doc.set_element_mode(element_mode)
        for command in _clay(ctx).values():
            if not command.enabled(ctx):
                assert command.why, f"{command.key} is greyed in {element_mode} with no reason"


def test_an_op_the_selection_refuses_is_greyed_with_the_ops_own_reason():
    ctx = _ctx(objects=1)
    doc = _doc(ctx)
    join = clay_ops.get("join")
    assert not join.enabled(doc), "one object cannot be merged"
    command = _clay(ctx)["clay-op:join"]
    assert not command.enabled(ctx)
    assert command.why == clay_ops.reason_for(join, doc) != ""


def test_running_an_op_row_calls_the_event_layers_op_door(monkeypatch: pytest.MonkeyPatch):
    ctx = _ctx()
    calls: list[tuple[Any, Any, str]] = []
    monkeypatch.setattr(
        clay_mode, "fire_op", lambda c, doc, op, **_kw: calls.append((c, doc, op.name))
    )
    _clay(ctx)["clay-op:mirror-x"].run(ctx)
    assert calls == [(ctx, _doc(ctx), "mirror-x")]


def test_an_op_with_parameters_asks_for_its_popup_instead_of_running_blind():
    ctx = _ctx()
    state = ctx.state.clay
    assert clay_ops.get("mirror-copy").params
    _clay(ctx)["clay-op:mirror-copy"].run(ctx)
    assert state.pending_op == "mirror-copy"
    assert state.open_op_popup is True


def test_an_op_row_really_runs_the_op():
    ctx = _ctx()
    doc = _doc(ctx)
    doc.select([obj.uid for obj in doc.objects])
    head = doc.history.head
    assert clay_ops.get("group").enabled(doc)
    _clay(ctx)["clay-op:group"].run(ctx)
    assert doc.history.head != head, "the op pushed its undo step"
    assert ctx.toasts == []


def test_an_op_row_is_shut_while_the_document_is_being_written():
    ctx = _ctx()
    ctx.state.clay.active.saving = True
    command = _clay(ctx)["clay-op:mirror-x"]
    assert not command.enabled(ctx)
    assert command.why == clay_header.SAVING_WHY


# --- tools, modes ------------------------------------------------------------------


@pytest.mark.parametrize("tool", ["select", "move", "rotate", "scale"])
def test_a_tool_row_sets_the_tool_the_rail_sets(tool: str):
    ctx = _ctx()
    other = "move" if tool != "move" else "scale"
    ctx.state.clay.tool = other
    _clay(ctx)[f"clay-tool:{tool}"].run(ctx)
    assert ctx.state.clay.tool == tool


def test_the_tool_rows_use_the_rails_words_and_the_tools_keys():
    found = _clay(_ctx())
    assert found["clay-tool:rotate"].label == "Turn tool"
    assert found["clay-tool:scale"].label == "Resize tool"
    assert found["clay-tool:move"].hint == "G"


@pytest.mark.parametrize("mode", ["object", "vertex", "edge", "face"])
def test_a_mode_row_switches_the_element_mode_the_header_switches(mode: str):
    ctx = _ctx()
    doc = _doc(ctx)
    doc.set_element_mode("vertex" if mode != "vertex" else "face")
    _clay(ctx)[f"clay-mode:{mode}"].run(ctx)
    assert doc.element_mode == mode


def test_the_mode_rows_are_shut_while_saving_like_the_headers_switch():
    ctx = _ctx()
    ctx.state.clay.active.saving = True
    command = _clay(ctx)["clay-mode:face"]
    assert not command.enabled(ctx)
    assert command.why == clay_header.SAVING_WHY
    command.run(ctx)
    assert _doc(ctx).element_mode == "object"


# --- view ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "stored"),
    [
        ("shading-solid", "solid"),
        ("shading-material", "material"),
        ("shading-wireframe", "wireframe"),
    ],
)
def test_a_shading_row_sets_the_state_the_shading_switch_sets(key: str, stored: str):
    ctx = _ctx()
    ctx.state.clay.shading = "material" if stored != "material" else "solid"
    _clay(ctx)[f"clay-view:{key}"].run(ctx)
    assert ctx.state.clay.shading == stored


def test_the_xray_row_flips_the_xray_flag():
    ctx = _ctx()
    row = _clay(ctx)["clay-view:xray"]
    row.run(ctx)
    assert ctx.state.clay.xray is True
    row.run(ctx)
    assert ctx.state.clay.xray is False


def test_the_overlay_rows_flip_what_the_view_popover_flips_and_persist_it():
    ctx = _ctx()
    state = ctx.state.clay
    found = _clay(ctx)
    found["clay-view:grid"].run(ctx)
    assert state.grid is False
    found["clay-view:god_light"].run(ctx)
    assert state.god_light is True
    found["clay-view:wire"].run(ctx)
    assert state.overlays["wire"] is True
    found["clay-view:stats"].run(ctx)
    assert state.overlays["stats"] is True
    assert ctx.settings.store["clay"]["view"]["grid"] is False
    assert ctx.settings.store["clay"]["view"]["god_light"] is True


def test_the_camera_rows_drive_the_live_viewports_camera():
    ctx = _ctx(with_view=True)
    found = _clay(ctx)
    found["clay-view:axis-top"].run(ctx)
    found["clay-view:axis-back"].run(ctx)
    assert ctx.clay_view.camera.looked == ["top", "back"]
    found["clay-view:ortho"].run(ctx)
    assert ctx.clay_view.camera.orthographic is True


def test_the_camera_rows_are_shut_until_the_viewport_exists():
    ctx = _ctx(with_view=False)
    found = _clay(ctx)
    for key in ("clay-view:ortho", "clay-view:axis-front", "clay-view:axis-bottom"):
        assert not found[key].enabled(ctx)
        assert found[key].why == clay_header.NO_VIEW_WHY
    assert len([k for k in found if k.startswith("clay-view:axis-")]) == len(
        clay_header.AXIS_ROWS
    )


def test_the_frame_row_asks_the_viewport_to_frame():
    ctx = _ctx()
    assert ctx.state.clay.frame_pending is False
    _clay(ctx)["clay-view:frame"].run(ctx)
    assert ctx.state.clay.frame_pending is True


# --- export ----------------------------------------------------------------------------


def test_the_export_rows_call_the_headers_export_doors(monkeypatch: pytest.MonkeyPatch):
    ctx = _ctx()
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(clay_mode, "export_asset", lambda c, tab: calls.append(("library",)))
    monkeypatch.setattr(
        clay_mode, "export_mesh_file", lambda c, tab, kind: calls.append((kind,))
    )
    monkeypatch.setattr(clay_mode, "save_screenshot", lambda c, tab: calls.append(("shot",)))
    found = _clay(ctx)
    for key in ("library", "glb", "obj", "screenshot"):
        assert found[f"clay-export:{key}"].enabled(ctx)
        found[f"clay-export:{key}"].run(ctx)
    assert calls == [("library",), ("glb",), ("obj",), ("shot",)]


def test_the_export_rows_grey_with_the_sentence_every_output_button_uses():
    ctx = _ctx()
    doc = _doc(ctx)
    for obj in doc.objects:
        doc.set_props(obj.uid, visible=False)
    found = _clay(ctx)
    for key in ("library", "glb", "obj", "screenshot"):
        row = found[f"clay-export:{key}"]
        assert not row.enabled(ctx)
        assert row.why == "Nothing visible to send -- every object is hidden."


# --- the way in -----------------------------------------------------------------------


def test_the_import_mesh_row_calls_the_picker_the_model_page_and_the_file_menu_call(
    monkeypatch: pytest.MonkeyPatch,
):
    """Old code: Import Mesh... was a row of the Add menu in the deleted menu strip and
    the palette had no row for it, so with the strip gone it was one search away from
    nobody."""
    ctx = _ctx()
    asked: list[Any] = []
    monkeypatch.setattr(clay_mode, "ask_import_mesh", lambda c: asked.append(c))
    row = _clay(ctx)["clay-file:import-mesh"]
    assert row.label == "Import Mesh..." and row.group == "Clay"
    assert row.enabled(ctx)
    row.run(ctx)
    assert asked == [ctx]
    assert _top(ctx, "import mesh", 1) == ["clay-file:import-mesh"]


def test_the_import_mesh_row_is_shut_while_the_document_is_being_written():
    ctx = _ctx()
    ctx.state.clay.active.saving = True
    row = _clay(ctx)["clay-file:import-mesh"]
    assert not row.enabled(ctx)
    assert row.why == clay_header.SAVING_WHY


# --- the list itself --------------------------------------------------------------------


def test_clay_command_keys_are_unique_and_labels_stay_in_latin_1():
    ctx = _ctx(with_view=True)
    commands = palette.commands(ctx)
    keys = [c.key for c in commands]
    assert len(keys) == len(set(keys))
    for command in commands:
        for text in (command.label, command.hint, command.group):
            assert all(ord(ch) < 0x100 for ch in text), text


def test_no_clay_command_leaks_into_the_menus_or_the_assistants_destinations():
    from realmspinner.studio import menus
    from realmspinner.studio.assistant import doors

    ctx = _ctx()
    commands = palette.commands(ctx)
    assert any(c.key.startswith("clay-") for c in commands)
    specs = menus._command_specs(ctx, commands)
    assert not [s for s in specs if s.identity.startswith("command:clay-")]
    assert not [d for d in doors.destinations(ctx) if d.key.startswith("clay-")]


# --- the cache ----------------------------------------------------------------------------


def _builds() -> int:
    return palette._clay_cache.builds


def test_the_list_is_not_rebuilt_while_nothing_changed():
    ctx = _ctx()
    palette.commands(ctx)
    start = _builds()
    first = _clay(ctx)
    for _ in range(5):
        palette.commands(ctx)
    assert _builds() == start
    assert _clay(ctx).keys() == first.keys()


def test_the_list_is_rebuilt_when_the_document_changes():
    ctx = _ctx()
    doc = _doc(ctx)
    palette.commands(ctx)
    start = _builds()
    doc.set_props(doc.objects[0].uid, name="Renamed")
    palette.commands(ctx)
    assert _builds() == start + 1


def test_the_list_is_rebuilt_when_the_element_mode_changes():
    ctx = _ctx()
    palette.commands(ctx)
    start = _builds()
    _doc(ctx).set_element_mode("face")
    found = _clay(ctx)
    assert _builds() == start + 1
    assert found["clay-op:extrude"].why != "Switch to Points, Edges or Faces mode first."


def test_the_list_is_rebuilt_when_the_selection_changes():
    ctx = _ctx()
    doc = _doc(ctx)
    palette.commands(ctx)
    start = _builds()
    doc.select([doc.objects[0].uid, doc.objects[1].uid])
    found = _clay(ctx)
    assert _builds() == start + 1
    assert clay_ops.get("join").enabled(doc)
    assert found["clay-op:join"].enabled(ctx)


def test_the_list_is_rebuilt_when_a_save_starts_and_ends():
    ctx = _ctx()
    tab = ctx.state.clay.active
    palette.commands(ctx)
    start = _builds()
    tab.saving = True
    palette.commands(ctx)
    tab.saving = False
    palette.commands(ctx)
    assert _builds() == start + 2


def test_a_second_document_never_reads_the_first_documents_cached_reasons():
    first = _ctx(objects=1)
    palette.commands(first)
    start = _builds()
    second = _ctx(objects=3)
    doc = _doc(second)
    doc.select([o.uid for o in doc.objects])
    found = _clay(second)
    assert _builds() == start + 1, "a different document is a different question"
    assert found["clay-op:join"].enabled(second)


# --- finding things -----------------------------------------------------------------------


def _top(ctx: Any, query: str, count: int = 6) -> list[str]:
    ranked = palette.rank(query, palette.commands(ctx), lambda c: c.label)
    return [c.key for c in ranked[:count]]


def test_searching_mirror_finds_the_mirror_ops():
    top = _top(_ctx(), "mirror", 4)
    assert set(top) == {
        "clay-op:mirror-x",
        "clay-op:mirror-y",
        "clay-op:mirror-z",
        "clay-op:mirror-copy",
    }


@pytest.mark.parametrize(
    ("query", "key"),
    [
        ("extrude", "clay-op:extrude"),
        ("merge objects", "clay-op:join"),
        ("drop to ground", "clay-op:drop-to-ground"),
        ("box unwrap", "clay-op:unwrap"),
        ("turn tool", "clay-tool:rotate"),
        ("faces mode", "clay-mode:face"),
        ("god light", "clay-view:god_light"),
        ("export glb", "clay-export:glb"),
    ],
)
def test_an_existing_tool_is_found_by_its_own_label_first(query: str, key: str):
    assert _top(_ctx(), query, 1) == [key]


def test_frame_in_clay_lands_on_clays_row_not_the_create_only_one():
    ctx = _ctx()
    assert _top(ctx, "frame", 1) == ["clay-view:frame"]


def test_the_pose_row_enters_pose_mode_and_a_shape_row_leaves_it():
    ctx = _ctx()
    doc = _doc(ctx)
    _clay(ctx)["clay-mode:pose"].run(ctx)
    assert doc.pose_mode
    _clay(ctx)["clay-mode:vertex"].run(ctx)
    assert not doc.pose_mode and doc.element_mode == "vertex"
