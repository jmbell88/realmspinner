"""Round-two repairs of the 2026-10-09 second Clay audit: wiring that the first
round's fixers built the function for but did not connect (clay-120), the toast
scanner's view of ``ops.py``, and the agent door's clay-102 behaviour.

Each test name is the claim; the clay-120 ones were run against the unconnected
call sites first.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import menus, palette
from realmspinner.studio.modes.clay import doc_io
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

SRC = Path(__file__).resolve().parents[3] / "src" / "realmspinner" / "studio"
CLAY = SRC / "modes" / "clay"


# --- clay-120: every output control passes the gesture --------------------------


def _doc() -> bd.ClayDoc:
    doc = bd.ClayDoc()
    doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Box",
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
        )
    )
    return doc


def _menu_ctx(tab: Any, *, dragging: bool) -> Any:
    return SimpleNamespace(
        state=SimpleNamespace(mode="clay", clay=SimpleNamespace(active=tab)),
        clay_view=SimpleNamespace(dragging=dragging),
        settings=SimpleNamespace(get=lambda key, default=None: default),
    )


def test_the_file_menus_exports_are_greyed_with_the_drag_sentence_during_a_live_drag(
    monkeypatch,
) -> None:
    monkeypatch.setattr(clay_mode, "recent_paths", lambda ctx: [])
    tab = SimpleNamespace(doc=_doc(), saving=False, uv_view=None)
    idle = {r.identity: r for r in menus._clay_file_specs(_menu_ctx(tab, dragging=False), [])}
    live = {r.identity: r for r in menus._clay_file_specs(_menu_ctx(tab, dragging=True), [])}
    for identity in ("clay:export-glb", "clay:export-obj", "clay:screenshot"):
        assert idle[identity].enabled and not idle[identity].disabled_reason, identity
        assert not live[identity].enabled, identity
        assert live[identity].disabled_reason == doc_io.GESTURE_WHY, identity
    # Saving still outranks the drag, the way it does for the buttons.
    tab.saving = True
    saving = {r.identity: r for r in menus._clay_file_specs(_menu_ctx(tab, dragging=True), [])}
    assert saving["clay:export-glb"].disabled_reason == "Saving..."


def _outputs_why_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "outputs_why"
    ]


def test_every_clay_pane_call_of_outputs_why_passes_the_live_gesture() -> None:
    """``props`` and ``header`` draw through imgui, which a test cannot press, so
    the claim is read off the call: a third argument that asks
    ``clay_mode._gesture_live``. Four sites: the Inspector's row, the header's
    Export menu, the header's split button -- and the File menu above."""
    expected = {"props.py": 1, "header.py": 2}
    for name, count in expected.items():
        calls = _outputs_why_calls(CLAY / "ui" / "panes" / name)
        assert len(calls) == count, (name, len(calls))
        for call in calls:
            third = call.args[2] if len(call.args) > 2 else None
            assert third is not None and "_gesture_live" in ast.unparse(third), (
                f"{name}:{call.lineno} greys the export buttons without the live drag"
            )


def test_the_palettes_export_rows_say_the_drag_sentence_through_doc_io() -> None:
    tab = SimpleNamespace(doc=_doc(), saving=False)
    rows = [c for c in palette._clay_build(tab.doc, tab, True) if c.key.startswith("clay-export:")]
    assert len(rows) == 4
    assert {c.why for c in rows} == {doc_io.GESTURE_WHY}
    assert doc_io.GESTURE_WHY == palette._CLAY_GESTURE_WHY
    idle = [c for c in palette._clay_build(tab.doc, tab, False) if c.key.startswith("clay-export:")]
    assert {c.why for c in idle} == {"Nothing to send yet."}


# --- the toast-level scanner ----------------------------------------------------


def test_ops_holds_the_group_sentence_in_a_constant_not_a_toast_literal() -> None:
    tree = ast.parse((CLAY / "ops.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and len(node.args) >= 2):
            continue
        # The module's own ``toast(ctx, message)`` -- a bare name. ``ctx.toast(text,
        # "error")`` is the real API and its second argument really is a level.
        func = node.func
        second = node.args[1]
        if isinstance(func, ast.Name) and func.id == "toast":
            assert not (isinstance(second, ast.Constant) and isinstance(second.value, str)), (
                f"ops.py:{node.lineno} puts a string literal where the scan reads a level"
            )
    assert "group has no geometry" in clay_ops.GROUP_HAS_NO_GEOMETRY


# --- clay-102 at the agent door -------------------------------------------------


class _Cache:
    def invalidate(self) -> None:
        pass


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.svc = None
        self.cache = _Cache()
        self.toasts: list[tuple[str, str]] = []
        self.settings = SimpleNamespace()

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _cylinder(ctx: _Ctx, session: agent_clay.Session) -> int:
    result = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "cylinder"})
    assert result["isError"] is False, result
    return json.loads(result["content"][0]["text"])["uid"]


def _steps(ctx: _Ctx, session: agent_clay.Session) -> int:
    return len(clay_mode.ensure(ctx).get(session.tab_uid).doc.history.history())


def test_clay_set_params_reasserting_the_current_values_reports_no_change_and_pushes_no_step() -> (
    None
):
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _cylinder(ctx, session)
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    current = dict(tab.doc.by_uid(uid).params)
    before = _steps(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_set_params", {"uid": uid, "params": {"segments": current["segments"]}}
    )

    assert result["isError"] is False, result
    payload = json.loads(result["content"][0]["text"])
    assert payload["changed"] is False
    assert _steps(ctx, session) == before


def test_clay_set_params_uids_reasserting_the_current_values_pushes_no_step_and_no_relabel() -> (
    None
):
    ctx, session = _Ctx(), agent_clay.Session()
    uids = [_cylinder(ctx, session), _cylinder(ctx, session)]
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    segments = tab.doc.by_uid(uids[0]).params["segments"]
    before = _steps(ctx, session)
    label_before = tab.doc.history.history()[-1][0]

    result = agent_clay.call(
        ctx, session, "clay_set_params", {"uids": uids, "params": {"segments": segments}}
    )

    assert result["isError"] is False, result
    payload = json.loads(result["content"][0]["text"])
    assert payload["changed"] is False
    assert [row["changed"] for row in payload["objects"]] == [False, False]
    assert _steps(ctx, session) == before
    assert tab.doc.history.history()[-1][0] == label_before, "the previous step was renamed"
