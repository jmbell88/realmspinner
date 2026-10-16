"""car-07: a ``clay_op`` that changes the document's OBJECT selection says so.

An agent did ``mirror-copy`` then ``clay_element_mode face`` and read ``editing``
(the copies, which is true state) as stale, because the ``clay_op`` reply listed
only ``changed: [{uid, ...}]`` and never mentioned that the selection had moved.
The reply key is ``object_selection`` rather than ``selection``: ``selection``
is already the element-mode pick report (a list of per-object dicts) and one key
must not hold two shapes.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay


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


def _payload(result: dict) -> Any:
    return json.loads(result["content"][0]["text"])


def _doc(ctx: _Ctx, session: agent_clay.Session) -> Any:
    return clay_mode.ensure(ctx).get(session.tab_uid).doc


def _tab_with_selected_box() -> tuple[_Ctx, agent_clay.Session, int]:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    assert result["isError"] is False, result
    uid = _payload(result)["uid"]
    selected = agent_clay.call(ctx, session, "clay_select", {"uids": [uid]})
    assert selected["isError"] is False, selected
    return ctx, session, uid


def test_clay_op_mirror_copy_reply_reports_the_new_selection() -> None:
    ctx, session, uid = _tab_with_selected_box()

    result = agent_clay.call(
        ctx, session, "clay_op", {"name": "mirror-copy", "params": {"axis": 0, "offset": 0}}
    )

    assert result["isError"] is False, result
    held = sorted(_doc(ctx, session).selection)
    assert held != [uid], "the premise: mirror-copy moves the selection onto the copies"
    assert result["structuredContent"]["object_selection"] == held
    assert _payload(result)["object_selection"] == held


def test_clay_op_that_does_not_change_the_selection_omits_the_key() -> None:
    ctx, session, uid = _tab_with_selected_box()

    result = agent_clay.call(ctx, session, "clay_op", {"name": "shade-smooth"})

    assert result["isError"] is False, result
    assert sorted(_doc(ctx, session).selection) == [uid]
    assert "object_selection" not in result["structuredContent"]
    assert "object_selection" not in _payload(result)


def test_a_refused_clay_op_has_no_selection_key() -> None:
    ctx, session, _uid = _tab_with_selected_box()

    result = agent_clay.call(ctx, session, "clay_op", {"name": "no-such-op"})

    assert result["isError"] is True, result
    assert "object_selection" not in result["structuredContent"]


def test_the_instructions_say_select_by_adds_a_uid_to_the_edited_set() -> None:
    text = agent_clay.instructions()

    assert "ADD that uid to the edited set" in text
    assert "object_selection" in text
