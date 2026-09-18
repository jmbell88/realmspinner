"""Regression tests for the 2026-09-14 audit's Clay agent-door findings:
docs-06 (TODO F9, ``clay_op`` crashing on a wrong-typed ``params`` value),
agents-09 (``clay_reference_get`` skipping the post-``bounded_png`` frame
budget check ``_h_render`` already performs), and the clay-01 follow-up
(``clay_analyze``'s output schema declaring ``intersects`` a bare boolean
after ``analyze.py``'s own clay-01 fix started returning ``None`` for it).

Kept out of ``tests/test_agent_clay.py`` deliberately -- that file carries
the user's own uncommitted work, and this brief's constraints say every new
test for this session's fixes goes in this file instead.
"""

from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from warlock.studio.modes.clay.agent import dispatch as agent_clay


class _Ctx:
    """The same minimal ``ctx`` double ``tests/test_agent_clay.py`` uses --
    duplicated here rather than imported, since that module carries the
    user's own uncommitted work and this file must not import from it."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.settings = SimpleNamespace()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _payload(result: dict) -> Any:
    return json.loads(result["content"][0]["text"])


def _new_agent_tab(ctx: _Ctx, session: agent_clay.Session, generator: str = "box") -> int:
    result = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": generator})
    assert result["isError"] is False, result
    return _payload(result)["uid"]


def _tiny_png() -> bytes:
    im = Image.new("RGB", (4, 4), "white")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _add_inline_reference(
    ctx: _Ctx, session: agent_clay.Session, name: str = "ref1", view: str = "other"
) -> dict:
    b64 = base64.b64encode(_tiny_png()).decode("ascii")
    result = agent_clay.call(
        ctx, session, "clay_reference_add", {"name": name, "png_base64": b64, "view": view}
    )
    assert result["isError"] is False, result
    return result


# --- docs-06 / TODO F9: clay_op refuses a wrong-typed params value -----------


@pytest.mark.parametrize(
    "op_name, params, expected_snippet",
    [
        # mirror-x declares no params at all (its axis is closed over, not
        # passed) -- before the fix, an unrecognised key like this one
        # reached clay_ops.run's op.run(ctx, doc, **values) and collided
        # with the bound axis: TypeError: mirror() got multiple values for
        # argument 'axis', surfaced as "failed unexpectedly; see the log."
        ("mirror-x", {"axis": 0}, "not a parameter"),
        # mirror-copy's axis is a declared `choices` param, stored as an
        # index -- before the fix, a string reached run's own
        # float(values[param.name]) and leaked "could not convert string to
        # float: 'x'" verbatim.
        ("mirror-copy", {"axis": "x"}, "must be a single number"),
        # A list anywhere in params raised TypeError at that same float()
        # call before the fix.
        ("mirror-copy", {"offset": [1, 2]}, "must be a single number"),
    ],
)
def test_clay_op_refuses_a_wrong_typed_param_value_instead_of_crashing(
    op_name: str, params: dict, expected_snippet: str
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)

    result = agent_clay.call(ctx, session, "clay_op", {"name": op_name, "params": params})

    assert result["isError"] is True, result
    assert result["structuredContent"]["field"] == "params", result
    text = result["content"][0]["text"]
    assert expected_snippet in text, text
    # A refused call runs nothing -- pushed a step is exactly the harm a
    # crash surfaced as "failed unexpectedly" would otherwise hide behind.
    assert result["structuredContent"]["changed"] is False, result


def test_clay_op_still_runs_a_correctly_typed_declared_param() -> None:
    """The type check must not refuse what already worked: mirror-copy's own
    declared, correctly-typed params still run the op."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_op", {"name": "mirror-copy", "params": {"axis": 0.0, "offset": 0.5}}
    )

    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["ran"] is True, payload


# --- agents-09: clay_reference_get checks the frame budget after bounded_png -


def test_reference_get_refuses_a_reply_too_large_for_the_wire_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``_h_reference_get`` used to hand ``image_png(bounded_png(
    ref.png))`` straight back with no check of the size that actually
    crosses the wire -- the same check ``_h_render``'s ``_over_frame_budget``
    already performs after its own ``bounded_png`` calls, and
    ``bounded_png``'s own docstring says a caller must make. ``MAX_FRAME`` is
    monkeypatched absurdly small (the same technique
    ``test_clay_render_compare_is_refused_when_the_sheet_would_not_fit_one_frame``
    in ``tests/test_agent_clay.py`` uses for ``clay_render``) so the tiny
    fake PNG this file uses is still over budget, rather than needing a
    multi-megabyte reference to prove the same point. Fails against the
    unfixed handler, which never performs this check at all.
    """
    from warlock.mcp import rpc

    ctx = _Ctx()
    session = agent_clay.Session()
    _add_inline_reference(ctx, session, "ref1")
    monkeypatch.setattr(rpc, "MAX_FRAME", 10)

    result = agent_clay.call(ctx, session, "clay_reference_get", {"name": "ref1"})

    assert result["isError"] is True, result


def test_reference_get_still_returns_the_picture_under_budget() -> None:
    """The check must not refuse the ordinary case: a small stored reference
    still comes back, unrefused, at today's real budget constants."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _add_inline_reference(ctx, session, "ref1")

    result = agent_clay.call(ctx, session, "clay_reference_get", {"name": "ref1"})

    assert result["isError"] is False, result
    assert len(result["content"]) == 2, result


# --- clay-01 follow-up: clay_analyze's schema allows an unknown intersects --


def test_clay_analyze_output_schema_allows_an_unknown_intersects_past_the_pair_cap() -> None:
    """``analyze.py``'s own clay-01 fix lets ``PairAnalysis.intersects`` be
    ``None`` -- unknown, not "no" -- once a pair clears
    ``MAX_TRIANGLE_PAIRS`` and the full narrow phase never runs. Before this
    fix, ``clay_analyze``'s declared ``outputSchema`` still called
    ``intersects`` a bare ``{"type": "boolean"}`` with no ``null`` admitted,
    unlike ``distance``'s own ``anyOf`` beside it -- so ``_h_analyze`` could
    emit a value ("null") the tool's own advertised schema forbade.
    """
    tools = {t.name: t for t in agent_clay.tools()}
    pairs_schema = tools["clay_analyze"].output_schema["properties"]["pairs"]
    intersects_schema = pairs_schema["items"]["properties"]["intersects"]
    expected = {"anyOf": [{"type": "null"}, {"type": "boolean"}]}
    assert intersects_schema == expected, intersects_schema
