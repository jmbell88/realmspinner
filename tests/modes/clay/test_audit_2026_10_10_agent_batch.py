"""Regression tests for the 2026-10-10 audit's Clay agent batch/measure findings,
second fixer: clay-29 (a checkpoint inside a batch could never be restored),
clay-64 (mode-specific arguments were read and dropped), clay-65 (a refused
first call in a batch left an empty tab) and clay-68 (agent files citing a
deleted plan file).

Each test's name is the claim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import realmspinner.studio.modes.clay.agent as agent_pkg
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

from .test_agent_clay import _add_inline_reference_raw, _Ctx, _new_agent_tab, _payload


def _ctx() -> _Ctx:
    """The shared fake ctx plus the one attribute closing a tab reaches for
    (``state.preview``, the imgui texture registry the UV pane releases from)."""
    ctx = _Ctx()
    ctx.state.preview = {}
    return ctx


def _text(result: dict) -> str:
    return result["content"][0]["text"]


def _call(ctx: _Ctx, session: agent_clay.Session, name: str, args: dict) -> dict:
    return agent_clay.call(ctx, session, name, args)


def _tab_count(ctx: _Ctx) -> int:
    return len(clay_mode.ensure(ctx).docs)


_BOX = {"generator": "box"}


# --- clay-29 -----------------------------------------------------------------


def test_a_clay_checkpoint_inside_a_batch_is_refused_and_says_where_to_put_it() -> None:
    ctx, session = _ctx(), agent_clay.Session()
    result = _call(
        ctx,
        session,
        "clay_batch",
        {
            "calls": [
                {"name": "clay_add_primitive", "arguments": {**_BOX, "name": "a"}},
                {"name": "clay_checkpoint", "arguments": {"name": "after_a"}},
                {"name": "clay_add_primitive", "arguments": {**_BOX, "name": "b"}},
            ],
            "rollback_on_error": True,
        },
    )
    # The 2026-10-14 audit's clay-agent-tools-04: refused up front, before anything runs, so
    # there is no ``stopped_at``/``rolled_back`` -- the batch never started.
    assert result["isError"] is True
    assert "before or after the batch" in _text(result)
    assert result["structuredContent"]["field"] == "calls"
    assert result["structuredContent"]["recovery"] == "fix_arguments"
    assert "stopped_at" not in result["structuredContent"]
    # Nothing ran and nothing was named, so there is no checkpoint to find "gone" later.
    # (the refusal came before the batch minted a document, so there is not even a tab)
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert tab is None or (
        "after_a" not in tab.doc.checkpoints and len(tab.doc.objects) == 0
    )


def test_a_checkpoint_set_around_a_batch_is_still_restorable() -> None:
    ctx, session = _ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session)
    assert _call(ctx, session, "clay_checkpoint", {"name": "before"})["isError"] is False
    batch = _call(
        ctx,
        session,
        "clay_batch",
        {"calls": [{"name": "clay_add_primitive", "arguments": {**_BOX, "name": "n"}}]},
    )
    assert batch["isError"] is False
    assert _call(ctx, session, "clay_checkpoint", {"name": "after"})["isError"] is False
    doc = clay_mode.ensure(ctx).get(session.tab_uid).doc
    restored = _call(ctx, session, "clay_restore", {"name": "before"})
    assert restored["isError"] is False, _text(restored)
    assert len(doc.objects) == 1
    back = _call(ctx, session, "clay_restore", {"name": "after"})
    assert back["isError"] is False, _text(back)
    assert len(doc.objects) == 2


# --- clay-64 -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "stray"),
    [
        ({"kind": "distance", "a": [0, 0, 0], "b": [1, 0, 0], "uid": 99}, "uid"),
        ({"kind": "distance", "a": [0, 0, 0], "b": [1, 0, 0], "faces": [999]}, "faces"),
        ({"kind": "angle", "a": [1, 0, 0], "b": [0, 0, 0], "c": [0, 1, 0], "faces": [0]}, "faces"),
        ({"kind": "volume", "faces": [0, 1]}, "faces"),
        ({"kind": "volume", "a": [0, 0, 0]}, "a"),
        ({"kind": "area", "b": [0, 0, 0]}, "b"),
    ],
)
def test_clay_measure_refuses_an_argument_its_kind_does_not_take(args: dict, stray: str) -> None:
    ctx, session = _ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    if args["kind"] in ("volume", "area"):
        args = {**args, "uid": uid}
    result = _call(ctx, session, "clay_measure", args)
    assert result["isError"] is True, _text(result)
    assert result["structuredContent"]["field"] == stray
    assert repr(stray) in _text(result)


def test_clay_measure_still_answers_each_kinds_own_arguments() -> None:
    ctx, session = _ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    for args in (
        {"kind": "distance", "a": [0, 0, 0], "b": [1, 0, 0]},
        {"kind": "angle", "a": [1, 0, 0], "b": [0, 0, 0], "c": [0, 1, 0]},
        {"kind": "area", "uid": uid, "faces": [0]},
        {"kind": "volume", "uid": uid},
    ):
        result = _call(ctx, session, "clay_measure", args)
        assert result["isError"] is False, (args, _text(result))


@pytest.mark.parametrize("stray", ["compare_mode", "alpha"])
def test_clay_render_refuses_a_compare_argument_given_without_compare(stray: str) -> None:
    ctx, session = _ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session)
    value = "overlay" if stray == "compare_mode" else 0.3
    result = _call(ctx, session, "clay_render", {stray: value})
    assert result["isError"] is True, _text(result)
    assert result["structuredContent"]["field"] == stray
    assert "compare" in _text(result)


def test_clay_reference_add_refuses_file_beside_png_base64() -> None:
    ctx, session = _ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session)
    import base64

    from .test_agent_clay import _tiny_png

    result = _call(
        ctx,
        session,
        "clay_reference_add",
        {
            "name": "r",
            "png_base64": base64.b64encode(_tiny_png()).decode("ascii"),
            "file": "bogus.png",
        },
    )
    assert result["isError"] is True, _text(result)
    assert result["structuredContent"]["field"] == "file"
    assert not session.references
    # The inline form on its own is untouched.
    assert _add_inline_reference_raw(ctx, session, "r")["isError"] is False


# --- clay-65 -----------------------------------------------------------------


def test_a_batch_whose_first_call_is_refused_mints_no_document() -> None:
    ctx, session = _ctx(), agent_clay.Session()
    before = _tab_count(ctx)
    result = _call(
        ctx,
        session,
        "clay_batch",
        {"calls": [{"name": "clay_add_primitive", "arguments": {"generator": "nope"}}]},
    )
    assert result["isError"] is True
    assert _payload(result)["stopped_at"] == 0
    assert session.tab_uid == ""
    assert _tab_count(ctx) == before


def test_a_batch_that_placed_something_before_a_refusal_keeps_the_tab_it_minted() -> None:
    ctx, session = _ctx(), agent_clay.Session()
    result = _call(
        ctx,
        session,
        "clay_batch",
        {
            "calls": [
                {"name": "clay_add_primitive", "arguments": _BOX},
                {"name": "clay_add_primitive", "arguments": {"generator": "nope"}},
            ]
        },
    )
    assert result["isError"] is True
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert tab is not None
    assert len(tab.doc.objects) == 1


def test_a_refused_batch_never_closes_a_tab_the_session_already_owned() -> None:
    ctx, session = _ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session)
    pinned = session.tab_uid
    result = _call(
        ctx,
        session,
        "clay_batch",
        {"calls": [{"name": "clay_add_primitive", "arguments": {"generator": "nope"}}]},
    )
    assert result["isError"] is True
    assert session.tab_uid == pinned
    assert clay_mode.ensure(ctx).get(pinned) is not None


# --- clay-68 -----------------------------------------------------------------


def test_no_clay_agent_file_cites_dev_restructure_md() -> None:
    """The plan file was deleted; its name in a docstring sends a reader to
    nothing. Scoped to the Clay agent directory -- other packages carry the
    same citation and are not this finding's to fix."""
    root = Path(agent_pkg.__file__).parent
    offenders = [
        path.name
        for path in sorted(root.glob("*.py"))
        if "RESTRUCTURE.md" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
