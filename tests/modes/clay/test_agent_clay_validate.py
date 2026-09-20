"""Regression and derivation tests for ``clay_validate`` (advisory readiness
checks against ``kernels.mesh.readiness.PROFILES``) and for the plumbing its
sibling ``clay_op`` row ``decimate`` needs to run a real gltfpack child
process synchronously inside one MCP call: ``_OpCtx.inline`` (default
``True``) and ``_OpCtx.gltfpack_exe``. See ``studio/modes/clay/agent/
dispatch.py``'s own module docstring's ``clay_op`` paragraphs and
``agent_clay_tools_ops._OpCtx``'s own docstring for the design these tests
pin.

Kept out of ``tests/modes/clay/test_agent_clay.py`` deliberately -- the same
rule ``test_agent_clay_door_types.py`` states for itself: that file carries
the user's own uncommitted work, and this session's own new tests go in
their own file instead, with their own minimal ``ctx`` double rather than an
import across files.

Some tests here exercise a different session's own concurrent work
(``kernels/mesh/readiness.py`` and three new ``clay_ops.OPS`` rows --
``clean-mesh``, ``recalc-normals``, ``decimate``); each says so in its own
docstring, in case a future run of this suite catches one of them mid-flight
again before that work has landed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from warlock.kernels.mesh import readiness
from warlock.studio.modes.clay import mode as clay_mode
from warlock.studio.modes.clay import ops as clay_ops
from warlock.studio.modes.clay.agent import dispatch as agent_clay

# --- a ctx double, the same minimal shape test_agent_clay.py's own _Ctx is --


class _Cache:
    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


class _Ctx:
    """The same minimal ``ctx`` double ``tests/modes/clay/test_agent_clay.py``
    uses -- duplicated here rather than imported, the same reason
    ``test_agent_clay_door_types.py`` gives for its own copy: that file
    carries the user's own uncommitted work and this one must not import
    from it.
    """

    def __init__(self, svc: Any = None) -> None:
        self.state = SimpleNamespace(clay=None)
        self.svc = svc
        self.cache = _Cache()
        self.toasts: list[tuple[str, str]] = []
        self.settings = SimpleNamespace()

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _payload(result: dict) -> Any:
    return json.loads(result["content"][0]["text"])


def _new_agent_tab(ctx: _Ctx, session: agent_clay.Session, generator: str = "box") -> int:
    result = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": generator})
    assert result["isError"] is False, result
    return _payload(result)["uid"]


def _history_len(ctx: _Ctx, session: agent_clay.Session) -> int:
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    return len(tab.doc.history.history())


# --- the bidirectional derivation gate, clay_validate's own profile enum ----


def test_every_readiness_profile_is_a_clay_validate_enum_option_and_vice_versa() -> None:
    tools = {t.name: t for t in agent_clay.tools()}
    enum = set(tools["clay_validate"].schema["properties"]["profile"]["enum"])
    assert enum == set(readiness.PROFILES)


def test_a_fourth_readiness_profile_reaches_the_agent_surface_with_no_edit_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live-gate half of the derivation claim -- the same shape
    ``test_a_thirteenth_generator_reaches_the_agent_surface_with_no_edit_here``
    and its three siblings in ``test_agent_clay.py`` already prove for their
    own registries (``bp.GENERATORS``, ``presets.ASSEMBLIES``, ``select.
    QUERIES``): monkeypatch a new entry into ``readiness.PROFILES``, restored
    automatically, and assert it shows up in ``clay_validate``'s own enum
    with no code here touched at all. Reuses an existing ``Profile`` value
    under a new key rather than constructing a fresh one -- this test is
    about the enum reaching the wire, not about what a ``Profile`` is made
    of.
    """
    existing = readiness.PROFILES[readiness.DEFAULT_PROFILE]
    monkeypatch.setitem(readiness.PROFILES, "fourth_profile", existing)
    tools = {t.name: t for t in agent_clay.tools()}
    enum = tools["clay_validate"].schema["properties"]["profile"]["enum"]
    assert "fourth_profile" in enum


def test_every_readiness_fix_is_a_clay_op_name() -> None:
    """Every entry of ``readiness.FIX_OPS`` names a real ``clay_ops.OPS``
    row -- the cross-registry gate that makes a Fix button, and an agent's
    own next ``clay_op`` call following ``clay_validate``'s own ``fix``
    field, actually runnable rather than a name with nothing behind it.

    Failed against this branch's tree before ``clean-mesh``,
    ``recalc-normals`` and ``decimate`` landed in
    ``studio/modes/clay/ops.py`` (a different session's own concurrent
    work) -- proof this gate does its job: it is exactly what would also
    catch a readiness fix naming an op that was never registered, or a typo
    between the two registries' spelling of the same op name.
    """
    op_names = {op.name for op in clay_ops.OPS}
    missing = sorted(readiness.FIX_OPS - op_names)
    assert not missing, f"readiness.FIX_OPS names ops clay_ops.OPS does not have: {missing}"


# --- clay_validate itself -----------------------------------------------------


def test_clay_validate_pushes_no_undo_step() -> None:
    """Read-only, the same shape ``clay_diagnose``/``clay_analyze`` already
    hold to: a successful call leaves the document's own undo history
    exactly as it stood."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    before = _history_len(ctx, session)
    result = agent_clay.call(ctx, session, "clay_validate", {})
    assert result["isError"] is False, result
    assert _history_len(ctx, session) == before


def test_an_unknown_profile_is_refused_naming_the_field() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    result = agent_clay.call(ctx, session, "clay_validate", {"profile": "not-a-real-profile"})
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["field"] == "profile"
    assert structured["recovery"] == "fix_arguments"
    assert structured["changed"] is False


def test_an_empty_document_answers_fail_status_rather_than_a_refusal() -> None:
    """A document with nothing left in it -- every creator tool places an
    object immediately, so this session's document is minted and then
    emptied by deleting the object ``clay_add_primitive`` just placed -- is
    not a malformed ``clay_validate`` call. Having nothing to check is
    itself the finding an agent asked for, so this answers a ``"fail"``
    status rather than refusing the call outright.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    deleted = agent_clay.call(ctx, session, "clay_delete", {"uids": [uid]})
    assert deleted["isError"] is False, deleted

    result = agent_clay.call(ctx, session, "clay_validate", {})
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["status"] == "fail"


def test_clay_validates_own_result_carries_the_same_json_twice() -> None:
    """The structured-results claim ``test_agent_clay.py``'s own module
    docstring pins for the rest of this fold, exercised for the new tool: a
    successful ``clay_validate`` answers through ``_json``, so its text
    block and its ``structuredContent`` carry the identical payload."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    result = agent_clay.call(ctx, session, "clay_validate", {})
    assert result["isError"] is False, result
    assert result["structuredContent"] == _payload(result)


# --- decimate: a background op that runs inline through _OpCtx --------------


def test_decimate_runs_its_gltfpack_subprocess_inline_as_one_undo_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``decimate`` is Clay's first ``clay_op`` row whose ``run`` spawns a
    real child process (gltfpack, via ``pipelines.optimize.simplify_bytes``)
    rather than only editing a mesh in memory. Interactively it would run on
    a task thread, the shape every blocking op in this codebase takes; an
    MCP ``clay_op`` call has no later frame to hand that off to, so
    ``_OpCtx.inline`` (default ``True``) makes ``ops._decimate`` call
    ``simplify_bytes`` synchronously instead, inside ``clay_ops.run``.
    ``optimize.simplify_bytes`` monkeypatched to an identity function proves
    the roundtrip runs inside this one call and folds into the one undo step
    every other op already gives -- with no real gltfpack subprocess needed
    to prove it, only a real *file* at ``gltfpack_exe`` (``ops._decimate``'s
    own ``exe.is_file()`` gate refuses before ever reaching
    ``simplify_bytes`` otherwise, which a bare ``Path("fake-gltfpack.exe")``
    with nothing on disk silently tripped the first time this test was
    written -- caught by the ``ran`` assertion below, not the top-level
    ``isError`` one: a per-object refusal inside ``clay_ops.run`` still
    answers a successful-looking ``clay_op`` reply with ``ran: false``).
    """
    from warlock.pipelines import optimize as pipe_optimize

    calls: list[dict[str, Any]] = []

    def _fake_simplify_bytes(data: bytes, **kwargs: Any) -> bytes:
        calls.append(kwargs)
        return data

    monkeypatch.setattr(pipe_optimize, "simplify_bytes", _fake_simplify_bytes)

    fake_exe = tmp_path / "fake-gltfpack.exe"
    fake_exe.write_bytes(b"")
    ctx = _Ctx(svc=SimpleNamespace(config=SimpleNamespace(gltfpack_exe=fake_exe)))
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    before = _history_len(ctx, session)

    result = agent_clay.call(ctx, session, "clay_op", {"name": "decimate"})
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["ran"] is True, payload
    assert calls, "decimate did not call optimize.simplify_bytes"
    assert _history_len(ctx, session) == before + 1
