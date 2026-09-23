"""Regression tests for the 2026-09-23 audit's agents-familiar findings.

Each test here targets exactly one finding and is written to fail against
the tree as it stood before this fixer's changes:

* agents-01 -- ``realmspinner_status`` after a task-mode result has already
  been fetched via the ``status`` RPC op used to still claim it "is still
  waiting" (``op.fetched`` was set but ``op.delivered`` never was).
* agents-03 -- task-mode ``tools/call`` coerced a malformed, falsy
  ``arguments`` (``[]``/``0``/``""``) into ``{}`` instead of refusing with
  -32602, unlike the synchronous path and ``prompts/get``.
* familiar-01 -- landing a Familiar build after only a tab switch (no
  document edit at all) reported the generic "the document changed --
  preview again" sentence instead of ``familiar_preview.apply()``'s own
  "not the one in front" sentence for the identical situation.
* familiar-02 -- ``service.familiar._reason_for`` fell through to "http" for
  three ``LlamaServer`` sentences it never matched: "exited during startup",
  "was stopped during startup" (a GPU lease taken during the health poll),
  and every ``_reclaim_port`` sentence.

See "the 2026-09-23 audit, finding <id>" in each test's docstring rather
than this file's own name, per this repo's rule against citing audit/plan
file names from ``src/``.
"""

from __future__ import annotations

import json
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.mcp import protocol as p
from realmspinner.mcp import rpc
from realmspinner.service import familiar as svc_familiar
from realmspinner.studio import agent_host
from realmspinner.studio.assistant import ui as familiar_ui
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.tasks import Done

# --- agents-01 ---------------------------------------------------------------


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.settings = SimpleNamespace()

    def toast(self, message: str, level: str = "info") -> None:
        pass

    def submit(self, key, fn, *args, tag=None, **kwargs) -> bool:
        import threading

        threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()
        return True


def _bare_host() -> agent_host.AgentHost:
    host = agent_host.AgentHost(_Ctx(), Path("unused-for-these-tests"))
    host._queue = queue.Queue()
    return host


def test_realmspinner_status_does_not_claim_a_fetched_task_mode_result_is_still_waiting() -> None:
    """The 2026-09-23 audit, finding agents-01.

    Once ``_task_status`` has handed a task-mode operation's terminal result
    out over the ``status`` RPC op, ``realmspinner_status`` must not still say
    "Its result is still waiting -- sending the same call again will hand it
    back" -- that call has already happened, and (per ``_call_task``'s own
    docstring) a task-mode resend mints a fresh operation and reruns the
    tool rather than replaying anything.
    """
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()

    header = host._call_task(session, calls, "clay_scene", {})
    op_id = header["operation_id"]
    host.pump(budget=1.0)

    # Fetch once via the RPC v1 `status` op -- what a real task-mode client
    # does for `tasks/get`. (An empty session refusing `clay_scene` with
    # isError=True is expected here and beside the point -- what matters is
    # that the fetch happened at all.)
    reply = host._task_status(calls, op_id)
    _status_header, body = rpc.split_reply(reply)
    assert json.loads(body).get("content")

    op = calls.get(op_id)
    assert op.fetched is True

    status_result = host._call(session, calls, agent_host.STATUS_TOOL, {"operation_id": op_id})
    payload = json.loads(status_result["content"][0]["text"])

    assert payload["delivered"] is True, (
        "a task-mode operation must be marked delivered once its result has "
        "gone out over the status RPC op, or the note below keeps firing"
    )
    assert "note" not in payload, (
        f"realmspinner_status still claims a fetched result 'is still waiting': {payload}"
    )


# --- agents-03 ---------------------------------------------------------------


def test_task_mode_tools_call_refuses_non_dict_arguments() -> None:
    """The 2026-09-23 audit, finding agents-03.

    ``arguments: []`` (a list, not an object) must be refused with -32602
    "'arguments' must be an object", the same as the synchronous
    ``tools/call`` path and ``prompts/get`` -- not silently coerced to
    ``{}`` by an ``or {}`` default and run anyway.
    """
    captured: dict = {}

    def call_tool_task(name, arguments):
        captured["arguments"] = arguments
        return ("op-1", "working")

    state = p.BridgeEra()
    item = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "clay_scene",
            "arguments": [],
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientCapabilities": {
                    "extensions": {"io.modelcontextprotocol/tasks": {}}
                },
            },
        },
    }

    reply = p._dispatch_one(
        item,
        state,
        catalogue={"tools": [], "server": {"name": "realmspinner", "version": "0.0.0"}},
        call_tool=lambda name, args: (_ for _ in ()).throw(AssertionError("sync path called")),
        call_tool_task=call_tool_task,
    )

    assert "arguments" not in captured, (
        f"call_tool_task was invoked with coerced arguments {captured.get('arguments')!r} "
        "instead of the request being refused"
    )
    decoded = json.loads(reply)
    assert decoded["error"]["code"] == -32602
    assert "arguments" in decoded["error"]["message"]


# --- familiar-01 ---------------------------------------------------------------


class _FakeThreads:
    def __init__(self) -> None:
        self._data: dict = {}

    def append(self, key, turn) -> None:
        self._data.setdefault(key, []).append(turn)

    def get(self, key):
        return tuple(self._data.get(key, ()))

    def drop(self, mode, uid) -> None:
        self._data.pop((mode, uid), None)


class _FakeFamiliarCtx:
    def __init__(self, doc, mode: str = "clay") -> None:
        tab = clay_mode.ClayTab(doc=doc)
        other_doc = bd.ClayDoc()
        other_tab = clay_mode.ClayTab(doc=other_doc)
        clay_state = clay_mode.ClayState(docs=[tab, other_tab], active_uid=tab.uid)
        self.state = SimpleNamespace(clay=clay_state, mode=mode, familiar=None, preview={})
        self.settings = SimpleNamespace()
        self.svc = SimpleNamespace(worker=SimpleNamespace(familiar=object()))
        self.toasts: list[str] = []
        self.familiar_threads = _FakeThreads()
        self._tab = tab
        self._other_tab = other_tab
        self._pending: dict = {}
        self.clay_view = SimpleNamespace(cleared=0, previewed=None, grabbing=False)
        self.clay_view.clear_preview = lambda: setattr(
            self.clay_view, "cleared", self.clay_view.cleared + 1
        )
        self.clay_view.set_preview = lambda diff, scratch: setattr(
            self.clay_view, "previewed", (diff, scratch)
        )

    @property
    def tab(self):
        return self._tab

    def submit(self, key, fn, *args, tag=None, **kwargs) -> bool:
        if key in self._pending:
            return False
        self._pending[key] = (fn, args, kwargs, tag)
        return True

    def busy(self, key) -> bool:
        return key in self._pending

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append(message)


def test_landing_a_build_after_switching_tabs_says_switch_to_it_not_that_the_document_changed() -> (
    None
):
    """The 2026-09-23 audit, finding familiar-01.

    Switching to another open tab while Familiar is building must report the
    same "that document is not the one in front -- switch to it, then
    preview again" sentence ``familiar_preview.apply()`` already uses for
    the identical situation -- not the generic "the document changed --
    preview again" sentence, which is wrong here: nothing about the document
    moved, only which tab is on screen.
    """
    doc = bd.ClayDoc()
    ctx = _FakeFamiliarCtx(doc, mode="clay")
    calls = [{"name": "clay_add_primitive", "arguments": {"generator": "box", "name": "crate"}}]

    done = Done(
        key=familiar_ui.BUILD_KEY,
        result=calls,
        tag={"thread_key": ("clay", ctx.tab.uid), "tab_uid": ctx.tab.uid},
    )
    familiar_ui.on_task_done(ctx, done)
    assert familiar_ui.LAND_KEY in ctx._pending

    # Between phase one and phase two, the user only switches to the other,
    # unrelated tab -- nothing about ctx.tab.doc moves.
    ctx.state.clay.active_uid = ctx._other_tab.uid

    fn, args, kwargs, tag = ctx._pending.pop(familiar_ui.LAND_KEY)
    result = fn(*args, **kwargs)
    familiar_ui.on_task_done(ctx, Done(key=familiar_ui.LAND_KEY, result=result, tag=tag))

    ui = familiar_ui.ensure(ctx)
    assert ui.message is not None
    assert "not the one in front" in ui.message, (
        f"expected apply()'s own 'not the one in front' sentence, got: {ui.message!r}"
    )
    assert "the document changed" not in ui.message


# --- familiar-02 ---------------------------------------------------------------


def test_a_gpu_lease_taken_during_the_health_poll_is_classified_as_lease_not_http() -> None:
    """The 2026-09-23 audit, finding familiar-02.

    ``LlamaServer.ensure_started``'s health poll raises exactly
    "llama-server was stopped during startup" when ``stop_for_gpu_job`` (see
    its own docstring: it sets ``_leased = True`` before calling ``stop()``)
    clears ``self._proc`` out from under a poll in progress. That is a lease
    taken, not an http-shaped failure, and must classify as ``"lease"``.
    """
    assert svc_familiar._reason_for("llama-server was stopped during startup") == "lease"


def test_llama_server_exiting_during_startup_is_classified_as_unhealthy_not_http() -> None:
    """The 2026-09-23 audit, finding familiar-02: the child process dying on
    its own during the health poll is the same "never became healthy" story
    as the existing "did not become healthy in time" branch."""
    assert (
        svc_familiar._reason_for("llama-server exited during startup (code 3221225477)")
        == "unhealthy"
    )


@pytest.mark.parametrize(
    "message",
    [
        "port 8971 is already in use, probably by an orphaned llama-server.exe "
        "left behind by a previous crash. Run `Get-Process llama-server` and "
        "stop it before retrying.",
        "port 8971 is held by pid 1234 (C:\\Windows\\notepad.exe), which is not "
        "this Realmspinner's llama-server (C:\\llama-server.exe). Stop it or "
        "change REALMSPINNER_FAMILIAR_PORT before retrying.",
        "port 8971 is held by a llama-server (pid 1234) that this Realmspinner "
        "did not start. Stop it, or change REALMSPINNER_FAMILIAR_PORT, before "
        "retrying.",
        "port 8971 is held by a llama-server started by a Realmspinner that is "
        "still running (pid 1234). Close it, or give this one its own "
        "REALMSPINNER_FAMILIAR_PORT, before retrying.",
        "port 8971 is still held after terminating pid 1234",
    ],
)
def test_every_reclaim_port_sentence_is_classified_as_unhealthy_not_http(message: str) -> None:
    """The 2026-09-23 audit, finding familiar-02: every sentence
    ``LlamaServer._reclaim_port`` raises means "the server could not be
    reached" to a caller, the same story as "did not become healthy in
    time" -- not the generic "http" bucket."""
    assert svc_familiar._reason_for(message) == "unhealthy"


# --- familiar-04 (docstring only; a light guard against it regressing) -------


def test_check_card_sha_docstring_no_longer_calls_contract_py_future_work() -> None:
    """The 2026-09-23 audit, finding familiar-04: ``_check_card_sha``'s
    docstring used to say T3 (``familiar/contract.py``) "will supply" a real
    ``expected_card_sha`` -- future tense for work that has since landed
    (``contract.CARDS``/``contract.card_sha`` are live, and
    ``models.FamiliarModel.card_shas`` is populated for the Clay pin)."""
    from realmspinner.pipelines import llama

    doc = llama.LlamaServer._check_card_sha.__doc__ or ""
    assert "will supply" not in doc
    assert "until it lands" not in doc
