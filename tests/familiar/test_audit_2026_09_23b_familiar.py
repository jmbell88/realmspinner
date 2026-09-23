"""Regression tests for the 2026-09-23 audit (second run), familiar findings
familiar-01 through familiar-06.

Each test's name is the claim; see the docstring for how it fails against
the pre-fix code (``git show HEAD:<path>`` for the modules this session
touched -- ``studio/assistant/ui.py``, ``studio/assistant/doors.py``,
``pipelines/llama.py``, ``service/familiar.py``, ``familiar/character_plan.py``).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from realmspinner import fetch, models
from realmspinner.familiar import character_plan
from realmspinner.kernels.mesh import document as bd
from realmspinner.pipelines import llama as llama_mod
from realmspinner.pipelines.llama import LlamaServer
from realmspinner.studio.assistant import doors as familiar_doors
from realmspinner.studio.assistant import ui as familiar_ui
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.panes import model_gate
from realmspinner.studio.state import ManualState

# --- a small local stand-in for tests/familiar/test_familiar_ui.py's own
# _FakeCtx/_FakeThreads -- not imported cross-file, since this file owns
# nothing there and the shape it needs is a handful of attributes. ----------


class _FakeThreads:
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], list] = {}

    def append(self, key, turn) -> None:
        self._data.setdefault(key, []).append(turn)

    def get(self, key):
        return tuple(self._data.get(key, ()))


class _FakeCtx:
    def __init__(self, *, mode: str = "home") -> None:
        doc = bd.ClayDoc()
        tab = clay_mode.ClayTab(doc=doc)
        clay_state = clay_mode.ClayState(docs=[tab], active_uid=tab.uid)
        from realmspinner.studio.modes.create.engine.state import CreateState

        self.state = SimpleNamespace(
            clay=clay_state, mode=mode, familiar=None, preview={},
            manual=ManualState(), create=CreateState(),
        )
        self.settings = SimpleNamespace()
        self.svc = SimpleNamespace(
            worker=SimpleNamespace(familiar=object()),
            config=SimpleNamespace(palette_dir=Path("___audit_2026_09_23b_no_palette___")),
        )
        self.toasts: list[str] = []
        self.familiar_threads = _FakeThreads()
        self._tab = tab
        self._pending: dict[str, tuple] = {}
        self.model_rows: list[dict] = []
        self.model_picks: set[str] = set()

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


def _character_plan_action() -> dict:
    return {
        "kind": "character_plan",
        "prompt": "make me a goblin in the swamp",
        "plan": {"family": "goblin", "theme": "swamp", "movements": ["walk"]},
        "overrides": {"family": "goblin", "theme": "swamp", "animations": {"walk": None}},
        "summary": {
            "species": "Goblin", "theme": "swamp", "movements": ["walk"],
            "directions": None, "cells": 24, "estimate_minutes": 3.5, "ignored": [],
        },
    }


# --- familiar-01 -------------------------------------------------------


def test_open_in_create_refusal_leaves_the_pending_plan_intact():
    """studio/assistant/ui.py's ``open_character_in_create`` used to clear
    ``ui.plan`` whether or not the draft actually landed -- a gated Create
    (``model_gate.mode_gate`` refusing) dropped the proposed plan along with
    the refusal sentence, leaving the user with neither a drafted brief nor
    the card to retry from. Gate Create by putting one of its NEEDS_ROWS
    rows in ``ctx.model_rows`` as absent, with no library work on the ctx to
    exempt it (``_library_has_work`` reads ``ctx.cache``, absent here)."""
    ctx = _FakeCtx(mode="home")
    ctx.model_rows = [{"row_key": "engine:trellis_gguf", "present": False, "size_gib": 1.0}]
    where, _blocked = model_gate.mode_gate(ctx, "create")
    assert where, "the gate must actually be refusing for this test to mean anything"

    ui = familiar_ui.ensure(ctx)
    ui.plan = _character_plan_action()

    familiar_ui.open_character_in_create(ctx)

    assert ui.plan is not None, "a refused draft must leave the plan card in place"
    turns = ctx.familiar_threads.get(familiar_ui.thread_key(ctx))
    assert turns[-1].text == model_gate.mode_reason(ctx, "create")


def test_open_in_create_acceptance_still_clears_the_plan(monkeypatch):
    """The flip side of the fix above: an *un*-gated Create must still clear
    the plan, exactly as it did before -- this is the same shape
    ``tests/familiar/test_familiar_ui.py::
    test_open_in_create_drafts_the_character_brief_with_the_plan_fields``
    already covers, kept here so a regression in the gate-read alone (not
    the draft call) is caught in this file too."""
    ctx = _FakeCtx(mode="home")
    ui = familiar_ui.ensure(ctx)
    ui.plan = _character_plan_action()

    monkeypatch.setattr(
        familiar_doors, "draft_in_create",
        lambda *a, **k: "Drafted in Create -- check the brief and press Generate.",
    )

    familiar_ui.open_character_in_create(ctx)

    assert ui.plan is None


# --- familiar-02 ---------------------------------------------------------


def _srv(tmp_path, **kwargs) -> LlamaServer:
    exe = tmp_path / "llama-server.exe"
    weights = tmp_path / "models" / "familiar" / models.FAMILIAR_GGUF_FILE
    return LlamaServer(
        exe, weights, 17973, key_dir=tmp_path / "keys", log_path=tmp_path / "familiar.log",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_idle_eviction_during_a_cold_start_health_poll_is_not_misclassified_as_a_gpu_lease(
    tmp_path, monkeypatch
):
    """The 2026-09-23 audit (familiar-02): ``ensure_started``'s health poll
    only stamped ``last_used`` at spawn and on a healthy 200 -- so a cold
    start slow enough to run several poll ticks still read as idle "since
    boot" once elapsed time passed ``idle_timeout``. ``Worker._maybe_evict_
    idle`` then stopped the very server that was still starting, clearing
    ``self._proc`` mid-poll -- and ``service.familiar._reason_for`` reads
    that exact "was stopped during startup" sentence as a GPU lease (the
    only *other* caller documented to clear ``_proc`` mid-poll), so a slow
    cold start was misreported to the user as another job holding the card.

    ``idle_timeout`` here is set far below the poll loop's own 0.1s tick, so
    the unfixed code (last_used stamped only at spawn) is idle by the second
    tick every time; the fix (last_used touched every tick) must never let
    that happen no matter how many 503s precede the 200.
    """
    from realmspinner.config import Config
    from realmspinner.db import JobStore
    from realmspinner.queue import Worker

    srv = _srv(tmp_path, idle_timeout=0.01)
    srv._resolve_exe().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_exe().write_bytes(b"")
    srv._resolve_weights().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_weights().write_bytes(b"")
    monkeypatch.setattr(
        llama_mod.fetch, "verify_manifest",
        lambda dest: fetch.Verification(dest=dest, status=fetch.VERIFY_UNKNOWN),
    )
    monkeypatch.setattr(srv, "_check_vram", lambda: None)
    monkeypatch.setattr(llama_mod, "_port_in_use", lambda port: False)
    monkeypatch.setattr(llama_mod.winjob, "assign", lambda pid: None)
    monkeypatch.setattr(llama_mod.winjob, "track", lambda pid, name: None)
    monkeypatch.setattr(srv, "_claim_port", lambda pid: None)
    monkeypatch.setattr(srv, "_pump", lambda: None)
    fake_proc = type("P", (), {"pid": 4242, "returncode": None, "poll": lambda self: None})()
    monkeypatch.setattr(llama_mod.subprocess, "Popen", lambda *a, **k: fake_proc)

    config = Config(
        data_dir=tmp_path / "assets", db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe", trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models",
    )
    worker = Worker(config, JobStore(config.db_path))
    worker.familiar = srv

    health_calls: list[str] = []

    async def fake_get(self, url, **kwargs):
        health_calls.append(url)
        # The idle sweep runs synchronously inside this fake, between two
        # poll ticks -- the real worker's own idle tick reaches exactly this
        # window in the app (see test_a_cold_start_still_loading_its_weights_
        # is_not_evicted_as_idle in tests/familiar/test_familiar.py for the
        # sibling, spawn-time-only version of this race).
        if len(health_calls) in (2, 3):
            await worker._maybe_evict_idle()
        if len(health_calls) < 4:
            return httpx.Response(503)
        return httpx.Response(200)

    monkeypatch.setattr(llama_mod.httpx.AsyncClient, "get", fake_get)

    await srv.ensure_started()

    assert len(health_calls) >= 4
    assert srv.running, "a server still cold-starting was evicted as idle mid-poll"


# --- familiar-03 -----------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_started_reaps_a_dead_child_without_blocking_the_loop_thread(
    tmp_path, monkeypatch
):
    """The 2026-09-23 audit (familiar-03): ``ensure_started`` called
    ``self._reap_if_dead()`` directly on the ``realmspinner-loop`` thread --
    ``_reap_if_dead`` -> ``stop()`` can run ``proc.terminate()``/``wait()``,
    ``proc.kill()``/``wait()`` and ``self._reader.join(timeout=5)`` in
    sequence, the same class of blocking call the 2026-09-18 audit
    (familiar-01) moved ``_check_manifest`` off this thread for. Same proof
    shape as that fix's own test just above
    (``test_ensure_started_runs_manifest_verification_off_the_loop_thread``):
    record which thread actually calls ``stop()`` (``_reap_if_dead``'s own
    caller once it sees a dead ``proc.poll()``).
    """
    srv = _srv(tmp_path, idle_timeout=300.0)

    class _FakeProc:
        pid = 1234
        returncode = 7

        def poll(self):
            return 7  # dead

    srv._proc = _FakeProc()
    srv._spawned_at = time.monotonic()

    calling_thread_ids: list[int] = []

    def _fake_stop() -> None:
        calling_thread_ids.append(threading.get_ident())
        srv._proc = None

    monkeypatch.setattr(srv, "stop", _fake_stop)

    loop_thread_id = threading.get_ident()
    # No exe on disk -- ensure_started refuses right after the reap, which
    # is all this test needs: the reap itself must have already happened
    # off this thread by the time that refusal is raised.
    with pytest.raises(RuntimeError, match="llama-server not found at"):
        await srv.ensure_started()

    assert calling_thread_ids, "stop() (via _reap_if_dead) was never called"
    assert loop_thread_id not in calling_thread_ids, (
        "_reap_if_dead ran stop() on the calling/loop thread instead of "
        "being offloaded with asyncio.to_thread"
    )


# --- familiar-04 -------------------------------------------------------


def test_manual_overview_lists_show_trash_among_familiars_navigable_destinations():
    """studio/assistant/doors.py's ``_NAV_COMMAND_KEYS`` includes
    ``"show-trash"`` -- Familiar can navigate there -- but
    docs/manual/20-overview.md's own sentence listing what "take me
    somewhere" can reach ("switches modes, opens the right Settings page,
    opens the Manual, a tour, the keyboard shortcuts list or the workspace
    layout picker") never mentions Trash. This file may not edit
    docs/manual/ (not among the files this fixer owns); it will pass once
    the orchestrator adds the sentence for show-trash, per familiar-04's own
    instruction in the audit."""
    from realmspinner.studio.assistant import doors as familiar_doors

    assert "show-trash" in familiar_doors._NAV_COMMAND_KEYS

    text = Path("docs/manual/20-overview.md").read_text(encoding="utf-8")
    marker = "switches modes, opens the right Settings page"
    start = text.find(marker)
    assert start != -1, "the navigate-door sentence itself moved or was reworded"
    # The sentence runs to its own period, "...whichever you asked for --".
    end = text.find("whichever you asked for", start)
    assert end != -1
    sentence = text[start:end]
    assert "trash" in sentence.lower(), (
        "the sentence describing where Familiar's navigate door can send "
        "you does not mention Trash, though show-trash is one of "
        "doors._NAV_COMMAND_KEYS -- familiar-04"
    )


# --- familiar-05 -----------------------------------------------------------


def test_a_missing_familiar_reason_offers_an_install_action_in_the_expanded_pane():
    """The 2026-09-23 audit (familiar-05): ``ui.reason`` was set on every
    refusal but never read anywhere in ``draw_expanded``, contrary to this
    module's own docstring and ``service.familiar``'s ("the pane can choose
    an icon or an action -- an Install... button"). ``install_missing_
    familiar`` is the action ``draw_expanded`` now calls for ``reason ==
    "missing"``; tested headless here since ``draw_expanded`` itself needs a
    live imgui frame (its own module docstring)."""
    ctx = _FakeCtx(mode="home")
    ui = familiar_ui.ensure(ctx)
    ui.reason = "missing"
    ui.message = "llama-server not found at ..."

    familiar_ui.install_missing_familiar(ctx)

    assert ctx.model_picks >= set(familiar_ui.MISSING_FAMILIAR_ROWS)
    from realmspinner.studio.modes.settings.ui.panes import app_settings

    assert ctx.state.preview[app_settings.CATEGORY_SLOT] == "models"
    assert ctx.state.mode == "settings"


# --- familiar-06 -------------------------------------------------------


def test_a_name_over_64_characters_is_dropped_and_named_in_the_dropped_list():
    """The 2026-09-23 audit (familiar-06): every sibling drop branch in
    ``character_plan.parse_plan`` (movement, theme, directions, size) has a
    "dropped and named" test; the over-64-character name branch (lines
    229-237) never got one. An evidence gap, not a defect -- this is
    expected to pass both before and after this session's other fixes, and
    is added purely to close the coverage hole the audit found."""
    options = {
        "families": [{"key": "goblin", "label": "Goblin", "themes": ["swamp"]}],
        "movements": ["idle"], "directions": [1], "size_range": (8, 256),
    }
    long_name = "G" * 65
    reply = json.dumps({"family": "goblin", "name": long_name})

    plan, dropped = character_plan.parse_plan(reply, options)

    assert plan is not None
    assert "name" not in plan
    assert {"kind": "name", "text": long_name[:64], "reason": "over 64 characters"} in dropped
