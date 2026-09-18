"""The dev-only Familiar recorder: ``service/familiar_log.py``.

Gated by :data:`familiar_log.ENV_KEY` (``WARLOCK_FAMILIAR_LOG``) -- off by
default, so the first test proves the off-state is truly silent (no file at
all, not just an empty one) before the rest turn it on with
``monkeypatch.setenv``. Every test calls :func:`familiar_log.reset` first so
an earlier test's open session file (and its cached path) never leaks into
the next one's ``WARLOCK_HOME``.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from warlock.familiar import llama_client
from warlock.service import familiar as svc_familiar
from warlock.service import familiar_log


class _FakeSvc:
    """The same shape ``tests/test_familiar_service.py``'s own fake uses:
    just ``worker.familiar`` and a synchronous ``call_on_loop``."""

    def __init__(self) -> None:
        self.worker = SimpleNamespace(familiar=object())

    def call_on_loop(self, coro_factory, timeout: float = 30.0):
        return asyncio.run(coro_factory())


def _lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_record_writes_no_file_when_the_env_var_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv(familiar_log.ENV_KEY, raising=False)
    monkeypatch.setenv("WARLOCK_HOME", str(tmp_path))
    familiar_log.reset()

    familiar_log.record("submit", submit_kind="chat", prompt="hello")

    assert not (tmp_path / "familiar-log").exists()


def test_an_unwritable_log_directory_does_not_raise(monkeypatch, tmp_path):
    """A file where the log directory should be (``_dir`` monkeypatched to
    point under it) must never turn a logging attempt into a crash -- the
    module docstring's "a logger must never break a chat"."""
    monkeypatch.setenv(familiar_log.ENV_KEY, "1")
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(familiar_log, "_dir", lambda: blocker / "familiar-log")
    familiar_log.reset()

    familiar_log.record("submit", submit_kind="chat", prompt="hello")  # must not raise


def test_call_failure_records_the_error_and_still_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("WARLOCK_HOME", str(tmp_path))
    monkeypatch.setenv(familiar_log.ENV_KEY, "1")
    familiar_log.reset()

    async def failing_chat(server, messages, *, slot, sampling, skill=None,
                            expected_card_sha=None, response_format=None, transport=None):
        raise RuntimeError(
            "Familiar cannot start while a GPU job holds the card -- "
            "it will restart on your next message."
        )

    monkeypatch.setattr(llama_client, "chat", failing_chat)
    svc = _FakeSvc()

    with pytest.raises(svc_familiar.FamiliarRefusal) as excinfo:
        svc_familiar._call(
            svc,
            [{"role": "user", "content": "hi"}],
            skill=None,
            sampling={},
            expected_card_sha=None,
        )

    path = familiar_log._session_path()
    lines = _lines(path)
    assert lines[-1]["kind"] == "request"
    assert lines[-1]["error_type"] == "FamiliarRefusal"
    assert lines[-1]["reason"] == "lease"
    assert excinfo.value.reason == "lease"


def test_submit_request_and_outcome_share_one_exchange_id(tmp_path, monkeypatch):
    """One real Send round trip -- ``submit_chat`` on the frame thread, its
    closure on the worker, ``on_task_done`` back on the frame -- must write a
    submit, the request ``_call`` logs and an outcome under one exchange id,
    so a reader of the jsonl file can group one user turn's whole story back
    together. Driven through ``familiar_ui`` rather than hand-written records:
    the id reaching the worker thread through ``run()``'s closure is the part
    that can silently break."""
    from test_familiar_ui import _FakeCtx

    from warlock.studio.assistant import ui as familiar_ui
    from warlock.studio.tasks import Done

    monkeypatch.setenv("WARLOCK_HOME", str(tmp_path))
    monkeypatch.setenv(familiar_log.ENV_KEY, "1")
    familiar_log.reset()

    async def fake_chat(server, messages, *, slot, sampling, skill=None,
                         expected_card_sha=None, response_format=None, transport=None):
        return "hello back"

    monkeypatch.setattr(llama_client, "chat", fake_chat)

    def fake_ask(svc, prompt, **_kwargs):
        reply = svc_familiar._call(
            _FakeSvc(),
            [{"role": "user", "content": prompt}],
            skill=None,
            sampling={},
            expected_card_sha=None,
        )
        return svc_familiar.Answer(skill="chat", text=reply)

    monkeypatch.setattr(svc_familiar, "ask", fake_ask)
    ctx = _FakeCtx(mode="home")

    assert familiar_ui.submit_chat(ctx, "hi") is True
    fn, _args, _kwargs, tag = ctx._pending[familiar_ui.CHAT_KEY]
    result = fn()  # the worker thread running the submitted closure
    ctx.finish(familiar_ui.CHAT_KEY)
    familiar_ui.on_task_done(ctx, Done(key=familiar_ui.CHAT_KEY, result=result, tag=tag))

    lines = _lines(familiar_log._session_path())
    assert sorted(line["kind"] for line in lines) == ["outcome", "request", "submit"]
    assert {line["exchange"] for line in lines} == {tag["exchange"]}
    request = next(line for line in lines if line["kind"] == "request")
    assert request["reply"] == "hello back"
