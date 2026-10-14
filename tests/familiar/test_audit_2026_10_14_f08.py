"""The 2026-10-14 audit, fixer F08: the Familiar core's seven findings.

Each test names the claim it pins and fails against the code the audit read.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import httpx
import pytest

from realmspinner.familiar import contract, doors, embed_client, llama_client, retrieval, router
from realmspinner.pipelines import llama as pipelines_llama
from realmspinner.service import familiar as svc_familiar
from realmspinner.service.familiar import FamiliarRefusal


def _turn(role: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, text=text)


# --- familiar-core-03 -------------------------------------------------------


def test_chat_history_drops_oldest_turns_until_it_fits_the_slot():
    """A thread kept by turn count alone lets one oversized turn ride in every
    later request. Budgeted by size too, the oversized turn (and anything older
    than it) is forgotten, and the newest turns that fit are all kept."""
    history = (
        _turn("user", "first question"),
        _turn("familiar", "first answer"),
        _turn("user", "x" * 200_000),  # the failed paste, still in the thread
        _turn("familiar", "short reply"),
        _turn("user", "a small follow-up"),
    )
    messages = contract.build_chat_messages("and now?", history)
    assert not any(len(m["content"]) > 100_000 for m in messages)
    total = sum(len(m["content"]) for m in messages)
    assert total <= contract.CHAT_PROMPT_CHAR_BUDGET
    # Newest turns survive, oldest are what is forgotten, order is preserved.
    texts = [m["content"] for m in messages]
    assert texts[-1] == "and now?"
    assert "a small follow-up" in texts
    assert "short reply" in texts
    assert "first question" not in texts


def test_a_chat_prompt_past_the_door_cap_is_refused_before_any_request(monkeypatch):
    """No Familiar door bounded a prompt's length: a long paste went straight
    into a request the 8192-token slot could not hold."""
    called = []

    async def fake_chat(*args, **kwargs):
        called.append(True)
        return "never"

    monkeypatch.setattr(svc_familiar.llama_client, "chat", fake_chat)

    class _Svc:
        worker = SimpleNamespace(familiar=object())

        def call_on_loop(self, factory, timeout=30.0):
            return asyncio.run(factory())

    huge = "y" * 100_000
    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.chat_reply(_Svc(), huge)
    assert excinfo.value.reason == "too_large"
    with pytest.raises(FamiliarRefusal) as excinfo:
        svc_familiar.ask(_Svc(), huge, mode="home", history=())
    assert excinfo.value.reason == "too_large"
    assert called == []


# --- familiar-core-08 -------------------------------------------------------


@pytest.mark.parametrize("bad", [["image"], {"k": "image"}, 3, None])
def test_parse_draft_returns_none_for_a_non_string_asset_type(bad):
    """``asset_type not in keys`` hashes its operand: a list or dict raised
    TypeError out of a function documented as never raising."""
    import json

    text = json.dumps({"asset_type": bad, "prompt": "a sword"})
    assert doors.parse_draft(text, (("image", "Image"),)) is None


def test_a_clay_call_whose_name_is_a_list_is_refused_not_raised():
    reply = '```json\n{"calls": [{"name": ["clay_add"], "args": {}}]}\n```'
    calls, error = svc_familiar._gate_reply("clay", reply)
    assert calls is None
    assert "outside Clay's trained vocabulary" in error


# --- familiar-core-07 -------------------------------------------------------


class _EmbedServer:
    def __init__(self, key_path):
        self.key_path = key_path

    base_url = "http://127.0.0.1:9998"

    async def ensure_started(self):
        pass

    def touch(self):
        pass


async def test_an_embedding_with_an_unrepresentable_number_is_a_bad_response(tmp_path):
    key = tmp_path / "k.key"
    key.write_text("k\n", encoding="utf-8")
    vector = [0.5] * 767 + [10**400]
    # json.dumps writes the integer out in full; the server's reply carries it.
    import json

    body = json.dumps({"data": [{"index": 0, "embedding": vector}]})
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=body.encode()))
    with pytest.raises(embed_client.EmbedUnavailable):
        await embed_client.embed(_EmbedServer(key), ["a"], role="query", transport=transport)


# --- familiar-core-05 / -06 -------------------------------------------------


class _ChatServer:
    def __init__(self, key_path):
        self.key_path = key_path

    base_url = "http://127.0.0.1:9999"

    async def ensure_started(self, *, expected_card_sha=None):
        pass

    def touch(self):
        pass


def _chat_server(tmp_path):
    key = tmp_path / "k.key"
    key.write_text("k\n", encoding="utf-8")
    return _ChatServer(key)


async def _chat(tmp_path, response: httpx.Response, **kw):
    return await llama_client.chat(
        _chat_server(tmp_path),
        [{"role": "user", "content": "hi"}],
        slot=1,
        sampling=contract.SAMPLING["chat"],
        transport=httpx.MockTransport(lambda r: response),
        **kw,
    )


async def test_a_non_utf8_reply_body_is_a_runtime_error_not_a_value_error(tmp_path):
    """``UnicodeDecodeError`` is a ``ValueError``; ``_call`` files every
    ``ValueError`` as ``too_large``, so a broken server read as "does not fit"."""
    with pytest.raises(RuntimeError, match="not valid UTF-8"):
        await _chat(tmp_path, httpx.Response(200, content=b"\xff\xfe{not utf8"))


async def test_a_null_content_reply_is_a_runtime_error(tmp_path):
    body = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
    with pytest.raises(RuntimeError, match="no message content"):
        await _chat(tmp_path, httpx.Response(200, json=body))


async def test_a_reply_cut_by_max_tokens_is_marked_as_cut_off(tmp_path):
    body = {"choices": [{"message": {"content": "The answer begins"}, "finish_reason": "length"}]}
    text = await _chat(tmp_path, httpx.Response(200, json=body))
    assert text.startswith("The answer begins")
    assert text != "The answer begins"
    assert llama_client.TRUNCATED_NOTE in text


async def test_a_constrained_reply_is_never_annotated_when_cut(tmp_path):
    """A router/door reply is parsed as JSON; a note appended to it would only
    turn a clean parse failure into a different one."""
    body = {"choices": [{"message": {"content": '{"skill": "ma'}, "finish_reason": "length"}]}
    text = await _chat(
        tmp_path,
        httpx.Response(200, json=body),
        response_format={"type": "json_schema", "json_schema": {"schema": {}}},
    )
    assert text == '{"skill": "ma'


async def test_a_complete_reply_is_returned_unchanged(tmp_path):
    body = {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}
    assert await _chat(tmp_path, httpx.Response(200, json=body)) == "done"


# --- familiar-core-02 -------------------------------------------------------


def test_a_capped_regrouped_citation_keeps_the_best_scoring_chunk():
    """Hit chunks of one section were joined in document order and cut at the
    cap, so earlier weak hits crowded out the best-scoring chunk."""

    def chunk(text: str) -> retrieval.Chunk:
        return retrieval.Chunk("13-x", 13, "X", "sec", "13 X > sec", text)

    filler = " ".join(["filler"] * 400)
    chunks = [chunk(f"weak{i} {filler}") for i in range(4)] + [chunk("BESTHIT " + filler)]
    index = retrieval.Index(chunks, [], [], [], 0.0, retrieval.Counter(), len(chunks))
    # indices are best first: the last chunk of the document ranks first.
    (citation,) = index.citations([4, 0, 1, 2, 3])
    assert "BESTHIT" in citation.text
    assert retrieval._whitespace_token_count(citation.text) <= retrieval._MAX_CITATION_TOKENS + 1
    # Kept chunks read in document order, not rank order.
    assert citation.text.index("weak0") < citation.text.index("BESTHIT")


# --- familiar-core-10 -------------------------------------------------------


def test_router_slot_plan_docstring_names_the_real_llama_constants():
    doc = router.__doc__
    assert "llama.py:62" not in doc and "llama.py:63" not in doc
    assert "PARALLEL_SLOTS" in doc and "CTX_SIZE" in doc
    assert f"``PARALLEL_SLOTS = {pipelines_llama.PARALLEL_SLOTS}``" in doc
    assert f"``CTX_SIZE = {pipelines_llama.CTX_SIZE}``" in doc
    # The sampling comment named a model that is no longer the pin.
    assert "re-picked against Qwen3-VL" not in inspect.getsource(contract)
