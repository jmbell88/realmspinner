"""Phase 2's HTTP half: ``familiar/embed_client.py``.

Everything runs against ``httpx.MockTransport`` and a fake server object
duck-typed to ``LlamaServer`` -- no subprocess, no network (the one real-card
embedding test lives with the lane that owns the real server).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from realmspinner.familiar import embed_client


class _FakeServer:
    def __init__(self, key_path, *, start_error: Exception | None = None):
        self._key_path = key_path
        self._start_error = start_error
        self.touches = 0
        self.starts = 0

    @property
    def base_url(self) -> str:
        return "http://127.0.0.1:9998"

    @property
    def key_path(self):
        return self._key_path

    async def ensure_started(self) -> None:
        self.starts += 1
        if self._start_error is not None:
            raise self._start_error

    def touch(self) -> None:
        self.touches += 1


def _server(tmp_path, key="embed-key", **kw):
    key_path = tmp_path / "embed-9998.key"
    key_path.write_text(key + "\n", encoding="utf-8")
    return _FakeServer(key_path, **kw)


def _vec(seed: int, dim: int = 768, norm: float = 1.0) -> list[float]:
    v = np.random.default_rng(seed).standard_normal(dim)
    return (v / np.linalg.norm(v) * norm).tolist()


def _echo(*, shuffle: bool = False, dim: int = 768, drop: int = 0):
    """A handler answering ``/v1/embeddings`` with a deterministic vector per
    input *position-in-text* (seeded by the prompt's own hash-free content: its
    length and first characters), recording each request body."""
    bodies: list[dict] = []
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        bodies.append(body)
        items = [
            {"object": "embedding", "index": i, "embedding": _vec(_seed(t), dim)}
            for i, t in enumerate(body["input"])
        ]
        if drop:
            items = items[:-drop]
        if shuffle:
            items = items[::-1]
        return httpx.Response(200, json={"data": items})

    return handler, bodies, requests


def _seed(prompt: str) -> int:
    return sum(ord(c) * (i + 1) for i, c in enumerate(prompt)) % 100003


def _expected(prompt: str) -> np.ndarray:
    v = np.asarray(_vec(_seed(prompt)), dtype=np.float32)
    return v / np.linalg.norm(v)


# --- the prompts ------------------------------------------------------------


async def test_a_query_is_sent_with_the_exact_model_card_query_prefix(tmp_path):
    handler, bodies, _ = _echo()
    await embed_client.embed(
        _server(tmp_path),
        ["where do models go"],
        role="query",
        transport=httpx.MockTransport(handler),
    )
    assert bodies == [
        {"input": ["task: search result | query: where do models go"], "model": "embeddinggemma"}
    ]


async def test_a_document_pair_is_sent_with_the_title_and_text_prefix(tmp_path):
    handler, bodies, _ = _echo()
    await embed_client.embed(
        _server(tmp_path),
        [("Clay > Lathe", "Spin a profile."), "bare text", ("", "blank title")],
        role="document",
        transport=httpx.MockTransport(handler),
    )
    assert bodies[0]["input"] == [
        "title: Clay > Lathe | text: Spin a profile.",
        "title: none | text: bare text",
        "title: none | text: blank title",
    ]


async def test_a_query_must_be_a_plain_string_and_the_role_must_be_known(tmp_path):
    handler, _, requests = _echo()
    transport = httpx.MockTransport(handler)
    with pytest.raises(TypeError):
        await embed_client.embed(_server(tmp_path), [("t", "x")], role="query", transport=transport)
    with pytest.raises(ValueError, match="role"):
        await embed_client.embed(_server(tmp_path), ["x"], role="doc", transport=transport)
    assert requests == []


# --- batching and ordering --------------------------------------------------


async def test_texts_go_out_in_batches_of_sixteen_and_come_back_in_order(tmp_path):
    handler, bodies, _ = _echo()
    texts = [f"chunk {i}" for i in range(40)]
    out = await embed_client.embed(
        _server(tmp_path), texts, role="document", transport=httpx.MockTransport(handler)
    )
    assert [len(b["input"]) for b in bodies] == [16, 16, 8]
    assert out.shape == (40, 768) and out.dtype == np.float32
    for i in (0, 15, 16, 39):
        np.testing.assert_allclose(
            out[i], _expected(embed_client.format_document(texts[i])), atol=1e-6
        )


async def test_the_response_is_reordered_by_each_items_index(tmp_path):
    handler, _, _ = _echo(shuffle=True)
    texts = ["alpha", "beta", "gamma"]
    out = await embed_client.embed(
        _server(tmp_path), texts, role="document", transport=httpx.MockTransport(handler)
    )
    for i, t in enumerate(texts):
        np.testing.assert_allclose(out[i], _expected(embed_client.format_document(t)), atol=1e-6)


async def test_rows_come_back_unit_norm_even_if_the_server_did_not_normalise(tmp_path):
    def handler(request):
        n = len(json.loads(request.content)["input"])
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": _vec(i, norm=7.0)} for i in range(n)]}
        )

    out = await embed_client.embed(
        _server(tmp_path), ["a", "b"], role="query", transport=httpx.MockTransport(handler)
    )
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


async def test_no_texts_returns_an_empty_matrix_without_starting_the_server(tmp_path):
    server = _server(tmp_path)
    out = await embed_client.embed(server, [], role="document")
    assert out.shape == (0, 768) and out.dtype == np.float32
    assert server.starts == 0


# --- the server door --------------------------------------------------------


async def test_the_api_key_is_read_from_the_key_file_and_sent_as_a_bearer_header(tmp_path):
    handler, _, requests = _echo()
    await embed_client.embed(
        _server(tmp_path, key="the-real-embed-key"),
        ["x"],
        role="query",
        transport=httpx.MockTransport(handler),
    )
    assert requests[0].headers["authorization"] == "Bearer the-real-embed-key"
    assert requests[0].url.path == "/v1/embeddings"


async def test_every_batch_touches_the_server_so_a_long_index_is_not_evicted_as_idle(tmp_path):
    handler, _, _ = _echo()
    server = _server(tmp_path)
    await embed_client.embed(
        server,
        [f"t{i}" for i in range(33)],
        role="document",
        transport=httpx.MockTransport(handler),
    )
    assert server.starts == 1
    assert server.touches >= 1 + 3  # before the first batch, and after each of three


async def test_the_client_never_trusts_the_proxy_environment(tmp_path, monkeypatch):
    """An HTTP(S)_PROXY without a NO_PROXY entry would route every Manual
    chunk and Library prompt through the proxy; the TRELLIS clients carry the
    same guard (tests/pipelines/test_trellis.py)."""
    seen: list[dict] = []
    real = httpx.AsyncClient

    class _Spy(real):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _Spy)
    handler, _, _ = _echo()
    await embed_client.embed(
        _server(tmp_path), ["x"], role="query", transport=httpx.MockTransport(handler)
    )
    assert seen and all(kw.get("trust_env") is False for kw in seen)


def test_every_async_client_in_the_module_sets_trust_env_false_by_ast():
    tree = ast.parse(Path(embed_client.__file__).read_text(encoding="utf-8"))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "AsyncClient"
    ]
    assert calls
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert isinstance(kw.get("trust_env"), ast.Constant) and kw["trust_env"].value is False


# --- refusals ---------------------------------------------------------------


async def test_a_short_response_is_refused_with_a_clear_error(tmp_path):
    handler, _, _ = _echo(drop=1)
    with pytest.raises(
        embed_client.EmbedBadResponse, match="asked for 3 embeddings, the server returned 2"
    ):
        await embed_client.embed(
            _server(tmp_path), ["a", "b", "c"], role="query", transport=httpx.MockTransport(handler)
        )


async def test_a_wrong_dimension_response_is_refused_with_a_clear_error(tmp_path):
    handler, _, _ = _echo(dim=256)
    with pytest.raises(embed_client.EmbedBadResponse, match=r"expected 768-d embeddings.*\(256,\)"):
        await embed_client.embed(
            _server(tmp_path), ["a"], role="query", transport=httpx.MockTransport(handler)
        )


@pytest.mark.parametrize(
    "body",
    [
        {"data": [{"index": 0, "embedding": "nope"}]},
        {"data": [{"index": 5, "embedding": _vec(1)}]},
        {"data": [{"index": 0, "embedding": [0.0] * 768}]},
        {"data": "x"},
        {"nodata": []},
        [1, 2],
    ],
)
async def test_a_malformed_or_degenerate_200_body_is_an_embed_unavailable(tmp_path, body):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    with pytest.raises(embed_client.EmbedUnavailable):
        await embed_client.embed(_server(tmp_path), ["a"], role="query", transport=transport)


async def test_two_items_claiming_one_index_is_refused(tmp_path):
    body = {"data": [{"index": 0, "embedding": _vec(1)}, {"index": 0, "embedding": _vec(2)}]}
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    with pytest.raises(embed_client.EmbedBadResponse, match="index 0"):
        await embed_client.embed(_server(tmp_path), ["a", "b"], role="query", transport=transport)


async def test_a_non_json_200_body_is_an_embed_unavailable(tmp_path):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text="<html>wedged</html>"))
    with pytest.raises(embed_client.EmbedUnavailable, match="non-JSON"):
        await embed_client.embed(_server(tmp_path), ["a"], role="query", transport=transport)


async def test_an_oversize_response_is_refused_while_it_streams(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_client, "MAX_RESPONSE_BYTES", 1000)
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"x" * 5000))
    with pytest.raises(embed_client.EmbedUnavailable, match="before it finished"):
        await embed_client.embed(_server(tmp_path), ["a"], role="query", transport=transport)


async def test_a_refusing_server_is_an_embed_unavailable_and_its_body_is_not_in_the_message(
    tmp_path,
):
    secret = "my private prompt about dragons"
    transport = httpx.MockTransport(lambda r: httpx.Response(500, text=f"failed on: {secret}"))
    with pytest.raises(embed_client.EmbedUnavailable) as info:
        await embed_client.embed(_server(tmp_path), [secret], role="query", transport=transport)
    assert "500" in str(info.value)
    assert secret not in str(info.value) and "dragons" not in str(info.value)


async def test_a_connection_error_maps_to_embed_unavailable_without_the_text(tmp_path):
    secret = "tell me about the secret lathe"

    def handler(request):
        raise httpx.ConnectError(f"refused while sending {secret}", request=request)

    with pytest.raises(embed_client.EmbedUnavailable) as info:
        await embed_client.embed(
            _server(tmp_path), [secret], role="query", transport=httpx.MockTransport(handler)
        )
    assert "ConnectError" in str(info.value) and "secret" not in str(info.value)


async def test_a_timeout_maps_to_embed_unavailable(tmp_path):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(embed_client.EmbedUnavailable):
        await embed_client.embed(
            _server(tmp_path), ["a"], role="query", transport=httpx.MockTransport(handler)
        )


async def test_a_server_that_cannot_start_maps_to_embed_unavailable(tmp_path):
    server = _server(tmp_path, start_error=RuntimeError("embedding weights not installed"))
    with pytest.raises(embed_client.EmbedUnavailable, match="weights not installed"):
        await embed_client.embed(
            server, ["a"], role="query", transport=httpx.MockTransport(_echo()[0])
        )


async def test_a_missing_key_file_maps_to_embed_unavailable(tmp_path):
    server = _FakeServer(None)
    with pytest.raises(embed_client.EmbedUnavailable, match="not running"):
        await embed_client.embed(
            server, ["a"], role="query", transport=httpx.MockTransport(_echo()[0])
        )
    gone = _FakeServer(tmp_path / "gone.key")
    with pytest.raises(embed_client.EmbedUnavailable):
        await embed_client.embed(
            gone, ["a"], role="query", transport=httpx.MockTransport(_echo()[0])
        )


def test_a_bad_response_is_an_embed_unavailable_so_one_except_covers_every_fallback():
    assert issubclass(embed_client.EmbedBadResponse, embed_client.EmbedUnavailable)
    assert issubclass(embed_client.EmbedUnavailable, RuntimeError)
