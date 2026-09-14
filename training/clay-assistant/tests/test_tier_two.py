"""Tests for ``eval/tier_two.py``'s MCP client and per-subject driver.

No app, no GPU, no ``llama-server``, and no ``warlock`` import at all: the fake peer below
is a pure-stdlib stdio script speaking exactly the newline-delimited JSON-RPC framing
``src/warlock/mcp/protocol.py`` defines and ``tests/mcp/test_bridge_e2e.py`` already proves
a real ``warlock mcp`` answers with (legacy ``initialize`` -> ``tools/list`` ->
``tools/call``) -- these tests exercise ``tier_two.MCPClient`` against that framing, and
``tier_two.run_subject``'s folding/refusal/render bookkeeping against a scripted reply, with
nothing that could reach a real Studio, a real model or a real GPU.

    uv run pytest training/clay-assistant/tests -n 0 -p no:cacheprovider
"""

from __future__ import annotations

import base64
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # training/clay-assistant
_EVAL = _PACKAGE_ROOT / "eval"
for _p in (_PACKAGE_ROOT, _EVAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import tier_two  # noqa: E402

FAKE_PNG = b"not-a-real-png-just-some-bytes-tier-two-writes-to-disk"


def _write_fake_server(tmp_path: Path, log_path: Path) -> Path:
    """A tiny stdio MCP peer: legacy ``initialize``, a fixed ``tools/list``, and
    ``tools/call`` for exactly the five tools ``tier_two.py`` drives. Pure stdlib, no
    ``warlock`` import -- this is not a stand-in for Studio's own behaviour (that is what
    ``tests/mcp/test_bridge_e2e.py`` already proves against the real thing), only for the
    wire shape ``MCPClient`` has to speak correctly, and for ``_over_wire_check``'s own
    reading of ``clay_scene``/``clay_diagnose``'s reply shapes.

    A ``clay_batch`` call whose ``calls`` contains a call named ``"boom_call"`` is refused,
    the same ``stopped_at``/``results`` shape ``clay_batch`` itself uses
    (``src/warlock/studio/agent_clay.py``'s batch handler); every other batch is accepted and
    sets this server's own scenario for the ``clay_scene``/``clay_diagnose`` replies that
    follow, from the other sentinel call names a test's fenced reply can use: ``"dirty_call"``
    (``clay_diagnose`` reports one object unclean) and ``"below_ground_call"`` (``clay_scene``
    reports an object whose bbox dips below ``tier_two.GROUND_EPS``); anything else is the
    "clean" scenario -- a grounded, non-degenerate two-object scene with nothing to report.
    ``clay_scene`` always reports two objects (so a render is always attempted regardless of
    whether the batch was accepted, exercising the "partial scene still renders" path) plus
    ``bounds``/``size``/``bbox`` fields shaped like ``agent_clay._object_row_output_schema``.
    Every ``clay_delete`` call appends its ``uids`` to *log_path* as one JSON line, so a test
    can assert what :func:`tier_two._reset_scene` actually asked to delete.
    """
    script = textwrap.dedent(
        f"""
        import base64, json, sys

        LOG = {str(log_path)!r}
        PNG_B64 = {base64.b64encode(FAKE_PNG).decode("ascii")!r}
        SCENARIO = ["clean"]

        TOOLS = [
            {{"name": "clay_batch", "title": "t", "description": "d",
              "inputSchema": {{"type": "object"}}}},
            {{"name": "clay_scene", "title": "t", "description": "d",
              "inputSchema": {{"type": "object"}}}},
            {{"name": "clay_diagnose", "title": "t", "description": "d",
              "inputSchema": {{"type": "object"}}}},
            {{"name": "clay_delete", "title": "t", "description": "d",
              "inputSchema": {{"type": "object"}}}},
            {{"name": "clay_render", "title": "t", "description": "d",
              "inputSchema": {{"type": "object"}}}},
        ]

        def reply(msg_id, result):
            envelope = {{"jsonrpc": "2.0", "id": msg_id, "result": result}}
            sys.stdout.write(json.dumps(envelope) + "\\n")
            sys.stdout.flush()

        def tool_call(msg_id, name, arguments):
            if name == "clay_batch":
                calls = arguments.get("calls", [])
                names = {{c.get("name") for c in calls}}
                boom = next((i for i, c in enumerate(calls) if c.get("name") == "boom_call"), None)
                if boom is not None:
                    inner = {{"content": [{{"type": "text", "text": "no object named 'x'."}}],
                              "isError": True}}
                    result = {{"content": [{{"type": "text", "text": "batch stopped"}}],
                               "isError": True,
                               "structuredContent": {{"stopped_at": boom, "results": [inner]}}}}
                    reply(msg_id, result)
                    return
                if "dirty_call" in names:
                    SCENARIO[0] = "dirty"
                elif "below_ground_call" in names:
                    SCENARIO[0] = "below_ground"
                else:
                    SCENARIO[0] = "clean"
                result = {{"content": [{{"type": "text", "text": "ok"}}], "isError": False}}
                reply(msg_id, result)
                return
            if name == "clay_scene":
                if SCENARIO[0] == "below_ground":
                    bbox = {{"min": [-0.5, -0.3, -0.5], "max": [0.5, 0.3, 0.5]}}
                    size = [1.0, 0.6, 1.0]
                else:
                    bbox = {{"min": [-0.5, 0.0, -0.5], "max": [0.5, 1.0, 0.5]}}
                    size = [1.0, 1.0, 1.0]
                objects = [
                    {{"uid": 1, "name": "Box", "bbox": bbox, "size": size}},
                    {{"uid": 2, "name": "Box.001", "bbox": bbox, "size": size}},
                ]
                bounds = {{"min": bbox["min"], "max": bbox["max"], "size": size,
                           "center": [0.0, 0.0, 0.0]}}
                structured = {{"objects": objects, "object_count": 2, "bounds": bounds}}
                reply(msg_id, {{"content": [{{"type": "text", "text": "scene"}}],
                                 "isError": False, "structuredContent": structured}})
                return
            if name == "clay_diagnose":
                if SCENARIO[0] == "dirty":
                    objects = [
                        {{"uid": 1, "name": "Box", "clean": False,
                          "findings": [{{"kind": "non_manifold", "label": "l", "count": 1,
                                         "mode": "edge"}}]}},
                        {{"uid": 2, "name": "Box.001", "clean": True, "findings": []}},
                    ]
                else:
                    objects = [
                        {{"uid": 1, "name": "Box", "clean": True, "findings": []}},
                        {{"uid": 2, "name": "Box.001", "clean": True, "findings": []}},
                    ]
                reply(msg_id, {{"content": [{{"type": "text", "text": "diagnose"}}],
                                 "isError": False, "structuredContent": {{"objects": objects}}}})
                return
            if name == "clay_delete":
                with open(LOG, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(arguments.get("uids", [])) + "\\n")
                reply(msg_id, {{"content": [], "isError": False}})
                return
            if name == "clay_render":
                result = {{"content": [{{"type": "text", "text": "{{}}"}},
                                        {{"type": "image", "data": PNG_B64,
                                          "mimeType": "image/png"}}],
                           "isError": False}}
                reply(msg_id, result)
                return
            reply(msg_id, {{"content": [{{"type": "text", "text": "unknown tool"}}],
                             "isError": True}})

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            method = msg.get("method")
            has_id = "id" in msg
            msg_id = msg.get("id")
            params = msg.get("params") or {{}}
            if method == "initialize":
                server_info = {{"name": "fake", "version": "0"}}
                reply(msg_id, {{"protocolVersion": "2025-11-25",
                                 "capabilities": {{}}, "serverInfo": server_info}})
            elif method == "notifications/initialized":
                pass  # a notification: no reply, ever
            elif method == "tools/list":
                reply(msg_id, {{"tools": TOOLS}})
            elif method == "tools/call":
                tool_call(msg_id, params.get("name"), params.get("arguments") or {{}})
            elif has_id:
                reply(msg_id, {{"error": {{"code": -32601, "message": "unknown method"}}}})
        """
    )
    path = tmp_path / "fake_mcp_server.py"
    path.write_text(script, encoding="utf-8")
    return path


def _client(tmp_path: Path, log_path: Path) -> tier_two.MCPClient:
    server = _write_fake_server(tmp_path, log_path)
    return tier_two.MCPClient(
        tmp_path / "home",  # never read by the fake server; WARLOCK_HOME just gets set
        timeout=10.0,
        argv=(sys.executable, str(server)),
    )


def test_handshake_tools_list_and_ok_tools_call(tmp_path: Path) -> None:
    client = _client(tmp_path, tmp_path / "deletes.log")
    try:
        client.initialize()
        names = {t["name"] for t in client.tools_list()}
        assert names == {
            "clay_batch",
            "clay_scene",
            "clay_diagnose",
            "clay_delete",
            "clay_render",
        }

        result = client.tools_call("clay_batch", {"calls": [{"name": "clay_add_primitive"}]})
        assert result["isError"] is False
    finally:
        client.close()


def test_dead_child_raises_mcp_error(tmp_path: Path) -> None:
    """A child that exits before answering (Studio down, no snapshot -- ``bridge.main``'s
    own ``exit(1)`` path) must surface as a clear :class:`tier_two.MCPError`, never a hang
    or a bare ``BrokenPipeError``."""
    client = tier_two.MCPClient(
        tmp_path / "home",
        timeout=5.0,
        argv=(sys.executable, "-c", "import sys; print('not accepting', file=sys.stderr)"),
    )
    try:
        try:
            client.initialize()
        except tier_two.MCPError as exc:
            assert "not accepting" in str(exc) or "not answering" in str(exc)
        else:
            raise AssertionError("expected MCPError from a child that never replied")
    finally:
        client.close()


def test_reset_scene_deletes_every_reported_uid(tmp_path: Path) -> None:
    log_path = tmp_path / "deletes.log"
    client = _client(tmp_path, log_path)
    try:
        client.initialize()
        tier_two._reset_scene(client)
    finally:
        client.close()
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert lines == [[1, 2]]


def _fenced(calls: list[dict[str, Any]]) -> str:
    return "```json\n" + json.dumps({"calls": calls}) + "\n```"


def _fake_chat(content: str) -> Any:
    def chat(url, system, user, timeout, *, temperature, top_k, top_p, max_tokens, seed):
        del url, system, user, timeout, temperature, top_k, top_p, max_tokens, seed
        return {"content": content, "finish": "stop", "completion_tokens": 42}

    return chat


def _record() -> dict[str, Any]:
    return {"id": "corpus-1", "family": "corpus", "kind": "build", "prompt": "a wooden chair"}


def test_run_subject_no_parse_makes_no_tool_call(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tier_two, "_chat", _fake_chat("no fence here, just prose"))
    log_path = tmp_path / "deletes.log"
    client = _client(tmp_path, log_path)
    try:
        client.initialize()
        row = tier_two.run_subject(
            client,
            _record(),
            "chair",
            "system card",
            url="unused",
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            max_tokens=100,
            seed=0,
            chat_timeout=10.0,
            out_dir=tmp_path,
        )
    finally:
        client.close()
    assert row["outcome"] == "no parse"
    assert row["sub_calls"] == 0
    assert row["render"] is None
    assert "calls" not in row


def test_run_subject_accepted_writes_render(tmp_path: Path, monkeypatch) -> None:
    """A clean, grounded, non-degenerate two-object scene -- ``_over_wire_check`` finds
    nothing to reject, so the outcome is ``accepted`` (not merely "the batch did not
    refuse"), matching tier one's own four-outcome vocabulary."""
    reply = _fenced([{"name": "clay_add_primitive", "arguments": {"generator": "box"}}])
    monkeypatch.setattr(tier_two, "_chat", _fake_chat(reply))
    client = _client(tmp_path, tmp_path / "deletes.log")
    try:
        client.initialize()
        row = tier_two.run_subject(
            client,
            _record(),
            "chair",
            "system card",
            url="unused",
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            max_tokens=100,
            seed=0,
            chat_timeout=10.0,
            out_dir=tmp_path,
        )
    finally:
        client.close()
    assert row["outcome"] == "accepted"
    assert "detail" not in row
    assert row["sub_calls"] == 1
    assert row["objects"] == 2
    assert row["tool_result"]["isError"] is False
    assert row["diagnose_result"]["isError"] is False
    assert row["diagnose_result"]["structuredContent"]["objects"][0]["clean"] is True
    assert row["check_notes"]  # names what verify.check rules this cannot reproduce
    assert row["render"] == str(tmp_path / "chair.png")
    assert Path(row["render"]).read_bytes() == FAKE_PNG


def test_run_subject_built_failed_check_diagnose_dirty(tmp_path: Path, monkeypatch) -> None:
    """A batch that does not refuse but leaves an unclean object per ``clay_diagnose`` is
    ``built, failed check``, not ``accepted`` -- the gap tier two existed to close (the old
    code called this outcome ``ok`` from a bare "did not refuse" reading)."""
    reply = _fenced([{"name": "dirty_call", "arguments": {}}])
    monkeypatch.setattr(tier_two, "_chat", _fake_chat(reply))
    client = _client(tmp_path, tmp_path / "deletes.log")
    try:
        client.initialize()
        row = tier_two.run_subject(
            client,
            _record(),
            "chair",
            "system card",
            url="unused",
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            max_tokens=100,
            seed=0,
            chat_timeout=10.0,
            out_dir=tmp_path,
        )
    finally:
        client.close()
    assert row["outcome"] == "built, failed check"
    assert "not clean per clay_diagnose" in row["detail"]
    assert "'Box'" in row["detail"]
    assert row["diagnose_result"]["structuredContent"]["objects"][0]["clean"] is False


def test_run_subject_built_failed_check_below_ground(tmp_path: Path, monkeypatch) -> None:
    """A batch that leaves an object with a bbox dipping below ``tier_two.GROUND_EPS`` is
    ``built, failed check`` even though ``clay_diagnose`` reports every object clean --
    verify.check's rule 5 (the named ground check), reproduced over the wire from
    ``clay_scene``'s own ``bbox`` field."""
    reply = _fenced([{"name": "below_ground_call", "arguments": {}}])
    monkeypatch.setattr(tier_two, "_chat", _fake_chat(reply))
    client = _client(tmp_path, tmp_path / "deletes.log")
    try:
        client.initialize()
        row = tier_two.run_subject(
            client,
            _record(),
            "chair",
            "system card",
            url="unused",
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            max_tokens=100,
            seed=0,
            chat_timeout=10.0,
            out_dir=tmp_path,
        )
    finally:
        client.close()
    assert row["outcome"] == "built, failed check"
    assert "sits below the ground plane" in row["detail"]
    assert "'Box'" in row["detail"]


def test_run_subject_below_ground_allowed_when_record_says_so(tmp_path: Path, monkeypatch) -> None:
    """``record["allow_below_ground"]`` opts a subject out of exactly the ground check, per
    ``gen/verify._ground_reasons_for``'s own docstring -- everything else (diagnose,
    degenerate size, document bounds) still applies."""
    reply = _fenced([{"name": "below_ground_call", "arguments": {}}])
    monkeypatch.setattr(tier_two, "_chat", _fake_chat(reply))
    client = _client(tmp_path, tmp_path / "deletes.log")
    record = {**_record(), "allow_below_ground": True}
    try:
        client.initialize()
        row = tier_two.run_subject(
            client,
            record,
            "chair",
            "system card",
            url="unused",
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            max_tokens=100,
            seed=0,
            chat_timeout=10.0,
            out_dir=tmp_path,
        )
    finally:
        client.close()
    assert row["outcome"] == "accepted"


def test_run_subject_refused_still_renders_partial_scene(tmp_path: Path, monkeypatch) -> None:
    reply = _fenced([{"name": "boom_call", "arguments": {}}])
    monkeypatch.setattr(tier_two, "_chat", _fake_chat(reply))
    client = _client(tmp_path, tmp_path / "deletes.log")
    try:
        client.initialize()
        row = tier_two.run_subject(
            client,
            _record(),
            "chair",
            "system card",
            url="unused",
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            max_tokens=100,
            seed=0,
            chat_timeout=10.0,
            out_dir=tmp_path,
        )
    finally:
        client.close()
    assert row["outcome"] == "refused"
    assert row["detail"] == "no object named 'x'."
    # clay_scene still reports objects (the fake always does), so this is the
    # "partial scene still worth a human's eyes" path -- a render is still written.
    assert row["render"] == str(tmp_path / "chair.png")


def test_load_tier_one_outcomes_keeps_only_corpus_ids_first_sample(tmp_path: Path) -> None:
    """:func:`tier_two._load_tier_one_outcomes` reads the same ``eval-<tag>.json`` shape
    ``run_val.py --corpus`` writes -- each row's top-level ``outcome`` is already the first
    sample's (``run_val.main``'s own ``row = {..., "outcome": first["outcome"]}``) -- and
    keeps only the five pre-registered corpus subjects, by id, dropping every ``val.jsonl``
    row (``dataset-*``, not ``corpus-*``) that rode along in the same file."""
    eval_path = tmp_path / "eval-A-q8-t0.json"
    eval_path.write_text(
        json.dumps(
            {
                "tag": "A-q8-t0",
                "rows": [
                    {"id": "dataset-1", "outcome": "accepted"},
                    {"id": "corpus-1", "outcome": "accepted"},
                    {"id": "corpus-2", "outcome": "refused"},
                    {"id": "corpus-3", "outcome": "built, failed check"},
                    {"id": "corpus-4", "outcome": "no parse"},
                    {"id": "corpus-5", "outcome": "accepted"},
                ],
            }
        ),
        encoding="utf-8",
    )
    outcomes = tier_two._load_tier_one_outcomes(eval_path)
    assert outcomes == {
        "corpus-1": "accepted",
        "corpus-2": "refused",
        "corpus-3": "built, failed check",
        "corpus-4": "no parse",
        "corpus-5": "accepted",
    }


def test_tier_one_outcome_joins_onto_the_row_by_corpus_id(tmp_path: Path, monkeypatch) -> None:
    """The join ``main`` performs after :func:`tier_two.run_subject` returns: a row's
    ``tier_one_outcome`` is looked up by *this subject's own* corpus id, not position, and is
    ``None`` for a subject tier one never scored -- the same "absent, not an error" contract
    :func:`tier_two._load_tier_one_outcomes`'s own docstring names."""
    eval_path = tmp_path / "eval-A-q8-t0.json"
    eval_path.write_text(
        json.dumps({"rows": [{"id": "corpus-1", "outcome": "built, failed check"}]}),
        encoding="utf-8",
    )
    tier_one_outcomes = tier_two._load_tier_one_outcomes(eval_path)

    reply = _fenced([{"name": "clay_add_primitive", "arguments": {"generator": "box"}}])
    monkeypatch.setattr(tier_two, "_chat", _fake_chat(reply))
    client = _client(tmp_path, tmp_path / "deletes.log")
    try:
        client.initialize()
        scored_row = tier_two.run_subject(
            client,
            _record(),
            "chair",
            "system card",
            url="unused",
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            max_tokens=100,
            seed=0,
            chat_timeout=10.0,
            out_dir=tmp_path,
        )
        unscored_row = tier_two.run_subject(
            client,
            {**_record(), "id": "corpus-2"},
            "table",
            "system card",
            url="unused",
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            max_tokens=100,
            seed=0,
            chat_timeout=10.0,
            out_dir=tmp_path,
        )
    finally:
        client.close()

    # This is the same one-line join main() performs once run_subject returns.
    scored_row["tier_one_outcome"] = tier_one_outcomes.get(scored_row["id"])
    unscored_row["tier_one_outcome"] = tier_one_outcomes.get(unscored_row["id"])

    assert scored_row["outcome"] == "accepted"
    assert scored_row["tier_one_outcome"] == "built, failed check"
    assert unscored_row["tier_one_outcome"] is None
