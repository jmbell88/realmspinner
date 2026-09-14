"""Tier two of the assistant eval: the fine-tuned model driving the real app.

Tier one (``run_val.py``) scores a reply by replaying it through ``gen/verify``'s headless
door -- a bare ``ClayDoc``, no window, no GL, no MCP wire, ``agent_clay.call`` invoked
in-process. That is deliberately not what production looks like: a real assistant reaches
Clay over ``warlock mcp``'s stdio, through the dual-era JSON-RPC ``bridge.py`` speaks, over
the pipe RPC v1 carries to a running Studio. Tier one cannot catch a defect that lives in any
of those three hops (the bridge's own framing, a tab-per-connection assumption, an argument
shape ``agent_clay.call`` tolerates but the MCP envelope mangles) because it never goes
through them.

So tier two does. A human first stands up a real session with
``scripts/agent_bench.py --serve`` -- a throwaway ``WARLOCK_HOME``, the agent bridge switched
on, Studio's own pipe listening for exactly one bridge connection at a time (``src/warlock/
mcp/pipe.py``'s module docstring). This script then dials that pipe the way a real MCP client
does: it spawns ``uv run --no-sync warlock mcp`` as a child process and speaks newline-
delimited JSON-RPC on its stdin/stdout (``protocol.encode``/``decode``, ``src/warlock/mcp/
protocol.py:69-97`` -- there is no ``Content-Length`` framing here, unlike some other MCP
transports) rather than importing ``agent_clay`` and calling it directly. One connection is
opened for the whole run and reused across all five subjects rather than one per subject --
see ``_reset_scene``'s own docstring for why.

For each of the five pre-registered corpus subjects (``docs/measurements/corpora/
clay-agent-v1.txt``, read via ``run_val._corpus_records``) this sends exactly one chat turn
(the card as the system turn, the prompt as the user turn -- the same shape ``run_val.py``
uses for a ``build`` row) and, if it parses, exactly one ``clay_batch`` tools/call with the
parsed calls folded in, matching ``clay_batch``'s own argument shape
(``src/warlock/studio/agent_clay.py:2247-2260``: ``{"calls": [{"name", "arguments"}, ...]}``,
the same folding ``gen/verify.replay``/``run_val._score`` use against the headless door). No
repair turns: what the model gets right or wrong on the first try, through the exact channel
production drives, is the whole point of a tier tier one's replay cannot answer. A render
(``clay_render``, ``shading: lit``) is still attempted whenever anything was built, even a
batch that refused partway through -- ``render_corpus.py``'s own reasoning applies here too:
a partial scene is still worth a human's eyes.

Every subject lands in exactly the same four outcomes tier one's own ``_score`` uses (``no
parse`` / ``refused`` / ``built, failed check`` / ``accepted``), because this exists to be
compared with tier one per subject, not to answer a looser question -- a non-refused batch is
graded by ``_over_wire_check``, which applies ``gen/verify.check``'s own rules against what is
reachable over MCP (``clay_diagnose``, ``clay_scene``'s bounds/size/bbox), and says in its own
``check_notes`` which two of ``verify.check``'s rules it cannot reproduce this way (3a's
in-process cross-check, 6's serialize round trip) rather than silently dropping them. With
``--tier-one <eval-*.json>`` (a ``run_val.py --corpus`` run) each row also carries
``tier_one_outcome`` and the two tiers print side by side.

    uv run python scripts/agent_bench.py --serve --transcript out/run-A/tier-two/bench.jsonl
    # in a second shell, once the app has printed WARLOCK_HOME=... and is running:
    uv run --no-sync python training/clay-assistant/eval/tier_two.py --run run-A \\
        --card training/clay-assistant/out/run-A/card.txt --warlock-home <home printed above>
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))

# SLUGS is corpus-1..corpus-5's own human-readable names, in the corpus file's line order --
# defined once in render_corpus.py (which needs the same mapping to name its own PNGs) and
# reused here rather than re-derived, so the two tools can never disagree about which slug a
# given corpus id means.
from render_corpus import SLUGS  # noqa: E402

# Reused rather than copied, per the module docstring: SETTINGS is the one place the
# sampler defaults live, _corpus_records/_chat are the exact prompt shape training used,
# and a drift between tier one and tier two's idea of either would be a silent divergence
# in what "the same eval" means. parse_calls comes straight from the contract module
# (T3) rather than through run_val, now that the reply grammar lives there.
from run_val import SETTINGS, _chat, _corpus_records  # noqa: E402

from warlock.studio.familiar.contract import parse_calls  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:8081/v1/chat/completions"

DEFAULT_ARGV: tuple[str, ...] = ("uv", "run", "--no-sync", "warlock", "mcp")
"""The real production child command. A parameter of :class:`MCPClient` rather than
hard-coded in its body so ``tests/test_tier_two.py`` can point the same client at a tiny
fake stdio server instead -- exercising the framing, the handshake and the timeout/EOF
handling with no app, no GPU and no ``warlock`` import at all."""


class MCPError(RuntimeError):
    """This client could not complete an MCP round trip -- the child never answered in
    time, answered something this client cannot parse, or is not running any more."""


class MCPClient:
    """A minimal MCP stdio client, speaking exactly the wire ``warlock mcp``
    (``src/warlock/mcp/bridge.py``) answers: newline-delimited JSON-RPC 2.0, the *legacy*
    era (a bare ``initialize`` with no ``protocolVersion`` negotiates ``LEGACY[0]``, per
    ``protocol._legacy_initialize``, ``src/warlock/mcp/protocol.py:363-384``), never the
    no-``initialize`` "modern" era -- there is no reason for a benchmark client to opt into
    the newer, per-request ``_meta`` versioning tier one's own inference path does not use
    either. This class's request shapes mirror ``tests/mcp/test_bridge_e2e.py`` line for
    line (``initialize`` with no params, then ``tools/list``, then ``tools/call`` with
    ``{"name": ..., "arguments": ...}``) -- that file already proves this exact sequence
    works end to end against a real ``AgentHost``, so this client makes no protocol choice
    that test has not already exercised.

    One tab per connection, one connection for this whole run. Studio hands out a fresh
    Clay tab the instant a bridge's pipe connection is accepted (``agent_host.py``'s
    ``_serve``: ``self._run_on_frame(lambda: agent_clay._tab(self.ctx, session,
    create=True))``, ``src/warlock/studio/agent_host.py:1057-1060``) and serves only one
    such connection at a time (``pipe.py``'s own module docstring: "One connection at a
    time is the v1 decision, not an oversight"). So this client spawns exactly one
    ``warlock mcp`` child and keeps its one pipe connection open across every subject --
    see ``_reset_scene`` for how the tab is cleared between subjects instead of being
    reopened.

    A background thread drains the child's stdout into a queue so every read can carry a
    real timeout (``queue.Queue.get(timeout=...)``) rather than the unbounded
    ``readline()`` a synchronous read would need; a second thread drains stderr so a
    child that logs anything cannot fill its stderr pipe and deadlock on a full buffer --
    the same shape of hazard ``pipelines/_workerio.py`` documents for this codebase's other
    child processes, here on the parent's side of one it does not own.
    """

    def __init__(
        self,
        warlock_home: pathlib.Path,
        *,
        timeout: float,
        argv: tuple[str, ...] = DEFAULT_ARGV,
    ) -> None:
        env = dict(os.environ)
        env["WARLOCK_HOME"] = str(warlock_home)
        # cwd=ROOT: `uv run` resolves the project environment from pyproject.toml/uv.lock,
        # which live at the repo root, not under training/clay-assistant/eval -- and
        # --no-sync so this benchmark run never mutates the env mid-training-run (the
        # caller's own instructions: a multi-hour training run owns the GPU right now, and
        # a `uv sync` here is exactly the kind of surprise side effect that must not happen).
        # Harmless (and ignored) for a test's fake-server argv, which needs no project env.
        self._proc = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(ROOT),
            env=env,
        )
        self._timeout = timeout
        self._next_id = 1
        self._out_q: queue.Queue[bytes | None] = queue.Queue()
        self._err_lines: list[bytes] = []
        self._out_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._err_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._out_thread.start()
        self._err_thread.start()

    def _drain_stdout(self) -> None:
        stdout = self._proc.stdout
        assert stdout is not None
        try:
            for line in iter(stdout.readline, b""):
                self._out_q.put(line)
        finally:
            self._out_q.put(None)  # EOF sentinel -- the child's stdout closed.

    def _drain_stderr(self) -> None:
        stderr = self._proc.stderr
        assert stderr is not None
        for line in iter(stderr.readline, b""):
            self._err_lines.append(line)

    def _dead_child_message(self) -> str:
        self._proc.poll()
        stderr = b"".join(self._err_lines).decode("utf-8", "replace")
        return (
            f"`warlock mcp` (pid {self._proc.pid}) is not answering "
            f"(exit code {self._proc.returncode}). stderr:\n{stderr}"
        )

    def _send(self, message: dict[str, Any]) -> None:
        line = json.dumps(message).encode("utf-8") + b"\n"
        stdin = self._proc.stdin
        assert stdin is not None
        try:
            stdin.write(line)
            stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPError(self._dead_child_message()) from exc

    def _readline(self) -> dict[str, Any]:
        try:
            line = self._out_q.get(timeout=self._timeout)
        except queue.Empty as exc:
            raise MCPError(f"no reply from `warlock mcp` within {self._timeout}s") from exc
        if line is None:
            raise MCPError(self._dead_child_message())
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise MCPError(f"unparseable line from `warlock mcp`: {line!r}") from exc

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        msg_id = self._next_id
        self._next_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)
        reply = self._readline()
        if reply.get("id") != msg_id:
            raise MCPError(f"reply id mismatch: sent {msg_id}, got {reply.get('id')!r}: {reply}")
        if "error" in reply:
            raise MCPError(f"{method} refused at the JSON-RPC level: {reply['error']}")
        return reply["result"]

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

    def initialize(self) -> dict[str, Any]:
        result = self.request("initialize")
        # A notification carries no "id" and gets no reply, by JSON-RPC's own rule
        # (protocol._dispatch_one's has_id gate) -- sent for a real client's sake even
        # though the legacy branch's own notifications/initialized handler is a no-op
        # (protocol.py:624-625), so this client's handshake looks like any other MCP
        # client's rather than skipping a step a real tool runner always takes.
        self.notify("notifications/initialized")
        return result

    def tools_list(self) -> list[dict[str, Any]]:
        return list(self.request("tools/list").get("tools", []))

    def tools_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        try:
            self._proc.wait(timeout=self._timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=self._timeout)


def _refusal_text(result: dict[str, Any]) -> str:
    """*result*'s own first content line. Mirrors ``gen/verify._refusal_text`` exactly,
    but reimplemented rather than imported: that module's package (``gen/verify.py``)
    imports ``warlock.studio`` at module scope to reach the real door, which this process
    has no other reason to load at all -- it never touches Clay except over the wire, and
    the caller's own brief is explicit that nothing here may import warlock config or touch
    the GPU while a training run owns it."""
    content = result.get("content") or []
    if content and isinstance(content[0], dict):
        return str(content[0].get("text", ""))
    return "refused (no message on the result)"


def _batch_refusal_reason(result: dict[str, Any]) -> str:
    """The same ``stopped_at`` lookup ``gen/verify.refusal_reason`` does -- a top-level
    ``clay_batch`` refusal's own ``content[0].text`` is the *whole batch result* serialised
    as JSON, unreadable as a one-line reason; the sub-call actually named by
    ``structuredContent["stopped_at"]`` has the real sentence."""
    structured = result.get("structuredContent") or {}
    stopped_at = structured.get("stopped_at")
    results = structured.get("results") or []
    if isinstance(stopped_at, int) and 0 <= stopped_at < len(results):
        return _refusal_text(results[stopped_at])
    return _refusal_text(result)


def _first_image_bytes(result: dict[str, Any]) -> bytes | None:
    """The first ``image`` content block's decoded PNG bytes, or ``None`` -- the MCP image
    content shape ``rpc.image_png`` builds (``src/warlock/mcp/rpc.py:220-225``):
    ``{"type": "image", "data": <base64>, "mimeType": "image/png"}``."""
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "image" and "data" in item:
            return base64.b64decode(item["data"])
    return None


def _summarize_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    """*result* verbatim, except an image content block is replaced by its byte length --
    ``tier-two.json`` records ok/refused plus the text a human or a later script would read,
    not a second copy of a PNG that is already saved to its own file (see ``_render_subject``)."""
    content = []
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "image" and "data" in item:
            content.append({"type": "image", "bytes": len(base64.b64decode(item["data"]))})
        else:
            content.append(item)
    summary: dict[str, Any] = {"isError": result.get("isError"), "content": content}
    if "structuredContent" in result:
        summary["structuredContent"] = result["structuredContent"]
    return summary


MIN_DOC_SIZE = 0.005
MAX_DOC_SIZE = 20.0
"""``gen/verify.check``'s rule 4a bounds, copied rather than imported: importing
``gen.verify`` pulls in ``warlock.studio`` at module scope (its own docstring says so),
which this script must never touch while a training run owns the GPU. Keep these in step
with ``gen/verify.py``'s own ``MIN_DOC_SIZE``/``MAX_DOC_SIZE`` by hand."""

MIN_COMPONENT = 1e-3
"""``gen/verify.check``'s rule 4b floor, copied for the same reason as above -- see
``gen/verify.py``'s own ``MIN_COMPONENT`` docstring for why ``<=`` and not ``<``."""

GROUND_EPS = -0.02
"""``gen/verify.check``'s rule 5 floor, copied for the same reason as above -- see
``gen/verify.py``'s own ``GROUND_EPS`` docstring."""


def _ground_reasons_for(row: dict[str, Any]) -> list[str]:
    """``gen/verify._ground_reasons_for``, reimplemented against the wire shape
    ``clay_scene``'s own ``objects`` rows carry (``src/warlock/studio/agent_clay.py``'s
    ``_object_row_output_schema``: ``bbox`` is ``{"min": [...], "max": [...]}`` or ``None``)
    -- not imported, for the same reason the constants above are copied rather than
    imported."""
    bbox = row.get("bbox")
    if bbox is None:
        return []
    min_y = bbox["min"][1]
    if min_y < GROUND_EPS:
        return [f"{row.get('name')!r} sits below the ground plane (bbox min y={min_y})"]
    return []


def _over_wire_check(
    diag_result: dict[str, Any] | None,
    scene_structured: dict[str, Any],
    *,
    allow_below_ground: bool,
) -> tuple[list[str], list[str]]:
    """The same acceptance rule ``gen/verify.check`` applies, replicated against what is
    reachable over the MCP wire -- called only once a subject's batch did *not* refuse
    (rules 1a/1b are already the ``refused``/``no parse`` outcomes by the time this runs;
    rule 7, "an edit made no change", never applies either, because tier two only ever
    sends one prior-less ``build`` batch per subject, never an edit's own ``prior`` -- see
    the module docstring). Returns ``(reasons, notes)``: *reasons* is the same "why this
    would be rejected" list ``verify.check`` returns, and *notes* names every
    ``verify.check`` rule this cannot reproduce over the wire, and why -- so a silent gap
    is never mistaken for "nothing wrong here".
    """
    notes = [
        "rule 3a (clay/diagnose.findings called directly, in-process, on each object's own "
        "mesh) is unreachable over MCP by construction: clay_diagnose is the only door "
        "available here, and it already runs exactly that function (see agent_clay._h_diagnose) "
        "-- so only verify.check's rule 3b (the tool-call cross-check against the same "
        "findings) is reproduced below. The divergence-between-two-doors defect rule 3a "
        "exists to catch cannot be seen from this side of the wire.",
        "rule 6 (the document survives a real serialize.wblk_bytes/read_wblk round trip) is "
        "not reproduced: no MCP tool exposes that round trip, and importing gen/verify or "
        "warlock.studio.clay.serialize here would pull in warlock.studio, which this script "
        "must never import while a training run owns the GPU (see the module docstring).",
    ]
    reasons: list[str] = []

    # Rule 2: nothing was built at all.
    object_count = scene_structured.get("object_count", 0)
    if not object_count:
        reasons.append("no objects were created")
        return reasons, notes  # nothing below has anything to look at, same as verify.check

    # Rule 3b: the clay_diagnose tool's own per-object "clean" flag.
    diag_result = diag_result or {}
    if diag_result.get("isError"):
        reasons.append(f"clay_diagnose refused: {_refusal_text(diag_result)}")
    else:
        for row in (diag_result.get("structuredContent") or {}).get("objects", []):
            if not row.get("clean", False):
                reasons.append(f"{row.get('name')!r} is not clean per clay_diagnose")

    # Rule 4a: the document-wide bounds, from clay_scene's own payload.
    bounds = scene_structured.get("bounds")
    if bounds is None:
        reasons.append("clay_scene reports no bounds (nothing visible)")
    else:
        for i, s in enumerate(bounds.get("size") or []):
            if s < MIN_DOC_SIZE or s > MAX_DOC_SIZE:
                reasons.append(
                    f"document bounds size[{i}]={s} outside [{MIN_DOC_SIZE}, {MAX_DOC_SIZE}]"
                )

    # Rule 4b/5: per-object degenerate size, and the named ground check.
    for row in scene_structured.get("objects", []):
        name = row.get("name")
        size = row.get("size")
        if size is not None:
            for i, s in enumerate(size):
                if s <= MIN_COMPONENT:
                    reasons.append(f"{name!r} has a near-zero size component (size[{i}]={s})")
        if not allow_below_ground:
            reasons.extend(_ground_reasons_for(row))

    return reasons, notes


def _reset_scene(client: MCPClient) -> None:
    """Empty the shared tab between subjects, rather than reconnecting for a fresh one.

    A fresh ``clay_scene``/``clay_delete`` round trip is cheaper and more realistic than a
    second ``uv run --no-sync warlock mcp`` per subject would be: reconnecting means a new
    process spawn, a new pipe handshake and a new tab for every one of the five subjects
    (see :class:`MCPClient`'s own docstring for why one connection is one tab), for no
    benefit tier three cares about -- the live assistant this eval is standing in for will
    face a session that already has a tab open, asked to build a second, unrelated thing in
    it, not a guarantee of a pristine process every time. ``clay_delete`` of every uid is
    the same shape of reset a person clearing a scene by hand would do.
    """
    scene = client.tools_call("clay_scene", {})
    structured = scene.get("structuredContent") or {}
    uids = [row["uid"] for row in structured.get("objects", [])]
    if not uids:
        return
    result = client.tools_call("clay_delete", {"uids": uids})
    if result.get("isError"):
        raise MCPError(f"clay_delete failed while resetting the scene: {_refusal_text(result)}")


def run_subject(
    client: MCPClient,
    record: dict[str, Any],
    slug: str,
    system: str,
    *,
    url: str,
    temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
    seed: int,
    chat_timeout: float,
    out_dir: pathlib.Path,
) -> dict[str, Any]:
    """One corpus subject, end to end: one chat turn, one ``clay_batch`` if it parses, a
    render if anything was built (even partially), then the row ``tier-two.json`` records.
    No repair turn -- see the module docstring for why."""
    reply = _chat(
        url,
        system,
        record["prompt"],
        chat_timeout,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )
    row: dict[str, Any] = {
        "id": record["id"],
        "slug": slug,
        "prompt": record["prompt"],
        "reply": reply["content"],
        "finish": reply["finish"],
        "tokens": reply["completion_tokens"],
    }

    calls, detail = parse_calls(reply["content"])
    if calls is None:
        row.update(outcome="no parse", detail=detail, sub_calls=0, objects=0, render=None)
        return row
    row["calls"] = calls
    row["sub_calls"] = len(calls)

    batch_result = client.tools_call("clay_batch", {"calls": calls})
    row["tool_result"] = _summarize_tool_result(batch_result)

    scene_result = client.tools_call("clay_scene", {})
    scene_structured = scene_result.get("structuredContent") or {}
    object_count = scene_structured.get("object_count", 0)
    row["objects"] = object_count

    if batch_result.get("isError"):
        row["outcome"] = "refused"
        row["detail"] = _batch_refusal_reason(batch_result)
    else:
        # Not refused -- apply the same acceptance rule gen/verify.check applies, over
        # the wire (see _over_wire_check's own docstring for what it cannot reproduce).
        diag_result = None
        if object_count:
            diag_result = client.tools_call("clay_diagnose", {})
            row["diagnose_result"] = _summarize_tool_result(diag_result)
        reasons, notes = _over_wire_check(
            diag_result,
            scene_structured,
            allow_below_ground=bool(record.get("allow_below_ground")),
        )
        row["check_notes"] = notes
        if reasons:
            row["outcome"] = "built, failed check"
            row["detail"] = "; ".join(reasons)[:300]
        else:
            row["outcome"] = "accepted"

    row["render"] = None
    if object_count:
        # A render even for a refused-partway batch, matching render_corpus.py's own
        # reasoning: a partial scene is still worth a human's eyes.
        render_result = client.tools_call("clay_render", {"shading": "lit"})
        if render_result.get("isError"):
            row["render_error"] = _refusal_text(render_result)
        else:
            png_bytes = _first_image_bytes(render_result)
            if png_bytes is not None:
                path = out_dir / f"{slug}.png"
                path.write_bytes(png_bytes)
                row["render"] = str(path)

    return row


def _load_tier_one_outcomes(path: pathlib.Path) -> dict[str, str]:
    """``corpus-1``..``corpus-5`` -> that subject's tier-one ``outcome`` (the first sample's,
    same as every other top-level field ``run_val.py``'s own row copies up from ``samples[0]``
    -- see its ``main``'s ``row = {...}`` block), read from an ``eval-<tag>.json`` written by
    ``run_val.py --corpus``. Not every eval run includes the corpus subjects, so a row named
    here that ``run_val.py`` never scored (``--corpus`` omitted) is simply absent from the
    returned mapping, not an error."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["id"]: row["outcome"]
        for row in data.get("rows", [])
        if str(row.get("id", "")).startswith("corpus-")
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="run-A", help="out/<run>/tier-two/ to write into")
    ap.add_argument("--card", type=pathlib.Path, required=True, help="system prompt, read verbatim")
    ap.add_argument(
        "--warlock-home",
        type=pathlib.Path,
        required=True,
        help="the WARLOCK_HOME printed by `scripts/agent_bench.py --serve`",
    )
    ap.add_argument("--url", default=DEFAULT_URL, help="llama-server chat endpoint")
    ap.add_argument("--temperature", type=float, default=SETTINGS["temperature"])
    ap.add_argument("--top-k", type=int, default=SETTINGS["top_k"])
    ap.add_argument("--top-p", type=float, default=SETTINGS["top_p"])
    ap.add_argument("--max-tokens", type=int, default=SETTINGS["max_tokens"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chat-timeout", type=float, default=600.0, help="per-subject chat timeout")
    ap.add_argument(
        "--mcp-timeout",
        type=float,
        default=60.0,
        help="per-round-trip timeout waiting on `warlock mcp` (spawn, handshake, each call)",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="restrict to corpus subjects whose slug or prompt contains this substring",
    )
    ap.add_argument(
        "--tier-one",
        type=pathlib.Path,
        default=None,
        help=(
            "an eval-<tag>.json written by `run_val.py --corpus`; when given, each "
            "subject's row also carries tier_one_outcome and the summary prints both "
            "tiers' outcomes side by side"
        ),
    )
    args = ap.parse_args()

    system = args.card.read_bytes().decode("utf-8")
    card_sha256 = hashlib.sha256(system.encode("utf-8")).hexdigest()

    records = _corpus_records()
    slug_by_id = {f"corpus-{i}": slug for i, slug in enumerate(SLUGS, start=1)}
    if args.only:
        needle = args.only.lower()
        records = [
            r
            for r in records
            if needle in slug_by_id[r["id"]].lower() or needle in r["prompt"].lower()
        ]
        if not records:
            raise SystemExit(f"--only {args.only!r} matched no corpus subject")

    out_dir = PKG / "out" / args.run / "tier-two"
    out_dir.mkdir(parents=True, exist_ok=True)

    tier_one_outcomes = _load_tier_one_outcomes(args.tier_one) if args.tier_one else None

    client = MCPClient(args.warlock_home, timeout=args.mcp_timeout)
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    try:
        client.initialize()
        tool_names = {t["name"] for t in client.tools_list()}
        if "clay_batch" not in tool_names:
            raise SystemExit(
                "`warlock mcp` never advertised clay_batch -- is the agent bridge on for "
                f"{args.warlock_home}? (Settings -> Advanced -> Allow AI agents to drive "
                "the Studio, or scripts/agent_bench.py --serve)"
            )

        for record in records:
            slug = slug_by_id[record["id"]]
            row = run_subject(
                client,
                record,
                slug,
                system,
                url=args.url,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed,
                chat_timeout=args.chat_timeout,
                out_dir=out_dir,
            )
            if tier_one_outcomes is not None:
                row["tier_one_outcome"] = tier_one_outcomes.get(record["id"])
            rows.append(row)
            _reset_scene(client)
            render_note = row.get("render") or row.get("render_error") or "-"
            tier_one_note = (
                f" tier1={row['tier_one_outcome']}" if tier_one_outcomes is not None else ""
            )
            print(
                f"{slug:<12} {row['outcome']:<20} calls={row.get('sub_calls', 0):<3} "
                f"objects={row.get('objects', 0):<3} render={render_note}{tier_one_note}"
            )
    finally:
        client.close()

    elapsed = time.time() - t0
    accepted = sum(1 for r in rows if r["outcome"] == "accepted")
    print(f"\n{accepted}/{len(rows)} subjects accepted, in {elapsed:.0f}s")

    if tier_one_outcomes is not None:
        print("\ntier one vs tier two:")
        for row in rows:
            tier_one = row.get("tier_one_outcome") or "(not scored)"
            match = "match" if tier_one == row["outcome"] else "DIFF"
            print(f"  {row['slug']:<12} tier1={tier_one:<20} tier2={row['outcome']:<20} {match}")

    settings = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "url": args.url,
        "card": str(args.card),
        "card_sha256": card_sha256,
        "warlock_home": str(args.warlock_home),
    }
    (out_dir / "tier-two.json").write_text(
        json.dumps({"run": args.run, "settings": settings, "subjects": rows}, indent=1),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
