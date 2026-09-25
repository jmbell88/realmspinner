"""The service door onto Familiar: a chat reply, or a Clay build's tool calls.

Both functions block -- called from a ``TaskRunner`` worker thread (see
``studio/tasks.py``), never the frame thread -- and both go through
``svc.call_on_loop`` to run the actual ``httpx`` request on the
``realmspinner-loop`` thread where ``LlamaServer`` and its asyncio lock live.

**Every refusal here is a :class:`FamiliarRefusal`**, a ``ServiceError`` that
adds one field, ``reason``, drawn from a fixed vocabulary
(:data:`REASONS`) so the pane can choose an icon or an action (an Install…
button, a "try again" hint) without parsing ``message`` -- the same idea as
``ServiceError.field`` for a form control, one level more specific because a
chat refusal has no control to point at.

``pipelines.llama.LlamaServer.ensure_started`` and
``familiar.llama_client.chat`` both raise plain ``RuntimeError``/``ValueError``
with a handful of fixed sentences (see each module's own docstring for the
exact wording); :func:`_reason_for` classifies them by substring rather than
by a new exception hierarchy in ``pipelines/``/``familiar/`` -- neither
module may import ``service`` (the offline/layering invariant: both are
reached by a training script and by ``queue.py``, neither of which should
have to know this hierarchy exists), so the door that *does* know about
``ServiceError`` is the one place the mapping can live.
"""

from __future__ import annotations

import base64
import concurrent.futures
import dataclasses
import threading
import time
from typing import Any

import httpx

from .. import models
from ..familiar import character_plan, contract, doors, llama_client, retrieval, router
from ..pipelines import llama
from . import familiar_log
from .errors import ServiceError

#: The fixed vocabulary a :class:`FamiliarRefusal` names itself with. Every
#: value here is one a pane can act on distinctly; anything this module
#: cannot classify more precisely falls back to ``"http"``, the same bucket a
#: non-200 response from the server itself lands in.
REASONS = frozenset(
    {"lease", "card", "vram", "backoff", "missing", "unhealthy", "too_large", "parse", "http"}
)


class FamiliarRefusal(ServiceError):
    """Familiar refused to answer. ``reason`` is one of :data:`REASONS`."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        assert reason in REASONS, f"unknown FamiliarRefusal reason {reason!r}"
        self.reason = reason


def _reason_for(message: str) -> str:
    """Classify a ``RuntimeError`` message from ``LlamaServer.ensure_started``
    or ``llama_client.chat`` into one of :data:`REASONS`. Substring matching
    against the fixed sentences those two modules raise -- see their own
    docstrings/source for the exact wording each refusal uses."""
    if "GPU job holds the card" in message:
        return "lease"
    if "does not match any card this weights pin was validated against" in message:
        return "card"
    if "not enough VRAM headroom" in message:
        return "vram"
    if "refusing to respawn for another" in message:
        return "backoff"
    if "not found at" in message or "failed manifest verification" in message:
        return "missing"
    if "did not become healthy in time" in message:
        return "unhealthy"
    if "was stopped during startup" in message:
        # The 2026-09-23 audit (familiar-02): ``ensure_started``'s health
        # poll raises this exact sentence when it finds ``self._proc`` gone
        # mid-poll -- and the only thing that clears ``_proc`` out from under
        # a poll in progress is ``stop_for_gpu_job`` (see that method's own
        # docstring), which sets ``_leased = True`` first. That is a lease
        # taken, not an http-shaped failure; unmatched, it fell to "http" and
        # the pane offered a retry instead of the lease story.
        return "lease"
    if "exited during startup" in message:
        # The 2026-09-23 audit (familiar-02): the child process dying on its
        # own during the health poll is the same "never became healthy"
        # story as the timeout branch just above, and was unmatched here for
        # the same reason -- the exact sentence only exists in
        # ``ensure_started``'s poll loop, never in the timeout path.
        return "unhealthy"
    if "probably by an orphaned llama-server.exe" in message or (
        "is held by" in message and "llama-server" in message
    ) or "is still held after terminating pid" in message:
        # The 2026-09-23 audit (familiar-02): every sentence ``_reclaim_port``
        # raises (port already in use by an orphan, held by a foreign
        # process, held by a llama-server this Realmspinner did not start,
        # held by another still-running Realmspinner, or still held after
        # terminating the orphan) means the same thing to a caller as
        # "did not become healthy in time" -- the server could not be
        # reached -- but matched none of the substrings above and fell to
        # the generic "http" bucket.
        return "unhealthy"
    if "has no key file" in message:
        # The 2026-09-20 audit (familiar-02): llama_client._headers raises
        # this RuntimeError when a chat lands after LlamaServer.stop() has
        # already cleared server.key_path -- the same "server is simply not
        # answering anymore" story the sibling race (a raw FileNotFoundError
        # when the key *file* vanishes mid-read, between _headers reading it
        # and pipelines.llama._release_key_file unlinking it) was already
        # mapped to "unhealthy" for by the 2026-09-18 familiar-01 fix, a few
        # lines earlier in _call's own OSError handler. Without this branch
        # the message matched none of the substrings above and fell through
        # to "http", so a stopped server was classified two different ways
        # depending on which side of the race a request landed on.
        return "unhealthy"
    if "bytes, over the" in message and "byte ceiling" in message:
        return "too_large"
    return "http"


#: How long the door waits on the loop. It must outlast the *whole* coroutine,
#: not just the chat round trip: a cold ``ensure_started`` alone may take
#: ``STARTUP_TIMEOUT`` before the client's own ``CHAT_TIMEOUT`` even begins.
#: Matching only the chat timeout let the door give up on a cold start with an
#: unmapped ``TimeoutError`` while the request ran on regardless.
LOOP_TIMEOUT = llama.STARTUP_TIMEOUT + llama_client.CHAT_TIMEOUT + 30.0


def _call(
    svc: Any,
    messages: list[dict[str, str]],
    *,
    skill: str | None,
    sampling: dict[str, float | int],
    expected_card_sha: str | None,
    slot: int = router.SKILL_SLOT,
    response_format: dict[str, Any] | None = None,
) -> str:
    if getattr(svc, "worker", None) is None:
        # ``call_on_loop`` returns None with no worker rather than raising, so
        # without this a chat would "succeed" with no reply at all. Not a
        # round trip that ever reached the model, so it is not logged below --
        # there is no request to have a record of.
        raise FamiliarRefusal("Familiar is not available in this session.", reason="missing")
    started = time.monotonic()
    reply: str | None = None
    error: FamiliarRefusal | None = None
    try:
        reply = svc.call_on_loop(
            lambda: llama_client.chat(
                svc.worker.familiar,
                messages,
                slot=slot,
                sampling=sampling,
                skill=skill,
                expected_card_sha=expected_card_sha,
                response_format=response_format,
            ),
            timeout=LOOP_TIMEOUT,
        )
        return reply
    except (TimeoutError, concurrent.futures.TimeoutError) as exc:
        # The 2026-09-17 audit (familiar-05): this used to just refuse and
        # walk away, leaving the abandoned chat coroutine running on the loop
        # with its llama-server slot -- a retry then queued behind a request
        # nobody was still waiting for, and its late end was logged nowhere.
        # The cancel itself lives in svc.call_on_loop (service/core.py),
        # the one place every caller of that primitive shares, not here --
        # this except clause still only maps the timeout to a refusal; the
        # `finally` below still records this request (reply=None, this
        # error) regardless of whether the coroutine had already been
        # cancelled by the time it runs.
        error = FamiliarRefusal(
            "Familiar did not answer in time -- try again.", reason="unhealthy"
        )
        raise error from exc
    except ValueError as exc:
        # contract.output_budget's own refusal: the prompt left no room for a
        # real reply. Not a RuntimeError, so it needs its own except clause,
        # but it is the same "too_large" bucket a reply over MAX_RESPONSE_BYTES
        # lands in -- both mean "this request does not fit", one on the way in
        # and one on the way out.
        error = FamiliarRefusal(str(exc), reason="too_large")
        raise error from exc
    except httpx.HTTPError as exc:
        # The 2026-09-16 audit (familiar-01): httpx's own transport exceptions
        # (ReadTimeout, ConnectError, RemoteProtocolError, ...) are not
        # TimeoutError/concurrent.futures.TimeoutError, so without this clause
        # they escaped _call unclassified -- and this is the failure mode
        # LOOP_TIMEOUT's own commentary says it exists to catch: the client's
        # CHAT_TIMEOUT always elapses before the outer LOOP_TIMEOUT possibly
        # could (LOOP_TIMEOUT = STARTUP_TIMEOUT + CHAT_TIMEOUT + 30), so a
        # slow or hung llama-server reply raised a raw httpx exception instead
        # of the "did not answer in time" sentence this bucket exists for.
        error = FamiliarRefusal(
            "Familiar did not answer in time -- try again.", reason="unhealthy"
        )
        raise error from exc
    except RuntimeError as exc:
        error = FamiliarRefusal(str(exc), reason=_reason_for(str(exc)))
        raise error from exc
    except OSError as exc:
        # familiar-01 (2026-09-18 audit, second run): a chat racing
        # LlamaServer.stop() can land between _headers reading the key file
        # (llama_client.py's _headers, called synchronously inside chat())
        # and pipelines.llama._release_key_file unlinking it -- a raw
        # FileNotFoundError, which is an OSError, not a TimeoutError,
        # ValueError, httpx.HTTPError or RuntimeError, so it used to escape
        # this door unclassified. The stopped server is simply not
        # answering anymore, the same story as an unhealthy llama-server.
        error = FamiliarRefusal(
            "Familiar did not answer in time -- try again.", reason="unhealthy"
        )
        raise error from exc
    finally:
        # dev-only (REALMSPINNER_FAMILIAR_LOG, familiar_log.py's own docstring):
        # one record per model round trip regardless of which of the above
        # exits it took, so try/finally rather than a record call duplicated
        # at every return/raise site.
        if familiar_log.enabled():
            familiar_log.record(
                "request",
                skill=skill,
                slot=slot,
                messages=messages,
                sampling=sampling,
                response_format=response_format,
                elapsed_ms=round((time.monotonic() - started) * 1000, 1),
                reply=reply,
                error_type=None if error is None else type(error).__name__,
                error_message=None if error is None else error.message,
                reason=None if error is None else error.reason,
            )


def _with_image(messages: list[dict[str, str]], image: bytes | None) -> list[dict[str, Any]]:
    """*messages*, with the last message's own ``content`` turned into an
    OpenAI-style content-part list carrying *image* alongside its text --
    unchanged when *image* is ``None`` (every caller before vision existed).

    Only the *last* message: every builder in :mod:`~.contract`
    (:func:`~.contract.build_chat_messages`, :func:`~.contract.build_messages`,
    :func:`~.contract.build_repair_messages`, ...) puts the live turn -- the
    one a picture actually illustrates -- last, and none of those functions
    themselves may change shape (a fine-tuning harness and a training script
    both depend on their exact plain-string output), so the wrapping happens
    here, one layer up, never inside :mod:`~.contract` itself.

    PNG bytes only, base64-encoded into a ``data:image/png;base64,...`` URI
    -- images never leave the machine (the offline invariant): this never
    touches the network itself, it only shapes the payload
    :func:`~..familiar.llama_client.chat` sends to the *resident*,
    loopback-only server.
    """
    if image is None:
        return messages
    out = [dict(m) for m in messages]
    last = out[-1]
    text = last["content"]
    last["content"] = [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()},
        },
    ]
    return out


def chat_reply(
    svc: Any, prompt: str, history: tuple[Any, ...] = (), image: bytes | None = None
) -> str:
    """A plain-chat reply on the base testing pin: no card, no scene.

    ``expected_card_sha=None`` -- the same "no card in play" shape
    ``LlamaServer._check_card_sha`` never refuses -- so plain chat starts
    Familiar in every mode, on the testing pin, regardless of whether the
    Clay fine-tune has ever been installed.

    *image* (2026-09-24, vision) -- optional PNG bytes, attached to the
    user's own turn (:func:`_with_image`) -- lets a question be asked about a
    picture (a reference image, or Clay's own ghost render) rather than text
    alone. ``None`` (every pre-vision caller) sends the same request as
    always.
    """
    messages = _with_image(contract.build_chat_messages(prompt, history), image)
    return _call(
        svc, messages, skill=None, sampling=contract.SAMPLING["chat"], expected_card_sha=None
    )


#: How many follow-up turns Familiar's self-repair loop may send after a
#: refusal from any gate *past* the card gate -- ``contract.parse_calls``,
#: ``contract.allowed_calls`` (both checked here), or the scratch run's own
#: refusal (``studio.assistant.preview.run_scratch``, checked by the caller
#: and fed back in through :func:`clay_repair`). Two, not open-ended, for the
#: same reason a person is not asked to keep re-typing a prompt forever: each
#: retry is a full extra round trip through an 8,192-token slot (``contract.
#: TRAINED_WINDOW``), and a model whose *second* corrected attempt still
#: fails a gate it was shown the exact refusal for is not going to be talked
#: into a third one -- the honest answer at that point is the refusal itself,
#: not another guess. The budget is shared across *both* refusal sources
#: (:data:`ClayBuildResult.repairs_used` is what lets :func:`clay_repair`
#: know how much of it :func:`clay_build`'s own parse/vocabulary retries
#: already spent), because the fix a person cares about is "how many times
#: did Familiar guess wrong before giving up", not which gate did the
#: refusing.
MAX_REPAIRS = 2


@dataclasses.dataclass(frozen=True)
class ClayBuildResult:
    """What :func:`clay_build`/:func:`clay_repair` hand back on success: the
    gated ``calls``, the reply they were parsed out of, and the self-repair
    bookkeeping a later gate needs if it has to resume this same budget.

    *reply* is carried along (not just the parsed ``calls``) because a gate
    this module cannot see -- the scratch run, off the frame thread in
    ``studio.assistant.preview.run_scratch`` -- may still refuse *this*
    reply's calls once they are actually tried against a document clone, and
    :func:`~.contract.build_repair_messages` needs the assistant's own prior
    turn played back verbatim to build a correct follow-up.

    *repairs_used* is how much of :data:`MAX_REPAIRS` this call already
    spent (0 when the very first reply passed the gate clean) -- the number
    a caller chaining a further :func:`clay_repair` call must pass back in so
    the two never together exceed the cap. *retries* is that same call's own
    refusal sentences, oldest first, for the caller to show in the
    transcript, each one visibly marked as a retry -- never the final,
    successful reply's own sentence, since that one is not a refusal.
    """

    calls: list[dict]
    reply: str
    repairs_used: int = 0
    retries: tuple[str, ...] = ()


def _gate_reply(skill: str, reply: str) -> tuple[list[dict] | None, str | None]:
    """*reply* through Clay's own gate -- :func:`~.contract.parse_calls` then
    :func:`~.contract.allowed_calls` -- worded exactly as :func:`clay_build`
    has always refused with, so a caller showing this sentence (in a retry
    turn, or as the final refusal) never drifts from what it said before
    self-repair existed. -> ``(calls, None)`` on success, ``(None, sentence)``
    naming which of the two failed.
    """
    calls, error = contract.parse_calls(reply)
    if calls is None:
        return None, f"Familiar's reply could not be read as Clay tool calls ({error})."
    # The 2026-09-18 audit (familiar-05): contract.allowed_calls exists
    # precisely to answer "what did this frozen card actually train the
    # model to name" (parsed from the card's own clay_batch schema, not the
    # live agent_clay registry) but was never actually called here, so a
    # reply naming a tool outside the card's own vocabulary -- a decoding
    # fluke, or a weights pin whose card has drifted from what this build
    # ships -- ran through to the preview/apply path unchecked. Refused in
    # the same "parse" bucket an unreadable reply already uses: a call this
    # door does not trust is no more actionable than one it could not read.
    allowed = contract.allowed_calls(skill)
    for call in calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name not in allowed:
            return None, (
                f"Familiar's reply named a tool ({name!r}) outside Clay's "
                "trained vocabulary."
            )
    return calls, None


def _build_with_repairs(
    svc: Any,
    prompt: str,
    scene: dict[str, Any] | None,
    card_sha: str,
    *,
    image: bytes | None = None,
    seed_reply: str | None = None,
    seed_refusal: str | None = None,
    repairs_used: int = 0,
) -> ClayBuildResult:
    """The self-repair loop shared by :func:`clay_build` (a fresh build) and
    :func:`clay_repair` (resuming after a refusal this module could not see
    itself). One gated reply, then up to :data:`MAX_REPAIRS` follow-up turns
    (:func:`~.contract.build_repair_messages`) when :func:`_gate_reply`
    refuses -- *repairs_used* is whatever budget an earlier refusal (of
    either kind) already spent, carried in so the two sources never together
    exceed the cap.

    **Every attempt re-runs the full gate**, never just the half that failed
    last time: a corrected reply could just as easily land back in the
    *other* half (a fixed vocabulary violation whose new call is itself
    unparseable, say) -- trusting the previous attempt's own gate for
    anything it did not just re-check would ship a reply this loop never
    actually validated.

    Raises :class:`FamiliarRefusal` (reason ``"parse"``) with the last
    attempt's own refusal sentence once the budget is spent -- the exact
    shape :func:`clay_build` always refused with, before self-repair
    existed.

    *image* (2026-09-24, vision) -- optional PNG bytes, attached
    (:func:`_with_image`) to *every* attempt's own last turn, retries
    included: a corrected reply is still being asked to build against the
    same picture, so a retry that dropped it would be answering a different
    question than the one that was refused.
    """
    retries: list[str] = []
    if seed_reply is None:
        messages = contract.build_messages("clay", prompt, scene)
    else:
        messages = contract.build_repair_messages(
            "clay", prompt, scene, seed_reply, seed_refusal or ""
        )
    while True:
        reply = _call(
            svc,
            _with_image(messages, image),
            skill="clay",
            sampling=contract.SAMPLING["clay"],
            expected_card_sha=card_sha,
        )
        calls, refusal = _gate_reply("clay", reply)
        if calls is not None:
            return ClayBuildResult(
                calls=calls, reply=reply, repairs_used=repairs_used, retries=tuple(retries)
            )
        if repairs_used >= MAX_REPAIRS:
            raise FamiliarRefusal(refusal, reason="parse")
        retries.append(refusal)
        repairs_used += 1
        messages = contract.build_repair_messages("clay", prompt, scene, reply, refusal)


def clay_build(
    svc: Any, prompt: str, scene: dict[str, Any], image: bytes | None = None
) -> ClayBuildResult:
    """A Clay build: the frozen card, *scene*, *prompt* -> a
    :class:`ClayBuildResult` whose ``calls`` a preview run then executes one
    at a time.

    *scene* is expected already compacted (``contract.compact_scene``) --
    the caller (the Familiar dock, on the frame thread) reads ``clay_scene``'s
    full structured content and compacts it *before* handing the request to
    this door, because that read has to happen against the live document on
    the frame thread anyway (the same GL-adjacent reason ``clay_scene``
    itself is drawn from ``agent_clay``, not from here).

    Refuses with reason ``"card"`` *before any request* when the running
    weights pin was never validated against Clay's frozen card -- checked
    here rather than left to ``ensure_started``'s own card-sha check, because
    that check only runs at *spawn*: a server already running for plain chat
    (started with ``expected_card_sha=None``, which is never refused) would
    otherwise happily serve a Clay prompt it was never trained to answer.
    Run A's own measurement on the previous (Gemma 4 E2B) pin is why this
    matters: base Gemma scored 0% door acceptance on Clay builds against its
    own fine-tune's 74% (``dev/measurements/2026-09-12-clay-assistant-run-A.md``).
    The untuned base model is not trusted with Clay builds on the current
    (Qwen3-VL-4B-Instruct) pin either -- no eval corpus has scored it yet --
    so a fine-tune's card sha is required before this door opens, and serving
    a Clay request on the testing pin alone would not fail loudly -- it would
    just fail, every time, with no tool call in the reply for
    :func:`~.contract.parse_calls` to find.

    A reply that fails the gate (:func:`_gate_reply`: unparseable, or naming
    a tool outside Clay's trained vocabulary) is not refused on the spot --
    self-repair resends one follow-up turn (:func:`~.contract.
    build_repair_messages`, carrying the model's own failed reply and the
    refusal sentence back to it) and re-runs the *full* gate on what comes
    back, up to :data:`MAX_REPAIRS` times, before finally refusing exactly as
    this door always has.

    *image* (2026-09-24, vision) -- optional PNG bytes (the dock's own
    attach, or Clay's ghost render on a revision -- see
    ``studio.assistant.ui``'s own docstring), attached to every attempt's own
    turn via :func:`_with_image`.
    """
    card_sha = contract.card_sha("clay")
    if card_sha not in models.FAMILIAR_MODELS["familiar_gguf"].card_shas:
        raise FamiliarRefusal(
            "Building in Clay needs the trained Familiar model (familiar_v1.0).",
            reason="card",
        )
    return _build_with_repairs(svc, prompt, scene, card_sha, image=image)


def clay_repair(
    svc: Any,
    prompt: str,
    scene: dict[str, Any] | None,
    failed_reply: str,
    refusal: str,
    repairs_used: int,
    image: bytes | None = None,
) -> ClayBuildResult:
    """Resume Clay's self-repair loop after a refusal from a gate
    :func:`clay_build` cannot see for itself: the scratch run
    (``studio.assistant.preview.run_scratch``), which only runs once a caller
    already has a gated ``clay_batch`` and tries it against a real document
    clone, off the frame thread, well after :func:`clay_build` has already
    returned.

    *repairs_used* is whatever :func:`clay_build` (or an earlier call here,
    for a build with more than one scratch-run refusal in a row) already
    spent -- this call's own attempt counts as one more against that same
    budget before it ever sends a request, so the caller must not call this
    once :data:`MAX_REPAIRS` is already reached; :func:`_build_with_repairs`
    still re-checks the cap on every attempt after that, the same as
    :func:`clay_build`'s own loop, so a second scratch-run refusal in a row
    is refused rather than granted a third full budget.

    Re-checks the card gate -- the same reason :func:`clay_build` checks it
    up front rather than trusting ``ensure_started``'s own spawn-time check:
    an earlier reply in this same build having come from the Clay-trained
    pin is not proof the server still serves it now.
    """
    card_sha = contract.card_sha("clay")
    if card_sha not in models.FAMILIAR_MODELS["familiar_gguf"].card_shas:
        raise FamiliarRefusal(
            "Building in Clay needs the trained Familiar model (familiar_v1.0).",
            reason="card",
        )
    return _build_with_repairs(
        svc,
        prompt,
        scene,
        card_sha,
        image=image,
        seed_reply=failed_reply,
        seed_refusal=refusal,
        repairs_used=repairs_used + 1,
    )


# ---------------------------------------------------------------------------
# T6: the router. ``ask`` is the one door a caller who does not already know
# which skill it wants should use; ``chat_reply``/``clay_build`` above stay
# public for the two callers that already know (plain chat with no router
# involved is still reachable directly, and Clay's own explicit Build button
# is deliberately router-free -- see ``studio/assistant/ui.py``'s own
# docstring for why a button that already knows it means "build" should not
# pay for a routing round trip only to be told what it already is).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Answer:
    """What :func:`ask` hands back: which skill the router picked, plus
    whichever of *text*/*calls* that skill actually produced.

    *skill* is reported **honestly even when the routed skill fell back to
    plain chat** (an unbuilt skill -- character/create/navigate/other -- or
    clay_build/clay_edit routed outside Clay) -- the pane and the tests both
    want to see what the router actually decided, not just what ran.
    """

    skill: str
    text: str | None
    citations: tuple[retrieval.Citation, ...] = ()
    calls: list[dict] | None = None
    #: Self-repair's own bookkeeping, populated only when *calls* came from
    #: :func:`clay_build` (``skill`` a Clay route) -- see
    #: :class:`ClayBuildResult`'s own docstring for what each carries.
    #: Defaulted so every pre-self-repair ``Answer(skill, text, ...)`` call
    #: site (including every hand-built one in a test) keeps working
    #: unchanged: no repair ever ran, nothing to retry.
    reply: str | None = None
    repairs_used: int = 0
    retries: tuple[str, ...] = ()
    #: T8: what a routed ``navigate``/``create`` decided to *do*, for the
    #: caller (``studio/assistant/doors.py``, on the frame thread) to act out
    #: -- ``{"kind": "navigate", "target": <destination key>}`` or
    #: ``{"kind": "draft", "asset_type": ..., "prompt": ...}``. T7 adds a
    #: third shape, ``{"kind": "character_plan", "prompt", "plan",
    #: "overrides", "summary"}`` -- *not* acted out automatically the way a
    #: navigate/draft is: the caller shows the plan and waits for a press
    #: (``studio/assistant/ui.py``'s own plan card) before ever calling
    #: :func:`create_planned_character`. ``None`` for every other skill,
    #: including a navigate/create/character call that fell back to chat
    #: because the model named nothing usable -- that case answers with
    #: *text* instead, exactly like an unbuilt skill does.
    action: dict[str, Any] | None = None


#: The retrieval index is expensive enough to build (~0.37s over 840 chunks,
#: ``retrieval.Index.build``'s own docstring) that it should happen once per
#: process, not once per Manual question -- and lazily, so a session that
#: never asks Familiar a Manual question never pays for it at all. Guarded by
#: a lock because ``ask`` runs on a ``TaskRunner`` worker thread and two
#: Manual questions in flight at once (unlikely -- ``ctx.submit`` already
#: refuses a second concurrent chat under the same key -- but not impossible
#: across two different tabs) must not race to build it twice.
_index_lock = threading.Lock()
_index: retrieval.Index | None = None


def _manual_index() -> retrieval.Index:
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = retrieval.Index.build()
    return _index


def _ask_manual(svc: Any, prompt: str) -> Answer:
    """A Manual question: retrieve, then either answer with citations or --
    with nothing retrieved -- say so with no model call at all.

    The no-citations case is deliberately *not* routed through
    :func:`chat_reply`: a plain-chat reply here would answer from whatever
    the base model happens to know about Realmspinner (nothing reliable -- it was
    never given the Manual), which is a worse answer than the honest "the
    Manual doesn't cover that" and costs a full round trip to produce.
    """
    citations = _manual_index().search(prompt)
    if not citations:
        return Answer(skill="manual", text="The Manual doesn't cover that.", citations=())
    messages = contract.build_manual_messages(prompt, citations)
    reply = _call(
        svc,
        messages,
        # skill=None: manual answers are not in contract.SIZED_SKILLS (no
        # trained window to fit -- SAMPLING["manual"]'s max_tokens is used
        # verbatim), so sizing off the skill name would do nothing extra
        # here beyond what passing None already does.
        skill=None,
        sampling=contract.SAMPLING["manual"],
        expected_card_sha=None,
    )
    return Answer(skill="manual", text=reply, citations=contract.cited(reply, citations))


def _ask_navigate(
    svc: Any, prompt: str, history: tuple[Any, ...], destinations: tuple[doors.Destination, ...]
) -> Answer:
    """A ``navigate`` route: ask which of *destinations* the message means,
    then hand back an :attr:`Answer.action` for the caller to act out --
    never an act itself, since this module never reaches the palette or
    ``state`` (the offline/layering invariant: ``service/`` acts through
    what a caller hands it, the acting half stays at studio level, see
    ``studio/assistant/doors.py``'s own docstring).

    Falls back to :func:`chat_reply` when the model names nothing usable
    (``doors.parse_target`` returned ``None``) -- the same "don't act, just
    answer" contract :func:`~.router.parse_route` already keeps for the
    router itself.
    """
    keys = tuple(d.key for d in destinations)
    reply = _call(
        svc,
        doors.build_navigate_messages(prompt, destinations),
        skill=None,
        sampling=contract.SAMPLING["navigate"],
        expected_card_sha=None,
        response_format={
            "type": "json_schema",
            "json_schema": {"schema": doors.navigate_schema(keys)},
        },
    )
    target = doors.parse_target(reply, keys)
    if target is None:
        return Answer(skill="navigate", text=chat_reply(svc, prompt, history))
    return Answer(skill="navigate", text=None, action={"kind": "navigate", "target": target})


def _ask_create(
    svc: Any, prompt: str, history: tuple[Any, ...], asset_types: tuple[tuple[str, str], ...]
) -> Answer:
    """A ``create`` route: ask for an asset type and a short prompt among
    *asset_types* (``create_assets.ASSET_TYPE_OPTIONS``, handed in by the
    caller -- see :func:`ask`'s own docstring for why this module never
    imports ``create_assets`` itself), then hand back an :attr:`Answer.action`
    for the caller to draft, never submit.

    Falls back to :func:`chat_reply` when the model's reply cannot be
    trusted as a draft (``doors.parse_draft`` returned ``None``), the same
    contract :func:`_ask_navigate` keeps for a routed navigation."""
    reply = _call(
        svc,
        doors.build_create_messages(prompt, asset_types),
        skill=None,
        sampling=contract.SAMPLING["create"],
        expected_card_sha=None,
        response_format={
            "type": "json_schema",
            "json_schema": {"schema": doors.create_schema(asset_types)},
        },
    )
    draft = doors.parse_draft(reply, asset_types)
    if draft is None:
        return Answer(skill="create", text=chat_reply(svc, prompt, history))
    asset_type, drafted_prompt = draft
    return Answer(
        skill="create",
        text=None,
        action={"kind": "draft", "asset_type": asset_type, "prompt": drafted_prompt},
    )


def _ask_character(
    svc: Any, prompt: str, history: tuple[Any, ...], character_options: dict[str, Any]
) -> Answer:
    """A ``character`` route: propose a plan among *character_options*
    (``{"families", "movements", "directions", "size_range"}`` -- see
    ``character_plan.build_character_messages``'s own docstring), then run
    it through :func:`~..service.characters.recipe_from_prompt` as a **dry
    run that mints nothing**. T7's whole point: the plan is free to discard,
    and only the user's own Create press (``studio/assistant/ui.py``'s
    ``submit_character``) ever calls :func:`create_planned_character`.

    Falls back to :func:`chat_reply` when the model names no species
    *character_options* offers (:func:`~.character_plan.parse_plan` returned
    ``None``) -- the same "don't act, just answer" contract
    :func:`_ask_navigate`/:func:`_ask_create` already keep.

    A refusal from ``recipe_from_prompt`` (a theme the species does not
    paint, a movement that is not this skeleton's clip, ...) is deliberately
    **not** re-raised as a :class:`FamiliarRefusal`: it is a conversational
    answer with a sentence the user can act on (try another species, drop a
    movement) exactly the way that door already answers a person typing into
    Create's own form, not a Familiar-specific failure -- and unlike a
    :class:`FamiliarRefusal` (Familiar itself could not answer), the model
    answered fine here; it is the *plan* the recipe would not build.
    """
    from . import characters as svc_characters
    from .errors import Invalid

    reply = _call(
        svc,
        character_plan.build_character_messages(prompt, character_options),
        skill=None,
        sampling=contract.SAMPLING["character"],
        expected_card_sha=None,
        response_format={
            "type": "json_schema",
            "json_schema": {"schema": character_plan.character_schema(character_options)},
        },
    )
    plan, dropped = character_plan.parse_plan(reply, character_options)
    if plan is None:
        return Answer(skill="character", text=chat_reply(svc, prompt, history))

    overrides = character_plan.plan_overrides(plan)
    try:
        built = svc_characters.recipe_from_prompt(svc, prompt, overrides=overrides)
    except Invalid as exc:
        return Answer(skill="character", text=str(exc))

    fam_label = next(
        (f["label"] for f in character_options["families"] if f["key"] == plan["family"]),
        plan["family"],
    )
    summary = {
        "species": fam_label,
        "theme": plan.get("theme"),
        "movements": plan.get("movements", []),
        "directions": plan.get("directions"),
        "cells": built["cells"],
        "estimate_minutes": built["estimate_minutes"],
        # The 2026-09-18 audit (familiar-02): docs/manual/20-overview.md
        # promises that a word the plan itself could not act on is "named
        # under the plan rather than silently dropped" -- but until this
        # merge only recipe_from_prompt's own ``ignored`` rode along here.
        # An unknown movement, an unoffered theme or an out-of-ladder
        # direction/size that character_plan.parse_plan drops on its own
        # (never reaching recipe_from_prompt at all) is what
        # character_plan.parse_plan's own ``dropped`` return now carries,
        # merged in here so the plan card shows every word the plan could
        # not act on, not only the ones the recipe rejects.
        "ignored": [*dropped, *built["ignored"]],
    }
    return Answer(
        skill="character",
        text=None,
        action={
            "kind": "character_plan",
            "prompt": prompt,
            "plan": plan,
            "overrides": overrides,
            "summary": summary,
        },
    )


def create_planned_character(
    svc: Any, prompt: str, overrides: dict[str, Any], name: str | None = None
) -> dict[str, Any]:
    """Mint the character T7's plan card proposed.

    ``recipe_from_prompt`` is **re-run** rather than trusted from the plan's
    own moment -- the world may have changed in between (a download that
    finished, a species that no longer resolves the way it did when the
    model last saw it) -- and only the fresh recipe it returns is handed to
    :func:`~..service.characters.create_character`, the same door Create's
    own submit calls (``modes/create/ui/settings_character.py``'s ``submit``): a
    character queued from a plan is queued exactly the way a person's own
    Generate press would have queued it, comment or no comment from an
    agent session in between.

    Raises whatever either door raises (``service.errors.Invalid`` for a
    refusal, e.g. "Rigging needs Blender, which is not installed.") --
    unlike :class:`FamiliarRefusal`, this is a plain service call with
    nothing Familiar-specific about the mint itself, so it carries no
    ``reason`` vocabulary of its own; the caller (``studio/assistant/ui.py``)
    shows it exactly like any other refused submit.
    """
    from . import characters as svc_characters

    built = svc_characters.recipe_from_prompt(svc, prompt, overrides=overrides)
    return svc_characters.create_character(
        svc, built["recipe"], name=name, prompt=prompt, resolution=built["resolution"]
    )


def ask(
    svc: Any,
    prompt: str,
    *,
    mode: str,
    history: tuple[Any, ...],
    scene: dict[str, Any] | None = None,
    destinations: tuple[doors.Destination, ...] = (),
    asset_types: tuple[tuple[str, str], ...] = (),
    character_options: dict[str, Any] | None = None,
    image: bytes | None = None,
) -> Answer:
    """Route *prompt* to a skill, then answer it.

    *image* (2026-09-24, vision) -- optional PNG bytes, forwarded to
    whichever door actually answers (:func:`chat_reply` for every fallback,
    :func:`clay_build` for a routed build) -- never to the router request
    itself, which only ever needs *prompt*'s own text to pick a skill.

    A refusal raised while routing (a lease, a missing worker, a timeout, ...)
    propagates as a :class:`FamiliarRefusal` exactly as :func:`chat_reply`/
    :func:`clay_build` already do -- routing is one more request to the same
    door, not a special case that should swallow what that door raises.

    A build is only ever dispatched to :func:`clay_build` -- which carries
    its own card gate -- when the router actually named a Clay skill *and*
    the caller is in Clay *and* a scene was captured for it; every other
    combination (an unbuilt skill, or a Clay-shaped request routed from
    outside Clay with no scene to build against) falls back to
    :func:`chat_reply`, with :attr:`Answer.skill` still naming what the
    router picked so a caller can tell "answered as chat" from "no skill
    wanted this at all".

    T8: ``navigate``/``create`` are the same shape, one step removed --
    *destinations*/*asset_types* are handed in by the caller rather than
    read here, because building either list touches studio-level machinery
    (the palette's own commands, ``app_settings.CATEGORIES``,
    ``create_assets``) this module must not import (the offline/layering
    invariant -- see ``studio/assistant/doors.py``'s own docstring for where
    that acting half actually lives). Empty (the caller's default, and what
    every pre-T8 call site still passes) means "nothing to route to", so
    both fall back to plain chat exactly like an unbuilt skill does, rather
    than asking the model to choose among zero options.

    T7: ``character`` is the same "empty means fall back" contract, but
    *character_options* is handed in for a narrower reason than
    *destinations*/*asset_types* -- what it is built from
    (``service.characters.character_options``) is already service-layer
    data, not studio machinery, so the caller (``studio/assistant/ui.py``)
    only exists as the one place already computing it, cached, for Create's
    own form (``modes/create/engine/character.options``); this module still never
    reads a registry to build it fresh.
    """
    route_reply = _call(
        svc,
        contract.build_router_messages(prompt, mode),
        skill="router",
        sampling=contract.SAMPLING["router"],
        expected_card_sha=None,
        slot=router.ROUTER_SLOT,
        response_format={"type": "json_schema", "json_schema": {"schema": router.ROUTE_SCHEMA}},
    )
    skill = router.parse_route(route_reply)

    if skill == "manual":
        return _ask_manual(svc, prompt)

    if skill in ("clay_build", "clay_edit") and mode == "clay" and scene is not None:
        # clay_edit shares clay_build's own frozen card and card gate: the
        # Clay card's own behaviour paragraph already covers "change what's
        # there" as well as "add something new" (``cards/clay-1.txt``), and
        # nothing about the card gate is build-specific -- a model untrained
        # on Clay tool calls at all is exactly as unable to edit them.
        build_result = clay_build(svc, prompt, scene, image=image)
        return Answer(
            skill=skill,
            text=None,
            calls=build_result.calls,
            reply=build_result.reply,
            repairs_used=build_result.repairs_used,
            retries=build_result.retries,
        )

    if skill == "navigate" and destinations:
        return _ask_navigate(svc, prompt, history, destinations)

    if skill == "create" and asset_types:
        return _ask_create(svc, prompt, history, asset_types)

    if skill == "character" and character_options:
        return _ask_character(svc, prompt, history, character_options)

    return Answer(skill=skill, text=chat_reply(svc, prompt, history, image=image))
