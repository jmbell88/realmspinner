"""The service door onto Familiar: a chat reply, or a Clay build's tool calls.

Both functions block -- called from a ``TaskRunner`` worker thread (see
``studio/tasks.py``), never the frame thread -- and both go through
``svc.call_on_loop`` to run the actual ``httpx`` request on the
``warlock-loop`` thread where ``LlamaServer`` and its asyncio lock live.

**Every refusal here is a :class:`FamiliarRefusal`**, a ``ServiceError`` that
adds one field, ``reason``, drawn from a fixed vocabulary
(:data:`REASONS`) so the pane can choose an icon or an action (an Install…
button, a "try again" hint) without parsing ``message`` -- the same idea as
``ServiceError.field`` for a form control, one level more specific because a
chat refusal has no control to point at.

``pipelines.llama.LlamaServer.ensure_started`` and
``pipelines.llama_client.chat`` both raise plain ``RuntimeError``/``ValueError``
with a handful of fixed sentences (see each module's own docstring for the
exact wording); :func:`_reason_for` classifies them by substring rather than
by a new exception hierarchy in ``pipelines/`` -- neither module may import
``service`` (the offline/layering invariant: ``pipelines/`` is reached by a
training script and by ``queue.py``, neither of which should have to know
this hierarchy exists), so the door that *does* know about ``ServiceError``
is the one place the mapping can live.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

from .. import models
from ..pipelines import llama, llama_client
from ..studio.familiar import contract, router
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
) -> str:
    if getattr(svc, "worker", None) is None:
        # ``call_on_loop`` returns None with no worker rather than raising, so
        # without this a chat would "succeed" with no reply at all.
        raise FamiliarRefusal("Familiar is not available in this session.", reason="missing")
    try:
        return svc.call_on_loop(
            lambda: llama_client.chat(
                svc.worker.familiar,
                messages,
                slot=router.SKILL_SLOT,
                sampling=sampling,
                skill=skill,
                expected_card_sha=expected_card_sha,
            ),
            timeout=LOOP_TIMEOUT,
        )
    except (TimeoutError, concurrent.futures.TimeoutError) as exc:
        raise FamiliarRefusal(
            "Familiar did not answer in time -- try again.", reason="unhealthy"
        ) from exc
    except ValueError as exc:
        # contract.output_budget's own refusal: the prompt left no room for a
        # real reply. Not a RuntimeError, so it needs its own except clause,
        # but it is the same "too_large" bucket a reply over MAX_RESPONSE_BYTES
        # lands in -- both mean "this request does not fit", one on the way in
        # and one on the way out.
        raise FamiliarRefusal(str(exc), reason="too_large") from exc
    except RuntimeError as exc:
        raise FamiliarRefusal(str(exc), reason=_reason_for(str(exc))) from exc


def chat_reply(svc: Any, prompt: str, history: tuple[Any, ...] = ()) -> str:
    """A plain-chat reply on the base testing pin: no card, no scene.

    ``expected_card_sha=None`` -- the same "no card in play" shape
    ``LlamaServer._check_card_sha`` never refuses -- so plain chat starts
    Familiar in every mode, on the testing pin, regardless of whether the
    Clay fine-tune has ever been installed.
    """
    messages = contract.build_chat_messages(prompt, history)
    return _call(
        svc, messages, skill=None, sampling=contract.SAMPLING["chat"], expected_card_sha=None
    )


def clay_build(svc: Any, prompt: str, scene: dict[str, Any]) -> list[dict]:
    """A Clay build: the frozen card, *scene*, *prompt* -> the parsed
    ``calls`` list a preview run then executes one at a time.

    *scene* is expected already compacted (``contract.compact_scene``) --
    the caller (the bottom pane, on the frame thread) reads ``clay_scene``'s
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
    Run A's own measurement is why this matters: base Gemma 4 E2B scored 0%
    door acceptance on Clay builds (the fine-tune: 74%,
    ``docs/measurements/2026-09-14-clay-assistant-ablation.md``), so serving
    a Clay request on the testing pin would not fail loudly -- it would just
    fail, every time, with no tool call in the reply for :func:`~.contract.
    parse_calls` to find.
    """
    card_sha = contract.card_sha("clay")
    if card_sha not in models.FAMILIAR_MODELS["familiar_gguf"].card_shas:
        raise FamiliarRefusal(
            "Building in Clay needs the trained Familiar model (familiar_v1.0).",
            reason="card",
        )
    messages = contract.build_messages("clay", prompt, scene)
    reply = _call(
        svc,
        messages,
        skill="clay",
        sampling=contract.SAMPLING["clay"],
        expected_card_sha=card_sha,
    )
    calls, error = contract.parse_calls(reply)
    if calls is None:
        raise FamiliarRefusal(
            f"Familiar's reply could not be read as Clay tool calls ({error}).",
            reason="parse",
        )
    return calls
