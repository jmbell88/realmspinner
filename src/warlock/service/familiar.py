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
import dataclasses
import threading
from typing import Any

import httpx

from .. import models
from ..pipelines import llama, llama_client
from ..studio.familiar import character_plan, contract, doors, retrieval, router
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
    slot: int = router.SKILL_SLOT,
    response_format: dict[str, Any] | None = None,
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
                slot=slot,
                sampling=sampling,
                skill=skill,
                expected_card_sha=expected_card_sha,
                response_format=response_format,
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
        raise FamiliarRefusal(
            "Familiar did not answer in time -- try again.", reason="unhealthy"
        ) from exc
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
    ``dev/measurements/2026-09-12-clay-assistant-run-A.md``), so serving
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


# ---------------------------------------------------------------------------
# T6: the router. ``ask`` is the one door a caller who does not already know
# which skill it wants should use; ``chat_reply``/``clay_build`` above stay
# public for the two callers that already know (plain chat with no router
# involved is still reachable directly, and Clay's own explicit Build button
# is deliberately router-free -- see ``studio/familiar_ui.py``'s own
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
    #: T8: what a routed ``navigate``/``create`` decided to *do*, for the
    #: caller (``studio/familiar_doors.py``, on the frame thread) to act out
    #: -- ``{"kind": "navigate", "target": <destination key>}`` or
    #: ``{"kind": "draft", "asset_type": ..., "prompt": ...}``. T7 adds a
    #: third shape, ``{"kind": "character_plan", "prompt", "plan",
    #: "overrides", "summary"}`` -- *not* acted out automatically the way a
    #: navigate/draft is: the caller shows the plan and waits for a press
    #: (``studio/familiar_ui.py``'s own plan card) before ever calling
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
    the base model happens to know about Warlock (nothing reliable -- it was
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
    ``studio/familiar_doors.py``'s own docstring).

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
    and only the user's own Create press (``studio/familiar_ui.py``'s
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
    plan = character_plan.parse_plan(reply, character_options)
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
        "ignored": built["ignored"],
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
    own submit calls (``panes/settings_character.py``'s ``submit``): a
    character queued from a plan is queued exactly the way a person's own
    Generate press would have queued it, comment or no comment from an
    agent session in between.

    Raises whatever either door raises (``service.errors.Invalid`` for a
    refusal, e.g. "Rigging needs Blender, which is not installed.") --
    unlike :class:`FamiliarRefusal`, this is a plain service call with
    nothing Familiar-specific about the mint itself, so it carries no
    ``reason`` vocabulary of its own; the caller (``studio/familiar_ui.py``)
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
) -> Answer:
    """Route *prompt* to a skill, then answer it.

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
    invariant -- see ``studio/familiar_doors.py``'s own docstring for where
    that acting half actually lives). Empty (the caller's default, and what
    every pre-T8 call site still passes) means "nothing to route to", so
    both fall back to plain chat exactly like an unbuilt skill does, rather
    than asking the model to choose among zero options.

    T7: ``character`` is the same "empty means fall back" contract, but
    *character_options* is handed in for a narrower reason than
    *destinations*/*asset_types* -- what it is built from
    (``service.characters.character_options``) is already service-layer
    data, not studio machinery, so the caller (``studio/familiar_ui.py``)
    only exists as the one place already computing it, cached, for Create's
    own form (``panes/settings_character.options``); this module still never
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
        calls = clay_build(svc, prompt, scene)
        return Answer(skill=skill, text=None, calls=calls)

    if skill == "navigate" and destinations:
        return _ask_navigate(svc, prompt, history, destinations)

    if skill == "create" and asset_types:
        return _ask_create(svc, prompt, history, asset_types)

    if skill == "character" and character_options:
        return _ask_character(svc, prompt, history, character_options)

    return Answer(skill=skill, text=chat_reply(svc, prompt, history))
