"""Familiar's in-session conversation: state, submission and the bottom
pane's expanded body. T5 of the Familiar programme.

**Why this lives at studio level, not inside ``studio/familiar/``.** Exactly
``studio/assistant/preview.py``'s own reason (see that module's docstring):
this reaches ``agent_clay`` (to read the live scene for a Build request) and
``clay_view``/``clay_mode`` (to show and apply a ghost preview), all of which
``studio/familiar/`` is pinned never to import, even lazily.

**State lives on the frame thread, submission does not block it.** Every
network round trip goes through ``ctx.submit`` under one of two keys
(:data:`CHAT_KEY`, :data:`BUILD_KEY`) running a service-layer call
(``service.familiar.chat_reply``/``clay_build``) on a ``TaskRunner`` worker
thread; the frame thread only ever reads :class:`FamiliarUIState` and calls
:func:`on_task_done` when a result lands, the same shape every other mode's
``ctx.submit``/``on_task_done`` pair already uses (``studio/modes/clay/mode.py``'s module
docstring states the rule this module follows). Landing a build that came
back with calls is itself two more of these round trips, not one inline
computation: :data:`LAND_KEY` carries the ``clay_batch`` run itself off the
frame thread too (the 2026-09-17 audit, familiar-01) -- see
:func:`_submit_build_preview`/:func:`_land_build_preview`.

**Threads are keyed by mode/tab, never by conversation content.** See
``studio/familiar/threads.py``'s own docstring: a thread is display and
refinement history, never fed back to the model as context -- the model
still sees only the current prompt (and, for Clay, the compacted scene).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The task-runner keys a chat send / Clay build submit under. Prefixed
#: ``familiar/`` (not ``familiar-``, unlike every Clay key's ``clay-``
#: prefix) simply because there is only ever one of each in flight at a time
#: -- no per-tab or per-job suffix is needed, since ``ctx.submit`` already
#: refuses a second submit under the same key while the first is running,
#: which is exactly "disable Send/Build while busy" for free.
CHAT_KEY = "familiar/chat"
BUILD_KEY = "familiar/build"
#: T7's Create press on a pending character plan. Its own key, not
#: ``CHAT_KEY`` -- a plan sits waiting for a press with no chat in flight,
#: and the two must be free to overlap the way any two independent
#: ``ctx.submit`` keys already are.
CHARACTER_KEY = "familiar/character"
#: The batch itself -- landing a build's ``clay_batch`` run, split off
#: ``CHAT_KEY``/``BUILD_KEY`` by the 2026-09-17 audit (familiar-01): up to
#: ``agent_clay.BATCH_MAX`` calls, booleans included, used to run inline
#: inside ``on_task_done``, on the pygame frame thread -- 624-705 ms wall for
#: 16 uv-sphere adds and 15 unions with no GPU or weights involved, freezing
#: the app for ~40 frames on an ordinary Build. Its own key for the same
#: reason ``CHAT_KEY``/``BUILD_KEY`` get theirs -- see :func:`_submit_build_
#: preview`/:func:`_land_build_preview` for the two-phase shape this key
#: exists to carry.
LAND_KEY = "familiar/land"

#: The bottom pane's expanded height, in design pixels, before
#: ``bottom_pane.max_height``'s own window-relative clamp -- room for a short
#: scrollable transcript plus one input row. Not derived from content height:
#: a pane that resizes itself every frame to fit a variable-length transcript
#: would fight the layout it sits in rather than the user's own toggle. This
#: is only the *starting* height now (2026-09-16) -- see :func:`pane_height`,
#: which is what every caller actually reads.
EXPANDED_H = 160.0

#: Where a user-dragged pane height is persisted (``ctx.settings``), a plain
#: top-level key the same way ``main.AGENT_SERVER_SETTING`` is -- one number,
#: not worth a dict entry of its own.
HEIGHT_SETTING = "familiar_pane_height"

#: The narrowest a user may drag the pane to before expanded stops being
#: worth the screen it costs -- room for roughly two transcript lines and the
#: input row.
MIN_EXPANDED_H = 100.0

#: The widest a user may drag the pane to, before ``bottom_pane.max_height``'s
#: own window-relative clamp gets a say. Generous -- a long back-and-forth is
#: exactly when someone wants the transcript tall -- but still bounded, so a
#: stray dependency on a settings file cannot hand this a value that eats the
#: whole window.
MAX_EXPANDED_H = 480.0


def pane_height(ctx: Any) -> float:
    """The expanded pane's height before :func:`bottom_pane.max_height`'s own
    clamp -- the user's last drag if there was one, else :data:`EXPANDED_H`."""
    stored = ctx.settings.get(HEIGHT_SETTING, EXPANDED_H)
    try:
        value = float(stored)
    except (TypeError, ValueError):
        return EXPANDED_H
    return min(max(value, MIN_EXPANDED_H), MAX_EXPANDED_H)


def set_pane_height(ctx: Any, value: float) -> None:
    """Persist a drag of the handle above the pane, clamped the same way
    :func:`pane_height` reads it back."""
    ctx.settings.set(HEIGHT_SETTING, min(max(value, MIN_EXPANDED_H), MAX_EXPANDED_H))


@dataclass
class FamiliarUIState:
    """The bottom pane's own session state -- never persisted, same as
    ``familiar.threads.Threads`` (see that module's docstring): a refusal or
    a pending preview describes this run, not something worth restoring
    across a restart."""

    #: Whether the pane is drawn at :data:`EXPANDED_H` (clamped) rather than
    #: ``bottom_pane.COLLAPSED_H``. Toggled by the user, never by a message
    #: arriving -- an answer landing while the pane is collapsed should not
    #: itself pop the layout open under whatever the user is doing.
    expanded: bool = False
    #: The input line's live text, kept here (not a local in ``draw``) so a
    #: reply landing mid-type does not require choosing what happens to
    #: unsent text -- there is exactly one input line for the whole app, and
    #: it survives a Done landing untouched.
    input_text: str = ""
    #: ``""`` | ``"chat"`` | ``"build"`` -- which key is in flight, so the
    #: pane can show "Thinking..." against the right control without asking
    #: ``ctx.busy`` twice for two different-shaped questions. "build" now
    #: spans two task keys in turn (``BUILD_KEY``/``CHAT_KEY``, then
    #: ``LAND_KEY`` -- see :func:`_submit_build_preview`), set again once the
    #: first lands, so this stays honest for the whole time a build is
    #: outstanding rather than dropping to "" while the batch itself runs.
    thinking: str = ""
    #: The last refusal's ``FamiliarRefusal.reason`` (see
    #: ``service.familiar.REASONS``), or ``None`` when the last outcome was
    #: not a refusal at all.
    reason: str | None = None
    #: The last refusal's or Done's own message, shown verbatim -- a lease
    #: refusal keeps its exact sentence (T5 brief), and every other refusal
    #: is already written for a person to read.
    message: str | None = None
    #: A ready Clay preview: the parsed tool calls, the scratch document they
    #: ran against, the diff computed from it, and which tab it previews for.
    #: All four are set together (:func:`_land_build_preview`) and cleared
    #: together (:func:`_clear_preview`) -- a partial set would let Apply run
    #: against a diff that does not describe ``preview_scratch``.
    preview_calls: list[dict] | None = None
    preview_scratch: Any = None
    preview_diff: Any = None
    preview_tab_uid: str = ""
    #: T7: a pending character plan -- ``service.familiar.Answer.action``'s
    #: own ``"character_plan"`` shape, verbatim, or ``None`` with nothing
    #: waiting on a press. Cleared by whichever of Create/Open in
    #: Create/Discard the user presses; landing a *new* plan while one is
    #: already pending simply replaces it -- the same "the document changed,
    #: preview again" spirit ``_staleness_refusal`` keeps, one plan at a time.
    plan: dict[str, Any] | None = None


def ensure(ctx: Any) -> FamiliarUIState:
    """The pane's state, built on first use -- ``AppState`` deliberately
    knows nothing about what a mode (or, here, the pane) keeps, the same
    reason ``clay_mode.ensure`` gives for ``ctx.state.clay``."""
    ui = ctx.state.familiar
    if ui is None:
        ui = FamiliarUIState()
        ctx.state.familiar = ui
    return ui


def install(ctx: Any) -> None:
    """Attach ``ctx.familiar_threads`` and register its ``drop`` with
    ``docmodes.TAB_CLOSED``, once. Called by the App at startup; a test that
    wants the same wiring calls this too, rather than appending to
    ``TAB_CLOSED`` itself, so "the listener is registered" is always proven
    through the one door that actually registers it in the running app.

    Idempotent: a bound method compares equal to another bound method of the
    same instance and function (``MethodType.__eq__``), so calling this
    twice on the same ``ctx`` does not double-register -- the guard a second
    ``App`` in one process (or a test that calls ``install`` more than once)
    needs.
    """
    from ...familiar import threads

    if getattr(ctx, "familiar_threads", None) is None:
        ctx.familiar_threads = threads.Threads()
    from .. import docmodes

    if ctx.familiar_threads.drop not in docmodes.TAB_CLOSED:
        docmodes.TAB_CLOSED.append(ctx.familiar_threads.drop)


# --- thread key --------------------------------------------------------------


def _active_tab_uid(ctx: Any) -> str:
    """Clay's active tab uid, or ``""`` in every other mode (or with no Clay
    tab open yet) -- Clay is the only mode with a Build action today, and
    every other mode's chat shares :data:`~.familiar.threads.STUDIO` via
    ``Threads.key_for``'s own falsy-uid rule."""
    mode = str(getattr(ctx.state, "mode", ""))
    if mode != "clay":
        return ""
    clay_state = getattr(ctx.state, "clay", None)
    return getattr(clay_state, "active_uid", "") if clay_state is not None else ""


def thread_key(ctx: Any) -> tuple[str, str]:
    """The ``(mode, tab_uid)`` key this frame's conversation reads/writes --
    see ``threads.Threads.key_for``."""
    from ...familiar import threads

    mode = str(getattr(ctx.state, "mode", ""))
    return threads.Threads.key_for(mode, _active_tab_uid(ctx))


# --- submission ----------------------------------------------------------


def _pending_ghost(ctx: Any, tab_uid: str) -> Any:
    """The scratch document of the preview pending on *tab_uid*, or ``None``.

    A prompt sent while a ghost is showing refines that ghost rather than the
    real document (user, 2026-09-16: follow-ups used to be impossible until
    Apply or Discard), so this is what the scene is read from and what the
    reply's calls later run on top of."""
    ui = ensure(ctx)
    if not tab_uid or ui.preview_calls is None or ui.preview_tab_uid != tab_uid:
        return None
    return ui.preview_scratch


def _capture_scene(ctx: Any, tab_uid: str) -> dict[str, Any] | None:
    """The compact scene for *tab_uid*, or ``None`` with no tab open.

    Read on the frame thread, exactly like ``submit_build`` always did --
    ``agent_clay.call`` walks the live ``ClayDoc``, and that read must
    happen against *this* frame's document, not whatever it is by the time
    a worker thread gets around to it. Factored out (T6) so
    :func:`submit_chat`'s routed path captures the same scene a router
    decision of "build" needs, without a second copy of this read.
    """
    if not tab_uid:
        return None
    from ...familiar import contract
    from ..modes.clay.agent import dispatch as agent_clay

    ghost = _pending_ghost(ctx, tab_uid)
    if ghost is not None:
        from . import preview as familiar_preview

        # Read from a clone of the ghost, so a refinement is written against
        # what the user is looking at and the ghost itself is never touched.
        scratch_ctx = familiar_preview.build(ghost)
        ctx, tab_uid = scratch_ctx, scratch_ctx.tab_uid
    session = agent_clay.Session(tab_uid=tab_uid)
    scene_result = agent_clay.call(ctx, session, "clay_scene", {})
    structured = (
        scene_result.get("structuredContent") if isinstance(scene_result, dict) else None
    )
    return contract.compact_scene(structured or {})


def _character_options(ctx: Any) -> dict[str, Any]:
    """The plan-shaped slice of ``service.characters.character_options`` for
    :func:`~.familiar.character_plan.build_character_messages`/
    ``character_schema`` -- species with their themes, every shipped
    movement across every archetype's own skeleton, the direction ladder and
    the custom-size range.

    Read through ``character_engine.options`` -- Create's own frame-thread
    cache, keyed on the palette directory's stamp
    (``modes/create/engine/character.py``'s own docstring) -- rather than
    calling ``service.characters.character_options`` fresh: it is the one
    place already paying for this read every frame Create's own form is
    open, and a second, uncached copy here would answer the same registries
    a frame later for no reason. Computed on the frame thread, alongside
    *destinations*/*asset_types* (T8's own precedent) rather than inside the
    worker closure below: unlike those two this touches no ``ctx.state``
    gate, but it does touch ``ctx.state.preview``'s own cache slot, which is
    frame-thread state exactly like the Clay scene capture is.
    """
    from ...kernels.rig import cliplib
    from ..modes.create.engine import character as character_engine

    raw = character_engine.options(ctx)
    families = [
        {"key": f["key"], "label": f["label"], "themes": [t["key"] for t in f["themes"]]}
        for f in raw["families"]
    ]
    templates = {a["template"] for a in raw["archetypes"]}
    movements = sorted({name for t in templates for name in cliplib.shipped_clip_names(t)})
    return {
        "families": families,
        "movements": movements,
        "directions": list(raw["directions"]),
        "size_range": tuple(raw["troupe"]["logical_size_range"]),
    }


def submit_chat(ctx: Any, prompt: str) -> bool:
    """Send *prompt* through Familiar's router. -> whether it was accepted.

    Refused (returns ``False``, no toast -- the disabled Send button already
    said why) when *prompt* is blank or a chat is already in flight. Appends
    the user's own turn to the thread immediately, before the network call
    even starts, so it appears in the transcript the same frame it was sent
    rather than only once a reply lands.

    T6: this now submits ``service.familiar.ask`` (route, then answer)
    rather than ``chat_reply`` directly -- Send no longer means "always
    plain chat"; the router decides. In Clay, with a tab open, the scene is
    captured up front (:func:`_capture_scene`) the same way :func:`submit_build`
    always has, so a message the router sends to ``clay_build``/``clay_edit``
    has a scene to build against without a second frame-thread round trip
    once the route comes back. The explicit **Build** button stays wired to
    :func:`submit_build` directly -- a press that already means "build"
    should not pay for a routing decision only to be told what it already
    knew.
    """
    prompt = prompt.strip()
    if not prompt or ctx.busy(CHAT_KEY):
        return False
    from ...familiar import threads

    key = thread_key(ctx)
    ctx.familiar_threads.append(key, threads.Turn("user", prompt))
    history = ctx.familiar_threads.get(key)[:-1]

    mode = str(getattr(ctx.state, "mode", ""))
    tab_uid = _active_tab_uid(ctx)
    scene = _capture_scene(ctx, tab_uid) if mode == "clay" else None

    # T8: computed here, on the frame thread, for ``_capture_scene``'s own
    # reason -- ``familiar_doors.destinations`` reads ``palette.commands``,
    # which reads ``ctx.state`` (the mode gate, the active document), so the
    # list the router is offered has to describe *this* frame, not whatever
    # it is by the time a worker thread gets around to it.
    from ..modes.create.engine import assets as create_assets
    from . import doors as familiar_doors

    destinations = familiar_doors.destinations(ctx)
    asset_types = create_assets.ASSET_TYPE_OPTIONS
    # T7: same frame-thread treatment as the two lines above -- see
    # ``_character_options``'s own docstring for why this one is cheap
    # rather than studio-gated.
    character_options = _character_options(ctx)

    from ...service import familiar as svc_familiar
    from ...service import familiar_log

    # T5's dev-only log (WARLOCK_FAMILIAR_LOG): minted here, on the frame
    # thread, and carried into ``run()``'s closure so every record the
    # request/outcome pair produces on the worker thread groups back to the
    # same round trip this submit accepted.
    exchange_id = familiar_log.new_exchange_id()
    refine = _pending_ghost(ctx, tab_uid)

    def run() -> Any:
        with familiar_log.exchange(exchange_id):
            return svc_familiar.ask(
                ctx.svc,
                prompt,
                mode=mode,
                history=history,
                scene=scene,
                destinations=tuple(destinations),
                asset_types=asset_types,
                character_options=character_options,
            )

    tag = {
        "thread_key": key,
        "tab_uid": tab_uid,
        "scene_captured": scene is not None,
        "refine": refine,
        "exchange": exchange_id,
    }
    if not ctx.submit(CHAT_KEY, run, tag=tag):
        return False
    if familiar_log.enabled():
        with familiar_log.exchange(exchange_id):
            familiar_log.record(
                "submit",
                submit_kind="chat",
                prompt=prompt,
                mode=mode,
                tab_uid=tab_uid,
                scene=scene,
                refine=refine is not None,
            )
    ui = ensure(ctx)
    ui.thinking = "chat"
    ui.reason = None
    ui.message = None
    return True


def submit_build(ctx: Any, prompt: str) -> bool:
    """Send *prompt* as a Clay build request. -> whether it was accepted.

    Refused, with no request ever sent, outside Clay or with no tab open --
    a Build button drawn only in Clay, against the active tab, should never
    reach this with either untrue, but the check is repeated here rather
    than trusted to the caller because this is also the door a test drives
    directly. Router-free by design (T6 brief): this is the explicit,
    always-build action, unlike Send's routed path in :func:`submit_chat`.
    """
    prompt = prompt.strip()
    tab_uid = _active_tab_uid(ctx)
    if not prompt or not tab_uid or ctx.busy(BUILD_KEY):
        return False
    scene = _capture_scene(ctx, tab_uid)

    from ...familiar import threads

    key = thread_key(ctx)
    ctx.familiar_threads.append(key, threads.Turn("user", prompt))

    from ...service import familiar as svc_familiar
    from ...service import familiar_log

    exchange_id = familiar_log.new_exchange_id()
    refine = _pending_ghost(ctx, tab_uid)

    def run() -> list[dict]:
        with familiar_log.exchange(exchange_id):
            return svc_familiar.clay_build(ctx.svc, prompt, scene)

    tag = {"thread_key": key, "tab_uid": tab_uid, "refine": refine, "exchange": exchange_id}
    if not ctx.submit(BUILD_KEY, run, tag=tag):
        return False
    if familiar_log.enabled():
        with familiar_log.exchange(exchange_id):
            familiar_log.record(
                "submit",
                submit_kind="build",
                prompt=prompt,
                mode="clay",
                tab_uid=tab_uid,
                scene=scene,
                refine=refine is not None,
            )
    ui = ensure(ctx)
    ui.thinking = "build"
    ui.reason = None
    ui.message = None
    return True


# --- landing ---------------------------------------------------------------


def _reason_and_message(done: Any) -> tuple[str | None, str]:
    error = done.error
    reason = getattr(error, "reason", None) if error is not None else None
    message = done.message or (str(error) if error is not None else "Something went wrong.")
    return reason, message


def _say(ctx: Any, thread_key: Any, text: str, *, toast: bool = True) -> None:
    """Append a Familiar turn to *thread_key*'s transcript and, unless
    *toast* is false, toast it too.

    The "done" confirmation (user, 2026-09-16: "after the assistant
    considers itself done, it needs to send a confirmation to the user that
    it is done with its job" -- transcript turn plus toast, the user's own
    choice): before this, a build preview landing or a refusal only ever set
    ``ui.message``/the ghost silently, so with the pane collapsed there was
    nothing to see. Tolerates a missing thread/``ctx.familiar_threads``
    exactly like the CHARACTER_KEY branch this was factored out of always
    did -- a caller with no thread key (no document tab, an early return
    before one was captured) still gets its toast.
    """
    if not text:
        # A refusal with no sentence (``familiar_preview.apply`` answering
        # ``ok: False`` without a ``message``) must not become a ``None``
        # turn: ``draw_expanded`` concatenates the prefix onto it.
        return
    threads_obj = getattr(ctx, "familiar_threads", None)
    if thread_key is not None and threads_obj is not None:
        from ...familiar import threads

        threads_obj.append(thread_key, threads.Turn("familiar", text))
    if toast:
        toast_fn = getattr(ctx, "toast", None)
        if toast_fn is not None:
            toast_fn(text)


def _log_outcome(done: Any, tag: dict) -> None:
    """Dev-only (WARLOCK_FAMILIAR_LOG): one ``outcome`` record per landed
    task, carrying the same exchange id its ``submit``/``request`` records
    used -- see ``familiar_log.py``'s own docstring."""
    from ...service import familiar_log

    if not familiar_log.enabled():
        return
    from ...service.familiar import Answer

    result = done.result
    skill = text = calls = action = None
    if isinstance(result, Answer):
        skill, text, calls, action = result.skill, result.text, result.calls, result.action
    elif isinstance(result, list):
        calls = result
    reason = getattr(done.error, "reason", None) if done.error is not None else None
    with familiar_log.exchange(tag.get("exchange")):
        familiar_log.record(
            "outcome",
            key=done.key,
            ok=done.ok,
            skill=skill,
            text=text,
            calls=calls,
            action=action,
            reason=reason,
            message=None if done.ok else done.message,
        )


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from the app for :data:`CHAT_KEY`/:data:`BUILD_KEY`/
    :data:`CHARACTER_KEY`, the same way every other mode's ``on_task_done``
    is called from ``main._on_task_done`` (any ``"familiar/"``-prefixed key
    reaches here)."""
    ui = ensure(ctx)
    tag = done.tag if isinstance(done.tag, dict) else {}
    _log_outcome(done, tag)

    if done.key == CHAT_KEY:
        ui.thinking = ""
        if done.ok:
            ui.reason = None
            ui.message = None
            result = done.result
            from ...service.familiar import Answer

            if isinstance(result, Answer) and result.calls is not None:
                # The router sent this one to Clay -- land it exactly like
                # an explicit Build's own result, calls and all.
                _submit_build_preview(
                    ctx,
                    ui,
                    tag.get("tab_uid", ""),
                    result.calls,
                    refine=tag.get("refine"),
                    thread_key=tag.get("thread_key"),
                    exchange=tag.get("exchange"),
                )
                return
            if (
                isinstance(result, Answer)
                and result.action is not None
                and result.action.get("kind") == "character_plan"
            ):
                # T7: a character plan waits for a press -- it is never acted
                # out the way a navigate/draft is (see ``Answer.action``'s
                # own docstring), just shown, so the plan card can draw
                # itself from ``ui.plan`` instead of the transcript.
                ui.plan = result.action
                text = "Here's a plan -- create it, open it in Create, or discard."
                citations = ()
            elif isinstance(result, Answer) and result.action is not None:
                # T8: a navigate/create route -- act it out here, on the
                # frame thread (``familiar_doors`` reaches the palette,
                # ``state.set_mode`` and Create's form, none of which a
                # worker thread may touch), then say what happened (or why
                # not) in the thread the same way every other reply does.
                text = _run_door(ctx, result.action)
                citations = ()
            elif isinstance(result, Answer):
                text, citations = result.text, result.citations
            else:
                # Defensive, not exercised by a real ``ask`` call: a bare
                # string result is still treated as an uncited chat reply
                # rather than crashing on ``.text``.
                text, citations = (result if isinstance(result, str) else str(result)), ()
            thread = tag.get("thread_key")
            threads_obj = getattr(ctx, "familiar_threads", None)
            if thread is not None and threads_obj is not None and text is not None:
                from ...familiar import threads

                threads_obj.append(thread, threads.Turn("familiar", text, citations))
        else:
            # A failed CHAT_KEY task is also where a routed build's own
            # refusal lands (``ask`` calls ``clay_build`` directly for a
            # ``clay_build``/``clay_edit`` route, so its ``FamiliarRefusal``
            # fails the whole task rather than coming back as an ``Answer``)
            # -- said here, once, for every CHAT_KEY failure rather than
            # only the build-routed ones, since there is nothing in ``done``
            # that tells the two apart.
            ui.reason, ui.message = _reason_and_message(done)
            _say(ctx, tag.get("thread_key"), ui.message)
        return

    if done.key == BUILD_KEY:
        ui.thinking = ""
        if not done.ok:
            ui.reason, ui.message = _reason_and_message(done)
            _say(ctx, tag.get("thread_key"), ui.message)
            if tag.get("refine") is None:
                _clear_preview(ui)
            return
        calls = done.result if isinstance(done.result, list) else []
        ui.reason = None
        ui.message = None
        _submit_build_preview(
            ctx,
            ui,
            tag.get("tab_uid", ""),
            calls,
            refine=tag.get("refine"),
            thread_key=tag.get("thread_key"),
            exchange=tag.get("exchange"),
        )
        return

    if done.key == LAND_KEY:
        # Phase two: the worker's ``clay_batch`` run has landed -- see
        # :func:`_land_build_preview`'s own docstring for why every
        # staleness fact is asked again here rather than trusted from the
        # submit that started it.
        _land_build_preview(ctx, ui, done)
        return

    if done.key == CHARACTER_KEY:
        if not done.ok:
            # A plain ``service.errors.Invalid`` (Blender missing, a stale
            # theme) has no ``.reason`` -- ``_reason_and_message`` already
            # answers ``None`` for that, the same shape a non-Familiar
            # refusal is shown in everywhere else.
            ui.reason, ui.message = _reason_and_message(done)
            return
        ui.reason = None
        ui.message = None
        _say(ctx, tag.get("thread_key"), "Character queued -- it will appear in the Library.")
        return


def _run_door(ctx: Any, action: dict[str, Any]) -> str:
    """T8: act out a routed ``navigate``/``create`` decision. -> the
    sentence the transcript should show.

    A shape neither of :func:`familiar_doors.navigate`/:func:`draft_in_create`
    itself refuses (an ``action`` this build does not recognise -- there is
    none today, but a future skill's own action kind must not crash the
    frame loop reading a reply that landed) answers plainly rather than
    raising."""
    from . import doors as familiar_doors

    kind = action.get("kind")
    if kind == "navigate":
        return familiar_doors.navigate(ctx, str(action.get("target") or ""))
    if kind == "draft":
        return familiar_doors.draft_in_create(
            ctx, str(action.get("asset_type") or ""), str(action.get("prompt") or "")
        )
    return "I'm not sure what to do with that."


def _refusal_sentence(result: dict) -> str:
    """The door's own sentence for a refused ``clay_batch``. A batch's text is
    its whole structured payload dumped as JSON, so the readable sentence is
    the one on the entry named by ``stopped_at``; a refusal of the batch
    itself (a malformed entry, say) has no such entry and keeps its text."""
    structured = result.get("structuredContent") or {}
    stopped_at = structured.get("stopped_at")
    results = structured.get("results") or []
    if isinstance(stopped_at, int) and 0 <= stopped_at < len(results):
        result = results[stopped_at]
    content = result.get("content") or []
    if content and isinstance(content[0], dict) and content[0].get("text"):
        return str(content[0]["text"])
    return "Familiar's build could not be previewed."


def _clear_preview(ui: FamiliarUIState) -> None:
    ui.preview_calls = None
    ui.preview_scratch = None
    ui.preview_diff = None
    ui.preview_tab_uid = ""


def _preview_sentence(diff: Any) -> str:
    """The "done" sentence a clean preview lands with (2026-09-16 brief: a
    transcript turn naming what the preview does, not just the silent
    ghost). Reads :class:`~.clay.scratch.PreviewDiff` -- ``added``/
    ``removed`` and ``changed`` minus ``added`` (an added object is not also
    counted as "changed") -- singular/plural per clause, and a clause is
    dropped rather than read as "adds 0 objects" when its count is zero.
    Removals are named too, not just additions and changes."""
    if diff.empty:
        return "Done, but the build changed nothing."
    added = len(diff.added)
    removed = len(diff.removed)
    changed = len(diff.changed - diff.added)
    clauses = []
    if added:
        clauses.append(f"adds {added} object{'s' if added != 1 else ''}")
    if removed:
        clauses.append(f"removes {removed} object{'s' if removed != 1 else ''}")
    if changed:
        clauses.append(f"changes {changed}")
    body = " and ".join(clauses) if clauses else "changes the document"
    return f"Done: the preview {body}. Apply to keep it or Discard to drop it."


def _log_preview(exchange_id: Any, *, diff: Any = None, refusal: str | None = None) -> None:
    """Dev-only (WARLOCK_FAMILIAR_LOG): one ``preview`` record per landed
    build -- diff counts on a clean preview, the refusal sentence
    otherwise."""
    from ...service import familiar_log

    if not familiar_log.enabled():
        return
    fields: dict[str, Any] = {}
    if refusal is not None:
        fields["refusal"] = refusal
    if diff is not None:
        fields.update(added=len(diff.added), removed=len(diff.removed), changed=len(diff.changed))
    with familiar_log.exchange(exchange_id):
        familiar_log.record("preview", **fields)


def _staleness_refusal(
    state: Any, tab: Any, tab_uid: str, ui: FamiliarUIState, refine: Any
) -> str | None:
    """The two facts a build preview's landing depends on, both re-asked by
    :func:`_land_build_preview` after the worker returns as well as checked
    here by :func:`_submit_build_preview` before it ever submits -- the
    2026-09-17 audit (familiar-01) split what used to be one inline check
    into two call sites once the batch itself moved to a worker, since the
    tab can close, the user can switch tabs, or the ghost being refined can
    be applied/discarded during the time the batch spends off the frame
    thread, none of which the pre-submit check alone could still see by the
    time the result lands.
    """
    if tab is None or tab_uid != state.active_uid:
        # The tab this build was requested against closed, or the user
        # switched to another tab, while Familiar was thinking -- the same
        # "preview again" sentence every other familiar_preview refusal uses,
        # for the same reason: there is nothing left to preview against.
        #
        # The 2026-09-15 audit's agents-01: this used to check only that the
        # tab still existed, not that it was still the one on screen. ``set_
        # preview``/``_ghost_draws`` carry no document identity of their own
        # on the shared ``clay_view``, so a build that landed after a tab
        # switch painted the ghost over whichever tab was now in front --
        # Apply already refused a stale base at that point, but the display
        # never did. Landing now refuses the same way Apply always has,
        # rather than showing a ghost for a document nobody is looking at.
        return "the document changed -- preview again"
    if refine is not None and (ui.preview_scratch is not refine or ui.preview_tab_uid != tab_uid):
        # Applied or discarded while the model was thinking: these calls were
        # written against a ghost that is gone, and on the real document they
        # would address objects that are not there.
        return "the preview changed -- preview again"
    return None


def _submit_build_preview(
    ctx: Any,
    ui: FamiliarUIState,
    tab_uid: str,
    calls: list[dict],
    *,
    refine: Any = None,
    thread_key: Any = None,
    exchange: Any = None,
) -> None:
    """Phase one of landing a build: the staleness checks and the scratch
    clone -- both cheap, both fine on the frame thread -- then hand the
    batch itself to a worker under :data:`LAND_KEY`.

    The 2026-09-17 audit (familiar-01): this function used to also run the
    batch (``familiar_preview.run_scratch(..., "clay_batch", ...)``) right
    here, inline on the frame thread -- up to ``agent_clay.BATCH_MAX`` calls,
    booleans included, 624-705 ms wall for 16 uv-sphere adds and 15 unions
    with no GPU or weights involved, freezing ``App.frame`` for ~40 frames on
    an ordinary Build. Now this only clones the base document (a numpy copy,
    not a batch of ops -- see ``clay.scratch.clone``) and submits the batch;
    :func:`_land_build_preview` is the second half, called back from
    :func:`on_task_done` once the worker is done, where the diff is taken and
    the ghost is shown.

    *thread_key* -- the same ``(mode, tab_uid)`` the request was submitted
    under -- is where the "done" sentence (:func:`_preview_sentence`, or a
    refusal) lands as a Familiar turn, via :func:`_say`; *exchange* is the
    dev-log id the same round trip's ``submit``/``request`` records used.
    Both are carried into :data:`LAND_KEY`'s own tag so the second phase can
    use them too.
    """
    from ..modes.clay import mode as clay_mode
    from . import preview as familiar_preview

    state = clay_mode.ensure(ctx)
    tab = state.get(tab_uid) if tab_uid else None
    refusal = _staleness_refusal(state, tab, tab_uid, ui, refine)
    if refusal is not None:
        ui.message = refusal
        ui.reason = None
        _log_preview(exchange, refusal=refusal)
        _say(ctx, thread_key, refusal)
        return

    scratch_ctx = familiar_preview.build(refine if refine is not None else tab.doc)

    # One clay_batch, never call by call: the model names objects made earlier
    # in the same reply as {"$ref": "<name>"}, which only a batch resolves --
    # it is the shape the training data, the eval and the door all share. Run
    # one at a time, the first $ref was refused ("Build the Eiffel Tower" came
    # back as clay_boolean's "uids must be a list of integers.", 2026-09-16).
    # This closure touches only ``scratch_ctx`` -- its own, private
    # ``ClayState`` holding nothing but the clone (see ``familiar_preview``'s
    # module docstring) -- never ``ctx.state``, GL or imgui, which is what
    # makes running it off the frame thread safe.
    def run() -> dict:
        return familiar_preview.run_scratch(scratch_ctx, "clay_batch", {"calls": calls})

    tag = {
        "thread_key": thread_key,
        "exchange": exchange,
        "tab_uid": tab_uid,
        "refine": refine,
        "calls": calls,
        "scratch_ctx": scratch_ctx,
    }
    if not ctx.submit(LAND_KEY, run, tag=tag):
        # A second build already landing on this same key -- CHAT_KEY and
        # BUILD_KEY are independent submits, so (rarely) both can land in the
        # same frame. Dropped rather than queued: the disabled Send/Build
        # button is the real guard against this in the ordinary case, and
        # ``ui.thinking`` is left as this branch found it (``on_task_done``
        # already cleared it to "" before calling here), so nothing is stuck
        # "thinking" over a build that was simply never submitted.
        return
    ui.thinking = "build"


def _land_build_preview(ctx: Any, ui: FamiliarUIState, done: Any) -> None:
    """Phase two of landing a build (see :func:`_submit_build_preview`),
    called from :func:`on_task_done` for :data:`LAND_KEY`: the worker's
    ``clay_batch`` run has landed, so this takes the diff and shows the
    ghost -- or refuses, re-asking :func:`_staleness_refusal` the same
    question the submit side already asked, since the tab or the ghost being
    refined can have moved again while the batch ran."""
    from ...kernels.mesh import scratch as clay_scratch
    from ..modes.clay import mode as clay_mode

    ui.thinking = ""
    tag = done.tag if isinstance(done.tag, dict) else {}
    thread_key, exchange = tag.get("thread_key"), tag.get("exchange")
    tab_uid, refine = tag.get("tab_uid", ""), tag.get("refine")
    scratch_ctx, calls = tag.get("scratch_ctx"), tag.get("calls") or []

    state = clay_mode.ensure(ctx)
    tab = state.get(tab_uid) if tab_uid else None
    refusal = _staleness_refusal(state, tab, tab_uid, ui, refine)
    if refusal is not None:
        ui.message = refusal
        ui.reason = None
        _log_preview(exchange, refusal=refusal)
        _say(ctx, thread_key, refusal)
        return

    if not done.ok:
        # ``run_scratch``/``agent_clay.call`` do not normally raise -- a
        # refusal comes back as ``isError`` in the result, handled below --
        # but a defensive path all the same, the same shape a raised
        # BUILD_KEY/CHAT_KEY task already lands with.
        ui.message = done.message or "Something went wrong."
        ui.reason = None
        _log_preview(exchange, refusal=ui.message)
        _say(ctx, thread_key, ui.message)
        return

    result = done.result if isinstance(done.result, dict) else {}
    if result.get("isError"):
        ui.message = _refusal_sentence(result)
        ui.reason = "parse"
        _log_preview(exchange, refusal=ui.message)
        _say(ctx, thread_key, ui.message)
        return

    base_calls = list(ui.preview_calls or []) if refine is not None else []
    scratch_doc = scratch_ctx.state.clay.get(scratch_ctx.tab_uid).doc
    diff = clay_scratch.diff(tab.doc, scratch_doc)
    ui.preview_calls = base_calls + list(calls)
    ui.preview_scratch = scratch_doc
    ui.preview_diff = diff
    ui.preview_tab_uid = tab_uid
    view = getattr(ctx, "clay_view", None)
    if view is not None:
        view.set_preview(diff, scratch_doc)
    _log_preview(exchange, diff=diff)
    _say(ctx, thread_key, _preview_sentence(diff))


def apply_preview(ctx: Any) -> None:
    """Apply the pending preview, if any -- see ``familiar_preview.apply``'s
    own docstring for the refusals it can still return (the document moved
    since the preview was computed)."""
    ui = ensure(ctx)
    if ui.preview_calls is None:
        return
    from ...service import familiar_log
    from . import preview as familiar_preview

    result = familiar_preview.apply(ctx, ui.preview_tab_uid, ui.preview_diff, ui.preview_scratch)
    key = thread_key(ctx)
    if familiar_log.enabled():
        familiar_log.record("apply", ok=bool(result.get("ok")), message=result.get("message"))
    if not result.get("ok"):
        ui.message = result.get("message")
        ui.reason = None
        _say(ctx, key, ui.message)
        return
    ui.message = None
    ui.reason = None
    _clear_preview(ui)
    _say(ctx, key, "Applied to the scene.")


def discard_preview(ctx: Any) -> None:
    """Drop the pending preview, if any, leaving the document untouched."""
    ui = ensure(ctx)
    if ui.preview_calls is None:
        return
    from ...service import familiar_log
    from . import preview as familiar_preview

    familiar_preview.discard(ctx, ui.preview_tab_uid)
    if familiar_log.enabled():
        familiar_log.record("discard")
    key = thread_key(ctx)
    _clear_preview(ui)
    # No toast (unlike Apply): discarding is a quiet "never mind", not a
    # result to be told about across the room.
    _say(ctx, key, "Preview discarded.", toast=False)


# --- T7: the character plan card ------------------------------------------


def _character_fields(plan: dict[str, Any]) -> dict[str, Any]:
    """*plan* (:func:`~.familiar.character_plan.parse_plan`'s own shape) ->
    the Create form's own ``character_*`` field names -- the same subset
    ``troupe_mode.vary_in_create`` writes for a recipe it is varying, built
    only from whichever of *plan*'s fields are actually present (never
    invents a theme, a camera or a name the plan itself does not carry).

    ``movements`` is filtered against ``character_engine.MOVEMENTS`` (the
    closed default trio Create's own action checkboxes offer), the same
    filter ``vary_in_create`` applies to a recipe's ``animations`` keys --
    Create's form has no control for a movement outside that ladder, so one
    from the plan's own wider vocabulary (any shipped clip, not just the
    default three) is silently absent from the brief rather than fed to a
    checkbox that cannot show it; the plan itself still built the full list
    into *its own* ``overrides``, for the Create button's own path.
    """
    from ..modes.create.engine import character as character_engine

    fields: dict[str, Any] = {}
    if "family" in plan:
        fields["character_family"] = plan["family"]
    if "theme" in plan:
        fields["character_theme"] = plan["theme"]
    if "movements" in plan:
        wanted = set(plan["movements"])
        fields["character_actions"] = ",".join(
            name for name, _frames in character_engine.MOVEMENTS if name in wanted
        )
    if "size" in plan:
        fields["character_pixel"] = str(int(plan["size"]))
    if "name" in plan:
        fields["character_name"] = plan["name"]
    return fields


def submit_character(ctx: Any) -> bool:
    """Mint the pending plan's character. -> whether the press was taken.

    Busy-guarded on :data:`CHARACTER_KEY` -- its own key, not :data:`CHAT_KEY`
    -- and refused outright with no plan waiting. The plan is cleared the
    moment the submit is *accepted*, not once it lands: the card should not
    keep showing Create/Open/Discard against a request already in flight,
    the same "clear on accept" rule :func:`draw_expanded` already applies to
    the input line.
    """
    ui = ensure(ctx)
    if ui.plan is None or ctx.busy(CHARACTER_KEY):
        return False
    action = ui.plan
    plan = action["plan"]

    from ...service import familiar as svc_familiar
    from ...service import familiar_log

    exchange_id = familiar_log.new_exchange_id()

    def run() -> Any:
        with familiar_log.exchange(exchange_id):
            return svc_familiar.create_planned_character(
                ctx.svc, action["prompt"], action["overrides"], plan.get("name")
            )

    tag = {"thread_key": thread_key(ctx), "exchange": exchange_id}
    if not ctx.submit(CHARACTER_KEY, run, tag=tag):
        return False
    if familiar_log.enabled():
        with familiar_log.exchange(exchange_id):
            familiar_log.record("submit", submit_kind="character", prompt=action["prompt"])
    ui.plan = None
    ui.reason = None
    ui.message = None
    return True


def open_character_in_create(ctx: Any) -> None:
    """Draft the pending plan's character into Create instead of minting it
    -- :func:`~.familiar_doors.draft_in_create`'s own ``character_fields``
    door, the one T8 built exactly for this caller. Clears the plan and
    appends the door's own sentence to the transcript, the same landing
    every other routed action already gets."""
    ui = ensure(ctx)
    if ui.plan is None:
        return
    action = ui.plan
    from . import doors as familiar_doors

    text = familiar_doors.draft_in_create(
        ctx, "character", action["prompt"], character_fields=_character_fields(action["plan"])
    )
    ui.plan = None
    threads_obj = getattr(ctx, "familiar_threads", None)
    if threads_obj is not None:
        from ...familiar import threads

        threads_obj.append(thread_key(ctx), threads.Turn("familiar", text))


def discard_character_plan(ctx: Any) -> None:
    """Drop the pending plan, if any -- nothing was ever queued, so there is
    nothing to undo, only the card to stop showing."""
    ui = ensure(ctx)
    ui.plan = None


# --- drawing -----------------------------------------------------------


def follow_citation(ctx: Any, citation: Any) -> None:
    """Open the Manual at *citation*'s own chapter/section.

    The click action behind a transcript's ``[n]`` link -- factored out of
    :func:`draw_expanded` so a headless test can drive it directly, the same
    reason :func:`submit_chat`/:func:`submit_build` are functions a button's
    ``if`` just calls rather than inline imgui-handler bodies.
    """
    from ..manual import render as manual_render

    manual_render.open_at(ctx, (citation.chapter, citation.anchor))


def _draw_plan_card(ctx: Any, ui: FamiliarUIState) -> None:
    """The pending plan: species, theme, movements, directions, an estimate,
    and whatever the prompt named that the plan could not use -- then
    Create/Open in Create/Discard. ``ui.message`` (a refused Create press)
    is drawn by the caller, above the transcript, the same way every other
    refusal already is -- this function only draws the plan's own summary
    and its three buttons.
    """
    from imgui_bundle import imgui

    from .. import controls, widgets

    action = ui.plan or {}
    summary = action.get("summary") or {}

    imgui.text_wrapped(str(summary.get("species") or ""))
    if summary.get("theme"):
        widgets.secondary(f"Theme: {summary['theme']}")
    movements = summary.get("movements") or []
    if movements:
        widgets.secondary("Movements: " + ", ".join(movements))
    if summary.get("directions"):
        widgets.secondary(f"Directions: {summary['directions']}")
    widgets.secondary(
        f"~{summary.get('estimate_minutes', 0):.1f} min, {summary.get('cells', 0)} cells"
    )
    for item in summary.get("ignored") or ():
        widgets.secondary(f"(not used: {item.get('text', '')})")

    busy = ctx.busy(CHARACTER_KEY)
    if controls.small_button("Create##familiar/character-create", enabled=not busy):
        submit_character(ctx)
    imgui.same_line()
    if controls.small_button("Open in Create##familiar/character-open", enabled=not busy):
        open_character_in_create(ctx)
    imgui.same_line()
    if controls.small_button("Discard##familiar/character-discard", enabled=not busy):
        discard_character_plan(ctx)


def draw_expanded(ctx: Any) -> None:
    """The pane's body once expanded: the thread transcript, the input line,
    Send, and -- in Clay, with a tab open -- Build, plus Apply/Discard once a
    preview is ready. Drawn by ``bottom_pane.draw`` inside its own child
    region; this function assumes it is already inside one.
    """
    from imgui_bundle import imgui

    from .. import controls, theme, tokens

    ui = ensure(ctx)
    key = thread_key(ctx)
    turns = ctx.familiar_threads.get(key) if getattr(ctx, "familiar_threads", None) else ()

    # The transcript takes whatever the pane's own drag (bottom_pane.draw)
    # leaves after a rough estimate of what still has to draw below it --
    # the message line, and either the input row, the preview buttons or the
    # taller plan card. Not exact (the plan card's real height depends on how
    # much of its summary is present), but the alternative -- a hardcoded
    # transcript height -- is the defect this pane shipped with: dragging the
    # handle above it would grow the pane while the transcript itself stayed
    # a fixed 60px, all of the new room going to blank space beneath it.
    footer = imgui.get_frame_height_with_spacing()
    if ui.message:
        footer += imgui.get_text_line_height_with_spacing()
    if ui.plan is not None:
        footer += imgui.get_frame_height_with_spacing() * 4.0
    if ui.preview_calls is not None:
        footer += imgui.get_frame_height_with_spacing()
    transcript_h = max(tokens.sp(40.0), imgui.get_content_region_avail().y - footer)

    # A turn already at the bottom stays pinned to it as new ones arrive; one
    # scrolled up to reread history is left alone. Read *before* this frame's
    # content is drawn, against last frame's scroll range, which is the
    # standard chat-log idiom -- there is no other point at which "was the
    # user already at the bottom" can be asked.
    was_at_bottom = imgui.get_scroll_y() >= imgui.get_scroll_max_y() - 1.0

    pad = tokens.sp(tokens.SP_2)
    imgui.push_style_var(imgui.StyleVar_.window_padding.value, (pad, pad))
    imgui.push_style_var(imgui.StyleVar_.item_spacing.value, (pad, pad * 0.5))
    imgui.begin_child("##familiar-transcript", (0, transcript_h), imgui.ChildFlags_.borders.value)
    wrap_width = imgui.get_content_region_avail().x
    for turn_idx, turn in enumerate(turns[-20:]):
        prefix = "You: " if turn.role == "user" else "Familiar: "
        text = prefix + turn.text
        bubble = theme.BUBBLE_USER if turn.role == "user" else theme.BUBBLE_ASSISTANT
        size = imgui.calc_text_size(text, None, False, wrap_width)
        origin = imgui.get_cursor_screen_pos()
        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect_filled(
            (origin.x - pad * 0.5, origin.y - pad * 0.25),
            (origin.x + size.x + pad * 0.5, origin.y + size.y + pad * 0.25),
            imgui.get_color_u32(imgui.ImVec4(*theme.rgba(bubble))),
            rounding=pad * 0.5,
        )
        imgui.text_wrapped(text)
        # A Manual answer's own [n] markers, each a small link back to the
        # section it came from -- ``cited`` already guarantees every one of
        # these actually appeared in the reply, so there is no dead link to
        # guard against here, only ids: turn_idx keeps two different turns'
        # citation buttons from colliding once imgui hashes the label.
        for cite_idx, citation in enumerate(turn.citations):
            if cite_idx:
                imgui.same_line()
            label = f"[{citation.n}] {citation.title_path}##familiar-cite-{turn_idx}-{cite_idx}"
            if controls.small_button(label):
                follow_citation(ctx, citation)
    if was_at_bottom:
        imgui.set_scroll_here_y(1.0)
    imgui.end_child()
    imgui.pop_style_var(2)

    if ui.message:
        imgui.text_colored(imgui.ImVec4(*theme.rgba(theme.WARN)), ui.message)

    pending = ui.preview_calls is not None
    if pending:
        # Disabled while Familiar is thinking: the reply is a refinement of
        # this ghost, and applying or discarding it underneath would only make
        # that reply land as "preview again".
        if controls.small_button("Apply##familiar/apply", enabled=not ui.thinking):
            apply_preview(ctx)
        imgui.same_line()
        if controls.small_button("Discard##familiar/discard", enabled=not ui.thinking):
            discard_preview(ctx)

    if ui.plan is not None:
        # T7: a Clay preview (checked above) always wins the pane's one row
        # of action buttons -- a build and a character plan cannot land in
        # the same turn today, but if one ever did, the ghost already sitting
        # in the viewport is the more urgent thing to resolve.
        _draw_plan_card(ctx, ui)
        return

    imgui.set_next_item_width(-1.0)
    hint = "Refine the preview..." if pending else "Ask Familiar..."
    _changed, ui.input_text = controls.input_text_with_hint(
        "##familiar-input", hint, ui.input_text
    )
    # Not ``changed and Enter``: pressing Enter does not change the text, so
    # that pairing never fired. Enter deactivates a single-line field on the
    # frame it is pressed, which is the frame to read the key on.
    enter_pressed = imgui.is_item_deactivated() and imgui.is_key_pressed(imgui.Key.enter)

    is_clay = str(getattr(ctx.state, "mode", "")) == "clay" and bool(_active_tab_uid(ctx))
    busy = bool(ui.thinking)
    if is_clay:
        # On the testing pin (empty card_shas) a Build press still submits --
        # ``service.familiar.clay_build`` is what refuses with reason "card",
        # naming familiar_v1.0, and that refusal is what the pane then shows.
        # Refusing here too, before the submit, would just move the same
        # sentence one frame earlier for no gain and a second place to keep
        # it in sync with ``FAMILIAR_V1_NAME``.
        # Clear the line only once a submit is accepted: a refused one (blank,
        # busy, no tab) must leave what the user typed where it was.
        if controls.small_button("Build##familiar/build", enabled=not busy) and submit_build(
            ctx, ui.input_text
        ):
            ui.input_text = ""
        imgui.same_line()
    send = controls.small_button("Send##familiar/send", enabled=not busy)
    if (send or (enter_pressed and not busy)) and submit_chat(ctx, ui.input_text):
        ui.input_text = ""
    if busy:
        from .. import widgets

        imgui.same_line()
        # A sentence the user has to read, so widgets.secondary rather than
        # imgui's disabled role, which fails contrast in both themes (UX-18).
        widgets.secondary("Thinking...")
