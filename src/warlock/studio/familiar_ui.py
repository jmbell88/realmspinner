"""Familiar's in-session conversation: state, submission and the bottom
pane's expanded body. T5 of the Familiar programme.

**Why this lives at studio level, not inside ``studio/familiar/``.** Exactly
``studio/familiar_preview.py``'s own reason (see that module's docstring):
this reaches ``agent_clay`` (to read the live scene for a Build request) and
``clay_view``/``clay_mode`` (to show and apply a ghost preview), all of which
``studio/familiar/`` is pinned never to import, even lazily.

**State lives on the frame thread, submission does not block it.** Every
network round trip goes through ``ctx.submit`` under one of two keys
(:data:`CHAT_KEY`, :data:`BUILD_KEY`) running a service-layer call
(``service.familiar.chat_reply``/``clay_build``) on a ``TaskRunner`` worker
thread; the frame thread only ever reads :class:`FamiliarUIState` and calls
:func:`on_task_done` when a result lands, the same shape every other mode's
``ctx.submit``/``on_task_done`` pair already uses (``clay_mode.py``'s module
docstring states the rule this module follows).

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

#: The bottom pane's expanded height, in design pixels, before
#: ``bottom_pane.max_height``'s own window-relative clamp -- room for a short
#: scrollable transcript plus one input row. Not derived from content height:
#: a pane that resizes itself every frame to fit a variable-length transcript
#: would fight the layout it sits in rather than the user's own toggle.
EXPANDED_H = 160.0


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
    #: ``ctx.busy`` twice for two different-shaped questions.
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
    #: All four are set together (:func:`_run_build_preview`) and cleared
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
    #: preview again" spirit ``_run_build_preview`` keeps, one plan at a time.
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
    from .familiar import threads

    if getattr(ctx, "familiar_threads", None) is None:
        ctx.familiar_threads = threads.Threads()
    from . import docmodes

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
    from .familiar import threads

    mode = str(getattr(ctx.state, "mode", ""))
    return threads.Threads.key_for(mode, _active_tab_uid(ctx))


# --- submission ----------------------------------------------------------


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
    from . import agent_clay
    from .familiar import contract

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

    Read through ``settings_character.options`` -- Create's own frame-thread
    cache, keyed on the palette directory's stamp
    (``panes/settings_character.py``'s own docstring) -- rather than calling
    ``service.characters.character_options`` fresh: it is the one place
    already paying for this read every frame Create's own form is open, and
    a second, uncached copy here would answer the same registries a frame
    later for no reason. Computed on the frame thread, alongside
    *destinations*/*asset_types* (T8's own precedent) rather than inside the
    worker closure below: unlike those two this touches no ``ctx.state``
    gate, but it does touch ``ctx.state.preview``'s own cache slot, which is
    frame-thread state exactly like the Clay scene capture is.
    """
    from .. import rigging
    from .panes import settings_character

    raw = settings_character.options(ctx)
    families = [
        {"key": f["key"], "label": f["label"], "themes": [t["key"] for t in f["themes"]]}
        for f in raw["families"]
    ]
    templates = {a["template"] for a in raw["archetypes"]}
    movements = sorted({name for t in templates for name in rigging.shipped_clip_names(t)})
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
    from .familiar import threads

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
    from . import create_assets, familiar_doors

    destinations = familiar_doors.destinations(ctx)
    asset_types = create_assets.ASSET_TYPE_OPTIONS
    # T7: same frame-thread treatment as the two lines above -- see
    # ``_character_options``'s own docstring for why this one is cheap
    # rather than studio-gated.
    character_options = _character_options(ctx)

    from ..service import familiar as svc_familiar

    def run() -> Any:
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

    tag = {"thread_key": key, "tab_uid": tab_uid, "scene_captured": scene is not None}
    if not ctx.submit(CHAT_KEY, run, tag=tag):
        return False
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

    from .familiar import threads

    key = thread_key(ctx)
    ctx.familiar_threads.append(key, threads.Turn("user", prompt))

    from ..service import familiar as svc_familiar

    def run() -> list[dict]:
        return svc_familiar.clay_build(ctx.svc, prompt, scene)

    tag = {"thread_key": key, "tab_uid": tab_uid}
    if not ctx.submit(BUILD_KEY, run, tag=tag):
        return False
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


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from the app for :data:`CHAT_KEY`/:data:`BUILD_KEY`/
    :data:`CHARACTER_KEY`, the same way every other mode's ``on_task_done``
    is called from ``main._on_task_done`` (any ``"familiar/"``-prefixed key
    reaches here)."""
    ui = ensure(ctx)
    tag = done.tag if isinstance(done.tag, dict) else {}

    if done.key == CHAT_KEY:
        ui.thinking = ""
        if done.ok:
            ui.reason = None
            ui.message = None
            result = done.result
            from ..service.familiar import Answer

            if isinstance(result, Answer) and result.calls is not None:
                # The router sent this one to Clay -- land it exactly like
                # an explicit Build's own result, calls and all.
                _run_build_preview(ctx, ui, tag.get("tab_uid", ""), result.calls)
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
                from .familiar import threads

                threads_obj.append(thread, threads.Turn("familiar", text, citations))
        else:
            ui.reason, ui.message = _reason_and_message(done)
        return

    if done.key == BUILD_KEY:
        ui.thinking = ""
        if not done.ok:
            ui.reason, ui.message = _reason_and_message(done)
            _clear_preview(ui)
            return
        calls = done.result if isinstance(done.result, list) else []
        ui.reason = None
        ui.message = None
        _run_build_preview(ctx, ui, tag.get("tab_uid", ""), calls)
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
        thread = tag.get("thread_key")
        threads_obj = getattr(ctx, "familiar_threads", None)
        text = "Character queued -- it will appear in the Library."
        if thread is not None and threads_obj is not None:
            from .familiar import threads

            threads_obj.append(thread, threads.Turn("familiar", text))
        toast = getattr(ctx, "toast", None)
        if toast is not None:
            toast(text)
        return


def _run_door(ctx: Any, action: dict[str, Any]) -> str:
    """T8: act out a routed ``navigate``/``create`` decision. -> the
    sentence the transcript should show.

    A shape neither of :func:`familiar_doors.navigate`/:func:`draft_in_create`
    itself refuses (an ``action`` this build does not recognise -- there is
    none today, but a future skill's own action kind must not crash the
    frame loop reading a reply that landed) answers plainly rather than
    raising."""
    from . import familiar_doors

    kind = action.get("kind")
    if kind == "navigate":
        return familiar_doors.navigate(ctx, str(action.get("target") or ""))
    if kind == "draft":
        return familiar_doors.draft_in_create(
            ctx, str(action.get("asset_type") or ""), str(action.get("prompt") or "")
        )
    return "I'm not sure what to do with that."


def _clear_preview(ui: FamiliarUIState) -> None:
    ui.preview_calls = None
    ui.preview_scratch = None
    ui.preview_diff = None
    ui.preview_tab_uid = ""


def _run_build_preview(ctx: Any, ui: FamiliarUIState, tab_uid: str, calls: list[dict]) -> None:
    """Run *calls* against a scratch clone of *tab_uid*'s document, one at a
    time, stopping at the first refusal (:func:`~.familiar_preview.
    run_scratch`'s own contract) -- and show the ghost the moment the whole
    list has run clean."""
    from . import clay_mode, familiar_preview
    from .clay import scratch as clay_scratch

    state = clay_mode.ensure(ctx)
    tab = state.get(tab_uid) if tab_uid else None
    if tab is None:
        # The tab this build was requested against closed while Familiar was
        # thinking -- the same "preview again" sentence every other
        # familiar_preview refusal uses, for the same reason: there is
        # nothing left to preview against.
        ui.message = "the document changed -- preview again"
        ui.reason = None
        return

    scratch_ctx = familiar_preview.build(tab.doc)
    for entry in calls:
        name = entry.get("name") if isinstance(entry, dict) else None
        args = entry.get("arguments") if isinstance(entry, dict) else {}
        result = familiar_preview.run_scratch(scratch_ctx, name, args or {})
        if result.get("isError"):
            content = result.get("content") or []
            text = (
                content[0].get("text")
                if content and isinstance(content[0], dict)
                else "Familiar's build could not be previewed."
            )
            ui.message = text
            ui.reason = "parse"
            return

    scratch_doc = scratch_ctx.state.clay.get(scratch_ctx.tab_uid).doc
    diff = clay_scratch.diff(tab.doc, scratch_doc)
    ui.preview_calls = calls
    ui.preview_scratch = scratch_doc
    ui.preview_diff = diff
    ui.preview_tab_uid = tab_uid
    view = getattr(ctx, "clay_view", None)
    if view is not None:
        view.set_preview(diff, scratch_doc)


def apply_preview(ctx: Any) -> None:
    """Apply the pending preview, if any -- see ``familiar_preview.apply``'s
    own docstring for the refusals it can still return (the document moved
    since the preview was computed)."""
    ui = ensure(ctx)
    if ui.preview_calls is None:
        return
    from . import familiar_preview

    result = familiar_preview.apply(ctx, ui.preview_tab_uid, ui.preview_diff, ui.preview_scratch)
    if not result.get("ok"):
        ui.message = result.get("message")
        ui.reason = None
        return
    ui.message = None
    ui.reason = None
    _clear_preview(ui)


def discard_preview(ctx: Any) -> None:
    """Drop the pending preview, if any, leaving the document untouched."""
    ui = ensure(ctx)
    if ui.preview_calls is None:
        return
    from . import familiar_preview

    familiar_preview.discard(ctx, ui.preview_tab_uid)
    _clear_preview(ui)


# --- T7: the character plan card ------------------------------------------


def _character_fields(plan: dict[str, Any]) -> dict[str, Any]:
    """*plan* (:func:`~.familiar.character_plan.parse_plan`'s own shape) ->
    the Create form's own ``character_*`` field names -- the same subset
    ``troupe_mode.vary_in_create`` writes for a recipe it is varying, built
    only from whichever of *plan*'s fields are actually present (never
    invents a theme, a camera or a name the plan itself does not carry).

    ``movements`` is filtered against ``settings_character.MOVEMENTS`` (the
    closed default trio Create's own action checkboxes offer), the same
    filter ``vary_in_create`` applies to a recipe's ``animations`` keys --
    Create's form has no control for a movement outside that ladder, so one
    from the plan's own wider vocabulary (any shipped clip, not just the
    default three) is silently absent from the brief rather than fed to a
    checkbox that cannot show it; the plan itself still built the full list
    into *its own* ``overrides``, for the Create button's own path.
    """
    from .panes import settings_character

    fields: dict[str, Any] = {}
    if "family" in plan:
        fields["character_family"] = plan["family"]
    if "theme" in plan:
        fields["character_theme"] = plan["theme"]
    if "movements" in plan:
        wanted = set(plan["movements"])
        fields["character_actions"] = ",".join(
            name for name, _frames in settings_character.MOVEMENTS if name in wanted
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

    from ..service import familiar as svc_familiar

    def run() -> Any:
        return svc_familiar.create_planned_character(
            ctx.svc, action["prompt"], action["overrides"], plan.get("name")
        )

    tag = {"thread_key": thread_key(ctx)}
    if not ctx.submit(CHARACTER_KEY, run, tag=tag):
        return False
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
    from . import familiar_doors

    text = familiar_doors.draft_in_create(
        ctx, "character", action["prompt"], character_fields=_character_fields(action["plan"])
    )
    ui.plan = None
    threads_obj = getattr(ctx, "familiar_threads", None)
    if threads_obj is not None:
        from .familiar import threads

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
    from .manual import render as manual_render

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

    from . import controls, widgets

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

    from . import controls

    ui = ensure(ctx)
    key = thread_key(ctx)
    turns = ctx.familiar_threads.get(key) if getattr(ctx, "familiar_threads", None) else ()

    imgui.begin_child("##familiar-transcript", (0, 60), imgui.ChildFlags_.borders.value)
    for turn_idx, turn in enumerate(turns[-20:]):
        prefix = "You: " if turn.role == "user" else "Familiar: "
        imgui.text_wrapped(prefix + turn.text)
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
    imgui.end_child()

    if ui.message:
        from . import theme

        imgui.text_colored(imgui.ImVec4(*theme.rgba(theme.WARN)), ui.message)

    if ui.preview_calls is not None:
        if controls.small_button("Apply##familiar/apply"):
            apply_preview(ctx)
        imgui.same_line()
        if controls.small_button("Discard##familiar/discard"):
            discard_preview(ctx)
        return

    if ui.plan is not None:
        # T7: a Clay preview (checked above) always wins the pane's one row
        # of action buttons -- a build and a character plan cannot land in
        # the same turn today, but if one ever did, the ghost already sitting
        # in the viewport is the more urgent thing to resolve.
        _draw_plan_card(ctx, ui)
        return

    imgui.set_next_item_width(-1.0)
    _changed, ui.input_text = controls.input_text_with_hint(
        "##familiar-input", "Ask Familiar...", ui.input_text
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
        from . import widgets

        imgui.same_line()
        # A sentence the user has to read, so widgets.secondary rather than
        # imgui's disabled role, which fails contrast in both themes (UX-18).
        widgets.secondary("Thinking...")
