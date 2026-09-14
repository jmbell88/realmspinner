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


def submit_chat(ctx: Any, prompt: str) -> bool:
    """Send *prompt* as a plain chat message. -> whether it was accepted.

    Refused (returns ``False``, no toast -- the disabled Send button already
    said why) when *prompt* is blank or a chat is already in flight. Appends
    the user's own turn to the thread immediately, before the network call
    even starts, so it appears in the transcript the same frame it was sent
    rather than only once a reply lands.
    """
    prompt = prompt.strip()
    if not prompt or ctx.busy(CHAT_KEY):
        return False
    from .familiar import threads

    key = thread_key(ctx)
    ctx.familiar_threads.append(key, threads.Turn("user", prompt))
    history = ctx.familiar_threads.get(key)[:-1]

    from ..service import familiar as svc_familiar

    def run() -> str:
        return svc_familiar.chat_reply(ctx.svc, prompt, history)

    if not ctx.submit(CHAT_KEY, run, tag={"thread_key": key}):
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
    directly. The scene is read and compacted here, on the frame thread,
    before the submit -- ``agent_clay.call`` walks the live ``ClayDoc``, and
    that read must happen against *this* frame's document, not whatever it
    is by the time a worker thread gets around to it.
    """
    prompt = prompt.strip()
    tab_uid = _active_tab_uid(ctx)
    if not prompt or not tab_uid or ctx.busy(BUILD_KEY):
        return False
    from . import agent_clay
    from .familiar import contract, threads

    session = agent_clay.Session(tab_uid=tab_uid)
    scene_result = agent_clay.call(ctx, session, "clay_scene", {})
    structured = (
        scene_result.get("structuredContent") if isinstance(scene_result, dict) else None
    )
    scene = contract.compact_scene(structured or {})

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
    """Called from the app for :data:`CHAT_KEY`/:data:`BUILD_KEY`, the same
    way every other mode's ``on_task_done`` is called from ``main.
    _on_task_done``."""
    ui = ensure(ctx)
    tag = done.tag if isinstance(done.tag, dict) else {}

    if done.key == CHAT_KEY:
        ui.thinking = ""
        if done.ok:
            ui.reason = None
            ui.message = None
            reply = done.result if isinstance(done.result, str) else str(done.result)
            thread = tag.get("thread_key")
            threads_obj = getattr(ctx, "familiar_threads", None)
            if thread is not None and threads_obj is not None:
                from .familiar import threads

                threads_obj.append(thread, threads.Turn("familiar", reply))
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


# --- drawing -----------------------------------------------------------


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
    for turn in turns[-20:]:
        prefix = "You: " if turn.role == "user" else "Familiar: "
        imgui.text_wrapped(prefix + turn.text)
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
