"""T5's bottom-pane conversation logic: ``studio/familiar_ui.py``.

Deliberately headless -- ``draw_expanded`` itself needs a live imgui frame
and is not exercised here (that is what ``/exercise-mode`` and the
screenshot debt ``TODO.md``'s P56 owns are for); what is under test is the
state machine ``draw_expanded`` calls into: submitting through the task
runner rather than blocking, landing a refusal or a Clay preview, and the
one app-level wire ``install`` adds to ``docmodes.TAB_CLOSED``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from warlock import models
from warlock.service import familiar as svc_familiar
from warlock.studio import clay_mode, docmodes, familiar_ui
from warlock.studio.clay import document as bd
from warlock.studio.clay import primitives as bp
from warlock.studio.familiar import retrieval
from warlock.studio.state import ManualState
from warlock.studio.tasks import Done


class _FakeThreads:
    """Just enough of ``familiar.threads.Threads`` for these tests."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], list] = {}

    def append(self, key, turn) -> None:
        self._data.setdefault(key, []).append(turn)

    def get(self, key):
        return tuple(self._data.get(key, ()))

    def drop(self, mode: str, uid: str) -> None:
        self._data.pop((mode, uid), None)


class _FakeCtx:
    """Enough of ``app_ctx.Ctx`` to drive ``familiar_ui`` without a window --
    the same shape ``tests/familiar/test_apply.py``'s own ``_FakeCtx`` uses
    for ``familiar_preview``, extended with the task-runner shorthands
    (``submit``/``busy``) and a thread store."""

    def __init__(self, doc: bd.ClayDoc | None = None, *, mode: str = "clay") -> None:
        doc = doc if doc is not None else bd.ClayDoc()
        tab = clay_mode.ClayTab(doc=doc)
        clay_state = clay_mode.ClayState(docs=[tab], active_uid=tab.uid)
        self.state = SimpleNamespace(
            clay=clay_state, mode=mode, familiar=None, preview={}, manual=ManualState()
        )
        self.settings = SimpleNamespace()
        # T7: ``config.palette_dir`` is all ``_character_options`` needs off
        # ``svc`` -- ``settings_character.options``/``service.characters.
        # character_options`` read every other registry in memory, and
        # ``palettes.available``/``stamps.stamp_ns`` both already answer "no
        # palettes" for a directory that is not there rather than raising.
        self.svc = SimpleNamespace(
            worker=SimpleNamespace(familiar=object()),
            config=SimpleNamespace(palette_dir=Path("___familiar_ui_test_no_palette_dir___")),
        )
        self.toasts: list[str] = []
        self.familiar_threads = _FakeThreads()
        self._tab = tab
        self._pending: dict[str, tuple] = {}
        self.clay_view = SimpleNamespace(cleared=0, previewed=None, grabbing=False)
        self.clay_view.clear_preview = lambda: setattr(
            self.clay_view, "cleared", self.clay_view.cleared + 1
        )
        self.clay_view.set_preview = lambda diff, scratch: setattr(
            self.clay_view, "previewed", (diff, scratch)
        )

    @property
    def tab(self):
        return self._tab

    def submit(self, key, fn, *args, tag=None, **kwargs) -> bool:
        if key in self._pending:
            return False
        self._pending[key] = (fn, args, kwargs, tag)
        return True

    def busy(self, key) -> bool:
        return key in self._pending

    def finish(self, key) -> None:
        self._pending.pop(key, None)

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append(message)


# --- submitting never blocks the frame --------------------------------------


def test_sending_submits_to_the_task_runner_rather_than_blocking_the_frame(monkeypatch):
    """``submit_chat`` must hand the network call to ``ctx.submit`` as a
    closure, never call it inline -- proven here by making the service door
    explode if it *is* called inline, then only calling the captured closure
    afterwards.

    T6: Send now routes through ``service.familiar.ask`` rather than
    ``chat_reply`` directly (the router decides which skill answers), so
    this patches ``ask`` -- patching the now-bypassed ``chat_reply`` here
    would prove nothing about what ``submit_chat`` actually calls."""

    def exploding_ask(*_args, **_kwargs):
        raise AssertionError("ask must not run on the frame thread")

    monkeypatch.setattr(svc_familiar, "ask", exploding_ask)
    ctx = _FakeCtx(mode="home")

    accepted = familiar_ui.submit_chat(ctx, "hello")

    assert accepted is True
    assert familiar_ui.CHAT_KEY in ctx._pending
    fn, _args, _kwargs, _tag = ctx._pending[familiar_ui.CHAT_KEY]
    with pytest.raises(AssertionError):
        fn()  # only now -- simulating the worker thread -- does it run


def test_a_blank_prompt_or_a_busy_key_is_not_submitted():
    ctx = _FakeCtx(mode="home")
    assert familiar_ui.submit_chat(ctx, "   ") is False
    assert ctx._pending == {}

    ctx.submit(familiar_ui.CHAT_KEY, lambda: None)
    assert familiar_ui.submit_chat(ctx, "hello") is False


# --- refusals land on the pane state ----------------------------------------


def test_refused_state_shows_the_lease_sentence():
    """``LlamaServer.ensure_started``'s lease sentence must reach
    ``FamiliarUIState.message`` verbatim -- the one refusal the T5 brief says
    keeps its exact wording."""
    ctx = _FakeCtx(mode="home")
    sentence = (
        "Familiar cannot start while a GPU job holds the card -- "
        "it will restart on your next message."
    )
    error = svc_familiar.FamiliarRefusal(sentence, reason="lease")
    done = Done(
        key=familiar_ui.CHAT_KEY,
        error=error,
        message=sentence,
        tag={"thread_key": ("studio", "")},
    )

    familiar_ui.on_task_done(ctx, done)

    ui = familiar_ui.ensure(ctx)
    assert ui.reason == "lease"
    assert ui.message == sentence
    assert ui.thinking == ""


def test_build_in_clay_on_the_testing_pin_names_familiar_v1():
    """A real ``clay_build`` refusal on the testing pin (empty ``card_shas``)
    must land as reason ``"card"`` naming ``familiar_v1.0`` -- run through the
    actual service door, not a hand-typed sentence, so this cannot drift from
    what ``clay_build`` really raises."""
    assert models.FAMILIAR_MODELS["familiar_gguf"].card_shas == ()
    fake_svc = SimpleNamespace(worker=SimpleNamespace(familiar=object()))
    with pytest.raises(svc_familiar.FamiliarRefusal) as excinfo:
        svc_familiar.clay_build(fake_svc, "build a box", {"objects": []})
    error = excinfo.value

    ctx = _FakeCtx(mode="clay")
    done = Done(
        key=familiar_ui.BUILD_KEY,
        error=error,
        message=error.message,
        tag={"thread_key": ("clay", ctx.tab.uid), "tab_uid": ctx.tab.uid},
    )

    familiar_ui.on_task_done(ctx, done)

    ui = familiar_ui.ensure(ctx)
    assert ui.reason == "card"
    assert models.FAMILIAR_V1_NAME in ui.message


# --- the closed-tab listener is the app's own wiring ------------------------


def test_closing_a_tab_drops_its_thread_through_the_registered_listener():
    """``familiar_ui.install`` -- the function the App itself calls at
    startup -- must be what puts ``threads.drop`` on ``docmodes.TAB_CLOSED``;
    this drives the drop through that registered listener rather than
    appending one of its own."""
    ctx = _FakeCtx(mode="clay")
    before = list(docmodes.TAB_CLOSED)
    try:
        familiar_ui.install(ctx)
        assert ctx.familiar_threads.drop in docmodes.TAB_CLOSED

        from warlock.studio.familiar.threads import Turn

        ctx.familiar_threads.append(("clay", "tab-1"), Turn("user", "hi"))
        assert ctx.familiar_threads.get(("clay", "tab-1"))

        for listener in list(docmodes.TAB_CLOSED):
            listener("clay", "tab-1")

        assert ctx.familiar_threads.get(("clay", "tab-1")) == ()

        # Calling install again on the same ctx must not double-register.
        familiar_ui.install(ctx)
        assert docmodes.TAB_CLOSED.count(ctx.familiar_threads.drop) == 1
    finally:
        docmodes.TAB_CLOSED[:] = before


# --- a canned Clay build previews as a ghost, then applies or refuses -------


def _canned_calls() -> list[dict]:
    return [{"name": "clay_add_primitive", "arguments": {"generator": "box", "name": "crate"}}]


def test_a_canned_build_previews_as_a_ghost_and_apply_lands_it():
    doc = bd.ClayDoc()
    ctx = _FakeCtx(doc, mode="clay")
    calls = _canned_calls()
    done = Done(
        key=familiar_ui.BUILD_KEY,
        result=calls,
        tag={"thread_key": ("clay", ctx.tab.uid), "tab_uid": ctx.tab.uid},
    )

    familiar_ui.on_task_done(ctx, done)

    ui = familiar_ui.ensure(ctx)
    assert ui.preview_calls == calls
    assert ctx.clay_view.previewed is not None  # the ghost was set

    familiar_ui.apply_preview(ctx)

    assert any(o.name == "crate" for o in doc.objects)
    assert familiar_ui.ensure(ctx).preview_calls is None
    assert ctx.clay_view.cleared == 1


def test_apply_after_the_document_changed_returns_preview_again():
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="existing", mesh=bp.box()))
    ctx = _FakeCtx(doc, mode="clay")
    calls = _canned_calls()
    done = Done(
        key=familiar_ui.BUILD_KEY,
        result=calls,
        tag={"thread_key": ("clay", ctx.tab.uid), "tab_uid": ctx.tab.uid},
    )
    familiar_ui.on_task_done(ctx, done)
    assert familiar_ui.ensure(ctx).preview_calls is not None

    # The user edits the real document in between -- the base head moves.
    doc.set_transform(doc.objects[0].uid, translation=[1.0, 0.0, 0.0])

    familiar_ui.apply_preview(ctx)

    ui = familiar_ui.ensure(ctx)
    assert ui.message is not None and "preview again" in ui.message
    # Not silently discarded: the calls list stays offered, matching every
    # other familiar_preview refusal's own "fix is: preview again" contract.
    assert ui.preview_calls is not None


def test_discard_clears_the_ghost_without_touching_the_document():
    doc = bd.ClayDoc()
    ctx = _FakeCtx(doc, mode="clay")
    calls = _canned_calls()
    done = Done(
        key=familiar_ui.BUILD_KEY,
        result=calls,
        tag={"thread_key": ("clay", ctx.tab.uid), "tab_uid": ctx.tab.uid},
    )
    familiar_ui.on_task_done(ctx, done)

    familiar_ui.discard_preview(ctx)

    assert doc.objects == []
    assert familiar_ui.ensure(ctx).preview_calls is None
    assert ctx.clay_view.cleared == 1


# --- T6: Send routes through the router --------------------------------


def _fake_citation() -> retrieval.Citation:
    return retrieval.Citation(
        n=1, chapter="12-plotter", anchor="exporting",
        title_path="12 Plotter > Exporting", text="Export via File > Export > Tiled.",
    )


def test_send_routes_through_ask_and_a_cited_answer_lands_with_its_citations(monkeypatch):
    """A Manual answer routed by ``ask`` must land in the thread with its own
    citations attached -- not just its text -- so the pane can draw the
    ``[n]`` links :func:`familiar_ui.follow_citation` opens."""
    citation = _fake_citation()
    answer = svc_familiar.Answer(
        skill="manual", text="Use File > Export > Tiled [1].", citations=(citation,)
    )
    monkeypatch.setattr(svc_familiar, "ask", lambda *a, **k: answer)
    ctx = _FakeCtx(mode="home")

    accepted = familiar_ui.submit_chat(ctx, "how do I export a map?")

    assert accepted is True
    fn, _args, _kwargs, tag = ctx._pending[familiar_ui.CHAT_KEY]
    result = fn()  # simulating the worker thread running the submitted closure
    done = Done(key=familiar_ui.CHAT_KEY, result=result, tag=tag)

    familiar_ui.on_task_done(ctx, done)

    turns = ctx.familiar_threads.get(tag["thread_key"])
    assert turns[-1].role == "familiar"
    assert turns[-1].text == answer.text
    assert turns[-1].citations == (citation,)


def test_a_routed_build_in_clay_lands_as_a_ghost_preview():
    """A router decision of ``clay_build``, reached through Send rather than
    the explicit Build button, must land exactly the same way an explicit
    Build's result does: a ghost preview, not a plain chat line."""
    doc = bd.ClayDoc()
    ctx = _FakeCtx(doc, mode="clay")
    calls = _canned_calls()
    answer = svc_familiar.Answer(skill="clay_build", text=None, calls=calls)
    done = Done(
        key=familiar_ui.CHAT_KEY,
        result=answer,
        tag={"thread_key": ("clay", ctx.tab.uid), "tab_uid": ctx.tab.uid},
    )

    familiar_ui.on_task_done(ctx, done)

    ui = familiar_ui.ensure(ctx)
    assert ui.preview_calls == calls
    assert ctx.clay_view.previewed is not None


# --- T8: navigate/create routes land as an acted-out door -------------


def test_a_navigate_answer_moves_the_app_and_says_so_in_the_thread():
    """An ``Answer.action`` of kind "navigate" must be acted out on the
    frame thread by ``on_task_done`` (never on the worker thread that
    produced it) -- the mode actually switches, and the sentence
    ``familiar_doors.navigate`` returns lands in the thread, not the
    (``None``) ``Answer.text`` it arrived with."""
    ctx = _FakeCtx(mode="home")
    answer = svc_familiar.Answer(
        skill="navigate", text=None, action={"kind": "navigate", "target": "go:clay"}
    )
    done = Done(key=familiar_ui.CHAT_KEY, result=answer, tag={"thread_key": ("home", "")})

    familiar_ui.on_task_done(ctx, done)

    assert ctx.state.mode == "clay"
    turns = ctx.familiar_threads.get(("home", ""))
    assert turns[-1].role == "familiar"
    assert "Clay" in turns[-1].text


def test_a_draft_answer_opens_create_with_the_brief_filled():
    """An ``Answer.action`` of kind "draft" must fill Create's own form and
    move the stage, never submit -- landed the same way a navigate action
    is, through ``on_task_done`` on the frame thread."""
    ctx = _FakeCtx(mode="home")
    ctx.state.form_2d = {}
    answer = svc_familiar.Answer(
        skill="create",
        text=None,
        action={"kind": "draft", "asset_type": "image", "prompt": "a lantern"},
    )
    done = Done(key=familiar_ui.CHAT_KEY, result=answer, tag={"thread_key": ("home", "")})

    familiar_ui.on_task_done(ctx, done)

    assert ctx.state.mode == "create"
    assert ctx.state.form_2d["asset_type"] == "image"
    assert ctx.state.form_2d["prompt"] == "a lantern"
    turns = ctx.familiar_threads.get(("home", ""))
    assert turns[-1].role == "familiar"
    assert "Generate" in turns[-1].text


def test_following_a_citation_opens_the_manual_at_its_section():
    """The click action behind a transcript's ``[n]`` link must raise the
    Manual overlay at exactly that citation's chapter/section -- driven
    headlessly through :func:`familiar_ui.follow_citation`, the function a
    real button press in :func:`familiar_ui.draw_expanded` just calls."""
    ctx = _FakeCtx(mode="home")
    citation = _fake_citation()

    familiar_ui.follow_citation(ctx, citation)

    assert ctx.state.manual.open is True
    assert ctx.state.manual.chapter == citation.chapter
    assert ctx.state.manual.pending_anchor == citation.anchor


# --- T7: the character plan card -------------------------------------------


def _character_plan_action() -> dict:
    return {
        "kind": "character_plan",
        "prompt": "make me a goblin in the swamp",
        "plan": {"family": "goblin", "theme": "swamp", "movements": ["walk"]},
        "overrides": {"family": "goblin", "theme": "swamp", "animations": {"walk": None}},
        "summary": {
            "species": "Goblin",
            "theme": "swamp",
            "movements": ["walk"],
            "directions": None,
            "cells": 24,
            "estimate_minutes": 3.5,
            "ignored": [],
        },
    }


def test_a_character_plan_waits_for_a_press_before_anything_is_queued():
    """A ``character_plan`` action must only ever populate ``ui.plan`` --
    unlike navigate/draft it is never acted out by ``on_task_done`` itself,
    and unlike a Clay build it never queues anything: nothing is submitted
    until the user presses Create."""
    ctx = _FakeCtx(mode="home")
    action = _character_plan_action()
    answer = svc_familiar.Answer(skill="character", text=None, action=action)
    done = Done(key=familiar_ui.CHAT_KEY, result=answer, tag={"thread_key": ("home", "")})

    familiar_ui.on_task_done(ctx, done)

    ui = familiar_ui.ensure(ctx)
    assert ui.plan == action
    assert ctx._pending == {}, "a plan landing must never itself submit anything"
    turns = ctx.familiar_threads.get(("home", ""))
    assert turns[-1].role == "familiar"
    assert "plan" in turns[-1].text.lower()


def test_create_on_a_plan_submits_the_character_job_off_the_frame_thread(monkeypatch):
    """Pressing Create must hand ``create_planned_character`` to ``ctx.submit``
    as a closure -- proven by an exploding stand-in that only ever runs once
    the captured closure is invoked -- and must clear the pending plan the
    moment the submit is *accepted*, not once it lands."""
    ctx = _FakeCtx(mode="home")
    ui = familiar_ui.ensure(ctx)
    ui.plan = _character_plan_action()

    def exploding_create(*_args, **_kwargs):
        raise AssertionError("create_planned_character must not run on the frame thread")

    monkeypatch.setattr(svc_familiar, "create_planned_character", exploding_create)

    accepted = familiar_ui.submit_character(ctx)

    assert accepted is True
    assert ui.plan is None
    assert familiar_ui.CHARACTER_KEY in ctx._pending
    fn, _args, _kwargs, _tag = ctx._pending[familiar_ui.CHARACTER_KEY]
    with pytest.raises(AssertionError):
        fn()  # only now -- simulating the worker thread -- does it run


def test_open_in_create_drafts_the_character_brief_with_the_plan_fields(monkeypatch):
    """Open in Create must call ``familiar_doors.draft_in_create`` with the
    plan's own fields mapped onto Create's ``character_*`` form names, and
    must clear the plan -- it drafts, it never mints."""
    ctx = _FakeCtx(mode="home")
    ui = familiar_ui.ensure(ctx)
    ui.plan = _character_plan_action()

    captured: dict = {}

    def fake_draft(ctx_arg, asset_type, prompt, *, character_fields=None):
        captured["asset_type"] = asset_type
        captured["prompt"] = prompt
        captured["character_fields"] = character_fields
        return "Drafted in Create -- check the brief and press Generate."

    from warlock.studio import familiar_doors

    monkeypatch.setattr(familiar_doors, "draft_in_create", fake_draft)

    familiar_ui.open_character_in_create(ctx)

    assert ui.plan is None
    assert captured["asset_type"] == "character"
    assert captured["prompt"] == "make me a goblin in the swamp"
    assert captured["character_fields"]["character_family"] == "goblin"
    assert captured["character_fields"]["character_theme"] == "swamp"
    turns = ctx.familiar_threads.get(familiar_ui.thread_key(ctx))
    assert turns[-1].text == "Drafted in Create -- check the brief and press Generate."


def test_discard_clears_the_plan():
    """Discard must drop the pending plan -- nothing was ever queued, so
    there is nothing to undo, only the card to stop showing."""
    ctx = _FakeCtx(mode="home")
    ui = familiar_ui.ensure(ctx)
    ui.plan = _character_plan_action()

    familiar_ui.discard_character_plan(ctx)

    assert ui.plan is None
