"""Regression tests for the 2026-09-15 audit's Clay/Familiar findings.

Four unrelated defects, one file because one fixer owned all four: clay-01
(a live keyboard/gizmo drag survives a tab switch and settles against the
wrong document), clay-04 (the copy-family reader mis-reads a four-digit
suffix), agents-01 (a Familiar build preview can land as a ghost over a tab
nobody asked to preview) and agents-07 (the character-plan landing path had
no test at all, success or failure).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from test_familiar_ui import _canned_calls, _FakeCtx  # shared rather than duplicated

from warlock.service import errors as service_errors
from warlock.service import familiar as svc_familiar
from warlock.studio import clay_mode, clay_view, familiar_ui
from warlock.studio.clay import diagnose
from warlock.studio.clay import document as bd
from warlock.studio.clay import primitives as bp
from warlock.studio.tasks import Done

RECT = (0.0, 0.0, 128.0, 96.0)


# --- clay-01: a tab switch mid-drag ------------------------------------------


def test_switching_the_active_tab_via_the_tab_bar_commits_or_cancels_a_live_drag(gl) -> None:
    """``ClayState.activate`` -- what ``docmodes.tab_bar``'s click handler,
    ``cycle`` (Ctrl+Tab) and both ``clay-open`` dedupe paths all call to
    switch tabs -- used to clear only its own ``drag_axis``/``ref``
    bookkeeping. A live G/R/S drag has already written TRS onto the object in
    place by the time any of those run, and nothing recorded that: the
    drag's eventual commit or cancel (a release, an Esc) then ran against
    whichever document had *become* active, not the one the drag began on.
    ``close_tab``'s ``release`` already cancelled a live drag first for the
    tab-*closing* case; this is the tab-*switching* half.

    Drives ``ClayState.activate`` directly against a real, headless
    ``ClayView`` -- the same shape ``tests/test_clay_view.py`` already
    drives keyboard drags through -- rather than a fake, so the actual
    ``settle_drag`` wiring (``clay_mode.ensure``) and commit path
    (``_view_drag.DragOps.settle_drag``) are what is under test.

    Fails against the unfixed code with:
        AssertionError: the drag must be settled, not left live on the new tab
    (``ClayState.activate`` cleared ``drag_axis``/``ref`` and nothing else,
    so ``view._grab`` was still ``"keydrag"`` after the switch.)
    """
    doc_a = bd.ClayDoc()
    obj = bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())
    doc_a.add_object(obj)
    doc_a.select([obj.uid])
    doc_b = bd.ClayDoc()

    state = clay_mode.ClayState()
    tab_a = clay_mode.ClayTab(doc=doc_a)
    tab_b = clay_mode.ClayTab(doc=doc_b)
    state.add(tab_a)
    state.add(tab_b)
    state.active_uid = tab_a.uid

    ctx = SimpleNamespace(state=SimpleNamespace(clay=state, mode="clay"), settings=None)
    view = clay_view.ClayView(gl, ctx)
    ctx.clay_view = view
    try:
        clay_mode.ensure(ctx)  # wires ClayState.settle_drag to this view

        view.draw(doc_a, RECT, 0.0)
        view._last_mouse = (64.0, 48.0)
        assert view.begin_keyboard_drag(doc_a, "move")
        view._motion(doc_a, (100.0, 48.0))
        assert view.dragging
        head_a = doc_a.history.head
        dragged_to = np.array(obj.translation, copy=True)

        state.activate(tab_b.uid)  # exactly what the tab bar's click does

        assert state.active_uid == tab_b.uid
        assert not view.dragging, (
            "the drag must be settled, not left live on the new tab"
        )
        assert doc_a.history.head != head_a, "one history step for the whole gesture"
        assert np.allclose(obj.translation, dragged_to), (
            "the commit must land on tab A, the tab the drag began on"
        )
        assert doc_b.history.head == 0, "tab B -- the tab switched to -- must be untouched"
    finally:
        view.release()


# --- 2026-09-16 audit: add() (new/open/import/recover) settles nothing -------


def test_creating_or_opening_a_document_mid_drag_settles_the_drag_on_the_tab_it_replaces(
    gl,
) -> None:
    """``ClayState.add`` -- what ``new_document``, both ``clay-open`` adopt
    branches, ``clay-recover`` and both ``clay-import`` adopt branches all
    call to bring a *new* tab in -- had no settle at all, unlike ``activate``
    (fixed for exactly this class of bug by the 2026-09-15 audit's clay-01,
    the test above). New/Open carry no drag gate in the UI either
    (``widgets.document_header``'s buttons), and the async adopt paths are
    inherently decoupled from whatever drag is in progress when their result
    lands, so a live G/R/S drag on the tab being replaced was left with its
    TRS already written in place and no history step behind it -- Ctrl+Z can
    never revert a move nothing recorded.

    Fails against the unfixed code with:
        AssertionError: the drag must be settled, not left live on the tab it was replaced on
    (``ClayState.add`` only appended the new tab and cleared its own
    ``drag_axis``/``ref``, so ``view._grab`` was still ``"keydrag"`` after
    the add.)
    """
    doc_a = bd.ClayDoc()
    obj = bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())
    doc_a.add_object(obj)
    doc_a.select([obj.uid])

    state = clay_mode.ClayState()
    tab_a = clay_mode.ClayTab(doc=doc_a)
    state.add(tab_a)
    state.active_uid = tab_a.uid

    ctx = SimpleNamespace(state=SimpleNamespace(clay=state, mode="clay"), settings=None)
    view = clay_view.ClayView(gl, ctx)
    ctx.clay_view = view
    try:
        clay_mode.ensure(ctx)  # wires ClayState.settle_drag to this view

        view.draw(doc_a, RECT, 0.0)
        view._last_mouse = (64.0, 48.0)
        assert view.begin_keyboard_drag(doc_a, "move")
        view._motion(doc_a, (100.0, 48.0))
        assert view.dragging
        head_a = doc_a.history.head
        dragged_to = np.array(obj.translation, copy=True)

        doc_b = bd.ClayDoc()
        tab_b = clay_mode.ClayTab(doc=doc_b)
        state.add(tab_b)  # exactly what new_document/open/import/recover do

        assert state.active_uid == tab_b.uid
        assert not view.dragging, (
            "the drag must be settled, not left live on the tab it was replaced on"
        )
        assert doc_a.history.head != head_a, "one history step for the whole gesture"
        assert np.allclose(obj.translation, dragged_to), (
            "the commit must land on tab A, the tab the drag began on"
        )
        assert doc_b.history.head == 0, (
            "the new tab -- what it was replaced with -- must be untouched"
        )
    finally:
        view.release()


# --- clay-04: a copy family with a four-digit suffix -------------------------


def test_family_strips_a_four_digit_copy_suffix() -> None:
    """``ops.next_name`` counts a copy up past ``.999`` into ``.1000``,
    ``.10000``, and so on, never resetting the suffix width. ``diagnose.
    _family`` used to check only the character exactly four from the end for
    a literal ``.`` -- true for ``Box.001`` but false for ``Box.1000`` and
    every wider suffix after it -- so a family duplicated past 999 copies
    silently stopped being read as one: ``Box.1000`` came back as its own
    family, named after itself, instead of joining ``Box``.

    Fails against the unfixed code with:
        AssertionError: assert 'Box.1000' == 'Box'
    """
    assert diagnose._family("Box.1000") == "Box"
    assert diagnose._family("Box.10000") == "Box"
    # The ordinary three-digit case, and the bare (uncopied) name, both still
    # read the way they always did.
    assert diagnose._family("Box.001") == "Box"
    assert diagnose._family("Box") == "Box"
    # A name that merely contains a dot-digits run earlier, not at the end,
    # is not a copy suffix and must be left alone.
    assert diagnose._family("Box.001.glb") == "Box.001.glb"


# --- agents-01: a build preview landing after a tab switch -------------------


def test_a_build_preview_landing_after_switching_tabs_does_not_ghost_the_inactive_tab() -> None:
    """``_run_build_preview`` used to check only that the tab a build was
    requested against still existed, not that it was still the one on
    screen. ``ClayView.set_preview``/``_ghost_draws`` carry no document
    identity of their own -- the "added"/"changed" half draws straight from
    the scratch clone regardless of which tab's document is being rendered
    -- so a build that landed after the user switched tabs painted the ghost
    over whatever tab was now in front, a diff computed against a document
    nobody is looking at. Apply already refused a stale base at that point;
    landing a preview did not.

    Fails against the unfixed code with:
        AssertionError: a build for a now-inactive tab must not offer Apply/Discard
        assert [{'name': 'clay_add_primitive', ...}] is None
    (the preview landed -- ``ui.preview_calls`` was set and the ghost was
    handed to ``ctx.clay_view`` -- although the build was requested against a
    tab that is no longer active.)
    """
    doc_a = bd.ClayDoc()
    ctx = _FakeCtx(doc_a, mode="clay")
    tab_a = ctx.tab
    tab_b = clay_mode.ClayTab(doc=bd.ClayDoc())
    ctx.state.clay.add(tab_b)  # the user switched away from tab_a while Familiar thought
    assert ctx.state.clay.active_uid == tab_b.uid

    calls = _canned_calls()
    done = Done(
        key=familiar_ui.BUILD_KEY,
        result=calls,
        tag={"thread_key": ("clay", tab_a.uid), "tab_uid": tab_a.uid},
    )

    familiar_ui.on_task_done(ctx, done)

    ui = familiar_ui.ensure(ctx)
    assert ui.preview_calls is None, "a build for a now-inactive tab must not offer Apply/Discard"
    assert ctx.clay_view.previewed is None, "and must not have set the ghost either"
    assert ui.message is not None and "preview again" in ui.message


# --- agents-07: the character-plan landing path -------------------------


def test_on_task_done_lands_a_queued_character_as_a_thread_turn_and_toast() -> None:
    """A successful ``CHARACTER_KEY`` result adds a Familiar turn to the
    active thread and raises a toast -- neither had a test before this."""
    ctx = _FakeCtx(mode="clay")
    thread = ("clay", ctx.tab.uid)
    text = "Character queued -- it will appear in the Library."
    done = Done(
        key=familiar_ui.CHARACTER_KEY,
        result={"job_id": "job-1"},
        tag={"thread_key": thread},
    )

    familiar_ui.on_task_done(ctx, done)

    ui = familiar_ui.ensure(ctx)
    assert ui.reason is None
    assert ui.message is None
    turns = ctx.familiar_threads.get(thread)
    assert turns and turns[-1].role == "familiar" and turns[-1].text == text
    assert ctx.toasts == [text]


def test_on_task_done_reports_a_refused_character_with_no_thread_turn_or_toast() -> None:
    """A failed ``CHARACTER_KEY`` result -- a plain ``service.errors.Invalid``
    with no ``.reason``, the shape a missing Blender or a stale theme raises
    -- must land on the pane's refusal fields rather than as a chat turn, and
    must not toast: the card itself is what shows the refusal."""
    ctx = _FakeCtx(mode="clay")
    thread = ("clay", ctx.tab.uid)
    error = service_errors.Invalid("Blender isn't installed.")
    done = Done(
        key=familiar_ui.CHARACTER_KEY,
        error=error,
        message=error.message,
        tag={"thread_key": thread},
    )

    familiar_ui.on_task_done(ctx, done)

    ui = familiar_ui.ensure(ctx)
    assert ui.reason is None
    assert ui.message == "Blender isn't installed."
    assert ctx.familiar_threads.get(thread) == ()
    assert ctx.toasts == []


def test_on_task_done_reports_a_refused_character_familiar_refusal_reason() -> None:
    """The other failure shape ``CHARACTER_KEY`` can land: a
    ``svc_familiar.FamiliarRefusal`` (Familiar itself refused, rather than
    the character door) carries a ``.reason`` the pane surfaces."""
    ctx = _FakeCtx(mode="clay")
    thread = ("clay", ctx.tab.uid)
    error = svc_familiar.FamiliarRefusal("weights are missing", reason="missing")
    done = Done(
        key=familiar_ui.CHARACTER_KEY,
        error=error,
        message=error.message,
        tag={"thread_key": thread},
    )

    familiar_ui.on_task_done(ctx, done)

    ui = familiar_ui.ensure(ctx)
    assert ui.reason == "missing"
    assert ui.message == "weights are missing"
    assert ctx.familiar_threads.get(thread) == ()
    assert ctx.toasts == []
