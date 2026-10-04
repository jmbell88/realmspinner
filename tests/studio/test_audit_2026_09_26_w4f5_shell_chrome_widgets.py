"""Wave 4, fixer 5 of the 2026-09-26 audit fix pass.

Findings closed here: shell-chrome-03 (Mason's export chord), shell-chrome-06
(Empty the trash outside the loaded window), shell-chrome-07 (a raising
``on_confirm``/``on_accept`` dropping the next queued question), shell-widgets-01
(an orphaned slider gesture blocking undo eviction), and the Low findings
shell-chrome-08/09/10/11 and shell-widgets-02/03/04/05/06 that a design
decision or another fixer's file left open (see each test's docstring for the
disposition).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.studio import dialogs, palette

# --- shell-chrome-03: the printed export chord must match what the mode's
# own ``handle_key`` actually binds --------------------------------------------


def test_the_export_commands_printed_chord_is_the_one_masons_handle_key_binds():
    """The 2026-09-26 audit, finding shell-chrome-03: ``_doc_export_hint``
    printed "Ctrl+Shift+E" for every mode but Clay, while
    ``mason.mode._ctrl_key`` binds ``export_glb`` to plain ``e`` (``elif name
    == "e" and not shift:``) -- the same chord Clay uses and for the same
    reason (Mason's file export *is* its library export). Reproduced against
    the unfixed ``_doc_export_hint`` by asserting Mason's hint equals Clay's;
    before the fix Mason returned "Ctrl+Shift+E" and this failed.
    """
    ctx = SimpleNamespace(state=SimpleNamespace(mode="mason"))
    assert palette._doc_export_hint(ctx) == "Ctrl+E"

    clay_ctx = SimpleNamespace(state=SimpleNamespace(mode="clay"))
    assert palette._doc_export_hint(ctx) == palette._doc_export_hint(clay_ctx)

    # Every other document mode still prints the Ctrl+Shift+E its own
    # ``handle_key`` binds -- this is a Mason-specific carve-out, not a
    # blanket change.
    for mode in ("inker", "plotter", "packwright", "poser", "sirens"):
        other = SimpleNamespace(state=SimpleNamespace(mode=mode))
        assert palette._doc_export_hint(other) == "Ctrl+Shift+E"


# --- shell-chrome-06: "Empty the trash" must see the whole trash, not just
# the loaded window -------------------------------------------------------------


def _ctx_with_trash(window_jobs: list[dict[str, Any]], trashed: list[dict[str, Any]]) -> Any:
    # The store-wide answer arrives as ``cache.trash_present``, taken by the
    # job-list read task (the 2026-10-03 audit, shell-18, moved it off a
    # ``store.trashed()`` call the gate made on the frame thread); the store is
    # a tripwire so this ctx also proves the gate never asks it.
    class _Tripwire:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"the gate reached the store: {name}")

    return SimpleNamespace(
        cache=SimpleNamespace(jobs=window_jobs, trash_present=bool(trashed)),
        svc=SimpleNamespace(store=_Tripwire()),
    )


def test_empty_trash_is_enabled_when_the_only_trashed_rows_are_older_than_the_loaded_window():
    """The 2026-09-26 audit, finding shell-chrome-06: the palette's
    "Empty the trash..." command greyed itself from ``ctx.cache.jobs``, which
    is only the newest 200 rows (``jobs_cache.LIST_LIMIT``). A trash entirely
    older than that window is invisible to the cache, so the command reported
    "The trash is empty" while the store held rows to delete. Before the fix,
    the command read ``ctx.cache.jobs`` directly and this assertion failed
    with an empty, unpopulated window standing in for "no trash".
    """
    from studio.test_palette import _ctx as _full_ctx

    # Building the row needs the full command-list ctx (``commands()`` builds
    # every row, not just this one); the row's own ``enabled`` lambda then
    # only ever asks the trash question, so the minimal ctx below is enough
    # to call it with.
    command = next(c for c in palette.commands(_full_ctx()) if c.key == "empty-trash")

    empty_ctx = _ctx_with_trash(window_jobs=[{"id": "a", "deleted_at": None}], trashed=[])
    assert command.enabled(empty_ctx) is False

    # The window's newest 200 rows hold nothing trashed, but the store still
    # reports a trashed row outside that window.
    stale_trash_ctx = _ctx_with_trash(
        window_jobs=[{"id": "a", "deleted_at": None}],
        trashed=[{"id": "old", "deleted_at": 1.0}],
    )
    assert command.enabled(stale_trash_ctx) is True


def test_empty_trash_falls_back_to_the_cache_window_when_ctx_has_no_svc():
    """``specs()`` (menus and the palette both) builds every command's
    ``enabled`` on every call, including from callers -- ``test_menus.py`` and
    ``test_editor_shell.py`` among them -- that hand it a ctx built for a
    narrower question and never gave it a service door at all. This is the
    gate failure that fix produced: ``AttributeError: 'SimpleNamespace' object
    has no attribute 'svc'``. ``_any_trashed`` must fall back to the old
    ``cache.jobs`` scan rather than assume ``ctx.svc`` exists.
    """
    bare_ctx = SimpleNamespace(cache=SimpleNamespace(jobs=[{"id": "a", "deleted_at": None}]))
    assert palette._any_trashed(bare_ctx) is False

    bare_ctx_with_trash = SimpleNamespace(
        cache=SimpleNamespace(jobs=[{"id": "a", "deleted_at": 1.0}])
    )
    assert palette._any_trashed(bare_ctx_with_trash) is True

    # No ``cache`` either -- the emptiest ctx a caller could hand it.
    assert palette._any_trashed(SimpleNamespace()) is False


# --- shell-chrome-08: the viewport greyed reason must name a real place -------


def test_the_viewport_greyed_reason_names_a_real_place_rather_than_stale_panes():
    """The 2026-09-26 audit, finding shell-chrome-08: ``modes.VIEWPORT_MODES``
    has held only ``{"create"}`` for a while (Create stopped being a two-pane
    2D/3D split), but the four viewport commands' shared greyed reason still
    said "Only in the 2D and 3D panes, which are where the viewport is" --
    panes that no longer exist, in a mode this sentence never named.
    """
    from realmspinner.studio import modes

    assert frozenset({"create"}) == modes.VIEWPORT_MODES
    assert "panes" not in palette._VIEWPORT_WHY.lower()
    assert "create" in palette._VIEWPORT_WHY.lower()


# --- shell-chrome-07: a raising on_confirm/on_accept must not cost the next
# queued question -----------------------------------------------------------------


class _Enum:
    value = 0


class _Flags:
    always_auto_resize = _Enum()
    alpha = _Enum()
    enter_returns_true = _Enum()


class _Key:
    enter = object()
    keypad_enter = object()
    escape = object()


class _FakeConfirmImgui:
    """Enough imgui to run ``ConfirmQueue.draw`` once with a chosen button
    press. Modelled on ``test_dialogs_prompt.py``'s ``_FakeImgui`` -- the
    module docstring there notes no test yet drove ``ConfirmQueue.draw``
    against a fake, which this fills."""

    WindowFlags_ = _Flags
    StyleVar_ = _Flags
    Key = _Key

    def __init__(self, *, confirm: bool = False, cancel: bool = False) -> None:
        self._confirm = confirm
        self._cancel = cancel
        self.ended = False

    def open_popup(self, _title: str) -> None: ...
    def push_style_var(self, *_a: Any) -> None: ...
    def pop_style_var(self, *_a: Any) -> None: ...
    def set_next_window_bg_alpha(self, *_a: Any) -> None: ...
    def begin_popup_modal(self, *_a: Any) -> tuple[bool, Any]:
        return (True, None)

    def dummy(self, *_a: Any) -> None: ...
    def text_wrapped(self, *_a: Any) -> None: ...
    def same_line(self) -> None: ...
    def is_any_item_active(self) -> bool:
        return False

    def set_item_default_focus(self) -> None: ...
    def is_key_pressed(self, _key: Any) -> bool:
        return False

    def close_current_popup(self) -> None: ...
    def end_popup(self) -> None:
        self.ended = True


def _run_confirm_draw(
    monkeypatch: pytest.MonkeyPatch, queue: dialogs.ConfirmQueue, fake: Any
) -> None:
    from realmspinner.studio import widgets

    monkeypatch.setattr(widgets, "frosted", lambda: False)
    monkeypatch.setattr(widgets, "destructive_button", lambda *a, **k: fake._confirm)
    monkeypatch.setattr(widgets, "ghost_button", lambda *a, **k: fake._cancel)
    # ``waiting`` is nonzero whenever a second question is queued behind the
    # one on screen, which draws through the real ``widgets.muted`` -- an
    # access violation with no imgui context, the same reason
    # ``test_dialogs_prompt.py`` pins ``frosted``/``field_label``.
    monkeypatch.setattr(widgets, "muted", lambda *a, **k: None)
    monkeypatch.setattr(dialogs, "imgui", fake)
    queue.draw()


def test_a_raising_on_confirm_does_not_drop_the_next_queued_question(monkeypatch):
    """The 2026-09-26 audit, finding shell-chrome-07: ``draw`` popped the
    answered ``Confirm`` before calling ``on_confirm``. Wired as
    ``guard.run(..., ctx.confirms.draw, on_failure=ctx.confirms.dismiss)``, a
    raising callback let the exception reach ``dismiss``, which popped again --
    dropping the *next* queued question rather than the one that failed.

    Reproduced directly against the unfixed queue (no ``_answering`` guard):
    calling ``_answered()`` then letting the callback raise, then calling
    ``dismiss()`` the way ``guard.run`` does, removed the second Confirm too.
    """
    seen: list[str] = []

    def bad() -> None:
        raise RuntimeError("boom")

    first = dialogs.Confirm(title="First", message="m", on_confirm=bad)
    second = dialogs.Confirm(title="Second", message="m", on_confirm=lambda: seen.append("second"))

    queue = dialogs.ConfirmQueue()
    queue.ask(first)
    queue.ask(second)

    fake = _FakeConfirmImgui(confirm=True)
    with pytest.raises(RuntimeError):
        _run_confirm_draw(monkeypatch, queue, fake)
    # Mirrors guard.run's except clause: on_failure=queue.dismiss.
    queue.dismiss()

    assert queue.pending is second, "the next queued question must survive a raising callback"
    assert fake.ended, "end_popup must run even though the callback raised"

    fake2 = _FakeConfirmImgui(confirm=True)
    _run_confirm_draw(monkeypatch, queue, fake2)
    assert seen == ["second"]
    assert queue.pending is None


def test_a_raising_on_accept_does_not_drop_the_next_queued_prompt(monkeypatch):
    """The ``PromptQueue`` twin of the test above -- same fix, same shape."""
    from realmspinner.studio import widgets

    monkeypatch.setattr(widgets, "frosted", lambda: False)
    monkeypatch.setattr(widgets, "field_label", lambda *a, **k: None)
    monkeypatch.setattr(widgets, "muted", lambda *a, **k: None)

    class _PromptImgui(_FakeConfirmImgui):
        Cond_ = _Flags
        InputTextFlags_ = _Flags

        def __init__(self, *, typed: str, press: str) -> None:
            super().__init__()
            self.typed = typed
            self.press = press

        def get_main_viewport(self) -> Any:
            return type("V", (), {"get_center": staticmethod(lambda: (0.0, 0.0))})()

        def set_next_window_pos(self, *_a: Any) -> None: ...
        def set_next_item_width(self, _w: float) -> None: ...
        def set_keyboard_focus_here(self) -> None: ...
        def input_text(self, _label: str, _value: str, _flags: int) -> tuple[bool, str]:
            return (False, self.typed)

        def button(self, label: str, _size: Any = None) -> bool:
            return label == self.press

        @staticmethod
        def ImVec4(*values: float) -> tuple[float, ...]:
            return values

        def text_colored(self, *_a: Any) -> None: ...

    def bad(_value: str) -> None:
        raise RuntimeError("boom")

    seen: list[str] = []
    first = dialogs.Prompt(title="First", label="Name", value="a", on_accept=bad)
    second = dialogs.Prompt(title="Second", label="Name", value="b", on_accept=seen.append)

    queue = dialogs.PromptQueue()
    queue.ask(first)
    queue.ask(second)

    monkeypatch.setattr(dialogs, "imgui", _PromptImgui(typed="a", press="Save"))
    with pytest.raises(RuntimeError):
        queue.draw()
    queue.dismiss()

    assert queue.pending is second, "the next queued prompt must survive a raising on_accept"

    monkeypatch.setattr(dialogs, "imgui", _PromptImgui(typed="b", press="Save"))
    queue.draw()
    assert seen == ["b"]
    assert queue.pending is None


# --- shell-widgets-01: an orphaned gesture must close on tab close and mode
# switch, not only on the next activation ---------------------------------------


class _PlainStep:
    """A minimal ``undo.Edit`` -- only ``undo``/``redo`` need to exist, and
    neither is ever called here."""

    cost = 0
    serial = 0
    label = ""

    def undo(self, doc: Any) -> None: ...
    def redo(self, doc: Any) -> None: ...


def _open_orphaned_gesture(monkeypatch: pytest.MonkeyPatch, history: Any) -> None:
    """Open a gesture on ``history`` the way a slider mid-drag does, then leave
    it open: the item has stopped drawing (its pane closed under a tab close or
    a mode switch), so nothing will ever report the deactivation that would
    otherwise close it (``test_an_orphaned_gesture_is_closed_by_the_next_activation``
    in ``tests/test_undo_gesture_doors.py`` already covers the "something else
    activates" case; this is the case nothing does).
    """
    from imgui_bundle import imgui

    from realmspinner.studio import controls

    monkeypatch.setattr(imgui, "is_item_activated", lambda: True)
    monkeypatch.setattr(imgui, "is_item_deactivated", lambda: False)
    monkeypatch.setattr(controls, "_gesture", None)
    controls.fold_undo(history)
    for _ in range(3):
        history.push(_PlainStep())


def _old_set_mode(state_ns: Any, key: str) -> bool:
    """``state.set_mode`` exactly as of ``git show HEAD`` for this file, before
    the shell-widgets-01 fix added the ``controls.close_gesture()`` call. Run
    as a throwaway function per this pass's constraints (never checked out
    over the fixed tree file) to prove the finding against the code as it
    stood.
    """
    from realmspinner.studio import state as state_mod

    if key == state_ns.mode:
        return False
    if state_mod._MODE_AVAILABLE is not None and not state_mod._MODE_AVAILABLE(key):
        return False
    leaving = state_ns.mode
    state_ns.previous_mode = state_ns.mode
    state_ns.mode_observed = key
    state_ns.mode = key
    if state_mod._MODE_LEAVE is not None:
        state_mod._MODE_LEAVE(leaving)
    return True


def test_an_orphaned_slider_gesture_does_not_switch_off_undo_eviction_for_later_edits(
    monkeypatch,
):
    """The 2026-09-26 audit, finding shell-widgets-01: a slider drag whose
    item stops drawing -- its pane closed, or the mode switched away -- leaves
    ``controls._gesture`` open forever, since nothing but *another* item's
    activation ever closes a stray one. ``UndoStack._open_gestures`` then
    never drops back to zero, so ``push``'s deferred eviction never runs again
    for that document: the audit measured 500 steps piling up against a cap of
    64.

    Reproduced first against the pre-fix ``set_mode`` (``_old_set_mode``,
    ``git show HEAD``): switching modes left the gesture open. Then against
    the fixed ``state.set_mode`` and ``docmodes.close_tab``, both of which now
    close it, and eviction resumes.
    """
    from realmspinner.core import undo
    from realmspinner.studio import controls, docmodes
    from realmspinner.studio import state as state_mod

    # -- pre-fix: reproduced, not assumed --------------------------------------
    stuck = undo.UndoStack()
    _open_orphaned_gesture(monkeypatch, stuck)
    assert stuck._open_gestures == 1
    moved = _old_set_mode(
        SimpleNamespace(mode="create", previous_mode="create", mode_observed="create"), "clay"
    )
    assert moved is True
    assert stuck._open_gestures == 1, "pre-fix: set_mode left the stray gesture open"

    # -- fixed: mode switch -----------------------------------------------------
    via_mode_switch = undo.UndoStack()
    _open_orphaned_gesture(monkeypatch, via_mode_switch)
    assert via_mode_switch._open_gestures == 1
    state_mod.set_mode(
        SimpleNamespace(mode="create", previous_mode="create", mode_observed="create"), "clay"
    )
    assert via_mode_switch._open_gestures == 0
    assert controls._gesture is None
    for _ in range(undo.UNDO_MAX_DEPTH * 8):
        via_mode_switch.push(_PlainStep())
    assert len(via_mode_switch) <= undo.UNDO_MAX_DEPTH, "eviction must have resumed"

    # -- fixed: tab close ---------------------------------------------------------
    via_tab_close = undo.UndoStack()
    _open_orphaned_gesture(monkeypatch, via_tab_close)
    assert via_tab_close._open_gestures == 1

    class _Tab:
        saving = False
        dirty = False

    class _Tabs:
        def __init__(self, tab: Any) -> None:
            self._tab = tab

        def get(self, _uid: str) -> Any:
            return self._tab

        def close(self, _uid: str) -> None: ...

    docmodes.close_tab(SimpleNamespace(), _Tabs(_Tab()), "uid", release=lambda _tab: None)
    assert via_tab_close._open_gestures == 0
    assert controls._gesture is None
    for _ in range(undo.UNDO_MAX_DEPTH * 8):
        via_tab_close.push(_PlainStep())
    assert len(via_tab_close) <= undo.UNDO_MAX_DEPTH, "eviction must have resumed"


# --- shell-widgets-02: the grade row and the tag rows must be able to say why
# they are disabled together, not just individually -----------------------------


def test_grade_buttons_and_tag_toggles_can_explain_why_they_are_disabled():
    """The 2026-09-26 audit, finding shell-widgets-02: both rows took an
    ``enabled`` flag and no ``reason``, so a caller that greys the whole row
    at once -- an ungradeable candidate, a locked review -- had nowhere to
    say why: 21 dead-looking buttons (11 grades, 10 tags) while scanning.
    """
    import ast
    import inspect

    from realmspinner.studio import widgets

    for name in ("grade_buttons", "tag_toggles"):
        func = getattr(widgets, name)
        assert "reason" in inspect.signature(func).parameters, f"{name} takes no reason"
        tree = ast.parse(inspect.getsource(func))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "disabled_button"
        ]
        assert calls, f"{name} no longer calls disabled_button"
        for call in calls:
            names = {kw.arg for kw in call.keywords}
            assert "reason" in names, f"{name}'s disabled_button call forwards no reason"


# --- shell-widgets-03: a non-finite stored UI scale must not silently pick
# the smallest step -----------------------------------------------------------


def _old_nearest_ui_scale(value: float, monitor_scale: float = 1.0) -> float:
    """``tokens.nearest_ui_scale`` exactly as of ``git show HEAD`` for this
    file, before the shell-widgets-03 fix added the ``math.isfinite`` check.
    Run as a throwaway function per this pass's constraints.
    """
    from realmspinner.studio import tokens as tokens_mod

    steps = tokens_mod.ui_scale_steps(monitor_scale)
    try:
        wanted = float(value)
    except (TypeError, ValueError):
        wanted = 1.0
    return min(steps, key=lambda step: (abs(step - wanted), step))


def test_nearest_ui_scale_refuses_a_non_finite_wanted_value():
    """The 2026-09-26 audit, finding shell-widgets-03: ``float("inf")``/
    ``float("nan")`` do not raise in ``float(value)``, so a corrupt or
    hand-edited settings file carrying either one reached
    ``abs(step - wanted)`` as ``inf``/``nan`` against every step alike --
    ``min`` then has nothing it can call smaller than anything else and
    silently returns the first step it was handed, 0.5x, on every launch.
    """
    from realmspinner.studio import tokens

    # -- pre-fix: reproduced, not assumed --------------------------------------
    assert _old_nearest_ui_scale(float("inf")) == 0.5
    assert _old_nearest_ui_scale(float("nan")) == 0.5

    # -- fixed ------------------------------------------------------------------
    assert tokens.nearest_ui_scale(float("inf")) == 1.0
    assert tokens.nearest_ui_scale(float("-inf")) == 1.0
    assert tokens.nearest_ui_scale(float("nan")) == 1.0


# --- shell-widgets-04: a non-serialisable settings value must not raise out
# of flush, every debounced tick -------------------------------------------------


def test_a_save_with_a_non_serialisable_value_does_not_raise_out_of_flush(tmp_path):
    """The 2026-09-26 audit, finding shell-widgets-04: ``json.dumps`` ran
    before ``flush``'s own ``try``, and the ``try`` caught only ``OSError`` --
    so one non-serialisable value in ``self.data`` raised straight out of
    ``flush``, and out of ``tick``, which calls it on every debounced frame:
    the app never stopped raising once it had.
    """
    import json

    from realmspinner.studio import settings as settings_mod

    settings = settings_mod.Settings.load(tmp_path)
    settings.set("bad", {1, 2, 3})  # a set: valid Python, not valid JSON

    # -- pre-fix: reproduced against the exact statement ``flush`` ran
    # unguarded (``git show HEAD``: ``json.dumps`` was the first line in the
    # function, before its ``try``). -------------------------------------------
    with pytest.raises(TypeError):
        json.dumps({"version": settings_mod.VERSION, "data": settings.data}, indent=2)

    # -- fixed: the encode now runs inside flush's own try/except -------------
    assert settings.flush() is False
    notice = settings.take_notice()
    assert notice is not None and "cannot be saved" in notice
    # And it does not raise a fresh notice on every retry -- the same latch
    # the OSError arm already had.
    assert settings.flush() is False
    assert settings.take_notice() is None


# --- shell-widgets-05: the DPI docstrings must not claim the scale is never
# re-sampled, now that a display change re-samples it --------------------------


def test_the_dpi_docstrings_no_longer_claim_the_scale_is_never_resampled():
    """The 2026-09-26 audit, finding shell-widgets-05: both ``tokens.py`` and
    ``dpi.py`` said the DPI scale is sampled once at startup and never again --
    stale since ``shell.events._resample_display_scale`` (UX-22) started
    re-sampling it on a display change.
    """
    import inspect

    from realmspinner.studio import dpi, tokens

    # The module summary no longer states the stale claim as a flat,
    # unqualified fact (the fix's own explanation still quotes the old
    # wording historically, which is why this is not a whole-file substring
    # check).
    assert (dpi.__doc__ or "").splitlines()[0] == "Windows DPI awareness."
    tokens_summary = "Design tokens: the numbers the UI is drawn from."
    assert (tokens.__doc__ or "").splitlines()[0] == tokens_summary
    # And both now cite the mechanism that makes the old claim false.
    assert "_resample_display_scale" in inspect.getsource(tokens)
    assert "_resample_display_scale" in inspect.getsource(dpi)


# --- shell-widgets-06: a late background decode of an older mtime must not
# replace a newer texture -------------------------------------------------------


class _FakeThumbTexture:
    def __init__(self, size: Any) -> None:
        self.size = size
        self.filter = None
        self.repeat_x = True
        self.repeat_y = True
        self.released = False
        self.glo = id(self)

    def release(self) -> None:
        self.released = True


class _FakeThumbGL:
    NEAREST = "nearest"
    LINEAR = "linear"

    def texture(self, size: Any, _components: int, _data: bytes) -> Any:
        return _FakeThumbTexture(size)


def _done_future(result: Any) -> Any:
    from concurrent.futures import Future

    future: Future = Future()
    future.set_result(result)
    return future


def _old_adopt_finished_decodes(cache: Any) -> None:
    """``ThumbnailCache._adopt_finished_decodes`` exactly as of ``git show
    HEAD`` for this file, before the shell-widgets-06 fix added the
    newer-entry check below. Run as a throwaway bound function per this
    pass's constraints (never by editing the tree) to prove the finding
    against the code as it stood.
    """
    done = [key for key, future in cache._inflight.items() if future.done()]
    for key in done:
        future = cache._inflight.pop(key)
        job_id, mtime, nearest, _max_side = key
        try:
            decoded = future.result()
        except Exception:
            decoded = None
        if decoded is None:
            cache._missing.add(key)
            cache._missing_by_key.setdefault(job_id, set()).add(key)
            continue
        size, data = decoded
        try:
            texture = cache.ctx.texture(size, 4, data)
        except Exception:
            continue
        mode = cache.ctx.NEAREST if nearest else cache.ctx.LINEAR
        texture.filter = (mode, mode)
        texture.repeat_x = texture.repeat_y = False
        cache._supersede(job_id, mtime)
        cache._insert(key, texture)


def test_a_late_background_decode_of_an_older_mtime_does_not_replace_a_newer_texture():
    """The 2026-09-26 audit, finding shell-widgets-06: the decode pool has two
    workers and no ordering guarantee, so a decode queued for an older mtime
    can land (``_adopt_finished_decodes``, a *later* frame) after one queued
    afterwards for a newer mtime already landed and was inserted.
    ``_supersede`` retires every entry under the job id with a *different*
    mtime, older or not -- so the stale decode retired the texture that was
    actually current and installed the old one in its place.

    Reproduced first against the pre-fix method (``_old_adopt_finished_decodes``,
    ``git show HEAD``), then against the fixed one.
    """
    from realmspinner.studio.textures import MAX_SIDE, ThumbnailCache

    job_id = "job:x"
    old_key = (job_id, 100.0, False, MAX_SIDE)
    new_key = (job_id, 200.0, False, MAX_SIDE)
    decoded = ((8, 8), b"\x00" * (8 * 8 * 4))

    # -- pre-fix: reproduced, not assumed --------------------------------------
    stuck = ThumbnailCache(_FakeThumbGL())
    stuck._inflight[new_key] = _done_future(decoded)
    _old_adopt_finished_decodes(stuck)
    assert new_key in stuck._entries
    stuck._inflight[old_key] = _done_future(decoded)
    _old_adopt_finished_decodes(stuck)
    assert new_key not in stuck._entries, (
        "pre-fix: the late, older decode evicted the newer texture"
    )
    assert old_key in stuck._entries

    # -- fixed ------------------------------------------------------------------
    cache = ThumbnailCache(_FakeThumbGL())
    cache._inflight[new_key] = _done_future(decoded)
    cache._adopt_finished_decodes()
    assert new_key in cache._entries
    newer_texture = cache._entries[new_key]

    cache._inflight[old_key] = _done_future(decoded)
    cache._adopt_finished_decodes()
    assert new_key in cache._entries, "the newer texture must survive a late, stale decode"
    assert cache._entries[new_key] is newer_texture
    assert old_key not in cache._entries, "the stale decode must be discarded, not installed"
