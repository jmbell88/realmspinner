"""Sirens' controller: documents, the caret, rendering, playback and keys.

The layer that knows about task threads and a sound device; the engine under
``sirens/`` knows about neither, and that separation is the whole reason a
machine with no card can still use this mode. The panes draw, this decides.

**Rendering is a task, and re-arming it is a flag.** A three-minute song is a
tick loop over every channel plus a 4x-oversampled decimation per voice, which
is seconds of numpy -- not frame-thread work -- so a render goes through
``ctx.submit`` under ``sirens-render:<uid>`` and the result is adopted in
:func:`on_task_done`. What re-arms it is ``SongTab.render_dirty``, pumped from
the grid pane's draw and cleared *only when a submit is accepted*: this is
``PackTab.pack_dirty`` verbatim, including its rule that the flag is cleared at
the submit and never at the adoption. ``TaskRunner.submit`` refuses a key
already in flight and nothing re-arms it, so a burst of keystrokes -- which is
what typing a bar *is* -- would otherwise render the song as it stood at the
first one and drop every note after it.

**Rendering reads a snapshot, not the document.** ``rsng.rsng_bytes`` is the
snapshot: the task re-reads a document from those bytes and renders *that*, so
the frame thread may keep editing the live one while the task runs. It costs a
zip round-trip per render and it buys the only version of this that cannot tear
-- a numpy view handed to a thread is a view of an array the caret is writing
into.

**Playback never blocks and never fails loudly.** ``sirens_audio`` answers
rather than raises, so :func:`play` on a machine with no device toasts once and
leaves the document exactly as it was.

Every task key carries the ``sirens-`` prefix, because the app claims results
by prefix: a key without one is a result delivered nowhere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ... import dialogs, docmodes, journal
from ...state import set_mode
from . import audio as sirens_audio
from . import fileio as sirens_io
from . import state as sirens_state

# ``ensure`` and ``active`` live in :mod:`.sirens_state` -- they touch nothing
# but ``ctx.state.sirens`` -- and the file layer lives in :mod:`.sirens_io`.
# Both are re-exported here, as **plain imports rather than wrappers**, because
# every pane, every key binding and every test says ``sirens_mode.save(ctx)``:
# a wrapper would be a second object where the callers reach for one.
from .fileio import (  # noqa: F401
    EXPORT_PREFIX,
    SAMPLE_FILTER,
    SAMPLE_PREFIX,
    SFX_DIR,
    SONG_FILTER,
    SONG_NAME,
    STEM_DIR,
    WAV_FILTER,
    _load,
    _start,
    ask_open,
    ask_sample,
    export_files,
    export_plan,
    export_to,
    free_sample_key,
    import_sample,
    open_path,
    save,
    save_as,
    save_to,
)
from .state import (  # noqa: F401
    COLUMN_DIGITS,
    SirensState,
    SongTab,
    active,
    ensure,
)

log = logging.getLogger(__name__)


# The three recents wrappers every document mode carries, over the one
# list Home's Resume rows are built from (``docmodes.recents_for``).
remember_path, forget_path, recent_paths = docmodes.recents_for("sirens")


def persist(ctx: Any) -> None:
    """Nothing to write: the recent list is :mod:`.recents`, which persists
    itself on every write. Kept as a no-op for ``packwright_mode.persist``'s
    reason -- it is called after every open and save, and turning each of those
    into "call this only if the mode still has settings" is how one of them
    comes to skip a write that mattered later."""


# --- documents ----------------------------------------------------------------


def adopt(ctx: Any, doc: Any, *, path: Path | None = None, title: str | None = None) -> SongTab:
    state = ensure(ctx)
    tab = SongTab(
        doc=doc,
        title=title or sirens_state.title_for(path),
        path=path,
        saved_head=doc.history.head,
    )
    state.add(tab)
    remember_path(ctx, path)
    persist(ctx)
    return tab


def new_document(ctx: Any) -> SongTab:
    """A song that plays the moment somebody types into it.

    ``document.new_song`` rather than a bare ``SongDoc()``: an empty order list
    is a document where Space does nothing and there is no way to find out why.
    """
    from .engine import document

    return adopt(ctx, document.new_song(), title="Untitled")


# --- the caret ----------------------------------------------------------------


def caret_pattern(ctx: Any, tab: SongTab | None = None) -> Any:
    """The pattern the caret is in, or ``None``.

    Every caller wants the object rather than the uid, and reaching for it
    through ``doc.pattern(state.pattern)`` at six sites is six places to forget
    that the uid can be stale -- a pattern the user has just deleted is the
    ordinary way that happens.
    """
    state = ensure(ctx)
    tab = tab or state.active
    if tab is None or state.pattern is None:
        return None
    return tab.doc.pattern(state.pattern)


def caret_pattern_label(ctx: Any, tab: SongTab | None = None) -> str:
    """What the grid is editing, in words. Empty when it is editing nothing.

    **Adding a sound effect silently repoints the grid.** ``add_oneshot`` mints
    the effect *and* a pattern of its own and moves the caret onto it, which is
    the right behaviour -- the grid is the effect editor and there is no second
    one -- but until this readout existed the only thing on screen that said so
    was a line of muted text in a panel in the right-hand column, and a reader
    who had scrolled it away had no way to tell a song pattern from an effect.
    Which one is loaded decides what every keystroke is editing, so it belongs
    above the grid rather than beside it.

    A reverse lookup, because a one-shot *is* a pattern (``sirens/document``'s
    "One-shots are patterns"): the effect owns a pattern uid, so the question
    "is this pattern an effect" is answered by asking the effects, not the
    pattern. Pure state to string, so it is assertable without a frame.
    """
    state = ensure(ctx)
    tab = tab or state.active
    if tab is None or state.pattern is None:
        return ""
    pattern = tab.doc.pattern(state.pattern)
    if pattern is None:
        return ""
    for one in tab.doc.oneshots:
        if one.pattern == state.pattern:
            # The effect's name, not the pattern's: the effect is the thing the
            # user named and the thing the exported WAV is called after.
            return f"{one.name or 'effect'} - sound effect"
    return f"pattern {pattern.name}" if pattern.name else "song pattern"


def oneshot_name_for_caret(ctx: Any, tab: SongTab | None = None) -> str:
    """The sound effect the grid is editing, by name, or "".

    The reverse lookup ``caret_pattern_label`` already does, as a *fact* rather
    than a sentence, because two doors need it: the label above the grid, and
    the "+ To order" button -- an effect's pattern is not part of the song, and
    appending one used to put a coin pickup in the middle of it silently.
    """
    state = ensure(ctx)
    tab = tab or state.active
    if tab is None or state.pattern is None:
        return ""
    for one in tab.doc.oneshots:
        if one.pattern == state.pattern:
            return one.name or "effect"
    return ""


def confirm_remove_pattern(ctx: Any, tab: SongTab, uid: int, used: int) -> None:
    """Delete a pattern, asking first when the order list refers to it.

    ``remove_pattern`` drops every order entry naming the pattern in the same
    undo step, which is right -- a song that refers to a pattern which does not
    exist must not be one Ctrl+Z away -- and is exactly the sort of thing a
    person should be told before the press rather than after it. A pattern
    nothing plays is deleted without a question, because there is nothing to
    warn about.
    """
    def go() -> None:
        from . import edit as sirens_edit
        from . import play as sirens_play

        if tab.doc.remove_pattern(uid):
            sirens_play.request_rerender(ctx, tab)
            remaining = tab.doc.patterns
            if remaining:
                sirens_edit.set_caret(ctx, pattern=remaining[0].uid)

    if not used:
        go()
        return
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Delete this pattern?",
            message=(
                f"It is in the order list {used} time(s), and those entries go "
                "with it. One Ctrl+Z brings both back."
            ),
            confirm_label="Delete",
            cancel_label="Keep it",
            on_confirm=go,
        )
    )


def audible_channels(doc: Any, view: Any) -> tuple[int, ...]:
    """Which channel *indices* the mix should play. Pure.

    Solo wins over mute, which is every tracker's rule and the one that makes
    the pair usable: soloing to check a bass line and then unsoloing must not
    have to undo four mutes. A solo naming a channel the song no longer has
    falls back to "everything", because a removed channel is not a reason for
    silence.

    ``view`` is the **tab**, not the app state: the mask is per document since
    S5, because channel uids restart per document and a mask on ``SirensState``
    silenced a channel of the same number in every other song.
    """
    channels = list(getattr(doc, "channels", ()))
    solo = int(getattr(view, "solo", -1))
    muted = set(getattr(view, "muted", ()) or ())
    order = [one.uid for one in channels]
    if solo >= 0 and solo in order:
        return (order.index(solo),)
    return tuple(index for index, uid in enumerate(order) if uid not in muted)


def toggle_mute(ctx: Any, uid: int, tab: SongTab | None = None) -> bool:
    """Mute or unmute one channel. -> whether it is now muted.

    Re-renders, because the mute lives in the mix rather than in the document:
    the buffer Space plays is the render, and a mute that only changed a button
    would be a control that does nothing until the next edit.
    """
    from . import play as sirens_play

    state = ensure(ctx)
    tab = tab or state.active
    if tab is None:
        return False
    uid = int(uid)
    if uid in tab.muted:
        tab.muted.discard(uid)
        muted = False
    else:
        tab.muted.add(uid)
        muted = True
    sirens_play.request_rerender(ctx, tab)
    return muted


def toggle_solo(ctx: Any, uid: int, tab: SongTab | None = None) -> int:
    """Solo one channel, or clear the solo when it is already this one. -> the
    soloed uid, or ``-1``."""
    from . import play as sirens_play

    state = ensure(ctx)
    tab = tab or state.active
    if tab is None:
        return -1
    tab.solo = -1 if tab.solo == int(uid) else int(uid)
    sirens_play.request_rerender(ctx, tab)
    return tab.solo


def channel_state(ctx: Any, uid: int) -> tuple[bool, bool, bool]:
    """``(muted, soloed, audible)`` for one channel -- what the header draws.

    Three answers rather than two because "muted" and "not heard" are different
    facts: a channel that is not the soloed one is silent without being muted,
    and a button that lit up for both would be lying about what a click undoes.
    """
    state = ensure(ctx)
    tab = state.active
    if tab is None:
        # No document, so no mask: nothing is muted and nothing is soloed.
        return False, False, True
    muted = int(uid) in tab.muted
    soloed = tab.solo == int(uid)
    order = [one.uid for one in tab.doc.channels]
    if int(uid) not in order:
        return muted, soloed, False
    audible = order.index(int(uid)) in audible_channels(tab.doc, tab)
    return muted, soloed, audible


# --- task results -------------------------------------------------------------


def _still_wanted(state: Any, done: Any) -> bool:
    """Whether the completion in ``done`` is the sound the user is still asking
    for (S1, 2026-09-05). -> ``False`` for a stale one, which is dropped.

    The three audition branches below hand their result straight to the mixer,
    and used to do it with no freshness check whatever: press a note, press
    Stop, hear the note a second later, because the render was still on a task
    thread when the device was silenced. ``SirensState.play_request`` counts
    requests and ``stop`` bumps it; ``TaskRunner`` carries the value the request
    was made under in ``Done.tag`` -- the slot has been threaded through
    ``submit`` since it was written and had no reader until now -- so a stale
    completion is one whose tag is not the current count. No cancellation is
    involved and none is needed: the render finishes and its samples are simply
    not played, which is ``MuseState.audition_job``'s answer (M11) with a
    counter in place of a job id, because not everything that sounds here has
    one.
    """
    tag = getattr(done, "tag", None)
    return tag is None or int(tag) == int(state.play_request)


def on_task_done(ctx: Any, done: Any) -> None:
    from . import play as sirens_play

    state = ensure(ctx)
    key, result = done.key, done.result
    name = key.split(":", 1)[0]

    if name == "sirens-open":
        if isinstance(result, dict):
            # The 2026-09-18 audit (second run, finding sirens-01) found this
            # arm adopting unconditionally while ``open_path`` already guards
            # with "Focus rather than fork: two tabs over one path would race
            # on save" -- the same gap Clay closed on 2026-09-12 (clay-02).
            path = Path(result["path"]) if result.get("path") else None
            existing = state.find_path(path) if path is not None else None
            if existing is not None:
                state.activate(existing.uid)
            else:
                adopt(
                    ctx,
                    result["doc"],
                    path=path,
                    title=result.get("title"),
                )
            set_mode(ctx.state, "sirens")
        return

    tab = state.get(key.split(":", 1)[1]) if ":" in key else None
    if tab is None:
        return

    if key.startswith(sirens_play.RENDER_PREFIX):
        if isinstance(result, dict):
            tab.adopt_render(
                result["pcm"], result.get("loop"), result.get("marks") or ()
            )
        else:
            tab.rendering = False
        return

    if name == "sirens-pattern":
        # The caret's pattern, alone, straight to the mixer -- the audition's
        # shape, and tagged so the song's playhead does not bisect the song's
        # row map against a single pattern's clock.
        if not _still_wanted(state, done):
            return
        if isinstance(result, dict) and not sirens_audio.play(
            result["pcm"], tag=f"pattern:{result.get('pattern', '')}"
        ):
            docmodes.refuse(ctx, "That pattern could not be played; see the log.")
        return

    if name == "sirens-preview":
        # Straight to the mixer, under a tag of its own so a preview never
        # displaces the playhead's reading of what the song is doing.
        if not _still_wanted(state, done):
            return
        if isinstance(result, dict):
            sirens_audio.play(result["pcm"], tag="preview")
        return

    if name == "sirens-audition":
        # Straight to the mixer. Deliberately not through :func:`play`, which
        # is about ``tab.pcm`` -- see :data:`AUDITION_PREFIX` for why an effect
        # never lands there.
        if not _still_wanted(state, done):
            return
        if isinstance(result, dict) and not sirens_audio.play(result["pcm"]):
            docmodes.refuse(ctx, "That sound effect could not be played; see the log.")
        return

    if name == "sirens-sample":
        if isinstance(result, dict):
            from . import edit as sirens_edit

            # The mode switch is the *last* thing, and only on a key: a sample
            # the document refused (a full table, a name it would not take) has
            # not landed, and moving the window for it would put the sentence
            # about the refusal in a mode the user did not ask to be in.
            if sirens_edit.adopt_sample(ctx, tab, result) and result.get("switch"):
                set_mode(ctx.state, "sirens")
        return

    if name == "sirens-export":
        # Its own arm rather than the fall-through below, because an export is
        # not a save: it writes files *derived* from the document and leaves the
        # document exactly as dirty as it was. Falling through would call
        # ``mark_saved`` and drop the crash copy of work still only in memory.
        tab.saving = False
        if isinstance(result, dict):
            ctx.toast(f"Exported {result['files']} file(s) to {result['directory']}")
        return

    tab.saving = False
    if not isinstance(result, dict):
        return  # a cancelled dialog

    tab.mark_saved(result.get("head"))
    # Saved is the moment the crash copy stops describing anything at risk
    # (UX-05); see ``inker_mode``.
    journal.drop(ctx, tab)
    if result.get("retitle") and result.get("path"):
        tab.path = Path(result["path"])
        tab.title = sirens_state.title_for(tab.path)
        remember_path(ctx, tab.path)
        persist(ctx)
    ctx.toast("Saved.")


def on_task_failed(ctx: Any, done: Any) -> None:
    """A failed save must not leave the document locked, and a failed *render*
    must clear ``rendering`` and record why -- a dead Play button with no
    sentence beside it is the worst outcome of a song that would not render."""
    from . import play as sirens_play

    if done.key.startswith(sirens_io.OPEN_PREFIX):
        # Before the tab lookup, because an open that failed has no tab: what
        # it has is a path that does not open, and a Resume list that keeps
        # offering one is worse than a short one.
        forget_path(ctx, done.key.split(":", 1)[1])
        return
    if done.key.startswith(
        (
            sirens_play.AUDITION_PREFIX,
            sirens_play.PATTERN_PREFIX,
            sirens_play.PREVIEW_PREFIX,
            sirens_io.SAMPLE_PREFIX,
        )
    ):
        # A refused sample or audition has nothing to unlock: the tab was never
        # locked for either, and clearing ``saving`` here would unlock a *save*
        # that happened to be running alongside. The sentence the user is owed
        # is already the toast the classifier drew before this was called. An
        # *export* is not in this clause, deliberately -- it does lock the tab
        # (``docmodes.start_save``), so it falls through to the unlock below.
        return
    state = ctx.state.sirens
    if state is None or ":" not in done.key:
        return
    tab = state.get(done.key.split(":", 1)[1])
    if tab is None:
        return
    tab.saving = False
    if done.key.startswith(sirens_play.RENDER_PREFIX):
        tab.rendering = False
        tab.render_error = done.message or "That song did not render."


# --- guard and keys -----------------------------------------------------------


def guard(ctx: Any, verb: str, proceed: Any) -> bool:
    """Ask before losing unsaved work. -> whether it went ahead now.

    One question for all of them, the ``packwright_mode.guard`` shape. Only
    quitting and closing a tab are destructive: switching modes is not, because
    Sirens is a mode rather than a takeover and its tabs are still there on the
    way back.
    """
    return docmodes.guard(ctx, "sirens", "song", "songs", verb, proceed)


def close_tab(ctx: Any, uid: str) -> None:
    """``docmodes.close_tab``; what is Sirens' is the release."""
    from . import play as sirens_play

    state = ensure(ctx)

    def release(_tab: SongTab) -> None:
        if state.active_uid == uid:
            # The buffer on this tab is what the mixer is playing; a tab closed
            # mid-bar would otherwise keep sounding with nothing on screen to
            # stop it.
            sirens_play.stop(ctx)

    docmodes.close_tab(ctx, state, uid, release)


# --- crash recovery (UX-05) ---------------------------------------------------
#
# ``packwright_mode``'s four answers, for songs. See :mod:`studio.journal`.
#
# The rendered PCM is deliberately not journalled, for the reason it is not
# saved: it is derived, and a re-render of an unchanged document is identical
# with nothing to invalidate. A recovered song re-renders itself, which is
# seconds and is the only answer that cannot be stale.


def _journal_encode(tab: Any) -> bytes:
    from .engine import rsng

    return rsng.rsng_bytes(tab.doc)


def _journal_adopt(ctx: Any, path: Path, meta: dict[str, Any]) -> bool:
    from .engine import rsng

    ensure(ctx)
    try:
        doc = rsng.read_rsng(sirens_io._within_ceiling(Path(path)))
    except Exception:
        log.exception("could not reopen the recovered song at %s", path)
        journal.adopt_failed(ctx, "song")
        return False
    title = f"{meta.get('title') or Path(path).stem} (recovered)"
    tab = adopt(ctx, doc, path=None, title=title)
    docmodes.mark_recovered(tab, path, doc)
    return True


JOURNAL = journal.tab_provider(
    "sirens", ".rsng", "song", encode=_journal_encode, adopt=_journal_adopt
)


# --- what moved out, and the one door back ------------------------------------
#
# ``sirens_edit``, ``sirens_play`` and ``sirens_keys`` -- the modules T7 split
# this one into (the 2026-09-02 review, section 8). Every name they define is
# still reachable as ``sirens_mode.<name>``, because that is what the seven
# panes, the app's key routing and most of this mode's tests name.
#
# ``inker_mode``'s mechanism exactly: a PEP 562 ``__getattr__`` over a table
# rather than imports at the bottom of this file. Each moved module imports
# *this* one as a module object, so its own attribute lookups happen at call
# time, and a bottom ``from .sirens_edit import ...`` here would fail whenever
# something imported the pair the other way round. Resolving on demand has no
# order at all -- which is also what lets the three siblings call each other
# through this name without knowing which file anything ended up in.
#
# **A name appears exactly once**, so the table is the record of where each
# thing went rather than a second place to keep in step.
_MOVED: dict[str, str] = {
    "AUDITION_PREFIX": "play",
    "ENVELOPE_FIELDS": "edit",
    "PATTERN_PREFIX": "play",
    "PIANO_KEYS": "keys",
    "PREVIEW_PREFIX": "play",
    "PREVIEW_ROWS": "play",
    "RENDER_PREFIX": "play",
    "_MUTATING_CTRL": "keys",
    "_caret_kind": "play",
    # Added by the 2026-09-13 audit, finding sirens-02: both names were
    # reachable from their own modules but missing from this table, breaking
    # the promise that every split-out name stays reachable as
    # ``sirens_mode.<name>``. The ghost test that should have caught it
    # checked only that every table entry still resolves, not that every
    # module-level name in the split modules is in the table.
    "_caret_offset": "play",
    "_column_ceiling": "edit",
    "_ctrl_key": "keys",
    "step_history": "edit",
    "_playable": "play",
    "_touch": "edit",
    "_write_at_caret": "edit",
    "adopt_sample": "edit",
    "audition": "play",
    "begin_envelope_drag": "edit",
    "clamp_caret": "edit",
    "clear_cell": "edit",
    "clear_selection": "edit",
    "copy_selection": "edit",
    "cut_selection": "edit",
    "cycle_instrument": "edit",
    "end_envelope_drag": "edit",
    "follow_playhead": "play",
    "handle_key": "keys",
    "interpolate_selection": "edit",
    "jump_row": "edit",
    "move_caret": "edit",
    "paste": "edit",
    "_piano_elsewhere": "keys",
    "play": "play",
    "play_from_caret": "play",
    "play_pattern": "play",
    "playhead_mark": "play",
    "playhead_row": "play",
    "preview_note": "play",
    "pump": "play",
    "redo": "edit",
    "release_all": "keys",
    "remove_sample": "edit",
    "request_render": "play",
    "request_rerender": "play",
    "set_caret": "edit",
    "set_sequence": "edit",
    "shift_rows": "edit",
    "stop": "play",
    "synth_rate": "play",
    "toggle_play": "play",
    "transpose": "edit",
    "undo": "edit",
    "update_channel": "edit",
    "write_cell": "edit",
    "write_effect": "edit",
    "write_hex": "edit",
    "write_note": "edit",
}


def __getattr__(name: str) -> Any:
    module = _MOVED.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __package__), name)


def __dir__() -> list[str]:
    """The moved names included, so ``dir`` and tab completion still find them."""
    return sorted({*globals(), *_MOVED})
