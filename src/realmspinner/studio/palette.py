"""The command palette's data half: what the commands are, and what matches.

Pure in the way :mod:`~realmspinner.vram` and :mod:`~realmspinner.memlog` are -- stdlib
only, no imgui, no moderngl, no pygame -- so every rule about *which* command a
query finds is assertable without a window. The drawing half is
:mod:`.panes.palette`, and it owns no list of its own.

Two things here are deliberate and easy to undo.

**The mode commands are derived from** :data:`.modes.MODES`, not written out.
A palette is a second index of everything the app can do, and a hand-written
one is a second index that drifts -- a fourteenth mode would gain a switch
segment and be missing from the one surface whose entire job is telling the
user what exists. It is also, since the positional Alt+digit bindings went
away, the *only* keyboard route to a mode, which is what turns the derivation
from tidiness into a requirement.
For the same reason what a mode command *does* is :func:`.state.set_mode` and
not four lines of its own: that copy existed, and it had already drifted.

**Every command carries ``enabled``, and a disabled command is *listed*.** The
tempting version filters the list to what can run right now, which makes the
palette a worse menu: a user searching for "wireframe" from Home learns
nothing from an empty result, and learns where to go from a greyed row that
says which mode it belongs to. The one thing a disabled row may not do is run.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import mode_manifest, modes, state
from .tour import scripts as tour_scripts

# How many assets the quick-open section offers at once. A palette is a
# shortlist: past a handful the eye is scanning rather than recognising, and
# the list is the newest N of M anyway, so a longer one would still not be
# "every asset".
MAX_ASSETS = 8


@dataclass(frozen=True)
class Command:
    """One thing the palette can do.

    ``key`` is stable and is what a test names; ``label`` is what the user
    reads and may be reworded freely. ``group`` is a heading, ``hint`` is the
    keyboard shortcut.

    ``why`` is the reason the row is greyed, and it is the half of the module
    docstring's argument that was never built: "learns where to go from a
    greyed row that says which mode it belongs to" was the whole case for
    listing disabled commands, and the row said nothing -- it greyed out and
    Enter on it returned in silence. A command with ``enabled`` and no ``why``
    is the gap, not the norm.
    """

    key: str
    label: str
    group: str
    run: Callable[[Any], None]
    hint: str = ""
    enabled: Callable[[Any], bool] = field(default=lambda _ctx: True)
    why: str = ""


# --- matching ----------------------------------------------------------------


def match(query: str, text: str) -> int | None:
    """Score ``text`` against ``query``, or ``None`` for no match.

    A subsequence match, case-insensitive: "wf" finds "Toggle wireframe" and
    "gtc" finds "Go to Clay". Higher is better. The scoring is deliberately
    crude and deliberately *stated*, because the alternative -- ranking by
    whatever order the commands happen to be built in -- makes the first row a
    property of this file's layout rather than of what was typed, and the first
    row is the one Enter runs.

    Three things earn points, in the order they matter: a prefix match on the
    whole string, a match that starts at a word boundary, and adjacency between
    consecutive matched characters. A shorter haystack breaks ties, so "Clay"
    outranks "Open in Clay" for the query "clay".
    """
    if not query:
        return 0
    needle = query.casefold()
    hay = text.casefold()
    if needle in hay:
        # A contiguous hit always beats a scattered one, and a prefix beats
        # anything inside the string.
        base = 1000 if hay.startswith(needle) else 700
        if hay.startswith(needle) or hay[hay.index(needle) - 1] in " -/:":
            base += 100
        return base - len(hay)
    score = 0
    position = 0
    previous = -2
    for char in needle:
        found = hay.find(char, position)
        if found < 0:
            return None
        if found == 0 or hay[found - 1] in " -/:":
            score += 20
        if found == previous + 1:
            score += 10
        previous = found
        position = found + 1
        score += 1
    return score - len(hay)


def rank(query: str, items: list[Any], text_of: Callable[[Any], str]) -> list[Any]:
    """Everything that matches, best first, stable within a score.

    Stable because the input order is meaningful -- commands arrive grouped,
    assets arrive newest first -- and an unstable sort would reshuffle equal
    rows as the user types a character that changes nothing.
    """
    scored = []
    for index, item in enumerate(items):
        value = match(query, text_of(item))
        if value is not None:
            scored.append((-value, index, item))
    scored.sort(key=lambda row: (row[0], row[1]))
    return [item for _score, _index, item in scored]


# --- the commands ------------------------------------------------------------


def _go(key: str) -> Callable[[Any], None]:
    """A mode command, through the app's one mode switch.

    It used to be a second copy of ``App._set_mode``'s four lines, and the two
    had already drifted: this one had no early return, so choosing the mode you
    were already in recorded ``previous_mode == mode`` and the next Esc from a
    pass-through mode fell back to Home instead of to the work you left.
    """

    def run(ctx: Any) -> None:
        state.set_mode(ctx.state, key)

    return run


def _generate(ctx: Any) -> None:
    from .modes.create.ui import stages as create_stages
    from .modes.create.ui.panes import settings_2d, settings_3d

    if create_stages.at(ctx.state, "reference"):
        settings_2d.generate(ctx, ctx.state.form_2d)
    elif create_stages.at(ctx.state, "mesh"):
        settings_3d.promote(ctx, ctx.cache.get(ctx.state.source_job), ctx.state.form_3d)


def _in_generate_mode(ctx: Any) -> bool:
    """Whether Create is at a stage that has a Generate of its own.

    The same two stages ``shell.events``' Ctrl+Enter acts on. This used to be
    "Create, any stage", so the palette and File menu rows (which print that
    chord as their hint) opened the cutout dialog for a fresh mesh job from
    Rig, Pose and Export while the chord itself did nothing there -- the
    2026-10-03 audit, shell-17.
    """
    from .modes.create.ui import stages as create_stages

    return create_stages.at(ctx.state, "reference") or create_stages.at(ctx.state, "mesh")


def _viewport(ctx: Any) -> bool:
    return ctx.state.mode in modes.VIEWPORT_MODES


# Shared by the four viewport commands: one sentence, so they cannot drift into
# four different accounts of the same gate. Named after ``modes.VIEWPORT_MODES``
# itself rather than "the 2D and 3D panes" (the 2026-09-26 audit, finding
# shell-chrome-08): Create stopped being a two-pane 2D/3D split well before
# this sentence was last touched, and ``VIEWPORT_MODES`` has held only
# ``{"create"}`` since -- a greyed reason naming panes that no longer exist
# teaches nothing about where to go instead.
_VIEWPORT_WHY = "Only in Create, which is where the viewport is."


def _selected(ctx: Any) -> Any:
    """The library row ``ctx.state.selected`` names, or ``None``.

    Tolerant of a ``ctx`` with no ``cache`` at all -- ``commands()`` now
    calls this unconditionally while *building* the Reroll row's ``why``
    (finding shell-09's fix), and an enumerator that only wants the
    navigation-shaped rows -- ``familiar_doors.destinations``, which never
    even looks at Reroll's ``why`` -- used to pay nothing for the library
    cache and should still pay nothing for it. A ``ctx`` missing ``cache``
    reads the same as one with an empty cache: nothing selected.
    """
    cache = getattr(ctx, "cache", None)
    if cache is None:
        return None
    return cache.get(getattr(ctx.state, "selected", None))


def _any_trashed(ctx: Any) -> bool:
    """Whether there is anything for "Empty the trash..." to empty.

    Prefers ``cache.trash_present``, the store-wide answer the job-list read
    task took (``JobsCache.read``), over ``ctx.cache.jobs``, which is only the
    newest ``jobs_cache.LIST_LIMIT`` (200) rows -- a trash entirely older than
    that window used to read as empty here while the store still held rows to
    delete (shell-chrome-06, the 2026-09-26 audit).

    **Never asks the store.** The first fix called ``store.trashed()`` -- a
    full ``SELECT *`` that decodes every trashed row under the store's lock --
    from this gate, and the gate runs for every command row on every frame a
    menu root or the palette is open: 5.5 ms a call at 500 trashed rows, 35 ms
    at 2000 (the 2026-10-03 audit, shell-18). The figure is one frame stale at
    worst, and ``invalidate`` -- which every trash, restore and empty calls --
    re-reads it.

    Falls back to the ``cache.jobs`` scan until a read has landed, and for a
    ``ctx`` with no cache at all: ``specs()`` builds every command's
    ``enabled`` on every call it makes (menus and the palette both), including
    from callers that hand it a ctx built for a narrower question. The same
    tolerance :func:`_selected` above already has, for the same reason.
    """
    cache = getattr(ctx, "cache", None)
    present = getattr(cache, "trash_present", None)
    if present is not None:
        return bool(present)
    jobs = getattr(cache, "jobs", None) or ()
    return any(job.get("deleted_at") for job in jobs)


# --- the document modes, as one table ----------------------------------------
#
# Save, Save as, Export and Undo/Redo are four commands whose whole content is
# "whichever document mode is in front". Written out per mode they would be
# sixteen commands, four of which would be forgotten the next time a mode is
# added; written as a dispatch they are four, and the dispatch is one line per
# mode.
#
# The entry points are the modes' *own*: ``inker_mode.save`` and the rest are
# what the bridge buttons and the Ctrl+S handlers already call, so the palette
# cannot drift into a second way of saving. Undo goes to the document's own
# ``undo()``, which is uid-addressed inside -- the palette never touches a
# history directly.
#
# Derived from :mod:`.mode_manifest` -- mode -> (module name, export command
# label) -- rather than written out here a second time: that is the one
# manifest field this table needed, character for character (Sirens' label
# names a folder because it is the one export in the app that writes a family
# -- song.wav, stems/, sfx/ -- into a directory the user picks; Poser's empty
# label suppresses the command rather than forcing a fake one, since a pose
# is a library record, saved and never exported -- and its character-sheet
# stage, folded in from Troupe in P9 (2026-09-18), changes nothing here: a
# sheet is a selection over files a worker already published, the same "no
# document" reason the empty label already covers).
_DOC_MODES: dict[str, tuple[str, str]] = mode_manifest.export_table()

_DOC_WHY = "Open a drawing, model, map, atlas or song first."


def _doc_mode(ctx: Any):
    """``(module, tab)`` for the document mode in front, or ``(None, None)``."""
    from importlib import import_module

    entry = _DOC_MODES.get(ctx.state.mode)
    if entry is None:
        return None, None
    module = import_module(f".{entry[0]}", __package__)
    return module, module.active(ctx)


def _has_doc(ctx: Any) -> bool:
    _module, tab = _doc_mode(ctx)
    return tab is not None


def _doc_save(ctx: Any) -> None:
    module, tab = _doc_mode(ctx)
    if tab is not None:
        module.save(ctx, tab)


def _doc_save_as(ctx: Any) -> None:
    module, tab = _doc_mode(ctx)
    if tab is not None:
        module.save_as(ctx, tab)


def _doc_export(ctx: Any) -> None:
    module, tab = _doc_mode(ctx)
    if tab is None:
        return
    # Each mode's *file* export -- the one its bridge puts first. The others
    # stay panel-only: a palette listing four exports per mode is a list
    # nobody reads. **Its chord is Ctrl+Shift+E, not Ctrl+E**, in every mode
    # but Clay: Ctrl+E is the library export everywhere it exists (Inker's is
    # "Save as reference"), and Clay's file export *is* its library export.
    # This comment said Ctrl+E for all of them until 2026-09-05, and
    # ``_doc_export_hint`` is what keeps the row's printed chord honest.
    if not _doc_export_label(ctx):
        return
    if ctx.state.mode == "inker":
        module.export_png(ctx, tab)
    elif ctx.state.mode == "clay":
        module.export_asset(ctx, tab)
    elif ctx.state.mode == "mason":
        module.export_glb(ctx, tab)
    elif ctx.state.mode == "plotter":
        module.export_map(ctx, "tmx", tab)
    else:
        module.export_files(ctx, tab)


def _doc_export_label(ctx: Any) -> str:
    entry = _DOC_MODES.get(ctx.state.mode)
    return entry[1] if entry else "Export"


#: Modes whose ``handle_key`` binds the file export to plain Ctrl+E rather
#: than Ctrl+Shift+E, because the file export *is* the mode's library export
#: (see the comment on ``_doc_export`` above). Mason joined Clay here in the
#: 2026-09-26 audit, finding shell-chrome-03: ``mason.mode._ctrl_key`` binds
#: ``export_glb`` to ``"e" and not shift``, but this table still printed
#: "Ctrl+Shift+E" -- a chord that did nothing -- while the working one went
#: unmentioned.
_EXPORT_CTRL_E_MODES = {"clay", "mason"}


def _doc_export_hint(ctx: Any) -> str:
    """The chord the mode actually binds the file export to; see ``_doc_export``."""
    return "Ctrl+E" if ctx.state.mode in _EXPORT_CTRL_E_MODES else "Ctrl+Shift+E"


def _can_export(ctx: Any) -> bool:
    """Whether this mode has an export at all. See :data:`_DOC_MODES`."""
    return bool(_doc_export_label(ctx)) and _has_doc(ctx)


def _doc_history(ctx: Any):
    """The active document's undo stack, or None."""
    _module, tab = _doc_mode(ctx)
    doc = getattr(tab, "doc", None)
    return getattr(doc, "history", None)


def _can_undo(ctx: Any) -> bool:
    history = _doc_history(ctx)
    return bool(history is not None and history.can_undo)


def _can_redo(ctx: Any) -> bool:
    history = _doc_history(ctx)
    return bool(history is not None and history.can_redo)


def _step(ctx: Any, name: str) -> None:
    """Undo or redo through the *mode*, never straight at the document.

    ``tab.doc.undo()`` is not the whole verb anywhere it has a bookkeeping
    twin. Plotter's ``undo`` also clears ``state.selected_object``, and without
    it Ctrl+K -> Undo could retire the object a form was still drawing against
    -- a dead uid the panel keeps rendering, from the one surface that reaches
    every mode. ``getattr`` rather than a required attribute: a mode with no
    such wrapper is answered by the document directly, which is what this
    always did and is correct for the modes that need nothing more.
    """
    module, tab = _doc_mode(ctx)
    if tab is None or getattr(tab, "doc", None) is None:
        return
    through = getattr(module, name, None)
    if callable(through):
        through(ctx, tab)
        return
    getattr(tab.doc, name)()


def _doc_undo(ctx: Any) -> None:
    _step(ctx, "undo")


def _doc_redo(ctx: Any) -> None:
    _step(ctx, "redo")


def _mode_commands(ctx: Any) -> list[Command]:
    """One per switch segment, in the switch's order.

    Derived rather than listed -- see the module docstring. No hint: there is
    no per-mode key to advertise any more, and a hint naming Ctrl+K would be
    the palette telling you how to open the palette you are looking at.

    **Consults ``model_gate.mode_gate``, the way the rail does.** A gated
    mode's "Go to" row used to be drawn ``enabled`` with an empty ``why``: it
    matched ``state.set_mode``'s own silent refusal on Enter -- no toast, no
    explanation -- on exactly the fresh-install case ``model_gate`` exists
    for, while the rail's matching item greyed out and routed to Settings.
    The 2026-09-07 audit found the two surfaces disagreeing (shell-05). Wiring
    ``enabled``/``why`` here is also the whole fix for the toast: the shared
    ``_run`` in ``panes/palette.py`` already turns any disabled command's
    ``why`` into a warning toast on Enter, so a correct ``why`` is all this
    needed.

    **An ungated mode keeps the dataclass default ``enabled``**, rather than
    an equivalent always-true lambda of its own --
    ``test_every_command_with_a_gate_says_why_it_is_shut`` tells the two apart
    by identity against that default, so a fresh lambda here for every mode,
    gated or not, would make every ungated "Go to" command look like a silent
    gate with no ``why`` to show.
    """
    from .panes import model_gate

    out = []
    for key, label, _icon, _purpose in modes.MODES:
        where, _blocked = model_gate.mode_gate(ctx, key)
        if not where:
            out.append(
                Command(key=f"go:{key}", label=f"Go to {label}", group="Go to", run=_go(key))
            )
            continue
        out.append(
            Command(
                key=f"go:{key}",
                label=f"Go to {label}",
                group="Go to",
                run=_go(key),
                enabled=lambda _ctx: False,
                why=model_gate.mode_reason(ctx, key),
            )
        )
    return out


# --- Clay's commands ---------------------------------------------------------
#
# A mode cannot hand the palette commands of its own, and Clay needs to: its
# menu strip went away (Landing 11 of the Clay UI rebuild) and every tool on it
# has to stay one search from the user ("Extrude", "Mirror", "Box Unwrap",
# "Import Mesh..."), plus the view, mode, tool and export controls that live on its
# toolbar and rail. So the palette asks Clay
# for them here, **only in the Clay workspace and only with a document open**,
# and every row is a thin door onto what the mode already does:
#
# * an op's row is ``clay_mode.fire_op`` -- the event layer's door, which opens the
#   parameter popup next frame for an op with parameters and toasts a refusal;
# * a view, mode or export row calls what the header's popover and Export menu
#   call (``ui/panes/header.py``), so the two surfaces cannot drift;
# * every ``enabled`` and ``run`` resolves the *live* tab through ``ctx``, never a
#   captured document, so a cached row cannot act on a document that was since
#   closed.
#
# **Cached.** ``commands()`` runs every frame -- the palette's rows and the menu
# bar both ask for it -- and this block is ~65 ``Command`` objects with a
# reason apiece. The list is rebuilt only when the question it answers changes:
# a different tab or document, a ``rev`` bump (every edit, undo and selection
# change touches it), the element mode, the selection, or a save starting or
# ending. ``enabled`` stays a lambda over the live document, so what a row
# *does* is never stale; only the ``why`` string and the row set are cached.
#
# Function-scope imports only: ``studio/modes/clay`` is a sibling the palette
# sits above, and ``tests/test_layering.py`` reads module scope.

_CLAY_GROUP = "Clay"
# Said when an op is greyed by a gate that has no sentence of its own, so a row
# is never shut in silence (``test_every_command_with_a_gate_says_why_it_is_shut``).
_CLAY_GENERIC_WHY = "Not available for the current selection."


class _ClayCache:
    """The last Clay command list and the question it answered."""

    key: tuple[Any, ...] | None = None
    doc_ref: Any = None
    commands: list[Command] = []
    #: How many times the list was actually built, for the test that pins the cache.
    builds: int = 0


_clay_cache = _ClayCache()


def _clay_tab(ctx: Any) -> Any:
    """The Clay tab in front, or ``None`` outside Clay or with no document."""
    if getattr(ctx.state, "mode", None) != "clay":
        return None
    clay = getattr(ctx.state, "clay", None)
    return None if clay is None else clay.active


def _clay_run(act: Callable[[Any, Any, Any], None]) -> Callable[[Any], None]:
    """A ``run`` that resolves the live tab, then calls ``act(ctx, state, tab)``."""

    def run(ctx: Any) -> None:
        tab = _clay_tab(ctx)
        if tab is not None:
            act(ctx, ctx.state.clay, tab)

    return run


def _clay_has_view(ctx: Any) -> bool:
    return getattr(ctx, "clay_view", None) is not None


#: The 2026-10-09 audit's clay-13/14: a live G/R/S, gizmo or UV drag owns the
#: document until it commits; a row that edits or exports it waits.
_CLAY_GESTURE_WHY = "Finish the drag in progress first (Enter or click to commit, Esc to cancel)."


def _clay_op_command(op: Any, doc: Any, tab: Any, gesture: bool = False) -> Command:
    from .modes.clay import mode as clay_mode
    from .modes.clay import ops as clay_ops
    from .modes.clay.ui.panes import header as clay_header

    if tab.saving:
        why = clay_header.SAVING_WHY
    elif gesture:
        why = _CLAY_GESTURE_WHY
    elif doc.element_mode not in op.modes:
        # Listed, not hidden: a user searching "Extrude" from Shape mode learns
        # which mode it lives in from the greyed row.
        why = clay_ops.wrong_mode_reason(op)
    else:
        why = clay_ops.reason_for(op, doc) or _CLAY_GENERIC_WHY

    def enabled(ctx: Any) -> bool:
        live = _clay_tab(ctx)
        if live is None or live.saving or clay_mode._gesture_live(ctx, live):
            return False
        return live.doc.element_mode in op.modes and bool(op.enabled(live.doc))

    def run(ctx: Any) -> None:
        live = _clay_tab(ctx)
        if live is not None:
            clay_mode.fire_op(ctx, live.doc, op)

    return Command(
        key=f"clay-op:{op.name}",
        label=op.label,
        group=_CLAY_GROUP,
        run=run,
        hint=op.key,
        enabled=enabled,
        why=why,
    )


def _clay_build(doc: Any, tab: Any, gesture: bool = False) -> list[Command]:
    from . import verbs
    from .modes.clay import doc_io
    from .modes.clay import mode as clay_mode
    from .modes.clay import ops as clay_ops
    from .modes.clay import state as clay_state
    from .modes.clay.ui.panes import header as clay_header
    from .modes.clay.ui.panes import rail as clay_rail
    from .modes.clay.ui.panes import tools as clay_tools

    out: list[Command] = []

    # The registry's ops. "Frame Selection" is left to the view row below, which
    # uses the header's own door; two rows for one camera move would be noise.
    for op in clay_ops.OPS:
        if op.name != "frame":
            out.append(_clay_op_command(op, doc, tab, gesture))

    # The four tools: the rail's words, the registry's keys and shortcuts.
    for key, label, shortcut in clay_state.TOOLS:

        def set_tool(_ctx: Any, state: Any, _tab: Any, key: str = key) -> None:
            state.tool = key

        out.append(
            Command(
                key=f"clay-tool:{key}",
                label=f"{clay_rail.TOOL_WORDS.get(key, label)} tool",
                group=_CLAY_GROUP,
                run=_clay_run(set_tool),
                hint=shortcut,
            )
        )

    # The four modes, through the door the header's switch uses.
    def can_switch(ctx: Any) -> bool:
        live = _clay_tab(ctx)
        return live is not None and not live.saving

    for mode, label, key in clay_tools.MODE_BUTTONS:

        def set_mode(_ctx: Any, _state: Any, live: Any, mode: str = mode) -> None:
            if not live.saving:
                if mode == "pose":
                    live.doc.set_pose_mode(True)
                else:
                    live.doc.set_element_mode(mode)

        out.append(
            Command(
                key=f"clay-mode:{mode}",
                label=f"{label} mode",
                group=_CLAY_GROUP,
                run=_clay_run(set_mode),
                hint=key,
                enabled=can_switch,
                why=clay_header.SAVING_WHY,
            )
        )

    # Shading: stored names (solid / material / wireframe), the header's words.
    for key, label, _short, _tip in clay_header.SHADING:

        def set_shading(_ctx: Any, state: Any, _tab: Any, key: str = key) -> None:
            state.shading = key

        out.append(
            Command(
                key=f"clay-view:shading-{key}",
                label=f"{label} shading",
                group=_CLAY_GROUP,
                run=_clay_run(set_shading),
            )
        )

    def toggle_xray(_ctx: Any, state: Any, _tab: Any) -> None:
        state.xray = not state.xray

    out.append(
        Command(
            key="clay-view:xray",
            label="Toggle X-ray",
            group=_CLAY_GROUP,
            run=_clay_run(toggle_xray),
            hint="Alt+Z",
        )
    )

    # The overlays, off the header popover's own table.
    for key, label, _tip in clay_header.OVERLAY_ROWS:

        def toggle_overlay(ctx: Any, state: Any, _tab: Any, key: str = key) -> None:
            clay_header.set_overlay(state, key, not clay_header.overlay_value(state, key))
            clay_mode.persist(ctx)

        out.append(
            Command(
                key=f"clay-view:{key}",
                label=f"Toggle {label.lower()}",
                group=_CLAY_GROUP,
                run=_clay_run(toggle_overlay),
            )
        )

    # The camera. These need the live viewport, which is absent on the first frame.
    def toggle_ortho(ctx: Any, _state: Any, _tab: Any) -> None:
        view = getattr(ctx, "clay_view", None)
        if view is not None:
            clay_header.toggle_orthographic(view)

    out.append(
        Command(
            key="clay-view:ortho",
            label="Toggle orthographic",
            group=_CLAY_GROUP,
            run=_clay_run(toggle_ortho),
            hint="Ctrl+5",
            enabled=_clay_has_view,
            why=clay_header.NO_VIEW_WHY,
        )
    )
    for name, label, chord in clay_header.AXIS_ROWS:

        def look(ctx: Any, _state: Any, _tab: Any, name: str = name) -> None:
            view = getattr(ctx, "clay_view", None)
            if view is not None:
                clay_header.look_along(view, name)

        out.append(
            Command(
                key=f"clay-view:axis-{name}",
                label=f"{label} view",
                group=_CLAY_GROUP,
                run=_clay_run(look),
                hint=chord,
                enabled=_clay_has_view,
                why=clay_header.NO_VIEW_WHY,
            )
        )

    def frame(_ctx: Any, state: Any, _tab: Any) -> None:
        clay_header.request_frame(state)

    out.append(
        Command(
            key="clay-view:frame",
            # Fifteen characters, the same as the Create-only "Frame the model":
            # ``rank`` breaks a tie toward the earlier row, and Clay's rows are
            # built first.
            label="Frame selection",
            group=_CLAY_GROUP,
            run=_clay_run(frame),
            hint="F",
        )
    )

    # The way in. Import Mesh... was a row of the Add menu in the menu strip; the
    # strip is gone, and this is one of its three homes (the Inspector's Model &
    # history page and File > Import Mesh... are the others). It opens the file as a
    # new document tab, so it needs the viewport's tab to exist (this block only
    # does) and no more.
    def import_mesh(ctx: Any, _state: Any, _live: Any) -> None:
        clay_mode.ask_import_mesh(ctx)

    out.append(
        Command(
            key="clay-file:import-mesh",
            label="Import Mesh...",
            group=_CLAY_GROUP,
            run=_clay_run(import_mesh),
            enabled=can_switch,
            why=clay_header.SAVING_WHY,
        )
    )

    # The ways out, greyed with the one sentence every output button uses.
    def can_export(ctx: Any) -> bool:
        live = _clay_tab(ctx)
        return (
            live is not None
            and not clay_mode._gesture_live(ctx, live)
            and not doc_io.outputs_why(live.doc, bool(live.saving))
        )

    # The one function the buttons use, so a row and a button greyed by the same
    # state say the same sentence (saving outranks the drag, as it does there).
    export_why = doc_io.outputs_why(doc, bool(tab.saving), gesture) or "Nothing to send yet."

    def export_library(ctx: Any, _state: Any, live: Any) -> None:
        clay_mode.export_asset(ctx, live)

    def export_glb(ctx: Any, _state: Any, live: Any) -> None:
        clay_mode.export_mesh_file(ctx, live, "glb")

    def export_obj(ctx: Any, _state: Any, live: Any) -> None:
        clay_mode.export_mesh_file(ctx, live, "obj")

    def screenshot(ctx: Any, _state: Any, live: Any) -> None:
        clay_mode.save_screenshot(ctx, live)

    for key, label, act, hint in (
        ("library", verbs.EXPORT_TO_LIBRARY, export_library, "Ctrl+E"),
        ("glb", "Export GLB...", export_glb, ""),
        ("obj", "Export OBJ...", export_obj, ""),
        ("screenshot", "Save screenshot...", screenshot, ""),
    ):
        out.append(
            Command(
                key=f"clay-export:{key}",
                label=label,
                group=_CLAY_GROUP,
                run=_clay_run(act),
                hint=hint,
                enabled=can_export,
                why=export_why,
            )
        )
    return out


def _clay_commands(ctx: Any) -> list[Command]:
    """Clay's commands for the document in front, or ``[]``. See the block comment."""
    tab = _clay_tab(ctx)
    if tab is None:
        return []
    doc = tab.doc
    from .modes.clay import mode as clay_mode

    gesture = clay_mode._gesture_live(ctx, tab)
    key = (
        tab.uid,
        doc.rev,
        doc.element_mode,
        tuple(sorted(doc.selection)),
        bool(tab.saving),
        # The greyed rows' reason differs while a drag is live, so the memo must
        # not hand the idle build to a frame that is mid-drag.
        gesture,
    )
    cache = _clay_cache
    held = cache.doc_ref() if cache.doc_ref is not None else None
    if cache.key == key and held is doc:
        return cache.commands
    cache.commands = _clay_build(doc, tab, gesture)
    cache.key = key
    cache.doc_ref = weakref.ref(doc)
    cache.builds += 1
    return cache.commands


def commands(ctx: Any) -> list[Command]:
    """Every command, in a sensible reading order.

    Takes ``ctx`` so a future command can be present only where it exists at
    all (as the rig actions are, on a host with no bpy) -- as opposed to being
    merely *disabled*, which is the default and is what almost everything here
    wants.
    """
    from ..service._jobs_resubmit import rerollable_reason as _reroll_why
    from .modes.library.ui.panes import library

    def reroll(ctx: Any) -> None:
        from ..service import jobs as svc_jobs

        job = _selected(ctx)
        if job is not None:
            ctx.submit(
                f"rerun:{job['id']}", svc_jobs.rerun_job, ctx.svc, job["id"], mode="reroll"
            )

    def rerollable(ctx: Any) -> bool:
        # ``service.jobs.rerollable``, the one spelling. This and the context
        # menu's own copy (panes/library.py) were the two that disagreed.
        from ..service import jobs as svc_jobs

        job = _selected(ctx)
        return bool(job and svc_jobs.rerollable(job))

    def delete(ctx: Any) -> None:
        # The 2026-09-26 audit's shell-chrome-01: this used to open a confirm
        # titled "Delete this asset?" claiming the job is "removed from disk",
        # when ``library.delete_asset`` only ever trashes it -- the confirm's
        # own text was wrong, and it contradicted this same gesture's other
        # door (``shell/events.py``'s Delete key on a library row), whose
        # comment states the house rule: "delete-to-trash is confirm-free here
        # because the trash *is* the confirmation." Manual Chapter 36 (The
        # trash): "Nothing is removed from disk and no question is asked --
        # the trash *is* the question." No confirm here either, now -- the
        # same call the keyboard shortcut already makes.
        job = _selected(ctx)
        if job is None:
            return
        library.delete_asset(ctx, job["id"])

    def new_drawing(ctx: Any) -> None:
        from .modes.inker import mode as inker_mode

        inker_mode.new_document(ctx, 1024, 1024)

    def new_clay(ctx: Any) -> None:
        from .modes.clay import mode as clay_mode

        clay_mode.new_document(ctx)

    def new_mason_scene(ctx: Any) -> None:
        from .modes.mason import mode as mason_mode

        mason_mode.new_document(ctx)

    def new_atlas(ctx: Any) -> None:
        from .modes.packwright import mode as packwright_mode

        packwright_mode.new_document(ctx)

    def new_song(ctx: Any) -> None:
        from .modes.sirens import mode as sirens_mode

        sirens_mode.new_document(ctx)

    def wireframe(ctx: Any) -> None:
        ctx.state.wireframe = not ctx.state.wireframe
        if ctx.viewer is not None:
            ctx.viewer.set_wireframe(ctx.state.wireframe)

    def turntable(ctx: Any) -> None:
        ctx.state.turntable = not ctx.state.turntable
        if ctx.viewer is not None:
            ctx.viewer.set_turntable(ctx.state.turntable)

    def frame(ctx: Any) -> None:
        if ctx.viewer is not None:
            ctx.viewer.frame()

    def clear_viewport(ctx: Any) -> None:
        # Through the App's helper rather than ``viewer.clear()``, because
        # emptying the viewer is only half of it -- the pin against
        # ``_sync_viewer`` is the other half, and a palette command that cleared
        # the canvas for one frame would look broken rather than useful.
        if ctx.clear_viewport is not None:
            ctx.clear_viewport()

    def fps(ctx: Any) -> None:
        ctx.state.show_fps = not ctx.state.show_fps

    def new_map(ctx: Any) -> None:
        from .modes.plotter import mode as plotter_mode

        plotter_mode.ask_new_document(ctx)

    def quit_app(ctx: Any) -> None:
        # Through the app's own guard, never straight to the exit: it is the
        # chain that asks about unsaved pixels, geometry and a pose in turn,
        # and a palette entry that bypassed it would be the one way out of the
        # app that loses work.
        ask = getattr(ctx, "ask_quit", None)
        if ask is not None:
            ask()

    def open_log(ctx: Any) -> None:
        ctx.open_log()

    def show_trash(ctx: Any) -> None:
        ctx.state.filters.trash = not ctx.state.filters.trash
        state.set_mode(ctx.state, "library")

    def empty_trash(ctx: Any) -> None:
        from ..service import jobs as svc_jobs
        from . import dialogs

        ctx.confirms.ask(
            dialogs.Confirm(
                title="Empty the trash?",
                message="Every trashed asset and everything derived from it is "
                "removed from disk. This cannot be undone.",
                confirm_label="Empty",
                cancel_label="Cancel",
                on_confirm=lambda: ctx.submit("empty-trash", svc_jobs.empty_trash, ctx.svc),
            )
        )

    def manual(ctx: Any) -> None:
        # The Manual stopped being a mode in the UI redesign, wave 3, so
        # ``_mode_commands`` no longer derives an entry for it -- and a
        # reference reachable only by a function key is a reference most people
        # do not know exists. This is that door, kept in the palette where
        # every other whole-app action is.
        from .manual import render as manual_render

        manual_render.open_at(ctx, (ctx.state.manual.chapter, None))

    def _tour(key: str) -> Any:
        # One closure per tour rather than one command taking an argument: the
        # palette matches on a *label*, and "Take the tour" for two different
        # tours would be one row the query could not tell apart.
        def run(ctx: Any) -> None:
            from .panes import tour as tour_pane

            tour_pane.start(ctx, key)

        return run

    def workspace_layout(ctx: Any) -> None:
        # The switcher, through the same one-shot: the rail's footer has no
        # room for a tenth item (its own test catches the one that pushes the
        # column out of the window at the resize floor), so the palette and
        # Settings are the two doors.
        from . import rail

        rail.request("layouts")

    def rearrange_panes(ctx: Any) -> None:
        from . import layout_edit

        layout_edit.toggle(ctx.state, ctx)

    def _has_skeleton(ctx: Any) -> bool:
        from . import skeletons

        return bool(skeletons.for_mode(ctx, ctx.state.mode))

    def shortcuts(ctx: Any) -> None:
        # A flag rather than a call, because a palette command cannot open an
        # imgui popup: the palette's own window is closing on this frame, and
        # ``open_popup`` names a popup registered inside whatever window is
        # current. The draw site consumes it, which is also what lets Ctrl+/
        # and this command be the same door.
        ctx.state.shortcuts_requested = True

    return [
        *_mode_commands(ctx),
        # Before the generic rows on purpose: a tie in ``rank`` goes to the
        # earlier command, and Clay's "Frame selection" must not lose to the
        # Create-only "Frame the model" the workspace cannot run.
        *_clay_commands(ctx),
        Command(
            key="generate",
            label="Generate / Make 3D",
            group="Actions",
            run=_generate,
            hint="Ctrl+Enter",
            enabled=_in_generate_mode,
            why="Go to Create's Reference or Mesh stage first.",
        ),
        Command(key="new-drawing", label="New drawing", group="Actions", run=new_drawing),
        Command(key="new-clay", label="New Clay document", group="Actions", run=new_clay),
        # The 2026-09-15 audit, finding shell-03: Mason is a document mode --
        # it is in ``_DOC_MODES`` above, so Save/Export/Undo/Redo all work on
        # it -- but its New had no command at all, which is the "reads as the
        # mode not having one" failure the new-map comment below already names
        # for a different mode.
        Command(
            key="new-mason-scene", label="New Mason scene", group="Actions", run=new_mason_scene
        ),
        Command(key="new-atlas", label="New atlas", group="Actions", run=new_atlas),
        # The fourth of the four, missing since Plotter shipped: three of the
        # document modes could be started from here and the map editor could
        # not, which reads as Plotter not having a New at all.
        Command(key="new-map", label="New map", group="Actions", run=new_map),
        Command(key="new-song", label="New song", group="Actions", run=new_song),
        Command(
            key="reroll",
            label="Reroll the selected asset",
            group="Actions",
            run=reroll,
            enabled=rerollable,
            # The 2026-09-15 audit, finding shell-09: this was a fixed string,
            # true only when nothing is selected -- a finished, non-rerollable
            # asset (built, a stem split, a LoRA run, a hand-made reference)
            # read the same "select a finished asset" sentence a fresh install
            # with an empty library would, which is not why the row is grey.
            # Computed from the *outer* ``ctx``, exactly as ``model_gate.mode_reason``
            # is above -- ``why`` is a value fixed at list-build time, not a
            # second callable alongside ``enabled``.
            why=_reroll_why(_selected(ctx)),
        ),
        Command(
            key="delete",
            # No ellipsis: the 2026-10-03 audit's shell-55. Chapter 20 says a
            # label ending in one opens a dialog, and this command trashes at
            # once (see ``delete`` above -- the trash is the confirmation).
            label="Delete the selected asset",
            group="Actions",
            run=delete,
            enabled=lambda ctx: _selected(ctx) is not None,
            why="Select an asset in the library first.",
        ),
        Command(
            key="wireframe",
            label="Toggle wireframe",
            group="Viewport",
            run=wireframe,
            hint="W",
            enabled=_viewport,
            why=_VIEWPORT_WHY,
        ),
        Command(
            key="turntable",
            label="Toggle turntable",
            group="Viewport",
            run=turntable,
            hint="S",
            enabled=_viewport,
            why=_VIEWPORT_WHY,
        ),
        Command(
            key="frame",
            label="Frame the model",
            group="Viewport",
            run=frame,
            hint="F",
            enabled=_viewport,
            why=_VIEWPORT_WHY,
        ),
        Command(
            key="clear-viewport",
            label="Clear the viewport",
            group="Viewport",
            run=clear_viewport,
            enabled=_viewport,
            why=_VIEWPORT_WHY,
        ),
        Command(
            key="fps",
            label="Toggle the frame-rate readout",
            group="Viewport",
            run=fps,
            hint="F10",
        ),
        # --- the document commands, whichever document mode is in front -----
        Command(
            key="save",
            label="Save",
            group="Document",
            run=_doc_save,
            hint="Ctrl+S",
            enabled=_has_doc,
            why=_DOC_WHY,
        ),
        Command(
            key="save-as",
            label="Save as...",
            group="Document",
            run=_doc_save_as,
            hint="Ctrl+Shift+S",
            enabled=_has_doc,
            why=_DOC_WHY,
        ),
        Command(
            key="export",
            # Labelled for the mode it will act on, because "Export" alone in a
            # searchable list is a command whose result the user cannot predict.
            label=_doc_export_label(ctx),
            group="Document",
            run=_doc_export,
            hint=_doc_export_hint(ctx),
            enabled=_can_export,
            why=_DOC_WHY
            if _doc_export_label(ctx)
            else "A pose is saved into the library rather than exported.",
        ),
        Command(
            key="undo",
            label="Undo",
            group="Document",
            run=_doc_undo,
            hint="Ctrl+Z",
            enabled=_can_undo,
            why="Nothing to undo in the document in front.",
        ),
        Command(
            key="redo",
            label="Redo",
            group="Document",
            run=_doc_redo,
            hint="Ctrl+Shift+Z",
            enabled=_can_redo,
            why="Nothing to redo: this is the newest step.",
        ),
        # --- the application commands ---------------------------------------
        Command(
            key="manual",
            label="Open the manual",
            group="Application",
            run=manual,
            hint="F1",
        ),
        *[
            Command(
                key=f"tour:{one.key}",
                label=f"Take the tour: {one.title}",
                group="Application",
                run=_tour(one.key),
            )
            for one in tour_scripts.TOURS
        ],
        Command(
            key="shortcuts",
            label="Keyboard shortcuts",
            group="Application",
            run=shortcuts,
            hint="Ctrl+/",
        ),
        Command(
            key="workspace-layout",
            label="Workspace layout...",
            group="Application",
            run=workspace_layout,
        ),
        Command(
            key="rearrange-panes",
            label="Rearrange panes",
            group="Application",
            run=rearrange_panes,
            hint="Shift+W",
            enabled=_has_skeleton,
            why="This workspace's panes cannot be rearranged.",
        ),
        Command(
            key="show-trash",
            label="Show the trash",
            group="Application",
            run=show_trash,
        ),
        Command(
            key="empty-trash",
            label="Empty the trash...",
            group="Application",
            run=empty_trash,
            enabled=_any_trashed,
            why="The trash is empty.",
        ),
        Command(key="open-log", label="Open the log", group="Application", run=open_log),
        Command(key="quit", label="Quit", group="Application", run=quit_app),
    ]


def asset_label(job: Any) -> str:
    """What quick-open matches against and shows. The id is in there because
    it is what a log line, a toast and a bug report all name."""
    name = job.get("name") or job.get("prompt") or ""
    return f"{name} {job['id']}".strip()


def assets(ctx: Any, query: str) -> list[Any]:
    """The quick-open half: assets whose name, prompt or id matches.

    Empty for an empty query -- a palette that lists the newest eight assets
    before a key is pressed pushes the commands off the screen, and the library
    is right there.

    Searches ``ctx.cache.jobs`` -- the newest ``jobs_cache.LIST_LIMIT`` (200)
    rows, not the whole store -- so an asset older than that window will not
    turn up here even though the Library's own search can still find it
    (shell-chrome-09, the 2026-09-26 audit: the manual's "searches your
    assets" reads as unqualified, which this window is not). Returned to the
    orchestrator as a manual wording fix rather than changed here: widening
    this to a store-wide search is the async ``request_widen``/``SEARCH_KEY``
    machinery ``jobs_cache.py`` already built for the Library's own search,
    wired into a palette call that is synchronous today -- a bigger change
    than a Low-tier wording mismatch calls for.
    """
    if not query.strip():
        return []
    # Trashed assets are excluded here rather than by ``Filters``: quick-open
    # does not go through the library's filter bar at all, so without this the
    # one surface that searches by name would be the one place a deleted asset
    # still turns up.
    live = [job for job in ctx.cache.jobs if not job.get("deleted_at")]
    return rank(query, live, asset_label)[:MAX_ASSETS]
