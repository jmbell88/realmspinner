"""Per-document-mode facts, the one place each mode states them once.

Five places used to discover a document mode by hand-maintained table, one
module name at a time: ``journal._PROVIDER_MODULES`` (which modules to import
so crash recovery can see every kind), ``palette._DOC_MODES`` (module name
and export label, keyed by mode), ``panes/landing.KIND_OPENERS`` and
``_KIND_MODES`` (which module reopens a recent row of a kind, and which mode
it switches to), and ``main.teardown``'s hand-called ``_persist_inker`` /
``_persist_clay`` / ``_persist_mason`` / ``_persist_plotter`` /
``_persist_packwright``. The first four tables can each be exactly
reconstructed from one list of facts below; the fifth -- teardown -- stops
needing a table at all once "does this module define ``persist``" is *asked*
of the module instead of copied into a fifth place to forget. That fifth
place is exactly where the bug this phase exists to fix was living:
``sirens_mode.persist`` had no caller anywhere, although its own docstring
says it is called after every open and save, because ``main.teardown``
hand-listed five of the six modules that define ``persist`` and never grew a
sixth line.

**A sibling of :mod:`.modes`, not a section inside it.** ``modes.py``'s own
docstring reserves that module for the rail's tuple -- key, label, icon,
purpose -- and says a table of anything else there would be a second place
deciding something, which is exactly what this is not: it is data about the
*document* modes, a strict subset of ``modes.KEYS`` (Home, Library, Create,
Troupe, Review, Muse and Settings own no document and are not here); it needs
:data:`.verbs.EXPORT_TO_LIBRARY`, a second import ``modes.py`` has no other
reason to carry; and the plan this phase works from (``dev/RESTRUCTURE.md``,
P2) puts a mode's manifest inside that mode's own future package -- a shape a
sibling list of dataclasses splits into seven files with no change to what
each one declares, where a second section of ``modes.py`` would have to be
carved out of the one table that module is not allowed to grow.

**Module names stay strings, never imported here at module scope** -- the
same discipline ``journal.py`` already keeps, and for the same reason: it
keeps a session that never opens Clay from paying for the mesh engine. Every
lookup below that needs to ask a module something (:func:`persisting_modes`)
imports lazily, at the call site, exactly as ``journal.ensure_providers`` and
``palette._doc_mode`` already do -- and only :mod:`.verbs` (itself
stdlib-only past :mod:`.modes` and :mod:`.icons`) is imported at module scope
here, so this file stays as cheap to import as ``journal.py`` requires.

**Declared vs. derived.** ``key``, ``module``, ``kind``, ``export_label`` and
``opener`` are declared below, because none of them is a fact a module can be
asked for without importing it first -- they are what a name and a label
*are*, not something the code computes. Whether a module defines ``persist``
or exposes a ``JOURNAL`` is asked with ``getattr`` instead, wherever a table
used to assert it by copying the answer down: that is a fact about the code,
and a table that restates it is a table that can disagree with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from . import verbs

#: Every document-mode module lives flat under this package today. Isolating
#: the prefix here is what changes, and only here, the day these modules fold
#: into ``studio.modes.<name>`` (RESTRUCTURE.md's own future shape for this
#: file: one manifest per mode package).
_PACKAGE = "warlock.studio"


@dataclass(frozen=True)
class ModeManifest:
    """One document mode's facts, against the four tables that used to
    hand-list them. See the module docstring for declared-vs-derived."""

    #: The mode key, one of ``modes.KEYS``.
    key: str
    #: The module (flat under :data:`_PACKAGE`) that owns ``active``,
    #: ``JOURNAL`` and (for six of the seven) ``persist``.
    module: str
    #: The journal/document kind string. Equal to ``key`` for six of the
    #: seven; Poser's is ``"pose"`` -- a pose is authored in Poser, but
    #: nothing else in the app ever calls the *document* a "poser".
    kind: str
    #: ``palette._DOC_MODES``'s export command label, character for
    #: character. Empty suppresses the command instead of forcing a fake one:
    #: Poser's document is a library record, saved and never exported.
    export_label: str
    #: The module whose ``open_path`` reopens a recent row of this kind, or
    #: ``None`` when this kind has no recent-row opener at all (Poser: a pose
    #: begins by rigging a library asset already there, never by reopening a
    #: path, so it is not offered through Recents at all). Usually equal to
    #: ``module`` -- Sirens is the one exception, opened by ``sirens_io``
    #: rather than ``sirens_mode``.
    opener: str | None


#: One row per document mode, in the order ``journal._PROVIDER_MODULES`` used
#: to list them. Every function below rebuilds exactly the old table it
#: replaces; see each one's docstring for which shape that is.
DOC_MODES: tuple[ModeManifest, ...] = (
    ModeManifest("inker", "modes.inker.mode", "inker", "Export PNG", "modes.inker.mode"),
    ModeManifest("clay", "modes.clay.mode", "clay", verbs.EXPORT_TO_LIBRARY, "modes.clay.mode"),
    ModeManifest(
        "mason", "modes.mason.mode", "mason", "Export .glb + manifest", "modes.mason.mode"
    ),
    ModeManifest("plotter", "plotter_mode", "plotter", "Export .tmx", "plotter_mode"),
    ModeManifest(
        "packwright",
        "modes.packwright.mode",
        "packwright",
        "Export atlas + JSON",
        "modes.packwright.mode",
    ),
    # Named for the folder rather than for a file: this is the one export in
    # the app that writes a family (song.wav, stems/, sfx/) into a directory
    # the user picks, so "Export WAV" would describe a third of what happens.
    # Its recent rows are reopened by ``fileio``, not ``mode``.
    ModeManifest(
        "sirens", "modes.sirens.mode", "sirens", "Export WAV + stems", "modes.sirens.fileio"
    ),
    # An empty export label suppresses the command (see the field's own
    # docstring); no opener, for the same reason a pose has no New command.
    ModeManifest("poser", "modes.poser.mode", "pose", "", None),
)


def by_key(key: str) -> ModeManifest | None:
    return next((m for m in DOC_MODES if m.key == key), None)


def journal_modules() -> tuple[str, ...]:
    """``journal._PROVIDER_MODULES``'s shape: every module to import so
    crash recovery can see every registered kind."""
    return tuple(m.module for m in DOC_MODES)


def export_table() -> dict[str, tuple[str, str]]:
    """``palette._DOC_MODES``'s shape: mode -> (module, export label)."""
    return {m.key: (m.module, m.export_label) for m in DOC_MODES}


def opener_table() -> dict[str, str]:
    """``landing.KIND_OPENERS``'s shape: recent-doc kind -> opener module.
    Poser is left out -- see :attr:`ModeManifest.opener`."""
    return {m.kind: m.opener for m in DOC_MODES if m.opener is not None}


def kind_mode_table() -> dict[str, str]:
    """``landing._KIND_MODES``'s shape: journal kind -> the mode a recovered
    (or reopened) row of it should switch to."""
    return {m.kind: m.key for m in DOC_MODES}


def _import(module_name: str) -> Any:
    return import_module(f"{_PACKAGE}.{module_name}")


def module_of(entry: ModeManifest) -> Any:
    """The module that owns one mode's document -- ``active``, ``persist`` and
    its ``JOURNAL``.

    The public door onto :func:`_import`, for the call sites that used to spell
    a module out of the mode key (``status_bar`` built ``f".{mode}_mode"``).
    That spelling was only ever true while every mode module sat directly under
    ``studio/``, and it was already false for Sirens in the one table that
    names openers rather than document modules.
    """
    return _import(entry.module)


def persisting_modes() -> tuple[ModeManifest, ...]:
    """Doc modes whose module defines ``persist`` -- asked, never declared.

    This is the fix the module docstring names: whether a module defines
    ``persist`` is a fact the module can answer for itself, through
    ``getattr``, and a table that copies the answer into a sixth place is a
    table that can (and did) forget one.
    """
    return tuple(
        m for m in DOC_MODES if callable(getattr(_import(m.module), "persist", None))
    )


def call_persist(ctx: Any, entry: ModeManifest) -> None:
    """Call one mode's ``persist``. The module is already imported (by
    :func:`persisting_modes`, or by ordinary use earlier this session);
    importing an already-imported name is a cache hit, not a fresh load."""
    _import(entry.module).persist(ctx)
