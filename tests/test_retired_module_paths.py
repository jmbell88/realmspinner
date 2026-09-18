"""No source or test file names a module path the 2026-09-18 restructure retired.

``dev/RESTRUCTURE.md`` moved a fixed set of modules (P0-P8, released 0.0.50).
The 2026-09-18 audit, finding docs-01, counted 188 leftover citations of the
old spellings across ``src/`` (101) and ``tests/`` (43), plus 44 more in
``dev/INVARIANTS.md`` (paid separately -- this file never reads ``dev/``,
since a public clone of this repo does not carry it).

This is the guard that keeps that count from growing back. It walks every
``.py`` file under ``src/warlock`` and ``tests``, greps each for
:data:`RETIRED`'s patterns, and fails -- one ``path:line: old -> new`` per
hit -- on anything :data:`ALLOWED` does not excuse by name. A comment that
plainly tells history (dated, past tense: "lived in X until", "used to be
Y") is not a stale citation and does not need an entry here; one that reads
as naming the *current* location does, whether or not it is correct.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "warlock"
TESTS = ROOT / "tests"

# Mode packages that used to draw as flat ``panes/<mode>_<name>.py`` files
# before P5/P6 folded them into ``studio/modes/<mode>/ui/panes/<name>.py``.
# Home, Library, Create, Review and Settings never used that prefix (P6's
# own note: "Library, Home and Settings had no ``<m>_`` prefix"), so they
# are deliberately not in this list -- adding them would make the pattern
# below match live files like ``studio/panes/library.py``.
_PREFIXED_MODES = (
    "inker|clay|mason|poser|troupe|plotter|packwright|muse|sirens"
)

# old spelling (regex) -> where it actually lives now, for the reader.
RETIRED: list[tuple[str, re.Pattern[str], str]] = [
    (
        "studio/undo.py / studio.undo",
        re.compile(r"studio[./]undo\b"),
        "warlock/core/undo.py",
    ),
    (
        "studio/atomic.py / studio.atomic",
        re.compile(r"studio[./]atomic\b"),
        "warlock/core/safeio/atomic.py",
    ),
    (
        "studio/manual/{loader,parser,targets}.py",
        re.compile(r"studio[./]manual[./](loader|parser|targets)\b"),
        "warlock/kernels/manual/{loader,parser,targets}.py",
    ),
    (
        "studio/sirens_audio.py",
        re.compile(r"studio[./]sirens_audio\b"),
        "warlock/studio/modes/sirens/audio.py",
    ),
    (
        "studio/troupe/ (the flat headless package)",
        re.compile(r"studio[./]troupe[./]"),
        "warlock/studio/modes/troupe/engine/",
    ),
    (
        "pipelines/sheet.py / pipelines.sheet (not sheetcheck, not charsheet)",
        re.compile(r"pipelines[./]sheet\b"),
        "warlock/kernels/sheet.py",
    ),
    (
        "tilegrid (bare or studio-prefixed)",
        re.compile(r"\btilegrid\b"),
        "warlock/kernels/grid2d/",
    ),
    (
        "panes/<mode>_<name>.py / panes.<mode>_<name>",
        re.compile(rf"panes[./](?:{_PREFIXED_MODES})_[a-z][a-z_]*"),
        "warlock/studio/modes/<mode>/ui/panes/<name>.py",
    ),
    (
        "tests/test_exercise_mode.py",
        re.compile(r"tests[./]test_exercise_mode\b"),
        "dev/scripts/exercise_mode.py (now a dev tool, not a pytest file)",
    ),
]

# (file relative to repo root, retired name from RETIRED's first column) ->
# one-line reason. May shrink; should not grow. Every entry here is a place
# that names the *old* spelling on purpose -- an import pin asserting the
# old module is gone, or a comment plainly phrased as history -- not a
# leftover citation.
ALLOWED: dict[tuple[str, str], str] = {
    (
        "src/warlock/studio/modes/inker/ui/panes/drag.py",
        "panes/<mode>_<name>.py / panes.<mode>_<name>",
    ): (
        "\"Lifted out of panes/inker_canvas on 2026-09-04\" -- true of that "
        "date, before this restructure."
    ),
    (
        "src/warlock/studio/modes/inker/ui/panes/gestures.py",
        "panes/<mode>_<name>.py / panes.<mode>_<name>",
    ): (
        "\"Lifted out of panes/inker_canvas on 2026-09-04\" -- true of that "
        "date, before this restructure."
    ),
    (
        "src/warlock/studio/modes/inker/ui/panes/slices.py",
        "panes/<mode>_<name>.py / panes.<mode>_<name>",
    ): (
        "\"Lifted out of panes/inker_canvas on 2026-09-04\" -- true of that "
        "date, before this restructure."
    ),
    (
        "src/warlock/studio/modes/sirens/engine/envelope.py",
        "panes/<mode>_<name>.py / panes.<mode>_<name>",
    ): (
        "\"It lived in panes/sirens_envelopes.py until 2026-09-04\" -- "
        "explicitly past tense, names the pre-restructure home."
    ),
    (
        "src/warlock/cliptransfer.py",
        "pipelines/sheet.py / pipelines.sheet (not sheetcheck, not charsheet)",
    ): "\"this module's old location, before...\" -- explicitly past tense.",
    (
        "src/warlock/kernels/mesh/regen.py",
        "panes/<mode>_<name>.py / panes.<mode>_<name>",
    ): "panes/clay_props._carry_shading is named \"now deleted\" -- not a live citation.",
    (
        "tests/modes/troupe/test_troupe_imports.py",
        "studio/troupe/ (the flat headless package)",
    ): (
        "the 2026-09-15 audit finding it quotes (troupe-04) predates this "
        "restructure and used the real names of that day."
    ),
    (
        "src/warlock/studio/modes/inker/mode.py",
        "panes/<mode>_<name>.py / panes.<mode>_<name>",
    ): (
        "docstring quotes a historical docstring's own wrong claim (\"it "
        "named a pane that does not exist\") -- the fake name is the point."
    ),
    (
        "tests/modes/settings/test_app_settings_pickers.py",
        "tests/test_exercise_mode.py",
    ): (
        "\"found by the pipelines-04 fixer's tests/test_exercise_mode.py "
        "picker-count test\" -- names the test that found the bug, by its "
        "name at the time."
    ),
    (
        "tests/test_layering.py",
        "studio/undo.py / studio.undo",
    ): (
        "both hits narrate the P3 move itself (\"studio/undo.py itself is "
        "gone -- P3 moved it to core/undo.py\", \"P3 answered studio/undo.py's "
        "layer... by moving them\") -- explicitly the old name, describing "
        "its own retirement."
    ),
    (
        "tests/test_layering.py",
        "tilegrid (bare or studio-prefixed)",
    ): (
        "a verbatim quote of dev/RESTRUCTURE.md (\"RESTRUCTURE.md's own "
        "words\"), which this fixer may not edit and which still spells it "
        "tilegrid."
    ),
    (
        "tests/test_layering.py",
        "panes/<mode>_<name>.py / panes.<mode>_<name>",
    ): (
        "\"the four pairs that used to sit here ... are gone\" -- explicitly "
        "the pre-move dotted names, past tense."
    ),
    (
        "tests/test_layering.py",
        "pipelines/sheet.py / pipelines.sheet (not sheetcheck, not charsheet)",
    ): (
        "narrates the P3/P4 move itself (\"pipelines/sheet ... dissolved "
        "when sheet.py ... moved to kernels/\") -- the old name, describing "
        "its own retirement."
    ),
    (
        "src/warlock/core/safeio/zipguard.py",
        "tilegrid (bare or studio-prefixed)",
    ): (
        "\"kernels/grid2d (né tilegrid)\" -- né means \"born\"; this names "
        "the pre-restructure name on purpose."
    ),
    (
        "tests/modes/inker/test_inker_imports.py",
        "pipelines/sheet.py / pipelines.sheet (not sheetcheck, not charsheet)",
    ): (
        "\"it used to be pipelines.sheet until P4 ... moved that module to "
        "warlock.kernels.sheet\" -- explicitly the pre-move name."
    ),
    (
        "tests/modes/inker/test_inker_imports.py",
        "tilegrid (bare or studio-prefixed)",
    ): "\"(studio/tilegrid/ before P3 ... moved it)\" -- explicitly the pre-move name.",
    (
        "tests/modes/sirens/test_sirens_imports.py",
        "tilegrid (bare or studio-prefixed)",
    ): "quotes the hard-coded tuple this test's derivation replaced, by its pre-move member name.",
    (
        "tests/modes/muse/test_muse_imports.py",
        "tilegrid (bare or studio-prefixed)",
    ): "quotes the hard-coded tuple this test's derivation replaced, by its pre-move member name.",
    (
        "tests/modes/plotter/test_plotter_imports.py",
        "studio/undo.py / studio.undo",
    ): (
        "\"(2026-09-17: moved from studio/undo.py to warlock/core/undo.py "
        "in P3...)\" -- explicitly the pre-move name."
    ),
    (
        "tests/modes/plotter/test_plotter_imports.py",
        "tilegrid (bare or studio-prefixed)",
    ): "\"(studio/tilegrid/ before the same move)\" -- explicitly the pre-move name.",
    (
        "tests/_pure_packages.py",
        "tilegrid (bare or studio-prefixed)",
    ): (
        "lists the paths P3 moved *out of* studio/ (studio/clay/, "
        "studio/inker/, ..., studio/sirens/wavout.py) -- studio/tilegrid/ "
        "belongs in that list by its old name for the same reason its "
        "neighbours do."
    ),
    (
        "tests/modes/mason/test_mason_imports.py",
        "studio/undo.py / studio.undo",
    ): (
        "\"(2026-09-17: this moved from studio/undo.py to "
        "warlock/core/undo.py in P3...)\" -- explicitly the pre-move name."
    ),
    (
        "tests/modes/clay/test_clay_imports.py",
        "studio/undo.py / studio.undo",
    ): (
        "\"P3 ... moved this module from studio/undo.py to "
        "warlock/core/undo.py\" -- explicitly the pre-move name."
    ),
    (
        "src/warlock/studio/modes/inker/state.py",
        "panes/<mode>_<name>.py / panes.<mode>_<name>",
    ): (
        "\"moved here on 2026-09-03 from panes/inker_canvas\" -- true of "
        "that 2026-09-03 move, before this restructure."
    ),
}


_SELF = Path(__file__).resolve()


def _iter_py_files() -> list[Path]:
    files: list[Path] = []
    for base in (SRC, TESTS):
        files.extend(
            p
            for p in base.rglob("*.py")
            if "__pycache__" not in p.parts and p.resolve() != _SELF
        )
    return files


def _scan() -> list[str]:
    hits: list[str] = []
    for path in _iter_py_files():
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern, new in RETIRED:
            if ALLOWED.get((rel, label)) is not None:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    hits.append(f"{rel}:{i}: {label} -> {new}")
    return hits


def test_no_source_or_test_cites_a_module_path_the_restructure_retired() -> None:
    hits = _scan()
    assert not hits, (
        f"{len(hits)} citation(s) of a module path the restructure retired "
        "(see RETIRED in this file for where each one actually lives now; "
        "add a reason to ALLOWED only for a citation that is genuinely "
        "history, not a leftover):\n" + "\n".join(hits)
    )


def test_allowed_entries_still_match_a_real_hit() -> None:
    """ALLOWED "may shrink but should not grow" -- and every entry in it
    must still be excusing something real, or it is dead weight nobody will
    notice go stale."""
    all_hits: list[str] = []
    for path in _iter_py_files():
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern, _new in RETIRED:
            if pattern.search(text):
                all_hits.append((rel, label))
    missing = [key for key in ALLOWED if key not in all_hits]
    assert not missing, f"ALLOWED entries with nothing left to excuse: {missing}"
