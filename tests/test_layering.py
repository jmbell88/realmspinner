"""Whether the tree already obeys the restructure's target layers, measured.

``dev/RESTRUCTURE.md`` names six layers and one rule -- a module may import a
module of the same or lower layer, plus a non-UI layer (0-3) may not import
anything that still lives under ``studio/`` even when that thing's own future
layer would be low enough, because until it actually moves it is still
entangled with the UI tree it sits in -- and one extra rule at the top layer:
a mode may never import a sibling mode. The point of this file is the same
one :mod:`_pure_packages` already makes for the six per-mode import pins:
the question "does the tree obey the rule" is answerable by walking the
tree, and a pin that answers it by *reading the code* survives every file
move the restructure makes, because nothing here is keyed on where a file
sits today except the one table below that says where it is *going*.

**Layer by planned destination, not by current directory.** So
``src/warlock/studio/clay/document.py`` is already layer 1 (bound for
``kernels/mesh/``), ``studio/atomic.py`` is already layer 0, and
``studio/familiar/router.py`` is already layer 3 -- each mapped by the same
table ``dev/RESTRUCTURE.md``'s "Target architecture" section gives, transcribed
once here rather than re-derived, because *where a file is going* is a plan
decision, not a fact the tree can answer by itself. What the tree answers,
and what actually gets walked with :mod:`ast`, is which mode owns a
``studio/<mode>_*.py`` / ``studio/panes/<mode>_*.py`` / ``studio/<mode>/``
file -- derived from :data:`warlock.studio.modes.KEYS`, the one authoritative
mode list, exactly the way the task that produced this file asked for, so a
fifteenth mode enrols itself in the sibling-import ban the day its files
appear rather than waiting for a hand list to notice.

**Module scope only**, matching :mod:`_pure_packages`'s own choice, and for
the same reason stated there plus one this repo already writes down twice.
``studio/familiar/contract.py`` imports ``agent_clay`` *inside a function*
specifically so the module keeps importing with no imgui/moderngl/pygame in
the process, and its own docstring names that as the reason; ``poser_mode.py``
imports ``clay_mode`` and ``troupe_mode`` the same way, inside functions, and
says so ("the ``clay_mode.py`` pattern -- state and logic here, drawing in
``main.py``"). A lazy, function-scope import is this codebase's accepted way
to reach across a boundary rarely and by name -- treating it the same as a
module-scope import would fail two patterns the code is deliberately, visibly
using as an escape hatch, not two bugs. It also means this pin cannot see
``importlib.import_module(f".{mode}_mode")`` (``status_bar.py``'s own
dynamic dispatch, named in ``dev/RESTRUCTURE.md``'s hazards list) -- a
string built at runtime is invisible to an import walk by construction, and
nothing about widening the scope to function bodies would fix that; it would
need a second, ``import_module``-call-shaped check, which this file does not
attempt.

**Violations today are real, and :data:`EXCEPTIONS` is the restructure's own
remaining work made countable rather than a waiver.** Every pair is one
edge this tree actually has right now, grouped by the ``dev/RESTRUCTURE.md``
phase that removes it (a short header comment per group, not one repeated per
pair -- the phase is a property of the group, not of each tuple), with a
handful genuinely not owned by any phase as written and marked as such rather
than pinned to a made-up one. Two tests carry the claim: violations outside
this set fail by name, and every entry in this set must still be a real edge
in the tree, so a landed phase makes the list *shrink* -- an entry that no
longer names a real import is a completed phase, not a still-open one, and the
test says so.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

from warlock.studio import modes

SRC = Path(__file__).resolve().parents[1] / "src" / "warlock"

# ---------------------------------------------------------------------------
# The target architecture (dev/RESTRUCTURE.md's own table), as data.
# ---------------------------------------------------------------------------

#: The mode set the sibling-import ban applies to, derived from the one
#: authoritative list rather than written out again. Familiar and Tour are
#: deliberately absent -- CLAUDE.md is explicit that neither is a mode, and
#: RESTRUCTURE.md's own P5 bullet keeps Familiar "a pane, not a mode... kept
#: out of modes/" on purpose.
MODE_KEYS: frozenset[str] = frozenset(modes.KEYS)

#: Files whose name does not follow the ``studio/<mode>_*.py`` /
#: ``studio/panes/<mode>_*.py`` convention but that RESTRUCTURE.md's own
#: measurement (or this file's own verification of it) attributes to one mode
#: anyway. Each comment says how it was confirmed.
STRAY_MODE_FILES: dict[str, str] = {
    # RESTRUCTURE.md's "Mode-specific code is trapped in shared places" list,
    # verified by reading each file.
    "studio/ants.py": "inker",
    "studio/colorwheel.py": "inker",
    "studio/layout_edit.py": "plotter",
    "studio/probe.py": "troupe",
    "studio/quality.py": "review",
    "studio/artifacts.py": "create",
    # Create's real 2D/3D/character recipe engine under a stale "settings"
    # name -- RESTRUCTURE.md measured this by name: "plus its real engine
    # buried in panes/settings_2d.py (2,917 lines of non-UI plan/validate/
    # kwargs logic)"; P5's Create bullet is the same file.
    "studio/panes/settings_2d.py": "create",
    "studio/panes/settings_3d.py": "create",
    "studio/panes/settings_character.py": "create",
    # The actual Settings-mode pane -- not studio/settings.py, which is the
    # persisted-JSON engine (imported by every mode, not Settings-owned).
    "studio/panes/app_settings.py": "settings",
    # CLAUDE.md: "studio/agent_clay.py is the Clay tool surface"; P5's Clay
    # bullet folds it and agent_program.py into modes/clay/agent/ together.
    "studio/agent_clay.py": "clay",
    "studio/agent_program.py": "clay",
}

CORE_TOP = frozenset({
    "config.py", "db.py", "hashes.py", "memlog.py", "vram.py", "winjob.py",
    "native.py", "instance.py", "migrate.py", "provenance.py", "leases.py",
    "progress.py", "__init__.py",
    # Not named in RESTRUCTURE.md's table at all -- both are stdlib-only and
    # explicitly compare themselves in their own docstrings to vram/memlog,
    # this tree's two canonical "pure, importable from anywhere" examples.
    "errors.py", "changelog.py",
})
CORE_SAFEIO = frozenset({
    "studio/atomic.py", "studio/sizeguard.py", "studio/zipguard.py",
    "studio/xmlguard.py", "studio/npyguard.py", "studio/pixelguard.py",
})
GEOM3D_TOP = frozenset({"glbio.py", "meshaudit.py", "meshreport.py", "tiercheck.py"})
GEOM3D_VIEWER = frozenset({
    "studio/viewer/math3d.py", "studio/viewer/gltf.py", "studio/viewer/glbwrite.py",
})
RIG_TOP = frozenset({"rigging.py", "clips.py", "clipmaps.py", "cliptransfer.py", "poselib.py"})
AUDIO_KERNEL = frozenset({"studio/sirens/wavout.py"})
WEIGHTS_TOP = frozenset({"models.py", "fetch.py", "packs.py", "publish.py"})
#: Top-level, torch-free, pure planning/writer/contract modules with no row
#: of their own in RESTRUCTURE.md's table -- grouped beside pipelines because
#: each is read by pipelines and/or service and imports at most ``models``
#: (see this file's own docstring companion, the landing report, for the
#: verification: none of the five import anything above weights).
PIPELINES_ADJACENT_TOP = frozenset({
    "guidance.py", "generation.py", "asset_workflows.py", "godotscene.py", "judge.py",
})
JOBS_TOP_EXTRA = frozenset({"queue.py", "vectors.py", "followups.py"})
SHELL_NAMED = frozenset({
    "studio/main.py", "studio/widgets.py", "studio/theme.py", "studio/tokens.py",
    "studio/controls.py", "studio/icons.py", "studio/dialogs.py", "studio/state.py",
    "studio/docmodes.py", "studio/undo.py", "studio/journal.py",
    "studio/imgui_backend.py", "studio/tasks.py",
})
VIEWPORT_NAMED = frozenset({
    "studio/_view_bounds.py", "studio/_view_cache.py", "studio/_view_drag.py",
    "studio/_view_frame.py", "studio/_view_overlay.py", "studio/_view_pick.py",
    "studio/_viewer_pose.py",
})
#: Familiar's UI half -- P5: "Familiar is a pane, not a mode -- keep it out
#: of modes/", folding these into studio/assistant/. Classified shell (L4)
#: rather than invented as its own layer for exactly that reason.
FAMILIAR_UI = frozenset({
    "studio/familiar_ui.py", "studio/familiar_preview.py", "studio/familiar_doors.py",
})
#: True CLI/entrypoint/dev-tooling -- outside the six-layer table entirely
#: (nothing in the table's rows names ``cli.py``, ``doctor.py``, ``sweep.py``,
#: the console entry point, or the bench harness). Excluded from the check in
#: both directions rather than forced into a row; see the landing report.
EXCLUDED_TOP = frozenset({"doctor.py", "sweep.py", "cli.py", "__main__.py"})


@dataclasses.dataclass(frozen=True)
class Layer:
    number: int
    kind: str
    mode: str | None = None


def classify(rel: str) -> Layer:
    """The layer *rel* (a ``/``-separated path under ``src/warlock``) is
    bound for, by :data:`dev/RESTRUCTURE.md`'s table -- today's path, not
    today's directory listing.
    """
    if rel in CORE_TOP or rel in CORE_SAFEIO:
        return Layer(0, "core")
    if rel in GEOM3D_TOP or rel in GEOM3D_VIEWER:
        return Layer(1, "kernel:geom3d")
    if rel.startswith("studio/clay/"):
        return Layer(1, "kernel:mesh")
    if rel.startswith("studio/inker/"):
        return Layer(1, "kernel:pixel")
    if rel.startswith("studio/tilegrid/"):
        return Layer(1, "kernel:grid2d")
    if rel in RIG_TOP:
        return Layer(1, "kernel:rig")
    if rel in AUDIO_KERNEL:
        return Layer(1, "kernel:audio")
    if rel in WEIGHTS_TOP:
        return Layer(2, "weights")
    if rel.startswith("pipelines/"):
        return Layer(2, "pipelines")
    if rel in PIPELINES_ADJACENT_TOP:
        return Layer(2, "pipelines-adjacent")
    if rel in JOBS_TOP_EXTRA or rel.startswith("_q_"):
        return Layer(3, "jobs")
    if rel.startswith("service/"):
        return Layer(3, "service")
    if rel.startswith("studio/familiar/"):
        return Layer(3, "familiar")
    if rel.startswith("characters/"):
        return Layer(3, "characters")
    if rel.startswith("mcp/"):
        return Layer(3, "mcp")
    if rel.startswith("studio/viewer/"):
        return Layer(4, "viewport")
    if rel in VIEWPORT_NAMED:
        return Layer(4, "viewport")
    if rel in SHELL_NAMED or rel in FAMILIAR_UI:
        return Layer(4, "shell")
    if rel in STRAY_MODE_FILES:
        return Layer(5, "mode", STRAY_MODE_FILES[rel])
    parts = rel.split("/")
    if parts[0] == "studio" and len(parts) >= 2:
        base = parts[-1]
        stem = base[:-3] if base.endswith(".py") else base
        if parts[1] == "panes" and len(parts) == 3:
            for mode in MODE_KEYS:
                if stem == mode or stem.startswith(mode + "_"):
                    return Layer(5, "mode", mode)
        if len(parts) == 2:
            for mode in MODE_KEYS:
                if stem == mode + "_mode" or stem.startswith(mode + "_"):
                    return Layer(5, "mode", mode)
        if len(parts) >= 2 and parts[1] in MODE_KEYS:
            return Layer(5, "mode", parts[1])
    # Everything else under studio/ that is neither a kernel, a named shell
    # file nor mode-prefixed defaults to shell (L4): either genuinely shared
    # chrome (state.py, dialogs.py, the design system's other files) or a
    # residual the P4/P5/P6 mode folds have not sorted yet. Not in the table
    # by name -- see the landing report for the files this catches.
    return Layer(4, "shell-default")


def is_excluded(rel: str) -> bool:
    return rel in EXCLUDED_TOP or rel.startswith("bench/")


# ---------------------------------------------------------------------------
# The import graph: module-scope edges only (see the module docstring).
# ---------------------------------------------------------------------------


def _rel_of(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def _dotted_of(rel: str) -> str:
    if rel.endswith("/__init__.py"):
        rel = rel[: -len("/__init__.py")]
    elif rel.endswith(".py"):
        rel = rel[:-3]
    return "warlock." + rel.replace("/", ".")


def _module_file(base: Path) -> Path | None:
    as_module = base.with_suffix(".py")
    if as_module.is_file():
        return as_module
    as_package = base / "__init__.py"
    if as_package.is_file():
        return as_package
    return None


def _climb(path: Path, level: int) -> Path:
    """``__package__`` is *path*'s own containing directory whether *path* is
    a plain module or a package's ``__init__.py`` -- level 1 means "relative
    to that directory" (zero climbs), each further level climbs one more.
    Same rule as ``tests/_pure_packages.py``'s own ``_relative_targets``.
    """
    base = path.parent
    for _ in range(level - 1):
        base = base.parent
    return base


def _module_scope_targets(path: Path) -> list[tuple[Path, int]]:
    """Every first-party file *path* imports at module scope, submodule-first.

    ``from ..viewer import gltf`` names the submodule ``viewer/gltf.py`` when
    that file exists; only when it does not is ``viewer/__init__.py`` itself
    the target (an attribute pulled off the package). Trying both and keeping
    both, the way an early draft of this walk did, double-counts: it made
    every one of Clay's ``from ..viewer import math3d`` imports also count as
    a dependency on ``viewer/__init__.py`` (viewport, L4) beside the real,
    harmless one on ``viewer/math3d.py`` (a geom3d kernel, L1) -- manufacturing
    a layer violation nothing in the source actually commits.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - nothing in this tree fails to parse
        return []
    out: list[tuple[Path, int]] = []
    for node in tree.body:  # module scope only
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != "warlock":
                    continue
                rest = alias.name.split(".", 1)[1] if "." in alias.name else ""
                target = SRC.joinpath(*rest.split(".")) if rest else SRC
                resolved = _module_file(target)
                if resolved is not None:
                    out.append((resolved, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if not node.module or node.module.split(".")[0] != "warlock":
                    continue
                rest = node.module.split(".", 1)[1] if "." in node.module else ""
                base = SRC.joinpath(*rest.split(".")) if rest else SRC
            else:
                base = _climb(path, node.level)
                if node.module:
                    base = base.joinpath(*node.module.split("."))
            names = node.names or []
            for alias in names:
                resolved = _module_file(base / alias.name)
                if resolved is None:
                    resolved = _module_file(base)
                if resolved is not None:
                    out.append((resolved, node.lineno))
            if not names:  # `from X import *`
                resolved = _module_file(base)
                if resolved is not None:
                    out.append((resolved, node.lineno))
    return out


@dataclasses.dataclass(frozen=True)
class Edge:
    importer: str  # dotted
    imported: str  # dotted
    file: Path
    lineno: int


def _all_edges() -> list[Edge]:
    edges: list[Edge] = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = _rel_of(path)
        if is_excluded(rel):
            continue
        importer = _dotted_of(rel)
        for target_path, lineno in _module_scope_targets(path):
            target_rel = _rel_of(target_path)
            if is_excluded(target_rel) or target_rel == rel:
                continue
            edges.append(Edge(importer, _dotted_of(target_rel), path, lineno))
    return edges


_EDGES: list[Edge] = _all_edges()
#: Every edge this session's own walk found, in dotted-pair form -- what
#: :data:`EXCEPTIONS` staleness is checked against.
_EDGE_PAIRS: frozenset[tuple[str, str]] = frozenset((e.importer, e.imported) for e in _EDGES)


def _dotted_to_rel(dotted: str) -> str:
    """The reverse of :func:`_dotted_of`, resolved through the same
    :func:`_module_file` lookup the edge walk itself uses -- a naive
    ``dotted.replace(".", "/") + ".py"`` is wrong for any package (
    ``warlock.studio.viewer`` is ``studio/viewer/__init__.py``, not
    ``studio/viewer.py``, which does not exist).
    """
    rest = dotted.removeprefix("warlock.")
    base = SRC.joinpath(*rest.split("."))
    resolved = _module_file(base)
    assert resolved is not None, f"{dotted} does not resolve to a file under {SRC}"
    return _rel_of(resolved)


def _violations() -> list[tuple[Edge, str]]:
    """Every edge in :data:`_EDGES` that breaks a layering rule, paired with
    which rule it broke (for the failure message only).
    """
    out: list[tuple[Edge, str]] = []
    seen: set[tuple[str, str]] = set()
    for edge in _EDGES:
        key = (edge.importer, edge.imported)
        if key in seen:
            continue
        seen.add(key)
        imp_rel = _dotted_to_rel(edge.importer)
        tgt_rel = _dotted_to_rel(edge.imported)
        imp = classify(imp_rel)
        tgt = classify(tgt_rel)
        if tgt.number > imp.number:
            reason = f"L{imp.number}({imp.kind}) may not import L{tgt.number}({tgt.kind})"
            out.append((edge, reason))
            continue
        if (
            imp.number <= 3
            and not imp_rel.startswith("studio/")
            and tgt_rel.startswith("studio/")
            and tgt_rel not in CORE_SAFEIO
        ):
            reason = f"L{imp.number}({imp.kind}) (non-UI) may not import anything under studio/"
            out.append((edge, reason))
            continue
        if imp.number == 5 and tgt.number == 5 and imp.mode and tgt.mode and imp.mode != tgt.mode:
            out.append((edge, f"mode {imp.mode!r} may not import sibling mode {tgt.mode!r}"))
    return out


# ---------------------------------------------------------------------------
# EXCEPTIONS -- today's real violations, grouped by the RESTRUCTURE.md phase
# that removes each group. A pair leaves this set the day its own edge is
# gone, never before -- test_exceptions_has_no_stale_entries enforces that
# nothing lingers once a phase actually lands.
# ---------------------------------------------------------------------------

# P2 -- mode manifests. `studio/main.py` and `studio/panes/landing.py` reach
# mode UI modules by name today; once each mode carries a manifest.py and
# main.py/landing.py read `modes.ALL` instead, these become indirection
# through the mode's own package (L5 importing L5's own manifest) rather than
# L4 naming L5 directly. (`journal.py`, `palette.py` and `status_bar.py` are
# RESTRUCTURE.md's other three named examples of this family; they currently
# reach mode code only through `importlib.import_module(f"...")`, which this
# AST walk cannot see -- see the module docstring.)
_P2_SHELL_DISPATCH: frozenset[tuple[str, str]] = frozenset({
    ("warlock.studio.main", "warlock.studio.clay_viewport"),
    ("warlock.studio.main", "warlock.studio.create_brief"),
    ("warlock.studio.main", "warlock.studio.mason_viewport"),
    ("warlock.studio.main", "warlock.studio.poser_viewport"),
    ("warlock.studio.main", "warlock.studio.probe"),
    ("warlock.studio.main", "warlock.studio.review_panes"),
    ("warlock.studio.panes.landing", "warlock.studio.create_stages"),
})

# P3 -- shared code moves out of studio/. Familiar's headless half (contract,
# router, retrieval, doors, character_plan) is the pilot: service/familiar.py
# and pipelines/llama_client.py depend on it today while it still lives under
# studio/, which the "non-UI layer may not import under studio/" rule catches
# regardless of Familiar's own future layer (3) being low enough on paper.
_P3_FAMILIAR_MOVES_OUT: frozenset[tuple[str, str]] = frozenset({
    ("warlock.service.familiar", "warlock.studio.familiar.character_plan"),
    ("warlock.service.familiar", "warlock.studio.familiar.contract"),
    ("warlock.service.familiar", "warlock.studio.familiar.doors"),
    ("warlock.service.familiar", "warlock.studio.familiar.retrieval"),
    ("warlock.service.familiar", "warlock.studio.familiar.router"),
    ("warlock.pipelines.llama_client", "warlock.studio.familiar.contract"),
})

# P3/P7 -- "Packwright stays a mode... the overlap was tilegrid and the
# texture caches, which P3 and P7 already fix" (RESTRUCTURE.md's own words).
# Packwright's atlas writers reuse Plotter's PNG/TSX writers directly today.
_P3_P7_PACKWRIGHT_PLOTTER_OVERLAP: frozenset[tuple[str, str]] = frozenset({
    ("warlock.studio.packwright.compose", "warlock.studio.plotter.pngio"),
    ("warlock.studio.packwright.tsxout", "warlock.studio.plotter.tsx"),
    ("warlock.studio.packwright.wpack", "warlock.studio.plotter.pngio"),
})

# P4 -- the god-file split. RESTRUCTURE.md: "widgets.py (3,423) sheds
# Create's stage rail and Review's grade buttons." controls.py's Troupe probe
# reference is the same shape, in the same design-system layer, not named as
# its own bullet but resolved the same way.
_P4_GOD_FILE_SPLIT: frozenset[tuple[str, str]] = frozenset({
    ("warlock.studio.widgets", "warlock.studio.artifacts"),
    ("warlock.studio.widgets", "warlock.studio.probe"),
    ("warlock.studio.widgets", "warlock.studio.quality"),
    ("warlock.studio.controls", "warlock.studio.probe"),
    ("warlock.studio.panes.inspector", "warlock.studio.quality"),
})

# P5 -- the pilot four. Two shapes: Familiar's UI half folding into
# studio/assistant/ (its Clay-preview and Create-doors reach), and Inker's
# `PaintView` (inker_state.py:858-1476) promoting to shell/paintview.py,
# named explicitly as "already imported by Plotter and Packwright". Create's
# own UI fold (create_brief/create_stages/generation_workspace/settings_*
# into modes/create/ui/) and the Clay agent fold (agent_clay.py/
# agent_program.py into modes/clay/agent/, taking agent_host.py's and
# agent_transcript.py's reach into agent_clay.py with them) are the other two
# pilot-four bullets, and cover the remaining pairs below.
_P5_PILOT_FOUR: frozenset[tuple[str, str]] = frozenset({
    # Familiar UI -> Clay / Create
    ("warlock.studio.familiar_preview", "warlock.studio.agent_clay"),
    ("warlock.studio.familiar_preview", "warlock.studio.clay_mode"),
    ("warlock.studio.familiar_preview", "warlock.studio.clay_state"),
    # Clay agent fold
    ("warlock.studio.agent_host", "warlock.studio.agent_clay"),
    ("warlock.studio.agent_transcript", "warlock.studio.agent_clay"),
    # Create UI fold
    ("warlock.studio.asset_exits", "warlock.studio.create_stages"),
    ("warlock.studio.generation_workspace", "warlock.studio.create_assets"),
    ("warlock.studio.panes.inspector", "warlock.studio.create_stages"),
    # Create's recipe engine lifting out of panes/settings_*.py means Settings
    # can import the engine module directly instead of a pane object.
    ("warlock.studio.panes.app_settings", "warlock.studio.panes.settings_3d"),
    # PaintView promotion (inker_state.py -> shell/paintview.py)
    ("warlock.studio.packwright_state", "warlock.studio.inker_state"),
    ("warlock.studio.panes.packwright_preview", "warlock.studio.inker_state"),
    ("warlock.studio.panes.plotter_canvas", "warlock.studio.inker_state"),
    ("warlock.studio.plotter_state", "warlock.studio.inker_state"),
})

# P6 -- the remaining modes, one agent per mode. Best-fit rather than named:
# RESTRUCTURE.md's P6 bullet does not call either edge out by name, but both
# targets are owned by a mode landing in this wave (Plotter's own tileset
# editor; Mason's own asset picker), and that landing is the plausible place
# either dependency gets resolved.
_P6_REMAINING_MODES: frozenset[tuple[str, str]] = frozenset({
    ("warlock.studio.panes.inker_tiles", "warlock.studio.plotter_tilesets"),
    ("warlock.studio.panes.mason_palette", "warlock.studio.panes.library"),
})

# P10 -- Muse folds into Create's audio stage, explicitly removing "the
# muse <-> sirens cross-import" (RESTRUCTURE.md's own words) once
# kernels/audio/ (P3) holds what the two shared.
_P10_MUSE_FOLDS_INTO_CREATE: frozenset[tuple[str, str]] = frozenset({
    ("warlock.studio.muse_mode", "warlock.studio.sirens_audio"),
    ("warlock.studio.muse_mode", "warlock.studio.sirens_io"),
    ("warlock.studio.muse_mode", "warlock.studio.sirens_mode"),
    ("warlock.studio.muse_mode", "warlock.studio.sirens_state"),
    ("warlock.studio.panes.muse_player", "warlock.studio.sirens_audio"),
    ("warlock.studio.panes.muse_results", "warlock.studio.sirens_audio"),
})

# P11 -- Review folds into a Library view; P12 -- Home folds into Library's
# empty state. `panes/candidates_panel.py` and `panes/library.py` reaching
# into both Review and Create today are exactly the seam P9-P12 close.
_P11_P12_LIBRARY_ABSORBS: frozenset[tuple[str, str]] = frozenset({
    ("warlock.studio.panes.candidates_panel", "warlock.studio.panes.library"),
    ("warlock.studio.panes.candidates_panel", "warlock.studio.review_mode"),
    ("warlock.studio.panes.library", "warlock.studio.review_mode"),
    ("warlock.studio.panes.library", "warlock.studio.artifacts"),
})

# Not owned by any phase as dev/RESTRUCTURE.md is written today -- real,
# current edges the plan does not yet say who fixes. Kept in EXCEPTIONS
# (removing them would just make the suite red for a gap in the plan, not in
# the code) and named here for whoever picks the plan back up. See the P1
# landing report for the case each one earns.
_UNRESOLVED: frozenset[tuple[str, str]] = frozenset({
    # studio/undo.py is the layer table's own contradiction: its own
    # docstring says it was extracted "for" Clay precisely so a pure engine
    # would not depend on the raster editor for history, and
    # tests/clay/test_clay_imports.py's OUTWARD_IMPORTS already excepts it on
    # exactly that reasoning. Moving Inker's and Clay's engines to
    # kernels/pixel and kernels/mesh (P3) does not resolve this edge -- an L1
    # kernel would still import an L4 shell module the day P3 lands. Whoever
    # runs P3 needs to also decide undo.py's real layer (core, beside
    # vram/memlog, is the shape its own docstring already argues for).
    ("warlock.studio.clay.document", "warlock.studio.undo"),
    ("warlock.studio.clay.edits", "warlock.studio.undo"),
    ("warlock.studio.inker.anim_edits", "warlock.studio.undo"),
    ("warlock.studio.inker.tile_edits", "warlock.studio.undo"),
    ("warlock.studio.inker.undo", "warlock.studio.undo"),
    # studio/manual/ has no row in RESTRUCTURE.md's table at all. Familiar's
    # retrieval already depends on manual.loader/parser (pure text chunking,
    # no imgui) at module scope; manual.render.py is the only piece that
    # actually draws. The same kernel/viewport split the plan already applies
    # to viewer/ would resolve this, but no phase proposes it yet.
    ("warlock.studio.familiar.retrieval", "warlock.studio.manual.loader"),
    ("warlock.studio.familiar.retrieval", "warlock.studio.manual.parser"),
    # config.py (core) importing models.DEFAULT_BASE_MODEL (weights) for one
    # constant; clips.py (kernels/rig) and inker/sheetout.py (kernels/pixel)
    # each importing a pipelines/ writer directly. None of the three is a
    # "shared code trapped in studio/" case RESTRUCTURE.md's P3 describes --
    # they are core/kernels reaching into weights/pipelines outright -- and
    # no phase names any of them.
    ("warlock.config", "warlock.models"),
    ("warlock.clips", "warlock.pipelines.charsheet"),
    ("warlock.clips", "warlock.pipelines.sheet"),
    ("warlock.studio.inker.sheetout", "warlock.pipelines.sheet"),
    # mason -> clay, in the direction Mason's own code says is banned:
    # clay_ops.py's `_align` docstring states "Mason may not import Clay (its
    # own import pin says so, and for a real reason)" while arguing the
    # *reverse* direction is fine -- but tests/mason/test_mason_imports.py
    # only guards the headless studio/mason/ engine package (`pure_packages`
    # is scoped to directories, and panes/ is excluded from it everywhere),
    # so studio/panes/mason_palette.py reaching into Clay's own tool-palette
    # pane is invisible to every existing pin. Not named by any phase.
    ("warlock.studio.panes.mason_palette", "warlock.studio.panes.clay_tools"),
    # Create referencing Review's mode module directly for its own settings
    # panes -- plausibly resolved once Create's engine (P5) has its own
    # verdict/grade vocabulary to import instead, but P5's bullet does not
    # say so, so this is not filed under it.
    ("warlock.studio.panes.settings_2d", "warlock.studio.review_mode"),
    ("warlock.studio.panes.settings_3d", "warlock.studio.review_mode"),
})

EXCEPTIONS: frozenset[tuple[str, str]] = (
    _P2_SHELL_DISPATCH
    | _P3_FAMILIAR_MOVES_OUT
    | _P3_P7_PACKWRIGHT_PLOTTER_OVERLAP
    | _P4_GOD_FILE_SPLIT
    | _P5_PILOT_FOUR
    | _P6_REMAINING_MODES
    | _P10_MUSE_FOLDS_INTO_CREATE
    | _P11_P12_LIBRARY_ABSORBS
    | _UNRESOLVED
)


# ---------------------------------------------------------------------------
# The two tests.
# ---------------------------------------------------------------------------


def test_no_undocumented_layering_violations() -> None:
    """Every layering violation in the tree today is named in EXCEPTIONS.

    A new violation not in that set means either a real regression (an import
    added that should not have been) or a real decision (add it to
    EXCEPTIONS, under the phase that removes it, or to ``_UNRESOLVED`` with a
    reason if no phase does).
    """
    offenders = [
        (edge, reason)
        for edge, reason in _violations()
        if (edge.importer, edge.imported) not in EXCEPTIONS
    ]
    if not offenders:
        return
    lines = [
        f"{edge.file}:{edge.lineno}: {edge.importer} -> {edge.imported} ({reason})"
        for edge, reason in offenders
    ]
    raise AssertionError(
        f"{len(offenders)} layering violation(s) not in tests/test_layering.py's "
        "EXCEPTIONS -- add each to the group for the dev/RESTRUCTURE.md phase "
        "that removes it (or to _UNRESOLVED with a reason):\n" + "\n".join(lines)
    )


def test_exceptions_has_no_stale_entries() -> None:
    """Every pair in EXCEPTIONS is still a real edge in the tree.

    A pair that stops matching a real import is a phase that already landed
    -- delete the entry rather than leave it; a shrinking EXCEPTIONS is the
    restructure's own progress made visible in a diff.
    """
    stale = sorted(pair for pair in EXCEPTIONS if pair not in _EDGE_PAIRS)
    if not stale:
        return
    lines = [f"{imp} -> {tgt}" for imp, tgt in stale]
    raise AssertionError(
        f"{len(stale)} stale EXCEPTIONS entry(ies) in tests/test_layering.py -- "
        "the import is gone, so delete the entry (the phase that removed it "
        "already landed):\n" + "\n".join(lines)
    )
