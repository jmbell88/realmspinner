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

**Layer by planned destination, not by current directory -- and that trick is
now half-spent.** P3 landed 2026-09-17: ``core/safeio/``, ``core/undo.py``,
``kernels/{mesh,pixel,grid2d,geom3d,audio}/`` and ``kernels/manual/
{loader,parser,targets}.py`` are today's real paths, not a plan destination
any more, so :func:`classify` matches them directly (a plain prefix or a
literal set of the files that actually moved) instead of guessing from a
``studio/...`` name that no longer exists on disk. ``warlock/familiar/``
(the pilot) is the same story: ``familiar/router.py`` is layer 3 because
``familiar/`` *is* where it lives now. What is left to move under this
trick -- still classified by planned destination because the file has not
moved yet -- is everything P4 onward names: the god-file splits, the
per-mode folds (P5/P6), Muse into Create (P10), Review/Home into Library
(P11/P12), plus the three still-undecided core/kernels edges in
``_UNRESOLVED`` below. What the tree answers, and what actually gets walked
with :mod:`ast`, is which mode owns a ``studio/<mode>_*.py`` /
``studio/panes/<mode>_*.py`` / ``studio/<mode>/`` file -- derived from
:data:`warlock.studio.modes.KEYS`, the one authoritative mode list, exactly
the way the task that produced this file asked for, so a fifteenth mode
enrols itself in the sibling-import ban the day its files appear rather than
waiting for a hand list to notice.

**Module scope only**, matching :mod:`_pure_packages`'s own choice, and for
the same reason stated there plus one this repo already writes down twice.
``familiar/contract.py`` imports ``agent_clay`` *inside a function*
specifically so the module keeps importing with no imgui/moderngl/pygame in
the process, and its own docstring names that as the reason; ``poser_mode.py``
imports ``clay_mode`` and ``troupe_mode`` the same way, inside functions, and
says so ("the ``studio/modes/clay/mode.py`` pattern -- state and logic here, drawing in
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
    # verified by reading each file. ``ants.py`` and ``colorwheel.py`` left
    # this table in P5 for ``modes/inker/ui/``.
    "studio/layout_edit.py": "plotter",
    # Three files RESTRUCTURE.md's list named as mode-owned and this file
    # copied over, all three wrong -- checked by reading them, P4,
    # 2026-09-17. Each is now left to the L4 "shell-default" fall-through,
    # and the edges that used to be violations because of the
    # misattribution are gone rather than waived:
    #
    # * ``studio/probe.py`` is not Troupe's. Its own docstring: a per-frame
    #   census of everything ``controls._finish_item`` saw, with the rect a
    #   driver can click -- the half that makes a control *addressable* to
    #   ``/exercise-mode``. Shell instrumentation. Nothing under Troupe
    #   imports it (the one mention in ``troupe_mode.py`` is a comment
    #   citing ``probe.record``'s reasoning), and its two importers are
    #   ``widgets`` and ``controls``, i.e. the design system itself.
    # * ``studio/quality.py`` is not Review's. Its docstring states the
    #   opposite outright: one wording for what a mesh measurement is
    #   allowed to say, read by *three* surfaces (the quality badge, the
    #   inspector's remesh line, Review's mesh lines) precisely because
    #   each had spelled the caveat itself -- one of them by importing
    #   imgui-bearing ``widgets`` from inside a per-frame function.
    # * ``studio/artifacts.py`` is not Create's. It is the headless table
    #   of what a finished stage can hand the user, and it has three
    #   readers across two modes and the shell: ``create_stages`` (every
    #   frame, which is why it was lifted out of ``widgets`` on
    #   2026-09-03), ``panes/library.py`` and ``panes/inspector.py``.
    #   Job-artifact vocabulary, the same shape as ``quality.py``.
    # ``panes/settings_{2d,3d,character}.py`` sat here until P5 moved them to
    # ``studio/modes/create/ui/``, where the path itself says Create.
    # The actual Settings-mode pane -- not studio/settings.py, which is the
    # persisted-JSON engine (imported by every mode, not Settings-owned).
    "studio/panes/app_settings.py": "settings",
    # ``agent_clay.py`` (and its five P4 siblings) and ``agent_program.py``
    # sat here until P5 moved them to ``studio/modes/clay/agent/``, where the
    # path itself says Clay.
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
#: P3 landed: ``core/safeio/*`` and ``core/undo.py`` are today's paths, not a
#: plan destination -- so this is a plain prefix now rather than a hand list
#: of the pre-move ``studio/{atomic,sizeguard,...}.py`` names.
CORE_NESTED_PREFIX = "core/"
GEOM3D_TOP = frozenset({"meshaudit.py", "meshreport.py", "tiercheck.py"})
#: P3 landed: ``viewer/{math3d,gltf,glbwrite}.py`` and ``glbio.py`` are both
#: at ``kernels/geom3d/`` today.
GEOM3D_PREFIX = "kernels/geom3d/"
#: P4 landed: ``rigging.py`` is gone, split into ``kernels/rig/{templates,
#: cliplib,skeleton,poses,store,blender_spec}.py`` -- so this is a prefix
#: now, the way GEOM3D_PREFIX and AUDIO_PREFIX already are. Leaving the bare
#: ``"rigging.py"`` string here would not have been a *stale* entry the
#: second test catches: it would simply have stopped matching, and every
#: file under ``kernels/rig/`` would have fallen through to the L4
#: "shell-default" default -- a kernel silently reclassified as UI, which is
#: the one direction this pin exists to refuse. The four top-level names
#: stay until the wave that moves them.
RIG_PREFIX = "kernels/rig/"
RIG_TOP = frozenset({"clips.py", "clipmaps.py", "cliptransfer.py", "poselib.py"})
#: P3 landed: ``sirens/wavout.py`` is at ``kernels/audio/wavout.py`` today.
AUDIO_PREFIX = "kernels/audio/"
#: P3 landed: ``manual/{loader,parser,targets}.py`` are at ``kernels/manual/``
#: today -- ``render.py`` (the only piece that draws) stays behind at L4, per
#: RESTRUCTURE.md's own table, so this is a prefix on the three files that
#: actually moved rather than the whole former ``studio/manual/`` directory.
MANUAL_KERNEL = frozenset({
    "kernels/manual/loader.py", "kernels/manual/parser.py", "kernels/manual/targets.py",
})
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
    "studio/docmodes.py", "studio/journal.py",
    "studio/imgui_backend.py", "studio/tasks.py",
    # studio/undo.py itself is gone -- P3 moved it to core/undo.py (see
    # CORE_NESTED_PREFIX above); it is not renamed to a shell file here.
})
VIEWPORT_NAMED = frozenset({
    # The leaf every document viewport shares. Its five ``_view_*`` siblings
    # key on Clay's objects and moved to ``modes/clay/ui/`` in P5.
    "studio/_view_frame.py",
    "studio/_viewer_pose.py",
})
#: Familiar's UI half -- P5 landed: "Familiar is a pane, not a mode -- keep it
#: out of modes/", so ``studio/assistant/`` is shell (L4) by prefix rather
#: than invented as its own layer.
FAMILIAR_UI_PREFIX = "studio/assistant/"
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
    if rel in CORE_TOP or rel.startswith(CORE_NESTED_PREFIX):
        return Layer(0, "core")
    if rel in GEOM3D_TOP or rel.startswith(GEOM3D_PREFIX):
        return Layer(1, "kernel:geom3d")
    if rel.startswith("kernels/mesh/"):
        return Layer(1, "kernel:mesh")
    if rel.startswith("kernels/pixel/"):
        return Layer(1, "kernel:pixel")
    if rel.startswith("kernels/grid2d/"):
        return Layer(1, "kernel:grid2d")
    if rel in RIG_TOP or rel.startswith(RIG_PREFIX):
        return Layer(1, "kernel:rig")
    if rel.startswith(AUDIO_PREFIX):
        return Layer(1, "kernel:audio")
    if rel in MANUAL_KERNEL:
        return Layer(1, "kernel:manual")
    if rel.startswith("kernels/"):
        # Anything else under kernels/, named or not. P4 wave two added the
        # first *flat* modules there (``kernels/sheet.py``,
        # ``kernels/charsheet.py`` -- the layer-1 table sanctions flat
        # modules, ``palettes.py`` is one), and with only the per-package
        # prefixes above they fell through to the L4 "shell-default" at the
        # bottom of this function: a kernel classified as UI, silently, which
        # is the one direction this pin exists to refuse. A catch-all rather
        # than another prefix constant, so the next kernel -- package or
        # module -- is born at the right layer instead of waiting for someone
        # to notice it was not.
        return Layer(1, "kernel")
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
    if rel.startswith("familiar/"):
        return Layer(3, "familiar")
    if rel.startswith("characters/"):
        return Layer(3, "characters")
    if rel.startswith("mcp/"):
        return Layer(3, "mcp")
    if rel.startswith("studio/viewer/"):
        return Layer(4, "viewport")
    if rel in VIEWPORT_NAMED:
        return Layer(4, "viewport")
    if rel in SHELL_NAMED or rel.startswith(FAMILIAR_UI_PREFIX):
        return Layer(4, "shell")
    if rel in STRAY_MODE_FILES:
        return Layer(5, "mode", STRAY_MODE_FILES[rel])
    parts = rel.split("/")
    # P5: a mode's own package. ``studio/modes/__init__.py`` is the mode list
    # itself -- imported by the shell and every mode alike, and importing
    # nothing but ``icons`` -- so it is shell; everything *under* a mode's
    # directory is that mode's, whatever its file is called. Without this the
    # first file to move in (``modes/create/ui/brief.py``, no ``create_``
    # prefix left to match) fell through to the L4 default below, and every
    # shell -> Create edge the fold was meant to leave countable became
    # invisible instead.
    if parts[:2] == ["studio", "modes"]:
        if len(parts) >= 4 and parts[2] in MODE_KEYS:
            return Layer(5, "mode", parts[2])
        return Layer(4, "shell")
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
            # No carve-out for core/safeio/* here any more: P3 actually moved
            # those files out of studio/, so tgt_rel no longer starts with
            # "studio/" for them at all -- the old exemption (for when they
            # were physically under studio/ but logically core) is dead code
            # now, not a rule this check still needs.
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
    # P4 wave two moved this dispatch one file over without changing its
    # nature: ``main.py`` (5,971 lines) is the process entry now, and the
    # ``App`` class is ``shell/app.py``, assembled from fourteen mixins. The
    # five ``main -> *`` pairs that used to sit here are gone because the
    # imports are gone from that file, not because the relationship ended --
    # it is ``shell.app`` naming each mode's pane mixin now, and six of the
    # ten are *new* edges rather than moved ones, because those workspaces
    # were inline methods on ``App`` before the split and an inline method is
    # invisible to an import walk. Which is the honest reading: the coupling
    # was always there, and making it an import is what made it countable.
    ("warlock.studio.shell.app", "warlock.studio.modes.clay.ui.viewport"),
    ("warlock.studio.shell.app", "warlock.studio.modes.inker.ui.workspace"),
    ("warlock.studio.shell.app", "warlock.studio.modes.mason.ui.viewport"),
    ("warlock.studio.shell.app", "warlock.studio.modes.muse.ui.workspace"),
    ("warlock.studio.shell.app", "warlock.studio.modes.packwright.ui.workspace"),
    ("warlock.studio.shell.app", "warlock.studio.modes.plotter.ui.workspace"),
    ("warlock.studio.shell.app", "warlock.studio.modes.poser.ui.viewport"),
    ("warlock.studio.shell.app", "warlock.studio.modes.review.ui.workspace"),
    ("warlock.studio.shell.app", "warlock.studio.modes.sirens.ui.workspace"),
    ("warlock.studio.shell.app", "warlock.studio.modes.troupe.ui.workspace"),
    ("warlock.studio.shell.frame", "warlock.studio.modes.create.ui.brief"),
    # No ``create_rail`` row, though P4 moved Create's stage rail out of
    # ``widgets.py`` and ``_stage_rail`` (now in ``shell/frame.py``) calls
    # it: that import is function-scope, and this walk is module-scope only,
    # for the reasons the module docstring gives. Named here because an entry
    # *was* added on the reasoning that it would be an edge, and the pin
    # refused it as stale -- which is the pin working.
    ("warlock.studio.panes.landing", "warlock.studio.modes.create.ui.stages"),
})

# P3 -- shared code moves out of studio/, DONE for Familiar's headless half
# (contract, router, retrieval, doors, character_plan): it now lives at
# warlock/familiar/, and service/familiar.py importing it is layer 3
# importing layer 3, not a violation any more -- the group that used to sit
# here (_P3_FAMILIAR_MOVES_OUT) is gone. One of its six pairs survives under
# a different name: see _UNRESOLVED's "pipelines/llama_client.py" entry --
# the move fixed the "-> studio/" shape but not the underlying layer number,
# because pipelines/ (L2) importing warlock/familiar/ (L3) is banned by
# dev/RESTRUCTURE.md's own table regardless of studio/ being involved, and
# no phase says who fixes that.

# P3/P7 -- "Packwright stays a mode... the overlap was tilegrid and the
# texture caches, which P3 and P7 already fix" (RESTRUCTURE.md's own words).
# Packwright's atlas writers reuse Plotter's PNG/TSX writers directly today.
_P3_P7_PACKWRIGHT_PLOTTER_OVERLAP: frozenset[tuple[str, str]] = frozenset({
    ("warlock.studio.modes.packwright.engine.compose", "warlock.studio.modes.plotter.engine.pngio"),
    ("warlock.studio.modes.packwright.engine.tsxout", "warlock.studio.modes.plotter.engine.tsx"),
    ("warlock.studio.modes.packwright.engine.wpack", "warlock.studio.modes.plotter.engine.pngio"),
})

# P4 -- the god-file split. The group this comment used to head is empty and
# gone (2026-09-17). One of its five pairs was a real edge and the move
# removed it: ``widgets.py`` re-exported ``artifacts.ARTIFACTS*`` and
# ``artifacts_for`` under its own name, and that alias -- not any drawing --
# was the whole of the dependency, so deleting the five aliases and pointing
# the callers at ``artifacts`` directly ended it. The other four were never
# violations: they only looked like ones because ``probe.py`` and
# ``quality.py`` were mis-filed as mode-owned (see STRAY_MODE_FILES, where
# the evidence for each is written out). RESTRUCTURE.md's own P4 bullet
# asked for Review's grade buttons to leave ``widgets`` too; the callers
# refused it -- ``grade_buttons`` and ``tag_toggles`` are drawn by
# ``panes/inspector.py``'s "Was this any good?" section on any mesh job,
# Create's Mesh stage included, not only by Review -- so they stayed, and
# the plan line was corrected rather than obeyed.

# P5 -- the pilot four. Two shapes: Familiar's UI half folding into
# studio/assistant/ (its Clay-preview and Create-doors reach), and Inker's
# `PaintView` (formerly inker_state.py:858-1476) promoting to
# shell/paintview.py, named explicitly as "already imported by Plotter and
# Packwright" -- landed, which is why the four pairs that used to sit here
# (``packwright_state``/``panes.packwright_preview``/``panes.plotter_canvas``/
# ``plotter_state`` -> ``modes.inker.state``) are gone rather than struck
# through: all four now import ``shell.paintview`` instead, a real shell
# import rather than a sibling-mode one, so there is nothing left to except.
# Create's own UI fold (landed: modes/create/) and the Clay fold (landed:
# modes/clay/{ui,agent}/) are the other two pilot-four bullets; what is left
# of them below is the shell and Familiar still naming a mode directly.
_P5_PILOT_FOUR: frozenset[tuple[str, str]] = frozenset({
    # Familiar UI -> Clay / Create
    ("warlock.studio.assistant.preview", "warlock.studio.modes.clay.agent.dispatch"),
    ("warlock.studio.assistant.preview", "warlock.studio.modes.clay.mode"),
    ("warlock.studio.assistant.preview", "warlock.studio.modes.clay.state"),
    # Clay agent fold
    # The MCP listener is shell and names Clay's surface and its transcript
    # recorder. The transcript edge is not new: ``agent_transcript.py`` sat
    # flat in ``studio/`` with no mode prefix, so it classified as shell and
    # the reach was invisible until P5 put it under ``modes/clay/agent/``.
    ("warlock.studio.agent_host", "warlock.studio.modes.clay.agent.dispatch"),
    ("warlock.studio.agent_host", "warlock.studio.modes.clay.agent.transcript"),
    # Create UI fold
    ("warlock.studio.asset_exits", "warlock.studio.modes.create.ui.stages"),
    ("warlock.studio.panes.inspector", "warlock.studio.modes.create.ui.stages"),
    # Create's recipe engine lifting out of panes/settings_*.py means Settings
    # can import the engine module directly instead of a pane object.
    ("warlock.studio.panes.app_settings", "warlock.studio.modes.create.ui.panes.settings_3d"),
})

# P6 -- the remaining modes, one agent per mode. Best-fit rather than named:
# RESTRUCTURE.md's P6 bullet does not call either edge out by name, but both
# targets are owned by a mode landing in this wave (Plotter's own tileset
# editor; the Library mode's drag payload, ``can_drag_mesh``/``draggable_mesh``,
# which Mason's palette offers the same rows through), and that landing is the
# plausible place either dependency gets resolved.
_P6_REMAINING_MODES: frozenset[tuple[str, str]] = frozenset({
    ("warlock.studio.modes.inker.ui.panes.tiles", "warlock.studio.modes.plotter.tilesets"),
    ("warlock.studio.modes.mason.ui.panes.palette", "warlock.studio.panes.library"),
})

# P10 -- Muse folds into Create's audio stage, explicitly removing "the
# muse <-> sirens cross-import" (RESTRUCTURE.md's own words) once
# kernels/audio/ (P3) holds what the two shared.
_P10_MUSE_FOLDS_INTO_CREATE: frozenset[tuple[str, str]] = frozenset({
    ("warlock.studio.modes.muse.mode", "warlock.studio.modes.sirens.audio"),
    ("warlock.studio.modes.muse.mode", "warlock.studio.modes.sirens.fileio"),
    ("warlock.studio.modes.muse.mode", "warlock.studio.modes.sirens.mode"),
    ("warlock.studio.modes.muse.mode", "warlock.studio.modes.sirens.state"),
    ("warlock.studio.modes.muse.ui.panes.player", "warlock.studio.modes.sirens.audio"),
    ("warlock.studio.modes.muse.ui.panes.results", "warlock.studio.modes.sirens.audio"),
})

# P11 -- Review folds into a Library view; P12 -- Home folds into Library's
# empty state. `panes/candidates_panel.py` and `panes/library.py` reaching
# into both Review and Create today are exactly the seam P9-P12 close.
_P11_P12_LIBRARY_ABSORBS: frozenset[tuple[str, str]] = frozenset({
    ("warlock.studio.panes.candidates_panel", "warlock.studio.panes.library"),
    ("warlock.studio.panes.candidates_panel", "warlock.studio.modes.review.mode"),
    ("warlock.studio.panes.library", "warlock.studio.modes.review.mode"),
})

# Not owned by any phase as dev/RESTRUCTURE.md is written today -- real,
# current edges the plan does not yet say who fixes. Kept in EXCEPTIONS
# (removing them would just make the suite red for a gap in the plan, not in
# the code) and named here for whoever picks the plan back up. See the P1
# landing report for the case each one earns.
#
# This block has emptied twice now, and both times by the same means: the
# files moved rather than the rule bending. P3 (2026-09-17) answered
# studio/undo.py's layer and studio/manual/{loader,parser}.py's by moving
# them (core/undo.py, kernels/manual/), and P4 wave two (2026-09-17)
# answered the five that were left -- config -> models became a constant
# living at the layer that actually owns it (which checkpoint the app
# defaults to is configuration, so DEFAULT_BASE_MODEL is config.py's now);
# clips.py -> pipelines/{charsheet,sheet} and kernels/pixel/sheetout.py ->
# pipelines/sheet both dissolved when sheet.py and charsheet.py moved to
# kernels/, which is where two stdlib-only modules that decide what cell 137
# depicts always belonged; and pipelines/llama_client.py -> familiar/contract
# was closed by moving the client beside the thing it is a client of, with
# one recorded httpx exemption in familiar's own import ban. Each is deleted
# here rather than struck through, per this file's own second test's rule for
# a landed phase -- what remains below is what genuinely has no phase.
# Empty since P6 (2026-09-18). Its last entry was Mason's palette reaching
# into Clay's Tools pane for ``PRIMITIVE_ICONS``, in the direction
# ``modes/clay/ops.py``'s ``_align`` docstring says is banned and no pin could
# see while the palette sat in ``studio/panes/``. The table was a glyph per
# mesh primitive -- shared vocabulary, not Clay's -- and is ``icons.py``'s now.
_UNRESOLVED: frozenset[tuple[str, str]] = frozenset()

EXCEPTIONS: frozenset[tuple[str, str]] = (
    _P2_SHELL_DISPATCH
    | _P3_P7_PACKWRIGHT_PLOTTER_OVERLAP
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
    """Every pair in EXCEPTIONS is still a real edge *and still a violation*.

    A pair that stops matching a real import is a phase that already landed
    -- delete the entry rather than leave it; a shrinking EXCEPTIONS is the
    restructure's own progress made visible in a diff.

    Asking only "is the import still there" was not enough. P5 moved Create's
    workspace and its asset table into one package, so
    ``create.ui.workspace -> create.engine.assets`` became a mode importing
    itself -- a legal edge -- and its waiver stayed green because the import
    had not gone anywhere. A waiver for an edge the rules allow is a phase
    that landed without anyone deleting its entry.
    """
    violating = {(edge.importer, edge.imported) for edge, _reason in _violations()}
    stale = sorted(pair for pair in EXCEPTIONS if pair not in violating)
    if not stale:
        return
    lines = [
        f"{imp} -> {tgt}" + ("" if (imp, tgt) in _EDGE_PAIRS else "  (import gone)")
        for imp, tgt in stale
    ]
    raise AssertionError(
        f"{len(stale)} stale EXCEPTIONS entry(ies) in tests/test_layering.py -- "
        "the import is gone or no longer breaks a rule, so delete the entry "
        "(the phase that removed it already landed):\n" + "\n".join(lines)
    )


def test_every_stray_mode_file_still_exists() -> None:
    """Every hand-attributed path in STRAY_MODE_FILES names a real file.

    A row keyed on a path that has moved does not fail anything -- it simply
    stops matching, and the file it was about falls through to whatever
    :func:`classify` says by default. P5 moved three of these (the Create
    settings panes) and the rows stayed behind naming nothing; this is the
    check that would have said so.
    """
    missing = sorted(rel for rel in STRAY_MODE_FILES if not (SRC / rel).is_file())
    assert missing == [], f"STRAY_MODE_FILES names files that are gone: {missing}"
