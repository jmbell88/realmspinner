"""The 2026-10-03 audit's Low findings docs-30 .. docs-89 owned by fixer docs2.

Two are parser fixes (``changelog.parse`` and ``kernels.manual.parser.slugify``);
the rest are document wording that drifted from the code it describes. The three
findings that live in ``dev/INVARIANTS.md`` (docs-42, docs-44, docs-48) have no
test here on purpose: ``dev/`` is gitignored and nothing under ``tests/`` may
read it, so those are carried to the ledger pass instead.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "docs" / "manual"
SRC = ROOT / "src" / "realmspinner"


def _flat(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


# --- docs-30 / docs-31: changelog.parse --------------------------------------


def test_an_indented_sub_bullet_stays_inside_its_parent_bullet():
    from realmspinner import changelog

    text = (
        "## 0.1.0\n"
        "\n"
        "- **Audit fixes.** Many small ones.\n"
        "  - *Shell:* a crash copy is kept.\n"
        "  - *Clay:* delete reparents.\n"
        "- Another release note.\n"
    )
    (release,) = changelog.parse(text)
    assert len(release.bullets) == 2, release.bullets
    assert "Shell: a crash copy is kept." in release.bullets[0]
    assert "Clay: delete reparents." in release.bullets[0]
    # The parent's lead is still the card's lead, not the first sub-item's.
    assert changelog.lead(release.bullets[0]).startswith("Audit fixes.")


def test_the_shipped_changelog_has_no_peer_bullet_that_was_a_sub_bullet():
    from realmspinner import changelog

    raw = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    nested = [line for line in raw if re.match(r"^\s+[-*]\s+\S", line)]
    top = [line for line in raw if re.match(r"^[-*]\s+\S", line)]
    parsed = sum(len(r.bullets) for r in changelog.parse("\n".join(raw)))
    assert nested, "the shipped file no longer nests a bullet; retire this test"
    assert parsed == len(top)


def test_a_glob_inside_a_code_span_keeps_its_asterisk():
    from realmspinner import changelog

    text = (
        "## 0.1.0\n\n"
        "- Fetches `*.gguf` and skips `*.part` files.\n"
        "- Moved `studio/panes/inker_*.py` and **kept** the *rest*.\n"
    )
    (release,) = changelog.parse(text)
    assert release.bullets[0] == "Fetches `*.gguf` and skips `*.part` files."
    assert release.bullets[1] == "Moved `studio/panes/inker_*.py` and kept the rest."


# --- docs-32: slugify --------------------------------------------------------


def test_slugify_matches_github_for_a_heading_with_punctuation_between_spaces():
    from realmspinner.kernels.manual.parser import slugify
    from tests.test_external_doc_links import _slug

    # GitHub drops the "&" and keeps both spaces, one hyphen each.
    assert slugify("Setup & operations") == "setup--operations"
    assert _slug("Setup & operations") == "setup--operations"
    assert slugify("Triangle budget") == "triangle-budget"
    assert _slug("What `doctor` checks") == "what-doctor-checks"


# --- docs-34: citations of paths that moved ------------------------------------


_CITED = (
    "service/tilesheets.py",
    "service/sweeps.py",
    "service/_jobs_lifecycle.py",
    "service/_jobs_resubmit.py",
)
_PATH = re.compile(r"``(?:src/realmspinner/)?(studio/[\w/]+\.py)``")


def test_service_and_queue_comments_cite_paths_that_exist():
    missing = []
    for name in _CITED:
        text = (SRC / name).read_text(encoding="utf-8")
        for cited in _PATH.findall(text):
            if not (SRC / cited).is_file():
                missing.append(f"{name}: {cited}")
        for dead in ("studio.panes.settings_2d", "studio/review_mode"):
            if dead in text:
                missing.append(f"{name}: {dead}")
    assert not missing, missing


# --- docs-35: SECURITY.md ----------------------------------------------------


def test_security_md_inventory_includes_palette_audio_and_ase_import_formats():
    from realmspinner.kernels.pixel.asein import ASEPRITE_SUFFIXES
    from realmspinner.service.palettes import SUFFIXES

    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    wanted = sorted(set(SUFFIXES) | set(ASEPRITE_SUFFIXES) | {".wav"})
    missing = [s for s in wanted if f"`{s}`" not in security]
    assert not missing, f"SECURITY.md's file inventory is missing {missing}"


# --- docs-36 / docs-37: README -------------------------------------------------


def test_readme_bench_subcommand_list_matches_the_parser():
    from realmspinner.bench.__main__ import build_parser

    parser = build_parser()
    (action,) = [a for a in parser._actions if a.dest == "command"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    line = next(
        ln for ln in readme.splitlines() if ln.lstrip().startswith("`python -m realmspinner.bench`")
    )
    listed = set(re.findall(r"`([a-z-]+)`", line.split("Subcommands:", 1)[1]))
    assert listed == set(action.choices)


def test_readme_does_not_call_the_downloaded_engine_vendored():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    flat = " ".join(readme.split())
    assert "vendored `trellis-server.exe`" not in flat
    assert "THIRD-PARTY-NOTICES.md" in flat
    start = flat.index("Third-party components")
    assert "downloads" in flat[start : start + 600]


# --- docs-38: THIRD-PARTY-NOTICES.md -------------------------------------------


def test_notices_python_dependency_rows_exist_in_the_lockfile():
    notices = (ROOT / "THIRD-PARTY-NOTICES.md").read_text(encoding="utf-8")
    section = notices.split("Installed from PyPI by `uv`", 1)[1].split("\n## ", 1)[0]
    names: set[str] = set()
    for line in section.splitlines():
        if line.startswith("| ") and not line.startswith(("| Package", "|---")):
            first = line.split("|")[1]
            names.update(re.findall(r"`([A-Za-z0-9_.-]+)`", first))
    assert names
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    locked = {n.lower().replace("_", "-") for n in re.findall(r'^name = "([^"]+)"', lock, re.M)}
    missing = sorted(n for n in names if n.lower().replace("_", "-") not in locked)
    assert not missing, f"the notices name packages the lockfile lacks: {missing}"


# --- docs-73: docstrings naming surfaces that are gone ----------------------------


def test_service_and_queue_docstrings_name_no_removed_surface():
    from realmspinner import queue
    from realmspinner.service import judge

    status = inspect.getdoc(judge.status) or ""
    assert "realmspinnerc" not in status
    wake = inspect.getdoc(queue.Worker.wake) or ""
    assert "routes" not in wake
    assert "wake_worker" in wake
    stopping = inspect.getdoc(queue._Cancel.stopping) or ""
    assert "nothing in" not in " ".join(stopping.split())


# --- docs-77 / docs-81: chapter 45 ------------------------------------------------


def test_manual_45_does_not_count_blender_ops_by_ordinal():
    text = _flat(MANUAL / "45-pipelines.md")
    assert "adds a fourth operation" not in text
    assert "Rigging, pose baking and sprite-sheet rendering all need Blender" not in text
    # The op list named is the worker's own.
    section = text.split("## Blender out of process", 1)[1][:600]
    for op in ("clay", "remesh"):
        assert op in section.lower()


def test_manual_45_256px_claim_is_scoped_to_the_character_sheet_ladder():
    from realmspinner.kernels import charsheet

    text = _flat(MANUAL / "45-pipelines.md")
    assert "The size ladder now runs to 256 px" not in text
    assert "character-sheet size ladder" in text
    assert max(charsheet.SIZES) == 256


# --- docs-84: chapter 21 ----------------------------------------------------------


def test_manual_21_describes_what_home_draws():
    from realmspinner.studio.modes.home.ui.panes import landing

    text = _flat(MANUAL / "21-home.md")
    assert "the only place in the UI it appears" not in text
    assert "first three lines" not in text
    assert landing.NEWS_BULLETS == 3
    assert "three bullets" in text
    assert "You are running Realmspinner" in text
    assert "Unsaved work" in text
    assert "28-inker.md#autosave-and-recovery" in text


# --- docs-89: chapter 30 ----------------------------------------------------------


def test_manual_30_op_names_match_registered_labels():
    from realmspinner.studio.modes.clay import ops

    labels = {op.label.rstrip(".") for op in ops.OPS}
    text = (MANUAL / "30-clay.md").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    for name in ("Inset Faces", "Bevel Edges", "Bake Transform", "Retopologize"):
        assert name in labels, name
        assert name in flat, name
    assert "| Faces | Inset |" not in flat
    assert "| Edges | Bevel |" not in flat
    assert "**Retopologise**" not in flat
    assert "**Bake** folds" not in flat
