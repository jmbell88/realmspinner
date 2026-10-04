"""The 2026-10-03 audit's docs1 Low findings: prose that had drifted from the
tree it describes, each pinned to the thing it claims about.

One file for the whole slice, as the fixers' rules ask. Findings whose only
home is a gitignored file (``dev/INVARIANTS.md``, ``CLAUDE.md``) are not here:
nothing under ``tests/`` may read ``dev/``, since CI clones without it.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "realmspinner"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _docstring_of(path: Path, name: str | None = None) -> str:
    tree = ast.parse(_text(path))
    if name is None:
        return ast.get_docstring(tree) or ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_docstring(node) or ""
    raise AssertionError(f"{name} not found in {path}")


# --- docs-47: the jobs facade's sibling list -------------------------------------


def test_jobs_facade_docs_list_every_jobs_sibling_module():
    siblings = sorted(
        p.stem for p in (SRC / "service").glob("_jobs_*.py") if p.stem != "_jobs_support"
    )
    doc = _docstring_of(SRC / "service" / "jobs.py")
    for stem in siblings:
        assert stem in doc, f"service/jobs.py's docstring does not name {stem}"
    words = {5: "five", 6: "six", 7: "seven", 8: "eight"}
    # Every sibling that carries doors is named; the count word must agree with the list.
    named = [s for s in siblings if s in doc]
    assert f"over {words[len(named)]} siblings" in doc


# --- docs-52: which import pins still carry a hand list --------------------------


def _unconverted_pins() -> set[str]:
    """Pins that exist and never call ``siblings_of``, derived from the tests tree."""
    converted: set[str] = set()
    existing: set[str] = set()
    for pin in (ROOT / "tests" / "modes").glob("*/test_*imports*.py"):
        mode = pin.parent.name
        existing.add(mode)
        if "siblings_of(" in _text(pin):
            converted.add(mode)
    unconverted = existing - converted
    grid = ROOT / "tests" / "kernels" / "grid2d"
    pins = list(grid.glob("test_*imports*.py"))
    if pins and not any("siblings_of(" in _text(p) for p in pins):
        unconverted.add("grid2d")
    return unconverted


def test_the_unconverted_sibling_pin_list_matches_the_pins_that_do_not_call_siblings_of():
    doc = _docstring_of(ROOT / "tests" / "_pure_packages.py")
    match = re.search(r"\*\*Not yet adopted by (.*?),\s+and that is a live loose end", doc, re.S)
    assert match, "the loose-end sentence in _pure_packages.py's docstring moved"
    named = set(re.findall(r"``(\w+)``", match.group(1)))
    assert named == _unconverted_pins()


# --- docs-59 / docs-60: the lost-measurement markers -----------------------------

_MARKER = "re-measure to change"

#: Constants the stored corpus is keyed on whose cited dated document is gone.
_KEYED = [
    ("pipelines/seam.py", r"^SEAM_MAX = "),
    ("pipelines/seam.py", r"^SEAM_DOMINANCE_MAX = "),
    ("kernels/pixel/tiling.py", r"^SEAM_MAX = "),
    ("kernels/pixel/tiling.py", r"^SEAM_DOMINANCE_MAX = "),
    ("bench/metrics.py", r"^HASH_FLOOR = "),
    ("vectors.py", r"^GRADE_MIN, GRADE_MAX = "),
    ("service/loras.py", r"^DUPLICATE_SIMILARITY = "),
    ("config.py", r"^    mesh_hole_max: float"),
    ("config.py", r"^    mesh_profile: str"),
    ("config.py", r"^    trellis_band: int"),
    ("config.py", r"^    lowpoly_triangles: int"),
]


@pytest.mark.parametrize(("rel", "anchor"), _KEYED)
def test_every_corpus_keyed_constant_with_a_lost_measurement_says_so(rel: str, anchor: str):
    lines = _text(SRC / rel).splitlines()
    at = next(i for i, line in enumerate(lines) if re.search(anchor, line))
    # The marker sits in the comment block directly above the definition.
    block: list[str] = []
    for line in reversed(lines[:at]):
        if not line.lstrip().startswith("#"):
            break
        block.append(line)
    joined = " ".join(part.strip("# :").strip() for part in reversed(block))
    assert _MARKER in joined, f"{rel}: {anchor!r} carries no lost-measurement marker"


def test_max_total_bytes_comment_does_not_claim_both_no_document_and_a_lost_document():
    lines = _text(SRC / "kernels" / "geom3d" / "gltf.py").splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("MAX_TOTAL_BYTES ="))
    block = []
    for line in reversed(lines[:at]):
        if not line.lstrip().startswith("#"):
            break
        block.append(line)
    joined = " ".join(block)
    assert "none is owed" in joined
    assert "re-measure" not in joined and "was lost" not in joined


def test_lowpoly_marker_does_not_misdate_the_loss_to_the_2026_09_20_restore():
    text = _text(SRC / "config.py")
    at = text.index("    lowpoly_triangles: int")
    block = text[text.rindex("# (dev/measurements/2026-09-23", 0, at) : at]
    assert "2026-09-20 restore" not in block
    assert _MARKER in block.replace("\n    #", "")


# --- docs-61..64: docs/MODELS.md --------------------------------------------------


def _models_md() -> str:
    return _text(ROOT / "docs" / "MODELS.md")


def test_models_md_names_the_models_row_delete_button_by_its_label():
    ui = _text(SRC / "studio" / "modes" / "settings" / "ui" / "panes" / "app_settings.py")
    assert "Delete##{row_key}" in ui
    para = next(p for p in _models_md().split("\n\n") if "removes them again" in p)
    assert "**Delete**" in para and "**Remove**" not in para


def test_models_md_does_not_claim_a_single_non_hub_entry():
    text = _models_md()
    assert "the one entry on this page that is **not a Hugging Face" not in text
    assert "only optional\ndownload in this list" not in text
    assert "the second of them" not in text
    assert "the first of them" in text


def test_models_md_flux1_note_sits_under_the_image_models_heading():
    text = _models_md()
    at = text.index("**FLUX.1 is not offered; FLUX.2 klein is.**")
    heading = re.findall(r"^#{2,3} (.+)$", text[:at], re.M)[-1]
    assert heading == "Image models and style LoRAs"


def test_models_md_licence_row_agrees_with_the_registry_for_the_lcm_recipe():
    from realmspinner import models

    lcm = next(m for m in models.BASE_MODELS.values() if "LCM" in m.label)
    text = _models_md()
    table = text[text.index("| Model | Licence"):text.index("The OpenRAIL family")]
    row = next(line for line in table.splitlines() if "LCM" in line)
    assert lcm.license in row
    # And the combined row no longer sweeps the LCM recipe in under plain OpenRAIL++-M.
    assert "Hyper-SD / LCM / Lightning" not in table


# --- docs-65: COMPAT.md names the dropped dpi ------------------------------------


def test_compat_ledger_names_dpi_as_dropped_on_aseprite_write():
    text = _text(ROOT / "docs" / "COMPAT.md")
    start = text.index("### ORA → aseprite")
    end = text.index("### aseprite → ORA")
    rows = [r for r in text[start:end].splitlines() if "`Document.dpi`" in r]
    assert len(rows) == 1 and "| dropped |" in rows[0]


# --- docs-71: job_dir_file's name rule -------------------------------------------


def test_job_dir_file_docstring_says_the_name_can_come_from_an_agent():
    doc = _docstring_of(SRC / "service" / "files.py", "job_dir_file")
    assert "never user input" not in doc
    assert "clay_reference_add" in doc and "MCP agent" in doc


@pytest.mark.parametrize("name", ["../x.png", "..\\x.png", "a/b.png", "", "..", "."])
def test_job_dir_file_refuses_a_name_that_leaves_the_job_directory(tmp_path, name):
    from realmspinner.service import files
    from realmspinner.service.errors import Invalid

    class _Svc:
        def job_dir(self, job_id):
            return tmp_path / job_id

    job_id = "0" * 12
    with pytest.raises(Invalid):
        files.job_dir_file(_Svc(), job_id, name)


# --- docs-72: counts in service docstrings ---------------------------------------


def test_service_docstring_counts_match_their_registries():
    service = SRC / "service"
    rework = _docstring_of(service / "_jobs_rework.py")
    assert "remesh" in rework and "stems" in rework  # the module holds those doors too
    require = _docstring_of(service / "_jobs_rework.py", "_require_no_dependents")
    assert "Both doors here" not in require
    assert "separate_stems" not in require and "separate_job" in require
    # Every name it cites exists.
    assert "def separate_job" in _text(service / "_jobs_rework.py")

    save = _docstring_of(service / "files.py", "_save_source")
    callers = len(re.findall(r"return _save_source\(", _text(service / "files.py")))
    # save_clay_source is the model the helper copies; the callers are the rest.
    assert callers == 3 and "three documents" in save

    attach = _docstring_of(service / "files.py", "attach_files")
    assert "One stat per row" not in attach and "Two stats per row" in attach

    # queued_sheets: the number of doors that reserve through check_sheet_cap.
    doors = 0
    for path in service.glob("*.py"):
        for node in ast.walk(ast.parse(_text(path))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "check_sheet_cap":
                doors += 1
    queued = _docstring_of(service / "sheets.py", "queued_sheets")
    words = {3: "Three", 4: "Four", 5: "Five", 6: "Six"}
    assert f"{words[doors]} doors reserve" in " ".join(queued.split())

    chars = _docstring_of(service / "characters.py")
    assert "four ordinary rows" not in chars


# --- docs-79: chapter 41's home tree ---------------------------------------------


def test_manual_41_home_tree_lists_engine_packs_and_updates():
    text = _text(ROOT / "docs" / "manual" / "41-configuration.md")
    tree = text[text.index("~/.realmspinner/\n"):]
    tree = tree[: tree.index("```")]
    for entry in ("engine/trellis/", "packs/", "updates/", "mcp.token", "studio_settings.json"):
        assert entry in tree, f"chapter 41's home tree omits {entry}"


# --- docs-82: the re-texture view count ------------------------------------------


def test_retexture_docstrings_state_the_view_count():
    from realmspinner.pipelines import retexture

    assert len(retexture.VIEWS) == 10
    for rel in ("studio/panes/texture_panel.py", "service/_jobs_rework.py"):
        text = _text(SRC / rel)
        assert "six SDXL" not in text and "six conditioned" not in text, rel
        assert "ten " in text, rel


# --- docs-83: which entries ignore a negative prompt -----------------------------


def test_manual_22_negative_prompt_inert_entries_match_the_registry():
    from realmspinner import models

    inert = [m for m in models.BASE_MODELS.values() if m.guidance_scale <= 1.0]
    assert len(inert) == 5
    text = _text(ROOT / "docs" / "manual" / "22-generating-references.md")
    flat = " ".join(text.split())
    assert "the two four-step distilled defaults ignore it" not in flat
    assert "on the three four-step entries" not in flat
    assert "the five distilled entries" in flat
    for label in ("Turbo", "Hyper-SD", "Lightning", "LCM", "FLUX.2 klein distilled"):
        assert label in flat
