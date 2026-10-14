"""Regression tests for the 2026-10-14 audit's root-document findings (fixer D6).

Every test reads a document through ``ROOT`` so the claim it makes can be pointed at an older copy
of the file. The ``dev/TODO.md`` tests skip where ``dev/`` is absent (a public clone, CI): that
directory is gitignored and nothing under ``tests/`` may require it.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import realmspinner

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(realmspinner.__file__).parent


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _todo() -> str:
    path = ROOT / "dev" / "TODO.md"
    if not path.is_file():
        pytest.skip("dev/TODO.md is gitignored and absent on a public checkout")
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


# --- bpy never runs in the app process (poser-poses-01, poser-rig-01) ----------------------------


def test_no_module_but_blender_worker_imports_bpy():
    """``bpy`` is process-global and crashes, rather than raising, on the geometry trellis makes.

    The 2026-10-14 audit (poser-poses-01, poser-rig-01): ``eed32549`` deleted the only tree-wide
    scan for a stray ``import bpy`` along with the Poser mode's import pins, and nothing replaced
    it. ``tests/test_rigging.py`` names only the ``kernels/rig`` files it reads. This walks every
    module under ``src/realmspinner``; only ``pipelines/blender_worker.py``, which runs in a child,
    may import it. CONTRIBUTING.md names this test as the enforcer.
    """
    exempt = SRC / "pipelines" / "blender_worker.py"
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if path == exempt:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                hit = any(a.name == "bpy" or a.name.startswith("bpy.") for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                hit = bool(node.module) and (node.module == "bpy" or node.module.startswith("bpy."))
            else:
                continue
            if hit:
                offenders.append(str(path.relative_to(SRC)))
    assert offenders == []


def test_the_pure_rig_storage_modules_import_with_no_studio_and_no_bpy():
    """``poselib``, ``clipmaps`` and ``cliptransfer`` outlived the Poser mode and still claim to be
    decidable with no studio and no Blender; their import pins went with the mode's test file, so
    the claim is executed here in a fresh interpreter (``None`` in ``sys.modules`` makes a hidden
    import raise)."""
    stubs = ("imgui", "imgui_bundle", "moderngl", "pygame", "bpy", "realmspinner.studio")
    script = (
        "import sys\n"
        f"for name in {stubs!r}: sys.modules[name] = None\n"
        "import realmspinner.poselib, realmspinner.clipmaps, realmspinner.cliptransfer\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_contributing_names_a_bpy_scan_that_exists():
    """docs-A-B-04: CONTRIBUTING.md sent contributors to ``tests/modes/poser/``, which is gone."""
    text = _read("CONTRIBUTING.md")
    assert "tests/modes/poser" not in text
    named = re.search(
        r"`(tests/[\w/]+\.py)::(test_no_module_but_blender_worker_imports_bpy)`", text
    )
    assert named, "CONTRIBUTING.md no longer names the bpy scan test"
    assert (ROOT / named.group(1)).is_file()
    assert f"def {named.group(2)}" in (ROOT / named.group(1)).read_text(encoding="utf-8")


# --- INSTALL.md (pipelines-install-04, docs-A-B-06) ----------------------------------------------


def test_install_md_does_not_call_pack_restoration_after_an_upgrade_a_known_gap():
    """Since 0.1.4 ``packs.restore_at_startup`` puts the packs back behind the splash, and manual 42
    says so; INSTALL.md told users to reinstall by hand and called that a known gap."""
    text = " ".join(_read("INSTALL.md").split())
    assert "known gap" not in text
    assert "nothing restores them automatically" not in text
    assert "Restore packs" in text, "the fallback button is the one the upgrade section must name"
    assert "puts them back" in text or "reinstalls the packs" in text


def test_install_md_installer_size_matches_the_release_manifest():
    """docs-A-B-06: INSTALL.md stated 169,666,529 bytes, a figure that matched no release since
    2026-09-10. It states a rounded size now, and (where the release manifest is on disk) the
    rounded figure must sit within 5% of the manifest's ``size_bytes``."""
    text = _read("INSTALL.md")
    assert not re.search(r"\d{1,3}(?:,\d{3}){2,} bytes", text), (
        "INSTALL.md states an exact byte count"
    )
    claimed = re.search(r"Installer download: \*\*about (\d+) MB\*\*", text)
    assert claimed, "INSTALL.md lost its rounded installer size"
    manifest = ROOT / "dist" / "update-manifest.json"
    if not manifest.is_file():
        pytest.skip("dist/update-manifest.json is not on disk")
    size = json.loads(manifest.read_text(encoding="utf-8"))["installer"]["size_bytes"] / 1e6
    assert abs(int(claimed.group(1)) - size) <= 0.05 * size, (size, claimed.group(1))


# --- SECURITY.md (docs-A-B-01, -02, -12) ---------------------------------------------------------


def test_security_md_describes_only_the_clay_agent_surface():
    """The 2026-10-14 audit, finding docs-A-B-01: the MCP bullet still scoped a character-pipeline
    surface (read any Library row, additive-only charsheet writes, ``agent_export_stem``) that
    ``eed32549`` removed. The bridge is Clay-only while no character tool surface exists."""
    assert not list((SRC / "studio").glob("agent_character*.py"))
    text = _read("SECURITY.md")
    for stale in (
        "charsheet",
        "agent_export_stem",
        "read any Library row",
        "character-pipeline's read-any",
    ):
        assert stale not in text, stale
    assert "character pipeline" not in text.replace("character-pipeline tool surface", "")


def test_security_md_import_formats_are_only_those_a_door_can_reach():
    """docs-A-B-02: SECURITY.md credited "Poser's clip importer" with ``.fbx`` and ``.gltf``, but no
    code in ``src/`` calls ``service/clip_import.py``; Clay's Import Animation reads ``.glb``
    only."""
    text = _read("SECURITY.md")
    assert "Poser" not in text
    assert "clip_import" not in text
    assert "`.fbx`" not in text and "`.gltf`" not in text
    assert "kernels.mesh.animimport" in text


def test_security_md_subprocess_list_names_every_spawn_site():
    """docs-A-B-12: the child-process inventory omitted the text-to-image worker, ``gltfpack`` and
    Flourish's ``recipe_worker``; each has a Popen site inside the kill-on-close job."""
    text = " ".join(_read("SECURITY.md").split())
    for name in (
        "t2i_client",
        "gltfpack",
        "recipe_worker",
        "trellis-server.exe",
        "llama-server.exe",
    ):
        assert name in text, f"SECURITY.md's subprocess list never names {name}"
    for module in ("pipelines/t2i_client.py", "pipelines/optimize.py"):
        assert "Popen" in (SRC / module).read_text(encoding="utf-8") or "subprocess" in (
            SRC / module
        ).read_text(encoding="utf-8")


# --- README.md (docs-A-B-03, -07, and the GGUF size) ---------------------------------------------


def test_readme_does_not_promise_rendered_sheets_from_a_rig():
    """docs-A-B-03: the ``sheet``/``pixel_sheet``/``charsheet`` kinds are retired and the Pose-stage
    turnaround is gone, so the rigging paragraph may not promise sprite or turnaround sheets."""
    text = " ".join(_read("README.md").split())
    assert "turnaround sheet" not in text
    assert "and sprite sheets" not in text


def test_readme_licence_sentence_agrees_with_the_notices_file():
    """docs-A-B-07: README said two models restrict commercial use and that the picker shows every
    model's licence. The notices file says one (Hybrid Demucs) and only some entries carry
    a line."""
    readme = " ".join(_read("README.md").split())
    notices = " ".join(_read("THIRD-PARTY-NOTICES.md").split())
    assert "two of them restrict commercial" not in readme
    assert "one of them restricts commercial use" in notices
    assert (
        "one of them (the optional Hybrid Demucs stem separator) restricts commercial use" in readme
    )
    assert "shows a licence line" in readme and "each model's licence in the picker" not in readme


def test_readme_trellis_gguf_size_is_the_registry_rows_decimal_size():
    """The 2026-10-14 audit, finding docs-D-03, corrected the row to 15.34 GiB; README quoted the
    old 16.1. The README quotes decimal GB, so the figure is derived from the row."""
    from realmspinner import models

    gib = models.ENGINE_MODELS["trellis_gguf"].fetch[0].size_gib
    decimal = f"{gib * 1.073741824:.1f} GB"
    text = _read("README.md")
    assert text.count(f"({decimal})") + text.count(f"{decimal} of TRELLIS.2 GGUF") == 2, decimal
    assert "16.1 GB" not in text


# --- docs/MODELS.md and docs/create-quality.md (docs-D-09, docs-D-01) -----------------------------


def test_models_md_retrieval_row_says_the_file_is_the_text_block():
    """docs-D-09: "a 740M-parameter embedding model" beside a 0.29 GiB download. The row installs
    the text block only."""
    text = " ".join(_read("docs/MODELS.md").split())
    assert "of a 740M-parameter embedding model" in text
    assert not re.search(r"(?<!of )a 740M-parameter embedding model", text)
    assert "the text block of a 740M-parameter embedding model" in text


def test_create_quality_cites_no_gitignored_measurement_path():
    """docs-D-01: ``docs/create-quality.md`` cited ``dev/measurements/...``, which a public clone
    does not carry."""
    assert "dev/measurements" not in _read("docs/create-quality.md")


def test_external_doc_links_walks_create_quality():
    assert "create-quality.md" in (ROOT / "tests" / "test_external_doc_links.py").read_text(
        encoding="utf-8"
    )


# --- dev/TODO.md (docs-E-*) -----------------------------------------------------------------------


def test_todo_header_names_every_entry_that_claims_deliberately_unstarted():
    """docs-E-14: the header said T7 was the only fully-specified, deliberately unstarted entry
    while P54 and P100 made the same claim."""
    text = _todo()
    header = text[: text.index("**The moment an item could be built")]
    assert "P54" in header and "P109" in header


def test_todo_struck_entries_stay_struck():
    """docs-E-01, -02, -03: P72 (release cut), P70 (retired plans) and P44 (retired LoRA row) were
    built, decided or moot but still read as open."""
    text = _todo()
    for number in (44, 70, 72, 90):
        assert re.search(rf"^## ~~P{number}\.", text, re.M), f"P{number} is open again"


def test_todo_p54_names_only_modules_that_exist():
    """docs-E-08: P54 step 1 named ``agent_character.HANDLERS`` (deleted with the character
    surface) and a ``clay-1.txt`` exemption in ``test_old_name_gone.py`` that never existed."""
    text = _todo()
    start = text.index("## P54.")
    section = text[start : text.index("\n## ", start + 3)]
    assert "agent_character.HANDLERS" not in section
    assert "character-1.txt" not in section
    assert "test_old_name_gone" not in section
    cards = {p.name for p in (SRC / "familiar" / "cards").glob("*.txt")}
    for card in re.findall(r"`((?:clay|router|character)-\d\.txt)`", section):
        assert card in cards or "retired" in section, card


def test_every_todo_entry_number_it_cites_has_a_heading_or_a_closed_record():
    """docs-E-11: P28 was cited as the pending sitting from P8, P22 and the closed P5, with no
    heading and no closed record anywhere."""
    text = _todo()
    cited = {int(n) for n in re.findall(r"\bP(\d+)\b", text)}
    headed = {int(n) for n in re.findall(r"^## (?:~~)?P(\d+)\.", text, re.M)}
    closed: set[int] = set()
    for line in text.splitlines():
        lead = re.match(r"- (?:~~)?\*\*((?:P\d+(?:[/,–-]\s*)?)+)", line)
        if lead:
            closed.update(int(n) for n in re.findall(r"P(\d+)", lead.group(1)))
    assert sorted(cited - headed - closed) == []


def test_todo_manual_chapter_citations_resolve_to_the_named_topic():
    """docs-E-12: chapter numbers written before the 2026-10-14 renumbering named other chapters.
    Every cited chapter file exists, and a topic keyword beside a chapter number agrees with it."""
    text = _todo()
    chapters = {p.name[:2]: p.name for p in (ROOT / "docs" / "manual").glob("[0-9][0-9]-*.md")}
    for number in re.findall(r"(?:[Cc]hapters?|manual) (\d\d)\b", text):
        assert number in chapters, f"chapter {number} does not exist"
    topic = {
        "Stems": {"32"},
        "Make prefab": {"28"},
        "Simplify": {"27"},
        "Rows run bottom-up": {"25"},
    }
    for match in re.finditer(r"(?:[Cc]hapters?|manual) (\d\d)\b", text):
        window = text[max(0, match.start() - 160) : match.end() + 160]
        for keyword, allowed in topic.items():
            if keyword in window:
                assert match.group(1) in allowed, (match.group(0), keyword)


def test_todo_cites_no_test_or_script_the_tree_lacks():
    """docs-E-13 / docs-E-11: an open entry may not send a human to a path that is gone (the
    ``docs/manual/26-poser.md`` chapter was one). Struck entries and the closed records are
    records, so they are not held to it."""
    text = _todo()
    live = text[: text.index("## Closed records")]
    missing: list[str] = []
    for section in re.split(r"\n(?=## )", live):
        if section.startswith("## ~~"):
            continue
        for path in re.findall(r"`((?:tests|scripts|docs)/[\w./\-]+\.(?:py|md|ps1))`", section):
            if not (ROOT / path).exists():
                missing.append(f"{section.splitlines()[0][:40]}: {path}")
    assert missing == []
