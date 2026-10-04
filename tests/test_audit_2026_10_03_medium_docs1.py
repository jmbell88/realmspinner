"""The 2026-10-03 audit's Medium documentation findings, batch 1: docs-03,
docs-17 (the config.py comment half), docs-18 and docs-19.

Each test names the claim and was run red against the unfixed tree first.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "realmspinner"

_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def _squash(text: str) -> str:
    """Whitespace folded and curly quotes straightened, so a ledger row that wraps
    a line or uses a typographic apostrophe still quotes the code's sentence."""
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text)


# --- docs-03: a pasted curl must be able to create the folder it writes into ----


def test_a_url_fetch_command_creates_its_destination_directory():
    """docs-03. ``curl -o`` into a directory that does not exist exits 23
    ("client returned ERROR on write"); the engine and Hybrid Demucs
    directories do not exist on a machine that has not downloaded them, which
    is exactly the machine the paste-able line is for. ``Fetch.command()`` and
    both curl blocks in docs/MODELS.md must say ``--create-dirs``."""
    from realmspinner import fetch

    seen = 0
    for entry in fetch.entries():
        for one in entry.fetch:
            if one.url:
                seen += 1
                assert "--create-dirs" in one.command(), (
                    f"{entry.row_key}: {one.command()!r} cannot write into a folder "
                    "that does not exist yet"
                )
    assert seen, "no url record left in the registry -- the docs-03 guard has nothing to guard"

    models_md = (ROOT / "docs" / "MODELS.md").read_text(encoding="utf-8")
    # One logical command per block: a trailing backtick continues the line.
    commands = re.findall(r"^curl .*(?:`\r?\n[ \t]+.*)*", models_md, flags=re.M)
    assert commands, "docs/MODELS.md has no curl command left"
    for command in commands:
        assert "--create-dirs" in command, f"MODELS.md curl without --create-dirs: {command!r}"


# --- docs-18: every warning the aseprite reader raises has a ledger row ---------


def _asein_warning_texts() -> list[tuple[int, str]]:
    """Every literal handed to a ``warn(...)`` / ``state.warn(...)`` in
    asein.py, as the longest static run of the message (an f-string's literal
    pieces; a plain or implicitly concatenated string whole)."""
    tree = ast.parse((SRC / "kernels" / "pixel" / "asein.py").read_text(encoding="utf-8"))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) != 1:
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name != "warn":
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.append((node.lineno, arg.value))
        elif isinstance(arg, ast.JoinedStr):
            pieces = [v.value for v in arg.values if isinstance(v, ast.Constant)]
            out.append((node.lineno, max(pieces, key=len)))
    return out


def test_every_asein_warning_has_a_row_in_the_aseprite_ledger():
    """docs-18. The "aseprite -> ORA (the asein.py reader's warning table)"
    section of docs/COMPAT.md omitted four warnings -- among them "a cel
    reaching past the canvas was cropped to it", the one that discards picture
    data -- so a user could not find it in the ledger. Collected from the AST,
    so the next ``warn("...")`` added without a row fails here."""
    compat = (ROOT / "docs" / "COMPAT.md").read_text(encoding="utf-8")
    start = compat.index("### aseprite → ORA")
    end = compat.index("\n## ", start)
    section = _squash(compat[start:end])

    found = _asein_warning_texts()
    assert len(found) > 25, (
        f"the AST walk found only {len(found)} warn() calls -- the collector broke"
    )
    missing = [
        f"asein.py:{line}: {text!r}"
        for line, text in found
        if _squash(text).strip(" ;,.:") not in section
    ]
    assert not missing, (
        "asein.py raises these warnings and the aseprite -> ORA ledger quotes none of them:\n  "
        + "\n  ".join(missing)
    )


# --- docs-19: five recipes share the SDXL 1.0 checkpoint -----------------------


def _sdxl_sharing_count() -> int:
    from realmspinner.models import BASE_MODELS

    sdxl_dir = BASE_MODELS["sdxl_cfg"].dir_name
    return len([m for m in BASE_MODELS.values() if m.dir_name == sdxl_dir])


def test_models_md_counts_the_sdxl_recipes_that_share_the_base_checkpoint():
    """docs-19. MODELS.md said "four registered recipes share" the 7 GB
    download and manual chapter 42 said "four of the image models" and "the
    other three are still standing on"; five registry entries carry the same
    checkpoint. The number is read from the registry, not restated."""
    count = _sdxl_sharing_count()
    word, rest = _NUMBER_WORDS[count], _NUMBER_WORDS[count - 1]

    models_md = _squash((ROOT / "docs" / "MODELS.md").read_text(encoding="utf-8"))
    assert f"base download {word} registered recipes share" in models_md, (
        f"{count} registry entries share the SDXL checkpoint but docs/MODELS.md does not say {word}"
    )

    manual = _squash((ROOT / "docs" / "manual" / "42-app-settings.md").read_text(encoding="utf-8"))
    assert f"{word} of the image models share one set of SDXL 1.0 weights" in manual
    assert f"picking all {word} downloads them once" in manual
    assert f"the {word} SDXL 1.0 recipes share one 7 GB checkpoint" in manual
    assert f"the weights the other {rest} are still standing on" in manual


def test_no_comment_still_counts_four_sdxl_recipes():
    """docs-19, the comments and docstrings the record lists: config.py,
    fetch.py and service/downloads.py each restated the stale four."""
    stale = re.compile(
        r"\bfour (?:SDXL 1\.0 )?recipes\b|one of four recipes|all four (?:SDXL|are chosen)"
        r"|shared by four recipes|sdxl_cfg`?,? ``?pixel``? and\s+``?lightning",
        re.I,
    )
    hits = []
    for rel in ("config.py", "fetch.py", "service/downloads.py"):
        text = (SRC / rel).read_text(encoding="utf-8")
        flat = re.sub(r"\n\s*(?:#:?|\"\"\")?\s*", " ", text)
        for m in stale.finditer(flat):
            hits.append(f"{rel}: ...{flat[max(0, m.start() - 30) : m.end() + 30]}...")
    assert not hits, "stale four-recipe count:\n  " + "\n  ".join(hits)


# --- docs-17 (the config.py half): the trellis exe is a download, not vendored ---


def test_gltfpack_comment_does_not_call_trellis_server_exe_vendored_and_never_downloaded():
    """docs-17. The comment above ``gltfpack_exe`` said "Vendored like
    trellis-server.exe: a pinned native binary, never downloaded at runtime".
    Since 2026-09-10 trellis-server.exe *is* a Settings -> Models download
    (``Config.resolve_trellis_exe`` prefers it, ``vendor/trellis`` is only the
    fallback), so the analogy sends a maintainer looking in the checkout."""
    text = (SRC / "config.py").read_text(encoding="utf-8")
    assert "Vendored like trellis-server.exe" not in text
