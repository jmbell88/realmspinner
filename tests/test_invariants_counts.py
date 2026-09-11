"""Counts `docs/INVARIANTS.md` states, read against the tables that own them.

That document is the authoritative record of this codebase's hard constraints,
which makes a number inside it a claim like any other -- and two of them had
gone stale by the 2026-09-11 audit (findings docs-05 and docs-06). Both were
the same shape: a table grew, the guard that scans it kept working, and the
prose describing the guard did not follow.

The publisher count is the sharper of the two, because the paragraph it sits in
is *about* a guard going stale -- "``PUBLISHERS`` is a hand-written table and
none of the four was in it, which is how a rule with a scan behind it was
broken four more times" -- and then its own count drifted when ``_remesh``
joined the table.

`tests/manual/test_manual_promises.py` does this for the manual's chapters.
This file does it for INVARIANTS, and it reads each number from the table
rather than keeping a second copy of it here.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

INVARIANTS = Path(__file__).resolve().parents[1] / "docs" / "INVARIANTS.md"


def _from_sibling(module: str, name: str) -> Any:
    """One constant out of a sibling test module, by path.

    ``tests/`` is not a package, so a plain import of a sibling does not
    resolve; loading by file path keeps this file working whether pytest is
    run from the root or from inside ``tests/``.
    """
    path = Path(__file__).with_name(f"{module}.py")
    spec = importlib.util.spec_from_file_location(f"_counts_{module}", path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return getattr(loaded, name)

#: Spelled numbers, because this document writes its counts as words.
WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}


def _paragraph(lead_in: str) -> str:
    """The paragraph whose bold lead-in starts with ``lead_in``."""
    text = INVARIANTS.read_text(encoding="utf-8")
    start = text.index(lead_in)
    end = text.find("\n\n", start)
    return text[start : end if end != -1 else len(text)]


def test_invariants_publishers_count_matches_the_guard_table():
    count = len(_from_sibling("test_job_durability", "PUBLISHERS"))
    para = _paragraph("**A stage that has published onto a served name commits")

    assert f"{WORDS[count].capitalize()} stages publish" in para, (
        f"docs/INVARIANTS.md does not say {WORDS[count]!r} stages publish; "
        f"tests/test_job_durability.py's PUBLISHERS table has {count} rows"
    )
    assert f"scans all {WORDS[count]} publishers" in para, (
        f"docs/INVARIANTS.md does not say the guard scans all {WORDS[count]} "
        f"publishers; PUBLISHERS has {count} rows"
    )


def test_invariants_raw_imgui_controls_count_matches_the_pinned_probe_constant():
    raw_imgui_controls = _from_sibling("test_probe", "RAW_IMGUI_CONTROLS")

    para = _paragraph("**The control probe is env-gated")
    stated = re.search(r"call imgui directly -- (\w+) of them", para)

    assert stated is not None, (
        "docs/INVARIANTS.md's control-probe paragraph no longer states how "
        "many controls call imgui directly"
    )
    assert stated.group(1) == WORDS[raw_imgui_controls], (
        f"docs/INVARIANTS.md says {stated.group(1)!r} controls call imgui "
        f"directly; tests/test_probe.py pins RAW_IMGUI_CONTROLS = "
        f"{raw_imgui_controls} ({WORDS[raw_imgui_controls]})"
    )
