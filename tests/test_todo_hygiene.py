"""Regressions for the 2026-09-06 docs audit's TODO.md findings (docs-15 to docs-18).

``TODO.md:28`` states the file's own rule: "Entry numbers are stable... a
closed entry's number is not reused." The audit found four ways the file had
drifted from its own claims -- two duplicated ``## P<n>`` headings, one
number quietly reused across a closed record and an open entry, a struck
finding still cited as an open gap, a code step described as owed after it
shipped, and a branch-turned-tag still called a branch. Each check here reads
the live file rather than a frozen copy, so a later edit that reintroduces
one of these mistakes fails it again.
"""

from __future__ import annotations

import re
from pathlib import Path

TODO = Path(__file__).resolve().parents[1] / "TODO.md"


def _text() -> str:
    return TODO.read_text(encoding="utf-8")


def _entry_numbers(text: str) -> list[int]:
    """Every ``## P<n>`` heading's number, in file order, duplicates kept."""
    return [int(n) for n in re.findall(r"(?m)^## P(\d+)\.", text)]


# --- docs-15: entry numbers are not unique -----------------------------------

def test_todo_entry_numbers_are_unique():
    """No ``## P<n>`` heading number may appear twice.

    Before the fix this failed with P30 heading both the 2D walk-cycle
    verdict (:839) and the Inker pane-height decision (:936), and P31 heading
    both the sweep-abort spec (:354) and the Clay judgement entry (:976).
    """
    numbers = _entry_numbers(_text())
    seen: set[int] = set()
    dupes: set[int] = set()
    for n in numbers:
        if n in seen:
            dupes.add(n)
        seen.add(n)
    assert not dupes, f"duplicated ## P<n> headings: {sorted(dupes)}"


def test_closed_p3_number_is_not_reused_by_an_open_entry():
    """P3 is closed (the graded mesh run, ``Closed records``); no open
    ``## P3.`` heading may exist alongside it -- that is the reuse
    ``TODO.md:28`` forbids, which the tex-res-pin entry committed by staying
    numbered P3 after the graded mesh run had already closed under that
    number.
    """
    text = _text()
    assert "## P3." not in text
    assert "**P3, the graded mesh run.**" in text, (
        "the closed record itself must keep citing P3"
    )


# --- docs-16: P22 cites the closed F4 pack-gate finding as a live gap -------

def test_p22_no_longer_cites_the_closed_pack_gate_finding():
    """P22 must not cite F4 as an open gap -- F4 (:1093-1100 in the unfixed
    file) is struck as "Built 2026-09-05" in the same file, and
    ``model_gate.mode_gate``/``mode_reason`` already sends a user with
    missing packs to Packs first and names what is blocked
    (``tests/test_pack_gate.py``).
    """
    text = _text()
    p22_start = text.index("## P22.")
    p22_end = text.index("\n## P23.")
    p22 = text[p22_start:p22_end]
    assert "(F4)" not in p22
    assert "F4" not in p22


# --- docs-17: P17 describes a shipped branch as an owed code step ----------

def test_p17_no_longer_describes_the_direction_selector_step_as_unbuilt():
    """P17 must not describe the readonly-label-vs-combo branch as a step
    still owed -- ``settings_2d.py`` already branches ``form_ui.readonly``
    under two discovered counts and ``form_ui.segmented_choice`` otherwise.
    The mention may stay as a struck, closed step; it may not stand live.
    """
    text = _text()
    p17_start = text.index("## P17.")
    p17_end = text.index("\n## P19.")
    p17 = text[p17_start:p17_end]
    assert "~~**The one step that is code" in p17, "the step must be struck"
    assert "Built" in p17


# --- docs-18: P11 calls the archived plotter-wave-2 tag a branch -----------

def test_p11_describes_plotter_wave_2_as_an_archived_tag():
    """P11 must call ``plotter-wave-2`` an archived tag, not "the branch",
    and must not offer deleting a branch that no longer exists -- it became
    ``refs/tags/archive/plotter-wave-2`` at the commit the entry itself
    cites (d1995fad, 2026-08-14).
    """
    text = _text()
    p11_start = text.index("## P11.")
    p11_end = text.index("\n## P13.")
    p11 = text[p11_start:p11_end]
    assert "archive/plotter-wave-2" in p11 or "archived tag" in p11
    assert "**A branch delete needs an explicit ask.**" not in p11
    assert "The branch last moved" not in p11
