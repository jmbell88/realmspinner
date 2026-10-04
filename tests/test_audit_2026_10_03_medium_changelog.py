"""The 2026-10-03 audit, finding docs-11: the 0.0.54 note quoted a mid-run count.

The entry said the 2026-09-26 audit's fix pass "has closed 319 of its 463
findings". 517e4fb2 closed 181 more to reach 319; 6c5ffa90 then closed 141
more before the release commit f8cb9354, so the figure the release shipped
(and Home's What's new lead, which quotes the entry's opening sentence) was
about 140 findings short. 319 + 141 is the total the commit messages hold.
"""

from __future__ import annotations

import re
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"

WAVE_1_TO_5 = 319  # 517e4fb2: "319 of 463 total"
WAVES_6_TO_8 = 141  # 6c5ffa90: "closes 141 more findings"
TOTAL = 463


def test_changelog_audit_fix_pass_count_names_the_final_total():
    text = CHANGELOG.read_text(encoding="utf-8")
    entry = re.search(
        r"2026-09-26 audit's fix pass (?:has )?closed (\d+) of its (\d+) findings", text
    )
    assert entry, "the 0.0.54 entry no longer states the audit fix pass's count"
    closed, total = int(entry.group(1)), int(entry.group(2))
    assert total == TOTAL
    assert closed == WAVE_1_TO_5 + WAVES_6_TO_8, (
        f"the entry says {closed} closed; the commit messages sum to "
        f"{WAVE_1_TO_5 + WAVES_6_TO_8} (319 from 517e4fb2 plus 141 from 6c5ffa90)"
    )
