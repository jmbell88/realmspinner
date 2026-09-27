"""service-gates-03, the 2026-09-26 audit: chapter 37's cancel-siblings claim,
pinned against the grouping the code actually cancels by.
"""

from __future__ import annotations

import re
from pathlib import Path

from realmspinner.service import sweeps

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "docs" / "manual"


def _chapter(name: str) -> str:
    return (MANUAL / name).read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """Prose with hand-wrapped line breaks collapsed to single spaces, so a
    phrase that happens to straddle two lines in the manual's own wrapping
    still matches a plain substring check."""
    return re.sub(r"\s+", " ", text)


def test_review_manual_names_the_server_axes_the_cancel_uses():
    """Chapter 37 said a failed sweep unit cancels every sibling that shares
    "the six `trellis_` flags plus resolution", but ``sweeps.server_group_of``
    groups by the seven :data:`sweeps.SERVER_AXES` values and ``resolution``
    is not one of them -- a unit differing only in resolution is not a
    sibling for this purpose, and the manual claimed the opposite."""
    assert len(sweeps.SERVER_AXES) == 7
    assert "resolution" not in sweeps.SERVER_AXES

    text = _flat(_chapter("37-review.md"))
    assert "six `trellis_` flags plus resolution" not in text
    assert "seven `trellis_` flags" in text
