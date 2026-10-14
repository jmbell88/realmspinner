"""Chapter labels against the controls that actually draw them.

A sibling of ``test_manual_labels.py``'s idea, in its own module because this
audit's finding (docs-03) is the first one filed here rather than against the
2026-09-06 batch that file documents.
"""

from __future__ import annotations

from pathlib import Path

MANUAL = Path(__file__).resolve().parents[2] / "docs" / "manual"


def _read(key: str) -> str:
    return (MANUAL / f"{key}.md").read_text(encoding="utf-8")
