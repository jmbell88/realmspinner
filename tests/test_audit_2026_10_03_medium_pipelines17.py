"""The 2026-10-03 audit's pipelines-17 (Medium): no document claims remote code runs."""

from __future__ import annotations

from pathlib import Path


def test_the_manual_and_pyproject_do_not_claim_birefnet_runs_remote_code():
    root = Path(__file__).resolve().parents[1]
    for rel in ("docs/manual/40-installation.md", "pyproject.toml"):
        text = (root / rel).read_text(encoding="utf-8")
        assert "trust_remote_code" not in text, (
            f"{rel} says BiRefNet runs downloaded modelling code; it has been vendored "
            "at pipelines/birefnet/ since MDL-03 and src/ never passes the flag"
        )
