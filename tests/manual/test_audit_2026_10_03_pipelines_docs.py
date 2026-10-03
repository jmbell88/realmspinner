"""Regression tests for the 2026-10-03 audit's pipelines-02 and pipelines-03
(manual sentences that had gone stale against the code)."""

from __future__ import annotations

import re
from pathlib import Path

MANUAL = Path(__file__).resolve().parents[2] / "docs" / "manual"


def _flat(name: str) -> str:
    return re.sub(r"\s+", " ", (MANUAL / f"{name}.md").read_text(encoding="utf-8"))


def test_manual_22_does_not_promise_a_template_for_the_image_type() -> None:
    """Since prompt policy 9 an Image reference is the bare prompt (prompt.build)."""
    text = _flat("22-generating-references")
    assert "studio render: the app wraps" not in text
    assert "Your text is composed into a fixed template before the image model sees it" not in text
    assert "Image** type is the exception" in text
    assert "no template" in text


def test_the_manual_does_not_say_the_installer_carries_the_reconstruction_engine() -> None:
    install = _flat("40-installation")
    settings = _flat("42-app-settings")
    assert "its renderer and the reconstruction engine" not in install
    assert "its renderer and the reconstruction engine" not in settings
    assert "reconstruction engine is a Settings" in install
    assert "reconstruction engine is a Models download" in settings
