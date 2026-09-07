"""Regression for the 2026-09-06 audit, finding docs-14.

The "Insect" figure preset's label read "Insect / spider (six-legged)", which
promises an eight-legged spider the template does not provide -- ``insect.json``
rigs a six-legged insect. Whether a genuine spider template is wanted is a
separate, still-open decision; this only checks the label stops naming a
subtype the template does not build.
"""

from __future__ import annotations

import json
from pathlib import Path

from warlock.studio.clay import presets

TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "warlock" / "templates" / "insect.json"
)
CLAY_TOOLS_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "warlock"
    / "studio"
    / "panes"
    / "clay_tools.py"
)


def test_insect_figure_label_no_longer_names_the_spider_subtype() -> None:
    label, _builder = presets.ASSEMBLIES["insect"]
    assert label == "Insect"
    assert "spider" not in label.lower()

    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert template["label"] == "Insect"
    assert template["key"] == "insect"  # only the human-facing label changes

    clay_tools_source = CLAY_TOOLS_PATH.read_text(encoding="utf-8")
    assert "spider" not in clay_tools_source.lower()
