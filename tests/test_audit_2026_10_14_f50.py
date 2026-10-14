"""The 2026-10-14 audit, finding pipelines-26: the ruff target is the declared floor."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_ruff_target_version_matches_requires_python():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    floor = re.fullmatch(r">=\s*(\d+)\.(\d+)", cfg["project"]["requires-python"].strip())
    assert floor, cfg["project"]["requires-python"]
    expected = f"py{floor.group(1)}{floor.group(2)}"
    assert cfg["tool"]["ruff"]["target-version"] == expected
