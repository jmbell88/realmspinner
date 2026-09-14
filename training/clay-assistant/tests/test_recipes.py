"""Every ``drafts/_gen_<family>.py`` recipe must reproduce its own committed
``drafts/<family>.jsonl`` byte-for-byte -- the README's own "Authoring in
parallel" promise ("Regenerating a family must reproduce its JSONL
byte-for-byte") and the whole reason each recipe seeds its own randomness
from a stable string (``FAMILY-<template>-<phrasing>``) rather than leaving
it to chance.

Not part of the main suite -- see ``tests/test_scaffold.py``'s own docstring
for why (``pyproject.toml``'s ``testpaths`` is ``["tests"]``):

    uv run pytest training/clay-assistant/tests/test_recipes.py -n 0 -p no:cacheprovider

Each recipe is loaded by its file path via ``importlib`` (never a bare
``import _gen_x``, since ``drafts/`` is a sibling scratch package of several
authors' own in-flight files sharing generic module names -- see
``test_scaffold.py::test_gen_creatures_part_names_match_the_live_presets_registry``
for the same convention) with the drafts directory pushed onto ``sys.path``
first: ``_gen_figures.py`` does ``from _gen_creatures import PART_NAMES``, a
bare sibling import that only resolves once ``drafts/`` is on the path.

A recipe that does not expose the ``OUT_PATH``/``main()`` shape this test
assumes is reported, not special-cased: ``monkeypatch.setattr(module,
"OUT_PATH", ..., raising=True)`` raises loudly if the module has no such
attribute, rather than this test falling back to guess another name (four
recipes -- ``_gen_containers.py``, ``_gen_edits.py``, ``_gen_furniture.py``,
``_gen_mechanical.py`` -- write to a plain module-level ``OUT`` instead of
``OUT_PATH`` as of 2026-09-13; that mismatch is exactly what this test is
supposed to surface).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # training/clay-assistant
DRAFTS = _PACKAGE_ROOT / "drafts"

if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from gen import schema  # noqa: E402

RECIPE_PATHS = sorted(DRAFTS.glob("_gen_*.py"))
RECIPE_IDS = [p.stem.removeprefix("_gen_") for p in RECIPE_PATHS]


def _load_recipe(path: Path) -> ModuleType:
    """Load *path* as a standalone module, with ``drafts/`` on ``sys.path``
    so a recipe's own sibling import (``_gen_figures.py`` imports
    ``_gen_creatures``) resolves. Loaded under a name distinct from the file's
    own stem so it never collides with a real ``sys.modules`` entry a sibling
    import creates for the same file."""
    if str(DRAFTS) not in sys.path:
        sys.path.insert(0, str(DRAFTS))
    spec = importlib.util.spec_from_file_location(f"{path.stem}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", RECIPE_PATHS, ids=RECIPE_IDS)
def test_recipe_reproduces_its_committed_jsonl_byte_for_byte(
    path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = path.stem.removeprefix("_gen_")
    module = _load_recipe(path)

    assert hasattr(module, "OUT_PATH"), (
        f"{path.name} has no module-level OUT_PATH -- it does not follow the "
        "OUT_PATH/main() shape this test assumes (check for a differently "
        "named output path, e.g. a plain OUT)"
    )
    assert callable(getattr(module, "main", None)), f"{path.name} has no callable main()"

    tmp_out = tmp_path / f"{family}.jsonl"
    # raising=True (the default): fails loudly if OUT_PATH is not actually
    # the attribute the recipe writes through, rather than silently letting
    # main() write over the real drafts/<family>.jsonl.
    monkeypatch.setattr(module, "OUT_PATH", tmp_out, raising=True)
    module.main()

    committed = DRAFTS / f"{family}.jsonl"
    assert committed.is_file(), (
        f"drafts/{family}.jsonl does not exist yet (the recipe itself ran "
        "fine) -- an author may still be writing this family; that is a "
        "real gap to report, not something for this test to skip over"
    )
    assert tmp_out.read_bytes() == committed.read_bytes(), (
        f"drafts/{family}.jsonl is stale: re-running {path.name} no longer "
        "reproduces the committed file byte-for-byte"
    )


def test_every_schema_family_has_both_a_recipe_and_a_committed_jsonl() -> None:
    missing_recipe = [f for f in schema.FAMILIES if not (DRAFTS / f"_gen_{f}.py").is_file()]
    missing_jsonl = [f for f in schema.FAMILIES if not (DRAFTS / f"{f}.jsonl").is_file()]

    assert not missing_recipe, (
        f"families in schema.FAMILIES with no drafts/_gen_<family>.py recipe: {missing_recipe}"
    )
    assert not missing_jsonl, (
        f"families in schema.FAMILIES with no committed drafts/<family>.jsonl: {missing_jsonl}"
    )
