"""The three network-touching child processes describe themselves consistently.

The 2026-09-14 audit (pipelines-06): ``fetch_worker.py`` opened "The one
process in this project that is allowed to touch the network" -- true when it
was the only one, false since ``pack_worker`` (2026-09-04, dependency packs)
and ``update_worker`` (the release-feed/installer worker) were added, each of
which correctly names itself "the second"/"the third". Nothing caught the
stale claim because no test read these docstrings against each other.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "warlock" / "pipelines"


def _module_docstring(name: str) -> str:
    tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
    doc = ast.get_docstring(tree)
    assert doc, f"{name} has no module docstring"
    return doc


def test_fetch_worker_docstring_does_not_claim_to_be_the_only_network_process():
    doc = _module_docstring("fetch_worker.py")
    assert "the one process in this project that is allowed" not in doc.lower()
    assert "first of three" in doc.lower()


def test_all_three_network_worker_docstrings_agree_on_their_own_ordinal():
    fetch_doc = _module_docstring("fetch_worker.py").lower()
    pack_doc = _module_docstring("pack_worker.py").lower()
    update_doc = _module_docstring("update_worker.py").lower()
    assert "first of three" in fetch_doc
    assert "second process allowed to touch the network" in pack_doc
    assert "third process allowed to touch the network" in update_doc
