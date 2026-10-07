"""The old product name (Warlock Studio) stays out of the tree.

Checked in both directions: a tracked file that names it fails, and so does an
allowance that is no longer needed. The upgrade path off the old name was
removed on 2026-09-23 (no install of it remains), so what is left here is only
the guard and the agent pipe's address.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Where the old name survives on purpose, and why. Every entry is a record of
# something that already happened or a door held open for it -- never a
# description of what the app is now. Checked in both directions below, so an
# entry that stops being needed fails just as loudly as a file that starts
# spelling the old name: a hand list that only fails open is how ``PUBLISHERS``
# went stale twice.
_ALLOWED: dict[str, str] = {
    "CHANGELOG.md": "entries below 0.0.52 describe releases published under that name",
    "tests/kernels/geom3d/test_glbwrite.py": "records why the pinned glTF bytes moved",
    "tests/modes/inker/test_cel_z.py": "records why the pinned .ora bytes moved",
    "tests/test_old_name_gone.py": "this file",
}

_NAME = re.compile(rb"[Ww]arlock|WARLOCK")


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def _names_the_old_product(rel: str) -> bool:
    path = ROOT / rel
    if not path.is_file():
        return False
    return bool(_NAME.search(path.read_bytes()))


def test_the_old_product_name_is_gone_from_every_tracked_file():
    """The guard that stops the old name creeping back in.

    A rename this size lands as one scripted sweep, and the way a sweep like
    that fails is not loudly -- it is one file that was open in an editor, or a
    path the file list did not reach, sitting there spelling the old name at a
    user months later. Checked over ``git ls-files`` rather than a directory
    walk, so ``dist/``, ``.venv/`` and the gitignored maintainer-only trees
    cannot make it pass or fail by accident. Binary files are checked too: the
    shipped character meshes carried the product in their glTF ``generator``.
    """
    offenders = [
        rel
        for rel in _tracked_files()
        if not any(rel.replace("\\", "/").startswith(a) for a in _ALLOWED)
        and _names_the_old_product(rel)
    ]
    assert not offenders, (
        "the old product name survives in:\n  "
        + "\n  ".join(offenders)
        + "\n\nRename it, or -- if it is a record of something that already "
        "happened -- add it to _ALLOWED with the reason."
    )


def test_every_allowance_for_the_old_name_is_still_needed():
    """The other direction, so the list above cannot quietly go stale.

    An exemption that no longer matches anything is worse than no exemption:
    it is a hole the next sweep falls through silently.
    """
    tracked = [rel.replace("\\", "/") for rel in _tracked_files()]
    for prefix, reason in _ALLOWED.items():
        matched = [rel for rel in tracked if rel.startswith(prefix)]
        assert matched, f"_ALLOWED names {prefix!r}, which matches no tracked file"
        assert any(_names_the_old_product(rel) for rel in matched), (
            f"_ALLOWED still exempts {prefix!r} ({reason}), but nothing under it "
            "names the old product any more -- drop the entry"
        )


@pytest.mark.skipif(os.name != "nt", reason="the pipe name is the Windows address")
def test_the_agent_pipe_no_longer_answers_to_the_old_name():
    """An external surface: an agent dials this name, so it had to move too."""
    from realmspinner.mcp import pipe, rpc

    address = pipe.address_for(Path(sys.prefix))
    assert "realmspinner-mcp-" in address
    assert "warlock" not in address.lower()
    assert rpc.SERVER_NAME == "realmspinner"
