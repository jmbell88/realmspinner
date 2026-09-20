"""Every hand-spelled ``tour_pane.start(ctx, "...")`` call names a real tour.

``panes/tour.py``'s own docstring says ``start`` ignores an unknown key rather
than raising -- deliberately, because a key restored from saved settings must
not take the frame down if a tour was renamed between releases. That mercy
has a cost the 2026-09-18 audit named (tour-01): a call site that *hand-spells*
the key, rather than naming ``Tour.key`` off the real ``Tour`` object, gets the
same silent nothing a renamed tour gets if the literal is ever wrong -- the
button just stops opening a tour, with no test and no toast to say why.
``sirens-basics`` (``sirens/ui/panes/patterns.py``) and ``first-hour``
(``panes/first_run.py``) are both spelled correctly today; this is the sweep
that keeps them that way, and it also catches the next hand-spelled call site
rather than needing a per-site test.
"""

from __future__ import annotations

import ast
from pathlib import Path

from realmspinner.studio.tour.scripts import TOURS

SRC = Path(__file__).resolve().parents[3] / "src" / "realmspinner"

REAL_TOUR_KEYS = {tour.key for tour in TOURS}


def _hardcoded_start_calls(path: Path) -> list[tuple[int, str]]:
    """Every ``tour_pane.start(ctx, "<literal>")`` in *path*, as (line, key).

    Scoped to the ``tour_pane`` alias every call site in this tree uses for
    ``from ...panes import tour as tour_pane`` (or the sibling-relative
    spelling), so a coincidental ``.start(x, "y")`` on an unrelated object --
    a thread, a job -- is never mistaken for a tour key.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "start":
            continue
        if not isinstance(func.value, ast.Name) or func.value.id != "tour_pane":
            continue
        key_arg = node.args[1]
        if isinstance(key_arg, ast.Constant) and isinstance(key_arg.value, str):
            found.append((node.lineno, key_arg.value))
    return found


def test_the_sweep_finds_the_known_hardcoded_call_sites():
    """Guard on the guard: an empty result would pass the real test vacuously,
    the way an empty module list would have for ``test_tour_imports.py``."""
    offenders = [
        (path.relative_to(SRC).as_posix(), n, key)
        for path in sorted(SRC.rglob("*.py"))
        for n, key in _hardcoded_start_calls(path)
    ]
    assert len(offenders) >= 2, offenders


def test_every_hardcoded_tour_start_call_names_a_real_tour_key():
    offenders = [
        f"{path.relative_to(SRC).as_posix()}:{n} -> {key!r}"
        for path in sorted(SRC.rglob("*.py"))
        for n, key in _hardcoded_start_calls(path)
        if key not in REAL_TOUR_KEYS
    ]
    assert offenders == []
