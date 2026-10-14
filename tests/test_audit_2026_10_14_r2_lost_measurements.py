"""Every ``dev/measurements/<date>-...`` citation in ``src/`` that names a lost document says so.

The backup that held ``dev/measurements/`` is gone (confirmed 2026-09-28), and every document dated
before 2026-10-03 went with it. About a hundred citations in ``src/`` still name those files; a
reader who follows one goes looking for a file that does not exist and may take its absence for an
omission. The 2026-10-14 audit (finding docs-11) put a "lost; re-measure" note beside each, and
this gate keeps the next citation honest.

The rule has two halves so it also holds on a public checkout, where ``dev/`` is absent: a date
before the cut-off is lost whether or not ``dev/`` is here to say so, and when ``dev/`` is present a
cited name that does not exist under it is lost whatever its date.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
MEASUREMENTS = ROOT / "dev" / "measurements"

# Documents dated before this day went with the lost backup.
FIRST_SURVIVING_DAY = "2026-10-03"

# ``dev/measurements/2026-08-06-pixel-art-xl.md``, and the same citation wrapped after the slash
# with a comment leader on the next line (``# dev/measurements/`` / ``# 2026-08-06-...``).
_CITATION = re.compile(
    r"dev/measurements/[ \t]*(?:\r?\n[ \t]*(?:#:?|\*|//)?[ \t]*)?(\d{4}-\d{2}-\d{2})([\w.\-]*)"
)
_NOTE = re.compile(r"\blost\b|re-?measure", re.IGNORECASE)
_WINDOW = 5


def _exists(date: str, tail: str) -> bool:
    name = (date + tail).rstrip(".-")
    return any(p.name.startswith(name) for p in MEASUREMENTS.iterdir())


def _citations():
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in {".py", ".md", ".c", ".h", ".txt", ".json", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for m in _CITATION.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            yield path, lineno, m.group(1), m.group(2), lines


def test_every_lost_measurement_citation_says_so():
    have_dev = MEASUREMENTS.is_dir()
    bare = []
    seen = 0
    for path, lineno, date, tail, lines in _citations():
        seen += 1
        lost = date < FIRST_SURVIVING_DAY or (have_dev and not _exists(date, tail))
        if not lost:
            continue
        lo = max(0, lineno - 1 - _WINDOW)
        hi = min(len(lines), lineno + _WINDOW)
        if not any(_NOTE.search(ln) for ln in lines[lo:hi]):
            bare.append(f"{path.relative_to(ROOT).as_posix()}:{lineno}: {date}{tail}")
    assert seen, "the citation pattern matched nothing: the gate has gone blind"
    assert not bare, (
        "these cite a dev/measurements document that no longer exists without saying so "
        '(add "document lost with the 2026-09-20 backup; re-measure before changing this"):\n'
        + "\n".join(bare)
    )
