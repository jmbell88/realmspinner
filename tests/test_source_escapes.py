"""No module under ``src/realmspinner`` carries an invalid string escape.

Python 3.12 turned ``"\\s"`` in a non-raw string from a silent
DeprecationWarning into a SyntaxWarning, and a later release makes it a
SyntaxError -- at which point the module stops importing. The vendored
ACE-Step ``lyric_normalizer.py`` held ``re.compile("(?<!^)\\s+(?!$)")`` from
Muse's first commit until 2026-09-14: correct today only because an unknown
escape is kept verbatim, and invisible because the warning fires once, at
bytecode compile, inside a child process nobody reads the stderr of.
"""

from __future__ import annotations

import warnings
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "realmspinner"


def test_no_module_under_src_has_an_invalid_escape_sequence():
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntaxWarning)
            try:
                compile(path.read_bytes(), str(path), "exec")
            except SyntaxError as exc:
                offenders.append(f"{path.relative_to(SRC)}:{exc.lineno}: {exc.msg}")
    assert not offenders, offenders
