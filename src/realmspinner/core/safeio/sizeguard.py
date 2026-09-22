"""One "is this file small enough to open" question, for every mode that opens one.

``plotter_io`` and ``packwright_io`` each carried a private ``_within_ceiling``
-- correct, well argued, and wired only to the door it sat next to. Clay had
none at all, though ``service.files.MAX_CLAY_SOURCE_BYTES`` has existed since
the format did and is applied at exactly one place, the *upload*; Inker read a
``.aseprite`` and two ``.gpl`` files with no ceiling either.

That is ``zipguard``'s finding again: the number exists, the rule exists, and
the rule holds at the call sites that remember it. A shared helper cannot make
a caller remember, but it can make remembering cost one line and make the scan
test that checks for it possible to write -- which is the pair
``atomic.staged`` and its ``dialogs.save_file`` scan already form for writes.

**A ``ServiceError``, not a ``ValueError``.** This is the *mode's* refusal about
a file the user picked, not an engine's about a document's contents, and its
text reaches the user verbatim through the task classifier. The service layer is
imported inside the function so a mode pays nothing for it until it opens a
file, and the ceiling is read at call time so a test lowers it rather than
building half a gigabyte.
"""

from __future__ import annotations

from pathlib import Path


def within_ceiling(path: Path, ceiling: int, *, field: str = "file") -> Path:
    """Refuse a file past *ceiling* bytes, before a byte of it is read.

    Returns the path so a caller reads ``within_ceiling(p, N).read_bytes()`` in
    one expression -- the shape both private copies already had, kept because a
    helper whose result is easy to drop on the floor is a helper that gets
    called and ignored.

    **shell-07 (the 2026-09-18 audit):** this only checks ``stat()`` at the
    call, and every caller in the tree reads the shape
    ``within_ceiling(p, N).read_bytes()`` -- two syscalls, not one, so a file
    that grows between the ``stat`` and the ``read_bytes`` (another process
    still writing it, a symlink swapped underfoot) sails the ceiling on the
    read it was meant to bound. :func:`read_bytes_within_ceiling` below closes
    that window by bounding the read itself rather than trusting a stat taken
    a moment earlier; existing callers keep working unchanged; a new one, or
    one revisited for this class of bug, should prefer it.
    """
    from ...service.errors import TooLarge

    if Path(path).stat().st_size > ceiling:
        raise TooLarge(
            f"{Path(path).name} is past the {ceiling} bytes this build will open",
            field=field,
        )
    return Path(path)


def read_bytes_within_ceiling(path: Path, ceiling: int, *, field: str = "file") -> bytes:
    """Read *path*, refusing past *ceiling* bytes -- without the stat/read race.

    ``within_ceiling(p, N).read_bytes()`` is two separate syscalls: the size
    checked by ``stat()`` is not the size actually read a moment later, so a
    file that grows in between (a concurrent writer, a swapped symlink) can
    still land more than *ceiling* bytes in memory even though the guard
    "passed" -- shell-07, the 2026-09-18 audit. This reads at most
    ``ceiling + 1`` bytes in one call and refuses if that many came back,
    which bounds the read itself rather than trusting an earlier stat.
    """
    from ...service.errors import TooLarge

    p = Path(path)
    with p.open("rb") as f:
        data = f.read(ceiling + 1)
    if len(data) > ceiling:
        raise TooLarge(
            f"{p.name} is past the {ceiling} bytes this build will open",
            field=field,
        )
    return data
