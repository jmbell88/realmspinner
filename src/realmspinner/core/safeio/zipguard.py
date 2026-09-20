"""A zip reader that refuses a member claiming less than it unpacks to.

**Why this exists, measured.** Six container doors -- ``pixel/ora.py`` (.ora),
``mesh/serialize.py``, ``mason/serialize.py`` (.rscn),
``packwright/rpack.py``, ``plotter/rmap.py`` and ``sirens/rsng.py`` -- each
opened with the same precheck: sum ``info.file_size`` over the central
directory and refuse an archive claiming more than
:data:`MAX_DECOMPRESSED_BYTES` unpacked. That is the cheapest possible
refusal and it is worth keeping, but it asks the *archive* how big it is,
and an attacker writes the central directory.
``safeio/npyguard.py``'s ``read_npz`` opens a bounded zip too, one layer down,
for the ``.npz`` blobs those six doors embed inside their own archives --
shell-08 (the 2026-09-14 audit) found this docstring naming only the
original four (Mason and Sirens gained their doors later and were never
added here); the regression test derives the caller list from the tree by
grep rather than repeating a second hand-written copy of it, so it cannot go
stale the same way again. (P3 of the restructure, ``dev/RESTRUCTURE.md``,
moved ``inker/ora.py`` and ``clay/serialize.py`` out from under ``studio/``
into ``kernels/pixel/`` and ``kernels/mesh/`` respectively -- the names above
are each caller's own directory, which is what stays stable across a move
like that one, rather than a full path through whichever package layer an
engine currently sits in.)

The gap is not theoretical. A member whose directory entry declares **10 bytes**
and whose deflate stream actually inflates to 512 MiB passes the sum untouched:
``claimed`` is 10. ``zipfile`` then discovers the lie by CRC and raises
``BadZipFile`` -- *after* ``ZipExtFile.read()`` has accumulated the whole
inflated stream. Measured with ``tracemalloc`` on CPython 3.13: peak Python
allocation **1,070 MiB** for a **510 KiB** archive, against a ceiling nominally
set at 1 GiB. The refusal arrives, correctly, once the memory is already spent.

So the sum is a courtesy and this is the guarantee: every member is read through
its own declared size as a hard bound, one chunk at a time, and a stream that
runs past what its directory entry promised is refused at the byte after rather
than at the CRC. ``plotter/tmx.py``'s ``_decompress`` and ``pixel/asein.py``'s
``_inflate`` are the same idea one layer down, on a raw deflate stream.

**A subclass rather than a helper function**, which is the one design decision
here worth stating. There are dozens of ``zf.read`` call sites across those six
doors and there will be more; a ``bounded_read(zf, name)`` helper is a rule that
holds only as long as every future call site remembers it, and "remembered at
all but one site" is indistinguishable from not having the rule.
Overriding ``read`` means the bound is a property of the *archive object* the
door opened, so a new call site gets it by construction.

A shared leaf under ``core/safeio/`` for the reason ``kernels/grid2d`` (né
``tilegrid``) and ``core/undo`` are shared leaves: several engines needed one
rule, and another copy of a security bound is the kind of thing that drifts
without any of them noticing. (P3 of the restructure, ``dev/RESTRUCTURE.md``,
moved this module itself out from under ``studio/`` to sit beside the other
shared, no-GL kernels the engines it guards were being moved out from under
``studio/`` to reach.)
"""

from __future__ import annotations

import zipfile

#: The absolute ceiling on any one member's unpacked size. The constant the four
#: doors already carried, now in one place; each still re-exports it under its
#: own name so their existing refusal messages are unchanged.
MAX_DECOMPRESSED_BYTES = 1 << 30

#: Read granularity. Peak allocation for one member is its declared size plus
#: this, so it trades a bounded overshoot for not calling ``read`` a million
#: times on a large layer.
_CHUNK = 1 << 20


class BoundedZip(zipfile.ZipFile):
    """A ``ZipFile`` whose ``read`` will not outrun the directory's own promise.

    Every other ``ZipFile`` method is inherited untouched -- ``infolist``,
    ``namelist`` and ``getinfo`` only ever read the central directory, which is
    already in memory by the time the constructor returns.
    """

    #: Per-instance so a test can lower it rather than building a gigabyte --
    #: the rule the four doors already stated about their own copies.
    ceiling: int = MAX_DECOMPRESSED_BYTES

    def read(self, name, pwd=None) -> bytes:  # type: ignore[override]
        info = name if isinstance(name, zipfile.ZipInfo) else self.getinfo(name)
        declared = int(info.file_size)
        if declared < 0 or declared > self.ceiling:
            raise ValueError(
                f"{info.filename!r} declares {declared} bytes unpacked, past the"
                f" {self.ceiling} this build will read"
            )
        out = bytearray()
        with self.open(info, "r", pwd) as fh:
            while True:
                # ``declared - len(out) + 1`` and never simply ``_CHUNK``: the
                # last read is deliberately allowed to fetch one byte more than
                # the promise, because that byte is the whole test. Reading
                # exactly ``declared`` would leave a lying archive
                # indistinguishable from an honest one until the CRC check that
                # this class exists to get in front of.
                want = min(_CHUNK, declared - len(out) + 1)
                if want <= 0:
                    break
                block = fh.read(want)
                if not block:
                    break
                out += block
                if len(out) > declared:
                    raise ValueError(
                        f"{info.filename!r} unpacks past the {declared} bytes its"
                        " directory entry declares"
                    )
        return bytes(out)
