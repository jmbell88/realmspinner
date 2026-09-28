"""Muse's headless half: the arithmetic behind the player.

``studio/sirens/``, ``studio/inker/``, ``studio/clay/`` -- the sixth instance of
the same shape, and it earns it for the same reason they do. ``muse_mode`` may
import ``service`` freely because it is a controller; this package may not,
because what it computes is a picture and a pair of sample offsets, and neither
of those is a question about jobs.

**No document, unlike every other headless package here.** ``sirens/`` owns a
``.rsng`` and reaches for ``core.undo`` and the two container guards; there is
nothing here to undo, so the outward-import set is *empty*. A take is a file a
worker wrote, and the store owns it.

**``scipy`` is banned, and the argument is ``sirens/wavout``'s, one module
further out.** An exported loop is an *artifact*: the crossfade at its seam is
samples this code wrote, and ``wavout``'s byte-identity rule reaches every one
of them. A filter whose coefficients come from a dependency is a filter that can
change under a ``uv sync``, which would make two exports of the same take and
the same markers differ. The decimator here is written out for that reason and
no other.

The one deliberate consequence: ``fileio.read_track`` (``muse_mode._read_track``,
under the name a test patches) may go on using ``scipy.signal.resample_poly``.
It retimes the in-memory buffer handed to the mixer, and it lives outside this
package precisely so that stays true.

**"nothing exported depends on it" needed a second half, not just this
paragraph asserting it (the 2026-09-26 audit, finding muse-engine-02).** A
legacy-rate take -- anything below ``sirens_audio.RATE``, which is only ever
an older take, since ``REALMSPINNER 5/6`` writes 44.1 kHz -- had
``fileio.export_loop``/``export_with_points`` write straight from that same
resampled buffer, so its export *was* scipy's arithmetic wearing this
package's byte-identity guarantee. Both functions now read the take's own
file again, at its own rate, whenever it differs from the player's (see
``fileio._export_source``); the resample stays exactly what this paragraph
always said it was, a decode-time convenience for the mixer, because export
no longer reaches through it.
"""

from __future__ import annotations

__all__: list[str] = []
