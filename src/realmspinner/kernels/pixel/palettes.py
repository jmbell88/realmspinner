"""Named colour tables a caller can hand to ``Document.set_palette``.

Constants only, and nothing imported: a table is data, and the document that
adopts it is what knows how to snap onto one.
"""

from __future__ import annotations

# PICO-8's sixteen colours in its own numbering (0 = black ... 15 = peach), as
# opaque RGBA. The order is the console's, not sorted: a game's sprite sheet
# refers to a colour by that number, so a palette in another order would
# round-trip into the wrong picture.
PICO8: tuple[tuple[int, int, int, int], ...] = tuple(
    (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
    for h in (
        "000000",
        "1D2B53",
        "7E2553",
        "008751",
        "AB5236",
        "5F574F",
        "C2C3C7",
        "FFF1E8",
        "FF004D",
        "FFA300",
        "FFEC27",
        "00E436",
        "29ADFF",
        "83769C",
        "FF77A8",
        "FFCCAA",
    )
)
