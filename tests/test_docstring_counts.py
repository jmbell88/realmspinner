"""Regression test for the 2026-09-06 audit, finding docs-23.

``attach_files``'s docstring quoted ``LISTED`` as "eleven names" to justify the
per-row stat cost it states right after; ``LISTED`` actually held fifteen (the
original nine plus ``track.wav``, the four ``STEM_FILES`` entries and
``error.log``), understating that cost by nearly 40%. The word is derived from
``files.LISTED`` itself here so the docstring cannot drift out from under the
count again without this test noticing.
"""

from realmspinner.service import files

_NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}


def test_attach_files_docstring_matches_the_listed_table():
    count = len(files.LISTED)
    word = _NUMBER_WORDS[count]
    doc = files.attach_files.__doc__ or ""
    assert word in doc, (
        f"attach_files docstring does not say {word!r}, but LISTED holds "
        f"{count} names -- the docstring has drifted again (docs-23)"
    )
