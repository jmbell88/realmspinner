"""Regression tests for the 2026-09-26 audit's w1f7 fixer batch, for the one
finding this batch closed whose owned file lives under ``kernels/pixel``:
inker-codecs-07 (a BOM-prefixed ``.hex`` or headerless ``.gpl`` losing rows).

The rest of this batch's findings live in
``tests/modes/inker/test_audit_2026_09_26_w1f7.py``.
"""

from __future__ import annotations

from realmspinner.kernels.pixel import gpl

# -- inker-codecs-07: parse_any sniffs a BOM-stripped copy but parses the original --------


def test_a_bom_prefixed_headerless_gpl_parses_every_row():
    """``parse`` never stripped the BOM itself (unlike ``parse_jasc`` and
    ``parse_txt``, which already did), and ``parse_any`` sniffed a
    BOM-stripped copy of the text but handed the *original*, BOM-prefixed
    text to the reader it picked. ``int("\\ufeff10")`` raises -- U+FEFF is
    not whitespace to Python's own int parser -- so row 1 silently vanished
    into the same ``except ValueError: continue`` every malformed row does."""

    text = "﻿10 20 30 Name1\n40 50 60 Name2\n"
    assert gpl.parse_any(text) == [(10, 20, 30, 255), (40, 50, 60, 255)]


def test_a_bom_prefixed_hex_palette_parses_instead_of_being_refused():
    """Same root cause as the headerless ``.gpl`` case, but ``parse_hex``
    raises rather than skips a row it cannot read -- so the whole file was
    refused as "not a hex colour" instead of merely losing one row."""

    text = "﻿aabbcc\n112233\n"
    assert gpl.parse_any(text) == [(170, 187, 204, 255), (17, 34, 51, 255)]
