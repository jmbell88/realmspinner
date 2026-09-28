"""Regression for the 2026-09-26 audit's clay-document-09 (fixer w6f4, the
comment half): ``_normalize_seams``'s own docstring named a "scratch clone"
mechanism as one reason a call site has no trustworthy mesh size yet --
Familiar's ``kernels/mesh/scratch.py``, removed the same day Familiar itself
was (2026-09-26). Left in place, the comment pointed a future reader at a
module that no longer exists.
"""

from __future__ import annotations

from realmspinner.kernels.mesh import document as bd


def test_normalize_seams_docstring_no_longer_cites_familiars_removed_scratch_clone() -> None:
    doc = (bd._normalize_seams.__doc__ or "")
    assert "scratch clone" not in doc, (
        "the docstring must not point at Familiar's removed scratch-clone mechanism"
    )
