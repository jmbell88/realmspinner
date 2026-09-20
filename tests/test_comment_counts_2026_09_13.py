"""Comment-drift regressions from the 2026-09-13 audit (shell-12, shell-14,
shell-15): three prose comments each stated the size of a tuple/expression
in words, and each had drifted from the actual count after the collection it
described grew. Each check reads the source, finds the comment immediately
above the declaration, normalises line-wrapped ``\\n# ``/``\\n#: `` breaks to
a single space, and asserts no stale number word survives -- robust to
rewording, since the fix here is to drop the number rather than restate it,
but also catching a future comment that names a number without matching the
real count.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "realmspinner" / "studio"

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
}


def _normalise(comment: str) -> str:
    """Join a wrapped comment block into one line, the way a reader does."""

    return re.sub(r"\n\s*#:?\s?", " ", comment)


def _count_words_before(comment: str, noun: str) -> set[str]:
    """Number words that directly modify *noun* in this comment (e.g. "the
    five document modes", "is thirteen\\nrows") -- not any incidental
    occurrence of a number word elsewhere in the prose. A plain whole-comment
    word search flagged "one"/"two" inside unrelated sentences here, which is
    not what shell-12/14/15 are about: each finding is specifically a stale
    count word sitting right next to the thing it counts."""

    text = _normalise(comment)
    return {
        w
        for w in _NUMBER_WORDS
        if re.search(rf"\b{w}\s+{noun}\b", text, re.IGNORECASE)
    }


def _leading_comment_block(lines: list[str], decl_lineno: int) -> str:
    """The contiguous ``#``/``#:`` comment lines directly above a declaration
    (1-indexed ``decl_lineno``, as AST reports it)."""

    i = decl_lineno - 2  # 0-indexed line just above the declaration
    block: list[str] = []
    while i >= 0 and lines[i].lstrip().startswith("#"):
        block.insert(0, lines[i])
        i -= 1
    return "\n".join(block)


def test_doc_modes_is_derived_rather_than_a_second_hand_written_tuple():
    """shell-12 (the 2026-09-13 audit) guarded a stale count in the comment
    above a hand-written ``DOC_MODES`` tuple ("five" when the tuple held
    six). shell-05 (the 2026-09-20 audit) replaced that tuple with a live
    derivation off ``mode_manifest.DOC_MODES`` (``docmodes._doc_modes``,
    read through a module ``__getattr__`` so every existing
    ``docmodes.DOC_MODES`` call site keeps working) precisely because a
    second hand-written copy is what let it drift in the first place -- so
    there is no longer a literal tuple for a comment to go stale against.
    This asserts that stays true, rather than scanning a comment that no
    longer exists.
    """

    path = SRC / "docmodes.py"
    tree = ast.parse(path.read_text())
    literal = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "DOC_MODES"
    ]
    assert not literal, (
        "DOC_MODES is a hand-written tuple again in docmodes.py -- derive it "
        "from mode_manifest.DOC_MODES instead (the 2026-09-20 audit, shell-05)"
    )

    from realmspinner.studio import docmodes

    assert set(docmodes.DOC_MODES) == {
        "inker",
        "clay",
        "mason",
        "plotter",
        "packwright",
        "sirens",
    }


def test_new_items_comment_count_matches_its_own_length():
    """shell-15: the comment above ``NEW_ITEMS`` no longer names a stale
    length ("eight" when the tuple held nine)."""

    path = SRC / "modes/home/ui/panes/landing.py"
    lines = path.read_text().splitlines()
    tree = ast.parse("\n".join(lines))
    (node,) = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "NEW_ITEMS"
    ]
    comment = _leading_comment_block(lines, node.lineno)
    words = _count_words_before(comment, "things")
    actual = len(node.value.elts)
    stale = {w for w in words if _NUMBER_WORDS[w] != actual}
    assert not stale, f"comment names {stale} but NEW_ITEMS has {actual} entries"


def test_rail_comment_row_count_matches_rail_groups():
    """shell-14: the comment beside the section-gap ladder in
    ``rail.fitted_height`` no longer names a stale row count ("thirteen")
    for an expression that evaluates to fourteen (``sum(len(g) for g in
    modes.RAIL_GROUPS)``)."""

    from realmspinner.studio import modes

    actual = sum(len(g) for g in modes.RAIL_GROUPS)

    path = SRC / "rail.py"
    text = path.read_text()
    # The ladder's third rung is the paragraph that used to cite "thirteen
    # rows"; find it by its stable neighbour, the MIN_ITEM_H floor line.
    match = re.search(r"# \*\*And then the section gaps.*?\n(?:    #.*\n){1,12}", text)
    assert match, "could not find the section-gap ladder comment in rail.py"
    words = _count_words_before(match.group(0), "rows")
    stale = {w for w in words if _NUMBER_WORDS[w] != actual}
    assert not stale, f"comment names {stale} but the rail has {actual} rows"
