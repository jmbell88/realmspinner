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

SRC = Path(__file__).resolve().parents[1] / "src" / "warlock" / "studio"

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


def test_doc_modes_comment_count_matches_the_tuple_length():
    """shell-12: the comment above ``DOC_MODES`` no longer names a stale
    tuple length ("five" when the tuple held six)."""

    path = SRC / "docmodes.py"
    lines = path.read_text().splitlines()
    tree = ast.parse("\n".join(lines))
    (node,) = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "DOC_MODES"
    ]
    comment = _leading_comment_block(lines, node.lineno)
    words = _count_words_before(comment, "document modes")
    actual = len(ast.literal_eval(node.value))
    stale = {w for w in words if _NUMBER_WORDS[w] != actual}
    assert not stale, f"comment names {stale} but DOC_MODES has {actual} entries"


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

    from warlock.studio import modes

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
