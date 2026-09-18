"""``MuseState``'s own field, tested at the level a doc-comment lives at.

New file: nothing tested ``muse_state.py``'s dataclass fields as *text* before
this -- ``test_muse_mode.py`` covers :attr:`muse_state.MuseState.loop_memory`'s
runtime behaviour (its cap and eviction), but a comment attached to the wrong
field is invisible to any test that only imports the module and reads values
off it. This file is for defects a `#:` comment itself can have.
"""

from __future__ import annotations

import re
from pathlib import Path

from warlock.studio.modes.muse import state as muse_state


def _source() -> str:
    return Path(muse_state.__file__).read_text(encoding="utf-8")


def _preceding_doc_comment(source: str, field_pattern: str) -> str:
    """The run of ``    #: ...`` lines immediately above a field declaration.

    Fails loudly (rather than returning "") if the field cannot be found, so a
    field renamed out from under this test is a clear error and not a
    trivially-passing empty comparison.
    """
    match = re.search(
        r"((?:^ {4}#:.*\n)+)^ {4}" + field_pattern, source, re.M
    )
    assert match is not None, f"no doc-comment block found directly above {field_pattern!r}"
    return match.group(1)


def test_the_compose_strength_doc_comment_is_not_attached_to_loop_memory():
    """muse-06 (2026-09-07 audit).

    A ``#:`` block describing :attr:`muse_state.MuseState.compose_strength`
    (opening "How near the model stays to a song composed from Sirens...") was
    attached to :attr:`muse_state.MuseState.loop_memory` instead -- one
    comment run straight from that paragraph into loop_memory's own
    ("``{job id: (loop_start, loop_end, xfade_ms)}``..."), with nothing
    marking where one field's documentation ended and the other's began. A
    reader of ``loop_memory`` met a paragraph about a different field first.

    Fails against the unfixed code, whose ``loop_memory`` doc-comment block
    still opens with the ``compose_strength`` paragraph instead of its own.
    """
    source = _source()
    loop_memory_doc = _preceding_doc_comment(source, r"loop_memory: dict\[")
    compose_strength_doc = _preceding_doc_comment(
        source, r"compose_strength: float = field\("
    )

    assert "How near the model stays" not in loop_memory_doc, (
        "loop_memory's doc-comment still carries compose_strength's paragraph"
    )
    assert "{job id: (loop_start, loop_end, xfade_ms)}" in loop_memory_doc

    assert "{job id: (loop_start, loop_end, xfade_ms)}" not in compose_strength_doc, (
        "compose_strength's doc-comment still carries loop_memory's paragraph"
    )
    assert "How near the model stays" in compose_strength_doc
