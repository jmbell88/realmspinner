"""``familiar.retrieval``: the BM25 index Familiar cites the Manual through.

The index is built once at module scope (a fixture with session scope) since
building it parses all 46 chapters -- doing that per test would make the
suite pay ~46-chapter parse cost dozens of times over for no reason.
"""

from __future__ import annotations

import re
import time

import pytest

from warlock.studio.familiar import retrieval
from warlock.studio.manual import loader, parser


@pytest.fixture(scope="module")
def index() -> retrieval.Index:
    start = time.perf_counter()
    idx = retrieval.Index.build()
    elapsed = time.perf_counter() - start
    # Not an assertion -- reported so a future regression is visible in test
    # output without needing a benchmark harness. "About 2s" is the brief's
    # own ballpark, not a budget this test enforces.
    print(f"\nretrieval.Index.build() took {elapsed:.3f}s for {len(idx.chunks)} chunks")
    return idx


@pytest.fixture(scope="module")
def anchors_by_chapter() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for chapter in loader.chapters():
        blocks = parser.parse(loader.load(chapter.key))
        out[chapter.key] = {s.anchor for s in loader.sections(blocks)}
    return out


@pytest.mark.parametrize(
    "query",
    ["export a GLB", "tile map Tiled", "undo", "materials in Clay"],
)
def test_retrieval_cites_only_anchors_the_manual_really_has(
    index: retrieval.Index, anchors_by_chapter: dict[str, set[str]], query: str
) -> None:
    citations = index.search(query)
    assert citations, f"expected at least one citation for {query!r}"
    for expected_n, citation in enumerate(citations, start=1):
        assert citation.n == expected_n
        if citation.anchor is not None:
            assert citation.anchor in anchors_by_chapter[citation.chapter]


def test_a_code_identifier_query_finds_the_paragraph_that_names_it(
    index: retrieval.Index,
) -> None:
    # WARLOCK_VRAM_BUDGET is named in exactly one chapter (docs/manual/
    # 41-configuration.md) -- confirmed with a literal grep over docs/manual
    # before writing this test, so a false pass (some other chunk happening
    # to rank first) is not on the table.
    citations = index.search("WARLOCK_VRAM_BUDGET")
    assert citations
    assert "WARLOCK_VRAM_BUDGET" in citations[0].text


def test_citations_stay_inside_the_token_budget(index: retrieval.Index) -> None:
    budget = 400
    citations = index.search("export a GLB", limit=6, budget_tokens=budget)
    assert citations
    total = sum(len(c.text.split()) for c in citations)
    # The first citation always lands regardless of size (the "always return
    # at least one" rule), so the budget is only a bound on everything after
    # it plus that first one's own length.
    if len(citations) > 1:
        assert total <= budget + len(citations[0].text.split())


def test_a_query_with_no_matching_terms_returns_no_citations(
    index: retrieval.Index,
) -> None:
    assert index.search("xyzzyzzyplghqwertyzzznonexistentnonsense") == []


def test_top_hits_for_godot_export(index: retrieval.Index) -> None:
    # Not a hard assertion of exact chapters (that's a content decision, not
    # a retrieval contract) -- prints the top 3 so it is visible in output,
    # and asserts the shape of the response rather than which chapter wins.
    citations = index.search("how do I export to Godot")
    assert citations
    for c in citations[:3]:
        print(f"  {c.title_path}")


_CANDIDATE_WORD = re.compile(r"[A-Za-z]{6,}")


def test_retrieval_section_ownership_agrees_with_the_manual_loader() -> None:
    """``retrieval._chapter_chunks`` copies ``loader.matching_sections``'s own
    ownership walk rather than reusing it (see that function's docstring) --
    a copy that can silently drift. Exact parity is not expressible: a long
    section is split by ``_split_long_section`` into several retrieval chunks
    sharing one anchor, so "the chunk a word is in" and "the section a word
    is in" are not quite the same question. The strongest checkable claim is
    ownership, not chunking: a word that appears in exactly one section of a
    chapter (by retrieval's own reckoning) must have that section's anchor
    among what ``loader.matching_sections`` returns for the same word over
    the same blocks.
    """
    checked_words = 0
    for chapter in loader.chapters():
        blocks = parser.parse(loader.load(chapter.key))

        # Group retrieval's own chunks by anchor -- a split section's pieces
        # share one anchor, so this is retrieval's idea of "one section".
        by_anchor: dict[str, list[str]] = {}
        for chunk in retrieval._chapter_chunks(chapter, blocks):
            if chunk.anchor is not None:
                by_anchor.setdefault(chunk.anchor, []).append(chunk.text)

        if len(by_anchor) < 1:
            continue

        # Words appearing in more than one section can't tell ownership
        # apart, so they are not a fair sample -- count each candidate word's
        # sections across the whole chapter first.
        word_sections: dict[str, set[str]] = {}
        for anchor, texts in by_anchor.items():
            for word in _CANDIDATE_WORD.findall(" ".join(texts).lower()):
                word_sections.setdefault(word, set()).add(anchor)

        for anchor, texts in by_anchor.items():
            words = _CANDIDATE_WORD.findall(" ".join(texts).lower())
            sample = [w for w in dict.fromkeys(words) if word_sections[w] == {anchor}][:3]
            for word in sample:
                hits = {s.anchor for s in loader.matching_sections(word, blocks)}
                assert anchor in hits, (
                    f"{chapter.key}: {word!r} is unique to {anchor!r} by retrieval's own "
                    f"chunking, but loader.matching_sections did not find it there: {hits}"
                )
                checked_words += 1

    assert checked_words > 0, "no chapter offered a section-unique word to check ownership with"


def test_one_section_is_cited_once_even_when_split(index: retrieval.Index) -> None:
    """ "how do I export to Godot" used to cite "13 Putting it in a game ›
    3D engines" twice: the section is long enough that ``_split_long_section``
    breaks it into two chunks sharing one ``(chapter, anchor)``, and both
    scored high enough to both make the citation list. A reader seeing the
    same section number twice in one answer has no way to tell that is not a
    bug in the citation numbering itself."""
    citations = index.search("how do I export to Godot")
    seen = set()
    for c in citations:
        key = (c.chapter, c.anchor)
        assert key not in seen, f"{c.title_path!r} cited twice"
        seen.add(key)
