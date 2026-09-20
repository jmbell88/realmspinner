"""Manual retrieval for Familiar: a BM25 index over Manual chunks.

Pure and synchronous by design -- T6 builds ``Index`` off the frame thread
(a `TaskRunner` job, the same shape every other one-shot CPU job in this
app takes) and hands the finished object to the listener. Nothing here
touches imgui, moderngl, pygame, ``service`` or the network: it imports
only the stdlib plus the two pure Manual modules, so it can be built and
tested headlessly the same way ``manual/loader.py`` is.

Chunking reuses ``loader``'s section-ownership rule (a ``##``/``###``
heading owns every block until the next heading at or above its level)
rather than re-deriving it, so retrieval's idea of "a section" can never
drift from the TOC tree's.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from realmspinner.kernels.manual import loader, parser

# A chunk over roughly this many whitespace tokens is split at block
# boundaries so a single citation never dumps a whole long section on the
# model -- Familiar's answer has a token budget too, long before BM25 does.
_MAX_CHUNK_TOKENS = 300

# BM25's usual defaults (Robertson/Sparck Jones); nothing in this corpus
# argues for tuning them.
_K1 = 1.2
_B = 0.75

# search() regroups a section's split chunks (see _split_long_section) back
# into one citation so a reader sees one source, not several near-duplicates
# -- but a section with many split chunks (28-inker#tools has ten) rejoins
# them all with no cap of its own. The 2026-09-18 audit (familiar-06) found
# this made the "Inker tools" query's first (always-admitted) citation 2209
# tokens, over 7x _MAX_CHUNK_TOKENS, because the per-citation size check in
# search() only ever ran for citations after the first. A regrouped citation
# is now built chunk by chunk up to this cap instead of joining the whole
# group unconditionally.
_MAX_CITATION_TOKENS = 3 * _MAX_CHUNK_TOKENS

# An identifier is kept whole (a query for "clay_batch" must find the chunk
# naming it exactly) *and* split into its parts (a query for "batch" should
# still find it) -- so both a wholesale grep-like use and a keyword-style use
# of the manual's own vocabulary work. No stemming: the manual's vocabulary is
# small and technical (tool names, env vars, mode names) where a stemmer would
# either be a no-op or actively wrong ("Poser" -> "Pose").
_WORD = re.compile(r"[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)+|[A-Za-z0-9]+")


def _identifier_parts(token: str) -> list[str]:
    """Split ``clay_batch``/``mirror-copy``/``REALMSPINNER_HOME`` into pieces."""
    return [p for p in re.split(r"[_-]+", token) if p]


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, identifiers kept whole and split into parts."""
    tokens: list[str] = []
    for raw in _WORD.findall(text.lower()):
        tokens.append(raw)
        parts = _identifier_parts(raw)
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


def _whitespace_token_count(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class Chunk:
    chapter: str
    chapter_number: int
    chapter_title: str
    anchor: str | None
    title_path: str
    text: str


@dataclass(frozen=True)
class Citation:
    n: int
    chapter: str
    anchor: str | None
    title_path: str
    text: str


def _split_long_section(text_blocks: list[str]) -> list[str]:
    """Group a section's block texts into chunks under ``_MAX_CHUNK_TOKENS``.

    Splits at block (paragraph/code-block/list-item/table) boundaries only --
    never mid-block -- so a citation's text is always whole blocks glued back
    together with blank lines, matching how the chapter reads.
    """
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for block_text in text_blocks:
        block_tokens = _whitespace_token_count(block_text)
        if current and current_tokens + block_tokens > _MAX_CHUNK_TOKENS:
            pieces.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(block_text)
        current_tokens += block_tokens
    if current:
        pieces.append("\n\n".join(current))
    return pieces or [""]


def _chapter_chunks(chapter: loader.Chapter, blocks: list[parser.Block]) -> list[Chunk]:
    """One chapter's chunks: lead-in prose, then each ``##``/``###`` section.

    Mirrors ``loader.matching_sections``'s ownership bookkeeping (open_at by
    level) rather than reusing it directly, because that function returns
    only the *sections*, discarding exactly the block text this needs; the
    walk itself -- what closes an open section, what a heading at level N
    displaces -- is copied rather than shared. Copying does not by itself
    guarantee the two never disagree; ``tests/familiar/test_retrieval.py::
    test_retrieval_section_ownership_agrees_with_the_manual_loader`` is what
    actually checks it, over every real chapter.
    """
    chunks: list[Chunk] = []
    lead: list[str] = []
    open_at: dict[int, tuple[loader.Section, list[str]]] = {}
    parent_title: dict[int, str] = {}

    def flush_lead() -> None:
        if lead:
            text = "\n\n".join(lead)
            if text.strip():
                chunks.append(
                    Chunk(
                        chapter=chapter.key,
                        chapter_number=chapter.number,
                        chapter_title=chapter.title,
                        anchor=None,
                        title_path=f"{chapter.number:02d} {chapter.title}",
                        text=text,
                    )
                )
            lead.clear()

    def flush_section(section: loader.Section, texts: list[str]) -> None:
        prefix = f"{chapter.number:02d} {chapter.title}"
        if section.level == 3 and 2 in parent_title:
            title_path = f"{prefix} › {parent_title[2]} › {section.title}"
        else:
            title_path = f"{prefix} › {section.title}"
        for piece in _split_long_section(texts):
            chunks.append(
                Chunk(
                    chapter=chapter.key,
                    chapter_number=chapter.number,
                    chapter_title=chapter.title,
                    anchor=section.anchor,
                    title_path=title_path,
                    text=piece,
                )
            )

    for block in blocks:
        if isinstance(block, parser.Heading):
            for level in list(open_at):
                if level >= block.level:
                    section, texts = open_at.pop(level)
                    flush_section(section, texts)
            for level in list(parent_title):
                if level >= block.level:
                    del parent_title[level]
            if 2 <= block.level <= 3:
                flush_lead()
                section = loader.Section(block.level, block.text, block.anchor)
                open_at[block.level] = (section, [])
                parent_title[block.level] = block.text
            continue
        text = loader.block_text(block)
        if not text.strip():
            continue
        if not open_at:
            lead.append(text)
        else:
            open_level = max(open_at)
            open_at[open_level][1].append(text)

    flush_lead()
    for level in sorted(open_at):
        section, texts = open_at[level]
        flush_section(section, texts)
    return chunks


def _all_chunks(chapters: list[loader.Chapter], root: Path | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for chapter in chapters:
        blocks = parser.parse(loader.load(chapter.key, root=root))
        chunks.extend(_chapter_chunks(chapter, blocks))
    return chunks


@dataclass
class Index:
    """A BM25 index over Manual chunks.

    Pure and synchronous -- T6's job is to call ``Index.build`` on a
    ``TaskRunner`` worker thread, never the frame thread, the same rule
    every other CPU-bound Familiar step already follows.
    """

    chunks: list[Chunk]
    _doc_tokens: list[list[str]]
    _doc_freqs: list[Counter[str]]
    _doc_lens: list[int]
    _avg_len: float
    _df: Counter[str]
    _n: int

    @classmethod
    def build(cls, chapters: list[loader.Chapter] | None = None) -> Index:
        chs = chapters if chapters is not None else loader.chapters()
        chunks = _all_chunks(chs)
        doc_tokens: list[list[str]] = []
        doc_freqs: list[Counter[str]] = []
        doc_lens: list[int] = []
        df: Counter[str] = Counter()
        for chunk in chunks:
            # Heading/title_path terms count twice: a query using the manual's
            # own section names ("materials in Clay") should outrank a chunk
            # that merely mentions the words in passing.
            tokens = tokenize(chunk.title_path) * 2 + tokenize(chunk.text)
            doc_tokens.append(tokens)
            freqs = Counter(tokens)
            doc_freqs.append(freqs)
            doc_lens.append(len(tokens))
            df.update(freqs.keys())
        n = len(chunks)
        avg_len = (sum(doc_lens) / n) if n else 0.0
        return cls(
            chunks=chunks,
            _doc_tokens=doc_tokens,
            _doc_freqs=doc_freqs,
            _doc_lens=doc_lens,
            _avg_len=avg_len,
            _df=df,
            _n=n,
        )

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        # BM25's standard idf, floored at a small positive value so a term
        # in every chunk still contributes something rather than going
        # negative and penalising documents that contain it.
        return max(
            math.log((self._n - df + 0.5) / (df + 0.5) + 1.0),
            1e-9,
        )

    def _score(self, doc_index: int, query_terms: list[str]) -> float:
        freqs = self._doc_freqs[doc_index]
        doc_len = self._doc_lens[doc_index]
        score = 0.0
        for term in query_terms:
            f = freqs.get(term, 0)
            if f == 0:
                continue
            idf = self._idf(term)
            denom = f + _K1 * (1 - _B + _B * doc_len / (self._avg_len or 1.0))
            score += idf * (f * (_K1 + 1)) / denom
        return score

    def search(self, query: str, limit: int = 6, budget_tokens: int = 2500) -> list[Citation]:
        query_terms = tokenize(query)
        if not query_terms or self._n == 0:
            return []
        scored: list[tuple[float, int]] = []
        for i in range(self._n):
            score = self._score(i, query_terms)
            if score > 0:
                scored.append((score, i))
        if not scored:
            return []
        # Deterministic tie-break: score descending, then source order (the
        # chunk index), so equal scores never depend on sort stability across
        # runs or Python versions.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))

        # A long section is split into several chunks by _split_long_section,
        # each of which can score high enough on its own to make the top-N
        # cut -- e.g. "13 Putting it in a game > 3D engines" split in two,
        # both about Godot. Collapse hits sharing a (chapter, anchor) into one
        # citation at the best-scoring chunk's rank rather than let a reader
        # see the same section cited twice with no way to tell that apart
        # from a real second source.
        groups: dict[tuple[str, str | None], list[int]] = {}
        order: list[tuple[str, str | None]] = []
        for _score, idx in scored[: max(limit, 1) * 4]:
            chunk = self.chunks[idx]
            key = (chunk.chapter, chunk.anchor)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(idx)

        citations: list[Citation] = []
        used_tokens = 0
        for key in order:
            if len(citations) >= limit:
                break
            # Join the group's chunks in document order (source order == the
            # order they were produced in _chapter_chunks) so a joined
            # citation reads the way the chapter does, not by score -- but
            # stop joining once _MAX_CITATION_TOKENS is reached rather than
            # gluing the whole group back together regardless of size (the
            # 2026-09-18 audit, familiar-06): this is the size check the old
            # code only ran for citations after the first, so it must run
            # here, before the "is this citation even a fit" budget check
            # below, and not be skipped for the first citation the way that
            # one deliberately still is.
            idxs = sorted(groups[key])
            chunk = self.chunks[idxs[0]]
            pieces: list[str] = []
            regrouped_tokens = 0
            for i in idxs:
                piece = self.chunks[i].text
                piece_tokens = _whitespace_token_count(piece)
                if pieces and regrouped_tokens + piece_tokens > _MAX_CITATION_TOKENS:
                    break
                pieces.append(piece)
                regrouped_tokens += piece_tokens
            text = "\n\n".join(pieces)
            chunk_tokens = regrouped_tokens
            if citations and used_tokens + chunk_tokens > budget_tokens:
                break
            citations.append(
                Citation(
                    n=len(citations) + 1,
                    chapter=chunk.chapter,
                    anchor=chunk.anchor,
                    title_path=chunk.title_path,
                    text=text,
                )
            )
            used_tokens += chunk_tokens
        return citations
