"""Chapter discovery: the packaged copy if installed, the repo docs otherwise.

The canonical files live at docs/manual/ -- readable on GitHub -- and hatchling
force-includes them into the wheel as realmspinner/manual, the same
single-source-two-locations bargain CHANGELOG.md makes. Pure: no
imgui anywhere, so tests run headless.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from . import parser

# A chapter's number decides both its order and its part, so a new chapter is
# a renumbering rather than an append, and every cross-link, every HELP_TARGETS
# entry and the index are asserted in both directions by
# ``tests/manual/test_docs.py`` (``test_index_sections_match_the_loaders_parts``
# pins the grouping against ``docs/manual/00-index.md``). The ranges below are
# the one place a chapter number is load-bearing: Part I is the tutorials, Part
# II "Using Realmspinner" (the screens, in the rail's order), Part III setup and
# operations, Part IV architecture. 01-19 are reserved for the tutorial series
# as a block, so the next tutorial is a new file and an ``EXPECTED_KEYS`` line,
# not another round of renames across the cross-links; the gap is legal because
# ``chapters()`` globs and sorts and needs the numbers ordered, not dense.
#
# The 2026-10-14 audit, finding docs-A-B-14, found this comment narrating five
# renumberings in a vocabulary the index no longer uses (it filed the shortcuts
# chapter under "Part I" and cited chapters and gotcha ids that are gone). The
# dated history is CHANGELOG's.
PARTS: tuple[tuple[str, range], ...] = (
    ("Tutorials", range(1, 20)),
    ("Using Realmspinner", range(20, 40)),
    ("Setup & operations", range(40, 44)),
    ("Architecture", range(44, 47)),
)

_H1 = re.compile(r"^# +(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Chapter:
    key: str  # filename stem: "20-overview"
    number: int
    title: str
    part: str


def manual_dir() -> Path:
    packaged = Path(str(resources.files("realmspinner"))) / "manual"
    if packaged.is_dir():
        return packaged
    # Dev checkout: src/realmspinner/kernels/manual/loader.py -> repo root is
    # parents[4], and the canonical files are docs/manual there.
    return Path(__file__).resolve().parents[4] / "docs" / "manual"


def chapters(root: Path | None = None) -> list[Chapter]:
    base = root or manual_dir()
    found: list[Chapter] = []
    for path in sorted(base.glob("[0-9][0-9]-*.md")):
        number = int(path.stem[:2])
        match = _H1.search(path.read_text(encoding="utf-8"))
        title = match.group(1).strip() if match else path.stem
        part = next((label for label, rng in PARTS if number in rng), "")
        found.append(Chapter(path.stem, number, title, part))
    return found


def load(key: str, root: Path | None = None) -> str:
    base = root or manual_dir()
    if not any(c.key == key for c in chapters(root=base)):
        raise KeyError(key)
    return (base / f"{key}.md").read_text(encoding="utf-8")


# The sub-chapter half of navigation. The TOC listed chapters and stopped
# there, which was survivable while the longest chapter was a screen or two
# and stopped being so as Inker reached 1110 lines: a reader who knew exactly
# which section they wanted still had to open the chapter and scroll for it,
# and the (?) buttons were the only thing in the app that could land on a
# section directly. Everything below is pure for the same reason the parser is
# -- the rules about what the tree offers are assertable without a frame.


@dataclass(frozen=True)
class Section:
    """One ``##`` or ``###`` inside a chapter: a row of the TOC tree."""

    level: int
    title: str
    anchor: str


def sections(blocks: Iterable[parser.Block]) -> list[Section]:
    """The headings under a chapter's title, in source order.

    The H1 is excluded deliberately: it is the chapter, and the tree already
    draws it as the row these hang from. Including it would put every chapter's
    own name inside itself.
    """
    return [
        Section(block.level, block.text, block.anchor)
        for block in blocks
        if isinstance(block, parser.Heading) and 2 <= block.level <= 3
    ]


def active_anchor(offsets: Sequence[tuple[str, float]], scroll_y: float) -> str | None:
    """Which section a scroll position is inside. -> its anchor, or None.

    ``offsets`` are content-space y positions, which is what imgui's
    ``get_cursor_pos_y`` returns -- it adds the scroll back in, so a heading's
    recorded y is the same number on every frame and compares directly against
    ``get_scroll_y``.

    The answer is the *last* heading at or above the line rather than the
    nearest one, so a long section keeps its row lit all the way down instead
    of handing over to the next heading at the halfway mark. Above the first
    heading the answer is None: the chapter's opening prose belongs to the
    chapter, and lighting a section there would point at text the reader has
    not reached.

    Sorted rather than trusted: the renderer appends in draw order, which is
    source order, but a cache rebuilt mid-scroll can hand over a partial list
    and a wrong highlight is worse than a missing one.
    """
    found: str | None = None
    for anchor, y in sorted(offsets, key=lambda pair: pair[1]):
        if y > scroll_y:
            break
        found = anchor
    return found


def needs_scroll_reset(previous: str, current: str, anchor: str | None) -> bool:
    """Whether arriving at ``current`` should put the page back at the top.

    The manual's page is one imgui child with one id, so its scroll offset
    belongs to the child rather than to the chapter drawn into it. Opening a
    short chapter while scrolled deep into a long one therefore landed the
    reader at the short chapter's footer, past everything it had to say --
    which looked exactly like a chapter that begins with its own "next
    chapter" buttons. It was survivable while the only way to navigate was to
    pick another chapter out of a flat list; the TOC tree made it constant.

    An anchor answers False because ``set_scroll_here_y`` already scrolls to
    the heading as it draws, and two things scrolling the same child on the
    same frame is one of them losing.
    """
    return anchor is None and previous != current


def matching_sections(needle: str, blocks: Iterable[parser.Block]) -> list[Section]:
    """The sections of one chapter whose heading or body contains ``needle``.

    Takes parsed blocks rather than a chapter key, and deliberately: the
    renderer calls this once per matching chapter per frame while somebody is
    typing in the search box, and ``render._blocks`` already holds every
    chapter parsed and warmed. Reading the file here would put a read and a
    parse per chapter into a keystroke.

    A section owns the blocks between its heading and the next heading *at or
    above its own level* -- so prose under ``### Image brushes`` matches the
    subsection, and prose directly under ``## Tools`` matches Tools. Text
    before the first heading matches nothing: it belongs to the chapter, and
    filing it under whichever heading happens to follow would scroll the reader
    past the words they searched for.

    Case-insensitive, and every block type is searched, for the reason
    ``render._block_text`` gives: a reader arrives with "gltfpack" or
    "REALMSPINNER_VRAM_BUDGET", and those live in paragraphs and code blocks rather
    than in headings.
    """
    needle = needle.strip().lower()
    if not needle:
        return []
    blocks = list(blocks)
    # open_at[level] is the section currently accumulating text at that level.
    open_at: dict[int, Section] = {}
    hits: dict[str, Section] = {}
    for block in blocks:
        if isinstance(block, parser.Heading):
            for level in list(open_at):
                if level >= block.level:
                    del open_at[level]
            if 2 <= block.level <= 3:
                section = Section(block.level, block.text, block.anchor)
                open_at[block.level] = section
                if needle in block.text.lower():
                    hits[section.anchor] = section
            continue
        if not open_at or needle not in block_text(block).lower():
            continue
        # The innermost open section only. A paragraph under ``### Image
        # brushes`` is inside ``## Tools`` as well, and listing both would put
        # the parent of every hit beside it -- a result list where half the
        # rows scroll to somewhere the word does not appear.
        hits_at = open_at[max(open_at)]
        hits[hits_at.anchor] = hits_at
    order = {s.anchor: i for i, s in enumerate(sections(blocks))}
    return sorted(hits.values(), key=lambda s: order[s.anchor])


def block_text(block: parser.Block) -> str:
    """Everything readable in one block, as a flat string.

    Every block type, because a search that sees only headings is a search over
    the table of contents: "gltfpack", "prompt_hash" and "REALMSPINNER_VRAM_BUDGET"
    are each named in one paragraph and in no heading anywhere, so the three
    things a reader is most likely to arrive with found nothing.

    Code blocks are included deliberately. A command line is exactly the kind
    of string somebody pastes into a search box, and it is the one kind that
    never appears in prose.

    Public, and living here rather than in ``render``, because the chapter
    search and the section search are the same question asked at two
    granularities -- they were briefly two copies of this function, which is
    one edit away from a search that finds a chapter and then no section in it.
    """
    if isinstance(block, parser.Heading | parser.CodeBlock):
        return block.text
    if isinstance(block, parser.Paragraph | parser.ListItem):
        return " ".join(span.text for span in block.spans)
    if isinstance(block, parser.Table):
        cells = [*block.header, *(cell for row in block.rows for cell in row)]
        return " ".join(span.text for cell in cells for span in cell)
    return ""
