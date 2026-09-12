"""Five chapters whose prose drifted from the tree, pinned against it.

The 2026-09-12 audit's manual-prose findings (docs-12 through docs-16), each
read from the module or chapter that makes the claim true or false rather
than repeated as a second hand-written copy -- the same shape as
``test_manual_promises.py`` and ``test_manual_claims.py``, kept separate
because those belong to different fixers' file lists.
"""

from __future__ import annotations

import re
from pathlib import Path

MANUAL = Path(__file__).resolve().parents[2] / "docs" / "manual"
STUDIO = Path(__file__).resolve().parents[2] / "src" / "warlock" / "studio"


def _chapter(name: str) -> str:
    return (MANUAL / name).read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The prose under one ``##`` heading, up to the next ``##`` (or EOF)."""
    pattern = rf"^## {re.escape(heading)}\s*$"
    match = re.search(pattern, text, re.MULTILINE)
    assert match, f"no '## {heading}' heading found"
    rest = text[match.end():]
    nxt = re.search(r"^## ", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


# --- docs-12: the tutorial chain's "what to read next" links ---------------


def _index_tutorials() -> list[str]:
    """The tutorial chapter filenames, in the order 00-index.md's own
    Tutorials list gives them.

    Derived from the index rather than hard-coded, so a chapter appended to
    that list later (18-...) walks into this chain gate automatically
    instead of sliding past it the way 15, 16 and 17 did in the 2026-09-12
    audit.
    """
    section = _section(_chapter("00-index.md"), "Tutorials")
    return re.findall(r"\]\((\d\d-[a-z0-9-]+\.md)\)", section)


def _next_section(text: str) -> str:
    """The "what next" section of a tutorial chapter, under either of the
    two headings the tutorials actually use."""
    match = re.search(r"^## (?:What to read next|Where to go next)\s*$", text, re.MULTILINE)
    assert match, "no 'what to read next' / 'where to go next' heading found"
    rest = text[match.end():]
    nxt = re.search(r"^## ", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def test_tutorial_next_links_form_an_unbroken_chain_through_the_last_tutorial():
    """The 2026-09-12 audit, finding docs-12.

    Chapter 13 said "One tutorial left" and chapter 14 said "That is the
    last tutorial" while three more tutorials (15, 16, 17) sat right after
    them in 00-index.md's own Tutorials list. A reader following the
    manual's stated method ("Start with the tutorials... they walk one
    path") was walked off the end of the chain three chapters early, and
    neither 15 nor 16 pointed forward to the next tutorial either (15
    looped back to 13; 16 named 36/35/14 but not 17).
    """
    tutorials = _index_tutorials()
    assert len(tutorials) >= 2, "the index's Tutorials list moved or emptied; update this test"

    broken = []
    for current, following in zip(tutorials, tutorials[1:], strict=False):
        section = _next_section(_chapter(current))
        if following not in section:
            broken.append((current, following))

    assert not broken, (
        "these tutorials' 'what to read next' section does not link to the "
        "next tutorial in 00-index.md's own list: "
        + ", ".join(f"{a} -> {b}" for a, b in broken)
    )


# --- docs-13: the index's download-count blurb for chapter 1 ---------------


_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                  6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

_COUNTED_DOWNLOADS = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+downloads?\b",
    re.IGNORECASE,
)


def test_index_blurb_download_count_matches_the_chapter_it_summarizes():
    """The 2026-09-12 audit, finding docs-13.

    The index's one-line summary of chapter 1 said "the two downloads" while
    the chapter's own table lists three (TRELLIS.2 engine, TRELLIS.2 GGUF
    weights, SDXL 1.0) since the reconstruction engine stopped shipping in
    the installer on 2026-09-10. The count has already gone stale once and
    will again, so this reads the chapter's actual table rather than
    checking the blurb against a literal "three": if the blurb names no
    count at all (the suggested fix) this passes; if it names one, that one
    must match the table.
    """
    chapter_text = _chapter("01-before-you-begin.md")
    launch_section = _section(chapter_text, "The first launch")
    row_count = len(re.findall(r"^\|\s*\*\*", launch_section, re.MULTILINE))
    assert row_count >= 1, "the downloads table moved or changed shape; update this test"

    index_section = _section(_chapter("00-index.md"), "Tutorials")
    line = next(
        line for line in index_section.splitlines() if "01-before-you-begin.md" in line
    )

    match = _COUNTED_DOWNLOADS.search(line)
    if match is None:
        return  # the blurb names no count -- nothing to go stale

    stated = match.group(1).lower()
    expected = _NUMBER_WORDS.get(row_count)
    assert stated == expected, (
        f"index says {stated!r} downloads but 01-before-you-begin.md's table "
        f"now lists {row_count} ({expected!r})"
    )


# --- docs-14: the leftover "your the" in the App settings chapter ----------


_POSSESSIVE_THEN_ARTICLE = re.compile(
    r"\b(your|my|his|her|its|our|their)\s+(a|an|the)\b",
    re.IGNORECASE,
)


def test_app_settings_chapter_has_no_duplicate_leftover_words():
    """The 2026-09-12 audit, finding docs-14.

    "also holds your\\nthe sidebar's internal split" is a leftover word from
    an edit, wrapping across the paragraph's soft line-break so a same-line
    scan would miss it. Scoped to a possessive pronoun directly followed by
    an article -- not a general "same word twice" scan, which is noisy over
    ordinary prose (e.g. "that the", "the A-pose" are legitimate and appear
    throughout the manual) -- because two determiners in a row is never
    grammatical and is exactly the shape this leftover took.
    """
    text = _chapter("42-app-settings.md")
    for paragraph in text.split("\n\n"):
        joined = " ".join(paragraph.splitlines())
        match = _POSSESSIVE_THEN_ARTICLE.search(joined)
        assert match is None, (
            f"leftover word: {match.group(0)!r} in paragraph starting "
            f"{paragraph.strip().splitlines()[0]!r}"
        )


# --- docs-15: which modes the axis-view chords are said to work in ---------


def _modes_sharing_axis_view_key() -> set[str]:
    """The display label of every mode whose ``*_mode.py`` calls
    ``axis_view_key`` (directly, or -- like Poser -- via another mode's
    re-exported handle onto the same ``_view_frame`` function).
    """
    from warlock.studio import modes as modes_module

    labels = {key: label for key, label, *_rest in modes_module.MODES}
    found = set()
    for path in STUDIO.glob("*_mode.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"\baxis_view_key\(", text):
            key = path.stem.removesuffix("_mode")
            found.add(labels[key])
    return found


def test_overview_axis_view_sentence_names_every_mode_that_shares_axis_view_key():
    """The 2026-09-12 audit, finding docs-15.

    20-overview.md's "same gestures" bullet said the Ctrl+1/3/5/7 axis-view
    chords work "in Clay and in Poser alike", but Mason calls the same
    shared ``axis_view_key`` function (mason_mode.py:1134) with identical
    behaviour and went unmentioned. ``test_ux_consistency_pass2.py`` asserts
    Clay and Poser share the function but predates Mason and doesn't check
    it either -- this is the doc-facing gate that one doesn't provide.
    """
    modes_sharing = _modes_sharing_axis_view_key()
    assert modes_sharing, "no *_mode.py calls axis_view_key any more; update this test"

    section = _section(_chapter("20-overview.md"), "What is the same in every workspace")
    bullet_match = re.search(r"The same gestures\..*?(?=\n- \*\*|\Z)", section, re.DOTALL)
    assert bullet_match, "the 'same gestures' bullet moved; update this test"
    bullet = bullet_match.group(0)

    missing = [label for label in modes_sharing if label not in bullet]
    assert not missing, (
        f"20-overview.md's 'same gestures' bullet doesn't name: {missing} "
        f"-- each calls axis_view_key"
    )


# --- docs-16: the Home chapter's quoted Setup status line -------------------


def _setup_status_constants() -> list[str]:
    """The literal (non-interpolated) text pieces of the f-string
    ``_setup_status`` builds in ``panes/landing.py``, in order.
    """
    import ast

    source = (STUDIO / "panes" / "landing.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            constants = [v.value for v in node.values if isinstance(v, ast.Constant)]
            text = "".join(constants)
            if "Generation is not set up yet" in text:
                return constants
    raise AssertionError("no 'Generation is not set up yet' f-string found in landing.py")


def test_home_setup_line_quote_matches_landing_setup_status_format():
    """The 2026-09-12 audit, finding docs-16.

    21-home.md's Status table quotes the Setup line as "Generation is not
    set up yet - N downloads", presented as an exact quote, but
    ``_setup_status`` also appends ", about {total:.0f} GB" after the
    download count. This reads the format from landing.py (not a pasted
    literal) so a future change to the size clause's wording is what this
    test tracks, not a fixed string.
    """
    constants = _setup_status_constants()
    tail = "".join(constants[-2:])
    assert "about" in tail and "GB" in tail, (
        "landing.py's _setup_status format changed shape; update this test"
    )

    chapter_text = _chapter("21-home.md")
    line = next(
        line for line in chapter_text.splitlines() if "Generation is not set up yet" in line
    )

    # The manual's house style uses an em dash where the source uses a plain
    # hyphen -- a stylistic difference, not this finding -- so normalise
    # before comparing.
    normalized = line.replace("—", "-").replace("–", "-")

    downloads_at = normalized.find("downloads")
    assert downloads_at != -1, "21-home.md no longer quotes the download count"
    after = normalized[downloads_at:]
    assert "about" in after and "GB" in after, (
        "21-home.md's quoted Setup line is missing the ', about N GB' clause "
        "that _setup_status actually appends after the download count"
    )
