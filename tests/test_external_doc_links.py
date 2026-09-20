"""Links *into* the manual from outside it, which nothing else was checking.

``tests/manual/test_docs.py`` walks the manual's own links and anchors and is
strict about them -- but it only ever opens files under ``docs/manual/``. Every
link that points *at* the manual from the repo root or from a measurement
document was therefore unchecked, and both kinds had rotted the same way: the
manual has been renumbered, and `README.md` and `LEFTOVERS.md` were still
sending readers to ``docs/manual/14-configuration.md``, which is the *shortcuts*
chapter. LEFTOVERS presented that one as a citation it had already corrected,
which is the sharpest version of the problem: a correction is not exempt from
the rule it enforces.

So this is the outward half of the same gate. A renumbering now fails here
rather than silently pointing a reader at the wrong chapter, and it fails at the
file that carries the stale link rather than inside the manual.

Anchors are checked as well as filenames, because a chapter can survive a
renumbering with its headings rewritten -- ``#optional-image-models-and-style-loras``
is exactly such a link, from a measurement document, and it is the kind that
lands the reader on the right page and the wrong part of it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "docs" / "manual"

# The files outside docs/manual/ that are allowed to link into it. Named rather
# than globbed: a glob would quietly stop covering a file that was renamed, and
# the whole point here is that a link nobody walks is a link that rots.
# ``docs/TODO.md`` sat here until 2026-08-11, when the roadmap file was deleted
# outright (`de87838`; there is no roadmap file now -- see dev/INVARIANTS.md).
# Its entry is gone rather than commented into the tuple, because a declared
# source that does not exist used to be *silent*: ``_manual_links`` skipped a
# missing path, so the link count simply fell, and the global-count guard below
# was slack enough to absorb it. That is the exact silent-shrink failure this
# file's own comments warn about, and it is why
# ``test_every_declared_source_exists`` now fails on the missing file itself
# rather than leaving the shrink to be inferred from a total.
#
# ``CLAUDE.md``, ``docs/INVARIANTS.md`` (now ``dev/INVARIANTS.md``),
# ``TODO.md`` (now ``dev/TODO.md``) and the ``docs/measurements/*.md`` glob
# (now ``dev/measurements/``) moved out of this tuple on 2026-09-16, when they
# moved out of the public checkout entirely -- they do not exist on a clean
# clone, so declaring them here would fail ``test_every_declared_source_exists``
# on CI. The same link-and-anchor sweep for them now lives in
# ``dev/tests/test_external_doc_links.py``, which only runs where ``dev/``
# exists.
SOURCES = (
    ROOT / "README.md",
    # The optional-model catalogue the README's download sections moved into on
    # 2026-08-10. It points readers at the guidance-panel chapter, so its links
    # rot with a renumbering exactly the way README.md's do.
    ROOT / "docs" / "MODELS.md",
    # Created in the 2026-08-21 consolidation: the merged interop ledger, which
    # cites the fixture inventories, the manual and this file's other sources.
    ROOT / "docs" / "COMPAT.md",
    # The 2026-09-06 docs audit, finding docs-10: the docs slice's own root
    # sources stopped at the seven above, so INSTALL.md:124's citation of a
    # manual chapter and THIRD-PARTY-NOTICES.md's links to docs/MODELS.md were
    # never checked for a live target. Named here for the same reason every
    # other entry is: a manual renumbering breaks these exactly the way it
    # breaks README.md, and this file exists to catch that before a reader does.
    ROOT / "INSTALL.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "SECURITY.md",
    ROOT / "THIRD-PARTY-NOTICES.md",
    ROOT / "CHANGELOG.md",
)

# ``CHANGELOG.md`` records what was true on a date, not what is true now --
# the same reasoning ``tests/test_findings_followups.py`` already applies to it
# under its own ``_HISTORIES``. Its old entries cite chapter numbers and
# filenames as they stood *at the time*, including ones since renumbered or
# deleted (``docs/manual/07-sprite-sheets.md``, ``FINDINGS.md``), so checking
# those as live citations would fail on history rather than on drift. Declared
# in SOURCES so ``test_every_declared_source_exists`` still watches the file
# itself; skipped by the two walkers below so its past does not have to match
# the present.
_HISTORIES = {"CHANGELOG.md"}

# A citation into the manual is only ever a chapter (``.md``, optionally with
# an anchor) -- but the generic ``LINK`` regex used to pick up a manual *image*
# too, since ``![alt](docs/manual/img/x.png)`` also contains ``[...](...)``
# with ``manual/`` in the target. That was invisible until INSTALL.md (finding
# docs-10) joined SOURCES with exactly such an image in its first-run
# screenshot, and the image's parent directory then failed the "must be a
# top-level chapter" assertion below for a reason that has nothing to do with
# a stale citation. Filtering on the ``.md`` suffix here keeps this file
# checking chapter links, not every asset the manual happens to hold.
_MD_TARGET = re.compile(r"\.md(?:#[\w-]+)?$")

# Two shapes, and the second is the one that caused this test to exist.
#
# ``LINK`` is an inline markdown link. Reference-style links are used nowhere in
# these files and adding one should be a deliberate act.
#
# ``MENTION`` is a bare backticked path. LEFTOVERS' broken
# ``docs/manual/14-configuration.md`` was one of these, not a link -- so a
# checker that only walked ``[](...)`` would have missed the very citation the
# file presents as already corrected. A path a reader is expected to open is a
# link whether or not it has brackets round it.
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
# A mention is resolved relative to the file that carries it, exactly as a link
# is. It used to be treated as repo-root-relative, which was indistinguishable
# from correct only while every source lived at the repo root: `docs/LEFTOVERS.md`
# moved into `docs/` on 2026-08-09 and correctly re-spelled its citation as
# `manual/18-configuration.md`, which the old root-anchored pattern then stopped
# matching altogether -- a silent loss of coverage rather than a failure, which is
# what the guard-on-the-guard below exists to turn into a failure.
#
# The optional prefix is what keeps both spellings honest: `README.md` and
# `CLAUDE.md` are at the root and say `docs/manual/...`, a measurement document
# says `../manual/...`, and each resolves against its own parent to the same
# place. Anchoring on the opening backtick is load-bearing -- it is why
# `tests/manual/test_docs.py` is not mistaken for a manual chapter.
MENTION = re.compile(r"`((?:\.\./|docs/)?manual/[0-9A-Za-z._/-]+\.md(?:#[\w-]+)?)`")


# The outward half again, one level up: a citation of a *non-manual* document.
#
# The manual checks above only ever look at ``manual/...`` targets, so a source
# could cite `docs/CAMERA.md` -- a file that never existed in any of this repo's
# commits -- and three documents did, for months. The same shape produced live
# citations of `../LEFTOVERS.md`, which did exist and was deleted.
#
# Those two cases want opposite treatment, so the rule is: a cited document must
# either exist, or the citing line must *say* it is gone. Anything containing
# "deleted" on the same line is taken as that admission -- which is what the
# annotated LEFTOVERS cites and INVARIANTS' roll-call of executed plans already
# say in prose. A dead citation with no such word is a reader sent to nowhere.
#
# ``memory`` is the second exemption and a different kind: CLAUDE.md and
# INVARIANTS.md both point at `realmspinner-stack.md`, which is a note in the agent's
# own memory store rather than a file in this repo. A line that says "memory" is
# citing that store, and there is nothing here to resolve it against.
DEAD_CITE_MARKER = "deleted"
EXTERNAL_STORE_MARKER = "memory"
# The 2026-09-06 audit, finding docs-10 (second round): widening SOURCES to the
# five root documents surfaced THIRD-PARTY-NOTICES.md:69 --
# ``[`ATTRIBUTION.md`](tests/fixtures/humanoid/ATTRIBUTION.md)`` -- a link whose
# href resolves. DOC_MENTION re-matched the backticked *label* inside it as a
# second, bare citation of `ATTRIBUTION.md`, which does not exist at the repo
# root, even though nothing here asks a reader to open the label on its own --
# LINK already walks the href next to it. The negative lookbehind excludes
# exactly that shape (a backtick opened immediately after ``[``, i.e. a link's
# own label) without touching a real bare mention, which is never preceded by
# ``[``.
DOC_MENTION = re.compile(r"(?<!\[)`((?:\.\./|\./|docs/)?[0-9A-Za-z._/-]+\.md)`")


def _doc_citations() -> list[tuple[Path, int, str]]:
    """(source, 1-based line, target) for every backticked or linked ``.md``
    citation that is not a manual chapter and not marked as deleted."""
    found: list[tuple[Path, int, str]] = []
    for source in SOURCES:
        if not source.exists() or source.name in _HISTORIES:
            continue
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            lowered = line.lower()
            if DEAD_CITE_MARKER in lowered or EXTERNAL_STORE_MARKER in lowered:
                continue
            targets = list(DOC_MENTION.findall(line))
            targets += [t for t in LINK.findall(line) if t.endswith(".md")]
            for target in dict.fromkeys(targets):
                if "manual/" in target:
                    continue  # the tests above own these
                found.append((source, number, target))
    return found


def test_doc_mention_does_not_double_count_a_resolving_links_own_label():
    """Pins the docs-10 (second round) fix above: a backticked label inside a
    markdown link -- ``[`x.md`](href)`` -- is the link's own label, not an
    independent bare citation, so DOC_MENTION must not also match it. LINK
    already walks ``href`` on its own, so nothing here loses coverage; the
    label itself was the false positive on THIRD-PARTY-NOTICES.md:69.

    A genuine bare mention -- backticked, no enclosing link -- must still be
    caught, so this also guards against the lookbehind swallowing real cases.
    """
    linked = (
        "That file's own [`ATTRIBUTION.md`](tests/fixtures/humanoid/ATTRIBUTION.md)"
        " carries the terms."
    )
    assert DOC_MENTION.findall(linked) == []
    assert LINK.findall(linked) == ["tests/fixtures/humanoid/ATTRIBUTION.md"]

    bare = "The full table is in `docs/MODELS.md`."
    assert DOC_MENTION.findall(bare) == ["docs/MODELS.md"]


def test_the_document_citation_sweep_finds_something():
    """The same guard-on-the-guard the manual sweep gets, for the same reason:
    if ``DOC_MENTION`` or either exemption ever widens far enough to match
    nothing, every case below passes vacuously and the sweep says nothing."""
    cites = _doc_citations()
    assert len(cites) >= 10, f"expected many .md citations across SOURCES, found {cites}"


@pytest.mark.parametrize(
    "source,line,target",
    _doc_citations(),
    ids=lambda v: v.name if isinstance(v, Path) else str(v),
)
def test_a_cited_document_exists_or_says_it_is_gone(source: Path, line: int, target: str):
    path, _anchor = _resolve(source, target)
    if not path.exists():
        path = (ROOT / target.lstrip("./")).resolve()
    assert path.exists(), (
        f"{source.name}:{line} cites `{target}`, which does not exist. Either fix "
        f"the path, or -- if the document was deleted -- say so on the same line "
        f"(the word '{DEAD_CITE_MARKER}' is the marker) so a reader knows to look "
        f"in git history instead of hunting for a file that is not there."
    )


def _slug(heading: str) -> str:
    """GitHub's anchor slug, which is also what the in-app manual generates."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s]+", "-", text)


def _anchors(path: Path) -> set[str]:
    return {
        _slug(line.lstrip("#"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    }


def _manual_links() -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for source in SOURCES:
        if not source.exists() or source.name in _HISTORIES:
            continue
        text = source.read_text(encoding="utf-8")
        targets = [t for t in LINK.findall(text) if "manual/" in t and _MD_TARGET.search(t)]
        # No rewriting: a mention resolves against its own file's parent, which
        # is what ``_resolve`` already does for a link. One rule for both shapes.
        targets.extend(MENTION.findall(text))
        found.extend((source, t) for t in dict.fromkeys(targets))
    return found


def _resolve(source: Path, target: Path | str) -> tuple[Path, str]:
    """A link target -> (the file it names, the anchor it names or "")."""
    raw, _, anchor = str(target).partition("#")
    return (source.parent / raw).resolve(), anchor


# The named sources that are expected to point readers into the manual. Not
# every declared source is here, and the difference is deliberate: SECURITY.md,
# CONTRIBUTING.md and the others carry no manual link today, and asserting one
# would be asserting a wish rather than a fact. (``dev/INVARIANTS.md`` and
# ``CLAUDE.md`` used to be named here for the same reason before they moved to
# ``dev/`` on 2026-09-16; that reasoning now lives with them in
# ``dev/tests/test_external_doc_links.py``.) Everything in this set has links
# today, so a file that loses its last one is a regression rather than an edit.
LINKED_SOURCES = (
    ROOT / "README.md",
    ROOT / "docs" / "MODELS.md",
)


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.name)
def test_every_declared_source_exists(source: Path):
    """A declared source that is missing must fail *here*, not shrink coverage.

    ``docs/TODO.md`` was deleted on 2026-08-11 while still named in ``SOURCES``,
    and nothing failed: ``_manual_links`` skips a path that does not exist, so
    the only symptom was a lower total that the count guard below still passed.
    A source is either present and walked, or removed from the tuple on purpose.
    """
    assert source.exists(), (
        f"{source} is declared in SOURCES but does not exist. Either restore it "
        f"or delete its entry -- a named source that silently vanishes is how "
        f"this file stops covering what it claims to cover."
    )


def test_the_sources_actually_carry_manual_links():
    """A guard on the guard: if the regex or the filter ever stops matching,
    every assertion below passes vacuously and says nothing."""
    links = _manual_links()
    assert len(links) >= 3, f"expected several links into docs/manual/, found {links}"


@pytest.mark.parametrize("source", LINKED_SOURCES, ids=lambda p: p.name)
def test_each_linking_source_still_carries_its_own_links(source: Path):
    """Per-source, because a global count hides a file going quiet.

    The total above is satisfied by three links from one file; this is what
    notices that README stopped linking into the manual while MODELS.md grew
    two more.
    """
    mine = [target for src, target in _manual_links() if src == source]
    assert mine, f"{source.name} carries no link or mention of docs/manual/ any more"


@pytest.mark.parametrize("source,target", _manual_links(), ids=str)
def test_a_link_into_the_manual_names_a_chapter_that_exists(source: Path, target: str):
    path, _anchor = _resolve(source, target)
    assert path.exists(), (
        f"{source.name} links to {target}, which does not exist. "
        f"The manual is numbered 00-index..23-extending -- check whether the "
        f"chapter has been renumbered."
    )
    assert path.parent == MANUAL.resolve(), (
        f"{source.name} links to {target}, which is outside docs/manual/"
    )


@pytest.mark.parametrize("source,target", _manual_links(), ids=str)
def test_a_link_into_the_manual_names_an_anchor_that_exists(source: Path, target: str):
    path, anchor = _resolve(source, target)
    if not anchor or not path.exists():
        pytest.skip("no anchor, or the filename check already covers it")
    assert anchor in _anchors(path), (
        f"{source.name} links to {target}, but {path.name} has no heading "
        f"slugging to '{anchor}'. Its headings slug to: {sorted(_anchors(path))}"
    )


#: The five root documents finding docs-10 added to SOURCES. Named rather than
#: derived from SOURCES itself -- a test that reads its own fixture back out of
#: the thing it is meant to be checking would pass no matter how far SOURCES
#: shrank.
_DOCS_10_ROOT_SOURCES = {
    "INSTALL.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "THIRD-PARTY-NOTICES.md",
    "CHANGELOG.md",
}


#: The 2026-09-08 audit, finding docs-04: SECURITY.md's "Any file the app
#: opens" inventory named ``.tmx``/``.tsx`` but omitted ``.tmj``/``.tsj``,
#: Tiled's JSON spellings of the same map and tileset formats -- read from the
#: same untrusted external sources by ``plotter/tmx.py``'s ``read_tmj`` and
#: ``plotter/tsx.py``'s ``.tsj`` tileset loader, routed by suffix in
#: ``plotter_io.py``. The traversal-allowance carve-out named only ``.tmx``/
#: ``.tsx`` too. A security researcher using either list to scope what is in
#: scope would not know the JSON spelling carries the same class of risk.
SECURITY = ROOT / "SECURITY.md"


def test_security_md_file_format_list_includes_tiled_json_spellings():
    """Both SECURITY.md mentions of the Tiled XML formats must also name the
    JSON spellings the app reads through the very same code paths.

    ``plotter_io.py`` routes ``.tmj``/``.tsj`` to ``tmx.read_tmj`` and
    ``tsx``'s JSON tileset loader by suffix alongside ``.tmx``/``.tsx`` --
    there is no code-level distinction in risk between the two spellings, so
    the inventory and the traversal-allowance note must not draw one either.
    """
    text = SECURITY.read_text(encoding="utf-8")

    def _slice(start_marker: str, end_marker: str) -> str:
        start = text.index(start_marker) + len(start_marker)
        end = text.index(end_marker, start)
        return text[start:end]

    inventory = _slice("Any file the app opens.", "Path traversal")
    assert ".tmj" in inventory and ".tsj" in inventory, (
        "SECURITY.md's 'Any file the app opens' inventory names .tmx/.tsx "
        "but not their .tmj/.tsj JSON spellings, which the app reads through "
        "the same plotter_io.py suffix routing"
    )

    traversal = _slice("not in scope", "## Supported versions")
    assert ".tmj" in traversal and ".tsj" in traversal, (
        "SECURITY.md's documented .tmx/.tsx traversal allowance does not "
        "mention the same allowance in .tmj/.tsj"
    )


def test_the_five_ungated_root_docs_are_swept_for_dead_citations():
    """The 2026-09-06 audit, finding docs-10: SOURCES covered README.md,
    CLAUDE.md, dev/INVARIANTS.md, docs/MODELS.md, TODO.md, docs/COMPAT.md and
    dev/measurements/*.md, and omitted five of the docs slice's own root
    sources -- so INSTALL.md:124's citation of a manual chapter and
    THIRD-PARTY-NOTICES.md's links to docs/MODELS.md were never checked for a
    live target. A manual renumbering would have broken them silently, which is
    the exact failure this file exists to prevent for every other source.

    (CLAUDE.md, dev/INVARIANTS.md, TODO.md and dev/measurements/*.md later
    moved to dev/ on 2026-09-16 and dropped out of this file's own SOURCES --
    see the comment above it -- but that is a different, later change and does
    not touch what docs-10 itself found or fixed.)
    """
    names = {p.name for p in SOURCES}
    missing = _DOCS_10_ROOT_SOURCES - names
    assert not missing, f"SOURCES is still missing {sorted(missing)}"
