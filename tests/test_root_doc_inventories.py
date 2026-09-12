"""Regressions for the smaller entries of the 2026-09-06 ``docs`` audit: an
inventory sentence that had drifted from the thing it counts, and two figures
for one artifact that disagreed with each other.

Each of these mirrors ``tests/test_findings_followups.py``'s shape -- a claim
in prose is checked against the code or the other prose that makes it true,
so the next drift fails here instead of waiting for a reader to notice.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_contributing_test_count_is_not_a_stale_exact_number():
    """The 2026-09-06 audit, finding docs-11: CONTRIBUTING.md:17 said "The
    suite is 16,392 tests in about two minutes" while collection at the
    default markers reported 18,603 -- 13% higher. CONTRIBUTING.md's own
    prose warns two paragraphs later that a hand-kept number drifts, and this
    was that warning proven right about itself.

    CLAUDE.md already phrases the same fact as a range that cannot stale
    ("~2 min for 16k+ tests"); the regression is that no *exact*,
    comma-grouped test count is claimed anywhere as current fact.
    """
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    exact_count = re.search(r"\b\d{1,3}(?:,\d{3})+\s+tests\b", text)
    assert exact_count is None, (
        f"CONTRIBUTING.md claims an exact test count ({exact_count.group(0)!r}) "
        f"that collection will outgrow again; use a range like CLAUDE.md's "
        f"'~2 min for 16k+ tests' instead."
    )


def test_models_md_flux1_paragraph_names_flux1_before_dev_and_schnell():
    """The 2026-09-06 audit, finding docs-12: docs/MODELS.md's closing
    paragraph began "Both `dev` and `schnell` are click-through gated..."
    with no antecedent anywhere in the file for what `dev` and `schnell`
    *are* -- two blank lines sat where the lead sentence should be. The same
    paragraph in docs/manual/40-installation.md opens "**FLUX.1 is not
    offered; FLUX.2 klein is.**"; this is the swallowed-lead-in failure the
    2026-09-04 audit's L05 already found once in this exact file.

    Checked as the paragraph immediately before, not merely present earlier
    in the file -- "somewhere earlier" is what let the lead-in go missing
    while other FLUX mentions stayed in the file.
    """
    text = (ROOT / "docs" / "MODELS.md").read_text(encoding="utf-8")
    paragraphs = text.split("\n\n")
    target = next(
        (i for i, p in enumerate(paragraphs) if p.startswith("Both `dev` and `schnell`")),
        None,
    )
    assert target is not None and target > 0, (
        "docs/MODELS.md no longer has the 'Both `dev` and `schnell`' closing "
        "paragraph -- update this test if it moved or was reworded."
    )
    lead_in = paragraphs[target - 1]
    assert "FLUX.1" in lead_in, (
        f"the paragraph immediately before 'Both `dev` and `schnell`...' does not "
        f"name FLUX.1, so 'dev'/'schnell' have no antecedent: {lead_in!r}"
    )


#: Number words this project actually writes for a recipe count, mirroring
#: tests/test_findings_followups.py's _WORKSPACE_WORDS for the same reason:
#: derived counts should read as English, not "5 of the registered recipes".
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def test_readme_sdxl_recipe_count_matches_the_registry():
    """The 2026-09-06 audit, finding docs-19: README.md said the one SDXL
    download "powers four of the registered recipes", but a fifth entry --
    `sdxl_cfg_pag` ("SDXL 1.0 + PAG") -- shares the identical `sdxl-base-1.0`
    dir_name at zero extra download, and "PAG" appeared nowhere in README.md.

    Derived from `warlock.models.BASE_MODELS` rather than hardcoded a second
    time, the way docs-19's own fix instruction requires: the registry is the
    count that can change under this sentence, so the sentence must read it
    rather than restate it.
    """
    from warlock.models import BASE_MODELS

    sdxl_dir = BASE_MODELS["sdxl_cfg"].dir_name  # the default recipe's own weights
    sharing = [m for m in BASE_MODELS.values() if m.dir_name == sdxl_dir]
    count = len(sharing)
    assert count in _NUMBER_WORDS, f"no number word registered for {count} -- extend _NUMBER_WORDS"
    word = _NUMBER_WORDS[count]

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"powers {word} of the registered recipes" in readme, (
        f"BASE_MODELS has {count} recipes sharing dir_name {sdxl_dir!r} "
        f"({sorted(m.key for m in sharing)}), but README.md doesn't say "
        f"'powers {word} of the registered recipes'"
    )
    if count == 5:
        assert "PAG" in readme, (
            "the registry's fifth SDXL recipe is PAG (sdxl_cfg_pag) and README.md "
            "names four of the five by name -- PAG should be the fifth, not silently "
            "counted and left unnamed"
        )


def test_install_md_states_the_installer_download_size_once():
    """The 2026-09-06 audit, finding docs-20: INSTALL.md:15 said "Installer
    download: about 810 MB (846,946,556 bytes)" and INSTALL.md:25 said "size
    about 813 MB" for the same artifact -- two rounded figures for one file,
    disagreeing with each other and with the exact byte count either could
    have been derived from.
    """
    text = (ROOT / "INSTALL.md").read_text(encoding="utf-8")
    sized_under_needs = re.search(r"Installer download:\s*\*\*about (\d+) MB\*\*", text)
    sized_in_step_one = re.search(
        r"download `WarlockSetup-[^`]*`[^.]*?\*\*about (\d+) MB\*\*", text
    )
    assert sized_under_needs and sized_in_step_one, (
        "couldn't find both of INSTALL.md's installer-size sentences -- "
        "update this test if their wording changed"
    )
    assert sized_under_needs.group(1) == sized_in_step_one.group(1), (
        f"INSTALL.md states two different installer sizes: "
        f"{sized_under_needs.group(1)} MB under 'What you'll need' vs "
        f"{sized_in_step_one.group(1)} MB in Step 1"
    )
    # The exact byte count is the more precise figure the two roundings
    # should agree with; it must not itself grow a second, possibly
    # different, copy the way the two MB figures did.
    byte_figures = re.findall(r"[\d,]{9,}\s+bytes", text)
    assert len(byte_figures) <= 1, f"the exact byte figure appears more than once: {byte_figures}"


# The tests below are the 2026-09-07 audit's smaller docs findings, in the
# same shape: a claim in one file checked against the code or commit history
# that makes it true. None of them touch a module another fixer is editing --
# each reads a doc plus, at most, the one source file its own citation names
# -- so running this file alone during a fix pass cannot collide with a
# sibling fixer's edit the way importing a live subsystem package could.


def test_claude_md_rebuild_comment_does_not_claim_the_native_dll_rebuilds_by_default():
    """The 2026-09-07 audit, finding docs-02: CLAUDE.md's ``rebuild.ps1``
    comment claimed the plain invocation rebuilds the native DLL, but
    ``-Native`` is opt-in (``scripts/rebuild.ps1``: ``[switch]$Native``,
    ``if (-not $Native)``) -- the default run only checks the DLL is present
    and silently keeps a stale one, in the script's own words "the default is
    to leave the DLL alone". A contributor editing ``native/*.c`` and testing
    against the old kernel would silently break bit-identical parity.
    """
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    line = next(row for row in text.splitlines() if "scripts\\rebuild.ps1" in row)
    assert "then the native DLL" not in line, (
        f"CLAUDE.md's rebuild.ps1 line still claims the plain invocation "
        f"rebuilds the native DLL by default: {line!r}"
    )
    assert "-Native" in line, (
        f"CLAUDE.md's rebuild.ps1 line should name -Native as the opt-in "
        f"that rebuilds the DLL: {line!r}"
    )


def test_invariants_measurement_doc_count_is_not_a_stale_exact_number():
    """The 2026-09-07 audit, finding docs-04: INVARIANTS.md:585 said the
    measurement documents under ``docs/measurements/`` numbered "a dozen
    now"; there are 54. The sentence's own hedge -- "the count is the wrong
    thing to track" -- shows the author expected it to go stale, so the fix
    deletes the count rather than replacing it with a number that will only
    go stale again the same way.
    """
    count = len(list((ROOT / "docs" / "measurements").glob("*.md")))
    assert count > 12, f"sanity: expected more than a dozen measurement docs, found {count}"

    text = (ROOT / "docs" / "INVARIANTS.md").read_text(encoding="utf-8")
    assert "a dozen now" not in text, (
        "docs/INVARIANTS.md still claims the measurement corpus is 'a dozen "
        f"now', which is stale by {count - 12} documents"
    )


def test_invariants_commit_citations_no_longer_name_the_six_dead_hashes():
    """The 2026-09-07 audit, finding docs-05: six commit hashes in
    INVARIANTS.md did not resolve with ``git rev-parse`` -- ``bf8889c``,
    ``28e35d9``, ``de87838``, ``09c64b4`` (line 585) and ``2a56df6`` (225),
    ``cc7ee724`` (290) -- while the control hashes ``1fc1573`` and
    ``3476f114`` did, so the repository is not shallow. Each was re-derived
    with ``git log --diff-filter=D`` and the equivalent merge search rather
    than dropped, since each still names a real, findable commit:
    ``09c64b4`` -> ``59f47bf7`` (docs/REPORT.md's actual deletion, which the
    commit that fixed the *other* stale reference in this same paragraph
    already named correctly), ``de87838`` -> ``f815f35e`` (docs/TODO.md's
    deletion, same date the prose already states), ``bf8889c``/``28e35d9``
    -> ``34e48562`` (REDESIGN.md and INKER_UPDATE.md were merged and deleted
    in the same commit), ``2a56df6`` -> ``a366fc3b`` (the timeline-absorbs-
    layers commit, same date the prose already states), and ``cc7ee724`` ->
    ``8b5de0af`` (the commit immediately before per-cel z-index landed,
    which is what "the wave" means in that sentence).
    """
    text = (ROOT / "docs" / "INVARIANTS.md").read_text(encoding="utf-8")
    dead = ["bf8889c", "28e35d9", "de87838", "09c64b4", "2a56df6", "cc7ee724"]
    still_present = [h for h in dead if h in text]
    assert not still_present, (
        f"docs/INVARIANTS.md still cites the dead hash(es) {still_present}, "
        "which do not resolve with git rev-parse"
    )
    # One of the finding's own two control hashes lives in this file (the
    # other, 3476f114, is TODO.md's) -- a sanity check that this test is not
    # vacuous by scrubbing every hash-shaped string rather than the six
    # named ones.
    assert "1fc1573" in text, "sanity: expected the control hash to still be present"


def test_ground_reduction_measurement_names_current_tilesheet_symbols():
    """The 2026-09-07 audit, finding docs-08: this document's own "why it
    exists" paragraph and prompt-arm description cited
    ``pipelines/ground.py:reduce_texture``, ``GROUND_VERSION`` and
    ``GROUND_NEGATIVE_PROMPT`` -- a module deleted on 2026-08-18 -- even
    though its own front-matter already says the result moved to
    ``pipelines/tilesheet.py`` unchanged. Its front-matter half-corrected
    this; the body did not.
    """
    text = (
        ROOT / "docs" / "measurements" / "2026-08-17-ground-reduction.md"
    ).read_text(encoding="utf-8")
    assert "pipelines/ground.py:reduce_texture" not in text, (
        "the measurement doc still cites the deleted pipelines/ground.py "
        "module by its old path"
    )

    paragraphs = text.split("\n\n")
    exists_para = next(p for p in paragraphs if p.startswith("This document exists"))
    assert "GROUND_VERSION" not in exists_para, (
        "the 'This document exists because' paragraph still cites the "
        "deleted GROUND_VERSION constant"
    )
    assert "tilesheet" in exists_para.lower(), (
        "the 'This document exists because' paragraph does not name "
        "pipelines/tilesheet.py, where the constants it cites now live"
    )

    arms_section = text.split("## The arms")[1].split("## Decision rules")[0]
    assert "GROUND_NEGATIVE_PROMPT" not in arms_section, (
        "the arms section still cites the deleted GROUND_NEGATIVE_PROMPT "
        "constant"
    )
    assert "SHEET_NEGATIVE_PROMPT" in arms_section, (
        "the arms section does not name SHEET_NEGATIVE_PROMPT, the constant "
        "GROUND_NEGATIVE_PROMPT was renamed to in pipelines/tilesheet.py"
    )


def test_todo_p28_cites_the_public_family_accessor():
    """The 2026-09-07 audit, finding docs-09: TODO.md's P28 cited
    ``family.ARCHETYPES[key].channels``; ``family`` (now
    ``warlock.characters.family``) keeps its archetype table private
    (``_ARCHETYPES``) and only exposes it through ``get_archetype()`` /
    ``archetypes()``.
    """
    from warlock.characters import family

    assert not hasattr(family, "ARCHETYPES"), (
        "sanity: family.ARCHETYPES exists now -- TODO.md's citation may be "
        "correct after all"
    )
    assert hasattr(family, "get_archetype"), (
        "sanity: family.get_archetype was expected to be the public accessor"
    )

    text = (ROOT / "TODO.md").read_text(encoding="utf-8")
    assert "family.ARCHETYPES[key].channels" not in text, (
        "TODO.md's P28 still cites family.ARCHETYPES, which does not exist"
    )
    assert "family.get_archetype(key).channels" in text, (
        "TODO.md's P28 does not cite the real accessor, "
        "family.get_archetype(key).channels"
    )


def test_invariants_asein_ud_owner_citation_matches_the_code():
    """The 2026-09-07 audit, finding inker-12: INVARIANTS.md:207 cited
    ``_Parse.tileset_ud_run``, which does not exist; the mechanism that
    tracks which chunk a following ``USER_DATA`` chunk belongs to is the
    generic ``_Parse.ud_owner`` tuple (``asein.py``'s own docstring on the
    field), used for tilesets among everything else.
    """
    asein_src = (
        ROOT / "src" / "warlock" / "studio" / "inker" / "asein.py"
    ).read_text(encoding="utf-8")
    assert "self.ud_owner" in asein_src, (
        "sanity: asein.py's _Parse no longer has a ud_owner field"
    )
    assert "tileset_ud_run" not in asein_src, (
        "sanity: asein.py now has a tileset_ud_run -- INVARIANTS.md's "
        "original citation may have been correct after all"
    )

    text = (ROOT / "docs" / "INVARIANTS.md").read_text(encoding="utf-8")
    assert "_Parse.tileset_ud_run" not in text, (
        "docs/INVARIANTS.md still cites _Parse.tileset_ud_run, which does "
        "not exist in asein.py"
    )
    assert "_Parse.ud_owner" in text, (
        "docs/INVARIANTS.md does not cite _Parse.ud_owner, the real "
        "mechanism"
    )
