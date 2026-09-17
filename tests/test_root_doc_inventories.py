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


def test_invariants_asein_ud_owner_citation_matches_the_code():
    """The 2026-09-07 audit, finding inker-12: INVARIANTS.md:207 cited
    ``_Parse.tileset_ud_run``, which does not exist; the mechanism that
    tracks which chunk a following ``USER_DATA`` chunk belongs to is the
    generic ``_Parse.ud_owner`` tuple (``asein.py``'s own docstring on the
    field), used for tilesets among everything else.

    The INVARIANTS.md half of this finding is checked in
    ``dev/tests/test_root_doc_inventories.py`` (the doc moved to
    ``dev/INVARIANTS.md`` on 2026-09-16); this half only needs the source
    file, which stays public.
    """
    # P3 of the restructure (dev/RESTRUCTURE.md) moved studio/inker/ to
    # warlock/kernels/pixel/, asein.py included.
    asein_src = (
        ROOT / "src" / "warlock" / "kernels" / "pixel" / "asein.py"
    ).read_text(encoding="utf-8")
    assert "self.ud_owner" in asein_src, (
        "sanity: asein.py's _Parse no longer has a ud_owner field"
    )
    assert "tileset_ud_run" not in asein_src, (
        "sanity: asein.py now has a tileset_ud_run -- INVARIANTS.md's "
        "original citation may have been correct after all"
    )
