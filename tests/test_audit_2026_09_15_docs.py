"""Regressions for the 2026-09-15 audit, findings docs-01 through docs-06.

Six small documentation drifts, each a stale fact left behind by a code
change that moved on without it:

* docs-01 -- ``docs/manual/28-inker.md``'s seam ×N paragraph still described
  the retired ``seam_ratio``/``SEAM_MAX`` statistic ("against the picture's
  own grain", amber above 3.5) after the 2026-09-07 fix moved
  ``inker_canvas.seam_text`` onto ``seam_dominance``/``SEAM_DOMINANCE_MAX``
  (1.0) -- ``dev/measurements/2026-08-30-seam-dominance.md``.
* docs-02 -- README.md's Settings -> Models paragraph said "four of the
  registered recipes share one 7 GB checkpoint"; ``sdxl_cfg_pag`` (PAG) is a
  fifth entry on the same ``sdxl-base-1.0`` weights, as the 2026-09-06
  audit's docs-19 fix already established for the sibling sentence two
  paragraphs down (``tests/test_root_doc_inventories.py``
  ``test_readme_sdxl_recipe_count_matches_the_registry``).
* docs-03 -- SECURITY.md's "Any file the app opens" inventory listed
  ``.wmap``, ``.wblk``, ``.wpack`` and ``.wsng`` but not ``.wscn``, Mason's
  native document (``service.files.MASON_SOURCE``).
* docs-04 -- ``docs/manual/40-installation.md`` said "The SDXL 1.0 weights
  serve three entries in the model list" and named only Hyper-SD, full-CFG
  and pixel-art; PAG (``sdxl_cfg_pag``) and Lightning also share the
  identical ``sdxl-base-1.0`` weights, for five total.
* docs-05 -- ``dev/INVARIANTS.md``'s Muse paragraph opens "five
  modifications are marked ``WARLOCK n/6``", counting the marker family's
  own denominator down by one; the vendored tree carries six
  (``pipelines/acestep/ATTRIBUTION.md``, ``tests/test_music_format.py``).
* docs-06 -- the same file's fps-refusal paragraph justified
  ``field="fps"`` with "``panes/troupe_settings.py`` draws no fps control
  yet"; ``_frame_rate`` has drawn one (a ``form_ui.combo("fps", ...)``)
  since fa2fee2a.

docs-05 and docs-06 are ``dev/INVARIANTS.md`` paragraphs, which this fixer
does not own; their tests here only prove the stale sentences are still
stale, so the orchestrator's replacement paragraphs have something to make
fail-then-pass.

``docs/INVARIANTS.md`` moved to ``dev/INVARIANTS.md`` on 2026-09-16, so
docs-05's test and docs-06's INVARIANTS.md half moved to
``dev/tests/test_audit_2026_09_15_docs.py``; docs-06's source-only half
(``troupe_settings.py`` still draws the fps control) stays here.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "docs" / "manual"

#: Mirrors tests/test_root_doc_inventories.py's own _NUMBER_WORDS -- this
#: project writes recipe/entry counts as English words, not digits.
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def _sdxl_sharing_count() -> int:
    from warlock.models import BASE_MODELS

    sdxl_dir = BASE_MODELS["sdxl_cfg"].dir_name
    return len([m for m in BASE_MODELS.values() if m.dir_name == sdxl_dir])


def test_manual_ch28_seam_threshold_matches_seam_dominance_max():
    """docs-01: the seam x N paragraph's amber threshold must be the real
    ``SEAM_DOMINANCE_MAX``, not the retired ratio's 3.5, and must no longer
    describe the statistic as measured "against the picture's own grain" --
    ``seam_dominance`` divides by the interior's *worst* join, not its
    "grain" (a word this chapter used for the ratio it no longer computes).
    """
    from warlock.kernels.pixel.tiling import SEAM_DOMINANCE_MAX

    chapter = (MANUAL / "28-inker.md").read_text(encoding="utf-8")
    match = re.search(r"turning amber above ([\d.]+)", chapter)
    assert match, (
        "docs/manual/28-inker.md no longer has a 'turning amber above N' sentence to check"
    )
    threshold = float(match.group(1))
    assert threshold == SEAM_DOMINANCE_MAX, (
        f"docs/manual/28-inker.md's seam paragraph says 'turning amber above "
        f"{match.group(1)}', but inker_canvas.seam_text has used "
        f"SEAM_DOMINANCE_MAX ({SEAM_DOMINANCE_MAX}) since the 2026-09-07 fix"
    )
    assert "own grain" not in chapter, (
        "docs/manual/28-inker.md still describes the seam figure as measured "
        "'against the picture's own grain' -- that is the retired "
        "seam_ratio/SEAM_MAX statistic's description, not seam_dominance's "
        "(the picture's own worst interior join)"
    )


def test_readme_models_removal_paragraph_recipe_count_matches_the_registry():
    """docs-02: the Settings -> Models removal paragraph (README.md, the
    sentence ending "...share one 7 GB checkpoint") must count the same
    ``BASE_MODELS`` sharing group the sibling sentence two paragraphs down
    already counts correctly, derived from the registry rather than
    hardcoded a second time.
    """
    count = _sdxl_sharing_count()
    assert count in _NUMBER_WORDS, f"no number word registered for {count} -- extend _NUMBER_WORDS"
    word = _NUMBER_WORDS[count]

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"([a-z]+) of the registered recipes\s*\nshare one 7 GB checkpoint", readme)
    assert match, "README.md's 'share one 7 GB checkpoint' sentence has moved or changed shape"
    assert match.group(1) == word, (
        f"BASE_MODELS has {count} recipes sharing the sdxl-base-1.0 dir_name, "
        f"but README.md's removal paragraph says '{match.group(1)} of the "
        f"registered recipes share one 7 GB checkpoint' instead of '{word}'"
    )


def test_security_md_file_format_inventory_includes_every_native_document_suffix():
    """docs-03: every ``*_SOURCE`` native-document filename ``service.files``
    defines must have its suffix named in SECURITY.md's "Any file the app
    opens" inventory -- derived from the module rather than hand-listed, so
    the next workspace's own document format enrolls itself instead of
    silently sitting outside the scope statement the way ``.wscn`` did.
    """
    from warlock.service import files

    suffixes = sorted(
        {
            Path(getattr(files, name)).suffix
            for name in dir(files)
            if name.endswith("_SOURCE") and isinstance(getattr(files, name), str)
        }
    )
    assert ".wscn" in suffixes, (
        "MASON_SOURCE moved or was renamed -- update this test's expectations"
    )

    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    missing = [suffix for suffix in suffixes if suffix not in security]
    assert not missing, (
        f"SECURITY.md's 'Any file the app opens' inventory is missing "
        f"{missing} from service.files's native-document *_SOURCE constants"
    )


def test_installation_chapter_sdxl_recipe_count_matches_the_registry():
    """docs-04: ch. 40's "The SDXL 1.0 weights serve N entries" sentence
    must count every ``BASE_MODELS`` entry sharing the ``sdxl-base-1.0``
    weights (five: Hyper-SD, full-CFG, PAG, pixel-art/LCM and Lightning),
    and must name PAG and Lightning rather than silently counting them.
    """
    count = _sdxl_sharing_count()
    assert count in _NUMBER_WORDS, f"no number word registered for {count} -- extend _NUMBER_WORDS"
    word = _NUMBER_WORDS[count]

    chapter = (MANUAL / "40-installation.md").read_text(encoding="utf-8")
    match = re.search(
        r"The SDXL 1\.0 weights serve ([a-z]+) entries in the model list.*?"
        r"Downloading them once gets you all [a-z]+\.",
        chapter,
        re.DOTALL,
    )
    assert match, (
        "docs/manual/40-installation.md's SDXL entry-count sentence has moved or changed shape"
    )
    paragraph = match.group(0)
    assert match.group(1) == word, (
        f"BASE_MODELS has {count} recipes sharing the sdxl-base-1.0 dir_name, "
        f"but ch. 40 says 'serve {match.group(1)} entries' instead of "
        f"'serve {word} entries'"
    )
    assert "PAG" in paragraph, "ch. 40's SDXL entry-count paragraph still doesn't name PAG"
    assert "Lightning" in paragraph, (
        "ch. 40's SDXL entry-count paragraph still doesn't name Lightning"
    )


def test_troupe_settings_draws_the_fps_control_the_invariant_says_it_lacks():
    """docs-06, source half: ``troupe_settings.py`` must still draw the fps
    control that INVARIANTS.md's fps-refusal paragraph (checked in
    ``dev/tests/test_audit_2026_09_15_docs.py``) says it lacks --
    ``_frame_rate`` (added in fa2fee2a) draws one, a
    ``form_ui.combo("fps", "Frame rate", ...)``.
    """
    troupe_settings_path = (
        ROOT / "src" / "warlock" / "studio" / "panes" / "troupe_settings.py"
    )
    troupe_settings = troupe_settings_path.read_text(encoding="utf-8")
    assert 'form_ui.combo(\n        "fps",' in troupe_settings or re.search(
        r'form_ui\.combo\(\s*"fps"', troupe_settings
    ), (
        "troupe_settings.py no longer draws an 'fps' combo -- re-check docs-06 "
        "against the current source"
    )
