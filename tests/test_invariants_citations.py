"""``docs/INVARIANTS.md`` citing a symbol that still exists under ``src/``.

The 2026-09-08 audit found two stale citations in the same file, both the
same shape: a paragraph names a function that has since been renamed or
moved, so a reader grepping for it to find or modify the rule the paragraph
describes finds nothing.

- docs-07: the "Admission control is at the door" paragraph cited
  ``start_pixel_sheet`` three times; the function is
  ``service.sheets.create_pixel_sheet``.
- docs-08: the "And its stdin reader may never leave a read pending"
  paragraph cited ``text2image_worker._lines_from``; the function was
  extracted into a shared module and renamed to
  ``pipelines._workerio.lines_from`` (with ``peek_stdin`` as the
  ``PeekNamedPipe`` wrapper it now uses), shared with ``music_worker``.

This fixer does not own ``docs/INVARIANTS.md`` -- the corrected paragraphs
are returned to the orchestrator rather than edited here -- so these two
tests are pinned against the file as it stands *before* that paragraph is
applied, and will fail until the orchestrator applies it. That is stated
here rather than left implicit, per the brief's instruction for this pair.
"""

from __future__ import annotations

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVARIANTS = ROOT / "docs" / "INVARIANTS.md"


def _text() -> str:
    return INVARIANTS.read_text(encoding="utf-8")


def test_invariants_pixel_sheet_admission_cites_the_real_function_name():
    """docs-07: the admission-control paragraph must name the function that
    actually exists (``create_pixel_sheet``), not the pre-rename spelling.

    Fails until the orchestrator applies the returned replacement text --
    the paragraph as it stands today still says ``start_pixel_sheet``.
    """
    sheets = importlib.import_module("warlock.service.sheets")
    assert hasattr(sheets, "create_pixel_sheet"), (
        "service.sheets.create_pixel_sheet does not exist -- the finding's "
        "premise (the function was renamed, not removed) no longer holds"
    )
    assert not hasattr(sheets, "start_pixel_sheet"), (
        "service.sheets.start_pixel_sheet exists again -- the citation this "
        "test guards may have been correct all along"
    )

    text = _text()
    assert "start_pixel_sheet" not in text, (
        "docs/INVARIANTS.md still cites the pre-rename name "
        "'start_pixel_sheet'; it should read 'create_pixel_sheet'"
    )
    assert "create_pixel_sheet" in text, (
        "docs/INVARIANTS.md's admission-control paragraph should cite "
        "service.sheets.create_pixel_sheet by name"
    )


def test_invariants_stdin_reader_paragraph_cites_the_shared_workerio_module():
    """docs-08: the stdin-reader paragraph must cite the shared module the
    reader actually lives in now (``pipelines._workerio``), not the
    ``text2image_worker._lines_from`` spelling that predates the extraction
    into a module ``music_worker`` also uses.

    Fails until the orchestrator applies the returned replacement text --
    the paragraph as it stands today still says ``text2image_worker._lines_from``.
    """
    workerio = importlib.import_module("warlock.pipelines._workerio")
    assert hasattr(workerio, "lines_from"), (
        "pipelines._workerio.lines_from does not exist -- the finding's "
        "premise (the reader was extracted into this shared module) no "
        "longer holds"
    )
    assert hasattr(workerio, "peek_stdin"), (
        "pipelines._workerio.peek_stdin does not exist -- the finding's "
        "premise about the PeekNamedPipe wrapper no longer holds"
    )

    text = _text()
    assert "text2image_worker._lines_from" not in text, (
        "docs/INVARIANTS.md still cites the pre-extraction name "
        "'text2image_worker._lines_from'"
    )
    assert "_workerio.lines_from" in text, (
        "docs/INVARIANTS.md's stdin-reader paragraph should cite "
        "pipelines._workerio.lines_from by name"
    )
