"""The 2026-10-14 audit, fixer F52: prose that named code that no longer exists.

Sits at the top of ``tests/`` because the files it reads span ``service/``,
``pipelines/`` and the package root. The behavioural tests are in
``tests/service/test_audit_2026_10_14_f52.py``.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from realmspinner.service import pixelopts

SRC = Path(__file__).resolve().parents[1] / "src" / "realmspinner"


def _text(relative: str) -> str:
    return (SRC / relative).read_text(encoding="utf-8")


def test_no_comment_in_the_gates_files_names_a_removed_troupe_or_pixel_sheet_pipeline():
    """service-gates-12: a reader chasing "pixel sheet", "troupe" or ``migrate``
    found nothing, or the wrong module."""
    errors = _text("service/errors.py")
    assert "No such job / pose / sheet / file" not in errors

    retired = _text("retired.py")
    assert "``migrate`` flips" not in retired
    assert "retire_stuck_rows" in retired

    registry = _text("models.py")
    assert "the pixel sheets run img2img" not in registry
    assert "The style LoRA a pixel sheet restyle" not in registry
    assert "Its own modelling code runs on load" not in registry


def test_queue_and_familiar_comments_cite_live_names():
    """docs-E-18: citations that pointed at a retired card and a moved function."""
    assert "clay-1.txt" not in _text("service/familiar.py")
    assert "queue.py:_apply_scale" not in _text("service/files.py")
    assert "_apply_scale" in _text("_q_mesh.py")


def test_the_files_door_comments_name_the_modules_that_exist():
    """service-assets-06: ``studio.mason_io`` is ``studio/modes/mason/fileio.py``."""
    assert "studio.mason_io" not in _text("service/files.py")
    assert (SRC / "studio" / "modes" / "mason" / "fileio.py").is_file()


def test_pixelopts_has_no_parameter_without_a_caller():
    """service-assets-06: ``size_range`` served Troupe's custom sprite size and
    nothing in the package passes it any more. The module says so, in the
    current vocabulary, rather than leaving the Troupe wording to invite a
    caller."""
    doc = inspect.getdoc(pixelopts.check_pixel_options) or ""
    helper = inspect.getdoc(pixelopts._on_the_ladder) or ""
    source = _text("service/pixelopts.py")
    assert "Troupe" not in doc and "Troupe" not in helper
    assert "_charsheet" not in source
    callers = [
        path
        for path in SRC.rglob("*.py")
        if path.name != "pixelopts.py" and re.search(r"\bsize_range\s*=", path.read_text("utf-8"))
        and "pixelopts" in path.read_text("utf-8")
    ]
    # The cleanup round deleted ``size_range`` (and ``_on_the_ladder``'s ``value_range``) outright
    # rather than documenting them as caller-less, so this pins that a parameter with no caller
    # does not come back.
    assert "size_range" not in inspect.signature(pixelopts.check_pixel_options).parameters
    assert "value_range" not in inspect.signature(pixelopts._on_the_ladder).parameters
    assert not callers, callers
