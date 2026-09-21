"""What a follow-up row made, drawn in the inspector.

A sprite draft, a sheet, a rig and a retexture write into their *source's*
directory, so selecting one showed settings and nothing else -- the pictures were
one directory over. ``product_images`` is pure path arithmetic, so which file
each kind draws is a table.
"""

from __future__ import annotations

from pathlib import Path

from realmspinner.studio.panes import followup_preview as fp


def _job_dir(job_id):
    return Path("lib") / job_id


def _row(kind, status="done", **params):
    return {"id": "ROW", "kind": kind, "status": status, "params": {"source_job": "SRC", **params}}


def _names(products):
    return [(p.label, p.path.as_posix(), p.nearest) for p in products]


def test_a_sprite_draft_shows_both_candidates_from_the_source_reference():
    got = fp.product_images(
        _row("sprite_synthesis", draft_id="97025e7b7334", candidates=2), _job_dir
    )
    assert _names(got) == [
        ("a", "lib/SRC/sprites/97025e7b7334.a.png", True),
        ("b", "lib/SRC/sprites/97025e7b7334.b.png", True),
    ]


def test_a_single_candidate_draft_shows_one_picture():
    """A big sheet is drawn as one candidate; asking for ``b`` would be a path
    that was never written."""
    got = fp.product_images(
        _row("sprite_synthesis", draft_id="97025e7b7334", candidates=1), _job_dir
    )
    assert [p.label for p in got] == ["a"]


def test_a_sheet_and_a_pixel_sheet_read_the_sources_sheets_directory():
    sheet = fp.product_images(_row("sheet", sheet_id="abcdef123456"), _job_dir)
    pixel = fp.product_images(_row("pixel_sheet", sheet_id="abcdef123456"), _job_dir)

    assert _names(sheet) == [("sheet", "lib/SRC/sheets/abcdef123456.png", False)]
    assert _names(pixel) == [("pixel sheet", "lib/SRC/sheets/abcdef123456.pixel.png", True)]


def test_a_rig_and_a_retexture_read_the_sources_own_pictures():
    assert _names(fp.product_images(_row("rig"), _job_dir)) == [
        ("rig check", "lib/SRC/rig_qa.png", False)
    ]
    for kind in ("retexture", "remesh"):
        assert _names(fp.product_images(_row(kind), _job_dir)) == [
            ("mesh", "lib/SRC/thumb.png", False)
        ]


def test_nothing_is_drawn_for_a_row_that_is_unfinished_or_has_no_picture():
    assert fp.product_images(_row("rig", status="running"), _job_dir) == []
    assert fp.product_images(_row("rig", status="error"), _job_dir) == []
    # Poser draws a character sheet, and a LoRA run has no image.
    assert fp.product_images(_row("charsheet", sheet_id="abcdef123456"), _job_dir) == []
    assert fp.product_images(_row("lora_train"), _job_dir) == []
    # An ordinary asset is drawn by the inspector's own sections.
    assert fp.product_images({"id": "A", "kind": "text", "status": "done"}, _job_dir) == []
    assert fp.product_images(None, _job_dir) == []


def test_a_malformed_id_never_becomes_a_path():
    """The ids come out of a params blob; ``store.sprite_draft_png_path`` raises
    on a bad one, and the inspector runs every frame."""
    assert fp.product_images(_row("sprite_synthesis", draft_id="../x"), _job_dir) == []
    assert fp.product_images(_row("sprite_synthesis"), _job_dir) == []
    assert fp.product_images(_row("sheet", sheet_id="..\\x"), _job_dir) == []


def test_pixel_art_is_drawn_at_a_whole_multiple_and_a_render_is_never_upscaled():
    assert fp.scaled((64, 32), 240, nearest=True) == (192, 96)
    assert fp.scaled((300, 300), 240, nearest=True) == (300, 300), "never below 1x"
    assert fp.scaled((256, 128), 240, nearest=False) == (240, 120)
    assert fp.scaled((100, 50), 240, nearest=False) == (100, 50), "no blur-up of a thumbnail"
    assert fp.scaled((0, 0), 240, nearest=False) == (1, 1)


def test_the_real_row_from_the_report_resolves_to_the_source_jobs_drafts():
    """85c9054a3626, as it sits in the library: kind sprite_synthesis, stage
    model, no files of its own, ``source_job`` the reference it was drawn from."""
    row = {
        "id": "85c9054a3626",
        "kind": "sprite_synthesis",
        "stage": "model",
        "status": "done",
        "files": [],
        "params": {"source_job": "cd5a18c265dc", "draft_id": "97025e7b7334", "candidates": 2},
    }
    got = fp.product_images(row, _job_dir)
    assert [p.path.as_posix() for p in got] == [
        "lib/cd5a18c265dc/sprites/97025e7b7334.a.png",
        "lib/cd5a18c265dc/sprites/97025e7b7334.b.png",
    ]
