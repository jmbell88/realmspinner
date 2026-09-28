"""Regression test for the 2026-09-26 audit, finding poser-rig-04:
``list_sheets``/``list_sprite_drafts`` sorted on a raw
``s.get("created", 0.0)`` instead of the tolerant ``_created_key`` helper
``list_poses`` already uses -- so one sidecar with a string-typed
``created`` (a hand edit, a file from another program) raised
``TypeError: '<' not supported between instances of 'float' and 'str'`` out
of ``sort()`` and emptied the *whole* listing, not just that one record.
"""

from __future__ import annotations

from pathlib import Path

from realmspinner.kernels.rig import store


def _touch_png(path: Path) -> None:
    path.write_bytes(b"")


def test_list_sheets_tolerates_a_string_typed_created_sidecar(tmp_path):
    job_dir = tmp_path / "job"
    (job_dir / store.SHEET_DIR_NAME).mkdir(parents=True)
    good_id, bad_id = store.new_id(), store.new_id()
    for sid, created in ((good_id, 100.0), (bad_id, "not-a-number")):
        path = store.sheet_path(job_dir, sid)
        store.write_json_staged(path, {"id": sid, "created": created}, prefix=f".{sid}.")
        _touch_png(store.sheet_png_path(job_dir, sid))
    sheets = store.list_sheets(job_dir)
    assert {s["id"] for s in sheets} == {good_id, bad_id}


def test_list_sprite_drafts_tolerates_a_string_typed_created_sidecar(tmp_path):
    job_dir = tmp_path / "job"
    (job_dir / store.SPRITE_DIR_NAME).mkdir(parents=True)
    good_id, bad_id = store.new_id(), store.new_id()
    for did, created in ((good_id, 100.0), (bad_id, "not-a-number")):
        path = store.sprite_draft_path(job_dir, did)
        store.write_json_staged(
            path, {"id": did, "created": created, "candidates": ["a"]}, prefix=f".{did}."
        )
        _touch_png(store.sprite_draft_png_path(job_dir, did, "a"))
    drafts = store.list_sprite_drafts(job_dir)
    assert {d["id"] for d in drafts} == {good_id, bad_id}
