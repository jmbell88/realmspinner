"""poser-jobs-06/07 (2026-09-26 audit): a bare sidecar read, no ceiling on the
atlas file, and a stale manual citation, all in ``service/characters.py``.

**poser-jobs-06.** ``export_frames`` read the sidecar's ``frame_size`` with a
bare ``int(record.get("frame_size") or 0)`` -- a corrupted or hand-edited
value (a string, say) raised a raw ``TypeError``/``ValueError`` instead of the
"refused, not crashed" contract every other sidecar field in this door already
keeps (``run["start"]``/``run["end"]`` learned this lesson under
service-assets-08). Separately, its ``Image.open(png_path)`` went straight
into ``.load()`` with no ceiling on the file itself -- the sibling of
service-queue-03's fix for ``sheet_preview_png`` (2026-09-16 audit), never
carried across to this door.

**poser-jobs-07.** A comment cited ``docs/manual/34-troupe.md``; P9
(2026-09-18) folded Troupe into Poser and chapter 34 is Sirens now.
"""

from __future__ import annotations

import inspect
import io
import json
import struct
import zlib

import pytest
from PIL import Image

from realmspinner.kernels.rig import store
from realmspinner.service import characters as svc_characters
from realmspinner.service.errors import Invalid


def _mesh_with_sheet(svc, *, frame_size=16, columns=2, rows=1):
    """A finished mesh plus a one-movement character sheet sidecar -- the
    shape ``tests/service/test_audit_2026_09_23b_character_exports.py``'s
    ``_mesh_with_sheet`` already uses for this exact door."""
    from realmspinner.service import jobs as svc_jobs

    mesh_id = svc_jobs.create_job(svc, kind="text", prompt="a knight")["id"]
    svc.store.set_status(mesh_id, "done")
    job_dir = svc.job_dir(mesh_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"glb")

    sheet_id = store.new_id()
    atlas = Image.new("RGBA", (frame_size * columns, frame_size * rows), (0, 0, 0, 255))
    png_path = store.sheet_png_path(job_dir, sheet_id)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(png_path, format="PNG")

    movements = [
        {"key": "walk", "label": "Walk", "frames": columns * rows, "loop": False,
         "duration_ms": 100, "directions": [{"key": "front", "label": "Front", "yaw": 0.0}]},
    ]
    runs = [
        {"movement": "walk", "direction": "front", "yaw": 0.0, "start": 0,
         "end": columns * rows - 1},
    ]
    cells = [
        {"index": i, "x": (i % columns) * frame_size, "y": (i // columns) * frame_size,
         "w": frame_size, "h": frame_size}
        for i in range(columns * rows)
    ]
    sidecar = {
        "version": 1, "id": sheet_id, "name": "export test", "source_job": mesh_id,
        "created": 0.0, "image": png_path.name, "frame_size": frame_size,
        "columns": columns, "rows": rows, "width": frame_size * columns,
        "height": frame_size * rows, "yaws": [0.0], "poses": [], "cells": cells,
        "troupe": {
            "version": 3, "columns": columns, "movements": movements, "runs": runs,
            "cell_count": len(cells),
        },
    }
    store.sheet_path(job_dir, sheet_id).write_text(json.dumps(sidecar), encoding="utf-8")
    return mesh_id, sheet_id


def test_export_frames_refuses_a_corrupted_frame_size_instead_of_a_bare_type_error(
    svc, tmp_path
):
    mesh_id, sheet_id = _mesh_with_sheet(svc)
    job_dir = svc.job_dir(mesh_id)
    record = json.loads(store.sheet_path(job_dir, sheet_id).read_text(encoding="utf-8"))
    record["frame_size"] = "not-a-number"
    store.sheet_path(job_dir, sheet_id).write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(Invalid) as excinfo:
        svc_characters.export_frames(svc, mesh_id, sheet_id, dest_dir=tmp_path / "out")
    assert excinfo.value.field == "sheet_id"


def _giant_header_png() -> bytes:
    """A structurally valid PNG whose IHDR claims a huge width/height, built
    by patching the header of a genuine 1x1 PNG -- the same trick
    ``tests/test_audit_2026_09_15_troupe.py``'s own ``_giant_header_png``
    uses for ``sheet_preview_png``'s identical header-only guard."""
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (0, 0, 0)).save(buf, "PNG")
    data = bytearray(buf.getvalue())
    assert data[12:16] == b"IHDR"
    width = height = 9000  # > kernels.sheet.MAX_ATLAS_PX (8192)
    struct.pack_into(">II", data, 16, width, height)
    crc = zlib.crc32(bytes(data[12:29])) & 0xFFFFFFFF
    struct.pack_into(">I", data, 29, crc)
    return bytes(data)


def test_export_frames_refuses_before_decoding_an_oversized_atlas_png(svc, tmp_path, monkeypatch):
    mesh_id, sheet_id = _mesh_with_sheet(svc)
    job_dir = svc.job_dir(mesh_id)
    store.sheet_png_path(job_dir, sheet_id).write_bytes(_giant_header_png())

    calls: list[bool] = []
    real_load = Image.Image.load

    def spy_load(self, *a, **k):
        calls.append(True)
        return real_load(self, *a, **k)

    monkeypatch.setattr(Image.Image, "load", spy_load)

    with pytest.raises(Invalid) as excinfo:
        svc_characters.export_frames(svc, mesh_id, sheet_id, dest_dir=tmp_path / "out")
    assert excinfo.value.field == "sheet_id"
    assert not calls, f"Image.load was reached before the refusal: {calls!r}"


def test_the_rig_minutes_comment_names_a_manual_chapter_that_still_exists():
    """The 2026-09-26 audit, finding poser-jobs-07: this comment used to cite
    ``docs/manual/34-troupe.md``, a chapter P9 (2026-09-18) renamed to Sirens
    when Troupe folded into Poser -- Poser is chapter 26 now."""
    source = inspect.getsource(svc_characters)
    assert "34-troupe.md" not in source
    assert "26-poser.md" in source
