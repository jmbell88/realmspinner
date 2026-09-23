"""Regression tests for the 2026-09-23 (second run) audit's findings
service-04 and poser-04, both in ``service.characters.export_frames``.

Fixture shape follows ``tests/test_audit_2026_09_15_troupe.py``'s
``_mesh_with_sheet``: a finished mesh row plus a hand-built character-sheet
sidecar, small enough that a real Blender render is never needed -- these
tests corrupt the sidecar's own ``troupe`` block the way a hand edit or a
foreign job id would, which is exactly the input ``export_frames`` must not
trust blind.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from realmspinner.kernels.rig import store
from realmspinner.service import characters as svc_characters
from realmspinner.service.errors import Invalid


def _mesh_with_sheet(svc, *, runs, movements, frame_size=16, columns=2, rows=1):
    """A finished mesh plus a one-movement character sheet sidecar, with
    *runs*/*movements* substituted in verbatim -- the corruption under test."""
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


# --- service-04: a run's span must be checked before it is materialised ----


def test_export_frames_refuses_a_corrupted_run_span_before_allocating_it(svc, tmp_path, monkeypatch):  # noqa: E501
    """The 2026-09-23 audit, finding service-04: ``export_frames`` built
    ``needed_indices`` with ``range(int(run["start"]), int(run["end"]) + 1)``
    straight from the sidecar's own ``runs``, with no ceiling at all, before
    the "does this sheet even have these cells" refusal that follows it ever
    ran. The sibling fix for the same shape of bug
    (``sheet_preview_png``, troupe-03, 2026-09-15) was never carried across to
    this door. ``builtins.range`` is spied on the same way that fix's own test
    spies on ``Image.new``: the claim under test is squarely "never asked to
    count a run this large", which materialising the corrupted range first --
    and only failing once it is compared against the sidecar's real cells --
    would not prove.
    """
    movements = [
        {"key": "walk", "label": "Walk", "frames": 1, "loop": False, "duration_ms": 100,
         "directions": [{"key": "front", "label": "Front", "yaw": 0.0}]},
    ]
    # A corrupted claim no real render would ever write: charsheet.MAX_CELLS
    # is 512, and this run alone claims a span of a billion cells.
    runs = [{"movement": "walk", "direction": "front", "yaw": 0.0, "start": 0, "end": 10**9}]
    mesh_id, sheet_id = _mesh_with_sheet(svc, runs=runs, movements=movements)

    calls: list[tuple] = []
    real_range = range

    def spy_range(*args):
        calls.append(args)
        return real_range(*args)

    monkeypatch.setattr("builtins.range", spy_range)

    with pytest.raises(Invalid) as excinfo:
        svc_characters.export_frames(svc, mesh_id, sheet_id, dest_dir=tmp_path / "out")
    assert excinfo.value.field == "sheet_id"
    huge = [c for c in calls if c and c[-1] > 10_000]
    assert not huge, f"range() was asked to count the corrupted span before the refusal: {huge!r}"


# --- poser-04: a run naming a missing movement must refuse, not KeyError ---


def test_export_frames_refuses_a_sidecar_whose_run_names_a_movement_the_movements_table_no_longer_has(  # noqa: E501
    svc, tmp_path
):
    """The 2026-09-23 audit, finding poser-04: ``clips_meta``'s
    ``movements[clip]`` was a bare subscript, so a sidecar whose ``runs``
    names a movement missing from its own ``troupe.movements`` table raised
    ``KeyError`` rather than ``Invalid`` -- the same shape of bug
    ``_sheet_movements_by_key`` (service-02, this same audit) already fixed
    for the *first* read of the movements table; this is its second read,
    inside ``write()``, and it had not been fixed the same way.

    A rig's clip library can rename or drop a clip after a sheet naming it
    was already rendered, so this is reachable without any hand-editing at
    all -- not only a corrupted-sidecar scenario.
    """
    movements = [
        {"key": "walk", "label": "Walk", "frames": 1, "loop": False, "duration_ms": 100,
         "directions": [{"key": "front", "label": "Front", "yaw": 0.0}]},
    ]
    # The run names a movement ("sprint") that was since renamed or dropped
    # from the layout's own movements table -- a real, reachable staleness,
    # not only a hand-edited file.
    runs = [{"movement": "sprint", "direction": "front", "yaw": 0.0, "start": 0, "end": 0}]
    mesh_id, sheet_id = _mesh_with_sheet(svc, runs=runs, movements=movements)

    with pytest.raises(Invalid) as excinfo:
        svc_characters.export_frames(svc, mesh_id, sheet_id, dest_dir=tmp_path / "out")
    assert excinfo.value.field == "sheet_id"
