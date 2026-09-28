"""poser-jobs-04/05 (2026-09-26 audit): two sheet-delete races.

**poser-jobs-04.** ``rerender_charsheet`` used to read and validate the base
sheet *before* taking the "sheets" lock ``delete_sheet`` holds for its own
checks -- so a concurrent ``delete_sheet`` could remove the sheet in the gap
between that read and the mint, leaving a queued ``charsheet`` row naming a
``base_sheet`` already gone from disk, which failed only minutes later at
dispatch. Fixed by reading the record inside the same "sheets" hold.

**poser-jobs-05.** ``delete_sheet`` refuses while a *derived* restyle or
re-render job is in flight (``_restyle_in_flight``/``_rerender_in_flight``),
but not while the ``sheet``/``charsheet`` job that will *publish* the sheet in
the first place is still active. That job writes its PNG onto the served path
well before its sidecar (the completion marker ``list_sheets``/``read_sheet``
key on), so a delete landing in that window found only the PNG, removed it,
and the still-running job went on to publish a sidecar with no image beside
it. Fixed with ``_source_sheet_in_flight``.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from realmspinner.kernels import charsheet
from realmspinner.kernels.rig import store
from realmspinner.service import sheets as svc_sheets
from realmspinner.service import troupe as svc_troupe
from realmspinner.service.errors import Conflict


def _rigged_mesh(svc, template="humanoid"):
    job_id = svc.store.create("image", "a ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.json").write_text(json.dumps({"template": template}), "utf-8")
    svc.store.set_status(job_id, "done")
    return job_id


def _published(svc, job_id, *, logical_size=32):
    """A sheet on disk plus the finished row that produced it -- the same
    fixture ``tests/test_rerender_door.py`` uses."""
    made = svc_troupe.create_charsheet(svc, job_id, logical_size=logical_size)
    sheet_id = made["sheet_id"]
    job_dir = svc.job_dir(job_id)
    sheets = job_dir / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    (sheets / f"{sheet_id}.png").write_bytes(b"fake-png")
    layout = charsheet.resolve_layout().as_dict()
    record = {
        "id": sheet_id,
        "image": f"{sheet_id}.png",
        "created": time.time(),
        "cells": [],
        "troupe": layout,
    }
    (sheets / f"{sheet_id}.json").write_text(json.dumps(record), "utf-8")
    svc.store.set_status(made["id"], "done")
    return made["id"], sheet_id


def _runs(n=1):
    return [
        {"animation": animation, "direction": direction}
        for animation, direction, *_ in charsheet.spans()[:n]
    ]


def test_a_rerender_never_mints_a_row_naming_a_base_sheet_a_concurrent_delete_removed(
    svc, monkeypatch
):
    job_id = _rigged_mesh(svc)
    _row_id, sheet_id = _published(svc, job_id)
    job_dir = svc.job_dir(job_id)

    real_read_sheet = store.read_sheet
    read_started = threading.Event()

    def _slow_read(directory, sid):
        record = real_read_sheet(directory, sid)
        read_started.set()
        # A real window for a concurrent caller to act in -- long enough that
        # a lock actually held across this call blocks the deleter for the
        # whole span, short enough not to slow the suite down noticeably.
        time.sleep(0.3)
        return record

    monkeypatch.setattr(svc_troupe.store, "read_sheet", _slow_read)

    results: dict[str, object] = {}

    def _delete():
        read_started.wait(timeout=5)
        try:
            results["delete"] = svc_sheets.delete_sheet(svc, job_id, sheet_id)
        except Exception as exc:  # noqa: BLE001 - recorded, not raised, in a thread
            results["delete_error"] = exc

    thread = threading.Thread(target=_delete)
    thread.start()
    try:
        made = svc_troupe.rerender_charsheet(svc, job_id, sheet_id=sheet_id, subset=_runs(1))
    finally:
        thread.join(timeout=5)

    assert not thread.is_alive(), "the concurrent delete_sheet call never returned"

    base_sheet = svc.store.get(made["id"])["params"]["base_sheet"]
    assert store.read_sheet(job_dir, base_sheet) is not None, (
        "rerender_charsheet minted a charsheet row naming a base_sheet that a "
        "concurrent delete_sheet had already removed from disk -- it would "
        "fail only at dispatch, minutes later"
    )
    # And the race resolved the *other* way: the still-in-flight rerender row
    # is exactly what should have made the concurrent delete refuse.
    assert results.get("delete_error") is not None or results.get("delete") is None


def test_delete_sheet_refuses_while_the_job_that_will_publish_it_is_still_active(svc):
    job_id = _rigged_mesh(svc)
    made = svc_troupe.create_charsheet(svc, job_id, logical_size=32)
    sheet_id = made["sheet_id"]

    # The job is still queued -- never marked done -- but the worker has
    # already written its PNG onto the served path (``_q_troupe.py``'s
    # ``tmp.replace(png)``), well before the sidecar that ``read_sheet``/
    # ``list_sheets`` key completion on.
    job_dir = svc.job_dir(job_id)
    sheets_dir = job_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    png_path = sheets_dir / f"{sheet_id}.png"
    png_path.write_bytes(b"in-flight-png")

    with pytest.raises(Conflict):
        svc_sheets.delete_sheet(svc, job_id, sheet_id)

    assert png_path.exists(), (
        "delete_sheet removed the PNG of a sheet whose own publishing job is "
        "still active -- its sidecar, written later, would land with no "
        "image beside it, forever"
    )
