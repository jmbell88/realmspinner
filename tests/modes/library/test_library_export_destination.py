"""Regression for shell-01, the 2026-09-18 audit: the library's export popup
routed "Replace" through ``svc_export.export_to_folder``, which reads its
destination straight off ``svc.config.export_dir`` and never sees the
``dest``/``plan`` a "Browse..." pick published to the popup. The popup showed
the browsed folder; the write landed in the configured one instead (or
refused outright when none was configured).

``_run_export`` runs its whole loop on a background thread and blocks on
``popup.decisions`` -- the same shape ``test_shell_audit_2026_09_14.py``'s
shell-02 test already drives, reused here with the real service instead of a
stub so the write actually lands on disk where the assertion can see it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from realmspinner.service import jobs as svc_jobs
from realmspinner.studio.modes.library.ui.panes import library


def _done_job(svc, assets, name: str = "model.glb") -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / name).write_bytes(b"glb")
    svc.store.set_status(job_id, "done")
    return job_id


def _drive_export(ctx, ids, names, dest, *, browse_to: Path, decision: str) -> dict | None:
    """Start ``_run_export`` on a thread, wait for its popup, browse to
    ``browse_to``, then send ``decision`` and wait for the result."""
    result: dict[str, Any] = {}

    def run() -> None:
        result["value"] = library._run_export(
            ctx, "Save to project", ids, names, dest, as_zip=False
        )

    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while ctx.state._library_export is None and time.monotonic() < deadline:
            time.sleep(0.005)
        popup = ctx.state._library_export
        assert popup is not None, "the export task never published a popup"

        popup.decisions.put("browse")
        deadline = time.monotonic() + 5.0
        while popup.dest != browse_to and time.monotonic() < deadline:
            time.sleep(0.005)
        assert popup.dest == browse_to, "the browse pick never landed on the popup"

        popup.decisions.put(decision)
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "the export task never finished"
    finally:
        if thread.is_alive():
            ctx.state._library_export.decisions.put("cancel")
            thread.join(timeout=5.0)
    return result.get("value")


def test_replacing_after_browse_writes_to_the_browsed_folder_not_the_configured_export_dir(
    svc, tmp_path, monkeypatch
):
    """Unfixed: ``_run_export``'s "replace" branch called
    ``export_to_folder(ctx.svc, ids, names)``, which copies to
    ``svc.config.export_dir`` -- the folder configured before the popup ever
    opened -- ignoring the folder the popup was showing on screen after
    Browse.... Fixed: "Replace" writes to ``popup.plan``'s destinations, the
    same door "Keep both" already used correctly.
    """
    assets = svc.config.data_dir
    configured_dir = tmp_path / "configured-export-dir"
    browsed_dir = tmp_path / "user-picked-dir"
    configured_dir.mkdir()
    browsed_dir.mkdir()
    svc.config.export_dir = configured_dir

    job_id = _done_job(svc, assets)

    monkeypatch.setattr(library.dialogs, "select_folder", lambda *a, **k: browsed_dir)

    state = SimpleNamespace(_library_export=None)
    ctx = SimpleNamespace(state=state, svc=svc)

    result = _drive_export(
        ctx, [job_id], ["model.glb"], configured_dir, browse_to=browsed_dir, decision="replace"
    )

    assert result is not None
    written = browsed_dir / job_id / "model.glb"
    ignored = configured_dir / job_id / "model.glb"
    assert written.exists(), (
        f"Replace after Browse... did not write to the browsed folder {browsed_dir}"
    )
    assert not ignored.exists(), (
        "Replace after Browse... wrote to the configured export_dir "
        f"{configured_dir} instead of the browsed folder"
    )
    assert result["dir"] == str(browsed_dir), (
        f"the export result named {result['dir']!r} as the destination, not the browsed "
        f"folder {browsed_dir}"
    )


def test_replacing_after_browse_succeeds_with_no_export_dir_configured(svc, tmp_path, monkeypatch):
    """The same browsed-folder write, but with ``REALMSPINNER_EXPORT_DIR`` never
    set at all. Unfixed, "Replace" refused with "no export folder configured"
    even though the popup had a real, user-picked destination in hand --
    because the guard checked ``svc.config.export_dir`` instead of the plan
    that was actually about to be written.
    """
    assets = svc.config.data_dir
    browsed_dir = tmp_path / "user-picked-dir"
    browsed_dir.mkdir()
    assert svc.config.export_dir is None

    job_id = _done_job(svc, assets)

    monkeypatch.setattr(library.dialogs, "select_folder", lambda *a, **k: browsed_dir)

    state = SimpleNamespace(_library_export=None)
    ctx = SimpleNamespace(state=state, svc=svc)

    result = _drive_export(
        ctx, [job_id], ["model.glb"], Path(""), browse_to=browsed_dir, decision="replace"
    )

    assert result is not None, "Replace refused even though the popup had a browsed destination"
    assert (browsed_dir / job_id / "model.glb").exists()
