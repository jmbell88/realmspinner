"""shell-shell-pkg-01, the 2026-09-26 audit: exported-asset thumbnails.

``_on_task_done``'s ``plotter-``/``packwright-`` branches used to call
``self._capture_clay_thumbnail`` on an "exported_asset" completion -- which
photographs *Clay's* viewport regardless of which mode actually built the
asset, exactly the hazard Mason already has its own branch for
(``_capture_thumbnail_from(..., self.mason_view)``, see
``tests/modes/mason/test_mason_library.py::
test_the_exported_card_is_photographed_from_mason_s_own_viewport``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


class _Done:
    def __init__(self, key: str, result: Any = None) -> None:
        self.key = key
        self.result = result


class _App:
    """Just enough of ``App`` for ``_on_task_done``'s routing to run."""

    def __init__(self, ctx: Any) -> None:
        self.app_ctx = ctx
        self.clay_view = "clay-viewport"
        self.captured: list[tuple[str, Any]] = []

    def _capture_clay_thumbnail(self, job_id: str) -> None:
        self.captured.append((job_id, self.clay_view))

    def _capture_thumbnail_from(self, job_id: str, view: Any) -> None:
        self.captured.append((job_id, view))


@pytest.mark.parametrize("prefix", ["plotter-export:doc1", "packwright-export:doc1"])
def test_plotter_and_packwright_library_export_never_capture_clays_viewport(
    monkeypatch, prefix
):
    from realmspinner.studio import main as main_mod
    from realmspinner.studio.modes.packwright import mode as packwright_mode
    from realmspinner.studio.modes.plotter import mode as plotter_mode

    monkeypatch.setattr(plotter_mode, "on_task_done", lambda ctx, done: None)
    monkeypatch.setattr(packwright_mode, "on_task_done", lambda ctx, done: None)

    ctx = SimpleNamespace(state=SimpleNamespace(preview={}))
    app = _App(ctx)
    done = _Done(prefix, {"job_id": "j1", "exported_asset": True})

    main_mod.App._on_task_done(app, done)

    assert app.captured == [], f"a library export captured a viewport: {app.captured}"
