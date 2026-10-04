"""Regression test for the 2026-09-26 audit, finding create-brief-03.

``TasksMixin._adopt_model``'s failed-GLB-load ``except`` branch returned
without clearing ``state.preview``'s rig side data (poses/sheets/bones/...),
so a failed load -- or a selection that moved on before the parse landed --
left the *previous* asset's evidence on screen. ``create.ui.stages._reached_pose``
reads exactly that evidence (``state.preview["poses"]``) to tick the Pose
stage on the rail, so a broken load could still show "Pose" as reached for
an asset that has none.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


class _FailingViewer:
    """A viewer whose GLB adoption always raises -- the corrupt-file path
    ``_adopt_model``'s ``except Exception`` branch exists for."""

    def __init__(self) -> None:
        self.pending: Path | None = None
        self.path: Path | None = None

    def adopt_model(self, model, path):
        raise RuntimeError("not a real GLB")

    def clear(self):
        # The 2026-10-03 audit (create-18): a failed adoption now empties the
        # viewer and pins the failed path, so the fake needs the real method.
        self.path = None
        self.pending = None


class _RigDataCtx:
    """A minimal ``app_ctx`` that starts with a previous asset's rig side
    data already in ``state.preview``, the way a real selection would after
    ``_refresh_rig_side_data`` last ran for a *different*, rigged job.
    """

    def __init__(self) -> None:
        self.state = SimpleNamespace(
            preview={
                "poses": ["stale-pose"],
                "sheets": ["stale-sheet"],
                "bones": ["stale-bone"],
            }
        )
        self.toasts: list[tuple] = []

    def job(self):
        # No job selected any more (or the new selection has no rig) --
        # ``_refresh_rig_side_data`` clears the preview keys and returns here.
        return None

    def toast(self, *a, **k):
        self.toasts.append((a, k))

    def submit(self, *a, **k):
        raise AssertionError("must not submit rig-data fetches when job() is None")

    def capture_thumbnail(self, *a, **k):
        raise AssertionError("no thumbnail is ever captured on a failed load")


def _app(ctx):
    from realmspinner.studio import main as main_mod

    app = main_mod.App.__new__(main_mod.App)
    app.viewer = _FailingViewer()
    app.app_ctx = ctx
    return app


def test_a_failed_glb_load_clears_the_previous_assets_rig_side_data():
    ctx = _RigDataCtx()
    app = _app(ctx)
    wanted = Path("/tmp/broken.glb")
    app.viewer.pending = wanted

    app._adopt_model(SimpleNamespace(tag=wanted, result=object()))

    assert ctx.toasts, "a failed load must still toast"
    assert "poses" not in ctx.state.preview, (
        "a failed load left the previous asset's poses in state.preview, so "
        "the Pose stage still ticked as reached for an asset with no rig"
    )
    assert "sheets" not in ctx.state.preview
    assert "bones" not in ctx.state.preview
