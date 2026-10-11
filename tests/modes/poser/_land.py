"""Land Poser's asset parse in a ``FakeCtx`` test.

The 2026-10-10 audit, finding create-34: ``open_asset``'s bind no longer parses
the rig's GLB on the frame thread (``Viewer.load_model``). It dispatches
``Viewer.parse_model`` under ``ASSET_LOAD_KEY`` and leaves ``viewer.pending``
set; the viewer is only showing the asset (``pose_mode``, ``pose_job_id``)
once ``on_task_done`` has adopted the result. A ``FakeCtx.submit`` runs the
task inline and stores its result, so a test that needs "the asset is open and
bound" calls :func:`land_asset` after ``open_asset`` -- the same seam the other
asset-load tests in ``test_poser_mode.py`` drive by hand.

Opt-in on purpose: a test that is *about* the in-flight state (a parse not yet
adopted, a stale landing) simply does not call it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from realmspinner.studio.modes.poser import mode as poser_mode


def land_asset(ctx: Any) -> bool:
    """Deliver the captured ``ASSET_LOAD_KEY`` result, if one is in flight.

    -> whether a landing was delivered. Nothing is delivered when no parse was
    dispatched or when the viewer has no ``pending`` path (already adopted, or
    cleared since) -- the same freshness rule ``_land_asset_load`` applies, so
    this can be called after any ``open_asset`` without double-adopting.
    """
    key = poser_mode.ASSET_LOAD_KEY
    viewer = poser_mode.viewer_of(ctx)
    if key not in ctx.results or viewer is None or viewer.pending is None:
        return False
    poser_mode.on_task_done(
        ctx, SimpleNamespace(key=key, result=ctx.results[key], tag=ctx.tags[key])
    )
    return True
