"""Regression tests for the 2026-10-11 audit, finding create-03.

``TasksMixin._adopt_model`` captured a new mesh job's card thumbnail straight
after ``viewer.adopt_model``. Adoption only uploads and frames the model; the
viewport's texture is still whatever the last ``Viewer.render`` drew, so the
card was photographed with the *previous* asset (or an empty frame) in it. The
capture now waits for the first draw of the adopted model.

Needs a standalone GL 3.3 context (the ``gl`` fixture skips without one).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from realmspinner.kernels.geom3d import gltf
from realmspinner.studio.viewer_embed import Viewer

RED = (1.0, 0.05, 0.05)
BLUE = (0.05, 0.05, 1.0)
RECT = (0, 0, 64, 64)
PROBE = (32, 40)


def _model(rgb) -> gltf.Model:
    p = np.array([[-5, -5, 0], [5, -5, 0], [0, 5, 0]], dtype="f4")
    mat = dataclasses.replace(gltf.Material(), double_sided=True, base_color_factor=(*rgb, 1.0))
    prim = gltf.Primitive(
        positions=p,
        indices=np.array([0, 1, 2], dtype="u4"),
        normals=np.tile(np.array([0, 0, 1], dtype="f4"), (3, 1)),
        material=mat,
    )
    return gltf.Model([gltf.Node(mesh=0)], [0], [[prim]], [])


class _Ctx:
    """The slice of ``AppContext`` ``_adopt_model`` touches. ``capture_thumbnail``
    records the pixel the viewport holds at the moment it is asked."""

    def __init__(self, viewer: Viewer, job_id: str = "job-b") -> None:
        self.viewer = viewer
        self.job_id = job_id
        self.svc = object()
        self.state = SimpleNamespace(preview={})
        self.captures: list[tuple[str, tuple[int, ...]]] = []
        self.has_thumb = False

    def job(self):
        return {"id": self.job_id, "files": ["thumb.png"] if self.has_thumb else []}

    def toast(self, *a, **k):
        pass

    def submit(self, *a, **k):
        return True

    def capture_thumbnail(self, job_id: str) -> None:
        # Relative, not absolute: before any draw the viewport is its 16-pixel
        # default, which is itself part of what the bug photographs.
        img = self.viewer.screenshot()
        px = img.getpixel((img.width // 2, int(img.height * PROBE[1] / RECT[3])))
        self.captures.append((job_id, px))


def _app(viewer: Viewer, ctx: _Ctx):
    from realmspinner.studio import main as main_mod

    app = main_mod.App.__new__(main_mod.App)
    app.viewer = viewer
    app.app_ctx = ctx
    return app


def _adopt(app, viewer: Viewer, rgb, name: str) -> None:
    path = Path(name)
    viewer.pending = path
    app._adopt_model(SimpleNamespace(tag=path, result=_model(rgb)))


def _frame(app, viewer: Viewer) -> None:
    """One frame of the shell: draw the viewport, then service the request."""
    viewer.render(RECT, 0.016)
    service = getattr(app, "_service_thumbnail_request", None)
    if service is not None:
        service()


def _is_blue(px) -> bool:
    return px[2] > 100 and px[0] < 60


def _is_red(px) -> bool:
    return px[0] > 100 and px[2] < 60


def test_a_thumbnail_captured_on_adopt_shows_the_adopted_model_not_the_previous_frame(gl):
    viewer = Viewer(gl)
    ctx = _Ctx(viewer)
    app = _app(viewer, ctx)
    ctx.job_id = "job-a"
    _adopt(app, viewer, RED, "a.glb")
    _frame(app, viewer)
    ctx.captures.clear()

    ctx.job_id = "job-b"
    _adopt(app, viewer, BLUE, "b.glb")
    _frame(app, viewer)

    assert [c[0] for c in ctx.captures] == ["job-b"], ctx.captures
    px = ctx.captures[0][1]
    assert _is_blue(px), (
        f"the card for the blue model was photographed as {px}: the capture ran before "
        "the viewport had drawn the adopted model, so it shows the previous (red) frame"
    )


def test_the_capture_waits_for_a_draw_and_happens_once(gl):
    viewer = Viewer(gl)
    ctx = _Ctx(viewer)
    app = _app(viewer, ctx)
    _adopt(app, viewer, BLUE, "b.glb")

    service = getattr(app, "_service_thumbnail_request", None)
    assert service is not None
    service()  # no draw yet: nothing may be captured
    assert ctx.captures == []
    _frame(app, viewer)
    _frame(app, viewer)
    assert len(ctx.captures) == 1


def test_a_second_adoption_before_the_draw_replaces_the_request(gl):
    viewer = Viewer(gl)
    ctx = _Ctx(viewer)
    app = _app(viewer, ctx)
    ctx.job_id = "job-a"
    _adopt(app, viewer, RED, "a.glb")
    ctx.job_id = "job-b"
    _adopt(app, viewer, BLUE, "b.glb")
    _frame(app, viewer)

    assert [c[0] for c in ctx.captures] == ["job-b"]
    assert _is_blue(ctx.captures[0][1])


def test_a_request_is_dropped_when_the_viewer_changes_model_without_one(gl):
    viewer = Viewer(gl)
    ctx = _Ctx(viewer)
    app = _app(viewer, ctx)
    _adopt(app, viewer, RED, "a.glb")
    ctx.has_thumb = True  # the next job already has its picture
    _adopt(app, viewer, BLUE, "b.glb")
    _frame(app, viewer)
    assert ctx.captures == []

    ctx.has_thumb = False
    _adopt(app, viewer, RED, "c.glb")
    viewer.clear()
    viewer.adopt_model(_model(BLUE), Path("d.glb"))  # another path, no request
    _frame(app, viewer)
    assert ctx.captures == []
