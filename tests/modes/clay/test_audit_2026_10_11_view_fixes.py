"""The 2026-10-11 audit's Clay viewport findings: clay-01, 44, 45, 46.

The viewport owns a renderer, so these ride the session ``gl`` fixture and skip
where there is no GPU, like every sibling in this package. The clay-46 test
drives the real viewport pane, through the fixture ``test_first_try_path.py``
already builds for it.
"""

from __future__ import annotations

import gc
import weakref
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import docmodes
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.state import ClayState
from realmspinner.studio.modes.clay.ui import view as clay_view
from realmspinner.studio.modes.clay.ui._view_overlay import _SelOverlay

from .test_first_try_path import workspace as workspace  # noqa: F401, PLC0414

RECT = (0.0, 0.0, 128.0, 96.0)


class _Ctx:
    def __init__(self) -> None:
        self.state = type("S", (), {"clay": ClayState()})()
        self.clay_view = None
        self.errors: list[str] = []

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.errors.append(message)


@pytest.fixture
def ctx():
    return _Ctx()


@pytest.fixture
def view(gl, ctx):
    v = clay_view.ClayView(gl, ctx)
    ctx.clay_view = v
    yield v
    v.release()


def _box_doc(mode: str = "face") -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    uid = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box())).uid
    doc.set_element_mode(mode)
    return doc, uid


def _drawn_counts(entry: Any) -> list[tuple[int, int]]:
    return [(len(p.positions), len(p.indices)) for p in entry.model.meshes[0]]


def _counts_of(prims: Any) -> list[tuple[int, int]]:
    return [(len(p.positions), len(p.indices)) for p in prims]


# --- clay-01: the op-drag preview draws the kernel's mesh ----------------------


def test_an_op_drag_preview_draws_the_kernels_mesh_not_the_documents(view, ctx) -> None:
    """The 2026-10-11 audit's clay-01: ``_swap_preview`` handed ``_build`` the
    previewed mesh and ``_build`` ignored it (``to_primitives(obj, materials)``),
    so an Inset drag redrew the document's own box at every value and the picture
    the pointer steered by never changed. The old guards compared ``entry.gpu``
    with ``is not``, which a rebuilt-from-the-same-mesh model also satisfies."""
    doc, uid = _box_doc()
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (64.0, 48.0)
    obj = doc.by_uid(uid)
    base_counts = _counts_of(bd.to_primitives(obj, doc.materials))

    assert view.begin_op_drag(doc, clay_ops.get("inset"))
    od = view._op_drag
    real, made = od.kernel, []

    def kernel(mesh: Any, sel: Any, **params: Any) -> Any:
        out = real(mesh, sel, **params)
        made.append(out[0])
        return out

    od.kernel = kernel
    view._motion(doc, (100.0, 48.0))
    view.draw(doc, RECT, 0.0)

    assert made, "the kernel never ran"
    expected = _counts_of(bd.to_primitives(obj, doc.materials, made[-1]))
    assert expected != base_counts, "the inset must change the geometry for this to mean anything"
    assert _drawn_counts(view._cache[uid]) == expected, (
        f"the viewport drew {_drawn_counts(view._cache[uid])}, the previewed mesh is {expected}"
    )

    # Esc puts the document's own geometry back on the next frame.
    assert view.cancel_drag(doc)
    view.draw(doc, RECT, 0.0)
    assert _drawn_counts(view._cache[uid]) == base_counts


def test_a_refused_op_drag_value_puts_the_documents_geometry_back_on_screen(view, ctx) -> None:
    """The restore half of clay-01: ``_op_drag_restore`` swaps the base mesh
    back, which only means something once the swap draws the mesh it is given."""
    doc, uid = _box_doc()
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (64.0, 48.0)
    obj = doc.by_uid(uid)
    base_counts = _counts_of(bd.to_primitives(obj, doc.materials))

    assert view.begin_op_drag(doc, clay_ops.get("inset"))
    view._motion(doc, (100.0, 48.0))
    assert _drawn_counts(view._cache[uid]) != base_counts
    view._op_drag_restore(doc, [uid])
    assert _drawn_counts(view._cache[uid]) == base_counts
    view.cancel_drag(doc)


# --- clay-44: a stale element pick costs a highlight, not the frame ------------


@pytest.mark.parametrize(
    "mode, stale",
    [
        ("face", el.ElementSel(faces=[61])),
        ("edge", el.ElementSel(edges=[(61, 62)])),
        ("vertex", el.ElementSel(verts=[61])),
    ],
)
def test_the_element_overlay_survives_a_pick_past_the_end_of_the_mesh(
    view, monkeypatch, mode: str, stale: el.ElementSel
) -> None:
    """The 2026-10-11 audit's clay-44: ``_overlay_specs`` and ``_face_outline``
    indexed ``mesh.starts[faces + 1]`` and ``adj.edge_ids(sel.edges)`` unguarded
    where ``_hover_specs`` bounds-checks, so one stale pick (reached through a
    raw ``doc.history.undo``, the path the repo names as unsafe) would raise out of
    ``draw`` on every frame for as long as it stayed."""
    doc, uid = _box_doc(mode)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    doc.element_sel[uid] = stale
    doc.touch()
    # What reaches the GPU as an index buffer: an index past the position
    # buffer is a read past the end of it, which raises nothing on the CPU
    # side and is the vertex and edge modes' half of the same defect.
    sent: list[int] = []
    real = _SelOverlay.indexed

    def spy(self: Any, indices: Any, gl_mode: int, *, hover: bool = False) -> Any:
        flat = np.asarray(indices).reshape(-1)
        if len(flat):
            sent.append(int(flat.max()))
        return real(self, indices, gl_mode, hover=hover)

    monkeypatch.setattr(_SelOverlay, "indexed", spy)

    view.draw(doc, RECT, 0.0)  # must not raise

    assert uid in view._overlays
    assert max(sent) < len(doc.by_uid(uid).mesh.positions), sent


def test_a_stale_pick_keeps_the_picks_that_are_still_in_range(view) -> None:
    doc, uid = _box_doc("face")
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    doc.element_sel[uid] = el.ElementSel(faces=[0, 61])
    doc.touch()
    view.draw(doc, RECT, 0.0)
    drawn = [spec for spec in view._overlays[uid].specs]
    # guides + fill + outline: the in-range face is still highlighted.
    assert len(drawn) >= 3


# --- clay-45: closing the last tab frees the document and its buffers ----------


def test_clearing_the_view_lets_a_closed_document_be_collected(view) -> None:
    """The 2026-10-11 audit's clay-45: ``clear`` released ``_cache`` but not the
    overlays, ``_last_doc`` or the memos, so the closed ``ClayDoc`` stayed alive
    and its overlay VBOs resident until another document was drawn."""
    doc, uid = _box_doc("face")
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    view.element_centre(doc)
    view.world_bounds(doc)
    assert view._centre_memo is not None and view._bounds_memo, "the memos pin the document"
    assert view._overlays, "the element overlay should be live before the clear"
    ref = weakref.ref(doc)

    view.clear()
    del doc
    gc.collect()

    assert ref() is None, "the cleared view still holds the closed document"
    assert not view._overlays and not view._obj_overlays


def test_a_cleared_view_redraws_the_next_frame(view) -> None:
    doc, _uid = _box_doc("object")
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    view.clear()
    assert view._render_dirty is True
    assert view._last_doc is None and view._last_render_key is None


# --- clay-46: a pending frame on a tab-switch frame ----------------------------


def _far_box_doc(x: float, size: float) -> bd.ClayDoc:
    doc = bd.ClayDoc()
    doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Box",
            mesh=bp.box(),
            translation=m3.vec3(x, 0.0, 0.0),
            scale=m3.vec3(size, size, size),
        )
    )
    return doc


def test_a_pending_frame_on_the_frame_a_tab_switches_does_not_overwrite_the_outgoing_tabs_camera(
    workspace,  # noqa: F811
    monkeypatch,
) -> None:
    """The 2026-10-11 audit's clay-46: the ``frame_pending`` block framed the live
    camera for the *new* tab before ``sync_active_camera`` and the hand-off ran,
    so the outgoing tab's stored camera was overwritten with the other
    document's framing."""
    d, ctx = workspace.driver, workspace.ctx
    state = clay_mode.ensure(ctx)
    tab_a = clay_mode.adopt(ctx, _far_box_doc(0.0, 1.0), title="A")
    tab_b = clay_mode.adopt(ctx, _far_box_doc(40.0, 6.0), title="B")
    state.activate(tab_a.uid)
    d.frame()
    d.settle(3)
    view = workspace.app.clay_view
    view.camera.look_angles(1.234, 1.0)
    d.frame()
    d.frame()
    expected_target = np.array(tab_a.view.target, dtype="f8")
    assert tab_a.view.yaw == pytest.approx(1.234)
    # An event-layer switch (Ctrl+Tab) moves ``active_uid`` outside the tab
    # bar; imgui's own bar would re-assert the tab it still has open and hide
    # the frame this test is about.
    monkeypatch.setattr(docmodes, "tab_bar", lambda *args, **kwargs: None)

    # One frame carries both events: the switch and F.
    state.activate(tab_b.uid)
    state.frame_pending = True
    state.frame_whole = True
    d.frame()

    assert tab_a.view.yaw == pytest.approx(1.234), "A's orbit was overwritten"
    assert np.allclose(np.array(tab_a.view.target, dtype="f8"), expected_target), (
        f"A's stored target {tab_a.view.target} should still be {expected_target.tolist()}"
    )
    # And B did get the framing that was asked of it.
    assert np.linalg.norm(np.array(view.camera.target) - expected_target) > 10.0
    assert state.frame_pending is False
