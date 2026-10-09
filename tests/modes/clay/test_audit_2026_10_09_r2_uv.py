"""The 2026-10-09 second ``clay`` audit, texturing: clay-78, clay-80, clay-106, clay-107.

Each test's name is the claim.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_topo, uvtools
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import uv as bmuv
from realmspinner.kernels.mesh.elements import ElementSel, OpError
from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv

_UV_PANE = Path(clay_uv.__file__)


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


# --- clay-107 ----------------------------------------------------------------------


def test_pack_islands_refuses_a_negative_margin_instead_of_overlapping_islands() -> None:
    mesh = bp.cylinder()
    assert not uvtools.overlap_faces(mesh).any()
    for margin in (-0.3, -1e-6, 0.5000001, 2.0):
        with pytest.raises(OpError, match="margin"):
            uvtools.pack_islands(mesh, margin=margin)
    # The ends of the allowed range stay allowed.
    for margin in (0.0, 0.5):
        packed = uvtools.pack_islands(mesh, margin=margin)
        assert not uvtools.overlap_faces(packed).any()


# --- clay-78 -----------------------------------------------------------------------


class _Tab:
    def __init__(self, doc, view_state) -> None:
        self.doc = doc
        self.uv_view = view_state
        self.saving = False


class _State:
    def __init__(self, tab) -> None:
        self.active = tab


def _frame(ui, monkeypatch, doc, view_state) -> None:
    monkeypatch.setattr(clay_uv.clay_mode, "ensure", lambda ctx: _State(_Tab(doc, view_state)))
    ui.new_frame()
    ui.begin("##host")
    try:
        clay_uv._body(ctx=None)
    finally:
        ui.end()
        ui.end_frame()


def _box_doc() -> tuple[bd.ClayDoc, bd.Obj, clay_uv.UvPaneState]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    obj = doc.by_uid(obj.uid)
    doc.select([obj.uid])
    return doc, obj, clay_uv.UvPaneState()


def test_deleting_a_face_while_the_unwrap_page_is_open_does_not_leave_a_stale_island_selection(
    ui, monkeypatch
) -> None:
    """The 2026-10-09 second audit's clay-78: ``selected_islands`` was keyed on the
    object's uid alone, and ``uvtools.islands`` renumbers from the lowest face on
    every topology change, so deleting face 0 in the 3-D view left "island 4"
    naming what had been face 5's island (and "island 5" naming none)."""
    doc, obj, view_state = _box_doc()
    _frame(ui, monkeypatch, doc, view_state)  # the page opens on the whole box
    view_state.selected_islands = frozenset({4})
    _frame(ui, monkeypatch, doc, view_state)
    assert view_state.selected_islands == frozenset({4}), "an untouched mesh keeps its boxing"

    after, _ = ops_topo.delete_faces(obj.mesh, ElementSel(faces=np.array([0])))
    doc.set_mesh(obj.uid, after)
    _frame(ui, monkeypatch, doc, view_state)
    assert view_state.selected_islands == frozenset()


def test_an_undo_that_renumbers_the_islands_clears_the_selection_too(ui, monkeypatch) -> None:
    doc, obj, view_state = _box_doc()
    after, _ = ops_topo.delete_faces(obj.mesh, ElementSel(faces=np.array([0])))
    doc.set_mesh(obj.uid, after)
    _frame(ui, monkeypatch, doc, view_state)
    view_state.selected_islands = frozenset({3})
    doc.undo()
    _frame(ui, monkeypatch, doc, view_state)
    assert view_state.selected_islands == frozenset()


def test_a_position_only_change_keeps_the_island_selection(ui, monkeypatch) -> None:
    """A 3-D element drag replaces ``obj.mesh`` every frame and changes only
    positions: the islands are the same, so the boxing must stay."""
    from dataclasses import replace

    doc, obj, view_state = _box_doc()
    _frame(ui, monkeypatch, doc, view_state)
    view_state.selected_islands = frozenset({2, 4})
    doc.set_mesh(obj.uid, replace(obj.mesh, positions=obj.mesh.positions + np.float32(0.5)))
    _frame(ui, monkeypatch, doc, view_state)
    assert view_state.selected_islands == frozenset({2, 4})


def test_a_topology_change_under_a_live_rotate_settles_the_gesture_and_drops_the_selection(
    ui, monkeypatch
) -> None:
    doc, obj, view_state = _box_doc()
    _frame(ui, monkeypatch, doc, view_state)
    view_state.selected_islands = frozenset({4})
    ids = uvtools.islands(obj.mesh)
    before = len(doc.history)
    assert clay_uv.begin_live_transform(doc, view_state, "rotate", obj.mesh, ids, (0.1, 0.1))
    clay_uv.update_live_transform(doc, obj.uid, view_state, (0.3, 0.1))
    assert doc.history._open_gestures == 1

    current = doc.by_uid(obj.uid).mesh
    after, _ = ops_topo.delete_faces(current, ElementSel(faces=np.array([0])))
    doc.set_mesh(obj.uid, after)
    _frame(ui, monkeypatch, doc, view_state)

    assert doc.history._open_gestures == 0, "the open undo gesture must be closed, not leaked"
    assert view_state.drag_mode == ""
    assert view_state.selected_islands == frozenset()
    assert len(doc.history) >= before + 1


# --- clay-80 -----------------------------------------------------------------------


class _CountingDrawList:
    """Counts every primitive call; the draw list's other methods are never reached."""

    def __init__(self) -> None:
        self.calls = 0

    def __getattr__(self, name: str):
        if not name.startswith(("add_", "path_")):
            raise AttributeError(name)

        def count(*args, **kwargs) -> None:
            self.calls += 1

        return count


def _disjoint_quads(n: int) -> bm.Mesh:
    """*n* separate unit quads, each its own island -- the worst case for outlines."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    uv = []
    faces = []
    for i in range(n):
        x, y = (i % side) * 2.0, (i // side) * 2.0
        positions += [(x, y, 0), (x + 1, y, 0), (x + 1, y + 1, 0), (x, y + 1, 0)]
        faces.append([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3])
        u, v = (i % side) / side, (i // side) / side
        e = 0.5 / side
        uv.append([(u, v), (u + e, v), (u + e, v + e), (u, v + e)])
    return bm.from_faces(np.array(positions, dtype="f4"), faces, uv=np.array(uv, dtype="f4"))


def test_the_uv_canvas_draws_a_bounded_number_of_primitives_per_frame_on_a_large_mesh(
    ui,
) -> None:
    """The 2026-10-09 second audit's clay-80: two Python imgui calls per face per
    frame with no ceiling (29 ms at 10,000 faces, 147 ms at 52,900)."""
    from realmspinner.studio.shell import paintview

    n = 100_000
    cases = {
        "100k disjoint quads, every island boxed": _disjoint_quads(n),
        "a 100k-face grid, one island": bmuv.box_unwrap(bp.grid((1, 1), 316)),
    }
    bound = clay_uv.CANVAS_PRIMITIVE_CEILING
    assert bound < 25_000, "the ceiling is what makes the frame cost independent of the mesh"
    for label, mesh in cases.items():
        assert bm.face_count(mesh) > 90_000, label
        ids = uvtools.islands(mesh)
        # The pane's own overlap check refuses a mesh this size; a stand-in array
        # still exercises the red-outline path.
        overlap = np.zeros(bm.face_count(mesh), dtype=bool)
        overlap[::7] = True
        state = clay_uv.UvPaneState()
        paintview.fit(state.view, (1024.0, 1024.0), (400.0, 400.0))
        draw_list = _CountingDrawList()
        everything = set(np.unique(ids).tolist())
        clay_uv._faces(draw_list, state.view, (0.0, 0.0), mesh, overlap, state.geo, ids)
        clay_uv._island_outlines(
            draw_list, state.view, (0.0, 0.0), mesh, ids, everything, state.geo
        )
        assert 0 < draw_list.calls <= bound, (label, draw_list.calls)


def test_the_outline_view_draws_through_a_real_imgui_draw_list(ui) -> None:
    """A counting stub accepts any arguments; the real ``add_rect`` has its own
    signature (rounding, thickness, flags), which the first draft got wrong."""
    from realmspinner.studio.shell import paintview

    mesh = _disjoint_quads(clay_uv.FACE_DETAIL_CEILING + 100)
    ids = uvtools.islands(mesh)
    state = clay_uv.UvPaneState()
    paintview.fit(state.view, (1024.0, 1024.0), (400.0, 400.0))
    ui.new_frame()
    ui.begin("##host")
    try:
        draw_list = ui.get_window_draw_list()
        clay_uv._faces(draw_list, state.view, (0.0, 0.0), mesh, None, state.geo, ids)
        clay_uv._island_outlines(
            draw_list, state.view, (0.0, 0.0), mesh, ids, {0, 1, 2}, state.geo
        )
        # Reaching here is the assertion: neither call raised on the real draw list.
    finally:
        ui.end()
        ui.end_frame()


def test_the_canvas_plan_is_per_face_up_to_the_ceiling_and_outlines_only_past_it() -> None:
    ceiling = clay_uv.FACE_DETAIL_CEILING
    small = clay_uv.canvas_plan(ceiling, ceiling * 4, ceiling, ceiling)
    assert small.per_face
    assert small.primitives == 2 * ceiling + ceiling * 4
    big = clay_uv.canvas_plan(ceiling + 1, (ceiling + 1) * 4, ceiling + 1, ceiling + 1)
    assert not big.per_face
    huge = clay_uv.canvas_plan(10**7, 4 * 10**7, 10**7, 10**7)
    assert not huge.per_face
    assert huge.primitives <= clay_uv.CANVAS_PRIMITIVE_CEILING
    assert "island outlines" in huge.note and "10,000,000" in huge.note
    # The per-face detail the pane always had is spent only where it stays cheap:
    # the 2026-10-09 probe measured about 3 ms per 1,000 faces.
    assert small.primitives <= clay_uv.CANVAS_PRIMITIVE_CEILING


def test_picking_and_the_marquee_still_see_every_island_of_a_large_mesh() -> None:
    mesh = _disjoint_quads(10_000)
    ids = uvtools.islands(mesh)
    everything = clay_uv.islands_in_rect(mesh, ids, (-1.0, -1.0, 2.0, 2.0))
    assert everything == set(range(10_000))
    first = clay_uv.islands_in_rect(mesh, ids, (0.0, 0.0, 0.0, 0.0))
    assert first == {0}


# --- clay-106 ----------------------------------------------------------------------


def test_the_uv_pane_docstrings_describe_the_marquee_the_memo_key_and_the_inspector_page() -> None:
    text = " ".join(_UV_PANE.read_text(encoding="utf-8").split())
    stale = (
        "catches every island with at least one uv corner inside it",
        "keyed by ``mesh`` identity",
        "keyed on mesh *identity*, but",
        "as exactly as much a function of mesh identity",
        "both are exactly as much a pure function of mesh identity",
        "The same body as a tab of the Properties pane",
        "and the properties panel's own (?)",
        "the Properties pane's own (?)",
        "armed by E or R",
        "beside the outliner and the properties panel",
        "properties panel's own generator-parameter rebuild",
    )
    for needle in stale:
        assert needle not in text, f"uv.py still says {needle!r}"
    assert "Inspector's Unwrap page" in text
    keyed = " ".join((clay_uv._measurements.__doc__ or "").split())
    assert "``uv``, ``loops`` and ``starts``" in keyed
    marquee = " ".join((clay_uv.islands_in_rect.__doc__ or "").split())
    assert "whose face the box lies inside" in marquee
    state_source = inspect.getsource(clay_uv.UvPaneState).replace("#", " ")
    assert "armed by R or S" in " ".join(state_source.split())
