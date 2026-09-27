"""Regression tests for the 2026-09-26 audit's ``w1f4`` fixer brief, covering
the ``studio/modes/clay`` files it owns: ``ops.py``, ``ui/_view_drag.py``,
``ui/panes/outliner.py``, ``ui/panes/props.py`` and ``ui/panes/uv.py``.

clay-ops-tail-01: Retopologize/Smart Unwrap send Blender the *evaluated*
mesh (base run through the modifier stack) and fold the result back as the
new base -- but used to leave the modifier stack itself in place, so the
next evaluation ran it a second time on a mesh that already had it baked in.

clay-panes-01: the UV pane's overlap/stretch/island memo is keyed on mesh
identity, but a live move/rotate/scale drag calls ``doc.set_mesh`` every
frame, so the memo missed on every single frame of the drag instead of only
on the frames that actually changed the mesh.

clay-view-01: a gizmo drag applied its delta to every selected object,
including a selected descendant of an already-selected, already-moved
ancestor -- so the descendant moved twice.

clay-document-03: ``ClayDoc.separate`` (not owned by this brief) adds every
new piece to ``selection`` unconditionally; ``_separate_selection`` (owned)
is the one caller that runs in face mode, where that broke the module's own
invariant ("selection holds exactly the uids with a non-empty
element_sel").

clay-document-05 (+clay-panes-04, -05): the outliner's own tree walk and
``props._relations``'s descendants query each cost O(objects) *per node
visited*, because ``ClayDoc.children_of``/``by_uid`` (not owned by this
brief) scan the whole object list every call; and the outliner drew every
row whether or not the pane could show it.
"""

from __future__ import annotations

import inspect
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.geom3d import gltf as gltf_mod
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import modifiers as mods
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import uvtools
from realmspinner.pipelines import clay_blender
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.ui import view as clay_view
from realmspinner.studio.modes.clay.ui.panes import outliner as clay_outliner
from realmspinner.studio.modes.clay.ui.panes import props as clay_props
from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv

# --- clay-ops-tail-01: retopo/unwrap must not run the modifier stack twice --


class _InlineCtx:
    def __init__(self) -> None:
        self.toasted: list[tuple[str, str]] = []
        self.inline = True

    def toast(self, message: str, level: str = "info") -> None:
        self.toasted.append((message, level))


def _fake_unwrap_bytes(
    glb: bytes, *, angle_limit: float = 66.0, island_margin: float = 0.003, timeout: Any = None,
) -> tuple[bytes, dict]:
    del angle_limit, island_margin, timeout
    model = gltf_mod.load(glb)
    objects = [{"name": n.name, "islands": 1} for n in model.nodes if n.mesh is not None]
    return glb, {"ok": True, "objects": objects}


def test_unwrap_apply_on_an_object_with_a_modifier_stack_does_not_apply_the_stack_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-09-26 audit's clay-ops-tail-01, reproduced exactly: a box (12
    triangles) under an Array modifier (count=3, 36 triangles evaluated) sent
    to a faked "Blender" that echoes the same bytes back landed at 108
    triangles after Smart Unwrap -- the 36-triangle result folded back in as
    the new base, with the Array modifier still on the stack to run a second
    time on the next evaluation.
    """
    monkeypatch.setattr(clay_blender, "available", lambda: (True, ""))
    monkeypatch.setattr(clay_blender, "unwrap_bytes", _fake_unwrap_bytes)

    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box", params={"size": 1.0})
    )
    doc.set_modifiers(obj.uid, (mods.make("array", {"count": 3.0, "offset_x": 2.0}, id=1),))
    doc.select([obj.uid])

    before_tris = clay_ops._tri_count(doc.evaluated(obj.uid))
    assert before_tris == 36, "fixture sanity: 12 base triangles times an Array of 3"

    ctx = _InlineCtx()
    assert clay_ops.run(ctx, doc, clay_ops.get("smart-unwrap")) is True

    obj_after = doc.by_uid(obj.uid)
    assert obj_after.modifiers == (), (
        "the modifier stack must be cleared -- Blender's own result already "
        "has it baked in once"
    )
    after_tris = clay_ops._tri_count(doc.evaluated(obj_after.uid))
    assert after_tris == before_tris, (
        f"expected {before_tris} triangles (the modifier baked in exactly "
        f"once), got {after_tris} -- the stack ran a second time"
    )


def test_retopo_apply_on_an_object_with_a_modifier_stack_also_clears_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retopologize's own apply gets the identical fix -- same cause, same
    shape, ``_retopo_apply`` beside ``_unwrap_apply``."""

    def _fake_retopo_bytes(
        glb: bytes, *, target_faces: int, close_holes: bool = False, seed: int = 0,
        keep_uvs: bool = False, timeout: Any = None,
    ) -> tuple[bytes, dict]:
        del target_faces, close_holes, seed, keep_uvs, timeout
        model = gltf_mod.load(glb)
        objects = [
            {"name": n.name, "method": "quadriflow", "faces_before": 12, "faces": 12, "quads": 1.0}
            for n in model.nodes
            if n.mesh is not None
        ]
        return glb, {"ok": True, "objects": objects}

    monkeypatch.setattr(clay_blender, "available", lambda: (True, ""))
    monkeypatch.setattr(clay_blender, "retopo_bytes", _fake_retopo_bytes)

    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    doc.set_modifiers(obj.uid, (mods.make("array", {"count": 3.0, "offset_x": 2.0}, id=1),))
    doc.select([obj.uid])

    before_tris = clay_ops._tri_count(doc.evaluated(obj.uid))

    ctx = _InlineCtx()
    assert clay_ops.run(ctx, doc, clay_ops.get("retopo"), target_faces=1000) is True

    obj_after = doc.by_uid(obj.uid)
    assert obj_after.modifiers == ()
    after_tris = clay_ops._tri_count(doc.evaluated(obj_after.uid))
    assert after_tris == before_tris


# --- clay-panes-01: the uv pane's measurement memo during a live drag -------


def _two_island_mesh() -> bm.Mesh:
    positions = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0],
    ]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7]]
    uv = [
        [[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]],
        [[0.6, 0.6], [1.0, 0.6], [1.0, 1.0], [0.6, 1.0]],
    ]
    return bm.from_faces(positions, faces, uv=uv)


class _FakeUvTab:
    def __init__(self, doc: bd.ClayDoc) -> None:
        self.doc = doc
        self.uv_view = clay_uv.UvPaneState()
        self.saving = False


def test_uv_pane_does_not_recompute_overlap_on_every_frame_of_a_drag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-09-26 audit's clay-panes-01, reproduced: three simulated
    frames of a live "move" drag, each handing the canvas a *new* mesh
    object the way ``apply_translate``'s own ``doc.set_mesh`` does every
    frame it runs -- ``uvtools.overlap_faces`` must be called once before
    the drag starts and not again until it ends.
    """
    doc = bd.ClayDoc()
    added = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=_two_island_mesh()))
    obj = doc.by_uid(added.uid)

    calls: list[int] = []
    original = uvtools.overlap_faces

    def counting(m: Any) -> Any:
        calls.append(1)
        return original(m)

    monkeypatch.setattr(clay_uv.uvtools, "overlap_faces", counting)

    view_state = clay_uv.UvPaneState()
    tab = _FakeUvTab(doc)

    with imgui_context(monkeypatch) as imgui:

        def frame() -> None:
            imgui.new_frame()
            imgui.begin("##host")
            try:
                clay_uv._canvas(ctx=None, tab=tab, doc=doc, obj=obj, view_state=view_state)
            finally:
                imgui.end()
                imgui.end_frame()

        frame()
        assert len(calls) == 1, "the first, not-dragging frame must measure once"

        view_state.drag_mode = "move"
        for _ in range(3):
            obj.mesh = _two_island_mesh()  # a fresh mesh object, same shape
            frame()
        assert len(calls) == 1, (
            "overlap must not be recomputed on every frame of a live drag -- "
            f"it was called {len(calls)} times across 1 pre-drag + 3 drag frames"
        )

        view_state.drag_mode = ""
        obj.mesh = _two_island_mesh()
        frame()
        assert len(calls) == 2, "and resumes once the drag ends"


# --- clay-view-01: a selected descendant of a selected ancestor ------------


class _DragState:
    def __init__(self, tool: str = "select") -> None:
        self.tool = tool
        self.snap = False
        self.snap_translate = 0.125
        self.snap_rotate = 15.0
        self.snap_vertex = False
        self.snap_edge = False
        self.snap_face = False


class _DragCtx:
    def __init__(self, tool: str = "select") -> None:
        self.state = SimpleNamespace(clay=_DragState(tool))
        self.toasted: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasted.append((message, level))


@pytest.fixture
def view(gl):
    v = clay_view.ClayView(gl, _DragCtx())
    yield v
    v.release()


def _parent_and_child() -> tuple[bd.ClayDoc, bd.Obj, bd.Obj]:
    doc = bd.ClayDoc()
    parent = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="P", mesh=bp.box(), translation=m3.vec3(3.0, 0.0, 0.0))
    )
    child = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="C", mesh=bp.box(), translation=m3.vec3(1.0, 0.0, 0.0))
    )
    doc.set_parent(child.uid, parent.uid, keep_world=True)
    return doc, parent, child


def test_dragging_a_parent_and_its_selected_child_moves_the_child_by_the_drag_delta_once(
    view,
) -> None:
    """The 2026-09-26 audit's clay-view-01, reproduced exactly: with both
    the parent and the child selected, ``_apply``'s own per-object loop used
    to move the child by 10 against the parent's 5 for an identical 5-unit
    drag, because the child's ``before_world`` already carried the parent's
    move (read fresh, every call) and then had the same delta applied to it
    directly, on top.
    """
    doc, parent, child = _parent_and_child()
    doc.select([parent.uid, child.uid])
    view.app_ctx.state.clay.tool = "move"

    assert view._begin_gizmo_drag(doc) is True
    assert child.uid not in view._drag_start, (
        "a selected descendant of a selected ancestor must not get its own "
        "drag delta -- the ancestor's own move already carries it along"
    )

    parent_world_before = doc.world_matrix(parent.uid)[:3, 3].copy()
    child_world_before = doc.world_matrix(child.uid)[:3, 3].copy()
    view._drag_origin = parent_world_before.copy()
    displacement = np.array([5.0, 0.0, 0.0])
    delta = parent_world_before + displacement  # the gizmo's own new world position

    # ``_drag_gizmo``'s own loop: the identical ``delta`` handed to every uid
    # still in ``_drag_start``.
    for uid, was in view._drag_start.items():
        view._apply(doc, doc.by_uid(uid), was, delta, view.state)

    parent_world_after = doc.world_matrix(parent.uid)[:3, 3]
    child_world_after = doc.world_matrix(child.uid)[:3, 3]
    assert np.allclose(parent_world_after, parent_world_before + displacement)
    assert np.allclose(child_world_after, child_world_before + displacement), (
        "the child must move by exactly the drag delta once, carried along "
        f"by its parent -- moved by {child_world_after - child_world_before} instead"
    )


# --- clay-document-03: separate-by-selection keeps the invariant -----------


def test_separate_by_selection_in_face_mode_keeps_selection_equal_to_element_sel_keys() -> None:
    """The 2026-09-26 audit's clay-document-03: ``ClayDoc.separate`` adds
    every new piece to ``selection`` unconditionally, but a piece just split
    off has nothing selected inside it -- in face mode, the only mode
    ``_separate_selection`` runs in, that broke the module's own invariant
    ("selection holds exactly the uids with a non-empty element_sel").
    """
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    doc.select([obj.uid])
    doc.set_element_mode("face")
    n_faces = len(obj.mesh.starts) - 1
    doc.set_element_sel(obj.uid, el.ElementSel(faces=list(range(n_faces // 2))))

    ctx = _InlineCtx()
    assert clay_ops._separate_selection(ctx, doc) is True

    assert doc.selection == set(doc.element_sel.keys()), (
        f"selection {doc.selection} must equal element_sel's own keys "
        f"{set(doc.element_sel.keys())}"
    )


# --- clay-document-05 (+clay-panes-04, -05): hierarchy queries are linear --


def _flat_family(n_children: int) -> tuple[bd.ClayDoc, bd.Obj]:
    """One parent, *n_children* direct children, all sharing one mesh
    object -- the shape both the audit's own reproduction and this test use:
    a wide, shallow tree is exactly what turns a per-node linear scan into a
    quadratic walk."""
    doc = bd.ClayDoc()
    shared_mesh = bp.box()
    parent = bd.Obj(uid=bd.new_uid(), name="Parent", mesh=shared_mesh)
    doc.objects.append(parent)
    for i in range(n_children):
        doc.objects.append(
            bd.Obj(uid=bd.new_uid(), name=f"C{i}", mesh=shared_mesh, parent=parent.uid)
        )
    return doc, parent


def test_descendants_of_a_4095_child_parent_is_linear_not_quadratic() -> None:
    """The 2026-09-26 audit's clay-document-05: reproduced at 0.40 s/frame
    against the unmemoised ``ClayDoc.descendants`` (``children_of`` scans
    every object, once per node visited). ``fast_descendants`` answers the
    identical question from one parent->children map built once per
    ``doc.rev``.
    """
    doc, parent = _flat_family(4095)

    start = time.perf_counter()
    result = clay_outliner.fast_descendants(doc, parent.uid)
    elapsed = time.perf_counter() - start

    assert len(result) == 4095
    assert elapsed < 0.2, (
        f"took {elapsed:.3f}s for a 4,095-child parent -- still a "
        "children_of scan of the whole object list per node?"
    )


def test_props_relations_reads_the_memoised_descendants_not_the_documents_own_scan() -> None:
    """clay-document-05's other half: the properties panel's own combo must
    actually call the fast path, not just have it sitting unused nearby."""
    source = inspect.getsource(clay_props._relations)
    assert "fast_descendants" in source, (
        "_relations must read clay_outliner.fast_descendants, not "
        "doc.descendants, or the panel is back to a linear scan per node"
    )


def test_outliner_tree_rows_are_linear_in_object_count() -> None:
    """The outliner's own half of clay-document-05: reproduced at
    0.63-0.72 s/frame for 4,096 objects against the unmemoised
    ``children_of``/``by_uid`` walk this pane's own ``_tree_rows`` used to
    make once per node visited.
    """
    doc, parent = _flat_family(4095)

    start = time.perf_counter()
    rows = clay_outliner._tree_rows(doc)
    elapsed = time.perf_counter() - start

    assert len(rows) == 4096
    assert elapsed < 0.2, (
        f"took {elapsed:.3f}s for 4,096 objects -- still children_of/by_uid's "
        "own linear scan per node?"
    )


def test_outliner_body_clips_rows_with_imguis_own_list_clipper() -> None:
    """clay-panes-05: every row used to be submitted to imgui whether or not
    the pane could show it. Pinned the way this codebase already pins its
    other wiring facts (``test_uv_pane_editing_controls_are_disabled_while_
    the_tab_is_saving``'s own source check) rather than measuring an actual
    scroll position.
    """
    source = inspect.getsource(clay_outliner._body)
    assert "imgui.ListClipper()" in source
    assert "clipper.begin(" in source
    assert "clipper.step()" in source
