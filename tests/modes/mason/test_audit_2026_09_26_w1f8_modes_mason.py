"""Regressions for the 2026-09-26 audit, findings closed against the files
this fixer owns: ``mason/engine/terrain.py``, ``mason/engine/objout.py``,
``mason/mode.py``, ``mason/ui/view.py``, ``mason/ui/panes/tools.py``,
``mason/ui/panes/menu.py`` and ``mason/ui/panes/palette.py``.

Each test's name is the claim it makes, ``clay_tools``'s own convention
(``test_mason_mode.py``'s docstring restates it for this package).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.geom3d.gltf import Material
from realmspinner.studio import probe
from realmspinner.studio.modes.mason import mode as mason_mode
from realmspinner.studio.modes.mason import state as mason_state
from realmspinner.studio.modes.mason.engine import document as md
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import objout
from realmspinner.studio.modes.mason.engine import refs as mason_refs
from realmspinner.studio.modes.mason.engine import scene as msc
from realmspinner.studio.modes.mason.engine import terrain as T
from realmspinner.studio.modes.mason.ui import view as mason_view
from realmspinner.studio.modes.mason.ui.panes import menu as mason_menu
from realmspinner.studio.modes.mason.ui.panes import palette as mason_palette
from realmspinner.studio.modes.mason.ui.panes import tools as mason_tools

# --- mason-engine-02: the terrain mesh/GPU memo missed a config change ------


def test_terrain_mesh_rebuilds_after_a_size_or_material_config_change():
    """``MasonDoc.set_terrain_config`` (``document.py``) writes
    ``size_x``/``size_z``/``material`` onto the live ``Terrain`` with
    ``setattr`` and never rebinds ``heights`` -- so ``terrain_mesh``'s memo,
    keyed on ``heights`` identity alone, kept returning the old mesh after a
    config edit. Against the unfixed memo this fails: the second call is
    identical to the first and the mesh's own X extent never grows past the
    original 8 m even though ``size_x`` is now 100.
    """
    heights = np.zeros((5, 5), dtype="f4")
    terrain = T.Terrain(heights=heights, size_x=8.0, size_z=8.0, material=Material())
    first = T.terrain_mesh(terrain)

    terrain.size_x = 100.0
    second = T.terrain_mesh(terrain)

    assert second is not first
    extent = float(second.positions[:, 0].max() - second.positions[:, 0].min())
    assert extent == pytest.approx(100.0)


def test_terrain_mesh_rebuilds_after_a_material_only_config_change():
    """The material half of the same key: a ``Terrain`` whose ``heights`` and
    sizes are untouched but whose ``material`` was replaced (an override
    reaching through ``set_terrain_config``) must not keep drawing the old
    surface either."""
    heights = np.zeros((5, 5), dtype="f4")
    terrain = T.Terrain(heights=heights, size_x=8.0, size_z=8.0, material=Material(name="mud"))
    first = T.terrain_mesh(terrain)

    terrain.material = Material(name="grass")
    second = T.terrain_mesh(terrain)

    assert second is not first
    assert second.material.name == "grass"


def test_mason_view_terrain_upload_rebuilds_after_a_size_config_change(gl):
    """The viewport's own copy of the same memo (``MasonView.sync_terrain``),
    which used to key the GPU upload on ``heights`` identity alone the same
    way. Against the unfixed cache, ``gpu2 is gpu1`` and the second model's
    primitive still measures the old 8 m extent.
    """
    from realmspinner.kernels.geom3d import gltf as _gltf

    view = mason_view.MasonView(gl, _SimpleCtx())
    try:
        doc = md.MasonDoc()
        doc.set_terrain(
            T.Terrain(
                heights=np.zeros((5, 5), dtype="f4"),
                size_x=8.0,
                size_z=8.0,
                material=_gltf.Material(name="ground"),
            )
        )
        model1, gpu1 = view.sync_terrain(doc)
        doc.set_terrain_config(size_x=100.0, size_z=100.0)
        model2, gpu2 = view.sync_terrain(doc)

        assert gpu2 is not gpu1
        prim2 = model2.meshes[0][0]
        extent = float(prim2.positions[:, 0].max() - prim2.positions[:, 0].min())
        assert extent == pytest.approx(100.0)
    finally:
        view.release()


class _SimpleCtx:
    def __init__(self) -> None:
        self.state = mason_state.MasonState()


# --- mason-mode-01: Ctrl+A then G with a node and its own ancestor ----------


class FakeCtx:
    def __init__(self) -> None:
        self.svc = None
        self.state = _AppState()
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.result: Any = None

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info", *_a: Any, **_k: Any) -> None:
        self.toasts.append((message, kind))


class _AppState:
    def __init__(self) -> None:
        self.mason = None
        self.mode = "home"


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


def _armed_ctx() -> Any:
    ctx = FakeCtx()
    mason_mode.new_document(ctx)
    return ctx


def test_group_selected_with_a_node_and_its_selected_ancestor_keeps_the_hierarchy():
    """The 2026-09-26 audit's mason-mode-01 (was Critical): grouping a node
    together with one of its own selected ancestors used to hand every uid to
    ``move_node`` unconditionally. The group can land as a child of the very
    ancestor being grouped -- it is added beside the first selected uid's own
    parent, and a lower uid reparented under a higher one (an ordinary
    outliner drag) makes that parent's own parent itself -- so moving the
    ancestor into its own new child raised ``move_node``'s "cannot be moved
    inside its own descendant".

    Against the unfixed function this raises ``ValueError`` right out of
    ``group_selected`` (verified separately by loading the pre-fix function
    from ``git show HEAD`` in the scratchpad, per this pass's own ground
    rules about never checking out an old file): a scratch run printed
    ``REPRODUCED: ValueError: a node cannot be moved inside its own
    descendant``.
    """
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    b = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="b"))
    a = doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="a"))
    assert b.uid < a.uid
    doc.move_node(b.uid, 0, parent_uid=a.uid)
    doc.select([a.uid, b.uid])

    mason_mode.group_selected(ctx)  # must not raise

    assert not any(kind == "error" for _msg, kind in ctx.toasts)
    group_uid = next(iter(doc.selection))
    group = doc.node(group_uid)
    assert isinstance(group, nd.GroupNode)
    # b rides along inside a, which is the only thing actually moved into the
    # new group -- the hierarchy a -> b is unchanged.
    assert doc.parent_uid_of(a.uid) == group_uid
    assert doc.parent_uid_of(b.uid) == a.uid


# --- mason-mode-04: Array (radial) always centred on the node itself --------


def test_array_radial_button_places_copies_at_distinct_positions_around_a_centre(monkeypatch):
    """``centre=node.translation`` is exactly ``base_trs``'s own translation,
    so ``array_radial``'s ``t0 - centre`` was zero for every copy and all of
    them landed on top of the original -- reproduced against the unfixed
    button handler in the scratchpad (``verify_mason_mode_04.py``): 3 copies,
    1 distinct position, all at (5.0, 0.0, 0.0).
    """
    with imgui_context(monkeypatch) as imgui:
        ctx = _PaneCtx()
        doc = md.MasonDoc()
        node = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="post"))
        node.translation = np.array([5.0, 0.0, 0.0])
        doc.select([node.uid])
        mason_tools._PENDING["count"] = 4
        mason_tools._PENDING["degrees"] = 360.0

        label = "Array (radial)##masonarrayradial"

        def frame(pos=(-100.0, -100.0), down=False):
            io = imgui.get_io()
            io.add_mouse_pos_event(pos[0], pos[1])
            io.add_mouse_button_event(0, down)
            probe.begin_frame()
            imgui.new_frame()
            imgui.set_next_window_size((320.0, 900.0))
            imgui.begin("##host")
            mason_tools._array(ctx, None, doc)
            imgui.end()
            imgui.end_frame()
            return list(probe.FRAME_CONTROLS)

        found = [c for c in frame() if c.label == label]
        assert found, "no Array (radial) button drawn -- the pane changed shape"
        cx, cy = found[0].centre

        frame((cx, cy), down=False)
        frame((cx, cy), down=True)
        frame((cx, cy), down=False)

        copies = [n for n in doc.all_nodes() if n.uid != node.uid]
        assert len(copies) == 3
        positions = {tuple(np.round(c.translation, 6)) for c in copies}
        assert len(positions) == 3, "the radial array collapsed every copy onto one spot"


class _PaneCtx:
    def toast(self, *_a: Any, **_k: Any) -> None:
        pass


# --- mason-mode-07: a gizmo drag over a node and its selected parent -------


class _ViewState:
    def __init__(self) -> None:
        self.tool = "move"
        self.pivot = "median"
        self.snap = False
        self.snap_translate = 0.0
        self.snap_rotate = 0.0
        self.snap_ground = False


class _ViewCtx:
    def __init__(self) -> None:
        self.state = _ViewState()


class _NullSource:
    def resolve(self, ref: Any) -> list[Any]:
        return []


def test_gizmo_drag_moves_a_selected_child_with_its_selected_parent_by_the_drag_delta_once():
    """The 2026-09-26 audit's mason-mode-07: a drag over a node and one of
    its own selected ancestors moved the child twice -- once as its own
    independent local-translation write in ``_apply_drag``, and again for
    free because the ancestor's world transform already carries every child
    along. Headless, ``test_a_locked_node_is_not_dragged_even_though_it_is_selected``'s
    own shape one gesture over.

    Against the unfixed ``_begin_gizmo_drag`` (verified in the scratchpad
    against the pre-fix function loaded via ``git show HEAD``) a delta of 1.0
    on a child starting one metre out from its parent lands the child's
    world x at 3.0 -- the parent's own +1.0 plus the child's *own* +1.0 on
    top of a local translation that already carried the parent's move.
    """
    doc = md.MasonDoc()
    parent = doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="parent"))
    child = doc.add_node(
        nd.MeshNode(uid=nd.new_uid(), name="child", translation=m3.vec3(1.0, 0.0, 0.0)),
        parent_uid=parent.uid,
    )
    doc.select([parent.uid, child.uid])

    view = mason_view.MasonView.__new__(mason_view.MasonView)
    view.app_ctx = _ViewCtx()
    view._placed = []
    view._placed_key = None
    view._last_doc = None
    view._begin_gizmo_drag(doc, _NullSource())

    # The fix's own shape: the child, having a selected ancestor, is not one
    # of the nodes the drag writes directly.
    assert child.uid not in view._drag_start
    assert parent.uid in view._drag_start

    view._apply_drag(doc, delta=np.array([1.0, 0.0, 0.0]))

    child_world_x = float(msc.resolved_for(doc, child.uid).world[:3, 3][0])
    assert child_world_x == pytest.approx(2.0)  # 1.0 (original) + 1.0 (the delta), once


# --- mason-mode-05: a library row places at once instead of arming ---------


class _LibraryCache:
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self.jobs = jobs


class _LibraryState:
    def __init__(self) -> None:
        self.dragging_job = None


class _LibraryCtx:
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self.cache = _LibraryCache(jobs)
        self.state = _LibraryState()
        self.textures = None


def test_library_row_click_behaviour_matches_manual_chapter_17(monkeypatch):
    """Chapter 17 ("Putting something in it"): "Nothing happens yet --
    clicking *arms* the asset rather than placing it ... Click in the
    viewport, on the ground, and the asset lands there." Chapter 31's
    "Arming and placing" section promises the identical gesture for every row
    in the Assets panel. Before the fix a library row called
    ``mason_mode.place_job`` straight from the click handler -- this must
    fail against that: the row is drawn once, pressed once, and
    ``state.place_kind`` never gets a chance to hold anything because the
    click already placed the job.
    """
    with imgui_context(monkeypatch) as imgui:
        ctx = _LibraryCtx(
            [{"id": "job-1", "name": "Barrel", "stage": "model", "status": "done"}]
        )
        state = mason_state.MasonState()
        label = "Barrel##masonlib"

        def frame(pos=(-100.0, -100.0), down=False):
            io = imgui.get_io()
            io.add_mouse_pos_event(pos[0], pos[1])
            io.add_mouse_button_event(0, down)
            probe.begin_frame()
            imgui.new_frame()
            imgui.set_next_window_size((320.0, 900.0))
            imgui.begin("##host")
            mason_palette._library(ctx, state)
            imgui.end()
            imgui.end_frame()
            return list(probe.FRAME_CONTROLS)

        found = [c for c in frame() if c.label == label]
        assert found, "no library row drawn -- the pane changed shape"
        cx, cy = found[0].centre

        frame((cx, cy), down=False)
        frame((cx, cy), down=True)
        frame((cx, cy), down=False)

        assert state.place_kind == "job:job-1", "the row must arm rather than place at once"
        assert state.place_prefab == ""


def test_place_armed_places_an_armed_library_job_when_the_viewport_is_clicked():
    """The other half of the same gesture: once a row has armed
    ``state.place_kind = "job:<id>"``, ``mason_mode.place_armed`` (the
    viewport's own click handler) must actually place it, the same as it
    already does for a primitive or a light."""
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    state = mason_mode.ensure(ctx)
    job = {"id": "job-1", "name": "Barrel", "stage": "model", "status": "done"}
    ctx.cache = _LibraryCache([job])
    state.place_kind = "job:job-1"

    uid = mason_mode.place_armed(ctx, point=(3.0, 0.0, 0.0))

    assert uid is not None
    node = doc.node(uid)
    assert node is not None
    assert node.name == "Barrel"
    assert node.translation.tolist() == [3.0, 0.0, 0.0]


# --- mason-mode-11: export against an unresolved library ref ---------------


class _MixedSource:
    """One ref resolves to a box, the other -- ``job_id == "missing"`` --
    resolves to nothing, the way a library asset whose background parse has
    not landed yet does (``gltfout._Builder._mesh_for``'s own docstring)."""

    rev = 0

    def primitives(self, ref: Any) -> list[Any]:
        from realmspinner.kernels.geom3d import gltf as _gltf

        if getattr(ref, "job_id", None) == "missing":
            return []
        return [
            _gltf.Primitive(
                positions=np.array(
                    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="f4"
                ),
                indices=np.array([0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2], dtype="u4"),
                material=_gltf.Material(name="stone"),
            )
        ]


def _mixed_scene() -> md.MasonDoc:
    doc = md.MasonDoc()
    doc.add_node(
        nd.MeshNode(
            uid=nd.new_uid(),
            name="Crate",
            ref=mason_refs.LibraryRef(job_id="present", name="Crate"),
        )
    )
    doc.add_node(
        nd.MeshNode(
            uid=nd.new_uid(),
            name="Ghost",
            ref=mason_refs.LibraryRef(job_id="missing", name="Ghost"),
        )
    )
    return doc


def test_export_library_refuses_rather_than_minting_a_row_missing_geometry_when_a_ref_is_unresolved(
    svc, monkeypatch
):
    """The 2026-09-26 audit's mason-mode-11: ``export_library`` used to hand
    every scene straight to ``svc_jobs.import_mesh`` with no look at
    ``gltfout.scene_model``'s own ``unresolved`` list, so a library asset
    whose background parse had not landed yet minted a *permanent* row
    missing that prop's geometry -- there is no re-export of a row already
    minted, only a fresh one.

    Verified against the pre-fix function in the scratchpad (loaded from
    ``git show HEAD``, never checked out): with the same scene this printed
    ``jobs before=0, after=1`` and no exception.
    """
    from realmspinner.service.errors import NotReady
    from realmspinner.studio.modes.mason import assets as mason_assets

    monkeypatch.setattr(mason_assets, "ensure", lambda ctx: _MixedSource())
    ctx = _ExportCtx(svc)
    tab = mason_mode.adopt(ctx, _mixed_scene(), title="Empty Room")
    before = len(svc.store.list())

    with pytest.raises(NotReady):
        mason_mode.export_library(ctx, tab)

    assert len(svc.store.list()) == before


def test_export_glb_refuses_and_writes_nothing_when_a_ref_is_unresolved(tmp_path, monkeypatch):
    """The GLB half of the same finding: this used to write ``scene.glb``
    through ``mason_io.glb_bundle`` (which discards ``SceneExport.unresolved``
    entirely) and then toast "Exported." over a file missing the unparsed
    prop's geometry.

    Verified against the pre-fix function in the scratchpad: the same scene
    wrote a real file and returned ``{"exported": True}`` with nothing about
    the missing prop.
    """
    from realmspinner.service.errors import NotReady
    from realmspinner.studio import dialogs
    from realmspinner.studio.modes.mason import assets as mason_assets

    monkeypatch.setattr(mason_assets, "ensure", lambda ctx: _MixedSource())
    out_path = tmp_path / "scene.glb"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: str(out_path))
    ctx = _ExportCtx(None)
    tab = mason_mode.adopt(ctx, _mixed_scene(), title="Scene")

    with pytest.raises(NotReady):
        mason_mode.export_glb(ctx, tab)

    assert not out_path.exists()


class _ExportCtx:
    def __init__(self, svc: Any) -> None:
        self.svc = svc
        self.state = _AppState()
        self.settings = _Settings()
        self.cache = _ExportCache()
        self.toasts: list[tuple[str, str]] = []
        self.result: Any = None

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info", *_a: Any, **_k: Any) -> None:
        self.toasts.append((message, kind))


class _ExportCache:
    def invalidate(self) -> None:
        pass


# --- mason-engine-03: the OBJ writer never flips winding for a mirror ------


def test_obj_export_flips_winding_for_a_mirrored_node():
    """The 2026-09-26 audit's mason-engine-03: ``_write_item`` bakes world
    transforms into positions but used to carry every triangle's index order
    straight through, so a node mirrored on one axis (a negative-determinant
    world basis) exported inside-out -- its shading normal (correctly
    transformed by the inverse-transpose) points one way and the face's own
    vertex order, read by the right-hand rule, points the other.

    Checked geometrically: for each exported face, the cross product of its
    own two edges (in the order the ``f`` line gives them) must point the
    same way as its ``vn`` -- against the unfixed writer this is negative for
    the mirrored node's one face.
    """
    d = md.MasonDoc()
    mesh = nd.MeshNode(uid=nd.new_uid(), name="Mirrored", ref=mason_refs.primitive_ref("box", {}))
    mesh.scale = m3.vec3(-1.0, 1.0, 1.0)  # one negative axis: a mirror
    d.add_node(mesh)

    class _Source:
        rev = 0

        def primitives(self, ref):
            return [
                gltf.Primitive(
                    positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"),
                    indices=np.array([0, 1, 2], dtype="u4"),
                    normals=np.array([[0, 0, 1]] * 3, dtype="f4"),
                )
            ]

        def box(self, ref):
            return self.primitives(ref)[0].box()

    out = objout.obj_export(d, _Source())
    text = out.files[objout.OBJ].decode("utf-8")
    lines = text.splitlines()
    v = np.array([[float(x) for x in ln.split()[1:]] for ln in lines if ln.startswith("v ")])
    vn = np.array([[float(x) for x in ln.split()[1:]] for ln in lines if ln.startswith("vn ")])
    faces = [ln for ln in lines if ln.startswith("f ")]
    assert faces

    for face in faces:
        tokens = face[2:].split()
        v_idx = [int(t.split("/")[0]) - 1 for t in tokens]
        vn_idx = [int(t.split("/")[2]) - 1 for t in tokens]
        p0, p1, p2 = v[v_idx[0]], v[v_idx[1]], v[v_idx[2]]
        geometric_normal = np.cross(p1 - p0, p2 - p0)
        shading_normal = vn[vn_idx[0]]
        assert geometric_normal @ shading_normal > 0, "face wound backward from its own normal"


# --- mason-mode-08: Align/Distribute/Drop-to-ground ignore parent scale ----


def test_align_and_drop_to_ground_move_a_child_of_a_scaled_group_by_the_world_delta():
    """The 2026-09-26 audit's mason-mode-08: ``_apply_deltas`` (Align,
    Distribute and the sidebar's own Drop-to-ground button all share it) used
    to add a *world*-space delta straight onto ``node.translation``, which is
    expressed in the node's own *parent* space -- no conversion through the
    parent's inverse basis, the fix ``MasonView._apply_drag`` already has for
    a gizmo drag.

    A child of a group scaled 2x, moved by a world delta of (2, 0, 0), must
    land at world x = 2.0 (its own local delta is halved by the parent's
    inverse scale before the parent's forward scale doubles it back on the
    way to world space). Against the unfixed function the child's local
    translation absorbs the whole (2, 0, 0) directly, so its world position
    -- doubled again by the parent's 2x scale -- lands at 4.0.
    """
    doc = md.MasonDoc()
    group = doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="Group", scale=m3.vec3(2.0, 2.0, 2.0)))
    child = doc.add_node(
        nd.MeshNode(uid=nd.new_uid(), name="Child"), parent_uid=group.uid
    )

    mason_tools._apply_deltas(doc, {child.uid: np.array([2.0, 0.0, 0.0])})

    world_x = float(msc.resolved_for(doc, child.uid).world[:3, 3][0])
    assert world_x == pytest.approx(2.0)


def test_context_menu_drop_to_ground_shares_apply_deltas_with_the_sidebar():
    """``mason_menu._drop_to_ground`` used to carry its own hand-copied loop,
    identical to ``mason_tools._apply_deltas`` on the day it was written but
    free to drift -- and it had: the 2026-09-26 audit found the sidebar's copy
    had grown the parent-basis conversion above and the menu's copy had not.
    Delegating rather than copying is what makes a second drift impossible;
    checked at the identity rather than by re-running the arithmetic, which
    the test above already covers."""
    assert mason_menu._apply_deltas is mason_tools._apply_deltas


# --- mason-mode-10: Delete/Duplicate/Array on the Terrain node -------------


def test_deleting_the_terrain_node_removes_the_height_field_too_and_duplicate_refuses_it():
    """The 2026-09-26 audit's mason-mode-10: ``delete_selected`` used to call
    plain ``remove_node`` on a selected ``TerrainNode``, which takes the
    outliner row but leaves ``doc.terrain`` -- the document-singleton height
    field the node only refers to -- behind it, so ``add_terrain`` (which
    refuses whenever ``doc.terrain is not None``) could never give the scene
    a new ground again. ``duplicate_selected`` had the companion bug: nothing
    stopped it minting a second ``TerrainNode`` pointing at the very same
    array.

    Against the unfixed functions: after deleting the ground,
    ``doc.terrain`` is still set and a fresh ``add_terrain`` call returns
    ``None``; and duplicating the (re-added) ground leaves two
    ``TerrainNode``s in the tree.
    """
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    mason_mode.add_terrain(ctx)
    terrain_uid = next(n.uid for n in doc.all_nodes() if isinstance(n, nd.TerrainNode))
    assert doc.terrain is not None

    doc.select([terrain_uid])
    mason_mode.delete_selected(ctx)

    assert doc.terrain is None
    assert doc.node(terrain_uid) is None
    # Add ground must work again now that the singleton was actually cleared.
    new_uid = mason_mode.add_terrain(ctx)
    assert new_uid is not None
    assert doc.terrain is not None

    doc.select([new_uid])
    steps = len(doc.history)
    mason_mode.duplicate_selected(ctx)

    assert len(doc.history) == steps, "duplicating the ground must be a no-op, not a new undo step"
    terrain_nodes = [n for n in doc.all_nodes() if isinstance(n, nd.TerrainNode)]
    assert len(terrain_nodes) == 1


def test_array_button_refuses_the_terrain_node(monkeypatch):
    """The Array buttons' own half of mason-mode-10: arraying the ground
    would build several outliner rows all resolving and drawing the one
    document-singleton height field a second, third and fourth time."""
    with imgui_context(monkeypatch) as imgui:
        ctx = _PaneCtx()
        doc = md.MasonDoc()
        terrain = doc.add_node(nd.TerrainNode(uid=nd.new_uid(), name="Terrain"))
        doc.select([terrain.uid])

        def frame():
            probe.begin_frame()
            imgui.new_frame()
            imgui.set_next_window_size((320.0, 900.0))
            imgui.begin("##host")
            mason_tools._array(ctx, None, doc)
            imgui.end()
            imgui.end_frame()
            return list(probe.FRAME_CONTROLS)

        drawn = frame()
        radial = next(c for c in drawn if c.label == "Array (radial)##masonarrayradial")
        linear = next(c for c in drawn if c.label == "Array (linear)##masonarraylinear")
        assert radial.enabled is False
        assert linear.enabled is False
