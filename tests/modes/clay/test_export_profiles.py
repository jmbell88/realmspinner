"""Tranche 7 (integration half): engine profiles applied at export.

Three layers, each pinned separately: :func:`~.objexport.collider_export_names`
(the shared naming/numbering rule), :func:`~.objexport.claydoc_to_obj` and
``clay_mode._rename_collider_nodes`` (the two file-format-specific doors that
use it), and ``clay_mode``'s own settings-backed engine choice. The one claim
that threads all of it: **the written file's collider names and, for OBJ, its
axis/scale change with the chosen engine; the document's own ``Obj.name``
and TRS never do.**
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pygame
import pytest

from realmspinner.kernels.geom3d import glbio
from realmspinner.kernels.mesh import colliders as cl
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import engines as eng
from realmspinner.kernels.mesh import objexport
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.elements import OpError
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import state as clay_state


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    """``handle_key`` reads live modifier state on some paths -- pinned here
    the same way ``test_clay_xray_shortcut.py`` does, so this file does not
    need a real display to construct and dispatch a ``pygame`` key event."""
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


def _doc_with_collider(source_name: str = "Crate") -> tuple[bd.ClayDoc, bd.Obj, bd.Obj]:
    doc = bd.ClayDoc()
    source = doc.add_object(bd.Obj(uid=bd.new_uid(), name=source_name, mesh=bp.box()))
    collider = doc.add_collider(source.uid, cl.fit_box(source.mesh))
    return doc, source, collider


def _parse_obj_vertices(obj_text: str) -> list[np.ndarray]:
    out = []
    for line in obj_text.splitlines():
        if line.startswith("v "):
            out.append(np.array([float(x) for x in line.split()[1:4]]))
    return out


def _apply(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    """A plain affine apply, independent of anything under test -- the
    "reconstruct the expected answer with different tools" half of a test
    that is not just calling the code under test a second time."""
    return matrix[:3, :3] @ point + matrix[:3, 3]


# --- collider_export_names: the shared naming/numbering rule -----------------


def test_collider_export_names_numbers_per_source_mesh_in_document_order() -> None:
    doc = bd.ClayDoc()
    crate = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Crate", mesh=bp.box()))
    barrel = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Barrel", mesh=bp.box()))
    crate_a = doc.add_collider(crate.uid, cl.fit_box(crate.mesh))
    barrel_col = doc.add_collider(barrel.uid, cl.fit_sphere(barrel.mesh))
    crate_b = doc.add_collider(crate.uid, cl.fit_sphere(crate.mesh))

    names = objexport.collider_export_names(doc, "unreal")
    assert names[crate_a.uid] == "UBX_Crate_00"
    assert names[crate_b.uid] == "USP_Crate_01"  # second collider on Crate -> index 1
    assert names[barrel_col.uid] == "USP_Barrel_00"  # Barrel's own count starts at 0


def test_collider_export_names_returns_nothing_for_a_document_with_no_colliders() -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Crate", mesh=bp.box()))
    assert objexport.collider_export_names(doc, "godot4") == {}


def test_collider_export_names_refuses_an_unknown_engine() -> None:
    doc, _source, _collider = _doc_with_collider()
    with pytest.raises(OpError):
        objexport.collider_export_names(doc, "cryengine")


# --- claydoc_to_obj: OBJ's own naming + conversion ----------------------------


@pytest.mark.parametrize(
    ("engine_key", "expect_prefix"),
    [("unreal", "UBX_"), ("godot4", ""), ("unity", ""), ("webgl", "collider_")],
)
def test_obj_export_renames_colliders_to_each_engines_own_convention(
    engine_key: str, expect_prefix: str
) -> None:
    doc, source, collider = _doc_with_collider("Crate")
    original_name = collider.name

    obj_text, _mtl = objexport.claydoc_to_obj(doc, engine=engine_key)

    expected = eng.collider_name(engine_key, collider.collider_kind, source.name, 0)
    assert f"o {expected}" in obj_text
    assert f"o {original_name}" not in obj_text
    if expect_prefix:
        assert expected.startswith(expect_prefix)

    # The document itself never learns about any of this.
    assert doc.by_uid(collider.uid).name == original_name
    assert doc.by_uid(source.uid).name == "Crate"


def test_obj_export_with_no_engine_keeps_the_documents_own_names() -> None:
    """The pre-tranche-7 default: ``engine=None`` renames nothing."""
    doc, _source, collider = _doc_with_collider("Crate")
    obj_text, _mtl = objexport.claydoc_to_obj(doc)
    assert f"o {collider.name}" in obj_text


def test_obj_export_writes_collider_geometry_not_a_skipped_node() -> None:
    """Rule 2's "colliders export as ordinary nodes; they are not skipped" --
    the collider's own faces show up as ``f`` lines under its renamed ``o``."""
    doc, _source, collider = _doc_with_collider("Crate")
    obj_text, _mtl = objexport.claydoc_to_obj(doc, engine="unreal")
    lines = obj_text.splitlines()
    o_index = next(i for i, line in enumerate(lines) if line.startswith("o UBX_Crate_00"))
    following = lines[o_index + 1 : o_index + 40]
    assert any(line.startswith("f ") for line in following)


def test_obj_export_applies_the_engines_conversion_only_when_an_engine_is_given() -> None:
    """The OBJ half of rule 3: a chosen engine's ``obj_conversion`` composes
    with the object's own world matrix; ``engine=None`` applies none at all
    (the identity), reproducing every OBJ export before this tranche."""
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Crate", mesh=bp.box(), translation=(1.0, 2.0, 3.0))
    )
    local_corner = np.array([0.5, 0.5, 0.5])  # one corner of the unit box, in local space
    world = doc.world_matrix(obj.uid)

    # No engine: plain world placement, nothing more.
    obj_text_none, _ = objexport.claydoc_to_obj(doc)
    verts_none = _parse_obj_vertices(obj_text_none)
    expected_none = _apply(world, local_corner)
    assert any(np.allclose(v, expected_none, atol=1e-5) for v in verts_none)

    for engine_key in ("unity", "unreal"):
        conversion = eng.ENGINES[engine_key].obj_conversion
        expected = _apply(conversion, _apply(world, local_corner))
        obj_text, _ = objexport.claydoc_to_obj(doc, engine=engine_key)
        verts = _parse_obj_vertices(obj_text)
        assert any(np.allclose(v, expected, atol=1e-3) for v in verts), (
            engine_key,
            expected,
            verts,
        )
        # And it actually moved something -- a no-op conversion would make
        # this assertion meaningless.
        assert not np.allclose(expected, expected_none, atol=1e-3)


def test_obj_export_leaves_godot4_and_webgl_numerically_unchanged() -> None:
    """Both already match glTF's own convention -- their ``obj_conversion``
    is the identity (pinned in ``test_engines.py``), so passing them should
    not move a single vertex relative to ``engine=None``."""
    doc = bd.ClayDoc()
    doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Crate", mesh=bp.box(), translation=(1.0, 2.0, 3.0))
    )
    obj_text_none, _ = objexport.claydoc_to_obj(doc)
    verts_none = _parse_obj_vertices(obj_text_none)
    for engine_key in ("godot4", "webgl"):
        obj_text, _ = objexport.claydoc_to_obj(doc, engine=engine_key)
        verts = _parse_obj_vertices(obj_text)
        assert len(verts) == len(verts_none)
        assert all(np.array_equal(a, b) for a, b in zip(verts, verts_none, strict=True))


# --- clay_mode._rename_collider_nodes: GLB's own naming, no conversion -------


def test_rename_collider_nodes_uses_the_engines_convention_and_leaves_the_document_alone() -> None:
    doc, source, collider = _doc_with_collider("Crate")
    model = bd.to_model(doc)
    original_document_name = collider.name

    clay_mode._rename_collider_nodes(doc, model, "unreal")

    names = [node.name for node in model.nodes]
    assert "UBX_Crate_00" in names
    assert original_document_name not in names
    # The document itself: untouched.
    assert doc.by_uid(collider.uid).name == original_document_name


def test_rename_collider_nodes_touches_only_the_name_not_the_transform() -> None:
    doc, _source, collider = _doc_with_collider("Crate")
    model = bd.to_model(doc)
    before = next(n for n in model.nodes if n.name == collider.name)
    translation_before = before.translation.copy()
    rotation_before = before.rotation.copy()
    scale_before = before.scale.copy()
    mesh_before = before.mesh
    children_before = list(before.children)

    clay_mode._rename_collider_nodes(doc, model, "webgl")

    after = next(n for n in model.nodes if n.name == "collider_Crate_00")
    assert np.array_equal(after.translation, translation_before)
    assert np.array_equal(after.rotation, rotation_before)
    assert np.array_equal(after.scale, scale_before)
    assert after.mesh == mesh_before  # not skipped -- still a real mesh node
    assert after.children == children_before


def test_rename_collider_nodes_is_a_no_op_with_no_collider_in_the_document() -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Crate", mesh=bp.box()))
    model = bd.to_model(doc)
    names_before = [n.name for n in model.nodes]
    clay_mode._rename_collider_nodes(doc, model, "unreal")
    assert [n.name for n in model.nodes] == names_before


def test_a_parented_collider_exports_at_the_right_world_place() -> None:
    """The rename touches a name; it must not touch *where* the node ends up
    once an importer walks the hierarchy -- checked against
    ``doc.world_matrix``, the same source of truth every other export path
    in this package already trusts."""
    doc = bd.ClayDoc()
    table = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Table", mesh=bp.box(), translation=(5.0, 0.0, 0.0))
    )
    leg = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(), name="Leg", mesh=bp.box(), parent=table.uid,
            translation=(1.0, -1.0, 0.5),
        )
    )
    collider = doc.add_collider(leg.uid, cl.fit_box(leg.mesh))

    model = bd.to_model(doc)
    clay_mode._rename_collider_nodes(doc, model, "unreal")
    model.update_world()

    node = next(n for n in model.nodes if n.name == "UBX_Leg_00")
    expected_world = doc.world_matrix(collider.uid)
    assert np.allclose(node.world, expected_world, atol=1e-9)


# --- export_engine / set_export_engine: app settings, not per-tab -----------


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _AppState:
    def __init__(self) -> None:
        self.clay = None
        self.mode = "home"


class FakeCtx:
    """Reproduced locally rather than imported -- ``test_clay_mode.py``'s own
    convention (see its docstring): a private test double of another module
    is not a public fixture."""

    def __init__(self, svc: Any = None, *, accept: bool = True) -> None:
        self.svc = svc
        self.state = _AppState()
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.accept = accept
        self.result: Any = None

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        if not self.accept:
            return False
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))


def test_export_engine_defaults_to_whatever_targets_the_default_readiness_profile() -> None:
    from realmspinner.kernels.mesh import readiness

    ctx = FakeCtx()
    default_key = clay_mode.export_engine(ctx)
    assert eng.ENGINES[default_key].readiness_profile == readiness.DEFAULT_PROFILE
    assert default_key == "godot4"  # today's answer, read off the mapping above


def test_export_engine_reads_a_missing_settings_store_the_same_way(monkeypatch) -> None:
    """No ``ctx.settings`` at all -- the agent host / headless context shape
    ``clay_mode.ensure`` already tolerates -- must not raise."""
    ctx = FakeCtx()
    monkeypatch.setattr(ctx, "settings", None, raising=False)
    assert clay_mode.export_engine(ctx) == clay_mode._default_export_engine()


def test_set_export_engine_persists_and_survives_a_fresh_read() -> None:
    ctx = FakeCtx()
    clay_mode.set_export_engine(ctx, "unreal")
    # A second, independent read -- not the same call cached -- must see it,
    # which is the whole point of storing it in ``ctx.settings`` rather than
    # on any in-memory object this test could be accidentally sharing.
    assert clay_mode.export_engine(ctx) == "unreal"


def test_set_export_engine_merges_rather_than_clobbers_the_view_block() -> None:
    """The ``"clay"`` settings block already carries a ``"view"`` sub-key
    (grid/god light, ``persist``'s own docstring) -- setting the engine must
    not blow that away, the same merge rule ``persist`` itself follows."""
    ctx = FakeCtx()
    ctx.settings.set("clay", {"view": {"grid": True, "grid_size": 42.0, "god_light": False}})
    clay_mode.set_export_engine(ctx, "webgl")
    stored = ctx.settings.get("clay")
    assert stored["view"]["grid_size"] == 42.0
    assert stored["export_engine"] == "webgl"


def test_set_export_engine_refuses_an_unknown_key() -> None:
    ctx = FakeCtx()
    with pytest.raises(ValueError, match="cryengine"):
        clay_mode.set_export_engine(ctx, "cryengine")
    # Refused before anything was written.
    assert ctx.settings.get("clay") is None


# --- the two export doors, end to end ----------------------------------------


def _tab_with_collider(ctx: Any) -> tuple[clay_state.ClayTab, bd.Obj]:
    doc, _source, collider = _doc_with_collider("Crate")
    tab = clay_mode.adopt(ctx, doc, title="Scene")
    return tab, collider


def test_export_asset_renames_colliders_in_the_built_glb_under_the_persisted_engine(
    svc,
) -> None:
    ctx = FakeCtx(svc)
    clay_mode.set_export_engine(ctx, "unreal")
    tab, _collider = _tab_with_collider(ctx)

    clay_mode.export_asset(ctx, tab)
    job_id = ctx.result["job_id"]

    data = (svc.job_dir(job_id) / "model.glb").read_bytes()
    gltf_json, _rest = glbio.read_glb(data)
    names = [node.get("name", "") for node in gltf_json.get("nodes", [])]
    assert any(name.startswith("UBX_") for name in names)

    # The tab's own document, untouched -- ``build_asset`` never mutates it.
    assert tab.doc.objects[1].name == "Crate Box"


def test_build_asset_with_no_engine_renames_nothing(svc) -> None:
    """The agent's own call (``agent/tools_ops.py``) passes no ``engine`` at
    all -- this tranche must not change what that path writes."""
    doc, _source, collider = _doc_with_collider("Crate")

    job_id = clay_mode.build_asset(svc, doc, title="Scene")

    data = (svc.job_dir(job_id) / "model.glb").read_bytes()
    gltf_json, _rest = glbio.read_glb(data)
    names = {node.get("name", "") for node in gltf_json.get("nodes", [])}
    assert collider.name in names


@pytest.fixture
def _no_dialogs(monkeypatch, tmp_path):
    from realmspinner.studio import dialogs

    def _save_file(title: str, default_name: str, filt: Any) -> Any:
        del title, filt
        return tmp_path / default_name

    monkeypatch.setattr(dialogs, "save_file", _save_file)
    monkeypatch.setattr(dialogs, "open_file", lambda *a, **k: None)
    return tmp_path


def test_export_mesh_file_glb_renames_colliders_under_the_persisted_engine(
    _no_dialogs,
) -> None:
    ctx = FakeCtx()
    clay_mode.set_export_engine(ctx, "godot4")
    tab, _collider = _tab_with_collider(ctx)

    clay_mode.export_mesh_file(ctx, tab, "glb")

    written = _no_dialogs / "Scene.glb"
    gltf_json, _rest = glbio.read_glb(written.read_bytes())
    names = [node.get("name", "") for node in gltf_json.get("nodes", [])]
    assert any(name.endswith("-convcolonly") for name in names)
    assert tab.doc.objects[1].name == "Crate Box"  # document untouched


def test_export_mesh_file_obj_renames_and_converts_under_the_persisted_engine(
    _no_dialogs,
) -> None:
    ctx = FakeCtx()
    clay_mode.set_export_engine(ctx, "unity")
    tab, _collider = _tab_with_collider(ctx)

    clay_mode.export_mesh_file(ctx, tab, "obj")

    written = (_no_dialogs / "Scene.obj").read_text(encoding="utf-8")
    assert "o Crate_collider_00" in written
    assert "o Crate Box" not in written


# --- Escape cancels an armed knife -------------------------------------------


class _FakeView:
    """Just enough of ``ClayView`` for ``_escape``'s knife branch: an armed
    flag and a ``cancel_drag`` that records whether it ran and disarms."""

    def __init__(self, *, armed: bool) -> None:
        self._knife_armed = armed
        self.cancel_drag_calls = 0

    def cancel_drag(self, doc: Any) -> bool:
        del doc
        self.cancel_drag_calls += 1
        self._knife_armed = False
        return True


def test_escape_cancels_an_armed_knife_before_the_staged_clear() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=np.array([0], dtype="i4")))
    view = _FakeView(armed=True)
    state = clay_state.ClayState()
    tab = clay_state.ClayTab(doc=doc)

    clay_mode._escape(state, tab, doc, view)

    assert view.cancel_drag_calls == 1
    assert view._knife_armed is False  # disarmed
    # The staged clearing below the knife check did not run: the element
    # selection an armed knife has nothing to do with is still there.
    assert doc.element_mode == "face"
    assert obj.uid in doc.element_sel
    assert list(doc.element_sel[obj.uid].faces) == [0]


def test_escape_runs_the_staged_clear_when_no_knife_is_armed() -> None:
    """The existing behaviour, unaffected by the new branch: with no view at
    all (``view=None``, ``handle_key``'s own default) or an unarmed one,
    Escape still steps element selection -> element mode -> object
    selection, exactly as it always did."""
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=np.array([0], dtype="i4")))
    state = clay_state.ClayState()
    tab = clay_state.ClayTab(doc=doc)

    clay_mode._escape(state, tab, doc, None)
    assert doc.element_mode == "face"
    assert doc.element_sel == {}  # selection cleared first

    clay_mode._escape(state, tab, doc, _FakeView(armed=False))
    assert doc.element_mode == "object"  # then the mode itself


def test_handle_key_passes_its_own_view_through_to_escape() -> None:
    """The wiring in ``handle_key`` itself: a real Escape keypress must reach
    ``_escape`` with ``ctx.clay_view``, not ``None`` -- otherwise an armed
    knife elsewhere in the app would never see this branch at all."""
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    ctx = FakeCtx()
    clay_mode.adopt(ctx, doc, title="Scene")
    view = _FakeView(armed=True)
    ctx.clay_view = view

    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0)
    assert clay_mode.handle_key(ctx, event) is True
    assert view.cancel_drag_calls == 1
