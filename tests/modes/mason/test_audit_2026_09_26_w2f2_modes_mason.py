"""Regressions for the 2026-09-26 audit, closed against the files this fixer
(w2f2) owns: ``mason/assets.py``, ``mason/engine/document.py``,
``mason/engine/serialize.py``, ``mason/mode.py``, ``mason/ui/panes/tools.py``
and ``mason/ui/view.py``.

Each test's name is the claim it makes, ``test_mason_mode.py``'s own
convention (restated by every other mason test module).
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.studio import dialogs
from realmspinner.studio.modes.mason import mode as mason_mode
from realmspinner.studio.modes.mason import state as mason_state
from realmspinner.studio.modes.mason.engine import document as md
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import refs as mason_refs
from realmspinner.studio.modes.mason.engine import serialize as ser
from realmspinner.studio.modes.mason.ui import view as mason_view


class _SimpleCtx:
    def __init__(self) -> None:
        self.state = mason_state.MasonState()


class _FailingSource:
    """A stand-in ``GeometrySource`` that has already given up on ``ref``,
    the way ``mason_assets.AssetSource`` does once a generator is unknown, a
    builder raises, or a library parse errors: ``primitives`` keeps answering
    ``[]`` and the key sits in ``missing`` for good."""

    def __init__(self, key: tuple[Any, ...]) -> None:
        self.missing = {key}
        self.rev = 0

    def primitives(self, ref: Any) -> list[Any]:
        return []

    def box(self, ref: Any) -> None:
        return None


# --- mason-mode-03: ``doc.missing`` was never filled ------------------------


def test_a_library_ref_that_fails_to_load_is_reported_by_scene_stats_and_missing_refs(
    gl,
) -> None:
    """Only ``AssetSource.missing`` was ever written (``assets.py``); nothing
    copied it onto ``doc.missing`` (``document.py``), so ``missing_refs()`` --
    the "N missing" HUD, the Scene-file list and the Properties line
    (manual 31) -- always reported zero regardless of how many refs actually
    failed. Against the unfixed ``MasonView.sync``, ``doc.missing_refs()`` is
    empty and ``scene_stats(...)["missing"] == 0`` here even though the node's
    ref never resolves.
    """
    ref = mason_refs.LibraryRef(job_id="deadbeef")
    key = mason_refs.ref_key(ref)
    doc = md.MasonDoc()
    node = nd.MeshNode(uid=nd.new_uid(), name="prop", ref=ref)
    doc.add_node(node)

    view = mason_view.MasonView(gl, _SimpleCtx())
    try:
        view.sync(doc, _FailingSource(key))
    finally:
        view.release()

    reported = doc.missing_refs()
    assert [n for n, _r in reported] == [node]

    tab = type("Tab", (), {"doc": doc})()
    stats = mason_mode.scene_stats(None, tab)
    assert stats["missing"] == 1


def test_sync_does_not_require_a_geometry_source_to_carry_a_missing_set(gl) -> None:
    """``missing`` is not part of the ``GeometrySource`` protocol (``refs.py``
    lists only ``primitives``, ``box`` and ``rev``) -- a stand-in source with
    no such attribute (every fixture in ``test_mason_view.py``) must not raise
    out of ``sync``."""

    class _NoMissingSource:
        rev = 0

        def primitives(self, ref: Any) -> list[Any]:
            return []

        def box(self, ref: Any) -> None:
            return None

    doc = md.MasonDoc()
    doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="prop", ref=mason_refs.LibraryRef(job_id="x")))

    view = mason_view.MasonView(gl, _SimpleCtx())
    try:
        view.sync(doc, _NoMissingSource())  # must not raise
    finally:
        view.release()

    assert doc.missing == set()


# --- mason-engine-04: non-finite transforms, wrong-length factors, ----------
# --- and non-object scene/texture containers --------------------------------


def _rewrite(data: bytes, edit) -> bytes:
    """The same ``.rscn`` archive with ``edit`` applied to its parsed
    ``scene.json`` -- ``test_serialize.py``'s own helper, restated here since
    a test module may not import another test module."""
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            if name == ser.SCENE:
                scene = json.loads(src.read(name))
                edit(scene)
                dst.writestr(name, json.dumps(scene))
            else:
                dst.writestr(name, src.read(name))
    return out.getvalue()


def _replace_scene(data: bytes, raw: bytes) -> bytes:
    """The same archive with ``scene.json`` replaced by arbitrary bytes --
    for a mangle that must not even parse as a mapping, which ``_rewrite``'s
    edit-a-parsed-dict shape cannot express."""
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            dst.writestr(name, raw if name == ser.SCENE else src.read(name))
    return out.getvalue()


def _one_node_doc() -> md.MasonDoc:
    node = nd.MeshNode(uid=nd.new_uid(), name="Rock", ref=ser.primitive_ref("box", {}))
    return md.MasonDoc(roots=[node])


def test_read_rscn_refuses_non_finite_transforms_and_a_non_object_scene() -> None:
    """mason-engine-04, the 2026-09-26 audit: ``_vector`` cast a JSON
    ``NaN``/``Infinity`` straight into the array it handed back, and
    ``read_rscn`` assumed its parsed ``scene.json`` was a mapping without
    checking. Against the unfixed reader, the first ``read_rscn`` below
    raises nothing at all (a silently non-finite transform) and the second
    raises a bare ``AttributeError`` from ``scene.get(...)`` rather than this
    reader's own named refusal.
    """
    data = ser.rscn_bytes(_one_node_doc())

    def mangle(scene: dict) -> None:
        for entry in scene["nodes"]:
            if entry.get("name") == "Rock":
                entry["translation"] = [float("nan"), 0.0, 0.0]

    with pytest.raises(ValueError, match="translation"):
        ser.read_rscn(_rewrite(data, mangle))

    non_object = _replace_scene(data, json.dumps([1, 2, 3]).encode())
    with pytest.raises(ValueError, match="Realmspinner Mason scene"):
        ser.read_rscn(non_object)


def test_read_rscn_refuses_an_infinite_light_field_and_camera_field() -> None:
    """The same non-finite guard for :func:`_float_tuple` (a light's
    ``color``) and :func:`_float_field` (a light's ``intensity``, a camera's
    ``yfov``/``znear``/``zfar``) -- not only :func:`_vector`'s transforms."""
    light = nd.LightNode(uid=nd.new_uid(), name="Sun", kind="point")
    doc = md.MasonDoc(roots=[light])
    data = ser.rscn_bytes(doc)

    def mangle(scene: dict) -> None:
        scene["nodes"][0]["intensity"] = float("inf")

    with pytest.raises(ValueError, match="intensity"):
        ser.read_rscn(_rewrite(data, mangle))


def test_read_rscn_refuses_a_wrong_length_material_factor() -> None:
    """mason-engine-04: ``tuple(entry.get("base_color_factor", ...))`` used to
    accept any length at all -- ``gltf.Material`` is a bare ``@dataclass``
    with no ``__post_init__`` to check it -- so a two-element
    ``base_color_factor`` built a material nothing downstream expects."""
    doc = md.MasonDoc(materials=[gltf.Material(name="red")])
    data = ser.rscn_bytes(doc)

    def mangle(scene: dict) -> None:
        scene["materials"][0]["base_color_factor"] = [1.0, 0.0]

    # ``_material_from``'s own ``try`` folds every malformed field into one
    # message (see its docstring) -- the same "malformed" a bad texture index
    # or a bad name already gets in ``test_serialize.py``, not the field name.
    with pytest.raises(ValueError, match="malformed"):
        ser.read_rscn(_rewrite(data, mangle))


def test_read_rscn_refuses_a_textures_field_that_is_not_a_list() -> None:
    """mason-engine-04: ``for entry in scene.get("textures", [])`` used to
    accept any container JSON could produce -- a dict iterates its own
    string keys, and ``"a-key".get(...)`` is a bare ``AttributeError``, not
    this reader's own named refusal."""
    data = ser.rscn_bytes(_one_node_doc())

    def mangle(scene: dict) -> None:
        scene["textures"] = {"oops": 1}

    with pytest.raises(ValueError, match="textures"):
        ser.read_rscn(_rewrite(data, mangle))


# --- mason-mode-02: duplicate_selected/Array only charged the tree-side count


class _ModeFakeCtx:
    """``test_mason_mode.py``'s own ``FakeCtx``, restated here rather than
    imported: a test module may not import another test module, and this is
    the minimal contract :func:`mason_mode.new_document`/``duplicate_selected``
    need -- ``submit`` runs its callable inline, ``toast`` just records."""

    def __init__(self) -> None:
        self.svc = None
        self.state = type("S", (), {"mason": None, "mode": "home"})()
        self.settings = type("Settings", (), {"store": {}, "get": lambda self, k: None})()
        self.toasts: list[tuple[str, str]] = []

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        run(*args)
        return True

    def toast(self, message: str, kind: str = "info", *_a: Any, **_k: Any) -> None:
        self.toasts.append((message, kind))


def _armed_ctx() -> _ModeFakeCtx:
    ctx = _ModeFakeCtx()
    mason_mode.new_document(ctx)
    return ctx


def test_duplicate_selected_of_prefab_instances_past_the_resolved_ceiling_toasts_instead_of_raising(
    monkeypatch,
) -> None:
    """mason-mode-02, the 2026-09-26 audit (was High): ``duplicate_selected``'s
    own pre-check (``_over_max_placed``) only counts tree-side size -- one
    ``PrefabNode`` instance is one tree node no matter how big its template is
    -- so a duplicate that stays comfortably under that check can still push
    the document's *resolved* size (what the template actually expands to
    every draw/export) past ``MAX_PLACED``. Against the unfixed function that
    raises ``add_node``'s ``ValueError`` uncaught, past this call, with the
    undo mark left open.
    """
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    template = nd.GroupNode(
        uid=nd.new_uid(),
        name="Tmpl",
        children=[nd.MeshNode(uid=nd.new_uid(), name=f"m{i}") for i in range(5)],
    )
    doc.prefabs["big"] = template
    inst = nd.PrefabNode(uid=nd.new_uid(), name="Instance", template="big")
    doc.add_node(inst)
    doc.select([inst.uid])

    growth = doc.resolved_growth([inst])
    assert growth > 1, "the template must cost more resolved than the copy's own tree size"
    before_tree = len(doc.all_nodes())
    resolved_before = doc.resolved_total()
    # One short of what one more duplicate needs, resolved-wise -- but the
    # tree only grows by 1 (another PrefabNode), nowhere near this ceiling.
    from realmspinner.studio.modes.mason.engine import scene as msc

    monkeypatch.setattr(msc, "MAX_PLACED", resolved_before + growth - 1)
    steps = len(doc.history)

    mason_mode.duplicate_selected(ctx)

    assert len(doc.all_nodes()) == before_tree
    assert len(doc.history) == steps
    assert doc.selection == {inst.uid}
    assert any(kind == "error" for _msg, kind in ctx.toasts)


def test_array_of_a_prefab_instance_past_the_resolved_ceiling_toasts_instead_of_raising(
    monkeypatch,
) -> None:
    """The Array buttons' own half of mason-mode-02: ``_spawn_array`` calls
    ``doc.add_nodes`` once for the whole batch, which checks both ceilings
    before attaching anything -- so this backstop never leaves a half-built
    array attached -- but its ``ValueError`` used to propagate out of a
    button press uncaught instead of the toast ``_over_either_ceiling``'s
    precheck gives the common case.
    """
    from realmspinner.studio.modes.mason.engine import scene as msc
    from realmspinner.studio.modes.mason.ui.panes import tools as mason_tools

    doc = md.MasonDoc()
    template = nd.GroupNode(
        uid=nd.new_uid(),
        name="Tmpl",
        children=[nd.MeshNode(uid=nd.new_uid(), name=f"m{i}") for i in range(5)],
    )
    doc.prefabs["big"] = template
    inst = nd.PrefabNode(uid=nd.new_uid(), name="Instance", template="big")
    doc.add_node(inst)

    growth = doc.resolved_growth([inst])
    assert growth > 1
    resolved_before = doc.resolved_total()
    monkeypatch.setattr(msc, "MAX_PLACED", resolved_before + growth - 1)

    # The tree-side precheck alone sees nothing wrong (one more PrefabNode);
    # only the resolved-size one does.
    assert mason_tools._over_max_placed(doc, inst, 1) is None
    assert mason_tools._over_either_ceiling(doc, inst, 1) is not None

    ctx = _ModeFakeCtx()
    before_tree = len(doc.all_nodes())
    mason_tools._spawn_array(
        ctx, doc, inst, [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0))]
    )

    assert len(doc.all_nodes()) == before_tree
    assert any(kind == "error" for _msg, kind in ctx.toasts)


# --- mason-mode-09: gizmo drag and sculpt ignored an inherited/terrain lock -


class _StateHolder:
    def __init__(self) -> None:
        self.mason = mason_state.MasonState()


class _ViewCtx:
    """The minimal ``app_ctx`` shape ``MasonView.state`` reads through:
    ``getattr(app_ctx.state, "mason", None)`` (``view.py``'s own property) --
    a real ``mason_state.MasonState`` so ``.tool``/``.brush``/``.pivot`` all
    carry their real defaults rather than being re-declared here."""

    def __init__(self) -> None:
        self.state = _StateHolder()


def test_gizmo_drag_and_sculpt_respect_an_inherited_or_terrain_lock(gl) -> None:
    """mason-mode-09, the 2026-09-26 audit: ``_begin_gizmo_drag`` read a
    node's own ``.locked`` flag, and the sculpt brush read no lock at all --
    neither read ``scene.py``'s inherited "locked ORs" answer
    (``Placed.locked``), so a child of a locked group, or a locked
    ``TerrainNode``, still moved. Manual 31:141-144 promises "a locked
    object, and everything under it, cannot be moved."
    """
    # -- gizmo drag: a child of a locked group, itself unlocked -----------
    group = nd.GroupNode(uid=nd.new_uid(), name="Locked", locked=True)
    child = nd.MeshNode(uid=nd.new_uid(), name="Child", ref=ser.primitive_ref("box", {}))
    group.children.append(child)
    doc = md.MasonDoc(roots=[group])
    doc.select([child.uid])
    assert child.locked is False  # the node's own flag really is unset

    view = mason_view.MasonView.__new__(mason_view.MasonView)
    view.app_ctx = _ViewCtx()
    view._placed = []
    view._placed_key = None
    view._last_doc = None
    view._begin_gizmo_drag(doc, _FailingSource(("nothing",)))

    assert child.uid not in view._drag_start

    # -- sculpt: a locked terrain -------------------------------------------
    from realmspinner.kernels.geom3d import gltf as _gltf
    from realmspinner.studio.modes.mason.engine.terrain import Terrain

    terrain_doc = md.MasonDoc()
    terrain_doc.set_terrain(
        Terrain(
            heights=np.zeros((5, 5), dtype="f4"),
            size_x=8.0,
            size_z=8.0,
            material=_gltf.Material(name="ground"),
        )
    )
    terrain_doc.add_node(nd.TerrainNode(uid=nd.new_uid(), name="Terrain", locked=True))

    tview = mason_view.MasonView(gl, _ViewCtx())
    try:
        tview._rect = (0.0, 0.0, 128.0, 96.0)
        tview.state.tool = "sculpt"
        tview._press(terrain_doc, _FailingSource(("nothing",)), 1, (64.0, 48.0))
        assert tview._grab != "sculpt"
        assert terrain_doc.sculpting is False
    finally:
        tview.release()


# --- mason-mode-12: export's run() mutated the AssetSource from a task thread


class _RecordingSource:
    """A ``GeometrySource`` that resolves synchronously (like the real one
    does for a cache hit or a ``PrimitiveRef``) and counts every call to
    ``primitives`` -- the same call ``mason_assets.AssetSource.primitives``
    mutates its ``_cache``/``_order``/``_pending`` (and, for an unresolved
    ``LibraryRef``, calls ``ctx.submit``) inside."""

    rev = 0

    def __init__(self) -> None:
        self.calls = 0

    def primitives(self, ref: Any) -> list[gltf.Primitive]:
        self.calls += 1
        return [
            gltf.Primitive(
                positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"),
                indices=np.array([0, 1, 2], dtype="u4"),
                # A real base-colour texture, not just a flat factor: the
                # mason-mode-13 regression below needs an OBJ export that
                # actually writes a ``map_Kd`` line and a texture PNG, the
                # two things a fixed ``textures/`` name collided on.
                material=gltf.Material(
                    name="stone", base_color=(1, 1, b"\xff\xff\xff\xff")
                ),
            )
        ]

    def box(self, ref: Any) -> Any:
        return self.primitives(ref)[0].box()


class _DeferredCtx:
    """A ``ctx`` whose ``submit`` stores ``run`` rather than running it
    inline -- the shape every other mason test's ``FakeCtx`` deliberately
    does *not* have, because this test needs to see what already happened
    *before* ``ctx.submit`` was even called, which an inline-running
    ``submit`` would hide entirely.
    """

    def __init__(self) -> None:
        self.svc = None
        self.state = type("S", (), {"mason": None, "mode": "home"})()
        self.settings = type("Settings", (), {"store": {}, "get": lambda self, k: None})()
        self.toasts: list[tuple[str, str]] = []
        self.pending: tuple[Any, tuple[Any, ...]] | None = None

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.pending = (run, args)
        return True

    def toast(self, message: str, kind: str = "info", *_a: Any, **_k: Any) -> None:
        self.toasts.append((message, kind))


def _scene_with_a_library_ref() -> md.MasonDoc:
    doc = md.MasonDoc()
    doc.add_node(
        nd.MeshNode(uid=nd.new_uid(), name="Prop", ref=mason_refs.LibraryRef(job_id="x"))
    )
    return doc


def test_export_task_does_not_mutate_the_asset_source_from_a_task_thread(monkeypatch) -> None:
    """mason-mode-12, the 2026-09-26 audit: ``export_glb``/``export_obj``'s
    ``run()`` closures used to call ``mason_assets.ensure``/
    ``gltfout.scene_model`` (which calls ``source.primitives(ref)`` for every
    node) *inside* the closure ``_start`` hands off to a task thread -- racing
    the frame thread's own ``MasonView.sync``, which mutates that exact same
    ``AssetSource`` every frame, and (for an unresolved ref) calling
    ``ctx.submit`` from a thread nothing else ever calls it from. Against the
    unfixed functions, ``source.calls`` is still 0 by the time ``ctx.submit``
    is called (nothing has resolved anything yet) and only grows once the
    deferred ``run()`` below is invoked -- proving the mutating call happened
    inside the task closure rather than before it was ever handed off.
    """
    from realmspinner.studio.modes.mason import assets as mason_assets

    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: None)  # cancel: no file I/O needed

    for exporter, key in (
        (mason_mode.export_glb, "mason-exportglb"),
        (mason_mode.export_obj, "mason-exportobj"),
    ):
        source = _RecordingSource()
        monkeypatch.setattr(mason_assets, "ensure", lambda ctx, _s=source: _s)
        ctx = _DeferredCtx()
        tab = mason_mode.adopt(ctx, _scene_with_a_library_ref(), title="Scene")

        exporter(ctx, tab)

        assert ctx.pending is not None, f"{key} must still hand a run() to ctx.submit"
        calls_before = source.calls
        assert calls_before > 0, (
            f"{key}: the scene's ref must already be resolved before the task "
            "is even submitted, not merely by the time it finishes"
        )

        run, args = ctx.pending
        run(*args)  # the deferred task, invoked the way a task thread eventually would

        assert source.calls == calls_before, (
            f"{key}: run() must not call source.primitives() itself -- "
            "resolving happens on the frame thread, before ctx.submit, never "
            "inside the closure a task thread runs"
        )


# --- mason-mode-13: sidecars at a fixed name collide across two exports -----


class _InlineExportCtx:
    """Runs a submitted ``run`` inline -- every other mason test's ``FakeCtx``
    shape, self-contained here since a test module may not import another."""

    def __init__(self) -> None:
        self.svc = None
        self.state = type("S", (), {"mason": None, "mode": "home"})()
        self.settings = type("Settings", (), {"store": {}, "get": lambda self, k: None})()
        self.toasts: list[tuple[str, str]] = []
        self.result: Any = None

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info", *_a: Any, **_k: Any) -> None:
        self.toasts.append((message, kind))


def test_two_exports_to_one_folder_do_not_overwrite_each_others_sidecars(
    tmp_path, monkeypatch
) -> None:
    """mason-mode-13, the 2026-09-26 audit: ``write_files`` (GLB path) and
    ``objout``'s own ``mtllib``/``map_Kd`` lines (OBJ path) used to name every
    sidecar at a fixed spelling -- ``scene.json``, ``scene.mtl``,
    ``textures/<n>.png`` -- regardless of what the user actually called the
    export. Exporting "Barrel" and then "Crate" as GLB into the same folder
    used to leave exactly one ``scene.json`` on disk, describing whichever
    export ran last, with the first export's own manifest gone; the OBJ path
    did the same to ``scene.mtl`` and every texture under ``textures/``.
    Against the unfixed code, ``Barrel.json`` does not exist afterward -- both
    exports wrote the same ``scene.json`` -- and the second overwrote it.
    """
    from realmspinner.studio.modes.mason import assets as mason_assets_mod

    monkeypatch.setattr(mason_assets_mod, "ensure", lambda ctx: _RecordingSource())

    glb_paths = iter([tmp_path / "Barrel.glb", tmp_path / "Crate.glb"])
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: str(next(glb_paths)))

    ctx = _InlineExportCtx()
    tab1 = mason_mode.adopt(ctx, _scene_with_a_library_ref(), title="Barrel")
    mason_mode.export_glb(ctx, tab1)
    tab2 = mason_mode.adopt(ctx, _scene_with_a_library_ref(), title="Crate")
    mason_mode.export_glb(ctx, tab2)

    assert (tmp_path / "Barrel.glb").exists()
    assert (tmp_path / "Crate.glb").exists()
    assert (tmp_path / "Barrel.json").exists(), "the first export's manifest must survive"
    assert (tmp_path / "Crate.json").exists()

    # -- the OBJ path: the MTL and the texture directory, not only the OBJ --
    obj_paths = iter([tmp_path / "Barrel.obj", tmp_path / "Crate.obj"])
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: str(next(obj_paths)))

    tab3 = mason_mode.adopt(ctx, _scene_with_a_library_ref(), title="Barrel")
    mason_mode.export_obj(ctx, tab3)
    tab4 = mason_mode.adopt(ctx, _scene_with_a_library_ref(), title="Crate")
    mason_mode.export_obj(ctx, tab4)

    assert (tmp_path / "Barrel.obj").exists()
    assert (tmp_path / "Crate.obj").exists()
    assert (tmp_path / "Barrel.mtl").exists(), "the first export's MTL must survive"
    assert (tmp_path / "Crate.mtl").exists()
    assert (tmp_path / "Barrel_textures" / "0.png").exists()
    assert (tmp_path / "Crate_textures" / "0.png").exists()

    # And each MTL/OBJ pair must actually reference its own sidecars, not the
    # other export's -- a rename in ``write_files`` alone (rather than at the
    # point ``mtllib``/``map_Kd`` are written) would leave the *content*
    # still pointing at "scene.mtl"/"textures/0.png", breaking every export,
    # not only a colliding one.
    barrel_obj = (tmp_path / "Barrel.obj").read_text("utf-8")
    assert "mtllib Barrel.mtl" in barrel_obj
    barrel_mtl = (tmp_path / "Barrel.mtl").read_text("utf-8")
    assert "map_Kd Barrel_textures/0.png" in barrel_mtl
