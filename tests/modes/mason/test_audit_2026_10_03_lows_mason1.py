"""The 2026-10-03 audit's Low findings for Mason (fixer mason1).

Every test name is the claim, written to fail against the code the audit read.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.studio.modes.mason.engine import document as doc
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import objout, refs

# --- mason-24 ----------------------------------------------------------------


def _tiny_scale_export(scale: float) -> objout.ObjExport:
    d = doc.MasonDoc()
    mesh = nd.MeshNode(uid=nd.new_uid(), name="Prop", ref=refs.primitive_ref("box", {}))
    mesh.scale = m3.vec3(scale, scale, scale)
    d.add_node(mesh)

    class _Src:
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

    return objout.obj_export(d, _Src())


def test_obj_keeps_normals_for_a_uniformly_tiny_but_invertible_scale():
    """A millimetre prop at 0.0009 has |det| < 1e-9 yet a perfectly good inverse."""
    out = _tiny_scale_export(0.0009)
    text = out.files[objout.OBJ].decode("utf-8")
    assert "vn " in text
    assert not any("zero scale" in s for s in out.skipped)


# --- mason-25 ----------------------------------------------------------------


def _bare_view_with_live_gizmo_drags():
    from types import SimpleNamespace

    from realmspinner.studio.modes.mason.ui import view as mason_view
    from realmspinner.studio.viewer.gizmo import Drag, Gizmo

    def gizmo() -> Gizmo:
        g = Gizmo.__new__(Gizmo)
        g.hover = None
        g.drag = Drag(
            axis="x", start=np.zeros(3), origin=np.zeros(3), normal=np.array([1.0, 0.0, 0.0])
        )
        return g

    view = mason_view.MasonView.__new__(mason_view.MasonView)
    view.translate_gizmo = gizmo()
    view.rotate_gizmo = gizmo()
    view.scale_gizmo = gizmo()
    view._drag_start = {}
    view._sculpt_doc = None
    view._grab = "gizmo"
    view._render_dirty = False
    return view, SimpleNamespace


def test_releasing_a_gizmo_drag_clears_the_gizmos_own_drag_state():
    """Clay calls ``end_drag`` at release; Mason never did, so ``draws()`` kept
    painting the last-dragged axis as active instead of the hovered one."""
    d = doc.MasonDoc()
    view, _ = _bare_view_with_live_gizmo_drags()
    view._end_gizmo_drag(d)
    for g in (view.translate_gizmo, view.rotate_gizmo, view.scale_gizmo):
        assert g.drag is None


def test_cancelling_a_gizmo_drag_clears_the_gizmos_own_drag_state():
    d = doc.MasonDoc()
    view, _ = _bare_view_with_live_gizmo_drags()
    view.cancel_drag(d)
    for g in (view.translate_gizmo, view.rotate_gizmo, view.scale_gizmo):
        assert g.drag is None


# --- mason-26 ----------------------------------------------------------------


def test_properties_walks_the_document_once_per_revision(monkeypatch):
    """Properties re-ran the root-down ``resolved_for`` walk and
    ``missing_refs`` every frame with nothing changed; both are now answered
    once per ``(doc, doc.rev)`` (and, for the missing list, ``doc.missing``)."""
    from realmspinner.studio.modes.mason import mode as mason_mode
    from realmspinner.studio.modes.mason.engine import scene as mscene
    from realmspinner.studio.modes.mason.ui.panes import props

    d = doc.MasonDoc()
    node = nd.MeshNode(uid=nd.new_uid(), name="", ref=refs.primitive_ref("box", {}))
    d.add_node(node)

    walks = {"resolved": 0, "missing": 0}
    real_resolved = mscene.resolved_for
    real_missing = doc.MasonDoc.missing_refs

    def counting_resolved(*a, **k):
        walks["resolved"] += 1
        return real_resolved(*a, **k)

    def counting_missing(self):
        walks["missing"] += 1
        return real_missing(self)

    monkeypatch.setattr(mscene, "resolved_for", counting_resolved)
    monkeypatch.setattr(doc.MasonDoc, "missing_refs", counting_missing)

    for _ in range(5):
        props._placed_for(d, node)
        mason_mode.missing_refs_cached(d)
    assert walks == {"resolved": 1, "missing": 1}

    d.touch()
    props._placed_for(d, node)
    mason_mode.missing_refs_cached(d)
    assert walks == {"resolved": 2, "missing": 2}

    # ``doc.missing`` is rewritten by the viewport without moving ``doc.rev``.
    d.missing = {refs.ref_key(node.ref)}
    assert [n.uid for n, _ in mason_mode.missing_refs_cached(d)] == [node.uid]
    assert walks["missing"] == 3


# --- mason-27 ----------------------------------------------------------------


def _rscn_with_a_material() -> bytes:
    from realmspinner.studio.modes.mason.engine import serialize as ser

    d = doc.MasonDoc()
    d.add_node(nd.MeshNode(uid=nd.new_uid(), name="M", ref=refs.primitive_ref("box", {})))
    d.materials.append(gltf.Material(name="stone"))
    return ser.rscn_bytes(d)


def _rewritten(data: bytes, edit) -> bytes:
    import json
    import zipfile
    from io import BytesIO

    from realmspinner.studio.modes.mason.engine import serialize as ser

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


def test_read_rscn_refuses_a_non_numeric_version_and_non_finite_material_scalars_by_name():
    import pytest

    from realmspinner.studio.modes.mason.engine import serialize as ser

    base = _rscn_with_a_material()
    assert ser.read_rscn(base).materials  # the fixture itself reads cleanly

    for bad in (None, [1], "x", float("nan")):
        data = _rewritten(base, lambda s, bad=bad: s.__setitem__("version", bad))
        with pytest.raises(ValueError, match="version"):
            ser.read_rscn(data)

    for key in ("metallic_factor", "roughness_factor", "alpha_cutoff"):
        for bad in (float("nan"), float("inf")):
            data = _rewritten(
                base, lambda s, key=key, bad=bad: s["materials"][0].__setitem__(key, bad)
            )
            with pytest.raises(ValueError, match=f"{key}.*not finite|malformed"):
                ser.read_rscn(data)


# --- mason-28 ----------------------------------------------------------------


class _Done:
    def __init__(self, key, tag, result=None, error=None):
        self.key, self.tag, self.result, self.error = key, tag, result, error


def _asset_ctx(root):
    class _Config:
        def job_dir(self, job_id):
            return root / job_id

    class _Svc:
        config = _Config()

    class _Ctx:
        svc = _Svc()

        def __init__(self):
            self.submitted = []

        def submit(self, key, fn, *args, tag=None, **kwargs):
            self.submitted.append((key, tag))
            return True

    return _Ctx()


def _baked_triangle():
    from realmspinner.studio.modes.mason import assets as mason_assets

    prim = gltf.Primitive(
        positions=np.zeros((3, 3), dtype="f4"), indices=np.array([0, 1, 2], dtype="u4")
    )
    return mason_assets._bake_model(gltf.Model([gltf.Node(mesh=0)], [0], [[prim]], []))


def _rewrite_in_place(path, payload: bytes) -> None:
    import os

    before = path.stat().st_mtime_ns
    path.write_bytes(payload)
    os.utime(path, ns=(before + 5_000_000_000, before + 5_000_000_000))


def test_an_asset_rebuilt_in_place_is_re_resolved_instead_of_served_from_the_cache(tmp_path):
    """``optimize_job`` rewrites ``model.glb`` in place; the cache keyed on
    ``(job_id, artifact)`` kept serving the old geometry until restart."""
    from realmspinner.studio.modes.mason import assets as mason_assets

    job = tmp_path / "0000000000a1"
    job.mkdir()
    glb = job / "model.glb"
    glb.write_bytes(b"v1")
    ctx = _asset_ctx(tmp_path)
    source = mason_assets.ensure(ctx)
    ref = refs.LibraryRef(job_id="0000000000a1", artifact="model.glb")

    assert source.primitives(ref) == []
    task_key, tag = ctx.submitted[-1]
    assert mason_assets.on_task_done(ctx, _Done(task_key, tag, result=_baked_triangle()))
    assert source.primitives(ref) != []
    rev = source.rev

    source.revalidate(force=True)  # nothing changed: nothing happens
    assert source.primitives(ref) != [] and source.rev == rev

    _rewrite_in_place(glb, b"version two, different size")
    source.revalidate(force=True)
    assert source.rev == rev + 1
    assert source.box(ref) is None
    assert refs.ref_key(ref) in source.take_invalidated()
    assert source.primitives(ref) == []  # re-resolving: a second parse was submitted
    assert len(ctx.submitted) == 2


def test_a_ref_that_failed_once_is_retried_when_its_artifact_appears(tmp_path):
    """"Put the asset back" must take effect without a restart: a ref in
    ``missing`` stayed there for the whole session."""
    from realmspinner.studio.modes.mason import assets as mason_assets

    (tmp_path / "0000000000a2").mkdir()
    ctx = _asset_ctx(tmp_path)
    source = mason_assets.ensure(ctx)
    ref = refs.LibraryRef(job_id="0000000000a2", artifact="model.glb")

    assert source.primitives(ref) == []
    task_key, tag = ctx.submitted[-1]
    mason_assets.on_task_done(ctx, _Done(task_key, tag, error=FileNotFoundError("gone")))
    assert refs.ref_key(ref) in source.missing

    (tmp_path / "0000000000a2" / "model.glb").write_bytes(b"restored")
    source.revalidate(force=True)
    assert refs.ref_key(ref) not in source.missing
    source.primitives(ref)
    assert len(ctx.submitted) == 2


def test_the_viewport_drops_the_gpu_upload_of_a_ref_the_source_invalidated():
    """The GPU cache is keyed by ref too: dropping only the source's geometry
    would leave the old upload drawn."""
    from realmspinner.studio.modes.mason.engine import scene as msc
    from realmspinner.studio.modes.mason.ui import view as mason_view

    ref = refs.LibraryRef(job_id="0000000000a1", artifact="model.glb")
    d = doc.MasonDoc()
    d.add_node(nd.MeshNode(uid=nd.new_uid(), name="M", ref=ref))
    key = mason_view._entry_key(msc.resolve(d)[0])
    released = []

    class _Gpu:
        def release(self):
            released.append(True)

    class _Entry:
        gpu = _Gpu()

    entry = _Entry()
    entry.key = key

    class _Source:
        rev = 0

        def __init__(self, invalidated):
            self._invalidated = invalidated

        def take_invalidated(self):
            taken, self._invalidated = self._invalidated, set()
            return taken

        def primitives(self, ref):
            return []  # re-resolving: nothing to draw yet

    view = mason_view.MasonView.__new__(mason_view.MasonView)
    view._cache = {key: entry}
    view._placed = []
    view._placed_key = None
    view._last_doc = None
    view._terrain = None

    view.sync(d, _Source(set()))  # not invalidated: the upload is kept
    assert key in view._cache and not released

    view.sync(d, _Source({key[0]}))
    assert key not in view._cache and released


# --- mason-29 ----------------------------------------------------------------


def test_the_export_thumbnail_is_not_taken_from_a_different_scene_tab(monkeypatch):
    """Tab switching is not blocked while an export runs, and the capture reads
    whichever scene the viewport shows when the task lands: the new library
    card must not be a picture of the other scene."""
    from types import SimpleNamespace

    from realmspinner.studio import main as main_mod
    from realmspinner.studio.modes.mason import mode as mason_mode

    monkeypatch.setattr(mason_mode, "on_task_done", lambda ctx, done: None)

    class _App:
        def __init__(self, active_uid):
            tab = SimpleNamespace(uid=active_uid)
            self.app_ctx = SimpleNamespace(state=SimpleNamespace(mason=SimpleNamespace(active=tab)))
            self.mason_view = "mason-viewport"
            self.captured = []

        def _capture_thumbnail_from(self, job_id, view):
            self.captured.append((job_id, view))

    class _Done:
        def __init__(self, key):
            self.key = key
            self.result = {"job_id": "j1", "exported_asset": True}

    shown_elsewhere = _App("ms-b")
    main_mod.App._on_task_done(shown_elsewhere, _Done("mason-library:ms-a"))
    assert shown_elsewhere.captured == []

    shown_here = _App("ms-a")
    main_mod.App._on_task_done(shown_here, _Done("mason-library:ms-a"))
    assert shown_here.captured == [("j1", "mason-viewport")]


# --- mason-30 ----------------------------------------------------------------


def test_a_recovered_scene_comes_back_with_the_camera_the_journal_saved(tmp_path):
    from typing import Any

    from realmspinner.studio.modes.mason import mode as mason_mode

    class _Store:
        def __init__(self) -> None:
            self.store: dict[str, Any] = {}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value):
            self.store[key] = value

        def invalidate(self):
            pass

    class _Ctx:
        def __init__(self) -> None:
            self.state = type("S", (), {"mason": None, "mode": "home"})()
            self.settings = _Store()
            self.cache = _Store()
            self.toasts: list[Any] = []

        def submit(self, key, fn, *args):
            return True

        def toast(self, *a, **k):
            self.toasts.append(a)

    ctx = _Ctx()
    original = mason_mode.adopt(ctx, doc.MasonDoc(), title="Courtyard")
    original.view.yaw = 1.25
    original.view.pitch = 0.5
    original.view.distance = 17.0
    original.view.target = np.array([3.0, 1.0, -2.0])
    saved = tmp_path / "copy.rscn"
    saved.write_bytes(mason_mode._journal_encode(original))

    result = mason_mode._load_recovery(saved, {"title": "Courtyard"})
    assert result is not None

    class _Done:
        key = "mason-recover:abc"

        def __init__(self, result):
            self.result = result

    mason_mode.on_task_done(ctx, _Done(result))
    recovered = mason_mode.active(ctx)
    assert recovered is not original
    assert recovered.view.yaw == 1.25
    assert recovered.view.distance == 17.0
    assert list(recovered.view.target) == [3.0, 1.0, -2.0]


# --- mason-31 ----------------------------------------------------------------


def test_a_stale_armed_prefab_does_not_swallow_viewport_clicks():
    """Armed in one scene, then a tab switch or an undo of the definition: the
    view turned every click into a placement request that placed nothing, so
    clicks neither placed nor selected until Esc."""
    from types import SimpleNamespace

    from realmspinner.studio.modes.mason.ui import view as mason_view

    state = SimpleNamespace(place_kind="", place_prefab="ghost")
    view = mason_view.MasonView.__new__(mason_view.MasonView)
    view.app_ctx = SimpleNamespace(state=SimpleNamespace(mason=state))

    d = doc.MasonDoc()  # no "ghost" template in this document
    assert view._placement_armed(d) is False
    assert state.place_prefab == ""

    d.prefabs["ghost"] = nd.GroupNode(uid=nd.new_uid(), name="Ghost")
    state.place_prefab = "ghost"
    assert view._placement_armed(d) is True
    assert state.place_prefab == "ghost"


def test_placing_a_prefab_the_document_lacks_disarms_and_says_so():
    from types import SimpleNamespace

    from realmspinner.studio.modes.mason import mode as mason_mode

    toasts = []
    ctx = SimpleNamespace(
        toast=lambda message, kind="info", *a, **k: toasts.append((message, kind)),
        state=SimpleNamespace(mason=None),
        settings=SimpleNamespace(get=lambda k: None, set=lambda k, v: None),
        cache=SimpleNamespace(invalidate=lambda: None),
    )
    mason_mode.adopt(ctx, doc.MasonDoc(), title="A")
    state = mason_mode.ensure(ctx)
    state.place_prefab = "ghost"

    assert mason_mode.place_armed(ctx, (0.0, 0.0, 0.0)) is None
    assert state.place_prefab == ""
    assert toasts and "ghost" in toasts[0][0]


def test_the_hint_names_the_armed_thing_not_its_internal_key():
    from types import SimpleNamespace

    from realmspinner.studio.modes.mason.ui.panes import hud

    def hint(kind: str, jobs=()) -> str:
        state = SimpleNamespace(place_prefab="", place_kind=kind, tool="select")
        return hud._hint(state, jobs)

    assert "light:" not in hint("light:point")
    assert "point light" in hint("light:point")
    assert "job:" not in hint("job:3f2a9c")
    assert "Crate" in hint("job:3f2a9c", [{"id": "3f2a9c", "name": "Crate"}])
    assert "3f2a9c" not in hint("job:3f2a9c")
    assert "torus" in hint("torus")


# --- mason-32 ----------------------------------------------------------------


def test_a_failed_open_forgets_the_recent_path(tmp_path):
    """The key was ``mason-open:{sha1 of path}``, so ``on_task_failed`` could not
    recover the path and a moved .rscn stayed in Resume for ever."""
    from pathlib import Path
    from typing import Any

    from realmspinner.studio.modes.mason import mode as mason_mode

    class _Settings:
        def __init__(self) -> None:
            self.store: dict[str, Any] = {}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value):
            self.store[key] = value

    class _Ctx:
        def __init__(self) -> None:
            self.state = type("S", (), {"mason": None, "mode": "home"})()
            self.settings = _Settings()
            self.submitted: list[str] = []

        def submit(self, key, fn, *args):
            self.submitted.append(key)
            return True

    class _Done:
        def __init__(self, key):
            self.key = key

    ctx = _Ctx()
    gone = Path(tmp_path) / "moved.rscn"
    mason_mode.remember_path(ctx, gone)
    assert str(gone) in mason_mode.recent_paths(ctx)

    mason_mode.open_path(ctx, gone)
    mason_mode.on_task_failed(ctx, _Done(ctx.submitted[-1]))
    assert str(gone) not in mason_mode.recent_paths(ctx)


# --- mason-33 ----------------------------------------------------------------


def _placing_ctx():
    from types import SimpleNamespace

    from realmspinner.studio.modes.mason import mode as mason_mode

    ctx = SimpleNamespace(
        toast=lambda *a, **k: None,
        state=SimpleNamespace(mason=None, mode="home"),
        settings=SimpleNamespace(get=lambda k: None, set=lambda k, v: None),
        cache=SimpleNamespace(invalidate=lambda: None),
        svc=None,
    )
    mason_mode.adopt(ctx, doc.MasonDoc(), title="A")
    return ctx


def test_placing_with_drop_to_ground_on_rests_the_new_node_on_the_surface():
    """``snap_ground`` was honoured by a drag and not by a placement click, so an
    armed Box clicked on the ground sat centred on the surface, half sunk."""
    from realmspinner.studio.modes.mason import assets as mason_assets
    from realmspinner.studio.modes.mason import mode as mason_mode
    from realmspinner.studio.modes.mason.engine import scene as msc

    for drop in (False, True):
        ctx = _placing_ctx()
        state = mason_mode.ensure(ctx)
        state.place_kind = "box"
        state.snap_ground = drop
        uid = mason_mode.place_armed(ctx, (2.0, 0.0, 3.0))
        assert uid is not None
        tab = mason_mode.active(ctx)
        source = mason_assets.ensure(ctx)
        source.primitives(tab.doc.node(uid).ref)  # resolve, as the next draw would
        lo, hi = msc.world_bounds(tab.doc, source, uids=[uid])
        if drop:
            assert lo[1] == pytest.approx(0.0, abs=1e-6)
            assert tab.doc.node(uid).translation[0] == pytest.approx(2.0)
            # One placement is one undo step, drop included.
            assert tab.doc.undo()
            assert tab.doc.node(uid) is None
        else:
            assert lo[1] < -0.1  # centred on the click: the unchanged behaviour


# --- mason-34 ----------------------------------------------------------------


def test_export_glb_does_not_overwrite_an_unrelated_json_beside_it(tmp_path):
    """The dialog only asked about ``level.glb``; the manifest goes to
    ``level.json`` through ``staged_set`` with no check of its own."""
    from realmspinner.service.errors import Conflict
    from realmspinner.studio.modes.mason import fileio as mason_io

    target = tmp_path / "level.glb"
    theirs = tmp_path / "level.json"
    theirs.write_text('{"my": "own file"}', encoding="utf-8")
    files = {"scene.glb": b"glb-bytes", "level.json": b'{"format": "realmspinner-mason-scene"}'}

    with pytest.raises(Conflict, match="level.json"):
        mason_io.write_files(files, target, primary="scene.glb")
    assert theirs.read_text(encoding="utf-8") == '{"my": "own file"}'
    assert not target.exists()  # nothing half-written


def test_export_replaces_a_previous_mason_export_sidecar_without_asking(tmp_path):
    from realmspinner.studio.modes.mason import fileio as mason_io
    from realmspinner.studio.modes.mason.engine import manifest

    target = tmp_path / "level.glb"
    previous = tmp_path / "level.json"
    previous.write_text(
        f'{{"format": "{manifest.FORMAT}", "version": 1}}', encoding="utf-8"
    )
    mason_io.write_files(
        {"scene.glb": b"new", "level.json": b'{"format": "%s"}' % manifest.FORMAT.encode()},
        target,
        primary="scene.glb",
    )
    assert target.read_bytes() == b"new"
    assert b"realmspinner-mason-scene" in previous.read_bytes()


def test_export_obj_does_not_overwrite_an_unrelated_mtl_or_texture_directory(tmp_path):
    from realmspinner.service.errors import Conflict
    from realmspinner.studio.modes.mason import fileio as mason_io
    from realmspinner.studio.modes.mason.engine import objout

    header = objout.EXPORT_HEADER.encode()
    files = {
        "level.obj": b"o x\n",
        "level.mtl": header + b"\nnewmtl m\n",
        "level_textures/0.png": b"png",
    }
    # Someone else's material library under the name this export would take.
    (tmp_path / "level.mtl").write_text("newmtl theirs\n", encoding="utf-8")
    with pytest.raises(Conflict, match="level.mtl"):
        mason_io.write_files(files, tmp_path / "level.obj", primary="level.obj")
    assert (tmp_path / "level.mtl").read_text(encoding="utf-8") == "newmtl theirs\n"

    # Their texture directory, with no Mason material library beside it, is theirs too.
    (tmp_path / "level.mtl").unlink()
    (tmp_path / "level_textures").mkdir()
    (tmp_path / "level_textures" / "0.png").write_bytes(b"theirs")
    with pytest.raises(Conflict, match="0.png"):
        mason_io.write_files(files, tmp_path / "level.obj", primary="level.obj")
    assert (tmp_path / "level_textures" / "0.png").read_bytes() == b"theirs"

    # A previous export of ours (header on the .mtl) is replaced.
    (tmp_path / "level_textures" / "0.png").unlink()
    (tmp_path / "level.mtl").write_bytes(header + b"\nold\n")
    mason_io.write_files(files, tmp_path / "level.obj", primary="level.obj")
    assert (tmp_path / "level_textures" / "0.png").read_bytes() == b"png"


# --- mason-35 ----------------------------------------------------------------


def test_palette_docstring_matches_that_library_rows_arm_rather_than_place():
    """The module docstring said only the library rows place immediately; the
    mason-mode-05 fix made them arm like every other row."""
    import inspect

    from realmspinner.studio.modes.mason.ui.panes import palette

    text = " ".join((palette.__doc__ or "").split())
    assert "Only the library rows place" not in text
    assert "Every** row **arms" in text
    # And the code agrees with the corrected sentence: a library row arms ``job:<id>``.
    assert 'f"job:{' in inspect.getsource(palette) or "job:" in inspect.getsource(palette)


def test_missing_refs_docstring_does_not_promise_a_proxy_or_a_relink_gesture():
    """The viewport draws nothing for an unresolved ref and no control calls
    ``set_ref``; the docstring promised both."""
    text = " ".join((doc.MasonDoc.missing_refs.__doc__ or "").split())
    assert "drawn as a missing-asset proxy" not in text
    assert "offered **Relink" not in text
    assert "draws **nothing**" in text
