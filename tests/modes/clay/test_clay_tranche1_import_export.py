"""``clay_mode``'s file doors: import through one function for every mesh
suffix, and the two flavours of file export (GLB, OBJ+MTL) beside the
library export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import trimesh

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import meshimport
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import dialogs
from realmspinner.studio.modes.clay import mode as clay_mode


class FakeCtx:
    """``test_clay_mode.py``'s own double, reproduced locally rather than
    imported: a private test helper of another module, not a public fixture."""

    def __init__(self, *, accept: bool = True) -> None:
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


class _AppState:
    def __init__(self) -> None:
        self.clay = None
        self.mode = "home"


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


def _glb_bytes() -> bytes:
    from realmspinner.kernels.geom3d import glbwrite

    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    return glbwrite.write_glb(bd.to_model(doc))


def _obj_bytes() -> bytes:
    return b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


def _cube_bytes(file_type: str) -> bytes:
    data = trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(file_type=file_type)
    return data if isinstance(data, bytes) else data.encode("utf-8")


def _tab(ctx: FakeCtx) -> Any:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    return clay_mode.adopt(ctx, doc, title="Scene")


class _Done:
    def __init__(self, key: str, result: Any) -> None:
        self.key = key
        self.result = result


_SUFFIX_BYTES = {
    ".glb": _glb_bytes,
    ".obj": _obj_bytes,
    ".stl": lambda: _cube_bytes("stl"),
    ".ply": lambda: _cube_bytes("ply"),
}


# --- import: one door for every suffix ----------------------------------------


@pytest.mark.parametrize("suffix", meshimport.SUPPORTED_SUFFIXES)
def test_import_mesh_path_routes_every_supported_suffix(tmp_path: Path, suffix: str) -> None:
    ctx = FakeCtx()
    path = tmp_path / f"thing{suffix}"
    path.write_bytes(_SUFFIX_BYTES[suffix]())

    clay_mode.import_mesh_path(ctx, path)

    assert ctx.submitted[-1].startswith("clay-import:")
    assert ctx.result is not None
    assert len(ctx.result["doc"].objects) >= 1
    assert ctx.result["title"] == "thing"


def test_import_glb_path_is_a_thin_alias_of_import_mesh_path(tmp_path: Path) -> None:
    ctx = FakeCtx()
    path = tmp_path / "thing.glb"
    path.write_bytes(_glb_bytes())

    clay_mode.import_glb_path(ctx, path)

    assert ctx.submitted[-1].startswith("clay-import:")
    assert len(ctx.result["doc"].objects) == 1


def test_import_mesh_path_remembers_scale_and_up_for_the_next_import(tmp_path: Path) -> None:
    ctx = FakeCtx()
    state = clay_mode.ensure(ctx)
    assert state.import_scale == 1.0
    assert state.import_up == "y"

    path = tmp_path / "a.glb"
    path.write_bytes(_glb_bytes())
    clay_mode.import_mesh_path(ctx, path, scale=0.01, up="z")
    assert state.import_scale == 0.01
    assert state.import_up == "z"

    # A second import with no explicit scale/up reuses what was just set --
    # the whole point of remembering it rather than defaulting every time.
    path2 = tmp_path / "b.glb"
    path2.write_bytes(_glb_bytes())
    doc_default_scale, _up = state.import_scale, state.import_up
    clay_mode.import_mesh_path(ctx, path2)
    assert state.import_scale == doc_default_scale == 0.01
    assert state.import_up == "z"


def test_import_mesh_path_applies_the_remembered_scale(tmp_path: Path) -> None:
    from realmspinner.kernels.mesh import mesh as bm

    ctx = FakeCtx()
    path_plain = tmp_path / "plain.glb"
    path_plain.write_bytes(_glb_bytes())
    clay_mode.import_mesh_path(ctx, path_plain)
    plain_lo, plain_hi = bm.bounds(ctx.result["doc"].objects[0].mesh)

    path_scaled = tmp_path / "scaled.glb"
    path_scaled.write_bytes(_glb_bytes())
    clay_mode.import_mesh_path(ctx, path_scaled, scale=2.0, up="y")
    lo, hi = bm.bounds(ctx.result["doc"].objects[0].mesh)

    assert (hi - lo)[0] == pytest.approx((plain_hi - plain_lo)[0] * 2.0)


def test_ask_import_mesh_opens_a_picker_then_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "thing.glb"
    path.write_bytes(_glb_bytes())
    monkeypatch.setattr(dialogs, "open_file", lambda *a, **k: path)

    ctx = FakeCtx()
    clay_mode.ask_import_mesh(ctx)

    assert ctx.submitted[-1] == "clay-import"
    assert ctx.result["doc"].objects


def test_ask_import_mesh_a_cancelled_picker_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dialogs, "open_file", lambda *a, **k: None)
    ctx = FakeCtx()
    clay_mode.ask_import_mesh(ctx)
    assert ctx.result is None


def test_an_unsupported_suffix_names_the_full_list_in_the_refusal() -> None:
    from realmspinner.kernels.mesh.elements import OpError

    with pytest.raises(OpError) as excinfo:
        meshimport.import_file(b"nope", ".fbx", "X")
    for suffix in meshimport.SUPPORTED_SUFFIXES:
        assert suffix in str(excinfo.value)


# --- export: GLB and OBJ+MTL beside the library export ------------------------


def test_export_mesh_file_glb_writes_a_readable_glb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from realmspinner.kernels.geom3d import gltf as gltf_mod

    ctx = FakeCtx()
    tab = _tab(ctx)
    out = tmp_path / "out.glb"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: out)

    clay_mode.export_mesh_file(ctx, tab, "glb")
    clay_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))

    assert out.is_file()
    model = gltf_mod.load(out.read_bytes())
    assert model.nodes
    assert any("Exported to" in m for m, _ in ctx.toasts)


def test_export_mesh_file_obj_writes_both_files_and_the_obj_names_the_mtl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = FakeCtx()
    tab = _tab(ctx)
    out = tmp_path / "renamed.obj"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: out)

    clay_mode.export_mesh_file(ctx, tab, "obj")
    clay_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))

    assert out.is_file()
    mtl = tmp_path / "renamed.mtl"
    assert mtl.is_file()
    text = out.read_text(encoding="utf-8")
    assert "mtllib renamed.mtl" in text
    assert "mtllib Scene.mtl" not in text, "the tab's own title, not the chosen filename"


def test_export_mesh_file_refuses_when_nothing_is_visible(tmp_path: Path) -> None:
    ctx = FakeCtx()
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    tab = clay_mode.adopt(ctx, doc, title="Scene")
    doc.set_props(obj.uid, visible=False)

    clay_mode.export_mesh_file(ctx, tab, "glb")

    assert not ctx.submitted
    assert any(level == "error" for _m, level in ctx.toasts)


def test_export_mesh_file_uses_the_shared_saving_flag_and_key_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_start`` -- ``save_to``/``save_as``'s own submit helper -- is what
    this reuses rather than a hand-rolled submit, so a refused submit (a
    second export already in flight) unlocks the tab exactly as a refused
    save does."""
    ctx = FakeCtx(accept=False)
    tab = _tab(ctx)
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "out.glb")

    clay_mode.export_mesh_file(ctx, tab, "glb")

    assert ctx.submitted[-1].startswith("clay-exportfile:")
    assert tab.saving is False, "a refused submit must not leave the tab locked"


def test_export_mesh_file_does_not_mark_the_document_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain file export writes a copy; it is not the ``.rblk`` that makes
    the document itself clean, so a successful export leaves the tab exactly
    as dirty as it was and touches no journal entry."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    doc.set_props(doc.objects[0].uid, name="Edited")
    assert tab.dirty
    saved_head_before = tab.saved_head
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "out.glb")

    clay_mode.export_mesh_file(ctx, tab, "glb")
    clay_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))

    assert tab.dirty, "the document is still exactly as unsaved as before"
    assert tab.saved_head == saved_head_before, "mark_saved was never called"
    assert (tmp_path / "out.glb").is_file()


# --- the round trip: an exported OBJ keeps its colours coming back ------------


def test_an_obj_exported_from_clay_imports_with_its_material_colour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = FakeCtx()
    tab = _tab(ctx)
    red = (0.8, 0.1, 0.1, 1.0)
    tab.doc.materials[0] = _with_colour(tab.doc.materials[0], red)
    out = tmp_path / "painted.obj"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: out)
    clay_mode.export_mesh_file(ctx, tab, "obj")
    clay_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))

    importer = FakeCtx()
    clay_mode.import_mesh_path(importer, out)
    doc = importer.result["doc"]

    used = {int(i) for obj in doc.objects for i in obj.mesh.material}
    colours = [tuple(round(c, 4) for c in doc.materials[i].base_color_factor) for i in used]
    assert tuple(round(c, 4) for c in red) in colours


def test_an_mtllib_with_a_directory_part_is_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "stolen.mtl").write_text("newmtl m\nKd 1 0 0\n", encoding="utf-8")
    obj = tmp_path / "sneaky.obj"
    obj.write_bytes(b"mtllib elsewhere/stolen.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl m\nf 1 2 3\n")

    assert clay_mode._sibling_mtl(obj, obj.read_bytes()) is None


def _with_colour(material: Any, rgba: tuple[float, ...]) -> Any:
    import dataclasses

    return dataclasses.replace(material, base_color_factor=rgba)
