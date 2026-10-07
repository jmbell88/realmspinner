"""The 2026-10-07 Clay audit, the ``.rblk`` serializer and the v3 migration.

clay-01 (writer half), clay-05, clay-43, clay-44, clay-45, clay-56 and clay-57.
Each test's name is its claim; each was run against the unfixed code first.
"""

from __future__ import annotations

import ast
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import realmspinner
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import glbimport
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser

SRC = Path(realmspinner.__file__).parent


def _box_doc(n: int = 1) -> bd.ClayDoc:
    doc = bd.ClayDoc()
    for i in range(n):
        doc.add_object(bd.Obj(uid=bd.new_uid(), name=f"B{i}", mesh=bp.box()))
    return doc


def _rewrite(data: bytes, edit_scene=None, replace: dict[str, bytes] | None = None) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            raw = src.read(name)
            if name == ser.SCENE and edit_scene is not None:
                scene = json.loads(raw)
                edit_scene(scene)
                raw = json.dumps(scene).encode()
            if replace and name in replace:
                raw = replace[name]
            dst.writestr(name, raw)
    return out.getvalue()


def _mesh_with(positions=None, uv=None) -> bm.Mesh:
    base = bp.box()
    return bm.Mesh(
        positions=base.positions if positions is None else positions,
        loops=base.loops,
        starts=base.starts,
        material=base.material,
        smooth=base.smooth,
        uv=uv,
    )


# --- clay-01, the writer half ---------------------------------------------------


def test_a_document_whose_object_has_zero_scale_is_refused_by_the_writer_with_a_sentence() -> None:
    doc = _box_doc()
    doc.objects[0].scale = np.zeros(3)  # set directly: ``set_transform`` would refuse it
    with pytest.raises(ValueError, match=r"B0.*scale|scale.*B0"):
        ser.snapshot_bytes(ser.snapshot(doc))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("translation", (0.0, float("nan"), 0.0)),
        ("translation", (float("inf"), 0.0, 0.0)),
        ("rotation", (0.0, 0.0, float("nan"), 1.0)),
        ("rotation", (0.0, 0.0, 0.0, 0.0)),
        ("scale", (1.0, float("inf"), 1.0)),
    ],
)
def test_the_writer_refuses_every_transform_the_reader_would_refuse(field, value) -> None:
    doc = _box_doc()
    setattr(doc.objects[0], field, np.array(value, dtype="f8"))
    with pytest.raises(ValueError, match=field):
        ser.rblk_bytes(doc)


def test_a_document_with_ordinary_transforms_still_saves_and_reopens() -> None:
    doc = _box_doc(2)
    doc.objects[1].scale = np.array([2.0, 0.5, 1.0])
    again = ser.read_rblk(ser.rblk_bytes(doc))
    assert [o.name for o in again.objects] == ["B0", "B1"]


# --- clay-05 ----------------------------------------------------------------------


def test_a_v3_bake_never_leaves_a_document_past_the_triangle_ceiling_the_writer_refuses(
    monkeypatch,
) -> None:
    n, count = 10, 50
    data = ser.rblk_bytes(_box_doc(n))  # 12 triangles a box: 120 to begin with

    def to_v3(scene):
        scene["version"] = 3
        for entry in scene["objects"]:
            entry["modifiers"] = [
                {"id": 1, "kind": "array", "enabled": True, "params": {"count": count}}
            ]

    data = _rewrite(data, to_v3)
    # Each array alone (600) is inside the ceiling; ten of them (6,000) are not.
    monkeypatch.setattr(glbimport, "MAX_TRIANGLES", 1_000)
    doc = ser.read_rblk(data)
    total = sum(ser.triangle_count(o.mesh) for o in doc.objects)
    assert total <= 1_000, f"the bake left {total:,} triangles"
    ser.rblk_bytes(doc)  # the writer must accept what the reader handed back
    text = "\n".join(doc.notices)
    assert "Array modifier could not run" in text and "triangles" in text
    assert any(ser.triangle_count(o.mesh) > 12 for o in doc.objects), "nothing baked at all"


# --- clay-43 ----------------------------------------------------------------------


def test_a_version_4_file_naming_a_shape_clay_does_not_offer_opens_frozen() -> None:
    assert "lathe" not in bp.CLAY_GENERATOR_NAMES
    data = ser.rblk_bytes(_box_doc())

    def edit(scene):
        scene["objects"][0]["generator"] = "lathe"
        scene["objects"][0]["params"] = {"segments": 12}

    doc = ser.read_rblk(_rewrite(data, edit))
    obj = doc.objects[0]
    assert obj.generator is None and obj.params == {}
    assert len(obj.mesh.positions) > 0
    assert any("no longer builds" in n and '"B0"' in n for n in doc.notices)


def test_a_version_4_file_naming_a_shape_clay_offers_keeps_its_generator() -> None:
    doc = bd.ClayDoc()
    doc.add_object(
        bd.Obj(
            uid=bd.new_uid(), name="B", mesh=bp.box(), generator="box",
            params={"size": (1.0, 1.0, 1.0)},
        )
    )
    again = ser.read_rblk(ser.rblk_bytes(doc))
    assert again.objects[0].generator == "box" and again.notices == ()


# --- clay-44 ----------------------------------------------------------------------


def _nested_scene_archive() -> bytes:
    depth = 200_000
    deep = ("[" * depth + "]" * depth).encode()
    return _rewrite(ser.rblk_bytes(_box_doc()), replace={ser.SCENE: deep})


def test_a_scene_json_nested_past_the_recursion_limit_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="not a Realmspinner Clay document"):
        ser.read_rblk(_nested_scene_archive())


def test_a_scene_json_nested_past_the_recursion_limit_has_no_view() -> None:
    assert ser.read_view(_nested_scene_archive()) is None


def test_a_scene_json_that_is_not_an_object_has_no_view() -> None:
    data = _rewrite(ser.rblk_bytes(_box_doc()), replace={ser.SCENE: b"[1, 2, 3]"})
    assert ser.read_view(data) is None


# --- clay-45 ----------------------------------------------------------------------


def _uid_member(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return next(n for n in zf.namelist() if n.startswith(f"{ser.MESH_DIR}/"))


def _archive_with_member(mesh: bm.Mesh) -> bytes:
    data = ser.rblk_bytes(_box_doc())
    arrays = {name: getattr(mesh, name) for name in ser._MESH_FIELDS}
    if mesh.uv is not None:
        arrays["uv"] = mesh.uv
    return _rewrite(data, replace={_uid_member(data): ser._npz_bytes(arrays)})


def test_read_rblk_refuses_a_mesh_with_a_non_finite_position() -> None:
    positions = bp.box().positions.copy()
    positions[3, 1] = np.nan
    with pytest.raises(ValueError, match="position"):
        ser.read_rblk(_archive_with_member(_mesh_with(positions=positions)))


def test_read_rblk_refuses_a_mesh_with_a_non_finite_uv() -> None:
    base = bp.box()
    uv = np.zeros((len(base.loops), 2), dtype="f4")
    uv[2, 0] = np.inf
    with pytest.raises(ValueError, match="uv"):
        ser.read_rblk(_archive_with_member(_mesh_with(uv=uv)))


def test_the_writer_refuses_a_mesh_with_a_non_finite_position() -> None:
    positions = bp.box().positions.copy()
    positions[0, 0] = np.inf
    doc = _box_doc()
    doc.objects[0].mesh = _mesh_with(positions=positions)
    with pytest.raises(ValueError, match="B0.*position|position.*B0"):
        ser.rblk_bytes(doc)


def test_the_writer_refuses_a_mesh_with_a_non_finite_uv() -> None:
    base = bp.box()
    uv = np.zeros((len(base.loops), 2), dtype="f4")
    uv[0, 1] = np.nan
    doc = _box_doc()
    doc.objects[0].mesh = _mesh_with(uv=uv)
    with pytest.raises(ValueError, match="B0.*uv|uv.*B0"):
        ser.rblk_bytes(doc)


# --- clay-56 ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["legacy.py", "serialize.py"])
def test_no_docstring_in_kernels_mesh_names_the_deleted_modifiers_module(name) -> None:
    text = (SRC / "kernels" / "mesh" / name).read_text(encoding="utf-8")
    assert not (SRC / "kernels" / "mesh" / "modifiers.py").exists()
    for gone in (":mod:`.modifiers`", "via .modifiers", "files.unready_reason"):
        assert gone not in text, f"{name} still names {gone}"


# --- clay-57 ----------------------------------------------------------------------

DATA = Path(__file__).parent / "data"


def test_a_v3_file_bakes_the_kernels_the_first_fixture_does_not_pin() -> None:
    """radial-array (closed and open), solidify, subdivide, triangulate, smooth and
    a four-modifier chain, against what the pre-112c69e2 evaluator returned for
    ``data/v3_kernels.rblk`` (``ClayDoc.evaluation`` from that commit's tree,
    loaded as a throwaway package, never checked out)."""
    doc = ser.read_rblk((DATA / "v3_kernels.rblk").read_bytes())
    expected = np.load(DATA / "v3_kernels_expected.npz")
    assert [o.name for o in doc.objects] == [
        "radial", "radial_arc", "solid", "subdiv", "tri", "smooth", "chain"
    ]
    for obj in doc.objects:
        for field in ("positions", "loops", "starts", "material", "smooth"):
            got = np.asarray(getattr(obj.mesh, field))
            want = expected[f"{obj.name}.{field}"]
            assert got.shape == want.shape, (obj.name, field)
            assert np.array_equal(got, want), (obj.name, field)
    assert doc.notices and "modifiers" in doc.notices[0]


LEGACY = "realmspinner.kernels.mesh.legacy"


def _imports_of(path: Path) -> set[str]:
    """Every absolute dotted name *path* could be importing, bare or ``from``."""
    module = ".".join(path.relative_to(SRC.parent).with_suffix("").parts)
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
            if node.level == 0:
                base = ""
            target = ".".join(p for p in (base, node.module or "") if p)
            found.add(target)
            found.update(f"{target}.{alias.name}" for alias in node.names)
    return found


def test_only_serialize_imports_the_legacy_migration_module() -> None:
    importers = {
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if path.name != "legacy.py" and LEGACY in _imports_of(path)
    }
    assert importers == {"kernels/mesh/serialize.py"}
