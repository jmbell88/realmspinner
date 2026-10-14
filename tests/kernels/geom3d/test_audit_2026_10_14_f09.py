"""The 2026-10-14 audit, fixer F09: the glTF writer and loader, the charsheet prose, wavout.

One file for every finding F09 closed, because ``tests/`` has no ``__init__.py`` and a basename
must be unique across the tree: sirens-engine-11 (wavout's unstaged writer), poser-render-05
(charsheet prose naming a removed Poser), clay-io-02/03/04/05/06/07/09 and create-viewer-04
(``glbwrite`` / ``gltf``) and clay-io-03 (non-finite skin weights).
"""

from __future__ import annotations

import ast
import re
import struct
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from realmspinner.kernels.audio import wavout
from realmspinner.kernels.geom3d import glbwrite, gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.geom3d.glbio import read_glb, rebuild_glb, split_glb
from realmspinner.kernels.mesh import rigimport
from realmspinner.kernels.mesh.elements import OpError

SRC = Path(gltf.__file__).resolve().parents[2]
Q0 = [0.0, 0.0, 0.0, 1.0]
Q90 = [0.0, 0.0, float(np.sin(np.pi / 4)), float(np.cos(np.pi / 4))]


def _model(animations: list[gltf.Animation]) -> gltf.Model:
    nodes = [
        gltf.Node(name="root", children=[1]),
        gltf.Node(name="child", translation=m3.vec3(0, 1, 0)),
    ]
    return gltf.Model(nodes, [0], [], [], animations=animations)


def _anim(*channels: gltf.AnimChannel, name: str = "A") -> gltf.Animation:
    return gltf.Animation(name, list(channels))


# -- sirens-engine-11 ------------------------------------------------------------------------


def test_wavout_has_no_unstaged_file_writer(tmp_path: Path) -> None:
    tree = ast.parse(Path(wavout.__file__).read_text(encoding="utf-8"))
    opens = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open"
    ]
    assert opens == [], f"wavout opens a served name in place at line(s) {opens}"
    # And the behaviour the docstring used to ask the caller for: a write that fails while
    # encoding leaves the file that was there, and a clean one leaves no temp sibling.
    target = tmp_path / "a.wav"
    target.write_bytes(b"previous")
    with pytest.raises(ValueError):
        wavout.write(target, np.zeros((2, 2, 2), dtype=np.float32), 44100)
    assert target.read_bytes() == b"previous"
    pcm = np.zeros((16, 2), dtype=np.float32)
    wavout.write(target, pcm, 44100)
    assert target.read_bytes() == wavout.wav_bytes(pcm, 44100)
    assert [p.name for p in tmp_path.iterdir()] == ["a.wav"]


# -- poser-render-05 -------------------------------------------------------------------------


def test_charsheet_docstring_names_no_removed_module_or_test_path() -> None:
    stale = (
        "Poser's sheet planning",
        "modes.poser",
        "modes/poser",
        "retired Troupe door",
        "Troupe's for a rendered sheet",
        'is the "rendered character sheet"',
        "for a rendered character sheet",
    )
    for rel in ("kernels/charsheet.py", "kernels/sheet.py", "pipelines/spritesynth.py"):
        text = (SRC / rel).read_text(encoding="utf-8")
        for phrase in stale:
            assert phrase not in text, f"{rel} still says {phrase!r}"


def test_charsheet_refusals_name_no_removed_feature() -> None:
    tree = ast.parse((SRC / "kernels/charsheet.py").read_text(encoding="utf-8"))
    named = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            for part in ast.walk(node.exc):
                if (
                    isinstance(part, ast.Constant)
                    and isinstance(part.value, str)
                    and "Troupe" in part.value
                ):
                    named.append((node.lineno, part.value))
    assert named == [], f"a user-facing refusal names the removed Troupe: {named}"


# -- clay-io-02, 05, 06: the animation writer ------------------------------------------------


def test_write_glb_does_not_emit_an_animation_with_no_channels() -> None:
    walk = _anim(gltf.AnimChannel(1, "translation", np.array([0.0]), np.array([[0.0, 1.0, 0.0]])))
    doc, _ = read_glb(glbwrite.write_glb(_model([_anim(name="Empty"), walk])))
    assert [a["name"] for a in doc["animations"]] == ["A"]
    for animation in doc["animations"]:
        assert animation["samplers"] and animation["channels"], "both arrays have minItems 1"
    doc, _ = read_glb(glbwrite.write_glb(_model([_anim(name="Empty")])))
    assert "animations" not in doc


def test_the_key_time_accessor_bounds_equal_the_float32_values_written() -> None:
    times = np.array([0.1, 7 / 24, 1.3])
    channel = gltf.AnimChannel(1, "translation", times, np.zeros((3, 3)))
    data = glbwrite.write_glb(_model([_anim(channel)]))
    doc, binary = read_glb(data)
    accessor = doc["accessors"][doc["animations"][0]["samplers"][0]["input"]]
    view = doc["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stored = np.frombuffer(binary, dtype="<f4", count=accessor["count"], offset=start)
    assert accessor["min"] == [float(stored.min())]
    assert accessor["max"] == [float(stored.max())]


def _writes(channel: gltf.AnimChannel, *more: gltf.AnimChannel) -> bytes:
    return glbwrite.write_glb(_model([_anim(channel, *more)]))


def test_write_glb_refuses_decreasing_key_times_and_non_unit_rotation_keys() -> None:
    two = np.zeros((2, 3))
    for times in ([1.0, 0.5], [-0.5, 1.0], [0.5, 0.5], [500.0, 500.00001]):
        with pytest.raises(ValueError, match="key times"):
            _writes(gltf.AnimChannel(1, "translation", np.array(times), two))
    wide = np.array([[0.0, 0.0, 0.0, 2.0], Q0])
    with pytest.raises(ValueError, match="unit"):
        _writes(gltf.AnimChannel(1, "rotation", np.array([0.0, 1.0]), wide))
    with pytest.raises(ValueError, match="unit"):
        _writes(gltf.AnimChannel(1, "rotation", np.array([0.0]), np.zeros((1, 4))))


def test_write_glb_refuses_the_other_spec_invalid_animation_shapes() -> None:
    one = np.array([0.0])
    with pytest.raises(ValueError, match="more than once"):
        _writes(
            gltf.AnimChannel(1, "translation", one, np.zeros((1, 3))),
            gltf.AnimChannel(1, "translation", one, np.zeros((1, 3))),
        )
    with pytest.raises(ValueError, match="CUBICSPLINE"):
        _writes(gltf.AnimChannel(1, "translation", one, np.zeros((3, 3)), "CUBICSPLINE"))
    skin = gltf.Skin(joints=[1, 1], inverse_bind=np.stack([np.eye(4), np.eye(4)]))
    nodes = [gltf.Node(name="root", children=[1]), gltf.Node(name="bone")]
    with pytest.raises(ValueError, match="more than once"):
        glbwrite.write_glb(gltf.Model(nodes, [0], [], [skin]))


def test_a_unit_rotation_within_float_noise_is_still_written() -> None:
    quat = np.array(Q90) * (1.0 + 1e-6)
    channel = gltf.AnimChannel(1, "rotation", np.array([0.0, 1.0]), np.stack([quat, quat]))
    out = gltf.load(_writes(channel))
    assert np.allclose(np.linalg.norm(out.animations[0].channels[0].values, axis=1), 1.0)


# -- clay-io-09 ------------------------------------------------------------------------------


def test_no_comment_in_geom3d_names_the_removed_poser_mode() -> None:
    hits = []
    for path in sorted((SRC / "kernels/geom3d").glob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\bPoser\b|\bTroupe\b", line):
                hits.append(f"{path.name}:{number}")
    assert hits == [], f"geom3d still names the removed Poser/Troupe at {hits}"


# -- create-viewer-04 ------------------------------------------------------------------------


def _with_json(data: bytes, edit) -> bytes:
    header, doc, rest = split_glb(data)
    edit(doc)
    return rebuild_glb(header, doc, rest)


def _scene_glb() -> bytes:
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4")
    prim = gltf.Primitive(positions=pos, indices=np.array([0, 1, 2], dtype="u4"))
    nodes = [
        gltf.Node(name="m", mesh=0),
        gltf.Node(name="c", camera=0),
        gltf.Node(name="l", light=0),
    ]
    model = gltf.Model(
        nodes, [0, 1, 2], [[prim]], [], cameras=[gltf.Camera()], lights=[gltf.Light(kind="point")]
    )
    return glbwrite.write_glb(model)


def test_a_huge_integer_in_a_material_factor_falls_back_to_the_default() -> None:
    huge = 10**400

    def edit(doc: dict) -> None:
        pbr = doc["materials"][0].setdefault("pbrMetallicRoughness", {})
        pbr["baseColorFactor"] = [huge, 1, 1, 1]
        pbr["metallicFactor"] = huge
        pbr["roughnessFactor"] = huge
        doc["materials"][0]["emissiveFactor"] = [huge, 0, 0]
        doc["cameras"][0]["perspective"]["yfov"] = huge
        doc["cameras"][0]["perspective"]["znear"] = huge
        light = doc["extensions"]["KHR_lights_punctual"]["lights"][0]
        light["color"] = [huge, 1, 1]
        light["intensity"] = huge

    model = gltf.load(_with_json(_scene_glb(), edit))
    material = model.meshes[0][0].material
    assert tuple(material.base_color_factor) == (1.0, 1.0, 1.0, 1.0)
    assert material.metallic_factor == 1.0 and material.roughness_factor == 1.0
    assert model.cameras[0].yfov == gltf.Camera.yfov
    assert model.lights[0].intensity == gltf.Light.intensity


# -- clay-io-03: non-finite skin weights -----------------------------------------------------


def _skinned(weights) -> gltf.Model:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype="f4")
    prim = gltf.Primitive(
        positions=positions,
        indices=np.array([0, 1, 2, 1, 3, 2], dtype="u4"),
        joints=np.array([[0, 1, 0, 0]] * 4, dtype="i4"),
        weights=np.array(weights, dtype="f4"),
    )
    nodes = [
        gltf.Node(name="root", children=[1]),
        gltf.Node(name="hip", translation=m3.vec3(0, 1, 0), children=[2]),
        gltf.Node(name="knee", translation=m3.vec3(0, 1, 0)),
        gltf.Node(name="body", mesh=0, skin=0),
    ]
    skin = gltf.Skin(joints=[1, 2], inverse_bind=np.stack([np.eye(4), np.eye(4)]))
    return gltf.Model(nodes, [0, 3], [[prim]], [skin])


def _glb_with_weight(value: float) -> bytes:
    """A valid skinned GLB whose first vertex then has *value* written over a weight."""
    data = glbwrite.write_glb(_skinned([[0.5, 0.5, 0.0, 0.0]] * 4))
    header, doc, rest = split_glb(data)
    accessor = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["WEIGHTS_0"]]
    view = doc["bufferViews"][accessor["bufferView"]]
    where = 8 + view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    blob = bytearray(rest)
    struct.pack_into("<f", blob, where, value)
    return rebuild_glb(header, doc, bytes(blob), allow_nan=True)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_glb_with_a_nan_skin_weight_is_refused_at_import(bad: float) -> None:
    with pytest.raises(ValueError, match="weight"):
        gltf.load(_glb_with_weight(bad))
    from realmspinner.kernels.mesh import meshimport

    with pytest.raises(ValueError, match="weight"):
        meshimport.import_file(_glb_with_weight(bad), ".glb")


def test_rigimport_refuses_a_non_finite_weight_by_name() -> None:
    weights = np.array([[np.nan, 0.5, 0.0, 0.0]] * 2)
    with pytest.raises(OpError, match="weight"):
        rigimport._normalised(weights)
    assert np.allclose(rigimport._normalised(np.array([[1.0, 1.0, 0.0, 0.0]])).sum(), 1.0)


def test_write_glb_refuses_non_finite_skin_weights() -> None:
    with pytest.raises(ValueError, match="weight"):
        glbwrite.write_glb(_skinned([[np.nan, 0.5, 0.0, 0.0]] * 4))
    with pytest.raises(ValueError, match="weight"):
        glbwrite.write_glb(_skinned([[np.inf, 0.5, 0.0, 0.0]] * 4))


# -- clay-io-04 ------------------------------------------------------------------------------


def _peak_of_loading(component_type: int, rows: int) -> int:
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{}],
        "accessors": [
            {"componentType": 5126, "type": "SCALAR", "count": 1},
            {"componentType": component_type, "type": "VEC4", "count": rows, "normalized": True},
        ],
        "animations": [
            {
                "samplers": [{"input": 0, "output": 1}],
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "rotation"}}],
            }
        ],
    }
    data = rebuild_glb(struct.pack("<III", 0x46546C67, 2, 0), doc, b"")
    tracemalloc.start()
    try:
        with pytest.raises(ValueError):
            gltf.load(data)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_a_signed_normalised_animation_output_peaks_inside_the_document_budget() -> None:
    rows = 1_000_000
    unsigned = _peak_of_loading(5123, rows)  # clamps nothing: the baseline for the same decode
    signed = _peak_of_loading(5122, rows)  # clamps in place, so no third array exists
    one_array = rows * 4 * 8
    assert signed < unsigned + one_array // 2, (signed, unsigned, one_array)


# -- clay-io-07 ------------------------------------------------------------------------------


def test_a_non_unit_node_rotation_is_normalised_on_load() -> None:
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4")
    prim = gltf.Primitive(positions=pos, indices=np.array([0, 1, 2], dtype="u4"))
    data = glbwrite.write_glb(gltf.Model([gltf.Node(name="m", mesh=0)], [0], [[prim]], []))

    def edit(doc: dict) -> None:
        doc["nodes"][0]["rotation"] = [0.0, 0.0, 0.5, 0.5]

    node = gltf.load(_with_json(data, edit)).nodes[0]
    assert np.isclose(np.linalg.norm(node.rotation), 1.0)
    assert np.allclose(node.rotation, [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    assert np.allclose(
        m3.compose(node.translation, node.rotation, node.scale)[:3, :3].T
        @ m3.compose(node.translation, node.rotation, node.scale)[:3, :3],
        np.eye(3),
    )

    def zero(doc: dict) -> None:
        doc["nodes"][0]["rotation"] = [0.0, 0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="rotation"):
        gltf.load(_with_json(data, zero))
