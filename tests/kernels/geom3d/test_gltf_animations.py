"""glTF animations: read by the loader, written by the writer.

Tranche 3 of the 2026-10-13 Clay rig plan. Nothing read or wrote ``"animations"`` before. Verified
the way the writer's other claims are -- through the independent loader -- plus the container facts
a tolerant loader cannot see (the ``min``/``max`` the spec requires of an animation input, one
shared input accessor for channels that share their key times).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite, gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.geom3d.glbio import read_glb, rebuild_glb

Q0 = [0.0, 0.0, 0.0, 1.0]
Q90 = [0.0, 0.0, float(np.sin(np.pi / 4)), float(np.cos(np.pi / 4))]


def _model(animations: list[gltf.Animation]) -> gltf.Model:
    nodes = [
        gltf.Node(name="root", children=[1]),
        gltf.Node(name="child", translation=m3.vec3(0, 1, 0)),
    ]
    return gltf.Model(nodes, [0], [], [], animations=animations)


def _walk() -> gltf.Animation:
    times = np.array([0.0, 0.5, 1.0])
    return gltf.Animation(
        "Walk",
        [
            gltf.AnimChannel(1, "rotation", times, np.array([Q0, Q90, Q0]), "LINEAR"),
            gltf.AnimChannel(
                1, "translation", times, np.array([[0, 1, 0], [0, 2, 0], [0, 1, 0]]), "STEP"
            ),
            gltf.AnimChannel(0, "scale", np.array([0.0]), np.array([[2.0, 2.0, 2.0]]), "LINEAR"),
        ],
    )


def test_animations_round_trip_through_the_loader() -> None:
    out = gltf.load(glbwrite.write_glb(_model([_walk()])))
    [walk] = out.animations
    assert walk.name == "Walk" and len(walk.channels) == 3
    rot, trans, scale = walk.channels
    assert (rot.node, rot.path, rot.interpolation) == (1, "rotation", "LINEAR")
    assert np.allclose(rot.times, [0, 0.5, 1]) and np.allclose(rot.values[1], Q90)
    assert (trans.path, trans.interpolation) == ("translation", "STEP")
    assert (scale.node, scale.path) == (0, "scale") and np.allclose(scale.values, [[2, 2, 2]])
    assert out.skipped_animation_channels == 0


def test_a_model_without_animations_writes_no_animations_key() -> None:
    doc, _ = read_glb(glbwrite.write_glb(_model([])))
    assert "animations" not in doc


def test_the_input_accessor_carries_min_and_max_and_equal_times_share_one_accessor() -> None:
    times = np.array([0.25, 1.5])
    shared = [
        gltf.AnimChannel(1, "rotation", times, np.array([Q0, Q90])),
        gltf.AnimChannel(1, "translation", times, np.array([[0, 0, 0], [1, 0, 0]])),
    ]
    doc, _ = read_glb(glbwrite.write_glb(_model([gltf.Animation("A", shared)])))
    samplers = doc["animations"][0]["samplers"]
    assert samplers[0]["input"] == samplers[1]["input"], "one array, one accessor"
    accessor = doc["accessors"][samplers[0]["input"]]
    assert accessor["min"] == [0.25] and accessor["max"] == [1.5]
    assert accessor["componentType"] == 5126 and accessor["type"] == "SCALAR"
    assert doc["accessors"][samplers[0]["output"]]["type"] == "VEC4"
    assert doc["accessors"][samplers[1]["output"]]["type"] == "VEC3"


def test_extras_survive_and_a_cubic_spline_keeps_its_tangents() -> None:
    times = np.array([0.0, 1.0])
    values = np.zeros((6, 3))
    values[1] = [0, 0, 0]
    values[4] = [1, 0, 0]
    cubic = gltf.AnimChannel(1, "translation", times, values, "CUBICSPLINE")
    out = gltf.load(
        glbwrite.write_glb(_model([gltf.Animation("Smooth", [cubic], {"loop": False})]))
    )
    [animation] = out.animations
    assert animation.extras == {"loop": False}
    [channel] = animation.channels
    assert channel.interpolation == "CUBICSPLINE" and channel.values.shape == (6, 3)
    assert np.allclose(channel.values[4], [1, 0, 0])


# -- the writer refuses what a reader could not play -------------------------------------------


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: gltf.AnimChannel(9, "rotation", np.array([0.0]), np.array([Q0])), "node 9"),
        (lambda: gltf.AnimChannel(1, "weights", np.array([0.0]), np.array([[0.0] * 3])), "weights"),
        (
            lambda: gltf.AnimChannel(1, "rotation", np.array([0.0]), np.array([Q0]), "SMOOTHSTEP"),
            "SMOOTHSTEP",
        ),
        (
            lambda: gltf.AnimChannel(1, "rotation", np.array([0.0, 1.0]), np.array([Q0])),
            "key time",
        ),
        (lambda: gltf.AnimChannel(1, "rotation", np.array([]), np.zeros((0, 4))), "key time"),
        (
            lambda: gltf.AnimChannel(
                1, "translation", np.array([float("nan")]), np.array([[0.0, 0.0, 0.0]])
            ),
            "finite",
        ),
        (
            lambda: gltf.AnimChannel(1, "translation", np.array([0.0]), np.array([[0.0, 0.0]])),
            "key time",
        ),
    ],
)
def test_the_writer_refuses_a_channel_that_does_not_hold_together(make, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        glbwrite.write_glb(_model([gltf.Animation("Bad", [make()])]))


# -- the loader's own refusals ------------------------------------------------------------------


def _edited(edit) -> bytes:
    """The writer's bytes for a walk, with the glTF JSON passed through *edit*."""
    data = glbwrite.write_glb(_model([_walk()]))
    doc, binary = read_glb(data)
    edit(doc)
    header = data[:12]
    chunk = b"" if not binary else (len(binary)).to_bytes(4, "little") + b"BIN\x00" + binary
    return rebuild_glb(header, doc, chunk)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda d: d["animations"][0]["channels"][0]["target"].__setitem__("node", 99),
            "does not exist",
        ),
        (lambda d: d["animations"][0]["channels"][0].__setitem__("sampler", 99), "sampler"),
        (
            lambda d: d["animations"][0]["samplers"][0].__setitem__("interpolation", "BOUNCE"),
            "BOUNCE",
        ),
        (lambda d: d["animations"][0].__setitem__("channels", {}), "channels"),
        (lambda d: d.__setitem__("animations", {}), "animations"),
        (lambda d: d["animations"].__setitem__(0, 5), "animations"),
        (lambda d: d["animations"][0]["channels"].__setitem__(0, "x"), "channel"),
    ],
)
def test_the_loader_refuses_an_animation_that_does_not_hold_together_by_name(edit, message) -> None:
    with pytest.raises(ValueError, match=message):
        gltf.load(_edited(edit))


def test_a_channel_that_drives_something_else_is_counted_and_left_out() -> None:
    def edit(doc: dict) -> None:
        doc["animations"][0]["channels"][0]["target"]["path"] = "weights"

    out = gltf.load(_edited(edit))
    assert out.skipped_animation_channels == 1 and len(out.animations[0].channels) == 2


def test_a_channel_with_a_missing_node_is_counted_not_refused() -> None:
    # KHR_animation_pointer targets have a pointer and no node.
    def edit(doc: dict) -> None:
        del doc["animations"][0]["channels"][0]["target"]["node"]

    assert gltf.load(_edited(edit)).skipped_animation_channels == 1


def test_a_normalised_integer_output_is_read_as_floats() -> None:
    data = glbwrite.write_glb(
        _model(
            [
                gltf.Animation(
                    "Q", [gltf.AnimChannel(1, "rotation", np.array([0.0]), np.array([Q0]))]
                )
            ]
        )
    )
    doc, binary = read_glb(data)
    output = doc["animations"][0]["samplers"][0]["output"]
    accessor = doc["accessors"][output]
    view = doc["bufferViews"][accessor["bufferView"]]
    ints = np.array([[0, 0, 0, 32767]], dtype="<i2").tobytes()
    patched = bytearray(binary)
    patched[view["byteOffset"] : view["byteOffset"] + len(ints)] = ints
    accessor.update(componentType=5122, normalized=True)
    view["byteLength"] = len(ints)
    chunk = len(patched).to_bytes(4, "little") + b"BIN\x00" + bytes(patched)
    out = gltf.load(rebuild_glb(data[:12], doc, chunk))
    assert np.allclose(out.animations[0].channels[0].values, [[0, 0, 0, 1]])
    # And an integer output that is not marked normalised is refused.
    accessor["normalized"] = False
    with pytest.raises(ValueError, match="without normalising"):
        gltf.load(rebuild_glb(data[:12], doc, chunk))


def test_the_declared_counts_are_capped_before_anything_is_decoded(monkeypatch) -> None:
    monkeypatch.setattr(gltf, "MAX_ANIMATIONS", 1)
    with pytest.raises(ValueError, match="2 animations"):
        gltf.load(glbwrite.write_glb(_model([_walk(), _walk()])))
    monkeypatch.setattr(gltf, "MAX_ANIMATIONS", 100)
    monkeypatch.setattr(gltf, "MAX_ANIMATION_CHANNELS", 2)
    with pytest.raises(ValueError, match="3 animation channels"):
        gltf.load(glbwrite.write_glb(_model([_walk()])))


def test_the_json_the_tests_edit_is_what_the_writer_emitted() -> None:
    doc, _ = read_glb(glbwrite.write_glb(_model([_walk()])))
    assert json.dumps(doc["animations"][0]["channels"][0]["target"]) == (
        '{"node": 1, "path": "rotation"}'
    )
