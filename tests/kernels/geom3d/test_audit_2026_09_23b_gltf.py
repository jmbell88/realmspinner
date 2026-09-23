"""Regression tests for the 2026-09-23 audit (second run)'s clay-13 and
clay-14, both in ``kernels/geom3d/gltf.py``.
"""

from __future__ import annotations

import struct

import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d.glbio import rebuild_glb


def _glb(gltf_json: dict, binary: bytes = b"") -> bytes:
    """Wrap a JSON chunk and a BIN chunk as a GLB, the way an exporter would."""
    binary += b"\x00" * (-len(binary) % 4)
    chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    header = struct.pack("<III", 0x46546C67, 2, 0)
    return rebuild_glb(header, gltf_json, chunk)


# --- clay-13: len() on a non-list array before checking it is one -----------


def test_load_refuses_a_non_list_nodes_array_with_a_named_value_error() -> None:
    """``load()`` called ``len(gltf.get("nodes", []))`` with no check that a
    *present* ``"nodes"`` was itself a list -- ``.get(key, [])``'s default
    only ever covers the key being absent. A GLB with ``"nodes": {}`` (legal
    JSON, wrong shape) reached ``len()`` as a bare, un-messaged ``TypeError``
    instead of the named ``ValueError`` every other malformed shape in this
    loader raises.
    """
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}], "nodes": {}}
    with pytest.raises(ValueError, match="\"nodes\""):
        gltf.load(_glb(doc))


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("materials", "not-a-list"),
        ("meshes", 5),
        ("cameras", {"oops": True}),
    ],
)
def test_load_refuses_other_non_list_top_level_arrays_with_a_named_value_error(
    field, bad_value
) -> None:
    """Same clay-13 gap, the other four call sites the finding names
    (``"materials"``, ``"meshes"``, ``"cameras"``, and a mesh's own
    ``"primitives"`` -- covered separately below since it needs a valid
    ``"meshes"`` array to reach)."""
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        field: bad_value,
    }
    with pytest.raises(ValueError, match=f'"{field}"'):
        gltf.load(_glb(doc))


def test_load_refuses_a_non_list_primitives_array_on_an_otherwise_valid_mesh_entry() -> None:
    """The fifth clay-13 call site: ``declared_primitives += len(mesh.get
    ("primitives") or [])`` inside the per-mesh loop, once ``mesh`` itself
    is a valid dict (past ``_check_dict_entry``, clay-04's own fix) but its
    ``"primitives"`` field is not a list."""
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        "meshes": [{"primitives": "not-a-list"}],
    }
    with pytest.raises(ValueError, match='"primitives"'):
        gltf.load(_glb(doc))


# --- clay-14: scenes[index]["nodes"] indexed unguarded -----------------------


def test_roots_refuses_a_malformed_scene_with_a_named_value_error() -> None:
    """``_roots`` indexed ``scenes[index]`` and then ``scenes[index]
    ["nodes"]`` with no shape check on either -- a scene entry that is not a
    JSON object reached the ``"nodes" in scenes[index]`` membership test as
    a bare exception instead of this loader's own named refusal."""
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": ["not-a-scene-object"],
        "nodes": [],
    }
    with pytest.raises(ValueError, match="scene"):
        gltf.load(_glb(doc))


def test_roots_refuses_a_scene_whose_nodes_field_is_not_a_list() -> None:
    """The other half of clay-14: a scene that *is* an object but whose own
    ``"nodes"`` is not a list -- ``for root in roots`` would otherwise
    either misread it (a string, a dict) or raise a bare exception (a
    number) instead of this loader's named refusal."""
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": "not-a-list"}],
        "nodes": [],
    }
    with pytest.raises(ValueError, match='"nodes"'):
        gltf.load(_glb(doc))


def test_roots_still_falls_back_to_unparented_nodes_when_the_chosen_scene_omits_nodes() -> None:
    """Sanity: a scene object with no ``"nodes"`` key at all is not
    malformed -- glTF allows an empty scene -- and must still reach the
    unparented-node fallback exactly as before this fix."""
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{}],
        "nodes": [{}],
    }
    model = gltf.load(_glb(doc))
    assert model.roots == [0]
