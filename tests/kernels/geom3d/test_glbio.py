"""GLB container helpers in ``glbio.py`` beyond split/rebuild.

Fixtures here are built by hand with ``json``/``struct`` rather than through
``studio.viewer.glbwrite`` -- these helpers operate on the JSON chunk alone,
so a fixture only needs to be a well-formed GLB, not a renderable model.
"""

from __future__ import annotations

import json
import struct

import pytest

from warlock.kernels.geom3d.glbio import (
    CHUNK_BIN,
    CHUNK_JSON,
    GLB_MAGIC,
    animation_names,
    rename_animations,
    root_extras,
    set_root_extras,
)


def _make_glb(gltf: dict, bin_payload: bytes = b"") -> bytes:
    """A minimal, spec-padded GLB carrying ``gltf`` as its JSON chunk and, if
    given, a BIN chunk -- built independently of ``rebuild_glb`` so a bug
    shared between the fixture and the code under test can't hide."""
    json_payload = json.dumps(gltf, separators=(",", ":")).encode()
    json_payload += b" " * (-len(json_payload) % 4)
    chunks = struct.pack("<II", len(json_payload), CHUNK_JSON) + json_payload
    if bin_payload:
        padded = bin_payload + b"\x00" * (-len(bin_payload) % 4)
        chunks += struct.pack("<II", len(padded), CHUNK_BIN) + padded
    total = 12 + len(chunks)
    header = struct.pack("<III", GLB_MAGIC, 2, total)
    return header + chunks


def _animated_gltf(names: list[str]) -> dict:
    return {
        "asset": {"version": "2.0"},
        "animations": [{"name": n, "channels": [], "samplers": []} for n in names],
    }


# --- animations --------------------------------------------------------------


def test_renaming_animations_keeps_the_binary_chunk_byte_identical() -> None:
    bin_payload = bytes(range(37))  # length not a multiple of 4: exercises padding
    data = _make_glb(_animated_gltf(["walk", "idle", "attack"]), bin_payload)
    before = data[data.index(bin_payload) :]  # the BIN chunk's bytes, padding included

    out = rename_animations(data, {"walk": "walk-loop", "idle": "idle-loop"})

    assert out[out.index(bin_payload) :] == before
    assert animation_names(out) == ["walk-loop", "idle-loop", "attack"]


def test_renaming_an_animation_the_file_lacks_is_refused() -> None:
    data = _make_glb(_animated_gltf(["walk", "idle"]))
    with pytest.raises(ValueError, match="no animation named"):
        rename_animations(data, {"run": "run-loop"})


def test_a_rename_that_collides_two_animations_is_refused() -> None:
    data = _make_glb(_animated_gltf(["walk", "idle"]))
    with pytest.raises(ValueError, match="two animations named"):
        rename_animations(data, {"walk": "idle"})


def test_a_rename_to_an_empty_name_is_refused() -> None:
    data = _make_glb(_animated_gltf(["walk", "idle"]))
    with pytest.raises(ValueError, match="must not be empty"):
        rename_animations(data, {"walk": ""})


def test_an_animation_not_named_in_the_mapping_is_untouched() -> None:
    data = _make_glb(_animated_gltf(["walk", "idle", "attack"]))
    out = rename_animations(data, {"walk": "walk-loop"})
    assert animation_names(out) == ["walk-loop", "idle", "attack"]


def test_a_swap_between_two_mapped_names_is_not_a_collision() -> None:
    data = _make_glb(_animated_gltf(["walk", "idle"]))
    out = rename_animations(data, {"walk": "idle", "idle": "walk"})
    assert animation_names(out) == ["idle", "walk"]


def test_animation_names_reports_file_order_and_empty_for_unnamed() -> None:
    gltf = {"animations": [{"name": "walk"}, {}]}
    data = _make_glb(gltf)
    assert animation_names(data) == ["walk", ""]


def test_animation_names_on_a_file_with_no_animations_is_empty() -> None:
    data = _make_glb({"asset": {"version": "2.0"}})
    assert animation_names(data) == []


# --- root extras ---------------------------------------------------------


def test_root_extras_round_trip_and_keep_other_keys() -> None:
    gltf = {"asset": {"version": "2.0"}, "extras": {"kept": "as-is", "num": 1}}
    data = _make_glb(gltf)

    out = set_root_extras(data, "digest", "abc123")

    extras = root_extras(out)
    assert extras["digest"] == "abc123"
    assert extras["kept"] == "as-is"
    assert extras["num"] == 1


def test_root_extras_creates_the_extras_object_when_absent() -> None:
    data = _make_glb({"asset": {"version": "2.0"}})
    assert root_extras(data) == {}

    out = set_root_extras(data, "digest", "abc123")
    assert root_extras(out) == {"digest": "abc123"}


def test_setting_root_extras_keeps_the_binary_chunk_byte_identical() -> None:
    bin_payload = bytes(range(53))
    data = _make_glb({"asset": {"version": "2.0"}}, bin_payload)
    before = data[data.index(bin_payload) :]  # the BIN chunk's bytes, padding included

    out = set_root_extras(data, "digest", {"clips": ["walk", "idle"]})

    assert out[out.index(bin_payload) :] == before
    assert root_extras(out) == {"digest": {"clips": ["walk", "idle"]}}


def test_setting_root_extras_with_an_empty_key_is_refused() -> None:
    data = _make_glb({"asset": {"version": "2.0"}})
    with pytest.raises(ValueError, match="non-empty string"):
        set_root_extras(data, "", "x")


def test_setting_root_extras_with_a_non_serialisable_value_is_refused() -> None:
    data = _make_glb({"asset": {"version": "2.0"}})
    with pytest.raises(ValueError, match="JSON-serialisable"):
        set_root_extras(data, "digest", object())


# --- container validity -------------------------------------------------


def test_the_rebuilt_glb_is_still_a_valid_container() -> None:
    bin_payload = bytes(range(11))  # deliberately not a multiple of 4
    data = _make_glb(_animated_gltf(["walk", "idle"]), bin_payload)
    data = set_root_extras(data, "digest", "abc123")
    data = rename_animations(data, {"walk": "walk-loop"})

    magic, version, total_length = struct.unpack_from("<III", data, 0)
    assert magic == GLB_MAGIC
    assert version == 2
    assert total_length == len(data)

    offset = 12
    seen_kinds = []
    while offset < len(data):
        chunk_len, chunk_kind = struct.unpack_from("<II", data, offset)
        seen_kinds.append(chunk_kind)
        assert chunk_len % 4 == 0
        assert offset + 8 + chunk_len <= len(data)
        offset += 8 + chunk_len
    assert offset == len(data)
    assert seen_kinds == [CHUNK_JSON, CHUNK_BIN]
