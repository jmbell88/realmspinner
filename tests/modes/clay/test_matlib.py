"""Clay's material library (``studio/modes/clay/matlib.py``, tranche 6).

Pure and imgui-free by the module's own rule, so every test here drives it
directly against a sandboxed ``tmp_path`` standing in for ``WARLOCK_HOME`` --
never the real one (``dev/audits/audit-2026-09-13.md``'s own lesson, restated
in this session's brief).
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.kernels.geom3d import gltf
from warlock.studio.modes.clay import matlib


def _texture(width: int = 2, height: int = 2, value: int = 200) -> tuple[int, int, bytes]:
    data = np.full((height, width, 4), value, dtype=np.uint8)
    return (width, height, data.tobytes())


def test_save_then_list_finds_it_by_name(tmp_path) -> None:
    material = gltf.Material(name="Untitled", base_color_factor=(0.2, 0.4, 0.6, 1.0))
    entry = matlib.save_material(tmp_path, "Rusty Metal", material)

    entries = matlib.list_materials(tmp_path)
    assert [e.name for e in entries] == ["Rusty Metal"]
    assert entries[0].id == entry.id
    assert entries[0].base_color_factor == (0.2, 0.4, 0.6, 1.0)
    assert entries[0].texture_slots == ()


def test_list_is_empty_before_anything_is_saved(tmp_path) -> None:
    assert matlib.list_materials(tmp_path) == []


def test_two_materials_may_share_a_name(tmp_path) -> None:
    """``ClayDoc.add_material`` places no uniqueness rule on a material's
    name; the shelf follows the same rule -- "Save current as..." is always
    an add, never a silent overwrite of an existing entry."""
    a = matlib.save_material(tmp_path, "Wood", gltf.Material(name="Wood"))
    b = matlib.save_material(tmp_path, "Wood", gltf.Material(name="Wood"))
    assert a.id != b.id
    assert len(matlib.list_materials(tmp_path)) == 2


def test_save_writes_texture_side_cars_and_load_round_trips_them(tmp_path) -> None:
    base_color = _texture(3, 2, value=10)
    normal = _texture(2, 2, value=250)
    material = gltf.Material(
        name="Painted Wood",
        base_color_factor=(1.0, 1.0, 1.0, 1.0),
        metallic_factor=0.1,
        roughness_factor=0.7,
        emissive_factor=(0.2, 0.0, 0.0),
        double_sided=True,
        alpha_mode="MASK",
        alpha_cutoff=0.3,
        base_color=base_color,
        normal=normal,
    )
    entry = matlib.save_material(tmp_path, "Painted Wood", material)
    assert set(entry.texture_slots) == {"base_color", "normal"}

    # The side cars are real PNG files under the library folder -- not, say,
    # base64 folded into the JSON manifest (see the module's own docstring
    # for why a texture is a sibling file, not inline).
    png_files = list(matlib.library_dir(tmp_path).glob(f"{entry.id}_*.png"))
    assert len(png_files) == 2

    loaded = matlib.load_material(tmp_path, entry.id)
    assert loaded is not None
    assert loaded.name == "Painted Wood"
    assert loaded.metallic_factor == pytest.approx(0.1)
    assert loaded.roughness_factor == pytest.approx(0.7)
    assert loaded.emissive_factor == pytest.approx((0.2, 0.0, 0.0))
    assert loaded.double_sided is True
    assert loaded.alpha_mode == "MASK"
    assert loaded.alpha_cutoff == pytest.approx(0.3)
    assert loaded.metallic_roughness is None
    assert loaded.base_color is not None
    assert loaded.base_color[0] == 3 and loaded.base_color[1] == 2
    assert loaded.base_color[2] == base_color[2]
    assert loaded.normal is not None
    assert loaded.normal[2] == normal[2]


def test_delete_removes_the_entry_and_its_textures(tmp_path) -> None:
    material = gltf.Material(name="Glass", base_color=_texture())
    entry = matlib.save_material(tmp_path, "Glass", material)
    folder = matlib.library_dir(tmp_path)
    assert list(folder.glob(f"{entry.id}*")), "the entry actually wrote something"

    assert matlib.delete_material(tmp_path, entry.id) is True
    assert matlib.list_materials(tmp_path) == []
    assert list(folder.glob(f"{entry.id}*")) == [], "no orphaned side car left behind"
    # Deleting again is a clean "nothing was there", not an error.
    assert matlib.delete_material(tmp_path, entry.id) is False


def test_load_of_an_unknown_id_is_none_not_a_raise(tmp_path) -> None:
    assert matlib.load_material(tmp_path, "does-not-exist") is None


def test_list_skips_a_corrupt_entry_and_keeps_the_rest(tmp_path) -> None:
    """A corrupt or half-written entry must not break the pane -- the brief's
    own words. One bad JSON file among several good ones is a skipped row,
    never a raised exception the properties panel would have to catch."""
    matlib.save_material(tmp_path, "Good One", gltf.Material(name="Good One"))
    matlib.save_material(tmp_path, "Good Two", gltf.Material(name="Good Two"))
    folder = matlib.library_dir(tmp_path)
    (folder / "half-written.json").write_text("{not valid json", encoding="utf-8")
    (folder / "wrong-shape.json").write_text('{"base_color_factor": [1, 2]}', encoding="utf-8")

    entries = matlib.list_materials(tmp_path)
    assert {e.name for e in entries} == {"Good One", "Good Two"}


def test_load_degrades_a_missing_or_corrupt_texture_slot_rather_than_failing(tmp_path) -> None:
    material = gltf.Material(name="Tiles", base_color=_texture(), normal=_texture())
    entry = matlib.save_material(tmp_path, "Tiles", material)
    # Corrupt one side car by hand, as a crash mid-write would leave it.
    normal_path = matlib.library_dir(tmp_path) / f"{entry.id}_normal.png"
    normal_path.write_bytes(b"not a png")

    loaded = matlib.load_material(tmp_path, entry.id)
    assert loaded is not None, "a bad texture must not sink the whole material"
    assert loaded.name == "Tiles"
    assert loaded.base_color is not None, "the other slot is unaffected"
    assert loaded.normal is None, "the corrupt slot degrades to empty"


def test_load_tolerates_a_manifest_naming_a_texture_file_that_is_gone(tmp_path) -> None:
    material = gltf.Material(name="Brick", base_color=_texture())
    entry = matlib.save_material(tmp_path, "Brick", material)
    (matlib.library_dir(tmp_path) / f"{entry.id}_base_color.png").unlink()

    loaded = matlib.load_material(tmp_path, entry.id)
    assert loaded is not None
    assert loaded.base_color is None


def test_library_dir_is_under_the_given_home_not_the_real_one(tmp_path) -> None:
    folder = matlib.library_dir(tmp_path)
    assert str(folder).startswith(str(tmp_path))
    assert folder.parts[-2:] == ("clay", "materials")
