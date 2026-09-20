"""Clay's material library (``studio/modes/clay/matlib.py``, tranche 6).

Pure and imgui-free by the module's own rule, so every test here drives it
directly against a sandboxed ``tmp_path`` standing in for ``REALMSPINNER_HOME`` --
never the real one (``dev/audits/audit-2026-09-13.md``'s own lesson, restated
in this session's brief).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.studio.modes.clay import matlib


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


def _orphaned_pngs(folder) -> set[str]:
    """A texture side car under *folder* whose manifest is not there to name
    it -- what nothing in this module ever lists or sweeps."""
    if not folder.is_dir():
        return set()
    manifests = {p.stem for p in folder.glob("*.json")}
    orphans: set[str] = set()
    for png in folder.glob("*.png"):
        stem = png.stem
        for slot in matlib.TEXTURE_SLOTS:
            suffix = f"_{slot}"
            if stem.endswith(suffix):
                entry_id = stem[: -len(suffix)]
                break
        else:
            entry_id = stem
        if entry_id not in manifests:
            orphans.add(png.name)
    return orphans


def test_an_interrupted_save_or_delete_leaves_no_orphaned_texture_side_cars(
    tmp_path, monkeypatch
) -> None:
    """The 2026-09-19 audit, finding clay-32. ``save_material`` used to write
    every texture PNG *before* the manifest that names them, and
    ``delete_material`` unlinked the manifest *before* its PNGs -- an
    interruption between either pair left PNGs under
    ``REALMSPINNER_HOME/clay/materials`` that nothing lists or sweeps
    (``list_materials`` only ever walks ``*.json``). The fix writes the
    manifest first on save and last on delete, so a crash in the middle
    always leaves either no manifest and no textures, or a manifest whose
    texture slots ``load_material`` already tolerates missing.
    """
    folder = matlib.library_dir(tmp_path)

    # --- an interrupted save: the manifest lands, then the crash hits partway
    # through the textures -- one PNG makes it to disk, the rest do not.
    real_write_png = matlib._write_png
    calls = {"n": 0}

    def _write_png_crash_after_first(path, image):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_write_png(path, image)
        raise RuntimeError("simulated crash mid-texture-write")

    material = gltf.Material(name="Rusty", base_color=_texture(), normal=_texture())
    with monkeypatch.context() as m:
        m.setattr(matlib, "_write_png", _write_png_crash_after_first)
        with pytest.raises(RuntimeError):
            matlib.save_material(tmp_path, "Rusty", material)

    assert calls["n"] == 2, "the crash must land mid-write, not before or after every texture"
    assert _orphaned_pngs(folder) == set(), (
        "a save interrupted after the manifest write must not orphan the PNG that did land"
    )
    entries = matlib.list_materials(tmp_path)
    assert [e.name for e in entries] == ["Rusty"], "the manifest itself landed intact"
    loaded = matlib.load_material(tmp_path, entries[0].id)
    assert loaded is not None, "a half-written material must still load, degraded"

    matlib.delete_material(tmp_path, entries[0].id)  # a clean delete, for the next half

    # --- an interrupted delete: both textures are unlinked, then the crash
    # hits before the manifest itself is.
    entry = matlib.save_material(
        tmp_path, "Glass", gltf.Material(name="Glass", base_color=_texture(), normal=_texture())
    )
    real_unlink = Path.unlink

    def _unlink_crash_on_manifest(self, *args, **kwargs):
        if self.suffix == ".json":
            raise RuntimeError("simulated crash before the manifest unlink")
        return real_unlink(self, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", _unlink_crash_on_manifest)
        with pytest.raises(RuntimeError):
            matlib.delete_material(tmp_path, entry.id)

    assert _orphaned_pngs(folder) == set(), (
        "a delete interrupted before the manifest unlink must not orphan a PNG -- "
        "the textures are already gone by then"
    )


def test_library_dir_is_under_the_given_home_not_the_real_one(tmp_path) -> None:
    folder = matlib.library_dir(tmp_path)
    assert str(folder).startswith(str(tmp_path))
    assert folder.parts[-2:] == ("clay", "materials")
