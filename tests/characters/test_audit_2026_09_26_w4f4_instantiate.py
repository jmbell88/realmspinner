"""poser-characters-07 (2026-09-26 audit): a bare mask-npz index and a fixed
staging name in ``instantiate.py``.

``_load_base``/``instantiate`` indexed a species' ``.masks.npz`` with a bare
``arrays[key]`` for every structural channel (``positions_digest``, ``joints``,
``joint_names``, ``joint_parents``, ``prim_offsets``, ``prim_regions``) --
unlike the per-appearance ``disp/``/``jdisp/`` channels a few lines below,
which already guard the identical lookup and raise ``CharacterError``. A mask
file missing one of these raised a raw ``KeyError`` with no ``field``,
invisible to the ``except ValueError`` doors ``CharacterError`` exists to be
caught by. Separately, ``_write`` staged every artifact through one fixed
``.<name>.tmp`` name regardless of caller, so two concurrent writers onto the
same served path shared one staging file instead of each getting their own.
"""

from __future__ import annotations

import pytest

from realmspinner.characters import DEFAULT_RECIPE, CharacterError
from realmspinner.characters import instantiate as instantiate_mod


def test_a_masks_npz_missing_a_structural_channel_refuses_by_name_not_a_bare_keyerror(
    tmp_path, monkeypatch
):
    real_load_base = instantiate_mod._load_base

    def _stripped(fam):
        prims, positions, arrays = real_load_base(fam)
        arrays = dict(arrays)
        del arrays["joint_names"]
        return prims, positions, arrays

    monkeypatch.setattr(instantiate_mod, "_load_base", _stripped)

    with pytest.raises(CharacterError) as excinfo:
        instantiate_mod.instantiate(DEFAULT_RECIPE, tmp_path)
    assert excinfo.value.field == "family"


def test_required_refuses_a_missing_channel_by_name_rather_than_a_bare_keyerror():
    from realmspinner.characters.family import get_family

    fam = get_family(DEFAULT_RECIPE.family)
    with pytest.raises(CharacterError) as excinfo:
        instantiate_mod._required({}, "prim_offsets", fam)
    assert excinfo.value.field == "family"
    assert "prim_offsets" in str(excinfo.value)


def test_write_stages_a_temp_name_unique_per_call_not_shared_across_writers(tmp_path):
    """Two calls onto the same served path must not reuse one staging file --
    a fixed ``.<name>.tmp`` let one call's in-flight ``write_bytes`` be
    clobbered by another's before either reached ``os.replace``."""
    seen: list[str] = []
    path = tmp_path / "model.glb"
    real_write_bytes = type(path).write_bytes

    def _spy(self, data):
        seen.append(self.name)
        return real_write_bytes(self, data)

    import pathlib

    original = pathlib.Path.write_bytes
    pathlib.Path.write_bytes = _spy
    try:
        instantiate_mod._write(path, b"AAAA")
        instantiate_mod._write(path, b"BBBB")
    finally:
        pathlib.Path.write_bytes = original

    assert len(seen) == 2
    assert seen[0] != seen[1], (
        "two calls to _write onto the same served path used the same "
        "staging filename -- a concurrent pair could clobber each other"
    )
    assert path.read_bytes() == b"BBBB"
