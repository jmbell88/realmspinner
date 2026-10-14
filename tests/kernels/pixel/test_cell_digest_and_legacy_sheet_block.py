"""What survived the character-sheet correction loop's removal.

Two things outlived it, and each is a claim about stored data rather than about
a feature:

* ``cell_digest`` moved from ``sheetmerge`` into ``_doc_flourish``. Every
  ``.ora`` an effect was saved into carries digests of this exact function, so
  a different hash would read every saved cel as painted over. The golden
  values below were computed with the function as it stood in ``sheetmerge``.
* An ``.ora`` written before the removal may carry a ``sheet`` block in
  ``animation.json`` (the recorded render a re-rendered sheet was merged
  against). Nothing reads it any more; it has to open, and a re-save drops it.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from realmspinner.kernels.pixel import ora
from realmspinner.kernels.pixel._doc_flourish import cell_digest
from realmspinner.kernels.pixel.sheetin import document_from_grid

CELL = 8
FRAMES = 4


def _cell(value: int = 0, size: int = 4) -> np.ndarray:
    cell = np.zeros((size, size, 4), dtype=np.uint8)
    cell[..., 3] = 255
    cell[..., 0] = value
    return cell


# -- the digest ---------------------------------------------------------------


def test_cell_digest_values_are_the_ones_stored_files_were_written_with():
    ramp = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    assert cell_digest(_cell(7)) == "1b6a7851c6a4af368ac8f9e651a9fcb1"
    assert cell_digest(ramp) == "bdc339938ec5e6c24fa5d53bbf2c5734"


def test_the_same_pixels_digest_the_same_and_different_ones_do_not():
    assert cell_digest(_cell(7)) == cell_digest(_cell(7))
    assert cell_digest(_cell(7)) != cell_digest(_cell(8))


def test_a_view_digests_as_its_own_contents_rather_than_its_base():
    """A cel's plane can be a view after a geometry op, and a view's ``tobytes``
    is its *base's* buffer -- so without the contiguity copy a cropped cell
    would digest as the thing it was cropped out of."""
    whole = _cell(3, size=8)
    view = whole[2:6, 2:6]
    assert not view.flags["C_CONTIGUOUS"]
    assert cell_digest(view) == cell_digest(np.array(view))
    assert cell_digest(view) != cell_digest(whole)


def test_the_shape_is_part_of_the_digest():
    """A canvas resized between two renders must read as changed, not as a
    coincidence. Same bytes, different shape, different answer."""
    flat = np.zeros((4, 8, 4), dtype=np.uint8)
    tall = np.zeros((8, 4, 4), dtype=np.uint8)
    assert flat.tobytes() == tall.tobytes()
    assert cell_digest(flat) != cell_digest(tall)


# -- an .ora that still carries the retired ``sheet`` block ---------------------


def _animated_ora(tmp_path: Path) -> Path:
    atlas = np.zeros((CELL, FRAMES * CELL, 4), dtype=np.uint8)
    atlas[..., 3] = 255
    for i in range(FRAMES):
        atlas[:, i * CELL : (i + 1) * CELL, 0] = 30 + i * 40
    doc = document_from_grid(atlas, (CELL, CELL))
    path = tmp_path / "plain.ora"
    ora.write_ora(doc, path)
    return path


def _animation_member(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return next(n for n in archive.namelist() if n.endswith("animation.json"))


def _with_sheet_block(source: Path, target: Path, block: object) -> Path:
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    key = _animation_member(source)
    payload = json.loads(members[key])
    payload["sheet"] = block
    members[key] = json.dumps(payload, indent=2).encode("utf-8")
    with zipfile.ZipFile(target, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return target


def _legacy_block() -> dict:
    return {
        "version": 1,
        "algorithm": "blake2b-16",
        "source": {"job": "abc", "sheet": "def"},
        "cells": [{"frame": i, "digest": "0" * 32} for i in range(FRAMES)],
        "conflicts": [2],
    }


def test_an_ora_carrying_a_sheet_base_block_still_opens_and_drops_it_on_resave(tmp_path):
    plain = _animated_ora(tmp_path)
    legacy = _with_sheet_block(plain, tmp_path / "legacy.ora", _legacy_block())
    assert b'"sheet"' in zipfile.ZipFile(legacy).read(_animation_member(legacy))

    back = ora.read_ora(legacy)
    assert len(back.anim.frames) == FRAMES, "the timeline must survive intact"
    assert not hasattr(back, "sheet_base"), "nothing holds the retired record any more"

    resaved = tmp_path / "resaved.ora"
    ora.write_ora(back, resaved)
    written = zipfile.ZipFile(resaved).read(_animation_member(resaved))
    assert b'"sheet"' not in written
    assert len(ora.read_ora(resaved).anim.frames) == FRAMES


def test_a_malformed_sheet_block_cannot_cost_the_document_its_timeline(tmp_path):
    plain = _animated_ora(tmp_path)
    for index, block in enumerate(
        ("not a mapping", {"cells": "nope"}, {"algorithm": "from-the-future", "cells": []}, None)
    ):
        broken = _with_sheet_block(plain, tmp_path / f"broken{index}.ora", block)
        assert len(ora.read_ora(broken).anim.frames) == FRAMES
