"""Regressions for the 2026-10-14 audit's Inker codec, sheet and walk findings
(inker-04, 05, 07, 33, 35, 44, 45, 69, 73, 81).

Each test's name is its claim. The shape most of them share is the one the
2026-10-03 codec audit stated: a document this build can save is one it can
reopen, and a file it opens costs memory in proportion to its bytes rather than
to what it names.
"""

from __future__ import annotations

import io
import json
import re
import tracemalloc
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.core.safeio import pixelguard
from realmspinner.kernels.pixel import ora, sheetout
from realmspinner.kernels.pixel.document import Document
from realmspinner.kernels.pixel.walk import rig as R


def _png(width: int, height: int, colour=(9, 9, 9, 255)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (width, height), colour).save(buf, "PNG")
    return buf.getvalue()


def _animated(
    frames: int, tracks: int = 1, *, size: int = 8, fill: bool = True
) -> Document:
    """``tracks`` rows held across ``frames`` frames: one distinct cel per row,
    so ``tracks * frames`` cel slots over only ``tracks`` planes."""
    from realmspinner.kernels.pixel.layers import Layer

    doc = Document.blank(size, size)
    doc.ensure_animation()
    for _ in range(tracks - 1):
        doc.add_layer()
    for _ in range(frames - 1):
        doc.add_frame(link=True)
    anim = doc.anim
    for track in anim.tracks if fill else ():
        held = Layer.empty(size, size, track.name)
        for frame in anim.frames:
            anim.cels[(track.uid, frame.uid)] = held
    return doc


# -- inker-04: the flat reader's memo of full-size decodes ----------------------


def test_a_flat_ora_of_many_distinct_oversized_layer_pngs_does_not_retain_each_decode(
    monkeypatch, tmp_path
):
    # inker-codecs-03: ``decoded_members`` kept the whole decoded plane of every
    # distinct member for the loop's duration though each layer is cropped to the
    # canvas, and nothing summed them: 6 distinct 4096x4096 PNGs in a 1x1 canvas
    # peaked at 448 MiB. Scaled down here by lowering the decode cap the memo is
    # now charged against: 40 distinct 1024x1024 members, room for two.
    monkeypatch.setattr(pixelguard, "MAX_DECODE_PIXELS", 2 * 1024 * 1024)
    count = 40
    layers = "".join(f'<layer src="data/l{i}.png" name="L{i}"/>' for i in range(count))
    path = tmp_path / "many.ora"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", b"image/openraster")
        zf.writestr("stack.xml", f'<image w="1" h="1"><stack>{layers}</stack></image>')
        for i in range(count):
            zf.writestr(f"data/l{i}.png", _png(1024, 1024, (i, i, i, 255)))
    tracemalloc.start()
    try:
        doc = ora.read_ora(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(doc.stack) == count
    # Retaining every decode is 40 x 4 MiB = 160 MiB; holding two and one in
    # flight is a fraction of that.
    assert peak < 48 * 1024 * 1024, f"peak {peak / 2**20:.0f} MiB"


# -- inker-05: write_ora checks every count the reader refuses ------------------


def test_write_ora_refuses_an_animation_whose_frames_or_cel_slots_the_reader_will_not_reopen(
    monkeypatch,
):
    # inker-codecs-04: five tracks of one held cel each over 1,000 frames is
    # 5,000 cel slots over five distinct cels. It passed every write check and
    # ``_read_animation`` then refused the cel list, so the timeline fell back to
    # a flat read that refused it again.
    monkeypatch.setattr(ora, "MAX_ORA_METADATA_ENTRIES", 20)
    doc = _animated(6, tracks=4)  # 24 cel slots, 4 distinct cels
    assert len(list(doc.anim.unique_cel_layers())) == 4
    with pytest.raises(ValueError, match="cel"):
        ora.ora_bytes(doc)

    monkeypatch.setattr(ora, "MAX_ORA_METADATA_ENTRIES", 1 << 20)
    monkeypatch.setattr(ora, "MAX_ORA_FRAMES", 5)
    with pytest.raises(ValueError, match="frames"):
        ora.ora_bytes(_animated(6))
    # The ceiling itself still saves.
    monkeypatch.setattr(ora, "MAX_ORA_FRAMES", 6)
    assert ora.ora_bytes(_animated(6))


def test_write_ora_refuses_tags_slices_and_group_nodes_the_reader_will_not_reopen(
    monkeypatch,
):
    monkeypatch.setattr(ora, "MAX_ORA_METADATA_ENTRIES", 2)
    doc = _animated(2)
    for i in range(3):
        doc.add_slice((0, 0, 2, 2), name=f"s{i}")
    with pytest.raises(ValueError, match="slices"):
        ora.ora_bytes(doc)

    doc = _animated(2)
    for start, end in ((0, 0), (1, 1), (0, 1)):
        assert doc.add_tag(f"t{start}{end}", start, end)
    with pytest.raises(ValueError, match="tags"):
        ora.ora_bytes(doc)

    doc = _animated(1, tracks=4, fill=False)
    doc.group_layers([0, 1], "A")
    doc.group_layers([2, 3], "B")
    doc.group_layers([0, 1, 2, 3], "C")
    with pytest.raises(ValueError, match="group"):
        ora.ora_bytes(doc)


# -- inker-07: a JSON member is bounded before it is parsed ---------------------


def test_an_oversized_animation_json_member_is_refused_before_it_is_parsed(
    monkeypatch, tmp_path
):
    # inker-codecs-06: ``json.loads`` ran on the whole member and only then were
    # the count ceilings applied -- 8M empty frame objects peaked at 601 MiB and
    # 4.9 s from a 24 KB archive, then were refused by the frame count.
    monkeypatch.setattr(ora, "MAX_ORA_JSON_BYTES", 1024)
    body = json.dumps({"version": 1, "frames": [{}] * 400, "tracks": [], "cels": []})
    assert len(body) > 1024
    path = tmp_path / "huge.ora"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", b"image/openraster")
        zf.writestr(
            "stack.xml", '<image w="2" h="2"><stack><layer src="data/a.png"/></stack></image>'
        )
        zf.writestr("data/a.png", _png(2, 2))
        zf.writestr("animation.json", body)
    parsed: list[int] = []
    real = json.loads

    def spy(raw, *a, **k):
        parsed.append(len(raw))
        return real(raw, *a, **k)

    monkeypatch.setattr(ora.json, "loads", spy)
    doc = ora.read_ora(path)
    assert not any(n >= 1024 for n in parsed), "the oversized member was parsed"
    # It degrades like every other bad animation.json: a flat, one-frame read.
    assert doc.anim is None
    assert len(doc.stack) == 1


# -- inker-45: imported slices are fitted ---------------------------------------


def test_an_imported_slice_is_clamped_into_the_canvas_and_its_centre_into_its_bounds(
    tmp_path,
):
    # inker-sheets-02: a 16x12 canvas with an ``.ora`` slice (-4,-4,36,36) and
    # centre (30,30,35,35) read back exactly like that, and ``Export slices``
    # then cropped ``flat[-4:36, -4:36]``, which numpy wraps.
    path = tmp_path / "slices.ora"
    payload = {
        "version": 1,
        "slices": [
            {
                "name": "panel",
                "bounds": {"x": -4, "y": -4, "w": 40, "h": 40},
                "center": {"x": 30, "y": 30, "w": 5, "h": 5},
                "keys": [],
            }
        ],
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", b"image/openraster")
        zf.writestr(
            "stack.xml", '<image w="16" h="12"><stack><layer src="data/a.png"/></stack></image>'
        )
        zf.writestr("data/a.png", _png(16, 12))
        zf.writestr("realmspinner.json", json.dumps(payload))
    doc = ora.read_ora(path)
    (entry,) = doc.slices
    assert entry.bounds == (0, 0, 16, 12)
    cx0, cy0, cx1, cy1 = entry.center
    assert 0 <= cx0 < cx1 <= entry.width
    assert 0 <= cy0 < cy1 <= entry.height


# -- inker-44: trim keeps the nine-slice centre inside the clipped slice --------


def test_trim_keeps_a_slice_centre_inside_the_clipped_slice_bounds():
    # inker-sheets-01: a 10x10 canvas with art in 3..7 and a slice (0,0,10,10)
    # with centre (1,1,8,8) exported bounds {0,0,5,5} and centre {-2,-2,8,8}.
    trims = {0: {"x": 3, "y": 3, "w": 5, "h": 5}}
    cell_slices = [
        {
            "pivot": None,
            "slices": [
                {
                    "name": "panel",
                    "x": 0,
                    "y": 0,
                    "w": 10,
                    "h": 10,
                    "pivot": None,
                    "center": (1, 1, 8, 8),
                }
            ],
        }
    ]
    (meta,) = sheetout._trim_shift_slices(cell_slices, trims)
    (entry,) = meta["slices"]
    cx, cy, cw, ch = entry["center"]
    assert (entry["x"], entry["y"], entry["w"], entry["h"]) == (0, 0, 5, 5)
    assert cx >= entry["x"] and cy >= entry["y"]
    assert cx + cw <= entry["x"] + entry["w"]
    assert cy + ch <= entry["y"] + entry["h"]
    assert cw > 0 and ch > 0


# -- inker-81: Columns never exceeds the frame count ----------------------------


def test_columns_arrange_never_exceeds_the_frame_count():
    # inker-sheets-05: three 16px frames with ``wrap=10`` gave ``columns=10``, a
    # 160x16 atlas and seven empty slots, while Rows clamped to the content.
    plan = sheetout.plan_frames(3, 16, 16, arrange="columns", wrap=10)
    assert plan.columns == 3
    assert plan.rows == 1
    assert plan.width == 48


# -- inker-33: a walk bake that ora could not reopen is refused -----------------


def test_a_walk_bake_refuses_a_canvas_the_ora_writer_could_not_reopen(monkeypatch):
    # inker-mode-05: fourteen parts over eight frames is 112 independent cels and
    # ``ora._layer_budget`` allows 104 at 800x800. The bake opened in a new tab
    # and could never be saved as ``.ora``.
    from realmspinner.studio.modes.inker import walk as inker_walk

    rig = R.blank((800, 800))
    for name in R.PART_NAMES:
        rig.parts[name] = R.Part(pixels=np.full((2, 2, 4), 255, dtype=np.uint8))
    monkeypatch.setattr(inker_walk.walk, "refusal", lambda _rig: "")
    for size in ((800, 800), (64, 64)):
        session = SimpleNamespace(tab_uid="t", rig=rig, size=size)
        state = SimpleNamespace(walk=session)
        tab = SimpleNamespace(uid="t", busy=False)
        reason = inker_walk.bake_reason(state, tab)
        if size == (800, 800):
            assert reason, "a canvas whose cels ora cannot reopen was accepted"
            assert ".ora" in reason or "layers" in reason
        else:
            assert reason == ""


# -- inker-73: no docstring names a mode that is gone ---------------------------


def test_no_source_docstring_names_a_removed_mode():
    # inker-mode-17: ``gait.py`` defended its design by contrast with "Poser's
    # clips", whose ``closed`` flag went with Poser (eed32549).
    src = Path(__file__).resolve().parents[3] / "src" / "realmspinner" / "kernels" / "pixel"
    text = (src / "walk" / "gait.py").read_text(encoding="utf-8")
    assert not re.search(r"\bPoser\b", text)


# -- inker-35: a full-width part is a copy --------------------------------------


def test_a_full_width_part_does_not_share_memory_with_its_source_layer():
    # inker-mode-07: a crop spanning the canvas width is already C-contiguous,
    # so ``np.ascontiguousarray`` returned a view and the part aliased the layer.
    plane = np.zeros((16, 16, 4), dtype=np.uint8)
    plane[4:9, :] = (200, 0, 0, 255)  # every column, some rows
    part = R.part_from_plane(plane)
    assert not np.shares_memory(part.pixels, plane)
    plane[5, 5] = (0, 0, 0, 0)
    assert tuple(part.pixels[1, 5]) == (200, 0, 0, 255)


# -- inker-69: a blank part is reported missing ---------------------------------


def test_a_part_cut_from_a_blank_layer_is_reported_missing():
    # inker-mode-13: ``trim`` hands back a 1x1 transparent crop that counts as
    # ``assigned``, so a blank torso gave ``missing_parts == ()`` and a bake of
    # an invisible torso with no warning.
    rig = R.blank((32, 32))
    full = np.zeros((32, 32, 4), dtype=np.uint8)
    full[4:20, 4:20] = (200, 0, 0, 255)
    for name in R.PART_NAMES:
        rig.parts[name] = R.part_from_plane(full)
    assert R.missing_parts(rig) == ()
    rig.parts["torso"] = R.part_from_plane(np.zeros((32, 32, 4), dtype=np.uint8))
    assert "torso" in R.missing_parts(rig)
