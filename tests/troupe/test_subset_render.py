"""The subset half of a re-render: which cells, and how they get composited.

Mostly pure -- no worker, no Blender, no job store, ``charsheet`` being
filesystem-free by design and ``sheet.pack``/``compose_cells`` taking paths and
nothing else -- except the one section at the end that needs the worker: D5's
claim that an HD subset re-render pins no palette is about ``_q_troupe``'s own
branching, not about this module's pure functions.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock.pipelines import charsheet
from warlock.pipelines import sheet as sheetlib


def _runs(n=1):
    return [
        {"animation": animation, "direction": direction}
        for animation, direction, *_ in charsheet.spans()[:n]
    ]


# -- which runs -----------------------------------------------------------


def test_a_subset_normalises_to_animation_and_direction_pairs():
    wanted = charsheet.check_subset(_runs(3))
    assert len(wanted) == 3
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in wanted)


@pytest.mark.parametrize(
    ("subset", "says"),
    [
        (None, "at least one"),
        ([], "at least one"),
        ([{"animation": "walk"}], "needs an animation and a direction"),
        ([{"animation": "", "direction": "front"}], "needs an animation and a direction"),
        ([{"animation": "moonwalk", "direction": "front"}], "not an animation"),
        ([{"animation": "walk", "direction": "upwards"}], "no walk facing upwards"),
    ],
)
def test_a_subset_is_refused_by_name(subset, says):
    """``check_frame_counts``' rule one function up: a request naming something
    the sheet does not carry is a mistake about the sheet, and rendering the
    rest of it would leave the user hunting a change that never happened."""
    with pytest.raises(ValueError, match=says):
        charsheet.check_subset(subset)


def test_the_same_run_twice_is_refused():
    run = _runs(1)
    with pytest.raises(ValueError, match="named twice"):
        charsheet.check_subset(run + run)


def test_every_run_at_once_is_refused_as_a_full_render():
    """Not pedantry: it is a full render taking the slower path, paying for a
    copy step and a pinned palette it does not need. The ordinary door exists."""
    everything = [
        {"animation": animation, "direction": direction}
        for animation, direction, *_ in charsheet.spans()
    ]
    with pytest.raises(ValueError, match="every run"):
        charsheet.check_subset(everything)


# -- which cells ----------------------------------------------------------


def test_the_indices_are_exactly_the_runs_spans():
    """Expanded from ``spans`` rather than from arithmetic of its own, so this
    and ``sheetscope.runs`` come from the two copies the geometry-agreement
    test owns rather than from a third nothing owns."""
    animation, direction, start, end, _loop = charsheet.spans()[2]
    indices = charsheet.subset_indices([{"animation": animation, "direction": direction}])
    assert indices == tuple(range(start, end + 1))


def test_the_indices_of_several_runs_are_sorted_and_disjoint():
    indices = charsheet.subset_indices(_runs(4))
    assert list(indices) == sorted(indices)
    assert len(set(indices)) == len(indices)


def test_a_subset_never_names_a_cell_the_full_table_does_not_have():
    every = {cell.index for cell in charsheet.frame_table()}
    assert set(charsheet.subset_indices(_runs(5))) <= every


# -- the geometry claim the whole feature rests on ------------------------


def _plan(**kw):
    records = {
        movement.name: [{"id": movement.name, "frame": i} for i in range(movement.frames)]
        for movement in charsheet.resolve_layout().movements
    }
    return charsheet.plan(records, frame_size=16, **kw)


def test_a_subset_changes_no_cell_geometry_at_all():
    """**The claim the merge rests on.** A re-rendered cell has to land on the
    rectangle it has always had, or the composite is a smear. The plan is built
    unfiltered and the *spec list* is what gets filtered, so this is a property
    of the design rather than something to be careful about."""
    plan = _plan()
    geometry = [
        (cell.index, cell.row, cell.column, cell.x, cell.y) for cell in plan.cells
    ]
    wanted = set(charsheet.subset_indices(_runs(2)))
    subset = [cell for cell in plan.cells if cell.index in wanted]

    for cell in subset:
        assert (cell.index, cell.row, cell.column, cell.x, cell.y) in geometry


# -- packing a subset ------------------------------------------------------


def _write_cells(tmp_path, plan, indices, value):
    out = {}
    for cell in plan.cells:
        if cell.index not in indices:
            continue
        pixels = np.zeros((plan.cell_h, plan.cell_w, 4), dtype=np.uint8)
        pixels[..., 0] = value
        pixels[..., 3] = 255
        path = tmp_path / f"{cell.index:04d}.png"
        Image.fromarray(pixels, "RGBA").save(path)
        out[cell.index] = path
    return out


def test_packing_a_subset_leaves_every_other_cell_transparent(tmp_path):
    plan = _plan()
    wanted = set(charsheet.subset_indices(_runs(1)))
    frames = _write_cells(tmp_path, plan, wanted, 200)

    out = tmp_path / "subset.png"
    trims = sheetlib.pack(plan, frames, out, only=wanted)

    assert set(trims) == wanted, "a cell outside the subset is not measured"
    with Image.open(out) as opened:
        atlas = np.asarray(opened.convert("RGBA"))
    assert atlas.shape[:2] == (plan.height, plan.width)

    outside = next(cell for cell in plan.cells if cell.index not in wanted)
    patch = atlas[outside.y : outside.y + plan.cell_h, outside.x : outside.x + plan.cell_w]
    assert patch[..., 3].max() == 0


def test_a_named_cell_that_did_not_render_is_still_refused(tmp_path):
    """The filter must not become a relaxation. A dropped render frame and a
    subset have to stay distinguishable in the one function positioned to
    notice."""
    plan = _plan()
    wanted = set(charsheet.subset_indices(_runs(1)))
    frames = _write_cells(tmp_path, plan, wanted, 200)
    frames.pop(sorted(wanted)[0])

    with pytest.raises(ValueError, match="no rendered frame"):
        sheetlib.pack(plan, frames, tmp_path / "out.png", only=wanted)


# -- composing -------------------------------------------------------------


def _atlas(plan, value, path):
    pixels = np.zeros((plan.height, plan.width, 4), dtype=np.uint8)
    pixels[..., 1] = value
    pixels[..., 3] = 255
    Image.fromarray(pixels, "RGBA").save(path)
    return path


def test_composing_replaces_only_the_named_cells(tmp_path):
    plan = _plan()
    wanted = set(charsheet.subset_indices(_runs(1)))
    base = _atlas(plan, 50, tmp_path / "base.png")
    overlay = _atlas(plan, 250, tmp_path / "overlay.png")

    out = tmp_path / "merged.png"
    sheetlib.compose_cells(base, overlay, plan, wanted, out)

    with Image.open(out) as opened:
        merged = np.asarray(opened.convert("RGBA"))

    inside = next(cell for cell in plan.cells if cell.index in wanted)
    outside = next(cell for cell in plan.cells if cell.index not in wanted)
    assert merged[inside.y, inside.x, 1] == 250
    assert merged[outside.y, outside.x, 1] == 50


def test_composing_twice_is_byte_identical(tmp_path):
    """**The test the outline finding is about.** ``outline`` in the shipped
    ``outer`` mode grows a silhouette by a pixel on every side, so a design that
    composed first and quantised after would fatten every copied cell once per
    re-render -- and the sheet would go a pixel thinner in the runs nobody
    touched, with nothing to say why. Composing is a paste and nothing else."""
    plan = _plan()
    wanted = set(charsheet.subset_indices(_runs(1)))
    base = _atlas(plan, 50, tmp_path / "base.png")
    overlay = _atlas(plan, 250, tmp_path / "overlay.png")

    once = tmp_path / "once.png"
    twice = tmp_path / "twice.png"
    sheetlib.compose_cells(base, overlay, plan, wanted, once)
    sheetlib.compose_cells(once, overlay, plan, wanted, twice)

    assert once.read_bytes() == twice.read_bytes()


def test_a_base_of_the_wrong_size_is_refused_by_name(tmp_path):
    plan = _plan()
    overlay = _atlas(plan, 250, tmp_path / "overlay.png")
    wrong = tmp_path / "wrong.png"
    Image.fromarray(
        np.zeros((plan.height // 2, plan.width, 4), dtype=np.uint8), "RGBA"
    ).save(wrong)

    with pytest.raises(ValueError, match="re-rendered is"):
        sheetlib.compose_cells(wrong, overlay, plan, {0}, tmp_path / "out.png")


def test_a_replaced_cell_does_not_show_the_old_silhouette_through(tmp_path):
    """Pasted, not alpha-composited. Blending would leave the previous pose
    showing wherever the new one is transparent, which is a ghost."""
    plan = _plan()
    base = _atlas(plan, 50, tmp_path / "base.png")
    clear = np.zeros((plan.height, plan.width, 4), dtype=np.uint8)
    overlay = tmp_path / "overlay.png"
    Image.fromarray(clear, "RGBA").save(overlay)

    out = tmp_path / "out.png"
    sheetlib.compose_cells(base, overlay, plan, {0}, out)

    with Image.open(out) as opened:
        merged = np.asarray(opened.convert("RGBA"))
    cell = plan.cells[0]
    patch = merged[cell.y : cell.y + plan.cell_h, cell.x : cell.x + plan.cell_w]
    assert patch[..., 3].max() == 0


# -- D5: an HD subset re-render has no palette to pin -------------------------
#
# Self-contained -- its own worker fixture, its own render fake -- rather than
# importing ``test_troupe_chain.py``'s private helpers, ``test_charsheet_cancel.py``'s
# convention: that module is not owned by this fix and its internals are free
# to change. Everything above this section stays pure; this is the one case in
# the file that needs the worker, because "does not pin a palette" is a claim
# about ``_q_troupe._charsheet``'s own branching, not about ``sheet.py`` alone.


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    from warlock.config import Config
    from warlock.db import JobStore
    from warlock.queue import Worker

    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    yield w
    store.close()


async def _wait_until(predicate, timeout: float = 20.0) -> None:
    import asyncio

    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


def _gradient_render(monkeypatch):
    """A Blender fake that paints a colour gradient with soft alpha. A flat
    fill quantises losslessly and would prove nothing about whether the
    quantise pass ran; a gradient could not survive it."""
    from pathlib import Path

    from warlock.pipelines import blender_run

    def fake(spec, **kwargs):
        frames_dir = Path(spec["frames_dir"])
        size = spec["frame_size"]
        ys, xs = np.mgrid[0:size, 0:size]
        red = (xs * 255 // max(size - 1, 1)).astype(np.uint8)
        green = (ys * 255 // max(size - 1, 1)).astype(np.uint8)
        blue = np.full((size, size), 128, dtype=np.uint8)
        cx = cy = size / 2.0
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / (size / 2.0)
        alpha = np.clip(255 * (1.0 - dist), 0, 255).astype(np.uint8)
        rgba = np.dstack([red, green, blue, alpha])
        for cell in spec["cells"]:
            Image.fromarray(rgba, "RGBA").save(frames_dir / f"{cell['index']:04d}.png")
        return {
            "ok": True,
            "pivot": [0.5, 0.9],
            "framing": {"extent": 2.24, "margin": spec.get("margin") or 1.12},
        }

    monkeypatch.setattr(blender_run, "run_worker", fake)


async def test_an_hd_subset_rerender_does_not_pin_a_palette(worker, monkeypatch):
    """A subset re-render's pinned-palette path (``_atlas_entries``) exists to
    keep the re-rendered runs the same shade as the ones beside them -- a
    question a full-colour HD atlas never asks. Pre-fix, the row's absent
    ``colors`` defaulted to 64 and the sheet was quantised anyway; post-fix,
    the composed atlas keeps the same wide colour range and soft alpha the
    base did."""
    import json

    from warlock.kernels.rig import store as rig_store

    _gradient_render(monkeypatch)
    source = worker.store.create("image", "a ranger", {}, stage="model")
    source_dir = worker.config.job_dir(source)
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "model.glb").write_bytes(b"fake-glb")
    (source_dir / "rig.glb").write_bytes(b"fake-rig")
    (source_dir / "rig.json").write_text(json.dumps({"template": "humanoid"}), "utf-8")
    worker.store.set_status(source, "done")

    layout = {
        "version": 2,
        "movements": [
            {"key": "idle", "frames": 3, "directions": 1},
            {"key": "attack", "frames": 2, "directions": 1},
        ],
    }

    def _queue(**extra):
        return worker.store.create(
            "charsheet",
            "a ranger",
            {
                "source_job": source,
                "sheet_id": rig_store.new_id(),
                "logical_size": 16,
                "pixel_art": False,
                "layout": layout,
                **extra,
            },
        )

    first = _queue()
    worker.start()
    try:
        await _wait_until(
            lambda: worker.store.get(first)["status"] in ("done", "error"), 60.0
        )
        assert worker.store.get(first)["error"] is None
        base_sheet = worker.store.get(first)["params"]["sheet_id"]

        rerun = _queue(
            subset=[{"animation": "attack", "direction": "front"}],
            base_sheet=base_sheet,
        )
        await _wait_until(
            lambda: worker.store.get(rerun)["status"] in ("done", "error"), 60.0
        )
    finally:
        await worker.shutdown()

    assert worker.store.get(rerun)["error"] is None
    report = worker.store.get(rerun)["params"]["pixel_report"]
    assert report["style"] == "hd"
    assert "palette" not in report and "palette_name" not in report

    rerun_sheet = worker.store.get(rerun)["params"]["sheet_id"]
    png = rig_store.sheet_png_path(source_dir, rerun_sheet)
    with Image.open(png) as opened:
        opened.load()
        atlas = np.asarray(opened.convert("RGBA"))
    pixels = atlas.reshape(-1, 4)
    opaque = pixels[pixels[:, 3] > 0]
    colours = {tuple(int(v) for v in rgb) for rgb in opaque[:, :3]}
    assert len(colours) > 64
    alpha = pixels[:, 3]
    assert ((alpha > 0) & (alpha < 255)).any()
