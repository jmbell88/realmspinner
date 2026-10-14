"""Regression tests closing 2026-09-14 audit findings with no better-fitting
existing test module. See ``docs/audit-2026-09-14.md`` for the full findings;
do not cite it from ``src/`` (``tests/test_ux_todo_fixes.py`` refuses that)."""

from __future__ import annotations

import inspect


def test_land_tileset_docstring_does_not_claim_the_projection_arm_is_test_only():
    """The 2026-09-14 audit, plotter-04: ``land_tileset``'s comment claimed
    its ``projection`` arm was "currently reached only from tests". Untrue:
    ``use_as_tileset`` submits the ``plotter-tileset:<uid>`` task, whose
    ``run()`` calls ``_sheet_or_tileset``, which sets
    ``out["projection"] = recorded_lattice`` for an unpainted map whose
    lattice disagrees with a recorded sheet's -- and
    ``plotter_mode.on_task_done``'s ``plotter-tileset`` arm passes that
    result straight into ``land_tileset``.
    """
    from realmspinner.studio.modes.plotter import tilesets as plotter_tilesets

    source = inspect.getsource(plotter_tilesets.land_tileset)
    assert "reached only from tests" not in source
    assert "use_as_tileset" in source


# --- plotter-02: ``_erase_seam`` has no behavioural test ---------------------
#
# Evidence gap, not a reproduced defect: ``_erase_seam`` (``_q_tileset.py``,
# called through the materials mode's ``seam_erase`` option) had only its
# cancel-flag door covered (``test_job_durability.py``'s stage-kind sweep),
# never whether it actually rewrites the material or leaves it alone on a
# cancel. Both tests below call the function directly with a fake ``t2i`` --
# a plain object with a ``generate`` method -- rather than through the async
# ``Worker``, since ``_erase_seam`` is itself synchronous (``to_thread``
# calls it) and needs nothing else the queue provides. **Both pass against
# HEAD's unfixed code**: this closes the evidence gap by pinning the
# behaviour the code already has, not a discovered defect.


def _material_image(path, fill):
    from PIL import Image

    Image.new("RGB", (32, 32), fill).save(path)


def test_seam_erase_actually_redraws_the_material_and_a_cancel_mid_pass_leaves_it_untouched(
    tmp_path,
):
    """Normal pass: ``_erase_seam`` reads ``material``, runs it through the
    fake ``t2i.generate`` and writes the rolled-back result over ``material``
    -- staged to a ``.tmp`` sibling and ``os.replace``'d, per this repo's
    stage-then-replace rule for a served name."""
    from realmspinner import _q_tileset

    material = tmp_path / "material.png"
    _material_image(material, (10, 20, 30))
    original_bytes = material.read_bytes()

    class FakeT2I:
        def generate(self, prompt, out, **kwargs):
            _material_image(out, (200, 100, 50))

    _q_tileset._erase_seam(
        FakeT2I(),
        material,
        tmp_path / "scratch",
        prompt="a mossy wall",
        seed=1,
        lora=None,
        lora_weight=0.0,
        negative_prompt="",
        size=(32, 32),
        cancel_event=None,
    )

    assert material.read_bytes() != original_bytes, "the material was not redrawn"
    assert not material.with_name(material.name + ".tmp").exists(), (
        "a staged temp sibling was left behind instead of being replaced"
    )


def test_a_cancel_mid_erase_pass_leaves_the_material_file_untouched(tmp_path):
    """A cancel discovered during ``t2i.generate`` (the fake sets the caller's
    own ``cancel_event`` the way the real pipeline would report one mid-draw)
    must leave ``material`` exactly as it was: ``_erase_seam`` checks
    ``cancel_event`` before it ever opens the redrawn output or touches the
    served file."""
    import threading

    from realmspinner import _q_tileset

    material = tmp_path / "material.png"
    _material_image(material, (10, 20, 30))
    original_bytes = material.read_bytes()
    cancel_event = threading.Event()

    class FakeT2I:
        def generate(self, prompt, out, *, cancel_event, **kwargs):
            # The pipeline discovers the cancel mid-draw and stops without
            # finishing ``out`` -- nothing here writes to it.
            cancel_event.set()

    _q_tileset._erase_seam(
        FakeT2I(),
        material,
        tmp_path / "scratch",
        prompt="a mossy wall",
        seed=1,
        lora=None,
        lora_weight=0.0,
        negative_prompt="",
        size=(32, 32),
        cancel_event=cancel_event,
    )

    assert material.read_bytes() == original_bytes
    assert not material.with_name(material.name + ".tmp").exists()


# --- the 2026-10-14 audit, finding plotter-tiles-10 ----------------------------
#
# ``_erase_seam``'s annotation said ``size: int`` while the worker hands it the
# ``(1024, 1024)`` tuple from ``geom.source_size`` -- and its only unit tests
# above passed ``size=32``, so nothing ever used the real call shape. The
# ordering the branch exists for (redraw the seam cross, *then* measure the seam,
# with the style lock on and a cancel possible mid-pass) was asserted by no test.


def test_erase_seam_is_annotated_with_the_frame_the_worker_passes():
    from realmspinner import _q_tileset

    assert _q_tileset._erase_seam.__annotations__["size"] == "tuple[int, int]"


def test_a_seam_erased_materials_set_measures_the_seam_after_the_redraw(
    tmp_path, fake_pipelines, monkeypatch
):
    """Runs the real ``_tile_set`` through the fake pipeline with ``seam_erase``
    and the style lock on: every material is redrawn (a masked img2img pass at
    the worker's real ``(w, h)`` frame) *before* ``seam.report`` measures it, and
    the first pass is not an erase."""
    import asyncio
    import time

    from realmspinner import _q_tileset
    from realmspinner.config import Config
    from realmspinner.db import JobStore
    from realmspinner.pipelines import seam, tileatlas
    from realmspinner.queue import Worker

    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )
    store = JobStore(config.db_path)
    worker = Worker(config, store)

    events: list[tuple[str, str, object]] = []
    real_erase = _q_tileset._erase_seam
    real_report = seam.report

    def spy_erase(t2i, material, scratch, **kwargs):
        events.append(("erase", material.name, kwargs["size"]))
        return real_erase(t2i, material, scratch, **kwargs)

    def spy_report(path):
        events.append(("measure", path.name, None))
        return real_report(path)

    monkeypatch.setattr(_q_tileset, "_erase_seam", spy_erase)
    monkeypatch.setattr(seam, "report", spy_report)

    prompts = ("moss", "gravel", "water")
    seeds = tileatlas.material_seeds(100, len(prompts))
    geom = tileatlas.material_geometry(32, "top_down", len(prompts))
    params = {
        "seed": 100,
        "base_model": "sdxl_cfg",
        "colors": 64,
        "negative_prompt": "",
        "sheet": {
            "version": 3, "mode": "materials", "tile_w": geom.tile_w, "tile_h": geom.tile_h,
            "projection": "top_down", "columns": geom.columns, "rows": geom.rows,
            "layout": "grid",
            "materials": [
                {"index": i, "prompt": p, "variant": 1, "seed": s}
                for i, (p, s) in enumerate(zip(prompts, seeds, strict=True))
            ],
            "variants": 1, "style_lock": True, "seam_erase": True,
        },
    }
    job_id = store.create("tile_sheet", "a dungeon", params, stage="tilesheet")

    async def run():
        worker.start()
        try:
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                if store.get(job_id)["status"] in ("done", "error", "cancelled"):
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("job did not finish before timeout")
        finally:
            await worker.shutdown()

    try:
        asyncio.run(run())
        row = store.get(job_id)
    finally:
        store.close()
    assert row["status"] == "done", row.get("error")

    names = [f"material-{i:02d}.png" for i in range(len(prompts))]
    assert [e[:2] for e in events] == [
        (kind, name) for name in names for kind in ("erase", "measure")
    ]
    assert all(e[2] == geom.source_size for e in events if e[0] == "erase")
    # Three generations plus three masked redraws, the redraws carrying the mask.
    pipe = worker._text2image
    masked = [c for c in pipe.conditionings if c is not None and c.mask_image is not None]
    assert len(pipe.conditionings) == 2 * len(prompts) and len(masked) == len(prompts)
