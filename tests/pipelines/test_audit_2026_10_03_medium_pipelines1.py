"""The 2026-10-03 audit's Medium findings pipelines-06 .. pipelines-17 (batch 1).

Each test's name is the claim. None needs a GPU, weights or ``bpy``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

# --- pipelines-06: the alpha mode is read before the first bake -------------


def _drive_remesh(monkeypatch, tmp_path, gltf_materials):
    """Run ``op_remesh`` against a mock bpy; -> (bake calls, outcome)."""
    from realmspinner.kernels.geom3d import glbio
    from realmspinner.pipelines import blender_worker as bw

    source = tmp_path / "source.glb"
    source.write_bytes(b"glTF")
    bakes: list[tuple[str, ...]] = []

    def fake_bake(*_a, maps, **_k):
        bakes.append(tuple(maps))
        return mock.MagicMock(), {}

    monkeypatch.setattr(glbio, "read_glb", lambda _p: ({"materials": gltf_materials}, b""))
    monkeypatch.setattr(bw, "_reset_scene", lambda _b: None)
    monkeypatch.setattr(bw, "_import_measured", lambda _b, _p: mock.MagicMock())
    monkeypatch.setattr(bw, "_face_stats", lambda _m: (10, 1.0))
    monkeypatch.setattr(bw, "_world_bounds", lambda _m: ([0, 0, 0], [1, 1, 1]))
    monkeypatch.setattr(bw, "weld_distance", lambda _lo, _hi: 0.0)
    monkeypatch.setattr(bw, "_remesh_object", lambda *_a, **_k: "quadriflow")
    monkeypatch.setattr(bw, "_smart_unwrap", lambda *_a, **_k: None)
    monkeypatch.setattr(bw, "_bake_maps", fake_bake)

    def stand_in_for_preserve(_source, _material, materials):
        # The real function needs a node tree; the refusal is the part under test.
        modes = {m.get("alphaMode", "OPAQUE") for m in materials}
        if {"MASK", "BLEND"} <= modes:
            raise RuntimeError("repair cannot combine MASK and BLEND alpha modes in one atlas")

    monkeypatch.setattr(bw, "_preserve_alpha_mode", stand_in_for_preserve)
    monkeypatch.setattr(bw, "_export", lambda *_a, **_k: None)
    monkeypatch.setattr(bw, "_tri_count", lambda _m: 10)
    monkeypatch.setattr(bw, "progress", lambda *_a, **_k: None)
    spec = {
        "source_glb": str(source), "out_glb": str(tmp_path / "out.glb"),
        "target_faces": 100, "texture_size": 64,
    }
    try:
        return bakes, bw.op_remesh(mock.MagicMock(), spec)
    except RuntimeError as exc:
        return bakes, exc


def test_repair_refuses_mixed_mask_and_blend_before_any_bake_runs(monkeypatch, tmp_path):
    bakes, outcome = _drive_remesh(
        monkeypatch, tmp_path, [{"alphaMode": "MASK"}, {"alphaMode": "BLEND"}]
    )
    assert isinstance(outcome, RuntimeError)
    assert "MASK and BLEND" in str(outcome)
    assert bakes == [], "the refusal came after the Cycles bakes had already been paid for"


def test_repair_of_an_opaque_source_does_not_bake_a_discarded_alpha_map(monkeypatch, tmp_path):
    bakes, outcome = _drive_remesh(monkeypatch, tmp_path, [{"alphaMode": "OPAQUE"}, {}])
    assert isinstance(outcome, dict)
    assert len(bakes) == 1 and "alpha" not in bakes[0]
    assert "base_color" in bakes[0] and "metallic" in bakes[0]


def test_repair_of_a_blended_source_still_bakes_alpha(monkeypatch, tmp_path):
    bakes, _outcome = _drive_remesh(monkeypatch, tmp_path, [{"alphaMode": "BLEND"}])
    assert "alpha" in bakes[0]


# --- pipelines-07: a dead image child says why -------------------------------


def test_a_child_that_dies_says_why_in_the_failure_it_raises(monkeypatch, tmp_path):
    from realmspinner import models
    from realmspinner.pipelines import t2i_client
    from realmspinner.pipelines.t2i_client import ChildFailed, Text2ImageClient

    startup = tmp_path / "dies_at_startup.py"
    startup.write_text(
        "import sys\n"
        "sys.stdout.write('Traceback (most recent call last):\\n')\n"
        "sys.stdout.write(\"ModuleNotFoundError: No module named 'diffusers'\\n\")\n"
        "sys.stdout.flush()\n"
        "raise SystemExit(1)\n"
    )
    fake = Path(__file__).parents[1] / "fixtures" / "fake_t2i_worker.py"

    monkeypatch.setattr(t2i_client, "CHILD_ARGV", [sys.executable, str(startup)])
    at_startup = Text2ImageClient(models.BASE_MODELS["sdxl_cfg"], tmp_path)
    try:
        with pytest.raises(ChildFailed) as startup_err:
            at_startup.load()
    finally:
        at_startup.close()
    assert "No module named 'diffusers'" in str(startup_err.value)

    monkeypatch.setattr(t2i_client, "CHILD_ARGV", [sys.executable, str(fake)])
    mid_request = Text2ImageClient(models.BASE_MODELS["sdxl_cfg"], tmp_path)
    try:
        with pytest.raises(ChildFailed) as mid_err:
            mid_request._request({"op": "generate", "output": "x", "chatter": True, "die": True})
    finally:
        mid_request.close()
    assert "Loading pipeline components" in str(mid_err.value)


# --- pipelines-08: an opaque RGBA picture is not "transparent" ----------------


def test_a_jpeg_from_an_opaque_rgba_picture_is_written_not_refused(tmp_path):
    from PIL import Image

    from realmspinner.pipelines import imageout

    source = tmp_path / "input.png"
    Image.new("RGBA", (16, 16), (10, 20, 30, 255)).save(source, "PNG")
    out = tmp_path / "input.jpg"
    imageout.convert(source, out, "input.jpg")
    with Image.open(out) as written:
        assert written.format == "JPEG" and written.mode == "RGB"
        r, g, b = written.getpixel((8, 8))
        assert abs(r - 10) <= 3 and abs(g - 20) <= 3 and abs(b - 30) <= 3

    # The refusal still stands once any pixel really is transparent.
    holed = Image.new("RGBA", (16, 16), (10, 20, 30, 255))
    holed.putpixel((0, 0), (10, 20, 30, 0))
    holed.save(source, "PNG")
    with pytest.raises(imageout.AlphaUnsupported):
        imageout.convert(source, tmp_path / "again.jpg", "input.jpg")


# --- pipelines-09: the Colours cap belongs to the subject --------------------

_SUBJECT_COLOURS = {(220, 40, 40), (40, 200, 60), (50, 60, 230), (240, 220, 40)}
_BACKGROUND = (200, 190, 30)


def _round_four_colour_subject(cells=32, scale=4):
    """A round subject of four flat colours on a flat background, as art cells
    blown up ``scale`` times -> (image, mask)."""
    import numpy as np
    from PIL import Image

    art = np.empty((cells, cells, 3), np.uint8)
    art[:] = _BACKGROUND
    yy, xx = np.mgrid[0:cells, 0:cells]
    centre = (cells - 1) / 2
    disc = (yy - centre) ** 2 + (xx - centre) ** 2 <= (cells * 0.4) ** 2
    colours = sorted(_SUBJECT_COLOURS)
    quadrant = (yy >= cells // 2).astype(int) * 2 + (xx >= cells // 2).astype(int)
    for index, colour in enumerate(colours):
        art[disc & (quadrant == index)] = colour
    big = np.kron(art, np.ones((scale, scale, 1), np.uint8))
    mask = np.kron(disc, np.ones((scale, scale), bool))
    return Image.fromarray(big, "RGB"), mask


def _subject_colours(out):
    import numpy as np

    rgba = np.asarray(out.convert("RGBA"))
    return {tuple(int(v) for v in px[:3]) for px in rgba[rgba[:, :, 3] > 0]}


def test_a_colour_cap_is_spent_on_the_subject_not_on_the_discarded_background(monkeypatch):
    from realmspinner.pipelines import asset2d
    from realmspinner.pipelines import pixel as pixelmod

    image, mask = _round_four_colour_subject()
    # The legacy branch (no lattice detected): a plain reduction of the cutout.
    no_grid = {"scale": None, "phase": None, "residual": 1.0}
    monkeypatch.setattr(pixelmod, "detect_grid", lambda _im: no_grid)
    out, meta = asset2d.pixel(image, mask, size=64, colors=4)
    assert meta["palette"] == 4
    assert _subject_colours(out) == _SUBJECT_COLOURS

    # The grid branch quantizes on its own, and had the same defect.
    on_grid = {"scale": 4, "phase": (0, 0), "residual": 0.0}
    monkeypatch.setattr(pixelmod, "detect_grid", lambda _im: on_grid)
    out, meta = asset2d.pixel(image, mask, size=64, colors=4)
    assert meta["palette"] == 4
    assert _subject_colours(out) == _SUBJECT_COLOURS


# --- pipelines-10: a forced health recheck forgets a matting load failure -----


def test_a_forced_health_recheck_forgets_a_matting_load_failure_after_a_pack_install(
    monkeypatch, tmp_path
):
    from realmspinner import doctor, fetch
    from realmspinner.config import Config
    from realmspinner.pipelines import matting

    (tmp_path / "assets").mkdir()
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        trellis_port=0,
    )
    # The weights are on disk and the probe loads; the only thing left that can
    # keep the row red is the failure memo an earlier load left behind.
    monkeypatch.setattr(fetch, "present", lambda *_a, **_k: True)
    monkeypatch.setattr(doctor, "_load_probe", lambda *_a, **_k: (True, "loads"))
    monkeypatch.setattr(doctor, "_missing_modules", lambda *_a, **_k: [])
    monkeypatch.setitem(matting._cache, "stale-model-key", matting._FAILED)
    monkeypatch.setattr(matting, "_last_error", "ModuleNotFoundError: No module named 'einops'")

    def matting_row():
        rows = [c for c in doctor.static_checks(config, probe_slow=False, force=True)
                if "atting" in c.name]
        assert rows, "the matting row is gone from static_checks"
        return rows[0]

    row = matting_row()
    assert "last load failed" not in row.detail
    assert matting.last_error() is None
    # The memo that makes ``matting.mask`` raise ``_AlreadyFailed`` is gone too.
    assert matting._FAILED not in matting._cache.values()


# --- pipelines-11: sweep_refill names only units nobody has refilled ----------


def _load_sweep_refill():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "sweep_refill.py"
    spec = importlib.util.spec_from_file_location("sweep_refill_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _svc_with_rows(rows):
    class _Store:
        @staticmethod
        def sweep_jobs(_sweep_id):
            return list(rows)

    class _Svc:
        store = _Store()

    return _Svc()


def test_sweep_refill_does_not_requeue_a_unit_that_was_already_refilled(monkeypatch):
    refill = _load_sweep_refill()
    shutdown = f"{refill.SHUTDOWN} (the app was closed)"
    rows = [
        # Cancelled, then refilled and still waiting: not lost any more.
        {"sweep_unit": "baseline s1", "status": "cancelled", "error": None},
        {"sweep_unit": "baseline s1", "status": "queued", "error": None},
        # Interrupted, then refilled and finished.
        {"sweep_unit": "baseline s2", "status": "error", "error": shutdown},
        {"sweep_unit": "baseline s2", "status": "done", "error": None},
        # Interrupted, refilled, and the refill is running.
        {"sweep_unit": "lora=0.6 s1", "status": "error", "error": shutdown},
        {"sweep_unit": "lora=0.6 s1", "status": "running", "error": None},
        # Still lost: nobody has refilled it.
        {"sweep_unit": "lora=0.6 s2", "status": "cancelled", "error": None},
        # A genuine refusal stays a measurement.
        {"sweep_unit": "lora=1.2 s1", "status": "error", "error": "composition gate"},
    ]
    lost, refused = refill.lost_units(_svc_with_rows(rows), "sweep")
    assert lost == ["lora=0.6 s2"]
    assert refused == ["lora=1.2 s1"]


# --- pipelines-15: no pack installed is "set up", not "needs attention" -------


def _no_pack_installed(monkeypatch):
    """A fresh installer's host: no pack's modules resolve, and every probe
    child fails the way an import of an absent module does."""
    from types import SimpleNamespace

    from realmspinner import doctor, packs

    monkeypatch.setattr(packs, "missing_modules", lambda names: list(names))
    monkeypatch.setitem(sys.modules, "torch", None)  # ``import torch`` -> ImportError
    monkeypatch.setattr(
        doctor.winjob, "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="", stderr="ModuleNotFoundError: No module named 'absent'"
        ),
    )
    monkeypatch.setattr(doctor, "_blender", None)
    monkeypatch.setattr(doctor, "_music_deps", None)
    monkeypatch.setattr(doctor, "_t2i_deps", None)
    return doctor


def test_a_fresh_install_with_no_packs_counts_no_dependency_row_as_an_issue(monkeypatch):
    doctor = _no_pack_installed(monkeypatch)
    rows = [
        doctor._cuda_check(probe=True),
        doctor.blender_check(force=True),
        doctor.text2image_deps_check(force=True),
        doctor.music_deps_check(force=True),
    ]
    assert [r.name for r in rows] == [
        "CUDA", "Blender (rigging)", "Create (dependencies)", "Muse (dependencies)"
    ]
    for row in rows:
        assert not row.ok and not row.fatal
        assert row.pending_install, f"{row.name} would be counted as an issue on a fresh install"
        assert "uv sync" not in row.detail, f"{row.name} prints a command a build cannot run"
        assert "Settings -> Packs" in row.detail
    # The same filter the menu bar, Home and Settings -> Health apply.
    assert [r for r in rows if not r.ok and not r.pending_install] == []


def test_a_pack_that_resolves_but_will_not_import_is_still_an_issue(monkeypatch):
    doctor = _no_pack_installed(monkeypatch)
    from realmspinner import packs

    monkeypatch.setattr(packs, "missing_modules", lambda names: [])  # every module resolves
    for row in (
        doctor.blender_check(force=True),
        doctor.text2image_deps_check(force=True),
        doctor.music_deps_check(force=True),
    ):
        assert not row.ok and not row.pending_install, row.name


# --- pipelines-16: the retained RotSprite scratch has the stated bound --------


def test_the_retained_rotsprite_scratch_never_exceeds_the_interactive_budget(monkeypatch):
    import numpy as np

    from realmspinner import native
    from realmspinner.kernels.pixel import transform as tf
    from realmspinner.kernels.pixel.walk import render

    if not native.available():
        pytest.skip("the native kernels are not built here")
    monkeypatch.setattr(native, "_rotsprite_scratch", None)
    # What native.py documents: 80 bytes a pixel per channel at the interactive
    # budget, i.e. an RGBA plane of ROTSPRITE_MAX_PIXELS is about 80 MiB.
    bound = 80 * tf.ROTSPRITE_MAX_PIXELS * 4

    # An ordinary interactive-sized call still keeps its buffer (the point of
    # retaining one is the free-transform drag's per-move page-fault storm).
    small = np.zeros((64, 64, 4), np.uint8)
    small[16:48, 16:48] = 255
    assert native.rotsprite_u8(small, 33.0) is not None
    assert native._rotsprite_scratch is not None

    # The walk bake's 4x budget: a plane just past the interactive ceiling.
    side = int(tf.ROTSPRITE_MAX_PIXELS**0.5) + 40
    assert side * side > tf.ROTSPRITE_MAX_PIXELS
    assert side * side <= render.ROTSPRITE_BUDGET
    big = np.zeros((side, side, 4), np.uint8)
    big[side // 4 : side // 2, side // 4 : side // 2] = 255
    assert native.rotsprite_u8(big, 33.0) is not None
    held = native._rotsprite_scratch
    assert held is None or held.nbytes <= bound, (
        f"{held.nbytes:,} bytes pinned for the life of the process; the stated "
        f"bound is {bound:,}"
    )
