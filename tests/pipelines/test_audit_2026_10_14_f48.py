"""The 2026-10-14 audit's findings muse-03, muse-12, pipelines-07, pipelines-13,
pipelines-22, inker-51, inker-82, inker-80 (the reference-only half), poser-23 and
poser-25. Each test's name is the claim.
"""

from __future__ import annotations

import struct
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from realmspinner import leases, models
from realmspinner.pipelines import music_client, pixel, prompt, reference, retexture, spritesynth
from realmspinner.pipelines.music_client import ChildFailed, MusicCancelled, MusicClient

SRC = Path(__file__).parents[2] / "src" / "realmspinner"
FAKE = Path(__file__).parents[1] / "fixtures" / "fake_music_worker.py"


def _source(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


# --- muse-03 and muse-12: the music child ------------------------------------

_PRELUDE = textwrap.dedent(
    '''
    import json, sys, time
    MARKER = "@@realmspinner-music@@ "

    def emit(msg):
        sys.stdout.write("\\n" + MARKER + json.dumps(msg) + "\\n")
        sys.stdout.flush()
    '''
)


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "scripted_worker.py"
    path.write_text(_PRELUDE + textwrap.dedent(body), encoding="utf-8")
    return path


def _client(monkeypatch, tmp_path: Path, script: Path) -> MusicClient:
    monkeypatch.setattr(music_client, "CHILD_ARGV", [sys.executable, str(script)])
    return MusicClient(models.MUSIC_MODELS["ace_step_v1"], tmp_path / "ace-step")


def test_a_music_child_that_dies_at_startup_reports_what_it_printed(monkeypatch, tmp_path):
    """The 2026-10-14 audit, finding muse-jobs-01: the client threw the child's
    output away, so a missing torch or a broken CUDA DLL surfaced as the bare
    "exited during startup" with the traceback nowhere a user could read it."""
    script = _script(
        tmp_path,
        """
        sys.stderr.write("Traceback (most recent call last):\\n")
        sys.stderr.write("ImportError: DLL load failed while importing torch\\n")
        sys.stderr.flush()
        sys.exit(1)
        """,
    )
    client = _client(monkeypatch, tmp_path, script)
    try:
        with pytest.raises(ChildFailed) as caught:
            client.load()
    finally:
        client.close()
    assert "exited during startup" in str(caught.value)
    assert "DLL load failed while importing torch" in str(caught.value)


def test_a_music_child_that_dies_mid_request_reports_what_it_printed(monkeypatch, tmp_path):
    script = _script(
        tmp_path,
        """
        emit({"kind": "ready"})
        sys.stdin.readline()
        print("torch.OutOfMemoryError: CUDA out of memory", flush=True)
        sys.exit(3)
        """,
    )
    client = _client(monkeypatch, tmp_path, script)
    try:
        with pytest.raises(ChildFailed) as caught:
            client._request({"op": "generate", "output": "x"})
    finally:
        client.close()
    assert "exited without answering" in str(caught.value)
    assert "CUDA out of memory" in str(caught.value)


def test_the_childs_tail_is_bounded_and_does_not_carry_over_to_the_next_child(
    monkeypatch, tmp_path
):
    script = _script(
        tmp_path,
        """
        for i in range(500):
            print(f"noise line {i}", flush=True)
        sys.exit(1)
        """,
    )
    client = _client(monkeypatch, tmp_path, script)
    try:
        with pytest.raises(ChildFailed) as caught:
            client.load()
    finally:
        client.close()
    text = str(caught.value)
    assert "noise line 499" in text
    assert "noise line 0\n" not in text
    assert len(text) < music_client.ERROR_TAIL_CHARS + 200


def test_cancel_during_the_load_phase_stops_the_child_without_waiting_for_the_first_step(
    monkeypatch, tmp_path
):
    """The 2026-10-14 audit, finding muse-jobs-04: a Cancel was relayed to a
    child that checks it only at the sampler's first step, so a cold checkpoint
    load or a source-audio encode ran to its end for a job that was already
    being unloaded. The child here ignores the cancel line, as a loading one
    does; the silence timeout is shortened so a client that relays and waits
    fails in seconds rather than after fifteen minutes."""
    monkeypatch.setattr(music_client, "SILENCE_TIMEOUT", 6.0)
    script = _script(
        tmp_path,
        """
        emit({"kind": "ready"})
        sys.stdin.readline()
        emit({"kind": "state", "text": "load"})
        time.sleep(60)
        """,
    )
    client = _client(monkeypatch, tmp_path, script)
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    started = time.monotonic()
    try:
        with pytest.raises(MusicCancelled):
            client.generate("x", tmp_path / "track.wav", cancel_event=cancel)
        proc = client._proc
        assert proc is None, "the child that was still loading must have been stopped"
    finally:
        client.close()
    assert time.monotonic() - started < 5.0


def test_cancel_while_the_child_is_still_starting_stops_it_without_waiting_for_ready(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(music_client, "READY_TIMEOUT", 6.0)
    script = _script(
        tmp_path,
        """
        time.sleep(60)
        """,
    )
    client = _client(monkeypatch, tmp_path, script)
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    started = time.monotonic()
    try:
        with pytest.raises(MusicCancelled):
            client.generate("x", tmp_path / "track.wav", cancel_event=cancel)
        assert client._proc is None
    finally:
        client.close()
    assert time.monotonic() - started < 5.0


def test_a_cancel_after_the_first_step_is_still_relayed_and_keeps_the_child(
    monkeypatch, tmp_path
):
    """Once the sampler is stepping it honours the cancel itself, and the warm
    child survives the cancelled take."""
    script = _script(
        tmp_path,
        """
        emit({"kind": "ready"})
        sys.stdin.readline()
        emit({"kind": "step", "step": 1, "total": 60})
        for raw in sys.stdin:
            if '"cancel"' in raw:
                emit({"kind": "error", "error": "cancelled", "cancelled": True})
                break
        time.sleep(30)
        """,
    )
    client = _client(monkeypatch, tmp_path, script)
    cancel = threading.Event()
    seen = threading.Event()

    def on_step(_i, _n):
        seen.set()
        cancel.set()

    try:
        with pytest.raises(MusicCancelled):
            client.generate("x", tmp_path / "track.wav", on_step=on_step, cancel_event=cancel)
        assert seen.is_set()
        assert client._proc is not None and client._proc.poll() is None
    finally:
        client.close()


def test_load_and_generate_hold_the_model_lease_and_unload_does_not(monkeypatch, tmp_path):
    """The 2026-10-14 audit, finding service-queue-07 as it lands on the music
    client: with no lease a Download could start while the child was mid
    ``from_pretrained``. ``unload``/``close`` run on the loop thread and must
    not park behind a maintainer."""
    client = MusicClient(models.MUSIC_MODELS["ace_step_v1"], tmp_path / "ace-step")
    seen: dict[str, int] = {}

    def fake_request(self, payload, **_kwargs):
        seen[payload["op"]] = leases.MODELS.users
        return {"cancelled": False, "recipe": {}}

    monkeypatch.setattr(MusicClient, "_request", fake_request)
    client.load()
    client.generate("x", tmp_path / "track.wav")
    assert seen == {"load": 1, "generate": 1}
    assert leases.MODELS.users == 0

    def fake_stop(self):
        seen["unload"] = leases.MODELS.users

    monkeypatch.setattr(MusicClient, "_stop_child", fake_stop)
    client.unload()
    assert seen["unload"] == 0


# --- pipelines-27: the optimize door (the service half is in the service file) ---


# --- inker-51: stale module path ----------------------------------------------


def test_pixel_pipeline_comments_name_the_real_gpl_module():
    text = _source("pipelines/pixel.py")
    assert "studio/inker/gpl" not in text
    assert (SRC / "kernels" / "pixel" / "gpl.py").is_file()
    assert "kernels/pixel/gpl" in text


# --- inker-82: the orphan count -----------------------------------------------


def _dot() -> Image.Image:
    arr = np.zeros((7, 7, 4), np.uint8)
    arr[3, 3] = (255, 0, 0, 255)
    return Image.fromarray(arr, "RGBA")


def test_clean_orphans_counts_only_pixels_it_recoloured():
    """The 2026-10-14 audit, finding inker-sheets-06: a floating dot has no
    opaque neighbour to take a colour from, so it was left alone and still
    counted as cleaned."""
    out, cleaned = pixel.clean_orphans(_dot())
    assert cleaned == 0
    assert tuple(np.asarray(out)[3, 3]) == (255, 0, 0, 255)

    arr = np.zeros((7, 7, 4), np.uint8)
    arr[:, :, :3] = (20, 20, 20)
    arr[:, :, 3] = 255
    arr[1, 1, :3] = (255, 0, 0)
    out, cleaned = pixel.clean_orphans(Image.fromarray(arr, "RGBA"))
    assert cleaned == 1
    assert tuple(np.asarray(out)[1, 1, :3]) == (20, 20, 20)


def test_the_report_still_measures_a_floating_dot_as_a_lone_pixel():
    """``report`` reads the count as "lone pixels found", which a dot is; fixing
    what ``clean_orphans`` calls cleaned must not blind the measurement."""
    rep = pixel.report(_dot())
    assert rep["orphan_fraction"] == pytest.approx(1.0)


# --- pipelines-22: the build docstring -----------------------------------------


def test_the_prompt_build_docstring_states_the_order_the_code_uses():
    assert prompt.build("a barrel", {}, trigger="TRIG").startswith("TRIG, ")
    doc = prompt.build.__doc__ or ""
    assert "then the LoRA trigger" not in doc
    assert "LoRA trigger first" in doc


# --- pipelines-07: normalisation must not launder a refusal --------------------

BG = (200, 200, 200)


def _cropped_subject() -> Image.Image:
    """A subject cut off by the frame on three sides."""
    im = Image.new("RGB", (256, 256), BG)
    ImageDraw.Draw(im).rectangle([0, 0, 255, 180], fill=(40, 90, 160))
    return im


def test_prepare_with_normalisation_still_refuses_a_subject_cropped_by_the_source_frame(
    tmp_path,
):
    """The 2026-10-14 audit, finding pipelines-image-02: the report was measured
    off the recentred canvas, which has a margin by construction, so a subject
    the frame cut off re-measured clean and went to TRELLIS."""
    src = tmp_path / "ref.png"
    _cropped_subject().save(src)
    off = reference.prepare(src, tmp_path / "off.png", enabled=False)
    assert not off.ok and "edge" in off.codes
    on = reference.prepare(src, tmp_path / "on.png", enabled=True)
    assert on.normalised
    assert not on.ok
    assert "edge" in on.codes
    assert len(on.codes) == len(on.reasons)
    assert any("edge of the frame" in r for r in on.reasons)


def test_normalisation_still_rescues_a_small_subject_that_the_frame_did_not_cut():
    """What the option is for: a small, well-framed subject is enlarged and passes,
    so only the refusal the canvas hides is carried over."""
    small = Image.new("RGB", (256, 256), BG)
    ImageDraw.Draw(small).rectangle([100, 100, 130, 130], fill=(40, 90, 160))
    assert not reference.measure(small).ok
    _out, report = reference.normalise(small)
    assert report.ok, report.reasons


# --- pipelines-13: an albedo that cannot be read back --------------------------


def _glb(path: Path, texture: dict | None) -> None:
    from realmspinner.kernels.geom3d import glbio

    gltf: dict = {"asset": {"version": "2.0"}}
    if texture is not None:
        gltf["textures"] = [texture]
        gltf["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
    header = struct.pack("<III", glbio.GLB_MAGIC, 2, 0)
    path.write_bytes(glbio.rebuild_glb(header, gltf, b""))


def test_a_retexture_of_a_mesh_whose_albedo_cannot_be_read_back_says_so_instead_of_greying_the_unseen_texels(  # noqa: E501
    tmp_path,
):
    """The 2026-10-14 audit, finding pipelines-mesh-02: an ``EXT_texture_webp``-only
    base returned False exactly as an untextured mesh does, so the unseen ~63% of
    the atlas became mid grey under a job that still said done."""
    webp_only = tmp_path / "webp.glb"
    _glb(webp_only, {"extensions": {"EXT_texture_webp": {"source": 0}}})
    with pytest.raises(retexture.BaseColourUnreadable, match="albedo"):
        retexture.require_base_colour(webp_only, tmp_path / "base.png", size=8)

    bare = tmp_path / "bare.glb"
    _glb(bare, None)
    assert retexture.require_base_colour(bare, tmp_path / "b2.png", size=8) is False


def test_the_retexture_stage_goes_through_the_refusing_reader():
    text = _source("_q_sprite.py")
    assert "retexture.require_base_colour" in text
    assert "retexture.extract_base_colour" not in text


# --- poser-23 and poser-25: spritesynth ----------------------------------------


def test_no_spritesynth_reference_guide_function_is_unreachable_from_a_producer():
    """The T-pose aliases had no caller anywhere ("Kept for its callers" was
    untrue); the two reference-guide functions stay, for a reroll of a reference
    row made before the producer was removed, and say so."""
    assert not hasattr(spritesynth, "load_tpose_guide")
    assert not hasattr(spritesynth, "render_tpose_guide")
    text = _source("pipelines/spritesynth.py")
    assert "Kept for its callers" not in text
    assert "Retained for rerolls" in text
    # What does remain has a producer in the worker.
    assert "spritesynth.render_reference_guide" in _source("_q_generate.py")


def test_spritesynth_cast_playback_matches_the_shipped_cast_clip():
    import json

    clips = sorted((SRC / "templates" / "clips").glob("*.json"))
    assert len(clips) >= 4
    ms = spritesynth.ACTION_PLAYBACK["cast"][1]
    for path in clips:
        data = json.loads(path.read_text(encoding="utf-8"))
        cast = next(c for c in data["clips"] if c["name"] == "cast")
        assert cast["duration_ms"] == ms, path.name
    text = _source("pipelines/spritesynth.py")
    assert "is the only clip file that ships" not in text
    assert "``hurt`` and ``cast`` have no clips" not in text


# --- inker-80: the retired sheet helpers' prose ---------------------------------
#
# The marker test that stood here ("Reference only" in three modules) went when C2
# deleted the helpers themselves; see test_audit_2026_10_14_c2_dead_chains.py.


def test_the_restyle_and_sheet_docstrings_no_longer_describe_a_live_pipeline():
    assert "Turning a rendered sprite sheet into a pixel-art sprite sheet" not in _source(
        "pipelines/pixelsheet.py"
    )
