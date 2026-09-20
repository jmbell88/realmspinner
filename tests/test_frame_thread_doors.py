"""Nothing expensive runs on the frame thread -- the doors the review named.

The 2026-09-02 review's theme T2 listed nine places where a decode, an encode
or a batch of inserts ran on the pygame frame thread: the Review sweep launch
and its mesh loads, the reference PNG on job completion, the Clay crash
recovery, the Troupe atlas, the Inker revert reload, the Packwright save
encodes, the Sirens sample re-encode per snapshot, and the settings flush
during a splitter drag. Each had a sibling in the tree that already did it
right (the GLB parse/adopt split, the ``inker-recover`` task), and each is
now that shape.

**Muse joined the list on 2026-09-05**, for two doors the 2026-09-02 sweep
never reached because Muse did not exist for it: ``muse_io.export_loop`` and
``export_with_points`` ran their crossfade blend and their whole WAV byte
encode before ``ctx.submit`` was ever called (muse-02), and
``muse_mode._play_from`` paid for that same blend on the frame thread on
exactly the press that follows a fresh region or crossfade, because
``loop_body``'s cache always misses on the params that just changed
(muse-03). Both are now the same shape as the rest of this file: the compute
moved inside what ``ctx.submit`` runs, either directly (the export) or as a
precompute fired wherever a region or crossfade *settles* so the next call is
a cache hit (the player).

The guard here is behavioural where it can be: a ctx whose ``submit`` runs
the task on a *real worker thread* and joins, with the expensive function
spied to record which thread it ran on. A regression that moves the decode
back into the frame-thread half shows up as the test thread's name. The
Review launch and mesh load are pinned in ``tests/modes/review/test_review_mode.py``.
"""

from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from modes.clay.test_clay_mode import FakeCtx as ClayCtx
from modes.clay.test_clay_mode import _tab as clay_tab
from modes.inker.test_inker_mode import _PaletteCtx
from modes.sirens.test_sirens_mode import FakeCtx as SirensCtx
from modes.sirens.test_sirens_mode import _tab as sirens_tab
from PIL import Image

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.mesh import document as clay_document
from realmspinner.kernels.mesh import serialize as clay_serialize
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker.state import InkerDoc
from realmspinner.studio.modes.muse import fileio as muse_io
from realmspinner.studio.modes.muse import mode as muse_mode
from realmspinner.studio.modes.muse import state as muse_state
from realmspinner.studio.modes.packwright import fileio as packwright_io
from realmspinner.studio.modes.packwright.engine import rpack
from realmspinner.studio.modes.packwright.engine.document import PackDoc
from realmspinner.studio.modes.packwright.engine.sources import Sprite
from realmspinner.studio.modes.poser import mode as poser_mode
from realmspinner.studio.modes.sirens.engine import rsng
from realmspinner.studio.viewer_embed import Viewer

WORKER = "realmspinner-task-test"


@dataclass
class _Done:
    key: str
    result: Any = None
    tag: Any = None
    error: Any = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


class _Threaded:
    """``submit`` on a real worker thread, joined -- so the test sees the task
    half run where it would run, and the recorder below can name the thread."""

    submitted: list[str]
    tags: list[Any]
    result: Any

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        self.submitted.append(key)
        self.tags.append(tag)
        box: dict[str, Any] = {}

        def go() -> None:
            try:
                box["result"] = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
                box["error"] = exc

        worker = threading.Thread(target=go, name=WORKER)
        worker.start()
        worker.join()
        if "error" in box:
            raise box["error"]
        self.result = box["result"]
        return True


def _spy(monkeypatch: pytest.MonkeyPatch, owner: Any, name: str) -> list[str]:
    """Wrap ``owner.name`` to record the thread each call ran on."""
    real = getattr(owner, name)
    threads: list[str] = []

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        threads.append(threading.current_thread().name)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapped)
    return threads


class _GL:
    """A texture factory: the one GL call the frame-thread half makes."""

    NEAREST = 0
    LINEAR = 1

    def __init__(self) -> None:
        self.uploads: list[tuple[tuple[int, int], int]] = []

    def texture(self, size: Any, components: int, data: bytes) -> Any:
        assert len(data) == size[0] * size[1] * components
        self.uploads.append((tuple(size), len(data)))
        return SimpleNamespace(size=tuple(size), filter=None, release=lambda: None)


def _png(path: Path, size: tuple[int, int] = (8, 4)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, (10, 20, 30, 255)).save(path)
    return path


# --- 1. the reference picture on job completion -------------------------------


def test_the_reference_picture_is_decoded_apart_from_its_upload(tmp_path):
    parsed = Viewer.parse_reference(_png(tmp_path / "input.png"))
    assert parsed[0] == (8, 4) and len(parsed[1]) == 8 * 4 * 4

    viewer = object.__new__(Viewer)
    viewer.ctx = _GL()
    viewer.reference = None
    viewer._forget = lambda texture: None
    viewer.adopt_reference(parsed)
    assert viewer.reference.size == (8, 4)
    assert viewer.reference.filter == (_GL.LINEAR, _GL.LINEAR)


def test_the_viewer_sync_submits_the_decode_and_adopts_it_on_landing():
    """The timer path never calls the blocking ``load_reference``."""
    from realmspinner.studio import main

    sync = inspect.getsource(main.App._sync_viewer)
    assert "parse_reference" in sync and "load_reference(" not in sync
    adopt = inspect.getsource(main.App._adopt_model)
    assert "adopt_reference" in adopt


def test_the_character_preview_submits_the_parse_and_adopts_it_on_landing():
    """The tenth door, and the same shape as the first nine.

    The 2026-09-11 audit (finding create-02) found "Preview character"'s
    landing calling ``viewer.load_model`` -- a glTF parse plus a per-slot PNG
    decode -- inline in ``_on_task_done``, which is the T2 bug the 2026-09-02
    review had already fixed for the reference picture and the mesh. This file
    is the guard for that class and did not name this door, so the fix arrived
    with nothing standing behind it.
    """
    from realmspinner.studio import main

    dispatch = inspect.getsource(main.App._dispatch_character_preview)
    assert "parse_model" in dispatch and "load_model(" not in dispatch
    adopt = inspect.getsource(main.App._adopt_character_preview)
    assert "adopt_model" in adopt


# --- 2. the character-sheet atlas (Troupe's own, folded into Poser by P9) ---


class _PoserSheetCtx(_Threaded):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.state = SimpleNamespace(poser=None, preview={}, mode="poser")
        self.viewer = SimpleNamespace(ctx=_GL())
        self.submitted, self.tags, self.result = [], [], None
        self.toasts: list[tuple[str, str]] = []

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def busy(self, key: str) -> bool:
        return False

    def toast(self, text: str, level: str = "info", *a: Any, **k: Any) -> None:
        self.toasts.append((text, level))


def _sheet(ctx: _PoserSheetCtx) -> tuple[str, str]:
    from realmspinner.kernels.rig import store

    state = poser_mode.ensure(ctx)
    state.job_id, state.sheet_id = "job1", store.new_id()
    _png(store.sheet_png_path(ctx.job_dir("job1"), state.sheet_id), (16, 8))
    return "job1", state.sheet_id


def test_the_sheet_atlas_is_decoded_on_a_task_and_uploaded_when_it_lands(tmp_path, monkeypatch):
    ctx = _PoserSheetCtx(tmp_path)
    key = _sheet(ctx)
    threads = _spy(monkeypatch, poser_mode, "_decode_atlas")

    assert poser_mode.atlas_texture(ctx) is None, "not yet: the decode is in flight"
    assert ctx.submitted == [poser_mode.atlas_key(*key)]
    assert threads == [WORKER]
    assert ctx.viewer.ctx.uploads == [], "nothing touched GL on the frame thread"

    poser_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result, tag=key))
    texture = poser_mode.atlas_texture(ctx)
    assert texture is not None and texture.size == (16, 8)
    assert ctx.viewer.ctx.uploads == [((16, 8), 16 * 8 * 4)]
    assert len(ctx.submitted) == 1, "cached: the second ask decodes nothing"


def test_a_decoded_atlas_for_a_sheet_no_longer_on_screen_is_dropped(tmp_path):
    from realmspinner.kernels.rig import store

    ctx = _PoserSheetCtx(tmp_path)
    key = _sheet(ctx)
    poser_mode.atlas_texture(ctx)
    poser_mode.ensure(ctx).sheet_id = store.new_id()
    poser_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result, tag=key))
    assert ctx.viewer.ctx.uploads == []
    assert "troupe_texture" not in ctx.state.preview


def test_an_unreadable_atlas_is_tried_once(tmp_path):
    ctx = _PoserSheetCtx(tmp_path)
    key = _sheet(ctx)
    poser_mode.atlas_texture(ctx)
    poser_mode.on_task_failed(ctx, _Done(ctx.submitted[-1], tag=key, error=OSError("bad png")))
    for _ in range(3):
        assert poser_mode.atlas_texture(ctx) is None
    assert len(ctx.submitted) == 1


# --- 3. the Clay crash recovery -----------------------------------------------


class _ClayCtx(_Threaded, ClayCtx):
    def __init__(self) -> None:
        ClayCtx.__init__(self)
        self.tags: list[Any] = []


def test_a_recovered_clay_model_is_read_on_a_task_and_adopted_dirty(tmp_path, monkeypatch):
    ctx = _ClayCtx()
    authored = clay_tab(ctx).doc
    path = tmp_path / "crate.rblk"
    path.write_bytes(clay_serialize.rblk_bytes(authored))
    threads = _spy(monkeypatch, clay_serialize, "read_rblk")

    assert clay_mode._journal_adopt(ctx, path, {"title": "Crate"}) is True
    assert ctx.submitted[-1].startswith("clay-recover:")
    assert threads == [WORKER]

    clay_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))
    tab = next(t for t in ctx.state.clay.docs if t.title == "Crate (recovered)")
    assert tab.path is None
    assert tab.saved_head == -1 and tab.dirty
    assert tab.journal_name == path.name


def test_a_recovered_clay_model_that_will_not_parse_says_so(tmp_path):
    """Through ``journal.adopt_failed`` since 2026-09-05 -- a warning with the
    log behind it, the sentence every provider says -- where a raise here
    arrived as an *error* toast no other mode's copy raised."""
    path = tmp_path / "bad.rblk"
    path.write_bytes(b"not a zip")
    assert clay_mode._load_recovery(path, {}) is None


def test_clay_save_to_encodes_off_the_frame_thread(tmp_path, monkeypatch):
    """clay-03 (2026-09-06 audit): ``save_to`` called ``serialize.rblk_bytes``
    -- the zip-and-PNG encode this module's own docstring says never runs on
    the frame thread -- directly on the calling thread, before ``ctx.submit``
    was ever reached; only ``atomic.write_bytes`` was inside the submitted
    closure. Same shape as ``test_a_packwright_save_encodes_its_pngs_on_the_task``
    above, for the sibling that already does this split correctly.
    """
    ctx = _ClayCtx()
    tab = clay_tab(ctx)
    threads = _spy(monkeypatch, clay_serialize, "snapshot_bytes")
    out = tmp_path / "scene.rblk"

    clay_mode.save_to(ctx, tab, out)
    assert ctx.submitted == [f"clay-save:{tab.uid}"]
    assert threads == [WORKER]
    assert clay_serialize.read_rblk(out.read_bytes()).objects[0].name == "Box"


def test_clay_save_as_encodes_off_the_frame_thread(tmp_path, monkeypatch):
    """The picker half of clay-03: ``save_as`` built the snapshot on the frame
    thread (correctly, for the reason its own docstring gives -- the picker is
    unbounded and modal) but then encoded it there too, before handing the
    result to the picker's own task closure."""
    ctx = _ClayCtx()
    tab = clay_tab(ctx)
    out = tmp_path / "scene.rblk"
    monkeypatch.setattr(clay_mode.dialogs, "save_file", lambda *a, **k: out)
    threads = _spy(monkeypatch, clay_serialize, "snapshot_bytes")

    clay_mode.save_as(ctx, tab)
    assert ctx.submitted == [f"clay-saveas:{tab.uid}"]
    assert threads == [WORKER]
    assert clay_serialize.read_rblk(out.read_bytes()).objects[0].name == "Box"


def test_clay_export_asset_encodes_off_the_frame_thread(svc, monkeypatch):
    """The other half of clay-03: ``export_asset`` also built its GLB
    (``glbwrite.write_glb``) on the calling thread, ahead of ``rblk_bytes``.
    Both encodes now run inside ``run()``, against a real service so the job
    it mints and the sidecar it writes are checked as well as the thread."""
    from realmspinner.kernels.geom3d import glbwrite
    from realmspinner.service import files as svc_files

    ctx = _ClayCtx()
    ctx.svc = svc
    tab = clay_tab(ctx)
    glb_threads = _spy(monkeypatch, glbwrite, "write_glb")
    rblk_threads = _spy(monkeypatch, clay_serialize, "snapshot_bytes")

    clay_mode.export_asset(ctx, tab)
    assert ctx.submitted == [f"clay-export:{tab.uid}"]
    assert glb_threads == [WORKER]
    assert rblk_threads == [WORKER]
    job_id = ctx.result["job_id"]
    assert svc_files.clay_source_status(svc, job_id)["exists"] is True


def test_the_clay_uid_counter_survives_a_reserve_from_a_task_thread():
    """``reserve_uid`` swaps the counter out from under ``new_uid``; both hold
    the lock now, and a burst from both sides mints no duplicate."""
    minted: list[int] = []
    stop = threading.Event()

    def reserve_loop() -> None:
        base = clay_document.new_uid()
        while not stop.is_set():
            clay_document.reserve_uid(base + 5)

    worker = threading.Thread(target=reserve_loop, name=WORKER)
    worker.start()
    try:
        for _ in range(2000):
            minted.append(clay_document.new_uid())
    finally:
        stop.set()
        worker.join()
    assert len(set(minted)) == len(minted)
    assert minted == sorted(minted), "monotonic through every reserve"


# --- 4. the Inker revert reload ----------------------------------------------


class _InkerCtx(_Threaded, _PaletteCtx):
    def __init__(self) -> None:
        _PaletteCtx.__init__(self)
        self.tags: list[Any] = []
        self.svc = None
        self.viewer = None
        self.cache = SimpleNamespace(invalidate=lambda: None)
        self.confirms = SimpleNamespace(pending=None)
        self.confirms.ask = lambda confirm: setattr(self.confirms, "pending", confirm)


def test_a_revert_decodes_the_restored_image_on_the_task(tmp_path, monkeypatch):
    from realmspinner.service import files as svc_files

    monkeypatch.setattr(svc_files, "revert_reference", lambda svc, job_id: None)
    monkeypatch.setattr(svc_files, "discard_inker_working", lambda svc, job_id: None)
    ctx = _InkerCtx()
    path = _png(tmp_path / "input.png", (6, 5))
    doc = inker.Document.blank(4, 4)
    tab = InkerDoc(doc=doc, title="input.png", path=path, saved_head=doc.history.head)
    tab.job_id, tab.has_original = "job7", True
    inker_mode.ensure(ctx).add(tab)
    threads = _spy(monkeypatch, inker.Document, "load")

    inker_mode.revert(ctx, tab)
    ctx.confirms.pending.on_confirm()
    assert ctx.submitted[-1] == f"inker-revert:{tab.uid}"
    assert threads == [WORKER]
    assert ctx.result["doc"] is not None and tab.doc is doc, "adopted on landing, not here"

    inker_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))
    assert tab.doc is not doc
    assert tab.doc.size == (6, 5)
    assert tab.has_original is False
    assert ctx.toasts[-1][0] == "Back to the original image."


def test_a_revert_whose_image_will_not_reopen_still_reports_the_revert(tmp_path, monkeypatch):
    from realmspinner.service import files as svc_files

    monkeypatch.setattr(svc_files, "revert_reference", lambda svc, job_id: None)
    monkeypatch.setattr(svc_files, "discard_inker_working", lambda svc, job_id: None)
    ctx = _InkerCtx()
    path = tmp_path / "input.png"
    path.write_bytes(b"not a png")
    doc = inker.Document.blank(4, 4)
    tab = InkerDoc(doc=doc, title="input.png", path=path, saved_head=doc.history.head)
    tab.job_id, tab.has_original = "job7", True
    inker_mode.ensure(ctx).add(tab)

    inker_mode.revert(ctx, tab)
    ctx.confirms.pending.on_confirm()
    assert ctx.result["reverted"] is True and ctx.result["doc"] is None
    inker_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))
    assert tab.doc is doc
    assert "could not be reopened" in ctx.toasts[-1][0]
    assert ctx.toasts[-1][1] == "error"


# --- 5. the Packwright save --------------------------------------------------


def _sprite(key: str) -> Sprite:
    pixels = np.zeros((6, 8, 4), dtype=np.uint8)
    pixels[1:-1, 1:-1] = (int(key[-1]) * 20, 40, 60, 255)
    return Sprite(key=key, name=f"name-{key}", pixels=pixels)


def _pack_doc() -> PackDoc:
    doc = PackDoc()
    for i in range(3):
        doc.add_source(_sprite(f"s{i}"))
    doc.mark_saved()
    return doc


def test_a_snapshot_is_what_the_document_was_when_the_save_was_pressed():
    """The document goes on being edited while the task encodes."""
    doc = _pack_doc()
    before = rpack.rpack_bytes(doc)
    snap = rpack.snapshot(doc)
    doc.add_source(_sprite("s9"))
    doc.rename_source(doc.sources[0].uid, "renamed")
    assert rpack.snapshot_bytes(snap) == before
    assert len(rpack.read_rpack(rpack.snapshot_bytes(snap)).sources) == 3


class _PackCtx(_Threaded):
    def __init__(self) -> None:
        self.submitted, self.tags, self.result = [], [], None


def test_packwright_crash_recovery_reads_on_a_task_and_adopts_on_landing():
    """The eleventh door. Clay's recovery is pinned above; Packwright's was not.

    The 2026-09-11 audit (finding packwright-01) found ``_journal_adopt``
    reading and decoding a recovered ``.rpack`` inline -- 137.7 ms for an
    ordinary 300-sprite atlas, and ``MAX_PACK_SOURCE_BYTES`` allows a great
    deal more -- because the Recover button calls ``journal.take`` straight
    from ``draw``. Clay's and Inker's providers already deferred; this one is
    now the same shape.
    """
    from realmspinner.studio.modes.packwright import mode as packwright_mode

    adopt = inspect.getsource(packwright_mode._journal_adopt)
    assert "submit" in adopt, "the recovery read must be handed to a task"
    assert "read_rpack" not in adopt, "the decode must not run in the provider"
    load = inspect.getsource(packwright_mode._load_recovery)
    assert "read_rpack" in load


def test_a_packwright_save_encodes_its_pngs_on_the_task(tmp_path, monkeypatch):
    threads = _spy(monkeypatch, rpack, "png_bytes")
    ctx = _PackCtx()
    tab = SimpleNamespace(doc=_pack_doc(), uid="t1", saving=False, title="atlas")
    out = tmp_path / "atlas.rpack"

    packwright_io.save_to(ctx, tab, out)
    assert ctx.submitted == ["packwright-save:t1"]
    assert threads and set(threads) == {WORKER}
    assert rpack.read_rpack(out.read_bytes()).sources[1].key == "s1"


# --- 6. the Sirens sample re-encode ------------------------------------------


def test_a_sample_is_encoded_to_wav_once_across_snapshots(monkeypatch):
    doc = sirens_tab(SirensCtx()).doc
    doc.set_sample("kick", np.linspace(-1.0, 1.0, 256, dtype=np.float32))
    reference = rsng.rsng_bytes(doc)
    encodes = _spy(monkeypatch, rsng.wavout, "wav_bytes")

    first = rsng.rsng_bytes(doc)
    second = rsng.rsng_bytes(doc)
    assert encodes == [], "already cached from the first snapshot"
    assert first == second == reference

    doc.set_sample("kick", np.zeros(64, dtype=np.float32))
    third = rsng.rsng_bytes(doc)
    assert len(encodes) == 1, "a replaced array is a new encode"
    assert third != first
    assert len(rsng.read_rsng(third).samples["kick"]) == 64


# --- 7. the settings flush under a drag ---------------------------------------


def test_the_settings_flush_waits_for_the_mouse_button_to_come_up():
    """``frame()`` -- and the guard it wraps this call in -- moved out of
    ``studio/main.py`` in the P4 restructure, into ``studio/shell/frame.py``.
    """
    from realmspinner.studio.shell import frame

    source = inspect.getsource(frame)
    guarded = "if not imgui.is_any_mouse_down():\n            self.app_ctx.settings.tick()"
    assert guarded in source
    assert source.count("settings.tick()") == 1


# --- 8/9. Muse's loop-cache precompute and export encode ---------------------


class _MuseCtx(_Threaded):
    def __init__(self) -> None:
        self.state = SimpleNamespace(muse=None)
        self.submitted, self.tags, self.result = [], [], None
        self.toasts: list[tuple[str, str]] = []

    def toast(self, text: str, level: str = "info", *a: Any, **k: Any) -> None:
        self.toasts.append((text, level))


class _MuseDevice:
    """``sirens_audio``, stubbed to the one call ``play_region`` needs."""

    def play(self, pcm: Any, rate: int = 44100, *, tag: str = "", loops: int = 0) -> bool:
        return True


def _muse_take(ctx: Any, seconds: float = 240.0) -> Any:
    pcm = (
        np.random.default_rng(0).standard_normal((int(seconds * 44100), 2)) * 5000
    ).astype(np.int16)
    state = muse_mode.ensure(ctx)
    state.player = muse_state.Player(job="take", pcm=pcm, rate=44100, duration=seconds)
    return state.player


def test_export_loop_does_not_block_the_frame_thread_encoding_a_240_second_take(
    monkeypatch, tmp_path
):
    """muse-02: the crossfade blend and the whole WAV byte encode used to run
    before ``ctx.submit`` was ever reached in ``export_loop``, so both ran on
    the frame thread on every press -- and again on every press, since nothing
    cached the encode itself. Both now live inside the closure ``_save`` hands
    to ``ctx.submit``.
    """
    ctx = _MuseCtx()
    one = _muse_take(ctx)
    muse_mode.set_region(ctx, 10.0, 230.0)
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: tmp_path / "loop.wav")
    blend = _spy(monkeypatch, muse_io.loops_mod, "crossfade")
    encode = _spy(monkeypatch, muse_io, "_wav")

    muse_io.export_loop(ctx, one)

    assert blend == [WORKER]
    assert encode == [WORKER]
    assert (tmp_path / "loop.wav").exists()


def test_playing_a_freshly_marked_loop_region_does_not_block_the_frame_thread(monkeypatch):
    """muse-03: ``loop_body``'s cache key is ``(start, end, fade)``, so it
    always missed on the very first Play after a region changed -- the press
    that follows ``Find loop points`` (``choose_candidate``, here) landing a
    result. ``choose_candidate`` now precomputes the body on a task, so
    ``play_region``'s own call into ``loop_body`` is a cache hit and never
    calls the blend itself.
    """
    from realmspinner.studio.modes.muse.engine.loops import Candidate

    ctx = _MuseCtx()
    monkeypatch.setattr(muse_mode, "sirens_audio", _MuseDevice())
    one = _muse_take(ctx)
    one.candidates = [Candidate(int(10.0 * 44100), int(230.0 * 44100), 0.9)]
    blend = _spy(monkeypatch, muse_io.loops_mod, "crossfade")

    muse_mode.choose_candidate(ctx, 0)
    assert blend == [WORKER], "the blend ran on the task when the region settled"
    # incident-2026-09-05b: the task only answers with a value now; landing
    # it is ``on_task_done``'s job, on the frame thread.
    muse_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))

    muse_mode.play_region(ctx)
    assert blend == [WORKER], "and play found it already cached -- no second call"


# --- 10. the library search widen ---------------------------------------------


def test_widen_for_filters_runs_the_store_query_off_the_frame_thread(monkeypatch):
    """shell-01 (the 2026-09-08 audit): ``JobsCache.widen_for_filters``, called
    unconditionally at the top of both Library panes' ``draw()``, ran
    ``self.svc.store.search_ids`` -- a real sqlite query behind ``JobStore``'s
    shared RLock -- directly on the frame thread on essentially every
    keystroke in the filter box: the exact stall :meth:`JobsCache.read`'s own
    ``request``/``adopt`` split exists to prevent for the ordinary list poll.
    ``request_widen`` now submits the query through ``TaskRunner`` the same
    way, and this is the sibling proof this file already keeps for every other
    door: a ``ctx`` whose ``submit`` runs the task on a real worker thread,
    with the store call spied to record which thread it ran on.
    """
    from realmspinner.studio import jobs_cache
    from realmspinner.studio.state import Filters

    svc = SimpleNamespace(store=SimpleNamespace(search_ids=lambda *a, **k: ["match-1"]))
    threads = _spy(monkeypatch, svc.store, "search_ids")
    cache = jobs_cache.JobsCache(svc)
    runner = _Threaded()
    runner.submitted, runner.tags, runner.result = [], [], None

    filters = Filters(text="dragon")
    submitted = cache.request_widen(filters, runner)

    assert submitted is True
    assert runner.submitted == [jobs_cache.SEARCH_KEY]
    assert threads == [WORKER], "search_ids must not run on the frame thread"
    # And nothing was merged into ``self.jobs`` yet -- that is the frame
    # thread's half, landing only once ``adopt_widen`` is handed the result,
    # exactly as ``read``/``adopt`` split for the ordinary list poll.
    assert cache.jobs == []


def test_a_stale_precompute_result_cannot_pair_its_key_with_a_newer_buffer(monkeypatch):
    """incident-2026-09-05b, a correction to muse-03: submitting
    ``muse_io.loop_body`` itself as the precompute task moved the blend off
    the frame thread, but ``loop_body`` also writes
    ``player.loop_cache``/``loop_cache_key`` -- two separate statements,
    straight onto the shared ``Player`` -- and it was doing that from the
    task thread. A ``_play_from`` call for a *different*, newer region,
    landing between those two statements, paired one key with the other's
    buffer:

        1. task (key A): loop_cache = A
        2. frame (key B): loop_cache = B
        3. frame (key B): loop_cache_key = B
        4. task (key A): loop_cache_key = A

    -- leaving the player claiming key A while holding B's buffer, which
    ``export_loop`` would then write to the file the user named. That is
    exactly the ordinary case of marking a region and pressing Play before a
    ~100 ms blend has finished.

    Deterministic, not timing-dependent: ``compute_loop_cache`` is called
    directly to get A's answer as a plain value (proving, first, that doing
    so touches nothing on ``one`` -- there is nothing left here to tear), a
    newer region B is set and precomputed for real in the ordinary way, and
    only then does A's stale answer arrive at ``on_task_done``. The claim
    this closes: the buffer the player ends up holding must match the key it
    claims to be.
    """
    ctx = _MuseCtx()
    one = _muse_take(ctx, seconds=100.0)
    muse_mode.set_region(ctx, 10.0, 90.0)
    a_key = muse_io.loop_cache_key(one)

    a_result = muse_io.compute_loop_cache(one)
    assert a_result[0] == a_key, "the pair returned is at least internally consistent"
    assert one.loop_cache is None and one.loop_cache_key is None, (
        "compute_loop_cache must not touch the player -- that is the whole "
        "fix: nothing but on_task_done, on the frame thread, may write the "
        "cache pair"
    )

    # The frame thread moves on before A's answer arrives: a newer region,
    # precomputed and landed in full -- a self-consistent pair, the one
    # write that installs both fields.
    muse_mode.set_region(ctx, 0.0, 40.0)
    muse_mode.precompute_loop(ctx)
    muse_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))
    b_key, b_buffer = ctx.result

    # A's stale-but-internally-consistent answer lands last.
    muse_mode.on_task_done(
        ctx, _Done(f"{muse_io.CACHE_PREFIX}{one.job}", a_result)
    )

    assert one.loop_cache_key == b_key, "a stale key must not overwrite a newer one"
    assert one.loop_cache is b_buffer, "nor may a stale buffer sit under a newer key"
    # The claim M09 exists to make, checked directly: whatever key the player
    # is claiming, the buffer underneath it must be that key's own blend.
    expected = muse_io.loops_mod.crossfade(one.pcm, *one.loop_cache_key)
    assert np.array_equal(one.loop_cache, expected)


# --- 11. Familiar's Clay build landing -----------------------------------


class _FamiliarCtx(_Threaded):
    """Enough of ``app_ctx.Ctx`` to land a Familiar Clay build -- the
    thread-checking twin of ``test_familiar_ui._FakeCtx``, which records a
    submitted closure rather than running one on a real, joined worker."""

    def __init__(self) -> None:
        from realmspinner.familiar import threads as familiar_threads_mod

        doc = clay_document.ClayDoc()
        tab = clay_mode.ClayTab(doc=doc)
        clay_state = clay_mode.ClayState(docs=[tab], active_uid=tab.uid)
        self.state = SimpleNamespace(clay=clay_state, mode="clay", familiar=None, preview={})
        self.tab = tab
        self.familiar_threads = familiar_threads_mod.Threads()
        self.clay_view = SimpleNamespace(cleared=0, previewed=None, grabbing=False)
        self.clay_view.clear_preview = lambda: setattr(
            self.clay_view, "cleared", self.clay_view.cleared + 1
        )
        self.clay_view.set_preview = lambda diff, scratch: setattr(
            self.clay_view, "previewed", (diff, scratch)
        )
        self.toasts: list[str] = []
        self.submitted, self.tags, self.result = [], [], None

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append(message)


def test_a_familiar_build_landing_runs_its_clay_batch_off_the_frame_thread(monkeypatch):
    """familiar-01 (the 2026-09-17 audit): a landed Familiar build ran its
    whole ``clay_batch`` -- up to ``agent_clay.BATCH_MAX`` = 32 calls,
    booleans included -- inline inside ``familiar_ui.on_task_done``, which
    ``App.frame`` calls on the pygame frame thread (624-705 ms wall,
    reproduced with 16 uv-sphere adds and 15 unions, no GPU or weights
    involved -- the app froze for ~40 frames on an ordinary Build).
    ``on_task_done`` now only clones the base document -- a numpy copy, not
    a batch of ops -- inline, and submits the batch itself under
    ``familiar_ui.LAND_KEY``; this is that class's own proof, the same shape
    every other door in this file already keeps: a ``ctx`` whose ``submit``
    runs the task on a real worker thread, with the expensive call spied to
    record which thread it ran on.
    """
    from realmspinner.studio.assistant import preview as familiar_preview
    from realmspinner.studio.assistant import ui as familiar_ui

    ctx = _FamiliarCtx()
    threads = _spy(monkeypatch, familiar_preview, "run_scratch")
    calls = [{"name": "clay_add_primitive", "arguments": {"generator": "box", "name": "crate"}}]
    done = _Done(
        key=familiar_ui.BUILD_KEY,
        result=calls,
        tag={"thread_key": ("clay", ctx.tab.uid), "tab_uid": ctx.tab.uid},
    )

    familiar_ui.on_task_done(ctx, done)

    assert ctx.submitted == [familiar_ui.LAND_KEY]
    assert threads == [WORKER], "clay_batch must not run on the frame thread"
    assert ctx.clay_view.previewed is None, "nothing landed yet -- only submitted"

    land_tag = ctx.tags[-1]
    familiar_ui.on_task_done(ctx, _Done(key=familiar_ui.LAND_KEY, result=ctx.result, tag=land_tag))

    ui = familiar_ui.ensure(ctx)
    assert ui.preview_calls == calls, "the batch's own result still lands as the ghost preview"
    assert ctx.clay_view.previewed is not None
