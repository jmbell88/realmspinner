"""The 2026-09-23 (second run) audit, findings pipelines-04 and pipelines-05.

pipelines-04: ``Text2ImageClient.close()``/``MusicClient.close()`` read
``self._proc`` with no lock (by design -- see each ``close()``'s own
docstring), but ``_start_child`` used to assign it only *after* ``Popen``,
``winjob.assign``/``track`` and the reader thread were all set up. A close()
landing in that window found ``self._proc`` still ``None``, killed nothing,
and then blocked on ``self._lock`` for however long the rest of the spawn
(and ``READY_TIMEOUT``) took. ``self._proc`` is now published the instant
``Popen`` returns.

pipelines-05: ``Text2Image._generate``'s conditioning-on-non-SDXL refusal ran
only inside ``_conditioned``/``_attach_conditioning``, which this method does
not reach until after ``self.load(on_state)`` has already paid for a full
checkpoint load -- the same shape pipelines-07 (2026-09-14) fixed for the
``tile`` refusal beside it.
"""

from __future__ import annotations

import contextlib
import io
import threading
import time

import pytest

from realmspinner import models
from realmspinner.pipelines import music_client, t2i_client, text2image

# --- pipelines-04 -------------------------------------------------------------


class _FakeProc:
    """A ``subprocess.Popen``-shaped double. Never a real process."""

    def __init__(self) -> None:
        self.pid = 999_999
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.killed = False

    def poll(self):
        return 1 if self.killed else None

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None):
        return 1


def _spawn_race(monkeypatch, module, make_client, spec_and_dir):
    """Start a spawn, pause it mid-flight inside ``winjob.assign``, and prove
    ``close()`` kills the fake child without waiting for the spawn to finish.

    Shared by the t2i and music clients: same bug, same fix, same shape.
    """
    fake = _FakeProc()
    winjob_reached = threading.Event()
    release = threading.Event()

    def fake_assign(pid):
        winjob_reached.set()
        # Bounded so a broken fix (or a broken test) cannot hang the suite --
        # not a race-timing dependency of the assertion itself.
        release.wait(timeout=10)

    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: fake)
    monkeypatch.setattr(module.winjob, "assign", fake_assign)
    monkeypatch.setattr(module.winjob, "track", lambda pid, label: None)
    monkeypatch.setattr(module.winjob, "untrack", lambda pid: None)

    client = make_client(*spec_and_dir)

    def spawner():
        # The fake child "exits" once release() lets it through.
        with client._lock, contextlib.suppress(Exception):
            client._start_child()

    spawn_thread = threading.Thread(target=spawner, daemon=True)
    spawn_thread.start()
    try:
        assert winjob_reached.wait(timeout=5), "winjob.assign was never reached"
        # Give the spawner a moment to settle into release.wait() -- the
        # window this finding is about.
        time.sleep(0.1)

        closer = threading.Thread(target=client.close, daemon=True)
        closer.start()
        try:
            # close()'s kill of an already-published proc does not need the
            # spawn to finish -- only its own unlocked read plus proc.kill().
            deadline = time.monotonic() + 5.0
            while not fake.killed and time.monotonic() < deadline:
                time.sleep(0.01)
            assert fake.killed, (
                "close() did not kill the child that was still mid-spawn -- "
                "self._proc was not published until after winjob.assign"
            )
        finally:
            release.set()
            closer.join(timeout=5)
    finally:
        release.set()
        spawn_thread.join(timeout=5)


def test_t2i_client_close_kills_a_child_that_is_still_mid_spawn(monkeypatch, tmp_path):
    _spawn_race(
        monkeypatch,
        t2i_client,
        t2i_client.Text2ImageClient,
        (models.BASE_MODELS["sdxl_cfg"], tmp_path),
    )


def test_music_client_close_kills_a_child_that_is_still_mid_spawn(monkeypatch, tmp_path):
    _spawn_race(
        monkeypatch,
        music_client,
        music_client.MusicClient,
        (models.MUSIC_MODELS["ace_step_v1"], tmp_path / "ace-step"),
    )


# --- pipelines-05 --------------------------------------------------------------


@pytest.fixture
def flux_t2i(tmp_path):
    # A non-SDXL family, and no weights anywhere under tmp_path: if
    # ``self.load()`` ran before the refusal, it would fail on the missing
    # ``model_index.json`` instead -- a different exception, proving the
    # ordering rather than just the outcome.
    return text2image.Text2Image(models.BASE_MODELS["flux_klein"], tmp_path)


def test_conditioning_on_a_non_sdxl_family_is_refused_before_the_checkpoint_loads(
    flux_t2i, tmp_path
):
    with pytest.raises(RuntimeError, match="cannot take conditioning"):
        flux_t2i._generate(
            "a prompt",
            tmp_path / "out.png",
            conditioning={"control": "depth"},
        )
    # Never reached load(): no pipe was ever assigned.
    assert flux_t2i._pipe is None
