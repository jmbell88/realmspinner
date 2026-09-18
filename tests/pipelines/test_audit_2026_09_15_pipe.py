"""Regression tests for the 2026-09-15 audit's pipelines findings.

Five independent findings, five independent tests, grouped here because they
share no code -- the installer, the release gate, Familiar's chat client, the
image pipeline's LoRA loading and the update checker.
"""

from __future__ import annotations

import importlib.util
import io
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from warlock import models
from warlock.familiar import contract, llama_client
from warlock.pipelines import download, update_worker
from warlock.pipelines.text2image import Text2Image

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# pipelines-01 -- installer upgrade leaves the old trellis engine behind
# ---------------------------------------------------------------------------


def test_upgrade_install_delete_removes_the_previous_versions_staged_trellis_engine() -> None:
    """``build.ps1`` stopped staging ``vendor\\trellis`` on 2026-09-10 (the
    engine is a Settings -> Models download now), but ``[InstallDelete]`` was
    never told -- so an in-place upgrade from <=0.0.41 left the old 838 MB
    engine sitting in ``{app}\\vendor\\trellis`` forever. Worse than dead
    weight: ``Config.resolve_trellis_exe`` falls back to exactly that path
    when no downloaded engine is present, so the stale, unpinned engine from
    the previous release kept being *used*, not just wasting disk."""
    iss = (ROOT / "installer" / "warlock.iss").read_text(encoding="utf-8")
    # Section-header-aware, not a plain str.split on "[Files]" -- a comment a
    # few lines into [InstallDelete] itself reads "...replaced in place by
    # [Files]." and a naive split truncates there, well before the real
    # [Files] section header (and before the line under test).
    sections = re.split(r"(?m)^\[(\w+)\]\s*$", iss)
    install_delete = sections[sections.index("InstallDelete") + 1]
    assert r'Type: filesandordirs; Name: "{app}\vendor\trellis"' in install_delete


# ---------------------------------------------------------------------------
# pipelines-02 -- release gate does not check uv.lock's own warlock version
# ---------------------------------------------------------------------------


def _load_preflight():
    spec = importlib.util.spec_from_file_location(
        "warlock_preflight_audit_2026_09_15", ROOT / "scripts" / "preflight.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_release_tree(tmp_path: Path, *, version: str, uv_lock_version: str) -> None:
    (tmp_path / "src" / "warlock").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(f'version = "{version}"\n', encoding="utf-8")
    (tmp_path / "src" / "warlock" / "__init__.py").write_text(
        f'__version__ = "{version}"\n', encoding="utf-8"
    )
    (tmp_path / "CHANGELOG.md").write_text(f"## {version}\n\n- notes\n", encoding="utf-8")
    (tmp_path / "INSTALL.md").write_text(
        f"Download `WarlockSetup-v{version}.exe` from Releases.\n", encoding="utf-8"
    )
    (tmp_path / "uv.lock").write_text(
        "[[package]]\n"
        'name = "warlock"\n'
        f'version = "{uv_lock_version}"\n'
        'source = { editable = "." }\n',
        encoding="utf-8",
    )


def test_check_versions_catches_a_stale_uv_lock_version(tmp_path: Path, capsys) -> None:
    """CLAUDE.md defines a release as five files in lockstep, but
    ``check_versions`` only ever compared four -- a release commit that bumped
    ``pyproject.toml``/``__init__.py``/``CHANGELOG.md``/``INSTALL.md`` and
    forgot to run ``uv sync`` afterwards would pass this gate with a stale
    lockfile."""
    module = _load_preflight()
    _write_release_tree(tmp_path, version="0.0.50", uv_lock_version="0.0.49")
    module.ROOT = tmp_path

    assert module.check_versions() is False
    out = capsys.readouterr().out
    assert "uv.lock" in out


def test_check_versions_passes_when_uv_lock_agrees(tmp_path: Path) -> None:
    module = _load_preflight()
    _write_release_tree(tmp_path, version="0.0.50", uv_lock_version="0.0.50")
    module.ROOT = tmp_path

    assert module.check_versions() is True


# ---------------------------------------------------------------------------
# pipelines-03 -- Familiar's llama client buffered the whole reply before
# checking MAX_RESPONSE_BYTES
# ---------------------------------------------------------------------------


class _CountingStream(httpx.AsyncByteStream):
    """A response body delivered as separate chunks, counting how many of
    them the client actually consumed -- the only way to tell "streamed and
    aborted early" from "read in full, then measured" from outside."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.yielded = 0

    async def __aiter__(self):
        for chunk in self._chunks:
            self.yielded += 1
            yield chunk

    async def aclose(self) -> None:  # pragma: no cover -- nothing to release
        pass


class _StreamingTransport(httpx.AsyncBaseTransport):
    def __init__(self, stream: _CountingStream) -> None:
        self._stream = stream

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=self._stream)


class _FakeServer:
    def __init__(self, key_path: Path) -> None:
        self._key_path = key_path

    @property
    def base_url(self) -> str:
        return "http://127.0.0.1:9999"

    @property
    def key_path(self) -> Path:
        return self._key_path

    async def ensure_started(self, *, expected_card_sha: str | None = None) -> None:
        return None

    def touch(self) -> None:
        return None


async def test_chat_refuses_an_oversized_reply_before_it_is_fully_buffered(
    monkeypatch, tmp_path: Path
) -> None:
    """``MAX_RESPONSE_BYTES`` used to be checked with ``len(response.content)``
    after a plain ``client.post`` -- by the time that ran, httpx had already
    read the whole body into memory, exactly the risk this module's own
    docstring says it avoids by streaming (parity with ``trellis.generate``).

    Proven here by chunk-counting: three chunks are queued, each over a third
    of the (monkeypatched, small) ceiling, so the ceiling is crossed partway
    through the second chunk. Streamed-and-aborted code must never ask the
    transport for the third chunk; buffer-then-check code (``client.post``,
    which internally calls ``aread()``) always consumes every chunk before
    any length check runs.
    """
    monkeypatch.setattr(llama_client, "MAX_RESPONSE_BYTES", 100)
    key_path = tmp_path / "familiar.key"
    key_path.write_text("k", encoding="utf-8")
    server = _FakeServer(key_path)
    chunk = b"x" * 60  # 3 chunks x 60 bytes = 180 bytes, over the 100-byte cap
    stream = _CountingStream([chunk, chunk, chunk])
    transport = _StreamingTransport(stream)

    with pytest.raises(RuntimeError, match="byte ceiling"):
        await llama_client.chat(
            server,
            [{"role": "user", "content": "hi"}],
            slot=0,
            sampling=contract.SAMPLING["chat"],
            transport=transport,
        )

    assert stream.yielded < 3, (
        "the client consumed every queued chunk -- it buffered the full reply "
        "before checking the ceiling instead of aborting mid-stream"
    )


# ---------------------------------------------------------------------------
# pipelines-04 -- a directory where a LoRA file belongs read as "present"
# ---------------------------------------------------------------------------


class _FakePipe:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    def load_lora_weights(self, _dir: Any, weight_name: str, adapter_name: str, **_kw: Any) -> None:
        self.loaded.append(adapter_name)


def test_a_directory_where_the_base_lora_file_belongs_is_reported_missing(tmp_path: Path) -> None:
    """``_load_loras`` and ``_ensure_adapter`` used ``path.exists()``, which is
    True for a directory -- a partial/broken download that left a directory
    where the LoRA file belongs read as "present" and would fall through into
    ``pipe.load_lora_weights`` (and, with a real diffusers pipe, its own
    low-level traceback) instead of this module's clear remedy message. The
    three sibling checks fixed by pipelines-06 (2026-09-11 audit) were already
    ``is_file()``; these two were the ones it missed."""
    base_spec = models.BASE_MODELS["sdxl"]
    assert base_spec.base_lora is not None
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    (loras_dir / base_spec.base_lora).mkdir()  # a directory, not the weights file

    t2i = Text2Image(base_spec, tmp_path)
    pipe = _FakePipe()

    with pytest.raises(RuntimeError, match="requires"):
        t2i._load_loras(pipe)

    style_key = "render3d"
    style_spec = models.STYLE_LORAS[style_key]
    (loras_dir / style_spec.filename).mkdir()

    with pytest.raises(RuntimeError, match="not downloaded"):
        t2i._ensure_adapter(pipe, style_key)


# ---------------------------------------------------------------------------
# pipelines-05 -- the update checker read a JSON response with no size ceiling
# ---------------------------------------------------------------------------


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False


def test_check_refuses_a_release_feed_response_over_the_manifest_size_ceiling(
    monkeypatch,
) -> None:
    """``_get_json`` used ``response.read()`` with no limit -- a compromised
    or misconfigured feed host answering ``/releases/latest`` (or the
    ``update-manifest.json`` asset it points at) with an unbounded body would
    have this process buffer all of it before ``json.loads`` ever got a
    chance to reject it. Same host-exhaustion shape ``trellis.py``'s
    ``MAX_GLB_BYTES`` and ``llama_client.py``'s ``MAX_RESPONSE_BYTES`` already
    guard against."""
    oversized = b"[" + b"1," * update_worker.MAX_MANIFEST_BYTES + b"1]"

    def fake_open_url(url: str, *, timeout: float | None = None) -> _Response:
        return _Response(oversized)

    monkeypatch.setattr(download, "open_url", fake_open_url)

    with pytest.raises(ValueError, match="more than"):
        update_worker.check({})
