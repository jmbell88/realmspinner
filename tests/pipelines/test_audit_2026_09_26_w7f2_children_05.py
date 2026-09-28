"""Regression tests for the 2026-09-26 audit, finding pipelines-children-05.

The three URL-download loops (``fetch_worker._fetch_url``, ``update_worker.fetch``,
``pack_worker.collect``) each already knew a declared size -- the registry's
``size_gib``, the release feed's ``size_bytes``, the pack manifest's
``size_bytes`` -- but only ever used it to shape a progress bar. Nothing
stopped the loop itself if the server kept sending past it, so a server that
lied about size (or a MITM without TLS validation, though this is an offline
app) could exhaust memory/disk before the digest check ever got a look at the
bytes. Every case below serves far more than was declared and checks that the
loop aborts partway through instead of writing it all to disk.

Nothing here touches a socket: ``download.open_url`` is replaced throughout,
the same seam ``tests/pipelines/test_update_worker.py`` and ``test_fetch.py``
already use.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from realmspinner.pipelines import download


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _serve(monkeypatch, routes: dict[str, bytes]):
    def fake(url, *, timeout=None):
        return _Response(routes[url])

    monkeypatch.setattr(download, "open_url", fake)


# --- fetch_worker -------------------------------------------------------------


def _import_fetch_worker():
    """``fetch_worker`` flips ``HF_HUB_OFFLINE`` to "0" in this process on
    import -- correct for the child it is written to be, wrong for every test
    that runs after this one in the same pytest process. Same guard
    ``test_fetch.py``'s ``_import_fetch_worker`` uses."""
    import os

    before = os.environ.get("HF_HUB_OFFLINE")
    from realmspinner.pipelines import fetch_worker

    if before is None:
        os.environ.pop("HF_HUB_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = before
    return fetch_worker


def test_fetch_worker_aborts_a_url_asset_that_outgrows_its_declared_size(
    tmp_path, monkeypatch
):
    worker = _import_fetch_worker()
    # size_gib=0.000001 GiB declares ~1 KB; the margin is generous (3x), so
    # serve a payload an order of magnitude past even that before it can pass
    # as "the margin was just tight".
    payload = b"x" * (2 * 1024 * 1024)
    spec = {
        "repo_id": "",
        "dest": str(tmp_path / "engine"),
        "url": "https://example.invalid/trellis-cuda-windows-x64.zip",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "filename": "trellis-cuda-windows-x64.zip",
        "extract": ".",
        "size_gib": 0.000001,
        "retries": 1,
    }
    _serve(monkeypatch, {spec["url"]: payload})

    with pytest.raises(ValueError, match="past what this fetch declared"):
        worker.fetch_one(spec)

    # No half-filled staging tree left for a retry to trip over.
    assert not list(tmp_path.glob("*.fetch.part"))


# --- update_worker -------------------------------------------------------------


def test_update_worker_aborts_an_installer_that_outgrows_its_declared_size(
    tmp_path, monkeypatch
):
    from realmspinner.pipelines import update_worker

    url = "https://example.invalid/RealmspinnerSetup-v0.0.37.exe"
    declared = 4096
    payload = b"MZ" + b"x" * (declared * 10)
    _serve(monkeypatch, {url: payload})
    spec = {
        "mode": "download",
        "installer_url": url,
        "installer_name": "RealmspinnerSetup-v0.0.37.exe",
        "size_bytes": declared,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "dest_dir": str(tmp_path / "updates"),
    }

    with pytest.raises(ValueError, match="past what the release declared"):
        update_worker.fetch(spec)

    # Nothing survives under either the final or the staging name.
    assert list((tmp_path / "updates").iterdir()) == []


# --- pack_worker ---------------------------------------------------------------


def test_pack_worker_aborts_a_wheel_that_outgrows_its_declared_size(tmp_path, monkeypatch):
    from realmspinner.pipelines import pack_worker

    url = "https://example.invalid/thing-1.0-py3-none-any.whl"
    declared = 4096
    payload = b"a wheel-shaped pile of bytes" * 1000  # far past declared * margin
    _serve(monkeypatch, {url: payload})
    wheel = {
        "filename": "thing-1.0-py3-none-any.whl",
        "url": url,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": declared,
    }
    spec = {"pack_dir": str(tmp_path), "wheels": [wheel], "probe": []}

    with pytest.raises(ValueError, match="past what the manifest declared"):
        pack_worker.collect(spec)

    assert list(Path(tmp_path).glob("*.part")) == []
