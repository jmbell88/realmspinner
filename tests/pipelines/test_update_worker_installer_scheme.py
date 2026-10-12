"""An installer is only ever fetched over https, whatever the feed says.

The 2026-10-10 audit's F18-8. ``check`` already sanitised ``release_url`` (http
or https) and the installer *filename*, but the installer's own
``browser_download_url`` went through ``str()`` and straight into the spec the
download worker hands ``download.open_url`` -- which is ``urllib``, and takes
``file:`` and ``ftp:`` as readily as ``https:``. An installer is an executable
the user is about to double-click, so its transport is the one URL that has to
be https (a plain-http fetch is also readable and rewritable in transit; the
sha256 is the second line of defence, not the first).
"""

from __future__ import annotations

import io
import json

import pytest

from realmspinner.pipelines import download, update_worker

ASSET = update_worker.MANIFEST_ASSET
INSTALLER = "RealmspinnerSetup-v0.0.37.exe"
MANIFEST_URL = "https://example.invalid/m.json"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _serve(monkeypatch, installer_url: str) -> None:
    routes = {
        update_worker.RELEASES_URL: json.dumps(
            {
                "tag_name": "v0.0.37",
                "html_url": "https://example.invalid/releases/v0.0.37",
                "assets": [
                    {"name": ASSET, "browser_download_url": MANIFEST_URL, "size": 0},
                    {"name": INSTALLER, "browser_download_url": installer_url, "size": 10},
                ],
            }
        ).encode(),
        MANIFEST_URL: json.dumps(
            {
                "version": "0.0.37",
                "installer": {"filename": INSTALLER, "size_bytes": 10, "sha256": "a" * 64},
            }
        ).encode(),
    }

    def fake(url, *, timeout=None):
        return _Response(routes[url])

    monkeypatch.setattr(download, "open_url", fake)


@pytest.mark.parametrize(
    "bad",
    [
        "http://example.invalid/x.exe",
        "file:///C:/Windows/System32/calc.exe",
        "ftp://example.invalid/x.exe",
        "javascript:alert(1)",
        "https://",
        "https:///x.exe",
        "//example.invalid/x.exe",
        "",
    ],
)
def test_check_refuses_an_installer_url_that_is_not_an_https_url_with_a_host(monkeypatch, bad):
    _serve(monkeypatch, bad)
    with pytest.raises(ValueError, match="https"):
        update_worker.check({})


def test_check_still_offers_an_https_installer(monkeypatch):
    _serve(monkeypatch, "https://github.com/o/r/releases/download/v1/" + INSTALLER)
    out = update_worker.check({})
    assert out["installer_url"].startswith("https://github.com/")


@pytest.mark.parametrize("bad", ["http://example.invalid/x.exe", "file:///C:/x.exe", ""])
def test_fetch_refuses_a_non_https_installer_url_before_opening_it(monkeypatch, tmp_path, bad):
    """The parent writes this spec from ``check``'s answer, so this is the second
    door on the same field -- a spec that did not come through ``check`` (a stale
    cached answer, a test, a future caller) still cannot reach ``open_url``."""

    def never(url, *, timeout=None):
        raise AssertionError(f"opened {url!r}")

    monkeypatch.setattr(download, "open_url", never)
    with pytest.raises(ValueError, match="https"):
        update_worker.fetch(
            {
                "mode": "download",
                "installer_url": bad,
                "installer_name": INSTALLER,
                "size_bytes": 10,
                "sha256": "a" * 64,
                "dest_dir": str(tmp_path / "updates"),
            }
        )
    assert not (tmp_path / "updates" / INSTALLER).exists()
