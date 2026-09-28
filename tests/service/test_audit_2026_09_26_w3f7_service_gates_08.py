"""service-gates-08 (2026-09-26 audit): every ``_run_worker`` failure raised
"Could not check for updates", including a failed installer *download*.

The timeout branch already varies its wording by ``op``
(``test_a_stalled_download_says_download_not_check``, service-05 from the
2026-09-07 audit) but the ``ok: False`` branch a few lines below it did not --
a stub child that writes ``{"ok": false, "error": ...}`` during
``download()`` was still reported to the user as a failed *check*, naming the
wrong worry: the release feed answered fine, the ~100 MB installer transfer
is what failed.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from realmspinner.service import updates as svc_updates
from realmspinner.service.errors import Invalid


class FakeService:
    """Only what this module touches. ``RealmspinnerService`` needs a database."""

    def __init__(self, home: Path) -> None:
        self.config = type("C", (), {"home": home})()


@pytest.fixture
def svc(tmp_path):
    return FakeService(tmp_path / "home")


def _stub(monkeypatch, body: str) -> None:
    monkeypatch.setattr(
        svc_updates, "worker_argv", lambda: [sys.executable, "-c", textwrap.dedent(body)]
    )


def test_a_failed_installer_download_says_download_not_check(svc, monkeypatch):
    _stub(
        monkeypatch,
        """
        import json, sys
        spec = json.loads(sys.stdin.read())
        open(spec["result_path"], "w").write(
            json.dumps({"ok": False, "error": "the transfer reset partway through"})
        )
        """,
    )
    info = {
        "installer_url": "https://example.invalid/x.exe",
        "installer_name": "x.exe",
        "size_bytes": 100,
        "sha256": "a" * 64,
    }
    with pytest.raises(Invalid) as caught:
        svc_updates.download(svc, info)
    message = str(caught.value)
    assert "download" in message
    assert "check" not in message
    assert "the transfer reset partway through" in message
