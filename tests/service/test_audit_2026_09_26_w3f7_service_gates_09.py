"""service-gates-09 (2026-09-26 audit): ``selected.json`` written in place.

``_record_selected`` used to write with ``Path.write_text`` directly onto the
served ``selected.json`` -- and ``write_text`` truncates before it writes a
byte, so a crash or a full disk mid-write did not just lose *this* update, it
destroyed every pack this file was the only record of, with
``selected_packs``'s own ``except (OSError, ValueError)`` reading the
resulting empty/torn file back as "nothing installed" rather than raising.
Fixed by staging through ``core/safeio/atomic.write_text`` -- the same
stage-then-``os.replace`` this project uses for every other served name.

``selected_packs`` had a second hole even on an intact file: a bare JSON
*list* (valid JSON, so ``json.loads`` raises nothing) reached ``raw.get`` and
crashed with an uncaught ``AttributeError``, since only ``OSError``/
``ValueError`` were caught.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from realmspinner.service import packs as svc_packs


class FakeService:
    """Only what this module touches. ``RealmspinnerService`` needs a database."""

    def __init__(self, home: Path) -> None:
        self.config = type("C", (), {"home": home})()


@pytest.fixture
def svc(tmp_path):
    return FakeService(tmp_path / "home")


def test_a_bare_json_list_reads_as_no_selection_instead_of_crashing(svc):
    path = svc_packs._selection_path(svc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(["rig", "music"]), encoding="utf-8")

    assert svc_packs.selected_packs(svc) == []


def _truncate_then_raise(self: Path, data: str, *args, **kwargs) -> None:
    """Simulates the real hazard ``write_text`` in place has: opening in ``"w"``
    mode truncates immediately, and a crash before the rest of the write
    completes leaves the file empty. Whichever path this lands on -- the real
    destination (unfixed code) or a staging temp sibling (fixed code) -- is
    the one left destroyed, which is exactly what makes this a fair probe of
    which one the write actually touches first.
    """
    self.write_bytes(b"")
    raise OSError("disk full (simulated)")


def test_record_selected_never_truncates_the_real_file_when_the_write_fails(
    svc, monkeypatch
):
    path = svc_packs._selection_path(svc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"packs": ["rig"]}), encoding="utf-8")

    monkeypatch.setattr(Path, "write_text", _truncate_then_raise)

    # Best-effort: a write failure here must not raise out of an otherwise
    # successful install (the function's own contract).
    svc_packs._record_selected(svc, ["music"])

    assert json.loads(path.read_text(encoding="utf-8")) == {"packs": ["rig"]}
