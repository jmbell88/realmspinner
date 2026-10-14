"""The 2026-10-14 audit, fixer F07: finding service-assets-05.

``files._staged_write`` and ``derive._staged`` did one bare ``os.replace``, so
a reader holding the served name open for a moment (WinError 5 on Windows)
failed the write where the rig and postprocess doors ride it out.
"""

from __future__ import annotations

import os

import pytest

from realmspinner.core.safeio import atomic
from realmspinner.service import derive, files


def _flaky_replace(monkeypatch, failures: int) -> list[int]:
    real = os.replace
    calls: list[int] = []

    def replace(src, dst):
        calls.append(1)
        if len(calls) <= failures:
            raise PermissionError(13, "Access is denied", str(dst))
        return real(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(atomic, "REPLACE_PAUSE", 0.0, raising=False)
    return calls


def test_staged_write_rides_out_a_reader_holding_the_target_open(
    tmp_path, monkeypatch
) -> None:
    dest = tmp_path / "input.png"
    dest.write_bytes(b"old")
    calls = _flaky_replace(monkeypatch, 2)
    files._staged_write(dest, b"new")
    assert dest.read_bytes() == b"new"
    assert len(calls) == 3
    assert [p.name for p in tmp_path.iterdir()] == ["input.png"]


def test_staged_write_still_raises_and_cleans_up_when_the_reader_never_leaves(
    tmp_path, monkeypatch
) -> None:
    dest = tmp_path / "input.png"
    dest.write_bytes(b"old")
    _flaky_replace(monkeypatch, 10**6)
    with pytest.raises(PermissionError):
        files._staged_write(dest, b"new")
    assert dest.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["input.png"]


def test_derive_staged_rides_out_a_reader_holding_the_artifact_open(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "model.stl").write_bytes(b"old")
    calls = _flaky_replace(monkeypatch, 2)
    derive._staged(tmp_path, "model.stl", lambda tmp: tmp.write_bytes(b"new"))
    assert (tmp_path / "model.stl").read_bytes() == b"new"
    assert len(calls) == 3
    assert [p.name for p in tmp_path.iterdir()] == ["model.stl"]
