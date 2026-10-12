"""Fix-pass-7: ``realmspinner sweep`` had no ``--matte``, so a sweep could only ever
run under whatever ``guidance.default_bg_removal`` resolved on the host; create-13
fixed the default, this lets a human pin the matte a measurement is taken under."""

from __future__ import annotations

import asyncio
import sys

import pytest

from realmspinner import cli, guidance, sweep
from realmspinner.config import Config


class _FakeServer:
    calls: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def generate(self, image, out, **kwargs):
        type(self).calls.append(kwargs)
        out.write_bytes(b"glb")

    def stop(self) -> None:
        pass


@pytest.fixture
def fake_sweep(monkeypatch, tmp_path):
    from realmspinner import meshaudit

    _FakeServer.calls = []
    monkeypatch.setattr(sweep, "TrellisServer", _FakeServer)
    monkeypatch.setattr(
        meshaudit, "hole_fraction", lambda *a, **k: {"worst": 0.1, "mean": 0.05, "faces": 1000}
    )
    models_dir = tmp_path / "trellis-models"
    models_dir.mkdir()
    (models_dir / "birefnet.gguf").write_bytes(b"weights")
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_models_dir=models_dir,
    )
    return config, tmp_path


def test_run_band_sends_the_matte_it_was_told_to_not_the_hosts_default(fake_sweep):
    config, tmp_path = fake_sweep
    row = asyncio.run(
        sweep.run_band(
            config, 4, tmp_path / "ref.png", tmp_path,
            seed=1, resolution=512, audit_resolution=64, matte="threshold",
        )
    )
    assert _FakeServer.calls[-1]["bg_removal"] == "threshold"
    assert row["matte"] == "threshold"


def test_run_band_without_a_matte_keeps_the_hosts_default(fake_sweep):
    config, tmp_path = fake_sweep
    asyncio.run(
        sweep.run_band(
            config, 4, tmp_path / "ref.png", tmp_path,
            seed=1, resolution=512, audit_resolution=64,
        )
    )
    assert _FakeServer.calls[-1]["bg_removal"] == "birefnet"


def _argv(monkeypatch, *extra: str) -> None:
    monkeypatch.setattr(sys, "argv", ["realmspinner", "sweep", "--image", "ref.png", *extra])


def test_the_sweep_command_passes_matte_through_to_the_sweep(monkeypatch):
    seen: dict = {}

    async def fake_sweep(*args, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(sweep, "sweep", fake_sweep)
    monkeypatch.setattr(sweep, "print_table", lambda *a, **k: None)
    _argv(monkeypatch, "--matte", "auto")
    cli.main()
    assert seen["matte"] == "auto"

    _argv(monkeypatch)
    cli.main()
    assert seen["matte"] is None


def test_the_sweep_command_refuses_a_matte_the_door_does_not_accept(monkeypatch, capsys):
    _argv(monkeypatch, "--matte", "magic")
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert all(choice in err for choice in guidance.BG_REMOVAL)
