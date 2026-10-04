"""The 2026-10-04 audit's first-run kernel findings: create-07 (judge), create-08
(the size row), create-09 (sweep's recommendation) and create-10 (birefnet gate).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from realmspinner import guidance, judge, sweep
from realmspinner.studio import readiness


def _corpus(n_per_class: int = 10, dim: int = 6):
    rng = np.random.default_rng(0)
    pos = rng.normal(1.0, 0.3, size=(n_per_class, dim))
    neg = rng.normal(-1.0, 0.3, size=(n_per_class, dim))
    x = np.vstack([pos, neg])
    y = np.array([1] * n_per_class + [0] * n_per_class)
    return x, y


# --- create-07 ---------------------------------------------------------------


def test_fit_refuses_a_non_finite_embedding_and_score_never_returns_nan(tmp_path):
    """One NaN row used to fit a NaN probe that ``save``/``load`` accepted, and
    ``score`` then returned ``nan`` rather than ``None``."""
    x, y = _corpus()
    assert judge.fit(x, y, stage="reference") is not None  # the control

    bad = x.copy()
    bad[3, 2] = np.nan
    assert judge.fit(bad, y, stage="reference") is None
    inf = x.copy()
    inf[0, 0] = np.inf
    assert judge.fit(inf, y, stage="reference") is None

    # A probe that is already non-finite (an older file, a hand-built one) must
    # not be loadable, and must not score.
    poisoned = judge.Probe(
        weights=np.full(6, np.nan), bias=0.0, stage="reference", labels=20, positives=10
    )
    path = judge.save(poisoned, tmp_path / "probe-reference.npz")
    assert judge.load(path, stage="reference") is None
    assert judge.score(poisoned, np.ones(6)) is None

    good = judge.fit(x, y, stage="reference")
    assert judge.score(good, np.full(6, np.nan)) is None
    result = judge.score(good, np.ones(6))
    assert result is not None and math.isfinite(result)


# --- create-08 ---------------------------------------------------------------


@pytest.mark.parametrize("bad", ["0", "-1", -1.0, float("nan"), float("inf"), "inf"])
def test_size_row_survives_a_zero_or_negative_size_target(bad):
    """``float(size_m)`` was divided by unguarded: ``"0"`` raised
    ``ZeroDivisionError`` on the frame thread and a negative target reported
    "matching the -1.000 m requested"."""
    report = {"achieved_size_m": 1.0}
    row = readiness._size_row(report, {"size_m": bad}, None)
    assert row.state == "attention"
    assert "could not be read" in row.detail
    assert "matching" not in row.detail


# --- create-09 ---------------------------------------------------------------


def test_print_table_never_recommends_a_band_whose_face_count_collapsed(capsys):
    """A band whose mesh collapsed to a slab measures worst 0.0 with a dozen
    faces, and ranking on ``worst`` alone announced it "clearly better than
    'auto'" with "Set config.DEFAULT_TRELLIS_BAND = 16"."""
    rows = [
        {"band": "auto", "worst": 0.30, "mean": 0.10, "faces": 267_000, "seconds": 100.0},
        {"band": "16", "worst": 0.0, "mean": 0.0, "faces": 12, "seconds": 90.0},
    ]
    sweep.print_table(rows, 1024)
    out = capsys.readouterr().out
    assert "DEFAULT_TRELLIS_BAND = 16" not in out
    assert "clearly better" not in out
    assert "degenerate" in out


def test_a_collapsed_band_does_not_hide_a_real_winner(capsys):
    rows = [
        {"band": "auto", "worst": 0.30, "mean": 0.10, "faces": 267_000, "seconds": 100.0},
        {"band": "16", "worst": 0.0, "mean": 0.0, "faces": 12, "seconds": 90.0},
        {"band": "4", "worst": 0.02, "mean": 0.01, "faces": 290_000, "seconds": 100.0},
    ]
    sweep.print_table(rows, 1024)
    out = capsys.readouterr().out
    assert "DEFAULT_TRELLIS_BAND = 4" in out
    assert "DEFAULT_TRELLIS_BAND = 16" not in out


# --- create-10 ---------------------------------------------------------------


def test_default_bg_removal_falls_back_when_birefnet_weights_is_a_directory(
    tmp_path, monkeypatch
):
    """``exists() and st_size > 0`` passed a directory at the weights' name
    wherever a directory reports a non-zero size (ext4, APFS, and a ReFS
    volume); the size is pinned here so the proof does not depend on the
    filesystem the suite runs on."""
    (tmp_path / guidance.BIREFNET_WEIGHTS).mkdir()
    real_stat = type(tmp_path).stat

    def stat(self, *a, **k):
        result = real_stat(self, *a, **k)
        if self.name == guidance.BIREFNET_WEIGHTS:
            return type(
                "S", (), {"st_size": 4096, "st_mode": result.st_mode}
            )()
        return result

    monkeypatch.setattr(type(tmp_path), "stat", stat)
    assert guidance.default_bg_removal(tmp_path) == "auto"
