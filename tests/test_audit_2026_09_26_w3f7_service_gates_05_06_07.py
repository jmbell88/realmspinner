"""service-gates-05/06/07 (2026-09-26 audit): three ``config.py`` env parsers
that accepted a value they should have refused.

* **gates-05** -- ``_env_path`` read ``os.environ.get(name, default)`` raw, so
  an explicitly empty override (``REALMSPINNER_DATA_DIR=""``) was not "unset",
  it was a real value, and ``Path("").resolve()`` is the current working
  directory -- whatever it happens to be when the app launches. ``export_dir``
  had the same hole one layer up: its own ``if os.environ.get(...)`` guard let
  a whitespace-only value (truthy as a Python string) through to ``_env_path``,
  turning the opt-in export feature on for a directory nobody named.
* **gates-06** -- ``rank_candidates``/``pose_fit``/``deform_qa`` parsed their
  flag with a private ``.lower() not in ("0", "false", "off", "no")`` instead
  of ``_env_bool``: a typo (``of`` for ``off``) isn't in that list, so it fell
  through to *true* and never reached ``INVALID_ENV`` for Doctor to report.
* **gates-07** -- ``_env_float``/``_env_opt_float`` parsed with bare ``float()``,
  which accepts ``nan``/``inf``/``-inf`` and negative numbers without raising.
  A NaN timeout or budget compares false against everything it is checked
  against, which is not "unlimited" or "unset" -- it is "refuse every job".
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from realmspinner import config as config_mod
from realmspinner.config import Config


@pytest.fixture(autouse=True)
def _clean_notes():
    config_mod.INVALID_ENV.clear()
    yield
    config_mod.INVALID_ENV.clear()


# --- gates-05: empty/whitespace env values must read as unset ---------------


def test_an_empty_data_dir_override_falls_back_instead_of_resolving_to_the_cwd(
    monkeypatch,
):
    baseline = Config().data_dir
    monkeypatch.setenv("REALMSPINNER_DATA_DIR", "")
    built = Config()
    assert built.data_dir == baseline
    assert built.data_dir != Path("").resolve()


def test_a_whitespace_only_export_dir_is_unset_not_a_directory(monkeypatch):
    monkeypatch.setenv("REALMSPINNER_EXPORT_DIR", "   ")
    assert Config().export_dir is None


def test_a_real_export_dir_still_works(tmp_path, monkeypatch):
    monkeypatch.setenv("REALMSPINNER_EXPORT_DIR", str(tmp_path))
    assert Config().export_dir == tmp_path.resolve()


# --- gates-06: the ad-hoc boolean vocabulary must match _env_bool -----------


@pytest.mark.parametrize(
    "env_name,attr",
    [
        ("REALMSPINNER_RANK", "rank_candidates"),
        ("REALMSPINNER_POSE_FIT", "pose_fit"),
        ("REALMSPINNER_DEFORM_QA", "deform_qa"),
    ],
)
def test_a_typo_on_a_default_true_flag_falls_back_and_is_recorded(
    env_name, attr, monkeypatch
):
    baseline = getattr(Config(), attr)
    assert baseline is True
    monkeypatch.setenv(env_name, "of")
    built = Config()
    assert getattr(built, attr) is True
    assert any(entry[0] == env_name for entry in config_mod.INVALID_ENV)


@pytest.mark.parametrize(
    "env_name,attr",
    [
        ("REALMSPINNER_RANK", "rank_candidates"),
        ("REALMSPINNER_POSE_FIT", "pose_fit"),
        ("REALMSPINNER_DEFORM_QA", "deform_qa"),
    ],
)
def test_the_flag_can_still_be_turned_off(env_name, attr, monkeypatch):
    monkeypatch.setenv(env_name, "off")
    assert getattr(Config(), attr) is False
    assert config_mod.INVALID_ENV == []


# --- gates-07: nan/inf/negative floats must not survive parsing -------------


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "-5"])
def test_a_non_finite_or_negative_timeout_falls_back_to_the_default(raw, monkeypatch):
    baseline = Config().rig_timeout
    monkeypatch.setenv("REALMSPINNER_RIG_TIMEOUT", raw)
    built = Config()
    assert built.rig_timeout == baseline
    assert math.isfinite(built.rig_timeout)
    assert any(entry[0] == "REALMSPINNER_RIG_TIMEOUT" for entry in config_mod.INVALID_ENV)


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "-5"])
def test_a_non_finite_or_negative_vram_budget_is_unset_not_nan(raw, monkeypatch):
    monkeypatch.setenv("REALMSPINNER_VRAM_BUDGET", raw)
    built = Config()
    assert built.vram_budget_gib is None
    assert any(
        entry[0] == "REALMSPINNER_VRAM_BUDGET" for entry in config_mod.INVALID_ENV
    )


def test_a_positive_vram_budget_still_parses(monkeypatch):
    monkeypatch.setenv("REALMSPINNER_VRAM_BUDGET", "12.5")
    assert Config().vram_budget_gib == 12.5
    assert config_mod.INVALID_ENV == []
