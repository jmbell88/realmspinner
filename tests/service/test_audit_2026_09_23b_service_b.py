"""Regression tests for the 2026-09-23 (second run) audit's findings
service-05 and service-06.

Fixture shapes for the rename test follow ``tests/test_rename.py``'s own
``_Cfg``/``_populate``/``_migrating`` (duplicated locally rather than
imported -- ``tests/`` is not a package, and neither ``tests/`` nor
``tests/service/`` carries an ``__init__.py``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from realmspinner import models, rename

# --- service-06: rename._ROOT_VARS omitted the export dir and trellis exe ---


class _Cfg(SimpleNamespace):
    home: Path


def _populate(legacy: Path) -> None:
    """The minimum ``rename.run`` needs to consider the move safe otherwise."""
    (legacy / "assets").mkdir(parents=True)
    (legacy / "models" / "familiar").mkdir(parents=True)
    (legacy / "assets" / "jobs.sqlite").write_bytes(b"not really a database")


@pytest.fixture
def _migrating(monkeypatch, tmp_path):
    monkeypatch.delenv("REALMSPINNER_NO_MIGRATE", raising=False)
    monkeypatch.delenv("REALMSPINNER_MIGRATE_KEEP", raising=False)
    monkeypatch.delenv("REALMSPINNER_HOME", raising=False)
    for var in rename._ROOT_VARS:
        monkeypatch.delenv(var, raising=False)
    legacy = tmp_path / ".warlock"
    dest = tmp_path / ".realmspinner"
    _populate(legacy)
    monkeypatch.setattr(rename, "legacy_home", lambda: legacy)
    return SimpleNamespace(legacy=legacy, dest=dest, config=_Cfg(home=dest))


@pytest.mark.parametrize("var", ["REALMSPINNER_EXPORT_DIR", "REALMSPINNER_TRELLIS_EXE"])
def test_rename_leaves_the_legacy_home_alone_when_export_dir_or_trellis_exe_points_inside_it(
    _migrating, monkeypatch, var
):
    """The 2026-09-23 audit, finding service-06: ``_ROOT_VARS`` -- the table
    ``_blocked_by`` walks to decide whether a root the user pointed by hand
    sits inside the legacy home -- carried every path setting except
    ``REALMSPINNER_EXPORT_DIR`` and ``REALMSPINNER_TRELLIS_EXE``. A user who
    pointed either inside ``~/.warlock`` (an export folder kept beside the
    library, or a trellis-server.exe unpacked into it by hand) had it moved
    out from under them the moment this rename ran -- the same silent
    breakage every other entry in the table already exists to prevent (see
    ``test_a_root_pointed_inside_the_legacy_home_stops_the_move`` in
    ``tests/test_rename.py`` for the sibling case this one completes).
    """
    monkeypatch.setenv(var, str(_migrating.legacy / "somewhere"))
    assert rename.run(_migrating.config) is None
    assert (_migrating.legacy / "assets" / "jobs.sqlite").is_file()


# --- service-05: sdxl_cfg's description claimed exclusive ControlNet/CFG ---


def test_no_base_model_description_claims_exclusive_controlnet_or_cfg_support_the_registry_contradicts():  # noqa: E501
    """The 2026-09-23 audit, finding service-05: ``sdxl_cfg``'s description
    said it was "the only family that takes ControlNet, and the only one
    where the negative prompt carries full weight" -- but four other bases
    (``playground``, ``sdxl_cfg_pag``, ``juggernaut``, ``dreamshaper``) are
    also ``controlnet=True`` with the same full-CFG recipe. Swept over every
    row rather than pinned to ``sdxl_cfg`` alone, so a future entry that
    reintroduces the same claim fails this test too.
    """
    controlnet_bases = [m for m in models.BASE_MODELS.values() if m.controlnet]
    assert len(controlnet_bases) > 1, "fixture assumption: more than one base takes ControlNet"
    for base in models.BASE_MODELS.values():
        lowered = base.description.lower()
        assert "the only family" not in lowered, base.key
        claims_exclusive_cfg = "only one" in lowered and (
            "controlnet" in lowered or "cfg" in lowered
        )
        assert not claims_exclusive_cfg, base.key
