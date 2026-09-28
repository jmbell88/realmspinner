"""Regression test for the 2026-09-26 audit, finding pipelines-install-03.

``fetch.removal_plan`` already refused to delete a base model's directory when
it was the ``REALMSPINNER_T2I_DIR`` override -- a directory the user pointed
Realmspinner at, not one it fetched. The two engine directories
(``REALMSPINNER_TRELLIS_MODELS``, ``REALMSPINNER_TRELLIS_RUNTIME``) had no
equivalent guard, and the general "outside the model root" check cannot supply
one: it treats ``trellis_models_dir``/``trellis_runtime_dir`` themselves as
allowed roots, so anything claimed under an overridden one is definitionally
"inside" it and the refusal never fires.
"""

from __future__ import annotations

from realmspinner import fetch
from realmspinner.config import Config


def _entry(row_key: str) -> fetch.Entry:
    entry = fetch.find(row_key)
    assert entry is not None, row_key
    return entry


def test_a_trellis_models_env_override_directory_is_never_deleted(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere-gguf"
    monkeypatch.setenv("REALMSPINNER_TRELLIS_MODELS", str(elsewhere))
    cfg = Config()
    assert cfg.trellis_models_dir == elsewhere

    removal = fetch.removal_plan(cfg, [_entry("engine:trellis_gguf")])

    assert removal.paths == ()
    assert removal.freed_gib == 0.0
    assert any("REALMSPINNER_TRELLIS_MODELS" in reason for reason in removal.blocked)


def test_a_trellis_runtime_env_override_directory_is_never_deleted(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere-runtime"
    monkeypatch.setenv("REALMSPINNER_TRELLIS_RUNTIME", str(elsewhere))
    cfg = Config()
    assert cfg.trellis_runtime_dir == elsewhere

    removal = fetch.removal_plan(cfg, [_entry("engine:trellis_runtime")])

    assert removal.paths == ()
    assert removal.freed_gib == 0.0
    assert any("REALMSPINNER_TRELLIS_RUNTIME" in reason for reason in removal.blocked)


def test_trellis_dirs_are_still_removable_when_not_env_overridden(tmp_path, monkeypatch):
    """The guard must not become a blanket refusal: an un-overridden engine
    directory under the real model root is exactly what Remove is for."""
    monkeypatch.delenv("REALMSPINNER_TRELLIS_MODELS", raising=False)
    monkeypatch.delenv("REALMSPINNER_TRELLIS_RUNTIME", raising=False)
    cfg = Config()
    cfg.trellis_models_dir = tmp_path / "models" / "trellis2-gguf"
    cfg.trellis_runtime_dir = tmp_path / "engine" / "trellis"

    removal = fetch.removal_plan(cfg, [_entry("engine:trellis_gguf")])

    assert removal.blocked == ()
    assert removal.paths == (cfg.trellis_models_dir,)
