"""Regression tests for the 2026-09-15 audit's pipelines findings.

Four independent findings, four independent tests, grouped here because they
share no code -- the installer, the release gate, the image pipeline's LoRA
loading and the update checker.
"""

from __future__ import annotations

import importlib.util
import io
import re
from pathlib import Path
from typing import Any

import pytest

from realmspinner import models
from realmspinner.pipelines import download, update_worker
from realmspinner.pipelines.text2image import Text2Image

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# pipelines-01 -- installer upgrade leaves the old trellis engine behind
# ---------------------------------------------------------------------------


def test_upgrade_install_delete_removes_the_previous_versions_staged_trellis_engine() -> None:
    """``build.ps1`` stopped staging ``vendor\\trellis`` on 2026-09-10 (the
    engine is a Settings -> Models download now), but ``[InstallDelete]`` was
    never told -- so an in-place upgrade from <=0.0.41 left the old 838 MB
    engine sitting in ``{app}\\vendor\\trellis`` forever. Worse than dead
    weight: ``Config.resolve_trellis_exe`` falls back to exactly that path
    when no downloaded engine is present, so the stale, unpinned engine from
    the previous release kept being *used*, not just wasting disk."""
    iss = (ROOT / "installer" / "realmspinner.iss").read_text(encoding="utf-8")
    # Section-header-aware, not a plain str.split on "[Files]" -- a comment a
    # few lines into [InstallDelete] itself reads "...replaced in place by
    # [Files]." and a naive split truncates there, well before the real
    # [Files] section header (and before the line under test).
    sections = re.split(r"(?m)^\[(\w+)\]\s*$", iss)
    install_delete = sections[sections.index("InstallDelete") + 1]
    assert r'Type: filesandordirs; Name: "{app}\vendor\trellis"' in install_delete


# ---------------------------------------------------------------------------
# pipelines-02 -- release gate does not check uv.lock's own realmspinner version
# ---------------------------------------------------------------------------


def _load_preflight():
    spec = importlib.util.spec_from_file_location(
        "realmspinner_preflight_audit_2026_09_15", ROOT / "scripts" / "preflight.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_release_tree(tmp_path: Path, *, version: str, uv_lock_version: str) -> None:
    (tmp_path / "src" / "realmspinner").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(f'version = "{version}"\n', encoding="utf-8")
    (tmp_path / "src" / "realmspinner" / "__init__.py").write_text(
        f'__version__ = "{version}"\n', encoding="utf-8"
    )
    (tmp_path / "CHANGELOG.md").write_text(f"## {version}\n\n- notes\n", encoding="utf-8")
    (tmp_path / "INSTALL.md").write_text(
        f"Download `RealmspinnerSetup-v{version}.exe` from Releases.\n", encoding="utf-8"
    )
    (tmp_path / "uv.lock").write_text(
        "[[package]]\n"
        'name = "realmspinner"\n'
        f'version = "{uv_lock_version}"\n'
        'source = { editable = "." }\n',
        encoding="utf-8",
    )


def test_check_versions_catches_a_stale_uv_lock_version(tmp_path: Path, capsys) -> None:
    """CLAUDE.md defines a release as five files in lockstep, but
    ``check_versions`` only ever compared four -- a release commit that bumped
    ``pyproject.toml``/``__init__.py``/``CHANGELOG.md``/``INSTALL.md`` and
    forgot to run ``uv sync`` afterwards would pass this gate with a stale
    lockfile."""
    module = _load_preflight()
    _write_release_tree(tmp_path, version="0.0.50", uv_lock_version="0.0.49")
    module.ROOT = tmp_path

    assert module.check_versions() is False
    out = capsys.readouterr().out
    assert "uv.lock" in out


def test_check_versions_passes_when_uv_lock_agrees(tmp_path: Path) -> None:
    module = _load_preflight()
    _write_release_tree(tmp_path, version="0.0.50", uv_lock_version="0.0.50")
    module.ROOT = tmp_path

    assert module.check_versions() is True


# ---------------------------------------------------------------------------
# pipelines-04 -- a directory where a LoRA file belongs read as "present"
# ---------------------------------------------------------------------------


class _FakePipe:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    def load_lora_weights(self, _dir: Any, weight_name: str, adapter_name: str, **_kw: Any) -> None:
        self.loaded.append(adapter_name)


def test_a_directory_where_the_base_lora_file_belongs_is_reported_missing(tmp_path: Path) -> None:
    """``_load_loras`` and ``_ensure_adapter`` used ``path.exists()``, which is
    True for a directory -- a partial/broken download that left a directory
    where the LoRA file belongs read as "present" and would fall through into
    ``pipe.load_lora_weights`` (and, with a real diffusers pipe, its own
    low-level traceback) instead of this module's clear remedy message. The
    three sibling checks fixed by pipelines-06 (2026-09-11 audit) were already
    ``is_file()``; these two were the ones it missed."""
    base_spec = models.BASE_MODELS["sdxl"]
    assert base_spec.base_lora is not None
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()
    (loras_dir / base_spec.base_lora).mkdir()  # a directory, not the weights file

    t2i = Text2Image(base_spec, tmp_path)
    pipe = _FakePipe()

    with pytest.raises(RuntimeError, match="requires"):
        t2i._load_loras(pipe)

    style_key = "render3d"
    style_spec = models.STYLE_LORAS[style_key]
    (loras_dir / style_spec.filename).mkdir()

    with pytest.raises(RuntimeError, match="not downloaded"):
        t2i._ensure_adapter(pipe, style_key)


# ---------------------------------------------------------------------------
# pipelines-05 -- the update checker read a JSON response with no size ceiling
# ---------------------------------------------------------------------------


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False


def test_check_refuses_a_release_feed_response_over_the_manifest_size_ceiling(
    monkeypatch,
) -> None:
    """``_get_json`` used ``response.read()`` with no limit -- a compromised
    or misconfigured feed host answering ``/releases/latest`` (or the
    ``update-manifest.json`` asset it points at) with an unbounded body would
    have this process buffer all of it before ``json.loads`` ever got a
    chance to reject it. Same host-exhaustion shape ``trellis.py``'s
    ``MAX_GLB_BYTES`` already guards against."""
    oversized = b"[" + b"1," * update_worker.MAX_MANIFEST_BYTES + b"1]"

    def fake_open_url(url: str, *, timeout: float | None = None) -> _Response:
        return _Response(oversized)

    monkeypatch.setattr(download, "open_url", fake_open_url)

    with pytest.raises(ValueError, match="more than"):
        update_worker.check({})
