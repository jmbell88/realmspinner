"""F21: an installed engine that is not the pinned build says so.

The 2026-10-11 re-pin of ``trellis_runtime`` (v0.6.0 -> v0.8.1) reached only a
*new* download, because ``fetch.present`` asks whether files exist and
``verify_manifest`` compares against the digests the download itself recorded.
These pin the question neither asks: does the last completed download match
what the registry pins today.
"""

from __future__ import annotations

import json

from realmspinner import fetch, models
from realmspinner.config import Config
from realmspinner.service import downloads

OLD = "0" * 64


def _config(tmp_path, *, exe=None):
    runtime = tmp_path / "engine" / "trellis"
    return Config(
        home=tmp_path / "home",
        data_dir=tmp_path / "home" / "assets",
        db_path=tmp_path / "home" / "jobs.sqlite",
        t2i_model_root=tmp_path / "image-models",
        trellis_models_dir=tmp_path / "engine-models",
        trellis_runtime_dir=runtime,
        trellis_server_exe=exe if exe is not None else runtime / "trellis-server.exe",
    )


def _install_runtime(root, digests):
    spec = models.ENGINE_MODELS["trellis_runtime"]
    root.mkdir(parents=True, exist_ok=True)
    for name in spec.probe:
        (root / name).write_bytes(b"engine")
    if digests is not None:
        record = {
            "repo_id": "",
            "revision": "",
            "files": sorted(digests),
            "digests": {n: {"size": 6, "sha256": d} for n, d in digests.items()},
        }
        (root / fetch.MANIFEST_NAME).write_text(
            json.dumps({"repos": {"": record}}), encoding="utf-8"
        )
    return spec


def _row(config, key="engine:trellis_runtime"):
    class _Svc:
        def __init__(self, cfg):
            self.config = cfg
            self.vram_plan = None

    return next(r for r in downloads.rows(_Svc(config)) if r["row_key"] == key)


def _pinned():
    return dict(models.TRELLIS_RUNTIME_DIGESTS)


def test_an_engine_downloaded_before_the_repin_reads_as_installed_but_outdated(tmp_path) -> None:
    config = _config(tmp_path)
    old = {name: OLD for name in _pinned()}
    spec = _install_runtime(config.trellis_runtime_dir, old)

    assert fetch.present(config, "engine", spec) is True
    assert set(fetch.outdated(config, "engine", spec)) == set(_pinned())
    row = _row(config)
    assert row["present"] is True
    assert row["outdated"] is True
    assert row["downloadable"] is True


def test_one_changed_library_is_enough_and_is_named(tmp_path) -> None:
    config = _config(tmp_path)
    digests = _pinned()
    digests["ggml-cuda.dll"] = OLD
    spec = _install_runtime(config.trellis_runtime_dir, digests)
    assert fetch.outdated(config, "engine", spec) == ("ggml-cuda.dll",)


def test_a_library_the_pin_added_since_the_download_counts_as_outdated(tmp_path) -> None:
    config = _config(tmp_path)
    digests = _pinned()
    del digests["cudart64_13.dll"]
    spec = _install_runtime(config.trellis_runtime_dir, digests)
    assert fetch.outdated(config, "engine", spec) == ("cudart64_13.dll",)


def test_a_current_engine_has_no_outdated_key(tmp_path) -> None:
    config = _config(tmp_path)
    spec = _install_runtime(config.trellis_runtime_dir, _pinned())
    assert fetch.outdated(config, "engine", spec) == ()
    assert "outdated" not in _row(config)


def test_an_engine_with_no_manifest_is_unknown_not_outdated(tmp_path) -> None:
    """A hand copy or a pre-manifest download is never red: there is nothing to compare."""
    config = _config(tmp_path)
    spec = _install_runtime(config.trellis_runtime_dir, None)
    assert fetch.outdated(config, "engine", spec) == ()
    assert "outdated" not in _row(config)


def test_a_vendored_engine_is_not_offered_a_download_that_would_not_replace_it(tmp_path) -> None:
    """The resolver found the exe elsewhere: Download again would change nothing it runs."""
    elsewhere = tmp_path / "vendor" / "trellis"
    config = _config(tmp_path, exe=elsewhere / "trellis-server.exe")
    _install_runtime(elsewhere, None)
    # A stale manifest sits in the runtime directory the engine is not run from.
    spec = _install_runtime(config.trellis_runtime_dir, {n: OLD for n in _pinned()})
    assert fetch.outdated(config, "engine", spec) == ()
    assert "outdated" not in _row(config)


def _gguf_manifest(config, revision):
    spec = models.ENGINE_MODELS["trellis_gguf"]
    root = config.trellis_models_dir
    root.mkdir(parents=True, exist_ok=True)
    for name in spec.probe:
        (root / name).write_bytes(b"weights")
    repo = spec.fetch[0].repo_id
    (root / fetch.MANIFEST_NAME).write_text(
        json.dumps({"repos": {repo: {"repo_id": repo, "revision": revision, "digests": {}}}}),
        encoding="utf-8",
    )
    return spec


def test_weights_downloaded_at_another_commit_are_outdated(tmp_path) -> None:
    config = _config(tmp_path)
    spec = _gguf_manifest(config, "f" * 40)
    pin = spec.fetch[0]
    assert fetch.outdated(config, "engine", spec) == (f"{pin.repo_id}@{pin.revision[:8]}",)
    assert _row(config, "engine:trellis_gguf")["outdated"] is True


def test_weights_at_the_pinned_commit_or_with_no_recorded_commit_are_current(tmp_path) -> None:
    config = _config(tmp_path)
    spec = _gguf_manifest(config, models.ENGINE_MODELS["trellis_gguf"].fetch[0].revision)
    assert fetch.outdated(config, "engine", spec) == ()
    # A download that predates pinning recorded "": unknown, never red.
    _gguf_manifest(config, "")
    assert fetch.outdated(config, "engine", spec) == ()


def test_only_engine_rows_are_ever_outdated(tmp_path) -> None:
    config = _config(tmp_path)
    spec = models.FAMILIAR_MODELS[next(iter(models.FAMILIAR_MODELS))]
    assert fetch.outdated(config, "familiar", spec) == ()
