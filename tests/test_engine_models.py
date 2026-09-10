from __future__ import annotations

from warlock import fetch, models
from warlock.config import Config
from warlock.service import downloads


def _config(tmp_path):
    return Config(
        home=tmp_path / "home",
        data_dir=tmp_path / "home" / "assets",
        db_path=tmp_path / "home" / "jobs.sqlite",
        t2i_model_root=tmp_path / "image-models",
        trellis_models_dir=tmp_path / "engine-models",
    )


def test_the_engine_kind_leads_with_its_binaries_then_its_weights(tmp_path) -> None:
    """Order is the install order, and the two halves land in different roots.

    The binaries come first because they are the smaller download and the one
    the weights are useless without -- the same packs-before-weights argument
    the mode gate makes, one level down. Before 2026-09-10 there was a single
    engine entry and ``fetch.py`` hardcoded the weights directory for the whole
    kind, which was correct with one entry in it and would have published the
    engine's DLLs on top of the GGUFs the moment there were two.
    """
    config = _config(tmp_path)
    first, second = fetch.entries()[0], fetch.entries()[1]
    assert first.row_key == "engine:trellis_runtime"
    assert second.row_key == "engine:trellis_gguf"
    assert fetch.destination(config, first, first.fetch[0]) == config.trellis_runtime_dir
    assert fetch.destination(config, second, second.fetch[0]) == config.trellis_models_dir


def test_engine_presence_requires_the_exact_pipeline(tmp_path) -> None:
    config = _config(tmp_path)
    spec = models.ENGINE_MODELS["trellis_gguf"]
    config.trellis_models_dir.mkdir(parents=True)
    for name in spec.probe[:-1]:
        (config.trellis_models_dir / name).write_bytes(b"weights")
    assert fetch.present(config, "engine", spec) is False
    (config.trellis_models_dir / spec.probe[-1]).write_bytes(b"weights")
    assert fetch.present(config, "engine", spec) is True


def test_engine_uninstall_stages_on_the_engine_volume(svc, monkeypatch) -> None:
    """WARLOCK_TRELLIS_MODELS may point at a drive unlike image models."""
    spec = models.ENGINE_MODELS["trellis_gguf"]
    svc.config.trellis_models_dir.mkdir(parents=True, exist_ok=True)
    for name in spec.probe:
        (svc.config.trellis_models_dir / name).write_bytes(b"weights")

    real_rename = downloads.os.rename

    def same_parent(src, dst):
        assert src.parent == dst.parent
        return real_rename(src, dst)

    monkeypatch.setattr(downloads.os, "rename", same_parent)
    result = downloads.uninstall(svc, ["engine:trellis_gguf"])

    assert result["removed"] == [str(svc.config.trellis_models_dir)]
    assert not svc.config.trellis_models_dir.exists()


def test_engine_download_command_uses_literal_powershell_quoting(tmp_path) -> None:
    config = _config(tmp_path / "a $literal directory")
    text = fetch.download_text(config, "engine", models.ENGINE_MODELS["trellis_gguf"])
    assert fetch.quote_for_shell(config.trellis_models_dir) in text
    assert f'"{config.trellis_models_dir}"' not in text


def _runtime_config(tmp_path, *, exe):
    """A config whose engine locations are all inside ``tmp_path``.

    Both have to be pinned by hand. ``Config(home=...)`` does **not** move
    ``trellis_runtime_dir``: every root resolves ``_home()`` independently so
    that ``WARLOCK_HOME`` moves all of them at once while a per-root variable
    still wins, which means a ``home=`` keyword moves none of them. Without
    this, these tests read the *developer's* ``vendor/trellis/`` through the
    resolver's third fallback and pass or fail depending on whose machine they
    are on.
    """
    return Config(
        home=tmp_path / "home",
        data_dir=tmp_path / "home" / "assets",
        db_path=tmp_path / "home" / "jobs.sqlite",
        t2i_model_root=tmp_path / "image-models",
        trellis_models_dir=tmp_path / "engine-models",
        trellis_runtime_dir=tmp_path / "engine" / "trellis",
        trellis_server_exe=exe,
    )


def test_an_engine_the_resolver_finds_reads_as_present_wherever_it_is(tmp_path) -> None:
    """Doctor and the Models pane must not disagree about the same engine.

    The engine has three possible homes -- ``WARLOCK_TRELLIS_EXE``, the
    downloaded runtime directory, and the checkout's ``vendor/trellis/`` -- and
    a download can only ever go to the second. A presence probe that asked only
    about the second called the engine missing on a machine that was running
    it: on a source checkout, doctor's ``trellis-server.exe`` row said OK (it
    probes ``resolve_trellis_exe``) while this row said "not downloaded", so
    the pane offered 0.7 GB of something already present and
    ``modes.NEEDS_ROWS`` would have greyed Create on an empty library over a
    working engine. Two surfaces reading the same question must not disagree.

    Exercised through the override, which is one of the three homes and the
    only one a test can put somewhere hermetic; ``tests/test_home_migration.py``
    owns the claim about the order they are tried in.
    """
    spec = models.ENGINE_MODELS["trellis_runtime"]
    elsewhere = tmp_path / "somewhere-else" / "trellis"
    config = _runtime_config(tmp_path, exe=elsewhere / "trellis-server.exe")
    assert fetch.present(config, "engine", spec) is False

    elsewhere.mkdir(parents=True)
    for name in spec.probe:
        (elsewhere / name).write_bytes(b"engine")
    assert fetch.present(config, "engine", spec) is True
    assert config.resolve_trellis_exe().is_file()

    # And a download would still go to the runtime directory, never on top of
    # the copy that was found.
    entry = fetch.find("engine:trellis_runtime")
    assert fetch.destination(config, entry, entry.fetch[0]) == config.trellis_runtime_dir


def test_a_partial_engine_is_absent_the_way_every_other_row_is(tmp_path) -> None:
    """Eight of nine files is "not downloaded", not "downloaded and broken"."""
    spec = models.ENGINE_MODELS["trellis_runtime"]
    config = _runtime_config(
        tmp_path, exe=tmp_path / "engine" / "trellis" / "trellis-server.exe"
    )
    config.trellis_runtime_dir.mkdir(parents=True)
    for name in spec.probe[:-1]:
        (config.trellis_runtime_dir / name).write_bytes(b"engine")
    assert fetch.present(config, "engine", spec) is False
    (config.trellis_runtime_dir / spec.probe[-1]).write_bytes(b"engine")
    assert fetch.present(config, "engine", spec) is True


def test_a_vendored_engine_offers_no_delete_it_cannot_perform(tmp_path) -> None:
    """Present-but-not-downloaded must not offer to free 0.7 GB it will not free.

    The engine's binaries are the only row that can read present from a place
    Warlock may not delete from: presence resolves through
    ``resolve_trellis_exe`` (on a source checkout, ``vendor/trellis/``) while
    the claim is the download location, because that is the only place a Remove
    may act. Before this, a checkout offered Delete on its vendored engine,
    staged nothing, and reported 0.7 GB freed -- a false success, which is
    worse than a refusal.

    Asserted on ``downloads.rows``, not ``fetch.removal_plan``: the plan is
    pure by contract and answers about the registry rather than the disk, so
    the row is the layer that owns this question and already asks the disk for
    ``present``.
    """
    spec = models.ENGINE_MODELS["trellis_runtime"]
    elsewhere = tmp_path / "vendor" / "trellis"
    elsewhere.mkdir(parents=True)
    for name in spec.probe:
        (elsewhere / name).write_bytes(b"engine")
    config = _runtime_config(tmp_path, exe=elsewhere / "trellis-server.exe")

    row = _engine_row(config)
    assert row["present"] is True
    assert row["removable"] is False
    assert row["freed_gib"] == 0.0

    # And once it really is downloaded, Remove works on the copy Warlock owns.
    config.trellis_runtime_dir.mkdir(parents=True)
    for name in spec.probe:
        (config.trellis_runtime_dir / name).write_bytes(b"engine")
    row = _engine_row(config)
    assert row["removable"] is True
    assert row["freed_gib"] > 0.0


def _engine_row(config):
    class _Svc:
        def __init__(self, cfg):
            self.config = cfg
            self.vram_plan = None

    rows = downloads.rows(_Svc(config))
    return next(r for r in rows if r["row_key"] == "engine:trellis_runtime")
