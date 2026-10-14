"""The 2026-10-14 audit, fixer F03: the model store, the Familiar rows and a few
small doors around them.

One file at the top of ``tests/`` rather than one per mirror directory:
``tests/pipelines`` and ``tests/service`` carry no ``__init__.py``, so two files
sharing this basename would collide under pytest's default import mode.
"""

from __future__ import annotations

import json
import logging
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from realmspinner import doctor, fetch, generation, hashes, judge, leases, meshreport, models
from realmspinner.config import Config
from realmspinner.service import downloads
from realmspinner.service.errors import Conflict

# --- helpers -------------------------------------------------------------------


def _config(tmp_path: Path) -> Config:
    """Every directory a row can land in, pinned inside ``tmp_path``."""
    return Config(
        home=tmp_path / "home",
        data_dir=tmp_path / "home" / "assets",
        db_path=tmp_path / "home" / "jobs.sqlite",
        t2i_model_root=tmp_path / "models",
        trellis_models_dir=tmp_path / "engine-models",
        trellis_runtime_dir=tmp_path / "engine" / "trellis",
        trellis_server_exe=tmp_path / "engine" / "trellis" / "trellis-server.exe",
        familiar_runtime_dir=tmp_path / "engine" / "llama",
        familiar_models_dir=tmp_path / "models" / "familiar",
    )


def _entries(*row_keys: str) -> list[fetch.Entry]:
    chosen = []
    for row_key in row_keys:
        entry = fetch.find(row_key)
        assert entry is not None, row_key
        chosen.append(entry)
    return chosen


# --- pipelines-25: the two doctor rows that still said ``uv sync`` -------------


def _checkpoint(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(b"weights")


def test_the_matting_and_text_rows_name_settings_packs_not_uv_sync(tmp_path, monkeypatch):
    """The 2026-10-03 audit (pipelines-15) swapped the ``uv sync --extra`` literal
    for ``_pack_hint`` on the CUDA, Blender and Muse rows and missed these two:
    weights present, imports missing -- reachable on a packaged install because
    the weights and the pack arrive in either order -- still told the user to
    run a command there is no venv to run it in."""
    from realmspinner.studio.modes.inker import flourish as inker_flourish

    config = _config(tmp_path)
    config.t2i_model_root.mkdir(parents=True, exist_ok=True)
    for spec in models.MATTING_MODELS.values():
        _checkpoint(config.t2i_model_root / spec.dir_name)
    monkeypatch.setattr(doctor, "_probes", {})
    monkeypatch.setattr(doctor, "_run_load_probe", lambda which, path: (True, "loaded"))
    monkeypatch.setattr(doctor.matting, "last_error", lambda: None)
    monkeypatch.setattr(doctor, "_missing_modules", lambda names: ["einops"])
    monkeypatch.setattr(inker_flourish, "text_model_present", lambda cfg: True)

    rows = doctor._matting_checks(config, probe_slow=True) + doctor._text_checks(config)
    named = [row for row in rows if "cannot import" in row.detail]
    # Both a matting row and the Flourish text row reached the missing-import branch.
    assert len(named) == len(models.MATTING_MODELS) + 1
    for row in named:
        assert "uv sync" not in row.detail, row.name
        assert doctor._pack_hint("text2image") in row.detail, row.name


# --- pipelines-03 / service-gates-01: two URL rows, one destination ------------


def test_planning_both_familiar_runtime_rows_keeps_both_archives(tmp_path):
    """Both runtime rows are URL fetches with ``repo_id`` "" into one directory,
    so ``plan``'s ``(repo_id, destination)`` key folded them into one job and
    ``_merge`` kept only the first URL: the CUDA redistributable zip was never
    downloaded while its size was still added to the total."""
    config = _config(tmp_path)
    jobs = fetch.plan(
        config, _entries("familiar:familiar_runtime", "familiar:familiar_runtime_cudart")
    )
    assert sorted(job.url for job in jobs) == sorted(
        [models.FAMILIAR_RUNTIME_MAIN_URL, models.FAMILIAR_RUNTIME_CUDART_URL]
    )
    assert sorted(job.sha256 for job in jobs) == sorted(
        [models.FAMILIAR_RUNTIME_MAIN_SHA256, models.FAMILIAR_RUNTIME_CUDART_SHA256]
    )
    assert fetch.total_gib(jobs) == pytest.approx(0.24 + 0.37)


def test_planning_both_familiar_runtime_rows_yields_two_jobs_with_both_urls(tmp_path):
    config = _config(tmp_path)
    jobs = fetch.plan(
        config,
        _entries(
            "familiar:familiar_runtime",
            "familiar:familiar_runtime_cudart",
            "familiar:familiar_gguf",
        ),
    )
    by_url = {job.url: job for job in jobs if job.url}
    assert set(by_url) == {models.FAMILIAR_RUNTIME_MAIN_URL, models.FAMILIAR_RUNTIME_CUDART_URL}
    assert {job.filename for job in by_url.values()} == {
        models.FAMILIAR_RUNTIME_MAIN_ASSET,
        models.FAMILIAR_RUNTIME_CUDART_ASSET,
    }
    # The weights are a Hub fetch and stay one job of their own.
    assert len(jobs) == 3


def test_merge_refuses_to_drop_a_second_archive():
    """The collision was silent: ``_merge`` kept ``into.url or one.url``. If the
    dedupe key ever lets two different archives meet again it must say so."""
    main = models.FAMILIAR_MODELS["familiar_runtime"].fetch[0]
    cudart = models.FAMILIAR_MODELS["familiar_runtime_cudart"].fetch[0]
    job = fetch.Job(
        repo_id="", dest=Path("x"), url=main.url, sha256=main.sha256, filename=main.filename
    )
    with pytest.raises(ValueError):
        fetch._merge(job, cudart)


def test_two_url_archives_into_one_directory_keep_two_manifest_records(tmp_path):
    """With two jobs for one directory the completion record is keyed by
    ``repo_id`` and both are "": the second overwrote the first, so the runtime
    directory recorded the digests of one zip and ``verify`` and ``outdated``
    went blind to the other."""
    config = _config(tmp_path)
    dest = config.familiar_runtime_dir
    dest.mkdir(parents=True)
    jobs = fetch.plan(
        config, _entries("familiar:familiar_runtime", "familiar:familiar_runtime_cudart")
    )
    for index, job in enumerate(jobs):
        staging = tmp_path / f"staging-{index}"
        staging.mkdir()
        (staging / f"file{index}.dll").write_bytes(b"x" * (index + 1))
        downloads._stage_manifest_for(job, staging)
        # Publish the manifest the way the move does, so the next job reads it.
        (dest / fetch.MANIFEST_NAME).write_bytes((staging / fetch.MANIFEST_NAME).read_bytes())
    repos = json.loads((dest / fetch.MANIFEST_NAME).read_text())["repos"]
    recorded = {name for record in repos.values() for name in record["digests"]}
    assert recorded == {"file0.dll", "file1.dll"}


# --- pipelines-09 / service-gates-04: Familiar weights rows share a directory --


def test_each_familiar_weights_row_can_be_removed_without_the_others(tmp_path):
    """The three weights rows all claimed ``models/familiar`` whole, so each
    found a sibling claiming it and none was removable; the pane draws Remove
    only for a removable row, so there was no button and no reason."""
    config = _config(tmp_path)
    for key, file, gib in (
        ("familiar_gguf", models.FAMILIAR_GGUF_FILE, 6.50),
        ("familiar_mmproj", models.FAMILIAR_MMPROJ_FILE, 0.16),
        ("familiar_embed", models.FAMILIAR_EMBED_GGUF_FILE, 0.29),
    ):
        removal = fetch.removal_plan(config, _entries(f"familiar:{key}"))
        assert removal.blocked == (), key
        assert removal.paths == (config.familiar_models_dir / file,), key
        assert removal.freed_gib == pytest.approx(gib), key


def test_each_familiar_row_is_removable_on_its_own_and_frees_only_its_files(tmp_path):
    config = _config(tmp_path)
    everything = fetch.removal_plan(
        config,
        _entries("familiar:familiar_gguf", "familiar:familiar_mmproj", "familiar:familiar_embed"),
    )
    assert set(everything.paths) == {
        config.familiar_models_dir / models.FAMILIAR_GGUF_FILE,
        config.familiar_models_dir / models.FAMILIAR_MMPROJ_FILE,
        config.familiar_models_dir / models.FAMILIAR_EMBED_GGUF_FILE,
    }
    # The directory itself is never claimed, so removing one row cannot take the
    # others' files or the manifest beside them.
    assert config.familiar_models_dir not in everything.paths
    # The runtime pair still genuinely shares one directory: either alone leaves
    # the other's claim standing.
    alone = fetch.removal_plan(config, _entries("familiar:familiar_runtime"))
    assert alone.paths == ()


def _install_familiar_weights(config: Config) -> dict[str, Path]:
    dest = config.familiar_models_dir
    dest.mkdir(parents=True, exist_ok=True)
    files = {
        "gguf": dest / models.FAMILIAR_GGUF_FILE,
        "mmproj": dest / models.FAMILIAR_MMPROJ_FILE,
    }
    digests = {}
    for index, path in enumerate(files.values()):
        path.write_bytes(bytes([index + 1]) * 8)
        digests[path.name] = {"size": 8, "sha256": hashes.sha256(path)}
    record = {
        "repo_id": models.FAMILIAR_GGUF_REPO,
        "revision": models.FAMILIAR_GGUF_REVISION,
        "files": sorted(digests),
        "digests": digests,
    }
    (dest / fetch.MANIFEST_NAME).write_text(
        json.dumps({"repos": {models.FAMILIAR_GGUF_REPO: record}}), encoding="utf-8"
    )
    return files


def test_removing_one_familiar_weights_file_forgets_it_in_the_manifest(tmp_path):
    """A per-file removal in a shared directory leaves the manifest standing, and
    ``verify_manifest`` would then report the removed optional file as missing --
    a corrupt-install verdict for a row the user deliberately dropped."""
    config = _config(tmp_path)
    files = _install_familiar_weights(config)
    removal = fetch.removal_plan(config, _entries("familiar:familiar_mmproj"))
    svc = types.SimpleNamespace(config=config)
    downloads._uninstall(svc, removal, None)
    assert not files["mmproj"].exists()
    assert files["gguf"].exists()
    verdict = fetch.verify_manifest(config.familiar_models_dir)
    assert verdict.status == fetch.VERIFY_OK
    assert verdict.checked == 1
    # Removing the last recorded file leaves no manifest to lie about it.
    downloads._uninstall(
        svc, fetch.removal_plan(config, _entries("familiar:familiar_gguf")), None
    )
    assert not (config.familiar_models_dir / fetch.MANIFEST_NAME).exists()


# --- pipelines-10 / service-gates-05: an old Familiar runtime is outdated ------


def _install_familiar_runtime(config: Config, key: str, digests: dict[str, str] | None) -> object:
    spec = models.FAMILIAR_MODELS[key]
    root = config.familiar_runtime_dir
    root.mkdir(parents=True, exist_ok=True)
    for name in spec.probe:
        (root / name).write_bytes(b"runtime")
    if digests is not None:
        record = {
            "repo_id": "",
            "revision": "",
            "files": sorted(digests),
            "digests": {n: {"size": 7, "sha256": d} for n, d in digests.items()},
        }
        (root / fetch.MANIFEST_NAME).write_text(
            json.dumps({"repos": {"": record}}), encoding="utf-8"
        )
    return spec


def test_a_familiar_runtime_downloaded_before_the_repin_reads_as_outdated(tmp_path):
    """b10948 and b11457 ship the same filenames, so the b10948 install reads
    present and verifies clean against its own manifest -- and cannot load
    EmbeddingGemma 2. ``outdated`` answered () for every kind but ``engine``."""
    config = _config(tmp_path)
    spec = models.FAMILIAR_MODELS["familiar_runtime"]
    old = {name: "0" * 64 for name, _ in spec.digests}
    _install_familiar_runtime(config, "familiar_runtime", old)
    assert fetch.present(config, "familiar", spec) is True
    assert set(fetch.outdated(config, "familiar", spec)) == {n for n, _ in spec.digests}

    class _Svc:
        def __init__(self, cfg):
            self.config = cfg
            self.vram_plan = None

    row = next(
        r for r in downloads.rows(_Svc(config)) if r["row_key"] == "familiar:familiar_runtime"
    )
    assert row["present"] is True
    assert row["outdated"] is True


def test_a_familiar_runtime_installed_at_an_older_pin_reads_outdated(tmp_path):
    config = _config(tmp_path)
    spec = models.FAMILIAR_MODELS["familiar_runtime"]
    digests = dict(spec.digests)
    digests["llama-server.exe"] = "0" * 64
    _install_familiar_runtime(config, "familiar_runtime", digests)
    assert fetch.outdated(config, "familiar", spec) == ("llama-server.exe",)


def test_a_current_or_unrecorded_familiar_runtime_is_never_outdated(tmp_path):
    config = _config(tmp_path)
    main = models.FAMILIAR_MODELS["familiar_runtime"]
    cudart = models.FAMILIAR_MODELS["familiar_runtime_cudart"]
    # No manifest at all: a hand copy, unknown rather than stale.
    _install_familiar_runtime(config, "familiar_runtime", None)
    assert fetch.outdated(config, "familiar", main) == ()
    # Current.
    _install_familiar_runtime(config, "familiar_runtime", dict(main.digests))
    assert fetch.outdated(config, "familiar", main) == ()
    # A directory whose manifest recorded only the *other* zip (two separate
    # installs once overwrote one shared record) says nothing about this row.
    _install_familiar_runtime(config, "familiar_runtime_cudart", dict(cudart.digests))
    assert fetch.outdated(config, "familiar", main) == ()
    assert fetch.outdated(config, "familiar", cudart) == ()


def test_familiar_weights_at_another_commit_are_outdated(tmp_path):
    config = _config(tmp_path)
    spec = models.FAMILIAR_MODELS["familiar_embed"]
    root = config.familiar_models_dir
    root.mkdir(parents=True, exist_ok=True)
    (root / models.FAMILIAR_EMBED_GGUF_FILE).write_bytes(b"w")
    repo = spec.fetch[0].repo_id
    (root / fetch.MANIFEST_NAME).write_text(
        json.dumps({"repos": {repo: {"repo_id": repo, "revision": "f" * 40, "digests": {}}}}),
        encoding="utf-8",
    )
    assert fetch.outdated(config, "familiar", spec) == (f"{repo}@{spec.fetch[0].revision[:8]}",)


# --- create-workspace-03: an imported adapter is listed once, with its licence -


_ARCHIVE = (
    len(b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}').to_bytes(8, "little")
    + b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    + b"\x00" * 4
)


@pytest.fixture
def _clean_style_registry():
    before = dict(models.STYLE_LORAS)
    yield
    models.STYLE_LORAS.clear()
    models.STYLE_LORAS.update(before)


def test_the_catalog_lists_an_imported_style_once_with_its_own_licence(
    tmp_path, _clean_style_registry
):
    config = _config(tmp_path)
    source = tmp_path / "mystyle.safetensors"
    source.write_bytes(_ARCHIVE)
    manifest = generation.import_lora(
        config, source, label="Cosmos", commercial=False, license="CC-BY-NC"
    )
    rows = [row for row in generation.lora_catalog(config) if row["key"] == manifest.key]
    assert len(rows) == 1
    assert rows[0]["source"] == "local file"
    assert rows[0]["commercial"] is False
    assert rows[0]["license"] == "CC-BY-NC"
    # The built-in styles are still listed, once each, as built-in.
    builtin = [row for row in generation.lora_catalog(config) if row["source"] == "built-in"]
    assert {row["key"] for row in builtin} >= set(generation._BUILTIN_STYLE_LORAS)


# --- create-workspace-06: an infinite number in a document ---------------------


def test_from_dict_survives_an_infinite_seed_and_validate_refuses_it():
    """``json.loads`` accepts ``Infinity``; ``int(inf)`` raises ``OverflowError``,
    which the coercers (catching ``TypeError``/``ValueError``) let out of
    ``from_dict`` as a raw exception instead of an issue ``validate_request``
    can name."""
    inf = float("inf")
    for field in ("seed", "count", "schema_version"):
        request = generation.GenerationRequest.from_dict({"prompt": "a cat", field: inf})
        resolved = generation.resolve_recipe(request, None)
        issues = generation.validate_request(request, resolved)
        if field == "seed":
            assert any(issue.field == "seed" for issue in issues)
    request = generation.GenerationRequest.from_dict(
        {
            "prompt": "x",
            "generation_type": "tileset",
            "tile": {"variants": inf, "target_cell_px": inf},
        }
    )
    assert request.tile.variants == inf
    assert request.tile.target_cell_px == inf
    request = generation.GenerationRequest.from_dict(
        {
            "prompt": "x",
            "generation_type": "sprite_sheet",
            "sprite": {"directions": inf, "target_cell_px": inf},
        }
    )
    assert request.sprite.directions == inf
    # The form-decoding adapter takes the same values from a saved form.
    generation.request_from_legacy({"prompt": "x", "seed": inf, "count": inf, "variants": inf})
    generation.request_from_legacy({"prompt": "x", "target_cell_px": inf, "sprite_candidates": inf})


# --- create-workspace-10: a failed manifest replace leaves no temp -------------


def test_a_failed_manifest_replace_leaves_no_temp_file(
    tmp_path, monkeypatch, _clean_style_registry
):
    import os

    config = _config(tmp_path)
    source = tmp_path / "mystyle.safetensors"
    source.write_bytes(_ARCHIVE)
    root = config.t2i_model_root / "loras"

    real_replace = os.replace

    def refuse_manifests(src, dst, *args, **kwargs):
        if Path(dst).name == "manifests.json":
            raise PermissionError("a reader holds manifests.json")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", refuse_manifests)
    # ``Path.replace`` is its own call into the OS, so route it the same way.
    monkeypatch.setattr(
        Path,
        "replace",
        lambda self, target: refuse_manifests(self, target),
    )
    with pytest.raises(PermissionError):
        generation.import_lora(config, source, label="Cosmos")
    assert [p.name for p in root.iterdir() if p.name.endswith(".tmp")] == []


# --- create-brief-07: a directory is not a probe -------------------------------


def test_judge_load_treats_a_directory_at_the_probe_path_as_missing(tmp_path, caplog):
    path = judge.probe_path(tmp_path, "reference")
    path.mkdir()
    with caplog.at_level(logging.ERROR, logger=judge.log.name):
        assert judge.load(path, stage="reference") is None
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    # Sanity: a real probe still loads.
    probe = judge.Probe(
        weights=np.zeros(4), bias=0.0, stage="reference", labels=2, positives=1,
        corpus="c", schema=judge.SCHEMA_VERSION,
    )
    good = tmp_path / "good" / "probe-reference.npz"
    good.parent.mkdir()
    judge.save(probe, good)
    assert judge.load(good, stage="reference") is not None


# --- service-queue-07: the child-process pipe takes the model lease ------------


def test_a_download_is_refused_while_the_out_of_process_image_pipe_is_generating(
    tmp_path, monkeypatch
):
    """The shipped default runs the image pipe in a child, and the client took no
    lease, so ``maintain()`` saw no users and a Download could start while the
    child was mid-``from_pretrained``. The lease is held across the exchange."""
    from realmspinner.pipelines.t2i_client import Text2ImageClient

    monkeypatch.setattr(leases, "MAINTAIN_TIMEOUT", 0.05)
    client = Text2ImageClient(models.BASE_MODELS["sdxl_cfg"], tmp_path)
    seen: dict[str, object] = {}

    def fake_request(payload, **kwargs):
        # What a download from the service thread does while the child works.
        outcome: list[BaseException | None] = []

        def attempt() -> None:
            try:
                with downloads._maintenance("downloading a model"):
                    outcome.append(None)
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=attempt)
        thread.start()
        thread.join(5)
        seen[payload["op"]] = outcome[0]
        return {"path": str(tmp_path / "out.png"), "prompt": "p", "recipe": {}}

    monkeypatch.setattr(client, "_request", fake_request)
    client.load()
    client.generate("a cat", tmp_path / "out.png")
    assert isinstance(seen["load"], Conflict)
    assert isinstance(seen["generate"], Conflict)
    # And the lease is released afterwards: a download may run between jobs.
    with downloads._maintenance("downloading a model"):
        pass


def test_unload_and_close_do_not_wait_on_the_lease(tmp_path):
    """``unload`` and ``close`` run on the event loop thread; taking a shared lease
    there would park the loop behind a download that holds the exclusive one for
    as long as the fetch lasts."""
    from realmspinner.pipelines.t2i_client import Text2ImageClient

    client = Text2ImageClient(models.BASE_MODELS["sdxl_cfg"], tmp_path)
    done = threading.Event()

    def run() -> None:
        client.unload()
        client.close()
        done.set()

    with leases.MODELS.maintain():
        thread = threading.Thread(target=run)
        thread.start()
        assert done.wait(5), "unload/close blocked behind the maintainer"
    thread.join(5)


# --- pipelines-28: the WELD_TOLERANCE comment ----------------------------------


def test_weld_tolerance_comment_names_blender_worker_as_a_consumer():
    source = Path(meshreport.__file__).read_text(encoding="utf-8")
    block = source[: source.index("WELD_TOLERANCE = ")]
    comment = block[block.rindex("# Two vertices closer") :]
    assert "blender_worker" in comment
    assert "nothing in the stored" not in comment


# --- docs-D-03: the GGUF row's size --------------------------------------------


#: The sum of the ``*.gguf`` files (minus ``q4/`` and ``q8/``) at the pinned
#: revision, in bytes, as the completed download recorded it in its manifest
#: (``.realmspinner-fetch.json``, 2026-10-14): 16,467,590,304 B.
_TRELLIS_GGUF_BYTES = 16_467_590_304


def test_the_trellis_gguf_row_size_matches_the_revisions_file_total():
    spec = models.ENGINE_MODELS["trellis_gguf"]
    assert spec.fetch[0].revision == "a57397bd3d351599d9729fc144b3f87c3f87d65b"
    assert spec.fetch[0].size_gib == pytest.approx(_TRELLIS_GGUF_BYTES / 1024**3, abs=0.01)


# --- docs-D-05: the separation licence note ------------------------------------


def test_the_separation_license_note_names_the_checkpoint_it_ships():
    spec = models.SEPARATION_MODELS[models.DEFAULT_SEPARATION]
    assert "htdemucs" not in spec.license_note
    assert spec.fetch[0].filename.removesuffix(".pt") in spec.license_note
