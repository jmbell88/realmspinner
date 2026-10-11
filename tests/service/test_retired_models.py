"""The retired-model ledger's service half: find what is left, delete it safely.

``downloads.uninstall`` works on registered rows, so once a row leaves the
registry its folder has no button. ``retired_present`` / ``remove_retired`` are
that button's two halves, and every test here runs against the ``svc`` fixture's
throwaway model root.
"""

from __future__ import annotations

import pytest

from realmspinner import models
from realmspinner.service import downloads as svc_downloads
from realmspinner.service.errors import Conflict, Invalid, NotFound


def _root(svc):
    return svc.config.t2i_model_root


def _litter(svc):
    """Drop a retired checkpoint folder, a retired LoRA file and a stranded folder."""
    root = _root(svc)
    (root / "juggernaut-xl-v9").mkdir()
    (root / "juggernaut-xl-v9" / "a.bin").write_bytes(b"x" * 1000)
    (root / "juggernaut-xl-v9" / "b.bin").write_bytes(b"x" * 24)
    (root / "loras").mkdir(exist_ok=True)
    (root / "loras" / "Hyper-SDXL-4steps-lora.safetensors").write_bytes(b"y" * 10)
    (root / "familiar-q2").mkdir()


def test_nothing_retired_on_disk_lists_nothing(svc):
    assert svc_downloads.retired_present(svc) == []


def test_retired_leftovers_are_listed_with_measured_sizes(svc):
    _litter(svc)
    found = {item["key"]: item for item in svc_downloads.retired_present(svc)}
    assert set(found) == {"juggernaut", "sdxl", "familiar-q2"}
    assert found["juggernaut"]["size_bytes"] == 1024
    assert found["sdxl"]["size_bytes"] == 10
    assert found["juggernaut"]["label"] == "Juggernaut XL v9"
    assert found["juggernaut"]["paths"] == [str(_root(svc) / "juggernaut-xl-v9")]


def test_removing_a_retired_folder_deletes_it_and_reports_what_it_freed(svc):
    _litter(svc)
    result = svc_downloads.remove_retired(svc, "juggernaut")
    assert not (_root(svc) / "juggernaut-xl-v9").exists()
    assert result["removed"] == [str(_root(svc) / "juggernaut-xl-v9")]
    assert not [p for p in _root(svc).iterdir() if p.name.startswith(svc_downloads.TRASH_PREFIX)]
    # The rest of the ledger is untouched, and the listing no longer names it.
    assert (_root(svc) / "familiar-q2").exists()
    assert "juggernaut" not in {i["key"] for i in svc_downloads.retired_present(svc)}


def test_removing_a_retired_tier_takes_only_its_adapter_never_the_shared_checkpoint(svc):
    """The retired ``sdxl`` tier shared ``sdxl-base-1.0`` with the default."""
    _litter(svc)
    base = _root(svc) / "sdxl-base-1.0"
    assert base.is_dir()
    svc_downloads.remove_retired(svc, "sdxl")
    assert not (_root(svc) / "loras" / "Hyper-SDXL-4steps-lora.safetensors").exists()
    assert base.is_dir(), "the checkpoint sdxl_cfg stands on must survive"


def test_a_queued_job_blocks_a_retired_removal_with_nothing_deleted(svc):
    _litter(svc)
    svc.store.create("text", "a barrel", {"seed": 1}, stage="reference")
    with pytest.raises(Conflict, match="queued or running"):
        svc_downloads.remove_retired(svc, "juggernaut")
    assert (_root(svc) / "juggernaut-xl-v9").exists()


def test_a_key_outside_the_ledger_is_not_found(svc):
    with pytest.raises(NotFound):
        svc_downloads.remove_retired(svc, "sdxl_cfg")
    with pytest.raises(NotFound):
        svc_downloads.remove_retired(svc, "../outside")


def test_an_absent_retired_model_has_nothing_to_remove(svc):
    with pytest.raises(Invalid, match="nothing to remove"):
        svc_downloads.remove_retired(svc, "dreamshaper")


def test_a_ledger_path_a_live_row_stands_on_is_refused(svc, monkeypatch):
    """The guard behind the pin in ``test_models``: a ledger entry that listed
    the shared checkpoint would otherwise delete the default model."""
    bad = models.RetiredModel("bad", "folder", "Bad", "2026-10-11", ("sdxl-base-1.0",))
    monkeypatch.setitem(models.RETIRED_MODELS, "bad", bad)
    with pytest.raises(Invalid, match="still used"):
        svc_downloads.remove_retired(svc, "bad")
    assert (_root(svc) / "sdxl-base-1.0").is_dir()


def test_a_ledger_path_that_escapes_the_model_root_is_refused(svc, monkeypatch, tmp_path):
    outside = tmp_path / "outside-the-root"
    outside.mkdir()
    bad = models.RetiredModel("bad", "folder", "Bad", "2026-10-11", ("../outside-the-root",))
    monkeypatch.setitem(models.RETIRED_MODELS, "bad", bad)
    with pytest.raises(Invalid, match="outside the model root"):
        svc_downloads.remove_retired(svc, "bad")
    assert outside.is_dir()
