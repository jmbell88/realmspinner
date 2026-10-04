"""plotter-23 (2026-10-03 audit): the reroll gate ignored the cell count the door applies."""

from __future__ import annotations

from realmspinner import models
from realmspinner.service import jobs as svc_jobs
from realmspinner.service import tilesheets


def _delete_ip_adapter_weights(svc) -> None:
    adapter = models.IP_ADAPTERS["plus"]
    root = svc.config.t2i_model_root / adapter.dir_name
    (root / adapter.subfolder / adapter.weight_name).unlink()
    (root / adapter.image_encoder_dir / "config.json").unlink()


def test_rerolling_a_one_material_style_locked_sheet_does_not_require_the_adapter(svc):
    made = tilesheets.create_tile_sheet(
        svc, prompt="a temperate ruin", tile_size=32, view="top_down",
        mode=tilesheets.MODE_MATERIALS, prompt_items=("mossy cobblestone",),
        style_lock=True,
    )
    svc.store.set_status(made["id"], "done")
    # The sheet was admitted and drawn without the encoder; the host has since
    # lost the adapter. A reroll runs the same one pass and needs nothing new.
    _delete_ip_adapter_weights(svc)

    new = svc_jobs.rerun_job(svc, made["id"], mode="reroll")

    assert new["id"] != made["id"]
