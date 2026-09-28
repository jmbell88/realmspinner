"""plotter-tiles-04 (2026-09-26 audit): the door and ``vram`` both treated
``style_lock`` as needing the IP-Adapter even for a one-cell sheet the worker
will never actually lock.

``_q_tileset.py``'s own worker only hands the first material to the adapter
``if style_lock and count > 1`` -- a one-material request has nothing after
the first material to condition on. ``tilesheets._check_weights`` and
``vram.estimate``'s ``tile_sheet`` branch both required the encoder's weights
whenever ``style_lock`` was set, regardless of how many materials were named,
refusing a one-cell style-locked request on a host that would have fit the
job the worker actually runs.
"""

from __future__ import annotations

import pytest

from realmspinner import models
from realmspinner.service import tilesheets
from realmspinner.service.errors import Invalid


def _delete_ip_adapter_weights(svc) -> None:
    adapter = models.IP_ADAPTERS["plus"]
    root = svc.config.t2i_model_root / adapter.dir_name
    (root / adapter.subfolder / adapter.weight_name).unlink()
    (root / adapter.image_encoder_dir / "config.json").unlink()


def test_a_one_material_style_locked_sheet_builds_without_the_ip_adapter(svc):
    _delete_ip_adapter_weights(svc)

    result = tilesheets.create_tile_sheet(
        svc, prompt="a temperate ruin", tile_size=32, view="top_down",
        mode=tilesheets.MODE_MATERIALS, prompt_items=("mossy cobblestone",),
        style_lock=True,
    )
    assert result["id"]


def test_a_multi_material_style_locked_sheet_still_needs_the_ip_adapter(svc):
    """The fix must not remove the requirement for the request that actually
    uses the encoder -- style_lock with more than one material really does
    reach ``_q_tileset.py``'s ``later_cond`` branch."""
    _delete_ip_adapter_weights(svc)

    with pytest.raises(Invalid) as excinfo:
        tilesheets.create_tile_sheet(
            svc, prompt="a temperate ruin", tile_size=32, view="top_down",
            mode=tilesheets.MODE_MATERIALS,
            prompt_items=("mossy cobblestone", "cracked dry mud"),
            style_lock=True,
        )
    assert "adapter" in excinfo.value.field or "IP-Adapter" in excinfo.value.message
