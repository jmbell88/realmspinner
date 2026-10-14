"""Six bare-exception/logic gaps from the 2026-09-26 audit's "Remaining
findings", closed together because each is a small, self-contained fix in a
door under ``service/``.

**service-assets-04.** ``export._safe_export_name`` ran its drive-letter and
separator checks on the unstripped ``name`` but returned ``name.strip()``:
leading whitespace shifted a Windows drive letter past position 0 for the
check while the stripped return value still had it at position 0.

**service-assets-05.** ``sweeps.expand`` built a dict keyed by
``unit.server_group(...)`` (a tuple of raw axis values); an axis value that is
a list or dict makes that tuple unhashable and raises a bare ``TypeError``
before any unit's own validation gets a chance to refuse it by name.

**service-assets-06.** ``sprites.resolve_sheet_kind``'s ``int(directions)``
and ``sprites.sprite_cost``'s ``int(logical_size)`` raised bare
``ValueError``/``TypeError`` instead of ``service.errors.Invalid``/a
"not drawable" reply.

**service-assets-07.** ``tilesheets.create_tile_sheet`` iterated
``prompt_items`` directly in materials mode: a plain string is itself
iterable, so a caller that sent the whole materials text as one string (not a
list) had it walked character by character -- ``"grass"`` became five
one-letter materials.

**service-assets-08.** ``characters.export_frames`` -- removed with the
character sheets (``retired.py``), and its regressions with it.

**service-assets-09.** ``loras.library_training_set``'s favourites source had
no stage filter, unlike ``accepted_references``/``usable_meshes`` -- a
favourited ``tile_sheet`` row's own ``input.png`` (a 64-cell grid) could reach
the training corpus as if it were a single-subject reference.
"""

from __future__ import annotations

import pytest
from PIL import Image

from realmspinner.service import loras as svc_loras
from realmspinner.service import sprites as svc_sprites
from realmspinner.service import sweeps as svc_sweeps
from realmspinner.service import tilesheets as svc_tilesheets
from realmspinner.service.errors import Invalid
from realmspinner.service.export import _safe_export_name

# --- service-assets-04 -------------------------------------------------------


def test_a_leading_space_does_not_smuggle_a_drive_letter_past_the_check():
    with pytest.raises(Invalid):
        _safe_export_name(" C:evil")


def test_an_ordinary_name_still_round_trips():
    assert _safe_export_name("  Barrel  ") == "Barrel"


# --- service-assets-05 -------------------------------------------------------


def test_expand_does_not_raise_unhashable_when_a_server_axis_value_is_a_list():
    plan = svc_sweeps.SweepPlan(
        label="s", prompt="a crate", base={}, seeds=(1,),
        axes=(svc_sweeps.Axis(param="trellis_atlas", values=([1, 2],)),),
    )
    units = svc_sweeps.expand(plan)
    # Two units: the baseline and the one list-valued axis unit -- ordering
    # ran to completion rather than raising, which is the whole claim.
    assert len(units) == 2


# --- service-assets-06 -------------------------------------------------------


def test_resolve_sheet_kind_refuses_a_non_numeric_directions_by_name():
    with pytest.raises(Invalid) as excinfo:
        svc_sprites.resolve_sheet_kind(action="turnaround", directions="many")
    assert excinfo.value.field == "directions"


def test_sprite_cost_reports_not_drawable_for_a_non_numeric_logical_size():
    result = svc_sprites.sprite_cost("turnaround", logical_size="huge")
    assert result["drawable"] is False
    assert result["logical_size"] == "huge"


# --- service-assets-07 -------------------------------------------------------


def test_a_bare_materials_string_is_one_line_not_one_material_per_character(svc):
    result = svc_tilesheets.create_tile_sheet(
        svc, prompt="a dungeon", mode="materials", prompt_items="grass",
    )
    job = svc.store.get(result["id"])
    materials = job["params"]["sheet"]["materials"]
    assert len(materials) == 1
    assert materials[0]["prompt"] == "grass"


# --- service-assets-09 -------------------------------------------------------


def _favourited_reference(svc, name: str) -> str:
    job_id = svc.store.create("text", "a thing", {}, stage="reference", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (10, 20, 30)).save(job_dir / name)
    svc.store.set_meta(job_id, favorite=True)
    return job_id


def test_a_favourited_tile_sheet_never_reaches_the_training_corpus(svc):
    # Padded past ``lora_train.MIN_IMAGES`` with two ordinary favourited
    # references, so the refusal under test is "excluded", not "too few
    # candidates to even try".
    kept_a = _favourited_reference(svc, "reference.png")
    kept_b = _favourited_reference(svc, "input.png")
    kept_c = _favourited_reference(svc, "reference.png")

    tile_job = svc.store.create(
        "tile_sheet", "a dungeon floor", {}, stage="tilesheet", status="done",
    )
    tile_dir = svc.job_dir(tile_job)
    tile_dir.mkdir(parents=True, exist_ok=True)
    # The 64-cell grid a tile-sheet job's own door writes -- see
    # ``_q_tileset.py``/``_q_tilesheet.py``'s ``out_png = job_dir /
    # "input.png"`` -- not a single subject a style LoRA should train on.
    Image.new("RGB", (512, 512), (10, 20, 30)).save(tile_dir / "input.png")
    svc.store.set_meta(tile_job, favorite=True)

    result = svc_loras.library_training_set(
        svc, accepted_references=False, usable_meshes=False, dedupe=False,
    )

    paths = set(result["paths"])
    assert svc.job_dir(kept_a) / "reference.png" in paths
    assert svc.job_dir(kept_b) / "input.png" in paths
    assert svc.job_dir(kept_c) / "reference.png" in paths
    assert tile_dir / "input.png" not in paths
