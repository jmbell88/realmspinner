"""``rerollable()``'s whole job is to stop the app *offering* a press that
would only buy an error toast (its own docstring's words) -- so every kind
``rerun_job`` refuses unconditionally, by name, must read back as not
rerollable. Two escaped that rule (the 2026-09-14 audit):

* service-04 -- a finished "separate" (stem split) job. ``rerun_job`` always
  refuses it: a split is deterministic and has no seed to change.
* service-05 -- a finished "lora_train" job. ``rerun_job`` never refuses it by
  name, but the row it mints gets none of the ``train/`` directory of images
  the worker needs (``rerun_job`` copies only image dirs, ``ref.png`` and
  ``source.wav``), so the reroll queues and then fails at dispatch with "this
  job has no training images" -- a failure with no field to point a refusal
  at, because it never reaches a door.

Both are pure-predicate tests: ``rerollable`` takes a job dict, no ``svc``, no
disk, so the frame thread can call it per card.
"""

from __future__ import annotations

import pytest

from realmspinner.service._jobs_resubmit import rerollable, rerun_job
from realmspinner.service.errors import Invalid


def _done(kind: str, **extra) -> dict:
    return {"id": "a", "kind": kind, "status": "done", "stage": "model", "params": {}, **extra}


# --- service-04: a stem split has no seed to change --------------------------


def test_a_finished_stem_split_is_not_offered_a_reroll():
    assert rerollable(_done("separate")) is False


@pytest.mark.parametrize("status", ["error", "cancelled"])
def test_a_stem_split_is_not_offered_a_reroll_whatever_terminal_status_it_ended_in(status):
    assert rerollable(_done("separate", status=status)) is False


def test_rerun_job_itself_still_refuses_a_stem_split_reroll(svc):
    """rerollable's own refusal must agree with the door it guards: this is
    the failure a stray Reroll press against the door still gets, and it is
    unconditional -- both modes, by name."""
    job_id = svc.store.create(
        "separate", "a take", {}, "0123456789ab", stage="model", status="done"
    )
    with pytest.raises(Invalid, match="stem split has no seed"):
        rerun_job(svc, job_id, mode="reroll")


# --- service-05: a LoRA training row has no train/ images to carry forward ---


def test_a_finished_lora_train_job_is_not_offered_a_reroll():
    assert rerollable(_done("lora_train")) is False


@pytest.mark.parametrize("status", ["error", "cancelled"])
def test_a_lora_train_job_is_not_offered_a_reroll_whatever_terminal_status_it_ended_in(status):
    assert rerollable(_done("lora_train", status=status)) is False


# --- the existing rules must survive both additions ---------------------------


def test_an_ordinary_finished_mesh_is_still_rerollable():
    assert rerollable(_done("image", stage="model")) is True


def test_a_built_asset_is_still_not_rerollable():
    assert rerollable(_done("image", stage="model", params={"built": True})) is False


def test_a_hand_made_reference_is_still_not_rerollable():
    assert rerollable(_done("image", stage="reference")) is False


def test_a_job_still_in_flight_is_not_rerollable():
    assert rerollable({"id": "a", "kind": "separate", "status": "running", "params": {}}) is False
