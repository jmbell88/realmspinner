"""service-kinds-09 (2026-09-26 audit): ``rerun_job`` never refused a
``lora_train`` row by name, though ``rerollable`` already says no to it
(the 2026-09-14 audit's service-05, closed for the *predicate* in
``tests/service/test_reroll_offers.py``). ``rerollable`` only decides
whether the Reroll button is offered; ``rerun_job`` is the door that
actually matters, and a caller that reached it anyway -- a stale UI state,
a direct service or MCP call -- got a job admitted and queued, which then
failed at dispatch with "this job has no training images" instead of being
refused here with a reason and a field to point at.

Mirrors ``test_rerun_job_itself_still_refuses_a_stem_split_reroll`` in that
file, the one other kind ``rerollable`` already refuses unconditionally.
"""

from __future__ import annotations

import pytest

from realmspinner.service._jobs_resubmit import rerollable, rerun_job
from realmspinner.service.errors import Invalid


def test_rerun_job_itself_refuses_a_lora_train_reroll_before_it_is_ever_queued(svc):
    job_id = svc.store.create(
        "lora_train", "Cosmos",
        {"base_model": "sdxl_cfg", "label": "Cosmos", "trigger": "cosmos style"},
        "0123456789ab",
        stage="model",
        status="done",
    )
    with pytest.raises(Invalid, match="LoRA training run has no seed"):
        rerun_job(svc, job_id, mode="reroll")
    # And it was refused before a row was ever minted -- no stray queued
    # job left behind for the worker to fail later.
    assert svc.store.get(job_id)["status"] == "done"


@pytest.mark.parametrize("mode", ["reroll", "remesh"])
def test_rerun_job_refuses_a_lora_train_row_in_either_mode(svc, mode):
    job_id = svc.store.create(
        "lora_train", "Cosmos",
        {"base_model": "sdxl_cfg", "label": "Cosmos", "trigger": "cosmos style"},
        "0123456789ac",
        stage="model",
        status="done",
    )
    with pytest.raises(Invalid, match="LoRA training run has no seed"):
        rerun_job(svc, job_id, mode=mode)


def test_rerollable_and_rerun_job_now_agree_about_lora_train():
    """The predicate already said no (2026-09-14 audit); this is the door
    catching up to it."""
    assert (
        rerollable({"id": "a", "kind": "lora_train", "status": "done", "params": {}})
        is False
    )
