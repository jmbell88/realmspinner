"""A create-side cap greys the button before the press, not after.

The 2026-09-06 audit, second run, findings create2-06 and create2-08: the
same defect in two files. ``sprite_panel._submit`` graded its button without
ever reading ``MAX_SPRITE_DRAFTS`` (and the since-removed ``sheet_panel`` the
same for ``MAX_SHEETS``), so the cap was discovered only after a wasted press
as a fieldless ``Conflict`` toast. The pane now carries a pure
``*_cap_reason`` function beside the draw call, following the shape
``retarget_panel.dependent_job_reason`` established earlier the same day:
a function over state the frame already has (``draft_records`` and
``ctx.cache.jobs``), with no sqlite and no directory listing of its own --
testable here with no imgui in play.
"""

from __future__ import annotations

from typing import Any

from realmspinner.kernels.rig import store
from realmspinner.studio.panes import sprite_panel


def _job(status: str, source_job: str, kind: str, **extra: Any) -> dict[str, Any]:
    params = {"source_job": source_job, **extra}
    return {"id": "x", "status": status, "kind": kind, "params": params}


def test_sprite_panel_states_the_draft_cap_before_the_button_is_pressed():
    records = [{"id": str(i)} for i in range(store.MAX_SPRITE_DRAFTS)]
    assert sprite_panel.sprite_draft_cap_reason(records, [], "draw1") is not None
    assert sprite_panel.sprite_draft_cap_reason(records[:-1], [], "draw1") is None


def test_sprite_draft_cap_counts_a_queued_sibling_synthesis_too():
    records = [{"id": str(i)} for i in range(store.MAX_SPRITE_DRAFTS - 1)]
    queued = [_job("running", "draw1", "sprite_synthesis")]
    assert sprite_panel.sprite_draft_cap_reason(records, queued, "draw1") is not None


def test_sprite_draft_cap_ignores_an_unrelated_or_finished_job():
    records = [{"id": str(i)} for i in range(store.MAX_SPRITE_DRAFTS - 1)]
    other_asset = [_job("queued", "draw2", "sprite_synthesis")]
    finished = [_job("done", "draw1", "sprite_synthesis")]
    assert sprite_panel.sprite_draft_cap_reason(records, other_asset, "draw1") is None
    assert sprite_panel.sprite_draft_cap_reason(records, finished, "draw1") is None
