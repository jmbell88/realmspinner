"""``history_panel``: the Earlier meshes list and its one door, Restore.

The row-building functions are pure (no imgui), so they are tested directly
against a ``model_history`` list -- and ``_confirm_restore`` is tested the
way ``candidates_panel``'s own confirm-then-act functions are: drive it
against a fake ``ctx`` and check nothing is submitted until the confirm's
``on_confirm`` is actually called.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class _FakeConfirms:
    def __init__(self) -> None:
        self.asked: list[Any] = []

    def ask(self, confirm: Any) -> None:
        self.asked.append(confirm)


def _ctx(**extra: Any) -> SimpleNamespace:
    submits: list[tuple] = []
    ns = SimpleNamespace(
        confirms=_FakeConfirms(),
        svc=object(),
        submits=submits,
        submit=lambda key, fn, *a, **k: submits.append((key, fn, a, k)) or True,
    )
    for key, value in extra.items():
        setattr(ns, key, value)
    return ns


# --- pure row data ------------------------------------------------------------


def test_entries_for_lists_newest_first():
    from realmspinner.studio.panes import history_panel

    job = {
        "params": {
            "model_history": [
                {"n": 1, "kind": "optimize"},
                {"n": 2, "kind": "remesh"},
            ]
        }
    }
    assert [e["n"] for e in history_panel.entries_for(job)] == [2, 1]


def test_entries_for_an_asset_with_no_history_is_empty():
    from realmspinner.studio.panes import history_panel

    assert history_panel.entries_for({"params": {}}) == []
    assert history_panel.entries_for({}) == []


def test_row_label_reads_the_kind_table():
    from realmspinner.studio.panes import history_panel

    assert history_panel.row_label({"kind": "remesh"}) == "Game-ready remesh"
    assert history_panel.row_label({"kind": "retexture"}) == "Surface texture"
    assert history_panel.row_label({"kind": "optimize"}) == "Triangle budget"
    assert history_panel.row_label({"kind": "revert"}) == "Restored mesh"
    # An unfamiliar kind (a hand-edited row, a future rework) falls back to
    # the raw string rather than raising.
    assert history_panel.row_label({"kind": "future-thing"}) == "future-thing"


def test_row_detail_combines_the_recorded_text_and_the_kept_size():
    from realmspinner.studio.panes import history_panel

    detail = history_panel.row_detail({"detail": "8,000 quads", "bytes": 2048})
    assert detail == "8,000 quads (2 KB)"
    assert history_panel.row_detail({"detail": "", "bytes": 512}) == "512 B"
    no_size = history_panel.row_detail({"detail": "6 views restyled", "bytes": None})
    assert no_size == "6 views restyled"


# --- the confirm-then-submit door ---------------------------------------------


def test_restore_confirms_before_it_submits():
    from realmspinner.studio.panes import history_panel

    ctx = _ctx()
    job = {
        "id": "aaaaaaaaaaaa",
        "params": {"model_history": [{"n": 3, "kind": "remesh", "geometry": True}]},
    }
    entry = job["params"]["model_history"][0]

    history_panel._confirm_restore(ctx, job, entry)

    assert ctx.submits == [], "must not submit until the confirm is answered"
    assert len(ctx.confirms.asked) == 1
    confirm = ctx.confirms.asked[0]
    assert confirm.confirm_label == "Restore"

    confirm.on_confirm()

    assert len(ctx.submits) == 1
    key, fn, args, kwargs = ctx.submits[0]
    assert key == "model-revert:aaaaaaaaaaaa"
    assert fn is history_panel.svc_jobs.revert_model
    assert args == (ctx.svc, "aaaaaaaaaaaa")
    assert kwargs == {"version": 3}


def test_restore_names_the_rig_when_geometry_crossed():
    from realmspinner.studio.panes import history_panel

    ctx = _ctx()
    job = {
        "id": "aaaaaaaaaaaa",
        "params": {"model_history": [{"n": 1, "kind": "remesh", "geometry": True}]},
    }
    entry = job["params"]["model_history"][0]

    history_panel._confirm_restore(ctx, job, entry)

    message = ctx.confirms.asked[0].message
    assert "rig" in message.lower()


def test_restore_of_a_pure_surface_version_says_the_rig_is_unaffected():
    from realmspinner.studio.panes import history_panel

    ctx = _ctx()
    job = {
        "id": "aaaaaaaaaaaa",
        "params": {"model_history": [{"n": 1, "kind": "retexture", "geometry": False}]},
    }
    entry = job["params"]["model_history"][0]

    history_panel._confirm_restore(ctx, job, entry)

    message = ctx.confirms.asked[0].message
    assert "unaffected" in message.lower()
