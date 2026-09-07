"""Everywhere one asset can go, answered once and drawn twice.

Before :mod:`warlock.studio.asset_exits` existed, the library's overflow menu
and the inspector's "Take it somewhere" section each grew their own list of
destinations, one bridge at a time, and stopped agreeing about what was on it
-- the library offered Poser and the two reopen doors, the inspector did not.
This file is the table both surfaces are now measured against, modelled on
``tests/test_inspector_edit_actions.py``: a fake ``ctx``, no imgui, no GL.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from warlock.studio import asset_exits, modes, verbs
from warlock.studio.panes import inspector, library
from warlock.studio.state import AppState


class FakeCtx:
    def __init__(self, svc: Any, mode: str = "library", stage: str = "reference") -> None:
        self.svc = svc
        self.state = AppState()
        self.state.mode = mode
        self.state.create_stage = stage
        self.state.selected = None

    def job(self) -> Any:
        return None


def _job(svc, kind: str, *, stage: str, status: str, params: dict | None = None) -> dict:
    job_id = svc.store.create(kind, "a chest", params or {}, stage=stage, status=status)
    return svc.store.get(job_id)


def _reference(svc, *, status: str = "done", tileset: bool = False, ready: bool = True) -> dict:
    params = {"asset_intent": "tileset"} if tileset else {}
    job = _job(svc, "text", stage="reference", status=status, params=params)
    job["files"] = ["input.png"] if ready else []
    return job


def _mesh(svc, *, rigged: bool = False, status: str = "done") -> dict:
    job = _job(svc, "image", stage="model", status=status, params={})
    job["files"] = ["model.glb", "rig.glb"] if rigged else ["model.glb"]
    return job


def _charsheet(svc, source_id: str, *, status: str = "done") -> dict:
    job = _job(
        svc, "charsheet", stage="model", status=status,
        params={"source_job": source_id, "sheet_id": "s1"},
    )
    job["files"] = []
    return job


def _authored(svc, authored: str) -> dict:
    job = _job(svc, "text", stage="reference", status="done", params={"authored": authored})
    job["files"] = ["input.png"]
    return job


def _music(svc, *, status: str = "done", has_track: bool = True) -> dict:
    job = _job(svc, "music", stage="music", status=status, params={})
    job["files"] = ["track.wav"] if has_track else []
    return job


def _rows(svc) -> dict[str, dict]:
    mesh_id = svc.store.create("image", "a chest", {}, stage="model", status="done")
    mesh_row = svc.store.get(mesh_id)
    mesh_row["files"] = ["model.glb"]

    deleted = _mesh(svc, rigged=True)
    deleted["deleted_at"] = 12345.0

    return {
        "reference": _reference(svc),
        "tileset_reference": _reference(svc, tileset=True),
        "mesh": _mesh(svc),
        "rigged_mesh": _mesh(svc, rigged=True),
        "charsheet": _charsheet(svc, mesh_row["id"]),
        "authored_map": _authored(svc, "plotter"),
        "authored_atlas": _authored(svc, "packwright"),
        "music_take": _music(svc),
        "errored_reference": _reference(svc, status="error", ready=False),
        "deleted": deleted,
    }


def _labels(exits: list) -> dict[str, tuple[str, bool]]:
    """``{mode: (label, dimmed)}`` -- good enough for a table this small,
    where no row hands back two exits on the same mode with the same
    dimmed-ness."""
    return {e.mode: (e.label, bool(e.reason)) for e in exits}


# --- the table ----------------------------------------------------------


def test_reference_offers_inker_and_both_add_doors(svc):
    ctx = FakeCtx(svc)
    exits = asset_exits.exits_for(ctx, _rows(svc)["reference"])
    modes_present = {e.mode for e in exits if not e.reason}
    assert modes_present == {"inker", "plotter", "packwright"}
    assert not any(e.reason for e in exits)


def test_a_tileset_reference_offers_exactly_the_same_run(svc):
    """The inspector used to gate the two Add-to doors on
    ``asset_intent == "tileset"``, which the library's own menu never did --
    the drift this module exists to close. A plain reference and a
    tileset-flagged one must now reach identical destinations."""
    ctx = FakeCtx(svc)
    rows = _rows(svc)
    plain = {e.mode for e in asset_exits.exits_for(ctx, rows["reference"])}
    tileset = {e.mode for e in asset_exits.exits_for(ctx, rows["tileset_reference"])}
    assert plain == tileset == {"inker", "plotter", "packwright"}


def test_an_unrigged_mesh_offers_clay_and_troupe_and_dims_poser(svc):
    ctx = FakeCtx(svc)
    exits = asset_exits.exits_for(ctx, _rows(svc)["mesh"])
    by_mode = _labels(exits)
    assert set(by_mode) == {"clay", "poser", "troupe"}
    assert by_mode["clay"][1] is False
    assert by_mode["troupe"][1] is False
    assert by_mode["poser"][1] is True
    poser = next(e for e in exits if e.mode == "poser")
    assert "Rig" in poser.reason


def test_a_rigged_mesh_dims_nothing(svc):
    ctx = FakeCtx(svc)
    exits = asset_exits.exits_for(ctx, _rows(svc)["rigged_mesh"])
    by_mode = _labels(exits)
    assert set(by_mode) == {"clay", "poser", "troupe"}
    assert not any(dimmed for _label, dimmed in by_mode.values())


def test_a_charsheet_offers_only_the_way_back_into_troupe(svc):
    """Regression: a charsheet row carries ``stage == "model"`` -- the column
    default every follow-up product wears, per ``asset_open``'s own docstring
    -- and without excluding rows that carry a ``source_job`` this showed a
    dimmed "Send to Troupe" for the mesh a character sheet does not have."""
    ctx = FakeCtx(svc)
    exits = asset_exits.exits_for(ctx, _rows(svc)["charsheet"])
    by_mode = _labels(exits)
    assert set(by_mode) == {"troupe"}
    assert by_mode["troupe"][1] is False
    assert by_mode["troupe"][0] == verbs.open_in("troupe")


def test_an_authored_map_offers_both_plotter_doors_plus_inker_and_packwright(svc):
    ctx = FakeCtx(svc)
    exits = asset_exits.exits_for(ctx, _rows(svc)["authored_map"])
    modes_present = [e.mode for e in exits]
    assert modes_present.count("plotter") == 2
    labels = {e.label for e in exits if e.mode == "plotter"}
    assert labels == {verbs.open_in("plotter"), verbs.add_to("plotter", "as a tileset")}
    assert {"inker", "packwright"} <= {e.mode for e in exits}
    assert not any(e.reason for e in exits)


def test_an_authored_atlas_offers_both_packwright_doors_plus_inker_and_plotter(svc):
    ctx = FakeCtx(svc)
    exits = asset_exits.exits_for(ctx, _rows(svc)["authored_atlas"])
    modes_present = [e.mode for e in exits]
    assert modes_present.count("packwright") == 2
    labels = {e.label for e in exits if e.mode == "packwright"}
    assert labels == {verbs.open_in("packwright"), verbs.add_to("packwright", "as an atlas source")}
    assert {"inker", "plotter"} <= {e.mode for e in exits}
    assert not any(e.reason for e in exits)


def test_a_finished_take_offers_sirens_and_nothing_else(svc):
    ctx = FakeCtx(svc)
    exits = asset_exits.exits_for(ctx, _rows(svc)["music_take"])
    by_mode = _labels(exits)
    assert set(by_mode) == {"sirens"}
    assert by_mode["sirens"][1] is False


def test_an_errored_reference_dims_every_reachable_door_and_hides_nothing_extra(svc):
    ctx = FakeCtx(svc)
    exits = asset_exits.exits_for(ctx, _rows(svc)["errored_reference"])
    by_mode = _labels(exits)
    assert set(by_mode) == {"inker", "plotter", "packwright"}
    assert all(dimmed for _label, dimmed in by_mode.values())
    for e in exits:
        assert "failed" in e.reason.lower()


def test_a_deleted_row_offers_nothing_at_all(svc):
    """Every action a builder could offer either fails outright or quietly
    resurrects a trashed row into the workshop without saying so -- the same
    rule ``panes.library._trash_menu`` already applies to its own menu."""
    ctx = FakeCtx(svc)
    assert asset_exits.exits_for(ctx, _rows(svc)["deleted"]) == []


# --- cross-cutting rules --------------------------------------------------


def test_every_label_is_a_verbs_product(svc):
    """No hand-written strings -- ``tests/test_ux_shared_vocabulary.py``
    polices this app-wide, and this is the module every exit's label now
    flows through."""

    def _looks_like_a_verb(label: str) -> bool:
        bare = label[:-3] if label.endswith("...") else label
        for key in modes.KEYS:
            if bare == verbs.open_in(key) or bare == verbs.send_to(key):
                return True
            add_bare = verbs.add_to(key)
            if bare == add_bare or bare.startswith(add_bare + " "):
                return True
        return False

    ctx = FakeCtx(svc)
    for name, job in _rows(svc).items():
        for exit_ in asset_exits.exits_for(ctx, job):
            assert _looks_like_a_verb(exit_.label), (name, exit_.label)


def test_no_gate_touches_the_filesystem(svc, monkeypatch):
    """Both call sites ask ``exits_for`` every frame -- the same argument
    ``inker_open.can_edit_job``'s docstring makes about the toolbar -- so a
    gate that reached for the disk would be a stat per asset per frame.

    Every module a builder might lazily import is imported *before* the
    patch below, so this only catches a stat made by a gate itself and not
    one made by some unrelated module's own import machinery.
    """
    from warlock.studio import (  # noqa: F401
        clay_mode,
        create_stages,
        inker_mode,
        muse_mode,
        packwright_mode,
        plotter_mode,
        troupe_mode,
    )
    from warlock.studio.panes import pose_panel, troupe_send  # noqa: F401

    def _raise(self, *_a, **_k):
        raise AssertionError(f"a gate touched the filesystem: {self}")

    # Built before the patch lands: creating the rows goes through the real
    # store and the real config, and neither is what this test is about.
    rows = _rows(svc)
    ctx = FakeCtx(svc)
    monkeypatch.setattr(Path, "stat", _raise)
    for job in rows.values():
        asset_exits.exits_for(ctx, job)


def test_the_inspector_and_the_library_menu_cannot_drift_again(svc):
    """The claim the whole module exists to make: both surfaces draw the
    identical list ``exits_for`` returns. Asserted on the source of the two
    draw functions rather than by rendering them -- neither can be driven
    without imgui and a GL context -- each must call ``exits_for`` directly,
    and neither may still carry one of the three hand-written helpers this
    replaced."""
    edit_actions_src = inspect.getsource(inspector._edit_actions)
    overflow_src = inspect.getsource(library._overflow)
    assert "asset_exits.exits_for(ctx, job)" in edit_actions_src
    assert "asset_exits.exits_for(ctx, job)" in overflow_src
    for name in ("_send_to_troupe_item", "_troupe_item", "_map_and_atlas_items"):
        assert not hasattr(library, name), name
