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


class FakeCache:
    """The one method ``_mesh_for`` calls on ``ctx.cache`` -- a dict the test
    populates by hand, not ``svc.store`` itself: the real ``JobsCache.get``
    answers from the loaded *page*, already carrying ``files``
    (``attach_files``' doing), and a bare ``store.get`` row does not carry
    that column at all -- it would make a resolved mesh look fileless no
    matter what the test built. A dict of exactly the rows a test wants
    "loaded" is what actually stands in for the cache's contract.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def get(self, job_id: Any) -> Any:
        return self.rows.get(job_id)


class FakeCtx:
    def __init__(self, svc: Any, mode: str = "library", stage: str = "reference") -> None:
        self.svc = svc
        self.cache = FakeCache()
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


def _rig_followup(svc, source_id: str, *, status: str = "done") -> dict:
    """A rig job row, the shape ``asset_open``'s docstring names: minted with
    ``params["source_job"]``, writing into the *source*'s directory, so its
    own ``files`` is empty and its own ``stage`` is the ``model`` column
    default it never earned."""
    job = _job(svc, "rig", stage="model", status=status, params={"source_job": source_id})
    job["files"] = []
    return job


def _authored(svc, authored: str) -> dict:
    job = _job(svc, "text", stage="reference", status="done", params={"authored": authored})
    job["files"] = ["input.png"]
    return job


def _mason_row(svc, *, status: str = "done", ready: bool = True) -> dict:
    """What ``mason_mode.export_library`` mints: an ordinary ``model`` row --
    ``import_mesh``'s own -- carrying ``params["authored"] == "mason"`` and a
    ``scene.wscn`` beside it that nothing in ``job["files"]`` ever mentions."""
    job = _job(svc, "image", stage="model", status=status, params={"authored": "mason"})
    job["files"] = ["model.glb"] if ready else []
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


def test_a_mesh_can_be_added_to_a_scene_and_the_door_opens_the_mesh(svc, monkeypatch):
    """The outward half of Stage G's round trip, from the *library*: a mesh
    row's exits include somewhere to place it, and the door is handed the mesh
    rather than the selected row. Both call sites invoke ``exit_.open(ctx,
    job)`` with the row the user picked, so a rig row's Mason door that read
    its own ``job`` argument would place a reference to a job id with no
    ``model.glb`` behind it -- the exact trap ``_clay``'s and ``_poser``'s
    closures already name."""
    from warlock.studio import mason_mode

    mesh = _mesh(svc, rigged=True)
    rig_row = _rig_followup(svc, mesh["id"])
    ctx = FakeCtx(svc)
    ctx.cache.rows[mesh["id"]] = mesh

    placed: list = []
    monkeypatch.setattr(mason_mode, "add_asset_to_scene", lambda ctx, job: placed.append(job))
    mason = next(e for e in asset_exits.exits_for(ctx, rig_row) if e.mode == "mason")
    assert mason.label == verbs.add_to("mason", "as a scene item")
    assert not mason.reason
    mason.open(ctx, rig_row)
    assert placed and placed[0]["id"] == mesh["id"], "the door must place the source mesh"


def test_an_unfinished_mesh_dims_the_scene_door_with_the_row_s_own_reason(svc):
    """The near-miss rule: a mesh that is still reconstructing is one step
    from a scene, so the button is drawn with a reason rather than left off
    the list -- and the reason is the service's own sentence, not a second
    spelling of it."""
    from warlock.service.validation import not_done_message

    ctx = FakeCtx(svc)
    exits = asset_exits.exits_for(ctx, _mesh(svc, status="running"))
    mason = next(e for e in exits if e.mode == "mason")
    assert mason.reason == not_done_message("This", "running")


def test_a_mason_authored_row_offers_the_way_back_into_the_scene(svc, monkeypatch):
    """The inward half. ``params["authored"]`` is the whole gate -- no
    ``stat``, no service call -- because a reopen has no fallback: a merged
    scene GLB is not a lesser scene, and ``edit_asset_in_mason`` refuses to
    substitute it."""
    from warlock.studio import mason_mode

    row = _mason_row(svc)
    ctx = FakeCtx(svc)
    # Both Mason exits are live on this row and they are not the same door --
    # checked on the unfiltered list, since ``_labels`` collapses by mode and
    # would show only whichever of the two came last.
    mason_exits = [e for e in asset_exits.exits_for(ctx, row) if e.mode == "mason"]
    assert not any(e.reason for e in mason_exits)
    assert [e.label for e in mason_exits] == [
        verbs.open_in("mason"),
        verbs.add_to("mason", "as a scene item"),
    ]

    opened: list = []
    monkeypatch.setattr(mason_mode, "edit_asset_in_mason", lambda ctx, job: opened.append(job))
    mason_exits[0].open(ctx, row)
    assert opened and opened[0]["id"] == row["id"]


def test_an_ordinary_mesh_does_not_offer_to_reopen_a_scene_it_never_was(svc):
    """The marker is absent rather than empty on every other row, and the
    reopen door must be gated on that and nothing looser -- offering it for a
    row with no ``scene.wscn`` is a button that can only fail."""
    ctx = FakeCtx(svc)
    labels = [e.label for e in asset_exits.exits_for(ctx, _mesh(svc)) if e.mode == "mason"]
    assert labels == [verbs.add_to("mason", "as a scene item")]


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


def test_an_unrigged_mesh_offers_clay_mason_and_troupe_and_dims_poser(svc):
    """The pinned set gained ``mason`` in Stage G, deliberately: a mesh is a
    thing a *scene* is built out of, so every row that offers Clay, Poser and
    Troupe now offers somewhere to place it as well. The set is asserted
    exactly, not with ``<=``, because the whole reason this module exists is
    that two surfaces grew different lists -- an assertion that only checked
    for presence would let a sixth destination appear on one and not the
    other without saying so."""
    ctx = FakeCtx(svc)
    exits = asset_exits.exits_for(ctx, _rows(svc)["mesh"])
    by_mode = _labels(exits)
    assert set(by_mode) == {"clay", "mason", "poser", "troupe"}
    assert by_mode["clay"][1] is False
    assert by_mode["troupe"][1] is False
    assert by_mode["poser"][1] is True
    poser = next(e for e in exits if e.mode == "poser")
    assert "Rig" in poser.reason


def test_a_rigged_mesh_dims_nothing(svc):
    ctx = FakeCtx(svc)
    exits = asset_exits.exits_for(ctx, _rows(svc)["rigged_mesh"])
    by_mode = _labels(exits)
    assert set(by_mode) == {"clay", "mason", "poser", "troupe"}
    assert not any(dimmed for _label, dimmed in by_mode.values())


def test_a_rig_row_offers_its_mesh_destinations_and_poser_opens_the_source(svc, monkeypatch):
    """A rig row is a dead end on its own -- ``asset_open``'s own docstring:
    its artifacts land beside the mesh, never in its own directory -- so
    selecting it must offer exactly what its *mesh* reaches, and a press on
    Poser must open the mesh the rig belongs to, never the rig row itself
    (the door is invoked with the selected row, ``exit_.open(ctx, job)``, so
    a door that read its own ``job`` argument would get this wrong)."""
    from warlock.studio.panes import pose_panel

    mesh = _mesh(svc, rigged=True)
    rig_row = _rig_followup(svc, mesh["id"])
    ctx = FakeCtx(svc)
    ctx.cache.rows[mesh["id"]] = mesh
    exits = asset_exits.exits_for(ctx, rig_row)
    by_mode = _labels(exits)
    assert set(by_mode) == {"clay", "mason", "poser", "troupe"}
    assert not any(dimmed for _label, dimmed in by_mode.values())

    opened: list = []
    monkeypatch.setattr(pose_panel, "open_in_poser", lambda ctx, job: opened.append(job))
    poser = next(e for e in exits if e.mode == "poser")
    poser.open(ctx, rig_row)
    assert opened and opened[0]["id"] == mesh["id"], "the door must open the source mesh"


def test_a_rig_row_over_an_unrigged_mesh_dims_poser_with_the_mesh_reason(svc):
    """The near-miss reason has to be the *mesh*'s, not a generic one --
    ``asset_exits``'s own near-miss rule names why: a dimmed button without a
    reason is noise."""
    mesh = _mesh(svc, rigged=False)
    rig_row = _rig_followup(svc, mesh["id"])
    ctx = FakeCtx(svc)
    ctx.cache.rows[mesh["id"]] = mesh
    exits = asset_exits.exits_for(ctx, rig_row)
    by_mode = _labels(exits)
    assert set(by_mode) == {"clay", "mason", "poser", "troupe"}
    assert by_mode["clay"][1] is False
    assert by_mode["troupe"][1] is False
    assert by_mode["poser"][1] is True
    poser = next(e for e in exits if e.mode == "poser")
    assert "Rig" in poser.reason


def test_a_rig_row_whose_source_is_not_in_the_cache_offers_nothing_at_all(svc):
    """A source that has fallen off the loaded page -- or was never a real
    row -- is the honest floor ``asset_open.open_asset`` already takes for
    the same reason: a row this module cannot see is a row it offers nothing
    for, not a crash and not a guess."""
    rig_row = _rig_followup(svc, "000000000000")
    ctx = FakeCtx(svc)
    assert asset_exits.exits_for(ctx, rig_row) == []


def test_a_charsheet_offers_only_the_way_back_into_troupe(svc):
    """Regression: a charsheet row carries ``stage == "model"`` -- the column
    default every follow-up product wears, per ``asset_open``'s own docstring
    -- and it also carries ``params["source_job"]``, the same field a rig or a
    sheet carries. ``charsheet`` is deliberately not a key of
    ``asset_open.FOLLOWUP_STAGES`` (it opens in Troupe, not in Create), so
    ``_mesh_for`` must not hop for it even when its source mesh really is in
    the cache -- a version of that gate keyed on ``source_job`` alone did hop,
    resolving straight back to the mesh and offering Clay and Poser it has no
    files to back, plus a *second*, duplicate "Open in Troupe" beside
    ``_troupe_out``'s own. The mesh is loaded into the cache here on purpose:
    an empty cache only proves the fallback branch (no mesh found), not that
    the kind gate actually holds when a hop is otherwise possible."""
    ctx = FakeCtx(svc)
    mesh = _mesh(svc, rigged=True)
    ctx.cache.rows[mesh["id"]] = mesh
    exits = asset_exits.exits_for(ctx, _charsheet(svc, mesh["id"]))
    by_mode = _labels(exits)
    assert set(by_mode) == {"troupe"}
    assert by_mode["troupe"][1] is False
    assert by_mode["troupe"][0] == verbs.open_in("troupe")
    # ``_labels`` collapses by mode, which is exactly why a second "Open in
    # Troupe" from a resolved mesh's own ``_troupe_in`` would be invisible to
    # the assertions above -- checked separately, on the unfiltered list.
    troupe_exits = [e for e in exits if e.mode == "troupe"]
    assert len(troupe_exits) == 1, "a resolved mesh must not add a second Troupe door"


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


def test_asset_exits_status_reason_reuses_the_services_status_sentences(svc):
    """The 2026-09-11 audit, finding create-06.

    ``_status_reason`` used to be a second, independently hand-written mapping
    of the same four job-status values ``service.validation.STATUS_SENTENCES``
    already owns, worded differently ("Still queued." vs "is still waiting in
    the queue"). ``create_stages.available``'s own docstring states the rule
    this violated: "the wording is the service's own, verbatim, so the
    tooltip on the disabled segment and the toast from the refusal it is
    predicting are one sentence and not two paraphrases." Asserted through a
    live near-miss exit (an undone reference offering Plotter, dimmed)
    rather than by calling the private helper directly, so the check follows
    what a reader actually sees.
    """
    from warlock.service.validation import not_done_message

    ctx = FakeCtx(svc)
    for status in ("queued", "running", "error", "cancelled"):
        reference = _reference(svc, status=status, ready=False)
        exits = asset_exits.exits_for(ctx, reference)
        plotter = next(e for e in exits if e.mode == "plotter")
        assert plotter.reason == not_done_message("This", status), status


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
