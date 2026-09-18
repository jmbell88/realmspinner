"""Three items of the 2026-09-07 Create friction sweep: the rail's ticks
(item 3.5), progress reachable from every stage (item 5.7), and a finished
mesh becoming the selection (item 5.8).

Pure where the claim is pure (``create_stages.ticked``, the mesh
auto-selection) and a source check where the claim is about which imgui call
a dispatch makes (the progress row) -- the same split ``test_create_stages.py``
and ``test_generation_workspace.py`` already draw between behaviour and
wiring.

Filename note: an earlier version of these tests lived in
``test_create_friction_sweep_2026_09_07.py``, which a parallel agent's own
file of the same name overwrote (untracked files do not merge). This name
cannot collide with theirs.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.studio import create_stages
from warlock.studio.state import AppState


def job(**kwargs):
    base = {
        "id": "0123456789ab",
        "kind": "text",
        "stage": "model",
        "status": "done",
        "files": ["model.glb", "input.png"],
        "prompt": "a barrel",
        "parent_id": None,
        "created_at": "2026-01-01T00:00:00",
    }
    return {**base, **kwargs}


class FakeCache:
    def __init__(self, jobs=()):
        self.jobs = list(jobs)
        self.by_id = {j["id"]: j for j in self.jobs}

    def get(self, job_id):
        return None if job_id is None else self.by_id.get(job_id)


class FakeCtx:
    """Only what ``create_stages.go`` and ``library.select`` touch."""

    def __init__(self, jobs=(), selected=None, mode=create_stages.MODE, stage="mesh"):
        self.state = AppState()
        self.state.selected = selected
        self.state.mode = mode
        self.state.create_stage = stage
        self.cache = FakeCache(jobs)

    def job(self, job_id=None):
        return self.cache.get(job_id or self.state.selected)


# --- item 3.5: the rail ticks what a finished prop actually has -------------


def rigged(**kwargs):
    return job(files=["model.glb", "input.png", "rig.glb"], **kwargs)


def test_a_finished_prop_ticks_export_without_a_rig_or_a_pose():
    """The verified bug: ``reached`` stops at the first stage a job has not
    got to, so Rig failing (no rig.glb, no rig.json) stranded Export behind
    it forever on every unrigged prop -- ``create_stages.reached`` -> "mesh"
    for exactly this job, which is what the unfixed ``ticked`` would inherit
    if it only forwarded to ``reached``."""
    done_model = job()
    assert create_stages.reached(done_model) == "mesh"  # the bug, pinned

    ticks = create_stages.ticked(done_model)
    assert ticks == frozenset({"reference", "mesh", "export"})
    assert "rig" not in ticks
    assert "pose" not in ticks


def test_a_bare_reference_still_ticks_nothing_past_reference():
    """The guard ``_reached_export``'s docstring exists to keep: Mesh is
    required, not optional, so failing it still ends the walk before Export
    is ever asked about."""
    reference = job(stage="reference", files=["input.png"])
    assert create_stages.ticked(reference) == frozenset({"reference"})


def test_ticked_agrees_with_reached_once_rig_and_pose_are_both_earned():
    """Nothing changes for an asset that actually has both -- the two
    functions only diverge on a stage a prop may legitimately skip."""
    fully = rigged()
    assert create_stages.reached(fully, poses=[{"name": "idle"}]) == "export"
    assert create_stages.ticked(fully, poses=[{"name": "idle"}]) == frozenset(
        {"reference", "mesh", "rig", "pose", "export"}
    )


def test_reached_stays_monotone_and_distinct_from_ticked():
    """``reached`` was not deleted: it is still the "furthest in strict
    pipeline order" answer and the rail no longer asks it. This pins that it
    still disagrees with ``ticked`` on exactly the case 3.5 exists for."""
    done_model = job()
    assert create_stages.reached(done_model) != create_stages.ticked(done_model)


def test_optional_hints_name_only_the_optional_stages():
    assert set(create_stages.OPTIONAL_HINTS) == create_stages.OPTIONAL_STAGES
    for text in create_stages.OPTIONAL_HINTS.values():
        # Latin-1 only (CLAUDE.md): no arrow, ellipsis or dash wider than a
        # hyphen may reach imgui's default atlas.
        text.encode("latin-1")


def test_the_stage_rail_ticks_a_set_not_a_single_furthest_key():
    """``create_rail.stage_rail`` takes ``done`` as a container of keys now,
    and a finished, unrigged prop's set has three members with a gap in the
    pipeline order -- which a single "furthest reached" key cannot express at
    all. A regression here is a signature the rail cannot be fed the right
    answer through, whatever ``create_stages`` computes."""
    import inspect

    from warlock.studio import create_rail

    signature = inspect.signature(create_rail.stage_rail)
    assert "optional" in signature.parameters
    source = inspect.getsource(create_rail.stage_rail)
    assert "done_index" not in source, "the old furthest-key comparison is still there"
    assert "key in done_keys" in source


# --- item 5.7: progress is reachable from every stage ------------------------


def test_the_rig_stage_shows_the_progress_of_a_job_it_started(monkeypatch):
    """The verified bug: the tray's "Working now" row was only ever drawn
    from ``shell.frame.FrameMixin._viewport_pane``'s Reference-stage tray, so
    a remesh or a rig bake started from the Rig stage showed nothing here but
    the floating card. ``_stage_pane`` now draws the same row before it
    dispatches to any stage's own panel."""
    from warlock.studio import generation_workspace
    from warlock.studio.panes import stage_rig
    from warlock.studio.shell import frame

    calls: list[object] = []
    monkeypatch.setattr(
        generation_workspace, "progress_row", lambda ctx: calls.append(ctx) or False
    )
    monkeypatch.setattr(stage_rig, "draw", lambda ctx: None)

    ctx = SimpleNamespace(state=SimpleNamespace(create_stage="rig"))
    frame._stage_pane(ctx)

    assert calls == [ctx], "the Rig stage never asked the tray for its progress row"


def test_the_reference_stage_does_not_draw_the_tray_progress_row_twice(monkeypatch):
    """Reference already carries the canvas tray and the floating card; a
    third copy from ``_stage_pane`` would put the count back up to three
    instead of trading one restatement for reach on every other stage."""
    from warlock.studio import generation_workspace
    from warlock.studio.panes import settings_2d
    from warlock.studio.shell import frame

    calls: list[object] = []
    monkeypatch.setattr(
        generation_workspace, "progress_row", lambda ctx: calls.append(ctx) or False
    )
    monkeypatch.setattr(settings_2d, "draw", lambda ctx: None)

    ctx = SimpleNamespace(state=SimpleNamespace(create_stage="reference"))
    frame._stage_pane(ctx)

    assert calls == []


def test_the_canvas_tray_no_longer_draws_its_own_working_now_row():
    """The restatement dropped from the Reference stage (2026-09-07 Create
    review, item 5.7): the tray used to call ``_progress`` on the active job
    directly from ``draw``. That call moved out to ``progress_row``, which
    ``shell.frame.FrameMixin._stage_pane`` now calls instead -- the pick was
    forced rather than chosen, since the other two restatements (the settings
    column's "Queue: ..." line and the floating card) live in panes this
    change does not own.
    """
    import inspect

    from warlock.studio import generation_workspace as gw

    draw_source = inspect.getsource(gw.draw)
    assert "_progress(ctx, active)" not in draw_source
    assert "_progress(ctx, active)" in inspect.getsource(gw.progress_row)


# --- item 5.8: a finished mesh becomes the selection, narrowly --------------


def test_a_finished_mesh_becomes_the_selection_when_the_user_waited_on_the_mesh_stage():
    ref = job(id="aaaaaaaaaaaa", stage="reference", files=["input.png"])
    mesh = job(id="bbbbbbbbbbbb", parent_id="aaaaaaaaaaaa")
    ctx = FakeCtx([ref, mesh], selected="aaaaaaaaaaaa", stage="mesh")
    app = SimpleNamespace(app_ctx=ctx)

    from warlock.studio.main import App

    App._select_finished_mesh_if_waiting(app, mesh)

    assert ctx.state.selected == "bbbbbbbbbbbb"
    assert ctx.state.create_stage == "mesh"


def test_it_does_not_move_a_selection_the_user_changed():
    """All three conditions matter: a user who selected something else while
    the mesh was building keeps what they picked."""
    ref = job(id="aaaaaaaaaaaa", stage="reference", files=["input.png"])
    mesh = job(id="bbbbbbbbbbbb", parent_id="aaaaaaaaaaaa")
    other = job(id="cccccccccccc", stage="reference", files=["input.png"])
    ctx = FakeCtx([ref, mesh, other], selected="cccccccccccc", stage="mesh")
    app = SimpleNamespace(app_ctx=ctx)

    from warlock.studio.main import App

    App._select_finished_mesh_if_waiting(app, mesh)

    assert ctx.state.selected == "cccccccccccc"


def test_it_does_not_move_a_selection_from_another_stage():
    """A user who navigated away from Mesh -- to Rig, to Reference, anywhere
    -- does not have their selection stolen out from under them."""
    ref = job(id="aaaaaaaaaaaa", stage="reference", files=["input.png"])
    mesh = job(id="bbbbbbbbbbbb", parent_id="aaaaaaaaaaaa")
    ctx = FakeCtx([ref, mesh], selected="aaaaaaaaaaaa", stage="reference")
    app = SimpleNamespace(app_ctx=ctx)

    from warlock.studio.main import App

    App._select_finished_mesh_if_waiting(app, mesh)

    assert ctx.state.selected == "aaaaaaaaaaaa"


def test_it_does_not_move_the_selection_for_an_unrelated_meshs_landing():
    """The landed job has to be *this* selection's child, not merely some
    mesh that happened to finish while the user stood on the Mesh stage."""
    ref = job(id="aaaaaaaaaaaa", stage="reference", files=["input.png"])
    other_ref = job(id="dddddddddddd", stage="reference", files=["input.png"])
    unrelated_mesh = job(id="eeeeeeeeeeee", parent_id="dddddddddddd")
    ctx = FakeCtx([ref, other_ref, unrelated_mesh], selected="aaaaaaaaaaaa", stage="mesh")
    app = SimpleNamespace(app_ctx=ctx)

    from warlock.studio.main import App

    App._select_finished_mesh_if_waiting(app, unrelated_mesh)

    assert ctx.state.selected == "aaaaaaaaaaaa"
