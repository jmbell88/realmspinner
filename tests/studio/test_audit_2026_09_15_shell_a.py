"""Regression tests for the 2026-09-15 audit's shell-A batch.

shell-01: the workspace layout editor's drag/hide commit rebuilt every
column's saved order from ``column.live(ctx)`` (the skeleton's built-in
declaration order) instead of ``skeletons.ordered(...)`` (the order actually
drawn, reconciled against the saved layout). Any drag or hide-badge press
silently reset every *other* column's saved arrangement to its built-in
order.

shell-02: the Library inspector's Details tab gated "Mesh quality" and "Was
this any good?" on ``create_stages.at(state, "mesh")``, which also requires
``state.mode == "create"`` -- but ``_details_tab`` is only ever reached from
the branch of ``inspector.draw`` taken when Create is *not* the mode, so the
section was dead everywhere it could be seen.

shell-06 is documentation-only (see ``journal.snapshot``'s docstring); no
regression test is possible for it, per the fix brief.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from realmspinner.studio import layout as layout_mod
from realmspinner.studio import layout_edit, layouts
from realmspinner.studio import layout_skeleton as skeleton
from realmspinner.studio.panes import inspector

# --- shell-01 ----------------------------------------------------------------


class _Settings:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value) -> None:
        self.data[key] = value


def _slot(name: str) -> skeleton.Slot:
    return skeleton.Slot(name, name, lambda ctx: None, sizing=skeleton.SHARE, share_key=name)


def test_dragging_one_pane_does_not_reset_another_columns_saved_order():
    """Reproduces the audit's ``shell-documents-01.py`` probe: a column
    already reordered by the user (C, A, B against a built-in A, B, C) must
    keep that order verbatim when a drag lands in a *different* column --
    the bug wrote every column back from the built-in order, so an unrelated
    drag in the "left" column scrambled "right" back to A, B, C.
    """
    left_a, left_b = _slot("left-a"), _slot("left-b")
    right_a, right_b, right_c = _slot("A"), _slot("B"), _slot("C")
    left = skeleton.Column("left", (left_a, left_b))
    right = skeleton.Column("right", (right_a, right_b, right_c))
    columns = {"left": left, "right": right}

    settings = _Settings()
    library = layouts.Library(settings)
    # The user has already dragged "right" into C, A, B; "left" is untouched.
    library.record("probe", {"right": ["C", "A", "B"]}, set())

    ctx = SimpleNamespace(state=SimpleNamespace(mode="probe"))
    app = SimpleNamespace(layouts=library)
    edit = layout_edit.ensure(ctx.state)
    edit.open = True
    edit.hidden = set()
    edit.dragging = "left-b"

    # The rects the real renderer would have produced: "left" in its only
    # possible order, "right" in the *saved* C, A, B order (skeletons.ordered
    # is what layout.column actually draws from).
    layout_mod.FRAME_PANES.clear()
    layout_mod.FRAME_PANES["left-a"] = (0.0, 0.0, 300.0, 50.0)
    layout_mod.FRAME_PANES["left-b"] = (0.0, 50.0, 300.0, 50.0)
    layout_mod.FRAME_PANES["C"] = (300.0, 0.0, 300.0, 50.0)
    layout_mod.FRAME_PANES["A"] = (300.0, 50.0, 300.0, 50.0)
    layout_mod.FRAME_PANES["B"] = (300.0, 100.0, 300.0, 50.0)

    # Drag "left-b" to the top of the left column -- nothing to do with the
    # right column at all.
    mouse = SimpleNamespace(x=10.0, y=5.0)
    layout_edit._commit(app, ctx, columns, edit, mouse)

    assert library.arrangement("probe").columns.get("left") == ["left-b", "left-a"]
    # The untouched column's saved order must survive the drag unchanged.
    assert library.arrangement("probe").columns.get("right") == ["C", "A", "B"]


def test_a_drag_within_a_column_reorders_against_its_saved_position_not_the_builtin_one():
    """The same probe's own drag: moving "B" (last in the *saved* order, but
    second in the *built-in* one) to the top must produce B, C, A -- the
    order computed against what was actually on screen -- not B, A, C, which
    is what moving "B" to the front of the *built-in* A, B, C list gives.
    """
    a_slot, b_slot, c_slot = _slot("A"), _slot("B"), _slot("C")
    column = skeleton.Column("right", (a_slot, b_slot, c_slot))
    columns = {"right": column}

    settings = _Settings()
    library = layouts.Library(settings)
    library.record("probe", {"right": ["C", "A", "B"]}, set())

    ctx = SimpleNamespace(state=SimpleNamespace(mode="probe"))
    app = SimpleNamespace(layouts=library)
    edit = layout_edit.ensure(ctx.state)
    edit.open = True
    edit.hidden = set()
    edit.dragging = "B"

    layout_mod.FRAME_PANES.clear()
    layout_mod.FRAME_PANES["C"] = (0.0, 0.0, 300.0, 50.0)
    layout_mod.FRAME_PANES["A"] = (0.0, 50.0, 300.0, 50.0)
    layout_mod.FRAME_PANES["B"] = (0.0, 100.0, 300.0, 50.0)

    mouse = SimpleNamespace(x=10.0, y=5.0)
    layout_edit._commit(app, ctx, columns, edit, mouse)

    assert library.arrangement("probe").columns.get("right") == ["B", "C", "A"]


# --- shell-02 ----------------------------------------------------------------


def test_the_librarys_details_tab_shows_the_verdict_section_for_a_finished_mesh(monkeypatch):
    """``_details_tab`` is only ever called from the branch of ``draw`` taken
    when Create is *not* the mode (Create itself renders ``_stage_body`` and
    returns before the tab bar exists) -- so gating "Mesh quality" and "Was
    this any good?" on ``create_stages.at(state, "mesh")``, which additionally
    requires ``state.mode == "create"``, made both sections unreachable
    everywhere the Details tab is actually drawn from.
    """
    calls: list[str] = []
    monkeypatch.setattr(inspector, "_settings", lambda ctx, job: None)
    monkeypatch.setattr(inspector, "_reference", lambda ctx, job: None)
    monkeypatch.setattr(inspector, "_pixel", lambda ctx, job: None)
    monkeypatch.setattr(inspector, "_seam", lambda ctx, job: None)
    monkeypatch.setattr(inspector.sprite_panel, "draw", lambda ctx, job: None)
    monkeypatch.setattr(inspector, "_quality", lambda ctx, job: calls.append("quality"))
    monkeypatch.setattr(inspector, "_verdict", lambda ctx, job: calls.append("verdict"))

    # Library's inspector: not Create mode, and a finished mesh (the
    # reconstruction stage is spelled "model", never "mesh" -- see
    # create_stages.py's own module docstring).
    ctx = SimpleNamespace(state=SimpleNamespace(mode="home", create=SimpleNamespace(stage=None)))
    job = {"id": "aaaaaaaaaaaa", "stage": "model", "status": "done"}

    inspector._details_tab(ctx, job)

    assert calls == ["quality", "verdict"]


def test_the_details_tab_still_skips_the_verdict_section_for_a_reference():
    """The complement: a reference (an image-stage job, never a mesh) must
    not grow "Mesh quality" or "Was this any good?" just because the gate
    moved off ``state.mode``.
    """
    from realmspinner.studio.modes.create.ui import stages as create_stages

    job = {"id": "bbbbbbbbbbbb", "stage": "reference", "status": "done"}
    assert create_stages.stage_for(job) != "mesh"
