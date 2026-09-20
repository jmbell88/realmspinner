"""The "Game check" pane's own pure half: ``bridge.validator_rows`` turns a
``readiness.Report`` into plain tuples with no imgui anywhere in the path, and
``clay_mode.check_readiness`` is the on-demand trigger that fills the tab's
cache.
"""

from __future__ import annotations

from typing import Any

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import readiness
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.ui.panes import bridge as clay_bridge


def test_validator_rows_maps_every_check_to_a_tuple() -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    report = readiness.validate(doc, readiness.DEFAULT_PROFILE)

    rows = clay_bridge.validator_rows(report)

    assert len(rows) == len(report.checks)
    for row, check in zip(rows, report.checks, strict=True):
        status, label, message, fix, uids = row
        assert status == check.status
        assert label == check.label
        assert message == check.message
        assert fix == check.fix
        assert uids == check.uids


def test_validator_rows_only_names_a_fix_where_the_check_actually_has_one() -> None:
    """The claim the "Fix" button's own gating rests on: a row with an empty
    ``fix`` must stay empty here too, or the pane would draw a button that
    runs nothing."""
    doc = bd.ClayDoc()  # empty document: every check but "objects" is skipped
    report = readiness.validate(doc, readiness.DEFAULT_PROFILE)

    rows = clay_bridge.validator_rows(report)
    objects_row = next(row for row in rows if row[1] == "Objects")
    assert objects_row[0] == "fail"
    assert objects_row[3] == "", "an empty document names no fix for its own emptiness"


def test_validator_rows_names_a_real_fix_op_for_every_fixable_check() -> None:
    """Every non-empty ``fix`` in ``readiness.FIX_OPS`` -- checked at the
    registry, not merely at the constant, so a name that stops being a real
    op is caught here rather than only when a user presses the button."""
    from realmspinner.studio.modes.clay import ops as clay_ops

    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    report = readiness.validate(doc, readiness.DEFAULT_PROFILE)

    for status, _label, _message, fix, _uids in clay_bridge.validator_rows(report):
        del status
        if fix:
            assert fix in readiness.FIX_OPS
            assert clay_ops.get(fix) is not None


# --- clay_mode.check_readiness -------------------------------------------


class _AppState:
    """The one field ``clay_mode.ensure`` reads/writes: ``ctx.state.clay``,
    ``None`` until the mode's own state is built on first use. See
    ``test_clay_mode.py``'s identical fixture -- ``on_task_done`` (which the
    round trip below now goes through, since the 2026-09-20 audit's clay-05
    moved ``check_readiness`` onto ``TaskRunner``) calls ``ensure(ctx)`` at
    its very first line, and a bare ``_Ctx`` with no ``.state`` at all raised
    ``AttributeError`` there before this fixture existed.
    """

    def __init__(self) -> None:
        self.clay = None


class _Ctx:
    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.result: Any = None
        self.tag: Any = None
        self.state = _AppState()

    def toast(self, message: str, level: str = "info") -> None:  # pragma: no cover - unused
        pass

    def submit(self, key: str, run: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        """Runs the task inline. The 2026-09-20 audit's clay-05 moved
        ``check_readiness`` off the frame thread and onto ``TaskRunner``
        (measured at up to 15.1 s on an ordinary document); these tests want
        the same result that call used to compute directly, so the submitted
        callable runs immediately rather than on a real background thread --
        see ``clay_mode.on_task_done``, which is what actually applies it
        onto the tab now, the same two-step shape ``test_clay_mode.py``'s own
        ``FakeCtx``/``_save`` already use for every other backgrounded call
        in this module.
        """
        self.submitted.append(key)
        self.tag = tag
        self.result = run(*args, **kwargs)
        return True


def _tab(ctx: _Ctx) -> Any:
    """A registered tab, not just a bare ``ClayTab``: ``on_task_done`` (which
    the round trip below now goes through) finds its tab by uid through
    ``ensure(ctx).get(...)`` -- ``ClayState.add``, the same registration
    ``clay_mode.adopt`` itself calls, not the settings/recents machinery
    ``adopt`` also does, which this ``_Ctx`` has no need to fake."""
    from realmspinner.studio.modes.clay import state as clay_state

    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    tab = clay_state.ClayTab(doc=doc, title="Scene")
    clay_mode.ensure(ctx).add(tab)
    return tab


class _Done:
    """The frame loop's own delivery shape -- see ``test_clay_mode.py``'s
    identical fake, used the same way: ``check_readiness`` only submits now
    (2026-09-20 audit, clay-05); applying the result onto the tab is
    ``clay_mode.on_task_done``'s job."""

    def __init__(self, key: str, result: Any = None, *, tag: Any = None) -> None:
        self.key = key
        self.result = result
        self.tag = tag


def _check_readiness(ctx: _Ctx, tab: Any, profile: str) -> None:
    """The whole round trip: submit, then the result landing through
    ``on_task_done``, exactly as the real ``TaskRunner`` cycle does."""
    clay_mode.check_readiness(ctx, tab, profile)
    clay_mode.on_task_done(ctx, _Done(f"clay-readiness:{tab.uid}", ctx.result, tag=ctx.tag))


def test_check_readiness_caches_the_report_and_the_head_it_was_computed_at() -> None:
    ctx = _Ctx()
    tab = _tab(ctx)
    assert tab.readiness_report is None
    assert tab.readiness_head == -1

    _check_readiness(ctx, tab, "godot-desktop")

    assert tab.readiness_report is not None
    assert tab.readiness_report.profile == "godot-desktop"
    assert tab.readiness_head == tab.doc.history.head
    assert tab.readiness_profile == "godot-desktop"


def test_check_readiness_result_goes_stale_once_the_document_moves_on() -> None:
    ctx = _Ctx()
    tab = _tab(ctx)
    _check_readiness(ctx, tab, "unity")
    head_at_check = tab.readiness_head

    tab.doc.set_props(tab.doc.objects[0].uid, name="Renamed")

    assert tab.readiness_head == head_at_check, "the cached head does not update itself"
    assert tab.readiness_head != tab.doc.history.head, "the document has moved past it"


def test_check_readiness_runs_again_for_a_different_profile() -> None:
    ctx = _Ctx()
    tab = _tab(ctx)
    # Both calls share one ctx/tab, the same registered tab both submissions
    # must find through ``ensure(ctx).get(...)`` when their results land.
    _check_readiness(ctx, tab, "godot-desktop")
    first = tab.readiness_report
    _check_readiness(ctx, tab, "webgl")
    second = tab.readiness_report

    assert first.profile == "godot-desktop"
    assert second.profile == "webgl"
    assert second is not first


def test_a_fresh_tab_has_a_readiness_profile_before_any_check_runs() -> None:
    """The Game check pane reads ``tab.readiness_profile`` on its first draw.
    It was declared on ``ClayState`` instead, and only ``check_readiness``
    ever set it on a tab, so drawing the pane before pressing Check raised
    ``AttributeError`` and took the whole Clay side panel down with it."""
    from realmspinner.studio.modes.clay import state as clay_state

    names = {f.name for f in __import__("dataclasses").fields(clay_state.ClayTab)}
    assert "readiness_profile" in names
