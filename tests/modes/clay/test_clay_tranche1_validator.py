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


class _Ctx:
    def toast(self, message: str, level: str = "info") -> None:  # pragma: no cover - unused
        pass


def _tab() -> Any:
    from realmspinner.studio.modes.clay import state as clay_state

    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    return clay_state.ClayTab(doc=doc, title="Scene")


def test_check_readiness_caches_the_report_and_the_head_it_was_computed_at() -> None:
    tab = _tab()
    assert tab.readiness_report is None
    assert tab.readiness_head == -1

    clay_mode.check_readiness(_Ctx(), tab, "godot-desktop")

    assert tab.readiness_report is not None
    assert tab.readiness_report.profile == "godot-desktop"
    assert tab.readiness_head == tab.doc.history.head
    assert tab.readiness_profile == "godot-desktop"


def test_check_readiness_result_goes_stale_once_the_document_moves_on() -> None:
    tab = _tab()
    clay_mode.check_readiness(_Ctx(), tab, "unity")
    head_at_check = tab.readiness_head

    tab.doc.set_props(tab.doc.objects[0].uid, name="Renamed")

    assert tab.readiness_head == head_at_check, "the cached head does not update itself"
    assert tab.readiness_head != tab.doc.history.head, "the document has moved past it"


def test_check_readiness_runs_again_for_a_different_profile() -> None:
    tab = _tab()
    clay_mode.check_readiness(_Ctx(), tab, "godot-desktop")
    first = tab.readiness_report
    clay_mode.check_readiness(_Ctx(), tab, "webgl")
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
