"""Clay's tool rail centres its cells, and the header row is grouped by utility.

Two reports from 2026-10-09: the left rail's buttons sat against the rail's left
edge instead of in the middle of it, and the row above the viewport ran its
twelve controls together with nothing to say which belong to which.

* **The rail.** A child window with no border ignores ``window_padding`` unless it
  is given ``always_use_window_padding``, so the 46 px cells sat flush left in a
  56 px column with the whole 10 px of slack on the right.
* **The row.** ``toolbar.Item.group`` / ``Field.group`` name what a control is for
  and the row draws a thin rule where two drawn neighbours differ; the rule is part
  of the width arithmetic, so a grouped row that "fits" fits.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from _ui_context import imgui_context
from _ui_driver import UiDriver

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import toolbar
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.ui.panes import header as clay_header
from realmspinner.studio.modes.clay.ui.panes import rail as clay_rail


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.settings = _Settings()

    def toast(self, message: str, level: str = "info") -> None:
        pass

    def busy(self, key: str) -> bool:
        return False


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(clay_mode, "recent_paths", lambda ctx: [])


def _open(boxes: int = 1):
    ctx = _Ctx()
    doc = bd.ClayDoc()
    for index in range(boxes):
        doc.add_object(bd.Obj(uid=bd.new_uid(), name=f"B{index}", mesh=bp.box()))
    tab = clay_mode.adopt(ctx, doc, title="Scene")
    return ctx, tab


# --- the rail ----------------------------------------------------------------------


def _rail_slack(ui, monkeypatch) -> list[tuple[float, float]]:
    """``(left, right)`` gap in px between each rail cell and its child window's edges."""
    ctx, _tab = _open()
    seen: list[tuple[float, float]] = []
    real = clay_rail._cell

    def spy(glyph, ident, word, **kw):
        clicked = real(glyph, ident, word, **kw)
        low, high = ui.get_item_rect_min(), ui.get_item_rect_max()
        pos, size = ui.get_window_pos(), ui.get_window_size()
        seen.append((low.x - pos.x, (pos.x + size.x) - high.x))
        return clicked

    monkeypatch.setattr(clay_rail, "_cell", spy)
    ui.new_frame()
    ui.set_next_window_pos((0.0, 0.0))
    ui.set_next_window_size((400.0, 1400.0))
    ui.begin("##host", None, ui.WindowFlags_.no_decoration.value)
    clay_rail.draw(ctx, 1200.0)
    ui.end()
    ui.end_frame()
    return seen


def test_the_rail_cells_are_centred_in_the_rail_column(monkeypatch, ui):
    seen = _rail_slack(ui, monkeypatch)
    assert len(seen) >= 5, "the four tools and the +, at least"
    for left, right in seen:
        assert left > 1.0, "a cell flush against the rail's left edge is not centred"
        assert abs(left - right) <= 1.0, (left, right)


def test_the_rail_child_asks_for_its_window_padding(monkeypatch, ui):
    """The cause, pinned: a borderless child without the flag drops the padding."""
    flags: list[int] = []
    real = ui.begin_child

    def spy(str_id, size=None, child_flags=0, window_flags=0):
        if str_id == "##clay-rail":
            flags.append(int(child_flags))
        return real(str_id, size, child_flags, window_flags)

    monkeypatch.setattr(ui, "begin_child", spy)
    _rail_slack(ui, monkeypatch)
    assert flags and flags[0] & ui.ChildFlags_.always_use_window_padding.value


# --- the grouping arithmetic (pure) -------------------------------------------------


def test_a_rule_sits_between_drawn_neighbours_of_different_named_groups_only():
    groups = ["a", "a", "b", "", "b", "c"]
    assert toolbar.separators(groups, range(6)) == 2, "a|b and b|c; the unnamed entry has none"
    assert toolbar.separators(groups, [0, 1]) == 0
    assert toolbar.separators(groups, [1, 2]) == 1
    assert toolbar.separators(None, range(3)) == 0
    assert toolbar.separators([], range(3)) == 0


def test_a_group_that_moves_into_the_menu_takes_its_rule_with_it():
    groups = ["a", "b", "c"]
    assert toolbar.separators(groups, [0, 1, 2]) == 2
    assert toolbar.separators(groups, [0, 2]) == 1, "b is in the menu: one rule, not two"


def test_the_planner_counts_the_rules_so_a_grouped_row_that_fits_on_paper_fits_on_screen():
    full = [100.0, 100.0, 100.0]
    icon = [20.0, 20.0, 20.0]
    prio = [0, 1, 2]
    pinned = [False, False, False]
    # Exactly the width of the three entries and two gaps: fits ungrouped...
    avail = 300.0 + 2 * 4.0
    assert toolbar.plan(full, icon, prio, pinned, avail, 20.0, gap=4.0) == [toolbar.FULL] * 3
    # ...and does not once two rules, each worth 5, are in it.
    tiers = toolbar.plan(
        full, icon, prio, pinned, avail, 20.0, gap=4.0, groups=["a", "b", "c"], extra=5.0
    )
    assert tiers != [toolbar.FULL] * 3


def test_an_ungrouped_row_plans_exactly_as_before():
    full = [80.0, 90.0, 70.0, 60.0]
    icon = [20.0, 20.0, 20.0, 20.0]
    prio = [0, 1, 1, 2]
    pinned = [False, False, False, False]
    for avail in (400.0, 300.0, 200.0, 120.0, 60.0):
        assert toolbar.plan(full, icon, prio, pinned, avail, 20.0, gap=4.0) == toolbar.plan(
            full, icon, prio, pinned, avail, 20.0, gap=4.0, groups=["", "", "", ""], extra=5.0
        )


# --- Clay's row ---------------------------------------------------------------------


def test_the_clay_row_is_four_groups_in_the_order_they_are_read(ui):
    ctx, tab = _open()
    state = clay_mode.ensure(ctx)
    ui.new_frame()
    items, fields, _trailing = clay_header._row(ctx, state, tab, None)
    ui.end_frame()
    ordered = [(entry.key, entry.group) for entry in [*items, *fields]]
    assert ordered == [
        ("undo", "history"),
        ("redo", "history"),
        ("history", "history"),
        ("mode", "mode"),
        ("shading", "display"),
        ("xray", "display"),
        ("view", "aids"),
        ("snap", "aids"),
    ]
    assert toolbar.separators([group for _key, group in ordered], range(len(ordered))) == 3


def test_the_clay_row_draws_one_rule_between_each_pair_of_groups(monkeypatch, ui):
    ctx, _tab = _open()
    rules: list[int] = []
    real = toolbar._group_rule

    def spy() -> None:
        rules.append(1)
        real()

    monkeypatch.setattr(toolbar, "_group_rule", spy)

    def draw() -> None:
        ui.new_frame()
        ui.set_next_window_pos((0.0, 0.0))
        ui.set_next_window_size((1500.0, 400.0))
        ui.begin("##host", None, ui.WindowFlags_.no_decoration.value)
        clay_header.draw(ctx, None)
        ui.end()
        ui.end_frame()

    driver = UiDriver(ui, draw)
    driver.settle(3)
    rules.clear()
    driver.settle(1)
    assert len(rules) == 3, "history | mode | display | aids"


def test_the_clay_row_still_fits_at_the_window_floor_with_its_rules_counted(ui):
    ctx, tab = _open()
    state = clay_mode.ensure(ctx)
    gap = ui.get_style().item_spacing.x
    overflow = ui.get_frame_height()
    for avail in (1100.0, 760.0, 640.0, 520.0):
        ui.new_frame()
        plan = clay_header.row_plan(ctx, state, tab, avail)
        ui.end_frame()
        assert plan.extra > 0.0
        pinned_only = all(
            tier == toolbar.MENU or entry.pinned
            for entry, tier in zip(plan.entries, plan.tiers, strict=True)
        )
        assert plan.used(gap, overflow) <= avail or pinned_only, (avail, plan.tiers)
