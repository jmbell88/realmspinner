"""Compare: never itself, never inline, never fatal.

UX-04. Three defects in one small feature, and each is reachable by ordinary
use:

* Right-clicking a card *selects* it before the menu is drawn -- deliberately,
  because a menu acting on a card other than the marked one is how the wrong
  asset gets deleted. But "Compare with selected" then read that same selection
  when the item was clicked, so it compared the target with itself.
* ``Viewer.compare`` parsed the GLB and uploaded GPU resources synchronously on
  the frame thread. ``_sync_viewer`` grew its ``pending``/task split precisely
  because inline parsing froze the frame; compare was not among the blocking
  survivors that are argued for by name.
* And it had no error boundary, so a malformed GLB propagated out of the frame
  loop and exited the app -- for a feature whose entire job is to *look* at
  something.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from warlock.studio.panes import library


class _Viewer:
    def __init__(self) -> None:
        self.parsed: list[Path] = []
        self.adopted: list[Any] = []
        self.exited = 0

    @staticmethod
    def parse_model(path: Path) -> Any:
        return ("model", path)

    def adopt_compare(self, model: Any) -> None:
        self.adopted.append(model)

    def exit_compare(self) -> None:
        self.exited += 1


class _Ctx:
    def __init__(self, tmp_path: Path) -> None:
        self.viewer = _Viewer()
        self.state = SimpleNamespace(
            comparing=None, selected=None, compare_baseline=None, compare_pending=None
        )
        self.submitted: list[tuple[str, Any]] = []
        self.toasts: list[tuple[str, str]] = []
        self._root = tmp_path
        self.accept = True

    def job_dir(self, job_id: str) -> Path:
        return self._root / job_id

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kw: Any) -> bool:
        if not self.accept:
            return False
        self.submitted.append((key, tag))
        return True

    def toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))


def test_comparing_an_asset_with_itself_is_refused(tmp_path):
    ctx = _Ctx(tmp_path)
    # What the right-click gesture leaves behind: the baseline captured when
    # the menu opened, and the selection already moved onto the target.
    ctx.state.compare_baseline = "aaaaaaaaaaaa"
    ctx.state.selected = "aaaaaaaaaaaa"

    library.compare(ctx, "aaaaaaaaaaaa")

    assert ctx.submitted == [], "nothing should be parsed for a self-comparison"
    assert ctx.state.comparing is None
    assert ctx.toasts and "different asset" in ctx.toasts[0][0]


def test_a_comparison_parses_off_the_frame_thread(tmp_path):
    ctx = _Ctx(tmp_path)
    ctx.state.compare_baseline = "aaaaaaaaaaaa"
    ctx.state.selected = "bbbbbbbbbbbb"

    library.compare(ctx, "bbbbbbbbbbbb")

    assert ctx.state.comparing == "bbbbbbbbbbbb"
    assert ctx.submitted == [
        (library.COMPARE_KEY, tmp_path / "bbbbbbbbbbbb" / "model.glb")
    ]
    # Nothing has been uploaded yet -- that is the frame thread's half, and it
    # happens when the parse lands.
    assert ctx.viewer.adopted == []
    assert ctx.state.compare_pending == tmp_path / "bbbbbbbbbbbb" / "model.glb"


def test_a_second_comparison_while_one_is_parsing_is_dropped_not_queued(tmp_path):
    """``_sync_viewer``'s rule: a refused submit means a load is already in
    flight, and its result is checked against ``pending`` before adoption."""
    ctx = _Ctx(tmp_path)
    ctx.state.compare_baseline = "aaaaaaaaaaaa"
    ctx.accept = False

    library.compare(ctx, "bbbbbbbbbbbb")

    assert ctx.state.comparing is None
    assert ctx.state.compare_pending is None


def test_toggling_the_same_comparison_off_still_works(tmp_path):
    ctx = _Ctx(tmp_path)
    ctx.state.comparing = "bbbbbbbbbbbb"

    library.compare(ctx, "bbbbbbbbbbbb")

    assert ctx.state.comparing is None
    assert ctx.viewer.exited == 1
    assert ctx.submitted == []


def test_the_menu_item_says_which_way_round_the_comparison_goes():
    """"Compare with selected" was ambiguous *and* wrong -- by the time it ran,
    "selected" was the card it was on."""
    source = Path(library.__file__).read_text(encoding="utf-8")
    assert 'menu_item("Compare selected with this"' in source
    # The old label survives only inside the comment explaining why it went.
    assert 'menu_item("Compare with selected"' not in source


def test_the_baseline_is_captured_when_the_menu_opens():
    """Not read when the item is clicked: the right-click that opens the menu
    has already moved the selection onto the target.

    Captured *before* ``select`` runs and assigned *after* it -- not the other
    way round, since shell-02 (the 2026-09-08 audit) made ``AppState.select``
    itself clear ``compare_baseline`` on an ordinary selection change, and
    setting it ahead of the call would have this line undo its own
    assignment.
    """
    source = Path(library.__file__).read_text(encoding="utf-8")
    opened = source.index("imgui.open_popup(\"more\")")
    before = source[:opened]
    assert "previous = ctx.state.selected" in before
    assert "ctx.state.compare_baseline = previous" in before
    select_call = before.rindex("select(ctx, job[\"id\"])")
    previous_capture = before.index("previous = ctx.state.selected")
    baseline_assign = before.rindex("ctx.state.compare_baseline = previous")
    assert previous_capture < select_call < baseline_assign < opened


@pytest.mark.parametrize("attr", ["compare_baseline", "compare_pending"])
def test_the_state_fields_exist_on_the_real_app_state(attr):
    from warlock.studio.state import AppState

    assert hasattr(AppState(), attr)


def test_a_selection_change_drops_a_comparison_parse_in_flight():
    """A result landing after the selection moved would be adopted into a
    comparison that no longer exists."""
    from warlock.studio.state import AppState

    state = AppState()
    state.select("aaaaaaaaaaaa")
    state.comparing = "bbbbbbbbbbbb"
    state.compare_pending = Path("somewhere/model.glb")
    state.select("cccccccccccc")
    assert state.comparing is None
    assert state.compare_pending is None


def test_a_refused_second_compare_leaves_the_first_one_alone(tmp_path):
    """``COMPARE_KEY`` is a single key, so a second compare started before the
    first parse lands is refused -- and only the second one should be lost.

    ``comparing`` and ``compare_pending`` *are* the record of the compare in
    flight, and ``compare()`` overwrites them before it asks the runner. On
    refusal it used to clear both to None, which cancelled the earlier compare
    too: ``_adopt_compare`` checks the arriving tag against ``compare_pending``
    and dropped the result that was already on its way. Neither mesh appeared
    and nothing was said, so the menu item read as inert and the user clicked
    it again.
    """
    ctx = _Ctx(tmp_path)
    ctx.state.compare_baseline = "aaaaaaaaaaaa"
    ctx.state.selected = "aaaaaaaaaaaa"
    library.compare(ctx, "bbbbbbbbbbbb")
    first = ctx.state.compare_pending
    assert first is not None

    ctx.accept = False  # the runner refuses: the first parse is still running
    library.compare(ctx, "cccccccccccc")

    assert ctx.state.comparing == "bbbbbbbbbbbb", "the first compare was cancelled"
    assert ctx.state.compare_pending == first
    assert len(ctx.submitted) == 1


def test_compare_from_the_ellipsis_menu_does_not_reuse_a_stale_right_click_baseline(tmp_path):
    """shell-02 (the 2026-09-08 audit): ``compare_baseline`` was written only
    by the card's right-click handler and never cleared on an ordinary
    selection change, or when the same overflow menu is instead opened via
    the small ellipsis button -- so "Compare selected with this" clicked from
    a menu opened that way could silently reuse whichever job was
    right-clicked earliest in the session instead of the asset actually
    selected.

    Reproduced here without any right-click at all: a stale baseline from
    some earlier gesture, then an ordinary ``state.select`` (a plain card
    click, or the selection already in place when the ellipsis menu is
    opened) -- which must retire it. If it does not, comparing the now
    -selected card with itself is not refused.
    """
    from warlock.studio.state import AppState

    ctx = _Ctx(tmp_path)
    ctx.state = AppState()
    # An earlier right-click, on a different card entirely, left this behind.
    ctx.state.compare_baseline = "zzzzzzzzzzzz"
    # The user then selects another card the ordinary way -- no right-click,
    # so nothing here ever touches ``compare_baseline`` except ``select``
    # itself.
    ctx.state.select("aaaaaaaaaaaa")

    # Opens that same card's overflow via the ellipsis and clicks "Compare
    # selected with this" -- comparing the selected asset with itself, which
    # must be refused.
    library.compare(ctx, "aaaaaaaaaaaa")

    assert ctx.submitted == [], "the stale baseline let a self-comparison through"
    assert ctx.toasts and "different asset" in ctx.toasts[0][0]


def test_a_refused_first_compare_leaves_nothing_behind(tmp_path):
    """With nothing in flight the fields were already None, so restoring them
    is the same as clearing them -- the viewer must not be left claiming a
    comparison that was never parsed."""
    ctx = _Ctx(tmp_path)
    ctx.state.compare_baseline = "aaaaaaaaaaaa"
    ctx.state.selected = "aaaaaaaaaaaa"
    ctx.accept = False

    library.compare(ctx, "bbbbbbbbbbbb")

    assert ctx.state.comparing is None
    assert ctx.state.compare_pending is None
