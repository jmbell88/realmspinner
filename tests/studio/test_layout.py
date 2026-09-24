"""The three-column skeleton's measurements.

Two sidebars that the user could drag became two fixed ones, and the tests here
are about the *leftover*: a settings file written by the version that stored
widths is still on every machine that has ever run Realmspinner, and it must not
resurrect a width nothing reads or leave one behind for a future reader to find
and half-honour.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import pytest

from realmspinner.studio import layout as layout_mod


class _Settings:
    def __init__(self, stored: Any = None) -> None:
        self.store: dict[str, Any] = {"layout": stored} if stored is not None else {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


def test_the_sidebars_are_one_of_three_named_sizes():
    """Named sizes rather than a drag (M106): a form has a width that reads
    well, and what the old free drag bought was a way to make the app look
    broken. ``SIDEBAR_W`` is the one in force -- module state, exactly as
    ``tokens.SCALE`` is, because eight call sites read it directly."""
    assert layout_mod.SIDEBAR_WIDTHS["default"] == 300.0
    assert layout_mod.SIDEBAR_W in layout_mod.SIDEBAR_WIDTHS.values()


def test_a_pane_is_inset_by_a_step_of_the_spacing_scale():
    """This used to read ``PANE_PADDING == 5.0``, which froze one afternoon's
    answer exactly as the spacing-scale test once did: 5 was not a step of
    anything, and the number is a *taste* call UX.md Phase 2 settled against a
    screenshot. What is not taste is that a pane's inset comes from the scale
    rather than being invented, which is the rule this asserts instead.
    """
    from realmspinner.studio import tokens

    steps = {v for k, v in vars(tokens).items() if k.startswith("SP_")}
    assert layout_mod.PANE_PADDING in steps
    # And it is tighter than the host window's own gutter: a pane sits inside
    # that gutter already, so matching it would double the inset on the two
    # sidebars, which are the width-constrained case.
    assert layout_mod.PANE_PADDING < tokens.SP_4


def test_an_unknown_stored_width_falls_back_rather_than_stopping_the_window():
    try:
        assert layout_mod.set_sidebar("enormous") == "default"
        assert layout_mod.SIDEBAR_W == 300.0
        assert layout_mod.set_sidebar("wide") == "wide"
        assert layout_mod.SIDEBAR_WIDTHS["wide"] == layout_mod.SIDEBAR_W
    finally:
        layout_mod.set_sidebar("default")


def test_a_stored_width_is_a_name_and_never_a_number():
    """So a settings file can never carry a size this build does not offer."""
    settings = _Settings({"sidebar": "wide", "settings_share": 0.4})
    try:
        lay = layout_mod.Layout(settings)
        assert lay.sidebar == "wide"
        lay.save()
        assert settings.store["layout"]["sidebar"] == "wide"
    finally:
        layout_mod.set_sidebar("default")


def test_stored_widths_are_ignored_and_only_the_share_is_read():
    lay = layout_mod.Layout(
        _Settings({"sidebar_w": 480.0, "inspector_w": 280.0, "settings_share": 0.4})
    )
    assert lay.settings_share == 0.4
    assert not hasattr(lay, "sidebar_w")
    assert not hasattr(lay, "inspector_w")


def test_a_save_writes_no_width_key_at_all():
    # Settings.set replaces the whole dict rather than merging into it, so the
    # stale keys die the first time anything saves rather than lingering.
    settings = _Settings({"sidebar_w": 480.0, "inspector_w": 280.0, "settings_share": 0.4})
    layout_mod.Layout(settings).save()
    assert settings.store["layout"] == {
        "settings_share": 0.4,
        # The per-split shares (empty here: nothing has been dragged). Written
        # every time for the same reason the rest of this dict is.
        "settings_shares": {},
        "sidebar": "default",
        # The navigation rail's labels-or-icons preference (the UI redesign, wave 3).
        # It has to be written here every time for the reason the whole test
        # exists: the dict is replaced, so a key ``save`` forgets is a
        # preference that silently resets the next time the other one changes.
        "rail": "icons",
    }


def test_the_rail_preference_round_trips_and_survives_nonsense():
    settings = _Settings({"rail": "icons"})
    assert layout_mod.Layout(settings).rail == "icons"
    # A fresh or unknown preference uses the compact editor-first default;
    # only an explicit stored "labels" keeps an existing expanded rail.
    assert layout_mod.Layout(_Settings({"rail": "enormous"})).rail == "icons"
    assert layout_mod.Layout(_Settings({})).rail == "icons"

    layout = layout_mod.Layout(settings)
    layout.set_rail("icons")
    assert settings.store["layout"]["rail"] == "icons"
    assert layout_mod.Layout(settings).rail == "icons"
    layout.set_rail("labels")
    assert settings.store["layout"]["rail"] == "labels"
    assert layout_mod.Layout(settings).rail == "labels"


def test_a_nonsense_share_falls_back_rather_than_raising():
    assert layout_mod.Layout(_Settings({"settings_share": "wide"})).settings_share == 0.55
    assert layout_mod.Layout(_Settings({"settings_share": 9.0})).settings_share == (
        layout_mod.SHARE_MAX
    )


def test_a_split_starts_at_the_shared_default_and_then_goes_its_own_way():
    """One number behind every workspace's split meant Inker's toolbox handle
    silently re-split Create, Clay, Plotter, Packwright, Troupe and Review."""
    lay = layout_mod.Layout(_Settings({"settings_share": 0.4}))
    assert lay.share("clay") == 0.4
    assert lay.share("inker-tools") == 0.4
    lay.set_share("inker-tools", 0.7)
    assert lay.share("inker-tools") == 0.7
    assert lay.share("clay") == 0.4, "a keyed drag must not move another split"
    assert lay.share("inker-tiles") == 0.4, "Inker's handles are separate splits"


def test_a_stored_split_is_clamped_and_junk_is_dropped():
    lay = layout_mod.Layout(
        _Settings(
            {
                "settings_shares": {
                    "clay-tools": 9.0,
                    "review-runs": "wide",
                    "troupe-cast": 0.42,
                }
            }
        )
    )
    assert lay.share("clay-tools") == layout_mod.SHARE_MAX
    assert lay.share("review-runs") == lay.settings_share
    assert lay.share("troupe-cast") == 0.42
    lay.set_share("packwright-items", -3.0)
    assert lay.share("packwright-items") == layout_mod.SHARE_MIN


def test_a_retired_workspace_key_seeds_both_splits_it_used_to_serve():
    """Clay, Plotter, Troupe and Packwright each stacked two panes on the left
    *and* two on the right and read one key for both, so one handle moved a
    column the user was not looking at. Two keys now -- seeded from the old
    one, so an existing profile opens on the proportion it had."""
    lay = layout_mod.Layout(_Settings({"settings_shares": {"clay": 0.31, "create": 0.62}}))
    assert lay.share("clay-tools") == 0.31
    assert lay.share("clay-outliner") == 0.31
    assert lay.share("create-inspector") == 0.62
    # Deleted, not left alongside: ``save`` writes ``shares`` wholesale, so a
    # key left in place would be re-seeded from on every launch for ever.
    assert "clay" not in lay.shares
    assert "create" not in lay.shares


def test_migration_never_overwrites_a_split_the_user_has_already_moved():
    lay = layout_mod.Layout(
        _Settings({"settings_shares": {"plotter": 0.30, "plotter-layers": 0.66}})
    )
    assert lay.share("plotter-tools") == 0.30
    assert lay.share("plotter-layers") == 0.66


# --- The keying, derived from the source rather than kept in step by hand ----
#
# Wave 0's whole finding was that ``Layout.shares`` had been keyed per split
# since the day it was written, and the callers passed the same string twice
# anyway. A list of expected keys maintained here would be one more thing to
# forget beside them; these two read ``main.py`` and check the property.


def _main_source() -> str:
    """Every file that builds a hand-composed split, as source.

    **One file since 2026-09-04, now six since the P4 restructure -- five
    since P9 (2026-09-18) folded Troupe into Poser as a stage, which draws no
    split of its own.** Review's nine hundred lines of pane drawing moved to
    ``studio/review_panes.py`` as a mixin on ``App`` (T7 of the 2026-09-02
    review), and it draws splits and handles like every other workspace. The
    P4 restructure (``dev/RESTRUCTURE.md``) then moved
    ``_split_column``/``_right_column`` themselves out of ``studio/main.py``
    into ``studio/shell/frame.py``, and the two workspaces that called them by
    name at the time (Troupe's and Packwright's) into
    ``studio/troupe_workspace.py`` and ``studio/packwright_workspace.py`` --
    along with Inker's own hand-built timeline splitter, in
    ``studio/modes/inker/ui/workspace.py``. ``main.py`` itself has drawn no split since
    that move; it stays in this list so a future one landing back on the
    shell's entry module is not silently invisible to this scan.
    """
    from realmspinner.studio import main as main_mod
    from realmspinner.studio.modes.inker.ui import workspace as inker_workspace
    from realmspinner.studio.modes.packwright.ui import workspace as packwright_workspace
    from realmspinner.studio.modes.review.ui import workspace as review_panes
    from realmspinner.studio.shell import frame

    sources = [
        pathlib.Path(module.__file__).read_text(encoding="utf-8")
        for module in (
            main_mod,
            review_panes,
            frame,
            inker_workspace,
            packwright_workspace,
        )
    ]
    return "".join(sources)


def _share_literals() -> list[str]:
    """Every string literal ``main.py`` names a split by.

    Both spellings count: ``lay.share("x")`` in hand-built code, and
    ``split_id="x"`` passed to ``_split_column``, which derives the ``share``
    call and the handle's id from it.
    """
    tree = ast.parse(_main_source())
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in {"share", "set_share"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                found.append(node.args[0].value)
            for kw in node.keywords:
                if kw.arg == "split_id" and isinstance(kw.value, ast.Constant):
                    found.append(kw.value.value)
        elif isinstance(node, ast.FunctionDef):
            # ``_right_column``'s ``share_key`` default -- Create names its
            # split in the signature and hands it straight to ``split_id``.
            args = node.args
            for name, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                if name.arg == "share_key" and isinstance(default, ast.Constant):
                    found.append(default.value)
    return found


def _splitter_ids() -> list[str]:
    tree = ast.parse(_main_source())
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "splitter" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant):
                found.append(arg.value)
            elif isinstance(arg, ast.JoinedStr):
                # ``f"{split_id}-share"`` -- the derived form. Recorded by the
                # suffix it contributes, since the id itself is not a literal.
                found.append("<derived>")
    return found


def _skeleton_share_keys() -> list[str]:
    """The share keys declared in ``skeletons.py``'s slot tables.

    Wave 5 moved two workspaces' columns out of ``main`` and into data, so the
    scan has two sources -- and the *union* is what the rules below are about.
    Declared, never derived: a key is in users' settings files forever, and
    computing one from ``f"{workspace}/{slot}"`` would silently reset every
    dragged proportion with no failure anywhere.
    """
    import ast
    import inspect

    from realmspinner.studio import skeletons

    tree = ast.parse(inspect.getsource(skeletons))
    found: list[str] = []
    for node in ast.walk(tree):
        named = isinstance(node, ast.keyword) and node.arg == "share_key"
        if named and isinstance(node.value, ast.Constant) and node.value.value:
            found.append(node.value.value)
    return found


def test_no_split_key_is_used_by_two_splits():
    """The defect this wave fixed, stated so it cannot come back by copy-paste:
    Clay, Plotter, Troupe, Packwright and Review each passed one key to both
    their left and their right column."""
    literals = _share_literals() + _skeleton_share_keys()
    duplicates = sorted({key for key in literals if literals.count(key) > 1})
    assert not duplicates, f"one key serving two splits: {duplicates}"


def test_every_split_has_a_handle_and_every_handle_a_split():
    """Six of the workspaces drew a proportion with no way to change it.

    ``_split_column`` derives the handle's id from ``split_id``, so the check
    is that nothing builds a split outside it -- a hand-built ``lay.share``
    with no matching ``splitter`` is a proportion the user cannot drag, and a
    hand-built ``splitter`` with no share is a handle that moves nothing.
    """
    ids = _splitter_ids()
    # A *set*: Inker's timeline strip is a second hand-composed split (its
    # height is a drag along the bottom of the centre column, which no column
    # renderer owns), and it derives its handle from its key exactly the way
    # ``_split_column`` does. What matters is that no id is a bare literal.
    #
    # The Familiar dock's own grip is gone (2026-09-23's proportional shell:
    # the dock's width is a fixed share of the room, ``layout.proportions``,
    # never a drag), so it no longer appears here.
    assert sorted(set(ids)) == ["<derived>"], (
        "every column's handle should come from _split_column, which derives "
        f"its id from split_id; hand-built splitters found: {sorted(set(ids))}"
    )
    # ``layout.column`` derives its handle the same way, from the slot's own
    # ``share_key``, so a declared key *is* a handle there too.
    keys = set(_share_literals()) | set(_skeleton_share_keys())
    assert keys == {
        # Clay's right column stacks the outliner over the selected object's
        # own settings over the document, so it carries two handles. The left
        # column is one FILL pane since the viewport header took the tool grid,
        # the mode row and the view aids -- and a column of one has nothing to
        # share against, so ``clay-tools`` is no longer a key at all.
        "clay-outliner",
        "clay-props",
        # Tranche 6: the UV pane joins the right column below Properties, a
        # third SHARE slot -- an island layout wants its own canvas the same
        # reason the outliner and the properties pane each already have
        # theirs. See ``skeletons.clay``'s own docstring.
        "clay-uv",
        # Mason stacks two shares in each column: the asset palette over the
        # tools on the left, and the outliner over the properties on the
        # right, with the document pane taking the FILL underneath. Four
        # declared keys, three handles -- the bridge shares against nothing.
        "mason-assets",
        "mason-outliner",
        "mason-props",
        # The Prefabs pane is a *conditional* slot -- in the column only while
        # the scene defines a template -- and a conditional slot still declares
        # a share key: ``layout.column`` derives its handle from the key of
        # whichever slots are live that frame, so the proportion is draggable
        # exactly when the pane is there to drag.
        "mason-prefabs",
        "create-inspector",
        "inker-colors",
        # Inker's right column stacks three shareable panes, and the strip
        # along the bottom of its centre column is a fourth split -- keyed
        # rather than fixed so its height is a drag that persists.
        "inker-tools",
        "inker-tiles",
        # The walk-cycle setup panel, present only while a session is open --
        # ``plotter-objects``' case, and its answer: a ``when`` slot carries a
        # handle, because the column it shares with is the column it is drawn in.
        "inker-walk",
        "inker-timeline",
        "packwright-sources",
        "packwright-items",
        # Plotter took Tiled's arrangement for the left column -- the selected
        # thing over the tile stamps -- and puts its file block at the bottom
        # of the right, where Clay's and Sirens' already are. So the right
        # column stacks the layer list, the objects dock and the tileset
        # palette over the map file, and carries the handles those splits
        # need. ``plotter-tools`` left this set with the pane -- the tools are
        # a strip inside the centre column now, and a strip is not a slot. An
        # orphaned share under ``plotter-tools`` or ``plotter-stamps`` in a
        # user's settings file is inert; see ``skeletons.plotter``.
        "plotter-layers",
        # The Objects dock, between the stack and the palette and present only
        # on a map that has an object layer. A ``when`` slot still carries a
        # handle: the column it shares with is the column it is drawn in.
        "plotter-objects",
        "plotter-properties",
        # The tileset palette, over the map file at the bottom of the right
        # column. The numbered stamps left this set on 2026-09-05: the map file
        # went to the right column, so the stamps became the left column's
        # fill and a fill carries no handle of its own.
        "plotter-tileset",
        "review-runs",
        "sirens-transport",
        # The right column stacks the instrument list over the envelope editor
        # over the song file, so it carries two split handles as Inker's does.
        "sirens-instruments",
        "sirens-envelopes",
        # The sound-effect list is the third shareable pane of that column
        # (Phase 4), so the right column carries three handles.
        "sirens-effects",
        # Troupe's own "troupe-cast"/"troupe-sheets" left this set when P9
        # (2026-09-18) folded that mode into Poser: the character-sheet
        # section draws inline in Poser's existing library/controls panes
        # rather than as a split column of its own, so there is no handle
        # left for it to carry.
    }


def test_measure_shrinks_the_sidebars_as_familiar_opens(monkeypatch):
    """UX-01's failure mode one edge over: the Familiar dock (2026-09-23)
    sits on the window's right edge outside every mode's columns, exactly
    where the old bottom pane never had to be subtracted from (it ran the
    window's full width). ``measure`` now derives every column from
    :func:`layout_mod.proportions`, keyed on :data:`layout_mod.FAMILIAR_OPEN_T`
    -- opening the dock takes width from both sidebars by construction, so
    the right sidebar can never be pushed off-screen the way an
    independently-fitted dock could.
    """
    from _ui_context import imgui_context

    with imgui_context(monkeypatch) as imgui:
        imgui.get_io().display_size = (1600.0, 900.0)
        imgui.new_frame()
        try:
            layout_mod.FAMILIAR_OPEN_T = 0.0
            without_dock = layout_mod.measure()

            layout_mod.FAMILIAR_OPEN_T = 1.0
            with_dock = layout_mod.measure()
        finally:
            imgui.end_frame()
            imgui.render()
            layout_mod.FAMILIAR_OPEN_T = 0.0

    assert with_dock < without_dock


# --- proportions(): the proportional shell (2026-09-23) ----------------------
#
# The rail, the two sidebars, the centre and the Familiar dock used to be
# fitted by two independent pure functions (``fit_widths`` and
# ``familiar_dock.fit``) against the same room, with nothing keeping their
# combined claim under the window -- the reported bug this wave fixes.
# ``proportions`` is the one function that divides the room, so the
# regression below is the shape of the fix: assert the five numbers plus
# the four gaps between them equal the room exactly, at every scale and in
# both Familiar states, which the old ``fit_widths``/``familiar_dock.fit``
# pair could not promise because neither knew about the other.


def test_rail_left_centre_right_dock_and_gaps_equal_room_exactly():
    """The regression this wave exists for. Proven to fail against the old
    ``fit_widths``/``familiar_dock.fit`` pair before this test was written:
    reconstructing their old formulas at a 1600x900 window, 2x UI scale,
    Familiar open and 300 dp panel preferences on both sides gave sidebars
    fitted to 193 physical px each -- well under half of
    :data:`layout_mod.SIDEBAR_MIN` at that scale (400) -- because
    ``fit_widths`` floors only the *centre*, never the sidebars, once the
    dock (fitted independently, with no notion of the rail or the sidebar
    floors at all) has taken its own share first. ``proportions`` cannot
    reproduce that: it divides the room once, so every floor in its own
    ladder is met by moving width between the other four columns, never by
    quietly letting one of them collapse.
    """
    spacing = 8.0
    for room in (1100.0, 1920.0, 2560.0, 3800.0):
        for scale in (1.0, 1.5, 2.0):
            for t in (0.0, 0.35, 1.0):
                rail, left, centre, right, dock = layout_mod.proportions(
                    room, t, spacing, scale=scale
                )
                total = rail + left + centre + right + dock + 4.0 * spacing
                assert total == pytest.approx(room), (room, scale, t)


def test_the_rail_and_the_closed_dock_are_icon_strips_not_shares():
    """2026-09-24: as flat 5% shares the rail and the closed dock were each
    two to three times wider than the 44 dp icon they hold (96 px apiece on a
    1920 window). Both are icon strips again, at every window width and
    scale -- a share would grow with the room, which is what this pins."""
    spacing = 8.0
    for room in (1600.0, 1920.0, 2560.0, 3800.0):
        for scale in (1.0, 1.5):
            rail, _l, _c, _r, dock = layout_mod.proportions(
                room, 0.0, spacing, rail=44.0 * scale, scale=scale
            )
            assert rail == pytest.approx(44.0 * scale), (room, scale)
            assert dock == pytest.approx(44.0 * scale), (room, scale)


def test_the_canvas_gets_what_the_icon_strips_gave_up():
    """The centre is the remainder: on a 1920 window with Familiar closed it
    is wider than the 40% it was held at while the rail and the closed dock
    were 5% shares, by exactly what those two shares gave up."""
    spacing = 8.0
    room = 1920.0
    content = room - 4.0 * spacing
    rail, left, centre, right, dock = layout_mod.proportions(
        room, 0.0, spacing, rail=44.0, scale=1.0
    )
    assert left == right == pytest.approx(content * 0.25)
    assert centre == pytest.approx(content * 0.50 - 88.0)
    assert centre > content * 0.40 + 100.0


def test_proportions_at_the_open_extreme():
    """Open, each sidebar is 20% and the dock 15% of the room (spacing taken
    out), on a window generous enough that no floor engages."""
    spacing = 8.0
    room = 3800.0
    content = room - 4.0 * spacing
    rail, left, centre, right, dock = layout_mod.proportions(
        room, 1.0, spacing, rail=44.0, scale=1.0
    )
    assert rail == pytest.approx(44.0)
    assert left == right == pytest.approx(content * 0.20)
    assert dock == pytest.approx(content * 0.15)
    assert centre == pytest.approx(content * 0.45 - 44.0)


def test_opening_familiar_takes_five_points_from_each_sidebar():
    """Halfway open, each sidebar should have given up exactly half of the
    five points the fully-open state takes -- ``proportions`` interpolates
    linearly, and this is what "linearly" has to mean for a share."""
    spacing = 8.0
    room = 3800.0
    content = room - 4.0 * spacing
    _rail, left_closed, _c, right_closed, _dock = layout_mod.proportions(room, 0.0, spacing)
    _rail, left_half, _c, right_half, _dock = layout_mod.proportions(room, 0.5, spacing)
    assert left_closed - left_half == pytest.approx(content * 0.025)
    assert right_closed - right_half == pytest.approx(content * 0.025)


def test_the_rail_argument_defaults_to_what_rail_tick_reserved(monkeypatch):
    """``measure`` and ``familiar_dock.tick`` read the rail's width from
    :data:`RAIL_RESERVED` -- a labelled rail must narrow the centre, not
    overflow it."""
    monkeypatch.setattr(layout_mod, "RAIL_RESERVED", 188.0)
    rail, *_ = layout_mod.proportions(1920.0, 0.0, 8.0, scale=1.0)
    assert rail == pytest.approx(188.0)
    monkeypatch.setattr(layout_mod, "RAIL_RESERVED", 0.0)
    rail, *_ = layout_mod.proportions(1920.0, 0.0, 8.0, scale=1.0)
    assert rail == pytest.approx(44.0)


def test_the_dock_floor_pulls_from_the_sidebars_then_the_canvas():
    """At a 1100 px window, scale 1, fully open, the dock's raw 15% share is
    well under its 260 dp floor. The floor is met in full: the sidebars give
    what they can above :data:`SIDEBAR_MIN`, and the centre -- the remainder
    -- carries the rest."""
    spacing = 8.0
    room = 1100.0
    rail, left, centre, right, dock = layout_mod.proportions(
        room, 1.0, spacing, rail=44.0, scale=1.0
    )
    assert dock == pytest.approx(260.0)
    assert left == pytest.approx(layout_mod.SIDEBAR_MIN)
    assert right == pytest.approx(layout_mod.SIDEBAR_MIN)
    assert rail == pytest.approx(44.0), "the rail is an icon strip; it never gives"
    assert centre > 0.0
    assert rail + left + centre + right + dock + 4.0 * spacing == pytest.approx(room)


def test_a_closed_dock_never_borrows_the_open_floor():
    """The failure mode a naive "floor at 260 whenever the share is under
    260" rule would have: at t=0 the dock is a slim strip the width of the
    collapsed rail, however narrow the window."""
    spacing = 8.0
    room = 1100.0
    rail, _left, _centre, _right, dock = layout_mod.proportions(
        room, 0.0, spacing, rail=44.0, scale=1.0
    )
    assert dock == pytest.approx(rail)


def test_a_degenerate_room_still_sums_exactly():
    """A room narrower than every floor combined takes the overdraw back off
    the dock, the sidebars and the rail -- never a negative centre and never
    a sum past the room."""
    spacing = 8.0
    for room in (200.0, 400.0, 600.0):
        for t in (0.0, 1.0):
            parts = layout_mod.proportions(room, t, spacing, rail=44.0, scale=2.0)
            assert min(parts) >= 0.0, (room, t, parts)
            assert sum(parts) + 4.0 * spacing == pytest.approx(room), (room, t)
