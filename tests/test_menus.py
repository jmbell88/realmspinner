"""The application menu model: whether a command reaches the menu bar.

``menus.py`` is an adapter over the command palette's list -- see
``palette.py``'s own docstring -- so most of its behaviour is exercised
through ``tests/test_editor_shell.py``'s parity checks between the two
surfaces. This file is for defects that are ``menus.py``'s own: a command
present in the palette that ``_COMMAND_PATHS`` sends nowhere, or sends
somewhere wrong.
"""

from __future__ import annotations

from types import SimpleNamespace


def _ctx(mode: str = "home"):
    return SimpleNamespace(
        state=SimpleNamespace(
            mode=mode,
            selected=None,
            wireframe=False,
            turntable=False,
            show_fps=False,
            filters=SimpleNamespace(trash=False),
            errors=[],
        ),
        cache=SimpleNamespace(get=lambda _key: None, jobs=[]),
        runtime=SimpleNamespace(checks=[]),
        viewer=None,
    )


def test_reroll_is_reachable_from_the_menu_bar_in_the_library():
    """Shell-06, the 2026-09-07 audit. ``reroll`` had no ``_COMMAND_PATHS``
    entry, so ``_command_specs`` fell through to the mode-specific contextual
    menu instead of ``Edit`` -- and that branch only fires outside Home,
    Settings and Library, so the command was absent from exactly the two
    modes where selecting an asset and rerolling it actually happens, while
    turning up in the contextual menu of nine unrelated workspace modes.
    Chapter 38 states the palette and the menu bar carry the same commands.
    """
    from warlock.studio import menus

    for mode in ("home", "library"):
        rows = menus.specs(_ctx(mode))
        row = next((r for r in rows if r.identity == "command:reroll"), None)
        assert row is not None, f"reroll is missing from the menu bar in {mode}"
        assert row.path == ("Edit",)


def test_specs_builds_the_command_list_once_per_call(monkeypatch):
    """Shell-11, the 2026-09-07 audit. ``_inker_export_specs`` rebuilt the
    ~30-command palette list a second time per call -- once via
    ``_command_specs`` and again for its own lookup of the "export" row's
    index -- and ``specs`` itself is called up to twice a frame (the bar's
    shape, then whichever root menu is open), so a frame with a menu open
    rebuilt the list up to four times. ``specs`` now builds it once and hands
    it down to both.
    """
    from warlock.studio import inker_state, menus, palette

    ctx = _ctx("inker")
    ctx.state.inker = inker_state.InkerState()

    calls: list[int] = []
    real_commands = palette.commands

    def counting(c):
        calls.append(1)
        return real_commands(c)

    monkeypatch.setattr(palette, "commands", counting)

    menus.specs(ctx)

    assert calls == [1], f"palette.commands(ctx) ran {len(calls)} times, want 1"


# --- T0, the Familiar programme: the menu bar's status group ---------------


def test_status_items_drop_lowest_priority_first_when_the_menus_need_the_room():
    """``fit_status_rows`` drops one key at a time off ``STATUS_DROP_ORDER``
    (resources, then zoom, then tool, then document, then queue) until what
    is left fits -- proven by an available width that only fits after three
    of the five droppable rows are gone, and checking it is exactly those
    three, in that order, rather than merely a count.
    """
    from warlock.studio import menus, status_bar

    rows = [
        status_bar.StatusItem("workspace", "Inker"),
        status_bar.StatusItem("document", "Untitled"),
        status_bar.StatusItem("tool", "Brush"),
        status_bar.StatusItem("zoom", "100%"),
        status_bar.StatusItem("resources", "RAM 1/8"),
        status_bar.StatusItem("queue", "Queue 1 active"),
        status_bar.StatusItem("health", "1 issue(s)", True),
    ]

    def measure(item):
        return 10.0

    # 7 rows * 10 = 70. Dropping resources (60), then zoom (50), then tool
    # (40) is exactly enough to fit 40 -- document and queue must survive.
    fitted = menus.fit_status_rows(rows, 40.0, measure)
    assert {row.key for row in fitted} == {"workspace", "document", "queue", "health"}


def test_health_is_never_dropped_from_the_menu_bar():
    """However little room is left, ``health`` (and the leading ``workspace``
    row) must survive -- ``STATUS_DROP_ORDER`` never names them, so the loop
    that removes keys off it cannot touch them even once every droppable key
    is gone.
    """
    from warlock.studio import menus, status_bar

    rows = [
        status_bar.StatusItem("workspace", "Inker"),
        status_bar.StatusItem("document", "Untitled"),
        status_bar.StatusItem("tool", "Brush"),
        status_bar.StatusItem("zoom", "100%"),
        status_bar.StatusItem("resources", "RAM 1/8"),
        status_bar.StatusItem("queue", "Queue 1 active"),
        status_bar.StatusItem("health", "1 issue(s)", True),
    ]

    fitted = menus.fit_status_rows(rows, 0.0, lambda item: 10.0)

    assert "health" in {row.key for row in fitted}
    assert "workspace" in {row.key for row in fitted}
    for key in menus.STATUS_DROP_ORDER:
        assert key not in {row.key for row in fitted}


def test_status_items_render_right_aligned_in_the_menu_bar(monkeypatch):
    """A real imgui frame: the status group's last item (``health``, since
    it is never dropped) must land in the right half of a wide menu bar, and
    the ``Familiar`` menu must have been drawn -- proving the group is placed
    after every root and the reserved Familiar entry rather than immediately
    following ``File``.
    """
    from _ui_context import imgui_context

    from warlock.studio import menus

    ctx = _ctx("home")
    ctx.state.errors = ["boom"]  # forces a "health" row to exist

    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.set_next_window_size((1600, 950))
        imgui.begin("##host", None, imgui.WindowFlags_.menu_bar.value)
        menus.draw(ctx)
        rect_min = imgui.get_item_rect_min()
        imgui.end()
        imgui.render()

    assert rect_min.x > 1600 * 0.55, rect_min.x
