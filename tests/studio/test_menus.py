"""The application menu model: whether a command reaches the menu bar.

``menus.py`` is an adapter over the command palette's list -- see
``palette.py``'s own docstring -- so most of its behaviour is exercised
through ``tests/studio/test_editor_shell.py``'s parity checks between the two
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
    from realmspinner.studio import menus

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
    from realmspinner.studio import menus, palette
    from realmspinner.studio.modes.inker import state as inker_state

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
    from realmspinner.studio import menus, status_bar

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
    from realmspinner.studio import menus, status_bar

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

    from realmspinner.studio import menus

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


def test_status_group_items_do_not_overlap(monkeypatch):
    """Familiar T0 (ef853790): every status readout landed at the same x as
    the one before it. ``_draw_status_group`` chained ``same_line(0.0, 0.0)``
    calls after an initial absolute jump, which is how an ordinary window's
    line-wrapping layout accumulates left-to-right -- but a menu bar runs its
    own cursor bookkeeping, and there ``same_line(0.0, 0.0)`` kept landing
    back at the first item's start instead of after the previous item, so
    "Inker" and "Loaded 31.2/32 RAM..." printed on top of each other.

    Wraps ``imgui.text_colored`` to record each drawn item's on-screen rect
    and asserts none overlap the next, and that the last one ends near the
    menu bar's right edge.
    """
    from _ui_context import imgui_context

    from realmspinner.studio import menus

    ctx = _ctx("home")
    ctx.state.errors = ["boom"]  # forces a "health" row to exist

    rects: list[tuple[float, float]] = []
    with imgui_context(monkeypatch) as imgui:
        original = imgui.text_colored

        def recording_text_colored(color, text):
            start = imgui.get_cursor_screen_pos().x
            result = original(color, text)
            rects.append((start, start + imgui.calc_text_size(text).x))
            return result

        monkeypatch.setattr(imgui, "text_colored", recording_text_colored)

        imgui.new_frame()
        imgui.set_next_window_size((1600, 950))
        imgui.begin("##host", None, imgui.WindowFlags_.menu_bar.value)
        menus.draw(ctx)
        imgui.end()
        imgui.render()

    assert len(rects) >= 2, "expected at least two status readouts to be drawn"
    for (_, prev_end), (next_start, _) in zip(rects, rects[1:]):  # noqa: B905 (offset pairing)
        assert next_start >= prev_end, f"status items overlap: {rects}"
    assert rects[-1][1] > 1600 * 0.9, rects


def test_the_status_group_never_runs_past_the_menu_bar_edge(monkeypatch):
    """Familiar T0 (ef853790): at 1100x700 the status group clipped at the
    right edge ("RAM 23." cut off) instead of ``fit_status_rows`` dropping
    ``resources`` first. ``_draw_status_group`` fit the group against
    ``get_content_region_avail()`` -- the content-region right edge, which
    already excludes the window's frame padding -- but then placed it against
    ``get_window_width()``, the *full* window width. That disagreement let a
    group the fit had approved land past where the fit thought the edge was.

    A real host window at 1100x700 (the narrowest ``screenshot_modes.py``
    size) with Inker's own roots drawn, at window position (0, 0) so screen
    coordinates and window-local coordinates coincide -- proving the last
    status item's rect never extends past the content region's own right
    edge, not just "some" edge.
    """
    from _ui_context import imgui_context

    from realmspinner.studio import menus
    from realmspinner.studio.modes.inker import state as inker_state

    ctx = _ctx("inker")
    ctx.state.inker = inker_state.InkerState()
    ctx.state.errors = ["boom"]  # forces a "health" row to exist

    width = 1100.0
    rects: list[tuple[float, float]] = []
    with imgui_context(monkeypatch) as imgui:
        original = imgui.text_colored

        def recording_text_colored(color, text):
            start = imgui.get_cursor_screen_pos().x
            result = original(color, text)
            rects.append((start, start + imgui.calc_text_size(text).x))
            return result

        monkeypatch.setattr(imgui, "text_colored", recording_text_colored)

        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((width, 700))
        imgui.begin("##host", None, imgui.WindowFlags_.menu_bar.value)
        menus.draw(ctx)
        window_padding_x = imgui.get_style().window_padding.x
        imgui.end()
        imgui.render()

    content_max_x = width - window_padding_x

    assert rects, "expected at least one status readout to be drawn"
    assert rects[-1][1] <= content_max_x, (
        f"status group runs past the content region edge: last item ends at "
        f"{rects[-1][1]}, content region ends at {content_max_x}"
    )


def test_the_generate_command_does_not_spawn_a_stray_menu_root_outside_create():
    """The 2026-09-18 audit, shell-03 (second run). ``generate`` (Generate /
    Make 3D) is enabled only in Create (``palette._in_generate_mode``) and had
    no ``_COMMAND_PATHS`` entry, so ``_command_specs``' contextual branch
    caught it in every other mode that branch reaches and spawned a stray
    one-item disabled menu root named after that mode."""
    from realmspinner.studio import menus

    modes_outside_create = (
        "clay", "mason", "poser", "troupe", "plotter", "packwright", "muse", "sirens", "review",
    )
    for mode in modes_outside_create:
        rows = menus.specs(_ctx(mode), evaluate=False)
        row = next((r for r in rows if r.identity == "command:generate"), None)
        assert row is not None, f"generate is missing from the menu bar in {mode}"
        assert row.path == ("File",)
        assert not any(r.path[0] not in menus.ROOTS for r in rows), (
            f"a stray menu root should not exist in {mode}, got paths "
            f"{sorted({r.path for r in rows if r.path[0] not in menus.ROOTS})}"
        )


def _row(path):
    """A minimal ``MenuSpec`` naming only ``path``, which is all ``roots()``
    reads."""
    from realmspinner.studio import menus

    return menus.MenuSpec(
        identity=f"test:{'/'.join(path)}",
        path=path,
        order=0,
        label="x",
        enabled=True,
        checked=False,
        shortcut="",
        disabled_reason="",
        callback=lambda: None,
    )


def test_roots_places_a_contextual_workspace_menu_before_view():
    """shell-09 (the 2026-09-20 audit): ``roots()``'s own docstring promises
    "contextual workspace menus before View" and nothing in the suite ever
    called it -- so a mode's contextual root landing after View/Window would
    read as the app's own fixed menus having reordered themselves, with
    nothing failing to say otherwise."""
    from realmspinner.studio import menus

    rows = [_row(("File",)), _row(("Edit",)), _row(("Clay",)), _row(("View",))]
    order = menus.roots(rows)
    assert order.index("Clay") < order.index("View")
    assert order == ["File", "Edit", "Clay", "View", "Workspace", "Window", "Help"]


def test_roots_is_just_the_six_fixed_roots_when_no_row_is_contextual():
    """The six named in :data:`menus.ROOTS` are permanent menu-bar sections
    (drawn even empty) -- unlike a contextual root, which only appears when a
    row actually asked for it. With no contextual row at all, ``roots()``
    must return exactly the fixed six, in their declared order, regardless of
    which of them a particular frame's rows happen to populate."""
    from realmspinner.studio import menus

    assert menus.roots([]) == list(menus.ROOTS)
    assert menus.roots([_row(("File",)), _row(("Help",))]) == list(menus.ROOTS)
