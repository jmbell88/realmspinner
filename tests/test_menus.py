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
