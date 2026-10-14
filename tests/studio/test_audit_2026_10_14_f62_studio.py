"""Regression test for the 2026-10-14 audit, fixer F62, finding shell-chrome-10:
the menu bar's "no stray root" test iterated a hand list of modes."""

from __future__ import annotations

from types import SimpleNamespace


def _ctx(mode: str):
    from realmspinner.studio.modes.inker import state as inker_state

    ctx = SimpleNamespace(
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
    ctx.state.inker = inker_state.InkerState()  # ``_inker_specs`` ensures it in Inker
    return ctx


def _stray_command_roots(mode: str):
    """Roots a *command* row spawned outside ``menus.ROOTS`` in ``mode``. A mode
    may own contextual roots of its own (Inker's Layer, Frame, ...); what a
    command row must never do is open one."""
    from realmspinner.studio import menus

    rows = menus.specs(_ctx(mode), evaluate=False)
    return sorted(
        {r.path for r in rows if r.identity.startswith("command:") and r.path[0] not in menus.ROOTS}
    )


def test_no_mode_in_modes_keys_spawns_a_stray_menu_root():
    """The 2026-10-14 audit, finding shell-chrome-10. The test it replaces
    iterated a hand list that named the retired ``poser`` and ``troupe`` and
    left out ``home``, ``library``, ``create``, ``inker`` and ``settings``, so a
    stray root in any of those would have passed."""
    from realmspinner.studio import modes

    strays = {mode: _stray_command_roots(mode) for mode in modes.KEYS}
    assert not any(strays.values()), strays


def test_the_stray_root_scan_sees_a_stray_root_in_a_mode_the_old_list_omitted(monkeypatch):
    """The negative control: a command row sent to a made-up root in Inker (a
    mode the hand list omitted) is reported, and only there."""
    from realmspinner.studio import menus

    real = menus._command_specs

    def planted(ctx, commands, *, evaluate=True):
        rows = real(ctx, commands, evaluate=evaluate)
        if ctx.state.mode == "inker":
            rows = [r.__class__(**{**r.__dict__, "path": ("Stray",)}) for r in rows]
        return rows

    monkeypatch.setattr(menus, "_command_specs", planted)
    assert _stray_command_roots("inker") == [("Stray",)]
    assert _stray_command_roots("clay") == []
