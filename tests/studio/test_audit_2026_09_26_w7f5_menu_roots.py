"""The 2026-09-26 audit, finding shell-chrome-02.

Manual 20 (``docs/manual/20-overview.md``) used to promise a per-mode menu
root -- naming Clay, Plotter and Mason as examples -- that no mode actually
built: every ``Actions``/``Viewport`` command already has a ``_COMMAND_PATHS``
entry, so ``menus._command_specs``'s contextual per-mode branch (which would
have synthesized a root named after ``ctx.state.mode``) never fired for a real
mode. The manual was corrected to say no mode gets a root of its own, except
Inker, which contributes six of its own (Sprite, Layer, Frame, Select, Sheet
and Flourish) via its own op table -- and the dead branch was deleted from
``_command_specs``.

This test is the regression: it walks every mode in the one authoritative
list (``modes.MODES``) and checks the menu bar's actual roots against exactly
that claim.
"""

from __future__ import annotations

from types import SimpleNamespace


def _ctx(mode: str):
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


def test_command_specs_never_synthesizes_a_contextual_root_for_an_unmapped_action():
    """Direct regression for the dead branch itself: an ``Actions``/``Viewport``
    command with no ``_COMMAND_PATHS`` entry used to fall through to a
    contextual root named after ``ctx.state.mode`` -- reachable only in
    theory, since every real command has had a path since the ``generate``
    and ``reroll`` fixes, but still live code a future command could
    resurrect it for. It must now be dropped instead."""
    from realmspinner.studio import menus

    class FakeCommand:
        key = "shell-chrome-02-unmapped-action"
        group = "Actions"
        label = "Do a thing"
        hint = ""
        why = ""

        def enabled(self, ctx):
            return True

        def run(self, ctx):
            pass

    rows = menus._command_specs(_ctx("clay"), [FakeCommand()], evaluate=False)
    assert rows == []


def test_every_mode_the_manual_says_has_its_own_menu_root_has_one():
    """The manual now claims exactly one mode (Inker) has mode-specific menu
    roots of its own; every other mode folds its actions into the six fixed
    roots. Before the fix this failed for no mode (the dead branch never
    fired), but it stands as the gate against that branch -- or an
    equivalent one -- coming back for a mode with an ungated command."""
    from realmspinner.studio import menus, modes

    for key, _label, _icon, _purpose in modes.MODES:
        if key == "inker":
            continue
        rows = menus.specs(_ctx(key), evaluate=False)
        stray = sorted({row.path[0] for row in rows if row.path and row.path[0] not in menus.ROOTS})
        assert not stray, f"mode {key!r} has a menu root of its own: {stray}"

    from realmspinner.studio.modes.inker import state as inker_state

    inker_ctx = _ctx("inker")
    inker_ctx.state.inker = inker_state.InkerState()
    inker_rows = menus.specs(inker_ctx, evaluate=False)
    inker_roots = menus.roots(inker_rows)
    contextual = [r for r in inker_roots if r not in menus.ROOTS]
    assert contextual == ["Sprite", "Layer", "Frame", "Select", "Sheet", "Flourish"]
