"""Regression tests for the 2026-09-15 audit, shell-B fix batch.

shell-04 (Mason's empty scene had no working action), shell-08 (Overlay's
dead Mason placeholder entry), shell-05 (Settings claiming to import before
Add style is pressed) and shell-07 (Launch sweep greying with no reason).
shell-10 (a stale count in a ``modes.py`` comment) is docstring-only and has
no regression test -- see the fixer's return for that one.
"""

from __future__ import annotations

import ast
import inspect


def test_masons_empty_scene_placeholder_has_a_working_action(monkeypatch):
    """shell-04: ``mason_viewport`` asks ``overlay.action_for(ctx, "mason")``
    for the button under its "Add something" empty state. Before this fix
    ``ACTIONS`` had no ``"mason"`` key, so ``action_for`` returned ``None``
    and the placeholder drew with no button at all.
    """
    from realmspinner.studio.modes.mason import mode as mason_mode
    from realmspinner.studio.panes import overlay

    assert "mason" in overlay.ACTIONS

    calls: list[str] = []
    monkeypatch.setattr(
        mason_mode, "place_primitive", lambda ctx, generator: calls.append(generator)
    )

    entry = overlay.action_for(object(), "mason")
    assert entry is not None, 'ACTIONS has no "mason" key, so the button never draws'
    label, run = entry
    assert label

    run()
    assert len(calls) == 1


def test_the_mason_button_never_names_a_generator():
    """Mirrors ``_clay_box``: the generator name comes off the registry
    rather than being spelled in the pane, so a rename in
    ``primitives.GENERATORS`` cannot leave this button pointing at nothing.
    """
    from realmspinner.kernels.mesh import primitives as bp
    from realmspinner.studio.panes import overlay

    source = inspect.getsource(overlay._mason_box)
    for name in bp.GENERATORS:
        assert f'"{name}"' not in source, f"{name} is hardcoded in overlay"


def test_the_mason_placeholder_entry_matches_what_mason_viewport_draws():
    """shell-08: ``PLACEHOLDERS["mason"]`` described a mode with no document
    at all ("Stage A", before Mason had one) and nothing ever dispatched
    through it -- ``mason_viewport`` reaches its empty state straight out of
    ``centred_empty``, the way Clay's "Add a shape" does, not through
    ``overlay.placeholder``.

    Removing the entry (rather than repointing it) was tried first and
    reopened: it satisfied this file but broke that table's own sibling gate,
    ``test_notifications.py``'s
    ``test_every_mode_that_draws_the_viewport_has_its_own_placeholder``, which
    requires an entry for every key in ``modes.WORK_MODES`` -- Mason included,
    the same as every other engine. So the entry stays, repointed to the exact
    icon/title/hint ``mason_viewport`` hardcodes at its own call site, so the
    two cannot silently drift back apart the way "Stage A" did once Mason grew
    a real document.
    """
    from realmspinner.studio import icons
    from realmspinner.studio.modes.mason.ui import viewport as mason_viewport
    from realmspinner.studio.panes import overlay

    assert overlay.PLACEHOLDERS["mason"] == (
        icons.BLOCKS,
        "Add something",
        "Pick one from the Assets panel.",
    )
    # Nothing dispatches through the table yet -- ``mason_viewport`` still
    # calls ``centred_empty`` with its own literals rather than reading this
    # entry -- so the equality above is the only thing keeping the two from
    # drifting apart again; it is not itself proof that the table is wired up.
    source = inspect.getsource(mason_viewport)
    assert "overlay.placeholder(" not in source


def _top_level_busy_calls(func) -> list[ast.Call]:
    """Calls to ``widgets.busy`` directly in ``func``'s body -- not nested
    inside an ``if`` -- which is what "drawn as soon as the picker returns"
    looks like in the AST."""

    source = inspect.getsource(func)
    tree = ast.parse(source)
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)

    found: list[ast.Call] = []
    for node in fn.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "busy"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "widgets"
            ):
                found.append(call)
    return found


def test_the_lora_import_form_does_not_claim_to_be_importing_before_add_style_is_pressed():
    """shell-05: this form drew ``widgets.busy("Importing ...")`` the instant
    the file picker returned a source, before "Add style" was ever pressed --
    unconditionally, at the top of the function. It must now be gated on
    ``ctx.busy("lora:import")``, the key "Add style" itself submits under.
    """
    from realmspinner.studio.modes.settings.ui.panes import app_settings

    assert _top_level_busy_calls(app_settings._lora_import_form) == [], (
        "widgets.busy(...) is drawn unconditionally in _lora_import_form, "
        "before ctx.busy('lora:import') gates it"
    )

    source = inspect.getsource(app_settings._lora_import_form)
    assert 'ctx.busy("lora:import")' in source


def test_launch_sweep_button_greys_with_a_reason_while_a_scan_or_submit_is_in_flight():
    """shell-07: "Launch sweep" used to grey during a scan or a submit with
    no ``reason=`` at all, unlike Rescan and Remove on the same pane."""
    from realmspinner.studio.modes.review.ui.workspace import _launch_sweep_reason

    assert _launch_sweep_reason(3, submitting=False, scanning=True) != ""
    assert _launch_sweep_reason(3, submitting=True, scanning=False) != ""
    # Live and ready: no reason to show, because there is nothing to explain.
    assert _launch_sweep_reason(3, submitting=False, scanning=False) == ""
    # Scanning wins over submitting when (improbably) both are true, so the
    # copy never claims two things are happening in one sentence.
    assert _launch_sweep_reason(3, submitting=True, scanning=True) == (
        _launch_sweep_reason(3, submitting=False, scanning=True)
    )
