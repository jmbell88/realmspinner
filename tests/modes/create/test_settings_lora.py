"""Regression tests for the 2026-09-16 audit, finding shell-settings-01.

``_lora_import_form``'s "Add style" button and ``_lora_train_form``'s "Train
style" button used to gate their ``enabled`` state on their own exact task
key (``ctx.busy("lora:import")`` / ``ctx.busy("lora:train")``) instead of the
pane's ``ctx.tasks.any_busy("lora:")`` prefix rule every other LoRA-mutating
control in this section (Remove, Import, Train-from-folder,
Train-from-library) already shares via ``app_settings._LORA_BUSY_REASON`` --
so pressing one of these two submit buttons while a *different*
``lora:``-prefixed task was already in flight was not blocked, and a second
concurrent LoRA mutation could start.
"""

from __future__ import annotations

import ast
import inspect


def _disabled_button_call(func, label: str) -> ast.Call:
    """The ``widgets.disabled_button(label, ...)`` call inside ``func``."""

    source = inspect.getsource(func)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "disabled_button"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == label
        ):
            return node
    raise AssertionError(f"no widgets.disabled_button({label!r}, ...) call found")


def _uses_any_busy_lora_prefix(func) -> bool:
    """Whether ``func`` calls ``ctx.tasks.any_busy("lora:")`` anywhere,
    the prefix rule ``_LORA_BUSY_REASON`` documents for this section."""

    source = inspect.getsource(func)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "any_busy"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "lora:"
        ):
            return True
    return False


def test_add_style_is_disabled_while_a_different_lora_operation_is_running():
    """shell-settings-01: "Add style" used to gate on
    ``ctx.busy("lora:import")`` -- its own exact submit key -- alone.
    """
    from warlock.studio.modes.settings.ui.panes import app_settings

    assert _uses_any_busy_lora_prefix(app_settings._lora_import_form), (
        "_lora_import_form never calls ctx.tasks.any_busy(\"lora:\"), the "
        "prefix rule _LORA_BUSY_REASON documents for this section"
    )
    call = _disabled_button_call(app_settings._lora_import_form, "Add style")
    enabled_expr = ast.unparse(call.args[1])
    assert 'ctx.busy("lora:import")' not in enabled_expr, (
        f"'Add style' still gates on its own exact task key: {enabled_expr!r}"
    )


def test_train_style_is_disabled_while_a_different_lora_operation_is_running():
    """shell-settings-01: "Train style" used to gate on
    ``ctx.busy("lora:train")`` -- its own exact submit key -- alone, so
    pressing it while, say, an import was in flight was not blocked.
    """
    from warlock.studio.modes.settings.ui.panes import app_settings

    assert _uses_any_busy_lora_prefix(app_settings._lora_train_form), (
        "_lora_train_form never calls ctx.tasks.any_busy(\"lora:\"), the "
        "prefix rule _LORA_BUSY_REASON documents for this section"
    )
    call = _disabled_button_call(app_settings._lora_train_form, "Train style")
    enabled_expr = ast.unparse(call.args[1])
    assert 'ctx.busy("lora:train")' not in enabled_expr, (
        f"'Train style' still gates on its own exact task key: {enabled_expr!r}"
    )
