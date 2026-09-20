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
    from realmspinner.studio.modes.settings.ui.panes import app_settings

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
    from realmspinner.studio.modes.settings.ui.panes import app_settings

    assert _uses_any_busy_lora_prefix(app_settings._lora_train_form), (
        "_lora_train_form never calls ctx.tasks.any_busy(\"lora:\"), the "
        "prefix rule _LORA_BUSY_REASON documents for this section"
    )
    call = _disabled_button_call(app_settings._lora_train_form, "Train style")
    enabled_expr = ast.unparse(call.args[1])
    assert 'ctx.busy("lora:train")' not in enabled_expr, (
        f"'Train style' still gates on its own exact task key: {enabled_expr!r}"
    )


def test_lora_remove_and_import_do_not_toast_completion_before_the_task_runs():
    """shell-05 (2026-09-18 audit).

    ``_loras``' Remove button and ``_lora_import_form``'s Add style button
    used to toast "Removed {label}." / "Style added." the instant
    ``ctx.submit`` accepted the task -- proof only that the job reached the
    queue, not that ``svc_loras.remove_lora``/``import_lora`` ever ran. A
    refusal (a built-in key, an already-deleted manifest, a form the loader
    rejects) or a plain disk error then toasted a success the task never
    earned. Both submit sites now toast something progressive instead, and
    the real outcome is reported once ``shell/tasks.py``'s
    ``TasksMixin._on_task_done`` lands the task under the exact key each
    site submits (``"lora:remove:"``/``"lora:import"``) -- both were
    previously unclaimed there, so a landed removal or import fell through
    to the silent "nothing claimed it" path with no toast at all.
    """
    from realmspinner.studio.modes.settings.ui.panes import app_settings
    from realmspinner.studio.shell import tasks as shell_tasks

    remove_source = inspect.getsource(app_settings._loras)
    import_source = inspect.getsource(app_settings._lora_import_form)
    landing_source = inspect.getsource(shell_tasks.TasksMixin._on_task_done)

    # The submit sites must not claim the outcome before it happened.
    assert 'ctx.toast(f"Removed {row.label}.")' not in remove_source, (
        "the Remove button still toasts completion wording at submit"
    )
    assert 'ctx.toast("Style added.")' not in import_source, (
        "the Add style button still toasts completion wording at submit"
    )

    # The landing handler is the one place that now claims the outcome, and
    # it is keyed on the exact submit key each site uses.
    assert '"lora:remove:"' in landing_source, (
        "_on_task_done never claims the \"lora:remove:\" key the Remove "
        "button's own ctx.submit uses"
    )
    assert '"lora:import"' in landing_source, (
        "_on_task_done never claims the \"lora:import\" key the Add style "
        "button's own ctx.submit uses"
    )
    assert "Removed" in landing_source
    assert "added" in landing_source
