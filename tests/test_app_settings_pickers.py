"""The two LoRA-picker buttons in Settings never open a picker inline.

Regression test for the 2026-09-14 audit, finding shell-09.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP_SETTINGS = (
    Path(__file__).resolve().parent.parent
    / "src" / "warlock" / "studio" / "panes" / "app_settings.py"
)


def _inline_picker_calls(source: str) -> list[str]:
    """Every ``dialogs.open_file``/``select_folder``/``save_file`` call that
    sits directly in a pane function's body, rather than inside a nested
    ``def run(): ...`` a task thread runs it on.

    Depth-counted rather than name- or text-matched, so it survives
    reformatting and still catches a new inline call added later -- the same
    failure mode that let this pair through the first time (found by the
    pipelines-04 fixer's ``tests/test_exercise_mode.py`` picker-count test).
    A call nested one ``def`` deep from module level (directly in a pane
    function such as ``_loras``) is inline: it runs on the frame thread when
    the button handler executes. One nested a level deeper -- inside a
    closure defined in that function and handed to ``ctx.submit`` -- runs on
    a task thread instead, which is the shape every other picker site under
    ``studio/`` already uses (``app_ctx.py``'s ``save_artifact``,
    ``inker_open.py``'s ``ask_open``, and about fifty more).
    """

    tree = ast.parse(source)
    found: list[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in ("open_file", "select_folder", "save_file")
                and isinstance(func.value, ast.Name)
                and func.value.id == "dialogs"
                and self.depth <= 1
            ):
                found.append(func.attr)
            self.generic_visit(node)

    Visitor().visit(tree)
    return found


def test_the_lora_import_and_train_buttons_never_open_a_picker_on_the_frame_thread():
    """The 2026-09-14 audit, finding shell-09.

    ``_loras``' "Import a LoRA file..." and "Train from a folder..." buttons
    used to call ``dialogs.open_file``/``select_folder`` straight from the
    draw call, on the frame thread. A native picker is modal to the OS, so
    ``App.frame`` blocked for as long as the dialog stayed open -- no
    repaint, a "not responding" window -- while every other picker site
    under ``studio/`` opened it inside a task-thread closure instead. Both
    buttons now submit a ``def run(): ...`` closure under the "preview" key,
    the same landing spot "Train from my library..." two lines below already
    used.
    """

    source = APP_SETTINGS.read_text(encoding="utf-8")
    assert _inline_picker_calls(source) == []
