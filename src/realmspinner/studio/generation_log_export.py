"""Export the generation log: pick a folder, write the JSONL into it.

One module for the two buttons that do it -- Settings -> Advanced and Create's
History header -- so there is one folder picker, one task key and one landing
(``shell/tasks.py`` toasts the path). It sits at ``studio/`` rather than in
either mode because neither may import the other, and the work itself
(``service.generation_log.export_jsonl``) is not UI.
"""

from __future__ import annotations

from typing import Any

from ..service import generation_log as svc_generation_log
from . import dialogs

#: The task key both buttons submit under, and the one the landing branches on.
KEY = "generation-log-export"


def task(svc: Any) -> str | None:
    """Pick a folder, then write the log into it. -> the file's path, or ``None``
    when the picker was cancelled.

    On the task thread: the picker is modal to the OS (the shell-09 rule in the
    Settings pane) and the export reads the whole verdict history.
    """
    folder = dialogs.select_folder("Where should the generation log go?")
    if folder is None:
        return None
    return str(svc_generation_log.export_jsonl(svc, folder))


def request(ctx: Any) -> bool:
    """Start an export unless one is already running. -> whether it started."""
    return bool(ctx.submit(KEY, task, ctx.svc))
