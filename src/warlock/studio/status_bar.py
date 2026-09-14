"""Compact application status shared by every workspace.

Pure data only (:func:`items`, :func:`resource_item`) -- T0 of the Familiar
programme moved the imgui drawing itself into ``menus.draw`` (the per-item
row) and ``panes/bottom_pane.py`` (the one collapsed row at the foot of the
window). This module's own ``draw`` and ``STATUS_H`` stopped being called
the same day and sat here unreferenced until shell-04 (the 2026-09-14 audit)
found them still claiming to be live -- with INVARIANTS, ``bottom_pane.py``'s
docstring and ``test_editor_shell.py`` all already saying otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StatusItem:
    key: str
    text: str
    warning: bool = False


def _document_name(tab: Any) -> str:
    """A tab's label with imgui's id suffix taken off.

    **Every mode, not just Inker.** A tab label carries the id the widget that
    draws it is keyed on -- ``Untitled##pd1`` in Inker, ``Untitled###pl1`` in
    Plotter -- and that is markup, not a name. Inker learned to split it and
    the branch covering the other five modes did not, so a Plotter map read
    ``Untitled###pl1 *`` down here for as long as the status bar has existed.
    Found by looking at a screenshot; no test compared the two branches,
    because both were only ever asserted on their *keys*.

    Split on ``##`` and take the head, which handles ``###`` too -- the third
    hash lands at the start of the discarded tail.
    """
    return str(getattr(tab, "label", "Untitled")).split("##")[0] or "Untitled"


def _document_modes() -> frozenset[str]:
    """The modes this bar reports a document/tool/zoom row for -- everything
    :data:`palette._DOC_MODES` calls a document mode except "poser", which is
    handled by its own branch above (a viewer, not a tab).

    Derived rather than a second, hand-written tuple: the two used to be two
    independent literals, and a document mode added to ``_DOC_MODES`` in
    future had no test tying this one to it -- the "seventh kind left out of
    a hand-written list" pattern this codebase repeats (shell-08, the
    2026-09-08 audit).
    """
    from . import palette

    return frozenset(palette._DOC_MODES) - {"poser"}


def items(ctx: Any) -> list[StatusItem]:
    """Current status as data so the shell and tests share one account."""

    from . import modes

    mode = str(getattr(ctx.state, "mode", "home"))
    label = next(
        (name for key, name, _icon, _purpose in modes.MODES if key == mode), mode.title()
    )
    out = [StatusItem("workspace", label)]

    if mode == "poser":
        # Poser's "tab" is the viewer (``poser_mode.active``), which has no
        # label and no ``dirty``: the row read a constant ``Untitled`` however
        # dirty the pose was, until 2026-09-05. The name and the flag come
        # from the mode, which knows which library record is being edited.
        from . import poser_mode

        named = poser_mode.document_label(ctx)
        if named is not None:
            name, dirty = named
            out.append(StatusItem("document", f"{name}{' *' if dirty else ''}"))
    elif mode in _document_modes():
        # **One branch for every document mode.** Inker alone reported its
        # tool and zoom, from a branch of its own; Plotter and Packwright
        # carry the identical ``PaintView.zoom`` and Plotter has tools, and
        # the bar is "shared by every workspace" (line 1). What a mode has is
        # asked for by name and drawn if it answers (2026-09-05).
        try:
            from importlib import import_module

            module = import_module(f".{mode}_mode", __package__)
            tab = module.active(ctx)
            if tab is not None:
                dirty = " *" if bool(getattr(tab, "dirty", False)) else ""
                out.append(StatusItem("document", f"{_document_name(tab)}{dirty}"))
                tool = getattr(module, "tool_label", None)
                label = tool(ctx) if tool is not None else ""
                if label:
                    out.append(StatusItem("tool", label))
                view = getattr(tab, "view", None)
                zoom = getattr(view, "zoom", None)
                if zoom is not None:
                    out.append(StatusItem("zoom", f"{float(zoom) * 100:.0f}%"))
        except (AttributeError, ImportError):
            pass

    agent_host = getattr(ctx, "agent_host", None)
    if agent_host is not None and agent_host.connected:
        # Not a warning: an attached agent is an ordinary, invited state (the
        # setting that allows it says so at length), so this reads in the
        # same muted register as "workspace" and "document" rather than the
        # amber "health" chip's.
        out.append(StatusItem("agent", "Agent connected"))

    jobs = list(getattr(getattr(ctx, "cache", None), "jobs", []) or [])
    queued = sum(1 for job in jobs if job.get("status") == "queued")
    running_jobs = [job for job in jobs if job.get("status") in ("running", "processing")]
    running = len(running_jobs)
    if queued or running:
        # **What**, not just **how many**. "Queue 1 active" alone never said
        # what the active job was doing.
        #
        # The name comes off the running job's own progress label -- the
        # string the pipeline is already publishing ("Generating", "Starting
        # 3D generation") -- and *not* from a stage. The row's ``stage`` is
        # the record vocabulary (``model``), while ``create_stages.LABELS``
        # is the UI one (``mesh``); ``create_stages`` keeps those two apart
        # on purpose, so translating between them here would be the second
        # representation that module exists to avoid. Progress already holds
        # a sentence meant for a human, which is what this row wants.
        #
        # Left in its own case: lowercasing it turns "Starting 3D generation"
        # into "3d". Falls back to the old wording whenever the running job
        # publishes no label, which covers every job that is not the one the
        # worker is currently on.
        stage_text = ""
        active = running_jobs[0] if running_jobs else None
        if isinstance(active, dict):
            progress = active.get("progress")
            label = (progress or {}).get("label") if isinstance(progress, dict) else None
            if label:
                stage_text = f" ({label})"
        out.append(
            StatusItem("queue", f"Queue {running} active{stage_text} / {queued} waiting")
        )

    checks = list(getattr(getattr(ctx, "runtime", None), "checks", []) or [])
    # Downloads not made yet are not issues. Counting them put "28 issue(s)"
    # in the status bar of a fresh install with nothing wrong with it -- the
    # same false alarm the startup banner used to raise, in the one place a
    # user glances at to decide whether the app is healthy.
    failures = sum(
        1
        for check in checks
        if not getattr(check, "ok", False)
        and not getattr(check, "pending_install", False)
    )
    errors = len(getattr(ctx.state, "errors", []) or [])
    if failures or errors:
        out.append(StatusItem("health", f"{failures + errors} issue(s)", True))
    return out


def resource_item(ctx: Any) -> StatusItem | None:
    """The machine's own line, or None when it is off or unreadable.

    **Not in** :func:`items`. That list is unit-tested on its key set and is
    drawn left to right with the tail elided, so putting the meter in it would
    make it the *first* thing dropped as the window narrows -- which is
    backwards for the one item that has to be readable while a generation is
    being decided on. ``menus.draw`` right-anchors it instead (the flat status
    bar this module used to draw itself is gone -- see the module docstring),
    following ``overlay.doctor_banner``'s rule: reserve the trailing item
    before trimming the leading detail.
    """
    if not getattr(ctx.state, "show_resources", False):
        return None
    sampler = getattr(ctx, "resources", None)
    if sampler is None:
        return None
    text = sampler.reading.text()
    return StatusItem("resources", text) if text else None
