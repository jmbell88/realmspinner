"""Run a spec-speaking child process and return its result JSON.

Split out of the former ``src/realmspinner/rigging.py`` (P4 of
``dev/RESTRUCTURE.md``): ``run_worker`` spawns and supervises a subprocess
(``winjob.assign``, a deadline, a drain thread) rather than deciding anything
about a rig or a pose, so it is process control -- Layer 2, not the Layer 1
``kernels/rig/`` package the rest of the old file split into.

``module``/``marker``/``name`` already generalised this past Blender before
this move: ``pipelines/lora_train.py`` and ``pipelines/separation_worker.py``
both describe themselves as running "under ``blender_run.run_worker``", and
``pipelines/blender_worker.py`` is only the first and most literal caller.
Renamed for what it actually is now that it no longer lives inside a module
named for one of its three callers.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import winjob

log = logging.getLogger(__name__)

# Progress lines the worker prints. Anything else on its stdout is log noise.
PROGRESS_PREFIX = "[blender]"
RE_PROGRESS = re.compile(r"^\[blender\]\s+([\d.]+)\s+(.*)$")

# Rigging a 300k-face mesh with automatic weights is minutes of CPU, not hours.
BLENDER_TIMEOUT = 1800.0

# Tail of the worker's output kept for the error message when it fails.
ERROR_TAIL_CHARS = 2000


class BlenderError(RuntimeError):
    """The worker exited non-zero, timed out, or produced no result file."""


def _terminate_worker(proc: subprocess.Popen[str], timeout: float = 10.0) -> None:
    """Kill and reap a Blender child without ever introducing another hang."""
    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=timeout)


def run_worker(
    spec: dict[str, Any],
    *,
    on_progress: Callable[[float, str], None] | None = None,
    on_start: Callable[[subprocess.Popen[str]], None] | None = None,
    timeout: float = BLENDER_TIMEOUT,
    module: str = "realmspinner.pipelines.blender_worker",
    marker: str = "blender",
    name: str = "Blender worker",
) -> dict[str, Any]:
    """Run one Blender operation out-of-process and return its result JSON.

    ``module``/``marker``/``name`` generalise the contract to any child that
    speaks it -- the LoRA trainer (``pipelines.lora_train_worker``) is the
    second -- without a second copy of the deadline, the drain threads and
    the staged-result rules below. The defaults keep every Blender caller
    exactly what it was.

    Synchronous and blocking -- every caller in the app dispatches it through
    ``asyncio.to_thread``, like every other multi-second call in this codebase.

    ``spec`` is handed over on stdin; the worker writes its result to
    ``spec["result_path"]`` rather than stdout, so a stray print from bpy (and
    it does print) can never corrupt the payload.

    ``on_start`` receives the live ``Popen`` so a cancel can kill it. Like
    trellis-server, there is no polite abort: bpy is inside a C weighting solve
    and checks nothing, so killing the process is the only thing that stops it.
    """
    result_path = Path(spec["result_path"])
    # The worker stages its result beside the served name and renames it in, so
    # the same three cleanup sites that clear a stale result have to clear the
    # staging file too -- a worker killed mid-write leaves the .tmp, not the
    # result, and it would otherwise sit in the source job's directory forever.
    result_tmp = result_path.with_name(result_path.name + ".tmp")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.unlink(missing_ok=True)
    result_tmp.unlink(missing_ok=True)

    re_progress = (
        RE_PROGRESS
        if marker == "blender"
        else re.compile(rf"^\[{re.escape(marker)}\]\s+([\d.]+)\s+(.*)$")
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", module],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    # Same kill-on-close job as trellis-server: a bpy solve holds multiple GB,
    # and a parent that dies mid-rig used to leave it running indefinitely.
    winjob.assign(proc.pid)
    winjob.track(proc.pid, name.lower())
    tail: list[str] = []
    if on_start is not None:
        try:
            on_start(proc)
        except BaseException:
            _terminate_worker(proc)
            winjob.untrack(proc.pid)
            raise
    # stdout is drained on a helper thread so the *whole* run has a deadline, not
    # just the wait() after EOF: a bpy process that hangs mid-solve (or mid-render)
    # produces no further output and never closes stdout, and reading it inline
    # would block past any timeout and wedge the serial job queue forever.
    lines: queue.Queue[str | None] = queue.Queue()

    def _pump(stream: Any) -> None:
        try:
            for raw in stream:
                lines.put(raw)
        finally:
            lines.put(None)

    assert proc.stdin is not None and proc.stdout is not None
    reader = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
    reader.start()

    # The spec goes out on a thread of its own for the same reason stdout is
    # drained on one: it has to be inside the deadline. A sheet spec carries
    # every cell's bones inline, and a 200-cell clip is far larger than the OS
    # pipe buffer -- so a worker that dies before draining stdin (a failed
    # `import bpy`, a driver that will not load) made this write block
    # indefinitely, wedging the serial GPU queue well past `timeout`. The
    # BrokenPipeError the other outcome raises is swallowed here too: the exit
    # code and the captured tail below say far more about what went wrong.
    def _send() -> None:
        try:
            proc.stdin.write(json.dumps(spec))
            proc.stdin.close()
        except OSError:
            pass

    writer = threading.Thread(target=_send, daemon=True)
    writer.start()

    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # The deadline can elapse in the same queue-poll tick the
                # child's stdout closes -- drain whatever is already queued
                # (the EOF sentinel, or a final line) before calling this a
                # timeout, or a job that legitimately finished gets its
                # result deleted and a spurious BlenderError raised.
                try:
                    raw = lines.get_nowait()
                except queue.Empty:
                    raise subprocess.TimeoutExpired(proc.args, timeout) from None
            else:
                try:
                    raw = lines.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    continue
            if raw is None:
                break
            line = raw.rstrip()
            m = re_progress.match(line)
            if m and on_progress is not None:
                on_progress(float(m.group(1)), m.group(2))
            tail.append(line)
            if len(tail) > 200:
                del tail[:100]
        # A small floor on the final wait: the worker has already signalled
        # done (EOF/sentinel seen above), but its exit may still be
        # finalising -- deadline - now can be ~0 here and give the OS no
        # grace to reap it, re-raising the very timeout we just avoided.
        code = proc.wait(timeout=max(deadline - time.monotonic(), 1.0))
    except subprocess.TimeoutExpired:
        _terminate_worker(proc)
        result_path.unlink(missing_ok=True)
        result_tmp.unlink(missing_ok=True)
        raise BlenderError(f"{name} timed out after {timeout:.0f}s") from None
    finally:
        if proc.poll() is None:
            _terminate_worker(proc)
        # Every exit path leaves the child dead or killed, so the registry
        # entry comes out with it -- winjob's contract is that entries are
        # removed on reap, or terminate_tracked later opens a recycled pid
        # with PROCESS_TERMINATE and can kill an unrelated process.
        winjob.untrack(proc.pid)

    output = "\n".join(tail)[-ERROR_TAIL_CHARS:]
    if code != 0:
        # A killed-late worker may still have written the handoff file; it
        # would otherwise sit in the source job's directory until the next run.
        result_path.unlink(missing_ok=True)
        result_tmp.unlink(missing_ok=True)
        raise BlenderError(f"{name} exited with code {code}:\n{output}")
    if not result_path.exists():
        raise BlenderError(f"{name} wrote no result:\n{output}")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        # An exit 0 with an unreadable result is the one way this could still
        # raise a raw decoder error at the caller. Typed like every other way
        # the worker can disappoint, and carrying the worker's own tail.
        result_path.unlink(missing_ok=True)
        raise BlenderError(f"{name} wrote an unreadable result:\n{output}") from exc
    # The handoff file has served its purpose; a rig writes it into the source
    # job's directory, where it would otherwise sit next to model.glb forever.
    result_path.unlink(missing_ok=True)
    result_tmp.unlink(missing_ok=True)
    return payload
