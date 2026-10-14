"""The 2026-10-14 audit's Library/Home findings shell-28, -29, -52, -55, -59 and poser-20.

One dated file with a name unique across the tree, per the repo's convention.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from realmspinner.studio import state as studio_state
from realmspinner.studio import widgets
from realmspinner.studio.modes.home.ui.panes import landing
from realmspinner.studio.modes.library.ui.panes import library
from realmspinner.studio.state import AppState, Filters

_SRC = Path(library.__file__).resolve().parents[5]


def _trashed(**over: Any) -> dict[str, Any]:
    row = {"id": "t1", "status": "done", "deleted_at": 1.0, "name": "chest"}
    row.update(over)
    return row


# --- shell-28: a filtered trash is not an empty one --------------------------


def test_a_filtered_trash_does_not_claim_to_be_empty(monkeypatch):
    said: list[tuple[str, str]] = []
    monkeypatch.setattr(
        widgets, "empty_state", lambda icon, title, body="": said.append((title, body))
    )
    trashed = _trashed()
    filters = Filters(trash=True, text="lantern")
    assert not filters.matches(trashed)
    ctx = SimpleNamespace(
        state=SimpleNamespace(filters=filters), cache=SimpleNamespace(jobs=[trashed])
    )

    library._empty(ctx)

    assert said and said[0][0] == "Nothing matches."


def test_a_truly_empty_trash_still_says_so(monkeypatch):
    said: list[tuple[str, str]] = []
    monkeypatch.setattr(
        widgets, "empty_state", lambda icon, title, body="": said.append((title, body))
    )
    ctx = SimpleNamespace(
        state=SimpleNamespace(filters=Filters(trash=True)), cache=SimpleNamespace(jobs=[])
    )

    library._empty(ctx)

    assert said[0][0] == "The trash is empty."


# --- shell-29: rigging by hand in Clay needs no Blender ----------------------


def test_rig_by_hand_in_clay_is_offered_without_blender():
    row = {
        "id": "m1",
        "kind": "text",
        "status": "done",
        "stage": "model",
        "files": ["model.glb"],
        "params": {},
    }

    assert "clay" in studio_state.result_actions(row, rigging_available=False)
    # The Blender-only verb stays gated: only weighting needs Blender.
    source = inspect.getsource(library._overflow)
    clay_item = source.index('"Rig by hand in Clay"')
    gate = source[source.rindex("\n        if", 0, clay_item) : clay_item]
    assert "rigging_available" not in gate


# --- shell-52: Enter honours the same gate the double-click does -------------


def test_enter_does_not_open_a_trashed_asset(monkeypatch):
    from realmspinner.studio import asset_open

    opened: list[Any] = []
    monkeypatch.setattr(asset_open, "open_asset", lambda ctx, job: opened.append(job))
    jobs = {
        "t1": _trashed(),
        "r1": {"id": "r1", "status": "running"},
        "ok": {"id": "ok", "status": "done"},
    }
    ctx = SimpleNamespace(
        state=SimpleNamespace(selected="t1"), cache=SimpleNamespace(get=lambda jid: jobs.get(jid))
    )

    library.open_selected(ctx)
    ctx.state.selected = "r1"
    library.open_selected(ctx)
    assert opened == []

    ctx.state.selected = "ok"
    library.open_selected(ctx)
    assert opened == [jobs["ok"]]


# --- shell-55: a background measurement the user never asked for is silent ---


class _TaskCtx:
    """A ctx whose ``submit`` runs inline and counts what ``TaskRunner`` would
    have reported as a failure (each one is an uncoalesced error toast)."""

    def __init__(self) -> None:
        self.state = AppState()
        self.svc = SimpleNamespace()
        self.submits = 0
        self.failures = 0

    def submit(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> bool:
        kwargs.pop("tag", None)
        self.submits += 1
        try:
            fn(*args, **kwargs)
        except Exception:
            self.failures += 1
        return True


def test_a_failed_trash_measurement_does_not_toast_every_frame(monkeypatch):
    def boom(_svc):
        raise OSError("the store read failed")

    monkeypatch.setattr(library.svc_jobs, "trash_size", boom)
    ctx = _TaskCtx()
    rows = [{"id": "a"}]

    for _ in range(30):
        assert library.measure_trash(ctx, rows) is None

    assert ctx.failures == 0  # a failure is swallowed, so no toast is raised
    assert ctx.submits <= 1  # and it is not re-asked every frame


def test_a_failed_unreviewed_count_does_not_toast(monkeypatch):

    def boom(**_kw):
        raise OSError("the store read failed")

    svc = SimpleNamespace(store=SimpleNamespace(unverdicted_models=boom))

    assert landing._count_unreviewed(svc) is None


# --- shell-59 and poser-20: nothing names what was removed -------------------

_REMOVED = re.compile(r"Poser|Troupe|poser_mode|poser_send|pose library", re.IGNORECASE)


def test_no_comment_in_the_library_or_home_panes_names_a_removed_mode():
    for path in (
        _SRC / "studio/modes/library/ui/panes/library.py",
        _SRC / "studio/modes/home/ui/panes/landing.py",
        _SRC / "studio/panes/inspector.py",
        _SRC / "studio/state.py",
    ):
        text = path.read_text(encoding="utf-8")
        hits = [m.group(0) for m in _REMOVED.finditer(text)]
        assert not hits, (path.name, hits)


def test_the_delete_everything_dialog_names_only_things_the_app_still_has():
    source = inspect.getsource(library)
    start = source.index('title="Delete every asset?"')
    dialog = source[start : start + 900]
    assert "pose library" not in dialog.lower()
