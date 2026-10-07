"""Regressions for the 2026-09-23 audit's clay-03: a document past the reopen
ceiling can still be saved.

The harness is a trimmed copy of ``tests/modes/clay/test_clay_generate.py``'s
own ``_Ctx`` (duplicating a few lines is cheaper than coupling to another
module).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize
from realmspinner.service.errors import TooLarge
from realmspinner.service.files import MAX_CLAY_SOURCE_BYTES
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.state import DEFAULT_FORM_3D, default_form_2d
from realmspinner.studio.tasks import Done

# --- shared harness (test_clay_generate.py's own _Ctx shape) -----------------


class _Store:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}

    def get(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)


class _Svc:
    def __init__(self, root: Path) -> None:
        self.store = _Store()
        self.root = root
        self.config = None

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id


class _Confirms:
    def __init__(self) -> None:
        self.pending: Any = None

    def ask(self, confirm: Any) -> None:
        self.pending = confirm


class _AppState:
    def __init__(self) -> None:
        self.clay = None
        self.mode = "home"
        self.form_2d = default_form_2d()
        self.form_3d = dict(DEFAULT_FORM_3D)


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _Ctx:
    def __init__(self, tmp_path: Path) -> None:
        self.svc = _Svc(tmp_path)
        self.state = _AppState()
        self.settings = _Settings()
        self.toasts: list[tuple[str, str]] = []
        self.confirms = _Confirms()
        self.submitted: list[str] = []
        self._busy: set[str] = set()
        self._queue: list[Done] = []
        self.clay_view = None

    def toast(self, message: str, kind: str = "info", action: Any = None) -> None:
        self.toasts.append((message, kind))

    def busy(self, key: str) -> bool:
        return key in self._busy

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        if key in self._busy:
            return False
        self.submitted.append(key)
        try:
            done = Done(key=key, result=fn(*args, **kwargs), tag=tag)
        except Exception as exc:  # noqa: BLE001 - the same failure a real pool reports
            done = Done(key=key, error=exc, message=str(exc), tag=tag)
        self._busy.add(key)
        self._queue.append(done)
        return True

    def land_all(self) -> None:
        while self._queue:
            done = self._queue.pop(0)
            self._busy.discard(done.key)
            if done.ok:
                clay_mode.on_task_done(self, done)
            else:
                clay_mode.on_task_failed(self, done)


def _tab(ctx: Any) -> Any:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    return clay_mode.adopt(ctx, doc, title="Scene")


# --- clay-03: the save doors have no ceiling ----------------------------------


def test_a_clay_document_past_the_reopen_ceiling_is_refused_at_save(tmp_path, monkeypatch):
    """``_load`` refuses anything over ``MAX_CLAY_SOURCE_BYTES``; before this
    fix ``save_to`` wrote past it with no check at all, producing a file the
    app would then refuse to reopen."""
    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    oversized = b"0" * (MAX_CLAY_SOURCE_BYTES + 1)
    monkeypatch.setattr(serialize, "snapshot_bytes", lambda snap: oversized)

    path = tmp_path / "too-big.rblk"
    clay_mode.save_to(ctx, tab, path)
    ctx.land_all()

    assert not path.exists(), "an oversized document must not be written at all"
    assert tab.saving is False, "a refused save must not leave the tab locked"


def test_a_clay_document_past_the_reopen_ceiling_is_refused_at_save_as(tmp_path, monkeypatch):
    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    oversized = b"0" * (MAX_CLAY_SOURCE_BYTES + 1)
    monkeypatch.setattr(serialize, "snapshot_bytes", lambda snap: oversized)
    path = tmp_path / "too-big-as.rblk"
    monkeypatch.setattr(
        "realmspinner.studio.dialogs.save_file", lambda *a, **kw: path
    )

    clay_mode.save_as(ctx, tab)
    ctx.land_all()

    assert not path.exists()
    assert tab.saving is False


def test_the_save_ceiling_check_raises_too_large_with_a_field(tmp_path):
    """Refusals raise ``service.errors`` exceptions carrying a ``field`` --
    the app-wide contract every other refusal in this codebase follows."""
    with pytest.raises(TooLarge) as excinfo:
        clay_mode._refuse_oversized_save(b"0" * (MAX_CLAY_SOURCE_BYTES + 1))
    assert excinfo.value.field == "save"
