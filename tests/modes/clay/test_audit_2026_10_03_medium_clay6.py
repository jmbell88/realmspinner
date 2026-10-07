"""The 2026-10-03 audit's Medium Clay findings, batch 6 (clay-58, 59): a
non-finite stored grid size and the mesh-file export's encode thread."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import dialogs
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.state import ClayState
from realmspinner.studio.tasks import Done

WORKER = "realmspinner-task-test"


class _Ctx:
    """``submit`` runs the task on a real, joined worker thread -- so a spy on
    the encode can name the thread it ran on -- and queues its ``Done``."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None, mode="home")
        self.settings = SimpleNamespace(
            get=lambda key: None, set=lambda key, value: None
        )
        self.toasts: list[tuple[str, str]] = []
        self.clay_view = None
        self._busy: set[str] = set()
        self._queue: list[Done] = []

    def toast(self, message: str, kind: str = "info", action: Any = None) -> None:
        self.toasts.append((message, kind))

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        if key in self._busy:
            return False
        box: dict[str, Any] = {}

        def go() -> None:
            try:
                box["result"] = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                box["error"] = exc

        worker = threading.Thread(target=go, name=WORKER)
        worker.start()
        worker.join()
        done = (
            Done(key=key, error=box["error"], message=str(box["error"]), tag=tag)
            if "error" in box
            else Done(key=key, result=box.get("result"), tag=tag)
        )
        self._busy.add(key)
        self._queue.append(done)
        return True


def _tab(ctx: Any) -> Any:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    return clay_mode.adopt(ctx, doc, title="Scene")


# --- clay-58: a non-finite stored grid size ---------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 1e999])
def test_a_non_finite_stored_grid_size_is_ignored_not_raised(bad):
    state = ClayState()
    default = state.grid_size
    clay_mode._restore_view(state, {"grid_size": bad, "grid": False})
    assert state.grid_size == default
    assert state.grid is False  # the rest of the block still restores


def test_a_finite_stored_grid_size_is_still_clamped():
    state = ClayState()
    clay_mode._restore_view(state, {"grid_size": 5000})
    assert state.grid_size == 1000.0


# --- clay-59: the mesh-file export's encode ---------------------------------------


def _spy(monkeypatch: pytest.MonkeyPatch, owner: Any, name: str) -> list[str]:
    threads: list[str] = []
    real = getattr(owner, name)

    def spy(*args: Any, **kwargs: Any) -> Any:
        threads.append(threading.current_thread().name)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, spy)
    return threads


def test_clay_export_mesh_file_encodes_off_the_frame_thread(tmp_path, monkeypatch):
    from realmspinner.kernels.geom3d import glbwrite
    from realmspinner.kernels.mesh import objexport

    ctx = _Ctx()
    tab = _tab(ctx)
    glb_threads = _spy(monkeypatch, glbwrite, "write_glb")
    obj_threads = _spy(monkeypatch, objexport, "claydoc_to_obj")
    model_threads = _spy(monkeypatch, bd, "to_model")
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "out.glb")

    clay_mode.export_mesh_file(ctx, tab, "glb")
    assert glb_threads == ["realmspinner-task-test"]
    assert model_threads == ["realmspinner-task-test"]
    assert (tmp_path / "out.glb").is_file()

    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "out.obj")
    ctx._busy.clear()
    tab.saving = False
    clay_mode.export_mesh_file(ctx, tab, "obj")
    assert obj_threads == ["realmspinner-task-test"]
    assert (tmp_path / "out.obj").is_file()
