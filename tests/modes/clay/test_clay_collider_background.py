"""P60: the hull-backed collider fits (Convex Hull, Compound, Box (Oriented))
run under a ``clay-bg:<tab uid>`` task in decimate's own shape -- snapshot on
the frame thread, fit on a task thread, apply only to the mesh that was read --
while the agent's inline ctx still runs them synchronously.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.kernels.mesh import colliders as colliders_mod
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay import state as clay_state


class _Ctx:
    def __init__(self) -> None:
        self.toasted: list[tuple[str, str]] = []
        self.submitted: list[tuple[str, Any, tuple, dict]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasted.append((message, level))

    def submit(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> bool:
        self.submitted.append((key, fn, args, kwargs))
        return True


class _Done:
    def __init__(self, key: str, result: Any) -> None:
        self.key = key
        self.result = result


def _interactive() -> tuple[_Ctx, bd.ClayDoc, int, clay_state.ClayTab]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Crate", mesh=bp.box()))
    doc.select([obj.uid])
    state = clay_state.ClayState()
    tab = clay_state.ClayTab(doc=doc, title="Scene")
    state.add(tab)
    ctx = _Ctx()
    ctx.state = SimpleNamespace(clay=state)
    return ctx, doc, obj.uid, tab


def _spy_kind(monkeypatch: pytest.MonkeyPatch, kind: str) -> list[str]:
    """Record which thread each fit of *kind* ran on."""
    import threading

    threads: list[str] = []
    label, real, defaults = colliders_mod.COLLIDER_KINDS[kind]

    def spy(mesh: object, **kwargs: object) -> object:
        threads.append(threading.current_thread().name)
        return real(mesh, **kwargs)

    monkeypatch.setitem(colliders_mod.COLLIDER_KINDS, kind, (label, spy, defaults))
    return threads


HULL_RUNS = (
    ("collider-convex", {}),
    ("collider-compound", {}),
    ("collider-box", {"oriented": 1.0}),
)


@pytest.mark.parametrize(("name", "params"), HULL_RUNS)
def test_submitting_a_hull_fit_returns_without_running_the_hull_on_the_calling_thread(
    monkeypatch: pytest.MonkeyPatch, name: str, params: dict[str, float]
) -> None:
    kind = name.removeprefix("collider-")
    ran = _spy_kind(monkeypatch, kind)
    ctx, doc, uid, tab = _interactive()
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get(name), **params) is True

    assert ran == [], "the fit must not run on the frame thread"
    assert len(ctx.submitted) == 1
    key, fn, args, kwargs = ctx.submitted[0]
    assert key == f"clay-bg:{tab.uid}"
    assert tab.bg_busy
    assert len(doc.history) == depth, "a submit is not a result"

    fn(*args, **kwargs)  # the task-thread half
    assert len(ran) == 1


def test_a_stale_mesh_stamp_drops_the_collider_fit_result() -> None:
    ctx, doc, uid, tab = _interactive()
    assert clay_ops.run(ctx, doc, clay_ops.get("collider-convex")) is True
    key, fn, args, kwargs = ctx.submitted[0]
    result = fn(*args, **kwargs)

    edited = bp.cone()
    doc.set_mesh(uid, edited)  # the user edits before the result lands
    depth = len(doc.history)
    count = len(doc.objects)

    clay_mode.on_task_done(ctx, _Done(key, result))

    assert tab.bg_busy == ""
    assert len(doc.objects) == count, "no collider was added for a mesh that moved on"
    assert len(doc.history) == depth
    assert doc.by_uid(uid).mesh is edited
    assert any("changed while" in m for m, _ in ctx.toasted), ctx.toasted


def test_a_hull_fit_result_for_an_unchanged_mesh_adds_one_collider_in_one_step() -> None:
    ctx, doc, uid, tab = _interactive()
    assert clay_ops.run(ctx, doc, clay_ops.get("collider-convex")) is True
    key, fn, args, kwargs = ctx.submitted[0]
    result = fn(*args, **kwargs)
    depth = len(doc.history)

    clay_mode.on_task_done(ctx, _Done(key, result))

    assert tab.bg_busy == ""
    assert len(doc.history) == depth + 1
    children = [o for o in doc.objects if o.role == "collider"]
    assert len(children) == 1 and children[0].parent == uid


def test_a_deleted_source_drops_the_collider_fit_result() -> None:
    ctx, doc, uid, tab = _interactive()
    assert clay_ops.run(ctx, doc, clay_ops.get("collider-convex")) is True
    key, fn, args, kwargs = ctx.submitted[0]
    result = fn(*args, **kwargs)
    doc.remove_object(uid)

    clay_mode.on_task_done(ctx, _Done(key, result))

    assert not [o for o in doc.objects if o.role == "collider"]


def test_the_agent_inline_path_still_fits_and_applies_synchronously(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    real = clay_ops._collider_work
    monkeypatch.setattr(
        clay_ops, "_collider_work", lambda *a, **k: calls.append("work") or real(*a, **k)
    )
    ctx, doc, uid, tab = _interactive()
    ctx.inline = True
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("collider-convex")) is True

    assert calls == ["work"]
    assert ctx.submitted == [], "inline: nothing crosses to a task thread"
    assert len(doc.history) == depth + 1
    assert [o for o in doc.objects if o.role == "collider"]


def test_the_agent_collider_tool_fits_inline_on_the_calling_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``clay_collider`` follows decimate's agent rule -- inline -- and does
    not grow a second one: the hull runs on the calling thread and the
    collider is there when the call returns."""
    import threading

    from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

    from .test_agent_clay import _Ctx as AgentCtx
    from .test_agent_clay import _new_agent_tab

    threads = _spy_kind(monkeypatch, "convex")
    ctx = AgentCtx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)

    result = agent_clay.call(ctx, session, "clay_collider", {"uids": [uid], "kind": "convex"})

    assert result["isError"] is False, result
    assert threads == [threading.current_thread().name]
    doc = clay_mode.ensure(ctx).get(session.tab_uid).doc
    assert [o for o in doc.objects if o.role == "collider"]
