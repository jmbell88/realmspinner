"""Three small Clay items left behind by earlier fixers on 2026-10-03.

1. A non-finite number pasted into a generator field raised on the frame thread
   (``clamp_params`` began refusing it, and the call sat outside the ``try``).
2. A save past the object/triangle ceilings surfaced as "Something went wrong;
   see the log", because ``snapshot_bytes`` raises a plain ``ValueError`` that
   the task runner does not treat as a message for a person.
3. Four callers still named N copies through a list they appended to, paying
   the copy and the probe that ``mesh_ops.UsedNames`` removes.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import glbimport, merge, selection
from realmspinner.kernels.mesh import ops as mesh_ops
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.service.errors import TooLarge
from realmspinner.studio import tasks as studio_tasks
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.ui.panes import props as clay_props

from .test_clay_mode import FakeCtx, _Done, _tab


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


class _Ctx:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []
        self.inline = True

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _add(doc: bd.ClayDoc, name: str, **kw: Any) -> bd.Obj:
    return doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box(), **kw))


# --- item 1: a pasted inf/nan in a float field is refused, not raised --------


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_float_pasted_into_a_generator_field_is_refused_without_raising(
    monkeypatch, ui, bad
) -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Cyl",
            mesh=bp.cylinder(),
            generator="cylinder",
            params={"radius": 0.5, "height": 1.0, "segments": 16},
        )
    )
    mesh_before, params_before = obj.mesh, dict(obj.params)
    head_before = doc.history.head

    def fake_widget(key, value, default):
        return (bad, True) if key == "radius" else (value, False)

    monkeypatch.setattr(clay_props, "_widget", fake_widget)

    ui.new_frame()
    ui.begin("##host")
    try:
        clay_props._generator(doc, doc.by_uid(obj.uid))  # must not raise
    finally:
        ui.end()
        ui.end_frame()

    after = doc.by_uid(obj.uid)
    assert after.mesh is mesh_before
    assert dict(after.params) == params_before
    assert doc.history.head == head_before, "a refused edit is not an undo step"
    assert np.isfinite(after.mesh.positions).all()


# --- item 2: the save refusal names its cause --------------------------------


def _two_object_tab(ctx: FakeCtx):
    tab = _tab(ctx)
    tab.doc.add_object(bd.Obj(uid=bd.new_uid(), name="Second", mesh=bp.box()))
    return tab


def test_save_to_past_the_object_ceiling_is_a_toastable_refusal_not_a_bare_valueerror(
    monkeypatch, svc, tmp_path
) -> None:
    ctx = FakeCtx(svc)
    tab = _two_object_tab(ctx)
    monkeypatch.setattr(glbimport, "MAX_OBJECTS", 1)
    target = tmp_path / "scene.rblk"

    with pytest.raises(TooLarge) as caught:
        clay_mode.save_to(ctx, tab, target)

    assert caught.value.field == "save"
    assert "2 objects" in caught.value.message
    # What the task runner shows a person is the message, not "see the log".
    assert isinstance(caught.value, studio_tasks.CARRIES_ITS_OWN_MESSAGE)
    assert not target.exists(), "nothing half-written"
    # ...and the tab is released by the failure path, the document untouched.
    clay_mode.on_task_failed(ctx, _Done(f"clay-save:{tab.uid}"))
    assert tab.saving is False
    assert len(tab.doc.objects) == 2


def test_save_as_past_the_triangle_ceiling_is_a_toastable_refusal_not_a_bare_valueerror(
    monkeypatch, svc, tmp_path
) -> None:
    ctx = FakeCtx(svc)
    tab = _two_object_tab(ctx)  # two boxes, 12 triangles each
    monkeypatch.setattr(glbimport, "MAX_TRIANGLES", 12)
    target = tmp_path / "scene.rblk"
    monkeypatch.setattr(clay_mode.dialogs, "save_file", lambda *a, **k: target)

    with pytest.raises(TooLarge) as caught:
        clay_mode.save_as(ctx, tab)

    assert caught.value.field == "save"
    assert "triangles" in caught.value.message
    assert not target.exists()
    clay_mode.on_task_failed(ctx, _Done(f"clay-saveas:{tab.uid}"))
    assert tab.saving is False


def test_a_save_at_the_ceiling_still_writes(monkeypatch, svc, tmp_path) -> None:
    ctx = FakeCtx(svc)
    tab = _two_object_tab(ctx)
    monkeypatch.setattr(glbimport, "MAX_OBJECTS", 2)
    target = tmp_path / "scene.rblk"
    clay_mode.save_to(ctx, tab, target)
    assert target.exists()


# --- item 3: one UsedNames per naming loop ------------------------------------


@pytest.fixture
def spy(monkeypatch):
    """Record the type of every ``taken`` handed to ``next_name``."""
    seen: list[type] = []
    real = mesh_ops.next_name

    def wrapper(name, taken=()):
        seen.append(type(taken))
        return real(name, taken)

    monkeypatch.setattr(mesh_ops, "next_name", wrapper)
    return seen


# The unpatched function, for the oracles: the spy must see only the callers'
# calls, never the test's own.
_REAL_NEXT_NAME = mesh_ops.next_name


def _old_names(existing: list[str], sources: list[str]) -> list[str]:
    """What the list-and-append callers produced: the oracle."""
    taken = list(existing)
    out = []
    for source in sources:
        name = _REAL_NEXT_NAME(source, taken)
        taken.append(name)
        out.append(name)
    return out


_EXISTING = ["Post", "Post.001", "Post.004", "Rail", "Rail.002", "Other"]


def _kit() -> tuple[bd.ClayDoc, list[bd.Obj]]:
    doc = bd.ClayDoc()
    objs = [_add(doc, name) for name in _EXISTING]
    return doc, objs


def test_array_linear_names_its_copies_through_one_used_names_and_keeps_the_names(spy) -> None:
    doc, objs = _kit()
    doc.select([objs[0].uid, objs[3].uid])  # Post, Rail
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("array-linear"), count=5, x=1.0) is True
    made = [o.name for o in doc.objects[len(_EXISTING):]]
    assert made == _old_names(_EXISTING, ["Post", "Rail"] * 4)
    assert spy and all(t is mesh_ops.UsedNames for t in spy), set(spy)


def test_array_radial_names_its_copies_through_one_used_names_and_keeps_the_names(spy) -> None:
    doc, objs = _kit()
    doc.select([objs[0].uid, objs[3].uid])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("array-radial"), count=4) is True
    made = [o.name for o in doc.objects[len(_EXISTING):]]
    assert made == _old_names(_EXISTING, ["Post", "Rail"] * 3)
    assert spy and all(t is mesh_ops.UsedNames for t in spy), set(spy)


def test_mirror_copy_names_its_copies_through_one_used_names_and_keeps_the_names(spy) -> None:
    doc, objs = _kit()
    doc.select([objs[0].uid, objs[1].uid, objs[3].uid])  # Post, Post.001, Rail
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("mirror-copy"), axis=0.0) is True
    made = [o.name for o in doc.objects[len(_EXISTING):]]
    assert made == _old_names(_EXISTING, ["Post", "Post.001", "Rail"])
    assert spy and all(t is mesh_ops.UsedNames for t in spy), set(spy)


def test_duplicate_selected_names_its_copies_through_one_used_names_and_keeps_the_names(
    spy,
) -> None:
    doc, objs = _kit()
    doc.select([objs[3].uid, objs[0].uid, objs[2].uid])  # selection order is not doc order
    new = selection.duplicate_selected(doc)
    assert len(new) == 3
    made = [doc.by_uid(u).name for u in new]
    assert made == _old_names(_EXISTING, ["Post", "Post.004", "Rail"])
    assert spy and all(t is mesh_ops.UsedNames for t in spy), set(spy)


def test_separate_names_its_pieces_through_one_used_names_and_keeps_the_names(spy) -> None:
    doc, objs = _kit()
    source = _add(doc, "Rail")
    pieces = [bp.box() for _ in range(5)]
    new = doc.separate(source.uid, pieces)
    expected = _old_names(_EXISTING + ["Rail"], ["Rail"] * 5)
    assert [o.name for o in new] == expected
    assert spy and all(t is mesh_ops.UsedNames for t in spy), set(spy)


def test_merge_into_names_arrivals_through_one_used_names_and_keeps_the_names(spy) -> None:
    target = bd.ClayDoc()
    for name in ["Rock", "Rock.002", "Tree"]:
        _add(target, name)
    incoming = bd.ClayDoc()
    arriving = ["Rock", "Rock", "Fern", "Tree", "Rock", "Rock.002"]
    for name in arriving:
        _add(incoming, name)

    added = merge.merge_into(target, incoming, offset=np.zeros(3, dtype="f8"))

    taken = {"Rock", "Rock.002", "Tree"}
    expected = []
    for name in arriving:
        if name in taken:
            name = _REAL_NEXT_NAME(name, taken)
        taken.add(name)
        expected.append(name)
    assert [o.name for o in added] == expected
    assert spy and all(t is mesh_ops.UsedNames for t in spy), set(spy)


def _array_seconds(copies: int, monkeypatch) -> float:
    monkeypatch.setattr(clay_ops, "MAX_ARRAY_COPIES", 10**6)
    best = float("inf")
    for _ in range(2):
        doc = bd.ClayDoc()
        post = _add(doc, "Post")
        doc.select([post.uid])
        started = time.perf_counter()
        # The op body directly: ``run`` would clamp ``count`` to the catalogue's
        # own ``MAX_ARRAY_COUNT`` before it got here.
        assert clay_ops._array_linear(None, doc, count=copies + 1, x=0.1)
        best = min(best, time.perf_counter() - started)
        assert len({o.name for o in doc.objects}) == copies + 1
    return best


def test_naming_an_arrays_copies_does_not_grow_quadratically(monkeypatch) -> None:
    # ``MAX_ARRAY_COPIES`` is lifted here only so the cost can be measured over
    # a span where a quadratic term shows; one real press stays under 2,000.
    small, large = 1500, 6000
    ratio = _array_seconds(large, monkeypatch) / max(_array_seconds(small, monkeypatch), 1e-6)
    # 4x the copies: linear is ~4x, a list copied and probed per copy is ~16x.
    assert ratio < 9.0, f"an array of {large} copies cost {ratio:.1f}x one of {small}"
