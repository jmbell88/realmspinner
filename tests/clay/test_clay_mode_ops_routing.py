"""Clay: two doors that used to bypass the op registry.

``clay_ops.run`` is the one place that folds whatever an op pushes into a
single undo step and names it -- every surface is supposed to funnel through
it (``clay_ops.py``'s own module docstring). Two did not:

* the Delete **key** called ``clay.selection.delete_selected`` directly, so a
  face selection spanning two objects (each ``set_mesh`` call is its own
  push) cost one press to make and two presses of Ctrl+Z to undo -- the
  2026-09-06 audit fixed the identical class of bug for the *menu's* Delete
  row and for multi-object Duplicate, and missed these two doors next door.
* Ctrl+J and the outliner's "Duplicate" row both called
  ``clay_mode._duplicate_selection``, which called
  ``clay.selection.duplicate_selected`` directly -- so the edit landed in
  history under ``add_objects``'s generic "object add" label instead of
  "Duplicate".

The 2026-09-07 audit is clay-02 and clay-07.
"""

from __future__ import annotations

from typing import Any

import pygame
import pytest

from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import elements as el
from warlock.kernels.mesh import primitives as bp
from warlock.studio import clay_mode, clay_ops


class FakeCtx:
    """Just enough of ``Ctx`` for ``clay_mode.handle_key`` and ``clay_ops.run``:
    a Clay state slot, a settings store ``adopt``'s recents wrapper reads (a
    no-op here since every test opens with no path), and a place refusals
    land."""

    def __init__(self) -> None:
        self.state = _AppState()
        self.settings = _Settings()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))


class _AppState:
    def __init__(self) -> None:
        self.clay = None


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    """``pygame.key.get_mods`` needs a video system; there is none in a test.
    Same discipline ``test_clay_mode.py`` uses, for the same reason."""
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


def _two_objects_with_faces_selected() -> tuple[bd.ClayDoc, list[int]]:
    """A document with two boxes, one face selected on each -- the shape
    ``clay-02``'s finding reproduced against."""
    doc = bd.ClayDoc()
    uids = []
    for name in ("A", "B"):
        obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box()))
        uids.append(obj.uid)
    doc.set_element_mode("face")
    for uid in uids:
        doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    return doc, uids


def _delete_event() -> Any:
    return pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE, mod=0)


def test_deleting_faces_across_two_objects_with_the_delete_key_is_one_undo_step() -> None:
    ctx = FakeCtx()
    doc, uids = _two_objects_with_faces_selected()
    clay_mode.adopt(ctx, doc, title="Scene")
    before_faces = [len(doc.by_uid(uid).mesh.starts) - 1 for uid in uids]
    depth = len(doc.history)

    assert clay_mode.handle_key(ctx, _delete_event()) is True

    for uid, before in zip(uids, before_faces, strict=True):
        assert len(doc.by_uid(uid).mesh.starts) - 1 < before, "the face was actually removed"
    assert len(doc.history) == depth + 1, "one press, one undo step"
    assert doc.undo() is True
    for uid, before in zip(uids, before_faces, strict=True):
        assert len(doc.by_uid(uid).mesh.starts) - 1 == before, (
            "one Ctrl+Z restored both objects' faces"
        )


def test_menu_delete_and_key_delete_push_the_same_number_of_steps() -> None:
    """The comparison the finding's reproduction made: the menu path (already
    through ``clay_ops.run``) and the keyboard path must agree."""
    ctx = FakeCtx()

    menu_doc, menu_uids = _two_objects_with_faces_selected()
    menu_depth = len(menu_doc.history)
    clay_ops.run(ctx, menu_doc, clay_ops.get("delete"))
    menu_steps = len(menu_doc.history) - menu_depth

    key_doc, key_uids = _two_objects_with_faces_selected()
    clay_mode.adopt(ctx, key_doc, title="Scene2")
    key_depth = len(key_doc.history)
    clay_mode.handle_key(ctx, _delete_event())
    key_steps = len(key_doc.history) - key_depth

    del menu_uids, key_uids
    assert menu_steps == key_steps == 1


def _doc_with_two_selected_objects() -> bd.ClayDoc:
    doc = bd.ClayDoc()
    for name in ("A", "B"):
        obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box()))
        doc.select(list(doc.selection) + [obj.uid])
    return doc


def _duplicate_via_ctrl_j(ctx: Any, tab: Any) -> bool:
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_j, mod=pygame.KMOD_CTRL)
    return clay_mode.handle_key(ctx, event)


def test_ctrl_j_duplicate_is_named_and_one_undo_step_not_object_add() -> None:
    """Before the fix this landed under ``add_objects``'s generic label; the
    named step is what the history panel and the finding's fix both promise."""
    ctx = FakeCtx()
    doc = _doc_with_two_selected_objects()
    clay_mode.adopt(ctx, doc, title="Scene")
    before_count = len(doc.objects)
    depth = len(doc.history)

    assert _duplicate_via_ctrl_j(ctx, None) is True

    assert len(doc.objects) == before_count + 2, "both selected objects were duplicated"
    assert len(doc.history) == depth + 1, "one press, one undo step"
    assert doc.history.top.label == "Duplicate", (
        f"expected the op's own label, got {doc.history.top.label!r}"
    )


def test_duplicate_selection_helper_routes_through_the_op_registry() -> None:
    """``clay_mode._duplicate_selection`` is what the outliner's context menu
    calls by name (``panes/clay_outliner.py``); fixing it here is what fixes
    that row without touching a file this fix does not own."""
    ctx = FakeCtx()
    doc = _doc_with_two_selected_objects()
    clay_mode.adopt(ctx, doc, title="Scene")
    before_count = len(doc.objects)
    depth = len(doc.history)

    clay_mode._duplicate_selection(ctx, None, doc)

    assert len(doc.objects) == before_count + 2
    assert len(doc.history) == depth + 1
    assert doc.history.top.label == "Duplicate"
