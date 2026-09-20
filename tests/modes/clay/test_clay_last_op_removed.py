"""Clay: ``last_op`` bookkeeping, removed rather than built into a control.

The 2026-09-07 audit's clay-10: ``ClayState.last_op`` was written by every
``clay_ops.run`` call (through the now-removed ``_remember``) and read by no
pane -- the "adjust last operation" card and Repeat it was recorded for do not
exist in the shipped UI. Removed rather than kept on the chance a future card
wants it; a card that does can rebuild the record from the undo stack it
would need anyway.

No sibling test module exists in ``tests/modes/clay/`` for ``studio/modes/clay/ops.py``,
``studio/modes/clay/mode.py`` or ``studio/modes/clay/state.py`` (all three live directly under
``studio/``, not under ``studio/clay/``, and their usual homes --
``tests/modes/clay/test_clay_ops.py``, ``tests/modes/clay/test_clay_mode.py`` -- sit outside this
fix's allowed test directory), so this is a new file.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pygame
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay import state as clay_state


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _Ctx:
    def __init__(self, state: clay_state.ClayState) -> None:
        self.state = type("S", (), {"clay": state})()
        self.settings = _Settings()

    def toast(self, message: str, kind: str = "info") -> None:
        pass


def test_lastop_no_longer_exists() -> None:
    assert not hasattr(clay_state, "LastOp")


def test_claystate_carries_no_last_op_field() -> None:
    fields = {f.name for f in dataclasses.fields(clay_state.ClayState)}
    assert "last_op" not in fields


def test_running_an_op_leaves_no_last_op_attribute_behind() -> None:
    """A partial removal that left ``_remember`` calling ``setattr`` on a
    plain instance would silently reintroduce the attribute; this is what
    would catch it."""
    state = clay_state.ClayState()
    ctx = _Ctx(state)
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    doc.select([obj.uid])

    assert clay_ops.run(ctx, doc, clay_ops.get("duplicate")) is True
    assert not hasattr(state, "last_op")


def test_step_history_still_moves_the_document_with_the_bookkeeping_gone() -> None:
    """``clay_mode.step_history`` used to also clear ``ClayState.last_op``;
    with that field gone it must still do the one thing its callers need."""
    state = clay_state.ClayState()
    ctx = _Ctx(state)
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    tab = clay_mode.adopt(ctx, doc, title="Scene")
    after_add = len(doc.history)
    doc.set_props(obj.uid, name="Renamed")

    # Back to just after the object was added, not to an empty document --
    # index 0 would also undo ``add_object`` itself, which is not what this
    # is testing.
    assert clay_mode.step_history(ctx, tab, after_add) is True
    assert doc.by_uid(obj.uid).name == "Box"
