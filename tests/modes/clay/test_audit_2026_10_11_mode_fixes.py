"""Regression tests for the 2026-10-11 audit's Clay mode findings: clay-14 (an
agent minting its tab must not take the person's active tab), clay-32 (F with no
document open frames nothing later), clay-33 and clay-34 (stale docstring and an
orphaned comment) and clay-35 (the Import Mesh picker's filter follows the
supported suffixes). clay-15 is pinned in ``test_texture_link.py``.

Each test's name is the claim.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

import pygame

from realmspinner.kernels.mesh import meshimport
from realmspinner.studio.modes.clay import menutree
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

from .test_agent_clay import _Ctx
from .test_clay_mode import FakeCtx


class _DragView:
    """The slice of the viewport ``_settle_drag_on_tab_switch`` reads: a live
    drag, and a record of whether anything settled or cancelled it."""

    def __init__(self) -> None:
        self.dragging = True
        self.settled: list[Any] = []
        self.marquees = 0

    def settle_drag(self, doc: Any) -> None:
        self.settled.append(doc)
        self.dragging = False

    def cancel_marquee(self) -> None:
        self.marquees += 1


# --- clay-14 ---------------------------------------------------------------------


def test_an_agent_minting_its_tab_leaves_the_users_active_tab_active() -> None:
    ctx = _Ctx()
    user_tab = clay_mode.new_document(ctx)
    ctx.clay_view = _DragView()
    state = clay_mode.ensure(ctx)
    assert state.active_uid == user_tab.uid

    session = agent_clay.Session()
    result = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    assert result["isError"] is False, result

    assert session.tab_uid not in ("", user_tab.uid)
    assert state.active_uid == user_tab.uid, "the person's viewport and Inspector stay on theirs"
    assert state.active is user_tab
    assert ctx.clay_view.settled == [], "their live drag is not committed from under them"
    assert ctx.clay_view.dragging

    # The agent's tab is still a tab the person can reach, and the agent keeps to it.
    agent_tab = state.get(session.tab_uid)
    assert agent_tab is not None and agent_tab in state.docs
    state.activate(agent_tab.uid)
    assert state.active is agent_tab


def test_an_agent_minting_the_first_tab_of_an_empty_clay_shows_it() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    assert result["isError"] is False, result
    state = clay_mode.ensure(ctx)
    assert state.active_uid == session.tab_uid, "nothing of the person's to protect"


# --- clay-32 ---------------------------------------------------------------------


def test_f_with_no_document_open_does_not_frame_the_next_document_opened() -> None:
    ctx = FakeCtx()
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f, mod=0)
    clay_mode.handle_key(ctx, event)
    from realmspinner.kernels.mesh import document as bd

    clay_mode.adopt(ctx, bd.ClayDoc(), title="Opened")
    assert clay_mode.ensure(ctx).frame_pending is False, (
        "an F pressed over the empty state must not overwrite the camera the next file carries"
    )


def test_f_with_a_document_open_still_asks_for_a_frame() -> None:
    from realmspinner.kernels.mesh import document as bd

    ctx = FakeCtx()
    clay_mode.adopt(ctx, bd.ClayDoc(), title="Open")
    clay_mode.handle_key(ctx, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f, mod=0))
    assert clay_mode.ensure(ctx).frame_pending is True


# --- clay-33 and clay-34 ---------------------------------------------------------


def test_clay_guard_docstring_does_not_claim_the_confirm_queue_is_single_slot() -> None:
    doc = inspect.getdoc(clay_mode.guard) or ""
    assert "single pending" not in doc
    assert "silently drop" not in doc


def test_menutree_docstrings_name_no_deleted_function() -> None:
    doc = inspect.getdoc(menutree._mesh_menu) or ""
    defined = {name for name, _ in inspect.getmembers(menutree, inspect.isfunction)}
    for named in re.findall(r":func:`(\w+)`", doc):
        assert named in defined, f"_mesh_menu's docstring names {named}, which is not defined"
    assert "resolve" not in doc


def test_the_axis_view_comment_sits_with_the_import_it_describes_not_over_mutating_ctrl() -> None:
    source = inspect.getsource(clay_mode)
    marker = "_MUTATING_CTRL = "
    above = source[: source.index(marker)].rstrip().splitlines()[-12:]
    assert not any("axis view" in line.lower() for line in above), above


# --- clay-35 ---------------------------------------------------------------------


def test_import_mesh_picker_filter_matches_the_supported_suffixes() -> None:
    label, patterns = clay_mode.IMPORT_MESH_FILTER
    globs = patterns.split()
    assert tuple(g.removeprefix("*") for g in globs) == meshimport.SUPPORTED_SUFFIXES
    assert label == f"Mesh files ({patterns})"
