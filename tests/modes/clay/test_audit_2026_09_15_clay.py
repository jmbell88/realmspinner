"""Regression tests for the 2026-09-15 audit's Clay findings.

Two unrelated defects, one file because one fixer owned both: clay-01
(a live keyboard/gizmo drag survives a tab switch and settles against the
wrong document) and clay-04 (the copy-family reader mis-reads a four-digit
suffix).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from realmspinner.kernels.mesh import diagnose
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.ui import view as clay_view

RECT = (0.0, 0.0, 128.0, 96.0)


# --- clay-01: a tab switch mid-drag ------------------------------------------


def test_switching_the_active_tab_via_the_tab_bar_commits_or_cancels_a_live_drag(gl) -> None:
    """``ClayState.activate`` -- what ``docmodes.tab_bar``'s click handler,
    ``cycle`` (Ctrl+Tab) and both ``clay-open`` dedupe paths all call to
    switch tabs -- used to clear only its own ``drag_axis``/``ref``
    bookkeeping. A live G/R/S drag has already written TRS onto the object in
    place by the time any of those run, and nothing recorded that: the
    drag's eventual commit or cancel (a release, an Esc) then ran against
    whichever document had *become* active, not the one the drag began on.
    ``close_tab``'s ``release`` already cancelled a live drag first for the
    tab-*closing* case; this is the tab-*switching* half.

    Drives ``ClayState.activate`` directly against a real, headless
    ``ClayView`` -- the same shape ``tests/modes/clay/test_clay_view.py`` already
    drives keyboard drags through -- rather than a fake, so the actual
    ``settle_drag`` wiring (``clay_mode.ensure``) and commit path
    (``_view_drag.DragOps.settle_drag``) are what is under test.

    Fails against the unfixed code with:
        AssertionError: the drag must be settled, not left live on the new tab
    (``ClayState.activate`` cleared ``drag_axis``/``ref`` and nothing else,
    so ``view._grab`` was still ``"keydrag"`` after the switch.)
    """
    doc_a = bd.ClayDoc()
    obj = bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())
    doc_a.add_object(obj)
    doc_a.select([obj.uid])
    doc_b = bd.ClayDoc()

    state = clay_mode.ClayState()
    tab_a = clay_mode.ClayTab(doc=doc_a)
    tab_b = clay_mode.ClayTab(doc=doc_b)
    state.add(tab_a)
    state.add(tab_b)
    state.active_uid = tab_a.uid

    ctx = SimpleNamespace(state=SimpleNamespace(clay=state, mode="clay"), settings=None)
    view = clay_view.ClayView(gl, ctx)
    ctx.clay_view = view
    try:
        clay_mode.ensure(ctx)  # wires ClayState.settle_drag to this view

        view.draw(doc_a, RECT, 0.0)
        view._last_mouse = (64.0, 48.0)
        assert view.begin_keyboard_drag(doc_a, "move")
        view._motion(doc_a, (100.0, 48.0))
        assert view.dragging
        head_a = doc_a.history.head
        dragged_to = np.array(obj.translation, copy=True)

        state.activate(tab_b.uid)  # exactly what the tab bar's click does

        assert state.active_uid == tab_b.uid
        assert not view.dragging, (
            "the drag must be settled, not left live on the new tab"
        )
        assert doc_a.history.head != head_a, "one history step for the whole gesture"
        assert np.allclose(obj.translation, dragged_to), (
            "the commit must land on tab A, the tab the drag began on"
        )
        assert doc_b.history.head == 0, "tab B -- the tab switched to -- must be untouched"
    finally:
        view.release()


# --- 2026-09-16 audit: add() (new/open/import/recover) settles nothing -------


def test_creating_or_opening_a_document_mid_drag_settles_the_drag_on_the_tab_it_replaces(
    gl,
) -> None:
    """``ClayState.add`` -- what ``new_document``, both ``clay-open`` adopt
    branches, ``clay-recover`` and both ``clay-import`` adopt branches all
    call to bring a *new* tab in -- had no settle at all, unlike ``activate``
    (fixed for exactly this class of bug by the 2026-09-15 audit's clay-01,
    the test above). New/Open carry no drag gate in the UI either
    (``widgets.document_header``'s buttons), and the async adopt paths are
    inherently decoupled from whatever drag is in progress when their result
    lands, so a live G/R/S drag on the tab being replaced was left with its
    TRS already written in place and no history step behind it -- Ctrl+Z can
    never revert a move nothing recorded.

    Fails against the unfixed code with:
        AssertionError: the drag must be settled, not left live on the tab it was replaced on
    (``ClayState.add`` only appended the new tab and cleared its own
    ``drag_axis``/``ref``, so ``view._grab`` was still ``"keydrag"`` after
    the add.)
    """
    doc_a = bd.ClayDoc()
    obj = bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())
    doc_a.add_object(obj)
    doc_a.select([obj.uid])

    state = clay_mode.ClayState()
    tab_a = clay_mode.ClayTab(doc=doc_a)
    state.add(tab_a)
    state.active_uid = tab_a.uid

    ctx = SimpleNamespace(state=SimpleNamespace(clay=state, mode="clay"), settings=None)
    view = clay_view.ClayView(gl, ctx)
    ctx.clay_view = view
    try:
        clay_mode.ensure(ctx)  # wires ClayState.settle_drag to this view

        view.draw(doc_a, RECT, 0.0)
        view._last_mouse = (64.0, 48.0)
        assert view.begin_keyboard_drag(doc_a, "move")
        view._motion(doc_a, (100.0, 48.0))
        assert view.dragging
        head_a = doc_a.history.head
        dragged_to = np.array(obj.translation, copy=True)

        doc_b = bd.ClayDoc()
        tab_b = clay_mode.ClayTab(doc=doc_b)
        state.add(tab_b)  # exactly what new_document/open/import/recover do

        assert state.active_uid == tab_b.uid
        assert not view.dragging, (
            "the drag must be settled, not left live on the tab it was replaced on"
        )
        assert doc_a.history.head != head_a, "one history step for the whole gesture"
        assert np.allclose(obj.translation, dragged_to), (
            "the commit must land on tab A, the tab the drag began on"
        )
        assert doc_b.history.head == 0, (
            "the new tab -- what it was replaced with -- must be untouched"
        )
    finally:
        view.release()


# --- clay-04: a copy family with a four-digit suffix -------------------------


def test_family_strips_a_four_digit_copy_suffix() -> None:
    """``ops.next_name`` counts a copy up past ``.999`` into ``.1000``,
    ``.10000``, and so on, never resetting the suffix width. ``diagnose.
    _family`` used to check only the character exactly four from the end for
    a literal ``.`` -- true for ``Box.001`` but false for ``Box.1000`` and
    every wider suffix after it -- so a family duplicated past 999 copies
    silently stopped being read as one: ``Box.1000`` came back as its own
    family, named after itself, instead of joining ``Box``.

    Fails against the unfixed code with:
        AssertionError: assert 'Box.1000' == 'Box'
    """
    assert diagnose._family("Box.1000") == "Box"
    assert diagnose._family("Box.10000") == "Box"
    # The ordinary three-digit case, and the bare (uncopied) name, both still
    # read the way they always did.
    assert diagnose._family("Box.001") == "Box"
    assert diagnose._family("Box") == "Box"
    # A name that merely contains a dot-digits run earlier, not at the end,
    # is not a copy suffix and must be left alone.
    assert diagnose._family("Box.001.glb") == "Box.001.glb"
