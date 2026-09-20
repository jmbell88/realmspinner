"""Tranche 3: named checkpoints for agents -- keyed on ``history.head``'s
serial, never a stack position.

A position (a plain count of done steps) drifts under both eviction (which
pops from the *front* of the done list, shifting every later position down by
one with nothing to say so) and a redo (which replaces the very steps it
undid, at the same positions, under a *different* serial once anything has
diverged) -- ``mark()``'s own docstring in ``core/undo.py`` states the same
argument for why a gesture token is a serial. Every test here is really a
test of that argument, at the ``ClayDoc`` door instead of the ``UndoStack``
one.
"""

from __future__ import annotations

import numpy as np

from realmspinner.core.undo import UNDO_MAX_DEPTH
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser


def _obj(name: str, mesh: bm.Mesh | None = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=bp.box() if mesh is None else mesh,
        **kwargs,  # type: ignore[arg-type]
    )


# --- basic set / status / restore --------------------------------------------


def test_checkpoint_status_is_unknown_for_a_name_never_set() -> None:
    doc = bd.ClayDoc()
    assert doc.checkpoint_status("nope") == "unknown"
    assert doc.restore_checkpoint("nope") is False


def test_checkpoint_status_is_current_right_after_setting_it() -> None:
    doc = bd.ClayDoc()
    doc.add_object(_obj("A"))
    doc.set_checkpoint("cp")
    assert doc.checkpoint_status("cp") == "current"


def test_an_edit_after_the_checkpoint_makes_it_reachable_not_current() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_checkpoint("cp")
    doc.set_transform(a.uid, translation=(1.0, 0.0, 0.0))
    assert doc.checkpoint_status("cp") == "reachable"


def test_restore_checkpoint_moves_the_document_back() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_checkpoint("cp")
    doc.set_transform(a.uid, translation=(1.0, 0.0, 0.0))

    assert doc.restore_checkpoint("cp") is True
    assert np.allclose(doc.by_uid(a.uid).translation, [0.0, 0.0, 0.0])
    assert doc.checkpoint_status("cp") == "current"


def test_restore_checkpoint_is_false_when_already_current() -> None:
    doc = bd.ClayDoc()
    doc.add_object(_obj("A"))
    doc.set_checkpoint("cp")
    assert doc.restore_checkpoint("cp") is False


def test_restore_checkpoint_can_move_forward_too() -> None:
    """A checkpoint set *after* some edits, restored to from an earlier
    position -- the same call moves forward (redo) as well as back."""
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_transform(a.uid, translation=(1.0, 0.0, 0.0))
    doc.set_checkpoint("cp")
    doc.undo()
    assert doc.checkpoint_status("cp") == "reachable"

    assert doc.restore_checkpoint("cp") is True
    assert np.allclose(doc.by_uid(a.uid).translation, [1.0, 0.0, 0.0])


def test_re_setting_a_checkpoint_overwrites_its_position() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_checkpoint("cp")
    doc.set_transform(a.uid, translation=(1.0, 0.0, 0.0))
    doc.set_checkpoint("cp")  # moved
    doc.set_transform(a.uid, translation=(2.0, 0.0, 0.0))

    doc.restore_checkpoint("cp")
    assert np.allclose(doc.by_uid(a.uid).translation, [1.0, 0.0, 0.0])


def test_a_checkpoint_at_the_very_start_is_always_reachable() -> None:
    """Serial 0 -- ``history.head`` for a document with nothing done -- is
    never an edit's own serial (they start at 1), so it is always the
    "everything undone" position regardless of what has since been evicted."""
    doc = bd.ClayDoc()
    doc.set_checkpoint("empty")
    a = doc.add_object(_obj("A"))
    doc.set_transform(a.uid, translation=(1.0, 0.0, 0.0))
    doc.set_transform(a.uid, translation=(2.0, 0.0, 0.0))

    assert doc.checkpoint_status("empty") == "reachable"
    assert doc.restore_checkpoint("empty") is True
    assert doc.objects == []


# --- survives undo + redo ----------------------------------------------------


def test_checkpoint_survives_an_undo_and_a_redo() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_transform(a.uid, translation=(1.0, 0.0, 0.0))
    doc.set_checkpoint("cp")
    doc.set_transform(a.uid, translation=(2.0, 0.0, 0.0))

    doc.undo()  # back to the checkpoint's own position
    assert doc.checkpoint_status("cp") == "current"
    doc.redo()  # forward past it again, replaying the *same* edit
    assert doc.checkpoint_status("cp") == "reachable"
    assert doc.restore_checkpoint("cp") is True
    assert np.allclose(doc.by_uid(a.uid).translation, [1.0, 0.0, 0.0])


# --- gone after a divergent push ---------------------------------------------


def test_checkpoint_reports_gone_after_a_divergent_push() -> None:
    """The regression the tranche 3 spec names: a checkpoint's own serial,
    once its step has been undone and then discarded by a *different* push
    (``UndoStack.push`` clears the whole redo branch), is unreachable from
    either branch -- not "reachable" by coincidence, not silently treated as
    "current"."""
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_transform(a.uid, translation=(1.0, 0.0, 0.0))
    doc.set_checkpoint("cp")  # names the transform's own step

    doc.undo()  # cp's edit now sits on the redo branch
    assert doc.checkpoint_status("cp") == "reachable"

    doc.set_transform(a.uid, translation=(9.0, 0.0, 0.0))  # diverges; discards it
    assert doc.checkpoint_status("cp") == "gone"
    assert doc.restore_checkpoint("cp") is False
    # The document is exactly where the divergent push left it.
    assert np.allclose(doc.by_uid(a.uid).translation, [9.0, 0.0, 0.0])


def test_a_checkpoint_at_zero_stays_reachable_while_another_goes_gone() -> None:
    """Serial 0's special-casing (always reachable) must not bleed into an
    ordinary checkpoint's answer -- the two are checked side by side so a
    bug conflating them would show up as one of these two assertions, not
    both."""
    doc = bd.ClayDoc()
    doc.set_checkpoint("start")  # serial 0, before anything exists
    a = doc.add_object(_obj("A"))
    doc.set_transform(a.uid, translation=(1.0, 0.0, 0.0))
    doc.set_checkpoint("mid")  # names the transform's own step

    doc.undo()  # "mid"'s edit is now on the redo branch
    doc.set_transform(a.uid, translation=(9.0, 0.0, 0.0))  # diverges, discards it

    assert doc.checkpoint_status("start") == "reachable"
    assert doc.checkpoint_status("mid") == "gone"


# --- eviction never renumbers a survivor -------------------------------------


def test_eviction_reports_an_evicted_checkpoint_gone_and_leaves_a_survivor_reachable() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))  # push 1
    doc.set_transform(a.uid, translation=(0.0, 0.0, 0.0))  # push 2
    doc.set_checkpoint("early")  # names push 2 -- guaranteed off the front below

    extra = UNDO_MAX_DEPTH + 20
    survives_at = extra - 5  # comfortably inside the last UNDO_MAX_DEPTH pushes kept
    # Enough further pushes to run the done list well past UNDO_MAX_DEPTH,
    # each one a genuine step (a no-op push records nothing, so every one of
    # these -- a strictly increasing translation -- actually pushes).
    for i in range(1, extra):
        doc.set_transform(a.uid, translation=(float(i), 0.0, 0.0))  # push 2 + i
        if i == survives_at:
            doc.set_checkpoint("survives")

    assert len(doc.history) == UNDO_MAX_DEPTH  # the depth cap actually fired
    assert doc.checkpoint_status("early") == "gone"
    assert doc.restore_checkpoint("early") is False
    assert doc.checkpoint_status("survives") in ("reachable", "current")
    assert doc.restore_checkpoint("survives") is True
    assert np.allclose(doc.by_uid(a.uid).translation, [float(survives_at), 0.0, 0.0])


# --- not serialized ------------------------------------------------------


def test_checkpoints_are_not_serialized() -> None:
    doc = bd.ClayDoc()
    doc.add_object(_obj("A"))
    doc.set_checkpoint("cp")
    assert doc.checkpoints  # sanity: it really was set

    reloaded = ser.read_rblk(ser.rblk_bytes(doc))
    assert reloaded.checkpoints == {}
    assert reloaded.checkpoint_status("cp") == "unknown"
