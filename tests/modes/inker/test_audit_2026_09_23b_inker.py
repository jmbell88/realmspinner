"""Regression tests for the 2026-09-23 (second run) audit's inker fixer
batch, for the findings whose owned files live under ``studio/modes/inker``.

inker-02 (Flourish Restyle landing only the first facing of a multi-facing
effect) and inker-05 (Paste as new layer saying nothing on an empty
clipboard) live here. inker-01, inker-03 and inker-04 (``kernels/pixel``)
live in ``tests/kernels/pixel/test_audit_2026_09_23b_pixel.py``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.pixel.flourish import bake as B
from realmspinner.kernels.pixel.flourish import presets
from realmspinner.studio.modes.inker import flourish as inker_flourish
from realmspinner.studio.modes.inker import ops as inker_ops
from realmspinner.studio.modes.inker import state as inker_state
from realmspinner.studio.tasks import Done

# -- inker-02: Restyle on a multi-facing effect --------------------------------------------


class _Store:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}

    def get(self, job_id):
        return self.jobs.get(job_id)


class _Ctx:
    """``tests/modes/inker/test_flourish_restyle.py``'s harness, trimmed to
    what this file needs: a synchronous ``submit`` and a toast log."""

    def __init__(self, root: Path) -> None:
        self.state = SimpleNamespace(inker=inker_state.InkerState())
        self.svc = SimpleNamespace(store=_Store(), job_dir=lambda j: root / j, config=None)
        self.toasts: list = []
        self.tasks = SimpleNamespace(set_progress=lambda *a, **k: None)
        self.pending: list[Done] = []
        self._busy: set[str] = set()

    def toast(self, text, level="info", **_):
        self.toasts.append((text, level))

    def busy(self, key):
        return key in self._busy

    def progress(self, key):
        return None

    def submit(self, key, fn, *args, **kwargs):
        if key in self._busy:
            return False
        self._busy.add(key)
        self.pending.append(Done(key=key, result=None))
        return True


def _multi_facing_scene(tmp_path):
    ctx = _Ctx(tmp_path)
    tab = inker_state.InkerDoc(doc=inker.Document.blank(32, 32))
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    rec = dataclasses.replace(
        presets.load("sword_impact"), width=32, height=32, supersample=1, directions=4
    )
    group = tab.doc.insert_flourish(B.bake(rec, directions=4))
    return ctx, tab, group


def test_restyle_refuses_a_multi_direction_effect_instead_of_landing_only_the_first_facing(
    tmp_path,
):
    """The 2026-09-23 (second run) audit, finding inker-02: with more than
    one facing, ``bake.tags()`` names every phase's span once per facing
    (``phase/E``, ``phase/SE``, ...) and ``_phase_span`` matched the first
    tag whose name equalled ``phase`` or started with ``phase/`` -- so
    Restyle rendered and landed only the ``E`` facing's span while the toast
    claimed the whole phase, leaving the other facings' procedural frames
    untouched. Restyle must refuse outright rather than land a partial
    result the user has no way to tell apart from a full one.
    """
    ctx, tab, group = _multi_facing_scene(tmp_path)
    state = ctx.state.inker
    held = tab.doc.flourish_state(group)
    assert held.recipe.directions == 4  # the scene is actually multi-facing

    # The button itself must be greyed, with a reason that says why.
    assert not inker_flourish.can_restyle(state, tab)
    assert inker_flourish.restyle_reason(state, tab) == inker_flourish.MULTI_FACING_RESTYLE

    # And the door refuses too, even called directly -- no task submitted,
    # nothing landed, no jobs queued.
    phase = inker_flourish.phase_names(state, tab)[0]
    submitted = inker_flourish.submit_restyle(ctx, state, tab, phase=phase, subject="test")
    assert submitted is False
    assert ctx.pending == []
    assert state.flourish_restyle_pending is None
    # ``state.say`` (the refusal door for a gesture) puts a tip under the
    # canvas -- it does not toast.
    assert state.tip is not None and state.tip.text == inker_flourish.MULTI_FACING_RESTYLE


def test_restyle_still_works_on_a_one_facing_effect(tmp_path):
    """The refusal above must not catch the ordinary, One-facing case --
    it is scoped to ``directions > 1``, not to Restyle in general."""
    ctx = _Ctx(tmp_path)
    tab = inker_state.InkerDoc(doc=inker.Document.blank(32, 32))
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    rec = dataclasses.replace(presets.load("sword_impact"), width=32, height=32, supersample=1)
    tab.doc.insert_flourish(B.bake(rec))
    state = ctx.state.inker

    assert inker_flourish.can_restyle(state, tab)
    assert inker_flourish.restyle_reason(state, tab) == ""


# -- inker-05: paste as new layer on an empty clipboard ------------------------------------


def test_paste_as_layer_says_why_when_the_clipboard_is_empty(monkeypatch):
    """The 2026-09-23 (second run) audit, finding inker-05: plain Paste
    already toasts "There is nothing on the clipboard." when the clipboard
    is empty, but Ctrl+Shift+V's ``as_layer`` branch returned
    ``paste_as_layer()``'s ``False`` straight through, saying nothing."""
    tab = inker_state.InkerDoc(doc=inker.Document.blank(8, 8))
    state = inker_state.InkerState()
    state.docs.append(tab)
    state.active_uid = tab.uid
    toasts: list[str] = []
    monkeypatch.setattr(state, "say", toasts.append)

    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))
    monkeypatch.setattr(
        "realmspinner.studio.modes.inker.mode.paste_from_os", lambda ctx, tab: None
    )
    assert tab.doc.paste_as_layer() is False  # sanity: this doc's clipboard is empty

    result = inker_ops._paste(ctx, tab, as_layer=True)

    assert result is False
    assert toasts == ["There is nothing on the clipboard."]
