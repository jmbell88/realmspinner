"""The 2026-10-14 audit, finding tour-06.

``_notes`` memoises on the song's edit generation, but the Clay conditions the
torch tour gates on were recomputed every frame the step was on screen:
``clay_painted_at_least`` read ``(o.mesh.material != 0).any()`` for every object
and ``clay_shapes_at_least`` walked ``doc.objects``. "Skip step" lets a reader
run the tour inside a large existing model, so the pass is real frame time.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.studio.panes import tour as tour_pane


class _CountingMesh:
    """A mesh whose ``material`` read is counted -- the per-object pass."""

    def __init__(self, owner: _Doc, value: int) -> None:
        self._owner = owner
        self._value = value

    @property
    def material(self):
        self._owner.reads += 1
        return np.full(4, self._value, dtype="i4")


class _Doc:
    def __init__(self, painted: int, total: int = 5) -> None:
        self.reads = 0
        self.rev = 0
        self.history = SimpleNamespace(head=0)
        self.materials = [object(), object()]
        self.objects = [
            SimpleNamespace(
                generator="box", mesh=_CountingMesh(self, 1 if i < painted else 0)
            )
            for i in range(total)
        ]


def _ctx(doc: _Doc, uid: str = "bd1") -> SimpleNamespace:
    tab = SimpleNamespace(doc=doc, uid=uid, path=None, dirty=False)
    clay = SimpleNamespace(active=tab)
    return SimpleNamespace(state=SimpleNamespace(mode="clay", clay=clay, inker=None, sirens=None))


@pytest.fixture(autouse=True)
def _fresh_cache():
    cache = getattr(tour_pane, "_clay_cache", None)
    if cache is not None:
        cache.clear()
    yield
    if cache is not None:
        cache.clear()


def test_clay_painted_condition_is_recomputed_only_when_the_document_changes():
    doc = _Doc(painted=2)
    ctx = _ctx(doc)

    assert tour_pane.satisfied(ctx, "clay_painted_at_least", "2") is True
    first_pass = doc.reads
    assert first_pass == len(doc.objects)

    for _ in range(30):  # thirty frames with the step on screen and nothing edited
        assert tour_pane.satisfied(ctx, "clay_painted_at_least", "2") is True
    assert doc.reads == first_pass, "an unchanged document was re-scanned on a later frame"

    doc.history.head += 1  # an edit (or an undo) moves the generation
    tour_pane.satisfied(ctx, "clay_painted_at_least", "2")
    assert doc.reads == first_pass * 2


def test_clay_painted_condition_follows_the_document_through_a_change():
    doc = _Doc(painted=1)
    ctx = _ctx(doc)
    assert tour_pane.satisfied(ctx, "clay_painted_at_least", "2") is False
    doc.objects[1].mesh._value = 1
    doc.rev += 1  # every Clay door touches
    assert tour_pane.satisfied(ctx, "clay_painted_at_least", "2") is True


def test_clay_shape_count_is_memoised_and_a_second_tab_does_not_inherit_it():
    doc_a, doc_b = _Doc(painted=0, total=2), _Doc(painted=0, total=3)
    assert tour_pane.satisfied(_ctx(doc_a, "bd1"), "clay_shapes_at_least", "box:3") is False
    # Same generation, a different tab: the answer must be its own.
    assert tour_pane.satisfied(_ctx(doc_b, "bd2"), "clay_shapes_at_least", "box:3") is True
    assert tour_pane.satisfied(_ctx(doc_a, "bd1"), "clay_shapes_at_least", "box:3") is False
