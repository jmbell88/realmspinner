"""The 2026-09-26 audit, finding tour-1-02.

``_notes()``'s single-entry cache used to key its identity half on
``id(doc)``. ``id()`` is a memory address, and CPython is free to hand the
same address to a brand-new object once the old one holding it has been
garbage-collected -- so a later, unrelated document landing at the same
address with the same ``history.head`` could be served the previous
document's cached note count instead of scanning its own.

Reliably forcing CPython to reuse an address in a test is exactly the kind
of GC-timing bet ``dev/INVARIANTS.md`` warns against relying on, so this
proves the fix two ways instead: (1) construct two distinct, simultaneously
live "documents" that only ``id()`` could ever confuse -- same
``history.head``, different note counts -- and show each gets its own
answer while both are alive, which is the scenario a same-address collision
would break if the code still trusted addresses; and (2) assert directly
that the cache's identity half is no longer ``id(doc)`` at all, by using
``SongTab.uid`` (minted from ``sirens.state``'s ``itertools.count``, which
never repeats for the life of the process) -- an identity that cannot
collide by construction, address reuse or not.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.studio.modes.sirens.engine import document as D
from realmspinner.studio.modes.sirens.state import SongTab
from realmspinner.studio.panes import tour as tour_pane


def _pattern(rows: list[list[int]]) -> SimpleNamespace:
    cells = np.full((len(rows), len(rows[0]), D.COLUMNS), -1, dtype="i2")
    cells[:, :, D.NOTE] = np.asarray(rows, dtype="i2")
    return SimpleNamespace(cells=cells)


def _doc(head: int, *patterns: list[list[int]]) -> SimpleNamespace:
    return SimpleNamespace(
        history=SimpleNamespace(head=head),
        patterns=[_pattern(rows) for rows in patterns],
    )


def _ctx(tab: SimpleNamespace) -> SimpleNamespace:
    sirens = SimpleNamespace(active=tab)
    return SimpleNamespace(state=SimpleNamespace(mode="sirens", inker=None, sirens=sirens))


@pytest.fixture(autouse=True)
def _reset_notes_cache():
    tour_pane._notes_cache = None
    yield
    tour_pane._notes_cache = None


def test_notes_cache_key_is_not_id_of_doc():
    """The identity half of ``_notes_cache`` can no longer be ``id(doc)``.

    Fails against the unfixed code, which builds its key as
    ``(id(doc), int(head))``: the first populated cache row's identity
    component is exactly ``id(doc)`` there. Passes against the fix, where the
    identity is ``tab.uid`` -- a ``SongTab`` always has one, and it is a
    string minted from a monotonic counter, never a recycled address.
    """
    doc = _doc(0, [[24, -1]])
    tab = SongTab(doc=doc)
    tour_pane._notes(_ctx(tab))

    assert tour_pane._notes_cache is not None
    identity, _head, _count = tour_pane._notes_cache
    assert identity == tab.uid
    assert identity != id(doc), (
        "the cache is still keyed on id(doc) -- exactly the address that a "
        "later, unrelated document can inherit once this one is collected"
    )


def test_two_documents_sharing_a_history_head_never_cross_contaminate():
    """Two live, distinct documents at the same ``history.head`` must each
    answer with their own note count -- the shape of the bug an id()-reused
    address would produce, proven here without depending on GC timing: two
    real, simultaneously-alive objects are never the same key under the
    fixed ``tab.uid``-keyed cache, regardless of what CPython later does with
    either object's address.
    """
    doc_a = _doc(0, [[24, -1], [26, -1]])  # 2 real pitches
    doc_b = _doc(0, [[24, 24, 24]])  # 3 real pitches, same history.head == 0
    tab_a = SongTab(doc=doc_a)
    tab_b = SongTab(doc=doc_b)

    count_a = tour_pane._notes(_ctx(tab_a))
    count_b = tour_pane._notes(_ctx(tab_b))
    # Re-read A after B primed the single-entry cache at the same head, the
    # exact moment the old id(doc)-keyed cache could serve B's cached row
    # back to A if their addresses ever coincided.
    count_a_again = tour_pane._notes(_ctx(tab_a))

    assert count_a == 2
    assert count_b == 3
    assert count_a_again == 2


def test_notes_still_memoises_within_one_tab():
    """The fix must not cost the memoisation tour-02 added: a second call at
    the same ``history.head`` for the same tab is still a cache hit."""
    doc = _doc(0, [[24, -1], [26, -1]])
    tab = SongTab(doc=doc)
    ctx = _ctx(tab)

    first = tour_pane._notes(ctx)
    cached_after_first = tour_pane._notes_cache
    second = tour_pane._notes(ctx)

    assert first == second == 2
    assert tour_pane._notes_cache == cached_after_first

    doc.history.head += 1
    third = tour_pane._notes(ctx)
    assert third == 2
    assert tour_pane._notes_cache[1] == 1
