"""Job kinds that no longer exist, and what a row of one is told.

The character-sheet pipeline (``charsheet``, ``sheet`` and ``pixel_sheet``) left Realmspinner
with Poser (the 2026-10-13 Clay rig plan, Tranche 5): a rigged character is posed and animated in
Clay now, and the sprite-sheet renderers had no mode left to start them. A **row** of one of these
kinds can still be in a user's library -- the database outlives the feature -- so the kinds are
named here once, in a module with no imports, which the queue, the service layer and the studio
may all read (``service`` is not importable from the queue, which is why this does not live there).

What a retired row does: it is never run again (``Worker._generate`` marks a stuck queued or
running one as an error, and ``migrate`` flips them at open), it is never offered a button,
rerun, retarget or export (every "can this row ..." predicate answers no), it opens the Library on
its own card with :data:`NOTE`, and its files are **never deleted** -- the user's work is theirs
until they trash it.

``tests/test_retired_kinds.py`` is the gate: each of these kinds is absent from every table that
decides what a live kind is offered (the phase tables, the products, the follow-up stages, the
publisher list), so a retired kind cannot creep back in through the one table nobody re-read.
"""

from __future__ import annotations

RETIRED_KINDS: frozenset[str] = frozenset({"charsheet", "sheet", "pixel_sheet"})

#: What the Library says when a retired row is opened.
NOTE = (
    "Character sheets were removed from Realmspinner. This row stays in your library as it "
    "was, but it can no longer be rerun, opened or exported."
)


def is_retired(kind: object) -> bool:
    """Whether *kind* is one of :data:`RETIRED_KINDS`."""
    return isinstance(kind, str) and kind in RETIRED_KINDS
