"""service-queue-04 (2026-09-26 audit): a corrupt sweep spec took down the
whole rescan.

``list_sweeps``/``get_sweep`` decoded the ``spec`` column with a bare
``json.loads``, unlike every other JSON column this store holds
(``observations``/``verdicts`` go through ``_blob``, ``params`` through
``_params_blob``, both added after the exact same failure class). One row
with an unparseable ``spec`` -- a hand edit, a torn write, disk corruption --
raised straight out of the loop in ``list_sweeps`` and took every *other*
sweep down with it, which is Review's own rescan on every visit until that
row is repaired or deleted by hand.
"""

from __future__ import annotations

import pytest

from realmspinner.db import JobStore


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.sqlite")
    yield s
    s.close()


def _corrupt_spec(store: JobStore, sweep_id: str) -> None:
    with store._lock:
        store._conn.execute(
            "UPDATE sweeps SET spec = ? WHERE id = ?", ("{not json", sweep_id)
        )
        store._commit()


def test_list_sweeps_skips_a_corrupt_spec_instead_of_raising(store):
    good_id = store.create_sweep("a good sweep", "a barrel", {"seeds": [1, 2]})
    bad_id = store.create_sweep("a bad sweep", "a crate", {"seeds": [3]})
    _corrupt_spec(store, bad_id)

    rows = {row["id"]: row for row in store.list_sweeps()}

    assert set(rows) == {good_id, bad_id}
    assert rows[good_id]["spec"] == {"seeds": [1, 2]}
    # Unreadable, not fatal: the corrupt row's own spec reads as None --
    # ``_blob``'s "a row nobody can read is one row of evidence lost" -- rather
    # than taking the whole list down with it.
    assert rows[bad_id]["spec"] is None


def test_get_sweep_tolerates_a_corrupt_spec_too(store):
    bad_id = store.create_sweep("a bad sweep", "a crate", {"seeds": [3]})
    _corrupt_spec(store, bad_id)

    row = store.get_sweep(bad_id)

    assert row is not None
    assert row["spec"] is None
