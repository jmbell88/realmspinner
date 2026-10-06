"""The Library's meaning index at the storage layer: ``job_embeddings``.

The table, how a stale row is told from a current one, that deleting a job takes
its vector with it, and the ranking query. Nothing here starts an embedder --
the vectors are hand-built unit vectors, because this layer stores and ranks
what it is handed and must never reach the network.
"""

from __future__ import annotations

import pathlib
import sqlite3

import numpy as np
import pytest

from realmspinner.db import MIGRATIONS, JobStore

SHA = "a" * 64
OTHER_SHA = "b" * 64
DIM = 8


def _unit(index: int, dim: int = DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[index % dim] = 1.0
    return v


def _noisy(index: int, dim: int, seed: int = 0) -> np.ndarray:
    """A unit axis plus a little noise, unit length. Exactly-orthogonal vectors
    give a median absolute deviation of zero (half the scores identical), which
    no real embedding does -- so the ranking tests use these."""
    rng = np.random.default_rng(seed * 1000 + index)
    v = _unit(index, dim) + 0.05 * rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _text_of(row):
    """A text builder shaped like ``library_index.job_text``: the name and
    prompt, hashed by their content."""
    text = f"{row['name']}|{row['prompt']}"
    return text, f"hash:{text}"


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.sqlite")
    yield s
    s.close()


def _done(store: JobStore, prompt: str, name: str = "", **kw) -> str:
    job_id = store.create("text", prompt, kw.pop("params", {}), status="done", **kw)
    if name:
        store.set_meta(job_id, name=name)
    return job_id


def _table_sql(path) -> str:
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'job_embeddings'"
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else ""


# --- the table ---------------------------------------------------------------


def test_a_fresh_database_has_the_job_embeddings_table(tmp_path):
    path = tmp_path / "fresh.sqlite"
    JobStore(path).close()
    sql = _table_sql(path)
    assert "job_id" in sql and "model_sha" in sql and "dim" in sql
    assert "text_hash" in sql and "vec" in sql


def test_a_database_at_the_previous_version_gains_the_table_on_open(tmp_path):
    """Migration 12: a DB sitting at user_version 11 with no ``job_embeddings``
    -- every library written before the meaning index existed -- opens, gains
    the table and lands on the new version, rows intact."""
    path = tmp_path / "old.sqlite"
    store = JobStore(path)
    keep = _done(store, "an old sword", name="old")
    store.close()
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE job_embeddings")
    conn.execute(f"PRAGMA user_version = {len(MIGRATIONS) - 1}")
    conn.commit()
    conn.close()
    assert _table_sql(path) == ""

    reopened = JobStore(path)
    try:
        assert reopened.get(keep)["prompt"] == "an old sword"
        assert reopened.upsert_embeddings(SHA, DIM, [(keep, "h", _unit(0))]) == 1
    finally:
        reopened.close()
    assert "job_embeddings" in _table_sql(path)
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    conn.close()


def test_the_migration_that_adds_the_table_is_the_last_one_and_is_idempotent(tmp_path):
    """The statement is IF NOT EXISTS, so a replay on a database that already
    has the table (every fresh one, which gets it from the schema script too)
    is a no-op rather than an error that stops the app starting."""
    last = MIGRATIONS[-1]
    assert any("job_embeddings" in stmt for stmt in last)
    path = tmp_path / "again.sqlite"
    JobStore(path).close()
    conn = sqlite3.connect(path)
    for stmt in last:
        conn.execute(stmt)
    conn.close()


# --- upsert and what counts as stale ------------------------------------------


def test_a_finished_job_with_no_vector_is_pending(store):
    job_id = _done(store, "a rusty longsword")
    pending = store.pending_embeddings(SHA, DIM, _text_of, 10)
    assert [p[0] for p in pending] == [job_id]


def test_a_stored_vector_is_no_longer_pending(store):
    job_id = _done(store, "a rusty longsword")
    (_id, _text, text_hash), = store.pending_embeddings(SHA, DIM, _text_of, 10)
    assert store.upsert_embeddings(SHA, DIM, [(job_id, text_hash, _unit(0))]) == 1
    assert store.pending_embeddings(SHA, DIM, _text_of, 10) == []
    assert store.embedding_stamps([job_id]) == {job_id: (SHA, DIM, text_hash)}


def test_a_vector_from_another_model_is_stale_and_gets_replaced(store):
    job_id = _done(store, "a rusty longsword")
    (_i, _t, text_hash), = store.pending_embeddings(SHA, DIM, _text_of, 10)
    store.upsert_embeddings(SHA, DIM, [(job_id, text_hash, _unit(0))])
    again = store.pending_embeddings(OTHER_SHA, DIM, _text_of, 10)
    assert [p[0] for p in again] == [job_id]
    store.upsert_embeddings(OTHER_SHA, DIM, [(job_id, text_hash, _unit(1))])
    # One row per job: replaced, never appended.
    count = store._conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0]
    assert count == 1
    assert store.pending_embeddings(OTHER_SHA, DIM, _text_of, 10) == []


def test_a_vector_of_another_width_is_stale(store):
    job_id = _done(store, "a rusty longsword")
    (_i, _t, text_hash), = store.pending_embeddings(SHA, DIM, _text_of, 10)
    store.upsert_embeddings(SHA, DIM, [(job_id, text_hash, _unit(0))])
    wider = store.pending_embeddings(SHA, 2 * DIM, _text_of, 10)
    assert [p[0] for p in wider] == [job_id]


def test_a_renamed_job_is_found_by_the_verify_scan_and_only_by_it(store):
    job_id = _done(store, "a rusty longsword", name="Sword")
    (_i, _t, text_hash), = store.pending_embeddings(SHA, DIM, _text_of, 10)
    store.upsert_embeddings(SHA, DIM, [(job_id, text_hash, _unit(0))])
    store.set_meta(job_id, name="Excalibur")
    # The cheap SQL-only pass cannot know the text moved ...
    assert store.pending_embeddings(SHA, DIM, _text_of, 10) == []
    # ... the verify pass recomputes the hash and does.
    drifted = store.pending_embeddings(SHA, DIM, _text_of, 10, verify=True)
    assert [d[0] for d in drifted] == [job_id]
    assert "Excalibur" in drifted[0][1]


def test_pending_skips_unfinished_jobs_sweep_units_and_rows_with_nothing_to_say(store):
    queued = store.create("text", "still queued", {})
    errored = store.create("text", "failed one", {}, status="error")
    sweep_id = store.create_sweep("s", "p", {})
    unit = store.create("text", "sweep unit", {}, status="done", sweep_id=sweep_id)
    blank = _done(store, "   ")
    good = _done(store, "kept")
    ids = [p[0] for p in store.pending_embeddings(SHA, DIM, _text_of, 50)]
    assert ids == [good]
    for skipped in (queued, errored, unit, blank):
        assert skipped not in ids


def test_pending_is_newest_first_and_bounded_by_the_limit(store):
    ids = [_done(store, f"job {i}") for i in range(5)]
    got = [p[0] for p in store.pending_embeddings(SHA, DIM, _text_of, 3)]
    assert got == list(reversed(ids))[:3]


def test_a_text_builder_that_declines_a_row_cannot_starve_the_rest(store):
    """A prompt that is only control characters passes the SQL's TRIM and is
    declined by the builder; returned first on every pass it would hide every
    row behind it. Paging past it is what the query's OFFSET loop is for."""
    wanted = [_done(store, f"real {i}") for i in range(3)]
    for _ in range(70):
        _done(store, "\t\n")

    def builder(row):
        return None if not row["prompt"].strip() else _text_of(row)

    got = [p[0] for p in store.pending_embeddings(SHA, DIM, builder, 3)]
    assert sorted(got) == sorted(wanted)


def test_an_embedding_of_the_wrong_length_is_refused(store):
    job_id = _done(store, "x")
    with pytest.raises(ValueError):
        store.upsert_embeddings(SHA, DIM, [(job_id, "h", np.zeros(DIM + 1, dtype=np.float32))])


def test_an_embedding_for_a_job_that_was_deleted_meanwhile_is_not_written(store):
    """The vector is computed on a worker thread; the delete that already ran
    cannot clean up a row written after it."""
    job_id = _done(store, "x")
    store.delete(job_id)
    assert store.upsert_embeddings(SHA, DIM, [(job_id, "h", _unit(0))]) == 0
    assert store._conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0] == 0


# --- deletion -------------------------------------------------------------------


def _count(store: JobStore) -> int:
    return store._conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0]


def test_deleting_a_job_removes_its_embedding(store):
    keep = _done(store, "keep")
    gone = _done(store, "gone")
    store.upsert_embeddings(SHA, DIM, [(keep, "h", _unit(0)), (gone, "h", _unit(1))])
    store.delete(gone)
    assert set(store.embedding_stamps()) == {keep}


def test_purging_a_job_that_is_not_running_removes_its_embedding(store):
    job_id = _done(store, "x")
    store.upsert_embeddings(SHA, DIM, [(job_id, "h", _unit(0))])
    assert store.delete_if_not_running(job_id) is True
    assert _count(store) == 0


def test_a_refused_delete_leaves_the_embedding_alone(store):
    job_id = store.create("text", "running now", {})
    store.set_status(job_id, "running")
    # A finished row's vector is the only kind there is; fake one for a row the
    # worker owns to prove a *refused* delete does not touch it.
    store._conn.execute(
        "INSERT INTO job_embeddings VALUES (?, ?, ?, ?, ?)",
        (job_id, SHA, DIM, "h", _unit(0).tobytes()),
    )
    store._conn.commit()
    assert store.delete_if_not_running(job_id) is False
    assert _count(store) == 1


def test_the_trash_keeps_the_vector_so_a_restore_loses_nothing(store):
    job_id = _done(store, "x")
    store.upsert_embeddings(SHA, DIM, [(job_id, "h", _unit(0))])
    assert store.set_deleted_if_not_running(job_id, 123.0) is True
    assert _count(store) == 1


def test_prune_removes_vectors_whose_job_is_gone(store):
    live = _done(store, "live")
    store.upsert_embeddings(SHA, DIM, [(live, "h", _unit(0))])
    store._conn.execute(
        "INSERT INTO job_embeddings VALUES ('orphan', ?, ?, 'h', ?)",
        (SHA, DIM, _unit(1).tobytes()),
    )
    store._conn.commit()
    assert store.prune_embeddings() == 1
    assert set(store.embedding_stamps()) == {live}


# --- ranking -------------------------------------------------------------------


def _library(store: JobStore, n: int = 30, *, dim: int = DIM) -> list[str]:
    """*n* finished jobs whose vectors are the unit axes plus noise (so each is
    nearly orthogonal to every other), oldest first."""
    ids = [_done(store, f"job {i}", name=f"name {i}") for i in range(n)]
    store.upsert_embeddings(
        SHA, dim, [(job_id, "h", _noisy(i, dim)) for i, job_id in enumerate(ids)]
    )
    return ids


def _rank(store, query, **kw):
    args = dict(model_sha=SHA, dim=DIM, z_floor=3.0, cap=20, min_rows=20)
    args.update(kw)
    return store.semantic_ranked(query, **args)


def test_the_closest_row_is_ranked_first_and_an_unrelated_one_is_not_returned(store):
    ids = _library(store, 30, dim=64)
    query = _unit(7, 64)
    hits = store.semantic_ranked(
        query, model_sha=SHA, dim=64, z_floor=3.0, cap=20, min_rows=20
    )
    assert [h[0] for h in hits] == [ids[7]]
    assert hits[0][1] > 0.9


def test_a_query_close_to_everything_adds_nothing_the_floor_holds(store):
    """The failure the floor exists for: a query that is a little like every
    row (nonsense, in practice) must not flood the list with the least-bad ones."""
    _library(store, 30, dim=64)
    flat = np.ones(64, dtype=np.float32)
    assert store.semantic_ranked(
        flat, model_sha=SHA, dim=64, z_floor=3.0, cap=20, min_rows=20
    ) == []


def test_the_cap_holds(store):
    ids = [_done(store, f"job {i}") for i in range(60)]
    rows = []
    for i, job_id in enumerate(ids):
        v = _noisy(26 + (i - 25) if i >= 25 else 0, 64, seed=i)
        if i < 25:
            v = v.copy()
            v[0] += 1.0  # 25 near-duplicates of the query, each a little different
            v /= np.linalg.norm(v)
        rows.append((job_id, "h", v))
    store.upsert_embeddings(SHA, 64, rows)
    hits = store.semantic_ranked(
        _unit(0, 64), model_sha=SHA, dim=64, z_floor=3.0, cap=20, min_rows=20
    )
    assert len(hits) == 20
    scores = [s for _id, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_a_small_library_is_not_searched_by_meaning(store):
    ids = _library(store, 10, dim=64)
    assert store.semantic_ranked(
        _unit(3, 64), model_sha=SHA, dim=64, z_floor=3.0, cap=20, min_rows=20
    ) == []
    assert ids  # the rows exist; the floor is about having a baseline


def test_a_query_matching_a_large_cluster_still_finds_it(store):
    """Why the baseline is the median and MAD, not mean and standard deviation:
    a third of the library agreeing with the query inflates a standard deviation
    until nothing clears three of them."""
    ids = [_done(store, f"chest {i}") for i in range(60)]
    rows = []
    for i, job_id in enumerate(ids):
        v = _noisy(0 if i < 20 else 2 + (i % 60), 64, seed=i).copy()
        if i < 20:
            v[0] += 1.0
            v /= np.linalg.norm(v)
        rows.append((job_id, "h", v))
    store.upsert_embeddings(SHA, 64, rows)
    hits = store.semantic_ranked(
        _unit(0, 64), model_sha=SHA, dim=64, z_floor=3.0, cap=40, min_rows=20
    )
    assert {h[0] for h in hits} == set(ids[:20])
    # The same data under a plain mean/std baseline finds nothing -- the claim
    # this test pins the median/MAD choice against.
    scores = np.stack([r[2] for r in rows]) @ _unit(0, 64)
    assert scores.max() < scores.mean() + 3.0 * scores.std()


def test_a_trashed_job_never_appears_in_the_workshop_ranking(store):
    ids = _library(store, 30, dim=64)
    store.set_deleted_if_not_running(ids[7], 5.0)
    kw = dict(model_sha=SHA, dim=64, z_floor=3.0, cap=20, min_rows=20)
    assert store.semantic_ranked(_unit(7, 64), **kw) == []
    # ... and is the *only* thing the trash view ranks.
    assert [h[0] for h in store.semantic_ranked(_unit(7, 64), trash=True, **kw)] == [ids[7]]


def test_the_scope_clauses_narrow_the_meaning_path_like_the_text_path(store):
    ids = _library(store, 30, dim=64)
    store.set_meta(ids[7], tags="wood")
    kw = dict(model_sha=SHA, dim=64, z_floor=3.0, cap=20, min_rows=20)
    assert [h[0] for h in store.semantic_ranked(_unit(7, 64), tags=("wood",), **kw)] == [ids[7]]
    assert store.semantic_ranked(_unit(7, 64), tags=("stone",), **kw) == []
    assert store.semantic_ranked(_unit(7, 64), names=("nomatch",), **kw) == []
    assert [h[0] for h in store.semantic_ranked(_unit(7, 64), names=("name 7",), **kw)] == [ids[7]]
    assert store.semantic_ranked(_unit(7, 64), status="error", **kw) == []


def test_a_sweep_unit_is_never_ranked(store):
    ids = _library(store, 30, dim=64)
    sweep_id = store.create_sweep("s", "p", {})
    store._conn.execute("UPDATE jobs SET sweep_id = ? WHERE id = ?", (sweep_id, ids[7]))
    store._conn.commit()
    assert store.semantic_ranked(
        _unit(7, 64), model_sha=SHA, dim=64, z_floor=3.0, cap=20, min_rows=20
    ) == []


def test_only_vectors_of_the_current_model_and_width_are_ranked(store):
    ids = _library(store, 30, dim=64)
    kw = dict(dim=64, z_floor=3.0, cap=20, min_rows=20)
    assert store.semantic_ranked(_unit(7, 64), model_sha=OTHER_SHA, **kw) == []
    assert [h[0] for h in store.semantic_ranked(_unit(7, 64), model_sha=SHA, **kw)] == [ids[7]]
    # A query of another width is refused outright, not mis-multiplied.
    assert store.semantic_ranked(_unit(7, 32), model_sha=SHA, **kw) == []


def test_the_store_never_imports_the_network_or_the_embedder():
    """``db.py`` is layer 0: it stores and ranks vectors it is handed. The
    embedding itself belongs to ``service.library_index``."""
    import realmspinner.db as db_mod

    source = pathlib.Path(db_mod.__file__).read_text(encoding="utf-8")
    for banned in ("httpx", "embed_client", "familiar", "library_index"):
        assert f"import {banned}" not in source
        assert f"from .{banned}" not in source
