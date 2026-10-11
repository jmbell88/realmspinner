"""The generation log's backend: a verdict or observation carries what ran, a
hard delete stops losing it, and the log reads back and exports.

The incident: the 2026-10-11 look at the real store found 13 of 17 verdicts
pointing at deleted jobs, each holding only the config vector -- which omits
steps, guidance, scheduler, seed and prompt on purpose.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from realmspinner import queue as queue_mod
from realmspinner import vectors
from realmspinner.db import MIGRATIONS, JobStore
from realmspinner.service import generation_log
from realmspinner.service import verdicts as svc_verdicts
from realmspinner.service.errors import Invalid

PARAMS = {
    "generation_type": "object",
    "base_model": "sdxl_cfg",
    "style_lora": "render3d",
    "lora_weight": 0.9,
    "seed": 1204285292,
    "profile": "standard",
    "recipe": {
        "reference": {
            "base_model": "sdxl_cfg",
            "family": "sdxl",
            "steps": 30,
            "guidance_scale": 7.0,
            "scheduler": "EulerDiscreteScheduler",
            "seed": 1204285292,
            "prompt": "3d style, a chest",
            "negative_prompt": "blurry",
            "style_lora": "render3d",
            "lora_weight": 0.9,
            "models": {"base_model": "abc"},
            "versions": {"torch": "2.9"},
        },
        "trellis": {"seed": 7, "band": 4, "tex_res": 1024, "engine_version": "x"},
    },
    "reference_attempts": [{}, {}],
    "reference_report": {"ok": True, "codes": [], "bbox": [1, 2, 3, 4]},
    "mesh_audit": {"worst": 0.01, "mean": 0.0},
    "mesh_report": {"triangles": 100, "status": "ready", "welded_watertight": True},
}


def _done_job(svc, params=None):
    return svc.store.create(
        "image", "a chest", dict(PARAMS if params is None else params),
        stage="model", status="done",
    )


def test_a_verdict_on_a_hard_deleted_job_still_carries_steps_guidance_and_seed(svc):
    job_id = _done_job(svc)
    svc_verdicts.record_verdict(svc, job_id, grade=4)
    svc.store.delete(job_id)

    rows = generation_log.history(svc)
    assert [r["job_id"] for r in rows] == [job_id]
    row = rows[0]
    assert row["exists"] is False
    snap = row["snapshot"]
    assert (snap["steps"], snap["guidance_scale"], snap["seed"]) == (30, 7.0, 1204285292)
    assert row["prompt"] == "a chest"
    assert "30 steps" in row["settings_line"] and "seed 1204285292" in row["settings_line"]


def test_an_observation_carries_a_snapshot(svc):
    job_id = _done_job(svc)
    assert queue_mod._observe_finished(svc.store, job_id)
    obs = svc.store.latest_observations()[0]
    assert obs["snapshot"]["steps"] == 30
    assert obs["snapshot"]["v"] == 1


def test_a_note_round_trips_and_the_newest_note_wins(svc):
    job_id = _done_job(svc)
    svc_verdicts.record_verdict(svc, job_id, grade=2, note="  first thoughts  ")
    assert svc.store.latest_verdicts()[0]["note"] == "first thoughts"
    svc_verdicts.record_verdict(svc, job_id, grade=2, note="changed my mind")
    assert svc.store.latest_verdicts()[0]["note"] == "changed my mind"
    assert generation_log.history(svc)[0]["mesh"]["note"] == "changed my mind"
    with pytest.raises(Invalid) as exc:
        svc_verdicts.record_verdict(svc, job_id, grade=2, note=5)  # type: ignore[arg-type]
    assert exc.value.field == "note"


def test_a_verdict_without_a_note_stores_none_and_a_long_note_is_capped(svc):
    job_id = _done_job(svc)
    svc_verdicts.record_verdict(svc, job_id, grade=1)
    assert svc.store.latest_verdicts()[0]["note"] is None
    svc_verdicts.record_verdict(svc, job_id, grade=1, note="x" * 5000)
    assert len(svc.store.latest_verdicts()[0]["note"]) == svc_verdicts.NOTE_MAX


def test_the_snapshot_is_not_part_of_the_vector(svc):
    job = svc.store.get(_done_job(svc))
    assert "steps" not in vectors.config_vector(job)
    assert vectors.vector_key(vectors.config_vector(job)) == vectors.vector_key(
        vectors.config_vector(job)
    )
    assert "snapshot" not in vectors.VECTOR_PARAMS and "note" not in vectors.VECTOR_PARAMS


def test_a_snapshot_survives_malformed_params(svc):
    snap = vectors.settings_snapshot(
        {"params": {"recipe": "nope", "reference_report": 3, "resolved_recipe": []}}
    )
    assert snap == {"v": 1}


def test_a_database_at_the_previous_version_is_upgraded_in_place(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE verdicts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
            source TEXT NOT NULL, verdict TEXT NOT NULL,
            reasons TEXT NOT NULL DEFAULT '[]', vector TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL, sweep_id TEXT,
            sweep_unit TEXT NOT NULL DEFAULT '', seed INTEGER,
            prompt_hash TEXT NOT NULL DEFAULT '',
            stage TEXT NOT NULL DEFAULT 'model', grade INTEGER);
        CREATE TABLE observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
            sweep_id TEXT, sweep_unit TEXT NOT NULL DEFAULT '', seed INTEGER,
            prompt_hash TEXT NOT NULL DEFAULT '', vector TEXT NOT NULL DEFAULT '{}',
            metrics TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL);
        INSERT INTO verdicts (job_id, source, verdict, created_at, grade)
            VALUES ('gone', 'human', 'accept', 1.0, 3);
        INSERT INTO observations (job_id, created_at) VALUES ('gone', 1.0);
        """
    )
    conn.execute(f"PRAGMA user_version = {len(MIGRATIONS) - 1}")
    conn.commit()
    conn.close()

    store = JobStore(path)
    try:
        row = store.latest_verdicts()[0]
        assert (row["job_id"], row["grade"], row["snapshot"], row["note"]) == (
            "gone", 3, None, None,
        )
        assert store.latest_observations()[0]["snapshot"] is None
        cols = {r[1] for r in store._conn.execute("PRAGMA table_info(verdicts)")}
        assert {"snapshot", "note"} <= cols
        assert store._conn.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    finally:
        store.close()


def test_history_includes_deleted_jobs_and_its_filters_work(svc):
    kept = _done_job(svc)
    svc_verdicts.record_verdict(svc, kept, grade=3)
    other_recipe = {"reference": {**PARAMS["recipe"]["reference"], "style_lora": "other"}}
    gone = _done_job(
        svc, {**PARAMS, "generation_type": "tileset", "style_lora": "other", "recipe": other_recipe}
    )
    svc_verdicts.record_verdict(svc, gone, grade=-3, reasons=["holes"])
    svc.store.delete(gone)
    unjudged = _done_job(svc)

    ids = {r["job_id"] for r in generation_log.history(svc)}
    assert ids == {kept, gone, unjudged}
    by_id = {r["job_id"]: r for r in generation_log.history(svc)}
    assert by_id[gone]["exists"] is False and by_id[kept]["exists"] is True

    assert {r["job_id"] for r in generation_log.history(svc, judged=True)} == {kept, gone}
    assert [r["job_id"] for r in generation_log.history(svc, judged=False)] == [unjudged]
    assert [r["job_id"] for r in generation_log.history(svc, generation_type="tileset")] == [gone]
    assert {r["job_id"] for r in generation_log.history(svc, style_lora="render3d")} == {
        kept, unjudged,
    }
    assert len(generation_log.history(svc, base_model="sdxl_cfg")) == 3
    assert len(generation_log.history(svc, limit=1)) == 1
    assert len(generation_log.history(svc, limit=5, offset=2)) == 1


def test_history_labels_a_retired_model_and_finds_the_thumbnail_only_when_it_exists(svc):
    job_id = _done_job(svc, {**PARAMS, "recipe": {"reference": {"base_model": "turbo"}}})
    svc_verdicts.record_verdict(svc, job_id, grade=1)
    row = generation_log.history(svc)[0]
    assert row["settings_line"].startswith("SDXL-Turbo")
    assert row["thumbnail"] is None
    d = svc.job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "reference.png").write_bytes(b"x")
    assert generation_log.history(svc)[0]["thumbnail"] == str(d / "reference.png")


def test_export_jsonl_writes_valid_lines_with_snapshot_and_note_atomically(svc, tmp_path):
    job_id = _done_job(svc)
    svc_verdicts.record_verdict(svc, job_id, grade=2, note="nice lid")
    queue_mod._observe_finished(svc.store, job_id)
    dest = tmp_path / "out"
    dest.mkdir()

    path = generation_log.export_jsonl(svc, dest)

    assert path.parent == dest
    assert path.name.startswith("realmspinner-generation-log-") and path.suffix == ".jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    tables = {r["table"]: r for r in records}
    assert set(tables) == {"verdict", "observation"}
    assert tables["verdict"]["note"] == "nice lid"
    assert tables["verdict"]["snapshot"]["steps"] == 30
    assert isinstance(tables["verdict"]["vector"], dict)
    assert tables["observation"]["snapshot"]["seed"] == 1204285292
    # Staged then renamed: no staging sibling is left behind.
    assert [p.name for p in dest.iterdir()] == [path.name]


def test_export_jsonl_refuses_a_missing_destination(svc, tmp_path):
    with pytest.raises(Invalid) as exc:
        generation_log.export_jsonl(svc, tmp_path / "nope")
    assert exc.value.field == "dest"
