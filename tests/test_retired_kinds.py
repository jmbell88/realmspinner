"""The retired job kinds -- ``charsheet``, ``sheet`` and ``pixel_sheet`` -- stay retired.

The character-sheet pipeline left Realmspinner with Poser (the 2026-10-13 Clay rig
plan, Tranche 5; ``src/realmspinner/retired.py`` says what that means for a row that
is still in someone's library). This is the **backend half** of the gate: each
table that decides what a *live* kind is offered -- the phase tables, the VRAM
estimate, the follow-up products, the cancel-discard arms, the dispatch chain, the
publisher list -- carries no row for a retired kind, so one cannot creep back in
through the table nobody re-read. The studio half (the Library card, the palette,
the openers) lives beside the studio.

Every test fails against the tree as it stood before the removal: the sweep is the
point, and a retired kind found in a table is the one-line regression this file
exists to name.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import sqlite3
from pathlib import Path

import pytest

from realmspinner import followups, progress, retired, vram
from realmspinner.config import Config
from realmspinner.db import JobStore, retire_stuck_rows
from realmspinner.queue import Worker

SRC = Path(__file__).resolve().parents[1] / "src" / "realmspinner"

#: Every kind a row can still carry today and be run. A new kind is a sweep of the
#: stage-keyed tables (``dev/INVARIANTS.md``); it is named here so the first test
#: below fails loudly if a retired name is ever re-used for one of them.
LIVE_KINDS = (
    "text", "image", "rig", "sprite_synthesis", "retexture", "tile_sheet", "remesh",
    "lora_train", "music", "separate",
)


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    yield w
    store.close()


# --- what retired means --------------------------------------------------------


def test_is_retired_is_true_for_exactly_the_three_sheet_kinds():
    assert frozenset({"charsheet", "sheet", "pixel_sheet"}) == retired.RETIRED_KINDS
    for kind in retired.RETIRED_KINDS:
        assert retired.is_retired(kind) is True, kind
    for kind in LIVE_KINDS:
        assert retired.is_retired(kind) is False, kind
    # Not a kind at all: a row with no kind (or a junk one) is not "retired".
    for junk in (None, "", 7, ["sheet"], b"sheet"):
        assert retired.is_retired(junk) is False, junk


def test_the_note_says_what_happened_to_the_row():
    assert "removed" in retired.NOTE.lower()
    assert "library" in retired.NOTE.lower()


# --- the stage-keyed tables ----------------------------------------------------


def test_no_retired_kind_has_a_progress_phase_table():
    assert not set(progress._PHASES_BY_KIND) & retired.RETIRED_KINDS
    for kind in retired.RETIRED_KINDS:
        # Unregistered kinds fall back to the image table, which is exactly what
        # nothing should rely on -- so it is also the proof the row is gone.
        assert progress.phases_for(kind) is progress.PHASES_IMAGE, kind
    for name in ("PHASES_SHEET", "PHASES_PIXEL_SHEET", "PHASES_CHARSHEET"):
        assert not hasattr(progress, name), name


def test_no_retired_kind_is_priced_by_vram():
    for kind in retired.RETIRED_KINDS:
        for exclusive in (True, False):
            assert vram.estimate_parts(kind, "model", {}, exclusive=exclusive) == (0.0, 0.0), kind
    # And the pixel sheet's checkpoint term does not live on in the source: the
    # estimate has no branch that names a retired kind.
    tree = ast.parse((SRC / "vram.py").read_text(encoding="utf-8"))
    compared: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "kind"
        ):
            for comparator in node.comparators:
                compared |= _string_constants(comparator)
    assert {"retexture", "tile_sheet", "lora_train"} <= compared  # the scan works
    assert not compared & retired.RETIRED_KINDS


def test_no_retired_kind_is_a_followup_product_label_or_stage():
    for table in (followups.PRODUCTS, followups.LABELS, followups.APPLICABLE_STAGES):
        assert not set(table) & retired.RETIRED_KINDS, table


def test_the_publisher_list_names_no_retired_stage():
    """``PUBLISHERS`` is a hand list that fails open, and has gone stale twice: a
    row naming a stage that no longer exists would be skipped, not failed."""
    from test_job_durability import PUBLISHERS

    for module, func, _publish in PUBLISHERS:
        assert module != "realmspinner._q_troupe", (module, func)
        assert func not in {"_sheet", "_charsheet", "_pixel_sheet"}, (module, func)
        # Every remaining row still names a stage that exists.
        assert importlib.import_module(module) is not None


# --- the dispatch chain and the cancel arm -------------------------------------


def _string_constants(node: ast.AST) -> set[str]:
    return {
        n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def _kinds_compared(node: ast.AST) -> set[str]:
    """Every string a ``job["kind"]`` is compared with, ``==`` or ``in (...)``."""
    found: set[str] = set()
    for n in ast.walk(node):
        if not isinstance(n, ast.Compare):
            continue
        left = n.left
        if not (
            isinstance(left, ast.Subscript)
            and isinstance(left.slice, ast.Constant)
            and left.slice.value == "kind"
        ):
            continue
        for comparator in n.comparators:
            found |= _string_constants(comparator)
    return found


def _function(path: Path, name: str) -> ast.AST:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )


def test_the_worker_dispatches_no_retired_kind_and_carries_no_retired_stage():
    generate = _function(SRC / "_q_generate.py", "_generate")
    dispatched = _kinds_compared(generate)
    assert {"rig", "sprite_synthesis", "tile_sheet", "music"} <= dispatched  # the scan works
    assert not dispatched & retired.RETIRED_KINDS
    for name in (
        "_sheet", "_charsheet", "_pixel_sheet",
        "_maybe_queue_charsheet", "_maybe_queue_sheet_after_rig",
        "_render_charsheet", "_charsheet_roots",
    ):
        assert not hasattr(Worker, name), name
    import realmspinner.queue as queue_mod

    assert not hasattr(queue_mod, "TroupeOps")
    assert not hasattr(queue_mod, "_sheet_root_offsets")
    assert importlib.util.find_spec("realmspinner._q_troupe") is None


def test_the_discard_arm_names_no_retired_kind_and_touches_a_retired_row_never(worker):
    """``_discard_artifacts`` falls through to a generic mesh-filename arm for a
    kind it has no branch for. A retired row must not reach it: the user's work is
    theirs until they trash it, so a cancel (or a failed run) deletes nothing."""
    discard = _function(SRC / "_q_jobs.py", "_discard_artifacts")
    arms = _kinds_compared(discard)
    assert {"rig", "retexture", "separate", "tile_sheet"} <= arms  # the scan works
    assert not arms & retired.RETIRED_KINDS

    source = worker.store.create("text", "a knight", {"seed": 1})
    source_dir = worker.config.job_dir(source)
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "model.glb").write_bytes(b"the mesh")
    sheets = source_dir / "sheets"
    sheets.mkdir()
    (sheets / "abcdef012345.png").write_bytes(b"an old sheet")
    (sheets / ".abcdef012345.png.tmp").write_bytes(b"a staged one")
    for kind in sorted(retired.RETIRED_KINDS):
        row_id = worker.store.create(
            kind, "a knight", {"source_job": source, "sheet_id": "abcdef012345"}
        )
        own = worker.config.job_dir(row_id)
        own.mkdir(parents=True, exist_ok=True)
        (own / "model.glb").write_bytes(b"its own file")
        (own / "input.png").write_bytes(b"its own image")
        worker._discard_artifacts(worker.store.get(row_id))
        assert (own / "model.glb").read_bytes() == b"its own file", kind
        assert (own / "input.png").read_bytes() == b"its own image", kind
    assert (source_dir / "model.glb").read_bytes() == b"the mesh"
    assert (sheets / "abcdef012345.png").read_bytes() == b"an old sheet"
    assert (sheets / ".abcdef012345.png.tmp").read_bytes() == b"a staged one"


@pytest.mark.parametrize("kind", sorted(retired.RETIRED_KINDS))
async def test_a_retired_row_that_reaches_the_worker_is_an_error_with_the_note(worker, kind):
    source = worker.store.create("text", "a knight", {"seed": 1})
    source_dir = worker.config.job_dir(source)
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "model.glb").write_bytes(b"fake-glb")
    (source_dir / "rig.glb").write_bytes(b"fake-rig")
    worker.store.set_status(source, "done")

    job_id = worker.store.create(
        kind, None, {"source_job": source, "sheet_id": "abcdef012345", "frame_size": 64}
    )
    await worker._process(worker.store.get(job_id))

    row = worker.store.get(job_id)
    assert row["status"] == "error"
    assert row["error"] == retired.NOTE
    # Nothing was rendered, published or minted, and the source is untouched.
    assert not (source_dir / "sheets").exists()
    assert (source_dir / "model.glb").read_bytes() == b"fake-glb"
    assert [r["id"] for r in worker.store.list(limit=50) if r["id"] not in (source, job_id)] == []


# --- opening the store ---------------------------------------------------------


def _raw_row(path: Path, job_id: str, kind: str, status: str) -> None:
    """A row an earlier build could have written, straight into the file: the
    store's own ``create`` is what a current build would use."""
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "INSERT INTO jobs (id, kind, status, prompt, params, created_at) "
            "VALUES (?, ?, ?, '', '{}', 1.0)",
            (job_id, kind, status),
        )
        conn.commit()
    finally:
        conn.close()


def test_opening_the_store_flips_a_stuck_retired_row_to_error_and_leaves_its_files(tmp_path):
    path = tmp_path / "assets" / "jobs.sqlite"
    store = JobStore(path)
    store.close()
    kept = tmp_path / "assets" / "aaaaaaaaaaaa"
    kept.mkdir(parents=True)
    (kept / "sheets").mkdir()
    (kept / "sheets" / "bbbbbbbbbbbb.png").write_bytes(b"a rendered sheet")
    stuck = {
        "aaaaaaaaaaa1": ("charsheet", "queued"),
        "aaaaaaaaaaa2": ("sheet", "running"),
        "aaaaaaaaaaa3": ("pixel_sheet", "queued"),
        "aaaaaaaaaaa4": ("pixel_sheet", "done"),
        "aaaaaaaaaaa5": ("rig", "queued"),
        "aaaaaaaaaaa6": ("text", "running"),
    }
    for job_id, (kind, status) in stuck.items():
        _raw_row(path, job_id, kind, status)

    store = JobStore(path)
    try:
        flipped = {j: store.get(j) for j in ("aaaaaaaaaaa1", "aaaaaaaaaaa2", "aaaaaaaaaaa3")}
        for job_id, row in flipped.items():
            assert row["status"] == "error", job_id
            assert row["error"] == retired.NOTE, job_id
            assert row["finished_at"], job_id
        # A finished row of a retired kind is history, not stuck; a live kind is
        # not the migration's business at all (``reconcile_startup`` has running ones).
        assert store.get("aaaaaaaaaaa4")["status"] == "done"
        assert store.get("aaaaaaaaaaa5")["status"] == "queued"
        assert store.get("aaaaaaaaaaa6")["status"] == "running"
        assert store.next_queued()["id"] == "aaaaaaaaaaa5"
    finally:
        store.close()
    # Files are never deleted by opening the store.
    assert (kept / "sheets" / "bbbbbbbbbbbb.png").read_bytes() == b"a rendered sheet"


def test_retiring_is_idempotent_and_reports_how_many_rows_it_changed(tmp_path):
    path = tmp_path / "jobs.sqlite"
    store = JobStore(path)
    store.close()
    _raw_row(path, "cccccccccccc", "charsheet", "queued")
    _raw_row(path, "dddddddddddd", "sheet", "running")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        assert retire_stuck_rows(conn) == 2
        assert retire_stuck_rows(conn) == 0
    finally:
        conn.close()


# --- the doors -----------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(retired.RETIRED_KINDS))
def test_a_retired_row_is_never_offered_a_reroll_and_the_door_refuses_it(svc, kind):
    from realmspinner.service import jobs as svc_jobs
    from realmspinner.service._jobs_resubmit import rerollable, rerollable_reason
    from realmspinner.service.errors import Invalid

    job_id = svc.store.create(
        kind, "a knight", {"source_job": "0123456789ab", "sheet_id": "abcdef012345"}
    )
    svc.store.set_status(job_id, "done")
    row = svc.store.get(job_id)
    assert rerollable(row) is False
    assert rerollable_reason(row) == retired.NOTE
    for mode in ("reroll", "remesh"):
        with pytest.raises(Invalid) as refused:
            svc_jobs.rerun_job(svc, job_id, mode=mode)
        assert str(refused.value) == retired.NOTE
    assert len(svc.store.list(limit=50)) == 1


def test_the_service_has_no_door_left_that_makes_or_exports_a_sheet():
    for module in ("sheets", "troupe", "poses"):
        assert importlib.util.find_spec(f"realmspinner.service.{module}") is None, module
    assert importlib.util.find_spec("realmspinner.pipelines.sheetcheck") is None

    from realmspinner.service import _jobs_create, characters, export

    assert "troupe" not in inspect.signature(_jobs_create.create_job).parameters
    for name in ("export_package", "export_frames", "sheet_preview_png", "_plan_cells"):
        assert not hasattr(characters, name), name
    assert "has_sheets" not in characters.ASSET_FILTERS
    assert set(export.CHARACTER_EXPORTS) == {"animated_glb", "godot_scene"}


def test_the_character_options_carry_no_sheet_ladders(svc):
    from realmspinner.service import characters

    assert "troupe" not in characters.character_options(svc)


def test_the_rig_store_keeps_the_qa_sheet_and_nothing_else_of_the_sheet_storage():
    from realmspinner.kernels.rig import store

    assert callable(store.rig_qa_path) and callable(store.rig_qa_png_path)
    for name in (
        "sheet_dir", "sheet_path", "sheet_png_path", "sheet_pixel_path",
        "sheet_pixel_png_path", "read_sheet", "read_sheet_pixel", "list_sheets",
        "delete_sheet", "MAX_SHEETS", "SHEET_DIR_NAME", "MAX_SHEET_NAME",
    ):
        assert not hasattr(store, name), name
