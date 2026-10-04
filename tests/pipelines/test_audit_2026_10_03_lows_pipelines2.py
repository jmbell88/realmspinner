"""The 2026-10-03 audit's Low findings pipelines-26, -27, -36, -37 and -39."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest
import trimesh

from realmspinner.pipelines import finishing, modelhistory, remesh

SRC = Path(__file__).resolve().parents[2] / "src" / "realmspinner"


# --- pipelines-26 ------------------------------------------------------------


def test_every_from_pretrained_call_in_the_package_passes_local_files_only():
    """The manual and INVARIANTS state "every model load is local_files_only"
    as unconditional. ``recipe_worker`` was the one load that held only by
    ``HF_HUB_OFFLINE`` being inherited and ``model_dir`` happening to exist.

    ``pipelines/acestep`` is excluded: it is the vendored ACE-Step source, whose
    loads take checkpoint paths the app resolves itself and which this audit
    does not own.
    """
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if "acestep" in path.relative_to(SRC).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_pretrained"
                and not any(k.arg == "local_files_only" for k in node.keywords)
            ):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, f"from_pretrained without local_files_only=True: {offenders}"


# --- pipelines-27 ------------------------------------------------------------


def test_every_prompt_version_has_a_ledger_entry():
    """prompt.py says its version counter is bumped "whenever ... changes" and
    keeps a ledger of what each number meant. 9 had no entry, so nothing in the
    tree said what changed relative to 8."""
    from realmspinner.pipelines import prompt

    text = (SRC / "pipelines" / "prompt.py").read_text(encoding="utf-8")
    ledgered = {int(m.group(1)) for m in re.finditer(r"^# (\d+): ", text, re.MULTILINE)}
    missing = [n for n in range(2, prompt.PROMPT_VERSION + 1) if n not in ledgered]
    assert not missing, f"PROMPT_VERSION entries missing from the ledger comment: {missing}"


# --- pipelines-39 ------------------------------------------------------------


def test_the_seam_max_copies_agree():
    """``kernels/pixel/tiling.py`` copies ``SEAM_MAX`` with a comment saying the
    same measurement document governs both; only ``SEAM_DOMINANCE_MAX`` had an
    equality test, so this copy could drift silently."""
    from realmspinner.kernels.pixel import tiling
    from realmspinner.pipelines import seam

    assert tiling.SEAM_MAX == seam.SEAM_MAX


# --- pipelines-37 ------------------------------------------------------------


def _plate(tmp_path: Path, name: str, *, hole: bool, drop_corner: bool = False,
           scale: float = 1.0) -> Path:
    """A 3x3 grid of unit quads in the XY plane; ``hole`` omits the centre quad
    and ``drop_corner`` the (0, 0) one."""
    verts = np.array([[i, j, 0.0] for j in range(4) for i in range(4)]) * scale
    faces = []
    for j in range(3):
        for i in range(3):
            if hole and (i, j) == (1, 1):
                continue
            if drop_corner and (i, j) == (0, 0):
                continue
            a, b, c, d = j * 4 + i, j * 4 + i + 1, (j + 1) * 4 + i + 1, (j + 1) * 4 + i
            faces += [[a, b, c], [a, c, d]]
    path = tmp_path / name
    trimesh.Trimesh(verts, faces, process=False).export(path)
    return path


def test_compare_geometry_measures_identical_filled_and_missing_geometry(tmp_path):
    source = _plate(tmp_path, "src.glb", hole=True)

    same = finishing.compare_geometry(source, _plate(tmp_path, "same.glb", hole=True), 128)
    assert same["measured"] is True
    assert same["worst_lost_coverage"] == 0.0 and same["worst_added_coverage"] == 0.0
    assert all(v["iou"] == 1.0 and v["closed_openings"] == 0.0 for v in same["views"])

    filled = finishing.compare_geometry(source, _plate(tmp_path, "fill.glb", hole=False), 128)
    assert filled["worst_lost_coverage"] == 0.0
    assert filled["worst_added_coverage"] > 0.05
    assert all(v["closed_openings"] == pytest.approx(1.0) for v in filled["views"])

    lossy = finishing.compare_geometry(
        source, _plate(tmp_path, "lossy.glb", hole=True, drop_corner=True), 128
    )
    assert lossy["worst_lost_coverage"] > 0.05
    assert lossy["worst_added_coverage"] == 0.0


def test_finishing_lines_reports_lost_and_added_silhouette_and_closed_openings(tmp_path):
    source = _plate(tmp_path, "src.glb", hole=True)

    lost = remesh.finishing_lines({
        "requested": 100, "triangles": 12,
        "geometry": finishing.compare_geometry(
            source, _plate(tmp_path, "lossy.glb", hole=True, drop_corner=True), 128
        ),
        "tiercheck": {"failures": []},
    })
    lost_text = "\n".join(lost)
    lost_pct = float(re.search(r"lost: ([\d.]+)%", lost_text).group(1))
    assert lost_pct > 5.0
    assert "added: 0.0%" in lost_text
    assert "Source opening pixels filled: 0.0%." in lost_text

    closed = "\n".join(remesh.finishing_lines({
        "requested": 100, "triangles": 12,
        "geometry": finishing.compare_geometry(
            source, _plate(tmp_path, "fill.glb", hole=False), 128
        ),
        "tiercheck": {"failures": []},
    }))
    assert "lost: 0.0%" in closed
    assert "Source opening pixels filled: 100.0%." in closed
    assert "Finished mesh: 12 triangles; requested 100." in closed

    # No opening in the source: the sentence must say unknown, not "0.0%".
    solid = _plate(tmp_path, "solid.glb", hole=False)
    no_openings = "\n".join(remesh.finishing_lines({
        "requested": 100, "triangles": 12,
        "geometry": finishing.compare_geometry(solid, solid, 128),
        "tiercheck": {"failures": []},
    }))
    assert "Opening preservation: unknown" in no_openings
    assert "filled" not in no_openings


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_finishing_lines_for_a_failed_or_cancelled_finish_say_the_source_survives(status):
    lines = remesh.finishing_lines({"status": status, "error": "blender died"})
    assert lines[0] == f"Finishing {status}: blender died"
    assert "source.glb" in lines[1]
    assert not any("Finished mesh" in line for line in lines)


def test_finishing_lines_name_each_tiercheck_failure_and_a_missing_verdict_as_unknown():
    failed = remesh.finishing_lines({
        "triangles": 5000,
        "tiercheck": {"failures": ["UVs were lost", "base color texture is gone"]},
    })
    assert "Material/UV change: UVs were lost" in failed
    assert "Material/UV change: base color texture is gone" in failed
    assert "Material preservation: unknown." not in failed

    unverified = remesh.finishing_lines({"triangles": 5000})
    assert "Material preservation: unknown." in unverified
    assert "Shape preservation: unknown (measurement unavailable)." in unverified
    assert "Finished mesh: 5,000 triangles; requested unknown." in unverified


# --- pipelines-36 ------------------------------------------------------------


def test_two_overlapping_retargets_keep_both_replaced_meshes(svc, monkeypatch):
    """``optimize_job`` read the row before taking the model lock and staged
    from that copy, where ``revert_model`` re-reads inside it. A second
    retarget that waited on the lock therefore staged the same ``n`` as the
    first and overwrote the first's kept mesh under that name.

    The wait is simulated at the lock itself: entering the lock for the first
    time first lets "the other retarget" publish, as it would have while this
    one was blocked.
    """
    from realmspinner.pipelines import optimize
    from realmspinner.service import jobs as svc_jobs

    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "source.glb").write_bytes(b"source-bytes")
    (job_dir / "model.glb").write_bytes(b"model-v1")
    svc.store.set_status(job_id, "done")

    def fake_run(source, out, **_kwargs):
        out.write_bytes(b"model-from-the-waiting-retarget")
        return {"ok": True}

    monkeypatch.setattr(optimize, "run", fake_run)

    real_lock = svc.convert_lock
    interleaved = {"done": False}

    class _Guard:
        def __init__(self, lock) -> None:
            self._lock = lock

        def __enter__(self):
            self._lock.acquire()
            if not interleaved["done"]:
                interleaved["done"] = True
                # The retarget that held the lock first, publishing.
                row = svc.store.get(job_id)
                entries, _ = modelhistory.stage(
                    job_dir, modelhistory.entries_of(row["params"]), row["params"],
                    kind="optimize", geometry=True, detail="first retarget", now=1.0,
                )
                entries = modelhistory.commit(job_dir, entries)
                (job_dir / "model.glb").write_bytes(b"model-from-the-first-retarget")
                svc.store.merge_params(job_id, {"model_history": entries})
            return self

        def __exit__(self, *exc):
            self._lock.release()

    def convert_lock(jid, name):
        lock = real_lock(jid, name)
        return _Guard(lock) if name == modelhistory.MODEL_LOCK else lock

    monkeypatch.setattr(svc, "convert_lock", convert_lock)

    svc_jobs.optimize_job(svc, job_id, profile="raw")

    entries = svc.store.get(job_id)["params"]["model_history"]
    numbers = [e["n"] for e in entries]
    assert len(numbers) == 2 and len(set(numbers)) == 2, numbers
    assert modelhistory.version_path(job_dir, numbers[0]).read_bytes() == b"model-v1"
    assert (
        modelhistory.version_path(job_dir, numbers[1]).read_bytes()
        == b"model-from-the-first-retarget"
    )
