"""The 2026-10-14 audit, finding pipelines-24: bench documents are staged, not truncated."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from realmspinner.bench import manifest, quality


def _review_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    run.mkdir()
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    family = next(iter(quality.CRITERIA))
    criteria = quality.CRITERIA[family]
    (review_dir / "mapping.private.json").write_text(
        json.dumps({"c1": {"run": str(run), "key": "k1"}}), "utf-8"
    )
    (review_dir / "review.json").write_text(
        json.dumps(
            [
                {
                    "id": "c1",
                    "family": family,
                    "acceptance": dict.fromkeys(criteria, True),
                    "aesthetic_preference": None,
                }
            ]
        ),
        "utf-8",
    )
    return run, review_dir


def _fail_write_text(self: Path, *a, **k):
    # Model a kill mid-write: the target is truncated and then the write dies.
    with open(self, "w", encoding="utf-8") as fh:
        fh.write("{")
    raise OSError("disk full")


def test_import_review_leaves_the_old_grades_intact_when_the_write_fails(tmp_path, monkeypatch):
    run, review_dir = _review_fixture(tmp_path)
    grades = run / "human-grades.json"
    old = json.dumps({"older": {"a": True}}, indent=2)
    grades.write_text(old, "utf-8")

    monkeypatch.setattr(Path, "write_text", _fail_write_text)
    monkeypatch.setattr(quality, "report", lambda *a, **k: {})
    with pytest.raises(OSError):
        quality.import_review(review_dir)
    monkeypatch.undo()

    assert grades.read_text("utf-8") == old
    assert not [p for p in run.iterdir() if p.name != "human-grades.json"]


def test_write_manifest_leaves_the_old_manifest_intact_when_the_write_fails(tmp_path, monkeypatch):
    path = tmp_path / manifest.FILENAME
    path.write_text('{"version": 1}', "utf-8")
    monkeypatch.setattr(Path, "write_text", _fail_write_text)
    with pytest.raises(OSError):
        manifest.write_manifest(tmp_path, {"version": 2})
    monkeypatch.undo()
    assert path.read_text("utf-8") == '{"version": 1}'


def test_write_scores_leaves_the_old_scores_intact_when_the_write_fails(tmp_path, monkeypatch):
    from realmspinner.bench import score

    path = tmp_path / score.FILENAME
    path.write_text('{"version": 1}', "utf-8")
    monkeypatch.setattr(Path, "write_text", _fail_write_text)
    with pytest.raises(OSError):
        score.write_scores(tmp_path, {"version": 2})
    monkeypatch.undo()
    assert path.read_text("utf-8") == '{"version": 1}'


def test_write_report_leaves_the_old_calibration_intact_when_the_write_fails(
    tmp_path, monkeypatch
):
    from realmspinner.bench import calibrate

    path = tmp_path / "calibrate.json"
    path.write_text('{"results": []}', "utf-8")
    monkeypatch.setattr(Path, "write_text", _fail_write_text)
    monkeypatch.setattr(calibrate.metrics_mod, "available", lambda config=None: [])
    with pytest.raises(OSError):
        calibrate.write_report(tmp_path, [])
    monkeypatch.undo()
    assert path.read_text("utf-8") == '{"results": []}'


def test_no_bench_document_is_written_in_place():
    # The 2026-10-14 audit, finding pipelines-24, remainder: scores.json,
    # purged.json, job.json, calibrate.json and the views sidecar were still
    # written in place, so a kill mid-write truncated a run's record. This
    # sweeps the package so the next writer cannot go back to ``write_text``.
    import realmspinner.bench as bench

    root = Path(bench.__file__).parent
    offenders = [
        f"{path.name}:{number}"
        for path in sorted(root.glob("*.py"))
        for number, line in enumerate(path.read_text("utf-8").splitlines(), 1)
        if ".write_text(json.dumps" in line
    ]
    assert offenders == []
