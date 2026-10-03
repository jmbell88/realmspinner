from __future__ import annotations

import json
from collections import Counter

import pytest

from realmspinner.bench import quality, runner, suite


def test_suite_gives_every_family_equal_coverage_and_fixed_seeds():
    doc = suite.load("create-v1")
    assert doc.seeds == (42, 1337)
    assert Counter(i.output_family for i in doc.items) == dict.fromkeys(quality.CRITERIA, 2)


def test_unmeasured_or_ungraded_outputs_do_not_pass():
    records = [{"key": "x", "output_family": "image", "status": "done", "seconds": 1}]
    result = quality.summarise(records)
    assert result["families"]["image"]["unknown"] == 1
    assert result["families"]["image"]["usable_output_rate"] is None
    assert result["equal_family_usable_rate"] is None
    assert quality.readiness("image", {"prompt_fidelity": False}) is False
    assert quality.readiness("image", {"prompt_fidelity": True}) is None


def test_a_large_mesh_corpus_cannot_outweigh_other_families():
    records, grades = [], {}
    for family in quality.CRITERIA:
        for i in range(20 if family == "3d_model" else 1):
            key = f"{family}-{i}"
            records.append({"key": key, "output_family": family, "status": "done"})
            grades[key] = dict.fromkeys(quality.CRITERIA[family], family == "3d_model")
            grades[key]["aesthetic_preference"] = 5
    assert quality.summarise(records, grades)["equal_family_usable_rate"] == pytest.approx(1 / 6)


def make_run(path, *, reference=b"same", family="3d_model"):
    item_dir = path / "items" / "case--s42"
    item_dir.mkdir(parents=True)
    (item_dir / "input.png").write_bytes(reference)
    (item_dir / "model.glb").write_bytes(b"asset")
    (item_dir / "job.json").write_text('{"model":"secret"}')
    runner.append_item(
        path,
        {
            "key": "case--s42",
            "item": "case",
            "seed": 42,
            "status": "done",
            "output_family": family,
            "prompt": "an open ring",
            "reference_sha256": quality.reference_hash(item_dir / "input.png"),
        },
    )


def test_mesh_blind_review_refuses_different_reference_images(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    make_run(left)
    make_run(right, reference=b"different")
    with pytest.raises(ValueError, match="same reference"):
        quality.blind_review(left, right, tmp_path / "review")
    assert not (tmp_path / "review").exists()


def test_blind_packet_omits_provenance_and_imports_readiness_separately(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    make_run(left)
    make_run(right)
    review_dir = quality.blind_review(left, right, tmp_path / "review")
    review_path = review_dir / "review.json"
    rows = json.loads(review_path.read_text())
    assert len(rows) == 2
    assert not list((review_dir / "assets").rglob("job.json"))
    assert "left" not in review_path.read_text() and "right" not in review_path.read_text()
    for row in rows:
        assert all(value is None for value in row["acceptance"].values())
        row["acceptance"] = dict.fromkeys(row["acceptance"], True)
        row["aesthetic_preference"] = 1
    review_path.write_text(json.dumps(rows))
    quality.import_review(review_dir)
    for path in (left, right):
        result = json.loads((path / "quality.json").read_text())
        assert result["families"]["3d_model"]["usable_output_rate"] == 1
        assert result["equal_family_usable_rate"] is None
        assert quality.report(path)["families"]["3d_model"]["usable_output_rate"] == 1


def test_edited_paired_reference_refuses_resume(tmp_path):
    from realmspinner.bench import manifest, recipe
    from realmspinner.config import Config

    source = tmp_path / "references"
    path = source / "items" / "mesh-lantern--s42" / "input.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original")
    kwargs = dict(
        stage="model", started="t", ids=("mesh-lantern",), seeds=(42,), reference_run=source
    )
    config = Config(data_dir=tmp_path / "assets", bench_dir=tmp_path / "bench")
    s, r = suite.load("create-v1"), recipe.load("sdxl-cfg-raw")
    _dir, _units, old = runner.plan_run(config, s, r, **kwargs)
    path.write_bytes(b"edited")
    _dir, _units, new = runner.plan_run(config, s, r, **kwargs)
    with pytest.raises(manifest.NotResumable, match="reference images changed"):
        manifest.assert_resumable(old, new)
