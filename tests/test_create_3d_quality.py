"""create-3d-v1: the 3D measurement harness.

The claims here are about what the harness refuses and what it will not say:
a run that names every missing image at once, a rubric that never changes a
create-v1 report, and a time-to-accepted that stays unknown rather than being
guessed. Nothing in this file touches a GPU or the network.
"""

from __future__ import annotations

import json
import socket

import pytest

from realmspinner.bench import manifest, quality, recipe, runner, suite
from realmspinner.config import Config

SUITE = "create-3d-v1"
CATEGORIES_3D = (
    "solid_prop", "thin_weapon", "open_structure", "hard_surface", "humanoid", "creature"
)


def _config(tmp_path):
    return Config(data_dir=tmp_path / "assets", bench_dir=tmp_path / "bench")


def _plan(tmp_path, **kwargs):
    kwargs.setdefault("stage", "model")
    return runner.plan_run(
        _config(tmp_path), suite.load(SUITE), recipe.load("sdxl-cfg-raw"), started="t", **kwargs
    )


def _supply(root, items, *, only=None):
    for item in items:
        if item.input_mode == "image" and (only is None or item.reference_image in only):
            path = root / item.reference_image
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png:" + item.reference_image.encode())


def test_the_corpus_has_six_categories_of_three_images_and_one_prompt():
    doc = suite.load(SUITE)
    assert doc.seeds == (42, 1337, 7)
    by_category = doc.by_category()
    assert tuple(by_category) == CATEGORIES_3D
    for items in by_category.values():
        assert sorted(i.input_mode for i in items) == ["image", "image", "image", "text"]
    for item in doc.items:
        assert item.rubric == "3d_v2" and item.output_family == "3d_model"
        assert (item.reference_image is None) == (item.input_mode == "text")


@pytest.mark.parametrize(
    "bad", ["/etc/x.png", "C:/x.png", "..\\x.png", "a/../../x.png", "", "D:\\refs\\x.png"]
)
def test_an_image_item_cannot_name_a_path_outside_the_references_root(tmp_path, bad):
    raw = {
        "seeds": [1],
        "items": [{"id": "a", "prompt": "x", "category": "prop", "input": "image",
                   "reference_image": bad}],
    }
    with pytest.raises(ValueError, match="relative reference_image"):
        suite.parse(raw, tmp_path / "s.json")


def test_a_text_item_cannot_carry_a_reference_and_a_rubric_must_exist(tmp_path):
    base = {"id": "a", "prompt": "x", "category": "prop"}
    with pytest.raises(ValueError, match="names a reference_image"):
        suite.parse({"seeds": [1], "items": [{**base, "reference_image": "a.png"}]}, tmp_path / "s")
    with pytest.raises(ValueError, match="unknown rubric"):
        suite.parse({"seeds": [1], "items": [{**base, "rubric": "nope"}]}, tmp_path / "s")


def test_missing_reference_images_are_all_named_in_one_refusal(tmp_path):
    root = tmp_path / "refs"
    _supply(root, suite.load(SUITE).items, only={"solid_prop/01.png"})
    with pytest.raises(ValueError) as caught:
        _plan(tmp_path, references=root)
    text = str(caught.value)
    assert "17 reference image(s) missing" in text
    for name in ("solid_prop/02.png", "thin_weapon/01.png", "creature/03.png"):
        assert name in text
    assert "solid_prop/01.png" not in text


def test_the_default_references_root_is_under_the_bench_dir_not_the_repo(tmp_path):
    with pytest.raises(ValueError) as caught:
        _plan(tmp_path)
    assert str(tmp_path / "bench" / "references" / SUITE) in str(caught.value)


def test_skip_missing_runs_what_exists_and_lists_the_rest(tmp_path):
    root = tmp_path / "refs"
    _supply(root, suite.load(SUITE).items, only={"solid_prop/01.png"})
    _dir, todo, doc = _plan(tmp_path, references=root, skip_missing=True)
    images = [u for u in todo if u.item.input_mode == "image"]
    assert [u.item.id for u in images] == ["solid_prop-img-01"] * 3
    assert all(u.reference == root / "solid_prop" / "01.png" for u in images)
    assert len([u for u in todo if u.item.input_mode == "text"]) == 6 * 3
    assert len(doc["skipped"]) == 17 * 3
    assert set(doc["references"]) == {u.key for u in images}


def test_a_changed_reference_image_refuses_resume(tmp_path):
    root = tmp_path / "refs"
    _supply(root, suite.load(SUITE).items)
    _d, _t, old = _plan(tmp_path, references=root)
    (root / "humanoid" / "02.png").write_bytes(b"re-photographed")
    _d, _t, new = _plan(tmp_path, references=root)
    with pytest.raises(manifest.NotResumable, match="reference images changed"):
        manifest.assert_resumable(old, new)


def test_attempt_is_the_seeds_slot_in_the_suite_whichever_seeds_a_run_asks_for(tmp_path):
    root = tmp_path / "refs"
    _supply(root, suite.load(SUITE).items)
    _d, todo, _doc = _plan(tmp_path, references=root, ids=("creature-txt-01",), seeds=(7, 1337))
    assert {u.seed: u.attempt for u in todo} == {7: 3, 1337: 2}


def test_reference_bytes_prefers_the_frozen_paired_run(tmp_path):
    root = tmp_path / "refs"
    _supply(root, suite.load(SUITE).items)
    _d, todo, _doc = _plan(tmp_path, references=root, ids=("humanoid-img-01",), seeds=(42,))
    unit = todo[0]
    assert runner.reference_bytes(unit, None) == b"png:humanoid/01.png"
    paired = tmp_path / "paired"
    (paired / "items" / unit.key).mkdir(parents=True)
    (paired / "items" / unit.key / "input.png").write_bytes(b"frozen")
    assert runner.reference_bytes(unit, paired) == b"frozen"
    text = suite.load(SUITE).filter(ids=("humanoid-txt-01",))[0]
    assert runner.reference_bytes(runner.Unit(text, 42), None) is None


# ---- rubric and review ------------------------------------------------------


def test_animation_suitability_is_asked_only_of_things_that_get_rigged():
    for category in CATEGORIES_3D:
        names = quality.criteria_for("3d_model", "3d_v2", category)
        assert ("animation_suitability" in names) == (category in ("humanoid", "creature"))
        assert names[:5] == quality.RUBRICS["3d_v2"][:5]
    assert quality.criteria_for("3d_model") == quality.CRITERIA["3d_model"]


def test_a_static_prop_is_accepted_without_an_animation_grade():
    grades = dict.fromkeys(quality.criteria_for("3d_model", "3d_v2", "solid_prop"), True)
    assert quality.readiness("3d_model", grades, "3d_v2", "solid_prop") is True
    assert quality.readiness("3d_model", grades, "3d_v2", "humanoid") is None


def _record(item, attempt, *, status="done", seconds=60.0, category="solid_prop", **extra):
    return {
        "key": f"{item}--s{attempt}", "item": item, "category": category, "seed": attempt,
        "attempt": attempt, "status": status, "seconds": seconds, "output_family": "3d_model",
        "rubric": "3d_v2", "input_mode": "image", **extra,
    }


def _accept(*keys, repair=None):
    names = quality.criteria_for("3d_model", "3d_v2", "solid_prop")
    return {k: {**dict.fromkeys(names, True), "repair_minutes": repair} for k in keys}


def test_time_to_accepted_sums_attempts_up_to_first_accept_plus_repair():
    records = [_record("a", 1, seconds=100.0), _record("a", 2, seconds=70.0),
               _record("a", 3, seconds=500.0)]
    names = quality.criteria_for("3d_model", "3d_v2", "solid_prop")
    grades = {"a--s1": dict.fromkeys(names, False), **_accept("a--s2", "a--s3", repair=2)}
    overall = quality.attempt_metrics(records, grades)["overall"]
    # 100 + 70 seconds of generation through the first accept, plus two minutes by hand;
    # the 500 s third attempt came after acceptance and is not charged to it.
    assert overall["time_to_accepted_seconds"] == 100 + 70 + 120
    assert overall["first_attempt"] == 0.0 and overall["best_of_3"] == 1.0
    assert overall["generation_seconds_total"] == 670.0


def test_time_to_accepted_stays_unknown_when_the_repair_time_or_a_duration_is_unknown():
    records = [_record("a", 1), _record("b", 1, seconds=None)]
    grades = _accept("a--s1", "b--s1", repair=None)
    assert quality.attempt_metrics(records, grades)["overall"]["time_to_accepted_seconds"] is None
    grades = _accept("a--s1", "b--s1", repair=0)
    overall = quality.attempt_metrics(records, grades)["overall"]
    assert overall["time_to_accepted_seconds"] == 60.0 and overall["time_to_accepted_known"] == 1


def test_ungraded_attempts_are_unknown_not_failures_and_failed_ones_are_failures():
    ungraded = quality.attempt_metrics([_record("a", 1)], {})["overall"]
    assert ungraded["first_attempt"] is None and ungraded["best_of_3"] is None
    failed = quality.attempt_metrics([_record("a", 1, status="error")], {})["overall"]
    assert failed["first_attempt"] == 0.0 and failed["best_of_3"] == 0.0


def test_best_of_3_ignores_a_fourth_slot_and_splits_by_input_mode_and_category():
    records = [_record("a", n) for n in (1, 2, 3, 4)] + [
        _record("t", 1, input_mode="text", category="humanoid")
    ]
    grades = {"a--s4": _accept("a--s4")["a--s4"]}
    for n in (1, 2, 3):
        grades[f"a--s{n}"] = dict.fromkeys(quality.RUBRICS["3d_v2"], False)
    out = quality.attempt_metrics(records, grades)
    assert out["by_input_mode"]["image"]["best_of_3"] == 0.0
    assert out["by_input_mode"]["text"]["items"] == 1
    assert set(out["by_category"]) == {"solid_prop", "humanoid"}


def _make_run(path, *, item="case", attempt=1, rubric=None, category="solid_prop",
              reference=b"same", seconds=30.0):
    key = f"{item}--s{40 + attempt}"
    item_dir = path / "items" / key
    item_dir.mkdir(parents=True)
    (item_dir / "input.png").write_bytes(reference)
    (item_dir / "model.glb").write_bytes(b"glTF" + b"\0" * 8)
    runner.append_item(path, {
        "key": key, "item": item, "category": category, "seed": 40 + attempt,
        "attempt": attempt, "status": "done", "seconds": seconds,
        "output_family": "3d_model", "prompt": "a thing", "rubric": rubric,
        "input_mode": "image", "reference_sha256": quality.reference_hash(item_dir / "input.png"),
    })
    return key


def test_a_rubric_packet_imports_its_own_criteria_and_repair_minutes(tmp_path):
    left, right = tmp_path / "l", tmp_path / "r"
    key = _make_run(left, rubric="3d_v2")
    _make_run(right, rubric="3d_v2")
    review_dir = quality.blind_review(left, right, tmp_path / "review")
    rows = json.loads((review_dir / "review.json").read_text())
    for row in rows:
        assert "animation_suitability" not in row["acceptance"]
        assert row["rubric"] == "3d_v2" and row["repair_minutes"] is None
        row["acceptance"] = dict.fromkeys(row["acceptance"], True)
        row["repair_minutes"] = 3
    (review_dir / "review.json").write_text(json.dumps(rows))
    quality.import_review(review_dir)
    graded = json.loads((left / "human-grades.json").read_text())[key]
    assert graded["repair_minutes"] == 3 and graded["silhouette"] is True
    result = json.loads((left / "quality.json").read_text())
    assert result["families"]["3d_model"]["usable_output_rate"] == 1
    assert result["attempts"]["overall"]["time_to_accepted_seconds"] == 30 + 180


def test_a_bad_repair_value_is_refused_on_import(tmp_path):
    left, right = tmp_path / "l", tmp_path / "r"
    _make_run(left, rubric="3d_v2")
    _make_run(right, rubric="3d_v2")
    review_dir = quality.blind_review(left, right, tmp_path / "review")
    rows = json.loads((review_dir / "review.json").read_text())
    rows[0]["repair_minutes"] = -5
    (review_dir / "review.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="repair_minutes"):
        quality.import_review(review_dir)


def test_old_review_without_rubric_still_imports(tmp_path):
    left, right = tmp_path / "l", tmp_path / "r"
    _make_run(left)
    _make_run(right)
    review_dir = quality.blind_review(left, right, tmp_path / "review")
    rows = json.loads((review_dir / "review.json").read_text())
    assert all("rubric" not in row and "repair_minutes" not in row for row in rows)
    # A mapping written before rubrics existed carries no criteria list.
    mapping = json.loads((review_dir / "mapping.private.json").read_text())
    for entry in mapping.values():
        entry.pop("criteria")
    (review_dir / "mapping.private.json").write_text(json.dumps(mapping))
    for row in rows:
        row["acceptance"] = dict.fromkeys(quality.CRITERIA["3d_model"], True)
    (review_dir / "review.json").write_text(json.dumps(rows))
    quality.import_review(review_dir)
    assert json.loads((left / "quality.json").read_text())["families"]["3d_model"][
        "usable_output_rate"
    ] == 1


def test_create_v1_equal_family_rate_unchanged_by_rubrics(tmp_path):
    records, grades = [], {}
    for family in quality.CRITERIA:
        key = f"{family}--s42"
        records.append({"key": key, "item": family, "output_family": family, "status": "done"})
        grades[key] = dict.fromkeys(quality.CRITERIA[family], True)
    assert quality.summarise(records, grades)["equal_family_usable_rate"] == 1.0
    # And a run with no attempt fields reports exactly the keys it always did.
    for record in records:
        runner.append_item(tmp_path, record)
    (tmp_path / "human-grades.json").write_text(json.dumps(grades))
    assert set(quality.report(tmp_path)) == {
        "families", "equal_family_usable_rate", "aesthetic_preference"
    }


# ---- importing another service's meshes -------------------------------------


def _glb(path, body=b"\0" * 12):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"glTF" + body)


def test_external_import_files_a_run_that_blind_review_pairs(tmp_path):
    ours = tmp_path / "ours"
    _make_run(ours, attempt=1, rubric="3d_v2")
    _make_run(ours, attempt=2, rubric="3d_v2", reference=b"same")
    glbs = tmp_path / "tripo"
    _glb(glbs / "case" / "attempt-1.glb")
    (glbs / "timings.json").write_text(json.dumps({"case/attempt-1.glb": 12.5}))
    _glb(glbs / "stray" / "attempt-1.glb")

    result = quality.import_external(ours, glbs, tmp_path / "ext", "tripo")

    assert result["imported"] == ["case/attempt-1.glb"]
    assert result["missing"] == ["case/attempt-2.glb"]
    assert result["unmatched"] == ["stray/attempt-1.glb"]
    record = runner.latest_items(tmp_path / "ext")["case--s41"]
    assert (record["attempt"], record["seconds"], record["backend"]) == (1, 12.5, "tripo")
    assert record["reference_sha256"] == runner.latest_items(ours)["case--s41"]["reference_sha256"]

    review_dir = quality.blind_review(ours, tmp_path / "ext", tmp_path / "review")
    # One paired unit is two anonymous cases, A and B; the one pair the files cover.
    rows = json.loads((review_dir / "review.json").read_text())
    assert {row["id"][:8] for row in rows} == {"case-001"} and len(rows) == 2
    assert "tripo" not in (review_dir / "review.json").read_text()


def test_external_import_refuses_a_file_that_is_not_a_glb_or_is_too_big(tmp_path, monkeypatch):
    ours = tmp_path / "ours"
    _make_run(ours)
    bad = tmp_path / "bad"
    (bad / "case").mkdir(parents=True)
    (bad / "case" / "attempt-1.glb").write_bytes(b"<html>login</html>")
    with pytest.raises(ValueError, match="not a binary glTF"):
        quality.import_external(ours, bad, tmp_path / "o1", "x")
    assert not (tmp_path / "o1" / "items").exists()

    big = tmp_path / "big"
    _glb(big / "case" / "attempt-1.glb", b"\0" * 64)
    monkeypatch.setattr(quality, "EXTERNAL_MAX_BYTES", 16)
    with pytest.raises(ValueError, match="cap"):
        quality.import_external(ours, big, tmp_path / "o2", "x")


def test_external_import_makes_no_network_call(tmp_path, monkeypatch):
    ours = tmp_path / "ours"
    _make_run(ours)
    glbs = tmp_path / "g"
    _glb(glbs / "case" / "attempt-1.glb")

    def boom(*_a, **_k):
        raise AssertionError("import_external touched the network")

    for name in ("socket", "create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, boom)
    assert quality.import_external(ours, glbs, tmp_path / "ext", "meshy")["imported"]


def test_external_import_with_nothing_matching_leaves_no_run_behind(tmp_path):
    ours = tmp_path / "ours"
    _make_run(ours)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no <item>/attempt-<n>.glb"):
        quality.import_external(ours, empty, tmp_path / "ext", "meshy")
    assert not (tmp_path / "ext").exists()
