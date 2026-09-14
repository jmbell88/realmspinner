"""Tests for ``train/make_arm.py`` -- the ablation arm builder that pairs a
dataset (optionally with some families dropped) against a system-prompt
card. Run with the rest of this package's tests, not the main suite (see
``README.md``'s "Running the tests" section):

    uv run pytest training/clay-assistant/tests -n 0 -p no:cacheprovider

All against a tiny fixture dataset built fresh per test in ``tmp_path`` --
never against the real ``dataset/``, so these run in a few milliseconds and
never race the parallel authors regenerating it for real.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # training/clay-assistant
_TRAIN_DIR = _PACKAGE_ROOT / "train"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

import make_arm  # noqa: E402 -- train/ is not a package; path-inserted above

BUILT_CARD = "You are Warlock's Clay assistant (built card).\n\nTools:\n- clay_batch: ..."
NEW_CARD = "You are Warlock's Clay assistant (new card).\n\nTools:\n- clay_batch: ..."


def _record(rec_id: str, family: str, prompt: str, *, tools_sha: str = "sha-old") -> dict[str, Any]:
    return {
        "id": rec_id,
        "family": family,
        "kind": "build",
        "prompt": prompt,
        "calls": [
            {
                "name": "clay_add_primitive",
                "arguments": {"generator": "box", "name": "a", "params": {"size": [1, 1, 1]}},
            }
        ],
        "verified": {
            "object_count": 1,
            "bounds": {"size": [1, 1, 1], "center": [0, 0.5, 0]},
            "diagnose_clean": True,
            "call_count": 1,
            "tools_sha": tools_sha,
        },
    }


def _unsloth_row(
    record: dict[str, Any], *, card: str = BUILT_CARD, prompt: str | None = None
) -> dict[str, Any]:
    args_json = json.dumps({"calls": record["calls"]}, sort_keys=True, separators=(",", ":"))
    return {
        "messages": [
            {"role": "system", "content": card},
            {"role": "user", "content": prompt if prompt is not None else record["prompt"]},
            {
                "role": "assistant",
                "content": f"```json\n{args_json}\n```",
                "tool_calls": [
                    {"type": "function", "function": {"name": "clay_batch", "arguments": args_json}}
                ],
            },
        ]
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _make_dataset(
    tmp_path: Path,
    *,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    card: str = BUILT_CARD,
    corrupt_alignment_id: str | None = None,
    corrupt_card_id: str | None = None,
) -> Path:
    """A fixture dataset dir shaped like the real ``dataset/`` -- train/val
    jsonl plus their unsloth exports and a manifest -- with two knobs to
    deliberately misalign one row (*corrupt_alignment_id*: the row's user
    turn no longer ends with the record's prompt; *corrupt_card_id*: the
    row's system turn no longer matches *card*) so the refusal tests have
    something real to trip over.
    """
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    for split, records in (("train", train_records), ("val", val_records)):
        _write_jsonl(dataset_dir / f"{split}.jsonl", records)
        rows = []
        for record in records:
            row_card = card
            row_prompt = None
            if corrupt_card_id is not None and record["id"] == corrupt_card_id:
                row_card = "a completely different card that was never built"
            if corrupt_alignment_id is not None and record["id"] == corrupt_alignment_id:
                row_prompt = "a prompt belonging to some other record entirely"
            rows.append(_unsloth_row(record, card=row_card, prompt=row_prompt))
        _write_jsonl(dataset_dir / f"unsloth_{split}.jsonl", rows)

    (dataset_dir / "manifest.json").write_text(
        json.dumps({"accepted": len(train_records) + len(val_records)}, indent=2), encoding="utf-8"
    )
    return dataset_dir


def _write_card(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    # write_bytes, not write_text: text mode on Windows translates "\n" -> "\r\n" on
    # write, which would make the file's own bytes.decode("utf-8") -- what make_arm.py
    # actually reads -- differ from this fixture's own *text* even though nothing
    # meaningful changed. make_arm.py reads every card the same way (read_bytes().
    # decode("utf-8")), so the fixture must write them the same way.
    path.write_bytes(text.encode("utf-8"))
    return path


# --- 1. card swap: the new card lands in the output, the built card does not ---


def test_card_swap_replaces_the_system_turn(tmp_path: Path) -> None:
    train = [_record("furniture-0001", "furniture", "a wooden stool")]
    val = [_record("furniture-0002", "furniture", "a low bench")]
    dataset_dir = _make_dataset(tmp_path, train_records=train, val_records=val)
    built_card = _write_card(tmp_path, "built.txt", BUILT_CARD)
    new_card = _write_card(tmp_path, "new.txt", NEW_CARD)
    out_dir = tmp_path / "out"

    make_arm.build_arm(
        card_path=new_card,
        drop_families=set(),
        expect_rev=None,
        out_dir=out_dir,
        dataset_dir=dataset_dir,
        built_card_path=built_card,
    )

    written = json.loads(
        (out_dir / "unsloth_train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    system_content = written["messages"][0]["content"]
    assert system_content == NEW_CARD, "the new card must be what ships in the arm's output"
    assert system_content != BUILT_CARD, "the built card must not survive the swap"

    arm = json.loads((out_dir / "arm.json").read_text(encoding="utf-8"))
    assert arm["card_sha256"] == make_arm._sha256_text(NEW_CARD)
    assert arm["built_card_sha256"] == make_arm._sha256_text(BUILT_CARD)
    assert arm["dropped_families"] == []
    assert arm["splits"] == {"train": {"before": 1, "after": 1}, "val": {"before": 1, "after": 1}}


# --- 2. family drop: a dropped family's records never reach the output ---


def test_family_drop_removes_only_the_named_family(tmp_path: Path) -> None:
    train = [
        _record("furniture-0001", "furniture", "a wooden stool"),
        _record("figures-0001", "figures", "a standing knight"),
    ]
    val: list[dict[str, Any]] = []
    dataset_dir = _make_dataset(tmp_path, train_records=train, val_records=val)
    built_card = _write_card(tmp_path, "built.txt", BUILT_CARD)
    out_dir = tmp_path / "out"

    make_arm.build_arm(
        card_path=built_card,
        drop_families={"figures"},
        expect_rev=None,
        out_dir=out_dir,
        dataset_dir=dataset_dir,
        built_card_path=built_card,
    )

    kept_ids = [
        json.loads(line)["messages"][1]["content"]
        for line in (out_dir / "unsloth_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert kept_ids == ["a wooden stool"], "the figures record must not survive the drop"

    arm = json.loads((out_dir / "arm.json").read_text(encoding="utf-8"))
    assert arm["dropped_families"] == ["figures"]
    assert arm["splits"]["train"] == {"before": 2, "after": 1}


# --- 3. alignment refusal ----------------------------------------------------


def test_alignment_refusal_when_a_row_no_longer_matches_its_record(tmp_path: Path) -> None:
    train = [
        _record("furniture-0001", "furniture", "a wooden stool"),
        _record("furniture-0002", "furniture", "a round table"),
    ]
    dataset_dir = _make_dataset(
        tmp_path, train_records=train, val_records=[], corrupt_alignment_id="furniture-0002"
    )
    built_card = _write_card(tmp_path, "built.txt", BUILT_CARD)
    out_dir = tmp_path / "out"

    with pytest.raises(make_arm.ArmError, match="furniture-0002"):
        make_arm.build_arm(
            card_path=built_card,
            drop_families=set(),
            expect_rev=None,
            out_dir=out_dir,
            dataset_dir=dataset_dir,
            built_card_path=built_card,
        )
    assert not out_dir.exists(), "a refused arm must write nothing"


# --- 4. built-card refusal ----------------------------------------------------


def test_built_card_refusal_when_a_row_carries_a_different_system_turn(tmp_path: Path) -> None:
    train = [_record("furniture-0001", "furniture", "a wooden stool")]
    dataset_dir = _make_dataset(
        tmp_path, train_records=train, val_records=[], corrupt_card_id="furniture-0001"
    )
    built_card = _write_card(tmp_path, "built.txt", BUILT_CARD)
    out_dir = tmp_path / "out"

    with pytest.raises(make_arm.ArmError, match="system turn"):
        make_arm.build_arm(
            card_path=built_card,
            drop_families=set(),
            expect_rev=None,
            out_dir=out_dir,
            dataset_dir=dataset_dir,
            built_card_path=built_card,
        )
    assert not out_dir.exists()


# --- 5. --expect-rev: id-set and content checks, via a monkeypatched git ----


def test_expect_rev_accepts_a_matching_rev_ignoring_tools_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = [_record("furniture-0001", "furniture", "a wooden stool", tools_sha="sha-new")]
    dataset_dir = _make_dataset(tmp_path, train_records=train, val_records=[])
    built_card = _write_card(tmp_path, "built.txt", BUILT_CARD)
    out_dir = tmp_path / "out"

    # The rev's own record differs only in verified.tools_sha -- exactly what
    # a rebuild against a newer tool card is expected to change.
    rev_record = _record("furniture-0001", "furniture", "a wooden stool", tools_sha="sha-old-rev")
    rev_train_text = json.dumps(rev_record, sort_keys=True) + "\n"

    def fake_git_show(rev: str, rel_path: str, *, root: Path = make_arm._ROOT) -> str:
        assert rev == "deadbeef"
        if rel_path.endswith("train.jsonl"):
            return rev_train_text
        return ""

    monkeypatch.setattr(make_arm, "git_show", fake_git_show)

    make_arm.build_arm(
        card_path=built_card,
        drop_families=set(),
        expect_rev="deadbeef",
        out_dir=out_dir,
        dataset_dir=dataset_dir,
        built_card_path=built_card,
    )

    arm = json.loads((out_dir / "arm.json").read_text(encoding="utf-8"))
    assert arm["expect_rev"] == "deadbeef"
    assert arm["differing_records"] == {"train": 0, "val": 0}


def test_expect_rev_refuses_on_id_set_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = [_record("furniture-0001", "furniture", "a wooden stool")]
    dataset_dir = _make_dataset(tmp_path, train_records=train, val_records=[])
    built_card = _write_card(tmp_path, "built.txt", BUILT_CARD)
    out_dir = tmp_path / "out"

    other_record = _record("furniture-9999", "furniture", "a different stool entirely")
    rev_train_text = json.dumps(other_record, sort_keys=True) + "\n"

    def fake_git_show(rev: str, rel_path: str, *, root: Path = make_arm._ROOT) -> str:
        return rev_train_text if rel_path.endswith("train.jsonl") else ""

    monkeypatch.setattr(make_arm, "git_show", fake_git_show)

    with pytest.raises(make_arm.ArmError, match="id set"):
        make_arm.build_arm(
            card_path=built_card,
            drop_families=set(),
            expect_rev="deadbeef",
            out_dir=out_dir,
            dataset_dir=dataset_dir,
            built_card_path=built_card,
        )
    assert not out_dir.exists()


def test_expect_rev_refuses_and_names_the_differing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = [_record("furniture-0001", "furniture", "a wooden stool")]
    dataset_dir = _make_dataset(tmp_path, train_records=train, val_records=[])
    built_card = _write_card(tmp_path, "built.txt", BUILT_CARD)
    out_dir = tmp_path / "out"

    # Same id, but the rev's own prompt differs -- a real content drift, not
    # just the expected tools_sha rebuild noise.
    rev_record = _record("furniture-0001", "furniture", "a completely different prompt")
    rev_train_text = json.dumps(rev_record, sort_keys=True) + "\n"

    def fake_git_show(rev: str, rel_path: str, *, root: Path = make_arm._ROOT) -> str:
        return rev_train_text if rel_path.endswith("train.jsonl") else ""

    monkeypatch.setattr(make_arm, "git_show", fake_git_show)

    with pytest.raises(make_arm.ArmError, match="furniture-0001"):
        make_arm.build_arm(
            card_path=built_card,
            drop_families=set(),
            expect_rev="deadbeef",
            out_dir=out_dir,
            dataset_dir=dataset_dir,
            built_card_path=built_card,
        )
    assert not out_dir.exists()


# --- 6. main()'s CLI plumbing, end to end -------------------------------------


def test_main_writes_arm_json_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    train = [_record("furniture-0001", "furniture", "a wooden stool")]
    dataset_dir = _make_dataset(tmp_path, train_records=train, val_records=[])
    built_card = _write_card(tmp_path, "built.txt", BUILT_CARD)
    new_card = _write_card(tmp_path, "new.txt", NEW_CARD)
    out_dir = tmp_path / "out"

    code = make_arm.main(
        [
            "--card",
            str(new_card),
            "--out",
            str(out_dir),
            "--dataset",
            str(dataset_dir),
            "--built-card",
            str(built_card),
        ]
    )
    assert code == 0
    assert (out_dir / "arm.json").is_file()
    captured = capsys.readouterr()
    assert "wrote" in captured.out


def test_main_returns_nonzero_and_prints_refusal_on_bad_pairing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    train = [_record("furniture-0001", "furniture", "a wooden stool")]
    dataset_dir = _make_dataset(tmp_path, train_records=train, val_records=[])
    built_card = _write_card(tmp_path, "built.txt", BUILT_CARD)
    wrong_card_as_built = _write_card(tmp_path, "wrong.txt", "not the built card at all")
    out_dir = tmp_path / "out"

    code = make_arm.main(
        [
            "--card",
            str(built_card),
            "--out",
            str(out_dir),
            "--dataset",
            str(dataset_dir),
            "--built-card",
            str(wrong_card_as_built),
        ]
    )
    assert code != 0
    captured = capsys.readouterr()
    assert "refusing" in captured.err
    assert not out_dir.exists()


# --- 7. this module is importable on any Python (used by importlib elsewhere) --


def test_make_arm_module_has_no_third_party_imports() -> None:
    """Guards the "stdlib only, runnable with any Python 3.11+" claim in
    ``make_arm.py``'s own module docstring -- reloaded from its file path
    (rather than trusting the already-imported module, which this test file
    itself put on ``sys.path``) so a future import this script adds at
    module scope would actually be exercised here."""
    path = _TRAIN_DIR / "make_arm.py"
    spec = importlib.util.spec_from_file_location("make_arm_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # raises ImportError if a non-stdlib import crept in
    assert hasattr(module, "build_arm")
