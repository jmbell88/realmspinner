from __future__ import annotations

import ast
import inspect

import realmspinner._q_jobs as q_jobs
from realmspinner import followups
from realmspinner.db import JobStore
from realmspinner.service.validation import DERIVED_PARAMS
from realmspinner.studio.modes.library.ui.panes import library
from realmspinner.studio.panes import inspector


def test_persist_keeps_parent_params_and_each_followup(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    try:
        job_id = store.create("image", "a ranger", {"seed": 7})
        assert followups.persist(store, job_id, "rig", OSError("read-only"))
        assert followups.persist(store, job_id, "sprite_synthesis", "sprite row was not queued")

        params = store.get(job_id)["params"]
        assert params["seed"] == 7
        assert set(params[followups.PARAM_KEY]) == {"rig", "sprite_synthesis"}
        assert params[followups.PARAM_KEY]["rig"]["error_type"] == "OSError"
    finally:
        store.close()


def test_records_rejects_malformed_entries_and_bounds_messages():
    record = followups.failure_record("rig", "x" * 900, recorded_at=12.0)
    params = {
        followups.PARAM_KEY: {
            "rig": record,
            "bad": "not an object",
            "empty": {"message": ""},
        }
    }
    assert len(record["message"]) == followups.MAX_MESSAGE
    assert followups.records(params) == [
        {
            "kind": "rig",
            "label": "Automatic rig",
            "message": "x" * followups.MAX_MESSAGE,
            "error_type": "Unavailable",
            "recorded_at": 12.0,
        }
    ]


def test_studio_wording_names_the_missing_followup_and_reason():
    job = {
        "params": {
            followups.PARAM_KEY: {
                "sprite_synthesis": followups.failure_record(
                    "sprite_synthesis", RuntimeError("database is locked")
                )
            }
        }
    }
    assert inspector.followup_failure_lines(job) == [
        ("Sprite sheet was not queued.", "database is locked")
    ]
    assert library.followup_failure_tooltip(job) == (
        "Sprite sheet was not queued: database is locked"
    )


def test_a_rerun_cannot_inherit_an_earlier_followup_failure():
    assert followups.PARAM_KEY in DERIVED_PARAMS


# --- the 2026-09-11 audit, findings service-04 and muse-08 --------------------
#
# ``tests/test_jobs_followups.py`` does not exist in this tree; the regression
# lands here because this module is the one that already covers ``followups``
# (muse-08 is a ``followups.PRODUCTS`` gap). service-04's ``source_job`` check
# on ``_maybe_queue_sheet_after_rig`` left with that follow-up (``retired.py``).


def _source_job_carrying_kinds() -> set[str]:
    """Every job kind ``_discard_artifacts`` treats as writing into a
    *source* job's directory, derived from its own source via ``ast`` rather
    than hand-listed here -- the same one-more-kind drift muse-08 found in
    ``followups.PRODUCTS`` itself (and service-06 before it), so this test
    must not become a second place that count can go stale.

    A branch counts only if it both switches on ``job["kind"]`` and reads
    ``params["source_job"]`` somewhere in its body: ``tile_sheet`` also
    switches on ``job["kind"]`` in this function but writes into its own
    directory, not a source's, and must not be swept in by a looser match.
    """
    tree = ast.parse(inspect.getsource(q_jobs))
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_discard_artifacts"
    )
    kinds: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Subscript)
            and isinstance(test.left.value, ast.Name)
            and test.left.value.id == "job"
        ):
            continue
        branch_kinds: set[str] = set()
        for op, comparator in zip(test.ops, test.comparators, strict=True):
            if isinstance(op, ast.Eq) and isinstance(comparator, ast.Constant):
                branch_kinds.add(comparator.value)
            elif isinstance(op, ast.In) and isinstance(comparator, (ast.Tuple, ast.List)):
                branch_kinds.update(
                    elt.value for elt in comparator.elts if isinstance(elt, ast.Constant)
                )
        if not branch_kinds:
            continue
        uses_source_job = any(
            isinstance(n, ast.Constant) and n.value == "source_job" for n in ast.walk(node)
        )
        if uses_source_job:
            kinds.update(branch_kinds)
    return kinds


def test_followups_products_names_every_source_job_carrying_kind_including_separate():
    """muse-08: a correspondence check, not a hand-listed eight names -- an
    eighth (or ninth) kind that starts writing into a source job's directory
    must fail this the moment it is added to ``_discard_artifacts``, the way
    ``separate`` (added for stem splitting) failed it silently until now.
    """
    carrying = _source_job_carrying_kinds()
    # The premise: more than the empty set, and specifically including the
    # kind this finding is about -- otherwise a bug in the ast extraction
    # above would make this test vacuously true.
    assert "separate" in carrying
    assert "rig" in carrying and "sprite_synthesis" in carrying
    missing = carrying - set(followups.PRODUCTS)
    assert not missing, (
        f"followups.PRODUCTS has no entry for: {sorted(missing)} -- every kind "
        "that carries params['source_job'] needs one, or its finished toast "
        "reads as the generic '{name} finished.' instead of a named one"
    )
