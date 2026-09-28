"""Wave 3, fixer 6 of the fix pass over the 2026-09-26 audit
(the 2026-09-26 audit): the ``pipelines-install-*`` findings that
land in the bench/install/models corner of the tree -- ``bench/findings.py``,
``bench/metrics.py``, ``bench/calibrate.py`` and the deleted-module docstrings
in ``bench/__init__.py``.

Each test name is the specific claim the unfixed code failed; the docstring
under it names the finding.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import realmspinner.bench as bench_pkg
from realmspinner.bench import calibrate as calibrate_mod
from realmspinner.bench import findings as findings_mod
from realmspinner.bench import metrics

# --- pipelines-install-07: findings.load / hint raise on a malformed doc ----


def test_load_returns_none_for_a_top_level_list_rather_than_a_dict(tmp_path):
    """``json.loads`` can parse to anything JSON allows, not only a dict; a
    findings.json that parses to a bare list used to come back from ``load``
    unchanged and blow up the first ``.get`` call in ``hint``/``best_value``."""
    findings_mod._CACHE.clear()
    path = tmp_path / "findings.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    assert findings_mod.load(path) is None


def test_hint_does_not_raise_when_the_document_is_a_list():
    """Before the fix: ``hint([1, 2, 3], "trellis_gss", 0.6)`` raised
    ``AttributeError: 'list' object has no attribute 'get'`` from
    ``rendered``'s first line, on the frame thread."""
    assert findings_mod.hint([1, 2, 3], "trellis_gss", 0.6) is None


def test_hint_does_not_raise_when_a_bucket_entry_has_a_null_n():
    """A bucket entry of ``{"n": null, ...}`` is valid JSON (an explicit write
    of nothing) but ``entry.get("n", 0) >= min_n`` only applies its default
    when the key is *absent* -- with the key present and null this compared
    ``None >= 5`` and raised ``TypeError``."""
    doc = {"params": {"trellis_gss": {"0.6": {"n": None, "accepts": 0}}}}
    assert findings_mod.hint(doc, "trellis_gss", 0.6, min_n=5) is None


def test_hint_does_not_raise_when_the_param_bucket_is_a_list():
    """``section.get(param)`` can itself be a non-dict (a list here) on a
    hand-edited file; ``_lookup`` called ``.get`` on it unguarded and raised."""
    doc = {"params": {"trellis_gss": [1, 2, 3]}}
    assert findings_mod.hint(doc, "trellis_gss", 0.6) is None


# --- pipelines-install-08: dino_available reads a half-downloaded dir as OK ---


def test_dino_available_is_false_for_an_empty_directory(tmp_path, monkeypatch):
    """A killed download (or a ``mkdir`` that ran ahead of the transfer) left
    the bare ``dinov2-base`` folder on disk with no weights in it, and the old
    ``.exists()``-only check called that "available" -- unlike
    ``pickscore_available`` beside it, which already checks for its weights
    file."""
    empty = tmp_path / "dinov2-base"
    empty.mkdir()
    monkeypatch.setattr(metrics, "_dino_dir", lambda config=None: empty)

    assert metrics.dino_available() is False


def test_dino_available_is_true_once_a_safetensors_file_exists(tmp_path, monkeypatch):
    weights_dir = tmp_path / "dinov2-base"
    weights_dir.mkdir()
    (weights_dir / "model.safetensors").write_bytes(b"")
    monkeypatch.setattr(metrics, "_dino_dir", lambda config=None: weights_dir)

    assert metrics.dino_available() is True


# --- pipelines-install-09: sweep_job scores a leftover PNG from a prior run --


def test_sweep_job_does_not_score_a_stale_png_left_by_an_earlier_run(tmp_path, monkeypatch):
    """A resumed/re-run sweep reuses ``out_dir``'s ``e<elev>`` folders. Before
    the fix, a PNG left there by an earlier run (or an earlier, larger
    ``yaws``) was never cleared, so a render that produced nothing this time
    still "found" the old file at the same index and scored it as if it were
    the current run's -- silently. Blender itself is stubbed out here: the
    worker call is mocked to write nothing, so the only way a cell can appear
    in the result is if the stale file was (wrongly) picked up."""
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "model.glb").write_bytes(b"")
    Image.new("RGB", (8, 8)).save(job_dir / "input.png")

    out_dir = tmp_path / "out"
    elevation = 20.0
    frames = out_dir / f"e{elevation:06.2f}".replace(".", "_")
    frames.mkdir(parents=True)
    # A leftover from a previous run at the same elevation, same index.
    Image.new("RGB", (8, 8)).save(frames / "0000.png")

    monkeypatch.setattr(
        "realmspinner.pipelines.blender_run.run_worker",
        lambda spec, timeout=None: {},
    )
    monkeypatch.setattr(
        calibrate_mod.metrics_mod,
        "score_view",
        lambda reference, render, config=None: {"silhouette_iou": 1.0},
    )

    result = calibrate_mod.sweep_job(
        job_dir, out_dir, yaws=1, elevations=(elevation,), config=None
    )

    assert result["cells"] == []


# --- pipelines-install-11: bench docstrings name a module that no longer exists --


def test_bench_docstrings_name_the_real_agent_caller_not_the_deleted_module():
    """``studio.agent_clay`` does not exist in this tree; the Clay agent
    surface these docstrings describe moved to
    ``modes/clay/agent/tools_ops.py``."""
    assert "agent_clay" not in (bench_pkg.__doc__ or "")
    assert "agent_clay" not in (metrics.__doc__ or "")
    assert "agent_clay" not in (metrics.silhouette_iou_masks.__doc__ or "")
    assert "agent_clay" not in (metrics.compare_silhouette.__doc__ or "")


# --- pipelines-install-10: scripts/sweep_refill.py's usage lines are escaped -


def test_sweep_refill_usage_lines_have_no_html_entities():
    """The module docstring's usage examples were pasted through an HTML
    escaper at some point and kept the literal ``&lt;sweep_id&gt;`` rather
    than ``<sweep_id>`` -- copy-pasting the command as printed fails.

    Read as source text rather than imported: ``scripts/`` carries no
    ``__init__.py`` and is not on the import path pytest collects under.
    """
    path = Path(__file__).resolve().parents[1] / "scripts" / "sweep_refill.py"
    doc = path.read_text(encoding="utf-8")
    assert "&lt;" not in doc
    assert "&gt;" not in doc
    assert "<sweep_id>" in doc
