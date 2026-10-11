"""The 2026-10-11 audit's History and Retarget findings (create-10, create-20,
create-22). Each test's name is the claim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from realmspinner.service import generation_log
from realmspinner.service import verdicts as svc_verdicts
from realmspinner.service.errors import Invalid
from realmspinner.studio.modes.create.engine import history as engine
from realmspinner.studio.modes.create.ui.panes import history
from realmspinner.studio.panes import retarget_panel

PARAMS = {
    "generation_type": "object",
    "base_model": "sdxl_cfg",
    "style_lora": "render3d",
    "lora_weight": 0.8,
    "seed": 1204285292,
    "recipe": {"reference": {"base_model": "sdxl_cfg", "steps": 30, "guidance_scale": 7.0}},
}


def _image_job(svc, *, input_png: bool = True, params: dict | None = None) -> str:
    job_id = svc.store.create(
        "text", "a chest", dict(PARAMS if params is None else params),
        stage="reference", status="done",
    )
    directory = svc.job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    if input_png:
        (directory / "input.png").write_bytes(b"x")
    return job_id


# --- create-10 ------------------------------------------------------------------------


def test_history_row_for_an_image_only_attempt_has_a_thumbnail(svc):
    """The 2026-10-11 audit, finding create-10: a thumbnail came from
    ``reference.png`` alone, which only the mesh stage writes, so every
    reference-stage attempt (most of what History lists) drew a flat swatch."""
    job_id = _image_job(svc)
    (row,) = generation_log.history(svc)
    assert row["thumbnail"] == str(svc.job_dir(job_id) / "input.png")


def test_history_thumbnail_prefers_reference_png_over_input_png(svc):
    job_id = _image_job(svc)
    (svc.job_dir(job_id) / "reference.png").write_bytes(b"x")
    (row,) = generation_log.history(svc)
    assert row["thumbnail"] == str(svc.job_dir(job_id) / "reference.png")


def test_history_row_with_no_image_on_disk_has_no_thumbnail(svc):
    _image_job(svc, input_png=False)
    (row,) = generation_log.history(svc)
    assert row["thumbnail"] is None


# --- create-20 ------------------------------------------------------------------------


def test_history_page_does_not_stat_rows_outside_the_page(svc, monkeypatch):
    """The 2026-10-11 audit, finding create-20: the whole log was joined, every
    unjudged job rebuilt a ``settings_snapshot`` and every job's image was
    stat-ed before the page was sliced off the sorted list."""
    ids = [_image_job(svc) for _ in range(8)]
    dirs = {str(svc.job_dir(j)): j for j in ids}

    snapshots: list[str] = []
    real_snapshot = generation_log.settings_snapshot

    def counting_snapshot(job):
        snapshots.append(job["id"])
        return real_snapshot(job)

    stats: list[str] = []
    real_is_file = Path.is_file

    def counting_is_file(self, *a, **k):
        if str(self.parent) in dirs:
            stats.append(dirs[str(self.parent)])
        return real_is_file(self, *a, **k)

    monkeypatch.setattr(generation_log, "settings_snapshot", counting_snapshot)
    monkeypatch.setattr(Path, "is_file", counting_is_file)

    page = generation_log.history(svc, limit=2)

    assert len(page) == 2
    shown = {row["job_id"] for row in page}
    assert set(snapshots) <= shown, "a snapshot was rebuilt for a row off the page"
    assert set(stats) <= shown, "an image was stat-ed for a row off the page"


def test_history_load_reads_the_library_once_even_when_it_wants_options(svc, monkeypatch):
    """Options used to be a second ``history(limit=1_000_000)`` on top of the
    page read, so each finished generation cost two whole-library joins."""
    for _ in range(3):
        _image_job(svc)
    reads = []
    real = svc.store.history_jobs

    def counting():
        reads.append(1)
        return real()

    monkeypatch.setattr(svc.store, "history_jobs", counting)

    loaded = history.load(svc, dict(engine.NO_FILTERS), 2, True)

    assert len(reads) == 1
    assert len(loaded["rows"]) == 3, "the page asks for one row more than it shows"
    assert ("sdxl_cfg", generation_log.base_label("sdxl_cfg")) in loaded["options"]["base_model"]


def test_history_options_name_models_a_filter_hid_from_the_page(svc):
    """The one pass derives options from the unfiltered log, so a filtered page
    still names every model it filtered out."""
    _image_job(svc)
    other = dict(PARAMS, base_model="turbo", recipe={"reference": {"base_model": "turbo"}})
    _image_job(svc, params=other)

    loaded = history.load(svc, dict(engine.NO_FILTERS, base_model="turbo"), 50, True)

    assert [r["base_model"] for r in loaded["rows"]] == ["turbo"]
    keys = {key for key, _label in loaded["options"]["base_model"]}
    assert keys == {"sdxl_cfg", "turbo"}


def test_a_filtered_page_matches_filtering_the_whole_list(svc):
    ids = []
    for i in range(6):
        params = dict(PARAMS, seed=i)
        if i % 2:
            params = dict(params, base_model="turbo", recipe={"reference": {"base_model": "turbo"}})
        ids.append(_image_job(svc, params=params))
    svc.store.set_stage(ids[1], "model")
    svc_verdicts.record_verdict(svc, ids[1], grade=1)

    everything = generation_log.history(svc, limit=1000)
    expected = [r["job_id"] for r in everything if r["base_model"] == "turbo"]
    page = generation_log.history(svc, base_model="turbo", limit=2, offset=1)

    assert [r["job_id"] for r in page] == expected[1:3]


# --- create-22 ------------------------------------------------------------------------


def _refuse_with(monkeypatch, field: str):
    def refuse(svc, job_id, **kwargs):
        raise Invalid("nope", field=field)

    monkeypatch.setattr(retarget_panel.svc_jobs, "optimize_job", refuse)


@pytest.mark.parametrize(
    ("door_field", "ring"),
    [("profile", "retarget_profile"), ("custom_triangles", "retarget_custom_triangles")],
)
def test_a_retarget_refusal_does_not_ring_the_create_budget_combo(monkeypatch, door_field, ring):
    """The 2026-10-11 audit, finding create-22: ``optimize_job`` refuses with
    ``profile``/``custom_triangles``, the very ids Create's Budget combo rings
    in the one flat ``field_errors`` dict, so a refusal on one control lit the
    other. The panel's door relabels them to its own."""
    _refuse_with(monkeypatch, door_field)

    with pytest.raises(Invalid) as exc:
        retarget_panel._optimize_job(object(), "aaaaaaaaaaaa", profile="x", custom_triangles=None)

    assert exc.value.field == ring
    assert exc.value.field != door_field


def test_the_retarget_panel_draws_its_rings_under_the_relabelled_ids():
    import inspect
    import re

    source = inspect.getsource(retarget_panel.draw)
    assert re.search(r'form_ui\.combo\(\s*"retarget_profile"', source)
    assert re.search(r'form_ui\.number\(\s*"retarget_custom_triangles"', source)


def test_a_retarget_refusal_of_another_field_is_left_alone(monkeypatch):
    _refuse_with(monkeypatch, "source_job")
    with pytest.raises(Invalid) as exc:
        retarget_panel._optimize_job(object(), "aaaaaaaaaaaa", profile="x", custom_triangles=None)
    assert exc.value.field == "source_job"
