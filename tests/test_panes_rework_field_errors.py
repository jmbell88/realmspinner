"""The rework panels' field ids stop colliding in ``ctx.state.field_errors``.

The 2026-09-07 audit, finding create-01: ``ctx.state.field_errors`` is one
flat, unnamespaced dict, and three rework panels reused the bare ids
"strength", "texture_size" and "custom_faces" for unrelated controls
(``texture_panel``'s restyle strength and atlas size, ``remesh_panel``'s bake
size, ``sheet_panel``'s pixel-restyle strength) -- so a refusal from one door
rang a sibling panel's unrelated control, and the inspector draws these panels
together routinely. Each panel's submit door is now wrapped to relabel its
service call's colliding refusal before it ever reaches
``ctx.state.field_errors``; the widget drawing the control is keyed by the
same, now-unique, name (see each panel's own ``_FIELD_PREFIX``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.service.errors import Invalid
from warlock.studio.panes import remesh_panel, sheet_panel, texture_panel


def test_a_remesh_texture_size_refusal_does_not_ring_the_retexture_atlas_control(monkeypatch):
    """The finding's own regression name: a remesh's atlas-size refusal must
    not carry the bare "texture_size" id that ``texture_panel`` also uses for
    its own, unrelated atlas-size control."""

    def _boom(svc, job_id, **kwargs):
        raise Invalid("texture_size must be one of (...)", field="texture_size")

    monkeypatch.setattr(remesh_panel.svc_jobs, "remesh_job", _boom)

    with pytest.raises(Invalid) as excinfo:
        remesh_panel._remesh_job(object(), "job-1", texture_size=999999)

    assert excinfo.value.field == "remesh_texture_size"
    assert excinfo.value.field != "texture_size"
    assert excinfo.value.field != texture_panel._FIELD_PREFIX["texture_size"]


def test_a_retexture_strength_refusal_does_not_ring_the_sheet_pixel_strength_control(monkeypatch):
    """Same collision, the other pairing: a re-texture strength refusal must
    not carry the bare "strength" id ``sheet_panel`` also uses for its pixel
    restyle's own strength control."""

    def _boom(svc, job_id, prompt, **kwargs):
        raise Invalid("strength must be between 0 and 1", field="strength")

    monkeypatch.setattr(texture_panel.svc_jobs, "retexture_job", _boom)

    with pytest.raises(Invalid) as excinfo:
        texture_panel._retexture_job(object(), "job-1", "mossy stone", strength=5.0)

    assert excinfo.value.field == "retexture_strength"
    assert excinfo.value.field != "strength"
    assert excinfo.value.field != sheet_panel._FIELD_PREFIX["strength"]


def test_a_pixel_sheet_strength_refusal_does_not_ring_the_retexture_strength_control(monkeypatch):
    def _boom(svc, job_id, sheet_id, **kwargs):
        raise Invalid("strength must be between 0 and 1", field="strength")

    monkeypatch.setattr(sheet_panel.svc_sheets, "create_pixel_sheet", _boom)

    with pytest.raises(Invalid) as excinfo:
        sheet_panel._create_pixel_sheet(object(), "job-1", "sheet-1", strength=5.0)

    assert excinfo.value.field == "sheet_pixel_strength"
    assert excinfo.value.field != "strength"
    assert excinfo.value.field != texture_panel._FIELD_PREFIX["strength"]


def test_an_unrelated_refusal_field_passes_through_unchanged(monkeypatch):
    """Only the ids the finding names collide; ``control`` names no other
    panel's control on these three doors, so it is not remapped.

    The 2026-09-08 audit, finding create-02: this used to make the same
    assertion about ``"prompt"``, which was true only among these three
    rework panels -- it went false once ``create_brief.py``'s own bare
    "prompt" field (the Reference stage's main asset prompt) is considered,
    since a refusal there rang this panel's Surface field too. ``"prompt"``
    is now remapped (see ``test_a_create_brief_empty_prompt_refusal_...``
    below); ``"control"`` is the still-unmapped case this test was written to
    cover.
    """

    def _boom(svc, job_id, prompt, **kwargs):
        raise Invalid("control must be a known ControlNet", field="control")

    monkeypatch.setattr(texture_panel.svc_jobs, "retexture_job", _boom)

    with pytest.raises(Invalid) as excinfo:
        texture_panel._retexture_job(object(), "job-1", "mossy stone", control="bogus")

    assert excinfo.value.field == "control"


def test_a_create_brief_empty_prompt_refusal_does_not_ring_the_retexture_surface_field(
    monkeypatch,
):
    """The 2026-09-08 audit, finding create-02: ``texture_panel``'s Surface
    prompt field was keyed by the bare id "prompt" -- the same id
    ``create_brief.py``'s main asset-prompt field uses, and the same id
    ``settings_2d.validate()`` files an empty-prompt refusal under -- in the
    shared, flat ``ctx.state.field_errors`` dict. An ordinary "the asset
    prompt is empty" refusal from Create's Reference stage carried
    ``field="prompt"`` and so rang the Surface field of an unrelated,
    already-built mesh asset shown in the inspector at the same time.
    """

    def _boom(svc, job_id, prompt, **kwargs):
        raise Invalid("describe the surface you want", field="prompt")

    monkeypatch.setattr(texture_panel.svc_jobs, "retexture_job", _boom)

    with pytest.raises(Invalid) as excinfo:
        texture_panel._retexture_job(object(), "job-1", "")

    assert excinfo.value.field == "retexture_prompt"
    assert excinfo.value.field != "prompt"


def test_texture_panel_warn_does_not_touch_the_filesystem_from_the_frame_thread(monkeypatch):
    """The 2026-09-08 audit, finding create-04: ``texture_panel._warn`` used
    to call ``svc_jobs.stale_surface_artifacts(ctx.svc.job_dir(job["id"]))``
    -- three ``Path.exists()`` filesystem checks -- every frame the "Surface
    texture" section is open, unmemoized: disk I/O on the frame thread. Its
    siblings (``remesh_panel._warn_stale``, ``retarget_panel._warn_stale``)
    answer the equivalent question from the already-cached ``job.get("files")``
    list instead, with no disk access. Proven here by making
    ``ctx.svc.job_dir`` explode if it is ever called: the fixed ``_warn``
    never needs a job directory at all.
    """
    calls: list[tuple] = []
    monkeypatch.setattr(
        texture_panel.widgets, "text_colored", lambda *a, **k: calls.append(a)
    )
    monkeypatch.setattr(texture_panel.widgets, "muted", lambda *a, **k: calls.append(a))

    def _boom(job_id):
        raise AssertionError("_warn touched the filesystem via ctx.svc.job_dir")

    ctx = SimpleNamespace(svc=SimpleNamespace(job_dir=_boom))
    job = {"id": "job-1", "files": ["model.glb", "model_obj.zip", "textures.zip"]}
    form = {"depth": True}

    texture_panel._warn(ctx, job, form)  # must not raise -- ctx.svc.job_dir is never called

    assert any("model_obj.zip" in call[1] and "textures.zip" in call[1] for call in calls)


def test_the_three_panels_field_prefix_tables_do_not_reintroduce_the_collision():
    """A direct check on the fix's data, not just one instance of it: every
    renamed id these three panels hand out is unique across the three tables."""
    renamed = (
        list(texture_panel._FIELD_PREFIX.values())
        + list(remesh_panel._FIELD_PREFIX.values())
        + list(sheet_panel._FIELD_PREFIX.values())
    )
    assert len(renamed) == len(set(renamed))
