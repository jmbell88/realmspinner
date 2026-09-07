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
    """Only the ids the finding names collide; ``prompt`` names no other
    panel's control on these three doors, so it is not remapped."""

    def _boom(svc, job_id, prompt, **kwargs):
        raise Invalid("describe the surface you want", field="prompt")

    monkeypatch.setattr(texture_panel.svc_jobs, "retexture_job", _boom)

    with pytest.raises(Invalid) as excinfo:
        texture_panel._retexture_job(object(), "job-1", "")

    assert excinfo.value.field == "prompt"


def test_the_three_panels_field_prefix_tables_do_not_reintroduce_the_collision():
    """A direct check on the fix's data, not just one instance of it: every
    renamed id these three panels hand out is unique across the three tables."""
    renamed = (
        list(texture_panel._FIELD_PREFIX.values())
        + list(remesh_panel._FIELD_PREFIX.values())
        + list(sheet_panel._FIELD_PREFIX.values())
    )
    assert len(renamed) == len(set(renamed))
