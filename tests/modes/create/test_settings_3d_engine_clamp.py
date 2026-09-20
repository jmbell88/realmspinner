"""The 3D pane's Texture resolution and Band can no longer hold a value the door refuses.

The incident (2026-09-18): the field clamped only at 0, so a typed 64 was kept
and persisted, and every Make 3D -> Accept after it was refused by
``check_trellis_tex_res`` with a toast pointing at a control inside the
collapsed Engine header. A good reference could never become a mesh, and the
log said nothing, because a refusal is toasted rather than logged. Band, two
controls above it in the same header, had the identical defect (1..64 at the
door, only >= 0 at the field).

The 2026-09-20 audit, finding create-01: the other three of the five Engine
fields (Token budget, Decimation, Atlas resolution) had the same shape of
defect and no clamp helper at all -- ``settings_3d.py`` committed them with a
floor-only ``max(...)`` and no ceiling, so every one of them could hold a
value ``check_trellis_max_tokens``/``check_trellis_decim``/
``check_trellis_atlas`` refuses at every Accept.
"""

import contextlib

import pytest

from realmspinner.service.errors import Invalid
from realmspinner.service.validation import (
    MAX_TRELLIS_ATLAS,
    MAX_TRELLIS_BAND,
    MAX_TRELLIS_DECIM,
    MAX_TRELLIS_MAX_TOKENS,
    MAX_TRELLIS_TEX_RES,
    MIN_TRELLIS_ATLAS,
    MIN_TRELLIS_BAND,
    MIN_TRELLIS_TEX_RES,
    check_trellis_atlas,
    check_trellis_band,
    check_trellis_decim,
    check_trellis_max_tokens,
    check_trellis_tex_res,
)
from realmspinner.studio import forms
from realmspinner.studio.modes.create.engine import mesh as create_mesh
from realmspinner.studio.settings import restore_form
from realmspinner.studio.state import DEFAULT_FORM_3D


@pytest.mark.parametrize(
    ("typed", "kept"),
    [
        (-5, 0),
        (0, 0),
        (1, MIN_TRELLIS_TEX_RES),
        (64, MIN_TRELLIS_TEX_RES),
        (MIN_TRELLIS_TEX_RES, MIN_TRELLIS_TEX_RES),
        (1024, 1024),
        (MAX_TRELLIS_TEX_RES, MAX_TRELLIS_TEX_RES),
        (99999, MAX_TRELLIS_TEX_RES),
    ],
)
def test_a_typed_texture_resolution_lands_inside_the_doors_range(typed, kept):
    assert create_mesh.clamp_tex_res(typed) == kept


@pytest.mark.parametrize("typed", [-1, 0, 1, 64, 127, 128, 512, 4096, 4097, 10**6])
def test_whatever_the_field_keeps_the_promotion_door_accepts(typed):
    kept = create_mesh.clamp_tex_res(typed)
    # 0 is the sentinel promote_kwargs leaves out entirely; anything else is
    # sent, and must pass the same check promote_to_model runs.
    if kept:
        check_trellis_tex_res(kept)


def test_the_value_that_blocked_the_donut_is_still_refused_at_the_door():
    # The clamp is the UI's half; the door keeps its own bound, so the fix
    # must not have been made by loosening it.
    with pytest.raises(Invalid):
        check_trellis_tex_res(64)


@pytest.mark.parametrize("stored", [64, 1, 127, 4097])
def test_a_saved_out_of_range_texture_resolution_restores_as_unset(stored):
    form = restore_form(DEFAULT_FORM_3D, {**DEFAULT_FORM_3D, "trellis_tex_res": stored})
    assert form["trellis_tex_res"] == 0


@pytest.mark.parametrize("stored", [0, MIN_TRELLIS_TEX_RES, 512, MAX_TRELLIS_TEX_RES])
def test_a_saved_in_range_texture_resolution_survives_a_restore(stored):
    form = restore_form(DEFAULT_FORM_3D, {**DEFAULT_FORM_3D, "trellis_tex_res": stored})
    assert form["trellis_tex_res"] == stored


def test_form_number_passes_commit_through_so_the_clamp_waits_for_the_edit_to_end(
    monkeypatch,
):
    # Per-keystroke, a clamp would turn the "2" of a typed 2048 into 128.
    seen: dict[str, object] = {}

    def fake_input_int(label, value, **kwargs):
        seen.update(kwargs)
        return (False, value)

    monkeypatch.setattr(forms.controls, "input_int", fake_input_int)
    form = forms.Form("t", available_width=300.0)
    monkeypatch.setattr(
        form, "field", lambda *a, **k: contextlib.nullcontext("")
    )
    form.number("trellis_tex_res", "Texture resolution", 0, commit=True)
    assert seen.get("commit") is True


# -- Band ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "kept"),
    [
        (-5, 0),
        (0, 0),
        (MIN_TRELLIS_BAND, MIN_TRELLIS_BAND),
        (8, 8),
        (MAX_TRELLIS_BAND, MAX_TRELLIS_BAND),
        (65, MAX_TRELLIS_BAND),
        (1000, MAX_TRELLIS_BAND),
    ],
)
def test_a_typed_band_lands_inside_the_doors_range(typed, kept):
    assert create_mesh.clamp_band(typed) == kept


@pytest.mark.parametrize("typed", [-1, 0, 1, 8, 64, 65, 10**6])
def test_whatever_the_band_field_keeps_the_promotion_door_accepts(typed):
    kept = create_mesh.clamp_band(typed)
    if kept:
        check_trellis_band(kept)


def test_an_out_of_range_band_is_still_refused_at_the_door():
    with pytest.raises(Invalid):
        check_trellis_band(MAX_TRELLIS_BAND + 1)


@pytest.mark.parametrize("stored", [65, 1000])
def test_a_saved_out_of_range_band_restores_as_unset(stored):
    form = restore_form(DEFAULT_FORM_3D, {**DEFAULT_FORM_3D, "trellis_band": stored})
    assert form["trellis_band"] == 0


@pytest.mark.parametrize("stored", [0, MIN_TRELLIS_BAND, 8, MAX_TRELLIS_BAND])
def test_a_saved_in_range_band_survives_a_restore(stored):
    form = restore_form(DEFAULT_FORM_3D, {**DEFAULT_FORM_3D, "trellis_band": stored})
    assert form["trellis_band"] == stored


# -- Token budget / Decimation / Atlas resolution (the 2026-09-20 audit, ------
# finding create-01: the three siblings that shipped with no clamp at all) ---


def test_engine_advanced_numeric_fields_clamp_to_the_doors_ceiling_like_band_and_tex_res():
    # Token budget: sentinel is 0 (like band/tex_res), door range is 1..MAX.
    assert create_mesh.clamp_max_tokens(-5) == 0
    assert create_mesh.clamp_max_tokens(0) == 0
    assert create_mesh.clamp_max_tokens(1) == 1
    assert create_mesh.clamp_max_tokens(MAX_TRELLIS_MAX_TOKENS) == MAX_TRELLIS_MAX_TOKENS
    assert create_mesh.clamp_max_tokens(MAX_TRELLIS_MAX_TOKENS + 1) == MAX_TRELLIS_MAX_TOKENS
    check_trellis_max_tokens(create_mesh.clamp_max_tokens(10**9))

    # Decimation: sentinel is -1 (engine_kwargs' own comment -- 0 is "off", a
    # real value the exe must receive), door range is 0..MAX_TRELLIS_DECIM.
    assert create_mesh.clamp_decim(-5) == -1
    assert create_mesh.clamp_decim(-1) == -1
    assert create_mesh.clamp_decim(0) == 0
    assert create_mesh.clamp_decim(MAX_TRELLIS_DECIM) == MAX_TRELLIS_DECIM
    assert create_mesh.clamp_decim(MAX_TRELLIS_DECIM + 1) == MAX_TRELLIS_DECIM
    check_trellis_decim(create_mesh.clamp_decim(10**9))

    # Atlas resolution: sentinel is 0 (like band/tex_res), door range is
    # MIN_TRELLIS_ATLAS..MAX_TRELLIS_ATLAS.
    assert create_mesh.clamp_atlas(-5) == 0
    assert create_mesh.clamp_atlas(0) == 0
    assert create_mesh.clamp_atlas(1) == MIN_TRELLIS_ATLAS
    assert create_mesh.clamp_atlas(MAX_TRELLIS_ATLAS) == MAX_TRELLIS_ATLAS
    assert create_mesh.clamp_atlas(MAX_TRELLIS_ATLAS + 1) == MAX_TRELLIS_ATLAS
    check_trellis_atlas(create_mesh.clamp_atlas(10**9))


def test_the_values_that_blocked_the_donut_are_still_refused_at_the_three_new_doors():
    # The clamp is the UI's half; the doors keep their own bounds, so the fix
    # must not have been made by loosening any of them.
    with pytest.raises(Invalid):
        check_trellis_max_tokens(MAX_TRELLIS_MAX_TOKENS + 1)
    with pytest.raises(Invalid):
        check_trellis_decim(MAX_TRELLIS_DECIM + 1)
    with pytest.raises(Invalid):
        check_trellis_atlas(MAX_TRELLIS_ATLAS + 1)


@pytest.mark.parametrize(
    ("field", "stored"),
    [
        ("trellis_max_tokens", MAX_TRELLIS_MAX_TOKENS + 1),
        ("trellis_decim", MAX_TRELLIS_DECIM + 1),
        ("trellis_atlas", MAX_TRELLIS_ATLAS + 1),
        ("trellis_atlas", 1),  # below MIN_TRELLIS_ATLAS but above the sentinel
    ],
)
def test_a_saved_out_of_range_engine_field_restores_as_unset(field, stored):
    form = restore_form(DEFAULT_FORM_3D, {**DEFAULT_FORM_3D, field: stored})
    assert form[field] == DEFAULT_FORM_3D[field]


@pytest.mark.parametrize(
    ("field", "stored"),
    [
        ("trellis_max_tokens", 0),
        ("trellis_max_tokens", 1),
        ("trellis_max_tokens", MAX_TRELLIS_MAX_TOKENS),
        ("trellis_decim", -1),
        ("trellis_decim", 0),
        ("trellis_decim", MAX_TRELLIS_DECIM),
        ("trellis_atlas", 0),
        ("trellis_atlas", MIN_TRELLIS_ATLAS),
        ("trellis_atlas", MAX_TRELLIS_ATLAS),
    ],
)
def test_a_saved_in_range_engine_field_survives_a_restore(field, stored):
    form = restore_form(DEFAULT_FORM_3D, {**DEFAULT_FORM_3D, field: stored})
    assert form[field] == stored
