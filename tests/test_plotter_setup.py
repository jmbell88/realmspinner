"""What a new map is, before the map exists.

Plotter answered "New map" with a 32x32 grid of 32px orthogonal cells and told
nobody. Two of those numbers are one-way -- the projection is fixed the moment
anything is painted, and the tile size is what a plain image is sliced at when
it is added -- so the interesting property of this module is not that it holds
defaults but that it makes the caps and the one-way choices *explicit* before
the map is built. The pure half is tested here; the popup that collects it is a
pane, and ``test_studio_smoke`` draws that.
"""

from __future__ import annotations

import pytest

from warlock.studio.modes.plotter import setup as plotter_setup
from warlock.studio.modes.plotter.engine import project


def test_a_blank_form_is_a_preset_rather_than_an_invention():
    """Every number the dialog opens on has to come from a row a reader can
    find, or the defaults drift away from the presets that claim to be them."""
    form = plotter_setup.blank_form()
    label, width, height, tile_w, tile_h, projection, hex_side = plotter_setup.DEFAULT
    assert plotter_setup.DEFAULT in plotter_setup.PRESETS
    assert plotter_setup.size_of(form) == (width, height, tile_w, tile_h)
    assert form["projection"] == projection
    assert form["hex_side"] == hex_side
    assert form["preset"] == label


def test_every_preset_is_a_projection_this_app_draws():
    for label, _w, _h, _tw, _th, projection, _hex_side in plotter_setup.PRESETS:
        assert projection in project.PROJECTIONS, label


def test_the_hexagonal_preset_has_a_nonzero_flat_run():
    """The 2026-09-08 audit, plotter-02: ``MapDoc`` starts every map at
    ``hex_side = 0``, which ``project.py``'s own ``Lattice`` docstring says is
    the *staggered* lattice, not the hexagonal one -- so a preset that left
    ``hex_side`` at that default would draw identically to ``"Staggered, 32
    px"`` despite the module's own comment that the two "differ by exactly the
    hex side"."""
    hexagonal = [p for p in plotter_setup.PRESETS if p[5] == project.HEXAGONAL]
    assert hexagonal, "no hexagonal preset"
    for preset in hexagonal:
        label, _w, _h, _tw, _th, _proj, hex_side = preset
        assert hex_side > 0, label

    staggered = [p for p in plotter_setup.PRESETS if p[5] == project.STAGGERED]
    assert staggered, "no staggered preset"
    for preset in staggered:
        label, _w, _h, _tw, _th, _proj, hex_side = preset
        assert hex_side == 0, f"{label} is staggered and must keep hex_side at 0"

    form = plotter_setup.apply_preset(plotter_setup.blank_form(), "Hexagonal, 32 px")
    assert form["hex_side"] > 0


def test_the_isometric_preset_is_two_to_one():
    """The preset that exists to be correct has to *be* correct: an isometric
    cell is conventionally twice as wide as it is tall, and a preset that
    shipped square would hand every new isometric map the warning."""
    iso = [p for p in plotter_setup.PRESETS if p[5] == project.ISOMETRIC]
    assert iso, "no isometric preset"
    for preset in iso:
        _label, _w, _h, tile_w, tile_h, _proj, _hex_side = preset
        assert tile_w == tile_h * 2
        assert plotter_setup.isometric_warning(_form_of(preset)) == ""


def _form_of(preset) -> dict:
    _label, width, height, tile_w, tile_h, projection, hex_side = preset
    return {
        "width": width,
        "height": height,
        "tile_w": tile_w,
        "tile_h": tile_h,
        "projection": projection,
        "hex_side": hex_side,
        "next": plotter_setup.NEXT_EMPTY,
        "preset": _label,
    }


def test_a_preset_moves_every_number_it_names():
    form = plotter_setup.blank_form()
    label, width, height, tile_w, tile_h, projection, hex_side = plotter_setup.PRESETS[-1]
    plotter_setup.apply_preset(form, label)
    assert plotter_setup.size_of(form) == (width, height, tile_w, tile_h)
    assert form["projection"] == projection
    assert form["hex_side"] == hex_side


def test_an_unknown_preset_leaves_the_numbers_alone():
    """The label is a note about where the numbers came from; the numbers are
    the answer. A typed size that reset itself to a default would be the dialog
    arguing with the user."""
    form = plotter_setup.blank_form()
    before = plotter_setup.size_of(form)
    plotter_setup.apply_preset(form, "no such preset")
    assert plotter_setup.size_of(form) == before


@pytest.mark.parametrize(
    "field,cap",
    [
        ("width", plotter_setup.MAX_TILES),
        ("height", plotter_setup.MAX_TILES),
        ("tile_w", plotter_setup.MAX_TILE_PX),
        ("tile_h", plotter_setup.MAX_TILE_PX),
    ],
)
def test_a_typed_number_is_capped_rather_than_believed(field, cap):
    """One stray digit on a field that multiplies: 512 tiles square at 32 px is
    already the 268-megapixel composite the library export warns about."""
    form = plotter_setup.blank_form()
    form[field] = 99999
    plotter_setup.clamp(form)
    assert form[field] == cap


@pytest.mark.parametrize("field", ["width", "height", "tile_w", "tile_h"])
@pytest.mark.parametrize("bad", [0, -3])
def test_a_dimension_below_one_becomes_one(field, bad):
    form = plotter_setup.blank_form()
    form[field] = bad
    plotter_setup.clamp(form)
    assert form[field] == 1


def test_an_unknown_projection_falls_back_rather_than_raising():
    """The form is typed into and restored from ``state.preview``; a stale key
    from an older session must not be able to stop the dialog drawing."""
    form = plotter_setup.blank_form()
    # A projection this build genuinely does not have. "hexagonal" used to be
    # the example and stopped being one when the editor learned to place it --
    # this test is about a *stale key*, so it needs a word that is not a
    # projection rather than one that merely was not yet.
    form["projection"] = "spiral"
    plotter_setup.clamp(form)
    assert form["projection"] == project.ORTHOGONAL


def test_the_offset_projections_survive_the_clamp():
    for projection in project.OFFSET_PROJECTIONS:
        form = plotter_setup.blank_form()
        form["projection"] = projection
        plotter_setup.clamp(form)
        assert form["projection"] == projection


def test_the_summary_states_both_units():
    """Maps are authored in tiles and exported in pixels, and the gap between
    those two numbers is the whole surprise this line exists to remove."""
    form = plotter_setup.blank_form()
    form.update(width=10, height=4, tile_w=32, tile_h=16)
    line = plotter_setup.summary(form)
    assert "10 x 4 tiles" in line
    assert "32 x 16 px" in line
    assert "320 x 64 px" in line


def test_a_square_isometric_cell_is_warned_about_and_still_allowed():
    form = plotter_setup.blank_form()
    form.update(projection=project.ISOMETRIC, tile_w=32, tile_h=32)
    assert "2:1" in plotter_setup.isometric_warning(form)
    # Still a legal size: the warning is advice, and the generator's own
    # projection control words it the same way.
    assert plotter_setup.size_of(form)[2:] == (32, 32)


def test_an_orthogonal_map_is_never_warned_about_its_ratio():
    form = plotter_setup.blank_form()
    form.update(projection=project.ORTHOGONAL, tile_w=32, tile_h=8)
    assert plotter_setup.isometric_warning(form) == ""
