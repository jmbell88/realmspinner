"""The Seed control on the 2D pane, and Mesh seed on the 3D pane, can in fact
be refused by name -- and until the 2026-09-11 audit (finding create-07),
neither drew a ring for it.

``service.validation.check_seed(name, value)`` raises
``Invalid(..., field=name)`` for a seed outside ``0..MAX_SEED`` or not a
plain ``int``, and ``_jobs_create.create_job`` calls it as
``check_seed("seed", seed)`` / ``check_seed("mesh_seed", mesh_seed)`` -- a
refusal these two controls can be named in. The comment that used to sit
beside ``settings_2d._seed_row`` claimed the opposite: "nothing in ``service``
raises a refusal naming it (the range check is fieldless...)". What actually
kept the refusal unreachable *through the widget* was incidental, not
structural -- Dear ImGui's plain InputInt stores into a C int32 whose range
happens to coincide with ``MAX_SEED = 2**31-1``. A seed can still arrive out
of range from a hand-edited ``settings.json``, since Python ints on load are
unbounded, and a submit built from it was refused with no control on either
pane ringing to say why.

Modelled on ``tests/test_field_error_wiring.py``'s source-inspection style:
neither pane's seed row can be driven without imgui and a GL context, so the
wiring is asserted on the source rather than by rendering a frame.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from warlock.service.errors import Invalid
from warlock.service.validation import MAX_SEED, check_seed


def _source(rel: str) -> str:
    from warlock import studio

    return (Path(studio.__file__).parent / rel).read_text(encoding="utf-8")


# --- the door really does refuse these by name --------------------------------


def test_check_seed_names_the_field_it_is_passed():
    """The fact the old comment on ``settings_2d._seed_row`` got wrong."""
    with pytest.raises(Invalid) as caught:
        check_seed("seed", -1)
    assert caught.value.field == "seed"

    with pytest.raises(Invalid) as caught:
        check_seed("mesh_seed", MAX_SEED + 1)
    assert caught.value.field == "mesh_seed"

    with pytest.raises(Invalid) as caught:
        check_seed("seed", 1.5)  # a float from a hand-edited settings.json
    assert caught.value.field == "seed"


# --- and now both panes actually ring it ---------------------------------------


def test_a_malformed_seed_from_a_settings_file_still_rings_the_seed_control():
    """The regression: both the 2D pane's Seed row and the 3D pane's Mesh
    seed row must ring their own control and clear the ring on edit, the same
    contract every other refusable control on these panes already keeps
    (``tests/test_field_error_wiring.py``)."""
    from warlock.studio.modes.create.ui import settings_2d

    seed_row_src = inspect.getsource(settings_2d._seed_row)
    assert 'field_error(ctx.state, "seed")' in seed_row_src, (
        "settings_2d._seed_row never rings the seed control on a refusal"
    )
    assert 'clear_field_error("seed")' in seed_row_src, (
        "settings_2d._seed_row never clears the seed ring when the control is edited"
    )

    # settings_3d's mesh-seed block is inline rather than its own function
    # (unlike settings_2d, which split it out) -- checked over the pane's
    # whole source for that reason, the same way test_field_error_wiring.py
    # checks stage_rig's bare combo.
    settings_3d_src = _source("modes/create/ui/settings_3d.py")
    assert 'field_error(ctx.state, "mesh_seed")' in settings_3d_src, (
        "settings_3d never rings the mesh_seed control on a refusal"
    )
    assert 'clear_field_error("mesh_seed")' in settings_3d_src, (
        "settings_3d never clears the mesh_seed ring when the control is edited"
    )
