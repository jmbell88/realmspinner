"""The symbols for the two node kinds that have no geometry.

Each test's name is its claim. The substance of this module is that a light and
a camera were **invisible** before it: the resolver gave them a ``Placed`` with
``ref is None``, and the composite draws only placements that have a ref, so
placing a point light put a row in the outliner and changed the picture not at
all. So the first test here is a regression against that in the only form it can
take -- the marker pass answers for a light and does not for a mesh -- and the
rest pin the three decisions that make a symbol readable: a fixed world size, a
cone at its own angle, and a node's scale ignored.

The shape builders and ``_shape_for``/``_placement`` need no GL and are asserted
headlessly; ``SceneMarks.draws`` needs a context and takes the ``gl`` fixture.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from warlock.kernels.geom3d import math3d as m3
from warlock.studio.modes.mason.engine import document as md
from warlock.studio.modes.mason.engine import nodes as nd
from warlock.studio.modes.mason.engine import scene as msc
from warlock.studio.modes.mason.ui import marks as mason_marks


def _doc_with_a_light_and_a_mesh() -> md.MasonDoc:
    doc = md.MasonDoc()
    doc.add_node(nd.LightNode(uid=nd.new_uid(), name="Bulb", kind="point"))
    doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Prop"))
    return doc


# --- the gap this module closes ----------------------------------------------


def test_a_light_and_a_camera_get_a_symbol_and_a_mesh_does_not() -> None:
    """The whole reason this module exists.

    A ``LightNode`` and a ``CameraNode`` resolve to a placement with no ref and
    therefore draw nothing through the GPU cache -- which is keyed on a ref. Each
    has to be described as an overlay instead or it is invisible; a mesh must
    *not* be, or it would draw twice.
    """
    assert mason_marks._shape_for(nd.LightNode(uid=1, kind="point")) is not None
    assert mason_marks._shape_for(nd.CameraNode(uid=2)) is not None
    assert mason_marks._shape_for(nd.MeshNode(uid=3)) is None
    assert mason_marks._shape_for(nd.GroupNode(uid=4)) is None
    assert mason_marks._shape_for(nd.TerrainNode(uid=5)) is None


def test_every_light_kind_has_its_own_shape() -> None:
    """Three kinds, three symbols: a bulb, a cone and parallel rays. One glyph
    for all three would make the only difference between a sun and a spot a
    colour nobody has a legend for."""
    shapes = {
        kind: mason_marks._shape_for(nd.LightNode(uid=1, kind=kind))[0]
        for kind in ("point", "spot", "directional")
    }
    assert len(set(shapes.values())) == 3


# --- the three decisions that make a symbol readable -------------------------


def test_a_spot_is_drawn_at_its_own_outer_cone_angle() -> None:
    """The rim radius is ``THROW * tan(outer)``, which is what makes widening
    the cone in Properties visibly widen the cone in the viewport -- and it is a
    matrix, so it costs no upload."""
    narrow = mason_marks._shape_for(nd.LightNode(uid=1, kind="spot", outer_cone_angle=0.2))
    wide = mason_marks._shape_for(nd.LightNode(uid=2, kind="spot", outer_cone_angle=1.0))
    assert narrow[1][0] == pytest.approx(mason_marks.THROW * math.tan(0.2))
    assert wide[1][0] > narrow[1][0]
    # And the length along -Z is the throw either way: a wider cone is wider,
    # not longer.
    assert narrow[1][2] == wide[1][2] == mason_marks.THROW


def test_a_camera_frustum_widens_with_its_field_of_view() -> None:
    tight = mason_marks._shape_for(nd.CameraNode(uid=1, yfov=0.4))
    loose = mason_marks._shape_for(nd.CameraNode(uid=2, yfov=1.2))
    assert loose[1][1] > tight[1][1]
    # Wider than it is tall, by the assumed aspect -- a square frustum would
    # read as a camera nobody has ever used.
    assert tight[1][0] == pytest.approx(tight[1][1] * mason_marks.MARK_ASPECT)


def test_a_marker_keeps_its_size_however_the_node_is_scaled() -> None:
    """A light scaled to five is not a bigger light -- ``intensity`` is the
    field that says how much light there is. A symbol that grew with the scale
    would say something the document does not mean, and would stop matching the
    fixed radius ``pick.ray_marker`` tests a click against.
    """
    world = m3.translation(m3.vec3(2.0, 3.0, 4.0)) @ m3.scaling((5.0, 5.0, 5.0))
    placement = mason_marks._placement(world)
    # The translation survives; the basis is unit-length again.
    assert placement[:3, 3] == pytest.approx([2.0, 3.0, 4.0])
    for column in range(3):
        assert float(np.linalg.norm(placement[:3, column])) == pytest.approx(1.0)


def test_a_marker_keeps_the_rotation_it_is_pointed_by() -> None:
    """The other half: normalizing the scale away must not normalize the
    *rotation* away, or a spot cone and a camera frustum would all face -Z in
    world space and the symbol would stop saying which way anything looks."""
    rotated = m3.compose(
        m3.vec3(0.0, 0.0, 0.0), m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), math.pi / 2),
        m3.vec3(3.0, 3.0, 3.0),
    )
    placement = mason_marks._placement(rotated)
    forward = placement[:3, :3] @ np.array([0.0, 0.0, -1.0])
    expected = np.asarray(rotated)[:3, :3] @ np.array([0.0, 0.0, -1.0])
    expected = expected / np.linalg.norm(expected)
    assert forward == pytest.approx(expected, abs=1e-9)


def test_the_pick_radius_and_the_drawn_size_are_one_number() -> None:
    """Imported, not restated: a symbol drawn at one size and clicked at another
    is a control whose hit area is a lie, and the two would drift the first time
    either was tuned."""
    from warlock.studio.modes.mason.engine import pick as mpick

    assert mason_marks.MARK_SIZE is mpick.MARK_SIZE


# --- the draw list, against a real context -----------------------------------


@pytest.fixture
def marks(gl):
    from warlock.studio.viewer.render import Renderer

    renderer = Renderer(gl)
    made = mason_marks.SceneMarks(gl, renderer.programs)
    yield made
    made.release()
    renderer.release()


def test_one_draw_item_per_light_and_camera_and_none_for_a_mesh(marks) -> None:
    doc = _doc_with_a_light_and_a_mesh()
    doc.add_node(nd.CameraNode(uid=nd.new_uid(), name="Cam"))
    items = marks.draws(msc.resolve(doc), ())
    assert len(items) == 2


def test_a_hidden_light_draws_no_symbol(marks) -> None:
    """"Hidden nodes do not render, export or pick" is the sentence the
    outliner's eye button makes; an overlay that ignored it would make the
    symbol the one thing hiding a light does not hide."""
    doc = md.MasonDoc()
    light = doc.add_node(nd.LightNode(uid=nd.new_uid(), kind="point"))
    assert len(marks.draws(msc.resolve(doc), ())) == 1
    doc.set_props(light.uid, visible=False)
    assert marks.draws(msc.resolve(doc), ()) == []


def test_the_selected_light_is_drawn_in_the_selection_colour(marks) -> None:
    doc = md.MasonDoc()
    light = doc.add_node(nd.LightNode(uid=nd.new_uid(), kind="point"))
    idle = marks.draws(msc.resolve(doc), ())[0].color
    chosen = marks.draws(msc.resolve(doc), {light.uid})[0].color
    assert idle != chosen
    assert chosen[:3] == pytest.approx(mason_marks._rgb(mason_marks.ACTIVE))


def test_a_symbol_is_drawn_as_lines_and_is_depth_tested(marks) -> None:
    """Both are decisions the module docstring argues: lines because a light is
    a symbol rather than a prop that exports, and depth-tested because a light
    drawn over the wall in front of it would make the viewport unreadable about
    what is behind what."""
    import moderngl

    doc = md.MasonDoc()
    doc.add_node(nd.LightNode(uid=nd.new_uid(), kind="spot"))
    item = marks.draws(msc.resolve(doc), ())[0]
    assert item.mode == moderngl.LINES
    assert item.depth is True
