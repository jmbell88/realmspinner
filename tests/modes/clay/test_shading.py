"""Insertion decides shading: the 2026-09-06 audit's organic-shapes decision.

The user's call that day: organic shapes -- the eight figure assemblies and
the curved primitives -- insert smooth-shaded; structural shapes keep hard
edges. The mechanism is not new: it is ``clay_ops._shade_auto``'s existing
angle rule (a face is smooth only when *every* one of its edges is under the
threshold), extracted here as :func:`clay.shading.auto_smooth` so the two
insertion doors -- ``modes/clay/ui/panes/tools.add_primitive`` for a shape off the grid
and ``modes/clay/ui/panes/tools.add_assembly`` for a figure's parts -- and the manual
"Shade Auto..." op all read one rule rather than three copies of it.

Consequence, stated by the rule itself and pinned here rather than assumed:
**a cylinder and a cone stay flat**, because every side face meets a cap at a
right angle -- see :func:`shading.auto_smooth`'s own docstring for why that is
correct for this renderer rather than a gap. Do not "fix" that by
special-casing either generator into partial smoothing.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import regen, shading
from realmspinner.kernels.mesh.adjacency import adjacency
from realmspinner.studio.modes.clay.ui.panes import props as clay_props
from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` for why it is not a
    conftest fixture."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui


# --- the extracted rule: byte identity with the pre-extraction inline code --


def _old_inline_auto_smooth(mesh: bm.Mesh, angle: float = 30.0) -> np.ndarray:
    """A verbatim copy of ``clay_ops._shade_auto``'s inline computation as it
    stood at ``git show HEAD:src/realmspinner/studio/modes/clay/ops.py`` before the
    2026-09-06 audit's extraction, kept independent of
    :func:`shading.auto_smooth` so this test cannot pass merely by calling the
    thing it exists to check.
    """
    faces = bm.face_count(mesh)
    if faces == 0:
        return np.asarray(mesh.smooth)
    limit = float(np.cos(np.radians(max(0.0, min(180.0, float(angle))))))
    normals = np.asarray(bm.face_normals(mesh), dtype="f8")
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)
    counts = np.diff(np.asarray(mesh.starts, dtype="i8"))
    face_of = np.repeat(np.arange(faces, dtype="i8"), counts)
    twin = np.asarray(adjacency(mesh).twin, dtype="i8")
    paired = np.flatnonzero(twin >= 0)
    smooth = np.ones(faces, dtype=bool)
    if len(paired):
        left, right = face_of[paired], face_of[twin[paired]]
        sharp = np.einsum("ij,ij->i", normals[left], normals[right]) < limit
        smooth[left[sharp]] = False
        smooth[right[sharp]] = False
    return smooth


@pytest.mark.parametrize(
    "build", [bp.uv_sphere, bp.cylinder, bp.box], ids=["sphere", "cylinder", "box"]
)
def test_auto_smooth_is_byte_identical_to_the_old_inline_computation(build) -> None:
    mesh = build()
    expected = _old_inline_auto_smooth(mesh)
    got = shading.auto_smooth(mesh)
    assert np.array_equal(got.smooth, expected)


def test_auto_smooth_marks_a_flipped_normal_seam_sharp_rather_than_smooth() -> None:
    """The 2026-09-07 audit's clay-03: ``auto_smooth`` only measured pairs
    where ``twin >= 0``, so a flipped-normal pair -- two faces winding the
    *same* direction around the edge they share, which is exactly what denies
    them a twin -- read as a boundary with nothing to disagree with, and both
    faces came out smooth across a seam that is a full 180 degrees off.
    ``adjacency.py``'s own module docstring says importing a real-world GLB
    routinely produces exactly this.

    The mesh is ``test_adjacency.py``'s own flipped-pair reproduction (two
    quads sharing the edge between vertices 1 and 4, the second one wound the
    same way round it): ``Adjacency.flipped_pairs`` there is the corner pair
    ``[[1, 7]]``, whose shared edge is that vertex pair -- the audit's finding
    named the edge, not the corner ids. Before this fix both faces measured
    smooth; the seam must come back sharp on both sides of it.
    """
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
        ],
        dtype="f4",
    )
    flipped = bm.Mesh(
        positions=positions,
        loops=np.array([0, 1, 4, 3, 4, 5, 2, 1], dtype="i4"),
        starts=np.array([0, 4, 8], dtype="i4"),
        material=np.zeros(2, dtype="i4"),
        smooth=np.zeros(2, dtype=bool),
    )
    bm.validate(flipped)
    assert adjacency(flipped).flipped_pairs.tolist() == [[1, 7]]

    result = shading.auto_smooth(flipped)
    assert not result.smooth.any(), "both faces of the flipped seam must be sharp"


def test_auto_smooth_marks_faces_sharp_across_a_nonmanifold_edge() -> None:
    """The 2026-09-11 audit's clay-05: ``twin >= 0`` is -1 for a boundary edge
    *and* for a non-manifold edge (``edge_uses >= 3``), and the old code
    treated both the same way -- "nothing to disagree with" -- even though a
    non-manifold edge has two or more neighbours that very much can disagree.
    Three quads share the edge between vertices 0 and 1 here, each roughly
    perpendicular to the others, so every pair across that edge is sharp; all
    three faces must come out flat. Before this fix ``twin`` was -1 for every
    corner on that edge and all three faces read as smooth.
    """
    positions = [
        [0, 0, 0], [0, 1, 0],   # shared edge: v0, v1
        [1, 0, 0], [1, 1, 0],   # face A extends +X
        [0, 0, 1], [0, 1, 1],   # face B extends +Z, sharp against A
        [-1, 0, 0], [-1, 1, 0],  # face C extends -X, sharp against both
    ]
    faces = [
        [0, 2, 3, 1],
        [0, 1, 5, 4],
        [1, 0, 6, 7],
    ]
    mesh = bm.from_faces(positions, faces)
    a = adjacency(mesh)
    edge01 = next(
        e for e in range(a.n_edges) if set(a.edge_verts[e].tolist()) == {0, 1}
    )
    assert int(a.edge_uses[edge01]) >= 3, "the shared edge must be non-manifold"

    result = shading.auto_smooth(mesh)
    assert not result.smooth.any(), "every face on the non-manifold edge must be sharp"


def test_auto_smooth_returns_the_same_object_when_nothing_changes() -> None:
    """A box is already flat everywhere the rule would leave it, so applying
    the rule must not allocate a new ``Mesh`` -- which is what lets both
    insertion doors call it unconditionally on every shape without paying for
    a GPU cache miss on the ones the rule leaves alone."""
    mesh = bp.box()
    assert shading.auto_smooth(mesh) is mesh


# --- the twelve generators, measured through the insertion door ------------


def test_a_sphere_placed_from_the_grid_arrives_smooth() -> None:
    doc = bd.ClayDoc()
    obj = clay_tools.add_primitive(None, doc, "uv_sphere")
    assert obj.mesh.smooth.all()


def test_a_box_placed_from_the_grid_arrives_flat() -> None:
    doc = bd.ClayDoc()
    obj = clay_tools.add_primitive(None, doc, "box")
    assert not obj.mesh.smooth.any()


def test_a_cylinder_placed_from_the_grid_arrives_flat() -> None:
    """Not a gap in the rule: every side quad meets a cap at a right angle, so
    every face on a capped cylinder has at least one sharp neighbour under
    :func:`shading.auto_smooth`'s angle rule. Smoothing the band while the
    caps stayed flat would average the cap normals into the rim and round the
    very edge the caps exist to define -- the reason is spelled out in
    ``shading.auto_smooth``'s own docstring. A future reader who finds a
    cylinder looking faceted in the viewport should not "fix" this by
    special-casing the generator into partial smoothing; that is the
    consequence the 2026-09-06 audit's organic-shapes decision explicitly
    accepted.
    """
    doc = bd.ClayDoc()
    obj = clay_tools.add_primitive(None, doc, "cylinder")
    assert not obj.mesh.smooth.any()


# --- a figure's parts -------------------------------------------------------


def test_a_figures_box_and_sphere_parts_come_out_flat_and_smooth() -> None:
    doc = bd.ClayDoc()
    objs = clay_tools.add_assembly(None, doc, "humanoid")
    by_name = {obj.name: obj for obj in objs}

    hand = by_name["Hand.L"]
    assert hand.generator == "box"
    assert not hand.mesh.smooth.any()

    head = by_name["Head"]
    assert head.generator == "uv_sphere"
    assert head.mesh.smooth.all()


def test_a_figures_capsule_limbs_come_out_smooth_rather_than_beaded() -> None:
    """The outcome the 2026-09-06 decision was actually for.

    A capsule has no caps, so by :func:`shading.auto_smooth`'s rule nothing on
    one is inherently sharp, and a capsule at the grid's own defaults comes
    back fully smooth. A figure's limbs did not, and the arithmetic is the
    whole story: a hemisphere divides 90 degrees by its ring count, so
    ``presets.LIMB_RINGS = 3`` stepped by exactly 30 -- precisely
    :data:`shading.DEFAULT_ANGLE`. Landing *on* the threshold is not a margin;
    quad-normal blending tipped enough bands past it that a humanoid's upper
    arm measured 33% smooth and the figure went on reading as a string of
    beads, which is the complaint the decision existed to answer.

    ``LIMB_RINGS`` is 4 since 2026-09-06: the step is 22.5 degrees, the same
    limb measures ~93%, and the silhouette is untouched because ring count is
    tessellation density rather than proportion. This test asserts the *wanted*
    number rather than the measured-today one -- it is the claim, not a pin on
    an accident.
    """
    doc = bd.ClayDoc()
    objs = clay_tools.add_assembly(None, doc, "humanoid")
    by_name = {obj.name: obj for obj in objs}

    limb = by_name["Upper arm.L"]
    assert limb.generator == "capsule"
    limb_fraction = float(limb.mesh.smooth.mean())

    standalone = shading.auto_smooth(bp.capsule())
    assert standalone.smooth.all(), "a capsule at its own defaults is fully smooth"

    assert limb_fraction > 0.85, (
        f"Upper arm.L is {limb_fraction:.0%} smooth; limbs must read as round, "
        "and a drop back towards a third means a ring count has landed on "
        "shading.DEFAULT_ANGLE again"
    )

    # A figure's boxy parts keep their hard edges under the same rule -- the
    # half of the decision that says structural geometry is left alone.
    assert not by_name["Hand.L"].mesh.smooth.any(), "a box part stays flat"


# --- surviving a properties-panel rebuild -----------------------------------
#
# ``clay_props._generator`` rebuilds the mesh from edited params and calls
# ``set_generator_params``; the rebuilt mesh always arrives flat (every
# generator does), so without ``clay.regen.carry_over`` a Shade Smooth the
# user had applied -- or the shading an insertion door had already given the
# object -- would be silently discarded the moment any field was touched.
# (``regen.carry_over`` also carries per-face *material* now; see
# ``test_clay_props_regen.py`` for that half.) ``_widget`` is monkeypatched to
# report "changed" without a live imgui frame typing into a field: the panel
# does not care whether the change came from a keystroke or from this fixed
# answer, only from ``_widget``'s return.


def _placed_box(doc: bd.ClayDoc) -> bd.Obj:
    return doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Box",
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
        )
    )


def test_a_hand_set_shading_survives_a_rebuild_that_keeps_the_face_count(monkeypatch, ui) -> None:
    doc = bd.ClayDoc()
    obj = _placed_box(doc)
    # A hand-set Shade Smooth on every face of a box -- nonsense under the
    # angle rule, and exactly the point: it must come back exactly as set,
    # not re-derived, because the rebuild below does not change the face
    # count (a box is always six quads).
    hand_set = np.ones(bm.face_count(obj.mesh), dtype=bool)
    doc.set_mesh(obj.uid, replace(obj.mesh, smooth=hand_set), keep_generator=True)

    def fake_widget(key, value, default):
        if key == "size":
            return (2.0, 1.0, 1.0), True
        return value, False

    monkeypatch.setattr(clay_props, "_widget", fake_widget)

    ui.new_frame()
    ui.begin("##host")
    clay_props._generator(doc, doc.by_uid(obj.uid))
    ui.end()
    ui.end_frame()

    rebuilt = doc.by_uid(obj.uid)
    assert tuple(rebuilt.params["size"]) == (2.0, 1.0, 1.0)
    assert np.array_equal(rebuilt.mesh.smooth, hand_set)


def test_a_rebuild_that_changes_face_count_re_derives_shading_by_the_rule(monkeypatch, ui) -> None:
    """A sphere, not a cylinder: a cylinder is flat both before this fix (no
    shading logic ran at all) and after it (the angle rule leaves a capped
    cylinder flat regardless), so a cylinder cannot tell "re-derived" apart
    from "nothing ran". A sphere can -- ``clay.shading.auto_smooth`` leaves a
    freshly built one **fully** smooth, which only a re-derive (not the
    always-flat mesh a bare rebuild produces) can reach.
    """
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Ball",
            mesh=bp.uv_sphere(segments=16, rings=4),
            generator="uv_sphere",
            params={"radius": 0.5, "segments": 16, "rings": 4},
        )
    )
    original_faces = bm.face_count(obj.mesh)
    # Deliberately wrong for a sphere, and the wrong length for the rebuilt
    # mesh too -- there is no reading of "carry this over" that could produce
    # it, so its survival would only mean the rebuild fell back to the flat
    # mesh a bare ``build()`` call returns.
    doc.set_mesh(
        obj.uid,
        replace(obj.mesh, smooth=np.zeros(original_faces, dtype=bool)),
        keep_generator=True,
    )

    def fake_widget(key, value, default):
        if key == "rings":
            return 8, True
        return value, False

    monkeypatch.setattr(clay_props, "_widget", fake_widget)

    ui.new_frame()
    ui.begin("##host")
    clay_props._generator(doc, doc.by_uid(obj.uid))
    ui.end()
    ui.end_frame()

    rebuilt = doc.by_uid(obj.uid)
    assert rebuilt.params["rings"] == 8
    assert bm.face_count(rebuilt.mesh) != original_faces, (
        "the rebuild must have actually changed face count"
    )
    # Re-derived by the rule: a sphere comes back fully smooth.
    assert rebuilt.mesh.smooth.all()


def test_carry_shading_keeps_the_old_array_verbatim_when_face_count_matches() -> None:
    """The pure half of the rebuild rule, independent of imgui entirely.

    ``_carry_shading`` moved to ``clay.regen.carry_over`` (and now carries
    ``material`` too -- ``test_regen.py`` is the module's own test); this
    keeps pinning the shading half by the name this file already searches
    for it under.
    """
    old = bp.box()
    old = replace(old, smooth=np.array([True, False, True, False, True, False]))
    rebuilt = bp.box(size=(2.0, 1.0, 1.0))
    carried = regen.carry_over(old, rebuilt, material=0)
    assert np.array_equal(carried.smooth, old.smooth)


def test_carry_shading_re_derives_when_face_count_differs() -> None:
    old = bp.cylinder(segments=8)
    old = replace(old, smooth=np.ones(bm.face_count(old), dtype=bool))
    rebuilt = bp.cylinder(segments=16)
    carried = regen.carry_over(old, rebuilt, material=0)
    assert bm.face_count(carried) == bm.face_count(rebuilt)
    assert np.array_equal(carried.smooth, shading.auto_smooth(rebuilt).smooth)
