"""What Clay's statistics overlay reports, checked against known solids.

Every number here was unavailable anywhere in Clay before the overlay existed:
the outliner counted objects and nothing counted vertices, edges, faces or
triangles -- so "is this mesh 500 triangles or 50,000" was a question the app
could not answer about the thing on screen, which is the question that decides
whether a game asset is finished.

Which makes the numbers being *right* the whole of the feature, and Euler is how
that is checked rather than asserted: V - E + F = 2 for any closed surface of
genus 0, so a cube reporting 24 edges (the corner count, which is what a cheap
implementation reports) fails here rather than being read off a screenshot by
somebody who happens to remember a cube has twelve.
"""

from __future__ import annotations

import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import viewport_hints as clay_hints


def _doc(*meshes):
    doc = bd.ClayDoc()
    uids = [
        doc.add_object(bd.Obj(uid=bd.new_uid(), name=f"o{index}", mesh=mesh)).uid
        for index, mesh in enumerate(meshes)
    ]
    return doc, uids


def _numbers(line: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for chunk in line.split("  "):
        parts = chunk.split()
        if len(parts) == 2 and parts[0].replace(",", "").isdigit():
            out[parts[1].rstrip("s")] = int(parts[0].replace(",", ""))
    return out


# --- the counts --------------------------------------------------------------


def test_a_cube_reports_the_numbers_a_cube_has():
    doc, _uids = _doc(bp.box())
    got = _numbers(clay_hints.stats(doc))
    assert got["vert"] == 8
    assert got["edge"] == 12, "24 is the corner count -- every edge is shared"
    assert got["face"] == 6
    assert got["tri"] == 12


@pytest.mark.parametrize(
    "name", ["box", "cylinder", "cone", "uv_sphere", "icosphere", "capsule", "torus"]
)
def test_every_closed_primitive_satisfies_eulers_formula(name):
    """V - E + F = 2 for a closed surface of genus 0, and 0 for a torus. The
    check that makes the edge count right rather than plausible."""
    make = getattr(bp, name)
    doc, _uids = _doc(make())
    got = _numbers(clay_hints.stats(doc))

    characteristic = got["vert"] - got["edge"] + got["face"]
    assert characteristic == (0 if name == "torus" else 2), (name, got)


def test_an_open_surface_is_not_expected_to_close():
    """A plane is one face with a boundary: 4 - 4 + 1 = 1, which is the disc's
    characteristic and not a failure."""
    doc, _uids = _doc(bp.plane())
    got = _numbers(clay_hints.stats(doc))
    assert got["vert"] - got["edge"] + got["face"] == 1


def test_an_empty_document_reports_zeros_rather_than_nothing():
    assert "0 objects" in clay_hints.stats(bd.ClayDoc())


def test_counts_are_summed_over_the_objects():
    doc, _uids = _doc(bp.box(), bp.box())
    got = _numbers(clay_hints.stats(doc))
    assert got["vert"] == 16 and got["face"] == 12


def test_stats_excludes_collider_geometry_from_the_triangle_count():
    """clay-33 (2026-09-19 audit): this summed every visible object with no
    ``role`` filter, so a collider's triangles inflated the same budget
    ``readiness.validate`` deliberately excludes them from (dev/INVARIANTS.md's
    collider-rows paragraph: "an engine does not draw them"). A collider gets
    its own line instead of vanishing outright -- still worth knowing about,
    just not mixed into "is this too heavy to render"."""
    doc, uids = _doc(bp.box())
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A collider", mesh=bp.box(), role="collider"))

    got = _numbers(clay_hints.stats(doc))
    line = clay_hints.stats(doc)

    assert got["vert"] == 8, "the collider's 8 verts must not join the render count"
    assert got["tri"] == 12, "the collider's 12 tris must not join the render count"
    assert got["object"] == 2, "the collider is still an object in the document"
    assert "12 collider tris" in line


def test_a_document_with_no_collider_names_no_collider_line():
    doc, _uids = _doc(bp.box())
    assert "collider" not in clay_hints.stats(doc)


def test_a_hidden_object_is_not_counted():
    """The overlay describes what is on screen. An object you have hidden is
    not on screen, and counting it would make the numbers disagree with the
    picture they sit on."""
    doc, uids = _doc(bp.box(), bp.box())
    doc.set_visibility({uids[0]: False})

    got = _numbers(clay_hints.stats(doc))

    assert got["vert"] == 8 and got["object"] == 1


# --- what is selected --------------------------------------------------------


def test_nothing_selected_says_nothing():
    doc, _uids = _doc(bp.box())
    assert "selected" not in clay_hints.stats(doc)


def test_object_mode_counts_objects():
    doc, uids = _doc(bp.box(), bp.box())
    doc.select(uids)
    assert "2 selected" in clay_hints.stats(doc)


@pytest.mark.parametrize(
    ("mode", "sel", "expected"),
    [
        ("vertex", {"verts": [0, 1, 2]}, "3 verts selected"),
        ("face", {"faces": [0]}, "1 face selected"),
        ("edge", {"edges": [[0, 1]]}, "1 edge selected"),
    ],
)
def test_an_element_mode_counts_the_elements_it_is_about(mode, sel, expected):
    """A face count while vertices are being picked is a number about a
    selection the user does not have."""
    doc, uids = _doc(bp.box())
    doc.element_mode = mode
    doc.set_element_sel(uids[0], el.ElementSel(**sel))

    assert expected in clay_hints.stats(doc)


def test_element_counts_are_summed_across_objects():
    doc, uids = _doc(bp.box(), bp.box())
    doc.element_mode = "face"
    for uid in uids:
        doc.set_element_sel(uid, el.ElementSel(faces=[0, 1]))

    assert "4 faces selected" in clay_hints.stats(doc)


# --- the memo ----------------------------------------------------------------


def test_the_edge_count_is_memoised_on_the_mesh_itself():
    """Keyed on the ``Mesh`` rather than an id, which is ``ClayState.manifold``'s
    rule: a mesh is immutable and every op replaces it, so ``mesh is measured``
    is exactly "this count is still about what is on screen" -- where an id
    would be recycled onto a different mesh and report the last edit's edges."""
    mesh = bp.box()
    clay_hints._EDGE_CACHE.clear()

    first = clay_hints._unique_edges(mesh)
    assert mesh in clay_hints._EDGE_CACHE
    assert clay_hints._unique_edges(mesh) == first


def test_the_memo_does_not_thrash_once_more_than_sixty_four_meshes_are_live():
    """The 2026-09-26 audit's clay-view-03: the memo used to be a plain
    ``dict`` capped at 64 entries and dropped *wholesale* past that -- so a
    scene of more than 64 objects with Stats on cleared the entire cache
    partway through every single frame's loop over its own objects, and every
    mesh missed on every single frame (386 ms/call measured at 70 objects).
    Kept alive in ``meshes`` here exactly as a real scene's own objects would
    keep their meshes alive, so nothing is evicted to make room for a later
    entry the way the old wholesale clear did."""
    clay_hints._EDGE_CACHE.clear()
    meshes = [bp.box() for _ in range(70)]
    for mesh in meshes:
        clay_hints._unique_edges(mesh)

    for mesh in meshes:
        assert mesh in clay_hints._EDGE_CACHE, "an earlier mesh must not be evicted by a later one"


def test_the_memo_evicts_on_its_own_once_a_mesh_is_no_longer_referenced():
    """A readout must not become a second undo stack (the old cache's own
    reason for capping itself, which is what made it thrash -- see the test
    above). A ``WeakKeyDictionary`` needs no cap: an entry disappears once
    nothing else holds the mesh it counted, which is exactly when a document
    edit has already replaced it and the readout has no more use for it."""
    import gc

    clay_hints._EDGE_CACHE.clear()
    for _ in range(200):
        clay_hints._unique_edges(bp.box())
    gc.collect()

    assert len(clay_hints._EDGE_CACHE) < 200, (
        "an unreferenced mesh's entry must not be pinned forever"
    )


def test_a_mesh_with_no_faces_counts_no_edges_rather_than_raising():
    class Bare:
        loops = None
        starts = None
        positions = ()

    assert clay_hints._unique_edges(Bare()) == 0
