"""Loops, rings, linked, grow, shrink, boundary and mirror pairs.

Every one of these is a thing a modeller does dozens of times an hour and none
of them existed: Clay could select an element and add another with Shift, and
that was the whole vocabulary. So selecting the ring of edges round a cylinder
meant clicking each of them, and selecting one of two shapes welded into one
mesh was not possible at all.

The counts are checked against solids whose answers are arithmetic rather than
remembered. A 4x4 grid has 25 vertices, 40 edges and 16 faces; a row of it is
four edges and the ring across that row is five, because a ring of n quads has
n+1 edges. Those are the numbers a wrong walk gets wrong.
"""

from __future__ import annotations

import time

import numpy as np

from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import select, topo


def _grid():
    return bp.grid()


def _interior_edge(mesh):
    """An edge of the grid whose endpoints both have four edges, which is where
    a loop is defined at all."""
    a = adj.adjacency(mesh)
    valence = np.zeros(len(mesh.positions), dtype=int)
    for index in range(a.n_edges):
        valence[a.edge_verts[index]] += 1
    for index in range(a.n_edges):
        pair = a.edge_verts[index]
        if valence[pair[0]] == 4 and valence[pair[1]] == 4:
            return tuple(int(v) for v in pair)
    raise AssertionError("the grid has no interior edge")


# --- loops and rings ----------------------------------------------------------


def test_a_loop_runs_the_width_of_the_grid():
    mesh = _grid()
    assert len(select.edge_loop(mesh, _interior_edge(mesh))) == 4


def test_a_ring_crosses_the_quads_and_is_one_longer():
    """A ring of n quads has n+1 edges, which is the arithmetic that catches a
    walk that stops one short or wraps one too far."""
    mesh = _grid()
    assert len(select.edge_ring(mesh, _interior_edge(mesh))) == 5


def test_a_loop_stops_at_a_pole():
    """A cube's vertices have three edges, not four, so there is no edge
    opposite the one arrived on -- and a loop that ran through one would wander
    off round the mesh. The seed alone is the honest answer."""
    mesh = bp.box()
    a = adj.adjacency(mesh)
    seed = tuple(int(v) for v in a.edge_verts[0])

    assert len(select.edge_loop(mesh, seed)) == 1


def test_a_ring_still_works_where_a_loop_does_not():
    """The two are different traversals, and the cube is the case that shows
    it: no loop, and a four-edge band round the cube."""
    mesh = bp.box()
    a = adj.adjacency(mesh)
    seed = tuple(int(v) for v in a.edge_verts[0])

    assert len(select.edge_ring(mesh, seed)) == 4


def test_a_loop_and_a_ring_both_contain_their_seed():
    mesh = _grid()
    seed = _interior_edge(mesh)
    for pairs in (select.edge_loop(mesh, seed), select.edge_ring(mesh, seed)):
        rows = {tuple(sorted(int(v) for v in row)) for row in pairs}
        assert tuple(sorted(seed)) in rows


def test_an_edge_that_is_not_on_the_mesh_selects_nothing():
    """Reached by a keystroke during a selection, so a refusal is empty rather
    than an exception -- a key that raises is a key that takes the window down."""
    mesh = _grid()
    assert len(select.edge_loop(mesh, (999, 998))) == 0
    assert len(select.edge_ring(mesh, (999, 998))) == 0


def test_a_face_loop_takes_both_strips_through_the_face():
    """A quad sits on two loops at right angles, and picking one arbitrarily
    would make the result depend on corner order rather than on anything the
    user can see."""
    mesh = _grid()
    # A corner face: 4 along one strip and 4 along the other, sharing itself.
    assert len(select.face_loop(mesh, 0)) == 7


def test_a_face_loop_refuses_a_face_that_is_not_a_quad():
    mesh = bp.cylinder()
    arity = np.diff(np.asarray(mesh.starts, dtype="i8"))
    ngon = int(np.flatnonzero(arity != 4)[0])
    assert len(select.face_loop(mesh, ngon)) == 0


# --- linked -------------------------------------------------------------------


def test_linked_takes_the_whole_shell():
    mesh = _grid()
    assert len(select.linked(mesh, [0])) == len(mesh.positions)


def test_linked_stops_at_a_shell_boundary():
    """The verb that makes two shapes welded into one mesh separable again."""
    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import ops

    far = bd.Obj(
        uid=bd.new_uid(),
        name="b",
        mesh=bp.box(),
        translation=np.array([9.0, 0.0, 0.0]),
    )
    near = bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())
    joined = ops.join([near, far])

    first = select.linked(joined, [0])

    assert len(first) == 8, "one cube's worth, not both"
    assert len(first) < len(joined.positions)


def test_linked_from_nothing_selects_nothing():
    assert len(select.linked(_grid(), [])) == 0


def _strip_mesh(n_verts: int, islands: int = 4, width: int = 2):
    """*islands* disconnected quad strips *width* faces wide, ~*n_verts* total.

    Mirrors ``dev/scripts/bench_native.py``'s ``_linked_mesh`` fixture (the one
    behind ``dev/measurements/2026-09-13-native-batch-10-candidates.md``
    §1): a long thin strip is label propagation's worst case, since its pass
    count is the strip's length, not the square root of its size. Kept as its
    own copy here because ``bench_native.py`` is a script, not an importable
    module.
    """
    length = max(1, n_verts // islands // (width + 1))
    positions, faces = [], []
    for island in range(islands):
        xx, zz = np.meshgrid(np.arange(width + 1.0), np.arange(length + 1.0), indexing="ij")
        base = sum(len(p) for p in positions)
        positions.append(
            np.stack([xx.ravel() + island * 1000.0, np.zeros(xx.size), zz.ravel()], axis=1)
        )
        faces += [
            [
                base + a
                for a in (
                    i * (length + 1) + j,
                    i * (length + 1) + j + 1,
                    (i + 1) * (length + 1) + j + 1,
                    (i + 1) * (length + 1) + j,
                )
            ]
            for i in range(width)
            for j in range(length)
        ]
    return bm.Mesh(
        positions=np.concatenate(positions),
        loops=np.asarray([c for f in faces for c in f], dtype="i4"),
        starts=topo.starts_from_counts([4] * len(faces)),
        material=np.zeros(len(faces), dtype="i4"),
        smooth=np.zeros(len(faces), dtype=bool),
    )


def _linked_by_label_propagation(mesh, verts) -> np.ndarray:
    """Reference copy of the propagation ``select.linked`` used to ship with
    (see the 2026-09-13 measurement, §1) -- kept only so the fast path can be
    checked against it, never as the shipped implementation."""
    seeds = np.unique(np.asarray(verts, dtype="i8").reshape(-1))
    count = len(mesh.positions)
    if not len(seeds) or count == 0:
        return np.zeros(0, dtype="i4")
    a = adj.adjacency(mesh)
    inside = np.zeros(count, dtype=bool)
    inside[seeds[(seeds >= 0) & (seeds < count)]] = True
    if a.n_edges == 0:
        return np.flatnonzero(inside).astype("i4")
    lo = a.edge_verts[:, 0].astype("i8")
    hi = a.edge_verts[:, 1].astype("i8")
    while True:
        grown = inside.copy()
        grown[lo[inside[hi]]] = True
        grown[hi[inside[lo]]] = True
        if bool(np.array_equal(grown, inside)):
            break
        inside = grown
    return np.flatnonzero(inside).astype("i4")


def test_linked_on_a_long_thin_strip_finishes_well_under_a_second():
    """The regression this batch is about. Label propagation's pass count is
    a strip's *length*, not its size, so four 200k-vertex-total quad strips
    two faces wide took 11.1 s (dev/measurements/
    2026-09-13-native-batch-10-candidates.md §1) -- 11 seconds on the frame
    thread for one L key. ``connected_components`` over the edge graph is
    flat in the strip's length, not linear in it, so a generous 1 s bound
    still fails hard against the unfixed propagation and passes easily
    against the fix."""
    mesh = _strip_mesh(200_000)

    start = time.perf_counter()
    result = select.linked(mesh, [0])
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"took {elapsed:.3f}s -- still O(passes x mesh)?"
    # One island's worth, not all four.
    assert 0 < len(result) < len(mesh.positions)


def test_linked_matches_label_propagation_on_several_seeded_islands():
    """Parity with the propagation ``linked`` used to be: several small
    meshes, seeds landing in one, several, and no island."""
    rng = np.random.default_rng(0)
    for islands, width, seed_islands in (
        (1, 2, (0,)),
        (3, 2, (0,)),
        (3, 2, (0, 2)),
        (4, 3, (1, 3)),
        (5, 1, ()),
    ):
        mesh = _strip_mesh(400, islands=islands, width=width)
        per_island = len(mesh.positions) // islands
        seeds = np.concatenate(
            [rng.integers(i * per_island, (i + 1) * per_island, size=2) for i in seed_islands]
        ) if seed_islands else np.zeros(0, dtype="i8")

        fast = select.linked(mesh, seeds)
        reference = _linked_by_label_propagation(mesh, seeds)

        assert fast.dtype == reference.dtype
        assert np.array_equal(np.sort(fast), np.sort(reference))


# --- more and less ------------------------------------------------------------


def test_grow_takes_one_ring_outward():
    mesh = _grid()
    corner = select.grow(mesh, [0])
    assert len(corner) == 3, "a corner vertex and its two neighbours"


def test_shrink_peels_the_border_off():
    """The definition that matters: shrinking leaves "the middle of what I
    have", which is what a user reaching for it wants."""
    mesh = _grid()
    everything = np.arange(len(mesh.positions))

    inner = select.shrink(mesh, everything)

    assert len(inner) == 9, "the 3x3 interior of a 5x5 lattice"
    assert len(inner) < len(everything)


def test_shrinking_a_selection_with_no_interior_leaves_nothing():
    mesh = _grid()
    assert len(select.shrink(mesh, [0])) == 0


def test_grow_then_shrink_returns_the_interior_rather_than_the_original():
    """They are inverses only on an infinite lattice, and saying so is the
    point: a selection touching the border loses that border to the shrink."""
    mesh = _grid()
    seed = select.linked(mesh, [0])
    assert len(select.shrink(mesh, select.grow(mesh, seed))) <= len(seed)


# --- boundary and material ----------------------------------------------------


def test_the_boundary_of_a_grid_is_its_border():
    assert len(select.boundary(_grid())) == 16


def test_a_closed_solid_has_no_boundary():
    assert len(select.boundary(bp.box())) == 0


def test_by_material_finds_the_faces_using_a_slot():
    mesh = bp.box()
    assert len(select.by_material(mesh, 0)) == 6
    assert len(select.by_material(mesh, 7)) == 0


# --- verts_of / sel_from_verts (moved down from clay_ops, 2026-09-10) ---------
#
# Pure moves, unchanged in behaviour: the proof they changed nothing is the
# existing selection-verb tests in ``tests/modes/clay/test_clay_ops.py``, which exercise
# them through the five ``_verb_*`` wrappers exactly as before. These two are
# just the functions' own tests, which they had no home for while they lived
# one level up in the ops layer.


def test_verts_of_and_sel_from_verts_round_trip_a_face_selection():
    from realmspinner.kernels.mesh import elements as el

    mesh = bp.box()
    faces = el.ElementSel(faces=[1])  # the top face: four corners, all its own

    verts = select.verts_of(mesh, faces, "face")
    assert len(verts) == 4

    back = select.sel_from_verts(mesh, verts, "face")
    assert back.faces.tolist() == [1], (
        "every corner of face 1 and nothing else -- face 1 is the only face "
        "all of whose corners are in that vertex set"
    )


def test_verts_of_and_sel_from_verts_round_trip_an_edge_selection():
    from realmspinner.kernels.mesh import elements as el

    mesh = _grid()
    edge = _interior_edge(mesh)
    edges = el.ElementSel(edges=[edge])

    verts = select.verts_of(mesh, edges, "edge")
    assert len(verts) == 2

    back = select.sel_from_verts(mesh, verts, "edge")
    assert {tuple(sorted(int(v) for v in row)) for row in back.edges} == {tuple(sorted(edge))}


# --- faces_by_normal ------------------------------------------------------------


def test_faces_by_normal_takes_the_upward_faces_of_a_box_and_not_its_sides():
    mesh = bp.box()
    faces = select.faces_by_normal(mesh, (0.0, 1.0, 0.0))
    assert faces.tolist() == [1], "face 1 is +Y in primitives.box()'s own face table -- the top"


def test_faces_by_normal_widens_with_the_angle_and_takes_nothing_at_zero_on_a_sphere():
    mesh = bp.uv_sphere(segments=16, rings=8)
    tiny = select.faces_by_normal(mesh, (0.0, 1.0, 0.0), max_angle=0.0)
    narrow = select.faces_by_normal(mesh, (0.0, 1.0, 0.0), max_angle=60.0)
    wide = select.faces_by_normal(mesh, (0.0, 1.0, 0.0), max_angle=170.0)
    assert len(tiny) == 0, "no discrete face normal lands on exactly zero degrees off"
    assert 0 < len(narrow) < len(wide)


def test_faces_by_normal_ignores_a_degenerate_face_rather_than_matching_every_direction():
    """The trap the docstring names: a zero-length normal, guarded to
    ``[0, 0, 0]`` by the same ``np.divide(..., where=...)`` rule ``shading.
    auto_smooth`` uses on the identical array, dots to exactly zero with any
    unit direction -- which is ``>= cos(max_angle)`` the moment ``max_angle``
    reaches 90 degrees, so an unguarded wide query would treat a degenerate
    face as facing every direction there is. A second, real face is the
    control: it must still match at the same wide angle, so the assertion
    proves the degenerate one is excluded *because* it is degenerate, not
    because nothing in this mesh would have matched anyway.
    """
    from realmspinner.kernels.mesh import mesh as bm

    real = bp.plane()  # one quad, facing +Y exactly, by its own docstring
    degenerate = np.full((4, 3), 2.0, dtype="f4")  # one point, four times over
    mesh = bm.Mesh(
        positions=np.concatenate([real.positions, degenerate]),
        loops=np.concatenate([real.loops, np.array([4, 5, 6, 7], dtype="i4")]),
        starts=np.concatenate([real.starts, [8]]).astype("i4"),
        material=np.concatenate([real.material, [0]]).astype("i4"),
        smooth=np.concatenate([real.smooth, [False]]),
    )

    faces = select.faces_by_normal(mesh, (0.0, 1.0, 0.0), max_angle=180.0)
    assert faces.tolist() == [0], (
        "the real quad matches at 180 degrees; the degenerate one never does"
    )


# --- faces_in_bounds ------------------------------------------------------------


def test_faces_in_bounds_takes_a_face_only_when_every_corner_is_inside():
    mesh = bp.box()
    # Every corner of the box except the top face's own sits at y = -0.5.
    faces = select.faces_in_bounds(mesh, (-10.0, 0.4, -10.0), (10.0, 10.0, 10.0))
    assert faces.tolist() == [1], "only the top face has every one of its corners at y = +0.5"


def test_faces_in_bounds_reads_the_positions_it_is_given_rather_than_the_meshs_own():
    mesh = bp.box()
    lo, hi = (-10.0, 0.4, -10.0), (10.0, 10.0, 10.0)
    assert select.faces_in_bounds(mesh, lo, hi).tolist() == [1]

    dropped = np.asarray(mesh.positions, dtype="f8") + [0.0, -1.0, 0.0]  # push the box down
    assert len(select.faces_in_bounds(mesh, lo, hi, positions=dropped)) == 0, (
        "the mesh's own positions would put the top face in these bounds; the "
        "positions actually passed in do not, and those are the ones that count"
    )


# --- QUERIES --------------------------------------------------------------------


def test_every_query_names_modes_it_can_actually_answer_in():
    grid = _grid()
    box = bp.box()
    edge = _interior_edge(grid)
    field_to_mode = {"verts": "vertex", "edges": "edge", "faces": "face"}
    fixtures = {
        "loop": (grid, {"edge": edge}),
        "ring": (grid, {"edge": edge}),
        "face_loop": (grid, {"face": 0}),
        "material": (box, {"slot": 0}),
        "normal": (box, {"direction": (0.0, 1.0, 0.0)}),
        "similar_area": (box, {"faces": [0], "tolerance": 0.1}),
        "similar_normal": (box, {"faces": [0], "tolerance": 5.0}),
        "similar_material": (box, {"faces": [0]}),
        "similar_sides": (box, {"faces": [0], "tolerance": 0}),
        "similar_length": (grid, {"edges": [edge], "tolerance": 0.5}),
        "similar_valence": (grid, {"verts": [edge[0]], "tolerance": 0}),
    }
    for name, query in select.QUERIES.items():
        if name == "bounds":
            sel = query.run(box, (-10.0, -10.0, -10.0), (10.0, 10.0, 10.0))
        else:
            mesh, kwargs = fixtures[name]
            sel = query.run(mesh, **kwargs)
        touched = [field for field in ("verts", "edges", "faces") if len(getattr(sel, field))]
        assert touched, f"query {name!r} found nothing on its own fixture"
        for field in touched:
            assert field_to_mode[field] in query.modes, name


def test_the_query_registry_does_not_duplicate_a_verb_that_is_already_an_op():
    """The anti-drift gate. ``all``, ``none``, ``invert``, ``linked``, ``more``,
    ``less`` and ``boundary`` are already ``select-*`` rows in ``clay_ops.OPS``
    and already in the agent's derived ``clay_op`` enum -- dead only because no
    element mode can be set from an agent yet, not a hole for ``QUERIES`` to
    fill a second time."""
    from realmspinner.studio.modes.clay import ops as clay_ops

    banned = {"all", "none", "invert", "linked", "more", "less", "boundary"}
    op_verbs = {
        op.name.removeprefix("select-") for op in clay_ops.OPS if op.name.startswith("select-")
    }
    assert op_verbs == banned, "this pin's own idea of the seven must match the real registry"
    assert set(select.QUERIES) & banned == set()
    assert set(select.QUERIES).isdisjoint(op_verbs)


# --- mirror pairs -------------------------------------------------------------


def test_a_symmetric_mesh_pairs_every_vertex():
    pairs = select.mirror_pairs(bp.box(), 0)
    assert len(pairs) == 8
    # And the pairing is an involution: the mirror of a vertex's mirror is
    # itself, which is what a mirrored drag depends on.
    for index, twin in pairs.items():
        assert pairs[twin] == index


def test_a_vertex_on_the_plane_maps_to_itself():
    """The case that has to be handled rather than excluded: those are the ones
    a mirrored drag must slide *along* the plane instead of moving off it."""
    mesh = bp.grid()
    pairs = select.mirror_pairs(mesh, 0)
    on_plane = [
        index
        for index in range(len(mesh.positions))
        if abs(float(mesh.positions[index][0])) < 1e-6
    ]
    assert on_plane
    for index in on_plane:
        assert pairs.get(index) == index


def test_an_asymmetric_mesh_reports_what_it_can_rather_than_pretending():
    """X-mirror can only be as good as the mesh: one that is not symmetric has
    no pairs to find, and reporting the ones it has beats inventing the rest."""
    import dataclasses

    mesh = bp.box()
    moved = np.asarray(mesh.positions, dtype="f8").copy()
    moved[0][0] += 0.5
    shifted = dataclasses.replace(mesh, positions=moved)

    pairs = select.mirror_pairs(shifted, 0)

    assert len(pairs) < len(moved)


def test_the_mirror_axis_is_a_parameter():
    box = bp.box()
    assert len(select.mirror_pairs(box, 0)) == 8
    assert len(select.mirror_pairs(box, 1)) == 8
    assert len(select.mirror_pairs(box, 2)) == 8


def test_mirror_pairs_is_reachable_from_a_live_code_path_or_its_docstring_says_it_is_not():
    """The 2026-09-19 audit's clay-26: ``mirror_pairs``'s docstring reads as
    though X-mirror editing already calls it ("what X-mirror editing needs"),
    but ``header.py``'s own docstring admits the feature is not built yet --
    X-mirror is explicitly listed there as "not here yet, deliberately".
    ``elements.restrict()`` had exactly this gap (the 2026-09-08 audit's
    clay-09) and ``test_elements.py`` closed it with this same self-adjusting
    gate; ``mirror_pairs`` never got the mirror. Either a live caller exists,
    or the docstring has to say plainly that none does, so a future caller
    does not assume X-mirror dragging is already wired up.
    """
    import inspect
    import re
    from pathlib import Path

    import realmspinner

    root = Path(realmspinner.__file__).parent
    callers = [
        path
        for path in root.rglob("*.py")
        if path.name != "select.py"
        and re.search(r"\bmirror_pairs\s*\(", path.read_text(encoding="utf-8"))
    ]
    if callers:
        return  # a live caller exists -- nothing more to prove

    doc = inspect.getdoc(select.mirror_pairs) or ""
    assert "not currently called" in doc.lower() or "not built" in doc.lower(), (
        "mirror_pairs() has no live caller anywhere under realmspinner/, but "
        "its docstring no longer admits that -- either wire it into X-mirror "
        "editing, or restore the honest docstring"
    )


# --- delete and duplicate, one undo step per gesture --------------------------
#
# The 2026-09-06 audit, finding clay-01: delete_selected and duplicate_selected
# pushed one ObjectRemoveEdit/ObjectAddEdit per object instead of one
# CompoundEdit for the whole gesture, unlike add_objects, set_visibility and
# join_objects, which this same package bundles for exactly this reason. A
# user who selected three objects and pressed Delete once got all three gone,
# but a single Ctrl+Z restored only one of them.


def _three_boxes():
    from realmspinner.kernels.mesh import document as bd

    doc = bd.ClayDoc()
    for i in range(3):
        doc.add_object(bd.Obj(bd.new_uid(), f"Box{i}", bp.box()))
    doc.history.clear()  # the three add_object calls are not part of the gesture under test
    return doc


def test_deleting_several_selected_objects_undoes_in_one_step():
    from realmspinner.kernels.mesh import selection

    doc = _three_boxes()
    doc.select([obj.uid for obj in doc.objects])

    steps_before = len(doc.history)
    selection.delete_selected(doc)

    assert len(doc.history) - steps_before == 1, "one keystroke, one undo step"
    assert doc.objects == []

    doc.undo()

    assert len(doc.objects) == 3, "one Ctrl+Z must restore every object the keystroke removed"


# The 2026-09-08 audit's second run, finding clay-10: the element-mode branch of
# delete_selected pushed one MeshEdit (via doc.set_mesh) per affected object
# instead of bundling the whole keystroke into one step the way the
# object-mode branch above it does. Three boxes, all faces selected, one
# Delete pushed three undo steps; one Ctrl+Z restored one box and left two
# empty -- a state the user never produced.


def test_deleting_selected_faces_across_several_objects_undoes_in_one_step():
    from realmspinner.kernels.mesh import elements as el
    from realmspinner.kernels.mesh import selection

    doc = _three_boxes()
    doc.set_element_mode("face")
    for obj in doc.objects:
        doc.set_element_sel(obj.uid, el.select_all(obj.mesh, "face"))
    doc.history.clear()  # the three set_element_sel calls push no steps; be explicit anyway

    face_counts_before = [len(obj.mesh.starts) - 1 for obj in doc.objects]
    assert all(count > 0 for count in face_counts_before)

    steps_before = len(doc.history)
    refusals = selection.delete_selected(doc)

    assert refusals == []
    assert len(doc.history) - steps_before == 1, "one keystroke, one undo step"
    assert all(len(obj.mesh.starts) - 1 == 0 for obj in doc.objects), "every box's faces are gone"

    doc.undo()

    assert [len(obj.mesh.starts) - 1 for obj in doc.objects] == face_counts_before, (
        "one Ctrl+Z must restore every object's faces the keystroke removed"
    )


def test_duplicating_several_selected_objects_undoes_in_one_step():
    from realmspinner.kernels.mesh import selection

    doc = _three_boxes()
    doc.select([obj.uid for obj in doc.objects])

    steps_before = len(doc.history)
    fresh = selection.duplicate_selected(doc)

    assert len(fresh) == 3
    assert len(doc.history) - steps_before == 1, "one keystroke, one undo step"
    assert len(doc.objects) == 6

    doc.undo()

    assert len(doc.objects) == 3, "one Ctrl+Z must remove every copy the keystroke made"


def test_duplicate_selected_preserves_the_objects_original_relative_order():
    """The 2026-09-14 audit's clay-07: ``duplicate_selected`` iterated
    ``doc.selection`` directly, a plain ``set``, so the copies landed in
    whatever order the set's hash buckets gave rather than the order the
    objects sit in the outliner.

    Three consecutive uids in a ``set`` iterate in ascending numeric order in
    CPython (each hashes to itself and lands in its own bucket), so reversing
    ``doc.objects`` in place -- without touching any uid -- puts the
    document's own order (index 0, 1, 2 -> Box2, Box1, Box0) at odds with
    that ascending set order (Box0, Box1, Box2). The unfixed code copies in
    the set's order; the fix copies in the document's.
    """
    from realmspinner.kernels.mesh import selection

    doc = _three_boxes()
    doc.objects.reverse()  # document order is now Box2, Box1, Box0
    doc.select([obj.uid for obj in doc.objects])
    expected_bases = [obj.name for obj in doc.objects]

    fresh = selection.duplicate_selected(doc)

    copy_bases = [doc.by_uid(uid).name.split(".")[0] for uid in fresh]
    assert copy_bases == expected_bases, (
        "the copies must land in the objects' document order, not set-iteration order"
    )
