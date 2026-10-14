"""The 2026-10-14 audit, fixer F12: the format-3 reader and migration, the rigged GLB import with a
scale or an up axis, the booleans' docstring, Simplify's refusal sentence, Edge Slide and
``regen.carry_over``. One claim per test; each name is the finding it closes."""

from __future__ import annotations

import time

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import (
    anim,
    legacy,
    meshimport,
    objimport,
    ops_boolean,
    ops_simplify,
    ops_slide,
    regen,
    serialize,
)
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.mesh import from_faces

# --- clay-22: dropping colliders ---------------------------------------------------------------


def _collider_chain(n: int) -> tuple[bd.ClayDoc, dict[int, legacy.LegacyFields], bd.Obj]:
    box = bp.box()
    objs: list[bd.Obj] = []
    parent = None
    for i in range(n):
        o = bd.Obj(
            uid=bd.new_uid(),
            name=f"c{i}",
            mesh=box,
            parent=parent,
            translation=np.array([0.01, 0.0, 0.0]),
        )
        objs.append(o)
        parent = o.uid
    keep = bd.Obj(
        uid=bd.new_uid(),
        name="keep",
        mesh=box,
        parent=parent,
        translation=np.array([0.0, 1.0, 0.0]),
    )
    doc = bd.ClayDoc(objects=[*objs, keep])
    fields = {o.uid: legacy.LegacyFields(role="collider") for o in objs}
    return doc, fields, keep


def test_dropping_a_long_chain_of_nested_colliders_does_not_compute_a_world_matrix_per_collider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-10-14 audit, finding clay-mesh-model-03: every doomed collider's world matrix was
    composed (an O(depth) walk each) and then thrown away; 4,000 nested colliders took 68 s."""
    n = 300
    doc, fields, keep = _collider_chain(n)
    calls = 0
    real = m3.compose

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(m3, "compose", counting)
    legacy.migrate(doc, fields)
    assert [o.name for o in doc.objects] == ["keep"]
    # One compose per object is the floor; the quadratic walk was n * n / 2 = 45,000.
    assert calls < 10 * n, f"{calls} compose calls for a chain of {n}"
    assert doc.by_uid(keep.uid).parent is None


def test_dropping_colliders_keeps_every_survivor_in_its_world_place() -> None:
    """A survivor under a chain of removed colliders is lifted onto its nearest surviving ancestor
    with its world placement unchanged, whatever the depth it was lifted from."""
    box = bp.box()
    top = bd.Obj(
        uid=bd.new_uid(),
        name="top",
        mesh=box,
        translation=np.array([1.0, 2.0, 3.0]),
        rotation=m3.quat_from_axis_angle((0.0, 1.0, 0.0), 0.7),
        scale=np.array([2.0, 2.0, 2.0]),
    )
    c1 = bd.Obj(
        uid=bd.new_uid(), name="c1", mesh=box, parent=top.uid, translation=np.array([1.0, 0.0, 0.0])
    )
    c2 = bd.Obj(
        uid=bd.new_uid(), name="c2", mesh=box, parent=c1.uid, translation=np.array([0.0, 1.0, 0.0])
    )
    low = bd.Obj(
        uid=bd.new_uid(), name="low", mesh=box, parent=c2.uid, translation=np.array([0.0, 0.0, 1.0])
    )
    mid = bd.Obj(
        uid=bd.new_uid(), name="mid", mesh=box, parent=c1.uid, translation=np.array([0.5, 0.0, 0.0])
    )
    doc = bd.ClayDoc(objects=[top, c1, c2, low, mid])
    before = {u: doc.world_matrix(u) for u in (low.uid, mid.uid)}
    legacy.migrate(
        doc,
        {
            c1.uid: legacy.LegacyFields(role="collider"),
            c2.uid: legacy.LegacyFields(role="collider"),
        },
    )
    assert [o.name for o in doc.objects] == ["top", "low", "mid"]
    assert doc.by_uid(low.uid).parent == top.uid and doc.by_uid(mid.uid).parent == top.uid
    for uid, world in before.items():
        assert np.allclose(doc.world_matrix(uid), world, atol=1e-9)


# --- clay-23: the modifier list's length ------------------------------------------------------


def test_read_fields_refuses_a_modifier_list_past_a_ceiling_before_parsing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-10-14 audit, finding clay-mesh-model-04: a count read from an untrusted file had no
    ceiling, so a million modifiers cost seconds, and a million sentences when malformed."""
    parsed = 0
    real = legacy._parse_modifier

    def counting(item):
        nonlocal parsed
        parsed += 1
        return real(item)

    monkeypatch.setattr(legacy, "_parse_modifier", counting)
    entry = {"modifiers": [{"id": i, "kind": "weld"} for i in range(5_000)]}
    out = legacy.read_fields(entry)
    assert parsed == 0, "the list was parsed before its length was checked"
    assert out.modifiers == ()
    assert len(out.unreadable) == 1 and "5,000" in out.unreadable[0]


def test_a_long_run_of_malformed_modifiers_is_a_few_sentences_not_one_each() -> None:
    entry = {"modifiers": [{"id": 1, "kind": "zz"} for _ in range(200)]}
    out = legacy.read_fields(entry)
    assert len(out.unreadable) <= 4, len(out.unreadable)
    assert any("more" in s for s in out.unreadable)
    assert any("200" in s or "197" in s or "196" in s for s in out.unreadable)


def test_a_modifier_list_under_the_ceiling_is_still_read_whole() -> None:
    entry = {"modifiers": [{"id": i, "kind": "weld"} for i in range(1, 51)]}
    out = legacy.read_fields(entry)
    assert len(out.modifiers) == 50 and out.unreadable == []


# --- clay-24: a bake that overflows float32 ----------------------------------------------------


def test_a_modifier_whose_result_overflows_float32_is_dropped_and_named_not_adopted() -> None:
    """The 2026-10-14 audit, finding clay-mesh-model-05: the migration adopted a baked mesh whose
    positions had gone to inf, and the document could then never be saved or journalled."""
    box = bp.box()
    big = bm.Mesh(
        positions=box.positions.astype("f8") * 3.0e38 / np.abs(box.positions).max(),
        loops=box.loops,
        starts=box.starts,
        material=box.material,
        smooth=box.smooth,
        uv=box.uv,
    )
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="big", mesh=big))
    mod = legacy._parse_modifier(
        {"id": 1, "kind": "radial-array", "params": {"count": 8, "angle": 360, "axis": "Y"}}
    )
    notices = legacy.migrate(doc, {obj.uid: legacy.LegacyFields(modifiers=(mod,))})
    mesh = doc.by_uid(obj.uid).mesh
    assert np.isfinite(mesh.positions).all()
    assert not any("applied to the mesh of big" in n for n in notices), notices
    assert any("could not run" in n and "big" in n for n in notices), notices
    serialize.rblk_bytes(doc)  # the document can still be saved


# --- clay-05: a rigged GLB imported at a scale and an up axis ----------------------------------


def _rigged_glb() -> tuple[bytes, dict[str, np.ndarray]]:
    """hip (1 m up) > knee (1 m up, keyed on all three properties) > ankle (1 m up, unkeyed)."""
    doc = bd.ClayDoc()
    hip = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Hip",
            mesh=bp.box(),
            bone=True,
            translation=np.array([0.0, 1.0, 0.0]),
        )
    ).uid
    knee = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Knee",
            mesh=bp.box(),
            bone=True,
            parent=hip,
            translation=np.array([0.0, 1.0, 0.0]),
        )
    ).uid
    doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Ankle",
            mesh=bp.box(),
            bone=True,
            parent=knee,
            translation=np.array([0.0, 1.0, 0.0]),
        )
    )
    doc.add_animation("Walk")
    quarter = m3.quat_from_axis_angle((1.0, 0.0, 0.0), np.pi / 2)
    doc.key_pose(
        {
            knee: {
                "translation": (0.5, 1.0, 0.25),
                "rotation": tuple(quarter),
                "scale": (1.0, 2.0, 3.0),
            }
        }
    )
    return glbwrite.write_glb(bd.to_model(doc)), {}


def _posed_worlds(doc: bd.ClayDoc) -> dict[str, np.ndarray]:
    over = anim.pose(doc.animation(), 0.0)
    return {o.name: doc.posed_world(o.uid, over) for o in doc.objects if o.bone}


@pytest.mark.parametrize(
    "options",
    [{"scale": 0.01}, {"up": "z"}, {"scale": 2.0, "up": "z"}],
    ids=["scale", "up-z", "scale-and-up-z"],
)
def test_a_rigged_glb_imported_at_a_scale_and_up_axis_plays_its_keys_in_the_same_frame(
    options: dict,
) -> None:
    """The 2026-10-14 audit, finding clay-io-01: the bone objects were conjugated by the import
    matrix and their animation keys were not, so a clip played with the bones in the wrong place."""
    blob, _ = _rigged_glb()
    plain = _posed_worlds(meshimport.import_file(blob, ".glb"))
    moved = _posed_worlds(meshimport.import_file(blob, ".glb", **options))
    matrix = objimport.axis_matrix(scale=options.get("scale", 1.0), up=options.get("up", "y"))
    assert set(plain) == {"Hip", "Knee", "Ankle"}
    for name, world in plain.items():
        expected = matrix @ world @ np.linalg.inv(matrix)
        assert np.allclose(moved[name], expected, atol=1e-6), name


# --- clay-70: the booleans' docstring ----------------------------------------------------------


def test_ops_boolean_docstring_does_not_claim_clay_lacks_the_booleans() -> None:
    doc = ops_boolean.__doc__ or ""
    assert "no longer offers" not in doc
    assert "2026-10-13" in doc and "Union" in doc
    assert "legacy" in doc and "characters" in doc


# --- clay-69: Simplify's refusal sentence ------------------------------------------------------


def test_simplify_refusal_names_the_cost_it_charged_for_an_ngon_mesh() -> None:
    """The 2026-10-14 audit, finding clay-mesh-ops-04: 950 twenty-gons (19,000 corners) were refused
    with a sentence naming a 20,000-corner ceiling the mesh is under."""
    n_faces, sides = 950, 20
    ring = np.arange(sides)
    positions = np.zeros((sides * n_faces, 3))
    faces = []
    for f in range(n_faces):
        base = f * sides
        angle = 2 * np.pi * ring / sides
        positions[base : base + sides] = np.stack(
            [np.cos(angle) + 3.0 * f, np.sin(angle), np.zeros(sides)], axis=1
        )
        faces.append((base + ring).tolist())
    mesh = from_faces(positions, faces)
    assert len(mesh.loops) == 19_000 and ops_simplify.MAX_SIMPLIFY_CORNERS > 19_000
    cost = ops_simplify.simplify_cost(mesh)
    assert cost > ops_simplify.MAX_SIMPLIFY_CORNERS
    with pytest.raises(el.OpError) as raised:
        ops_simplify.simplify(mesh, 10)
    message = str(raised.value)
    assert f"{cost:,}" in message, message
    assert "-- 20,000 corners" not in message, message


# --- clay-06 / clay-26: Edge Slide -------------------------------------------------------------


def _ring_cylinder(sides: int, order: list[int]) -> tuple[bm.Mesh, list[int]]:
    """Three rings at y = -1, 0, 1 closed into quads, vertices renumbered by *order*
    (``order[logical] -> stored``). -> ``(mesh, the middle ring's stored indices)``."""
    count = 3 * sides
    positions = np.zeros((count, 3))
    for r, y in enumerate((-1.0, 0.0, 1.0)):
        for i in range(sides):
            a = 2 * np.pi * i / sides
            positions[order[r * sides + i]] = (np.cos(a), y, np.sin(a))
    faces = []
    for r in range(2):
        for i in range(sides):
            j = (i + 1) % sides
            faces.append(
                [
                    order[r * sides + i],
                    order[r * sides + j],
                    order[(r + 1) * sides + j],
                    order[(r + 1) * sides + i],
                ]
            )
    return from_faces(positions, faces), [order[sides + i] for i in range(sides)]


def _loop_edges(ring: list[int]) -> np.ndarray:
    return np.array([[ring[i], ring[(i + 1) % len(ring)]] for i in range(len(ring))], dtype="i4")


def test_edge_slide_moves_every_vertex_of_a_loop_the_same_way_on_a_renumbered_mesh() -> None:
    """The 2026-10-14 audit, finding clay-mesh-ops-01: the first rail was the neighbour with the
    lower vertex index, so on a mesh not numbered along the strip a loop was torn in two."""
    sides = 16
    rng = np.random.default_rng(7)
    order = rng.permutation(3 * sides).tolist()
    mesh, ring = _ring_cylinder(sides, order)
    sel = el.ElementSel(edges=_loop_edges(ring))
    out, _ = ops_slide.edge_slide(mesh, sel, t=0.5)
    bm.validate(out)
    heights = out.positions[ring, 1]
    assert np.allclose(heights, heights[0], atol=1e-9), heights.round(3).tolist()
    assert abs(abs(heights[0]) - 0.5) < 1e-9
    # Flipping the sign sends the whole loop to the other side.
    back, _ = ops_slide.edge_slide(mesh, sel, t=-0.5)
    assert np.allclose(back.positions[ring, 1], -heights[0], atol=1e-9)


def test_edge_slide_on_a_generator_numbered_loop_is_still_one_height() -> None:
    mesh = bp.cylinder(1.0, 2.0, 12)
    a = adj.adjacency(mesh)
    ys = mesh.positions[:, 1]
    # Any edge whose two ends sit at the same height strictly between the caps is a loop edge only
    # on a mesh with a middle ring; the cylinder has none, so cut one first via the strip helper.
    del a, ys
    order = list(range(36))
    mesh, ring = _ring_cylinder(12, order)
    out, _ = ops_slide.edge_slide(mesh, el.ElementSel(edges=_loop_edges(ring)), t=0.4)
    heights = out.positions[ring, 1]
    assert np.allclose(heights, heights[0])


def test_edge_slide_on_a_high_valence_vertex_stays_under_the_frame_budget_or_refuses() -> None:
    """The 2026-10-14 audit, finding clay-mesh-ops-03: one spoke of a 4,000-triangle fan ran an
    O(valence^2) pure-Python pair search, 5.4 s for two touched vertices."""
    n = 4_000
    angle = 2 * np.pi * np.arange(n) / n
    positions = np.vstack(
        [[0.0, 0.0, 0.0], np.stack([np.cos(angle), np.zeros(n), np.sin(angle)], 1)]
    )
    faces = [[0, 1 + i, 1 + (i + 1) % n] for i in range(n)]
    fan = from_faces(positions, faces)
    sel = el.ElementSel(edges=np.array([[0, 1]], dtype="i4"))
    start = time.perf_counter()
    try:
        out, _ = ops_slide.edge_slide(fan, sel, t=0.5)
    except el.OpError:
        out = None  # a refusal is the other allowed answer
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"{elapsed:.2f} s for one spoke of a {n}-triangle fan"
    if out is not None:
        bm.validate(out)


def test_edge_slide_refuses_a_hub_with_more_rails_than_it_can_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ops_slide, "MAX_SLIDE_RAILS", 5)
    n = 12
    angle = 2 * np.pi * np.arange(n) / n
    positions = np.vstack(
        [[0.0, 0.0, 0.0], np.stack([np.cos(angle), np.zeros(n), np.sin(angle)], 1)]
    )
    fan = from_faces(positions, [[0, 1 + i, 1 + (i + 1) % n] for i in range(n)])
    with pytest.raises(el.OpError, match="past the 5"):
        ops_slide.edge_slide(fan, el.ElementSel(edges=np.array([[0, 1]], dtype="i4")), t=0.5)


def test_the_blocked_pair_search_agrees_with_the_one_pair_at_a_time_search() -> None:
    """The vectorised search picks the same pair, ties included (first in (i, j) order)."""
    rng = np.random.default_rng(3)
    for trial in range(30):
        k = int(rng.integers(3, 600))
        pts = rng.normal(size=(k + 1, 3))
        if trial % 3 == 0:
            pts = np.round(pts)  # many exact ties
        far = list(range(1, k + 1))
        units = []
        for fv in far:
            d = pts[fv] - pts[0]
            n = float(np.linalg.norm(d))
            units.append(d / n if n > 1e-12 else d)
        best = None
        for i in range(k):
            for j in range(i + 1, k):
                score = float(units[i] @ units[j])
                if best is None or score < best[0]:
                    best = (score, far[i], far[j])
        assert ops_slide._most_opposite(pts, 0, far) == (best[1], best[2]), (trial, k)


# --- clay-61: carry_over ------------------------------------------------------------------------


def test_carry_over_does_not_carry_uv_across_a_rebuild_with_a_different_corner_count() -> None:
    """The 2026-10-14 audit, finding clay-document-12: with ``changed_keys`` omitted the same face
    count was trusted as the same faces, and a uv array of the wrong length rode on."""
    old, new = bp.uv_sphere(0.5, 6, 4), bp.uv_sphere(0.5, 4, 6)
    assert bm.face_count(old) == bm.face_count(new) and len(old.loops) != len(new.loops)
    old = bm.Mesh(
        positions=old.positions,
        loops=old.loops,
        starts=old.starts,
        material=old.material,
        smooth=old.smooth,
        uv=np.zeros((len(old.loops), 2), dtype="f4"),
    )
    bm.validate(regen.carry_over(old, new))


def test_carry_over_still_carries_everything_when_the_faces_are_the_same_faces() -> None:
    old = bp.uv_sphere(0.5, 6, 4)
    old = bm.Mesh(
        positions=old.positions,
        loops=old.loops,
        starts=old.starts,
        material=np.full(bm.face_count(old), 3, dtype="i4"),
        smooth=old.smooth,
        uv=np.full((len(old.loops), 2), 0.25, dtype="f4"),
    )
    out = regen.carry_over(old, bp.uv_sphere(0.9, 6, 4))
    assert np.array_equal(out.uv, old.uv) and np.array_equal(out.material, old.material)
