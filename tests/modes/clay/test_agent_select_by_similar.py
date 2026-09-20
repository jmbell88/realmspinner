"""``clay_select_by``'s six "similar" queries (tranche 5's kernel additions --
``kernels/mesh/select.py``'s own "similar: a property of a seed set" section:
``similar_area``, ``similar_normal``, ``similar_material``, ``similar_sides``,
``similar_length``, ``similar_valence``), proved through the real agent tool
rather than only the pure kernel functions ``tests/modes/clay/
test_select_similar.py`` already exercises.

**Coverage.** Each test below computes the identical ``select.similar_*``
call directly and asserts ``clay_select_by``'s own selection matches it
exactly -- the same "cannot drift from the verb it wraps" shape
``test_agent_clay.py``'s own
``test_clay_select_by_loop_selects_the_ring_of_edges_a_human_alt_click_would``
already uses for ``loop``/``ring``/``normal``.

**Refusals.** ``schema.py``'s ``_QUERY_ARG_SCHEMAS`` declared a shape for
``faces``/``edges``/``verts``/``tolerance`` (the four argument names these
six queries add) that ``validate.py``'s ``_validate_query_arg`` never
checked -- every one of those declared constraints was a promise
``_h_select_by`` did not keep, caught by ``tests/test_agent_schemas.py``'s
generic discovery walk and fixed in ``_validate_query_arg`` itself, not
here. This file is what proves the fix from the handler's own public
surface, one malformed shape at a time: wrong type, out of range (negative),
an empty seed list, and an edge pair of the wrong length.

Shares ``_Ctx``/``_new_agent_tab``/``_payload`` with ``test_agent_clay.py``
rather than duplicating them -- see that module's own fixtures.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from realmspinner.kernels.mesh import select as clay_select_mod
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

from .test_agent_clay import _Ctx, _new_agent_tab, _payload

# A quad and a triangle sharing an edge -- the one mesh in this file with
# mixed face arity, which is what similar_sides needs to have anything to
# distinguish (a box's six faces are all quads).
_QUAD_AND_TRIANGLE = {
    "positions": [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0],
    ],
    "faces": [[0, 1, 2, 3], [1, 4, 2]],
}


def _doc(ctx: _Ctx, session: agent_clay.Session):
    return clay_mode.ensure(ctx).get(session.tab_uid).doc


def _new_box(ctx: _Ctx, session: agent_clay.Session, size=(1.0, 1.0, 1.0)) -> int:
    """A fresh box, a given size -- ``clay_add_primitive``'s own ``params``,
    not ``_new_agent_tab``'s plain default, so ``similar_area``/
    ``similar_length`` have a mesh whose faces/edges are not all the same
    size to actually tell apart."""
    result = agent_clay.call(
        ctx, session, "clay_add_primitive",
        {"generator": "box", "params": {"size": list(size)}},
    )
    assert result["isError"] is False, result
    return _payload(result)["uid"]


def _set_element_mode(ctx: _Ctx, session: agent_clay.Session, mode: str) -> None:
    result = agent_clay.call(ctx, session, "clay_element_mode", {"mode": mode})
    assert result["isError"] is False, result


# --- coverage: the real tool matches the kernel verb it wraps -----------------


def test_similar_area_selects_the_four_square_faces_of_a_stretched_box() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_box(ctx, session, size=(2.0, 1.0, 1.0))
    mesh = _doc(ctx, session).by_uid(uid).mesh
    areas = 0.5 * np.linalg.norm(clay_select_mod.bm.face_normals(mesh), axis=1)
    # A 2x1x1 box: the +-X end caps are 1x1 (area 1), the other four faces
    # are 2x1 (area 2) -- the seed below is one of those four.
    seed = int(np.flatnonzero(np.isclose(areas, 2.0))[0])
    expected = set(clay_select_mod.similar_area(mesh, [seed], tolerance=0.01).tolist())
    assert len(expected) == 4, "a 2x1x1 box has exactly four area-2 faces"

    _set_element_mode(ctx, session, "face")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_area", "faces": [seed], "tolerance": 0.01},
    )
    assert result["isError"] is False, result
    got = set(_doc(ctx, session).element_sel_of(uid).faces.tolist())
    assert got == expected


def test_similar_normal_selects_only_the_seed_face_at_a_narrow_tolerance() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    mesh = _doc(ctx, session).by_uid(uid).mesh
    expected = set(clay_select_mod.similar_normal(mesh, [0], tolerance=1.0).tolist())
    assert expected == {0}, "no two faces of a cube share a normal within 1 degree"

    _set_element_mode(ctx, session, "face")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_normal", "faces": [0], "tolerance": 1.0},
    )
    assert result["isError"] is False, result
    got = set(_doc(ctx, session).element_sel_of(uid).faces.tolist())
    assert got == expected


def test_similar_material_selects_every_face_sharing_the_seeds_slot() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    doc = _doc(ctx, session)
    obj = doc.by_uid(uid)
    material = obj.mesh.material.copy()
    material[2] = 1  # one face repainted onto a second slot
    doc.set_mesh(uid, replace(obj.mesh, material=material), keep_generator=True)
    mesh = doc.by_uid(uid).mesh
    expected = set(clay_select_mod.similar_material(mesh, [0]).tolist())
    assert 2 not in expected, "face 2 carries the repainted slot, not the seed's"

    _set_element_mode(ctx, session, "face")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_material", "faces": [0]},
    )
    assert result["isError"] is False, result
    got = set(doc.element_sel_of(uid).faces.tolist())
    assert got == expected


def test_similar_sides_at_zero_tolerance_keeps_the_quad_and_triangle_apart() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    added = agent_clay.call(ctx, session, "clay_add_mesh", _QUAD_AND_TRIANGLE)
    assert added["isError"] is False, added
    uid = _payload(added)["uid"]
    mesh = _doc(ctx, session).by_uid(uid).mesh
    expected = set(clay_select_mod.similar_sides(mesh, [0], tolerance=0).tolist())
    assert expected == {0}, "the seed quad must not match the triangle at zero tolerance"

    _set_element_mode(ctx, session, "face")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_sides", "faces": [0], "tolerance": 0},
    )
    assert result["isError"] is False, result
    got = set(_doc(ctx, session).element_sel_of(uid).faces.tolist())
    assert got == expected


def test_similar_length_selects_every_edge_matching_the_seeds_length() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_box(ctx, session, size=(2.0, 1.0, 1.0))
    mesh = _doc(ctx, session).by_uid(uid).mesh
    expected_pairs = clay_select_mod.similar_length(
        mesh, np.array([[0, 1]], dtype="i4"), tolerance=0.01
    )
    expected = {tuple(sorted(p)) for p in expected_pairs.tolist()}
    assert len(expected) == 4, "a 2x1x1 box has four length-2 edges"

    _set_element_mode(ctx, session, "edge")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_length", "edges": [[0, 1]], "tolerance": 0.01},
    )
    assert result["isError"] is False, result
    got = {tuple(sorted(p)) for p in _doc(ctx, session).element_sel_of(uid).edges.tolist()}
    assert got == expected


def test_similar_valence_selects_every_corner_of_a_cube() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    mesh = _doc(ctx, session).by_uid(uid).mesh
    expected = set(clay_select_mod.similar_valence(mesh, [0], tolerance=0).tolist())
    assert len(expected) == 8, "every corner of a cube has the same valence"

    _set_element_mode(ctx, session, "vertex")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_valence", "verts": [0], "tolerance": 0},
    )
    assert result["isError"] is False, result
    got = set(_doc(ctx, session).element_sel_of(uid).verts.tolist())
    assert got == expected


# --- refusals: one per malformed shape _QUERY_ARG_SCHEMAS declares ------------


def test_similar_area_refuses_a_non_integer_seed_face() -> None:
    """Wrong type."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    _set_element_mode(ctx, session, "face")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_area", "faces": ["not-a-number"], "tolerance": 0.1},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "faces"


def test_similar_area_refuses_a_negative_seed_face() -> None:
    """Out of range -- ``_QUERY_ARG_SCHEMAS["faces"]`` declares each item's
    own ``minimum: 0``."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    _set_element_mode(ctx, session, "face")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_area", "faces": [-1], "tolerance": 0.1},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "faces"


def test_similar_area_refuses_an_empty_seed_list() -> None:
    """Empty list -- ``_QUERY_ARG_SCHEMAS["faces"]`` declares ``minItems: 1``."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    _set_element_mode(ctx, session, "face")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_area", "faces": [], "tolerance": 0.1},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "faces"


def test_similar_length_refuses_an_edge_pair_of_the_wrong_length() -> None:
    """An edge pair with three entries instead of two."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    _set_element_mode(ctx, session, "edge")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_length", "edges": [[0, 1, 2]], "tolerance": 0.1},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "edges"


def test_similar_length_refuses_a_one_entry_edge_pair() -> None:
    """The other wrong-length edge pair -- one entry rather than two."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    _set_element_mode(ctx, session, "edge")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_length", "edges": [[0]], "tolerance": 0.1},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "edges"


def test_similar_length_refuses_a_non_list_edges_argument() -> None:
    """Wrong type."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    _set_element_mode(ctx, session, "edge")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_length", "edges": "not-a-list", "tolerance": 0.1},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "edges"


def test_similar_length_refuses_an_empty_edges_list() -> None:
    """Empty list."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    _set_element_mode(ctx, session, "edge")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_length", "edges": [], "tolerance": 0.1},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "edges"


def test_similar_valence_refuses_an_empty_seed_list() -> None:
    """Empty list."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    _set_element_mode(ctx, session, "vertex")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_valence", "verts": [], "tolerance": 0},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "verts"


def test_similar_valence_refuses_a_negative_seed_vertex() -> None:
    """Out of range."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    _set_element_mode(ctx, session, "vertex")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_valence", "verts": [-1], "tolerance": 0},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "verts"


def test_similar_area_refuses_a_non_numeric_tolerance() -> None:
    """Wrong type."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    _set_element_mode(ctx, session, "face")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_area", "faces": [0], "tolerance": "wide"},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "tolerance"


def test_similar_area_refuses_a_negative_tolerance() -> None:
    """Out of range -- ``_QUERY_ARG_SCHEMAS["tolerance"]`` declares
    ``minimum: 0.0``."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    _set_element_mode(ctx, session, "face")
    result = agent_clay.call(
        ctx, session, "clay_select_by",
        {"uid": uid, "query": "similar_area", "faces": [0], "tolerance": -0.1},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "tolerance"
