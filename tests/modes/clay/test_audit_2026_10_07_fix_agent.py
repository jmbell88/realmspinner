"""Regression tests for the 2026-10-07 audit's Clay agent-tool findings that
close the float32 family and the loose-argument doors: clay-01 and clay-04
(the agent half), clay-17 (the inline bounds test), clay-20, clay-35, clay-36,
clay-73, clay-74, clay-75 and clay-76.

Each test's name is the claim. None of them can hang the suite: the one that
used to (a ``clay_scene`` over a mesh whose uv is infinite in float32) asserts
the refusal at the door, and drives ``_uv_facts`` itself with the overlap
measurement stubbed to raise, so a regression fails fast instead of looping.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import uvtools
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import tools as agent_tools
from realmspinner.studio.modes.clay.agent import validate as agent_validate

from .test_agent_clay import (
    _TETRA_MESH_ARGS,
    _Ctx,
    _history_len,
    _new_agent_tab,
    _payload,
)


def _doc(ctx: _Ctx, session: agent_clay.Session):
    return clay_mode.ensure(ctx).get(session.tab_uid).doc


def _call(ctx: _Ctx, session: agent_clay.Session, tool: str, **arguments: Any) -> dict:
    return agent_clay.call(ctx, session, tool, arguments)


def _refused(result: dict, field: str) -> None:
    assert result["isError"] is True, result["content"][0]["text"][:200]
    assert result["structuredContent"]["field"] == field, result["structuredContent"]
    assert result["structuredContent"].get("changed") is False


def _fresh() -> tuple[_Ctx, agent_clay.Session]:
    return _Ctx(), agent_clay.Session()


def _no_nan_text(result: dict) -> bool:
    text = result["content"][0]["text"]
    return "NaN" not in text and "Infinity" not in text


# --- clay-01 / clay-04 --------------------------------------------------------


def test_clay_add_primitive_refuses_a_param_that_overflows_float32_instead_of_hanging_clay_scene() -> (  # noqa: E501
    None
):
    """A box ``size`` of 1e300 is finite in float64, so it passed; the mesh's f4
    positions were inf, and the same call's scene row then ran the overlap
    bucketing over the garbage and never returned."""
    ctx, session = _fresh()
    result = _call(
        ctx, session, "clay_add_primitive", generator="box", params={"size": [1e300, 1, 1]}
    )
    _refused(result, "params")
    assert "size" in result["content"][0]["text"]
    # Refused before the tab was even minted: nothing was built or placed.
    assert session.tab_uid == ""

    uid = _new_agent_tab(ctx, session)
    before = _history_len(ctx, session)
    result = _call(
        ctx, session, "clay_add_primitive", generator="cylinder", params={"radius": 1e300}
    )
    _refused(result, "params")
    assert _history_len(ctx, session) == before
    scene = _call(ctx, session, "clay_scene")
    assert scene["isError"] is False and _no_nan_text(scene)
    assert [o["uid"] for o in _payload(scene)["objects"]] == [uid]


def test_every_clay_generator_refuses_a_length_past_float32_at_the_agent_door() -> None:
    """Sweep ``CLAY_GENERATOR_NAMES`` at 1e39 (past float32's 3.4e38) through
    ``clay_add_primitive`` and ``clay_set_params``: each key of each generator
    is refused by name, and nothing lands."""
    for generator in sorted(bp.CLAY_GENERATOR_NAMES):
        defaults = bp.GENERATORS[generator][0]
        for key, default in defaults.items():
            huge = [1e39] * len(default) if isinstance(default, (list, tuple)) else 1e39
            ctx, session = _fresh()
            result = _call(
                ctx, session, "clay_add_primitive", generator=generator, params={key: huge}
            )
            assert result["isError"] is True, (generator, key)
            assert result["structuredContent"]["field"] == "params", (generator, key)
            assert key in result["content"][0]["text"], (generator, key)
            assert session.tab_uid == "", (generator, key)

            uid = _new_agent_tab(ctx, session, generator)
            doc = _doc(ctx, session)
            before = _history_len(ctx, session)
            result = _call(ctx, session, "clay_set_params", uid=uid, params={key: huge})
            _refused(result, "params")
            assert _history_len(ctx, session) == before, (generator, key)
            assert np.isfinite(doc.by_uid(uid).mesh.positions).all(), (generator, key)


def test_clay_add_mesh_refuses_positions_that_overflow_float32() -> None:
    ctx, session = _fresh()
    for bad in (1e300, 1e39, 3e38):
        result = _call(
            ctx,
            session,
            "clay_add_mesh",
            positions=[[0, 0, 0], [bad, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
        )
        _refused(result, "positions")
    assert session.tab_uid == ""


def test_clay_add_mesh_refuses_a_uv_that_overflows_float32_instead_of_hanging_clay_scene() -> None:
    ctx, session = _fresh()
    result = _call(
        ctx,
        session,
        "clay_add_mesh",
        positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        uv=[[[0, 0], [1e300, 0], [0, 1]]],
    )
    _refused(result, "uv")
    assert session.tab_uid == ""


def test_clay_set_params_refuses_a_param_that_overflows_float32() -> None:
    ctx, session = _fresh()
    uid = _new_agent_tab(ctx, session, "cylinder")
    before = _history_len(ctx, session)
    result = _call(ctx, session, "clay_set_params", uid=uid, params={"height": 1e300})
    _refused(result, "params")
    assert _history_len(ctx, session) == before
    assert _no_nan_text(_call(ctx, session, "clay_scene"))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_built_mesh_that_is_not_finite_is_refused_before_the_object_is_added(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Belt and braces behind the bounds: whatever a generator (or the f4
    narrowing) produces, a mesh with an infinite position never reaches the
    document."""

    def poisoned_box(size=(1.0, 1.0, 1.0)):
        good = bp.box(size)
        pos = np.array(good.positions, dtype="f4")
        pos[0, 0] = np.inf
        return bm.Mesh(**{**good.__dict__, "positions": pos})

    monkeypatch.setitem(bp.GENERATORS, "box", (bp.GENERATORS["box"][0], poisoned_box))

    ctx, session = _fresh()
    result = _call(ctx, session, "clay_add_primitive", generator="box", params={"size": [2, 2, 2]})
    _refused(result, "params")
    assert session.tab_uid == ""

    # clay_set_params, on an object placed before the generator went bad.
    monkeypatch.undo()
    uid = _new_agent_tab(ctx, session, "box")
    monkeypatch.setitem(bp.GENERATORS, "box", (bp.GENERATORS["box"][0], poisoned_box))
    before = _history_len(ctx, session)
    result = _call(ctx, session, "clay_set_params", uid=uid, params={"size": [2, 2, 2]})
    _refused(result, "params")
    assert _history_len(ctx, session) == before

    # clay_add_mesh: from_faces handed back a non-finite mesh.
    monkeypatch.undo()
    real_from_faces = bm.from_faces

    def poisoned_from_faces(positions, faces, uv=None):
        good = real_from_faces(positions, faces, uv)
        pos = np.array(good.positions, dtype="f4")
        pos[0, 0] = np.nan
        return bm.Mesh(**{**good.__dict__, "positions": pos})

    monkeypatch.setattr(bm, "from_faces", poisoned_from_faces)
    objects = len(_doc(ctx, session).objects)
    result = _call(ctx, session, "clay_add_mesh", **_TETRA_MESH_ARGS)
    _refused(result, "positions")
    assert len(_doc(ctx, session).objects) == objects


def test_uv_facts_returns_no_overlap_for_a_mesh_whose_uv_is_not_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_uv_facts`` must not hand an infinite uv to ``overlap_faces`` (its
    bucket loop cast the garbage to int64 and did not finish); it answers a
    null overlap instead. Built by hand, past the door that now refuses it."""
    good = bm.from_faces([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], [[[0, 0], [1, 0], [0, 1]]])
    uv = np.array(good.uv, dtype="f4")
    uv[1, 0] = np.inf
    mesh = bm.Mesh(**{**good.__dict__, "uv": uv})

    def must_not_run(_mesh):
        raise AssertionError("overlap_faces was called on a non-finite uv")

    monkeypatch.setattr(uvtools, "overlap_faces", must_not_run)
    facts = agent_validate._uv_facts(mesh)
    assert facts is not None
    assert facts["overlapping_faces"] is None

    nan_uv = np.array(good.uv, dtype="f4")
    nan_uv[0, 1] = np.nan
    facts = agent_validate._uv_facts(bm.Mesh(**{**good.__dict__, "uv": nan_uv}))
    assert facts is not None and facts["overlapping_faces"] is None


# --- clay-17 ------------------------------------------------------------------


@pytest.mark.parametrize("size", [0.6, 0.1])
@pytest.mark.parametrize("space", ["local", "world"])
def test_bounds_query_selects_everything_when_the_box_is_exactly_the_objects_extent_for_a_non_f4_exact_size(  # noqa: E501
    space: str,
    size: float,
) -> None:
    """``box(0.6)`` has f4 corners at +-f4(0.3), which are not +-0.3: an
    inclusive float64 box equal to the extent used to select nothing."""
    ctx, session = _fresh()
    uid = _payload(
        _call(ctx, session, "clay_add_primitive", generator="box", params={"size": [size] * 3})
    )["uid"]
    _call(ctx, session, "clay_element_mode", mode="vertex")
    result = _call(
        ctx,
        session,
        "clay_select_by",
        uid=uid,
        query="bounds",
        min=[-size / 2] * 3,
        max=[size / 2] * 3,
        space=space,
    )
    assert result["isError"] is False, result["content"][0]["text"]
    assert _payload(result)["selected"]["verts"] == 8


# --- clay-20 ------------------------------------------------------------------


def test_clay_scene_does_not_recompute_uv_overlap_for_an_unchanged_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, session = _fresh()
    uid = _new_agent_tab(ctx, session)
    assert _call(ctx, session, "clay_uv", uid=uid, action="unwrap")["isError"] is False

    calls = []
    real = uvtools.overlap_faces

    def counting(mesh):
        calls.append(mesh)
        return real(mesh)

    monkeypatch.setattr(uvtools, "overlap_faces", counting)
    first = _payload(_call(ctx, session, "clay_scene"))
    second = _payload(_call(ctx, session, "clay_scene"))
    third = _payload(_call(ctx, session, "clay_scene"))
    assert len(calls) <= 1, "an unchanged mesh was re-measured"
    assert first["objects"][0]["uv"] == second["objects"][0]["uv"] == third["objects"][0]["uv"]

    # A mesh that is not the same object is measured afresh.
    calls.clear()
    moved = _call(ctx, session, "clay_uv", uid=uid, action="transform", translate=[0.25, 0.0])
    assert moved["isError"] is False
    assert len(calls) == 1
    again = len(calls)
    _call(ctx, session, "clay_scene")
    assert len(calls) == again


def test_the_uv_facts_memo_is_bounded() -> None:
    mesh = bm.from_faces([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], [[[0, 0], [1, 0], [0, 1]]])
    keep = []
    for _ in range(agent_validate.UV_FACTS_MEMO_MAX + 40):
        m = bm.Mesh(**{**mesh.__dict__})
        keep.append(m)
        agent_validate._uv_facts(m)
    assert len(agent_validate._UV_FACTS_MEMO) <= agent_validate.UV_FACTS_MEMO_MAX


# --- clay-35 ------------------------------------------------------------------

_BAD_INTEGERS = (1.9, True)


def _integer_paths(schema: Any, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    if not isinstance(schema, dict):
        return out
    if schema.get("type") == "integer":
        out.append(path)
    for key, sub in (schema.get("properties") or {}).items():
        out.extend(_integer_paths(sub, (*path, key)))
    if "items" in schema:
        out.extend(_integer_paths(schema["items"], (*path, "[]")))
    for sub in schema.get("anyOf", []):
        out.extend(_integer_paths(sub, path))
    return out


# Every integer argument that is not an object address. Addresses (``uid``,
# ``uids[]``, ``focus[]``) are pinned by the clay-83 tests already.
_ADDRESS_PATHS = {"uid", "uids.[]", "focus.[]"}


def _vertex_mode(ctx, session, mode="face"):
    _call(ctx, session, "clay_element_mode", mode=mode)


def _integer_doors(bad: Any):
    """(tool, field, builder) for every non-address integer path the catalogue
    declares. ``builder(ctx, session, uid)`` -> the arguments carrying *bad*."""
    tetra = _TETRA_MESH_ARGS
    return {
        ("clay_scene", "offset"): ("clay_scene", "offset", lambda c, s, u: {"offset": bad}),
        ("clay_scene", "limit"): ("clay_scene", "limit", lambda c, s, u: {"limit": bad}),
        ("clay_add_primitive", "material"): (
            "clay_add_primitive",
            "material",
            lambda c, s, u: {"generator": "box", "material": bad},
        ),
        ("clay_add_mesh", "material"): (
            "clay_add_mesh",
            "material",
            lambda c, s, u: {**tetra, "material": bad},
        ),
        ("clay_add_mesh", "faces.[].[]"): (
            "clay_add_mesh",
            "faces",
            lambda c, s, u: {**tetra, "faces": [[0, bad, 1], *tetra["faces"][1:]]},
        ),
        ("clay_material", "faces.[]"): (
            "clay_material",
            "faces",
            lambda c, s, u: {"uids": [u], "color": [1, 0, 0], "faces": [bad]},
        ),
        ("clay_material", "index"): (
            "clay_material",
            "index",
            lambda c, s, u: {"uids": [u], "index": bad},
        ),
        ("clay_select_elements", "verts.[]"): (
            "clay_select_elements",
            "verts",
            lambda c, s, u: (_vertex_mode(c, s, "vertex"), {"uid": u, "verts": [bad]})[1],
        ),
        ("clay_select_elements", "edges.[].[]"): (
            "clay_select_elements",
            "edges",
            lambda c, s, u: (_vertex_mode(c, s, "edge"), {"uid": u, "edges": [[0, bad]]})[1],
        ),
        ("clay_select_elements", "faces.[]"): (
            "clay_select_elements",
            "faces",
            lambda c, s, u: (_vertex_mode(c, s, "face"), {"uid": u, "faces": [bad]})[1],
        ),
        ("clay_select_elements", "expect_stamp"): (
            "clay_select_elements",
            "expect_stamp",
            lambda c, s, u: (
                _vertex_mode(c, s, "face"),
                {"uid": u, "faces": [0], "expect_stamp": bad},
            )[1],
        ),
        ("clay_select_by", "expect_stamp"): (
            "clay_select_by",
            "expect_stamp",
            lambda c, s, u: (
                _vertex_mode(c, s, "face"),
                {"uid": u, "query": "material", "slot": 0, "expect_stamp": bad},
            )[1],
        ),
        ("clay_select_by", "slot"): (
            "clay_select_by",
            "slot",
            lambda c, s, u: (
                _vertex_mode(c, s, "face"),
                {"uid": u, "query": "material", "slot": bad},
            )[1],
        ),
        ("clay_elements", "offset"): (
            "clay_elements",
            "offset",
            lambda c, s, u: {"uid": u, "offset": bad},
        ),
        ("clay_elements", "limit"): (
            "clay_elements",
            "limit",
            lambda c, s, u: {"uid": u, "limit": bad},
        ),
        ("clay_render", "size"): ("clay_render", "size", lambda c, s, u: {"size": bad}),
        ("clay_uv", "islands.[]"): (
            "clay_uv",
            "islands",
            lambda c, s, u: (
                _call(c, s, "clay_uv", uid=u, action="unwrap"),
                {"uid": u, "action": "transform", "islands": [bad]},
            )[1],
        ),
        ("clay_undo", "steps"): ("clay_undo", "steps", lambda c, s, u: {"steps": bad}),
        ("clay_redo", "steps"): ("clay_redo", "steps", lambda c, s, u: {"steps": bad}),
        ("clay_measure", "faces.[]"): (
            "clay_measure",
            "faces",
            lambda c, s, u: {"kind": "area", "uid": u, "faces": [bad]},
        ),
    }


def test_every_integer_argument_refuses_a_fractional_or_boolean_value() -> None:
    declared: set[tuple[str, str]] = set()
    for tool in agent_clay.tools():
        for path in _integer_paths(tool.schema):
            dotted = ".".join(path)
            if dotted in _ADDRESS_PATHS:
                continue
            declared.add((tool.name, dotted))
    walked = set(_integer_doors(1.9))
    assert declared == walked, (
        f"integer arguments with no door test: {sorted(declared - walked)}; "
        f"tests for arguments the catalogue no longer declares: {sorted(walked - declared)}"
    )

    for bad in _BAD_INTEGERS:
        for key, (tool, field, build) in _integer_doors(bad).items():
            ctx, session = _fresh()
            uid = _new_agent_tab(ctx, session)
            args = build(ctx, session, uid)
            history = _history_len(ctx, session)
            doc = _doc(ctx, session)
            selection = {k: v for k, v in doc.element_sel.items()}
            result = agent_clay.call(ctx, session, tool, args)
            assert result["isError"] is True, (key, bad, result["content"][0]["text"][:160])
            assert result["structuredContent"]["field"] == field, (key, bad)
            assert _history_len(ctx, session) == history, (key, bad)
            assert doc.element_sel == selection or not selection, (key, bad)


def test_clay_measure_area_refuses_a_string_of_digits_and_does_not_double_count_a_repeated_face() -> (  # noqa: E501
    None
):
    ctx, session = _fresh()
    uid = _new_agent_tab(ctx, session)
    result = _call(ctx, session, "clay_measure", kind="area", uid=uid, faces="123")
    _refused(result, "faces")
    once = _payload(_call(ctx, session, "clay_measure", kind="area", uid=uid, faces=[0]))
    twice = _payload(_call(ctx, session, "clay_measure", kind="area", uid=uid, faces=[0, 0]))
    assert twice["faces"] == 1 == once["faces"]
    assert twice["value"] == once["value"]


# --- clay-36 ------------------------------------------------------------------


def test_select_elements_refuses_a_selection_kind_that_is_not_the_documents_mode() -> None:
    ctx, session = _fresh()
    uid = _new_agent_tab(ctx, session)
    doc = _doc(ctx, session)

    # Object mode, no ``mode``: the selection would be a state the document
    # never holds (entering object mode clears it).
    assert doc.element_mode == "object"
    result = _call(ctx, session, "clay_select_elements", uid=uid, faces=[0, 1])
    _refused(result, "mode")
    assert doc.element_mode == "object"
    assert not doc.element_sel

    # ``mode='object'`` is no better.
    result = _call(ctx, session, "clay_select_elements", uid=uid, mode="object", faces=[0])
    _refused(result, "mode")
    assert doc.element_mode == "object"

    # In face mode, vertices are not the currency.
    _call(ctx, session, "clay_element_mode", mode="face")
    result = _call(ctx, session, "clay_select_elements", uid=uid, verts=[0, 1, 2])
    assert result["isError"] is True and result["structuredContent"]["field"] == "verts"
    assert "face" in result["content"][0]["text"]
    assert result["structuredContent"].get("changed") is False
    assert len(doc.element_sel_of(uid).verts) == 0

    # ...and neither is it when ``mode`` is given alongside.
    result = _call(ctx, session, "clay_select_elements", uid=uid, mode="vertex", faces=[0])
    assert result["isError"] is True and result["structuredContent"]["field"] == "faces"
    assert doc.element_mode == "face"

    # The matching kind still works, with or without ``mode``.
    ok_face = _call(ctx, session, "clay_select_elements", uid=uid, faces=[0, 1])
    assert ok_face["isError"] is False
    assert _payload(ok_face)["selected"]["faces"] == 2
    ok_vert = _call(ctx, session, "clay_select_elements", uid=uid, mode="vertex", verts=[0, 1])
    assert ok_vert["isError"] is False and _payload(ok_vert)["mode"] == "vertex"


# --- clay-73 ------------------------------------------------------------------


def test_clay_measure_refuses_a_point_that_makes_the_distance_overflow() -> None:
    ctx, session = _fresh()
    _new_agent_tab(ctx, session)
    result = _call(ctx, session, "clay_measure", kind="distance", a=[1e300, 0, 0], b=[-1e300, 0, 0])
    _refused(result, "a")
    assert _no_nan_text(result)
    # Within the translation ceiling a distance is always finite.
    ok = _call(ctx, session, "clay_measure", kind="distance", a=[1e7, 0, 0], b=[-1e7, 0, 0])
    assert ok["isError"] is False and _payload(ok)["value"] == 2e7


def test_clay_measure_angle_refuses_a_zero_length_ray() -> None:
    ctx, session = _fresh()
    _new_agent_tab(ctx, session)
    result = _call(
        ctx, session, "clay_measure", kind="angle", a=[1, 0, 0], b=[1, 0, 0], c=[0, 1, 0]
    )
    _refused(result, "a")
    result = _call(
        ctx, session, "clay_measure", kind="angle", a=[1, 0, 0], b=[0, 0, 0], c=[0, 0, 0]
    )
    _refused(result, "c")
    good = _call(ctx, session, "clay_measure", kind="angle", a=[1, 0, 0], b=[0, 0, 0], c=[0, 1, 0])
    assert good["isError"] is False and _payload(good)["value"] == 90.0


# --- clay-74 ------------------------------------------------------------------


def test_select_by_refuses_a_zero_direction_and_an_inverted_box() -> None:
    ctx, session = _fresh()
    uid = _new_agent_tab(ctx, session)
    _call(ctx, session, "clay_element_mode", mode="face")
    result = _call(ctx, session, "clay_select_by", uid=uid, query="normal", direction=[0, 0, 0])
    _refused(result, "direction")
    result = _call(
        ctx, session, "clay_select_by", uid=uid, query="bounds", min=[1, 1, 1], max=[-1, -1, -1]
    )
    _refused(result, "min")
    # One inverted axis is enough.
    result = _call(
        ctx, session, "clay_select_by", uid=uid, query="bounds", min=[-1, 2, -1], max=[1, 1, 1]
    )
    _refused(result, "min")
    # A flat box (min == max on an axis) is a legitimate slab.
    flat = _call(
        ctx, session, "clay_select_by", uid=uid, query="bounds", min=[-1, 0.5, -1], max=[1, 0.5, 1]
    )
    assert flat["isError"] is False


# --- clay-75 ------------------------------------------------------------------


def test_clay_render_description_claims_only_what_is_checked_before_rendering() -> None:
    render = next(t for t in agent_clay.tools() if t.name == "clay_render")
    description = " ".join(render.description.split())
    # Only the pixel budget is judged before the GPU is touched; the encoded
    # size can be known only after the pictures exist.
    for sentence in description.split(". "):
        if "reply frame" in sentence:
            assert "before any GPU work" not in sentence, sentence
            assert "after" in sentence, sentence
    assert "before any GPU work" in description
    assert "pixel render budget" in description


# --- clay-76 ------------------------------------------------------------------


def test_the_agent_modules_state_the_real_tool_count_and_scene_row_width() -> None:
    from pathlib import Path

    from realmspinner.familiar import contract

    count = len(agent_clay.tools())
    src = Path(agent_clay.__file__).read_text(encoding="utf-8")
    assert f"{count} ``Tool`` objects" in src
    assert "25 ``Tool`` objects" not in src

    ctx, session = _fresh()
    uid = _new_agent_tab(ctx, session)
    _call(ctx, session, "clay_uv", uid=uid, action="unwrap")
    child = _payload(_call(ctx, session, "clay_add_primitive", generator="box"))["uid"]
    _call(ctx, session, "clay_parent", uid=child, parent=uid)
    row = next(
        o for o in _payload(_call(ctx, session, "clay_scene"))["objects"] if o["uid"] == child
    )
    base = len([k for k in row if k not in ("uv", "local", "skin")])  # optional additions
    words = {18: "eighteen"}
    contract_src = Path(contract.__file__).read_text(encoding="utf-8")
    assert "carries fifteen fields" not in contract_src
    assert "carries seventeen fields" not in contract_src
    assert f"carries {words[base]} fields" in contract_src
    json.dumps(row)
    assert agent_tools is not None


# --- the document ceilings (clay-13) reach the agent doors as tool errors ------


def _full_document(ctx: _Ctx, session: agent_clay.Session, extra_loose_parts: bool = False):
    """Replace the session's document with one holding ``MAX_OBJECTS`` objects,
    built through the ``ClayDoc`` constructor (which does not refuse)."""
    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh.glbimport import MAX_OBJECTS

    tri = bm.from_faces([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    objects = [
        bd.Obj(uid=bd.new_uid(), name=f"o{i}", mesh=tri, generator=None, params={})
        for i in range(MAX_OBJECTS - 1)
    ]
    two_parts = bm.from_faces(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [5, 1, 0]],
        [[0, 1, 2], [3, 4, 5]],
    )
    last = bd.Obj(uid=bd.new_uid(), name="last", mesh=two_parts, generator=None, params={})
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    tab.doc = bd.ClayDoc([*objects, last])
    return tab.doc, last.uid


def test_every_agent_door_that_adds_an_object_refuses_a_full_document_as_a_tool_error() -> None:
    ctx, session = _fresh()
    _new_agent_tab(ctx, session)
    doc, last_uid = _full_document(ctx, session)
    count = len(doc.objects)
    history = doc.history.total

    doors = {
        "clay_add_primitive": {"generator": "box"},
        "clay_add_mesh": dict(_TETRA_MESH_ARGS),
        "clay_group": {"uids": [last_uid]},
        "clay_separate": {"uid": last_uid, "by": "loose_parts"},
    }
    for tool, arguments in doors.items():
        result = agent_clay.call(ctx, session, tool, arguments)
        assert result["isError"] is True, (tool, result["content"][0]["text"][:120])
        assert "4,096" in result["content"][0]["text"], tool
        assert result["structuredContent"].get("changed") is False, tool
        assert len(doc.objects) == count, tool
        assert doc.history.total == history, tool
        # No gesture left open: an open one defers the undo stack's eviction.
        assert doc.history._open_gestures == 0, tool
