"""``clay_collider`` (``dev/CLAY-PLAN.md`` tranche 7's integration half,
``studio/modes/clay/agent/tools_collider.py``): fits a box/sphere/capsule/
convex/compound proxy against one or more objects' own evaluated meshes,
one undo step per call, and is deliberately *not* a locking door.

Shares ``_Ctx``/``_payload``/``_new_agent_tab``/``_history_len`` with
``test_agent_clay.py`` -- see that module's own fixtures.
"""

from __future__ import annotations

import pytest

from realmspinner.kernels.mesh import colliders
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

from .test_agent_clay import _Ctx, _history_len, _new_agent_tab, _payload


def _doc(ctx, session):
    return clay_mode.ensure(ctx).get(session.tab_uid).doc


def _head(ctx, session) -> int:
    return _doc(ctx, session).history.head


# --- the derived enum, both ways ----------------------------------------------


def test_the_collider_kind_enum_matches_collider_kinds_both_ways() -> None:
    tool = next(t for t in agent_clay.tools() if t.name == "clay_collider")
    assert set(tool.schema["properties"]["kind"]["enum"]) == set(colliders.COLLIDER_KINDS)


def test_a_sixth_collider_kind_reaches_the_agent_surface_with_no_edit_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live half of the same gate: a kind added to ``COLLIDER_KINDS`` at
    runtime shows up in ``clay_collider``'s own schema enum with nothing here
    touched to make it happen -- the same "monkeypatch a registry, watch the
    tool list move" proof ``test_agent_clay.py`` already gives
    ``GENERATORS``/``ASSEMBLIES``/``OPS``."""
    monkeypatch.setitem(
        colliders.COLLIDER_KINDS, "sixth_kind", ("Sixth", colliders.fit_sphere, {})
    )
    tool = next(t for t in agent_clay.tools() if t.name == "clay_collider")
    assert "sixth_kind" in tool.schema["properties"]["kind"]["enum"]


# --- one call, one undo step, and undoable ------------------------------------


@pytest.mark.parametrize("kind", sorted(colliders.COLLIDER_KINDS))
def test_every_kind_is_one_undo_step_and_undoable(kind: str) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    before_len = _history_len(ctx, session)
    before_head = _head(ctx, session)
    before_count = len(_doc(ctx, session).objects)

    result = agent_clay.call(ctx, session, "clay_collider", {"uids": [uid], "kind": kind})
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["kind"] == kind
    assert len(payload["uids"]) == 1
    assert _history_len(ctx, session) == before_len + 1

    doc = _doc(ctx, session)
    new_obj = doc.by_uid(payload["uids"][0])
    assert new_obj.role == "collider"
    assert new_obj.collider_kind == kind
    assert new_obj.parent == uid
    assert len(doc.objects) == before_count + 1

    undo = agent_clay.call(ctx, session, "clay_undo", {})
    assert undo["isError"] is False, undo
    assert _head(ctx, session) == before_head
    assert len(_doc(ctx, session).objects) == before_count


def test_several_uids_fold_into_one_undo_step() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid1 = _new_agent_tab(ctx, session)
    second = agent_clay.call(
        ctx, session, "clay_add_primitive", {"generator": "box", "translation": [3.0, 0.0, 0.0]}
    )
    assert second["isError"] is False, second
    uid2 = _payload(second)["uid"]
    before_len = _history_len(ctx, session)
    before_head = _head(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_collider", {"uids": [uid1, uid2], "kind": "sphere"}
    )
    assert result["isError"] is False, result
    assert len(_payload(result)["uids"]) == 2
    assert _history_len(ctx, session) == before_len + 1

    undo = agent_clay.call(ctx, session, "clay_undo", {})
    assert undo["isError"] is False, undo
    assert _head(ctx, session) == before_head
    assert not [o for o in _doc(ctx, session).objects if o.role == "collider"]


def test_box_kind_takes_the_oriented_param() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    result = agent_clay.call(
        ctx,
        session,
        "clay_collider",
        {"uids": [uid], "kind": "box", "params": {"oriented": 1.0}},
    )
    assert result["isError"] is False, result


# --- refusals -------------------------------------------------------------


def test_an_unknown_kind_is_refused_by_name() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    result = agent_clay.call(ctx, session, "clay_collider", {"uids": [uid], "kind": "bogus"})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "kind"
    assert result["structuredContent"]["changed"] is False


def test_an_unknown_param_is_refused_by_name() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    result = agent_clay.call(
        ctx,
        session,
        "clay_collider",
        {"uids": [uid], "kind": "sphere", "params": {"oriented": 1.0}},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "params"


_FLAT_TRIANGLE_ARGS = {
    "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    "faces": [[0, 1, 2]],
}
"""Three coplanar points -- a valid ``clay_add_mesh`` call (one real face),
but not four non-coplanar points, so :func:`colliders.convex_hull` refuses
it (``OpError``: "Every point is coplanar...")."""


def test_convex_hull_refuses_too_few_points() -> None:
    """``colliders.convex_hull`` itself refuses (``OpError``) an object with
    no real hull to build, which this handler surfaces naming ``uids``
    rather than leaving the generic backstop to log it."""
    ctx = _Ctx()
    session = agent_clay.Session()
    added = agent_clay.call(ctx, session, "clay_add_mesh", _FLAT_TRIANGLE_ARGS)
    assert added["isError"] is False, added
    uid = _payload(added)["uid"]
    result = agent_clay.call(ctx, session, "clay_collider", {"uids": [uid], "kind": "convex"})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "uids"
    assert result["structuredContent"]["changed"] is False


def test_a_refusal_partway_through_a_multi_uid_call_leaves_nothing_added() -> None:
    """Every fit runs before any object is added (see ``tools_collider.py``'s
    own module docstring) -- a bad uid among several good ones must not
    leave the good ones' colliders standing while the call as a whole
    refuses."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid1 = _new_agent_tab(ctx, session)
    added = agent_clay.call(ctx, session, "clay_add_mesh", _FLAT_TRIANGLE_ARGS)
    assert added["isError"] is False, added
    bad_uid = _payload(added)["uid"]
    before_count = len(_doc(ctx, session).objects)

    result = agent_clay.call(
        ctx, session, "clay_collider", {"uids": [uid1, bad_uid], "kind": "convex"}
    )
    assert result["isError"] is True
    assert len(_doc(ctx, session).objects) == before_count


# --- not a locking door ------------------------------------------------------


def test_a_locked_object_can_still_grow_a_collider() -> None:
    """``ClayDoc.add_collider`` says plainly that adding a collider changes
    nothing about the *source* object itself, so a locked source can still
    grow one -- unlike every mesh-mutating ``clay_uv`` action."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    lock = agent_clay.call(ctx, session, "clay_lock", {"uids": [uid], "locked": True})
    assert lock["isError"] is False, lock

    result = agent_clay.call(ctx, session, "clay_collider", {"uids": [uid], "kind": "box"})
    assert result["isError"] is False, result


# --- clay_scene's own role/collider_kind ---------------------------------------


def test_clay_scene_names_the_collider_role_and_kind() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    added = agent_clay.call(ctx, session, "clay_collider", {"uids": [uid], "kind": "sphere"})
    assert added["isError"] is False, added
    collider_uid = _payload(added)["uids"][0]

    scene = agent_clay.call(ctx, session, "clay_scene", {})
    assert scene["isError"] is False, scene
    row = next(o for o in _payload(scene)["objects"] if o["uid"] == collider_uid)
    assert row["role"] == "collider"
    assert row["collider_kind"] == "sphere"
    assert row["parent"] == uid


# --- clay_catalog: the collider_kinds topic ------------------------------------


def test_clay_catalog_collider_kinds_names_every_kind() -> None:
    from realmspinner.studio.modes.clay.agent import schema as agent_clay_schema

    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(ctx, session, "clay_catalog", {"topic": "collider_kinds"})
    assert result["isError"] is False, result
    catalogue = _payload(result)["catalogue"]
    assert catalogue == agent_clay_schema._collider_kind_catalog()
    for name in colliders.COLLIDER_KINDS:
        assert name in catalogue


def test_clay_catalog_refuses_an_unknown_topic() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(ctx, session, "clay_catalog", {"topic": "bogus"})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "topic"


# --- clay_export's engine argument ---------------------------------------------


def test_clay_export_validates_and_echoes_engine(svc) -> None:
    ctx = _Ctx(svc=svc)
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    result = agent_clay.call(ctx, session, "clay_export", {"engine": "godot4"})
    assert result["isError"] is False, result
    assert _payload(result)["engine"] == "godot4"


def test_clay_export_refuses_an_unknown_engine(svc) -> None:
    ctx = _Ctx(svc=svc)
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    result = agent_clay.call(ctx, session, "clay_export", {"engine": "bogus"})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "engine"
