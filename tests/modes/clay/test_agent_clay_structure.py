"""Regression and behaviour tests for Clay tranche 3's agent surface -- scene
structure (``dev/CLAY-PLAN.md``): parenting and groups, locking, tags,
separate, set origin, measure, and named checkpoints
(``studio/modes/clay/agent/tools_structure.py``), plus the world/local split
that lands on ``clay_scene`` and the locked-refusal wiring on
``clay_transform``/``clay_delete``.

Kept out of ``tests/modes/clay/test_agent_clay.py`` deliberately -- the same
rule ``test_agent_clay_modifiers.py`` states for itself: that file carries
the user's own uncommitted work, and this session's own new tests go in
their own file instead, with their own minimal ``ctx`` double rather than an
import across files.

**The bidirectional derivation gate does not apply here.** ``clay_separate``'s
``by``, ``clay_set_origin``'s ``mode`` and ``clay_measure``'s ``kind`` are
each a fixed three/four/four-member tuple (``schema.SEPARATE_MODES``/
``ORIGIN_MODES``/``MEASURE_KINDS``), not a live registry the way
``GENERATORS``/``OPS``/``QUERIES``/``MODIFIERS`` are -- there is no growing
source for a "thirteenth entry reaches the surface with no edit here" test
to prove anything about, the same reason ``RENDER_SHADINGS`` carries no such
gate either.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

# --- a ctx double, the same minimal shape test_agent_clay.py's own _Ctx is --


class _Cache:
    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


class _Ctx:
    """The same minimal ``ctx`` double ``tests/modes/clay/test_agent_clay.py``
    uses -- duplicated here rather than imported, the same reason
    ``test_agent_clay_modifiers.py`` gives for its own copy."""

    def __init__(self, svc: Any = None) -> None:
        self.state = SimpleNamespace(clay=None)
        self.svc = svc
        self.cache = _Cache()
        self.toasts: list[tuple[str, str]] = []
        self.settings = SimpleNamespace()

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _payload(result: dict) -> Any:
    return json.loads(result["content"][0]["text"])


def _row(result: dict) -> dict:
    return _payload(result)


def _new_world(ctx: _Ctx | None = None) -> tuple[_Ctx, agent_clay.Session, int, int]:
    """A fresh session with two boxes, apart in space. ->
    ``(ctx, session, uid1, uid2)``."""
    ctx = ctx or _Ctx()
    session = agent_clay.Session()
    r1 = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    assert r1["isError"] is False, r1
    uid1 = _payload(r1)["uid"]
    r2 = agent_clay.call(
        ctx, session, "clay_add_primitive", {"generator": "box", "translation": [3.0, 0.0, 0.0]}
    )
    assert r2["isError"] is False, r2
    uid2 = _payload(r2)["uid"]
    return ctx, session, uid1, uid2


def _doc(ctx: _Ctx, session: agent_clay.Session) -> Any:
    return clay_mode.ensure(ctx).get(session.tab_uid).doc


def _history_len(ctx: _Ctx, session: agent_clay.Session) -> int:
    return len(_doc(ctx, session).history.history())


# --- clay_parent -------------------------------------------------------------


def test_parent_writes_local_trs_and_is_one_undo_step() -> None:
    ctx, session, uid1, uid2 = _new_world()
    before = _history_len(ctx, session)

    result = agent_clay.call(ctx, session, "clay_parent", {"uid": uid1, "parent": uid2})
    assert result["isError"] is False, result
    row = _row(result)
    assert row["parent"] == uid2
    # keep_world (the default): uid1's world translation is unchanged even
    # though its local one is now relative to uid2.
    assert row["translation"] == [0.0, 0.0, 0.0]
    assert row["local"]["translation"] == [-3.0, 0.0, 0.0]
    assert _history_len(ctx, session) == before + 1

    doc = _doc(ctx, session)
    assert doc.undo()
    assert doc.by_uid(uid1).parent is None


def test_parent_null_clears_the_parent() -> None:
    ctx, session, uid1, uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_parent", {"uid": uid1, "parent": uid2})
    result = agent_clay.call(ctx, session, "clay_parent", {"uid": uid1, "parent": None})
    assert result["isError"] is False, result
    assert _row(result)["parent"] is None
    assert "local" not in _row(result)


def test_parent_missing_key_is_refused_distinctly_from_null() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_parent", {"uid": uid1})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "parent"
    assert result["structuredContent"]["changed"] is False


def test_parent_refuses_a_self_cycle() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    before = _history_len(ctx, session)
    result = agent_clay.call(ctx, session, "clay_parent", {"uid": uid1, "parent": uid1})
    assert result["isError"] is True
    assert "itself" in result["content"][0]["text"] or "descendant" in result["content"][0]["text"]
    assert _history_len(ctx, session) == before


def test_parent_refuses_a_descendant_cycle() -> None:
    ctx, session, uid1, uid2 = _new_world()
    made_child = agent_clay.call(ctx, session, "clay_parent", {"uid": uid2, "parent": uid1})
    assert made_child["isError"] is False, made_child
    before = _history_len(ctx, session)
    result = agent_clay.call(ctx, session, "clay_parent", {"uid": uid1, "parent": uid2})
    assert result["isError"] is True
    assert "descendant" in result["content"][0]["text"]
    assert _history_len(ctx, session) == before


def test_parent_refuses_an_unknown_parent_uid() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_parent", {"uid": uid1, "parent": 999999})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "parent"
    assert result["structuredContent"]["recovery"] == "read_scene"


def test_parent_is_batchable() -> None:
    assert "clay_parent" in set(agent_clay._HANDLERS) - agent_clay.BATCH_EXCLUDED


# --- clay_group / clay_ungroup ------------------------------------------------


def test_group_parents_every_member_onto_a_new_empty_as_one_step() -> None:
    ctx, session, uid1, uid2 = _new_world()
    before = _history_len(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_group", {"uids": [uid1, uid2], "name": "MyGroup"}
    )
    assert result["isError"] is False, result
    row = _row(result)
    assert row["name"] == "MyGroup"
    assert row["faces"] == 0  # an empty draws nothing
    assert _history_len(ctx, session) == before + 1

    doc = _doc(ctx, session)
    group_uid = row["uid"]
    assert doc.by_uid(uid1).parent == group_uid
    assert doc.by_uid(uid2).parent == group_uid

    assert doc.undo()
    assert doc.by_uid(uid1).parent is None
    assert doc.by_uid(uid2).parent is None
    assert group_uid not in {o.uid for o in doc.objects}


def test_group_disambiguates_a_colliding_name_rather_than_refusing() -> None:
    ctx, session, uid1, uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_rename", {"uid": uid1, "name": "Group"})
    result = agent_clay.call(ctx, session, "clay_group", {"uids": [uid2], "name": "Group"})
    assert result["isError"] is False, result
    assert _row(result)["name"] != "Group"


def test_group_refuses_an_empty_uids_list() -> None:
    ctx, session, _uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_group", {"uids": []})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "uids"


def test_ungroup_releases_children_with_world_placement_kept() -> None:
    ctx, session, uid1, uid2 = _new_world()
    grouped = agent_clay.call(ctx, session, "clay_group", {"uids": [uid1, uid2]})
    group_uid = _row(grouped)["uid"]
    before = _history_len(ctx, session)

    result = agent_clay.call(ctx, session, "clay_ungroup", {"uid": group_uid})
    assert result["isError"] is False, result
    assert sorted(_row(result)["released"]) == sorted([uid1, uid2])
    assert _history_len(ctx, session) == before + 1

    doc = _doc(ctx, session)
    assert doc.by_uid(uid1).parent is None
    assert doc.by_uid(uid2).parent is None
    # World placement survived the round trip through the group.
    assert doc.by_uid(uid2).translation.tolist() == pytest.approx([3.0, 0.0, 0.0])

    assert doc.undo()
    assert doc.by_uid(uid1).parent == group_uid
    assert doc.by_uid(uid2).parent == group_uid


def test_ungroup_refuses_an_object_with_no_children() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_ungroup", {"uid": uid1})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "uid"
    assert result["structuredContent"]["changed"] is False


def test_ungroup_refuses_an_object_that_carries_geometry() -> None:
    ctx, session, uid1, uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_parent", {"uid": uid2, "parent": uid1})
    result = agent_clay.call(ctx, session, "clay_ungroup", {"uid": uid1})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "uid"
    assert "geometry" in result["content"][0]["text"]


def test_group_and_ungroup_are_batchable() -> None:
    handlers = set(agent_clay._HANDLERS) - agent_clay.BATCH_EXCLUDED
    assert "clay_group" in handlers
    assert "clay_ungroup" in handlers


# --- clay_lock -----------------------------------------------------------------


def test_lock_and_unlock_toggle_as_one_step_each() -> None:
    ctx, session, uid1, uid2 = _new_world()
    before = _history_len(ctx, session)

    locked = agent_clay.call(ctx, session, "clay_lock", {"uids": [uid1, uid2], "locked": True})
    assert locked["isError"] is False, locked
    assert _row(locked)["changed"] == [uid1, uid2]
    assert _history_len(ctx, session) == before + 1

    doc = _doc(ctx, session)
    assert doc.by_uid(uid1).locked is True
    assert doc.by_uid(uid2).locked is True

    assert doc.undo()
    assert doc.by_uid(uid1).locked is False
    assert doc.by_uid(uid2).locked is False


def test_lock_already_locked_pushes_no_step() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_lock", {"uids": [uid1], "locked": True})
    before = _history_len(ctx, session)
    result = agent_clay.call(ctx, session, "clay_lock", {"uids": [uid1], "locked": True})
    assert result["isError"] is False, result
    assert _row(result)["changed"] == []
    assert _history_len(ctx, session) == before


def test_a_locked_object_refuses_transform_by_name() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_lock", {"uids": [uid1], "locked": True})
    before = _history_len(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_transform", {"uid": uid1, "translation": [1, 1, 1]}
    )
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["field"] == "uid"
    assert structured["changed"] is False
    assert "locked" in result["content"][0]["text"]
    assert "Box" in result["content"][0]["text"]  # names the object
    assert _history_len(ctx, session) == before


def test_a_locked_object_refuses_transform_through_a_locked_ancestor() -> None:
    ctx, session, uid1, uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_parent", {"uid": uid2, "parent": uid1})
    agent_clay.call(ctx, session, "clay_lock", {"uids": [uid1], "locked": True})

    result = agent_clay.call(
        ctx, session, "clay_transform", {"uid": uid2, "translation": [1, 1, 1]}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "uid"


def test_a_locked_object_refuses_delete_by_name() -> None:
    ctx, session, uid1, uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_lock", {"uids": [uid1], "locked": True})
    before = _history_len(ctx, session)

    result = agent_clay.call(ctx, session, "clay_delete", {"uids": [uid1, uid2]})
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["field"] == "uids"
    assert structured["changed"] is False
    assert structured["uids"] == [uid1]
    assert "Box" in result["content"][0]["text"]
    assert _history_len(ctx, session) == before
    # Nothing was removed, including uid2 -- validated before any mutation.
    doc = _doc(ctx, session)
    assert {o.uid for o in doc.objects} == {uid1, uid2}


def test_locking_still_allows_rename_visibility_and_tags() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_lock", {"uids": [uid1], "locked": True})

    renamed = agent_clay.call(ctx, session, "clay_rename", {"uid": uid1, "name": "StillLocked"})
    assert renamed["isError"] is False, renamed
    tagged = agent_clay.call(ctx, session, "clay_tag", {"uids": [uid1], "add": ["prop"]})
    assert tagged["isError"] is False, tagged
    unlocked = agent_clay.call(ctx, session, "clay_lock", {"uids": [uid1], "locked": False})
    assert unlocked["isError"] is False, unlocked


def test_lock_is_batchable() -> None:
    assert "clay_lock" in set(agent_clay._HANDLERS) - agent_clay.BATCH_EXCLUDED


# --- clay_tag --------------------------------------------------------------------


def test_tag_normalizes_dedupes_sorts_and_lower_cases() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    before = _history_len(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_tag", {"uids": [uid1], "add": ["Prop", "prop", "Furniture"]}
    )
    assert result["isError"] is False, result
    row = _row(result)["objects"][0]
    assert row["uid"] == uid1
    assert row["tags"] == ["furniture", "prop"]  # sorted, deduped, lower-cased
    assert _history_len(ctx, session) == before + 1

    doc = _doc(ctx, session)
    assert doc.undo()
    assert doc.by_uid(uid1).tags == ()


def test_tag_remove_drops_a_tag_spelled_differently() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_tag", {"uids": [uid1], "add": ["Prop"]})
    result = agent_clay.call(ctx, session, "clay_tag", {"uids": [uid1], "remove": ["PROP"]})
    assert result["isError"] is False, result
    assert _row(result)["objects"][0]["tags"] == []


def test_tag_applies_the_same_add_remove_to_every_named_object_as_one_step() -> None:
    ctx, session, uid1, uid2 = _new_world()
    before = _history_len(ctx, session)
    result = agent_clay.call(
        ctx, session, "clay_tag", {"uids": [uid1, uid2], "add": ["batch_tag"]}
    )
    assert result["isError"] is False, result
    rows = {r["uid"]: r["tags"] for r in _row(result)["objects"]}
    assert rows[uid1] == ["batch_tag"]
    assert rows[uid2] == ["batch_tag"]
    assert _history_len(ctx, session) == before + 1


def test_tag_refuses_giving_neither_add_nor_remove() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_tag", {"uids": [uid1]})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "add"
    assert result["structuredContent"]["changed"] is False


def test_tag_is_not_a_locking_door() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_lock", {"uids": [uid1], "locked": True})
    result = agent_clay.call(ctx, session, "clay_tag", {"uids": [uid1], "add": ["prop"]})
    assert result["isError"] is False, result


def test_select_by_tag_unions_with_uids() -> None:
    ctx, session, uid1, uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_tag", {"uids": [uid1], "add": ["prop"]})
    result = agent_clay.call(ctx, session, "clay_select", {"uids": [uid2], "tag": "Prop"})
    assert result["isError"] is False, result
    assert sorted(_payload(result)["selection"]) == sorted([uid1, uid2])


def test_select_by_tag_alone_with_empty_uids() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_tag", {"uids": [uid1], "add": ["prop"]})
    result = agent_clay.call(ctx, session, "clay_select", {"uids": [], "tag": "prop"})
    assert result["isError"] is False, result
    assert _payload(result)["selection"] == [uid1]


def test_tag_is_batchable() -> None:
    assert "clay_tag" in set(agent_clay._HANDLERS) - agent_clay.BATCH_EXCLUDED


# --- clay_separate ---------------------------------------------------------------


_TWO_TETRA_ARGS = {
    "positions": [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, 0.0, 1.0],
    ],
    "faces": [
        [0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3],
        [4, 6, 5], [4, 5, 7], [5, 6, 7], [6, 4, 7],
    ],
}


def test_separate_by_loose_parts_as_one_step() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    added = agent_clay.call(ctx, session, "clay_add_mesh", _TWO_TETRA_ARGS)
    uid = _row(added)["uid"]
    before = _history_len(ctx, session)

    result = agent_clay.call(ctx, session, "clay_separate", {"uid": uid, "by": "loose_parts"})
    assert result["isError"] is False, result
    row = _row(result)
    assert len(row["uids"]) == 2
    assert _history_len(ctx, session) == before + 1

    doc = _doc(ctx, session)
    assert uid not in {o.uid for o in doc.objects}
    assert doc.undo()
    assert {o.uid for o in doc.objects} == {uid}


def test_separate_by_material() -> None:
    """Half the box's faces get a second material -- the tool surface has no
    door for painting part of one object's own faces (clay_material always
    repaints the whole object), so this reaches into the document directly
    for test setup only; the assertion below is entirely about
    clay_separate's own behaviour."""
    from dataclasses import replace as _replace

    ctx = _Ctx()
    session = agent_clay.Session()
    added = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    uid = _row(added)["uid"]
    agent_clay.call(
        ctx, session, "clay_material", {"name": "red", "uids": [uid], "color": [1, 0, 0]}
    )
    doc = _doc(ctx, session)
    obj = doc.by_uid(uid)
    material = obj.mesh.material.copy()
    material[: len(material) // 2] = 0  # half the faces keep slot 0
    doc.set_mesh(uid, _replace(obj.mesh, material=material), keep_generator=True)

    result = agent_clay.call(ctx, session, "clay_separate", {"uid": uid, "by": "material"})
    assert result["isError"] is False, result
    assert len(_row(result)["uids"]) == 2


def test_separate_by_selection() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    mode_result = agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    assert mode_result["isError"] is False, mode_result
    sel_result = agent_clay.call(
        ctx, session, "clay_select_elements", {"uid": uid1, "faces": [0]}
    )
    assert sel_result["isError"] is False, sel_result

    result = agent_clay.call(ctx, session, "clay_separate", {"uid": uid1, "by": "selection"})
    assert result["isError"] is False, result
    assert len(_row(result)["uids"]) == 2


def test_separate_by_selection_refuses_with_nothing_selected() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_separate", {"uid": uid1, "by": "selection"})
    assert result["isError"] is True
    assert result["structuredContent"]["changed"] is False


def test_separate_refuses_when_the_split_would_be_a_single_piece() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_separate", {"uid": uid1, "by": "loose_parts"})
    assert result["isError"] is True
    assert result["structuredContent"]["changed"] is False


def test_separate_refuses_a_locked_source() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    added = agent_clay.call(ctx, session, "clay_add_mesh", _TWO_TETRA_ARGS)
    uid = _row(added)["uid"]
    agent_clay.call(ctx, session, "clay_lock", {"uids": [uid], "locked": True})
    result = agent_clay.call(ctx, session, "clay_separate", {"uid": uid, "by": "loose_parts"})
    assert result["isError"] is True
    assert "locked" in result["content"][0]["text"]


def test_separate_keeps_the_sources_parent_and_transform_on_every_piece() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    added = agent_clay.call(ctx, session, "clay_add_mesh", _TWO_TETRA_ARGS)
    uid = _row(added)["uid"]
    parent_added = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    parent_uid = _row(parent_added)["uid"]
    agent_clay.call(ctx, session, "clay_parent", {"uid": uid, "parent": parent_uid})

    result = agent_clay.call(ctx, session, "clay_separate", {"uid": uid, "by": "loose_parts"})
    doc = _doc(ctx, session)
    for piece_uid in _row(result)["uids"]:
        assert doc.by_uid(piece_uid).parent == parent_uid


def test_separate_is_batchable() -> None:
    assert "clay_separate" in set(agent_clay._HANDLERS) - agent_clay.BATCH_EXCLUDED


# --- clay_set_origin ---------------------------------------------------------------


def test_set_origin_bounds_is_a_no_op_on_an_already_centred_box() -> None:
    """A default box's own bounds centre already *is* its origin, so this is
    the one mode with nothing to move for it -- ``set_origin`` pushes no
    step for a no-op, the same rule ``set_transform`` follows."""
    ctx, session, uid1, _uid2 = _new_world()
    before = _history_len(ctx, session)

    result = agent_clay.call(ctx, session, "clay_set_origin", {"uid": uid1, "mode": "bounds"})
    assert result["isError"] is False, result
    row = _row(result)
    assert row["changed"] is False
    assert row["bbox"]["min"] == pytest.approx([-0.5, -0.5, -0.5])
    assert _history_len(ctx, session) == before


def test_set_origin_base_uses_the_lowest_y() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_set_origin", {"uid": uid1, "mode": "base"})
    assert result["isError"] is False, result
    assert _row(result)["changed"] is True
    assert _row(result)["point"] == pytest.approx([0.0, -0.5, 0.0])
    # Nothing moved on screen: the box's own world bounds are unchanged.
    assert _row(result)["bbox"]["min"] == pytest.approx([-0.5, -0.5, -0.5])
    # The origin itself did move: the object's own translation now names the
    # new pivot's world position.
    assert _row(result)["translation"] == pytest.approx([0.0, -0.5, 0.0])


def test_set_origin_world_moves_the_pivot_to_the_world_origin() -> None:
    """The mesh and everything on screen stays exactly where it was -- the
    world-space bbox is unchanged -- while the object's own translation
    (naming where the pivot now sits) becomes the target point, and its
    local mesh absorbs the opposite shift to compensate."""
    ctx, session, uid1, uid2 = _new_world()  # uid2 sits at [3, 0, 0]
    scene_before = agent_clay.call(ctx, session, "clay_scene", {})
    row_before = next(o for o in _payload(scene_before)["objects"] if o["uid"] == uid2)

    result = agent_clay.call(ctx, session, "clay_set_origin", {"uid": uid2, "mode": "world"})
    assert result["isError"] is False, result
    row = _row(result)
    assert row["point"] == pytest.approx([0.0, 0.0, 0.0])
    assert row["translation"] == pytest.approx([0.0, 0.0, 0.0])  # the pivot's new position
    assert row["bbox"]["min"] == pytest.approx(row_before["bbox"]["min"])  # unmoved on screen
    doc = _doc(ctx, session)
    assert doc.by_uid(uid2).translation.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_set_origin_selection_uses_the_mean_selected_position() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "vertex"})
    agent_clay.call(ctx, session, "clay_select_elements", {"uid": uid1, "verts": [0]})
    result = agent_clay.call(ctx, session, "clay_set_origin", {"uid": uid1, "mode": "selection"})
    assert result["isError"] is False, result
    # The moved-to point is one of the box's own corners.
    point = _row(result)["point"]
    assert all(abs(abs(c) - 0.5) < 1e-6 for c in point)


def test_set_origin_explicit_point() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_set_origin", {"uid": uid1, "point": [1, 2, 3]})
    assert result["isError"] is False, result
    assert _row(result)["translation"] == pytest.approx([1.0, 2.0, 3.0])


def test_set_origin_refuses_giving_both_mode_and_point() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(
        ctx, session, "clay_set_origin", {"uid": uid1, "mode": "world", "point": [0, 0, 0]}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "mode"


def test_set_origin_refuses_giving_neither() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_set_origin", {"uid": uid1})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "mode"


def test_set_origin_freezes_the_generator() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_set_origin", {"uid": uid1, "mode": "base"})
    assert result["isError"] is False, result
    assert _row(result)["changed"] is True
    assert _row(result)["generator"] is None


def test_set_origin_is_not_a_locking_door() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_lock", {"uids": [uid1], "locked": True})
    result = agent_clay.call(ctx, session, "clay_set_origin", {"uid": uid1, "mode": "world"})
    assert result["isError"] is False, result


def test_set_origin_is_batchable() -> None:
    assert "clay_set_origin" in set(agent_clay._HANDLERS) - agent_clay.BATCH_EXCLUDED


# --- clay_measure ------------------------------------------------------------------
#
# A default box is a unit cube centred on the origin -- volume 1, one quad
# face area 1, an edge 1 metre, a right-angle corner 90 degrees -- so every
# number below is a known, hand-checkable ground truth, not merely "some
# plausible float".


def test_measure_volume_of_a_unit_cube() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_measure", {"kind": "volume", "uid": uid1})
    assert result["isError"] is False, result
    assert _row(result)["value"] == pytest.approx(1.0)


def test_measure_area_of_one_face() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(
        ctx, session, "clay_measure", {"kind": "area", "uid": uid1, "faces": [0]}
    )
    assert result["isError"] is False, result
    assert _row(result)["value"] == pytest.approx(1.0)


def test_measure_area_defaults_to_the_current_face_selection() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    agent_clay.call(ctx, session, "clay_select_elements", {"uid": uid1, "faces": [0, 1]})
    result = agent_clay.call(ctx, session, "clay_measure", {"kind": "area", "uid": uid1})
    assert result["isError"] is False, result
    assert _row(result)["value"] == pytest.approx(2.0)
    assert _row(result)["faces"] == 2


def test_measure_area_refuses_with_nothing_selected_and_no_faces_given() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_measure", {"kind": "area", "uid": uid1})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "faces"


def test_measure_distance_between_explicit_points() -> None:
    ctx, session, _uid1, _uid2 = _new_world()
    result = agent_clay.call(
        ctx, session, "clay_measure",
        {"kind": "distance", "a": [-0.5, -0.5, -0.5], "b": [0.5, -0.5, -0.5]},
    )
    assert result["isError"] is False, result
    assert _row(result)["value"] == pytest.approx(1.0)


def test_measure_angle_at_a_right_angle_corner() -> None:
    ctx, session, _uid1, _uid2 = _new_world()
    result = agent_clay.call(
        ctx, session, "clay_measure",
        {
            "kind": "angle",
            "a": [0.5, -0.5, -0.5],
            "b": [-0.5, -0.5, -0.5],
            "c": [-0.5, 0.5, -0.5],
        },
    )
    assert result["isError"] is False, result
    assert _row(result)["value"] == pytest.approx(90.0)


def test_measure_distance_via_uid_alone_reads_world_translation() -> None:
    ctx, session, uid1, uid2 = _new_world()  # uid1 at origin, uid2 at [3,0,0]
    result = agent_clay.call(
        ctx, session, "clay_measure", {"kind": "distance", "a": {"uid": uid1}, "b": {"uid": uid2}}
    )
    assert result["isError"] is False, result
    assert _row(result)["value"] == pytest.approx(3.0)


def test_measure_distance_via_uid_and_vertex() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(
        ctx, session, "clay_measure",
        {"kind": "distance", "a": {"uid": uid1, "vertex": 0}, "b": {"uid": uid1, "vertex": 0}},
    )
    assert result["isError"] is False, result
    assert _row(result)["value"] == pytest.approx(0.0)


def test_measure_refuses_an_out_of_range_vertex() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(
        ctx, session, "clay_measure",
        {"kind": "distance", "a": {"uid": uid1, "vertex": 99999}, "b": [0, 0, 0]},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "a"


def test_measure_never_selects_or_pushes_a_step() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    before = _history_len(ctx, session)
    before_sel = set(_doc(ctx, session).selection)
    agent_clay.call(ctx, session, "clay_measure", {"kind": "volume", "uid": uid1})
    assert _history_len(ctx, session) == before
    assert set(_doc(ctx, session).selection) == before_sel


def test_measure_is_batchable() -> None:
    assert "clay_measure" in set(agent_clay._HANDLERS) - agent_clay.BATCH_EXCLUDED


# --- clay_checkpoint / clay_restore ------------------------------------------------


def test_checkpoint_pushes_no_undo_step() -> None:
    ctx, session, _uid1, _uid2 = _new_world()
    before = _history_len(ctx, session)
    result = agent_clay.call(ctx, session, "clay_checkpoint", {"name": "cp"})
    assert result["isError"] is False, result
    assert _history_len(ctx, session) == before


def test_checkpoint_survives_undo_and_redo() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    first = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    assert first["isError"] is False, first
    agent_clay.call(ctx, session, "clay_checkpoint", {"name": "cp"})
    agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})

    doc = _doc(ctx, session)
    assert doc.undo()  # back to just after "cp"
    assert doc.redo()  # forward past it again

    result = agent_clay.call(ctx, session, "clay_restore", {"name": "cp"})
    assert result["isError"] is False, result
    assert _row(result)["moved"] is True
    assert len(doc.objects) == 1


def test_restore_reports_gone_after_a_divergent_push() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    agent_clay.call(ctx, session, "clay_checkpoint", {"name": "cp"})  # at 2 objects

    doc = _doc(ctx, session)
    assert doc.undo()  # back to 1 object, "cp"'s own serial now only reachable by redo
    # A divergent push discards the redo branch "cp" pointed into.
    agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})

    assert doc.checkpoint_status("cp") == "gone"
    result = agent_clay.call(ctx, session, "clay_restore", {"name": "cp"})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "name"
    assert "gone" in result["content"][0]["text"] or "not" in result["content"][0]["text"] \
        or "reachable" in result["content"][0]["text"]


def test_restore_unknown_name_is_refused_naming_read_scene() -> None:
    ctx, session, _uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_restore", {"name": "never-set"})
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["field"] == "name"
    assert structured["recovery"] == "read_scene"


def test_restore_already_current_is_not_a_refusal() -> None:
    ctx, session, _uid1, _uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_checkpoint", {"name": "cp"})
    result = agent_clay.call(ctx, session, "clay_restore", {"name": "cp"})
    assert result["isError"] is False, result
    assert _row(result)["moved"] is False
    assert _row(result)["status"] == "current"


def test_checkpoint_is_batchable_but_restore_is_not() -> None:
    handlers = set(agent_clay._HANDLERS) - agent_clay.BATCH_EXCLUDED
    assert "clay_checkpoint" in handlers
    assert "clay_restore" not in handlers
    assert "clay_restore" in agent_clay.BATCH_EXCLUDED


# --- clay_scene: world TRS plus local for a parented object -----------------------


def test_scene_reports_world_trs_for_a_root_unchanged() -> None:
    """A root's world translation/rotation/scale is bit-identical to what it
    always reported -- the whole point of tranche 3's "a root's local TRS is
    its world TRS" invariant."""
    ctx, session, uid1, _uid2 = _new_world()
    agent_clay.call(
        ctx, session, "clay_transform",
        {"uid": uid1, "translation": [1.0, 2.0, 3.0], "rotation": [10.0, 20.0, 30.0]},
    )
    scene = agent_clay.call(ctx, session, "clay_scene", {})
    row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid1)
    assert row["translation"] == pytest.approx([1.0, 2.0, 3.0])
    assert row["rotation"] == pytest.approx([10.0, 20.0, 30.0])
    assert "local" not in row
    assert row["parent"] is None
    assert row["locked"] is False
    assert row["tags"] == []


def test_scene_reports_world_trs_and_a_local_block_for_a_parented_object() -> None:
    ctx, session, uid1, uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_transform", {"uid": uid2, "translation": [5.0, 0.0, 0.0]})
    agent_clay.call(ctx, session, "clay_parent", {"uid": uid1, "parent": uid2})
    agent_clay.call(ctx, session, "clay_transform", {"uid": uid1, "translation": [1.0, 0.0, 0.0]})

    scene = agent_clay.call(ctx, session, "clay_scene", {})
    row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid1)
    # World: parent's [5,0,0] plus this object's own local [1,0,0].
    assert row["translation"] == pytest.approx([6.0, 0.0, 0.0])
    assert row["local"]["translation"] == pytest.approx([1.0, 0.0, 0.0])
    assert row["parent"] == uid2


def test_scene_reports_tags_and_locked() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_tag", {"uids": [uid1], "add": ["prop", "furniture"]})
    agent_clay.call(ctx, session, "clay_lock", {"uids": [uid1], "locked": True})

    scene = agent_clay.call(ctx, session, "clay_scene", {})
    row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid1)
    assert row["tags"] == ["furniture", "prop"]
    assert row["locked"] is True


def test_boolean_consumes_the_world_placement_of_a_parented_operand() -> None:
    """world= threaded into ops_boolean.boolean: a parented operand's own
    local TRS alone would put it in the wrong place -- this proves the
    boolean actually reads its world matrix, not merely its own fields."""
    ctx, session, uid1, uid2 = _new_world()
    # Move uid2's own parent far away and re-parent uid2 under it, keeping
    # world placement -- uid2's *local* TRS is now nowhere near [3, 0, 0],
    # but its world placement (what the boolean must actually consume) still
    # is, since keep_world=True is the default.
    far = agent_clay.call(
        ctx, session, "clay_add_primitive", {"generator": "box", "translation": [100.0, 0.0, 0.0]}
    )
    far_uid = _row(far)["uid"]
    agent_clay.call(ctx, session, "clay_parent", {"uid": uid2, "parent": far_uid})

    result = agent_clay.call(ctx, session, "clay_boolean", {"kind": "union", "uids": [uid1, uid2]})
    assert result["isError"] is False, result
    survivor = _payload(result)["uid"]
    doc = _doc(ctx, session)
    lo, hi = agent_clay.clay_geom_ops.world_box(doc.by_uid(survivor), doc.evaluated(survivor))
    # Two boxes centred at 0 and at [3,0,0] union to a box spanning roughly
    # -0.5..3.5 on X -- not the ~100 a un-worlded read of uid2 would give.
    assert hi[0] < 10.0
