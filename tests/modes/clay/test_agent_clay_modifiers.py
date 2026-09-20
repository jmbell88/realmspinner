"""Regression and derivation tests for Clay tranche 2's agent surface -- the
modifier stack (``kernels.mesh.modifiers``, ``ClayDoc.set_modifiers``/
``apply_modifiers``/``evaluated``/``evaluation``) as five new tools:
``clay_modifier_add``, ``clay_modifier_set``, ``clay_modifier_remove``,
``clay_modifier_move`` and ``clay_modifier_apply``
(``studio/modes/clay/agent/tools_modifiers.py``), plus the base/evaluated
split that lands on every tool that already existed (``clay_scene``,
``clay_boolean``, ``clay_diagnose``, ``clay_analyze``, ``clay_program``'s own
facts).

Kept out of ``tests/modes/clay/test_agent_clay.py`` deliberately -- the same
rule ``test_agent_clay_validate.py`` states for itself: that file carries the
user's own uncommitted work, and this session's own new tests go in their own
file instead, with their own minimal ``ctx`` double rather than an import
across files. The two bidirectional registry gates for the ``kind`` enum
(``test_every_modifier_kind_is_a_clay_modifier_add_enum_option_and_vice_versa``,
``test_an_eleventh_modifier_kind_reaches_the_agent_surface_with_no_edit_here``)
stay in ``test_agent_clay.py`` itself, beside the other three registries'
identical pair -- that file's own module docstring's instruction.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from warlock.kernels.mesh import modifiers as clay_modifiers
from warlock.kernels.mesh import scratch as clay_scratch
from warlock.studio.modes.clay import mode as clay_mode
from warlock.studio.modes.clay.agent import dispatch as agent_clay

# --- a ctx double, the same minimal shape test_agent_clay.py's own _Ctx is --


class _Cache:
    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


class _Ctx:
    """The same minimal ``ctx`` double ``tests/modes/clay/test_agent_clay.py``
    uses -- duplicated here rather than imported, the same reason
    ``test_agent_clay_validate.py`` gives for its own copy: that file carries
    the user's own uncommitted work and this one must not import from it.
    """

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


def _new_world(ctx: _Ctx | None = None) -> tuple[_Ctx, agent_clay.Session, int, int]:
    """A fresh session with two boxes, apart in space -- one closed solid
    each, so a boolean modifier has something real to consume. ->
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


def _row(result: dict) -> dict:
    return _payload(result)


# --- the bidirectional derivation gate lives in test_agent_clay.py; see the
# module docstring above for why. The rest of this file is behaviour.


# --- clay_modifier_add ---------------------------------------------------


def test_add_appends_by_default_and_is_one_undo_step() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    before = _history_len(ctx, session)
    result = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "mirror"})
    assert result["isError"] is False, result
    row = _row(result)
    assert row["modifiers"] == [
        {
            "id": row["modifier"],
            "kind": "mirror",
            "enabled": True,
            "params": {"axis": 0, "weld": 0.0001},
        }
    ]
    assert _history_len(ctx, session) == before + 1

    doc = _doc(ctx, session)
    assert doc.undo()
    assert doc.by_uid(uid1).modifiers == ()


def test_add_with_index_inserts_rather_than_appends() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    first = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "mirror"})
    first_id = _row(first)["modifier"]
    second = agent_clay.call(
        ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "weld", "index": 0}
    )
    assert second["isError"] is False, second
    row = _row(second)
    assert [m["id"] for m in row["modifiers"]] == [row["modifier"], first_id]


def test_add_clamps_an_out_of_range_param_rather_than_refusing() -> None:
    """The same trade ``clay_op``'s own ``run`` makes for its declared
    parameters -- an array count of 99999 is silently pulled back to the
    kind's own ceiling (200), not a refusal an agent has to write around."""
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(
        ctx, session, "clay_modifier_add",
        {"uid": uid1, "kind": "array", "params": {"count": 99999}},
    )
    assert result["isError"] is False, result
    assert _row(result)["modifiers"][0]["params"]["count"] == 200


def test_add_refuses_an_unknown_kind_naming_the_valid_ones() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    before = _history_len(ctx, session)
    result = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "nope"})
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["field"] == "kind"
    assert structured["changed"] is False
    message = result["content"][0]["text"]
    for name in sorted(clay_modifiers.MODIFIERS):
        assert name in message
    assert _history_len(ctx, session) == before


def test_add_refuses_an_unknown_param_naming_the_valid_ones() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(
        ctx, session, "clay_modifier_add",
        {"uid": uid1, "kind": "mirror", "params": {"bogus": 1.0}},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "params"
    message = result["content"][0]["text"]
    assert "bogus" in message
    assert "mirror" in message


def test_add_refuses_a_non_numeric_param_value_by_name() -> None:
    """The exact crash class :func:`~.validate._op_params_type_refusal`'s
    own docstring names for ``clay_op`` -- a non-numeric string reaching a
    bare ``float()`` call two frames deep -- closed here the same way, before
    it ever reaches ``modifiers.make``."""
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(
        ctx, session, "clay_modifier_add",
        {"uid": uid1, "kind": "mirror", "params": {"weld": "abc"}},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "params"
    assert "weld" in result["content"][0]["text"]


def test_add_refuses_a_self_target_by_name() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    before = _history_len(ctx, session)
    result = agent_clay.call(
        ctx, session, "clay_modifier_add",
        {"uid": uid1, "kind": "boolean", "params": {"target": uid1}},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "params"
    assert "own uid" in result["content"][0]["text"]
    assert _history_len(ctx, session) == before


def test_add_refuses_a_target_naming_no_live_object() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(
        ctx, session, "clay_modifier_add",
        {"uid": uid1, "kind": "boolean", "params": {"target": 999999}},
    )
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["field"] == "params"
    assert structured["recovery"] == "read_scene"
    assert "999999" in result["content"][0]["text"]


def test_add_accepts_target_zero_as_none_chosen() -> None:
    """A boolean modifier with no target yet is legal to add -- it simply
    contributes nothing until evaluated, the same "skipped, not fatal" rule
    every other refusing modifier gets (see the error tests below)."""
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "boolean"})
    assert result["isError"] is False, result
    row = _row(result)
    assert row["modifiers"][0]["error"] == "Choose a target object."


def test_add_refuses_an_out_of_range_index_naming_the_field() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(
        ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "mirror", "index": 5}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "index"


def test_add_starts_this_session_owning_no_document_is_refused() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": 1, "kind": "mirror"})
    assert result["isError"] is True
    assert result["structuredContent"]["recovery"] == "start_document"


# --- clay_modifier_set -----------------------------------------------------


def test_set_merges_params_and_toggles_enabled_as_one_step() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    added = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "mirror"})
    modifier_id = _row(added)["modifier"]
    before = _history_len(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_modifier_set",
        {"uid": uid1, "modifier": modifier_id, "params": {"weld": 0.02}, "enabled": False},
    )
    assert result["isError"] is False, result
    row = _row(result)
    assert row["modifiers"][0]["params"] == {"axis": 0, "weld": 0.02}
    assert row["modifiers"][0]["enabled"] is False
    assert _history_len(ctx, session) == before + 1


def test_set_refuses_giving_neither_params_nor_enabled() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    added = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "mirror"})
    modifier_id = _row(added)["modifier"]
    result = agent_clay.call(
        ctx, session, "clay_modifier_set", {"uid": uid1, "modifier": modifier_id}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["changed"] is False


def test_set_refuses_an_unknown_modifier_id_and_recommends_a_re_read() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(
        ctx, session, "clay_modifier_set",
        {"uid": uid1, "modifier": 999, "enabled": False},
    )
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["field"] == "modifier"
    assert structured["recovery"] == "read_scene"


def test_set_enabled_must_be_a_boolean() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    added = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "mirror"})
    modifier_id = _row(added)["modifier"]
    result = agent_clay.call(
        ctx, session, "clay_modifier_set",
        {"uid": uid1, "modifier": modifier_id, "enabled": "yes"},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "enabled"


def test_set_target_change_is_validated_the_same_way_add_is() -> None:
    ctx, session, uid1, uid2 = _new_world()
    added = agent_clay.call(
        ctx, session, "clay_modifier_add",
        {"uid": uid1, "kind": "boolean", "params": {"target": uid2}},
    )
    modifier_id = _row(added)["modifier"]
    result = agent_clay.call(
        ctx, session, "clay_modifier_set",
        {"uid": uid1, "modifier": modifier_id, "params": {"target": uid1}},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "params"
    assert "own uid" in result["content"][0]["text"]


# --- clay_modifier_remove ---------------------------------------------------


def test_remove_drops_it_as_one_step_and_undo_restores_it() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    added = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "mirror"})
    modifier_id = _row(added)["modifier"]
    before = _history_len(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_modifier_remove", {"uid": uid1, "modifier": modifier_id}
    )
    assert result["isError"] is False, result
    assert _row(result)["removed"] == modifier_id
    assert "modifiers" not in _row(result)
    assert _history_len(ctx, session) == before + 1

    doc = _doc(ctx, session)
    assert doc.undo()
    assert [m.id for m in doc.by_uid(uid1).modifiers] == [modifier_id]


def test_remove_refuses_an_unknown_modifier_id() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(
        ctx, session, "clay_modifier_remove", {"uid": uid1, "modifier": 12345}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "modifier"


# --- clay_modifier_move -----------------------------------------------------


def test_move_reorders_the_stack_as_one_step() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    first = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "mirror"})
    first_id = _row(first)["modifier"]
    second = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "weld"})
    second_id = _row(second)["modifier"]
    before = _history_len(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_modifier_move", {"uid": uid1, "modifier": second_id, "index": 0}
    )
    assert result["isError"] is False, result
    row = _row(result)
    assert [m["id"] for m in row["modifiers"]] == [second_id, first_id]
    assert _history_len(ctx, session) == before + 1

    doc = _doc(ctx, session)
    assert doc.undo()
    assert [m.id for m in doc.by_uid(uid1).modifiers] == [first_id, second_id]


def test_move_order_changes_the_evaluated_result() -> None:
    """Order matters: a box's edges are all 90-degree corners, well past
    ``bevel``'s own default 30-degree threshold, so bevel-then-subdivide
    bevels the sharp box and then smooths the beveled result further, while
    subdivide-then-bevel first smooths the box into shallow-angle geometry
    that ``bevel``'s own angle test then finds nothing sharp enough to
    round -- a real, measured difference in face count between the two
    orders (bevel, subdivide = 96 faces; subdivide, bevel = 98, measured
    directly against this build), not merely a reordered list."""
    ctx, session, uid1, _uid2 = _new_world()
    b = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "bevel"})
    bevel_id = _row(b)["modifier"]
    s = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "subdivide"})
    subdivide_id = _row(s)["modifier"]
    scene = agent_clay.call(ctx, session, "clay_scene", {})
    row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid1)
    faces_bevel_first = row["evaluated"]["faces"]

    moved = agent_clay.call(
        ctx, session, "clay_modifier_move", {"uid": uid1, "modifier": subdivide_id, "index": 0}
    )
    assert moved["isError"] is False, moved
    scene = agent_clay.call(ctx, session, "clay_scene", {})
    row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid1)
    faces_subdivide_first = row["evaluated"]["faces"]

    assert faces_bevel_first != faces_subdivide_first
    del bevel_id  # named for readability only


def test_move_refuses_an_out_of_range_index() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    added = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "mirror"})
    modifier_id = _row(added)["modifier"]
    result = agent_clay.call(
        ctx, session, "clay_modifier_move", {"uid": uid1, "modifier": modifier_id, "index": 7}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "index"


# --- clay_modifier_apply ----------------------------------------------------


def test_apply_with_no_modifier_bakes_the_whole_stack() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "mirror"})
    agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "weld"})
    before = _history_len(ctx, session)
    before_faces = _doc(ctx, session).by_uid(uid1).mesh
    scene_before = agent_clay.call(ctx, session, "clay_scene", {})
    evaluated_before = next(
        o for o in _payload(scene_before)["objects"] if o["uid"] == uid1
    )["evaluated"]

    result = agent_clay.call(ctx, session, "clay_modifier_apply", {"uid": uid1})
    assert result["isError"] is False, result
    row = _row(result)
    assert row["changed"] is True
    assert "modifiers" not in row
    assert row["faces"] == evaluated_before["faces"]
    assert _history_len(ctx, session) == before + 1

    doc = _doc(ctx, session)
    assert doc.undo()
    assert doc.by_uid(uid1).mesh is before_faces
    assert len(doc.by_uid(uid1).modifiers) == 2


def test_apply_with_a_modifier_id_bakes_only_the_prefix() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    first = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "mirror"})
    first_id = _row(first)["modifier"]
    second = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "weld"})
    second_id = _row(second)["modifier"]

    result = agent_clay.call(
        ctx, session, "clay_modifier_apply", {"uid": uid1, "modifier": first_id}
    )
    assert result["isError"] is False, result
    row = _row(result)
    assert [m["id"] for m in row["modifiers"]] == [second_id]
    assert row["generator"] is None  # frozen, the same as document.set_mesh's own rule


def test_apply_skips_a_disabled_modifier_in_the_prefix_without_baking_it() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    added = agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "array"})
    modifier_id = _row(added)["modifier"]
    agent_clay.call(
        ctx, session, "clay_modifier_set", {"uid": uid1, "modifier": modifier_id, "enabled": False}
    )
    base_verts = len(_doc(ctx, session).by_uid(uid1).mesh.positions)

    result = agent_clay.call(ctx, session, "clay_modifier_apply", {"uid": uid1})
    assert result["isError"] is False, result
    row = _row(result)
    assert row["verts"] == base_verts  # the disabled array never contributed
    assert "modifiers" not in row


def test_apply_refuses_a_currently_erroring_prefix_with_its_own_message_and_bakes_nothing() -> (
    None
):
    ctx = _Ctx()
    session = agent_clay.Session()
    box = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    uid1 = _row(box)["uid"]
    # An open sheet (one quad, no volume) is not a closed solid -- the exact
    # target a boolean's own "needs every selected object to be a closed
    # solid" refusal exists for.
    sheet = agent_clay.call(
        ctx, session, "clay_add_mesh",
        {
            "positions": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            "faces": [[0, 1, 2, 3]],
            "translation": [3.0, 0.0, 0.0],
        },
    )
    uid2 = _row(sheet)["uid"]

    added = agent_clay.call(
        ctx, session, "clay_modifier_add",
        {"uid": uid1, "kind": "boolean", "params": {"target": uid2}},
    )
    assert added["isError"] is False, added
    # The exact "skipped, not fatal" case clay_scene already reports as an
    # 'error' on the row.
    scene = agent_clay.call(ctx, session, "clay_scene", {})
    row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid1)
    error_message = row["modifiers"][0]["error"]
    assert error_message

    before = _history_len(ctx, session)
    result = agent_clay.call(ctx, session, "clay_modifier_apply", {"uid": uid1})
    assert result["isError"] is True
    assert result["content"][0]["text"] == error_message
    assert _history_len(ctx, session) == before


def test_apply_refuses_on_an_empty_stack() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    result = agent_clay.call(ctx, session, "clay_modifier_apply", {"uid": uid1})
    assert result["isError"] is True
    assert result["structuredContent"]["changed"] is False


def test_apply_refuses_an_unknown_modifier_id() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    agent_clay.call(ctx, session, "clay_modifier_add", {"uid": uid1, "kind": "mirror"})
    result = agent_clay.call(
        ctx, session, "clay_modifier_apply", {"uid": uid1, "modifier": 404}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "modifier"


# --- cycle refusal, through set_modifiers -----------------------------------


def test_an_indirect_cycle_through_two_objects_is_refused_at_the_closing_modifier() -> None:
    ctx, session, uid1, uid2 = _new_world()
    first = agent_clay.call(
        ctx, session, "clay_modifier_add",
        {"uid": uid1, "kind": "boolean", "params": {"target": uid2}},
    )
    assert first["isError"] is False, first

    before = _history_len(ctx, session)
    second = agent_clay.call(
        ctx, session, "clay_modifier_add",
        {"uid": uid2, "kind": "boolean", "params": {"target": uid1}},
    )
    assert second["isError"] is True
    assert "cycle" in second["content"][0]["text"]
    assert _history_len(ctx, session) == before


# --- clay_scene / clay_add_primitive: modifiers and evaluated -------------


def test_a_fresh_object_carries_neither_modifiers_nor_evaluated() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    scene = agent_clay.call(ctx, session, "clay_scene", {})
    row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid1)
    assert "modifiers" not in row
    assert "evaluated" not in row


def test_bbox_size_center_measure_the_evaluated_mesh_not_the_base() -> None:
    """An array modifier translated off the object's own local origin grows
    the evaluated box past the base's own -- the one difference that proves
    ``_scene_row`` reads ``doc.evaluated``, not ``obj.mesh``, for
    bbox/size/center while ``faces``/``verts`` stay the base's own counts."""
    ctx, session, uid1, _uid2 = _new_world()
    scene = agent_clay.call(ctx, session, "clay_scene", {})
    base_row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid1)
    base_size_x = base_row["size"][0]
    base_faces = base_row["faces"]
    base_verts = base_row["verts"]

    added = agent_clay.call(
        ctx, session, "clay_modifier_add",
        {"uid": uid1, "kind": "array", "params": {"count": 3, "offset_x": 2.0}},
    )
    assert added["isError"] is False, added

    scene = agent_clay.call(ctx, session, "clay_scene", {})
    row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid1)
    assert row["size"][0] > base_size_x
    assert row["faces"] == base_faces  # base counts, unchanged
    assert row["verts"] == base_verts
    assert row["evaluated"]["vertices"] == base_verts * 3
    assert row["evaluated"]["faces"] == base_faces * 3
    assert row["evaluated"]["triangles"] == row["evaluated"]["faces"] * 2  # every face a quad

    document_bounds = _payload(scene)["bounds"]
    assert document_bounds["size"][0] >= row["size"][0]


def test_clay_add_primitive_and_clay_add_mesh_agree_with_clay_scene_about_the_row_shape() -> (
    None
):
    """A freshly placed object (no modifiers yet) answers the identical row
    shape from all three doors -- the shared ``_scene_row`` helper's own
    claim, still true once ``modifiers``/``evaluated`` are conditional."""
    ctx = _Ctx()
    session = agent_clay.Session()
    add_result = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    assert add_result["isError"] is False, add_result
    add_row = _row(add_result)
    assert "modifiers" not in add_row and "evaluated" not in add_row

    mesh_result = agent_clay.call(
        ctx, session, "clay_add_mesh",
        {
            "positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "faces": [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        },
    )
    assert mesh_result["isError"] is False, mesh_result
    mesh_row = _row(mesh_result)
    assert "modifiers" not in mesh_row and "evaluated" not in mesh_row


# --- clay_boolean consumes evaluated meshes ---------------------------------


def test_boolean_consumes_a_targets_evaluated_mesh_and_clears_its_stack() -> None:
    """A target with a solidify modifier is booleaned as the *shelled* solid,
    not the open surface its base mesh alone would be -- the base surface
    has no volume to union at all, so a union of it with anything is a
    strong, observable proof the evaluated mesh is what actually got
    consumed. The survivor's own stack is cleared in the same step: the
    modifier is now baked into what the union absorbed."""
    ctx, session, uid1, uid2 = _new_world()
    added = agent_clay.call(
        ctx, session, "clay_modifier_add", {"uid": uid2, "kind": "solidify"}
    )
    assert added["isError"] is False, added

    result = agent_clay.call(
        ctx, session, "clay_boolean", {"kind": "union", "uids": [uid1, uid2]}
    )
    assert result["isError"] is False, result
    survivor_uid = _payload(result)["uid"]
    doc = _doc(ctx, session)
    survivor = doc.by_uid(survivor_uid)
    assert survivor.modifiers == ()
    # Two boxes each solidified/unioned closed keep their own volumes;
    # nothing here collapsed to the open, zero-thickness surface a bare
    # base-mesh union of the target's un-shelled shell would have produced.
    assert len(survivor.mesh.positions) > 8


def test_boolean_refuses_in_element_mode_exactly_as_before_the_evaluated_change() -> None:
    """Guards against the evaluated-mesh change above having disturbed the
    unrelated element-mode refusal this tool already carried."""
    ctx, session, uid1, uid2 = _new_world()
    mode_result = agent_clay.call(ctx, session, "clay_element_mode", {"mode": "vertex"})
    assert mode_result["isError"] is False, mode_result
    result = agent_clay.call(
        ctx, session, "clay_boolean", {"kind": "union", "uids": [uid1, uid2]}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["recovery"] == "switch_mode"


# --- clay_diagnose stays on the base mesh -----------------------------------


def test_diagnose_reports_the_base_meshs_own_findings_not_the_evaluated_ones() -> None:
    """A hand-built open sheet is not closed -- diagnose says so -- and
    stacking a solidify on top (which would close it, were diagnose reading
    the evaluated mesh) changes nothing about that finding."""
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx, session, "clay_add_mesh",
        {
            "positions": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            "faces": [[0, 1, 2, 3]],
        },
    )
    assert result["isError"] is False, result
    uid = _row(result)["uid"]
    assert result["structuredContent"]["closed"] is False

    added = agent_clay.call(
        ctx, session, "clay_modifier_add", {"uid": uid, "kind": "solidify"}
    )
    assert added["isError"] is False, added

    diag = agent_clay.call(ctx, session, "clay_diagnose", {"uid": uid})
    assert diag["isError"] is False, diag
    findings = _payload(diag)["objects"][0]["findings"]
    assert any(f["kind"] == "hole" for f in findings)


# --- clay_analyze reads the evaluated mesh (kernels.mesh.analyze's doc=) ---


def test_analyze_bounds_and_volume_measure_the_evaluated_mesh() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    before = agent_clay.call(ctx, session, "clay_analyze", {"uids": [uid1]})
    base_volume = _payload(before)["objects"][0]["volume"]

    added = agent_clay.call(
        ctx, session, "clay_modifier_add",
        {"uid": uid1, "kind": "array", "params": {"count": 3, "offset_x": 2.0}},
    )
    assert added["isError"] is False, added

    after = agent_clay.call(ctx, session, "clay_analyze", {"uids": [uid1]})
    row = _payload(after)["objects"][0]
    assert row["volume"] == pytest.approx(base_volume * 3, rel=1e-6)
    assert row["bounds"]["max"][0] - row["bounds"]["min"][0] > 1.0  # wider than one box


# --- clay_program facts read the evaluated mesh (tools_batch._ConditionAccess) --


def test_condition_access_bounds_and_volume_read_the_evaluated_mesh() -> None:
    """``tools_batch._ConditionAccess`` -- what ``clay_program``'s own
    ``lo``/``hi``/``size``/``center``/``touches``/``grounded``/``floating``/
    ``volume`` facts are built from -- now measures ``doc.evaluated(uid)``
    rather than the base mesh.

    Exercised directly against ``_ConditionAccess`` rather than through a
    full ``clay_program`` call: a bare ``assert``-only program's own
    compiler checks a condition's ids against *that program's own* ``add``/
    ``figure``/``mesh``/``group`` steps (``program.py``'s own "unknown id"
    check, ``_validate_condition``), never against the live document by
    name -- and a modifier stack has no ``clay_program`` step kind of its
    own to add one inside that same program. This is the same live document
    and the same ``_ConditionAccess`` class a real program's ``assert``
    step reaches for at run time (``_run_live_assert``), so the claim is
    identical either way.
    """
    from warlock.studio.modes.clay.agent import tools_batch as agent_tools_batch

    ctx, session, uid1, _uid2 = _new_world()
    doc = _doc(ctx, session)
    access = agent_tools_batch._ConditionAccess(doc=doc, groups={})
    base_volume = access.volume(uid1)

    added = agent_clay.call(
        ctx, session, "clay_modifier_add",
        {"uid": uid1, "kind": "array", "params": {"count": 3, "offset_x": 2.0}},
    )
    assert added["isError"] is False, added

    lo, hi = access.bounds(uid1)
    assert hi[0] - lo[0] > 1.5  # wider than one 1x1x1 box
    assert access.volume(uid1) == pytest.approx(base_volume * 3, rel=1e-6)


# --- one undo step, batchable, reachable from clay_batch --------------------


@pytest.mark.parametrize(
    "name",
    [
        "clay_modifier_add",
        "clay_modifier_set",
        "clay_modifier_remove",
        "clay_modifier_move",
        "clay_modifier_apply",
    ],
)
def test_every_modifier_tool_is_batchable(name: str) -> None:
    assert name in set(agent_clay._HANDLERS) - agent_clay.BATCH_EXCLUDED


def test_a_batch_can_build_a_stack_and_apply_it_in_one_undo_step() -> None:
    ctx, session, uid1, _uid2 = _new_world()
    before = _history_len(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_batch",
        {
            "calls": [
                {"name": "clay_modifier_add", "arguments": {"uid": uid1, "kind": "mirror"}},
                {"name": "clay_modifier_add", "arguments": {"uid": uid1, "kind": "weld"}},
                {"name": "clay_modifier_apply", "arguments": {"uid": uid1}},
            ]
        },
    )
    assert result["isError"] is False, result
    assert _payload(result)["completed"] == 3
    assert _history_len(ctx, session) == before + 1

    doc = _doc(ctx, session)
    assert doc.by_uid(uid1).modifiers == ()
    assert doc.undo()
    assert doc.by_uid(uid1).modifiers == ()  # undo reverses the whole fold


# --- kernels.mesh.scratch: a batch that adds a modifier previews and
# transplants as one step ----------------------------------------------------


def test_a_batch_adding_a_modifier_previews_and_transplants_as_one_step() -> None:
    """``clay_scratch.clone``'s own comment: the stack is shared (an
    immutable tuple of frozen modifiers) and ``"modifiers"`` rides in
    ``_PROP_FIELDS`` -- a batch that adds one on the scratch clone previews
    it and, transplanted, lands on the real document as one step. Before
    that pair of one-line fixes, a preview drew every mirrored or arrayed
    object as its bare base mesh and a transplant back wrote the stack away
    entirely. Driven through ``studio.assistant.preview``'s own
    ``build``/``run_scratch`` -- the real door a Familiar-driven preview
    takes, not a hand-rolled scratch ctx -- the same way
    ``tests/familiar/test_scratch_ctx.py`` already does for every other
    tool.
    """
    from warlock.studio.assistant import preview as familiar_preview

    ctx, session, uid1, _uid2 = _new_world()
    doc = _doc(ctx, session)
    before_head = doc.history.head
    before_count = len(doc.history.history())

    scratch_ctx = familiar_preview.build(doc)
    batch_result = familiar_preview.run_scratch(
        scratch_ctx, "clay_batch",
        {"calls": [{"name": "clay_modifier_add", "arguments": {"uid": uid1, "kind": "mirror"}}]},
    )
    assert batch_result["isError"] is False, batch_result
    scratch = scratch_ctx.state.clay.docs[0].doc

    # Previewed on the clone; the real document is untouched.
    assert scratch.by_uid(uid1).modifiers
    assert doc.by_uid(uid1).modifiers == ()

    diff = clay_scratch.diff(doc, scratch)
    assert uid1 in diff.props_changed
    assert "modifiers" in diff.props_changed[uid1]

    changed = clay_scratch.transplant(doc, scratch, diff)
    assert changed
    assert doc.history.head != before_head
    # One step, folded: history() lists every step ever pushed on this
    # branch (done and undone both -- see UndoStack.history's own
    # docstring), so a single new entry here is what "one step" means,
    # exactly the shape test_scratch.py's own transplant tests already
    # check.
    assert len(doc.history.history()) == before_count + 1
    assert doc.by_uid(uid1).modifiers

    assert doc.undo()
    assert doc.history.head == before_head
    assert doc.by_uid(uid1).modifiers == ()


def test_scratch_preview_draws_the_modifier_not_the_bare_base_mesh() -> None:
    """The measured incident ``clay_scratch.clone``'s own comment names,
    reproduced directly: without ``modifiers=obj.modifiers`` a clone's copy
    of an object dropped its stack, so a preview of a batch that had already
    added one drew the object as its bare base mesh -- fewer vertices than
    the real evaluated result the batch actually built."""
    ctx, session, uid1, _uid2 = _new_world()
    doc = _doc(ctx, session)
    agent_clay.call(
        ctx, session, "clay_modifier_add",
        {"uid": uid1, "kind": "array", "params": {"count": 3, "offset_x": 2.0}},
    )
    real_evaluated_verts = len(doc.evaluated(uid1).positions)

    scratch = clay_scratch.clone(doc)
    scratch_evaluated_verts = len(scratch.evaluated(uid1).positions)
    assert scratch_evaluated_verts == real_evaluated_verts
    assert scratch.by_uid(uid1).modifiers == doc.by_uid(uid1).modifiers
