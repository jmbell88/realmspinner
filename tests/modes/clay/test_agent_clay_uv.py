"""``clay_uv`` (``dev/CLAY-PLAN.md`` tranche 6's integration half,
``studio/modes/clay/agent/tools_uv.py``): five uv actions behind one tool, and
what each promises -- one undo step, undoable, and a refusal that names a
``field`` for the four the brief calls out: a missing uv, no seams marked, an
unknown action, and a locked object.

Shares ``_Ctx``/``_payload``/``_new_agent_tab``/``_history_len`` with
``test_agent_clay.py`` rather than duplicating them -- see that module's own
fixtures; every box ``clay_add_primitive`` places already carries a uv
(``primitives.box`` calls ``box_unwrap`` on the way out), which is what makes
a bare ``_new_agent_tab`` enough setup for ``pack``/``density`` and every
refusal test below that does not need one.
"""

from __future__ import annotations

from typing import Any

import pytest

from warlock.studio.modes.clay import mode as clay_mode
from warlock.studio.modes.clay.agent import dispatch as agent_clay
from warlock.studio.modes.clay.agent import schema as agent_clay_schema
from warlock.studio.modes.clay.agent import tools_uv as agent_clay_tools_uv

from .test_agent_clay import _Ctx, _history_len, _new_agent_tab, _payload

# A hand-built mesh with no ``uv`` of its own -- ``clay_add_mesh`` never
# invents one -- so ``mesh.uv is None`` for every refusal below that needs an
# object genuinely missing one.
_TETRA_NO_UV = {
    "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "faces": [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
}


def _new_mesh_tab(ctx: _Ctx, session: agent_clay.Session) -> int:
    result = agent_clay.call(ctx, session, "clay_add_mesh", _TETRA_NO_UV)
    assert result["isError"] is False, result
    return _payload(result)["uid"]


def _doc(ctx: _Ctx, session: agent_clay.Session) -> Any:
    return clay_mode.ensure(ctx).get(session.tab_uid).doc


def _head(ctx: _Ctx, session: agent_clay.Session) -> int:
    """The document's own undo-position serial (``UndoStack.head``) --
    unlike ``_history_len`` (``len(history())``, the *combined* done+undone
    log this file's sibling module uses to prove a step was pushed), this is
    what actually moves backwards on ``clay_undo``, so it is what this
    module's own "and undoable" half of each test checks instead."""
    return _doc(ctx, session).history.head


# --- the derived enum, both ways ----------------------------------------------


def test_the_uv_action_enum_matches_uv_actions_both_ways() -> None:
    """``clay_uv``'s own schema enum, ``UV_ACTIONS``' keys and
    ``tools_uv._UV_ACTION_HANDLERS``' keys are the same five names, from three
    different files -- a same-membership pin across all three, so a fourth
    file drifting from the other two (an action added to one dispatch table
    but not the schema, or vice versa) fails here rather than at whichever
    action a caller happens to try first."""
    tool = next(t for t in agent_clay.tools() if t.name == "clay_uv")
    schema_enum = set(tool.schema["properties"]["action"]["enum"])
    assert schema_enum == set(agent_clay_schema.UV_ACTIONS)
    assert schema_enum == set(agent_clay_tools_uv._UV_ACTION_HANDLERS)


# --- one call, one undo step, and undoable ------------------------------------


def test_pack_is_one_undo_step_and_undoable() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    before_uv = _doc(ctx, session).by_uid(uid).mesh.uv.copy()
    before_len = _history_len(ctx, session)
    before_head = _head(ctx, session)

    result = agent_clay.call(ctx, session, "clay_uv", {"uid": uid, "action": "pack"})
    assert result["isError"] is False, result
    assert _history_len(ctx, session) == before_len + 1

    undo = agent_clay.call(ctx, session, "clay_undo", {})
    assert undo["isError"] is False, undo
    assert _head(ctx, session) == before_head
    assert (_doc(ctx, session).by_uid(uid).mesh.uv == before_uv).all()


def test_density_is_one_undo_step_and_undoable() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    before_uv = _doc(ctx, session).by_uid(uid).mesh.uv.copy()
    before_len = _history_len(ctx, session)
    before_head = _head(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_uv", {"uid": uid, "action": "density", "target": 256.0}
    )
    assert result["isError"] is False, result
    assert _history_len(ctx, session) == before_len + 1

    undo = agent_clay.call(ctx, session, "clay_undo", {})
    assert undo["isError"] is False, undo
    assert _head(ctx, session) == before_head
    assert (_doc(ctx, session).by_uid(uid).mesh.uv == before_uv).all()


def test_mark_seam_is_one_undo_step_and_undoable() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    before_len = _history_len(ctx, session)
    before_head = _head(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_uv", {"uid": uid, "action": "mark_seam", "edges": [[0, 1]]}
    )
    assert result["isError"] is False, result
    assert _payload(result)["seam_count"] == 1
    assert _history_len(ctx, session) == before_len + 1
    assert _doc(ctx, session).by_uid(uid).seams == ((0, 1),)

    undo = agent_clay.call(ctx, session, "clay_undo", {})
    assert undo["isError"] is False, undo
    assert _head(ctx, session) == before_head
    assert _doc(ctx, session).by_uid(uid).seams == ()


def test_clear_seam_is_one_undo_step_and_undoable() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    marked = agent_clay.call(
        ctx, session, "clay_uv", {"uid": uid, "action": "mark_seam", "edges": [[0, 1]]}
    )
    assert marked["isError"] is False, marked
    before_len = _history_len(ctx, session)
    before_head = _head(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_uv", {"uid": uid, "action": "clear_seam", "edges": [[0, 1]]}
    )
    assert result["isError"] is False, result
    assert _payload(result)["seam_count"] == 0
    assert _history_len(ctx, session) == before_len + 1
    assert _doc(ctx, session).by_uid(uid).seams == ()

    undo = agent_clay.call(ctx, session, "clay_undo", {})
    assert undo["isError"] is False, undo
    assert _head(ctx, session) == before_head
    assert _doc(ctx, session).by_uid(uid).seams == ((0, 1),)


def test_mark_seam_falls_back_to_the_current_edge_selection() -> None:
    """``edges`` omitted reads the object's *current* edge selection -- the
    same shape the tranche 6 integration spec gives the interactive Mark
    Seam button, so an agent that has already selected edges through
    ``clay_select_elements`` need not repeat the pair by hand."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    selected = agent_clay.call(
        ctx,
        session,
        "clay_select_elements",
        {"uid": uid, "mode": "edge", "edges": [[0, 1]]},
    )
    assert selected["isError"] is False, selected

    result = agent_clay.call(ctx, session, "clay_uv", {"uid": uid, "action": "mark_seam"})
    assert result["isError"] is False, result
    assert _doc(ctx, session).by_uid(uid).seams == ((0, 1),)


def _all_edges(mesh: Any) -> list[list[int]]:
    """Every edge of *mesh*, as ``[vertex, vertex]`` pairs -- marking all of
    them as seams is the simplest way to guarantee an island actually opens
    (see ``test_unwrap_seams_is_one_undo_step_and_undoable``'s own comment):
    a single seam edge on a closed solid like a box does not disconnect its
    face-adjacency graph at all, so ``islands_by_seams`` still reports one
    closed island and ``unwrap_lscm`` correctly refuses it -- every edge cut
    turns each face into its own single-face, unclosed island instead."""
    from warlock.kernels.mesh.adjacency import adjacency

    return adjacency(mesh).edge_verts.tolist()


def test_unwrap_seams_is_one_undo_step_and_undoable() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    edges = _all_edges(_doc(ctx, session).by_uid(uid).mesh)
    marked = agent_clay.call(
        ctx, session, "clay_uv", {"uid": uid, "action": "mark_seam", "edges": edges}
    )
    assert marked["isError"] is False, marked
    before_uv = _doc(ctx, session).by_uid(uid).mesh.uv.copy()
    before_len = _history_len(ctx, session)
    before_head = _head(ctx, session)

    result = agent_clay.call(ctx, session, "clay_uv", {"uid": uid, "action": "unwrap_seams"})
    assert result["isError"] is False, result
    assert _history_len(ctx, session) == before_len + 1

    undo = agent_clay.call(ctx, session, "clay_undo", {})
    assert undo["isError"] is False, undo
    assert _head(ctx, session) == before_head
    assert (_doc(ctx, session).by_uid(uid).mesh.uv == before_uv).all()


# --- refusals -------------------------------------------------------------


@pytest.mark.parametrize("action", ["pack", "density"])
def test_pack_and_density_refuse_an_object_with_no_uv(action: str) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_mesh_tab(ctx, session)
    args = {"uid": uid, "action": action}
    if action == "density":
        args["target"] = 256.0
    result = agent_clay.call(ctx, session, "clay_uv", args)
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "uid"
    assert result["structuredContent"]["changed"] is False


def test_unwrap_seams_refuses_with_no_seams_marked() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    result = agent_clay.call(ctx, session, "clay_uv", {"uid": uid, "action": "unwrap_seams"})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "uid"
    assert result["structuredContent"]["changed"] is False


def test_an_unknown_action_is_refused_by_name() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    result = agent_clay.call(ctx, session, "clay_uv", {"uid": uid, "action": "bogus"})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "action"


@pytest.mark.parametrize(
    ("action", "extra"),
    [
        ("pack", {}),
        ("density", {"target": 256.0}),
        ("mark_seam", {"edges": [[0, 1]]}),
        ("clear_seam", {"edges": [[0, 1]]}),
    ],
)
def test_every_mesh_editing_action_refuses_a_locked_object(action: str, extra: dict) -> None:
    """``set_mesh``/``set_seams`` both refuse a locked object -- a seam or a
    uv layout is authoring intent about the object's own geometry, the
    identical footing a mesh edit already stands on (see
    ``tools_uv.py``'s own module docstring)."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    lock = agent_clay.call(ctx, session, "clay_lock", {"uids": [uid], "locked": True})
    assert lock["isError"] is False, lock

    result = agent_clay.call(ctx, session, "clay_uv", {"uid": uid, "action": action, **extra})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "uid"
    assert result["structuredContent"]["changed"] is False


def test_mark_seam_with_no_edges_and_nothing_selected_names_the_field() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    result = agent_clay.call(ctx, session, "clay_uv", {"uid": uid, "action": "mark_seam"})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "edges"


def test_mark_seam_refuses_a_pair_that_is_not_a_real_edge() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    result = agent_clay.call(
        ctx, session, "clay_uv", {"uid": uid, "action": "mark_seam", "edges": [[0, 6]]}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "edges"


# --- clay_scene's own uv facts -------------------------------------------------


def test_clay_scene_reports_uv_facts_only_once_the_object_has_one() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)  # a box: already has a uv
    scene = agent_clay.call(ctx, session, "clay_scene", {})
    assert scene["isError"] is False, scene
    row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid)
    assert "uv" in row
    assert row["uv"]["islands"] > 0

    no_uv_uid = _new_mesh_tab(ctx, session)
    scene2 = agent_clay.call(ctx, session, "clay_scene", {})
    assert scene2["isError"] is False, scene2
    row2 = next(o for o in _payload(scene2)["objects"] if o["uid"] == no_uv_uid)
    assert "uv" not in row2


def test_clay_scene_reports_role_and_collider_kind_on_an_ordinary_object() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    scene = agent_clay.call(ctx, session, "clay_scene", {})
    row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid)
    assert row["role"] == "mesh"
    assert row["collider_kind"] == ""


# --- clay_catalog: the uv_actions topic ----------------------------------------


def test_clay_catalog_uv_actions_matches_uv_actions_verbatim() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(ctx, session, "clay_catalog", {"topic": "uv_actions"})
    assert result["isError"] is False, result
    assert _payload(result)["catalogue"] == agent_clay_schema._uv_action_catalog()
    for name in agent_clay_schema.UV_ACTIONS:
        assert name in _payload(result)["catalogue"]


# --- a batch mixing a uv action and a collider fit folds into one step --------


def test_a_batch_mixing_a_uv_action_and_a_collider_fit_folds_into_one_step() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    before_len = _history_len(ctx, session)
    before_head = _head(ctx, session)

    result = agent_clay.call(
        ctx,
        session,
        "clay_batch",
        {
            "calls": [
                {"name": "clay_uv", "arguments": {"uid": uid, "action": "pack"}},
                {"name": "clay_collider", "arguments": {"uids": [uid], "kind": "box"}},
            ]
        },
    )
    assert result["isError"] is False, result
    assert _history_len(ctx, session) == before_len + 1

    doc = _doc(ctx, session)
    colliders = [o for o in doc.objects if o.role == "collider"]
    assert len(colliders) == 1
    assert colliders[0].parent == uid

    undo = agent_clay.call(ctx, session, "clay_undo", {})
    assert undo["isError"] is False, undo
    assert _head(ctx, session) == before_head
    assert not [o for o in _doc(ctx, session).objects if o.role == "collider"]
