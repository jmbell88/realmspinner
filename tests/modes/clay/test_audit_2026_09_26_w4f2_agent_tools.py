"""Regressions for the 2026-09-26 audit's Clay-agent findings (fixer w4f2),
all inside ``studio/modes/clay/agent/`` (the tool doors, not the kernels the
findings' own text also touches -- see each test's own docstring for the
split).

clay-agent-tools-04: ``clay_delete`` with a repeated uid removed on pass one
and raised ``KeyError`` on pass two, before ``collapse_since`` ever ran.

clay-agent-tools-06: an enum argument was checked with ``x not in a_dict`` or
``a_dict.get(x)`` -- both of which raise a bare, unhashable ``TypeError`` for
a list or object argument, reaching ``call()``'s generic "failed
unexpectedly" backstop instead of a field-named refusal.

clay-agent-tools-07: ``_object_row_output_schema`` declared ``generator`` a
plain ``string``, but ``_scene_row`` answers ``None`` for a hand-built
(``clay_add_mesh``) or frozen (topology-edited) object -- so ``clay_add_mesh``
and ``clay_scene`` both violate their own declared ``outputSchema``.

clay-agent-tools-08: ``_resolve_uids`` never deduplicated, so
``clay_analyze uids:[u, u]`` reported a self-pair overlap and
``clay_collider`` with a repeated uid added two colliders for one object --
the same missing de-duplication as clay-agent-tools-04, in the one function
several handlers share.

clay-agent-tools-09: ``int(float('inf'))`` and ``round(float('inf'))`` raise
``OverflowError``, which is not among the exceptions ``int(...)``'s own
``except (TypeError, ValueError)`` clauses caught -- another route to the
same generic backstop as -06.

clay-document-02 (agent-door half only -- the generator-level "floor the
extents" fix for ``arch``/``stairs``/``doorway`` is kernels/mesh territory,
owned by a different fixer in this pass, and landed during this same audit
pass: ``arch(width=0)`` now floors to a buildable shape instead of raising,
so the two tests below monkeypatch a generator to still raise, the way any
future generator that does not floor its own degenerate input would):
``clay_add_primitive`` opened its undo gesture and placed the default-params
object *before* building the mesh with the caller's own params, so a
generator that raises on a degenerate value left that default object
standing in the document and ``history``'s own ``_open_gestures`` counter
permanently incremented, because nothing ever reached ``collapse_since``.
``clay_set_params`` shared the same "build, then refuse" order, minus the
leaked object (it never places one) but with the identical field-less
generic backstop.

agents-clay-01: ``schema.REFERENCE_TOOLS`` has no consumer anywhere in
``src/`` or ``tests/`` after the Familiar removal (2026-09-26) -- only its
own docstring and ``BATCH_EXCLUDED``'s point at it.
"""

from __future__ import annotations

import jsonschema
import pytest

from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import program as ap
from realmspinner.studio.modes.clay.agent import schema as agent_clay_schema

from .test_agent_clay import _TETRA_MESH_ARGS, _Ctx, _history_len, _new_agent_tab, _payload

# --- clay-agent-tools-04 / -08: _resolve_uids never deduplicated ------------


def test_clay_delete_with_a_repeated_uid_deletes_once_and_reports_one_undo_step() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    before = _history_len(ctx, session)

    result = agent_clay.call(ctx, session, "clay_delete", {"uids": [uid, uid]})

    assert result["isError"] is False, result
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert list(tab.doc.objects) == []
    assert _history_len(ctx, session) == before + 1


def test_clay_analyze_with_a_repeated_uid_does_not_report_a_self_pair_overlap() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid1 = _new_agent_tab(ctx, session, "box")
    uid2 = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "cylinder"})
    assert uid2["isError"] is False, uid2

    result = agent_clay.call(ctx, session, "clay_analyze", {"uids": [uid1, uid1]})
    assert result["isError"] is False, result
    pairs = _payload(result)["pairs"]
    assert pairs == [], f"a single named object reported a pair against itself: {pairs}"


def test_clay_collider_with_a_repeated_uid_adds_one_collider_not_two() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    before = {o.uid for o in tab.doc.objects}

    result = agent_clay.call(ctx, session, "clay_collider", {"uids": [uid, uid], "kind": "box"})

    assert result["isError"] is False, result
    after = {o.uid for o in tab.doc.objects}
    assert len(after - before) == 1, "a repeated uid added more than one collider"


# --- clay-agent-tools-06: a non-string enum argument must not crash ---------

_NON_STRING_ENUM_CASES = [
    ("clay_add_primitive", {"generator": ["box"]}, "generator", False),
    ("clay_add_figure", {"key": ["a"]}, "key", False),
    ("clay_catalog", {"topic": ["ops"]}, "topic", False),
    ("clay_collider", {"uids": [], "kind": ["box"]}, "kind", True),
    ("clay_modifier_add", {"uid": 0, "kind": ["mirror"]}, "kind", True),
    ("clay_select_by", {"uid": 0, "query": ["all"]}, "query", True),
    ("clay_render", {"view": ["front"]}, "view", True),
    ("clay_validate", {"profile": ["godot-desktop"]}, "profile", True),
    ("clay_export", {"engine": ["godot4"]}, "engine", True),
    ("clay_uv", {"uid": 0, "action": ["pack"]}, "action", True),
    ("clay_reference_add", {"name": "r", "png_base64": "", "view": ["other"]}, "view", False),
]


@pytest.mark.parametrize("name, args, field, needs_tab", _NON_STRING_ENUM_CASES)
def test_a_non_string_enum_argument_is_refused_naming_its_field_not_the_generic_backstop(
    name: str, args: dict, field: str, needs_tab: bool
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    if needs_tab:
        uid = _new_agent_tab(ctx, session, "box")
        args = {**args}
        for key in ("uid",):
            if key in args and args[key] == 0:
                args[key] = uid
        if "uids" in args and args["uids"] == []:
            args["uids"] = [uid]

    result = agent_clay.call(ctx, session, name, args)

    assert result["isError"] is True, result
    assert not result.get("content", [{}])[0].get("text", "").startswith(f"{name} failed"), (
        f"{name} reached the generic backstop instead of a field-named refusal: {result}"
    )
    assert result["structuredContent"].get("field") == field, result


# --- clay-agent-tools-07: outputSchema must admit a null generator ----------


def test_add_mesh_reply_validates_against_its_declared_output_schema() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(ctx, session, "clay_add_mesh", _TETRA_MESH_ARGS)
    assert result["isError"] is False, result
    tool = next(t for t in agent_clay.tools() if t.name == "clay_add_mesh")
    assert result["structuredContent"]["generator"] is None, (
        "sanity: a hand-built object must have no generator"
    )
    jsonschema.validate(instance=result["structuredContent"], schema=tool.output_schema)


def test_clay_scene_reply_validates_against_its_declared_output_schema_for_a_hand_built_object() -> None:  # noqa: E501
    ctx = _Ctx()
    session = agent_clay.Session()
    added = agent_clay.call(ctx, session, "clay_add_mesh", _TETRA_MESH_ARGS)
    assert added["isError"] is False, added

    result = agent_clay.call(ctx, session, "clay_scene", {})
    assert result["isError"] is False, result
    tool = next(t for t in agent_clay.tools() if t.name == "clay_scene")
    jsonschema.validate(instance=result["structuredContent"], schema=tool.output_schema)


# --- clay-agent-tools-09: int(float('inf'))/round(...) raise OverflowError --


def test_an_infinite_material_index_is_refused_naming_field_material_not_the_generic_backstop() -> None:  # noqa: E501
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx, session, "clay_add_primitive", {"generator": "box", "material": float("inf")}
    )
    assert result["isError"] is True
    assert result["structuredContent"].get("field") == "material", result


def test_an_infinite_uid_in_clay_boolean_is_refused_naming_field_uids_not_the_generic_backstop() -> None:  # noqa: E501
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    result = agent_clay.call(
        ctx, session, "clay_boolean", {"kind": "union", "uids": [float("inf")]}
    )
    assert result["isError"] is True
    assert result["structuredContent"].get("field") == "uids", result


def test_an_infinite_collider_param_is_refused_naming_field_params_not_the_generic_backstop() -> None:  # noqa: E501
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    result = agent_clay.call(
        ctx,
        session,
        "clay_collider",
        {"uids": [uid], "kind": "convex", "params": {"max_faces": float("inf")}},
    )
    assert result["isError"] is True
    assert result["structuredContent"].get("field") == "params", result


def test_an_infinite_uid_argument_is_refused_naming_its_field_not_the_generic_backstop() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    result = agent_clay.call(ctx, session, "clay_transform", {"uid": float("inf")})
    assert result["isError"] is True
    assert result["structuredContent"].get("field") == "uid", result


# --- clay-document-02 (agent-door half): a degenerate generator value must
# not leave a default object standing or an undo gesture open --------------


def _raise_zero_division(**_kwargs) -> None:
    """A stand-in generator build function -- see the two tests below for
    why a real one no longer serves this purpose."""
    raise ZeroDivisionError("synthetic: this generator always refuses")


def test_clay_add_primitive_on_a_degenerate_generator_value_leaves_no_object_and_no_open_gesture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``arch(width=0)`` used to raise ``ZeroDivisionError`` deep inside the
    generator (its own UV-island helper divided by ``2.0 * r_out``), which is
    what originally reproduced this finding -- but the kernels/mesh half of
    this same audit pass (clay-document-02) now floors ``arch``'s own
    ``width`` before it can reach that division, so ``arch(width=0)`` builds
    a valid mesh rather than raising, and can no longer stand in for "a
    generator refuses". Repointed at a generator monkeypatched to still
    raise, exactly the shape a future generator that does not floor its own
    degenerate input would still hit.

    What belongs to this test is that ``_h_add_primitive`` used to open
    ``doc.history.mark()`` and place the default-params object *before*
    trying to build the mesh with the caller's own params, so a raising
    generator left that default object standing in the document and the
    mark's own ``_open_gestures`` counter permanently incremented -- nothing
    ever called ``collapse_since`` to close it. Fixed by building (and
    refusing) the mesh before the tab is even minted -- exactly the order
    ``_h_add_mesh`` already validates in
    (``test_a_face_index_past_positions_is_refused_naming_the_face``'s own
    ``session.tab_uid == ""`` assertion is the precedent this now matches).
    """
    monkeypatch.setitem(bp.GENERATORS, "box", (bp.GENERATORS["box"][0], _raise_zero_division))
    ctx = _Ctx()
    session = agent_clay.Session()

    result = agent_clay.call(
        ctx,
        session,
        "clay_add_primitive",
        {"generator": "box", "params": {"size": [2.0, 2.0, 2.0]}},
    )

    assert result["isError"] is True, result
    assert result["structuredContent"].get("field") == "params", result
    assert session.tab_uid == "", "a refused params build must mint no document at all"


def test_clay_set_params_on_a_degenerate_generator_value_refuses_naming_field_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """See the sibling test above for why a monkeypatched generator, not
    ``arch(width=0)``, is what still reproduces "the generator itself
    refuses" now that the kernels/mesh half of clay-document-02 floors
    ``arch``'s own extents."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    before_params = dict(tab.doc.by_uid(uid).params)
    before_mesh = tab.doc.by_uid(uid).mesh

    # Patched only after the object above is safely built with the real
    # generator -- this test is about a rebuild refusing, not the initial
    # placement.
    monkeypatch.setitem(bp.GENERATORS, "box", (bp.GENERATORS["box"][0], _raise_zero_division))

    result = agent_clay.call(
        ctx, session, "clay_set_params", {"uid": uid, "params": {"size": [2.0, 2.0, 2.0]}}
    )

    assert result["isError"] is True, result
    assert result["structuredContent"].get("field") == "params", result
    obj = tab.doc.by_uid(uid)
    assert obj.params == before_params
    assert obj.mesh is before_mesh


# --- agents-clay-01: REFERENCE_TOOLS has no consumer ------------------------


def test_reference_tools_constant_is_gone_now_that_nothing_consumes_it() -> None:
    """The 2026-09-26 audit: ``REFERENCE_TOOLS`` was only ever read by its own
    docstring and ``BATCH_EXCLUDED``'s cross-reference to it -- no consumer
    survived the Familiar removal (see ``BATCH_EXCLUDED`` immediately below
    it in ``schema.py`` for where the same reasoning now lives directly)."""
    assert not hasattr(agent_clay_schema, "REFERENCE_TOOLS")


# --- clay_program: the same non-string enum crash, through the compiler ----


def test_a_non_string_generator_in_a_program_step_is_a_program_error_not_a_bare_typeerror() -> None:  # noqa: E501
    with pytest.raises(ap.ProgramError):
        ap.compile_program({"steps": [{"add": {"generator": ["box"]}}]})


def test_a_non_string_figure_key_in_a_program_step_is_a_program_error_not_a_bare_typeerror() -> None:  # noqa: E501
    with pytest.raises(ap.ProgramError):
        ap.compile_program({"steps": [{"figure": {"key": ["a"]}}]})
