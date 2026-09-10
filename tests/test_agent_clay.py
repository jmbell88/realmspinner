"""What Clay's agent tool surface (``studio/agent_clay.py``) promises to hold.

Four claims are pinned here, each stated in that module's own docstring.

**The bidirectional derivation gate.** ``tools()`` builds its enums from
``primitives.GENERATORS``, ``presets.ASSEMBLIES`` and ``clay_ops.OPS`` rather
than naming shapes and ops by hand -- the whole point being that a thirteenth
primitive needs no edit here. A same-membership assertion only proves today's
three lists agree; it says nothing about *tomorrow's* fourth generator finding
its way in with nobody touching this file. So each pair is pinned twice: once
as a set equality (today's registry vs. today's enum) and once as a live gate
-- monkeypatch a new entry into the registry and assert it shows up in
``tools()`` with no code here changed at all. That second half is what makes
this a gate rather than a snapshot; see ``.claude/skills/warlock-sweep/
references/gate-patterns.md``. ``clay_batch``'s own name enum gets the same
treatment against ``_HANDLERS`` minus ``BATCH_EXCLUDED``.

**The blast-radius claim.** An agent session reaches exactly one document,
named by ``Session.tab_uid`` and resolved fresh through ``ClayState`` on every
call -- never a fallback onto whichever tab the user has open, and never a
reach across sessions into another tab's objects. Both halves are pinned:
every document-scoped tool refuses on a session whose tab is gone (the four
reference tools are per-session and hold no document, so they are exempt --
see ``_SESSION_ONLY``), and a uid that belongs to a *different* tab is refused
rather than acted on, with that other document provably untouched (a
byte-for-byte comparison of ``serialize.wblk_bytes``, which is deterministic
by construction -- see that module's docstring).

**Everyday behaviour**: one tool call is one undo step (with the two
documented exceptions, ``clay_batch`` and ``clay_undo``/``clay_redo``), a
clamp is reported back rather than silently applied, ``clay_material``
repaints every face rather than only the object's default slot, a refusal
names its ``field`` where one is knowable, and ``call()`` truly never raises.

**The structured-results claim.** Every tool's JSON answer is duplicated into
``structuredContent`` (``_json``, the shape most of this module's tools
answer through), so a client can branch on a field instead of re-parsing the
text block a model reads -- except a result that carries a picture, which
never duplicates its header into ``structuredContent``: ``clay_render``
(``ok(header, *pngs)``) and ``clay_reference_get`` (``ok(text(...),
image_png(...))``) both build their result directly rather than through
``_json``, because an image block has no JSON to duplicate. Stated
structurally rather than as a name or a count, and pinned exhaustively --
walking every entry in ``_HANDLERS`` rather than a hand-kept subset -- by
``test_every_tool_answers_with_structured_content_unless_its_reply_carries_a_picture``.
Three tools -- ``clay_scene``, ``clay_add_primitive`` and ``clay_diagnose``
-- also declare an ``outputSchema`` describing that shape, and none declares
``required`` or ``additionalProperties: false``, because a refusal shares
the same result envelope and its ``structuredContent`` is only ever whatever
``field`` it names.

``clay_render`` needs a real moderngl context to build its private viewport
(``ctx.viewer.ctx``); this suite runs with no GL at all. Most of its tests
below pin only the shape of a refusal -- a missing context is a clean one,
never an unhandled exception -- but a handful use ``_install_fake_view`` to
swap in a fake ``ClayView`` that returns a tiny real PNG without touching GL,
which is enough to exercise the multi-view and reference-comparison result
shapes end to end. Real pixels from a real GPU are not covered by this file.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from warlock.studio import agent_clay, clay_mode, clay_ops
from warlock.studio.clay import document as bd
from warlock.studio.clay import presets, serialize
from warlock.studio.clay import primitives as bp
from warlock.studio.panes import clay_tools as pane_clay_tools
from warlock.studio.viewer import math3d as m3

# --- a ctx double, no imgui, no GL, no pygame --------------------------------


class _Cache:
    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


class _Ctx:
    """Exactly what ``agent_clay`` and ``clay_mode`` read off ``ctx``.

    No ``viewer`` attribute -- ``clay_render`` is the one tool this omission
    is meant to exercise; see the module docstring.
    """

    def __init__(self, svc: Any = None) -> None:
        self.state = SimpleNamespace(clay=None)
        self.svc = svc
        self.cache = _Cache()
        self.toasts: list[tuple[str, str]] = []
        # ``clay_mode.adopt`` calls ``remember_path(ctx, path)`` on every new
        # document, and ``path`` is always ``None`` here (an agent's tab has
        # no file) -- so ``recents.remember`` no-ops before ever reading this,
        # but the attribute access itself (``ctx.settings``) still has to
        # succeed for that no-op to be reached at all.
        self.settings = SimpleNamespace()

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _payload(result: dict) -> Any:
    """The JSON a successful tool call answered with."""
    return json.loads(result["content"][0]["text"])


def _new_agent_tab(ctx: _Ctx, session: agent_clay.Session, generator: str = "box") -> int:
    """Mint the session's document via the real tool path. -> the object's uid."""
    result = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": generator})
    assert result["isError"] is False, result
    return _payload(result)["uid"]


def _history_len(ctx: _Ctx, session: agent_clay.Session) -> int:
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    return len(tab.doc.history.history())


def _tiny_png() -> bytes:
    im = Image.new("RGB", (4, 4), "white")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _add_inline_reference_raw(
    ctx: _Ctx, session: agent_clay.Session, name: str, view: str = "other"
) -> dict:
    b64 = base64.b64encode(_tiny_png()).decode("ascii")
    return agent_clay.call(
        ctx, session, "clay_reference_add", {"name": name, "png_base64": b64, "view": view}
    )


def _add_inline_reference(
    ctx: _Ctx, session: agent_clay.Session, name: str = "ref1", view: str = "other"
) -> dict:
    result = _add_inline_reference_raw(ctx, session, name, view)
    assert result["isError"] is False, result
    return result


class _FakeView:
    """A ``ClayView`` stand-in that returns a real tiny PNG without touching GL.

    Records every call's kwargs so a test can assert what ``clay_render``
    asked for -- what view/angles, what bounds, what grid setting -- without
    a moderngl context to actually read pixels back from.
    """

    def __init__(self, png: bytes | None = None) -> None:
        self.png = png or _tiny_png()
        self.calls: list[dict[str, Any]] = []

    def render_png(
        self,
        doc: Any,
        *,
        size: int,
        view: str | None = None,
        angles: Any = None,
        bounds: Any = None,
        grid: bool = False,
        frame: bool = True,
    ) -> bytes:
        del doc, frame
        self.calls.append(
            {"size": size, "view": view, "angles": angles, "bounds": bounds, "grid": grid}
        )
        return self.png


def _install_fake_view(monkeypatch: pytest.MonkeyPatch, png: bytes | None = None) -> _FakeView:
    fake = _FakeView(png)
    monkeypatch.setattr(agent_clay, "_view_for", lambda ctx: fake)
    return fake


# --- the bidirectional derivation gate ---------------------------------------


def test_every_generator_key_is_a_clay_add_primitive_enum_option_and_vice_versa() -> None:
    tools = {t.name: t for t in agent_clay.tools()}
    enum = set(tools["clay_add_primitive"].schema["properties"]["generator"]["enum"])
    assert enum == set(bp.GENERATORS)


def test_every_assembly_key_is_a_clay_add_figure_enum_option_and_vice_versa() -> None:
    tools = {t.name: t for t in agent_clay.tools()}
    enum = set(tools["clay_add_figure"].schema["properties"]["key"]["enum"])
    assert enum == set(presets.ASSEMBLIES)


def test_every_op_name_is_a_clay_op_enum_option_and_vice_versa() -> None:
    tools = {t.name: t for t in agent_clay.tools()}
    enum = set(tools["clay_op"].schema["properties"]["name"]["enum"])
    assert enum == {op.name for op in clay_ops.OPS}


def test_a_thirteenth_generator_reaches_the_agent_surface_with_no_edit_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claim that makes the gate above a gate rather than a snapshot.

    ``monkeypatch.setitem`` on the real registry, restored automatically --
    the module docstring's own instruction is not to mutate ``GENERATORS`` by
    hand, and this is why: a hand-restore that a later assertion failure
    skipped would leave a fake generator live for every test after this one.
    """
    monkeypatch.setitem(bp.GENERATORS, "thirteenth_shape", ({"size": 1.0}, bp.box))
    tools = {t.name: t for t in agent_clay.tools()}
    enum = tools["clay_add_primitive"].schema["properties"]["generator"]["enum"]
    assert "thirteenth_shape" in enum


def test_a_ninth_figure_reaches_the_agent_surface_with_no_edit_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(presets.ASSEMBLIES, "ninth_figure", ("Ninth", lambda: ()))
    tools = {t.name: t for t in agent_clay.tools()}
    assert "ninth_figure" in tools["clay_add_figure"].schema["properties"]["key"]["enum"]


def test_every_element_mode_is_a_clay_element_mode_enum_option_and_vice_versa() -> None:
    from warlock.studio.clay import elements as clay_elements

    tools = {t.name: t for t in agent_clay.tools()}
    enum = set(tools["clay_element_mode"].schema["properties"]["mode"]["enum"])
    assert enum == set(clay_elements.MODES)


def test_every_query_name_is_a_clay_select_by_enum_option_and_vice_versa() -> None:
    from warlock.studio.clay import select as clay_select_mod

    tools = {t.name: t for t in agent_clay.tools()}
    enum = set(tools["clay_select_by"].schema["properties"]["query"]["enum"])
    assert enum == set(clay_select_mod.QUERIES)


def test_a_seventh_query_reaches_the_agent_surface_with_no_edit_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same live-gate shape as the thirteenth-generator and ninth-figure
    tests above, for the fourth derived registry: monkeypatch a new entry
    into ``select.QUERIES``, restored automatically, and assert it shows up
    in ``clay_select_by``'s own enum with no code here touched at all."""
    from warlock.studio.clay import elements as clay_elements
    from warlock.studio.clay import select as clay_select_mod

    fake = clay_select_mod.Query(
        name="seventh",
        modes=("face",),
        args=("slot",),
        run=lambda mesh, slot: clay_elements.ElementSel(),
        hint="a fake seventh query",
    )
    monkeypatch.setitem(clay_select_mod.QUERIES, "seventh", fake)
    tools = {t.name: t for t in agent_clay.tools()}
    assert "seventh" in tools["clay_select_by"].schema["properties"]["query"]["enum"]


def test_every_query_argument_name_has_a_schema_fragment_and_vice_versa() -> None:
    """``select.py`` holds no JSON-schema knowledge of its own (its own
    module docstring's rule) -- ``agent_clay._QUERY_ARG_SCHEMAS`` is the one
    place that vocabulary is spelled out, and this gate is what stops a query
    growing an argument nobody here can express, or an entry here nothing
    asks for any more."""
    from warlock.studio.clay import select as clay_select_mod

    all_args = {a for q in clay_select_mod.QUERIES.values() for a in q.args}
    assert all_args == set(agent_clay._QUERY_ARG_SCHEMAS)


def test_every_tool_not_excluded_from_batching_is_in_the_batch_name_enum() -> None:
    tools = {t.name: t for t in agent_clay.tools()}
    call_schema = tools["clay_batch"].schema["properties"]["calls"]["items"]
    enum = set(call_schema["properties"]["name"]["enum"])
    assert enum == set(agent_clay._HANDLERS) - agent_clay.BATCH_EXCLUDED


def test_every_handler_has_a_tool_and_every_tool_has_a_handler() -> None:
    tool_names = {t.name for t in agent_clay.tools()}
    assert tool_names == set(agent_clay._HANDLERS)


def test_every_tool_schema_is_a_plausible_json_schema_object() -> None:
    for tool in agent_clay.tools():
        schema = tool.schema
        assert schema["type"] == "object"
        assert isinstance(schema["properties"], dict)
        for required in schema.get("required", []):
            assert required in schema["properties"]


# --- the blast-radius claim ---------------------------------------------------

# One call per tool, with just enough arguments to pass whatever this handler
# validates *before* it resolves the session's tab -- ``clay_add_primitive``,
# ``clay_add_figure`` and ``clay_batch`` check their own arguments first,
# every other handler here calls ``_tab`` before touching ``args`` at all.
# See ``agent_clay.py``'s source for that ordering; getting it backwards here
# would test argument validation instead of the blast-radius gate.
_NEEDS_A_TAB = [
    ("clay_scene", {}),
    ("clay_transform", {}),
    ("clay_set_params", {}),
    ("clay_material", {}),
    ("clay_boolean", {}),
    ("clay_select", {}),
    ("clay_element_mode", {}),
    ("clay_select_elements", {}),
    ("clay_select_by", {}),
    ("clay_elements", {}),
    ("clay_op", {}),
    ("clay_render", {}),
    ("clay_diagnose", {}),
    ("clay_export", {}),
    ("clay_undo", {}),
    ("clay_redo", {}),
    ("clay_delete", {}),
    ("clay_rename", {}),
    ("clay_batch", {"calls": [{"name": "clay_scene", "arguments": {}}]}),
]

# The four reference tools hold no document at all -- requiring one in order
# to hold a picture would be a rule with no reason behind it, since a
# reference lives on the session (see the module docstring), never in the
# ``ClayDoc``. Empty until B8 landed; now the whole set the dead-tab gate
# above deliberately does not cover.
_SESSION_ONLY = [
    "clay_reference_add",
    "clay_reference_list",
    "clay_reference_get",
    "clay_reference_remove",
]

# The two tools that can start a document from nothing, and are therefore the
# two that a *dead* pin must not refuse: the refusal every other tool gives
# names these as the way out, and while they refused too that sentence was
# impossible to follow, which bricked the session for the rest of the
# connection. They mint rather than substitute -- what arrives is a new empty
# document, never one already open -- so the one-tab blast radius the list
# above gates is unchanged. See ``_tab``'s own comment.
_MINTS_A_TAB = [
    ("clay_add_primitive", {"generator": "box"}),
    ("clay_add_figure", {"key": sorted(presets.ASSEMBLIES)[0]}),
]


def test_every_tool_is_covered_by_the_dead_tab_and_session_only_lists() -> None:
    assert {n for n, _ in _NEEDS_A_TAB} | set(_SESSION_ONLY) | {
        n for n, _ in _MINTS_A_TAB
    } == set(agent_clay._HANDLERS)


# The four ``_SESSION_ONLY`` tools carry no ready-made args table the way
# ``_NEEDS_A_TAB``/``_MINTS_A_TAB`` do -- that list is only names, on purpose
# (see its own comment) -- so this is the one small table the test below adds,
# built from the same inline-base64 helpers the B8 reference tests already
# use rather than anything new. ``clay_reference_get`` names "ref1", added by
# the test itself before this table is walked; ``clay_reference_add`` and
# ``clay_reference_remove`` name a second reference of their own so the
# add/remove pair does not fight over the same slot.
_SESSION_ONLY_ARGS = {
    "clay_reference_add": {
        "name": "exhaustive_ref",
        "png_base64": base64.b64encode(_tiny_png()).decode("ascii"),
    },
    "clay_reference_list": {},
    "clay_reference_get": {"name": "ref1"},
    "clay_reference_remove": {"name": "exhaustive_ref"},
}


def test_every_tool_answers_with_structured_content_unless_its_reply_carries_a_picture(
    monkeypatch: pytest.MonkeyPatch, svc
) -> None:
    """The rule the module docstring states: a result that carries a picture
    (an image content block) never duplicates its header into
    ``structuredContent``, and every other successful reply's
    ``structuredContent`` is ``json.loads`` of its own first text block.
    Walked exhaustively over every entry in ``agent_clay._HANDLERS`` -- not a
    hand-kept subset -- so a tool added later is covered with nobody having
    to remember to extend a list for it.

    Reuses ``_NEEDS_A_TAB`` and ``_MINTS_A_TAB``'s own (name, args) pairs
    rather than inventing a second "call every tool with minimal arguments"
    table -- a second copy of that machinery is exactly the drift this file's
    own module docstring warns about. Those args are deliberately minimal
    (just enough to pass whatever a handler checks before it resolves a tab),
    so several handlers refuse outright with them -- ``clay_transform`` with
    no uid, say. A refusal is not a counterexample to the rule (see the
    module docstring's own refusal-envelope paragraph), so it is skipped
    rather than asserted on either way -- but the number of successes this
    walk actually produced is asserted with a hard floor, so a future
    regression that turned every reply into a refusal could not make this
    test pass having proven nothing.
    """
    ctx = _Ctx(svc=svc)
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)  # a real, open tab with one object on it
    _add_inline_reference(ctx, session, "ref1")  # so clay_reference_get can succeed too
    _install_fake_view(monkeypatch)  # so clay_render can succeed too, image and all

    covered = {n for n, _ in _NEEDS_A_TAB} | {n for n, _ in _MINTS_A_TAB} | set(_SESSION_ONLY)
    assert covered == set(agent_clay._HANDLERS)

    calls = list(_NEEDS_A_TAB) + list(_MINTS_A_TAB)
    calls += [(name, _SESSION_ONLY_ARGS[name]) for name in _SESSION_ONLY]

    successes = 0
    image_successes = 0
    for name, args in calls:
        result = agent_clay.call(ctx, session, name, args)
        if result["isError"]:
            continue
        successes += 1
        carries_image = any(block.get("type") == "image" for block in result["content"])
        if carries_image:
            image_successes += 1
            assert "structuredContent" not in result, name
        else:
            assert "structuredContent" in result, name
            assert result["structuredContent"] == json.loads(result["content"][0]["text"]), name

    # A floor, not a target: proves the walk actually exercised the rule on a
    # real mix of tools rather than skipping (almost) everything as
    # refusals. Both image-carrying tools reach a real success here:
    # clay_render because of the fake view installed above, and
    # clay_reference_get because "ref1" already exists by the time this walk
    # reaches it. Measured at 15 successes (2 of them image-carrying) out of
    # 25 calls on this tree; the bound below leaves headroom rather than
    # pinning that exact count, since a handler gaining one more required
    # argument tomorrow should not make this test start failing for an
    # unrelated reason.
    assert successes >= 12
    assert image_successes >= 2


@pytest.mark.parametrize("name,args", _MINTS_A_TAB, ids=[n for n, _ in _MINTS_A_TAB])
def test_a_session_whose_document_was_closed_can_start_another(name: str, args: dict) -> None:
    """The recovery the other tools' refusal names, actually reachable.

    Before ``_tab`` released a dead pin, this call took the same branch the
    refusal came from and answered with the identical sentence -- so an agent
    told to "call clay_add_primitive to start a new one" did exactly that and
    was refused again, with no way out short of reconnecting.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    closed = session.tab_uid
    state = clay_mode.ensure(ctx)
    assert state.close(closed)

    result = agent_clay.call(ctx, session, name, args)
    assert result["isError"] is False
    assert session.tab_uid and session.tab_uid != closed
    assert state.get(session.tab_uid) is not None


@pytest.mark.parametrize("name,args", _NEEDS_A_TAB, ids=[n for n, _ in _NEEDS_A_TAB])
def test_a_session_naming_a_tab_that_no_longer_exists_is_refused_for_every_tool(
    name: str, args: dict
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session(tab_uid="no-such-tab")
    result = agent_clay.call(ctx, session, name, args)
    assert result["isError"] is True


def test_a_tab_closed_mid_session_is_seen_as_gone_on_the_very_next_call() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    state = clay_mode.ensure(ctx)
    assert state.close(session.tab_uid)

    result = agent_clay.call(ctx, session, "clay_scene", {})
    assert result["isError"] is True


def test_a_session_with_no_document_never_falls_back_to_the_users_active_tab() -> None:
    ctx = _Ctx()
    user_tab = clay_mode.new_document(ctx)
    pane_clay_tools.add_primitive(ctx, user_tab.doc, "box")

    session = agent_clay.Session()
    result = agent_clay.call(ctx, session, "clay_scene", {})
    assert result["isError"] is True


def test_an_object_uid_in_the_users_tab_is_refused_and_that_document_is_untouched() -> None:
    ctx = _Ctx()
    user_tab = clay_mode.new_document(ctx)
    user_obj = pane_clay_tools.add_primitive(ctx, user_tab.doc, "box")
    before = serialize.wblk_bytes(user_tab.doc)

    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    assert session.tab_uid != user_tab.uid

    result = agent_clay.call(
        ctx, session, "clay_transform", {"uid": user_obj.uid, "translation": [1.0, 2.0, 3.0]}
    )
    assert result["isError"] is True

    after = serialize.wblk_bytes(user_tab.doc)
    assert after == before


def test_clay_add_primitive_mints_a_document_that_is_not_the_users_active_one() -> None:
    ctx = _Ctx()
    user_tab = clay_mode.new_document(ctx)

    session = agent_clay.Session()
    result = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    assert result["isError"] is False
    assert session.tab_uid not in ("", user_tab.uid)


# --- one tool call, one undo step ---------------------------------------------


def test_add_primitive_is_one_undo_step_and_undo_reverts_it_completely() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    before = _history_len(ctx, session)

    agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "cylinder"})
    assert _history_len(ctx, session) == before + 1

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert tab.doc.undo()
    assert len(tab.doc.objects) == 1


def test_set_params_is_one_undo_step_and_undo_reverts_it_completely() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "cylinder")
    before = _history_len(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_set_params", {"uid": uid, "params": {"radius": 2.0}}
    )
    assert result["isError"] is False
    assert _history_len(ctx, session) == before + 1

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert tab.doc.by_uid(uid).params["radius"] == pytest.approx(2.0)
    assert tab.doc.undo()
    assert tab.doc.by_uid(uid).params["radius"] == pytest.approx(0.5)  # cylinder's default


def test_material_is_one_undo_step_however_many_objects_it_paints() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid1 = _new_agent_tab(ctx, session, "box")
    uid2 = _payload(agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "cylinder"}))[
        "uid"
    ]
    before = _history_len(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_material", {"uids": [uid1, uid2], "color": [1.0, 0.0, 0.0]}
    )
    assert result["isError"] is False
    assert _history_len(ctx, session) == before + 1

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert tab.doc.undo()


def test_transform_is_one_undo_step_and_undo_reverts_it_completely() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    before = _history_len(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_transform", {"uid": uid, "translation": [1.0, 2.0, 3.0]}
    )
    assert result["isError"] is False
    assert _history_len(ctx, session) == before + 1

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert list(tab.doc.by_uid(uid).translation) == pytest.approx([1.0, 2.0, 3.0])
    assert tab.doc.undo()
    assert list(tab.doc.by_uid(uid).translation) == pytest.approx([0.0, 0.0, 0.0])


def test_boolean_is_one_undo_step_and_undo_reverts_it_completely() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid1 = _new_agent_tab(ctx, session, "box")
    added = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    uid2 = _payload(added)["uid"]
    before = _history_len(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_boolean", {"kind": "union", "uids": [uid1, uid2]}
    )
    assert result["isError"] is False, result
    assert _history_len(ctx, session) == before + 1

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert len(tab.doc.objects) == 1
    assert tab.doc.undo()
    assert len(tab.doc.objects) == 2


# --- clamping is reported, never silent ---------------------------------------


def test_set_params_clamps_and_reports_the_clamped_value_back() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "cylinder")

    result = agent_clay.call(
        ctx, session, "clay_set_params", {"uid": uid, "params": {"segments": 2}}
    )
    assert result["isError"] is False
    assert _payload(result)["params"]["segments"] == 3  # bp.MIN_SEGMENTS


# --- clay_material paints every face, not just the object's default slot -----


def test_material_repaints_every_face_of_the_mesh_not_only_the_objects_default_slot() -> None:
    """The trap the module docstring names: ``Obj.material`` is only the slot
    *future* faces are stamped with. What actually exports is the per-face
    ``mesh.material`` array, grouped by ``document.to_primitives``."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")

    result = agent_clay.call(
        ctx, session, "clay_material", {"uids": [uid], "color": [1.0, 0.0, 0.0]}
    )
    assert result["isError"] is False
    index = _payload(result)["index"]
    assert index != 0  # not the document's default palette slot

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    obj = tab.doc.by_uid(uid)
    prims = bd.to_primitives(obj, tab.doc.materials)
    # One group, at the new index: if only ``Obj.material`` had been set, the
    # per-face array would still read all-zeros and this would come back as
    # one group at the *old*, default material instead.
    assert len(prims) == 1
    assert prims[0].material.base_color_factor == (1.0, 0.0, 0.0, 1.0)


# --- a rebuild carries per-face material and shading, the same as the panel ---


def test_clay_set_params_keeps_per_face_material_through_a_rebuild() -> None:
    """The defect this closes: every generator funnels through
    ``primitives._mesh``, which stamps a fresh all-zero ``material`` array on
    every call, and ``_h_set_params`` used to rebuild with a bare
    ``shading.auto_smooth(bp.GENERATORS[obj.generator][1](**merged))`` -- no
    carry at all. There is no per-face paint tool over MCP yet (``clay_material``
    repaints every face of an object, not a selection within one -- see the
    module docstring's "everyday behaviour" claim), so face 2 is painted
    directly the way a face-mode paint op would leave it, then a same-face-
    count params edit is sent through the real tool and face 2's slot is
    checked. Fails today: it comes back 0.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    obj = tab.doc.by_uid(uid)
    painted = np.array(obj.mesh.material)
    painted[2] = 1
    tab.doc.set_mesh(uid, replace(obj.mesh, material=painted), keep_generator=True)

    result = agent_clay.call(
        ctx, session, "clay_set_params", {"uid": uid, "params": {"size": [2.0, 1.0, 1.0]}}
    )
    assert result["isError"] is False, result

    obj = tab.doc.by_uid(uid)
    assert int(obj.mesh.material[2]) == 1, "a size edit must not grey out a painted face"


def test_clay_set_params_keeps_per_face_shading_through_a_rebuild() -> None:
    """Same rebuild, for ``smooth``. A box reads flat under
    ``shading.auto_smooth`` on its own, so re-deriving from scratch (what
    ``_h_set_params`` did before this fix) happens to look right on an
    untouched box -- the defect only shows once a face's shading has been
    hand-picked *away* from what the angle rule would choose, which is
    forced here by smoothing every face of a box, something ``auto_smooth``
    itself would never produce. Fails today: a same-face-count rebuild
    re-derives instead of carrying, so the hand-picked flags come back false.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    obj = tab.doc.by_uid(uid)
    hand_picked = np.ones(len(obj.mesh.smooth), dtype="?")
    tab.doc.set_mesh(uid, replace(obj.mesh, smooth=hand_picked), keep_generator=True)

    result = agent_clay.call(
        ctx, session, "clay_set_params", {"uid": uid, "params": {"size": [2.0, 1.0, 1.0]}}
    )
    assert result["isError"] is False, result

    obj = tab.doc.by_uid(uid)
    assert obj.mesh.smooth.tolist() == hand_picked.tolist()


# --- refusals name their field where one is knowable --------------------------


def test_an_unknown_generator_value_is_refused_with_field_generator() -> None:
    result = agent_clay.call(
        _Ctx(), agent_clay.Session(), "clay_add_primitive", {"generator": "nope"}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "generator"


def test_an_unknown_param_key_is_refused_with_field_params() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "cylinder")

    result = agent_clay.call(
        ctx, session, "clay_set_params", {"uid": uid, "params": {"not_a_real_param": 1.0}}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "params"


def test_clay_transform_refuses_a_two_element_translation_and_the_document_still_describes_itself() -> None:  # noqa: E501
    """The headline regression. An unvalidated ``clay_transform`` used to cast
    each element of ``translation`` with no length check at all, commit the
    two-element result straight onto the object via ``set_transform``, and
    only fall over three calls later: ``clay_scene`` -> ``_scene_row`` ->
    ``clay_geom_ops.world_box`` -> ``viewer/math3d.py``'s ``compose`` does
    ``m[:3, 3] = t``, which raises trying to broadcast a length-2 array into
    a length-3 slot -- and ``agent_clay.call``'s blanket ``except ValueError``
    turns that into a refusal for *every* object in the document, not just
    the one that was moved. The only recovery was a blind ``clay_undo`` an
    agent had no reason to reach for, since the call that broke it had
    reported success. Refusing the malformed vector at the handler is what
    keeps the document able to describe itself afterwards.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")

    result = agent_clay.call(
        ctx, session, "clay_transform", {"uid": uid, "translation": [1.0, 2.0]}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "translation"

    scene = agent_clay.call(ctx, session, "clay_scene", {})
    assert scene["isError"] is False, scene


def test_clay_transform_refuses_a_non_finite_rotation_rather_than_poisoning_the_quaternion() -> None:  # noqa: E501
    """``rotation`` was only *accidentally* safe against a wrong-length list --
    ``_quat_from_euler_xyz``'s ``rx, ry, rz = (...)`` unpack raises on that --
    but nothing caught a non-finite element: ``math.radians(float("nan"))``
    passes straight through and the resulting quaternion is nan in every
    component, committed to the object exactly like a good one.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    agent_clay.call(ctx, session, "clay_transform", {"uid": uid, "rotation": [0.0, 90.0, 0.0]})

    result = agent_clay.call(
        ctx, session, "clay_transform", {"uid": uid, "rotation": [float("nan"), 0.0, 0.0]}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "rotation"

    scene = agent_clay.call(ctx, session, "clay_scene", {})
    row = _payload(scene)["objects"][0]
    assert row["rotation"] == pytest.approx([0.0, 90.0, 0.0], abs=1e-3)


def test_clay_set_params_refuses_a_non_finite_value_rather_than_baking_it_into_positions() -> None:
    """A NaN in a generator's own params used to sail past ``bp.clamp_params``
    (which only clamps the keys it knows a floor or a relational limit for)
    straight into the generator function and out the other side as vertex
    positions nothing downstream checks -- the same unvalidated-number hole
    ``clay_transform`` had, one call over.
    """
    import numpy as np

    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")

    result = agent_clay.call(
        ctx,
        session,
        "clay_set_params",
        {"uid": uid, "params": {"size": [1.0, float("inf"), 1.0]}},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "params"

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    obj = tab.doc.by_uid(uid)
    assert np.isfinite(obj.mesh.positions).all()


def test_clay_material_refuses_a_non_finite_metallic() -> None:
    """``float("nan")`` passed the old ``isinstance(c, int | float)`` colour
    check just as readily as a real number -- NaN *is* a float -- and the
    bare ``float(args.get("metallic", 0.0))`` had no check at all, so a NaN
    metallic landed straight in the palette.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    scene_before = agent_clay.call(ctx, session, "clay_scene", {})
    palette_before = len(_payload(scene_before)["materials"])

    result = agent_clay.call(
        ctx,
        session,
        "clay_material",
        {"uids": [uid], "color": [1.0, 0.0, 0.0], "metallic": float("nan")},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "metallic"

    scene_after = agent_clay.call(ctx, session, "clay_scene", {})
    assert len(_payload(scene_after)["materials"]) == palette_before


# --- call() never raises -------------------------------------------------------


def test_call_never_raises_on_an_unknown_tool_name() -> None:
    result = agent_clay.call(_Ctx(), agent_clay.Session(), "clay_not_a_real_tool", {})
    assert result["isError"] is True


def test_call_never_raises_on_missing_required_arguments() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)

    result = agent_clay.call(ctx, session, "clay_transform", {})  # no uid
    assert result["isError"] is True


def test_call_never_raises_on_a_wrong_typed_argument() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_material", {"uids": "not-a-list-of-ints", "color": [1.0, 1.0, 1.0]}
    )
    assert result["isError"] is True


# --- end to end: clay_export mints a real, finished asset ---------------------


def test_clay_export_end_to_end_mints_a_finished_model_row_with_its_source_sidecar(svc) -> None:
    """Mirrors ``tests/test_clay_service.py``'s own shape for ``import_mesh``."""
    from warlock.service import files as svc_files

    ctx = _Ctx(svc=svc)
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")

    result = agent_clay.call(ctx, session, "clay_export", {})
    assert result["isError"] is False, result
    job_id = _payload(result)["job_id"]

    job = svc.store.get(job_id)
    assert job["status"] == "done"
    assert job["stage"] == "model"

    job_dir = svc.job_dir(job_id)
    assert (job_dir / "source.glb").exists()
    assert (job_dir / "model.glb").exists()
    assert svc_files.clay_source_status(svc, job_id) == {"exists": True}


# ==============================================================================
# B1 -- the Euler inverse and the instructions text
# ==============================================================================


def test_the_instructions_state_metres_y_up_ground_at_zero_and_the_half_height_rule() -> None:
    body = agent_clay.instructions()
    assert "metre" in body
    assert "Y is up" in body
    assert "y=0" in body
    assert "h/2" in body


def test_the_instructions_name_every_generator_the_registry_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(bp.GENERATORS, "thirteenth_shape", ({"size": 1.0}, bp.box))
    assert "thirteenth_shape" in agent_clay.instructions()


def test_euler_xyz_round_trips_through_the_quaternion_on_a_grid_of_angles() -> None:
    """The claim is about the *rotation*, not the three numbers -- at gimbal
    lock a whole family of (rx, rz) pairs describes one orientation, so ±90
    degree yaw is included precisely to hold the round trip to that weaker,
    correct claim rather than to an equality on angles that legitimately
    differ there."""
    import numpy as np

    angles = (-180.0, -90.0, -45.0, 0.0, 30.0, 45.0, 90.0, 135.0, 180.0)
    for rx in (0.0, 30.0, 90.0):
        for ry in angles:
            for rz in (0.0, 45.0, -90.0):
                q = agent_clay._quat_from_euler_xyz((rx, ry, rz))
                back = agent_clay._euler_xyz_from_quat(q)
                q2 = agent_clay._quat_from_euler_xyz(back)
                assert np.allclose(m3.quat_to_mat4(q), m3.quat_to_mat4(q2), atol=1e-6)


# ==============================================================================
# B2 -- a readable scene
# ==============================================================================


def test_clay_scene_reports_each_objects_translation_rotation_scale_size_and_center() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    agent_clay.call(
        ctx,
        session,
        "clay_transform",
        {"uid": uid, "translation": [1.0, 2.0, 3.0], "rotation": [0.0, 90.0, 0.0]},
    )
    result = agent_clay.call(ctx, session, "clay_scene", {})
    row = _payload(result)["objects"][0]
    assert row["translation"] == pytest.approx([1.0, 2.0, 3.0])
    assert row["rotation"] == pytest.approx([0.0, 90.0, 0.0], abs=1e-3)
    assert row["scale"] == pytest.approx([1.0, 1.0, 1.0])
    assert row["size"] is not None
    assert row["center"] is not None
    assert row["verts"] > 0


def test_clay_scene_keeps_every_key_it_already_had() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    result = agent_clay.call(ctx, session, "clay_scene", {})
    row = _payload(result)["objects"][0]
    old_keys = {"uid", "name", "visible", "generator", "params", "faces", "material", "bbox"}
    assert old_keys <= set(row)


def test_clay_scene_reports_the_documents_own_bounds_and_palette() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    result = agent_clay.call(ctx, session, "clay_scene", {})
    payload = _payload(result)
    assert payload["object_count"] == 1
    assert payload["bounds"] is not None
    assert payload["materials"][0]["index"] == 0


def test_clay_scene_bounds_ignore_a_hidden_object() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    tab.doc.set_props(uid, visible=False)
    result = agent_clay.call(ctx, session, "clay_scene", {})
    assert _payload(result)["bounds"] is None


# ==============================================================================
# B3 -- place in one call
# ==============================================================================


def test_add_primitive_with_params_translation_rotation_scale_and_name_is_one_undo_step() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx,
        session,
        "clay_add_primitive",
        {
            "generator": "cylinder",
            "params": {"radius": 2.0},
            "translation": [1.0, 2.0, 3.0],
            "rotation": [0.0, 90.0, 0.0],
            "scale": [1.0, 2.0, 1.0],
            "name": "Pillar",
            "material": 0,
        },
    )
    assert result["isError"] is False, result
    row = _payload(result)
    assert row["name"] == "Pillar"
    assert row["params"]["radius"] == pytest.approx(2.0)
    assert row["translation"] == pytest.approx([1.0, 2.0, 3.0])
    assert row["material"] == 0

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert len(tab.doc.history.history()) == 1
    assert tab.doc.undo()
    assert len(tab.doc.objects) == 0


def test_add_primitive_with_a_taken_name_is_refused_with_field_name_and_places_nothing() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")  # named "Box"
    result = agent_clay.call(
        ctx, session, "clay_add_primitive", {"generator": "cylinder", "name": "Box"}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "name"
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert len(tab.doc.objects) == 1


def test_add_primitive_with_a_material_index_past_the_palette_is_refused_and_places_nothing(
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx, session, "clay_add_primitive", {"generator": "box", "material": 5}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "material"
    scene = agent_clay.call(ctx, session, "clay_scene", {})
    assert _payload(scene)["objects"] == []


def test_add_primitive_returns_the_row_clay_scene_would_have_shown() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    added = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    added_row = _payload(added)
    scene = agent_clay.call(ctx, session, "clay_scene", {})
    scene_row = _payload(scene)["objects"][0]
    assert added_row == scene_row


def test_add_figure_yaw_rotates_part_positions_about_the_group_origin_not_each_part_in_place(
) -> None:
    plain = agent_clay.call(_Ctx(), agent_clay.Session(), "clay_add_figure", {"key": "humanoid"})
    assert plain["isError"] is False, plain
    plain_rows = {r["name"]: r for r in _payload(plain)["objects"]}

    yawed = agent_clay.call(
        _Ctx(), agent_clay.Session(), "clay_add_figure", {"key": "humanoid", "yaw": 90.0}
    )
    assert yawed["isError"] is False, yawed
    yawed_rows = {r["name"]: r for r in _payload(yawed)["objects"]}

    for name, row in plain_rows.items():
        x, y, z = row["translation"]
        expected = (z, y, -x)  # R_y(+90 deg): x' = z, y' = y, z' = -x
        got = yawed_rows[name]["translation"]
        assert got == pytest.approx(expected, abs=1e-3), name


def test_add_figure_scale_multiplies_both_the_offsets_and_the_parts() -> None:
    plain = agent_clay.call(_Ctx(), agent_clay.Session(), "clay_add_figure", {"key": "humanoid"})
    plain_rows = {r["name"]: r for r in _payload(plain)["objects"]}

    scaled = agent_clay.call(
        _Ctx(), agent_clay.Session(), "clay_add_figure", {"key": "humanoid", "scale": 2.0}
    )
    scaled_rows = {r["name"]: r for r in _payload(scaled)["objects"]}

    for name, row in plain_rows.items():
        got = scaled_rows[name]
        assert got["translation"] == pytest.approx([v * 2.0 for v in row["translation"]], abs=1e-3)
        assert got["scale"] == pytest.approx([v * 2.0 for v in row["scale"]], abs=1e-3)


def test_add_primitive_with_no_optional_arguments_behaves_exactly_as_it_did() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    assert result["isError"] is False
    row = _payload(result)
    assert row["generator"] == "box"
    assert row["translation"] == pytest.approx([0.0, 0.0, 0.0])

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert len(tab.doc.history.history()) == 1
    assert tab.doc.undo()
    assert len(tab.doc.objects) == 0


# ==============================================================================
# B4 -- history and object verbs
# ==============================================================================


def test_clay_undo_reverses_the_last_tool_call_and_clay_redo_restores_it() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)

    result = agent_clay.call(ctx, session, "clay_undo", {})
    assert result["isError"] is False
    assert _payload(result)["moved"] == 1
    assert len(tab.doc.objects) == 0

    result = agent_clay.call(ctx, session, "clay_redo", {})
    assert _payload(result)["moved"] == 1
    assert len(tab.doc.objects) == 1


def test_clay_undo_over_asking_moves_what_it_can_and_says_how_far_it_got() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")  # exactly one step

    result = agent_clay.call(ctx, session, "clay_undo", {"steps": 10})
    payload = _payload(result)
    assert payload["moved"] == 1
    assert payload["can_undo"] is False


def test_clay_undo_pushes_no_new_step_of_its_own() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "cylinder"})
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    before = len(tab.doc.history)

    agent_clay.call(ctx, session, "clay_undo", {})
    assert len(tab.doc.history) == before - 1


def test_clay_delete_still_works_in_an_element_mode() -> None:
    """A guard for a capability this change adds, not a regression test for
    one that broke: ``clay_select`` and ``clay_boolean`` now refuse in an
    element mode because they *write* object uids into ``doc.selection``,
    which can manufacture "selected with nothing selected inside it" the
    moment element mode is reachable at all. ``clay_delete`` is deliberately
    not given the same refusal -- it works from the uids given and only ever
    *removes* from ``selection``, which cannot manufacture that state -- so
    this extends the original face-mode delete test to also assert the
    derived-selection invariant holds afterwards.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    tab.doc.element_mode = "face"  # nothing is selected in this mode

    result = agent_clay.call(ctx, session, "clay_delete", {"uids": [uid]})
    assert result["isError"] is False, result
    assert len(tab.doc.objects) == 0
    assert all(u in tab.doc.element_sel for u in tab.doc.selection)


def test_clay_delete_of_several_objects_is_one_undo_step() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid1 = _new_agent_tab(ctx, session, "box")
    uid2 = _payload(agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "cylinder"}))[
        "uid"
    ]
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    before = len(tab.doc.history)

    result = agent_clay.call(ctx, session, "clay_delete", {"uids": [uid1, uid2]})
    assert result["isError"] is False, result
    assert len(tab.doc.history) == before + 1
    assert tab.doc.undo()
    assert len(tab.doc.objects) == 2


def test_clay_delete_forgets_the_manifold_cache_entry_of_what_it_removed() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    state = clay_mode.ensure(ctx)
    state.manifold[uid] = (object(), [])

    result = agent_clay.call(ctx, session, "clay_delete", {"uids": [uid]})
    assert result["isError"] is False, result
    assert uid not in state.manifold


def test_clay_rename_refuses_a_name_another_object_wears_with_field_name() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid1 = _new_agent_tab(ctx, session, "box")
    uid2 = _payload(agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "cylinder"}))[
        "uid"
    ]
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    other_name = tab.doc.by_uid(uid1).name

    result = agent_clay.call(ctx, session, "clay_rename", {"uid": uid2, "name": other_name})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "name"


def test_clay_rename_refuses_an_empty_name() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")

    result = agent_clay.call(ctx, session, "clay_rename", {"uid": uid, "name": "   "})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "name"


# ==============================================================================
# B5 -- clay_batch
# ==============================================================================


def test_clay_batch_runs_its_calls_in_order_as_one_undo_step() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    before = len(tab.doc.history)

    calls = [
        {"name": "clay_add_primitive", "arguments": {"generator": "cylinder"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "cone"}},
    ]
    result = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["completed"] == 2
    assert payload["stopped_at"] is None
    assert len(payload["results"]) == 2
    assert len(tab.doc.objects) == 3
    assert len(tab.doc.history) == before + 1
    assert tab.doc.undo()
    assert len(tab.doc.objects) == 1


def test_clay_batch_stops_at_the_first_refusal_keeps_the_prefix_and_reports_where_it_stopped(
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    before = len(tab.doc.objects)

    calls = [
        {"name": "clay_add_primitive", "arguments": {"generator": "cylinder"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "nope"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "cone"}},
    ]
    result = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})
    assert result["isError"] is True
    payload = _payload(result)
    assert payload["stopped_at"] == 1
    assert payload["completed"] == 1
    assert len(payload["results"]) == 2  # the cylinder, then the refusal itself
    assert len(tab.doc.objects) == before + 1


def test_clay_batch_refuses_more_than_batch_max_calls_and_runs_none_of_them() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    before = len(tab.doc.objects)

    calls = [{"name": "clay_add_primitive", "arguments": {"generator": "box"}}] * (
        agent_clay.BATCH_MAX + 1
    )
    result = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})
    assert result["isError"] is True
    assert len(tab.doc.objects) == before


def test_clay_batch_cannot_contain_a_batch() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")

    result = agent_clay.call(
        ctx,
        session,
        "clay_batch",
        {"calls": [{"name": "clay_batch", "arguments": {"calls": []}}]},
    )
    assert result["isError"] is True


def test_clay_batch_that_starts_with_an_add_mints_the_sessions_first_document() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    assert session.tab_uid == ""

    calls = [{"name": "clay_add_primitive", "arguments": {"generator": "box"}}]
    result = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})
    assert result["isError"] is False, result
    assert session.tab_uid != ""


# ==============================================================================
# B6 -- clay_op stops reaching the user's viewport
# ==============================================================================


class _FakeClayView:
    def __init__(self) -> None:
        self.frame_calls = 0

    def frame_selection(self, doc: Any) -> None:
        del doc
        self.frame_calls += 1


def test_clay_op_frame_never_touches_the_users_viewport() -> None:
    ctx = _Ctx()
    fake_view = _FakeClayView()
    ctx.clay_view = fake_view
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    agent_clay.call(ctx, session, "clay_select", {"uids": [uid]})

    result = agent_clay.call(ctx, session, "clay_op", {"name": "frame"})
    assert result["isError"] is False, result
    assert fake_view.frame_calls == 0


def test_clay_op_returns_refusal_messages_to_the_agent_instead_of_toasting_the_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    agent_clay.call(ctx, session, "clay_select", {"uids": [uid]})

    def fake_run(proxy: Any, doc: Any, op: Any, **params: Any) -> bool:
        del doc, op, params
        proxy.toast("could not do the thing", "error")
        return False

    monkeypatch.setattr(clay_ops, "run", fake_run)
    result = agent_clay.call(ctx, session, "clay_op", {"name": "duplicate"})
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["ran"] is False
    assert "could not do the thing" in payload["messages"]
    assert ctx.toasts == []  # the real ctx never saw it


def test_clay_op_still_prunes_the_manifold_cache_through_the_proxy() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid1 = _new_agent_tab(ctx, session, "box")
    uid2 = _payload(agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"}))[
        "uid"
    ]
    state = clay_mode.ensure(ctx)
    state.manifold[uid2] = (object(), [])
    agent_clay.call(ctx, session, "clay_select", {"uids": [uid1, uid2]})

    result = agent_clay.call(ctx, session, "clay_op", {"name": "join"})
    assert result["isError"] is False, result
    assert uid2 not in state.manifold


# ==============================================================================
# B7 -- render: several views, a free angle, a grid, a focus, a budget
# ==============================================================================


def test_clay_render_refuses_cleanly_with_no_moderngl_context() -> None:
    """See the module docstring: this pins the refusal shape only. Rendering
    real pixels needs ``ctx.viewer.ctx``, a live moderngl context this
    headless suite does not stand up, and belongs in a GL-backed test file."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)

    result = agent_clay.call(ctx, session, "clay_render", {})
    assert result["isError"] is True


def test_a_valid_render_request_reaches_gl_and_fails_there_rather_than_at_validation() -> None:
    """The ordering claim made real: no ``field`` in the refusal means this
    was never one of the named validation refusals -- it is the generic GL
    failure ``_view_for`` raises with no ``ctx.viewer`` to build a viewport
    from."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)

    result = agent_clay.call(ctx, session, "clay_render", {"view": "front"})
    assert result["isError"] is True
    assert "structuredContent" not in result


def test_clay_render_refuses_more_pixels_than_the_frame_budget_before_touching_gl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    fake = _install_fake_view(monkeypatch)

    views = [{"yaw": 0.0, "pitch": 0.0}, {"yaw": 45.0, "pitch": 0.0}]
    result = agent_clay.call(ctx, session, "clay_render", {"size": 2048, "views": views})
    assert result["isError"] is True
    assert fake.calls == []


def test_clay_render_refuses_both_view_and_views() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_render", {"view": "front", "views": ["back"]}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "views"


def test_clay_render_refuses_a_pitch_past_the_pole() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_render", {"views": [{"yaw": 0.0, "pitch": 95.0}]}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "views"


def test_clay_render_refuses_a_focus_uid_the_document_does_not_have() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)

    result = agent_clay.call(ctx, session, "clay_render", {"focus": [999999]})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "focus"


def test_clay_render_folds_a_single_view_into_the_views_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    fake = _install_fake_view(monkeypatch)

    result = agent_clay.call(ctx, session, "clay_render", {"view": "front"})
    assert result["isError"] is False, result
    assert len(fake.calls) == 1
    assert fake.calls[0]["view"] == "front"
    header = _payload(result)
    assert header["views"] == ["front"]
    # one text block, one image block
    assert len(result["content"]) == 2
    assert result["content"][1]["type"] == "image"


# ==============================================================================
# B8 -- references on the session
# ==============================================================================


def test_reference_add_from_a_library_job_reads_its_input_png(svc) -> None:
    ctx = _Ctx(svc=svc)
    session = agent_clay.Session()
    job_id = svc.store.create("image", None, {}, stage="reference", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(_tiny_png())

    result = agent_clay.call(
        ctx, session, "clay_reference_add", {"name": "ref1", "job_id": job_id}
    )
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["source"] == f"job:{job_id}:input.png"


def test_reference_add_from_a_library_job_names_all_four_candidates_when_none_is_ready(
    svc,
) -> None:
    ctx = _Ctx(svc=svc)
    session = agent_clay.Session()
    job_id = svc.store.create("image", None, {}, stage="reference", status="queued")
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)

    result = agent_clay.call(
        ctx, session, "clay_reference_add", {"name": "ref1", "job_id": job_id}
    )
    assert result["isError"] is True
    body = result["content"][0]["text"]
    for candidate in ("input.png", "ref.png", "reference.png", "thumb.png"):
        assert candidate in body


def test_reference_add_from_inline_base64_is_listed_and_returned_as_an_image_block() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    added = _add_inline_reference(ctx, session, "ref1", view="front")
    payload = _payload(added)
    assert payload["width"] > 0
    assert payload["height"] > 0
    assert payload["source"] == "inline"

    listing = agent_clay.call(ctx, session, "clay_reference_list", {})
    assert [r["name"] for r in _payload(listing)["references"]] == ["ref1"]

    got = agent_clay.call(ctx, session, "clay_reference_get", {"name": "ref1"})
    assert got["isError"] is False
    assert got["content"][1]["type"] == "image"


def test_reference_add_refuses_a_ninth_reference_and_names_the_cap() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    for i in range(agent_clay.MAX_REFERENCES):
        _add_inline_reference(ctx, session, f"ref{i}")

    result = _add_inline_reference_raw(ctx, session, "one_too_many")
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "name"
    assert str(agent_clay.MAX_REFERENCES) in result["content"][0]["text"]


def test_reference_add_under_a_name_already_taken_replaces_it_without_spending_a_slot() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    for i in range(agent_clay.MAX_REFERENCES):
        _add_inline_reference(ctx, session, f"ref{i}")

    result = _add_inline_reference(ctx, session, "ref0", view="top")
    payload = _payload(result)
    assert payload["replaced"] is True
    assert len(session.references) == agent_clay.MAX_REFERENCES


def test_reference_add_pushes_no_undo_step() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    before = len(tab.doc.history)

    _add_inline_reference(ctx, session, "ref1")
    assert len(tab.doc.history) == before


def test_references_are_per_session_so_a_second_session_lists_none() -> None:
    ctx = _Ctx()
    session1 = agent_clay.Session()
    _add_inline_reference(ctx, session1, "ref1")

    session2 = agent_clay.Session()
    listing = agent_clay.call(ctx, session2, "clay_reference_list", {})
    assert _payload(listing)["references"] == []


def test_clay_render_compare_with_an_unknown_reference_is_refused_with_field_compare_before_any_gl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    fake = _install_fake_view(monkeypatch)

    result = agent_clay.call(ctx, session, "clay_render", {"compare": "nope"})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "compare"
    assert fake.calls == []


def test_clay_render_compare_uses_the_references_view_when_none_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    _add_inline_reference(ctx, session, "ref1", view="left")
    fake = _install_fake_view(monkeypatch)

    result = agent_clay.call(ctx, session, "clay_render", {"compare": "ref1"})
    assert result["isError"] is False, result
    assert fake.calls[0]["view"] == "left"
    payload = _payload(result)
    assert payload["view"] == "left"
    assert payload["reference"] == "ref1"
    assert payload["mode"] == "beside"
    assert result["content"][1]["type"] == "image"


def test_clay_render_compare_refuses_more_than_one_view() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    _add_inline_reference(ctx, session, "ref1")

    result = agent_clay.call(
        ctx, session, "clay_render", {"compare": "ref1", "views": ["front", "back"]}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "views"


# ==============================================================================
# B9 -- the mesh, taken apart: element mode, explicit index, query, the read
# ==============================================================================


def test_clay_op_inset_refuses_until_the_agent_switches_to_face_mode_and_then_runs() -> None:
    """The headline capability this change adds. Before it, nothing in
    ``agent_clay.py`` ever called ``doc.set_element_mode`` or
    ``doc.set_element_sel``, so ``clay_op`` refused ``inset``/``bevel``/
    ``extrude`` unconditionally, forever -- an agent could place and boolean
    shapes but could never touch a single face. Fails today at the second
    half: the refusal text matches ``clay_ops.reason_for``, but there is no
    way to reach the run.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)

    refused = agent_clay.call(ctx, session, "clay_op", {"name": "inset"})
    assert refused["isError"] is True
    op = clay_ops.get("inset")
    assert refused["content"][0]["text"] == clay_ops.reason_for(op, tab.doc)

    mode_result = agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    assert mode_result["isError"] is False, mode_result
    sel_result = agent_clay.call(
        ctx, session, "clay_select_elements", {"uid": uid, "faces": [0]}
    )
    assert sel_result["isError"] is False, sel_result

    result = agent_clay.call(ctx, session, "clay_op", {"name": "inset"})
    assert result["isError"] is False, result
    assert _payload(result)["ran"] is True


@pytest.mark.parametrize(
    "op_name,mode,select_kwargs",
    [
        ("bevel", "edge", {"edges": [[0, 1]]}),
        ("extrude", "face", {"faces": [0]}),
    ],
)
def test_clay_op_bevel_and_extrude_become_reachable_the_same_way(
    op_name: str, mode: str, select_kwargs: dict
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")

    refused = agent_clay.call(ctx, session, "clay_op", {"name": op_name})
    assert refused["isError"] is True

    agent_clay.call(ctx, session, "clay_element_mode", {"mode": mode})
    sel = agent_clay.call(
        ctx, session, "clay_select_elements", {"uid": uid, **select_kwargs}
    )
    assert sel["isError"] is False, sel

    result = agent_clay.call(ctx, session, "clay_op", {"name": op_name})
    assert result["isError"] is False, result
    assert _payload(result)["ran"] is True


def test_clay_select_elements_replaces_adds_and_subtracts() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")  # 6 faces: 0..5
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})

    r1 = agent_clay.call(ctx, session, "clay_select_elements", {"uid": uid, "faces": [0, 1]})
    assert _payload(r1)["selected"]["faces"] == 2

    r2 = agent_clay.call(
        ctx, session, "clay_select_elements", {"uid": uid, "faces": [2], "how": "add"}
    )
    assert _payload(r2)["selected"]["faces"] == 3

    r3 = agent_clay.call(
        ctx, session, "clay_select_elements", {"uid": uid, "faces": [0], "how": "subtract"}
    )
    assert _payload(r3)["selected"]["faces"] == 2

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert sorted(tab.doc.element_sel_of(uid).faces.tolist()) == [1, 2]


def test_clay_select_elements_refuses_a_face_index_the_mesh_does_not_have_and_selects_nothing() -> (
    None
):
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")  # 6 faces: 0..5
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})

    result = agent_clay.call(ctx, session, "clay_select_elements", {"uid": uid, "faces": [99]})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "faces"

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert tab.doc.element_sel_of(uid).faces.tolist() == []


def test_clay_select_elements_refuses_a_vertex_pair_that_is_not_an_edge() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "edge"})

    # face 0 is [0, 1, 2, 3] -- 0 and 2 are a diagonal of that quad, not an edge.
    result = agent_clay.call(
        ctx, session, "clay_select_elements", {"uid": uid, "edges": [[0, 2]]}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "edges"
    assert "[0, 2]" in result["content"][0]["text"]


def test_clay_select_by_loop_selects_the_ring_of_edges_a_human_alt_click_would() -> None:
    """Assert equality with ``select.edge_loop`` called directly, so the tool
    cannot drift from the verb it wraps."""
    from warlock.studio.clay import select as clay_select_mod

    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    mesh = tab.doc.by_uid(uid).mesh
    expected = clay_select_mod.edge_loop(mesh, (0, 1))

    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "edge"})
    result = agent_clay.call(
        ctx, session, "clay_select_by", {"uid": uid, "query": "loop", "edge": [0, 1]}
    )
    assert result["isError"] is False, result

    got = tab.doc.element_sel_of(uid).edges
    assert got.tolist() == expected.tolist()


def test_clay_select_by_normal_takes_the_upward_faces_of_a_rotated_object_in_world_space() -> (
    None
):
    from warlock.studio.clay import select as clay_select_mod

    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    agent_clay.call(ctx, session, "clay_transform", {"uid": uid, "rotation": [90.0, 0.0, 0.0]})
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    obj = tab.doc.by_uid(uid)

    # The naive (uncorrected) answer, straight in local space -- if the tool
    # forgot to convert 'direction' through ``local_direction`` it would find
    # this instead, which after a 90 degree rotation is not what faces world
    # up any more.
    naive = set(clay_select_mod.faces_by_normal(obj.mesh, (0.0, 1.0, 0.0)).tolist())
    local_dir = agent_clay.clay_geom_ops.local_direction(obj, (0.0, 1.0, 0.0))
    expected = set(clay_select_mod.faces_by_normal(obj.mesh, local_dir).tolist())
    assert expected != naive, "the rotation has to actually matter for this test to prove anything"

    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    result = agent_clay.call(
        ctx,
        session,
        "clay_select_by",
        {"uid": uid, "query": "normal", "direction": [0.0, 1.0, 0.0]},
    )
    assert result["isError"] is False, result
    got = set(tab.doc.element_sel_of(uid).faces.tolist())
    assert got == expected


def test_clay_select_by_refuses_a_query_the_current_mode_cannot_answer() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)  # still object mode

    result = agent_clay.call(
        ctx, session, "clay_select_by", {"uid": uid, "query": "material", "slot": 0}
    )
    assert result["isError"] is True
    assert result["content"][0]["text"] == clay_ops._in_mode_reason("face")(tab.doc)


def test_clay_diagnose_can_select_the_finding_it_reports() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    obj = tab.doc.by_uid(uid)
    original_vert_count = len(obj.mesh.positions)
    # An unreferenced vertex, appended after every real one -- the cheapest
    # defect to manufacture by hand: the mesh's topology (starts/loops) does
    # not reference it, so ``clay_diagnose`` reports it as "unused".
    positions = np.concatenate([obj.mesh.positions, np.zeros((1, 3), dtype="f4")])
    tab.doc.set_mesh(uid, replace(obj.mesh, positions=positions), keep_generator=True)

    diag = agent_clay.call(ctx, session, "clay_diagnose", {"uid": uid})
    assert diag["isError"] is False, diag
    findings = _payload(diag)["objects"][0]["findings"]
    assert any(f["kind"] == "unused" for f in findings)

    result = agent_clay.call(
        ctx, session, "clay_diagnose", {"select": {"uid": uid, "kind": "unused"}}
    )
    assert result["isError"] is False, result
    assert tab.doc.element_mode == "vertex"
    assert tab.doc.element_sel_of(uid).verts.tolist() == [original_vert_count]


def test_an_element_selection_reports_a_stamp_that_changes_when_an_op_replaces_the_mesh() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    r1 = agent_clay.call(ctx, session, "clay_select_elements", {"uid": uid, "faces": [0]})
    stamp1 = _payload(r1)["stamp"]

    op_result = agent_clay.call(ctx, session, "clay_op", {"name": "extrude"})
    assert op_result["isError"] is False, op_result
    changed = _payload(op_result)["changed"]
    assert changed and changed[0]["uid"] == uid
    assert changed[0]["stamp"] != stamp1


def test_clay_select_elements_with_a_stale_expect_stamp_is_refused_and_changes_nothing() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    r1 = agent_clay.call(ctx, session, "clay_select_elements", {"uid": uid, "faces": [0]})
    stamp1 = _payload(r1)["stamp"]
    agent_clay.call(ctx, session, "clay_op", {"name": "extrude"})  # replaces the mesh

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    before = tab.doc.element_sel_of(uid)

    result = agent_clay.call(
        ctx,
        session,
        "clay_select_elements",
        {"uid": uid, "faces": [0], "expect_stamp": stamp1},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "expect_stamp"
    assert tab.doc.element_sel_of(uid).same_as(before)


@pytest.mark.parametrize(
    "name,needs_face_mode,make_args",
    [
        ("clay_element_mode", False, lambda uid: {"mode": "face"}),
        ("clay_select_elements", False, lambda uid: {"uid": uid, "mode": "face", "faces": [0]}),
        (
            "clay_select_by",
            # clay_select_by has no 'mode' of its own -- unlike
            # clay_select_elements, it only checks the mode already in
            # effect (see its own refusal test) -- so this case switches
            # mode first, through clay_element_mode, and includes that call
            # in what "no undo step" is checked against too.
            True,
            lambda uid: {"uid": uid, "query": "material", "slot": 0},
        ),
        ("clay_select", False, lambda uid: {"uids": [uid]}),
    ],
    ids=["clay_element_mode", "clay_select_elements", "clay_select_by", "clay_select"],
)
def test_the_selection_tools_push_no_undo_step(
    name: str, needs_face_mode: bool, make_args: Any
) -> None:
    """Sibling of ``test_reference_add_pushes_no_undo_step`` -- see the module
    docstring's undo enumeration. A capability this change adds rather than a
    regression: before it, nothing in this module ever called
    ``set_element_mode`` or ``set_element_sel``, so there was no selection
    tool to make this claim about at all.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    before = len(tab.doc.history)

    if needs_face_mode:
        mode_result = agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
        assert mode_result["isError"] is False, mode_result
        assert len(tab.doc.history) == before

    result = agent_clay.call(ctx, session, name, make_args(uid))
    assert result["isError"] is False, result
    assert len(tab.doc.history) == before


def test_clay_op_extrude_returns_the_caps_it_selected_so_the_next_call_needs_no_round_trip() -> (
    None
):
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    agent_clay.call(ctx, session, "clay_select_elements", {"uid": uid, "faces": [0]})

    result = agent_clay.call(ctx, session, "clay_op", {"name": "extrude"})
    assert result["isError"] is False, result
    changed = _payload(result)["changed"]
    assert changed and changed[0]["uid"] == uid
    assert changed[0]["selected"]["faces"] > 0

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert tab.doc.element_sel_of(uid).faces.tolist() != []

    # No re-selection in between: the caps ``set_mesh(select=...)`` handed
    # back are exactly what the very next op call needs.
    inset_result = agent_clay.call(ctx, session, "clay_op", {"name": "inset"})
    assert inset_result["isError"] is False, inset_result


def test_clay_op_reports_which_objects_meshes_changed_and_which_did_not() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid1 = _new_agent_tab(ctx, session, "box")
    uid2 = _payload(
        agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    )["uid"]
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    agent_clay.call(ctx, session, "clay_select_elements", {"uid": uid1, "faces": [0]})

    result = agent_clay.call(ctx, session, "clay_op", {"name": "extrude"})
    assert result["isError"] is False, result
    changed_uids = {row["uid"] for row in _payload(result)["changed"]}
    assert changed_uids == {uid1}
    assert uid2 not in changed_uids


def test_clay_op_never_returns_raw_element_indices() -> None:
    """The context bound the module docstring names, made executable: an
    op's own result is a diff read off the mesh -- counts, a stamp, whether
    it pushed -- never the element indices themselves. ``clay_elements``
    exists, paged, for the rarer moment an agent has to reason about which
    ones."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    agent_clay.call(ctx, session, "clay_select_elements", {"uid": uid, "faces": [0]})

    result = agent_clay.call(ctx, session, "clay_op", {"name": "extrude"})
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["changed"], "the test needs at least one changed row to check"
    for row in payload["changed"]:
        assert isinstance(row["faces"], int)
        assert isinstance(row["verts"], int)
        assert set(row["selected"]) == {"verts", "edges", "faces"}
        for value in row["selected"].values():
            assert isinstance(value, int)


def test_clay_elements_pages_a_large_selection_and_reports_the_total() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "cylinder")
    agent_clay.call(ctx, session, "clay_set_params", {"uid": uid, "params": {"segments": 64}})
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    agent_clay.call(ctx, session, "clay_op", {"name": "select-all"})

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    total_faces = len(tab.doc.element_sel_of(uid).faces)
    assert total_faces > 10, "the fixture needs to actually be a large selection"

    result = agent_clay.call(
        ctx, session, "clay_elements", {"uid": uid, "kind": "face", "limit": 5, "offset": 3}
    )
    assert result["isError"] is False, result
    row = _payload(result)["objects"][0]
    assert row["total"] == total_faces
    assert row["offset"] == 3
    assert len(row["indices"]) == 5
    assert row["indices"] == sorted(tab.doc.element_sel_of(uid).faces.tolist())[3:8]


def test_the_instructions_name_the_call_timeout_the_host_actually_uses() -> None:
    from warlock.studio import agent_host

    assert str(int(agent_host.CALL_TIMEOUT)) in agent_clay.instructions()


def test_the_instructions_tell_an_agent_which_timed_out_calls_are_safe_to_retry() -> None:
    text = agent_clay.instructions()

    # A dropped call (never started) changed nothing, so retrying is safe.
    assert "dropped" in text
    assert "safe to send the same call again" in text
    # A started call keeps running and wants a fresh clay_scene, not a retry.
    assert "re-read clay_scene" in text
    # The old claim this change overturns must not still be here.
    assert "still completes" not in text


def test_the_instructions_tell_an_agent_a_started_call_can_now_be_recovered() -> None:
    from warlock.studio import agent_host

    text = agent_clay.instructions()

    # Resending the identical call is a replay, not a second run of it.
    assert "replayed rather than run a second time" in text
    # warlock_status is the other way to ask, named by the constant it
    # actually publishes under -- never a hand-typed copy of that string.
    assert agent_host.STATUS_TOOL in text
    # This only recovers a call whose answer never arrived -- stated
    # plainly, not left for the agent to infer.
    assert "two identical calls that both got answered stay two calls" in text


# --- guards for a capability that did not previously exist --------------------
#
# Before this change an agent could never leave object mode, so
# ``clay_select`` and ``clay_boolean`` writing object uids straight into
# ``doc.selection`` was harmless -- the element mode that write could
# contradict was unreachable. These are not regression tests for something
# that broke; they pin a refusal this change had to add the moment element
# mode became reachable at all.


def test_clay_select_refuses_object_uids_while_the_document_is_in_an_element_mode() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    before = set(tab.doc.selection)

    result = agent_clay.call(ctx, session, "clay_select", {"uids": [uid]})
    assert result["isError"] is True
    assert "clay_element_mode" in result["content"][0]["text"]
    assert tab.doc.selection == before


def test_clay_boolean_refuses_in_an_element_mode_rather_than_breaking_the_derived_selection_invariant() -> (  # noqa: E501
    None
):
    ctx = _Ctx()
    session = agent_clay.Session()
    uid1 = _new_agent_tab(ctx, session, "box")
    uid2 = _payload(
        agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    )["uid"]
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})

    result = agent_clay.call(
        ctx, session, "clay_boolean", {"kind": "union", "uids": [uid1, uid2]}
    )
    assert result["isError"] is True
    assert "clay_element_mode" in result["content"][0]["text"]

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert len(tab.doc.objects) == 2


# ==============================================================================
# Structured results -- every tool answers its JSON payload twice
# ==============================================================================
#
# See the module docstring's structured-results claim. `_json` is what nearly
# every tool here answers through; a reply that carries a picture is the
# exclusion, stated as a rule rather than a list of names.


def test_every_tool_answers_with_its_json_payload_as_structured_content_too() -> None:
    """The claim, driven for a representative handful
    (``clay_scene``, ``clay_add_primitive`` and a tool with a small payload,
    ``clay_rename``) through the real call path, ``.get`` rather than
    subscripting so a HEAD with no ``structuredContent`` at all fails this
    assertion cleanly instead of raising ``KeyError``."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")

    scene = agent_clay.call(ctx, session, "clay_scene", {})
    assert scene["isError"] is False, scene
    assert scene.get("structuredContent") == _payload(scene)

    added = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "cylinder"})
    assert added["isError"] is False, added
    assert added.get("structuredContent") == _payload(added)

    renamed = agent_clay.call(ctx, session, "clay_rename", {"uid": uid, "name": "the box"})
    assert renamed["isError"] is False, renamed
    assert renamed.get("structuredContent") == _payload(renamed)


def test_a_render_does_not_duplicate_its_header_into_structured_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deliberate exclusion, driven through a real ``clay_render`` call
    via ``_install_fake_view`` (the file already has a way to fake the GL
    this test cannot reach -- see the module docstring), plus a check at the
    source of truth: ``_h_render`` never calls ``_json`` at all, which is
    what makes the exclusion structural rather than an accident of what its
    header happens to contain."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    _install_fake_view(monkeypatch)

    result = agent_clay.call(ctx, session, "clay_render", {"view": "front"})
    assert result["isError"] is False, result
    assert "structuredContent" not in result

    import inspect

    source = inspect.getsource(agent_clay._h_render)
    assert "_json(" not in source


def test_the_three_declared_output_schemas_describe_what_those_tools_actually_return() -> None:
    """The test that catches a schema drifting from ``_scene_row`` (or from
    ``_h_scene``/``_h_diagnose``'s own payload): every key a real call's
    ``structuredContent`` actually carries must appear in that tool's own
    declared ``outputSchema['properties']``."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    tools = {t.name: t for t in agent_clay.tools()}

    scene_schema = getattr(tools["clay_scene"], "output_schema", None)
    assert scene_schema is not None
    scene_result = agent_clay.call(ctx, session, "clay_scene", {})
    scene_structured = scene_result.get("structuredContent") or {}
    assert scene_structured, "clay_scene answered with no structuredContent at all"
    # Exact, not a subset: every key ``_h_scene`` builds is unconditional, so
    # equality catches drift in *both* directions -- a key the handler gained
    # and the schema does not describe, and a key the schema claims that the
    # handler does not actually answer with.
    assert set(scene_structured) == set(scene_schema["properties"])

    add_schema = getattr(tools["clay_add_primitive"], "output_schema", None)
    assert add_schema is not None
    add_result = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    add_structured = add_result.get("structuredContent") or {}
    assert add_structured, "clay_add_primitive answered with no structuredContent at all"
    # Exact for the same reason: this tool answers with ``_scene_row``, whose
    # every key is unconditional, so a row key added without touching the
    # shared schema helper fails here.
    assert set(add_structured) == set(add_schema["properties"])

    diag_schema = getattr(tools["clay_diagnose"], "output_schema", None)
    assert diag_schema is not None
    diag_result = agent_clay.call(ctx, session, "clay_diagnose", {})
    diag_structured = diag_result.get("structuredContent") or {}
    assert diag_structured, "clay_diagnose answered with no structuredContent at all"
    # A subset here, deliberately, and the one of the three where it has to
    # be: ``selected`` appears only when the call asked for a finding to be
    # selected, which needs a mesh that actually has one -- that half is
    # already pinned by
    # ``test_diagnose_hands_back_a_selection_the_agent_can_act_on``.
    assert set(diag_structured) <= set(diag_schema["properties"])
    assert "objects" in diag_schema["properties"]


def test_no_declared_output_schema_demands_required_keys_because_a_refusal_shares_the_envelope() -> (  # noqa: E501
    None
):
    """None of the three declared schemas names a ``required`` list or sets
    ``additionalProperties: false`` -- proven alongside the reason itself: a
    refusal from one of these same tools really does put ``field`` in
    ``structuredContent`` and nothing else, which a ``required`` list on the
    success shape would make non-conforming."""
    tools = {t.name: t for t in agent_clay.tools()}
    for name in ("clay_scene", "clay_add_primitive", "clay_diagnose"):
        schema = getattr(tools[name], "output_schema", None)
        assert schema is not None
        assert "required" not in schema
        assert schema.get("additionalProperties") is not False

    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    refusal = agent_clay.call(
        ctx, session, "clay_add_primitive", {"generator": "not-a-real-generator"}
    )
    assert refusal["isError"] is True
    assert refusal["structuredContent"] == {"field": "generator"}


def test_the_object_row_schema_is_shared_by_the_scene_and_the_primitive_tools() -> None:
    """``clay_scene``'s ``objects`` items and ``clay_add_primitive``'s own
    declared schema use the same row shape -- the one shared helper, not a
    hand-written second copy of it."""
    tools = {t.name: t for t in agent_clay.tools()}

    add_schema = getattr(tools["clay_add_primitive"], "output_schema", None)
    assert add_schema is not None
    assert add_schema == agent_clay._object_row_output_schema()

    scene_schema = getattr(tools["clay_scene"], "output_schema", None)
    assert scene_schema is not None
    assert scene_schema["properties"]["objects"]["items"] == agent_clay._object_row_output_schema()
