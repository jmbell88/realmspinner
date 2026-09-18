"""What Clay's agent tool surface (``studio/modes/clay/agent/dispatch.py``) promises to hold.

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
Five tools -- ``clay_scene``, ``clay_add_primitive``, ``clay_add_mesh``,
``clay_diagnose`` and ``clay_analyze`` -- also declare an ``outputSchema``
describing that shape, and none declares
``required`` or ``additionalProperties: false``, because a refusal shares
the same result envelope and its ``structuredContent`` is whatever
``fail()``'s ``**extra`` was given -- ``field`` where one is knowable, always
``changed``, and ``recovery``/``uids``/``op`` where they are.

**The refusal-recovery claim.** Every refusal's ``structuredContent`` carries
``changed`` -- whether *this session's own document* was modified before the
refusal fired, defaulted to ``False`` in :func:`agent_clay.fail` rather than
at each of its ~100 call sites -- proven empirically, not merely asserted
present, by ``test_every_refusal_says_whether_the_document_moved``: it walks
every handler, and for each refusal it captures the document's own history
length, ``dirty`` flag and object count before and after the call, and checks
a ``changed: false`` refusal really left all three untouched. ``recovery`` is
a closed vocabulary (:data:`agent_clay.RECOVERY`) naming what a client should
try next, pinned bidirectionally by
``test_every_recovery_a_refusal_names_is_in_the_vocabulary`` the same way the
derivation gate above is -- every value a refusal actually produces is a
vocabulary member, and every vocabulary member is findable somewhere in
``src/warlock/``. A refusal whose recovery is not known carries no
``recovery`` key at all, which is a real, distinct answer rather than an
omission.

**The unknown-argument claim.** ``call()`` -- the one door every tool call
passes through -- refuses an argument name a tool's own schema does not
declare in ``properties``, before the handler ever runs. The derived lookup
(:func:`agent_clay._allowed_argument_names`, memoised) is checked against a
fresh ``tools()`` call by ``test_the_allowed_argument_names_are_the_schemas_own``,
which is also what pins the memoisation safe: a tool's *property names* are
schema literals, never derived from a registry the way an *enum's values*
are, so caching this projection cannot go stale the way caching the whole
catalogue would. ``test_a_misspelled_argument_is_refused_with_the_name_it_
probably_meant`` is the exact measured incident that motivated this --
``clay_add_primitive`` given ``translaton`` used to place a box at the origin
and report success; it now refuses, names the key, suggests ``translation``,
and places nothing. Several bad keys are all named in one refusal rather than
costing a caller two round trips (``test_several_misspelled_arguments_are_
all_named_at_once``), an unknown *tool* is still refused before its
arguments are ever looked at
(``test_an_unknown_tool_is_still_refused_before_its_arguments_are_looked_at``),
and ``clay_batch``'s own nested calls go through the identical door -- a bad
argument at its third entry stops the batch there and keeps the first two,
exactly as its documented contract for any other refusal already promises
(``test_a_batch_refuses_at_the_entry_with_a_bad_argument_and_keeps_what_ran``).
Every *other* constraint a schema declares -- ``minimum``, ``enum``,
``minItems`` and the rest -- is pinned separately, by
``tests/test_agent_schemas.py``, which discovers them from the schemas
themselves rather than from a hand-written list.

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

from warlock.kernels.geom3d import math3d as m3
from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import presets, serialize
from warlock.kernels.mesh import primitives as bp
from warlock.studio.modes.clay import mode as clay_mode
from warlock.studio.modes.clay import ops as clay_ops
from warlock.studio.modes.clay.agent import dispatch as agent_clay
from warlock.studio.modes.clay.ui.panes import tools as pane_clay_tools

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

    def __init__(self, png: bytes | None = None, id_rows: list[tuple] | None = None) -> None:
        self.png = png or _tiny_png()
        self.calls: list[dict[str, Any]] = []
        self.id_calls: list[dict[str, Any]] = []
        # A per-uid (hex, px) row an object_id test can shape; empty by
        # default, since most callers of this fake never touch render_ids.
        self.id_rows = id_rows if id_rows is not None else []

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
        shading: str = "unlit",
    ) -> bytes:
        del doc, frame
        self.calls.append(
            {
                "size": size, "view": view, "angles": angles, "bounds": bounds,
                "grid": grid, "shading": shading,
            }
        )
        return self.png

    def render_ids(
        self,
        doc: Any,
        *,
        size: int,
        view: str | None = None,
        angles: Any = None,
        bounds: Any = None,
        frame: bool = True,
    ) -> tuple[bytes, list[tuple]]:
        del doc, frame
        self.id_calls.append({"size": size, "view": view, "angles": angles, "bounds": bounds})
        return self.png, self.id_rows


def _install_fake_view(
    monkeypatch: pytest.MonkeyPatch, png: bytes | None = None, id_rows: list[tuple] | None = None
) -> _FakeView:
    fake = _FakeView(png, id_rows)
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


def test_the_placement_ops_reach_the_clay_op_enum() -> None:
    """align/distribute/drop-to-ground/snap-to-grid are ``clay_ops`` rows like
    any other, so the bidirectional gate above already covers them -- this
    names the four directly so an agent-visible regression (one dropped from
    the registry, or renamed) fails here rather than only as a shrinking set
    the test above would not explain."""
    tools = {t.name: t for t in agent_clay.tools()}
    enum = set(tools["clay_op"].schema["properties"]["name"]["enum"])
    assert {"align", "distribute", "drop-to-ground", "snap-to-grid"} <= enum


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


def test_clay_add_figure_description_names_every_part_of_every_figure() -> None:
    """Run A of the Clay-assistant fine-tune (2026-09-12) refused 15 calls
    with ``no object named '...'``, nine of them a creatures-family guess at
    a generated figure's own part names (``hound_Beak``, ``t_Shank.R``,
    ``s_Tail 01``) -- nothing in ``clay_add_figure``'s description told the
    model what :func:`presets.build` actually calls a figure's parts.

    Fails against the unfixed code: the old description ends at
    ``name_prefix``'s collision rule and names no part of any figure.
    """
    tools = {t.name: t for t in agent_clay.tools()}
    description = tools["clay_add_figure"].description
    for key in sorted(presets.ASSEMBLIES):
        for part in presets.build(key):
            assert part.name in description, f"{key}'s part {part.name!r} is missing"


# --- an array-of-arrays param, lathe's own shape -----------------------------


def test_a_profile_param_survives_the_whole_agent_door() -> None:
    """``lathe``'s ``profile`` -- an array of ``[radius, y]`` pairs -- makes
    the same round trip a flat vector like a box's ``size`` already does:
    placed through ``clay_add_primitive``, read back unchanged through
    ``clay_scene``, and changed through ``clay_set_params``.

    Fails today: the old ``_validate_number_or_vec``'s flat-array branch
    calls ``float(v)`` on each *row* of the profile, and a row is itself a
    list -- ``float([0.0, -0.5])`` raises ``TypeError``, caught by that
    branch's own ``except`` and turned into a clean but wrong refusal,
    ``"params must be a number or an array of numbers."``, before a single
    vertex is placed.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    profile = [[0.0, -0.5], [0.3, 0.0], [0.3, 0.5]]
    result = agent_clay.call(
        ctx,
        session,
        "clay_add_primitive",
        {"generator": "lathe", "params": {"profile": profile}},
    )
    assert result["isError"] is False, result
    row = _payload(result)
    uid = row["uid"]
    assert row["params"]["profile"] == profile

    scene = agent_clay.call(ctx, session, "clay_scene", {})
    scene_row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid)
    assert scene_row["params"]["profile"] == profile

    new_profile = [[0.0, -0.5], [0.4, 0.0], [0.1, 0.5]]
    changed = agent_clay.call(
        ctx, session, "clay_set_params", {"uid": uid, "params": {"profile": new_profile}}
    )
    assert changed["isError"] is False, changed
    scene = agent_clay.call(ctx, session, "clay_scene", {})
    scene_row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid)
    assert scene_row["params"]["profile"] == new_profile


def test_a_tubes_path_survives_the_whole_agent_door() -> None:
    """``tube``'s ``path`` -- an array of ``[x, y, z]`` triples -- is the
    second real generator parameter of this shape, after ``lathe``'s
    ``profile``, and the claim this module's own docstring makes about that
    widening (``_validate_number_or_vec``/``_params_value_schema`` went
    generic for *any* array-of-arrays param, not one hand-listed for
    ``lathe``) is that ``tube`` needed no further edit here to reach the
    same door. Placed through ``clay_add_primitive``, read back unchanged
    through ``clay_scene``, and changed through ``clay_set_params`` -- the
    same three steps
    :func:`test_a_profile_param_survives_the_whole_agent_door` proves for
    ``lathe``.
    """
    # Already centred on its own bounding box on all three axes -- the same
    # care ``test_a_profile_param_survives_the_whole_agent_door``'s profile
    # takes -- so a round trip through ``_clamp_path`` changes nothing here
    # and the claim is about the door, not about the clamp.
    ctx = _Ctx()
    session = agent_clay.Session()
    path = [[-0.3, -0.1, -0.05], [0.0, 0.05, 0.02], [0.3, 0.1, 0.05]]
    result = agent_clay.call(
        ctx,
        session,
        "clay_add_primitive",
        {"generator": "tube", "params": {"path": path}},
    )
    assert result["isError"] is False, result
    row = _payload(result)
    uid = row["uid"]
    assert row["params"]["path"] == path

    scene = agent_clay.call(ctx, session, "clay_scene", {})
    scene_row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid)
    assert scene_row["params"]["path"] == path

    new_path = [[-0.3, -0.1, -0.05], [0.1, -0.05, 0.03], [0.3, 0.1, 0.05]]
    changed = agent_clay.call(
        ctx, session, "clay_set_params", {"uid": uid, "params": {"path": new_path}}
    )
    assert changed["isError"] is False, changed
    scene = agent_clay.call(ctx, session, "clay_scene", {})
    scene_row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid)
    assert scene_row["params"]["path"] == new_path


def test_an_array_of_arrays_param_survives_the_agent_door_for_any_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claim is about the door, not about ``lathe`` specifically -- a
    fake generator whose only parameter is an array of arrays, monkeypatched
    into the registry exactly the way
    ``test_a_thirteenth_generator_reaches_the_agent_surface_with_no_edit_here``
    does it, reaches ``clay_add_primitive``, ``clay_scene`` and
    ``clay_set_params`` with nothing here naming ``lathe`` at all.
    """

    def fake_builder(rows: Any = ((1.0, 2.0), (3.0, 4.0))) -> bp.Mesh:
        del rows
        return bp.box(size=(1.0, 1.0, 1.0))

    monkeypatch.setitem(
        bp.GENERATORS, "fake_rows", ({"rows": ((1.0, 2.0), (3.0, 4.0))}, fake_builder)
    )
    ctx = _Ctx()
    session = agent_clay.Session()
    rows = [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    result = agent_clay.call(
        ctx, session, "clay_add_primitive", {"generator": "fake_rows", "params": {"rows": rows}}
    )
    assert result["isError"] is False, result
    row = _payload(result)
    uid = row["uid"]
    assert row["params"]["rows"] == rows

    scene = agent_clay.call(ctx, session, "clay_scene", {})
    scene_row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid)
    assert scene_row["params"]["rows"] == rows

    new_rows = [[1.0, 1.0]]
    changed = agent_clay.call(
        ctx, session, "clay_set_params", {"uid": uid, "params": {"rows": new_rows}}
    )
    assert changed["isError"] is False, changed


def test_every_element_mode_is_a_clay_element_mode_enum_option_and_vice_versa() -> None:
    from warlock.kernels.mesh import elements as clay_elements

    tools = {t.name: t for t in agent_clay.tools()}
    enum = set(tools["clay_element_mode"].schema["properties"]["mode"]["enum"])
    assert enum == set(clay_elements.MODES)


def test_every_query_name_is_a_clay_select_by_enum_option_and_vice_versa() -> None:
    from warlock.kernels.mesh import select as clay_select_mod

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
    from warlock.kernels.mesh import elements as clay_elements
    from warlock.kernels.mesh import select as clay_select_mod

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
    from warlock.kernels.mesh import select as clay_select_mod

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


# --- the unknown-argument claim ------------------------------------------------


def test_a_misspelled_argument_is_refused_with_the_name_it_probably_meant() -> None:
    """The exact measured case from the module docstring's own incident:
    ``clay_add_primitive`` given ``translaton`` (not ``translation``) used to
    place a box at the origin and report success."""
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx, session, "clay_add_primitive", {"generator": "box", "translaton": [0, 9, 0]}
    )
    assert result["isError"] is True
    message = result["content"][0]["text"]
    assert "translaton" in message
    assert "translation" in message  # the did-you-mean
    structured = result["structuredContent"]
    assert structured["field"] == "translaton"
    assert structured["recovery"] == "fix_arguments"
    assert structured["changed"] is False
    # The part that matters most: nothing was placed. A refused call runs
    # nothing, so this session was never even minted a document to place a
    # box on -- the strongest available proof, stronger than "no box exists"
    # alone would be.
    assert session.tab_uid == ""


def test_several_misspelled_arguments_are_all_named_at_once() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx,
        session,
        "clay_add_primitive",
        {"generator": "box", "prams": {"size": 4}, "translaton": [0, 9, 0]},
    )
    assert result["isError"] is True
    message = result["content"][0]["text"]
    assert "prams" in message
    assert "translaton" in message
    assert session.tab_uid == ""  # neither round trip is needed to learn this


def test_an_unknown_tool_is_still_refused_before_its_arguments_are_looked_at() -> None:
    """The ordering claim: an unknown *tool* name is refused first, never
    reinterpreted as an unknown-argument refusal for some other tool."""
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx, session, "clay_add_primitve", {"generator": "box", "bogus": 1}
    )
    assert result["isError"] is True
    assert "no such tool" in result["content"][0]["text"]
    assert "field" not in result.get("structuredContent", {})


def test_the_allowed_argument_names_are_the_schemas_own() -> None:
    """The memoisation-safety check :func:`agent_clay._allowed_argument_names`'s
    own docstring promises: the cached lookup agrees with a fresh ``tools()``
    call, and its key set is exactly ``_HANDLERS``."""
    fresh = {t.name: frozenset(t.schema.get("properties", {})) for t in agent_clay.tools()}
    cached = agent_clay._allowed_argument_names()
    assert cached == fresh
    assert set(cached) == set(agent_clay._HANDLERS)


def test_a_batch_refuses_at_the_entry_with_a_bad_argument_and_keeps_what_ran() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx,
        session,
        "clay_batch",
        {
            "calls": [
                {"name": "clay_add_primitive", "arguments": {"generator": "box"}},
                {"name": "clay_add_primitive", "arguments": {"generator": "box"}},
                {
                    "name": "clay_add_primitive",
                    "arguments": {"generator": "box", "translaton": [0, 9, 0]},
                },
            ]
        },
    )
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["completed"] == 2
    assert payload["stopped_at"] == 2
    assert payload["changed"] is True
    assert payload["results"][2]["isError"] is True
    assert "translaton" in payload["results"][2]["content"][0]["text"]
    # The prefix genuinely ran and is kept: two objects exist on the document
    # this batch itself minted (its first call was the creator).
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert len(tab.doc.objects) == 2


# --- the blast-radius claim ---------------------------------------------------

# One call per tool, with just enough arguments to pass whatever this handler
# validates *before* it resolves the session's tab -- ``clay_add_primitive``,
# ``clay_add_figure`` and ``clay_batch`` check their own arguments first,
# every other handler here calls ``_tab`` before touching ``args`` at all.
# See ``studio/modes/clay/agent/dispatch.py``'s source for that ordering; getting it backwards here
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
    ("clay_analyze", {}),
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

# A hand-built tetrahedron -- four triangles, closed -- reused wherever a
# test just needs *some* valid clay_add_mesh call rather than one that
# exercises a specific validation rule.
_TETRA_MESH_ARGS = {
    "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "faces": [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
}

# The three tools that can start a document from nothing, and are therefore
# the three that a *dead* pin must not refuse: the refusal every other tool
# gives names these as the way out, and while they refused too that sentence
# was impossible to follow, which bricked the session for the rest of the
# connection. They mint rather than substitute -- what arrives is a new empty
# document, never one already open -- so the one-tab blast radius the list
# above gates is unchanged. See ``_tab``'s own comment.
_MINTS_A_TAB = [
    ("clay_add_primitive", {"generator": "box"}),
    ("clay_add_figure", {"key": sorted(presets.ASSEMBLIES)[0]}),
    ("clay_add_mesh", _TETRA_MESH_ARGS),
]

# clay_program mints a document the identical way (a non-dry-run call with
# no session tab yet calls ``_tab(..., create=True)`` unconditionally, the
# same as the three above) but is deliberately *not* one of
# ``agent_clay.MINTS_A_DOCUMENT`` -- it can never be a clay_batch entry at
# all (``BATCH_EXCLUDED``), so it has no business in the sentence that names
# what a batch may open with. Kept as its own list rather than folded into
# ``_MINTS_A_TAB`` so ``test_clay_batchs_published_description_names_every_
# tool_that_can_start_one``'s own ``MINTS_A_DOCUMENT == {n for n, _ in
# _MINTS_A_TAB}`` pin stays exactly what it already proves, with nothing
# here for it to accidentally start disagreeing with.
_ALSO_MINTS_A_TAB = [
    ("clay_program", {"steps": [{"add": {"generator": "box"}}]}),
]


def test_every_tool_is_covered_by_the_dead_tab_and_session_only_lists() -> None:
    assert {n for n, _ in _NEEDS_A_TAB} | set(_SESSION_ONLY) | {
        n for n, _ in _MINTS_A_TAB
    } | {n for n, _ in _ALSO_MINTS_A_TAB} == set(agent_clay._HANDLERS)


def test_clay_batchs_published_description_names_every_tool_that_can_start_one() -> None:
    """The catalogue sentence an agent reads before its first call must agree
    with the membership check that actually runs.

    It did not: ``clay_add_mesh`` joined ``_h_batch``'s own tuple when that
    tool landed and the description was left saying "the first call must be
    clay_add_primitive or clay_add_figure", so the published contract refused
    in prose a batch the code ran without complaint. Both now read
    ``agent_clay.MINTS_A_DOCUMENT``, and this pins that tuple against the
    same ``_MINTS_A_TAB`` table the exhaustiveness test above already keeps
    honest -- so a fourth document-starting tool cannot be added without
    this file's own classification and the agent-visible sentence both
    moving with it.
    """
    assert set(agent_clay.MINTS_A_DOCUMENT) == {n for n, _ in _MINTS_A_TAB}
    description = next(
        tool.description for tool in agent_clay.tools() if tool.name == "clay_batch"
    )
    for name in agent_clay.MINTS_A_DOCUMENT:
        assert name in description, name


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

    covered = (
        {n for n, _ in _NEEDS_A_TAB}
        | {n for n, _ in _MINTS_A_TAB}
        | {n for n, _ in _ALSO_MINTS_A_TAB}
        | set(_SESSION_ONLY)
    )
    assert covered == set(agent_clay._HANDLERS)

    calls = list(_NEEDS_A_TAB) + list(_MINTS_A_TAB) + list(_ALSO_MINTS_A_TAB)
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
    # reaches it. Measured at 17 successes (2 of them image-carrying) out of
    # 28 calls on this tree; the bound below leaves headroom rather than
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


def test_a_refused_boolean_leaves_the_selection_it_found() -> None:
    """A boolean that names too few visible objects refuses -- and must leave
    the person's own selection exactly as it was.

    It used to write ``doc.select(wanted)`` *before* the "at least two"
    count check, because re-reading the selection back was how it got the
    document's own object order for picking the survivor. So a refused call
    still overwrote whatever the human at the keyboard had selected, from a
    call that changed nothing else and reported a refusal. The order is now
    derived by walking ``doc.objects`` instead, which writes nothing -- and
    that is what lets the refusal report ``changed: false`` honestly rather
    than owning up to a mutation it had no reason to make.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    uid1 = _new_agent_tab(ctx, session, "box")
    added = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    uid2 = _payload(added)["uid"]

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    tab.doc.select([uid1, uid2])
    before = set(tab.doc.selection)
    history_before = _history_len(ctx, session)

    # One visible object named, where a boolean needs two.
    result = agent_clay.call(ctx, session, "clay_boolean", {"kind": "union", "uids": [uid1]})

    assert result["isError"] is True
    structured = result.get("structuredContent") or {}
    assert structured.get("changed") is False
    assert set(tab.doc.selection) == before  # the selection it found, untouched
    assert _history_len(ctx, session) == history_before


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


# --- clay_set_params gains a plural form (tranche 5) --------------------------


def test_clay_set_params_uids_retunes_every_named_object_at_once() -> None:
    """The reviewer's ask, verbatim: "named parts that stay coherent -- make
    the wheels larger and they all change together" is one ``clay_set_params``
    call against several uids, not one call per wheel."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid1 = _new_agent_tab(ctx, session, "cylinder")
    uid2 = _new_agent_tab(ctx, session, "cylinder")
    uid3 = _new_agent_tab(ctx, session, "cylinder")

    result = agent_clay.call(
        ctx, session, "clay_set_params", {"uids": [uid1, uid2, uid3], "params": {"segments": 16}}
    )
    assert result["isError"] is False, result

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    for uid in (uid1, uid2, uid3):
        assert tab.doc.by_uid(uid).params["segments"] == 16


def test_clay_set_params_uids_is_one_undo_step_and_one_undo_restores_every_object() -> None:
    """Retuning six wheels is one call *and* one undo step -- a single
    ``clay_undo`` must put every one of them back, not just the last."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid1 = _new_agent_tab(ctx, session, "cylinder")
    uid2 = _new_agent_tab(ctx, session, "cylinder")
    uid3 = _new_agent_tab(ctx, session, "cylinder")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    originals = {uid: tab.doc.by_uid(uid).params["segments"] for uid in (uid1, uid2, uid3)}
    before = _history_len(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_set_params", {"uids": [uid1, uid2, uid3], "params": {"segments": 16}}
    )
    assert result["isError"] is False, result
    assert _history_len(ctx, session) == before + 1, "three rebuilds must fold into one step"
    for uid in (uid1, uid2, uid3):
        assert tab.doc.by_uid(uid).params["segments"] == 16

    undo = agent_clay.call(ctx, session, "clay_undo", {"steps": 1})
    assert undo["isError"] is False, undo
    for uid in (uid1, uid2, uid3):
        assert tab.doc.by_uid(uid).params["segments"] == originals[uid]


def test_clay_set_params_singular_uid_is_still_one_step_with_no_plural_label() -> None:
    """The fold is only worth its keep once there is more than one step to
    fold -- a lone ``uid`` already pushes exactly one step on its own
    (``document.set_generator_params`` folds its own params-edit/mesh-edit
    pair), so wrapping it in the same ``mark()``/``collapse_since()``/
    ``_label_top`` dance the plural form uses would relabel that step "Set
    Params" for a call whose behaviour never changed -- exactly the
    over-relabelling ``_label_top``'s own docstring warns against."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "cylinder")
    before = _history_len(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_set_params", {"uid": uid, "params": {"segments": 16}}
    )
    assert result["isError"] is False, result
    assert _history_len(ctx, session) == before + 1

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    label, is_done = tab.doc.history.history()[-1]
    assert is_done
    assert label != "Set Params"


def test_clay_set_params_bad_key_among_several_refuses_the_whole_call() -> None:
    """A ``radius`` handed to a box among two cylinders must refuse the whole
    call and name the offending uid and generator -- rebuilding four
    cylinders and leaving one box untouched would be a caller having to
    guess which of its six wheels did not actually change."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid_box = _new_agent_tab(ctx, session, "box")
    uid_cyl = _new_agent_tab(ctx, session, "cylinder")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    before = {u: tab.doc.by_uid(u).mesh for u in (uid_box, uid_cyl)}
    history_before = _history_len(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_set_params", {"uids": [uid_box, uid_cyl], "params": {"radius": 2.0}}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "params"
    # ``uids`` (a list), never a singular ``uid``: ``agent_clay.fail``'s own
    # docstring and the agent paragraph in ``dev/INVARIANTS.md`` both
    # enumerate the four extras a refusal may carry, and this is the one
    # that means "these objects" -- the same key ``_resolve_uids`` already
    # answers a missing uid with.
    assert result["structuredContent"]["uids"] == [uid_box]
    message = result["content"][0]["text"]
    assert str(uid_box) in message
    assert "box" in message

    # All-or-nothing: not just the box (whose 'radius' is illegal) but the
    # cylinder too -- an eligible object must come out of a refused call
    # exactly as it went in. ``Mesh`` is ``eq=False`` and immutable, so
    # identity is the honest check, the same one ``_h_op`` already uses to
    # tell which objects an op actually touched.
    for u in (uid_box, uid_cyl):
        assert tab.doc.by_uid(u).mesh is before[u]
    assert _history_len(ctx, session) == history_before


def test_clay_set_params_frozen_object_among_several_refuses_and_names_it() -> None:
    """A mesh placed through ``clay_add_mesh`` has no generator -- see
    ``test_clay_add_mesh_placed_object_has_no_generator_and_clay_set_params_refuses_it``
    for the singular case -- and the same freeze must stop the whole plural
    call, not just skip the frozen object."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid_normal = _new_agent_tab(ctx, session, "box")
    added = agent_clay.call(ctx, session, "clay_add_mesh", _TETRA_MESH_ARGS)
    assert added["isError"] is False, added
    uid_frozen = _payload(added)["uid"]
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    before = {u: tab.doc.by_uid(u).mesh for u in (uid_normal, uid_frozen)}
    history_before = _history_len(ctx, session)

    result = agent_clay.call(
        ctx,
        session,
        "clay_set_params",
        {"uids": [uid_normal, uid_frozen], "params": {"size": [2.0, 1.0, 1.0]}},
    )
    assert result["isError"] is True
    # ``uids``, not ``uid``: the refusal points at the argument the caller
    # actually sent, the rule ``_resolve_uid``'s own docstring already holds
    # itself to -- telling a plural call to fix its ``uid`` would name an
    # argument that is not in the call.
    assert result["structuredContent"]["field"] == "uids"
    assert result["structuredContent"]["uids"] == [uid_frozen]
    assert str(uid_frozen) in result["content"][0]["text"]

    for u in (uid_normal, uid_frozen):
        assert tab.doc.by_uid(u).mesh is before[u]
    assert _history_len(ctx, session) == history_before


def test_clay_set_params_both_uid_and_uids_refuses() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")

    result = agent_clay.call(
        ctx,
        session,
        "clay_set_params",
        {"uid": uid, "uids": [uid], "params": {"size": [2.0, 1.0, 1.0]}},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "uid"
    assert "exactly one" in result["content"][0]["text"]


def test_clay_set_params_neither_uid_nor_uids_refuses() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")

    result = agent_clay.call(
        ctx, session, "clay_set_params", {"params": {"size": [2.0, 1.0, 1.0]}}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "uid"
    assert "exactly one" in result["content"][0]["text"]


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


def test_a_bad_params_value_names_which_param_is_bad() -> None:
    """A lathe has two numeric-shaped params -- ``profile`` (array of
    arrays) and ``segments`` (a plain number) -- so a NaN in one beside a
    good value in the other is exactly the case that used to come back as
    the bare ``"params must be finite numbers."``: true, but silent about
    which of the two keys was wrong, leaving an agent that cannot see its
    own document to guess. ``field`` must still be the top-level ``"params"``
    (``tests/test_agent_schemas.py``'s ``_run_case`` walks only
    ``case.path[0]``); only the message may name the bad key.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "lathe")

    result = agent_clay.call(
        ctx,
        session,
        "clay_set_params",
        {"uid": uid, "params": {"segments": 8, "profile": [[1.0, float("nan")]]}},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "params"
    message = result["content"][0]["text"]
    assert "profile" in message
    assert "segments" not in message


def test_two_bad_params_are_both_named_in_one_refusal() -> None:
    """A caller that got two params wrong in the same call should not need
    two round trips to learn about the second -- the same determinism rule
    :func:`agent_clay._unknown_argument_refusal` already holds unknown
    argument names to.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "lathe")

    result = agent_clay.call(
        ctx,
        session,
        "clay_set_params",
        {
            "uid": uid,
            "params": {"segments": float("inf"), "profile": [[1.0, float("nan")]]},
        },
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "params"
    message = result["content"][0]["text"]
    assert "profile" in message
    assert "segments" in message


def test_clay_add_primitive_also_names_which_param_is_bad() -> None:
    """The same duplicated loop lived in ``_h_add_primitive`` -- this is the
    other of the two doors the bug shipped through.
    """
    ctx = _Ctx()
    session = agent_clay.Session()

    result = agent_clay.call(
        ctx,
        session,
        "clay_add_primitive",
        {
            "generator": "lathe",
            "params": {"segments": 8, "profile": [[1.0, float("nan")]]},
        },
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "params"
    message = result["content"][0]["text"]
    assert "profile" in message
    assert "segments" not in message


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
    """Mirrors ``tests/modes/clay/test_clay_service.py``'s own shape for ``import_mesh``."""
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


# --- clay_add_mesh: geometry an agent hands over directly --------------------


def test_a_hand_built_tetrahedron_places_and_reports_itself_closed() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(ctx, session, "clay_add_mesh", _TETRA_MESH_ARGS)
    assert result["isError"] is False, result
    row = _payload(result)
    assert row["faces"] == 4
    assert row["verts"] == 4
    assert row["generator"] is None
    assert row["closed"] is True
    assert row["findings"] == []

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert len(tab.doc.history.history()) == 1
    assert tab.doc.undo()
    assert len(tab.doc.objects) == 0


def test_an_open_sheet_places_and_reports_itself_not_closed() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx,
        session,
        "clay_add_mesh",
        {
            "positions": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            "faces": [[0, 1, 2, 3]],
        },
    )
    assert result["isError"] is False, result
    row = _payload(result)
    assert row["closed"] is False
    kinds = {f["kind"] for f in row["findings"]}
    assert "hole" in kinds


def test_a_face_index_past_positions_is_refused_naming_the_face() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx,
        session,
        "clay_add_mesh",
        {"positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "faces": [[0, 1, 5]]},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "faces"
    assert "face 0" in result["content"][0]["text"]
    assert "corner 2" in result["content"][0]["text"]
    # A refused call places nothing -- no document minted for it either.
    assert session.tab_uid == ""


def test_a_uv_with_the_wrong_corner_count_for_one_face_is_refused_naming_that_face() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx,
        session,
        "clay_add_mesh",
        {
            "positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "faces": [[0, 1, 2]],
            "uv": [[[0.0, 0.0], [1.0, 0.0]]],  # 2 corners for a 3-corner face
        },
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "uv"
    assert "uv[0]" in result["content"][0]["text"]
    assert "face 0" in result["content"][0]["text"]


def test_a_ragged_positions_row_is_refused_by_field_rather_than_crashing() -> None:
    """The off-wire shape ``Mesh.__post_init__`` handles badly: a ragged row
    reaching ``np.array(value, dtype=...)`` raises a bare exception with no
    field name attached, which only ``call()``'s generic backstop would
    catch. This must be refused before that point is ever reached."""
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx,
        session,
        "clay_add_mesh",
        {"positions": [[0, 0, 0], [1, 0]], "faces": [[0, 1, 0]]},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "positions"
    assert "failed unexpectedly" not in result["content"][0]["text"]


def test_clay_add_mesh_refuses_above_the_size_ceiling_before_allocating() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    oversized = [[0.0, 0.0, 0.0]] * (agent_clay.MAX_MESH_VERTICES + 1)
    result = agent_clay.call(
        ctx, session, "clay_add_mesh", {"positions": oversized, "faces": [[0, 1, 2]]}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "positions"
    assert session.tab_uid == ""

    oversized_faces = [[0, 1, 2]] * (agent_clay.MAX_MESH_FACES + 1)
    result = agent_clay.call(
        ctx,
        session,
        "clay_add_mesh",
        {"positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "faces": oversized_faces},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "faces"
    assert session.tab_uid == ""


def test_clay_add_mesh_placed_object_has_no_generator_and_clay_set_params_refuses_it() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    added = agent_clay.call(ctx, session, "clay_add_mesh", _TETRA_MESH_ARGS)
    uid = _payload(added)["uid"]

    result = agent_clay.call(
        ctx, session, "clay_set_params", {"uid": uid, "params": {"radius": 1.0}}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "uid"


def test_clay_add_mesh_with_a_taken_name_is_refused_with_field_name_and_places_nothing() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")  # named "Box"
    result = agent_clay.call(
        ctx, session, "clay_add_mesh", {**_TETRA_MESH_ARGS, "name": "Box"}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "name"
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert len(tab.doc.objects) == 1


def test_clay_add_mesh_places_with_translation_rotation_scale_name_and_material() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx,
        session,
        "clay_add_mesh",
        {
            **_TETRA_MESH_ARGS,
            "translation": [1.0, 2.0, 3.0],
            "rotation": [0.0, 90.0, 0.0],
            "scale": [1.0, 2.0, 1.0],
            "name": "Wedge",
            "material": 0,
        },
    )
    assert result["isError"] is False, result
    row = _payload(result)
    assert row["name"] == "Wedge"
    assert row["translation"] == pytest.approx([1.0, 2.0, 3.0])
    assert row["material"] == 0

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert len(tab.doc.history.history()) == 1
    assert tab.doc.undo()
    assert len(tab.doc.objects) == 0


def test_clay_add_mesh_returns_the_row_clay_scene_would_have_shown_plus_closed_and_findings() -> (
    None
):
    ctx = _Ctx()
    session = agent_clay.Session()
    added = agent_clay.call(ctx, session, "clay_add_mesh", _TETRA_MESH_ARGS)
    added_row = _payload(added)
    scene = agent_clay.call(ctx, session, "clay_scene", {})
    scene_row = _payload(scene)["objects"][0]
    extra = {"closed", "findings"}
    assert {k: v for k, v in added_row.items() if k not in extra} == scene_row


def test_clay_add_mesh_is_a_valid_clay_batch_first_call() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx,
        session,
        "clay_batch",
        {"calls": [{"name": "clay_add_mesh", "arguments": _TETRA_MESH_ARGS}]},
    )
    assert result["isError"] is False, result
    assert session.tab_uid != ""


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


# --- B5a -- rollback_on_error -------------------------------------------------
#
# Tranche 5's fifth piece. ``clay_batch``'s documented contract -- stop at the
# first refusal, keep the successful prefix -- is unchanged and untouched by
# any of this; ``rollback_on_error`` is an opt-in, default-false argument for
# an agent that would rather the partial work never existed. Because the
# whole run already folds into one undo step, reversing it is one
# ``history.undo(doc, redoable=False)`` -- see that method's own docstring
# (``src/warlock/studio/undo.py``) for the cancelled-lift incident that
# argument exists for.


def test_rollback_on_error_true_leaves_the_document_exactly_as_it_was_before_the_batch() -> (
    None
):
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    objects_before = len(tab.doc.objects)
    history_before = len(tab.doc.history)
    dirty_before = tab.doc.dirty

    calls = [
        {"name": "clay_add_primitive", "arguments": {"generator": "cylinder"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "nope"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "cone"}},
    ]
    result = agent_clay.call(
        ctx, session, "clay_batch", {"calls": calls, "rollback_on_error": True}
    )
    assert result["isError"] is True
    payload = _payload(result)
    assert payload["stopped_at"] == 1
    assert payload["completed"] == 1

    assert len(tab.doc.objects) == objects_before
    assert len(tab.doc.history) == history_before
    assert tab.doc.dirty == dirty_before


def test_without_rollback_on_error_the_successful_prefix_survives_unchanged() -> None:
    """The contract-preservation claim: omitting the flag must behave
    byte-for-byte as it does today, not merely "similarly"."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    objects_before = len(tab.doc.objects)

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
    assert payload["changed"] is True
    assert payload["rolled_back"] is False
    assert len(tab.doc.objects) == objects_before + 1
    assert tab.doc.undo()
    assert len(tab.doc.objects) == objects_before


def test_a_rolled_back_batch_cannot_be_redone() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    objects_before = len(tab.doc.objects)

    calls = [
        {"name": "clay_add_primitive", "arguments": {"generator": "cylinder"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "nope"}},
    ]
    result = agent_clay.call(
        ctx, session, "clay_batch", {"calls": calls, "rollback_on_error": True}
    )
    assert result["isError"] is True
    payload = _payload(result)
    assert payload["completed"] == 1
    assert payload["rolled_back"] is True
    assert len(tab.doc.objects) == objects_before

    redo_result = agent_clay.call(ctx, session, "clay_redo", {})
    assert redo_result["isError"] is False, redo_result
    assert _payload(redo_result)["moved"] == 0
    assert len(tab.doc.objects) == objects_before


def test_rolled_back_batch_reports_changed_false_and_rolled_back_true() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")

    calls = [
        {"name": "clay_add_primitive", "arguments": {"generator": "cylinder"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "nope"}},
    ]
    result = agent_clay.call(
        ctx, session, "clay_batch", {"calls": calls, "rollback_on_error": True}
    )
    payload = _payload(result)
    assert payload["changed"] is False
    assert payload["rolled_back"] is True
    # "completed" stays diagnostic -- how many calls succeeded before the
    # refusal -- and stays true whether or not that work was then reversed.
    assert payload["completed"] == 1


def test_kept_prefix_batch_reports_changed_true_and_rolled_back_false() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")

    calls = [
        {"name": "clay_add_primitive", "arguments": {"generator": "cylinder"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "nope"}},
    ]
    result = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})
    payload = _payload(result)
    assert payload["changed"] is True
    assert payload["rolled_back"] is False


def test_rollback_on_error_on_a_batch_that_succeeds_changes_nothing() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)

    calls = [
        {"name": "clay_add_primitive", "arguments": {"generator": "cylinder"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "cone"}},
    ]
    result = agent_clay.call(
        ctx, session, "clay_batch", {"calls": calls, "rollback_on_error": True}
    )
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["rolled_back"] is False
    assert payload["changed"] is True
    assert len(tab.doc.objects) == 3


def test_rollback_on_error_on_a_batch_that_minted_the_document_leaves_it_existing_and_empty() -> (
    None
):
    """Rule 7's limit, pinned as a stated behaviour: only the document's own
    undo stack is unwound. The mint ``_h_batch`` does itself when the session
    owns no tab yet pushes no undo step (see ``_h_add_primitive``'s own
    comment on why), so it happens before ``mark`` and a rollback cannot
    touch it -- the session is left with an empty document, not a dead pin
    naming a tab that no longer exists."""
    ctx = _Ctx()
    session = agent_clay.Session()
    assert session.tab_uid == ""

    calls = [{"name": "clay_add_primitive", "arguments": {"generator": "nope"}}]
    result = agent_clay.call(
        ctx, session, "clay_batch", {"calls": calls, "rollback_on_error": True}
    )
    assert result["isError"] is True
    payload = _payload(result)
    assert payload["rolled_back"] is False
    assert session.tab_uid != ""

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert tab is not None
    assert len(tab.doc.objects) == 0


# --- B5b -- $ref inside a clay_batch entry -----------------------------------
#
# Tranche 5's fourth piece: a batch entry never sees an earlier entry's own
# result until the whole batch returns, so it has no uid to pass an object
# an earlier entry in the same batch just created. ``{"$ref": "<name>"}``
# resolves against the object's own ``name`` -- the namespace
# ``clay_add_primitive``/``clay_add_figure``/``clay_add_mesh`` already refuse
# a collision on and ``clay_scene`` already reports -- at the moment its
# entry runs, so a hub can be built and then addressed by name without a
# ``clay_scene`` read splitting the work across two batches.


def test_a_ref_in_a_later_batch_entry_names_an_object_an_earlier_entry_just_created() -> (
    None
):
    """The motivating case from the tranche brief: build a named hub, then
    act on it by name, both inside one batch."""
    ctx = _Ctx()
    session = agent_clay.Session()

    calls = [
        {"name": "clay_add_primitive", "arguments": {"generator": "box", "name": "hub"}},
        {
            "name": "clay_transform",
            "arguments": {"uid": {"$ref": "hub"}, "translation": [1.0, 2.0, 3.0]},
        },
    ]
    result = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["completed"] == 2
    assert payload["stopped_at"] is None

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    hub = next(o for o in tab.doc.objects if o.name == "hub")
    assert list(hub.translation) == pytest.approx([1.0, 2.0, 3.0])


def test_a_ref_resolves_an_object_already_in_the_document_before_the_batch_began() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    add = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box", "name": "pre"})
    assert add["isError"] is False, add
    pre_uid = _payload(add)["uid"]

    calls = [
        {
            "name": "clay_transform",
            "arguments": {"uid": {"$ref": "pre"}, "translation": [4.0, 5.0, 6.0]},
        }
    ]
    result = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})
    assert result["isError"] is False, result

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert list(tab.doc.by_uid(pre_uid).translation) == pytest.approx([4.0, 5.0, 6.0])


def test_a_ref_inside_a_list_of_uids_resolves() -> None:
    """``clay_material``'s ``uids`` is a list, not a single value -- the
    recursive-into-a-list half of the contract, distinct from the
    single-value case the other tests here cover."""
    ctx = _Ctx()
    session = agent_clay.Session()

    calls = [
        {"name": "clay_add_primitive", "arguments": {"generator": "box", "name": "a"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "cylinder", "name": "b"}},
        {
            "name": "clay_material",
            "arguments": {
                "uids": [{"$ref": "a"}, {"$ref": "b"}],
                "color": [1.0, 0.0, 0.0],
            },
        },
    ]
    result = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})
    assert result["isError"] is False, result
    assert _payload(result)["completed"] == 3

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    a = next(o for o in tab.doc.objects if o.name == "a")
    b = next(o for o in tab.doc.objects if o.name == "b")
    assert a.material == b.material != 0


def test_a_ref_to_a_name_no_object_has_stops_the_batch_and_keeps_the_prefix() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()

    calls = [
        {"name": "clay_add_primitive", "arguments": {"generator": "box", "name": "hub"}},
        {
            "name": "clay_transform",
            "arguments": {"uid": {"$ref": "ghost"}, "translation": [1.0, 0.0, 0.0]},
        },
        {"name": "clay_add_primitive", "arguments": {"generator": "cone", "name": "never"}},
    ]
    result = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})
    assert result["isError"] is True
    payload = _payload(result)
    assert payload["stopped_at"] == 1
    assert payload["completed"] == 1

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    names = {o.name for o in tab.doc.objects}
    assert "hub" in names
    assert "never" not in names

    ref_failure = payload["results"][1]
    assert ref_failure["isError"] is True
    assert ref_failure["structuredContent"]["field"] == "uid"
    assert ref_failure["structuredContent"]["recovery"] == "read_scene"
    # Distinguishes a real ``$ref`` miss from the generic "not an int" refusal
    # a bare unresolved ``{"$ref": ...}`` dict would otherwise fall through
    # to (which also names field="uid"/recovery="read_scene", coincidentally)
    # -- this message names the object by the name that was actually looked
    # up, not the raw dict ``int()`` choked on.
    assert "no object named" in ref_failure["content"][0]["text"]
    assert "ghost" in ref_failure["content"][0]["text"]


def test_a_dict_carrying_ref_beside_another_key_is_refused() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()

    calls = [
        {"name": "clay_add_primitive", "arguments": {"generator": "box", "name": "hub"}},
        {
            "name": "clay_transform",
            "arguments": {
                "uid": {"$ref": "hub", "extra": 1},
                "translation": [1.0, 0.0, 0.0],
            },
        },
    ]
    result = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})
    assert result["isError"] is True
    payload = _payload(result)
    assert payload["stopped_at"] == 1
    ref_failure = payload["results"][1]
    assert ref_failure["structuredContent"]["field"] == "uid"
    # "fix_arguments", not "read_scene" -- this is a malformed call, not a
    # name the document happens not to hold, and the message names $ref by
    # name rather than falling through to the generic bad-uid refusal (which
    # would also land on field="uid", coincidentally, but with recovery
    # "read_scene" and no mention of $ref at all).
    assert ref_failure["structuredContent"]["recovery"] == "fix_arguments"
    assert "$ref" in ref_failure["content"][0]["text"]


def test_a_ref_to_an_ambiguous_name_is_refused_naming_both_uids() -> None:
    """Names are unique at the agent door (``clay_add_primitive`` and friends
    refuse a collision) but not globally -- the human door's own rename can
    still land two objects on one name, so ``$ref`` has to cope with more
    than one match rather than silently picking the first."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid1 = _new_agent_tab(ctx, session, "box")
    uid2 = _payload(agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "cone"}))[
        "uid"
    ]
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    # Bypasses ``_h_rename``'s own collision refusal on purpose: ``doc.set_props``
    # is the lower-level door the human-facing rename panel and ``_h_rename``
    # both sit on top of, and only the agent handlers refuse a name collision
    # -- so this is exactly how two objects really do end up sharing a name.
    tab.doc.set_props(uid1, name="dup")
    tab.doc.set_props(uid2, name="dup")

    calls = [
        {
            "name": "clay_transform",
            "arguments": {"uid": {"$ref": "dup"}, "translation": [1.0, 0.0, 0.0]},
        }
    ]
    result = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})
    assert result["isError"] is True
    payload = _payload(result)
    ref_failure = payload["results"][0]
    assert ref_failure["isError"] is True
    assert ref_failure["structuredContent"]["field"] == "uid"
    assert set(ref_failure["structuredContent"]["uids"]) == {uid1, uid2}


def test_a_ref_in_an_ordinary_non_batched_call_is_not_resolved() -> None:
    """The boundary the brief pins: outside a batch an agent already has the
    creating call's own result in hand, so ``$ref`` is batch-only and a
    direct call must not learn about it -- a ``$ref`` there is refused as
    the malformed uid it plainly is (``int({"$ref": "hub"})`` raises the same
    as any other non-numeric ``uid``), not specially resolved."""
    ctx = _Ctx()
    session = agent_clay.Session()
    hub_uid = _payload(
        agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box", "name": "hub"})
    )["uid"]

    result = agent_clay.call(
        ctx,
        session,
        "clay_transform",
        {"uid": {"$ref": "hub"}, "translation": [1.0, 0.0, 0.0]},
    )
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["field"] == "uid"
    assert structured["recovery"] == "read_scene"

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    # Untouched -- proof this was refused, not resolved and then acted on.
    assert list(tab.doc.by_uid(hub_uid).translation) == pytest.approx([0.0, 0.0, 0.0])


# --- clay_program --------------------------------------------------------------


def test_clay_program_builds_a_four_leg_table_in_one_undo_step() -> None:
    """A slab plus a ``repeat`` over four legs, each named from the loop
    variable -- the shape a hand-built ``clay_batch`` run of the identical
    five calls already proves, compiled from a program instead."""
    ctx = _Ctx()
    session = agent_clay.Session()
    program = {
        "steps": [
            {
                "add": {
                    "generator": "box",
                    "params": {"size": [1.0, 0.1, 1.0]},
                    "translation": [0, 1, 0],
                    "id": "top",
                }
            },
            {
                "repeat": {
                    "ranges": {"i": [0, 1, 2, 3]},
                    "steps": [
                        {
                            "add": {
                                "generator": "box",
                                "params": {"size": [0.1, 1, 0.1]},
                                "translation": [
                                    "0.4*cos($i*90)", 0.5, "0.4*sin($i*90)",
                                ],
                                "id": "leg_{i}",
                            }
                        }
                    ],
                }
            },
        ]
    }
    result = agent_clay.call(ctx, session, "clay_program", program)
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["dry_run"] is False
    assert payload["validated"] == "execute"
    assert payload["stopped_at"] is None
    assert payload["completed"] == 5
    assert payload["changed"] is True
    assert payload["rolled_back"] is False
    assert {row["name"] for row in payload["objects"]} == {
        "top", "leg_0", "leg_1", "leg_2", "leg_3",
    }
    assert all("uid" in row for row in payload["objects"])

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert {o.name for o in tab.doc.objects} == {"top", "leg_0", "leg_1", "leg_2", "leg_3"}
    assert len(tab.doc.history) == 1
    assert tab.doc.history.top.label == "Agent program"
    assert tab.doc.undo()
    assert len(tab.doc.objects) == 0


def test_a_failing_call_mid_program_rolls_back_pushes_no_step_and_restores_selection() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    a_uid = _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    agent_clay.call(ctx, session, "clay_select", {"uids": [a_uid]})
    assert tab.doc.selection == {a_uid}
    history_before = len(tab.doc.history)

    program = {
        "steps": [
            {"add": {"generator": "box", "id": "c"}},
            # No object in this document ever gets this uid -- refuses at
            # _resolve_uid the same way an ordinary clay_transform would.
            {"transform": {"uid": 999999, "translation": [1.0, 0.0, 0.0]}},
        ]
    }
    result = agent_clay.call(ctx, session, "clay_program", program)
    assert result["isError"] is True
    payload = _payload(result)
    assert payload["rolled_back"] is True
    assert payload["changed"] is False
    assert payload["completed"] == 1
    assert payload["stopped_at"] == {"step": "steps[1].transform", "call": "clay_transform"}
    assert payload["failure"]["isError"] is True
    assert "999999" in payload["failure"]["content"][0]["text"]
    assert payload["objects"] == []

    # No step pushed -- the folded step this run would have pushed is fully
    # undone, not merely left off the history for some other reason.
    assert len(tab.doc.history) == history_before
    # The object this program itself placed is gone; the one that predates
    # it survives, and the selection is exactly what it was before this call.
    assert {o.uid for o in tab.doc.objects} == {a_uid}
    assert tab.doc.selection == {a_uid}


def test_a_dry_run_leaves_the_scene_and_dirty_flag_unchanged_and_returns_calls_with_no_uids() -> (
    None
):
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    history_before = len(tab.doc.history)
    dirty_before = tab.doc.dirty
    objects_before = {o.name for o in tab.doc.objects}

    result = agent_clay.call(
        ctx, session, "clay_program",
        {"steps": [{"add": {"generator": "cylinder", "id": "c"}}], "dry_run": True},
    )
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["dry_run"] is True
    assert payload["validated"] == "execute"
    assert payload["rolled_back"] is True
    assert payload["changed"] is False
    assert payload["objects"] == [{"id": "c", "name": "c"}]  # no uid
    assert payload["calls"] == [
        {"name": "clay_add_primitive", "arguments": {"generator": "cylinder", "name": "c"}}
    ]

    assert len(tab.doc.history) == history_before
    assert tab.doc.dirty == dirty_before
    assert {o.name for o in tab.doc.objects} == objects_before


def test_a_dry_run_with_no_document_compiles_only_and_mints_no_tab() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx, session, "clay_program",
        {"steps": [{"add": {"generator": "box"}}], "dry_run": True},
    )
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["validated"] == "compile"
    assert payload["objects"] == []
    assert payload["calls"] == [
        {"name": "clay_add_primitive", "arguments": {"generator": "box"}}
    ]
    assert session.tab_uid == ""


def test_a_compile_refusal_carries_field_and_path() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx, session, "clay_program",
        {"steps": [{"add": {"generator": "not-a-generator"}}]},
    )
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["field"] == "steps"
    assert "steps[0].add" in result["content"][0]["text"]
    assert "unknown generator" in result["content"][0]["text"]


def test_clay_program_is_refused_in_element_mode() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})

    result = agent_clay.call(
        ctx, session, "clay_program", {"steps": [{"add": {"generator": "box"}}]}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["recovery"] == "switch_mode"


def test_clay_programs_deadline_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deadline so tight that only the (always-unconditional) first call
    beats it -- the second is refused before it ever runs, and the whole
    attempt rolls back exactly as any other mid-program refusal would."""
    ctx = _Ctx()
    session = agent_clay.Session()
    monkeypatch.setattr(agent_clay, "PROGRAM_DEADLINE_S", 0.0)

    result = agent_clay.call(
        ctx, session, "clay_program",
        {"steps": [{"add": {"generator": "box", "id": "a"}}, {"add": {"generator": "cylinder"}}]},
    )
    assert result["isError"] is True
    payload = _payload(result)
    assert payload["rolled_back"] is True
    assert "deadline" in payload["failure"]["content"][0]["text"]
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert len(tab.doc.objects) == 0


def test_clay_program_is_batch_excluded() -> None:
    assert "clay_program" in agent_clay.BATCH_EXCLUDED
    program_args = {"steps": [{"add": {"generator": "box"}}]}
    result = agent_clay.call(
        _Ctx(), agent_clay.Session(), "clay_batch",
        {"calls": [{"name": "clay_program", "arguments": program_args}]},
    )
    assert result["isError"] is True
    assert "not a batchable tool" in result["content"][0]["text"]


def test_the_compiler_never_emits_a_batch_excluded_tool() -> None:
    """The reason ``clay_program`` folds into ``BATCH_EXCLUDED`` rather than
    being nested is moot if the compiler could still emit one of the other
    excluded names -- it cannot: every kind it compiles maps to a fixed,
    small set of tools, none of them in that set."""
    from warlock.studio.modes.clay.agent import program as ap

    compiled = ap.compile_program(
        {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"add": {"generator": "box", "id": "b"}},
                {"transform": {"uid": "a", "translation": [1, 0, 0]}},
                {"params": {"uid": "a", "params": {"size": [2, 2, 2]}}},
                {"material": {"uids": "a", "color": [1, 0, 0]}},
                {"boolean": {"kind": "union", "uids": ["a", "b"]}},
                {"select": {"uids": "a"}},
                {"delete": {"uids": "a"}},
            ]
        }
    )
    for call in compiled.calls:
        if call[0] != "live":
            assert call[0] not in agent_clay.BATCH_EXCLUDED, call[0]


def test_a_relative_move_composes_with_an_earlier_absolute_transform() -> None:
    """move's ``by`` adds to whatever the object already holds -- an
    absolute clay_transform-shaped step earlier in the same program included
    -- rather than starting over from the origin."""
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx, session, "clay_program",
        {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"transform": {"uid": "a", "translation": [1.0, 0.0, 0.0]}},
                {"move": {"uid": "a", "by": [0.0, 1.0, 0.0]}},
            ]
        },
    )
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["rolled_back"] is False
    assert payload["stopped_at"] is None
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    obj = next(o for o in tab.doc.objects if o.name == "a")
    assert list(obj.translation) == pytest.approx([1.0, 1.0, 0.0])
    assert tab.doc.history.top.label == "Agent program"


def test_turn_composes_rotations_and_scale_by_multiplies() -> None:
    """A single-axis composition well clear of the +-180 degree ambiguity a
    3-angle Euler readout can otherwise pick a different-looking (but
    equivalent) representation for -- 30 then 20 more is unambiguously 50."""
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx, session, "clay_program",
        {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"transform": {"uid": "a", "rotation": [0.0, 30.0, 0.0], "scale": [2.0, 2.0, 2.0]}},
                {"turn": {"uid": "a", "by": [0.0, 20.0, 0.0]}},
                {"scale_by": {"uid": "a", "factor": 1.5}},
            ]
        },
    )
    assert result["isError"] is False, result
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    obj = next(o for o in tab.doc.objects if o.name == "a")
    rx, ry, rz = agent_clay._euler_xyz_from_quat(obj.rotation)
    assert (rx, ry, rz) == pytest.approx((0.0, 50.0, 0.0), abs=1e-4)
    assert list(obj.scale) == pytest.approx([3.0, 3.0, 3.0])


def test_a_group_move_moves_every_member_as_one_undo_step() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    history_before = _history_len(ctx, session) if session.tab_uid else 0
    result = agent_clay.call(
        ctx, session, "clay_program",
        {
            "steps": [
                {"add": {"generator": "box", "id": "g1"}},
                {"add": {"generator": "box", "id": "g2", "translation": [2.0, 0.0, 0.0]}},
                {"group": {"id": "pair", "members": ["g1", "g2"]}},
                {"move": {"uid": "pair", "by": [0.0, 3.0, 0.0]}},
            ]
        },
    )
    assert result["isError"] is False, result
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    g1 = next(o for o in tab.doc.objects if o.name == "g1")
    g2 = next(o for o in tab.doc.objects if o.name == "g2")
    assert list(g1.translation) == pytest.approx([0.0, 3.0, 0.0])
    assert list(g2.translation) == pytest.approx([2.0, 3.0, 0.0])
    # Four compiled calls (2 adds + 1 move per member) still fold into the
    # one undo step every clay_program run promises.
    assert _history_len(ctx, session) == history_before + 1


def test_an_assert_failure_rolls_back_pushes_no_step_and_names_the_step_path() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx, session, "clay_program",
        {
            "steps": [
                {"add": {"generator": "box", "id": "a", "translation": [0.0, 0.5, 0.0]}},
                {"assert": {"condition": "size(a, 1) > 100"}},
            ]
        },
    )
    assert result["isError"] is True
    payload = _payload(result)
    assert payload["rolled_back"] is True
    assert payload["changed"] is False
    assert payload["stopped_at"] == {"step": "steps[1].assert", "call": "live:assert"}
    assert "size(a, 1) > 100" in payload["failure"]["content"][0]["text"]
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert len(tab.doc.objects) == 0


def test_a_passing_assert_over_touches_and_grounded_lets_the_program_commit() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx, session, "clay_program",
        {
            "steps": [
                {"add": {"generator": "box", "id": "a", "translation": [0.0, 0.5, 0.0]}},
                {"add": {"generator": "box", "id": "b", "translation": [1.0, 0.5, 0.0]}},
                {"assert": {"condition": "grounded(a) and touches(a, b)"}},
            ]
        },
    )
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["rolled_back"] is False
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert {o.name for o in tab.doc.objects} == {"a", "b"}


def test_a_dry_run_with_live_steps_still_leaves_the_scene_unchanged() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    history_before = len(tab.doc.history)
    objects_before = {o.name for o in tab.doc.objects}

    result = agent_clay.call(
        ctx, session, "clay_program",
        {
            "steps": [
                {"add": {"generator": "box", "id": "c", "translation": [0.0, 0.5, 0.0]}},
                {"move": {"uid": "c", "by": [1.0, 0.0, 0.0]}},
                {"assert": {"condition": "grounded(c)"}},
            ],
            "dry_run": True,
        },
    )
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["rolled_back"] is True
    assert payload["changed"] is False
    assert len(tab.doc.history) == history_before
    assert {o.name for o in tab.doc.objects} == objects_before


def test_an_unknown_fact_inside_assert_is_refused_at_compile_time_with_a_path() -> None:
    from warlock.studio.modes.clay.agent import program as ap

    err = None
    try:
        ap.compile_program(
            {
                "steps": [
                    {"add": {"generator": "box", "id": "a"}},
                    {"assert": {"condition": "frobnicate(a)"}},
                ]
            }
        )
    except ap.ProgramError as exc:
        err = exc
    assert err is not None
    assert err.field == "steps"
    assert err.path == "steps[1].assert"
    assert "frobnicate" in err.reason


def test_facts_read_lo_hi_size_center_count_and_exists_off_known_boxes() -> None:
    """A 1x1x1 box sat with its bottom on the ground at [0, 0.5, 0] has a
    known box (y from 0 to 1) -- and a group of two such boxes, and a name
    nothing was ever given, exercise count/exists the same way."""
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx, session, "clay_program",
        {
            "steps": [
                {"add": {"generator": "box", "id": "a", "translation": [0.0, 0.5, 0.0]}},
                {"add": {"generator": "box", "id": "b", "translation": [5.0, 0.5, 0.0]}},
                {"group": {"id": "pair", "members": ["a", "b"]}},
                {"assert": {
                    "condition": (
                        "lo(a, 1) > -0.01 and lo(a, 1) < 0.01 "
                        "and hi(a, 1) > 0.99 and hi(a, 1) < 1.01 "
                        "and size(a, 1) > 0.99 and size(a, 1) < 1.01 "
                        "and center(a, 1) > 0.49 and center(a, 1) < 0.51 "
                        "and count(pair) == 2 "
                        "and exists(a) and not exists(nope)"
                    )
                }},
            ]
        },
    )
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["rolled_back"] is False


def test_an_unknown_id_inside_assert_is_refused_at_compile_time_with_a_path() -> None:
    from warlock.studio.modes.clay.agent import program as ap

    err = None
    try:
        ap.compile_program(
            {
                "steps": [
                    {"add": {"generator": "box", "id": "a"}},
                    {"assert": {"condition": "touches(a, ghost)"}},
                ]
            }
        )
    except ap.ProgramError as exc:
        err = exc
    assert err is not None
    assert err.path == "steps[1].assert"
    assert "unknown id 'ghost'" in err.reason


def test_a_session_whose_document_was_closed_can_start_another_via_clay_program() -> None:
    """The same recovery ``_MINTS_A_TAB``'s own parametrized test proves for
    the three creator tools, reached separately here because clay_program is
    deliberately not one of ``agent_clay.MINTS_A_DOCUMENT`` -- see
    ``_ALSO_MINTS_A_TAB``'s own comment."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    closed = session.tab_uid
    state = clay_mode.ensure(ctx)
    assert state.close(closed)

    result = agent_clay.call(
        ctx, session, "clay_program", {"steps": [{"add": {"generator": "box"}}]}
    )
    assert result["isError"] is False, result
    assert session.tab_uid and session.tab_uid != closed
    assert state.get(session.tab_uid) is not None


def _clay_op_description() -> str:
    tool = next(t for t in agent_clay.tools() if t.name == "clay_op")
    return tool.description


def test_clay_op_catalog_describes_a_boolean_param_as_a_boolean():
    """``_op_catalog`` used to fold every param into "name (low-high, default
    x)" -- a bare range is a poor description of a checkbox, and this prose is
    the only thing a model is ever told about an op's arguments."""
    description = _clay_op_description()
    assert "fit (0.0-1.0, default 1.0)" not in description
    assert "fit (boolean" in description


def test_clay_op_catalog_describes_axis_as_a_named_three_way_choice():
    """Not a boolean, and not a bare 0-2 range either: naming what each value
    means is the whole point, since a model reads only this sentence before
    its first call."""
    description = _clay_op_description()
    assert "axis (0.0-2.0, default" not in description
    assert "0=X" in description and "1=Y" in description and "2=Z" in description


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
    from. ``structuredContent`` itself is no longer absent here -- every
    refusal now carries ``changed`` (see ``fail()``'s own docstring) -- so
    this checks the one key that really does distinguish a validation
    refusal from this generic one, rather than the envelope's presence."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)

    result = agent_clay.call(ctx, session, "clay_render", {"view": "front"})
    assert result["isError"] is True
    assert "field" not in (result.get("structuredContent") or {})


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


# --- shading -------------------------------------------------------------


def test_clay_render_shading_defaults_to_unlit_and_threads_through_to_render_png(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    fake = _install_fake_view(monkeypatch)

    default = agent_clay.call(ctx, session, "clay_render", {})
    assert default["isError"] is False, default
    assert fake.calls[0]["shading"] == "unlit"

    explicit = agent_clay.call(ctx, session, "clay_render", {"shading": "wireframe"})
    assert explicit["isError"] is False, explicit
    assert fake.calls[1]["shading"] == "wireframe"


def test_clay_render_refuses_grid_combined_with_object_id_shading() -> None:
    """The id pass never draws a grid -- a grid line has no uid behind it,
    which would corrupt the very pixel counts 'ids' exists to report. Refused
    before any GL work, exactly like every other named-field render refusal
    above."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_render", {"shading": "object_id", "grid": True}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "grid"


def test_clay_render_refuses_object_id_combined_with_compare() -> None:
    """A compare reply is a picture-vs-picture comparison with no room for
    the uid/colour/pixel table object_id answers with, and a stored
    reference was captured as an ordinary picture -- comparing it against
    flat id colours is not a coherent question."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    _add_inline_reference(ctx, session, "ref1")

    result = agent_clay.call(
        ctx, session, "clay_render", {"shading": "object_id", "compare": "ref1"}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "shading"


def test_clay_render_object_id_shares_one_colour_map_and_sums_pixels_across_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'share one map' means the same uid gets the same colour in every
    requested view, and this file has no GPU to actually draw one -- so the
    fake's own per-call rows stand in for two views seeing the same object,
    and this checks ``_h_render`` merges them into one row summed across
    views rather than keeping (or overwriting) one view's count."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    fake = _install_fake_view(monkeypatch, id_rows=[(uid, "#112233", 40)])

    result = agent_clay.call(
        ctx, session, "clay_render", {"shading": "object_id", "views": ["front", "back"]}
    )
    assert result["isError"] is False, result
    assert len(fake.id_calls) == 2
    header = _payload(result)
    ids = {row[0]: row for row in header["ids"]}
    assert ids[uid] == [uid, "#112233", 80]


# --- regressions: the compare path's clamp and skipped frame-size check ------


def test_clay_render_compare_refuses_a_size_over_1024_instead_of_clamping_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the silent ``size = min(size, 1024)`` clamp: before the
    fix this call answered ``isError: False`` with a 1024-square picture
    nobody asked for instead of refusing the 1500 actually given -- the same
    'Refused, not clamped' rule the plain ``size`` check already states for
    itself just upstream, not followed here until now. Fails against the
    unfixed handler, which reports success with the clamped size silently
    substituted."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    _add_inline_reference(ctx, session, "ref1")
    _install_fake_view(monkeypatch)

    result = agent_clay.call(ctx, session, "clay_render", {"compare": "ref1", "size": 1500})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "size"


def test_clay_render_compare_is_refused_when_the_sheet_would_not_fit_one_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the compare branch used to ``return`` its sheet before the
    reply-frame size check further down ever ran, so a beside/overlay sheet
    over ``protocol.MAX_FRAME`` reached the caller with ``isError: False``
    instead of being refused the way the ordinary multi-view path already
    was. ``MAX_FRAME`` is monkeypatched absurdly small so the tiny fake PNG
    this file uses is still over budget, rather than needing a real
    multi-megapixel sheet to prove the same point. Fails against the
    unfixed handler, which never reaches the check on this path at all."""
    from warlock.mcp import rpc

    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    _add_inline_reference(ctx, session, "ref1")
    _install_fake_view(monkeypatch)
    monkeypatch.setattr(rpc, "MAX_FRAME", 10)

    result = agent_clay.call(ctx, session, "clay_render", {"compare": "ref1"})
    assert result["isError"] is True


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


# --- compare's silhouette header ---------------------------------------------


def _shape_png(size: int, rect: tuple[int, int, int, int], *, alpha: bool = False) -> bytes:
    """A ``size`` x ``size`` picture with ``rect`` (x0, y0, x1, y1) filled in
    and the rest of the canvas white (or, ``alpha=True``, transparent) --
    a hand-built stand-in for both a ``render_ids`` picture (never white
    where an object is) and an alpha-carrying reference, so a test can put an
    exact, known footprint on either side of the comparison."""
    x0, y0, x1, y1 = rect
    if alpha:
        arr = np.zeros((size, size, 4), dtype=np.uint8)
        arr[y0:y1, x0:x1] = (10, 20, 30, 255)
        mode = "RGBA"
    else:
        arr = np.full((size, size, 3), 255, dtype=np.uint8)
        arr[y0:y1, x0:x1] = (10, 20, 30)
        mode = "RGB"
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, "PNG")
    return buf.getvalue()


def _busy_reference_png(size: int = 300) -> bytes:
    """An RGB (no alpha) reference with no clean background to flood-fill:
    a uniform patch in each corner (so the fill seeds and spreads there,
    exactly as ``pipelines.reference.subject_mask``'s own corner sampling
    expects) surrounded by salt-and-pepper noise the tight fill tolerance
    cannot cross -- so the 'background' the fill finds is a sliver and the
    'subject' it leaves behind covers the whole frame, over
    ``bench.metrics.MASK_COVERAGE_CEILING``. Seeded, so the noise pattern is
    fixed rather than a source of a flaky test."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 2, size=(size, size, 3), dtype=np.uint8) * 255
    p = 16
    corner = np.array([200, 200, 200], dtype=np.uint8)
    arr[:p, :p] = corner
    arr[:p, -p:] = corner
    arr[-p:, :p] = corner
    arr[-p:, -p:] = corner
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, "PNG")
    return buf.getvalue()


def _add_reference_png(
    ctx: _Ctx, session: agent_clay.Session, name: str, png: bytes, view: str = "other"
) -> dict:
    b64 = base64.b64encode(png).decode("ascii")
    result = agent_clay.call(
        ctx, session, "clay_reference_add", {"name": name, "png_base64": b64, "view": view}
    )
    assert result["isError"] is False, result
    return result


def test_clay_render_compare_silhouette_scores_identical_footprints_near_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reference whose alpha matches the render's own footprint exactly --
    taken, in the real app, from a ``render_ids`` pass of the same scene at
    the same size -- is the case the metric exists to score highest."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    rect = (32, 32, 96, 96)
    render_png = _shape_png(128, rect)
    _add_reference_png(ctx, session, "ref1", _shape_png(128, rect, alpha=True))
    _install_fake_view(monkeypatch, png=render_png)

    result = agent_clay.call(ctx, session, "clay_render", {"compare": "ref1"})
    assert result["isError"] is False, result
    payload = _payload(result)
    silhouette = payload["silhouette"]
    assert silhouette["iou"] >= 0.99, silhouette
    assert silhouette["reference_mask"] == "alpha"
    assert silhouette["aspect_error"] == 0.0
    assert result["content"][1]["type"] == "image"  # the picture, regardless


def test_clay_render_compare_silhouette_is_null_with_a_reason_for_a_busy_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reference with no alpha and no clean corner-to-fill background is
    not a silhouette to score -- null with a reason (a leak or near-total
    coverage), and the comparison picture still comes back regardless."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    _add_reference_png(ctx, session, "ref1", _busy_reference_png())
    _install_fake_view(monkeypatch, png=_shape_png(64, (16, 16, 48, 48)))

    result = agent_clay.call(ctx, session, "clay_render", {"compare": "ref1"})
    assert result["isError"] is False, result
    payload = _payload(result)
    silhouette = payload["silhouette"]
    assert silhouette["iou"] is None
    assert silhouette["reason"]
    assert result["content"][1]["type"] == "image"


def test_clay_render_compare_silhouette_is_null_when_the_reference_has_no_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-white reference (no alpha, nothing for the flood fill to find
    but background) refuses no call -- the picture still returns, with a
    null silhouette naming why."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    _add_reference_png(ctx, session, "ref1", _tiny_png())
    _install_fake_view(monkeypatch, png=_shape_png(64, (16, 16, 48, 48)))

    result = agent_clay.call(ctx, session, "clay_render", {"compare": "ref1"})
    assert result["isError"] is False, result
    silhouette = _payload(result)["silhouette"]
    assert silhouette == {"iou": None, "reason": silhouette["reason"]}
    assert silhouette["reason"]
    assert result["content"][1]["type"] == "image"


def test_clay_render_compare_silhouette_is_null_when_the_render_has_no_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The render side of the same rule: the fake's default picture is a
    blank white square, so ``render_ids_mask`` finds nothing even though the
    reference has a real, alpha-clean subject."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    _add_reference_png(ctx, session, "ref1", _shape_png(64, (16, 16, 48, 48), alpha=True))
    _install_fake_view(monkeypatch)  # default fake.png is a blank white square

    result = agent_clay.call(ctx, session, "clay_render", {"compare": "ref1"})
    assert result["isError"] is False, result
    silhouette = _payload(result)["silhouette"]
    assert silhouette["iou"] is None
    assert silhouette["reason"]
    assert result["content"][1]["type"] == "image"


def test_clay_render_compare_returns_the_picture_even_if_silhouette_measurement_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'Never a refusal' stated as a test: whatever goes wrong measuring the
    silhouette, the comparison picture an agent asked for is not held
    hostage to it."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    _add_inline_reference(ctx, session, "ref1")
    _install_fake_view(monkeypatch)

    from warlock.bench import metrics as bench_metrics

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(bench_metrics, "compare_silhouette", _boom)

    result = agent_clay.call(ctx, session, "clay_render", {"compare": "ref1"})
    assert result["isError"] is False, result
    silhouette = _payload(result)["silhouette"]
    assert silhouette["iou"] is None
    assert silhouette["reason"]
    assert result["content"][1]["type"] == "image"


# ==============================================================================
# B9 -- the mesh, taken apart: element mode, explicit index, query, the read
# ==============================================================================


def test_clay_op_inset_refuses_until_the_agent_switches_to_face_mode_and_then_runs() -> None:
    """The headline capability this change adds. Before it, nothing in
    ``studio/modes/clay/agent/dispatch.py`` ever called ``doc.set_element_mode`` or
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
    from warlock.kernels.mesh import select as clay_select_mod

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
    from warlock.kernels.mesh import select as clay_select_mod

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


def test_a_call_with_no_uid_is_told_to_give_one_rather_than_that_uid_none_does_not_exist() -> None:
    """The 2026-09-15 Clay agent benchmark sitting: ``clay_select_by {}`` answered "no object
    with uid None." with ``recovery: read_scene`` -- a uid the model never
    passed, and a recovery that could not help. ``clay_transform`` shares the
    same resolver, so it is held to the same sentence."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")

    for tool in ("clay_select_by", "clay_transform"):
        result = agent_clay.call(ctx, session, tool, {})
        assert result["isError"] is True, tool
        assert result["content"][0]["text"] == "give a value for 'uid'.", tool
        assert result["structuredContent"]["recovery"] == "fix_arguments", tool


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


def test_clay_analyze_reports_bounds_area_volume_and_ground_for_a_box() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")

    result = agent_clay.call(ctx, session, "clay_analyze", {})
    assert result["isError"] is False, result
    payload = _payload(result)
    row = next(o for o in payload["objects"] if o["uid"] == uid)
    assert row["closed"] is True
    assert row["volume"] == pytest.approx(1.0, abs=1e-4)
    assert row["area"] == pytest.approx(6.0, abs=1e-4)
    assert row["bounds"] is not None
    assert "floating" in payload  # a whole-document call: no uids were given
    assert payload["tolerances"] == {"contact_tol": 0.001, "near": 0.05, "symmetry_tol": 0.002}


def test_clay_analyze_with_uids_skips_floating_and_reports_only_those_objects() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid1 = _new_agent_tab(ctx, session)
    add2 = agent_clay.call(
        ctx, session, "clay_add_primitive", {"generator": "box", "translation": [5.0, 5.0, 0.0]}
    )
    uid2 = _payload(add2)["uid"]

    result = agent_clay.call(ctx, session, "clay_analyze", {"uids": [uid1]})
    assert result["isError"] is False, result
    payload = _payload(result)
    assert {o["uid"] for o in payload["objects"]} == {uid1}
    assert "floating" not in payload
    del uid2


def test_clay_analyze_pushes_no_undo_step() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    before = _history_len(ctx, session)

    result = agent_clay.call(ctx, session, "clay_analyze", {})
    assert result["isError"] is False, result
    assert _history_len(ctx, session) == before


def test_clay_analyze_is_batchable() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_batch", {"calls": [{"name": "clay_analyze", "arguments": {}}]}
    )
    assert result["isError"] is False, result
    batch_payload = _payload(result)
    assert batch_payload["completed"] == 1
    assert "objects" in _payload(batch_payload["results"][0])


def test_clay_analyze_refuses_an_out_of_range_tolerance() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)

    result = agent_clay.call(ctx, session, "clay_analyze", {"near": 100.0})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "near"


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


def test_the_five_declared_output_schemas_describe_what_those_tools_actually_return() -> None:
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

    mesh_schema = getattr(tools["clay_add_mesh"], "output_schema", None)
    assert mesh_schema is not None
    mesh_result = agent_clay.call(ctx, session, "clay_add_mesh", _TETRA_MESH_ARGS)
    mesh_structured = mesh_result.get("structuredContent") or {}
    assert mesh_structured, "clay_add_mesh answered with no structuredContent at all"
    # Exact for the same reason ``clay_add_primitive``'s is: ``closed`` and
    # ``findings`` are unconditional too, and both are declared on the
    # composed schema (see ``_mesh_row_output_schema``), not left off it.
    assert set(mesh_structured) == set(mesh_schema["properties"])

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

    analyze_schema = getattr(tools["clay_analyze"], "output_schema", None)
    assert analyze_schema is not None
    analyze_result = agent_clay.call(ctx, session, "clay_analyze", {})
    analyze_structured = analyze_result.get("structuredContent") or {}
    assert analyze_structured, "clay_analyze answered with no structuredContent at all"
    # A subset, like clay_diagnose's: ``truncated`` only appears when true,
    # and ``floating`` only for a whole-document call (this one -- no uids
    # were given).
    assert set(analyze_structured) <= set(analyze_schema["properties"])
    assert "objects" in analyze_schema["properties"]
    assert "pairs" in analyze_schema["properties"]


def test_no_declared_output_schema_demands_required_keys_because_a_refusal_shares_the_envelope() -> (  # noqa: E501
    None
):
    """None of the five declared schemas names a ``required`` list or sets
    ``additionalProperties: false`` -- proven alongside the reason itself: a
    refusal from one of these same tools really does put ``field`` in
    ``structuredContent`` -- and, since ``changed`` was added, nothing else
    beyond that -- which a ``required`` list on the success shape would make
    non-conforming. The exact equality below (not a subset check) is the
    point: it is what would catch an accidental extra key landing in this
    envelope, ``changed`` among them if its default ever drifted."""
    tools = {t.name: t for t in agent_clay.tools()}
    for name in (
        "clay_scene",
        "clay_add_primitive",
        "clay_add_mesh",
        "clay_diagnose",
        "clay_analyze",
    ):
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
    assert refusal["structuredContent"] == {
        "field": "generator",
        "changed": False,
        "recovery": "fix_arguments",
    }


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

    # ``clay_add_mesh`` composes the same shared row rather than copying it
    # -- every key the row schema declares must still be there, plus exactly
    # the two this tool alone answers with.
    mesh_schema = getattr(tools["clay_add_mesh"], "output_schema", None)
    assert mesh_schema is not None
    row_properties = agent_clay._object_row_output_schema()["properties"]
    assert row_properties.items() <= mesh_schema["properties"].items()
    assert set(mesh_schema["properties"]) - set(row_properties) == {"closed", "findings"}


# ==============================================================================
# C -- a refusal reports whether the document moved, and what to try next
# ==============================================================================


def test_every_refusal_says_whether_the_document_moved(svc) -> None:
    """The load-bearing test for this change, and it is empirical rather
    than a source scan. Walks every entry in ``agent_clay._HANDLERS``,
    reusing ``_NEEDS_A_TAB``/``_MINTS_A_TAB``/``_SESSION_ONLY`` and
    ``_SESSION_ONLY_ARGS`` exactly as
    ``test_every_tool_answers_with_structured_content_unless_its_reply_carries_a_picture``
    already does, rather than a second argument table. For every call that
    refuses, this captures the document's own history length, ``dirty`` flag
    and object count *before* and *after* the call, and proves a
    ``changed: false`` refusal really left all three untouched -- the part
    that makes this a gate on the document itself rather than a restatement
    of whatever the handler happened to report.

    Most of the calls in these tables are minimal on purpose (just enough to
    pass whatever a handler checks before it resolves a tab), so several of
    them succeed rather than refuse against a tab that already holds an
    object (``clay_scene``, ``clay_elements``,
    ``clay_diagnose``, ``clay_export`` with a real ``svc``, ``clay_undo``/
    ``clay_redo``, ``clay_batch``, ``clay_program``, all three ``_MINTS_A_TAB``
    creators, and every ``_SESSION_ONLY`` tool but ``clay_reference_get``
    naming a reference this session was never given). Those successes are
    skipped rather than
    asserted on either way, the same as that other exhaustive walk -- but the
    number of refusals this walk actually exercised is asserted with a hard
    floor, so a future regression that turned every refusal green could not
    make this test pass having proven nothing.
    """
    ctx = _Ctx(svc=svc)
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")  # a real, open tab with one object on it
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    doc = tab.doc

    covered = (
        {n for n, _ in _NEEDS_A_TAB}
        | {n for n, _ in _MINTS_A_TAB}
        | {n for n, _ in _ALSO_MINTS_A_TAB}
        | set(_SESSION_ONLY)
    )
    assert covered == set(agent_clay._HANDLERS)

    calls = list(_NEEDS_A_TAB) + list(_MINTS_A_TAB) + list(_ALSO_MINTS_A_TAB)
    calls += [(name, _SESSION_ONLY_ARGS[name]) for name in _SESSION_ONLY]

    refusals = 0
    for name, args in calls:
        witness_before = (len(doc.history), doc.dirty, len(doc.objects))
        result = agent_clay.call(ctx, session, name, args)
        witness_after = (len(doc.history), doc.dirty, len(doc.objects))
        if not result["isError"]:
            continue
        refusals += 1
        structured = result.get("structuredContent") or {}
        assert "changed" in structured, name
        assert isinstance(structured["changed"], bool), name
        if structured["changed"] is False:
            assert witness_after == witness_before, (name, witness_before, witness_after)

    # A floor, not a target -- see the docstring above for which of these
    # calls succeed rather than refuse against an already-open tab. Measured
    # at 13 refusals out of 28 calls on this tree.
    assert refusals >= 10


def test_every_recovery_a_refusal_names_is_in_the_vocabulary(svc) -> None:
    """Bidirectional, the way this file's own derivation gate already is
    (see the module docstring): every ``recovery`` value a refusal actually
    produces during the same walk as the test above is a member of
    :data:`agent_clay.RECOVERY`, and every member of that vocabulary is
    findable somewhere in ``src/warlock/`` -- a source scan is the only way
    to reach ``agent_host``'s own transport-level members (``"retry"``,
    ``"wait"``) from this file, which never calls those refusals directly.
    Both directions, or it is half a gate.
    """
    import pathlib

    import warlock

    ctx = _Ctx(svc=svc)
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")

    calls = list(_NEEDS_A_TAB) + list(_MINTS_A_TAB) + list(_ALSO_MINTS_A_TAB)
    calls += [(name, _SESSION_ONLY_ARGS[name]) for name in _SESSION_ONLY]

    seen: set[str] = set()
    for name, args in calls:
        result = agent_clay.call(ctx, session, name, args)
        if not result["isError"]:
            continue
        structured = result.get("structuredContent") or {}
        recovery = structured.get("recovery")
        if recovery is not None:
            seen.add(recovery)

    assert seen, "the walk produced no recovery value at all -- it proved nothing"
    assert seen <= agent_clay.RECOVERY

    src_root = pathlib.Path(warlock.__file__).resolve().parent
    all_text = "\n".join(p.read_text(encoding="utf-8") for p in src_root.rglob("*.py"))
    for member in agent_clay.RECOVERY:
        assert f'"{member}"' in all_text or f"'{member}'" in all_text, member


def test_every_refusal_that_names_a_field_also_names_how_to_fix_it(svc) -> None:
    """A refusal that names the argument it is unhappy with is already
    saying which one to change, so it must carry a ``recovery`` too -- and
    it does without any call site spelling one out, because :func:`fail`
    derives ``"fix_arguments"`` from ``field=`` itself. Walked over every
    handler rather than sampled, so a refusal added later cannot name a
    field and leave a client to guess.
    """
    ctx = _Ctx(svc=svc)
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")

    calls = list(_NEEDS_A_TAB) + list(_MINTS_A_TAB) + list(_ALSO_MINTS_A_TAB)
    calls += [(name, _SESSION_ONLY_ARGS[name]) for name in _SESSION_ONLY]

    named_a_field = 0
    for name, args in calls:
        result = agent_clay.call(ctx, session, name, args)
        if not result["isError"]:
            continue
        structured = result.get("structuredContent") or {}
        if not structured.get("field"):
            continue
        named_a_field += 1
        assert structured.get("recovery") in agent_clay.RECOVERY, (name, structured)

    assert named_a_field >= 8  # a floor, so this cannot pass having seen none


def test_a_stale_stamp_says_to_re_read_rather_than_to_fix_the_argument() -> None:
    """The one refusal that names a field and is deliberately *not*
    ``"fix_arguments"``: the stamp the client sent was well-formed and was
    true when it read it, so what is stale is its picture of the mesh. Its
    own message says to go and read the stamp again, and its ``recovery``
    says the same thing to a program."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")

    result = agent_clay.call(
        ctx,
        session,
        "clay_select_elements",
        {"uid": uid, "mode": "face", "faces": [0], "expect_stamp": 9999},
    )

    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["field"] == "expect_stamp"
    assert structured["recovery"] == "read_scene"
    assert structured["changed"] is False


def test_a_validation_refusal_tells_a_client_to_fix_its_arguments_and_a_missing_uid_to_re_read() -> (  # noqa: E501
    None
):
    """The two commonest refusal paths, proven through real calls rather
    than by calling ``_validate_vec3``/``_resolve_uid`` directly."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")

    malformed = agent_clay.call(
        ctx, session, "clay_transform", {"uid": uid, "translation": [1.0, 2.0]}
    )
    assert malformed["isError"] is True
    assert (malformed.get("structuredContent") or {}).get("recovery") == "fix_arguments"

    missing_uid = agent_clay.call(
        ctx, session, "clay_transform", {"uid": uid + 999, "translation": [1.0, 2.0, 3.0]}
    )
    assert missing_uid["isError"] is True
    assert (missing_uid.get("structuredContent") or {}).get("recovery") == "read_scene"


def test_a_batch_that_stopped_early_reports_that_the_document_did_move() -> None:
    """The one refusal where ``changed`` is ``true``, and the reason the
    field is not a constant: the cylinder before the bad generator name
    really was placed and really is still there, folded into the one undo
    step ``clay_batch`` keeps as its successful prefix."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")

    calls = [
        {"name": "clay_add_primitive", "arguments": {"generator": "cylinder"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "nope"}},
    ]
    result = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})
    assert result["isError"] is True
    payload = _payload(result)
    assert payload["stopped_at"] == 1
    assert payload["completed"] == 1
    assert payload.get("changed") is True
    assert (result.get("structuredContent") or {}).get("changed") is True


# --- the tool catalogue is fixed context an agent pays before its first call --


def test_the_tool_catalogue_stays_inside_the_context_budget_an_agent_pays_for_it() -> None:
    """The ``tools/list`` payload and the ``initialize`` instructions prose are
    context every agent session pays for once, up front, before it has made a
    single useful call -- and unlike a reply it asked for, it never chose this
    cost and cannot shrink it. Both halves are derived rather than hand-kept:
    ``agent_clay.tools()`` builds its schemas from ``primitives.GENERATORS``,
    ``presets.ASSEMBLIES`` and ``clay_ops.OPS`` (see this module's own
    docstring on the derivation gate), so the catalogue grows every time one
    of those registries does, with nobody at the call site deciding it should.
    Tranche 4's six new generators grew ``clay_add_primitive``'s schema alone
    to the largest of any tool here, and nothing noticed. The point of this
    pin is not "keep it small" -- it is that growing this budget becomes a
    decision someone makes (raise the ceiling, and say why, in this same
    commit) rather than a thing that happens as a side effect of an unrelated
    registry change.

    This builds the payload the way ``agent_host`` actually serves it at
    ``initialize``/``tools/list`` -- ``[*agent_clay.tools(), *
    agent_host._transport_tools()]`` (see ``agent_host._serve``'s
    ``tools=lambda: ...``) -- rather than ``agent_clay.tools()`` alone, and so
    imports ``agent_host`` the way two tests above already do. Leaving
    ``warlock_status`` out would undercount what a connecting agent is
    actually billed for by one whole tool; the honest number includes it.

    Measured on 2026-09-13, after ``clay_render`` grew a ``shading`` enum
    (six values, plus the description explaining what each one draws and the
    new ``object_id``/``compare``/``grid`` refusals): catalogue JSON 41,861
    chars + instructions 6,499 chars = 48,360 chars total (still 27 Clay
    tools plus ``warlock_status`` -- ``shading`` is a property on an
    existing tool, not a 28th one -- at ``rpc.tool_dict`` encoding).
    ``clay_render`` itself is now 2,883 chars, the previous measurement's
    entire 636-char headroom plus more, so the ceiling below is raised to
    48,500 -- just past this measurement, the same "minimal, and say why"
    rule the previous raise (47,364 of 48,000) already followed. The next
    tool that grows the catalogue at all will need to raise it again.

    A second growth the same day: ``clay_render``'s description gained one
    more sentence naming the compare header's new ``silhouette`` block (shape
    IoU, aspect, a null reading's ``reason``) -- catalogue JSON 42,136 chars +
    instructions 6,499 chars = 48,635 chars total, over the 48,500 raised
    above by 135. ``clay_render`` itself is now 3,158 chars. No schema
    changed (``silhouette`` is a reply field, not an argument), so this is the
    description alone; the ceiling is raised to 48,700, again just past the
    measurement.

    A 28th tool the same class of growth as tranche 4's six new generators:
    ``clay_program`` (2026-09-13) is a new registry-independent tool -- its
    own schema and grammar-card description are 3,065 chars on their own --
    and ``instructions()`` gained a paragraph on when to reach for it over
    ``clay_batch`` plus the "two exceptions" -> "three exceptions" edit,
    growing from 6,499 to 7,279 chars. Catalogue JSON 45,217 chars +
    instructions 7,279 chars = 52,496 chars total, well over the 48,700
    ceiling above -- a whole new tool, not an unnoticed drift, so the ceiling
    moves with it rather than being defended against it. Raised to 52,600,
    again just past the measurement.

    The same day, ``clay_program``'s four ``LIVE_KINDS`` (move/turn/
    scale_by/assert) went from compiling to a refused placeholder to
    actually running: its grammar-card description grew a paragraph naming
    those four steps and the ``FACTS`` an assert condition may call
    (lo/hi/size/center/count/exists/touches/grounded/floating/volume), and
    ``instructions()`` gained one sentence in its own clay_program
    paragraph. Catalogue JSON 45,961 chars + instructions 7,565 chars =
    53,526 chars total, over the 52,600 ceiling above by 926 -- again a
    whole grammar growing, not drift. Raised to 53,600, just past this
    measurement.

    2026-09-13, same day, a different registry: ``clay_add_figure``'s own
    description gained a "Parts, each prefixed by name_prefix: ..." sentence
    naming every part of every figure preset (``presets.ASSEMBLIES``), so a
    training run or a live agent is told what a figure's own parts are
    called instead of guessing (run A of the Clay-assistant fine-tune, 2026-
    09-12, showed 15 refusals of the shape ``no object named '...'``, nine
    of them creatures-family guesses at a generated figure's own part names
    -- ``hound_Beak``, ``t_Shank.R``, ``s_Tail 01`` -- that this sentence
    would have made unnecessary). Catalogue JSON 47,377 chars + instructions
    7,565 chars = 54,942 chars total, over the 53,600 ceiling above by 1,342
    -- a whole new catalogue sentence, not drift. Raised to 55,000, just
    past this measurement.
    """
    from warlock.mcp import rpc
    from warlock.studio import agent_host

    CEILING = 55_000

    tools = [*agent_clay.tools(), *agent_host._transport_tools()]
    tool_jsons = [rpc.tool_dict(t) for t in tools]
    catalogue = json.dumps({"tools": tool_jsons})
    instructions = agent_clay.instructions()
    total = len(catalogue) + len(instructions)

    sizes = sorted(
        ((len(json.dumps(tj)), tj.get("name", "?")) for tj in tool_jsons), reverse=True
    )
    biggest = ", ".join(f"{name}={size}" for size, name in sizes[:5])

    assert total <= CEILING, (
        f"the tools/list catalogue ({len(catalogue)} chars) plus the "
        f"initialize instructions ({len(instructions)} chars) now total "
        f"{total} chars, over the {CEILING}-char budget an agent pays before "
        f"its first useful call. Largest tool schemas by size: {biggest}. If "
        f"this growth is deliberate, raise CEILING in this test and say why "
        f"in the same commit; if it is not, find what grew unnoticed among "
        f"the tools above."
    )


def test_a_params_value_is_held_to_the_shape_its_generator_default_declares() -> None:
    """``pyramid`` given a list for ``base`` is a refusal that names the key,
    not a crash.

    Fails today: the wire schema says a param value is ``number |
    array-of-numbers | array-of-arrays`` for every key of every generator,
    and nothing checked which of the three *this* key wants -- so a list
    reached ``primitives.pyramid``'s ``float(base)`` and came back through
    ``call()``'s generic backstop as "failed unexpectedly; see the log",
    with a ``TypeError`` traceback beside it. The same hole ran the other
    way for ``box``'s ``size``: a bare number, or two numbers instead of
    three, raised out of the generator just as unhelpfully.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)

    bad = agent_clay.call(
        ctx,
        session,
        "clay_add_primitive",
        {"generator": "pyramid", "params": {"base": [1.0, 1.0, 1.0]}},
    )
    assert bad["isError"] is True
    assert bad["structuredContent"]["field"] == "params"
    text = bad["content"][0]["text"]
    assert "base" in text and "pyramid" in text
    assert "failed unexpectedly" not in text

    for value in (1.0, [1.0, 1.0], [[1.0, 1.0, 1.0]]):
        wrong = agent_clay.call(
            ctx, session, "clay_add_primitive", {"generator": "box", "params": {"size": value}}
        )
        assert wrong["isError"] is True, value
        assert "size" in wrong["content"][0]["text"]

    # A row array's *width* is fixed by the default too, even though how many
    # rows it has is the caller's to choose -- an over-wide row used to have
    # its extra column silently dropped.
    ragged = agent_clay.call(
        ctx,
        session,
        "clay_add_primitive",
        {"generator": "lathe", "params": {"profile": [[0.5, 0.0, 9.0], [0.4, 0.5, 9.0]]}},
    )
    assert ragged["isError"] is True
    assert "profile" in ragged["content"][0]["text"]

    # And the same gate on the other door, where the refusal must say which
    # object it is talking about.
    on_set = agent_clay.call(ctx, session, "clay_set_params", {"uid": uid, "params": {"size": 2.0}})
    assert on_set["isError"] is True
    assert f"uid {uid}" in on_set["content"][0]["text"]

    # What the shapes really are still passes, both doors.
    good = agent_clay.call(
        ctx, session, "clay_add_primitive", {"generator": "pyramid", "params": {"base": 2.0}}
    )
    assert good["isError"] is False, good
    fine = agent_clay.call(
        ctx, session, "clay_set_params", {"uid": uid, "params": {"size": [2.0, 1.0, 2.0]}}
    )
    assert fine["isError"] is False, fine


def test_diagnose_reports_a_copy_family_that_no_longer_agrees_on_its_material() -> None:
    """Place a box, array it, then paint only the original: the copies keep
    the material they were made with, and that is the right behaviour -- a
    copy is an independent object. What was wrong is that it had no symptom
    at all short of a render, which is the one thing an agent over a pipe
    cannot read cheaply.

    Fails today: ``clay_diagnose`` measures meshes and nothing else, so a
    whole-document call on a half-painted array answered "clean" for every
    object in it.
    """
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)

    select = agent_clay.call(ctx, session, "clay_select", {"uids": [uid]})
    assert select["isError"] is False, select
    array = agent_clay.call(
        ctx, session, "clay_op", {"name": "array-linear", "params": {"count": 3}}
    )
    assert array["isError"] is False, array

    before = agent_clay.call(ctx, session, "clay_diagnose", {})
    assert "scene" not in _payload(before), "a uniform family says nothing"

    painted = agent_clay.call(
        ctx, session, "clay_material", {"uids": [uid], "name": "Red", "color": [1.0, 0.0, 0.0]}
    )
    assert painted["isError"] is False, painted

    after = _payload(agent_clay.call(ctx, session, "clay_diagnose", {}))
    assert "scene" in after, "a half-painted family has something to say"
    row = next(r for r in after["scene"] if r["kind"] == "copies_disagree_on_material")
    assert uid in row["uids"]
    assert len(row["uids"]) == 3

    # A single named object is asked about itself only, so a finding about how
    # three objects relate has no business in that answer.
    assert "scene" not in _payload(agent_clay.call(ctx, session, "clay_diagnose", {"uid": uid}))
