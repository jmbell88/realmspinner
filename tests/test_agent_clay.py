"""What Clay's agent tool surface (``studio/agent_clay.py``) promises to hold.

Three claims are pinned here, each stated in that module's own docstring.

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
from types import SimpleNamespace
from typing import Any

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
    ("clay_add_primitive", {"generator": "box"}),
    ("clay_add_figure", {"key": sorted(presets.ASSEMBLIES)[0]}),
    ("clay_transform", {}),
    ("clay_set_params", {}),
    ("clay_material", {}),
    ("clay_boolean", {}),
    ("clay_select", {}),
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


def test_every_tool_is_covered_by_the_dead_tab_and_session_only_lists() -> None:
    assert {n for n, _ in _NEEDS_A_TAB} | set(_SESSION_ONLY) == set(agent_clay._HANDLERS)


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


def test_clay_delete_deletes_objects_even_in_face_mode() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    tab.doc.element_mode = "face"  # nothing is selected in this mode

    result = agent_clay.call(ctx, session, "clay_delete", {"uids": [uid]})
    assert result["isError"] is False, result
    assert len(tab.doc.objects) == 0


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
