"""Regression tests for the 2026-10-03 audit's Medium Clay findings
clay-20 through clay-26 (the agent tool surface and the mesh document)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import measure
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.mcp import rpc
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay


class _Cache:
    def invalidate(self) -> None:
        pass


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.svc = None
        self.cache = _Cache()
        self.toasts: list[tuple[str, str]] = []
        self.settings = SimpleNamespace()

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _text(result: dict) -> str:
    return result["content"][0]["text"]


def _payload(result: dict) -> Any:
    return json.loads(_text(result))


def _session_with_boxes(n: int = 2) -> tuple[_Ctx, agent_clay.Session]:
    ctx = _Ctx()
    session = agent_clay.Session()
    for i in range(n):
        r = agent_clay.call(
            ctx, session, "clay_add_primitive",
            {"generator": "box", "translation": [3.0 * i, 0, 0]},
        )
        assert r["isError"] is False, r
    return ctx, session


def _doc(ctx: _Ctx, session: agent_clay.Session) -> Any:
    return clay_mode.ensure(ctx).get(session.tab_uid).doc


def _framed(result: dict) -> int:
    """The bytes a result occupies on the wire, the way the host encodes it."""
    return len(json.dumps(result, separators=(",", ":")).encode("utf-8"))


# --- clay-20 ---------------------------------------------------------------


def test_a_batch_refused_for_reply_size_reports_changed_true_when_its_edits_were_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, session = _session_with_boxes(1)
    doc = _doc(ctx, session)
    before = len(doc.objects)
    monkeypatch.setattr(rpc, "MAX_FRAME", 10)

    result = agent_clay.call(
        ctx, session, "clay_batch",
        {"calls": [{"name": "clay_add_primitive", "arguments": {"generator": "box"}}] * 2},
    )

    assert result["isError"] is True
    assert "too large" in _text(result)
    assert len(doc.objects) == before + 2, "sanity: the batch's edits were kept"
    assert result["structuredContent"]["changed"] is True, result["structuredContent"]
    assert "kept" in _text(result), "the refusal must say not to repeat the edits"


def test_a_program_refused_for_reply_size_reports_changed_true_when_its_edits_were_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, session = _session_with_boxes(1)
    doc = _doc(ctx, session)
    before = len(doc.objects)
    monkeypatch.setattr(rpc, "MAX_FRAME", 10)

    result = agent_clay.call(
        ctx, session, "clay_program",
        {"steps": [{"add": {"generator": "box"}}, {"add": {"generator": "box"}}]},
    )

    assert result["isError"] is True
    assert "too large" in _text(result)
    assert len(doc.objects) == before + 2, "sanity: the program's edits were kept"
    assert result["structuredContent"]["changed"] is True, result["structuredContent"]


# --- clay-21 ---------------------------------------------------------------


def test_the_frame_guard_counts_the_escaped_text_twin_of_a_batch_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, session = _session_with_boxes(6)
    calls = [{"name": "clay_scene", "arguments": {}}] * 3

    measured = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})
    assert measured["isError"] is False, measured
    old_estimate = 2 * len(json.dumps(measured["structuredContent"]))
    wire = _framed(measured)
    assert wire > old_estimate, (wire, old_estimate)  # the 2x estimate under-counts

    # A ceiling the old estimate clears and the real frame does not.
    monkeypatch.setattr(rpc, "MAX_FRAME", wire - 1 + 64 * 1024)
    assert old_estimate <= rpc.MAX_FRAME - 64 * 1024
    result = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})

    assert result["isError"] is True, "a reply past MAX_FRAME was sent after its edit was kept"
    assert "too large" in _text(result)


# --- clay-22 ---------------------------------------------------------------


def test_the_scene_over_budget_refusal_names_a_narrowing_the_tool_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, session = _session_with_boxes(5)
    scene = next(t for t in agent_clay.tools() if t.name == "clay_scene")
    accepted = set(scene.schema["properties"])

    full = agent_clay.call(ctx, session, "clay_scene", {})
    assert full["isError"] is False
    # Room for two object rows and no more.
    one_row = len(json.dumps(_payload(full)["objects"][0]))
    monkeypatch.setattr(rpc, "MAX_FRAME", 64 * 1024 + 2 * 2 * one_row + 2000)
    refused = agent_clay.call(ctx, session, "clay_scene", {})
    assert refused["isError"] is True
    message = _text(refused)

    # The refusal may only name arguments clay_scene really has ...
    assert accepted, "clay_scene must accept some narrowing argument"
    assert any(name in message for name in accepted), message
    for suggestion in ("a single uid", "fewer objects"):
        assert suggestion not in message, message
    # ... and following it must work.
    paged = agent_clay.call(ctx, session, "clay_scene", {"offset": 0, "limit": 2})
    assert paged["isError"] is False, paged
    body = _payload(paged)
    assert len(body["objects"]) == 2
    assert body["object_count"] == 5
    rest = _payload(agent_clay.call(ctx, session, "clay_scene", {"offset": 2, "limit": 2}))
    assert [o["uid"] for o in rest["objects"]] == [
        o["uid"] for o in _payload(full)["objects"][2:4]
    ]


# --- clay-23 ---------------------------------------------------------------


def test_clay_elements_for_the_whole_selection_is_refused_when_the_reply_would_not_fit_a_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, session = _session_with_boxes(4)
    doc = _doc(ctx, session)
    r = agent_clay.call(ctx, session, "clay_select", {"uids": [o.uid for o in doc.objects]})
    assert r["isError"] is False, r
    r = agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    assert r["isError"] is False, r
    r = agent_clay.call(ctx, session, "clay_op", {"name": "select-all"})
    assert r["isError"] is False, r
    assert len(doc.element_sel) == 4, "sanity: every object has a selection"

    ok = agent_clay.call(ctx, session, "clay_elements", {"kind": "face", "limit": 4096})
    assert ok["isError"] is False, ok
    monkeypatch.setattr(rpc, "MAX_FRAME", 64 * 1024 + 100)

    result = agent_clay.call(ctx, session, "clay_elements", {"kind": "face", "limit": 4096})

    assert result["isError"] is True, "a reply the bridge cannot take reached the wire"
    assert "too large" in _text(result)
    assert result["structuredContent"]["changed"] is False


# --- clay-24 ---------------------------------------------------------------


def _scene_is_strict_json(ctx: _Ctx, session: agent_clay.Session) -> bool:
    scene = agent_clay.call(ctx, session, "clay_scene", {})
    try:
        json.loads(_text(scene), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    except ValueError:
        return False
    return True


def test_an_extreme_transform_is_refused_rather_than_putting_infinity_or_nan_in_the_document(
) -> None:
    ctx, session = _session_with_boxes(2)
    doc = _doc(ctx, session)
    a, b = (o.uid for o in doc.objects)
    before = [o.trs() for o in doc.objects]

    refused = []
    for args in (
        {"translation": [1e308, 0, 0]},
        {"scale": [1e-320, 1e-320, 1e-320]},
        {"scale": [1e308, 1, 1]},
    ):
        r = agent_clay.call(ctx, session, "clay_transform", {"uid": a, **args})
        refused.append(r)
        assert r["isError"] is True, (args, r)
        assert r["structuredContent"]["changed"] is False
        assert r["structuredContent"]["field"] in ("translation", "scale")

    for o, trs in zip(doc.objects, before, strict=True):
        assert all(np.array_equal(x, y) for x, y in zip(o.trs(), trs, strict=True))
    assert _scene_is_strict_json(ctx, session)
    del b


def test_an_extreme_transform_is_refused_by_every_tool_that_places_something(
) -> None:
    ctx, session = _session_with_boxes(1)
    for tool, args in (
        ("clay_add_primitive", {"generator": "box", "translation": [1e308, 0, 0]}),
        ("clay_add_primitive", {"generator": "box", "scale": [1e-320, 1, 1]}),
        ("clay_add_figure", {"key": "humanoid", "translation": [1e308, 0, 0]}),
    ):
        count = len(_doc(ctx, session).objects)
        r = agent_clay.call(ctx, session, tool, args)
        assert r["isError"] is True, (tool, args)
        assert len(_doc(ctx, session).objects) == count


def test_reparenting_under_a_denormal_scale_refuses_rather_than_writing_nan() -> None:
    doc = bd.ClayDoc()
    parent = doc.add_object(bd.Obj(uid=bd.new_uid(), name="p", mesh=bp.box()))
    child = doc.add_object(bd.Obj(uid=bd.new_uid(), name="c", mesh=bp.box()))
    # The document's own door accepts a finite, nonzero, denormal scale.
    doc.set_transform(parent.uid, scale=[1e-320, 1e-320, 1e-320])
    before = tuple(np.array(v, copy=True) for v in child.trs())

    with pytest.raises(el.OpError):
        doc.set_parent(child.uid, parent.uid, keep_world=True)

    assert child.parent is None
    for kept, now in zip(before, child.trs(), strict=True):
        assert np.array_equal(kept, now)
        assert np.isfinite(now).all()


# --- clay-25 ---------------------------------------------------------------


def _open_box_session() -> tuple[_Ctx, agent_clay.Session, int]:
    ctx = _Ctx()
    session = agent_clay.Session()
    positions = [
        [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1],
        [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1],
    ]
    faces = [[0, 3, 2, 1], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]  # no top
    added = agent_clay.call(
        ctx, session, "clay_add_mesh", {"positions": positions, "faces": faces}
    )
    assert added["isError"] is False, added
    return ctx, session, _payload(added)["uid"]


def test_clay_measure_volume_of_an_open_mesh_does_not_answer_a_number() -> None:
    ctx, session, uid = _open_box_session()

    analysed = _payload(agent_clay.call(ctx, session, "clay_analyze", {"uids": [uid]}))
    assert analysed["objects"][0]["volume"] is None, "sanity: analyze says open"

    measured = agent_clay.call(ctx, session, "clay_measure", {"kind": "volume", "uid": uid})
    body = measured["structuredContent"] if measured["isError"] else _payload(measured)
    assert body.get("value") is None, body
    assert body.get("closed") is False or measured["isError"], body


def test_clay_measure_volume_of_a_closed_mesh_still_answers() -> None:
    ctx, session = _session_with_boxes(1)
    uid = _doc(ctx, session).objects[0].uid
    body = _payload(agent_clay.call(ctx, session, "clay_measure", {"kind": "volume", "uid": uid}))
    assert body["value"] == pytest.approx(1.0, abs=1e-3)


def test_the_measure_kernel_has_a_volume_that_declines_an_open_mesh() -> None:
    assert measure.volume_if_closed(bp.box()) == pytest.approx(1.0, abs=1e-5)
    open_box = bp.box()
    # Drop one face: the same mesh, now with a hole.
    from realmspinner.kernels.mesh import mesh as bm

    starts, loops = open_box.starts, open_box.loops
    faces = [loops[starts[i]:starts[i + 1]].tolist() for i in range(bm.face_count(open_box) - 1)]
    open_mesh = bm.from_faces(open_box.positions, faces)
    assert measure.volume_if_closed(open_mesh) is None


# --- clay-26 ---------------------------------------------------------------


def test_a_batch_opening_with_a_creator_restarts_a_session_whose_tab_was_closed() -> None:
    ctx, session = _session_with_boxes(1)
    state = clay_mode.ensure(ctx)
    assert state.close(session.tab_uid)

    result = agent_clay.call(
        ctx, session, "clay_batch",
        {"calls": [{"name": "clay_add_primitive", "arguments": {"generator": "box"}}]},
    )

    assert result["isError"] is False, _text(result)
    assert state.get(session.tab_uid) is not None
    assert len(_doc(ctx, session).objects) == 1


def test_a_batch_opening_with_a_non_creator_still_refuses_on_a_closed_tab() -> None:
    ctx, session = _session_with_boxes(1)
    assert clay_mode.ensure(ctx).close(session.tab_uid)

    result = agent_clay.call(
        ctx, session, "clay_batch", {"calls": [{"name": "clay_scene", "arguments": {}}]}
    )

    assert result["isError"] is True
    assert result["structuredContent"]["recovery"] == "start_document"
