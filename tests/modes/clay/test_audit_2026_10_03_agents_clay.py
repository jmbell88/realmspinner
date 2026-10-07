"""Regression tests for the 2026-10-03 audit's agents-06..09 (Clay agent surface)."""

from __future__ import annotations

import contextlib
import io

from PIL import Image

from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import refs as agent_refs
from realmspinner.studio.modes.clay.agent import transcript as agent_transcript

from .test_agent_clay import _Ctx


def test_a_program_assert_with_a_non_finite_axis_rolls_the_whole_program_back_and_says_so() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx,
        session,
        "clay_program",
        {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"add": {"generator": "box", "id": "b"}},
                {"assert": {"condition": "size(a, 1e400) > 0"}},
            ]
        },
    )
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "failed unexpectedly" not in text
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert {o.name for o in tab.doc.objects} == set()
    assert _payload_or_none(result).get("changed") is False


def _payload_or_none(result: dict) -> dict:
    return result.get("structuredContent") or {}


def test_fold_run_rolls_back_when_an_entry_raises(monkeypatch) -> None:
    """Any exception escaping an entry, not just the axis one, leaves the
    document at its mark."""
    from realmspinner.studio.modes.clay.agent import tools_batch

    ctx = _Ctx()
    session = agent_clay.Session()
    agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    doc = tab.doc
    before = doc.history.head
    calls = {"n": 0}
    real = tools_batch._run_entry

    def flaky(c, s, d, entry):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return real(c, s, d, entry)

    monkeypatch.setattr(tools_batch, "_run_entry", flaky)
    entry = ("clay_add_primitive", {"generator": "box"})
    with contextlib.suppress(RuntimeError):
        tools_batch._fold_run(ctx, session, doc, [entry, entry], rollback=True, label=None)
    assert doc.history.head == before


def _png(size, color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_overlay_scales_a_small_reference_up_to_fill_the_cell_like_the_render() -> None:
    out = agent_refs.overlay(_png((64, 64)), _png((256, 256), (255, 255, 255)), 0.0, size=256)
    with Image.open(io.BytesIO(out)) as img:
        # alpha 0.0 is the reference alone: it must fill the cell, not sit as a
        # 64 px patch in the middle of white.
        assert img.convert("RGB").getpixel((5, 5)) == (255, 0, 0)


def test_remap_rewrites_the_parent_and_focus_uids_a_replay_would_otherwise_leave_stale() -> None:
    m = {1: 11, 2: 22, 3: 33}
    assert agent_transcript.remap({"uid": 1, "parent": 2}, m, "t", 1)["parent"] == 22
    assert agent_transcript.remap({"parent": None, "uid": 1}, m, "t", 1)["parent"] is None
    assert agent_transcript.remap({"focus": [1, 3]}, m, "t", 1)["focus"] == [11, 33]
    # a numeric target of another tool (clay_uv texel density) is not a uid
    assert agent_transcript.remap({"uid": 1, "target": 1024}, m, "t", 1, tool="clay_uv") == {
        "uid": 11,
        "target": 1024,
    }
    entry = {"tool": "clay_uv", "arguments": {"uid": 1, "parent": 2}}
    out = agent_transcript.remap({"calls": [entry]}, m, "t", 1)
    assert out["calls"][0]["arguments"]["uid"] == 11
    assert out["calls"][0]["arguments"]["parent"] == 22
    live = {"steps": [{"move": {"uid": {"uid": 2}, "by": [0, 1, 0]}}]}
    assert agent_transcript.remap(live, m, "t", 1)["steps"][0]["move"]["uid"] == {"uid": 22}


def test_reference_add_refuses_a_file_name_outside_the_four_it_documents_before_touching_the_disk(
    svc, tmp_path, monkeypatch
) -> None:
    from realmspinner.service import files as svc_files

    ctx = _Ctx(svc=svc)
    session = agent_clay.Session()
    job_id = svc.store.create("image", None, {}, stage="reference", status="done")
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    touched: list[str] = []
    real = svc_files.ready
    monkeypatch.setattr(
        svc_files, "ready", lambda job, d, name: touched.append(name) or real(job, d, name)
    )
    result = agent_clay.call(
        ctx,
        session,
        "clay_reference_add",
        {"name": "r", "job_id": job_id, "file": "../../etc/hosts"},
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "file"
    assert touched == []
