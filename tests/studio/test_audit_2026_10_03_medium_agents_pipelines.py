"""The 2026-10-03 audit's Medium findings for the MCP host and the mesh pipelines.

agents-10 and agents-11 (the bridge's set-up fallbacks) were already closed in
``tests/mcp/test_bridge.py`` by the Criticals-and-Highs commit, so they are not
repeated here. What is: agents-14 (a timed-out read is re-run, never replayed
stale), pipelines-12 (a GLB cut short inside its BIN chunk is refused),
pipelines-13 (a "match the mesh" re-texture is bounded), pipelines-14
(components counted through non-manifold edges) and pipelines-18 (the Mesh
quality block reads the key the report writes).
"""

from __future__ import annotations

import inspect
import json
import struct
import threading
import time

import numpy as np
import pytest
import trimesh

from realmspinner import meshreport
from realmspinner.pipelines import retexture
from realmspinner.pipelines import trellis as trellis_mod
from realmspinner.studio import agent_host, widgets
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

from .test_agent_host import (
    RUNNING_WAIT,
    WAIT,
    _bare_host,
    _pump_loop,
    _shorten_call_timeout,
)

# ---------------------------------------------------------------- agents-14


def _time_out_then_finish(monkeypatch, tool: str, args: dict, replies: list[str]):
    """Drive ``tool`` so its first call outruns the timeout while running and
    finishes afterwards; return (host, calls, session, runs, release_and_wait).

    ``replies[n]`` is what the n-th real run answers with, so a test can tell a
    re-run (a fresh answer) from a replay (the remembered one)."""
    host = _bare_host()
    calls = agent_host._Calls()
    _shorten_call_timeout(monkeypatch, host, timeout=RUNNING_WAIT)
    runs = {"n": 0}
    started = threading.Event()
    release = threading.Event()

    def fake_call(ctx, session, name, arguments):  # noqa: ARG001
        n = runs["n"]
        runs["n"] += 1
        started.set()
        if n == 0:
            assert release.wait(WAIT), "release never came"
        return {"content": [{"type": "text", "text": replies[min(n, len(replies) - 1)]}]}

    monkeypatch.setattr(agent_clay, "call", fake_call)
    stop = threading.Event()
    pumper = threading.Thread(target=_pump_loop, args=(host, stop), daemon=True)
    pumper.start()
    session = agent_clay.Session()
    first = threading.Thread(
        target=lambda: host._call(session, calls, tool, args), daemon=True
    )
    first.start()
    assert started.wait(WAIT), "the job never started running"
    first.join(timeout=WAIT)
    assert not first.is_alive()
    release.set()
    op = calls.get("op-1")
    deadline = time.monotonic() + WAIT
    while (
        op.status(host._job_lock) not in (agent_host.DONE, agent_host.RAISED)
        and time.monotonic() < deadline
    ):
        time.sleep(0.005)

    def finish() -> None:
        stop.set()
        pumper.join(timeout=WAIT)

    return host, calls, session, runs, finish


def test_a_timed_out_read_is_re_run_not_replayed_after_the_document_changed(monkeypatch) -> None:
    """The 2026-10-03 audit, agents-14: a pure read that outran the timeout was
    handed back from memory on the retry, as a picture of a document that had
    since changed -- and the bridge's own timeout text tells the agent to
    re-read with clay_scene."""
    host, calls, session, runs, finish = _time_out_then_finish(
        monkeypatch, "clay_scene", {}, ["old scene", "new scene"]
    )
    try:
        retry = host._call(session, calls, "clay_scene", {})
    finally:
        finish()
    assert runs["n"] == 2
    assert retry["content"][0]["text"] == "new scene"


@pytest.mark.parametrize(
    "tool",
    [
        "clay_scene",
        "clay_measure",
        "clay_elements",
    ],
)
def test_every_pure_read_the_audit_names_is_exempt_from_replay(tool: str) -> None:
    assert tool in agent_host.REPLAY_EXEMPT_READS


def test_every_replay_exempt_read_is_a_real_tool() -> None:
    """A renamed tool must not leave a dead name here that quietly re-enables
    stale replay for its successor."""
    real = {t.name for t in agent_clay.tools()}
    assert real >= agent_host.REPLAY_EXEMPT_READS


def test_a_timed_out_edit_is_still_replayed_not_run_twice(monkeypatch) -> None:
    """The exemption is for reads only: re-running a timed-out placement would
    put a second box in the document, which is what replay exists to prevent."""
    host, calls, session, runs, finish = _time_out_then_finish(
        monkeypatch, "clay_add_primitive", {"kind": "box"}, ["placed"]
    )
    try:
        retry = host._call(session, calls, "clay_add_primitive", {"kind": "box"})
    finally:
        finish()
    assert runs["n"] == 1
    assert retry["structuredContent"]["replayed"] is True


# -------------------------------------------------------------- pipelines-12


def _glb_with_bin(binary: bytes) -> bytes:
    gltf = json.dumps({"asset": {"version": "2.0"}, "meshes": [{"primitives": []}]})
    payload = gltf.encode()
    payload += b" " * (-len(payload) % 4)
    chunk = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    chunk += struct.pack("<II", len(binary), 0x004E4942) + binary
    return struct.pack("<III", 0x46546C67, 2, 12 + len(chunk)) + chunk


def test_validate_glb_accepts_a_whole_glb() -> None:
    trellis_mod._validate_glb(_glb_with_bin(b"\x00" * 64))


def test_generate_rejects_a_glb_truncated_inside_its_bin_chunk() -> None:
    """The 2026-10-03 audit, pipelines-12: only the 12-byte header and the JSON
    chunk were parsed, so a body cut short inside BIN was written onto
    source.glb and the job went done with a mesh trimesh could not load."""
    whole = _glb_with_bin(b"\x00" * 64)
    with pytest.raises(RuntimeError, match="invalid GLB"):
        trellis_mod._validate_glb(whole[:-20])


def test_generate_rejects_a_glb_whose_header_length_disagrees_with_the_body() -> None:
    whole = _glb_with_bin(b"\x00" * 64)
    with pytest.raises(RuntimeError, match="invalid GLB"):
        trellis_mod._validate_glb(whole + b"\x00" * 8)


# -------------------------------------------------------------- pipelines-13


def test_a_match_the_mesh_retexture_is_bounded_by_the_largest_texture_size(
    tmp_path, monkeypatch
) -> None:
    """The 2026-10-03 audit, pipelines-13: with texture_size unset a re-texture
    baked at whatever square albedo the mesh carried, so a 4096 atlas put about
    8.4 GiB of ten-view float32 arrays in the app process, past the
    TEXTURE_SIZES the door enforces for an explicit size."""
    monkeypatch.setattr(retexture, "atlas_size", lambda path: 4096)
    assert retexture.match_the_mesh_size(tmp_path / "model.glb") == max(retexture.TEXTURE_SIZES)
    monkeypatch.setattr(retexture, "atlas_size", lambda path: 1024)
    assert retexture.match_the_mesh_size(tmp_path / "model.glb") == 1024
    monkeypatch.setattr(retexture, "atlas_size", lambda path: None)
    assert retexture.match_the_mesh_size(tmp_path / "model.glb") is None


def test_the_retexture_job_asks_the_bounded_size_not_the_raw_atlas() -> None:
    from realmspinner import _q_sprite

    source = inspect.getsource(_q_sprite)
    assert "retexture.match_the_mesh_size" in source
    assert "retexture.atlas_size" not in source


# -------------------------------------------------------------- pipelines-14


def test_components_counts_faces_joined_through_a_non_manifold_edge_as_one() -> None:
    """The 2026-10-03 audit, pipelines-14: three fins on one edge reported 3
    components on trimesh's face_adjacency, which omits any edge shared by more
    than two faces."""
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0, 1], [0.5, -1, 0]], dtype=float
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components, _boundary, nonmanifold = meshreport._topology(trimesh, np, mesh)
    assert nonmanifold == 1
    assert components == 1


def test_components_still_counts_separate_shells_apart() -> None:
    a = trimesh.creation.box(extents=(1, 1, 1))
    b = trimesh.creation.box(extents=(1, 1, 1))
    b.apply_translation((5, 0, 0))
    mesh = trimesh.util.concatenate([a, b])
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    assert meshreport._topology(trimesh, np, mesh)[0] == 2


def test_a_lone_floating_triangle_is_its_own_component() -> None:
    box = trimesh.creation.box(extents=(1, 1, 1))
    vertices = np.vstack([box.vertices, [[9, 9, 9], [10, 9, 9], [9, 10, 9]]])
    faces = np.vstack([box.faces, [[8, 9, 10]]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert meshreport._topology(trimesh, np, mesh)[0] == 2


# -------------------------------------------------------------- pipelines-18


def test_mesh_quality_lists_the_reports_material_count(monkeypatch) -> None:
    """The 2026-10-03 audit, pipelines-18: the pane read report["materials"]
    while meshreport.build writes "material_count", so the line never drew."""
    from realmspinner.studio.panes import inspector

    lines: list[str] = []
    monkeypatch.setattr(widgets, "header", lambda label, **kwargs: True)
    monkeypatch.setattr(widgets, "muted", lines.append)
    monkeypatch.setattr(widgets, "quality_badge", lambda job: None)
    job = {"params": {"mesh_report": {"triangles": 12, "material_count": 3}}}
    inspector._quality(None, job)
    assert "materials: 3" in lines


def test_the_pane_and_the_report_agree_on_the_key(tmp_path) -> None:
    path = tmp_path / "m.glb"
    trimesh.Scene(trimesh.creation.box(extents=(1, 1, 1))).export(path)
    assert "material_count" in meshreport.build(path)
