"""The 2026-10-14 audit, fixer F47: the Blender rig worker, the children's failure sentences,
Familiar's llama child, the matting child, and the docstrings that named a pin that is gone.

Every test is named for the claim it pins. The Blender ones need ``bpy`` (skipped without it,
like every sibling in this directory) and run the real ``op_rig`` against a real Blender, because
each of those defects was invisible to the fake ``bpy`` the older rig tests use.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner import fetch, models
from realmspinner.pipelines import llama as llama_mod
from realmspinner.pipelines.llama import LlamaServer

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "realmspinner"


# --- the Blender rig worker (pipelines-children-02, -04, -05) ----------------


def _weight_in_blender(doc, work: Path):
    """What "Weight with Blender" runs, minus the queue. -> (op_rig's result, the imported rig)."""
    bpy = pytest.importorskip("bpy")
    from realmspinner.kernels.geom3d import glbwrite
    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import glbimport
    from realmspinner.kernels.rig import blender_spec, clayrig, store
    from realmspinner.pipelines import blender_worker, postprocess

    model = work / "model.glb"
    merged = clayrig.weight_source_doc(doc)
    model.write_bytes(glbwrite.write_glb(bd.to_model(merged, skinned=False)))
    moved = postprocess.normalize_glb(model, None)
    place = np.diag([moved["scale"]] * 3 + [1.0])
    place[:3, 3] = moved["translation"]
    skeleton = clayrig.heat_skeleton(doc, place)
    spec = blender_spec.rig_spec(
        work, "blank", skeleton["bones"], skeleton="custom", root="Hip",
        mirror_pairs=skeleton["mirror_pairs"],
    )
    result = blender_worker.op_rig(bpy, spec)
    store.finalize_rig(work)
    imported, _ = glbimport.glb_to_claydoc_report((work / "rig.glb").read_bytes(), name="rig")
    return result, imported


def _doc_with(*shapes):
    """Hip and Spine bones plus whatever shapes ``shapes`` adds (each ``(name, kwargs)``)."""
    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import primitives as bp

    doc = bd.ClayDoc()
    ids: dict[str, int] = {}

    def add(name, **kw):
        kw.setdefault("mesh", bp.box())
        ids[name] = doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, **kw)).uid

    add("Hip", bone=True, translation=(0.0, 0.5, 0.0))
    add("Spine", bone=True, parent=ids["Hip"], translation=(0.0, 0.5, 0.0))
    for name, kw in shapes:
        add(name, **kw)
    return doc


def test_an_envelope_fallback_that_weights_nothing_fails_the_rig_instead_of_publishing_an_unskinned_one(  # noqa: E501
    tmp_path,
):
    """The 2026-10-14 audit, finding pipelines-children-02: a torso and two one-sided cards defeat
    bone-heat, and the envelope bind that followed reached no vertex at all -- ``op_rig`` still said
    ``ok: True, weighting: "envelope"`` and wrote a ``rig.glb`` with ``skins: 0``."""
    pytest.importorskip("bpy")
    from realmspinner.kernels.mesh import primitives as bp

    doc = _doc_with(
        ("Torso", {"translation": (0.0, 1.0, 0.0), "scale": (0.6, 1.4, 0.4)}),
        ("Card", {"mesh": bp.plane((1.0, 1.0)), "translation": (1.5, 2.5, 0.0)}),
        ("Card2", {"mesh": bp.plane((1.0, 1.0)), "translation": (-1.5, 2.5, 0.0)}),
    )
    with pytest.raises(RuntimeError, match="weight"):
        _weight_in_blender(doc, tmp_path)


def test_weighting_an_inside_out_shell_does_not_flip_its_faces(tmp_path):
    """The 2026-10-14 audit, finding pipelines-children-04: ``_weld`` runs
    ``normals_make_consistent(inside=False)`` and the welded mesh is what reached the export, so a
    room modelled with its walls facing in came back with every wall facing out."""
    pytest.importorskip("bpy")
    from realmspinner.kernels.mesh import mesh as bm
    from realmspinner.kernels.mesh import ops_topo
    from realmspinner.kernels.mesh import primitives as bp
    from realmspinner.kernels.mesh.elements import ElementSel

    inside_out, _ = ops_topo.flip_normals(bp.box(), ElementSel())
    doc = _doc_with(
        ("Room", {"mesh": inside_out, "translation": (0.0, 1.0, 0.0), "scale": (3.0, 2.0, 3.0)}),
        ("Crate", {"translation": (0.0, 0.3, 0.0), "scale": (0.4, 0.4, 0.4)}),
    )
    result, imported = _weight_in_blender(doc, tmp_path)
    assert result["weighting"] == "automatic-welded", (
        "the scene no longer reaches the welded success path this test is about"
    )
    body = next(o for o in imported.objects if not o.bone and bm.face_count(o.mesh))
    mesh = body.mesh
    assert mesh.weights is not None and float(mesh.weights.sum()) > 0.0, (
        "restoring the facing must not cost the skin its weights"
    )
    normals, positions = bm.face_normals(mesh), mesh.positions
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    centre = (lo + hi) / 2
    inward = total = 0
    for face in range(bm.face_count(mesh)):
        verts = positions[mesh.loops[mesh.starts[face]:mesh.starts[face + 1]]]
        on_wall = any(
            np.ptp(verts[:, a]) < 1e-5
            and (abs(verts[0, a] - lo[a]) < 1e-5 or abs(verts[0, a] - hi[a]) < 1e-5)
            for a in range(3)
        )
        if on_wall:
            total += 1
            inward += float(np.dot(normals[face], verts.mean(axis=0) - centre)) < 0
    assert total > 0
    assert inward == total, f"{total - inward} of {total} inward-facing walls came back facing out"


def test_the_manual_says_weighting_triangulates_the_mesh():
    """The 2026-10-14 audit, finding pipelines-children-05: Blender's glTF export triangulates, so
    a weighted Clay mesh loses its quads, and the chapter that describes Weight with Blender never
    said so. The sentence it carries is what this pins."""
    text = (ROOT / "docs" / "manual" / "27-clay.md").read_text(encoding="utf-8")
    start = text.index("## Rigging with Blender weights")
    section = re.split(r"\n## ", text[start:], maxsplit=1)[0]
    assert re.search(r"triangulat", section, re.IGNORECASE), (
        "27-clay.md does not tell the reader that weighting turns the mesh's quads into triangles"
    )


# --- the children's failure sentences (pipelines-children-07, -06) ------------


def test_describe_failure_names_an_http_status_not_a_network_problem():
    """The 2026-10-14 audit, finding pipelines-children-07: ``_looks_like_a_network_error``
    matched the class name ``HTTPError``, so GitHub's 403/429 rate limit on the update check told
    the user to check their firewall."""
    import io
    import urllib.error

    from realmspinner.pipelines.download import describe_failure

    def failure(code: int, reason: str) -> str:
        err = urllib.error.HTTPError("https://api.github.com/x", code, reason, {}, io.BytesIO(b""))
        return describe_failure(err, resume_note="Nothing was kept.")

    for code, reason in ((403, "rate limit exceeded"), (429, "Too Many Requests")):
        said = failure(code, reason)
        assert str(code) in said, said
        assert "firewall" not in said and "antivirus" not in said and "online" not in said, said
        assert "rate" in said.lower() or "limit" in said.lower(), said
    assert "404" in failure(404, "Not Found")
    assert "firewall" not in failure(404, "Not Found")
    assert "503" in failure(503, "Service Unavailable")
    # A transport reset is still the network's fault, which is what the sentence is for.
    reset = OSError("An existing connection was forcibly closed by the remote host")
    reset.winerror = 10054
    assert "firewall" in describe_failure(reset)


def _gated_repo_error():
    httpx = pytest.importorskip("httpx")
    errors = pytest.importorskip("huggingface_hub.errors")
    request = httpx.Request("GET", "https://huggingface.co/api/models/x/y")
    response = httpx.Response(401, request=request)
    return errors.GatedRepoError("401 Client Error. Cannot access gated repo", response=response)


def test_a_gated_repo_is_not_retried_as_a_lost_connection(monkeypatch):
    """The 2026-10-14 audit, finding pipelines-children-06: a gated or mistyped repo is the same
    answer every time, but ``_TERMINAL`` held only ``download.Refusal``, so it was retried four
    times with 2+8+20 s of sleeps, each announced as "connection lost"."""
    # Importing the worker sets HF_HUB_OFFLINE=0; monkeypatch restores whatever was there.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from realmspinner.pipelines import fetch_worker

    slept: list[float] = []
    emitted: list[dict] = []
    monkeypatch.setattr(fetch_worker.time, "sleep", slept.append)
    monkeypatch.setattr(fetch_worker, "_emit", lambda **p: emitted.append(p))
    calls: list[int] = []
    error = _gated_repo_error()

    def run():
        calls.append(1)
        raise error

    with pytest.raises(type(error)):
        fetch_worker._with_retries(run, {})
    assert len(calls) == 1, f"a gated repo was attempted {len(calls)} times"
    assert slept == [] and not any("connection lost" in e.get("label", "") for e in emitted)


def test_a_dropped_connection_is_still_retried(monkeypatch):
    """The guard on the fix above: the reset the retry exists for must keep being retried."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from realmspinner.pipelines import fetch_worker

    monkeypatch.setattr(fetch_worker.time, "sleep", lambda _s: None)
    monkeypatch.setattr(fetch_worker, "_emit", lambda **_p: None)
    calls: list[int] = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("An existing connection was forcibly closed by the remote host")

    fetch_worker._with_retries(flaky, {"retries": 3})
    assert len(calls) == 2


# --- Familiar's llama child (familiar-host-08, -12, -09) ----------------------


def _chat(tmp_path, **kwargs) -> LlamaServer:
    return LlamaServer(
        tmp_path / "engine" / "llama-server.exe",
        tmp_path / "models" / "familiar" / models.FAMILIAR_GGUF_FILE,
        17972,
        key_dir=tmp_path / "keys",
        log_path=tmp_path / "familiar.log",
        **kwargs,
    )


def _arm(srv: LlamaServer, monkeypatch, popen) -> None:
    """Every check before the spawn made to pass, and ``popen`` standing in for the spawn."""
    srv._resolve_exe().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_exe().write_bytes(b"")
    srv._resolve_weights().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_weights().write_bytes(b"")
    monkeypatch.setattr(
        llama_mod.fetch, "verify_manifest",
        lambda dest, **kw: fetch.Verification(dest=dest, status=fetch.VERIFY_UNKNOWN),
    )
    monkeypatch.setattr(srv, "_check_vram", lambda: None)
    monkeypatch.setattr(llama_mod, "_port_in_use", lambda port: False)
    monkeypatch.setattr(llama_mod.winjob, "assign", lambda pid: None)
    monkeypatch.setattr(llama_mod.winjob, "track", lambda pid, name: None)
    monkeypatch.setattr(srv, "_claim_port", lambda pid: None)
    monkeypatch.setattr(srv, "_pump", lambda: None)
    monkeypatch.setattr(llama_mod.subprocess, "Popen", popen)


@pytest.mark.asyncio
async def test_a_popen_failure_releases_the_log_and_key_and_counts_as_a_start_failure(
    tmp_path, monkeypatch
):
    """The 2026-10-14 audit, finding familiar-host-08: when ``Popen`` itself refuses (Defender
    block, WinError 216/225) the log handle and key file were already made and never released, no
    failure was counted, and the raw ``OSError`` reached the user as "did not answer in time"."""
    srv = _chat(tmp_path)
    spawns: list[int] = []

    def refuse(*a, **k):
        spawns.append(1)
        raise OSError(216, "This version of llama-server.exe is not compatible")

    _arm(srv, monkeypatch, refuse)
    with pytest.raises(RuntimeError, match="llama-server"):
        await srv.ensure_started()
    assert srv._logfh is None, "the log handle was leaked"
    assert not list((tmp_path / "keys").glob("*.key")), "the key file was left behind"
    assert srv._start_failures == 1
    # Counted, so the next message is refused with the backoff sentence rather than re-spawning.
    with pytest.raises(RuntimeError, match="refusing to respawn"):
        await srv.ensure_started()
    assert len(spawns) == 1


def _write_manifest(dest: Path, files: dict[str, bytes]) -> None:
    digests = {name: {"sha256": hashlib.sha256(data).hexdigest()} for name, data in files.items()}
    fetch.manifest_path(dest).write_text(
        json.dumps({"repos": {"acme/familiar": {"digests": digests}}}), encoding="utf-8"
    )


def test_chat_start_ignores_a_missing_optional_row_file_in_the_shared_manifest(tmp_path):
    """The 2026-10-14 audit, finding familiar-host-12: the chat child verified every file the
    shared ``models/familiar`` manifest names, so a deleted ~160 MB mmproj (or a damaged
    retrieval file) refused chat although the argv code runs without the mmproj."""
    srv = _chat(tmp_path)
    weights = srv._resolve_weights()
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"chat weights")
    (tmp_path / "engine").mkdir()
    (weights.parent / models.FAMILIAR_EMBED_GGUF_FILE).write_bytes(b"torn retrieval file")
    _write_manifest(
        weights.parent,
        {
            weights.name: b"chat weights",
            models.FAMILIAR_MMPROJ_FILE: b"projector, since deleted",
            models.FAMILIAR_EMBED_GGUF_FILE: b"the retrieval file as it was downloaded",
        },
    )
    srv._check_manifest()  # must not raise


def test_chat_start_still_refuses_damaged_files_it_will_load(tmp_path):
    """The other half: the files the chat child *does* load keep the strict check."""
    weights_dir = tmp_path / "models" / "familiar"
    weights_dir.mkdir(parents=True)
    (tmp_path / "engine").mkdir()
    mmproj = weights_dir / models.FAMILIAR_MMPROJ_FILE

    srv = _chat(tmp_path)
    srv._resolve_weights().write_bytes(b"damaged weights")
    _write_manifest(weights_dir, {srv._resolve_weights().name: b"chat weights"})
    with pytest.raises(RuntimeError, match="manifest verification"):
        srv._check_manifest()

    srv = _chat(tmp_path, mmproj_path=mmproj)
    srv._resolve_weights().write_bytes(b"chat weights")
    mmproj.write_bytes(b"torn projector")
    _write_manifest(
        weights_dir,
        {srv._resolve_weights().name: b"chat weights", mmproj.name: b"the projector"},
    )
    with pytest.raises(RuntimeError, match="manifest verification"):
        srv._check_manifest()


@pytest.mark.asyncio
async def test_a_cancelled_start_does_not_leave_a_child_that_the_next_start_trusts(
    tmp_path, monkeypatch
):
    """The 2026-10-14 audit, finding familiar-host-09: a start cancelled mid health-poll (the
    caller's 20 s timeout against a 120 s load) left a spawned, never-verified child, and every
    later ``ensure_started`` returned at once because the child was ``running`` -- so the next
    embedding request hit a server still loading."""
    srv = _chat(tmp_path)
    fake_proc = SimpleNamespace(pid=4242, returncode=None, poll=lambda: None)
    _arm(srv, monkeypatch, lambda *a, **k: fake_proc)
    health: list[str] = []
    status = {"code": 503}

    async def fake_get(self, url, **kwargs):
        health.append(url)
        import httpx

        return httpx.Response(status["code"])

    monkeypatch.setattr(llama_mod.httpx.AsyncClient, "get", fake_get)
    # The stop() of a fake process has nothing to wait for.
    monkeypatch.setattr(srv, "_release_port_claim", lambda: None)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(srv.ensure_started(), timeout=0.4)
    assert srv.running, "the cancelled start's child is expected to be left loading"
    polled = len(health)
    assert polled > 0

    # Still 503: the next start must not report a healthy server.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(srv.ensure_started(), timeout=0.4)
    assert len(health) > polled, "ensure_started trusted a child nobody had seen healthy"

    # Once it does answer, the next start waits for that answer and then returns.
    status["code"] = 200
    before = len(health)
    await asyncio.wait_for(srv.ensure_started(), timeout=5)
    assert len(health) > before


# --- the matting child (pipelines-image-04) -----------------------------------


def test_a_failed_spawn_of_the_matting_child_is_not_memoized_as_a_load_failure(
    tmp_path, monkeypatch
):
    """The 2026-10-14 audit, finding pipelines-image-04: a transient failure to *spawn* the child
    (WinError 1455, the paging file) became ``stage="load"``, so every later ``mask()`` used the
    corner fill without a second attempt and doctor showed a red "last load failed" for a
    checkpoint that was fine."""
    from realmspinner.pipelines import matting

    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"x")
    config = SimpleNamespace(t2i_model_root=tmp_path)

    from PIL import Image

    attempts: list[int] = []

    def no_spawn(key):
        attempts.append(1)
        raise OSError(1455, "The paging file is too small for this operation to complete")

    matting.unload()
    try:
        monkeypatch.setattr(matting, "_start_child", no_spawn)
        for _ in range(2):
            _mask, source = matting.mask(Image.new("RGB", (64, 64), (200, 200, 200)), config)
            assert source == "flood"
        assert len(attempts) == 2, "the second mask() never tried to spawn the child again"
        assert matting.last_error() is None
    finally:
        matting.unload()


# --- docstring currency (docs-A-B-16, familiar-host-14) -----------------------


def test_the_birefnet_annotation_counts_the_network_children_correctly():
    """The 2026-10-14 audit, finding docs-A-B-16: the vendored annotation said the fetch child was
    "the only process allowed online"; there are three user-initiated children."""
    text = (SRC / "pipelines" / "birefnet" / "modeling.py").read_text(encoding="utf-8")
    flat = re.sub(r"\s*\r?\n#\s*", " ", text)  # the annotation wraps across comment lines
    assert "the only process allowed online" not in flat
    assert "one of three" in flat


def test_public_docs_do_not_cite_gitignored_dev_files():
    """The 2026-10-14 audit, finding docs-A-B-16: ``docs/create-quality.md`` cited a measurement
    under ``dev/``, which a public clone does not carry."""
    public = [ROOT / n for n in ("README.md", "INSTALL.md", "CONTRIBUTING.md", "SECURITY.md")]
    public += sorted((ROOT / "docs").glob("*.md"))
    offenders = [
        f"{path.relative_to(ROOT)}: {m.group(0)}"
        for path in public
        if path.is_file()
        for m in re.finditer(r"`dev/[^`\s]+`", path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_create_quality_is_one_of_the_documents_whose_manual_links_are_walked():
    """The same finding: ``docs/create-quality.md`` was not in the external-link test's SOURCES."""
    text = (ROOT / "tests" / "test_external_doc_links.py").read_text(encoding="utf-8")
    assert "create-quality.md" in text


def test_familiar_docstrings_name_only_the_current_pin():
    """The 2026-10-14 audit, finding familiar-host-14: the Clay-build gate and the process header
    still described Qwen3-VL-4B Q8_0, and ``doors.py`` cited a Poser function that is gone."""
    for relative in (
        "pipelines/llama.py",
        "service/familiar.py",
        "studio/assistant/doors.py",
    ):
        text = (SRC / relative).read_text(encoding="utf-8")
        assert "Qwen" not in text, f"{relative} still names the retired Qwen pin"
        assert "poser_mode" not in text, f"{relative} still cites the removed Poser mode"


def test_the_assistant_attachment_note_names_the_current_pin():
    """The same finding for ``studio/assistant/ui.py``'s image-token note, which still said the
    512 px measurement was Qwen3-VL's 258 tokens (Gemma 4 12B measures 123, ceiling 150)."""
    text = (SRC / "studio" / "assistant" / "ui.py").read_text(encoding="utf-8")
    assert "cost 258 tokens" not in text


# --- joints="measured" has no writer left (poser-rig-02) ----------------------


def test_every_joint_source_in_the_rig_spec_has_a_production_writer():
    """The 2026-10-14 audit, finding poser-rig-02: Troupe's ``_jobs_create`` and
    its service module were the only writers of ``joints="measured"`` / ``rig_joints``, and
    both were deleted with the character pipeline. A joint source nothing can ask for is dead code
    that still looked live (and had tests keeping it green)."""
    from realmspinner.kernels.rig import blender_spec

    plumbing = {
        "pipelines/jointfit.py",
        "kernels/rig/blender_spec.py",
        "pipelines/blender_worker.py",
        "_q_rig.py",
        "_q_jobs.py",
    }
    writers = ""
    for path in SRC.rglob("*.py"):
        if path.relative_to(SRC).as_posix() in plumbing:
            continue
        writers += path.read_text(encoding="utf-8")
    # ``JOINT_SOURCES`` itself was deleted with the chain (C2, same audit): a ratchet
    # that stays armed if a joint source is ever added back without a writer.
    for source in getattr(blender_spec, "JOINT_SOURCES", ()):
        if source == "template":
            continue  # the default: nothing has to ask for it
        assert re.search(rf"joints\s*=\s*[\"']{source}[\"']|rig_joints", writers), (
            f"joints={source!r} is a rig-spec joint source with no production writer"
        )
