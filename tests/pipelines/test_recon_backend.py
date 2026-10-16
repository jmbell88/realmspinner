"""The reconstruction-backend seam: a protocol, a request type, a capability gate.

Nothing here changes what TRELLIS.2 does -- every claim is that the seam is *only*
a seam: the one call forwards to the ``generate`` it always had, the one backend
declares what it always was (one image in), and admission refuses what it cannot
take on the field a caller would set.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from realmspinner.pipelines import recon, trellis
from realmspinner.service import Invalid
from realmspinner.service import validation as svc_validation


def _server(tmp_path) -> trellis.TrellisServer:
    return trellis.TrellisServer(lambda: tmp_path / "trellis-server.exe", tmp_path, 59999)


def test_trellis_declares_single_view_only():
    assert trellis.TrellisServer.capabilities == recon.TRELLIS_CAPS
    assert recon.TRELLIS_CAPS.max_views == 1
    assert trellis.TrellisServer.name == recon.DEFAULT_BACKEND == "trellis.2"
    assert recon.active_capabilities(object()) is recon.TRELLIS_CAPS
    assert recon.active_backend_name(object()) == "trellis.2"


def test_trellis_server_satisfies_recon_backend(tmp_path):
    assert isinstance(_server(tmp_path), recon.ReconBackend)


def test_the_test_double_satisfies_it_too(tmp_path):
    """The queue's tests run against ``FakeTrellisServer``; if it drifted from the
    protocol they would be testing a backend the queue does not hold."""
    from conftest import FakeTrellisServer

    assert isinstance(FakeTrellisServer(), recon.ReconBackend)


def test_validation_refuses_multi_view_request_on_views_field():
    with pytest.raises(Invalid) as caught:
        svc_validation.check_recon_views(recon.TRELLIS_CAPS, 2)
    assert caught.value.field == "views"
    assert "1 image" in str(caught.value)
    svc_validation.check_recon_views(recon.TRELLIS_CAPS, 1)
    svc_validation.check_recon_views(recon.ReconCapabilities(max_views=4), 4)
    for bad in (0, -1, True, "2", None):
        with pytest.raises(Invalid) as caught:
            svc_validation.check_recon_views(recon.TRELLIS_CAPS, bad)
        assert caught.value.field == "views"


async def test_reconstruct_refuses_a_second_image_rather_than_ignoring_it(tmp_path):
    request = recon.ReconRequest(
        images=(tmp_path / "a.png", tmp_path / "b.png"), output=tmp_path / "out.glb"
    )
    with pytest.raises(ValueError, match="takes 1 image, got 2"):
        await _server(tmp_path).reconstruct(request)


class _Capture:
    """A stand-in for ``httpx.AsyncClient`` that records the one POST and answers
    with a body ``_validate_glb`` is told to accept."""

    posts: list = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, *, files, data):
        _Capture.posts.append((method, url, files["image"][0], dict(data)))
        return _Response()


class _Response:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_bytes(self):
        yield b"glb-bytes"


async def test_reconstruct_posts_identical_form_to_generate(tmp_path, monkeypatch):
    monkeypatch.setattr(trellis.httpx, "AsyncClient", _Capture)
    monkeypatch.setattr(trellis, "_validate_glb", lambda data: None)
    server = _server(tmp_path)

    async def started() -> None:
        return None

    monkeypatch.setattr(server, "ensure_started", started)
    image = tmp_path / "reference.png"
    image.write_bytes(b"png")
    _Capture.posts = []

    await server.generate(
        image, tmp_path / "direct.glb", seed=7, resolution=1536, bg_removal="birefnet"
    )
    await server.reconstruct(
        recon.ReconRequest(
            images=(image,), output=tmp_path / "via.glb", seed=7, resolution=1536,
            bg_removal="birefnet",
        )
    )
    direct, via = _Capture.posts
    assert direct == via
    assert direct[3] == {"seed": "7", "resolution": "1536", "bg_removal": "birefnet"}
    assert (tmp_path / "via.glb").read_bytes() == (tmp_path / "direct.glb").read_bytes()


async def test_reconstruct_omits_bg_removal_exactly_when_generate_does(tmp_path, monkeypatch):
    monkeypatch.setattr(trellis.httpx, "AsyncClient", _Capture)
    monkeypatch.setattr(trellis, "_validate_glb", lambda data: None)
    server = _server(tmp_path)

    async def started() -> None:
        return None

    monkeypatch.setattr(server, "ensure_started", started)
    image = tmp_path / "reference.png"
    image.write_bytes(b"png")
    _Capture.posts = []
    await server.reconstruct(recon.ReconRequest(images=(image,), output=tmp_path / "o.glb"))
    assert "bg_removal" not in _Capture.posts[0][3]


def test_the_queue_reaches_its_backend_only_through_reconstruct():
    """``generate`` is TRELLIS.2's own method, not part of the protocol; a call to
    it from the queue would be the seam leaking."""
    import realmspinner._q_generate as q_generate
    import realmspinner.queue as queue_mod

    for module in (q_generate, queue_mod):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                target = node.func
                if (
                    target.attr == "generate"
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "trellis"
                ):
                    pytest.fail(f"{module.__name__} calls trellis.generate directly")


def test_the_worker_holds_its_backend_as_the_protocol():
    import realmspinner.queue as queue_mod

    assert "ReconBackend" in inspect.getsource(queue_mod.Worker.__init__)


def test_a_manifest_records_the_backend_and_an_older_one_reads_as_trellis(tmp_path):
    from realmspinner.bench import manifest, recipe, suite
    from realmspinner.config import Config

    config = Config(data_dir=tmp_path / "assets", bench_dir=tmp_path / "bench")
    doc = manifest.build_manifest(
        config, suite.load("create-v1"), recipe.load("sdxl-cfg-raw"),
        stage="model", items=["mesh-lantern"], seeds=[42], started="t",
    )
    assert doc["recon_backend"] == "trellis.2"
    older = {k: v for k, v in doc.items() if k != "recon_backend"}
    assert manifest.compare_manifests(older, doc) == []
    other = {**doc, "recon_backend": "multiview.1"}
    assert any("recon backend" in line for line in manifest.compare_manifests(doc, other))
    with pytest.raises(manifest.NotResumable, match="recon backend: trellis.2 -> multiview.1"):
        manifest.assert_resumable(older, other)


def test_recon_imports_nothing_outside_the_standard_library():
    tree = ast.parse(Path(recon.__file__).read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
    assert roots <= {"__future__", "dataclasses", "pathlib", "typing"}


async def test_generate_sends_faces_only_when_set(tmp_path, monkeypatch):
    """``faces`` is the engine's QEM face target, a per-request multipart field.

    Sent only when set, ``bg_removal``'s rule: an engine without ``--faces`` is not
    handed a field it may answer with a 400, and ``None`` leaves the exe's own ~300k.
    """
    monkeypatch.setattr(trellis.httpx, "AsyncClient", _Capture)
    monkeypatch.setattr(trellis, "_validate_glb", lambda data: None)
    server = _server(tmp_path)

    async def started() -> None:
        return None

    monkeypatch.setattr(server, "ensure_started", started)
    image = tmp_path / "reference.png"
    image.write_bytes(b"png")
    _Capture.posts = []

    await server.generate(image, tmp_path / "plain.glb", seed=7, resolution=1024)
    await server.generate(image, tmp_path / "budget.glb", seed=7, resolution=1024, faces=5000)
    plain, budget = (post[3] for post in _Capture.posts)
    assert "faces" not in plain
    assert budget["faces"] == "5000"


async def test_reconstruct_forwards_the_requests_faces_to_generate(tmp_path, monkeypatch):
    monkeypatch.setattr(trellis.httpx, "AsyncClient", _Capture)
    monkeypatch.setattr(trellis, "_validate_glb", lambda data: None)
    server = _server(tmp_path)

    async def started() -> None:
        return None

    monkeypatch.setattr(server, "ensure_started", started)
    image = tmp_path / "reference.png"
    image.write_bytes(b"png")
    _Capture.posts = []
    await server.reconstruct(
        recon.ReconRequest(images=(image,), output=tmp_path / "o.glb", faces=2000)
    )
    assert _Capture.posts[0][3]["faces"] == "2000"
