"""The 2026-10-09 audit's Clay mode findings: clay-03, 13, 14, 15, 42 and 43."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pygame
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize
from realmspinner.studio import palette
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops

from .test_clay_mode import FakeCtx
from .test_clay_palette import _clay, _ctx


class _Done:
    def __init__(self, key: str, result: Any = None) -> None:
        self.key = key
        self.result = result


def _key(key: int, mod: int = 0) -> Any:
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod)


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


def _sidecar_ctx(tmp_path: Path) -> tuple[FakeCtx, str]:
    job_id = "0123456789ab"
    job = tmp_path / job_id
    job.mkdir()
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.cone(), generator="cone"))
    (job / "build.rblk").write_bytes(serialize.rblk_bytes(doc))
    ctx = FakeCtx(SimpleNamespace(job_dir=lambda _id: job))
    return ctx, job_id


# --- clay-03 / clay-15 ---------------------------------------------------------


def test_open_in_clay_on_a_sidecar_titles_the_tab_with_the_jobs_name(tmp_path: Path) -> None:
    ctx, job_id = _sidecar_ctx(tmp_path)
    clay_mode.edit_asset_in_clay(ctx, {"id": job_id, "name": "Job One"})
    assert ctx.result["title"] == "Job One"


def test_a_sidecar_past_the_slow_threshold_asks_before_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, job_id = _sidecar_ctx(tmp_path)
    clay_mode.edit_asset_in_clay(ctx, {"id": job_id, "name": "Job One"})
    assert ctx.result["triangles"] > 0
    monkeypatch.setattr(clay_mode, "SLOW_TRIANGLES", ctx.result["triangles"] - 1)
    clay_mode.on_task_done(ctx, _Done(f"clay-import:{job_id}", ctx.result))
    assert ctx.confirms.pending is not None, "a big sidecar opened without the slow-edit confirm"


# --- clay-13 -------------------------------------------------------------------


def test_export_waits_for_a_live_drag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clay_mode, "camera_of", lambda ctx, tab: None)
    monkeypatch.setattr(clay_mode, "build_asset", lambda *a, **k: "job")
    ctx = _ctx(with_view=True)
    ctx.clay_view.dragging = True
    ctx.clay_view.grabbing = False
    before = list(ctx.submitted)

    assert clay_mode.handle_key(ctx, _key(pygame.K_e, pygame.KMOD_CTRL)) is True
    assert ctx.submitted == before, "Ctrl+E started an export under a live drag"

    rows = _clay(ctx)
    for name in ("library", "glb", "obj", "screenshot"):
        row = rows[f"clay-export:{name}"]
        assert row.enabled(ctx) is False, f"{name} row is live during a drag"
        assert row.why

    ctx.clay_view.dragging = False
    assert _clay(ctx)["clay-export:library"].enabled(ctx) is True


# --- clay-14 -------------------------------------------------------------------


def _live_op(ctx: Any) -> Any:
    doc = ctx.state.clay.active.doc
    for op in clay_ops.OPS:
        if op.name == "duplicate" and doc.element_mode in op.modes and op.enabled(doc):
            return op
    raise AssertionError("no enabled op")


def test_palette_op_rows_are_shut_during_a_live_drag() -> None:
    ctx = _ctx(with_view=True)
    ctx.clay_view.dragging = True
    ctx.clay_view.grabbing = False
    op = _live_op(ctx)
    doc = ctx.state.clay.active.doc

    row = _clay(ctx)[f"clay-op:{op.name}"]
    assert row.enabled(ctx) is False, "an op row stayed live under a drag"
    assert row.why

    head = doc.history.head
    clay_mode.fire_op(ctx, doc, op)
    assert doc.history.head == head, "fire_op ran an op under a live drag"

    ctx.clay_view.dragging = False
    assert _clay(ctx)[f"clay-op:{op.name}"].enabled(ctx) is True
    assert palette is not None


# --- clay-42 / clay-43 ---------------------------------------------------------


def test_sibling_mtl_ignores_a_trailing_comment_on_the_mtllib_line(tmp_path: Path) -> None:
    (tmp_path / "m.mtl").write_text("newmtl red\nKd 1 0 0\n", encoding="utf-8")
    obj = tmp_path / "m.obj"
    data = b"mtllib m.mtl # the colours\nv 0 0 0\n"
    obj.write_bytes(data)
    assert (clay_mode._sibling_mtl(obj, data) or "").splitlines() == ["newmtl red", "Kd 1 0 0"]


def test_sibling_mtl_read_is_bounded_in_one_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "m.mtl").write_text("newmtl red\n", encoding="utf-8")
    obj = tmp_path / "m.obj"
    data = b"mtllib m.mtl\n"
    obj.write_bytes(data)
    calls: list[int] = []
    real = clay_mode.sizeguard.read_bytes_within_ceiling

    def spy(path: Path, ceiling: int, **kw: Any) -> bytes:
        calls.append(ceiling)
        return real(path, ceiling, **kw)

    monkeypatch.setattr(clay_mode.sizeguard, "read_bytes_within_ceiling", spy)
    assert (clay_mode._sibling_mtl(obj, data) or "").splitlines() == ["newmtl red"]
    assert calls == [clay_mode.MAX_MTL_BYTES]
    # Past the ceiling is a missing library, not an error.
    (tmp_path / "m.mtl").write_bytes(b"x" * (clay_mode.MAX_MTL_BYTES + 1))
    assert clay_mode._sibling_mtl(obj, data) is None
