"""``mason_assets.AssetSource``: the host half of ``mason.refs.GeometrySource``.

Headless throughout -- no GL, no imgui. A fake ``ctx`` records ``submit``
calls instead of running them, so the async half (a ``LibraryRef``'s parse) is
driven deterministically by calling ``on_task_done`` by hand, exactly the way
``main.py``'s real task-collection loop would once a ``TaskRunner`` result
lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from warlock.kernels.geom3d import gltf
from warlock.studio import mason_assets
from warlock.studio.mason import refs as mason_refs


class _Done:
    def __init__(self, key: str, tag: Any, result: Any = None, error: BaseException | None = None):
        self.key = key
        self.tag = tag
        self.result = result
        self.error = error


class _Config:
    def __init__(self, root: Path) -> None:
        self.root = root

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id


class _Svc:
    def __init__(self, root: Path) -> None:
        self.config = _Config(root)


class _Ctx:
    """Records submissions instead of running them -- see the module docstring."""

    def __init__(self, root: Path) -> None:
        self.svc = _Svc(root)
        self.submitted: list[tuple[str, Any]] = []

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        if any(k == key for k, _ in self.submitted):
            return False
        self.submitted.append((key, (fn, args, kwargs, tag)))
        return True

    def run(self, key: str) -> tuple[Any, BaseException | None]:
        """Actually run one recorded submission's function -- the test's stand
        in for the thread pool -- and return (result, error)."""
        for k, (fn, args, kwargs, _tag) in self.submitted:
            if k == key:
                try:
                    return fn(*args, **kwargs), None
                except Exception as exc:  # noqa: BLE001 - mirrors TaskRunner's own catch-all
                    return None, exc
        raise KeyError(key)

    def tag_for(self, key: str) -> Any:
        for k, (_fn, _args, _kwargs, tag) in self.submitted:
            if k == key:
                return tag
        raise KeyError(key)


def _box_ref() -> mason_refs.PrimitiveRef:
    return mason_refs.primitive_ref("box", {"size": (1.0, 1.0, 1.0)})


def test_a_primitive_ref_resolves_synchronously() -> None:
    ctx = _Ctx(Path("."))
    source = mason_assets.ensure(ctx)
    prims = source.primitives(_box_ref())
    assert len(prims) >= 1
    assert prims[0].positions.shape[1] == 3


def test_the_same_primitive_ref_twice_is_one_build(monkeypatch: pytest.MonkeyPatch) -> None:
    from warlock.kernels.mesh import primitives as clay_primitives

    calls = {"n": 0}
    real_box = clay_primitives.GENERATORS["box"][1]

    def counting_box(**kwargs: Any):
        calls["n"] += 1
        return real_box(**kwargs)

    defaults = clay_primitives.GENERATORS["box"][0]
    monkeypatch.setitem(clay_primitives.GENERATORS, "box", (defaults, counting_box))

    ctx = _Ctx(Path("."))
    source = mason_assets.ensure(ctx)
    ref = _box_ref()
    source.primitives(ref)
    source.primitives(ref)
    assert calls["n"] == 1


def test_resolving_a_primitive_synchronously_does_not_move_rev() -> None:
    ctx = _Ctx(Path("."))
    source = mason_assets.ensure(ctx)
    before = source.rev
    source.primitives(_box_ref())
    assert source.rev == before


def test_adopting_a_library_parse_moves_rev() -> None:
    ctx = _Ctx(Path("."))
    source = mason_assets.ensure(ctx)
    ref = mason_refs.LibraryRef(job_id="job1", artifact="model.glb")
    before = source.rev

    node = gltf.Node(mesh=0)
    prim = gltf.Primitive(
        positions=np.zeros((3, 3), dtype="f4"),
        indices=np.array([0, 1, 2], dtype="u4"),
    )
    model = gltf.Model([node], [0], [[prim]], [])

    assert source.primitives(ref) == []
    task_key = f"{mason_assets.TASK_PREFIX}job1:model.glb"
    # ``result`` is what ``_start_library.run`` now hands back -- already
    # baked, per the 2026-09-15 audit's mason-02 (baking moved off the frame
    # thread onto the task thread).
    done = _Done(task_key, ctx.tag_for(task_key), result=mason_assets._bake_model(model))
    claimed = mason_assets.on_task_done(ctx, done)
    assert claimed is True
    assert source.rev == before + 1
    assert source.primitives(ref) != []


def test_an_unknown_generator_resolves_to_empty_and_is_recorded_missing() -> None:
    ctx = _Ctx(Path("."))
    source = mason_assets.ensure(ctx)
    ref = mason_refs.primitive_ref("not-a-real-generator", {})
    prims = source.primitives(ref)
    assert prims == []
    assert mason_refs.ref_key(ref) in source.missing


def test_a_primitive_builder_that_raises_resolves_to_empty_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from warlock.kernels.mesh import primitives as clay_primitives

    def exploding(**kwargs: Any):
        raise ValueError("bad params")

    monkeypatch.setitem(clay_primitives.GENERATORS, "box", ({}, exploding))

    ctx = _Ctx(Path("."))
    source = mason_assets.ensure(ctx)
    ref = _box_ref()
    prims = source.primitives(ref)
    assert prims == []
    assert mason_refs.ref_key(ref) in source.missing


def test_an_unresolved_library_ref_is_empty_but_not_missing() -> None:
    ctx = _Ctx(Path("."))
    source = mason_assets.ensure(ctx)
    ref = mason_refs.LibraryRef(job_id="job2", artifact="model.glb")
    prims = source.primitives(ref)
    assert prims == []
    assert mason_refs.ref_key(ref) not in source.missing


def test_a_library_asset_with_no_such_artifact_is_recorded_missing() -> None:
    ctx = _Ctx(Path("."))  # no job dir on disk at all
    source = mason_assets.ensure(ctx)
    ref = mason_refs.LibraryRef(job_id="does-not-exist", artifact="model.glb")
    source.primitives(ref)
    task_key = f"{mason_assets.TASK_PREFIX}does-not-exist:model.glb"
    result, error = ctx.run(task_key)
    assert error is not None
    done = _Done(task_key, ctx.tag_for(task_key), result=result, error=error)
    mason_assets.on_task_done(ctx, done)
    assert mason_refs.ref_key(ref) in source.missing
    assert source.primitives(ref) == []


def test_box_is_none_before_resolution() -> None:
    ctx = _Ctx(Path("."))
    source = mason_assets.ensure(ctx)
    ref = mason_refs.LibraryRef(job_id="job3", artifact="model.glb")
    assert source.box(ref) is None


def test_box_is_a_real_aabb_after_resolution_and_not_recomputed_per_call() -> None:
    ctx = _Ctx(Path("."))
    source = mason_assets.ensure(ctx)
    ref = _box_ref()
    source.primitives(ref)
    box1 = source.box(ref)
    box2 = source.box(ref)
    assert box1 is not None
    lo, hi = box1
    assert np.all(hi > lo)
    # Same cached tuple object handed back both times -- proof it was not
    # recomputed by walking the primitives' vertices again on the second call.
    assert box1 is box2


def test_baking_folds_a_node_translation_into_the_positions() -> None:
    ctx = _Ctx(Path("."))
    source = mason_assets.ensure(ctx)
    ref = mason_refs.LibraryRef(job_id="job4", artifact="model.glb")

    node = gltf.Node(translation=np.array([5.0, 0.0, 0.0]), mesh=0)
    prim = gltf.Primitive(
        positions=np.array([[0.0, 0.0, 0.0]], dtype="f4"),
        indices=np.array([0, 0, 0], dtype="u4"),
    )
    model = gltf.Model([node], [0], [[prim]], [])

    source.primitives(ref)
    task_key = f"{mason_assets.TASK_PREFIX}job4:model.glb"
    done = _Done(task_key, ctx.tag_for(task_key), result=mason_assets._bake_model(model))
    mason_assets.on_task_done(ctx, done)

    prims = source.primitives(ref)
    assert len(prims) == 1
    # The node's own translation is already in the position: this is the one
    # flat primitive list a placed library asset gets, per the module
    # docstring's "baking" section -- Mason never re-applies a per-node
    # transform on top of what this cache hands back.
    np.testing.assert_allclose(prims[0].positions[0], [5.0, 0.0, 0.0])


def test_mason_library_asset_bake_runs_on_the_task_thread_not_on_landing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The 2026-09-15 audit's mason-02: ``_bake_model`` -- a per-vertex
    matrix and normal transform, up to a 100 MB GLB -- used to run inside
    ``AssetSource.on_task``, which ``on_task_done`` calls from the frame
    thread that adopts a finished parse (see the module docstring's
    "adopted later, off on_task_done, on the frame that the task lands").
    Moved into the ``run()`` closure ``ctx.submit`` hands to the task pool,
    so the bake's cost lands on a task-pool thread, never a dropped frame.

    Proven with a spy on ``_bake_model``: it must already have run by the
    time ``ctx.run(task_key)`` -- standing in for the task pool actually
    running the submitted function -- returns, and ``on_task_done``
    (standing in for the frame-thread landing) must call it zero more times.
    Against the unfixed code the first assertion fails: ``run()`` only
    parsed the GLB, so ``calls`` is still empty until ``on_task_done`` bakes
    it on "landing".
    """
    calls: list[Any] = []
    real_bake = mason_assets._bake_model

    def spy_bake(model: Any) -> list[gltf.Primitive]:
        calls.append(model)
        return real_bake(model)

    monkeypatch.setattr(mason_assets, "_bake_model", spy_bake)

    node = gltf.Node(mesh=0)
    prim = gltf.Primitive(
        positions=np.zeros((3, 3), dtype="f4"),
        indices=np.array([0, 1, 2], dtype="u4"),
    )
    model = gltf.Model([node], [0], [[prim]], [])
    monkeypatch.setattr(mason_assets.gltf, "load", lambda data: model)

    class _FakePath:
        def read_bytes(self) -> bytes:
            return b""

    monkeypatch.setattr(
        mason_assets.sizeguard, "within_ceiling", lambda path, ceiling: _FakePath()
    )

    job_dir = tmp_path / "job5"
    job_dir.mkdir()
    (job_dir / "model.glb").write_bytes(b"")

    ctx = _Ctx(tmp_path)
    source = mason_assets.ensure(ctx)
    ref = mason_refs.LibraryRef(job_id="job5", artifact="model.glb")
    assert source.primitives(ref) == []

    task_key = f"{mason_assets.TASK_PREFIX}job5:model.glb"
    result, error = ctx.run(task_key)  # stands in for the task pool running run()
    assert error is None
    assert len(calls) == 1  # baked already, on the "task thread"

    done = _Done(task_key, ctx.tag_for(task_key), result=result, error=error)
    mason_assets.on_task_done(ctx, done)  # stands in for the frame-thread landing
    assert len(calls) == 1  # landing did not bake again


def test_on_task_done_ignores_a_key_that_is_not_ours() -> None:
    ctx = _Ctx(Path("."))
    mason_assets.ensure(ctx)
    done = _Done("clay-import:whatever", tag=None, result=None)
    assert mason_assets.on_task_done(ctx, done) is False


def test_ensure_returns_the_same_source_on_a_second_call() -> None:
    ctx = _Ctx(Path("."))
    first = mason_assets.ensure(ctx)
    second = mason_assets.ensure(ctx)
    assert first is second


def test_release_drops_cached_geometry() -> None:
    ctx = _Ctx(Path("."))
    source = mason_assets.ensure(ctx)
    ref = _box_ref()
    source.primitives(ref)
    assert source.box(ref) is not None
    source.release()
    assert source.box(ref) is None
    assert source.primitives(ref) != []  # re-resolves on next use, per _evict's docstring
