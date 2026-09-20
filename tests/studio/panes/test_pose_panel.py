"""The inspector's Pose tab: applying a saved pose, and re-saving one.

poser-01, the 2026-09-20 audit: ``_apply`` (the "Apply" button in the saved
poses list) called ``viewer.set_pose`` alone -- no ``reset_all`` first and no
``set_root_translation`` after -- so a pose authored with "Move root" lost its
offset the moment it was re-applied from this panel, and ``_save``'s payload
carried no ``root_translation`` either, so re-saving that same pose erased the
offset from disk for good (``store.save_pose`` rebuilds the record wholesale
from what it is given). Both halves mirror an omission the Poser's own doors,
``poser_mode.apply_asset_pose`` and ``poser_mode.save_pose_to_asset``, were
fixed for on 2026-09-11 (poser-02) -- this pane writes to the same per-job
pose store and was never brought into line.

A lightweight fake stands in for the viewer/editor pair here rather than the
real ``PoseEditor``: what matters is the *sequence* of calls and the payload
``_save`` builds, both of which this fake records without needing a GL
context or a real GLTF skeleton bound to it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.studio.panes import pose_panel


class _FakeEditor:
    def __init__(self, *, current: str | None = None, root: list[float] | None = None) -> None:
        self.mode = "pose"
        self.current = current
        self._root = list(root or [0.0, 0.0, 0.0])
        self._bones: dict[str, Any] = {}

    def root_translation(self) -> list[float]:
        return list(self._root)

    def pose(self) -> dict[str, Any]:
        return dict(self._bones)


class _FakeViewer:
    """Records what was called and in what order, and keeps enough state
    (root offset, bones, ``current``) for the assertions to read back."""

    def __init__(self, *, pose_mode: bool = True, current: str | None = None,
                 root: list[float] | None = None) -> None:
        self.pose_mode = pose_mode
        self.editor = _FakeEditor(current=current, root=root)
        self.calls: list[tuple] = []

    def reset_all(self, *, dirty: bool = True) -> None:
        self.calls.append(("reset_all", dirty))
        self.editor._bones = {}
        self.editor._root = [0.0, 0.0, 0.0]

    def set_pose(self, bones: dict, *, pose_id: str | None = None, dirty: bool = True) -> None:
        self.calls.append(("set_pose", dict(bones), pose_id, dirty))
        self.editor._bones = dict(bones)
        self.editor.current = pose_id

    def set_root_translation(self, v: list[float], *, dirty: bool = True) -> None:
        self.calls.append(("set_root_translation", list(v), dirty))
        self.editor._root = list(v)

    def get_pose(self) -> dict[str, Any]:
        return dict(self.editor._bones)


def test_applying_a_saved_pose_in_the_inspector_restores_its_root_offset_and_resaving_it_does_not_erase_it():  # noqa: E501
    job = {"id": "abc123456789", "files": ["model.glb", "rig.glb"]}
    # A stale offset left over from whatever the editor was showing before --
    # the exact shape that a bare set_pose (no reset_all first) would leave
    # untouched.
    viewer = _FakeViewer(root=[1.0, 0.0, 0.0])
    ctx = SimpleNamespace(viewer=viewer)
    pose = {
        "id": "p1",
        "name": "Crouch",
        "bones": {"hips": [0.0, 0.0, 0.0, 1.0]},
        "root_translation": [0.1, 0.0, 0.25],
    }

    # -- half one: applying restores the record's own offset -----------------
    pose_panel._apply_saved_pose(ctx, job, pose, "p1")
    assert viewer.editor.root_translation() == pytest.approx([0.1, 0.0, 0.25]), (
        "the saved pose's own root offset must land, not the stale one the "
        "editor started with"
    )
    call_names = [c[0] for c in viewer.calls]
    assert call_names.index("reset_all") < call_names.index("set_pose"), (
        "reset_all must run before set_pose, apply_asset_pose's own order"
    )
    assert "set_root_translation" in call_names

    # -- half two: re-saving does not erase it --------------------------------
    ctx.svc = object()
    submitted: dict[str, Any] = {}

    def fake_submit(key: str, fn: Any, *args: Any) -> bool:
        submitted["key"] = key
        submitted["payload"] = args[-1]
        return True

    ctx.submit = fake_submit
    prompts_asked: list[Any] = []
    ctx.prompts = SimpleNamespace(ask=prompts_asked.append)

    pose_panel._save(ctx, job, viewer)
    assert prompts_asked, "the name prompt must have been raised"
    prompts_asked[-1].on_accept("Crouch")

    assert submitted["payload"]["root_translation"] == pytest.approx([0.1, 0.0, 0.25]), (
        "root_translation must be in the payload _save submits, or "
        "store.save_pose rebuilds the record without it and the offset is "
        "gone from disk"
    )
