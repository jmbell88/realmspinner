"""Residue R3 of the 2026-10-14 audit: the bone-line paint parameters stay live."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from realmspinner.studio.viewer import bonelines
from realmspinner.studio.viewer.markers import _rgb


class _Gl:
    def buffer(self, **_kw):
        return SimpleNamespace(write=lambda _b: None, release=lambda: None)

    def vertex_array(self, *_a, **_kw):
        return SimpleNamespace(release=lambda: None)


def test_bone_lines_draws_takes_the_colour_and_alpha_clay_passes():
    """Clay's pose view calls ``draws(..., colour=IDLE, alpha=alpha)``; the cleanup
    that removed them as onion-ghost-only broke 24 Clay tests."""
    parameters = inspect.signature(bonelines.BoneLines.draws).parameters
    assert parameters["colour"].default is None
    assert parameters["alpha"].default == 1.0

    lines = bonelines.BoneLines(_Gl(), SimpleNamespace(get=lambda _n: object()))
    positions = {"a": np.zeros(3), "b": np.ones(3)}
    (item,) = lines.draws(positions, [("a", "b")], None, colour=0x102030, alpha=0.7)
    assert item.color == (*_rgb(0x102030), 0.7)

    (default,) = lines.draws(positions, [("a", "b")], None)
    assert default.color == (*_rgb(bonelines.IDLE), 1.0)
