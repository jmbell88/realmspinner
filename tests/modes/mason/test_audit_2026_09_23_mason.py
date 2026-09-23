"""Regressions for the 2026-09-23 audit's mason-01, mason-02 and mason-03.

Each test's name is the claim, and each failed against the unfixed code
before the fix in this same change -- see the fixer's return for the pasted
failing output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.geom3d.gltf import Material
from realmspinner.studio.modes.mason import mode as mason_mode
from realmspinner.studio.modes.mason.engine import document as md
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import scene as msc
from realmspinner.studio.modes.mason.engine import terrain as terrain_mod


def _terrain(side: int = 4, size: float = 4.0) -> terrain_mod.Terrain:
    heights = np.zeros((side + 1, side + 1), dtype="f4")
    return terrain_mod.Terrain(heights=heights, size_x=size, size_z=size, material=Material())


class _FakeCtx:
    """Runs a submitted callable inline -- the same shape
    ``test_mason_mode.py``'s ``FakeCtx`` uses, kept local so this file edits
    nothing another fixer owns."""

    def __init__(self) -> None:
        self.svc = None
        self.state = _AppState()
        self.settings = _Settings()
        self.toasts: list[tuple[str, str]] = []

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        run(*args)
        return True

    def toast(self, message: str, kind: str = "info", *_a: Any, **_kw: Any) -> None:
        self.toasts.append((message, kind))


class _AppState:
    def __init__(self) -> None:
        self.mason = None
        self.mode = "home"


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


def _box() -> nd.MeshNode:
    return nd.MeshNode(uid=nd.new_uid())


# --- mason-01 --------------------------------------------------------------


def test_placing_many_prefab_instances_refuses_before_the_resolved_count_exceeds_max_placed(
    monkeypatch,
) -> None:
    """A placed ``PrefabNode`` instance costs the scene tree exactly one
    node, but ``scene.resolve`` expands it into its whole template subtree
    on every draw, export and pick. ``place_prefab`` used to charge only the
    tree-side count, so a second instance -- tree total 2, nowhere near
    ``MAX_PLACED`` -- was let through even though *resolving* the document
    afterwards would blow straight past the ceiling."""
    ctx = _FakeCtx()
    tab = mason_mode.new_document(ctx)
    doc = tab.doc

    template = nd.GroupNode(uid=nd.new_uid(), name="cluster")
    for _ in range(5):
        template.children.append(_box())
    doc.define_prefab("Cluster", template)

    monkeypatch.setattr(msc, "MAX_PLACED", 10)

    first = mason_mode.place_prefab(ctx, "Cluster")
    assert first is not None  # tree=1, resolved=6 (the group root + 5 meshes)

    steps = len(doc.history)
    second = mason_mode.place_prefab(ctx, "Cluster")

    assert second is None
    assert len(doc.history) == steps
    assert len(doc.all_nodes()) == 1  # still just the one instance -- tree side never noticed
    assert any(kind == "error" for _msg, kind in ctx.toasts)
    # The refusal actually protected what it claims to: the scene, as left,
    # still resolves under the ceiling. ``max_items`` is passed explicitly --
    # ``resolve``'s own default parameter is bound once at import time, so a
    # monkeypatched ``MAX_PLACED`` never reaches it unless the caller forwards
    # the live value itself.
    assert len(msc.resolve(doc, max_items=msc.MAX_PLACED)) <= msc.MAX_PLACED


# --- mason-02 ----------------------------------------------------------------


def test_resolve_max_items_above_the_module_ceiling_is_honoured(monkeypatch) -> None:
    """``resolve``'s own ``max_items`` parameter used to never reach
    ``walk``, so ``walk``'s default -- the *module* ``MAX_PLACED``, not
    whatever ``resolve`` was handed -- was what actually bounded the
    traversal. A caller raising its own ceiling above the module default was
    refused at the lower, module number instead."""
    doc = md.MasonDoc()
    for _ in range(5):
        doc.add_node(_box())
    monkeypatch.setattr(msc, "MAX_PLACED", 2)

    # The module ceiling (2) sits below the document's own resolved count
    # (5); a caller-supplied ceiling above the module default must still be
    # honoured rather than refusing at the module's smaller number.
    placed = msc.resolve(doc, max_items=100)
    assert len(placed) == 5


# --- mason-03 ----------------------------------------------------------------


def test_set_terrain_config_refuses_a_non_positive_size_rather_than_writing_an_unreopenable_document():  # noqa: E501
    """``set_terrain_config`` used to validate nothing and write straight
    onto the live ``Terrain`` with ``setattr``, which never re-runs
    ``Terrain.__post_init__`` -- so a zero size saved clean and then refused
    to ever reopen (``read_rscn`` rebuilds a ``Terrain`` from scratch and its
    ``__post_init__`` catches it there instead)."""
    doc = md.MasonDoc()
    doc.set_terrain(_terrain())
    assert doc.terrain is not None

    with pytest.raises(ValueError, match="must both be positive"):
        doc.set_terrain_config(size_x=0.0)

    # Refused before anything was written or pushed.
    assert doc.terrain.size_x == 4.0
    assert doc.terrain.size_z == 4.0


def test_validate_terrain_size_is_the_one_place_both_terrain_and_set_terrain_config_check():
    """Pins the shared validator directly, so a future setter cannot reopen
    this gap by hand-rolling its own copy of the check."""
    with pytest.raises(ValueError, match="must both be positive"):
        terrain_mod.validate_terrain_size(0.0, 4.0)
    with pytest.raises(ValueError, match="must both be positive"):
        terrain_mod.validate_terrain_size(4.0, -1.0)
    terrain_mod.validate_terrain_size(4.0, 4.0)  # does not raise
