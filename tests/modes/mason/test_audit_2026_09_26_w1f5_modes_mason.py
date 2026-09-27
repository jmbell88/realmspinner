"""Findings closed from the 2026-09-26 audit, Mason engine slice (w1f5).

mason-engine-01 -- Make prefab: ``define_prefab`` stored the template as
``nodes.copy_subtree(node, fresh_uids=False)``, keeping the selected node's
own translation/rotation/scale on the template *root*. The instance that
replaces the node in the tree (``mode.py``'s
``define_prefab_from_selection``) is given that same TRS, so
``scene.py``'s prefab expansion applied it twice -- once composing the
instance's own ``local()`` into its parent's world matrix, once more
composing the template root's ``local()`` on top of that -- landing every
freshly made instance at double its original transform. Reproduced: a node
moved to x=5 came back from a moved prefab at x=10.

``unpack_instance`` (a few lines below ``define_prefab`` in the same file)
already discards the template root's TRS and substitutes the instance's own,
which is what let the bug hide on a round trip through unpack -- "world x
10, expected 5" only shows up while the node is still expressed as a
:class:`~.nodes.PrefabNode` instance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from realmspinner.studio import dialogs
from realmspinner.studio.modes.mason import mode as mason_mode
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import scene as sc


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _FakeCtx:
    """The same minimal ctx ``tests/modes/mason/test_prefab_naming.py``
    already builds by hand for this same controller layer -- restated here
    rather than imported across test modules, per that file's own comment."""

    def __init__(self) -> None:
        self.svc = None
        self.state = type("S", (), {"mason": None, "mode": "home"})()
        self.settings = _Settings()
        self.prompts = dialogs.PromptQueue()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, kind: str = "info", *_a: Any, **_k: Any) -> None:
        self.toasts.append((message, kind))


def _ctx() -> _FakeCtx:
    ctx = _FakeCtx()
    mason_mode.new_document(ctx)
    return ctx


def test_make_prefab_from_a_moved_node_leaves_its_world_transform_unchanged() -> None:
    ctx = _ctx()
    doc = mason_mode.ensure(ctx).active.doc
    node = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Barrel"))
    node.translation = np.array([5.0, 0.0, 0.0])
    doc.select([node.uid])

    name = mason_mode.define_prefab_from_selection(ctx, "Barrel")
    assert name == "Barrel", ctx.toasts

    placed = sc.resolve(doc)
    assert len(placed) == 1
    world_x = placed[0].world[0, 3]
    assert world_x == 5.0, (
        f"Make prefab must not move the node: expected world x=5.0, got {world_x}"
    )
