"""Regressions for the 2026-09-26 audit, closed against the files this fixer
(w6f8) owns: ``mason/engine/document.py``, ``mason/engine/scene.py``,
``mason/engine/terrain.py``, ``mason/engine/serialize.py``, ``mason/mode.py``
(``place_dropped_mesh`` only), ``mason/ui/view.py`` (``drop_point`` only),
``mason/ui/viewport.py``, ``mason/ui/panes/outliner.py`` and
``mason/ui/panes/prefabs.py``.

Each test's name is the claim it makes, ``test_mason_mode.py``'s own
convention (restated by every other mason test module).
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from typing import Any

import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.studio.modes.mason import mode as mason_mode
from realmspinner.studio.modes.mason.engine import document as md
from realmspinner.studio.modes.mason.engine import gltfout
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import refs as mason_refs
from realmspinner.studio.modes.mason.engine import scene as msc
from realmspinner.studio.modes.mason.engine import serialize as ser
from realmspinner.studio.modes.mason.engine import terrain as terrain_mod
from realmspinner.studio.modes.mason.ui import view as mason_view
from realmspinner.studio.modes.mason.ui.panes import outliner as mason_outliner
from realmspinner.studio.modes.mason.ui.panes import prefabs as mason_prefabs

# --- mason-engine-05: no write door enforced nodes.MAX_DEPTH -----------------


def test_add_node_refuses_a_subtree_that_would_nest_past_max_depth(monkeypatch) -> None:
    """mason-engine-05, the 2026-09-26 audit: nothing on the write side ever
    enforced ``nodes.MAX_DEPTH`` -- ``add_node`` attached a chain of groups
    arbitrarily deep, and a scene built past the ceiling saved clean
    (``serialize.scene_json`` writes ``children`` recursively with no depth
    ceiling of its own) and then refused to ever reopen
    (``serialize._read_node`` is the one place the ceiling was ever checked).
    Against the unfixed ``add_node``, the final call below succeeds instead of
    raising.
    """
    monkeypatch.setattr(nd, "MAX_DEPTH", 2)
    doc = md.MasonDoc()
    g0 = nd.GroupNode(uid=nd.new_uid(), name="g0")
    doc.add_node(g0)
    g1 = nd.GroupNode(uid=nd.new_uid(), name="g1")
    doc.add_node(g1, parent_uid=g0.uid)
    g2 = nd.GroupNode(uid=nd.new_uid(), name="g2")
    doc.add_node(g2, parent_uid=g1.uid)  # depth 2, exactly at the ceiling: allowed

    g3 = nd.GroupNode(uid=nd.new_uid(), name="g3")
    with pytest.raises(ValueError, match="deep"):
        doc.add_node(g3, parent_uid=g2.uid)  # depth 3: past the ceiling

    # And the one that *was* allowed really does round-trip.
    data = ser.rscn_bytes(doc)
    reopened = ser.read_rscn(data)
    assert len(reopened.roots) == 1


def test_move_node_refuses_a_reparent_that_would_nest_past_max_depth(monkeypatch) -> None:
    """The reparent half of mason-engine-05: ``move_node`` had the identical
    gap -- dragging a node in the outliner onto a deeply nested group had no
    depth ceiling either, only the acyclicity refusals :meth:`move_node`
    already had. A pure reorder among the same siblings is unaffected (depth
    does not change), which is why this drags across parents.
    """
    monkeypatch.setattr(nd, "MAX_DEPTH", 2)
    doc = md.MasonDoc()
    g0 = nd.GroupNode(uid=nd.new_uid(), name="g0")
    doc.add_node(g0)
    g1 = nd.GroupNode(uid=nd.new_uid(), name="g1")
    doc.add_node(g1, parent_uid=g0.uid)
    g2 = nd.GroupNode(uid=nd.new_uid(), name="g2")
    doc.add_node(g2, parent_uid=g1.uid)  # depth 2

    x = nd.MeshNode(uid=nd.new_uid(), name="x")
    doc.add_node(x)  # root, depth 0

    with pytest.raises(ValueError, match="deep"):
        doc.move_node(x.uid, 0, parent_uid=g2.uid)  # would land at depth 3

    # Refused, so the node is still where it started.
    assert doc.parent_uid_of(x.uid) is None


def test_add_nodes_refuses_a_batch_past_max_depth_before_attaching_any(monkeypatch) -> None:
    """:meth:`add_nodes` -- a prefab placement or an array op's own door --
    gets the identical check, ahead of its own loop, so a batch attach never
    leaves some copies attached and the rest refused mid-loop."""
    monkeypatch.setattr(nd, "MAX_DEPTH", 0)
    doc = md.MasonDoc()
    parent = nd.GroupNode(uid=nd.new_uid(), name="p")
    doc.add_node(parent)  # depth 0, exactly at the ceiling: allowed

    batch = [nd.MeshNode(uid=nd.new_uid(), name=f"m{i}") for i in range(3)]
    with pytest.raises(ValueError, match="deep"):
        doc.add_nodes(batch, parent_uid=parent.uid)  # would land at depth 1: past it
    # None of the batch attached.
    assert parent.children == []


# --- mason-engine-06: visit/enter counted independently, doubling the ceiling


def _mesh(name: str = "box") -> nd.MeshNode:
    return nd.MeshNode(uid=nd.new_uid(), name=name, ref=mason_refs.primitive_ref("box", {}))


class _Source:
    """A minimal ``GeometrySource``: one triangle for any ref at all."""

    rev = 0

    def primitives(self, ref: Any) -> list[gltf.Primitive]:
        import numpy as np

        return [
            gltf.Primitive(
                positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"),
                indices=np.array([0, 1, 2], dtype="u4"),
                material=gltf.Material(name=""),
            )
        ]

    def box(self, ref: Any) -> Any:
        return self.primitives(ref)[0].box()


def test_walk_shares_one_counter_between_visit_and_enter_so_the_ceiling_is_not_doubled(
    monkeypatch,
) -> None:
    """mason-engine-06, the 2026-09-26 audit: ``visit`` and ``enter`` each got
    their own ``_bounded`` wrapper with its own private counter, so a scene
    that fires both (``gltfout.scene_model`` hands the *same* bound method to
    each -- ``enter=builder.visit``) could place up to roughly twice
    ``MAX_PLACED`` before either counter -- each seeing only its own half of
    the calls -- ever tripped. Five prefab instances, each templated directly
    on a single bare mesh (no wrapping group -- a ``GroupNode`` in the
    template would itself draw a ``visit`` call and muddy the count), against
    a ceiling of 5 contribute exactly 5 ``enter`` calls and 5 ``visit`` calls:
    each independently at the ceiling, so before the fix neither ever raised,
    for ten actual placed items -- twice what the ceiling promises. Against
    the unfixed ``walk``, the ``scene_model`` call below returns a finished
    export instead of raising.
    """
    doc = md.MasonDoc()
    doc.prefabs["leaf"] = _mesh()
    for i in range(5):
        doc.add_node(nd.PrefabNode(uid=nd.new_uid(), name=f"Instance{i}", template="leaf"))

    monkeypatch.setattr(msc, "MAX_PLACED", 5)
    with pytest.raises(ValueError, match="MAX_PLACED"):
        gltfout.scene_model(doc, _Source())


# --- mason-engine-07: validate_terrain_size let an infinite extent through --


def test_validate_terrain_size_refuses_infinite_and_nan_extents() -> None:
    """mason-engine-07, the 2026-09-26 audit: ``size_x > 0`` is true of
    ``float("inf")`` -- Python has no special case for it -- so an infinite
    (or NaN) extent sailed through this check and on into ``_build_mesh``,
    which divides by it and multiplies it back into every vertex position,
    producing a NaN-filled ground mesh far from this refusal. Against the
    unfixed function, neither call below raises at all.
    """
    with pytest.raises(ValueError, match="finite"):
        terrain_mod.validate_terrain_size(float("inf"), 4.0)
    with pytest.raises(ValueError, match="finite"):
        terrain_mod.validate_terrain_size(4.0, float("nan"))


def test_terrain_construction_refuses_an_infinite_size_too() -> None:
    """``Terrain.__post_init__`` calls :func:`validate_terrain_size` itself,
    so building one directly with an infinite extent is refused the same way
    a config edit through ``MasonDoc.set_terrain_config`` is."""
    import numpy as np

    with pytest.raises(ValueError, match="finite"):
        terrain_mod.Terrain(
            heights=np.zeros((3, 3), dtype="f4"),
            size_x=float("inf"),
            size_z=4.0,
            material=gltf.Material(name="ground"),
        )


# --- the new finding: PIL.UnidentifiedImageError escaped _read_textures -----


def _one_node_with_texture_doc() -> md.MasonDoc:
    material = gltf.Material(name="tex", base_color=(2, 2, bytes(range(16))))
    node = nd.MeshNode(uid=nd.new_uid(), name="Rock", ref=mason_refs.primitive_ref("box", {}))
    node.material = material
    return md.MasonDoc(roots=[node])


def _replace_member(data: bytes, member: str, raw: bytes) -> bytes:
    """The same archive with one member's bytes replaced outright."""
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            dst.writestr(name, raw if name == member else src.read(name))
    return out.getvalue()


def test_read_rscn_refuses_a_texture_that_is_not_a_decodable_image() -> None:
    """The 2026-09-26 audit's fix pass: ``_read_textures`` let Pillow's
    ``UnidentifiedImageError`` (an ``OSError`` subclass) escape uncaught for a
    texture member that is not real image bytes -- a truncated write, a
    hand-edited zip member -- mirroring the identical gap Clay's own
    ``kernels/mesh/serialize.py`` closed under finding clay-document-07.
    Against the unfixed reader this raises Pillow's own unnamed
    ``UnidentifiedImageError`` instead of this module's refusal.
    """
    data = ser.rscn_bytes(_one_node_with_texture_doc())
    mangled = _replace_member(data, f"{ser.TEXTURE_DIR}/0.png", b"not a png at all")
    with pytest.raises(ValueError, match="not a usable image"):
        ser.read_rscn(mangled)


# --- mason-mode-06: a dropped library mesh had nowhere in the app to land --


class _FakeCtx:
    """``test_mason_mode.py``'s own ``FakeCtx``, restated here rather than
    imported: a test module may not import another test module."""

    def __init__(self) -> None:
        self.svc = None
        self.state = type("S", (), {"mason": None, "mode": "home"})()
        self.settings = type("Settings", (), {"store": {}, "get": lambda self, k: None})()
        self.toasts: list[tuple[str, str]] = []

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        run(*args)
        return True

    def toast(self, message: str, kind: str = "info", *_a: Any, **_k: Any) -> None:
        self.toasts.append((message, kind))


def _armed_ctx() -> _FakeCtx:
    ctx = _FakeCtx()
    mason_mode.new_document(ctx)
    return ctx


def test_place_dropped_mesh_places_the_job_at_the_drop_point_as_one_undo_step() -> None:
    """mason-mode-06, the 2026-09-26 audit: a library row already offers
    itself as a ``library.DRAG_MESH`` drag-and-drop payload
    (``library.draggable_mesh``), built for Mason's viewport, but nothing in
    the mode layer -- or anywhere in ``studio/`` -- ever accepted it, so a
    card that lifted had no controller function to land in. Mirrors
    ``place_armed``'s own "add, then move, as one step" shape. Against the
    unfixed ``mode.py``, ``mason_mode.place_dropped_mesh`` does not exist at
    all (``AttributeError``).
    """
    ctx = _armed_ctx()
    doc = mason_mode.ensure(ctx).active.doc
    before = len(doc.history)

    uid = mason_mode.place_dropped_mesh(
        ctx, {"id": "abc123", "name": "Barrel"}, (3.0, 0.0, -1.0)
    )

    assert uid is not None
    node = doc.node(uid)
    assert isinstance(node, nd.MeshNode)
    assert node.name == "Barrel"
    assert list(node.translation) == [3.0, 0.0, -1.0]
    # One undo step, not an add followed by a move.
    assert len(doc.history) == before + 1
    doc.undo()
    assert doc.node(uid) is None


def test_masonview_drop_point_is_public_for_a_drop_target_outside_the_class() -> None:
    """mason-mode-06: the viewport's own new drop target
    (``ui/viewport.py``'s ``_mason_accept_dropped_mesh``) needs this method
    from outside :class:`MasonView`, which a leading underscore refuses to
    offer. Against the unfixed class this method is named ``_drop_point`` and
    ``drop_point`` does not exist.
    """
    assert hasattr(mason_view.MasonView, "drop_point")
    assert not hasattr(mason_view.MasonView, "_drop_point")


# --- mason-mode-14: no clipper, and a full tree walk paid every frame -------


def test_outliner_body_clips_rows_with_imguis_own_list_clipper() -> None:
    """Pinned the way this codebase already pins its other wiring facts
    (``clay/ui/panes/outliner.py``'s own clay-panes-05 regression test) rather
    than measuring an actual scroll position."""
    import inspect

    source = inspect.getsource(mason_outliner._body)
    assert "imgui.ListClipper()" in source
    assert "clipper.begin(" in source
    assert "clipper.step()" in source


def test_outliner_cached_rows_reuses_the_walk_until_the_document_changes() -> None:
    """mason-mode-14, the 2026-09-26 audit: this pane used to rebuild
    ``list(doc.walk())`` by walking the whole tree on every single frame it
    drew, whether or not the document had changed since the frame before.
    ``_cached_rows`` now answers a second call with the *same* list object
    when nothing has happened to the document in between, and only re-walks
    once something has -- ``doc.rev`` moves on every edit
    (``document.py``'s own module docstring names exactly this cache as what
    ``rev`` is for). Against the unfixed pane, ``_cached_rows`` does not exist
    (``AttributeError``); the old code rebuilt a fresh list every call, so
    ``first is second`` would be ``False``.
    """
    doc = md.MasonDoc()
    doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="g"))
    tab = type("Tab", (), {"doc": doc})()

    first = mason_outliner._cached_rows(tab)
    second = mason_outliner._cached_rows(tab)
    assert second is first

    doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="h"))
    third = mason_outliner._cached_rows(tab)
    assert third is not first
    assert len(third) == 2


def test_prefab_instance_counts_are_cached_until_the_document_changes() -> None:
    """The same gap in ``ui/panes/prefabs.py``'s own walk of the tree
    (``_instance_counts``, over ``doc.all_nodes()``): the pane rebuilt it from
    scratch on every frame it drew regardless of whether the scene had
    changed. ``_instance_counts`` itself stays a plain, uncached query --
    ``test_mason_panes.py``'s own
    ``test_the_prefabs_pane_counts_the_instances_in_the_scene_tree`` calls it
    directly with a bare document -- and ``_cached_instance_counts`` is the
    new wrapper the pane calls instead. Against the unfixed pane, this
    function does not exist at all (``AttributeError``).
    """
    doc = md.MasonDoc()
    doc.prefabs["big"] = nd.GroupNode(uid=nd.new_uid(), name="Tmpl")
    doc.add_node(nd.PrefabNode(uid=nd.new_uid(), name="Instance", template="big"))
    tab = type("Tab", (), {"doc": doc})()

    first = mason_prefabs._cached_instance_counts(tab)
    second = mason_prefabs._cached_instance_counts(tab)
    assert second is first
    assert first == {"big": 1}

    doc.add_node(nd.PrefabNode(uid=nd.new_uid(), name="Instance2", template="big"))
    third = mason_prefabs._cached_instance_counts(tab)
    assert third is not first
    assert third == {"big": 2}
