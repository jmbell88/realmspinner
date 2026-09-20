"""Mason's library round trip: a scene out as an asset, and back in as a scene.

Stage G's whole claim, in one file. The outward half is
``mason_mode.export_library`` -- an ordinary ``model`` row, minted by the same
``import_mesh`` Clay's builds go through, so rigging, sprite sheets, the
triangle retarget and every mesh export work on it without any of them learning
that Mason exists. The inward half is ``edit_asset_in_mason``, which reads the
``scene.rscn`` sidecar written beside the mesh and gets the *arrangement* back:
the groups, the lights, the links.

The two halves are tested together rather than apart, because what would
actually break is the join. Each half can be perfectly correct about itself
while the file one writes is not the file the other reads.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.service import files as svc_files
from realmspinner.studio.modes.mason import assets as mason_assets
from realmspinner.studio.modes.mason import mode as mason_mode
from realmspinner.studio.modes.mason.engine import document as md
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import refs as mason_refs
from realmspinner.studio.modes.mason.engine import scene as mason_scene
from realmspinner.studio.modes.mason.engine import serialize as mason_ser


class _Cache:
    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _AppState:
    def __init__(self) -> None:
        self.mason = None
        self.mode = "home"


class FakeCtx:
    """Runs a submitted callable inline -- ``tests/modes/mason/test_mason_mode.py``'s own
    fake, with a real service under it, because everything this file is about
    happens inside ``run()``."""

    def __init__(self, svc: Any) -> None:
        self.svc = svc
        self.state = _AppState()
        self.settings = _Settings()
        self.cache = _Cache()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.result: Any = None

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info", *_args: Any, **_kwargs: Any) -> None:
        self.toasts.append((message, kind))


class _Done:
    def __init__(self, key: str, result: Any = None) -> None:
        self.key = key
        self.result = result


class _Source:
    """A ``GeometrySource`` answering one box for every ref.

    Mason's real source resolves a ``LibraryRef`` through the job directory and
    a ``PrimitiveRef`` through ``clay.primitives``; neither is what this file is
    testing, and a fake keeps the round trip's own failure modes from hiding
    behind a missing asset.
    """

    rev = 0

    def primitives(self, ref: Any) -> list[gltf.Primitive]:
        return [
            gltf.Primitive(
                positions=np.array(
                    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="f4"
                ),
                indices=np.array([0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2], dtype="u4"),
                material=gltf.Material(name="stone"),
            )
        ]

    def box(self, ref: Any) -> Any:
        return self.primitives(ref)[0].box()


@pytest.fixture(autouse=True)
def _fake_source(monkeypatch):
    monkeypatch.setattr(mason_assets, "ensure", lambda ctx: _Source())


def _scene() -> md.MasonDoc:
    """A scene with the three things a merged mesh cannot carry: a group, a
    light, and a mesh node that is a *link* to a library job rather than
    geometry."""
    doc = md.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), name="Props")
    doc.add_node(group)
    doc.add_node(
        nd.MeshNode(
            uid=nd.new_uid(),
            name="Crate",
            ref=mason_refs.LibraryRef(job_id="abcdef123456", name="Crate"),
        ),
        parent_uid=group.uid,
    )
    doc.add_node(nd.LightNode(uid=nd.new_uid(), name="Key", kind="point"))
    return doc


def _exported(ctx: FakeCtx) -> tuple[Any, str]:
    """A whole export: the submit, and the result coming back. ``on_task_done``
    is half of it -- the tab's memory of which row it became is written there,
    not in the task."""
    tab = mason_mode.adopt(ctx, _scene(), title="Courtyard")
    mason_mode.export_library(ctx, tab)
    mason_mode.on_task_done(ctx, _Done(f"mason-library:{tab.uid}", ctx.result))
    return tab, tab.job_id


# --- out ----------------------------------------------------------------------


def test_a_scene_exports_as_an_ordinary_finished_mesh_row(svc):
    """The payoff is that the row is *ordinary*: ``done``, at stage ``model``,
    with a ``model.glb`` -- so every mesh-shaped path in the app reaches it
    with no knowledge of Mason at all."""
    ctx = FakeCtx(svc)
    _tab, job_id = _exported(ctx)
    job = svc.store.get(job_id)
    assert job["status"] == "done"
    assert job["stage"] == "model"
    assert (svc.config.job_dir(job_id) / "model.glb").exists()
    assert job["name"] == "Courtyard"


def test_the_row_names_mason_as_the_mode_that_can_reopen_it(svc):
    """The marker is the whole of what the library's reopen door is gated on:
    ``asset_exits`` answers every gate from the cached row, with no ``stat`` on
    the frame thread, and a reopen has no fallback to take if it guessed
    wrong."""
    ctx = FakeCtx(svc)
    _tab, job_id = _exported(ctx)
    assert svc.store.get(job_id)["params"]["authored"] == "mason"


def test_the_scene_document_lands_beside_the_mesh_and_is_never_listed(svc):
    """``build.rblk``'s precedent exactly: the sidecar is a real file in the
    job directory and is absent from every table that would serve it or draw
    a row for it."""
    ctx = FakeCtx(svc)
    _tab, job_id = _exported(ctx)
    sidecar = svc_files.mason_source_path(svc, job_id)
    assert sidecar.exists()
    assert zipfile.ZipFile(BytesIO(sidecar.read_bytes())).namelist()
    assert svc_files.MASON_SOURCE not in svc_files.MEDIA
    assert svc_files.MASON_SOURCE not in svc_files.LISTED
    job = dict(svc.store.get(job_id))
    svc_files.attach_files(job, svc.job_dir(job_id))
    assert svc_files.MASON_SOURCE not in job["files"]


def test_the_exported_mesh_keeps_the_scene_graph_rather_than_one_merged_lump(svc):
    """The GLB written here is ``gltfout.scene_model``'s, not ``objout``'s
    merge: ``source.glb`` has to hold the node graph, because a flattened one
    would lose the grouping and the light on the way *in* to the library and
    no sidecar could put it back -- the reopen would be the only surviving
    copy of the arrangement, which is not what "the mesh derives from the
    source" means anywhere else in this app."""
    ctx = FakeCtx(svc)
    _tab, job_id = _exported(ctx)
    model = gltf.load((svc.config.job_dir(job_id) / "source.glb").read_bytes())
    names = {node.name for node in model.nodes}
    assert "Props" in names, "the group survived"
    assert model.lights, "the light survived"


def test_an_empty_scene_is_refused_before_a_job_directory_exists(svc):
    """Refused in the mode, not at the service door -- ``clay_mode``'s reason:
    ``check_glb`` would refuse the same bytes, but only after the user had been
    told an export was under way and a row had been started for it."""
    ctx = FakeCtx(svc)
    tab = mason_mode.adopt(ctx, md.MasonDoc(), title="Empty")
    mason_mode.export_library(ctx, tab)
    assert not ctx.submitted
    assert not svc.store.list()
    assert tab.job_id == ""


def test_the_export_tells_the_library_its_cache_is_stale(svc):
    """A row minted behind the cache's back is invisible until something says
    so, and a user told the export worked who then finds nothing in the
    workshop has been lied to."""
    ctx = FakeCtx(svc)
    _exported(ctx)
    assert ctx.cache.invalidated == 1
    assert ("Exported to the library.", "info") in ctx.toasts


# --- and back in ----------------------------------------------------------------


def test_the_row_reopens_as_the_scene_it_was_and_not_as_the_merged_mesh(svc):
    """The round trip's join, and the reason the two halves are tested
    together. The arrangement that comes back has to be the *document* -- the
    group, the light and the link by job id -- none of which a merged mesh has
    any way to express."""
    ctx = FakeCtx(svc)
    _tab, job_id = _exported(ctx)
    job = dict(svc.store.get(job_id))
    job["name"] = "Courtyard"

    mason_mode.edit_asset_in_mason(ctx, job)
    assert ctx.submitted[-1] == f"mason-reopen:{job_id}"
    mason_mode.on_task_done(ctx, _Done(f"mason-reopen:{job_id}", ctx.result))

    tab = mason_mode.active(ctx)
    assert tab is not None
    assert ctx.state.mode == "mason"
    doc = tab.doc
    assert [node.name for node in doc.roots] == ["Props", "Key"]
    group = doc.roots[0]
    assert isinstance(group, nd.GroupNode)
    crate = group.children[0]
    assert isinstance(crate.ref, mason_refs.LibraryRef)
    assert crate.ref.job_id == "abcdef123456"
    assert isinstance(doc.roots[1], nd.LightNode)


def test_a_reopened_scene_is_not_a_file_and_remembers_the_row_it_came_from(svc):
    """``path`` stays None -- there is no file on disk the user chose, so
    nothing goes in the recent list and a plain Save has nothing to write over
    -- while ``job_id`` carries, so the Document pane can still say which row
    this scene last became."""
    ctx = FakeCtx(svc)
    _tab, job_id = _exported(ctx)
    mason_mode.edit_asset_in_mason(ctx, dict(svc.store.get(job_id)))
    mason_mode.on_task_done(ctx, _Done(f"mason-reopen:{job_id}", ctx.result))
    tab = mason_mode.active(ctx)
    assert tab.path is None
    assert tab.job_id == job_id
    assert not tab.dirty


def test_reopening_a_row_whose_scene_is_gone_raises_rather_than_substituting(svc):
    """The asymmetry with ``clay_mode.edit_asset_in_clay``, asserted rather
    than only documented. Clay falls back to importing ``model.glb`` because a
    Clay document *is* geometry; a Mason document is an arrangement, and its
    merged mesh is one mesh node where there were sixty. Opening that and
    calling it the scene would show the user finished work that is not there
    and let them save over it."""
    ctx = FakeCtx(svc)
    _tab, job_id = _exported(ctx)
    before = len(mason_mode.ensure(ctx).docs)
    svc_files.mason_source_path(svc, job_id).unlink()
    with pytest.raises(OSError):
        mason_mode.edit_asset_in_mason(ctx, dict(svc.store.get(job_id)))
    # The failure is reported (the task raises and the app's failed path has
    # it); what must *not* happen is a second tab appearing, holding the
    # merged mesh and calling itself the scene.
    assert len(mason_mode.ensure(ctx).docs) == before


# --- the picture ----------------------------------------------------------------


def test_the_exported_card_is_photographed_from_mason_s_own_viewport(svc):
    """A library card's picture comes from the viewport that is already drawn,
    and there are two of them now.

    Until Stage G the only caller of this was Clay, so the App's capture
    reached for ``self.clay_view`` by name -- and Plotter's and Packwright's
    exports have been routed through it ever since. For Mason that is not a
    near miss but the wrong picture: Clay's viewport is either holding an
    unrelated document or, in a session that never opened Clay, is None, which
    would leave the card on its placeholder and read as an export that
    produced nothing.
    """
    from realmspinner.studio import main as main_mod

    class _App:
        def __init__(self, ctx: Any) -> None:
            self.app_ctx = ctx
            self.clay_view = "clay-viewport"
            self.mason_view = "mason-viewport"
            self.captured: list[tuple[str, Any]] = []

        def _capture_thumbnail_from(self, job_id: str, view: Any) -> None:
            self.captured.append((job_id, view))

    ctx = FakeCtx(svc)
    _tab, job_id = _exported(ctx)
    app = _App(ctx)
    done = _Done("mason-library:ms-x", {"job_id": job_id, "exported_asset": True})
    main_mod.App._on_task_done(app, done)
    assert app.captured == [(job_id, "mason-viewport")]


# --- the 2026-09-18 audit's second-run mason-01: writer and reader must agree ---


def _light(name: str) -> nd.LightNode:
    return nd.LightNode(uid=nd.new_uid(), name=name, kind="point")


def test_define_prefab_refuses_a_template_that_would_push_roots_plus_prefabs_past_max_placed(
    monkeypatch,
):
    """``define_prefab`` used to charge nothing against ``MAX_PLACED`` at all
    -- its own comment at :func:`~realmspinner.studio.modes.mason.engine.document.
    MasonDoc.define_prefab`'s call site said so -- while
    ``serialize.read_rscn`` counts scene roots *and* every template's nodes
    against that same shared ceiling. A document built through guarded calls
    alone (this one: three roots, then ``define_prefab``) could save clean
    and then permanently refuse to reopen. The fix charges the template here,
    against roots plus every *other* prefab, so the writer refuses at the
    same total the reader would -- before anything is written, not after.
    """
    monkeypatch.setattr(mason_scene, "MAX_PLACED", 5)
    doc = md.MasonDoc(roots=[_light("A"), _light("B"), _light("C")])
    assert doc._node_count == 3
    # A 3-node template: 3 roots + 3 template nodes = 6, past the ceiling of 5.
    template_source = nd.GroupNode(
        uid=nd.new_uid(), name="Big", children=[_light("D"), _light("E")]
    )
    with pytest.raises(ValueError, match="MAX_PLACED"):
        doc.define_prefab("big", template_source)
    assert doc.prefabs == {}


def test_a_document_with_several_large_named_prefabs_can_reopen_after_it_saves_successfully(
    monkeypatch,
):
    """The full round trip the finding describes: build a document whose
    roots plus prefab templates stay within ``MAX_PLACED`` (through
    ``define_prefab``'s new pre-flight check, which the previous test proves
    fires), save it, and reopen it. Before the fix, nothing stopped
    ``define_prefab`` from building a document whose combined total
    ``read_rscn`` would refuse -- this proves a document built entirely
    through the guarded API always reopens.
    """
    monkeypatch.setattr(mason_scene, "MAX_PLACED", 10)
    doc = md.MasonDoc(roots=[_light("A"), _light("B")])
    doc.define_prefab(
        "one", nd.GroupNode(uid=nd.new_uid(), name="One", children=[_light("D"), _light("E")])
    )
    doc.define_prefab(
        "two", nd.GroupNode(uid=nd.new_uid(), name="Two", children=[_light("F"), _light("G")])
    )
    # roots(2) + template "one"(3) + template "two"(3) = 8, within the ceiling
    # of 10 -- and a ninth or tenth node would still fit, an eleventh would not.
    with pytest.raises(ValueError, match="MAX_PLACED"):
        doc.define_prefab(
            "three",
            nd.GroupNode(
                uid=nd.new_uid(), name="Three", children=[_light("H"), _light("I"), _light("J")]
            ),
        )

    data = mason_ser.rscn_bytes(doc)
    reopened = mason_ser.read_rscn(data)
    assert set(reopened.prefabs) == {"one", "two"}
    assert len(list(nd.walk(reopened.roots))) == 2
