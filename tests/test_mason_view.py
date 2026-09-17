"""Mason's viewport: one upload per ref, one proxy per placement, one redraw rule.

Three of the tests here are regressions against bugs the plan predicted before
the code existed, and each is worded as the claim it makes rather than as the
code it touches.

The interesting half is what Mason does *differently* from Clay, because Clay's
own viewport tests already cover everything the two share. Clay's GPU cache is
one entry per object and its composite writes ``node.world`` on the cached
entry's own ``gltf.Node``, which is exactly right at one entry per object.
Mason's cache is one entry per **ref** -- that is what makes five hundred
instances one upload -- and the same composite would then have N placements
sharing one node, the last write winning, and N-1 drawing stacked at whichever
was composited last. Nothing raises and the frame renders, so only a test can
see it.

Everything that needs a context is skipped where there is no GPU, per the ``gl``
fixture. The cache key, the culling arithmetic and the drag's rebind rule do
not, and are asserted headlessly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from warlock.kernels.geom3d import gltf
from warlock.kernels.geom3d import math3d as m3
from warlock.studio import mason_view
from warlock.studio.mason import document as md
from warlock.studio.mason import nodes as nd
from warlock.studio.mason import refs as mrefs
from warlock.studio.mason import scene as msc

RECT = (0.0, 0.0, 128.0, 96.0)


class _State:
    """The *app*'s Mason state. The view reads the tool and the pivot off it
    and holds neither, because both are app settings shared across documents.

    The four ``snap*`` fields default off/zero, matching ``MasonState``'s own
    defaults, so every test that does not care about snapping is unaffected by
    their presence -- only the docs-01/docs-02 regression tests below set them.
    """

    def __init__(
        self,
        tool: str = "select",
        pivot: str = "median",
        snap: bool = False,
        snap_translate: float = 0.0,
        snap_rotate: float = 0.0,
        snap_ground: bool = False,
    ) -> None:
        self.tool = tool
        self.pivot = pivot
        self.snap = snap
        self.snap_translate = snap_translate
        self.snap_rotate = snap_rotate
        self.snap_ground = snap_ground


class _Ctx:
    def __init__(
        self,
        tool: str = "select",
        pivot: str = "median",
        snap: bool = False,
        snap_translate: float = 0.0,
        snap_rotate: float = 0.0,
        snap_ground: bool = False,
    ) -> None:
        self.state = type(
            "S",
            (),
            {"mason": _State(tool, pivot, snap, snap_translate, snap_rotate, snap_ground)},
        )()


def _box_primitive(material: Any = None) -> gltf.Primitive:
    """A unit cube's two front triangles -- enough geometry to upload and to
    raycast, and small enough that a test reads as a test."""
    positions = np.array(
        [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]],
        dtype="f4",
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype="u4")
    return gltf.Primitive(
        positions=positions,
        indices=indices,
        material=material if material is not None else gltf.Material(name="test"),
    )


class _Source:
    """A stand-in :class:`~warlock.studio.mason.refs.GeometrySource`.

    Deliberately *not* ``mason_assets.AssetSource``: what these tests are about
    is what the viewport does with whatever a source answers, and driving the
    real one would mean a service, a job directory and a task runner in a test
    about a cache key. ``rev`` is public and settable so the redraw test can
    make an asset "arrive" without a document edit, which is the whole
    situation that test exists for.
    """

    def __init__(self, primitive: Any = None) -> None:
        self._prims = [primitive if primitive is not None else _box_primitive()]
        self.rev = 0
        self.calls = 0

    def primitives(self, ref: Any) -> list[gltf.Primitive]:
        self.calls += 1
        return self._prims

    def box(self, ref: Any) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5])


class _EmptySource(_Source):
    """A source that has resolved nothing yet -- an asset still parsing."""

    def primitives(self, ref: Any) -> list[gltf.Primitive]:
        self.calls += 1
        return []

    def box(self, ref: Any) -> None:
        return None


def _scene(count: int = 6, *, spacing: float = 3.0) -> md.MasonDoc:
    """``count`` placements of **one** ref, spread along X."""
    doc = md.MasonDoc()
    ref = mrefs.primitive_ref("box", {})
    for i in range(count):
        doc.add_node(
            nd.MeshNode(
                uid=nd.new_uid(),
                name=f"prop{i}",
                translation=m3.vec3(float(i) * spacing, 0.0, 0.0),
                ref=ref,
            )
        )
    return doc


@pytest.fixture
def view(gl):
    v = mason_view.MasonView(gl, _Ctx())
    yield v
    v.release()


# --- the cache key ------------------------------------------------------------


def test_the_cache_key_is_the_ref_rather_than_the_node() -> None:
    """Two placements of one asset must produce **one** key.

    Headless, because this is arithmetic over the resolver's output and needs
    no GPU at all. Clay's equivalent key carries the object's own mesh by
    identity; if this one carried the node, five hundred barrels would be five
    hundred uploads and the whole instancing claim would be false.
    """
    doc = _scene(count=4)
    placed = msc.resolve(doc)
    keys = {mason_view._entry_key(item) for item in placed}

    assert len(placed) == 4
    assert len(keys) == 1


def test_a_material_override_is_a_second_key_and_therefore_a_second_upload() -> None:
    """The stated price of putting the override in the key, asserted rather
    than assumed: an instance that overrides its material uploads identical
    geometry twice. If overrides turn out to be common the fix is a per-draw
    material uniform, which is a renderer change -- but until then this is the
    behaviour, and a test is where a known cost is recorded."""
    doc = _scene(count=2)
    first, second = doc.roots
    doc.set_props(second.uid, material=gltf.Material(name="override"))
    placed = msc.resolve(doc)

    keys = {mason_view._entry_key(item) for item in placed}
    assert len(keys) == 2
    del first


def test_five_hundred_placements_of_one_ref_are_one_upload(view) -> None:
    """The headline claim, against the real GPU cache."""
    doc = _scene(count=500)
    view.sync(doc, _Source())

    assert view.rebuilds == 1
    assert len(view._cache) == 1


def test_a_ref_that_has_not_resolved_yet_is_not_cached_as_empty(view) -> None:
    """An asset still parsing in the background contributes no entry and must
    not poison the cache with one. Poisoning it would mean the parse finishing
    changed nothing, because a subsequent sync would find a valid-looking entry
    and skip the build -- the asset would never appear at all."""
    doc = _scene(count=3)
    source = _EmptySource()
    view.sync(doc, source)
    assert view.rebuilds == 0
    assert view._cache == {}

    view.sync(doc, _Source())
    assert view.rebuilds == 1


# --- the per-draw proxy -------------------------------------------------------


def test_every_instance_draws_at_its_own_place_rather_than_the_last_one_written(
    view,
) -> None:
    """**The quietest bug in the mode**, and the reason ``scene.DrawNode`` and
    ``scene.NodePool`` exist.

    ``clay_view._composite`` does ``node.world = world`` on the cached entry's
    own ``gltf.Node``. That is sound at one cache entry per *object* and
    silently wrong at one entry per *ref*: the six placements below share one
    cached node, so the last write wins and five of them draw stacked at the
    sixth's position. Nothing raises, nothing crashes, and the frame renders --
    which is why this is asserted on the *worlds the composite actually
    submits* rather than on a screenshot.
    """
    doc = _scene(count=6)
    source = _Source()
    view.sync(doc, source)
    composite = view._composite(doc, source)

    worlds = [node.world[0, 3] for node, _primitive in composite.draws]
    assert len(composite.draws) == 6
    # Six distinct X positions, 3 metres apart, in the order the resolver
    # walked them -- not six copies of the last.
    assert worlds == pytest.approx([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])
    # And six distinct proxy objects: one shared object holding the last world
    # would pass the assertion above only by accident of ordering.
    assert len({id(node) for node, _primitive in composite.draws}) == 6


def test_the_proxy_pool_does_not_allocate_a_new_object_every_frame(view) -> None:
    """The other half of the fix. ``DrawNode`` per placement is what makes it
    correct; pooling them is what makes paying for it affordable -- a thousand
    placements must cost a thousand allocations once, not once a frame."""
    doc = _scene(count=64)
    source = _Source()
    view.sync(doc, source)

    first = {id(node) for node, _primitive in view._composite(doc, source).draws}
    second = {id(node) for node, _primitive in view._composite(doc, source).draws}

    assert first == second


# --- the redraw key -----------------------------------------------------------


def test_an_asset_arriving_redraws_a_frame_the_document_never_changed(view) -> None:
    """``(id(source), source.rev)`` in the redraw key, and this is the whole
    reason ``GeometrySource.rev`` exists.

    An asset finishing its background parse changes the picture with **no
    document edit**: no node moved and no undo step was pushed, so ``doc.rev``
    does not move either. Without the source's revision in the key, this second
    frame is skipped as "nothing moved" and the asset appears only when
    something else happens to force a redraw.
    """
    doc = _scene(count=2)
    source = _EmptySource()
    view.draw(doc, source, RECT, 0.0)
    settled = view._last_render_key
    # Nothing at all has changed: the frame is skipped, which is the behaviour
    # the key is *supposed* to have and what makes the next assertion mean
    # something.
    view.draw(doc, source, RECT, 0.0)
    assert view._last_render_key == settled

    source.rev += 1
    view.draw(doc, source, RECT, 0.0)
    assert view._last_render_key != settled


def test_the_document_and_the_source_are_pinned_by_the_cache_that_names_them(
    view,
) -> None:
    """Both are held by identity in the redraw key, and an id is only sound
    while its object is alive -- the 2026-09-07 audit's clay-09, where a closed
    document could be collected and a new one minted at the same address with
    ``rev`` back at 0, handing back the closed tab's texture."""
    doc = _scene(count=1)
    source = _Source()
    view.draw(doc, source, RECT, 0.0)

    assert view._last_doc is doc
    assert view._last_source is source


# --- the drag -----------------------------------------------------------------


def test_a_drag_rebinds_the_transform_arrays_so_the_local_memo_cannot_go_stale() -> None:
    """``Node.local()`` memoizes its matrix against the *identity* of the
    node's three transform arrays, which is sound only because every transform
    write rebinds rather than writes through them -- ``nodes.Node.trs``'s own
    docstring states the rule.

    A drag that did ``node.translation[:] = ...`` would change the numbers
    without changing the objects, so the memo would keep handing back the
    matrix from before the drag: the node would hold the new transform and the
    viewport would draw the old one, with nothing in the data to say why.
    Headless -- no GPU is involved in moving a node.
    """
    doc = _scene(count=1)
    node = doc.roots[0]
    view = mason_view.MasonView.__new__(mason_view.MasonView)
    view.app_ctx = _Ctx()
    view._drag_start = {node.uid: tuple(np.array(v, copy=True) for v in node.trs())}
    view._drag_pivot = np.zeros(3)

    before = node.local()
    view._apply_drag(doc, delta=np.array([0.0, 2.0, 0.0]))

    assert node.translation[1] == pytest.approx(2.0)
    # The memo, not the field: this is the assertion that fails against a
    # write-through, where ``translation`` is right and ``local()`` is stale.
    assert node.local()[1, 3] == pytest.approx(2.0)
    assert before[1, 3] == pytest.approx(0.0)


def test_a_locked_node_is_not_dragged_even_though_it_is_selected() -> None:
    """A lock is *reported, never enforced* in the engine -- the standing rule,
    because a lock stops the user and not the document. The view is the layer
    that stops the user, so the refusal has to be asserted here."""
    doc = _scene(count=2)
    first, second = doc.roots
    doc.set_props(second.uid, locked=True)
    doc.select([first.uid, second.uid])

    view = mason_view.MasonView.__new__(mason_view.MasonView)
    view.app_ctx = _Ctx(tool="move")
    view._placed = []
    view._placed_key = None
    view._last_doc = None
    view._begin_gizmo_drag(doc, _Source())

    assert first.uid in view._drag_start
    assert second.uid not in view._drag_start


def test_dragging_a_node_with_the_move_gizmo_snaps_translation_to_the_grid_when_snap_is_on() -> (
    None
):
    """The 2026-09-12 audit's docs-01: Chapter 17 has the reader turn on Snap
    with **grid (m)** at 1 and drag a box, promising "it lands on whole
    metres," but ``_apply_drag`` never read ``state.snap`` or
    ``state.snap_translate`` at all -- only the first-click placement path
    (``_drop_point``) did. The audit's own probe dragged a node by 2.37 m with
    Snap on and a 1 m grid and it landed at exactly 2.37, unsnapped; this
    encodes that same drag and asserts the grid-aligned answer instead.
    """
    doc = _scene(count=1)
    node = doc.roots[0]
    view = mason_view.MasonView.__new__(mason_view.MasonView)
    view.app_ctx = _Ctx(tool="move", snap=True, snap_translate=1.0)
    view._drag_start = {node.uid: tuple(np.array(v, copy=True) for v in node.trs())}
    view._drag_pivot = np.zeros(3)

    view._apply_drag(doc, delta=np.array([2.37, 0.0, 0.0]))

    assert node.translation[0] == pytest.approx(2.0)


def test_dragging_a_node_with_drop_to_ground_enabled_keeps_it_on_the_terrain_surface() -> None:
    """The 2026-09-12 audit's docs-02: ``state.snap_ground`` is written by its
    own toggle in ``panes/mason_tools.py`` and read nowhere else, so Chapter
    17's "Drop to ground ... it lands on the ground rather than floating above
    or sinking into it" did nothing during the one gesture -- dragging -- the
    chapter tells the reader to use it for. The audit's own probe dragged a
    node starting 5 m above the ground with the toggle on and its height was
    unchanged after the drag; this drags the same node and asserts its box
    actually rests on the ground plane afterward.
    """
    doc = _scene(count=1)
    node = doc.roots[0]
    view = mason_view.MasonView.__new__(mason_view.MasonView)
    view.app_ctx = _Ctx(tool="move", snap_ground=True)
    view._drag_start = {node.uid: tuple(np.array(v, copy=True) for v in node.trs())}
    view._drag_pivot = np.zeros(3)
    source = _Source()

    view._apply_drag(doc, source, delta=np.array([0.0, 5.0, 0.0]))

    box = msc.world_bounds(doc, source, uids=[node.uid])
    assert box is not None
    assert box[0][1] == pytest.approx(0.0)


# --- culling ------------------------------------------------------------------


def test_a_box_behind_the_camera_is_culled_and_one_in_front_is_not() -> None:
    """The frustum test itself, headlessly. Conservative on purpose: a false
    positive costs one draw call the GPU discards, where a false negative costs
    a prop that is missing from the picture."""
    from warlock.studio.viewer.camera import Camera

    camera = Camera()
    camera.aspect = 1.0
    planes = mason_view._frustum_planes(camera.projection() @ camera.view())
    box = (np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5]))

    at_origin = m3.compose(m3.vec3(0.0, 0.0, 0.0), m3.quat_identity(), m3.vec3(1, 1, 1))
    far_behind = m3.compose(
        m3.vec3(0.0, 0.0, 500.0), m3.quat_identity(), m3.vec3(1, 1, 1)
    )

    assert mason_view._box_visible(planes, box, at_origin)
    assert not mason_view._box_visible(planes, box, far_behind)


def test_culling_is_skipped_under_the_threshold(view) -> None:
    """The test is not free, and under a few hundred placements the GPU throws
    an off-screen draw away more cheaply than Python decides not to make it.
    Asserted by the counter the HUD reads, which is the only way to see culling
    working from outside."""
    doc = _scene(count=4, spacing=1000.0)
    source = _Source()
    view.sync(doc, source)
    view._composite(doc, source)

    assert len(doc.roots) < mason_view.CULL_THRESHOLD
    assert view.culled == 0


# --- selection ----------------------------------------------------------------


def test_the_gizmo_sits_at_the_pivot_the_user_chose() -> None:
    """Three pivots rather than one, because a scene editor's selection is a
    set. ``origin`` is the one with no ambiguity in it, so it is what pins the
    plumbing: the median of two props three metres apart is not the origin."""
    doc = _scene(count=2)
    doc.select([node.uid for node in doc.roots])

    median_view = mason_view.MasonView.__new__(mason_view.MasonView)
    median_view.app_ctx = _Ctx(pivot="median")
    median_view._placed = []
    median_view._placed_key = None
    median_view._last_doc = None
    origin_view = mason_view.MasonView.__new__(mason_view.MasonView)
    origin_view.app_ctx = _Ctx(pivot="origin")
    origin_view._placed = []
    origin_view._placed_key = None
    origin_view._last_doc = None

    source = _Source()
    assert median_view.selection_centre(doc, source)[0] == pytest.approx(1.5)
    assert origin_view.selection_centre(doc, source)[0] == pytest.approx(0.0)


def test_active_pivot_uses_the_last_clicked_node_not_document_order() -> None:
    """The "Active" pivot's own tooltip promises "the last node clicked", but
    the 2026-09-16 audit found ``selection_centre`` returning ``points[0]`` --
    the first selected node in ``resolve()``'s document-walk order -- which
    for a fixed selection is the same node however the user built it up, not
    whichever one was actually clicked last. Selecting both nodes with the
    *second* recorded as ``active`` (``doc.select``'s new keyword) must put
    the pivot at the second node's position (x=3.0) -- against the unfixed
    ``points[0]`` this would instead land on the first node (x=0.0).
    """
    doc = _scene(count=2)
    first, second = doc.roots
    doc.select([first.uid, second.uid], active=second.uid)

    active_view = mason_view.MasonView.__new__(mason_view.MasonView)
    active_view.app_ctx = _Ctx(pivot="active")
    active_view._placed = []
    active_view._placed_key = None
    active_view._last_doc = None

    source = _Source()
    assert active_view.selection_centre(doc, source)[0] == pytest.approx(3.0)


# --- Stage F: the ground, which has geometry and no ref -----------------------


def _ground(side: int = 4, size: float = 8.0) -> md.MasonDoc:
    """A flat terrain and the node that places it."""
    from warlock.kernels.geom3d import gltf as _gltf
    from warlock.studio.mason.terrain import Terrain

    doc = md.MasonDoc()
    doc.set_terrain(
        Terrain(
            heights=np.zeros((side + 1, side + 1), dtype="f4"),
            size_x=size,
            size_z=size,
            material=_gltf.Material(name="ground"),
        )
    )
    doc.add_node(nd.TerrainNode(uid=nd.new_uid(), name="Terrain"))
    return doc


def test_the_ground_is_drawn_even_though_it_has_no_ref(view) -> None:
    """The second half of the "no ref, no picture" gap -- the first being lights.

    Terrain geometry is generated from ``doc.terrain`` rather than resolved from
    a ``GeometrySource``, so the ref-keyed cache has nowhere to put it and the
    composite's ``ref is None`` skip threw it away. Picking already worked: the
    ray found ground the eye could not see.
    """
    doc = _ground()
    source = _Source()
    view.sync(doc, source)
    composite = view._composite(doc, source)

    assert composite is not None
    assert len(composite.draws) == 1
    assert view.terrain_rebuilds == 1


def test_the_ground_is_uploaded_once_and_reused_until_a_brush_rebinds_it(view) -> None:
    """Keyed on the identity of ``heights``, which is sound precisely because
    every brush **rebinds** that array rather than writing into it -- the same
    property ``terrain_mesh``'s own memo rests on, so the mesh build and the
    upload invalidate together on one signal rather than on two."""
    doc = _ground()
    source = _Source()
    view.sync(doc, source)
    view.sync(doc, source)
    assert view.terrain_rebuilds == 1

    doc.begin_sculpt()
    doc.sculpt((0, 0, 2, 2), np.full((2, 2), 1.0, dtype="f4"))
    doc.end_sculpt()
    view.sync(doc, source)
    assert view.terrain_rebuilds == 2


def test_a_ground_nothing_places_any_more_gives_its_buffers_back(view) -> None:
    """The rule the ref cache already follows, kept true of the one drawable
    that has no ref: holding an upload for geometry that does not draw, does not
    export and is not picked would make it the one of the three that is only
    half true."""
    doc = _ground()
    source = _Source()
    view.sync(doc, source)
    assert view._terrain is not None

    for node in [n for n in doc.all_nodes() if isinstance(n, nd.TerrainNode)]:
        doc.remove_node(node.uid)
    view.sync(doc, source)
    assert view._terrain is None


def test_a_sculpt_stroke_changes_the_picture_without_changing_the_document(view) -> None:
    """**The frame that would otherwise be skipped.**

    ``document.sculpt`` pushes nothing and touches no revision on purpose -- the
    whole drag is one undo step that only ``end_sculpt`` commits -- so the
    document does not see a stroke in progress. Without the height array's
    identity in the redraw key, every frame of a drag is skipped as "nothing
    moved" and the ground jumps to its new shape on release.
    """
    doc = _ground()
    source = _Source()
    view.draw(doc, source, RECT, 0.0)
    settled, rev = view._last_render_key, doc.rev
    # Nothing changed: the frame is skipped, which is what makes the rest mean
    # something.
    view.draw(doc, source, RECT, 0.0)
    assert view._last_render_key == settled

    doc.begin_sculpt()
    doc.sculpt((0, 0, 3, 3), np.full((3, 3), 2.0, dtype="f4"))
    # The claim, stated as the assertion the key has to survive: the document is
    # *unchanged* as far as every other redraw signal is concerned.
    assert doc.rev == rev

    view.draw(doc, source, RECT, 0.0)
    assert view._last_render_key != settled
    assert view.terrain_rebuilds == 2


# --- the sculpt session, from the pointer -------------------------------------


def test_a_whole_drag_is_one_undo_step(view) -> None:
    """A pane that pushed a ``TerrainEdit`` per mouse-move would make one stroke
    fifty presses of Ctrl+Z. The session is what makes it one, and the view is
    what opens and closes it."""
    doc = _ground(side=16, size=16.0)
    view._rect = RECT
    view.camera.target = m3.vec3(0.0, 0.0, 0.0)
    state = view.state
    state.tool = "sculpt"
    state.brush = "raise"
    state.brush_radius = 4.0
    steps = len(doc.history)
    centre = (RECT[2] / 2.0, RECT[3] / 2.0)

    assert view._press(doc, _Source(), 1, centre) is True
    assert view._grab == "sculpt"
    for step in range(1, 6):
        view._motion(doc, _Source(), (centre[0] + step, centre[1]))
    assert doc.sculpting is True
    assert len(doc.history) == steps

    view._release(doc, 1)
    assert doc.sculpting is False
    assert len(doc.history) == steps + 1
    assert float(doc.terrain.heights.max()) > 0.0
    doc.undo()
    assert float(doc.terrain.heights.max()) == pytest.approx(0.0)


def test_a_sculpt_press_selects_nothing_and_starts_no_orbit(view) -> None:
    """The brush owns the left button for the whole stroke, which is what makes
    Sculpt a tool rather than an armed placement: a press that also picked would
    change the selection under every stroke."""
    doc = _ground(side=16, size=16.0)
    view._rect = RECT
    state = view.state
    state.tool = "sculpt"
    view._press(doc, _Source(), 1, (RECT[2] / 2.0, RECT[3] / 2.0))
    assert doc.selection == set()
    assert view._grab == "sculpt"


def test_a_press_that_misses_the_ground_starts_no_stroke(view) -> None:
    """A sky click with the brush in hand must fall through to the ordinary
    press behaviour rather than opening a session nothing will ever close."""
    doc = _ground(side=8, size=8.0)
    view._rect = RECT
    view.state.tool = "sculpt"
    # Straight up, away from the ground plane.
    view.camera.target = m3.vec3(0.0, 0.0, 0.0)
    view.camera.pitch = -80.0
    view.camera.update(1.0)
    view._press(doc, _Source(), 1, (RECT[2] / 2.0, 1.0))
    assert doc.sculpting is False


def test_a_cancelled_drag_still_commits_the_stroke_it_had_already_made(view) -> None:
    """The asymmetry is deliberate: a gizmo drag's "before" is three arrays the
    view still holds, where a stroke's is a whole height field the session
    snapshotted -- and ``end_sculpt`` is the only thing that turns what is
    already on the ground into something Ctrl+Z can reach. Dropping the session
    would leave a sculpted ground with no undo step for it at all."""
    doc = _ground(side=16, size=16.0)
    view._rect = RECT
    view.state.tool = "sculpt"
    steps = len(doc.history)
    view._press(doc, _Source(), 1, (RECT[2] / 2.0, RECT[3] / 2.0))
    view.cancel_drag(doc)
    assert doc.sculpting is False
    assert len(doc.history) == steps + 1


def test_a_live_stroke_counts_as_dragging_so_an_undo_cannot_land_inside_it(view) -> None:
    """``mason_mode._DRAG_BLOCKED_CTRL`` reads ``view.dragging`` to refuse
    Ctrl+Z mid-drag, and an undo between two dabs would leave the session
    holding a "before" snapshot of a height field the history has replaced."""
    doc = _ground(side=16, size=16.0)
    view._rect = RECT
    view.state.tool = "sculpt"
    view._press(doc, _Source(), 1, (RECT[2] / 2.0, RECT[3] / 2.0))
    assert view.dragging is True
    view._release(doc, 1)
    assert view.dragging is False


# --- the armed placement is a request, not a placement ------------------------


def test_an_armed_click_asks_the_pane_to_place_and_does_not_place_itself(view) -> None:
    """The view owns the pointer and the camera; what a placement *means* --
    which document, which node kind, which undo step -- is ``mason_mode``'s, and
    this module does not import the controller. ``menu_request`` is drained the
    same way for the same reason."""
    doc = _scene(count=0)
    view._rect = RECT
    view.state.place_kind = "box"

    assert view._press(doc, _Source(), 1, (RECT[2] / 2.0, RECT[3] / 2.0)) is True
    assert view.place_request is not None
    # Nothing was added, and nothing was selected: this press is a question.
    assert doc.roots == []
    assert doc.selection == set()


def test_an_armed_click_lands_on_what_it_points_at(view) -> None:
    """The point is where the ray *hits*, so a prop drops onto the ground or onto
    the roof of another prop -- which is the answer a user pointing at a surface
    means. With nothing under the cursor it is the ground plane."""
    doc = _ground(side=8, size=8.0)
    view._rect = RECT
    view.state.place_kind = "box"
    view._press(doc, _Source(), 1, (RECT[2] / 2.0, RECT[3] / 2.0))

    point = view.place_request
    assert point is not None
    assert abs(float(point[1])) < 1e-6


def test_an_armed_click_snaps_to_the_grid_when_snapping_is_on(view) -> None:
    """Through ``ops.snap_translation`` rather than arithmetic in the view -- the
    rule the Tools pane already follows: this file decides *whether*, the engine
    decides *where*."""
    doc = _scene(count=0)
    view._rect = RECT
    state = view.state
    state.place_kind = "box"
    state.snap = True
    state.snap_translate = 5.0
    view._press(doc, _Source(), 1, (RECT[2] / 3.0, RECT[3] / 3.0))

    point = view.place_request
    assert point is not None
    for value in point:
        assert float(value) % 5.0 == pytest.approx(0.0, abs=1e-6)
