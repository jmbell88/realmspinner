"""Regression tests for the 2026-09-26 audit's w4f3 fixer batch: every
finding whose owned files live under ``kernels/pixel`` (all of it, including
``flourish/`` and ``walk/``) plus ``studio/modes/inker/_doc_flourish.py`` for
inker-flourish-03.

Covers, in order: inker-mode-09 and inker-document-10 (``to_background``),
inker-document-13 (``set_layer_props``/``set_group_props`` blend
validation), inker-document-09 (``insert_flourish`` undo inside a folder),
inker-document-08 (``paste_cels`` palette reconciliation), inker-document-07
(the transform ceiling under shear and rotation), inker-document-12
(``begin_transform`` on a refused lift), inker-document-14 (identity
whole-canvas geometry ops), inker-document-15 (premultiplied-scale
rounding), inker-flourish-03 (duplicate phase names), inker-flourish-04
(paint window sizing under erosion), inker-flourish-06
(``Recipe.from_dict`` exception types), inker-mode-14 (the walk renderer's
RotSprite budget), inker-paint-01/02 (``_resolve_indices`` scope against a
stroke's own coverage, not its dirty rect) and inker-paint-03 (blur/smudge
premultiply).

Not here: inker-flourish-05 needs ``studio/modes/inker/ui/panes/flourish.py``,
outside this batch's owned directories. inker-paint-04 is not a fix -- see
the fixer's return for why the finding is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.pixel import groups as gp
from realmspinner.kernels.pixel import index_plane as ixp
from realmspinner.kernels.pixel import selection as sel
from realmspinner.kernels.pixel import transform as tf
from realmspinner.kernels.pixel.document import Document
from realmspinner.kernels.pixel.flourish import bake as B
from realmspinner.kernels.pixel.flourish.prims import core as core_prim
from realmspinner.kernels.pixel.flourish.prims import ring as ring_prim
from realmspinner.kernels.pixel.flourish.prims import smoke as smoke_prim
from realmspinner.kernels.pixel.flourish.recipe import Layer as RecipeLayer
from realmspinner.kernels.pixel.flourish.recipe import Phase, Recipe, clamp, from_dict
from realmspinner.kernels.pixel.flourish.render import FrameCtx
from realmspinner.kernels.pixel.tiles import TilemapCel, strip
from realmspinner.kernels.pixel.walk import render as wr
from realmspinner.kernels.pixel.walk import rig as R
from realmspinner.kernels.pixel.walk.gait import Pose

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLACK = (0, 0, 0, 255)


# -- inker-mode-09 / inker-document-10: to_background --------------------------------------


def _blank_tile(w: int = 4, h: int = 4) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def _colour_tile(colour: tuple[int, int, int, int], w: int = 4, h: int = 4) -> np.ndarray:
    tile = np.zeros((h, w, 4), dtype=np.uint8)
    tile[..., 0], tile[..., 1], tile[..., 2], tile[..., 3] = colour
    return tile


def _tileset(*colours: tuple[int, int, int, int], w: int = 4, h: int = 4):
    stack = np.stack([_blank_tile(w, h), *[_colour_tile(c, w, h) for c in colours]], axis=0)
    return strip(stack)


def test_convert_to_background_on_a_tilemap_layer_changes_nothing():
    """inker-mode-09: mutating ``layer.pixels`` before ``_patch_edit_for``'s
    tilemap refusal left a tilemap bottom layer's pixels changed with no
    flag, step or dirty mark. Refusing first must leave it untouched."""
    doc = Document.blank(8, 8)
    slot = doc.add_tileset(_tileset(RED))
    doc.add_tilemap_layer(slot.uid)
    doc.remove_layer(0)
    cel = doc.stack[0]
    assert isinstance(cel, TilemapCel)
    before_pixels = cel.pixels.copy()
    depth = len(doc.history)

    with pytest.raises(ValueError):
        doc.to_background()

    assert np.array_equal(cel.pixels, before_pixels)
    assert len(doc.history) == depth


def test_to_background_on_an_empty_bottom_cel_autovivifies_instead_of_raising():
    """inker-document-10: an animated document's bottom row that nothing has
    ever drawn on is the shared read-only placeholder plane
    (``animation.blank_plane``). Converting it used to raise numpy's
    "assignment destination is read-only" instead of converting."""
    doc = Document.blank(8, 8)
    doc.add_frame()
    assert doc.anim is not None and doc.anim.is_placeholder(doc.stack[0])

    assert doc.to_background() is True

    assert not doc.anim.is_placeholder(doc.stack[0])
    assert doc.stack[0].background is True


# -- inker-document-13: set_layer_props / set_group_props validate blend -------------------


def test_set_layer_props_refuses_an_unknown_blend_before_mutating_or_pushing():
    """inker-document-13: ``setattr`` bypassed ``Layer.__post_init__``'s own
    blend-mode check, so a bad value landed and a step was pushed *before*
    the next recomposite raised several calls later."""
    doc = Document.blank(8, 8)
    doc.add_layer("L1")
    target = doc.stack[1]
    depth = len(doc.history)

    with pytest.raises(ValueError):
        doc.set_layer_props(1, blend="bogus-mode")

    assert target.blend == "normal"
    assert len(doc.history) == depth


def test_set_group_props_refuses_an_unknown_blend_before_mutating_or_pushing():
    doc = Document.blank(8, 8)
    doc.add_layer("L1")
    node = doc.group_layers([0, 1], name="G")
    assert node is not None
    depth = len(doc.history)

    with pytest.raises(ValueError):
        doc.set_group_props(node.uid, blend="bogus-mode")

    assert node.blend == "normal"
    assert len(doc.history) == depth


# -- inker-document-09: insert_flourish undone inside a folder -----------------------------


def _hand_recipe(with_layer: bool = True) -> Recipe:
    layers = (RecipeLayer(uid=1, kind="core", name="A"),) if with_layer else ()
    return Recipe(
        name="Hand", width=4, height=4, phases=(Phase("main", 2, False),), layers=layers
    )


def _hand_bake():
    opaque = np.zeros((4, 4, 4), np.uint8)
    opaque[..., :3] = (200, 40, 40)
    opaque[..., 3] = 255
    recipe = _hand_recipe(True)
    facing = B.Facing(
        name="E", degrees=0.0, composites={"main": [opaque, opaque]},
        layers={"main": {1: [opaque, opaque]}},
    )
    return B.Bake(recipe=recipe, facings=[facing], palette=None, palette_source="none")


def test_undoing_a_flourish_insert_inside_a_folder_leaves_a_well_formed_group_tree():
    """inker-document-09: ``GroupAddEdit.undo`` dissolves by *promoting*
    members to the group's own parent -- correct for ungrouping layers that
    outlive the group, wrong for tracks this same compound just created,
    which the following ``TrackAddEdit.undo`` calls remove from the stack
    without ever touching their now-stale ``group_of`` entry. At the root
    ``_drop_group`` happens to delete instead of promote, which is why this
    only ever showed up nested."""
    doc = inker.Document.blank(8, 8)
    doc.add_layer("L1")
    doc.add_layer("L2")
    folder = doc.group_layers([1, 2], name="Folder")
    assert folder is not None
    doc.set_active_layer(2)
    assert doc._parent_of_active() == folder.uid

    baked = _hand_bake()
    group = doc.insert_flourish(baked)
    assert doc.group_of.get(group) == folder.uid

    doc.history.undo(doc)
    gp.check(doc.groups, doc.group_of, doc.member_uids())  # must not raise

    doc.history.redo(doc)
    gp.check(doc.groups, doc.group_of, doc.member_uids())
    assert doc.group_of.get(group) == folder.uid


# -- inker-document-08: paste_cels reconciles indices against this palette -----------------


def test_pasting_indexed_cels_into_a_document_with_another_palette_keeps_the_materialisation():
    """inker-document-08: a copied plane's ``.indices`` names slots of the
    document it was copied from. Landing it unreconciled kept the old slot
    numbers, so a copied red cel materialised green after the destination's
    ``recolour_slot`` moved that slot -- and ``check_materialized`` would
    fail the moment anything rematerialised it."""
    palette = [BLACK, RED, GREEN, RED]  # slot 1 and slot 3 both start red
    doc = Document.blank(4, 4)
    weight = np.ones((4, 4), dtype=np.float32)
    assert doc.write_colour((0, 0, 4, 4), RED, weight)  # frame 0: solid red
    doc.add_frame()  # frame 1: the paste target, still blank
    assert doc.convert_to_indexed(palette)
    track_uid = doc.anim.tracks[0].uid
    frame1_uid = doc.anim.frames[1].uid

    clip = doc.copy_cels(0, 0, 0, 0)
    assert clip is not None
    assert doc.recolour_slot(1, GREEN)  # slot 1 turns green; slot 3 is still red

    assert doc.paste_cels(clip, 0, 1)
    pasted = doc.anim.cels[(track_uid, frame1_uid)]

    assert tuple(int(c) for c in pasted.pixels[0, 0]) == RED
    assert pasted.indices is not None and pasted.indices[0, 0] == 3
    doc.check_materialized()
    doc._rematerialize(pasted)
    assert tuple(int(c) for c in pasted.pixels[0, 0]) == RED


# -- inker-document-07: the transform ceiling under shear and rotation ---------------------


def test_the_transform_ceiling_holds_with_shear_and_rotation_applied(monkeypatch):
    """inker-document-07: ``MAX_TRANSFORM_SIDE`` clamped only ``scale``, so a
    shear near :data:`.transform.SHEAR_MAX` plus a rotation reached several
    times the ceiling -- 622px against a ceiling patched to 256 here, the
    same ratio the live 16384 ceiling reproduced at roughly 50,000px."""
    monkeypatch.setattr(sel, "MAX_TRANSFORM_SIDE", 256)
    pixels = np.full((64, 64, 4), 255, dtype=np.uint8)
    mask = np.full((64, 64), 255, dtype=np.uint8)
    buf = sel.FloatingBuffer(pixels=pixels, mask=mask, offset=(0, 0), layer_uid=1)

    buf.transform(scale=(4.0, 4.0), shear=(55.0, 0.0), angle=45.0)

    width, height = buf.size
    assert width <= sel.MAX_TRANSFORM_SIDE
    assert height <= sel.MAX_TRANSFORM_SIDE


# -- inker-document-12: begin_transform on a refused lift -----------------------------------


def test_begin_transform_on_a_locked_layer_leaves_no_stray_selection_or_step():
    """inker-document-12: with no selection, ``begin_transform`` selects the
    whole canvas and then calls ``lift()`` -- which can still refuse (a
    locked layer, a tilemap layer's raise). The select-all step used to
    survive the refusal, leaving a full-canvas selection and an extra undo
    step for a transform that never began."""
    doc = Document.blank(8, 8)
    doc.stack[0].locked = True
    depth = len(doc.history)

    assert doc.begin_transform() is False

    assert doc.mask is None
    assert len(doc.history) == depth


# -- inker-document-14: identity whole-canvas geometry ops push no step --------------------


@pytest.mark.parametrize(
    "op",
    [
        lambda doc: doc.rotate90(0),
        lambda doc: doc.rotate90(4),
        lambda doc: doc.scale((8, 8)),
        lambda doc: doc.resize_canvas((8, 8), (0, 0)),
        lambda doc: doc.crop((0, 0, 8, 8)),
    ],
    ids=["rotate90-0", "rotate90-4", "scale-same-size", "resize-identity", "crop-full-canvas"],
)
def test_an_identity_whole_canvas_geometry_op_pushes_no_step(op):
    """inker-document-14: ``_replay`` snapshots and pushes unconditionally,
    so a rotate by a multiple of four, a same-size scale or resize, and a
    full-canvas crop each cost a full-document copy and an undo step that
    reverses nothing back to itself."""
    doc = Document.blank(8, 8)
    depth = len(doc.history)
    op(doc)
    assert len(doc.history) == depth


# -- inker-document-15: premultiplied scale rounds instead of truncating -------------------


def test_a_smooth_identity_scale_does_not_darken_a_translucent_pixel():
    """inker-document-15: ``.astype(np.uint8)`` on a premultiplied float
    truncates towards zero, and a premultiplied channel's fractional part is
    never negative -- so every partially transparent pixel was rounded
    *down* before the filter ran, a systematic darkening."""
    pixels = np.zeros((1, 1, 4), dtype=np.uint8)
    pixels[0, 0] = (201, 50, 80, 200)

    out = tf.scale(pixels, (1, 1), resample="smooth")

    assert tuple(int(c) for c in out[0, 0]) == (201, 50, 80, 200)


# -- inker-flourish-03: duplicate phase names get unique names in clamp() -----------------


def test_bake_of_two_identically_named_phases_renders_each_phases_own_frames():
    """inker-flourish-03: nothing kept two phases from clamping to the same
    name, and every phase-keyed lookup downstream (``bake()``'s cel dict,
    ``_doc_flourish._cel_for``) is keyed by that name -- so a duplicate
    silently aliased the two phases' frames through the same dict slot while
    ``Bake.tags()`` still reported two distinct spans."""
    recipe = Recipe(
        name="Dup", width=8, height=8, supersample=1, directions=1,
        phases=(Phase("main", 2, False), Phase("main", 3, False)),
        layers=(RecipeLayer(uid=1, kind="core", name="A"),),
    )

    clamped = clamp(recipe)

    assert clamped.phases[0].name != clamped.phases[1].name
    baked = B.bake(clamped)
    assert baked.frame_count == len(baked.flat())
    tags = baked.tags()
    assert {name for name, *_ in tags} == {clamped.phases[0].name, clamped.phases[1].name}


# -- inker-flourish-04: paint window sizing matches the erosion's own worst case -----------


def _ctx(width: int, height: int, *, scale: float = 1.0, frame: int = 0) -> FrameCtx:
    return FrameCtx(
        seed=1, width=width, height=height, scale=scale, frame=frame,
        phase=Phase("main", 12, True), phase_index=0, phase_frame=frame, fps=18, assets={},
    )


def _zero_noise(ctx, seed, **kw):
    """The erosion's worst case: no noise at all, so the shrink factor is
    exactly ``1 - amount`` everywhere in the window."""
    return np.zeros(kw["win"].x.shape, dtype=np.float32)


def test_core_window_reaches_the_erosions_worst_case_radius(monkeypatch):
    """inker-flourish-04: ``core``'s erosion scales ``d`` by as little as
    ``1 - amount``, so the true reach of a raw distance that still shows
    after erosion is ``radius / (1 - amount)``, not the ``radius * (1 +
    amount)`` the window used to size against -- a gap that grows without
    bound as ``amount`` approaches its own ceiling of 1."""
    monkeypatch.setattr(core_prim, "fbm_plane", _zero_noise)
    layer = (
        RecipeLayer(uid=1, kind="core")
        .with_param("radius", 2.0)
        .with_param("noise", 0.9)
        .with_param("pulse", 0.0)
    )
    ctx = _ctx(128, 128)

    out = core_prim.render(layer, ctx, None)

    # n=0 (monkeypatched) -> factor = 1 - 0.9 = 0.1; the eroded edge still
    # shows at raw distance 15, well past the old radius*(1+0.9)=3.8 window.
    assert out[64, 64 + 15, 3] > 0.0


def test_ring_window_reaches_the_erosions_worst_case_radius(monkeypatch):
    monkeypatch.setattr(ring_prim, "fbm_plane", _zero_noise)
    radius, thick = 200.0, 2.0
    layer = (
        RecipeLayer(uid=1, kind="ring")
        .with_param("radius", radius)
        .with_param("thickness", thick)
        .with_param("unevenness", 1.0)
        .with_param("alpha", 1.0)
    )
    ctx = _ctx(700, 700)

    out = ring_prim.render(layer, ctx, None)

    # n=0 -> factor = 1 - 0.25 = 0.75; the ring's true outer edge sits at
    # (radius+thick/2)/0.75 = 268, past the old radius*1.3+thick=262 window.
    d_true = 201.0 / 0.75
    assert out[350, 350 + int(round(d_true)), 3] > 0.0


def test_smoke_window_reaches_the_erosions_worst_case_radius(monkeypatch):
    monkeypatch.setattr(smoke_prim, "fbm_plane", _zero_noise)
    monkeypatch.setattr(smoke_prim.native, "available", lambda: False)
    radius = 40.0
    layer = (
        RecipeLayer(uid=1, kind="smoke")
        .with_param("count", 1)
        .with_param("emission", "burst")
        .with_param("size", radius)
        .with_param("expand", 0.0)
        .with_param("rise", 0.0)
        .with_param("drift", 0.0)
        .with_param("spawn_radius", 0.0)
        .with_param("raggedness", 1.0)
        .with_param("lifetime", 5.0)
    )
    # Frame 27 at 18fps puts the one burst-born particle at u~0.3, inside the
    # alpha-over-life curve's nonzero span (it is 0 at birth, u=0).
    ctx = _ctx(800, 800, frame=27)

    out = smoke_prim.render(layer, ctx, None)

    # n=0 -> factor = 1 - 0.8 = 0.2; visible out to radius/0.2=200, past the
    # old radius*1.8=72 window.
    d_true = radius / 0.2
    assert out[400, 400 + int(round(d_true)) - 10, 3] > 0.0


# -- inker-flourish-06: Recipe.from_dict never raises past its own docstring ----------------


def test_from_dict_refuses_nothing_but_value_error_on_infinite_or_non_numeric_fields():
    """inker-flourish-06: ``Recipe.from_dict`` documents ``ValueError``
    alone; bare ``int()`` calls on ``Infinity`` (a legal JSON-decoded float)
    or a non-numeric value raised ``OverflowError``/``TypeError`` instead."""
    inf = float("inf")
    raw = {
        "name": "X",
        "seed": inf,
        "size": [inf, "not-a-number"],
        "supersample": inf,
        "fps": "abc",
        "colors": inf,
        "directions": [1, 2],
        "phases": [{"name": "main", "frames": inf, "loop": False}],
        "layers": [{"uid": 1, "kind": "core", "params": {}}],
    }

    recipe = from_dict(raw)  # must not raise TypeError/OverflowError

    assert recipe.width == 128 and recipe.height == 128
    assert recipe.phases[0].frames == 12


# -- inker-mode-14: the walk renderer's RotSprite budget ------------------------------------


def test_a_part_with_an_off_centre_joint_still_uses_rotsprite():
    """inker-mode-14: a part turns about its own joint, not its centre,
    through ``render_transform_about``'s pad-to-pivot trick -- padding that
    roughly doubles whichever side the joint sits off-centre on,
    angle-independently. A 260x520 part with a joint near one end padded
    past ``ROTSPRITE_MAX_PIXELS`` and silently fell back to nearest-neighbour
    for a turn the module's own docstring says never happens."""
    width, height = 260, 520
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[..., 0] = 200
    pixels[..., 3] = 255
    part = R.Part(pixels=pixels, origin=(0, 0))
    spec = R.PartSpec("thigh", "hip", ("hip", "knee"))
    rest = R.Rig(joints={"hip": (130.0, 10.0), "knee": (130.0, 500.0)})
    pose = Pose(
        joints={"hip": (130.0, 10.0), "knee": (130.0, 500.0)},
        angles={"thigh": 15.0},
        grounded={},
    )

    calls: list[tuple[int, ...]] = []
    real_rotsprite = tf.rotsprite

    def spy(pixels, degrees, *, expand=False):
        calls.append(pixels.shape)
        return real_rotsprite(pixels, degrees, expand=expand)

    original = tf.rotsprite
    tf.rotsprite = spy
    try:
        result = wr.part_frame(part, spec, rest, pose)
    finally:
        tf.rotsprite = original

    assert result is not None
    assert len(calls) > 0, "an off-centre joint must not silently fall back to nearest-neighbour"


# -- inker-paint-01 / -02: _resolve_indices touches only what a stroke changed --------------


def test_a_stroke_keeps_the_slot_of_a_duplicate_pixel_its_round_nib_never_covers():
    """inker-paint-01: ``_resolve_indices`` used to re-``resolve`` every pixel
    of the stroke's dirty *rect* -- a round nib's square bounding box, not its
    circular footprint -- so a pixel the stamp never actually reached (its
    coverage is exactly 0 there, ``test_a_hard_stamp...``'s own corner) fell
    off the higher of two duplicate palette slots onto the lowest (``resolve``
    has no ``prefer`` for a pixel outside the current gesture). Only a real
    stroke exercises this: ``end_stroke`` is where the brush's own coverage
    buffer reaches ``_resolve_indices``."""
    palette = [BLACK, RED, GREEN, RED, (0, 0, 255, 100)]
    doc = Document.blank(8, 8)
    assert doc.convert_to_indexed(palette)
    layer = doc.stack[0]
    table = doc._index_lut()
    layer.indices[:, :] = 0
    layer.indices[0, 0] = 3  # the duplicate red, at the round nib's own corner
    layer.pixels[:, :] = ixp.materialize(layer.indices, table)

    doc.begin_stroke((4.0, 4.0), GREEN, size=8, hardness=1.0, nib="soft")
    doc.end_stroke()

    doc.check_materialized()
    assert layer.indices[0, 0] == 3
    assert layer.indices[4, 4] == 2  # inside the stamp: painted green


def test_a_translucent_palette_slot_survives_a_stroke_elsewhere_in_its_rect():
    """inker-paint-02: ``resolve`` treats alpha under ``OPAQUE_THRESHOLD`` as
    a hole, so a pixel on a translucent slot -- alpha under the threshold by
    design, not by accident -- that a round nib's dirty rect merely includes,
    without the stamp ever covering it, was re-inferred as a hole."""
    palette = [BLACK, RED, GREEN, RED, (0, 0, 255, 100)]
    doc = Document.blank(8, 8)
    assert doc.convert_to_indexed(palette)
    layer = doc.stack[0]
    table = doc._index_lut()
    layer.indices[:, :] = 0
    layer.indices[7, 7] = 4  # the translucent slot, at the opposite corner
    layer.pixels[:, :] = ixp.materialize(layer.indices, table)

    doc.begin_stroke((4.0, 4.0), GREEN, size=8, hardness=1.0, nib="soft")
    doc.end_stroke()

    doc.check_materialized()
    assert layer.indices[7, 7] == 4
    assert tuple(int(c) for c in layer.pixels[7, 7]) == (0, 0, 255, 100)


# -- inker-paint-03: blur and smudge premultiply -------------------------------------------


def _brush_layer(size=(32, 32)):
    pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    pixels[:, :16] = RED
    pixels[:, 16:] = (0, 0, 0, 0)
    return pixels


def _brush_stroke(pixels, **kw):
    from realmspinner.kernels.pixel import brush

    kw.setdefault("colour", RED)
    return brush.StrokeState(
        layer_uid=1, size=(pixels.shape[1], pixels.shape[0]), before=pixels.copy(), **kw
    )


def test_blur_brush_keeps_edge_colour_next_to_transparency():
    """inker-paint-03: the Gaussian blur mixed neighbouring pixels' RGB in
    straight alpha, so a fully transparent neighbour's black RGB dragged a
    red edge dark as it faded, instead of just losing coverage."""
    pixels = _brush_layer()
    stroke = _brush_stroke(pixels, diameter=16, hardness=1.0, mode="blur", strength=1.0)

    stroke.begin((16, 16), pixels)

    row = pixels[16, 10:21]
    faded = [tuple(int(c) for c in px) for px in row if 0 < px[3] < 255]
    assert faded, "the blur should leave a partially transparent band to check"
    assert all(r == 255 and g == 0 and b == 0 for r, g, b, a in faded)


# -- inker-paint-05: stale docstrings ------------------------------------------------------


def test_index_plane_docstring_names_native_as_its_one_import_exception():
    """inker-paint-05: the module claimed to import nothing under
    realmspinner; it has imported ``realmspinner.native`` since the native
    palette kernels landed, which the import-pin test's own allowlist
    (``tests/modes/inker/test_inker_imports.py``) already tracks by name."""
    from realmspinner.kernels.pixel import index_plane

    doc = index_plane.__doc__
    assert "importing nothing under ``realmspinner``." not in doc
    assert "native" in doc


def test_indexed_docstring_acknowledges_the_index_plane_module_coexists():
    """inker-paint-05: ``indexed.py``'s header still argued a document that
    stored an index plane "would be a rewrite of the whole package" -- true
    indexed mode (``index_plane.py``) had already shipped and does exactly
    that, for the one thing constrain-on-write cannot represent."""
    from realmspinner.kernels.pixel import indexed

    doc = indexed.__doc__
    assert "index_plane" in doc
