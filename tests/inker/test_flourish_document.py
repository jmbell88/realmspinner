"""An effect on a document: insert, regenerate, the painted-cel rule, undo as
one step, and what survives a save to each format."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from warlock.studio import inker
from warlock.studio.inker import _doc_flourish, flourish, ora
from warlock.studio.inker.flourish import bake as B
from warlock.studio.inker.flourish import presets
from warlock.studio.inker.flourish.bake import Bake, Facing
from warlock.studio.inker.flourish.recipe import Layer as RecipeLayer
from warlock.studio.inker.flourish.recipe import Phase, Recipe

_PUFF = presets.load("smoke_puff")


def _recipe(name: str = "smoke_puff", **over):
    """One loaded preset, so every variant shares its layer uids -- the
    situation a regenerate is in: the recipe edited, not reloaded."""
    rec = _PUFF if name == "smoke_puff" else presets.load(name)
    return dataclasses.replace(rec, width=32, height=32, supersample=2, **over)


def _small_layer_recipe():
    """A two-layer painterly recipe whose geometry fits a 32px canvas."""
    rec = presets.load("sword_impact")
    return dataclasses.replace(rec, width=32, height=32, supersample=2)


def _hand_recipe(*, with_layer: bool) -> Recipe:
    layers = (RecipeLayer(uid=1, kind="fill", name="A"),) if with_layer else ()
    return Recipe(
        name="Hand",
        width=4,
        height=4,
        phases=(Phase("main", 2, False),),
        layers=layers,
    )


def _hand_bake(frame1_content: np.ndarray | None) -> Bake:
    """A bake built by hand rather than rendered, so the second frame's
    pixels are exactly whatever the caller hands it -- fully transparent, in
    particular, which no shipped preset can be relied on to line up on
    demand. ``None`` omits the layer from the recipe entirely, which is the
    bake ``insert_flourish`` gets first, so the track this test cares about
    is *added* by the regenerate rather than already present with a cel in
    it -- the ``existing is None`` branch the audit found the bug in.

    Frame 0 is always opaque paint; frame 1 is ``frame1_content``.
    """
    opaque = np.zeros((4, 4, 4), np.uint8)
    opaque[..., :3] = (200, 40, 40)
    opaque[..., 3] = 255
    if frame1_content is None:
        recipe = _hand_recipe(with_layer=False)
        facing = Facing(name="E", degrees=0.0, composites={"main": [opaque, opaque]}, layers={})
    else:
        recipe = _hand_recipe(with_layer=True)
        facing = Facing(
            name="E",
            degrees=0.0,
            composites={"main": [opaque, frame1_content]},
            layers={"main": {1: [opaque, frame1_content]}},
        )
    return Bake(recipe=recipe, facings=[facing], palette=None, palette_source="none")


def test_apply_flourish_records_no_digest_for_a_frame_it_wrote_nothing_to():
    """The 2026-09-07 audit (inker-09): ``apply_flourish`` recorded a digest
    for every rendered frame of a newly-added track, even the fully
    transparent ones the cel-creation branch right above it declines to turn
    into a cel -- 18 phantom entries from one ordinary call. A digest with no
    cel behind it means a later hand-drawn edit on that blank frame is judged
    against a digest that could never match real pixels."""
    doc = inker.Document.blank(4, 4)
    group = doc.insert_flourish(_hand_bake(None))
    state = doc.flourish_state(group)
    assert state.tracks == {}, "the layer was not in the first bake at all"

    blank = np.zeros((4, 4, 4), np.uint8)  # fully transparent: no ink at all
    counts = doc.apply_flourish(group, _hand_bake(blank), force=True)
    assert counts.added == 1
    state = doc.flourish_state(group)
    track_uid = next(iter(state.tracks.values()))
    frame_uids = [f.uid for f in doc.anim.frames[:2]]

    # The blank frame got no cel -- and, the fix, no digest for it either.
    assert doc.anim.cels.get((track_uid, frame_uids[1])) is None
    assert (track_uid, frame_uids[1]) not in state.digests
    # The opaque frame, which really was written, keeps its digest.
    assert doc.anim.cels.get((track_uid, frame_uids[0])) is not None
    assert (track_uid, frame_uids[0]) in state.digests


def test_insert_makes_a_group_of_tracks_with_a_tag_per_phase_in_one_step():
    doc = inker.Document.blank(32, 32)
    baked = B.bake(_small_layer_recipe())
    group = doc.insert_flourish(baked)
    assert doc.anim is not None
    assert len(doc.anim.frames) == baked.frame_count
    assert [t.name for t in doc.anim.tags] == [p.name for p in baked.recipe.phases]
    state = doc.flourish_state(group)
    assert state is not None
    members = [uid for uid, g in doc.group_of.items() if g == group]
    assert set(members) == set(state.tracks.values())
    assert len(members) == len(baked.recipe.layers)
    assert doc.groups[group].name == baked.recipe.name
    # One undo step back to a still document with no effect.
    doc.history.undo(doc)
    assert doc.anim is None
    assert doc.flourish == {} and doc.groups == {}
    doc.history.redo(doc)
    assert doc.flourish_state(group) is not None
    assert len(doc.anim.tracks) == 1 + len(baked.recipe.layers)


def test_the_effect_lands_centred_on_a_larger_document():
    doc = inker.Document.blank(64, 48)
    baked = B.bake(_recipe())
    group = doc.insert_flourish(baked)
    state = doc.flourish_state(group)
    assert state.offset == (16, 8)
    track_uid = next(iter(state.tracks.values()))
    frame = doc.anim.frames[6]
    cel = doc.anim.cels[(track_uid, frame.uid)]
    assert cel.pixels.shape == (48, 64, 4)
    # Nothing outside the recipe's canvas.
    assert not cel.pixels[:8].any() and not cel.pixels[:, :16].any()


def test_a_pixel_bake_is_one_track():
    doc = inker.Document.blank(32, 32)
    baked = B.bake(_recipe(mode="pixel", colors=6))
    group = doc.insert_flourish(baked)
    state = doc.flourish_state(group)
    assert list(state.tracks) == [_doc_flourish.COMPOSITE]


def test_the_active_layer_finds_its_effect():
    doc = inker.Document.blank(32, 32)
    baked = B.bake(_small_layer_recipe())
    group = doc.insert_flourish(baked)
    assert doc.flourish_group_of_active() == group
    doc.set_active_layer(0)  # the original background
    assert doc.flourish_group_of_active() is None


def test_regenerate_with_the_same_recipe_changes_nothing_but_is_one_step():
    doc = inker.Document.blank(32, 32)
    baked = B.bake(_recipe())
    group = doc.insert_flourish(baked)
    head = doc.history.head
    counts = doc.apply_flourish(group, baked)
    assert counts.taken == 0 and counts.conflicts == 0
    assert counts.agreed == baked.frame_count
    assert doc.history.head == head + 1
    doc.history.undo(doc)
    assert doc.history.head == head


def test_a_regenerate_keeps_the_groups_textures():
    """``apply_flourish``'s own ``FlourishState`` construction used to omit
    ``assets`` altogether, which defaults to an *empty* dict
    (``FlourishState.assets``'s ``default_factory``) -- so a plain
    regenerate, with no texture in flight at all, silently dropped every
    texture the group already held, whatever the recipe still named by id.
    Found chasing the 2026-09-14 audit's inker-06 second half (a texture just
    picked up would otherwise vanish the instant the render that names it
    lands), but it is a bug in its own right: nothing had exercised a
    regenerate closely enough after adding an asset to notice it."""
    doc = inker.Document.blank(32, 32)
    baked = B.bake(_recipe())
    group = doc.insert_flourish(baked)
    tex = np.full((4, 4, 4), 255, dtype=np.uint8)
    doc.add_flourish_asset(group, tex)
    assert list(doc.flourish_state(group).assets) == ["tex1"]
    doc.apply_flourish(group, baked)
    assert list(doc.flourish_state(group).assets) == ["tex1"], (
        "a regenerate must not drop the group's existing textures"
    )


def test_regenerate_takes_untouched_cels_and_keeps_painted_ones():
    doc = inker.Document.blank(32, 32)
    first = B.bake(_recipe(seed=1))
    group = doc.insert_flourish(first)
    state = doc.flourish_state(group)
    track_uid = next(iter(state.tracks.values()))
    # Paint on frame 3.
    frame = doc.anim.frames[3]
    painted = doc.anim.cels[(track_uid, frame.uid)]
    painted.pixels[0, 0] = (255, 0, 255, 255)
    second = B.bake(_recipe(seed=2))
    counts = doc.apply_flourish(group, second)
    assert counts.conflicts == 1 and counts.kept == 1
    assert counts.taken >= 1
    assert doc.flourish_conflicts(group) == [3]
    # The paint stands.
    assert tuple(doc.anim.cels[(track_uid, frame.uid)].pixels[0, 0]) == (255, 0, 255, 255)
    # Another untouched frame took the new render.
    other = doc.anim.frames[6]
    got = doc.anim.cels[(track_uid, other.uid)].pixels
    expected = _doc_flourish._place(second.flat()[6], (32, 32), (0, 0))
    assert np.array_equal(got, expected)
    # One Ctrl+Z puts the pixels and the record back together.
    doc.history.undo(doc)
    assert doc.flourish_conflicts(group) == []
    assert np.array_equal(
        doc.anim.cels[(track_uid, other.uid)].pixels,
        _doc_flourish._place(first.flat()[6], (32, 32), (0, 0)),
    )


def test_force_replaces_the_painted_cel():
    doc = inker.Document.blank(32, 32)
    group = doc.insert_flourish(B.bake(_recipe(seed=1)))
    state = doc.flourish_state(group)
    track_uid = next(iter(state.tracks.values()))
    frame = doc.anim.frames[3]
    doc.anim.cels[(track_uid, frame.uid)].pixels[0, 0] = (255, 0, 255, 255)
    counts = doc.apply_flourish(group, B.bake(_recipe(seed=2)), force=True)
    assert counts.conflicts == 0
    assert tuple(doc.anim.cels[(track_uid, frame.uid)].pixels[0, 0]) != (255, 0, 255, 255)


def test_resolve_clears_flags_as_its_own_step():
    doc = inker.Document.blank(32, 32)
    group = doc.insert_flourish(B.bake(_recipe(seed=1)))
    track_uid = next(iter(doc.flourish_state(group).tracks.values()))
    doc.anim.cels[(track_uid, doc.anim.frames[2].uid)].pixels[0, 0] = (1, 2, 3, 255)
    doc.apply_flourish(group, B.bake(_recipe(seed=2)))
    assert doc.flourish_conflicts(group) == [2]
    assert doc.resolve_flourish(group, [2])
    assert doc.flourish_conflicts(group) == []
    assert not doc.resolve_flourish(group, [2])
    doc.history.undo(doc)
    assert doc.flourish_conflicts(group) == [2]


def test_a_new_recipe_layer_gets_a_track_inside_the_group():
    doc = inker.Document.blank(32, 32)
    rec = _recipe()
    group = doc.insert_flourish(B.bake(rec))
    before = len(doc.anim.tracks)
    extra = dataclasses.replace(
        rec, layers=(*rec.layers, flourish.Layer(uid=flourish.new_uid(), kind="flash"))
    )
    counts = doc.apply_flourish(group, B.bake(flourish.clamp(extra)))
    assert counts.added == 1
    assert len(doc.anim.tracks) == before + 1
    state = doc.flourish_state(group)
    for track_uid in state.tracks.values():
        assert doc.group_of.get(track_uid) == group


def test_more_frames_are_appended_when_a_phase_grows():
    doc = inker.Document.blank(32, 32)
    rec = _recipe()
    group = doc.insert_flourish(B.bake(rec))
    longer = dataclasses.replace(rec, phases=(dataclasses.replace(rec.phases[0], frames=20),))
    doc.apply_flourish(group, B.bake(longer))
    assert len(doc.anim.frames) == 20


def test_regenerate_after_shrinking_a_phase_updates_the_tag_and_flags_the_orphaned_frames():
    """The 2026-09-13 audit (inker-02): ``apply_flourish`` never rewrote
    ``Document.anim.tags``. A phase that grows leaves its new frames outside
    every tag (checked by ``test_more_frames_are_appended_when_a_phase_grows``,
    which only counted frames); a phase that *shrinks* is the other half --
    the tag still spans the now-stale cels past its new end, and play or a
    per-tag export would show them with no notice."""
    doc = inker.Document.blank(32, 32)
    rec = _recipe()
    group = doc.insert_flourish(B.bake(rec))
    tag = next(t for t in doc.anim.tags if t.name == rec.phases[0].name)
    assert tag.end == rec.phases[0].frames - 1
    track_uid = next(iter(doc.flourish_state(group).tracks.values()))
    old_end = tag.end

    shorter = dataclasses.replace(
        rec, phases=(dataclasses.replace(rec.phases[0], frames=old_end),)
    )
    doc.apply_flourish(group, B.bake(shorter))

    new_tag = next(t for t in doc.anim.tags if t.name == rec.phases[0].name)
    assert new_tag.end == old_end - 1
    # The last frame of the old span held a rendered cel and is no longer
    # covered by the tag: it must be flagged, not left to look untouched.
    last_frame_uid = doc.anim.frames[old_end].uid
    assert (track_uid, last_frame_uid) in doc.flourish_state(group).conflicts
    assert old_end in doc.flourish_conflicts(group)


def test_detach_forgets_the_recipe_and_keeps_the_layers():
    doc = inker.Document.blank(32, 32)
    group = doc.insert_flourish(B.bake(_recipe()))
    tracks = len(doc.anim.tracks)
    assert doc.detach_flourish(group)
    assert doc.flourish_state(group) is None
    assert len(doc.anim.tracks) == tracks
    assert group in doc.groups
    doc.history.undo(doc)
    assert doc.flourish_state(group) is not None


def test_set_recipe_without_rendering_is_a_step_and_a_noop_when_equal():
    doc = inker.Document.blank(32, 32)
    rec = _recipe()
    group = doc.insert_flourish(B.bake(rec))
    assert not doc.set_flourish_recipe(group, rec)
    renamed = dataclasses.replace(rec, name="Puff 2")
    assert doc.set_flourish_recipe(group, renamed)
    assert doc.flourish_state(group).recipe.name == "Puff 2"
    doc.history.undo(doc)
    assert doc.flourish_state(group).recipe.name == rec.name


def test_apply_refuses_a_group_that_is_not_an_effect():
    doc = inker.Document.blank(32, 32)
    with pytest.raises(ValueError):
        doc.apply_flourish(999, B.bake(_recipe()))


def test_regenerate_with_a_linked_cel_raises_before_mutating_anything():
    """Finding #1. A linked cel used to be found mid-loop, after earlier
    frames of the same track were already rewritten and no undo step was
    pushed to cover them -- the raise that reached ``land`` left the document
    half-rendered. Every target must be checked before any of them is
    touched."""
    doc = inker.Document.blank(32, 32)
    first = B.bake(_recipe(seed=1))
    group = doc.insert_flourish(first)
    state = doc.flourish_state(group)
    track_uid = next(iter(state.tracks.values()))
    track_index = next(i for i, t in enumerate(doc.anim.tracks) if t.uid == track_uid)
    # Link frame 5's cel to frame 0's, so the two share one object -- the
    # regenerate must refuse it. Frame 0 is earlier in the render loop than
    # frame 5, so mutation of the earlier frames is exactly what a mid-loop
    # raise would have already done.
    assert doc.link_cel(0, track_index=track_index, frame_index=5)
    assert doc.anim.is_linked(track_uid, doc.anim.frames[5].uid)
    head = doc.history.head
    before_pixels = {
        i: doc.anim.cels[(track_uid, f.uid)].pixels.copy()
        for i, f in enumerate(doc.anim.frames)
        if (track_uid, f.uid) in doc.anim.cels
    }
    second = B.bake(_recipe(seed=2))
    with pytest.raises(ValueError, match="unlink"):
        doc.apply_flourish(group, second)
    # No undo step, and nothing -- including the frames the loop would have
    # reached before the linked one -- was rewritten.
    assert doc.history.head == head
    for i, pixels in before_pixels.items():
        assert np.array_equal(doc.anim.cels[(track_uid, doc.anim.frames[i].uid)].pixels, pixels)


def test_apply_flourish_leaves_no_added_frames_when_it_refuses_a_linked_cel_after_growing_the_grid():  # noqa: E501
    """The 2026-09-14 audit (inker-01): ``_flourish_ensure_frames`` used to
    run *before* the linked-cel validation loop above, so on a recipe whose
    phase grew, a refused regenerate still appended the new frames the
    would-be grid needed -- real blank frames left in the document with
    nothing in ``edits`` to cover them, because the raise unwinds before
    ``self.history.push`` ever runs. The test above only covers a grid that
    does not grow; this one covers the growing case the bug actually lived in."""
    doc = inker.Document.blank(32, 32)
    rec = _recipe()
    group = doc.insert_flourish(B.bake(rec))
    state = doc.flourish_state(group)
    track_uid = next(iter(state.tracks.values()))
    track_index = next(i for i, t in enumerate(doc.anim.tracks) if t.uid == track_uid)
    assert doc.link_cel(0, track_index=track_index, frame_index=1)
    assert doc.anim.is_linked(track_uid, doc.anim.frames[1].uid)

    frame_count_before = len(doc.anim.frames)
    head = doc.history.head
    longer = dataclasses.replace(
        rec, phases=(dataclasses.replace(rec.phases[0], frames=frame_count_before + 10),)
    )
    with pytest.raises(ValueError, match="unlink"):
        doc.apply_flourish(group, B.bake(longer))

    assert len(doc.anim.frames) == frame_count_before
    assert doc.history.head == head


def test_the_recipe_survives_an_ora_round_trip(tmp_path):
    doc = inker.Document.blank(40, 40)
    rec = _recipe(seed=5)
    group = doc.insert_flourish(B.bake(rec))
    track_uid = next(iter(doc.flourish_state(group).tracks.values()))
    doc.anim.cels[(track_uid, doc.anim.frames[1].uid)].pixels[2, 2] = (9, 9, 9, 255)
    doc.apply_flourish(group, B.bake(_recipe(seed=6)))
    assert doc.flourish_conflicts(group) == [1]
    path = tmp_path / "puff.ora"
    ora.write_ora(doc, path)
    again = inker.Document.load(path)
    assert len(again.flourish) == 1
    (guid, state), = again.flourish.items()
    assert guid in again.groups
    assert state.recipe == doc.flourish_state(group).recipe
    assert state.offset == (4, 4)
    assert len(state.tracks) == 1
    assert again.flourish_conflicts(guid) == [1]
    # And it still regenerates: the digests came back, so untouched cels take.
    counts = again.apply_flourish(guid, B.bake(_recipe(seed=7)))
    assert counts.conflicts == 1 and counts.taken >= 1


def test_opening_a_flourish_document_preserves_its_layer_uids(tmp_path):
    """The 2026-09-14 audit (inker-09): INVARIANTS and ``render.FrameCtx``'s
    own comment both said a layer's uid is reissued on every load, when
    ``_read_flourish`` calls ``recipe.from_dict`` directly and that keeps
    whatever uid the file carries -- only a *second* insert of the same
    preset reissues one (``presets.load`` -> ``recipe.bump_uids``). Pin the
    real behaviour, and pin the comment against drifting back into the wrong
    claim: a docstring assertion that fails against the old wording is the
    only proof a doc-only finding has that it once failed."""
    import inspect

    from warlock.studio.inker.flourish import render as render_mod

    doc = inker.Document.blank(40, 40)
    rec = _recipe(seed=5)
    group = doc.insert_flourish(B.bake(rec))
    original_uids = [layer.uid for layer in doc.flourish_state(group).recipe.layers]
    assert original_uids  # the preset actually has layers to lose uids from

    path = tmp_path / "uid_puff.ora"
    ora.write_ora(doc, path)
    again = inker.Document.load(path)
    (guid, state), = again.flourish.items()
    assert [layer.uid for layer in state.recipe.layers] == original_uids

    source = inspect.getsource(render_mod.FrameCtx)
    assert "uids are reissued on every load" not in source
    assert "uid is not reissued on load" in source


def test_read_flourish_refuses_past_a_metadata_ceiling(tmp_path, monkeypatch):
    """The 2026-09-16 audit found ``animation.json``'s "flourish" list had no
    ceiling at all, unlike every sibling list this module already bounds
    (``"tracks"``, ``"frames"``, a per-frame ``"palette"``): a crafted file
    could repeat one valid entry thousands of times and build one real
    ``FlourishState`` -- a full ``Recipe`` parse -- per copy with no refusal.
    Pins that the loop never starts once the list is over
    ``MAX_ORA_METADATA_ENTRIES``, by counting calls into the recipe parser
    rather than trusting the final ``doc.flourish`` size, which a repeated
    entry would leave looking identical either way (same uid, overwritten)."""
    import json
    import zipfile

    from warlock.studio.inker.flourish import recipe as flourish_recipe

    doc = inker.Document.blank(40, 40)
    rec = _recipe(seed=5)
    doc.insert_flourish(B.bake(rec))
    path = tmp_path / "many.ora"
    ora.write_ora(doc, path)

    with zipfile.ZipFile(path) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    payload = json.loads(members[ora.ANIMATION_MEMBER])
    entry = payload["flourish"][0]
    payload["flourish"] = [entry] * (ora.MAX_ORA_METADATA_ENTRIES + 1)
    members[ora.ANIMATION_MEMBER] = json.dumps(payload).encode("utf-8")
    broken = tmp_path / "broken.ora"
    with zipfile.ZipFile(broken, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)

    calls: list[int] = []
    original = flourish_recipe.from_dict

    def _counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(flourish_recipe, "from_dict", _counting)

    back = inker.Document.load(broken)
    assert calls == []
    assert not back.flourish


def test_an_ordinary_document_writes_no_flourish_key(tmp_path):
    doc = inker.Document.blank(8, 8)
    doc.add_frame()
    path = tmp_path / "plain.ora"
    ora.write_ora(doc, path)
    import zipfile

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        grid = next(n for n in names if n.endswith("animation.json"))
        assert b"flourish" not in zf.read(grid)


def test_aseprite_keeps_the_layers_and_drops_the_recipe(tmp_path):
    from warlock.studio.inker import asein, aseout

    doc = inker.Document.blank(32, 32)
    group = doc.insert_flourish(B.bake(_small_layer_recipe()))
    tracks = len(doc.anim.tracks)
    path = tmp_path / "fx.aseprite"
    aseout.write_aseprite(doc, path)
    again, _warnings = asein.document_from_aseprite(path.read_bytes())
    assert again.flourish == {}
    assert len(again.anim.tracks) == tracks
    assert [t.name for t in again.anim.tags] == [t.name for t in doc.anim.tags]
    assert group not in again.flourish


def test_flourish_conflicts_does_not_rebuild_the_frame_index_when_nothing_changed():
    """The 2026-09-13 audit, finding inker-07: the Flourish inspector calls
    ``flourish_conflicts`` two or three times per frame while it is open, and
    each call used to rebuild a frame-uid -> index dict over the whole
    animation from scratch. A second call with nothing changed must answer
    from the memo rather than walking ``anim.frames`` again -- proven here by
    monkeypatching ``enumerate`` calls off the table: instead, count calls
    into the frame-index build by wrapping ``anim.frames`` access is fragile,
    so this instruments the one thing that walk actually does: build a dict
    the size of the animation. A cheap proxy that is exact enough not to lie
    either way is to count how many times ``state.conflicts`` gets read while
    computing the *signature* is unavoidable (a cache still has to check
    freshness) -- what must NOT happen again is a fresh dict comprehension
    over every frame. That is asserted directly against the cache field.
    """
    doc = inker.Document.blank(32, 32)
    group = doc.insert_flourish(B.bake(_recipe(seed=1)))
    track_uid = next(iter(doc.flourish_state(group).tracks.values()))
    doc.anim.cels[(track_uid, doc.anim.frames[2].uid)].pixels[0, 0] = (1, 2, 3, 255)
    doc.apply_flourish(group, B.bake(_recipe(seed=2)))

    first = doc.flourish_conflicts(group)
    assert first == [2]
    state = doc.flourish_state(group)
    cached_after_first = state._conflicts_cache
    assert cached_after_first is not None

    # A second call with nothing changed must reuse the same cached result
    # object rather than recomputing it -- on the unfixed code, every call
    # rebuilds ``at`` and a new ``result`` list, so this identity check fails.
    second = doc.flourish_conflicts(group)
    assert second == [2]
    assert second is cached_after_first[1]

    # Second assertion: an edit that changes the answer -- resolving the
    # conflict -- must invalidate the memo rather than serve the stale list.
    assert doc.resolve_flourish(group, [2])
    assert doc.flourish_conflicts(group) == []
