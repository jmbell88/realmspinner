"""Regressions for the 2026-09-26 audit's inker-mode-01, inker-flourish-02 and
inker-codecs-01 findings. Independent defects, independent tests -- see each
test's docstring for the finding it closes.
"""

from __future__ import annotations

import dataclasses
import struct
from types import SimpleNamespace

import pytest

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.pixel import asein, aseout, ora
from realmspinner.kernels.pixel.flourish import bake as B
from realmspinner.kernels.pixel.flourish import presets
from realmspinner.studio import state as state_mod
from realmspinner.studio.modes.inker import flourish as inker_flourish
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker import ops as inker_ops
from realmspinner.studio.modes.inker import state as inker_state
from realmspinner.studio.tasks import Done


def _op(name):
    return next(op for op in inker_ops.OPS if op.name == name)


def test_next_and_previous_frame_ops_step_the_playhead():
    """inker-mode-01: ``next_frame``/``prev_frame`` were wired through
    ``_mode("step_frame", delta=1)``, which calls
    ``inker_mode.step_frame(ctx, tab, delta=1)`` -- but ``playback.step_frame``
    takes ``delta`` positionally *before* ``tab``
    (``step_frame(ctx, delta, tab=None)``), the one verb that does not follow
    every other ``_mode`` target's ``(ctx, tab, **kwargs)`` shape. That raised
    ``TypeError: step_frame() got multiple values for argument 'delta'`` for
    both menu rows and the ``.``/``,`` keys.
    """
    doc = inker.Document.blank(8, 8)
    doc.ensure_animation()
    doc.add_frame()
    doc.add_frame()
    doc.set_current_frame(0)
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")
    state = inker_state.InkerState()
    state.add(tab)
    app = SimpleNamespace(inker=state, toasts=[])
    from types import MethodType

    app.toast = MethodType(state_mod.AppState.toast, app)
    app.toast_once = MethodType(state_mod.AppState.toast_once, app)
    settings = SimpleNamespace(get=lambda key: {}, set=lambda key, value: None)
    ctx = SimpleNamespace(state=app, toast=app.toast, settings=settings)

    assert doc.anim.current == 0
    _op("next_frame").run(ctx, tab)
    assert doc.anim.current == 1
    _op("next_frame").run(ctx, tab)
    assert doc.anim.current == 2
    _op("prev_frame").run(ctx, tab)
    assert doc.anim.current == 1


class _Ctx:
    """Runs tasks and records toasts; mirrors ``test_flourish_ops.py``'s fake."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(inker=inker_state.InkerState())
        self.toasts: list[tuple[str, str]] = []
        self.tasks = SimpleNamespace(set_progress=lambda *a, **k: None)
        self.pending: list[Done] = []
        self.auto_land = True
        self._busy: set[str] = set()

    def toast(self, text: str, level: str = "info", **_: object) -> None:
        self.toasts.append((text, level))

    def busy(self, key: str) -> bool:
        return key in self._busy

    def submit(self, key: str, fn, *args, **kwargs) -> bool:
        if key in self._busy:
            return False
        try:
            done = Done(key=key, result=fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 -- the runner reports, never raises
            done = Done(key=key, error=exc)
        if self.auto_land:
            inker_mode.on_task_done(self, done)
        else:
            self._busy.add(key)
            self.pending.append(done)
        return True

    def land_all(self) -> None:
        while self.pending:
            done = self.pending.pop(0)
            self._busy.discard(done.key)
            inker_mode.on_task_done(self, done)


def _open(ctx: _Ctx, size=(32, 32)) -> inker_state.InkerDoc:
    tab = inker_state.InkerDoc(doc=inker.Document.blank(*size))
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    return tab


def _small(name: str = "smoke_puff"):
    return dataclasses.replace(presets.load(name), width=32, height=32, supersample=2)


@pytest.fixture
def ctx():
    return _Ctx()


def test_a_flourish_render_landing_on_a_busy_tab_is_refused(ctx):
    """inker-flourish-02: ``land`` pushed ``apply_flourish``'s history step
    with no re-check of ``tab.busy``, unlike the tileset landing
    (``_done_tileset_import`` re-checks ``busy`` at completion, not just at
    submit). A save or Play started while a bake was in flight saw a
    half-mutated layer stack.
    """
    tab = _open(ctx)
    state = ctx.state.inker
    rec = _small()
    group = tab.doc.insert_flourish(B.bake(rec))
    head = tab.doc.history.head
    edited = dataclasses.replace(rec, seed=99)
    inker_flourish.set_pending(state, group, edited, now=10.0)

    ctx.auto_land = False
    sent = inker_flourish.tick(ctx, state, tab, now=10.0 + inker_flourish.DEBOUNCE_SECONDS)
    assert sent == 1
    assert len(ctx.pending) == 1
    done = ctx.pending.pop(0)
    ctx._busy.discard(done.key)

    # A save starts while the bake result is sitting in the task queue --
    # ``land`` is called directly here (rather than through ``on_task_done``,
    # which stamps ``now`` from the real clock) so the test controls ``now``.
    tab.saving = True
    assert inker_flourish.land(ctx, state, done, now=15.0) is False

    assert tab.doc.history.head == head, "the busy tab's stack must not move"
    assert tab.doc.flourish_state(group).recipe == rec
    # Left due, so ``tick`` sends it again once the tab is free -- not lost.
    assert state.flourish_due.get(group) == 15.0

    tab.saving = False
    ctx.auto_land = True
    assert inker_flourish.tick(ctx, state, tab, now=15.0) == 1
    assert tab.doc.history.head == head + 1
    assert tab.doc.flourish_state(group).recipe == edited


def _animated_doc(size: tuple[int, int], frames: int) -> inker.Document:
    doc = inker.Document.blank(*size)
    doc.ensure_animation()
    for _ in range(frames - 1):
        # ``copy=True`` so each frame gets its own ``Layer`` object -- a
        # distinct decoded plane, exactly as one more drawn frame would --
        # rather than a link, which the reader (rightly) does not charge.
        doc.add_frame(copy=True)
    return doc


def test_a_clip_the_editor_can_build_is_a_clip_ora_and_aseprite_can_reopen():
    """inker-codecs-01: the editor's ``add_frame`` carries no pixel/plane
    budget, but both readers refuse an animation with more distinct cels than
    ``_layer_budget`` (``pixelguard.MAX_DECODE_PIXELS`` / canvas pixels) --
    so a 70-frame, 1024x1024 clip (each frame its own cel, well within what
    ``add_frame`` lets you build) saved to a 20 KB ``.ora`` and a 289 KB
    ``.aseprite`` that this same build then refused to reopen, and journal
    recovery's ``ora_bytes`` path hit the identical wall.

    ``_doc_anim.py``'s ``add_frame`` is not one of this fixer's owned files,
    so the budget is enforced on the writers instead (one of the two shapes
    the finding names): ``write_ora``/``ora_bytes`` and
    ``aseout.aseprite_bytes`` now refuse, by name, before writing a single
    byte, an animation with more distinct cels than this build can reopen --
    turning the silent "saved but unopenable" file into a loud refusal at
    save time. The editor itself still lets you build past that point
    (``_doc_anim.py::add_frame`` needs the identical guard for the refusal to
    happen at the point the extra frame is added, not just at Save).
    """
    size = (1024, 1024)
    allowed = ora._layer_budget(*size)
    assert allowed < 70  # the fixture, not the claim -- see _layer_budget

    small = _animated_doc(size, allowed)
    assert len(list(small.anim.unique_cel_layers())) == allowed
    ora.ora_bytes(small)  # at the budget: still reopens
    aseout.aseprite_bytes(small)

    over = _animated_doc(size, 70)
    assert len(list(over.anim.unique_cel_layers())) == 70
    with pytest.raises(ValueError, match="distinct cels"):
        ora.ora_bytes(over)
    with pytest.raises(ValueError, match="distinct cels"):
        aseout.aseprite_bytes(over)


# --- a minimal .aseprite builder, just what the test below needs -----------
# (mirrors ``test_asein.py``'s builders, which this fixer does not own and so
# does not edit; kept small on purpose.)


def _chunk(kind: int, payload: bytes) -> bytes:
    return struct.pack("<IH", len(payload) + 6, kind) + payload


def _ase_frame(chunks: list[bytes], duration: int = 100) -> bytes:
    body = b"".join(chunks)
    return struct.pack("<IHHHHI", len(body) + 16, 0xF1FA, len(chunks), duration, 0, 0) + body


def _ase_header(frames: int, width: int, height: int) -> bytes:
    head = struct.pack(
        "<IHHHHHIHIIB3sHBBhhHH",
        0, 0xA5E0, frames, width, height, 32, 1, 100, 0, 0, 0, b"\0\0\0", 0, 1, 1, 0, 0, 0, 0,
    )
    return head + b"\0" * 84


def _ase_file(header: bytes, frames: list[bytes]) -> bytes:
    body = header + b"".join(frames)
    return struct.pack("<I", len(body)) + body[4:]


def _ase_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<H", len(raw)) + raw


def _ase_layer_named(
    name: str = "Art", *, background: bool = False, reference: bool = False
) -> bytes:
    flags = 1 | 2 | (8 if background else 0) | (64 if reference else 0)
    body = struct.pack("<HHHHHHB3s", flags, 0, 0, 0, 0, 0, 255, b"\0\0\0") + _ase_string(name)
    return _chunk(0x2004, body)


def _ase_cel(
    layer: int, pixels: bytes, width: int, height: int, *, x: int = 0, y: int = 0
) -> bytes:
    return _chunk(
        0x2005,
        struct.pack("<HhhBHh5s", layer, x, y, 255, 0, 0, b"\0" * 5)
        + struct.pack("<HH", width, height)
        + pixels,
    )


def test_many_large_offcanvas_cels_are_refused_before_they_are_inflated(monkeypatch):
    """inker-codecs-02: each cel's own decompressed size was already bounded
    (``MAX_DECOMPRESSED_BYTES``, up to ~1 GiB), but nothing summed that total
    across every cel a file may declare -- and the one running total
    ``_build_cels`` did keep charged the *canvas*'s pixel count per cel, not
    the cel's own (possibly far larger, off-canvas) rectangle, so a 654 KB
    file naming many large off-canvas cels retained 1.34 GB before this fix.

    ``pixelguard.MAX_DECODE_PIXELS`` is lowered here, the way
    ``test_an_animated_aseprite_with_many_real_cels_has_a_pixel_budget`` in
    ``test_asein.py`` already does, so two small, individually-legal cels
    trip the *cel-pixel* running total without a multi-hundred-megabyte
    fixture. The canvas itself is tiny (2x2 = 4 pixels) so the old,
    canvas-charging total (2 cels x 4 = 8) would never have tripped at this
    budget -- only a running total in the cels' own pixels (2 x 16 = 32)
    catches it.
    """
    from realmspinner.core.safeio import pixelguard

    monkeypatch.setattr(pixelguard, "MAX_DECODE_PIXELS", 30)
    big = bytes(4) * (4 * 4)  # 4x4 RGBA, all zero -- the content is irrelevant
    frames = [_ase_frame([_ase_layer_named("Art"), _ase_cel(0, big, 4, 4, x=100, y=100)])]
    frames += [_ase_frame([_ase_cel(0, big, 4, 4, x=100, y=100)])]
    data = _ase_file(_ase_header(len(frames), 2, 2), frames)
    with pytest.raises(ValueError, match="pixels"):
        asein.document_from_aseprite(data)


def _ase_palette(colours: list[tuple[int, int, int, int]]) -> bytes:
    body = struct.pack("<III8s", len(colours), 0, len(colours) - 1, b"\0" * 8)
    for red, green, blue, alpha in colours:
        body += struct.pack("<HBBBB", 0, red, green, blue, alpha)
    return _chunk(0x2019, body)


def test_per_frame_palette_snapshots_are_bounded_by_a_total_budget(monkeypatch):
    """inker-codecs-03: any palette chunk at all marks its frame "touched"
    (``_read_palette``'s own comment says why: even a restated, unchanged
    table has to snapshot, or the forward fill loses the base one), and
    ``_read_frame`` stores one *whole*-table snapshot per touched frame with
    no running total across them -- so a 9.7 KB file naming 200 frames each
    with a full ``_MAX_PALETTE_ENTRIES`` (65536) table retained 217 MB before
    this fix.

    ``pixelguard.MAX_DECODE_PIXELS`` -- the shared running-total ceiling this
    reader's ``tileset_pixels``/``cel_pixels`` totals already use -- is
    lowered here so a handful of small, individually-legal palette chunks
    trip the total without the real fixture's size.
    """
    from realmspinner.core.safeio import pixelguard

    monkeypatch.setattr(pixelguard, "MAX_DECODE_PIXELS", 10)
    six = [(i, i, i, 255) for i in range(6)]
    frames = [
        _ase_frame([_ase_layer_named("Art"), _ase_cel(0, bytes(16), 2, 2), _ase_palette(six)])
    ]
    frames += [_ase_frame([_ase_cel(0, bytes(16), 2, 2), _ase_palette(six)])]
    data = _ase_file(_ase_header(len(frames), 2, 2), frames)
    with pytest.raises(ValueError, match="colours"):
        asein.document_from_aseprite(data)


def test_an_animated_aseprite_keeps_background_and_reference_flags_on_its_tracks():
    """inker-codecs-04: a ``Track``'s ``background``/``reference`` are
    authoritative over the materialised ``Layer``'s own -- ``Track``'s own
    docstring says so, since ``Document._materialize_frame`` copies them down
    every time a frame is shown -- but the animated branch of
    ``open_aseprite`` built every ``Track`` without passing either flag
    through, so the still branch (built from ``_build_cels``'s ``Layer``
    objects directly) kept both while the animated one silently lost them the
    moment any frame materialised, contradicting ``docs/COMPAT.md``.
    """
    art = bytes(16)  # 2x2 RGBA, all zero -- content is irrelevant

    def sprite() -> bytes:
        return _ase_file(
            _ase_header(2, 2, 2),
            [
                _ase_frame(
                    [
                        _ase_layer_named("BG", background=True),
                        _ase_cel(0, art, 2, 2),
                        _ase_layer_named("Trace", reference=True),
                        _ase_cel(1, art, 2, 2),
                    ]
                ),
                _ase_frame([_ase_cel(0, art, 2, 2), _ase_cel(1, art, 2, 2)]),
            ],
        )

    doc, _warnings = asein.document_from_aseprite(sprite())
    assert doc.anim is not None
    bg, trace = doc.anim.tracks
    assert bg.background is True
    assert trace.reference is True


def test_marking_a_layer_reference_or_background_on_a_copied_cel_survives_a_frame_change():
    """inker-document-01: ``_set_layer_flags``/``_flag_edit`` wrote the track
    side of a ``background``/``reference`` edit only when ``track.uid``
    matched the *materialised layer's* uid -- true only for the frame
    ``Track.of`` first built the track from. ``add_frame(copy=True)`` (and a
    paint stroke that autovivifies a cel) gives every later frame's cel a
    fresh uid, so marking a layer on one of those wrote the flag onto that one
    materialised ``Layer`` and never onto the track -- invisible until the
    frame was re-materialised (a frame change and back), which copied the
    track's untouched, still-``False`` flag back down over it.
    """
    doc = inker.Document.blank(8, 8)
    doc.ensure_animation()
    doc.add_frame(copy=True)  # frame 1's cel is a copy: its own uid, not the track's
    doc.set_current_frame(1)
    track_uid = doc.anim.tracks[0].uid
    cel_uid = doc.stack[0].uid
    assert track_uid != cel_uid, "the fixture, not the claim -- see add_frame(copy=True)"

    assert doc.to_background() is True
    assert doc.stack[0].background is True
    assert doc.anim.tracks[0].background is True

    # A frame change and back: what re-materialises frame 1's stack entry.
    # ``layers_for`` copies every ``CEL_PROPS`` entry down from the track onto
    # whichever ``Layer`` object lands here -- if the track never got the
    # flag, this is exactly where it reverted to ``False`` before the fix.
    doc.set_current_frame(0)
    doc.set_current_frame(1)
    assert doc.stack[0].background is True
    assert doc.anim.tracks[0].background is True

    # ``set_reference`` shares the same two helpers and the same shape of
    # fix: frame 1's cel is still ``add_frame(copy=True)``'s copy.
    assert doc.from_background() is True
    assert doc.stack[0].background is False
    assert doc.anim.tracks[0].background is False
    assert doc.set_reference(0, True) is True
    assert doc.stack[0].reference is True
    assert doc.anim.tracks[0].reference is True
    doc.set_current_frame(0)
    doc.set_current_frame(1)
    assert doc.stack[0].reference is True
    assert doc.anim.tracks[0].reference is True


def test_duplicating_a_background_layer_makes_an_ordinary_layer():
    """inker-document-02: ``duplicate_layer`` (both branches) and
    ``LayerStack.duplicate`` carried ``background`` onto the copy, which
    lands one row *above* its source -- never row 0, the only row the format
    and every writer let carry the flag -- so duplicating the bottom layer
    left two rows both reading ``background`` (``[True, True]``), the copy
    stuck there since ``LayerStack.move`` refuses a background away from row
    0 and ``from_background`` only ever acts on ``stack[0]``.
    """
    still = inker.Document.blank(8, 8)
    still.to_background()
    still.duplicate_layer(0)
    assert [bool(getattr(layer, "background", False)) for layer in still.stack] == [
        True,
        False,
    ]

    animated = inker.Document.blank(8, 8)
    animated.to_background()
    animated.ensure_animation()
    animated.duplicate_layer(0)
    assert [bool(track.background) for track in animated.anim.tracks] == [True, False]
    assert [bool(getattr(layer, "background", False)) for layer in animated.stack] == [
        True,
        False,
    ]


def test_pasting_a_tilemap_cel_clip_into_a_document_without_its_tileset_is_refused():
    """inker-document-04: ``paste_cels`` adopts a clip's planes with a plain
    ``.copy()`` and no check that a ``TilemapCel`` among them names a tileset
    the *target* document holds -- ``tileset_slot``'s own docstring calls a
    missing uid a bug upstream and raises ``KeyError`` rather than refusing
    by name, and every later stroke, flip or rotate on the pasted cel goes
    through it. ``ops.run`` does not catch a bare ``KeyError``, so pasting
    across tabs without also copying the tileset crashed on the next edit
    instead of refusing at paste time.
    """
    import numpy as np

    from realmspinner.kernels.pixel.tiles import TilemapCel, strip

    def _tile(colour: tuple[int, int, int, int]) -> np.ndarray:
        tile = np.zeros((4, 4, 4), dtype=np.uint8)
        tile[..., 0], tile[..., 1], tile[..., 2], tile[..., 3] = colour
        return tile

    def _tileset(colour: tuple[int, int, int, int]):
        blank = np.zeros((4, 4, 4), dtype=np.uint8)
        return strip(np.stack([blank, _tile(colour)], axis=0))

    doc = inker.Document.blank(8, 8)
    doc.ensure_animation()
    slot = doc.add_tileset(_tileset((255, 0, 0, 255)))
    layer = doc.add_tilemap_layer(slot.uid)  # a new track, above "Background"
    tile_track = doc.stack.active_index
    patch = np.array([[1]], dtype=np.uint32)
    assert doc.place_tiles(layer.uid, (0, 0), patch) is True
    doc.add_frame()
    doc.history.clear()

    clip = doc.copy_cels(tile_track, tile_track, 0, 1)
    assert clip is not None
    assert any(isinstance(plane, TilemapCel) for plane in clip.planes)

    other = inker.Document.blank(8, 8)  # no tileset at all
    other.ensure_animation()
    other.add_frame()
    other.history.clear()

    before_cels = dict(other.anim.cels)
    assert other.paste_cels(clip, 0, 0) is False
    assert len(other.history) == 0
    assert other.anim.cels == before_cels  # nothing adopted
