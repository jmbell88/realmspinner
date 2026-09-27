"""Regression tests for the 2026-09-26 audit's w1f7 fixer batch, for the
findings whose owned files live under ``studio/modes/inker``:

inker-panes-01 (the eye-column drag gated on ``is_item_hovered``, so it only
ever painted row 0), inker-mode-04 (Ctrl+S flattening a PNG source that has
since grown layers or frames), inker-mode-05 (closing a walk session's tab
leaving ``state.walk`` set), inker-mode-10 (slice filenames colliding
case-insensitively or landing on a Windows device name) and inker-mode-13
(an export magnifying by ``export_scale`` with no ceiling before allocating).

inker-codecs-07 lives in ``tests/kernels/pixel/test_audit_2026_09_26_w1f7.py``,
whose owned file (``gpl.py``) is under ``kernels/pixel``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _ui_context import imgui_context
from PIL import Image

from realmspinner.kernels import pixel as inker
from realmspinner.service.errors import TooLarge
from realmspinner.studio import probe
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker import walk as inker_walk
from realmspinner.studio.modes.inker.state import InkerDoc, InkerState
from realmspinner.studio.modes.inker.ui.panes import timeline as inker_timeline

# -- shared ------------------------------------------------------------------


class _Ctx:
    """A ctx that records toasts and runs a submitted task inline, ``_SaveCtx``'s
    shape from ``test_inker_save.py`` -- not imported from there because that
    module is not one of this batch's owned files."""

    def __init__(self, state: InkerState) -> None:
        self.state = SimpleNamespace(inker=state)
        self.toasts: list[tuple[str, str]] = []
        self.submitted: list[str] = []
        self.run: Any = None

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        self.run = run
        self.result = run(*args)
        return True

    def toast(self, text: str, level: str = "info", *_: Any) -> None:
        self.toasts.append((text, level))


def _open(doc=None):
    tab = InkerDoc(doc=doc or inker.Document.blank(16, 16), title="sprite.png")
    state = InkerState()
    state.add(tab)
    return _Ctx(state), state, tab


# -- inker-panes-01: the eye-column drag only ever painted row 0 -------------


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _frame(imgui, ctx, tab, *, pos, down):
    io = imgui.get_io()
    io.add_mouse_pos_event(pos[0], pos[1])
    io.add_mouse_button_event(0, down)
    probe.begin_frame()
    imgui.new_frame()
    imgui.set_next_window_size((520.0, 400.0))
    imgui.set_next_window_pos((0.0, 0.0))
    imgui.begin("##host")
    inker_timeline._grid(ctx, tab)
    imgui.end()
    imgui.end_frame()
    return list(probe.FRAME_CONTROLS)


def _eye_buttons(controls):
    from realmspinner.studio import icons

    return [
        c
        for c in controls
        if c.kind == "icon_button"
        and c.label.startswith(icons.EYE)
        and c.label.endswith("##visible")
    ]


def test_an_eye_drag_over_a_second_row_sets_that_rows_visibility(ui):
    """inker-panes-01 (2026-09-26 audit): ``_drag_toggle`` gated the paint on
    ``imgui.is_item_hovered()``, which imgui answers False for every row but
    the one whose button is the *active* item while the mouse stays down --
    the same trap ``cell_index``'s own docstring names for the range marquee.
    A drag starting on row 0 therefore never reached row 1. The fix hit-tests
    the row's own item rect against the mouse position instead."""

    from realmspinner.studio.state import AppState

    doc = inker.Document.blank(16, 16)
    doc.add_layer()
    app_state = AppState()
    app_state.inker = InkerState()
    tab = InkerDoc(doc=doc, uid="t1", title="Untitled")
    app_state.inker.add(tab)
    ctx = SimpleNamespace(state=app_state, viewer=None)

    controls = _frame(ui, ctx, tab, pos=(0.0, 0.0), down=False)
    eyes = _eye_buttons(controls)
    assert len(eyes) == 2, [c.label for c in controls]
    row0, row1 = eyes[0], eyes[1]

    x0, y0, w0, h0 = row0.rect
    press = (x0 + w0 / 2, y0 + h0 / 2)
    _frame(ui, ctx, tab, pos=press, down=True)  # press row 0's eye: arms the drag

    x1, y1, w1, h1 = row1.rect
    drag = (x1 + w1 / 2, y1 + h1 / 2)
    _frame(ui, ctx, tab, pos=drag, down=True)  # the press continues onto row 1

    assert doc.stack[1].visible is False, "the drag never reached row 1"

    _frame(ui, ctx, tab, pos=drag, down=False)  # release
    assert app_state.inker.eye_drag is None
    assert doc.history.can_undo
    doc.undo()
    assert doc.stack[1].visible is True


# -- inker-mode-04: Ctrl+S on a grown flat-PNG source ------------------------


def test_ctrl_s_on_a_layered_png_source_does_not_flatten_the_original(tmp_path):
    """A flat PNG opened here starts as one layer and no animation.
    ``png_bytes()`` is ``flatten()`` -- the *current* frame's composite alone
    -- so a user who grew that document into more than one layer and pressed
    Ctrl+S got the write ``WRITABLE_SUFFIXES`` exists to police for JPG/WebP/
    BMP/GIF: bytes over the source that "keep what I have" promised not to
    touch. Confirmed against the unfixed ``save()`` (git history): it
    submitted ``inker-save:...`` unconditionally and the source's pixels
    changed. The fix routes this case to Save As instead, leaving the
    original untouched."""

    src = tmp_path / "sprite.png"
    Image.new("RGBA", (8, 8), (200, 30, 30, 255)).save(src, "PNG")
    before = src.read_bytes()

    doc = inker.Document.load(src)
    tab = InkerDoc(doc=doc, title=src.name, path=src, file_format=doc.file_format)
    state = InkerState()
    state.add(tab)
    doc.add_layer()
    weight = np.ones((4, 4), dtype=np.float32)
    doc.write_colour((0, 0, 4, 4), (0, 255, 0, 255), weight)
    ctx = _Ctx(state)

    dest = tmp_path / "sprite.ora"
    from realmspinner.studio import dialogs

    orig_save_file = dialogs.save_file
    try:
        dialogs.save_file = lambda *a, **k: dest
        inker_mode.save(ctx, tab)
    finally:
        dialogs.save_file = orig_save_file

    assert any(key.startswith("inker-saveas:") for key in ctx.submitted)
    assert not any(key.startswith("inker-save:") for key in ctx.submitted)
    assert src.read_bytes() == before, "the flat PNG was overwritten with a flatten"
    assert dest.exists()
    assert ctx.toasts and "layer or frame" in ctx.toasts[0][0]


def test_ctrl_s_on_an_unchanged_flat_png_still_saves_in_place(tmp_path):
    """The routing is scoped to a document that has actually grown beyond one
    layer and no animation -- an ordinary flat PNG still saves the way the
    manual promises."""

    src = tmp_path / "sprite.png"
    Image.new("RGBA", (8, 8), (200, 30, 30, 255)).save(src, "PNG")
    doc = inker.Document.load(src)
    tab = InkerDoc(doc=doc, title=src.name, path=src, file_format=doc.file_format)
    state = InkerState()
    state.add(tab)
    ctx = _Ctx(state)

    inker_mode.save(ctx, tab)

    assert ctx.submitted == [f"inker-save:{tab.uid}"]


# -- inker-mode-05: closing a walk session's tab leaves state.walk set -------


def test_closing_the_tab_that_owns_a_walk_session_ends_the_session():
    """``_closed`` settled a transform, cleared drag and forgot held keys, but
    never checked whether the tab it just lost was ``state.walk``'s owner --
    so ``open_reason`` stayed pinned at ``ALREADY_OPEN`` (state.py) for the
    rest of the app run, on every other tab, with no session left for Cancel
    to throw away."""

    state = InkerState()
    owner = InkerDoc(doc=inker.Document.blank(16, 16), title="a")
    other = InkerDoc(doc=inker.Document.blank(16, 16), title="b")
    state.add(owner)
    state.add(other)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state), viewer=None)

    assert inker_walk.open_session(ctx, owner)
    assert state.walk is not None

    state.close(owner.uid)

    assert state.walk is None
    assert inker_walk.can_open(state, other)


def test_closing_an_unrelated_tab_leaves_a_walk_session_open():
    """The fix is keyed on the closed tab actually owning the session --
    closing some other tab must not throw away a session in progress
    elsewhere."""

    state = InkerState()
    owner = InkerDoc(doc=inker.Document.blank(16, 16), title="a")
    other = InkerDoc(doc=inker.Document.blank(16, 16), title="b")
    state.add(owner)
    state.add(other)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state), viewer=None)

    assert inker_walk.open_session(ctx, owner)
    state.close(other.uid)

    assert state.walk is not None
    assert inker_walk.session(state, owner) is not None


# -- inker-mode-10: slice filenames collide case-insensitively or land on a
#    Windows device name -----------------------------------------------------


def test_slice_filenames_are_distinct_case_insensitively_and_avoid_device_names():
    """``_slice_filenames`` sanitised but never folded case for its own
    collision set, so "Hitbox" and "hitbox" -- two different names to Python's
    ``set`` -- both survived as themselves and collided the moment NTFS (or
    APFS) compared them; and it never consulted ``sheetout.reserved_check``,
    so a slice literally named "NUL" sanitised straight through to
    ``NUL.png``, a file Windows refuses to create."""

    entries = [SimpleNamespace(name=n) for n in ("Hitbox", "hitbox", "NUL", "nul")]
    names = inker_mode._slice_filenames(entries)

    # Case-insensitively distinct: no two names may be the same filename on
    # a real (case-insensitive) filesystem.
    folded = [n.casefold() for n in names]
    assert len(set(folded)) == len(folded), names
    # No device name literally handed to a writer, whatever case it arrived in.
    assert not any(n.upper() in ("NUL",) for n in names), names


def test_a_single_device_named_slice_still_gets_a_usable_filename():
    entries = [SimpleNamespace(name="NUL")]
    names = inker_mode._slice_filenames(entries)
    assert names == ["NUL_2"]


# -- inker-mode-13: an export magnifying by scale with no ceiling -----------


def test_export_refuses_a_scale_that_would_exceed_the_pixel_ceiling_before_allocating(
    tmp_path,
):
    """Every export fed ``export_scale`` straight to ``transform.upscale``
    (``np.repeat``) with no ceiling of any kind: a 4096x4096 canvas at 8x is a
    ~4 GiB allocation, and a hand-edited sidecar's ``"scale": 100000`` is
    unbounded. Confirmed against the unfixed ``export_png`` (git history):
    given a 1x1 canvas and ``export_scale = 10000`` it happily allocated and
    wrote a 10000x10000 PNG with no refusal at all. The fix refuses before
    ``doc.png_bytes`` ever runs."""

    doc = inker.Document.blank(1, 1)
    ctx, state, tab = _open(doc)
    state.export_scale = 10_000

    dest = tmp_path / "sprite.png"
    from realmspinner.studio import dialogs

    orig_save_file = dialogs.save_file
    try:
        dialogs.save_file = lambda *a, **k: dest
        with pytest.raises(TooLarge) as excinfo:
            inker_mode.export_png(ctx, tab)
    finally:
        dialogs.save_file = orig_save_file

    assert excinfo.value.field == "export_scale"
    assert not dest.exists()


def test_a_reasonable_export_scale_is_not_refused(tmp_path):
    doc = inker.Document.blank(16, 16)
    ctx, state, tab = _open(doc)
    state.export_scale = 4

    dest = tmp_path / "sprite.png"
    from realmspinner.studio import dialogs

    orig_save_file = dialogs.save_file
    try:
        dialogs.save_file = lambda *a, **k: dest
        inker_mode.export_png(ctx, tab)
        ctx.run()
    finally:
        dialogs.save_file = orig_save_file

    assert dest.exists()


def test_a_hand_edited_scale_is_clamped_when_export_options_are_restored():
    """``_safe_int`` had a floor and no ceiling, so ``apply_export_options``
    put a hand-edited sidecar's ``"scale": 100000`` straight onto a live
    tab's shared controls, unbounded, between there and whichever export
    finally rejected it."""

    from realmspinner.studio.modes.inker.state import MAX_EXPORT_SCALE

    state = InkerState()
    state.apply_export_options({"scale": 100_000})
    assert state.export_scale == MAX_EXPORT_SCALE
