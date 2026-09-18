"""Regression tests for the 2026-09-14 audit's Inker findings: inker-07,
inker-12, inker-13, inker-14.

Each test's name is the claim; each failed against the code as the audit
found it before the fix beside it landed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from warlock.studio.modes.inker import opening as inker_open
from warlock.studio.modes.inker import palette_io as inker_palette_io
from warlock.studio.modes.inker import sheet as inker_sheet
from warlock.studio.modes.inker.state import InkerState
from warlock.studio.modes.inker.ui.panes import colors as inker_colors
from warlock.studio.modes.inker.ui.panes import tools as inker_tools
from warlock.studio.state import AppState

# --- a shared door harness ---------------------------------------------------
#
# ``inker_mode.ensure`` short-circuits to whatever is already sitting on
# ``ctx.state.inker`` (see its own body), so pre-seeding that field is enough
# to skip the settings machinery entirely -- none of the doors under test read
# ``ctx.settings``.


class _Ctx:
    """``submit`` that always refuses, the way a duplicate in-flight key does.

    A real ``TaskRunner.submit`` returns ``False`` without ever calling *fn*
    when the key is already claimed -- a second press of the same button
    while its picker is still up. Never invoking *fn* here mirrors that: no
    test in this file opens a real file dialog.
    """

    def __init__(self, state: InkerState) -> None:
        self.state = AppState()
        self.state.inker = state
        self.settings = None  # unused: ``ensure`` short-circuits on ``state``
        self.svc = object()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []

    def submit(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> bool:
        self.submitted.append(key)
        return False

    def toast(self, text: str, level: str = "info", **_kw: Any) -> None:
        self.toasts.append((text, level))


class _Tab:
    """The little a door needs of an open tab: a uid, a document, a path."""

    def __init__(self, palette: list[tuple[int, int, int, int]] | None = None) -> None:
        self.uid = "t1"
        self.busy = False
        self.path: Path | None = None
        self.doc = _Doc(palette)


class _Doc:
    def __init__(self, palette: list[tuple[int, int, int, int]] | None) -> None:
        self.palette = palette or []


def _state_with_tab() -> InkerState:
    state = InkerState()
    state.docs.append(_Tab(palette=[(0, 0, 0, 255), (255, 255, 255, 255)]))
    return state


# --- inker-12: a refused ``ctx.submit`` used to be silent --------------------


def test_pressing_open_twice_while_the_picker_is_up_says_something():
    """Every door in ``inker_open`` and ``inker_palette_io`` that submits a
    task discarded ``ctx.submit``'s ``False`` -- "a task with this key is
    already running" -- so a second press while the first was still in flight
    did nothing a user could see (the 2026-09-14 audit, inker-12; widened on
    2026-09-14 to ``import_aseprite_path``, ``open_sprite_draft``,
    ``open_rendered_sheet`` and ``open_pixel_artifact``, the four doors this
    finding's stated line range had not reached).

    Before the fix every one of these left ``ctx.toasts`` empty.
    """
    doors: list[tuple[str, Any]] = [
        ("inker-open", lambda ctx: inker_open.ask_open(ctx)),
        (
            f"inker-open:{abs(hash(str(Path('a.png'))))}",
            lambda ctx: inker_open.open_path(ctx, Path("a.png")),
        ),
        (
            "inker-open:pixels:Untitled",
            lambda ctx: inker_open.open_pixels(
                ctx, np.zeros((2, 2, 4), dtype=np.uint8)
            ),
        ),
        ("inker-sheetin", lambda ctx: inker_open.ask_import_sheet(ctx)),
        ("inker-open:aseprite", lambda ctx: inker_open.ask_import_aseprite(ctx)),
        (
            f"inker-open:aseprite:{abs(hash(str(Path('b.aseprite'))))}",
            lambda ctx: inker_open.import_aseprite_path(ctx, Path("b.aseprite")),
        ),
        (
            "inker-open:sprite:draft1:N",
            lambda ctx: inker_open.open_sprite_draft(ctx, "job1", "draft1", "N"),
        ),
        (
            "inker-open:sheet:sheet1:render",
            lambda ctx: inker_open.open_rendered_sheet(ctx, "job1", "sheet1"),
        ),
        (
            "inker-open:pixel:job1:pixel_001.png",
            lambda ctx: inker_open.open_pixel_artifact(
                ctx,
                "job1",
                "pixel_001.png",
                title="T",
                pixel_colors=16,
                pixel_palette=None,
                pixel_dither=False,
            ),
        ),
        ("inker-palette", lambda ctx: inker_palette_io.import_palette(ctx)),
        ("inker-palette-export", lambda ctx: inker_palette_io.export_palette(ctx)),
    ]
    for expected_key, press in doors:
        ctx = _Ctx(InkerState())
        press(ctx)
        assert ctx.submitted == [expected_key], expected_key
        assert ctx.toasts, f"{expected_key} refused a second press in silence"

    # The four palette doors that read the active tab rather than only the
    # session's swatch row.
    tab_doors: list[tuple[str, Any]] = [
        (
            "inker-index:t1",
            lambda ctx: inker_palette_io.import_document_palette(ctx),
        ),
        ("inker-palimg:t1", lambda ctx: inker_palette_io.palette_from_image(ctx)),
        (
            "inker-palette-export-doc",
            lambda ctx: inker_palette_io.export_document_palette(ctx),
        ),
        (
            "inker-palette-export-image",
            lambda ctx: inker_palette_io.export_palette_image(ctx),
        ),
    ]
    for expected_key, press in tab_doors:
        ctx = _Ctx(_state_with_tab())
        press(ctx)
        assert ctx.submitted == [expected_key], expected_key
        assert ctx.toasts, f"{expected_key} refused a second press in silence"


# --- inker-07: a refusal toast must not be followed by a no-op toast --------


class _MirrorAnim:
    current = 0


class _MirrorDoc:
    """Just enough of ``Document`` for ``inker_sheet.mirror_to``: an ``anim``
    with a ``current`` frame, and the one engine call under test."""

    def __init__(self, *, raises: bool = False, applied: bool = True) -> None:
        self.anim = _MirrorAnim()
        self._raises = raises
        self._applied = applied

    def mirror_to(self, track_uid: int, current: int, target: int, fraction: float) -> bool:
        if self._raises:
            raise ValueError("the face box is empty")
        return self._applied


class _MirrorTab:
    def __init__(self, doc: _MirrorDoc) -> None:
        self.doc = doc


class _ToastCtx:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, text: str, level: str = "info", **_kw: Any) -> None:
        self.toasts.append((text, level))


def test_mirror_to_does_not_follow_a_refusal_toast_with_an_already_matches_toast(
    monkeypatch: Any,
) -> None:
    """A raised refusal and a plain no-op used to look identical to
    ``mirror_to``, which is why a refused mirror was followed by "The mirror
    already matches outside the face." for the same press (the 2026-09-14
    audit, inker-07). Before the fix this saw two toasts, not one."""
    monkeypatch.setattr(inker_sheet, "active_track_uid", lambda tab: 1)
    monkeypatch.setattr(inker_sheet, "counterpart", lambda tab: 2)

    ctx = _ToastCtx()
    result = inker_sheet.mirror_to(ctx, _MirrorTab(_MirrorDoc(raises=True)))
    assert result is False
    assert len(ctx.toasts) == 1, ctx.toasts
    assert "was not applied" in ctx.toasts[0][0]

    # The plain no-op still gets its own, different toast.
    ctx2 = _ToastCtx()
    result2 = inker_sheet.mirror_to(
        ctx2, _MirrorTab(_MirrorDoc(raises=False, applied=False))
    )
    assert result2 is False
    assert len(ctx2.toasts) == 1, ctx2.toasts
    assert "already matches" in ctx2.toasts[0][0]


def test_mirror_run_does_not_follow_a_refusal_toast_with_an_already_matches_toast(
    monkeypatch: Any,
) -> None:
    """``mirror_run`` goes through the same ``_framed`` as ``mirror_to`` and
    carried the identical bug: a raised refusal was followed by "Every cell of
    that run already matches its mirror outside the face." for the same
    press. Found alongside ``mirror_to`` on 2026-09-14 and widened into
    inker-07 the same day."""
    monkeypatch.setattr(inker_sheet, "active_track_uid", lambda tab: 1)
    monkeypatch.setattr(inker_sheet, "run_of", lambda tab: (object(), 0))

    ctx = _ToastCtx()
    result = inker_sheet.mirror_run(ctx, _MirrorTab(_MirrorRunDoc(raises=True)))
    assert result is False
    assert len(ctx.toasts) == 1, ctx.toasts
    assert "was not applied" in ctx.toasts[0][0]

    # The plain no-op still gets its own, different toast.
    ctx2 = _ToastCtx()
    result2 = inker_sheet.mirror_run(
        ctx2, _MirrorTab(_MirrorRunDoc(raises=False, applied=False))
    )
    assert result2 is False
    assert len(ctx2.toasts) == 1, ctx2.toasts
    assert "already matches" in ctx2.toasts[0][0]


class _MirrorRunDoc:
    """``mirror_run``'s half of ``_MirrorDoc``: only the engine call differs."""

    def __init__(self, *, raises: bool = False, applied: bool = True) -> None:
        self._raises = raises
        self._applied = applied

    def mirror_run(self, track_uid: int, run: Any, fraction: float) -> bool:
        if self._raises:
            raise ValueError("the run's face box is empty")
        return self._applied


# --- inker-13: sorting or ramping an indexed palette is not free ------------


def test_sort_and_ramp_docstring_does_not_claim_indexed_sorts_are_free():
    """``_sort_and_ramp`` said "Neither pushes an undo step" for both the sort
    and the ramp; ``Document.sort_palette``'s own docstring says the opposite
    for an indexed document ("it moves index data and costs a Ctrl+Z, exactly
    as ``move_slot`` does"). The pane docstring was wrong (the 2026-09-14
    audit, inker-13)."""
    doc = inker_colors._sort_and_ramp.__doc__ or ""
    assert "neither pushes an undo step" not in doc.lower()
    assert "undo step" in doc.lower()


def test_insert_ramp_docstring_does_not_claim_an_indexed_ramp_is_free():
    """``insert_ramp``'s docstring said "adding a swatch repaints nothing, so
    there is no step to push" with no RGB-vs-indexed qualifier, but its own
    indexed branch pushes a ``CompoundEdit`` -- ``sort_palette``'s shape,
    inker-13's finding, widened to this sibling on 2026-09-14."""
    from warlock.kernels.pixel._doc_indexed import IndexedOps

    doc = (IndexedOps.insert_ramp.__doc__ or "").lower()
    assert "repaints nothing, so there is no step to push" not in doc
    assert "compoundedit" in doc


# --- inker-14: only Move has no per-tool options ----------------------------


def test_has_options_docstring_names_only_move_as_optionless():
    """``_has_options``' docstring said "The move and eyedropper tools have no
    options at all," but the function itself lists ``"eyedropper"`` beside
    fill, wand and text -- only ``"move"`` is missing from every branch (the
    2026-09-14 audit, inker-14). The code was already right; the comment
    lied."""
    assert inker_tools._has_options("move") is False
    assert inker_tools._has_options("eyedropper") is True

    doc = inker_tools._has_options.__doc__ or ""
    assert "move and eyedropper" not in doc.lower()
