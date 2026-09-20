"""Widget-reservation contracts: ``same_line_or_wrap``/``button_width`` used
correctly by their call sites.

Headless -- no imgui context. ``widgets.same_line_or_wrap`` and
``widgets.button_width`` are stubbed so what is asserted is the *sequence* of
reservation calls a toolbar makes, not anything imgui itself computes: the
defect this file is about is a call site reserving room for a label that is
never drawn, which is wrong before a single pixel is measured.
"""

from __future__ import annotations

from types import SimpleNamespace


def test_the_undo_and_redo_buttons_grey_while_the_document_is_saving(monkeypatch):
    """The 2026-09-18 audit (second run, finding shell-06), seen from both
    packwright/mode and sirens/edit: ``history_block`` gated Undo, Redo and
    the history popover button on ``can_undo``/``can_redo`` alone, so a click
    could mutate history under an in-flight save while
    ``docmodes.blocked_while_writing`` already refused the same edit's
    keyboard chord (Ctrl+Z/Y). One door -- ``history_block`` -- so every
    bridge pane that draws this pair inherits the guard."""
    from realmspinner.studio import widgets

    calls: list[tuple[str, bool, str]] = []

    def fake_disabled_button(label, enabled, size=(0, 0), *, reason="", tooltip=""):
        calls.append((label, enabled, reason))
        return False

    monkeypatch.setattr(widgets, "disabled_button", fake_disabled_button)
    monkeypatch.setattr(widgets.imgui, "same_line", lambda *a, **k: None)
    monkeypatch.setattr(widgets, "muted", lambda *a, **k: None)
    monkeypatch.setattr(widgets, "grid_width", lambda *a, **k: 100.0)

    class _History:
        can_undo = True
        can_redo = True

        def __len__(self):
            return 3

    tab = SimpleNamespace(doc=SimpleNamespace(history=_History()), busy=True)

    widgets.history_block(
        ctx=None,
        tab=tab,
        key="test",
        undo=lambda: None,
        redo=lambda: None,
    )

    by_label = {label: (enabled, reason) for label, enabled, reason in calls}
    undo_label = next(label for label in by_label if "Undo" in label)
    redo_label = next(label for label in by_label if "Redo" in label)
    assert by_label[undo_label] == (False, widgets.DOCUMENT_SAVING_WHY)
    assert by_label[redo_label] == (False, widgets.DOCUMENT_SAVING_WHY)


def test_the_frame_button_reservation_matches_its_own_width_not_the_tiled_toggle(
    monkeypatch,
):
    """Shell-08, the 2026-09-07 audit.

    ``offers_inker`` is looser than ``shows_tiled`` -- it only asks whether
    this is the 2D reference toolbar, not whether the job is a tile -- so the
    reservation for the Tiled toggle used to fire unconditionally once Open in
    Inker was drawn, even when the toggle itself never draws. The very next
    control in that case is the Frame icon button, which inherited a
    reservation sized for "Tiled 2x2" -- a label wider than the icon it never
    stands in for -- and wrapped the row onto a second line when the icon
    alone would have fit on the first.
    """
    from realmspinner.studio import icons
    from realmspinner.studio.panes import overlay

    events: list[tuple[str, str]] = []

    monkeypatch.setattr(
        overlay.widgets, "same_line_or_wrap", lambda width: events.append(("wrap", width))
    )
    monkeypatch.setattr(overlay.widgets, "button_width", lambda label: label)

    def fake_button(label, *_a, **_k):
        events.append(("button", label))
        return False

    def fake_icon_button(label, *_a, **_k):
        events.append(("icon_button", label))
        return False

    def fake_toggle(label, value, **_k):
        events.append(("toggle", label))
        return False, value

    monkeypatch.setattr(overlay.controls, "button", fake_button)
    monkeypatch.setattr(overlay.widgets, "icon_button", fake_icon_button)
    monkeypatch.setattr(overlay.widgets, "toggle", fake_toggle)
    monkeypatch.setattr(overlay.manual_render, "help_button_inline", lambda *a, **k: None)
    monkeypatch.setattr(overlay, "_has_content", lambda ctx, viewer: False)
    monkeypatch.setattr(overlay, "_texture_losses", lambda viewer: None)
    # The 2D reference toolbar, for a job that is not a tile: exactly the case
    # the finding reproduces -- Open in Inker offered, the Tiled toggle not.
    monkeypatch.setattr(overlay, "offers_inker", lambda ctx, job: True)
    monkeypatch.setattr(overlay, "shows_tiled", lambda ctx, job: False)

    ctx = SimpleNamespace(
        state=SimpleNamespace(
            create=SimpleNamespace(tile_preview=False),
            wireframe=False,
            turntable=False,
            comparing=None,
        ),
        viewer=SimpleNamespace(has_model=False),
        job=lambda: {},
        clear_viewport=None,
    )

    overlay.toolbar(ctx)

    frame_index = events.index(("icon_button", icons.MAXIMIZE))
    assert events[frame_index - 1] == ("wrap", icons.MAXIMIZE), (
        "the Frame button's reservation is sized for the Tiled toggle instead "
        f"of its own icon: {events}"
    )
