"""Sirens' right-top pane: the instrument list, and which one notes are stamped
with.

**A list, not an editor.** The envelope curves an instrument is made of --
volume, arpeggio, pitch and duty, each a sequence with a loop point and a
release point -- are ``sirens_envelopes``, the pane directly under this one,
because a sequence editor is a canvas rather than a row of fields and does not
belong squeezed into the list it edits.

What is here is what the grid needs to be usable at all: the instruments a
document already has (``new_song`` builds one per channel kind), a way to add
and remove them, and the selection that ``write_note`` stamps into the
instrument column. Without that last one a typed note plays *nothing*, which is
the single most confusing thing a tracker can do to a newcomer. The selection is
also what the envelope editor draws, so the two panes share one answer to "which
instrument" rather than each keeping its own.

**A row is numbered by its id, not by its position in the list.** That id is
what the grid draws in a cell's instrument column and what a user types there,
and it is a per-document slot bounded by ``document.MAX_INSTRUMENTS`` precisely
so it can be both. Numbering the rows by position instead would agree with the
grid only until an instrument was removed.
"""

from __future__ import annotations

from typing import Any

from .. import anchors, controls, icons, sirens_mode, tokens, widgets
from ..manual import render as manual_render
from ..sirens import document as D
from ..sirens import instruments as inst
from ..tokens import sp

_BUSY_WHY = "This song is being written; the buttons come back when it lands."


def instrument_room(doc: Any, editable: bool) -> tuple[bool, str]:
    """Whether "Add" can fire, and why not if not.

    Pulled out as a pure function so the 2026-09-07 audit's finding sirens-03
    has something a test can call with no imgui frame: the button used to call
    ``add_instrument`` with no cap check and no try/except, so filling a song
    to its own documented ``MAX_INSTRUMENTS`` ceiling raised a bare
    ``ValueError`` out of ``draw()`` -- and ``guard.py`` replaces the whole
    pane after three of those in a row.
    """
    if not editable:
        return False, _BUSY_WHY
    if len(doc.instruments) >= D.MAX_INSTRUMENTS:
        return False, f"A song holds {D.MAX_INSTRUMENTS} instruments."
    return True, ""


def delete_reason(editable: bool, selected: bool) -> str:
    """Why the instrument Delete button is disabled, or "" while it is live.

    Pulled out as a pure function so the 2026-09-07 audit's finding sirens-04
    has something a test can call with no imgui frame: the inline ternary this
    replaced tested ``editable`` inverted, so a busy song showed "No
    instrument is selected" and an idle one with nothing picked showed the
    busy sentence -- each state naming the other's reason.
    """
    if not editable:
        return _BUSY_WHY
    if not selected:
        return "No instrument is selected."
    return ""


def sample_delete_reason(editable: bool, held: bool) -> str:
    """Why the sample Delete button is disabled, or "" while it is live.

    Pulled out as a pure function so the 2026-09-07 audit's finding sirens-05
    has something a test can call with no imgui frame: the inline ternary this
    replaced fell through to the empty string while the song was saving,
    which is the exact silent failure ``disabled_button``'s ``reason``
    parameter exists to prevent, at the moment it most needs explaining.
    """
    if not editable:
        return _BUSY_WHY
    if not held:
        return "This instrument has no sample."
    return ""


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    anchors.mark_window("sirens/instruments")
    state = sirens_mode.ensure(ctx)
    tab = state.active
    widgets.section("Instruments")
    manual_render.help_button(ctx, "sirens-instruments")

    if tab is None:
        return

    doc = tab.doc
    editable = not tab.busy

    width = widgets.grid_width(2)
    addable, add_why = instrument_room(doc, editable)
    if widgets.disabled_button(f"{icons.PLUS} Add", addable, (width, 0), reason=add_why):
        instrument = doc.add_instrument()
        state.instrument = instrument.uid
        sirens_mode.request_rerender(ctx, tab)
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.TRASH} Delete",
        editable and state.instrument is not None,
        (width, 0),
        reason=delete_reason(editable, state.instrument is not None),
    ):
        if doc.remove_instrument(state.instrument):
            sirens_mode.request_rerender(ctx, tab)
        sirens_mode.clamp_caret(ctx, tab)

    if not doc.instruments:
        widgets.muted_wrapped(
            "No instruments: a note typed now would be silent. Add one."
        )
        return

    imgui.dummy((0, sp(tokens.SP_1)))
    for instrument in list(doc.instruments):
        label = instrument.name or f"Instrument {instrument.uid:02X}"
        if controls.selectable(
            f"{instrument.uid:02X}  {label}  [{instrument.kind}]###sirens-inst-{instrument.uid}",
            state.instrument == instrument.uid,
        )[0]:
            state.instrument = instrument.uid

    selected = None if state.instrument is None else doc.instrument(state.instrument)
    if selected is None:
        return
    imgui.dummy((0, sp(tokens.SP_2)))
    # ``##``-hidden with the name drawn above, the same rule the Kind combo a
    # few lines down already follows: imgui draws a field's label to its
    # *right*, and this field is set to width -1, so a visible "Name" here
    # would land past the content region and simply not be drawn.
    widgets.field_label("Name")
    imgui.set_next_item_width(-1)
    name = widgets.input_text(
        "##sirens-inst-name", selected.name, max_length=inst.MAX_NAME_LEN, commit=True
    )
    if name != selected.name and doc.update_instrument(selected.uid, name=name):
        sirens_mode.request_rerender(ctx, tab)
    # ``controls.combo`` takes (key, label) pairs and answers with a key, so a
    # fifth voice kind added to the engine reaches this list without an index
    # here silently becoming a different instrument.
    widgets.field_label("Kind")
    changed, kind = controls.combo(
        # ``##``-hidden, with the name drawn above: imgui puts a combo's label
        # to its *right*, so a named one at the default width writes past the
        # content region and the name is simply not drawn -- the rule
        # ``widgets.combo`` states, and which a scan test holds every pane to.
        "##sirens-inst-kind",
        selected.kind,
        [(name_, name_.title()) for name_ in inst.KINDS],
        enabled=editable,
    )
    if changed and doc.update_instrument(selected.uid, kind=kind):
        sirens_mode.request_rerender(ctx, tab)
    if selected.kind == "sample":
        _sample(ctx, tab, selected)


def _sample(ctx: Any, tab: Any, selected: Any) -> None:
    """Which recording a ``sample`` instrument plays, and how one gets here.

    Only for ``kind == "sample"``: the other three voices synthesise their
    waveform and a sample field beside them would be a control that does
    nothing, which is the failure ``Fxx``-as-speed was rejected for.

    **Import goes through ``sirens_io``**, the same door the window's ``.wav``
    drop uses, so the picker and the drop cannot advertise different formats or
    decode a file two different ways.
    """
    from imgui_bundle import imgui

    doc = tab.doc
    editable = not tab.busy
    keys = sorted(doc.samples)
    widgets.field_label("Sample")
    if keys:
        changed, key = controls.combo(
            "##sirens-inst-sample",
            selected.sample if selected.sample in keys else "",
            [("", "-- none --"), *((one, one) for one in keys)],
            enabled=editable,
        )
        if changed and doc.update_instrument(selected.uid, sample=key):
            sirens_mode.request_rerender(ctx, tab)
    else:
        widgets.muted_wrapped(
            "This song has no samples yet. Import a .wav, or drop one on the"
            " window."
        )

    width = widgets.grid_width(2)
    if widgets.disabled_button(
        f"{icons.UPLOAD} Import...",
        editable,
        (width, 0),
        reason=_BUSY_WHY,
    ):
        sirens_mode.ask_sample(ctx, tab, selected.uid)
    imgui.same_line()
    held = selected.sample if selected.sample in doc.samples else ""
    if widgets.disabled_button(
        f"{icons.TRASH} Delete",
        editable and bool(held),
        (width, 0),
        reason=sample_delete_reason(editable, bool(held)),
    ):
        sirens_mode.remove_sample(ctx, tab, held)

    if selected.sample and selected.sample not in doc.samples:
        # Named rather than blanked: ``remove_sample`` leaves the instruments
        # that pointed at a key alone, so this is the one place that says why
        # an instrument which used to make a sound now does not.
        widgets.muted_wrapped(
            f"This instrument names the sample {selected.sample}, which this"
            " song no longer holds, so it is silent."
        )
