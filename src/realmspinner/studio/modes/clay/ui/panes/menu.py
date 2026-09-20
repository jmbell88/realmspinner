"""The right-mouse context menu over the Clay viewport.

The Wings3D idea, and the reason the ops registry exists: the operations that
apply to what is selected, under the cursor, at the moment the user wants them
-- rather than in a toolbar the eye has to leave the model to find. Every row
comes from :mod:`~realmspinner.studio.modes.clay.ops`, so the menu cannot offer an op the
keyboard does not have or grey out one the tools pane would have run.

This is the only layer that knows imgui exists. The registry decides *what* is
invocable and *whether*; this decides where the popup opens and what a row looks
like, and nothing else.

**An op with parameters opens a second popup rather than running.** Bevel with
no width is a bevel of whatever the last one was, which is the sort of thing
that is right four times and destroys a model on the fifth; the popup follows
the raster editor's resize-dialog idiom -- fields plus Apply, with the last
values remembered on ``ClayState`` so the common case is two clicks.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ..... import controls, theme, tokens, widgets
from .....tokens import sp
from ... import mode as clay_mode
from ... import ops as clay_ops

POPUP = "clay-context"
PARAM_POPUP = "clay-op-params"

#: How wide an ``Op.hint`` is allowed to get before it wraps, in design pixels.
#: Chosen so the dialog stays narrower than the properties pane beside it -- a
#: popup that auto-sizes past its own panel reads as a window, not a prompt.
HINT_WRAP = 300


def draw(ctx: Any, view: Any) -> None:
    """Open and render the menu. Called from the viewport pane, after the image."""
    tab = clay_mode.active(ctx)
    if tab is None:
        return
    state = clay_mode.ensure(ctx)
    if view.menu_request is not None:
        view.menu_request = None
        imgui.open_popup(POPUP)

    if imgui.begin_popup(POPUP):
        widgets.popup_chrome(_imgui=imgui)
        _rows(ctx, state, tab, tab.doc)
        imgui.end_popup()
    params_popup(ctx, state, tab)


def _rows(ctx: Any, state: Any, tab: Any, doc: Any) -> None:
    widgets.secondary(f"{doc.element_mode} mode")
    controls.menu_separator()
    if tab.saving:
        # The gate every other control in the app has, and the one this menu
        # did not: ``enabled`` never consulted it, so every row stayed
        # clickable during a save and the click was then swallowed by the
        # ``or tab.saving`` below -- a live-looking menu that did nothing and
        # said nothing. Told once, at the top, rather than as fifteen greyed
        # rows with no reason attached.
        widgets.secondary("Saving...")
        controls.menu_separator()
    for op in clay_ops.menu(doc.element_mode):
        if op.separator_before:
            controls.menu_separator()
        enabled = op.enabled(doc) and not tab.saving
        # A saving document greys everything for one shared reason (the
        # "Saving..." row above); an op's own reason only applies once that
        # gate has already passed. clay-07 (2026-09-06 audit): this row used to
        # grey out with no reason at all, for every op that refuses -- Merge
        # Objects needing two visible objects, Bridge Loops needing edge mode,
        # every bevel/inset/weld needing an element selection.
        reason = "" if tab.saving else clay_ops.reason_for(op, doc)
        clicked, _ = controls.menu_item(op.label, op.key, False, enabled, reason=reason)
        if not clicked:
            continue
        if op.params:
            state.pending_op = op.name
            state.op_params.setdefault(op.name, clay_ops.defaults_for(op))
            imgui.close_current_popup()
            # Here the id stack *is* a window's, so this opens directly rather
            # than going through open_op_popup.
            imgui.open_popup(PARAM_POPUP)
        else:
            clay_ops.run(ctx, doc, op)


def params_popup(ctx: Any, state: Any, tab: Any) -> None:
    """The fields for a parameterised op, and its Apply button.

    Called from *both* the viewport (for a menu row) and the tools pane (for a
    button), because an imgui popup only renders inside the window whose id
    stack opened it -- a single call site would leave whichever half did not
    make it silently doing nothing when clicked.

    Opened by name rather than by holding the ``Op``: the popup survives across
    frames and the registry is the only thing allowed to own that object.
    """
    if not state.pending_op:
        return
    try:
        op = clay_ops.get(state.pending_op)
    except KeyError:  # pragma: no cover - a stale name from a removed op
        state.pending_op = ""
        state.open_op_popup = False
        return

    if state.open_op_popup:
        # A request from outside a window -- the keyboard path, which cannot
        # call open_popup itself. Cleared here whether or not the popup ends up
        # rendering, so a request can never outlive the frame that made it.
        state.open_op_popup = False
        imgui.open_popup(PARAM_POPUP)
    if not imgui.begin_popup(PARAM_POPUP):
        return
    widgets.popup_chrome(_imgui=imgui)
    values = state.op_params.setdefault(op.name, clay_ops.defaults_for(op))
    imgui.text(op.label.rstrip("."))
    controls.menu_separator()
    if op.hint:
        # Above the fields, not below them: it is about which op you are in
        # rather than about a number, so a user who is in the wrong one should
        # read it before they start typing into the right one's dialog.
        #
        # Wrapped at an explicit column rather than through ``muted_wrapped``,
        # which wraps at the content region's right edge. A popup *auto-sizes to
        # its content*, so in here that edge is whatever the widest item already
        # is -- there is nothing yet to be wide, so a three-line hint would have
        # sized the popup to its own single longest line instead of wrapping.
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + sp(HINT_WRAP))
        widgets.text_colored(theme.MUTED, op.hint)
        imgui.pop_text_wrap_pos()
        imgui.dummy((0, sp(tokens.SP_1)))
    for param in op.params:
        # Label above the field (2026-09-08 consistency pass); id kept
        # stable, "Foo##op-name" -> "##Foo##op-name".
        widgets.field_label(param.label)
        label = f"##{param.label}##{op.name}-{param.name}"
        if param.boolean:
            # A checkbox rather than an int spinner clamped to 0/1 -- "fit to
            # gap (0=off, 1=on)" was a label carrying the widget's job because
            # ``Param`` had no boolean until 2026-09-10. Stored exactly as the
            # int field it replaced did (0.0/1.0), so nothing downstream of
            # this loop had to change.
            changed, flag = controls.checkbox(
                label, bool(values.get(param.name, param.default))
            )
            if changed:
                values[param.name] = 1.0 if flag else 0.0
        elif param.choices:
            # A combo over named options rather than a spinner reading "axis
            # (0=X, 1=Y, 2=Z)". The value stored is still the option's index
            # -- the same int the field it replaced already wrote -- so
            # ``run`` and the op function need not know the widget changed.
            options = [(str(i), choice) for i, choice in enumerate(param.choices)]
            current = str(int(values.get(param.name, param.default)))
            changed, picked = controls.combo(label, current, options)
            if changed:
                values[param.name] = float(int(picked))
        elif param.integer:
            # Honoured rather than declared. Smooth's "levels" is the only
            # integer parameter and it was drawn as a float field, so it
            # accepted 1.5 and the op then truncated it -- a number the user
            # typed, silently becoming a different one.
            changed, value = controls.input_int(label, int(values.get(param.name, param.default)))
            if changed:
                values[param.name] = int(min(max(value, param.low), param.high))
        else:
            changed, value = controls.input_float(
                label,
                float(values.get(param.name, param.default)),
                param.step,
                0.0,
                clay_ops.format_for(param),
            )
            if changed:
                values[param.name] = min(max(float(value), param.low), param.high)
        if param.warn:
            widgets.secondary(param.warn)
    # Greyed rather than drawn live and ignored, which is what "and not
    # tab.saving" after the click amounted to.
    #
    # clay-13 (2026-09-08 audit, second run): this call passed no ``reason=``, unlike every
    # other saving-gated control in Clay (the tools-pane actions, the header's
    # mode field, the context-menu rows) -- so a user who opened this popup
    # mid-save saw Apply grey with nothing on screen saying why. ``widgets``
    # already carries the one wording every one of those uses
    # (``inker_export.BUSY_WHY``, ``inker_tiles.BUSY_WHY`` and
    # ``clay_header._SAVING`` are the same sentence copied three times);
    # reused here rather than defining a fourth copy local to this file.
    if widgets.disabled_button(
        f"Apply##{op.name}", not tab.saving, reason=widgets.DOCUMENT_SAVING_WHY
    ):
        clay_ops.run(ctx, tab.doc, op, **values)
        state.pending_op = ""
        imgui.close_current_popup()
    imgui.same_line()
    if controls.button(f"Cancel##{op.name}"):
        state.pending_op = ""
        imgui.close_current_popup()
    imgui.end_popup()
