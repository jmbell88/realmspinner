"""The question both silent Troupe doors used to skip.

``service.troupe.send_to_troupe`` has always accepted a sprite size and a rig
template, and the picker *inside* Troupe passes the mode's form so both apply.
The two doors a user actually reaches for -- the library's right-click item and
the inspector's button -- called it with no form at all, so ``logical_size``
arrived None and fell back to 32, and the skeleton was pinned to ``humanoid``.
A user who wanted 64 px sprites had to know to enter Troupe first and open a
collapsed sub-header; a user with a quadruped got human walk cycles, and the
manual's own advice was to go and re-rig it from Create.

So the doors ask. One modal, enqueued from anywhere and drawn at top level --
the ``dialogs.ConfirmQueue`` shape, because the library's item is inside an
imgui context popup and ``imgui.open_popup`` cannot be called there -- with the
single slot ``matte_preview`` uses, since at most one send is in flight.

**The answers are written back into the mode's form**, not kept per door, so
the size chosen at the library is the size the inspector opens on and Troupe's
own pane shows the same numbers. Two doors remembering separately would be two
defaults for one request.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from imgui_bundle import imgui

from ... import rigging
from ...pipelines import charsheet
from .. import controls, tokens, troupe_mode, widgets
from ..tokens import sp

TITLE = "Send to Troupe"

#: The floor ``widgets.modal_bounds`` is given. The combos are full-width, so
#: this is what decides how wide the dialog reads.
DIALOG_W = 420.0


@dataclass
class TroupeSend:
    """The mesh being sent, and the answers so far.

    The answers are held here rather than edited straight into the mode's form
    because Cancel has to mean cancel: writing live would leave a dismissed
    dialog's choices behind as the next send's defaults.
    """

    job_id: str = ""
    label: str = ""
    #: Whether the mesh already carries ``rig.glb``. Read off the row's cached
    #: ``files`` list -- the rig's *template* is a fact about a file on disk,
    #: and reading it here is the disk read ``can_send_to_troupe`` deliberately
    #: does not do on the frame thread.
    rigged: bool = False
    template: str = ""
    #: This mesh's own recorded front, read once at :func:`ask` -- a fact
    #: about the job, not a question this dialog asks. See ``_front_helper``.
    front_yaw: float = 0.0
    #: P4 (2026-09-13): whether the rig's own skeleton was edited away from its
    #: template (``rig.json["skeleton"] == "custom"``), and how many bones the
    #: template's clip library animates that this rig no longer has
    #: (``rigging.clip_coverage``). Read once here, not per frame -- a rig read
    #: is a file, and ``_skeleton`` draws every frame the dialog is open.
    custom_skeleton: bool = False
    custom_skeleton_missing: int = 0
    logical_size: int = 32
    #: Whether the size box is the "Custom..." input rather than the ladder
    #: combo. Its own field rather than inferred solely from ``logical_size``
    #: being off the ladder, so a user who *chose* Custom and then typed a
    #: value that happens to sit on a preset (say, 32) is not silently bounced
    #: back to the combo underneath them.
    custom_size: bool = False
    camera: str = ""
    outline: str = ""
    colors: int = 64
    palette: str = ""
    # ``imgui.open_popup`` must be called exactly once per question, and the
    # overlay redraws every frame: ``dialogs.Confirm._open``'s idiom.
    _open: bool = False


def ask(ctx: Any, job: dict[str, Any] | None) -> bool:
    """Put the question in front of the send. -> whether it was asked.

    Draws nothing itself; the overlay picks it up on the next frame, which is
    what lets the library's context menu call it from inside a popup.
    """
    job_id = str((job or {}).get("id") or "")
    if not job_id:
        return False
    form = troupe_mode.form(ctx)
    options = troupe_mode.options(ctx)
    logical_size = int(form.get("logical_size") or 32)
    rigged = "rig.glb" in ((job or {}).get("files") or [])
    custom_skeleton = False
    custom_missing = 0
    if rigged:
        # P4 (2026-09-13): a rig whose skeleton was edited away from its
        # template may no longer have every bone the template's clip library
        # animates -- Troupe warns, once, at the door, rather than a silently
        # thinner walk cycle discovered after the render. Read tolerantly:
        # an unreadable rig here is not this dialog's refusal to raise, only a
        # missed warning -- the send itself re-reads the rig and is the real
        # gate.
        from ...service import rig as svc_rig

        with contextlib.suppress(Exception):
            rig = svc_rig.get_rig(ctx.svc, job_id)
            if rig.get("skeleton") == "custom":
                custom_skeleton = True
                custom_missing = len(
                    rigging.clip_coverage(rig, str(rig.get("template") or ""))
                )
    ctx.state.troupe_send = TroupeSend(
        job_id=job_id,
        label=str((job or {}).get("prompt") or (job or {}).get("name") or "")[:48],
        rigged=rigged,
        custom_skeleton=custom_skeleton,
        custom_skeleton_missing=custom_missing,
        front_yaw=float(((job or {}).get("params") or {}).get("front_yaw") or 0.0),
        template=str(form.get("template") or ""),
        logical_size=logical_size,
        # Off-ladder means the field is already a custom answer -- the form
        # opens on the Custom box rather than silently snapping it to a
        # preset it does not hold.
        custom_size=logical_size not in (options.get("logical_sizes") or ()),
        camera=str(form.get("camera") or ""),
        outline=str(form.get("outline") or ""),
        colors=int(form.get("colors") or 64),
        palette=str(form.get("palette") or ""),
    )
    return True


def close(ctx: Any) -> None:
    state = getattr(getattr(ctx, "state", None), "troupe_send", None)
    if state is not None:
        ctx.state.troupe_send = None


def is_open(ctx: Any) -> bool:
    """Whether the dialog is up. Tolerant of a partial ctx.

    ``getattr`` rather than attribute access, ``matte_preview.is_open``'s rule
    and here for its reason: ``App._modal_open`` asks this on every key press
    and must not require a state object the caller has never built.
    """
    state = getattr(getattr(ctx, "state", None), "troupe_send", None)
    return state is not None and bool(state.job_id)


def draw(ctx: Any) -> None:
    """The modal. Beside the confirms, because it is one."""
    state = getattr(ctx.state, "troupe_send", None)
    if state is None or not state.job_id:
        return
    appearing = not state._open
    if appearing:
        imgui.open_popup(TITLE)
        state._open = True
    alpha, rise = widgets.popover_enter("troupe-send", appearing)
    frosted = widgets.frosted()
    if frosted:
        imgui.set_next_window_bg_alpha(0.0)
    imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
    radius = widgets.push_surface_rounding()
    widgets.modal_bounds(sp(DIALOG_W))
    opened, _ = imgui.begin_popup_modal(
        TITLE, None, imgui.WindowFlags_.always_auto_resize.value
    )
    widgets.pop_surface_rounding()
    if not opened:
        # Escape dismisses a modal without going through either button, and
        # imgui will not reopen a popup whose id it thinks is already open.
        imgui.pop_style_var()
        close(ctx)
        return
    widgets.window_shadow("overlay", radius=radius)
    if frosted:
        widgets.window_backdrop(radius=radius)
    if rise > 0.0:
        imgui.dummy((0, rise))
    _body(ctx, state)
    imgui.end_popup()
    imgui.pop_style_var()


def _body(ctx: Any, state: TroupeSend) -> None:
    options = troupe_mode.options(ctx)
    form = troupe_mode.form(ctx)
    # The body scrolls and the action row does not (INVARIANTS: a bounded
    # modal puts its body in ``modal_body`` and draws the buttons after it).
    with widgets.modal_body("troupe-send-body"):
        if state.label:
            widgets.muted(state.label)
        _skeleton(state, options)
        _size(state, options)
        presets = options.get("camera_presets") or {}
        state.camera = widgets.labeled_combo(
            "Camera",
            state.camera,
            [(key, str(entry.get("label") or key)) for key, entry in presets.items()],
        )
        helper = _camera_helper(presets, state.camera)
        if helper:
            widgets.muted(helper)
        widgets.muted(_front_helper(state.front_yaw))
        state.outline = widgets.labeled_combo(
            "Outline",
            state.outline,
            [(m, m) for m in options.get("outline_modes") or ()],
        )
        # Shown only when no palette is named, mirroring ``troupe_settings``:
        # the budget is what a *derived* palette gets, so offering it beside a
        # named one would be a control whose value is silently ignored.
        if state.palette:
            widgets.muted(f"Palette: {state.palette}")
        else:
            state.colors = int(
                widgets.labeled_combo(
                    "Colours",
                    str(state.colors),
                    [(str(n), f"{n} colours") for n in options.get("colors") or ()],
                )
            )
    imgui.dummy((0, sp(6)))
    _actions(ctx, state, form)


def _skeleton(state: TroupeSend, options: dict[str, Any]) -> None:
    """Which rig an unrigged mesh is built on, when there is a choice.

    A rigged mesh is not asked: the skeleton is already on disk and the service
    reads it off ``rig.json``, so a picker here would be a control whose value
    that branch discards.
    """
    if state.rigged:
        widgets.muted("This mesh is already rigged; its own skeleton is used.")
        if state.custom_skeleton and state.custom_skeleton_missing:
            n = state.custom_skeleton_missing
            widgets.muted_wrapped(
                f"Custom skeleton: clips skip {n} bone{'s' if n != 1 else ''} "
                "this rig does not have."
            )
        return
    choices = [
        (str(row.get("key")), str(row.get("label") or row.get("key")))
        for row in options.get("clip_templates") or ()
    ]
    if not choices:
        return
    if state.template not in {key for key, _label in choices}:
        state.template = choices[0][0]
    state.template = widgets.labeled_combo(
        "Skeleton",
        state.template,
        choices,
        help_text=(
            "The rig this mesh is built on, and the clip library its sheet is "
            "animated from. Only the skeletons with clips authored for them "
            "are offered."
        ),
    )


#: The combo's sentinel for "type your own number" -- distinct from every
#: ladder entry, which are all digit strings, so it can never collide with a
#: size the ladder actually offers.
_CUSTOM = "custom"


def _size(state: TroupeSend, options: dict[str, Any]) -> None:
    """Sprite size: the ladder, or a hand-typed value in ``logical_size_range``.

    ``troupe_settings._size``'s pane-side twin, kept in step because the two
    are the same question asked from two doors -- a size chosen here has to
    read back the same way in Troupe's own settings form.
    """
    choices = [(str(s), f"{s} px") for s in options.get("logical_sizes") or ()]
    choices.append((_CUSTOM, "Custom..."))
    combo_value = _CUSTOM if state.custom_size else str(state.logical_size)
    picked = widgets.labeled_combo("Sprite size", combo_value, choices)
    if picked == _CUSTOM:
        state.custom_size = True
    else:
        state.custom_size = False
        state.logical_size = int(picked)
    if state.custom_size:
        lo, hi = options.get("logical_size_range") or (8, 256)
        _changed, value = controls.input_int("##troupe-send-size", int(state.logical_size))
        state.logical_size = max(int(lo), min(int(hi), int(value)))
        if state.logical_size and charsheet.RENDER_SIZE % state.logical_size != 0:
            widgets.muted_wrapped(
                "Sizes that don't divide 512 are resized with nearest-neighbour."
            )


def _front_helper(front_yaw: float) -> str:
    """What this mesh's own front is, read-only -- ``_camera_helper``'s shape.

    **Read-only, no override control.** The front is set from Poser or the
    viewport toolbar, both places the user is looking at the model turning
    under the press; a number box in a send dialog the user opened to answer
    two questions about sprite size and skeleton would be the worse tool for
    the same job, with no picture beside it to judge the angle by.
    """
    if not front_yaw:
        return "This mesh has no front set; sheets are rendered from yaw 0."
    return f"This mesh's front is set to {front_yaw:.0f} degrees; sheets are rendered from it."


def _camera_helper(presets: dict[str, Any], key: str) -> str:
    """``troupe_settings._camera_helper``'s sentence, for the same reason."""
    entry = presets.get(key) or {}
    if "elevation" not in entry:
        return ""
    return f"{float(entry['elevation']):g} degrees above the horizon"


def _actions(ctx: Any, state: TroupeSend, form: dict[str, Any]) -> None:
    from . import troupe_settings

    count = troupe_settings.cell_count(form)
    note = f"{count} cells are rendered at {state.logical_size} px."
    if not state.rigged:
        note = f"A mesh that is not rigged is rigged first. Then {note}"
    widgets.cost_note(note)
    imgui.dummy((0, sp(tokens.SP_1)))
    if controls.button("Send", (sp(150), 0), role=controls.ButtonRole.PRIMARY):
        # The popup is closed here and not in ``_send``: that function is the
        # whole decision -- write back, then submit -- and a headless caller
        # (a test, the exerciser) must be able to run it with no imgui frame.
        imgui.close_current_popup()
        _send(ctx, state, form)
        return
    imgui.same_line()
    if controls.button("Cancel", (sp(110), 0)):
        imgui.close_current_popup()
        close(ctx)


def _send(ctx: Any, state: TroupeSend, form: dict[str, Any]) -> None:
    """Write the answers back, then submit with the form the pane also draws.

    No imgui: see ``_actions``.
    """
    form["logical_size"] = int(state.logical_size)
    form["camera"] = state.camera
    form["outline"] = state.outline
    if not state.palette:
        form["colors"] = int(state.colors)
    if not state.rigged and state.template:
        form["template"] = state.template
    job_id = state.job_id
    close(ctx)
    troupe_mode.send_to_troupe(ctx, {"id": job_id}, form)
