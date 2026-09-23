"""The question both silent sheet-rendering doors used to skip.

``service.troupe.send_to_troupe`` has always accepted a sprite size and a rig
template, and this modal passes the mode's sheet form so both apply. The two
doors a user actually reaches for -- the library's right-click item and the
inspector's button, both in :mod:`..... asset_exits` -- called it with no form
at all, so ``logical_size`` arrived None and fell back to 32, and the skeleton
was pinned to ``humanoid``. A user who wanted 64 px sprites had to know to
open Poser first and find a collapsed sub-header; a user with a quadruped got
human walk cycles, and the manual's own advice was to go and re-rig it from
Create.

So the doors ask. One modal, enqueued from anywhere and drawn at top level --
the ``dialogs.ConfirmQueue`` shape, because the library's item is inside an
imgui context popup and ``imgui.open_popup`` cannot be called there -- with
the single slot ``matte_preview`` uses, since at most one send is in flight.

**The answers are written back into the mode's form**, not kept per door, so
the size chosen at the library is the size a "Build another sheet" section
opens on. Two doors remembering separately would be two defaults for one
request.

**P9 (2026-09-18):** ported whole from Troupe's own ``ui/panes/send.py`` --
this was already reachable from outside that mode (the library and inspector
"ways out"), so folding Troupe into Poser moves this module rather than
retiring it. ``ctx.state.troupe_send`` becomes ``ctx.state.poser_send``; the
mode it hands off to is ``poser_mode`` throughout.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from imgui_bundle import imgui

from ......kernels import charsheet
from ......kernels.rig import skeleton
from ..... import controls, tokens, widgets
from .....tokens import sp
from ... import mode as poser_mode

# The 2026-09-23 audit, finding poser-08: this pane carried its own byte-
# identical copy of sheet.py's ``_camera_helper``, untested and free to drift
# from the one the sheet pane actually exercises. Shared rather than
# reproduced again.
from .sheet import _camera_helper

TITLE = "Send to Poser"

#: The floor ``widgets.modal_bounds`` is given. The combos are full-width, so
#: this is what decides how wide the dialog reads.
DIALOG_W = 420.0


@dataclass
class PoserSend:
    """The mesh being sent, and the answers so far.

    The answers are held here rather than edited straight into the mode's form
    because Cancel has to mean cancel: writing live would leave a dismissed
    dialog's choices behind as the next send's defaults.
    """

    job_id: str = ""
    label: str = ""
    #: Whether the mesh already carries ``rig.glb``. Read off the row's cached
    #: ``files`` list -- the rig's *template* is a fact about a file on disk,
    #: and reading it here is the disk read ``can_render_sheet`` deliberately
    #: does not do on the frame thread.
    rigged: bool = False
    #: The skeleton this send actually resolves against -- read off the
    #: mesh's own ``rig.json`` when it is already rigged, or the Skeleton
    #: combo's pick otherwise. **Not** whatever character happens to be bound
    #: to Poser's own session: see :func:`_send`.
    template: str = ""
    #: This mesh's own recorded front, read once at :func:`ask` -- a fact
    #: about the job, not a question this dialog asks. See ``_front_helper``.
    front_yaw: float = 0.0
    #: Whether the rig's own skeleton was edited away from its template
    #: (``rig.json["skeleton"] == "custom"``), and how many bones the
    #: template's clip library animates that this rig no longer has. Read
    #: once here, not per frame -- a rig read is a file, and ``_skeleton``
    #: draws every frame the dialog is open.
    custom_skeleton: bool = False
    custom_skeleton_missing: int = 0
    logical_size: int = 32
    #: Whether the size box is the "Custom..." input rather than the ladder
    #: combo. Its own field rather than inferred solely from ``logical_size``
    #: being off the ladder, so a user who *chose* Custom and then typed a
    #: value that happens to sit on a preset (say, 32) is not silently
    #: bounced back to the combo underneath them.
    custom_size: bool = False
    camera: str = ""
    outline: str = ""
    colors: int = 64
    palette: str = ""
    #: Pixel art or HD -- ``poser_mode.STYLE_PIXEL_ART``/``STYLE_HD``. HD
    #: disables Outline and Colours below.
    style: str = poser_mode.STYLE_PIXEL_ART
    #: A layout-wide rate, or ``None`` for "Authored". See ``poser_mode.
    #: _layout_request``: set, it moves the request's layout to version 3.
    fps: int | None = None
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
    form = poser_mode.sheet_form(ctx)
    options = poser_mode.sheet_options(ctx)
    logical_size = int(form.get("logical_size") or 32)
    rigged = "rig.glb" in ((job or {}).get("files") or [])
    custom_skeleton = False
    custom_missing = 0
    rig_template = ""
    if rigged:
        # This reads rig.json synchronously from a button handler, on the
        # frame thread. Recorded here rather than moved off-thread, because
        # the read is a single small JSON on an explicit click, not a loop or
        # a poll, and it is already bounded -- ``store.read_record`` (which
        # ``get_rig`` goes through) stats the file before reading and refuses
        # anything over ``MAX_RECORD_BYTES`` (1 MiB) rather than loading it.
        # See dev/INVARIANTS.md.
        from ......service import rig as svc_rig

        with contextlib.suppress(Exception):
            rig = svc_rig.get_rig(ctx.svc, job_id)
            rig_template = str(rig.get("template") or "")
            if rig.get("skeleton") == "custom":
                custom_skeleton = True
                custom_missing = len(
                    skeleton.clip_coverage(rig, str(rig.get("template") or ""))
                )
    ctx.state.poser_send = PoserSend(
        job_id=job_id,
        label=str((job or {}).get("prompt") or (job or {}).get("name") or "")[:48],
        rigged=rigged,
        custom_skeleton=custom_skeleton,
        custom_skeleton_missing=custom_missing,
        front_yaw=float(((job or {}).get("params") or {}).get("front_yaw") or 0.0),
        # Rigged: the mesh's own recorded skeleton, read above -- never the
        # form's, which names whichever character is bound to Poser's own
        # session. Unrigged: the form's remembered choice, the same default
        # the Skeleton combo below opens on and may still change before Send.
        template=rig_template if rigged else str(form.get("template") or ""),
        logical_size=logical_size,
        custom_size=logical_size not in (options.get("logical_sizes") or ()),
        camera=str(form.get("camera") or ""),
        outline=str(form.get("outline") or ""),
        colors=int(form.get("colors") or 64),
        palette=str(form.get("palette") or ""),
        style=poser_mode._style_choice(form),
        fps=form.get("fps"),
    )
    return True


def close(ctx: Any) -> None:
    state = getattr(getattr(ctx, "state", None), "poser_send", None)
    if state is not None:
        ctx.state.poser_send = None


def is_open(ctx: Any) -> bool:
    """Whether the dialog is up. Tolerant of a partial ctx.

    ``getattr`` rather than attribute access, ``matte_preview.is_open``'s rule
    and here for its reason: ``App._modal_open`` asks this on every key press
    and must not require a state object the caller has never built.
    """
    state = getattr(getattr(ctx, "state", None), "poser_send", None)
    return state is not None and bool(state.job_id)


def draw(ctx: Any) -> None:
    """The modal. Beside the confirms, because it is one."""
    state = getattr(ctx.state, "poser_send", None)
    if state is None or not state.job_id:
        return
    appearing = not state._open
    if appearing:
        imgui.open_popup(TITLE)
        state._open = True
    alpha, rise = widgets.popover_enter("poser-send", appearing)
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


def _body(ctx: Any, state: PoserSend) -> None:
    options = poser_mode.sheet_options(ctx)
    form = poser_mode.sheet_form(ctx)
    # The body scrolls and the action row does not (INVARIANTS: a bounded
    # modal puts its body in ``modal_body`` and draws the buttons after it).
    with widgets.modal_body("poser-send-body"):
        if state.label:
            widgets.muted(state.label)
        _skeleton(ctx, state, options)
        _size(state, options)
        widgets.field_error(ctx.state, "layout")
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
        _frame_rate(ctx, state, options)
        _style(state)
        hd = state.style == poser_mode.STYLE_HD
        state.outline = widgets.labeled_combo(
            "Outline",
            state.outline,
            [(m, m) for m in options.get("outline_modes") or ()],
            enabled=not hd,
            reason="Style is HD, so there is no outline pass." if hd else "",
        )
        if state.palette:
            widgets.muted(f"Palette: {state.palette}")
        else:
            state.colors = int(
                widgets.labeled_combo(
                    "Colours",
                    str(state.colors),
                    [(str(n), f"{n} colours") for n in options.get("colors") or ()],
                    enabled=not hd,
                    reason="Style is HD, so there is no colour budget." if hd else "",
                )
            )
    imgui.dummy((0, sp(6)))
    _actions(ctx, state, form)


def _skeleton(ctx: Any, state: PoserSend, options: dict[str, Any]) -> None:
    """Which rig an unrigged mesh is built on, when there is a choice.

    A rigged mesh is not asked: the skeleton is already on disk and the
    service reads it off ``rig.json``, so a picker here would be a control
    whose value that branch discards.
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
    before = state.template
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
    if state.template != before:
        ctx.state.clear_field_error("template")
    widgets.field_error(ctx.state, "template")


def _style(state: PoserSend) -> None:
    """Pixel art or HD -- the same control the sheet form draws, mirrored."""
    state.style = widgets.labeled_combo(
        "Style",
        state.style,
        [
            (poser_mode.STYLE_PIXEL_ART, "Pixel art"),
            (poser_mode.STYLE_HD, "HD"),
        ],
        help_text=(
            "Pixel art reduces the render to a logical size, a colour budget "
            "and an outline pass. HD keeps the render as painted, with no "
            "colour budget."
        ),
    )


def _frame_rate(ctx: Any, state: PoserSend, options: dict[str, Any]) -> None:
    """A layout-wide rate, or every clip's own recorded speed. The sheet
    form's control, mirrored."""
    choices = [("", "Authored")] + [
        (str(n), f"{n} fps") for n in options.get("fps_choices") or ()
    ]
    current = "" if state.fps in (None, "") else str(state.fps)
    before = current
    choice = widgets.labeled_combo(
        "Frame rate",
        current,
        choices,
        help_text=(
            "Authored keeps every included movement at its own recorded "
            "speed. A rate here overrides all of them to play at once."
        ),
    )
    state.fps = int(choice) if choice else None
    if choice != before:
        ctx.state.clear_field_error("fps")
    widgets.field_error(ctx.state, "fps")


#: The combo's sentinel for "type your own number".
_CUSTOM = "custom"


def _size(state: PoserSend, options: dict[str, Any]) -> None:
    """Sprite size: the ladder, or a hand-typed value in
    ``logical_size_range``. ``ui/panes/sheet.py``'s own combo, kept in step:
    a size chosen here has to read back the same way in that form."""
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
        _changed, value = controls.input_int("##poser-send-size", int(state.logical_size))
        state.logical_size = max(int(lo), min(int(hi), int(value)))
    # Outside the custom-only branch on purpose: the 2026-09-23 audit
    # (poser-07) found this hint gated on ``state.custom_size``, so a preset
    # that also fails to divide 512 -- 24, 48 and 96 among the ladder's own
    # choices -- was NEAREST-resized with no warning at all, while typing
    # that same number by hand got one.
    if state.logical_size and charsheet.RENDER_SIZE % state.logical_size != 0:
        widgets.muted_wrapped(
            "Sizes that don't divide 512 are resized with nearest-neighbour."
        )


def _front_helper(front_yaw: float) -> str:
    """What this mesh's own front is, read-only -- ``_camera_helper``'s shape.

    **Read-only, no override control.** The front is set from Poser or the
    viewport toolbar, both places the user is looking at the model turning
    under the press.
    """
    if not front_yaw:
        return "This mesh has no front set; sheets are rendered from yaw 0."
    return f"This mesh's front is set to {front_yaw:.0f} degrees; sheets are rendered from it."


def _actions(ctx: Any, state: PoserSend, form: dict[str, Any]) -> None:
    count = poser_mode.cell_count(form)
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


def _send(ctx: Any, state: PoserSend, form: dict[str, Any]) -> None:
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
    form["style"] = state.style
    form["fps"] = state.fps
    # ``form["layout"]`` is built by ``poser_mode.sheet_form``/
    # ``_default_sheet_layout`` against whichever character is bound to
    # Poser's own session, not against the mesh this dialog is actually
    # sending -- a separate id chosen from the Library or the inspector.
    # Rebuilt here whenever the two disagree, so a character open in Poser --
    # and any layout it carries, hand-edited or not -- cannot leak onto an
    # unrelated mesh sent through this door. ``state.template`` empty means
    # the skeleton could not be resolved (an unreadable ``rig.json``, read
    # tolerantly above) -- the form's own layout is kept rather than replaced
    # with one built for no template at all, and ``create_charsheet``
    # re-reads the rig and is the real gate.
    if state.template and state.job_id != poser_mode.ensure(ctx).job_id:
        form["layout"] = poser_mode._layout_for_sheet_template(ctx, state.template)
    job_id = state.job_id
    close(ctx)
    poser_mode.render_character_sheet(ctx, {"id": job_id}, form)
