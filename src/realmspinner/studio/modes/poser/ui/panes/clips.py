"""Poser's clip editor: authoring the keyframes a character sheet animates.

The Troupe programme's clip-authoring half. The clip *format* and its expansion
shipped with Troupe; what did not was any way to change a clip that is not
editing JSON in the package tree, and authoring those twenty-two keyframes is
the most important art task in the programme.

**The armature is the editor.** Poser already has a skeleton, gizmos and a pose
editor, so this pane adds no second posing surface: picking a key loads it onto
the preview armature, you drag joints with the controls on the right exactly as
you would for a library pose, and *Update key* puts it back. Everything a clip
adds on top of that is timing -- which keys, in what order, how many frames
apart -- and that is what the list here is.

**Scrubbing is not editing, and the pane says so.** A scrubbed frame is
interpolated between two keys and has nowhere to store an edit, so while the
scrubber is engaged the pane shows the frame number rather than a key and
*Update key* refuses by name. Coming back to a key is one click and re-applies
the authored pose.

Drawn in the left sidebar under the pose library rather than in a mode of its
own: a clip is a *library* of the same kind of thing the poses above it are,
and the right sidebar has to stay free for the joint controls that are the
actual editing surface.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ......kernels.rig import cliplib
from ..... import controls, forms, icons, theme, tokens, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import mode as poser_mode


def _provisional_note(record: dict[str, Any] | None) -> str:
    """The picker's badge text for *record*, or "" when none is owed.

    A pure lookup so the picker's badge is testable without an imgui frame,
    ``_update_key_reason``'s own reason for existing. The five clips shipped
    with ``provisional: true`` (attack_02, cast, fall, hit, death) are
    keyframed but not yet an animator's pass -- see
    ``docs/manual/26-poser.md`` -- and the picker is where an author decides
    which clip to open next, so the badge belongs beside the name rather than
    after the fact.
    """
    return "provisional" if record and record.get("provisional") else ""


def _key_pending(viewer: Any, frame: int) -> bool:
    """Whether the live pose differs from the selected key.

    ``PoseEditor.dirty`` already means exactly this once a key is loaded:
    ``apply_key`` clears it the moment a key lands on the armature, and every
    mutation that follows -- a gizmo drag's ``rotate_selected``/``move_root``,
    a numeric edit through the same calls, a mirror, a reset -- sets it back.
    No second comparison is needed, only reading the flag at the right time:
    never while scrubbing, where the armature holds an interpolated frame
    rather than the selected key at all.
    """
    return bool(
        viewer is not None
        and viewer.pose_mode
        and frame < 0
        and viewer.editor.has_unsaved_edits()
    )


def _not_posing_reason(error: str, asset_error: str) -> str:
    """Why the armature is not posable right now, when it is not.

    ``state.clips`` is refreshed independently of the preview build
    (``state.error``) and the bound asset's rig load (``state.asset_error``),
    so a Blender build failure -- or a broken rig on the asset this session
    has open -- left a perfectly good clip library on screen with both of
    this pane's buttons claiming the preview was "still loading" indefinitely
    (the 2026-09-11 audit, finding poser-07). Named after whichever actually
    failed; "still loading" is the fallback for the one case that really is
    in progress.
    """
    return error or asset_error or "The skeleton preview is still loading."


def _update_key_reason(posing: bool, frame: int, *, error: str = "", asset_error: str = "") -> str:
    """Why "Update key from pose" is disabled right now. -> the sentence.

    ``posing`` checked first: the 2026-09-07 audit (poser-07) found this
    tested ``frame`` first, so while the preview was still loading -- and
    ``frame`` happened to hold a stale in-between value from before the
    switch -- the button named "That is an in-between frame" when the real
    cause was that posing had not started yet. A pure function so the order
    is assertable without a GL context.
    """
    if not posing:
        return _not_posing_reason(error, asset_error)
    return "That is an in-between frame, not a key. Pick a key first."


def _new_key_reason(
    posing: bool, frame: int = -1, *, error: str = "", asset_error: str = ""
) -> str:
    """Why "New key from pose..." is disabled right now. -> the sentence.

    The 2026-09-11 audit (poser-07): this button stated its reason as a
    literal inline string rather than through a testable function at all,
    unlike its neighbour above -- given the same treatment here so a build or
    load failure says so instead of "still loading" on this button too.

    ``frame`` was added by the 2026-09-23 audit (poser-03): the button was
    gated on bare ``posing``, never on ``frame``, so scrubbing to an
    in-between frame and pressing it authored the *interpolated* pose as a
    brand-new key -- "Update key" next to it already refused the same state.
    Checked after ``posing`` for the reason ``_update_key_reason`` picks its
    own order: a build or load failure has to say so before anything about
    the scrubber, or the button would blame scrubbing while the real cause is
    that the preview never came up at all.
    """
    if not posing:
        return _not_posing_reason(error, asset_error)
    if frame >= 0:
        return "That is an in-between frame, not a key. Pick a key first."
    return ""


def _import_clip_reason(
    rigging_available: bool, has_library: bool, busy: bool, skeleton_editing: bool = False
) -> str:
    """Why "Import clip..." is disabled right now, or "" if it is not.

    Checked in this order for the same reason ``_update_key_reason`` picks its
    own: whichever fact actually explains the greyed button, named first.
    ``skeleton_editing`` goes first of all (P6, 2026-09-13): master hides the
    whole Clips section during a skeleton edit because every control in it
    reads or writes the armature's *pose*, which a skeleton draft holds at
    rest throughout, and this button is the one control this branch still
    draws through that state (see ``draw``'s comment) -- so while it is true,
    it is the only reason that matters, ahead of Blender or a busy import.
    Blender missing means nothing here can even sample the file; no library
    means :func:`poser_mode.adopt_imported_clips` would have nothing to merge
    into (``state.clips`` is empty for a template that ships none); busy means
    a previous import is still out sampling one.
    """
    if skeleton_editing:
        return "Apply or cancel the skeleton edit first."
    if not rigging_available:
        return "Importing an animation needs Blender, which is not installed."
    if not has_library:
        return "This skeleton has no clip library to import into."
    if busy:
        return "Still importing."
    return ""


def _import_button(ctx: Any, state: Any) -> None:
    busy = ctx.busy(poser_mode.CLIP_IMPORT_KEY)
    has_library = bool(state.clips.get("clips"))
    reason = _import_clip_reason(
        bool(ctx.rigging_available), has_library, busy, bool(state.skeleton_editing)
    )
    if widgets.disabled_button(
        "Import clip...",
        not reason,
        (-1, 0),
        reason=reason,
        tooltip=(
            "Bring in an animation from an FBX or GLB file (Mixamo or Rigify "
            "naming) as a new clip."
        ),
    ):
        poser_mode.import_clip(ctx)
    if not state.skeleton_editing:
        # The report is an account of a *finished* import; master hides this
        # whole section during a skeleton edit (``draw``, below) and a report
        # left visible above that early return would be the one piece of it
        # still on screen while nothing else is.
        _import_report(ctx, state)


def _import_report(ctx: Any, state: Any) -> None:
    """A per-clip account of the most recent import, collapsed by default.

    Drawn from ``state.clip_import_reports`` -- ``cliptransfer.transfer``'s
    own ``report`` dicts, kept verbatim -- rather than anything re-derived, so
    what an author reads here is exactly what the sample actually decided.

    ``state.clip_import_skipped`` is drawn first and separately: an action
    Blender refused to sample at all (today, only for exceeding
    ``op_clip_sample``'s frame limit) was never converted, so it has no
    ``report`` dict to sit inside the loop below. The 2026-09-18 audit,
    finding poser-01: a source file whose every action was skipped used to
    open this header (once there was anything to open it for -- before this
    fix, nothing, since ``reports`` was empty too) to nothing at all.
    """
    del ctx
    reports = state.clip_import_reports
    skipped = list(getattr(state, "clip_import_skipped", None) or ())
    if not reports and not skipped:
        return
    if not controls.collapsing_header("Import report##poser-clip-import-report"):
        return
    for line in skipped:
        widgets.muted(f"Skipped: {line}")
    if skipped:
        imgui.dummy((0, sp(tokens.SP_1)))
    for report in reports:
        loop = report.get("loop") or {}
        ignored = list(report.get("ignored") or ())
        left_at_rest = list(report.get("left_at_rest") or ())
        widgets.muted(f"map: {report.get('map') or 'unknown'}")
        widgets.muted(f"{len(left_at_rest)} bone(s) left at rest")
        if ignored:
            shown = ", ".join(ignored[:3])
            more = f" and {len(ignored) - 3} more" if len(ignored) > 3 else ""
            widgets.muted(f"{len(ignored)} source bone(s) ignored: {shown}{more}")
        else:
            widgets.muted("0 source bone(s) ignored")
        # The 2026-09-16 audit's poser-01 fix threaded this field through
        # cliptransfer.transfer's report but, correctly per its own brief,
        # left rendering it to whoever owns this pane; closed the same day.
        duplicates = dict(report.get("duplicate_source_names") or {})
        if duplicates:
            names = ", ".join(sorted(duplicates)[:3])
            more = f" and {len(duplicates) - 3} more" if len(duplicates) > 3 else ""
            widgets.muted(
                f"{len(duplicates)} normalized source bone name(s) collided: "
                f"{names}{more} -- only one raw name each kept its motion"
            )
        closed = "loops" if loop.get("closed") else "does not loop"
        residual = loop.get("residual_deg")
        residual_text = f", {residual:.1f} deg residual" if residual is not None else ""
        widgets.muted(f"{closed}{residual_text}")
        widgets.muted(
            f"{report.get('frames') or 0} frames, {report.get('keys') or 0} keys, "
            f"root motion: {report.get('root_motion') or 'none'}"
        )
        imgui.dummy((0, sp(tokens.SP_1)))


def draw(ctx: Any) -> None:
    state = poser_mode.ensure(ctx)
    # A section, not a collapsing header: every other workspace's column pane
    # opens with one, and Poser alone could fold its pane shut (2026-09-05).
    widgets.section("Clips")
    manual_render.help_button(ctx, "poser-clips")
    # Drawn before every bail-out below, and that is deliberate: three of this
    # button's four disabled reasons (a skeleton edit in progress, no Blender,
    # no clip library for this skeleton) are exactly the states those
    # bail-outs short-circuit on, so a button that only existed past them
    # could never say why it was missing.
    _import_button(ctx, state)
    if not ctx.rigging_available:
        widgets.muted("Editing clips needs Blender, which is not installed.")
        return
    if state.skeleton_editing:
        # P6 (2026-09-13): every control below reads or writes the armature's
        # *pose*, and a skeleton-editing session holds it at rest throughout
        # (``PoseEditor.enter_skeleton_mode`` resets it on the way in) -- key
        # capture and scrubbing would silently write or play a rest pose.
        # ``poser_mode.scrub``/``capture_key`` refuse this by name too, but
        # hiding the whole section is the honester picture of what a skeleton
        # draft actually has to offer a clip.
        widgets.muted_wrapped(
            "Editing the skeleton. Apply or cancel to get back to posing and clips."
        )
        return
    poser_mode.clips_pump(ctx)
    if state.clips_loading and not state.clips:
        widgets.muted("Reading the clip library...")
        return
    if not state.clips.get("clips"):
        widgets.muted_wrapped(
            "This skeleton ships no clips. Clips are what a character sheet "
            "animates; only the humanoid template has them today."
        )
        return

    with forms.Form("poser-clips"):
        _picker(ctx, state)
        _keys(ctx, state)
        _timing(ctx, state)
        _scrubber(ctx, state)
        _save(ctx, state)


def _picker(ctx: Any, state: Any) -> None:
    names = [str(c.get("name") or "") for c in state.clips.get("clips") or ()]
    # ``labeled_combo`` and not a named ``combo``: imgui draws a combo's name
    # to its *right*, and a full-width one in a sidebar puts that name past the
    # content region, where ``same_line`` clips rather than wrapping -- so the
    # name is simply never drawn. ``widgets.combo``'s docstring states the rule
    # and ``test_inker_ux`` enforces it.
    # ``labeled_combo`` answers with the key alone, not a (changed, key) pair
    # -- ``select_clip`` is a no-op on an unchanged name, so the comparison is
    # the whole of the change test.
    picked = widgets.labeled_combo(
        "Clip",
        state.clip,
        [(n, n) for n in names],
        help_text="Which animation's keyframes this list is showing.",
    )
    if picked != state.clip:
        poser_mode.select_clip(ctx, picked)
    note = _provisional_note(state.open_clip())
    if note:
        # The muted-label idiom the rest of the app uses for a fact beside a
        # name rather than a sentence of its own (``widgets.muted``).
        imgui.same_line()
        widgets.muted(note)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Placeholder keyframes; an animator's pass is still owed")
    if state.clips.get("edited"):
        widgets.muted_wrapped("edited - this skeleton is using your clips, not the shipped ones")


def _keys(ctx: Any, state: Any) -> None:
    record = state.open_clip()
    if record is None:
        return
    keys = list(record.get("keys") or ())
    segments = list(record.get("segments") or ())
    widgets.section("Keyframes")
    for index, name in enumerate(keys):
        selected = index == state.key_index and state.frame < 0
        # The frames *out of* this key, which is what a segment is -- shown on
        # the row it belongs to rather than in a second list, so reordering a
        # key visibly carries its timing.
        span = segments[index] if index < len(segments) else None
        label = f"{index + 1}. {name}" + (f"   {span}f" if span is not None else "")
        if controls.selectable_row(f"poser-clip-key/{index}", label, selected=selected):
            poser_mode.select_key(ctx, index)

    changed, onion = controls.checkbox(
        "Onion skin",
        state.onion,
        tooltip=(
            "Ghost the keys either side of this one in the viewport. A looping "
            "clip wraps, so the first key's neighbour is the last."
        ),
    )
    if changed:
        poser_mode.set_onion(ctx, onion)

    imgui.dummy((0, sp(tokens.SP_1)))
    if widgets.disabled_button(
        "Up",
        state.key_index > 0,
        reason="This is already the first key.",
        tooltip="Move this key earlier, carrying its own timing with it.",
    ):
        poser_mode.move_key(ctx, state.key_index, -1)
    imgui.same_line()
    if widgets.disabled_button(
        "Down",
        state.key_index < len(keys) - 1,
        reason="This is already the last key.",
        tooltip="Move this key later, carrying its own timing with it.",
    ):
        poser_mode.move_key(ctx, state.key_index, 1)
    imgui.same_line()
    if widgets.disabled_button(
        "Remove",
        len(keys) > 2,
        reason="A clip needs at least two keys; one key is a pose.",
        tooltip=(
            "Take this key out of the clip. The pose itself stays in the "
            "library -- another clip may use it."
        ),
    ):
        poser_mode.remove_key(ctx, state.key_index)

    viewer = poser_mode.viewer_of(ctx)
    posing = viewer is not None and viewer.pose_mode
    pending = _key_pending(viewer, state.frame)
    if widgets.disabled_button(
        "Update key from pose",
        posing and state.frame < 0,
        (-1, 0),
        reason=_update_key_reason(
            posing, state.frame, error=state.error, asset_error=state.asset_error
        ),
        tooltip="Store the joints as they are now into the selected key.",
    ):
        poser_mode.capture_key(ctx)
    if pending:
        # An accent dot, not a second sentence beside the button's own
        # tooltip: the button already says what it does, this says only that
        # doing it now would change something.
        imgui.same_line()
        widgets.text_colored(theme.ACCENT, icons.CIRCLE)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Pose differs from this key")
    if widgets.disabled_button(
        "New key from pose...",
        posing and state.frame < 0,
        (-1, 0),
        reason=_new_key_reason(
            posing, state.frame, error=state.error, asset_error=state.asset_error
        ),
        tooltip=(
            "Add the joints as they are now as a brand-new key pose, after "
            "the selected one."
        ),
    ):
        _ask_new_key(ctx)
    _insert_existing(ctx, state)


def _insert_existing(ctx: Any, state: Any) -> None:
    """Add a key the library already holds, after the selected one.

    ``poser_mode.insert_key`` and ``PoserState.key_names`` were both written
    for this and neither had a caller: the pane could author a *new* key pose
    and could not reuse one, so building a walk out of four poses meant
    authoring each of them twice. The combo lists what the library has, minus
    nothing -- a clip may legitimately visit the same key twice, which is how
    a there-and-back cycle is written without a pingpong.
    """
    names = [name for name in state.key_names() if name]
    if not names:
        return
    slot = "poser-insert-key"
    current = str(ctx.state.preview.get(slot) or names[0])
    if current not in names:
        current = names[0]
    picked = widgets.labeled_combo(
        "Existing key", current, [(name, name) for name in names]
    )
    if picked != current:
        ctx.state.preview[slot] = picked
        current = picked
    if widgets.disabled_button(
        "Insert this key",
        bool(state.open_clip()),
        (-1, 0),
        reason="Open a clip first.",
        tooltip="Put a key the library already holds after the selected one.",
    ):
        poser_mode.insert_key(ctx, current)


def _ask_new_key(ctx: Any) -> None:
    from ..... import dialogs

    ctx.prompts.ask(
        dialogs.Prompt(
            title="Name this key",
            label="Name",
            on_accept=lambda name: poser_mode.new_key(ctx, name),
        )
    )


def _timing(ctx: Any, state: Any) -> None:
    record = state.open_clip()
    if record is None:
        return
    from ......service import clips as svc_clips

    widgets.section("Timing")
    widgets.field_label("Frame time (ms)")
    changed, duration_ms = controls.input_int(
        "##Frame time (ms)",
        int(record.get("duration_ms") or cliplib.CLIP_DURATION_STEP_MS),
        cliplib.CLIP_DURATION_STEP_MS,
        cliplib.CLIP_DURATION_STEP_MS,
        tooltip="How long each rendered frame lasts in sprite sheets and in the animated GLB.",
    )
    if changed:
        poser_mode.set_duration(ctx, duration_ms)
    imgui.same_line()
    widgets.muted(f"≈ {poser_mode.clip_fps(record.get('duration_ms')):.1f} fps")

    segments = list(record.get("segments") or ())
    index = min(state.key_index, len(segments) - 1) if segments else -1
    if index >= 0:
        # Label above, matching "Easing" below it (2026-09-08 consistency
        # pass); id kept stable, "Frames after this key" -> "##Frames after
        # this key".
        widgets.field_label("Frames after this key")
        changed, value = controls.input_int(
            "##Frames after this key", int(segments[index])
        )
        if changed:
            poser_mode.set_segment(ctx, index, value)

    changed, closed = controls.checkbox(
        "Loops",
        bool(record.get("closed")),
        tooltip=(
            "A looping clip steps from its last key back to its first, so it "
            "needs one more segment than an open one. Changing this resizes "
            "the timing list to match."
        ),
    )
    if changed:
        poser_mode.set_closed(ctx, closed)

    easing = widgets.labeled_combo(
        "Easing",
        str(record.get("easing") or "linear"),
        [(name, name) for name in svc_clips.EASINGS],
        help_text="How the frames are spaced inside each step.",
    )
    poser_mode.set_easing(ctx, easing)
    if easing == "ease" and max(segments or [0]) < 3:
        # Stated rather than left to be discovered, and stated only for the one
        # easing it is true of. ``ease`` is a smoothstep whose value at a
        # two-frame segment's single interior sample is exactly one half, so it
        # renders identically to ``linear`` there. ``ease_in`` and ``ease_out``
        # sample 0.25 and 0.75 and genuinely differ at that length -- telling an
        # author otherwise sends them away from the option that would have
        # worked.
        widgets.muted_wrapped(
            '"ease" needs at least three frames in a step to do anything; '
            "every step here is shorter than that. Try ease_in or ease_out, "
            "which do act on a two-frame step."
        )

    total = sum(segments)
    widgets.muted(f"{total} frames  -  {len(record.get('keys') or ())} keys")


def _scrubber(ctx: Any, state: Any) -> None:
    widgets.section("Play")
    if state.clips_error:
        # Mid-edit inconsistency rather than a refusal: the clip momentarily
        # does not expand, the scrubber empties, and Save is what actually says
        # no. Coloured as a warning rather than an error for that reason.
        widgets.wrapped(theme.WARN, state.clips_error)
        return
    if not state.frames:
        widgets.muted("This clip has no frames yet.")
        return
    count = len(state.frames)
    current = state.frame if state.frame >= 0 else 0
    # Left beside the slider rather than moved above it (2026-09-08
    # consistency pass): this is the only control under "Play", so the
    # section heading already names it, and the label's own text is the
    # live frame range rather than a fixed field name -- reading it beside
    # the track keeps it next to the value it is describing.
    changed, value = controls.slider_int(
        f"Frame 1-{count}", int(current), 0, count - 1
    )
    if changed:
        poser_mode.scrub(ctx, value)
    if state.frame >= 0:
        widgets.text_colored(
            theme.ACCENT, f"frame {state.frame + 1} of {count} - in-between, not a key"
        )
        if controls.button(
            "Back to key",
            tooltip="Leave the scrubber and re-apply the selected key's own pose.",
        ):
            poser_mode.apply_key(ctx)


def _revert_clips_reason(state: Any, busy: bool) -> str:
    """Why "Revert to shipped clips" is disabled right now, or "" if it is not.

    The 2026-09-08 audit's poser-06: this button's disabled reason was the
    fixed string "These are already the clips the build ships." even when it
    was actually disabled because a save was in flight (``busy`` true while
    ``clips_unsaved``/``edited`` is also true) -- unlike "Save clips" two
    lines above it, which already branches its own reason on ``busy``. A
    greyed control naming the wrong reason it is unavailable is the class of
    bug this codebase's other audits already treat as a defect (see
    ``inker_mode._no_document_reason``, ``clay_ops.reason_for``), which is
    also why this is its own function rather than inline in the draw call:
    a greyed control's reason has to be testable without an imgui frame.
    """
    if (state.clips_unsaved or bool(state.clips.get("edited"))) and not busy:
        return ""
    return "Still saving." if busy else "These are already the clips the build ships."


def _save(ctx: Any, state: Any) -> None:
    imgui.dummy((0, sp(tokens.SP_2)))
    busy = ctx.busy(poser_mode.CLIPS_SAVE_KEY)
    if state.clips_unsaved:
        widgets.text_colored(theme.ACCENT, "unsaved clip changes")
    if widgets.disabled_button(
        "Save clips",
        state.clips_unsaved and not busy,
        (-1, 0),
        reason="Nothing has changed." if not state.clips_unsaved else "Still saving.",
        tooltip=(
            "Write these clips as your own copy. The clips the build ships are "
            "left alone, so an update cannot overwrite your work."
        ),
    ):
        poser_mode.save_clips(ctx)
    if widgets.disabled_button(
        "Revert to shipped clips",
        (state.clips_unsaved or bool(state.clips.get("edited"))) and not busy,
        (-1, 0),
        reason=_revert_clips_reason(state, busy),
        tooltip="Throw away every change to this skeleton's clips. Asks first.",
    ):
        poser_mode.revert_clips(ctx)
