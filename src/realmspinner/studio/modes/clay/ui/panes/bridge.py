"""What the document is, and the two ways out of it.

The counts and the save state on top, the pipeline buttons underneath -- the
same shape the raster editor's bridge takes, and for the same reason: a panel
that offers to send something somewhere should first say what it is going to
send.

**The two output paths are genuinely different things, not two encodings of
one.** Export puts the *exact* geometry in the library as an ordinary asset,
which is what a user wants when the shape they modelled is the shape they
meant. Make 3D renders the document flat and hands the picture to trellis,
which reinterprets it -- the blockout becomes a suggestion rather than a
specification, and what comes back is a reconstruction with surface detail
nobody modelled. Choosing between them is the whole point of having both, so
the panel says which is which rather than labelling them "Export" and "Export".

Both buttons are disabled while a save is in flight, for the reason the tool
panel states.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ..... import icons, theme, tokens, verbs, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import mode as clay_mode

#: What this pane refuses to shrink past, in design pixels: the path line, the
#: undo pair, the step count and the two ways out.
#:
#: It had none while the right column was a hand-composed pair. The declarative
#: layout stacks the outliner and the properties above it, and
#: ``layout_skeleton.heights`` gives each share its proportion of the room
#: before the fill sees any -- so two at the default 0.5 leave this pane exactly
#: zero pixels and the Document panel is not on screen at all. Plotter's map
#: file panel went the same way on the same day and for the same reason.
BRIDGE_FLOOR = 170.0


def draw(ctx: Any) -> None:
    state = clay_mode.ensure(ctx)
    tab = state.active
    # "Model file", the shape of every other bridge's heading ("Drawing file",
    # "Map file", "Song file", "Atlas file"); this one said "Document".
    widgets.section("Model file")
    manual_render.help_button(ctx, "clay-bridge")
    if tab is None:
        # The recent list and nothing else -- Plotter's bridge exactly (B5).
        # New/Open are on the empty canvas two columns to the left, and drawing
        # them here as well was one pair of buttons in two places; the *list*
        # is the opposite case, because the moment it matters most is the
        # moment there is nothing open.
        _recent(ctx)
        return
    _files(ctx, tab)

    _facts(tab)
    imgui.dummy((0, sp(tokens.SP_2)))
    _history(ctx, tab)
    imgui.dummy((0, sp(tokens.SP_2)))
    _outputs(ctx, tab)
    imgui.dummy((0, sp(tokens.SP_2)))
    _game_check(ctx, tab)
    _recent(ctx)


def _history(ctx: Any, tab: Any) -> None:
    """Undo and Redo, on screen.

    This mode had a full undo stack and no visible control for it, so the
    feature existed only for a user who already knew Ctrl+Z -- while Inker drew
    the same pair twice. ``clay_mode.undo``/``redo`` rather than
    ``tab.doc.undo()`` here, so the button and the chord carry the same side
    effects (see the history block in that module).
    """
    widgets.history_block(
        ctx,
        tab,
        key="clay",
        undo=lambda: clay_mode.undo(ctx, tab),
        redo=lambda: clay_mode.redo(ctx, tab),
        step=lambda index: clay_mode.step_history(ctx, tab, index),
    )


def _facts(tab: Any) -> None:
    doc = tab.doc
    visible = [obj for obj in doc.objects if obj.visible]
    # Evaluated: this is a count of what will actually export, not of the
    # pre-modifier base mesh -- a mirror or an array changes how many
    # triangles leave the document, and this line is a promise about that.
    triangles = sum(_triangles(doc.evaluated(obj.uid)) for obj in visible)
    widgets.muted(
        f"{len(visible)} of {len(doc.objects)} objects visible  -  "
        f"{triangles:,} triangles  -  {len(doc.materials)} materials"
    )


def _triangles(mesh: Any) -> int:
    """Counted the way the renderer fans them, so the number matches the
    exported file rather than the face count the outliner would give.

    Off array *lengths* alone, the way ``viewport_hints.stats()`` derives its
    own triangle count -- ``corners - 2*faces`` is exactly
    ``sum(max(n-2, 0))`` over every face's corner count ``n``, because
    ``validate`` guarantees every face has at least three corners, so the
    ``max(n-2, 0)`` floor never bites. The 2026-09-18 audit's clay-02 found
    this scanning the whole face array with ``np.diff``/``np.maximum``/``.sum``
    on every draw of the Document panel -- ~8 ms per call for a 1,000,000-face
    object, paid every imgui frame the panel is visible.
    """
    faces = max(0, len(mesh.starts) - 1)
    corners = len(mesh.loops)
    return max(0, corners - 2 * faces)


def _files(ctx: Any, tab: Any) -> None:
    """The file row for an *open* document -- the shared header, so it is the
    same four buttons and the same status ladder every other workspace has.
    ``draw`` returns before this when there is none, because New and Open also
    belong to the empty canvas."""
    widgets.document_header(
        tab,
        new=lambda: clay_mode.new_document(ctx),
        open_=lambda: clay_mode.ask_open(ctx),
        save=lambda: clay_mode.save(ctx, tab),
        save_as=lambda: clay_mode.save_as(ctx, tab),
    )
    imgui.dummy((0, sp(tokens.SP_2)))
    _import_mesh(ctx, tab)
    imgui.dummy((0, sp(tokens.SP_2)))


#: Scale choices for "Import Mesh...", key is the multiplier ``import_file``
#: takes -- a unit a modeller actually authors in, never a bare number a user
#: would have to already know the conversion for.
#: Keys are ``f"{value:g}"`` of the multiplier itself -- what :func:`_import_mesh`
#: formats ``state.import_scale`` as to look the option up, so the two can
#: never drift into two different spellings of the same number.
IMPORT_SCALE_OPTIONS = (
    ("1", "m"),
    ("0.01", "cm"),
    ("0.001", "mm"),
    ("0.0254", "in"),
    ("0.3048", "ft"),
)
IMPORT_UP_OPTIONS = (("y", "Y up"), ("z", "Z up"))


def _import_mesh(ctx: Any, tab: Any) -> None:
    """"Import Mesh...", plus the units/up-axis combo beside it a drop uses too.

    ``ClayState.import_scale``/``import_up`` are what both this button and a
    file dropped on the viewport read (``clay_mode.import_mesh_path``'s own
    defaults) -- set here, remembered for the next import in either form.
    """
    state = clay_mode.ensure(ctx)
    if widgets.disabled_button(
        f"{icons.FOLDER_OPEN} Import Mesh...",
        not tab.saving,
        reason="Saving..." if tab.saving else "",
    ):
        clay_mode.ask_import_mesh(ctx)
    imgui.same_line()
    scale_key = f"{state.import_scale:g}"
    picked = widgets.combo("##clay-import-scale", scale_key, IMPORT_SCALE_OPTIONS, sp(70))
    if picked != scale_key:
        state.import_scale = float(picked)
    imgui.same_line()
    state.import_up = widgets.combo("##clay-import-up", state.import_up, IMPORT_UP_OPTIONS, sp(90))


def _outputs_why(doc: Any, saving: bool) -> str:
    """Why both output buttons below are refused right now, or ``""`` when
    they are not.

    One sentence for both, because they are refused for the same two reasons
    and a user reading two different explanations of one state would look for
    two different problems. The ``_VIEWPORT_WHY`` pattern: a shared gate gets a
    shared sentence.

    Pulled out as its own function by the 2026-09-08 audit's clay-07: the
    "Make 3D" button next to Export received this sentence as its ``reason``,
    but Export itself did not, so it greyed out with no explanation while the
    comment two lines above it said both buttons share one. Extracting it is
    also what lets this be asserted without imgui -- panes cannot be driven
    headlessly, but the sentence a button greys with can still be a plain
    function of a document and a bool.
    """
    if saving:
        return "Saving..."
    if not any(obj.visible for obj in doc.objects):
        return "Nothing visible to send -- every object is hidden."
    return ""


def _outputs(ctx: Any, tab: Any) -> None:
    # The one heading every mode's exits are under. See ``inker_bridge``'s
    # ``_pipeline`` for why the five of them agree on a name.
    widgets.section("Take it somewhere")
    doc = tab.doc
    why = _outputs_why(doc, tab.saving)
    ready = not why

    if widgets.primary_button(
        f"{icons.DOWNLOAD} {verbs.EXPORT_TO_LIBRARY}", enabled=ready, reason=why
    ):
        clay_mode.export_asset(ctx, tab)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "The exact geometry, as an ordinary asset. It picks up rigging, posing, "
            "sprite sheets, the triangle retarget and every mesh export, because all "
            "of those are functions of model.glb."
        )

    # "Make 3D", matching the Mesh stage's own button and Inker's -- wave 5
    # left no "3D" to send anything to.
    if widgets.disabled_button(f"{icons.SEND} Make 3D", ready, reason=why):
        # The App owns the offscreen render: the picture has to be drawn on
        # the frame thread because it needs the GL context, and the bridge is
        # not where that belongs.
        send_to_3d(ctx, tab)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Renders the document flat and hands the picture to trellis, which "
            "reinterprets it: the blockout becomes a suggestion, and what comes back "
            "has surface detail nobody modelled."
        )

    # Two labelled buttons rather than "Export File..." plus a bare "OBJ": the
    # first spelling wrote GLB without saying so, and a format is exactly the
    # thing a user reading the row needs to see before pressing.
    tip = (
        "Saves the document as a plain mesh file on disk, for handing straight "
        "to another tool. The library never sees it."
    )
    if widgets.disabled_button(f"{icons.DOWNLOAD} Export GLB...", ready, reason=why):
        clay_mode.export_mesh_file(ctx, tab, "glb")
    if imgui.is_item_hovered():
        imgui.set_tooltip(tip)
    imgui.same_line()
    if widgets.disabled_button("Export OBJ...", ready, reason=why):
        clay_mode.export_mesh_file(ctx, tab, "obj")
    if imgui.is_item_hovered():
        imgui.set_tooltip(tip + " OBJ writes a .mtl of the same name beside it.")

    if tab.job_id:
        widgets.muted(f"Last exported as {tab.job_id}")


def send_to_3d(ctx: Any, tab: Any) -> None:
    """Hand the document to the App's offscreen render (``_clay_send_to_3d``).

    The indirection through ``ctx`` is the point: the render needs the GL
    context and therefore the frame thread, which is the App's business. A
    headless ctx that never attached the handler gets a clear refusal rather
    than a half-drawn frame.

    The refusal stays -- ``Ctx.clay_send_to_3d`` defaults to None and only the
    App assigns it, so a ctx built without one is a real construction and not a
    hypothetical -- but its wording did not. It said the feature was "not wired
    up yet", which was true of the branch's own first draft and has not been
    true of the app since: a user who saw it went looking for a setting to turn
    on. What the branch actually knows is that *this* window has nothing to
    render from, so that is what it now says.
    """
    handler = getattr(ctx, "clay_send_to_3d", None)
    if handler is None:
        ctx.toast("Could not make a mesh: this window has no viewport to render from.", "error")
        return
    handler(tab)


def _recent(ctx: Any) -> None:
    """The recent list, on the bridge, on **both** branches.

    Plotter's bridge already draws its list whether or not a document is open,
    and that is the answer: a recent list is how you get *back* to work, so the
    one moment it matters most is the moment there is nothing open. Clay's was
    on the empty canvas instead, which is the one screen it disappears from as
    soon as it becomes useful again.
    """
    from pathlib import Path

    widgets.recent_files(
        clay_mode.recent_paths(ctx),
        lambda path: clay_mode.open_path(ctx, Path(path)),
    )


# --- Game check ---------------------------------------------------------------
#
# ``readiness.validate`` is the one function the panel, an agent tool and a
# future warning badge all read (see that module's own docstring) -- what
# belongs here is only the on-demand trigger, the "out of date" staleness
# read and turning a ``Report`` into rows a "Fix" button can act on.

_STATUS_COLOR: dict[str, str] = {
    "pass": "OK",
    "warn": "ACCENT",
    "fail": "ERR",
    "skip": "MUTED",
}
_STATUS_ICON: dict[str, str] = {
    "pass": icons.CHECK,
    "warn": icons.TRIANGLE_ALERT,
    "fail": icons.X,
    "skip": icons.CIRCLE_ALERT,
}


def validator_rows(report: Any) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
    """A ``readiness.Report`` as plain ``(status, label, message, fix, uids)``
    row tuples -- the whole of what the section below draws, pulled into a
    function that takes no imgui so "does this Report produce a Fix button
    only where a check actually names one" is a plain assertion rather than a
    screenshot.
    """
    return [
        (check.status, check.label, check.message, check.fix, check.uids)
        for check in report.checks
    ]


def _run_fix(ctx: Any, tab: Any, fix: str, uids: tuple[int, ...]) -> None:
    """A row's "Fix" button: select what the check named (object mode, if it
    named anything), then run the op at its declared defaults.

    Defaults rather than the op's own param popup: reaching that popup from
    here would need the menu's own open-a-dialog machinery
    (``ClayState.pending_op``/``open_op_popup``), which is a keyboard/menu
    affordance this row is not one of. A user who wants non-default numbers
    still has the op's own row in the tools pane or the context menu, unaffected
    by this shortcut existing.
    """
    from ... import ops as clay_ops

    doc = tab.doc
    if uids:
        doc.set_element_mode("object")
        doc.select([uid for uid in uids if any(o.uid == uid for o in doc.objects)])
    # The 2026-09-22 audit's clay-05: with an empty ``uids`` (a check that ran
    # against nothing selectable) this used to fall through and run the op on
    # whatever was already selected -- the wrong object, or nothing at all
    # with no toast either way. ``clay_ops.run`` returns ``False`` when it
    # refused or had nothing to do; say so rather than pretending the press
    # did something.
    if not clay_ops.run(ctx, doc, clay_ops.get(fix)):
        ctx.toast("Nothing to fix.", "warn")


def _game_check(ctx: Any, tab: Any) -> None:
    """"Game check": a profile combo, a Check button, and one row per check."""
    from ......kernels.mesh import readiness

    if not widgets.header("Game check", default_open=False, persist_key="clay-game-check"):
        return

    profile = tab.readiness_profile or readiness.DEFAULT_PROFILE
    options = [(key, prof.label) for key, prof in readiness.PROFILES.items()]
    tab.readiness_profile = widgets.combo("##clay-readiness-profile", profile, options, sp(170))

    imgui.same_line()
    why = "Saving..." if tab.saving else ""
    if widgets.disabled_button("Check", not tab.saving, reason=why):
        clay_mode.check_readiness(ctx, tab, tab.readiness_profile)

    report = tab.readiness_report
    if report is None or report.profile != tab.readiness_profile:
        widgets.muted("Not checked against this profile yet -- press Check.")
        return
    if tab.readiness_head != tab.doc.history.head:
        widgets.muted("Out of date: the document has changed since this check ran.")

    for status, label, message, fix, uids in validator_rows(report):
        colour = getattr(theme, _STATUS_COLOR.get(status, "MUTED"))
        widgets.text_colored(colour, f"{_STATUS_ICON.get(status, '?')} {label}")
        imgui.same_line()
        widgets.muted_wrapped(message)
        if fix and widgets.disabled_button(f"Fix##{label}", not tab.saving, reason=why):
            _run_fix(ctx, tab, fix, uids)
