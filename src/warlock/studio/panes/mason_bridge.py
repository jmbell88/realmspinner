"""What the scene is, and the two ways out of it.

``clay_bridge``'s shape: the counts and the save state on top, the two export
buttons underneath -- a panel that offers to send something somewhere should
say what it is going to send first.

**The two exports are genuinely different files, not two encodings of one.**
GLB is the whole scene as one node graph an engine reads directly; OBJ is a
static mesh dump with no hierarchy, for a pipeline step that only wants
triangles. ``mason_mode.export_glb``/``export_obj`` are separate calls, not
one call with a format argument, because deciding *which* is a decision the
button labels have to carry rather than hide behind a shared verb.

**The scene-size warning is read off ``mason.scene.PLACED_WARN_THRESHOLD``,
never a literal typed here** -- that constant is the module's own measured
number for where ``resolve()`` stops fitting a 60 Hz frame, and a copy of it
in this file would be free to drift the day the measurement changes and this
file is not touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from .. import icons, mason_mode, tokens, widgets
from ..manual import render as manual_render
from ..tokens import sp

#: What this pane refuses to shrink past -- ``clay_bridge.BRIDGE_FLOOR``'s own
#: reason: the declarative layout gives every SHARE slot its proportion of the
#: room before the FILL slot sees any, so with no floor a tall outliner and
#: properties column can squeeze this one to nothing.
BRIDGE_FLOOR = 170.0


def draw(ctx: Any) -> None:
    state = mason_mode.ensure(ctx)
    tab = mason_mode.active(ctx)
    widgets.section("Scene file")
    manual_render.help_button(ctx, "mason-bridge")
    if tab is None:
        _recent(ctx)
        return
    _files(ctx, tab)
    _facts(ctx, state, tab)
    imgui.dummy((0, sp(tokens.SP_2)))
    _history(ctx, tab)
    imgui.dummy((0, sp(tokens.SP_2)))
    _missing(tab)
    imgui.dummy((0, sp(tokens.SP_2)))
    _outputs(ctx, tab)
    _recent(ctx)


def _files(ctx: Any, tab: Any) -> None:
    widgets.document_header(
        tab,
        new=lambda: mason_mode.new_document(ctx),
        open_=lambda: mason_mode.ask_open(ctx),
        save=lambda: mason_mode.save(ctx, tab),
        save_as=lambda: mason_mode.save_as(ctx, tab),
    )
    imgui.dummy((0, sp(tokens.SP_2)))


def _facts(ctx: Any, state: Any, tab: Any) -> None:
    stats = mason_mode.scene_stats(ctx, tab)
    placed = stats["placed"]
    widgets.muted(f"{placed:,} placed  -  {len(tab.doc.materials)} materials")
    if stats["warn"]:
        # ``stats["threshold"]`` is ``scene.PLACED_WARN_THRESHOLD``, read back
        # through ``mason_mode.scene_stats`` rather than imported a second
        # time here -- one call answers both "is this scene large" and "large
        # compared to what", so the sentence and the number it names cannot
        # disagree about which constant they mean.
        widgets.secondary(
            f"This scene has more than {stats['threshold']:,} placed items and "
            "may cost frame rate to edit."
        )
    del state


def _missing(tab: Any) -> None:
    missing = tab.doc.missing_refs()
    if not missing:
        return
    widgets.field_label("missing references")
    for node, ref in missing:
        widgets.secondary(f"{icons.TRIANGLE_ALERT} {node.name or node.uid}: {ref}")


def _history(ctx: Any, tab: Any) -> None:
    widgets.history_block(
        ctx,
        tab,
        key="mason",
        undo=lambda: mason_mode.undo(ctx, tab),
        redo=lambda: mason_mode.redo(ctx, tab),
        step=lambda index: mason_mode.step_history(ctx, tab, index),
    )


def _outputs_why(tab: Any) -> str:
    if tab.saving:
        return "Saving..."
    if not tab.doc.roots:
        return "Nothing placed to send -- the scene is empty."
    return ""


def _outputs(ctx: Any, tab: Any) -> None:
    widgets.section("Take it somewhere")
    why = _outputs_why(tab)
    ready = not why
    width = widgets.grid_width(2)
    if widgets.disabled_button(f"{icons.DOWNLOAD} Export GLB", ready, (width, 0), reason=why):
        mason_mode.export_glb(ctx, tab)
    if imgui.is_item_hovered():
        imgui.set_tooltip("The whole scene, hierarchy and all -- what an engine reads directly.")
    imgui.same_line()
    if widgets.disabled_button(f"{icons.DOWNLOAD} Export OBJ", ready, (width, 0), reason=why):
        mason_mode.export_obj(ctx, tab)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "A static mesh dump, no hierarchy -- for a pipeline step that only wants triangles."
        )
    if tab.job_id:
        widgets.muted(f"Last exported as {tab.job_id}")


def _recent(ctx: Any) -> None:
    widgets.recent_files(
        mason_mode.recent_paths(ctx),
        lambda path: mason_mode.open_path(ctx, Path(path)),
    )
