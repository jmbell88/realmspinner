"""Mason's Assets pane: what the next click (or the next drag) puts into the scene.

Named ``mason_palette`` and deliberately **not** ``mason_assets`` -- that name
already belongs to :mod:`~.mason_assets`, the host's
:class:`~.mason_assets.AssetSource`, the cache a placed :class:`~.mason.refs.Ref`
resolves through. Two modules one import apart sharing a name is how a reader
comes to believe this pane is holding that cache; it is not, it only *arms*
what the next placement will point one at.

The same three-group shape ``clay_tools`` settled on for "pick a thing to
add" -- library rows, then the derived primitive grid, then lights and a
camera -- because a scene editor's palette is the modeller's palette plus two
more kinds of thing a scene can hold. Only the library rows place
*immediately*, on click: a library mesh already has a size and an origin, so
there is nothing left for a viewport click to decide, and the row also lifts
for a drag onto a specific point (``panes.library.draggable_mesh``, which
this pane's own owner also built -- see that function's docstring for why
Mason gets a second drag payload rather than a widened one). Everything else
-- a primitive, a light, a camera -- has no size or position of its own until
the user says where, so a click here only **arms** ``state.place_kind`` and
the viewport's own click handler is what actually calls
``mason_mode.place_primitive``/``place_light``/``place_camera``.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, icons, mason_mode, tool_palette, widgets
from ..clay import primitives as bp
from ..manual import render as manual_render
from ..tokens import sp
from . import library, thumbs
from .clay_tools import PRIMITIVE_ICONS

COLUMNS = 4

#: ``PRIMITIVE_ICONS`` is ``clay_tools``'s own table, imported rather than
#: copied -- it is pinned bijective against ``primitives.GENERATORS`` by
#: ``tests/test_clay_wiring.py`` and this pane reads it only for its glyphs,
#: never for membership: a primitive this table has not caught up with still
#: gets ``icons.BOX`` at the draw site below, so a sixteenth generator gets a
#: button on the day it is written rather than on the day someone remembers
#: this file.

_LIGHT_KINDS = (
    ("light:point", "Point", "A bulb: falls off with distance in every direction"),
    ("light:spot", "Spot", "A cone, narrowed by its inner and outer angles"),
    (
        "light:directional",
        "Directional",
        "Parallel rays with no falloff, the way sunlight is modelled",
    ),
)


def draw(ctx: Any) -> None:
    """This pane's headings, on tinted blocks -- the shape every sidebar in
    this app opens with; see ``clay_tools.draw`` for why it is opened here
    rather than in ``layout.pane``."""
    with widgets.section_blocks():
        _body(ctx)


def _body(ctx: Any) -> None:
    state = mason_mode.ensure(ctx)
    tab = mason_mode.active(ctx)
    widgets.section("Assets")
    manual_render.help_button(ctx, "mason-assets")
    if tab is None:
        widgets.muted("Open or start a scene to place things in.")
        return

    imgui.begin_disabled(tab.saving)
    _library(ctx, state)
    imgui.dummy((0, sp(8)))
    _primitives(ctx, state)
    imgui.dummy((0, sp(8)))
    _lights(ctx, state)
    imgui.dummy((0, sp(8)))
    _camera(ctx, state)
    imgui.end_disabled()


def _mesh_jobs(ctx: Any) -> list[dict[str, Any]]:
    """Every finished mesh a row can offer, newest first.

    Read straight off the job cache rather than the filesystem --
    ``poser_mode.riggable_assets``'s own shape, one dimension over: this pane
    is drawn every frame Mason's Assets tab is open, so a disk walk here would
    be a disk walk every frame.
    """
    return [job for job in reversed(ctx.cache.jobs) if library.can_drag_mesh(job)]


def _library(ctx: Any, state: Any) -> None:
    widgets.field_label("library")
    jobs = _mesh_jobs(ctx)
    if not jobs:
        widgets.muted("No finished models yet -- build one in Create or Clay.")
        return
    for job in jobs:
        imgui.push_id(job["id"])
        thumbs.job_thumb(ctx, job, sp(32.0))
        imgui.same_line()
        label = job.get("name") or job.get("prompt") or job["id"]
        # ``controls.selectable`` and never ``imgui.selectable``: every
        # interactive widget in this app goes through the one chokepoint
        # ``controls._finish_item``, which is what ``studio/probe.py`` taps to
        # census a mode's controls. A raw imgui call draws the same row and is
        # invisible to that census -- so ``exercise_mode`` would press
        # everything in this pane except the library rows and report full
        # coverage, which is the blind spot ``test_probe`` pins a count
        # against precisely so it cannot grow quietly.
        if controls.selectable(f"{label}##masonlib", False)[0]:
            mason_mode.place_job(ctx, job)
        # The pane's own drag source, over the same row the click above
        # places from -- ``library.draggable_mesh`` is the predicate and the
        # payload this row and the viewport's drop target both have to agree
        # on; that agreement lives in ``panes/library.py``, not here.
        library.draggable_mesh(ctx, job)
        imgui.pop_id()
    del state


def _primitives(ctx: Any, state: Any) -> None:
    """The derived primitive grid -- ``clay_tools._add``'s grid, restated for
    an armed tool rather than an immediate placement."""
    for label, names in _sections():
        widgets.field_label(label)
        items = [
            (name, PRIMITIVE_ICONS.get(name, icons.BOX), name.replace("_", " "))
            for name in names
        ]
        clicked = tool_palette.icon_grid(items, COLUMNS, state.place_kind, id_prefix="masonadd")
        if clicked:
            state.place_kind = clicked
    del ctx


def _sections() -> list[tuple[str, tuple[str, ...]]]:
    """``primitives.CATEGORIES``, plus anything the table forgot --
    ``clay_tools._sections``, verbatim, for the identical reason: the table is
    a partition of ``GENERATORS`` and this is the fallback for the day it
    stops being one."""
    out = [(label, tuple(names)) for label, names in bp.CATEGORIES]
    filed = {name for _, names in bp.CATEGORIES for name in names}
    rest = tuple(sorted(set(bp.GENERATORS) - filed))
    if rest:
        out.append(("other", rest))
    return out


def _lights(ctx: Any, state: Any) -> None:
    widgets.field_label("lights")
    width = widgets.grid_width(1)
    for key, label, tip in _LIGHT_KINDS:
        if controls.button(
            f"{label}##masonlight{key}", (width, sp(28)), selected=state.place_kind == key,
            tooltip=tip,
        ):
            state.place_kind = key
    del ctx


def _camera(ctx: Any, state: Any) -> None:
    widgets.field_label("camera")
    width = widgets.grid_width(1)
    if controls.button(
        "Camera##masoncamera",
        (width, sp(28)),
        selected=state.place_kind == "camera",
        tooltip="A view into the scene, exported alongside it",
    ):
        state.place_kind = "camera"
    del ctx
