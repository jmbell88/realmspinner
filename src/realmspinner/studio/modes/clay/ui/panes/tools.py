"""Clay's Add palette: every shape you can place, and what the next click makes.

The same shape the raster editor's tool panel takes -- an icon grid, then the
options for whatever is selected rather than every option at once -- for the
same reason: a panel that shows all of them is unreadable, and a rotation snap
means nothing while the select tool is active.

**One grid, one selection, one options block.** Every add-tool writes
``state.generator``, and the options block right below the grid reads that one
field. ``tool_palette.icon_grid`` is the reusable half of the grammar -- the
plain "equal buttons, tooltip-named, one selected" shape -- because Inker's
toolbox and Plotter's tool rail are two more callers for the same idea.

**Every control that changes the document is disabled while a save is in
flight**, exactly as the layers panel is. Serialising reads the live document
on a task thread, so a control that restructured it mid-encode would write a
file describing a document that never existed. Disabling says so on screen
rather than swallowing the click.

**The operations are not here any more** (the 2026-10-02 menu regrouping). This
column used to end in about fifty op buttons generated from the registry in one
flat two-column grid, in the order the tranches landed in. They live in the
header's menu strip now (``strip.py``), grouped by ``menutree`` the way
Blender's Select / Add / Object / Mesh / UV menus group them, and in the
right-click menu, which reads the same table -- so what is left here is what a
sidebar is for: a palette of things to add, which is a list that wants height.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ......kernels.mesh import document as bd
from ......kernels.mesh import ops
from ......kernels.mesh import primitives as bp
from ..... import icons, tokens, tool_palette, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import mode as clay_mode

COLUMNS = 4

TOOL_ICONS = {
    "select": icons.SQUARE_DASHED,
    "move": icons.MOVE,
    "rotate": icons.ROTATE_CW,
    "scale": icons.SCALING,
}

AXES = (("x", "X"), ("y", "Y"), ("z", "Z"))

# The four element modes and the keys that switch them. Held as data beside
# ``TOOL_ICONS`` so the row and ``clay_mode.ELEMENT_KEYS`` are one edit apart
# rather than two files apart.
MODE_BUTTONS = (
    ("object", "Object", "4"),
    ("vertex", "Verts", "1"),
    ("edge", "Edges", "2"),
    ("face", "Faces", "3"),
)


def draw(ctx: Any) -> None:
    """This pane's headings, on tinted blocks.

    The blocks are opened *here* rather than in :func:`layout.pane`, which is
    flat: a pane on a wide canvas wants no tint, and this is one of the four
    narrow sidebars the grouping was written for (see
    ``tests/test_section_blocks.py`` for the report it came from). Wrapping
    ``_body`` rather than inlining the ``with`` keeps every early return inside
    the scope, and the scope closes its last block on the way out.
    """
    with widgets.section_blocks():
        _body(ctx)


def _body(ctx: Any) -> None:
    """What you can *add*, and what that add-tool will place -- and nothing else.

    Most of what this pane held has gone to the viewport header: the tool grid,
    the mode row, snapping and the view aids are settings
    changed between clicks in the viewport, and the operations are menus there.
    What is left is what a sidebar is right for -- lists that want the height
    and are read down rather than flicked between, plus the one block under
    them: which of those entries is the tool in hand right now.
    """

    state = clay_mode.ensure(ctx)
    tab = state.active
    widgets.section("Add")
    manual_render.help_button(ctx, "clay-tools")
    if tab is None:
        widgets.muted("Open or start a document to build in.")
        return

    imgui.begin_disabled(tab.saving)
    _add(ctx, state, tab.doc)
    _options(ctx, state, tab.doc)
    imgui.end_disabled()


def _add(ctx: Any, state: Any, doc: Any) -> None:
    """Every tool that places something, one grid per section.

    Enumerated rather than listed, so a sixteenth shape is a new entry in
    ``primitives.CLAY_GENERATORS`` and no edit here at all -- which is the
    whole reason that registry is data. Each button sets ``state.generator`` to
    the key it placed, so the options block below always names the tool in hand.
    """
    for label, names in sections():
        widgets.field_label(label)
        items = [
            (name, tool_palette.PRIMITIVE_ICONS.get(name, icons.BOX), name.replace("_", " "))
            for name in names
        ]
        clicked = tool_palette.icon_grid(items, COLUMNS, state.generator, id_prefix="add")
        if clicked:
            add_primitive(ctx, doc, clicked)
            state.generator = clicked


def sections() -> list[tuple[str, tuple[str, ...]]]:
    """``primitives.CLAY_GENERATORS``: the shapes Clay offers, by section.

    The table is Clay's own subset of ``GENERATORS`` (which stays whole for
    Mason's saved scenes), so this is not asserted to be a partition of it.
    """
    return [(label, tuple(names)) for label, names in bp.CLAY_GENERATORS]


def _options(ctx: Any, state: Any, doc: Any) -> None:
    """The selected tool's own name, and only its own defaults, beneath it.

    ``state.generator`` already carried this meaning before this pass --
    "what the properties panel offers when the user adds something", by its
    own docstring on ``ClayState`` -- and nothing read it, because the grid
    used to be one click and done; there was no "selected" for it to describe.

    **Read-only, deliberately.** The live, per-type dispatch that turns a
    generator's defaults into editable fields already exists, in
    ``clay_props._generator`` -- it edits a placed object's own params,
    folding every keystroke into that object's one undo step. Rebuilding that
    dispatch a second time here, against numbers that belong to no object yet,
    would be exactly the mistake this file's own docstring names: two lists of
    one thing, free to disagree about a clamp or a type the moment one of them
    changes and the other does not. What is shown here is what the next click
    starts *from*; adjusting a shape's own numbers is a door that already
    exists, once the shape does.
    """
    entry = _options_for(state.generator)
    if entry is None:
        del ctx, doc
        return
    heading, rows, note = entry
    imgui.dummy((0, sp(tokens.SP_2)))
    widgets.section(heading)
    for label, value in rows:
        widgets.field_label(label)
        widgets.muted(value)
    widgets.muted_wrapped(note)
    del ctx, doc


#: Shown under a primitive's defaults. Editing them is the Properties panel's
#: job, once the object exists -- see :func:`_options`'s docstring for why
#: this file does not offer a second door onto the same numbers.
_PRIMITIVE_NOTE = (
    "What the next click places. A shape's own numbers are edited afterwards, "
    "in Properties, once it exists."
)

def _options_for(name: str) -> tuple[str, tuple[tuple[str, str], ...], str] | None:
    """``(heading, rows, note)`` for one add-tool's options, or ``None`` for a
    name that is not one of Clay's shapes -- a remembered tool since removed
    from the palette, say.

    Pure: no imgui, no document. That is what makes "the options shown belong
    to the selected tool and change when the selection changes" a claim a test
    can prove without a GL context, the same way ``clay_header``'s own tables
    are checked.
    """
    entry = bp.GENERATORS.get(name) if name in bp.CLAY_GENERATOR_NAMES else None
    if entry is None:
        return None
    defaults, _build = entry
    rows = tuple(
        (key.replace("_", " "), _format_default(value)) for key, value in defaults.items()
    )
    return name.replace("_", " ").title(), rows, _PRIMITIVE_NOTE


def _format_default(value: Any) -> str:
    """One default value, in the units the generator itself takes."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (tuple, list)):
        return ", ".join(_format_default(v) for v in value)
    return str(value)


def add_primitive(ctx: Any, doc: Any, name: str) -> Any:
    """Place one primitive at its defaults, selected and ready to move.

    The generator name and the parameters it was built with are recorded on the
    object, so the properties panel can offer them and a change regenerates the
    mesh as one step -- until the first element op edits its topology, at which
    point ``clay_ops`` clears the field and the panel switches to counts.

    **Shading is decided here, not by the generator.** ``primitives.py`` always
    hands back a flat mesh (see its own module docstring); this is the door an
    object is placed through, and placing one is what smooths it.
    ``shading.auto_smooth`` runs unconditionally on every shape rather than
    against a membership list of "the organic ones" -- box, plane, pyramid and
    every other faceted primitive already come back flat under the angle rule
    (a capped cylinder's caps meet its band at a right angle, same as a box's
    faces meet each other), so a list here would only be a second, driftable
    statement of what the rule already decides.
    """
    from ......kernels.mesh import shading

    defaults, build = bp.GENERATORS[name]
    obj = bd.Obj(
        uid=bd.new_uid(),
        name=_unique_name(doc, name.replace("_", " ").title().replace(" ", "")),
        mesh=shading.auto_smooth(build(**defaults)),
        generator=name,
        params=dict(defaults),
    )
    doc.add_object(obj)
    # Object mode only (the 2026-09-13 audit's clay-03) -- see
    # ``ClayDoc.add_objects``'s own comment for why selecting unconditionally
    # leaves the Properties panel naming one object while editing another.
    if doc.element_mode == "object":
        doc.select([obj.uid])
    del ctx
    return obj


def _unique_name(doc: Any, base: str) -> str:
    """A name no object in *doc* wears."""
    taken = {obj.name for obj in doc.objects}
    if base not in taken:
        return base
    # The same counting-up rule ``ops.duplicate`` uses, so two objects never
    # wear one name whichever way they arrived.
    return ops.next_name(base, taken)
