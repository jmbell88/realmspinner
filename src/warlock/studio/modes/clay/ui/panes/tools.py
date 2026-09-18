"""Clay's tool column: what to add, how to transform it, and snapping.

The same shape the raster editor's tool panel takes -- an icon grid, then the
options for whatever is selected rather than every option at once -- for the
same reason: a panel that shows all of them is unreadable, and a rotation snap
means nothing while the select tool is active.

**One grid, one selection, one options block (the 2026-09-08 panel-grammar
pass).** This file used to draw three different affordances for "pick a
thing": primitives as an unlabelled icon grid with the name only in a
tooltip, figures as one full-width text button per row, and the ops as a
ragged two-column grid whose gaps were uneven because every button auto-sized
to its own label. Every add-tool -- primitive or figure -- now writes
``state.generator``, the same field regardless of which row placed it, and
the options block right below the grid reads that one field. Figures still
draw as labelled rows rather than icons; see :func:`_figures` for why that
half of the old inconsistency stays rather than being papered over with a
misleading glyph. ``tool_palette.icon_grid`` is the reusable half of the new
grammar -- the plain "equal buttons, tooltip-named, one selected" shape -- and
it is a new module rather than code grown in this file again, because Inker's
toolbox and Plotter's tool rail are two more callers for the same idea.

**Every control that changes the document is disabled while a save is in
flight**, exactly as the layers panel is. Serialising reads the live document
on a task thread, so a control that restructured it mid-encode would write a
file describing a document that never existed. Disabling says so on screen
rather than swallowing the click.

**The action buttons come from the ops registry**, not from a list here. There
were three lists of what Clay can do -- this pane, the key handler and now the
context menu -- and this is the one that stopped being one. Duplicate, Bake,
Mirror and Delete still look exactly as they did; they are just rows of
``clay_ops.menu(mode)`` now, so a button cannot offer an op the menu greys out.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ......kernels.mesh import document as bd
from ......kernels.mesh import ops
from ......kernels.mesh import primitives as bp
from ..... import controls, icons, tokens, tool_palette, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import mode as clay_mode
from ... import ops as clay_ops

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
    """What you can *add*, what that add-tool will place, and what you can
    *do* -- and nothing else.

    Half of what this pane held has gone to the viewport header: the tool grid,
    the mode row, snapping, proportional editing and the view aids. Every one of
    them is a setting changed between clicks in the viewport, and this sidebar
    is on the far side of the window from it.

    What is left is what a sidebar is right for -- lists that want the height
    and are read down rather than flicked between, plus the one field between
    them: which of those list entries is the tool in hand right now.
    """

    state = clay_mode.ensure(ctx)
    tab = state.active
    widgets.section("Tools")
    manual_render.help_button(ctx, "clay-tools")
    if tab is None:
        widgets.muted("Open or start a document to build in.")
        return

    from . import menu as clay_menu

    imgui.begin_disabled(tab.saving)
    _add(ctx, state, tab.doc)
    _options(ctx, state, tab.doc)
    imgui.dummy((0, sp(tokens.SP_2)))
    _actions(ctx, state, tab.doc)
    imgui.end_disabled()
    # *Outside* the disabled block: the popup greys its own Apply against
    # tab.saving and its Cancel must stay live, or a save that starts while it
    # is open leaves a modal the user cannot dismiss -- the exact trap
    # inker_bridge documents.
    clay_menu.params_popup(ctx, state, tab)


def _add(ctx: Any, state: Any, doc: Any) -> None:
    """Every tool that places something, in one grid plus one row-list.

    Enumerated rather than listed, so a seventh primitive is a new entry in
    ``primitives.GENERATORS`` and no edit here at all -- which is the whole
    reason that registry is data. Each button sets ``state.generator`` to the
    key it placed, exactly as :func:`_figures`'s rows do below, so the two
    groups share one idea of "the tool in hand" even though only one of them
    can show it as a selected icon.
    """
    for label, names in _sections():
        widgets.field_label(label)
        items = [
            (name, tool_palette.PRIMITIVE_ICONS.get(name, icons.BOX), name.replace("_", " "))
            for name in names
        ]
        clicked = tool_palette.icon_grid(items, COLUMNS, state.generator, id_prefix="add")
        if clicked:
            add_primitive(ctx, doc, clicked)
            state.generator = clicked
    _figures(ctx, state, doc)


def _sections() -> list[tuple[str, tuple[str, ...]]]:
    """``primitives.CATEGORIES``, plus anything the table forgot.

    The table is asserted to be a partition of ``GENERATORS``, so the trailing
    section is empty in every shipped build and the test that says so is an
    exact equality. It is drawn anyway for ``tool_palette.PRIMITIVE_ICONS.get``'s reason: a
    thirteenth generator added and not filed still gets a button on the day it
    is written rather than on the day someone remembers this file, which is the
    property that makes ``_add`` generated from data rather than a third list
    of what Clay can build.
    """
    out = [(label, tuple(names)) for label, names in bp.CATEGORIES]
    filed = {name for _, names in bp.CATEGORIES for name in names}
    rest = tuple(sorted(set(bp.GENERATORS) - filed))
    if rest:
        out.append(("other", rest))
    return out


def _figures(ctx: Any, state: Any, doc: Any) -> None:
    """The assembly presets, as labelled full-width rows -- deliberately not
    icons, even though the grid above is now the house shape for "pick a
    thing to add".

    Only one of the eight templates has a defensible glyph in the pinned
    lucide subset (``icons.py`` transcribes lucide-static 0.525.0 and its own
    docstring forbids guessing a codepoint): ``humanoid`` could honestly wear
    ``PERSON_STANDING``, and at a stretch so could ``biped_tail`` -- the same
    silhouette, plus a tail nothing in the set draws. The other six --
    quadruped, bird, serpent, insect, fish, blob -- have nothing that reads as
    them rather than as something else; the manual already said as much
    ("a humanoid and a blob look the same at sixteen pixels"). Icon-ing two of
    eight and leaving six as text would recreate the exact inconsistency this
    pass exists to remove, one level down, so the group stays uniform. The
    labels are also doing real work a tooltip could not: "Serpent (limbless
    chain)" is most of what a user needs to know before clicking it, and a
    16px glyph has no room for the parenthetical.

    Still one *selection* with the grid above, if not one affordance: a press
    here calls :func:`add_assembly` and sets ``state.generator`` exactly as a
    primitive button does, so the options block below always names whichever
    was placed last, from either row.
    """
    from ......kernels.mesh import presets

    if not presets.ASSEMBLIES:
        return
    widgets.field_label("figures")
    width = widgets.grid_width(1)
    for key, (label, _build) in presets.ASSEMBLIES.items():
        if controls.button(
            f"{label}##figure{key}", (width, sp(28)), selected=state.generator == key
        ):
            add_assembly(ctx, doc, key)
            state.generator = key
    imgui.new_line()


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

#: Shown under a figure's entry. Figures have no per-field defaults of their
#: own to preview -- ``presets.build`` computes a whole rig template's worth
#: of parts -- so this says what a figure *is* instead.
_FIGURE_NOTE = (
    "A preset arrangement of primitives, placed as one undo step. Each part "
    "opens in Properties like any other object once it is down."
)


def _options_for(name: str) -> tuple[str, tuple[tuple[str, str], ...], str] | None:
    """``(heading, rows, note)`` for one add-tool's options, or ``None`` for a
    name that names neither a generator nor a figure -- a document opened from
    an older save whose remembered tool was since removed from the registry,
    say.

    Pure: no imgui, no document. That is what makes "the options shown belong
    to the selected tool and change when the selection changes" a claim a test
    can prove without a GL context, the same way ``clay_header``'s own tables
    are checked.
    """
    entry = bp.GENERATORS.get(name)
    if entry is not None:
        defaults, _build = entry
        rows = tuple(
            (key.replace("_", " "), _format_default(value)) for key, value in defaults.items()
        )
        return name.replace("_", " ").title(), rows, _PRIMITIVE_NOTE
    from ......kernels.mesh import presets

    figure = presets.ASSEMBLIES.get(name)
    if figure is not None:
        label, _build = figure
        return label, (), _FIGURE_NOTE
    return None


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
    hands back a flat mesh (see its own module docstring); this is one of the
    two doors an object is placed through, and the 2026-09-06 audit's
    organic-shapes decision was that placing one is what smooths it.
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


def add_assembly(ctx: Any, doc: Any, key: str) -> list[Any]:
    """Place every part of a figure preset, as one undo step.

    Each part is an ordinary generated object -- same generator name, same
    recorded params -- so the properties panel offers a leg's radius the way it
    offers a lone cylinder's, and nothing about a figure is a special kind of
    document. What the preset adds is where the parts sit and what they are
    called; the parts themselves are the primitives that were already there.

    Through :func:`_unique_name` for the reason ``add_primitive`` is: two
    humanoids in one document must not have two objects called ``Head``, and
    the count is taken against the document *as it grows*, which is why the
    objects are appended to a list here and handed over in one call rather
    than named up front.

    Goes through ``presets.build`` rather than calling ``ASSEMBLIES[key][1]``
    itself, so the grounding rule from the 2026-09-06 audit's clay-08 finding
    (terrestrial figures sit on the ground, the two swimmers keep their
    authored placement) applies here without this pane knowing which key is
    which.

    **The other insertion door.** Every part gets ``shading.auto_smooth`` the
    same way ``add_primitive``'s lone shape does -- unconditionally, by the
    same rule, rather than a per-generator or per-part list of which parts are
    "organic": every figure part is a capsule, a sphere, an icosphere or a box
    (see ``presets.py``), so the rule alone gives a humanoid's boxy hands and
    feet hard edges with nothing here needing to know which parts those are.
    A capsule limb's *own* result is coarser than that: ``presets.py``'s
    ``LIMB_SEGMENTS``/``LIMB_RINGS`` put a limb's mesh right at the angle
    rule's threshold, so a limb comes back mostly rather than fully smooth --
    measured in ``tests/modes/clay/test_shading.py``, and a figure-proportions
    question this change is scoped out of touching.
    """
    from ......kernels.mesh import presets, shading

    label, _builder = presets.ASSEMBLIES[key]
    objs: list[Any] = []
    taken: set[str] = set()
    for part in presets.build(key):
        defaults, make = bp.GENERATORS[part.generator]
        params = {**defaults, **part.params}
        name = _unique_name(doc, part.name, taken)
        taken.add(name)
        objs.append(
            bd.Obj(
                uid=bd.new_uid(),
                name=name,
                mesh=shading.auto_smooth(make(**params)),
                generator=part.generator,
                params=dict(params),
                translation=list(part.translation),
                rotation=list(part.rotation),
                scale=list(part.scale),
            )
        )
    doc.add_objects(objs, label)
    del ctx
    return objs


def _unique_name(doc: Any, base: str, also: set[str] | None = None) -> str:
    """A name no object in *doc* wears.

    ``also`` is for the several objects an assembly places in one call: they
    are not in ``doc.objects`` yet, so counting against the document alone
    would hand every leg of a second humanoid the name the first leg already
    has. The caller adds each name it takes.
    """
    taken = {obj.name for obj in doc.objects} | (also or set())
    if base not in taken:
        return base
    # The same counting-up rule ``ops.duplicate`` uses, so two objects never
    # wear one name whichever way they arrived.
    return ops.next_name(base, taken)


def _actions(ctx: Any, state: Any, doc: Any) -> None:
    """One button per registry op that applies in the current mode.

    Two even columns, at ``widgets.grid_width(2)`` -- the one width rule the
    whole grid asks for, rather than each button auto-sizing to its own label
    the way it used to (that is what made the pairs ragged: "Duplicate" and
    "Smooth" are not the same number of characters, so neither were their
    buttons). ``grid_width`` is asked fresh, not assumed, for the reason its
    own docstring gives -- an unscaled gap literal was right at UI scale 1.0
    and short by 4.8px per gap at 1.5x, which is the exact incident that cost
    this row its fourth button before.

    Delete is drawn last and full width, in its own destructive styling,
    through ``widgets.destructive_button`` directly now rather than through a
    local reimplementation. The ``reason`` keyword that reimplementation
    existed to add is on ``destructive_button`` itself as of the 2026-09-08
    button-vocabulary pass.
    """
    widgets.field_label("actions")
    del state
    ops_here = [
        op
        for op in clay_ops.menu(doc.element_mode)
        if not op.name.startswith("select-") and op.name != "delete"
    ]
    # Two columns is the *shape*; the width is what the longest label needs.
    # ``grid_width(2)`` alone drew "Smooth (Catmull-Clark)" without its
    # closing bracket -- imgui renders a label straight past its frame and
    # the child clips it, so an even grid that is one character too narrow
    # loses the end of a word rather than looking tight.
    labels = [op.label.rstrip(".") for op in ops_here]
    columns = widgets.grid_columns_for(labels, maximum=2)
    width = widgets.grid_width(columns)
    for index, op in enumerate(ops_here):
        enabled = op.enabled(doc)
        label = op.label.rstrip(".")
        # clay-07 (2026-09-06 audit): a greyed row here used to say nothing
        # about why -- ``op.hint`` describes what the op does, not why it is
        # currently refused, and it is the only sentence a disabled action
        # used to carry. ``reason_for`` is derived from the same ``enabled``
        # predicate this row already greys on, so it cannot drift from it.
        if widgets.disabled_button(
            f"{label}##clayop{op.name}", enabled, (width, 0), reason=clay_ops.reason_for(op, doc)
        ):
            _invoke(ctx, doc, op)
        # The key *and* the sentence. ``Op.hint`` was written for the dialog a
        # parameterised op opens, which means the explanation of what an op is
        # for was reachable only by pressing the button -- and the two ops it
        # most has to tell apart sit side by side here.
        tip = "\n".join(part for part in (op.key, op.hint) if part)
        if imgui.is_item_hovered() and tip:
            imgui.set_tooltip(tip)
        # ``same_line`` on every button but the last of a row, and *nothing*
        # on the last: the next item wraps by itself. The ``new_line`` calls
        # that stood here added a whole blank row between every pair, which is
        # what put the visible ladder of gaps down this grid.
        #
        # ``index + 1 < len(ops_here)`` is the half that is not cosmetic. Delete
        # is drawn after this loop, and on an odd count the final op leaves its
        # row open -- so without this the destructive button landed *beside* an
        # ordinary one, which is the arrangement the comment below has always
        # existed to prevent.
        if (index + 1) % columns and index + 1 < len(ops_here):
            imgui.same_line()

    delete = next(
        (op for op in clay_ops.menu(doc.element_mode) if op.name == "delete"), None
    )
    if delete is None:
        return
    enabled = delete.enabled(doc)
    label = delete.label.rstrip(".")
    # Greyed like every other row here. Checking ``enabled`` *after* the click
    # drew a live red button that did nothing -- and this is the one button
    # where "nothing happened" is hardest to tell apart from "something
    # irreversible happened".
    if widgets.destructive_button(
        f"{icons.TRASH} {label}",
        (widgets.grid_width(1), 0),
        enabled=enabled,
        reason=clay_ops.reason_for(delete, doc),
    ):
        clay_ops.run(ctx, doc, delete)


def _invoke(ctx: Any, doc: Any, op: Any) -> None:
    """Run an op, or hand a parameterised one to the popup the menu also uses."""
    from . import menu as clay_menu

    if not op.params:
        clay_ops.run(ctx, doc, op)
        return
    state = clay_mode.ensure(ctx)
    state.pending_op = op.name
    state.op_params.setdefault(op.name, clay_ops.defaults_for(op))
    imgui.open_popup(clay_menu.PARAM_POPUP)
