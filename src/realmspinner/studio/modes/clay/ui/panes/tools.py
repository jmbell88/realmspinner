"""Clay's shapes: what can be placed, what it is called, and the flyout that lists it.

This was the left sidebar's Add pane, an icon grid with a read-only "what the
next click places" block under it. The sidebar is gone (``skeletons.clay``'s
left column is empty now) and the grid with it: the slim tool rail
(``rail.py``) keeps four quick shapes and a ``+`` that opens the flyout drawn
here, which lists **every** shape with its name beside its glyph. An unlabelled
icon grid of fifteen near-identical silhouettes made the user hover each one to
learn its name; a labelled list does not.

**What stays in this module is data and doors**, because other places reach
for them: :func:`sections` (the shape registry, grouped), :func:`add_primitive`
(the one door an object is placed through -- the rail, the Add menu, the
empty-state button and the agent all call it), :func:`display_name` (the one
spelling of a shape's name, so ``uv_sphere`` is "UV Sphere" in the rail, the
Add menu and the flyout alike), and the tables the header reads.

**Every control that changes the document is disabled while a save is in
flight**, as it is everywhere in Clay. Serialising reads the live document on a
task thread, so a control that restructured it mid-encode would write a file
describing a document that never existed. Disabling says so on screen rather
than swallowing the click.

**The operations are not here** (the 2026-10-02 menu regrouping). They live in
the header's menu strip (``strip.py``), grouped by ``menutree``, and in the
right-click menu, which reads the same table.
"""

from __future__ import annotations

from typing import Any

from ......kernels.mesh import document as bd
from ......kernels.mesh import ops
from ......kernels.mesh import primitives as bp
from ..... import controls, icons, tool_palette, widgets
from .....manual import render as manual_render

TOOL_ICONS = {
    "select": icons.SQUARE_DASHED,
    "move": icons.MOVE,
    "rotate": icons.ROTATE_CW,
    "scale": icons.SCALING,
}

AXES = (("x", "X"), ("y", "Y"), ("z", "Z"))

# The four element modes and the keys that switch them, **in the order of the
# keys** (1 2 3 4 = vertex, edge, face, object), which is the order the header's
# pill draws them in. The pill used to lead with Object, so the first segment
# was the key that is pressed last, and a reader counting segments left to
# right got "4 1 2 3". Held as data beside ``TOOL_ICONS`` so the row and
# ``clay_mode.ELEMENT_KEYS`` are one edit apart rather than two files apart.
MODE_BUTTONS = (
    ("vertex", "Verts", "1"),
    ("edge", "Edges", "2"),
    ("face", "Faces", "3"),
    ("object", "Object", "4"),
)

#: The shapes the rail keeps one click away; the rest are behind its ``+``.
#: Names, not glyphs -- the glyph and the label both come from the same tables
#: the flyout reads, so a quick shape cannot drift from its flyout entry.
QUICK_SHAPES = ("box", "cylinder", "uv_sphere", "cone")

#: Words a shape's key spells that ``str.capitalize`` would get wrong. A
#: "uv_sphere" is a UV sphere (latitude and longitude, as opposed to the
#: icosphere beside it), and "Uv Sphere" reads as a typo.
_ACRONYMS = frozenset({"uv"})


def display_name(name: str) -> str:
    """A shape's key as the words a person reads: ``uv_sphere`` -> ``UV Sphere``.

    One function for the rail, the Add menu, the flyout and the tooltips, which
    each spelled it their own way (``name.replace("_", " ")`` in lower case in
    one place, ``.title()`` in another) and so disagreed about the same sphere.
    """
    return " ".join(
        word.upper() if word in _ACRONYMS else word.capitalize() for word in name.split("_")
    )


def sections() -> list[tuple[str, tuple[str, ...]]]:
    """``primitives.CLAY_GENERATORS``: the shapes Clay offers, by section.

    The table is Clay's own subset of ``GENERATORS`` (which stays whole for
    Mason's saved scenes), so this is not asserted to be a partition of it.
    """
    return [(label, tuple(names)) for label, names in bp.CLAY_GENERATORS]


def draw_add_menu(ctx: Any, state: Any, tab: Any) -> None:
    """The flyout's body: every shape Clay offers, grouped, name beside glyph.

    Enumerated rather than listed, so a sixteenth shape is a new entry in
    ``primitives.CLAY_GENERATORS`` and no edit here at all -- which is the
    whole reason that registry is data. Each entry sets ``state.generator`` to
    the key it placed, the field the empty state reads.

    Drawn inside the rail's popup (``rail.py`` owns the open and close), with
    the heading's (?) here so the manual's "Adding a primitive" section stays
    one click from the list it describes.
    """
    widgets.section("Add")
    manual_render.help_button(ctx, "clay-tools")
    blocked = tab is None or bool(tab.saving)
    reason = (
        "Open or start a document to build in."
        if tab is None
        else "This document is being written; the shapes come back when it lands."
    )
    for label, names in sections():
        widgets.field_label(label)
        for name in names:
            glyph = tool_palette.PRIMITIVE_ICONS.get(name, icons.BOX)
            hit = controls.menu_item(
                f"{glyph} {display_name(name)}##clay-add/{name}",
                "",
                state.generator == name,
                not blocked,
                reason=reason,
                tooltip=_defaults_tooltip(name),
            )
            if bool(hit[0] if isinstance(hit, tuple) else hit) and tab is not None:
                add_primitive(ctx, tab.doc, name)
                state.generator = name


def _defaults_tooltip(name: str) -> str:
    """What the entry places, in its generator's own numbers.

    The old Add pane printed these under the grid for the selected shape; they
    are a tooltip now, so the numbers are there when asked for and the list is
    not a wall of them.
    """
    entry = _options_for(name)
    if entry is None:
        return ""
    heading, rows, _note = entry
    body = ", ".join(f"{label} {value}" for label, value in rows)
    return f"{heading}: {body}. Edit the numbers in Properties once it is placed."


#: Shown under a primitive's defaults. Editing them is the Properties panel's
#: job, once the object exists.
_PRIMITIVE_NOTE = (
    "What the next click places. A shape's own numbers are edited afterwards, "
    "in Properties, once it exists."
)


def _options_for(name: str) -> tuple[str, tuple[tuple[str, str], ...], str] | None:
    """``(heading, rows, note)`` for one shape's defaults, or ``None`` for a
    name that is not one of Clay's shapes -- a remembered tool since removed
    from the registry, say.

    Pure: no imgui, no document, so "the defaults shown belong to the named
    shape and change when it does" is a claim a test can prove without a GL
    context.
    """
    entry = bp.GENERATORS.get(name) if name in bp.CLAY_GENERATOR_NAMES else None
    if entry is None:
        return None
    defaults, _build = entry
    rows = tuple(
        (key.replace("_", " "), _format_default(value)) for key, value in defaults.items()
    )
    return display_name(name), rows, _PRIMITIVE_NOTE


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
