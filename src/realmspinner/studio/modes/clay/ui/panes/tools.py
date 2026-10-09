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
(the pure door -- the agent calls it, and a shape comes out at the origin) and
:func:`place_primitive` (the human door: the rail, the flyout, the Inspector and the
empty-state button call it, and a shape comes out on the ground, beside what
is there, in the colour last picked), :func:`display_name` (the one
spelling of a shape's name, so ``uv_sphere`` is "UV Sphere" in the rail, the
Inspector and the flyout alike), and the tables the header reads.

**Every control that changes the document is disabled while a save is in
flight**, as it is everywhere in Clay. Serialising reads the live document on a
task thread, so a control that restructured it mid-encode would write a file
describing a document that never existed. Disabling says so on screen rather
than swallowing the click.

**The operations are not here** (the 2026-10-02 menu regrouping). They live in the
Inspector's More actions and Shape it cards, the right-click menu and Ctrl+K, all
grouped by ``menutree``. (The menu strip that listed them over the viewport is gone.)
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

# The four modes as the header's switch reads them, **widest thing first**: Shape
# (the document's ``object`` mode -- you are moving whole shapes), then Points,
# Edges and Faces (``vertex``/``edge``/``face``, which is what a mesh is made of,
# smallest first). The second column is the plain word; the stored mode names are
# unchanged, and so are the keys: 1 2 3 are the three element modes and 4 is
# Shape, so the switch reads "4 1 2 3" left to right and every tooltip says which
# key is which. Held as data beside ``TOOL_ICONS`` so the row and
# ``clay_mode.ELEMENT_KEYS`` are one edit apart rather than two files apart.
MODE_BUTTONS = (
    ("object", "Shape", "4"),
    ("vertex", "Points", "1"),
    ("edge", "Edges", "2"),
    ("face", "Faces", "3"),
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

    One function for the rail, the Inspector, the flyout and the tooltips, which
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


def draw_add_menu(ctx: Any, state: Any, tab: Any, *, heading: bool = True) -> None:
    """The flyout's body: every shape Clay offers, grouped, name beside glyph.

    Enumerated rather than listed, so a sixteenth shape is a new entry in
    ``primitives.CLAY_GENERATORS`` and no edit here at all -- which is the
    whole reason that registry is data. Each entry sets ``state.generator`` to
    the key it placed, the field the empty state reads.

    Drawn inside the rail's popup (``rail.py`` owns the open and close), with
    the heading's (?) here so the manual's "Adding a primitive" section stays
    one click from the list it describes. ``heading=False`` is the inspector's
    "More shapes" disclosure, which is already under its own heading and (?).
    """
    if heading:
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
            clicked = bool(hit[0] if isinstance(hit, tuple) else hit)
            if clicked and tab is not None and place_primitive(ctx, tab.doc, name) is not None:
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
    return f"{heading}: {body}. Edit the numbers in the Inspector once it is placed."


#: Shown under a primitive's defaults. Editing them is the Properties panel's
#: job, once the object exists.
_PRIMITIVE_NOTE = (
    "What the next click places. A shape's own numbers are edited afterwards, "
    "in the Inspector, once it exists."
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
    del ctx
    obj = _new_primitive(doc, name)
    _commit_primitive(doc, obj)
    return obj


def _new_primitive(doc: Any, name: str) -> Any:
    """The object :func:`add_primitive` places, built and not yet in the document:
    the generator's defaults at the origin, auto-smoothed (see that function)."""
    from ......kernels.mesh import shading

    defaults, build = bp.GENERATORS[name]
    return bd.Obj(
        uid=bd.new_uid(),
        name=_unique_name(doc, name.replace("_", " ").title().replace(" ", "")),
        mesh=shading.auto_smooth(build(**defaults)),
        generator=name,
        params=dict(defaults),
    )


def _commit_primitive(doc: Any, obj: Any) -> None:
    """Add *obj* as one undo step and select it. Raises ``OpError`` when the
    document is full, with nothing pushed."""
    doc.add_object(obj)
    # Object mode only (the 2026-09-13 audit's clay-03) -- see
    # ``ClayDoc.add_objects``'s own comment for why selecting unconditionally
    # leaves the Properties panel naming one object while editing another.
    if doc.element_mode == "object":
        doc.select([obj.uid])


def place_primitive(ctx: Any, doc: Any, name: str) -> Any:
    """:func:`add_primitive` for a click: a refusal is a toast, never a raise.

    -> the placed object, or ``None`` when the document refused it.

    The 2026-10-07 audit's clay-13: ``ClayDoc.add_object`` now refuses a shape
    that would take the document past the object or triangle ceiling the saver
    holds it to (``OpError``, nothing pushed), and the Add buttons called
    ``add_primitive`` bare -- a click on a full document would have raised out of
    the draw call and into the pane's guard instead of telling the user why
    nothing appeared. Kept apart from :func:`add_primitive` because the agent's
    ``clay_add_primitive`` calls that and turns the same ``OpError`` into its own
    tool refusal with a gesture to close; this is the human door, and the
    decision lives here beside the draw call so it can be tested without a GL
    context. The refused click leaves no undo step and no open gesture: the
    document raises before it pushes.

    **What the click gets that the agent's door does not** (the first-click
    pass): the shape is put on the ground beside whatever is already there
    (``placement.spawn_translation``), painted with the palette slot the user
    last picked, and -- when it is the first shape, or lands outside the
    picture -- the camera is asked to frame it. All three are decided *before*
    ``add_object`` and written onto the new ``Obj``, so the add stays one undo
    step with nothing folded; moving it afterwards would be a second step and a
    Ctrl+Z that left a half-sunk box behind. ``add_primitive`` stays
    origin-placed, unpainted and camera-blind: the agent's boxes stand at
    ``[0, h/2, 0]`` after its own translation and the frozen Familiar card says
    so.
    """
    from ......kernels.mesh.elements import OpError
    from ... import mode as clay_mode

    first = not any(o.visible for o in doc.objects)
    try:
        obj = _new_primitive(doc, name)
        obj.translation = _spawn_translation(doc, obj)
        _paint_with_remembered_slot(ctx, doc, obj)
        _commit_primitive(doc, obj)
    except OpError as error:
        from ... import ops as clay_ops

        clay_ops.toast(ctx, str(error))
        return None
    clay_mode.frame_after_add(ctx, doc, obj, first=first)
    return obj


def _spawn_translation(doc: Any, obj: Any) -> Any:
    """Where the human door puts *obj*: on the ground, clear of the visible shapes.

    The boxes are the document's *world* boxes (a parented shape's local
    translation is not where it sits), and "beside" is the one selected
    object, when exactly one is selected and visible.
    """
    from ......kernels.mesh import mesh as bm
    from ... import ops as clay_ops
    from ... import placement

    visible = [o.uid for o in doc.objects if o.visible]
    boxes = clay_ops._world_boxes(doc, visible)
    near = None
    if len(doc.selection) == 1:
        near = boxes.get(next(iter(doc.selection)))
    if len(obj.mesh.positions) == 0:
        return obj.translation
    return placement.spawn_translation(bm.bounds(obj.mesh), list(boxes.values()), beside=near)


def _paint_with_remembered_slot(ctx: Any, doc: Any, obj: Any) -> None:
    """Give *obj* the palette slot the user last picked, before it is added.

    The object's default slot **and every face**, as ``ClayDoc.repaint_object``
    does it -- ``Obj.material`` is only what *new* faces are stamped with and
    the per-face array is what renders. Done on the unattached object so the
    paint is part of the add's one step. A textured slot arriving on a mesh with
    no UVs is box-unwrapped, ``repaint_object``'s own rule (``_with_uvs_for``).
    """
    from dataclasses import replace

    import numpy as np

    from ......kernels.mesh import uv as uv_projection
    from ... import mode as clay_mode

    slot = clay_mode.remembered_slot(ctx, doc)
    if slot is None:
        return
    mesh = replace(obj.mesh, material=np.full(len(obj.mesh.material), slot, dtype="i4"))
    if mesh.uv is None and doc.materials[slot].base_color is not None:
        mesh = uv_projection.box_unwrap(mesh)
    obj.mesh = mesh
    obj.material = slot


def _unique_name(doc: Any, base: str) -> str:
    """A name no object in *doc* wears."""
    taken = {obj.name for obj in doc.objects}
    if base not in taken:
        return base
    # The same counting-up rule ``ops.duplicate`` uses, so two objects never
    # wear one name whichever way they arrived.
    return ops.next_name(base, taken)
