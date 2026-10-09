"""The eight colours the inspector offers before a document has any palette to speak of.

A new model has one grey slot, and a first-time user who wants a blue box should
not have to learn that a palette exists, that ``+`` appends a slot, and that the
slot's colour is edited somewhere else before they can have one. The inspector's
Colour card lists these eight by name (the document's own slots are the row under
them); a click **finds the document's slot that already holds the colour, or adds
one named after it, and paints the selected shapes with it, as one undo step** --
the picked faces, in Faces mode (:func:`paint_faces`).

**They are UI-only, and that is the whole design.** Nothing here is ever seeded
into :class:`~realmspinner.kernels.mesh.document.ClayDoc` or its default palette:
a fresh document still holds exactly the one grey slot. Three things stand on
that -- the agent's ``clay_scene`` (a model an agent builds must not arrive with
eight colours it never asked for), the frozen Familiar card (it describes the
palette an empty document has) and the tour's ``clay_palette_at_least`` condition
(which counts the slots the *reader* added). A slot appears only when a click asks
for it, and an unused one stays until the user removes it like any other.

**The number stored is the number drawn.** ``base_color_factor`` is what the
palette row and the colour field show as-is (``swatches._swatch_colour`` draws
the factor, ``props._material`` edits it), so a starter's slot holds ``hex / 255``
per channel and its swatch and its slot's in the card are the same pixels. A
sRGB-to-linear conversion here would store a different number than the field
displays for the same colour and the "find the slot that already holds it" match
would stop working for a colour the user picked by hand.

Pure: no imgui, no GL, so what a click does is a claim a test can check.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from ....kernels.mesh import document as clay_document


@dataclass(frozen=True)
class StarterColour:
    """One named swatch: what it is called and the colour it paints."""

    name: str
    hex: str

    @property
    def factor(self) -> tuple[float, float, float, float]:
        """The ``base_color_factor`` a slot of this colour holds (opaque)."""
        digits = self.hex.lstrip("#")
        r, g, b = (int(digits[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        return (r, g, b, 1.0)


#: The eight, in the order the card draws them. Names are the plain words a
#: first-time user says ("make it blue"); the hexes are the redesign's.
STARTER_COLOURS: tuple[StarterColour, ...] = (
    StarterColour("Stone", "#D6D6DC"),
    StarterColour("Red", "#E5484D"),
    StarterColour("Orange", "#F7921E"),
    StarterColour("Yellow", "#F2C94C"),
    StarterColour("Green", "#4CC38A"),
    StarterColour("Blue", "#4A7BEA"),
    StarterColour("Purple", "#8B6CF0"),
    StarterColour("Charcoal", "#4B4F5E"),
)

#: How close two channels must be to count as the same colour. Half a step of an
#: 8-bit channel: tight enough that two colours a person can tell apart never
#: match, loose enough that a hex typed into the colour picker (which rounds to a
#: byte) finds the slot a click made.
_TOLERANCE = 0.5 / 255.0


def by_name(name: str) -> StarterColour:
    """The starter called *name*; ``KeyError`` for a name the card does not draw."""
    for colour in STARTER_COLOURS:
        if colour.name == name:
            return colour
    raise KeyError(f"no starter colour named {name!r}")


def holds(material: Any, colour: StarterColour) -> bool:
    """Whether palette *material* is a flat, opaque slot of exactly *colour*.

    A textured slot is a picture whose colour factor is white (``add_texture``
    moves the colour into the texels), and a see-through or cut-out one is not
    the plain colour the swatch promises, so none of them match.
    """
    if material.base_color is not None or material.alpha_mode != "OPAQUE":
        return False
    factor = material.base_color_factor
    return all(abs(float(a) - b) <= _TOLERANCE for a, b in zip(factor, colour.factor, strict=True))


def find_slot(doc: Any, colour: StarterColour) -> int | None:
    """The first palette slot of *doc* that holds *colour*, or ``None``."""
    for index, material in enumerate(doc.materials):
        if holds(material, colour):
            return index
    return None


def matching(doc: Any, slot: int | None) -> StarterColour | None:
    """The starter whose colour palette slot *slot* holds, or ``None`` ("custom").

    What the card outlines as the shape's current colour; ``None`` is a slot
    that is none of the eight (a textured one, an edited one, a stranger).
    """
    if slot is None or not 0 <= slot < len(doc.materials):
        return None
    material = doc.materials[slot]
    for colour in STARTER_COLOURS:
        if holds(material, colour):
            return colour
    return None


def paint(ctx: Any, doc: Any, uids: Iterable[int], colour: StarterColour) -> int:
    """Paint *uids* with *colour*, adding its slot first if the palette has none,
    as **one** undo step. -> the slot index used.

    ``repaint_object`` per shape (the default slot and every face, which is what
    renders), inside one mark, so a Ctrl+Z puts back the colour of every shape
    *and* removes the slot this click added, in a single step -- the rule
    ``add_material_and_assign`` and the slot row's multi-object click already keep.
    The slot is remembered (``ClayTab.last_slot``) so the next shape placed by
    hand arrives in it, as a press on a slot swatch does. With no *uids* (nothing
    selected) that is the whole of it: the colour the next shape will wear.
    """
    from . import mode as clay_mode

    mark = doc.history.mark()
    try:
        index = find_slot(doc, colour)
        if index is None:
            fresh = clay_document.default_material(colour.name)
            index = doc.add_material(replace(fresh, base_color_factor=colour.factor))
        for uid in sorted(uids):
            doc.repaint_object(uid, index)
    finally:
        doc.history.collapse_since(mark)
    clay_mode.remember_slot(ctx, doc, index)
    return index


def paint_faces(ctx: Any, doc: Any, colour: StarterColour) -> int | None:
    """Paint the picked faces with *colour*, adding its slot first if the palette has
    none, as **one** undo step. -> the slot index used, or ``None`` with no faces
    picked.

    The ``assign-material`` op, the door every face paint goes through (the Edit
    menu's Assign Material, a slot swatch's click, the agent's ``faces``): it leaves
    each object's default slot alone and is one step across however many objects have
    a pick. With no face picked nothing is added -- a click that painted nothing must
    not leave a stray slot behind -- and the refusal is said in a toast.
    """
    from . import mode as clay_mode
    from . import ops as clay_ops

    op = clay_ops.get("assign-material")
    if not op.enabled(doc):
        ctx.toast(clay_ops.reason_for(op, doc) or "Select faces to paint them.")
        return None
    mark = doc.history.mark()
    try:
        index = find_slot(doc, colour)
        if index is None:
            fresh = clay_document.default_material(colour.name)
            index = doc.add_material(replace(fresh, base_color_factor=colour.factor))
        clay_ops.run(ctx, doc, op, index=index)
    finally:
        doc.history.collapse_since(mark)
    clay_mode.remember_slot(ctx, doc, index)
    return index
