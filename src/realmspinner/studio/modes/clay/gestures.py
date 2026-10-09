"""What a left-button press in Clay's viewport means: a pure classifier.

No imgui, no GL, no pygame -- numbers and strings, so "what does a plain drag on
empty space do" is a table a headless test can walk rather than a sentence the
viewport has to be driven to prove.

**The rule, in every mode.** A plain drag orbits. A click without a drag picks
(or clears). Shift+drag sweeps a selection box that *adds* to the selection and
Ctrl+drag one that *subtracts*; Alt+drag orbits whatever is under the pointer
and is decided before this module is asked (``_view_drag._press``). Before this
module the answer depended on the mode: an element mode with the Select tool
swept a box on a plain drag, so a first-time user dragging empty space to look
around drew a rectangle and cleared their picks, and object mode cleared the
selection *at the press*, so the first orbit drag deselected the object being
looked at.

The cure is that a press which starts on nothing no longer decides anything
until it knows what the pointer did: :class:`Press` records the gesture's three
outcomes up front -- what the press does at once (``now``), what moving past the
slop turns it into (``drag``), and what a release inside the slop does
(``click``) -- and the viewport runs whichever the pointer's travel selects.
"""

from __future__ import annotations

from dataclasses import dataclass

#: How far the pointer may wander, in pixels, before a press stops being a
#: click. Four is the figure the right-button menu already uses ("a release
#: within four pixels of the press"), so the two buttons agree on what a click
#: is. Travel is the farthest the pointer ever got from the press, not where it
#: ended: dragging away and back is still a drag.
CLICK_SLOP = 4.0

#: What is under the pointer at the press, as the viewport reports it.
#: ``element`` -- a vertex, edge or face in an element mode; ``object`` -- an
#: object in object mode; ``surface`` -- an object's surface in an element mode
#: with no element in reach; ``nothing`` -- empty space.
UNDER = ("element", "object", "surface", "nothing")

ORBIT = "orbit"
MARQUEE_ADD = "marquee-add"
MARQUEE_SUB = "marquee-sub"
PICK = "pick"
CLEAR = "clear"


@dataclass(frozen=True)
class Press:
    """The three outcomes of one left-button press (see the module docstring).

    ``now`` is ``"pick"`` when the press itself selects what is under the
    pointer, else ``""``. ``drag`` is what moving past :data:`CLICK_SLOP`
    does: :data:`ORBIT`, :data:`MARQUEE_ADD` or :data:`MARQUEE_SUB`. ``click``
    is what a release that never left the slop does: :data:`PICK` (the thing
    under the pointer, deferred so a drag from it can be a marquee), :data:`CLEAR`
    (deselect) or ``""`` (nothing).
    """

    now: str
    drag: str
    click: str


def pick_how(shift: bool, ctrl: bool) -> str:
    """How a pick combines with the selection already held: Shift adds, Ctrl
    subtracts, neither replaces. Shift wins when both are down, as it always has."""
    return "add" if shift else ("subtract" if ctrl else "replace")


def classify_press(
    *, element_mode: bool, under: str, tool: str, shift: bool, ctrl: bool
) -> Press:
    """The outcomes of a left press that no handle and no Alt key has claimed.

    ``under`` is one of :data:`UNDER`; ``element_mode`` whether the document is
    in Points, Edges or Faces. Object mode has no box select (it never did, and
    adding one is not this rule's business), so Shift or Ctrl there only change
    how a click on an object combines and an empty-space drag stays an orbit.

    In an element mode the box is available under every tool, held behind a
    modifier so a plain drag is never ambiguous. An element under a Shift or Ctrl
    press is not picked at the press but on the release, so a box can start on
    the model itself -- in Faces mode that is nearly everywhere there is
    anything to box.

    A click on empty space clears the picks only under the Select tool (the
    gizmo tools leave them, as before: dragging a handle's neighbourhood is how
    a user looks around under Move), while a click on a surface with no element
    in reach clears that one object's picks under every tool, again as before.
    """
    if under not in UNDER:
        raise ValueError(f"unknown press target {under!r}")
    modified = shift or ctrl
    sweep = MARQUEE_ADD if shift else MARQUEE_SUB
    if not element_mode:
        if under == "object":
            return Press(now=PICK, drag=ORBIT, click="")
        return Press(now="", drag=ORBIT, click="" if modified else CLEAR)
    if under == "element":
        if modified:
            return Press(now="", drag=sweep, click=PICK)
        return Press(now=PICK, drag=ORBIT, click="")
    clears = under == "surface" or tool == "select"
    if modified:
        return Press(now="", drag=sweep, click="")
    return Press(now="", drag=ORBIT, click=CLEAR if clears else "")


def is_click(travel: float) -> bool:
    """Whether a press whose pointer got at most *travel* pixels away is a click."""
    return travel < CLICK_SLOP
