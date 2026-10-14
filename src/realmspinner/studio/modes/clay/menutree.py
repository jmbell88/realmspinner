"""Where every Clay op lives in a menu, as data, with no imgui.

:data:`ops.OPS` says what Clay can *do* and in which element modes. It never
said how those operations *group*: the only signal was registration order
(a row sat under whichever row was registered before it, so Shade Smooth sat
under Duplicate), which both surfaces that listed the registry -- the
right-click menu and the Inspector's action cards -- drew as one flat run.
(The 2026-10-07 audit's clay-42: this named an ``Op.separator_before`` field
that no longer exists.)

This module is the grouping. Two tables and a set, each answering one question:

* :data:`GROUPS` -- *which ops belong together*, and in what order. One group
  holds ops that a modeller thinks of as one family (Mirror, Origin, UV),
  whatever mode each op is registered for. Every registered op is in exactly one
  group, and ``tests/modes/clay/test_clay_menutree.py`` checks that in both
  directions, so an op registered tomorrow cannot be left out of every menu.
* :data:`BARS` -- *which groups a mode's menus show and in what order*, per
  element mode: Select | Object | UV in object mode, Select | Mesh in the three
  element modes (vertex, edge and face share the one Mesh menu). A group's ops are
  filtered by ``op.modes`` when a bar is read, so one group can serve the object
  bar (Origin to Bounds) and the element bars (Origin to Selection) at once, and a
  group with nothing for the mode simply is not drawn.

The Select | Add | Object | Mesh | UV strip that used to draw :data:`BARS` over the
viewport is gone (Landing 11 of the 2026-10 Clay UI rebuild), but the table is not:
it is still *the order and the grouping* -- :func:`flat` reads it for the right-click
menu, and the Inspector's More actions and Shape it cards group by :data:`GROUPS`.
The Add menu's non-op rows went with the strip: the shapes are the rail's ``+`` and
the Inspector's tiles and "More shapes", and Import Mesh is on the Inspector's Model &
history page, in the File menu and in Ctrl+K.

It holds op *names*, not ``Op`` objects, so it imports nothing but :mod:`ops`
and stays a plain table; :func:`flat` is the one function surfaces call, so they
cannot disagree about what the Object menu contains.

Op ids are the MCP surface (``clay_op`` is derived from ``ops.OPS``), so nothing
here renames one: a label or a menu may move, an id may not.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import ops as clay_ops

__all__ = [
    "BARS",
    "GROUPS",
    "OFF_BAR_GROUPS",
    "Entry",
    "Group",
    "Menu",
    "flat",
    "grouped_ops",
]


@dataclass(frozen=True)
class Group:
    """One family of ops: its heading and its members, in display order."""

    label: str
    ops: tuple[str, ...]


# Display order inside a group is the order below, not registration order,
# because registration order is the order tranches landed in -- the very thing
# this module exists to stop surfacing.
GROUPS: dict[str, Group] = {
    "select": Group(
        "Select",
        ("select-all", "select-none", "select-invert", "select-linked"),
    ),
    "basic": Group("Object", ("duplicate",)),
    "transform": Group("Transform", ("drop-to-ground", "snap-to-grid")),
    "origin": Group(
        "Set Origin",
        (
            "origin-to-bounds",
            "origin-to-base",
            "origin-to-world",
            "origin-to-selection",
        ),
    ),
    "mirror": Group("Mirror", ("mirror-x", "mirror-y", "mirror-z", "mirror-copy")),
    "parent": Group("Parent", ("group", "ungroup", "parent-to-last", "clear-parent")),
    "rig": Group(
        "Rig",
        (
            "mark-bone",
            "unmark-bone",
            "unbind-skin",
            "insert-key",
            "delete-key",
            "reset-pose",
            "apply-pose-as-rest",
        ),
    ),
    "join": Group("Join", ("join", "union", "difference", "intersection")),
    "shading": Group("Shading", ("shade-smooth", "shade-flat")),
    "cleanup": Group("Clean Up", ("recalc-normals", "simplify")),
    "smooth": Group("Smooth", ("smooth",)),
    "separate": Group("Separate", ("separate-loose", "separate-selection")),
    "uv": Group("UV", ("unwrap", "pack-uv")),
    "mesh": Group(
        "Mesh",
        (
            "repeat-last",
            "extrude",
            "inset",
            "bevel",
            "loop-cut",
            "edge-slide",
            "merge_faces",
            "weld",
            "flip",
            "assign-material",
            "triangulate",
            "subdivide",
        ),
    ),
    # Not in ``mesh``: that group sits open on the Inspector's Shape it card, and the
    # four ops that returned on 2026-10-13 (six with Loop Cut and Edge Slide) pushed
    # Delete 17 px past the open dock at 1600x950. A closed fold costs the card
    # nothing and the right-click menu still lists them.
    "topology": Group("Topology", ("dissolve", "bridge", "fill-hole", "collapse")),
    "delete": Group("Delete", ("delete",)),
    # Frame Selection is drawn by the toolbar's View popup, which owns the camera;
    # it is in the table so the coverage test can say "every op is in a group".
    "view": Group("View", ("frame",)),
}

#: Groups drawn somewhere other than a menu bar, named so the coverage test
#: knows that not appearing in :data:`BARS` is the intent rather than a gap.
OFF_BAR_GROUPS = frozenset({"view"})


@dataclass(frozen=True)
class Entry:
    """One line of a menu: a group as a submenu, or a group inline.

    ``kind`` is ``"sub"`` (a submenu headed by the group's label) or ``"inline"``
    (its ops drawn straight into the menu).
    """

    kind: str
    name: str


def _sub(name: str) -> Entry:
    return Entry("sub", name)


def _inline(name: str) -> Entry:
    return Entry("inline", name)


@dataclass(frozen=True)
class Menu:
    title: str
    entries: tuple[Entry, ...]


_SELECT = Menu("Select", (_inline("select"),))
_UV = Menu("UV", (_inline("uv"),))


def _mesh_menu(*extra: Entry) -> Menu:
    """The one edit-mode Mesh menu, shared by vertex, edge and face mode.

    ``extra`` is how Face mode adds Shading and Separate (both of which apply
    to selected faces) without the vertex and edge menus carrying a submenu
    that would be empty for them; naming only what a mode can have keeps the
    table honest.
    """
    return Menu(
        "Mesh",
        (
            _inline("mesh"),
            _sub("topology"),
            _sub("origin"),
            *extra,
            _inline("delete"),
        ),
    )


BARS: dict[str, tuple[Menu, ...]] = {
    "object": (
        _SELECT,
        Menu(
            "Object",
            (
                _inline("basic"),
                _sub("transform"),
                _sub("origin"),
                _sub("mirror"),
                _sub("parent"),
                _sub("rig"),
                _inline("join"),
                _sub("shading"),
                _inline("cleanup"),
                _inline("smooth"),
                _sub("separate"),
                _inline("delete"),
            ),
        ),
        _UV,
    ),
    "vertex": (_SELECT, _mesh_menu()),
    "edge": (_SELECT, _mesh_menu()),
    "face": (_SELECT, _mesh_menu(_sub("shading"), _sub("separate"))),
}


def grouped_ops(group: str, mode: str) -> tuple[clay_ops.Op, ...]:
    """The ops of *group* that apply in *mode*, in the group's own order."""
    out: list[clay_ops.Op] = []
    for name in GROUPS[group].ops:
        op = clay_ops.get(name)
        if mode in op.modes:
            out.append(op)
    return tuple(out)


def flat(mode: str) -> tuple[tuple[clay_ops.Op, ...], ...]:
    """Every op that applies in *mode*, as runs of one group each.

    What the right-click menu draws: one run per group in bar order, with a
    separator between runs. An op named by two bar entries (Shading is under
    both Object and Face's Mesh menu in different bars, never twice in one) is
    emitted once, at its first appearance, and the off-bar groups (Frame
    Selection) follow, so the flat list holds exactly ``ops.menu(mode)``.
    """
    seen: set[str] = set()
    runs: list[tuple[clay_ops.Op, ...]] = []

    def take(group: str) -> None:
        run = tuple(op for op in grouped_ops(group, mode) if op.name not in seen)
        if run:
            seen.update(op.name for op in run)
            runs.append(run)

    for menu in BARS[mode]:
        for entry in menu.entries:
            if entry.name != "delete":
                take(entry.name)
    for group in GROUPS:
        if group in OFF_BAR_GROUPS:
            take(group)
    # Delete last, as the old menu and the old button grid both drew it: it is the
    # one row whose neighbour must not be a thing you reach for by habit.
    take("delete")
    return tuple(runs)
