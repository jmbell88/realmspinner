"""Where every Clay op lives in a Blender-style menu, as data, with no imgui.

:data:`ops.OPS` says what Clay can *do* and in which element modes. It never
said how those operations *group*: the only signal was ``Op.separator_before``,
which is positional (a row sits under whichever row was registered before it, so
Shade Smooth sat under Duplicate and Smart Unwrap seven rows from Box Unwrap),
and which both surfaces that listed the registry -- the right-click menu and the
tools pane's button grid -- drew as one flat run.

This module is the grouping. Three tables, each answering one question:

* :data:`GROUPS` -- *which ops belong together*, and in what order. One group
  holds ops that a modeller thinks of as one family (Mirror, Array, Origin, UV),
  whatever mode each op is registered for. Every registered op is in exactly one
  group, and ``tests/modes/clay/test_clay_menutree.py`` checks that in both
  directions, so an op registered tomorrow cannot be left out of every menu.
* :data:`BARS` -- *which groups a mode's menu strip shows*, per element mode, the
  way Blender swaps Object for Mesh / Vertex / Edge / Face. A group's ops are
  filtered by ``op.modes`` when a bar is resolved, so one group can serve the
  object bar (Origin to Bounds) and the element bars (Origin to Selection) at
  once, and a group with nothing for the mode simply is not drawn.
* :data:`ADD_SOURCES` -- *what the Add menu draws that is not an op*: the
  generator categories, the figures, the colliders' own group. Named here so the
  header has one place to read them from rather than a list of its own.

It holds op *names*, not ``Op`` objects, so it imports nothing but :mod:`ops`
and stays a plain table; :func:`resolve` is the only function, and it is what
the header strip and the context menu both call so they cannot disagree about
what the Object menu contains.

Op ids are the MCP surface (``clay_op`` is derived from ``ops.OPS``), so nothing
here renames one: a label or a menu may move, an id may not.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import ops as clay_ops

__all__ = [
    "ADD_SOURCES",
    "BARS",
    "GROUPS",
    "OFF_BAR_GROUPS",
    "Entry",
    "Group",
    "Menu",
    "ResolvedMenu",
    "Section",
    "flat",
    "grouped_ops",
    "resolve",
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
        (
            "select-all",
            "select-none",
            "select-invert",
            "select-linked",
            "select-more",
            "select-less",
            "select-boundary",
        ),
    ),
    "basic": Group("Object", ("duplicate",)),
    "transform": Group(
        "Transform",
        ("bake", "align", "distribute", "place-between", "drop-to-ground", "snap-to-grid"),
    ),
    "origin": Group(
        "Set Origin",
        (
            "origin-to-bounds",
            "origin-to-base",
            "origin-to-world",
            "origin-to-selection",
        ),
    ),
    "mirror": Group("Mirror", ("mirror-x", "mirror-y", "mirror-z", "mirror-copy", "symmetrize")),
    "array": Group("Array", ("array-linear", "array-radial")),
    "parent": Group("Parent", ("group", "ungroup", "parent-to-last", "clear-parent")),
    "join": Group("Join & Boolean", ("join", "union", "difference", "intersection")),
    "shading": Group("Shading", ("shade-smooth", "shade-flat", "shade-auto")),
    "subdivide": Group("Subdivide", ("smooth", "subdivide")),
    "cleanup": Group(
        "Clean Up",
        ("clean-mesh", "recalc-normals", "apply-modifiers", "decimate", "retopo"),
    ),
    "separate": Group(
        "Separate", ("separate-loose", "separate-material", "separate-selection")
    ),
    "lock": Group("Lock", ("lock", "unlock")),
    "collider": Group(
        "Collider",
        (
            "collider-box",
            "collider-sphere",
            "collider-capsule",
            "collider-convex",
            "collider-compound",
        ),
    ),
    "uv": Group(
        "UV",
        (
            "unwrap",
            "mark-seam",
            "clear-seam",
            "unwrap-seams",
            "smart-unwrap",
            "pack-uv",
            "texel-density",
            "bake-detail",
        ),
    ),
    "mesh": Group(
        "Mesh",
        ("repeat-last", "extrude", "dissolve", "collapse", "bisect", "knife"),
    ),
    "vertex": Group("Vertex", ("weld", "vertex-slide")),
    "edge": Group(
        "Edge",
        (
            "bridge",
            "bevel",
            "loop-cut",
            "edge-slide",
            "rip",
            "fill-hole",
            "grid-fill",
            "spin",
            "screw",
        ),
    ),
    "face": Group(
        "Face",
        (
            "inset",
            "merge_faces",
            "flip",
            "poke",
            "triangulate",
            "tris-to-quads",
        ),
    ),
    "delete": Group("Delete", ("delete",)),
    # Frame Selection is drawn by the header's View popup, which owns the camera;
    # it is in the table so the coverage test can say "every op is in a group".
    "view": Group("View", ("frame",)),
}

#: Groups drawn somewhere other than a menu bar, named so the coverage test
#: knows that not appearing in :data:`BARS` is the intent rather than a gap.
OFF_BAR_GROUPS = frozenset({"view"})


@dataclass(frozen=True)
class Entry:
    """One line of a menu: a group as a submenu, a group inline, or a source.

    ``kind`` is ``"sub"`` (a submenu headed by the group's label), ``"inline"``
    (its ops drawn straight into the menu) or ``"add"`` (the Add menu's own
    non-op sources, which the header draws; ``name`` says which).
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


#: What the Add menu draws besides ops. The header reads these names; the
#: generator categories and figures come from the kernel registries, not from a
#: list here, so a new primitive or figure appears in Add on its own.
ADD_SOURCES: tuple[str, ...] = ("primitives", "figures", "import")

_SELECT = Menu("Select", (_inline("select"),))
_ADD = Menu(
    "Add",
    (
        Entry("add", "primitives"),
        Entry("add", "figures"),
        _sub("collider"),
        Entry("add", "import"),
    ),
)
_UV = Menu("UV", (_inline("uv"),))


def _mesh_menu(*extra: Entry) -> Menu:
    """The shared edit-mode Mesh menu, with the groups only one mode has.

    Blender's Mesh menu holds what every element mode shares and its Vertex /
    Edge / Face menus hold what one mode has; ``extra`` is how Face mode adds
    Shading and Separate (both of which apply to selected faces) without the
    vertex and edge menus carrying a submenu that would be empty for them.
    """
    return Menu(
        "Mesh",
        (
            _inline("mesh"),
            _sub("subdivide"),
            _sub("origin"),
            *extra,
            _inline("delete"),
        ),
    )


BARS: dict[str, tuple[Menu, ...]] = {
    "object": (
        _SELECT,
        _ADD,
        Menu(
            "Object",
            (
                _inline("basic"),
                _sub("transform"),
                _sub("origin"),
                _sub("mirror"),
                _sub("array"),
                _sub("parent"),
                _sub("join"),
                _sub("shading"),
                _sub("subdivide"),
                _sub("cleanup"),
                _sub("separate"),
                _sub("lock"),
                _inline("delete"),
            ),
        ),
        _UV,
    ),
    "vertex": (_SELECT, _ADD, _mesh_menu(), Menu("Vertex", (_inline("vertex"),))),
    "edge": (
        _SELECT,
        _ADD,
        _mesh_menu(),
        Menu("Edge", (_inline("edge"),)),
        _UV,
    ),
    "face": (
        _SELECT,
        _ADD,
        _mesh_menu(_sub("shading"), _sub("separate")),
        Menu("Face", (_inline("face"),)),
    ),
}


@dataclass(frozen=True)
class Section:
    """One resolved entry: a heading and the ops that are really in this mode.

    ``submenu`` says whether the surface draws it as a nested menu (``True``) or
    straight into its parent (``False``). ``source`` is non-empty only for an
    Add-menu source the header draws itself, in which case ``ops`` is empty.
    """

    label: str
    ops: tuple[clay_ops.Op, ...]
    submenu: bool
    source: str = ""


@dataclass(frozen=True)
class ResolvedMenu:
    title: str
    sections: tuple[Section, ...]


def grouped_ops(group: str, mode: str) -> tuple[clay_ops.Op, ...]:
    """The ops of *group* that apply in *mode*, in the group's own order."""
    out: list[clay_ops.Op] = []
    for name in GROUPS[group].ops:
        op = clay_ops.get(name)
        if mode in op.modes:
            out.append(op)
    return tuple(out)


def resolve(mode: str) -> tuple[ResolvedMenu, ...]:
    """The menu strip for *mode*, with every empty group and menu dropped.

    A menu whose groups hold nothing for this mode is omitted rather than drawn
    empty -- Blender's header does the same -- which is how the UV menu is
    absent in vertex mode (the only UV ops that reach an element mode are the
    edge-mode seams).
    """
    menus: list[ResolvedMenu] = []
    for menu in BARS[mode]:
        sections: list[Section] = []
        for entry in menu.entries:
            if entry.kind == "add":
                sections.append(Section(entry.name.title(), (), False, source=entry.name))
                continue
            ops = grouped_ops(entry.name, mode)
            if not ops:
                continue
            sections.append(
                Section(GROUPS[entry.name].label, ops, submenu=entry.kind == "sub")
            )
        if sections:
            menus.append(ResolvedMenu(menu.title, tuple(sections)))
    return tuple(menus)


def flat(mode: str) -> tuple[tuple[clay_ops.Op, ...], ...]:
    """Every op that applies in *mode*, as runs of one group each.

    What the right-click menu draws: one run per group in strip order, with a
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
            if entry.kind != "add" and entry.name != "delete":
                take(entry.name)
    for group in GROUPS:
        if group in OFF_BAR_GROUPS:
            take(group)
    # Delete last, as the old menu and the tools pane both drew it: it is the
    # one row whose neighbour must not be a thing you reach for by habit.
    take("delete")
    return tuple(runs)
