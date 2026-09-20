"""What is wrong with one mesh, as rows a panel can draw and a click can select.

:func:`~.adjacency.check_manifold` already measures every defect the CSR layout
cannot prevent; what it hands back is six arrays, and six arrays are not a user
interface. This turns them into :class:`Finding` rows -- a sentence, a count,
the element mode the defect is expressed in, and the
:class:`~.elements.ElementSel` that selects the offenders -- so the pane draws a
list and the click handler is one ``set_element_sel``.

It lives beside the report rather than in the pane for the reason every rule
about geometry in this package does: a defect's *name*, the mode it is selected
in and the elements it covers are all assertable from a hand-built mesh with no
window anywhere. The pane decides where the rows go and nothing else.

**A hole is a boundary loop, not a boundary edge.** The report lists edges,
because an edge pair survives a renumbering where a loop index does not, but
"37 holes" for one open quad face is a wrong answer rather than a coarse one --
so the count comes from :func:`~.adjacency.boundary_loops` while the selection
stays the report's edges. Both build the same weakly-cached
:class:`~.adjacency.Adjacency`, so asking twice costs once.

**Nothing here decides that a mesh is bad.** An open sheet is a legitimate mesh
and so is a plane with no thickness; ``clean`` is a strict reading rather than a
verdict, which is why the rows say what was measured and never how to feel about
it. That is the same doctrine ``widgets.quality_badge`` is written under.

**Three rows come from :mod:`.ops_clean` rather than :func:`~.adjacency.check_manifold`**,
because the report cannot measure them: ``degenerate`` (a face with under three
distinct vertices, or (near) zero area -- neither is a CSR-structure question),
``flipped_face`` (a face whose winding disagrees with the majority of its own
connected shell, read off the same BFS :func:`~.ops_clean.recalc_outside` walks
to fix it) and ``inside_out`` (a closed shell that is uniformly wound but the
wrong way round, which no per-face or per-edge measurement can see at all --
only a volume sign can). Duplicate faces and loose vertices are **not**
duplicated under a second name here even though :class:`~.ops_clean.Survey`
also counts both: the existing ``duplicate``/``unused`` rows already answer
them from `report` at no extra cost, and `Survey.duplicate_faces` counts a
narrower thing (the later face of a repeat only, matching what
:func:`~.ops_clean.remove_duplicate_faces` would delete) than
`report.duplicate_faces` does (every member of the group) -- switching would
change what an existing row selects, not just where its number comes from.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import elements as el
from . import ops_clean
from .adjacency import ManifoldReport, boundary_loops, check_manifold
from .mesh import Mesh

__all__ = ["Finding", "SceneFinding", "findings", "rows_for", "scene_findings"]


@dataclass(frozen=True)
class Finding:
    """One defect, said in a sentence and selectable.

    ``kind`` is the stable machine name (a test asserts on it, a label is free
    to be reworded); ``label`` is what the row draws, already pluralised.
    ``mode`` is the element mode the selection has to be read in -- a caller
    that sets the selection without setting the mode leaves the user looking at
    an empty vertex overlay over a face selection.
    """

    kind: str
    label: str
    count: int
    mode: str
    sel: el.ElementSel


def _plural(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def findings(mesh: Mesh) -> list[Finding]:
    """Every defect in *mesh*, most structural first. Empty means clean.

    O(corners) and not frame-thread work on a real model -- see the note on
    :func:`~.adjacency.check_manifold`. The caller runs it on demand and holds
    the result against the mesh it was measured from.
    """
    return rows_for(mesh, check_manifold(mesh))


def rows_for(mesh: Mesh, report: ManifoldReport) -> list[Finding]:
    """The rows for a report already in hand, so a caller measuring for its own
    reasons does not measure twice."""
    out: list[Finding] = []

    if len(report.boundary_edges):
        rings, _pinched = boundary_loops(mesh)
        # Rings can legitimately come back empty for a boundary the walk could
        # not close; the edges are still real, so the row is drawn about them.
        holes = len(rings) or 1
        out.append(
            Finding(
                kind="hole",
                label=_plural(holes, "hole", "holes"),
                count=holes,
                mode="edge",
                sel=el.ElementSel(edges=report.boundary_edges),
            )
        )

    out.extend(
        _edge_row(kind, one, many, edges)
        for kind, one, many, edges in (
            (
                "nonmanifold",
                "non-manifold edge",
                "non-manifold edges",
                report.nonmanifold_edges,
            ),
            ("flipped", "flipped edge", "flipped edges", report.flipped_edges),
        )
        if len(edges)
    )

    if len(report.repeated_corner_faces):
        out.append(
            Finding(
                kind="repeated",
                label=_plural(
                    len(report.repeated_corner_faces),
                    "face with a repeated vertex",
                    "faces with a repeated vertex",
                ),
                count=len(report.repeated_corner_faces),
                mode="face",
                sel=el.ElementSel(faces=report.repeated_corner_faces),
            )
        )
    if len(report.duplicate_faces):
        out.append(
            Finding(
                kind="duplicate",
                label=_plural(len(report.duplicate_faces), "duplicate face", "duplicate faces"),
                count=len(report.duplicate_faces),
                mode="face",
                sel=el.ElementSel(faces=report.duplicate_faces),
            )
        )
    if len(report.unused_verts):
        out.append(
            Finding(
                kind="unused",
                label=_plural(len(report.unused_verts), "unused vertex", "unused vertices"),
                count=len(report.unused_verts),
                mode="vertex",
                sel=el.ElementSel(verts=report.unused_verts),
            )
        )

    # ``ops_clean.survey``'s three defects `check_manifold` cannot measure at
    # all: degenerate area, winding that disagrees with a face's own shell, and
    # a shell that is uniformly wound but net inside-out. Duplicate faces and
    # loose vertices are already rows above (``duplicate``/``unused``, straight
    # off `report`) and are not duplicated here under a second name -- both
    # measure the same defect `ops_clean.Survey.duplicate_faces` and
    # `.loose_vertices` name, `report` already has them for free from the same
    # `check_manifold` pass, and `Survey.duplicate_faces` counts a repeat
    # narrower ("the later face only") than `report.duplicate_faces` does, so
    # switching would move the selection and count these existing rows already
    # promise. Both mode="face" (`inside_out`'s selection is every face of the
    # shell the volume sign convicted, not only the ones that individually
    # disagree with a neighbour -- see `ops_clean.FaceDefectMasks`).
    masks = ops_clean.face_defect_masks(mesh)
    if masks.degenerate.any():
        faces = np.flatnonzero(masks.degenerate).astype("i4")
        out.append(
            Finding(
                kind="degenerate",
                label=_plural(len(faces), "degenerate face", "degenerate faces"),
                count=len(faces),
                mode="face",
                sel=el.ElementSel(faces=faces),
            )
        )
    if masks.flipped.any():
        faces = np.flatnonzero(masks.flipped).astype("i4")
        out.append(
            Finding(
                kind="flipped_face",
                label=_plural(len(faces), "flipped face", "flipped faces"),
                count=len(faces),
                mode="face",
                sel=el.ElementSel(faces=faces),
            )
        )
    if masks.inside_out.any():
        faces = np.flatnonzero(masks.inside_out).astype("i4")
        # The count is *shells*, the same "count the loop, select the edges"
        # split the ``hole`` row above uses -- a shell reads as one defect
        # however many faces make it up.
        shells = ops_clean.survey(mesh).inside_out_shells
        out.append(
            Finding(
                kind="inside_out",
                label=_plural(shells, "inside-out shell", "inside-out shells"),
                count=shells,
                mode="face",
                sel=el.ElementSel(faces=faces),
            )
        )
    return out


def _edge_row(kind: str, one: str, many: str, edges: np.ndarray) -> Finding:
    return Finding(
        kind=kind,
        label=_plural(len(edges), one, many),
        count=len(edges),
        mode="edge",
        sel=el.ElementSel(edges=edges),
    )


@dataclass(frozen=True)
class SceneFinding:
    """One defect of the *document* rather than of a mesh, said in a sentence
    and pointing at the objects it is about.

    Deliberately not a :class:`Finding`: every field of that one exists to be
    read in an element mode (a ``mode`` and an ``ElementSel``), and a defect
    whose subject is "these three objects" has no elements to select. So it
    carries ``uids`` where the other carries a selection, and the two lists
    stay separate all the way out to the tool's answer.
    """

    kind: str
    label: str
    uids: tuple[int, ...]


_COPY_RE = re.compile(r"\.\d{3,}$")
"""``ops.next_name`` always counts up in 3-digit steps (``.001``, ..., ``.999``,
``.1000``, ``.10000``, ...), never resetting the width, so a copy suffix is
three digits *or more* -- ``\\d{3,}`` -- not the fixed four-character slice
this replaced. The 2026-09-15 audit's clay-04 found the fixed slice checking
only the character exactly four from the end for a literal ``.``: true for
``Box.001`` but false for ``Box.1000`` (four from the end is ``1``, not
``.``) and every wider suffix after it, so a family duplicated past 999
copies silently stopped being read as one -- ``Box.1000`` came back as its
own family, named after itself, rather than joining ``Box``. Anchored at the
end (``$``) rather than searched, so a name that merely contains a
dot-digits run earlier (an imported ``Box.001.glb``) is untouched."""


def _family(name: str) -> str:
    """The name a copy was counted up from -- ``Box.001`` -> ``Box``, ``Box``
    -> ``Box``.

    Read off the name rather than off a provenance field, because there is no
    provenance field and adding one would have to survive save, load and undo
    to be worth anything. That makes this a *hint*: renaming a copy takes it
    out of its family, which is exactly the reader's own signal that it is no
    longer one of a set. A hint is all this finding claims to be.
    """
    return _COPY_RE.sub("", name)


def scene_findings(objects: Sequence[Any]) -> list[SceneFinding]:
    """Every defect measurable from the document's object list alone.

    Today that is one: a family of copies (``Box``, ``Box.001``,
    ``Box.002`` -- what ``clay_ops``' arrays and Mirror Copy leave behind)
    whose members no longer agree about their material. Nothing refuses that
    and nothing could: a copy is an independent object and painting one of a
    set a different colour is a legitimate thing to want. What was wrong is
    that the *accidental* version of it -- placing an object, arraying it,
    and only then naming the original in ``clay_material``, which leaves
    every copy on the material it was made with -- had no symptom at all
    short of looking at a render, which is the one thing an agent working
    over a pipe cannot do cheaply.

    Objects are duck-typed on ``uid``, ``name`` and ``material``, so this
    reads a document without importing one.
    """
    families: dict[str, list[Any]] = {}
    for obj in objects:
        families.setdefault(_family(obj.name), []).append(obj)
    out: list[SceneFinding] = []
    for family in sorted(families):
        members = families[family]
        if len(members) < 2:
            continue
        slots = {int(obj.material) for obj in members}
        if len(slots) < 2:
            continue
        out.append(
            SceneFinding(
                kind="copies_disagree_on_material",
                label=(
                    f"{_plural(len(members), 'copy', 'copies')} of {family!r} "
                    f"use {len(slots)} different materials"
                ),
                uids=tuple(int(obj.uid) for obj in members),
            )
        )
    return out
