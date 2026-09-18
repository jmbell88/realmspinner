"""``_facts``'s triangle count, checked for cost rather than for a number.

The 2026-09-18 audit's clay-02: ``_triangles`` recomputed the triangle count
with ``np.diff(mesh.starts)`` + ``np.maximum`` + ``.sum()`` over every visible
object's whole face array on every draw of the Document panel -- no memo key
at all, unlike ``viewport_hints.stats()``, which derives the same number from
array *lengths* alone. ~8 ms per call for a 1,000,000-face object, paid every
imgui frame the panel is visible (which it is by default).

The stub mesh below stands in for "the whole face array", and it refuses to be
scanned: ``starts``/``loops`` answer ``len()`` but raise if anything tries to
read them as an array or index into them. A function that still gets the
right number out of it cannot have rescanned anything -- it can only have
read the two lengths, which is exactly what ``corners - 2*faces`` needs.
"""

from __future__ import annotations

from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import primitives as bp
from warlock.studio.modes.clay.ui.panes import bridge as clay_bridge


class _RefusesToBeScanned:
    """Answers ``len()`` and nothing else. Any attempt to read it as an array
    (``np.diff``, ``np.asarray``, iteration, indexing) is the thing under
    test, so each of those raises rather than returning data that would let a
    rescan silently produce the right answer."""

    def __init__(self, length: int) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __array__(self, *args, **kwargs):
        raise AssertionError("mesh array was scanned, not just measured by len()")

    def __getitem__(self, item):
        raise AssertionError("mesh array was indexed, not just measured by len()")

    def __iter__(self):
        raise AssertionError("mesh array was iterated, not just measured by len()")


class _StubMesh:
    def __init__(self, faces: int, corners: int) -> None:
        # starts is (F+1,): one offset per face plus the terminator.
        self.starts = _RefusesToBeScanned(faces + 1)
        self.loops = _RefusesToBeScanned(corners)


def test_facts_triangle_count_does_not_rescan_the_whole_mesh_every_frame():
    # 6 faces, 20 corners (a mix of tris/quads/an n-gon) -> 20 - 2*6 = 8 tris.
    mesh = _StubMesh(faces=6, corners=20)
    assert clay_bridge._triangles(mesh) == 8


def test_facts_triangle_count_matches_the_fan_a_real_mesh_exports():
    """The formula itself, proved against a real mesh rather than a stub:
    the docstring's promise is that this number matches the exported file's
    fan count, and a box fans into 12 triangles from 6 quad faces."""
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="o0", mesh=bp.box()))
    total = sum(clay_bridge._triangles(obj.mesh) for obj in doc.objects if obj.visible)
    assert total == 12


def test_facts_triangle_count_of_an_empty_mesh_is_zero():
    mesh = _StubMesh(faces=0, corners=0)
    assert clay_bridge._triangles(mesh) == 0
