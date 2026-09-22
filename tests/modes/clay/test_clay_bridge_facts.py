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

from dataclasses import replace

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import readiness
from realmspinner.studio.modes.clay.ui.panes import bridge as clay_bridge


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


# --- clay-05 (2026-09-22 audit): the pivot check's Fix ------------------------


class _FixCtx:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


class _FixTab:
    def __init__(self, doc: bd.ClayDoc) -> None:
        self.doc = doc


def _floating_box(y: float) -> bd.Obj:
    return bd.Obj(uid=bd.new_uid(), name="Floating", mesh=replace(bp.box(), uv=None),
                  translation=(0.0, y, 0.0))


def test_pivot_fix_grounds_the_actually_floating_object_when_nothing_is_selected():
    """Before this fix, ``readiness._check_pivot`` always reported
    ``uids=()`` (the 2026-09-22 audit's clay-05), so ``bridge._run_fix``'s
    ``if uids:`` guard never fired and ``drop-to-ground`` ran on whatever was
    already selected -- nothing, with an empty selection, so the press did
    nothing and gave no toast. The pivot check is document-wide, so its Fix
    must select every object it measured before running the op."""
    doc = bd.ClayDoc()
    floating = doc.add_object(_floating_box(5.0))
    ctx = _FixCtx()
    tab = _FixTab(doc)

    report = readiness.validate(doc)
    pivot = next(c for c in report.checks if c.key == "pivot")
    assert pivot.status == "warn"
    assert pivot.uids == (floating.uid,)
    assert doc.selection == set()  # nothing selected to start with

    clay_bridge._run_fix(ctx, tab, pivot.fix, pivot.uids)

    assert doc.objects[0].translation[1] == 0.5  # grounded: box half-height
    assert ctx.toasts == []


def test_pivot_fix_does_not_move_an_unrelated_selected_object_instead():
    """The second half of the same hole: with an unrelated object selected,
    the old ``if uids:`` guard being empty meant the *selected* object got
    grounded instead of the actually floating one -- the wrong object moved,
    silently, while the flagged object stayed exactly where it was. Both
    objects here are off the ground (so the document-wide check still warns
    with the unrelated one present), and only ``unrelated`` starts selected."""
    doc = bd.ClayDoc()
    floating = doc.add_object(_floating_box(5.0))
    unrelated = doc.add_object(bd.Obj(
        uid=bd.new_uid(), name="AlsoFloating", mesh=replace(bp.box(), uv=None),
        translation=(3.0, 2.0, 0.0),
    ))
    doc.select([unrelated.uid])
    ctx = _FixCtx()
    tab = _FixTab(doc)

    report = readiness.validate(doc)
    pivot = next(c for c in report.checks if c.key == "pivot")
    assert set(pivot.uids) == {floating.uid, unrelated.uid}

    clay_bridge._run_fix(ctx, tab, pivot.fix, pivot.uids)

    by_uid = {obj.uid: obj for obj in doc.objects}
    assert by_uid[floating.uid].translation[1] == 0.5, "the flagged object must be grounded"
    assert by_uid[unrelated.uid].translation[1] == 0.5, "grounded too, not left floating"


def test_run_fix_toasts_when_the_op_has_nothing_to_do():
    """``_run_fix`` must not pretend a no-op press did something: when
    ``clay_ops.run`` returns ``False`` (nothing selected, nothing to fix),
    the button now says so rather than silently swallowing the press."""
    doc = bd.ClayDoc()
    ctx = _FixCtx()
    tab = _FixTab(doc)

    clay_bridge._run_fix(ctx, tab, "drop-to-ground", ())

    assert ctx.toasts, "an empty document has nothing to ground and must toast"
    assert ctx.toasts[0][1] == "warn"
