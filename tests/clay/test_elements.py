"""The one selection type: canonical form, conversion and the click modifiers."""

from __future__ import annotations

import numpy as np
import pytest

from warlock.kernels.mesh import elements as el
from warlock.kernels.mesh import primitives as prim


def test_arrays_are_canonicalised_copied_and_frozen() -> None:
    src = np.array([5, 1, 5, 3], dtype="i4")
    sel = el.ElementSel(verts=src, edges=[[7, 2], [2, 7], [1, 0]], faces=[2, 2])
    assert sel.verts.tolist() == [1, 3, 5]
    assert sel.edges.tolist() == [[0, 1], [2, 7]], "low-first, lexsorted, deduped"
    assert sel.faces.tolist() == [2]

    src[0] = 99
    assert sel.verts.tolist() == [1, 3, 5], "the selection copied its input"
    with pytest.raises(ValueError):
        sel.verts[0] = 0


def test_an_empty_selection_is_empty_whichever_way_it_is_asked() -> None:
    assert el.is_empty(el.empty())
    assert el.is_empty(None)
    assert len(el.empty()) == 0
    assert not el.is_empty(el.ElementSel(verts=[0]))


def test_two_selections_of_the_same_elements_compare_the_same() -> None:
    a = el.ElementSel(edges=[[3, 1], [0, 2]])
    b = el.ElementSel(edges=[[2, 0], [1, 3]])
    assert a.same_as(b)
    assert a is not b, "identity stays distinct -- id(sel) keys the overlay cache"


def test_affected_verts_unions_every_kind() -> None:
    m = prim.box()
    sel = el.ElementSel(verts=[7], edges=[[0, 1]], faces=[0])
    got = el.affected_verts(m, sel).tolist()
    assert set(got) >= {0, 1, 7}
    assert got == sorted(set(got))
    assert el.affected_verts(m, el.empty()).tolist() == []


def test_going_down_is_a_union_and_going_up_is_a_conjunction() -> None:
    m = prim.box()
    face0 = el.ElementSel(faces=[0])
    corners = sorted(m.loops[m.starts[0] : m.starts[1]].tolist())

    verts = el.convert(m, face0, "vertex")
    assert verts.verts.tolist() == corners

    edges = el.convert(m, face0, "edge")
    assert len(edges.edges) == 4, "a quad's four edges, and no others"

    # Back up: those four corners belong to exactly one face all of whose
    # corners are selected. A neighbouring face shares two of them and is not
    # selected, which is the conjunction rule.
    back = el.convert(m, verts, "face")
    assert back.faces.tolist() == [0]

    assert el.convert(m, face0, "object").verts.size == 0


def test_converting_a_partial_vertex_set_selects_no_face() -> None:
    m = prim.box()
    two = el.ElementSel(verts=m.loops[:2])
    assert el.convert(m, two, "face").faces.tolist() == []
    assert el.convert(m, two, "edge").edges.tolist() == [sorted(m.loops[:2].tolist())], (
        "two adjacent verts imply the edge between them"
    )


def test_combine_replaces_adds_and_subtracts() -> None:
    a = el.ElementSel(verts=[1, 2, 3], faces=[0])
    b = el.ElementSel(verts=[3, 4], faces=[1])
    assert el.combine(a, b, "replace") is b
    assert el.combine(a, b, "add").verts.tolist() == [1, 2, 3, 4]
    assert el.combine(a, b, "add").faces.tolist() == [0, 1]
    assert el.combine(a, b, "subtract").verts.tolist() == [1, 2]
    assert el.combine(a, b, "subtract").faces.tolist() == [0]
    with pytest.raises(ValueError):
        el.combine(a, b, "nonsense")  # type: ignore[arg-type]


def test_subtracting_edges_matches_whole_pairs_not_endpoints() -> None:
    a = el.ElementSel(edges=[[0, 1], [1, 2], [2, 3]])
    b = el.ElementSel(edges=[[2, 1]])
    assert el.combine(a, b, "subtract").edges.tolist() == [[0, 1], [2, 3]]
    # Subtracting something that was never selected is a no-op, not an error:
    # a Ctrl-drag marquee sweeps over mostly-unselected geometry.
    assert el.combine(a, el.ElementSel(edges=[[9, 8]]), "subtract").same_as(a)


def test_combine_subtract_of_large_edge_selections_does_not_allocate_quadratically() -> None:
    """The 2026-09-14 audit's clay-03: ``_rows_minus`` broadcast ``a`` against
    ``b`` to a dense ``(len(a), len(b), 2)`` array, so select-all-edges on a
    large import followed by one Ctrl-drag subtract marquee allocated
    gigabytes on the frame thread. 8,000 edges each side is a 128 MB bool
    broadcast on the old code (measured ~190 MB peak); the fixed, key-based
    version never gets near that. A wall-clock bound would be flaky under
    load, so this checks the thing that actually matters -- peak traced
    memory -- instead of timing it.
    """
    import tracemalloc

    n = 8000
    a = np.stack([np.arange(n, dtype="i4"), np.arange(n, dtype="i4") + n], axis=1)
    b = np.stack([np.arange(n, dtype="i4"), np.arange(n, dtype="i4") + n + 1], axis=1)

    tracemalloc.start()
    try:
        out = el._rows_minus(a, b)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 20_000_000, f"peak {peak / 1e6:.1f} MB -- looks like a dense broadcast again"
    assert len(out) == n, "no row of a matches any row of b, so nothing should be dropped"


def test_select_all_and_invert_are_per_mode() -> None:
    m = prim.box()
    assert len(el.select_all(m, "vertex").verts) == len(m.positions)
    assert len(el.select_all(m, "face").faces) == 6
    assert len(el.select_all(m, "edge").edges) == 12
    assert el.select_all(m, "object").verts.size == 0

    some = el.ElementSel(faces=[0, 1])
    assert el.invert(m, some, "face").faces.tolist() == [2, 3, 4, 5]


def test_restrict_drops_what_a_shrunken_mesh_no_longer_has() -> None:
    m = prim.plane()  # 4 verts, 1 face
    wide = el.ElementSel(verts=[0, 9], edges=[[0, 1], [0, 9]], faces=[0, 4])
    got = el.restrict(m, wide)
    assert got.verts.tolist() == [0]
    assert got.edges.tolist() == [[0, 1]]
    assert got.faces.tolist() == [0]


def test_restrict_is_reachable_from_a_live_code_path_or_its_docstring_says_it_is_not() -> None:
    """The 2026-09-08 audit's clay-09: ``restrict``'s docstring described a
    safety net -- guarding the overlay build when an op shrinks a mesh under
    a selection that outlived it -- that no code in ``src/`` actually called;
    only this file's own test above does. Either a live caller exists, or the
    docstring has to say plainly that none does, so a future op that shrinks
    a mesh in place does not assume protection is already wired in.
    """
    import inspect
    import re
    from pathlib import Path

    import warlock

    # P3 of the restructure (dev/RESTRUCTURE.md) moved studio/clay/ -- the
    # one caller this test exists to find, ClayDoc.set_generator_params --
    # to warlock/kernels/mesh/, out from under studio/. Scanning the whole
    # ``warlock`` package rather than just ``studio/`` is what keeps this
    # test meaningful regardless of which side of that boundary a future
    # caller lands on.
    root = Path(warlock.__file__).parent
    callers = [
        path
        for path in root.rglob("*.py")
        if path.name != "elements.py"
        and re.search(r"\brestrict\s*\(", path.read_text(encoding="utf-8"))
    ]
    if callers:
        return  # a live caller exists -- nothing more to prove

    doc = inspect.getdoc(el.restrict) or ""
    assert "not currently called" in doc.lower(), (
        "restrict() has no live caller anywhere under warlock/, but its "
        "docstring no longer admits that -- either wire it into the caller "
        "that should use it, or restore the honest docstring"
    )


def test_op_error_is_a_value_error_so_a_forgotten_catch_still_fails_loudly() -> None:
    assert issubclass(el.OpError, ValueError)


def test_every_refusal_reads_like_a_sentence() -> None:
    """A refusal is the whole user interface for an op that cannot run.

    It is shown as a toast and nothing else happens, so the message has to say
    what was refused and -- wherever there is one -- what to do instead. The
    static half of that is checkable: every ``raise OpError`` site's literal
    text starts a sentence and ends one.
    """
    import ast
    from pathlib import Path

    import warlock.kernels.mesh as package

    root = Path(package.__file__).parent
    checked = 0
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)):
                continue
            name = getattr(node.exc.func, "id", "")
            if name != "OpError" or not node.exc.args:
                continue
            text = _literal_text(node.exc.args[0])
            if not text:
                continue
            checked += 1
            where = f"{path.name}:{node.lineno}"
            assert text[0].isupper() or text[0] == "{", f"{where}: {text!r}"
            # A trailing hole is a message that ends in an interpolated value
            # -- a wrapped exception's own text, which brings its own full stop.
            assert text.rstrip().endswith((".", "?", "{}")), f"{where}: {text!r}"
    assert checked > 20, "the sweep found suspiciously few refusals"


def _literal_text(node: object) -> str:
    """The constant text of a string or f-string expression, holes elided."""
    import ast

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value if isinstance(part, ast.Constant) else "{}"
            for part in node.values
        )
    return ""
