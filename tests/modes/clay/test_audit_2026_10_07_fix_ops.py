"""Regression tests for the 2026-10-07 Clay audit's ops-registry findings
(clay-08, 10, 11 [the ``run_mesh_op`` half], 41, 42, 43, 44, 45).

Each test name is the claim; each fails against the registry as it stood before
the fix.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops


class _Ctx:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.info: list[str] = []

    def toast(self, message: str, level: str = "info") -> None:
        (self.errors if level == "error" else self.info).append(message)


def _box(doc: bd.ClayDoc, name: str = "Box", **kw: Any) -> bd.Obj:
    return doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name=name,
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
            **kw,
        )
    )


def _names(doc: bd.ClayDoc) -> list[str]:
    return [o.name for o in doc.objects]


# --- clay-08 ------------------------------------------------------------------


def _grouped_doc() -> tuple[bd.ClayDoc, dict[str, bd.Obj]]:
    """Document order A, B, Q, Group; the outliner draws Q, Group, A, B."""
    doc = bd.ClayDoc()
    objs = {n: _box(doc, n) for n in ("A", "B", "Q")}
    doc.select([objs["A"].uid, objs["B"].uid])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("group"))
    objs["Group"] = next(o for o in doc.objects if o.name == "Group")
    return doc, objs


def test_merge_survivor_is_the_selected_object_the_outliner_draws_first() -> None:
    doc, objs = _grouped_doc()
    doc.select([objs["A"].uid, objs["Q"].uid])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("join"))
    # Q is a root drawn above Group's subtree (which holds A); A is document-first.
    assert "Q" in _names(doc) and "A" not in _names(doc)
    assert doc.selection == {objs["Q"].uid}


def test_parent_to_last_targets_the_object_the_outliner_draws_first() -> None:
    doc, objs = _grouped_doc()
    doc.select([objs["A"].uid, objs["Q"].uid])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("parent-to-last"))
    assert doc.by_uid(objs["A"].uid).parent == objs["Q"].uid
    assert doc.by_uid(objs["Q"].uid).parent is None


def test_parent_to_last_of_a_group_and_its_child_is_not_a_cycle_refusal() -> None:
    """The audit's reproduction: Group + A. The outliner draws Group first, so A
    (already its child) is the thing being parented, onto its own parent. Document
    order made Group the thing being parented onto its own child -- a toast."""
    doc, objs = _grouped_doc()
    doc.select([objs["Group"].uid, objs["A"].uid])
    ctx = _Ctx()
    head = doc.history.head
    assert clay_ops.run(ctx, doc, clay_ops.get("parent-to-last")) is False
    assert ctx.errors == []
    assert doc.history.head == head


def test_outliner_order_matches_the_outliner_pane_row_for_row() -> None:
    """The ops and the pane draw one order or the manual's "topmost in the
    outliner" is two different sentences."""
    from realmspinner.studio.modes.clay.ui.panes import outliner

    doc = bd.ClayDoc()
    a = _box(doc, "A")
    b = _box(doc, "B")
    c = _box(doc, "C")
    d = _box(doc, "D")
    e = _box(doc, "E")
    f = _box(doc, "F")
    # Interleave parents so document order and tree order differ at every level.
    doc.set_parent(a.uid, d.uid, keep_world=True)
    doc.set_parent(c.uid, d.uid, keep_world=True)
    doc.set_parent(b.uid, a.uid, keep_world=True)
    doc.set_parent(f.uid, e.uid, keep_world=True)
    doc.set_parent(e.uid, b.uid, keep_world=True)
    assert [o.uid for o in doc.objects] != [o.uid for o, _, _ in outliner._tree_rows(doc)]
    assert clay_ops.outliner_order(doc) == [o.uid for o, _, _ in outliner._tree_rows(doc)]


def test_outliner_order_keeps_an_object_the_tree_walk_cannot_reach() -> None:
    """An object whose parent chain never reaches a root (a hand-built cycle)
    must still rank somewhere, so a selected one is never dropped from a merge."""
    doc = bd.ClayDoc()
    a = _box(doc, "A")
    b = _box(doc, "B")
    a.parent, b.parent = b.uid, a.uid
    c = _box(doc, "C")
    assert clay_ops.outliner_order(doc) == [c.uid, a.uid, b.uid]


# --- clay-10 ------------------------------------------------------------------


def _doc_in_mode(mode: str) -> bd.ClayDoc:
    doc = bd.ClayDoc()
    a = _box(doc, "A")
    b = _box(doc, "B")
    doc.set_transform(b.uid, translation=np.array([3.0, 0.0, 0.0]))
    if mode == "object":
        doc.select([a.uid, b.uid])
        return doc
    doc.set_element_mode(mode)
    for obj in (a, b):
        kind = {"vertex": "verts", "edge": "edges", "face": "faces"}[mode]
        doc.set_element_sel(obj.uid, el.select_all(obj.mesh, mode))
        assert getattr(doc.element_sel_of(obj.uid), kind).size
    return doc


def _fingerprint(doc: bd.ClayDoc) -> tuple[Any, ...]:
    return (
        doc.history.head,
        doc.element_mode,
        tuple((o.uid, id(o.mesh), tuple(o.translation), o.parent, o.name) for o in doc.objects),
        frozenset(doc.selection),
        {uid: id(sel) for uid, sel in doc.element_sel.items()}.keys(),
    )


def test_run_refuses_an_object_level_op_in_an_element_mode() -> None:
    doc = _doc_in_mode("face")
    ctx = _Ctx()
    before = _fingerprint(doc)
    assert clay_ops.run(ctx, doc, clay_ops.get("join")) is False
    assert ctx.errors == ["Switch to object mode first."]
    assert _fingerprint(doc) == before
    assert doc.element_sel, "the element selection survived the refusal"


def test_every_op_run_in_a_mode_it_does_not_declare_refuses_and_changes_nothing() -> None:
    covered = 0
    for op in clay_ops.OPS:
        for mode in clay_ops.ALL_MODES:
            if mode in op.modes:
                continue
            doc = _doc_in_mode(mode)
            ctx = _Ctx()
            before = _fingerprint(doc)
            ran = clay_ops.run(ctx, doc, op)
            assert ran is False, f"{op.name} ran in {mode}"
            assert len(ctx.errors) == 1, (op.name, mode, ctx.errors)
            assert "mode first." in ctx.errors[0], (op.name, mode, ctx.errors)
            assert _fingerprint(doc) == before, f"{op.name} changed the document in {mode}"
            covered += 1
    assert covered > 30


def test_an_op_in_a_declared_mode_is_not_blocked_by_the_mode_check() -> None:
    """The check must be the declared set, not "object only": Shade works in
    object and face mode, Select All in all four."""
    doc = _doc_in_mode("face")
    ctx = _Ctx()
    clay_ops.run(ctx, doc, clay_ops.get("select-all"))  # ALL_MODES: no refusal
    assert ctx.errors == []
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("shade-smooth")) is True
    assert ctx.errors == []


def test_the_refusal_sentence_names_every_mode_the_op_works_in() -> None:
    doc = _doc_in_mode("object")
    ctx = _Ctx()
    clay_ops.run(ctx, doc, clay_ops.get("extrude"))
    assert ctx.errors == ["Switch to vertex, edge or face mode first."]
    doc = _doc_in_mode("vertex")
    ctx = _Ctx()
    clay_ops.run(ctx, doc, clay_ops.get("shade-smooth"))
    assert ctx.errors == ["Switch to object or face mode first."]
    ctx = _Ctx()
    clay_ops.run(ctx, doc, clay_ops.get("inset"))
    assert ctx.errors == ["Switch to face mode first."]


# --- clay-11 (run_mesh_op) ----------------------------------------------------


def test_an_element_op_leaves_a_hidden_object_with_a_stale_selection_untouched() -> None:
    doc = bd.ClayDoc()
    a = _box(doc, "a")
    b = _box(doc, "b")
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, el.select_all(a.mesh, "face"))
    doc.set_element_sel(b.uid, el.select_all(b.mesh, "face"))
    doc.set_visibility({b.uid: False})  # what H does; the element selection stays
    hidden_mesh = doc.by_uid(b.uid).mesh
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("inset"), thickness=0.1)
    assert doc.by_uid(b.uid).mesh is hidden_mesh, "a hidden object was rewritten"
    assert len(doc.by_uid(a.uid).mesh.positions) > 8, "the visible one was inset"


def test_an_element_op_with_only_hidden_objects_selected_reports_it_did_not_run() -> None:
    doc = bd.ClayDoc()
    a = _box(doc, "a")
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, el.select_all(a.mesh, "face"))
    doc.set_visibility({a.uid: False})
    head = doc.history.head
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("inset"), thickness=0.1) is False
    assert doc.history.head == head


# --- clay-41 ------------------------------------------------------------------


def test_run_reports_false_when_clear_parent_changed_nothing() -> None:
    doc = bd.ClayDoc()
    a = _box(doc)
    doc.select([a.uid])
    head = doc.history.head
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("clear-parent")) is False
    assert doc.history.head == head


def test_run_reports_true_when_clear_parent_released_a_child() -> None:
    doc = bd.ClayDoc()
    a = _box(doc)
    b = _box(doc, "B")
    doc.set_parent(b.uid, a.uid, keep_world=True)
    doc.select([b.uid])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("clear-parent")) is True
    assert doc.by_uid(b.uid).parent is None


@pytest.mark.parametrize(
    "name",
    ["recalc-normals", "snap-to-grid", "origin-to-bounds", "origin-to-base", "origin-to-world"],
)
def test_run_reports_false_for_a_no_op_object_op(name: str) -> None:
    doc = bd.ClayDoc()
    a = _box(doc)
    if name == "origin-to-base":
        doc.set_origin(a.uid, [0.0, -0.5, 0.0])  # the pivot is already at the base
    doc.select([a.uid])
    head = doc.history.head
    assert clay_ops.run(_Ctx(), doc, clay_ops.get(name)) is False, name
    assert doc.history.head == head, name


def test_run_reports_false_for_origin_to_bounds_on_an_empty_group() -> None:
    doc = bd.ClayDoc()
    a = _box(doc)
    b = _box(doc, "B")
    doc.select([a.uid, b.uid])
    clay_ops.run(_Ctx(), doc, clay_ops.get("group"))
    group = next(o for o in doc.objects if len(o.mesh.positions) == 0)
    doc.select([group.uid])
    head = doc.history.head
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("origin-to-bounds")) is False
    assert doc.history.head == head


def test_run_reports_false_for_origin_to_selection_on_an_empty_pick() -> None:
    doc = bd.ClayDoc()
    a = _box(doc)
    doc.set_element_mode("vertex")
    doc.set_element_sel(a.uid, el.ElementSel(verts=[0, 1, 2, 3]))
    doc.set_origin(a.uid, list(np.asarray(a.mesh.positions)[:4].mean(axis=0)))
    head = doc.history.head
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("origin-to-selection")) is False
    assert doc.history.head == head


def test_run_still_reports_true_when_a_snap_moved_an_object() -> None:
    doc = bd.ClayDoc()
    a = _box(doc)
    doc.set_transform(a.uid, translation=np.array([0.4, 0.0, 0.0]))
    doc.select([a.uid])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("snap-to-grid"), step=1.0) is True
    assert tuple(doc.by_uid(a.uid).translation) == (0.0, 0.0, 0.0)


def test_frame_reports_false_when_there_is_no_view_to_frame() -> None:
    doc = bd.ClayDoc()
    a = _box(doc)
    doc.select([a.uid])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("frame")) is False

    class _View:
        framed = 0

        def frame_selection(self, doc: Any) -> None:
            _View.framed += 1

    ctx = _Ctx()
    ctx.clay_view = _View()  # type: ignore[attr-defined]
    assert clay_ops.run(ctx, doc, clay_ops.get("frame")) is True
    assert _View.framed == 1


# --- clay-42 ------------------------------------------------------------------


@pytest.mark.parametrize("name", ["extrude", "select-linked", "origin-to-selection"])
def test_extrude_in_object_mode_names_the_mode_it_needs(name: str) -> None:
    doc = bd.ClayDoc()
    a = _box(doc)
    doc.select([a.uid])
    op = clay_ops.get(name)
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc) == "Switch to vertex, edge or face mode first."


@pytest.mark.parametrize("name", ["extrude", "select-linked", "origin-to-selection"])
def test_an_element_op_with_no_selection_in_an_element_mode_says_to_select(name: str) -> None:
    doc = bd.ClayDoc()
    _box(doc)
    doc.set_element_mode("face")
    assert clay_ops.reason_for(clay_ops.get(name), doc) == "Select something in the viewport first."


# --- clay-43 ------------------------------------------------------------------


def test_weld_refuses_a_distance_that_would_collapse_the_whole_mesh() -> None:
    doc = bd.ClayDoc()
    a = _box(doc)
    doc.set_element_mode("vertex")
    doc.set_element_sel(a.uid, el.select_all(a.mesh, "vertex"))
    before = doc.by_uid(a.uid).mesh
    head = doc.history.head
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("weld"), eps=5.0) is False
    assert len(ctx.errors) == 1
    assert "5 m" in ctx.errors[0] and "every face" in ctx.errors[0]
    assert doc.by_uid(a.uid).mesh is before
    assert doc.history.head == head


def test_weld_distance_is_bounded_to_a_few_tens_of_metres() -> None:
    doc = bd.ClayDoc()
    a = _box(doc)
    doc.set_element_mode("vertex")
    doc.set_element_sel(a.uid, el.select_all(a.mesh, "vertex"))
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("weld"), eps=1e6) is False
    assert bm.face_count(doc.by_uid(a.uid).mesh) == 6, "the box survived a 1e6 m weld"
    for name, param in (("weld", "eps"), ("join", "weld")):
        spec = next(p for p in clay_ops.get(name).params if p.name == param)
        assert spec.high <= 100.0, (name, spec.high)


def test_a_sensible_weld_still_welds() -> None:
    doc = bd.ClayDoc()
    a = _box(doc)
    doc.set_element_mode("vertex")
    doc.set_element_sel(a.uid, el.select_all(a.mesh, "vertex"))
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("weld"), eps=1e-4) is True


def test_merge_objects_refuses_a_weld_that_would_collapse_every_face() -> None:
    doc = bd.ClayDoc()
    a = _box(doc, "A")
    b = _box(doc, "B")
    doc.set_transform(b.uid, translation=np.array([2.0, 0.0, 0.0]))
    doc.select([a.uid, b.uid])
    ctx = _Ctx()
    head = doc.history.head
    assert clay_ops.run(ctx, doc, clay_ops.get("join"), weld=50.0) is False
    assert len(ctx.errors) == 1 and "50 m" in ctx.errors[0]
    assert _names(doc) == ["A", "B"]
    assert doc.history.head == head


# --- clay-44 ------------------------------------------------------------------


def test_the_snap_hint_says_world_position_and_the_param_docstring_is_current() -> None:
    hint = clay_ops.get("snap-to-grid").hint
    assert "world position" in hint and "translation" not in hint
    doc = clay_ops.Param.__doc__ or ""
    assert "is the one that is left" not in doc
    axis = next(p for p in clay_ops.get("mirror-copy").params if p.name == "axis")
    assert axis.choices == ("X", "Y", "Z")


# --- clay-45 ------------------------------------------------------------------

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}
#: The one function-scope reach into a sibling mode (Drop to Ground borrows
#: Mason's pure box arithmetic). Level 2 from ``studio/modes/clay/ops.py`` is
#: ``studio/modes``.
ALLOWED_SIBLING_REACH = {"mason.engine"}


def _import_violations(source: str) -> list[str]:
    bad: list[str] = []
    siblings: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BANNED_ROOTS or alias.name.startswith("realmspinner.service"):
                    bad.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0:
                root = module.split(".")[0]
                if root in BANNED_ROOTS or module.startswith("realmspinner.service"):
                    bad.append(module)
            else:
                names = {f"{module}.{a.name}".strip(".") for a in node.names}
                if module.split(".")[0] == "service" or any(
                    n.split(".")[0] == "service" for n in names
                ):
                    bad.append(f"{'.' * node.level}{module}")
                if node.level == 2 and module:
                    siblings.add(module)
    if siblings != ALLOWED_SIBLING_REACH:
        bad.append(f"sibling reach {sorted(siblings)}")
    return bad


def test_the_registry_imports_no_pygame_moderngl_or_service() -> None:
    source = Path(inspect.getsourcefile(clay_ops) or "").read_text(encoding="utf-8")
    assert _import_violations(source) == []


@pytest.mark.parametrize(
    "line",
    [
        "import pygame",
        "from moderngl import Context",
        "def f():\n    import moderngl",
        "from ....service import errors",
        "from .... import service",
        "from realmspinner.service import errors",
        "from ..inker import ops",
    ],
)
def test_the_registry_import_check_catches_what_it_claims_to(line: str) -> None:
    source = "from ..mason.engine import ops\n" + line + "\n"
    assert _import_violations(source), line
