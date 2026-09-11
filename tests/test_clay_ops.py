"""The Clay ops registry: one list, three surfaces, no imgui.

The registry exists because there used to be three lists -- the tools pane, the
key handler and (now) the context menu -- and the interesting one was always the
one nobody updated. So the assertions here are mostly about *agreement*: that a
key fires the op the menu shows for it, that enablement is one predicate, and
that nothing here reaches for a GUI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from warlock.studio import clay_ops
from warlock.studio.clay import document as bd
from warlock.studio.clay import elements as el
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import ops_topo
from warlock.studio.clay import primitives as bp


class _Toasts:
    """Only what ``Ctx`` really offers. The double used to carry an ``error``
    method under a ``ctx.toasts`` attribute the app has never had, which is
    exactly how every refusal passed here and vanished in the real app."""

    def __init__(self) -> None:
        self.errors: list[str] = []


class _Ctx:
    def __init__(self) -> None:
        self.toasts = _Toasts()

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.toasts.errors.append(message)


def _doc() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Box",
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
        )
    )
    return doc, obj.uid


def _faces(doc: bd.ClayDoc, uid: int, *faces: int) -> None:
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=list(faces)))


# --- the registry itself ----------------------------------------------------


def test_every_op_has_a_unique_name_and_at_least_one_mode() -> None:
    names = [op.name for op in clay_ops.OPS]
    assert len(names) == len(set(names))
    for op in clay_ops.OPS:
        assert op.modes
        assert set(op.modes) <= set(clay_ops.ALL_MODES)


def test_registering_a_duplicate_name_is_refused() -> None:
    existing = clay_ops.OPS[0]
    with pytest.raises(ValueError, match="already registered"):
        clay_ops.register(
            clay_ops.Op(name=existing.name, label="x", modes=("object",), run=lambda *_: None)
        )


def test_the_menu_is_filtered_by_mode_and_keeps_registration_order() -> None:
    face = [op.name for op in clay_ops.menu("face")]
    assert "extrude" in face
    assert "bevel" not in face, "bevel is an edge op"
    assert face == [op.name for op in clay_ops.OPS if "face" in op.modes]

    edges = [op.name for op in clay_ops.menu("edge")]
    assert "bevel" in edges and "loop-cut" in edges
    # Extrude is now one row across all three modes, dispatched on the mode the
    # way Dissolve is: it means the same thing everywhere and only the
    # implementation differs, so three rows would be exposing that.
    assert "extrude" in edges
    assert "bridge" in edges
    assert "bridge" not in face, "bridge joins two boundary loops, which is an edge selection"


def test_a_key_resolves_to_the_op_the_menu_shows_for_it() -> None:
    op = clay_ops.by_key("face", "E")
    assert op is not None and op.name == "extrude"
    # The same op, and therefore the same key, in every element mode.
    assert clay_ops.by_key("edge", "E") is op
    assert clay_ops.by_key("vertex", "E") is op
    assert clay_ops.by_key("face", "nope") is None


def test_the_registry_imports_no_gui() -> None:
    """``clay_ops`` is testable precisely because it draws nothing.

    The layering rule for the whole package: nothing under ``clay/`` and nothing
    in the registry knows imgui exists; only panes and ``main`` draw.
    """
    import ast
    import importlib
    from pathlib import Path

    source = importlib.import_module("warlock.studio.clay_ops").__file__
    assert source is not None
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "imgui_bundle" not in imported
    assert "imgui" not in imported


# --- enablement -------------------------------------------------------------


def test_an_element_op_is_disabled_with_nothing_selected() -> None:
    doc, uid = _doc()
    doc.set_element_mode("face")
    extrude = clay_ops.get("extrude")
    assert not extrude.enabled(doc)
    _faces(doc, uid, 0)
    assert extrude.enabled(doc)


def test_an_op_that_is_disabled_does_not_run() -> None:
    doc, _ = _doc()
    doc.set_element_mode("face")
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("extrude")) is False
    assert len(doc.history) == 1, "the add, and nothing else"


def test_object_ops_need_an_object_selection() -> None:
    doc, uid = _doc()
    assert not clay_ops.get("duplicate").enabled(doc)
    doc.select([uid])
    assert clay_ops.get("duplicate").enabled(doc)


# --- reasons (clay-07, 2026-09-06 audit) -------------------------------------
#
# The ``Op`` dataclass carried no explanation at all for a refusal: none of the
# three surfaces that grey a row -- the context menu, the tools-pane buttons,
# the Delete button -- passed anything to the ``reason``/``tooltip`` argument
# the widgets already accept, so Merge Objects, Bridge Loops and every element
# op (Bevel, Inset, Weld...) greyed out with nothing on screen saying why.


def test_every_disabled_clay_op_names_the_gate_that_refused_it() -> None:
    """A fresh, empty document refuses every op that has a non-default
    ``enabled`` at once -- no objects, object mode, nothing selected -- which
    is what makes this a sweep across the whole registry rather than one test
    per op. The four ops with no gate at all (``select-all``, ``select-invert``,
    ``select-boundary``, ``frame``) are the only ones this document does not
    refuse, and are checked for that instead.
    """
    doc = bd.ClayDoc()
    always_enabled = {"select-all", "select-invert", "select-boundary", "frame"}
    gated = 0
    for op in clay_ops.OPS:
        if op.name in always_enabled:
            assert op.enabled(doc), f"{op.name}: expected to have no gate"
            continue
        assert not op.enabled(doc), f"{op.name}: expected refused on an empty document"
        assert clay_ops.reason_for(op, doc), f"{op.name}: refused with no reason"
        gated += 1
    # Guards the sweep itself: a registry that grew no gated ops at all would
    # let every assertion above pass on an empty loop.
    assert gated >= 25


def test_a_reason_clears_the_moment_its_own_gate_passes() -> None:
    """The sentence must track ``enabled`` rather than drift from it -- picked
    across the shapes ``reason_for`` covers: an object selection, two visible
    objects, an element mode, and an element selection."""
    doc, uid = _doc()
    duplicate = clay_ops.get("duplicate")
    assert clay_ops.reason_for(duplicate, doc) == "Select an object first."
    doc.select([uid])
    assert clay_ops.reason_for(duplicate, doc) == ""

    doc, first = _doc()
    second = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box())).uid
    join = clay_ops.get("join")
    assert clay_ops.reason_for(join, doc) == "Select two visible objects first."
    doc.select([first, second])
    assert clay_ops.reason_for(join, doc) == ""

    doc, uid = _doc()
    bridge = clay_ops.get("bridge")
    assert "Switch to edge mode" in clay_ops.reason_for(bridge, doc)
    doc.set_element_mode("edge")
    assert clay_ops.reason_for(bridge, doc) == "Select something first."
    a = doc.by_uid(uid).mesh
    doc.set_element_sel(uid, el.ElementSel(edges=[[int(a.loops[0]), int(a.loops[1])]]))
    assert clay_ops.reason_for(bridge, doc) == ""


# --- running ----------------------------------------------------------------


def test_running_extrude_edits_the_mesh_and_selects_the_caps() -> None:
    doc, uid = _doc()
    _faces(doc, uid, 0)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("extrude")) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 10
    assert doc.element_sel_of(uid).faces.tolist() == [0]


def test_a_topology_op_freezes_the_generator_exactly_once() -> None:
    doc, uid = _doc()
    _faces(doc, uid, 0)
    assert doc.by_uid(uid).generator == "box"
    clay_ops.run(_Ctx(), doc, clay_ops.get("extrude"))
    assert doc.by_uid(uid).generator is None
    assert doc.by_uid(uid).params == {}

    depth = len(doc.history)
    _faces(doc, uid, 1)
    clay_ops.run(_Ctx(), doc, clay_ops.get("extrude"))
    assert len(doc.history) == depth + 1, "already frozen: one mesh step, no props step"


@pytest.mark.parametrize("key", ["bake", "mirror-x", "mirror-y", "mirror-z"])
def test_every_op_that_changes_geometry_freezes_it(key: str) -> None:
    """The regression: the freeze lived in ``run_mesh_op`` and Smooth, so Bake
    Transform and Mirror went straight to ``set_mesh`` and left the object
    still claiming to be "box, size 1". The properties panel keeps offering
    that size field, and touching it rebuilds a pristine box -- so the bake or
    the mirror vanished with no warning.
    """
    doc, uid = _doc()
    doc.select([uid])
    doc.set_transform(uid, translation=[1.0, 2.0, 3.0])

    clay_ops.run(_Ctx(), doc, clay_ops.get(key))

    assert doc.by_uid(uid).generator is None
    assert doc.by_uid(uid).params == {}


def test_bake_transform_undoes_the_mesh_and_the_transform_in_one_step() -> None:
    """clay-11 (2026-09-06 audit): ``_bake``'s own docstring claimed "an undo
    of a bake is two presses rather than a compound type that exists for one
    button" -- the opposite of what ``run`` has always actually done here.
    ``run``'s ``_one_step`` folds everything an op pushes (the mesh and the
    transform, for Bake) into one history entry for every op without
    exception, so a bake was already a single compound undo; only the
    docstring disagreed with the code beside it.
    """
    import inspect

    doc_text = inspect.getdoc(clay_ops._bake) or ""
    assert "two presses" not in doc_text, "docstring still claims the fold skips bake"
    assert "one" in doc_text.lower() and "compound" in doc_text.lower()

    doc, uid = _doc()
    doc.select([uid])
    doc.set_transform(uid, translation=[1.0, 2.0, 3.0])
    depth = len(doc.history)

    clay_ops.run(_Ctx(), doc, clay_ops.get("bake"))

    assert len(doc.history) == depth + 1, "one history entry, not two"
    assert doc.undo() is True
    obj = doc.by_uid(uid)
    assert np.allclose(obj.translation, [1.0, 2.0, 3.0]), "the transform came back"
    assert obj.generator == "box", "and the mesh's generator claim came back with it"


def test_deleting_elements_freezes_the_generator() -> None:
    """The same hole reached from ``clay_mode._delete`` rather than an op: it
    calls ``doc.set_mesh`` directly, and nothing there used to freeze."""
    doc, uid = _doc()
    _faces(doc, uid, 0)
    before = bm.face_count(doc.by_uid(uid).mesh)

    mesh, sel = ops_topo.delete_faces(doc.by_uid(uid).mesh, doc.element_sel_of(uid))
    doc.set_mesh(uid, mesh, select=sel)

    assert bm.face_count(doc.by_uid(uid).mesh) < before
    assert doc.by_uid(uid).generator is None


def test_one_undo_takes_back_the_edit_and_its_freeze_together() -> None:
    """The freeze was a second history step, so a single Ctrl+Z restored the
    generator claim over the still-edited mesh -- the exact state the freeze
    exists to prevent -- and only a second press undid the edit it belongs to.
    Touching the size field in between rebuilt a pristine box over the edit."""
    doc, uid = _doc()
    _faces(doc, uid, 0)
    mesh, sel = ops_topo.delete_faces(doc.by_uid(uid).mesh, doc.element_sel_of(uid))
    original = doc.by_uid(uid).mesh
    doc.set_mesh(uid, mesh, select=sel)

    assert doc.undo() is True

    obj = doc.by_uid(uid)
    assert obj.mesh is original
    assert obj.generator == "box"
    assert obj.params == {"size": (1.0, 1.0, 1.0)}


def test_a_rebuild_from_the_generator_is_the_one_thing_that_keeps_it() -> None:
    """Otherwise editing a box's size would freeze it on the first keystroke
    and the field would disappear under the user's hands."""
    doc, uid = _doc()

    doc.set_mesh(uid, bp.box(size=(2.0, 2.0, 2.0)), keep_generator=True)

    assert doc.by_uid(uid).generator == "box"


def test_a_refusal_becomes_a_toast_and_records_no_edit() -> None:
    doc, uid = _doc()
    doc.set_element_mode("edge")
    a = doc.by_uid(uid).mesh
    doc.set_element_sel(uid, el.ElementSel(edges=[[int(a.loops[0]), int(a.loops[1])]]))
    ctx = _Ctx()
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("fill-hole")) is False
    assert ctx.toasts.errors and "boundary" in ctx.toasts.errors[0]
    assert len(doc.history) == depth


def test_a_refusal_on_one_object_does_not_abandon_the_others() -> None:
    doc, first = _doc()
    second = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box())).uid
    doc.set_element_mode("face")
    doc.set_element_sel(first, el.ElementSel(faces=[0]))
    doc.set_element_sel(second, el.ElementSel(faces=[0]))
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("extrude")) is True
    assert bm.face_count(doc.by_uid(first).mesh) == 10
    assert bm.face_count(doc.by_uid(second).mesh) == 10


def test_parameters_fall_back_to_their_declared_defaults() -> None:
    doc, uid = _doc()
    _faces(doc, uid, 0)
    inset = clay_ops.get("inset")
    assert clay_ops.defaults_for(inset) == {"thickness": 0.1, "depth": 0.0}
    assert clay_ops.run(_Ctx(), doc, inset) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 10


def test_a_parameter_passed_in_overrides_the_default() -> None:
    doc, uid = _doc()
    _faces(doc, uid, 0)
    clay_ops.run(_Ctx(), doc, clay_ops.get("inset"), thickness=99.0)
    inner = doc.by_uid(uid).mesh
    corners = inner.positions[inner.loops[inner.starts[0] : inner.starts[1]]]
    assert np.allclose(corners, corners[0]), "collapsed onto the centroid"


def test_a_parameter_out_of_range_is_clamped_to_what_it_declared() -> None:
    """``run`` is the choke point, so the declared range holds on every surface.

    The popup clamps its own live fields, but the key path, the tools pane and
    a remembered value from a previous session all arrive here instead -- and a
    subdivision at 99 levels is not something each op should have to refuse for
    itself. An integer parameter also comes out an ``int``, so an op can index
    with it.
    """
    seen: dict[str, Any] = {}

    def record(ctx: Any, doc: Any, **params: Any) -> None:
        del ctx, doc
        seen.update(params)

    # Unregistered: the registry is global, and a probe op that joined it would
    # be there for every later test in the process.
    probe = clay_ops.Op(
        name="probe-clamp",
        label="Probe",
        modes=("object",),
        run=record,
        params=(
            clay_ops.Param("t", "position", 0.5, 0.05, low=0.0, high=1.0),
            clay_ops.Param("levels", "levels", 1.0, 1.0, low=1.0, high=4.0, integer=True),
        ),
    )

    assert clay_ops.run(_Ctx(), _doc()[0], probe, t=5.0, levels=99.0) is True
    assert seen == {"t": 1.0, "levels": 4}
    assert isinstance(seen["levels"], int)


def test_a_parameter_below_its_floor_is_clamped_up() -> None:
    seen: dict[str, Any] = {}

    def record(ctx: Any, doc: Any, **params: Any) -> None:
        del ctx, doc
        seen.update(params)

    probe = clay_ops.Op(
        name="probe-floor",
        label="Probe",
        modes=("object",),
        run=record,
        params=(clay_ops.Param("t", "position", 0.5, 0.05, low=0.25, high=1.0),),
    )

    clay_ops.run(_Ctx(), _doc()[0], probe, t=-3.0)
    assert seen == {"t": 0.25}


def test_dissolve_dispatches_on_the_mode() -> None:
    doc, uid = _doc()
    doc.set_element_mode("vertex")
    doc.set_element_sel(uid, el.ElementSel(verts=[0]))
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("dissolve")) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 4


def test_merge_faces_is_dissolve_under_the_name_a_user_looks_for() -> None:
    """Two adjacent faces of a box become one, leaving five."""
    doc, uid = _doc()
    _faces(doc, uid, 0, 2)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("merge_faces")) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 5


def test_merge_faces_offers_itself_only_in_face_mode() -> None:
    """Vertex and edge selections have their own dissolve; a Merge Faces row
    over an edge selection would be a third name for the same dispatch."""
    assert "merge_faces" in [op.name for op in clay_ops.menu("face")]
    assert "merge_faces" not in [op.name for op in clay_ops.menu("edge")]
    assert "merge_faces" not in [op.name for op in clay_ops.menu("vertex")]
    assert "merge_faces" not in [op.name for op in clay_ops.menu("object")]


def test_merge_faces_needs_a_face_selection() -> None:
    doc, uid = _doc()
    doc.set_element_mode("face")
    merge = clay_ops.get("merge_faces")
    assert not merge.enabled(doc)
    _faces(doc, uid, 0, 2)
    assert merge.enabled(doc)


def test_merge_faces_and_dissolve_agree_in_face_mode() -> None:
    """The whole licence for the second name: it must stay the same operation.

    Run on two documents that started identical, the two ops have to produce
    the same face count -- if they ever diverge, one of them has become a
    second implementation and the manual's claim that they are one thing is a
    lie the user finds out about by undoing the wrong result.
    """
    counts = []
    for name in ("merge_faces", "dissolve"):
        doc, uid = _doc()
        _faces(doc, uid, 0, 2)
        assert clay_ops.run(_Ctx(), doc, clay_ops.get(name)) is True
        counts.append(bm.face_count(doc.by_uid(uid).mesh))
    assert counts[0] == counts[1]


def test_merge_faces_refuses_a_selection_it_cannot_represent() -> None:
    """Opposite faces of a box touch nowhere, so there is no single n-gon to
    make. The refusal is a toast and the mesh is left alone."""
    doc, uid = _doc()
    before = bm.face_count(doc.by_uid(uid).mesh)
    _faces(doc, uid, 0, 1)
    ctx = _Ctx()
    clay_ops.run(ctx, doc, clay_ops.get("merge_faces"))
    assert bm.face_count(doc.by_uid(uid).mesh) == before


def test_smooth_ignores_the_element_selection_and_takes_whole_objects() -> None:
    doc, uid = _doc()
    _faces(doc, uid, 0)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("smooth"), levels=1) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 24


def test_delete_in_face_mode_removes_faces_rather_than_the_object() -> None:
    doc, uid = _doc()
    _faces(doc, uid, 0)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("delete")) is True
    assert len(doc.objects) == 1, "the object survives"
    assert bm.face_count(doc.by_uid(uid).mesh) == 5


def test_delete_in_object_mode_removes_the_object() -> None:
    doc, uid = _doc()
    doc.select([uid])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("delete")) is True
    assert doc.objects == []


def test_mirror_bakes_into_the_mesh_and_leaves_the_scale_positive() -> None:
    doc, uid = _doc()
    doc.select([uid])
    clay_ops.run(_Ctx(), doc, clay_ops.get("mirror-x"))
    assert np.allclose(doc.by_uid(uid).scale, [1.0, 1.0, 1.0])
    # Two steps: the transform, and the mesh change with its freeze compounded
    # into it. The freeze used to be a third, so one Ctrl+Z put the generator
    # claim back over the mirrored mesh.
    assert len(doc.history) == 2
    assert doc.by_uid(uid).generator is None


def _three_boxes_selected() -> tuple[bd.ClayDoc, list[int]]:
    doc, first = _doc()
    uids = [first]
    for name in ("Box.001", "Box.002"):
        obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box()))
        uids.append(obj.uid)
    doc.select(uids)
    return doc, uids


# --- array-linear, array-radial, mirror-copy (repetition and placement) -----


def test_array_linear_of_three_objects_five_times_is_one_undo_step() -> None:
    """clay-01's reason (2026-09-06 audit), at more scale: fifteen inserts
    must not be fifteen presses of Ctrl+Z to undo one keypress that made
    them."""
    doc, uids = _three_boxes_selected()
    depth = len(doc.history)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("array-linear"), count=5, x=1.0) is True

    assert len(doc.objects) == 3 + 3 * 4
    assert len(doc.history) == depth + 1
    assert doc.undo() is True
    assert len(doc.objects) == 3


def test_array_linears_copies_share_the_source_mesh() -> None:
    """``ops.duplicate``'s own property, carried straight through ``translated``:
    an array of many copies is one GPU upload, not one per copy."""
    doc, uid = _doc()
    doc.select([uid])
    source = doc.by_uid(uid).mesh

    clay_ops.run(_Ctx(), doc, clay_ops.get("array-linear"), count=4, x=1.0)

    copies = [obj for obj in doc.objects if obj.uid != uid]
    assert len(copies) == 3
    assert all(copy.mesh is source for copy in copies)


def test_array_linear_accepts_a_negative_step_and_runs_backwards() -> None:
    doc, uid = _doc()
    doc.select([uid])
    clay_ops.run(_Ctx(), doc, clay_ops.get("array-linear"), count=3, x=-2.0)
    xs = sorted(float(obj.translation[0]) for obj in doc.objects)
    assert xs == pytest.approx([-4.0, -2.0, 0.0])


def test_array_linear_leaves_the_whole_group_selected() -> None:
    """So arraying an array compounds, rather than losing the originals the
    next press would have wanted too."""
    doc, uid = _doc()
    doc.select([uid])
    clay_ops.run(_Ctx(), doc, clay_ops.get("array-linear"), count=3, x=1.0)
    assert doc.selection == {obj.uid for obj in doc.objects}


def test_array_linear_count_is_clamped_to_its_documented_ceiling() -> None:
    """``count`` needs a ceiling because every copy is one more outliner row
    and one more ``Obj`` recorded in the document, not because of mesh
    memory -- see ``clay_ops.MAX_ARRAY_COUNT``'s own comment for why 200."""
    doc, uid = _doc()
    doc.select([uid])
    clay_ops.run(_Ctx(), doc, clay_ops.get("array-linear"), count=1e9, x=0.001)
    assert len(doc.objects) == clay_ops.MAX_ARRAY_COUNT


def test_array_radial_of_four_over_360_does_not_repeat_a_position() -> None:
    """The off-by-one this op exists to avoid on a closed ring: dividing by
    ``count - 1`` instead of ``count`` would land the last spoke back on the
    first."""
    doc, uid = _doc()
    doc.select([uid])
    doc.set_transform(uid, translation=[2.0, 0.0, 0.0])

    clay_ops.run(_Ctx(), doc, clay_ops.get("array-radial"), count=4, angle=360.0, axis=1)

    positions = {tuple(np.round(obj.translation, 6)) for obj in doc.objects}
    assert len(positions) == 4


def test_array_radial_of_two_over_360_does_not_put_both_copies_in_the_same_place() -> None:
    """The sharpest case of the closed-ring rule: at ``count == 2`` the wrong
    (``count - 1``) divisor is 1, so the one copy would land exactly ``360``
    degrees from the original -- the same position -- which is invisible at
    any larger count where the copies are merely *evenly spaced* rather than
    provably distinct."""
    doc, uid = _doc()
    doc.select([uid])
    doc.set_transform(uid, translation=[2.0, 0.0, 0.0])

    clay_ops.run(_Ctx(), doc, clay_ops.get("array-radial"), count=2, angle=360.0, axis=1)

    positions = {tuple(np.round(obj.translation, 6)) for obj in doc.objects}
    assert len(positions) == 2


def test_array_radial_of_four_over_90_degrees_reaches_90_degrees() -> None:
    """The open-arc rule: a user who asks for 90 degrees means the copies
    *reach* 90, not fall short of it the way dividing by ``count`` (the
    closed-ring rule) would -- four copies over 90 would land at 22.5, 45 and
    67.5, nothing at 90."""
    from warlock.studio.clay import ops as clay_ops_geom

    doc, uid = _doc()
    doc.select([uid])
    doc.set_transform(uid, translation=[2.0, 0.0, 0.0])
    original = doc.by_uid(uid)
    expected_last = clay_ops_geom.rotated_about_origin(original, 1, 90.0).translation

    clay_ops.run(_Ctx(), doc, clay_ops.get("array-radial"), count=4, angle=90.0, axis=1)

    copies = [obj for obj in doc.objects if obj.uid != uid]
    assert len(copies) == 3
    assert any(np.allclose(obj.translation, expected_last, atol=1e-6) for obj in copies)


def test_array_radial_of_two_over_a_partial_sweep_lands_the_copy_at_the_full_angle() -> None:
    """``count == 2`` on an open arc divides by ``count - 1 == 1``, so the one
    copy lands at ``angle`` exactly -- the degenerate case of the open-arc
    rule, and the one most likely to silently regress to the closed-ring
    divisor since ``1`` and ``2`` differ by so little."""
    from warlock.studio.clay import ops as clay_ops_geom

    doc, uid = _doc()
    doc.select([uid])
    doc.set_transform(uid, translation=[2.0, 0.0, 0.0])
    original = doc.by_uid(uid)
    expected = clay_ops_geom.rotated_about_origin(original, 1, 40.0).translation

    clay_ops.run(_Ctx(), doc, clay_ops.get("array-radial"), count=2, angle=40.0, axis=1)

    copy = next(obj for obj in doc.objects if obj.uid != uid)
    assert np.allclose(copy.translation, expected, atol=1e-6)


def test_array_radial_leaves_its_copies_parametric_where_mirror_copy_freezes() -> None:
    """A radial array only ever changes a copy's transform, so it is still
    exactly the primitive its generator describes; a mirror-copy's mesh is
    reflected and baked, so it cannot be."""
    doc, uid = _doc()
    doc.select([uid])
    doc.set_transform(uid, translation=[2.0, 0.0, 0.0])
    clay_ops.run(_Ctx(), doc, clay_ops.get("array-radial"), count=2, angle=90.0, axis=1)
    radial_copy = next(obj for obj in doc.objects if obj.uid != uid)
    assert radial_copy.generator == "box"
    assert radial_copy.params == {"size": (1.0, 1.0, 1.0)}

    doc2, uid2 = _doc()
    doc2.select([uid2])
    clay_ops.run(_Ctx(), doc2, clay_ops.get("mirror-copy"), axis=0, offset=0.0)
    mirrored_copy = next(obj for obj in doc2.objects if obj.uid != uid2)
    assert mirrored_copy.generator is None
    assert mirrored_copy.params == {}


def test_array_radial_hint_names_the_world_origin_as_the_centre() -> None:
    """A separate centre would need a three-number parameter ``Param`` cannot
    express, so the world origin is a decision to state plainly rather than
    a limitation to discover by surprise."""
    hint = clay_ops.get("array-radial").hint.lower()
    assert "origin" in hint
    assert "own centre" in hint


def test_mirror_copy_reflects_across_the_named_world_plane_and_leaves_the_source() -> None:
    doc, uid = _doc()
    doc.select([uid])
    doc.set_transform(uid, translation=[1.0, 0.0, 0.0])

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("mirror-copy"), axis=0, offset=5.0) is True

    assert len(doc.objects) == 2
    source = doc.by_uid(uid)
    assert np.allclose(source.translation, [1.0, 0.0, 0.0]), "the source is untouched"
    copy = next(obj for obj in doc.objects if obj.uid != uid)
    assert np.allclose(copy.translation, [9.0, 0.0, 0.0])  # 2*5 - 1


def test_mirror_copy_is_one_undo_step_for_the_whole_selection() -> None:
    doc, uids = _three_boxes_selected()
    depth = len(doc.history)
    clay_ops.run(_Ctx(), doc, clay_ops.get("mirror-copy"), axis=0, offset=0.0)
    assert len(doc.objects) == 6
    assert len(doc.history) == depth + 1


def test_mirror_copy_distinguishes_itself_from_mirror_x_y_z_in_its_hint() -> None:
    """The manual draws the same distinction in the paragraph that already
    describes Mirror X/Y/Z (docs/manual/30-clay.md); the hint is the one
    place a user reaches for it mid-gesture."""
    assert "Mirror X/Y/Z" in clay_ops.get("mirror-copy").hint


# --- place-between (the third op this tranche adds) --------------------------


def _three_in_a_row() -> tuple[bd.ClayDoc, int, int, int]:
    """Two anchors added first, then the object to be placed, added last --
    document order is what ``place-between`` reads, not selection order."""
    doc, anchor_a = _doc()  # anchor_a ("Box") sits at the origin by default
    anchor_b = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box(), translation=[0.0, 10.0, 0.0])
    ).uid
    mover = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Strut", mesh=bp.box(), translation=[99.0, 99.0, 99.0])
    ).uid
    doc.select([anchor_a, anchor_b, mover])
    return doc, anchor_a, anchor_b, mover


def test_place_between_is_disabled_with_anything_but_three_selected() -> None:
    doc, anchor_a, anchor_b, mover = _three_in_a_row()
    op = clay_ops.get("place-between")
    assert op.enabled(doc)

    doc.select([anchor_a, anchor_b])
    assert not op.enabled(doc)
    assert "2 selected now" in clay_ops.reason_for(op, doc)

    fourth = doc.add_object(bd.Obj(uid=bd.new_uid(), name="D", mesh=bp.box())).uid
    doc.select([anchor_a, anchor_b, mover, fourth])
    assert not op.enabled(doc)
    assert "4 selected now" in clay_ops.reason_for(op, doc)


def test_place_between_moves_only_the_last_object_in_document_order() -> None:
    """The newcomer moves; the two anchors -- earlier in document order --
    are left exactly where they were, which is the opposite of Merge/Union's
    own "topmost (earliest) survives" rule, on purpose (see ``_place_between``'s
    docstring for why the roles differ)."""
    doc, anchor_a, anchor_b, mover = _three_in_a_row()
    a_before = np.array(doc.by_uid(anchor_a).translation)
    b_before = np.array(doc.by_uid(anchor_b).translation)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("place-between"), fit=0) is True

    assert np.allclose(doc.by_uid(anchor_a).translation, a_before)
    assert np.allclose(doc.by_uid(anchor_b).translation, b_before)
    assert np.allclose(doc.by_uid(mover).translation, [0.0, 5.0, 0.0])


def test_place_between_is_one_undo_step() -> None:
    doc, *_ = _three_in_a_row()
    depth = len(doc.history)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("place-between"), fit=0) is True
    assert len(doc.history) == depth + 1
    assert doc.undo() is True


def test_place_between_with_fit_stretches_the_mover_to_span_the_gap() -> None:
    doc, _a, _b, mover = _three_in_a_row()
    clay_ops.run(_Ctx(), doc, clay_ops.get("place-between"), fit=1)
    assert doc.by_uid(mover).scale[1] == pytest.approx(10.0)


def test_place_between_without_fit_leaves_the_movers_scale_alone() -> None:
    doc, _a, _b, mover = _three_in_a_row()
    doc.set_transform(mover, scale=[2.0, 2.0, 2.0])
    clay_ops.run(_Ctx(), doc, clay_ops.get("place-between"), fit=0)
    assert np.allclose(doc.by_uid(mover).scale, [2.0, 2.0, 2.0])


def test_place_between_appears_in_the_object_menu() -> None:
    assert "place-between" in [op.name for op in clay_ops.menu("object")]


def test_shade_smooth_in_face_mode_with_no_face_selection_is_refused_not_silent() -> None:
    """clay-06 (2026-09-08 audit): Shade Smooth/Flat are registered for both
    object and face mode, and were gated on ``has_objects`` alone -- an
    *object*-selection predicate -- even though the op body reads
    ``doc.element_sel`` in face mode.

    ``doc.selection`` can be non-empty in face mode with no face picked: the
    outliner selects an object by calling ``doc.select`` directly
    (``panes/clay_outliner.py``), which does not touch ``element_sel`` the way
    picking a face does. Before the fix that left ``has_objects`` reading True
    here, so the row drew enabled, the click ran an empty loop over
    ``doc.element_sel``, and nothing happened -- no ``set_shading`` call, no
    history step, no toast.
    """
    doc, uid = _doc()
    doc.set_element_mode("face")
    doc.select([uid])
    assert doc.selection == {uid}
    assert not doc.element_sel, "no face has been picked"

    smooth = clay_ops.get("shade-smooth")
    assert not smooth.enabled(doc), "nothing for the op body to act on"
    assert clay_ops.reason_for(smooth, doc)
    assert clay_ops.run(_Ctx(), doc, smooth) is False
    assert len(doc.history) == 1, "the add, and nothing else -- no silent no-op step"

    # Picking a face is what makes it live again.
    _faces(doc, uid, 0)
    assert smooth.enabled(doc)
    assert clay_ops.run(_Ctx(), doc, smooth) is True


# --- merge ------------------------------------------------------------------


def _two_boxes(apart: float = 5.0) -> tuple[bd.ClayDoc, int, int]:
    doc, first = _doc()
    second = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Box.001", mesh=bp.box(), translation=[apart, 0.0, 0.0])
    )
    doc.select([first, second.uid])
    return doc, first, second.uid


def test_merge_is_disabled_below_two_objects() -> None:
    """Merging one object is the identity, and an enabled button that does
    nothing is worse than a greyed one."""
    doc, uid = _doc()
    op = clay_ops.get("join")
    assert not op.enabled(doc)
    doc.select([uid])
    assert not op.enabled(doc)
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    doc.select([o.uid for o in doc.objects])
    assert op.enabled(doc)


def test_merge_keeps_the_topmost_object_and_absorbs_the_rest() -> None:
    doc, first, second = _two_boxes()
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("join"))
    assert [o.uid for o in doc.objects] == [first]
    assert doc.by_uid(first).name == "Box"
    assert doc.selection == {first}
    assert second not in {o.uid for o in doc.objects}


def test_merge_takes_the_geometry_of_everything_it_absorbed() -> None:
    doc, first, _ = _two_boxes()
    faces = bm.face_count(doc.by_uid(first).mesh)
    clay_ops.run(_Ctx(), doc, clay_ops.get("join"))
    merged = doc.by_uid(first).mesh
    bm.validate(merged)
    assert bm.face_count(merged) == faces * 2
    lo, hi = bm.bounds(merged)
    assert np.allclose(lo, [-0.5, -0.5, -0.5]) and np.allclose(hi, [5.5, 0.5, 0.5])


def test_merge_freezes_the_generator_and_undoes_in_one_press() -> None:
    doc, first, _ = _two_boxes()
    depth = len(doc.history)
    clay_ops.run(_Ctx(), doc, clay_ops.get("join"))
    assert doc.by_uid(first).generator is None
    assert len(doc.history) == depth + 1

    assert doc.undo()
    assert [o.name for o in doc.objects] == ["Box", "Box.001"]
    assert doc.by_uid(first).generator == "box"


def test_merge_welds_at_the_distance_it_is_given() -> None:
    """The parameter is what turns two shells into one surface; at zero it is
    a group, which is a legitimate thing to ask for."""
    doc, first, _ = _two_boxes(apart=0.0)
    verts = len(doc.by_uid(first).mesh.positions)
    clay_ops.run(_Ctx(), doc, clay_ops.get("join"), weld=0.0)
    assert len(doc.by_uid(first).mesh.positions) == 2 * verts

    doc, first, _ = _two_boxes(apart=0.0)
    clay_ops.run(_Ctx(), doc, clay_ops.get("join"), weld=1e-4)
    assert len(doc.by_uid(first).mesh.positions) == verts


def test_merge_refuses_a_zero_scaled_target_as_a_toast() -> None:
    doc, first, _ = _two_boxes()
    doc.set_transform(first, scale=[0.0, 1.0, 1.0])
    depth = len(doc.history)
    ctx = _Ctx()
    assert not clay_ops.run(ctx, doc, clay_ops.get("join"))
    assert ctx.toasts.errors
    assert len(doc.objects) == 2
    assert len(doc.history) == depth


# --- parameter formatting ---------------------------------------------------


def test_a_sub_millimetre_parameter_is_drawn_with_enough_decimals():
    """imgui's default "%.3f" printed both weld distances as 0.000 -- a field
    whose value cannot be read, whose step arrows appear to do nothing, and
    which hands back a different number than the one it was showing."""
    for name in ("weld", "join"):
        param = next(p for p in clay_ops.get(name).params if "distance" in p.label)
        assert param.default < 1e-3
        shown = clay_ops.format_for(param) % param.default
        assert float(shown) == pytest.approx(param.default), shown


def test_every_parameter_can_be_read_back_from_what_it_is_drawn_as():
    """The property, for the whole registry rather than for the one that was
    wrong: what the field shows must round-trip to what the op will be given."""
    for op in clay_ops.OPS:
        for param in op.params:
            if param.integer:
                continue
            shown = clay_ops.format_for(param) % param.default
            assert float(shown) == pytest.approx(param.default), f"{op.name}.{param.name}={shown}"


def test_an_ordinary_parameter_still_reads_at_three_decimals():
    """Widened only downwards: a parameter that was legible before must not
    grow a tail of zeros because of the parameter that was not."""
    assert clay_ops.format_for(clay_ops.Param("w", "width (m)", 0.05, 0.01)) == "%.3f"
    assert clay_ops.format_for(clay_ops.Param("t", "position", 0.5, 0.05)) == "%.3f"


# --- boolean and choice params (2026-09-10) ----------------------------------
#
# Four ops landed the same day carrying a number where a checkbox or a named
# choice belonged, the tell being a label doing the widget's job: "fit to gap
# (0=off, 1=on)", "axis (0=X, 1=Y, 2=Z)". A stray comment on ``place-between``
# called that a "checkbox" precedent too, which was wrong about ``axis`` --
# three values is a choice, not a boolean -- so this is two kinds, not one.


def test_a_param_cannot_be_both_boolean_and_a_choice():
    with pytest.raises(ValueError, match="both"):
        clay_ops.Param(
            "x", "x", 1.0, low=0.0, high=1.0, boolean=True, choices=("a", "b")
        )


def test_a_boolean_params_range_must_be_0_to_1():
    with pytest.raises(ValueError, match="0..1|boolean"):
        clay_ops.Param("x", "x", 1.0, low=0.0, high=2.0, boolean=True)


def test_a_choice_params_range_must_span_its_choices():
    with pytest.raises(ValueError, match="choice"):
        clay_ops.Param("x", "x", 0.0, low=0.0, high=1.0, choices=("X", "Y", "Z"))
    # The right range for three choices is fine.
    clay_ops.Param("x", "x", 0.0, low=0.0, high=2.0, choices=("X", "Y", "Z"))


def test_place_between_fit_is_a_boolean_not_a_bare_int():
    """The label carried the widget's job -- "(0=off, 1=on)" -- because the
    field itself couldn't say it. It should just say what the field is."""
    fit = next(p for p in clay_ops.get("place-between").params if p.name == "fit")
    assert fit.boolean is True
    assert fit.choices == ()
    assert fit.label == "fit to gap"
    assert "0=" not in fit.label and "1=" not in fit.label


def test_array_radial_and_mirror_copy_axis_is_a_three_way_choice_not_boolean():
    """The earlier note calling this a checkbox case was wrong: three values
    is a named choice, and conflating it with a boolean is the bug this
    change fixes, not a shape to repeat."""
    for op_name in ("array-radial", "mirror-copy"):
        axis = next(p for p in clay_ops.get(op_name).params if p.name == "axis")
        assert axis.boolean is False
        assert axis.choices == ("X", "Y", "Z")
        assert axis.label == "axis"
        assert "0=" not in axis.label


def test_place_between_fit_still_applies_as_0_or_1():
    """The whole change is a widget swap: what ``run`` hands the op, and what
    the op does with it, must be exactly what it was before ``fit`` grew a
    ``boolean`` flag."""
    doc, anchor_a = _doc()
    anchor_b = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box(), translation=[0.0, 10.0, 0.0])
    ).uid
    mover = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Strut", mesh=bp.box(), translation=[99.0, 99.0, 99.0])
    ).uid
    doc.select([anchor_a, anchor_b, mover])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("place-between"), fit=1.0) is True
    assert doc.by_uid(mover).scale[1] == pytest.approx(10.0)

    doc2, anchor_a2 = _doc()
    anchor_b2 = doc2.add_object(
        bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box(), translation=[0.0, 10.0, 0.0])
    ).uid
    mover2 = doc2.add_object(
        bd.Obj(uid=bd.new_uid(), name="Strut", mesh=bp.box(), translation=[99.0, 99.0, 99.0])
    ).uid
    doc2.select([anchor_a2, anchor_b2, mover2])
    assert clay_ops.run(_Ctx(), doc2, clay_ops.get("place-between"), fit=0.0) is True
    assert doc2.by_uid(mover2).scale[1] == pytest.approx(1.0)


def test_array_radial_axis_still_applies_as_its_index():
    """``axis`` growing ``choices`` must not change what index the op turns
    about -- 0/1/2 still mean X/Y/Z exactly as they did as a bare int."""
    doc = bd.ClayDoc()
    spoke = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), translation=[0.0, 0.0, 5.0])
    ).uid
    doc.select([spoke])
    assert (
        clay_ops.run(_Ctx(), doc, clay_ops.get("array-radial"), count=2, angle=90.0, axis=0)
        is True
    )
    made = [obj for obj in doc.objects if obj.uid != spoke]
    assert made
    # Rotated 90 degrees about X (index 0): a point at (0, 0, d) sweeps to
    # (0, +-d, 0) -- it must land on Y, not stay on Z or move onto X.
    for obj in made:
        assert obj.translation[2] == pytest.approx(0.0, abs=1e-6)
        assert abs(obj.translation[1]) == pytest.approx(5.0, abs=1e-6)


def test_the_registrys_own_boolean_and_choice_params_still_clamp_in_run():
    """``run``'s clamp -- ``min(max(value, low), high)`` -- must still reach
    a boolean/choice param even though it is no longer ``integer``: a stray
    2.0 for ``fit`` or a 9 for ``axis`` must come back inside range, exactly
    as an out-of-range int always has."""
    doc, anchor_a = _doc()
    anchor_b = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box(), translation=[0.0, 10.0, 0.0])
    ).uid
    mover = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Strut", mesh=bp.box(), translation=[99.0, 99.0, 99.0])
    ).uid
    doc.select([anchor_a, anchor_b, mover])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("place-between"), fit=7.0) is True
    assert doc.by_uid(mover).scale[1] == pytest.approx(10.0), "fit=7 should clamp to 1 (on)"


def _one_hidden() -> tuple[bd.ClayDoc, int, int]:
    """Two selected objects, the second hidden after the fact."""
    doc, first, second = _two_boxes()
    doc.set_props(second, visible=False)
    return doc, first, second


def test_merge_is_disabled_when_only_one_selected_object_is_visible():
    """Greyed rather than refused: the op would skip the hidden one, so a row
    enabled by an object the merge ignores is the enabled-button-that-does-
    nothing problem again."""
    doc, _first, _second = _one_hidden()
    assert not clay_ops.get("join").enabled(doc)


def test_merge_never_absorbs_a_hidden_object():
    """``_select_all`` no longer hands one over; this is the other half, an
    object hidden after it was selected. Forced past ``enabled`` because that is
    exactly what a stale predicate would do."""
    doc, first, second = _one_hidden()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="C", mesh=bp.box(), translation=[0.0, 5.0, 0.0]))
    doc.select([o.uid for o in doc.objects])
    faces = bm.face_count(doc.by_uid(first).mesh)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("join"))

    assert second in {o.uid for o in doc.objects}, "the hidden object survives untouched"
    assert bm.face_count(doc.by_uid(first).mesh) == faces * 2, "A and C, not A, B and C"


# --- Union's binding ----------------------------------------------------------
#
# Union Objects did the job the manual describes from the day it landed and
# almost nobody found it: it sat in the context menu with no key while the weld
# beside it held Ctrl+J, so the discoverable half of the pair was the half that
# leaves the interior walls in. The binding is the fix, and these pin it.


def test_union_is_bound_beside_merge_rather_than_buried() -> None:
    """The same key, shifted, for the same question answered the other way. A
    user who knows one is one keystroke from the other, which is the whole of
    what was missing. The pair moved from J to M when Clay's Ctrl+D/Ctrl+J
    stopped disagreeing with Inker's and Plotter's; what matters here is that
    they stayed a pair."""
    assert clay_ops.get("join").key == "Ctrl+M"
    assert clay_ops.get("union").key == "Ctrl+Shift+M"


def test_clay_agrees_with_the_other_editors_about_ctrl_d_and_ctrl_j() -> None:
    """Clay used to duplicate on Ctrl+D, which deselects in Inker and Plotter,
    and merge on Ctrl+J, which duplicates in Plotter. Two chords meaning two
    things in two workspaces of one app is a user pressing the one they learned
    and getting the other verb."""
    from warlock.studio import inker_ops

    assert clay_ops.get("duplicate").key == "Ctrl+J"
    assert not any(op.key == "Ctrl+D" for op in clay_ops.OPS)
    # Inker's, for the comparison the paragraph above rests on.
    assert inker_ops.get("deselect").key == "Ctrl+D"


def test_no_two_ops_in_one_mode_claim_the_same_key() -> None:
    """The registry's own promise, asserted rather than assumed -- adding a
    binding is exactly the change that can break it, and ``by_key`` returns the
    first match, so a collision fails silently in favour of registration order."""
    for mode in clay_ops.ALL_MODES:
        keys = [op.key for op in clay_ops.menu(mode) if op.key]
        duplicates = {key for key in keys if keys.count(key) > 1}
        assert not duplicates, f"{mode}: {sorted(duplicates)}"


def test_union_and_merge_are_enabled_together() -> None:
    """One predicate, so the pair is never half-offered: a selection that can
    be welded can be unioned, and the choice between them is the user's."""
    doc, _first, _second = _two_boxes()
    assert clay_ops.get("join").enabled(doc)
    assert clay_ops.get("union").enabled(doc)


def test_merge_points_at_union_from_inside_its_own_dialog() -> None:
    """The pair is only a choice if you know both halves exist.

    The dialog is where the pointer belongs rather than the menu: it is the one
    moment the user has committed to "make these one object" and can still pick
    which meaning of that they wanted, and it is the moment the weld's cost --
    the walls it is about to bury -- is still undone.
    """
    hint = clay_ops.get("join").hint
    assert "Union" in hint
    assert clay_ops.get("union").key in hint


def test_every_op_hint_names_a_binding_that_exists() -> None:
    """A hint citing a key is a second place the binding is written down, and
    the failure mode of those is that they go stale in silence."""
    bindings = {op.key for op in clay_ops.OPS if op.key}
    for op in clay_ops.OPS:
        for token in op.hint.replace(",", " ").replace("(", " ").replace(")", " ").split():
            if token.startswith("Ctrl+") or token.startswith("Shift+"):
                assert token in bindings, f"{op.name}: {token} is not bound to anything"
