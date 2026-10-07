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

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops as clay_ops_geom
from realmspinner.kernels.mesh import ops_topo
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops


class _Toasts:
    """Only what ``Ctx`` really offers. The double used to carry an ``error``
    method under a ``ctx.toasts`` attribute the app has never had, which is
    exactly how every refusal passed here and vanished in the real app."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.info: list[str] = []


class _Ctx:
    def __init__(self) -> None:
        self.toasts = _Toasts()

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.toasts.errors.append(message)
        else:
            self.toasts.info.append(message)


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
    assert "triangulate" in face
    assert face == [op.name for op in clay_ops.OPS if "face" in op.modes]

    edges = [op.name for op in clay_ops.menu("edge")]
    # Extrude is one row across all three modes, dispatched on the mode: it
    # means the same thing everywhere and only the implementation differs, so
    # three rows would be exposing that.
    assert "extrude" in edges
    assert "triangulate" not in edges, "triangulate is a face op"
    assert "inset" not in edges, "inset is a face op"


def test_a_key_resolves_to_the_op_the_menu_shows_for_it() -> None:
    op = clay_ops.by_key("face", "E")
    assert op is not None and op.name == "extrude"
    # The same op, and therefore the same key, in every element mode.
    assert clay_ops.by_key("edge", "E") is op
    assert clay_ops.by_key("vertex", "E") is op
    assert clay_ops.by_key("face", "nope") is None


def test_select_none_carries_no_key_that_by_key_can_never_resolve() -> None:
    """The 2026-09-20 audit's clay-22: ``select-none`` used to register
    ``key="Esc"``, but ``by_key`` is only ever called with
    ``pygame.key.name(...).upper()`` -- which spells the escape key
    ``"ESCAPE"``, never ``"Esc"`` -- so that binding could never actually
    fire and the context menu's "Select None  Esc" claimed a shortcut this
    op never had. Escape is genuinely handled by ``mode._escape`` outside
    this registry, so the fix is dropping the dead metadata rather than
    aliasing it: this asserts neither spelling resolves to any op, and that
    ``select-none`` itself carries no key at all.
    """
    for mode in clay_ops.ELEMENT_MODES:
        assert clay_ops.by_key(mode, "Esc") is None
        assert clay_ops.by_key(mode, "ESCAPE") is None
    assert clay_ops.get("select-none").key == ""


def test_the_registry_imports_no_gui() -> None:
    """``clay_ops`` is testable precisely because it draws nothing.

    The layering rule for the whole package: nothing under ``clay/`` and nothing
    in the registry knows imgui exists; only panes and ``main`` draw.
    """
    import ast
    import importlib
    from pathlib import Path

    source = importlib.import_module("realmspinner.studio.modes.clay.ops").__file__
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
    per op. The three ops with no gate at all (``select-all``,
    ``select-invert``, ``frame``) are the only ones this document does not
    refuse, and are checked for that instead.
    """
    doc = bd.ClayDoc()
    always_enabled = {"select-all", "select-invert", "frame"}
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
    inset = clay_ops.get("inset")
    assert "Switch to face mode" in clay_ops.reason_for(inset, doc)
    doc.set_element_mode("face")
    assert clay_ops.reason_for(inset, doc) == "Select something first."
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    assert clay_ops.reason_for(inset, doc) == ""


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


@pytest.mark.parametrize("key", ["mirror-x", "mirror-y", "mirror-z"])
def test_every_op_that_changes_geometry_freezes_it(key: str) -> None:
    """The regression: the freeze lived in ``run_mesh_op`` and Smooth, so
    Mirror went straight to ``set_mesh`` and left the object
    still claiming to be "box, size 1". The properties panel keeps offering
    that size field, and touching it rebuilds a pristine box -- so the mirror
    vanished with no warning.
    """
    doc, uid = _doc()
    doc.select([uid])
    doc.set_transform(uid, translation=[1.0, 2.0, 3.0])

    clay_ops.run(_Ctx(), doc, clay_ops.get(key))

    assert doc.by_uid(uid).generator is None
    assert doc.by_uid(uid).params == {}


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
    _faces(doc, uid, 0, 1)  # opposite faces of a box touch nowhere
    ctx = _Ctx()
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("merge_faces")) is False
    assert ctx.toasts.errors
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
    assert clay_ops.defaults_for(inset) == {
        "thickness": 0.1,
        "depth": 0.0,
        "region": 0.0,
    }
    assert clay_ops.run(_Ctx(), doc, inset) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 10


def test_inset_faces_region_mode_is_reachable_through_the_op_registry() -> None:
    """clay-08 (2026-09-11 audit): ``ops_topo.inset_faces``'s ``region=True`` --
    a documented, tested second mode that insets the outline of the whole
    selected block rather than each face on its own -- was tested only by
    calling ``inset_faces`` directly (``tests/modes/clay/test_ops_topo.py``); the
    registered "inset" ``Op`` declared just ``thickness``/``depth``, so the
    mode was unreachable from the menu, the tools pane, the keyboard, and
    (since ``studio/modes/clay/agent/dispatch.py`` derives its tool schema from the same
    ``Op.params``) the agent surface too.
    """
    inset = clay_ops.get("inset")
    region = next((p for p in inset.params if p.name == "region"), None)
    assert region is not None, "'inset' declares no 'region' param"
    assert region.boolean is True, "region is a toggle, not a bare number"
    assert region.default == 0.0, "default stays per-face -- the old behaviour"

    # Per-face (the declared default): each selected face gets its own rim,
    # so two adjacent faces double the edge between them.
    doc, uid = _doc()
    _faces(doc, uid, 0, 2)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("inset")) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 14

    # region=1.0 -- the checkbox's "on", exactly what the pane's own
    # checkbox writes -- reaches ``inset_faces(..., region=True)`` through
    # the same ``run`` call every other op's parameter takes.
    doc, uid = _doc()
    _faces(doc, uid, 0, 2)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("inset"), region=1.0) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 12


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


def test_merge_faces_joins_two_adjacent_faces_into_one() -> None:
    """Two adjacent faces of a box become one, leaving five."""
    doc, uid = _doc()
    _faces(doc, uid, 0, 2)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("merge_faces")) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 5


def test_merge_faces_offers_itself_only_in_face_mode() -> None:
    """Merge Faces is a face-selection op; a row over a vertex or edge
    selection would have nothing to merge."""
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


def test_merge_faces_refuses_a_selection_it_cannot_represent() -> None:
    """Opposite faces of a box touch nowhere, so there is no single n-gon to
    make. The refusal is a toast and the mesh is left alone."""
    doc, uid = _doc()
    before = bm.face_count(doc.by_uid(uid).mesh)
    _faces(doc, uid, 0, 1)
    ctx = _Ctx()
    clay_ops.run(ctx, doc, clay_ops.get("merge_faces"))
    assert bm.face_count(doc.by_uid(uid).mesh) == before


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


# --- mirror-copy (repetition) ------------------------------------------------


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


def test_mirror_copy_selects_its_copies_so_mirroring_twice_makes_four() -> None:
    """One leg, mirrored across X and then across Z, is four table legs --
    but only if the second press sees the pair. Leaving the copies
    unselected (what mirror-copy used to do, while both arrays selected
    theirs) meant the second press re-mirrored the original alone and the
    fourth corner silently never appeared."""
    doc, uid = _doc()
    doc.select([uid])
    doc.set_transform(uid, translation=[1.0, 0.0, 1.0])

    clay_ops.run(_Ctx(), doc, clay_ops.get("mirror-copy"), axis=0, offset=0.0)
    assert doc.selection == {obj.uid for obj in doc.objects}, "originals and copies both"

    clay_ops.run(_Ctx(), doc, clay_ops.get("mirror-copy"), axis=2, offset=0.0)
    assert len(doc.objects) == 4
    corners = sorted(
        (round(float(obj.translation[0]), 6), round(float(obj.translation[2]), 6))
        for obj in doc.objects
    )
    assert corners == [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]


def test_mirror_copy_distinguishes_itself_from_mirror_x_y_z_in_its_hint() -> None:
    """The manual draws the same distinction in the paragraph that already
    describes Mirror X/Y/Z (docs/manual/30-clay.md); the hint is the one
    place a user reaches for it mid-gesture."""
    assert "Mirror X/Y/Z" in clay_ops.get("mirror-copy").hint


def test_triangulate_faces_no_selection_fallback_is_unreachable_through_the_registered_op() -> None:
    """clay-36 (2026-09-19 audit): the hint used to promise "(or every face,
    with none selected)", but ``triangulate``'s own ``enabled=in_mode("face")``
    requires a *non-empty* element selection (``in_mode``'s own ``bool(doc.
    element_sel)`` check), so the row is greyed out exactly when the kernel's
    no-selection branch (``ops_topo.triangulate_faces``) would fire. That
    kernel branch is another fixer's file and stays untouched; the fix here is
    the sentence, which must stop promising a path this op can never reach."""
    doc, uid = _doc()
    doc.set_element_mode("face")
    op = clay_ops.get("triangulate")

    assert not op.enabled(doc), "an empty face selection must leave the row disabled"
    assert "none selected" not in op.hint
    assert "every face" not in op.hint


# --- shading ------------------------------------------------------------------


def test_shade_smooth_in_face_mode_with_no_face_selection_is_refused_not_silent() -> None:
    """clay-06 (2026-09-08 audit): Shade Smooth/Flat are registered for both
    object and face mode, and were gated on ``has_objects`` alone -- an
    *object*-selection predicate -- even though the op body reads
    ``doc.element_sel`` in face mode.

    ``doc.selection`` can be non-empty in face mode with no face picked: the
    outliner selects an object by calling ``doc.select`` directly
    (``studio/modes/clay/ui/outliner.py``), which does not touch ``element_sel`` the way
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
# Ops landed carrying a number where a checkbox or a named choice belonged, the
# tell being a label doing the widget's job: "axis (0=X, 1=Y, 2=Z)". Three
# values is a choice, not a boolean -- so this is two kinds, not one.


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


def test_mirror_copy_axis_is_a_three_way_choice_not_boolean():
    """The earlier note calling this a checkbox case was wrong: three values
    is a named choice, and conflating it with a boolean is the bug this
    change fixes, not a shape to repeat."""
    axis = next(p for p in clay_ops.get("mirror-copy").params if p.name == "axis")
    assert axis.boolean is False
    assert axis.choices == ("X", "Y", "Z")
    assert axis.label == "axis"
    assert "0=" not in axis.label


def test_the_registrys_own_boolean_and_choice_params_still_clamp_in_run():
    """``run``'s clamp -- ``min(max(value, low), high)`` -- must still reach
    a boolean/choice param even though it is no longer ``integer``: a stray
    7.0 for ``region`` must come back inside range, exactly as an
    out-of-range int always has."""
    doc, uid = _doc()
    _faces(doc, uid, 0, 2)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("inset"), region=7.0) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 12, "region=7 should clamp to 1 (on)"


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


# --- Merge's binding ----------------------------------------------------------


def test_merge_is_bound_to_ctrl_m() -> None:
    """Merge moved from J to M when Clay's Ctrl+D/Ctrl+J stopped disagreeing
    with Inker's and Plotter's."""
    assert clay_ops.get("join").key == "Ctrl+M"


def test_clay_agrees_with_the_other_editors_about_ctrl_d_and_ctrl_j() -> None:
    """Clay used to duplicate on Ctrl+D, which deselects in Inker and Plotter,
    and merge on Ctrl+J, which duplicates in Plotter. Two chords meaning two
    things in two workspaces of one app is a user pressing the one they learned
    and getting the other verb."""
    from realmspinner.studio.modes.inker import ops as inker_ops

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


def test_every_op_hint_names_a_binding_that_exists() -> None:
    """A hint citing a key is a second place the binding is written down, and
    the failure mode of those is that they go stale in silence."""
    bindings = {op.key for op in clay_ops.OPS if op.key}
    for op in clay_ops.OPS:
        for token in op.hint.replace(",", " ").replace("(", " ").replace(")", " ").split():
            if token.startswith("Ctrl+") or token.startswith("Shift+"):
                assert token in bindings, f"{op.name}: {token} is not bound to anything"


def test_every_clay_op_label_ending_in_ellipsis_actually_has_params() -> None:
    """The reverse of ``tests/test_label_conventions.py``'s own
    ``test_a_clay_op_that_opens_a_dialog_says_so``, which only ever checked
    "has params => label ends in an ellipsis". The 2026-09-19 audit
    (clay-30) found the other direction broken: ``union``, ``difference``
    and ``intersection`` ended in "..." -- the registry's own convention for
    "this opens a dialog" -- while declaring no ``params``, so each fired
    immediately with no dialog at all. Kept beside the registry itself
    (``test_clay_ops.py``) rather than in the shared, cross-mode
    ``tests/test_label_conventions.py``, which this fixer does not own."""
    for op in clay_ops.OPS:
        if op.label.endswith("..."):
            assert op.params, f"{op.name}: label promises a dialog but declares no params"


# --- drop-to-ground / snap-to-grid --------------------------------------------
#
# The box arithmetic itself (``mason.ops.drop_to_ground``) is Mason's own and
# tested there; what belongs here is the Clay-side wiring -- world boxes
# gathered from the *selection* and one undo step per gesture.


def test_drop_to_ground_rests_a_rotated_scaled_objects_world_box_on_y_zero() -> None:
    from realmspinner.kernels.geom3d import math3d as m3

    doc = bd.ClayDoc()
    rotation = m3.quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.radians(45.0))
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Tilted",
            mesh=bp.box(size=(1, 1, 1)),
            translation=[0, 5, 0],
            rotation=rotation,
            scale=[2, 2, 2],
        )
    )
    doc.select([obj.uid])

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("drop-to-ground")) is True

    lo, _hi = clay_ops_geom.world_box(doc.by_uid(obj.uid))
    assert float(lo[1]) == pytest.approx(0.0, abs=1e-6)


def test_drop_to_ground_is_one_undo_step_for_several_objects() -> None:
    doc, uids = _three_boxes_selected()  # default box, half-height 0.5
    for uid, y in zip(uids, (2.0, 5.0, -1.0), strict=True):
        doc.set_transform(uid, translation=[0, y, 0])
    depth = len(doc.history)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("drop-to-ground")) is True
    assert len(doc.history) == depth + 1
    for uid in uids:
        assert doc.by_uid(uid).translation[1] == pytest.approx(0.5)
    assert doc.undo() is True


def test_drop_to_ground_appears_in_the_object_menu() -> None:
    assert "drop-to-ground" in [op.name for op in clay_ops.menu("object")]


def test_snap_to_grid_rounds_a_translation_onto_the_step() -> None:
    doc, uid = _doc()
    doc.set_transform(uid, translation=[1.24, -0.63, 2.51])
    doc.select([uid])

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("snap-to-grid"), step=0.5) is True
    assert np.allclose(doc.by_uid(uid).translation, [1.0, -0.5, 2.5])


def test_snap_to_grid_is_one_undo_step_for_several_objects() -> None:
    doc, uids = _three_boxes_selected()
    for uid, x in zip(uids, (1.24, 2.6, 9.9), strict=True):
        doc.set_transform(uid, translation=[x, 0, 0])
    depth = len(doc.history)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("snap-to-grid"), step=1.0) is True
    assert len(doc.history) == depth + 1
    assert doc.undo() is True


def test_snap_to_grid_appears_in_the_object_menu() -> None:
    assert "snap-to-grid" in [op.name for op in clay_ops.menu("object")]
