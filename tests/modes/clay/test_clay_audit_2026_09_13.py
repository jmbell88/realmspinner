"""Regression tests for the 2026-09-13 audit's clay-02..04 and docs-05.

One module for the whole batch rather than editing existing suites, per the
fixer brief: these findings touch files owned by this fix pass and their
existing test modules are not.
"""

from __future__ import annotations

import inspect

import pytest

from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import mesh as bm
from warlock.kernels.mesh import ops
from warlock.kernels.mesh import primitives as bp
from warlock.kernels.mesh.elements import ElementSel, OpError


def _obj(name: str = "A", mesh: bm.Mesh | None = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=bp.box() if mesh is None else mesh,
        **kwargs,  # type: ignore[arg-type]
    )


# --- clay-01: the ordinary lathe still builds ---------------------------------


def test_lathe_accepts_its_own_default_profile():
    """The ceiling must not catch the ordinary case."""
    mesh = bp.lathe()
    bm.validate(mesh)


# --- clay-02: X-ray does not reach faces -------------------------------------


class _FakeCamera:
    orthographic = False
    theta = phi = distance = 0.0
    target = (0.0, 0.0, 0.0)

    def aspect(self):  # pragma: no cover - not exercised, present for shape
        return 1.0


def test_face_mode_pick_reaches_an_occluded_object_under_xray():
    """Two boxes on the same ray, near one in front of a far one. Under
    X-ray, face mode must be able to reach the far object's face -- vertex
    and edge mode already can (they drop ``surface_depth``); face mode fell
    back to ``hit.face if hit.uid == obj.uid else None``, which only ever
    named the *nearest* object's face, exactly like it would with X-ray off.
    """
    from warlock.studio.modes.clay.ui import _view_pick

    near = _obj("Near", translation=(0.0, 0.0, 1.0))
    far = _obj("Far", translation=(0.0, 0.0, -1.0))

    class FakeDoc:
        objects = [near, far]
        element_mode = "face"

    calls = []

    def fake_pick_face_on(self, doc, obj, origin, direction):
        calls.append(obj.name)
        # "Near" hits at t=0 (closer), "Far" hits at t=5 (farther), each on
        # its own object -- exactly what a real per-object raycast returns.
        t = 0.0 if obj.name == "Near" else 5.0
        return _view_pick.Hit(uid=obj.uid, t=t, face=0)

    class Picker(_view_pick.PickOps):
        xray = True

        def pick_face(self, doc, local):
            # The frontmost-only hit a real ``pick_face`` would return.
            return _view_pick.Hit(uid=near.uid, t=0.0, face=0)

        def _ray(self, local):
            return (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)

        def screen_of(self, doc, uid):
            return None  # face mode ranks by ray distance, not screen depth

    Picker._pick_face_on = fake_pick_face_on  # type: ignore[assignment]

    picker = Picker()
    result = picker.pick_element(FakeDoc(), (0.0, 0.0))
    assert result is not None
    uid, index = result
    # Both objects must have been probed independently -- the whole point of
    # the fix -- and the nearer one (by the per-object raycast's own t) wins.
    assert set(calls) == {"Near", "Far"}
    assert uid == near.uid


# --- clay-03: placing a primitive in element mode steals the selection ------


def test_adding_a_primitive_in_an_element_mode_does_not_steal_the_properties_panels_selection():
    """``ClayDoc.add_objects`` used to call ``select`` unconditionally, which
    in vertex/edge/face mode overwrote ``doc.selection`` with the new
    object's uid while ``doc.element_sel`` still named the object being
    edited -- the Properties panel would then describe one object while its
    rows edited another."""
    doc = bd.ClayDoc()
    editing = doc.add_object(_obj("Editing"))
    doc.set_element_mode("vertex")
    doc.element_sel = {editing.uid: ElementSel(verts=[0])}
    doc.selection = {editing.uid}

    doc.add_objects([_obj("New")])

    assert doc.selection == {editing.uid}


def test_adding_a_primitive_in_object_mode_still_selects_it():
    """The fix must not regress the ordinary object-mode case."""
    doc = bd.ClayDoc()
    added = doc.add_objects([_obj("New")])
    assert doc.selection == {added[0].uid}


# --- clay-04: Merge Objects has no ceiling -----------------------------------


def test_join_refuses_before_concatenating_more_corners_than_clay_can_weld_on_the_frame_thread(
    monkeypatch,
):
    monkeypatch.setattr(ops, "MAX_JOINED_CORNERS", 20)
    a, b = _obj("A"), _obj("B", translation=(5.0, 0.0, 0.0))
    with pytest.raises(OpError, match="merge"):
        ops.join([a, b])


def test_join_still_works_under_the_real_ceiling_for_two_boxes():
    a, b = _obj("A"), _obj("B", translation=(5.0, 0.0, 0.0))
    merged = ops.join([a, b], eps=0.0)
    bm.validate(merged)


# --- docs-05: agent_clay outputSchema comment --------------------------------


def test_output_schema_comment_lists_every_tool_that_declares_one():
    """Every tool that declares an ``outputSchema`` -- ``clay_scene``,
    ``clay_add_primitive``, ``clay_add_mesh``, ``clay_diagnose`` and
    ``clay_analyze`` -- is named in the module comment, which said three
    long after ``clay_add_mesh`` picked up its own schema. This reads the
    source rather than importing the schema builders, so it fails against the
    unfixed comment text directly.

    The "# --- output schemas" banner and its comment block moved to
    ``studio/modes/clay/agent/schema.py`` in the P4 restructure (``dev/RESTRUCTURE.md``);
    this test moved with it rather than reading ``agent_clay`` itself, which
    no longer carries that banner at all.
    """
    from warlock.studio.modes.clay.agent import schema as agent_clay_schema

    source = inspect.getsource(agent_clay_schema)
    # The dashes are part of the needle on purpose: the module docstring
    # itself quotes the bare banner name in prose (explaining where the
    # section moved from), and a search for that alone would match the
    # docstring instead of the real banner below it.
    start = source.index("# --- output schemas -----")
    # The comment block itself, up to the next top-level statement.
    block = source[start : start + 1200]
    assert "Five tools below declare an ``outputSchema``" in block
    assert "clay_scene" in block and "clay_add_primitive" in block
    assert "clay_add_mesh" in block and "clay_diagnose" in block
    assert "clay_analyze" in block
    assert "Only three tools" not in block
