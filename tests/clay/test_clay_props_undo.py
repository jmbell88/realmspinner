"""Clay's Properties panel: one undo step per gesture, not per keystroke.

The 2026-09-06 audit folded ``_material``'s three sliders (this module,
``_material``) but missed the transform and generator-parameter fields in the
same file. Both call ``set_transform``/``set_generator_params`` -- each an
unconditional ``history.push`` -- under a bare ``if changed:`` with no
``controls.fold_undo`` in between, and both fields are ``input_float3``/
``input_float4``, which fire on every keystroke exactly like the sliders do on
every frame of a drag.

Driven by source inspection rather than a live imgui frame, for the reason
``test_undo_gesture_doors.py``'s own packwright test states it for this exact
module: ``_transform`` and ``_generator`` each draw several fields in one
``draw`` call, and a headless frame has one active item at a time, which
cannot tell one field's activation from another's the way the real imgui item
stack does. The positional contract (draw, fold, act) is what a source check
proves instead.
"""

from __future__ import annotations

import inspect

from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import primitives as bp
from warlock.studio.panes import clay_props


def _fold_precedes(source: str, field_marker: str, write_marker: str) -> None:
    after = source.split(field_marker, 1)[1]
    assert "controls.fold_undo(" in after, (
        f"no controls.fold_undo(...) call after {field_marker!r}"
    )
    fold = after.index("controls.fold_undo(")
    write = after.index(write_marker)
    assert fold < write, f"{write_marker} runs before the fold"


def test_editing_a_transform_or_generator_field_by_keystroke_is_one_undo_step() -> None:
    """The 2026-09-07 audit's clay-01: typing a multi-digit number into
    Position, Scale, Rotation or any generator parameter pushed one undo step
    per digit, and a lone Ctrl+Z took back one typed character rather than the
    whole edit -- unlike ``_material``'s base colour/metallic/roughness
    fields in this same file, which already fold three times over."""
    transform_src = inspect.getsource(clay_props._transform)
    for field in ('"position##bt"', '"scale##bs"', '"rotation##br"'):
        _fold_precedes(transform_src, field, "doc.set_transform(")

    generator_src = inspect.getsource(clay_props._generator)
    _fold_precedes(generator_src, "_widget(key,", "doc.set_generator_params(")


def test_adding_or_removing_a_material_slot_is_one_undo_step() -> None:
    """The 2026-09-08 audit's clay-02: clicking Add pushed
    ``doc.add_material()`` and ``doc.set_props(...)`` as two separate undo
    steps for one click, and Remove pushed ``doc.remove_material(...)`` and
    ``doc.set_props(...)`` as two separate steps too -- neither pair folded
    by a ``history.mark()``/``collapse_since``, unlike every other
    document-changing gesture in Clay. One Ctrl+Z after clicking Add left a
    stray, unreferenced palette entry behind instead of restoring the
    object's original material; one Ctrl+Z after Remove left the object
    pointed at the reassigned slot instead of the one it started on.

    ``_palette_row`` now routes both buttons through a ``ClayDoc`` helper
    (``add_material_and_assign`` / ``remove_material_and_reassign``) that
    folds the pair into one step, the same shape ``join_objects`` already
    folds a mesh replacement and a set of removals into.
    """
    source = inspect.getsource(clay_props._palette_row)
    assert "doc.add_material_and_assign(" in source
    assert "doc.remove_material_and_reassign(" in source

    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))

    before_materials = list(doc.materials)
    before_history = len(doc.history)
    new_index = doc.add_material_and_assign(obj.uid)
    assert len(doc.materials) == len(before_materials) + 1
    assert obj.material == new_index
    assert len(doc.history) == before_history + 1, "one click, one undo step"
    assert doc.undo()
    assert list(doc.materials) == before_materials
    assert obj.material != new_index, "one Ctrl+Z restores the pre-click material too"

    extra = doc.add_material()
    doc.set_props(obj.uid, material=extra)
    before_materials = list(doc.materials)
    before_history = len(doc.history)
    assert doc.remove_material_and_reassign(obj.uid, extra) is True
    assert obj.material != extra
    assert len(doc.history) == before_history + 1, "one click, one undo step"
    assert doc.undo()
    # The removed entry comes back, in one press rather than the two the
    # unfolded pair needed. What ``obj.material`` itself lands on afterwards
    # is a separate, pre-existing question about ``_shift_materials``'s own
    # in-place renumbering (``edits.py``, not owned by this fix) racing an
    # ``ObjectPropsEdit`` undo on the same field -- present whether the pair
    # is folded or not, and not what this finding is about.
    assert list(doc.materials) == before_materials, "one Ctrl+Z restores the removed slot too"
