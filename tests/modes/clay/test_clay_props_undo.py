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

import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay.ui.panes import props as clay_props


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
    # Without the leading quote: tranche 3 (scene structure) made each label an
    # f-string with a "local " prefix for a parented object
    # (``f"{prefix}position##bt"``), so the *literal* ``"position##bt"``
    # (quote included) no longer appears verbatim -- the id suffix still does.
    for field in ('position##bt"', 'scale##bs"', 'rotation##br"'):
        _fold_precedes(transform_src, field, "doc.set_transform(")

    generator_src = inspect.getsource(clay_props._generator)
    _fold_precedes(generator_src, "_widget(key,", "doc.set_generator_params(")


def test_a_locked_objects_transform_fields_are_drawn_disabled_not_live_and_erroring() -> None:
    """The 2026-09-26 audit's clay-panes-07: a locked object's Position/
    Scale/Rotation fields stayed live and editable, so each keystroke reached
    ``set_transform``'s own refusal (``OpError``) and popped a fresh toast --
    typing "1.25" produced four toasts, one per character. The fields are now
    wrapped in ``imgui.begin_disabled(obj.locked)``, the same chrome
    ``clay_props.draw``'s own body already uses for "a save is in flight",
    so a locked object's fields cannot be edited at all rather than being
    edited and then refused."""
    source = inspect.getsource(clay_props._transform)
    disable_at = source.index("imgui.begin_disabled(obj.locked)")
    enable_at = source.index("imgui.end_disabled()")
    position_at = source.index('"position##bt"')
    dimensions_at = source.index("_dimensions(doc, obj)")
    assert disable_at < position_at < enable_at < dimensions_at, (
        "the position field must be drawn between begin_disabled(obj.locked) and end_disabled()"
    )


def test_a_locked_objects_generator_fields_are_drawn_disabled_not_live_and_erroring() -> None:
    """``_generator``'s own copy of clay-panes-07's gap, the same shape
    ``_transform``'s own test above checks."""
    source = inspect.getsource(clay_props._generator)
    disable_at = source.index("imgui.begin_disabled(obj.locked)")
    enable_at = source.index("imgui.end_disabled()")
    widget_at = source.index("_widget(key,")
    assert disable_at < widget_at < enable_at, (
        "the per-param widget loop must be drawn between begin_disabled(obj.locked) and "
        "end_disabled()"
    )


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


def test_add_material_and_assign_closes_its_gesture_even_when_set_props_fails() -> None:
    """The 2026-09-26 audit's clay-document-06: ``set_props`` ran outside a
    ``try`` here, so a bad uid (an object deleted out from under a stale
    panel reference) raised past ``collapse_since`` and left the gesture
    open forever -- ``UndoStack._open_gestures`` never dropped back to zero,
    so eviction stayed deferred for the rest of the session (the same shape
    ``test_clay_tranche1_decimate.py``'s own ``_open_gestures`` regression
    checks)."""
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))

    with pytest.raises(KeyError):
        doc.add_material_and_assign(999999)

    assert doc.history._open_gestures == 0, "the gesture must close even when set_props fails"


def test_remove_material_and_reassign_closes_its_gesture_even_when_set_props_fails() -> None:
    """``remove_material_and_reassign``'s own copy of clay-document-06's gap:
    ``remove_material`` succeeds (it takes no uid), but the ``set_props``
    call right after it can still fail on a bad uid, and the gesture must
    close either way."""
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    extra = doc.add_material()

    with pytest.raises(KeyError):
        doc.remove_material_and_reassign(999999, extra)

    assert doc.history._open_gestures == 0, "the gesture must close even when set_props fails"


def test_palette_row_material_users_is_memoised_on_doc_rev(monkeypatch) -> None:
    """The 2026-09-26 audit's clay-panes-06: ``_palette_row`` called
    ``doc.material_users`` fresh every single frame the properties panel was
    open with an object selected, and that method sums a face count over
    every object *and* every object an undo step is still holding out of the
    document (its own docstring) -- expensive, and unmemoised, for a number
    that does not change between frames unless the document does.
    ``clay_props._material_users`` now memoises it on ``doc.rev``."""
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    calls = []
    original = bd.ClayDoc.material_users

    def counting(self, index):
        calls.append(1)
        return original(self, index)

    monkeypatch.setattr(bd.ClayDoc, "material_users", counting)

    clay_props._material_users(doc, obj.material)
    clay_props._material_users(doc, obj.material)
    clay_props._material_users(doc, obj.material)
    assert len(calls) == 1, "an unchanged document must be counted once, not every call"

    extra = doc.add_material()
    doc.set_props(obj.uid, material=extra)
    clay_props._material_users(doc, extra)
    assert len(calls) == 2, "a real edit (it bumps rev) must still be counted"
