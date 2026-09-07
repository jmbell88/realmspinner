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
