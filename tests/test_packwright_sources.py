"""Packwright's sources pane: the rename field and the tile-set import popup.

No such module existed before the 2026-09-07 audit. Both rows below are
source-inspection checks in ``test_undo_gesture_doors.py``'s fifth-section
style, because the field they cover draws only inside a selected row's own
sub-tree (the rename field) or a popup (the tile-set import), neither of which
a headless frame opens on its own.
"""

from __future__ import annotations

import inspect

from warlock.studio.panes import packwright_sources


def test_renaming_a_packwright_source_is_one_undo_step_not_one_per_keystroke():
    """packwright-03: ``_row``'s rename field called ``widgets.input_text``
    with no ``commit=True``, while ``clay_outliner.py``'s identical widget
    passes it -- so ``rename_source`` (an unconditional ``history.push``)
    fired once per keystroke. Typing "lead" pushed 4 steps, and one Ctrl+Z
    left "lea" rather than undoing the whole rename."""
    source = inspect.getsource(packwright_sources._row)
    after_field = source.split('"##rename"', 1)[1]
    call_end = after_field.index(")")
    assert "commit=True" in after_field[:call_end]
