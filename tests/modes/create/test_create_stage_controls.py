"""Create's mesh-side controls: what a greyed control says, and to whom.

Pure source inspection, for ``dev/INVARIANTS.md``'s reason for every pane in
this file's neighbourhood: these are imgui panes and cannot be driven
headlessly, so the *decision* -- what argument a call is made with, what an
``if`` is conditioned on -- is what gets tested, the way
``tests/test_field_error_wiring.py`` already does for the field-error rings.
"""

from __future__ import annotations

import inspect
import re

from realmspinner.studio.modes.create.ui import brief as create_brief


def test_make_3d_button_states_a_reason_when_disabled_and_shows_ctrl_enter_only_when_live():
    """The 2026-09-05 audit, finding create-08, carried to where Make 3D lives now.

    Make 3D is the command bar's Generate on the Mesh stage: the same
    ``create_brief._generate`` passes the top refusal into ``primary_button`` as
    ``reason=`` -- shown only while the button is disabled -- and reserves the
    "Ctrl+Enter" tooltip for while it is live (``tooltip=``, which the shared
    button shows only when enabled). The old Mesh button did the opposite: no
    ``reason=`` at all, and a manual ``set_tooltip("Ctrl+Enter")`` on hover
    whether or not Make 3D was enabled -- advertising a shortcut that does
    nothing while the button is dead.
    """
    source = inspect.getsource(create_brief._generate)

    call = re.search(r"widgets\.primary_button\((.*?)\n        \)\n", source, re.S)
    assert call is not None, "create_brief._generate no longer draws a primary_button"
    assert "reason=" in call.group(1)
    assert "tooltip=_generate_tooltip(" in call.group(1)
    assert "set_tooltip" not in source, "a hover tooltip not gated on `enabled`"
    # And Mesh reaches the same function with its own label and door.
    draw = inspect.getsource(create_brief.submit_control)
    assert '"Make 3D" if mesh else' in draw
    assert "settings_3d.promote(" in draw
