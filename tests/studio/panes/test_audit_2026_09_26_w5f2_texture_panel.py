"""Regression test for the 2026-09-26 audit, finding create-panes-08.

``texture_panel``'s retexture "Anchor strength" slider used the bare id
``control_scale`` -- also ``settings_2d.py``'s bare id for the 2D pane's
ControlNet conditioning slider (``settings_2d.py:915-925``). Both panes read
and write ``ctx.state.field_errors``/``clear_field_error`` through that one
flat, unnamespaced dict, so a refusal or an edit on either field rang or
cleared the other's ring whenever a mesh's inspector and Create's Reference
stage were both open -- the exact shape ``_FIELD_PREFIX`` already exists to
prevent for "strength", "texture_size" and "prompt".
"""

from __future__ import annotations

import inspect
import re

from realmspinner.studio.modes.create.ui.panes import settings_2d
from realmspinner.studio.panes import texture_panel


def test_control_scale_is_relabelled_in_the_field_prefix_table():
    assert "control_scale" in texture_panel._FIELD_PREFIX
    assert texture_panel._FIELD_PREFIX["control_scale"] != "control_scale"


def test_the_anchor_strength_slider_does_not_use_the_bare_control_scale_id():
    source = inspect.getsource(texture_panel.draw)
    prefixed = texture_panel._FIELD_PREFIX["control_scale"]
    # Isolate the "Anchor strength" slider's own call -- the historical
    # comment above it (create-01, 2026-09-06) legitimately mentions the bare
    # name in prose, so the assertion below targets the widget call itself,
    # not every mention of the string in the function.
    match = re.search(
        r'form_ui\.slider\(\s*(?:#[^\n]*\n\s*)*"([^"]+)"\s*,\s*"Anchor strength"', source
    )
    assert match is not None, "could not find the Anchor strength slider's call"
    assert match.group(1) == prefixed, (
        f"the Anchor strength slider is drawn under {match.group(1)!r}, not "
        f"the prefixed {prefixed!r} -- it collides with settings_2d's own "
        "field of the same name"
    )


def test_settings_2d_still_uses_the_bare_control_scale_id_which_is_the_collision():
    """The control this module's fix protects against: Create's 2D recipe
    pane genuinely does use the bare ``control_scale`` id, unprefixed, so the
    collision this test module is named for is real and not hypothetical."""
    source = inspect.getsource(settings_2d)
    assert 'widgets.field_error(ctx.state, "control_scale")' in source
