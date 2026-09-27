"""shell-home-library-01, the 2026-09-26 audit: the full-window Library's
missing export/convert popups.

``full.draw`` composes ``library._bulk`` (Export zip / Save to project /
Convert, drawn from ``_grid``) but never drew ``library._draw_export_popup``/
``_draw_convert_popup`` -- so opening either popup ran ``imgui.open_popup``
with no matching ``begin_popup_modal`` anywhere in the frame's id stack. It
never appeared, the task waiting on ``popup.decisions.get()`` blocked
forever, and ``library.popup_open`` stayed latched, greying every button and
locking Ctrl+K and the mode keys behind a dialog nobody could see.
"""

from __future__ import annotations

import inspect

from realmspinner.studio.modes.library.ui.panes import full as library_full


def test_the_full_window_library_draws_the_export_and_convert_popups_its_bulk_bar_parks():
    source = inspect.getsource(library_full.draw)
    assert "_draw_export_popup" in source, "full.draw never draws the export-plan popup"
    assert "_draw_convert_popup" in source, "full.draw never draws the convert popup"
