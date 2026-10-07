"""Properties is tabs: Object, Material and Document.

The pane was one long scroll once. These claims pin the tab table and the
fallback a saved setting naming a retired tab (Modifiers, Data, Scene) lands
on; the pressed-through-real-imgui one is in ``tests/studio/test_studio_smoke.py``.
"""

from __future__ import annotations

from realmspinner.studio.modes.clay import state as clay_state
from realmspinner.studio.modes.clay.ui.panes import props as clay_props


def test_tab_keys_are_unique_and_the_default_is_one_of_them() -> None:
    keys = [key for key, *_ in clay_props.TABS]
    assert len(keys) == len(set(keys))
    assert clay_props.DEFAULT_TAB in keys
    assert set(keys) == {"object", "material", "document"}


def test_a_fresh_state_opens_on_object_and_an_unknown_tab_falls_back_to_it() -> None:
    state = clay_state.ClayState()
    assert clay_props.tab_key(state) == "object"
    state.props_tab = "retired-tab"
    assert clay_props.tab_key(state) == "object"
    state.props_tab = "document"
    assert clay_props.tab_key(state) == "document"


def test_a_saved_setting_naming_a_removed_tab_falls_back_to_object() -> None:
    state = clay_state.ClayState()
    for retired in ("modifiers", "data", "scene"):
        state.props_tab = retired
        assert clay_props.tab_key(state) == "object", retired


def test_every_tab_has_a_glyph_and_a_sentence_for_its_tooltip() -> None:
    for key, label, glyph, what in clay_props.TABS:
        assert label and glyph and what, key
