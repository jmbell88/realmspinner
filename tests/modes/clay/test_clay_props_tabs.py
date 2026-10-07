"""Properties is tabs: Object, Material, Document and UV.

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
    assert set(keys) == {"object", "material", "document", "uv"}
    assert keys == ["object", "material", "document", "uv"], "the order the strip draws them in"


def test_a_fresh_state_opens_on_object_and_an_unknown_tab_falls_back_to_it() -> None:
    state = clay_state.ClayState()
    assert clay_props.tab_key(state) == "object"
    state.props_tab = "retired-tab"
    assert clay_props.tab_key(state) == "object"
    state.props_tab = "document"
    assert clay_props.tab_key(state) == "document"
    state.props_tab = "uv"
    assert clay_props.tab_key(state) == "uv", "UV is a tab now, not a pane of its own"


def test_a_saved_setting_naming_a_removed_tab_falls_back_to_object() -> None:
    state = clay_state.ClayState()
    for retired in ("modifiers", "data", "scene"):
        state.props_tab = retired
        assert clay_props.tab_key(state) == "object", retired


def test_every_tab_is_named_in_words_and_has_a_sentence_for_its_tooltip() -> None:
    """The strip used to be three glyphs (a box, a palette, a cog) whose names
    were only in a tooltip. The label is the text drawn now, so it must be a
    real word and not a glyph left over from the old table."""
    for key, label, what in clay_props.TABS:
        assert label.isalpha() and what, key
        assert label.lower() == key or key == "uv", key
