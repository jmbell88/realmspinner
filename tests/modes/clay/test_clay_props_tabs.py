"""Properties is tabs now: Object, Modifiers, Material, Data and Scene.

The pane was one long scroll in which the mesh check sat between the modifier
stack and the material, and the game check lived two panes away under "Model
file". These claims pin the new homes; the pressed-through-real-imgui one is in
``tests/studio/test_studio_smoke.py``.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from realmspinner.kernels.mesh import engines as engines_mod
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import state as clay_state
from realmspinner.studio.modes.clay.ui.panes import bridge as clay_bridge
from realmspinner.studio.modes.clay.ui.panes import props as clay_props


def test_tab_keys_are_unique_and_the_default_is_one_of_them() -> None:
    keys = [key for key, *_ in clay_props.TABS]
    assert len(keys) == len(set(keys))
    assert clay_props.DEFAULT_TAB in keys
    assert set(keys) == {"object", "modifiers", "material", "data", "scene"}


def test_a_fresh_state_opens_on_object_and_an_unknown_tab_falls_back_to_it() -> None:
    state = clay_state.ClayState()
    assert clay_props.tab_key(state) == "object"
    state.props_tab = "retired-tab"
    assert clay_props.tab_key(state) == "object"
    state.props_tab = "scene"
    assert clay_props.tab_key(state) == "scene"


def test_every_tab_has_a_glyph_and_a_sentence_for_its_tooltip() -> None:
    for key, label, glyph, what in clay_props.TABS:
        assert label and glyph and what, key


def test_the_game_check_left_the_document_pane_for_the_scene_tab() -> None:
    """It measures the whole document, so it sits beside the export engine and
    not under the Recent list of a pane titled "Model file"."""
    assert "game_check" not in inspect.getsource(clay_bridge.draw)
    assert "game_check" in inspect.getsource(clay_props._scene)
    assert not hasattr(clay_bridge, "_game_check")


def test_the_export_engine_has_a_control_in_the_scene_tab() -> None:
    source = inspect.getsource(clay_props._scene)
    assert "set_export_engine" in source
    assert "ENGINES" in source


def test_the_export_engine_choice_round_trips_through_settings() -> None:
    store: dict = {}
    settings = SimpleNamespace(get=lambda key: store.get(key), set=store.__setitem__)
    ctx = SimpleNamespace(settings=settings)
    for key in engines_mod.ENGINES:
        clay_mode.set_export_engine(ctx, key)
        assert clay_mode.export_engine(ctx) == key
