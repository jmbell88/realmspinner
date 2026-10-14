"""The 2026-10-14 audit's Character-column findings: poser-characters-01 (a Seed
row that changes nothing) and poser-characters-04 (camera and sheet words the
column swallows without saying so).

Both are the column promising more than a rigged-mesh character can do. The
manual says what was not acted on is listed, and the Seed row, Reroll and Lock
seed sit on a build that reads no seed at all.
"""

from __future__ import annotations

import hashlib
import inspect

import pytest

from realmspinner.characters import Recipe
from realmspinner.characters import instantiate as instantiate_mod
from realmspinner.studio.modes.create.engine import assets as create_assets
from realmspinner.studio.modes.create.engine import character as character_engine
from realmspinner.studio.modes.create.ui.panes import settings_character
from realmspinner.studio.state import AppState


def _form(prompt: str = "") -> dict:
    form = AppState().form_2d
    form["asset_type"] = "character"
    form["prompt"] = prompt
    create_assets.sync_legacy_fields(form)
    return form


def _digest(recipe: Recipe, out_dir) -> str:
    instantiate_mod.instantiate(recipe, out_dir)
    return hashlib.sha256((out_dir / instantiate_mod.MODEL_NAME).read_bytes()).hexdigest()


# --- poser-characters-01 ---------------------------------------------------------


def test_a_character_recipe_seed_changes_the_built_mesh_or_the_character_column_draws_no_seed(
    tmp_path,
):
    """Two ogres built with seeds 1 and 999999 have byte-identical ``model.glb``
    (the only reader of the seed was the flame effect, which nothing calls), yet
    the column drew Seed, Reroll and Lock seed. Either the seed has a job in the
    build or the column does not offer it (the 2026-10-14 audit, finding
    poser-characters-01)."""
    base = Recipe.from_dict({"family": "ogre"})
    first = _digest(base.replace(seed=1), tmp_path / "a")
    second = _digest(base.replace(seed=999999), tmp_path / "b")
    if first != second:
        return  # the seed shapes the mesh: the row earns its place
    source = inspect.getsource(settings_character.draw_block)
    assert "seed_row" not in source, "the seed changes nothing, yet the column draws it"


class _Ctx:
    def __init__(self, svc) -> None:
        self.svc = svc
        self.state = AppState()
        self.rigging_available = True

    def busy(self, _key: str) -> bool:
        return False


def test_the_form_sends_the_door_no_seed_it_would_not_read(svc):
    form = _form("a fire ogre")
    form["seed"] = 4242
    character_engine.sync_from_prompt(form)
    kwargs = character_engine.recipe_kwargs(form, character_engine.options(_Ctx(svc)))
    assert "seed" not in kwargs


def test_the_door_mints_no_recipe_seed_nothing_reads():
    from realmspinner.service import characters as door

    assert "spec.replace(seed=" not in inspect.getsource(door.create_character)


# --- poser-characters-04 ---------------------------------------------------------


def test_not_interpreted_names_a_camera_word_the_character_cannot_act_on():
    """"a fire ogre, isometric view, 8 directions" resolved ``camera_preset`` and
    consumed "8 directions" as noise, so ``not_interpreted`` was empty while the
    manual promises what was not acted on is listed. A rigged mesh has no camera
    and no direction count (the 2026-10-14 audit, finding poser-characters-04)."""
    form = _form("a fire ogre, isometric view, 8 directions")
    character_engine.sync_from_prompt(form)
    said = character_engine.not_interpreted(form)
    assert "isometric view" in said, said
    assert "8 directions" in said, said


def test_not_interpreted_names_a_sprite_sheet_word():
    form = _form("sprite sheet of a wolf")
    character_engine.sync_from_prompt(form)
    assert "sprite sheet" in character_engine.not_interpreted(form)


@pytest.mark.parametrize("prompt", ["a wolf", "a fire ogre", "a detailed pixel art knight"])
def test_a_brief_with_no_camera_or_sheet_word_lists_nothing(prompt):
    form = _form(prompt)
    character_engine.sync_from_prompt(form)
    assert character_engine.not_interpreted(form) == ()


def test_a_second_camera_word_is_listed_too():
    """The span that lost the camera slot (``applied=False``) did nothing as well."""
    form = _form("an ogre, top down, isometric")
    character_engine.sync_from_prompt(form)
    said = character_engine.not_interpreted(form)
    assert "top down" in said and "isometric" in said, said
