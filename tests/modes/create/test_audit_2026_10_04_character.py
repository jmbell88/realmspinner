"""The 2026-10-04 audit's Character-column findings (create-23, -28, -29, -55).

All four are the Character column telling the user less than it knew: an action
the resolver understood and the sheet does not carry vanished without a word
(create-23), a Look picked under one species rode onto a species that cannot
paint it (create-28), a refusal's red ring came back the frame after the user
fixed the control (create-29), and the offer button misspelt the article its own
sentence had right (create-55).
"""

from __future__ import annotations

import pytest

from realmspinner.characters import resolve as resolve_mod
from realmspinner.characters.family import families
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


class _Ctx:
    def __init__(self, svc) -> None:
        self.svc = svc
        self.state = AppState()
        self.rigging_available = True

    def busy(self, _key: str) -> bool:
        return False


@pytest.fixture
def ctx(svc):
    return _Ctx(svc)


# --- create-23 ------------------------------------------------------------------


def test_an_understood_action_the_sheet_does_not_carry_is_listed_as_not_interpreted(
    monkeypatch,
):
    """"a running wolf": the resolver understood ``run``, ``_fill`` kept only the
    three movements a sheet carries, and "Not interpreted" read
    ``resolution.unrecognised`` -- empty -- so the sheet had no run and nothing
    said so (the 2026-10-04 audit, create-23)."""
    said: list[str] = []
    monkeypatch.setattr(settings_character.widgets, "muted_wrapped", said.append)
    form = _form("a running wolf")
    character_engine.sync_from_prompt(form)
    resolution = character_engine.resolution_of(form)
    carried = {name for name, _frames in character_engine.MOVEMENTS}
    dropped = [a for a in resolution.actions if a not in carried]
    assert dropped, "the premise: the resolver understood an action the sheet lacks"

    settings_character._unrecognised(form)

    assert said, "nothing on screen says the action was dropped"
    for action in dropped:
        assert action in said[0]


def test_a_carried_action_is_not_listed_as_not_interpreted(monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(settings_character.widgets, "muted_wrapped", said.append)
    form = _form("a walking wolf")
    character_engine.sync_from_prompt(form)
    settings_character._unrecognised(form)
    assert not said


# --- create-28 ------------------------------------------------------------------


def _species_pair_with_a_look_only_one_paints():
    """(painter, other, theme, painter_alias, other_alias), off the registry."""
    fams = families()
    for a_key, a in fams.items():
        for b_key, b in fams.items():
            only = sorted({t.key for t in a.themes} - {t.key for t in b.themes})
            if a_key != b_key and only and a.aliases and b.aliases:
                return a_key, b_key, only[0], a.aliases[0], b.aliases[0]
    raise AssertionError("no two species differ in their looks")


def test_a_look_the_user_picked_is_dropped_when_a_prompt_edit_moves_the_species_to_one_that_does_not_paint_it():  # noqa: E501
    painter, other, theme, painter_alias, other_alias = _species_pair_with_a_look_only_one_paints()
    form = _form(f"a {painter_alias}")
    character_engine.sync_from_prompt(form)
    assert form["character_family"] == painter
    form["character_theme"] = theme
    character_engine.touched(form, "character_theme")

    form["prompt"] = f"a {other_alias}"
    character_engine.sync_from_prompt(form)

    assert form["character_family"] == other
    assert form["character_theme"] == character_engine.THEME_UNSET, (
        "a Look the new species cannot paint must not ride along: the door refuses it"
    )


def test_a_theme_the_species_does_not_paint_is_a_problem_on_the_look_control(ctx):
    """A restored form can still carry the pair; Generate must say why it is
    refused rather than leave the Look combo showing "The species' own"."""
    _painter, _other, theme, _pa, other_alias = _species_pair_with_a_look_only_one_paints()
    form = _form(f"a {other_alias}")
    character_engine.sync_from_prompt(form)
    form["character_theme"] = theme
    character_engine.touched(form, "character_theme")

    found = [
        p
        for p in character_engine.problems(ctx, form)
        if getattr(p, "field", "") == "character_theme"
    ]
    assert len(found) == 1


def test_a_theme_the_species_paints_is_not_a_problem(ctx):
    _painter, _o, theme, painter_alias, _oa = _species_pair_with_a_look_only_one_paints()
    form = _form(f"a {painter_alias}")
    character_engine.sync_from_prompt(form)
    form["character_theme"] = theme
    assert not [
        p
        for p in character_engine.problems(ctx, form)
        if getattr(p, "field", "") == "character_theme"
    ]


# --- create-29 ------------------------------------------------------------------


class _EditingForm:
    """Answers every segmented control as if the user had just changed it."""

    def __init__(self, value: str) -> None:
        self.value = value

    def segmented_choice(self, *_a, **_k):
        return True, self.value


def test_editing_a_ringed_character_control_clears_its_refusal_for_good(ctx):
    """The door refuses ``logical_size``; ``mirror_errors`` copies it onto
    ``character_pixel``; the edit handler cleared only the copy, so the next
    frame's mirror re-filed the alias and the ring came back (the 2026-10-04
    audit, create-29)."""
    opts = character_engine.options(ctx)
    ctx.state.field_errors["logical_size"] = "Sprite size must be one of [32, 64]"
    form = _form("a wolf")

    character_engine.mirror_errors(ctx)
    assert "character_pixel" in ctx.state.field_errors  # frame 1: the ring

    settings_character._pixels(ctx, form, _EditingForm("32"), opts)  # the edit
    character_engine.mirror_errors(ctx)  # frame 2

    assert "character_pixel" not in ctx.state.field_errors
    assert "logical_size" not in ctx.state.field_errors


def test_every_character_control_edit_clears_its_recipe_aliases_too(ctx):
    for control, aliases in character_engine.RECIPE_FIELDS.items():
        for alias in aliases:
            ctx.state.field_errors[alias] = "refused"
        ctx.state.field_errors[control] = "refused"
        character_engine.clear_refusal(ctx.state, control)
        assert not ctx.state.field_errors, control


# --- create-55 ------------------------------------------------------------------


def test_the_offer_button_uses_the_same_article_as_the_offer_sentence(ctx, monkeypatch):
    form = _form("a minotaur")
    character_engine.sync_from_prompt(form)
    resolution = character_engine.resolution_of(form)
    sentence = resolve_mod.offer_sentence(resolution)
    assert sentence and "an ogre" in sentence

    labels: list[str] = []

    def fake_button(label, **_k):
        labels.append(label)
        return False

    monkeypatch.setattr(settings_character.controls, "button", fake_button)
    settings_character._offer_fixes(ctx, form)

    offer = next(label for label in labels if "character-offer" in label)
    assert offer.startswith("Make it an ogre"), offer
