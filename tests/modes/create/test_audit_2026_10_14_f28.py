"""The 2026-10-14 audit's Create findings that eed32549 (Poser and the character
sheet removed) left behind: a canvas preview that raised for every job, a Character
hint promising a sprite sheet, hand-listed follow-up kinds, comments describing a
mode that no longer exists, and three small first-screen inconsistencies.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from imgui_bundle import imgui

import realmspinner
from realmspinner.kernels.rig import store
from realmspinner.studio.modes.create.engine import workspace as families
from realmspinner.studio.modes.create.ui import brief as create_brief
from realmspinner.studio.modes.create.ui import canvas as create_canvas
from realmspinner.studio.modes.create.ui import preview as create_preview
from realmspinner.studio.modes.create.ui import workspace as create_workspace
from realmspinner.studio.state import AppState

SRC = Path(realmspinner.__file__).parent


# --- create-01: the canvas preview against the trimmed rig store ---------------


def _sprite_draft(job_dir: Path) -> str:
    draft_id = store.new_id()
    directory = store.sprite_dir(job_dir)
    directory.mkdir(parents=True)
    record = {"id": draft_id, "sheet_type": "walk", "candidates": ["a"], "created": 1.0}
    (directory / f"{draft_id}.json").write_text(json.dumps(record), encoding="utf-8")
    (directory / f"{draft_id}.a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return draft_id


def test_create_preview_load_runs_against_the_trimmed_rig_store(tmp_path):
    """The 2026-10-14 audit, findings create-brief-01 and service-kinds-01:
    ``load`` still called four functions eed32549 deleted from the rig store, so
    it raised ``AttributeError`` for every job it was asked about."""
    assert create_preview.load(tmp_path, {}) == []


def test_create_preview_load_lists_a_tileset_and_a_sprite_draft_without_touching_retired_sheet_storage(  # noqa: E501
    tmp_path,
):
    for retired in ("list_sheets", "read_sheet_pixel", "sheet_png_path", "sheet_pixel_png_path"):
        assert not hasattr(store, retired), f"the store grew {retired} back"
    draft_id = _sprite_draft(tmp_path)
    params = {"sheet": {"tile_w": 16, "tile_h": 16, "columns": 2, "rows": 2, "layout": "blob47"}}
    records = create_preview.load(tmp_path, params)
    assert [r["name"] for r in records] == ["Tileset", "walk a"]
    assert records[0]["terrain"] is True and len(records[0]["cells"]) == 4
    assert records[1]["id"] == draft_id
    assert records[1]["path"] == store.sprite_draft_png_path(tmp_path, draft_id, "a")


def test_create_preview_names_no_deleted_store_function():
    tree = ast.parse((SRC / "studio/modes/create/ui/preview.py").read_text(encoding="utf-8"))
    attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "store"
    }
    missing = {name for name in attrs if not hasattr(store, name)}
    assert not missing, f"preview.py calls rig.store.{sorted(missing)}, which do not exist"


# --- service-21: the follow-up kinds are the follow-up stage table --------------


def test_create_workspace_followup_kinds_are_derived_from_the_followup_stage_table():
    """The 2026-10-14 audit, finding service-kinds-03: a hand list that named
    ``character`` (not a job kind) and was not pinned to ``FOLLOWUP_STAGES``, so a
    fifth live follow-up added there would be treated as a result row."""
    from realmspinner.retired import RETIRED_KINDS
    from realmspinner.studio.asset_open import FOLLOWUP_STAGES

    # Live follow-ups, plus the retired kinds a library may still hold (they are
    # grouped into their source's family and kept out of ``results``).
    assert frozenset(FOLLOWUP_STAGES) | RETIRED_KINDS == families.FOLLOWUP_KINDS
    assert "character" not in families.FOLLOWUP_KINDS | families.ASSET_KINDS


def test_a_followup_row_is_never_a_result_row_and_a_retired_one_still_groups():
    from realmspinner.studio.asset_open import FOLLOWUP_STAGES

    mesh = {"id": "m1", "kind": "image", "stage": "model", "created_at": 1, "params": {}}
    rows = [
        mesh,
        *(
            {
                "id": f"f-{kind}",
                "kind": kind,
                "stage": "model",
                "created_at": 2,
                "params": {"source_job": "m1"},
            }
            for kind in (*FOLLOWUP_STAGES, "charsheet")
        ),
    ]
    index = families.build_index(rows)
    assert len(index.families) == 1
    (key,) = index.families
    assert [j["id"] for j in families.results(index, key)] == ["m1"]


# --- create-05: the Character hint ----------------------------------------------


def test_the_character_type_hint_does_not_promise_a_sprite_sheet():
    """The 2026-10-14 audit, findings create-brief-02 and create-workspace-02: a
    character ends at a rigged mesh since eed32549, and the tooltip still said it
    was "animated into a sprite sheet"."""
    hint = create_brief.TYPE_HINTS["character"]
    assert "sprite" not in hint.lower()
    assert "rigged" in hint and "Clay" in hint
    assert "No GPU needed." in hint


def test_the_character_asset_comments_do_not_say_a_sheet_is_rendered():
    source = (SRC / "studio/modes/create/engine/assets.py").read_text(encoding="utf-8")
    assert "renders the cells" not in source
    assert "is a sprite sheet by the time" not in source


# --- create-13: a picked recent prompt clears the refusal -----------------------


def test_picking_a_recent_prompt_clears_the_prompt_refusal(monkeypatch):
    """The 2026-10-14 audit, finding create-brief-04: picking from Recent prompts
    wrote the form directly and skipped the ring-clear and the touched flag an
    edit takes."""
    cleared: list[str] = []
    state = SimpleNamespace(
        history=["a mossy well"],
        create=SimpleNamespace(prompt_touched=False),
        clear_field_error=cleared.append,
    )
    ctx = SimpleNamespace(state=state)
    form = {"prompt": ""}
    monkeypatch.setattr(
        create_brief,
        "imgui",
        SimpleNamespace(begin_popup=lambda _id: True, end_popup=lambda: None),
    )
    monkeypatch.setattr(create_brief.widgets, "popup_chrome", lambda **_k: None)
    monkeypatch.setattr(create_brief.controls, "menu_item", lambda *_a, **_k: (True, False))
    create_brief._history_popup(ctx, form)
    assert form["prompt"] == "a mossy well"
    assert cleared == ["prompt"]
    assert state.create.prompt_touched is True


# --- create-14: the Images pills never hide a stored count ----------------------


@pytest.fixture
def frames():
    previous = imgui.get_current_context()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value

    def draw(build: Any) -> None:
        imgui.new_frame()
        imgui.begin("smoke")
        try:
            build()
        finally:
            imgui.end()
            imgui.end_frame()
            imgui.render()

    yield draw
    imgui.destroy_context(ctx)
    if previous is not None:
        imgui.set_current_context(previous)


@pytest.mark.parametrize(
    ("stored", "shown"), [(1, 1), (2, 2), (3, 2), (5, 4), (6, 4), (7, 8), (8, 8)]
)
def test_the_count_pills_never_hide_an_off_ladder_stored_count(frames, monkeypatch, stored, shown):
    """The 2026-10-14 audit, finding create-brief-05: the door takes 1..8 and the
    pills offer 1, 2, 4 and 8, so a form holding 3 showed no pill selected and the
    arrow keys stepped from index 0."""
    seen: list[str] = []
    real = create_brief.controls.segmented_choice

    def spy(key, options, current, *a, **k):
        seen.append(current)
        assert current in {value for value, _label in options}, "no pill is selected"
        return real(key, options, current, *a, **k)

    monkeypatch.setattr(create_brief.controls, "segmented_choice", spy)
    ctx = SimpleNamespace(state=AppState(mode="create"))
    form = {"count": stored}
    frames(lambda: create_brief._count(ctx, form, create_brief._COUNTS, stored))
    assert seen == [str(shown)]
    assert form["count"] == shown, "a press must draw the count the pills display"


def test_nearest_count_prefers_the_smaller_pill_on_a_tie():
    assert create_brief.nearest_count((1, 2, 4, 8), 3) == 2
    assert create_brief.nearest_count((1, 2, 4, 8), 6) == 4
    assert create_brief.nearest_count((1, 2, 3), 9) == 3
    assert create_brief.nearest_count((1, 2, 3), 0) == 1


# --- create-27: the tray card and the canvas agree on Make 3D -------------------


def _reference(kind: str, **params: Any) -> dict[str, Any]:
    return {
        "id": "r1",
        "kind": kind,
        "stage": "reference",
        "status": "done",
        "files": ["input.png"],
        "params": params,
    }


def test_a_sprite_sheet_reference_card_has_no_make_3d_when_the_canvas_has_none():
    """The 2026-10-14 audit, finding create-workspace-07: the tray card enabled Make
    3D for any finished reference row with an input.png, the canvas and Next step
    only for ``MAKE_3D_KINDS``."""
    sprite = _reference("sprite_synthesis", asset_type="sprite_sheet")
    plain = _reference("image", asset_type="3d_model")
    ctx = SimpleNamespace(job=lambda: sprite)
    assert create_canvas.make_3d_job(ctx) is None
    assert create_workspace.make_3d_offered(sprite) is False
    ctx = SimpleNamespace(job=lambda: plain)
    assert create_canvas.make_3d_job(ctx) is plain
    assert create_workspace.make_3d_offered(plain) is True


def test_make_3d_offered_follows_the_canvas_kind_set_and_needs_a_picture():
    no_picture = {**_reference("image", asset_type="3d_model"), "files": []}
    mesh = {**_reference("image", asset_type="3d_model"), "stage": "model"}
    assert create_workspace.make_3d_offered(no_picture) is False
    assert create_workspace.make_3d_offered(mesh) is False


def test_the_tray_card_gates_make_3d_on_the_shared_predicate():
    source = (SRC / "studio/modes/create/ui/workspace.py").read_text(encoding="utf-8")
    card = source[source.index("def _result_card") :]
    card = card[: card.index("\n\n\n")]
    assert "make_3d_offered(job)" in card
    assert 'is_reference = job.get("stage")' not in card


# --- create-18, create-15: comments naming a mode that is gone ------------------

#: Files this finding closed. Each phrase names a module or a role eed32549 removed.
_STALE_POSER = re.compile(
    r"poser_mode|poser_library|Poser's|inspector in Poser|Poser wears|Poser doors|entered Poser"
)
_OWNED = (
    "studio/modes/create/engine/recipe.py",
    "studio/modes/create/engine/workspace.py",
    "studio/modes/create/engine/assets.py",
    "studio/modes/create/ui/brief.py",
    "studio/modes/create/ui/preview.py",
    "studio/modes/create/ui/stages.py",
    "studio/modes/create/ui/workspace.py",
    "studio/modes/create/ui/canvas.py",
    "studio/modes/create/ui/panes/settings_character.py",
    "studio/panes/pose_panel.py",
    "studio/asset_exits.py",
    "studio/shell/frame.py",
    "studio/shell/tasks.py",
    "service/characters.py",
    "service/_jobs_create.py",
)


@pytest.mark.parametrize("relative", _OWNED)
def test_no_create_module_describes_the_removed_poser_mode_as_live(relative):
    text = (SRC / relative).read_text(encoding="utf-8")
    hits = sorted(set(_STALE_POSER.findall(text)))
    assert not hits, f"{relative} still speaks of the removed Poser mode: {hits}"


def test_comments_in_create_panes_name_no_deleted_poser_module():
    for relative in _OWNED:
        text = (SRC / relative).read_text(encoding="utf-8")
        for name in re.findall(r"\b(poser_\w+)\b", text):
            assert (SRC / "studio" / f"{name}.py").exists(), f"{relative} cites deleted {name}"
    assert "2600-line" not in (SRC / _OWNED[8]).read_text(encoding="utf-8")
