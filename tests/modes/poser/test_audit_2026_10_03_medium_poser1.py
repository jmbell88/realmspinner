"""The 2026-10-03 audit's poser-12, -17, -18 and -20 (Medium).

poser-12: the palette-flicker score is an exact-RGB histogram distance, which
only means "colours changed" on a quantised sheet. On an HD sheet (never
snapped to a palette) almost no exact value recurs between frames, so a
sprite moving a fraction of a pixel scored red.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from realmspinner.kernels.rig import store
from realmspinner.studio.modes.poser import mode as poser_mode
from realmspinner.studio.modes.poser.engine import qa
from realmspinner.studio.modes.poser.ui.panes import send as poser_send

from .test_send_door import _Ctx, _mesh
from .test_sheet_mode import _bind, _png_character, _SubmitCtx

CELL = 16
COLUMNS = 8


def _layout(frames=8):
    return {
        "version": 2,
        "movements": [
            {"key": "walk", "frames": frames, "loop": True, "directions": [{"key": "front"}]}
        ],
        "runs": [{"movement": "walk", "direction": "front", "start": 0, "end": frames - 1}],
        "cell_count": frames,
    }


def _shaded_atlas(frames=8):
    """A smoothly shaded square that slides a quarter of a pixel per frame:
    the silhouette never moves, every exact RGB value does."""
    rows = (frames + COLUMNS - 1) // COLUMNS
    atlas = np.zeros((rows * CELL, COLUMNS * CELL, 4), dtype=np.uint8)
    ys, xs = np.mgrid[0:CELL, 0:CELL].astype(np.float64)
    for index in range(frames):
        x0, y0, x1, y1 = qa.cell_box(index, COLUMNS, CELL, CELL)
        shift = 0.25 * index
        crop = atlas[y0:y1, x0:x1]
        crop[4:12, 4:12, 0] = np.clip(20 + 9.0 * (xs - shift) + 3.0 * ys, 0, 255)[4:12, 4:12]
        crop[4:12, 4:12, 1] = np.clip(40 + 5.0 * (xs - shift) + 7.0 * ys, 0, 255)[4:12, 4:12]
        crop[4:12, 4:12, 2] = np.clip(60 + 7.0 * (ys - shift) + 2.0 * xs, 0, 255)[4:12, 4:12]
        crop[4:12, 4:12, 3] = 255
    return atlas


def test_a_smoothly_shaded_sprite_moving_a_fraction_of_a_pixel_does_not_flag_palette_flicker():
    atlas = _shaded_atlas()
    score = qa.score_sheet(
        atlas, _layout(), columns=COLUMNS, frame_w=CELL, frame_h=CELL, pixel_art=False
    )
    assert score.flagged == 0
    assert all("palette_flicker" not in cell.metrics for cell in score.cells)
    assert all("flicker" not in cell.flags for cell in score.cells)
    assert "palette_flicker" in score.skipped


def test_the_palette_flicker_score_is_unchanged_on_a_pixel_art_sheet():
    atlas = _shaded_atlas()
    score = qa.score_sheet(atlas, _layout(), columns=COLUMNS, frame_w=CELL, frame_h=CELL)
    assert score.skipped == ()
    assert any("palette_flicker" in cell.metrics for cell in score.cells)


def test_an_hd_sheets_sidecar_makes_the_poser_scorer_skip_palette_flicker(svc):
    ctx = _SubmitCtx(svc)
    job_id, made = _png_character(svc)
    record_path = store.sheet_path(svc.job_dir(job_id), made[0])
    record = json.loads(record_path.read_text("utf-8"))
    record["pixel_art"] = False
    record_path.write_text(json.dumps(record), "utf-8")
    # Replace the flat atlas with the shaded, sliding one the sheet's first run covers.
    png = store.sheet_png_path(svc.job_dir(job_id), made[0])
    with Image.open(png) as flat:
        size = flat.size
    shaded = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    shaded[: CELL * 2, : CELL * 8] = np.tile(_shaded_atlas(8)[:CELL, : CELL * 8], (2, 1, 1))
    Image.fromarray(shaded, "RGBA").save(png)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])
    assert poser_mode.scores(ctx) is None
    _key, fn, args, _kwargs = ctx.submitted[0]
    result = fn(*args)
    assert result.cells
    assert all("palette_flicker" not in cell.metrics for cell in result.cells)


# -- poser-17 / poser-18: the Send dialog's layout is the submitted layout -----
#
# poser-17: the cost note counted the Poser form's standing layout while _send
# submitted a different one for any mesh that is not the bound character, so
# "32 cells" sat over a request for 256. poser-18: _send wrote that other
# layout into the shared form, resetting the bound character's hand edits.


@pytest.fixture
def ctx(svc):
    return _Ctx(svc)


def _bind_with_one_movement_ticked(ctx, svc):
    """A character bound in Poser whose form ticks exactly one movement."""
    bound = _mesh(svc, rigged=True)
    state = poser_mode.ensure(ctx)
    state.job_id = bound["id"]
    state.template = "humanoid"
    form = poser_mode.sheet_form(ctx)
    for row in form["layout"]["movements"]:
        row["enabled"] = False
    form["layout"]["movements"][0]["enabled"] = True
    return form


def _capture_sends(monkeypatch):
    sent: list[dict] = []

    def record(ctx, job, form=None):
        sent.append(dict(form or {}))
        return True

    monkeypatch.setattr(poser_mode, "render_character_sheet", record)
    return sent


def _shown_note(monkeypatch, ctx, state, form):
    """Run the dialog's cost-note half with no imgui frame and return the note."""
    notes: list[str] = []
    monkeypatch.setattr(
        poser_send,
        "imgui",
        SimpleNamespace(
            dummy=lambda *a, **k: None,
            same_line=lambda *a, **k: None,
            close_current_popup=lambda: None,
        ),
    )
    monkeypatch.setattr(poser_send.widgets, "cost_note", notes.append)
    monkeypatch.setattr(poser_send.controls, "button", lambda *a, **k: False)
    poser_send._actions(ctx, state, form)
    return notes[0]


def test_the_send_dialogs_cell_count_is_the_count_of_the_layout_it_submits_for_an_unbound_mesh(
    ctx, svc, monkeypatch
):
    form = _bind_with_one_movement_ticked(ctx, svc)
    standing = poser_mode.cell_count(form)
    sent = _capture_sends(monkeypatch)

    other = _mesh(svc, rigged=True)
    poser_send.ask(ctx, other)
    state = ctx.state.poser_send
    note = _shown_note(monkeypatch, ctx, state, form)
    poser_send._send(ctx, state, form)

    submitted = poser_mode.cell_count(sent[-1])
    assert submitted != standing, "premise: the two layouts really do disagree"
    assert note.startswith(f"{submitted} cells"), note


def test_sending_an_unbound_mesh_leaves_the_bound_characters_edited_layout_in_the_form(
    ctx, svc, monkeypatch
):
    form = _bind_with_one_movement_ticked(ctx, svc)
    standing_layout = form["layout"]
    before = json.dumps(standing_layout, sort_keys=True)
    sent = _capture_sends(monkeypatch)

    other = _mesh(svc, rigged=True)
    poser_send.ask(ctx, other)
    poser_send._send(ctx, ctx.state.poser_send, form)

    assert form["layout"] is standing_layout
    assert json.dumps(form["layout"], sort_keys=True) == before
    assert sent[-1]["layout"] is not standing_layout


def test_sending_the_bound_mesh_still_submits_the_forms_own_layout(ctx, svc, monkeypatch):
    form = _bind_with_one_movement_ticked(ctx, svc)
    sent = _capture_sends(monkeypatch)
    bound_id = poser_mode.ensure(ctx).job_id

    poser_send.ask(ctx, {"id": bound_id, "prompt": "bound", "files": ["rig.glb"]})
    poser_send._send(ctx, ctx.state.poser_send, form)

    assert sent[-1]["layout"] is form["layout"]


# -- poser-20: the nav keys are Poser's only while a sheet is on screen --------


def test_poser_does_not_reserve_the_nav_keys_while_no_sheet_is_on_screen():
    from realmspinner.studio import modes

    pose_view = SimpleNamespace(poser=poser_mode.PoserState())
    assert pose_view.poser.sheet_view is False
    assert modes.reserves_nav_keys("poser", pose_view) is False
    assert modes.reserves_nav_keys("poser", SimpleNamespace(poser=None)) is False

    sheet = poser_mode.PoserState()
    sheet.sheet_view = True
    assert modes.reserves_nav_keys("poser", SimpleNamespace(poser=sheet)) is True


def test_the_other_reserving_modes_still_reserve_unconditionally():
    from realmspinner.studio import modes

    state = SimpleNamespace(poser=None)
    for mode in sorted(modes.NAV_KEY_MODES - {"poser"}):
        assert modes.reserves_nav_keys(mode, state) is True, mode
    assert modes.reserves_nav_keys("clay", state) is False


def test_the_frame_loop_asks_the_function_not_the_set():
    import inspect

    from realmspinner.studio.shell import frame

    source = inspect.getsource(frame)
    assert "modes.reserves_nav_keys(" in source
    assert "in modes.NAV_KEY_MODES" not in source
