"""Poser's character-sheet section: what it plays, and what it does not hold.

Troupe's own ``mode.py`` test suite, ported by P9 (2026-09-18) onto the
functions that folded into ``poser_mode`` -- minus the cross-character cast
list and the "characters still on their way" tracking
(``characters``/``cast_and_pending``/``in_progress``/``sendable_meshes`` and
everything downstream of them), which decision 1 of the folding brief drops
outright: binding an asset through the existing "Rigged assets" picker is the
one way in now, so there is no second, independent selection for this file to
pin.

Three claims carry what is left:

* **the preview is a clock**, so a slow frame skips cells rather than falling
  behind, and two machines running at different frame rates play a run cycle at
  the same speed;
* **which cell is on screen comes from the frame table**, never from arithmetic
  over the animation lengths -- a third copy of that arithmetic is a third thing
  nobody owns;
* **the sheet section is not journal-tracked**, which is why nothing in
  ``poser_mode.JOURNAL`` ever reads a ``sheet_*`` field, and why entering the
  mode from Home creates nothing.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pygame
import pytest

from realmspinner.kernels import charsheet
from realmspinner.kernels.rig import store
from realmspinner.studio.modes.poser import mode as poser_mode
from realmspinner.studio.modes.poser.engine import spec as sheet_spec


class _Ctx:
    """The narrow slice of the app context this section's logic touches.

    ``FakeCtx``'s inline-submit pattern (``tests/modes/poser/test_poser_mode.py``):
    a submitted callable runs immediately, which is what lets
    :func:`poser_mode.open_asset`'s own ``refresh``/``clips_refresh``/
    ``refresh_asset_poses`` calls -- unavoidable now that binding an asset is
    this section's own door in -- run against the real ``svc`` fixture rather
    than crashing on a context double with no ``submit`` at all.
    """

    def __init__(self, svc):
        self.svc = svc
        self.cache = svc.store
        self.state = SimpleNamespace(poser=None, preview={}, mode="poser")
        self.viewer = None
        self.poser_viewer = None
        self.rig_default = "humanoid"
        self.toasts: list[tuple[str, str]] = []

    def job_dir(self, job_id):
        return self.svc.job_dir(job_id)

    def toast(self, text, level="info", *a, **k):
        self.toasts.append((text, level))

    def busy(self, key):
        return False

    def submit(self, key, fn, *args, tag=None, **kwargs):
        return True


@pytest.fixture
def ctx(svc):
    return _Ctx(svc)


def _character(svc, *, sheets=1, size=32):
    """A finished mesh with ``sheets`` character sheets and a row per sheet."""
    job_id = svc.store.create("image", "a hooded ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    svc.store.set_status(job_id, "done")
    made = []
    for index in range(sheets):
        sheet_id = store.new_id()
        path = store.sheet_path(job_dir, sheet_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "id": sheet_id,
                    "columns": charsheet.COLUMNS,
                    "rows": 32,
                    "frame_size": size + index * 16,
                    "created": 100.0 + index,
                    "animation": charsheet.animation_block(),
                }
            ),
            "utf-8",
        )
        store.sheet_png_path(job_dir, sheet_id).write_bytes(b"atlas")
        row = svc.store.create(
            "charsheet", "a hooded ranger", {"source_job": job_id, "sheet_id": sheet_id}
        )
        svc.store.set_status(row, "done")
        made.append(sheet_id)
    return job_id, made


def _v2_character(svc):
    job_id, made = _character(svc)
    path = store.sheet_path(svc.job_dir(job_id), made[0])
    record = json.loads(path.read_text("utf-8"))
    record["troupe"] = charsheet.resolve_layout(
        {
            "version": 2,
            "movements": [
                {"key": "idle", "frames": 3, "directions": 1},
                {"key": "walk", "frames": 6, "directions": 4},
            ],
        }
    ).as_dict()
    path.write_text(json.dumps(record), "utf-8")
    return job_id, made


def _bind(ctx, job_id):
    """Point the section at an asset without going through the real
    ``open_asset`` bind (a GL/rig-read door this file's claims do not need) --
    the same shortcut Troupe's own tests took by writing ``state.job_id``
    directly, adapted since that field is Poser's own bound-asset id now."""
    poser_mode.ensure(ctx).job_id = job_id


def test_v2_preview_uses_the_selected_sheets_movements_and_runs(ctx, svc):
    job_id, made = _v2_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])
    state = poser_mode.ensure(ctx)
    state.sheet_animation, state.sheet_direction, state.sheet_frame = "walk", "back", 5
    assert poser_mode.cell_index(ctx) == 20
    assert int(poser_mode.preview_movement(ctx)["frames"]) == 6


def test_v2_selection_reconciles_a_movement_missing_from_the_sheet(ctx, svc):
    state = poser_mode.ensure(ctx)
    state.sheet_animation, state.sheet_direction, state.sheet_frame = "jump", "back_right", 5
    job_id, made = _v2_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])
    assert (state.sheet_animation, state.sheet_direction, state.sheet_frame) == (
        "idle", "front", 0,
    )


# -- the sheet form's skeleton -------------------------------------------

#: Two templates whose vocabularies genuinely differ, so a test can tell which
#: one a set of rows came from by name alone -- the shipped humanoid and
#: quadruped libraries hold the same ten names today (only their frames,
#: timing and provisional flags differ), which would not catch a form that
#: read the wrong one.
_FAKE_LAYOUT_OPTIONS = {
    "defaults": {"template": "humanoid"},
    "clip_vocabulary": {
        "humanoid": [{"name": "walk", "frames": 8, "default": True}],
        "quadruped": [{"name": "gallop", "frames": 12, "default": True}],
    },
}


def _fake_layout_options(ctx):
    del ctx
    return dict(_FAKE_LAYOUT_OPTIONS)


def test_the_movement_rows_follow_the_rigs_own_skeleton(ctx, svc, monkeypatch):
    """A job rigged on the second template gets the second template's rows,
    not the door's default.

    Ported onto ``PoserState.template`` directly -- ``open_asset`` is what
    records a bound asset's own template at bind time now
    (:func:`poser_mode._bound_sheet_template`'s own docstring), which this
    test simulates rather than driving the real GL-adjacent bind.
    """
    monkeypatch.setattr(poser_mode, "sheet_options", _fake_layout_options)
    mesh_id = svc.store.create("image", "a wolf", {}, stage="model")
    svc.store.create(
        "rig", "a wolf", {"source_job": mesh_id, "template": "quadruped"}, status="done"
    )
    state = poser_mode.ensure(ctx)
    state.job_id = mesh_id
    state.template = "quadruped"

    layout = poser_mode._default_sheet_layout(ctx)
    assert layout["template"] == "quadruped"
    assert {m["key"] for m in layout["movements"]} == {"gallop"}


def test_rows_fall_back_to_the_default_skeleton_only_without_a_rig(ctx, svc, monkeypatch):
    """No character bound falls back to the door's default -- and nothing
    else does."""
    monkeypatch.setattr(poser_mode, "sheet_options", _fake_layout_options)

    layout = poser_mode._default_sheet_layout(ctx)
    assert layout["template"] == "humanoid"
    assert {m["key"] for m in layout["movements"]} == {"walk"}


def test_the_form_rebuilds_its_layout_when_the_bound_rig_changes(ctx, svc, monkeypatch):
    """``sheet_form`` calls ``_default_sheet_layout`` again once the bound
    character's own template no longer matches the one the current rows were
    built from."""
    monkeypatch.setattr(poser_mode, "sheet_options", _fake_layout_options)
    first = poser_mode.sheet_form(ctx)
    assert first["layout"]["template"] == "humanoid"

    mesh_id = svc.store.create("image", "a wolf", {}, stage="model")
    svc.store.create(
        "rig", "a wolf", {"source_job": mesh_id, "template": "quadruped"}, status="done"
    )
    state = poser_mode.ensure(ctx)
    state.job_id = mesh_id
    state.template = "quadruped"

    second = poser_mode.sheet_form(ctx)
    assert second is first, "the same form dict, rebuilt in place"
    assert second["layout"]["template"] == "quadruped"
    assert {m["key"] for m in second["layout"]["movements"]} == {"gallop"}


# -- the one door in ---------------------------------------------------------


def test_open_character_sheet_enters_poser_pointed_at_the_sheet(ctx, svc):
    job_id, sheets = _character(svc)
    ctx.state.mode = "library"

    assert poser_mode.open_character_sheet(ctx, job_id, sheets[0]) is True
    assert ctx.state.mode == "poser"
    state = poser_mode.ensure(ctx)
    assert state.sheet_id == sheets[0]
    assert state.job_id == job_id
    assert state.sheet_view is True


def test_open_character_sheet_refuses_rather_than_switching_into_an_empty_room(ctx, svc):
    """The sheet section would draw "No character on screen", which is the
    blank arrival the whole routing change exists to stop."""
    job_id = svc.store.create("image", "a ranger", {}, stage="model")
    ctx.state.mode = "library"

    assert poser_mode.open_character_sheet(ctx, job_id) is False
    assert ctx.state.mode == "library"


def test_only_character_sheets_are_listed_under_a_character(ctx, svc):
    """A mesh can also hold ordinary pose sheets. They have no animation block,
    no direction runs and nothing this section can play."""
    job_id, sheets = _character(svc)
    plain = store.new_id()
    store.sheet_path(svc.job_dir(job_id), plain).write_text(
        json.dumps({"id": plain, "columns": 8, "rows": 1, "frame_size": 64}), "utf-8"
    )

    listed = [r["id"] for r in poser_mode.sheets(ctx, job_id)]
    assert listed == sheets
    assert plain not in listed


def test_the_sheet_directory_is_not_re_read_every_frame(ctx, svc, monkeypatch):
    """``sheets`` is a glob plus a stat plus a read plus a JSON parse per sheet
    and the panes call it from their draw; ``active_sheet`` is another read.
    Between them Troupe hit the disk three or four times a frame for a
    directory that changes when a sheet is *built*."""
    from realmspinner.kernels.rig import store as rig_store

    job_id, listed = _character(svc, sheets=2)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx)

    globs: list[int] = []
    reads: list[int] = []
    real_list, real_read = rig_store.list_sheets, rig_store.read_sheet
    monkeypatch.setattr(
        rig_store,
        "list_sheets",
        lambda d: (globs.append(1), real_list(d))[1],
    )
    monkeypatch.setattr(
        rig_store,
        "read_sheet",
        lambda d, i: (reads.append(1), real_read(d, i))[1],
    )

    for _ in range(10):
        poser_mode.sheets(ctx, job_id)
        poser_mode.active_sheet(ctx)
    assert globs == [] and reads == [], "the selection already read both"

    # And a selection is read at once rather than waited out, which is the half
    # a bare interval would get wrong.
    poser_mode.select_sheet(ctx, listed[0])
    assert len(globs) == 1
    assert poser_mode.active_sheet(ctx)["id"] == listed[0]
    settled = len(reads)
    assert settled, "a selection reads at once rather than waiting out the interval"
    for _ in range(10):
        poser_mode.active_sheet(ctx)
    assert len(reads) == settled, "and then stops"


def test_selecting_a_character_takes_its_newest_sheet(ctx, svc):
    job_id, sheets = _character(svc, sheets=3)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx)
    state = poser_mode.ensure(ctx)
    assert state.sheet_id == sheets[-1]  # newest by ``created``


def test_selecting_resets_the_clock(ctx, svc):
    """Carried across a selection it would show the new character mid-stride at
    whatever frame the old one was on, which reads as a rendering fault."""
    job_id, _sheets = _character(svc)
    _bind(ctx, job_id)
    state = poser_mode.ensure(ctx)
    state.sheet_frame, state.sheet_clock = 5, 0.4
    poser_mode.select_sheet(ctx)
    assert (state.sheet_frame, state.sheet_clock) == (0, 0.0)


# -- the clock ---------------------------------------------------------------


def _table():
    return sheet_spec.load()


def test_a_walk_advances_one_frame_per_its_own_duration(ctx):
    state = poser_mode.ensure(ctx)
    state.sheet_playing = True
    state.sheet_animation = "walk"
    walk = _table().animation("walk")
    poser_mode.sheet_advance(ctx, walk.duration_ms / 1000.0)
    assert state.sheet_frame == 1


def test_the_frame_rate_does_not_change_the_playback_speed(ctx):
    """The whole reason this is a clock. Sixty small steps and six big ones
    covering the same wall-clock time have to land on the same frame."""
    fast = poser_mode.ensure(ctx)
    fast.sheet_animation = "walk"
    for _ in range(60):
        poser_mode.sheet_advance(ctx, 1.0 / 60.0)
    quick = fast.sheet_frame

    other = _Ctx(ctx.svc)
    slow = poser_mode.ensure(other)
    slow.sheet_animation = "walk"
    for _ in range(6):
        poser_mode.sheet_advance(other, 1.0 / 6.0)
    assert slow.sheet_frame == quick


def test_a_stalled_frame_skips_cells_rather_than_falling_behind(ctx):
    state = poser_mode.ensure(ctx)
    state.sheet_playing = True
    state.sheet_animation = "walk"
    walk = _table().animation("walk")
    poser_mode.sheet_advance(ctx, walk.duration_ms / 1000.0 * 3)
    assert state.sheet_frame == 3


def test_a_cycle_loops_and_a_one_shot_holds_its_last_frame(ctx):
    """A cycle wraps and a one-shot holds, once something is playing at all.

    Playback is armed here rather than assumed: the preview opens paused, so
    what this pins is the looping rule and not the default.
    """
    state = poser_mode.ensure(ctx)
    state.sheet_playing = True
    for name in ("walk", "attack"):
        poser_mode.set_sheet_animation(ctx, name)
        animation = _table().animation(name)
        poser_mode.sheet_advance(ctx, animation.duration_ms / 1000.0 * (animation.frames + 2))
        if animation.loop:
            assert state.sheet_frame < animation.frames
        else:
            assert state.sheet_frame == animation.frames - 1


def test_a_paused_preview_does_not_move(ctx):
    state = poser_mode.ensure(ctx)
    state.sheet_playing = False
    poser_mode.sheet_advance(ctx, 10.0)
    assert state.sheet_frame == 0


def test_stepping_pauses(ctx):
    state = poser_mode.ensure(ctx)
    state.sheet_playing = True
    poser_mode.sheet_step(ctx, 1)
    assert not state.sheet_playing and state.sheet_frame == 1


def test_stepping_back_from_zero_wraps_to_the_last_frame(ctx):
    state = poser_mode.ensure(ctx)
    state.sheet_animation = "walk"
    poser_mode.sheet_step(ctx, -1)
    assert state.sheet_frame == _table().animation("walk").frames - 1


def test_changing_animation_restarts_and_changing_direction_does_not(ctx):
    """Turning a character mid-stride should show the same frame from the other
    side; changing what it is *doing* should not."""
    state = poser_mode.ensure(ctx)
    state.sheet_frame = 3
    poser_mode.set_sheet_direction(ctx, "left")
    assert state.sheet_frame == 3
    poser_mode.set_sheet_animation(ctx, "run")
    assert state.sheet_frame == 0


# -- which cell --------------------------------------------------------------


def test_the_cell_comes_from_the_frame_table(ctx):
    """Against ``charsheet``'s copy rather than a number written here: the
    agreement between the two tables has exactly one owner, and a literal in
    this test would be a third."""
    state = poser_mode.ensure(ctx)
    for cell in charsheet.frame_table():
        state.sheet_animation, state.sheet_direction, state.sheet_frame = (
            cell.animation,
            cell.direction,
            cell.frame,
        )
        assert poser_mode.cell_index(ctx) == cell.index


def test_a_frame_past_the_end_of_a_clip_is_no_cell_rather_than_a_wrong_one(ctx):
    state = poser_mode.ensure(ctx)
    state.sheet_animation, state.sheet_direction = "walk", "front"
    state.sheet_frame = 99
    assert poser_mode.cell_index(ctx) is None


# -- registration --------------------------------------------------------


def test_the_sheet_section_is_not_journal_tracked():
    """Not an oversight, and asserted so that adding one is a deliberate act:
    the sheet section is a selection over sheets a worker published, so a Save
    command would have nothing to write and a journal provider nothing to
    recover.

    Decision 2 of the P9 folding brief, replacing Troupe's own
    ``test_troupe_holds_no_document_and_says_so_by_omission``: Poser *does*
    have a journal provider (``poser_mode.JOURNAL``, kind ``"pose"``) for the
    pose editor, so the claim here is narrower and sharper -- that provider's
    own machinery never reads a ``sheet_*`` field.
    """
    import dataclasses

    from realmspinner.studio.modes.poser.mode import PoserState

    sheet_fields = {
        f.name for f in dataclasses.fields(PoserState) if f.name.startswith("sheet")
    }
    assert sheet_fields, "the sheet section must actually have fields to check"

    journal_source = inspect.getsource(poser_mode._pose_payload)
    journal_source += inspect.getsource(poser_mode._journal_slots)
    journal_source += inspect.getsource(poser_mode._journal_slot_for)
    for name in sheet_fields:
        assert name not in journal_source, f"the journal payload reads {name!r}"


def test_entering_from_home_creates_nothing(ctx, svc):
    """Unlike the four document modes: entering Plotter *was* the act of
    creating a map, silently and at whatever the default happened to be."""
    from realmspinner.studio.modes.home.ui.panes import landing

    before = len(svc.store.list())
    landing.start_poser(ctx)
    assert ctx.state.mode == "poser"
    assert len(svc.store.list()) == before


def _press(key):
    """One KEYDOWN. ``handle_key`` is presses-only by contract."""
    return SimpleNamespace(type=pygame.KEYDOWN, key=key, mod=0)


def test_every_reserved_nav_key_does_something_in_the_sheet_section():
    """**The assertion that catches the whole class.** ``NAV_KEY_MODES``
    membership withholds all nine of ``imgui_backend._NAV_KEYS`` from imgui
    while the sheet section is up, and ``main._shortcut``'s Poser arm returns
    whatever ``handle_key`` answered -- so a reserved key this section does
    not bind is taken from one consumer and given to none. Six of the nine
    were dead that way in Troupe: Up, Down, PageUp, PageDown, Home and End.

    Asserted over the reserved set rather than over a list written here, so a
    tenth key joining ``_NAV_KEYS`` fails until somebody decides what it means
    here."""
    from realmspinner.studio import imgui_backend, modes

    assert "poser" in modes.NAV_KEY_MODES
    source = inspect.getsource(poser_mode.sheet_handle_key)
    # By the constant's own name, resolved off ``pygame`` -- ``key.name`` gives
    # "page up" with a space, which no source line spells.
    names = {
        value: attr
        for attr in dir(pygame)
        if attr.startswith("K_") and isinstance(value := getattr(pygame, attr), int)
    }
    missing = sorted(
        names.get(key, str(key))
        for key in imgui_backend._NAV_KEYS
        if names.get(key, "") not in source
    )
    assert not missing, f"reserved but unbound in the sheet section: {missing}"


def test_up_and_down_walk_the_directions_and_wrap(ctx, svc):
    """Through ``set_sheet_direction``, so the frame is held -- the manual
    promises you can turn the character mid-stride and see the same moment."""
    job_id, _made = _v2_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx)
    state = poser_mode.ensure(ctx)
    names = [
        d["key"] for d in (poser_mode.preview_movement(ctx) or {})["directions"]
    ]
    state.sheet_direction, state.sheet_frame = names[0], 1

    for expected in names[1:] + names[:1]:
        assert poser_mode.sheet_handle_key(ctx, _press(pygame.K_DOWN)) is True
        assert state.sheet_direction == expected
    assert state.sheet_frame == 1, "turning must not move the frame"

    poser_mode.sheet_handle_key(ctx, _press(pygame.K_UP))
    assert state.sheet_direction == names[-1]


def test_the_page_keys_change_animation_and_restart_the_clip(ctx, svc):
    """The documented difference from a direction change, and the reason the
    two pairs are different keys."""
    job_id, _made = _v2_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx)
    state = poser_mode.ensure(ctx)
    names = [m["key"] for m in poser_mode.preview_layout(ctx)["movements"]]
    state.sheet_animation, state.sheet_frame, state.sheet_clock = names[0], 3, 0.4

    assert poser_mode.sheet_handle_key(ctx, _press(pygame.K_PAGEDOWN)) is True
    assert state.sheet_animation == names[1]
    assert (state.sheet_frame, state.sheet_clock) == (0, 0.0)


def test_home_and_end_land_on_the_ends_of_the_run_and_pause(ctx, svc):
    job_id, _made = _v2_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx)
    state = poser_mode.ensure(ctx)
    state.sheet_playing = True

    assert poser_mode.sheet_handle_key(ctx, _press(pygame.K_END)) is True
    frames = int(poser_mode.preview_movement(ctx)["frames"])
    assert state.sheet_frame == frames - 1
    assert state.sheet_playing is False

    assert poser_mode.sheet_handle_key(ctx, _press(pygame.K_HOME)) is True
    assert state.sheet_frame == 0


def test_a_key_on_a_sheet_with_no_layout_is_consumed_and_does_nothing(ctx):
    """An invalid v2 snapshot resolves to a layout with no movements. A press
    must not invent a direction the sheet does not have."""
    state = poser_mode.ensure(ctx)
    state.job_id, state.sheet_id = "", ""
    before = (state.sheet_animation, state.sheet_direction)

    for key in (pygame.K_UP, pygame.K_DOWN, pygame.K_PAGEUP, pygame.K_PAGEDOWN):
        assert poser_mode.sheet_handle_key(ctx, _press(key)) is True
    assert (state.sheet_animation, state.sheet_direction) == before


def test_a_key_release_never_acts_twice(ctx):
    """``handle_key`` used to read ``event.key`` without looking at
    ``event.type``, so every binding ran twice per press: Space toggled play
    and toggled it straight back, and a tap of Right stepped two frames."""
    state = poser_mode.ensure(ctx)
    was = state.sheet_playing
    down = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_SPACE, mod=0)
    up = SimpleNamespace(type=pygame.KEYUP, key=pygame.K_SPACE, mod=0)
    assert poser_mode.sheet_handle_key(ctx, down) is True
    assert state.sheet_playing is not was
    assert poser_mode.sheet_handle_key(ctx, up) is False
    assert state.sheet_playing is not was, "a release must not undo the press"

    state.sheet_frame = 0
    right = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0)
    assert poser_mode.sheet_handle_key(ctx, right) is True
    stepped = state.sheet_frame
    release = SimpleNamespace(type=pygame.KEYUP, key=pygame.K_RIGHT, mod=0)
    assert poser_mode.sheet_handle_key(ctx, release) is False
    assert state.sheet_frame == stepped, "a tap of Right stepped two frames"


def test_handle_key_only_delegates_to_the_sheet_transport_while_viewing_a_sheet(ctx):
    """The merge point: Poser's pose-editing keys and the sheet section's own
    both live in ``handle_key`` now, and the branch has to pick the right one.
    With no sheet on screen, a sheet-only key must not be swallowed as
    consumed by a section that is not drawing."""
    state = poser_mode.ensure(ctx)
    state.sheet_view = False
    assert poser_mode.handle_key(ctx, _press(pygame.K_SPACE)) is False

    state.sheet_view = True
    assert poser_mode.handle_key(ctx, _press(pygame.K_SPACE)) is True


# --- the view marks and the transport pane -----------------------------------


def _preview_source() -> str:
    import pathlib

    from realmspinner.studio.modes.poser.ui.panes import sheet as poser_sheet

    return pathlib.Path(poser_sheet.__file__).read_text(encoding="utf-8")


def test_the_centre_pane_takes_the_wheel_and_now_gives_it_to_something() -> None:
    """``no_scroll_with_mouse`` said the wheel belonged to the zoom control.
    No Troupe pane read the wheel, so it belonged to nothing and turning it
    over the sprite did nothing at all."""
    source = _preview_source()
    assert "io.mouse_wheel" in source
    assert "state.sheet_zoom = max(1, min(int(state.sheet_zoom + io.mouse_wheel), 32))" in source


def test_playback_speed_has_a_control_at_last() -> None:
    """``sheet_advance`` has divided the frame interval by ``state.sheet_speed``
    since Troupe was written and nothing could ever change it, so every
    preview played at exactly 1x."""
    from realmspinner.studio.modes.poser.ui.panes import sheet as poser_sheet

    assert "##poser-sheet-speed" in _preview_source()
    keys = [float(key) for key, _ in poser_sheet._SPEEDS]
    assert 1.0 in keys and min(keys) < 1.0 < max(keys)
    # A stored value off the ladder resolves to its nearest rung rather than
    # showing a blank combo.
    assert poser_sheet._speed_key(0.9) == "1.0"
    assert poser_sheet._speed_key(0.3) == "0.25"


def test_the_transport_is_a_measured_row_not_a_same_line_chain() -> None:
    """``same_line`` clips rather than wraps, so on a narrow centre pane the
    zoom field went off the edge with no way to reach it."""
    source = _preview_source()
    assert 'toolbar.toolbar("poser-sheet-transport"' in source
    # Play/back/forward are the row's reason for existing: a Play collapsed
    # into an overflow menu is not a transport.
    assert source.count("pinned=True") == 3


# --- the "render a sheet" predicate ------------------------------------------


def _plain_mesh(svc, *, done=True, files=("model.glb",)):
    job_id = svc.store.create("image", "a hooded ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        (job_dir / name).write_bytes(b"fake-glb")
    if done:
        svc.store.set_status(job_id, "done")
    return job_id


def _row(svc, job_id):
    row = dict(svc.store.get(job_id))
    row["files"] = [p.name for p in svc.job_dir(job_id).iterdir()]
    return row


def test_the_predicate_answers_from_the_row_and_asks_for_no_rig(ctx, svc):
    """An unrigged mesh is exactly what the door is for.

    And no filesystem call: a toolbar asks this every frame, which is
    ``inker_mode.can_edit_job``'s rule.
    """
    job_id = _plain_mesh(svc)
    assert poser_mode.can_render_sheet(ctx, _row(svc, job_id))

    # The *code*, not the prose: this file's style is to name the rejected
    # alternative, so a raw scan would fail on the docstring explaining why
    # neither of these is read.
    body = [
        line
        for line in inspect.getsource(poser_mode.can_render_sheet).splitlines()
        if not line.lstrip().startswith("#")
    ]
    body = chr(10).join(body).split(chr(34) * 3)[-1]
    assert "rig.glb" not in body, "an unrigged mesh is the case this exists for"
    assert "source.glb" not in body, "the reconstruction is not the asset"
    for banned in ("job_dir", "exists()", "read_rig"):
        assert banned not in body, banned


def test_the_predicate_refuses_what_has_no_mesh_to_render(ctx, svc):
    unfinished = _plain_mesh(svc, done=False)
    assert not poser_mode.can_render_sheet(ctx, _row(svc, unfinished))

    no_mesh = _plain_mesh(svc, files=("input.png",))
    assert not poser_mode.can_render_sheet(ctx, _row(svc, no_mesh))

    reference = svc.store.create("image", "a drawing", {}, stage="reference")
    svc.store.set_status(reference, "done")
    assert not poser_mode.can_render_sheet(ctx, dict(svc.store.get(reference)))

    trashed = _row(svc, _plain_mesh(svc))
    trashed["deleted_at"] = "2026-08-23"
    assert not poser_mode.can_render_sheet(ctx, trashed)


def test_rendering_submits_under_its_own_key_and_does_not_switch_mode(ctx, svc):
    """``start_character`` switches to Create because the gate is there and
    the user has something to do. Here there is nothing to do, and pulling
    somebody out of the library to watch a spinner is the opposite of the
    affordance."""
    job_id = _plain_mesh(svc)
    submitted: list[str] = []
    ctx.submit = lambda key, run, *a: (submitted.append(key), True)[1]

    assert poser_mode.render_character_sheet(ctx, _row(svc, job_id))
    assert submitted == [f"troupe-send:{job_id}"]
    assert ctx.state.mode == "poser"


def test_a_sheet_can_be_named_from_the_form(ctx, svc):
    """**The whole path existed except the field.** The door validates
    ``name`` against ``store.MAX_SHEET_NAME``, the worker writes it into the
    sidecar and the chooser reads it back."""
    source = inspect.getsource(poser_mode.build_sheet)
    assert "name=" in source, "build_sheet must carry the form's name to the door"

    from realmspinner.studio.modes.poser.ui.panes import sheet as poser_sheet

    pane = inspect.getsource(poser_sheet)
    assert '"name"' in pane, "the form needs a name field for build_sheet to carry"
    assert "MAX_SHEET_NAME" in pane, "the field must cap at what the door accepts"


def test_the_cell_caps_are_read_from_charsheet_not_restated(ctx, svc):
    """troupe-05 (2026-09-07 audit): the pane wrote ``512``/``256`` as bare
    numbers instead of reading ``charsheet.MAX_CELLS``/``WARN_CELLS`` -- the
    door's own ladder, which the pane already imports ``charsheet`` for. A
    restated number goes stale the day the door's moves, because nothing but
    the number itself would then disagree.
    """
    from realmspinner.studio.modes.poser.ui.panes import sheet as poser_sheet

    source = inspect.getsource(poser_sheet)
    assert "<= 512" not in source
    assert "> 256" not in source
    assert "charsheet.MAX_CELLS" in source
    assert "charsheet.WARN_CELLS" in source


def test_the_render_door_still_carries_every_parameter_it_validates(ctx, svc):
    """``elevation`` and ``lighting`` have no control yet. The read stays, so
    adding one is a pane change -- which is the state ``name`` was in until its
    field landed."""
    source = inspect.getsource(poser_mode.render_character_sheet)
    for field in ("elevation", "lighting", "name"):
        assert f"{field}=" in source, field


def test_the_camera_preset_reaches_build_sheet_as_elevation(ctx, svc):
    """**The form holds a name and every door takes a number.**

    A preset is a name for an elevation, and the translation happens once, on
    the way out of the section -- so a combo reading "Side" cannot reach a
    door that renders at 30 degrees because nobody converted it. An elevation
    set explicitly still wins, because an angle off the ladder has to stay
    expressible or the ladder becomes the only thing anyone can render.
    """
    captured: dict = {}
    ctx.submit = lambda key, fn, *a, **kw: (captured.update(kw), True)[1]
    ctx.state.clear_field_errors = lambda: None
    angles = {key: angle for key, _label, angle in charsheet.CAMERA_PRESETS}

    for key, angle in angles.items():
        captured.clear()
        assert poser_mode.build_sheet(ctx, "a-job", {"camera": key})
        assert captured["elevation"] == angle, key

    captured.clear()
    poser_mode.build_sheet(ctx, "a-job", {"camera": "side", "elevation": 42.0})
    assert captured["elevation"] == 42.0, "a custom angle must stay expressible"

    # A form from before the control existed names no camera, and the door's
    # own default is what answers -- not a number invented here.
    captured.clear()
    poser_mode.build_sheet(ctx, "a-job", {})
    assert captured["elevation"] is None


def test_the_preview_opens_paused(ctx):
    """Overturned on request 2026-08-23, and pinned so it cannot drift back.

    A clip already moving when you arrive is one you have to stop before you
    can look at any frame in it, and looking at a frame -- a hand, a
    silhouette, which way the feet point -- is the first thing anyone does with
    a new sheet.
    """
    state = poser_mode.ensure(ctx)
    assert state.sheet_playing is False
    poser_mode.sheet_advance(ctx, 10.0)
    assert state.sheet_frame == 0, "a paused preview advanced on its own"


# -- the scores -------------------------------------------------------------
#
# ``poser.engine.qa`` is scored in the task runner and read back by the
# preview. What these pin: one submit per selection, adoption only for the
# sheet still on screen, no toast either way, and -- the one that matters --
# the pane never scores in the frame loop.


class _SubmitCtx(_Ctx):
    def __init__(self, svc):
        super().__init__(svc)
        self.submitted: list[tuple] = []

    def submit(self, key, fn, *args, **kwargs):
        self.submitted.append((key, fn, args, kwargs))

    def busy(self, key):
        return any(entry[0] == key for entry in self.submitted)


def _png_character(svc, size=16):
    """A character whose sheet PNG is a real atlas rather than four bytes."""
    import numpy as np
    from PIL import Image

    job_id, made = _v2_character(svc)
    atlas = np.zeros((size * 4, size * charsheet.COLUMNS, 4), dtype=np.uint8)
    atlas[..., :3] = 90
    atlas[..., 3] = 255
    path = store.sheet_png_path(svc.job_dir(job_id), made[0])
    Image.fromarray(atlas, "RGBA").save(path)
    record_path = store.sheet_path(svc.job_dir(job_id), made[0])
    record = json.loads(record_path.read_text("utf-8"))
    record["frame_size"] = size
    record_path.write_text(json.dumps(record), "utf-8")
    return job_id, made


def test_scores_are_submitted_once_per_selection_and_not_in_the_frame_loop(svc):
    ctx = _SubmitCtx(svc)
    job_id, made = _png_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])
    assert poser_mode.scores(ctx) is None
    assert len(ctx.submitted) == 1
    key, fn, args, _kwargs = ctx.submitted[0]
    assert key == poser_mode.scores_key(job_id, made[0])
    # Asking again while it runs submits nothing more.
    assert poser_mode.scores(ctx) is None
    assert len(ctx.submitted) == 1
    # The task itself is honest: run it here and it scores the real atlas.
    result = fn(*args)
    assert result.cells and result.worst is None


def test_a_landed_score_is_adopted_only_for_the_sheet_still_on_screen(svc):
    ctx = _SubmitCtx(svc)
    job_id, made = _png_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])
    poser_mode.scores(ctx)
    key, fn, args, _kwargs = ctx.submitted[0]
    result = fn(*args)
    stale = SimpleNamespace(key=poser_mode.scores_key(job_id, "other"), result=result)
    poser_mode.on_task_done(ctx, stale)
    assert poser_mode.scores(ctx) is None
    poser_mode.on_task_done(ctx, SimpleNamespace(key=key, result=result))
    assert poser_mode.scores(ctx) is result
    assert ctx.toasts == []


def test_a_failed_score_is_latched_and_never_toasted(svc):
    ctx = _SubmitCtx(svc)
    job_id, made = _png_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])
    poser_mode.scores(ctx)
    key = ctx.submitted[0][0]
    ctx.submitted.clear()
    poser_mode.on_task_failed(ctx, SimpleNamespace(key=key, error="boom"))
    assert poser_mode.scores(ctx) is None
    assert poser_mode.scores_failed(ctx)
    assert ctx.submitted == []
    assert ctx.toasts == []


def test_selecting_another_sheet_drops_the_scores(svc):
    ctx = _SubmitCtx(svc)
    job_id, made = _png_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])
    ctx.state.preview["troupe_scores"] = "stale"
    ctx.state.preview["troupe_scores:key"] = (job_id, made[0])
    poser_mode.select_sheet(ctx, "")
    assert "troupe_scores" not in ctx.state.preview


def test_goto_points_the_preview_at_a_cell_and_stops(ctx, svc):
    job_id, made = _v2_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])
    state = poser_mode.ensure(ctx)
    state.sheet_animation = "walk"
    state.sheet_playing = True
    poser_mode.sheet_goto(ctx, "back", 99)
    assert (state.sheet_direction, state.sheet_frame, state.sheet_playing) == (
        "back", 5, False,
    )


def test_a_sheet_that_does_not_say_its_cell_size_is_latched_rather_than_submitted(svc):
    ctx = _SubmitCtx(svc)
    job_id, made = _v2_character(svc)
    record_path = store.sheet_path(svc.job_dir(job_id), made[0])
    record = json.loads(record_path.read_text("utf-8"))
    record["frame_size"] = 0
    record_path.write_text(json.dumps(record), "utf-8")
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])
    assert poser_mode.scores(ctx) is None
    assert ctx.submitted == []
    assert poser_mode.scores_failed(ctx)


def test_the_pane_never_scores_in_the_frame_loop() -> None:
    source = _preview_source()
    assert "score_sheet" not in source
    assert "poser_mode.scores(" in source


def test_atlas_texture_and_scores_refuse_a_sheet_path_that_is_not_a_file(svc):
    """is_file(), not exists(): the 2026-09-08 audit's troupe-04 found
    ``scores`` and ``atlas_texture`` gating a task submission on
    ``path.exists()``, the same class of bug ``sheet.pack()`` was fixed for on
    2026-09-07 (see the comment there). A directory sitting where the sheet
    PNG should be still satisfies ``exists()``, so the task would be
    submitted and fail inside ``Image.open`` with no mention of which sheet,
    instead of being refused at the door the way ``pack()`` already is.
    """
    ctx = _SubmitCtx(svc)
    job_id, made = _v2_character(svc)
    ctx.viewer = SimpleNamespace()  # anything not None: atlas_texture only gates on identity
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])
    assert poser_mode.ensure(ctx).sheet_id == made[0], (
        "selection must actually take while the PNG is still a real file, or "
        "everything below it passes for the wrong reason"
    )

    png_path = store.sheet_png_path(svc.job_dir(job_id), made[0])
    png_path.unlink()
    png_path.mkdir()

    assert poser_mode.scores(ctx) is None
    assert poser_mode.scores_failed(ctx)
    assert poser_mode.atlas_texture(ctx) is None
    assert ctx.submitted == [], "a directory at the sheet path must not reach a task submit"


def test_cell_geometry_reads_a_non_square_plan_in_the_right_order():
    assert poser_mode.cell_geometry({"columns": 8, "frame_size": 32}) == (8, 32, 32)
    assert poser_mode.cell_geometry(
        {"columns": 4, "frame_size": 0, "frame_w": 24, "frame_h": 40}
    ) == (4, 24, 40)
    assert poser_mode.cell_geometry({"columns": 8}) is None
    assert poser_mode.cell_geometry(None) is None


class _TakenCtx(_SubmitCtx):
    """``_SubmitCtx`` whose ``submit`` answers like the real runner's: True for
    a key it took. The scoring tests above never read the answer; the export
    door does, because a refused press has to be distinguishable from a taken
    one."""

    def submit(self, key, fn, *args, **kwargs):
        super().submit(key, fn, *args, **kwargs)
        return True


def test_the_pivot_marker_reads_the_cell_and_not_a_constant():
    """**The claim: ``pivot_of`` looks the cell up.** The tempting shortcut is
    ``(w / 2, h)`` -- which is what ``sheet.sidecar`` itself writes when the
    renderer measured nothing -- and it would agree with the record on most
    cells and quietly disagree on the ones that matter: a character leaning
    into an attack does not stand on the middle of its own cell."""
    record = {
        "cells": [
            {"index": 0, "w": 32, "h": 32, "pivot_x": 16.0, "pivot_y": 32.0},
            {"index": 1, "w": 32, "h": 32, "pivot_x": 9.5, "pivot_y": 28.0},
        ]
    }
    assert poser_mode.pivot_of(record, 1) == (9.5, 28.0), "the cell, not the first"
    assert poser_mode.pivot_of(record, 0) == (16.0, 32.0)
    # A cell the sheet does not have is not the last cell's answer either.
    assert poser_mode.pivot_of(record, 7) is None


def test_a_cell_with_no_pivot_answers_none_rather_than_the_centre():
    """The regression the preview's marker depends on: a marker at a guessed
    origin is a lie about where the engine will place the sprite, and it looks
    exactly like a measured one."""
    assert poser_mode.pivot_of({"cells": [{"index": 0, "w": 32, "h": 32}]}, 0) is None
    assert poser_mode.pivot_of({"cells": [{"index": 0, "pivot_x": 4.0}]}, 0) is None
    assert poser_mode.pivot_of(None, 0) is None
    assert poser_mode.pivot_of({"cells": []}, None) is None


def test_c_and_p_toggle_the_view_marks_on_the_press_only(ctx, svc):
    """``handle_key`` acted on ``event.key`` without reading ``event.type``
    once already, and every binding ran twice per press. A toggle that runs
    twice is a toggle that does nothing."""
    state = poser_mode.ensure(ctx)
    assert (state.sheet_checker, state.sheet_show_pivot) == (False, True)

    assert poser_mode.sheet_handle_key(ctx, _press(pygame.K_c)) is True
    assert state.sheet_checker is True
    assert poser_mode.sheet_handle_key(
        ctx, SimpleNamespace(type=pygame.KEYUP, key=pygame.K_c, mod=0)
    ) is False
    assert state.sheet_checker is True, "a release must not undo the press"

    assert poser_mode.sheet_handle_key(ctx, _press(pygame.K_p)) is True
    assert state.sheet_show_pivot is False


def test_typing_a_c_into_a_field_is_typing_and_not_a_view_toggle(ctx, monkeypatch):
    """The rule every plain-letter shortcut in the app lives by
    (``main._passes_text_field``): a focused text field takes the plain keys.
    Naming a character "packwright" must not toggle the checkerboard four
    times on the way past -- and the arrows above are exempt because the nav
    reservation already withholds them while a field has the keyboard."""
    monkeypatch.setattr(poser_mode, "_typing", lambda: True)
    state = poser_mode.ensure(ctx)
    before = (state.sheet_checker, state.sheet_show_pivot)

    assert poser_mode.sheet_handle_key(ctx, _press(pygame.K_c)) is False
    assert poser_mode.sheet_handle_key(ctx, _press(pygame.K_p)) is False
    assert (state.sheet_checker, state.sheet_show_pivot) == before


def test_a_sheet_that_needs_repair_says_what_is_wrong_and_still_plays(ctx, svc):
    """**Structural validation is not the QA heatmap, and neither one gates.**
    ``qa.py`` ranks the drawing; ``sheetcheck`` says a cell is clipped, empty
    or was never rendered. This pins both halves of the second: the
    diagnostics are the ones ``sheetcheck.describe`` writes, and the preview
    plays the sheet exactly as it would a clean one -- a verdict the user may
    disagree with must not take their sheet away."""
    job_id, made = _v2_character(svc)
    path = store.sheet_path(svc.job_dir(job_id), made[0])
    record = json.loads(path.read_text("utf-8"))
    record["validation"] = {
        "version": 1,
        "ok": False,
        "clipped": [3, 4],
        "blank": [],
        "missing": [],
        "metadata": [],
        "reframed": False,
    }
    path.write_text(json.dumps(record), "utf-8")
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])
    live = poser_mode.active_sheet(ctx)

    assert poser_mode.needs_repair(live) is True
    notes = poser_mode.repair_notes(live)
    assert notes and "clipped at the frame edge" in notes[0]

    state = poser_mode.ensure(ctx)
    state.sheet_playing = True
    poser_mode.sheet_advance(ctx, 10.0)
    assert poser_mode.cell_index(ctx) is not None, "a flagged sheet still plays"
    assert state.sheet_playing is True, "nothing stops the clock over a verdict"


def test_an_unchecked_sheet_is_not_accused_of_anything(ctx, svc):
    """A sheet rendered before ``validation`` existed carries no block, and
    "we did not look" must not read as "we looked and it is broken"."""
    job_id, made = _v2_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])
    record = poser_mode.active_sheet(ctx)
    assert "validation" not in record
    assert poser_mode.needs_repair(record) is False
    assert poser_mode.repair_notes(record) == []
    assert poser_mode.needs_repair({"validation": {"ok": True}}) is False


def test_exporting_a_package_writes_the_png_and_the_sidecar_together(svc, tmp_path):
    """**The pair is the deliverable.** A folder holding the atlas without the
    JSON holds an asset nothing can interpret -- the sidecar is what says which
    cell is ``walk`` facing south-east. Submitted under its own key, so a
    second press while one is in flight is refused rather than racing it."""
    ctx = _TakenCtx(svc)
    ctx.export_dir = str(tmp_path / "project")
    job_id, made = _v2_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])

    assert poser_mode.export_package(ctx) is True
    assert len(ctx.submitted) == 1
    key, run, _args, _kwargs = ctx.submitted[0]
    assert key == f"troupe-export:{job_id}:{made[0]}"
    # In flight: the second press is refused rather than queued behind it.
    assert poser_mode.export_package(ctx) is False
    assert len(ctx.submitted) == 1

    written = run()
    assert Path(written["png"]).exists() and Path(written["json"]).exists()
    assert Path(written["png"]).parent == Path(written["json"]).parent

    poser_mode.on_task_done(ctx, SimpleNamespace(key=key, result=written))
    said = ctx.toasts[-1][0]
    assert Path(written["png"]).name in said and Path(written["json"]).name in said


def test_a_cancelled_export_picker_is_not_reported_as_an_export(svc):
    """``dialogs.select_folder`` answers None for a cancel and nothing else,
    and a toast naming two files nobody wrote would be the app claiming a
    write it did not make."""
    ctx = _SubmitCtx(svc)
    job_id, made = _v2_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])
    poser_mode.export_package(ctx)
    key = ctx.submitted[0][0]

    poser_mode.on_task_done(ctx, SimpleNamespace(key=key, result=None))
    assert ctx.toasts == []


def test_export_frames_asks_for_a_folder_on_the_task_thread(svc, monkeypatch):
    """``export_package``'s exact arrangement, one door over: the picker is
    asked inside the submitted ``run``, never before -- calling ``run()`` here
    rather than pressing a button is what proves "on the task thread", since a
    picker invoked while ``export_frames`` itself runs would have fired before
    this line."""
    from realmspinner.service import characters as svc_characters
    from realmspinner.studio import dialogs

    def _boom(*_a, **_k):
        raise AssertionError("the picker must not run before the task thread")

    monkeypatch.setattr(dialogs, "select_folder", _boom)
    ctx = _TakenCtx(svc)
    job_id, made = _v2_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])

    assert poser_mode.export_frames(ctx) is True
    assert len(ctx.submitted) == 1
    key, run, _args, _kwargs = ctx.submitted[0]
    assert key == f"troupe-frames:{job_id}:{made[0]}"
    # In flight: the second press is refused rather than queued behind it.
    assert poser_mode.export_frames(ctx) is False
    assert len(ctx.submitted) == 1

    monkeypatch.setattr(dialogs, "select_folder", lambda *_a, **_k: str(svc.job_dir(job_id)))
    recorded: list = []
    monkeypatch.setattr(
        svc_characters,
        "export_frames",
        lambda svc_, job_id_, sheet_id_, dest_dir=None: recorded.append(dest_dir)
        or Path(dest_dir) / "Stem",
    )
    folder = run()
    assert recorded == [str(svc.job_dir(job_id))]
    assert folder == Path(svc.job_dir(job_id)) / "Stem"

    poser_mode.on_task_done(ctx, SimpleNamespace(key=key, result=folder))
    said = ctx.toasts[-1][0]
    assert str(folder) in said


def test_export_frames_uses_the_configured_folder_without_asking(svc, monkeypatch, tmp_path):
    """A configured export folder is used outright -- the picker must not run
    at all when ``ctx.export_dir`` is set. ``service.characters.export_frames``
    is mocked here: what this test pins is the mode's own choice of
    destination, not the service's PNG cropping, which
    ``tests/service/test_character_exports.py`` already owns."""
    from realmspinner.service import characters as svc_characters
    from realmspinner.studio import dialogs

    def _boom(*_a, **_k):
        raise AssertionError("the picker must not run when a folder is configured")

    monkeypatch.setattr(dialogs, "select_folder", _boom)
    recorded: list = []
    monkeypatch.setattr(
        svc_characters,
        "export_frames",
        lambda svc_, job_id_, sheet_id_, dest_dir=None: recorded.append(dest_dir)
        or Path(dest_dir) / "Stem",
    )
    ctx = _TakenCtx(svc)
    ctx.export_dir = str(tmp_path / "project")
    job_id, made = _v2_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])

    assert poser_mode.export_frames(ctx) is True
    _key, run, _args, _kwargs = ctx.submitted[0]

    folder = run()
    assert recorded == [str(tmp_path / "project")]
    assert folder == tmp_path / "project" / "Stem"


def test_a_cancelled_frame_export_writes_nothing(svc, monkeypatch):
    """``None`` from the picker means the user cancelled. Run all the way
    through ``run()`` -- unlike the package export's cancel test, which stops
    at simulating the result -- to pin that a cancelled pick never reaches
    ``service.characters.export_frames`` at all, and that a toast naming a
    folder nobody wrote would be the app claiming a write it did not make."""
    from realmspinner.service import characters as svc_characters
    from realmspinner.studio import dialogs

    monkeypatch.setattr(dialogs, "select_folder", lambda *_a, **_k: None)

    def _boom(*_a, **_k):
        raise AssertionError("a cancelled pick must never reach the writer")

    monkeypatch.setattr(svc_characters, "export_frames", _boom)
    ctx = _TakenCtx(svc)
    job_id, made = _v2_character(svc)
    _bind(ctx, job_id)
    poser_mode.select_sheet(ctx, made[0])

    assert poser_mode.export_frames(ctx) is True
    key, run, _args, _kwargs = ctx.submitted[0]
    assert run() is None

    poser_mode.on_task_done(ctx, SimpleNamespace(key=key, result=None))
    assert ctx.toasts == []


def test_the_bridge_offers_export_frames_beside_the_package_export():
    """The pane draws a fourth way out next to "Export package...", wired to
    ``poser_mode.export_frames`` and gated the same way -- ready and not
    busy on the frames key, one call sharing the busy check with the button
    it sits beside."""
    from realmspinner.studio.modes.poser.ui.panes import sheet as poser_sheet

    source = inspect.getsource(poser_sheet._bridge)
    package_at = source.index('"Export package..."')
    frames_at = source.index('"Export frames..."')
    assert package_at < frames_at, "frames export belongs beside, after, the package export"
    assert "poser_mode.export_frames(ctx)" in source
    assert "poser_mode.frames_key(state.job_id, state.sheet_id)" in source


def test_varying_a_character_loads_its_recipe_as_the_users_own_choices(ctx, svc, monkeypatch):
    """**Every field is marked as an override, and that is the whole claim.**

    Create's character form follows the prompt for anything the user has not
    touched (``character_engine.sync_from_prompt``), and the prompt in the
    box is whatever was last typed there -- so a recipe loaded without the
    override marks would have its species, theme and camera silently rewritten
    on the next keystroke, by a brief that is not about this character. The
    press of the button *is* the touch."""
    from realmspinner.studio.modes.create.engine import character as character_engine
    from realmspinner.studio.modes.create.ui import stages as create_stages
    from realmspinner.studio.state import default_form_2d

    went: list[str] = []
    monkeypatch.setattr(create_stages, "go", lambda c, stage, **kw: went.append(stage))
    ctx.state.form_2d = default_form_2d()
    ctx.state.form_2d["prompt"] = "a completely different brief"
    record = {
        "character": {
            "family": "wyvern",
            "family_version": 3,
            "recipe": {
                "family": "wyvern",
                "family_version": 3,
                "theme": "ember",
                "camera": "isometric",
                "animations": {"walk": 8, "idle": 4},
                "logical_size": 48,
                "colors": 16,
                "appearance": {"horn": 0.75},
                "seed": 4242,
                "name": "Ash",
            },
        }
    }

    assert poser_mode.vary_in_create(ctx, record) is True
    form = ctx.state.form_2d
    assert went == ["reference"]
    assert form["asset_type"] == "character"
    assert form["character_family"] == "wyvern"
    assert form["character_camera"] == "isometric"
    assert form["character_pixel"] == "48" and form["character_colors"] == "16"
    assert '"horn"' in form["character_body"]
    assert form["seed"] == 4242

    marked = set(character_engine.overrides_of(form))
    assert set(character_engine.RECIPE_FIELDS) <= marked, "no field follows the prompt"

    # And the proof of what the marks are for: re-resolving the brief in the
    # box leaves every one of them exactly where this put it.
    before = {key: form[key] for key in character_engine.RECIPE_FIELDS}
    character_engine.sync_from_prompt(form)
    assert {key: form[key] for key in character_engine.RECIPE_FIELDS} == before


def test_a_sheet_with_no_recipe_offers_no_variation(ctx, svc):
    """A sheet built from a supplied mesh, or from the "Start a new character"
    form, was never described by a ``Recipe``. Refused rather than switching
    into a Create form that describes somebody else."""
    assert poser_mode.vary_in_create(ctx, {}) is False
    assert poser_mode.vary_in_create(ctx, {"character": {"family": "wolf"}}) is False
    assert poser_mode.recipe_of(None) == {}
