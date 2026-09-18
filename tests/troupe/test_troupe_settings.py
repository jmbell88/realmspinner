"""Troupe's New Character pane: the open clip vocabulary, Style and Frame rate.

Four claims new since the 2026-09-12 open vocabulary
(``dev/measurements/2026-09-12-troupe-open-clip-vocabulary.md`` and the
service-layer landing that followed it):

* every clip the rig's skeleton defines is a row on this form, not just the
  closed legacy five, and a provisional one says so;
* Style has two real states, and HD strips exactly the four fields the door
  refuses a truthy one of (``service.troupe._check_options``);
* a Frame rate choice rides the request's ``layout`` block, moving it to
  version 3 -- "Authored" sends neither, byte-identical to before either
  control existed;
* 256 is on the size ladder the pane reads, not a restated one that stops
  short of it.
"""

from __future__ import annotations

import inspect

import pytest
from _ui_context import imgui_context

from warlock.service import troupe as svc_troupe
from warlock.studio import probe
from warlock.studio.modes.troupe import mode as troupe_mode
from warlock.studio.modes.troupe.ui.panes import settings as troupe_settings
from warlock.studio.state import AppState


class _Ctx:
    """The narrow slice of the app context the pane's draw touches. No GL --
    ``test_troupe_mode``'s own reason: none of this needs one."""

    def __init__(self, svc):
        self.svc = svc
        self.state = AppState()

    def busy(self, key: str) -> bool:
        return False

    def toast(self, *a, **k) -> None:
        pass


@pytest.fixture
def ctx(svc):
    return _Ctx(svc)


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` for why it is not a
    fixture there."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _draw(ui, ctx):
    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    try:
        troupe_settings.draw(ctx)
    finally:
        ui.end()
        ui.end_frame()
    return probe.census()


def test_the_settings_pane_offers_every_clip_the_rigs_skeleton_defines(ui, ctx, svc):
    """Every row of ``clip_vocabulary["humanoid"]``, not just the legacy five
    ``options["animations"]`` still carries -- drawn as a real ``movement_<name>``
    switch, in a real imgui frame.
    """
    vocabulary = svc_troupe.troupe_options(svc)["clip_vocabulary"]["humanoid"]
    names = {str(row["name"]) for row in vocabulary}
    assert len(names) > 5, "the shipped humanoid library carries more than five clips"

    seen = _draw(ui, ctx)
    switches = {c.name for c in seen if c.kind == "switch" and c.name.startswith("movement_")}
    offered = {name.split("movement_", 1)[1] for name in switches}
    assert offered == names


#: Two templates whose vocabularies genuinely differ, so a test can tell which
#: one the rows came from by name alone -- the shipped humanoid and quadruped
#: libraries hold the same ten names today (only frames, timing and
#: provisional flags differ), which would not catch a form reading the wrong
#: one.
_FAKE_LAYOUT_OPTIONS = {
    "defaults": {"template": "humanoid"},
    "clip_vocabulary": {
        "humanoid": [{"name": "walk", "frames": 8, "default": True}],
        "quadruped": [{"name": "gallop", "frames": 12, "default": True}],
    },
}


def _offered_movements(seen) -> set[str]:
    switches = {c.name for c in seen if c.kind == "switch" and c.name.startswith("movement_")}
    return {name.split("movement_", 1)[1] for name in switches}


def test_the_movement_rows_follow_the_rigs_own_skeleton(ui, ctx, svc, monkeypatch):
    """The shared layout table reads the *bound* character's own skeleton, not
    the door's default -- the defect: a quadruped, bird or blob bound here
    used to get the default vocabulary's rows regardless of what its own clip
    library actually holds."""
    monkeypatch.setattr(troupe_mode, "options", lambda _ctx: dict(_FAKE_LAYOUT_OPTIONS))
    mesh_id = svc.store.create("image", "a wolf", {}, stage="model")
    svc.store.create(
        "rig", "a wolf", {"source_job": mesh_id, "template": "quadruped"}, status="done"
    )
    troupe_mode.ensure(ctx).job_id = mesh_id

    assert _offered_movements(_draw(ui, ctx)) == {"gallop"}


def test_rows_fall_back_to_the_default_skeleton_only_without_a_rig(ui, ctx, svc, monkeypatch):
    """Nothing bound -- the ordinary "New character" form -- still gets the
    door's default vocabulary."""
    monkeypatch.setattr(troupe_mode, "options", lambda _ctx: dict(_FAKE_LAYOUT_OPTIONS))

    assert _offered_movements(_draw(ui, ctx)) == {"walk"}


def test_a_provisional_clip_row_says_so():
    """The short muted note and its tooltip, sourced from the clip's own
    ``provisional`` flag rather than a guess -- a source claim, ``test_camera
    _presets``'s own reason: the failure is a row that looks finished and is
    not, which no widget geometry can tell apart from one that really is."""
    source = inspect.getsource(troupe_settings._layout)
    assert 'clip.get("provisional")' in source
    assert "Placeholder keyframes; an animator's pass is still owed" in source
    assert '"Provisional"' in source


def test_256_is_offered(svc):
    """The pane reads the door's own ladder and never restates a shorter one.

    Task G (2026-09-12, master) moved the sprite-size control out of ``_size``
    and into its own ``_logical_size`` -- the ladder plus a "Custom..." box --
    so this reads the source of the function that now actually draws it."""
    options = svc_troupe.troupe_options(svc)
    assert 256 in options["logical_sizes"]
    source = inspect.getsource(troupe_settings._logical_size)
    assert '"logical_sizes"' in source
    # ``(8, 256)`` does appear, as the fallback for ``logical_size_range`` --
    # task G's custom-size floor/ceiling, an unrelated number from the ladder
    # this test guards. What must never appear is the *ladder* restated as a
    # literal tuple, which is the shape a copy-pasted ``SIZES`` would take.
    assert "16, 24, 32" not in source, "the pane must read the ladder, not name a rung"


def _minimal_form(ctx) -> dict:
    """A form that submits cheaply: one movement, one frame, one direction."""
    form = troupe_mode.form(ctx)
    form["prompt"] = "a wizard"
    form["layout"] = {
        "version": 2,
        "columns": 8,
        "movements": [{"key": "idle", "enabled": True, "frames": 1, "directions": 1}],
    }
    return form


def _submitted_troupe_block(ctx, form: dict) -> dict:
    captured: dict = {}
    ctx.submit = lambda key, fn, *a, **kw: (captured.update(kw), True)[1]
    assert troupe_mode.start_character(ctx, form)
    return captured["troupe"]


def test_hd_style_sends_no_palette_options(ctx, svc):
    """HD sends ``pixel_art: False`` and none of the four fields a pixel-art
    render has -- even when the form still holds yesterday's values for them,
    from before the switch was flipped."""
    form = _minimal_form(ctx)
    form["style"] = troupe_mode.STYLE_HD
    form["palette"] = "nes"
    form["dither"] = True
    form["outline"] = "outer"
    form["colors"] = 8

    troupe_block = _submitted_troupe_block(ctx, form)
    assert troupe_block["pixel_art"] is False
    for field in ("colors", "palette", "dither", "outline"):
        assert field not in troupe_block, field


def test_pixel_art_style_sends_todays_request_unchanged(ctx, svc):
    """No ``pixel_art`` key at all, so a form that never touches Style mints
    the byte-identical row it always did."""
    form = _minimal_form(ctx)
    form["style"] = troupe_mode.STYLE_PIXEL_ART
    form["colors"] = 64
    form["outline"] = "outer"
    form["reduce_mode"] = "box"
    form["dither"] = False
    form["palette"] = ""

    troupe_block = _submitted_troupe_block(ctx, form)
    assert "pixel_art" not in troupe_block
    assert troupe_block["colors"] == 64
    assert troupe_block["outline"] == "outer"
    assert troupe_block["reduce_mode"] == "box"
    assert troupe_block["dither"] is False
    assert troupe_block["palette"] == ""


def test_a_frame_rate_choice_rides_the_layout(ctx, svc):
    """A chosen rate reaches the request on the layout block, at version 3."""
    form = _minimal_form(ctx)
    form["fps"] = 12

    troupe_block = _submitted_troupe_block(ctx, form)
    layout = troupe_block["layout"]
    assert layout["fps"] == 12
    assert layout["version"] == 3


def test_authored_frame_rate_sends_no_fps(ctx, svc):
    """"Authored" is the default, and sends neither key -- the request a form
    built before the control existed still mints."""
    form = _minimal_form(ctx)
    form["fps"] = None

    troupe_block = _submitted_troupe_block(ctx, form)
    layout = troupe_block["layout"]
    assert "fps" not in layout
    assert layout["version"] == 2
