"""Step 1 of the Create redesign: one vocabulary across the Reference and Mesh
stages, Ctrl+Enter meaning "this stage's Generate", and the Rig section that
stays on screen (disabled, with the rail's own reason) when Blender is missing.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pygame
import pytest

from realmspinner.studio.modes.create.engine import mesh as create_mesh
from realmspinner.studio.modes.create.ui import stages as create_stages
from realmspinner.studio.modes.create.ui.panes import settings_2d, settings_3d
from realmspinner.studio.shell.events import EventsMixin
from realmspinner.studio.state import DEFAULT_FORM_3D

# --- labels -----------------------------------------------------------------


def test_the_mesh_stage_labels_its_seed_seed():
    src = inspect.getsource(settings_3d._draw_form)
    assert '"Seed", int(form["mesh_seed"]' in src
    assert "Mesh seed" not in src


def test_both_stages_share_one_lock_seed_hint():
    assert create_stages.SEED_LOCK_HINT == "Unlocked, every Generate draws a fresh seed."
    assert "helper=create_stages.SEED_LOCK_HINT" in inspect.getsource(settings_2d._seed_row)
    assert "helper=create_stages.SEED_LOCK_HINT" in inspect.getsource(settings_3d._draw_form)


def test_the_mesh_stage_image_picker_reads_choose_an_image():
    src = inspect.getsource(settings_3d._source)
    assert '"Choose an image..."' in src
    assert "Open an image" not in src


def test_focus_pane_is_defined_once_in_settings_2d():
    src = inspect.getsource(inspect.getmodule(settings_2d))
    assert src.count('FOCUS_PANE = "2d"') == 1


# --- the 3D form's vocabulary ----------------------------------------------


def test_the_3d_form_calls_its_attempt_count_count():
    assert DEFAULT_FORM_3D["count"] == 1
    assert "candidates" not in DEFAULT_FORM_3D
    assert create_mesh.candidate_count({"count": 3}) == 3
    # The old key is no longer read.
    assert create_mesh.candidate_count({"candidates": 3}) == 1


def test_an_unset_mesh_seed_is_none_and_never_sent():
    assert DEFAULT_FORM_3D["mesh_seed"] is None
    form = dict(DEFAULT_FORM_3D)
    assert "mesh_seed" not in create_mesh.promote_kwargs(form)
    assert "mesh_seed" not in create_mesh.upload_kwargs(form)


def test_a_mesh_seed_of_zero_is_a_real_seed_and_is_sent():
    form = {**DEFAULT_FORM_3D, "mesh_seed": 0}
    assert create_mesh.promote_kwargs(form)["mesh_seed"] == 0
    assert create_mesh.upload_kwargs(form)["mesh_seed"] == 0


# --- the Rig section without Blender ---------------------------------------


def test_rig_section_shows_disabled_when_blender_is_missing(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        settings_3d.widgets, "section", lambda name: seen.setdefault("section", name)
    )

    def checkbox(label, value, **kw):
        seen["checkbox"] = (label, value, kw)
        return False, value

    monkeypatch.setattr(settings_3d.controls, "checkbox", checkbox)
    ctx = SimpleNamespace(rigging_available=False)
    settings_3d._rig(ctx, {"rig": True, "rig_template": ""})
    assert seen["section"] == "Rig"
    label, value, kw = seen["checkbox"]
    assert kw["enabled"] is False
    assert kw["reason"] == create_stages.available("rig", None, ctx)
    assert kw["reason"] == "Rigging needs Blender, which is not installed."
    assert value is False


def test_rig_section_is_enabled_when_blender_is_present(monkeypatch):
    seen = {}
    monkeypatch.setattr(settings_3d.widgets, "section", lambda name: None)

    def checkbox(label, value, **kw):
        seen["kw"] = kw
        return False, value

    monkeypatch.setattr(settings_3d.controls, "checkbox", checkbox)
    settings_3d._rig(SimpleNamespace(rigging_available=True), {"rig": False, "rig_template": ""})
    assert seen["kw"]["enabled"] is True


# --- Ctrl+Enter -------------------------------------------------------------


class _Host(EventsMixin):
    def __init__(self, stage):
        self.app_ctx = SimpleNamespace(
            state=SimpleNamespace(
                mode="create",
                create=SimpleNamespace(stage=stage),
                palette_open=False,
                shortcuts_requested=False,
                show_fps=False,
                manual=SimpleNamespace(open=False),
                tour=None,
                source_job=None,
                form_2d={},
                form_3d={},
            ),
            cache=SimpleNamespace(get=lambda _id: None),
        )
        self.viewer = SimpleNamespace(pose_mode=False)

    def _note_mode(self, state):
        pass


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("reference", ["generate"]),
        ("mesh", ["promote"]),
        ("rig", []),
        ("pose", []),
        ("export", []),
    ],
)
def test_ctrl_enter_runs_only_the_current_stages_generate(monkeypatch, stage, expected):
    calls = []
    monkeypatch.setattr(settings_2d, "generate", lambda *a, **k: calls.append("generate"))
    monkeypatch.setattr(settings_3d, "promote", lambda *a, **k: calls.append("promote"))
    from realmspinner.studio import docmodes

    monkeypatch.setattr(docmodes, "pose_undo_key", lambda *a, **k: False)
    host = _Host(stage)
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=pygame.KMOD_CTRL)
    host._shortcut(event)
    assert calls == expected


def test_ctrl_enter_does_nothing_on_the_rig_stage(monkeypatch):
    calls = []
    monkeypatch.setattr(settings_3d, "promote", lambda *a, **k: calls.append("promote"))
    from realmspinner.studio import docmodes

    monkeypatch.setattr(docmodes, "pose_undo_key", lambda *a, **k: False)
    _Host("rig")._shortcut(
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=pygame.KMOD_CTRL)
    )
    assert calls == []
