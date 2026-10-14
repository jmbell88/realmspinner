"""The 2026-10-14 audit, fixer F58: the inspector's panes, the first-run panel,
the palette row and the follow-up preview.

Findings closed here: shell-review-settings-01 (first-run CUDA verdict),
create-panes-09, service-kinds-02 and create-panes-04 (follow-up preview and
draft previews), shell-home-library-05 (a trashed asset's inspector),
pipelines-image-08 (the matte note's directory walk), shell-chrome-12 (palette
rows report to the probe census), create-panes-05 (Save GLB without Blender),
create-panes-07 (reorient panel decisions), create-panes-02 (the sprite cell
size ladder), create-panes-01 and create-panes-03 (re-texture).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from realmspinner import doctor, fetch, vram
from realmspinner.studio import textures
from realmspinner.studio.panes import (
    first_run,
    followup_preview,
    inspector,
    palette,
    pose_panel,
    remesh_panel,
    reorient_panel,
    sprite_panel,
    texture_panel,
)
from realmspinner.studio.state import AppState

# --- shell-review-settings-01 -----------------------------------------------


def _first_run_ctx(svc, checks, plan):
    svc.vram_plan = plan
    return SimpleNamespace(
        svc=svc,
        runtime=SimpleNamespace(
            checks=list(checks), device_memory=vram.DeviceMemory(24.0, 20.0, "Test GPU")
        ),
        state=AppState(),
        model_rows=[],
        model_picks=set(),
        pack_rows=[],
        rigging_available=False,
        first_run=True,
        first_run_info={},
        gpu_name="",
        toast=lambda *_args: None,
    )


def test_a_torchless_base_install_with_a_healthy_card_is_not_told_it_has_no_gpu(svc, monkeypatch):
    """On a base install torch is the Image generation pack, not yet installed:
    doctor's CUDA row is ``ok=False, pending_install=True``. That is "unmeasured",
    and TRELLIS needs no torch, so the NVML-backed VRAM row alone decides."""
    monkeypatch.setattr(fetch, "disk_refusal", lambda _jobs: None)
    plan = vram.plan(exclusive=False, total_gib=24.0)
    checks = (
        doctor.Check(
            "CUDA", False, "torch not installed -- install the Image generation pack",
            fatal=False, pending_install=True,
        ),
        doctor.Check("VRAM budget", True, plan.reason, fatal=False),
    )
    info = first_run.snapshot(_first_run_ctx(svc, checks, plan))
    assert info["three_d"]["ready"] is True
    assert info["three_d"]["detail"] == "Ready"


def test_a_real_cuda_failure_still_reads_as_not_ready(svc, monkeypatch):
    """The pending-install carve-out must not swallow a genuine fault."""
    monkeypatch.setattr(fetch, "disk_refusal", lambda _jobs: None)
    plan = vram.plan(exclusive=False, total_gib=24.0)
    checks = (
        doctor.Check("CUDA", False, "torch.cuda.is_available() is False", fatal=False),
        doctor.Check("VRAM budget", True, plan.reason, fatal=False),
    )
    assert first_run.snapshot(_first_run_ctx(svc, checks, plan))["three_d"]["ready"] is False


# --- service-kinds-02, create-panes-09 --------------------------------------


def _job_dir(job_id):
    return Path("lib") / job_id


def _row(kind, **params):
    return {
        "id": params.pop("row_id", "ROW"),
        "kind": kind,
        "status": params.pop("status", "done"),
        "created_at": params.pop("created_at", 1.0),
        "params": {"source_job": "SRC", **params},
    }


def test_a_rig_on_a_template_with_no_deformation_battery_draws_no_what_this_made_section():
    # Clay's Weight with Blender makes a ``blank`` rig; humanoid is the one with
    # a battery that renders rig_qa.png.
    assert followup_preview.product_images(_row("rig", template="blank"), _job_dir) == []
    assert followup_preview.product_images(_row("rig", template="fish"), _job_dir) == []
    humanoid = followup_preview.product_images(_row("rig", template="humanoid"), _job_dir)
    assert [p.label for p in humanoid] == ["rig check"]


def test_a_rig_with_an_unknown_template_draws_nothing_rather_than_raising():
    assert followup_preview.product_images(_row("rig", template="no-such"), _job_dir) == []


def test_an_older_followup_row_does_not_show_a_later_rows_picture():
    older = _row("retexture", row_id="OLD", created_at=1.0)
    newer = _row("remesh", row_id="NEW", created_at=2.0)
    siblings = [newer, older]
    assert followup_preview.product_images(older, _job_dir, siblings) == []
    assert [p.label for p in followup_preview.product_images(newer, _job_dir, siblings)] == ["mesh"]
    # A row of the other group, an unfinished row and another source's row do
    # not supersede it.
    unrelated = [
        _row("rig", row_id="R", created_at=3.0),
        _row("remesh", row_id="Q", created_at=3.0, status="running"),
        {**_row("remesh", row_id="Z", created_at=3.0), "params": {"source_job": "OTHER"}},
    ]
    assert [p.label for p in followup_preview.product_images(older, _job_dir, unrelated)] == [
        "mesh"
    ]


class _RecordingTextures:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, dict[str, Any]]] = []

    def get(self, key, path, **kwargs):
        self.calls.append((key, path, kwargs))
        return None


def test_a_sprite_draft_preview_is_not_resampled_below_its_atlas_size(monkeypatch):
    """Both draft call sites (the sprite panel's candidate and the follow-up
    preview) must pass a ceiling above the largest atlas (64 px cells: 256x512)."""
    rec = _RecordingTextures()
    ctx = SimpleNamespace(
        textures=rec, job_dir=lambda job_id: _job_dir(job_id), cache=SimpleNamespace(jobs=[])
    )
    monkeypatch.setattr(sprite_panel, "widgets", MagicMock())
    monkeypatch.setattr(sprite_panel, "imgui", MagicMock())
    controls = MagicMock()
    controls.small_button.return_value = False
    monkeypatch.setattr(sprite_panel, "controls", controls)
    sprite_panel._candidate(ctx, "SRC", "97025e7b7334", "a", {})
    monkeypatch.setattr(followup_preview, "widgets", MagicMock())
    monkeypatch.setattr(followup_preview, "imgui", MagicMock())
    followup_preview.draw(
        ctx, _row("sprite_synthesis", draft_id="97025e7b7334", candidates=2)
    )
    assert len(rec.calls) == 3
    for _key, _path, kwargs in rec.calls:
        assert kwargs["nearest"] is True
        assert kwargs.get("max_side", textures.MAX_SIDE) >= 512


def test_the_default_thumbnail_ceiling_really_resamples_a_draft_atlas(tmp_path):
    """The premise: ``ThumbnailCache`` shrinks a 256x512 atlas at its default."""
    from PIL import Image

    path = tmp_path / "a.png"
    Image.new("RGBA", (256, 512), (10, 20, 30, 255)).save(path)

    class _GL:
        NEAREST = 1
        LINEAR = 2

        def texture(self, size, comps, data):
            return SimpleNamespace(size=size, filter=None, repeat_x=False, repeat_y=False)

    cache = textures.ThumbnailCache(_GL())
    assert cache.get("k1", path, nearest=True).size == (128, 256)
    assert cache.get("k2", path, nearest=True, max_side=textures.PIXEL_MAX_SIDE).size == (256, 512)


# --- shell-home-library-05 ---------------------------------------------------


def _inspector_ctx(job):
    calls: list[str] = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(selected=job["id"], mode="library", checked=set()),
        job=lambda: job,
        svc=object(),
        viewer=None,
        confirms=SimpleNamespace(ask=lambda confirm: calls.append(f"confirm:{confirm.title}")),
        submit=lambda key, fn, *args: calls.append(f"submit:{key}"),
        toast=lambda *a: None,
        cache=SimpleNamespace(get=lambda job_id: job, jobs=[job]),
    )
    return ctx, calls


def test_the_inspector_offers_a_trashed_asset_nothing_but_restore_and_purge(monkeypatch):
    job = {
        "id": "trash0000001",
        "kind": "text",
        "stage": "model",
        "status": "done",
        "deleted_at": 5.0,
        "files": ["model.glb", "rig.glb"],
        "params": {},
    }
    ctx, calls = _inspector_ctx(job)
    drawn: list[str] = []
    for name in (
        "_header", "_meta", "_edit_actions", "_details_tab", "_rig_tab", "downloads",
        "_error", "_followup_failures",
    ):
        monkeypatch.setattr(inspector, name, lambda *a, _n=name, **k: drawn.append(_n))
    monkeypatch.setattr(inspector, "flush_unsent_for", lambda *a: None)
    monkeypatch.setattr(inspector.manual_render, "help_button", lambda *a, **k: None)
    tabs = MagicMock()
    monkeypatch.setattr(inspector.widgets, "tab_bar", tabs)
    monkeypatch.setattr(inspector.widgets, "muted_wrapped", lambda *a: None)
    monkeypatch.setattr(inspector.widgets, "empty_state", lambda *a: None)
    pressed = {"Restore": True, "Delete permanently...": False}
    monkeypatch.setattr(
        inspector.controls,
        "button",
        lambda label, *a, **k: pressed.get(label.split("##")[0], False),
    )
    inspector.draw(ctx)
    assert tabs.call_count == 0, "a trashed asset draws no Details/Rig & Pose/Export tabs"
    assert "_edit_actions" not in drawn
    assert calls == ["submit:restore:trash0000001"]
    pressed.update({"Restore": False, "Delete permanently...": True})
    calls.clear()
    inspector.draw(ctx)
    assert calls == ["confirm:Delete permanently?"]


# --- pipelines-image-08 -------------------------------------------------------


def test_matte_note_does_not_walk_the_model_directory_every_frame(monkeypatch):
    from realmspinner.pipelines import matting

    walks = []
    monkeypatch.setattr(matting, "available", lambda config=None: walks.append(1) or False)
    shown = []
    monkeypatch.setattr(inspector.widgets, "muted_wrapped", shown.append)
    class _Svc:  # a real service is a plain class, weakly referenceable
        config = object()

    ctx = SimpleNamespace(svc=_Svc())
    for _ in range(60):
        inspector._matte_note(ctx, None)
    assert len(walks) == 1
    assert len(shown) == 60, "the answer is memoised, not dropped"


# --- shell-chrome-12 ------------------------------------------------------------


def test_a_greyed_palette_row_reports_its_reason_to_the_probe_census(monkeypatch):
    seen: list[dict[str, Any]] = []

    def selectable(label, selected=False, *a, enabled=True, reason="", tooltip="", **k):
        seen.append({"label": label, "enabled": enabled, "reason": reason})
        return (False, False)

    imgui = MagicMock()
    imgui.get_cursor_screen_pos.return_value = SimpleNamespace(x=0.0, y=0.0)
    imgui.get_content_region_avail.return_value = SimpleNamespace(x=300.0, y=0.0)
    imgui.get_text_line_height.return_value = 16.0
    imgui.calc_text_size.return_value = SimpleNamespace(x=10.0, y=10.0)
    monkeypatch.setattr(palette, "imgui", imgui)
    monkeypatch.setattr(palette.controls, "selectable", selectable)
    item = SimpleNamespace(
        label="Rig this mesh", hint="", key="rig", why="Select a mesh first.",
        enabled=lambda ctx: False,
    )
    assert palette._row(object(), "command", item, False) is False
    assert seen == [
        {"label": "Rig this mesh##cmd/rig", "enabled": False, "reason": "Select a mesh first."}
    ]
    imgui.begin_disabled.assert_not_called()


# --- create-panes-05 --------------------------------------------------------------


def _save_glb_calls(monkeypatch, *, rigging_available):
    calls: list[dict[str, Any]] = []

    def disabled_button(label, enabled, *a, reason="", tooltip=""):
        calls.append({"label": label, "enabled": enabled, "reason": reason})
        return False

    widgets = MagicMock()
    widgets.disabled_button.side_effect = disabled_button
    widgets.list_filter.return_value = ""
    widgets.button_width.return_value = 10
    monkeypatch.setattr(pose_panel, "widgets", widgets)
    monkeypatch.setattr(pose_panel, "imgui", MagicMock())
    controls = MagicMock()
    controls.small_button.return_value = False
    monkeypatch.setattr(pose_panel, "controls", controls)
    ctx = SimpleNamespace(
        state=SimpleNamespace(preview={"poses": [{"id": "0123456789ab", "name": "idle"}]}),
        viewer=None,
        busy=lambda key: False,
        rigging_available=rigging_available,
    )
    pose_panel._saved_list(ctx, {"id": "job000000001"})
    return [c for c in calls if c["label"] == "Save GLB..."]


def test_save_glb_is_greyed_with_the_blender_reason_when_rigging_is_unavailable(monkeypatch):
    (call,) = _save_glb_calls(monkeypatch, rigging_available=False)
    assert call["enabled"] is False
    assert "Blender" in call["reason"]


def test_save_glb_stays_live_when_blender_is_available(monkeypatch):
    (call,) = _save_glb_calls(monkeypatch, rigging_available=True)
    assert call["enabled"] is True


# --- create-panes-07 ------------------------------------------------------------------


def test_reorient_panel_current_size_prefers_the_asked_size_then_the_measured_one():
    f = reorient_panel.current_size_m
    assert f({"params": {"size_m": 2.0, "mesh_report": {"achieved_size_m": 1.5}}}) == 2.0
    assert f({"params": {"mesh_report": {"achieved_size_m": 1.23456}}}) == 1.235
    assert f({"params": {}}) == reorient_panel.DEFAULT_SIZE_M
    # Zero, negative and bool are not sizes.
    assert f({"params": {"size_m": 0, "mesh_report": {"achieved_size_m": True}}}) == 1.0
    assert f({"params": {"size_m": -3}}) == 1.0


def test_reorient_panel_front_reason_matches_the_doors_refusals():
    r = reorient_panel.front_reason
    assert "no recorded front" in r({"params": {}, "files": []})
    assert "no recorded front" in r({"params": {"front_yaw": 0}, "files": []})
    assert r({"params": {"front_yaw": 30.0}, "files": ["model.glb"]}) == ""
    for name in ("rig.glb", "rig.json"):
        assert "rig fitted" in r({"params": {"front_yaw": 30.0}, "files": [name]})


def test_reorient_panel_submit_kwargs_sends_a_size_only_when_it_changed():
    job = {"params": {"size_m": 1.0}}
    same = reorient_panel.submit_kwargs({"size_m": 1.0002, "ground": True}, job)
    assert same == {"size_m": None, "ground": True}
    changed = reorient_panel.submit_kwargs({"size_m": 1.5, "ground": False}, job)
    assert changed == {"size_m": 1.5, "ground": False}


def test_reorient_panel_validate_states_the_doors_refusals_before_the_press():
    from realmspinner import guidance

    job = {"params": {"size_m": 1.0}}
    v = reorient_panel.validate
    assert v({"size_m": "abc", "ground": True}, job) == ["Size must be a number."]
    assert "between" in v({"size_m": guidance.SIZE_MAX_M * 2, "ground": True}, job)[0]
    assert "between" in v({"size_m": guidance.SIZE_MIN_M / 2, "ground": True}, job)[0]
    assert v({"size_m": 1.0, "ground": False}, job) == ["Change the size or turn Ground on."]
    assert v({"size_m": 1.0, "ground": True}, job) == []
    assert v({"size_m": 2.0, "ground": False}, job) == []


# --- create-panes-02 ------------------------------------------------------------------------


class _FakeForm:
    def __init__(self) -> None:
        self.combos: dict[str, list[tuple[str, str]]] = {}

    def combo(self, field, label, value, items, **kwargs):
        self.combos[field] = list(items)
        return False, value

    def number(self, field, label, value, **kwargs):
        return False, value


def test_the_sprite_panel_cell_size_ladder_is_the_selected_types_own(monkeypatch):
    monkeypatch.setattr(sprite_panel, "imgui", MagicMock())
    controls = MagicMock()
    controls.small_button.return_value = False
    monkeypatch.setattr(sprite_panel, "controls", controls)
    options = sprite_panel._options()
    for entry in options["sheet_types"]:
        form = {
            "sheet_type": entry["key"], "logical_size": 64, "colors": 32, "seed_a": 1, "seed_b": 2,
        }
        ui = _FakeForm()
        sprite_panel._controls(SimpleNamespace(), form, ui)
        offered = [int(value) for value, _label in ui.combos["logical_size"]]
        assert offered == sorted(entry["logical_sizes"]), entry["key"]
        assert form["logical_size"] in offered, f"{entry['key']} kept a size it refuses"
    eight = next(e for e in options["sheet_types"] if e["key"] == "walk8")
    assert eight["logical_sizes"] == [32]


def test_clamp_size_keeps_a_rung_and_otherwise_steps_down_then_up():
    assert sprite_panel.clamp_size([32], 64) == 32
    assert sprite_panel.clamp_size([32, 48, 64], 48) == 48
    assert sprite_panel.clamp_size([48, 64], 32) == 48
    assert sprite_panel.clamp_size([], 64) == 64


# --- create-panes-01, create-panes-03 ----------------------------------------------------------


def test_a_retexture_reports_or_rewrites_the_rig_glb_that_embeds_the_old_skin(monkeypatch):
    assert texture_panel.rig_skin_note(["model.glb"]) is None
    note = texture_panel.rig_skin_note(["model.glb", "rig.glb"])
    assert note and "rig" in note and "old surface" in note

    widgets = MagicMock()
    shown: list[tuple[str, str]] = []
    widgets.text_colored.side_effect = lambda _c, text: shown.append(("warn", text))
    widgets.muted.side_effect = lambda text: shown.append(("muted", text))
    monkeypatch.setattr(texture_panel, "widgets", widgets)
    texture_panel._warn(
        SimpleNamespace(), {"files": ["model.glb", "rig.glb"]}, {"depth": True}
    )
    text = " ".join(t for _k, t in shown)
    assert "unaffected" not in text
    assert any(kind == "warn" and "old surface" in t for kind, t in shown)


def test_retexture_panel_is_greyed_with_the_blender_sentence_when_blender_is_not_installed(
    monkeypatch,
):
    widgets = MagicMock()
    widgets.header.return_value = True
    monkeypatch.setattr(texture_panel, "widgets", widgets)
    monkeypatch.setattr(texture_panel, "manual_render", MagicMock())
    forms = MagicMock()
    monkeypatch.setattr(texture_panel, "forms", forms)
    monkeypatch.setattr(remesh_panel, "blender_available", lambda ctx: False)
    texture_panel.draw(SimpleNamespace(), {"id": "a", "files": ["model.glb"]})
    forms.Form.assert_not_called()
    sentence = widgets.muted_wrapped.call_args.args[0]
    assert "Blender" in sentence and "not installed" in sentence
