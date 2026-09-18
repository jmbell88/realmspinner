"""The 2026-09-07 Create review's friction sweep: items 5.1, 5.2, 5.4 and 7.1.

Pure source inspection for the two imgui-drawn shapes that cannot be driven
headlessly (5.1's dedup, 7.1's new button) -- ``tests/test_create_stage_
controls.py``'s own reasoning applies here too. 5.2 and 5.4 are pure enough
Python (a lookup, a dataclass check) to exercise directly against a stub
``ctx``, the way ``tests/test_matte_handoff.py``'s ``_Ctx`` already does.
"""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

from warlock.service import matte as svc_matte
from warlock.studio import matte_preview
from warlock.studio.modes.create.engine import mesh as create_mesh
from warlock.studio.modes.create.ui.panes import settings_3d
from warlock.studio.panes import sheet_panel, stage_rig
from warlock.studio.state import DEFAULT_FORM_3D, AppState


class _Settings:
    """Enough of ``studio.settings.Settings`` for a stub ctx: get/set on a
    plain dict, no file behind it."""

    def __init__(self, data: dict | None = None) -> None:
        self.data = dict(data or {})

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value) -> None:
        self.data[key] = value


class _Ctx:
    """Enough of ``Ctx`` for the frame-thread half of ``settings_3d``."""

    def __init__(self, jobs: dict | None = None) -> None:
        self.state = AppState()
        self.state.form_3d = dict(DEFAULT_FORM_3D)
        self._jobs = dict(jobs or {})
        self.svc = SimpleNamespace(job_dir=lambda _id: None)
        self.submitted: list = []
        self.cache = SimpleNamespace(
            get=lambda job_id: self._jobs.get(job_id),
            jobs=list(self._jobs.values()),
        )
        self.textures = None
        self.toasts: list = []
        self.settings = _Settings()
        self.rig_templates: list = []
        self.rig_default = "humanoid"
        self.rigging_available = True

    def job(self):
        return self._jobs.get(self.state.selected)

    def submit(self, key, fn, *args, **kwargs):
        self.submitted.append((key, fn, args, kwargs))
        return True

    def busy(self, key):
        return any(k == key for k, *_ in self.submitted)

    def toast(self, message, level="info", **extra):
        self.toasts.append((message, level))


# --- item 5.1: two identical skeleton combos, one field ----------------------


def test_the_rig_checkbox_and_the_rig_stage_draw_one_skeleton_picker():
    """The 2026-09-07 review, item 5.1.

    Both call sites route through ``stage_rig.skeleton_field`` rather than
    each drawing ``widgets.labeled_combo("Skeleton", ...)`` themselves --
    unfixed, this string appears twice across the two panes.
    """
    assert "def skeleton_field(" in inspect.getsource(stage_rig)
    assert "stage_rig.skeleton_field(" in inspect.getsource(settings_3d._rig)
    assert "skeleton_field(" in inspect.getsource(stage_rig._skeleton_picker)

    combined = inspect.getsource(stage_rig) + inspect.getsource(settings_3d)
    calls = re.findall(r'labeled_combo\(\s*\n?\s*"Skeleton"', combined)
    assert len(calls) == 1, (
        "more than one place still draws the Skeleton combo directly -- the "
        "two implementations this item was supposed to collapse"
    )


# --- item 5.2: a finished mesh should describe itself, not a missing reference


def test_selecting_a_finished_mesh_does_not_ask_the_mesh_stage_to_choose_a_reference():
    """The 2026-09-07 review, item 5.2.

    Verified at ``library.select`` (panes/library.py): ``state.source_job`` is
    only ever set for a *done reference* row, so selecting a finished mesh
    leaves it untouched (or None). Unfixed, ``create_mesh.validate`` then sees
    no source at all and refuses with "Choose a reference first." for a job
    that plainly has one.
    """
    reference = {
        "id": "ref1",
        "stage": "reference",
        "status": "done",
        "files": ["input.png"],
    }
    mesh = {
        "id": "mesh1",
        "stage": "model",
        "status": "done",
        "files": ["model.glb"],
        "parent_id": "ref1",
    }
    ctx = _Ctx({"ref1": reference, "mesh1": mesh})
    ctx.state.selected = "mesh1"
    ctx.state.source_job = None  # never followed a reference -- the bug's setup

    resolved = settings_3d._effective_source(ctx, ctx.cache.get(ctx.state.source_job))

    assert resolved is reference
    assert create_mesh.validate(resolved) == []


def test_an_explicit_source_pick_still_wins_over_the_selected_mesh():
    """The fallback must never override a real choice: dragging a *different*
    reference onto the form while a finished mesh is selected has to promote
    that reference, not the mesh's own parent."""
    reference = {"id": "ref1", "stage": "reference", "status": "done", "files": ["input.png"]}
    other = {"id": "ref2", "stage": "reference", "status": "done", "files": ["input.png"]}
    mesh = {
        "id": "mesh1",
        "stage": "model",
        "status": "done",
        "files": ["model.glb"],
        "parent_id": "ref1",
    }
    ctx = _Ctx({"ref1": reference, "ref2": other, "mesh1": mesh})
    ctx.state.selected = "mesh1"
    ctx.state.source_job = "ref2"

    resolved = settings_3d._effective_source(ctx, ctx.cache.get(ctx.state.source_job))

    assert resolved is other


# --- item 5.4: the matte modal opens on every Make 3D -------------------------


def _preview(**overrides) -> svc_matte.Preview:
    fields = dict(
        job_id="ref1",
        stamp=1,
        width=4,
        height=4,
        rgb=bytes(4 * 4 * 3),
        source="birefnet",
        approved=False,
        coverage=0.5,
        reasons=(),
        warnings=(),
    )
    fields.update(overrides)
    return svc_matte.Preview(**fields)


def _promoted(ctx: _Ctx, *, skip: bool) -> None:
    ctx.settings.set(settings_3d.SKIP_CLEAN_MATTE_SETTING, skip)
    source = {"id": "ref1", "status": "done", "files": ["input.png"], "params": {}}
    settings_3d.promote(ctx, source, ctx.state.form_3d)


def test_a_clean_birefnet_matte_skips_the_preview_when_the_user_asked_it_to():
    ctx = _Ctx({"ref1": {"id": "ref1", "stage": "reference", "status": "done"}})
    _promoted(ctx, skip=True)
    state = ctx.state.matte
    state.preview = _preview()
    state.stamp = state.preview.stamp

    handled = settings_3d._wants_auto_accept(ctx, state)

    assert handled is True
    assert not matte_preview.is_open(ctx), "the modal must never have opened"
    assert [key for key, *_ in ctx.submitted] == ["submit"], (
        "the skip must go through the identical submit_promotion path a "
        "pressed Accept uses"
    )


def test_a_warned_reference_still_opens_the_matte_preview_however_the_setting_is_set():
    ctx = _Ctx({"ref1": {"id": "ref1", "stage": "reference", "status": "done"}})
    _promoted(ctx, skip=True)
    state = ctx.state.matte
    state.preview = _preview(warnings=("low coverage",))
    state.stamp = state.preview.stamp

    handled = settings_3d._wants_auto_accept(ctx, state)

    assert handled is False
    assert matte_preview.is_open(ctx)
    assert ctx.submitted == []


def test_a_refused_or_fallback_matte_never_skips_even_with_the_setting_on():
    ctx = _Ctx({"ref1": {"id": "ref1", "stage": "reference", "status": "done"}})
    _promoted(ctx, skip=True)
    state = ctx.state.matte

    state.preview = _preview(reasons=("subject fills the frame",))
    state.stamp = state.preview.stamp
    assert settings_3d._wants_auto_accept(ctx, state) is False

    state.preview = _preview(source="flood")
    assert settings_3d._wants_auto_accept(ctx, state) is False


def test_the_setting_off_never_skips_a_clean_matte_either():
    ctx = _Ctx({"ref1": {"id": "ref1", "stage": "reference", "status": "done"}})
    _promoted(ctx, skip=False)
    state = ctx.state.matte
    state.preview = _preview()
    state.stamp = state.preview.stamp

    assert settings_3d._wants_auto_accept(ctx, state) is False
    assert ctx.submitted == []


# --- item 7.1: a turnaround needs no rig, but its button was on the Pose stage


def test_the_mesh_stage_renders_a_turnaround_for_an_unrigged_prop():
    """The 2026-09-07 review, item 7.1.

    ``docs/manual/27-sprite-sheets.md`` states a turnaround needs no rig; this
    checks the new Mesh-stage control submits through ``sheet_panel``'s own
    door (``svc_sheets.create_sheet``) under ``sheet_panel``'s own key
    (``f"sheet:{job_id}"``) with no rigged-mesh requirement anywhere in its
    guard, and that it is actually wired into the stage's draw.
    """
    source = inspect.getsource(settings_3d._turnaround)
    assert 'f"sheet:{job_id}"' in source, (
        "must submit under sheet_panel's own key, or a press here and a press "
        "on the Pose stage's own button would not be refused as one submit"
    )
    assert "svc_sheets.create_sheet" in source
    assert "poses=[]" in source, "a turnaround needs no rig and asks for no poses"
    assert "yaws=8" in source, "defaults to the 8-direction turnaround"
    assert "ctx.rigging_available" in source, "still respects the Blender gate"
    assert "sheet_panel.sheet_cap_reason(" in source
    assert "rig.glb" not in source, "no rigged-mesh requirement belongs in this guard"

    assert "_turnaround(ctx)" in inspect.getsource(settings_3d._draw_form)


def test_the_turnaround_key_matches_the_pose_stages_own_sheet_submit():
    """Both doors must agree on the in-flight key or a second press from the
    other stage would not be refused."""
    turnaround_src = inspect.getsource(settings_3d._turnaround)
    pose_stage_src = inspect.getsource(sheet_panel._submit)
    assert 'f"sheet:{job_id}"' in turnaround_src
    assert 'f"sheet:{job_id}"' in pose_stage_src
