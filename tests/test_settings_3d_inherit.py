"""W1.1: the Mesh stage's "keep the reference's" options name the value they
would actually inherit.

Before this fix ``_platform_options``/``_bg_options`` always offered the
generic "keep the reference's" wording for the unset entry, even when the
selected reference job had a recorded ``platform``/``bg_removal`` value sitting
right there in ``ctx.cache`` -- a user staring at "keep the reference's" had
no way to find out what that was without leaving the Mesh stage.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.studio.panes import settings_3d
from warlock.studio.state import DEFAULT_FORM_3D, AppState


class _Ctx:
    """Enough of ``Ctx`` for ``_platform_options``/``_bg_options``: a source
    job cache and the guidance catalog they read field labels from."""

    def __init__(self, jobs: dict, guidance: dict | None = None) -> None:
        self.state = AppState()
        self.state.form_3d = dict(DEFAULT_FORM_3D)
        self._jobs = dict(jobs)
        self.cache = SimpleNamespace(get=lambda job_id: self._jobs.get(job_id))
        self.guidance = guidance or {
            "fields": {
                "platform": [
                    {"key": "low", "label": "Low detail"},
                    {"key": "high", "label": "High detail"},
                ]
            },
            "bg_removal": ["birefnet", "flood"],
        }


def test_inherited_mesh_option_names_the_references_value():
    reference = {
        "id": "ref1",
        "status": "done",
        "params": {"platform": "high", "bg_removal": "birefnet"},
    }
    ctx = _Ctx({"ref1": reference})
    ctx.state.source_job = "ref1"

    platform_options = settings_3d._platform_options(ctx)
    bg_options = settings_3d._bg_options(ctx)

    assert platform_options[0] == ("", "From reference: High detail")
    assert bg_options[0] == ("", "From reference: birefnet")


def test_no_selected_source_falls_back_to_the_generic_wording():
    ctx = _Ctx({})
    ctx.state.source_job = None

    assert settings_3d._platform_options(ctx)[0] == ("", "keep the reference's")
    assert settings_3d._bg_options(ctx)[0] == ("", "keep the reference's")


def test_a_reference_with_no_recorded_value_also_falls_back():
    """An older reference job, or one whose params never recorded this key,
    must not be misreported as inheriting a value it doesn't have."""
    reference = {"id": "ref1", "status": "done", "params": {}}
    ctx = _Ctx({"ref1": reference})
    ctx.state.source_job = "ref1"

    assert settings_3d._platform_options(ctx)[0] == ("", "keep the reference's")
    assert settings_3d._bg_options(ctx)[0] == ("", "keep the reference's")
