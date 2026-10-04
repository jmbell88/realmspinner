"""The 2026-10-04 audit, finding create-20 (the 3D half) and the ControlNet refusal's wording.

create-20  the findings "Use" button on the 3D Budget wrote ``form["profile"]``
           alone, so on the default form (Game-ready 5k) the combo did not move
           and ``promote_kwargs`` kept sending ``lowpoly_triangles: 5000``.
create-54  ``validate_request``'s ControlNet refusal said "guidance 0", which is
           true of Turbo/Hyper/Lightning and false of LCM and FLUX.2 klein
           distilled (1.0).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from realmspinner import generation, models
from realmspinner.bench import findings as findings_lib
from realmspinner.studio.modes.create.engine import mesh as create_mesh
from realmspinner.studio.modes.create.ui.panes import settings_3d
from realmspinner.studio.state import DEFAULT_FORM_3D


def test_use_button_on_the_3d_profile_goes_through_the_budget_choice_path(tmp_path, monkeypatch):
    doc = {
        "version": 5,
        "generated": "x",
        "params": {
            "profile": {
                "standard": {"n": 8, "accepts": 7, "wilson_low": 0.55},
                "raw": {"n": 8, "accepts": 2, "wilson_low": 0.1},
            }
        },
    }
    (tmp_path / "findings.json").write_text(json.dumps(doc), encoding="utf-8")
    findings_lib._CACHE.clear()
    form = dict(DEFAULT_FORM_3D)
    assert form["lowpoly_triangles"] > 0, "the default form carries a Game-ready budget"
    ctx = SimpleNamespace(
        svc=SimpleNamespace(config=SimpleNamespace(bench_dir=tmp_path)),
        cache=SimpleNamespace(get=lambda _id: None),
        state=SimpleNamespace(source_job=None),
    )
    monkeypatch.setattr(settings_3d.controls, "button", lambda *a, **k: True)
    monkeypatch.setattr(settings_3d.widgets, "muted", lambda *a, **k: None)
    monkeypatch.setattr(settings_3d.imgui, "same_line", lambda *a, **k: None)
    before = settings_3d._budget_current(form)

    settings_3d._best_value_offer(ctx, form, "profile", form["profile"])

    assert form["profile"] == "standard"
    assert form["lowpoly_triangles"] == 0, "the Game-ready budget must not outlive the pick"
    assert settings_3d._budget_current(form) == "standard" != before
    kwargs = create_mesh.promote_kwargs(form)
    assert kwargs["profile"] == "standard"
    assert not kwargs.get("lowpoly_triangles")


def test_the_controlnet_refusal_does_not_claim_guidance_zero_for_lcm():
    # LCM (``pixel``) and FLUX.2 klein distilled both run at guidance 1.0.
    for key in ("pixel", "flux_klein_distilled"):
        spec = models.BASE_MODELS[key]
        assert spec.guidance_scale == 1.0, key
        assert not spec.controlnet, key
        request = generation.GenerationRequest(
            generation_type="image",
            prompt="a wooden crate",
            structure_control="canny",
            model_mode="advanced",
            model_override=key,
        )
        resolved = generation.resolve_recipe(request, None)
        issues = generation.validate_request(request, resolved)
        refusals = [i for i in issues if i.field == "base_model"]
        assert refusals, (key, issues)
        message = refusals[0].message
        assert "guidance 0" not in message, message
        assert "guidance 1.0 or lower" in message, message
        assert "ControlNet" in message and "full-CFG" in message
