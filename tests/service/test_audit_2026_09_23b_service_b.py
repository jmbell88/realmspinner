"""Regression test for the 2026-09-23 (second run) audit's finding service-05."""

from __future__ import annotations

from realmspinner import models

# --- service-05: sdxl_cfg's description claimed exclusive ControlNet/CFG ---


def test_no_base_model_description_claims_exclusive_controlnet_or_cfg_support_the_registry_contradicts():  # noqa: E501
    """The 2026-09-23 audit, finding service-05: ``sdxl_cfg``'s description
    said it was "the only family that takes ControlNet, and the only one
    where the negative prompt carries full weight" -- but four other bases
    (``playground``, ``sdxl_cfg_pag``, ``juggernaut``, ``dreamshaper``) are
    also ``controlnet=True`` with the same full-CFG recipe. Swept over every
    row rather than pinned to ``sdxl_cfg`` alone, so a future entry that
    reintroduces the same claim fails this test too.
    """
    controlnet_bases = [m for m in models.BASE_MODELS.values() if m.controlnet]
    assert len(controlnet_bases) > 1, "fixture assumption: more than one base takes ControlNet"
    for base in models.BASE_MODELS.values():
        lowered = base.description.lower()
        assert "the only family" not in lowered, base.key
        claims_exclusive_cfg = "only one" in lowered and (
            "controlnet" in lowered or "cfg" in lowered
        )
        assert not claims_exclusive_cfg, base.key
