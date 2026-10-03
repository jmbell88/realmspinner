"""The 2026-10-03 audit's docs-01 and docs-02: chapter 41 misstated the boolean
vocabulary, and chapter 12 called the Budget combo hidden and single-tier."""

from pathlib import Path

MANUAL = Path(__file__).resolve().parents[2] / "docs" / "manual"


def test_manual_41_boolean_vocabulary_matches_env_bool(monkeypatch):
    from realmspinner import config

    text = (MANUAL / "41-configuration.md").read_text(encoding="utf-8")
    assert "anything else is off" not in text
    for word in ("yes", "no", "false", "off"):
        assert f"`{word}`" in text
    # The code the sentence describes: yes is on, an unknown word is the default.
    monkeypatch.setenv("RS_PROBE", "yes")
    assert config._env_bool("RS_PROBE", False) is True
    monkeypatch.setenv("RS_PROBE", "maybe")
    assert config._env_bool("RS_PROBE", True) is True


def test_manual_12_budget_paragraph_describes_the_budget_combo_the_mesh_stage_draws():
    from realmspinner.studio.modes.create.ui.panes import settings_3d

    text = (MANUAL / "12-tuning-what-you-get.md").read_text(encoding="utf-8")
    assert "currently offers only one tier" not in text
    assert "zero of twenty" not in text
    assert "**Budget**" in text and "23-generating-meshes.md" in text
    # The label the paragraph names is the one the pane draws.
    assert '"Budget"' in __import__("inspect").getsource(settings_3d._budget)
