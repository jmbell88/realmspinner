"""Regression tests for the 2026-10-03 audit's agent_character findings
(agents-03, agents-04, agents-05)."""

from __future__ import annotations

from typing import Any

import pytest

from realmspinner.studio import agent_character as ac

pytestmark = pytest.mark.filterwarnings("ignore")


def test_character_sheet_create_refuses_a_malformed_job_id_before_touching_the_filesystem(
    svc: Any, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """agents-03: `job_id="../outside"` reached `svc.job_dir(job_id)/rig.glb`
    .exists() -- an existence oracle outside the data dir."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "rig.glb").write_bytes(b"x")
    probed: list[Any] = []
    real = svc.job_dir
    monkeypatch.setattr(svc, "job_dir", lambda jid: probed.append(jid) or real(jid), raising=False)
    result = ac.call(
        svc,
        ac.Session(),
        "character_sheet_create",
        {
            "job_id": "../outside",
            "template": next(iter(ac._enums().sheet_templates)),
            "movements": [{"name": next(iter(ac._enums().movements))}],
        },
    )
    assert result["isError"]
    assert "already rigged" not in result["content"][0]["text"]
    assert result["structuredContent"].get("field") == "job_id"
    assert probed == []


def test_a_movement_entry_with_an_unknown_key_is_refused_by_name(svc: Any) -> None:
    """agents-04: the item schema says additionalProperties false; a misspelt
    `frmaes` / a `directions` on character_create was dropped silently."""
    name = next(iter(ac._enums().movements))
    result = ac.call(
        svc,
        ac.Session(),
        "character_create",
        {"prompt": "a knight", "movements": [{"name": name, "directions": 4}]},
    )
    assert result["isError"]
    assert result["structuredContent"]["field"] == "movements"
    assert "directions" in result["content"][0]["text"]


def test_a_state_refusal_without_a_field_does_not_tell_the_agent_to_fix_job_id(
    svc: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """agents-05: a field-less Invalid/Conflict (Blender missing, job already
    finished) came back as recovery=fix_arguments on job_id."""
    from realmspinner.service import errors

    def boom(*a: Any, **k: Any) -> dict:
        raise errors.Conflict("a rig for this mesh is already running")

    monkeypatch.setitem(ac.HANDLERS, "character_rig", boom)
    result = ac.call(svc, ac.Session(), "character_rig", {"job_id": "0" * 12})
    sc = result["structuredContent"]
    assert sc.get("field") != "job_id"
    assert sc.get("recovery") != "fix_arguments"
    assert sc.get("recovery") == "wait"

    def boom2(*a: Any, **k: Any) -> dict:
        raise errors.Invalid("Rigging needs Blender, which is not installed.")

    monkeypatch.setitem(ac.HANDLERS, "character_rig", boom2)
    result = ac.call(svc, ac.Session(), "character_rig", {"job_id": "0" * 12})
    sc = result.get("structuredContent", {})
    assert sc.get("field") != "job_id"
    assert sc.get("recovery") != "fix_arguments"
