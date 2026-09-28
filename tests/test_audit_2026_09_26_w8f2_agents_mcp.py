"""Regression tests for the 2026-09-26 audit's wave-6 MCP findings:
character_cancel's garbled description (agents-character-01), a missing enum
check on character_clips (agents-character-02), character_rig's "never
replaces" check racing create_rig's own insert (agents-character-03),
character_sheet_create silently dropping `template` once a mesh is already
rigged (agents-character-04), a stale test citation in agent_prompts.py
(agents-clay-03), and agent_prompts.render stringifying a wrong-typed
argument straight into served prompt text (agents-clay-04).
"""

from __future__ import annotations

import inspect
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.kernels.rig import store
from realmspinner.service import rig as svc_rig
from realmspinner.service.errors import Conflict
from realmspinner.studio import agent_character as ac
from realmspinner.studio import agent_prompts

# --- agents-character-01: character_cancel's description ------------------


def test_character_cancel_description_sentence_is_grammatical() -> None:
    """character_cancel's tool description ended in a garbled, unparsable
    sentence since the surface's own first commit (14431faa): "Anything else
    is refused, even a job this same character the connection did not
    itself start." is missing "on" and "that". Pins the corrected wording,
    served verbatim to every connecting agent."""
    tool = next(t for t in ac.tools() if t.name == "character_cancel")
    assert (
        "even a job on this same character that the connection did not "
        "itself start" in tool.description
    )
    assert (
        "even a job this same character the connection did not itself start"
        not in tool.description
    )


# --- agents-character-02: character_clips's template enum -----------------


def test_character_clips_refuses_a_template_outside_the_sheet_template_enum(
    svc: Any,
) -> None:
    """The schema declares template's enum (sheet_templates) but
    _h_character_clips never checked args["template"] against it, unlike
    every sibling handler that takes an enum argument (see the module
    docstring's "Validated here, not left to the door"). "blank" is a real
    rig template (kernels.rig.templates accepts it -- Poser's manual-rig
    bootstrap) that ships no clips, so it is exactly the case
    svc_clips.library's own templates.get_template check does NOT catch (it
    accepts any real template key, not only a sheet-template-with-clips) --
    only the sheet_templates-specific check this finding asks for does."""
    e = ac._enums()
    assert "blank" not in e.sheet_templates  # sanity: real rig template, no clips

    result = ac.call(svc, ac.Session(), "character_clips", {"template": "blank"})
    assert result["isError"]
    assert result["structuredContent"]["field"] == "template"


# --- agents-character-03: character_rig's atomic "never replaces" check ---


def _finished_mesh_job(svc: Any) -> str:
    job_id = svc.store.create("image", "a knight", {}, stage="model", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    return job_id


def test_create_rig_refuse_existing_conflicts_with_an_existing_rig(svc: Any) -> None:
    job_id = _finished_mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    (job_dir / "rig.json").write_text(
        json.dumps({"template": "humanoid"}), encoding="utf-8"
    )
    with pytest.raises(Conflict, match="never replaces"):
        svc_rig.create_rig(svc, job_id, refuse_existing=True)


def test_create_rig_without_refuse_existing_still_lets_library_and_poser_rerig(
    svc: Any,
) -> None:
    """refuse_existing defaults to False -- Library's "Rig this mesh again"
    and Poser's re-rig (modes/poser/mode.py:rerig) both call this same door
    to replace an existing rig on purpose, so the new parameter must not
    narrow what they can do."""
    job_id = _finished_mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    (job_dir / "rig.json").write_text(
        json.dumps({"template": "humanoid"}), encoding="utf-8"
    )
    out = svc_rig.create_rig(svc, job_id, template="quadruped")
    assert out["template"] == "quadruped"


def test_create_rig_refuse_existing_is_checked_inside_the_same_lock_as_the_insert(
    svc: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """character_rig's own "never replaces an existing rig" check used to be
    a bare store.read_rig(job_dir) read in agent_character.py, entirely
    outside create_rig's convert_lock -- the same TOCTOU shape the
    2026-09-20 audit (finding agents-03) already fixed for rig_in_flight two
    lines below it. Widened deterministically like that test's own sibling
    (tests/test_character_agent_doors.py): store.read_rig is patched to
    pause, once, right after the *first* create_rig(refuse_existing=True)
    call reads it from inside the lock -- so a second call attempted in that
    exact gap is proven to block on the lock itself, rather than reach its
    own read the way agent_character.py's pre-fix outer check could.
    """
    monkeypatch.setattr(
        "realmspinner.doctor.blender_check", lambda: SimpleNamespace(ok=True, detail="")
    )
    job_id = _finished_mesh_job(svc)

    real_read_rig = store.read_rig
    checked = threading.Event()
    release = threading.Event()
    armed = True

    def spy_read_rig(job_dir):
        nonlocal armed
        result = real_read_rig(job_dir)
        if armed:
            armed = False
            checked.set()
            assert release.wait(5), "the first call's paused check never resumed"
        return result

    monkeypatch.setattr(store, "read_rig", spy_read_rig)

    results: list = []
    errors: list = []

    def call() -> None:
        try:
            results.append(svc_rig.create_rig(svc, job_id, refuse_existing=True))
        except Conflict as exc:
            errors.append(exc)

    first = threading.Thread(target=call)
    first.start()
    assert checked.wait(5), "the first call's read_rig check never ran"

    second = threading.Thread(target=call)
    second.start()
    # Pre-fix, refuse_existing did not exist at all (a TypeError, proving the
    # gap outright) and the "never replaces" read ran outside any lock in
    # agent_character.py, so a second caller's own read could land in this
    # exact gap. Post-fix, the first call already holds
    # convert_lock(job_id, "rig") before it ever calls store.read_rig, so the
    # second call blocks on that same lock instead of racing a read of its
    # own -- giving it a moment here is what makes that deterministic rather
    # than sleep-dependent.
    second.join(timeout=1.0)

    release.set()
    first.join(5)
    second.join(5)
    assert not first.is_alive()
    assert not second.is_alive()

    assert len(results) == 1
    assert len(errors) == 1
    rig_rows = [
        r
        for r in svc.store.list(limit=1000, kind="rig")
        if (r["params"] or {}).get("source_job") == job_id
    ]
    assert len(rig_rows) == 1


# --- agents-character-04: character_sheet_create's dropped template -------


def test_character_sheet_create_refuses_a_template_once_the_mesh_is_already_rigged(
    svc: Any,
) -> None:
    """svc_troupe.send_to_troupe silently drops `template` once
    job_dir/rig.glb already exists: it delegates straight to
    create_charsheet, which never mints a rig and so never reads template at
    all -- so a caller who asked for one template ("quadruped" here) got a
    different, silent skeleton (the mesh's own existing "humanoid" rig, pre-
    fix this call succeeds on that mismatched skeleton with no error at
    all). Honoring the request would mean re-rigging the mesh under the
    requested template, which this surface must never do; refused by name
    instead."""
    e = ac._enums()
    job_id = _finished_mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    (job_dir / "rig.json").write_text(
        json.dumps({"template": "humanoid"}), encoding="utf-8"
    )

    result = ac.call(
        svc,
        ac.Session(),
        "character_sheet_create",
        {
            "job_id": job_id,
            "movements": [{"name": e.movements[0]}],
            "template": "quadruped",
        },
    )
    assert result["isError"]
    assert result["structuredContent"]["field"] == "template"


# --- agents-clay-03: agent_prompts.py's stale test citation ----------------


def test_agent_prompts_cites_the_real_scan_test_not_a_nonexistent_one() -> None:
    """A comment in agent_prompts.py cited tests/test_agent_prompts.py, which
    has never existed -- the scan it describes
    (test_every_tool_name_a_prompt_mentions_is_a_real_tool) lives in
    tests/mcp/test_rpc_studio.py, which the module's own docstring already
    names correctly two paragraphs above the stale one."""
    import re

    source = Path(inspect.getfile(agent_prompts)).read_text(encoding="utf-8")
    citations = re.findall(r"against the real catalogue by `([^`]+)`", source)
    assert citations, "expected the 'against the real catalogue by ...' comment"
    assert all(c == "tests/mcp/test_rpc_studio.py" for c in citations), citations

    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "tests" / "mcp" / "test_rpc_studio.py").exists()
    assert not (repo_root / "tests" / "test_agent_prompts.py").exists()


# --- agents-clay-04: agent_prompts.render and wrong-typed arguments --------


def test_render_refuses_a_none_valued_required_argument_as_missing() -> None:
    """render() only checked whether a declared name was *present* in
    arguments, never whether its value actually was the string MCP prompt
    arguments always are. {"description": None} used to count as present
    and reach _character_sheets_from_description's own f-string, landing
    the literal text "this description: None" in the served prompt. A
    wrong-typed required value is now refused exactly like a missing one."""
    result = agent_prompts.render(
        "character_sheets_from_description", {"description": None}
    )
    assert result == ["description"]


def test_render_drops_a_wrong_typed_optional_argument_instead_of_stringifying_it() -> (
    None
):
    """An optional argument of the wrong type (movements sent as an int, say)
    must not reach a render function's own f-string either -- it is dropped,
    treated exactly like a caller who never sent it, instead of being
    stringified into the served text."""
    described = agent_prompts.render(
        "character_sheets_from_description",
        {"description": "a small red barrel", "movements": 12345},
    )
    assert described is not None
    _description, messages = described
    text = messages[0]["content"]["text"]
    assert "12345" not in text
    assert "movements omitted" in text
