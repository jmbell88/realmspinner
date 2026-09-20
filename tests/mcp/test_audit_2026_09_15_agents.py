"""Regression tests for the 2026-09-15 audit's agents findings.

Four unrelated defects, one file because one fixer owned all four:
agents-02 (a degraded ``animated_glb`` export is reported clean to an
agent), agents-03 (a locked-modern MCP connection can be downgraded to
legacy by a stray ``initialize``), agents-05 (a "never reached Realmspinner"
refusal carries a misleading ``read_scene`` recovery) and tour-01
(finishing a tour by running off its last step leaves a stale card hole
for the next one).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from realmspinner.mcp import bridge
from realmspinner.mcp import protocol as p
from realmspinner.studio import agent_character as ac

# --- agents-02: character_export animated_glb drops `degraded` -------------


def test_character_export_reports_a_degraded_mesh_to_the_agent(monkeypatch, svc, tmp_path) -> None:
    """``_normalise_export``'s ``"copied" in result`` branch (the
    ``animated_glb``/``export_to_folder`` shape) used to build its payload
    from only ``dir`` and ``copied``, dropping ``degraded`` on the floor --
    so an agent reading ``character_export`` for a mesh whose normalize step
    actually failed was told the export was clean.

    Fails against the unfixed code with:
        AssertionError: 'degraded' not in {'format': 'animated_glb', 'dir': ..., 'copied': 1}
    """
    from studio.test_agent_character import _real_sheet

    from realmspinner.service import export as svc_export

    svc.config.export_dir = tmp_path
    job_id, _sheet_id, _w, _h = _real_sheet(svc, job_name="Ranger")

    def fake_run_character_export(svc_arg, key, jid, sid=None, *, stem=None):
        return {"copied": 1, "dir": str(tmp_path), "degraded": ["normalize: non-manifold"]}

    monkeypatch.setattr(svc_export, "run_character_export", fake_run_character_export)

    result = ac.call(
        svc, ac.Session(), "character_export", {"job_id": job_id, "format": "animated_glb"}
    )
    assert result["isError"] is False, result
    payload = json.loads(next(c["text"] for c in result["content"] if c["type"] == "text"))
    assert "degraded" in payload, payload
    assert payload["degraded"] == ["normalize: non-manifold"]


# --- agents-03: `initialize` after `discover` can downgrade a locked era ----


def _catalogue() -> dict[str, Any]:
    return {
        "hash": "h1",
        "tools": [
            {"name": "clay_scene", "title": "Scene", "description": "d", "inputSchema": {}}
        ],
        "instructions": "Clay measures in metres.",
        "server": {"name": "realmspinner", "version": "1.2.3"},
    }


def _ok_call_tool(name, args):
    return json.dumps(p.ok(p.text(f"ran {name}"))).encode("utf-8")


def test_initialize_does_not_downgrade_an_already_modern_locked_era() -> None:
    """``BridgeEra``'s own docstring promises the era is "locked for the
    connection's life" once decided. The ``server/discover`` branch guards
    its own write with ``if state.era is None`` -- ``initialize`` set
    ``state.era = "legacy"`` unconditionally, so a connection already locked
    to "modern" was silently downgraded by a stray/late ``initialize``.

    Fails against the unfixed code with:
        AssertionError: assert 'legacy' == 'modern'
    """
    state = p.BridgeEra()
    state.era = "modern"
    raw = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    ).encode("utf-8")
    reply_bytes = p.bridge_dispatch(raw, state, catalogue=_catalogue(), call_tool=_ok_call_tool)
    reply = json.loads(reply_bytes)
    assert state.era == "modern", state.era
    assert reply["error"]["code"] == -32600


# --- agents-05: "not accepting" carries a misleading `read_scene` recovery -


def test_not_accepting_message_carries_no_misleading_read_scene_recovery(tmp_path) -> None:
    """This call never reached Realmspinner at all -- there is nothing here for
    "read the scene again" to be recovering from -- and ``agent_host``'s own
    refusal for the same "nothing ran" state (``rpc.fail("Realmspinner's agent
    server was switched off.")``) carries no recovery at all. ``bridge.py``'s
    ``call_tool`` used to attach ``recovery="read_scene"`` anyway.

    Fails against the unfixed code with:
        AssertionError: assert 'read_scene' != 'read_scene'
    """
    session = bridge._Session.from_snapshot(Path(tmp_path), _catalogue())
    reply_bytes = session.call_tool("clay_scene", {})
    result = json.loads(reply_bytes)
    assert result.get("structuredContent", {}).get("recovery") != "read_scene"


# --- tour-01: finishing via advance() leaves a stale card hole --------------


def test_finishing_a_tour_by_running_off_the_end_clears_the_stale_card_rect() -> None:
    """``advance()`` completing the tour by running off its last step called
    ``state.complete()`` directly rather than going through ``stop``'s
    cleanup, so ``_card_rect``/``_card_focused`` survived a finished tour --
    the next tour's first frame would veil around a hole the previous one
    left behind.

    Fails against the unfixed code with:
        AssertionError: assert (1.0, 2.0, 3.0, 4.0) is None
    """
    from types import SimpleNamespace

    from realmspinner.studio.panes import tour as tour_pane
    from realmspinner.studio.state import AppState
    from realmspinner.studio.tour import TOURS

    real_tour = TOURS[0]
    ctx = SimpleNamespace(
        state=AppState(),
        settings=SimpleNamespace(get=lambda *_a, **_k: None, set=lambda *_a, **_k: None),
        toast=lambda *_a, **_k: None,
    )
    ctx.state.tour.key = real_tour.key
    ctx.state.tour.index = len(real_tour.steps) - 1

    tour_pane._card_rect[0] = (1.0, 2.0, 3.0, 4.0)
    tour_pane._card_focused[0] = True
    try:
        tour_pane.advance(ctx, delta=1)

        assert not ctx.state.tour.running
        assert tour_pane._card_rect[0] is None
        assert tour_pane._card_focused[0] is False
    finally:
        # Leave the module's shared, process-wide state clean for whatever
        # test runs next in this worker.
        tour_pane._card_rect[0] = None
        tour_pane._card_focused[0] = False
