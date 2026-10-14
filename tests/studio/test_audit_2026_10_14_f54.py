"""The 2026-10-14 audit, fixer F54: agent host and resources, asset open, the
slider gesture, the modal slot, save filters, the focus ring, the embed
indexer, form coercion and the stale Poser prose.

Each test's name is the claim it pins, and each one fails against the code the
audit read.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _embed_fakes as fakes

import realmspinner
from realmspinner.core.undo import Edit, UndoStack
from realmspinner.familiar.embed_client import EmbedUnavailable
from realmspinner.service import library_index as li
from realmspinner.studio import (
    agent_host,
    agent_resources,
    asset_open,
    controls,
    dialogs,
    embed_indexer,
    focus,
    formvalues,
    widgets,
)
from realmspinner.studio.state import AppState

from .test_agent_host import _bare_host
from .test_audit_2026_10_03_medium_agents_pipelines import _time_out_then_finish

SRC = Path(realmspinner.__file__).resolve().parent

# --- agents-05 (agents-host-02) ------------------------------------------------


def _refuse_constant(token: str) -> None:
    raise AssertionError(f"a bare {token} token went out on the wire")


def test_result_body_never_emits_a_bare_nan_or_infinity_token() -> None:
    result = {
        "content": [{"type": "text", "text": "ok"}],
        "structuredContent": {"a": float("nan"), "b": [1.0, float("inf")], "c": float("-inf")},
    }
    body = agent_host._result_body(result)
    # ``parse_constant`` is what a strict parser does with NaN/Infinity.
    parsed = json.loads(body, parse_constant=_refuse_constant)
    assert parsed["content"][0]["text"] == "ok"
    assert parsed["structuredContent"]["a"] == "nan"
    assert parsed["structuredContent"]["b"] == [1.0, "inf"]
    assert parsed["structuredContent"]["c"] == "-inf"


# --- agents-18 (agents-host-04) ------------------------------------------------


def test_hello_is_answered_even_when_the_frame_thread_has_not_pumped_yet() -> None:
    """The host's queue is never pumped here, as at start-up before the first
    frame. The tab-open used to be waited on before the first frame was read,
    for up to CALL_TIMEOUT, against the bridge's SETUP_TIMEOUT of ten."""
    from realmspinner.mcp import rpc

    host = _bare_host()
    answered = threading.Event()

    class _Conn:
        def __init__(self) -> None:
            self._frame = rpc.encode_request("hello", versions=[1], bridge_version="test")
            self._delivered = False

        def recv_bytes(self, maxlength=None):  # noqa: ARG002
            if not self._delivered:
                self._delivered = True
                return self._frame
            assert answered.wait(5.0)
            raise EOFError()

        def send_bytes(self, data: bytes) -> None:  # noqa: ARG002
            answered.set()

        def close(self) -> None:
            pass

    thread = threading.Thread(target=host._serve, args=(_Conn(),), daemon=True)
    thread.start()
    assert answered.wait(3.0), "hello was not answered while the frame thread had not pumped"
    thread.join(timeout=5.0)
    assert not thread.is_alive()


# --- agents-20 (agents-host-06) ------------------------------------------------


def test_a_replayed_reference_list_is_re_run_not_remembered(monkeypatch) -> None:
    host, calls, session, runs, finish = _time_out_then_finish(
        monkeypatch, "clay_reference_list", {}, ["old list", "new list"]
    )
    try:
        retry = host._call(session, calls, "clay_reference_list", {})
    finally:
        finish()
    assert runs["n"] == 2
    assert retry["content"][0]["text"] == "new list"


# --- agents-14 (agents-clay-08) ------------------------------------------------


def test_the_operations_resource_marks_a_param_that_refuses_below_its_floor() -> None:
    from realmspinner.studio.modes.clay import ops as clay_ops

    mime, raw = agent_resources.read_static(agent_resources.OPERATIONS_URI)
    assert mime
    table = json.loads(raw)
    refusing = 0
    for op in clay_ops.OPS:
        rows = {row["name"]: row for row in table[op.name]["params"]}
        for param in op.params:
            if param.boolean or param.choices:
                continue
            row = rows[param.name]
            assert row.get("refuse_below", False) is bool(param.refuse_below), (op.name, param.name)
            assert row.get("integer", False) is bool(param.integer), (op.name, param.name)
            refusing += bool(param.refuse_below)
    assert refusing, "no op refuses below a floor any more; this test proves nothing"


# --- shell-49 (shell-documents-11) ---------------------------------------------


def test_a_follow_up_whose_source_is_outside_the_loaded_window_still_lands_somewhere_visible() -> (
    None
):
    state = AppState()
    state.mode = "create"
    row = {
        "id": "RETEX",
        "kind": "retexture",
        "stage": "model",
        "status": "done",
        "files": [],
        "params": {"source_job": "OLD-SOURCE"},
    }
    ctx = SimpleNamespace(
        state=state,
        cache=SimpleNamespace(get=lambda job_id: row if job_id == "RETEX" else None),
        toast=lambda *a, **k: None,
    )
    asset_open.open_asset(ctx, row)
    assert state.mode == "library"
    assert state.selected == "OLD-SOURCE"
    assert state.library_scroll_to == "OLD-SOURCE"


# --- shell-69 (shell-widgets-03) -----------------------------------------------


class _Step(Edit):
    cost = 10

    def undo(self, doc) -> None:  # noqa: ARG002
        pass

    def redo(self, doc) -> None:  # noqa: ARG002
        pass


def test_an_open_slider_gesture_closes_when_no_item_is_active_any_more(monkeypatch) -> None:
    item = SimpleNamespace(activated=False, deactivated=False, active_id=7)
    stub = SimpleNamespace(
        is_item_activated=lambda: item.activated,
        is_item_deactivated=lambda: item.deactivated,
        is_any_item_active=lambda: item.active_id != 0,
    )
    monkeypatch.setattr(controls, "imgui", stub)
    monkeypatch.setattr(controls, "_gesture", None)
    history = UndoStack()
    item.activated = True
    controls.fold_undo(history)
    item.activated = False
    history.push(_Step())
    # The slider is no longer drawn (a tab switched, a pane collapsed): imgui has
    # no active item, and the next control drawn anywhere is the first to know.
    item.active_id = 0
    for _ in range(20):
        history.push(_Step())
    controls._finish_item()
    assert controls._gesture is None
    assert history._open_gestures == 0


def test_a_drag_in_flight_keeps_its_gesture_open_while_another_control_draws(monkeypatch) -> None:
    item = SimpleNamespace(activated=False, deactivated=False, active_id=7)
    stub = SimpleNamespace(
        is_item_activated=lambda: item.activated,
        is_item_deactivated=lambda: item.deactivated,
        is_any_item_active=lambda: item.active_id != 0,
    )
    monkeypatch.setattr(controls, "imgui", stub)
    monkeypatch.setattr(controls, "_gesture", None)
    history = UndoStack()
    item.activated = True
    controls.fold_undo(history)
    item.activated = False
    controls._finish_item()
    assert controls._gesture is not None
    controls.close_gesture()


# --- shell-13 (shell-chrome-03) ------------------------------------------------


def test_a_confirm_and_a_prompt_pending_together_show_exactly_one_modal_each_frame(
    monkeypatch,
) -> None:
    opened_per_frame: dict[int, list[str]] = {}
    frame = {"n": 0}
    fake = mock.MagicMock()
    fake.get_frame_count.side_effect = lambda: frame["n"]
    fake.open_popup.side_effect = lambda name: opened_per_frame.setdefault(frame["n"], []).append(
        name
    )
    fake.begin_popup_modal.return_value = (False, False)
    fake.is_any_item_active.return_value = False
    monkeypatch.setattr(dialogs, "imgui", fake)
    monkeypatch.setattr(widgets, "popover_enter", lambda *a, **k: (1.0, 0.0))
    monkeypatch.setattr(widgets, "frosted", lambda: False)
    monkeypatch.setattr(widgets, "push_surface_rounding", lambda: 0.0)
    monkeypatch.setattr(widgets, "pop_surface_rounding", lambda: None)
    monkeypatch.setattr(widgets, "modal_bounds", lambda *a, **k: None)
    monkeypatch.setattr(dialogs, "_confirm_drew_on", None, raising=False)

    confirms, prompts = dialogs.ConfirmQueue(), dialogs.PromptQueue()
    confirms.ask(dialogs.Confirm(title="Quit?", message="m"))
    prompts.ask(dialogs.Prompt(title="Rename", label="Name"))
    for n in range(1, 5):
        frame["n"] = n
        confirms.draw()
        prompts.draw()
    for n in range(1, 5):
        assert opened_per_frame.get(n) == ["Quit?"], (n, opened_per_frame)


# --- shell-15 (shell-chrome-05) ------------------------------------------------


def test_filters_for_a_bare_extension_and_a_nine_patch_name_find_their_filters() -> None:
    assert dialogs.filters_for(".wav") is dialogs.ARTIFACT_FILTERS[".wav"]
    assert dialogs.filters_for("hero.9.png") is dialogs.ARTIFACT_FILTERS[".9.png"]
    assert dialogs.filters_for("hero.png") is dialogs.ARTIFACT_FILTERS[".png"]
    assert dialogs.filters_for("Song.WAV") is dialogs.ARTIFACT_FILTERS[".wav"]
    assert dialogs.filters_for("wav") == ["All files", "*"]
    assert dialogs.filters_for("noextension") == ["All files", "*"]


# --- shell-boot-01 (shell-08) --------------------------------------------------


def test_one_tab_press_moves_exactly_one_focus_ring_in_muse(monkeypatch) -> None:
    from realmspinner.studio.modes.muse.ui import brief as muse_brief
    from realmspinner.studio.modes.muse.ui.panes import recipe as muse_recipe

    stub = SimpleNamespace(
        is_popup_open=lambda *a: False,
        is_key_pressed=lambda key: True,
        is_key_down=lambda key: False,
        Key=SimpleNamespace(tab=1, left_shift=2, right_shift=3),
        PopupFlags_=SimpleNamespace(any_popup=SimpleNamespace(value=0)),
        get_current_context=lambda: object(),
        get_frame_count=lambda: 5,
    )
    monkeypatch.setattr(focus, "imgui", stub)
    state = AppState()
    panes = (muse_brief.FOCUS_PANE, muse_recipe.FOCUS_PANE)
    for pane in panes:
        state.focus_order[pane] = ["a", "b", "c"]
    for pane in panes:
        focus.pump(state, pane)
    assert len(state.focus_key) == 1, state.focus_key
    assert next(iter(state.focus_key.values())) == "a"


def test_two_panes_on_one_ring_key_record_one_order_per_frame(monkeypatch) -> None:
    from realmspinner.studio.modes.muse.ui import brief as muse_brief
    from realmspinner.studio.modes.muse.ui.panes import recipe as muse_recipe

    frame = {"n": 9}
    stub = SimpleNamespace(get_current_context=lambda: object(), get_frame_count=lambda: frame["n"])
    monkeypatch.setattr(focus, "imgui", stub)
    state = AppState()
    for pane, key in ((muse_brief.FOCUS_PANE, "prompt"), (muse_recipe.FOCUS_PANE, "seed")):
        focus.begin(state, pane)
        with focus.item(state, pane, key):
            pass
    assert state.focus_order[muse_brief.FOCUS_PANE] == ["prompt", "seed"]
    frame["n"] = 10
    focus.begin(state, muse_brief.FOCUS_PANE)
    assert state.focus_order[muse_brief.FOCUS_PANE] == []


# --- familiar-19 (familiar-host-04) --------------------------------------------


def test_a_batch_that_fails_before_pump_returns_keeps_its_ids_pending(svc, monkeypatch) -> None:
    fakes.install(monkeypatch, svc)

    def down(_svc, _ids):
        raise EmbedUnavailable("the embedder is in back-off")

    monkeypatch.setattr(li, "index_jobs", down)

    class _InlineRunner:
        """``TaskRunner.submit`` for a task that fails fast: the body has run
        (and re-queued its ids) before ``submit`` returns."""

        def submit(self, key, fn, *args, **kwargs):  # noqa: ARG002
            fn(*args)
            return True

    indexer = embed_indexer.LibraryIndexer(svc, clock=lambda: 1000.0)
    indexer.note("job-1")
    indexer.pump(_InlineRunner())
    assert indexer.pending == ("job-1",)


# --- shell-71 (shell-widgets-05) -----------------------------------------------


def test_coerce_form_value_refuses_non_finite_numbers_instead_of_raising_or_storing_them() -> None:
    for bad in (float("inf"), float("-inf"), float("nan")):
        assert formvalues.coerce_form_value(0.5, bad) == 0.5
        assert formvalues.coerce_form_value(3, bad) == 3
    assert formvalues.coerce_form_value(0.5, json.loads("Infinity")) == 0.5
    assert formvalues.coerce_form_value(0.5, "0.75") == 0.75
    assert formvalues.coerce_form_value(3, 4.0) == 4


# --- shell-51 (shell-documents-14) ---------------------------------------------

#: The files the finding cites, and the one Poser-naming file the audit named
#: outside the shell: every one is in this fixer's hands.
_POSER_SITES = (
    "studio/docmodes.py",
    "studio/artifacts.py",
    "studio/asset_open.py",
    "studio/mode_manifest.py",
    "studio/journal.py",
    "studio/modes/home/ui/panes/landing.py",
)


def test_no_source_comment_names_the_removed_poser_mode() -> None:
    offenders = []
    for rel in _POSER_SITES:
        text = (SRC / rel).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if re.search(r"\bPoser\b", line):
                offenders.append(f"{rel}:{number}: {line.strip()}")
    assert offenders == []
    docmodes = (SRC / "studio/docmodes.py").read_text(encoding="utf-8")
    assert "the two pose editors" not in docmodes
