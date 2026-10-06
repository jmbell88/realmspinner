"""The Manual pane's optional Semantic search (``studio/manual/semantic.py``).

The decisions live in ``semantic.view`` (no imgui), so most tests drive it with
a fake ctx whose ``submit`` *captures* the task instead of running it -- which
is also how "the embed client is never called on the frame thread" is proved:
nothing runs until the test runs it, on a thread of its own. ``render._draw_toc``
is exercised once with imgui, controls and widgets replaced, to pin what is
actually drawn.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from realmspinner.service import familiar as svc_familiar
from realmspinner.service import familiar_manual
from realmspinner.studio import state as studio_state
from realmspinner.studio.manual import render, semantic

HITS = familiar_manual.ManualHits(
    "ready",
    (
        ("05-drawing", "undo", "05 Drawing › Undo"),
        ("00-index", None, "00 Index"),
    ),
)


class _Ctx:
    """Just enough of ``AppCtx``: ``svc``, ``busy`` and a capturing ``submit``."""

    def __init__(self) -> None:
        self.svc = SimpleNamespace()
        self.queued: list[tuple[str, object, tuple]] = []
        self.running: set[str] = set()

    def busy(self, key: str) -> bool:
        return key in self.running

    def submit(self, key, fn, *args, tag=None, **kw) -> bool:
        if key in self.running:
            return False
        self.queued.append((key, fn, args))
        self.running.add(key)
        return True

    def run_queued(self) -> None:
        """What the pool does: run each captured task off the calling thread."""
        queued, self.queued = self.queued, []
        for key, fn, args in queued:
            worker = threading.Thread(target=fn, args=args)
            worker.start()
            worker.join()
            self.running.discard(key)


@pytest.fixture(autouse=True)
def _fresh():
    semantic.reset()
    yield
    semantic.reset()


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setattr(svc_familiar, "manual_semantic_available", lambda svc: True)


@pytest.fixture
def ranks(monkeypatch):
    """``manual_sections`` replaced; records each call's thread and query."""
    calls: list[tuple[threading.Thread, str]] = []
    answer = {"hits": HITS}

    def fake(svc, query, limit=12):
        calls.append((threading.current_thread(), query))
        hits = answer["hits"]
        if isinstance(hits, Exception):
            raise hits
        return hits

    monkeypatch.setattr(svc_familiar, "manual_sections", fake)
    fake.calls = calls
    fake.answer = answer
    return fake


def _manual(search: str = "", semantic_on: bool = False):
    ms = studio_state.ManualState()
    ms.search = search
    ms.semantic = semantic_on
    return ms


def _settle(ctx, ms, start: float = 100.0):
    """Frames until the debounce has passed and the task has run; -> final view."""
    needle = ms.search.strip().lower()
    semantic.view(ctx, ms, needle, now=start)
    semantic.view(ctx, ms, needle, now=start + semantic.DEBOUNCE + 0.05)
    ctx.run_queued()
    return semantic.view(ctx, ms, needle, now=start + semantic.DEBOUNCE + 0.1)


# --- the toggle ------------------------------------------------------------


def test_the_semantic_toggle_is_not_offered_without_the_retrieval_row(monkeypatch, ranks):
    monkeypatch.setattr(svc_familiar, "manual_semantic_available", lambda svc: False)
    ctx = _Ctx()
    ms = _manual("undo a mistake", semantic_on=True)

    plan = semantic.view(ctx, ms, "undo a mistake", now=1.0)

    assert not plan.show_toggle and not plan.rows and not plan.note
    assert ctx.queued == [] and ranks.calls == []


def test_the_semantic_toggle_is_off_by_default_and_then_does_nothing(installed, ranks):
    ctx = _Ctx()
    ms = _manual("undo a mistake")

    assert studio_state.ManualState().semantic is False
    for t in (1.0, 5.0):
        plan = semantic.view(ctx, ms, "undo a mistake", now=t)
        assert plan.show_toggle and not plan.rows
    assert ctx.queued == []


def test_the_toggle_probe_is_remembered_between_frames(monkeypatch):
    probes: list[int] = []
    monkeypatch.setattr(
        svc_familiar, "manual_semantic_available", lambda svc: probes.append(1) or True
    )
    ctx = _Ctx()
    for t in (1.0, 1.1, 1.2, 1.3):
        semantic.available(ctx, now=t)
    assert len(probes) == 1, "the frame loop must not stat the models directory per frame"


# --- the results -----------------------------------------------------------


def test_with_semantic_on_and_a_ready_matrix_the_result_list_is_the_semantic_one(installed, ranks):
    ctx = _Ctx()
    ms = _manual("undo a mistake", semantic_on=True)

    plan = _settle(ctx, ms)

    assert [(r.chapter, r.anchor, r.label) for r in plan.rows] == list(HITS.sections)
    assert ranks.calls[0][1] == "undo a mistake"


def test_the_query_is_embedded_on_a_task_thread_never_on_the_frame_thread(installed, ranks):
    ctx = _Ctx()
    ms = _manual("undo a mistake", semantic_on=True)
    frame_thread = threading.current_thread()

    semantic.view(ctx, ms, "undo a mistake", now=10.0)
    semantic.view(ctx, ms, "undo a mistake", now=10.0 + semantic.DEBOUNCE + 0.05)

    assert [key for key, _fn, _args in ctx.queued] == [semantic.KEY]
    assert ranks.calls == [], "view() submitted the work; it did not do it"
    ctx.run_queued()
    assert [thread for thread, _ in ranks.calls] and all(
        thread is not frame_thread for thread, _ in ranks.calls
    )


def test_the_query_waits_for_the_typed_text_to_settle(installed, ranks):
    ctx = _Ctx()
    ms = _manual("u", semantic_on=True)
    t = 50.0
    for text in ("un", "und", "undo"):
        ms.search = text
        semantic.view(ctx, ms, text, now=t)
        t += semantic.DEBOUNCE / 4  # keeps typing before it settles
    assert ctx.queued == [], "no request per keystroke"

    semantic.view(ctx, ms, "undo", now=t + semantic.DEBOUNCE + 0.05)
    assert len(ctx.queued) == 1
    assert ctx.queued[0][2][1] == "undo"


def test_until_a_result_is_back_the_pane_keeps_the_substring_results(installed, ranks):
    ctx = _Ctx()
    ms = _manual("undo a mistake", semantic_on=True)

    semantic.view(ctx, ms, "undo a mistake", now=1.0)
    waiting = semantic.view(ctx, ms, "undo a mistake", now=2.0)  # queued, not yet run

    assert ctx.queued and not waiting.rows and not waiting.note


def test_while_the_meaning_index_is_built_one_dim_line_says_so_and_nothing_spins(
    installed, ranks
):
    ranks.answer["hits"] = familiar_manual.ManualHits("building")
    ctx = _Ctx()
    ms = _manual("undo a mistake", semantic_on=True)

    plan = _settle(ctx, ms)

    assert plan.note == semantic.BUILDING_NOTE and not plan.rows
    # It asks again later, not every frame.
    semantic.view(ctx, ms, "undo a mistake", now=102.0)
    assert ctx.queued == []
    ranks.answer["hits"] = HITS
    semantic.view(ctx, ms, "undo a mistake", now=100.0 + semantic.RETRY_BUILDING + 1.0)
    ctx.run_queued()
    done = semantic.view(ctx, ms, "undo a mistake", now=200.0)
    assert done.rows and not done.note


def test_a_failure_falls_back_to_the_substring_results_silently(installed, ranks):
    ranks.answer["hits"] = RuntimeError("the embedder fell over")
    ctx = _Ctx()
    ms = _manual("undo a mistake", semantic_on=True)

    plan = _settle(ctx, ms)

    assert plan == semantic.View(show_toggle=True)
    # And it does not retry every frame for the same text.
    for t in (110.0, 111.0, 112.0):
        semantic.view(ctx, ms, "undo a mistake", now=t)
    assert ctx.queued == []


def test_a_one_character_query_is_never_embedded(installed, ranks):
    ctx = _Ctx()
    ms = _manual("u", semantic_on=True)
    semantic.view(ctx, ms, "u", now=1.0)
    semantic.view(ctx, ms, "u", now=9.0)
    assert ctx.queued == []


# --- what is drawn ---------------------------------------------------------


class _Recorder:
    def __init__(self, ms) -> None:
        self.ms = ms
        self.checkboxes: list[str] = []
        self.selectables: list[str] = []
        self.notes: list[str] = []
        self.click: str | None = None

    def install(self, monkeypatch) -> None:
        imgui = SimpleNamespace(
            set_next_item_width=lambda *_a: None,
            indent=lambda *_a: None,
            unindent=lambda *_a: None,
        )
        controls = SimpleNamespace(
            input_text_with_hint=lambda label, hint, value: (False, value),
            checkbox=self._checkbox,
            selectable=self._selectable,
        )
        widgets = SimpleNamespace(
            muted_wrapped=self.notes.append,
            section=lambda *_a: None,
            empty_state=lambda *_a: None,
        )
        monkeypatch.setattr(render, "imgui", imgui)
        monkeypatch.setattr(render, "controls", controls)
        monkeypatch.setattr(render, "widgets", widgets)

    def _checkbox(self, label, value, **_kw):
        self.checkboxes.append(label)
        return False, value

    def _selectable(self, label, active=False, **_kw):
        self.selectables.append(label)
        return (self.click is not None and label.startswith(self.click)), active


def test_the_toggle_checkbox_is_not_drawn_at_all_without_the_row(monkeypatch, ranks):
    monkeypatch.setattr(svc_familiar, "manual_semantic_available", lambda svc: False)
    ms = _manual("undo")
    rec = _Recorder(ms)
    rec.install(monkeypatch)

    render._draw_toc(_Ctx(), ms)

    assert rec.checkboxes == []


def test_the_drawn_semantic_results_open_the_right_chapter_and_anchor(
    monkeypatch, installed, ranks
):
    ctx = _Ctx()
    ms = _manual("undo a mistake", semantic_on=True)
    rec = _Recorder(ms)
    rec.install(monkeypatch)
    _settle(ctx, ms)

    rec.click = "05 Drawing › Undo"
    render._draw_toc(ctx, ms)

    assert rec.checkboxes == ["Semantic##manual-semantic"]
    assert rec.selectables == [
        "05 Drawing › Undo##sem-05-drawing-undo",
        "00 Index##sem-00-index-None",
    ], "the ranked sections replace the chapter list"
    assert (ms.chapter, ms.anchor, ms.pending_anchor) == ("05-drawing", "undo", "undo")


def test_the_drawn_building_note_is_one_muted_line_over_the_substring_list(
    monkeypatch, installed, ranks
):
    ranks.answer["hits"] = familiar_manual.ManualHits("building")
    ctx = _Ctx()
    ms = _manual("undo", semantic_on=True)
    rec = _Recorder(ms)
    rec.install(monkeypatch)
    _settle(ctx, ms)

    render._draw_toc(ctx, ms)

    assert rec.notes == [semantic.BUILDING_NOTE]
    assert all("##sem-" not in label for label in rec.selectables)
