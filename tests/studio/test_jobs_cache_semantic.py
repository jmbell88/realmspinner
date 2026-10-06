"""Library search by meaning: the filter predicate, the cache and the store together.

The embedder is ``tests/_embed_fakes``. The claim every test here defends is the
one the store's docstring makes: a row found for what it is *about* may skip the
free-words clause of ``Filters.matches`` and nothing else, so it can never appear
where the bar would have refused it for any other reason.
"""

from __future__ import annotations

import time
from typing import Any

import _embed_fakes as fakes
import pytest

from realmspinner.service import library_index as li
from realmspinner.studio import jobs_cache as cache_mod
from realmspinner.studio.state import Filters
from realmspinner.studio.tasks import TaskRunner


def _row(**over: Any) -> dict[str, Any]:
    row = {
        "id": "sword1",
        "kind": "text",
        "status": "done",
        "stage": "model",
        "name": "Iron sword",
        "prompt": "a rusty iron longsword",
        "tags": "",
        "params": {},
        "created_at": 1000.0,
        "favorite": 0,
        "deleted_at": None,
        "sweep_id": None,
        "candidate_group": None,
    }
    row.update(over)
    return row


# --- the predicate -------------------------------------------------------------------


def test_a_semantic_hit_skips_the_free_words_clause_and_only_that_one():
    job = _row()
    bar = Filters(text="blade")
    assert bar.matches(job) is False  # shares no substring
    assert bar.matches(job, frozenset({"sword1"})) is True
    assert bar.matches(job, frozenset({"another"})) is False
    assert bar.matches(job, frozenset()) is False
    assert bar.matches(job, None) is False


@pytest.mark.parametrize(
    "bar,over",
    [
        (Filters(text="blade", status="error"), {}),
        (Filters(text="blade", kind="music"), {}),
        (Filters(text="blade", favorites_only=True), {}),
        (Filters(text="blade", trash=True), {}),
        (Filters(text="blade"), {"deleted_at": 5.0}),
        (Filters(text="blade"), {"sweep_id": "s1"}),
        (Filters(text="blade"), {"candidate_group": "g1"}),
        (Filters(text="blade", usable_only=True), {}),
        (Filters(text="blade tag:wood"), {"tags": "stone"}),
        (Filters(text="blade name:dragon"), {}),
        (Filters(text="blade status:error"), {}),
        (Filters(text="blade id:zzz"), {}),
    ],
)
def test_a_semantic_hit_never_gets_past_any_other_clause(bar, over):
    job = _row(**over)
    assert bar.matches(job, frozenset({"sword1"})) is False


def test_a_semantic_hit_still_has_to_satisfy_a_field_term_it_does_match():
    assert Filters(text="blade tag:metal").matches(
        _row(tags="metal"), frozenset({"sword1"})
    ) is True
    assert Filters(text="blade name:sword").matches(_row(), frozenset({"sword1"})) is True


def test_failures_counts_the_same_rows_with_or_without_the_set():
    bar = Filters(text="blade")
    jobs = [_row(), _row(id="e1", status="error", name="blade one")]
    assert bar.failures(jobs) == 1
    assert bar.failures(jobs, frozenset({"sword1"})) == 1


# --- the cache, end to end ---------------------------------------------------------------


def _done(svc, prompt, name="", **kw):
    job_id = svc.store.create("text", prompt, kw.pop("params", {}), status="done", **kw)
    if name:
        svc.store.set_meta(job_id, name=name)
    return job_id


@pytest.fixture
def lib(svc, monkeypatch):
    """24 plain rows, then a sword, a chest and a dragon, all indexed. The sword
    is the *oldest* row so a one-row window cannot hold it."""
    fake = fakes.install(monkeypatch, svc)
    sword = _done(svc, "a rusty iron longsword", name="Iron sword")
    time.sleep(0.002)
    for i in range(24):
        _done(svc, f"plain filler text {i}", name=f"filler {i}")
    chest = _done(svc, "an old wooden chest bound in brass", name="Treasure chest")
    dragon = _done(svc, "a red fire dragon", name="Dragon")
    assert li.index_pending(svc, 100).indexed == 27
    fake.calls.clear()
    return svc, fake, {"sword": sword, "chest": chest, "dragon": dragon}


def _search_and_adopt(cache: cache_mod.JobsCache, filters: Filters) -> list[str]:
    """Run the widen the way the app does -- through a real ``TaskRunner``, the
    result adopted on this (the 'frame') thread -- and return the visible ids."""
    runner = TaskRunner(workers=1)
    try:
        cache.request_widen(filters, runner)
        done = fakes.wait_done(runner)
        assert done and done[0].key == cache_mod.SEARCH_KEY and done[0].ok, done
        cache.adopt_widen(done[0].result)
    finally:
        runner.shutdown(wait=True)
    return [j["id"] for j in cache.visible(filters)]


def test_a_job_is_found_by_what_it_is_about_though_it_shares_no_word_with_the_query(lib):
    svc, fake, ids = lib
    cache = cache_mod.JobsCache(svc)
    cache.tick()
    bar = Filters(text="blade")
    assert cache.visible(bar) == []  # the substring bar finds nothing, as before
    assert _search_and_adopt(cache, bar) == [ids["sword"]]
    assert [c["texts"] for c in fake.query_calls] == [["blade"]]


def test_a_meaning_hit_outside_the_loaded_window_is_fetched_and_shown(lib):
    svc, _fake, ids = lib
    cache = cache_mod.JobsCache(svc, limit=3)
    cache.tick()
    assert ids["sword"] not in cache.by_id
    assert _search_and_adopt(cache, Filters(text="blade")) == [ids["sword"]]
    assert ids["sword"] in cache.by_id


def test_a_search_still_runs_when_the_window_already_holds_the_whole_library(lib):
    """The window-holds-everything shortcut skips a widen that has nothing
    outside the window to fetch; a meaning search has something *inside* it
    (rows the substring clause rejects), so the shortcut must not apply."""
    svc, _fake, ids = lib
    cache = cache_mod.JobsCache(svc)
    cache.tick()
    assert cache.total and len(cache.jobs) >= cache.total
    assert _search_and_adopt(cache, Filters(text="blade")) == [ids["sword"]]


def test_a_trashed_job_never_appears_by_meaning(lib):
    svc, _fake, ids = lib
    svc.store.set_deleted_if_not_running(ids["sword"], 9.0)
    cache = cache_mod.JobsCache(svc)
    cache.tick()
    assert _search_and_adopt(cache, Filters(text="blade")) == []
    # The trash view is the one place it is found.
    assert _search_and_adopt(cache, Filters(text="blade", trash=True)) == [ids["sword"]]


def test_a_job_the_other_filters_exclude_never_appears_by_meaning(lib):
    svc, _fake, ids = lib
    cache = cache_mod.JobsCache(svc)
    cache.tick()
    assert _search_and_adopt(cache, Filters(text="blade", favorites_only=True)) == []
    assert _search_and_adopt(cache, Filters(text="blade", status="error")) == []
    assert _search_and_adopt(cache, Filters(text="blade", kind="music")) == []
    svc.store.set_meta(ids["sword"], favorite=True)
    cache.invalidate()
    cache.tick()
    assert _search_and_adopt(cache, Filters(text="blade", favorites_only=True)) == [
        ids["sword"]
    ]


def test_a_tag_term_alone_is_not_searched_by_meaning(lib):
    svc, fake, ids = lib
    svc.store.set_meta(ids["sword"], tags="metal")
    cache = cache_mod.JobsCache(svc, limit=3)
    cache.tick()
    got = _search_and_adopt(cache, Filters(text="tag:metal"))
    assert got == [ids["sword"]]  # found exactly as before: by the tag column
    assert fake.calls == []  # and nothing was embedded for the field part
    assert cache.semantic_ids(Filters(text="tag:metal")) == frozenset()


def test_free_words_beside_a_tag_are_searched_by_meaning_and_the_tag_still_narrows(lib):
    svc, fake, ids = lib
    svc.store.set_meta(ids["sword"], tags="metal")
    svc.store.set_meta(ids["chest"], tags="wood")
    cache = cache_mod.JobsCache(svc)
    cache.tick()
    assert _search_and_adopt(cache, Filters(text="tag:metal blade")) == [ids["sword"]]
    # Only the free word went to the embedder -- never the field term.
    assert [c["texts"] for c in fake.query_calls] == [["blade"]]
    assert _search_and_adopt(cache, Filters(text="tag:wood blade")) == []


def test_a_one_character_search_is_never_embedded(lib):
    svc, fake, _ids = lib
    cache = cache_mod.JobsCache(svc, limit=3)  # rows outside the window: LIKE widens
    cache.tick()
    _search_and_adopt(cache, Filters(text="b"))
    assert fake.calls == []


def test_a_set_found_for_one_query_is_not_applied_to_another(lib):
    svc, _fake, ids = lib
    cache = cache_mod.JobsCache(svc)
    cache.tick()
    _search_and_adopt(cache, Filters(text="blade"))
    assert cache.semantic_ids(Filters(text="blade")) == {ids["sword"]}
    assert cache.semantic_ids(Filters(text="something else")) == frozenset()
    assert cache.visible(Filters(text="something else")) == []


def test_clearing_the_box_drops_the_meaning_set(lib):
    svc, _fake, _ids = lib
    cache = cache_mod.JobsCache(svc)
    cache.tick()
    _search_and_adopt(cache, Filters(text="blade"))
    assert cache.semantic_ids(Filters(text="blade"))
    cache.request_widen(Filters(), TaskRunner(workers=1))
    assert cache._semantic == ("", frozenset())


def test_the_same_words_are_embedded_once_across_list_refreshes(lib):
    svc, fake, _ids = lib
    cache = cache_mod.JobsCache(svc)
    cache.tick()
    bar = Filters(text="blade")
    for _ in range(3):
        _search_and_adopt(cache, bar)
        cache._generation += 1  # a list refresh: the next frame searches again
    assert len(fake.query_calls) == 1


def test_the_search_task_returns_the_fused_order_with_the_both_ways_row_first(svc, monkeypatch):
    fakes.install(monkeypatch, svc)
    for i in range(24):
        _done(svc, f"plain filler text {i}")
    both = _done(svc, "a longsword, a sword of iron", name="Both")
    time.sleep(0.002)
    text_only = [_done(svc, f"swordfish number {i}") for i in range(3)]
    li.index_pending(svc, 100)
    cache = cache_mod.JobsCache(svc)
    reading = cache._search("sword", (), (), None, None, False, frozenset(), "sword")
    assert reading["ids"][0] == both
    assert set(reading["ids"]) >= {both, *text_only}
    assert reading["semantic"] == [both]
    assert reading["semantic_text"] == "sword"
    plain = svc.store.search_ids("sword", limit=cache_mod.SEARCH_LIMIT)
    assert plain[0] != both


# --- without meaning --------------------------------------------------------------------------


def test_with_the_row_absent_a_search_is_byte_for_byte_the_substring_search(svc, monkeypatch):
    fake = fakes.install(monkeypatch, svc)
    monkeypatch.setattr(li, "installed", lambda _svc: False)
    for i in range(5):
        _done(svc, f"a barrel number {i}", name=f"barrel {i}")
    cache = cache_mod.JobsCache(svc, limit=2)  # rows outside the window to widen for
    cache.tick()
    runner = TaskRunner(workers=1)
    try:
        cache.request_widen(Filters(text="barrel"), runner)
        done = fakes.wait_done(runner)
    finally:
        runner.shutdown(wait=True)
    reading = done[0].result
    assert reading["ids"] == svc.store.search_ids("barrel", limit=cache_mod.SEARCH_LIMIT)
    assert set(reading) == {"ids", "rows"}  # no "semantic" keys at all
    assert fake.calls == []
    assert cache.visible(Filters(text="blade")) == []


def test_an_embedder_that_is_down_leaves_the_substring_answer_untouched(lib):
    svc, fake, ids = lib
    fake.fail = True
    cache = cache_mod.JobsCache(svc)
    cache.tick()
    reading = cache._search("chest", (), (), None, None, False, frozenset(), "chest")
    assert reading["ids"] == svc.store.search_ids("chest", limit=cache_mod.SEARCH_LIMIT)
    assert "semantic" not in reading
    assert ids["chest"] in reading["ids"]


def test_a_cold_embedder_does_not_delay_the_substring_answer(lib, monkeypatch):
    svc, fake, _ids = lib
    fake.server.running = False

    class Loop:
        def is_running(self):
            return True

    svc.loop = Loop()
    warmed = []
    monkeypatch.setattr(li, "_warm", lambda s, server: warmed.append(server))
    cache = cache_mod.JobsCache(svc)
    reading = cache._search("chest", (), (), None, None, False, frozenset(), "chest")
    assert "semantic" not in reading
    assert warmed == [fake.server]
    assert fake.calls == []


def test_the_meaning_path_never_runs_on_the_frame_thread(lib):
    """``request_widen`` only submits; the embed happens in the task."""
    svc, fake, _ids = lib
    cache = cache_mod.JobsCache(svc)
    cache.tick()

    class Runner:
        submitted: list = []

        def submit(self, key, fn, *args):
            self.submitted.append(key)
            return True

    runner = Runner()
    assert cache.request_widen(Filters(text="blade"), runner) is True
    assert runner.submitted == [cache_mod.SEARCH_KEY]
    assert fake.calls == []  # nothing embedded yet: the task has not run
