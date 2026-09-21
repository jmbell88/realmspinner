"""Section J/O: the query syntax, the sorts, the trash, and the window.

The pure halves live in ``state.py`` and ``jobs_cache.py``, so most of this is
ordinary Python. The trash's rules are in the service and are tested against a
real store, because the interesting ones are about what the *worker* can still
do to a row the user has thrown away.
"""

from __future__ import annotations

import inspect
import threading
import time
from typing import Any

import pytest

from realmspinner.service import jobs as svc_jobs
from realmspinner.service.errors import Conflict
from realmspinner.studio import jobs_cache as cache_mod
from realmspinner.studio.modes.library.ui.panes import full as library_full
from realmspinner.studio.modes.library.ui.panes import library
from realmspinner.studio.state import SORTS, Filters, parse_query


def job(**over: Any) -> dict[str, Any]:
    row = {
        "id": "j1",
        "kind": "text",
        "status": "done",
        "stage": "model",
        "name": "",
        "prompt": "",
        "tags": "",
        "params": {},
        "created_at": 1000.0,
        "started_at": None,
        "finished_at": None,
    }
    row.update(over)
    return row


# --- J87: the query syntax ---------------------------------------------------


def test_a_plain_query_is_still_plain_text():
    """The box has always been plain text; a query with no colon in it must
    behave character for character as it did."""
    assert parse_query("a wooden chest") == (["a", "wooden", "chest"], [])


def test_a_known_prefix_becomes_a_constraint():
    terms, fields = parse_query("tag:wood status:error rusty")
    assert terms == ["rusty"]
    assert fields == [("tag", "wood"), ("status", "error")]


def test_an_unknown_prefix_stays_free_text():
    """Silently reinterpreting a colon somebody typed on purpose is how a
    search starts returning nothing with no explanation."""
    assert parse_query("http://example") == (["http://example"], [])
    assert parse_query("foo:bar") == (["foo:bar"], [])


def test_a_prefix_with_no_value_is_free_text():
    assert parse_query("tag:") == (["tag:"], [])


def test_a_value_may_be_quoted_for_a_space():
    assert parse_query('name:"a wooden chest"') == ([], [("name", "a wooden chest")])


def test_an_unbalanced_quote_does_not_break_the_box():
    """The user is mid-typing; a traceback or an empty list would both be
    wrong answers to a half-written query."""
    assert parse_query('name:"a wooden') == ([], [("name", "a wooden")])


def test_every_word_must_match_not_any():
    filters = Filters(text="wooden chest")
    assert filters.matches(job(name="a wooden chest"))
    assert not filters.matches(job(name="a wooden barrel"))


def test_a_tag_matches_a_whole_entry_and_not_a_substring():
    """``tag:wood`` finding ``driftwood`` makes the prefix useless for the one
    thing it is for."""
    filters = Filters(text="tag:wood")
    assert filters.matches(job(tags="wood,props"))
    assert not filters.matches(job(tags="driftwood"))


@pytest.mark.parametrize(
    ("query", "row", "hit"),
    [
        ("status:error", {"status": "error"}, True),
        ("status:error", {"status": "done"}, False),
        ("kind:model", {"stage": "model"}, True),
        ("kind:model", {"stage": "reference"}, False),
        ("stage:reference", {"stage": "reference"}, True),
        ("id:j1", {}, True),
        ("id:zz", {}, False),
        ("name:chest", {"name": "A Wooden Chest"}, True),
    ],
)
def test_each_field_narrows_what_it_says_it_does(query, row, hit):
    assert Filters(text=query).matches(job(**row)) is hit


def test_a_field_term_adds_to_the_combos_rather_than_overriding_them():
    """A term that quietly overrode a visible control would be worse than a
    contradiction that correctly shows nothing."""
    filters = Filters(text="status:error", status="done")
    assert not filters.matches(job(status="error"))
    assert not filters.matches(job(status="done"))


# --- J85: the sorts ----------------------------------------------------------


def test_every_sort_key_is_offered_and_labelled():
    keys = [key for key, _label in SORTS]
    assert keys[0] == "newest"  # the persisted default
    assert len(set(keys)) == len(keys)
    assert all(label and all(ord(c) < 0x100 for c in label) for _key, label in SORTS)


def test_the_date_sort_is_the_querys_own_order():
    rows = [job(id="a"), job(id="b"), job(id="c")]
    assert Filters(sort="newest").order(rows) == rows
    assert Filters(sort="newest", descending=False).order(rows) == rows[::-1]


def test_the_name_sort_uses_the_name_the_card_shows():
    """Sorting by a column the card does not show would look like no sort at
    all -- the displayed name is the prompt when there is no title."""
    rows = [job(id="a", prompt="zebra"), job(id="b", name="apple")]
    # ``descending`` is each key's own natural order, which for a name is A to Z.
    assert [j["id"] for j in Filters(sort="name").order(rows)] == ["b", "a"]
    assert [j["id"] for j in Filters(sort="name", descending=False).order(rows)] == ["a", "b"]


def test_an_unanswerable_row_sorts_last_in_both_directions():
    """"Unknown" is not a value at one end of a scale, it is the absence of
    one -- filing the whole unmeasured backlog under "smallest" until the
    storage walk lands would be a wrong answer, not a partial one."""
    rows = [job(id="a"), job(id="b"), job(id="c")]
    sizes = {"a": 10, "b": 500}
    for descending in (True, False):
        ordered = Filters(sort="size", descending=descending).order(rows, sizes=sizes)
        assert ordered[-1]["id"] == "c"
    assert [j["id"] for j in Filters(sort="size").order(rows, sizes=sizes)][:2] == ["b", "a"]


def test_the_duration_sort_ignores_a_job_that_never_ran():
    rows = [
        job(id="slow", started_at=0.0, finished_at=100.0),
        job(id="fast", started_at=0.0, finished_at=5.0),
        job(id="queued"),
    ]
    assert [j["id"] for j in Filters(sort="duration").order(rows)] == [
        "slow",
        "fast",
        "queued",
    ]


def test_the_best_sort_keeps_its_old_bucketing():
    rows = [
        job(id="unranked"),
        job(id="good", params={"rank": {"score": 0.9}}),
        job(id="poor", params={"rank": {"score": 0.1}}),
    ]
    assert [j["id"] for j in Filters(sort="best").order(rows)] == ["good", "poor", "unranked"]


def test_every_sort_is_stable_within_a_tie():
    """Two refreshes half a second apart must not reshuffle equal rows."""
    rows = [job(id=f"j{n}", name="same") for n in range(6)]
    for key, _label in SORTS:
        ordered = Filters(sort=key).order(rows, sizes={})
        assert [j["id"] for j in ordered] == [j["id"] for j in rows], key


def test_an_unknown_sort_key_is_the_querys_order():
    """A persisted value from a build that offered a key this one does not."""
    rows = [job(id="a"), job(id="b")]
    assert Filters(sort="kremlin").order(rows) == rows


# --- A3: grade-aware library ---------------------------------------------------


def test_the_grade_sort_files_ungraded_rows_last_in_both_directions():
    """"Ungraded" is not a value at one end of the scale, the same rule
    ``size`` and ``best`` already state -- most of a workshop predates
    grading, and a plain reverse would put that whole backlog first."""
    rows = [job(id="ungraded"), job(id="high", grade=4), job(id="low", grade=-2)]
    for descending in (True, False):
        ordered = Filters(sort="grade", descending=descending).order(rows)
        assert ordered[-1]["id"] == "ungraded"
    assert [j["id"] for j in Filters(sort="grade").order(rows)][:2] == ["high", "low"]


def test_usable_only_hides_a_plus_two_and_shows_a_plus_three_through_the_one_cut(monkeypatch):
    """The cut is Review's own scale (``+3`` is "usable"), and it must be the
    live constant rather than a second spelling of it -- patching the real
    ``vectors.USABLE_GRADE`` has to move the predicate with it."""
    import realmspinner.vectors as vectors_mod

    below = job(id="below", grade=2)
    at_cut = job(id="at-cut", grade=3)
    filters = Filters(usable_only=True)
    assert not filters.matches(below)
    assert filters.matches(at_cut)

    monkeypatch.setattr(vectors_mod, "USABLE_GRADE", 4)
    assert not filters.matches(at_cut)


def _finished_mesh(svc) -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    svc.store.set_status(job_id, "done")
    return job_id


def test_a_listed_job_carries_its_latest_human_mesh_grade_and_not_an_image_label(svc):
    """``list_jobs`` attaches the mesh grade onto the row -- and only the mesh
    grade: a reference on the same job id can carry a binary image label
    whose ``grade`` column is NULL, and that must not leak onto the card as a
    mesh verdict it never received."""
    from realmspinner.service import verdicts as svc_verdicts

    graded = _finished_mesh(svc)
    svc_verdicts.record_verdict(svc, graded, grade=4)

    labelled = _finished_mesh(svc)
    svc.job_dir(labelled).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(labelled) / "input.png").write_bytes(b"not really a png")
    svc_verdicts.record_verdict(svc, labelled, verdict="accept", stage="reference")

    rows = {j["id"]: j for j in svc_jobs.list_jobs(svc)}

    assert rows[graded]["grade"] == 4
    assert rows[labelled]["grade"] is None


def test_a_regraded_job_lists_the_newer_grade(svc):
    """Verdicts are append-only (a changed mind is a new row); the row must
    show what the reviewer thinks now, not what they filed first."""
    from realmspinner.service import verdicts as svc_verdicts

    job_id = _finished_mesh(svc)
    svc_verdicts.record_verdict(svc, job_id, grade=-3, reasons=["holes"])
    svc_verdicts.record_verdict(svc, job_id, grade=5)

    rows = {j["id"]: j for j in svc_jobs.list_jobs(svc)}

    assert rows[job_id]["grade"] == 5


# --- J91: the trash ----------------------------------------------------------


def test_a_trashed_row_is_in_exactly_one_of_the_two_views():
    trashed = job(deleted_at=123.0)
    live = job()
    assert Filters().matches(live) and not Filters().matches(trashed)
    assert Filters(trash=True).matches(trashed) and not Filters(trash=True).matches(live)


def test_trashing_leaves_the_files_alone(svc):
    """That is the whole feature: a "trash" that had already deleted the mesh
    would be a different, worse thing wearing the name."""
    created = svc_jobs.create_job(svc, kind="text", prompt="a barrel")
    job_id = created["id"]
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(job_id) / "model.glb").write_bytes(b"x")
    svc.store.set_status(job_id, "done")

    svc_jobs.trash_job(svc, job_id)
    assert (svc.job_dir(job_id) / "model.glb").exists()
    assert svc.store.get(job_id)["deleted_at"] is not None

    svc_jobs.restore_job(svc, job_id)
    assert svc.store.get(job_id)["deleted_at"] is None


def test_trashing_a_queued_job_cancels_it(svc):
    """A trashed row is still a row. Without the cancel the worker would pick
    up a job the user has thrown away, spend two minutes of GPU on it and write
    a mesh into a directory nothing shows."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    assert svc.store.get(job_id)["status"] == "queued"
    svc_jobs.trash_job(svc, job_id)
    assert svc.store.get(job_id)["status"] == "cancelled"
    assert svc.store.next_queued() is None


def test_a_running_job_cannot_be_trashed(svc):
    """The refusal is about the *filesystem*, which does not care that this
    delete is reversible."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    svc.store.set_status(job_id, "running")
    with pytest.raises(Conflict):
        svc_jobs.trash_job(svc, job_id)


def test_emptying_the_trash_reads_all_of_it_not_a_page(svc):
    """"Empty" that left the older half behind while reporting success would
    be the worst possible reading of the word."""
    source = inspect.getsource(svc_jobs.empty_trash)
    assert "store.trashed()" in source
    assert "list_jobs" not in source

    ids = []
    for _ in range(3):
        job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
        svc.store.set_status(job_id, "done")
        svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        svc_jobs.trash_job(svc, job_id)
        ids.append(job_id)
    # Exact rather than a subset: the count is the claim, and ``kept`` is 0
    # because none of these carries a verdict. A trashed job that does is held
    # back -- see ``test_emptying_the_trash_keeps_a_labelled_image``.
    assert svc_jobs.empty_trash(svc) == {"deleted": 3, "kept": 0}
    for job_id in ids:
        assert svc.store.get(job_id) is None
        assert not svc.job_dir(job_id).exists()


def test_the_trash_query_is_ordered_by_when_it_was_thrown_away(svc):
    ids = []
    for _ in range(3):
        job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
        svc.store.set_status(job_id, "done")
        svc_jobs.trash_job(svc, job_id)
        ids.append(job_id)
        time.sleep(0.002)
    assert [row["id"] for row in svc.store.trashed()] == ids[::-1]


def test_the_card_delete_no_longer_asks_and_the_permanent_one_does():
    """The trash *is* the confirmation, and a better one: an undo the user can
    take an hour later beats a question answered in half a second."""
    source = inspect.getsource(library._overflow)
    body = source.split('menu_item("Delete"', 1)[1]
    assert "delete_asset(ctx, job_id)" in body
    assert "confirms.ask" not in body
    assert "confirms.ask" in inspect.getsource(library._trash_menu)


def test_delete_asset_trashes_and_purge_asset_deletes():
    """Every caller keeps its old name: what the user asked for is "get this
    out of my library", and the reversibility of that is a property of the app
    rather than of each call site."""
    assert "trash_job" in inspect.getsource(library.delete_asset)
    assert "delete_job" in inspect.getsource(library.purge_asset)


def test_bulk_delete_uses_the_batch_helper_for_one_toast_one_undo():
    """The toolbar's Delete once looped ``delete_asset`` per ticked row --
    N toasts and N undos for one act. It must go through ``delete_assets``
    instead, which is exactly why that helper exists (see its docstring)."""
    source = inspect.getsource(library._bulk_action)
    body = source.split('key == "delete"', 1)[1]
    assert "delete_assets(ctx, picked)" in body
    assert "[delete_asset(ctx, j) for j in picked]" not in body


def test_quick_open_never_offers_a_trashed_asset():
    """It does not go through the filter bar at all, so without its own rule it
    would be the one surface where a deleted asset still turns up."""
    from types import SimpleNamespace

    from realmspinner.studio import palette

    rows = [job(id="live", name="chest"), job(id="gone", name="chest", deleted_at=1.0)]
    ctx = SimpleNamespace(cache=SimpleNamespace(jobs=rows))
    assert [j["id"] for j in palette.assets(ctx, "chest")] == ["live"]


# --- J88 / J89 / O119: the window and the rows -------------------------------


def test_filtering_reports_only_the_controls_that_hide_rows():
    """Sort and direction reorder; the trash re-addresses. Neither narrows, so
    neither may trigger the "this is only a window" warning."""
    assert not library.filtering(Filters())
    assert not library.filtering(Filters(sort="name", descending=False, trash=True))
    assert library.filtering(Filters(text="chest"))
    assert library.filtering(Filters(status="error"))
    assert library.filtering(Filters(kind="model"))
    assert library.filtering(Filters(favorites_only=True))


@pytest.mark.parametrize(
    ("age_days", "expected"),
    [(0, "Today"), (1, "Yesterday"), (3, "This week"), (30, None)],
)
def test_date_grouping_uses_calendar_days(age_days, expected):
    """Something made at 23:50 last night is "yesterday" at 00:10, and an
    elapsed-hours rule would call it "today"."""
    import datetime as dt

    now = dt.datetime(2026, 8, 8, 12, 0).timestamp()
    then = now - age_days * 86400
    group = library.date_group(then, now=now)
    assert group == (expected or dt.date.fromtimestamp(then).strftime("%Y-%m"))


def test_an_undated_row_says_so_rather_than_landing_in_today():
    assert library.date_group(None) == "Undated"


def test_usable_only_filter_warns_when_it_only_covers_the_loaded_window():
    """The 2026-09-20 audit, finding shell-02: ``usable_only`` narrows to the
    loaded newest-N window exactly as ``kind`` and the size sort already do
    (grade lives in the ``verdicts`` table, not a ``jobs`` column, so
    ``JobsCache.request_widen``'s SQL predicates cannot reach it any more
    than they can ``kind``) -- but only ``kind``/``sort`` tripped
    ``_narrows_the_window``, so turning the toggle on silently hid older
    usable rows with no "Load older" prompt telling the user their view is
    partial."""
    assert library._narrows_the_window(Filters(usable_only=True))
    assert not library._narrows_the_window(Filters())


def test_the_window_can_be_reset(svc):
    cache = cache_mod.JobsCache(svc)
    assert cache.limit == cache_mod.LIST_LIMIT
    cache.load_more()
    cache.load_more()
    assert cache.limit == cache_mod.LIST_LIMIT * 3
    cache.reset_window()
    assert cache.limit == cache_mod.LIST_LIMIT


def test_resetting_a_window_that_never_grew_does_not_force_a_re_read(svc):
    cache = cache_mod.JobsCache(svc)
    cache._dirty = False
    cache.reset_window()
    assert cache._dirty is False


def test_a_job_outside_the_loaded_window_is_found_by_its_prompt(svc):
    """W2.1: the cache only ever loads the newest page, so a search used to
    find nothing for a job the pager had not reached yet -- "Load older" was
    the only way in. ``request_widen`` merges a store-side match into the
    window before ``Filters.matches`` runs, so the search reaches it directly.

    Submitted through a real ``TaskRunner`` and collected like ``request``/
    ``adopt`` already are (shell-01, the 2026-09-08 audit): the store query
    used to run inline here, on whichever thread called ``widen_for_filters``,
    which on the frame thread was the exact stall the ordinary list poll's
    ``request``/``read``/``adopt`` split exists to prevent.
    """
    from realmspinner.studio.jobs_cache import SEARCH_KEY
    from realmspinner.studio.tasks import TaskRunner

    old_id = svc.store.create("text", "a rusty iron lantern", {})
    svc.store._conn.execute("UPDATE jobs SET created_at = 1.0 WHERE id = ?", (old_id,))
    svc.store._conn.commit()
    svc.store.create("text", "an unrelated crate", {})

    cache = cache_mod.JobsCache(svc, limit=1)
    cache.tick()
    assert old_id not in cache.by_id  # the one-row window missed it

    runner = TaskRunner(workers=1)
    try:
        filters = Filters(text="lantern")
        assert cache.request_widen(filters, runner) is True
        # Handed off, not run here: nothing about the cache changes merely
        # because a search was submitted.
        assert old_id not in cache.by_id

        deadline = time.monotonic() + 5
        done: list[Any] = []
        while time.monotonic() < deadline and not done:
            done = runner.poll()
        assert done and done[0].key == SEARCH_KEY, done

        cache.adopt_widen(done[0].result)
        assert old_id in [j["id"] for j in cache.visible(filters)]
    finally:
        runner.shutdown(wait=False)


@pytest.fixture(scope="module")
def imgui_ctx(gl):
    """See ``test_studio_smoke.imgui_ctx`` -- the same context shape, module
    scoped for the same reason: a session-scoped context left alive after the
    last test here collides with the next file's own context over one GL
    context."""
    from imgui_bundle import imgui

    from realmspinner.studio import imgui_backend, theme

    prev_screen = type(gl).__dict__.get("screen")
    fbo = gl.simple_framebuffer((1600, 950))
    fbo.use()
    type(gl).screen = property(lambda _self: fbo)

    imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    theme.apply(imgui)
    renderer = imgui_backend.ImguiRenderer(gl)
    yield imgui, renderer
    renderer.shutdown()
    imgui.destroy_context()
    if prev_screen is not None:
        type(gl).screen = prev_screen


@pytest.fixture
def app_ctx(gl, svc, tmp_path, imgui_ctx):
    from realmspinner.studio import textures
    from realmspinner.studio.app_ctx import Ctx
    from realmspinner.studio.runtime import Runtime
    from realmspinner.studio.tasks import TaskRunner
    from realmspinner.studio.viewer_embed import Viewer

    runtime = Runtime(svc.config)
    runtime.store = svc.store
    runtime.tasks = TaskRunner(workers=1)
    viewer = Viewer(gl)
    from realmspinner.studio.settings import Settings
    from realmspinner.studio.state import AppState

    ctx = Ctx(
        svc=svc,
        runtime=runtime,
        state=AppState(),
        cache=cache_mod.JobsCache(svc),
        tasks=runtime.tasks,
        settings=Settings.load(tmp_path),
        viewer=viewer,
        textures=textures.ThumbnailCache(gl),
    )
    yield ctx
    viewer.release()
    ctx.textures.release()
    runtime.tasks.shutdown(wait=False)


def _frame(imgui_ctx, build):
    imgui, renderer = imgui_ctx
    imgui.new_frame()
    imgui.set_next_window_size((1200, 900))
    imgui.begin("##host")
    build()
    imgui.end()
    imgui.render()
    renderer.render(imgui.get_draw_data())


def _drain_search(ctx: Any) -> None:
    """Poll a real ``TaskRunner`` for the search :meth:`request_widen`
    submitted and hand its result to ``adopt_widen`` -- the frame-thread half,
    which ``main._on_task_done`` performs in the app (shell-01)."""
    from realmspinner.studio.jobs_cache import SEARCH_KEY

    deadline = time.monotonic() + 5
    done: list[Any] = []
    while time.monotonic() < deadline and not done:
        done = ctx.tasks.poll()
    assert done and done[0].key == SEARCH_KEY, done
    ctx.cache.adopt_widen(done[0].result)


def test_a_tag_search_finds_an_asset_outside_the_loaded_window_in_both_library_views(
    app_ctx, imgui_ctx
):
    """A3: ``request_widen`` widens the store-side candidate set for a
    ``tag:`` filter, not only free text -- and both Library views call it, so
    a structured filter reaches past the loaded window whichever one is open.

    Before this fix, ``JobsCache.widen_for_search`` only ever ran from
    ``panes/library.py`` and only ever asked the store for a name/prompt
    substring -- so a ``tag:`` filter, and the full-window Library entirely,
    both stayed blind to anything outside the loaded page.
    """
    svc = app_ctx.svc
    old_id = svc.store.create("text", "an old wooden chest", {})
    svc.store.set_meta(old_id, tags="rusty")
    svc.store._conn.execute("UPDATE jobs SET created_at = 1.0 WHERE id = ?", (old_id,))
    svc.store._conn.commit()
    svc.store.create("text", "a fresh crate", {})

    app_ctx.state.filters = Filters(text="tag:rusty")
    app_ctx.cache = cache_mod.JobsCache(svc, limit=1)
    app_ctx.cache.tick()
    assert old_id not in app_ctx.cache.by_id  # the one-row window missed it

    _frame(imgui_ctx, lambda: library.draw(app_ctx))
    _drain_search(app_ctx)
    assert old_id in [j["id"] for j in app_ctx.cache.visible(app_ctx.state.filters)]

    # Fresh cache and fresh filters object: the full-window Library must reach
    # the same job on its own, not merely inherit the sidebar's merge.
    app_ctx.state.filters = Filters(text="tag:rusty")
    app_ctx.cache = cache_mod.JobsCache(svc, limit=1)
    app_ctx.cache.tick()
    assert old_id not in app_ctx.cache.by_id

    _frame(imgui_ctx, lambda: library_full.draw(app_ctx))
    _drain_search(app_ctx)
    assert old_id in [j["id"] for j in app_ctx.cache.visible(app_ctx.state.filters)]


def test_the_job_list_is_read_off_the_frame_thread_and_adopted_on_it(svc):
    """A2: ``JobsCache.tick`` used to call ``list_jobs`` inline -- one sqlite
    read plus a per-row ``attach_files`` stat over the whole window -- on
    whatever thread called it, and ``main.py`` calls it every frame. The fix
    splits it as ``refresh_storage``/``measure``/``adopt_storage`` already are:
    ``request`` submits :meth:`JobsCache.read` to a real :class:`TaskRunner`
    and touches nothing on the cache itself; only :meth:`JobsCache.adopt`,
    called with the task's result, may ever assign ``jobs`` or ``by_id``.
    """
    from realmspinner.studio.tasks import TaskRunner

    svc.store.create("text", "a rusty sword", {})
    cache = cache_mod.JobsCache(svc)
    runner = TaskRunner(workers=1)
    read_thread: dict[str, str] = {}
    real_read = cache.read

    def spying_read(files_snapshot: Any) -> Any:
        read_thread["name"] = threading.current_thread().name
        return real_read(files_snapshot)

    cache.read = spying_read  # type: ignore[method-assign]

    try:
        assert cache.request(runner) is True
        # Handed off, not run here: nothing about the cache has changed on
        # this thread merely because a read was submitted.
        assert cache.jobs == []
        assert cache.by_id == {}
        assert cache.error is None

        deadline = time.monotonic() + 5
        done: list[Any] = []
        while time.monotonic() < deadline and not done:
            done = runner.poll()
        assert done and done[0].key == "jobs-list", done

        # The read really ran on a different thread from this one.
        assert read_thread.get("name") not in (None, threading.current_thread().name)
        # And still nothing is published until ``adopt`` is called -- with the
        # task's result, on this (the frame) thread.
        assert cache.jobs == []

        assert cache.adopt(done[0].result) is True
        assert len(cache.jobs) == 1
        assert list(cache.by_id) == [cache.jobs[0]["id"]]
    finally:
        cache.read = real_read  # type: ignore[method-assign]
        runner.shutdown(wait=False)


def test_load_older_fetches_the_next_page_rather_than_re_reading_the_window(svc, monkeypatch):
    """O119/A2: ``load_more`` used to raise ``self.limit``, and every
    subsequent refresh re-read (and re-``attach_files``-statted) the *whole*
    growing window from the top. ``read`` now reuses whatever older rows are
    already held and only asks the store for the delta -- proved here by
    counting ``JobStore.list`` calls and the ``limit``/``before`` each one
    used rather than by any behaviour the old, unfixed code also happened to
    produce.
    """
    from realmspinner.studio import jobs_cache as cache_mod_local

    for i in range(cache_mod_local.LIST_LIMIT + 5):
        job_id = svc.store.create("text", f"asset {i}", {})
        svc.store._conn.execute(
            "UPDATE jobs SET created_at = ? WHERE id = ?", (float(i), job_id)
        )
    svc.store._conn.commit()

    calls: list[tuple[int, Any]] = []
    real_list = svc.store.list

    def counting_list(limit=100, before=None, kind=None):
        calls.append((limit, before))
        return real_list(limit, before, kind)

    monkeypatch.setattr(svc.store, "list", counting_list)

    cache = cache_mod.JobsCache(svc)
    cache.tick()
    assert len(calls) == 1
    first_limit = calls[0][0]
    assert first_limit == cache_mod_local.LIST_LIMIT

    calls.clear()
    cache.load_more()
    cache.tick()

    # The already-loaded top page plus whatever older rows were already held
    # must not be re-read: the newest page is refreshed (one call with no
    # cursor) and only the *new* page beyond it is fetched (one call with a
    # ``before`` cursor) -- never a single call asking for the whole widened
    # window in one go, which is what "re-reading the window" would look like.
    assert not any(before is None and limit > cache_mod_local.LIST_LIMIT for limit, before in calls)
    assert any(before is not None for limit, before in calls), calls
    assert len(cache.jobs) == cache_mod_local.LIST_LIMIT + 5


def test_the_size_sort_notices_a_measurement_landing(svc):
    """The storage walk is deferred and lands on a task thread long after the
    list did; without its own generation the memo would never reorder."""
    cache = cache_mod.JobsCache(svc)
    first = cache._filters_key(Filters(sort="size"))
    cache._dir_sizes = {"a": 1}
    cache._sizes_generation += 1
    assert cache._filters_key(Filters(sort="size")) != first


# --- O116: the prune count ---------------------------------------------------


def test_the_prune_count_is_asked_rather_than_assumed():
    source = inspect.getsource(library.ask_prune)
    assert "input_int" in source
    assert "body=body" in source
    # Reset every time it is asked: a destructive default somebody set once and
    # forgot is exactly the setting that deletes something they wanted.
    assert "PRUNE_KEEP_DEFAULT" in source
    assert "settings.set" not in source


def test_the_prune_message_does_not_promise_a_trash_it_does_not_use():
    """Prune exists to reclaim disk; one that moved two hundred jobs into the
    trash would free nothing while reporting that it had."""
    source = inspect.getsource(library.ask_prune)
    assert "deleted from disk" in source
    assert "cannot be undone" in source


# --- opening a job's folder ---------------------------------------------------


class _FolderCtx:
    def __init__(self, path):
        self._path = path
        self.toasts: list[tuple[str, str]] = []
        self.submitted: list[tuple[str, Any, tuple]] = []

    def job_dir(self, job_id):
        return self._path

    def toast(self, text, level="info", action=None):
        self.toasts.append((text, level))

    def submit(self, key, fn, *args, **kwargs):
        self.submitted.append((key, fn, args))
        return True


def test_opening_a_folder_goes_through_ctx_submit_not_inline(tmp_path):
    """``startfile`` is a blocking shell call, so it takes the same door
    ``ctx.open_log`` already uses rather than running on the frame thread."""
    import os

    ctx = _FolderCtx(tmp_path)

    library._open_folder(ctx, "j1")

    assert ctx.submitted, "startfile must be handed to ctx.submit, not called inline"
    key, fn, args = ctx.submitted[0]
    assert key.startswith("open-folder:")
    assert fn is os.startfile
    assert args == (str(tmp_path),)
    assert not ctx.toasts, "a folder that exists is not itself a toast"


def test_opening_a_missing_folder_still_toasts_inline():
    """The existence check stays synchronous -- it is a stat, not a shell
    call -- so a missing folder never reaches ``ctx.submit`` at all."""
    missing = type("P", (), {"exists": lambda self: False})()
    ctx = _FolderCtx(missing)

    library._open_folder(ctx, "j1")

    assert not ctx.submitted
    assert ctx.toasts and "not on disk" in ctx.toasts[0][0]


# --- Convert...: the format picker reached from a card and from the bulk bar --
#
# The popup itself needs an imgui context to draw (``_draw_convert_popup``,
# ``_convert_popup_body``), so what is tested here is the half that does not:
# which kinds offer a format list at all, and that both doors -- the card's
# overflow menu and the bulk bar -- route into the one function that decides
# it, the way ``test_bulk_delete_uses_the_batch_helper_for_one_toast_one_undo``
# above pins the delete door onto ``delete_assets`` rather than a restated loop.


def test_convert_formats_are_offered_only_for_the_kinds_that_have_any():
    from realmspinner.service import files as svc_files

    for kind in ("music", "reference", "tile", "tilesheet"):
        names = {n for n, _label in library._CONVERT_FORMATS[kind]}
        assert names, kind
        for name in names:
            assert name in svc_files.MEDIA, (kind, name)
    # A mesh, a rig, a character sheet -- nothing here has a "native format"
    # the way a take or a picture does, so none of them gets a row at all.
    for kind in ("model", "rig", "sheet", "sprite"):
        assert kind not in library._CONVERT_FORMATS


def test_convert_formats_pulls_only_the_reencodings_out_of_the_export_grid():
    """``ARTIFACTS_MUSIC``/``ARTIFACTS_2D`` carry rows -- WAV itself, the
    manifest, the source PNG -- that are not format *conversions*; the
    Convert picker must not offer a button for one of those."""
    music_names = {n for n, _label in library._CONVERT_FORMATS["music"]}
    assert "manifest.json" not in music_names
    ref_names = {n for n, _label in library._CONVERT_FORMATS["reference"]}
    assert "input.png" not in ref_names
    assert "manifest.json" not in ref_names
    assert "icon.png" not in ref_names


def test_the_format_buttons_wrap_instead_of_growing_the_dialog():
    """A music card offers five formats. On one line at ``_CONVERT_BUTTON_W``
    each that made the picker roughly 2.5x the width floor it asks
    ``modal_bounds`` for -- a floor, not a cap, so nothing clipped it, and at UI
    scale 2 the row ran into the viewport clamp with its last buttons cut off.
    """
    formats = library._CONVERT_FORMATS["music"]
    assert len(formats) > library._CONVERT_COLUMNS, "the row that grew the dialog"
    rows = library._convert_rows(formats)
    assert all(len(row) <= library._CONVERT_COLUMNS for row in rows)
    widest = max(len(row) for row in rows) * library._CONVERT_BUTTON_W
    assert widest <= library._CONVERT_WIDTH


def test_wrapping_the_formats_loses_none_of_them_and_reorders_nothing():
    for kind, formats in library._CONVERT_FORMATS.items():
        flat = [pair for row in library._convert_rows(formats) for pair in row]
        assert flat == list(formats), kind


def test_one_format_is_still_one_row():
    """The split is columns, not a grid the shortest list has to fill."""
    assert library._convert_rows((("a.wav", "WAV"),)) == ((("a.wav", "WAV"),),)
    assert library._convert_rows(()) == ()


def test_the_picker_draws_the_rows_rather_than_one_long_line():
    """The layout is fixed columns and not ``same_line_or_wrap``: this modal is
    ``always_auto_resize``, so the width left on the line is decided by the very
    row that would be asking about it."""
    source = inspect.getsource(library._convert_popup_body)
    assert "_convert_rows(popup.formats)" in source
    assert "same_line_or_wrap" not in source


def test_the_overflow_menu_offers_a_convert_entry_when_the_card_can():
    source = inspect.getsource(library._overflow)
    assert "_convert_formats(job)" in source
    assert "_start_convert(ctx, [job_id])" in source


def test_the_bulk_bar_routes_convert_through_start_convert():
    source = inspect.getsource(library._bulk_action)
    body = source.split('key == "convert"', 1)[1]
    assert "_start_convert(ctx, picked)" in body


class _ConvertCtx:
    def __init__(self, jobs):
        from types import SimpleNamespace

        self.cache = SimpleNamespace(get={j["id"]: j for j in jobs}.get)
        self.toasts: list[tuple[str, str]] = []
        self.submitted: list[tuple[str, Any]] = []
        # ``_convert_busy`` (shell-03, the 2026-09-11 audit) reads
        # ``ctx.tasks.any_busy``/``ctx.busy`` before ``_start_convert`` ever
        # gets to ``submit`` -- nothing here is ever actually in flight, so
        # both answer "no".
        self.tasks = SimpleNamespace(any_busy=lambda _prefix: False)

    def toast(self, text, level="info", action=None):
        self.toasts.append((text, level))

    def busy(self, key):
        return False

    def submit(self, key, fn, *args, **kwargs):
        self.submitted.append((key, fn))
        return True


def test_start_convert_toasts_rather_than_opening_a_picker_with_nothing_in_it():
    ctx = _ConvertCtx([job(id="m1", stage="model")])
    library._start_convert(ctx, ["m1"])
    assert not ctx.submitted
    assert ctx.toasts and "convert" in ctx.toasts[0][0].lower()


def test_start_convert_refuses_a_mixed_kind_selection():
    """A picker whose buttons only work for some of what is ticked is worse
    than no picker -- refused with a toast instead of shown half-broken."""
    ctx = _ConvertCtx(
        [job(id="t1", kind="music", stage="music"), job(id="r1", stage="reference")]
    )
    library._start_convert(ctx, ["t1", "r1"])
    assert not ctx.submitted
    assert ctx.toasts


def test_start_convert_submits_a_per_job_key_for_a_single_asset():
    """Distinct from the bulk key, and job-scoped: two cards' Convert menus
    opened in the same frame must not collide on one task key the way two
    concurrent saves of the same artifact are meant to (``TaskRunner.submit``
    refuses a key already in flight -- a shared key here would silently drop
    the second click)."""
    ctx = _ConvertCtx([job(id="t1", kind="music", stage="music")])
    library._start_convert(ctx, ["t1"])
    assert len(ctx.submitted) == 1
    assert ctx.submitted[0][0] == "convert:t1"


def test_start_convert_submits_the_export_prefix_for_several_assets():
    """``export-``: the same prefix ``_export_zip``/``_export_folder`` submit
    under, so ``main.py``'s existing "Exported to <path>" toast fires with no
    change there -- see ``_run_convert``'s own comment."""
    ctx = _ConvertCtx(
        [job(id="t1", kind="music", stage="music"), job(id="t2", kind="music", stage="music")]
    )
    library._start_convert(ctx, ["t1", "t2"])
    assert len(ctx.submitted) == 1
    assert ctx.submitted[0][0] == "export-convert"


# --- shell-03: two bulk flows racing over one popup slot ---------------------
#
# ``_export_zip``/``_export_folder`` both publish onto the single
# ``ctx.state._library_export`` slot (they are the two doors into
# ``_run_export``), and ``_start_convert``'s two keys -- ``convert:<id>`` for
# one asset, ``export-convert`` for several -- both publish onto the single
# ``ctx.state._library_convert`` slot. Nothing before shell-03 (the 2026-09-11
# audit) stopped a second flow from starting while the first was still parked
# waiting for its popup's decision: the second flow's popup silently replaced
# the first's in the slot, and when the first flow's own native dialog or
# picker was finally answered it wrote its popup back over the slot and then,
# in its ``finally``, cleared the slot to ``None`` -- so whichever popup was
# not the one currently in the slot at the moment a person answered it never
# received a decision, and its task thread stayed blocked forever on
# ``popup.decisions.get()``.
#
# These use a real ``TaskRunner`` behind ``ctx.submit``/``ctx.busy`` -- the
# refusal a fix has to make is "is this key already pending", and
# ``TaskRunner.submit`` answers that synchronously (it inserts into
# ``_pending`` under a lock before returning), so the second call below is
# already racing the *right* state the instant it runs, with no dialog, no
# sleep and no second thread required to force the interleaving. What proves
# the clobber is not watching a background thread run to completion; it is
# that both keys get *accepted* at all -- ``_run_export``/``_run_convert``
# each unconditionally write ``ctx.state._library_export``/``_library_convert``
# the moment they start, so two accepted keys is two writers of the one slot.


def _library_ctx(svc, *, export_dir: str = "proj", jobs: list[dict[str, Any]] | None = None):
    """A ctx whose ``submit`` records a key as in flight rather than running it.

    A refusal-to-start guard reads ``ctx.busy(key)`` on the frame thread before
    anything is submitted, so the task itself never has to run for this to be a
    faithful test -- and it must not. These flows park on
    ``popup.decisions.get()`` until a popup is answered, so a real ``TaskRunner``
    here leaves a worker blocked forever; ``shutdown(timeout=)`` returns but the
    thread survives, and ``concurrent.futures``' atexit join then hangs the whole
    pytest process after it has already reported. This stand-in keeps the
    bookkeeping the guard actually consults and owns no threads.
    """
    from types import SimpleNamespace

    in_flight: set[str] = set()
    accepted: dict[str, bool] = {}
    by_id = {j["id"]: j for j in (jobs or [])}

    def submit(key, fn, *args, **kwargs):
        # ``TaskRunner.submit``'s own contract: one flow per key, and a second
        # submit under a key already running is refused.
        ok = key not in in_flight
        if ok:
            in_flight.add(key)
        accepted[key] = ok
        return ok

    tasks = SimpleNamespace(
        is_busy=lambda key: key in in_flight,
        # ``convert:<id>`` is one key per asset, so its guard asks by prefix.
        any_busy=lambda prefix: any(k.startswith(prefix) for k in in_flight),
        shutdown=lambda timeout=0.0: None,
        finish=in_flight.discard,
    )
    ctx = SimpleNamespace(
        svc=svc,
        cache=SimpleNamespace(jobs=list(jobs or []), get=by_id.get),
        state=SimpleNamespace(_library_export=None, _library_convert=None),
        tasks=tasks,
        export_dir=export_dir,
        submit=submit,
        busy=lambda key: key in in_flight,
        toast=lambda *a, **k: None,
    )
    return ctx, tasks, accepted


def test_two_concurrent_bulk_export_tasks_do_not_clobber_each_others_popup_slot(svc):
    """shell-03: "Export zip..." then, before its native Save dialog is
    answered, "Save to project" -- reachable with two ordinary clicks because
    ``dialogs.save_file``/``select_folder`` carry no owner window and the bulk
    toolbar stays clickable while one is open. Both go through ``ctx.submit``
    under different keys and both, once running, write
    ``ctx.state._library_export`` -- so the second call being *accepted* at
    all, while the first is still in flight, is already the bug: whichever of
    the two writes that slot second erases the other's popup, and the loser's
    task thread is left parked on ``popup.decisions.get()`` with nothing left
    that can reach it.

    Nothing is stubbed and no task runs: the guard this checks lives on the
    frame thread, before ``submit``, so the flow never has to reach a picker
    or a popup for the refusal to be the thing under test. That also keeps a
    worker from parking on a decision nobody will answer -- see
    ``_library_ctx``.
    """
    ctx, _tasks, accepted = _library_ctx(svc)
    library._export_zip(ctx, [])
    assert accepted.get("export-zip"), "the zip export was not even submitted"

    # Same click sequence as shell-03: the second bulk action, while the
    # first is still in flight (its own dialog not yet answered, in the
    # real app -- here, simply never told to finish).
    library._export_folder(ctx, [])

    assert not accepted.get("export-folder", False), (
        "the folder export was allowed to start while the zip export "
        "was still in flight; both write ctx.state._library_export from "
        "_run_export, so the second one clobbers the first's popup "
        "there and the first flow's task thread is left permanently "
        "blocked on popup.decisions.get() once nothing can reach it to "
        "answer it any more (the 2026-09-11 audit, shell-03)"
    )


def test_two_concurrent_convert_tasks_do_not_clobber_each_others_popup_slot(svc):
    """shell-03's other slot: a single-asset "Convert..." (key
    ``convert:<id>``) and a several-asset one (key ``export-convert``) both
    write ``ctx.state._library_convert`` from ``_run_convert``. Driven
    through ``_start_convert`` itself, not ``_run_convert`` directly, because
    that is where a refusal-to-start guard has to live -- ``_run_convert``
    only ever runs after a flow has already been allowed to start, by which
    point refusing is too late.
    """
    jobs = [
    job(id="t1", kind="music", stage="music"),
    job(id="t2", kind="music", stage="music"),
    job(id="t3", kind="music", stage="music"),
    ]
    ctx, _tasks, accepted = _library_ctx(svc, jobs=jobs)
    library._start_convert(ctx, ["t1"])
    assert accepted.get("convert:t1"), "the first convert was not even submitted"

    library._start_convert(ctx, ["t2", "t3"])

    assert not accepted.get("export-convert", False), (
        "the second convert was allowed to start while the first was "
        "still in flight; both write ctx.state._library_convert from "
        "_run_convert, so the second one clobbers the first's popup "
        "there and the first flow's task thread is left permanently "
        "blocked on popup.decisions.get() (the 2026-09-11 audit, "
        "shell-03)"
    )


# --- the full-window grid's row layout and row clipping (A1) -----------------


def test_the_grids_row_layout_puts_a_date_heading_at_the_start_of_its_own_row():
    """Pure-function twin of ``library._convert_rows``: a date heading always
    breaks the row before it, even mid-row, because the column counter resets
    there -- a "Today" heading landing beside yesterday's cards on the same
    row would be a lie about what separates them (J89)."""
    now = time.time()
    jobs = [
        job(id="a", created_at=now),
        job(id="b", created_at=now),
        job(id="c", created_at=now - 90 * 86400),  # months back -> a new heading
    ]
    entries = library_full._row_layout(jobs, count=3, grouped=True)
    kinds = [kind for kind, _payload in entries]
    assert kinds.count("heading") == 2
    # The row that carried "a" and "b" is closed out *before* the second
    # heading, not left open with "c" appended onto it.
    row_entries = [payload for kind, payload in entries if kind == "row"]
    assert [j["id"] for j in row_entries[0]] == ["a", "b"]
    assert [j["id"] for j in row_entries[1]] == ["c"]


def test_the_grids_row_layout_wraps_at_the_column_count_with_no_headings():
    jobs = [job(id=str(i)) for i in range(7)]
    entries = library_full._row_layout(jobs, count=3, grouped=False)
    assert all(kind == "row" for kind, _payload in entries)
    assert [len(payload) for _kind, payload in entries] == [3, 3, 1]
    flat = [j["id"] for _kind, payload in entries for j in payload]
    assert flat == [j["id"] for j in jobs]


class _FakeGridImgui:
    """Just enough of imgui for ``library_full._row_clipper``'s arithmetic --
    ``test_resource_ceilings.py``'s ``_FakeImgui`` for ``library._clipper``,
    at row granularity."""

    def __init__(self, view: float, scroll: float) -> None:
        self._view = view
        self._scroll = scroll
        self.cursor = 0.0
        self.dummies: list[float] = []

    def get_window_size(self):
        from types import SimpleNamespace

        return SimpleNamespace(y=self._view)

    def get_scroll_y(self):
        return self._scroll

    def get_cursor_pos_y(self):
        return self.cursor

    def dummy(self, size):
        self.dummies.append(size[1])
        self.cursor += size[1]


def test_the_full_library_grid_submits_only_the_rows_on_screen(monkeypatch):
    """A1: ``_grid`` used to call ``_cell`` for every loaded job regardless of
    scroll position -- 5,000 assets paid ``push_id``, two
    ``get_cursor_screen_pos``, ``draggable_source``, ``is_item_hovered``,
    ``_card_context`` and ``fit_text`` per cell even scrolled far off screen
    (the measured 372 ms/frame). The fix is ``library._clipper``'s own
    arithmetic at row rather than cell granularity, gated on the same
    ``library.CLIP_THRESHOLD``."""
    row_h = 100.0
    count = 3
    total = 300  # well over CLIP_THRESHOLD
    fake = _FakeGridImgui(view=400.0, scroll=0.0)
    monkeypatch.setattr(library_full, "imgui", fake)
    from types import SimpleNamespace

    ctx = SimpleNamespace(state=SimpleNamespace(library_scroll_to=None, selected=None))
    skip = library_full._row_clipper(ctx, total)
    assert skip is not None

    rows_drawn = 0
    cells_drawn = 0
    for start in range(0, total, count):
        row_jobs = [job(id=str(i)) for i in range(start, min(start + count, total))]
        if skip(row_jobs, row_h):
            continue
        rows_drawn += 1
        cells_drawn += len(row_jobs)
        fake.cursor += row_h

    total_rows = -(-total // count)
    assert rows_drawn < total_rows, "an off-screen row must be skipped"
    assert cells_drawn < total, "a 400px viewport of 100px rows holds only a few"
    # Every skipped row still takes its own height, so the scrollbar and every
    # row's place in the list are what they would have been.
    assert fake.cursor == pytest.approx(total_rows * row_h)


def test_the_grids_row_clipper_never_skips_the_selected_or_scroll_to_row(monkeypatch):
    fake = _FakeGridImgui(view=400.0, scroll=0.0)
    monkeypatch.setattr(library_full, "imgui", fake)
    from types import SimpleNamespace

    ctx = SimpleNamespace(
        state=SimpleNamespace(library_scroll_to=None, selected="j250")
    )
    skip = library_full._row_clipper(ctx, 300)
    assert skip is not None
    row_jobs = [job(id="j250")]
    # Placed far past the viewport (cursor is at 0, row would land at 0 --
    # force it off screen by advancing the fake cursor first).
    fake.cursor = 10_000.0
    assert skip(row_jobs, 100.0) is False


def test_dialogs_offers_a_filter_for_every_new_convert_suffix():
    """Before this, saving any of these fell through to ``["All files", "*"]``
    -- no extension offered, none appended -- because ``ARTIFACT_FILTERS`` had
    no audio or web-image rows at all."""
    from realmspinner.studio import dialogs

    for suffix in (".wav", ".flac", ".mp3", ".ogg", ".aiff", ".webp", ".jpg"):
        assert suffix in dialogs.ARTIFACT_FILTERS, suffix
        assert dialogs.ARTIFACT_FILTERS[suffix][1] == f"*{suffix}", suffix


def test_a_follow_ups_folder_is_the_one_that_holds_its_files(tmp_path):
    """A sprite draft, a rig or a retexture never gets a directory of its own
    (``asset_open``'s docstring), so ``Open folder`` on the row asked for a path
    that was never created and toasted "not on disk" for every finished one."""
    asked: list[str] = []
    ctx = _FolderCtx(tmp_path)
    ctx.job_dir = lambda job_id: asked.append(job_id) or tmp_path
    row = {"id": "DRAFT", "kind": "sprite_synthesis", "params": {"source_job": "SRC"}}

    library._open_folder(ctx, "DRAFT", row)

    assert asked == ["SRC"], asked
    assert ctx.submitted and ctx.submitted[0][0] == "open-folder:DRAFT"
    assert not ctx.toasts


def test_an_ordinary_rows_folder_is_its_own(tmp_path):
    asked: list[str] = []
    ctx = _FolderCtx(tmp_path)
    ctx.job_dir = lambda job_id: asked.append(job_id) or tmp_path

    library._open_folder(ctx, "MESH", {"id": "MESH", "kind": "image", "params": {}})

    assert asked == ["MESH"], asked


def test_the_overflow_menu_offers_open_first_for_a_finished_row():
    """The menu is one imgui popup and needs a live frame to draw, so this pins
    the wiring rather than the pixels: an Open item, gated on ``done``, routed
    through ``run_action`` (and so ``asset_open``), and ahead of every other
    item -- a follow-up row had no way to open at all, from here or from its
    card, and Enter was the only door."""
    import inspect

    source = inspect.getsource(library._overflow)
    assert 'controls.menu_item("Open"' in source
    assert source.index('controls.menu_item("Open"') < source.index("Copy job id")
    assert 'run_action(ctx, job, "open")' in source
