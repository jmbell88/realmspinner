"""``service.library_index``: what a job says about itself, how it is indexed and
how a search by meaning is fused with the substring one.

The embedder is ``tests/_embed_fakes``'s concept-table stand-in, patched in at
``embed_client.embed``; nothing here starts a ``llama-server``.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import _embed_fakes as fakes
import numpy as np
import pytest

from realmspinner import fetch, models
from realmspinner.familiar import embed_client
from realmspinner.service import library_index as li


def _job(**over):
    row = {
        "id": "j1",
        "kind": "text",
        "stage": "model",
        "name": "",
        "prompt": "a rusty iron longsword",
        "params": {},
    }
    row.update(over)
    return row


# --- what is embedded ---------------------------------------------------------


def test_the_text_is_the_name_then_the_prompt_then_a_facet_line():
    text, _hash = li.job_text(
        _job(name="Iron sword", params={"asset_type": "prop", "style_lora": "pixel-art"})
    )
    title, body = li.split_text(text)
    assert title == "Iron sword"
    lines = body.split("\n")
    assert lines[0] == "a rusty iron longsword"
    assert lines[1] == "kind: 3D model; asset: prop; style: pixel art"


def test_the_text_is_deterministic():
    job = _job(name="x", params={"asset_type": "prop"})
    assert li.job_text(job) == li.job_text(dict(job))


@pytest.mark.parametrize(
    "key,value",
    [
        ("seed", 12345),
        ("reference_seed", 7),
        ("resolution", 1024),
        ("output_dir", "C:/Users/someone/secret"),
        ("ref_path", "C:/pics/ref.png"),
        ("lora_weight", 0.6),
        ("negative_prompt", "blurry, lowres"),
        ("steps", 30),
    ],
)
def test_a_seed_a_path_a_size_or_a_number_never_reaches_the_embedder(key, value):
    plain = li.job_text(_job(name="n"))
    with_param = li.job_text(_job(name="n", params={key: value}))
    assert with_param == plain
    assert str(value) not in with_param[0]


def test_renaming_a_job_or_editing_its_prompt_changes_the_hash_and_a_seed_does_not():
    base = li.job_text(_job(name="Sword"))
    assert li.job_text(_job(name="Excalibur"))[1] != base[1]
    assert li.job_text(_job(name="Sword", prompt="a golden blade"))[1] != base[1]
    assert li.job_text(_job(name="Sword", params={"seed": 9}))[1] == base[1]


def test_a_job_with_neither_name_nor_prompt_has_nothing_to_embed():
    assert li.job_text(_job(name="  ", prompt="\t")) is None
    assert li.job_text(_job(name="", prompt=None)) is None


def test_the_kind_is_said_in_words_a_person_would_search_by():
    assert "music track" in li.job_text(_job(kind="music"))[0]
    assert "sprite sheet" in li.job_text(_job(kind="sheet"))[0]
    assert "seamless tile" in li.job_text(_job(stage="tile"))[0]
    assert "reference image" in li.job_text(_job(stage="reference"))[0]


def test_a_prompt_that_repeats_the_name_is_not_said_twice():
    text, _ = li.job_text(_job(name="a dragon", prompt="a dragon"))
    assert text.count("a dragon") == 1


def test_a_very_long_prompt_is_cut():
    text, _ = li.job_text(_job(prompt="word " * 1000))
    assert len(text) < li.MAX_PROMPT_CHARS + 300


def test_only_a_search_with_two_real_characters_is_worth_embedding():
    assert not li.eligible_query("")
    assert not li.eligible_query("a")
    assert not li.eligible_query("  a  ")
    assert li.eligible_query("ab")
    assert li.eligible_query("a b")


# --- is there an embedder -------------------------------------------------------


def test_the_embedder_is_not_installed_until_both_rows_are_on_disk(svc):
    svc.worker = SimpleNamespace(familiar_embed=object())
    assert li.installed(svc) is False
    for key in ("familiar_runtime", "familiar_embed"):
        spec = models.FAMILIAR_MODELS[key]
        base = fetch.familiar_dir(svc.config, spec)
        base.mkdir(parents=True, exist_ok=True)
        for name in spec.probe:
            (base / name).write_bytes(b"x")
    assert li.installed(svc) is True


def test_no_worker_means_not_installed_even_with_the_files(svc):
    for key in ("familiar_runtime", "familiar_embed"):
        spec = models.FAMILIAR_MODELS[key]
        base = fetch.familiar_dir(svc.config, spec)
        base.mkdir(parents=True, exist_ok=True)
        for name in spec.probe:
            (base / name).write_bytes(b"x")
    svc.worker = None
    assert li.installed(svc) is False


# --- indexing -----------------------------------------------------------------------


def _done(svc, prompt, name="", **kw):
    job_id = svc.store.create("text", prompt, kw.pop("params", {}), status="done", **kw)
    if name:
        svc.store.set_meta(job_id, name=name)
    return job_id


def test_a_pass_embeds_the_pending_jobs_and_a_second_pass_finds_nothing(svc, monkeypatch):
    fake = fakes.install(monkeypatch, svc)
    ids = [_done(svc, f"thing {i}") for i in range(5)]
    first = li.index_pending(svc, 32)
    assert first.indexed == 5 and first.more is False
    assert set(svc.store.embedding_stamps()) == set(ids)
    stamp = next(iter(svc.store.embedding_stamps().values()))
    assert stamp[0] == li.MODEL_SHA and stamp[1] == li.DIM
    second = li.index_pending(svc, 32)
    assert second.indexed == 0
    assert len(fake.document_calls) == 1  # nothing embedded the second time


def test_documents_go_out_as_title_and_body_pairs_at_the_library_width(svc, monkeypatch):
    fake = fakes.install(monkeypatch, svc)
    _done(svc, "a rusty longsword", name="Sword")
    li.index_pending(svc, 32)
    assert fake.document_calls[0]["texts"] == [
        ("Sword", "a rusty longsword\nkind: 3D model")
    ]
    row = svc.store._conn.execute("SELECT dim, length(vec) FROM job_embeddings").fetchone()
    assert tuple(row) == (256, 256 * 4)


def test_a_full_batch_says_there_may_be_more(svc, monkeypatch):
    fakes.install(monkeypatch, svc)
    for i in range(5):
        _done(svc, f"thing {i}")
    assert li.index_pending(svc, 3).more is True
    assert li.index_pending(svc, 3).more is False  # two left, fewer than the batch


def test_an_unavailable_embedder_writes_nothing_and_says_so(svc, monkeypatch):
    fake = fakes.install(monkeypatch, svc)
    fake.fail = True
    _done(svc, "a rusty longsword")
    with pytest.raises(embed_client.EmbedUnavailable):
        li.index_pending(svc, 32)
    assert svc.store.embedding_stamps() == {}


def test_a_timeout_on_the_loop_is_an_unavailable_embedder_too(svc, monkeypatch):
    fakes.install(monkeypatch, svc)
    _done(svc, "a rusty longsword")

    def timing_out(factory, timeout=30.0):
        raise TimeoutError

    monkeypatch.setattr(svc, "call_on_loop", timing_out)
    with pytest.raises(embed_client.EmbedUnavailable):
        li.index_pending(svc, 32)


def test_no_worker_to_embed_with_is_an_unavailable_embedder(svc, monkeypatch):
    fakes.install(monkeypatch, svc)
    _done(svc, "x")
    svc.worker = None
    with pytest.raises(embed_client.EmbedUnavailable):
        li.index_pending(svc, 32)


def test_indexing_named_jobs_embeds_only_finished_unindexed_ordinary_rows(svc, monkeypatch):
    fake = fakes.install(monkeypatch, svc)
    done = _done(svc, "finished")
    queued = svc.store.create("text", "queued", {})
    sweep_id = svc.store.create_sweep("s", "p", {})
    unit = _done(svc, "unit", sweep_id=sweep_id)
    blank = _done(svc, "  ")
    assert li.index_jobs(svc, [done, queued, unit, blank, "nosuchjob"]) == 1
    assert set(svc.store.embedding_stamps()) == {done}
    # Already current: costs no embed call.
    assert li.index_jobs(svc, [done]) == 0
    assert len(fake.document_calls) == 1


def test_a_renamed_job_is_re_embedded_by_name_and_by_the_verify_pass(svc, monkeypatch):
    fake = fakes.install(monkeypatch, svc)
    job_id = _done(svc, "a rusty longsword", name="Sword")
    li.index_pending(svc, 32)
    before = svc.store.embedding_stamps()[job_id]
    svc.store.set_meta(job_id, name="Excalibur")
    assert li.index_jobs(svc, [job_id]) == 1
    assert svc.store.embedding_stamps()[job_id][2] != before[2]
    svc.store.set_meta(job_id, name="Caliburn")
    assert li.index_pending(svc, 32, verify=True).indexed == 1
    assert len(fake.document_calls) == 3


def test_a_vector_from_a_different_model_is_replaced_not_ranked(svc, monkeypatch):
    fakes.install(monkeypatch, svc)
    job_id = _done(svc, "a rusty longsword")
    li.index_pending(svc, 32)
    svc.store._conn.execute("UPDATE job_embeddings SET model_sha = ?", ("0" * 64,))
    svc.store._conn.commit()
    assert li.index_pending(svc, 32).indexed == 1
    assert svc.store.embedding_stamps()[job_id][0] == li.MODEL_SHA


# --- the query ---------------------------------------------------------------------


def test_a_query_vector_is_the_library_width_and_unit_length(svc, monkeypatch):
    fake = fakes.install(monkeypatch, svc)
    vec = li.embed_query(svc, "blade")
    assert vec.shape == (256,)
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)
    assert fake.query_calls[0]["texts"] == ["blade"]


def test_an_unavailable_embedder_gives_no_query_vector_and_no_error(svc, monkeypatch):
    fake = fakes.install(monkeypatch, svc)
    fake.fail = True
    assert li.embed_query(svc, "blade") is None


def test_a_stopped_embedder_is_started_in_the_background_and_the_search_goes_on(
    svc, monkeypatch
):
    """The substring result must not wait seconds for the child to load: with the
    child stopped, the query gets no vector this once and the child is asked to
    come up, so the next refresh finds it running."""
    fake = fakes.install(monkeypatch, svc)
    fake.server.running = False
    started = []

    class Loop:
        def is_running(self):
            return True

    svc.loop = Loop()
    monkeypatch.setattr(li, "_warm", lambda s, server: started.append(server))
    assert li.embed_query(svc, "blade") is None
    assert started == [fake.server]
    assert fake.calls == []  # it did not wait on, or even ask, the cold child


# --- searching ----------------------------------------------------------------------


def _library(svc, monkeypatch):
    """24 plain rows (a baseline to stand out from) plus a sword, a chest and a
    dragon whose prompts share no word with 'blade', 'container' or 'creature'."""
    fake = fakes.install(monkeypatch, svc)
    for i in range(24):
        _done(svc, f"plain filler text {i}", name=f"filler {i}")
    sword = _done(svc, "a rusty iron longsword", name="Iron sword")
    chest = _done(svc, "an old wooden chest bound in brass", name="Treasure chest")
    dragon = _done(svc, "a red fire dragon", name="Dragon")
    assert li.index_pending(svc, 100).indexed == 27
    return fake, sword, chest, dragon


def test_a_query_sharing_no_text_with_a_job_finds_it_by_meaning(svc, monkeypatch):
    fake, sword, chest, dragon = _library(svc, monkeypatch)
    vec = li.embed_query(svc, "blade")
    hits = li.search(svc, "blade", limit=50, query_vec=vec)
    assert hits.ids == [sword]
    assert hits.semantic == {sword}
    # ... and the substring search alone finds nothing: it shares no substring.
    assert svc.store.search_ids("blade", limit=50) == []


def test_without_a_query_vector_search_is_exactly_search_ids(svc, monkeypatch):
    _library(svc, monkeypatch)
    for text in ("sword", "filler 1", "blade", "%"):
        plain = svc.store.search_ids(text, limit=50)
        hits = li.search(svc, text, limit=50)
        assert hits.ids == plain
        assert hits.semantic == frozenset()


def test_a_row_found_both_ways_ranks_above_one_found_by_either(svc, monkeypatch):
    """Fusion by reciprocal rank: 'sword' is a substring of three newer rows that
    mean nothing by it ('swordfish') and of one older row that is also a weapon.
    Newest-first puts the swordfish first on the text path alone; the row that
    is a text hit *and* a meaning hit leads once the two rankings are fused."""
    fakes.install(monkeypatch, svc)
    for i in range(24):
        _done(svc, f"plain filler text {i}")
    # Older: found by meaning (a longsword) and by text (it says "sword" too).
    both = _done(svc, "a longsword, a sword of iron", name="Both")
    # Newer: text hits that mean nothing ("swordfish" is a creature of no concept).
    text_only = [_done(svc, f"swordfish number {i}") for i in range(3)]
    li.index_pending(svc, 100)
    vec = li.embed_query(svc, "sword")
    hits = li.search(svc, "sword", limit=50, query_vec=vec)
    plain = svc.store.search_ids("sword", limit=50)
    assert plain[0] != both  # newest-first puts a text-only row first
    assert hits.ids[0] == both
    assert set(hits.ids) >= {both, *text_only}
    assert hits.semantic == {both}


def test_a_trashed_job_never_appears_by_meaning(svc, monkeypatch):
    _fake, sword, _chest, _dragon = _library(svc, monkeypatch)
    svc.store.set_deleted_if_not_running(sword, 5.0)
    vec = li.embed_query(svc, "blade")
    assert li.search(svc, "blade", limit=50, query_vec=vec).ids == []
    assert li.search(svc, "blade", limit=50, trash=True, query_vec=vec).ids == [sword]


def test_a_tag_term_narrows_the_meaning_path_and_a_tag_alone_is_not_widened(svc, monkeypatch):
    _fake, sword, chest, _dragon = _library(svc, monkeypatch)
    svc.store.set_meta(sword, tags="metal")
    vec = li.embed_query(svc, "blade")
    assert li.search(svc, "blade", limit=50, tags=("metal",), query_vec=vec).ids == [sword]
    assert li.search(svc, "blade", limit=50, tags=("wood",), query_vec=vec).ids == []
    # With no free text there is nothing to embed: the cache never asks for a
    # vector, and the store's answer to a bare tag is the text path's, exactly.
    bare = li.search(svc, "", limit=50, tags=("metal",))
    assert bare.ids == [sword] and bare.semantic == frozenset()


def test_a_query_unlike_everything_adds_nothing(svc, monkeypatch):
    _library(svc, monkeypatch)
    vec = li.embed_query(svc, "completely unrelated gibberish zzz")
    assert li.search(svc, "completely unrelated gibberish zzz", limit=50, query_vec=vec).ids == []


def test_the_cap_is_twenty_and_the_floor_is_the_measured_one():
    assert li.SEMANTIC_CAP == 20
    assert li.SEMANTIC_Z_FLOOR == 3.0
    assert li.SEMANTIC_MIN_ROWS == 20


def test_a_search_never_embeds_while_the_store_is_queried(svc, monkeypatch):
    """``search_ids`` and ``semantic_ranked`` are handed a finished vector; the
    only embed call a search makes is the one the caller made beforehand."""
    fake, *_ = _library(svc, monkeypatch)
    vec = li.embed_query(svc, "blade")
    calls_before = len(fake.calls)
    li.search(svc, "blade", limit=50, query_vec=vec)
    assert len(fake.calls) == calls_before


def test_the_time_a_pass_takes_is_bounded_by_its_batch(svc, monkeypatch):
    fakes.install(monkeypatch, svc)
    for i in range(70):
        _done(svc, f"thing {i}")
    started = time.monotonic()
    result = li.index_pending(svc, li.BATCH)
    assert result.indexed == li.BATCH and result.more is True
    assert time.monotonic() - started < 30
