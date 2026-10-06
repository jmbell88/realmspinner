"""Phase 2's pure retrieval half: ``familiar/embed_index.py``.

No server, no Manual on disk -- vectors are built from a seeded generator and
chunks are a tiny dataclass with the four attributes ``tree_sha`` reads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np
import pytest

from realmspinner.familiar import embed_index as ei


def _unit(rng, n, d):
    m = rng.standard_normal((n, d)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


@dataclass(frozen=True)
class _Chunk:
    chapter: str
    anchor: str | None
    title_path: str
    text: str


# --- truncate ---------------------------------------------------------------


def test_truncate_equals_slicing_then_renormalising_and_keeps_rows_unit_norm():
    rng = np.random.default_rng(1)
    full = _unit(rng, 5, 768)
    out = ei.truncate(full, ei.LIBRARY_DIM)
    assert out.shape == (5, 256) and out.dtype == np.float32
    expect = full[:, :256] / np.linalg.norm(full[:, :256], axis=1, keepdims=True)
    np.testing.assert_allclose(out, expect, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    # The point of renormalising: the bare prefix is NOT unit length.
    assert np.linalg.norm(full[:, :256], axis=1).max() < 0.99


def test_truncate_to_the_native_width_is_the_identity_on_unit_rows():
    full = _unit(np.random.default_rng(2), 3, 768)
    np.testing.assert_allclose(ei.truncate(full, ei.MANUAL_DIM), full, atol=1e-6)


def test_truncate_takes_a_single_vector_and_returns_one():
    v = _unit(np.random.default_rng(3), 1, 768)[0]
    out = ei.truncate(v, 128)
    assert out.shape == (128,)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5


@pytest.mark.parametrize("dim", [0, 64, 300, 1024, -256, True])
def test_truncate_refuses_a_dimension_the_model_does_not_support_by_name(dim):
    with pytest.raises(ValueError, match="does not support an embedding dimension"):
        ei.truncate(np.ones((2, 768), dtype=np.float32), dim)


def test_truncate_refuses_a_vector_narrower_than_the_requested_width():
    with pytest.raises(ValueError, match="cannot truncate 256-d vectors to 512"):
        ei.truncate(np.ones((2, 256), dtype=np.float32), 512)


def test_truncate_leaves_a_zero_row_zero_not_nan():
    m = np.zeros((1, 768), dtype=np.float32)
    assert not np.isnan(ei.truncate(m, 256)).any()


def test_the_manual_and_library_widths_are_the_documented_ones():
    assert (ei.MANUAL_DIM, ei.LIBRARY_DIM) == (768, 256)
    assert set(ei.SUPPORTED_DIMS) == {768, 512, 256, 128}


# --- top_k ------------------------------------------------------------------


def test_top_k_orders_best_first_with_cosine_scores():
    rng = np.random.default_rng(4)
    m = _unit(rng, 20, 32)
    q = m[7].copy()
    hits = ei.top_k(q, m, 5)
    assert hits[0][0] == 7 and hits[0][1] == pytest.approx(1.0, abs=1e-5)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    brute = np.argsort(-(m @ q), kind="stable")[:5]
    assert [i for i, _ in hits] == [int(i) for i in brute]


def test_top_k_breaks_ties_by_the_lower_row_index():
    row = _unit(np.random.default_rng(5), 1, 16)[0]
    other = np.roll(row, 1)
    m = np.stack([other, row, row, row])  # rows 1..3 identical
    hits = ei.top_k(row, m, 4)
    assert [i for i, _ in hits[:3]] == [1, 2, 3]


def test_top_k_on_an_empty_matrix_is_empty():
    assert ei.top_k(np.ones(8, dtype=np.float32), np.empty((0, 8), dtype=np.float32), 3) == []


def test_top_k_with_k_larger_than_the_rows_returns_every_row():
    m = _unit(np.random.default_rng(6), 4, 8)
    assert len(ei.top_k(m[0], m, 100)) == 4


def test_top_k_with_a_non_positive_k_is_empty():
    m = _unit(np.random.default_rng(7), 4, 8)
    assert ei.top_k(m[0], m, 0) == [] and ei.top_k(m[0], m, -1) == []


def test_top_k_renormalises_an_unnormalised_matrix_instead_of_ranking_by_magnitude():
    q = np.array([1.0, 0.0], dtype=np.float32)
    m = np.array([[10.0, 10.0], [0.5, 0.0]], dtype=np.float32)  # row 1 is the aligned one
    assert ei.top_k(q, m, 2)[0][0] == 1


def test_top_k_refuses_a_width_mismatch():
    with pytest.raises(ValueError, match="query is 4-d"):
        ei.top_k(np.ones(4, dtype=np.float32), np.ones((2, 8), dtype=np.float32), 1)


def test_top_k_returns_plain_python_numbers():
    m = _unit(np.random.default_rng(8), 3, 8)
    index, score = ei.top_k(m[0], m, 1)[0]
    assert type(index) is int and type(score) is float


# --- rrf --------------------------------------------------------------------


def test_rrf_an_id_high_in_both_rankings_beats_an_id_first_in_only_one():
    fused = ei.rrf([["a", "b", "c"], ["z", "b", "y"]])
    ids = [i for i, _ in fused]
    assert ids[0] == "b"  # second in both beats first in only one
    assert ids.index("b") < ids.index("a") and ids.index("b") < ids.index("z")


def test_rrf_scores_are_the_reciprocal_rank_sum():
    fused = dict(ei.rrf([["a", "b"], ["b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_an_id_absent_from_a_list_contributes_nothing_from_it():
    fused = dict(ei.rrf([["a"], ["b"], []]))
    assert fused["a"] == fused["b"] == pytest.approx(1 / 61)


def test_rrf_is_stable_on_ties_best_rank_then_id():
    # a and b tie on score (1/61 + 1/62 each) and on best rank (1): id decides.
    fused = ei.rrf([["a", "b"], ["b", "a"]])
    assert [i for i, _ in fused] == ["a", "b"]
    assert [i for i, _ in ei.rrf([["x"], ["y"]])] == ["x", "y"]
    # same score, different best rank: the one that was first somewhere wins.
    tied = ei.rrf([["p", "q"], ["r", "p"]], k=0)  # p: 1+1/2, q: 1/2, r: 1 -> p first
    assert [i for i, _ in tied][0] == "p"
    same = ei.rrf([["m", "n"], ["n", "m"], ["o"]])  # m,n identical sums
    assert [i for i, _ in same][:2] == ["m", "n"]


def test_rrf_does_not_depend_on_the_order_the_rankings_are_given():
    r1, r2, r3 = ["a", "b", "c", "d"], ["d", "c", "b", "a"], ["b", "d", "a", "c"]
    assert ei.rrf([r1, r2, r3]) == ei.rrf([r3, r1, r2])


def test_rrf_counts_a_repeated_id_inside_one_ranking_once():
    assert dict(ei.rrf([["a", "a", "a"]]))["a"] == pytest.approx(1 / 61)


def test_rrf_limit_cuts_the_result_and_empty_input_is_empty():
    assert len(ei.rrf([list("abcdef")], limit=2)) == 2
    assert ei.rrf([]) == [] and ei.rrf([[], []]) == []


def test_rrf_works_over_row_indices_from_top_k():
    assert ei.rrf([[3, 1, 2], [1, 3]])[0][0] in {1, 3}


def test_rrf_refuses_a_negative_k():
    with pytest.raises(ValueError):
        ei.rrf([["a"]], k=-1)


# --- tree_sha ---------------------------------------------------------------


_CHUNKS = [
    _Chunk("01-intro", "start", "Intro", "Hello."),
    _Chunk("02-clay", None, "Clay > Lathe", "Spin a profile."),
]


def test_tree_sha_is_stable_when_nothing_changes():
    assert ei.tree_sha(_CHUNKS) == ei.tree_sha(list(_CHUNKS))
    assert len(ei.tree_sha(_CHUNKS)) == 64


@pytest.mark.parametrize("field", ["chapter", "anchor", "title_path", "text"])
def test_tree_sha_changes_when_one_field_of_one_chunk_changes(field):
    edited = [_CHUNKS[0], replace(_CHUNKS[1], **{field: "changed"})]
    assert ei.tree_sha(edited) != ei.tree_sha(_CHUNKS)


def test_tree_sha_distinguishes_a_missing_anchor_from_an_empty_one_and_order():
    a = [replace(_CHUNKS[0], anchor=None)]
    b = [replace(_CHUNKS[0], anchor="")]
    assert ei.tree_sha(a) != ei.tree_sha(b)
    assert ei.tree_sha(_CHUNKS) != ei.tree_sha(list(reversed(_CHUNKS)))


def test_tree_sha_does_not_let_text_bleed_into_the_next_field():
    a = [_Chunk("c", None, "ab", "c")]
    b = [_Chunk("c", None, "a", "bc")]
    assert ei.tree_sha(a) != ei.tree_sha(b)


# --- cache ------------------------------------------------------------------

_TREE = "a" * 64
_MODEL = "b" * 64


def _key(dim=256, tree=_TREE, model=_MODEL, kind="manual"):
    return ei.cache_key(tree, model, dim, kind)


def test_cache_key_changes_with_each_identity():
    base = _key()
    assert _key(tree="c" * 64) != base
    assert _key(model="c" * 64) != base
    assert _key(dim=768) != base
    assert _key(kind="library") != base
    assert _key() == base


@pytest.mark.parametrize("bad", ["../x", "a/b", "", "a-b", "a b"])
def test_cache_key_refuses_a_sha_that_could_escape_the_directory(bad):
    with pytest.raises(ValueError):
        ei.cache_key(bad, _MODEL, 256)


def test_cache_key_refuses_an_unsupported_dimension():
    with pytest.raises(ValueError, match="does not support"):
        ei.cache_key(_TREE, _MODEL, 300)


def test_the_cache_round_trips_a_matrix_exactly(tmp_path):
    m = _unit(np.random.default_rng(9), 12, 256)
    path = ei.save_matrix(tmp_path / "cache", _key(), m)
    assert path.parent == tmp_path / "cache" and path.name == _key() + ".npy"
    out = ei.load_matrix(tmp_path / "cache", _key(), rows=12)
    assert out is not None and out.dtype == np.float32
    np.testing.assert_array_equal(out, m)


def test_a_missing_file_is_a_miss(tmp_path):
    assert ei.load_matrix(tmp_path, _key()) is None
    assert ei.load_matrix(tmp_path / "no-such-dir", _key()) is None


def test_a_truncated_file_reads_as_a_miss_and_does_not_raise(tmp_path):
    m = _unit(np.random.default_rng(10), 10, 256)
    path = ei.save_matrix(tmp_path, _key(), m)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    assert ei.load_matrix(tmp_path, _key()) is None


def test_a_garbage_file_reads_as_a_miss(tmp_path):
    (tmp_path / (_key() + ".npy")).write_bytes(b"this is not a numpy file at all")
    assert ei.load_matrix(tmp_path, _key()) is None
    (tmp_path / (_key() + ".npy")).write_bytes(b"")
    assert ei.load_matrix(tmp_path, _key()) is None


def test_a_wrong_width_file_reads_as_a_miss(tmp_path):
    # Written under a 768 key, then renamed into the 256 key's slot.
    wide = ei.save_matrix(tmp_path, _key(768), _unit(np.random.default_rng(11), 4, 768))
    wide.replace(tmp_path / (_key(256) + ".npy"))
    assert ei.load_matrix(tmp_path, _key(256)) is None


def test_a_wrong_row_count_reads_as_a_miss_when_the_caller_knows_the_corpus_size(tmp_path):
    ei.save_matrix(tmp_path, _key(), _unit(np.random.default_rng(12), 10, 256))
    assert ei.load_matrix(tmp_path, _key(), rows=11) is None
    assert ei.load_matrix(tmp_path, _key(), rows=10) is not None


def test_a_wrong_dtype_or_rank_file_reads_as_a_miss(tmp_path):
    np.save(tmp_path / (_key() + ".npy"), np.zeros((3, 256), dtype=np.float64))
    assert ei.load_matrix(tmp_path, _key()) is None
    np.save(tmp_path / (_key() + ".npy"), np.zeros((256,), dtype=np.float32))
    assert ei.load_matrix(tmp_path, _key()) is None


def test_a_pickled_object_array_is_a_miss_not_code_execution(tmp_path):
    np.save(tmp_path / (_key() + ".npy"), np.array([{"a": 1}], dtype=object), allow_pickle=True)
    assert ei.load_matrix(tmp_path, _key()) is None


def test_a_non_finite_cache_is_a_miss(tmp_path):
    bad = np.zeros((2, 256), dtype=np.float32)
    bad[0, 0] = np.nan
    np.save(tmp_path / (_key() + ".npy"), bad)
    assert ei.load_matrix(tmp_path, _key()) is None


def test_save_refuses_a_matrix_whose_width_is_not_the_keys(tmp_path):
    with pytest.raises(ValueError, match="256-d"):
        ei.save_matrix(tmp_path, _key(256), np.zeros((2, 768), dtype=np.float32))
    with pytest.raises(ValueError):
        ei.save_matrix(tmp_path, _key(256), np.zeros((256,), dtype=np.float32))


def test_a_crash_between_the_temp_write_and_the_replace_leaves_nothing_served(
    tmp_path, monkeypatch
):
    """The 'never a partial file under the served name' claim, in both
    shapes: with no earlier file there is still none, and with an earlier good
    file it is the earlier one that is served, untouched."""
    m = _unit(np.random.default_rng(13), 6, 256)

    def boom(src, dst):
        raise OSError("disk went away")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        ei.save_matrix(tmp_path, _key(), m)
    assert not (tmp_path / (_key() + ".npy")).exists()
    assert list(tmp_path.iterdir()) == []  # the staged temp is cleaned up too
    monkeypatch.undo()

    ei.save_matrix(tmp_path, _key(), m)
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        ei.save_matrix(tmp_path, _key(), m * 0 + 1)
    monkeypatch.undo()
    np.testing.assert_array_equal(ei.load_matrix(tmp_path, _key()), m)


def test_prune_superseded_deletes_old_files_of_the_same_kind_and_width_only(tmp_path):
    m256 = np.zeros((1, 256), dtype=np.float32)
    m768 = np.zeros((1, 768), dtype=np.float32)
    keep = _key(256, tree="1" * 64)
    old1, old2 = _key(256, tree="2" * 64), _key(256, model="3" * 64)
    other_dim = _key(768, tree="2" * 64)
    other_kind = _key(256, tree="2" * 64, kind="library")
    for k in (keep, old1, old2, other_kind):
        ei.save_matrix(tmp_path, k, m256)
    ei.save_matrix(tmp_path, other_dim, m768)
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "weird.npy").write_bytes(b"x")

    removed = ei.prune_superseded(tmp_path, keep)

    assert sorted(removed) == sorted([old1 + ".npy", old2 + ".npy"])
    left = {p.name for p in tmp_path.iterdir()}
    assert left == {
        keep + ".npy",
        other_dim + ".npy",
        other_kind + ".npy",
        "notes.txt",
        "weird.npy",
    }


def test_prune_superseded_on_a_missing_directory_is_a_no_op(tmp_path):
    assert ei.prune_superseded(tmp_path / "nope", _key()) == []
