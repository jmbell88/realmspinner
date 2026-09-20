"""``mason/refs.py``: the two ways to name a mesh's source, and their caching key.

Every test name here is a claim about what makes two references the same
upload -- see ``refs.ref_key``'s own docstring for why that question is the
whole point of this module.
"""

from __future__ import annotations

import math

import pytest

from realmspinner.studio.modes.mason.engine import refs


def test_two_primitive_refs_built_from_equal_params_one_tuples_one_lists_are_equal() -> None:
    tupled = refs.primitive_ref("box", {"size": (1.0, 1.0, 1.0)})
    listed = refs.primitive_ref("box", {"size": [1.0, 1.0, 1.0]})
    assert tupled == listed
    assert hash(tupled) == hash(listed)


def test_an_int_and_a_float_of_the_same_value_produce_one_ref_key() -> None:
    """A spinbox that emits ``16`` one frame and ``16.0`` the next must not
    silently double the GPU upload for the same shape."""
    from_int = refs.primitive_ref("cylinder", {"segments": 16})
    from_float = refs.primitive_ref("cylinder", {"segments": 16.0})
    assert refs.ref_key(from_int) == refs.ref_key(from_float)


def test_params_in_a_different_key_order_still_produce_one_ref_key() -> None:
    a = refs.primitive_ref("box", {"size": (1.0, 2.0, 3.0), "extra": 5.0})
    b = refs.primitive_ref("box", {"extra": 5.0, "size": (1.0, 2.0, 3.0)})
    assert refs.ref_key(a) == refs.ref_key(b)


def test_a_library_ref_renamed_is_the_same_ref_key_but_not_the_same_object() -> None:
    """A job rename does not double the upload: the two refs disagree only on
    the human-facing ``name``, which ``ref_key`` deliberately drops."""
    before = refs.LibraryRef(job_id="job-1", name="Barrel")
    after = refs.LibraryRef(job_id="job-1", name="Barrel (renamed)")
    assert before != after  # the objects really do differ
    assert refs.ref_key(before) == refs.ref_key(after)  # but not the geometry they name


def test_a_library_ref_relinked_to_a_new_hash_is_still_the_same_ref_key() -> None:
    """``sha256`` exists only for relink to offer a match, never for identity."""
    before = refs.LibraryRef(job_id="job-1", sha256="")
    after = refs.LibraryRef(job_id="job-1", sha256="deadbeef")
    assert refs.ref_key(before) == refs.ref_key(after)


def test_two_library_refs_at_different_jobs_are_different_ref_keys() -> None:
    a = refs.LibraryRef(job_id="job-1")
    b = refs.LibraryRef(job_id="job-2")
    assert refs.ref_key(a) != refs.ref_key(b)


def test_ref_key_of_none_answers_rather_than_raising() -> None:
    """A ``GroupNode`` has no ref at all; a caller grouping placed items by
    shared geometry must be able to ask for its key without a crash."""
    key = refs.ref_key(None)
    assert isinstance(key, tuple)


def test_ref_key_of_none_is_distinct_from_any_real_ref() -> None:
    none_key = refs.ref_key(None)
    primitive_key = refs.ref_key(refs.primitive_ref("box", {}))
    library_key = refs.ref_key(refs.LibraryRef(job_id="job-1"))
    assert none_key != primitive_key
    assert none_key != library_key


def test_a_nan_in_params_is_refused_naming_the_parameter() -> None:
    with pytest.raises(ValueError, match="taper"):
        refs.primitive_ref("sweep", {"taper": math.nan})


def test_an_infinite_value_in_params_is_refused_naming_the_parameter() -> None:
    with pytest.raises(ValueError, match="depth"):
        refs.primitive_ref("sweep", {"depth": math.inf})


def test_a_nan_nested_inside_a_profile_is_refused_naming_the_parameter() -> None:
    """``lathe``'s ``profile`` is a list of ``[x, y]`` pairs -- the refusal
    must reach into that nesting, not just a params dict's top level."""
    with pytest.raises(ValueError, match="profile"):
        refs.primitive_ref("lathe", {"profile": [[0.0, 0.0], [math.nan, 1.0]]})


def test_a_non_string_param_key_is_refused() -> None:
    with pytest.raises(ValueError, match=r"key 1 is not a string"):
        refs.primitive_ref("box", {1: 2.0})


def test_an_unsupported_param_value_type_is_refused() -> None:
    with pytest.raises(ValueError):
        refs.primitive_ref("box", {"size": object()})


def test_as_params_round_trips_through_primitive_ref_to_an_equal_ref() -> None:
    original = refs.primitive_ref(
        "lathe", {"profile": [[0.0, 0.0], [0.5, 1.0], [0.0, 2.0]], "segments": 16}
    )
    rebuilt = refs.primitive_ref(original.generator, original.as_params())
    assert rebuilt == original


def test_as_params_returns_nested_tuples_not_lists() -> None:
    """Safe rather than lossy: every generator indexes a profile positionally,
    and a tuple supports the same positional indexing a list does."""
    ref = refs.primitive_ref("lathe", {"profile": [[0.0, 0.0], [0.5, 1.0]]})
    params = ref.as_params()
    assert isinstance(params["profile"], tuple)
    assert isinstance(params["profile"][0], tuple)
    assert params["profile"][0] == (0.0, 0.0)


def test_as_params_is_a_plain_dict_a_caller_can_splat() -> None:
    ref = refs.primitive_ref("box", {"size": (2.0, 3.0, 4.0)})
    params = ref.as_params()
    assert params == {"size": (2.0, 3.0, 4.0)}


def test_a_bool_param_survives_normalization_as_a_bool_not_a_number() -> None:
    """No generator takes a flag today, but a future one that does should not
    have ``True`` silently become ``1.0`` in a reported value."""
    ref = refs.primitive_ref("box", {"flag": True})
    assert ref.as_params()["flag"] is True
