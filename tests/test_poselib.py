"""What is left of ``poselib``: the pose constants and the user's clip folder.

``poselib`` was the global pose library (records, names, a preview cache); that
half left with Poser (``retired.py``, the 2026-10-13 Clay rig plan, Tranche 5).
What stays is shared by the single-mesh pose editors and the clip library, and
is decidable with a bare ``data_dir`` and no service -- which is the point of it
being its own module.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from realmspinner import poselib


def _config(tmp_path):
    return SimpleNamespace(data_dir=tmp_path)


# --- the root offset --------------------------------------------------------


def test_a_root_offset_is_normalized_to_three_floats():
    assert poselib.validate_root_translation((1, 0, -2)) == [1.0, 0.0, -2.0]


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ([1.0, 0.0], "3-vector"),
        ("abc", "3-vector"),
        (["x", 0.0, 0.0], "not numeric"),
        ([3.0, 0.0, 0.0], "character heights"),
        ([0.0, -2.5, 0.0], "character heights"),
    ],
)
def test_root_offset_refusals_carry_their_field(raw, match):
    with pytest.raises(poselib.RecordError, match=match) as info:
        poselib.validate_root_translation(raw)
    assert info.value.field == "root_translation"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_root_translation_is_finite_checked_before_range_checked(bad):
    """abs(nan) <= 2.0 is False, so the range message would blame a value that
    is not a number -- the finite check has to win."""
    with pytest.raises(poselib.RecordError, match="not numeric") as info:
        poselib.validate_root_translation([bad, 0.0, 0.0])
    assert info.value.field == "root_translation"


def test_record_error_is_a_value_error():
    """The GuidanceError shape: every caller that already catches ValueError
    keeps working, and invalid_from picks the field up off the instance."""
    assert issubclass(poselib.RecordError, ValueError)


# --- mirroring --------------------------------------------------------------


def test_mirror_root_translation_reflects_x_only():
    assert poselib.mirror_root_translation([0.5, 1.0, -0.25]) == [-0.5, 1.0, -0.25]


def test_mirroring_the_root_twice_is_the_identity():
    v = [0.3, -0.2, 0.9]
    assert poselib.mirror_root_translation(poselib.mirror_root_translation(v)) == v


def test_unit_box_constants():
    """Pinned because two things stand on them: fit_template's place() is the
    identity over exactly this box, and a root_translation is in
    character-height units only while the armature is exactly 1 tall."""
    assert poselib.UNIT_LO == (-0.5, -0.5, 0.0)
    assert poselib.UNIT_HI == (0.5, 0.5, 1.0)
    assert poselib.MAX_ROOT_TRANSLATION == 2.0


# --- the clip folder --------------------------------------------------------


def test_the_user_clip_folder_lives_under_data_dir_poser(tmp_path):
    config = _config(tmp_path)
    assert poselib.poser_dir(config) == tmp_path / "poser"
    assert poselib.clip_dir(config) == tmp_path / "poser" / "clips"
    assert poselib.clip_path(config, "humanoid") == tmp_path / "poser" / "clips" / "humanoid.json"


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "a b", "x.json"])
def test_clip_path_refuses_a_key_that_is_not_a_template_key(tmp_path, bad):
    with pytest.raises(poselib.RecordError) as info:
        poselib.clip_path(_config(tmp_path), bad)
    assert info.value.field == "template"


def test_the_pose_library_half_is_gone():
    """The library half left with Poser: nothing in this module lists, saves or
    applies a shared pose, so a caller that still reaches for one fails loudly."""
    for name in (
        "validate_record",
        "save_record",
        "read_record",
        "list_records",
        "delete_record",
        "find_name",
        "next_copy_name",
        "library_dir",
        "preview_path",
        "template_digest",
        "MAX_LIBRARY_POSES",
    ):
        assert not hasattr(poselib, name), name
