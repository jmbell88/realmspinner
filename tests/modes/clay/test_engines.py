"""Per-engine export conventions: collider naming, LOD naming, the readiness
gate and the axis/scale story -- tranche 7's other pure-kernel half. See
``engines.py``'s own module docstring for the engine facts these encode and
their confidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.kernels.mesh import engines as eng
from warlock.kernels.mesh import readiness
from warlock.kernels.mesh.elements import OpError

# --- registry shape -------------------------------------------------------


def test_every_engine_profile_points_at_a_real_readiness_profile():
    """The gate this tranche's brief asks for: an ``EngineProfile`` naming a
    ``readiness_profile`` key that does not exist in ``readiness.PROFILES``
    would silently break ``readiness.validate(doc, profile=eng.readiness_profile)``
    for whichever caller trusted this registry."""
    for key, profile in eng.ENGINES.items():
        assert profile.readiness_profile in readiness.PROFILES, (
            f"{key}'s readiness_profile {profile.readiness_profile!r} is not a real profile"
        )


def test_engines_cover_exactly_the_four_named_targets():
    assert set(eng.ENGINES) == {"godot4", "unity", "unreal", "webgl"}


def test_every_engine_profile_key_matches_its_registry_key():
    for key, profile in eng.ENGINES.items():
        assert profile.key == key


def test_glb_conversion_is_the_identity_for_every_engine():
    """Every target's own glTF importer performs whatever conversion it
    needs on the way in -- see the module docstring's opening paragraph."""
    for key, profile in eng.ENGINES.items():
        assert np.array_equal(profile.glb_conversion, np.eye(4)), key


def test_obj_conversion_is_the_identity_for_engines_already_in_gltf_convention():
    """Godot 4 and the WebGL profile match glTF's Y-up/right-handed/metres
    convention natively, so neither needs an OBJ conversion either."""
    for key in ("godot4", "webgl"):
        assert np.array_equal(eng.ENGINES[key].obj_conversion, np.eye(4)), key


def test_obj_conversion_is_not_the_identity_where_the_convention_differs():
    """Unity (handedness only) and Unreal (axis, scale and handedness) do
    differ from glTF's convention, and OBJ carries no metadata an importer
    could use to correct for it -- see the module docstring's Unreal/Unity
    entries for what is well documented here and what is this session's own
    best-effort judgement call."""
    for key in ("unity", "unreal"):
        assert not np.array_equal(eng.ENGINES[key].obj_conversion, np.eye(4)), key


def test_unity_obj_conversion_only_mirrors_handedness():
    """Unity already matches glTF's up axis and unit scale (both metres,
    both Y-up); only the right-vs-left-handedness differs, which a single Z
    negation corrects -- see ``engines.py``'s Unity docstring entry."""
    m = eng.ENGINES["unity"].obj_conversion
    expected = np.eye(4)
    expected[2, 2] = -1.0
    assert np.array_equal(m, expected)


def test_unreal_obj_conversion_scales_metres_to_centimetres():
    """High-confidence part of the Unreal OBJ conversion: whatever the axis
    mapping, a unit vector along Clay's own Y should come out 100x longer."""
    m = eng.ENGINES["unreal"].obj_conversion
    v = m @ np.array([0.0, 1.0, 0.0, 0.0])
    assert np.linalg.norm(v[:3]) == pytest.approx(100.0)


# --- collider naming ------------------------------------------------------


def test_unreal_collider_naming_uses_the_documented_prefixes():
    assert eng.collider_name("unreal", "convex", "Crate", 0) == "UCX_Crate_00"
    assert eng.collider_name("unreal", "box", "Crate", 1) == "UBX_Crate_01"
    assert eng.collider_name("unreal", "sphere", "Crate", 2) == "USP_Crate_02"
    assert eng.collider_name("unreal", "capsule", "Crate", 3) == "UCP_Crate_03"
    assert eng.collider_name("unreal", "compound", "Crate", 4) == "UCX_Crate_04"


def test_godot_collider_naming_is_the_same_convex_only_suffix_for_every_kind():
    """Godot has no per-primitive naming distinction -- every kind this
    package fits is convex, so every one of them round-trips through
    ``-convcolonly``; see ``engines.py``'s Godot docstring entry."""
    names = {
        kind: eng.collider_name("godot4", kind, "Rock", 0)
        for kind in ("box", "sphere", "capsule", "convex", "compound")
    }
    assert all(name.endswith("-convcolonly") for name in names.values())
    assert len(set(names.values())) == 1  # every kind produces the identical name at index 0


@pytest.mark.parametrize(
    ("engine_key", "kind", "mesh_name", "index"),
    [
        ("unreal", "convex", "Crate", 0),
        ("unreal", "box", "Anvil", 12),
        ("godot4", "sphere", "Rock", 3),
        ("unity", "capsule", "Barrel", 7),
        ("webgl", "compound", "Prop", 0),
    ],
)
def test_collider_naming_round_trips(engine_key: str, kind: str, mesh_name: str, index: int):
    name = eng.collider_name(engine_key, kind, mesh_name, index)
    assert eng.is_collider_name(engine_key, name)


def test_collider_name_refuses_an_unsupported_kind():
    with pytest.raises(OpError):
        eng.collider_name("unreal", "not-a-real-kind", "Crate", 0)


def test_collider_name_refuses_an_unknown_engine():
    with pytest.raises(OpError):
        eng.collider_name("cryengine", "convex", "Crate", 0)


def test_is_collider_name_recognises_godots_whole_documented_vocabulary():
    """Generous to Godot's whole suffix vocabulary, not only the one
    :func:`engines.collider_name` itself emits -- ``-col``, ``-convcol`` and
    the rest are still legitimate Godot collider-node suffixes."""
    for suffix in ("-col", "-colonly", "-convcol", "-convcolonly"):
        assert eng.is_collider_name("godot4", f"Rock{suffix}")
    assert not eng.is_collider_name("godot4", "Rock")


def test_is_collider_name_is_false_for_an_ordinary_mesh_name():
    assert not eng.is_collider_name("unreal", "Crate")
    assert not eng.is_collider_name("unity", "Crate")
    assert not eng.is_collider_name("webgl", "Crate")


# --- LOD naming -------------------------------------------------------------


def test_unreal_and_unity_lod_naming():
    assert eng.lod_name("unreal", "Crate", 0) == "Crate_LOD0"
    assert eng.lod_name("unreal", "Crate", 3) == "Crate_LOD3"
    assert eng.lod_name("unity", "Crate", 1) == "Crate_LOD1"


@pytest.mark.parametrize("engine_key", ["godot4", "webgl"])
def test_lod_name_refuses_where_there_is_no_naming_convention(engine_key: str):
    """Godot generates its own LODs on import; WebGL has no naming
    convention to hook into at all -- two different reasons, one refusal
    each, see ``engines.py``'s own module docstring for the distinction."""
    with pytest.raises(OpError):
        eng.lod_name(engine_key, "Rock", 0)


def test_lod_name_refuses_an_unknown_engine():
    with pytest.raises(OpError):
        eng.lod_name("cryengine", "Crate", 0)
