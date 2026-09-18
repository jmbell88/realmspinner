"""Direct unit tests for ``_q_troupe``'s five module-level helpers.

The 2026-09-18 audit, finding troupe-01: ``_socket_specs``, ``_sockets_in_cells``,
``_cells_by_index``, ``_composite_effects`` and ``_atlas_entries`` had no direct
test, only the full (Blender-dependent) ``_charsheet`` pipeline exercising them
end to end. All five are pure/filesystem-light enough to test in isolation --
that is the whole reason each was pulled out of the coroutine, per their own
docstrings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from warlock import _q_troupe as q
from warlock.kernels import charsheet

# --- _socket_specs -----------------------------------------------------------


def test_socket_specs_is_empty_with_no_character():
    assert q._socket_specs(None) == []
    assert q._socket_specs({}) == []


def test_socket_specs_returns_the_archetypes_sockets_in_bone_length_units():
    specs = q._socket_specs({"archetype": "humanoid"})
    names = {s["name"] for s in specs}
    assert "weapon_main" in names
    weapon = next(s for s in specs if s["name"] == "weapon_main")
    assert weapon["bone"] == "hand.R"
    assert weapon["offset"] == [1.0, 0.0, 0.0]
    assert weapon["reach"] == 3.0


def test_socket_specs_is_empty_for_an_unknown_archetype():
    assert q._socket_specs({"archetype": "not-a-real-archetype"}) == []


# --- _sockets_in_cells --------------------------------------------------------


def test_sockets_in_cells_converts_render_size_to_cell_pixels():
    """The regression this finding names.

    The worker projects sockets at ``charsheet.RENDER_SIZE`` (512); a 32px
    cell must see them scaled down by 32/512, the same conversion
    ``point_in_cell`` documents for the ground pivot.
    """
    result = {
        "sockets": {
            "0": {
                "weapon_main": {"x": 256.0, "y": 384.0, "depth": 0.5, "behind": True},
            },
        },
    }
    out = q._sockets_in_cells(result, frame_size=32)
    assert out == {
        0: {
            "weapon_main": {"x": 16.0, "y": 24.0, "depth": 0.5, "behind": True},
        },
    }


def test_sockets_in_cells_is_empty_when_the_result_carries_none():
    assert q._sockets_in_cells({}, frame_size=32) == {}
    assert q._sockets_in_cells("not a dict", frame_size=32) == {}
    assert q._sockets_in_cells({"sockets": "not a dict"}, frame_size=32) == {}


# --- _cells_by_index -----------------------------------------------------------


def _one_movement_layout(frames: int, duration_ms: int) -> charsheet.LayoutSpec:
    return charsheet.resolve_layout(
        {
            "version": 3,
            "movements": [
                {
                    "key": "idle",
                    "frames": frames,
                    "loop": True,
                    "duration_ms": duration_ms,
                    "directions": 1,
                }
            ],
        }
    )


def test_cells_by_index_reports_frame_index_frame_count_and_fps():
    layout = _one_movement_layout(frames=4, duration_ms=100)  # 1000/100 = 10 fps
    out = q._cells_by_index(layout)
    assert out == {
        0: {"frame": 0, "frames": 4, "fps": 10},
        1: {"frame": 1, "frames": 4, "fps": 10},
        2: {"frame": 2, "frames": 4, "fps": 10},
        3: {"frame": 3, "frames": 4, "fps": 10},
    }


def test_cells_by_index_accepts_a_raw_mapping_too():
    resolved = _one_movement_layout(frames=2, duration_ms=500)  # 2 fps
    raw = resolved.as_dict()
    out = q._cells_by_index(raw)
    assert out[0]["fps"] == 2
    assert out[0]["frames"] == 2


# --- _composite_effects --------------------------------------------------------


def test_composite_effects_is_empty_for_an_unknown_family():
    out = q._composite_effects(
        {"family": "not-a-real-species", "theme": "fire"},
        reduced={},
        sockets_px={},
        troupe_layout=None,
        logical=32,
        out_dir=Path("."),
    )
    assert out == {}


def test_composite_effects_is_empty_when_the_theme_has_no_effects():
    from warlock.characters import family

    fam = next(iter(family._FAMILIES.values()))
    theme = fam.themes[0]
    out = q._composite_effects(
        {"family": fam.key, "theme": theme.key},
        reduced={},
        sockets_px={},
        troupe_layout=None,
        logical=32,
        out_dir=Path("."),
    )
    if not theme.effects:
        assert out == {}


# --- _atlas_entries ------------------------------------------------------------


def _write_png(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels, mode="RGBA").save(path)


def test_atlas_entries_returns_sorted_unique_opaque_colours(tmp_path):
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    pixels[0, 0] = [10, 20, 30, 255]
    pixels[0, 1] = [10, 20, 30, 255]
    pixels[1, 0] = [200, 100, 50, 255]
    pixels[1, 1] = [0, 0, 0, 0]  # transparent: contributes nothing
    png = tmp_path / "atlas.png"
    _write_png(png, pixels)

    entries = q._atlas_entries(png, colors=8)
    assert entries == sorted(entries)
    assert set(entries) == {(10, 20, 30), (200, 100, 50)}


def test_atlas_entries_refuses_an_implausibly_large_colour_set(tmp_path):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, size=(16, 16, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    png = tmp_path / "noise.png"
    _write_png(png, pixels)

    # A tiny nominal palette against genuinely noisy pixels: comfortably over
    # the `colors * 4` plausibility ceiling.
    assert q._atlas_entries(png, colors=2) == []


def test_atlas_entries_is_empty_for_a_fully_transparent_image(tmp_path):
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    png = tmp_path / "blank.png"
    _write_png(png, pixels)
    assert q._atlas_entries(png, colors=8) == []
