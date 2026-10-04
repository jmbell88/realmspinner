"""Closing the grid2d Low findings plotter-28, -29 and -39 of the 2026-10-03 audit."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from realmspinner.kernels.grid2d import picking
from realmspinner.kernels.grid2d.tileset import TileRect, colour_text

# --- plotter-28 -----------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["#1_2345", "#+12345", "#-12345", "# 12345", "#12345 ", "#1234_567", "#٣23456"]
)
def test_colour_text_refuses_underscores_signs_and_spaces_in_the_digits(bad):
    with pytest.raises(ValueError):
        colour_text(bad, "a colour")


@pytest.mark.parametrize("good", ["#a1B2c3", "#80FFeeDD"])
def test_colour_text_still_accepts_either_case_of_hex(good):
    assert colour_text(good, "a colour") == good


# --- plotter-29 -----------------------------------------------------------------


@pytest.mark.parametrize("handle", ["s", "n"])
def test_resized_leaves_the_axis_the_handle_does_not_touch_alone(handle):
    shape = TileRect(x=4.0, y=2.0, w=0.4, h=8.0)
    at = (0.0, 12.0) if handle == "s" else (0.0, 0.5)
    out = picking.resized(shape, handle, at, 16, 16)
    assert picking.bounds(out)[2] == pytest.approx(0.4), "a vertical drag changed the width"
    assert picking.bounds(out)[0] == pytest.approx(4.0)

    wide = TileRect(x=2.0, y=4.0, w=8.0, h=0.4)
    out = picking.resized(wide, "e" if handle == "s" else "w", (12.0, 0.0), 16, 16)
    assert picking.bounds(out)[3] == pytest.approx(0.4), "a horizontal drag changed the height"


def test_resized_still_enforces_the_minimum_on_the_axis_it_moves():
    shape = TileRect(x=4.0, y=2.0, w=0.4, h=8.0)
    out = picking.resized(shape, "s", (0.0, 2.2), 16, 16)
    assert picking.bounds(out)[3] == pytest.approx(picking.MIN_SIDE)


# --- plotter-39 -----------------------------------------------------------------


def test_the_grid2d_leaf_imports_no_window_library():
    from _pure_packages import WINDOW_ROOTS

    root = Path(picking.__file__).parent
    found: dict[str, set[str]] = {}
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in WINDOW_ROOTS:
                    found.setdefault(path.name, set()).add(name)
    assert not found, f"kernels/grid2d imports a window library: {found}"


# --- plotter-38 -----------------------------------------------------------------


def test_roles_docstring_does_not_claim_no_terrain_set_is_generated():
    from realmspinner.kernels.grid2d import roles

    doc = " ".join(roles.__doc__.split())
    assert "Nothing in Realmspinner has produced a 47-column terrain set" not in doc
    assert "blob_atlas" in doc, "the docstring must name the generator that still exists"
