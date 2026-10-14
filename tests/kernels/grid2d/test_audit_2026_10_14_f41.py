"""The 2026-10-14 audit, finding plotter-tiles-03: a non-finite tile probability."""

from __future__ import annotations

import pytest

from realmspinner.kernels.grid2d.tileset import TileMeta
from realmspinner.kernels.grid2d.wang import WangColour


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_tile_probability_is_refused(bad):
    with pytest.raises(ValueError, match="finite"):
        TileMeta(probability=bad)
    with pytest.raises(ValueError, match="finite"):
        WangColour(name="c", probability=bad)


def test_an_ordinary_probability_still_loads():
    assert TileMeta(probability=0.0).probability == 0.0
    assert TileMeta(probability=2.5).probability == 2.5
    assert WangColour(name="c", probability=1e300).probability == 1e300
