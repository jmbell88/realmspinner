"""The render budget. ``uv run pytest -m perf -n 0``.

A regenerate runs in a task, so the frame loop never waits on it, but the
user does: a slider nudged and a second later the timeline updates is the
whole feel of the feature. The budget is the nine-layer fireball at the
default 128px / 4x, measured at ~130 ms a frame on the development machine
after the two cost rules in ``prims/__init__`` (noise at logical resolution,
discs in their windows); the first draft was 1,100 ms. The floor is set at
three times the measurement so a modest CI box passes and a regression that
doubles the cost fails here rather than under somebody's cursor.

Excluded from the parallel run for the reason every ``perf`` case is.
"""

from __future__ import annotations

import time

import pytest
from _recipes import FIREBALL

from warlock.studio.inker import flourish

MAX_MS_PER_FRAME = 400.0


def _default_size_fireball() -> flourish.Recipe:
    raw = dict(FIREBALL)
    raw["size"] = [128, 128]
    raw["supersample"] = 4
    # Scale the geometry up with the canvas so the work is representative.
    scaled = []
    for layer in raw["layers"]:
        params = dict(layer.get("params", {}))
        for key in ("radius", "size", "width", "height"):
            if isinstance(params.get(key), (int, float)):
                params[key] = params[key] * 2.5
        if "x" in params:
            params["x"] = {"keys": [[0, -40], [1, 40]]}
        scaled.append({**layer, "params": params})
    raw["layers"] = scaled
    return flourish.from_dict(raw)


@pytest.mark.perf
def test_a_default_size_fireball_frame_renders_inside_the_budget():
    rec = _default_size_fireball()
    flourish.render_frame(rec, 5)  # warm the primitive imports
    frames = list(range(rec.frame_count))
    start = time.perf_counter()
    for f in frames:
        flourish.to_uint8(flourish.render_frame(rec, f), rec.supersample)
    per_frame = (time.perf_counter() - start) / len(frames) * 1000.0
    assert per_frame < MAX_MS_PER_FRAME, f"{per_frame:.0f} ms/frame"


def test_check_bake_cost_refuses_a_particle_heavy_layer_that_would_blow_the_patience_budget():
    """The 2026-09-11 audit, finding inker-06: ``bake_cost`` multiplied only
    ``width * height * supersample**2 * frame_count * directions *
    len(layers)`` and never looked at what a layer's own parameters asked
    for, so a "particles" layer at its own published slider maximum
    (count=400, size=64) cost the formula exactly what an empty layer costs.

    This is the audit's own measured recipe -- three such layers, 100 frames,
    128px, 4x supersample, one direction -- which the *old* formula priced at
    78,643,200 of the 212,000,000-unit budget (comfortably under) while
    actually taking on the order of 2.19s/frame to render, ~656s total
    against the ~90s of patience ``MAX_BAKE_COST`` is calibrated to. A fixed
    cost model must refuse this recipe rather than wave it through.
    """
    from warlock.studio.inker.flourish import recipe as R

    rec = flourish.clamp(
        flourish.Recipe(
            width=128,
            height=128,
            supersample=4,
            directions=1,
            phases=(flourish.Phase("main", 100),),
            layers=tuple(
                flourish.Layer(uid=i, kind="particles", params={"count": 400, "size": 64.0})
                for i in range(3)
            ),
        )
    )
    # The layers really did clamp to the slider maximum this claims, not to
    # something smaller that would pass for an unrelated reason.
    for layer in rec.layers:
        assert layer.params["count"] == 400
        assert layer.params["size"] == 64.0
    with pytest.raises(ValueError, match="pixels of frames"):
        R.check_bake_cost(rec)


def test_check_bake_cost_refuses_a_trail_layer_that_would_blow_the_patience_budget():
    """The 2026-09-16 audit: ``_COST_PARAMS`` weighted ``particles``/``smoke``
    by their own ``count``/``size`` (inker-06, 2026-09-11) but never gained a
    ``trail`` entry, even though ``trail.render()`` loops once per ``samples``
    and paints a window sized by ``radius`` each time -- the identical
    "count x window-area" cost shape. A trail at its own published maximum
    (samples=64, radius=256) measured about 13x the wall time of an empty
    layer while ``bake_cost`` charged it the same flat 1.0 unit.

    Three such layers, 100 frames, 128px, 4x supersample, one direction --
    the same recipe shape as the particles regression above -- prices far
    past ``MAX_BAKE_COST`` once ``trail`` is weighted correctly, and must be
    refused rather than waved through.
    """
    from warlock.studio.inker.flourish import recipe as R

    rec = flourish.clamp(
        flourish.Recipe(
            width=128,
            height=128,
            supersample=4,
            directions=1,
            phases=(flourish.Phase("main", 100),),
            layers=tuple(
                flourish.Layer(uid=i, kind="trail", params={"samples": 64, "radius": 256.0})
                for i in range(3)
            ),
        )
    )
    # The layers really did clamp to the slider maximum this claims.
    for layer in rec.layers:
        assert layer.params["samples"] == 64
        assert layer.params["radius"] == 256.0
    with pytest.raises(ValueError, match="pixels of frames"):
        R.check_bake_cost(rec)


def test_check_bake_cost_still_allows_the_shipped_fireball_preset():
    """The fireball preset (``presets/fireball.json``) is what
    ``test_a_default_size_fireball_frame_renders_inside_the_budget`` scales
    up, and it is a real, shipped effect -- a cost model that closes the hole
    ``inker-06`` found must not also refuse the effect the manual's own
    "Casting a spell" chapter walks the reader through inserting. See
    ``docs/measurements`` for this preset's cost before and after the fix, if
    recorded there; here it is enough that it still fits.
    """
    from warlock.studio.inker.flourish import presets
    from warlock.studio.inker.flourish import recipe as R

    rec = presets.load("fireball")
    R.check_bake_cost(rec)  # must not raise
    assert R.bake_cost(rec) < R.MAX_BAKE_COST
