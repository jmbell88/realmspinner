"""Regression tests for the 2026-09-26 audit, findings create-brief-06 and
create-brief-07, both in ``judge.py`` (and, for the first, its caller in
``service/judge.py``).
"""

from __future__ import annotations

import numpy as np

from realmspinner import judge


def _separable(n: int = 20):
    rng = np.random.default_rng(0)
    pos = rng.normal(loc=2.0, size=(n, 4))
    neg = rng.normal(loc=-2.0, size=(n, 4))
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    return x, y


# --- create-brief-06: load must compare the stored stage to the one asked for


def test_load_refuses_a_probe_whose_stored_stage_disagrees_with_the_one_requested(
    tmp_path,
):
    x, y = _separable()
    probe = judge.fit(x, y, stage="reference")
    # The file is named for "blank" -- a rename, or a copy over by hand --
    # but its *contents* are still the "reference" probe.
    path = tmp_path / "probe-blank.npz"
    judge.save(probe, path)

    assert judge.load(path, stage="blank") is None, (
        "a probe fitted to one question must not answer for another just "
        "because it was found under that question's file name"
    )
    # The control: asking for the stage it was actually fitted to still works.
    loaded = judge.load(path, stage="reference")
    assert loaded is not None
    assert loaded.stage == "reference"
    # And the back-compatible form -- no opinion about the stage -- still
    # reads the file, since not every caller has one (a probe listing, this
    # module's own round-trip test).
    assert judge.load(path) is not None


def test_service_judge_probe_asks_load_to_check_the_stage_it_requested(tmp_path):
    """The service door (``service.judge.probe``) is what every real caller
    goes through, and it used to build the path from ``stage`` but never
    pass ``stage`` on to ``judge.load`` -- so the content check above never
    actually ran in the app, only in a direct ``judge.load`` call."""
    import inspect

    from realmspinner.service import judge as svc_judge

    source = inspect.getsource(svc_judge.probe)
    assert "stage=stage" in source, (
        "service.judge.probe must pass its own stage argument to judge.load "
        "so a mismatched probe file is refused, not silently trusted"
    )


# --- create-brief-07: the staging name must be unique per call -------------


def test_save_stages_under_a_name_unique_to_the_call_not_the_destination(
    tmp_path, monkeypatch
):
    """A fixed ``<stem>.tmp.npz`` name is exactly the M03 shape
    ``core.safeio.atomic._tmp_name``'s docstring already names as a bug:
    two concurrent stagings of one destination collide, and the earlier
    writer's cleanup can unlink the later writer's still-live file.
    """
    seen: list = []
    real_savez = judge.np.savez

    def spy_savez(tmp, **kwargs):
        seen.append(judge.Path(tmp))
        real_savez(tmp, **kwargs)

    monkeypatch.setattr(judge.np, "savez", spy_savez)

    x, y = _separable()
    probe = judge.fit(x, y, stage="blank")
    path = tmp_path / "probe-blank.npz"

    judge.save(probe, path)
    judge.save(probe, path)

    assert len(seen) == 2
    assert seen[0] != seen[1], (
        "two saves to the same destination staged into the same temp file"
    )
    # And the ordinary contract still holds: the destination itself lands.
    assert path.exists()
    assert not seen[0].exists()
    assert not seen[1].exists()
