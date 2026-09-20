"""Regression tests closing 2026-09-14 audit findings with no better-fitting
existing test module. See ``docs/audit-2026-09-14.md`` for the full findings;
do not cite it from ``src/`` (``tests/test_ux_todo_fixes.py`` refuses that)."""

from __future__ import annotations

import inspect


def test_land_tileset_docstring_does_not_claim_the_projection_arm_is_test_only():
    """The 2026-09-14 audit, plotter-04: ``land_tileset``'s comment claimed
    its ``projection`` arm was "currently reached only from tests". Untrue:
    ``use_as_tileset`` submits the ``plotter-tileset:<uid>`` task, whose
    ``run()`` calls ``_sheet_or_tileset``, which sets
    ``out["projection"] = recorded_lattice`` for an unpainted map whose
    lattice disagrees with a recorded sheet's -- and
    ``plotter_mode.on_task_done``'s ``plotter-tileset`` arm passes that
    result straight into ``land_tileset``.
    """
    from realmspinner.studio.modes.plotter import tilesets as plotter_tilesets

    source = inspect.getsource(plotter_tilesets.land_tileset)
    assert "reached only from tests" not in source
    assert "use_as_tileset" in source


# --- plotter-02: ``_erase_seam`` has no behavioural test ---------------------
#
# Evidence gap, not a reproduced defect: ``_erase_seam`` (``_q_tileset.py``,
# called through the materials mode's ``seam_erase`` option) had only its
# cancel-flag door covered (``test_job_durability.py``'s stage-kind sweep),
# never whether it actually rewrites the material or leaves it alone on a
# cancel. Both tests below call the function directly with a fake ``t2i`` --
# a plain object with a ``generate`` method -- rather than through the async
# ``Worker``, since ``_erase_seam`` is itself synchronous (``to_thread``
# calls it) and needs nothing else the queue provides. **Both pass against
# HEAD's unfixed code**: this closes the evidence gap by pinning the
# behaviour the code already has, not a discovered defect.


def _material_image(path, fill):
    from PIL import Image

    Image.new("RGB", (32, 32), fill).save(path)


def test_seam_erase_actually_redraws_the_material_and_a_cancel_mid_pass_leaves_it_untouched(
    tmp_path,
):
    """Normal pass: ``_erase_seam`` reads ``material``, runs it through the
    fake ``t2i.generate`` and writes the rolled-back result over ``material``
    -- staged to a ``.tmp`` sibling and ``os.replace``'d, per this repo's
    stage-then-replace rule for a served name."""
    from realmspinner import _q_tileset

    material = tmp_path / "material.png"
    _material_image(material, (10, 20, 30))
    original_bytes = material.read_bytes()

    class FakeT2I:
        def generate(self, prompt, out, **kwargs):
            _material_image(out, (200, 100, 50))

    _q_tileset._erase_seam(
        FakeT2I(),
        material,
        tmp_path / "scratch",
        prompt="a mossy wall",
        seed=1,
        lora=None,
        lora_weight=0.0,
        negative_prompt="",
        size=32,
        cancel_event=None,
    )

    assert material.read_bytes() != original_bytes, "the material was not redrawn"
    assert not material.with_name(material.name + ".tmp").exists(), (
        "a staged temp sibling was left behind instead of being replaced"
    )


def test_a_cancel_mid_erase_pass_leaves_the_material_file_untouched(tmp_path):
    """A cancel discovered during ``t2i.generate`` (the fake sets the caller's
    own ``cancel_event`` the way the real pipeline would report one mid-draw)
    must leave ``material`` exactly as it was: ``_erase_seam`` checks
    ``cancel_event`` before it ever opens the redrawn output or touches the
    served file."""
    import threading

    from realmspinner import _q_tileset

    material = tmp_path / "material.png"
    _material_image(material, (10, 20, 30))
    original_bytes = material.read_bytes()
    cancel_event = threading.Event()

    class FakeT2I:
        def generate(self, prompt, out, *, cancel_event, **kwargs):
            # The pipeline discovers the cancel mid-draw and stops without
            # finishing ``out`` -- nothing here writes to it.
            cancel_event.set()

    _q_tileset._erase_seam(
        FakeT2I(),
        material,
        tmp_path / "scratch",
        prompt="a mossy wall",
        seed=1,
        lora=None,
        lora_weight=0.0,
        negative_prompt="",
        size=32,
        cancel_event=cancel_event,
    )

    assert material.read_bytes() == original_bytes
    assert not material.with_name(material.name + ".tmp").exists()
