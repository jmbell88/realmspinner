"""Closes two findings from the 2026-09-27 fix pass over the 2026-09-26
audit's remaining Clay findings (fixer w5f3).

clay-document-02 (the kernel half; the agent door was already fixed
elsewhere): ``arch(width=0)`` raised ``ZeroDivisionError`` even after
``clamp_params``, because ``arch``'s own ``across`` UV helper divides by
``2.0 * r_out`` and nothing floored ``r_out`` the way ``depth`` already is a
few lines down; ``stairs``/``doorway`` at zero extents collapsed every
station of their swept profile to the same point, so ``_dedup_closed`` left
a single-corner face and ``mesh.validate`` raised "fewer than 3 corners" the
first time anything checked the mesh it returned. Reproduced against the
unfixed generators loaded from ``git show HEAD`` in this fix's own scratch
probes (``probe_primitives.py``, ``probe_primitives2.py``,
``probe_all_generators_head.py``, none checked into the tree): ``arch``
raised ``ZeroDivisionError`` directly, and ``stairs``/``doorway`` built
without error but failed ``validate()`` with "face 0 has fewer than 3
corners". A sweep of every other registered generator at every numeric
param zero already passed against the unfixed code, so this is exactly and
only those three.

A new finding the fixers surfaced (not in the 2026-09-26 audit; scope here
is Clay's copy only -- Mason's copy of the same gap is a different fixer's
file): ``_material_from`` cast ``base_color_factor``/``emissive_factor``
with a bare ``tuple(...)``, so a document whose material JSON carried the
wrong number of components for either field was accepted silently and
permanently disagreed with :class:`gltf.Material`'s own declared
``tuple[float, float, float, float]`` / ``tuple[float, float, float]``
shape. Reproduced against the unfixed ``_material_from`` loaded from
``git show HEAD`` (``probe_serialize.py``, not checked into the tree):
a 2-tuple ``base_color_factor`` came back accepted as-is.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from typing import Any

import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser


def _zero_numeric(value: Any) -> Any:
    """``value`` with every ``int``/``float`` leaf (never a ``bool``, which is
    an ``int`` subclass but not a numeric *param* here) replaced by ``0.0``,
    recursing into a tuple/list so a ``size=(1.0, 1.0, 1.0)``-shaped default
    zeroes every component rather than being left alone as "not a number"."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return 0.0
    if isinstance(value, (list, tuple)):
        return type(value)(_zero_numeric(v) for v in value)
    return value


@pytest.mark.parametrize("name", sorted(bp.GENERATORS))
def test_every_generator_builds_a_valid_mesh_when_every_numeric_param_is_zero(
    name: str,
) -> None:
    defaults, builder = bp.GENERATORS[name]
    zeroed = {key: _zero_numeric(value) for key, value in defaults.items()}
    mesh = builder(**zeroed)
    bm.validate(mesh)  # must not raise


def test_arch_at_zero_width_does_not_divide_by_zero_in_its_own_uv_builder() -> None:
    # arch's own docstring says height is raised to half of width and
    # thickness is clamped inside the head's radius, but nothing floored
    # ``r_out`` itself -- see the comment on ``r_out`` in primitives.py.
    mesh = bp.arch(width=0.0, height=0.0, depth=0.0, thickness=0.0)
    bm.validate(mesh)


def test_stairs_at_zero_height_and_depth_does_not_collapse_to_one_corner() -> None:
    mesh = bp.stairs(width=0.0, total_height=0.0, total_depth=0.0)
    bm.validate(mesh)


def test_doorway_at_zero_wall_length_and_height_does_not_collapse_to_one_corner() -> None:
    mesh = bp.doorway(
        wall_length=0.0,
        wall_height=0.0,
        wall_thickness=0.0,
        opening_width=0.0,
        opening_height=0.0,
        opening_offset=0.0,
    )
    bm.validate(mesh)


# --- base_color_factor / emissive_factor length -------------------------


def _doc() -> bd.ClayDoc:
    doc = bd.ClayDoc(materials=[gltf.Material(name="red")])
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), material=0))
    return doc


def _rewrite(data: bytes, edit: Any) -> bytes:
    """The same ``.rblk`` archive with ``edit`` applied to its parsed
    ``scene.json`` -- restated from ``test_audit_2026_09_26_w2f2_kernels_mesh.py``'s
    own helper, since a test module may not import another test module."""
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            if name == ser.SCENE:
                scene = json.loads(src.read(name))
                edit(scene)
                dst.writestr(name, json.dumps(scene))
            else:
                dst.writestr(name, src.read(name))
    return out.getvalue()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("base_color_factor", [1.0, 2.0]),
        ("base_color_factor", [1.0, 2.0, 3.0, 4.0, 5.0]),
        ("emissive_factor", [1.0]),
        ("emissive_factor", [1.0, 2.0, 3.0, 4.0]),
    ],
)
def test_a_material_factor_of_the_wrong_length_is_refused_with_a_named_valueerror(
    field: str, bad_value: list[float]
) -> None:
    data = ser.rblk_bytes(_doc())

    def mangle(scene: dict) -> None:
        scene["materials"][0][field] = bad_value

    with pytest.raises(ValueError, match="material"):
        ser.read_rblk(_rewrite(data, mangle))
