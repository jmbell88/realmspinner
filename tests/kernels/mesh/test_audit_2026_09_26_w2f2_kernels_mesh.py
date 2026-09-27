"""Closes the 2026-09-26 audit's clay-document-07: ``read_rblk`` promises a
named ``ValueError`` for every malformed shape (its own module docstring's
"half-read document is worse than a refused one" rule, restated at every
field this reader touches) but several shapes still reached the caller as a
bare, unnamed ``AttributeError``, ``TypeError``, ``OverflowError`` or
Pillow's ``UnidentifiedImageError`` instead.

Reproduced against the unfixed reader in this fix's own scratch probe
(``probe_clay_serialize.py``, never checked into the tree): a non-object
``scene.json`` and a non-mapping texture entry each raised ``AttributeError``;
an ``Infinity`` in any of seven integer fields (a version, a uid, a material
index, a parent, a modifier id, a seam vertex, a material's texture index)
raised ``OverflowError`` (``int(float("inf"))``'s own exception, not
``ValueError``); a modifier parameter of the wrong type (``null``, a list, a
dict where a number belongs) raised ``TypeError`` from inside
``modifiers.make``'s own ``float(value)`` coercion; and bytes that are not a
real image raised Pillow's ``UnidentifiedImageError``.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from typing import Any

import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser


def _doc() -> bd.ClayDoc:
    doc = bd.ClayDoc(materials=[gltf.Material(name="red")])
    doc.objects.append(
        bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), material=0)
    )
    return doc


def _textured_doc() -> bd.ClayDoc:
    tex = (2, 2, b"\xff" * (2 * 2 * 4))
    doc = bd.ClayDoc(materials=[gltf.Material(name="tex", base_color=tex)])
    doc.objects.append(
        bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), material=0)
    )
    return doc


def _rewrite(data: bytes, edit: Any) -> bytes:
    """The same ``.rblk`` archive with ``edit`` applied to its parsed
    ``scene.json`` -- ``test_serialize.py``'s own helper, restated here since
    a test module may not import another test module."""
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


def _replace_scene(data: bytes, raw: bytes) -> bytes:
    """The same archive with ``scene.json`` replaced by arbitrary bytes -- for
    a mangle that must not even parse as a mapping."""
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            dst.writestr(name, raw if name == ser.SCENE else src.read(name))
    return out.getvalue()


def _corrupt_textures(data: bytes) -> bytes:
    """The same archive with every ``textures/*`` member replaced by bytes
    that are not a real image."""
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            if name.startswith("textures/"):
                dst.writestr(name, b"not a real png")
            else:
                dst.writestr(name, src.read(name))
    return out.getvalue()


def test_a_non_object_top_level_scene_is_refused_with_a_named_valueerror() -> None:
    data = ser.rblk_bytes(_doc())
    non_object = _replace_scene(data, json.dumps([1, 2, 3]).encode())
    with pytest.raises(ValueError, match="Realmspinner Clay document"):
        ser.read_rblk(non_object)


def test_a_texture_entry_that_is_not_a_mapping_is_refused_with_a_named_valueerror() -> None:
    data = ser.rblk_bytes(_doc())

    def mangle(scene: dict) -> None:
        scene["textures"] = ["not-a-mapping"]

    with pytest.raises(ValueError, match="texture"):
        ser.read_rblk(_rewrite(data, mangle))


@pytest.mark.parametrize(
    ("mangle_key", "path"),
    [
        ("version", ()),
        ("uid", ("objects", 0, "uid")),
        ("material", ("objects", 0, "material")),
        ("parent", ("objects", 0, "parent")),
    ],
)
def test_an_infinite_integer_field_is_refused_rather_than_raising_overflowerror(
    mangle_key, path
) -> None:
    """``int(float("inf"))`` raises ``OverflowError``, not ``ValueError`` --
    and Python's own ``json`` module accepts an ``Infinity`` literal by
    default. Every one of these fields used to reach that uncaught.
    """
    data = ser.rblk_bytes(_doc())

    def mangle(scene: dict) -> None:
        if not path:
            scene[mangle_key] = float("inf")
            return
        target = scene
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = float("inf")

    with pytest.raises(ValueError):
        ser.read_rblk(_rewrite(data, mangle))


def test_an_infinite_modifier_id_is_refused_rather_than_raising_overflowerror() -> None:
    data = ser.rblk_bytes(_doc())

    def mangle(scene: dict) -> None:
        scene["objects"][0]["modifiers"] = [
            {"id": float("inf"), "kind": "bevel", "params": {"angle": 0.1, "width": 0.1}}
        ]

    with pytest.raises(ValueError, match="modifier"):
        ser.read_rblk(_rewrite(data, mangle))


def test_an_infinite_seam_vertex_is_refused_rather_than_raising_overflowerror() -> None:
    data = ser.rblk_bytes(_doc())

    def mangle(scene: dict) -> None:
        scene["objects"][0]["seams"] = [[float("inf"), 0]]

    with pytest.raises(ValueError, match="seam"):
        ser.read_rblk(_rewrite(data, mangle))


def test_an_infinite_material_texture_index_is_refused_rather_than_raising_overflowerror() -> (
    None
):
    data = ser.rblk_bytes(_doc())

    def mangle(scene: dict) -> None:
        scene["materials"][0]["textures"] = {"base_color": float("inf")}

    with pytest.raises(ValueError, match="material"):
        ser.read_rblk(_rewrite(data, mangle))


@pytest.mark.parametrize("bad_angle", [None, [1, 2], {"x": 1}])
def test_a_modifier_parameter_of_the_wrong_type_is_refused_rather_than_raising_typeerror(
    bad_angle,
) -> None:
    """A modifier parameter of the wrong type reaches ``modifiers.make``'s own
    ``float(value)`` coercion as a bare ``TypeError`` -- ``_modifiers_from``
    used to catch only ``el.OpError`` around that call.
    """
    data = ser.rblk_bytes(_doc())

    def mangle(scene: dict) -> None:
        scene["objects"][0]["modifiers"] = [
            {"id": 1, "kind": "bevel", "params": {"angle": bad_angle, "width": 0.1}}
        ]

    with pytest.raises(ValueError, match="modifier"):
        ser.read_rblk(_rewrite(data, mangle))


def test_a_corrupt_texture_is_refused_rather_than_raising_unidentifiedimageerror() -> None:
    data = ser.rblk_bytes(_textured_doc())
    with pytest.raises(ValueError, match="image"):
        ser.read_rblk(_corrupt_textures(data))
