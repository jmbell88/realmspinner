"""Regression tests for the 2026-10-03 audit's Medium Clay findings
clay-27 through clay-30 (the mesh document's ceilings, primitives,
serialization and separate)."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import glbimport, serialize
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import separate as sep


def _doc_with(n: int, mesh=None) -> bd.ClayDoc:
    doc = bd.ClayDoc()
    mesh = mesh if mesh is not None else bp.box()
    for i in range(n):
        doc.add_object(bd.Obj(uid=bd.new_uid(), name=f"O{i}", mesh=mesh))
    return doc


def _with_scene(data: bytes, edit) -> bytes:
    """*data*, a ``.rblk``, with ``scene.json`` rewritten by ``edit(scene)``."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            payload = src.read(info.filename)
            if info.filename == serialize.SCENE:
                scene = json.loads(payload)
                edit(scene)
                payload = json.dumps(scene).encode("utf-8")
            dst.writestr(info.filename, payload)
    return out.getvalue()


# --- clay-27 ---------------------------------------------------------------


def test_save_refuses_a_document_with_more_objects_than_read_rblk_will_reopen(monkeypatch):
    # The ceiling is lowered rather than 4,097 objects built: both sides read
    # ``glbimport.MAX_OBJECTS`` at call time, so one number moves them together.
    monkeypatch.setattr(glbimport, "MAX_OBJECTS", 3)
    at_ceiling = _doc_with(3)
    assert len(serialize.read_rblk(serialize.rblk_bytes(at_ceiling)).objects) == 3

    past = _doc_with(4)
    with pytest.raises(ValueError, match="4 objects"):
        serialize.rblk_bytes(past)


def test_save_refuses_a_document_with_more_triangles_than_read_rblk_will_reopen(monkeypatch):
    monkeypatch.setattr(glbimport, "MAX_TRIANGLES", 24)  # a box is 12
    assert len(serialize.read_rblk(serialize.rblk_bytes(_doc_with(2))).objects) == 2

    with pytest.raises(ValueError, match="triangles"):
        serialize.rblk_bytes(_doc_with(3))


def test_save_refuses_the_real_object_ceiling_as_read_rblk_states_it() -> None:
    doc = _doc_with(glbimport.MAX_OBJECTS + 1)
    with pytest.raises(ValueError, match=f"{glbimport.MAX_OBJECTS + 1:,} objects"):
        serialize.rblk_bytes(doc)


# --- clay-28 ---------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_clamp_params_refuses_a_non_finite_float_parameter(bad):
    with pytest.raises(ValueError, match="radius"):
        bp.clamp_params("cylinder", {"radius": bad, "height": 1.0, "segments": 8})
    with pytest.raises(ValueError, match="size"):
        bp.clamp_params("box", {"size": [1.0, bad, 1.0]})


def test_clamp_params_still_passes_finite_parameters_through_unchanged() -> None:
    out = bp.clamp_params("cylinder", {"radius": 0.25, "height": 2.0, "segments": 8})
    assert out == {"radius": 0.25, "height": 2.0, "segments": 8}
    assert bp.clamp_params("box", {"size": [1.0, 2.0, 3.0]})["size"] == [1.0, 2.0, 3.0]


# --- clay-29 ---------------------------------------------------------------


def test_read_rblk_refuses_a_translation_too_large_for_a_float_by_name_and_read_view_answers_none():
    data = serialize.rblk_bytes(_doc_with(1), view=None)
    huge = 10**400

    def translation(scene):
        scene["objects"][0]["translation"] = [huge, 0, 0]

    with pytest.raises(ValueError, match="translation"):
        serialize.read_rblk(_with_scene(data, translation))

    def yaw(scene):
        scene["view"] = {"yaw": huge, "pitch": 0.0, "distance": 1.0, "target": [0, 0, 0]}

    assert serialize.read_view(_with_scene(data, yaw)) is None


def test_read_view_answers_none_for_a_yaw_too_large_for_a_float() -> None:
    data = serialize.rblk_bytes(_doc_with(1), view=None)

    def yaw(scene):
        scene["view"] = {"yaw": 10**400, "pitch": 0.0, "distance": 1.0, "target": [0, 0, 0]}

    assert serialize.read_view(_with_scene(data, yaw)) is None


# --- clay-30 ---------------------------------------------------------------


def test_separate_in_face_mode_keeps_selection_equal_to_the_uids_with_an_element_selection():
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    doc.select([obj.uid])
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=[0, 1]))

    pieces = sep.by_selection(obj.mesh, doc.element_sel_of(obj.uid))
    doc.separate(obj.uid, pieces)

    assert doc.selection == {u for u, s in doc.element_sel.items() if not el.is_empty(s)}
    assert doc.selection == set()


def test_separate_in_object_mode_still_selects_the_new_pieces() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    pieces = sep.by_selection(obj.mesh, el.ElementSel(faces=[0, 1]))
    new = doc.separate(obj.uid, pieces)
    assert doc.selection == {o.uid for o in new}
    assert np.isfinite(new[0].mesh.positions).all()
