"""Wavefront OBJ into a Clay document."""

from __future__ import annotations

import numpy as np
import pytest

from warlock.kernels.mesh import mesh as bm
from warlock.kernels.mesh import objimport
from warlock.kernels.mesh.elements import OpError

# --- refusals -----------------------------------------------------------------


def test_an_empty_file_is_refused() -> None:
    with pytest.raises(OpError):
        objimport.obj_to_claydoc("")
    with pytest.raises(OpError):
        objimport.obj_to_claydoc("   \n\n  ")


def test_a_file_with_vertices_but_no_faces_is_refused() -> None:
    with pytest.raises(OpError):
        objimport.obj_to_claydoc("v 0 0 0\nv 1 0 0\nv 0 1 0\n")


def test_the_triangle_ceiling_refuses_before_any_array_is_built(monkeypatch) -> None:
    """A quad face is two triangles; with the ceiling monkeypatched to one, the
    refusal must land during the budget pre-pass, not after building a Mesh."""
    monkeypatch.setattr(objimport, "MAX_TRIANGLES", 1)
    text = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n"
    with pytest.raises(OpError, match="triangles"):
        objimport.obj_to_claydoc(text)


def test_the_object_ceiling_refuses(monkeypatch) -> None:
    monkeypatch.setattr(objimport, "MAX_OBJECTS", 1)
    text = "o A\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\no B\nv 0 0 1\nv 1 0 1\nv 0 1 1\nf 4 5 6\n"
    with pytest.raises(OpError, match="objects"):
        objimport.obj_to_claydoc(text)


def test_an_unknown_up_axis_is_refused() -> None:
    with pytest.raises(OpError):
        objimport.obj_to_claydoc("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", up="x")


# --- parsing shape --------------------------------------------------------------


def test_a_negative_index_resolves_relative_to_what_has_been_defined_so_far() -> None:
    text = (
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0.5 1.5 0\n"
        "v 0 1 0\n"
        "vn 0 0 1\n"
        "f -5//1 -4//1 -3//1 -2//1 -1//1\n"
    )
    doc = objimport.obj_to_claydoc(text)
    assert len(doc.objects) == 1
    mesh = doc.objects[0].mesh
    bm.validate(mesh)
    assert bm.face_count(mesh) == 1
    assert len(mesh.loops) == 5, "the n-gon keeps all five corners, not a fan of triangles"
    assert mesh.uv is None, "vn carries no uv, and no vt token was given"


def test_v_slash_slash_vn_form_is_accepted_and_vn_ignored() -> None:
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvn 0 0 1\nvn 0 0 1\nf 1//1 2//2 3//3\n"
    doc = objimport.obj_to_claydoc(text)
    mesh = doc.objects[0].mesh
    bm.validate(mesh)
    assert bm.face_count(mesh) == 1
    assert mesh.uv is None


def test_o_and_g_each_start_a_new_object() -> None:
    text = (
        "o First\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "f 1 2 3\n"
        "g Second\n"
        "v 0 0 1\nv 1 0 1\nv 0 1 1\n"
        "f 4 5 6\n"
    )
    doc = objimport.obj_to_claydoc(text)
    assert [obj.name for obj in doc.objects] == ["First", "Second"]
    for obj in doc.objects:
        bm.validate(obj.mesh)
        assert bm.face_count(obj.mesh) == 1


def test_a_face_missing_a_vt_turns_off_uv_for_the_whole_object_only() -> None:
    text = (
        "o Textured\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 0 1\n"
        "f 1/1 2/2 3/3\n"
        "o Bare\n"
        "v 0 0 2\nv 1 0 2\nv 0 1 2\n"
        "vt 0 0\nvt 1 0\n"  # global vt indices 4 and 5 -- 1..3 already belong to Textured
        "f 4/4 5/5 6\n"  # third corner has no vt
    )
    doc = objimport.obj_to_claydoc(text)
    textured, bare = doc.objects
    bm.validate(textured.mesh)
    bm.validate(bare.mesh)
    assert textured.mesh.uv is not None
    assert bare.mesh.uv is None, "one corner missing vt drops uv for the whole object"


def test_uv_v_is_flipped_from_objs_bottom_up_convention() -> None:
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0.25 1.0\nvt 0.75 1.0\nvt 0.0 0.0\nf 1/1 2/2 3/3\n"
    doc = objimport.obj_to_claydoc(text)
    mesh = doc.objects[0].mesh
    assert mesh.uv is not None
    assert np.allclose(mesh.uv[0], [0.25, 0.0])
    assert np.allclose(mesh.uv[2], [0.0, 1.0])


def test_s_off_and_s_1_set_the_per_face_smooth_flag() -> None:
    text = (
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 0 1 1\n"
        "s off\n"
        "f 1 2 3\n"
        "s 1\n"
        "f 4 5 6\n"
    )
    doc = objimport.obj_to_claydoc(text)
    mesh = doc.objects[0].mesh
    assert mesh.smooth.tolist() == [False, True]


def test_usemtl_and_mtl_text_build_the_palette() -> None:
    obj_text = (
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "usemtl Red\n"
        "f 1 2 3\n"
        "v 0 0 1\nv 1 0 1\nv 0 1 1\n"
        "usemtl Blue\n"
        "f 4 5 6\n"
    )
    mtl_text = (
        "newmtl Red\nKd 1 0 0\nd 1\nNs 0\n\nnewmtl Blue\nKd 0 0 1\nd 0.5\nNs 250\n"
    )
    doc = objimport.obj_to_claydoc(obj_text, mtl=mtl_text)
    mesh = doc.objects[0].mesh
    red_idx = int(mesh.material[0])
    blue_idx = int(mesh.material[1])
    red = doc.materials[red_idx]
    blue = doc.materials[blue_idx]
    assert np.allclose(red.base_color_factor, (1.0, 0.0, 0.0, 1.0))
    assert np.allclose(blue.base_color_factor, (0.0, 0.0, 1.0, 0.5))
    # Ns 0 -> roughness 1.0; Ns 250 -> a lower roughness. Exact via the module's
    # own invertible formula, see objimport.roughness_from_ns.
    assert red.roughness_factor == pytest.approx(objimport.roughness_from_ns(0.0))
    assert blue.roughness_factor == pytest.approx(objimport.roughness_from_ns(250.0))


def test_comments_blank_lines_and_line_continuations_are_tolerated() -> None:
    text = (
        "# a comment\n"
        "\n"
        "v 0 0 0  # trailing comment too\n"
        "v 1 0 0\n"
        "v 0 1 \\\n"
        "0\n"
        "an unrecognised statement here\n"
        "f 1 2 3\n"
    )
    doc = objimport.obj_to_claydoc(text)
    mesh = doc.objects[0].mesh
    bm.validate(mesh)
    assert bm.face_count(mesh) == 1
    assert np.allclose(mesh.positions[2], [0.0, 1.0, 0.0])


# --- axis and scale -------------------------------------------------------------


def _triangle(**kw: object):
    text = "v 0 0 1\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    return objimport.obj_to_claydoc(text, **kw)  # type: ignore[arg-type]


def test_up_z_rotates_z_up_into_gltfs_y_up() -> None:
    doc = _triangle(up="z")
    # (x, y, z) -> (x, z, -y): the first vertex (0, 0, 1) becomes (0, 1, 0).
    assert np.allclose(doc.objects[0].mesh.positions[0], [0.0, 1.0, 0.0])


def test_scale_multiplies_every_position() -> None:
    doc = _triangle(scale=2.0)
    assert np.allclose(doc.objects[0].mesh.positions[0], [0.0, 0.0, 2.0])


def test_mtllib_is_parsed_and_ignored() -> None:
    """The file's own ``mtllib`` names a path this pure kernel never opens --
    only the caller's own ``mtl`` text (if any) supplies material data."""
    text = "mtllib external.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    doc = objimport.obj_to_claydoc(text)
    assert bm.face_count(doc.objects[0].mesh) == 1
