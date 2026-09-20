"""``readiness.validate`` -- one game-engine readiness row per :data:`CHECKS`
entry, pinned against small hand-built documents the same way
``test_ops_clean.py`` pins ``survey``/``clean``: each test is exactly one
defect against an otherwise clean box, so a warn/fail/skip on the row that
names it -- and a pass everywhere else -- is the whole claim.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from warlock.kernels.geom3d import gltf
from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import elements as el
from warlock.kernels.mesh import ops_clean as oc
from warlock.kernels.mesh import ops_topo, readiness
from warlock.kernels.mesh import primitives as prim


def _obj(
    mesh, *, translation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
    generator=None, material=0, visible=True,
):
    return bd.Obj(
        uid=bd.new_uid(),
        name="Obj",
        mesh=mesh,
        translation=translation,
        scale=scale,
        generator=generator,
        material=material,
        visible=visible,
    )


def _grounded_box(**kwargs):
    """A unit box, UV stripped (real primitives carry UV -- see
    ``ops_clean``'s own module docstring), standing on the ground."""
    mesh = replace(prim.box(), uv=None)
    return _obj(mesh, translation=(0.0, 0.5, 0.0), **kwargs)


def _doc(*objects, materials=None):
    return bd.ClayDoc(objects=list(objects), materials=materials)


def _by_key(report: readiness.Report) -> dict[str, readiness.Check]:
    return {c.key: c for c in report.checks}


# --- structural ---------------------------------------------------------------


def test_checks_order_is_the_reports_order_and_each_appears_once():
    doc = _doc()
    report = readiness.validate(doc)
    assert tuple(c.key for c in report.checks) == readiness.CHECKS
    assert len(set(c.key for c in report.checks)) == len(readiness.CHECKS)


def test_every_fix_a_check_can_return_is_in_fix_ops(monkeypatch):
    """Drives a warn/fail path on every check that ever names a ``fix``, and
    asserts the whole set of names seen is exactly :data:`FIX_OPS`."""
    seen: set[str] = set()

    # triangles -> decimate
    base = readiness.PROFILES["godot-desktop"]
    low_tri = readiness.Profile(**{**base.__dict__, "triangles_warn": 1, "triangles_fail": 2})
    monkeypatch.setitem(readiness.PROFILES, "_test_low_tri", low_tri)
    report = readiness.validate(_doc(_grounded_box()), profile="_test_low_tri")
    seen.add(_by_key(report)["triangles"].fix)

    # geometry -> clean-mesh (a duplicate face)
    box = prim.box()
    corners = np.arange(int(box.starts[0]), int(box.starts[1]))
    loops = np.concatenate([box.loops.astype("i8"), box.loops.astype("i8")[corners]])
    starts = np.concatenate([box.starts.astype("i8"), [int(box.starts[-1]) + len(corners)]])
    material = np.concatenate([box.material, box.material[[0]]])
    smooth = np.concatenate([box.smooth, box.smooth[[0]]])
    dup = replace(box, loops=loops, starts=starts, material=material, smooth=smooth, uv=None)
    report = readiness.validate(_doc(_obj(dup, translation=(0.0, 0.5, 0.0))))
    seen.add(_by_key(report)["geometry"].fix)

    # normals -> recalc-normals (one flipped face)
    face_sel = el.ElementSel(faces=np.array([0], dtype="i4"))
    flipped, _sel = ops_topo.flip_normals(prim.box(), face_sel)
    report = readiness.validate(_doc(_obj(replace(flipped, uv=None), translation=(0.0, 0.5, 0.0))))
    seen.add(_by_key(report)["normals"].fix)

    # closed -> clean-mesh (an open box)
    opened, _sel = ops_topo.delete_faces(prim.box(), face_sel)
    report = readiness.validate(_doc(_obj(replace(opened, uv=None), translation=(0.0, 0.5, 0.0))))
    seen.add(_by_key(report)["closed"].fix)

    # transforms -> bake (negative scale)
    report = readiness.validate(_doc(_grounded_box(scale=(-1.0, 1.0, 1.0))))
    seen.add(_by_key(report)["transforms"].fix)

    # pivot -> drop-to-ground (floating)
    floating = _obj(replace(prim.box(), uv=None), translation=(0.0, 5.0, 0.0))
    report = readiness.validate(_doc(floating))
    seen.add(_by_key(report)["pivot"].fix)

    # uvs -> unwrap (textured material, no UV)
    tex_material = gltf.Material(name="Tex", base_color=(2, 2, bytes(2 * 2 * 4)))
    textured = _obj(replace(prim.box(), uv=None), translation=(0.0, 0.5, 0.0), material=0)
    report = readiness.validate(_doc(textured, materials=[tex_material]))
    seen.add(_by_key(report)["uvs"].fix)

    seen.discard("")
    assert seen == readiness.FIX_OPS


# --- the pass/warn/fail scenarios named in the brief --------------------------


def test_a_clean_primitive_document_passes_everything_except_uvs():
    obj = _grounded_box()
    report = readiness.validate(_doc(obj))
    checks = _by_key(report)
    for key, check in checks.items():
        if key == "uvs":
            assert check.status == "warn"
            assert check.fix == "unwrap"
        elif key == "textures":
            # The default (untextured) palette entry has no texture slot at
            # all, so there is nothing for this check to measure -- "skip",
            # not "pass": see the check's own semantics in the module
            # docstring ("skip when no used material has a texture").
            assert check.status == "skip"
        elif key == "collider_present":
            # Tranche 7: advisory, never a fail -- a document with no
            # collider at all is worth flagging, the same "soft but real"
            # reading ``uvs``' own "no UV anywhere" branch gets above.
            assert check.status == "warn"
        elif key in ("collider_triangles", "collider_convex"):
            # Nothing to measure with no collider in the document at all.
            assert check.status == "skip"
        else:
            assert check.status == "pass", f"{key}: {check.status} ({check.message})"


def test_an_empty_document_fails_objects_and_skips_everything_else():
    report = readiness.validate(_doc())
    checks = _by_key(report)
    assert checks["objects"].status == "fail"
    for key in readiness.CHECKS[1:]:
        assert checks[key].status == "skip", f"{key}: {checks[key].status}"
    assert report.status == "fail"


def test_triangle_warn_and_fail_thresholds(monkeypatch):
    low = readiness.Profile(
        key="low", label="Low",
        triangles_warn=1, triangles_fail=2,
        vertices_warn=10_000, materials_warn=8,
        texture_px_warn=4096, texture_bytes_warn=64 * 1024 * 1024,
        size_min_m=0.01, size_max_m=200.0,
    )
    monkeypatch.setitem(readiness.PROFILES, "low", low)

    # A box triangulates to 12 triangles -- past both thresholds.
    report = readiness.validate(_doc(_grounded_box()), profile="low")
    tri = _by_key(report)["triangles"]
    assert tri.status == "fail"
    assert tri.fix == "decimate"
    assert tri.limit == 2

    warn_only = readiness.Profile(**{**low.__dict__, "triangles_fail": 1_000_000})
    monkeypatch.setitem(readiness.PROFILES, "low", warn_only)
    report = readiness.validate(_doc(_grounded_box()), profile="low")
    tri = _by_key(report)["triangles"]
    assert tri.status == "warn"
    assert tri.limit == 1


def test_a_textured_material_on_a_uv_less_object_fails_uvs():
    tex_material = gltf.Material(name="Tex", base_color=(2, 2, bytes(2 * 2 * 4)))
    obj = _obj(replace(prim.box(), uv=None), translation=(0.0, 0.5, 0.0), material=0)
    report = readiness.validate(_doc(obj, materials=[tex_material]))
    uvs = _by_key(report)["uvs"]
    assert uvs.status == "fail"
    assert uvs.fix == "unwrap"
    assert obj.uid in uvs.uids


def test_a_flipped_face_warns_normals_with_fix_recalc_normals():
    face_sel = el.ElementSel(faces=np.array([0], dtype="i4"))
    flipped, _sel = ops_topo.flip_normals(prim.box(), face_sel)
    obj = _obj(replace(flipped, uv=None), translation=(0.0, 0.5, 0.0))
    report = readiness.validate(_doc(obj))
    normals = _by_key(report)["normals"]
    assert normals.status == "warn"
    assert normals.fix == "recalc-normals"
    assert obj.uid in normals.uids


def test_an_open_box_warns_closed_but_a_plane_generator_object_does_not():
    opened, _sel = ops_topo.delete_faces(prim.box(), el.ElementSel(faces=np.array([0], dtype="i4")))
    open_obj = _obj(replace(opened, uv=None), translation=(0.0, 0.5, 0.0), generator=None)
    report = readiness.validate(_doc(open_obj))
    closed = _by_key(report)["closed"]
    assert closed.status == "warn"
    assert closed.fix == "clean-mesh"
    assert open_obj.uid in closed.uids

    plane_obj = _obj(replace(prim.plane(), uv=None), generator="plane")
    report = readiness.validate(_doc(plane_obj))
    closed = _by_key(report)["closed"]
    assert closed.status == "pass"


def test_negative_scale_warns_transforms():
    obj = _grounded_box(scale=(-1.0, 1.0, 1.0))
    report = readiness.validate(_doc(obj))
    transforms = _by_key(report)["transforms"]
    assert transforms.status == "warn"
    assert transforms.fix == "bake"
    assert obj.uid in transforms.uids


def test_a_100m_cube_warns_scale_and_mentions_centimetres():
    # box() is a 1 m cube; scaled x100 it is 100 m across, well past this
    # profile's 50 m single-asset ceiling (readiness._SIZE_MAX_DESKTOP_M) --
    # exactly the "imported at centimetre scale" mistake this check exists
    # to catch.
    obj = _obj(
        replace(prim.box(), uv=None), translation=(0.0, 50.0, 0.0),
        scale=(100.0, 100.0, 100.0),
    )
    report = readiness.validate(_doc(obj))
    scale = _by_key(report)["scale"]
    assert scale.status == "warn"
    assert "centimetre" in scale.message.lower()


def test_a_floating_object_warns_pivot_with_fix_drop_to_ground():
    obj = _obj(replace(prim.box(), uv=None), translation=(0.0, 5.0, 0.0))
    report = readiness.validate(_doc(obj))
    pivot = _by_key(report)["pivot"]
    assert pivot.status == "warn"
    assert pivot.fix == "drop-to-ground"


def test_hidden_objects_are_ignored_unless_visible_only_is_false():
    # "transforms" (per-object, negative scale) rather than "pivot"
    # (document-wide lowest point) -- a hidden object floating above an
    # otherwise-grounded visible one would not move the document's own
    # minimum y at all, so pivot cannot tell the two calls apart here.
    good = _grounded_box()
    bad = _obj(
        replace(prim.box(), uv=None), translation=(0.0, 0.5, 0.0),
        scale=(-1.0, 1.0, 1.0), visible=False,
    )
    doc = _doc(good, bad)

    report = readiness.validate(doc, visible_only=True)
    transforms = _by_key(report)["transforms"]
    assert transforms.status == "pass"

    report = readiness.validate(doc, visible_only=False)
    transforms = _by_key(report)["transforms"]
    assert transforms.status == "warn"
    assert bad.uid in transforms.uids


def test_an_object_past_max_clean_corners_makes_survey_checks_skip(monkeypatch):
    monkeypatch.setattr(oc, "MAX_CLEAN_CORNERS", 10)

    called = []
    monkeypatch.setattr(oc, "survey", lambda *a, **k: called.append(1) or (_ for _ in ()).throw(
        AssertionError("survey should not be called past the ceiling")
    ))

    obj = _grounded_box()  # a box has 24 corners, past the monkeypatched 10
    report = readiness.validate(_doc(obj))
    checks = _by_key(report)
    for key in ("geometry", "normals", "closed"):
        assert checks[key].status == "skip"
        assert checks[key].limit == 10
        assert obj.uid in checks[key].uids
    assert not called


def test_report_status_is_the_worst_of_its_checks():
    # A clean-but-UV-less box has exactly one non-pass row (uvs, a warn), so
    # the whole report reads "warn" even though eleven of twelve rows pass.
    report = readiness.validate(_doc(_grounded_box()))
    assert _by_key(report)["uvs"].status == "warn"
    assert all(c.status in ("pass", "warn", "skip") for c in report.checks)
    assert report.status == "warn"

    # "objects" fails outright on an empty document, and fail beats warn.
    empty_report = readiness.validate(_doc())
    assert empty_report.status == "fail"

    # A textured material with no UV on the object using it is a "fail" row,
    # which must win over the "warn" rows elsewhere in the same report.
    tex_material = gltf.Material(name="Tex", base_color=(2, 2, bytes(2 * 2 * 4)))
    obj = _obj(replace(prim.box(), uv=None), translation=(0.0, 0.5, 0.0), material=0)
    fail_report = readiness.validate(_doc(obj, materials=[tex_material]))
    assert _by_key(fail_report)["uvs"].status == "fail"
    assert fail_report.status == "fail"
