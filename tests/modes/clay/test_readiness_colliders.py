"""Tranche 7 (integration half): the three collider rows the game check
gained -- ``collider_present``, ``collider_triangles``, ``collider_convex``.

``test_readiness.py`` pins the render-side checks against small hand-built
documents; this file does the same for the three rows this tranche added,
plus the one existing test its own arrival touches (see that file's own
``test_a_clean_primitive_document_passes_everything_except_uvs``, updated
alongside this one).
"""

from __future__ import annotations

from dataclasses import replace

from warlock.kernels.mesh import colliders as cl
from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import primitives as prim
from warlock.kernels.mesh import readiness


def _obj(mesh, *, translation=(0.0, 0.0, 0.0), **kwargs):
    return bd.Obj(uid=bd.new_uid(), name="Obj", mesh=mesh, translation=translation, **kwargs)


def _grounded_box(**kwargs):
    mesh = replace(prim.box(), uv=None)
    return _obj(mesh, translation=(0.0, 0.5, 0.0), **kwargs)


def _doc(*objects, materials=None):
    return bd.ClayDoc(objects=list(objects), materials=materials)


def _by_key(report: readiness.Report) -> dict[str, readiness.Check]:
    return {c.key: c for c in report.checks}


# --- collider_present: advisory, never a fail --------------------------------


def test_no_collider_at_all_warns_collider_present_but_never_fails():
    doc = _doc(_grounded_box())
    report = readiness.validate(doc)
    row = _by_key(report)["collider_present"]
    assert row.status == "warn"
    assert row.fix == ""
    assert report.status != "fail"


def test_a_document_with_a_collider_passes_collider_present():
    doc = bd.ClayDoc()
    source = doc.add_object(_grounded_box())
    doc.add_collider(source.uid, cl.fit_box(source.mesh))
    report = readiness.validate(doc)
    row = _by_key(report)["collider_present"]
    assert row.status == "pass"
    assert row.measured == 1


def test_collider_present_never_reaches_fail_even_with_many_missing():
    """"Advisory, never a fail" -- the brief's own words -- so however many
    render-side defects a document also has, this row's own status caps at
    warn."""
    doc = _doc(_grounded_box(scale=(-1.0, 1.0, 1.0)))  # also warns "transforms"
    report = readiness.validate(doc)
    assert _by_key(report)["collider_present"].status in ("warn", "pass")


def test_a_hidden_collider_is_ignored_unless_visible_only_is_false():
    doc = bd.ClayDoc()
    source = doc.add_object(_grounded_box())
    collider = doc.add_collider(source.uid, cl.fit_box(source.mesh))
    doc.set_visibility({collider.uid: False})

    report = readiness.validate(doc, visible_only=True)
    assert _by_key(report)["collider_present"].status == "warn"

    report = readiness.validate(doc, visible_only=False)
    assert _by_key(report)["collider_present"].status == "pass"


# --- collider_triangles -------------------------------------------------------


def test_collider_triangles_skips_with_no_collider():
    doc = _doc(_grounded_box())
    row = _by_key(readiness.validate(doc))["collider_triangles"]
    assert row.status == "skip"
    assert row.fix == ""


def test_a_light_collider_passes_triangle_budget():
    doc = bd.ClayDoc()
    source = doc.add_object(_grounded_box())
    doc.add_collider(source.uid, cl.fit_box(source.mesh))  # a box collider: 12 triangles
    row = _by_key(readiness.validate(doc))["collider_triangles"]
    assert row.status == "pass"
    assert row.measured == 12


def test_a_heavy_collider_warns_triangle_budget(monkeypatch):
    low = readiness.Profile(
        **{**readiness.PROFILES["godot-desktop"].__dict__, "collider_triangles_warn": 1}
    )
    monkeypatch.setitem(readiness.PROFILES, "_test_low_collider_tris", low)

    doc = bd.ClayDoc()
    source = doc.add_object(_grounded_box())
    collider = doc.add_collider(source.uid, cl.fit_box(source.mesh))  # 12 triangles > 1
    report = readiness.validate(doc, profile="_test_low_collider_tris")
    row = _by_key(report)["collider_triangles"]
    assert row.status == "warn"
    assert row.limit == 1
    assert collider.uid in row.uids
    # Never a fix -- see the module's own rule that these three rows never
    # name a remedy op.
    assert row.fix == ""


def test_collider_triangles_never_counted_against_the_render_budget():
    """The comment already guarding ``validate``'s own collider filter,
    turned into an assertion: a heavy collider must not make the *render*
    ``triangles`` row warn."""
    doc = bd.ClayDoc()
    source = doc.add_object(_grounded_box())
    doc.add_collider(source.uid, cl.fit_sphere(source.mesh))  # a real mesh, real triangles
    row = _by_key(readiness.validate(doc))["triangles"]
    assert row.status == "pass"


# --- collider_convex -----------------------------------------------------------


def test_collider_convex_skips_with_no_collider():
    doc = _doc(_grounded_box())
    row = _by_key(readiness.validate(doc))["collider_convex"]
    assert row.status == "skip"


def test_collider_convex_skips_for_a_box_collider_not_a_convex_one():
    """Box/sphere/capsule get an analytic collider on the engine side
    (``colliders.py``'s own module docstring); only convex/compound ever
    become a mesh a vertex cap could apply to."""
    doc = bd.ClayDoc()
    source = doc.add_object(_grounded_box())
    doc.add_collider(source.uid, cl.fit_box(source.mesh))
    row = _by_key(readiness.validate(doc))["collider_convex"]
    assert row.status == "skip"


def test_collider_convex_skips_on_a_profile_with_no_documented_cap():
    """Godot has no documented convex-hull vertex ceiling
    (``Profile.collider_convex_vertex_cap`` is ``None`` for it) -- a real
    convex collider still gets a "nothing to check against" skip, not a
    fabricated pass."""
    doc = bd.ClayDoc()
    source = doc.add_object(_grounded_box())
    doc.add_collider(source.uid, cl.convex_hull(source.mesh))
    row = _by_key(readiness.validate(doc, profile="godot-desktop"))["collider_convex"]
    assert row.status == "skip"
    assert "godot" in row.message.lower() or "documented" in row.message.lower()


def test_a_convex_collider_under_the_cap_passes_on_unity():
    doc = bd.ClayDoc()
    source = doc.add_object(_grounded_box())
    doc.add_collider(source.uid, cl.convex_hull(source.mesh))  # a box hull: 8 vertices
    row = _by_key(readiness.validate(doc, profile="unity"))["collider_convex"]
    assert row.status == "pass"
    assert row.limit == readiness.PROFILES["unity"].collider_convex_vertex_cap


def test_a_convex_collider_over_the_cap_warns_on_unity(monkeypatch):
    tight = readiness.Profile(
        **{**readiness.PROFILES["unity"].__dict__, "collider_convex_vertex_cap": 4}
    )
    monkeypatch.setitem(readiness.PROFILES, "_test_tight_cap", tight)

    doc = bd.ClayDoc()
    source = doc.add_object(_grounded_box())
    collider = doc.add_collider(source.uid, cl.convex_hull(source.mesh))  # 8 > 4
    report = readiness.validate(doc, profile="_test_tight_cap")
    row = _by_key(report)["collider_convex"]
    assert row.status == "warn"
    assert row.limit == 4
    assert collider.uid in row.uids
    assert row.fix == ""


def test_a_compound_collider_is_checked_as_one_concatenated_mesh(monkeypatch):
    """``compound``'s own stored mesh is every part's hull concatenated into
    one (``colliders.compound``'s own docstring) -- this row reads the whole
    thing, a conservative reading this module's own docstring states plainly
    rather than pretending part-level precision it does not have."""
    tight = readiness.Profile(
        **{**readiness.PROFILES["unity"].__dict__, "collider_convex_vertex_cap": 4}
    )
    monkeypatch.setitem(readiness.PROFILES, "_test_tight_cap", tight)

    doc = bd.ClayDoc()
    source = doc.add_object(_grounded_box())
    collider = doc.add_collider(source.uid, cl.compound(source.mesh))
    report = readiness.validate(doc, profile="_test_tight_cap")
    row = _by_key(report)["collider_convex"]
    assert row.status == "warn"
    assert collider.uid in row.uids


# --- CHECKS/FIX_OPS: shape and the existing bidirectional gate ---------------


def test_the_three_collider_checks_are_in_checks_and_labelled():
    for key in ("collider_present", "collider_triangles", "collider_convex"):
        assert key in readiness.CHECKS
        assert key in readiness._LABELS


def test_no_collider_row_ever_names_a_fix_op():
    """None of the three rows this tranche added name a remedy -- there is
    no single existing Clay op that "add a collider" or "shrink this
    collider" could honestly be (five different ``collider-*`` kinds exist,
    and none of them is a generic fix), so :data:`readiness.FIX_OPS`'s
    closed set is left exactly as it was. This is the other half of the
    brief's own "stay inside FIX_OPS's closed set, or join it deliberately"
    rule -- this tranche chose not to join it, which is only a safe choice
    if every row actually keeps its promise."""
    doc = bd.ClayDoc()
    source = doc.add_object(_grounded_box())
    doc.add_collider(source.uid, cl.convex_hull(source.mesh))
    report = readiness.validate(doc, profile="unity")
    for key in ("collider_present", "collider_triangles", "collider_convex"):
        assert _by_key(report)[key].fix == ""


def test_fix_ops_is_unchanged_by_the_three_new_rows():
    """``test_readiness.py``'s own
    ``test_every_fix_a_check_can_return_is_in_fix_ops`` is the bidirectional
    gate itself (every check that ever names a ``fix`` -> the set of names
    seen equals :data:`FIX_OPS` exactly, both directions) and keeps running
    unmodified as part of this same test package -- this is the narrower,
    local half: since none of the three rows this tranche added ever name a
    ``fix`` (:func:`test_no_collider_row_ever_names_a_fix_op` above), the
    closed set itself must be exactly what it was before this tranche
    touched the file."""
    assert frozenset(
        {"clean-mesh", "recalc-normals", "decimate", "bake", "drop-to-ground", "unwrap"}
    ) == readiness.FIX_OPS


# --- empty document: every new row skips, none fails --------------------------


def test_an_empty_document_skips_all_three_collider_rows():
    report = readiness.validate(_doc())
    checks = _by_key(report)
    for key in ("collider_present", "collider_triangles", "collider_convex"):
        assert checks[key].status == "skip"
