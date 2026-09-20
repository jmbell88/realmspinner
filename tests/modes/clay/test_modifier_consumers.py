"""Tranche 2's modifier stack: every DISPLAY/EXPORT/MEASURE consumer this
session touched reads the *evaluated* mesh -- the base run through the
modifier stack -- while editing keeps reading the base untouched. One
scenario runs through every consumer: a unit box shifted off the local
origin, carrying a mirror-on-X modifier with no weld, so the evaluated mesh
unambiguously doubles the base (a welded seam would make "did this read
the evaluated mesh" a question of a merge tolerance rather than a plain
count).

Every test here is built to fail against the unmodified consumer: before
this session's fix, each of these read ``obj.mesh`` (or, for picking,
raycast only the base mesh's own footprint) instead of
``doc.evaluated(obj.uid)``, so the mirror modifier was invisible to every one
of them. See the sanity-check note on ``test_the_view_cache_builds_the_evaluated_vertex_count``
and ``test_object_mode_pick_hits_the_mirrored_half`` for how that was checked
by reasoning.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels.mesh import analyze as clay_analyze
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import modifiers as mod
from realmspinner.kernels.mesh import objexport, readiness
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay.ui.panes import bridge as clay_bridge

# Reused rather than a second copy, the way ``test_clay_view_cache.py`` already
# reuses this file's GL-backed ``view`` fixture (its own docstring states why).
from .test_clay_view import view as view  # noqa: F401, PLC0414


def _half_box_doc() -> tuple[bd.ClayDoc, bd.Obj]:
    """A unit box shifted to x=2 -- nowhere near the local x=0 mirror plane,
    so a weld-off mirror concatenates two disjoint boxes rather than merging
    a seam (the same reasoning ``test_modifiers.py``'s own ``_shifted``
    states) -- carrying a mirror-on-X modifier. The evaluated mesh spans both
    x~2 (the base) and x~-2 (the reflection); the base mesh alone spans only
    x~2."""
    doc = bd.ClayDoc()
    mesh = bp.box()
    positions = np.array(mesh.positions, dtype="f8")
    positions[:, 0] += 2.0
    shifted = replace(mesh, positions=positions)
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Half", mesh=shifted))
    doc.set_modifiers(obj.uid, (mod.make("mirror", {"axis": 0, "weld": 0.0}, id=1),))
    return doc, obj


# --- the GPU cache (ui/_view_cache.py) ---------------------------------------


def test_the_view_cache_builds_the_evaluated_vertex_count(view) -> None:
    """Sanity check (by reasoning): before this session's fix, ``_build``
    called ``to_primitives(obj, doc.materials)`` with no ``mesh`` argument,
    which defaults to ``obj.mesh`` -- the base, 8 vertices -- and ``_Entry.mesh``
    pinned that same base object. Both assertions below would then read 8,
    not 16, so this fails against the unmodified consumer."""
    doc, obj = _half_box_doc()
    view.sync(doc)
    entry = view._cache[obj.uid]

    assert len(obj.mesh.positions) == 8  # the base: one box
    assert entry.mesh is doc.evaluated(obj.uid)
    assert len(entry.mesh.positions) == 16  # base + its unwelded reflection


def test_the_view_cache_does_not_rebuild_on_an_unrelated_second_sync(view) -> None:
    doc, _obj = _half_box_doc()
    view.sync(doc)
    before = view.rebuilds

    view.sync(doc)
    view.sync(doc)
    assert view.rebuilds == before


def test_the_view_cache_rebuilds_when_a_modifier_param_changes(view) -> None:
    doc, obj = _half_box_doc()
    view.sync(doc)
    before = view.rebuilds

    doc.set_modifiers(obj.uid, (mod.with_params(obj.modifiers[0], {"axis": "Y"}),))
    view.sync(doc)
    assert view.rebuilds == before + 1


# --- world bounds (ui/_view_bounds.py) ---------------------------------------


def test_world_bounds_covers_the_mirrored_half(view) -> None:
    """The base mesh alone spans x in [1.5, 2.5] -- entirely positive -- so a
    lower bound past -1.0 can only have come from the evaluated (mirrored)
    mesh reaching into negative x. Before the fix, ``_object_world_box``
    passed ``bops.world_box(obj)`` with no override, which boxes the base
    alone, so ``lo[0]`` would read ~1.5, failing the assertion below."""
    doc, _obj = _half_box_doc()
    lo, hi = view.world_bounds(doc)

    assert lo is not None
    assert lo[0] < -1.0
    assert hi[0] > 1.0


# --- object-mode picking (ui/_view_pick.py) ----------------------------------


def test_object_mode_pick_hits_the_mirrored_half(view, monkeypatch) -> None:
    """A ray aimed straight down at the mirrored copy's own position
    (~x=-2), nowhere near the base mesh (~x=2) -- so a hit here can only have
    come from the evaluated mesh. Before the fix, ``pick`` called
    ``pick_face`` with no ``evaluated=True``, which tests ``obj.mesh`` (the
    base, sitting at x~2) against every ray, so this ray would miss and
    ``pick`` would return ``None``, not the object's uid."""
    doc, obj = _half_box_doc()
    monkeypatch.setattr(view, "_ray", lambda local: ((-2.0, 0.0, 10.0), (0.0, 0.0, -1.0)))
    assert view.pick(doc, (0.0, 0.0)) == obj.uid


# --- analyze.py ----------------------------------------------------------------


def test_analyze_measures_the_evaluated_mesh() -> None:
    """A unit box's own surface area is 6 m^2; the unwelded mirror doubles it
    to 12 -- true only of the evaluated mesh, since the base alone is still
    one box. Before the fix, ``analyze`` had no ``doc`` parameter at all and
    always measured whatever ``.mesh`` the caller handed it (the base), so
    this would read 6.0, not 12.0."""
    doc, obj = _half_box_doc()
    analysis = clay_analyze.analyze([obj], doc=doc)
    row = analysis.objects[0]

    assert row.area == pytest.approx(12.0)


def test_analyze_with_no_doc_still_measures_the_base() -> None:
    """The back-compat path: every existing caller with no document in hand
    (this module is duck-typed, per its own docstring) keeps measuring
    exactly the mesh it was handed -- here, the *base*, since nothing
    evaluated it first."""
    doc, obj = _half_box_doc()
    analysis = clay_analyze.analyze([obj])
    row = analysis.objects[0]

    assert row.area == pytest.approx(6.0)


# --- readiness.py ----------------------------------------------------------------


def test_readiness_counts_evaluated_triangles() -> None:
    """A box triangulates to 12 (6 quads x 2 each); the unwelded mirror
    doubles the face count, so the evaluated mesh is 24. Before the fix,
    ``validate`` built its ``objects`` list straight off ``doc.objects``
    (the base meshes), so this would read 12, not 24."""
    doc, _obj = _half_box_doc()
    report = readiness.validate(doc)
    by_key = {check.key: check for check in report.checks}

    assert by_key["triangles"].measured == 24


# --- objexport.py ----------------------------------------------------------------


def test_obj_export_writes_evaluated_faces() -> None:
    """6 quad faces from the base, 6 more from its unwelded reflection.
    Before the fix, ``claydoc_to_obj`` baked ``obj`` directly (the base), so
    this would count 6 ``f`` lines, not 12."""
    doc, _obj = _half_box_doc()
    obj_text, _mtl_text = objexport.claydoc_to_obj(doc)
    face_lines = [line for line in obj_text.splitlines() if line.startswith("f ")]

    assert len(face_lines) == 12


# --- bridge.py `_facts` -----------------------------------------------------------


def test_bridge_facts_counts_evaluated_triangles(monkeypatch) -> None:
    """``_facts`` cannot be driven headlessly (no imgui context in this
    suite), so ``widgets.muted`` is captured rather than rendered -- the same
    technique ``test_quality_badge.py`` uses for the same reason. Before the
    fix, ``_facts`` summed ``_triangles(obj.mesh)`` (the base), so this line
    would read "12 triangles", not "24 triangles"."""
    doc, _obj = _half_box_doc()
    drawn: list[str] = []
    monkeypatch.setattr(clay_bridge.widgets, "muted", lambda text: drawn.append(text))

    clay_bridge._facts(SimpleNamespace(doc=doc))

    assert drawn, "the facts line should have drawn something"
    assert "24 triangles" in drawn[0]
