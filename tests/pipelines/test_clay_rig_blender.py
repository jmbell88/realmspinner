"""A Clay document weighted by a real Blender run and adopted back (Tranche 4 of the Clay rig plan).

The chain the "Weight with Blender" button runs, minus the queue: the visible non-bone shapes
merged into one mesh and grounded the way a library row grounds it, the document's bones handed
to ``op_rig`` as a custom skeleton in the row's frame, the ``rig.glb`` it writes read back as a
document, and ``ClayDoc.adopt_rig`` putting the weights and Blender's rest frames into the
original. Needs ``bpy``; skipped without it, like every sibling in this directory.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import glbimport, rig
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.rig import blender_spec, clayrig, store


def _figure() -> tuple[bd.ClayDoc, dict[str, int]]:
    """A torso, a sleeve and three bones: Hip (the root) at the waist, Spine, and Arm."""
    doc = bd.ClayDoc()
    ids: dict[str, int] = {}

    def add(name, **kw):
        kw.setdefault("mesh", bp.box())
        ids[name] = doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, **kw)).uid

    add("Hip", bone=True, translation=(0.0, 0.4, 0.0))
    add("Spine", bone=True, parent=ids["Hip"], translation=(0.0, 0.5, 0.0))
    add("Arm", bone=True, parent=ids["Spine"], translation=(0.35, 0.3, 0.0))
    add("Torso", translation=(0.0, 1.0, 0.0), scale=(0.6, 1.4, 0.4))
    add("Sleeve", translation=(0.8, 1.2, 0.0), scale=(0.9, 0.2, 0.2))
    return doc, ids


def test_blender_weights_a_clay_figure_and_the_document_adopts_them(tmp_path):
    bpy = pytest.importorskip("bpy")
    from realmspinner.pipelines import blender_worker, postprocess

    doc, ids = _figure()
    sources = clayrig.weight_sources(doc)
    assert {o.name for o in sources} == {"Torso", "Sleeve"}

    # The merged mesh, grounded as a library row grounds it.
    model = tmp_path / "model.glb"
    merged = clayrig.weight_source_doc(doc)
    model.write_bytes(glbwrite.write_glb(bd.to_model(merged, skinned=False)))
    moved = postprocess.normalize_glb(model, None)
    place = np.diag([moved["scale"]] * 3 + [1.0])
    place[:3, 3] = moved["translation"]

    skeleton = clayrig.heat_skeleton(doc, place)
    assert [b["name"] for b in skeleton["bones"]] == ["Hip", "Spine", "Arm"]
    spec = blender_spec.rig_spec(
        tmp_path, "blank", skeleton["bones"], skeleton="custom", root="Hip",
        mirror_pairs=skeleton["mirror_pairs"],
    )
    result = blender_worker.op_rig(bpy, spec)
    assert result["ok"] is True, result
    store.finalize_rig(tmp_path)

    imported, _ = glbimport.glb_to_claydoc_report((tmp_path / "rig.glb").read_bytes(), name="rig")
    assert {o.name for o in imported.objects if o.bone} >= {"Hip", "Spine", "Arm"}

    before = {o.uid for o in doc.objects}
    counts = doc.adopt_rig(
        imported,
        bone_names=dict(rig.export_names(doc.objects)),
        to_doc=np.linalg.inv(place),
        remove=[o.uid for o in sources],
    )
    assert counts["matched"] == 3 and counts["meshes"] >= 1 and counts["removed"] == 2
    assert doc.rig_problems() == []

    body = next(o for o in doc.objects if o.skin is not None)
    assert body.uid not in before
    weights = body.mesh.weights.astype("f8")
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-3)

    # At rest the skin is where the shapes were: the weights replaced them, not moved them.
    palette = doc.skin_palette(body, {})
    rest = rig.skin_positions(body.mesh.positions, body.mesh.joints, body.mesh.weights, palette)
    assert np.allclose(rest, body.mesh.positions, atol=1e-5)
    lo, hi = body.mesh.positions.min(axis=0), body.mesh.positions.max(axis=0)
    assert lo[1] == pytest.approx(0.3, abs=0.05) and hi[1] == pytest.approx(1.7, abs=0.05)

    # And the weights mean something: turning the arm moves the sleeve's end and not the feet.
    arm = ids["Arm"]
    turn = np.array([0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)])
    posed = rig.skin_positions(
        body.mesh.positions, body.mesh.joints, body.mesh.weights,
        doc.skin_palette(body, {arm: {"rotation": turn}}),
    )
    moved_by = np.linalg.norm(posed - body.mesh.positions, axis=1)
    sleeve_end = int(np.argmax(body.mesh.positions[:, 0]))
    low = int(np.argmin(body.mesh.positions[:, 1]))
    assert moved_by[sleeve_end] > 0.1, "the arm's far end did not follow the arm bone"
    assert moved_by[low] < 0.02, "the lowest point should be nowhere near the arm"

    # One undo puts the original shapes back.
    steps = len(doc.history)
    doc.undo()
    assert len(doc.history) == steps - 1
    assert {o.name for o in doc.objects} == {"Hip", "Spine", "Arm", "Torso", "Sleeve"}
