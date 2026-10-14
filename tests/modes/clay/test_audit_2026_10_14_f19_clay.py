"""The 2026-10-14 audit's rig-door findings: clay-rig-07 (a two-root rig published to the
library) and pipelines-children-01 ("Weight with Blender" on a one-shape document)."""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.rig import clayrig
from realmspinner.service import files as svc_files
from realmspinner.service.errors import Invalid
from realmspinner.studio.modes.clay import mode as clay_mode


def _add(doc: bd.ClayDoc, name: str, **kw) -> int:
    kw.setdefault("mesh", bp.box())
    return doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, **kw)).uid


def test_exporting_a_two_root_rig_to_the_library_is_refused_by_name(svc) -> None:
    good_doc = bd.ClayDoc()
    hip = _add(good_doc, "Hip", bone=True, translation=(0, 1, 0))
    _add(good_doc, "Arm", bone=True, parent=hip, translation=(0.5, 0.5, 0))
    job_id = clay_mode.build_asset(svc, good_doc, title="Hero")
    job_dir = svc.job_dir(job_id)
    published = (job_dir / "rig.json").read_bytes()
    glb = (job_dir / "rig.glb").read_bytes()

    two_roots = bd.ClayDoc()
    _add(two_roots, "Hip", bone=True)
    _add(two_roots, "Tail", bone=True, translation=(0, 0, 1))
    record = clayrig.rig_record(two_roots, place=np.eye(4))
    assert [b["parent"] for b in record["bones"]] == [None, None]

    with pytest.raises(Invalid, match=r"one root.*Hip.*Tail") as raised:
        svc_files.save_clay_rig(svc, job_id, glb, record)
    assert raised.value.field == "bones"
    assert (job_dir / "rig.json").read_bytes() == published, "a refusal leaves the rig alone"


def test_weight_with_blender_runs_on_a_document_with_a_single_shape() -> None:
    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True, translation=(0, 0.4, 0))
    _add(doc, "Spine", bone=True, parent=hip, translation=(0, 0.5, 0))
    _add(doc, "Body", translation=(0, 1.0, 0), scale=(0.6, 1.4, 0.4))
    scratch = clayrig.weight_source_doc(doc)
    [merged] = scratch.objects
    # The unit box is +-0.5 about the origin, so the body stands at y 0.3..1.7, x +-0.3.
    lo, hi = merged.mesh.positions.min(axis=0), merged.mesh.positions.max(axis=0)
    assert lo[1] == pytest.approx(0.3) and hi[1] == pytest.approx(1.7)
    assert lo[0] == pytest.approx(-0.3) and hi[0] == pytest.approx(0.3)
    assert not merged.bone and merged.skin is None
