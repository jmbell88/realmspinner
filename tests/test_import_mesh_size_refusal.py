"""F19-2: the mesh-import door refuses an unusable ``size_m`` with the service's
own ``Invalid`` naming the field, where a huge int escaped as a bare OverflowError
(the class create-42 closed in ``guidance.normalize``)."""

from __future__ import annotations

import pytest

from realmspinner.kernels.geom3d import glbwrite
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.service import jobs
from realmspinner.service.errors import Invalid


def _glb() -> bytes:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    return glbwrite.write_glb(bd.to_model(doc))


@pytest.mark.parametrize("bad", [10**400, "wide", float("nan"), float("inf")])
def test_import_mesh_refuses_an_unusable_size_m_as_invalid_naming_the_field(svc, bad) -> None:
    with pytest.raises(Invalid) as caught:
        jobs.import_mesh(svc, _glb(), size_m=bad)
    assert caught.value.field == "size_m"
