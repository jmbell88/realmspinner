"""Regression tests for the 2026-10-10 audit's clay-09 and clay-15.

clay-09: an undo step pins the ``Mesh`` objects it holds, and the weak-keyed
derived tables in ``adjacency.py`` live exactly as long as those meshes, but
``MeshEdit.cost`` counted only the mesh's own arrays -- so the byte budget
under-evicted on a large import.

clay-15: ``snapshot_bytes`` mirrored the reader's object, triangle, transform,
validity and array-size refusals but not its three document-wide ceilings, so
a document the writer accepted was one ``read_rblk`` refused.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from realmspinner.core.undo import UndoStack
from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import edits, ops_topo
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser

# --- clay-09 ----------------------------------------------------------------


def _grid(n: int) -> bm.Mesh:
    xs, ys = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    pos = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1).astype("f4")
    faces = [
        [j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i]
        for j in range(n)
        for i in range(n)
    ]
    return bm.from_faces(pos, faces)


def _build_every_derived_table(mesh: bm.Mesh) -> int:
    """Fill the four weak-keyed caches for *mesh*; the bytes they now hold."""
    a = adj.adjacency(mesh)
    t = adj.cached_triangulation(mesh)
    f8 = adj.cached_positions_f8(mesh)
    adj.closed_facts(mesh)
    return (
        sum(getattr(a, f.name).nbytes for f in dataclasses.fields(a))
        + sum(x.nbytes for x in t)
        + f8.nbytes
    )


def test_a_mesh_edit_in_history_does_not_pin_derived_adjacency_beyond_what_it_reports() -> None:
    """The 2026-10-10 audit's clay-09: ``MeshEdit.cost`` added up the two
    meshes' own arrays and nothing else, while the adjacency, triangulation and
    f8-position tables are weak-keyed to those same ``Mesh`` objects and so live
    as long as the step does -- 28.8 MB of tables on a 5.9 MB 160k-quad mesh,
    six extrude steps reporting 71 MB to the undo budget while pinning 154 MB.
    """
    before = _grid(60)
    after, _ = ops_topo.extrude_faces(before, el.ElementSel(faces=np.array([0])))
    pinned = _build_every_derived_table(before) + _build_every_derived_table(after)

    edit = edits.MeshEdit(1, before, after)

    own = edits.mesh_bytes(before) + edits.mesh_bytes(after)
    assert edit.cost - own >= pinned, (
        f"the step reports {edit.cost - own:,} bytes for derived tables "
        f"but pins {pinned:,}"
    )


def test_the_derived_cache_estimate_covers_what_each_cache_actually_holds() -> None:
    """The estimate is a function of counts, so it must be pinned against the
    real thing on meshes of different shapes -- a quad grid, a closed solid, a
    triangle fan and the empty mesh -- and must not balloon past a sane factor
    (a step that over-reports evicts history for no reason)."""
    meshes = [_grid(30), bp.uv_sphere(), bp.cylinder(), bm.from_faces(np.zeros((0, 3), "f4"), [])]
    for mesh in meshes:
        actual = _build_every_derived_table(mesh) if len(mesh.starts) > 1 else 0
        guess = adj.derived_cache_bytes(mesh)
        assert guess >= actual
        assert guess <= 2 * actual + 64


def test_history_with_large_meshes_reports_the_caches_it_pins() -> None:
    """The probe that found clay-09, shrunk: a stack of extrudes whose meshes
    have all been picked against (as any edit does) must report at least what
    the weak-keyed tables hold, which is what the byte budget spends."""
    cur = _grid(40)
    steps = UndoStack()
    pinned = 0
    for i in range(4):
        nxt, _ = ops_topo.extrude_faces(cur, el.ElementSel(faces=np.array([i])))
        pinned += _build_every_derived_table(cur)
        steps.push(edits.MeshEdit(1, cur, nxt))
        cur = nxt
    pinned += _build_every_derived_table(cur)

    assert steps.bytes >= pinned


# --- clay-15 ----------------------------------------------------------------


def _tex(seed: int = 0, side: int = 2) -> tuple[int, int, bytes]:
    return (side, side, bytes((seed + i) % 256 for i in range(side * side * 4)))


def _doc_with_textures(count: int, side: int = 2) -> bd.ClayDoc:
    materials = [
        gltf.Material(name=f"tex{i}", base_color=_tex(i, side)) for i in range(count)
    ]
    return bd.ClayDoc(
        objects=[bd.Obj(uid=bd.new_uid(), name="P", mesh=bp.plane())],
        materials=materials or None,
    )


def test_the_writer_refuses_a_document_whose_decoded_textures_pass_the_budget_the_reader_enforces(
    monkeypatch,
) -> None:
    """The 2026-10-10 audit's clay-15: ``read_rblk`` charges ``width*height*4``
    per texture against ``MAX_TOTAL_TEXTURE_BYTES`` and refuses past it, and
    ``snapshot_bytes`` charged nothing -- about 193 distinct 1024 x 1024 palette
    textures saved, autosaved and journalled a file that would not reopen,
    crash recovery included. Lowered through the module global the way the
    reader's own test does, so it needs no 768 MiB of RAM."""
    monkeypatch.setattr(ser, "MAX_TOTAL_TEXTURE_BYTES", 40)  # two 16-byte textures fit
    doc = _doc_with_textures(3)  # 48 decoded bytes

    with pytest.raises(ValueError, match="decoded texture") as caught:
        ser.rblk_bytes(doc)
    assert "tex2" in str(caught.value)  # the material whose texture crossed the line


def test_a_document_the_writer_accepts_at_the_texture_budget_the_reader_reads(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ser, "MAX_TOTAL_TEXTURE_BYTES", 48)  # exactly three 16-byte textures
    doc = _doc_with_textures(3)

    out = ser.read_rblk(ser.rblk_bytes(doc))

    assert len(out.materials) == 3


def test_the_writer_refuses_more_materials_than_the_reader_reads(monkeypatch) -> None:
    """The 2026-10-10 audit's clay-15, the materials half: ``read_rblk``
    refuses a scene declaring more than ``MAX_DECLARED_MATERIALS``."""
    monkeypatch.setattr(ser, "MAX_DECLARED_MATERIALS", 3)
    doc = bd.ClayDoc(
        objects=[bd.Obj(uid=bd.new_uid(), name="P", mesh=bp.plane())],
        materials=[gltf.Material(name=f"m{i}") for i in range(4)],
    )

    with pytest.raises(ValueError, match="materials"):
        ser.rblk_bytes(doc)


def test_a_document_at_the_material_ceiling_is_written_and_read_back(monkeypatch) -> None:
    monkeypatch.setattr(ser, "MAX_DECLARED_MATERIALS", 3)
    doc = bd.ClayDoc(
        objects=[bd.Obj(uid=bd.new_uid(), name="P", mesh=bp.plane())],
        materials=[gltf.Material(name=f"m{i}") for i in range(3)],
    )

    assert len(ser.read_rblk(ser.rblk_bytes(doc)).materials) == 3


def test_the_writer_refuses_more_textures_than_the_reader_reads(monkeypatch) -> None:
    """The textures half: ``MAX_DECLARED_TEXTURES`` bounds how many images a
    scene may name, and the writer named as many as the palette held."""
    monkeypatch.setattr(ser, "MAX_DECLARED_TEXTURES", 2)
    doc = _doc_with_textures(3)

    with pytest.raises(ValueError, match="textures"):
        ser.rblk_bytes(doc)


def test_a_document_at_the_texture_count_ceiling_is_written_and_read_back(monkeypatch) -> None:
    monkeypatch.setattr(ser, "MAX_DECLARED_TEXTURES", 3)
    doc = _doc_with_textures(3)

    assert len(ser.read_rblk(ser.rblk_bytes(doc)).materials) == 3
