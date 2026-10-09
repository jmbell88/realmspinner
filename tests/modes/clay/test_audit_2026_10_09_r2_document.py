"""The 2026-10-09 second audit's document, writer and scratch findings:
clay-57 (the kernel door), 82, 86, 91, 93, 102, 104 and 105.

Each test name is the claim, and each failed against the code it was written for.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import glbimport, ops, serialize
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import scratch as clay_scratch
from realmspinner.kernels.mesh import separate as sep

MESH_DIR = Path(bd.__file__).parent


def _obj(name: str = "o", mesh: bm.Mesh | None = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(uid=bd.new_uid(), name=name, mesh=mesh or bp.box(), **kwargs)  # type: ignore[arg-type]


def _painted(slot: int) -> bm.Mesh:
    box = bp.box()
    return bm.Mesh(
        positions=box.positions,
        loops=box.loops,
        starts=box.starts,
        material=np.full(len(box.material), slot, dtype="i4"),
        smooth=box.smooth,
        uv=box.uv,
    )


# --- clay-57 -----------------------------------------------------------------


def test_removing_a_ctrl_clicked_unused_slot_leaves_the_objects_default_slot_alone() -> None:
    """The panes' Remove button targets the Ctrl+clicked edit slot, which is not
    the object's own default.  The door used to repoint the default at
    ``min(index, last)`` regardless, so removing the unused slot 0 rewrote a
    shape wearing slot 2 to slot 0 -- while its faces were renumbered to 1."""
    doc = bd.ClayDoc()
    doc.add_material()
    doc.add_material()  # slots 0 (starter), 1, 2
    obj = doc.add_object(_obj(mesh=_painted(2), material=2))
    assert doc.material_users(0) == 0

    assert doc.remove_material_and_reassign(obj.uid, 0) is True

    assert set(obj.mesh.material.tolist()) == {1}
    assert obj.material == 1, "the default followed its material down one slot, no further"


def test_removing_the_slot_an_object_defaults_to_still_repoints_it_at_what_is_left() -> None:
    doc = bd.ClayDoc()
    doc.add_material()
    doc.add_material()
    obj = doc.add_object(_obj(mesh=_painted(0), material=1))  # default 1, faces 0
    assert doc.material_users(1) == 0

    assert doc.remove_material_and_reassign(obj.uid, 1) is True

    assert 0 <= obj.material < len(doc.materials)
    assert obj.material == 1, "slot 2 slid into slot 1, the nearest survivor"


def test_a_bystander_default_is_not_rewritten_by_remove_material_and_reassign() -> None:
    doc = bd.ClayDoc()
    doc.add_material()
    doc.add_material()
    mover = doc.add_object(_obj("mover", mesh=_painted(2), material=2))
    other = doc.add_object(_obj("other", mesh=_painted(1), material=1))

    assert doc.remove_material_and_reassign(other.uid, 0) is True

    assert (mover.material, other.material) == (1, 0)
    assert doc.undo()
    assert (mover.material, other.material) == (2, 1)


# --- clay-82 -----------------------------------------------------------------


def _one_corner_face() -> bm.Mesh:
    return bm.Mesh(
        positions=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype="f4"),
        loops=np.array([0, 1, 2, 3, 0], dtype="i4"),
        starts=np.array([0, 4, 5], dtype="i4"),
        material=np.zeros(2, dtype="i4"),
        smooth=np.zeros(2, dtype=bool),
    )


def test_rblk_bytes_refuses_a_mesh_read_rblk_would_refuse_to_reopen() -> None:
    doc = bd.ClayDoc()
    doc.add_object(_obj("Sheet", mesh=_one_corner_face()))
    with pytest.raises(ValueError, match="Sheet"):
        serialize.rblk_bytes(doc)


def test_set_mesh_refuses_a_mesh_that_is_not_a_valid_csr_and_changes_nothing() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(_obj("Sheet"))
    before, steps = obj.mesh, len(doc.history)
    with pytest.raises(el.OpError, match="Sheet"):
        doc.set_mesh(obj.uid, _one_corner_face())
    assert obj.mesh is before and len(doc.history) == steps


def test_the_mesh_module_header_does_not_claim_rebuild_validates() -> None:
    header = bm.__doc__ or ""
    assert "funnels through" not in header, (
        "topo.rebuild assembles a Mesh and validates nothing; the writer and set_mesh do"
    )


# --- clay-86 / clay-93 -------------------------------------------------------


def test_transplanting_a_hide_deselects_the_hidden_object() -> None:
    doc = bd.ClayDoc()
    a, b = _obj("A"), _obj("B")
    doc.add_objects([a, b])
    doc.select([a.uid])
    sc = clay_scratch.clone(doc)
    sc.set_visibility({a.uid: False})
    diff = clay_scratch.diff(doc, sc)
    assert diff.props_changed == {a.uid: {"visible"}}

    assert clay_scratch.transplant(doc, sc, diff)

    assert doc.by_uid(a.uid).visible is False
    assert doc.selection_problems() == []
    assert a.uid not in doc.selection


def test_transplanting_a_hide_is_still_one_undo_step() -> None:
    doc = bd.ClayDoc()
    a, b = _obj("A"), _obj("B")
    doc.add_objects([a, b])
    sc = clay_scratch.clone(doc)
    sc.set_visibility({a.uid: False})
    sc.set_props(b.uid, name="Renamed")
    diff = clay_scratch.diff(doc, sc)
    steps = len(doc.history)

    assert clay_scratch.transplant(doc, sc, diff)

    assert len(doc.history) == steps + 1
    assert doc.undo()
    assert doc.by_uid(a.uid).visible is True and doc.by_uid(b.uid).name == "B"


def test_transplanted_params_do_not_alias_the_scratch_objects_params() -> None:
    doc = bd.ClayDoc()
    o = doc.add_object(
        _obj("C", generator="cylinder", params={"radius": 0.5, "size": [1.0, 1.0, 1.0]})
    )
    sc = clay_scratch.clone(doc)
    sc.set_props(o.uid, params={"radius": 0.7, "size": [2.0, 2.0, 2.0]})
    diff = clay_scratch.diff(doc, sc)
    assert clay_scratch.transplant(doc, sc, diff)

    live, ghost = doc.by_uid(o.uid), sc.by_uid(o.uid)
    assert live.params == ghost.params
    assert live.params is not ghost.params
    ghost.params["radius"] = 99.0
    ghost.params["size"][0] = 99.0
    assert live.params["radius"] == 0.7
    assert live.params["size"] == [2.0, 2.0, 2.0], "a nested list is copied too"


def test_the_scratch_module_docstring_says_transplant_copies() -> None:
    text = " ".join((clay_scratch.__doc__ or "").replace("*", "").split())
    assert "moves the scratch's own objects" not in text
    assert "copy" in text.split("Transplant replays no tool call.")[1]


# --- clay-91 -----------------------------------------------------------------


def test_no_kernels_mesh_docstring_cites_the_dead_viewer_math3d_or_gltf() -> None:
    dead = re.compile(r"\bviewer\.(gltf|math3d)\b")
    found = [
        f"{name}:{n}: {line.strip()}"
        for name in ("document.py", "edits.py", "ops.py")
        for n, line in enumerate((MESH_DIR / name).read_text(encoding="utf-8").splitlines(), 1)
        if dead.search(line)
    ]
    assert not found, "\n".join(found)


# --- clay-102 ----------------------------------------------------------------


def test_set_generator_params_with_the_current_values_pushes_no_step_and_stays_clean() -> None:
    doc = bd.ClayDoc()
    defaults, build = bp.GENERATORS["cylinder"]
    obj = doc.add_object(
        _obj("c", mesh=build(**defaults), generator="cylinder", params=dict(defaults))
    )
    doc.mark_saved()
    steps, rev = len(doc.history), doc.rev
    rebuilt = build(**bp.clamp_params("cylinder", dict(obj.params)))
    assert rebuilt is not obj.mesh

    changed = doc.set_generator_params(
        obj.uid, dict(obj.params), rebuilt, was={"params": dict(obj.params)}
    )

    assert changed is False
    assert not doc.dirty and len(doc.history) == steps
    assert doc.rev == rev


def test_set_generator_params_still_pushes_a_step_when_only_the_mesh_differs() -> None:
    doc = bd.ClayDoc()
    defaults, build = bp.GENERATORS["cylinder"]
    obj = doc.add_object(
        _obj("c", mesh=build(**defaults), generator="cylinder", params=dict(defaults))
    )
    taller = bp.clamp_params("cylinder", {**defaults, "height": float(defaults["height"]) + 1.0})
    assert doc.set_generator_params(
        obj.uid, taller, build(**taller), was={"params": dict(defaults)}
    )
    assert doc.dirty


# --- clay-104 ----------------------------------------------------------------


def _fresh_total(doc: bd.ClayDoc) -> int:
    return sum(serialize.triangle_count(o.mesh) for o in doc.objects)


def _grown() -> bm.Mesh:
    defaults, build = bp.GENERATORS["uv_sphere"]
    return build(**defaults)


def test_set_mesh_does_not_resum_the_whole_document_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = bd.ClayDoc()
    objs = doc.add_objects([_obj(f"o{i}") for i in range(300)])
    doc.set_mesh(objs[0].uid, _grown())  # warms whatever the ceiling check keeps
    calls = 0
    real = bd._triangles_of

    def counting(mesh: bm.Mesh) -> int:
        nonlocal calls
        calls += 1
        return real(mesh)

    monkeypatch.setattr(bd, "_triangles_of", counting)
    for obj in objs[1:6]:
        assert doc.set_mesh(obj.uid, _grown())
    assert calls < 5 * 20, f"{calls} per-mesh lookups for 5 growing set_mesh calls over 300 objects"


def _warm(doc: bd.ClayDoc) -> None:
    assert doc._triangle_total() == _fresh_total(doc)


def test_the_triangle_total_cannot_go_stale_across_any_writer() -> None:
    doc = bd.ClayDoc()
    a, b, c = doc.add_objects([_obj("a"), _obj("b"), _obj("c")])

    def check(label: str) -> None:
        assert doc._triangle_total() == _fresh_total(doc), label

    _warm(doc)
    doc.set_mesh(a.uid, _grown())
    check("set_mesh grows")
    doc.set_mesh(a.uid, bp.box())
    check("set_mesh shrinks")
    assert doc.undo()
    check("undo")
    assert doc.redo()
    check("redo")
    # A caller that drives the stack directly, bypassing ClayDoc.undo's touch.
    assert doc.history.undo(doc)
    check("history.undo")
    assert doc.history.redo(doc)
    check("history.redo")
    extra = doc.add_object(_obj("extra", mesh=_grown()))
    check("add_object")
    doc.remove_object(extra.uid)
    check("remove_object")
    assert doc.undo()
    check("undo of a remove")
    joined = ops.join([a, b], eps=0.0, world=[doc.world_matrix(a.uid), doc.world_matrix(b.uid)])
    doc.join_objects(a.uid, joined, [b.uid])
    check("join_objects")
    pieces = sep.by_loose_parts(doc.by_uid(a.uid).mesh)
    assert len(pieces) >= 2
    doc.separate(a.uid, pieces)
    check("separate")
    defaults, build = bp.GENERATORS["uv_sphere"]
    g = doc.add_object(
        _obj("g", mesh=build(**defaults), generator="uv_sphere", params=dict(defaults))
    )
    check("add generator object")
    bigger = bp.clamp_params("uv_sphere", {**defaults, "segments": int(defaults["segments"]) * 2})
    assert doc.set_generator_params(g.uid, bigger, build(**bigger), was={"params": dict(defaults)})
    check("set_generator_params")
    sc = clay_scratch.clone(doc)
    sc.add_object(_obj("ghost", mesh=_grown()))
    sc.set_mesh(c.uid, _grown())
    diff = clay_scratch.diff(doc, sc)
    assert clay_scratch.transplant(doc, sc, diff)
    check("scratch transplant")
    assert doc.undo()
    check("undo of a transplant")


def test_the_ceiling_still_refuses_after_the_total_has_been_carried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = bd.ClayDoc()
    a, b = doc.add_objects([_obj("a"), _obj("b")])
    doc.set_mesh(a.uid, _grown())  # carries the total forward
    monkeypatch.setattr(glbimport, "MAX_TRIANGLES", _fresh_total(doc) + 1)
    with pytest.raises(el.OpError, match="triangles"):
        doc.set_mesh(b.uid, _grown())


# --- clay-105 ----------------------------------------------------------------


def test_the_module_docstring_lists_set_element_mode_among_the_selection_writers() -> None:
    text = " ".join((bd.__doc__ or "").split())
    assert "set_element_mode" in text
    assert not re.search(r"mode switch.{0,80}never touch it", text)
