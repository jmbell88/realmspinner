"""The 2026-10-03 audit's Clay follow-ups: the element counts and measure line
agree with what a drag would move (clay-70), the HUD volume agrees with
``clay_measure`` about an open mesh (clay-25), and ``next_name`` stops being
quadratic (the naming of a run of copies).
"""

from __future__ import annotations

import time

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops as mesh_ops
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import viewport_hints as clay_hints
from realmspinner.studio.modes.clay.ui.panes import props as clay_props


def _two_boxes() -> tuple[bd.ClayDoc, bd.Obj, bd.Obj]:
    doc = bd.ClayDoc()
    first = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    second = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    return doc, first, second


def _open_box() -> bm.Mesh:
    box = bp.box()
    starts, loops = box.starts, box.loops
    faces = [loops[starts[i] : starts[i + 1]].tolist() for i in range(bm.face_count(box) - 1)]
    return bm.from_faces(box.positions, faces)


# --- (a) Properties' "N selected" and the HUD follow the drag's eligibility --


def test_properties_selected_count_ignores_a_hidden_objects_element_selection() -> None:
    doc, first, second = _two_boxes()
    doc.set_element_mode("face")
    doc.set_element_sel(first.uid, el.ElementSel(faces=[0, 1]))
    doc.set_element_sel(second.uid, el.ElementSel(faces=[0, 1, 2]))
    assert clay_props.element_summary_text(doc) == "face mode -- 5 faces across 2 objects"

    doc.set_props(second.uid, visible=False)

    text = clay_props.element_summary_text(doc)
    assert text == "face mode -- 2 faces across 1 object", text


def test_properties_with_only_a_hidden_selection_says_nothing_selected() -> None:
    doc, first, _second = _two_boxes()
    doc.set_element_mode("edge")
    doc.set_element_sel(first.uid, el.ElementSel(edges=[[0, 1]]))
    doc.set_props(first.uid, visible=False)
    assert clay_props.element_summary_text(doc) == "edge mode -- nothing selected"


def test_hud_selected_count_and_measure_ignore_a_hidden_objects_element_selection() -> None:
    doc, first, second = _two_boxes()
    doc.set_element_mode("face")
    doc.set_element_sel(first.uid, el.ElementSel(faces=[0]))
    doc.set_element_sel(second.uid, el.ElementSel(faces=[0, 1]))
    assert "3 faces selected" in clay_hints.stats(doc)
    assert clay_hints.measure_line(doc) == "area  3.0000 m²"

    doc.set_props(second.uid, visible=False)

    assert "1 face selected" in clay_hints.stats(doc)
    assert clay_hints.measure_line(doc) == "area  1.0000 m²"


def test_hud_distance_ignores_a_hidden_objects_selected_vertices() -> None:
    doc, first, second = _two_boxes()
    doc.set_element_mode("vertex")
    doc.set_element_sel(first.uid, el.ElementSel(verts=[0]))
    doc.set_element_sel(second.uid, el.ElementSel(verts=[1]))
    assert clay_hints.measure_line(doc).startswith("distance")

    doc.set_props(second.uid, visible=False)

    assert clay_hints.measure_line(doc) == ""


# --- (b) the HUD volume of an open mesh ---------------------------------------


def test_hud_volume_of_an_open_mesh_is_not_a_number() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Open", mesh=_open_box()))
    doc.select([obj.uid])

    line = clay_hints.measure_line(doc)

    assert "m³" not in line, line
    assert "open" in line, line


def test_hud_volume_of_a_closed_mesh_is_unchanged_and_one_open_one_poisons_the_sum() -> None:
    doc = bd.ClayDoc()
    closed = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Closed", mesh=bp.box()))
    doc.select([closed.uid])
    assert clay_hints.measure_line(doc) == "volume  1.0000 m³"

    open_obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Open", mesh=_open_box()))
    doc.select([closed.uid, open_obj.uid])
    assert "m³" not in clay_hints.measure_line(doc)


# --- (c) next_name is not quadratic ---------------------------------------------


def test_next_name_does_not_copy_a_set_it_is_given(monkeypatch) -> None:
    constructed = []

    class CountingSet(set):
        def __init__(self, *args):
            super().__init__(*args)
            constructed.append(1)

    taken = CountingSet({"Box", "Box.001"})
    constructed.clear()
    # ``set`` resolves through the module's globals first, so this counts
    # exactly the constructions ``next_name`` itself makes.
    monkeypatch.setattr(mesh_ops, "set", CountingSet, raising=False)
    for _ in range(50):
        assert mesh_ops.next_name("Box", taken) == "Box.002"
    assert constructed == [], "a set must be read in place, not copied once per call"


def test_next_name_still_accepts_a_list_a_tuple_and_a_generator() -> None:
    assert mesh_ops.next_name("Box", ["Box", "Box.001"]) == "Box.002"
    assert mesh_ops.next_name("Box", ("Box", "Box.002")) == "Box.001"
    assert mesh_ops.next_name("Box.004", (n for n in ("Box.005",))) == "Box.006"
    assert mesh_ops.next_name("Box") == "Box.001"
    assert mesh_ops.next_name("Box", frozenset({"Box.001"})) == "Box.002"


def test_next_name_never_mutates_the_set_it_reads() -> None:
    taken = {"Box"}
    mesh_ops.next_name("Box", taken)
    assert taken == {"Box"}


def _naming_seconds(count: int) -> float:
    taken = mesh_ops.UsedNames({"Box"})
    start = time.perf_counter()
    for _ in range(count):
        taken.add(mesh_ops.next_name("Box", taken))
    return time.perf_counter() - start


def test_naming_copies_does_not_grow_quadratically() -> None:
    small, large = 1500, 6000
    ratio = _naming_seconds(large) / max(_naming_seconds(small), 1e-6)
    # 4x the copies: linear is ~4x, a probe from .001 every call is ~16x.
    assert ratio < 9.0, f"naming {large} copies cost {ratio:.1f}x naming {small}"


def test_a_used_names_set_gives_the_same_names_as_a_plain_one() -> None:
    plain: set[str] = {"Rock", "Rock.003", "Rock.007", "Tree.002"}
    fast = mesh_ops.UsedNames(plain)
    for source in ["Rock"] * 12 + ["Rock.003", "Tree.002", "Tree", "Rock"] * 3:
        expected = mesh_ops.next_name(source, plain)
        assert mesh_ops.next_name(source, fast) == expected, source
        plain.add(expected)
        fast.add(expected)


def test_a_used_names_hint_survives_a_caller_that_never_adds_the_name() -> None:
    fast = mesh_ops.UsedNames({"Box"})
    assert mesh_ops.next_name("Box", fast) == "Box.001"
    assert mesh_ops.next_name("Box", fast) == "Box.001"  # nothing was added, nothing skipped
