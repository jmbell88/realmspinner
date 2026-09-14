"""Plotter's Properties pane: whose fields it is showing, and the drafts it keeps.

The pane moved to the far side of the window from the layer list in wave A and
became a name/value table in wave C. Both changes turn on things that can be
asserted without a frame: what the subject line says, and where a half-typed
property name lives while it is being typed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.studio.panes import plotter_layers
from warlock.studio.plotter.props import Prop
from warlock.studio.plotter.tilemap import MapDoc


def _map():
    doc = MapDoc(8, 6, 16, 16)
    layer = doc.add_tile_layer()
    doc.set_active_layer(layer.uid)
    return doc, layer


def _state(**kwargs):
    base = {"selected_object": None, "selected_objects": set()}
    base.update(kwargs)
    state = SimpleNamespace(**base)
    state.select_objects = lambda values: setattr(state, "selected_objects", set(values))
    state.select_object = lambda uid: None
    return state


def _ctx():
    return SimpleNamespace(state=SimpleNamespace(preview={}))


# --- whose fields are these --------------------------------------------------


def test_the_subject_names_the_layer_when_a_layer_is_showing():
    """Until wave A this pane sat directly under the list it was about, where
    "the thing above" was answer enough. On the other side of the map it is
    not: a user reading a Name field has to be told whose name it is."""
    doc, layer = _map()
    doc.set_layer_props(layer.uid, name="Ground")

    assert plotter_layers._subject(doc, _state(), layer) == "Layer: Ground"


def test_an_unnamed_layer_still_reads_as_a_layer():
    doc, layer = _map()
    doc.set_layer_props(layer.uid, name="")
    assert plotter_layers._subject(doc, _state(), layer) == "Layer: Untitled"


def test_the_subject_names_the_object_and_its_id():
    """Two objects may share a name and the map addresses them by number, so
    the id is what makes the line unambiguous."""
    doc, _layer = _map()
    objects = doc.add_object_layer()
    obj = plotter_layers.add_object(doc, objects, "rect", 0.0, 0.0, 16.0, 16.0)
    doc.set_object(objects.uid, obj.uid, name="door_1")
    state = _state(selected_object=obj.uid, selected_objects={obj.uid})

    line = plotter_layers._subject(doc, state, objects)

    assert line == f"Object: door_1 (#{obj.uid})"


def test_an_unnamed_object_is_still_identified_by_its_id():
    """``add_object`` gives every object a default name, so this is the case a
    user reaches by clearing the Name field rather than one the door produces.
    The id is what keeps the line pointing at something."""
    doc, _layer = _map()
    objects = doc.add_object_layer()
    obj = plotter_layers.add_object(doc, objects, "point", 4.0, 4.0, 0.0, 0.0)
    doc.set_object(objects.uid, obj.uid, name="")
    state = _state(selected_object=obj.uid, selected_objects={obj.uid})

    assert plotter_layers._subject(doc, state, objects) == f"Object: Object (#{obj.uid})"


def test_a_multi_selection_reads_as_a_count():
    """The pane is a summary rather than a bulk editor there, and the line says
    so before the form does."""
    doc, layer = _map()
    state = _state(selected_objects={1, 2, 3})

    assert plotter_layers._subject(doc, state, layer) == "3 objects"


# --- the drafts the table keeps ----------------------------------------------


def test_the_new_key_draft_is_scoped_to_one_editor():
    """A layer's editor and the map's are drawn in one frame, and a name typed
    into one must not appear in the other."""
    ctx = _ctx()
    first = plotter_layers._prop_form(ctx, "plotter_layer_prop:1")
    second = plotter_layers._prop_form(ctx, "plotter_map_prop:1")

    first["name"] = "theme"

    assert second["name"] == ""
    assert plotter_layers._prop_form(ctx, "plotter_layer_prop:1")["name"] == "theme"


def test_the_draft_survives_being_reopened():
    ctx = _ctx()
    plotter_layers._prop_form(ctx, "k")["name"] = "half typed"
    assert plotter_layers._prop_form(ctx, "k")["name"] == "half typed"


def test_an_older_two_key_draft_is_filled_in_rather_than_replaced():
    """The dict parked here before wave C carried only a name and a type. A
    session upgrading mid-edit must not lose the name it was holding."""
    ctx = _ctx()
    ctx.state.preview["k"] = {"name": "cost", "type": "int"}

    form = plotter_layers._prop_form(ctx, "k")

    assert form["name"] == "cost"
    assert form["selected"] == ""
    assert form["folded"] == set()


def test_a_new_draft_starts_with_every_key_the_editor_reads():
    form = plotter_layers._prop_form(_ctx(), "k")
    assert set(form) == {"name", "type", "selected", "folded"}


# --- the tables the forms are built from -------------------------------------


def test_the_text_flags_name_real_fields_of_the_text_shape():
    """Six toggles that were six rows are one row of six checkboxes. A flag
    named here that the shape does not carry would be a box that writes an
    attribute nothing reads."""
    from warlock.studio.plotter.tilemap import Text

    shape = Text(text="hello")
    for key, label in plotter_layers.TEXT_FLAGS:
        assert hasattr(shape, key), key
        assert isinstance(getattr(shape, key), bool), key
        assert label


def test_every_addable_property_type_has_a_blank_value():
    for kind in plotter_layers.AUTHORABLE_TYPES:
        blank = plotter_layers._blank_value(kind)
        assert blank is not None or kind == "string"


# --- one gesture, one undo step (the 2026-09-07 audit, plotter-02) ----------


def test_layer_properties_name_class_offset_parallax_typing_is_one_undo_step():
    """``doc.set_layer_props`` pushes an unconditional ``history.push``, and
    Name, Class, Offset and Parallax each called it straight from a field with
    no ``fold_undo`` between the two -- one undo step per keystroke instead of
    one per gesture. Opacity and Tint in the same function already folded, and
    the comment beside them says why; these four had no such comment because
    they had no such fold.

    Each check is bounded to the gap between one field and the next one the
    function draws, rather than "a fold exists somewhere before the write" --
    which a neighbouring field's own fold would satisfy for free and prove
    nothing about the field actually being checked.
    """
    import inspect

    from warlock.studio.panes import plotter_layers

    table = inspect.getsource(plotter_layers._layer_table)

    after_name = table.split('"##layer-name"', 1)[1]
    before_class = after_name.split('"##layer-class"', 1)[0]
    assert "controls.fold_undo(" in before_class, (
        "Name field is not folded before the Class field is drawn"
    )

    after_class = table.split('"##layer-class"', 1)[1]
    before_opacity = after_class.split('"##layer-opacity"', 1)[0]
    assert "controls.fold_undo(" in before_opacity, (
        "Class field is not folded before the Opacity field is drawn"
    )

    after_offset = table.split('"##layer-offset"', 1)[1]
    before_parallax = after_offset.split('"##layer-parallax"', 1)[0]
    assert "controls.fold_undo(" in before_parallax, (
        "Offset field is not folded before the Parallax field is drawn"
    )

    after_parallax = table.split('"##layer-parallax"', 1)[1]
    before_write = after_parallax.split("doc.set_layer_props(", 1)[0]
    assert "controls.fold_undo(" in before_write, (
        "Parallax field writes before it is folded"
    )


def test_the_object_properties_form_folds_a_position_drag_into_one_step():
    """The 2026-09-08 audit, plotter-03: ``doc.set_object`` pushes an
    unconditional ``history.push``, and Name, Class, Position, Size and
    Rotation each called it straight from a field with no ``fold_undo``
    between the two -- one undo step per keystroke or per drag-report instead
    of one per gesture. This function's own Opacity row already folded, and
    the comment beside it says why; these five had no such comment because
    they had no such fold.

    Each check is bounded to the gap between one field and the next one the
    function draws, rather than "a fold exists somewhere before the write" --
    which a neighbouring field's own fold would satisfy for free and prove
    nothing about the field actually being checked.
    """
    import inspect

    from warlock.studio.panes import plotter_layers

    table = inspect.getsource(plotter_layers._object_fields)

    after_name = table.split('"##obj-name"', 1)[1]
    before_class = after_name.split('"##obj-class"', 1)[0]
    assert "controls.fold_undo(" in before_class, (
        "Name field is not folded before the Class field is drawn"
    )

    after_class = table.split('"##obj-class"', 1)[1]
    before_position = after_class.split('"##obj-position"', 1)[0]
    assert "controls.fold_undo(" in before_position, (
        "Class field is not folded before the Position field is drawn"
    )

    after_position = table.split('"##obj-position"', 1)[1]
    before_size = after_position.split('"##obj-size"', 1)[0]
    assert "controls.fold_undo(" in before_size, (
        "Position field is not folded before the Size field is drawn"
    )

    after_size = table.split('"##obj-size"', 1)[1]
    before_rotation = after_size.split('"##obj-rotation"', 1)[0]
    assert "controls.fold_undo(" in before_rotation, (
        "Size field is not folded before the Rotation field is drawn"
    )

    after_rotation = table.split('"##obj-rotation"', 1)[1]
    before_visible = after_rotation.split('"##obj-visible"', 1)[0]
    assert "controls.fold_undo(" in before_visible, (
        "Rotation field writes before it is folded"
    )


def test_a_custom_property_value_is_typed_as_one_undo_step():
    """The 2026-09-08 audit, plotter-04: ``property_editor``/``_value_editor``
    is the one shared custom-property editor the Map, Layer and Object
    Properties panes all reach for, and every value it draws is written back
    through an unconditional ``history.push`` (``doc.set_object``,
    ``set_layer_props``, ``set_map_properties``). None of its typed fields
    passed ``commit=True`` -- ``controls.py``'s own ``_field_call`` docstring
    names this exact failure ("typing '120' into a frame's duration pushed
    one step per character") as the reason the flag exists -- so typing into
    an int, float or string custom property spammed the undo stack one step
    per keystroke.

    Bounded to the gap between one field's own type check and the next,
    the same reason ``test_the_object_properties_form_folds_a_position_drag_
    into_one_step`` gives: an unrelated neighbour's fix must not satisfy a
    check that proves nothing about the field actually named.
    """
    import inspect

    from warlock.studio.panes import plotter_layers

    source = inspect.getsource(plotter_layers._value_editor)

    after_object_plain = source.split('if prop.type == "object":', 1)[1]
    before_int = after_object_plain.split('if prop.type == "int":', 1)[0]
    assert "commit=True" in before_int, "the plain object-id field is not committed"

    after_int = source.split('if prop.type == "int":', 1)[1]
    before_float = after_int.split('if prop.type == "float":', 1)[0]
    assert "commit=True" in before_float, "the int field is not committed"

    after_float = source.split('if prop.type == "float":', 1)[1]
    before_container = after_float.split("if prop.type in CONTAINER_TYPES:", 1)[0]
    assert "commit=True" in before_container, "the float field is not committed"

    after_container = after_float.split("if prop.type in CONTAINER_TYPES:", 1)[1]
    assert "commit=True" in after_container, "the string fallback field is not committed"

    children_source = inspect.getsource(plotter_layers._prop_children)
    after_class_field = children_source.split('"##property-class"', 1)[1]
    call_end = after_class_field.index(")")
    assert "commit=True" in after_class_field[:call_end], (
        "the class member's name field is not committed"
    )


def test_editing_a_tile_objects_gid_field_folds_into_one_undo_step():
    """The 2026-09-14 audit, plotter-01: ``_shape_fields``'s tile-gid field
    called ``controls.input_int`` with neither ``commit=True`` nor
    ``controls.fold_undo`` between it and ``doc.set_object``'s unconditional
    ``history.push`` -- typing a tile id pushed one undo step per keystroke.
    Its sibling custom-property fields in ``_value_editor`` already pass
    ``commit=True`` for the same reason (see
    ``test_a_custom_property_value_is_typed_as_one_undo_step`` above).
    """
    import inspect

    from warlock.studio.panes import plotter_layers

    source = inspect.getsource(plotter_layers._shape_fields)
    after_gid_field = source.split('"##obj-gid"', 1)[1]
    call_end = after_gid_field.index(")")
    assert "commit=True" in after_gid_field[:call_end], (
        "the tile object's gid field is not committed"
    )


@pytest.mark.parametrize("kind", ["class", "list"])
def test_a_container_property_summarises_rather_than_showing_nothing(kind):
    """A class arriving from Tiled used to look like an empty string until the
    recursive editor landed; the count is what says it carries something."""
    prop = Prop(type=kind, value={} if kind == "class" else [])
    assert plotter_layers._summary(prop)
