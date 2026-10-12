"""Regression tests for the 2026-10-12 Create leftovers (F18-3, F18-4, F19-3...).

Each test name is the claim, and each failed against the code it was written for.
"""

from __future__ import annotations

from realmspinner import generation


def test_null_sub_document_fields_fall_back_to_the_default_not_a_slot_descriptor():
    # The settings dataclasses are slots=True, so ``ModelSettings.output_profile``
    # on the class is a member descriptor, and ``_coerce_str(None, <descriptor>)``
    # stored ``str(<member ...>)`` as the field's value.
    request = generation.GenerationRequest.from_dict(
        {
            "model": {"output_profile": None, "mesh_finishing": None},
            "sprite": {"mode": None, "action": None, "palette": None},
            "tile": {"mode": None, "view": None, "palette": None},
        }
    )
    assert request.model.output_profile == generation.ModelSettings().output_profile
    assert request.model.mesh_finishing == generation.ModelSettings().mesh_finishing
    assert request.sprite.mode == generation.SpriteSettings().mode
    assert request.sprite.action == generation.SpriteSettings().action
    assert request.sprite.palette == generation.SpriteSettings().palette
    assert request.tile.mode == generation.TileSettings().mode
    assert request.tile.view == generation.TileSettings().view
    assert request.tile.palette == generation.TileSettings().palette


def test_every_coercion_default_is_the_default_instance_value():
    for table, cls in (
        (generation._TILE_COERCIONS, generation.TileSettings),
        (generation._SPRITE_COERCIONS, generation.SpriteSettings),
        (generation._MODEL_COERCIONS, generation.ModelSettings),
    ):
        fresh = cls()
        for name, (_kind, default) in table.items():
            assert default == getattr(fresh, name), (cls.__name__, name)


# --- F19-3: Create's panes name each other only through public names -----------


def test_create_ui_modules_reach_each_other_only_through_public_names():
    """``brief`` called ``settings_3d._reset``/``settings_2d._reset``,
    ``settings_2d`` called ``create_brief._with_pending_candidates_problem``,
    ``settings_character`` called ``settings_2d._seed_row`` and ``workspace`` read
    ``brief._TYPE_HINTS``: five private reaches between modules of one mode, each
    a rename away from breaking a caller no import check sees."""
    import ast
    import pathlib

    from realmspinner.studio.modes.create import ui

    root = pathlib.Path(ui.__file__).parent
    siblings = {p.stem for p in root.rglob("*.py")} | {"create_brief"}
    found: list[str] = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in siblings
                and node.attr.startswith("_")
                and not node.attr.startswith("__")
            ):
                found.append(f"{path.name}:{node.lineno} {node.value.id}.{node.attr}")
    assert found == []


# --- F20-5: create-2d's form clears a ring when its control is edited ---------


def test_create_2d_form_clears_a_field_ring_when_the_control_is_edited():
    """``create-3d`` got ``on_edit`` in the 2026-10-10 audit (create-18);
    ``create-2d`` still cleared only at the call sites that remembered to, so a
    control drawn through ``form_ui`` without one kept its ring after the edit."""
    import pathlib
    import re

    from realmspinner.studio.modes.create.ui.panes import settings_2d

    source = pathlib.Path(settings_2d.__file__).read_text(encoding="utf-8")
    opened = re.search(r'forms\.Form\(\s*"create-2d"(.*?)\)\s*as ', source, re.S)
    assert opened is not None
    assert "on_edit=ctx.state.clear_field_error" in opened.group(1)


# --- F20-11: a removed style the Create form still holds ----------------------


def test_removing_the_style_lora_the_create_form_holds_clears_it(monkeypatch):
    """The picker listed the removed style and let the user clear it, but the
    landing left ``form_2d["style_lora"]`` naming a key no registry holds, so the
    next Generate was refused for a style the user had just deleted."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent / "studio"))
    try:
        from test_audit_2026_10_04_lora_refresh import _App, _Ctx, _Done, _register
    finally:
        sys.path.pop(0)
    from realmspinner import models

    table = dict(models.STYLE_LORAS)
    monkeypatch.setattr(models, "STYLE_LORAS", table)
    _register(table)
    ctx = _Ctx()
    ctx.state.form_2d["style_lora"] = "imported_abc"
    ctx.state.form_2d["lora_weight"] = 1.3
    ctx.state.field_errors["style_lora"] = "That style is gone."
    with models.STYLE_LORAS_LOCK:
        table.pop("imported_abc")
    _App(ctx)._on_task_done(_Done("lora:remove:imported_abc", {"ok": True}, tag="Inkwash"))
    assert ctx.state.form_2d["style_lora"] == ""
    assert ctx.state.form_2d["lora_weight"] == models.DEFAULT_LORA_WEIGHT
    assert "style_lora" not in ctx.state.field_errors


def test_removing_a_different_style_leaves_the_forms_choice_alone(monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent / "studio"))
    try:
        from test_audit_2026_10_04_lora_refresh import _App, _Ctx, _Done, _register
    finally:
        sys.path.pop(0)
    from realmspinner import models

    table = dict(models.STYLE_LORAS)
    monkeypatch.setattr(models, "STYLE_LORAS", table)
    _register(table, "imported_a", "A")
    _register(table, "imported_b", "B")
    ctx = _Ctx()
    ctx.state.form_2d["style_lora"] = "imported_a"
    with models.STYLE_LORAS_LOCK:
        table.pop("imported_b")
    _App(ctx)._on_task_done(_Done("lora:remove:imported_b", {"ok": True}, tag="B"))
    assert ctx.state.form_2d["style_lora"] == "imported_a"


def test_a_host_with_neither_binary_does_not_refuse_a_stored_custom_budget(monkeypatch):
    """``host_budget_form`` sends Raw on a host with neither Blender nor
    gltfpack, but ``problems`` judged the stored form, so a stored Custom with a
    bad count blocked Make 3D with a message about a budget that was not going
    to be sent."""
    from types import SimpleNamespace

    from realmspinner.studio.modes.create.ui.panes import settings_3d

    monkeypatch.setattr(settings_3d.remesh_panel, "blender_available", lambda ctx: False)
    monkeypatch.setattr(settings_3d.retarget_panel, "gltfpack_available", lambda ctx: False)
    form = {"profile": "custom", "custom_triangles": 0, "lowpoly_triangles": 0}
    ctx = SimpleNamespace(model_rows=None, state=SimpleNamespace(form_3d=form))
    found = settings_3d.problems(ctx, None, form)
    assert not [p for p in found if "Custom triangles" in str(p)]


# --- F18-11: the four preserve_shape defaults agree ----------------------------


def test_every_default_mesh_finishing_is_the_same_word():
    """Create's engine cannot import ``generation`` (its import pin), so the four
    restatements stay literals and this is what keeps them one value."""
    import inspect

    from realmspinner.service import _jobs_create
    from realmspinner.studio.modes.create.engine import mesh as create_mesh
    from realmspinner.studio.state import DEFAULT_FORM_3D

    door = inspect.signature(_jobs_create.create_job).parameters["mesh_finishing"].default
    words = {
        generation.ModelSettings().mesh_finishing,
        create_mesh.DEFAULT_MESH_FINISHING,
        DEFAULT_FORM_3D["mesh_finishing"],
        door,
    }
    assert words == {"preserve_shape"}
