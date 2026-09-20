"""The properties panel's generator rebuild carries per-face *material* too,
not only shading.

``tests/modes/clay/test_shading.py`` already pins the shading half of this rebuild
(``clay_props._generator`` calling what used to be ``_carry_shading`` and is
now ``regen.carry_over``); this file is the material half, which the panel
never carried at all before this change -- ``primitives._mesh`` stamps a
fresh all-zero ``material`` array on every rebuild, and nothing downstream of
it put the old paint back. A box painted with a non-default palette slot
reverted to slot 0 the moment any field in the generator section was edited.

Also pinned here: the two rebuild doors, the properties panel and the agent
tool surface, both go through ``clay.regen.carry_over`` rather than each
carrying its own (partial, and now provably different) copy of the rule --
the thing that let them silently diverge the first two times this was
written.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay.ui.panes import props as clay_props


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` and ``test_shading.py``
    for why this is not a conftest fixture."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _placed_box(doc: bd.ClayDoc, material: int = 3) -> bd.Obj:
    return doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Box",
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
            material=material,
        )
    )


def test_the_properties_panel_keeps_per_face_material_through_a_params_edit(
    monkeypatch, ui
) -> None:
    """Face 2 of a box, painted slot 1 by hand, must still read slot 1 after a
    same-face-count params edit sent through the real panel door
    (``clay_props._generator``). Fails today: the rebuilt mesh comes back with
    ``material`` all zeros, because ``_carry_shading`` -- the only carry this
    door had -- never touched the material array at all.
    """
    doc = bd.ClayDoc()
    obj = _placed_box(doc, material=3)
    painted = np.array(obj.mesh.material)
    painted[2] = 1
    doc.set_mesh(obj.uid, bm.Mesh(
        positions=obj.mesh.positions,
        loops=obj.mesh.loops,
        starts=obj.mesh.starts,
        material=painted,
        smooth=obj.mesh.smooth,
    ), keep_generator=True)

    def fake_widget(key, value, default):
        if key == "size":
            return (2.0, 1.0, 1.0), True
        return value, False

    monkeypatch.setattr(clay_props, "_widget", fake_widget)

    ui.new_frame()
    ui.begin("##host")
    clay_props._generator(doc, doc.by_uid(obj.uid))
    ui.end()
    ui.end_frame()

    rebuilt = doc.by_uid(obj.uid)
    assert tuple(rebuilt.params["size"]) == (2.0, 1.0, 1.0)
    assert int(rebuilt.mesh.material[2]) == 1, "a size edit must not grey out a painted face"


def test_both_generator_rebuild_doors_go_through_clay_regen() -> None:
    """The gate that stops the panel and the agent silently diverging a third
    time. Each door's rebuild function must call ``regen.carry_over(...)``,
    and neither may rebuild with a bare ``shading.auto_smooth(...)`` over a
    generator call keyed on an *existing* object's own ``.generator`` --
    ``bp.GENERATORS[obj.generator][1](...)`` -- which is exactly the pattern
    that used to throw material away and (in the agent's case) re-derive
    shading instead of carrying it. The insertion doors' own, unrelated
    ``auto_smooth(bp.GENERATORS[generator][1](...))`` calls (a brand new
    object being placed, not an existing one being rebuilt) are untouched by
    this rule and deliberately outside the two functions this test inspects.
    """
    import realmspinner.studio.modes.clay.agent.tools as agent_clay_tools_mod
    import realmspinner.studio.modes.clay.ui.panes.props as clay_props_mod

    # ``_h_set_params`` lives in ``studio/modes/clay/agent/tools.py`` since the P4
    # restructure split it out of ``studio/modes/clay/agent/dispatch.py`` (dev/RESTRUCTURE.md) --
    # ``agent_clay`` only imports it now to build its own ``_HANDLERS``
    # table, so parsing ``studio/modes/clay/agent/dispatch.py``'s own source would no longer find
    # a ``FunctionDef`` for it at all.
    doors = {
        agent_clay_tools_mod: "_h_set_params",
        clay_props_mod: "_generator",
    }
    for module, func_name in doors.items():
        path = Path(inspect.getfile(module))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        func = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == func_name
        )
        calls = [n for n in ast.walk(func) if isinstance(n, ast.Call)]

        carries_over = any(
            isinstance(c.func, ast.Attribute)
            and c.func.attr == "carry_over"
            and isinstance(c.func.value, ast.Name)
            and c.func.value.id == "regen"
            for c in calls
        )
        assert carries_over, f"{path.name}:{func_name} never calls regen.carry_over(...)"

        bare_rebuild = any(
            isinstance(c.func, ast.Attribute) and c.func.attr == "auto_smooth"
            for c in calls
        )
        assert not bare_rebuild, (
            f"{path.name}:{func_name} rebuilds with a bare auto_smooth(...) "
            "instead of going through regen.carry_over(...)"
        )
