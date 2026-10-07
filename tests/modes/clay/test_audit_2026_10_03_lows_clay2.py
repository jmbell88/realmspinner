"""Regressions for the 2026-10-03 audit's Low findings, fixer ``clay2``.

clay-80: ``OverflowError`` joined the caught tuple at the sites clay-agent-tools-09
listed and nowhere else, so every other bare ``int()`` on ``inf`` and ``float()`` on
a JSON integer past 1e308 still reached ``call()``'s "failed unexpectedly" backstop.
clay-81: a non-image upload or an unhashable reference name did the same.
clay-99: ``clay_render``'s view count and the length of a name had no bound at
the tool door.
clay-85: ``kept_objects`` was cubic in a parent chain's depth.
clay-100 / clay-107 / clay-108 / clay-123: stale docstrings and comments.
clay-103: the three-vertex angle readout did not say which vertex is the apex.
clay-118: a plain click on the UV canvas never selected an island.
clay-126 / clay-127: evidence gaps (pane deciders, outward-import pins).
"""

from __future__ import annotations

import ast
import base64
import copy
import inspect
import io
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import ops_subdiv
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import viewport_hints as clay_hints
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import schema as agent_schema
from realmspinner.studio.modes.clay.ui.panes import props as clay_props
from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools
from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv

from .test_agent_clay import _Ctx, _new_agent_tab

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src" / "realmspinner"

_INF = float("inf")
_BIG = 10**400  # a JSON integer: ``int`` is fine with it, ``float()`` is not


def _backstop(result: dict) -> bool:
    return "failed unexpectedly" in result["content"][0]["text"]


# --- clay-80 ------------------------------------------------------------------

# (tool, arguments with "X" where the bad number goes, the field it must name).
# "U" stands for a real uid. Covers the sites that were still escaping.
_NUMERIC_DOORS = [
    ("clay_render", {"size": "X"}, "size"),
    ("clay_undo", {"steps": "X"}, "steps"),
    ("clay_redo", {"steps": "X"}, "steps"),
    ("clay_parent", {"uid": "U", "parent": "X"}, "parent"),
    ("clay_select_elements", {"uid": "U", "mode": "face", "faces": [0], "expect_stamp": "X"},
     "expect_stamp"),
    ("clay_add_primitive", {"generator": "box", "translation": ["X", 0, 0]}, "translation"),
    ("clay_add_primitive", {"generator": "box", "rotation": ["X", 0, 0]}, "rotation"),
    ("clay_add_primitive", {"generator": "box", "scale": ["X", 1, 1]}, "scale"),
    ("clay_transform", {"uid": "U", "translation": ["X", 0, 0]}, "translation"),
    ("clay_transform", {"uid": "U", "rotation": ["X", 0, 0]}, "rotation"),
    ("clay_transform", {"uid": "U", "scale": ["X", 1, 1]}, "scale"),
    ("clay_material", {"uids": ["U"], "color": ["X", 0, 0]}, "color"),
    ("clay_material", {"uids": ["U"], "color": [0.5, 0.5, 0.5], "metallic": "X"}, "metallic"),
    ("clay_material", {"uids": ["U"], "color": [0.5, 0.5, 0.5], "roughness": "X"}, "roughness"),
    ("clay_uv", {"uid": "U", "action": "pack", "margin": "X"}, "margin"),
    ("clay_measure", {"kind": "distance", "a": ["X", 0, 0], "b": [0, 0, 0]}, "a"),
]


@pytest.mark.parametrize("bad", [_INF, _BIG], ids=["inf", "past-1e308"])
@pytest.mark.parametrize(
    "tool, template, field", _NUMERIC_DOORS, ids=[f"{t}-{f}" for t, _a, f in _NUMERIC_DOORS]
)
def test_every_numeric_door_refuses_an_infinite_or_oversized_number_naming_its_field(
    tool: str, template: dict, field: str, bad: object
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")

    def fill(value: object) -> object:
        if value == "X":
            return bad
        if value == "U":
            return uid
        if isinstance(value, list):
            return [fill(v) for v in value]
        if isinstance(value, dict):
            return {k: fill(v) for k, v in value.items()}
        return value

    result = agent_clay.call(ctx, session, tool, fill(copy.deepcopy(template)))

    assert not _backstop(result), f"{tool} reached the generic backstop: {result}"
    assert result["isError"] is True, result
    assert result["structuredContent"].get("field") == field, result


# --- clay-81 ------------------------------------------------------------------


def _png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_reference_doors_refuse_a_non_image_and_a_non_string_name_naming_their_field() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")

    # Valid base64 that is not an image, and a PNG cut off after its header.
    garbage = base64.b64encode(b"this is not an image at all").decode("ascii")
    truncated = _png_b64()[:40]
    for payload in (garbage, truncated):
        result = agent_clay.call(
            ctx, session, "clay_reference_add", {"name": "r", "png_base64": payload}
        )
        assert not _backstop(result), result
        assert result["isError"] is True, result
        assert result["structuredContent"].get("field") == "png_base64", result

    ok = agent_clay.call(
        ctx, session, "clay_reference_add", {"name": "good", "png_base64": _png_b64()}
    )
    assert ok["isError"] is False, ok

    for bad_name in (["good"], {"a": 1}):
        for tool in ("clay_reference_get", "clay_reference_remove"):
            result = agent_clay.call(ctx, session, tool, {"name": bad_name})
            assert not _backstop(result), (tool, result)
            assert result["isError"] is True, result
            assert result["structuredContent"].get("field") == "name", (tool, result)
        compare = agent_clay.call(ctx, session, "clay_render", {"compare": bad_name})
        assert not _backstop(compare), compare
        assert compare["isError"] is True, compare
        assert compare["structuredContent"].get("field") == "compare", compare


# --- clay-99 ------------------------------------------------------------------


def test_clay_render_refuses_more_views_than_the_documented_maximum() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    too_many = ["front"] * (agent_schema.MAX_RENDER_VIEWS + 1)

    result = agent_clay.call(ctx, session, "clay_render", {"size": 64, "views": too_many})

    assert result["isError"] is True, result
    assert result["structuredContent"].get("field") == "views", result
    assert str(agent_schema.MAX_RENDER_VIEWS) in result["content"][0]["text"], result


def test_names_are_refused_past_their_length_cap() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    long_name = "n" * (agent_schema.MAX_NAME_LENGTH + 1)

    for tool, args in (
        ("clay_rename", {"uid": uid, "name": long_name}),
        ("clay_checkpoint", {"name": long_name}),
        ("clay_reference_add", {"name": long_name, "png_base64": _png_b64()}),
    ):
        result = agent_clay.call(ctx, session, tool, args)
        assert result["isError"] is True, (tool, result)
        assert result["structuredContent"].get("field") == "name", (tool, result)

    ok = agent_clay.call(ctx, session, "clay_rename", {"uid": uid, "name": "n" * agent_schema.MAX_NAME_LENGTH})  # noqa: E501
    assert ok["isError"] is False, ok


def test_checkpoints_are_capped_but_an_existing_name_can_still_be_reset() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    for i in range(agent_schema.MAX_CHECKPOINTS):
        assert agent_clay.call(ctx, session, "clay_checkpoint", {"name": f"c{i}"})["isError"] is False  # noqa: E501

    over = agent_clay.call(ctx, session, "clay_checkpoint", {"name": "one-too-many"})
    assert over["isError"] is True, over
    assert over["structuredContent"].get("field") == "name", over

    again = agent_clay.call(ctx, session, "clay_checkpoint", {"name": "c0"})
    assert again["isError"] is False, again


# --- clay-85 ------------------------------------------------------------------


def _chain_doc(depth: int, visible_leaf_only: bool = False) -> bd.ClayDoc:
    doc = bd.ClayDoc()
    mesh = bp.box()
    parent = None
    for i in range(depth):
        obj = bd.Obj(uid=i + 1, name=f"o{i}", mesh=mesh, parent=parent)
        obj.visible = (i == depth - 1) if visible_leaf_only else True
        doc.objects.append(obj)
        parent = obj.uid
    return doc


def test_kept_objects_on_a_thousand_deep_chain_runs_in_under_a_second() -> None:
    doc = _chain_doc(1000)
    start = time.perf_counter()
    kept = bd.kept_objects(doc)
    elapsed = time.perf_counter() - start
    assert len(kept) == 1000
    assert elapsed < 1.0, f"kept_objects took {elapsed:.2f}s on a 1,000-deep chain (was 4.9 s)"


def test_kept_objects_still_keeps_a_hidden_chain_above_one_visible_leaf() -> None:
    doc = _chain_doc(50, visible_leaf_only=True)
    assert [o.uid for o in bd.kept_objects(doc)] == list(range(1, 51))


def test_kept_objects_drops_a_hidden_object_with_no_visible_descendant_and_survives_a_cycle() -> None:  # noqa: E501
    doc = bd.ClayDoc()
    mesh = bp.box()
    root = bd.Obj(uid=1, name="root", mesh=mesh)
    hidden = bd.Obj(uid=2, name="hidden", mesh=mesh, parent=1)
    hidden.visible = False
    dangling = bd.Obj(uid=3, name="dangling", mesh=mesh, parent=99)
    cyc_a = bd.Obj(uid=4, name="a", mesh=mesh, parent=5)
    cyc_b = bd.Obj(uid=5, name="b", mesh=mesh, parent=4)
    cyc_a.visible = False
    for o in (root, hidden, dangling, cyc_a, cyc_b):
        doc.objects.append(o)
    assert [o.uid for o in bd.kept_objects(doc)] == [1, 3, 4, 5]


# --- clay-100 -----------------------------------------------------------------


def test_the_frame_budget_docstring_names_every_caller() -> None:
    from realmspinner.studio.modes.clay.agent import validate

    doc = inspect.getdoc(validate._over_frame_budget) or ""
    assert "Only two callers" not in doc

    callers: set[str] = set()
    agent_dir = SRC / "studio" / "modes" / "clay" / "agent"
    for path in sorted(agent_dir.glob("*.py")):
        if path.name == "validate.py":
            continue
        tree = ast.parse(path.read_text("utf-8"))
        for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
            if fn.name == "_h_render":  # its own nested, differently-shaped helper
                continue
            if any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == "_over_frame_budget"
                for c in ast.walk(fn)
            ):
                callers.add(fn.name)
    assert callers, "the scan found no callers at all"
    # ``_h_batch`` is the clay_batch door, and so on: either spelling counts.
    missing = sorted(
        c for c in callers if c not in doc and c.replace("_h_", "clay_") not in doc
    )
    assert not missing, f"_over_frame_budget's docstring does not name {missing}"


# --- clay-103 / clay-123 --------------------------------------------------------


def test_the_three_vertex_angle_readout_names_which_vertex_is_the_apex() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    doc.set_element_mode("vertex")
    doc.set_element_sel(obj.uid, el.ElementSel(verts=[0, 1, 2]))
    assert clay_hints.measure_line(doc) == "angle  90.00° at vertex 1"

    doc.set_element_sel(obj.uid, el.ElementSel(verts=[0, 1, 2]))
    other = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    doc.set_element_sel(obj.uid, el.ElementSel(verts=[0]))
    doc.set_element_sel(other.uid, el.ElementSel(verts=[1, 2]))
    assert clay_hints.measure_line(doc).endswith("at B vertex 1"), clay_hints.measure_line(doc)


def test_measure_line_docstring_matches_the_drag_touching_rev() -> None:
    drag = (SRC / "studio" / "modes" / "clay" / "ui" / "_view_drag.py").read_text("utf-8")
    assert drag.count("doc.touch()") >= 2, "a drag frame must still touch the document"
    doc = inspect.getdoc(clay_hints.measure_line) or ""
    assert "without ``touch()`` until release" not in doc
    assert "never asked during" in doc or "never asked" in doc


# --- clay-107 -----------------------------------------------------------------


def test_catmull_clark_docstring_matches_its_refusals() -> None:
    doc = inspect.getdoc(ops_subdiv.catmull_clark) or ""
    assert "Never refuses" not in doc
    assert "MAX_SUBDIVIDED_FACES" in doc


# --- clay-108 -----------------------------------------------------------------


def test_no_module_cites_the_retired_clay_plan_file() -> None:
    needle = "CLAY-PLAN" ".md"
    offenders = [
        f"{path.relative_to(REPO)}:{n}"
        for path in sorted(SRC.rglob("*.py"))
        for n, line in enumerate(path.read_text("utf-8").splitlines(), 1)
        if needle in line
    ]
    assert offenders == []

    ledger = (REPO / "tests" / "test_ux_todo_fixes.py").read_text("utf-8")
    assert '"CLAY-PLAN" ".md"' in ledger, "RETIRED_PLANS does not name the Clay plan file"


# --- clay-118 -----------------------------------------------------------------


def _two_island_mesh():
    from dataclasses import replace

    from realmspinner.kernels.mesh import mesh as bm
    from realmspinner.kernels.mesh import uvtools

    mesh = bm.from_faces(
        np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0]],  # noqa: E501
            dtype="f4",
        ),
        [[0, 1, 2, 3], [4, 5, 6, 7]],
    )
    uv = np.array(
        [[0, 0], [0.4, 0], [0.4, 0.4], [0, 0.4], [0.6, 0], [1, 0], [1, 0.4], [0.6, 0.4]],
        dtype="f4",
    )
    mesh = replace(mesh, uv=uv)
    return mesh, uvtools.islands(mesh)


def test_a_click_without_a_drag_on_the_uv_canvas_selects_the_island_under_it() -> None:
    mesh, ids = _two_island_mesh()
    state = clay_uv.UvPaneState()
    state.drag_mode = "box"
    state.drag_start = (0.2, 0.2)  # the middle of island 0, nowhere near a corner
    state.dragged = False

    assert clay_uv.finish_click(state, mesh, ids) is True
    assert state.selected_islands == frozenset({int(ids[0])})

    # A click on empty space deselects, like the first frame of a marquee.
    state.drag_mode = "box"
    state.drag_start = (0.5, 0.9)
    assert clay_uv.finish_click(state, mesh, ids) is True
    assert state.selected_islands == frozenset()


def test_a_release_after_a_real_drag_leaves_the_marquee_selection_alone() -> None:
    mesh, ids = _two_island_mesh()
    state = clay_uv.UvPaneState()
    state.selected_islands = frozenset({0, 1})
    state.drag_mode = "box"
    state.drag_start = (0.2, 0.2)
    state.dragged = True

    assert clay_uv.finish_click(state, mesh, ids) is False
    assert state.selected_islands == frozenset({0, 1})


def test_the_uv_canvas_applies_a_click_on_release() -> None:
    src = inspect.getsource(clay_uv._canvas)
    release = src.index("is_item_deactivated")
    assert "finish_click(" in src[release:], "_canvas never applies a no-drag click"
    assert "view_state.dragged = True" in src, "_canvas never records that a press became a drag"


# --- clay-126: pane deciders, table-driven -------------------------------------


def test_pane_deciders_face_fill_format_default_and_element_summary_are_pinned() -> None:
    # uv._face_fill: an overlapping face is tinted, every other face draws plain.
    overlap = np.array([True, False])

    over = clay_uv._face_fill(0, overlap)
    assert over is not None and over[3] == pytest.approx(0.55), "overlap must tint the face"
    assert clay_uv._face_fill(1, overlap) is None, "a face with no overlap draws plain"
    assert clay_uv._face_fill(0, None) is None

    # tools._format_default
    for value, text in (
        (True, "on"),
        (False, "off"),
        (0.5, "0.50"),
        (3, "3"),
        ("cap", "cap"),
        ((1.0, 2.5), "1.00, 2.50"),
        ([True, 0.25], "on, 0.25"),
    ):
        assert clay_tools._format_default(value) == text, value

    # props._element_summary's sentence (``element_summary_text`` is its pure half).
    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    assert clay_props.element_summary_text(doc) is None
    doc.set_element_mode("face")
    assert clay_props.element_summary_text(doc) == "face mode -- nothing selected"
    doc.set_element_sel(a.uid, el.ElementSel(faces=[0, 1]))
    assert clay_props.element_summary_text(doc) == "face mode -- 2 faces across 1 object"


# --- clay-127: outward-import pins -----------------------------------------------

_UI = SRC / "studio" / "modes" / "clay" / "ui"

#: Module-scope imports, as ``(relative-dots+module, name)``, each file may make.
#: A new one is a deliberate edit here, the way ``tests/studio/test_view_frame.py``
#: pins ``_view_frame``.
_PINNED_MODULE_IMPORTS = {
    SRC / "studio" / "viewport_hints.py": {
        ("__future__", "annotations"), ("math", None), ("weakref", None),
        ("dataclasses", "dataclass"), ("typing", "Any"), ("numpy", None),
    },
    _UI / "_view_bounds.py": {
        ("__future__", "annotations"), ("typing", "Any"), ("numpy", None),
    },
    _UI / "_view_cache.py": {
        ("__future__", "annotations"), ("typing", "Any"),
        (".....kernels.geom3d", "gltf"), (".....kernels.mesh", "document"),
        ("....viewer", "scene"),
    },
    _UI / "_view_drag.py": {
        ("__future__", "annotations"), ("dataclasses", "dataclass"), ("typing", "Any"),
        ("numpy", None), (".....kernels.geom3d", "math3d"), (".....kernels.mesh", "document"),
    },
    _UI / "_view_overlay.py": {
        ("__future__", "annotations"), ("typing", "Any"), ("moderngl", None), ("numpy", None),
        (".....kernels.geom3d", "math3d"), (".....kernels.mesh.topo", "corner_spans"),
        (".....kernels.mesh.topo", "flat_next"), ("....viewer.render", "DrawItem"),
    },
    _UI / "_view_pick.py": {
        ("__future__", "annotations"), ("dataclasses", "dataclass"), ("typing", "Any"),
        ("....viewer", "picking"),
    },
}


def _module_scope_imports(path: Path) -> set[tuple[str, str | None]]:
    """The runtime imports written at module scope (``TYPE_CHECKING`` ones excluded)."""
    tree = ast.parse(path.read_text("utf-8"))
    runtime: set[tuple[str, str | None]] = set()

    def walk(body: list, guarded: bool) -> None:
        for node in body:
            if isinstance(node, ast.Import) and not guarded:
                runtime.update((a.name, None) for a in node.names)
            elif isinstance(node, ast.ImportFrom) and not guarded:
                where = "." * (node.level or 0) + (node.module or "")
                runtime.update((where, a.name) for a in node.names)
            elif isinstance(node, ast.If):
                walk(node.body, guarded or "TYPE_CHECKING" in ast.unparse(node.test))
                walk(node.orelse, guarded)

    walk(tree.body, False)
    # ``from typing import TYPE_CHECKING`` is the guard itself, not a dependency.
    runtime.discard(("typing", "TYPE_CHECKING"))
    return runtime


@pytest.mark.parametrize("path", sorted(_PINNED_MODULE_IMPORTS), ids=lambda p: p.name)
def test_the_clay_view_mixins_and_viewport_hints_import_nothing_outward_at_module_scope(
    path: Path,
) -> None:
    runtime = _module_scope_imports(path)
    assert runtime == _PINNED_MODULE_IMPORTS[path], (
        f"{path.name}'s module-scope imports changed. These modules are held to a "
        "pinned outward set; widen it here, deliberately, if the new import is right."
    )
    # A mixin may name ClayView for annotations only: a runtime import of
    # ``.view`` is the cycle the mixin split exists to avoid.
    assert all(mod != ".view" for mod, _name in runtime), f"{path.name} imports .view at runtime"
    assert not any(
        mod.lstrip(".").split(".")[0] in {"imgui_bundle", "pygame"} for mod, _n in runtime
    ), f"{path.name} reaches a UI toolkit at module scope"
