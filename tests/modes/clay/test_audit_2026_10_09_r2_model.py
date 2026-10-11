"""The 2026-10-09 second Clay audit, the legacy migration, the clean ops and the
primitives: clay-85, clay-87, clay-94, clay-95, clay-97, clay-99, clay-100,
clay-101, clay-103.

Each test's name is its claim; each was run against the unfixed code first.
"""

from __future__ import annotations

import ast
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import legacy, ops_clean, ops_topo
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser


def _rewrite(data: bytes, edit) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            raw = src.read(name)
            if name == ser.SCENE:
                scene = json.loads(raw)
                edit(scene)
                raw = json.dumps(scene).encode()
            dst.writestr(name, raw)
    return out.getvalue()


# --- clay-85 --------------------------------------------------------------------

#: Measured 2026-10-09 on this machine, in microseconds per input triangle for one
#: pass of each modifier kind -- the worst figure over spheres of 1.1k to 261k
#: triangles, so the table is an upper bound a calibration may not fall under.
_MEASURED_US_PER_TRIANGLE = {
    "mirror": 3.1,
    "array": 4.5,
    "radial-array": 0.45,
    "solidify": 0.6,
    "bevel": 0.35,
    "subdivide": 2.0,
    "weld": 0.62,
    "triangulate": 0.81,
    "smooth": 0.19,
}


def test_a_stack_of_booleans_charged_to_the_bake_budget_opens_in_the_seconds_the_budget_claims(
    monkeypatch,
) -> None:
    # The audit's probe: forty tiny cutters against a ~200k-triangle target took
    # 21 s and every one was charged as if it cost a smooth iteration. Here the
    # target is half that size and the boolean itself is stubbed, so the test
    # counts what the budget lets run and prices each at the measured 0.23 s.
    target = bp.uv_sphere(0.5, 300, 170)
    tiny = bp.box()
    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=1, name="target", mesh=target))
    stacks = {}
    for i in range(40):
        doc.objects.append(bd.Obj(uid=10 + i, name=f"t{i}", mesh=tiny))
        stacks[10 + i] = (
            legacy._Modifier(id=1, kind="boolean", params=(("operation", 1), ("target", 1))),
        )
    ran: list[int] = []
    monkeypatch.setattr(
        legacy.ops_boolean, "boolean", lambda objs, op, world=None: ran.append(1) or objs[0].mesh
    )
    legacy._bake(doc, stacks)
    seconds_each = 0.23  # measured: one boolean over a 101k-triangle target
    assert len(ran) * seconds_each <= 3.0, (
        f"{len(ran)} booleans ran, about {len(ran) * seconds_each:.1f} s of work"
    )


@pytest.mark.parametrize("kind", sorted(_MEASURED_US_PER_TRIANGLE))
def test_every_modifier_kind_is_charged_what_it_costs_so_a_long_stack_opens_in_seconds(
    kind, monkeypatch
) -> None:
    mesh = bp.uv_sphere(0.5, 100, 60)
    triangles = legacy._triangle_count(mesh)
    ran: list[int] = []

    def stub(m, params, limit=None):
        ran.append(1)
        return m

    kinds = {k: (v[0], v[1], stub if k == kind else v[2]) for k, v in legacy._KINDS.items()}
    monkeypatch.setattr(legacy, "_KINDS", kinds)
    params = {p.name: p.default for p in legacy._KINDS[kind][1]}
    if kind == "array":
        params["weld"] = 0.001  # the measured case: copies that touch and weld
    if kind == "smooth":
        params["iterations"] = 1.0
    mods = tuple(
        legacy._Modifier(id=i + 1, kind=kind, params=tuple(sorted(params.items())))
        for i in range(2000)
    )
    doc = bd.ClayDoc(objects=[bd.Obj(uid=1, name="o", mesh=mesh)])
    legacy._bake(doc, {1: mods})
    seconds = len(ran) * triangles * _MEASURED_US_PER_TRIANGLE[kind] * 1e-6
    # The budget claims about a second and a half; twice that is the bar.
    assert seconds <= 3.0, f"{len(ran)} {kind} modifiers ran, about {seconds:.1f} s of work"


# --- the 2026-10-11 audit's clay-09 ---------------------------------------------

#: Measured 2026-10-11 on this machine, in microseconds for ONE modifier of each kind
#: through ``_bake`` over a 12-triangle box (kernel plus the loop's own checks, best of
#: three, the worst of a box and a 48-triangle sphere): a tiny mesh pays a fixed cost
#: the per-triangle table below 1 us a triangle cannot see.
_MEASURED_US_PER_TINY_MODIFIER = {
    "smooth": 135,
    "weld": 60,
    "triangulate": 150,
    "solidify": 270,
    "radial-array": 460,
    "subdivide": 500,
    "mirror": 1070,
    "array": 1070,
    "bevel": 1350,
    "boolean": 2400,
}


@pytest.mark.parametrize("kind", sorted(_MEASURED_US_PER_TINY_MODIFIER))
def test_a_stack_of_a_hundred_thousand_tiny_modifiers_is_refused_before_it_runs_for_seconds(
    kind, monkeypatch
) -> None:
    # The audit's probe: 2,000 weld, triangulate and smooth modifiers on a box ran
    # at 172, 135 and 119 us each, and the budget (6 to 16 passes a box) admitted
    # 166,666 to 641,025 of them, 19 to 76 s to open one hand-made file. The budget
    # is scaled to a hundredth here so the stack stays small enough to build; every
    # figure below reads one hundred times over.
    scale = 100
    monkeypatch.setattr(legacy, "MAX_BAKE_WORK", legacy.MAX_BAKE_WORK // scale)
    box = bp.box()
    ran: list[int] = []

    def stub(m, params, limit=None):
        ran.append(1)
        return m

    kinds = {k: (v[0], v[1], stub if k == kind else v[2]) for k, v in legacy._KINDS.items()}
    monkeypatch.setattr(legacy, "_KINDS", kinds)
    monkeypatch.setattr(
        legacy.ops_boolean, "boolean", lambda objs, op, world=None: ran.append(1) or objs[0].mesh
    )
    params = {p.name: p.default for p in legacy._KINDS[kind][1]}
    if kind == "boolean":
        params["target"] = 2
    if kind == "array":
        params["weld"] = 0.001
    mods = tuple(
        legacy._Modifier(id=i + 1, kind=kind, params=tuple(sorted(params.items())))
        for i in range(8_000)
    )
    doc = bd.ClayDoc(
        objects=[bd.Obj(uid=1, name="o", mesh=box), bd.Obj(uid=2, name="t", mesh=bp.box())]
    )
    legacy._bake(doc, {1: mods})
    seconds = len(ran) * _MEASURED_US_PER_TINY_MODIFIER[kind] * 1e-6 * scale
    assert seconds <= 3.0, f"{len(ran) * scale:,} tiny {kind} modifiers ran, about {seconds:.1f} s"


def test_the_modifiers_past_the_bake_budget_are_named_in_one_sentence_not_one_each(
    monkeypatch,
) -> None:
    # The same probe: every modifier past the budget added its own "could not run"
    # notice, so a 2,000-modifier stack opened under a screenful of identical
    # sentences. The first is kept in full; the rest are counted.
    doc = bd.ClayDoc(objects=[bd.Obj(uid=1, name="b", mesh=bp.box())])
    mods = tuple(
        legacy._Modifier(id=i + 1, kind="weld", params=(("distance", 0.001),)) for i in range(30)
    )
    room = 3 * legacy._bake_cost(mods[0], legacy._triangle_count(bp.box()))
    monkeypatch.setattr(legacy, "MAX_BAKE_WORK", room)
    _, errors = legacy._bake(doc, {1: mods})
    notices = legacy.migrate(
        bd.ClayDoc(objects=[bd.Obj(uid=1, name="b", mesh=bp.box())]),
        {1: legacy.LegacyFields(modifiers=mods)},
    )
    assert len(errors[1]) == 2, errors[1]
    assert "past the" in errors[1][0]
    assert "26 more" in errors[1][1]  # 27 refused: the first in full, the other 26 counted
    refusals = [n for n in notices if "could not run" in n or "more modifiers" in n]
    assert len(refusals) == 2, refusals
    assert refusals[1].endswith("were dropped."), refusals[1]


# --- clay-87--------------------------------------------------------------------


def _backtracks(m: bm.Mesh) -> list[int]:
    """Faces with a corner whose two edges fold straight back on each other."""
    found = []
    for f in range(bm.face_count(m)):
        pts = m.positions[m.loops[m.starts[f] : m.starts[f + 1]]].astype("f8")
        k = len(pts)
        for i in range(k):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % k]
            u, v = b - a, c - b
            cross = np.linalg.norm(np.cross(u, v))
            scale = max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12)
            if cross < 1e-9 * scale and np.dot(u, v) < 0:
                found.append(f)
                break
    return found


def _coincident_or_zero_width_faces(m: bm.Mesh) -> list[str]:
    problems = []
    seen: dict[frozenset, int] = {}
    for f in range(bm.face_count(m)):
        corners = m.loops[m.starts[f] : m.starts[f + 1]]
        key = frozenset(tuple(np.round(m.positions[c].astype("f8"), 5)) for c in corners)
        if key in seen:
            problems.append(f"face {f} sits on face {seen[key]}")
        seen[key] = f
        pts = m.positions[corners].astype("f8")
        area = np.linalg.norm(
            sum(np.cross(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))
        )
        if area < 1e-9:
            problems.append(f"face {f} has no area")
    return problems


@pytest.mark.parametrize(
    ("kwargs", "faces"),
    [
        ({"opening_offset": 99.0}, 8),
        ({"opening_offset": -99.0}, 8),
        ({"opening_width": 0.0}, 6),
        ({"opening_width": 0.0, "opening_offset": 0.7}, 6),
    ],
    ids=["right-pillar-gone", "left-pillar-gone", "no-opening", "no-opening-off-centre"],
)
def test_a_doorway_with_a_zero_width_pillar_or_opening_has_no_coincident_opposite_faces(
    kwargs, faces
) -> None:
    for build in (
        lambda: bp.doorway(**kwargs),
        lambda: bp.doorway(**bp.clamp_params("doorway", {**bp.GENERATORS["doorway"][0], **kwargs})),
    ):
        m = build()
        bm.validate(m)
        assert _coincident_or_zero_width_faces(m) == []
        assert _backtracks(m) == []
        assert bm.face_count(m) == faces


def test_a_doorway_with_both_pillars_present_keeps_its_eight_station_profile() -> None:
    assert bm.face_count(bp.doorway()) == 10
    assert _backtracks(bp.doorway()) == []


# --- clay-94 --------------------------------------------------------------------


def _cylinder_file(edit) -> bytes:
    doc = bd.ClayDoc()
    doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="C",
            mesh=bp.cylinder(),
            generator="cylinder",
            params={"radius": 0.5, "height": 1.0, "segments": 8},
        )
    )
    return _rewrite(ser.rblk_bytes(doc), edit)


def _assert_frozen_with_its_mesh(data: bytes) -> None:
    back = ser.read_rblk(data)
    got = back.objects[0]
    assert got.generator is None and got.params == {}
    assert bm.face_count(got.mesh) == bm.face_count(bp.cylinder())
    assert any('"C"' in n for n in back.notices)


def test_read_rblk_freezes_a_generator_param_that_is_a_four_hundred_digit_integer() -> None:
    def edit(scene):
        scene["objects"][0]["params"]["segments"] = 10**400

    _assert_frozen_with_its_mesh(_cylinder_file(edit))


def test_read_rblk_freezes_a_generator_whose_int_param_is_outside_int32() -> None:
    def edit(scene):
        scene["objects"][0]["params"]["segments"] = 3_000_000_000

    _assert_frozen_with_its_mesh(_cylinder_file(edit))


def test_read_rblk_freezes_a_generator_whose_float_param_is_outside_float32() -> None:
    def edit(scene):
        scene["objects"][0]["params"]["radius"] = 1e39

    _assert_frozen_with_its_mesh(_cylinder_file(edit))


def test_read_rblk_freezes_a_generator_with_a_tuple_param_outside_float32() -> None:
    doc = bd.ClayDoc()
    doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="C",
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
        )
    )

    def edit(scene):
        scene["objects"][0]["params"]["size"] = [1.0, 1e39, 1.0]

    back = ser.read_rblk(_rewrite(ser.rblk_bytes(doc), edit))
    assert back.objects[0].generator is None


def test_read_rblk_keeps_an_int_param_at_the_int32_edge_live() -> None:
    def edit(scene):
        scene["objects"][0]["params"]["segments"] = 64

    back = ser.read_rblk(_cylinder_file(edit))
    assert back.objects[0].generator == "cylinder"


# --- clay-95 --------------------------------------------------------------------


def _inward(mesh: bm.Mesh) -> bm.Mesh:
    faces = np.arange(bm.face_count(mesh), dtype="i4")
    out, _sel = ops_topo.flip_normals(mesh, el.ElementSel(faces=faces))
    return out


def _moved(mesh: bm.Mesh, offset) -> bm.Mesh:
    return replace(mesh, positions=(mesh.positions + np.asarray(offset, dtype="f4")).astype("f4"))


def _shell_volume(mesh: bm.Mesh, lo: int, hi: int) -> float:
    return float(ops_clean._face_fan_volume(mesh)[lo:hi].sum() / 6.0)


def test_recalculate_normals_leaves_the_inward_facing_wall_of_a_cavity_alone() -> None:
    hollow = legacy._concat([bp.box((2, 2, 2)), _inward(bp.box((1, 1, 1)))])
    assert ops_clean.recalc_outside(hollow) is hollow


def test_recalculate_normals_leaves_an_off_centre_cavity_wall_alone() -> None:
    cavity = _moved(_inward(bp.box((1, 1, 1))), (0.3, 0.2, -0.4))
    hollow = legacy._concat([bp.box((2, 2, 2)), cavity])
    assert ops_clean.recalc_outside(hollow) is hollow


def test_recalculate_normals_still_turns_an_inside_out_island_inside_a_cavity_outward() -> None:
    # A solid floating in the cavity, wound inward: that one is a mistake.
    scene = legacy._concat(
        [bp.box((4, 4, 4)), _inward(bp.box((2, 2, 2))), _inward(bp.box((0.5, 0.5, 0.5)))]
    )
    out = ops_clean.recalc_outside(scene)
    assert _shell_volume(out, 0, 6) > 0  # the container
    assert _shell_volume(out, 6, 12) < 0  # the cavity wall, untouched
    assert _shell_volume(out, 12, 18) > 0  # the island, turned outward


def test_recalculate_normals_still_turns_a_lone_inside_out_box_outward() -> None:
    out = ops_clean.recalc_outside(_inward(bp.box()))
    assert _shell_volume(out, 0, 6) > 0


def test_recalculate_normals_still_turns_a_disjoint_inside_out_box_outward() -> None:
    scene = legacy._concat([bp.box(), _moved(_inward(bp.box()), (5, 0, 0))])
    out = ops_clean.recalc_outside(scene)
    assert _shell_volume(out, 0, 6) > 0 and _shell_volume(out, 6, 12) > 0
    assert np.array_equal(out.loops[:24], scene.loops[:24])  # the first box untouched


def test_recalculate_normals_still_turns_an_inside_out_box_that_touches_another_outward() -> None:
    scene = legacy._concat([bp.box(), _moved(_inward(bp.box()), (1.0, 0, 0))])
    out = ops_clean.recalc_outside(scene)
    assert _shell_volume(out, 6, 12) > 0


def test_recalculate_normals_turns_outward_an_inside_out_box_in_a_concave_shells_bbox() -> None:
    # Inside the staircase's bounding box but in the air above its low end: a
    # bounding-box test alone would call it a cavity.
    stairs = bp.stairs()
    air = _moved(_inward(bp.box((0.1, 0.1, 0.1))), (0.0, 0.4, -0.45))
    out = ops_clean.recalc_outside(legacy._concat([stairs, air]))
    n = bm.face_count(stairs)
    assert _shell_volume(out, n, n + 6) > 0
    # ...and the same box buried in the solid end really is a cavity.
    buried = _moved(_inward(bp.box((0.1, 0.1, 0.1))), (0.0, 0.0, 0.45))
    scene = legacy._concat([stairs, buried])
    assert ops_clean.recalc_outside(scene) is scene


# --- clay-97 --------------------------------------------------------------------


def test_migrate_does_not_name_a_removed_colliders_unreadable_modifiers() -> None:
    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=1, name="visible", mesh=bp.box()))
    doc.objects.append(bd.Obj(uid=2, name="hull", mesh=bp.box()))
    fields = {
        2: legacy.LegacyFields(
            role="collider",
            unreadable=["the Weld modifier was dropped: it is malformed"],
        )
    }
    text = "\n".join(legacy.migrate(doc, fields))
    assert "dropped" not in text
    assert text.count('"hull"') == 1  # the collider-removal sentence alone


def test_a_boolean_cuts_with_the_baked_mesh_of_a_collider_target_it_then_removes(
    monkeypatch,
) -> None:
    # The old evaluator gave a boolean its target's *evaluated* mesh, collider or
    # not; the 2026-10-09 audit's clay-36 stopped baking a collider's stack and
    # so quietly cut with the unbaked mesh. The stack is baked as the target
    # only: nothing of it is kept or named.
    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=1, name="visible", mesh=bp.box()))
    doc.objects.append(bd.Obj(uid=2, name="hull", mesh=bp.box()))
    seen: list[int] = []

    def fake(objs, op, world=None):
        seen.append(bm.face_count(objs[1].mesh))
        return objs[0].mesh

    monkeypatch.setattr(legacy.ops_boolean, "boolean", fake)
    cut = legacy._Modifier(id=1, kind="boolean", params=(("operation", 1), ("target", 2)))
    double = legacy._Modifier(
        id=1,
        kind="array",
        params=(
            ("count", 2),
            ("offset_x", 3.0),
            ("offset_y", 0.0),
            ("offset_z", 0.0),
            ("weld", 0.0),
        ),
    )
    fields = {
        1: legacy.LegacyFields(modifiers=(cut,)),
        2: legacy.LegacyFields(modifiers=(double,), role="collider"),
    }
    notices = legacy.migrate(doc, fields)
    assert seen == [12]  # the target arrived doubled, not as the raw six faces
    assert [o.name for o in doc.objects] == ["visible"]
    text = "\n".join(notices)
    assert "applied" not in text
    assert text.count('"hull"') == 1


# --- clay-99 --------------------------------------------------------------------


def test_ops_clean_docstring_does_not_claim_the_ceiling_is_shared_by_every_op(
    monkeypatch,
) -> None:
    doc = ops_clean.__doc__ or ""
    assert "every Clay walking op" not in doc
    assert "Every internal helper" not in doc
    assert "The two op bodies below" in doc
    assert "Recalculate Normals alone" in doc
    # And the claim the new wording makes is true.
    monkeypatch.setattr(ops_clean, "MAX_CLEAN_CORNERS", 1)
    box = bp.box()
    assert ops_clean.merge_by_distance(box, 1e-5) is box
    with pytest.raises(el.OpError):
        ops_clean.recalc_outside(box)


# --- clay-100 -------------------------------------------------------------------


@pytest.mark.parametrize("distance", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("which", ["distinct", "coincident", "tiny"])
def test_merge_by_distance_refuses_a_non_positive_or_non_finite_distance_whatever_the_mesh(
    distance, which
) -> None:
    mesh = {
        "distinct": bp.box(),
        "coincident": legacy._concat([bp.box(), bp.box()]),
        "tiny": bp.plane(),
    }[which]
    with pytest.raises(el.OpError, match="distance"):
        ops_clean.merge_by_distance(mesh, distance)


def test_merge_by_distance_still_merges_at_a_positive_finite_distance() -> None:
    dup = legacy._concat([bp.box(), bp.box()])
    assert len(ops_clean.merge_by_distance(dup, 1e-5).positions) == 8


# --- clay-101 -------------------------------------------------------------------

_LEGACY = "realmspinner.kernels.mesh.legacy"


def _legacy_importers(root: Path) -> set[str]:
    """Every module under *root* (a ``realmspinner`` package directory) that
    imports the format-3 migration, by any spelling: ``import x.legacy``,
    ``from x import legacy``, ``from x.legacy import y``, either relative, or an
    ``importlib.import_module`` / ``__import__`` of its dotted name."""
    found: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root.parent).with_suffix("")
        parts = list(rel.parts)
        is_pkg = parts[-1] == "__init__"
        if is_pkg:
            parts = parts[:-1]
        module = ".".join(parts)
        package = parts if is_pkg else parts[:-1]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            hit = False
            if isinstance(node, ast.Import):
                hit = any(a.name == _LEGACY for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[: len(package) - node.level + 1]
                    target = ".".join(base + ([node.module] if node.module else []))
                else:
                    target = node.module or ""
                hit = target == _LEGACY or (
                    target == _LEGACY.rpartition(".")[0]
                    and any(a.name == "legacy" for a in node.names)
                )
            elif isinstance(node, ast.Call) and node.args:
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                arg = node.args[0]
                hit = (
                    name in {"import_module", "__import__"}
                    and isinstance(arg, ast.Constant)
                    and arg.value == _LEGACY
                )
            if hit:
                found.add(module)
    return found


def test_only_serialize_imports_the_legacy_migration() -> None:
    import realmspinner

    root = Path(realmspinner.__file__).parent
    importers = _legacy_importers(root)
    assert importers == {"realmspinner.kernels.mesh.serialize"}, importers


def test_the_legacy_import_sweep_sees_every_spelling_of_the_import(tmp_path: Path) -> None:
    pkg = tmp_path / "realmspinner"
    mesh = pkg / "kernels" / "mesh"
    mesh.mkdir(parents=True)
    other = pkg / "studio"
    other.mkdir()
    for d in (pkg, pkg / "kernels", mesh, other):
        (d / "__init__.py").write_text("", encoding="utf-8")
    spellings = {
        "a_abs": "import realmspinner.kernels.mesh.legacy\n",
        "b_from_pkg": "from realmspinner.kernels.mesh import legacy\n",
        "c_from_mod": "from realmspinner.kernels.mesh.legacy import migrate\n",
        "d_rel_pkg": "from . import legacy\n",
        "e_rel_mod": "from .legacy import migrate\n",
        "f_dyn": "import importlib\nimportlib.import_module('realmspinner.kernels.mesh.legacy')\n",
        "g_fn_scope": "def f():\n    from realmspinner.kernels.mesh import legacy\n",
    }
    for name, source in spellings.items():
        (mesh / f"{name}.py").write_text(source, encoding="utf-8")
    (mesh / "clean.py").write_text("from . import mesh\nimport math\n", encoding="utf-8")
    (other / "up.py").write_text(
        "from ..kernels.mesh.legacy import migrate\nfrom ..kernels.mesh import legacy\n",
        encoding="utf-8",
    )
    found = _legacy_importers(pkg)
    assert found == {f"realmspinner.kernels.mesh.{n}" for n in spellings} | {
        "realmspinner.studio.up"
    }


# --- clay-103 -------------------------------------------------------------------

#: Signed parameters: a direction, not an extent.
_SIGNED = {"opening_offset", "twist"}


def _extent_keys(generator: str) -> list[str]:
    defaults = bp.GENERATORS[generator][0]
    keys = []
    for key, value in defaults.items():
        if key in _SIGNED or isinstance(value, bool):
            continue
        if isinstance(value, float) or (
            isinstance(value, tuple) and value and all(isinstance(v, float) for v in value)
        ):
            keys.append(key)
    return keys


def _plain(value):
    return json.loads(json.dumps(value))


def _zeroed(value):
    return tuple(0.0 for _ in value) if isinstance(value, tuple) else 0.0


def _negated(value):
    if isinstance(value, tuple):
        return tuple(-v for v in value)
    return -value


def test_clamp_params_returns_the_extents_the_generators_actually_build_from() -> None:
    # Floors the builders apply to a zero extent, each read off its builder.
    zero = {"total_height": 1e-4, "total_depth": 1e-4}
    stairs = bp.GENERATORS["stairs"][0]
    got = bp.clamp_params("stairs", {**stairs, "total_height": 0.0, "total_depth": 0})
    assert got["total_height"] == zero["total_height"] and got["total_depth"] == zero["total_depth"]
    got = bp.clamp_params(
        "doorway", {**bp.GENERATORS["doorway"][0], "wall_length": 0.0, "wall_height": 0.0}
    )
    assert got["wall_length"] == 1e-4 and got["wall_height"] == 1e-4
    got = bp.clamp_params("arch", {**bp.GENERATORS["arch"][0], "width": 0.0, "depth": 0.0})
    assert got["width"] == 2e-4 and got["depth"] == 1e-4
    got = bp.clamp_params("column", {**bp.GENERATORS["column"][0], "radius": 0.0})
    assert got["radius"] == bp.MIN_PROFILE_RADIUS
    # A negative extent is stored as its magnitude.
    assert bp.clamp_params("cylinder", {"radius": -0.5, "height": -2.0, "segments": 8}) == {
        "radius": 0.5,
        "height": 2.0,
        "segments": 8,
    }
    assert bp.clamp_params("box", {"size": (-1.0, 2.0, -3.0)}) == {"size": (1.0, 2.0, 3.0)}
    assert bp.clamp_params("box", {"size": [-1.0, 2.0, -3.0]}) == {"size": [1.0, 2.0, 3.0]}
    got = bp.clamp_params("stairs", {**stairs, "width": -2.0, "total_height": -3.0})
    assert got["width"] == 2.0 and got["total_height"] == 3.0


@pytest.mark.parametrize("generator", sorted(bp.GENERATORS))
def test_clamp_params_stores_a_nonnegative_extent_and_builds_the_same_mesh(generator) -> None:
    defaults = bp.GENERATORS[generator][0]
    build = bp.GENERATORS[generator][1]
    keys = _extent_keys(generator)
    raw = {**defaults, **{k: _negated(defaults[k]) for k in keys}}
    clamped = bp.clamp_params(generator, raw)
    for key in keys:
        flat = clamped[key] if isinstance(clamped[key], tuple) else (clamped[key],)
        assert all(v >= 0 for v in flat), (generator, key, clamped[key])
    assert bp.clamp_params(generator, clamped) == clamped  # idempotent
    typed, stored = build(**raw), build(**clamped)
    assert np.array_equal(typed.positions, stored.positions)
    assert np.array_equal(typed.loops, stored.loops)
    # The defaults are already what the builders use (a profile comes back as a
    # list where the default holds a tuple, as it always has).
    assert _plain(bp.clamp_params(generator, defaults)) == _plain(defaults)


@pytest.mark.parametrize("generator", sorted(bp.GENERATORS))
def test_clamp_params_of_a_zero_extent_stores_what_the_builder_floors_it_to(generator) -> None:
    defaults = bp.GENERATORS[generator][0]
    build = bp.GENERATORS[generator][1]
    keys = _extent_keys(generator)
    zeroed = {
        **defaults,
        **{k: _zeroed(defaults[k]) for k in keys},
    }
    try:
        typed = build(**zeroed)
    except Exception:  # noqa: BLE001 -- a zero the builder refuses has nothing to mirror
        pytest.skip("the builder refuses a zero extent")
    clamped = bp.clamp_params(generator, zeroed)
    stored = build(**clamped)
    assert np.array_equal(typed.positions, stored.positions)
    # What was stored builds identically *and* is a fixed point of the clamp.
    assert bp.clamp_params(generator, clamped) == clamped
