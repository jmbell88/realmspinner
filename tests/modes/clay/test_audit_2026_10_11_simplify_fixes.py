"""The 2026-10-11 audit's Simplify findings: clay-03, -04, -06, -08, -24, -25, -27, -43.

Each test is named for the claim it holds and was written to fail against the code
the audit read. The kernel's geometry is ``test_ops_simplify.py``'s; what is here
is the shells floor, the dialog's refusal path, the selection-wide ceiling, the
ceiling's cost, the toast, the dissolve tolerance and the mode words.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import measure, topo
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_simplify as simp
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.studio.modes.clay import ops as clay_ops


class _Ctx:
    def __init__(self) -> None:
        self.toasted: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasted.append((message, level))

    @property
    def messages(self) -> list[str]:
        return [message for message, _ in self.toasted]

    @property
    def errors(self) -> list[str]:
        return [message for message, level in self.toasted if level == "error"]


def _bare(mesh: bm.Mesh) -> bm.Mesh:
    return bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
        uv=None,
    )


def _grid(nx: int, nz: int, *, uv: bool) -> bm.Mesh:
    """An ``nx`` by ``nz`` sheet of quads in the XZ plane."""
    xs, zs = np.arange(nx + 1, dtype="f4"), np.arange(nz + 1, dtype="f4")
    positions = np.stack(
        [np.repeat(xs, nz + 1), np.zeros((nx + 1) * (nz + 1), dtype="f4"), np.tile(zs, nx + 1)],
        axis=1,
    )

    def v(i: int, j: int) -> int:
        return i * (nz + 1) + j

    faces = [
        [v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)] for i in range(nx) for j in range(nz)
    ]
    uvs = [
        [
            (i / nx, j / nz),
            ((i + 1) / nx, j / nz),
            ((i + 1) / nx, (j + 1) / nz),
            (i / nx, (j + 1) / nz),
        ]
        for i in range(nx)
        for j in range(nz)
    ]
    return bm.from_faces(positions, faces, uvs if uv else None)


def _doc_of(*meshes: bm.Mesh) -> tuple[Any, list[Any]]:
    doc = bd.ClayDoc()
    objs = [
        doc.add_object(bd.Obj(uid=bd.new_uid(), name=f"O{i}", mesh=mesh))
        for i, mesh in enumerate(meshes)
    ]
    doc.select([obj.uid for obj in objs])
    return doc, objs


# --- clay-03: the four-face floor is per shell -------------------------------


def _two_shells(a: bm.Mesh, b: bm.Mesh, shift: float = 3.0) -> bm.Mesh:
    positions = np.concatenate(
        [a.positions.astype("f8"), b.positions.astype("f8") + [shift, 0.0, 0.0]]
    ).astype("f4")
    loops = np.concatenate([a.loops, b.loops + len(a.positions)]).astype(a.loops.dtype)
    counts = np.concatenate([np.diff(a.starts), np.diff(b.starts)])
    n = len(counts)
    return bm.Mesh(
        positions=positions,
        loops=loops,
        starts=topo.starts_from_counts(counts),
        material=np.zeros(n, dtype="i4"),
        smooth=np.zeros(n, dtype=bool),
        uv=None,
    )


def _shell_face_counts(mesh: bm.Mesh) -> list[int]:
    """Faces per edge-connected shell, by union-find over shared edges."""
    n = bm.face_count(mesh)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[tuple[int, int], int] = {}
    for f in range(n):
        loop = [int(v) for v in bm.face(mesh, f)]
        for p, q in zip(loop, loop[1:] + loop[:1], strict=True):
            key = (min(p, q), max(p, q))
            if key in owner:
                parent[find(f)] = find(owner[key])
            else:
                owner[key] = f
    sizes: dict[int, int] = {}
    for f in range(n):
        sizes[find(f)] = sizes.get(find(f), 0) + 1
    return sorted(sizes.values())


def test_simplify_never_collapses_a_closed_shell_below_four_faces_when_the_mesh_has_two_shells() -> (  # noqa: E501
    None
):
    pair = _two_shells(prim.icosphere(0.5, 1), _bare(prim.uv_sphere(0.5, 8, 4)))
    assert _shell_face_counts(pair) == [32, 80]

    for target in (4, 6):
        out = simp.simplify(pair, target)
        bm.validate(out)
        shells = _shell_face_counts(out)
        assert len(shells) == 2, (target, shells)
        assert min(shells) >= 4, f"a shell collapsed to a pancake at target {target}: {shells}"
        assert measure.volume(out) > 1e-3, f"the shells lost their volume at target {target}"


def test_a_single_shell_still_reaches_a_tetrahedron_at_target_four() -> None:
    out = simp.simplify(_bare(prim.icosphere(0.5, 1)), 4)
    assert bm.face_count(out) == 4
    assert measure.volume(out) > 1e-3


# --- clay-04: the dialog hands a below-floor number to the refusal -----------


@pytest.fixture
def widget_stubs(monkeypatch):
    """``param_widget`` with imgui's drawing stubbed: the fields answer *typed*."""
    from realmspinner.studio.modes.clay.ui.panes import menu

    typed = {"value": 0.0}
    monkeypatch.setattr(menu.widgets, "field_label", lambda *_a, **_k: None)
    monkeypatch.setattr(menu.widgets, "secondary", lambda *_a, **_k: None)
    monkeypatch.setattr(menu.controls, "input_int", lambda *_a, **_k: (True, int(typed["value"])))
    monkeypatch.setattr(menu.controls, "input_float", lambda *_a, **_k: (True, typed["value"]))
    return menu, typed


def test_param_widget_does_not_clamp_a_refuse_below_param_up_to_its_floor(widget_stubs) -> None:
    menu, typed = widget_stubs
    op = clay_ops.get("simplify")
    (param,) = op.params
    typed["value"] = 2.0

    changed, value = menu.param_widget(op.name, param, 100.0)

    assert changed and value == 2.0, "the typed 2 was raised to the floor before Apply"
    ctx, (doc, _obj) = _Ctx(), (_doc_of(prim.box())[0], None)
    assert clay_ops.run(ctx, doc, op, target=value) is False
    assert any("at least 4" in message for message in ctx.errors)


def test_every_refuse_below_param_survives_param_widget_unclamped(widget_stubs) -> None:
    menu, typed = widget_stubs
    seen = []
    for op in clay_ops.OPS:
        for param in op.params:
            if not param.refuse_below:
                continue
            seen.append((op.name, param.name))
            below = param.low - max(abs(param.low) * 0.5, 1.0)
            typed["value"] = below
            changed, value = menu.param_widget(op.name, param, param.default)
            assert changed and value == below, (op.name, param.name, value)
            typed["value"] = param.high * 10
            _, capped = menu.param_widget(op.name, param, param.default)
            assert capped == param.high, "the ceiling still clamps"
    assert ("simplify", "target") in seen, "the sweep found no refuse_below param at all"


def test_a_clamped_param_without_refuse_below_is_still_clamped_by_param_widget(
    widget_stubs,
) -> None:
    menu, typed = widget_stubs
    inset = clay_ops.get("inset")
    param = next(p for p in inset.params if not p.refuse_below)
    typed["value"] = param.low - 5.0
    _, value = menu.param_widget(inset.name, param, param.default)
    assert value == param.low


# --- clay-06 / clay-24: one selection-wide charge, the kernel's ---------------


def test_two_uv_objects_under_the_per_mesh_ceiling_but_over_the_halved_total_are_refused_as_a_selection(  # noqa: E501
    monkeypatch,
) -> None:
    monkeypatch.setattr(simp, "MAX_SIMPLIFY_CORNERS", 2_000)
    a, b = _grid(12, 12, uv=True), _grid(12, 12, uv=True)  # 576 corners each
    for mesh in (a, b):
        simp._refuse_if_too_large(mesh)  # the kernel takes either alone
    doc, objs = _doc_of(a, b)
    ctx = _Ctx()
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("simplify"), target=10) is False

    assert any("Select fewer objects" in message for message in ctx.errors), ctx.toasted
    assert len(doc.history) == depth
    assert all(doc.by_uid(o.uid).mesh is m for o, m in zip(objs, (a, b), strict=True))


def test_objects_already_under_the_target_do_not_count_toward_the_selection_ceiling(
    monkeypatch,
) -> None:
    monkeypatch.setattr(simp, "MAX_SIMPLIFY_CORNERS", 1_000)
    doc, _ = _doc_of(_grid(12, 12, uv=False), _grid(12, 12, uv=False))  # 576 corners each
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("simplify"), target=500) is False

    assert ctx.errors == [], ctx.toasted
    assert ctx.messages == ["Already at or under 500 faces; nothing to simplify."]


def test_the_selection_charge_is_the_kernels_own_per_mesh_charge() -> None:
    bare, textured = _grid(12, 12, uv=False), _grid(12, 12, uv=True)
    assert simp.simplify_cost(textured) == 2 * simp.simplify_cost(bare)
    assert simp.selection_cost([bare, textured], 10) == 3 * simp.simplify_cost(bare)
    assert simp.selection_cost([bare, textured], 144) == 0, "at the target is nothing to do"
    prism = _bare(prim.cylinder(0.5, 1.0, 100))
    assert simp.simplify_cost(prism) == 2 * (len(prism.loops) - 2 * bm.face_count(prism))


# --- clay-08: the ceiling is where a press stays about a second ---------------


def _cube_sphere(k: int) -> bm.Mesh:
    """A closed quad sphere of ``6 k^2`` faces (the dearest ordinary shape)."""
    ids: dict[tuple[float, ...], int] = {}
    positions: list[np.ndarray] = []

    def vid(p: np.ndarray) -> int:
        key = tuple(np.round(p, 7))
        if key not in ids:
            ids[key] = len(positions)
            positions.append(p)
        return ids[key]

    faces = []
    for sign in (1, -1):
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            grid = {}
            for i in range(k + 1):
                for j in range(k + 1):
                    p = np.zeros(3)
                    p[a], p[b], p[c] = sign, 2 * i / k - 1, 2 * j / k - 1
                    grid[i, j] = vid(p / np.linalg.norm(p) * 0.5)
            faces += [
                [grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]]
                for i in range(k)
                for j in range(k)
            ]
    pos = np.array(positions, dtype="f4")
    probe = bm.from_faces(pos, faces)
    centre = np.add.reduceat(probe.positions[probe.loops], probe.starts[:-1], axis=0) / 4
    flip = (bm.face_normals(probe) * centre).sum(axis=1) < 0
    return bm.from_faces(pos, [f[::-1] if fl else f for f, fl in zip(faces, flip, strict=True)])


def test_the_corner_ceiling_is_the_one_a_press_finishes_under_a_second_at() -> None:
    # Measured 2026-10-11: 48,600 corners of quad sphere took 1.34 s and 1.81 s
    # stretched; 18,816 corners took 0.45 s and 0.52 s.
    assert simp.MAX_SIMPLIFY_CORNERS <= 20_000


@pytest.mark.perf
def test_simplify_at_its_corner_ceiling_finishes_well_under_a_second() -> None:
    k = int((simp.MAX_SIMPLIFY_CORNERS / 24) ** 0.5)
    sphere = _cube_sphere(k)
    assert len(sphere.loops) <= simp.MAX_SIMPLIFY_CORNERS
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        out = simp.simplify(sphere, 50)
        best = min(best, time.perf_counter() - t)
    assert bm.face_count(out) <= 60
    assert best < 1.0, f"{len(sphere.loops):,} corners took {best:.2f} s on the frame thread"


# --- clay-25: the toast says the target is per object -------------------------


def _subdivided_box(levels: int) -> bm.Mesh:
    from realmspinner.kernels.mesh import ops_subdiv

    mesh = prim.box()
    for _ in range(levels):
        mesh, _sel = ops_subdiv.subdivide(mesh, el.ElementSel())
    return mesh


def test_a_multi_object_toast_says_the_target_is_per_object() -> None:
    doc, _ = _doc_of(_subdivided_box(2), _subdivided_box(2))
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("simplify"), target=10) is True

    assert ctx.toasted == [
        ("Simplified 192 faces to 12 across 2 objects, each toward 10.", "info")
    ]


def test_a_one_object_toast_keeps_its_plain_sentence() -> None:
    doc, _ = _doc_of(_subdivided_box(2))
    ctx = _Ctx()
    clay_ops.run(ctx, doc, clay_ops.get("simplify"), target=10)
    assert ctx.messages == ["Simplified 96 faces to 6."]


# --- clay-27: the straight-vertex dissolve keeps its stated tolerance ---------


def _disc(n: int) -> bm.Mesh:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rim = np.stack([np.cos(angles), np.zeros(n), np.sin(angles)], axis=1)
    positions = np.concatenate([[[0.0, 0.0, 0.0]], rim]).astype("f4")
    faces = [[0, 1 + (i + 1) % n, 1 + i] for i in range(n)]
    return bm.from_faces(positions, faces)


def _distance_to_outline(points: np.ndarray, outline: np.ndarray) -> np.ndarray:
    """Per point, the distance to the closed polyline *outline* (XZ plane)."""
    a = outline[:, [0, 2]]
    b = np.roll(a, -1, axis=0)
    p = points[:, [0, 2]][:, None, :]
    ab = (b - a)[None]
    t = np.clip(((p - a[None]) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-30), 0.0, 1.0)
    nearest = a[None] + t[..., None] * ab
    return np.linalg.norm(p - nearest, axis=-1).min(axis=1)


def test_a_finely_tessellated_flat_circle_never_drifts_past_the_flat_tolerance() -> None:
    disc = _disc(3000)
    tol = simp._FLAT_TOL * simp._extent(disc)

    out = simp.simplify(disc, 4)

    assert bm.face_count(out) == 1
    outline = out.positions[bm.face(out, 0)].astype("f8")
    assert len(outline) < 3000, "nothing was dissolved at all"
    rim = disc.positions[1:].astype("f8")
    worst = float(_distance_to_outline(rim, outline).max())
    assert worst <= tol * 1.001, f"sagitta {worst:.2e} is {worst / tol:.2f}x the {tol:.1e} bound"


# --- clay-43: the modes are named the way the header's switch names them -------


def test_greyed_reasons_name_the_modes_the_way_the_mode_switch_does() -> None:
    from realmspinner.studio.modes.clay.ui.panes import tools

    words = {mode: label for mode, label, _key in tools.MODE_BUTTONS}
    assert words == clay_ops.MODE_WORDS, "ops.py's table drifted from the header's switch"
    for stored in ("object", "vertex", "edge", "face"):
        assert f"{stored} mode" not in " ".join(words.values())

    extrude = clay_ops.get("extrude")
    doc, _ = _doc_of(prim.box())
    assert doc.element_mode == "object"
    reason = clay_ops.reason_for(extrude, doc)
    assert reason == "Switch to Points, Edges or Faces mode first.", reason

    mirror = clay_ops.get("mirror-copy")
    assert mirror.modes == ("object",)
    assert clay_ops.wrong_mode_reason(mirror) == "Switch to Shape mode first."
    doc.set_element_mode("face")

    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, mirror) is False
    assert ctx.errors == ["Switch to Shape mode first."]
    for op in clay_ops.OPS:
        text = clay_ops.wrong_mode_reason(op)
        for stored in ("object mode", "vertex mode", "edge mode", "face mode"):
            assert stored not in text, (op.name, text)


def test_the_right_click_menu_heading_names_the_mode_the_way_the_header_does(monkeypatch) -> None:
    from realmspinner.studio.modes.clay.ui.panes import menu

    heading: list[str] = []
    monkeypatch.setattr(menu.widgets, "secondary", lambda text, *_a, **_k: heading.append(text))
    monkeypatch.setattr(menu.controls, "menu_separator", lambda *_a, **_k: None)
    monkeypatch.setattr(menu, "_row", lambda *_a, **_k: None)
    doc, _ = _doc_of(prim.box())
    tab = type("Tab", (), {"saving": False})()

    for mode, word in (("object", "Shape"), ("vertex", "Points")):
        doc.set_element_mode(mode)
        heading.clear()
        menu._rows(_Ctx(), None, tab, doc)
        assert heading[0] == f"{word} mode", heading
