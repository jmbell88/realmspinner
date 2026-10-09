"""The selection rule under a randomised sequence of everything that can touch it.

``ClayDoc.selection_problems`` states the rule (object mode holds no picks; an
element mode's picks are on objects being edited; nothing hidden or deleted is
selected; every pick is in range) and this walks 60 seeds x 60 steps of the
doors that write ``selection``, ``element_sel`` or the objects they name,
asking it after every one. Each hand-written selection test pins one path; the
audits kept finding the *next* path that broke the rule (hide, separate,
undo of a show, a rolled-back batch), which is the case for a sequence test.

It also pins the claims that do not need a second document to compare with:
selection and mode calls push no step and leave ``dirty`` alone, undo and redo
never change the mode, and an enabled op either pushes a step or says it did
not run -- the gate that let Delete report ``ran`` over nothing.
"""

from __future__ import annotations

import random

import numpy as np

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops as mesh_ops
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import separate as sep
from realmspinner.studio.modes.clay import ops as clay_ops

SEEDS = 60
STEPS = 60
MODES = ("object", "vertex", "edge", "face")
# Ops that change what is selected, never the document.
NO_STEP_OPS = {"select-all", "select-none", "select-invert", "select-linked"}


class _Ctx:
    """The whole of what an op asks of its context: somewhere to toast."""

    def __init__(self) -> None:
        self.toasts: list[str] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append(message)


def _obj(name: str, rnd: random.Random) -> bd.Obj:
    shape = rnd.choice(["box", "box", "cylinder"])
    mesh = bp.cylinder(segments=8) if shape == "cylinder" else bp.box((1, 1, 1))
    return bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=mesh,
        generator="cylinder" if shape == "cylinder" else None,
        params={"radius": 0.5, "height": 1.0, "segments": 8} if shape == "cylinder" else {},
        translation=np.array([rnd.uniform(-3, 3), 0.0, 0.0]),
    )


def _random_sel(rnd: random.Random, mesh: bm.Mesh, mode: str) -> el.ElementSel:
    """A non-empty selection of *mode*'s own kind, inside the mesh."""
    full = el.select_all(mesh, mode)
    if mode == "vertex":
        pool = full.verts
    elif mode == "edge":
        pool = full.edges
    else:
        pool = full.faces
    n = len(pool)
    take = sorted(rnd.sample(range(n), rnd.randint(1, max(1, n // 2))))
    chosen = np.asarray(pool)[take]
    if mode == "vertex":
        return el.ElementSel(verts=chosen)
    if mode == "edge":
        return el.ElementSel(edges=chosen)
    return el.ElementSel(faces=chosen)


def _fuzz_once(seed: int) -> list[str]:
    rnd = random.Random(seed)
    doc = bd.ClayDoc()
    ctx = _Ctx()
    problems: list[str] = []

    def fail(step: int, kind: str, what: str) -> None:
        problems.append(f"seed {seed} step {step} {kind}: {what}")

    for step in range(STEPS):
        uids = [o.uid for o in doc.objects]
        visible = [o.uid for o in doc.objects if o.visible]
        kind = rnd.choice(
            ["add", "add", "add_many", "remove", "hide", "show", "isolate", "show_all",
             "select", "select", "mode", "mode", "pick", "pick", "pick", "clear",
             "set_mesh", "gen", "separate", "join", "undo", "undo", "redo", "unrecorded",
             "jump", "op", "op", "op", "op", "op", "op", "op", "op"]
        )  # fmt: skip
        if not uids and kind not in ("add", "add_many", "mode", "clear", "undo", "redo", "jump"):
            kind = "add"
        head, dirty, mode = doc.history.head, doc.dirty, doc.element_mode
        selection_before = set(doc.selection)
        pure_selection = kind in ("select", "mode", "pick", "clear")
        ran: bool | None = None
        toasts_before = len(ctx.toasts)
        try:
            if kind == "add":
                doc.add_object(_obj(f"o{step}", rnd))
            elif kind == "add_many":
                doc.add_objects([_obj(f"o{step}.{i}", rnd) for i in range(rnd.randint(1, 3))])
            elif kind == "remove":
                doc.remove_object(rnd.choice(uids))
            elif kind == "hide":
                doc.set_visibility({rnd.choice(uids): False})
            elif kind == "show":
                doc.set_visibility({rnd.choice(uids): True})
            elif kind == "isolate":
                doc.isolate(rnd.sample(uids, rnd.randint(0, min(2, len(uids)))))
            elif kind == "show_all":
                doc.show_all()
            elif kind == "select":
                doc.select(rnd.sample(visible, rnd.randint(0, min(3, len(visible)))))
            elif kind == "mode":
                doc.set_element_mode(rnd.choice(MODES))
            elif kind == "pick":
                if doc.element_mode != "object" and visible:
                    uid = rnd.choice(visible)
                    doc.set_element_sel(
                        uid, _random_sel(rnd, doc.by_uid(uid).mesh, doc.element_mode)
                    )
            elif kind == "clear":
                doc.clear_element_sel()
            elif kind == "set_mesh":
                uid = rnd.choice(visible or uids)
                obj = doc.by_uid(uid)
                moved = bm.transformed(obj.mesh, m3.translation(m3.vec3(1, 0, 0)))
                want = None
                if doc.element_mode != "object" and obj.visible and rnd.random() < 0.5:
                    want = _random_sel(rnd, moved, doc.element_mode)
                doc.set_mesh(uid, moved, select=want)
            elif kind == "gen":
                cyl = [o for o in doc.objects if o.generator == "cylinder"]
                if cyl:
                    obj = rnd.choice(cyl)
                    params = {
                        "radius": rnd.choice([0.4, 0.5, 0.6]),
                        "height": rnd.choice([1.0, 2.0]),
                        "segments": rnd.choice([6, 8, 12]),
                    }
                    doc.set_generator_params(
                        obj.uid, params, bp.cylinder(**params), was={"params": dict(obj.params)}
                    )
            elif kind == "separate":
                obj = doc.by_uid(rnd.choice(uids))
                doc.separate(obj.uid, sep.by_loose_parts(obj.mesh))
            elif kind == "join" and len(uids) >= 2:
                chosen = rnd.sample(uids, 2)
                objs = [doc.by_uid(u) for u in chosen]
                worlds = [doc.world_matrix(u) for u in chosen]
                joined = mesh_ops.join(objs, eps=1e-4, world=worlds)
                doc.join_objects(chosen[0], joined, chosen[1:])
            elif kind == "undo":
                doc.undo()
            elif kind == "redo":
                doc.redo()
            elif kind == "unrecorded":
                doc.undo_unrecorded()
            elif kind == "jump":
                doc.step_history(rnd.randint(0, len(doc.history.history())))
            elif kind == "op":
                choices = [
                    op
                    for op in clay_ops.menu(doc.element_mode)
                    if op.enabled(doc) and op.name != "frame"
                ]
                if choices:
                    op = rnd.choice(choices)
                    kind = f"op:{op.name}"
                    ran = clay_ops.run(ctx, doc, op)
                    if op.name in NO_STEP_OPS:
                        if doc.history.head != head:
                            fail(step, kind, "a selection op pushed an undo step")
                    elif ran and doc.history.head == head and len(ctx.toasts) == toasts_before:
                        fail(step, kind, "was enabled and reported it ran, but pushed no step")
        except (el.OpError, KeyError, ValueError):
            pass

        left = doc.selection_problems()
        if left:
            fail(step, kind, "; ".join(left))
            return problems
        if pure_selection:
            if doc.history.head != head:
                fail(step, kind, "a selection or mode call pushed an undo step")
            if doc.dirty != dirty:
                fail(step, kind, "a selection or mode call changed dirty")
        if kind == "mode" and doc.selection != selection_before:
            fail(step, kind, "set_element_mode changed the selection")
        if kind in ("undo", "redo", "unrecorded", "jump") and doc.element_mode != mode:
            fail(step, kind, f"changed the element mode from {mode} to {doc.element_mode}")
        if problems:
            return problems
    return problems


def test_the_selection_rule_holds_after_every_step_of_a_random_sequence() -> None:
    problems: list[str] = []
    for seed in range(SEEDS):
        problems.extend(_fuzz_once(seed))
        if problems:
            break
    assert not problems, problems[:3]
