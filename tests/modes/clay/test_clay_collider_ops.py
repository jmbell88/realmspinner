"""Tranche 7's OPS registry rows (integration half): one row per
``colliders.COLLIDER_KINDS`` entry.

``kernels.mesh.colliders`` is the kernel half (its own fits are pinned in
``test_colliders.py``, and ``ClayDoc.add_collider`` -- the document door one
of these rows calls five times over -- is pinned in
``test_collider_objects.py``); what belongs here is the registry wiring this
tranche's integration half owns (``dev/CLAY-PLAN.md``): that the context
menu and the tools pane reach ``collider-box``/``collider-sphere``/
``collider-capsule``/``collider-convex``/``collider-compound`` through the
one ``OPS`` list, that each is gated and greyed the same way every other
object-mode row is, that fitting several selected objects adds one collider
child per source as a *single* undo step and leaves the sources selected,
that a degenerate source is refused with the kernel's own sentence, that the
five rows are genuinely *derived* from ``COLLIDER_KINDS`` rather than hand-
written (a monkeypatched sixth kind proves it), and that the agent's derived
``clay_op`` enum picked up all five with nothing hand-listed there.
"""

from __future__ import annotations

import pytest

from realmspinner.kernels.mesh import colliders as colliders_mod
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops


class _Toasts:
    """Only what ``Ctx`` really offers -- ``test_clay_ops.py``'s own double."""

    def __init__(self) -> None:
        self.errors: list[str] = []


class _Ctx:
    def __init__(self) -> None:
        self.toasts = _Toasts()

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.toasts.errors.append(message)


COLLIDER_ROWS = tuple(f"collider-{kind}" for kind in colliders_mod.COLLIDER_KINDS)


def _obj(doc: bd.ClayDoc, name: str, mesh: object | None = None) -> bd.Obj:
    return doc.add_object(
        bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box() if mesh is None else mesh)
    )


# --- registry wiring: menu membership and greyed reasons ---------------------


def test_every_collider_row_is_registered_exactly_once() -> None:
    names = [op.name for op in clay_ops.OPS]
    for name in COLLIDER_ROWS:
        assert names.count(name) == 1, name


@pytest.mark.parametrize("name", COLLIDER_ROWS)
def test_every_collider_row_is_object_only_and_greyed_with_nothing_selected(name: str) -> None:
    doc = bd.ClayDoc()
    assert name in {op.name for op in clay_ops.menu("object")}
    for mode in ("vertex", "edge", "face"):
        assert name not in {op.name for op in clay_ops.menu(mode)}, f"{name}: leaked into {mode}"
    op = clay_ops.get(name)
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc)


def test_box_and_sphere_and_capsule_take_no_params_convex_and_compound_take_max_faces() -> None:
    """The bare-action vs. dialog split ``_collider_params`` derives from
    ``COLLIDER_KINDS``' own keyword defaults, named directly rather than
    just counted -- so a future kind that changes which of these takes a
    dialog is caught here, not just a length assertion passing by luck."""
    assert clay_ops.get("collider-box").params[0].name == "oriented"
    assert clay_ops.get("collider-sphere").params == ()
    assert clay_ops.get("collider-capsule").params == ()
    assert clay_ops.get("collider-convex").params[0].name == "max_faces"
    assert clay_ops.get("collider-compound").params[0].name == "max_faces"


# --- representative runs: one undo step, right role/kind/parent, selection --


@pytest.mark.parametrize("name", COLLIDER_ROWS)
def test_each_collider_kind_adds_one_child_with_the_right_role_kind_parent(name: str) -> None:
    ctx = _Ctx()
    doc = bd.ClayDoc()
    source = _obj(doc, "Crate")
    doc.select([source.uid])
    depth = len(doc.history)
    before = {o.uid for o in doc.objects}

    assert clay_ops.run(ctx, doc, clay_ops.get(name))

    assert len(doc.history) == depth + 1
    added = [o for o in doc.objects if o.uid not in before]
    assert len(added) == 1, "one collider child for one selected source"
    child = added[0]
    assert child.role == "collider"
    assert child.collider_kind == name.removeprefix("collider-")
    assert child.parent == source.uid
    assert not ctx.toasts.errors


def test_fitting_several_selected_sources_adds_one_child_each_as_a_single_step() -> None:
    ctx = _Ctx()
    doc = bd.ClayDoc()
    a = _obj(doc, "A")
    b = _obj(doc, "B")
    doc.select([a.uid, b.uid])
    depth = len(doc.history)
    before = {o.uid for o in doc.objects}

    assert clay_ops.run(ctx, doc, clay_ops.get("collider-sphere"))

    assert len(doc.history) == depth + 1, "two colliders, one undo step"
    added = [o for o in doc.objects if o.uid not in before]
    assert len(added) == 2
    assert {o.parent for o in added} == {a.uid, b.uid}
    assert all(o.role == "collider" and o.collider_kind == "sphere" for o in added)
    assert not ctx.toasts.errors

    assert doc.selection == {a.uid, b.uid}, "the sources stay selected"

    doc.history.undo(doc)
    assert {o.uid for o in doc.objects} == before, "one undo removes both children"


def test_collider_fits_the_evaluated_mesh_not_the_base_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A modifier stack changes what ``doc.evaluated`` returns without
    touching ``obj.mesh`` -- a mirror modifier doubles a box's 8 vertices to
    16. Spying on ``fit_box`` proves the row hands it the *evaluated* mesh,
    not the unmodified base one, the way the integration spec's own words
    ("the selected objects' evaluated meshes") require.

    Patches the *registry entry*, not the ``fit_box`` module attribute:
    ``COLLIDER_KINDS["box"]`` already holds a direct reference to the
    function object, captured once at import, so rebinding
    ``colliders_mod.fit_box`` afterwards would leave that stored reference
    -- and therefore ``_collider_op``'s own lookup -- pointed at the
    original.
    """
    from realmspinner.kernels.mesh import modifiers as mod

    seen: list[object] = []
    real = colliders_mod.fit_box

    def spy(mesh: object, **kwargs: object) -> object:
        seen.append(mesh)
        return real(mesh, **kwargs)

    monkeypatch.setitem(colliders_mod.COLLIDER_KINDS, "box", ("Box", spy, {"oriented": False}))

    doc = bd.ClayDoc()
    source = _obj(doc, "Crate")
    doc.set_modifiers(source.uid, (mod.make("mirror", {"axis": 0, "weld": 0.0}, id=1),))
    doc.select([source.uid])

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("collider-box"))

    assert len(seen) == 1
    assert len(source.mesh.positions) == 8, "the base mesh is untouched"
    assert len(seen[0].positions) == 16, "the fit ran against the evaluated (mirrored) mesh"


def test_collider_convex_forwards_max_faces_to_the_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same reasoning as the box test above: patch the registry entry the
    op actually looks up, not the module attribute."""
    seen: list[int] = []
    real = colliders_mod.convex_hull

    def spy(mesh: object, *, max_faces: int = 64) -> object:
        seen.append(max_faces)
        return real(mesh, max_faces=max_faces)

    monkeypatch.setitem(
        colliders_mod.COLLIDER_KINDS, "convex", ("Convex Hull", spy, {"max_faces": 64})
    )

    doc = bd.ClayDoc()
    source = _obj(doc, "Crate")
    doc.select([source.uid])

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("collider-convex"), max_faces=16.0)

    assert seen == [16]


# --- refusal: the kernel's own sentence, no edit -----------------------------


def test_collider_convex_refuses_a_coplanar_source_with_the_kernels_sentence() -> None:
    ctx = _Ctx()
    doc = bd.ClayDoc()
    source = _obj(doc, "Plane", mesh=bp.plane())
    doc.select([source.uid])
    depth = len(doc.history)
    before = {o.uid for o in doc.objects}

    assert clay_ops.run(ctx, doc, clay_ops.get("collider-convex")) is False

    assert len(doc.history) == depth, "a refusal pushes nothing"
    assert {o.uid for o in doc.objects} == before, "and adds no child"
    assert ctx.toasts.errors and "coplanar" in ctx.toasts.errors[0]


def test_collider_compound_refuses_a_coplanar_source_with_the_kernels_sentence() -> None:
    """``compound`` hulls each loose part with the same
    :func:`~.colliders.convex_hull`-backed ``_hull_from_points``, so a
    single flat source refuses the identical way -- named separately from
    the convex row's own test because it is a different registered op, not
    a re-run of one."""
    ctx = _Ctx()
    doc = bd.ClayDoc()
    source = _obj(doc, "Plane", mesh=bp.plane())
    doc.select([source.uid])
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("collider-compound")) is False

    assert len(doc.history) == depth
    assert ctx.toasts.errors and "coplanar" in ctx.toasts.errors[0]


# --- derivation: a sixth kind registers itself with no edit here -------------


def test_a_sixth_collider_kind_registers_itself_with_no_edit_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claim that makes ``_register_collider_ops`` a loop rather than a
    snapshot: add a kind to ``COLLIDER_KINDS`` the way ``fit_capsule``'s own
    signature already supports, drop the five existing ``collider-*`` rows
    (re-registering over them would hit ``register``'s own duplicate-name
    refusal) and call the registration function again -- the new row
    appears with nothing touched in ``ops.py`` for this test.

    ``monkeypatch`` restores both the dict and ``clay_ops.OPS`` automatically,
    so no other test in this session sees the fake kind or a half-populated
    registry.
    """
    monkeypatch.setitem(
        colliders_mod.COLLIDER_KINDS, "pill", ("Pill", colliders_mod.fit_capsule, {})
    )
    kept = [op for op in clay_ops.OPS if not op.name.startswith("collider-")]
    monkeypatch.setattr(clay_ops, "OPS", kept)

    clay_ops._register_collider_ops()

    names = {op.name for op in clay_ops.OPS}
    assert "collider-pill" in names
    # And every real kind still comes back too -- this is a re-derivation,
    # not a partial one.
    for name in COLLIDER_ROWS:
        assert name in names


# --- the agent's derived enum -------------------------------------------------


def test_the_agent_clay_op_enum_picks_up_every_collider_row_with_no_edit_there() -> None:
    """``clay_op``'s enum is built straight off ``clay_ops.OPS``
    (``agent/dispatch.py``'s own bidirectional derivation gate -- see
    ``test_agent_clay.py``'s identical claim). This is that same promise for
    tranche 7's five new rows specifically.
    """
    from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

    tools = {t.name: t for t in agent_clay.tools()}
    enum = set(tools["clay_op"].schema["properties"]["name"]["enum"])
    missing = [name for name in COLLIDER_ROWS if name not in enum]
    assert not missing, f"clay_op's enum is missing {missing}"
