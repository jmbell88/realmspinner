"""Clay's agent tool surface, the collider handler family (tranche 7,
``dev/CLAY-PLAN.md``): ``clay_collider`` alone.

A new family file, the same one-tool shape ``tools_uv.py`` lands in as
tranche 6's own family (see that module's own docstring for why one tool
with an enum, not several, is the house steer here). ``clay_collider``
fits :data:`~.colliders.COLLIDER_KINDS` against every named object's own
*evaluated* mesh and adds one child collider object per source, as one
undo step total (``ClayDoc.add_collider``, called once per uid, folded
through a single ``history.mark()``/``collapse_since`` pair the same way
``tools_structure._h_lock``/``_h_tag`` already fold a multi-uid edit).

**Every fit runs before any object is added.** :mod:`.colliders`' five fit
functions are pure -- they read a mesh's positions and return a
:class:`~.colliders.Collider`, touching no document at all -- so this
handler computes every fit first and only then calls ``add_collider`` in a
second pass, the same "validate everything before the first mutation" rule
every other multi-uid door in this fold already follows (``tools._h_delete``'s
own locked pre-check is the precedent). A refusal partway through fitting
(``convex``/``compound`` need four non-coplanar points; see
:mod:`.colliders`' own module docstring) therefore never leaves an earlier
uid's collider standing while a later one refused.

**Not a locking door.** ``ClayDoc.add_collider`` says so plainly: adding a
collider changes nothing about the *source* object itself, so a locked
source can still grow one -- this handler carries no locked pre-check to
match, unlike every mesh-mutating door in ``tools_uv.py``.
"""

from __future__ import annotations

from typing import Any

from .....kernels.mesh import colliders
from .....kernels.mesh.elements import OpError
from .validate import Session, _json, _label_top, _resolve_uids, _scene_row, _tab, fail


def _coerce_collider_param(default: Any, raw: Any) -> Any:
    """*raw* coerced to *default*'s own Python type -- ``bool`` (checked
    before ``int``, since ``bool`` is an ``int`` subclass) for a flag like
    ``oriented``, otherwise ``int`` for a count like ``max_faces``. Derived
    from the default's own type rather than a hand-listed pair of names, so
    a sixth collider kind's own extra keyword enrols itself here too."""
    if isinstance(default, bool):
        return bool(round(float(raw)))
    if isinstance(default, int):
        return int(round(float(raw)))
    return float(raw)


def _h_collider(ctx: Any, session: Session, args: dict) -> dict:
    """Fit *kind* against every named object's evaluated mesh and add one
    collider child each, as one undo step. See this module's own docstring
    for the two-pass (fit-then-add) shape and the locking exemption.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    uids, failure = _resolve_uids(doc, args.get("uids"), field="uids")
    if failure:
        return failure
    if not uids:
        return fail("give at least one uid.", field="uids")

    kind = args.get("kind")
    if kind not in colliders.COLLIDER_KINDS:
        return fail(
            f"kind must be one of {', '.join(sorted(colliders.COLLIDER_KINDS))}.",
            field="kind",
        )

    params_arg = args.get("params")
    if params_arg is None:
        params_arg = {}
    elif not isinstance(params_arg, dict):
        return fail("params must be an object.", field="params")
    _label, fit_func, defaults = colliders.COLLIDER_KINDS[kind]
    unknown = sorted(set(params_arg) - set(defaults))
    if unknown:
        return fail(
            f"params.{unknown[0]} is not a parameter of collider kind {kind!r}.",
            field="params",
        )
    kwargs: dict[str, Any] = {}
    for key, default in defaults.items():
        raw = params_arg.get(key, default)
        try:
            kwargs[key] = _coerce_collider_param(default, raw)
        except (TypeError, ValueError):
            return fail(f"params.{key} must be a number.", field="params")

    fits: list[tuple[int, colliders.Collider]] = []
    for uid in uids:
        mesh = doc.evaluated(uid)
        try:
            fit = fit_func(mesh, **kwargs)
        except OpError as error:
            return fail(str(error), field="uids", uids=[uid])
        fits.append((uid, fit))

    mark = doc.history.mark()
    new_objs = [doc.add_collider(uid, fit) for uid, fit in fits]
    doc.history.collapse_since(mark)
    _label_top(doc, mark, "Add Collider")

    return _json(
        {
            "kind": kind,
            "uids": [o.uid for o in new_objs],
            "objects": [_scene_row(doc, o) for o in new_objs],
        }
    )
