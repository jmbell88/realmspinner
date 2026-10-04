"""Clay's agent tool surface, the collider handler family (Clay tranche 7):
``clay_collider`` alone.

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

HULL_FACES_MIN = 12
"""The smallest ``max_faces`` the door accepts. Not the geometric floor (a
tetrahedron is 4 faces): the reduction hulls a farthest-point sample of
``max_faces // 2 + 2`` vertices, and a handful of points sampled off a round
mesh can all be coplanar, so a small value fails with "every point is
coplanar" on a perfectly good sphere. Measured over every generator at its
defaults (2026-10-03, the audit's clay-83): a uv_sphere, cylinder, capsule,
column and rounded_box need 6 or more, a ``tube`` needs 10. Twelve clears the
worst with a margin; the Clay panel's own control still bottoms out at 4."""

HULL_FACES_MAX = 1024
"""``max_faces`` ceiling: the Clay panel's own bound for the same parameter."""


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
    # isinstance checked first: the 2026-09-26 audit's clay-agent-tools-06 --
    # ``x not in a_dict`` hashes ``x``, and a list or object ``kind`` raised
    # a bare, unhashable ``TypeError`` that only ``call()``'s generic
    # "failed unexpectedly" backstop caught, instead of this refusal.
    if not isinstance(kind, str) or kind not in colliders.COLLIDER_KINDS:
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
            # OverflowError: the 2026-09-26 audit's clay-agent-tools-09 --
            # ``round(float("inf"))`` raises it (``_coerce_collider_param``'s
            # own ``int``/``bool`` branches both go through ``round``),
            # uncaught here before this fix.
            kwargs[key] = _coerce_collider_param(default, raw)
        except (TypeError, ValueError, OverflowError):
            return fail(f"params.{key} must be a number.", field="params")
        if key == "max_faces" and not (HULL_FACES_MIN <= kwargs[key] <= HULL_FACES_MAX):
            # The 2026-10-03 audit's clay-83: any value passed, and one below
            # the hull's floor failed with "every point is coplanar" -- blaming
            # a sphere for the caller's number -- while a huge one defeated the
            # face reduction the default exists for. The range is the Clay
            # panel's own for the same parameter (``ops._collider_params``).
            return fail(
                f"params.max_faces must be between {HULL_FACES_MIN} and "
                f"{HULL_FACES_MAX} (fewer cannot hull a round mesh reliably).",
                field="params",
            )

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
