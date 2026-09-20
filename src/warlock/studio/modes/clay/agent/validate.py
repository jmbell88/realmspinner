"""Clay's agent tool surface, the validation half: the session/document
resolver, the MCP result envelope, the Euler helper a rotation argument needs,
and every shared "check this argument, refuse naming the field" helper the
handlers in ``agent_clay_tools*.py`` call before they mutate anything.

Split out of ``studio/modes/clay/agent/dispatch.py`` in the P4 restructure (``dev/RESTRUCTURE.md``)
-- that module's own "# --- resolving the session's document", "# --- protocol
glue", "# --- Euler XYZ" and "# --- shared validation and mutation helpers"
sections, plus two members the brief's own section banners misfiled: the
module docstring's ``_OBJECT_SELECTION_DERIVED_REFUSAL`` (banner-adjacent to
the constants, but read as one of ``clay_select``/``clay_boolean``'s own
validation refusals, not dispatch state) and ``_validate_query_arg`` (living
under the old file's "# --- output schemas" banner because it happened to be
typed next to :data:`agent_clay_schema._QUERY_ARG_SCHEMAS`, when it is in
fact a validator with no schema-building code in it at all).

**This module is the true leaf of the split, and deliberately so.** It
imports no sibling of this fold -- not ``studio/modes/clay/agent/dispatch.py``, not
``agent_clay_schema``, not any ``agent_clay_tools*`` -- so every one of them
can import *this* module with no risk of a cycle. That is what makes
``ok``/``fail``/``text``/``image_png``/``_json`` and :class:`Session`/
:func:`_tab` live here rather than in ``studio/modes/clay/agent/dispatch.py`` itself: every handler
file needs the result envelope and the session/document resolver on
essentially every line, and ``studio/modes/clay/agent/dispatch.py`` needs to import those same
handler files (to build ``_HANDLERS``) -- so if the envelope lived in
``studio/modes/clay/agent/dispatch.py``, every handler file would have to reach back into the very
module that imports it. Keeping them here instead means the dependency runs
one way only: ``studio/modes/clay/agent/dispatch.py`` and every ``agent_clay_tools*.py`` import
*this* module; this module imports none of them back.

The handful of names that *do* stay behind in ``studio/modes/clay/agent/dispatch.py`` despite a
handler needing them (``call``, ``_HANDLERS``, ``_view_for``,
``PROGRAM_DEADLINE_S``) are exactly the ones a test monkeypatches on
``agent_clay`` by name, or that are ``studio/modes/clay/agent/dispatch.py``'s own dispatch table --
see that module's docstring for why those specific few are read back through
a lazy, function-scope ``from . import agent_clay`` at call time rather than
imported here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .....kernels.geom3d import math3d as m3
from .....kernels.mesh import elements as el
from .....kernels.mesh import mesh as bm
from .....kernels.mesh import ops as clay_geom_ops
from .....kernels.mesh import ops_modifiers as clay_ops_modifiers
from .....kernels.mesh import uvtools
from .....kernels.mesh.elements import OpError
from .. import mode as clay_mode

# --- protocol glue ------------------------------------------------------------
#
# Imported lazily inside functions rather than at module scope. The original
# reason was that ``warlock.mcp`` was still being written alongside
# ``studio/modes/clay/agent/dispatch.py`` and a module-scope import would have failed at collection
# time on however far that sibling had got; both modules exist now, so that
# reason is spent and is not what keeps this here. What keeps it is the
# direction of the dependency: this module (and every ``agent_clay_tools*.py``
# that imports it) is reachable from panes and their tests --
# ``modes.clay.ui.panes.tools`` among them -- none of which want the protocol leaf
# loaded to ask this module a question about Clay, and ``mcp/`` is a leaf that
# must never learn about ``studio`` in return (``tests/mcp/test_mcp_imports.py``
# pins that). One accessor, below, is the whole of the coupling.


def _protocol() -> Any:
    """Named `_protocol` for history, not for where the names now live:
    `Tool`/`ok`/`fail`/`text`/`image_png`/`MAX_FRAME` are `mcp/rpc.py`'s own
    vocabulary (`mcp/protocol.py` only re-exports them for the bridge's MCP
    path), and importing `rpc` directly here -- rather than `protocol` --
    is what keeps `warlock.studio` from ever importing `warlock.mcp.protocol`
    (`tests/mcp/test_mcp_imports.py` pins that)."""
    from .....mcp import rpc

    return rpc


def ok(*content: dict, structured: dict | None = None) -> dict:
    return _protocol().ok(*content, structured=structured)


def fail(message: str, *, changed: bool = False, **extra: Any) -> dict:
    """Thin wrapper over ``protocol.fail`` -- see that function's own
    docstring for the wire shape ``extra`` (``field=``, now also ``changed=``,
    ``recovery=``, ``uids=``, ``op=``) lands in.

    ``changed`` says whether *this session's own document* -- the ``ClayDoc``
    itself, walked by uid through ``tab.doc`` -- was modified before this
    refusal fired. Not whether a Library row was minted (``clay_export``
    refusing after ``clay_mode.build_asset`` has already run would still be
    ``changed: false``, because a model row is not this document) and not
    whether a session-scoped reference was added or removed (``clay_reference_add``
    never touches a ``ClayDoc`` at all) -- the document, exactly as
    ``agent_clay_tools._h_transform`` and ``agent_clay_tools._h_set_params``
    already use the word on the success side.

    Defaulted here, once, rather than passed at each of this fold's ~100
    ``fail(...)`` call sites, so every refusal answers the question by
    construction and a handler that refuses *after* it has already mutated
    the document is the only kind that has to say so explicitly. Exactly one
    does: ``clay_batch``, whose documented contract is that it stops at the
    first refusal and *keeps what already ran*, so it computes the answer
    from its own history mark rather than defaulting. Every other refusal in
    this fold validates before it mutates -- the rule
    ``docs/manual/46-extending.md`` states for a new tool -- and
    ``tests/modes/clay/test_agent_clay.py`` proves it against the document itself, by
    walking every handler and checking a ``changed: false`` refusal really
    did leave the history, the dirty flag and the object count alone.

    ``recovery`` is defaulted the same way, and from the field itself: a
    refusal that names a ``field`` is by definition telling the client which
    argument was wrong, which is ``"fix_arguments"`` -- so naming one is
    enough and the 80-odd refusals that already do get their recovery
    without a second keyword each. An explicit ``recovery=`` always wins,
    which is what the exceptions rely on: :func:`_resolve_uid`'s missing uid
    is ``"read_scene"`` even though it names ``field="uid"``, because the
    argument may be perfectly well-formed and the document simply no longer
    holds it, and the stale-``expect_stamp`` refusal is ``"read_scene"`` for
    the same reason -- its own sentence already says to go and read the
    stamp again. A refusal that names no field and passes no recovery keeps
    none, which is the honest answer for the blanket ``except`` in
    ``agent_clay.call``: nothing there knows what a client should do
    differently.

    ``RECOVERY`` (the closed vocabulary these strings are drawn from) stays
    in ``studio/modes/clay/agent/dispatch.py`` rather than here: nothing in this function -- or
    anywhere in this fold -- checks a ``recovery=`` value against it at
    runtime, so its only readers are ``studio/modes/clay/agent/dispatch.py``'s own module docstring
    and the tests that walk every refusal this fold produces, and it is
    exactly the kind of small, referenced-by-name-in-tests constant that
    stays put with the module callers already reach it through
    (``agent_clay.RECOVERY``).
    """
    if "recovery" not in extra and extra.get("field"):
        extra["recovery"] = "fix_arguments"
    return _protocol().fail(message, changed=changed, **extra)


def text(s: str) -> dict:
    return _protocol().text(s)


def image_png(data: bytes) -> dict:
    return _protocol().image_png(data)


def _over_frame_budget(payload: Any) -> dict | None:
    """Refuse *payload* before ``_json`` encodes a reply ``send_bytes`` can
    never carry, rather than let it reach the wire and fail there past the
    point a refusal could explain itself -- the same shape
    ``agent_clay_tools_ops._h_render``'s own ``_over_frame_budget`` already
    checks for a render's base64 payload, against ``protocol.MAX_FRAME``.

    Only two callers need this: ``_h_scene`` and whole-document
    ``_h_diagnose``, whose reply size scales with the *document's* own size
    (every object, every call) rather than with one call's own arguments --
    every other JSON-replying tool already bounds its own reply through a
    per-call ceiling (``clay_elements``' paging, ``clay_add_mesh``'s vertex
    and face caps, ``clay_analyze``'s 64-object limit). The 2026-09-18 audit's
    agents-03 found neither checked at all: a document of about 22,000
    primitives encodes to roughly 9.1 MB, past ``protocol.MAX_FRAME`` (8 MiB),
    with nothing in either handler that would refuse or page it.
    """
    import json

    encoded_len = len(json.dumps(payload))
    if encoded_len > _protocol().MAX_FRAME:
        return fail(
            "This reply is too large to send back in one frame; narrow the "
            "request (fewer objects, or a single uid) and try again."
        )
    return None


def _json(payload: Any) -> dict:
    """The shape every tool whose reply carries no picture answers in:
    *payload* as text (``json.dumps``, what a model actually reads) and,
    duplicated, as ``structuredContent`` (what a client branches on instead
    of re-parsing that text) -- see ``studio/modes/clay/agent/dispatch.py``'s own module docstring
    for the structured-results paragraph in full, and for the rule (a result
    that carries a picture does not duplicate its header) that excludes the
    two tools which do. The duplication is deliberate, not an oversight to
    dedupe away later.

    ``structured=payload`` only when *payload* is a ``dict`` -- MCP requires
    an object there, never a list or a scalar. Every one of this fold's own
    call sites already passes a dict, so this guard is a floor for whatever
    calls ``_json`` next, not a case any of them actually hits today.
    Routed through the already-serialized text (``json.loads`` of the same
    ``json.dumps`` the text block uses) rather than *payload* itself, so a
    tuple or a numpy scalar buried in ``params`` reaches ``structuredContent``
    as the plain list or number the wire format would have turned it into
    anyway -- the two blocks are meant to be the same JSON, not merely
    ``==``-comparable Python objects that happen to serialize the same way.

    ``clay_render`` and ``clay_reference_get`` never call this -- both answer
    with an image block, which is not JSON to duplicate, so each builds its
    own result directly with ``ok(...)`` instead. See
    ``agent_clay_tools_ops._h_render``'s and
    ``agent_clay_tools_batch._h_reference_get``'s own returns for where that
    exclusion is made.
    """
    import json

    encoded = json.dumps(payload)
    structured = json.loads(encoded) if isinstance(payload, dict) else None
    return ok(text(encoded), structured=structured)


# --- session and document resolution -----------------------------------------


@dataclass
class Session:
    """One MCP connection's claim on Clay. See ``studio/modes/clay/agent/dispatch.py``'s own module
    docstring."""

    tab_uid: str = ""
    references: dict[str, Any] = field(default_factory=dict)
    """Pictures this session has been handed to match against, by name --
    values are ``agent_refs.Reference``. In memory on the session only; see
    the module docstring's references paragraph for why never on the
    document."""

    last_render_png: bytes | None = field(default=None, repr=False)
    """The most recent PNG this session's ``clay_render`` produced -- read
    by ``agent_resources.read_dynamic`` for the ``warlock://clay/render/last``
    resource, ``None`` until the first render. Holds at most one picture,
    overwritten by the next render, never a history -- bounded the same way
    ``references`` is bounded by being session-scoped rather than kept
    forever."""


def _tab(ctx: Any, session: Session, *, create: bool = False) -> tuple[Any, dict | None]:
    """The session's own tab, or a failure result to return unchanged.

    ``create`` is only ever passed by the three tools that can act on an
    empty session -- adding the first primitive, figure or hand-built mesh --
    so every other tool refuses outright rather than silently starting a
    document nobody asked for. Resolved through ``ClayState`` on every call,
    never cached on the session, so a tab the user closed from the keyboard
    is seen as gone on the very next tool call rather than on whichever call
    happens to notice.
    """
    state = clay_mode.ensure(ctx)
    if session.tab_uid:
        tab = state.get(session.tab_uid)
        if tab is not None:
            return tab, None
        if not create:
            return None, fail(
                "This session's document was closed. Call clay_add_primitive, "
                "clay_add_figure or clay_add_mesh to start a new one.",
                recovery="start_document",
            )
        # The pin is released here rather than left standing, because leaving
        # it made the refusal above impossible to follow. It named
        # ``clay_add_primitive`` as the way out, but that tool is exactly the
        # one that arrives with ``create=True`` -- and a truthy ``tab_uid``
        # sent it straight back into this branch and out with the same
        # sentence, forever. A session whose document the user closed was
        # therefore bricked for the rest of the connection: every tool
        # refused, and the one the refusal told it to call refused
        # identically. Clearing the pin first is what makes the mint below
        # reachable, and it keeps the blast-radius rule intact rather than
        # widening it -- the session still owns exactly one tab and still
        # cannot name anybody else's, it is simply allowed to be handed a new
        # one after the old one is provably gone.
        session.tab_uid = ""
    if not create:
        return None, fail(
            "This session has no document yet. Call clay_add_primitive, "
            "clay_add_figure or clay_add_mesh first.",
            recovery="start_document",
        )
    tab = clay_mode.new_document(ctx)
    session.tab_uid = tab.uid
    return tab, None


# --- Euler XYZ, for clay_transform and clay_scene ----------------------------


def _quat_from_euler_xyz(degrees: Any) -> Any:
    """Three degrees -- rotate-X, then Y, then Z -- as this document's XYZW quaternion.

    ``viewer.math3d`` carries ``quat_from_axis_angle`` and ``quat_mul`` but no
    Euler helper at all, and that is not an oversight to fix upstream: nothing
    else in Clay needs one. A gizmo drag accumulates a single axis-angle
    increment directly into the object's quaternion and never decomposes it
    back into three numbers, so there has never been a second caller to share
    this with. An agent describing an orientation has no such luxury -- "face
    this way" arrives as three degrees -- so the composition lives here, once,
    for the handlers that take them.

    Intrinsic X, then Y, then Z, which is the order a person reaching for
    "rotation" with no further qualification expects (it is Blender's default
    Euler order). ``quat_mul(a, b)`` applies ``b`` first, so building the
    result as ``qz * qy * qx`` puts X innermost -- applied first -- exactly
    matching that order.
    """
    rx, ry, rz = (math.radians(float(v)) for v in degrees)
    qx = m3.quat_from_axis_angle(m3.vec3(1.0, 0.0, 0.0), rx)
    qy = m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), ry)
    qz = m3.quat_from_axis_angle(m3.vec3(0.0, 0.0, 1.0), rz)
    return m3.quat_mul(m3.quat_mul(qz, qy), qx)


def _euler_xyz_from_quat(q: Any) -> tuple[float, float, float]:
    """The exact inverse of :func:`_quat_from_euler_xyz`, in degrees.

    A gizmo drag accumulates axis-angle increments straight into an object's
    quaternion and never decomposes them back -- so until an agent needed to
    *read back* what it placed, nothing in Clay ever needed this inverse.
    ``clay_scene`` hands an agent three degrees rather than four quaternion
    components precisely so the readout is something ``clay_transform`` can
    be fed straight back into; a scene description an agent cannot act on is
    not a description worth giving it.

    Derived from ``m3.quat_to_mat4``, which is column-vector convention, so
    with ``R = Rz.Ry.Rx`` (the same composition order ``_quat_from_euler_xyz``
    builds): ``ry = asin(clamp(-R[2, 0], -1, 1))``; away from gimbal lock,
    ``rx = atan2(R[2, 1], R[2, 2])`` and ``rz = atan2(R[1, 0], R[0, 0])``; at
    gimbal lock (``|cos(ry)|`` tiny) ``rz`` is pinned to 0 and ``rx`` is read
    off row 0 instead -- ``atan2(R[0, 1], R[0, 2])`` at ``ry`` ~= +90 deg,
    ``atan2(-R[0, 1], -R[0, 2])`` at ``ry`` ~= -90 deg. The round-trip claim
    this exists to satisfy is about the *rotation* the three angles describe,
    not the three numbers themselves -- at gimbal lock a whole family of
    ``(rx, rz)`` pairs describes the same orientation, and picking ``rz = 0``
    is simply one member of it.
    """
    r = m3.quat_to_mat4(q)[:3, :3]
    sin_ry = -float(r[2, 0])
    ry = math.asin(max(-1.0, min(1.0, sin_ry)))
    if abs(math.cos(ry)) > 1e-6:
        rx = math.atan2(r[2, 1], r[2, 2])
        rz = math.atan2(r[1, 0], r[0, 0])
    else:
        rz = 0.0
        rx = (
            math.atan2(r[0, 1], r[0, 2])
            if sin_ry > 0
            else math.atan2(-r[0, 1], -r[0, 2])
        )
    return (math.degrees(rx), math.degrees(ry), math.degrees(rz))


# --- shared validation and mutation helpers -----------------------------------


def _resolve_uid(doc: Any, args: dict, key: str = "uid") -> tuple[Any, dict | None]:
    """*doc*'s object named by ``args[key]``, or a refusal naming ``field=key``.

    The ``int(args[key])`` / ``doc.by_uid`` / refusal dance that
    ``clay_transform``, ``clay_set_params`` and ``clay_diagnose`` each spelled
    out separately -- three copies of one lookup, free to drift on the wording
    or the field name the moment one of them was edited and the others were
    not.

    **An absent uid is a different refusal from an unknown one.** The
    2026-09-15 Clay agent benchmark sitting's model called ``clay_select_by`` with no
    arguments at all and was told "no object with uid None." -- naming a uid
    it never passed, and pointing it at ``read_scene``, when re-reading the
    scene could not have helped and the fix was in its own arguments. The
    wording is ``clay_select_by``'s own for a missing query argument, so the
    two missing-value refusals on one tool read as one sentence.
    """
    if args.get(key) is None:
        return None, fail(f"give a value for {key!r}.", field=key, recovery="fix_arguments")
    try:
        uid = int(args[key])
        obj = doc.by_uid(uid)
    except (KeyError, ValueError, TypeError):
        return None, fail(
            f"no object with uid {args.get(key)!r}.", field=key, recovery="read_scene"
        )
    return obj, None


def _resolve_uids(
    doc: Any, values: Any, field: str = "uids"
) -> tuple[list[int] | None, dict | None]:
    """*values* cast to ints, every one of them present in *doc*, or a refusal.

    The list version of :func:`_resolve_uid`, shared by ``clay_material`` and
    ``clay_select`` -- each of which cast to int, refused on a bad type, and
    refused again on an unknown uid, by hand. Deliberately silent about
    emptiness: ``clay_select`` means "clear the selection" by an empty list,
    while ``clay_material`` and ``clay_delete`` refuse one themselves, because
    only they have an opinion about it.
    """
    try:
        uids = [int(u) for u in values or []]
    except (TypeError, ValueError):
        return None, fail(
            f"{field} must be a list of integers.", field=field, recovery="fix_arguments"
        )
    known = {obj.uid for obj in doc.objects}
    missing = [u for u in uids if u not in known]
    if missing:
        return None, fail(
            f"no object with uid(s) {missing}.",
            field=field,
            recovery="read_scene",
            uids=missing,
        )
    return uids, None


def _validate_vec3(value: Any, field: str) -> tuple[list[float] | None, dict | None]:
    """Three finite numbers, or a refusal naming *field*.

    Shared by every optional TRS vector ``clay_add_primitive`` and
    ``clay_add_figure`` take, so a malformed one is caught before anything is
    placed -- see those tools' "validate everything before the first
    mutation" rule.
    """
    if not isinstance(value, list) or len(value) != 3:
        return None, fail(
            f"{field} must be an array of 3 numbers.", field=field, recovery="fix_arguments"
        )
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError):
        return None, fail(
            f"{field} must be an array of 3 numbers.", field=field, recovery="fix_arguments"
        )
    if not all(math.isfinite(v) for v in out):
        return None, fail(
            f"{field} must be finite numbers.", field=field, recovery="fix_arguments"
        )
    return out, None


def _validate_number(value: Any, field: str) -> tuple[float | None, dict | None]:
    """One finite number, unbounded -- the plain scalar case
    :func:`_validate_unit` (0..1) and :func:`_validate_number_or_vec` (number
    *or* array) both specialise. Added for ``clay_select_by``'s ``max_angle``,
    which is neither: a query argument this fold owns the schema for (see
    ``agent_clay_schema._QUERY_ARG_SCHEMAS``), not a colour component or a
    params value.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None, fail(f"{field} must be a number.", field=field, recovery="fix_arguments")
    if not math.isfinite(out):
        return None, fail(f"{field} must be finite.", field=field, recovery="fix_arguments")
    return out, None


def _validate_unit(value: Any, field: str) -> tuple[float | None, dict | None]:
    """One finite number in 0..1, or a refusal naming *field*.

    Added beside :func:`_validate_vec3` for the same reason: ``clay_material``
    used to check a colour component with a bare ``isinstance(c, int | float)``,
    which ``float("nan")`` passes as readily as a real number is a float, and
    checked ``metallic``/``roughness`` with nothing at all
    (``float(args.get("metallic", 0.0))``) -- so a NaN in any of the three
    landed straight in the palette and rode along into every export from
    then on.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None, fail(
            f"{field} must be a number, 0..1.", field=field, recovery="fix_arguments"
        )
    if not math.isfinite(out) or not (0.0 <= out <= 1.0):
        return None, fail(
            f"{field} must be a number, 0..1.", field=field, recovery="fix_arguments"
        )
    return out, None


def _validate_range(
    value: Any, field: str, lo: float, hi: float
) -> tuple[float | None, dict | None]:
    """One finite number in ``lo..hi``, or a refusal naming *field*.

    :func:`_validate_unit` fixed at 0..1 for a colour component; this is the
    same check with the bound as an argument, for ``clay_analyze``'s three
    tolerances, each declared with its own ``minimum``/``maximum`` in the
    schema and none of them 0..1.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None, fail(
            f"{field} must be a number, {lo}..{hi}.", field=field, recovery="fix_arguments"
        )
    if not math.isfinite(out) or not (lo <= out <= hi):
        return None, fail(
            f"{field} must be a number, {lo}..{hi}.", field=field, recovery="fix_arguments"
        )
    return out, None


def _validate_number_or_vec(
    value: Any, field: str
) -> tuple[float | list[float] | list[list[float]] | None, dict | None]:
    """A number, an array of numbers, or an array of arrays of numbers, every
    one of them finite -- the ``number | array-of-numbers | array-of-arrays``
    shape ``clay_set_params``'s own schema declares for a param value (a
    cylinder's ``radius`` is one number, a box's ``size`` is three, a
    lathe's ``profile`` is an array of ``[radius, y]`` pairs) -- or a refusal
    naming *field*.

    A schema declaring a shape does not enforce it on the wire:
    ``mcp/protocol.py``'s ``tools/call`` handling checks only that
    ``arguments`` as a whole is a dict before handing it to the handler, so a
    NaN or an infinity reaches here exactly as an agent typed it. Added
    alongside :func:`_validate_unit` when an unvalidated ``clay_transform``
    committed a two-element translation that bricked ``clay_scene`` for the
    whole document (see ``document.set_transform``'s own backstop) --
    ``clay_set_params`` had the identical hole: a non-finite value in
    ``size`` sailed past ``bp.clamp_params`` (which only clamps the keys it
    knows a floor for) and baked straight into the generator's vertex
    positions.

    The array-of-arrays branch was added for ``lathe``'s ``profile``, the
    first generator parameter whose own elements are arrays rather than
    numbers: before it, this function's flat-array branch tried
    ``float(v)`` on each *row* of a profile and raised ``TypeError``, which
    came back as "params must be a number or an array of numbers" -- true of
    the old schema and wrong about the new one, since an array of arrays is
    exactly what a profile is and exactly what ``agent_clay_schema.
    _params_value_schema`` now declares. Every row must itself be a
    non-empty array of finite numbers, and the outer array must not be empty
    either -- the same two rules the flat case already holds a bare array
    to, one level up.
    """
    if isinstance(value, list) and value and all(isinstance(row, list) for row in value):
        try:
            rows = [[float(v) for v in row] for row in value]
        except (TypeError, ValueError):
            return None, fail(
                f"{field} must be a number, an array of numbers, or an "
                "array of arrays of numbers.",
                field=field,
                recovery="fix_arguments",
            )
        if not all(row and all(math.isfinite(v) for v in row) for row in rows):
            return None, fail(
                f"{field} must be finite numbers, with no empty row.",
                field=field,
                recovery="fix_arguments",
            )
        return rows, None
    if isinstance(value, list):
        try:
            out = [float(v) for v in value]
        except (TypeError, ValueError):
            return None, fail(
                f"{field} must be a number, an array of numbers, or an "
                "array of arrays of numbers.",
                field=field,
                recovery="fix_arguments",
            )
        if not out or not all(math.isfinite(v) for v in out):
            return None, fail(
                f"{field} must be finite numbers.", field=field, recovery="fix_arguments"
            )
        return out, None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None, fail(
            f"{field} must be a number, an array of numbers, or an array "
            "of arrays of numbers.",
            field=field,
            recovery="fix_arguments",
        )
    if not math.isfinite(out):
        return None, fail(f"{field} must be finite.", field=field, recovery="fix_arguments")
    return out, None


def _validate_params_values(params: dict, field: str) -> dict | None:
    """Every value of a generator's ``params`` dict through
    :func:`_validate_number_or_vec`, naming *every* offending key in one
    refusal rather than only the first -- the shared body behind the
    identical loops ``_h_add_primitive`` and ``_h_set_params`` used to run,
    each passing the literal string ``"params"`` in as *field* for every
    value, so a lathe's bad ``profile`` beside a good ``segments`` came back
    as "params must be finite numbers." with nothing to say which of the
    two was wrong -- the same class of defect ``agent_clay._unknown_argument_
    refusal`` closed for a misspelled argument name.

    ``field`` on the returned refusal is always exactly *field* itself
    (``"params"`` for both callers): ``tests/test_agent_schemas.py``'s
    ``_run_case`` walks only a refusal's top-level property, never a
    sub-field, so widening it to ``"params.profile"`` would break that
    walk. Only the *message* may name a key, which is why each value is
    checked under a per-key display name (``"params.profile"``) that never
    leaves this function -- :func:`_validate_number_or_vec` builds its
    message from whatever field string it is handed, so handing it a
    compound one is enough to get the key into the text without teaching it
    anything about ``params`` itself. Every failing key's message is kept,
    sorted the same deterministic way ``agent_clay._unknown_argument_refusal``
    sorts its unknown names, so a caller that got two params wrong learns
    about both without a second round trip.

    Returns ``None`` when every value already validates.
    """
    messages = []
    for key in sorted(params):
        _, failure = _validate_number_or_vec(params[key], f"{field}.{key}")
        if failure:
            messages.append(failure["content"][0]["text"])
    if not messages:
        return None
    return fail(" ".join(messages), field=field)


def _params_shape_refusal(params: dict, defaults: dict, field: str, subject: str) -> dict | None:
    """Every value of *params* held to the **shape of the generator's own
    default** for that key, or a refusal naming every key that disagrees.

    :func:`_validate_params_values` checks a value is made of finite numbers
    and stops there, because that is all the wire schema declares: a param
    value is ``number | array-of-numbers | array-of-arrays``, one shape for
    every key of every generator. Which of the three a *particular* key
    wants is not in that schema, and nothing downstream asked either -- so
    ``clay_add_primitive("pyramid", params={"base": [1, 1, 1]})`` walked
    straight into ``primitives.pyramid``'s ``float(base)`` and came back as
    "failed unexpectedly; see the log" with a ``TypeError`` traceback in it,
    the generic backstop catching what should have been a field-named
    refusal (found by the furniture author, 2026-09-12). The same hole ran
    the other way: ``box`` with ``size=1.0`` raised ``TypeError`` on the
    unpack, and ``size=[1, 1]`` a ``ValueError`` about three values, both
    with the same unhelpful face.

    The rule is **derived from ``GENERATORS``' defaults, never listed**, for
    the reason ``agent_clay.tools`` is: every default dictionary is a
    complete call (``primitives.GENERATORS``' own docstring), so the default
    *is* the shape, and a sixteenth generator enrols itself. A scalar
    default wants a scalar; a flat sequence default wants a flat array of
    exactly that many numbers; a sequence-of-rows default (``lathe``'s
    ``profile``, ``tube``'s ``path``, ``sweep``'s ``outline``) wants an
    array of rows of exactly that row's width. The outer length of a row
    array is free -- that is how many points the profile has, which is the
    caller's to choose -- but the row width is not, and a three-number row
    handed to ``lathe`` silently dropped its third column rather than saying
    so.

    *subject* is what the message calls the generator (``'pyramid'`` for
    ``clay_add_primitive``, ``"'pyramid' (uid 4)"`` for ``clay_set_params``,
    which addresses many objects and must say which one). ``field`` on the
    refusal stays exactly *field*, for the reason
    :func:`_validate_params_values` gives: only the message may name a key.
    """
    messages = []
    for key in sorted(params):
        if key not in defaults:
            continue
        want, value = defaults[key], params[key]
        if not isinstance(want, list | tuple):
            if isinstance(value, list):
                messages.append(f"{field}.{key} must be a single number for {subject}.")
            continue
        rows = [r for r in want if isinstance(r, list | tuple)]
        if rows:
            width = len(rows[0])
            ok = (
                isinstance(value, list)
                and bool(value)
                and all(isinstance(row, list) and len(row) == width for row in value)
            )
            if not ok:
                messages.append(
                    f"{field}.{key} must be a non-empty array of "
                    f"{width}-number arrays for {subject}."
                )
        elif not (isinstance(value, list) and len(value) == len(want)):
            messages.append(
                f"{field}.{key} must be an array of {len(want)} numbers for {subject}."
            )
    if not messages:
        return None
    return fail(" ".join(messages), field=field, recovery="fix_arguments")


def _op_params_type_refusal(op: Any, params: dict, field: str = "params") -> dict | None:
    """Every value of ``clay_op``'s ``params`` held to **the parameter
    ``clay_ops`` itself declares** for that name -- a name in ``op.params``,
    and (for one it does declare) a single finite number, the only shape any
    ``clay_ops.Param`` ever stores (``Param``'s own docstring: ``boolean``
    and ``choices`` are widget kinds over the same float, a checkbox writing
    0.0/1.0 and a combo writing its index).

    Mirrors :func:`_params_shape_refusal`'s rule for ``clay_add_primitive``,
    but keyed on ``Op.params`` rather than a generator's own defaults --
    that is the registry ``clay_op`` actually dispatches into.

    Before this (2026-09-14 audit, docs-06 / TODO F9), a bad value in
    ``params`` reached ``clay_ops.run`` and crashed instead of refusing:
    ``mirror-x`` -- which declares no params at all, its axis closed over
    rather than passed -- given ``{"axis": 0}`` sailed past ``run``'s own
    clamp loop (empty, since ``op.params`` is empty) and into
    ``op.run(ctx, doc, **values)``, where the caller's ``axis`` collided
    with the one already bound in the closure: ``TypeError: mirror() got
    multiple values for argument 'axis'``, surfaced as "failed unexpectedly;
    see the log." ``mirror-copy`` (whose ``axis`` *is* declared, a
    ``choices`` param stored as an index) given ``{"axis": "x"}`` reached
    ``run``'s ``float(values[param.name])`` and leaked ``ValueError: could
    not convert string to float: 'x'`` verbatim (``call``'s own
    ``except ValueError`` forwards a bare message, unlike the generic
    backstop). A list anywhere in ``params`` raised ``TypeError`` at that
    same ``float()`` call. All three reproduced against ``git show
    HEAD:src/warlock/studio/modes/clay/agent/dispatch.py`` before this fix.
    """
    declared = {param.name: param for param in op.params}
    messages = []
    for key in sorted(params):
        if key not in declared:
            messages.append(f"{field}.{key} is not a parameter of {op.name!r}.")
            continue
        try:
            value = float(params[key])
        except (TypeError, ValueError):
            messages.append(f"{field}.{key} must be a single number for op {op.name!r}.")
            continue
        if not math.isfinite(value):
            messages.append(f"{field}.{key} must be finite for op {op.name!r}.")
    if not messages:
        return None
    return fail(" ".join(messages), field=field, recovery="fix_arguments")


def _modifier_params_type_refusal(
    kind_def: Any, params: dict, field: str = "params"
) -> dict | None:
    """Every value of a modifier's own ``params`` held to what that
    modifier's kind actually declares -- the identical shape
    :func:`_op_params_type_refusal` already checks for ``clay_op``'s own
    ``params`` (a name in the kind's declared parameter list, and a single
    finite number, the only shape any ``modifiers.ModParam`` ever stores),
    so a bad value here is refused by name before it ever reaches
    ``modifiers.make``/``with_params`` -- which trusts its ``params`` dict
    to already be coerced and clamped (:mod:`.ops_modifiers`'s own module
    docstring) and would otherwise let a non-numeric string reach a bare
    ``float()`` call and raise there, uncaught, exactly the crash class
    :func:`_op_params_type_refusal`'s own docstring names for ``clay_op``.
    """
    declared = {p.name: p for p in kind_def.params}
    messages = []
    for key in sorted(params):
        if key not in declared:
            messages.append(f"{field}.{key} is not a parameter of modifier {kind_def.name!r}.")
            continue
        try:
            value = float(params[key])
        except (TypeError, ValueError):
            messages.append(
                f"{field}.{key} must be a single number for modifier {kind_def.name!r}."
            )
            continue
        if not math.isfinite(value):
            messages.append(f"{field}.{key} must be finite for modifier {kind_def.name!r}.")
    if not messages:
        return None
    return fail(" ".join(messages), field=field, recovery="fix_arguments")


def _resolve_modifier(
    obj: Any, args: dict, key: str = "modifier"
) -> tuple[int | None, dict | None]:
    """*obj*'s modifier named by ``args[key]``, or a refusal naming
    ``field=key`` -- the modifier-stack counterpart of :func:`_resolve_uid`,
    shared by every ``clay_modifier_*`` handler but ``clay_modifier_add``
    (which mints a new one rather than looking an existing one up). An
    absent id and an unknown one are two different refusals for the same
    reason :func:`_resolve_uid`'s own docstring gives: a caller that never
    passed one has nothing to fix by re-reading the document, while a
    caller naming a stale id does.
    """
    if args.get(key) is None:
        return None, fail(f"give a value for {key!r}.", field=key, recovery="fix_arguments")
    try:
        modifier_id = int(args[key])
    except (TypeError, ValueError):
        return None, fail(f"{key} must be an integer.", field=key, recovery="fix_arguments")
    if not any(m.id == modifier_id for m in obj.modifiers):
        return None, fail(
            f"This object has no modifier {modifier_id}.", field=key, recovery="read_scene"
        )
    return modifier_id, None


def _validate_query_arg(name: str, value: Any) -> tuple[Any, dict | None]:
    """One ``clay_select_by`` argument, validated against the fixed
    vocabulary ``agent_clay_schema._QUERY_ARG_SCHEMAS`` describes -- the one
    place a query argument's shape is checked before it reaches a pure
    ``clay.select`` function that has no JSON-schema knowledge of its own to
    check it with.
    """
    if name == "edge":
        if not isinstance(value, list) or len(value) != 2:
            return None, fail(
                f"{name} must be a [vertex, vertex] pair.", field=name, recovery="fix_arguments"
            )
        try:
            return [int(v) for v in value], None
        except (TypeError, ValueError):
            return None, fail(
                f"{name} must be a [vertex, vertex] pair.", field=name, recovery="fix_arguments"
            )
    if name == "face":
        try:
            return int(value), None
        except (TypeError, ValueError):
            return None, fail(
                f"{name} must be an integer.", field=name, recovery="fix_arguments"
            )
    if name == "slot":
        try:
            slot = int(value)
        except (TypeError, ValueError):
            return None, fail(
                f"{name} must be an integer.", field=name, recovery="fix_arguments"
            )
        # ``_QUERY_ARG_SCHEMAS["slot"]`` declares ``minimum: 0`` -- a palette
        # has no negative indices -- and until this line nothing here checked
        # it, so a negative slot sailed through to ``_q_material`` and matched
        # no face, a silent no-op rather than the refusal the schema promised.
        if slot < 0:
            return None, fail(
                f"{name} must be a non-negative integer.", field=name, recovery="fix_arguments"
            )
        return slot, None
    if name in ("direction", "min", "max"):
        return _validate_vec3(value, name)
    if name == "max_angle":
        # ``_QUERY_ARG_SCHEMAS["max_angle"]`` declares ``minimum: 0.0,
        # maximum: 180.0`` -- past 180 degrees off a direction nothing is
        # excluded any more -- but ``_validate_number`` alone only checks
        # finiteness, not this query's own bound.
        out, failure = _validate_number(value, name)
        if failure:
            return None, failure
        if not (0.0 <= out <= 180.0):
            return None, fail(
                f"{name} must be between 0 and 180 degrees.",
                field=name,
                recovery="fix_arguments",
            )
        return out, None
    if name == "space":
        if value not in ("world", "local"):
            return None, fail(
                "space must be 'world' or 'local'.", field="space", recovery="fix_arguments"
            )
        return value, None
    if name in ("faces", "verts"):
        # Tranche 5's "similar" queries' own seed set (``similar_area``,
        # ``similar_normal``, ``similar_material``, ``similar_sides`` for
        # ``faces``; ``similar_valence`` for ``verts``).
        # ``_QUERY_ARG_SCHEMAS[name]`` declares an array of non-negative
        # integers with ``minItems: 1`` -- identical for both names, which is
        # why they share this branch the same way ``direction``/``min``/
        # ``max`` already share :func:`_validate_vec3` above, rather than
        # two copies of the same three checks.
        #
        # **This was the gap tests/test_agent_schemas.py's discovery walk
        # found (dev/CLAY-PLAN.md tranche 5's follow-up, 2026-09-19):**
        # before this branch existed, any name this function did not
        # recognise fell through to the ``unknown query argument`` refusal
        # below -- which is *correct* for a name nothing declares, but
        # ``faces``/``edges``/``verts``/``tolerance`` are declared, in
        # ``_QUERY_ARG_SCHEMAS``, and simply had no case here yet. Every
        # constraint the schema promised for the four of them -- minItems,
        # item type, item minimum -- was a promise the handler never checked.
        if not isinstance(value, list) or not value:
            return None, fail(
                f"{name} must be a non-empty list of integers.",
                field=name,
                recovery="fix_arguments",
            )
        try:
            indices = [int(v) for v in value]
        except (TypeError, ValueError):
            return None, fail(
                f"{name} must be a non-empty list of integers.",
                field=name,
                recovery="fix_arguments",
            )
        if any(i < 0 for i in indices):
            return None, fail(
                f"{name} must be non-negative integers.", field=name, recovery="fix_arguments"
            )
        return indices, None
    if name == "edges":
        # ``similar_length``'s own seed set: ``_QUERY_ARG_SCHEMAS["edges"]``
        # declares a non-empty array of ``[vertex, vertex]`` pairs, each a
        # non-negative integer -- the same pair shape ``clay_select_elements``'s
        # own ``edges`` argument already checks in ``_h_select_elements``,
        # minus that handler's further check that the pair is a real edge of
        # *this* mesh (``adjacency(obj.mesh).edge_ids``): a query argument
        # here only owns the JSON-shape half of that contract, the way
        # ``edge``'s own branch above does not check the loop seed is a real
        # edge either -- ``select.edge_loop``/``similar_length`` both answer
        # empty for a seed that is not, rather than needing this function to
        # refuse it first.
        if not isinstance(value, list) or not value:
            return None, fail(
                f"{name} must be a non-empty list of [vertex, vertex] pairs.",
                field=name,
                recovery="fix_arguments",
            )
        pairs: list[list[int]] = []
        for pair in value:
            if not isinstance(pair, list) or len(pair) != 2:
                return None, fail(
                    f"{name} must be a list of [vertex, vertex] pairs.",
                    field=name,
                    recovery="fix_arguments",
                )
            try:
                a, b = int(pair[0]), int(pair[1])
            except (TypeError, ValueError):
                return None, fail(
                    f"{name} must be a list of [vertex, vertex] pairs.",
                    field=name,
                    recovery="fix_arguments",
                )
            if a < 0 or b < 0:
                return None, fail(
                    f"{name} must be non-negative vertex indices.",
                    field=name,
                    recovery="fix_arguments",
                )
            pairs.append([a, b])
        return pairs, None
    if name == "tolerance":
        # Every "similar" query's own band, shared vocabulary the way
        # ``max_angle`` is shared by ``normal`` alone -- ``_QUERY_ARG_
        # SCHEMAS["tolerance"]`` declares ``minimum: 0.0`` (no maximum: a
        # fraction for area/length, degrees for a normal, an integer step
        # for sides/valence -- there is no one honest ceiling across all
        # four units), so this checks only the floor ``_validate_number``
        # alone does not.
        out, failure = _validate_number(value, name)
        if failure:
            return None, failure
        if out < 0.0:
            return None, fail(
                f"{name} must be non-negative.", field=name, recovery="fix_arguments"
            )
        return out, None
    return None, fail(
        f"unknown query argument {name!r}.", field=name, recovery="fix_arguments"
    )  # pragma: no cover


def _repaint(doc: Any, uids: Iterable[int], index: int) -> None:
    """Rewrite every face of each object in *uids* to material *index*.

    The trap: ``Obj.material`` is only the default slot *new* faces are
    stamped with. What actually renders and exports is the per-face
    ``mesh.material`` array -- ``to_primitives`` groups a mesh's faces by
    ``np.unique(mesh.material)`` and looks each index up in the palette.
    Pointing only ``obj.material`` at the new slot (a lone ``set_props``)
    would leave an existing box's faces still naming their old slot, so it
    would export in the wrong colour with nothing here to say why. Repainting
    means rewriting that array and rebuilding the mesh.
    """
    for uid in uids:
        obj = doc.by_uid(uid)
        mesh = replace(obj.mesh, material=np.full(len(obj.mesh.material), index, dtype="i4"))
        doc.set_mesh(uid, mesh, keep_generator=True)
        doc.set_props(uid, material=index)


def _label_top(doc: Any, mark: int, label: str) -> None:
    """Name the step this call just pushed -- but only when it actually pushed
    one. ``doc.history.head != mark`` is the guard: relabelling
    ``doc.history.top`` when nothing was pushed since ``mark`` would rename
    whatever step was already on top -- the user's *previous* action, not
    this call's."""
    if doc.history.head == mark:
        return
    top = doc.history.top
    if top is not None:
        top.label = label


def _round(value: Any, dp: int = 4) -> Any:
    """*value* rounded to *dp* decimal places -- a float, or a list/array of
    them. ``clay_scene``'s readout is meant to be read, and a world-space
    translation computed through several matrix multiplies comes back with
    sixteen digits of float noise that carries no information an agent could
    act on."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_round(v, dp) for v in value]
    if isinstance(value, (int, float, np.floating, np.integer)):
        return round(float(value), dp)
    return value


def _sel_counts(sel: el.ElementSel) -> dict:
    """One ``ElementSel``'s size, per kind -- the shape every element-mode
    result in this fold reports instead of the indices themselves (see
    ``clay_elements`` for the one tool that hands those back, paged)."""
    return {"verts": len(sel.verts), "edges": len(sel.edges), "faces": len(sel.faces)}


def _modifier_row(m: Any, evaluation: Any) -> dict:
    """One :class:`~.modifiers.Modifier` as JSON -- the shape ``clay_scene``'s
    own ``modifiers`` list and every ``clay_modifier_*`` tool's own reply
    share, so a stack an agent just edited never needs a second call to see
    what changed. ``error`` is present only for a modifier
    :meth:`~.document.ClayDoc.evaluation` skipped -- see :mod:`.modifiers`'s
    own "skipped, not fatal" rule; a modifier disabled outright carries no
    error at all, since :func:`~.modifiers.evaluate` never even tries it.
    """
    row: dict[str, Any] = {
        "id": m.id, "kind": m.kind, "enabled": m.enabled, "params": m.as_dict(),
    }
    error = next((msg for mid, msg in evaluation.errors if mid == m.id), None)
    if error is not None:
        row["error"] = error
    return row


def _uv_facts(mesh: Any) -> dict | None:
    """The four uv measurements :data:`~.schema._object_row_output_schema`'s
    own ``uv`` block declares, off *mesh*'s own already-assigned uv -- or
    ``None`` when it has none (:mod:`.uvtools`' own ``Mesh.uv is None`` gate,
    ``_require_uv``'s reason every function here would otherwise raise for).

    Read off the **base** mesh, never the evaluated one -- see
    :func:`_scene_row`'s own docstring's tranche 6 paragraph for why: a seam
    and an unwrap are both authoring concepts about the geometry an edit
    would actually touch.

    ``overlapping_faces`` is the one measurement that can refuse
    (:class:`~.elements.OpError`, past :data:`~.uvtools.MAX_OVERLAP_TRIANGLES`/
    :data:`~.uvtools.MAX_OVERLAP_BUCKET`) -- caught here and reported ``None``
    rather than left to blow up the whole ``clay_scene`` reply. ``clay_scene``
    reads *every* visible object's row on every call it answers, so a single
    dense, uv'd mesh must not turn a read into a refusal for the rest of the
    document; a caller that wants the real answer for that one object can
    still ask ``clay_uv`` directly, whose own refusal names it.
    """
    if mesh.uv is None:
        return None
    ids = uvtools.islands(mesh)
    n_islands = int(ids.max()) + 1 if len(ids) else 0
    stretch_vals = uvtools.stretch(mesh)
    mean_stretch = float(stretch_vals.mean()) if len(stretch_vals) else 0.0
    try:
        overlapping = int(uvtools.overlap_faces(mesh).sum())
    except OpError:
        overlapping = None
    return {
        "islands": n_islands,
        "overlapping_faces": overlapping,
        "mean_stretch": _round(mean_stretch),
        "texel_density": _round(uvtools.texel_density(mesh)),
    }


def _scene_row(doc: Any, obj: Any) -> dict:
    """Everything ``clay_scene`` says about one object -- and everything
    ``clay_add_primitive``/``clay_add_figure`` hand back too, so an agent that
    just placed something never needs a second call to learn where it landed.

    ``bbox``/``size``/``center`` are measured off *this object's evaluated
    mesh* (``doc.evaluation(uid)``, cheap -- an object with no enabled
    modifiers evaluates to its own base mesh, ``is``-identical, see
    :mod:`.modifiers`'s own module docstring), never the base alone: a
    mirror or an array modifier changes what is actually on screen, and a
    box measured only around half of it would contradict what
    ``clay_render`` draws. ``faces``/``verts`` stay the *base* mesh's own
    counts on purpose -- an element edit (and every element-only ``clay_op``
    row) acts on the base, so those two numbers describe what editing this
    object would actually change, not what a modifier stack turns it into.
    ``modifiers``/``evaluated`` are added only once the stack is non-empty --
    see :func:`~.schema._object_row_output_schema`'s own docstring for the
    exact shape.

    **Tranche 6/7: ``role``/``collider_kind`` and ``uv``.** ``role``/
    ``collider_kind`` are :class:`~.document.Obj`'s own two fields, always
    present. ``uv`` is added only when the *base* mesh carries texture
    coordinates -- the base, not ``evaluation.mesh``, because a seam and an
    unwrap are both authoring concepts about the geometry an element edit
    would touch, the identical reasoning ``faces``/``verts`` already follow;
    see :func:`_uv_facts` for the four measurements and why one of them can
    read ``null``.
    """
    world = doc.world_matrix(obj.uid)
    evaluation = doc.evaluation(obj.uid)
    box = clay_geom_ops.world_box(obj, evaluation.mesh, world=world)
    # Tranche 3: scene structure. A root's own world matrix *is* its local
    # TRS (document.py's own invariant), so reading the object's own fields
    # directly here -- rather than decomposing `world` back apart -- is not a
    # shortcut, it is the exact same number with none of a matrix round
    # trip's float noise, which is what keeps every pre-tranche-3 document's
    # clay_scene reply bit-identical to what it always reported. Only a
    # parented object's world TRS genuinely differs from its own fields, so
    # only there is `world` actually decomposed.
    if obj.parent is None:
        world_t, world_q, world_s = obj.translation, obj.rotation, obj.scale
    else:
        world_t, world_q, world_s = m3.decompose(world)
    wrx, wry, wrz = _euler_xyz_from_quat(world_q)
    size = center = None
    if box is not None:
        lo, hi = box
        size = _round(hi - lo)
        center = _round((lo + hi) * 0.5)
    row: dict[str, Any] = {
        "uid": obj.uid,
        "name": obj.name,
        "visible": obj.visible,
        "parent": obj.parent,
        "locked": obj.locked,
        "tags": list(obj.tags),
        "generator": obj.generator,
        "params": obj.params,
        "faces": bm.face_count(obj.mesh),
        "material": obj.material,
        "bbox": None if box is None else {"min": box[0].tolist(), "max": box[1].tolist()},
        "translation": _round(world_t),
        "rotation": _round([wrx, wry, wrz]),
        "scale": _round(world_s),
        "size": size,
        "center": center,
        "verts": len(obj.mesh.positions),
        # Purely additive -- see the module's element-mode paragraph. An
        # agent that has just switched mode or selected something does not
        # need a second call to learn what came across on this object.
        "stamp": doc.mesh_stamp(obj.uid),
        "selected": _sel_counts(doc.element_sel_of(obj.uid)),
        "role": obj.role,
        "collider_kind": obj.collider_kind,
    }
    uv_facts = _uv_facts(obj.mesh)
    if uv_facts is not None:
        row["uv"] = uv_facts
    if obj.parent is not None:
        lrx, lry, lrz = _euler_xyz_from_quat(obj.rotation)
        row["local"] = {
            "translation": _round(obj.translation),
            "rotation": _round([lrx, lry, lrz]),
            "scale": _round(obj.scale),
        }
    if obj.modifiers:
        row["modifiers"] = [_modifier_row(m, evaluation) for m in obj.modifiers]
        row["evaluated"] = {
            "vertices": len(evaluation.mesh.positions),
            "faces": bm.face_count(evaluation.mesh),
            "triangles": clay_ops_modifiers._triangle_count(evaluation.mesh),
        }
    return row


_OBJECT_SELECTION_DERIVED_REFUSAL = (
    "The object selection is derived from the element selection in "
    "vertex/edge/face mode. Call clay_element_mode with mode='object' first."
)
"""What ``clay_select`` and ``clay_boolean`` say in an element mode, verbatim
in both -- see each handler's own comment for why they refuse rather than
guess, and ``agent_clay_tools._h_delete``'s docstring for why it is
deliberately not a third."""
