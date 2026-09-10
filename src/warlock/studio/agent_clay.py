"""The tool surface an MCP agent drives Clay through.

**The tool list is derived, never hand-written.** Every schema in :func:`tools`
is built from a registry that already exists for a human surface --
``primitives.GENERATORS`` for what a primitive is and what it defaults to,
``presets.ASSEMBLIES`` for which figures exist, and ``clay_ops.OPS`` for the
whole of what an object or an element can be told to do. A thirteenth
primitive, a ninth figure or a new op in the registry needs no edit here: it
shows up in the next ``tools()`` call because the source it is drawn from
changed, which is the same property ``clay_ops.menu`` already gives the
context menu, the tools pane and the key handler -- one list, so nothing here
can drift out of step with what Clay can actually do. ``clay_batch``'s own
name enum is derived the same way, from ``_HANDLERS`` minus ``BATCH_EXCLUDED``
-- see that constant for which tools are left out and why.

**An agent reaches exactly one document, and never by falling back to
whichever tab the user has open.** ``Session.tab_uid`` names the one
:class:`~.clay_state.ClayTab` this session owns; every tool resolves it fresh,
every call, through ``clay_mode.ensure(ctx).get(session.tab_uid)`` -- read
*through* the same ``ClayState`` the interactive UI uses rather than a
snapshot taken once, so a document closed from the keyboard mid-session is
seen as gone on the very next call. A missing tab is a refusal, never a
substitution: an agent with no document of its own must never be handed the
user's, because that is the one way a scripted client could edit, export or
close something the person at the keyboard never offered it. The empty
default (``tab_uid == ""``) means "this session owns nothing yet," and only
the two tools that can start a document from nothing (:func:`clay_add_primitive`,
:func:`clay_add_figure`) are allowed to mint one and adopt it into the session
-- ``clay_batch`` is a documented third way in, but only because its first
call is one of those two; see its own docstring.

**Rendering owns a private viewport.** ``ClayView`` is a real GL object --
buffers, gizmos, a camera -- and the one already on screen
(``ctx.clay_view``) belongs to whatever tab the user is looking at. Reusing
it for ``clay_render`` would mean every render this module produces first
yanks the user's camera onto the agent's document and back, which is a visible
stutter in the middle of whatever the person is doing. This module builds and
keeps its own :class:`~.clay_view.ClayView`, off the moderngl context the app
already has (``ctx.viewer.ctx``) but with no ``app_ctx`` of its own, so it
never reads or writes ``ctx.clay_view`` at all. :func:`release` is the
matching teardown, called by ``agent_host`` at **teardown**, not when the
bridge disconnects -- a reconnect is routine, and releasing on disconnect
would either run GL work off the frame thread or race teardown's own
ordered stop-then-release; a viewport torn down and rebuilt per connection
would be churn bought for nothing.

**``clay_op`` is handed a sandboxed proxy, never the real app ``ctx``.**
Handing over the real one used to mean an op an agent ran through this
escape hatch could, like the same op fired from the keyboard, move the
viewport the user is looking through mid-gesture (Frame Selection chief
among them) -- the argument for doing so was that every op in the registry
is already written against the real shape (``ctx.toast``, ``ctx.clay_view``,
``ctx.state``), and reimplementing the registry against a second ``ctx``
shape it was never written for looked like the wrong trade. That argument
does not survive contact with a running agent: ``clay_ops`` reaches ``ctx``
in exactly three places (``toast``, ``getattr(ctx, "clay_view", None)`` in
``_frame``, ``getattr(ctx, "state", None)`` in ``_forget_manifold``), which
is few enough to sandbox properly rather than hand over wholesale. See
:class:`_OpCtx`: the absent ``clay_view`` makes Frame Selection the no-op it
should always have been for an agent with no viewport of its own, ``state``
passes through for real because the manifold-cache pop is real work that
still has to happen, and every ``toast`` lands in the result instead of the
running app -- so a per-object refusal inside ``run`` that used to become a
toast the user saw and the agent never did now comes back as a message the
agent can actually read.

**``clay_batch`` folds several tool calls into one undo step.** An agent
that wants to block out a scene one primitive at a time pays one round trip
per primitive and, worse, one Ctrl+Z per primitive for a user who wants to
back the whole attempt out; ``clay_batch`` runs up to ``BATCH_MAX`` calls
through :func:`call` under one ``history.mark()``/``collapse_since`` pair,
stopping at the first refusal and keeping the successful prefix. It is
itself the documented exception that makes "one tool call is one undo step"
true rather than approximately true.

**A call that outruns ``agent_host.CALL_TIMEOUT`` still completes.** The
timeout lives on the listener thread, which gives up waiting and answers
"no answer in time" -- but the job it queued is still sitting on the frame
thread's queue and :meth:`~.agent_host.AgentHost.pump` will run it exactly
once, on schedule, whether or not anyone is still waiting for the result.
An agent that sees a timeout must not retry blindly: the right recovery is
to re-read ``clay_scene`` and see what actually happened, because "no
answer" and "nothing happened" are not the same claim.

**``clay_render``'s payload is bounded before the GPU work, not after.**
``RENDER_PIXEL_BUDGET`` refuses a request for too many total pixels across
its views before a single frame is drawn, and the base64-encoded result is
checked again against ``protocol.MAX_FRAME`` (less ``RENDER_FRAME_RESERVE``)
before it is returned -- both refusals name the ceiling and suggest asking
for fewer or smaller views, because the alternative is minutes of rendering
spent on a reply nothing on the other end of the pipe could ever receive.

**References live on the session, never on the document.** ``clay_reference_add``
keeps a picture in memory on :class:`Session`, in memory only -- never in the
``ClayDoc`` and never in the ``.wblk``, because a :class:`~.clay_state.ClayTab`
outlives the session that opened it and putting pictures in it would drag in
journal and serialisation questions the format's VERSION 2 has no answer for.
Adding one pushes no undo step either, the second documented exception to
"one tool call is one undo step": nothing in the document changed.

**Two intentional departures from calling ``clay_mode.save_to`` and
``clay_mode.export_asset`` by name**, both because those functions hand their
result to whichever tab is on screen when a task finishes, and an MCP
``tools/call`` has no such moment: it returns once, from this frame's
``call()``, and the job id the contract asks for has to be in hand by then.

* ``save_to`` submits its encode to a task thread and, on completion,
  retitles the tab and repoints ``tab.path`` at wherever it was told to
  write -- exactly right for a user's own Ctrl+S, and exactly wrong here: an
  agent-chosen path must never become what the user's *next* Ctrl+S silently
  overwrites. :func:`clay_export` never calls it at all, so ``tab.path``,
  ``tab.title`` and ``tab.saving`` are left exactly as they were. The
  authored document survives anyway: :func:`clay_mode.build_asset` already
  writes it as the job's own ``build.wblk`` sidecar (via ``save_clay_source``),
  which is what ``clay_mode.edit_asset_in_clay`` reopens from the Library --
  there is no second copy for this module to keep of its own.
* ``export_asset`` submits its own encode the same way and only knows the new
  job's id once that task finishes and calls back into ``clay_mode.
  on_task_done`` -- there is no id to return from this call if this module
  goes through it as written. :func:`clay_export` instead calls
  ``clay_mode.build_asset`` -- the same document -> model-row chain
  ``export_asset`` itself now calls from its own task thread -- directly and
  synchronously, so the id is in hand before ``call()`` returns.

Both run once, off the interactive 60 fps loop, for a deliberate one-shot
action rather than every frame -- the stall a synchronous encode would be if
it ran on every draw is not what is happening here.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..service import files as svc_files
from ..service import validation as svc_validation
from ..service.errors import NotFound, ServiceError
from . import clay_mode, clay_ops
from .clay import diagnose as clay_diagnose
from .clay import mesh as bm
from .clay import ops as clay_geom_ops
from .clay import ops_boolean, presets, shading
from .clay import primitives as bp
from .clay.elements import OpError
from .clay_view import ClayView
from .panes import clay_tools as pane_clay_tools
from .viewer import gltf
from .viewer import math3d as m3
from .viewer.camera import Camera

log = logging.getLogger(__name__)


# --- module constants ---------------------------------------------------------

BATCH_MAX = 32
"""The most tool calls one ``clay_batch`` request may fold into a single undo
step. The byte budget (``UndoStack``) already bounds what a single step can
*cost*, but nothing bounded how many calls a batch could ask for before this
-- an unbounded batch would let one MCP round trip queue an arbitrarily long
run with no natural place to hit a ceiling first."""

BATCH_EXCLUDED = frozenset(
    {
        "clay_batch",  # nesting buys nothing and bounds nothing
        # An image block is a result a client must see as one; a batch can
        # only hand back JSON text, so neither of these has a shape a batch
        # result could carry.
        "clay_render",
        "clay_reference_get",
        "clay_export",  # a deliberate one-shot that mints a Library row
        # A batch is itself one undo step; moving the history head from
        # inside the very run that is about to fold into one is incoherent.
        "clay_undo",
        "clay_redo",
    }
)
"""Tools ``clay_batch`` refuses to run -- see :func:`_h_batch`'s docstring
for the derivation, and the comments above for why each one is excluded."""

MAX_REFERENCES = 8
"""How many pictures one session may hold at once. A session's references
live in memory for the session's whole life (:attr:`Session.references`) and
nothing ever evicts one on its own, so a ceiling is what keeps a client that
forgets ``clay_reference_remove`` from growing an unbounded set of decoded
PNGs behind a session nobody is watching."""

RENDER_PIXEL_BUDGET = 6 * 1024 * 1024
"""The most total pixels one ``clay_render`` call may ask for, summed across
every view it requests. A caller asking for the maximum -- a dozen views at
2048x2048 -- would otherwise spend real GPU minutes producing a reply that
was refused for size the moment it tried to leave; this stops that before
the first frame is drawn rather than after."""

RENDER_FRAME_RESERVE = 64 * 1024
"""Headroom subtracted from ``protocol.MAX_FRAME`` when checking whether a
render's base64-encoded payload will fit in one reply frame. The JSON
envelope around the image blocks costs bytes of its own; without this an
encode that is over by a few hundred bytes would reach ``send_bytes`` and
fail there, past the point a refusal could explain itself."""


@dataclass
class Session:
    """One MCP connection's claim on Clay. See the module docstring."""

    tab_uid: str = ""
    references: dict[str, Any] = field(default_factory=dict)
    """Pictures this session has been handed to match against, by name --
    values are ``agent_refs.Reference``. In memory on the session only; see
    the module docstring's references paragraph for why never on the
    document."""


# The private viewport :func:`_view_for` builds -- module-level rather than
# per-session because a second concurrent session cannot exist yet (see
# ``mcp/pipe.py``'s v1 decision, one connection at a time), so there is
# nothing for a per-session instance to isolate that a shared one does not
# already give for free.
_view: ClayView | None = None


def _view_for(ctx: Any) -> ClayView:
    """This module's own viewport, built the first time a render is asked for.

    ``ctx.viewer`` is the app's main 3D pane and always exists once the app has
    a window, which is what makes it the one place to borrow a moderngl
    context from without reaching for the interactive Clay viewport itself.
    ``app_ctx=None`` is what keeps :class:`ClayView` from reading
    ``ctx.clay_view`` or the shared tool setting through ``self.state`` --
    see the module docstring's rendering claim.
    """
    global _view
    if _view is None:
        _view = ClayView(ctx.viewer.ctx)
    return _view


def release() -> None:
    """Free the GL objects this module opened. Called by ``agent_host`` at
    teardown -- a pipe that is never reopened this session should not hold a
    viewport's worth of buffers and gizmos alive for the rest of the app's
    life."""
    global _view
    if _view is not None:
        _view.release()
        _view = None


# --- resolving the session's document ---------------------------------------


def _tab(ctx: Any, session: Session, *, create: bool = False) -> tuple[Any, dict | None]:
    """The session's own tab, or a failure result to return unchanged.

    ``create`` is only ever passed by the two tools that can act on an empty
    session -- adding the first primitive or figure -- so every other tool
    refuses outright rather than silently starting a document nobody asked
    for. Resolved through ``ClayState`` on every call, never cached on the
    session, so a tab the user closed from the keyboard is seen as gone on the
    very next tool call rather than on whichever call happens to notice.
    """
    state = clay_mode.ensure(ctx)
    if session.tab_uid:
        tab = state.get(session.tab_uid)
        if tab is not None:
            return tab, None
        return None, fail(
            "This session's document was closed. Call clay_add_primitive or "
            "clay_add_figure to start a new one."
        )
    if not create:
        return None, fail(
            "This session has no document yet. Call clay_add_primitive or "
            "clay_add_figure first."
        )
    tab = clay_mode.new_document(ctx)
    session.tab_uid = tab.uid
    return tab, None


# --- protocol glue ------------------------------------------------------------
#
# Imported lazily inside functions rather than at module scope: ``warlock.mcp``
# is built alongside this module by a different pass over the same plan (see
# ``CONTRACT.md``), and importing it at module scope would make every other
# caller of ``studio.agent_clay`` -- ``panes.clay_tools`` tests among them --
# fail at collection time on however far that sibling module has got.


def _protocol() -> Any:
    from ..mcp import protocol

    return protocol


def ok(*content: dict) -> dict:
    return _protocol().ok(*content)


def fail(message: str, **extra: Any) -> dict:
    return _protocol().fail(message, **extra)


def text(s: str) -> dict:
    return _protocol().text(s)


def image_png(data: bytes) -> dict:
    return _protocol().image_png(data)


def _json(payload: Any) -> dict:
    return ok(text(json.dumps(payload)))


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
    for the one tool that takes them.

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
    """
    try:
        uid = int(args[key])
        obj = doc.by_uid(uid)
    except (KeyError, ValueError, TypeError):
        return None, fail(f"no object with uid {args.get(key)!r}.", field=key)
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
        return None, fail(f"{field} must be a list of integers.", field=field)
    known = {obj.uid for obj in doc.objects}
    missing = [u for u in uids if u not in known]
    if missing:
        return None, fail(f"no object with uid(s) {missing}.", field=field)
    return uids, None


def _validate_vec3(value: Any, field: str) -> tuple[list[float] | None, dict | None]:
    """Three finite numbers, or a refusal naming *field*.

    Shared by every optional TRS vector ``clay_add_primitive`` and
    ``clay_add_figure`` take, so a malformed one is caught before anything is
    placed -- see those tools' "validate everything before the first
    mutation" rule.
    """
    if not isinstance(value, list) or len(value) != 3:
        return None, fail(f"{field} must be an array of 3 numbers.", field=field)
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError):
        return None, fail(f"{field} must be an array of 3 numbers.", field=field)
    if not all(math.isfinite(v) for v in out):
        return None, fail(f"{field} must be finite numbers.", field=field)
    return out, None


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


def _scene_row(doc: Any, obj: Any) -> dict:
    """Everything ``clay_scene`` says about one object -- and everything
    ``clay_add_primitive``/``clay_add_figure`` hand back too, so an agent that
    just placed something never needs a second call to learn where it landed.
    """
    box = clay_geom_ops.world_box(obj)
    rx, ry, rz = _euler_xyz_from_quat(obj.rotation)
    size = center = None
    if box is not None:
        lo, hi = box
        size = _round(hi - lo)
        center = _round((lo + hi) * 0.5)
    return {
        "uid": obj.uid,
        "name": obj.name,
        "visible": obj.visible,
        "generator": obj.generator,
        "params": obj.params,
        "faces": bm.face_count(obj.mesh),
        "material": obj.material,
        "bbox": None if box is None else {"min": box[0].tolist(), "max": box[1].tolist()},
        "translation": _round(obj.translation),
        "rotation": _round([rx, ry, rz]),
        "scale": _round(obj.scale),
        "size": size,
        "center": center,
        "verts": len(obj.mesh.positions),
    }


# --- the tools -----------------------------------------------------------------


def instructions() -> str:
    """The prose ``agent_host`` hands ``protocol.dispatch`` for ``initialize``.

    Read once, by whatever model is driving the bridge, before its first tool
    call -- so this is where the conventions no single schema field can carry
    live: which units, which axis is up, that a rotation is always degrees and
    never a quaternion, that undo is call-scoped with two named exceptions,
    and the working loop an agent that skips straight to numbers tends to
    get wrong. A function rather than a module constant, so ``BATCH_MAX`` and
    the live generator catalogue are embedded fresh rather than duplicated --
    the same reason ``tools()`` itself is rebuilt every time it is asked for.
    """
    return (
        "Warlock's Clay, over MCP. Units are metres; the axes are glTF's -- "
        "Y is up, Z is toward the viewer, and the ground is y=0. Every "
        "generator is centred on its own origin, so a box of height h "
        "stands on the ground at translation=[0, h/2, 0]; a figure placed "
        "with clay_add_figure arrives already grounded. Rotations are "
        "always Euler XYZ in degrees, never a quaternion.\n\n"
        "uids are the only addresses -- clay_scene reports one for every "
        "object, and every other tool that names an object takes one. "
        "Names are for humans and may be renamed (clay_rename); a uid never "
        "changes.\n\n"
        "One tool call is one undo step, with exactly two documented "
        "exceptions: clay_batch folds its whole run into one step, and "
        "clay_undo/clay_redo move the history head rather than pushing a "
        "step of their own. Adding a reference (clay_reference_add) pushes "
        "nothing either, because nothing in the document changed -- "
        "references live on this session, never in the document.\n\n"
        "The working loop that avoids building something plausible in "
        "numbers and wrong on screen: block out with primitives, "
        "clay_render from three_quarter and front, adjust, boolean, "
        "clay_diagnose, then clay_export. A boolean needs closed solids, "
        "and its survivor is whichever object comes first in the "
        "document's own order, never first in the uids list handed to "
        "it.\n\n"
        "Materials are linear RGB, 0..1. clay_scene's 'materials' lists the "
        "palette already in use -- reuse an index from it rather than "
        "appending a near-duplicate.\n\n"
        "References: clay_reference_add takes a Library job id or inline "
        "base64; pass 'compare' to clay_render to see the current render "
        "beside the reference or blended over it.\n\n"
        f"Up to {BATCH_MAX} tool calls can be folded into one clay_batch "
        "call; it stops at the first refusal and keeps everything that "
        "already ran.\n\n"
        "Known generators: " + _generator_catalog()
    )


def tools() -> list[Any]:
    """Every tool Clay's agent surface offers, built fresh from the registries
    named in the module docstring. Called once per ``tools/list`` request, so
    rebuilding it from ``GENERATORS``/``ASSEMBLIES``/``OPS``/``_HANDLERS`` each
    time costs nothing and can never go stale against an edit to any of
    them."""

    protocol = _protocol()
    primitive_names = sorted(bp.GENERATORS)
    figure_keys = sorted(presets.ASSEMBLIES)
    op_names = [op.name for op in clay_ops.OPS]
    axis_views = sorted(Camera.AXIS_VIEWS) + ["three_quarter"]
    batch_names = sorted(set(_HANDLERS) - BATCH_EXCLUDED)

    return [
        protocol.Tool(
            name="clay_scene",
            title="Describe the scene",
            description=(
                "Every object in this session's document -- its generator and "
                "parameters (or its shape once an edit has frozen them, see "
                "'params'), its world-space translation/rotation/scale, its "
                "bounding box (and the box's own size and center), its face "
                "and vertex counts and its material slot -- plus the "
                "document's own bounds over its visible objects, its "
                "material palette, its current selection, element mode and "
                "whether it has unsaved changes."
            ),
            schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        protocol.Tool(
            name="clay_add_primitive",
            title="Add a primitive",
            description=(
                "Place one primitive, selected, as one undo step. Starts "
                "this session's document if it has none yet. 'generator' "
                "picks the shape; 'params' overrides its own numbers "
                "(radius, segments and so on); 'translation'/'rotation'/"
                "'scale' place it directly rather than at the origin; "
                "'name' sets what it is called, refused if another object "
                "already wears it; 'material' paints it with an existing "
                "palette index (see clay_scene's 'materials') rather than "
                "the default grey -- appending a new material is "
                "clay_material's job, not this one's. Everything is "
                "validated before anything is placed, so a refused call "
                "places nothing. Returns the same row clay_scene would show "
                "for it. Known generators: " + _generator_catalog()
            ),
            schema={
                "type": "object",
                "properties": {
                    "generator": {"type": "string", "enum": primitive_names},
                    "params": {
                        "type": "object",
                        "additionalProperties": {
                            "anyOf": [
                                {"type": "number"},
                                {"type": "array", "items": {"type": "number"}},
                            ]
                        },
                    },
                    "translation": _vec3_schema("metres"),
                    "rotation": _vec3_schema("degrees, Euler XYZ"),
                    "scale": _vec3_schema("a multiplier per axis"),
                    "name": {"type": "string"},
                    "material": {"type": "integer", "minimum": 0},
                },
                "required": ["generator"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_add_figure",
            title="Add a figure",
            description=(
                "Place every part of a rigged figure preset -- a humanoid, a "
                "quadruped, a bird and so on -- as one grounded group, as "
                "one undo step. Starts this session's document if it has "
                "none yet. 'translation' offsets the whole group; 'yaw' "
                "turns it about Y, in degrees, around the group's own "
                "origin -- it rotates where the parts sit, not each part in "
                "place; 'scale' is one uniform number, since a figure is a "
                "proportioned thing and a per-axis scale is how you get a "
                "squashed head; 'name_prefix' is prepended to every part's "
                "name, refused if it would collide with an object already "
                "in the document."
            ),
            schema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "enum": figure_keys},
                    "translation": _vec3_schema("metres"),
                    "yaw": {"type": "number", "description": "degrees, about Y"},
                    "scale": {"type": "number", "exclusiveMinimum": 0.0},
                    "name_prefix": {"type": "string"},
                },
                "required": ["key"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_transform",
            title="Move, rotate or scale an object",
            description=(
                "Set one object's translation, rotation and/or scale, as one "
                "undo step. Rotation is Euler X, then Y, then Z, in degrees -- "
                "not a quaternion."
            ),
            schema={
                "type": "object",
                "properties": {
                    "uid": {"type": "integer"},
                    "translation": _vec3_schema("metres"),
                    "rotation": _vec3_schema("degrees, Euler XYZ"),
                    "scale": _vec3_schema("a multiplier per axis"),
                },
                "required": ["uid"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_set_params",
            title="Set a primitive's own parameters",
            description=(
                "Change how tall a cylinder is, how many segments it has, how "
                "thick a torus's tube is, or a column's base and capital -- "
                "the numbers a generator was built from, which scale alone "
                "cannot reach. Merges over the object's current params, then "
                "rebuilds the mesh, as one undo step. Only for an object whose "
                "generator is still set (clay_scene's 'generator' is not "
                "null) -- once an edit has frozen its topology there are no "
                "generator params left to set. Values may be a number or an "
                "array -- box's size is (x, y, z), plane's is (w, h). Known "
                "generators: " + _generator_catalog()
            ),
            schema={
                "type": "object",
                "properties": {
                    "uid": {"type": "integer"},
                    "params": {
                        "type": "object",
                        "additionalProperties": {
                            "anyOf": [
                                {"type": "number"},
                                {"type": "array", "items": {"type": "number"}},
                            ]
                        },
                        "description": "Only the keys to change; every other one keeps its value.",
                    },
                },
                "required": ["uid", "params"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_material",
            title="Give objects a material",
            description=(
                "Append one palette entry and paint every named object's "
                "whole surface with it, as one undo step -- 'a wooden barrel "
                "with iron bands' is two calls, one material each. Repaints "
                "every existing face, not just the object's default slot for "
                "future ones, so an already-built object comes out the "
                "colour asked for. Color is linear 0..1 RGB or RGBA; a "
                "3-element color exports fully opaque. Metallic and "
                "roughness default to a plain painted dielectric."
            ),
            schema={
                "type": "object",
                "properties": {
                    "uids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                    },
                    "name": {"type": "string"},
                    "color": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "minItems": 3,
                        "maxItems": 4,
                    },
                    "metallic": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "roughness": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["uids", "color"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_boolean",
            title="Boolean two or more objects",
            description=(
                "Union, subtract or intersect the given objects into the one "
                "that comes first in the document's own object order -- not "
                "the order given here, which only says which objects take "
                "part. A closed-solid requirement applies to all three; the "
                "refusal names which object is not one."
            ),
            schema={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(ops_boolean.KINDS)},
                    "uids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                    },
                },
                "required": ["kind", "uids"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_select",
            title="Set the object selection",
            description="Replace the document's object selection. An empty list clears it.",
            schema={
                "type": "object",
                "properties": {"uids": {"type": "array", "items": {"type": "integer"}}},
                "required": ["uids"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_op",
            title="Run a Clay op",
            description=(
                "The escape hatch over every op Clay can run, by name -- the "
                "same registry the context menu, the tools pane and the "
                "keyboard read. Runs sandboxed: it never moves the user's own "
                "viewport (Frame Selection is a no-op here) and any refusal "
                "an op raises per object comes back as a 'messages' list in "
                "the result rather than a toast only the person at the "
                "keyboard would see. Known ops: " + _op_catalog(op_names)
            ),
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": op_names},
                    "params": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": (
                            "Only for a parameterised op; missing fields fall "
                            "back to that op's own defaults."
                        ),
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_render",
            title="Render the document",
            description=(
                "One or more flat-shaded, white-background square renders "
                "of the document -- no gizmos -- from the standard "
                "three-quarter framing, a named axis view, or a free "
                "yaw/pitch pair. 'view' and 'views' are exclusive; giving "
                "both is refused. 'grid' draws the ground plane at y=0, the "
                "one scale cue available with no viewport to walk around in: "
                "16 cells across a span rounded up to a power of ten "
                "containing 2.5x the framed footprint, so one cell reads as "
                "span/16 metres. 'focus' points the camera at the union of "
                "the named objects' boxes -- everything else is still "
                "drawn, since the renderer has no per-object alpha. Refused, "
                "before any GPU work, when the requested views would exceed "
                f"the {RENDER_PIXEL_BUDGET:,}-pixel render budget or would "
                "not fit in one reply frame once encoded -- ask for fewer or "
                "smaller views instead. Pass 'compare' (a stored reference's "
                "name) to render exactly one view beside it or blended over "
                "it instead of the normal multi-view result -- see "
                "clay_reference_add."
            ),
            schema={
                "type": "object",
                "properties": {
                    "size": {"type": "integer", "minimum": 64, "maximum": 2048},
                    "view": {"type": "string", "enum": axis_views},
                    "views": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "anyOf": [
                                {"type": "string", "enum": axis_views},
                                {
                                    "type": "object",
                                    "properties": {
                                        "yaw": {"type": "number"},
                                        "pitch": {
                                            "type": "number",
                                            "minimum": -89.0,
                                            "maximum": 89.0,
                                        },
                                    },
                                    "required": ["yaw", "pitch"],
                                    "additionalProperties": False,
                                },
                            ]
                        },
                    },
                    "grid": {"type": "boolean"},
                    "focus": {"type": "array", "items": {"type": "integer"}},
                    "compare": {"type": "string"},
                    "compare_mode": {"type": "string", "enum": ["beside", "overlay"]},
                    "alpha": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_diagnose",
            title="Check a mesh for defects",
            description=(
                "Holes, non-manifold or flipped edges, duplicate faces and "
                "unused vertices -- for one object, or every visible object "
                "when none is named."
            ),
            schema={
                "type": "object",
                "properties": {"uid": {"type": "integer"}},
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_export",
            title="Export the document as an asset",
            description=(
                "Mint a finished model row from the document, the way Clay's "
                "own Export does, and also keep the authored document as a "
                ".wblk this session can be resumed from. Returns the new "
                "job's id."
            ),
            schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        protocol.Tool(
            name="clay_undo",
            title="Undo",
            description=(
                "Reverse up to 'steps' tool calls (default 1), moving the "
                "history head backwards rather than pushing a step of its "
                "own -- the documented exception to 'one call is one undo "
                "step'. Over-asking moves as far as it can rather than "
                "refusing; 'moved' in the result is a count of steps "
                "actually reversed, not a flag."
            ),
            schema={
                "type": "object",
                "properties": {"steps": {"type": "integer", "minimum": 1, "maximum": 64}},
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_redo",
            title="Redo",
            description=(
                "Replay up to 'steps' undone tool calls (default 1), moving "
                "the history head forwards -- clay_undo's own counterpart "
                "and the same documented exception to 'one call is one undo "
                "step'. Over-asking moves as far as it can rather than "
                "refusing; 'moved' is a count of steps actually replayed."
            ),
            schema={
                "type": "object",
                "properties": {"steps": {"type": "integer", "minimum": 1, "maximum": 64}},
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_delete",
            title="Delete objects",
            description=(
                "Remove the named objects, in any element mode, as one undo "
                "step -- never a wrapper over clay_op's own Delete row, "
                "which acts on the selection and in a face or edge mode can "
                "leave every object standing."
            ),
            schema={
                "type": "object",
                "properties": {
                    "uids": {"type": "array", "items": {"type": "integer"}, "minItems": 1}
                },
                "required": ["uids"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_rename",
            title="Rename an object",
            description=(
                "Give one object a new name. Refused if the name is empty "
                "or already worn by another object."
            ),
            schema={
                "type": "object",
                "properties": {"uid": {"type": "integer"}, "name": {"type": "string"}},
                "required": ["uid", "name"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_batch",
            title="Run several tools as one undo step",
            description=(
                "Run up to "
                f"{BATCH_MAX} tool calls in order, folded into a single undo "
                "step, stopping at the first refusal and keeping the "
                "successful prefix. If this session owns no document yet, "
                "the first call must be clay_add_primitive or "
                "clay_add_figure. clay_batch, clay_render, clay_export, "
                "clay_undo, clay_redo and clay_reference_get cannot be "
                "batched -- see their own tools for why."
            ),
            schema={
                "type": "object",
                "properties": {
                    "calls": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": BATCH_MAX,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "enum": batch_names},
                                "arguments": {"type": "object"},
                            },
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["calls"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_reference_add",
            title="Add a reference image",
            description=(
                "Hand this session a picture to match against -- from a "
                "Library job's own image, or inline as base64 -- kept in "
                "memory on the session only, never in the document and "
                f"never saved to disk. Up to {MAX_REFERENCES} at a time; "
                "re-adding an existing name replaces it without spending a "
                "slot. Pushes no undo step: nothing in the document "
                "changed."
            ),
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "job_id": {"type": "string"},
                    "file": {
                        "type": "string",
                        "description": (
                            "One of input.png/ref.png/reference.png/thumb.png, "
                            "with job_id. Omit to use the first one ready."
                        ),
                    },
                    "png_base64": {"type": "string"},
                    "view": {
                        "type": "string",
                        "enum": sorted(set(Camera.AXIS_VIEWS) | {"three_quarter", "other"}),
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_reference_list",
            title="List this session's references",
            description="Every reference this session holds, by name.",
            schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        protocol.Tool(
            name="clay_reference_get",
            title="Fetch one reference image",
            description="One stored reference's metadata and its picture, by name.",
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_reference_remove",
            title="Forget a reference image",
            description="Drop one stored reference, freeing its slot.",
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
    ]


def _vec3_schema(unit: str) -> dict:
    """The one shape every optional TRS vector in this module shares."""
    return {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
        "description": unit,
    }


def _op_catalog(names: list[str]) -> str:
    """Every op's params and bounds, folded into one sentence.

    Built from :data:`clay_ops.OPS` rather than written out, so a thirteenth
    op -- with or without parameters -- appears here the next time ``tools()``
    is called and nowhere needs editing for it to.
    """
    parts = []
    for op in clay_ops.OPS:
        modes = "/".join(op.modes)
        if op.params:
            fields = ", ".join(
                f"{p.name} ({p.low}-{p.high}, default {p.default})" for p in op.params
            )
            parts.append(f"{op.name} [{modes}: {fields}]")
        else:
            parts.append(f"{op.name} [{modes}]")
    del names
    return "; ".join(parts)


def _generator_catalog() -> str:
    """Every generator's own keys and their defaults, folded into one sentence.

    Built from :data:`primitives.GENERATORS` rather than written out, so a
    thirteenth primitive's parameters appear here the next time ``tools()`` is
    called and nowhere needs editing for it to -- the same rule
    :func:`_op_catalog` already follows for ``clay_op``.
    """
    parts = []
    for name in sorted(bp.GENERATORS):
        defaults = bp.GENERATORS[name][0]
        fields = ", ".join(f"{key}={value!r}" for key, value in defaults.items())
        parts.append(f"{name} [{fields}]")
    return "; ".join(parts)


# --- dispatch -----------------------------------------------------------------


def call(ctx: Any, session: Session, name: str, arguments: dict) -> dict:
    """Run one tool. Never raises -- see the module docstring's safety claim
    and the class of error each of these three turns into a refusal for."""
    args = arguments or {}
    handler = _HANDLERS.get(name)
    if handler is None:
        return fail(f"no such tool: {name!r}")
    try:
        return handler(ctx, session, args)
    except OpError as error:
        return fail(str(error))
    except ServiceError as error:
        return fail(error.message, field=error.field)
    except ValueError as error:
        return fail(str(error))
    except Exception:
        # Broad and logged, never silent -- ``tests/test_failure_paths.py``
        # scans for exactly this shape. An agent tool failing is not a crash
        # the app should show the user a traceback for; it is a refusal the
        # agent should be told about, with the detail kept in the log for
        # whoever has to work out what a library call did.
        log.exception("agent tool %r failed", name)
        return fail(f"{name} failed unexpectedly; see the log.")


def _h_scene(ctx: Any, session: Session, args: dict) -> dict:
    del args
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    objects = [_scene_row(doc, obj) for obj in doc.objects]

    boxes = [
        clay_geom_ops.world_box(obj)
        for obj in doc.objects
        if obj.visible and clay_geom_ops.world_box(obj) is not None
    ]
    bounds = None
    if boxes:
        lo = np.min([b[0] for b in boxes], axis=0)
        hi = np.max([b[1] for b in boxes], axis=0)
        bounds = {
            "min": _round(lo),
            "max": _round(hi),
            "size": _round(hi - lo),
            "center": _round((lo + hi) * 0.5),
        }

    materials = [
        {
            "index": i,
            "name": m.name,
            "color": _round(list(m.base_color_factor)),
            "metallic": _round(m.metallic_factor),
            "roughness": _round(m.roughness_factor),
        }
        for i, m in enumerate(doc.materials)
    ]

    return _json(
        {
            "objects": objects,
            "selection": sorted(doc.selection),
            "element_mode": doc.element_mode,
            "dirty": doc.dirty,
            "object_count": len(doc.objects),
            "bounds": bounds,
            "materials": materials,
        }
    )


def _h_add_primitive(ctx: Any, session: Session, args: dict) -> dict:
    """Place one primitive, with everything validated before the first
    mutation so a refused call places nothing -- see the tool's own
    description in :func:`tools` for the full argument list. Order:
    ``generator`` in the registry; ``params`` keys legal for it; the three
    TRS vectors well-formed; *then* the tab is resolved (minting one if the
    session owns none); *then* the object name (non-empty, not already
    taken) and the material index (in range) -- both of which need the
    document to answer.
    """
    generator = args.get("generator")
    if generator not in bp.GENERATORS:
        return fail(
            f"generator must be one of {', '.join(sorted(bp.GENERATORS))}.",
            field="generator",
        )

    params = args.get("params")
    if params is not None:
        if not isinstance(params, dict):
            return fail("params must be an object.", field="params")
        defaults = bp.GENERATORS[generator][0]
        unknown = sorted(set(params) - set(defaults))
        if unknown:
            return fail(
                f"unknown params {unknown} for {generator!r}; legal keys are "
                f"{sorted(defaults)}.",
                field="params",
            )

    translation = rotation_deg = scale = None
    if args.get("translation") is not None:
        translation, failure = _validate_vec3(args["translation"], "translation")
        if failure:
            return failure
    if args.get("rotation") is not None:
        rotation_deg, failure = _validate_vec3(args["rotation"], "rotation")
        if failure:
            return failure
    if args.get("scale") is not None:
        scale, failure = _validate_vec3(args["scale"], "scale")
        if failure:
            return failure

    tab, failure = _tab(ctx, session, create=True)
    if failure:
        return failure
    doc = tab.doc

    obj_name = args.get("name")
    if obj_name is not None:
        obj_name = str(obj_name)
        if not obj_name.strip():
            return fail("name must not be empty.", field="name")
        if any(o.name == obj_name for o in doc.objects):
            return fail(f"an object is already named {obj_name!r}.", field="name")

    material_index = args.get("material")
    if material_index is not None:
        try:
            material_index = int(material_index)
        except (TypeError, ValueError):
            return fail("material must be a palette index.", field="material")
        if not (0 <= material_index < len(doc.materials)):
            return fail(
                f"material must be an index into the palette (0..{len(doc.materials) - 1}).",
                field="material",
            )

    mark = doc.history.mark()
    obj = pane_clay_tools.add_primitive(ctx, doc, generator)
    if params:
        merged = bp.clamp_params(generator, {**obj.params, **params})
        mesh = shading.auto_smooth(bp.GENERATORS[generator][1](**merged))
        was = {"params": dict(obj.params)}
        doc.set_generator_params(obj.uid, merged, mesh, was=was)
    if translation is not None or rotation_deg is not None or scale is not None:
        doc.set_transform(
            obj.uid,
            translation=translation,
            rotation=None if rotation_deg is None else _quat_from_euler_xyz(rotation_deg),
            scale=scale,
        )
    if obj_name is not None:
        doc.set_props(obj.uid, name=obj_name)
    if material_index is not None:
        _repaint(doc, [obj.uid], material_index)
    doc.history.collapse_since(mark)
    _label_top(doc, mark, f"Add {obj.name}")
    return _json(_scene_row(doc, obj))


def _h_add_figure(ctx: Any, session: Session, args: dict) -> dict:
    """Place a figure preset as one group, one undo step. See
    :func:`tools`'s description for ``translation``/``yaw``/``scale``/
    ``name_prefix``.

    Per part, with ``T`` the translation, ``s`` the uniform scale and
    ``q_y`` the yaw quaternion: ``t' = R_y(yaw) . (s . t) + T``,
    ``q' = q_y (x) q`` and ``s' = s . s_part``. Yaw and scale are applied
    to every part's *offset from the group origin*, not to each part in
    its own local frame -- a yawed figure turns where its limbs sit, it
    does not spin each limb about its own centre.
    """
    key = args.get("key")
    if key not in presets.ASSEMBLIES:
        return fail(f"key must be one of {', '.join(sorted(presets.ASSEMBLIES))}.", field="key")

    translation = None
    if args.get("translation") is not None:
        translation, failure = _validate_vec3(args["translation"], "translation")
        if failure:
            return failure

    yaw_deg = args.get("yaw")
    if yaw_deg is not None:
        try:
            yaw_deg = float(yaw_deg)
        except (TypeError, ValueError):
            return fail("yaw must be a number.", field="yaw")
        if not math.isfinite(yaw_deg):
            return fail("yaw must be finite.", field="yaw")

    scale = args.get("scale")
    if scale is not None:
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            return fail("scale must be a number.", field="scale")
        if not (math.isfinite(scale) and scale > 0):
            return fail("scale must be a positive, finite number.", field="scale")

    name_prefix = args.get("name_prefix")
    if name_prefix is not None:
        name_prefix = str(name_prefix)

    tab, failure = _tab(ctx, session, create=True)
    if failure:
        return failure
    doc = tab.doc

    mark = doc.history.mark()
    objs = pane_clay_tools.add_assembly(ctx, doc, key)

    if name_prefix:
        # Checked *after* placement, against the names ``add_assembly`` chose
        # (already run through ``pane_clay_tools._unique_name`` for whatever
        # this document already held) rather than predicted beforehand
        # against ``presets.build``'s raw part names -- re-deriving that
        # de-duplication here to guess its answer would be a second copy of
        # it, free to drift the day it changes. A collision is undone rather
        # than left half-renamed, so a refused prefix still places nothing.
        placed = {o.uid for o in objs}
        existing = {o.name for o in doc.objects if o.uid not in placed}
        prefixed = [f"{name_prefix}{o.name}" for o in objs]
        if len(set(prefixed)) != len(prefixed) or existing & set(prefixed):
            doc.history.collapse_since(mark)
            doc.undo()
            return fail(
                f"{name_prefix!r} would collide with an existing object name.",
                field="name_prefix",
            )
        for obj, new_name in zip(objs, prefixed, strict=True):
            doc.set_props(obj.uid, name=new_name)

    if translation is not None or yaw_deg is not None or scale is not None:
        yaw_quat = m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), math.radians(yaw_deg or 0.0))
        s = 1.0 if scale is None else scale
        t = m3.vec3(*translation) if translation is not None else m3.vec3()
        for obj in objs:
            new_t = m3.quat_rotate(yaw_quat, obj.translation * s) + t
            new_q = m3.quat_mul(yaw_quat, obj.rotation)
            new_s = obj.scale * s
            doc.set_transform(obj.uid, translation=new_t, rotation=new_q, scale=new_s)

    doc.history.collapse_since(mark)
    label, _builder = presets.ASSEMBLIES[key]
    _label_top(doc, mark, f"Add {label}")
    return _json({"uids": [o.uid for o in objs], "objects": [_scene_row(doc, o) for o in objs]})


def _h_transform(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure
    translation = args.get("translation")
    rotation_deg = args.get("rotation")
    scale = args.get("scale")
    if translation is None and rotation_deg is None and scale is None:
        return fail("give at least one of translation, rotation or scale.")
    changed = doc.set_transform(
        obj.uid,
        translation=None if translation is None else [float(v) for v in translation],
        rotation=None if rotation_deg is None else _quat_from_euler_xyz(rotation_deg),
        scale=None if scale is None else [float(v) for v in scale],
    )
    return _json({"uid": obj.uid, "changed": changed})


def _h_set_params(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure
    if obj.generator is None:
        return fail(
            "This object's topology has been edited, so it is no longer a "
            "generated shape -- there are no generator params left to set "
            "(see document.set_mesh's freeze).",
            field="uid",
        )
    params = args.get("params")
    if not isinstance(params, dict) or not params:
        return fail("give at least one param to change.", field="params")
    defaults = bp.GENERATORS[obj.generator][0]
    unknown = sorted(set(params) - set(defaults))
    if unknown:
        return fail(
            f"unknown params {unknown} for {obj.generator!r}; legal keys are "
            f"{sorted(defaults)}.",
            field="params",
        )
    # Captured before anything below mutates ``obj.params`` -- the merge two
    # lines down edits it in place via a fresh dict, but ``set_generator_params``
    # itself reassigns ``obj.params`` to the very dict it is handed, so reading
    # "before" off the object once this call has run would compare a value
    # against itself. See that method's own docstring on why ``was`` is
    # mandatory for this caller.
    was = {"params": dict(obj.params)}
    merged = bp.clamp_params(obj.generator, {**obj.params, **params})
    mesh = shading.auto_smooth(bp.GENERATORS[obj.generator][1](**merged))
    changed = doc.set_generator_params(obj.uid, merged, mesh, was=was)
    # Reported back rather than echoed: a caller that asked for segments=2
    # learns here that clamp_params raised it to the generator's own floor.
    return _json(
        {"uid": obj.uid, "generator": obj.generator, "params": merged, "changed": changed}
    )


def _h_material(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    uids, failure = _resolve_uids(doc, args.get("uids"), field="uids")
    if failure:
        return failure
    if not uids:
        return fail("give at least one uid.", field="uids")
    color = args.get("color")
    if (
        not isinstance(color, list)
        or len(color) not in (3, 4)
        or not all(isinstance(c, int | float) for c in color)
    ):
        return fail("color must be an array of 3 or 4 numbers, 0..1.", field="color")
    rgba = tuple(float(c) for c in color)
    if len(rgba) == 3:
        rgba = (*rgba, 1.0)
    material = gltf.Material(
        name=str(args.get("name") or ""),
        base_color_factor=rgba,
        metallic_factor=float(args.get("metallic", 0.0)),
        roughness_factor=float(args.get("roughness", 0.6)),
    )

    # One material for the whole call -- never one per object -- folded into
    # one undo step the way ``add_material_and_assign`` folds its own pair,
    # so one tool call is one Ctrl+Z.
    mark = doc.history.mark()
    index = doc.add_material(material)
    _repaint(doc, uids, index)
    doc.history.collapse_since(mark)
    _label_top(doc, mark, "Set Material")
    return _json(
        {
            "index": index,
            "uids": uids,
            "color": list(rgba),
            "metallic": material.metallic_factor,
            "roughness": material.roughness_factor,
        }
    )


def _h_boolean(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    kind = args.get("kind")
    if kind not in ops_boolean.KINDS:
        return fail(f"kind must be one of {', '.join(ops_boolean.KINDS)}.", field="kind")
    try:
        wanted = [int(u) for u in args.get("uids") or []]
    except (TypeError, ValueError):
        return fail("uids must be a list of integers.", field="uids")
    # ``_union``'s own shape, generalised over the three kinds: the selection
    # is set from the request and then re-read the way every other selection-
    # based op reads it, so "first" means the target's place in the document's
    # own object order -- never the order this list happened to name them in.
    # See ``ops_boolean.KINDS``' own docstring for why that is the rule for a
    # difference, where the order changes the answer.
    doc.select(wanted)
    targets = [obj.uid for obj in doc.objects if obj.uid in doc.selection and obj.visible]
    if len(targets) < 2:
        return fail("Select at least two visible objects.", field="uids")
    mesh = ops_boolean.boolean([doc.by_uid(u) for u in targets], kind)
    doc.join_objects(targets[0], mesh, targets[1:])
    # clay-08 (2026-09-08 audit), the same pop ``clay_ops._join``/``_union``
    # make: the objects a boolean absorbs must not leave their manifold-check
    # cache entries pinned alive under a uid nothing owns any more.
    clay_ops._forget_manifold(ctx, targets[1:])
    doc.select([targets[0]])
    return _json({"uid": targets[0], "kind": kind})


def _h_select(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    uids, failure = _resolve_uids(doc, args.get("uids"), field="uids")
    if failure:
        return failure
    doc.select(uids)
    return _json({"selection": sorted(doc.selection)})


@dataclass
class _OpCtx:
    """A sandboxed stand-in for the real app ``ctx``, handed to ``clay_ops.run``.

    ``clay_ops`` reaches ``ctx`` in exactly three places (verified by reading
    the module before writing this): ``toast`` -- a module-level helper every
    refusal goes through -- and ``getattr(ctx, "clay_view", None)`` in
    ``_frame`` (Frame Selection), and ``getattr(ctx, "state", None)`` in
    ``_forget_manifold`` (the clay-08 manifold-cache pop). The absent
    ``clay_view`` is what makes Frame Selection the no-op it should always
    have been for an agent with no viewport of its own; ``state`` is passed
    through for real because the manifold-cache pop is real work that still
    has to happen. See the module docstring's ``clay_op`` paragraph for why
    the real ``ctx`` used to be handed over unsandboxed.
    """

    state: Any = None
    messages: list[str] = field(default_factory=list)

    def toast(self, message: str, level: str = "info") -> None:
        del level
        self.messages.append(message)


def _h_op(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    name = args.get("name")
    try:
        op = clay_ops.get(name)
    except KeyError:
        return fail(f"no op named {name!r}.", field="name")
    # ``run`` itself returns False, silently, for a disabled op -- exactly the
    # answer that is useless to an agent with no menu to look at and read the
    # greyed row's tooltip from. Checked here, once, so the refusal names the
    # gate (``reason_for`` is only ever consulted once ``enabled`` has already
    # said no, matching ``clay_ops``'s own rule for the two never disagreeing).
    if not op.enabled(doc):
        return fail(clay_ops.reason_for(op, doc))
    params = args.get("params") or {}
    proxy = _OpCtx(state=getattr(ctx, "state", None))
    ran = clay_ops.run(proxy, doc, op, **params)
    return _json({"op": op.name, "ran": ran, "messages": proxy.messages})


def _parse_view_entry(entry: Any, valid_views: set[str]) -> tuple[str, dict, dict | None]:
    """One ``views[]`` entry -> ``(label, render_png kwargs, failure)``.

    A named axis/``three_quarter`` view maps straight to ``render_png``'s own
    ``view`` keyword. A free ``{yaw, pitch}`` pair maps to ``angles=(theta,
    phi)`` in radians: ``theta = radians(yaw)``, ``phi = radians(90 - pitch)``.
    That makes yaw 0 the front (``Camera.AXIS_VIEWS["front"]`` is theta=0) and
    +pitch look *down* from above (``AXIS_VIEWS["top"]`` is phi ~= 0, i.e.
    pitch +90) -- the sign that is wrong half the time and invisible until a
    picture is looked at, so it is written down here rather than trusted to
    memory.
    """
    if isinstance(entry, str):
        if entry not in valid_views:
            return (
                "",
                {},
                fail(
                    f"each view must be one of {', '.join(sorted(valid_views))} or "
                    "an object with yaw and pitch.",
                    field="views",
                ),
            )
        return entry, {"view": entry}, None
    if isinstance(entry, dict) and set(entry) == {"yaw", "pitch"}:
        try:
            yaw = float(entry["yaw"])
            pitch = float(entry["pitch"])
        except (TypeError, ValueError):
            return "", {}, fail("yaw and pitch must be numbers.", field="views")
        if not math.isfinite(yaw) or not (-89.0 <= pitch <= 89.0):
            return "", {}, fail("pitch must be between -89 and 89 degrees.", field="views")
        angles = (math.radians(yaw), math.radians(90.0 - pitch))
        return f"yaw={yaw:g},pitch={pitch:g}", {"angles": angles}, None
    return "", {}, fail("each view must be a name or an object with yaw and pitch.", field="views")


def _h_render(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc

    try:
        size = max(64, min(int(args.get("size") or 1024), 2048))
    except (TypeError, ValueError):
        return fail("size must be an integer.", field="size")

    view = args.get("view")
    views_arg = args.get("views")
    if view is not None and views_arg is not None:
        return fail("give either view or views, not both.", field="views")

    valid_views = set(Camera.AXIS_VIEWS) | {"three_quarter"}
    if views_arg is not None:
        if not isinstance(views_arg, list) or not views_arg:
            return fail("views must be a non-empty list.", field="views")
        entries = views_arg
    elif view is not None:
        if view not in valid_views:
            return fail(f"view must be one of {', '.join(sorted(valid_views))}.", field="view")
        entries = [view]
    else:
        entries = ["three_quarter"]

    parsed: list[tuple[str, dict]] = []
    for entry in entries:
        label, kwargs, failure = _parse_view_entry(entry, valid_views)
        if failure:
            return failure
        parsed.append((label, kwargs))

    grid = bool(args.get("grid", False))

    focus = args.get("focus")
    bounds = None
    if focus is not None:
        uids, failure = _resolve_uids(doc, focus, field="focus")
        if failure:
            return failure
        boxes = [
            box for box in (clay_geom_ops.world_box(doc.by_uid(u)) for u in uids) if box is not None
        ]
        if boxes:
            lo = np.min([b[0] for b in boxes], axis=0)
            hi = np.max([b[1] for b in boxes], axis=0)
            bounds = (lo, hi)

    compare = args.get("compare")
    reference = None
    compare_mode = args.get("compare_mode", "beside")
    alpha = args.get("alpha", 0.5)
    if compare is not None:
        reference = session.references.get(compare)
        if reference is None:
            return fail(f"no reference named {compare!r}.", field="compare")
        if compare_mode not in ("beside", "overlay"):
            return fail("compare_mode must be 'beside' or 'overlay'.", field="compare_mode")
        if compare_mode == "overlay":
            try:
                alpha = float(alpha)
            except (TypeError, ValueError):
                return fail("alpha must be a number.", field="alpha")
        if len(parsed) > 1:
            return fail("compare renders exactly one view.", field="views")
        if views_arg is None and view is None:
            # No view was asked for: default to the reference's own angle, so
            # the comparison is framed the way the picture being matched was.
            label = reference.view if reference.view in valid_views else "three_quarter"
            parsed = [(label, {"view": label})]
        size = min(size, 1024)
    else:
        total_pixels = len(parsed) * size * size
        if total_pixels > RENDER_PIXEL_BUDGET:
            return fail(
                f"{len(parsed)} view(s) at {size}x{size} would be "
                f"{total_pixels:,} pixels, over the {RENDER_PIXEL_BUDGET:,}-pixel "
                "render budget for one call -- ask for fewer or smaller views."
            )

    try:
        view_obj = _view_for(ctx)
    except Exception:
        log.exception("agent render of a Clay document failed")
        return fail("That document could not be rendered; see the log.")

    try:
        pngs = [
            view_obj.render_png(doc, size=size, grid=grid, bounds=bounds, **kwargs)
            for _label, kwargs in parsed
        ]
    except Exception:
        log.exception("agent render of a Clay document failed")
        return fail("That document could not be rendered; see the log.")

    if compare is not None:
        from . import agent_refs

        if compare_mode == "beside":
            sheet = agent_refs.beside(reference.png, pngs[0], compare, "render", size=size)
        else:
            sheet = agent_refs.overlay(reference.png, pngs[0], alpha, size=size)
        import io

        from PIL import Image

        with Image.open(io.BytesIO(sheet)) as im:
            width, height = im.width, im.height
        return ok(
            text(
                json.dumps(
                    {
                        "view": parsed[0][0],
                        "reference": compare,
                        "mode": compare_mode,
                        "width": width,
                        "height": height,
                    }
                )
            ),
            image_png(sheet),
        )

    # base64 costs 4 bytes for every 3 of input, rounded up: the frame budget
    # is checked against what actually crosses the wire, not the raw PNG size.
    b64_total = sum(((len(png) + 2) // 3) * 4 for png in pngs)
    if b64_total > _protocol().MAX_FRAME - RENDER_FRAME_RESERVE:
        return fail(
            "This render is too large to send back in one reply frame; ask "
            "for fewer or smaller views."
        )

    header = text(json.dumps({"views": [label for label, _ in parsed], "size": size, "grid": grid}))
    return ok(header, *(image_png(png) for png in pngs))


def _h_diagnose(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    uid = args.get("uid")
    if uid is None:
        targets = [obj for obj in doc.objects if obj.visible]
    else:
        obj, failure = _resolve_uid(doc, args, "uid")
        if failure:
            return failure
        targets = [obj]
    report = []
    for obj in targets:
        rows = clay_diagnose.findings(obj.mesh)
        report.append(
            {
                "uid": obj.uid,
                "name": obj.name,
                "clean": not rows,
                "findings": [
                    {"kind": row.kind, "label": row.label, "count": row.count, "mode": row.mode}
                    for row in rows
                ],
            }
        )
    return _json({"objects": report})


def _h_export(ctx: Any, session: Session, args: dict) -> dict:
    del args
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    if tab.saving:
        return fail("A save for this document is already in progress.")
    if not any(obj.visible for obj in doc.objects):
        return fail("There is nothing visible to export.")

    # ``clay_mode.camera_of`` reads *whatever tab the interactive viewport is
    # currently showing*, which is never this one -- an agent's document is
    # never on screen by definition. Passing ``tab.view`` straight through
    # avoids stamping the agent's document with the user's current camera.
    #
    # The chain itself -- ``to_model``, the GLB write, ``import_mesh``, and the
    # ``.wblk`` source sidecar (``save_clay_source``) -- is
    # ``clay_mode.build_asset`` now; see its docstring for why both this call
    # and ``export_asset``'s go through it, and the module docstring's first
    # departure from ``save_to`` for why that sidecar is the only copy this
    # module ever needs -- there is no second one to keep of its own. There is
    # no frame boundary an MCP call can hand the encode across the way
    # ``export_asset`` hands it to a task thread, so it all runs right here,
    # synchronously, before this handler returns -- a deliberate one-shot
    # cost, not the per-frame stall the task-thread split exists to prevent.
    job_id = clay_mode.build_asset(ctx.svc, doc, title=tab.title, view=tab.view)

    tab.job_id = job_id
    ctx.cache.invalidate()
    return _json({"job_id": job_id})


def _move_history(ctx: Any, session: Session, args: dict, *, redo: bool) -> dict:
    """The shared body of ``clay_undo``/``clay_redo``. See ``tools()`` for the
    documented "moves the head, pushes nothing" exception both belong to."""
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    steps = args.get("steps", 1)
    try:
        steps = int(steps)
    except (TypeError, ValueError):
        return fail("steps must be an integer.", field="steps")
    if not (1 <= steps <= 64):
        return fail("steps must be between 1 and 64.", field="steps")

    step_fn = doc.redo if redo else doc.undo
    moved = 0
    for _ in range(steps):
        if not step_fn():
            break
        moved += 1

    # ``ObjectAddEdit.undo``/``ObjectRemoveEdit.redo`` already discard their
    # own uid from ``doc.selection`` when they run (read them) -- but that
    # self-pruning lives on those two edit types alone, and a compound step
    # can bundle them with edit types that carry no such rule (``join_objects``'s
    # own ``MeshEdit``/``ObjectPropsEdit`` siblings, say). Pruned here, once,
    # after every move, rather than trusted to every current and future
    # ``Edit.undo``/``redo`` to have covered it.
    known = {obj.uid for obj in doc.objects}
    doc.selection = {uid for uid in doc.selection if uid in known}

    return _json(
        {
            "moved": moved,
            "done_steps": len(doc.history),
            "can_undo": doc.history.can_undo,
            "can_redo": doc.history.can_redo,
        }
    )


def _h_undo(ctx: Any, session: Session, args: dict) -> dict:
    return _move_history(ctx, session, args, redo=False)


def _h_redo(ctx: Any, session: Session, args: dict) -> dict:
    return _move_history(ctx, session, args, redo=True)


def _h_delete(ctx: Any, session: Session, args: dict) -> dict:
    """Remove objects by uid, directly -- never through ``clay_op``'s own
    Delete row.

    ``clay_op`` ``delete`` acts on the document's *selection*, and in a face
    or edge mode ``selection.delete_selected`` never removes an object at
    all -- so an agent that had switched element mode (perhaps from an
    earlier ``clay_op`` call) would see a silent no-op where it asked for a
    deletion. Working from the uids given, in whatever element mode the
    document happens to be in, is what keeps "delete these objects" meaning
    that regardless.
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

    mark = doc.history.mark()
    for uid in uids:
        doc.remove_object(uid)
    doc.history.collapse_since(mark)
    _label_top(doc, mark, "Delete")
    # clay-08 (2026-09-08 audit): an object that leaves ``doc.objects`` must
    # not leave its manifold-check cache entry pinning a whole ``Mesh`` alive
    # under a uid nobody owns -- the same pop ``clay_ops._join``/``_union``
    # make when they absorb objects, here for the direct-delete path
    # ``clay_op``'s own Delete row does not take.
    clay_ops._forget_manifold(ctx, uids)
    return _json({"deleted": uids})


def _h_rename(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure
    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        return fail("name must not be empty.", field="name")
    if any(other.uid != obj.uid and other.name == name for other in doc.objects):
        return fail(f"an object is already named {name!r}.", field="name")
    doc.set_props(obj.uid, name=name)
    return _json({"uid": obj.uid, "name": name})


def _h_batch(ctx: Any, session: Session, args: dict) -> dict:
    """Run several tools as one undo step. See the module docstring's own
    paragraph on the fold and :data:`BATCH_EXCLUDED` for what this refuses to
    run and why.

    The whole list's shape is validated before anything runs, so a malformed
    batch runs nothing. If the session owns no tab yet, this refuses unless
    the *first* call is ``clay_add_primitive`` or ``clay_add_figure``, in
    which case it mints one through ``_tab(..., create=True)`` itself --
    ``_h_batch`` needs a document in hand before the loop starts (to open the
    ``history.mark()`` the whole run folds into), so the mint has to happen
    here rather than be left to the first sub-call, but it is still one of
    the two creator tools that is about to run, which is what keeps "only
    those two mint a document" true.
    """
    calls = args.get("calls")
    if not isinstance(calls, list) or not (1 <= len(calls) <= BATCH_MAX):
        return fail(f"calls must be a list of 1 to {BATCH_MAX} tool calls.", field="calls")
    allowed = set(_HANDLERS) - BATCH_EXCLUDED
    for entry in calls:
        if not isinstance(entry, dict):
            return fail("every call must be an object with a name.", field="calls")
        name = entry.get("name")
        if name not in allowed:
            return fail(f"{name!r} is not a batchable tool.", field="calls")
        arguments = entry.get("arguments")
        if arguments is not None and not isinstance(arguments, dict):
            return fail("each call's arguments must be an object.", field="calls")

    if not session.tab_uid:
        first_name = calls[0].get("name")
        if first_name not in ("clay_add_primitive", "clay_add_figure"):
            return fail(
                "This session has no document yet. The first call in a "
                "batch that starts one must be clay_add_primitive or "
                "clay_add_figure."
            )
        _, failure = _tab(ctx, session, create=True)
        if failure:
            return failure

    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc

    mark = doc.history.mark()
    results: list[dict] = []
    stopped_at: int | None = None
    for i, entry in enumerate(calls):
        result = call(ctx, session, entry["name"], entry.get("arguments") or {})
        results.append(result)
        if result.get("isError"):
            stopped_at = i
            break
    doc.history.collapse_since(mark)
    _label_top(doc, mark, "Agent batch")

    completed = len(results) - (1 if stopped_at is not None else 0)
    payload = {"completed": completed, "stopped_at": stopped_at, "results": results}
    result = ok(text(json.dumps(payload)))
    # Set by hand rather than through ``fail()``: a batch that stopped early
    # is a failure the agent must notice, but the payload it needs in order
    # to recover -- the successful prefix, and the failing call's own message
    # -- is a JSON result block, and ``fail()`` can only carry a message plus
    # flat ``structuredContent``, not both of those.
    result["isError"] = stopped_at is not None
    return result


def _h_reference_add(ctx: Any, session: Session, args: dict) -> dict:
    """Hand this session a picture from a Library job or inline base64. See
    the module docstring's references paragraph and :func:`tools`'s
    description for the full contract."""
    from . import agent_refs

    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        return fail("name must not be empty.", field="name")

    job_id = args.get("job_id")
    png_b64 = args.get("png_base64")
    if (job_id is None) == (png_b64 is None):
        return fail("give exactly one of job_id or png_base64.", field="job_id")

    view = args.get("view", "other")
    valid_views = set(Camera.AXIS_VIEWS) | {"three_quarter", "other"}
    if view not in valid_views:
        return fail(f"view must be one of {', '.join(sorted(valid_views))}.", field="view")

    if job_id is not None:
        try:
            job = ctx.svc.require_job(job_id)
        except NotFound as error:
            return fail(error.message, field="job_id")
        job_dir = ctx.svc.job_dir(job_id)
        file_arg = args.get("file")
        candidates = (
            [file_arg] if file_arg else ["input.png", "ref.png", "reference.png", "thumb.png"]
        )
        chosen = next((c for c in candidates if svc_files.ready(job, job_dir, c)), None)
        if chosen is None:
            return fail(
                "no reference image is ready for this job; looked for "
                + ", ".join(candidates)
                + ".",
                field="job_id",
            )
        path = svc_files.job_dir_file(ctx.svc, job_id, chosen)
        data = path.read_bytes()
        source = f"job:{job_id}:{chosen}"
    else:
        import base64
        import binascii

        try:
            data = base64.b64decode(png_b64, validate=True)
        except (binascii.Error, ValueError):
            return fail("png_base64 must be valid base64.", field="png_base64")
        if len(data) > svc_validation.MAX_UPLOAD_BYTES:
            # Belt and braces: the 8 MiB protocol frame this call arrived over
            # already bounds an inline upload's size, so this should never
            # actually be reachable -- but the ceiling is worth naming in
            # case that frame limit ever moves.
            return fail(
                f"png_base64 decodes to more than {svc_validation.MAX_UPLOAD_BYTES:,} bytes.",
                field="png_base64",
            )
        source = "inline"

    try:
        png, width, height = agent_refs.normalise(data)
    except svc_files.ImageTooLarge as error:
        return fail(str(error), field="job_id" if job_id is not None else "png_base64")

    replaced = name in session.references
    if not replaced and len(session.references) >= MAX_REFERENCES:
        return fail(
            f"this session already holds {MAX_REFERENCES} references, its "
            "cap; remove one with clay_reference_remove first.",
            field="name",
        )

    session.references[name] = agent_refs.Reference(
        name=name, png=png, width=width, height=height, view=view, source=source
    )
    # The whole of what the human at the keyboard is shown this round -- see
    # the module docstring for why that is a deliberate scope line, not an
    # oversight.
    ctx.toast(f"Agent reference {name!r} added.")
    return _json(
        {
            "name": name,
            "width": width,
            "height": height,
            "view": view,
            "source": source,
            "replaced": replaced,
        }
    )


def _h_reference_list(ctx: Any, session: Session, args: dict) -> dict:
    del ctx, args
    refs = sorted(session.references.values(), key=lambda r: r.name)
    return _json(
        {
            "references": [
                {
                    "name": r.name,
                    "width": r.width,
                    "height": r.height,
                    "view": r.view,
                    "source": r.source,
                }
                for r in refs
            ],
            "max": MAX_REFERENCES,
        }
    )


def _h_reference_get(ctx: Any, session: Session, args: dict) -> dict:
    from . import agent_refs

    del ctx
    name = args.get("name")
    ref = session.references.get(name)
    if ref is None:
        return fail(f"no reference named {name!r}.", field="name")
    meta = {
        "name": ref.name,
        "width": ref.width,
        "height": ref.height,
        "view": ref.view,
        "source": ref.source,
    }
    return ok(text(json.dumps(meta)), image_png(agent_refs.bounded_png(ref.png)))


def _h_reference_remove(ctx: Any, session: Session, args: dict) -> dict:
    del ctx
    name = args.get("name")
    if name not in session.references:
        return fail(f"no reference named {name!r}.", field="name")
    del session.references[name]
    return _json({"removed": name})


_HANDLERS = {
    "clay_scene": _h_scene,
    "clay_add_primitive": _h_add_primitive,
    "clay_add_figure": _h_add_figure,
    "clay_transform": _h_transform,
    "clay_set_params": _h_set_params,
    "clay_material": _h_material,
    "clay_boolean": _h_boolean,
    "clay_select": _h_select,
    "clay_op": _h_op,
    "clay_render": _h_render,
    "clay_diagnose": _h_diagnose,
    "clay_export": _h_export,
    "clay_undo": _h_undo,
    "clay_redo": _h_redo,
    "clay_delete": _h_delete,
    "clay_rename": _h_rename,
    "clay_batch": _h_batch,
    "clay_reference_add": _h_reference_add,
    "clay_reference_list": _h_reference_list,
    "clay_reference_get": _h_reference_get,
    "clay_reference_remove": _h_reference_remove,
}
