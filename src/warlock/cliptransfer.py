"""Import clip: the pure host math that poses a Warlock template rig from an
external animation, once :mod:`clipmaps` has said which source bone plays
which template bone.

**This is never called "retarget" here** -- see ``clipmaps``'s module
docstring for why that word is already spoken for
(``pipelines/optimize.py``'s triangle-budget re-optimisation). "Import clip"
is the feature name throughout the UI and this module.

The division of labour with the rest of "Import clip": a Blender operator
(``clip_sample``, in ``pipelines/blender_worker.py``) samples an externally
authored armature -- its rest bones and, per action, per frame, every posed
bone's world rotation and head position -- into the plain-dict shape this
module's :func:`transfer` takes as ``sample``. Nothing here touches ``bpy``;
nothing here writes a file. What comes out is exactly the v3 clip-library
shape :func:`warlock.cliplib.parse_clip_library` already knows how to read,
minus ``source`` (a service-layer provenance note the caller adds) and minus
``provisional`` (a caller decision, not a fact about the math).

**The rotation math is two changes of frame stacked on each other.** A
posed source bone's *world* rotation is turned into a *delta from that
bone's own world rest* (``Q_src(f) . Q_src_rest^-1``, the source's own
world frame); that delta is then re-expressed in the target's facing frame
by conjugating with ``C``, the one rotation that squares the source
skeleton's own left/up/forward onto the template's axes
(``C . delta . C^-1``, standard change-of-basis for a rotation). What
results, ``D(b, f)``, is "how far this bone swung, as the target skeleton
would see it" -- and it composes onto the target's own rest orientation the
same way ``pipelines.blender_worker._rest_local_rotation``'s docstring
already composes a posed bone from its rest and its own delta, just with the
source's swing standing in for the target's own ``basis``. ``G(b)`` is the
one further correction that formula does not have to make: the two
skeletons' *neutral* pose can point a bone in different directions (a
Mixamo T-pose's arm is not a Warlock A-pose's arm), and ``G(b)`` is the
shortest arc from the target's own rest direction onto the source's, folded
in before the source's motion is applied.

**``sheet.slerp``/``sheet.MAX_CLIP_FRAMES`` are imported, not restated, since
the 2026-09-17 restructure.** ``tests/modes/poser/test_poser_imports.py`` pins every
module here (``poselib``, ``clipmaps``, and now this one) to import no more
of ``warlock`` than a short, explicit set -- and one of its own generic
checks (``test_none_of_them_imports_the_queue_or_the_pipelines``) refuses a
``warlock.pipelines`` import from *any* of them, this module included. That
used to make ``warlock.pipelines.sheet`` (this module's old location, before
the move) unreachable from here, so
``sheet.slerp`` and ``sheet.MAX_CLIP_FRAMES`` were restated verbatim
instead -- a duplicate this codebase keeps finding has silently drifted
(``dev/RESTRUCTURE.md``'s own "constants get restated instead of imported"
failure shape). ``sheet.py`` moved to ``warlock.kernels.sheet`` because it
was always a kernel wearing a pipelines name (stdlib plus a lazy Pillow
import), and the ban above was never about kernels -- it exists to keep
"Import clip" decidable with no Blender and no torch behind it, which a pure
kernel does not carry. So the restatement's *reason* is gone, not just its
cost: this module now does ``from .kernels import sheet`` and the two names
are the real functions/constant, not a second copy of them.
``poselib.MAX_ROOT_TRANSLATION`` is a different case -- ``poselib`` is a
sibling module in this same import-pinned set, not something outside it --
and stays restated below, pinned against its source of truth by a test in
``tests/test_cliptransfer.py`` so the two cannot drift apart silently.

Quaternions are XYZW throughout, this package's convention everywhere else.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from . import clipmaps
from .kernels import sheet
from .kernels.rig import cliplib, templates

__all__ = [
    "ClipTransferError",
    "transfer",
    "MAX_CLIP_FRAMES",
    "MAX_ROOT_TRANSLATION",
    "LOOP_MATCH_DEG",
    "LOOP_MATCH_ROOT_Z",
    "KEY_TOLERANCE_DEG",
    "ROOT_TOLERANCE",
    "BONE_IDENTITY_DEG",
]


class ClipTransferError(ValueError):
    """A refused "Import clip" conversion, naming the field it came from.

    The ``kernels.rig.store.RigError`` / ``poselib.RecordError`` shape: a
    ``ValueError`` subclass so nothing that already catches ``ValueError``
    changes, with ``field`` carried so ``service.errors.invalid_from`` can
    point the UI at the right control.
    """

    def __init__(self, message: str, *, field: str) -> None:
        super().__init__(message)
        self.field = field


# --- constants -------------------------------------------------------
#
# See the module docstring: ``MAX_CLIP_FRAMES`` is the real
# ``kernels.sheet`` constant, not a copy of it, now that ``sheet.py`` is a
# kernel this module's import pin may reach. ``MAX_ROOT_TRANSLATION`` is
# still a restatement -- ``poselib`` is a sibling in the same pinned set,
# not something outside it -- pinned back to its source by a test in
# ``tests/test_cliptransfer.py``.

#: The real ``kernels.sheet.MAX_CLIP_FRAMES``, bound here so every call site
#: below keeps its own short name.
MAX_CLIP_FRAMES = sheet.MAX_CLIP_FRAMES

#: Restated from ``poselib.MAX_ROOT_TRANSLATION``: a root offset past two
#: character heights is a fat-fingered gizmo drag on the *editor* side and a
#: broken sample or a wildly different rig scale on this one -- either way,
#: not a translation an "Import clip" run should carry through silently.
MAX_ROOT_TRANSLATION = 2.0

#: Loop auto-detection: how close the first and last sampled frame have to be
#: -- in the worst-moved mapped bone, in degrees -- before a clip is treated
#: as already cyclic and its duplicate closing frame dropped.
LOOP_MATCH_DEG = 3.0

#: Loop auto-detection's other half: how close the root's *height* has to
#: return to where it started. Character-height units, like every other
#: root-translation figure in this codebase.
LOOP_MATCH_ROOT_Z = 0.01

#: Key-reduction (RDP) tolerances: how far a dropped frame's true pose may
#: stray from the straight-line (slerp/lerp) prediction between the two keys
#: either side of it before it has to be kept as a key of its own.
KEY_TOLERANCE_DEG = 1.5
ROOT_TOLERANCE = 0.005

#: A mapped bone whose basis never strays further than this from identity
#: across the whole clip is authored as if it were never mapped: omitting it
#: from every key is exactly what a "delta" pose already means for a bone it
#: does not name (``kernels.sheet._blend``'s docstring).
BONE_IDENTITY_DEG = 0.01

_IDENTITY_Q = (0.0, 0.0, 0.0, 1.0)
_EPS_LEN = 1e-9

#: XYZW throughout -- a short alias so a signature naming several of these
#: (``_pose_frame``'s ``geometry``, ``_resample``'s return) fits one line.
_Quat = tuple[float, float, float, float]
_Vec3 = tuple[float, float, float]


# --- kernels.sheet's own slerp, bound to this module's short name --------

#: The real ``kernels.sheet.slerp``, not a copy of it -- see the module
#: docstring. Bound under this module's own short name because every call
#: site below (and ``tests/test_cliptransfer.py``'s
#: ``cliptransfer._slerp``) already spells it this way.
_slerp = sheet.slerp


# --- small quaternion / vector math ---------------------------------------
#
# A private restatement of the one identity ``kernels.rig.poses`` already
# states for ``node``/``delta`` (``node = rest . delta``), rather than an
# import of its private ``_quat_mul``: the two modules happen to need the
# same arithmetic, not the same function object.


def _mul(a: Sequence[float], b: Sequence[float]) -> _Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _mul_all(*qs: Sequence[float]) -> _Quat:
    return functools.reduce(_mul, qs)


def _conj(q: Sequence[float]) -> _Quat:
    x, y, z, w = q
    return (-x, -y, -z, w)


def _normalize_q(q: Sequence[float]) -> _Quat:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return (x / n, y / n, z / n, w / n)


def _angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    """The angle between two rotations, in degrees, shortest-arc."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    dot = max(-1.0, min(1.0, abs(ax * bx + ay * by + az * bz + aw * bw)))
    return math.degrees(2.0 * math.acos(dot))


def _sub3(a: Sequence[float], b: Sequence[float]) -> _Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a: Sequence[float], b: Sequence[float]) -> _Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm3(a: Sequence[float]) -> float:
    return math.sqrt(_dot3(a, a))


def _normalize3(a: Sequence[float]) -> _Vec3:
    n = _norm3(a)
    if n < _EPS_LEN:
        # Only reached for a degenerate per-bone rest direction (a
        # zero-length authored bone); the facing axes are guarded by name
        # in ``_facing_alignment`` instead of falling back silently here.
        return (0.0, 0.0, 1.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def _lerp3(a: Sequence[float], b: Sequence[float], t: float) -> _Vec3:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _dist3(a: Sequence[float], b: Sequence[float]) -> float:
    return _norm3(_sub3(a, b))


def _rotate_vector(q: Sequence[float], v: Sequence[float]) -> _Vec3:
    """Rotate a 3-vector by a unit XYZW quaternion."""
    x, y, z, w = q
    qv = (x, y, z)
    t = tuple(2.0 * c for c in _cross3(qv, v))
    u = _cross3(qv, t)
    return (v[0] + w * t[0] + u[0], v[1] + w * t[1] + u[1], v[2] + w * t[2] + u[2])


def _from_two_vectors(a: Sequence[float], b: Sequence[float]) -> _Quat:
    """The shortest-arc rotation taking unit vector ``a`` onto unit vector ``b``."""
    a = _normalize3(a)
    b = _normalize3(b)
    d = _dot3(a, b)
    if d < -1.0 + 1e-7:
        # Antiparallel: the arc is 180 degrees around any axis perpendicular
        # to ``a`` -- there is no unique "shortest" one, so pick one.
        axis = _cross3(a, (1.0, 0.0, 0.0))
        if _norm3(axis) < 1e-6:
            axis = _cross3(a, (0.0, 1.0, 0.0))
        axis = _normalize3(axis)
        return (axis[0], axis[1], axis[2], 0.0)
    s = math.sqrt(max(0.0, (1.0 + d) * 2.0))
    inv_s = 1.0 / s
    axis = _cross3(a, b)
    return _normalize_q((axis[0] * inv_s, axis[1] * inv_s, axis[2] * inv_s, s * 0.5))


def _quat_from_rows(
    row0: Sequence[float], row1: Sequence[float], row2: Sequence[float]
) -> _Quat:
    """XYZW quaternion for the rotation matrix with these three rows.

    Shepperd's method: numerically stable across the whole rotation group by
    picking whichever of the four algebraically-equivalent formulas divides
    by the largest quantity.
    """
    m00, m01, m02 = row0
    m10, m11, m12 = row1
    m20, m21, m22 = row2
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(max(1e-12, 1.0 + m00 - m11 - m22))
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(max(1e-12, 1.0 + m11 - m00 - m22))
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(max(1e-12, 1.0 + m22 - m00 - m11))
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return _normalize_q((x, y, z, w))


# --- facing and rest alignment (rest-pose geometry, computed once) --------


def _facing_alignment(
    source_bones: Mapping[str, Any], resolved: Mapping[str, tuple[str, ...]]
) -> _Quat:
    """``C``: the one rotation squaring the source's own rest pose onto the
    template's axes (+X left, +Z up, -Y forward).

    Landmarks are read off four template bones by name -- ``thigh.L``,
    ``thigh.R``, ``hips``, ``head`` -- the shipped humanoid template's own
    spelling, because a facing direction has to be read off *some* named
    pair of legs and a spine, and every shipped clip map for this template
    maps all four.
    """
    try:
        thigh_l = resolved["thigh.L"]
        thigh_r = resolved["thigh.R"]
        hips = resolved["hips"]
        head = resolved["head"]
    except KeyError as exc:
        raise ClipTransferError(
            f"cannot tell which way this rig faces: {exc} did not map to a source bone",
            field="source",
        ) from exc

    left_raw = _sub3(source_bones[thigh_l[0]]["head"], source_bones[thigh_r[0]]["head"])
    if _norm3(left_raw) < _EPS_LEN:
        raise ClipTransferError(
            "the left and right thigh are at the same point; cannot tell which way this rig faces",
            field="source",
        )
    left = _normalize3(left_raw)

    up_raw = _sub3(source_bones[head[-1]]["tail"], source_bones[hips[0]]["head"])
    up_raw = _sub3(up_raw, tuple(c * _dot3(up_raw, left) for c in left))  # orthogonalise
    if _norm3(up_raw) < _EPS_LEN:
        raise ClipTransferError(
            "the head sits directly above the hips with no lean either way; "
            "cannot tell which way this rig faces",
            field="source",
        )
    up = _normalize3(up_raw)
    forward = _cross3(left, up)

    # C maps (left, up, forward) -> (+X, +Z, -Y). Applying C to the standard
    # basis vectors gives C's columns; read off as rows of the *source*
    # vectors this yields row0=left, row1=-forward, row2=up (worked out in
    # the module's implementation notes and checked by
    # ``test_a_source_facing_plus_y_is_turned_to_face_minus_y``, whose
    # left-at-+X/up-at-+Z/forward-at--Y source is the fixed point of this
    # matrix, i.e. C == identity there).
    neg_forward = (-forward[0], -forward[1], -forward[2])
    return _quat_from_rows(left, neg_forward, up)


def _rest_alignment(
    source_bones: Mapping[str, Any],
    target_bones: Mapping[str, Any],
    target_name: str,
    chain: tuple[str, ...],
    c: Sequence[float],
) -> _Quat:
    """``G(target_name)``: the shortest arc from the target's own rest
    direction onto the source's, both read at rest and the source's
    expressed in the ``C``-aligned frame.
    """
    first, last = chain[0], chain[-1]
    source_dir = _sub3(source_bones[last]["tail"], source_bones[first]["head"])
    d_s = _rotate_vector(c, _normalize3(source_dir))
    target = target_bones[target_name]
    d_t = _sub3(target["tail"], target["head"])
    return _from_two_vectors(d_t, d_s)


def _topo_order(target_bones: Mapping[str, Any]) -> list[str]:
    """Target bone names, parent before child."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        parent = target_bones[name].get("parent")
        if parent is not None and parent in target_bones:
            visit(parent)
        seen.add(name)
        order.append(name)

    for name in target_bones:
        visit(name)
    return order


# --- per-frame posing -------------------------------------------------------


def _root_height(
    source_bones: Mapping[str, Any], head_chain: tuple[str, ...], c: Sequence[float]
) -> float:
    """The source's own height, in its rest pose, after ``C`` -- what a
    stored root offset is measured in units of, exactly like
    ``poselib.UNIT_HI[2]``'s character-height convention.
    """
    lowest = min(
        _rotate_vector(c, pos)[2]
        for bone in source_bones.values()
        for pos in (bone["head"], bone["tail"])
    )
    top = _rotate_vector(c, source_bones[head_chain[-1]]["tail"])[2]
    height = top - lowest
    if height < _EPS_LEN:
        raise ClipTransferError(
            "this rig's rest pose has no measurable height", field="source"
        )
    return height


def _pose_frame(
    order: Sequence[str],
    target_bones: Mapping[str, Any],
    resolved: Mapping[str, tuple[str, ...]],
    geometry: Mapping[str, tuple[_Quat, _Quat]],
    c: Sequence[float],
    c_inv: Sequence[float],
    frame_bones: Mapping[str, Any],
) -> dict[str, _Quat]:
    """One frame's ``basis(b)`` for every *mapped* target bone.

    Walks the target hierarchy root-first so an unmapped bone's world
    orientation (needed only to pose a mapped *child* of it) can be carried
    down from its own parent -- see the module docstring's derivation:
    an unmapped bone's basis is always identity, so it is never added to the
    returned map, exactly what a "delta" pose already means for a bone it
    does not name.
    """
    world: dict[str, _Quat] = {}
    basis: dict[str, _Quat] = {}
    for name in order:
        parent = target_bones[name].get("parent")
        r_b = tuple(target_bones[name]["rest_rotation"])
        if name in resolved:
            chain = resolved[name]
            last = chain[-1]
            q_src_rest, g = geometry[name]
            src_entry = frame_bones.get(last)
            q_src_f = tuple(src_entry["rotation"]) if src_entry is not None else q_src_rest
            delta_local = _mul(q_src_f, _conj(q_src_rest))
            d = _mul_all(c, delta_local, c_inv)
            w_b = _mul_all(d, g, r_b)
            if parent is None:
                basis[name] = _mul(_conj(r_b), w_b)
            else:
                r_p = tuple(target_bones[parent]["rest_rotation"])
                w_p = world[parent]
                basis[name] = _mul_all(_conj(r_b), r_p, _conj(w_p), w_b)
            world[name] = w_b
        else:
            if parent is None:
                # Guarded rather than reached: every shipped clip map's
                # ``required`` list includes its ``root``, so a candidate
                # that qualifies in ``clipmaps.match`` always resolves it.
                raise ClipTransferError(
                    f"the root bone {name!r} did not map to a source bone", field="source"
                )
            r_p = tuple(target_bones[parent]["rest_rotation"])
            w_p = world[parent]
            world[name] = _mul_all(w_p, _conj(r_p), r_b)
    return basis


# --- loop closing, resampling, key reduction --------------------------------


def _fix_signs(values: list[_Quat]) -> None:
    """Flip each quaternion, in place, so it lands on the same hemisphere as
    the one before it -- sign continuity across a frame sequence."""
    for i in range(1, len(values)):
        a, b = values[i - 1], values[i]
        if a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] < 0.0:
            values[i] = (-b[0], -b[1], -b[2], -b[3])


def _close_loop(
    bone_names: Sequence[str],
    bones: list[dict[str, _Quat]],
    roots: list[_Vec3] | None,
) -> None:
    """Distribute the end-to-start residual linearly across every frame, in
    place, so the sequence loops with no seam -- ``loop="on"``.

    Frame 0 is untouched (weight 0) and the last frame is rotated/moved by
    ``(n-1)/n`` of the residual needed to land on frame 0's own value --
    one step short of it, not the full residual (weight 1.0). Landing
    exactly on frame 0 would make the last frame a duplicate of it, and
    since the clip is then resampled as closed (wrapping last back to
    frame 0), that duplicate turns the whole wrap segment into a held
    frame: a 1/n-of-the-cycle hitch with no motion, once per loop. Weight
    ``(n-1)/n`` instead leaves the last frame carrying its own step, so the
    wrap segment moves by as much as any other one does (the 2026-09-13
    review; see ``tests.test_cliptransfer.
    test_loop_on_leaves_no_held_frame_at_the_seam``).
    """
    n = len(bones)
    if n < 2:
        return
    for bone in bone_names:
        first = bones[0][bone]
        last = bones[-1][bone]
        residual = _mul(first, _conj(last))  # rotates "last" onto "first"
        for i in range(1, n):
            t = i / n
            corrected = _slerp(_IDENTITY_Q, residual, t)
            bones[i][bone] = tuple(_mul(corrected, bones[i][bone]))
    if roots is not None:
        drift = _sub3(roots[0], roots[-1])  # what "last" needs added to become "first"
        for i in range(1, n):
            t = i / n
            r = roots[i]
            roots[i] = (r[0] + drift[0] * t, r[1] + drift[1] * t, r[2] + drift[2] * t)


def _resample(
    bone_names: Sequence[str],
    bones: list[dict[str, _Quat]],
    roots: list[_Vec3] | None,
    frames: int,
    closed: bool,
) -> tuple[list[dict[str, _Quat]], list[_Vec3] | None]:
    """Sample ``frames`` normalized times across ``bones``/``roots``.

    The phase rule is ``kernels.sheet.resample_clip``'s, restated: a
    cycle samples ``i / frames`` (so it never lands a duplicate on the seam)
    and a one-shot samples ``i / (frames - 1)`` (so it lands on both
    endpoints), with every original frame treated as one equal-length
    segment -- the uniform case of that function's per-segment weighting.
    """
    m = len(bones)
    out_bones: list[dict[str, _Quat]] = []
    out_roots: list[_Vec3] | None = [] if roots is not None else None
    total = float(m if closed else m - 1)
    for index in range(frames):
        phase = index / frames if closed else (index / (frames - 1) if frames > 1 else 0.0)
        position = phase * total
        if not closed and phase >= 1.0:
            a_idx = b_idx = m - 1
            local = 0.0
        else:
            a_idx = int(position)
            if a_idx >= m:
                a_idx = m - 1
            b_idx = (a_idx + 1) % m
            local = position - a_idx
        out_bones.append(
            {
                bone: tuple(_slerp(bones[a_idx][bone], bones[b_idx][bone], local))
                for bone in bone_names
            }
        )
        if out_roots is not None:
            out_roots.append(_lerp3(roots[a_idx], roots[b_idx], local))
    return out_bones, out_roots


def _rdp(
    bone_names: Sequence[str],
    bones: Sequence[Mapping[str, _Quat]],
    roots: Sequence[_Vec3] | None,
    n: int,
    i0: int,
    i1: int,
    keep: set[int],
) -> None:
    """Ramer-Douglas-Peucker over a discrete pose sequence.

    ``i1`` may equal ``n`` for a closed clip's wrap segment (from the last
    kept key back around to frame 0); every index is read modulo ``n``, so
    that virtual endpoint is frame 0 itself.
    """
    if i1 - i0 <= 1:
        return
    span = i1 - i0
    a = bones[i0 % n]
    b = bones[i1 % n]
    worst_ratio = 0.0
    worst_idx = -1
    for k in range(i0 + 1, i1):
        t = (k - i0) / span
        actual = bones[k % n]
        ratio = 0.0
        for bone in bone_names:
            predicted = _slerp(a[bone], b[bone], t)
            ratio = max(ratio, _angle_deg(predicted, actual[bone]) / KEY_TOLERANCE_DEG)
        if roots is not None:
            predicted_root = _lerp3(roots[i0 % n], roots[i1 % n], t)
            ratio = max(ratio, _dist3(predicted_root, roots[k % n]) / ROOT_TOLERANCE)
        if ratio > worst_ratio:
            worst_ratio, worst_idx = ratio, k
    if worst_ratio > 1.0:
        keep.add(worst_idx % n)
        _rdp(bone_names, bones, roots, n, i0, worst_idx, keep)
        _rdp(bone_names, bones, roots, n, worst_idx, i1, keep)


# --- naming ------------------------------------------------------------

_JUNK_SEGMENTS = re.compile(r"^(mixamo\.com|take\s*\d+|armature\d*|layer\d*)$", re.IGNORECASE)


def _slug_from_action_name(name: str) -> str:
    """A legal clip name out of whatever an exporter called the action.

    Mixamo-style names arrive as e.g. ``Armature|mixamo.com|Layer0`` (pipe
    -separated, most of it boilerplate the exporter added) or a bare
    ``Take 001``. The boilerplate segments are dropped and whatever is left
    is slugged; an action that is *only* boilerplate falls back to
    ``"clip"`` rather than producing an empty name.
    """
    segments = [s for s in str(name or "").split("|") if s.strip()]
    kept = [s for s in segments if not _JUNK_SEGMENTS.match(s.strip())]
    base = kept[-1] if kept else (segments[-1] if segments else "clip")
    slug = re.sub(r"[^a-z0-9]+", "_", base.strip().lower()).strip("_")
    return slug or "clip"


def _validate_clip_name(name: str) -> str:
    name = str(name or "").strip()
    if not name:
        raise ClipTransferError("a clip needs a name", field="name")
    try:
        cliplib.reject_direction_named_clip(name)
    except ValueError as exc:
        raise ClipTransferError(str(exc), field="name") from exc
    return name


# --- the public API ----------------------------------------------------

_LOOP_MODES = ("auto", "on", "off")
_ROOT_MODES = ("in_place", "keep", "none")


def transfer(
    sample: Mapping[str, Any],
    *,
    template: str,
    clip_name: str | None = None,
    frames: int | None = None,
    loop: str = "auto",
    root_motion: str = "in_place",
    duration_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Convert every sampled action in ``sample`` onto ``template``.

    One entry per action: ``{"clip", "poses", "report"}`` -- see the module
    docstring for the shape and what is deliberately left for the caller
    (``source`` on the clip, ``provisional``). Raises
    :class:`ClipTransferError` naming the refusing field; see the class
    docstring.
    """
    if loop not in _LOOP_MODES:
        raise ClipTransferError(f"loop must be one of {_LOOP_MODES}, not {loop!r}", field="loop")
    if root_motion not in _ROOT_MODES:
        raise ClipTransferError(
            f"root_motion must be one of {_ROOT_MODES}, not {root_motion!r}", field="root_motion"
        )
    if frames is not None and not (1 <= int(frames) <= MAX_CLIP_FRAMES):
        raise ClipTransferError(
            f"frames must be 1-{MAX_CLIP_FRAMES}, not {frames}", field="frames"
        )
    # The 2026-09-15 audit, finding poser-03: an explicit ``clip_name`` used
    # to be handed to *every* sampled action unchanged, so a multi-action file
    # produced several clips sharing one name -- ``import_into_library``'s own
    # collision check then either raised "already exists" on the second one
    # (misreporting a name clash the caller never asked for) or, with
    # ``replace=True``, silently dropped every action but the last. An
    # explicit name only ever makes sense for a single-action sample; refused
    # here, before any of them are converted, rather than left for the door
    # three calls up to misdiagnose.
    if clip_name is not None and len(sample["actions"]) > 1:
        raise ClipTransferError(
            "clip_name names one clip, but this file sampled "
            f"{len(sample['actions'])} actions -- name each action instead, "
            "or drop clip_name and let the action names supply one each",
            field="clip_name",
        )
    try:
        templates.get_template(template)
    except ValueError as exc:
        raise ClipTransferError(str(exc), field="source") from exc

    try:
        match_result = clipmaps.match(sample["all_bone_names"], template=template)
    except clipmaps.ClipMapError as exc:
        raise ClipTransferError(str(exc), field="source") from exc

    source_bones = sample["source_bones"]
    target_bones = sample["target"]["bones"]
    resolved = match_result.resolved

    c = _facing_alignment(source_bones, resolved)
    c_inv = _conj(c)
    geometry = {
        name: (
            tuple(source_bones[chain[-1]]["rest_rotation"]),
            _rest_alignment(source_bones, target_bones, name, chain, c),
        )
        for name, chain in resolved.items()
    }
    order = _topo_order(target_bones)
    bone_names = tuple(resolved)
    height = _root_height(source_bones, resolved["head"], c)
    hips_chain = resolved["hips"]
    hips_rest = source_bones[hips_chain[-1]]["head"]

    out: list[dict[str, Any]] = []
    for action in sample["actions"]:
        out.append(
            _transfer_action(
                action,
                order=order,
                target_bones=target_bones,
                resolved=resolved,
                geometry=geometry,
                c=c,
                c_inv=c_inv,
                bone_names=bone_names,
                height=height,
                hips_chain=hips_chain,
                hips_rest=hips_rest,
                clip_name=clip_name,
                frames=frames,
                loop=loop,
                root_motion=root_motion,
                duration_ms=duration_ms,
                map_key=match_result.clip_map.key,
                left_at_rest=match_result.left_at_rest,
                ignored=match_result.ignored,
                duplicate_source_names=match_result.duplicate_source_names,
            )
        )
    return out


def _transfer_action(
    action: Mapping[str, Any],
    *,
    order: Sequence[str],
    target_bones: Mapping[str, Any],
    resolved: Mapping[str, tuple[str, ...]],
    geometry: Mapping[str, Any],
    c: Sequence[float],
    c_inv: Sequence[float],
    bone_names: Sequence[str],
    height: float,
    hips_chain: tuple[str, ...],
    hips_rest: Sequence[float],
    clip_name: str | None,
    frames: int | None,
    loop: str,
    root_motion: str,
    duration_ms: int | None,
    map_key: str,
    left_at_rest: tuple[str, ...],
    ignored: tuple[str, ...],
    duplicate_source_names: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    src_frames = list(action["frames"])
    if len(src_frames) < 1:
        raise ClipTransferError("this action has no sampled frames", field="source")
    fps = float(action.get("fps") or 30.0) or 30.0

    # Step 4-5: per original frame, every mapped bone's basis and the root's
    # world-space offset (always computed, even for root_motion="none",
    # because loop auto-detection reads the root's own bob to decide whether
    # the clip already cycles).
    per_frame_bones: list[dict[str, _Quat]] = []
    per_frame_roots: list[_Vec3] = []
    for frame in src_frames:
        frame_bones = frame.get("bones") or {}
        per_frame_bones.append(
            _pose_frame(order, target_bones, resolved, geometry, c, c_inv, frame_bones)
        )
        hips_entry = frame_bones.get(hips_chain[-1])
        hips_head = hips_entry["head"] if hips_entry is not None else hips_rest
        offset_world = _rotate_vector(c, _sub3(hips_head, hips_rest))
        per_frame_roots.append(tuple(v / height for v in offset_world))

    for bone in bone_names:
        # _fix_signs mutates a list of tuples in place; write the result
        # back onto each frame's dict afterward.
        values = [f[bone] for f in per_frame_bones]
        _fix_signs(values)
        for f, v in zip(per_frame_bones, values, strict=True):
            f[bone] = v

    residual_deg = max(
        (_angle_deg(per_frame_bones[0][b], per_frame_bones[-1][b]) for b in bone_names), default=0.0
    )
    root_z_delta = abs(per_frame_roots[-1][2] - per_frame_roots[0][2])

    closed = loop == "on" or (
        loop == "auto"
        and len(src_frames) > 1
        and residual_deg <= LOOP_MATCH_DEG
        and root_z_delta <= LOOP_MATCH_ROOT_Z
    )
    if loop == "auto" and closed and len(per_frame_bones) > 1:
        # The would-be duplicate closing sample: drop it, matching the seam
        # rule ``sheet.interpolate``'s own docstring states for a two-key
        # cycle.
        per_frame_bones = per_frame_bones[:-1]
        per_frame_roots = per_frame_roots[:-1]
    elif loop == "on":
        _close_loop(bone_names, per_frame_bones, per_frame_roots)

    if root_motion == "keep":
        peak = max(abs(v) for r in per_frame_roots for v in r)
        if peak > MAX_ROOT_TRANSLATION:
            raise ClipTransferError(
                f"root motion peaks at {peak:.3f} character heights, over the "
                f"+/-{MAX_ROOT_TRANSLATION} limit",
                field="root_motion",
            )
    elif root_motion == "in_place":
        first_xy = per_frame_roots[0][:2]
        last_xy = per_frame_roots[-1][:2]
        drift = (last_xy[0] - first_xy[0], last_xy[1] - first_xy[1])
        n = len(per_frame_roots)
        for i, r in enumerate(per_frame_roots):
            t = i / (n - 1) if n > 1 else 0.0
            per_frame_roots[i] = (r[0] - drift[0] * t, r[1] - drift[1] * t, r[2])

    # Step 8: resample to N frames.
    frame_end = action.get("frame_end", len(src_frames) - 1)
    duration_s = float(frame_end - action.get("frame_start", 0)) / fps
    n_frames = (
        frames if frames is not None else min(MAX_CLIP_FRAMES, max(2, round(duration_s * 15)))
    )
    if n_frames > MAX_CLIP_FRAMES:
        raise ClipTransferError(
            f"frames must be 1-{MAX_CLIP_FRAMES}, not {n_frames}", field="frames"
        )
    resampled_bones, resampled_roots = _resample(
        bone_names, per_frame_bones, per_frame_roots, n_frames, closed
    )

    proposed = clip_name if clip_name is not None else _slug_from_action_name(action.get("name"))
    name = _validate_clip_name(proposed)

    # Step 9: key reduction.
    keep = {0}
    if closed:
        _rdp(bone_names, resampled_bones, resampled_roots, n_frames, 0, n_frames, keep)
    else:
        keep.add(n_frames - 1)
        if n_frames > 1:
            _rdp(bone_names, resampled_bones, resampled_roots, n_frames, 0, n_frames - 1, keep)
    kept = sorted(keep)
    if closed:
        segments = [kept[i + 1] - kept[i] for i in range(len(kept) - 1)] + [n_frames - kept[-1]]
    else:
        segments = [kept[i + 1] - kept[i] for i in range(len(kept) - 1)]

    with_root = root_motion != "none"
    static_bones = {
        bone
        for bone in bone_names
        if max(_angle_deg(f[bone], _IDENTITY_Q) for f in resampled_bones) <= BONE_IDENTITY_DEG
    }
    key_names: list[str] = []
    poses: dict[str, Any] = {}
    for i, idx in enumerate(kept):
        key = f"{name} k{i:02d}"
        key_names.append(key)
        pose: dict[str, Any] = {
            "bones": {
                bone: list(resampled_bones[idx][bone])
                for bone in bone_names
                if bone not in static_bones
            }
        }
        if with_root:
            pose["root_translation"] = list(resampled_roots[idx])
        poses[key] = pose

    duration = duration_ms
    if duration is None:
        step = cliplib.CLIP_DURATION_STEP_MS
        # ``duration_ms`` is the time PER RENDERED FRAME, not the source
        # clip's own length (``clips.animation_tracks``'s ``step =
        # ANIMATION_FPS * duration_ms / 1000``, one hop per authored frame,
        # and ``charsheet``'s per-cell durations are the same idea). A
        # closed (looping) clip plays its ``n_frames`` samples and then
        # wraps the last back to the first -- ``n_frames`` hops per cycle --
        # while an open one plays them once through, ``n_frames - 1`` hops
        # with the last frame held. Divide the *source* clip's own span
        # (``duration_s``, from its authored frame range, not resampled
        # frame count) by however many hops the rendered clip makes, so the
        # resampled clip lasts as long as the source did instead of
        # ``(n_frames / fps) * 1000`` -- roughly the whole source clip's
        # length again -- which used to make an imported clip play
        # ``n_frames`` times too slowly (the 2026-09-13 review; see
        # ``tests.test_cliptransfer.test_an_imported_clip_keeps_its_source_length``).
        hops = n_frames if closed else max(1, n_frames - 1)
        raw = (duration_s * 1000.0) / hops
        rounded = round(raw / step) * step
        clamped = max(cliplib.MIN_CLIP_DURATION_MS, min(cliplib.MAX_CLIP_DURATION_MS, rounded))
        duration = int(clamped)
    try:
        duration = cliplib.validate_clip_duration_ms(duration, name)
    except ValueError as exc:
        raise ClipTransferError(str(exc), field="frames") from exc

    clip: dict[str, Any] = {
        "name": name,
        "keys": key_names,
        "segments": segments,
        "closed": closed,
        "easing": "linear",
        "duration_ms": duration,
    }
    report = {
        "map": map_key,
        "left_at_rest": list(left_at_rest),
        "ignored": list(ignored),
        # The 2026-09-16 audit: the 2026-09-15 fix (finding poser-04) taught
        # clipmaps.MatchResult to record a normalized name more than one raw
        # source bone collapsed onto, but nothing between clipmaps.match and
        # this report ever read the field -- so a colliding duplicate still
        # vanished with no trace once it reached "Import clip"'s own report,
        # one call frame further downstream of the fix. Surfaced the same way
        # ``ignored`` already is.
        "duplicate_source_names": {
            name: list(raw_names) for name, raw_names in duplicate_source_names.items()
        },
        "loop": {"closed": closed, "residual_deg": residual_deg},
        "frames": n_frames,
        "keys": len(kept),
        "root_motion": root_motion,
    }
    return {"clip": clip, "poses": poses, "report": report}
