"""A painted region on a mesh, as a mask over its UV atlas.

Masked texture repair paints a patch of *surface* in the viewport and has to hand
the worker a patch of *texture*: which texels of ``model.glb``'s base-colour atlas
belong to the faces the brush touched. This module is that translation, and it is
pure numpy -- no GL, no Blender -- so every claim is testable on a hand-built
quad.

Four steps, each a function:

* :func:`flatten` -- a loaded ``gltf.Model`` as one set of world-space positions,
  triangles and UVs (the viewer's own loader, so a triangle the brush picked is
  the triangle the mask is rasterised from by construction);
* :func:`faces_in_brush` -- which triangles lie within a radius of a picked point;
* :func:`rasterize_uv` -- those triangles' UV footprint as an ``L`` image at the
  atlas's own size, vectorised so a 300k-face mesh is not 300k Python iterations;
* :func:`feather_inside` -- a 0..1 weight that ramps up from the mask's edge and is
  exactly 0 outside it, which is what lets the repair leave every untouched texel
  byte-for-byte what it was.

The V axis is **not flipped**: glTF's texture origin is the image's top-left, so
a texel's row is ``v * height`` and a mask drawn here lines up with the atlas as
``retexture.extract_base_colour`` writes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Triangles are rasterised in batches grouped by the (power of two) size of their
#: pixel bounding box, so each batch is one broadcast over an ``(n, s, s)`` grid.
#: A triangle bigger than this goes through the one-at-a-time path -- there are
#: few, and their grids are large.
MAX_BATCH_SIDE = 64
#: Elements per batch grid, so a chunk of small triangles never allocates more
#: than a few tens of MB however many faces the mesh has.
BATCH_ELEMENTS = 4_000_000


@dataclass(frozen=True, slots=True)
class Flat:
    """The whole model as one indexed triangle soup in world space."""

    positions: np.ndarray  # (V, 3) f8, node transforms applied
    tris: np.ndarray  # (T, 3) i8
    uv: np.ndarray | None  # (V, 2) f8, or None if any primitive has no UVs


def flatten(model: Any) -> Flat:
    """Every mesh instance of ``model`` joined into one :class:`Flat`.

    World space through each node's ``world`` matrix, so a brush radius in
    metres means the same thing on a mesh with a grounding transform as on one
    without. ``uv`` is None when any primitive lacks UVs: a repair needs the
    whole atlas or nothing.
    """
    model.update_world()
    positions, tris, uvs = [], [], []
    offset = 0
    complete = True
    for node, primitives in model.mesh_instances():
        matrix = np.asarray(node.world, dtype=np.float64)
        for primitive in primitives:
            pos = np.asarray(primitive.positions, dtype=np.float64)
            positions.append(pos @ matrix[:3, :3].T + matrix[:3, 3])
            tris.append(np.asarray(primitive.indices, dtype=np.int64).reshape(-1, 3) + offset)
            if primitive.uvs is None:
                complete = False
                uvs.append(np.zeros((len(pos), 2)))
            else:
                uvs.append(np.asarray(primitive.uvs, dtype=np.float64))
            offset += len(pos)
    if not positions:
        return Flat(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), None)
    return Flat(
        np.concatenate(positions),
        np.concatenate(tris),
        np.concatenate(uvs) if complete else None,
    )


def centroids(flat: Flat) -> np.ndarray:
    """Each triangle's centre, ``(T, 3)``."""
    return flat.positions[flat.tris].mean(axis=1)


def faces_in_brush(centres: np.ndarray, point: Any, radius: float) -> np.ndarray:
    """Indices of the triangles whose centre is within ``radius`` of ``point``.

    Centre distance rather than any-vertex-inside: a brush should select what its
    disc visibly covers, and a long thin triangle with one vertex under the brush
    and the other two a metre away is not under it.
    """
    if radius <= 0 or len(centres) == 0:
        return np.zeros(0, dtype=np.int64)
    delta = centres - np.asarray(point, dtype=np.float64)
    return np.nonzero(np.einsum("ij,ij->i", delta, delta) <= radius * radius)[0]


def rasterize_uv(
    uv: np.ndarray, tris: np.ndarray, faces: np.ndarray | None, size: int
) -> np.ndarray:
    """The UV footprint of ``faces`` as a ``(size, size)`` uint8 mask (255 inside).

    A texel is inside a triangle when its **centre** is, which is the rule a GPU
    rasteriser follows and so the rule that decides which texels a face actually
    samples. ``faces=None`` means every triangle. Triangles that fall outside
    [0, 1] are clipped to the image, never wrapped.
    """
    out = np.zeros((size, size), dtype=np.uint8)
    selected = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if faces is not None:
        selected = selected[np.asarray(faces, dtype=np.int64)]
    if len(selected) == 0:
        return out
    corner = np.asarray(uv, dtype=np.float64)[selected] * size  # (n, 3, 2) in texels
    lo = np.floor(corner.min(axis=1)).astype(np.int64)
    hi = np.ceil(corner.max(axis=1)).astype(np.int64)
    lo = np.clip(lo, 0, size)
    hi = np.clip(hi, 0, size)
    side = np.maximum((hi - lo).max(axis=1), 1)
    keep = (hi > lo).all(axis=1)
    corner, lo, side = corner[keep], lo[keep], side[keep]

    cap = 1
    classes = []
    while cap < MAX_BATCH_SIDE:
        cap *= 2
        classes.append(cap)
    done = np.zeros(len(side), dtype=bool)
    for edge in classes:
        members = np.nonzero(~done & (side <= edge))[0]
        done[members] = True
        step = max(1, BATCH_ELEMENTS // (edge * edge))
        for start in range(0, len(members), step):
            chunk = members[start : start + step]
            _stamp(out, corner[chunk], lo[chunk], edge)
    for index in np.nonzero(~done)[0]:
        big = int(side[index])
        _stamp(out, corner[index : index + 1], lo[index : index + 1], big)
    return out


def _stamp(out: np.ndarray, corner: np.ndarray, lo: np.ndarray, edge: int) -> None:
    """Rasterise a batch whose bounding boxes all fit an ``edge`` x ``edge`` grid."""
    size = out.shape[0]
    grid = np.arange(edge, dtype=np.int64)
    xs = lo[:, 0, None] + grid[None, :]  # (n, edge) texel column indices
    ys = lo[:, 1, None] + grid[None, :]
    px = xs[:, None, :] + 0.5  # (n, 1, edge) texel centres
    py = ys[:, :, None] + 0.5  # (n, edge, 1)
    a, b, c = corner[:, 0], corner[:, 1], corner[:, 2]
    denom = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    live = np.abs(denom) > 1e-12
    denom = np.where(live, denom, 1.0)
    w0 = (
        (b[:, 1] - c[:, 1])[:, None, None] * (px - c[:, 0, None, None])
        + (c[:, 0] - b[:, 0])[:, None, None] * (py - c[:, 1, None, None])
    ) / denom[:, None, None]
    w1 = (
        (c[:, 1] - a[:, 1])[:, None, None] * (px - c[:, 0, None, None])
        + (a[:, 0] - c[:, 0])[:, None, None] * (py - c[:, 1, None, None])
    ) / denom[:, None, None]
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9) & live[:, None, None]
    inside &= (xs[:, None, :] < size) & (ys[:, :, None] < size)
    n_index, y_index, x_index = np.nonzero(inside)
    out[ys[n_index, y_index], xs[n_index, x_index]] = 255


def grow(mask: np.ndarray, px: int) -> np.ndarray:
    """``mask`` dilated by ``px`` texels (4-neighbourhood, never wrapped)."""
    out = mask > 0
    for _ in range(max(0, int(px))):
        step = out.copy()
        step[1:] |= out[:-1]
        step[:-1] |= out[1:]
        step[:, 1:] |= out[:, :-1]
        step[:, :-1] |= out[:, 1:]
        out = step
    return (out * 255).astype(np.uint8)


def coverage(mask: np.ndarray) -> float:
    """The share of the atlas the mask covers, 0..1."""
    return float((mask > 0).mean()) if mask.size else 0.0


def feather_inside(mask: np.ndarray, px: int) -> np.ndarray:
    """A float32 weight in 0..1 that is **exactly zero wherever ``mask`` is**.

    The mask is blurred with a box ``px`` texels each way (separable, via a
    running sum, so it is linear in the atlas size) and then multiplied by the
    mask itself: weight climbs from about a half at the region's edge to one a
    few texels in, and nothing outside the region is ever non-zero. That last
    property is the contract -- a repair blends with this weight, so a texel
    outside the mask is not *nearly* untouched, it is untouched.
    """
    inside = (mask > 0).astype(np.float32)
    if px <= 0 or not inside.any():
        return inside
    blurred = _box_blur(_box_blur(inside, px, axis=0), px, axis=1)
    return (blurred * inside).astype(np.float32)


def _box_blur(values: np.ndarray, radius: int, *, axis: int) -> np.ndarray:
    pad = [(0, 0), (0, 0)]
    pad[axis] = (radius + 1, radius)
    padded = np.pad(values, pad, mode="constant")
    total = np.cumsum(padded, axis=axis, dtype=np.float64)
    width = 2 * radius + 1
    upper = np.take(total, np.arange(width, total.shape[axis]), axis=axis)
    lower = np.take(total, np.arange(0, total.shape[axis] - width), axis=axis)
    return ((upper - lower) / width).astype(np.float32)
