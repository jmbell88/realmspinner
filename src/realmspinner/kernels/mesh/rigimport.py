"""A skinned GLB into a Clay document: bones, rigid parts and weighted meshes.

The other half of :func:`~.document.to_model`'s skinned export, and written to read what
Blender writes as well as what Clay does. :mod:`.glbimport` hands a model with skins here
instead of refusing it.

**Every joint node becomes a bone object**, parented to the nearest joint above it (any
non-joint nodes between two joints -- Blender's "Armature" empty, an exporter's scale node --
are folded into the placement, because Clay's outliner is a tree of objects and not of every
node a file happens to have). A bone's transform is its world matrix relative to its parent
bone, so a rig's rest pose is exactly the node transforms the file displays. A bone with shear
in that relative matrix, or with no inverse, is refused by name: nothing downstream could
place it.

**A mesh keeps one of three shapes**, and which one is decided from the weights, not from the
file's wording:

* a primitive whose every vertex has one influence, at weight 1, on the same joint is a
  **rigid part**: a child of that joint's bone, its vertices re-expressed in the bone's frame,
  at an identity transform. That is what Clay's own export writes for a shape under a bone, so
  a rigid export reopens as the structure it was. A part named like its joint, when the bone
  has no geometry of its own yet, is folded into the bone (the export writes a bone's own mesh
  under the bone's own name).
* anything else with weights is a **weighted mesh**: an object at the identity with a
  :class:`~.rig.Skin` naming its joints' bones and per-vertex ``joints``/``weights``.
* a mesh with no skin is imported exactly as before -- except that, in a file that has a
  skeleton, an unskinned mesh on a joint node or under one moves with that bone in glTF, and
  so becomes a child of it here.

**Vertices are posed, not taken raw.** A skinned vertex is placed by ``sum(w * joint_world *
inverse_bind)``; when the file's pose is its bind pose that is the identity and the positions
are used as they are, and when it is not (a file saved mid-animation) the mesh is baked at the
pose the bones are imported in, which is the only placement consistent with them. The skin's
inverse bind matrices are then ``inv(bone world)`` -- for a file at rest exactly what it
carried.

**Vertices are merged bitwise on position *and* weights** for a weighted mesh, so two points
that share a position but are weighted differently stay two points.

More than :data:`~.rig.MAX_BONES` joints across the file's skins is refused by name.

**Animations come in as keys on the bones** (:func:`import_animations`). A channel on a joint
becomes a channel on that bone; a channel on anything else is counted and dropped. glTF keys are
local to the joint's glTF parent, and a bone here is local to its parent *bone*, so when a
non-joint node (Blender's Armature empty) sits between them the keys are carried across it:
sampled at the union of the joint's key times, composed, multiplied by the fixed matrix between
the two frames and decomposed again. A ``CUBICSPLINE`` channel is resampled to linear keys and
the notice says so; ``STEP`` and ``LINEAR`` import as they are. The frame rate is guessed from
the smallest gap between keys and rounded to the nearest of 12, 24, 30 and 60. More than 64
animations, a channel past 4,096 keys or a clip longer than 600 seconds is refused by name: the
caps are Clay's, and trimming a clip silently would be worse than saying so.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..geom3d import gltf
from ..geom3d import math3d as m3
from . import anim, rig, topo
from . import mesh as bm
from .document import Obj, _empty_mesh, new_uid
from .elements import OpError

__all__ = ["import_animations", "import_rigged"]

#: How near the identity a joint's ``world @ inverse_bind`` must be for a file to count as
#: being at its bind pose, so the positions can be used as they are.
_BIND_TOLERANCE = 1e-5

#: Vertices posed per chunk, so a two-million-vertex skinned mesh does not allocate a 4x4
#: matrix per vertex all at once.
_POSE_CHUNK = 100_000

#: A vertex whose first influence is at least this heavy, with the rest at most its
#: complement, counts as following one joint alone.
_RIGID_WEIGHT = 0.999


def import_rigged(
    model: gltf.Model,
    base: str,
    materials: list[gltf.Material],
    palette: dict[int, int],
    placed: list[gltf.Node],
    *,
    material_index: Callable[..., int],
    unique: Callable[[str, set[str]], str],
    plain_object: Callable[..., Obj | None],
) -> tuple[list[Obj], anim.RigState, tuple[str, ...]]:
    """``(objects, animations, notices)`` of a model that has skins: bones first, then every
    placed mesh, then the keys.

    The three callables are :mod:`.glbimport`'s own (its palette lookup, its unique-name rule
    and its unskinned object builder), passed in so this module owes that one nothing at
    import time.
    """
    importer = _Importer(model, base, materials, palette, material_index, unique, plain_object)
    objects: list[Obj] = list(importer.bones)
    for node in placed:
        for prim in model.meshes[node.mesh]:
            made = importer.geometry(node, prim)
            if made is not None:
                objects.append(made)
    state, notices = import_animations(importer)
    return objects, state, notices


class _Importer:
    def __init__(self, model, base, materials, palette, material_index, unique, plain_object):
        self.model = model
        self.base = base
        self.materials = materials
        self.palette = palette
        self.material_index = material_index
        self.unique = unique
        self.plain_object = plain_object
        self.taken: set[str] = set()
        self.index_of = {id(node): i for i, node in enumerate(model.nodes)}
        self.parents = _parents(model)
        self.joints = _joint_nodes(model)
        self.bones: list[Obj] = []
        self.bone_of_node: dict[int, Obj] = {}
        #: A bone's world matrix by uid -- the pose it was imported in.
        self.world_of: dict[int, np.ndarray] = {}
        self._make_bones()

    # -- the skeleton -----------------------------------------------------------------------

    def _make_bones(self) -> None:
        joint_set = set(self.joints)
        order = sorted(self.joints, key=lambda i: (self._chain_length(i, joint_set), i))
        for index in order:
            node = self.model.nodes[index]
            label = f"The bone {node.name or index!r}"
            above = _joint_above(index, joint_set, self.parents)
            parent = self.bone_of_node[above] if above is not None else None
            parent_world = self.model.nodes[above].world if above is not None else None
            t, r, s = _relative(node.world, parent_world, label)
            bone = Obj(
                uid=new_uid(),
                name=self.unique(node.name or "Bone", self.taken),
                mesh=_empty_mesh(),
                translation=t,
                rotation=r,
                scale=s,
                parent=parent.uid if parent is not None else None,
                bone=True,
            )
            self.bone_of_node[index] = bone
            self.world_of[bone.uid] = np.asarray(node.world, dtype="f8")
            self.bones.append(bone)

    def frame_change(self, node_index: int) -> np.ndarray | None:
        """The fixed matrix from a joint's glTF parent frame to its bone's parent frame, or
        ``None`` when they are the same (the joint's parent in the file is its parent bone, or
        both are the world)."""
        bone = self.bone_of_node[node_index]
        above = self.parents.get(node_index)
        gltf_parent = (
            np.eye(4) if above is None else np.asarray(self.model.nodes[above].world, dtype="f8")
        )
        clay_parent = np.eye(4) if bone.parent is None else self.world_of[bone.parent]
        change = np.linalg.inv(clay_parent) @ gltf_parent
        return None if np.allclose(change, np.eye(4), atol=1e-9) else change

    def _chain_length(self, index: int, joint_set: set[int]) -> int:
        """How many joints sit above *index*, so bones are created parents first."""
        count = 0
        cursor = _joint_above(index, joint_set, self.parents)
        while cursor is not None:
            count += 1
            cursor = _joint_above(cursor, joint_set, self.parents)
        return count

    # -- the meshes -------------------------------------------------------------------------

    def geometry(self, node: gltf.Node, prim: gltf.Primitive) -> Obj | None:
        if len(prim.indices) < 3 or len(prim.positions) == 0:
            return None
        if node.skin is not None and prim.joints is not None and prim.weights is not None:
            return self._skinned(node, prim)
        home = _joint_at_or_above(self.index_of[id(node)], self.bone_of_node, self.parents)
        if home is None:
            return self.plain_object(
                prim, node, self.base, self.materials, self.palette, self.taken
            )
        # An unskinned mesh on a joint, or under one, moves with that bone in glTF.
        bone = self.bone_of_node[home]
        label = f"The mesh {node.name or self.base!r}"
        t, r, s = _relative(node.world, self.model.nodes[home].world, label)
        slot = self.material_index(prim.material, self.materials, self.palette)
        return Obj(
            uid=new_uid(),
            name=self.unique(node.name or self.base, self.taken),
            mesh=_plain_mesh(prim, slot),
            translation=t,
            rotation=r,
            scale=s,
            parent=bone.uid,
            material=slot,
        )

    def _skinned(self, node: gltf.Node, prim: gltf.Primitive) -> Obj | None:
        skin = self.model.skins[node.skin]
        skin_bones = [self.bone_of_node[int(j)] for j in skin.joints]
        positions, normals = _posed(self.model, skin, prim)
        slot = self.material_index(prim.material, self.materials, self.palette)
        joints = np.asarray(prim.joints, dtype="i4").reshape(-1, 4)
        weights = _normalised(np.asarray(prim.weights, dtype="f8").reshape(-1, 4))
        label = node.name or self.base

        alone = _rigid_joint(joints, weights)
        if alone is not None:
            bone = skin_bones[alone]
            local, local_normals = _into_frame(
                positions, normals, np.linalg.inv(self.world_of[bone.uid])
            )
            mesh = _plain_mesh(_primitive(prim, local, local_normals), slot)
            if len(bone.mesh.positions) == 0 and bone.name == label:
                bone.mesh, bone.material = mesh, slot
                return None
            return Obj(
                uid=new_uid(),
                name=self.unique(label, self.taken),
                mesh=mesh,
                parent=bone.uid,
                material=slot,
            )

        mesh = _weighted_mesh(prim, positions, normals, joints, weights, slot)
        inverse = np.array([np.linalg.inv(self.world_of[b.uid]) for b in skin_bones])
        try:
            skin_data = rig.Skin(tuple(b.uid for b in skin_bones), inverse)
        except ValueError as exc:
            raise OpError(f"The skin of {label!r} cannot be imported: {exc}") from exc
        return Obj(
            uid=new_uid(),
            name=self.unique(label, self.taken),
            mesh=mesh,
            material=slot,
            skin=skin_data,
        )


# -- animations -----------------------------------------------------------------------------


def _guess_fps(model: gltf.Model) -> int:
    """The nearest of Clay's frame rates to the smallest gap between any two keys."""
    smallest = np.inf
    for animation in model.animations:
        for channel in animation.channels:
            if len(channel.times) > 1:
                gap = float(np.diff(np.sort(channel.times)).min()) if len(channel.times) > 1 else 0
                if gap > 1e-6:
                    smallest = min(smallest, gap)
    if not np.isfinite(smallest):
        return 24
    wanted = 1.0 / smallest
    return min(anim.FPS_CHOICES, key=lambda fps: abs(np.log(fps / wanted)))


def _linear_keys(channel: gltf.AnimChannel, fps: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """``(times, values, was_cubic)``: the channel as keys Clay can hold, ascending and unique."""
    width = 4 if channel.path == "rotation" else 3
    cubic = channel.interpolation == "CUBICSPLINE"
    if cubic:
        times, values = anim.resample_cubic(channel.times, channel.values, fps=fps, width=width)
    else:
        times, values = np.asarray(channel.times), np.asarray(channel.values)
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    keep = np.ones(len(times), dtype=bool)
    keep[:-1] = np.diff(times) > anim.TIME_EPSILON  # a repeated time keeps its last key
    return times[keep], values[keep], cubic


def _fresh_name(name: str, taken: set[str]) -> str:
    base = name.strip() or "Animation"
    candidate, n = base, 0
    while candidate in taken:
        n += 1
        candidate = f"{base}.{n:03d}"
    taken.add(candidate)
    return candidate


def import_animations(importer: _Importer) -> tuple[anim.RigState, tuple[str, ...]]:
    """The file's animations as Clay's, on the bones :class:`_Importer` made. See the module
    docstring for what is carried, what is resampled and what is refused."""
    model = importer.model
    if not model.animations:
        return anim.RigState(), ()
    if len(model.animations) > anim.MAX_ANIMATIONS:
        raise OpError(
            f"This GLB has {len(model.animations)} animations, past the {anim.MAX_ANIMATIONS} a "
            "Clay rig holds. Delete the ones you do not need in Blender first."
        )
    fps = _guess_fps(model)
    taken: set[str] = set()
    animations: list[anim.Animation] = []
    cubic = dropped = mixed = 0
    for source in model.animations:
        label = f"The animation {source.name!r}" if source.name else "An animation"
        by_node: dict[int, dict[str, gltf.AnimChannel]] = {}
        interpolations: set[str] = set()
        for channel in source.channels:
            if channel.node not in importer.bone_of_node:
                dropped += 1
                continue
            by_node.setdefault(channel.node, {})[channel.path] = channel
            interpolations.add("step" if channel.interpolation == "STEP" else "linear")
        channels: dict[int, dict[str, anim.Channel]] = {}
        for node_index, props in by_node.items():
            made, resampled = _bone_channels(importer, node_index, props, fps, label)
            cubic += resampled
            channels[importer.bone_of_node[node_index].uid] = made
        if len(interpolations) > 1:
            mixed += 1
        loop = source.extras.get("loop", True) if isinstance(source.extras, dict) else True
        try:
            animations.append(
                anim.Animation(
                    _fresh_name(source.name, taken),
                    bool(loop),
                    "step" if interpolations == {"step"} else "linear",
                    channels,
                )
            )
        except ValueError as exc:
            raise OpError(f"{label} cannot be imported: {exc}") from exc
    notices: list[str] = []
    if cubic:
        notices.append(
            f"{cubic} smoothed (CUBICSPLINE) channel{'s' if cubic != 1 else ''} came in as "
            "linear keys, sampled at the frame rate: Clay keeps no tangent handles."
        )
    if dropped or model.skipped_animation_channels:
        n = dropped + model.skipped_animation_channels
        notices.append(
            f"{n} animation channel{'s' if n != 1 else ''} drove something that is not a bone "
            "(or morph weights) and were left out."
        )
    if mixed:
        notices.append(
            f"{mixed} animation{'s' if mixed != 1 else ''} mixed step and linear keys and came "
            "in as linear."
        )
    return anim.RigState(tuple(animations), fps), tuple(notices)


def _bone_channels(
    importer: _Importer,
    node_index: int,
    props: dict[str, gltf.AnimChannel],
    fps: int,
    label: str,
) -> tuple[dict[str, anim.Channel], int]:
    """One joint's channels, carried into its bone's frame. -> ``(channels, cubic count)``."""
    model = importer.model
    bone = importer.bone_of_node[node_index]
    keys: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    resampled = 0
    for path, channel in props.items():
        times, values, was_cubic = _linear_keys(channel, fps)
        resampled += int(was_cubic)
        keys[path] = (times, values)
        if len(times) > anim.MAX_KEYS_PER_CHANNEL:
            raise OpError(
                f"{label} has {len(times):,} keys on {bone.name!r}'s {path}, past the "
                f"{anim.MAX_KEYS_PER_CHANNEL:,} a Clay channel holds. Bake it to fewer keys "
                "in Blender first."
            )
        if len(times) and times[-1] > anim.MAX_ANIMATION_SECONDS:
            raise OpError(
                f"{label} runs {times[-1]:.0f} seconds, past the {anim.MAX_ANIMATION_SECONDS:.0f} "
                "a Clay clip holds."
            )
    fold = importer.frame_change(node_index)
    if fold is None:
        return {path: anim.Channel(t, v) for path, (t, v) in keys.items()}, resampled

    # A non-joint node sits between this joint and its parent bone, so every key crosses a
    # fixed matrix: sample all three properties at the union of the key times, multiply, and
    # decompose back. The ones the file does not animate hold the node's own value.
    node = model.nodes[node_index]
    union = np.unique(np.concatenate([t for t, _ in keys.values()]))
    if len(union) > anim.MAX_KEYS_PER_CHANNEL:
        raise OpError(
            f"{label} has {len(union):,} key times on {bone.name!r}, past the "
            f"{anim.MAX_KEYS_PER_CHANNEL:,} a Clay channel holds."
        )
    rest = {"translation": node.translation, "rotation": node.rotation, "scale": node.scale}
    out = {"translation": [], "rotation": [], "scale": []}
    interpolation = "step" if all(p.interpolation == "STEP" for p in props.values()) else "linear"
    for time in union:
        local = {
            path: (
                anim.sample(anim.Channel(*keys[path]), float(time), interpolation)
                if path in keys
                else np.asarray(rest[path], dtype="f8")
            )
            for path in rest
        }
        matrix = fold @ m3.compose(local["translation"], local["rotation"], local["scale"])
        t, r, s = m3.decompose(matrix)
        if not gltf.recomposes_without_shear(m3.compose(t, r, s), matrix):
            raise OpError(
                f"{label} moves {bone.name!r} through a non-uniform scale above it, which "
                "shears the keys; a Clay bone cannot hold that."
            )
        for path, value in (("translation", t), ("rotation", r), ("scale", s)):
            out[path].append(value)
    return {path: anim.Channel(union, np.array(rows)) for path, rows in out.items()}, resampled


# -- the skeleton's questions ---------------------------------------------------------------


def _joint_nodes(model: gltf.Model) -> list[int]:
    """Every node some skin uses as a joint, once, in node order. Refuses a skeleton Clay
    cannot hold or a joint the scene does not place."""
    seen: set[int] = set()
    for skin in model.skins:
        seen.update(int(j) for j in skin.joints)
    joints = sorted(seen)
    if len(joints) > rig.MAX_BONES:
        raise OpError(
            f"This GLB's skeleton has {len(joints)} joints, past the {rig.MAX_BONES} a Clay "
            "rig holds. Import it from the Library instead (Home > Import mesh...), or "
            "simplify the skeleton first."
        )
    for index in joints:
        if index not in model.reached:
            raise OpError(
                f"The joint {model.nodes[index].name or index!r} is not in the scene this GLB "
                "places, so it has no position to import it at."
            )
    return joints


def _parents(model: gltf.Model) -> dict[int, int]:
    parents: dict[int, int] = {}
    for index, node in enumerate(model.nodes):
        for child in node.children:
            parents.setdefault(int(child), index)
    return parents


def _joint_at_or_above(start: int, joints: Any, parents: dict[int, int]) -> int | None:
    cursor: int | None = start
    seen: set[int] = set()
    while cursor is not None and cursor not in seen:
        if cursor in joints:
            return cursor
        seen.add(cursor)
        cursor = parents.get(cursor)
    return None


def _joint_above(start: int, joints: Any, parents: dict[int, int]) -> int | None:
    above = parents.get(start)
    return None if above is None else _joint_at_or_above(above, joints, parents)


def _relative(world: np.ndarray, parent_world: np.ndarray | None, label: str):
    """``(translation, rotation, scale)`` of *world* relative to *parent_world*, refusing
    what a Clay object cannot be: sheared, non-finite or collapsed to nothing."""
    if not np.all(np.isfinite(world)):
        raise OpError(f"{label} has a non-finite transform.")
    matrix = np.asarray(world, dtype="f8")
    if parent_world is not None:
        if abs(np.linalg.det(np.asarray(parent_world, dtype="f8")[:3, :3])) < 1e-12:
            raise OpError(f"{label}'s parent bone has no scale, so nothing can sit under it.")
        matrix = np.linalg.inv(parent_world) @ matrix
    translation, rotation, scale = m3.decompose(matrix)
    if not gltf.recomposes_without_shear(m3.compose(translation, rotation, scale), matrix):
        raise OpError(
            f"{label} is sheared relative to its parent bone (a non-uniform scale above a "
            "turned bone), which a Clay bone cannot hold."
        )
    if not np.any(scale):
        raise OpError(f"{label} has a zero scale, so a Clay bone cannot be placed there.")
    return translation, rotation, scale


# -- the meshes' questions ------------------------------------------------------------------


def _plain_mesh(prim: gltf.Primitive, slot: int) -> bm.Mesh:
    from .glbimport import _mesh_for

    return _mesh_for(prim, slot)


def _primitive(prim: gltf.Primitive, positions: np.ndarray, normals: np.ndarray | None):
    """*prim*'s triangles over other positions and normals, without its cached box."""
    return gltf.Primitive(
        positions=positions.astype("f4"),
        indices=prim.indices,
        normals=None if normals is None else normals.astype("f4"),
        uvs=prim.uvs,
        material=prim.material,
    )


def _into_frame(positions: np.ndarray, normals: np.ndarray | None, matrix: np.ndarray):
    """*positions* and *normals* through *matrix* (normals by its inverse transpose)."""
    homogeneous = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
    moved = (matrix @ homogeneous.T).T[:, :3]
    if normals is None:
        return moved, None
    return moved, normals @ np.linalg.inv(matrix[:3, :3])


def _normalised(weights: np.ndarray) -> np.ndarray:
    """Weights summing to exactly one per vertex. A row of zeros (unskinned in glTF) is
    given to its first joint, the nearest honest reading of "no weights at all"."""
    weights = np.clip(weights, 0.0, None)
    total = weights.sum(axis=1)
    empty = total <= 0.0
    if empty.any():
        weights[empty] = [1.0, 0.0, 0.0, 0.0]
        total = weights.sum(axis=1)
    return (weights / total[:, None]).astype("f4")


def _rigid_joint(joints: np.ndarray, weights: np.ndarray) -> int | None:
    """The one skin joint every vertex follows alone at full weight, or ``None``."""
    first = int(joints[0, 0])
    if (
        (joints[:, 0] == first).all()
        and (weights[:, 0] >= _RIGID_WEIGHT).all()
        and (weights[:, 1:] <= 1.0 - _RIGID_WEIGHT).all()
    ):
        return first
    return None


def _posed(model: gltf.Model, skin: gltf.Skin, prim: gltf.Primitive):
    """The primitive's positions and normals as the file's pose displays them, float64."""
    positions = np.asarray(prim.positions, dtype="f8").reshape(-1, 3)
    normals = None if prim.normals is None else np.asarray(prim.normals, dtype="f8").reshape(-1, 3)
    palette = np.array(
        [
            np.asarray(model.nodes[int(j)].world, dtype="f8") @ np.asarray(ibm, dtype="f8")
            for j, ibm in zip(skin.joints, skin.inverse_bind, strict=True)
        ]
    )
    if np.allclose(palette, np.eye(4), atol=_BIND_TOLERANCE):
        return positions, normals
    joints = np.asarray(prim.joints, dtype="i8").reshape(-1, 4)
    weights = np.asarray(prim.weights, dtype="f8").reshape(-1, 4)
    out = np.zeros_like(positions)
    out_normals = None if normals is None else np.zeros_like(normals)
    for start in range(0, len(positions), _POSE_CHUNK):
        stop = min(start + _POSE_CHUNK, len(positions))
        for k in range(4):
            rows = start + np.flatnonzero(weights[start:stop, k] > 0.0)
            if not len(rows):
                continue
            m = palette[joints[rows, k]]
            w = weights[rows, k][:, None]
            out[rows] += w * (np.einsum("nij,nj->ni", m[:, :3, :3], positions[rows]) + m[:, :3, 3])
            if out_normals is not None:
                out_normals[rows] += w * np.einsum("nij,nj->ni", m[:, :3, :3], normals[rows])
    return out, out_normals


def _weighted_mesh(
    prim: gltf.Primitive,
    positions: np.ndarray,
    normals: np.ndarray | None,
    joints: np.ndarray,
    weights: np.ndarray,
    slot: int,
) -> bm.Mesh:
    """One weighted primitive as a CSR mesh, merging vertices that agree on position *and*
    weights (and on nothing else)."""
    from .glbimport import _smooth_flags

    positions = np.ascontiguousarray(positions, dtype="f4").reshape(-1, 3)
    if not np.all(np.isfinite(positions)):
        raise OpError("This mesh has a non-finite (NaN or infinite) vertex position.")
    indices = np.asarray(prim.indices, dtype="i8").reshape(-1)
    tris = indices[: (len(indices) // 3) * 3].reshape(-1, 3)

    key = np.concatenate([positions, joints.astype("f4"), weights.astype("f4")], axis=1)
    merged, inverse = np.unique(key, axis=0, return_inverse=True)
    loops = inverse.reshape(-1)[tris].reshape(-1)
    merged_positions = merged[:, :3]

    uv = None
    if prim.uvs is not None:
        uv = np.ascontiguousarray(prim.uvs, dtype="f4").reshape(-1, 2)[tris].reshape(-1, 2)
    n_faces = len(tris)
    flat = topo.rebuild(
        merged_positions,
        loops,
        topo.starts_from_counts(np.full(n_faces, 3, dtype="i8")),
        np.full(n_faces, int(slot), dtype="i4"),
        _smooth_flags(_primitive(prim, positions, normals), merged_positions, loops, tris),
        uv=uv,
    )
    return bm.Mesh(
        positions=flat.positions,
        loops=flat.loops,
        starts=flat.starts,
        material=flat.material,
        smooth=flat.smooth,
        uv=flat.uv,
        joints=merged[:, 3:7].astype("i4"),
        weights=merged[:, 7:11].astype("f4"),
    )
