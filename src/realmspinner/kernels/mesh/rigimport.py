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
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..geom3d import gltf
from ..geom3d import math3d as m3
from . import mesh as bm
from . import rig, topo
from .document import Obj, _empty_mesh, new_uid
from .elements import OpError

__all__ = ["import_rigged"]

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
) -> list[Obj]:
    """The objects of a model that has skins: bones first, then every placed mesh.

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
    return objects


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
