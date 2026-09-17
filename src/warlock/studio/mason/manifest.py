"""The engine-facing sidecar: the provenance a GLB cannot carry.

A Mason scene exports as ``scene.glb`` plus this file. The GLB is the
geometry, the hierarchy, the lights and the cameras -- everything glTF has a
place for. What it has no place for is *where each of these things came from*:
which library job a prop is, which generator and parameters a primitive was
built from, which user properties a designer typed, which nodes are copies of
one template, and whether a node was marked static. An import script building
an engine scene wants all of that, and the alternative -- glTF ``extras`` --
survives some importers and is silently dropped by others.

**Written for a reader with no glTF library.** The whole point is a file an
import script can parse with nothing but a JSON reader, then use to decorate
the objects its engine's own glTF importer produced. So every reference into
the GLB is by **node name**, and the names come from
:class:`~.gltfout.SceneExport` rather than being recomputed here -- the
manifest and the GLB name the same nodes by construction, not by running the
same uniquifier twice and hoping.

**Units and handedness are stated outright**, because that is the one thing an
import script gets wrong and the one thing no amount of care downstream can
recover: metres, Y-up, right-handed, ``-Z`` forward, rotations as XYZW
quaternions, angles in radians. This project now says so in
``dev/INVARIANTS.md`` as well; a scene unit that exists only inside an
exporter is a scene unit nobody can rely on.

**The schema is ours and no importer exists to test it against**, which is
recorded here rather than left implicit. It is verified by *shape* -- every
name it uses is a name the GLB carries, every node the GLB carries is a node
it describes, and the counts agree -- and not by a round trip through an
engine. When the first import script exists, that script becomes the test.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from ...kernels.geom3d import math3d as m3
from .gltfout import ExportedNode, SceneExport
from .nodes import CameraNode, LightNode, TerrainNode
from .refs import LibraryRef, PrimitiveRef, Ref

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import MasonDoc

__all__ = ["FORMAT", "GEOMETRY", "MANIFEST", "VERSION", "manifest_bytes", "scene_manifest"]

#: The member name the bundle writes this under, beside ``scene.glb``.
MANIFEST = "scene.json"
GEOMETRY = "scene.glb"

#: A tag an import script can check before trusting a single key below. A file
#: format with no name in it is one that can only be identified by guessing.
FORMAT = "warlock-mason-scene"
VERSION = 1

#: The sentence this whole file exists to make unambiguous, as data rather
#: than as a comment. glTF already fixes every one of these, and an import
#: script that assumed otherwise is precisely the failure this answers -- so
#: the manifest restates them where a reader with no glTF library will see
#: them.
CONVENTIONS: dict[str, Any] = {
    "length": "metre",
    "up": "+Y",
    "forward": "-Z",
    "handedness": "right",
    "rotation": "quaternion xyzw",
    "angles": "radians",
}


def _vec(values: Any) -> list[float]:
    return [float(v) for v in values]


def _trs(translation: Any, rotation: Any, scale: Any) -> dict[str, list[float]]:
    return {
        "translation": _vec(translation),
        "rotation": _vec(rotation),  # XYZW, as everywhere in this project
        "scale": _vec(scale),
    }


def _source_json(ref: Ref | None) -> dict[str, Any] | None:
    """Where a mesh node's geometry came from, or ``None`` for a node with no
    source at all (a group, a light, a mesh the user has not filled in yet).

    This is the half of the manifest that has no equivalent anywhere in the
    GLB: the file knows there are triangles, and only this says they are job
    ``3f2a...`` called "Barrel" or a ``cylinder`` with these seven numbers.
    Which makes re-exporting an updated asset and re-running an import script
    a thing a pipeline can actually do.
    """
    if ref is None:
        return None
    if isinstance(ref, PrimitiveRef):
        return {
            "kind": "primitive",
            "generator": ref.generator,
            "params": {key: _plain(value) for key, value in ref.params},
        }
    if isinstance(ref, LibraryRef):
        entry: dict[str, Any] = {
            "kind": "library",
            "job_id": ref.job_id,
            "artifact": ref.artifact,
        }
        if ref.name:
            entry["name"] = ref.name
        if ref.sha256:
            entry["sha256"] = ref.sha256
        return entry
    return None  # pragma: no cover - the Ref union is closed


def _plain(value: Any) -> Any:
    """A normalized param value as JSON. ``refs._normalize`` has already turned
    every number into a float and every sequence into a nested tuple, so this
    only has to turn the tuples back into lists -- JSON has no tuple, and
    ``json.dumps`` would do it silently and asymmetrically."""
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _light_json(node: LightNode) -> dict[str, Any]:
    """A light's own fields, in KHR_lights_punctual's names.

    Repeated from the GLB rather than left to it, and that is deliberate: this
    file's reader has no glTF library, and the extension block is the part of
    a GLB such a reader is least likely to be able to reach.
    """
    entry: dict[str, Any] = {
        "type": node.kind,
        "color": _vec(node.color),
        "intensity": float(node.intensity),
    }
    if node.range > 0.0 and node.kind != "directional":
        entry["range"] = float(node.range)
    if node.kind == "spot":
        entry["innerConeAngle"] = float(node.inner_cone_angle)
        entry["outerConeAngle"] = float(node.outer_cone_angle)
    return entry


def _camera_json(node: CameraNode) -> dict[str, Any]:
    entry: dict[str, Any] = {"yfov": float(node.yfov), "znear": float(node.znear)}
    if node.zfar > 0.0:
        entry["zfar"] = float(node.zfar)
    return entry


def _node_json(
    record: ExportedNode, parent: str | None, doc: MasonDoc
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": record.name,
        "kind": record.kind,
        "parent": parent,
        # The document uid, which is an address only for a node authored in
        # the scene tree. Every copy of a template leaf carries the
        # *template's* uid, because every copy is the same node object reached
        # by a different path -- that is what prefabs are. ``owner_uid`` below
        # is what disambiguates them, and ``name`` is what actually addresses
        # a node in the exported file.
        "uid": int(record.node.uid),
        "local": _trs(*record.node.trs()),
        # The composed answer as well as the authored one. An import script
        # that places one object without rebuilding the hierarchy has it
        # ready, and one that rebuilds the hierarchy ignores it -- either way
        # the decomposition is done once, here, by the module that owns the
        # convention, rather than by every reader against a 4x4 of numbers
        # whose handedness they would each have to assume.
        "world": _trs(*m3.decompose(np.asarray(record.world, dtype="f8"))),
    }
    if record.static:
        entry["static"] = True
    if record.locked:
        entry["locked"] = True
    if record.node.properties:
        # The user properties table, verbatim. This is the field the manifest
        # exists for as much as any other: a designer types ``spawn=true`` on
        # a node and an import script acts on it.
        entry["properties"] = dict(record.node.properties)
    source = _source_json(record.ref)
    if source is not None:
        entry["source"] = source
    if record.dangling:
        # Stated rather than omitted: a scene exported with a broken link
        # still exports, and the manifest is where the reader finds out that
        # this node's geometry is the one thing that is not in the GLB.
        entry["missing"] = True
    if isinstance(record.node, LightNode):
        entry["light"] = _light_json(record.node)
    elif isinstance(record.node, CameraNode):
        entry["camera"] = _camera_json(record.node)
    elif isinstance(record.node, TerrainNode) and doc.terrain is not None:
        entry["terrain"] = {
            "size_x": float(doc.terrain.size_x),
            "size_z": float(doc.terrain.size_z),
            "side": int(doc.terrain.side),
        }
    if record.prefab:
        entry["prefab"] = record.prefab
    if record.owner != record.node.uid:
        # The outliner row this node belongs to. For anything inside an
        # expanded instance that is the instance, never the template's own
        # node -- ``scene.Placed.owner``'s rule, carried through so an import
        # script groups an instance's contents the way the editor does.
        entry["owner_uid"] = int(record.owner)
    return entry


def _prefabs_json(export: SceneExport) -> dict[str, Any]:
    """Each template that was actually placed, and what each of its instances
    became in the file.

    **This is what lets an import script build engine prefabs rather than
    hundreds of loose meshes**, which is the difference between a scene a
    person can go on editing in their engine and a scene they can only look
    at. The shape is per-instance rather than one abstract template plus a
    list of transforms, because names are what the GLB carries: a script reads
    the first instance's node list to build the prefab, then places the rest
    at their own transforms.

    A template with no instances does not appear at all -- it is in the
    exported file in no form whatever, and a section naming nodes the GLB does
    not carry is worse than no section.

    Membership is decided by **path prefix**, which is the only thing that can
    decide it: an instance's copy of a template leaf and another instance's
    copy are the same ``node`` object by identity and differ solely in how the
    walk reached them. The longest matching prefix wins, so a nested instance's
    contents belong to the inner instance rather than to the outer one.
    """
    instances: dict[tuple[int, ...], dict[str, Any]] = {}
    out: dict[str, dict[str, Any]] = {}
    for record in export.nodes:
        if record.kind != "prefab":
            continue
        entry: dict[str, Any] = {"node": record.name, "nodes": []}
        instances[record.path] = entry
        template = str(getattr(record.node, "template", ""))
        out.setdefault(template, {"instances": []})["instances"].append(entry)
    for record in export.nodes:
        if not record.prefab:
            continue
        for length in range(len(record.path) - 1, 0, -1):
            owner = instances.get(record.path[:length])
            if owner is not None:
                owner["nodes"].append(record.name)
                break
    return out


def scene_manifest(doc: MasonDoc, export: SceneExport) -> dict[str, Any]:
    """The manifest for an export, as a plain dict.

    Takes the :class:`~.gltfout.SceneExport` rather than re-walking the
    document, which is the whole reason ``scene_model`` hands one back: the
    names, the worlds and the prefab membership in here are *the same values*
    the GLB was written from.
    """
    names = {record.path: record.name for record in export.nodes}
    nodes = [
        _node_json(record, names.get(record.path[:-1]), doc) for record in export.nodes
    ]
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "format_version": VERSION,
        "generator": "Warlock Studio",
        "units": dict(CONVENTIONS),
        "geometry": GEOMETRY,
        "nodes": nodes,
        "stats": {
            "nodes": len(nodes),
            "meshes": len(export.model.meshes),
            "lights": len(export.model.lights),
            "cameras": len(export.model.cameras),
            "triangles": int(export.model.triangle_count),
        },
    }
    prefabs = _prefabs_json(export)
    if prefabs:
        manifest["prefabs"] = prefabs
    if export.unresolved:
        # The same loss the GLB shows as a node with no mesh, said in words.
        # ``Model.skipped_textures`` is the precedent: a loss that is stated
        # is not the same as a loss that is invisible.
        manifest["missing"] = [
            {"node": export.nodes[index].name, **(_source_json(ref) or {})}
            for index, ref in export.unresolved
        ]
    return manifest


def manifest_bytes(doc: MasonDoc, export: SceneExport) -> bytes:
    """The manifest's bytes: sorted keys, indented, UTF-8.

    Sorted and indented because this half of the bundle exists to be *read* --
    by the person writing the import script, and by a diff showing what
    changed between two exports of one scene. There is no size argument for
    minifying it: the geometry is in the file next to it.
    """
    return json.dumps(scene_manifest(doc, export), sort_keys=True, indent=2).encode("utf-8")
