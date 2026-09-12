"""Mason's file layer: what a scene is read from and written to.

**No file dialog and no encode ever runs on the frame thread** -- the one rule
Clay's and Plotter's io layers each state for their own document: a native
picker is modal to the OS and blocks until dismissed, and a document of any
size is a zip (or a GLB, or a merged OBJ) to build. The cheap half --
``mason.serialize.snapshot``, a walk of the tree and the palette that
allocates nothing large -- runs on the frame thread, where it can read the
live document without a lock; the expensive half -- the zip container, the
``.npy`` encode, every PNG encode, the glTF or OBJ walk -- happens inside
``run()``, after ``ctx.submit`` has already moved it off that thread.

This module exists because Stage E named it: no earlier stage of Mason had a
document to save, export, or reopen, so nothing before this one needed an io
layer at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..service.files import MAX_SCENE_SOURCE_BYTES
from . import atomic, mason_state, sizeguard

WSCN_FILTER = ["Warlock scene (*.wscn)", "*.wscn"]
OBJ_FILTER = ["Wavefront OBJ (*.obj)", "*.obj"]
GLB_FILTER = ["glTF binary (*.glb)", "*.glb"]

# A ``.wscn`` links rather than embeds its geometry (``mason.serialize``'s own
# module docstring: no mesh is ever stored). What it actually carries is a
# node tree as text, one terrain height array, and whatever override
# materials' textures were painted directly onto a node -- orders of
# magnitude below ``service.files.MAX_CLAY_SOURCE_BYTES``, which has to hold a
# mesh's full vertex and index arrays. Fifty megabytes is generous against
# that shape and still small enough to refuse before an unreasonable file is
# ever unzipped.
#
# **Imported, not restated.** Stage G gave a scene a second door -- the
# ``scene.wscn`` sidecar beside an exported library row -- and the ceiling on
# the way *out* through that door is ``service.files.MAX_SCENE_SOURCE_BYTES``.
# Two numbers for one format is how a file this module opens happily comes to
# be one the service refuses to store, so there is one number and this is a
# view of it. The import is at module scope deliberately: ``mason_io`` already
# reaches into ``service`` nowhere else, but a lazy import here would put the
# only statement of the ceiling behind a function call, and the constant is
# read by ``sizeguard`` on a path that must not be able to find it missing.
MAX_WSCN_BYTES = MAX_SCENE_SOURCE_BYTES


def load(path: Path) -> dict[str, Any]:
    """Blocking; task thread only. Raises rather than returning a broken tab.

    The camera comes off the returned document itself, not from a second
    parse of the bytes: ``mason.serialize.read_wscn`` already restores
    ``doc.view`` as part of building the document, because Mason's format
    puts the camera inside ``scene.json`` alongside the tree it is a view of.
    This differs from Clay, whose ``read_view`` is a second, cheaper parse of
    the same bytes -- kept apart there because a ``.wblk``'s geometry is the
    expensive half of opening it and the camera is wanted before that cost is
    paid. A ``.wscn`` has no such expensive half to get ahead of, so there is
    only the one read here, and a caller comparing the two modules is not
    looking at an omission.
    """
    from .mason import serialize

    data = sizeguard.within_ceiling(path, MAX_WSCN_BYTES).read_bytes()
    doc = serialize.read_wscn(data)
    return {
        "doc": doc,
        "path": str(path),
        "title": mason_state.title_for(path),
        "view": doc.view or None,
    }


def write_wscn(path: Path, data: bytes) -> None:
    """Write a ``.wscn``: staged to a temp beside ``path``, then ``os.replace``d."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic.write_bytes(path, data)


def write_files(files: dict[str, bytes], path: Path, *, primary: str) -> None:
    """Write a multi-file export: ``files[primary]`` to ``path`` itself, every
    other member beside it at ``path.parent / name``.

    All through ``atomic.staged_set``, so either every file appears or none
    does. A partial export directory is worse than no export at all: a
    ``scene.glb`` written without the ``scene.json`` beside it (or the other
    way round) is not a smaller export, it is a broken one that looks intact
    until an import script goes looking for the half that never landed.
    """
    targets: dict[Path, bytes] = {}
    for name, blob in files.items():
        targets[path if name == primary else path.parent / name] = blob
    atomic.staged_set(targets)


def glb_bundle(doc: Any, source: Any) -> dict[str, bytes]:
    """A scene as ``scene.glb`` plus its ``scene.json`` sidecar manifest.

    Built from one walk, not two: ``gltfout.scene_glb`` would run
    ``scene_model`` a second time to get the bytes, so this calls
    ``scene_model`` itself and hands the resulting :class:`SceneExport` to
    both the manifest and the GLB writer -- the same model, the same node
    records, encoded twice rather than walked twice.
    """
    from .mason import gltfout, manifest
    from .viewer import glbwrite

    export = gltfout.scene_model(doc, source)
    return {
        "scene.glb": glbwrite.write_glb(export.model),
        manifest.MANIFEST: manifest.manifest_bytes(doc, export),
    }


def obj_bundle(doc: Any, source: Any) -> tuple[dict[str, bytes], list[str], int, int]:
    """The scene as a merged OBJ. A thin wrapper: the ``skipped`` sentences
    come straight from ``objout.obj_export`` and reach the caller intact --
    a light or a camera OBJ cannot carry is a loss that is *stated*, not one
    that vanishes silently between here and the toast that reports it.
    """
    from .mason import objout

    export = objout.obj_export(doc, source)
    return dict(export.files), list(export.skipped), export.vertices, export.triangles
