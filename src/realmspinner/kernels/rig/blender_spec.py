"""The worker specs -- pure dict construction, one function per op the
Blender subprocess understands.

Split out of the former ``rigging.py`` (P4 of ``dev/RESTRUCTURE.md``). Every
function here builds the JSON payload ``pipelines.blender_run.run_worker``
hands to ``pipelines/blender_worker.py`` on stdin; none of them touches a
process, a file beyond reading a template, or Blender itself -- that split is
what keeps this whole package importable with no ``bpy`` anywhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import store
from .store import pose_dir, pose_glb_path
from .templates import get_template

#: Where a rig's joints come from when the user has not moved them by hand.
#: ``template`` is the bbox-proportional fit every rig used before this existed
#: and is still the default; ``measured`` reads them off the mesh.
JOINT_SOURCES = ("template", "measured")


def rig_spec(
    job_dir: Path,
    template_key: str,
    bones: list[dict[str, Any]] | None = None,
    *,
    template_bones: list[dict[str, Any]] | None = None,
    fit: dict[str, Any] | None = None,
    joints: str | None = None,
    skeleton: str | None = None,
    root: str | None = None,
    mirror_pairs: list[Sequence[str]] | None = None,
) -> dict[str, Any]:
    """The worker spec for rigging a finished job's mesh.

    Three sources of joints, in a fixed order of preference, and the order is
    the design:

    * ``bones`` -- world-space joints the *user* moved, from the adjust-joints
      pass. Already validated against the template by ``skeleton.
      validate_joints``, and they win over everything: a correction is the
      last word by definition.
    * ``template_bones`` -- the template's own landmarks, replaced with ones
      measured off the reference image (``pipelines.pose2d``). Still normalized
      and still the template's shape, so the worker fits them onto the mesh
      bbox with exactly the ``skeleton.fit_template`` it uses for the shipped
      ones -- the scaling stays owned by the worker and this stays a *better
      template* rather than a second fitter.
    * nothing -- the shipped template, scaled bbox-proportionally, which is
      what every rig did before landmarks existed and is still right for a
      reference that really is standing in a T-pose.

    ``fit`` is what the host knows about how the second of those was found and
    the worker cannot: which model, how confident. It is recorded in rig.json
    and read by nothing that has to work, which is why it is a free-form dict.

    ``joints="measured"`` asks the worker to take ``pipelines.jointfit`` to the
    mesh's own vertices and use the result as if the user had corrected the
    joints by hand. It sits *below* ``bones`` in the same order of preference --
    a real correction is still the last word -- and above the template, because
    a measurement of the mesh in front of you beats a template scaled to its
    bounding box whenever the two disagree. The two supplied base meshes are
    exactly where they disagree: the shipped humanoid template is an A-pose and
    a T-pose mesh fits it badly.

    ``skeleton``/``root``/``mirror_pairs`` carry a *custom* skeleton's own
    structure across the pipe -- ``service.rig.edit_skeleton``'s
    ``skeleton.validate_skeleton`` output -- so the worker can build the
    armature from the caller's own parents instead of the template's. All
    three are written only when given, which is what keeps a spec built for
    an ordinary template rig, or for an old ``adjust_joints`` joint move,
    byte-identical to what it always was: the pin in ``tests/test_rigging.py``
    for exactly that.
    """
    get_template(template_key)  # fail here, not three seconds into a subprocess
    spec = {
        "op": "rig",
        "source_glb": str(job_dir / "model.glb"),
        "out_glb": str(job_dir / store.RIG_GLB_TMP),
        "out_json": str(job_dir / store.RIG_JSON_TMP),
        "result_path": str(job_dir / ".blender_result.json"),
        "template": template_key,
    }
    if bones is not None:
        spec["bones"] = bones
    if template_bones is not None:
        spec["template_bones"] = template_bones
    if fit is not None:
        spec["fit"] = fit
    if joints is not None:
        if joints not in JOINT_SOURCES:
            raise ValueError(f"joints must be one of {list(JOINT_SOURCES)}")
        spec["joints"] = joints
    if skeleton is not None:
        spec["skeleton"] = skeleton
    if root is not None:
        spec["root"] = root
    if mirror_pairs is not None:
        spec["mirror_pairs"] = [list(p) for p in mirror_pairs]
    return spec


def sheet_spec(
    source_glb: Path,
    frames_dir: Path,
    cells: list[dict[str, Any]],
    *,
    frame_size: int,
    elevation: float,
    lighting: str,
    margin: float | None = None,
    sockets: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The worker spec for rendering one frame per sheet cell.

    ``cells`` carries each cell's pose bones inline rather than a pose id: the
    worker has no access to the job directory's pose files, and shipping the
    rotations with the cell keeps it a pure renderer.

    ``margin`` widens the ortho window past ``sheet.FRAME_MARGIN``. Its one
    writer is ``_q_troupe``'s reframe retry, and it is written **only when
    given** -- a spec without it is byte-identical to the one this function
    produced before the key existed, which is what keeps every sheet that does
    not clip rendering exactly as it did.

    ``sockets`` is ``[{"name", "bone", "offset": [along, lateral, up],
    "reach"}]``: attachment points to project per cell, offsets in bone-length
    units and reach in character heights so a socket list survives a re-fit
    onto a character of another size. Written only when given, for the same
    reason, and its presence is what makes the worker emit a ``sockets`` block
    at all.
    """
    spec: dict[str, Any] = {
        "op": "sheet",
        "source_glb": str(source_glb),
        "frames_dir": str(frames_dir),
        "result_path": str(frames_dir / "result.json"),
        "frame_size": frame_size,
        "elevation": elevation,
        "lighting": lighting,
        "cells": cells,
    }
    if margin is not None:
        spec["margin"] = float(margin)
    if sockets is not None:
        spec["sockets"] = [dict(s) for s in sockets]
    return spec


def views_spec(
    source_glb: Path,
    views_dir: Path,
    views: list[tuple[float, float]],
    *,
    size: int,
    depth: bool = False,
) -> dict[str, Any]:
    """The worker spec for rendering one flat view per direction.

    Half of a re-texture. The other half (``project_spec``) runs *after* the
    host has restyled these renders, which is why it is two ops rather than
    one: the restyle is an SDXL pass and Blender is a separate interpreter with
    no way to call back into this one.

    ``depth`` adds a second render per view -- the camera-depth encoding the
    visibility test and the ControlNet hint both decode. Off by default so
    every other caller keeps meaning exactly what it meant before the depth
    pass existed.
    """
    return {
        "op": "views",
        "source_glb": str(source_glb),
        "views_dir": str(views_dir),
        "result_path": str(views_dir / ".views_result.json"),
        "size": size,
        "views": [list(v) for v in views],
        "depth": bool(depth),
    }


def project_spec(
    source_glb: Path,
    views_dir: Path,
    out_dir: Path,
    views: list[tuple[float, float]],
    *,
    size: int,
    texture_size: int,
    depth: bool = False,
) -> dict[str, Any]:
    """The worker spec for baking each restyled view into the mesh's atlas.

    ``size`` is the render size again rather than a second knob: it is what
    frames the camera, and a camera framed differently from the one that drew
    the views would land every projection somewhere the colours are not.

    ``depth`` adds a third bake per view: the depth the camera recorded at
    each texel's pixel, next to the texel's own depth, which is everything the
    host's visibility compare needs. It requires the ``depth`` renders from
    ``views_spec`` to exist, and it defaults off for ``views_spec``'s reason.
    """
    return {
        "op": "project",
        "source_glb": str(source_glb),
        "views_dir": str(views_dir),
        "out_dir": str(out_dir),
        "result_path": str(out_dir / ".project_result.json"),
        "size": size,
        "texture_size": texture_size,
        "views": [list(v) for v in views],
        "depth": bool(depth),
    }


def fbx_spec(source_glb: Path, out_fbx: Path, result_dir: Path) -> dict[str, Any]:
    """The worker spec for converting a GLB to FBX.

    Blender is the converter because it is already here and already
    out-of-process; adding an FBX library to the app process would mean a second
    importer disagreeing with the one that produces every other artifact.
    """
    return {
        "op": "fbx",
        "source_glb": str(source_glb),
        "out_fbx": str(out_fbx),
        "result_path": str(result_dir / ".fbx_result.json"),
    }


def pose_spec(
    job_dir: Path,
    pose_id: str,
    bones: dict[str, Any],
    *,
    root_bone: str | None = None,
    root_offset: Sequence[float] | None = None,
) -> dict[str, Any]:
    """The worker spec for baking one saved pose into its own GLB.

    ``root_bone``/``root_offset`` carry a library pose's root translation
    (world units, Blender axes) into the bake. Both keys are added only when
    there is a bone *and* a nonzero offset, so every spec built before the
    fields existed -- and every pose without an offset -- is byte-identical to
    what it always was: backward compatibility is structural, not versioned.
    """
    spec = {
        "op": "pose",
        "rig_glb": str(job_dir / "rig.glb"),
        "out_glb": str(pose_glb_path(job_dir, pose_id)),
        "result_path": str(pose_dir(job_dir) / f".{pose_id}.result.json"),
        "bones": bones,
    }
    if root_bone is not None and root_offset is not None and any(float(v) for v in root_offset):
        spec["root_bone"] = root_bone
        spec["root_offset"] = [float(v) for v in root_offset]
    return spec


def armature_spec(template_key: str, out_glb: Path, result_dir: Path) -> dict[str, Any]:
    """The worker spec for exporting one template's armature with no mesh.

    What the Poser preview stands on: the skeleton is built and exported by the
    *same* Blender code path a real rig uses (``_build_armature`` + ``_export``),
    fitted over the canonical unit box, so the bone frames the editor rotates
    are the frames every bake will see. ``out_glb`` must end in ``.glb`` -- the
    exporter appends one to any path that does not (the RIG_GLB_TMP rule).
    """
    get_template(template_key)  # fail here, not three seconds into a subprocess
    return {
        "op": "armature",
        "template": template_key,
        "out_glb": str(out_glb),
        # Named per template: two previews building concurrently share the
        # previews directory, and ``run_worker`` unlinks and then watches the
        # result path -- one shared name would let each build eat the other's
        # answer. The per-template lock in ``template_preview`` serializes one
        # template's builds; this is what keeps two *different* ones apart.
        "result_path": str(result_dir / f".{template_key}.armature_result.json"),
    }


def clip_sample_spec(
    source: Path,
    template: str,
    result_path: Path,
    *,
    max_frames: int = 900,
    max_actions: int = 64,
) -> dict[str, Any]:
    """The worker spec for "Import clip": sampling an external animation's
    world bone transforms so a later, Blender-free step can convert them onto
    ``template``'s own bone frames.

    **Never called "retarget"** -- see ``clipmaps``'s module docstring for why.
    ``candidates``/``strip`` are every shipped clip map's chain names and
    strip pattern, not just the one that will eventually match: the worker
    only has to recognise a source skeleton's *naming* well enough to pick the
    right armature and bones out of the file, and score which map actually
    resolves against ``source_bones``/``all_bone_names`` runs host-side,
    later, through ``clipmaps.match`` -- the same split ``skeleton.
    fit_template``'s bbox arithmetic keeps clear of ``_build_armature``'s bpy
    calls.

    ``result_path`` is the caller's to make unique (``store.new_id()``, the
    same rule a pose or sheet cell follows) -- unlike ``armature_spec``'s
    per-template name, two "Import clip" runs against the same template have
    no lock serialising them and must never share a result file.
    """
    get_template(template)  # fail here, not three seconds into a subprocess
    # Function-level: ``clipmaps`` imports this package's ``templates`` module
    # at its own top, so a top-level import here would be circular -- the same
    # shape ``cliplib.parse_clip_library``'s ``poselib`` import breaks the
    # other cycle in.
    from ... import clipmaps

    candidates: set[str] = set()
    strips: set[str] = set()
    for clip_map in clipmaps.load_clip_maps().values():
        strips.add(clip_map.strip.pattern)
        for chain in clip_map.bones.values():
            candidates.update(chain)
    return {
        "op": "clip_sample",
        "source": str(source),
        "template": template,
        "candidates": sorted(candidates),
        "strip": sorted(strips),
        "max_frames": int(max_frames),
        # The 2026-09-18 audit, finding poser-02: nothing bounded how many
        # actions a source FBX/GLB may carry before ``op_clip_sample`` sampled
        # every one of them, frame by frame, with no ceiling to refuse by.
        "max_actions": int(max_actions),
        "result_path": str(result_path),
    }


def root_offset_world(root_translation: Sequence[float], bounds: dict[str, Any]) -> list[float]:
    """A library pose's root offset, scaled from character heights to world units.

    ``root_translation`` is stored in character-height units because the Poser
    armature is exactly one character tall (``poselib.UNIT_LO``/``UNIT_HI``);
    the target rig's height comes from rig.json's ``bounds``, which are Blender
    world coordinates, Z up. A degenerate height falls back to the largest
    extent -- the ``skeleton.fit_template`` rule for a flat axis -- and to 1.0
    when the whole box is a point, so the offset degrades to "as authored"
    rather than collapsing to zero.
    """
    lo = [float(v) for v in bounds["min"]]
    hi = [float(v) for v in bounds["max"]]
    h = hi[2] - lo[2]
    if h <= 0:
        h = max(b - a for a, b in zip(lo, hi, strict=True))
        if h <= 0:
            h = 1.0
    return [float(u) * h for u in root_translation]


def remesh_spec(
    source_glb: Path,
    out_glb: Path,
    result_dir: Path,
    *,
    target_faces: int,
    texture_size: int,
    close_holes: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """The worker spec for remeshing a GLB to a quad budget and rebaking it.

    One op rather than the re-texture's two, because nothing on the host sits
    between the steps: the remesh, the unwrap and every bake are Blender's, and
    the host's only part is the publish afterwards. ``texture_size`` is
    resolved before this is built -- the worker never reads the atlas to decide
    it, so the spec is the whole record of what was asked.
    """
    return {
        "op": "remesh",
        "source_glb": str(source_glb),
        "out_glb": str(out_glb),
        "result_path": str(result_dir / ".remesh_result.json"),
        "target_faces": int(target_faces),
        "texture_size": int(texture_size),
        "close_holes": bool(close_holes),
        "seed": int(seed),
    }


#: Clamp range for the three Clay background ops' ``target_faces``. Lower
#: than ``remesh.FACES_MIN`` (500): a Clay selection can be a single small
#: prop, and 100 faces is still a mesh a quadriflow pass can act on, where
#: ``remesh``'s floor is tuned for a whole-character reconstruction.
CLAY_TARGET_FACES_MIN = 100
CLAY_TARGET_FACES_MAX = 200_000

#: Bake resolutions the Clay bake op accepts -- one entry finer than
#: ``pipelines.remesh.TEXTURE_SIZES`` (256) for a small prop, and one entry
#: coarser (4096) for a hero asset baked once rather than re-baked at a
#: reconstruction's atlas size.
CLAY_TEXTURE_SIZES = (256, 512, 1024, 2048, 4096)

#: What ``clay_bake_spec`` accepts in ``maps``, and what ``op_clay_bake``
#: bakes when none are named.
CLAY_BAKE_MAPS = ("base_color", "roughness", "normal")


def clay_retopo_spec(
    source_glb: Path,
    out_glb: Path,
    result_dir: Path,
    *,
    target_faces: int,
    close_holes: bool = False,
    seed: int = 0,
    keep_uvs: bool = False,
) -> dict[str, Any]:
    """The worker spec for Clay's "Retopologise" background op.

    Independent of ``remesh_spec``'s reconstruction pipeline -- no unwrap, no
    bake -- and every mesh object in ``source_glb`` keeps its own node rather
    than being joined into one: Clay may send a multi-object selection and
    expects the same shape back. ``target_faces`` is the *whole selection's*
    budget; ``op_clay_retopo`` shares it out per object proportionally to
    each one's own triangle count.
    """
    target = int(target_faces)
    if not CLAY_TARGET_FACES_MIN <= target <= CLAY_TARGET_FACES_MAX:
        raise ValueError(
            f"target_faces must be between {CLAY_TARGET_FACES_MIN:,} and "
            f"{CLAY_TARGET_FACES_MAX:,}"
        )
    return {
        "op": "clay_retopo",
        "source_glb": str(source_glb),
        "out_glb": str(out_glb),
        "result_path": str(result_dir / ".clay_retopo_result.json"),
        "target_faces": target,
        "close_holes": bool(close_holes),
        "seed": int(seed),
        "keep_uvs": bool(keep_uvs),
    }


def clay_unwrap_spec(
    source_glb: Path,
    out_glb: Path,
    result_dir: Path,
    *,
    angle_limit: float = 66.0,
    island_margin: float = 0.003,
) -> dict[str, Any]:
    """The worker spec for Clay's "Unwrap" background op: Smart UV Project
    per object, geometry untouched. Defaults match ``op_remesh``'s own
    unwrap pass."""
    return {
        "op": "clay_unwrap",
        "source_glb": str(source_glb),
        "out_glb": str(out_glb),
        "result_path": str(result_dir / ".clay_unwrap_result.json"),
        "angle_limit": float(angle_limit),
        "island_margin": float(island_margin),
    }


def clay_bake_spec(
    high_glb: Path,
    low_glb: Path,
    out_glb: Path,
    result_dir: Path,
    *,
    texture_size: int,
    cage_extrusion: float,
    maps: Sequence[str] = CLAY_BAKE_MAPS,
) -> dict[str, Any]:
    """The worker spec for Clay's "Bake high to low" background op.

    ``cage_extrusion`` is the caller's to set (a raw distance in the scene's
    own units) rather than derived here the way ``op_remesh`` derives its own
    from the mesh's bounding diagonal -- Clay knows its document's scale and
    this function does not read either GLB to find out.
    """
    if texture_size not in CLAY_TEXTURE_SIZES:
        raise ValueError(f"texture_size must be one of {CLAY_TEXTURE_SIZES}")
    chosen = list(maps)
    unknown = sorted(set(chosen) - set(CLAY_BAKE_MAPS))
    if unknown:
        raise ValueError(f"unknown bake map(s): {unknown}")
    if not chosen:
        raise ValueError("at least one bake map is required")
    return {
        "op": "clay_bake",
        "high_glb": str(high_glb),
        "low_glb": str(low_glb),
        "out_glb": str(out_glb),
        "result_path": str(result_dir / ".clay_bake_result.json"),
        "texture_size": int(texture_size),
        "cage_extrusion": float(cage_extrusion),
        "maps": chosen,
    }
