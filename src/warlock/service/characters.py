"""The door for a character: a recipe in, a finished mesh and a queued sheet out.

One press has to produce the whole chain -- a body, a skeleton, and 144 rendered
cells -- and the chain is still four ordinary rows rather than an orchestrator.
That is the shape ``troupe.send_to_troupe`` already establishes and this module
extends by exactly one link at the front: instead of taking a mesh the user
already has, it *builds* one from a :class:`~warlock.characters.recipe.Recipe`
and hands it straight to that door.

**Order is the whole design.** Everything knowable is refused before a byte is
written -- the recipe, the pixel block, the clip expansion, the frame plan, and
whether Blender exists at all -- because ``_charsheet_spec`` states the rule
this module obeys: an unrenderable request must cost the request, not a mesh
plus a rig plus 144 EEVEE frames. The mesh is built only once nothing left can
refuse it.

**The mesh row is minted finished**, ``import_mesh``'s arrangement line for
line: ``stage="model"``, ``status="done"``, ``params["built"]=True``. A
generated character has no generator behind it -- no seed a reconstruction could
re-roll, no reference image to re-derive from -- so ``rerun_job`` refuses it and
:func:`reroll_character` is the door that means "build that recipe again".

**``params["character"]`` rides every row of the chain** -- model, rig, sheet --
and is *nested* for the reason ``send_to_troupe`` gives about ``troupe_sheet``:
``VECTOR_PARAMS`` is an allowlist of flat settings, so a nested block cannot
have a field of it quietly promoted into a rerun vector. It is deliberately
**not** in ``DERIVED_PARAMS`` either: it is the *request*, not something a
worker learned about its output, and stripping it on a rerun would leave a
character sheet that no longer knows who it depicts.

Nothing here is humanoid-shaped. The rig template and the clip library come off
the species' archetype every time (``Family.template`` / ``Family.clip_library``),
because a door that works for one of four body plans is a door that fails for
three quarters of the registry the moment somebody picks a wolf.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import Invalid, NotFound, invalid_from
from .validation import (
    MAX_JOB_NAME,
    check_job_id,
    check_prompt,
    check_seed,
    note_degraded,
    random_seed,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..characters.recipe import Recipe
    from ..pipelines import charsheet
    from .core import WarlockService

log = logging.getLogger(__name__)

#: The ``params["character"]`` block's own format version. Bumped when a key in
#: it changes meaning -- not when the *recipe* format does, which carries its
#: own ``version`` inside ``recipe`` and moves for its own reasons.
CHARACTER_BLOCK_VERSION = 1

#: What the row is, in the two words the library and ``state.primary_action``
#: read. ``asset_intent`` is a fifth value beside the four ``create_job``
#: validates, and deliberately not passed through that door: this row is minted
#: directly, the way ``import_mesh`` mints one, so it is not a generation
#: request and never reaches that allowlist.
ASSET_TYPE = "character"
ASSET_INTENT = "character"

#: Roughly how long a character costs, for a form that has to say something
#: before the user presses the button. **An estimate, not a measurement** --
#: there is no dated document behind these two numbers, so nothing may key a
#: decision on them and they are only ever rendered as "about N minutes".
#: Rigging is the fixed half (``docs/manual/34-troupe.md``: "rigging is minutes
#: of CPU"), the cells are the linear half, and both are CPU: a character sheet
#: spends no GPU at all.
RIG_MINUTES = 1.5
SECONDS_PER_CELL = 1.0


def estimate_minutes(cells: int) -> float:
    """About how many minutes a character of *cells* frames will take.

    One function rather than the arithmetic inline in a pane, because the
    Create surface, the confirm dialog and the toast after the press all want
    the same sentence -- and three copies of a number a user is about to wait
    on is three chances to promise a different one.
    """
    count = max(0, int(cells))
    return RIG_MINUTES + count * SECONDS_PER_CELL / 60.0


def character_options(svc: WarlockService) -> dict[str, Any]:
    """Everything a character form may offer. One source for the whole surface.

    Read off the registries and off ``troupe.troupe_options`` rather than
    restated: ``recipe`` refuses against its own copies of the ladders (it may
    not import ``service``) and ``tests/characters/test_recipe.py`` owns the
    agreement between the two, so a form built from either offers the same set.
    The ladders here are the *recipe's*, because the recipe is what refuses.
    """
    from ..characters import family as family_mod
    from ..characters import recipe as recipe_mod
    from . import troupe as svc_troupe

    archetypes = family_mod.archetypes()
    families = family_mod.families()
    return {
        "archetypes": [
            {
                "key": key,
                "label": arch.label,
                "template": arch.template,
                "clip_library": arch.clip_library,
                "regions": list(arch.regions),
                "sockets": [s.name for s in arch.sockets],
            }
            for key, arch in archetypes.items()
        ],
        "families": [
            {
                "key": key,
                "label": fam.label,
                "version": fam.version,
                "archetype": fam.archetype,
                "silhouette": fam.silhouette,
                "height_m": fam.height_m,
                "aliases": list(fam.aliases),
                "nearest": list(fam.nearest),
                "themes": [{"key": t.key, "label": t.label} for t in fam.themes],
            }
            for key, fam in families.items()
        ],
        # Per species, not per archetype: the channel *set* belongs to the body
        # plan but the defaults belong to the species, and a slider column drawn
        # from the archetype's neutral defaults would start every wolf at the
        # quadruped average rather than at a wolf.
        "channels": {
            key: [
                {"key": c.key, "label": c.label, "default": c.default, "lo": c.lo, "hi": c.hi}
                for c in fam.channels
            ]
            for key, fam in families.items()
        },
        "themes": {key: list(values) for key, values in recipe_mod.THEMES.items()},
        "logical_sizes": list(recipe_mod.LOGICAL_SIZES),
        "colors": list(recipe_mod.COLOR_CHOICES),
        "outline_modes": list(recipe_mod.OUTLINE_MODES),
        "reduce_modes": list(recipe_mod.REDUCE_MODES),
        "directions": list(recipe_mod.DIRECTION_CHOICES),
        "default_recipe": recipe_mod.DEFAULT_RECIPE.as_dict(),
        # The Troupe ladders whole -- animations with their frame bounds, the
        # camera presets and their elevations, the palettes on disk, the cell
        # caps. Nested under one key rather than spread across this dict so a
        # reader can see which half of the form is Troupe's and which is the
        # character registry's.
        "troupe": svc_troupe.troupe_options(svc),
    }


def create_character(
    svc: WarlockService,
    recipe: Mapping[str, Any],
    *,
    name: str | None = None,
    prompt: str = "",
    resolution: Any = None,
) -> dict[str, Any]:
    """Build a character and queue its sheet. Two rows, one press.

    Returns ``{"id", "rig", "kind"}``: the finished model row, and the rig row
    that will mint the character sheet when it lands. Two ids and not one
    because they cancel independently and the library shows both -- the same
    property ``send_to_troupe`` keeps for the mesh it is handed.
    """
    from .. import meshreport
    from ..characters import instantiate as instantiate_mod
    from ..pipelines import postprocess
    from . import troupe as svc_troupe

    check_prompt(prompt or None)
    spec = _recipe(recipe)
    fam = spec.spec
    arch = fam.arch

    # 1. The pixel block, through the same function every Troupe door uses --
    #    so a colour count refused here is refused in Troupe's words, on
    #    Troupe's field. ``pixel_art`` rides along, and colour/outline/dither/
    #    palette are sent only when it is on: ``_check_options`` refuses an
    #    HD request (D5) that names any of the four outright, the same as
    #    every other caller of it -- and a recipe's HD switch is D5's own,
    #    per ``Recipe.pixel_art``.
    pixel_kwargs: dict[str, Any] = (
        {
            "colors": spec.colors,
            "outline": spec.outline,
            "dither": spec.dither,
            "palette": spec.palette or None,
        }
        if spec.pixel_art
        else {}
    )
    options = svc_troupe._check_options(
        svc,
        {
            "logical_size": spec.logical_size,
            "reduce_mode": spec.reduce_mode,
            "pixel_art": spec.pixel_art,
            **pixel_kwargs,
        },
    )

    # 2. The frame plan, expanded and planned -- and this time kept, not
    #    thrown away: ``_plan`` returns the ``LayoutSpec`` it resolved against
    #    the archetype's own clip library timing, and step 6 hands that
    #    exact object on to ``send_to_troupe`` rather than re-resolving the
    #    recipe's layout a second time with no timing in hand (which can only
    #    ever mean one of the closed legacy five -- see ``_plan``'s
    #    docstring).
    layout = _plan(spec, arch.clip_library, options["logical_size"])

    # 3. And Blender, before the mesh rather than after it. The chain's second
    #    row is a rig; without Blender it can never run, and a character whose
    #    body exists and whose skeleton never will is the half-built asset this
    #    ordering exists to prevent. The Rig segment's own sentence, verbatim,
    #    so the app has one wording for "this needs Blender" wherever it is met.
    from .. import doctor

    if not doctor.blender_check().ok:
        raise Invalid("Rigging needs Blender, which is not installed.")

    # 4. Seeds. The recipe's own seed is the *character's* -- it is what a
    #    sibling's procedural flames read, so it belongs in the recipe where a
    #    re-render copies it unchanged. ``mesh_seed`` is the row's, minted here
    #    because every model row carries one and ``rerun_job`` expects it.
    if not spec.seed:
        spec = spec.replace(seed=_fresh_seed())
    check_seed("seed", spec.seed)
    mesh_seed = _fresh_seed()

    block = character_block(spec, prompt=prompt, resolution=resolution)

    # 5. Now, and only now, the mesh. ``import_mesh``'s arrangement line for
    #    line, including the two non-fatal post-processing steps and the
    #    directory cleanup on any failure.
    job_id = uuid.uuid4().hex[:12]
    job_dir = svc.config.job_dir(job_id)
    params: dict[str, Any] = {
        # 0 for the reference stage that never ran, exactly as a built mesh
        # records it: there is no image behind this body.
        "seed": 0,
        "mesh_seed": mesh_seed,
        # An input, not a derived value. It is what ``rerun_job`` reads to
        # refuse a regenerate that has nothing to regenerate -- a character is
        # rebuilt from its recipe by ``reroll_character``, not by a new seed
        # through a reconstruction that was never run.
        "built": True,
        "character": block,
        "family": fam.key,
        "asset_type": ASSET_TYPE,
        "asset_intent": ASSET_INTENT,
    }
    joints: list[dict[str, Any]]
    try:
        instance = instantiate_mod.instantiate(spec, job_dir)
        joints = instance.joints
        # Recorded from what was *built* rather than from what was asked for: a
        # recipe may name an older family version, and the assets on disk are
        # this build's. A row claiming a version it does not carry is how a
        # saved character comes back looking like somebody else.
        params["family_version"] = instance.version
        model = job_dir / instantiate_mod.MODEL_NAME
        try:
            transform = postprocess.normalize_glb(model, fam.height_m)
            params["transform"] = transform
            params["scale_factor"] = transform["scale"]
            # **And the skeleton moves with the mesh.** ``instantiate`` already
            # grounds and scales to the species' height, so this is very nearly
            # the identity -- but "very nearly" is the whole problem: joints
            # left in the pre-normalisation frame sit millimetres off in every
            # pose, and that reads as bad weights rather than as a transform
            # nobody applied.
            joints = _moved_joints(joints, transform)
        except Exception as exc:
            # Logged, swallowed and *recorded*, ``import_mesh``'s handling of
            # the same step and for ART-01's reason: this row is inserted
            # ``done``, so without the note the user has a successful-looking
            # character whose pivot and scale are the engine's.
            log.exception("normalize failed for character %s; leaving the mesh as-is", job_id)
            note_degraded(
                params,
                "normalize",
                f"the character was not centred, grounded or resized ({exc}); its "
                f"pivot and scale are the engine's",
            )
        try:
            params["mesh_report"] = meshreport.build(model, target_size_m=fam.height_m)
        except Exception as exc:
            log.exception("mesh report failed for character %s", job_id)
            note_degraded(
                params,
                "report",
                f"the mesh could not be measured ({exc}); size, triangle count "
                f"and watertightness are unknown for this asset",
            )
        svc.store.create(
            "image", prompt or spec.name, params, job_id, stage="model", status="done"
        )
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    label = (name or spec.name or "").strip()
    if label:
        svc.store.set_meta(job_id, name=label[:MAX_JOB_NAME])

    # 6. And the rest of the chain, through the door that already knows how to
    #    build it. ``template`` and ``bones`` are the two things this door knows
    #    that no other caller does: the species' own skeleton, stated exactly,
    #    which also withholds ``joints="measured"`` -- measuring reads joints
    #    off a reference image this character never had.
    try:
        # No ``front_yaw`` here, and there never will be: it is set by orbiting
        # a viewport and pressing a button, and a registry-built character has
        # no viewport session behind it to have done that in. Absent means the
        # sheet renders its front row from yaw 0, same as every mesh before
        # this feature existed.
        rig = svc_troupe.send_to_troupe(
            svc,
            job_id,
            logical_size=spec.logical_size,
            reduce_mode=spec.reduce_mode,
            elevation=spec.elevation,
            name=spec.name,
            # ``layout`` is what step 2 already resolved against the
            # archetype's own clip library timing -- not
            # ``spec.layout_dict()``, which resolves the recipe's payload
            # with no timing at all and so can only ever mean one of the
            # closed legacy five. See ``_plan``'s docstring.
            layout=layout.as_dict(),
            template=arch.template,
            bones=joints,
            character=block,
            pixel_art=spec.pixel_art,
            **pixel_kwargs,
        )
    except Exception:
        # The 2026-09-07 audit (troupe-01) found a failure here left a
        # permanently orphaned "done" mesh row with no rig and no sheet --
        # this module's own docstring claims "the mesh is built only once
        # nothing left can refuse it", which was untrue the moment this call
        # could still fail after the row above was committed. Undo the row
        # the same way a failed ``instantiate`` undoes its directory, so a
        # refusal this late costs the request exactly as one raised earlier
        # would have.
        svc.store.delete(job_id)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    return {"id": job_id, "rig": rig["id"], "kind": ASSET_TYPE}


def reroll_character(svc: WarlockService, job_id: str) -> dict[str, Any]:
    """Build that character again, from the same recipe and a fresh seed.

    ``rerun_job`` refuses a character -- ``built`` is on the row and there is no
    generator behind it -- so this is the door "make me another" means. It is a
    door that copies ``params``, which puts it under
    ``test_every_door_that_copies_params_rerolls_the_seeds``: the recipe seed is
    re-rolled here and ``mesh_seed`` is minted fresh by ``create_character``, so
    neither comes through unchanged. A reroll that reproduced the previous
    seed would look like it ran, take the time, and return the same character.
    """
    check_job_id(job_id)
    row = svc.require_job(job_id)
    params = row.get("params") or {}
    block = params.get("character")
    if not isinstance(block, Mapping):
        raise NotFound("that job is not a character, so there is no recipe to build again")
    recipe = dict(block.get("recipe") or {})
    previous = recipe.get("seed")
    recipe["seed"] = _fresh_seed(avoid=previous if isinstance(previous, int) else None)
    return create_character(
        svc,
        recipe,
        name=str(row.get("name") or "") or None,
        prompt=str(block.get("prompt") or ""),
        resolution=block.get("resolution"),
    )


def preview_character(svc: WarlockService, recipe: Mapping[str, Any]) -> Path:
    """Build one character into a temp GLB for the viewer. **No row.**

    A preview is not an asset: it exists for as long as a slider is being
    dragged, and minting a library row per drag would bury the user's actual
    work under a hundred near-identical ogres. It lands under ``data_dir/tmp``
    keyed by a hash of the whole recipe, so dragging a slider back to where it
    was is a file that already exists rather than a rebuild.
    """
    from ..characters import instantiate as instantiate_mod

    spec = _recipe(recipe)
    digest = hashlib.sha256(
        json.dumps(spec.as_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    tmp_dir = svc.config.data_dir / "tmp"
    dest = tmp_dir / f"character-preview-{digest}.glb"
    if dest.exists():
        return dest
    # Built into a scratch directory and then moved: ``instantiate`` writes
    # three files under one name, and the served name here is the GLB alone.
    work = tmp_dir / f".character-preview-{digest}.{uuid.uuid4().hex[:8]}"
    try:
        instance_dir = work
        instantiate_mod.instantiate(spec, instance_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # ``os.replace``, never ``shutil.move``: the 2026-09-07 audit
        # (service-02) found that ``shutil.move`` onto a destination that
        # already exists degrades to non-atomic ``copy2`` on Windows -- two
        # concurrent previews racing the same digest could tear a GLB a
        # viewer is mid-read on, which is the exact hazard this comment used
        # to claim was handled. ``os.replace`` is atomic on the same volume
        # even when ``dest`` exists, which is the case ``dest.exists()``
        # above did not already return early for.
        os.replace(str(instance_dir / instantiate_mod.MODEL_NAME), str(dest))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return dest


def export_package(
    svc: WarlockService,
    job_id: str,
    sheet_id: str,
    dest_dir: Any = None,
) -> dict[str, Any]:
    """Copy one character sheet and its sidecar out as a pair.

    The deliverable is **both files**: the PNG is the atlas and the JSON is what
    says which cell is ``walk`` facing south-east. A folder that has one without
    the other holds an asset nothing can interpret, which is why the copy goes
    through ``export.staged_copy_all`` rather than two ``copyfile`` calls.

    Refuses in ``export_to_folder``'s exact words when no destination is given
    and none is configured -- one sentence for one condition, wherever it is met.
    """
    from .. import rigging
    from . import export as svc_export

    check_job_id(job_id)
    job = svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    if not rigging.is_valid_id(str(sheet_id or "")):
        raise Invalid("that is not a sheet id", field="sheet_id")

    dest = Path(dest_dir) if dest_dir is not None else svc.config.export_dir
    if dest is None:
        raise NotFound("no export folder configured (set WARLOCK_EXPORT_DIR)")

    png = rigging.sheet_png_path(job_dir, str(sheet_id))
    sidecar = rigging.sheet_path(job_dir, str(sheet_id))
    if not png.exists() or not sidecar.exists():
        raise NotFound("that sheet is no longer on disk", field="sheet_id")

    # Named after the *job*, not after the sheet id: a folder full of
    # ``a3f01c9b2d4e.png`` is a folder nobody can read, and the name is the one
    # thing the user actually chose. Falls back to the id when there is none.
    stem = _package_stem(job, str(sheet_id))
    dest.mkdir(parents=True, exist_ok=True)
    svc_export.staged_copy_all(
        [(png, dest / f"{stem}.png"), (sidecar, dest / f"{stem}.json")]
    )
    return {
        "png": str(dest / f"{stem}.png"),
        "json": str(dest / f"{stem}.json"),
        "dir": str(dest),
    }


#: A clip name used as a frame-folder path component must be plain: lowercase
#: ASCII, digits and underscores only. Everything a movement or clip can
#: actually be named already satisfies this in the shipped registries -- the
#: legacy five, the templates' clip libraries -- but a rig's clip library is
#: user-authored (Poser), and this is the one door that turns a clip's name
#: into a folder on the caller's own disk rather than a JSON string, so it
#: gets its own refusal instead of trusting the name a second time.
_CLIP_FOLDER_NAME_RE = re.compile(r"^[a-z0-9_]+$")


def export_frames(
    svc: WarlockService,
    job_id: str,
    sheet_id: str,
    dest_dir: Any = None,
) -> Path:
    """One folder per clip, one subfolder per compass direction, one PNG per frame.

    Where :func:`export_package` hands over the atlas whole, this cuts it up:
    an engine that wants ``AnimatedSprite2D``-style frame folders rather than
    an atlas-plus-sidecar pair gets ``<clip>/<COMPASS>/<nnn>.png`` for every
    cell, named the way
    ``docs/measurements/2026-09-12-troupe-open-clip-vocabulary.md``'s compass
    table names them -- never the raw facing key, which is Troupe's own
    internal spelling (``front_left``) and not a direction word an importer
    should have to know.

    The frame table comes from the sidecar's own ``"troupe"`` block -- the
    immutable snapshot ``_q_troupe`` wrote the moment the sheet was rendered
    -- rather than from today's registries, for the reason every re-render
    door already gives: a clip a user has since renamed or retimed must not
    reshuffle a sheet that was already handed to a player. A sheet with no
    ``"troupe"`` block (drawn by hand, or rendered before this feature
    existed) resolves through the closed legacy table instead, via
    ``charsheet.resolve_layout(None)`` -- the same fallback ``troupe_options``
    and every default row use.
    """
    from PIL import Image

    from .. import rigging
    from ..pipelines import charsheet
    from . import export as svc_export

    check_job_id(job_id)
    job = svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    if not rigging.is_valid_id(str(sheet_id or "")):
        raise Invalid("that is not a sheet id", field="sheet_id")

    dest = Path(dest_dir) if dest_dir is not None else svc.config.export_dir
    if dest is None:
        raise NotFound("no export folder configured (set WARLOCK_EXPORT_DIR)")

    png_path = rigging.sheet_png_path(job_dir, str(sheet_id))
    record = rigging.read_sheet(job_dir, str(sheet_id))
    if record is None or not png_path.exists():
        raise NotFound("that sheet is no longer on disk", field="sheet_id")

    troupe_block = record.get("troupe")
    layout = (
        troupe_block
        if isinstance(troupe_block, Mapping)
        else charsheet.resolve_layout(None).as_dict()
    )
    runs = layout.get("runs") or []
    movements = {m["key"]: m for m in (layout.get("movements") or [])}
    layout_fps = layout.get("fps")

    cells = record.get("cells") or []
    cell_by_index = {int(c["index"]): c for c in cells}
    frame_size = int(record.get("frame_size") or 0)
    if not frame_size:
        # Every 3D character sheet is square by construction (``charsheet.plan``
        # never sets ``frame_w``/``frame_h``) -- a 0 here means this sidecar is
        # something else ``sheet.sidecar`` also writes, not a Troupe sheet.
        raise Invalid(
            "that sheet is not square, so it cannot export to frame folders",
            field="sheet_id",
        )

    stem = _package_stem(job, str(sheet_id))
    # **Top-level first, then the recipe, then True.** ``_q_troupe`` writes
    # the worker's own D5 answer at the sidecar's top level (``"pixel_art":
    # False``, absent when the render was pixel art) -- the recipe's nested
    # copy is what was *asked for* and can be a species theme's HD default
    # rather than what this particular sheet actually rendered, so reading it
    # first mislabelled every HD Troupe sheet's manifest as pixel art.
    raw_pixel_art = record.get("pixel_art")
    if raw_pixel_art is None:
        raw_pixel_art = ((record.get("character") or {}).get("recipe") or {}).get(
            "pixel_art", True
        )
    pixel_art = bool(raw_pixel_art)

    # **Refused, not crashed.** A sheet with no ``"troupe"`` block whose cells
    # do not happen to match the closed legacy 256-cell table (a hand-drawn
    # sheet, or one from a format ``sheet.sidecar`` also writes) fell through
    # to a bare ``KeyError`` inside ``write`` the moment the plan below asked
    # for a cell index the sidecar never recorded. Checked here, before a
    # single frame is cropped -- the same "refused before anything renders"
    # rule every other export door in this module already keeps.
    needed_indices = {
        i for run in runs for i in range(int(run["start"]), int(run["end"]) + 1)
    }
    if not needed_indices <= set(cell_by_index):
        raise Invalid(
            "this sheet's cells don't match a character layout; export it as "
            "a package instead",
            field="sheet_id",
        )

    def write(tmp: Path) -> None:
        plan: list[tuple[str, str, int, int]] = []
        seen_folders: set[tuple[str, str]] = set()
        clip_directions: dict[str, list[str]] = {}
        for run in runs:
            clip = str(run["movement"])
            if not _CLIP_FOLDER_NAME_RE.match(clip):
                raise Invalid(
                    f"the clip {clip!r} cannot be used as an export folder name "
                    "(lowercase letters, digits and underscores only)",
                    field="sheet_id",
                )
            compass = charsheet.compass_name(float(run["yaw"]))
            key = (clip, compass)
            if key in seen_folders:
                raise Invalid(
                    f"two runs of {clip!r} would both export to {clip}/{compass}",
                    field="sheet_id",
                )
            seen_folders.add(key)
            clip_directions.setdefault(clip, []).append(compass)
            plan.append((clip, compass, int(run["start"]), int(run["end"])))

        with Image.open(png_path) as opened:
            opened.load()
            atlas = opened.convert("RGBA")
        try:
            for clip, compass, start, end in plan:
                folder = tmp / clip / compass
                folder.mkdir(parents=True, exist_ok=True)
                for frame_index, cell_index in enumerate(range(start, end + 1)):
                    cell = cell_by_index[cell_index]
                    box = (
                        cell["x"],
                        cell["y"],
                        cell["x"] + cell["w"],
                        cell["y"] + cell["h"],
                    )
                    atlas.crop(box).save(folder / f"{frame_index:03d}.png", "PNG")
        finally:
            atlas.close()

        clips_meta: dict[str, Any] = {}
        for clip, directions in clip_directions.items():
            movement = movements[clip]
            duration_ms = int(movement["duration_ms"])
            fps = float(layout_fps) if layout_fps is not None else 1000.0 / duration_ms
            clips_meta[clip] = {
                "loop": bool(movement["loop"]),
                "frames": int(movement["frames"]),
                "duration_ms": duration_ms,
                "fps": fps,
                "directions": directions,
            }
        manifest = {
            "format": "warlock-frames",
            "version": 1,
            "name": stem,
            "frame_size": frame_size,
            "pixel_art": pixel_art,
            "clips": clips_meta,
        }
        (tmp / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    return svc_export.staged_tree(dest, stem, write)


def export_godot(
    svc: WarlockService,
    job_id: str,
    dest_dir: Any = None,
) -> Path:
    """A Godot 4 scene beside a renamed copy of the served ``animated.glb``.

    The served file is never touched -- only read -- because it is what every
    other export and every re-render of it derives from, and Godot's own loop
    heuristic (``godotscene``'s docstring) reads a ``"-loop"`` suffix as
    licence to loop an animation, which is a rename this app's own copy must
    never carry. ``godotscene.rename_animations`` runs on a copy of the bytes
    instead, and only the copy lands in the export folder.

    Refuses in ``animated.glb``'s own words for an unrigged mesh or a
    skeleton with no clips authored: :func:`derive.get_file` is called
    directly rather than re-implemented, so a fresh rig-without-clips
    sentence and this door's sentence can never drift apart.
    """
    from .. import glbio, godotscene
    from . import derive as svc_derive
    from . import export as svc_export

    check_job_id(job_id)
    job = svc.require_job(job_id)

    dest = Path(dest_dir) if dest_dir is not None else svc.config.export_dir
    if dest is None:
        raise NotFound("no export folder configured (set WARLOCK_EXPORT_DIR)")

    # animated.glb's own door -- its refusal for an unrigged mesh (NotReady,
    # "This asset has not been rigged yet.") or an unauthored skeleton reaches
    # the caller exactly as that door states it, because this calls it rather
    # than restating ``files.ready``/``unready_reason``.
    animated_path = svc_derive.get_file(svc, job_id, "animated.glb")
    data = animated_path.read_bytes()

    extras = glbio.root_extras(data)
    stamp = extras.get("warlock_animation") if isinstance(extras, Mapping) else None
    loops = set(stamp.get("loops") or []) if isinstance(stamp, Mapping) else set()
    names = glbio.animation_names(data)

    stem = _package_stem(job, job_id)
    try:
        # **Inside the try, not before it.** ``godot_clip_name`` raises
        # ``ValueError`` for a clip name Godot's ``&"..."`` StringName syntax
        # cannot hold (a quote, a backslash, a newline) -- building this
        # mapping ahead of the ``try`` let that escape as a bare
        # ``ValueError`` instead of the refusal below.
        mapping = {
            name: godotscene.godot_clip_name(name, name in loops)
            for name in names
            if godotscene.godot_clip_name(name, name in loops) != name
        }
        renamed = glbio.rename_animations(data, mapping)
        scene = godotscene.scene_text(
            root_name=stem,
            glb_file=f"{stem}.glb",
            clips=[(name, name in loops) for name in names],
        )
    except ValueError as exc:
        raise invalid_from(
            exc, "That character cannot be exported to Godot", field="job_id"
        ) from exc

    def write(tmp: Path) -> None:
        (tmp / f"{stem}.glb").write_bytes(renamed)
        (tmp / f"{stem}.tscn").write_text(scene, encoding="utf-8")

    return svc_export.staged_tree(dest, stem, write)


# --- the pieces the doors above share ----------------------------------------


def character_block(
    spec: Recipe, *, prompt: str = "", resolution: Any = None
) -> dict[str, Any]:
    """The ``params["character"]`` block: who this is, and what was asked for.

    Nested rather than flattened onto the row, for the reason ``send_to_troupe``
    gives about ``troupe_sheet``: ``VECTOR_PARAMS`` is an allowlist of *flat*
    settings, so nesting is what stops ``seed`` or ``palette`` inside a recipe
    being picked up as a rerun vector for the row that carries it.

    ``resolution`` takes a :class:`~warlock.characters.resolve.Resolution`, the
    dict one serialises to, or nothing -- the three shapes a caller actually
    holds. Absent means an empty resolution rather than a missing key: the block
    has one shape on every row, and a reader that has to test for the key is a
    reader that will forget to.
    """
    from ..characters.resolve import Resolution

    if resolution is None:
        resolved = Resolution().to_dict()
    elif isinstance(resolution, Resolution):
        resolved = resolution.to_dict()
    elif isinstance(resolution, Mapping):
        resolved = Resolution.from_dict(resolution).to_dict()
    else:
        raise Invalid("that is not a prompt resolution", field="prompt")
    return {
        "version": CHARACTER_BLOCK_VERSION,
        "recipe": spec.as_dict(),
        "prompt": str(prompt or ""),
        "resolution": resolved,
    }


def _recipe(raw: Mapping[str, Any]) -> Recipe:
    """``Recipe.from_dict``, with the refusal framed for a person.

    ``CharacterError`` carries the ``field`` it came from, and ``invalid_from``
    passes that address straight through, so a bad colour count rings the
    colour combo rather than raising a wall of text at the top of the form.
    """
    from ..characters.errors import CharacterError
    from ..characters.recipe import Recipe

    try:
        return Recipe.from_dict(raw)
    except CharacterError as exc:
        raise invalid_from(exc, "That character cannot be made", field=exc.field) from exc


def _plan(spec: Recipe, clip_library: str, frame_size: int) -> charsheet.LayoutSpec:
    """Expand the clips and plan the frames -- the render is thrown away, the
    resolved layout is not.

    ``create_charsheet`` does exactly this planning step and says why: a clip
    library that does not fill the frame table, or a size whose atlas is over
    the texture limit, is refused now instead of failing a job that has
    already rendered. The library is the **archetype's**, never a constant --
    a wolf is animated from the quadruped clips, and pinning humanoid here
    would fill a quadruped frame table from a library whose skeleton it does
    not have.

    **The returned ``LayoutSpec`` is what ``create_character`` sends on to
    ``send_to_troupe``, not ``spec.layout_dict()``.** ``Recipe.layout_dict``
    resolves the recipe's v2/v3 *payload* with no ``timing`` at all, so it can
    only ever mean one of the closed :data:`charsheet.ANIMATIONS` five -- a
    recipe naming ``hit`` or any other clip the archetype's own library
    defines resolved here, against *this* ``timing``, and nowhere else, and
    then failed a second, untimed resolution downstream with a bare
    ``'hit' is not a Troupe movement`` after the mesh row was already
    committed. Reusing this call's own answer is what keeps there being only
    one resolution of the recipe's layout in the whole door.

    ``timing`` is passed the same way ``troupe._timed_layout`` passes it, so
    a recipe may name any clip *clip_library* actually defines -- not just the
    closed :data:`charsheet.ANIMATIONS` five -- per
    ``docs/measurements/2026-09-12-troupe-open-clip-vocabulary.md``. A clip the
    library has never heard of surfaces as ``resolve_layout``'s own
    ``ValueError`` now rather than ``expand_clips``' ``KeyError``, and falls
    into the same ``field="layout"`` branch below either way -- the existing
    ``Invalid`` path, unchanged.
    """
    from ..clips import clip_timing, expand_clips
    from ..pipelines import charsheet

    try:
        timing = clip_timing(clip_library)
        layout = charsheet.resolve_layout(spec.layout_payload(), timing=timing)
        records = expand_clips(clip_library, layout)
        charsheet.plan(
            records,
            frame_size=frame_size,
            elevation=spec.elevation,
            lighting="flat",
            layout=layout,
        )
    except KeyError as exc:
        raise Invalid(
            f"the {clip_library} clip library is missing {exc}", field="animations"
        ) from exc
    except ValueError as exc:
        # field="layout", the 2026-09-11 audit's finding troupe-01: the same
        # plan-and-throw-away shape ``troupe.check_troupe``/``create_charsheet``
        # carry, with the same fieldless branch -- see the comment in
        # ``troupe.check_troupe``.
        raise invalid_from(exc, "That character cannot be laid out", field="layout") from exc
    return layout


def _moved_joints(
    joints: list[dict[str, Any]], transform: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The skeleton through the grounding transform ``normalize_glb`` applied.

    **The axis swap is the load-bearing part.** ``normalize_glb`` reports its
    scale and translation in *glTF* axes (Y up), and the joints are in
    ``rigging.validate_joints``' shape, which is *Blender* axes (Z up). The
    scale is uniform and survives the swap unchanged; the translation does not,
    and applying it raw puts a character's feet through the floor along the
    wrong axis -- which looks exactly like a bad rig rather than like a
    coordinate convention nobody converted.
    """
    import numpy as np

    from ..characters.instantiate import transformed_joints

    scale = float(transform.get("scale", 1.0) or 1.0)
    tx, ty, tz = (float(v) for v in transform.get("translation", (0.0, 0.0, 0.0)))
    matrix = np.eye(4, dtype="f8")
    matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = scale
    # glTF (x, y, z) -> Blender (x, -z, y): ``instantiate._to_blender``, which
    # is the function that put these joints in Blender axes in the first place.
    matrix[0, 3], matrix[1, 3], matrix[2, 3] = tx, -tz, ty
    return transformed_joints(joints, matrix)


def _package_stem(job: Mapping[str, Any], sheet_id: str) -> str:
    """A filesystem-safe stem for the exported pair, from the job's name."""
    raw = str(job.get("name") or "").strip()
    safe = "".join(c if c.isalnum() or c in "-_ " else "-" for c in raw).strip()
    return safe[:MAX_JOB_NAME] if safe else sheet_id


def _fresh_seed(*, avoid: int | None = None) -> int:
    """A seed a recipe can actually carry, and one that is not ``avoid``.

    ``random_seed`` may answer 0, and 0 is the recipe's word for "nobody chose
    a seed" -- so a character built on a rolled zero would be re-rolled to a
    different one by the next door that read it. ``avoid`` is what makes a
    reroll observably a reroll rather than a very unlucky coincidence.
    """
    while True:
        seed = random_seed()
        if seed and seed != avoid:
            return seed
