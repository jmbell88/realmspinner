"""The door for Troupe: the T-pose reference block, and the character sheet.

Two entry points, because the program is deliberately two steps with a human
gate between them, and each step is a row the user can keep on its own:

* ``check_troupe`` validates the *request for a follow-up* that rides on a
  reference job, exactly as ``_jobs_create._check_sprite_sheet`` does for the
  sprite path -- and for the same reason, spelled out there: the worker mints
  the follow-up row itself, so a bad option discovered at that point would be a
  refusal an hour later on a row the user never submitted.
* ``create_charsheet`` is the direct door, for a mesh that already exists --
  a supplied base mesh, or a second sheet at a different size from the same
  character. Every refusal a character sheet has lives here rather than in the
  worker: an unrenderable request should cost the request, not a place in the
  queue and 256 EEVEE frames.

``expand_clips`` is re-exported from ``warlock.clips``: the worker needs the
same function and may not import ``service``, which is why that module exists.

The numbers this module offers come from ``pipelines.charsheet`` and
``pipelines.pixelize`` rather than being restated: those are the modules the
worker actually plans and reduces with, and a second copy here would be one
edit away from a form that offers a size the renderer refuses.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .. import followups, rigging
from ..clips import clip_timing, expand_clips
from ..pipelines import charsheet, pixelize, spritesynth
from .errors import Conflict, Invalid, NotFound, invalid_from
from .sheets import check_sheet_cap
from .validation import DERIVED_PARAMS, check_job_id, check_vram

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core import WarlockService

#: Which guide the reference stage conditions on. ``spritesynth`` owns the
#: list because it owns the templates.
TROUPE_VARIANTS: tuple[str, ...] = spritesynth.TPOSE_VARIANTS

#: And in which pose. The second axis of the same choice, crossed with the
#: variant rather than folded into it -- see ``spritesynth.REFERENCE_POSES``.
TROUPE_POSES: tuple[str, ...] = spritesynth.REFERENCE_POSES

#: The sizes a sheet may be laid out at. ``charsheet.SIZES``, which is also
#: what ``charsheet.plan`` validates against.
TROUPE_LOGICAL_SIZES: tuple[int, ...] = charsheet.SIZES

#: **Task G, 2026-09-12.** Below the ladder's floor, 8px, a cell holds too few
#: pixels for the outline and reduce passes to leave anything a player could
#: read as a character -- the same floor ``charsheet.MIN_FRAME_SIZE`` plans
#: against. Above 256px, a sheet is heading toward the atlas ceiling fast (32
#: columns * 256px is already 8192, ``sheet.MAX_ATLAS_PX``) and a size that big
#: has no reason to come from a hand-typed box rather than a fresh render at a
#: bigger ``RENDER_SIZE``. Between them, any whole number is a size
#: ``charsheet.plan``/``_q_troupe`` already build correctly -- the ladder is a
#: set of *presets*, not the renderer's actual limit -- so a size off the
#: ladder is not a mistake, only a choice with no button for it until now.
TROUPE_CUSTOM_SIZE_RANGE: tuple[int, int] = (8, 256)

#: Palette budgets, when no designed palette is named. The same ladder the
#: sprite path offers, and for the same reason: these are the counts a median
#: cut produces a usable sprite palette at.
TROUPE_COLOR_CHOICES: tuple[int, ...] = (8, 16, 32, 64)

TROUPE_OUTLINE_MODES: tuple[str, ...] = pixelize.OUTLINE_MODES
TROUPE_REDUCE_MODES: tuple[str, ...] = pixelize.REDUCE_MODES

DEFAULT_TROUPE_VARIANT = "male"

#: **A-pose, chosen 2026-08-23 over the T-pose that shipped before it.** The
#: shipped humanoid rig template is itself an A-pose, so this is the pose whose
#: mesh the template fits directly -- no joints measured off its vertices, and
#: so no dependency on the ViTPose weights a bare install does not have. The
#: T-pose is still on offer, and still the better reconstruction: it separates
#: the limbs more, which is the one thing a single view most needs.
DEFAULT_TROUPE_POSE = "apose"
DEFAULT_TROUPE_LOGICAL_SIZE = 32
DEFAULT_TROUPE_COLORS = 64
DEFAULT_TROUPE_OUTLINE = "outer"

#: How many ``charsheet`` rows deep the settings lookup looks. The library's
#: own page size, and ``studio.troupe_mode``'s: a sheet older than this is one
#: whose settings the door reports as no longer on record, by name.
_SCAN_LIMIT = 400

#: How far a candidate charsheet's own ``created_at`` may sit from the rig's
#: ``finished_at`` and still be recognised as *that* rig's follow-up.
#: ``_q_jobs._maybe_queue_sheet_after_rig`` runs in the same terminal-write
#: step that calls ``store.finish`` (queue.py's ``elif status == "done":``
#: block: the rig's ``finished_at`` is written first, and the follow-up sheet
#: is minted a few ``await``s later in the same step, normally milliseconds
#: apart) -- so this is generous headroom for scheduling jitter, not a
#: measured bound. Wide enough to never miss a genuine follow-up, narrow
#: enough that a human's later, independent ``create_charsheet`` call with
#: identical settings on the same mesh -- the false positive this window
#: exists to close -- has to land within a minute of the rig finishing to be
#: mistaken for it.
FOLLOW_UP_WINDOW_S = 60.0

#: The rig template every door here *defaults* to, and no longer the only one
#: allowed: what a character sheet actually needs is a template with clips
#: authored for it, which is what ``create_charsheet`` refuses on. The pin
#: survives as the default because it is the template the shipped clip library
#: carries -- and ``rigging.clip_library`` answers "no clips" for the rest
#: rather than failing, so without a refusal at the door the mismatch would
#: land in the worker as a frame-count error.
TROUPE_TEMPLATE = "humanoid"

#: Pinned rather than inherited from the character's row, the rule
#: ``_maybe_queue_sprite_sheet`` states at length: the guide is a ControlNet
#: hint, so the reference stage needs a base that can run one.
TROUPE_BASE_MODEL = "sdxl_cfg"


def has_clips(template: str) -> bool:
    """Whether a skeleton has a clip library to animate a sheet from.

    ``create_charsheet``'s question, asked as a function so the *door* offering
    the choice and the door refusing it read the same fact. An unknown or
    unrecorded template is False rather than an error, for the reason the
    refusal gives: from the user's side it is the same fact.
    """
    try:
        return bool(rigging.clip_library(str(template or "")).get("clips"))
    except ValueError:
        return False


def clip_templates() -> list[dict[str, str]]:
    """The skeletons a character sheet can actually be animated on.

    ``rigging.catalog()`` filtered by :func:`has_clips`, in the catalog's own
    order, so the Skeleton picker offers exactly the set the door accepts.
    """
    return [row for row in rigging.catalog() if has_clips(row["key"])]


def _clip_vocabulary(template_key: str) -> list[dict[str, Any]]:
    """*template_key*'s whole clip library, as the options block states it.

    ``troupe_options``' answer to what a layout may name a movement on this
    skeleton, beyond the closed :data:`charsheet.ANIMATIONS` five --
    ``clip_timing`` for the numbers every :class:`charsheet.ClipTiming` needs,
    and the library's own ``provisional`` flag alongside it, because a form
    offering a provisional clip has to be able to say so. ``default`` marks a
    name that is also one of the five the legacy layout already carries -- the
    boundary a pane can use to group "the sheet always had this" from "this
    rig also offers". See
    ``docs/measurements/2026-09-12-troupe-open-clip-vocabulary.md``.
    """
    library = rigging.clip_library(template_key)
    timing = clip_timing(template_key)
    legacy_names = {name for name, *_rest in charsheet.ANIMATIONS}
    out: list[dict[str, Any]] = []
    for clip in library["clips"]:
        name = str(clip["name"])
        clip_time = timing[name]
        out.append(
            {
                "name": name,
                "frames": clip_time.frames,
                "loop": clip_time.loop,
                "duration_ms": clip_time.duration_ms,
                "provisional": bool(clip.get("provisional", False)),
                "default": name in legacy_names,
            }
        )
    return out


def troupe_options(svc: WarlockService) -> dict[str, Any]:
    """What a Troupe request may ask for. One source for the form."""
    from . import palettes

    return {
        "variants": list(TROUPE_VARIANTS),
        # Derived, never a second hand-written list: what a character sheet
        # needs is a template with clips authored for it, and
        # ``rigging.clip_library`` is the one answer to that question. A
        # second list here would be one edit away from offering a skeleton
        # ``create_charsheet`` then refuses.
        "clip_templates": clip_templates(),
        "poses": list(TROUPE_POSES),
        "logical_sizes": list(TROUPE_LOGICAL_SIZES),
        # A pair, not a ladder: the "Custom..." size box clamps to this range
        # rather than offering a third list, since every whole number in it is
        # equally valid and there is no preset worth naming among them.
        "logical_size_range": list(TROUPE_CUSTOM_SIZE_RANGE),
        "colors": list(TROUPE_COLOR_CHOICES),
        "outline_modes": list(TROUPE_OUTLINE_MODES),
        "reduce_modes": list(TROUPE_REDUCE_MODES),
        "palettes": palettes.available(svc.config),
        "animations": [
            {
                "name": name,
                "frames": frames,
                "min_frames": charsheet.movement_min_frames(name),
                "max_frames": charsheet.MAX_FRAMES,
                "loop": loop,
                "duration_ms": ms,
            }
            for name, frames, loop, ms in charsheet.ANIMATIONS
        ],
        # ``template_key -> its whole clip library``, for every template a
        # sheet can actually be animated on -- the open vocabulary beyond the
        # closed five above. See :func:`_clip_vocabulary`.
        "clip_vocabulary": {
            row["key"]: _clip_vocabulary(row["key"]) for row in clip_templates()
        },
        "fps_choices": list(charsheet.FPS_CHOICES),
        "directions": [name for name, _yaw in charsheet.DIRECTIONS],
        # Read from ``charsheet`` rather than restated, this module's rule: the
        # worker frames the render from that same table, and a second copy here
        # would be one edit away from a form offering an angle nothing renders.
        "camera_presets": {
            key: {"label": label, "elevation": elevation}
            for key, label, elevation in charsheet.CAMERA_PRESETS
        },
        "direction_presets": list(charsheet.DIRECTION_PRESETS),
        "cells": len(charsheet.frame_table()),
        "warn_cells": charsheet.WARN_CELLS,
        "max_cells": charsheet.MAX_CELLS,
        "render_size": charsheet.RENDER_SIZE,
        "defaults": {
            "variant": DEFAULT_TROUPE_VARIANT,
            "pose": DEFAULT_TROUPE_POSE,
            "logical_size": DEFAULT_TROUPE_LOGICAL_SIZE,
            "colors": DEFAULT_TROUPE_COLORS,
            "outline": DEFAULT_TROUPE_OUTLINE,
            "reduce_mode": TROUPE_REDUCE_MODES[0],
            "camera": charsheet.DEFAULT_CAMERA_PRESET,
            "template": TROUPE_TEMPLATE,
            "layout": charsheet.resolve_layout().as_dict(),
            # D5 HD mode: on by default, so a form that never touches the
            # switch mints the same row it always has -- see ``_check_options``.
            "pixel_art": True,
        },
    }


def _check_options(svc: WarlockService, entries: dict[str, Any]) -> dict[str, Any]:
    """The pixelisation options every Troupe path shares, validated once.

    A thin wrapper over ``pixelopts.check_pixel_options`` since 2026-08-29:
    the body was the same four refusals every other pixel path needs, and the
    only Troupe-shaped things in it were the two ladders and the default
    outline, which are the parameters. Kept as a name here rather than having
    the three doors below call the shared function directly, because *this* is
    where the Troupe defaults live and a door should not have to restate them
    -- the delegation rule this module's docstring states, applied to itself.

    **D5 HD mode.** ``pixel_art`` (default True) is Troupe's own switch, not
    ``check_pixel_options``': a request that turns it off wants an unreduced,
    unpalletted render, so colour count, an authored palette, dithering and an
    outline pass are all questions this render never asks. Checked on the raw
    entries *before* ``check_pixel_options`` fills in its own defaults --
    catching the value the caller actually sent rather than the default
    ``check_pixel_options`` would otherwise substitute for it -- and refused on
    the option's own field, the same rule ``check_pixel_options`` already
    applies to ``outline``/``reduce_mode`` on a path that has neither. The row
    then carries ``"pixel_art": False`` and drops ``colors``/``palette``/
    ``dither``/``outline``/``reduce_mode`` outright, so ``_q_troupe`` never
    sees a value it would apply. ``True`` writes no key at all, so a form
    that never touches the switch mints the byte-identical row it always has.

    **Only a real bool, or absence, answers.** ``bool("false")`` is ``True``
    in Python, so ``entries.get("pixel_art")`` used to turn HD mode *on* by
    way of a string that spells "off" -- every pane sends a real bool here
    (``troupe_settings.py``'s Style combo resolves to one through
    ``troupe_mode._style_choice``, never a raw value passed through), so this
    refusal has no control on any pane to name and is deliberately left
    unfielded rather than pointed at an address nothing draws.
    """
    from .pixelopts import check_pixel_options

    raw_pixel_art = entries.get("pixel_art")
    if raw_pixel_art is None:
        pixel_art = True
    elif isinstance(raw_pixel_art, bool):
        pixel_art = raw_pixel_art
    else:
        raise Invalid("pixel_art must be true or false")

    if not pixel_art:
        palette = str(entries.get("palette") or "").strip()
        if palette:
            raise Invalid(
                "pixel_art is off, so there is no palette to choose",
                field="palette",
            )
        if entries.get("dither"):
            raise Invalid(
                "pixel_art is off, so there is no dithering to turn on",
                field="dither",
            )
        outline = str(entries.get("outline") or "none")
        if outline != "none":
            raise Invalid(
                "pixel_art is off, so there is no outline mode to set",
                field="outline",
            )

    options = check_pixel_options(
        svc,
        entries,
        sizes=TROUPE_LOGICAL_SIZES,
        size_default=DEFAULT_TROUPE_LOGICAL_SIZE,
        colors=TROUPE_COLOR_CHOICES,
        colors_default=DEFAULT_TROUPE_COLORS,
        outline_default=DEFAULT_TROUPE_OUTLINE,
        size_range=TROUPE_CUSTOM_SIZE_RANGE,
    )
    if not pixel_art:
        # The 2026-09-15 audit, finding service-07: ``reduce_mode`` was left
        # off this list, so an HD request still carried it onto the row --
        # and ``_q_troupe``'s own HD branch (``if not pixel_art:``) never
        # reads it, the atlas going straight through unquantised. Dead the
        # same way ``pixelopts``' own ``allow_reduce_mode=False`` comment
        # already names for ``_charsheet``'s path that has neither.
        for key in ("colors", "palette", "dither", "outline", "reduce_mode"):
            options.pop(key, None)
        options["pixel_art"] = False
    return options


def _timed_layout(
    payload: Mapping[str, Any] | None, template: str
) -> charsheet.LayoutSpec:
    """``charsheet.resolve_layout``, timed to *template*'s clip library.

    Every movement *template*'s own clip library defines is askable, per
    ``docs/measurements/2026-09-12-troupe-open-clip-vocabulary.md`` -- passing
    ``timing`` is what opens that door, in place of the closed
    :data:`charsheet.ANIMATIONS` five.

    Raises whatever ``resolve_layout`` raises, **unconverted**: each caller
    turns a ``ValueError``/``TypeError`` into ``Invalid`` on the field its own
    pane draws, and the ``field="..."`` literal has to sit in the *calling*
    function's own source for
    ``test_every_refusal_a_pane_can_provoke_names_something_that_pane_draws``
    to see it -- a field chosen here and merely handed up would be invisible
    to that scan, which is exactly the gap this vocabulary's ``fps`` field
    means to surface: ``panes/troupe_settings.py`` draws no ``fps`` control
    yet, and the wiring test is how that stays visible instead of silently
    passing.
    """
    return charsheet.resolve_layout(payload, timing=clip_timing(template))


def check_troupe(svc: WarlockService, block: Any) -> dict[str, Any]:
    """The Troupe follow-up's options, validated at the *reference* door.

    ``_jobs_create._check_sprite_sheet``'s shape and its argument. The chain a
    Troupe reference starts is reference -> gate -> mesh -> rig -> sheet, and
    only the first link exists when this runs; everything the later links will
    refuse that is knowable now is refused now.

    ``elevation`` rides along because the Troupe form now offers a camera
    preset, and the preset is only a name for an elevation -- resolved in the
    pane and carried here as the number, so a preset table that grows a row is
    not also a migration of every queued reference. Validated exactly as
    ``charsheet.plan`` validates it, because that is the function that would
    otherwise refuse it an hour later on a row the user never submitted.
    Absent means absent: the key is not written, so a row queued before the
    control existed is byte-identical to one queued after it.
    """
    entries = dict(block or {})
    variant = str(entries.get("variant") or DEFAULT_TROUPE_VARIANT)
    if variant not in TROUPE_VARIANTS:
        raise Invalid(
            f"variant must be one of {list(TROUPE_VARIANTS)}", field="variant"
        )
    pose = str(entries.get("pose") or DEFAULT_TROUPE_POSE)
    if pose not in TROUPE_POSES:
        raise Invalid(f"pose must be one of {list(TROUPE_POSES)}", field="pose")
    options = _check_options(svc, entries)
    try:
        layout = _timed_layout(entries.get("layout"), TROUPE_TEMPLATE)
    except (TypeError, ValueError) as exc:
        message = str(exc)
        if message.startswith("fps must be"):
            raise Invalid(message, field="fps") from exc
        raise Invalid(message, field="layout") from exc
    # The VRAM the *sheet* half needs is nothing -- EEVEE and CPU -- but the
    # mesh the gate promotes to is an ordinary image job and is admitted by
    # its own door. What is checked here is the reference stage's own base,
    # because this door pins it rather than inheriting it.
    check_vram(svc, "text", "reference", {"base_model": TROUPE_BASE_MODEL})
    checked = {"variant": variant, "pose": pose, "layout": layout.as_dict(), **options}
    raw_elevation = entries.get("elevation")
    elevation: float | None = None
    if raw_elevation is not None:
        # **Refused against ``camera``, not against ``elevation``.** The number
        # is what this door validates, but nothing on the Troupe form is called
        # ``elevation`` -- the control is the Camera combo, and
        # ``troupe_mode.camera_elevation`` turns its preset into this number on
        # the way here. A refusal naming the derived value would ring a field
        # that pane does not draw, which is precisely what
        # ``test_every_refusal_a_pane_can_provoke_names_something_that_pane_draws``
        # exists to catch: it caught this one.
        try:
            elevation = float(raw_elevation)
        except (TypeError, ValueError):
            raise Invalid("that camera angle is not a number", field="camera") from None
        if not -89.0 <= elevation <= 89.0:
            raise Invalid(
                "a camera angle must be between -89 and 89 degrees above the "
                "horizon",
                field="camera",
            )
        checked["elevation"] = elevation

    # **Planned and thrown away, exactly as ``create_charsheet``/``_charsheet_spec``
    # already plan and throw away.** The 2026-09-08 audit (finding troupe-01)
    # found this door validating every option a character sheet has except the
    # one question that actually decides whether the sheet can be built: this
    # is the same reference -> gate -> mesh -> rig -> sheet chain
    # ``create_charsheet`` argues for at length, applied one link earlier, and
    # without it a layout that cannot be planned against the configured rig
    # template -- an atlas over the texture limit, or a movement the
    # template's clip library has no clip for -- was accepted here and only
    # failed in ``_q_troupe._charsheet``, uncaught, after the reference
    # render, the human gate, the trellis reconstruction and the auto-rig had
    # all completed. Planned against ``TROUPE_TEMPLATE`` rather than a
    # caller-supplied one: this door has no rig yet to read a template off of,
    # and ``TROUPE_TEMPLATE`` is the template ``_maybe_queue_rig`` pins the
    # follow-up mesh to.
    from ..pipelines import sheet as sheetlib

    try:
        records = expand_clips(TROUPE_TEMPLATE, layout)
        charsheet.plan(
            records,
            frame_size=options["logical_size"],
            elevation=sheetlib.DEFAULT_ELEVATION if elevation is None else elevation,
            lighting="flat",
            layout=layout,
        )
    except KeyError as exc:
        raise Invalid(f"the {TROUPE_TEMPLATE} clip library is missing {exc}") from exc
    except ValueError as exc:
        # **field="layout", the 2026-09-11 audit's finding troupe-01.** This
        # branch is the one a real request reaches -- an atlas over the texture
        # limit, or a movement whose frame count the resolved layout and the
        # expanded clip disagree about -- and ``panes/troupe_settings.py``
        # calls ``form_ui.note("layout")`` on exactly this address to ring the
        # layout table. Left unfielded, the refusal reached a form wired to
        # catch it and rang nothing.
        raise invalid_from(exc, "That character sheet cannot be laid out", field="layout") from exc
    return checked


def create_charsheet(
    svc: WarlockService,
    job_id: str,
    *,
    logical_size: int | None = None,
    colors: int | None = None,
    outline: str | None = None,
    reduce_mode: str | None = None,
    dither: bool = False,
    palette: str | None = None,
    elevation: float | None = None,
    lighting: str | None = None,
    name: str | None = None,
    layout: Mapping[str, Any] | None = None,
    character: Mapping[str, Any] | None = None,
    pixel_art: bool | None = None,
) -> dict[str, Any]:
    """Queue a configured character sheet for a finished, rigged mesh.

    The output is an ordinary sheet -- ``sheets/<id>.png`` plus its sidecar, in
    the *source* job's directory -- and that is the whole reason it is not a
    format of its own: "Open in Inker", the library, the exporters and the
    Aseprite writer all already read that pair. What makes it a Troupe sheet is
    the frame table it was laid out on and the ``animation`` block in the
    sidecar, both of which ``pipelines.charsheet`` owns.

    ``character`` is the family block a caller may already hold about *who*
    this is -- carried onto the row untouched and *nested*, so
    ``VECTOR_PARAMS`` (an allowlist of flat settings) cannot pick a field of it
    up and quietly turn it into a rerun vector.
    """
    from ..pipelines import sheet as sheetlib

    check_job_id(job_id)
    source = svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    if source["status"] != "done" or not (job_dir / "model.glb").exists():
        raise Invalid("job has no finished mesh to render")
    if not (job_dir / "rig.glb").exists():
        # Every Troupe cell is a posed frame, so an unrigged mesh would render
        # 256 copies of one T-pose. Named as the missing step rather than as a
        # layout failure, because rigging it is what the user has to do next.
        raise Invalid("a character sheet needs a rigged mesh")

    rig_meta = rigging.read_rig(job_dir) or {}
    template = str(rig_meta.get("template") or "")
    # **What a sheet needs is clips, not the humanoid template.** The refusal
    # used to name ``humanoid`` and turned away every family that ships its own
    # clip library -- a rig authored with a walk cycle was refused for not
    # being the one template that happened to have one first.
    # ``rigging.clip_library`` answers with an empty library rather than
    # failing, so the question is asked here: without it the mismatch lands in
    # the worker as a frame-count error, an hour and 256 EEVEE frames later.
    # An unrecorded or unknown template answers False rather than raising --
    # from the user's side it is the same fact, there are no clips to animate
    # this rig from, and a second sentence for it would be a second wording of
    # one problem. See ``has_clips``.
    if not has_clips(template):
        raise Invalid(
            "a character sheet is animated from a clip library, and nothing is "
            f"authored for the {template or 'unknown'} rig"
        )

    options = _check_options(
        svc,
        {
            "logical_size": logical_size,
            "colors": colors,
            "outline": outline,
            "reduce_mode": reduce_mode,
            "dither": dither,
            "palette": palette,
            "pixel_art": pixel_art,
        },
    )

    try:
        resolved_layout = _timed_layout(layout, template)
    except (TypeError, ValueError) as exc:
        message = str(exc)
        if message.startswith("fps must be"):
            raise Invalid(message, field="fps") from exc
        raise Invalid(message, field="layout") from exc
    try:
        # Expanded and thrown away, exactly as ``create_sheet`` plans and
        # throws away: a clip library that does not fill the frame table, or a
        # size whose atlas is over the texture limit, is refused now instead of
        # failing a job that has already rendered.
        records = expand_clips(template, resolved_layout)
        charsheet.plan(
            records,
            frame_size=options["logical_size"],
            elevation=sheetlib.DEFAULT_ELEVATION if elevation is None else elevation,
            lighting=lighting or "flat",
            layout=resolved_layout,
        )
    except KeyError as exc:
        raise Invalid(f"the {template} clip library is missing {exc}") from exc
    except ValueError as exc:
        # field="layout", the 2026-09-11 audit's finding troupe-01 -- see the
        # identical comment in ``check_troupe``, the door this planning step
        # was copied from.
        raise invalid_from(exc, "That character sheet cannot be laid out", field="layout") from exc

    sheet_name = (name or "").strip()
    if len(sheet_name) > rigging.MAX_SHEET_NAME:
        raise Invalid(
            f"sheet name must be at most {rigging.MAX_SHEET_NAME} characters", field="name"
        )

    params = {
        "source_job": job_id,
        "sheet_id": rigging.new_id(),
        # The rig's own template, read off ``rig.json`` -- the mesh is already
        # rigged, so pinning ``humanoid`` here would have the worker expand a
        # clip library the skeleton on disk does not match.
        "template": template,
        "elevation": sheetlib.DEFAULT_ELEVATION if elevation is None else elevation,
        "lighting": lighting or "flat",
        "name": sheet_name,
        "layout": resolved_layout.as_dict(),
        **options,
    }
    if character is not None:
        params["character"] = dict(character)
    # Snapshotted off the source row rather than left for the worker to read
    # live: a user who re-presses the viewport's front button between this
    # sheet and a later subset re-render would otherwise get a sheet half
    # rendered from each front. Absent at zero -- ``set_front_yaw``'s own
    # discipline -- so a mesh nobody has oriented mints a row byte-identical
    # to one from before this feature existed.
    front_yaw = source["params"].get("front_yaw")
    if front_yaw:
        params["front_yaw"] = front_yaw
    # The sheet cap, counted the way ``create_sheet`` counts it and under the
    # same job-wide hold: the artifact lands minutes after the row is minted,
    # so counting files alone lets N rapid submits all read the same count.
    with svc.convert_lock(job_id, "sheets"):
        check_sheet_cap(svc, job_id, job_dir)
        new_id = svc.store.create(
            "charsheet", source["prompt"], params, uuid.uuid4().hex[:12]
        )
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id, "sheet_id": params["sheet_id"]}


def rerender_charsheet(
    svc: WarlockService,
    job_id: str,
    *,
    sheet_id: str,
    subset: Sequence[Mapping[str, Any]],
    name: str | None = None,
) -> dict[str, Any]:
    """Re-render some of a sheet's runs, copying the rest from the sheet itself.

    **It takes no pixel options, and that is the design.** The settings are
    copied verbatim from the row that produced ``sheet_id``, because the new
    cells have to be reduced, quantised and outlined exactly as the ones they
    will sit beside were -- and any option this door accepted separately would
    be an option a user could set to something else. That turns a whole class
    of "these twelve cells look wrong" into a lookup.

    The output is a **new sheet**, not a rewrite of the old one: sheets are
    write-once under a fresh id, the old one stays openable, and Inker's merge
    is what brings the two together over a document that has hand edits on it.

    -> ``{"id", "source_job", "sheet_id", "runs"}``
    """
    check_job_id(job_id)
    source = svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    if not rigging.is_valid_id(str(sheet_id or "")):
        raise Invalid("that is not a sheet id", field="sheet_id")
    record = rigging.read_sheet(job_dir, str(sheet_id))
    if not record:
        raise NotFound("that sheet is no longer on disk", field="sheet_id")
    snapshot = record.get("troupe")
    if not isinstance(snapshot, Mapping):
        raise Invalid(
            "that is not a character sheet, so it has no runs to re-render",
            field="sheet_id",
        )

    row = _charsheet_row(svc, job_id, str(sheet_id))
    if row is None:
        # Honest, and it names what to do instead. The settings are the whole
        # point of this door; without them the new cells could not be made to
        # match the ones they are landing beside.
        raise Invalid(
            "the settings that produced that sheet are no longer on record, so it "
            "cannot be re-rendered a run at a time -- build a new sheet instead",
            field="sheet_id",
        )

    try:
        resolved_layout = charsheet.resolve_layout(snapshot)
        runs = charsheet.check_subset(subset, resolved_layout)
    except ValueError as exc:
        raise invalid_from(exc, "Those runs cannot be re-rendered", field="subset") from exc

    sheet_name = (name or "").strip()
    if len(sheet_name) > rigging.MAX_SHEET_NAME:
        raise Invalid(
            f"sheet name must be at most {rigging.MAX_SHEET_NAME} characters", field="name"
        )

    params = dict(row.get("params") or {})
    # Not inherited: they are the *previous* run's answers about its own output
    # and a fresh row must not wear them. Stripped via ``DERIVED_PARAMS``
    # itself rather than a hand-copied subset of it -- the 2026-09-07 audit
    # found this door hand-stripping only three of the four relevant keys
    # (``validation``, the sheet's structural verdict, was missing), and the
    # 2026-09-11 audit (finding troupe-02) named the hand list itself as the
    # hazard: a duplicate of an allowlist is one future ``DERIVED_PARAMS``
    # addition away from silently reintroducing a stale-verdict row. Stripped
    # *before* the fields below are set, because ``sheet_id`` is itself one of
    # ``DERIVED_PARAMS``' entries -- stripping after would delete the fresh id
    # this door is about to mint.
    for derived in DERIVED_PARAMS:
        params.pop(derived, None)
    params.update(
        {
            "source_job": job_id,
            "sheet_id": rigging.new_id(),
            "base_sheet": str(sheet_id),
            "subset": [{"animation": a, "direction": d} for a, d in runs],
            "layout": resolved_layout.as_dict(),
            "name": sheet_name or str(params.get("name") or ""),
        }
    )

    # A re-render is a new sheet and draws on the same pool -- ``create_charsheet``'s
    # arrangement verbatim, under the same job-wide hold.
    with svc.convert_lock(job_id, "sheets"):
        check_sheet_cap(svc, job_id, job_dir)
        new_id = svc.store.create(
            "charsheet", source["prompt"], params, uuid.uuid4().hex[:12]
        )
    svc.wake_worker()
    return {
        "id": new_id,
        "source_job": job_id,
        "sheet_id": params["sheet_id"],
        "runs": [{"animation": a, "direction": d} for a, d in runs],
    }


def follow_up_sheet_job(svc: WarlockService, rig_job_id: str) -> str | None:
    """The ``charsheet`` row :func:`send_to_troupe`'s rig-and-sheet path will
    mint (or already has), for a rig job that carries a ``troupe_sheet``
    reservation. None until it lands.

    A character's mesh row is minted with a *rig* id, not a sheet id --
    ``_q_jobs._maybe_queue_sheet_after_rig`` mints the sheet with a fresh,
    random id once the rig finishes, with no back-link recorded anywhere. An
    agent (or a pane) polling "is my sheet ready yet" needs a way to find that
    row without knowing its id in advance, which is what this answers by
    re-deriving the same match ``_maybe_queue_sheet_after_rig`` would have made.

    Matched, not merely the newest ``charsheet`` row naming this mesh:
    ``source_job`` -- the mesh -- equal; ``base_sheet`` absent, which excludes
    a re-render of some other sheet on the same character; minted within
    :data:`FOLLOW_UP_WINDOW_S` of this rig's own ``finished_at``; and every
    setting the reservation actually pinned (``troupe_sheet`` minus
    ``sheet_id``, which it never carries) equal on the candidate's own params.
    **The oldest match wins** -- a re-render of *this* sheet is a second
    ``charsheet`` row with the same settings and a later ``created_at``, and
    the first one minted is the follow-up, not the redo.

    **A rig that has not finished yet, or did not finish at all, has no
    follow-up.** A ``queued``/``running`` rig answers None outright -- there
    is nothing to have minted a sheet yet -- and so does a ``cancelled`` or
    ``error`` one: ``_maybe_queue_sheet_after_rig`` only ever runs from the
    worker's ``done`` branch (queue.py). Without this, a later, unrelated
    ``create_charsheet`` call on the same mesh with the same settings --
    nothing here refuses that -- would have matched an abandoned reservation
    that never actually queued anything, purely because its own
    ``created_at`` happened to be no earlier than the dead rig's.
    """
    row = svc.require_job(rig_job_id)
    if row.get("kind") != "rig":
        return None
    if row.get("status") != "done":
        return None
    finished_at = row.get("finished_at")
    if not finished_at:
        return None
    params = row.get("params") or {}
    block = params.get("troupe_sheet")
    if not isinstance(block, Mapping):
        return None
    source = str(params.get("source_job") or "")
    if not rigging.is_valid_id(source):
        return None
    wanted = {k: v for k, v in block.items() if k != "sheet_id"}
    window_start = float(finished_at) - FOLLOW_UP_WINDOW_S
    window_end = float(finished_at) + FOLLOW_UP_WINDOW_S
    matches: list[tuple[Any, str]] = []
    for candidate in svc.store.list(limit=_SCAN_LIMIT, kind="charsheet"):
        cparams = candidate.get("params") or {}
        if str(cparams.get("source_job") or "") != source:
            continue
        if "base_sheet" in cparams:
            continue
        created = candidate.get("created_at") or 0
        if not (window_start <= created <= window_end):
            continue
        if all(cparams.get(k) == v for k, v in wanted.items()):
            matches.append((created, str(candidate["id"])))
    if not matches:
        return None
    matches.sort(key=lambda pair: (pair[0], pair[1]))
    return matches[0][1]


def follow_up_failure(svc: WarlockService, mesh_job_id: str) -> dict[str, Any] | None:
    """The recorded failure of a mesh's automatic character-sheet follow-up.

    ``followups.persist`` is what a failed
    ``_q_jobs._maybe_queue_sheet_after_rig`` writes onto the *mesh* row (its
    ``source_job``), not the rig -- so a caller that only has the rig id (an
    agent polling ``character_job``) reads it off the mesh instead.
    """
    row = svc.require_job(mesh_job_id)
    params = row.get("params") or {}
    failures = params.get(followups.PARAM_KEY)
    if not isinstance(failures, Mapping):
        return None
    record = failures.get("charsheet")
    return dict(record) if isinstance(record, Mapping) else None


def _charsheet_row(
    svc: WarlockService, job_id: str, sheet_id: str
) -> dict[str, Any] | None:
    """The ``charsheet`` row that produced one sheet, or None.

    Narrowed by ``kind`` in SQL rather than walked unfiltered: the answer is one
    row and the page this searches is the same one the mode's own sheet list is
    built from.
    """
    for row in svc.store.list(limit=_SCAN_LIMIT, kind="charsheet"):
        params = row.get("params") or {}
        if str(params.get("source_job") or "") != job_id:
            continue
        if str(params.get("sheet_id") or "") == sheet_id:
            return row
    return None


def send_to_troupe(
    svc: WarlockService,
    job_id: str,
    *,
    logical_size: int | None = None,
    colors: int | None = None,
    outline: str | None = None,
    reduce_mode: str | None = None,
    dither: bool = False,
    palette: str | None = None,
    elevation: float | None = None,
    lighting: str | None = None,
    name: str | None = None,
    layout: Mapping[str, Any] | None = None,
    template: str | None = None,
    bones: list[Any] | None = None,
    character: Mapping[str, Any] | None = None,
    pixel_art: bool | None = None,
) -> dict[str, Any]:
    """Take a mesh the user already has into Troupe, rigging it first if needed.

    The third door, and the one the *library* uses. ``create_charsheet`` above
    is the direct one and refuses an unrigged mesh by design -- every Troupe
    cell is a posed frame -- which left the common case ("I have a character;
    make me a sheet") with no route at all: the user had to know to rig it
    first, with a humanoid template, from a different pane.

    Two shapes, one press:

    * **Already rigged** -- delegate to ``create_charsheet`` verbatim. One row,
      the existing path, including the humanoid refusal and the sheet cap.
    * **Not rigged** -- mint a rig row carrying a nested ``troupe_sheet``
      block, and let the worker mint the sheet on the finished rig
      (``_maybe_queue_sheet_after_rig``). That keeps the "four ordinary jobs,
      not an orchestrator" property: one press mints one row, the second is
      minted by the same mechanism that already mints rigs and sheets, and
      both cancel independently.

    **Everything knowable is refused here**, before either row exists --
    the options, the layout, the clip expansion and the plan -- because an
    unrenderable request should cost the request and not a rig plus 256 EEVEE
    frames. That is ``create_charsheet``'s own argument, applied one link
    earlier.

    **The marker is ``troupe_sheet`` and not ``troupe``.** They are different
    claims: ``troupe`` on a reference means "run the whole chain, human gate
    included", and this means "render this sheet once the rig lands". It is
    also *nested*, so ``VECTOR_PARAMS`` -- an allowlist of flat settings --
    cannot pick it up.

    **The mesh row is not stamped.** Marking it would work, and it would
    change what a reroll means: ``rerun_job``/``promote_to_model`` copy
    everything that is not derived, so the next "Remesh" would silently spend
    a rig and 256 rendered cells nobody asked for.

    The rig template defaults to ``humanoid`` rather than being taken from the
    user's Rig-stage preference, because the clip library this animates from is
    humanoid; ``joints="measured"`` for the reason ``_jobs_create`` gives where
    it sets the same flag -- the shipped template is an A-pose and mis-fits a
    T-posed mesh badly enough to skin the arms to the chest.

    Three parameters that every existing caller leaves alone, all defaulting to
    None so the row a bare call mints is byte-identical to the one it minted
    before they existed:

    * ``template`` -- the rig template to use instead of the default. A family
      that ships its own clip library is rigged on its own skeleton, and
      ``expand_clips`` has to be given the same one or the frame table is
      filled from the wrong library.
    * ``bones`` -- an exact skeleton the caller already holds, validated the
      way ``rig.adjust_joints`` validates a user correction. Passing it also
      *withholds* ``joints="measured"``: measuring is a guess from the
      reference image, and re-deriving joints a family stated exactly would
      throw away the only precise answer in the chain.
    * ``character`` -- who this is, carried into the ``troupe_sheet`` block so
      ``_maybe_queue_sheet_after_rig`` copies it onto the sheet row. Nested,
      like the block that carries it, so ``VECTOR_PARAMS`` cannot pick a field
      of it up.
    """
    check_job_id(job_id)
    source = svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    if source["status"] != "done" or not (job_dir / "model.glb").exists():
        # ``create_charsheet``'s sentence, verbatim: one refusal, one wording.
        raise Invalid("job has no finished mesh to render")

    if (job_dir / "rig.glb").exists():
        return create_charsheet(
            svc,
            job_id,
            logical_size=logical_size,
            colors=colors,
            outline=outline,
            reduce_mode=reduce_mode,
            dither=dither,
            palette=palette,
            elevation=elevation,
            lighting=lighting,
            name=name,
            layout=layout,
            character=character,
            pixel_art=pixel_art,
        )

    spec = _charsheet_spec(
        svc,
        logical_size=logical_size,
        colors=colors,
        outline=outline,
        reduce_mode=reduce_mode,
        dither=dither,
        palette=palette,
        elevation=elevation,
        lighting=lighting,
        name=name,
        layout=layout,
        template=template,
        character=character,
        pixel_art=pixel_art,
        # Read off the source row this function already holds, rather than
        # grown as a parameter of ``send_to_troupe`` itself: the row is the
        # record of what the mesh's front was set to, and a caller-supplied
        # value could disagree with it.
        front_yaw=source["params"].get("front_yaw"),
    )
    from .. import doctor

    if not doctor.blender_check().ok:
        # The Rig segment's own sentence, verbatim, so the app has one wording
        # for "this needs Blender" wherever it is met. Checked through the
        # same probe ``rig_templates`` answers the pane with, rather than a
        # second test of the same thing.
        raise Invalid("Rigging needs Blender, which is not installed.")

    rig_template = str(template or TROUPE_TEMPLATE)
    params: dict[str, Any] = {
        "source_job": job_id,
        # Defaulted rather than taken from ``config.rig_template``: the sheet is
        # animated from a clip library, so a quadruped rig chosen elsewhere in
        # the app would produce a rig that the sheet then refuses.
        "template": rig_template,
        "auto": True,
        "joints": "measured",
        "troupe_sheet": spec,
    }
    if bones is not None:
        try:
            params["bones"] = rigging.validate_joints(
                {"bones": list(bones)}, rigging.get_template(rig_template)
            )
        except ValueError as exc:
            raise invalid_from(exc, "Those joint positions cannot be used") from exc
        # **And the measurement is dropped.** ``joints="measured"`` reads the
        # joints off the reference image, which is a guess; a family that ships
        # exact joints has already answered the question, and re-measuring
        # would overwrite the precise answer with the approximate one.
        params.pop("joints", None)
    # The rig row *is* a sheet reservation -- ``_maybe_queue_sheet_after_rig``
    # mints the charsheet from it with no door in between -- so the cap is
    # taken here, under the same hold the direct door takes it.
    with svc.convert_lock(job_id, "sheets"):
        check_sheet_cap(svc, job_id, job_dir)
        # **The mesh is not rigged, and a rig for it may already be running.**
        # Before this check existed (the 2026-09-13 brief for the agent's
        # character tools), a second press -- a doubled agent call, a pane
        # retried after a slow frame -- minted a second rig row with its own
        # ``troupe_sheet`` reservation, and the later rig overwrote the earlier
        # one's ``rig.glb``. Checked here, under the same hold that mints the
        # row, so two presses cannot both pass it; and after the cap, so a full
        # pool still answers with the cap's own sentence (the first cut checked
        # it before everything else and hid the cap from
        # ``test_the_sheet_cap_counts_every_door_that_reserves_a_slot``). No
        # ``field``: the pane that draws this door draws no ``job_id`` control
        # for a refusal to ring (``tests/test_field_error_wiring.py``).
        from .rig import rig_in_flight

        if rig_in_flight(svc, job_id) is not None:
            raise Conflict("a rig for this mesh is already running")
        new_id = svc.store.create("rig", source["prompt"], params, uuid.uuid4().hex[:12])
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id, "rigged": False}


def _charsheet_spec(
    svc: WarlockService,
    *,
    logical_size: int | None,
    colors: int | None,
    outline: str | None,
    reduce_mode: str | None,
    dither: bool,
    palette: str | None,
    elevation: float | None,
    lighting: str | None,
    name: str | None,
    layout: Mapping[str, Any] | None,
    template: str | None = None,
    character: Mapping[str, Any] | None = None,
    front_yaw: float | None = None,
    pixel_art: bool | None = None,
) -> dict[str, Any]:
    """Validate a sheet request and freeze it as the params the worker will use.

    Everything ``create_charsheet`` refuses that does not depend on the rig,
    checked here and then *snapshotted*: the row the worker mints later must
    describe the settings that were on screen when the button was pressed, not
    whatever the pane holds by the time the rig finishes.

    ``template`` is the rig the sheet will be animated on -- it has to be the
    one ``send_to_troupe`` puts on the rig row, or the clips expanded here come
    from one library and the skeleton from another. ``character`` is carried
    through untouched: ``_maybe_queue_sheet_after_rig`` copies this block onto
    the sheet row wholesale, which is the whole reason it is nested.

    ``front_yaw`` is the mesh's own front, read by ``send_to_troupe`` off the
    source row and passed through here rather than re-read later: this whole
    dict is splatted wholesale by ``_q_jobs._maybe_queue_sheet_after_rig``, so
    it has to already be the *value*, never ``None`` -- a present ``None``
    would arrive there as a key that exists and is not a number.
    """
    from ..pipelines import sheet as sheetlib

    options = _check_options(
        svc,
        {
            "logical_size": logical_size,
            "colors": colors,
            "outline": outline,
            "reduce_mode": reduce_mode,
            "dither": dither,
            "palette": palette,
            "pixel_art": pixel_art,
        },
    )
    sheet_template = str(template or TROUPE_TEMPLATE)
    # **Refused here, with a field, rather than as a missing key.** The
    # expansion below already fails for a clipless skeleton, but it fails as
    # ``KeyError`` turned into "the <x> clip library is missing 'walk'" -- a
    # message about a dictionary, with no ``field`` for the Skeleton control
    # that now asks the question. ``get_template`` answers an unknown key,
    # and the sentence is ``create_charsheet``'s own so one fact has one
    # wording.
    try:
        rigging.get_template(sheet_template)
    except ValueError as exc:
        raise Invalid(str(exc), field="template") from exc
    if not has_clips(sheet_template):
        raise Invalid(
            "a character sheet is animated from a clip library, and nothing is "
            f"authored for the {sheet_template} rig",
            field="template",
        )
    try:
        resolved_layout = _timed_layout(layout, sheet_template)
    except (TypeError, ValueError) as exc:
        message = str(exc)
        if message.startswith("fps must be"):
            raise Invalid(message, field="fps") from exc
        if message.endswith("is not a clip of this skeleton"):
            # field="template": the same address ``_charsheet_spec``'s sibling
            # ``except KeyError`` branch below already uses for the identical
            # fact reached a different way -- ``panes/troupe_send.py`` draws a
            # Skeleton control, not a layout table.
            raise Invalid(message, field="template") from exc
        raise Invalid(message, field="layout") from exc
    try:
        records = expand_clips(sheet_template, resolved_layout)
        charsheet.plan(
            records,
            frame_size=options["logical_size"],
            elevation=sheetlib.DEFAULT_ELEVATION if elevation is None else elevation,
            lighting=lighting or "flat",
            layout=resolved_layout,
        )
    except KeyError as exc:
        # **field="template", the 2026-09-11 audit's finding service-06.**
        # ``has_clips`` two calls up only proves the library is non-empty --
        # ``service.clips.save``'s ``_check_renders`` holds only
        # ``TROUPE_TEMPLATE`` to Troupe's frame table, so any other template's
        # library is accepted with any subset of clips, by design. A template
        # with some clips but not the one this layout names (an ordinary shape
        # for a user-edited library, not the "internal build consistency" case
        # the *shipped* libraries would give) reaches this branch with
        # ``has_clips`` having already answered True, and the Skeleton control
        # (``troupe_send._skeleton``) that offered the template is what a
        # refusal about it should ring.
        raise Invalid(
            f"the {sheet_template} clip library is missing {exc}", field="template"
        ) from exc
    except ValueError as exc:
        # field="layout", the 2026-09-11 audit's finding troupe-01 -- see the
        # identical comment in ``check_troupe``.
        raise invalid_from(exc, "That character sheet cannot be laid out", field="layout") from exc

    sheet_name = (name or "").strip()
    if len(sheet_name) > rigging.MAX_SHEET_NAME:
        raise Invalid(
            f"sheet name must be at most {rigging.MAX_SHEET_NAME} characters", field="name"
        )
    spec: dict[str, Any] = {
        "template": sheet_template,
        "elevation": sheetlib.DEFAULT_ELEVATION if elevation is None else elevation,
        "lighting": lighting or "flat",
        "name": sheet_name,
        "layout": resolved_layout.as_dict(),
        **options,
    }
    if character is not None:
        spec["character"] = dict(character)
    if front_yaw:
        spec["front_yaw"] = front_yaw
    return spec

# ``get_charsheet`` was deleted on 2026-08-22. It delegated to
# ``sheets.get_sheet`` so a Troupe pane would not have to know that a Troupe
# sheet *is* a sheet -- and all three of its would-be callers
# (``packwright_mode``, ``inker_mode``, ``panes.sheet_panel``) called
# ``sheets.get_sheet`` directly anyway, which is the thing it existed to spare
# them. A wrapper nobody reaches for is a second answer to drift from.
