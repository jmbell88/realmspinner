"""Everywhere one asset can go, as a list -- not the one door :mod:`.asset_open`
answers "where does this row open" for, but the panel of buttons under "Take it
somewhere".

**There were two of these lists, and they disagreed.** The library's overflow
menu (``modes/library/ui/panes/library.py``'s ``_overflow``) offered the full run -- Inker, Plotter,
Packwright, Clay and the rest, both directions -- because it grew one item at
a time as each bridge shipped. The inspector's "Take it somewhere" section
(``studio/panes/inspector.py``'s ``_edit_actions``) offered a hand-picked four, because it was
wired once, before the later bridges and the reopen doors existed, and nobody came back to
widen it. A mesh selected in the Library could reach a destination from the overflow
menu and not from the pane six inches to its right showing the same mesh. This
module is the one reading both surfaces draw from, so the two cannot drift
again -- ``tests/test_asset_exits.py`` is the test that says so by name.

**Every gate is answered from the cached row alone.** No ``stat``, no service
call, no filesystem -- both call sites ask this every frame, which is the same
argument ``inker_open.can_edit_job``'s docstring makes about the toolbar. A row
that has fallen out of date by the next frame is no worse than any other piece
of cached UI state in this app.

**A follow-up row resolves one hop through the cache -- kind-scoped, not
``source_job``-scoped.** A rig, a sheet, a retexture and a remesh all carry
``params["source_job"]`` and write their artifacts into *that* job's
directory, never their own (:func:`_is_mesh`'s docstring names the trap), so
selecting one of them used to offer nothing at all -- every mesh-shaped
builder gated on the row in hand and that row has no ``model.glb``.
:func:`_mesh_for` resolves ``source_job`` through ``ctx.cache.get`` -- still a
cached-row read, not a filesystem call -- for exactly the kinds
``asset_open.FOLLOWUP_STAGES`` already names as a mesh's own product, never on
``source_job`` alone: a character sheet (``retired.RETIRED_KINDS`` owns that
exclusion now; ``asset_open.FOLLOWUP_STAGES`` deliberately omits the retired
kinds) carries that same field but is not one of them, and hopping for it too
resolved it straight back to the mesh it was rendered from, offering Clay on a
row that has no mesh of its own.
The mesh-shaped builders (Clay and the Mason add door) read every gate off the
*resolved* mesh, closing their door over it rather than over the selected row.
A source that has fallen off the loaded page, or a kind the hop does not cover, answers None,
the same floor ``asset_open.open_asset`` already takes for the identical
reason.

**Every door is the mode's own, called verbatim.** This module does not
reimplement Clay's 200k-triangle confirm; it
calls the function that already asks them, so an exit taken from here behaves
exactly as the one that already shipped from wherever it shipped first.

**A near miss is not the same as absent.** ``modes/create/ui/rail.py``'s ``stage_rail``'s own
docstring already makes the argument this borrows: "'Rig' missing entirely is
a feature the user concludes does not exist; 'Rig -- Blender is not installed'
is an answer." So a destination this asset is one step away from -- a mesh with
no rig, a reference still generating, a reference whose ``input.png`` has not
landed -- is drawn dimmed with a reason rather than left off the list. A
destination this *kind* of asset can never reach stays off the list entirely:
a full matrix of dimmed buttons is noise, not information.

A mode's own UI submodule is imported lazily inside each function that needs
it, exactly as :mod:`.asset_open` keeps them: this module has to be
importable by a test with no GL context, and ``panes.inspector`` importing
this module while this module imports ``panes.inspector`` back (for
``offers_inker``) only works if neither import runs at module scope. That
rule does not reach ``icons``/``modes``/``verbs`` below -- plain data with no
GL and no import of this module in return, and needed at module scope to
build :data:`_MODE_ICONS` once rather than on every call to :func:`icon_for`.
The 2026-09-26 audit, finding create-brief-05, found the near-miss image
stages import breaking that promise at module scope for no such reason (nothing
here builds a module-level table from it) -- moved back into the two
functions that read it, ``_plotter_add`` and ``_packwright_add``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from ..service.files import EDITABLE_STAGES as _REFERENCE_STAGES
from . import icons, modes, verbs

# _REFERENCE_STAGES: the stages a job carries before it has a mesh -- a
# picture still being painted, generated or reconstructed from. Aliased
# directly from ``service.files.EDITABLE_STAGES`` rather than restated as a
# second tuple (the 2026-09-26 audit, finding create-brief-04: the hand copy
# here was unpinned, so the two could silently drift) -- that constant
# answers "can Inker open this" (a question about pixels) and this module
# answers "is this row shaped like a reference at all", and the two are the
# same question for this narrower purpose. Deliberately narrower than
# :data:`IMAGE_STAGES` (imported where it is used, below): Inker opens a
# reference or a single tile to paint over, never a whole generated tile
# sheet.


class Exit(NamedTuple):
    """One destination this asset could go to, ready or not.

    ``mode`` is a :data:`modes.KEYS` entry -- it supplies the icon (through
    :func:`icon_for`) and, together with :mod:`.verbs`, the label. ``label`` is
    always a :mod:`.verbs` product, never hand-written, so the four gestures
    this app hands an asset off with stay worded one way
    (``tests/test_ux_shared_vocabulary.py``). ``reason`` is empty when the
    exit is fully available; set, it says why the button is dimmed, and
    ``open`` is populated but never called for it -- the near-miss rule above.
    """

    mode: str
    label: str
    hint: str
    tooltip: str
    reason: str
    open: Callable[[Any, Any], None]


#: The rail's own glyph for a destination, unless a call site chose a
#: different one before this module existed and changing it now would just be
#: relabelling a button nobody asked to have relabelled. Inker's brush and
#: Clay's box read as "edit this" at button size faster than the rail's
#: pen-tool and ruler do. One rule -- the rail's icon, except these two,
#: kept for continuity with what shipped first.
_ICON_OVERRIDES: dict[str, str] = {
    "inker": icons.BRUSH,
    "clay": icons.BOX,
}

_MODE_ICONS: dict[str, str] = {key: icon for key, _label, icon, _purpose in modes.MODES}


def icon_for(mode: str) -> str:
    """The glyph a button for this destination draws -- see the override table
    above. Derived rather than a second hand-written table, so a mode cannot
    carry one icon on the rail and another one here by accident."""
    return _ICON_OVERRIDES.get(mode, _MODE_ICONS.get(mode, ""))


def _status_reason(job: Any) -> str:
    """The one line every "not done yet" near miss shows, worded off the
    row's own status rather than a generic "unavailable" that says nothing
    a reader could not already see from the status pill above it.

    Built from ``service.validation.STATUS_SENTENCES``/``not_done_message``
    rather than a second hand-written mapping of the same four values --
    ``modes/create/ui/stages.py``'s ``available`` names, in its own docstring, the rule this used to
    violate: "the wording is the service's own, verbatim, so the tooltip on
    the disabled segment and the toast from the refusal it is predicting are
    one sentence and not two paraphrases." Before the 2026-09-11 audit
    (finding create-06) this function spelled "Still queued." beside
    ``STATUS_SENTENCES``' "is still waiting in the queue" -- two spellings of
    one fact, one edit away from drifting the moment either changed alone.
    """
    from ..service.validation import STATUS_SENTENCES, not_done_message

    status = str(job.get("status") or "")
    if status not in STATUS_SENTENCES:
        # A status this table does not name -- including "", which a job
        # never actually carries. Honest rather than guessing at a sentence
        # for a word nobody defined one for.
        return "Not finished yet."
    return not_done_message("This", status)


def _files(job: Any) -> list[Any]:
    return job.get("files") or []


def _params(job: Any) -> dict[str, Any]:
    params = job.get("params")
    return params if isinstance(params, dict) else {}


def _is_mesh(job: Any) -> bool:
    """Whether ``job`` is a genuine reconstructed mesh, not a follow-up row
    wearing the ``model`` stage because that is ``db.Store.create``'s column
    default.

    ``asset_open``'s own docstring names the trap: a rig, a sheet, a pixel
    sheet, a sprite draft, a retexture, a remesh and a character sheet are all
    minted with ``params["source_job"]`` and write their artifacts into *that*
    job's directory, never into their own -- so their own row never gets a
    ``model.glb``, and without this check a finished character sheet showed a
    dimmed "Open in Clay" for the mesh it does not have. ``source_job`` is
    the exact field ``service._jobs_lifecycle.dependent_jobs`` filters on for
    the same fact, read from the other side.
    """
    # The 2026-10-10 audit, finding create-14: a ``lora_train`` row is minted
    # with no stage, so it takes the column default ``model``, and it carries
    # no ``source_job`` -- both halves of this test passed, and a finished LoRA
    # was offered Clay, Mason and two doors of the since-removed Poser mode,
    # dimmed with "This mesh has no model yet.", a reason that is false of a
    # LoRA. ``OPENS_ELSEWHERE`` is the one place that already names the kinds
    # whose row is not a mesh.
    from .asset_open import OPENS_ELSEWHERE

    if job.get("kind") in OPENS_ELSEWHERE:
        return False
    return job.get("stage") == "model" and not _params(job).get("source_job")


def _mesh_for(ctx: Any, job: Any) -> Any:
    """The mesh this row is about: itself, or the source it wrote into.

    ``job`` when :func:`_is_mesh` already agrees. Otherwise the hop is gated
    on membership of ``asset_open.FOLLOWUP_STAGES`` -- **kind-scoped, not
    ``source_job``-scoped** -- because a bare "carries a ``source_job``" test
    also matches ``charsheet``, a retired kind (``retired.RETIRED_KINDS``, absent
    from ``FOLLOWUP_STAGES`` by design) whose row is not a mesh follow-up at all.
    Gating on ``source_job`` alone hopped for one too and resolved it
    straight back to the mesh it was rendered from, so a charsheet row offered
    Clay it has no business offering -- exactly the outcome ``_is_mesh``'s
    docstring already names as the trap this module exists to avoid,
    reintroduced one layer down. Importing the mapping (lazily, as every other
    cross-module read here is) means the charsheet exclusion is inherited from
    the one place that already states it, rather than restated and risking a
    second copy that drifts.

    Resolved through ``ctx.cache.get`` -- a cached-row read, not a ``stat`` or
    a service call, so this survives the same filesystem ban every other gate
    in this module does -- guarded the way ``stages.parent``
    (``modes/create/ui/stages.py``) guards the identical lookup: a ``ctx``
    with no ``cache`` (the palette, a profile sheet) answers "no mesh" rather
    than raising. None when the row's kind is not in ``FOLLOWUP_STAGES``,
    there is no ``source_job``, the cache has never loaded that row, or the
    resolved row is not itself a mesh (a follow-up of a follow-up, or a source
    that has since been trashed) -- the honest floor ``asset_open.open_asset``
    already takes for the same reason: a row this module cannot see is a row it offers nothing for.
    """
    if _is_mesh(job):
        return job
    from .asset_open import FOLLOWUP_STAGES

    if job.get("kind") not in FOLLOWUP_STAGES:
        return None
    source = str(_params(job).get("source_job") or "")
    if not source:
        return None
    getter = getattr(getattr(ctx, "cache", None), "get", None)
    mesh = getter(source) if callable(getter) else None
    if mesh is None or not _is_mesh(mesh):
        return None
    # The 2026-09-23 audit, finding create-09: trashing a mesh does not touch
    # its surviving follow-up rows (`_jobs_lifecycle.dependent_jobs` only
    # scans active jobs) and a trashed row stays in the cache, so without
    # this floor a rig/sheet/retexture/remesh row kept resolving straight to
    # a mesh `exits_for`'s own docstring says must offer nothing -- the same
    # resurrection that check already refuses for the *selected* row.
    if mesh.get("deleted_at"):
        return None
    return mesh


# --- Inker --------------------------------------------------------------


def _inker(ctx: Any, job: Any) -> Exit | None:
    if job.get("stage") not in _REFERENCE_STAGES:
        return None
    from .modes.inker import mode as inker_mode
    from .panes import inspector

    hint = "Paint over the reference; saving updates this asset."

    def door(ctx: Any, job: Any) -> None:
        inker_mode.open_job_reference(ctx, job)

    if inspector.offers_inker(ctx, job):
        return Exit("inker", verbs.open_in("inker"), hint, "", "", door)

    # The viewport toolbar owns a *ready* reference's Inker button at the
    # Reference stage, and ``offers_inker``'s own docstring is built on exactly
    # one of the two ever being true -- a lit exit here would be the second
    # button its complement rule exists to prevent. Reaching this line means
    # ``inspector.offers_inker`` is False, so either the stage is not on screen
    # or the toolbar's own gate (``inker_mode.can_edit_job``: done, with an
    # ``input.png``) is shut and it draws nothing. The 2026-10-04 audit, finding
    # create-43: this used to return None for the whole Reference stage, so a
    # queued, running or failed reference in Create had no Inker affordance
    # anywhere, not even the dimmed, reasoned one the Library shows. The ready
    # case stays None below, which is where the toolbar's button lives.
    if job.get("status") != "done":
        reason = _status_reason(job)
    elif "input.png" not in _files(job):
        reason = "This reference has no image yet."
    else:
        return None
    return Exit("inker", verbs.open_in("inker"), hint, "", reason, door)


# --- Clay -----------------------------------------------------------------


def _clay(ctx: Any, job: Any) -> Exit | None:
    mesh = _mesh_for(ctx, job)
    if mesh is None:
        return None
    from .modes.clay import mode as clay_mode
    from .panes import inspector

    rigged = "rig.glb" in _files(mesh)
    hint = (
        "Opens the authored document when there is one, else the rig (its bones, weights and "
        "animations), else the mesh."
        if rigged
        else "Opens the authored document when there is one, else the mesh."
    )

    # Closed over ``mesh``, not the selected row: both call sites invoke
    # ``exit_.open(ctx, job)`` with the row the user actually picked, which
    # for a follow-up row is the rig/sheet/etc, not the mesh it belongs to --
    # a door that read its ``job`` argument would open Clay on a row with no
    # ``model.glb`` of its own.
    def door(ctx: Any, job: Any, _mesh: Any = mesh) -> None:
        clay_mode.edit_asset_in_clay(ctx, _mesh)

    if inspector.can_edit_in_clay(mesh):
        return Exit("clay", verbs.open_in("clay"), hint, "", "", door)

    if mesh.get("status") != "done":
        reason = _status_reason(mesh)
    elif "model.glb" not in _files(mesh):
        reason = "This mesh has no model yet."
    else:
        return None
    return Exit("clay", verbs.open_in("clay"), hint, "", reason, door)


# --- Plotter ----------------------------------------------------------------


def _plotter_reopen(ctx: Any, job: Any) -> Exit | None:
    if _params(job).get("authored") != "plotter":
        return None
    from .modes.plotter import mode as plotter_mode

    def door(ctx: Any, job: Any) -> None:
        plotter_mode.edit_asset_in_plotter(ctx, job)

    return Exit("plotter", verbs.open_in("plotter"), "Reopens the authored map.", "", "", door)


def _plotter_add(ctx: Any, job: Any) -> Exit | None:
    from .modes.create.ui.stages import IMAGE_STAGES as near_miss_image_stages
    from .modes.plotter import mode as plotter_mode

    label = verbs.add_to("plotter", "as a tileset")
    hint = "Use the generated grid as a map tileset."

    def door(ctx: Any, job: Any) -> None:
        plotter_mode.use_as_tileset(ctx, job)

    # The 2026-09-14 audit, finding create-02: this checked only
    # ``"input.png" in files``, and a finished *mesh* carries its
    # reference's ``input.png`` in its own files too (the promotion path in
    # ``_jobs_create.py`` copies it across) -- so every finished mesh was
    # offered a live "Add to Plotter as a tileset" door that acted on the
    # mesh's reference photo. The near-miss branch below already gates on
    # ``near_miss_image_stages``; the ready branch needs the same gate.
    #
    # The 2026-10-10 audit, finding create-15: it also never tested the row's
    # *status*, so a running, errored or cancelled tile row that had already
    # published its ``input.png`` was offered a live door while ``_inker``,
    # on the same row, was dimmed by status first. Unfinished falls through to
    # the near-miss branch below, which dims with the status sentence.
    if (
        job.get("stage") in near_miss_image_stages
        and job.get("status") == "done"
        and "input.png" in _files(job)
    ):
        return Exit("plotter", label, hint, "", "", door)

    # The near miss is scoped to a reference-shaped row (A2's own wording:
    # "a reference whose input.png has not landed"), not to every kind that
    # could theoretically carry the file -- a mesh missing it is not "close",
    # it is a promotion that did not copy the reference, which is a bug
    # elsewhere and not a state this button should be narrating.
    #
    # The 2026-09-13 audit, finding create-08: this checked
    # ``_REFERENCE_STAGES`` (``"reference"``, ``"tile"``), which left out
    # ``"tilesheet"`` -- so a running tile *sheet* job showed no button at
    # all instead of a greyed one, the one near-miss shape this whole
    # module exists to draw. ``stages.IMAGE_STAGES`` (``modes/create/ui/stages.py``) is the list
    # that already carries all three "this row is a picture, not a mesh
    # yet" stages.
    if job.get("stage") not in near_miss_image_stages:
        return None
    if job.get("status") != "done":
        reason = _status_reason(job)
    else:
        reason = "This reference has no image yet."
    return Exit("plotter", label, hint, "", reason, door)


# --- Packwright --------------------------------------------------------------


def _packwright_reopen(ctx: Any, job: Any) -> Exit | None:
    if _params(job).get("authored") != "packwright":
        return None
    from .modes.packwright import mode as packwright_mode

    def door(ctx: Any, job: Any) -> None:
        packwright_mode.edit_asset_in_packwright(ctx, job)

    return Exit(
        "packwright", verbs.open_in("packwright"), "Reopens the authored atlas.", "", "", door
    )


def _packwright_add(ctx: Any, job: Any) -> Exit | None:
    from .modes.create.ui.stages import IMAGE_STAGES as near_miss_image_stages
    from .modes.packwright import mode as packwright_mode

    label = verbs.add_to("packwright", "as an atlas source")
    hint = "Use the generated grid as an atlas source."

    def door(ctx: Any, job: Any) -> None:
        packwright_mode.add_job_source(ctx, job)

    # The 2026-09-14 audit, finding create-02: same fix as ``_plotter_add``
    # above -- gate the ready branch on the job being image-shaped too, not
    # on carrying "input.png" alone, since a finished mesh carries that file
    # in its own directory as well.
    #
    # The 2026-10-10 audit, finding create-15: status gate too -- see
    # ``_plotter_add``; an unfinished row falls through to the dimmed branch.
    if (
        job.get("stage") in near_miss_image_stages
        and job.get("status") == "done"
        and "input.png" in _files(job)
    ):
        return Exit("packwright", label, hint, "", "", door)

    # The 2026-09-13 audit, finding create-08: same fix as ``_plotter_add``
    # above -- ``stages.IMAGE_STAGES`` instead of ``_REFERENCE_STAGES``
    # so a running tile sheet job dims rather than vanishes.
    if job.get("stage") not in near_miss_image_stages:
        return None
    if job.get("status") != "done":
        reason = _status_reason(job)
    else:
        reason = "This reference has no image yet."
    return Exit("packwright", label, hint, "", reason, door)


# --- Mason --------------------------------------------------------------------


def _mason_reopen(ctx: Any, job: Any) -> Exit | None:
    """Back into the scene a library row was exported from.

    ``_plotter_reopen``'s shape exactly, including the reason it reads the
    marker rather than the disk: a reopen has **no fallback**
    (``mason_mode.edit_asset_in_mason``'s docstring says why a merged mesh is
    not a lesser scene), so the row alone has to answer whether the ``.rscn``
    is there -- and every gate in this module is answered from the cached row,
    with no ``stat`` on the frame thread.
    """
    if _params(job).get("authored") != "mason":
        return None
    from .modes.mason import mode as mason_mode

    def door(ctx: Any, job: Any) -> None:
        mason_mode.edit_asset_in_mason(ctx, job)

    return Exit("mason", verbs.open_in("mason"), "Reopens the authored scene.", "", "", door)


def _mason_add(ctx: Any, job: Any) -> Exit | None:
    """Drop a library mesh into the open scene.

    Gated on the *resolved* mesh like ``_clay``, for its reason: a rig or a retexture
    row writes into its source's directory and has no ``model.glb`` of its own, so a
    door keyed on the selected row would place a reference to a job with no mesh
    behind it.

    It offers to add rather than to open, so the label is ``add_to`` and not
    ``open_in`` -- ``verbs``' own distinction: the asset joins a document there
    as a *source*, and the scene it joins is whichever one is already open.
    A scene is minted if none is, which is what makes this a one-press exit
    from the library rather than a two-step errand.
    """
    mesh = _mesh_for(ctx, job)
    if mesh is None:
        return None
    from .modes.mason import mode as mason_mode

    label = verbs.add_to("mason", "as a scene item")
    hint = "Place this mesh in the open scene."

    # Closed over ``mesh``, ``_clay``'s reason: both call sites invoke the door
    # with the row the user picked, which for a follow-up is not the mesh.
    def door(ctx: Any, job: Any, _mesh: Any = mesh) -> None:
        mason_mode.add_asset_to_scene(ctx, _mesh)

    if mesh.get("status") == "done" and "model.glb" in _files(mesh):
        return Exit("mason", label, hint, "", "", door)

    if mesh.get("status") != "done":
        reason = _status_reason(mesh)
    elif "model.glb" not in _files(mesh):
        reason = "This mesh has no model yet."
    else:  # pragma: no cover - the ready branch above already claimed it
        return None
    return Exit("mason", label, hint, "", reason, door)


# --- Sirens -------------------------------------------------------------


def _sirens(ctx: Any, job: Any) -> Exit | None:
    """A finished take, landed in the tracker as a sample instrument.

    Not a near-miss destination: an unfinished or errored take is not "one
    step" from Sirens in the sense the other three bullets mean it -- there is
    no single missing file or status to name, only "wait" or "try again",
    which the take's own row in Muse already says.
    """
    if job.get("kind") != "music" or job.get("status") != "done":
        return None
    if "track.wav" not in _files(job):
        return None
    from .modes.muse import mode as muse_mode

    def door(ctx: Any, job: Any) -> None:
        muse_mode.open_in_sirens(ctx, job.get("id"))

    return Exit(
        "sirens",
        verbs.open_in("sirens"),
        "Imports this track into the tracker as a sample instrument.",
        "",
        "",
        door,
    )


_BUILDERS: tuple[Callable[[Any, Any], Exit | None], ...] = (
    _inker,
    _clay,
    _mason_reopen,
    _mason_add,
    _plotter_reopen,
    _plotter_add,
    _packwright_reopen,
    _packwright_add,
    _sirens,
)


def exits_for(ctx: Any, job: Any) -> list[Exit]:
    """Everywhere ``job`` can go, in the rail's own pipeline order.

    A trashed row gets none: every action a builder above could offer would
    either fail outright or quietly resurrect the row into the workshop
    without saying so, which is the same reasoning the Library's ``_overflow``
    already applies (a trashed asset's menu offers exactly Restore and Delete
    permanently, and nothing else).
    """
    if not isinstance(job, dict) or job.get("deleted_at"):
        return []
    exits: list[Exit] = []
    for build in _BUILDERS:
        exit_ = build(ctx, job)
        if exit_ is not None:
            exits.append(exit_)
    return exits
