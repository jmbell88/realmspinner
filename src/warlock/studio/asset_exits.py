"""Everywhere one asset can go, as a list -- not the one door :mod:`.asset_open`
answers "where does this row open" for, but the panel of buttons under "Take it
somewhere".

**There were two of these lists, and they disagreed.** The library's overflow
menu (``panes.library._overflow``) offered the full run -- Inker, Plotter,
Packwright, Clay, Poser, Troupe, both directions -- because it grew one item at
a time as each bridge shipped. The inspector's "Take it somewhere" section
(``panes.inspector._edit_actions``) offered a hand-picked four, because it was
wired once, before Poser and the reopen doors existed, and nobody came back to
widen it. A mesh selected in the Library could reach Poser from the overflow
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
``source_job`` alone: a character sheet carries that same field but is not
one of them (it opens in Troupe, not in Create), and hopping for it too
resolved it straight back to the mesh it was rendered from -- offering Clay
and Poser on a row that has neither, plus a second, duplicate "Open in
Troupe" beside :func:`_troupe_out`'s own. The three mesh-shaped builders
(Clay, Poser, Troupe-in) read every gate off the *resolved* mesh, closing
their door over it rather than over the selected row. A source that has
fallen off the loaded page, or a kind the hop does not cover, answers None,
the same floor ``asset_open.open_asset`` already takes for the identical
reason.

**Every door is the mode's own, called verbatim.** This module does not
reimplement Clay's 200k-triangle confirm or Troupe's skeleton question; it
calls the function that already asks them, so an exit taken from here behaves
exactly as the one that already shipped from wherever it shipped first.

**A near miss is not the same as absent.** ``create_rail.stage_rail``'s own
docstring already makes the argument this borrows: "'Rig' missing entirely is
a feature the user concludes does not exist; 'Rig -- Blender is not installed'
is an answer." So a destination this asset is one step away from -- a mesh with
no rig, a reference still generating, a reference whose ``input.png`` has not
landed -- is drawn dimmed with a reason rather than left off the list. A
destination this *kind* of asset can never reach stays off the list entirely:
a full matrix of dimmed buttons is noise, not information.

Imports stay lazy inside each function, exactly as :mod:`.asset_open` keeps
them: this module has to be importable by a test with no GL context, and
``panes.inspector`` importing this module while this module imports
``panes.inspector`` back (for ``offers_inker``) only works if neither import
runs at module scope.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from . import icons, modes, verbs
from .modes.create.ui.stages import IMAGE_STAGES as _NEAR_MISS_IMAGE_STAGES

#: The stages a job carries before it has a mesh -- a picture still being
#: painted, generated or reconstructed from. The same tuple
#: ``service.files.EDITABLE_STAGES`` names, restated here because that
#: constant answers "can Inker open this" (a question about pixels) and this
#: one answers "is this row shaped like a reference at all" -- the two
#: currently agree, and if they ever stop, that is a decision for whoever
#: changes one of them, not a second module quietly drifting off the first.
#: Deliberately narrower than :data:`_NEAR_MISS_IMAGE_STAGES` above: Inker
#: opens a reference or a single tile to paint over, never a whole
#: generated tile sheet.
_REFERENCE_STAGES = ("reference", "tile")


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
#: pen-tool and ruler do; Troupe's standing figure is the icon the "Send to
#: Troupe" button wore before the rail's own glyph for Troupe moved to FILM
#: (``modes.py``'s comment on that move: the rail needed a picture of what
#: Troupe *makes*, frames of a character, and this button is still about the
#: character going in). One rule -- the rail's icon, except these three,
#: kept for continuity with what shipped first.
_ICON_OVERRIDES: dict[str, str] = {
    "inker": icons.BRUSH,
    "clay": icons.BOX,
    "troupe": icons.PERSON_STANDING,
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
    ``create_stages.available``'s own docstring names the rule this used to
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
    dimmed "Send to Troupe" for the mesh it does not have. ``source_job`` is
    the exact field ``service._jobs_lifecycle.dependent_jobs`` filters on for
    the same fact, read from the other side.
    """
    return job.get("stage") == "model" and not _params(job).get("source_job")


def _mesh_for(ctx: Any, job: Any) -> Any:
    """The mesh this row is about: itself, or the source it wrote into.

    ``job`` when :func:`_is_mesh` already agrees. Otherwise the hop is gated
    on membership of ``asset_open.FOLLOWUP_STAGES`` -- **kind-scoped, not
    ``source_job``-scoped** -- because a bare "carries a ``source_job``" test
    also matches ``charsheet``, whose own row is not a mesh follow-up at all:
    it opens in Troupe, never in Create (``FOLLOWUP_STAGES``'s own comment
    says so). Gating on ``source_job`` alone hopped for a charsheet too and
    resolved it straight back to the mesh it was rendered from, so a
    charsheet row offered Clay and Poser it has no business offering, *and* a
    second, duplicate "Open in Troupe" beside ``_troupe_out``'s own --
    exactly the outcome ``_is_mesh``'s docstring already names as the trap
    this module exists to avoid, reintroduced one layer down. Importing the
    mapping (lazily, as every other cross-module read here is) means the
    charsheet exclusion is inherited from the one place that already states
    it, rather than restated and risking a second copy that drifts.

    Resolved through ``ctx.cache.get`` -- a cached-row read, not a ``stat`` or
    a service call, so this survives the same filesystem ban every other gate
    in this module does -- guarded the way ``create_stages.parent`` guards the
    identical lookup: a ``ctx`` with no ``cache`` (the palette, a profile
    sheet) answers "no mesh" rather than raising. None when the row's kind is
    not in ``FOLLOWUP_STAGES``, there is no ``source_job``, the cache has
    never loaded that row, or the resolved row is not itself a mesh (a
    follow-up of a follow-up, or a source that has since been trashed) -- the
    honest floor ``asset_open.open_asset`` already takes for the same reason:
    a row this module cannot see is a row it offers nothing for.
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
    return mesh


# --- Inker --------------------------------------------------------------


def _inker(ctx: Any, job: Any) -> Exit | None:
    if job.get("stage") not in _REFERENCE_STAGES:
        return None
    from .modes.create.ui import stages as create_stages
    from .modes.inker import mode as inker_mode
    from .panes import inspector

    hint = "Paint over the reference; saving updates this asset."

    def door(ctx: Any, job: Any) -> None:
        inker_mode.open_job_reference(ctx, job)

    if inspector.offers_inker(ctx, job):
        return Exit("inker", verbs.open_in("inker"), hint, "", "", door)

    # The viewport toolbar owns this asset's Inker affordance whenever the
    # Reference stage is on screen -- ready or not -- and ``offers_inker``'s
    # own docstring is built on exactly one of the two ever being true. A
    # near miss drawn here on top of that would be the second button its
    # complement rule exists to prevent, just dimmed instead of lit.
    if create_stages.at(ctx.state, "reference"):
        return None

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

    hint = "Opens the authored document when there is one, else the mesh."

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


# --- Poser ------------------------------------------------------------------


def _poser(ctx: Any, job: Any) -> Exit | None:
    mesh = _mesh_for(ctx, job)
    if mesh is None:
        return None
    from .panes import pose_panel

    hint = "Pose this mesh's own rig, or author clips for its skeleton."

    # Closed over ``mesh``, ``_clay``'s reason: the selected row a rig job's
    # own door is opened with (``exit_.open(ctx, job)``) is the rig row, and
    # ``pose_panel.open_in_poser`` needs the *mesh* id -- that is what
    # ``poser_mode.open_asset`` binds the session to.
    def door(ctx: Any, job: Any, _mesh: Any = mesh) -> None:
        pose_panel.open_in_poser(ctx, _mesh)

    if "rig.glb" in _files(mesh):
        return Exit("poser", verbs.open_in("poser"), hint, "", "", door)

    if mesh.get("status") != "done":
        reason = _status_reason(mesh)
    else:
        reason = "Rig this mesh first -- Poser edits poses on a rig."
    return Exit("poser", verbs.open_in("poser"), hint, "", reason, door)


# --- Troupe -----------------------------------------------------------------


def _troupe_in(ctx: Any, job: Any) -> Exit | None:
    """Take a mesh into Troupe -- rigging it first if it is not rigged.

    The hint and tooltip wording is carried over verbatim from the two call
    sites this module replaces (``inspector._edit_actions`` and
    ``library._send_to_troupe_item``), which is why it names the rig step even
    on the label's own line: for an unrigged mesh this button is minutes of
    CPU behind a button that is not called "Rig", and a user who is not told
    reads the quiet as a hang.
    """
    mesh = _mesh_for(ctx, job)
    if mesh is None:
        return None
    from .modes.troupe import mode as troupe_mode
    from .modes.troupe.ui.panes import send as troupe_send

    rigged = "rig.glb" in _files(mesh)
    hint = (
        "Render a character sheet from this mesh, on the skeleton it is "
        "already rigged on. Asks for the sprite size first."
        if rigged
        else "Rigs the mesh on a skeleton you choose, then renders a sheet."
    )
    tooltip = (
        "Render a character sheet from this mesh, rigging it first if it is "
        "not rigged yet. Asks for the sprite size -- and, for an unrigged "
        "mesh, the skeleton -- before anything is queued."
    )
    label = f"{verbs.send_to('troupe')}..."

    # Closed over ``mesh``, ``_clay``'s and ``_poser``'s reason: a follow-up
    # row's own send would ask Troupe to rig or sheet a row with no
    # ``model.glb``, since the mesh it actually needs is the one this door
    # already resolved to.
    def door(ctx: Any, job: Any, _mesh: Any = mesh) -> None:
        troupe_send.ask(ctx, _mesh)

    if troupe_mode.can_send_to_troupe(ctx, mesh):
        return Exit("troupe", label, hint, tooltip, "", door)

    if mesh.get("status") != "done":
        reason = _status_reason(mesh)
    elif "model.glb" not in _files(mesh):
        reason = "This mesh has no model yet."
    else:
        return None
    return Exit("troupe", label, hint, tooltip, reason, door)


def _troupe_out(ctx: Any, job: Any) -> Exit | None:
    """Back into Troupe from a finished character sheet.

    Not a near-miss destination (A2's three bullets do not name it): a
    charsheet row with no ``source_job`` is an old row or a hand-edited one,
    and there is nothing a dimmed button could offer a reason about that the
    row itself would still be true a moment later.
    """
    if job.get("kind") != "charsheet" or job.get("status") != "done":
        return None
    params = _params(job)
    source = str(params.get("source_job") or "")
    if not source:
        return None
    sheet_id = str(params.get("sheet_id") or "")
    from .modes.troupe import mode as troupe_mode

    def door(ctx: Any, job: Any, _source: str = source, _sheet: str = sheet_id) -> None:
        troupe_mode.open_sheet(ctx, _source, _sheet)

    return Exit(
        "troupe",
        verbs.open_in("troupe"),
        "Reopens this character sheet where it plays.",
        "",
        "",
        door,
    )


# --- Plotter ----------------------------------------------------------------


def _plotter_reopen(ctx: Any, job: Any) -> Exit | None:
    if _params(job).get("authored") != "plotter":
        return None
    from . import plotter_mode

    def door(ctx: Any, job: Any) -> None:
        plotter_mode.edit_asset_in_plotter(ctx, job)

    return Exit("plotter", verbs.open_in("plotter"), "Reopens the authored map.", "", "", door)


def _plotter_add(ctx: Any, job: Any) -> Exit | None:
    from . import plotter_mode

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
    # ``_NEAR_MISS_IMAGE_STAGES``; the ready branch needs the same gate.
    if job.get("stage") in _NEAR_MISS_IMAGE_STAGES and "input.png" in _files(job):
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
    # module exists to draw. ``create_stages.IMAGE_STAGES`` is the list
    # that already carries all three "this row is a picture, not a mesh
    # yet" stages.
    if job.get("stage") not in _NEAR_MISS_IMAGE_STAGES:
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
    from .modes.packwright import mode as packwright_mode

    label = verbs.add_to("packwright", "as an atlas source")
    hint = "Use the generated grid as an atlas source."

    def door(ctx: Any, job: Any) -> None:
        packwright_mode.add_job_source(ctx, job)

    # The 2026-09-14 audit, finding create-02: same fix as ``_plotter_add``
    # above -- gate the ready branch on the job being image-shaped too, not
    # on carrying "input.png" alone, since a finished mesh carries that file
    # in its own directory as well.
    if job.get("stage") in _NEAR_MISS_IMAGE_STAGES and "input.png" in _files(job):
        return Exit("packwright", label, hint, "", "", door)

    # The 2026-09-13 audit, finding create-08: same fix as ``_plotter_add``
    # above -- ``create_stages.IMAGE_STAGES`` instead of ``_REFERENCE_STAGES``
    # so a running tile sheet job dims rather than vanishes.
    if job.get("stage") not in _NEAR_MISS_IMAGE_STAGES:
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
    not a lesser scene), so the row alone has to answer whether the ``.wscn``
    is there -- and every gate in this module is answered from the cached row,
    with no ``stat`` on the frame thread.
    """
    if _params(job).get("authored") != "mason":
        return None
    from .modes.mason import mode as mason_mode

    def door(ctx: Any, job: Any) -> None:
        mason_mode.edit_asset_in_mason(ctx, job)

    return Exit(
        "mason", verbs.open_in("mason"), "Reopens the authored scene.", "", "", door
    )


def _mason_add(ctx: Any, job: Any) -> Exit | None:
    """Drop a library mesh into the open scene.

    Gated on the *resolved* mesh like ``_clay``, ``_poser`` and ``_troupe_in``,
    for their reason: a rig or a retexture row writes into its source's
    directory and has no ``model.glb`` of its own, so a door keyed on the
    selected row would place a reference to a job with no mesh behind it.

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
    _poser,
    _troupe_in,
    _troupe_out,
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
    without saying so, which is the same reasoning ``library._overflow``
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
