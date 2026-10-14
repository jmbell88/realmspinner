"""What a follow-up row made, drawn where the row is selected.

A rig, a sprite draft, a retexture and a remesh write into the *source*
job's directory and never their own (``asset_open``'s docstring says why), so
selecting one in the Library used to show a block of settings and nothing else:
the pictures it had made were one directory over, reachable only by opening the
source in Create and finding the right panel. This is the picture, in the
inspector, without leaving the Library.

Split the way :mod:`.sprite_panel` splits its drafts. :func:`product_images` is
path arithmetic with no I/O, no imgui and no ``ctx``, so which file each kind
draws is a table test; :func:`draw` owns the textures. It does not stat
anything itself -- ``ctx.textures.get`` answers ``None`` for a file that is not
there and remembers that, so a row whose files were removed by hand costs one
stat and not one per frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from imgui_bundle import imgui

from ...kernels.rig import poses, store
from .. import asset_open, widgets
from ..textures import MAX_SIDE, PIXEL_MAX_SIDE
from ..tokens import sp

#: How large the largest side of a preview is drawn, in design pixels. The
#: inspector is 340 wide; this leaves its gutters.
PREVIEW_SIZE = 240

#: Kinds whose picture is one shared file in the source's directory, grouped by
#: the file. A later row of any kind in a group overwrites what an earlier row
#: of the group drew from it: ``thumb.png`` is the *mesh's* current thumbnail,
#: not a record of what one rework made.
_SHARED_PICTURE_GROUPS: dict[str, frozenset[str]] = {
    "rig": frozenset({"rig"}),
    "retexture": frozenset({"retexture", "remesh"}),
    "remesh": frozenset({"retexture", "remesh"}),
}


class Product(NamedTuple):
    """One picture a row made."""

    #: What to call it under the picture -- and half of its texture key.
    label: str
    path: Path
    #: Pixel art is drawn at a whole multiple with the filter off; a render or
    #: a thumbnail is scaled to fit and filtered.
    nearest: bool


def superseded(job: Any, siblings: Any) -> bool:
    """Whether a later finished row of the same picture group shares ``job``'s source.

    The 2026-10-14 audit, finding create-panes-09: ``rig_qa.png`` and
    ``thumb.png`` live in the *source's* directory and are overwritten by each
    later rig, retexture or remesh, so an older row drew a later row's picture
    under its own heading. ``siblings`` is ``ctx.cache.jobs`` -- the recent
    window every pane already reads, never a live query on the frame thread.
    A row outside that window is simply not known to be superseded.
    """
    if not isinstance(job, dict) or not siblings:
        return False
    group = _SHARED_PICTURE_GROUPS.get(str(job.get("kind") or ""))
    params = job.get("params") if isinstance(job.get("params"), dict) else {}
    source = str(params.get("source_job") or "")
    if group is None or not source:
        return False
    mine = (float(job.get("created_at") or 0.0), str(job.get("id") or ""))
    for other in siblings:
        if not isinstance(other, dict) or other.get("id") == job.get("id"):
            continue
        if other.get("status") != "done" or str(other.get("kind") or "") not in group:
            continue
        theirs = other.get("params") if isinstance(other.get("params"), dict) else {}
        if str(theirs.get("source_job") or "") != source:
            continue
        if (float(other.get("created_at") or 0.0), str(other.get("id") or "")) > mine:
            return True
    return False


def product_images(job: Any, job_dir_of: Any, siblings: Any = None) -> list[Product]:
    """Every picture ``job`` made, oldest-decision-first. Pure: paths, no I/O.

    ``job_dir_of`` is ``ctx.job_dir``. The directory asked is the one
    :func:`asset_open.route` names -- the source's, for a follow-up -- which is
    the same fact ``Open folder`` and ``asset_open`` are built on.

    A row that is not finished, or a kind that draws no picture of its own (a
    retired sheet row keeps its files but shows nothing; a LoRA run has no
    image), answers
    ``[]``: the honest floor, and what makes the section draw nothing rather
    than a heading over nothing.
    """
    if not isinstance(job, dict) or job.get("status") != "done":
        return []
    kind = str(job.get("kind") or "")
    if kind not in asset_open.OPENS_ELSEWHERE:
        return []
    holder = asset_open.route(job).job_id
    if not holder:
        return []
    job_dir = Path(job_dir_of(holder))
    params = job.get("params") if isinstance(job.get("params"), dict) else {}

    if kind == "sprite_synthesis":
        draft_id = str(params.get("draft_id") or "")
        if not store.is_valid_id(draft_id):
            return []
        # ``candidates`` is a count in the row's params and a list in the
        # sidecar; the row is all this function reads. A big sheet is drawn as
        # one candidate (``store.list_sprite_drafts``), so it is the count and
        # not the constant that says how many letters exist.
        count = params.get("candidates")
        letters = store.SPRITE_CANDIDATES[: count if isinstance(count, int) and count > 0 else 2]
        return [
            Product(letter, store.sprite_draft_png_path(job_dir, draft_id, letter), True)
            for letter in letters
        ]
    if superseded(job, siblings):
        return []
    if kind == "rig":
        # ``rig_qa.png`` exists only for a template with a deformation battery
        # (``_q_rig`` renders nothing when ``deform_battery`` is empty). Clay's
        # Weight with Blender makes a ``blank`` rig and every non-humanoid
        # template has none, so offering the picture drew "Its pictures are not
        # on disk any more" for a file that was never made (the 2026-10-14
        # audit, finding service-kinds-02). A row that names no template keeps
        # the old answer.
        template = params.get("template")
        if isinstance(template, str) and template:
            try:
                if not poses.deform_battery(template):
                    return []
            except ValueError:
                return []
        return [Product("rig check", job_dir / "rig_qa.png", False)]
    if kind in ("retexture", "remesh"):
        return [Product("mesh", job_dir / "thumb.png", False)]
    return []


def scaled(size: tuple[int, int], limit: int, *, nearest: bool) -> tuple[int, int]:
    """The size a ``size`` picture is drawn at inside a ``limit`` square.

    A whole multiple for pixel art, ``pixel_scale``'s rule and for its reason (a
    fractional factor reads as banding), fit-to-limit for anything filtered.
    Never zero, and never larger than a filtered picture's own size -- upscaling
    a 256 px thumbnail to fill the column is a blur nobody asked for.
    """
    width, height = max(size[0], 1), max(size[1], 1)
    if nearest:
        factor = max(1, limit // max(width, height))
        return width * factor, height * factor
    factor = min(1.0, limit / max(width, height))
    return max(1, round(width * factor)), max(1, round(height * factor))


def draw(ctx: Any, job: Any) -> None:
    # Before ``ctx`` is touched: this runs for every asset the Library's Details
    # tab shows, and almost none of them are follow-ups.
    if not asset_open.opens_elsewhere(job) or job.get("status") != "done":
        return
    siblings = getattr(getattr(ctx, "cache", None), "jobs", None)
    products = product_images(job, ctx.job_dir, siblings)
    if not products and superseded(job, siblings):
        widgets.section("What this made")
        widgets.muted("A later rework of this asset replaced this row's picture.")
        return
    if not products or ctx.textures is None:
        return
    widgets.section("What this made")
    shown = 0
    limit = int(sp(PREVIEW_SIZE))
    for product in products:
        imgui.push_id(product.label)
        try:
            texture = ctx.textures.get(
                f"{job['id']}:followup:{product.label}",
                product.path,
                nearest=product.nearest,
                # Pixel art keeps its own size: the 256 px default shrinks and
                # smooths a 48 or 64 px draft atlas (the 2026-10-14 audit,
                # finding create-panes-04).
                max_side=PIXEL_MAX_SIDE if product.nearest else MAX_SIDE,
            )
            if texture is None:
                continue
            shown += 1
            widgets.muted(product.label)
            imgui.image(
                widgets.texture_ref(texture),
                scaled(texture.size, limit, nearest=product.nearest),
            )
        finally:
            imgui.pop_id()
    if not shown:
        # Reachable: a half-cleaned directory, a source trashed and purged, a
        # draft deleted from the source's own panel. Said, so an empty section
        # is not mistaken for one still rendering.
        widgets.muted("Its pictures are not on disk any more.")
