"""What a follow-up row made, drawn where the row is selected.

A rig, a sheet, a sprite draft, a retexture and a remesh write into the *source*
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

from ...kernels.rig import store
from .. import asset_open, widgets
from ..tokens import sp

#: How large the largest side of a preview is drawn, in design pixels. The
#: inspector is 340 wide; this leaves its gutters.
PREVIEW_SIZE = 240


class Product(NamedTuple):
    """One picture a row made."""

    #: What to call it under the picture -- and half of its texture key.
    label: str
    path: Path
    #: Pixel art is drawn at a whole multiple with the filter off; a render or
    #: a thumbnail is scaled to fit and filtered.
    nearest: bool


def product_images(job: Any, job_dir_of: Any) -> list[Product]:
    """Every picture ``job`` made, oldest-decision-first. Pure: paths, no I/O.

    ``job_dir_of`` is ``ctx.job_dir``. The directory asked is the one
    :func:`asset_open.route` names -- the source's, for a follow-up -- which is
    the same fact ``Open folder`` and ``asset_open`` are built on.

    A row that is not finished, or a kind that draws no picture of its own (a
    character sheet is Poser's to draw; a LoRA run has no image), answers
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
    if kind in ("sheet", "pixel_sheet"):
        sheet_id = str(params.get("sheet_id") or "")
        if not store.is_valid_id(sheet_id):
            return []
        if kind == "pixel_sheet":
            return [Product("pixel sheet", store.sheet_pixel_png_path(job_dir, sheet_id), True)]
        return [Product("sheet", store.sheet_png_path(job_dir, sheet_id), False)]
    if kind == "rig":
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
    products = product_images(job, ctx.job_dir)
    if not products or ctx.textures is None:
        return
    widgets.section("What this made")
    shown = 0
    limit = int(sp(PREVIEW_SIZE))
    for product in products:
        imgui.push_id(product.label)
        try:
            texture = ctx.textures.get(
                f"{job['id']}:followup:{product.label}", product.path, nearest=product.nearest
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
