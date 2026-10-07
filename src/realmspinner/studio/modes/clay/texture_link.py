"""Clay's texture round trip through Inker: the pull model Plotter's "Polish in
Inker" uses, with the document's revision as the trigger instead of a button.

A Clay tab records an :class:`InkerLink` -- *this* palette entry (by identity),
*that* Inker document (by uid), and the history head last landed -- and every
frame Clay is drawn :func:`pull_all` compares each link's head with the Inker
document's. Nothing moved is the common case and costs two comparisons; a moved
head is flattened on the frame thread (as Plotter's ``tileset_from_inker`` does,
because the composite fills and evicts the document's own cache) and lands as
exactly one ``set_material`` -- one undo step per return from Inker.

**The revision is ``history.head``, not ``doc.rev``.** ``rev`` moves every frame
a live preview is up (a brush hovering, a drag in flight) and would push an undo
step per frame; ``head`` moves when a step is *committed*.

**The link is the texture's identity, not the entry's or its index.** Removing a
slot below shifts every later index (the clay-16 rule), so the picture the link
was made for -- ``materials[index].base_color is link.material.base_color`` -- is
the only test that cannot land a pull on a neighbour. Editing the entry's colour,
cutout or double-sided flag replaces the material but keeps that same picture,
so the link survives it; an undo of a pull, or clearing the texture, puts a
different picture there and lets go of the link: the next "Edit texture in
Inker" opens a fresh document rather than guessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ....kernels.geom3d.gltf import Material
from ...state import set_mode

log = logging.getLogger(__name__)

#: The longest side a pulled texture may have. Clay's own textures are 32-128 px
#: and an Inker document can be anything; a 16k canvas flattened every revision
#: and uploaded to the GPU is a stall, so the pull refuses by name instead.
MAX_TEXTURE_SIDE = 1024


#: ``InkerLink.pulled_rev`` before anything has landed: no history head is ever
#: negative, so the first pull always counts the document as moved.
UNLANDED = -1


@dataclass
class InkerLink:
    """One palette entry <-> one Inker document. See the module docstring.

    ``doc`` is the Inker :class:`~realmspinner.kernels.pixel.Document` itself,
    held so a pull can still land its last committed head after the Inker *tab*
    is gone (the 2026-10-07 audit's clay-06). ``None`` for a link built without
    one, which then cannot outlive its tab.
    """

    index: int
    material: Material
    inker_uid: str
    pulled_rev: int
    doc: Any = None


# --- reading the Inker side -------------------------------------------------


def open_inker_docs(ctx: Any) -> list[Any]:
    """The open Inker tabs (each has ``uid``, ``title``, ``doc``); ``[]`` when
    Inker has not been used this session."""
    inker = getattr(getattr(ctx, "state", None), "inker", None)
    return list(getattr(inker, "docs", None) or [])


def _find(ctx: Any, uid: str) -> Any:
    for tab in open_inker_docs(ctx):
        if tab.uid == uid:
            return tab
    return None


def _links(tab: Any) -> list[Any]:
    return getattr(tab, "inker_links", None) or []


def _live(tab: Any, link: InkerLink) -> bool:
    """Whether the slot at *link.index* still holds the picture *link* was made for."""
    materials = tab.doc.materials
    return (
        0 <= link.index < len(materials)
        and materials[link.index].base_color is link.material.base_color
    )


def unlink(tab: Any, index: int) -> None:
    """Drop whatever link points at slot *index* (a texture was cleared)."""
    links = _links(tab)
    links[:] = [link for link in links if link.index != index]


def forget(tab: Any) -> None:
    """Drop every link (the tab is closing)."""
    _links(tab).clear()


# --- the push ---------------------------------------------------------------


def edit_in_inker(ctx: Any, tab: Any, index: int) -> None:
    """Open slot *index*'s texture in Inker, palette-locked to PICO-8, and link it."""
    materials = tab.doc.materials
    if not 0 <= index < len(materials):
        ctx.toast(f"There is no palette entry {index}.", "error")
        return
    material = materials[index]
    if material.base_color is None:
        ctx.toast("Add a texture first.", "error")
        return
    width, height, _data = material.base_color
    if max(width, height) > MAX_TEXTURE_SIDE:
        # The 2026-10-07 audit's clay-20: a texture assigned from a file may be
        # far past the pull's ceiling, and opening it anyway let the first
        # committed stroke be refused and the link dropped -- the user painted a
        # picture that could never come back. Refused here, at the door, by name.
        ctx.toast(
            f"That texture is {width} x {height}; Inker can only take one back at most "
            f"{MAX_TEXTURE_SIDE} px on a side.",
            "error",
        )
        return

    for link in _links(tab):
        if link.index == index and _live(tab, link):
            existing = _find(ctx, link.inker_uid)
            if existing is not None:
                # One document per slot: a second tab over the same texture would
                # be two pulls racing to land on one palette entry.
                set_mode(ctx.state, "inker")
                ctx.state.inker.activate(existing.uid)
                return

    from ....kernels.pixel.palettes import PICO8
    from ..inker import opening as inker_opening

    data = material.base_color[2]
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    name = material.name or f"slot {index}"

    def on_open(inker_tab: Any) -> None:
        # Frame thread, after adoption. The open ran on a task thread, so the
        # slot may have been edited or removed since the click; a changed slot
        # drops the link (the Inker tab simply opens unlinked).
        mats = tab.doc.materials
        if not 0 <= index < len(mats) or mats[index].base_color is not material.base_color:
            return
        _record(
            tab,
            InkerLink(index, material, inker_tab.uid, UNLANDED, doc=inker_tab.doc),
        )
        # The 2026-10-07 audit's clay-50: ``open_pixels`` snapped the picture to
        # PICO-8 before the tab was adopted, so the head recorded here already
        # held the snap and nothing pulled it -- the first real stroke then
        # recoloured every texel at once. ``UNLANDED`` makes the next pull count
        # the opened document as moved, and the snap returns on its own step.
        pull(ctx, tab)

    inker_opening.open_pixels(
        ctx, pixels, title=f"{tab.title} - {name} (texture)", palette=PICO8, on_open=on_open
    )


def _record(tab: Any, link: InkerLink) -> None:
    unlink(tab, link.index)
    tab.inker_links.append(link)


# --- the pull ---------------------------------------------------------------


def _flatten(ctx: Any, inker_doc: Any) -> tuple[int, int, bytes] | None:
    """The Inker document as a ``(w, h, rgba_bytes)`` texture, or ``None`` after
    a toast when its size is not one a texture may have."""
    pixels = np.asarray(inker_doc.flatten(matte=False), dtype=np.uint8)
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    if width < 1 or height < 1 or max(width, height) > MAX_TEXTURE_SIDE:
        ctx.toast(
            f"That drawing is {width} x {height}; a texture can be at most "
            f"{MAX_TEXTURE_SIDE} px on a side.",
            "error",
        )
        return None
    return width, height, np.ascontiguousarray(pixels).tobytes()


def _land(tab: Any, index: int, image: tuple[int, int, bytes]) -> Material:
    """One undo step: *image* as slot *index*'s texture. -> the entry now there.

    Byte-identical pixels push nothing: a pull that changed no texel (a palette
    swap and back, a selection) is not an edit of the model.
    """
    material = tab.doc.materials[index]
    if material.base_color == image and material.nearest:
        return material
    changed = replace(material, base_color=image, nearest=True)
    tab.doc.set_material(index, changed)
    return changed


def pull(ctx: Any, tab: Any) -> None:
    """Land every linked Inker document that has moved since it was last landed."""
    for link in list(_links(tab)):
        inker_tab = _find(ctx, link.inker_uid)
        if not _live(tab, link):
            _drop(tab, link)
            continue
        # The 2026-10-07 audit's clay-06: a closed Inker tab used to drop its
        # link here without landing what was last painted -- a pull runs only
        # while Clay is drawn, so paint, close Inker and open Clay lost the
        # picture. The link holds the document itself, which outlives its tab,
        # so the last committed head still lands and the link goes with it.
        closed = inker_tab is None
        inker_doc = link.doc if closed else inker_tab.doc
        if inker_doc is None:
            _drop(tab, link)
            continue
        head = inker_doc.history.head
        if head == link.pulled_rev:
            if closed:
                _drop(tab, link)
            continue
        try:
            image = _flatten(ctx, inker_doc)
            if image is None:
                _drop(tab, link)
                continue
            link.material = _land(tab, link.index, image)
            link.pulled_rev = head
            if closed:
                _drop(tab, link)
                ctx.toast("The Inker tab was closed; its last changes were taken back first.")
        except Exception:
            # A pull runs every frame Clay is drawn: one that raises would raise
            # every frame. Let go of the link and say so once.
            log.exception("could not pull the Inker texture onto slot %s", link.index)
            ctx.toast("Could not take the texture back from Inker.", "error")
            _drop(tab, link)


def _drop(tab: Any, link: Any) -> None:
    links = _links(tab)
    if link in links:
        links.remove(link)


def pull_all(ctx: Any) -> None:
    """Every Clay tab's pull -- what the viewport calls each frame it draws.

    Every tab with links, not only the drawn one: a texture painted in Inker
    while a different Clay tab is showing lands on its own tab, so switching
    back finds it done. Costs one attribute read per tab when nothing is linked.
    """
    state = getattr(getattr(ctx, "state", None), "clay", None)
    if state is None:
        return
    for tab in state.docs:
        if _links(tab):
            pull(ctx, tab)


# --- the manual fallback ----------------------------------------------------


def take_back(ctx: Any, tab: Any, index: int, inker_tab: Any) -> None:
    """Flatten *inker_tab*'s document onto slot *index* now, and keep following it.

    The fallback for a drawing that was not opened from this slot (or whose link
    let go): same landing and same refusals as the pull, and the same link is
    recorded afterwards so later edits to that Inker tab keep flowing in.
    """
    if not 0 <= index < len(tab.doc.materials):
        ctx.toast(f"There is no palette entry {index}.", "error")
        return
    head = inker_tab.doc.history.head
    image = _flatten(ctx, inker_tab.doc)
    if image is None:
        return
    before = tab.doc.materials[index]
    material = _land(tab, index, image)
    _record(tab, InkerLink(index, material, inker_tab.uid, head, doc=inker_tab.doc))
    ctx.toast(
        "Texture unchanged."
        if material is before
        else f"Texture taken back from {inker_tab.title}."
    )
