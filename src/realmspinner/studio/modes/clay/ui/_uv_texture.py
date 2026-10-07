"""The texture the UV pane draws behind the islands: which material's, and its GL copy.

Two halves, kept apart so the first is testable without a GL context.

:func:`material_for_pane` is pure: the palette entry whose base-colour texture
the pane should show. :func:`pane_texture` is the cache around the GL object,
and it follows the rules Plotter's tileset textures learned (see
``plotter/ui/panes/textures.py``): a texture is *registered* with the imgui
backend as well as created (``widgets.texture_ref``, at the draw), and
``docmodes.forget_texture``d before it is released -- there is one GL context,
and a texture the backend never hears of draws as the font atlas.

What differs from Plotter is the source. A tileset is frozen at construction;
a material's ``base_color`` is replaced wholesale (``dataclasses.replace``) by
every Inker pull, every "Add texture" and every undo, never mutated. So the
tuple's *identity* is the staleness stamp -- held as a strong reference beside
the texture, because ``id()`` alone is recycled once the old tuple is freed and
a replacement landing on the same address would be drawn as the old picture.
One texture per tab, whatever the palette: a new source releases the old one,
which is what keeps a session of Inker pulls from leaking one texture per pull,
and a material with no texture releases it too.
"""

from __future__ import annotations

from typing import Any

from .... import docmodes

PREFIX = "clay_uv_tex:"


def _slot(uid: str, name: str) -> str:
    # The trailing colon on the sweep prefix is what keeps ``bd1``'s release
    # from taking ``bd10``'s texture with it.
    return f"{PREFIX}{uid}:{name}"


def material_for_pane(doc: Any, obj: Any) -> Any:
    """The material whose texture the UV pane shows for *obj*, or ``None``.

    The slot of the first selected face when faces are selected -- what a
    painter is looking at when they box a few islands -- and the object's
    default slot otherwise. ``None`` for an index outside the palette (a stale
    selection against a mesh that has since changed) as well as for a material
    with no base-colour texture, so the caller has one "draw nothing extra"
    answer.
    """
    index = int(obj.material)
    selection = doc.element_sel.get(obj.uid)
    faces = getattr(selection, "faces", None)
    per_face = obj.mesh.material
    if faces is not None and len(faces) and 0 <= int(faces[0]) < len(per_face):
        index = int(per_face[int(faces[0])])
    if not 0 <= index < len(doc.materials):
        return None
    entry = doc.materials[index]
    return entry if entry.base_color is not None else None


def pane_texture(ctx: Any, uid: str, material: Any) -> Any:
    """The GL texture for *material*'s base colour, or ``None``.

    ``None`` when there is nothing to draw (no material, no texture, a
    malformed image) and when there is no GL context -- the headless suite and
    every state-only test run under one -- and in both cases whatever this tab
    held before is released, so "at most one texture per tab" holds on the
    way out as well as on the way in.
    """
    image = material.base_color if material is not None else None
    if image is not None:
        width, height, data = image
        if width <= 0 or height <= 0 or len(data) != int(width) * int(height) * 4:
            image = None
    viewer = getattr(ctx, "viewer", None)
    if image is None or viewer is None:
        release_doc(ctx, uid)
        return None
    key, stamp_key = _slot(uid, "img"), _slot(uid, "src")
    nearest = bool(getattr(material, "nearest", False))
    held = ctx.state.preview.get(key)
    stamp = ctx.state.preview.get(stamp_key)
    # Identity of the image tuple, value of the filter: a material that flips
    # ``nearest`` keeps its image but needs the other sampler.
    if held is not None and stamp is not None and stamp[0] is image and stamp[1] == nearest:
        return held
    if held is not None:
        docmodes.forget_texture(held)
    texture = viewer.ctx.texture((int(width), int(height)), 4, bytes(data))
    sampler = viewer.ctx.NEAREST if nearest else viewer.ctx.LINEAR
    texture.filter = (sampler, sampler)
    ctx.state.preview[key] = texture
    ctx.state.preview[stamp_key] = (image, nearest)
    return texture


def release_doc(ctx: Any, uid: str) -> None:
    """Drop the texture belonging to one tab -- on its close, and on a change to nothing."""
    if getattr(ctx, "state", None) is not None:
        docmodes.release_prefix(ctx, _slot(uid, ""))


def release_all(ctx: Any) -> None:
    if getattr(ctx, "state", None) is not None:
        docmodes.release_prefix(ctx, PREFIX)
