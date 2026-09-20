"""Flourish in the editor: predicates, the pending recipe, and the render task.

The headless half of the feature's UI, ``inker_sheet``'s shape: everything
the ops registry, the inspector and the tests ask is answered here with no
imgui in reach, so ``inker_ops`` can grey a row with the same sentence the
panel shows and a test can drive the whole loop with a fake ``ctx``.

**Renders run in a task; the document is written on the frame thread.** A
slider reports on every frame of a drag, and a bake is a hundred milliseconds
of numpy, so the inspector writes its edits to ``state.flourish_pending`` and
this module submits *one* render once the value has rested for
``DEBOUNCE_SECONDS``. The result comes back through ``inker_mode.on_task_done``
on the ``inker-flourish`` prefix and lands as one undo step
(``Document.apply_flourish``). A result for a tab that has closed, a group
that has been dissolved, or a recipe the user has since moved past is dropped
rather than shown -- ``viewer_embed``'s pending-marker rule, applied to cels.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

FLOURISH_POPUP = "inker-flourish-insert"
SNIPPET_POPUP = "inker-flourish-snippet"
TEXTURE_POPUP = "inker-flourish-texture"
TEXTURE_KEY = "inker-flourish-texture"
TEXTURE_LAND_KEY = "inker-flourish-texture-land"
PROMPT_KEY = "inker-flourish-prompt"
RESTYLE_POPUP = "inker-flourish-restyle"
RESTYLE_KEY = "inker-flourish-restyle"
RESTYLE_LAND_KEY = "inker-flourish-restyle-land"
RESTYLE_PENDING = "A restyle of this effect is already running."
#: The words around the user's own, for a keyframe the pixel model repaints.
RESTYLE_PROMPT_TEMPLATE = "{subject}, 2D game VFX frame, centered, transparent background"
#: How long the text model may take before the words fall back to the mapper.
PROMPT_TIMEOUT_S = 120.0
RENDER_KEY = "inker-flourish"
INSERT_KEY = "inker-flourish-insert"

#: How long a slider value has to rest before the render for it is submitted.
DEBOUNCE_SECONDS = 0.25


def clock() -> float:
    """The one clock the debounce runs on. Monotonic rather than imgui's,
    because ``land`` runs from the task-completion path where no imgui
    context need exist, and a due time compared across two clocks is never
    due or always due."""
    return time.monotonic()

NO_EFFECT = "The active layer is not part of a Flourish effect."
BUSY = "The document is busy -- a save, an export or playback is still running."
RENDERING = "A render of this effect is still running."
NO_CONFLICTS = "No cells of this effect are flagged."
NO_SELECTION = "Select the pixels to use as a texture first."
NO_TEXTURE_SLOT = "This layer has no texture parameter to take one."
TEXTURE_PENDING = "A texture is already being generated."
#: The words around the user's own, for a texture the pixel model paints.
TEXTURE_PROMPT_TEMPLATE = (
    "single {subject}, centered, on a plain black background, 2D game VFX texture, "
    "no objects, no text, high contrast"
)
#: How often the door asks the store about a pending texture.
TEXTURE_POLL_S = 0.5
#: A generated texture is brought down to this on its long side.
TEXTURE_MAX_PX = 256


def render_key(tab: Any, group_uid: int) -> str:
    return f"{RENDER_KEY}:{tab.uid}:{int(group_uid)}"


def insert_key(tab: Any) -> str:
    return f"{INSERT_KEY}:{tab.uid}"


# -- predicates (the ops registry reads these) ------------------------------------


def active_group(state: Any, tab: Any) -> int | None:
    doc = getattr(tab, "doc", None)
    if doc is None or not hasattr(doc, "flourish_group_of_active"):
        return None
    return doc.flourish_group_of_active()


def has_effect(state: Any, tab: Any) -> bool:
    return active_group(state, tab) is not None


def can_insert(state: Any, tab: Any) -> bool:
    return tab is not None and not getattr(tab, "busy", False)


def insert_reason(state: Any, tab: Any) -> str:
    if tab is None:
        return "Nothing is open."
    return BUSY if getattr(tab, "busy", False) else ""


def facing_afford(recipe: Any, directions: int) -> str:
    """Why *directions* would be refused for *recipe*, or "" if it would not.

    Checked against the preset's own recipe -- before ``inker_mode.flourish_insert``'s
    canvas-fit scale, which only ever shrinks ``width``/``height``. Shrinking
    geometry only ever lowers ``bake_cost`` (``inker/flourish/recipe.py``), so
    a combination flagged here is refused on every canvas the preset could be
    inserted into, never fewer -- the safe direction to be wrong in for a
    warning shown before Insert is pressed.

    The 2026-09-08 audit (finding inker-05): the Facings combo offered
    One/Four/Eight unconditionally for every one of the 29 shipped presets,
    but ``check_bake_cost`` (already gated at submission, the 2026-09-07
    audit's inker-10) silently refuses 4 presets at Four directions and 11 at
    Eight, ``fireball`` -- the manual's own walkthrough example -- among
    both. This is the predicate the popup greys options with, computed from
    the same ``check_bake_cost`` the submit path uses so the two can never
    disagree about which combinations are legal.
    """
    from ....kernels.pixel.flourish import recipe as flourish_recipe

    try:
        flourish_recipe.check_bake_cost(recipe, directions)
    except ValueError as exc:
        return str(exc)
    return ""


def can_regenerate(state: Any, tab: Any) -> bool:
    return has_effect(state, tab) and not getattr(tab, "busy", False)


def can_detach(state: Any, tab: Any) -> bool:
    """``can_regenerate``'s shape, kept separate rather than reused directly:
    Detach's op registration used to gate on bare ``has_effect``, the only one
    of this family's predicates missing the ``not busy`` every sibling here
    carries (the 2026-09-15 audit, finding inker-05) -- so the button stayed
    live, and its handler had nothing else standing between a click and a
    Detach landing on a document a save or an export was still writing."""
    return has_effect(state, tab) and not getattr(tab, "busy", False)


def regenerate_reason(state: Any, tab: Any) -> str:
    if tab is None:
        return "Nothing is open."
    if not has_effect(state, tab):
        return NO_EFFECT
    return BUSY if getattr(tab, "busy", False) else ""


def has_conflicts(state: Any, tab: Any) -> bool:
    group = active_group(state, tab)
    return group is not None and bool(tab.doc.flourish_conflicts(group))


def conflicts_reason(state: Any, tab: Any) -> str:
    if tab is None:
        return "Nothing is open."
    if not has_effect(state, tab):
        return NO_EFFECT
    return "" if has_conflicts(state, tab) else NO_CONFLICTS


# -- the pending recipe ---------------------------------------------------------------


def current_recipe(state: Any, tab: Any, group_uid: int) -> Any:
    """What the inspector shows: the pending edit if there is one, else the
    document's own."""
    pending = state.flourish_pending.get(int(group_uid))
    if pending is not None:
        return pending
    held = tab.doc.flourish_state(group_uid)
    return None if held is None else held.recipe


def set_pending(state: Any, group_uid: int, recipe: Any, *, now: float) -> None:
    """Record an edit and restart its debounce clock."""
    state.flourish_pending[int(group_uid)] = recipe
    state.flourish_due[int(group_uid)] = float(now) + DEBOUNCE_SECONDS


def due(state: Any, *, now: float) -> list[int]:
    """Groups whose pending edit has rested long enough to render."""
    return [g for g, at in state.flourish_due.items() if float(now) >= at]


def in_flight(ctx: Any, tab: Any, group_uid: int) -> bool:
    return bool(ctx.busy(render_key(tab, group_uid)))


# -- tasks --------------------------------------------------------------------------


#: The 2026-09-07 audit found every recipe field clamped individually but
#: nothing bounding their *product* before ``bake()``, and ``TaskRunner`` has
#: no cancel API -- so a hand-edited preset could freeze Regenerate
#: indefinitely or raise ``MemoryError``. ``flourish.recipe.check_bake_cost``
#: existed already; nothing called it ahead of either submit point.
BAKE_TOO_COSTLY = (
    "This effect is too large to bake at once -- lower the size, "
    "supersampling, frame count, directions or layers."
)


class SubmitResult(enum.Enum):
    """What answers a caller of :func:`submit_render`/:func:`submit_insert`.

    The two functions used to collapse every refusal into a bare ``False``,
    so a caller could not tell "the bake cost too much" (already toasted, by
    this module, right here) from "something is already running" (not
    toasted here, and the caller's job to say so). The 2026-09-08 audit
    (finding inker-09) found both ``inker_mode.flourish_insert`` and
    ``flourish_regenerate`` adding a *second*, contradicting message on top
    of the correct one whenever the refusal was actually the cost ceiling --
    "already being inserted"/"still running" when nothing was in flight.
    Finding inker-04 is the same ambiguity seen from :func:`tick`, which
    could not tell "refused, stop asking" from "accepted, stop asking" either
    and kept resubmitting -- and re-toasting -- every frame.
    """

    #: Handed to ``ctx.submit`` and accepted.
    ACCEPTED = "accepted"
    #: ``check_bake_cost`` refused it. Already toasted with ``BAKE_TOO_COSTLY``;
    #: retrying with the same recipe would only refuse again.
    TOO_COSTLY = "too_costly"
    #: ``ctx.submit`` refused a key already in flight. Not toasted here --
    #: the caller knows what it is that is already running.
    BUSY = "busy"


def submit_render(
    ctx: Any,
    tab: Any,
    group_uid: int,
    recipe: Any,
    *,
    force: bool = False,
    pending_assets: Mapping[str, np.ndarray] | None = None,
) -> SubmitResult:
    """Bake ``recipe`` off-thread for ``group_uid``. -> whether it was accepted.
    The group's textures go with it, read once here on the frame thread --
    the document's own plus ``pending_assets``, a texture picked up since the
    last render that ``recipe`` may already name but the document does not
    hold yet (``InkerState.flourish_pending_asset``): the bake needs it now,
    and it lands in the document only once the render this call starts comes
    back, through :meth:`Document.apply_flourish`'s own ``new_assets``."""
    from ....kernels.pixel.flourish import bake as flourish_bake
    from ....kernels.pixel.flourish import recipe as flourish_recipe

    try:
        flourish_recipe.check_bake_cost(recipe)
    except ValueError:
        ctx.toast(BAKE_TOO_COSTLY, "warn")
        return SubmitResult.TOO_COSTLY

    key = render_key(tab, group_uid)
    tab_uid = tab.uid
    group_uid = int(group_uid)
    held = tab.doc.flourish_state(group_uid)
    assets = dict(held.assets) if held is not None else {}
    if pending_assets:
        assets.update(pending_assets)

    def work() -> dict[str, Any]:
        def progress(done: int, total: int) -> None:
            ctx.tasks.set_progress(key, 100.0 * done / max(total, 1), f"{done}/{total}")

        baked = flourish_bake.bake(recipe, progress=progress, assets=assets)
        return {"tab": tab_uid, "group": group_uid, "baked": baked, "force": force}

    return SubmitResult.ACCEPTED if ctx.submit(key, work) else SubmitResult.BUSY


def submit_insert(ctx: Any, tab: Any, recipe: Any) -> SubmitResult:
    from ....kernels.pixel.flourish import bake as flourish_bake
    from ....kernels.pixel.flourish import recipe as flourish_recipe

    try:
        flourish_recipe.check_bake_cost(recipe)
    except ValueError:
        ctx.toast(BAKE_TOO_COSTLY, "warn")
        return SubmitResult.TOO_COSTLY

    key = insert_key(tab)
    tab_uid = tab.uid

    def work() -> dict[str, Any]:
        def progress(done: int, total: int) -> None:
            ctx.tasks.set_progress(key, 100.0 * done / max(total, 1), f"{done}/{total}")

        return {"tab": tab_uid, "baked": flourish_bake.bake(recipe, progress=progress)}

    return SubmitResult.ACCEPTED if ctx.submit(key, work) else SubmitResult.BUSY


def tick(ctx: Any, state: Any, tab: Any, *, now: float) -> int:
    """Submit every render that has become due for ``tab``. -> how many.

    **A refusal for cost is popped, not retried.** Before the 2026-09-08 audit
    (finding inker-04) this only popped ``flourish_due`` on acceptance, so a
    recipe over ``MAX_BAKE_COST`` stayed due forever -- ``draw_inspector``
    calls this every frame, and every frame called ``submit_render`` again,
    which re-toasted ``BAKE_TOO_COSTLY`` again, permanently occupying the
    toast stack's five visible slots even after the user clicked away. A
    ``BUSY`` refusal is left due, same as before: ``land`` re-arms the clock
    once the running render's result is in, and retrying it here would only
    race the same key.
    """
    if tab is None or getattr(tab, "busy", False):
        return 0
    sent = 0
    for group in due(state, now=now):
        recipe = state.flourish_pending.get(group)
        if recipe is None or tab.doc.flourish_state(group) is None:
            state.flourish_due.pop(group, None)
            state.flourish_pending.pop(group, None)
            # A texture picked up along the way (``_new_pending_asset``) is
            # still only pixels in ``state`` -- the document was never
            # touched -- so dropping it here is just forgetting them.
            _discard_pending_asset(state, group)
            continue
        if in_flight(ctx, tab, group):
            # Let it rest until the running render lands; ``land`` re-arms the
            # clock when the pending recipe has moved past what it rendered.
            continue
        result = submit_render(
            ctx, tab, group, recipe, pending_assets=state.flourish_pending_asset.get(group)
        )
        if result is SubmitResult.ACCEPTED:
            state.flourish_due.pop(group, None)
            sent += 1
        elif result is SubmitResult.TOO_COSTLY:
            state.flourish_due.pop(group, None)
            state.flourish_pending.pop(group, None)
            _discard_pending_asset(state, group)
    return sent


def _tab_by_uid(state: Any, uid: str) -> Any:
    for tab in getattr(state, "docs", []):
        if tab.uid == uid:
            return tab
    return None


def land(ctx: Any, state: Any, done: Any, *, now: float) -> bool:
    """Frame thread: put a finished bake onto its document. -> whether it did."""
    result = done.result
    if done.error is not None or not isinstance(result, dict):
        ctx.toast(f"The effect could not be rendered: {done.error or 'no result'}", "warn")
        return False
    tab = _tab_by_uid(state, result.get("tab", ""))
    if tab is None:
        # The tab closed; nothing to land on and nobody to tell. Whatever
        # asset write was riding along with this render has no document left
        # to belong to -- just forget it rather than leak the snapshot.
        group_field = result.get("group")
        if group_field is not None:
            state.flourish_pending_asset.pop(int(group_field), None)
        return False
    baked = result["baked"]
    if done.key.startswith(INSERT_KEY):
        group = tab.doc.insert_flourish(baked)
        state.flourish_layer[group] = baked.recipe.layers[0].uid if baked.recipe.layers else 0
        ctx.toast(f"Inserted {baked.recipe.name}: {baked.frame_count} frames.", "success")
        return True
    group = int(result["group"])
    if tab.doc.flourish_state(group) is None:
        ctx.toast("That effect was detached while it rendered; nothing landed.", "info")
        state.flourish_pending.pop(group, None)
        _discard_pending_asset(state, group)
        return False
    try:
        counts = tab.doc.apply_flourish(
            group,
            baked,
            force=bool(result.get("force")),
            new_assets=state.flourish_pending_asset.get(group),
        )
    except ValueError as exc:
        # A refusal from the document, not a crash: the linked-cel check is
        # the one that fires here. Framed as a sentence about the regenerate
        # (the house rule ``test_no_toast_forwards_a_bare_exception`` keeps).
        # A pending asset write is left exactly as it was -- still
        # uncommitted to a step -- so a later, successful regenerate can
        # still fold it in.
        ctx.toast(f"Could not regenerate the effect: {exc}", "warn")
        return False
    state.flourish_pending_asset.pop(group, None)
    pending = state.flourish_pending.get(group)
    if pending is not None and pending == baked.recipe:
        state.flourish_pending.pop(group, None)
    elif pending is not None:
        # The user kept editing while this rendered: render the newer one next.
        state.flourish_due[group] = float(now)
    ctx.toast(counts.sentence(), "warn" if counts.conflicts else "success")
    return True


# -- export and engine snippets -------------------------------------------------------------


def can_export(state: Any, tab: Any) -> bool:
    return (
        has_effect(state, tab)
        and not getattr(tab, "busy", False)
        and bool(getattr(tab.doc.anim, "tags", None))
    )


def export_reason(state: Any, tab: Any) -> str:
    if tab is None:
        return "Nothing is open."
    if not has_effect(state, tab):
        return NO_EFFECT
    if getattr(tab, "busy", False):
        return BUSY
    return "" if getattr(tab.doc.anim, "tags", None) else "This document has no tags to export by."


def tag_names(tab: Any) -> list[str]:
    anim = getattr(tab.doc, "anim", None)
    return [] if anim is None else [tag.name for tag in anim.tags]


def snippet_info(tab: Any, tag_name: str) -> dict[str, Any] | None:
    """What ``engines.snippet`` needs for one exported phase: the file the
    per-tag export writes (``sheetout.DEFAULT_TAG_TEMPLATE`` over the
    document's title), the frames the tag spans, the rate from the tag's
    first frame, the loop flag, and the origin -- the canvas centre, which is
    where ``bake`` puts an effect by construction."""
    from ....kernels.pixel import sheetout
    from ....kernels.pixel.flourish import engines

    anim = getattr(tab.doc, "anim", None)
    if anim is None:
        return None
    tag = next((t for t in anim.tags if t.name == tag_name), None)
    if tag is None:
        return None
    first, last = sheetout.tag_span(anim, tag)
    duration = max(1, int(anim.frames[first].duration_ms))
    title = Path(str(getattr(tab, "title", "") or "effect")).stem or "effect"
    stem = sheetout.filename_for(sheetout.DEFAULT_TAG_TEMPLATE, title=title, tag=tag.name)
    width, height = tab.doc.size
    return engines.describe(
        name=f"{title} {tag.name}",
        image=f"{stem}.png",
        frame_width=width,
        frame_height=height,
        frames=last - first + 1,
        fps=max(1, round(1000.0 / duration)),
        loop=bool(tag.loop),
        origin=(width // 2, height // 2),
    )


def snippet_text(tab: Any, tag_name: str, engine: str) -> str:
    from ....kernels.pixel.flourish import engines

    info = snippet_info(tab, tag_name)
    if info is None:
        return ""
    return engines.snippet(engine, info)


# -- textures --------------------------------------------------------------------------------


def _texture_target(state: Any, tab: Any, group: int | None) -> Any:
    """The recipe layer 'Use selection as texture' would write into: the
    inspector's active layer if the current recipe still carries it, else
    the stack's last layer -- ``_assign_texture``'s own resolution, read
    ahead of time so a caller can refuse before anything is committed. None
    with no group, no recipe, or a recipe with no layers at all."""
    if group is None:
        return None
    recipe = current_recipe(state, tab, group)
    if recipe is None or not recipe.layers:
        return None
    uid = state.flourish_layer.get(group)
    return next((each for each in recipe.layers if each.uid == uid), recipe.layers[-1])


def _has_texture_slot(state: Any, tab: Any, group: int | None) -> bool:
    from ....kernels.pixel.flourish import prims

    target = _texture_target(state, tab, group)
    return target is not None and "texture" in prims.params_of(target.kind)


def can_texture_selection(state: Any, tab: Any) -> bool:
    # The 2026-09-15 audit, inker-05: this predicate never checked ``busy`` at
    # all, unlike every sibling in this module (``can_regenerate``,
    # ``can_prompt``, ``can_restyle``...), and neither did the handler it
    # gates (``inker_mode.flourish_texture_selection``) -- so the button
    # stayed clickable, and pressing it while a save or an export was writing
    # the document pushed a pending recipe edit into ``state`` regardless.
    if (
        not has_effect(state, tab)
        or getattr(tab, "busy", False)
        or getattr(tab.doc, "mask", None) is None
    ):
        return False
    return _has_texture_slot(state, tab, active_group(state, tab))


def texture_selection_reason(state: Any, tab: Any) -> str:
    if tab is None:
        return "Nothing is open."
    if not has_effect(state, tab):
        return NO_EFFECT
    # Checked ahead of the slot/selection questions below, the same order
    # ``regenerate_reason`` settles busy in -- so a disabled button always has
    # a reason that matches ``can_texture_selection``'s own answer (the
    # 2026-09-15 audit, inker-05: before this, a busy document disabled the
    # button but this string stayed "", leaving the tooltip blank).
    if getattr(tab, "busy", False):
        return BUSY
    # The 2026-09-14 audit (inker-06): this used to say only NO_SELECTION,
    # so a glow layer -- no ``texture`` parameter at all -- let the button
    # through and ``texture_from_selection`` committed an asset nothing in
    # the document would ever reference. Check the slot before the selection:
    # a selection cannot fix a layer that has nowhere to put the result.
    if not _has_texture_slot(state, tab, active_group(state, tab)):
        return NO_TEXTURE_SLOT
    return "" if getattr(tab.doc, "mask", None) is not None else NO_SELECTION


def can_texture_generate(state: Any, tab: Any) -> bool:
    return has_effect(state, tab) and not getattr(tab, "busy", False)


def texture_generate_reason(state: Any, tab: Any) -> str:
    return regenerate_reason(state, tab)


def _new_pending_asset(
    state: Any, tab: Any, group: int, pixels: np.ndarray, *, stem: str = "tex"
) -> str:
    """Allocate an id for a texture that is not in the document yet, and hold
    its pixels in ``InkerState.flourish_pending_asset`` until the render
    that names it lands.

    **The document is not touched here at all** -- that is the whole point
    of this shape. An earlier cut of the fix committed the asset immediately
    (outside history, since the step covering it would not be pushed until
    the render landed) so the bake would have bytes to read; that mutation
    was invisible to everything else that could land on the same group in
    the meantime -- an undo of an earlier step, a resolve, a second texture
    -- each of which either restored a document state from before the
    texture existed (dropping it) or read/wrote past it, and the eventual
    landing step, built from a stale snapshot, put back the wrong thing (the
    2026-09-14 audit, inker-06, second half, redesigned once this surfaced).
    Keeping the pixels out of the document until :func:`land` folds them into
    ``apply_flourish``'s own step means nothing else can ever observe or
    restore a half-written state for them.

    ``next_asset_id``'s ``taken`` is checked against the group's *own*
    pending ids too (not only the document's), so a second texture picked
    before the first lands still gets a name of its own.
    """
    held = tab.doc.flourish_state(group)
    pending = state.flourish_pending_asset.setdefault(int(group), {})
    asset_id = held.next_asset_id(stem, taken=pending) if held is not None else f"{stem}1"
    pending[asset_id] = np.ascontiguousarray(pixels, dtype=np.uint8).copy()
    return asset_id


def _discard_pending_asset(state: Any, group: int) -> None:
    """Forget a write ``_new_pending_asset`` made that will never land. Not
    an undo -- the document was never touched -- just the pixels dropped
    from ``state`` before they can be mistaken for still wanting a step."""
    state.flourish_pending_asset.pop(int(group), None)


def texture_from_selection(ctx: Any, state: Any, tab: Any) -> str | None:
    """The selection's pixels become a texture of the active effect, and the
    inspector's layer takes it if it has a ``texture`` parameter. One step.

    Refuses with nothing pushed when the target layer has no texture slot at
    all (the 2026-09-14 audit, inker-06, first half): committing the asset
    first and discovering only afterwards that no layer would take it left
    an invisible, permanent undo step behind -- an asset with nothing in the
    document ever pointing at it. The asset itself is held only as pixels in
    ``state.flourish_pending_asset`` until the render that lands the pending
    recipe edit pointing at it folds the two into one ``FlourishEdit``
    (``_new_pending_asset``, ``Document.apply_flourish``'s ``new_assets``),
    so one undo reverses the texture and the render together, and nothing
    that lands on this group before then can observe an asset the document
    does not really have yet (inker-06, second half).
    """
    group = active_group(state, tab)
    if group is None:
        return None
    if not _has_texture_slot(state, tab, group):
        state.say(NO_TEXTURE_SLOT)
        return None
    cutout = tab.doc.selection_cutout()
    if cutout is None or not cutout[..., 3].any():
        state.say(NO_SELECTION)
        return None
    asset_id = _new_pending_asset(state, tab, group, cutout)
    if not _assign_texture(state, tab, group, asset_id):
        # Unreachable given the ``_has_texture_slot`` check above -- both
        # resolve the same target the same way and nothing between the two
        # calls can move it -- but a future refactor that lets the two
        # disagree must not leave orphaned pixels sitting in ``state``.
        _discard_pending_asset(state, group)
        return None
    return asset_id


def _assign_texture(state: Any, tab: Any, group: int, asset_id: str) -> bool:
    """Point the inspector's current layer at ``asset_id`` when it can take
    one, as a pending edit -- the render that lands it is one step. -> whether
    it did."""
    from ....kernels.pixel.flourish import prims

    recipe = current_recipe(state, tab, group)
    if recipe is None or not recipe.layers:
        return False
    uid = state.flourish_layer.get(group)
    layer = next((each for each in recipe.layers if each.uid == uid), recipe.layers[-1])
    if "texture" not in prims.params_of(layer.kind):
        return False
    edited = recipe.replace_layer(layer.with_param("texture", asset_id))
    set_pending(state, group, edited, now=clock())
    return True


def key_out_black(pixels: np.ndarray) -> np.ndarray:
    """A straight-alpha cutout of a picture painted on black: alpha from the
    brightest channel. The offline fallback when no matting model is present,
    and the *right* answer for an additive VFX texture, whose black *is*
    transparency."""
    rgb = pixels[..., :3].astype(np.float32)
    alpha = rgb.max(axis=-1)
    out = np.empty(pixels.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = pixels[..., :3]
    out[..., 3] = np.clip(np.rint(alpha), 0, 255).astype(np.uint8)
    return out


def submit_texture(ctx: Any, state: Any, tab: Any, subject: str) -> bool:
    """Ask the pixel model for a texture. The job goes to the queue like any
    reference job (``inker_bridge.submit_inpaint``'s door); ``poll_texture``
    watches for it and ``land_texture`` puts the cutout on the effect."""
    group = active_group(state, tab)
    if group is None or tab.busy:
        return False
    if state.flourish_texture_pending is not None:
        state.say(TEXTURE_PENDING)
        return False
    prompt = TEXTURE_PROMPT_TEMPLATE.format(subject=subject.strip() or "magical flame")
    pending = {
        "tab_uid": tab.uid,
        "group": int(group),
        "layer": state.flourish_layer.get(group),
        "job_id": "",
        "next_poll": 0.0,
        "subject": subject.strip(),
    }
    key = f"{TEXTURE_KEY}:{tab.uid}"

    def run() -> Any:
        from ....service import jobs as svc_jobs

        return svc_jobs.create_job(
            ctx.svc,
            kind="text",
            prompt=prompt,
            negative="photo, realistic, text, watermark, frame, border",
            output="reference",
            count=1,
        )

    if not ctx.submit(key, run):
        return False
    state.flourish_texture_pending = pending
    ctx.toast("Generating a texture...")
    return True


def on_texture_queued(ctx: Any, state: Any, done: Any) -> None:
    pending = state.flourish_texture_pending
    if pending is None:
        return
    result = done.result
    job_id = ""
    if done.error is None and isinstance(result, dict):
        job_id = str(result.get("id") or "")
        if not job_id:
            ids = result.get("ids") or result.get("jobs") or []
            job_id = str(ids[0]) if ids else ""
    if not job_id:
        state.flourish_texture_pending = None
        ctx.toast(f"The texture was not queued: {done.error or 'no job id'}.", "warn")
        return
    pending["job_id"] = job_id


def poll_texture(ctx: Any, state: Any, *, now: float) -> None:
    """Frame thread, cheap: once every ``TEXTURE_POLL_S`` ask whether the job
    is done, and hand the decode to a task."""
    pending = state.flourish_texture_pending
    if pending is None or not pending.get("job_id"):
        return
    if now < float(pending.get("next_poll") or 0.0):
        return
    pending["next_poll"] = now + TEXTURE_POLL_S
    try:
        job = ctx.svc.store.get(pending["job_id"])
    except Exception:  # noqa: BLE001 -- the store answers next tick
        return
    if job is None:
        state.flourish_texture_pending = None
        return
    status = job.get("status")
    if status in ("queued", "running"):
        return
    state.flourish_texture_pending = None
    if status != "done":
        ctx.toast(f"The texture {status}: {job.get('error') or 'no result'}.", "warn")
        return
    image_path = ctx.svc.job_dir(pending["job_id"]) / "input.png"
    key = f"{TEXTURE_LAND_KEY}:{pending['tab_uid']}"
    ctx.submit(key, decode_texture, pending, image_path, ctx.svc)


def decode_texture(
    pending: dict[str, Any], image_path: Any, svc: Any = None
) -> dict[str, Any] | None:
    """Task thread. The picture as a cutout, brought down to ``TEXTURE_MAX_PX``:
    the matting model where the machine has one, black-keyed otherwise."""
    from PIL import Image

    from ....core.safeio import pixelguard
    from ....pipelines import matting

    # The 2026-09-18 audit (inker-04): this used a bare ``Image.open``, one of
    # the two loaders in this file ``pixelguard``'s own docstring did not yet
    # count among its doors -- an untrusted PNG dropped on the texture picker
    # allocated before any size check ran.
    try:
        picture = Image.fromarray(pixelguard.decode_rgba(image_path, "a Flourish texture"), "RGBA")
    except (OSError, ValueError):
        return None
    picture.thumbnail((TEXTURE_MAX_PX, TEXTURE_MAX_PX), Image.Resampling.LANCZOS)
    pixels = np.asarray(picture, dtype=np.uint8).copy()
    config = getattr(svc, "config", None)
    source = "black-key"
    if config is not None and matting.available(config):
        try:
            found, source = matting.mask(picture, config)
            cut = pixels.copy()
            cut[..., 3] = np.where(np.asarray(found, dtype=bool), cut[..., 3], 0)
            pixels = cut
        except Exception:  # noqa: BLE001 -- the fallback is always right, if rougher
            source = "black-key"
            pixels = key_out_black(pixels)
    else:
        pixels = key_out_black(pixels)
    return {"pending": pending, "pixels": pixels, "source": source}


def land_texture(ctx: Any, state: Any, done: Any) -> bool:
    result = done.result
    if done.error is not None or not isinstance(result, dict):
        ctx.toast("The generated texture could not be read.", "warn")
        return False
    pending = result["pending"]
    tab = _tab_by_uid(state, pending["tab_uid"])
    if tab is None:
        return False
    group = int(pending["group"])
    if tab.doc.flourish_state(group) is None:
        ctx.toast("That effect was detached while its texture generated.", "info")
        return False
    asset_id = _new_pending_asset(state, tab, group, result["pixels"], stem="gen")
    if pending.get("layer") is not None:
        state.flourish_layer[group] = pending["layer"]
    if not _assign_texture(state, tab, group, asset_id):
        # Unlike ``texture_from_selection`` this door has no slot check ahead
        # of the (expensive, already-run) generation, so this is reachable:
        # the layer the inspector is showing when the picture comes back has
        # no ``texture`` parameter. Forget the pixels rather than leave them
        # in ``state`` with nothing that will ever land or discard them.
        _discard_pending_asset(state, group)
        ctx.toast("The generated texture has no layer to land on.", "info")
        return False
    ctx.toast(f"Texture {asset_id} added ({result.get('source', '')}).", "success")
    return True


# -- the prompt field --------------------------------------------------------------------------


#: The directory under the model root the prompt field looks in. **Not a
#: registry entry**: every ``models`` entry carries a fetch pinned to a
#: revision, and the pin comes from the measurement that picks the model
#: (still owed, on the human's list). Until then a user who wants to try one puts an instruct
#: model's ``config.json`` and safetensors here by hand, and doctor says so.
TEXT_MODEL_DIR = "text-instruct"


def text_model_dir(config: Any) -> Path | None:
    """Where the text model would be, or None with no config in reach."""
    if config is None:
        return None
    return Path(config.t2i_model_root) / TEXT_MODEL_DIR


def text_model_present(config: Any) -> bool:
    """Weights on disk: ``config.json`` and at least one safetensors file, the
    same two facts ``fetch.present`` asks of every helper model.

    Both checks require a *regular file*, not merely a path that exists with
    the right name -- ``.exists()``/``rglob`` alone are also true of a
    directory named ``config.json`` or a directory ending in ``.safetensors``.
    The 2026-09-08 audit (finding inker-12): this is the one "present is not
    usable" door doctor.py's suspect-file check and ``pack_worker``'s import
    probe already guard everywhere else, left unguarded on the door a user
    hand-populates -- a stray directory or a partial copy reported the model
    available, and ``text_model_available`` then spawned ``recipe_worker`` for
    every prompt before falling back to the keyword mapper.
    """
    base = text_model_dir(config)
    if base is None or not (base / "config.json").is_file():
        return False
    return any(p.is_file() for p in base.rglob("*.safetensors"))


#: :func:`text_model_available`'s memo, keyed on the resolved model directory.
#: The 2026-09-18 audit (inker-03): the Flourish inspector's "model"/"keywords"
#: indicator called this, unmemoised, on every imgui frame the inspector
#: stayed open -- an ``rglob`` of the model directory plus two
#: ``importlib.util.find_spec`` calls, repeated dozens of times a second for
#: no reason, the same shape as ``_popup_names_cached`` (inker-11,
#: 2026-09-11) fixed in ``ui/panes/flourish.py``. Module state rather than a
#: field on ``InkerState`` for the same reason as that fix: the cache is
#: about this directory, not about any one document.
_TEXT_MODEL_AVAILABLE: dict[str, bool] = {}


def reset_text_model_available_cache() -> None:
    """Drop the memo. Call after anything that could change the answer:
    weights fetched or removed, ``t2i_model_root`` edited in Settings."""
    _TEXT_MODEL_AVAILABLE.clear()


def text_model_available(config: Any) -> bool:
    """Weights on disk *and* the packages to run them, checked before any torch
    import -- ``tests/test_offline.py``'s ordering rule.

    Memoised per resolved model directory; call
    :func:`reset_text_model_available_cache` when the answer could have
    changed."""
    key = str(text_model_dir(config))
    cached = _TEXT_MODEL_AVAILABLE.get(key)
    if cached is not None:
        return cached
    if not text_model_present(config):
        result = False
    else:
        import importlib.util

        result = all(
            importlib.util.find_spec(name) is not None for name in ("torch", "transformers")
        )
    _TEXT_MODEL_AVAILABLE[key] = result
    return result


def can_prompt(state: Any, tab: Any) -> bool:
    return has_effect(state, tab) and not getattr(tab, "busy", False)


def prompt_reason(state: Any, tab: Any) -> str:
    return regenerate_reason(state, tab)


def ask_words(recipe: Any, text: str, *, model_dir: Path | None) -> tuple[Any, list[str], str]:
    """Task thread. -> ``(recipe, notes, source)``: the text model when there
    is one and it answers, the keyword mapper otherwise -- always something,
    and the source says which, because a change the user cannot attribute is
    a change they cannot trust."""
    from ....kernels.pixel.flourish import keywords

    if model_dir is not None:
        diff, why = run_text_model(recipe, text, model_dir)
        if diff is not None:
            changed, notes = keywords.apply_diff(recipe, diff)
            return changed, notes, "model"
        changed, notes = keywords.apply(recipe, text)
        return changed, [f"model: {why}; used the keyword mapper", *notes], "keywords"
    changed, notes = keywords.apply(recipe, text)
    return changed, notes, "keywords"


def run_text_model(recipe: Any, text: str, model_dir: Path) -> tuple[dict[str, Any] | None, str]:
    """One child, one answer. -> ``(diff, reason-if-none)``."""
    import json
    import subprocess
    import sys

    from .... import winjob
    from ....kernels.pixel.flourish import keywords

    request = {
        "model_dir": str(model_dir),
        "recipe": keywords.describe_for_model(recipe),
        "request": text,
        "schema": keywords.DIFF_SCHEMA,
    }
    argv = [sys.executable, "-m", "realmspinner.pipelines.recipe_worker"]
    try:
        proc = winjob.run(
            argv,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PROMPT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None, f"no answer within {PROMPT_TIMEOUT_S:.0f} s"
    except OSError as exc:
        return None, f"could not start the text model: {exc}"
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None, "the text model said nothing"
    try:
        answer = json.loads(line[-1])
    except json.JSONDecodeError:
        return None, "the text model's answer was not JSON"
    if not isinstance(answer, dict) or answer.get("kind") != "ok":
        reason = answer.get("error") if isinstance(answer, dict) else "malformed answer"
        return None, str(reason or "the text model failed")
    diff = answer.get("diff")
    return (diff if isinstance(diff, dict) else None), "the diff was not an object"


def submit_prompt(ctx: Any, state: Any, tab: Any, text: str) -> bool:
    """Words -> a pending recipe, off-thread; ``land_prompt`` sets it pending
    and the ordinary render then lands it as one step."""
    group = active_group(state, tab)
    text = (text or "").strip()
    if group is None or not text or tab.busy:
        return False
    recipe = current_recipe(state, tab, group)
    if recipe is None:
        return False
    key = f"{PROMPT_KEY}:{tab.uid}:{group}"
    config = getattr(getattr(ctx, "svc", None), "config", None)
    model_dir = text_model_dir(config) if text_model_available(config) else None
    tab_uid = tab.uid

    def work() -> dict[str, Any]:
        changed, notes, source = ask_words(recipe, text, model_dir=model_dir)
        # ``base`` is the snapshot ``ask_words`` actually started from -- the
        # 2026-09-16 audit found ``land_prompt`` had no way to tell that
        # ``state.flourish_pending[group]`` had moved on while this ran
        # off-thread, so it overwrote a manual edit staged after submit with
        # the prompt's own edit, itself derived from this now-stale ``recipe``.
        return {
            "tab": tab_uid,
            "group": group,
            "recipe": changed,
            "notes": notes,
            "source": source,
            "base": recipe,
        }

    if not ctx.submit(key, work):
        state.say("The last prompt is still being read.")
        return False
    return True


def land_prompt(ctx: Any, state: Any, done: Any, *, now: float) -> bool:
    result = done.result
    if done.error is not None or not isinstance(result, dict):
        ctx.toast(f"The prompt could not be applied: {done.error or 'no result'}", "warn")
        return False
    tab = _tab_by_uid(state, result.get("tab", ""))
    if tab is None:
        return False
    group = int(result["group"])
    if tab.doc.flourish_state(group) is None:
        return False
    notes = list(result.get("notes") or [])
    recipe = result["recipe"]
    live = current_recipe(state, tab, group)
    base = result.get("base")
    if base is not None and live != base:
        # The 2026-09-16 audit, inker-flourish-02: this used to apply
        # ``recipe`` unconditionally, no matter what ``state.flourish_pending``
        # had become while ``ask_words`` ran off-thread -- so a manual edit
        # staged after submit and before land vanished, replaced by the
        # prompt's own edit computed from the older ``base``. Detected here,
        # the newer edit is left exactly as it is (it already carries its own
        # debounce clock from whatever call to ``set_pending`` staged it) and
        # the prompt's own answer is dropped rather than overwrite it.
        ctx.toast("The effect changed while the prompt was read; resend it to apply.", "info")
        return False
    if recipe == live:
        ctx.toast("; ".join(notes) or "Nothing changed.", "info")
        return False
    set_pending(state, group, recipe, now=now)
    source = "model" if result.get("source") == "model" else "keywords"
    ctx.toast(f"[{source}] " + "; ".join(notes[:6]), "success")
    return True


# -- restyled keyframes -----------------------------------------------------------------------


def can_restyle(state: Any, tab: Any) -> bool:
    return has_effect(state, tab) and not getattr(tab, "busy", False)


def restyle_reason(state: Any, tab: Any) -> str:
    return regenerate_reason(state, tab)


def phase_names(state: Any, tab: Any) -> list[str]:
    group = active_group(state, tab)
    if group is None:
        return []
    held = tab.doc.flourish_state(group)
    return [p.name for p in held.recipe.phases] if held is not None else []


def _phase_span(state_held: Any, anim: Any, phase_name: str) -> tuple[int, int] | None:
    """The flat frame span of one phase of the effect, from its tag."""
    from ....kernels.pixel import sheetout

    for tag in anim.tags:
        if tag.name == phase_name or tag.name.startswith(phase_name + "/"):
            return sheetout.tag_span(anim, tag)
    return None


def submit_restyle(
    ctx: Any,
    state: Any,
    tab: Any,
    *,
    phase: str,
    subject: str,
    strength: float = 0.55,
    anchors: int = 3,
) -> bool:
    """Send ``anchors`` frames of ``phase`` -- the effect's own composite --
    through the image model as img2img, one reference job each; ``poll_restyle``
    collects them and ``land_restyle`` interpolates the rest and lands a
    snapshot track. Opt-in, never default: whether this beats the procedural
    frames is a measurement, not a setting."""
    from ....kernels.pixel import sheetout
    from ....kernels.pixel.flourish import keyframes

    group = active_group(state, tab)
    if group is None or tab.busy:
        return False
    if state.flourish_restyle_pending is not None:
        state.say(RESTYLE_PENDING)
        return False
    held = tab.doc.flourish_state(group)
    anim = tab.doc.anim
    if held is None or anim is None:
        return False
    span = _phase_span(held, anim, phase)
    if span is None:
        state.say(f"This effect has no phase called {phase!r}.")
        return False
    first, last = span
    frames = keyframes.anchor_frames(first, last, anchors)
    track_uids = [uid for uid in held.tracks.values()]
    uids = sheetout.frame_uids(tab.doc)
    # Composite here, on the frame thread, because ``flatten_subset`` reads
    # ``Document.frame_stack`` directly and the document must not move while
    # the task runs. The PNG encode does not need the document at all, so it
    # moves into ``run`` below -- the 2026-09-14 audit (inker-05) found this
    # loop encoding every anchor on the frame thread, ~51 ms per anchor at
    # 1024^2 and 100-300 ms per press, none of it needing to block a frame.
    planes: dict[int, np.ndarray] = {}
    for index in frames:
        plane = sheetout.flatten_subset(tab.doc, uids[index], track_uids)
        planes[index] = np.ascontiguousarray(plane, dtype=np.uint8).copy()
    prompt = RESTYLE_PROMPT_TEMPLATE.format(subject=subject.strip() or "painted magical effect")
    pending = {
        "tab_uid": tab.uid,
        "group": int(group),
        "phase": phase,
        "span": [first, last],
        "jobs": {},  # frame index -> job id
        "frames": list(frames),
        "next_poll": 0.0,
        "subject": subject.strip(),
    }
    key = f"{RESTYLE_KEY}:{tab.uid}"
    strength = min(0.95, max(0.1, float(strength)))

    def run() -> Any:
        from ....service import jobs as svc_jobs

        ids: dict[int, str] = {}
        for index, plane in planes.items():
            png = _png_bytes(plane)
            result = svc_jobs.create_job(
                ctx.svc,
                kind="text",
                prompt=prompt,
                negative="photo, text, watermark, frame, border",
                reference=png,
                init_image=True,
                init_strength=strength,
                output="reference",
                count=1,
                reference_prep=False,
            )
            job_id = str(result.get("id") or "") if isinstance(result, dict) else ""
            if not job_id and isinstance(result, dict):
                found = result.get("ids") or result.get("jobs") or []
                job_id = str(found[0]) if found else ""
            if not job_id:
                raise RuntimeError(f"frame {index} was not queued")
            ids[index] = job_id
        return ids

    if not ctx.submit(key, run):
        return False
    state.flourish_restyle_pending = pending
    ctx.toast(f"Restyling {len(frames)} keyframes of {phase}...")
    return True


def _png_bytes(plane: np.ndarray) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(plane, dtype=np.uint8), "RGBA").save(buf, "PNG")
    return buf.getvalue()


def on_restyle_queued(ctx: Any, state: Any, done: Any) -> None:
    pending = state.flourish_restyle_pending
    if pending is None:
        return
    if done.error is not None or not isinstance(done.result, dict) or not done.result:
        state.flourish_restyle_pending = None
        ctx.toast(f"The restyle was not queued: {done.error or 'no jobs'}.", "warn")
        return
    pending["jobs"] = {int(k): str(v) for k, v in done.result.items()}


def poll_restyle(ctx: Any, state: Any, *, now: float) -> None:
    pending = state.flourish_restyle_pending
    if pending is None or not pending.get("jobs"):
        return
    if now < float(pending.get("next_poll") or 0.0):
        return
    pending["next_poll"] = now + TEXTURE_POLL_S
    statuses: dict[int, str] = {}
    for index, job_id in pending["jobs"].items():
        try:
            job = ctx.svc.store.get(job_id)
        except Exception:  # noqa: BLE001 -- the store answers next tick
            return
        if job is None:
            state.flourish_restyle_pending = None
            return
        statuses[index] = str(job.get("status"))
        if statuses[index] not in ("queued", "running", "done"):
            state.flourish_restyle_pending = None
            ctx.toast(f"The restyle {statuses[index]}: {job.get('error') or 'no result'}.", "warn")
            return
    if any(s in ("queued", "running") for s in statuses.values()):
        return
    state.flourish_restyle_pending = None
    paths = {
        index: ctx.svc.job_dir(job_id) / "input.png" for index, job_id in pending["jobs"].items()
    }
    tab = _tab_by_uid(state, pending["tab_uid"])
    held = tab.doc.flourish_state(int(pending["group"])) if tab is not None else None
    recipe = held.recipe if held is not None else None
    size = tab.doc.size if tab is not None else None
    key = f"{RESTYLE_LAND_KEY}:{pending['tab_uid']}"
    ctx.submit(key, decode_restyle, pending, paths, recipe, size)


def decode_restyle(
    pending: dict[str, Any], paths: dict[int, Any], recipe: Any, size: tuple[int, int] | None
) -> dict[str, Any] | None:
    """Task thread: read every anchor, key it out, interpolate the span."""
    from PIL import Image

    from ....core.safeio import pixelguard
    from ....kernels.pixel.flourish import keyframes

    if recipe is None or size is None:
        return None
    anchors: dict[int, np.ndarray] = {}
    for index, path in paths.items():
        # The 2026-09-18 audit (inker-04): the second of the two bare
        # ``Image.open`` loaders in this file, same gap as ``decode_texture``.
        try:
            pixels = pixelguard.decode_rgba(path, "a Flourish restyle anchor")
            picture = Image.fromarray(pixels, "RGBA").resize(size, Image.Resampling.LANCZOS)
        except (OSError, ValueError):
            return None
        pixels = np.asarray(picture, dtype=np.uint8).copy()
        if not (pixels[..., 3] < 255).any():
            pixels = key_out_black(pixels)  # the model painted on an opaque ground
        anchors[int(index)] = pixels
    first, last = pending["span"]
    # ``size`` is the document's canvas -- what every anchor above was just
    # resized to -- not the recipe's own, which is centred and offset inside
    # it and can be smaller. The field has to share that shape or ``_shift``
    # broadcasts a (recipe h, recipe w) displacement against a (doc h, doc w)
    # plane (finding #12).
    field = keyframes.field_from_recipe(recipe, size=size)
    planes = keyframes.interpolate(anchors, int(first), int(last), field)
    cels = {int(first) + i: plane for i, plane in enumerate(planes)}
    return {"pending": pending, "cels": cels}


def land_restyle(ctx: Any, state: Any, done: Any) -> bool:
    result = done.result
    if done.error is not None or not isinstance(result, dict):
        ctx.toast("The restyled keyframes could not be read.", "warn")
        return False
    pending = result["pending"]
    tab = _tab_by_uid(state, pending["tab_uid"])
    if tab is None:
        return False
    group = int(pending["group"])
    held = tab.doc.flourish_state(group)
    if held is None:
        ctx.toast("That effect was detached while its keyframes restyled.", "info")
        return False
    anim = tab.doc.anim
    span = _phase_span(held, anim, pending["phase"]) if anim is not None else None
    if span is None or list(span) != list(pending["span"]):
        # The 2026-09-19 audit, finding inker-02: this used to land on
        # ``pending["span"]`` -- the phase's flat frame span captured once,
        # back at ``submit_restyle`` time -- no matter what the timeline had
        # become while the restyle rendered off-thread. A frame cut moves
        # ``tag_span``'s clamp (it bounds a tag to ``len(anim.frames) - 1``),
        # so the span recomputed here differs the moment frames the restyle
        # was keyed to no longer exist; landing on the stale one anyway made
        # ``insert_flourish_track``'s ``_flourish_ensure_frames`` regrow the
        # grid to fit indices the user had just cut, undoing their own edit
        # and placing the restyled cels on the wrong frames. ``land_prompt``
        # already refuses a stale snapshot this way; this is the same
        # discipline for a restyle.
        ctx.toast(
            "The effect's frames changed while the restyle rendered; it was dropped.",
            "info",
        )
        return False
    name = f"Restyled {pending['phase']}"
    tab.doc.insert_flourish_track(group, name, result["cels"])
    ctx.toast(
        f"{name}: {len(result['cels'])} frames from {len(pending['frames'])} keyframes.",
        "success",
    )
    return True
