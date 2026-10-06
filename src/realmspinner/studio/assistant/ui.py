"""Familiar's in-session conversation: state, submission and the Familiar
dock's expanded body. T5 of the Familiar programme.

**Why this lives at studio level, not inside ``studio/familiar/``.** Exactly
``studio/assistant/preview.py``'s own reason (see that module's docstring):
this reaches ``agent_clay`` (to read the live scene for a Build request) and
``clay_view``/``clay_mode`` (to show and apply a ghost preview), all of which
``studio/familiar/`` is pinned never to import, even lazily.

**State lives on the frame thread, submission does not block it.** Every
network round trip goes through ``ctx.submit`` under one of two keys
(:data:`CHAT_KEY`, :data:`BUILD_KEY`) running a service-layer call
(``service.familiar.chat_reply``/``clay_build``) on a ``TaskRunner`` worker
thread; the frame thread only ever reads :class:`FamiliarUIState` and calls
:func:`on_task_done` when a result lands, the same shape every other mode's
``ctx.submit``/``on_task_done`` pair already uses (``studio/modes/clay/mode.py``'s module
docstring states the rule this module follows). Landing a build that came
back with calls is itself two more of these round trips, not one inline
computation: :data:`LAND_KEY` carries the ``clay_batch`` run itself off the
frame thread too (the 2026-09-17 audit, familiar-01) -- see
:func:`_submit_build_preview`/:func:`_land_build_preview`.

**Threads are keyed by mode/tab, never by conversation content.** See
``studio/familiar/threads.py``'s own docstring: a thread is display and
refinement history, never fed back to the model as context -- the model
still sees only the current prompt (and, for Clay, the compacted scene).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The task-runner keys a chat send / Clay build submit under. Prefixed
#: ``familiar/`` (not ``familiar-``, unlike every Clay key's ``clay-``
#: prefix) simply because there is only ever one of each in flight at a time
#: -- no per-tab or per-job suffix is needed, since ``ctx.submit`` already
#: refuses a second submit under the same key while the first is running,
#: which is exactly "disable Send/Build while busy" for free.
CHAT_KEY = "familiar/chat"
BUILD_KEY = "familiar/build"
#: T7's Create press on a pending character plan. Its own key, not
#: ``CHAT_KEY`` -- a plan sits waiting for a press with no chat in flight,
#: and the two must be free to overlap the way any two independent
#: ``ctx.submit`` keys already are.
CHARACTER_KEY = "familiar/character"
#: The rows a "missing" refusal (:data:`service.familiar.REASONS`) names --
#: Familiar has no ``ServiceError.rows`` of its own the way a form-field
#: refusal does (there is no field to hang them off), so :func:`draw_expanded`
#: offers exactly these three rather than reading ``ctx.state.field_error_rows``.
#: The 2026-09-23 audit (familiar-05): ``ui.reason`` was set on every refusal
#: but never read anywhere, contrary to this module's own docstring and
#: ``service.familiar``'s ("the pane can choose an icon or an action -- an
#: Install... button"), so a missing engine/weights refusal offered no way
#: back to Settings -> Models short of the palette.
MISSING_FAMILIAR_ROWS = ("familiar_runtime", "familiar_runtime_cudart", "familiar_gguf")
#: The batch itself -- landing a build's ``clay_batch`` run, split off
#: ``CHAT_KEY``/``BUILD_KEY`` by the 2026-09-17 audit (familiar-01): up to
#: ``agent_clay.BATCH_MAX`` calls, booleans included, used to run inline
#: inside ``on_task_done``, on the pygame frame thread -- 624-705 ms wall for
#: 16 uv-sphere adds and 15 unions with no GPU or weights involved, freezing
#: the app for ~40 frames on an ordinary Build. Its own key for the same
#: reason ``CHAT_KEY``/``BUILD_KEY`` get theirs -- see :func:`_submit_build_
#: preview`/:func:`_land_build_preview` for the two-phase shape this key
#: exists to carry.
LAND_KEY = "familiar/land"

#: The Familiar dock no longer has a width preference of its own (2026-09-23's
#: proportional shell): its width is a fixed share of the room
#: (``layout.proportions``), the same for every profile, so there is nothing
#: left to persist here -- the old ``dock_width``/``set_dock_width`` pair and
#: the settings key they read/wrote are simply gone.


@dataclass
class FamiliarUIState:
    """The bottom pane's own session state -- never persisted, same as
    ``familiar.threads.Threads`` (see that module's docstring): a refusal or
    a pending preview describes this run, not something worth restoring
    across a restart."""

    #: Whether the Familiar dock is open (its proportional share of the room,
    #: up to 15%) rather than collapsed to its 5%-share strip.
    #: Toggled by the user, never by a message arriving -- an answer landing
    #: while the dock is closed should not itself pop it open under whatever
    #: the user is doing.
    expanded: bool = False
    #: The input line's live text, kept here (not a local in ``draw``) so a
    #: reply landing mid-type does not require choosing what happens to
    #: unsent text -- there is exactly one input line for the whole app, and
    #: it survives a Done landing untouched.
    input_text: str = ""
    #: ``""`` | ``"chat"`` | ``"build"`` -- which key is in flight, so the
    #: pane can show "Thinking..." against the right control without asking
    #: ``ctx.busy`` twice for two different-shaped questions. "build" now
    #: spans two task keys in turn (``BUILD_KEY``/``CHAT_KEY``, then
    #: ``LAND_KEY`` -- see :func:`_submit_build_preview`), set again once the
    #: first lands, so this stays honest for the whole time a build is
    #: outstanding rather than dropping to "" while the batch itself runs.
    thinking: str = ""
    #: The last refusal's ``FamiliarRefusal.reason`` (see
    #: ``service.familiar.REASONS``), or ``None`` when the last outcome was
    #: not a refusal at all.
    reason: str | None = None
    #: The last refusal's or Done's own message, shown verbatim -- a lease
    #: refusal keeps its exact sentence (T5 brief), and every other refusal
    #: is already written for a person to read.
    message: str | None = None
    #: A ready Clay preview: the parsed tool calls, the scratch document they
    #: ran against, the diff computed from it, and which tab it previews for.
    #: All four are set together (:func:`_land_build_preview`) and cleared
    #: together (:func:`_clear_preview`) -- a partial set would let Apply run
    #: against a diff that does not describe ``preview_scratch``.
    preview_calls: list[dict] | None = None
    preview_scratch: Any = None
    preview_diff: Any = None
    preview_tab_uid: str = ""
    #: T7: a pending character plan -- ``service.familiar.Answer.action``'s
    #: own ``"character_plan"`` shape, verbatim, or ``None`` with nothing
    #: waiting on a press. Cleared by whichever of Create/Open in
    #: Create/Discard the user presses; landing a *new* plan while one is
    #: already pending simply replaces it -- the same "the document changed,
    #: preview again" spirit ``_staleness_refusal`` keeps, one plan at a time.
    plan: dict[str, Any] | None = None
    #: Vision (2026-09-24): a file path the user typed into the dock's own
    #: attach control, kept as text (not the loaded bytes) so the input line
    #: survives a reply landing mid-type the same way ``input_text`` already
    #: does. Read and cleared by :func:`submit_chat`/:func:`submit_build` on
    #: an accepted submit, never by the frame draw itself.
    attach_path: str = ""
    #: The last attempt to load ``attach_path``'s own refusal sentence (a
    #: missing file, an unreadable one, ...), or ``None`` -- shown beside the
    #: attach control the same way a submit refusal is shown beside Send/
    #: Build, and cleared the moment a fresh path is typed or a load succeeds.
    attach_error: str | None = None

    def on_tab_closed(self, mode: str, uid: str) -> None:
        """:data:`~..docmodes.TAB_CLOSED` listener (registered by
        :func:`install`): drop a pending preview that belonged to the tab
        which just closed.

        The 2026-09-20 audit's familiar-03: ``clay_mode.close_tab``'s own
        ``release`` already clears the GL ghost (``view.clear_preview()``,
        unconditionally, on every close) but never touched this dataclass's
        four preview fields, so the bottom pane kept drawing Apply/Discard
        for a tab that was gone -- and Apply, reading ``ui.preview_tab_uid``
        against a ``state.get`` that now returns ``None``, refused "the
        document changed" forever with no Discard-shaped way out beyond
        guessing. *mode* is accepted (the listener signature every
        ``TAB_CLOSED`` entry shares) but not checked -- ``preview_tab_uid``
        is a bare uid with no mode of its own, the same way
        ``_staleness_refusal`` compares tab uids alone.
        """
        if self.preview_tab_uid == uid:
            _clear_preview(self)


def ensure(ctx: Any) -> FamiliarUIState:
    """The pane's state, built on first use -- ``AppState`` deliberately
    knows nothing about what a mode (or, here, the pane) keeps, the same
    reason ``clay_mode.ensure`` gives for ``ctx.state.clay``."""
    ui = ctx.state.familiar
    if ui is None:
        ui = FamiliarUIState()
        ctx.state.familiar = ui
    return ui


def install(ctx: Any) -> None:
    """Attach ``ctx.familiar_threads`` and register its ``drop``, plus the
    pane's own :meth:`FamiliarUIState.on_tab_closed`, with
    ``docmodes.TAB_CLOSED``, once. Called by the App at startup; a test that
    wants the same wiring calls this too, rather than appending to
    ``TAB_CLOSED`` itself, so "the listener is registered" is always proven
    through the one door that actually registers it in the running app.

    Idempotent: a bound method compares equal to another bound method of the
    same instance and function (``MethodType.__eq__``), so calling this
    twice on the same ``ctx`` does not double-register -- the guard a second
    ``App`` in one process (or a test that calls ``install`` more than once)
    needs. ``ensure(ctx)`` is called here (rather than leaving ``ui.on_tab_
    closed`` to be bound lazily) precisely so that guarantee holds: the same
    ``FamiliarUIState`` instance backs every ``ensure(ctx)`` call afterwards,
    so its bound method keeps comparing equal to itself the same way
    ``ctx.familiar_threads.drop`` already does.
    """
    from ...familiar import threads

    if getattr(ctx, "familiar_threads", None) is None:
        ctx.familiar_threads = threads.Threads()
    from .. import docmodes

    if ctx.familiar_threads.drop not in docmodes.TAB_CLOSED:
        docmodes.TAB_CLOSED.append(ctx.familiar_threads.drop)

    # familiar-03 (2026-09-20 audit): see ``FamiliarUIState.on_tab_closed``'s
    # own docstring -- without this, closing the previewed tab left Apply/
    # Discard drawn (and Apply refusing forever) for a document that no
    # longer existed.
    ui = ensure(ctx)
    if ui.on_tab_closed not in docmodes.TAB_CLOSED:
        docmodes.TAB_CLOSED.append(ui.on_tab_closed)


# --- thread key --------------------------------------------------------------


def _active_tab_uid(ctx: Any) -> str:
    """Clay's active tab uid, or ``""`` in every other mode (or with no Clay
    tab open yet) -- Clay is the only mode with a Build action today, and
    every other mode's chat shares :data:`~.familiar.threads.STUDIO` via
    ``Threads.key_for``'s own falsy-uid rule."""
    mode = str(getattr(ctx.state, "mode", ""))
    if mode != "clay":
        return ""
    clay_state = getattr(ctx.state, "clay", None)
    return getattr(clay_state, "active_uid", "") if clay_state is not None else ""


def thread_key(ctx: Any) -> tuple[str, str]:
    """The ``(mode, tab_uid)`` key this frame's conversation reads/writes --
    see ``threads.Threads.key_for``."""
    from ...familiar import threads

    mode = str(getattr(ctx.state, "mode", ""))
    return threads.Threads.key_for(mode, _active_tab_uid(ctx))


# --- submission ----------------------------------------------------------


def _pending_ghost(ctx: Any, tab_uid: str) -> Any:
    """The scratch document of the preview pending on *tab_uid*, or ``None``.

    A prompt sent while a ghost is showing refines that ghost rather than the
    real document (user, 2026-09-16: follow-ups used to be impossible until
    Apply or Discard), so this is what the scene is read from and what the
    reply's calls later run on top of."""
    ui = ensure(ctx)
    if not tab_uid or ui.preview_calls is None or ui.preview_tab_uid != tab_uid:
        return None
    return ui.preview_scratch


#: The longest side any image handed to Familiar may have, after
#: :func:`normalize_attachment_image` -- the resolution
#: ``familiar.llama_client.IMAGE_TOKEN_COST`` was actually measured at
#: (``dev/measurements/2026-09-24-familiar-mmproj-vram.md``: a 512x512 PNG
#: cost 258 tokens). Qwen3-VL's own vision encoder tiles a larger image into
#: more tokens -- roughly linearly in pixel count, so a 2048px image (16x the
#: pixels of 512px) costs on that order more -- and ``IMAGE_TOKEN_COST`` is a
#: flat ceiling over the 512px measurement, not a formula that scales with
#: whatever a caller hands it. **The orchestrator's 2026-09-24 review, second
#: finding:** without a cap here, an attached photo straight off a phone
#: (2048px+ on its long side) would be undercounted by roughly the same
#: factor it exceeds 512px by -- exactly what INVARIANTS' "a token count that
#: feeds a budget must err high" forbids, silently, only visible once a Clay
#: reply overran the 8,192-token trained window. Capping every image at this
#: module's one door, before it ever reaches ``llama_client``, is what makes
#: the flat constant true rather than merely convenient.
VISION_MAX_SIDE = 512


def normalize_attachment_image(data: bytes) -> bytes:
    """*data*, decoded, converted to RGB and downscaled -- never upscaled --
    so its longer side is at most :data:`VISION_MAX_SIDE`, re-encoded as PNG.

    The one door both image sources go through before either ever reaches
    Familiar: :func:`load_attachment` (a user-typed path) and
    :func:`_ghost_critique_image` (Clay's own render, already 512px, so this
    is a cheap re-encode for that path rather than a resize) -- so
    ``llama_client.IMAGE_TOKEN_COST``'s own measurement at 512px is a
    guarantee about every image Familiar is ever handed, not just the one
    this function's own measurement used.

    Raises ``ValueError`` naming why not when *data* cannot be decoded as an
    image at all -- the same "refuse with a sentence a person can act on"
    shape :func:`load_attachment`'s own missing-file refusal already keeps,
    never a raw ``PIL`` traceback reaching the dock.
    """
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"not a readable image: {exc}") from exc
    img = img.convert("RGB")
    longer = max(img.size)
    if longer > VISION_MAX_SIDE:
        scale = VISION_MAX_SIDE / longer
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(size, Image.LANCZOS)
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def load_attachment(path: str) -> bytes:
    """Read *path* (an image the user typed into the dock's own attach
    input), normalized through :func:`normalize_attachment_image`. -> the
    normalized PNG bytes, or raises ``ValueError`` naming why not -- missing,
    a directory, any other ``OSError`` reading it, or a file that is not a
    readable image -- the same "refuse with a sentence a person can act on"
    shape every other Familiar refusal already keeps, never a raw traceback
    from a bad path or an oversized photo.

    Pure and headless (no imgui, no GL): the file itself is the only thing
    this touches, so it is exactly as testable off a live frame as
    :func:`~..familiar.contract.build_repair_messages` is.
    """
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        raise ValueError(f"{path!r} is not a file")
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read {path!r}: {exc}") from exc
    return normalize_attachment_image(raw)


#: The view/size Clay's own reference render already ships to trellis --
#: ``ClayView.render_png``'s own module docstring names it "the standard
#: three-quarter framing" -- reused verbatim here (2026-09-24, vision) so a
#: revision's critique image is the same picture the user is already looking
#: at in the ghost preview, not a second, different render invented for this.
#: Equal to :data:`VISION_MAX_SIDE` on purpose (see that constant's own
#: docstring): the render is already the size the token-cost measurement
#: used, so :func:`normalize_attachment_image` is a no-op resize for this
#: path, never a downscale.
GHOST_CRITIQUE_SIZE = VISION_MAX_SIDE


def _ghost_critique_image(ctx: Any, ghost: Any) -> bytes | None:
    """*ghost* (a ``ClayDoc``, the pending preview's own scratch document),
    rendered the same way Clay's reference render already does -- lit,
    three-quarter, 512 px -- for a revision request to critique, then through
    :func:`normalize_attachment_image` like every other attached image.
    ``None`` with no ``clay_view`` to render through (a headless test's
    ``ctx``, or a session with no GL context at all), never a refusal: an
    auto-attached critique image is a bonus a text-only prompt already works
    without.
    """
    view = getattr(ctx, "clay_view", None)
    if view is None:
        return None
    render_png = getattr(view, "render_png", None)
    if render_png is None:
        return None
    raw = render_png(ghost, size=GHOST_CRITIQUE_SIZE, view="three_quarter", shading="lit")
    return normalize_attachment_image(raw)


def _resolve_attach_image(ctx: Any, ui: FamiliarUIState, refine: Any) -> tuple[bytes | None, bool]:
    """What image (if any) this submit should attach. -> ``(image, ok)`` --
    ``ok`` is ``False`` only when the user explicitly typed a path
    (``ui.attach_path``) that failed to load, in which case *this* submit
    must be refused rather than silently sent with no image at all
    (``ui.attach_error`` is set for the dock to show beside the control).

    An explicit attach always wins over the ghost critique: a user who typed
    a path meant that picture, not the ghost's own render. With nothing
    explicit and *refine* not ``None`` (a revision of a pending ghost --
    the follow-up shape :func:`_pending_ghost` already recognises), Clay's
    own ghost render is attached automatically (:func:`_ghost_critique_image`)
    so the model can see what it is being asked to revise, not just read the
    compacted scene JSON.
    """
    path = ui.attach_path.strip()
    if path:
        try:
            image = load_attachment(path)
        except ValueError as exc:
            ui.attach_error = str(exc)
            return None, False
        ui.attach_error = None
        return image, True
    if refine is not None:
        return _ghost_critique_image(ctx, refine), True
    return None, True


def _capture_scene(ctx: Any, tab_uid: str) -> dict[str, Any] | None:
    """The compact scene for *tab_uid*, or ``None`` with no tab open.

    Read on the frame thread, exactly like ``submit_build`` always did --
    ``agent_clay.call`` walks the live ``ClayDoc``, and that read must
    happen against *this* frame's document, not whatever it is by the time
    a worker thread gets around to it. Factored out (T6) so
    :func:`submit_chat`'s routed path captures the same scene a router
    decision of "build" needs, without a second copy of this read.
    """
    if not tab_uid:
        return None
    from ...familiar import contract
    from ..modes.clay.agent import dispatch as agent_clay

    ghost = _pending_ghost(ctx, tab_uid)
    if ghost is not None:
        from . import preview as familiar_preview

        # Read from a clone of the ghost, so a refinement is written against
        # what the user is looking at and the ghost itself is never touched.
        scratch_ctx = familiar_preview.build(ghost)
        ctx, tab_uid = scratch_ctx, scratch_ctx.tab_uid
    session = agent_clay.Session(tab_uid=tab_uid)
    scene_result = agent_clay.call(ctx, session, "clay_scene", {})
    structured = (
        scene_result.get("structuredContent") if isinstance(scene_result, dict) else None
    )
    return contract.compact_scene(structured or {})


def _character_options(ctx: Any) -> dict[str, Any]:
    """The plan-shaped slice of ``service.characters.character_options`` for
    :func:`~.familiar.character_plan.build_character_messages`/
    ``character_schema`` -- species with their themes, every shipped
    movement across every archetype's own skeleton, the direction ladder and
    the custom-size range.

    Read through ``character_engine.options`` -- Create's own frame-thread
    cache, keyed on the palette directory's stamp
    (``modes/create/engine/character.py``'s own docstring) -- rather than
    calling ``service.characters.character_options`` fresh: it is the one
    place already paying for this read every frame Create's own form is
    open, and a second, uncached copy here would answer the same registries
    a frame later for no reason. Computed on the frame thread, alongside
    *destinations*/*asset_types* (T8's own precedent) rather than inside the
    worker closure below: unlike those two this touches no ``ctx.state``
    gate, but it does touch ``ctx.state.preview``'s own cache slot, which is
    frame-thread state exactly like the Clay scene capture is.
    """
    from ...kernels.rig import cliplib
    from ..modes.create.engine import character as character_engine

    raw = character_engine.options(ctx)
    families = [
        {"key": f["key"], "label": f["label"], "themes": [t["key"] for t in f["themes"]]}
        for f in raw["families"]
    ]
    templates = {a["template"] for a in raw["archetypes"]}
    movements = sorted({name for t in templates for name in cliplib.shipped_clip_names(t)})
    return {
        "families": families,
        "movements": movements,
        "directions": list(raw["directions"]),
        "size_range": tuple(raw["troupe"]["logical_size_range"]),
    }


def submit_chat(ctx: Any, prompt: str) -> bool:
    """Send *prompt* through Familiar's router. -> whether it was accepted.

    Refused (returns ``False``, no toast -- the disabled Send button already
    said why) when *prompt* is blank or a chat is already in flight. Appends
    the user's own turn to the thread immediately, before the network call
    even starts, so it appears in the transcript the same frame it was sent
    rather than only once a reply lands.

    T6: this now submits ``service.familiar.ask`` (route, then answer)
    rather than ``chat_reply`` directly -- Send no longer means "always
    plain chat"; the router decides. In Clay, with a tab open, the scene is
    captured up front (:func:`_capture_scene`) the same way :func:`submit_build`
    always has, so a message the router sends to ``clay_build``/``clay_edit``
    has a scene to build against without a second frame-thread round trip
    once the route comes back. The explicit **Build** button stays wired to
    :func:`submit_build` directly -- a press that already means "build"
    should not pay for a routing decision only to be told what it already
    knew.
    """
    prompt = prompt.strip()
    if not prompt or ctx.busy(CHAT_KEY):
        return False
    from ...familiar import threads

    key = thread_key(ctx)
    ctx.familiar_threads.append(key, threads.Turn("user", prompt))
    history = ctx.familiar_threads.get(key)[:-1]

    mode = str(getattr(ctx.state, "mode", ""))
    tab_uid = _active_tab_uid(ctx)
    scene = _capture_scene(ctx, tab_uid) if mode == "clay" else None

    # T8: computed here, on the frame thread, for ``_capture_scene``'s own
    # reason -- ``familiar_doors.destinations`` reads ``palette.commands``,
    # which reads ``ctx.state`` (the mode gate, the active document), so the
    # list the router is offered has to describe *this* frame, not whatever
    # it is by the time a worker thread gets around to it.
    from ..modes.create.engine import assets as create_assets
    from . import doors as familiar_doors

    destinations = familiar_doors.destinations(ctx)
    asset_types = create_assets.ASSET_TYPE_OPTIONS
    # T7: same frame-thread treatment as the two lines above -- see
    # ``_character_options``'s own docstring for why this one is cheap
    # rather than studio-gated.
    character_options = _character_options(ctx)

    from ...service import familiar as svc_familiar
    from ...service import familiar_log

    # T5's dev-only log (REALMSPINNER_FAMILIAR_LOG): minted here, on the frame
    # thread, and carried into ``run()``'s closure so every record the
    # request/outcome pair produces on the worker thread groups back to the
    # same round trip this submit accepted.
    exchange_id = familiar_log.new_exchange_id()
    refine = _pending_ghost(ctx, tab_uid)

    ui = ensure(ctx)
    image, image_ok = _resolve_attach_image(ctx, ui, refine if mode == "clay" else None)
    if not image_ok:
        return False

    def run() -> Any:
        with familiar_log.exchange(exchange_id):
            return svc_familiar.ask(
                ctx.svc,
                prompt,
                mode=mode,
                history=history,
                scene=scene,
                destinations=tuple(destinations),
                asset_types=asset_types,
                character_options=character_options,
                image=image,
            )

    tag = {
        "thread_key": key,
        "tab_uid": tab_uid,
        "scene_captured": scene is not None,
        "refine": refine,
        "exchange": exchange_id,
        # Self-repair: if the router sends this to Clay and the scratch run
        # later refuses (a gate ``ask``/``clay_build`` cannot see -- see
        # ``preview.py``'s own docstring), ``_submit_build_preview`` needs
        # *this* prompt/scene again to build a follow-up turn
        # (``contract.build_repair_messages``) -- neither was carried in the
        # tag before self-repair existed, since nothing downstream of a
        # landed build ever needed them again.
        "prompt": prompt,
        "scene": scene,
    }
    if not ctx.submit(CHAT_KEY, run, tag=tag):
        return False
    if familiar_log.enabled():
        with familiar_log.exchange(exchange_id):
            familiar_log.record(
                "submit",
                submit_kind="chat",
                prompt=prompt,
                mode=mode,
                tab_uid=tab_uid,
                scene=scene,
                refine=refine is not None,
            )
    ui.thinking = "chat"
    ui.reason = None
    ui.message = None
    ui.attach_path = ""
    return True


def submit_build(ctx: Any, prompt: str) -> bool:
    """Send *prompt* as a Clay build request. -> whether it was accepted.

    Refused, with no request ever sent, outside Clay or with no tab open --
    a Build button drawn only in Clay, against the active tab, should never
    reach this with either untrue, but the check is repeated here rather
    than trusted to the caller because this is also the door a test drives
    directly. Router-free by design (T6 brief): this is the explicit,
    always-build action, unlike Send's routed path in :func:`submit_chat`.
    """
    prompt = prompt.strip()
    tab_uid = _active_tab_uid(ctx)
    if not prompt or not tab_uid or ctx.busy(BUILD_KEY):
        return False
    scene = _capture_scene(ctx, tab_uid)

    from ...familiar import threads

    key = thread_key(ctx)
    ctx.familiar_threads.append(key, threads.Turn("user", prompt))

    from ...service import familiar as svc_familiar
    from ...service import familiar_log

    exchange_id = familiar_log.new_exchange_id()
    refine = _pending_ghost(ctx, tab_uid)

    ui = ensure(ctx)
    image, image_ok = _resolve_attach_image(ctx, ui, refine)
    if not image_ok:
        return False

    def run() -> list[dict]:
        with familiar_log.exchange(exchange_id):
            return svc_familiar.clay_build(ctx.svc, prompt, scene, image=image)

    tag = {
        "thread_key": key,
        "tab_uid": tab_uid,
        "refine": refine,
        "exchange": exchange_id,
        # Self-repair: see the identical comment on ``submit_chat``'s own tag.
        "prompt": prompt,
        "scene": scene,
    }
    if not ctx.submit(BUILD_KEY, run, tag=tag):
        return False
    if familiar_log.enabled():
        with familiar_log.exchange(exchange_id):
            familiar_log.record(
                "submit",
                submit_kind="build",
                prompt=prompt,
                mode="clay",
                tab_uid=tab_uid,
                scene=scene,
                refine=refine is not None,
            )
    ui.attach_path = ""
    ui.thinking = "build"
    ui.reason = None
    ui.message = None
    return True


# --- landing ---------------------------------------------------------------


def _reason_and_message(done: Any) -> tuple[str | None, str]:
    error = done.error
    reason = getattr(error, "reason", None) if error is not None else None
    message = done.message or (str(error) if error is not None else "Something went wrong.")
    return reason, message


def _say(ctx: Any, thread_key: Any, text: str, *, toast: bool = True) -> None:
    """Append a Familiar turn to *thread_key*'s transcript and, unless
    *toast* is false, toast it too.

    The "done" confirmation (user, 2026-09-16: "after the assistant
    considers itself done, it needs to send a confirmation to the user that
    it is done with its job" -- transcript turn plus toast, the user's own
    choice): before this, a build preview landing or a refusal only ever set
    ``ui.message``/the ghost silently, so with the pane collapsed there was
    nothing to see. Tolerates a missing thread/``ctx.familiar_threads``
    exactly like the CHARACTER_KEY branch this was factored out of always
    did -- a caller with no thread key (no document tab, an early return
    before one was captured) still gets its toast.
    """
    if not text:
        # A refusal with no sentence (``familiar_preview.apply`` answering
        # ``ok: False`` without a ``message``) must not become a ``None``
        # turn: ``draw_expanded`` concatenates the prefix onto it.
        return
    threads_obj = getattr(ctx, "familiar_threads", None)
    if thread_key is not None and threads_obj is not None:
        from ...familiar import threads

        threads_obj.append(thread_key, threads.Turn("familiar", text))
    if toast:
        toast_fn = getattr(ctx, "toast", None)
        if toast_fn is not None:
            toast_fn(text)


def _log_outcome(done: Any, tag: dict) -> None:
    """Dev-only (REALMSPINNER_FAMILIAR_LOG): one ``outcome`` record per landed
    task, carrying the same exchange id its ``submit``/``request`` records
    used -- see ``familiar_log.py``'s own docstring."""
    from ...service import familiar_log

    if not familiar_log.enabled():
        return
    from ...service.familiar import Answer

    result = done.result
    skill = text = calls = action = None
    if isinstance(result, Answer):
        skill, text, calls, action = result.skill, result.text, result.calls, result.action
    elif isinstance(result, list):
        calls = result
    reason = getattr(done.error, "reason", None) if done.error is not None else None
    with familiar_log.exchange(tag.get("exchange")):
        familiar_log.record(
            "outcome",
            key=done.key,
            ok=done.ok,
            skill=skill,
            text=text,
            calls=calls,
            action=action,
            reason=reason,
            message=None if done.ok else done.message,
        )


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from the app for :data:`CHAT_KEY`/:data:`BUILD_KEY`/
    :data:`CHARACTER_KEY`, the same way every other mode's ``on_task_done``
    is called from ``main._on_task_done`` (any ``"familiar/"``-prefixed key
    reaches here)."""
    ui = ensure(ctx)
    tag = done.tag if isinstance(done.tag, dict) else {}
    _log_outcome(done, tag)

    if done.key == CHAT_KEY:
        ui.thinking = ""
        if done.ok:
            ui.reason = None
            ui.message = None
            result = done.result
            from ...service.familiar import Answer

            if isinstance(result, Answer) and result.calls is not None:
                # The router sent this one to Clay -- land it exactly like
                # an explicit Build's own result, calls and all.
                for refusal_sentence in result.retries:
                    _say(
                        ctx,
                        tag.get("thread_key"),
                        f"Retrying after a refusal: {refusal_sentence}",
                    )
                _submit_build_preview(
                    ctx,
                    ui,
                    tag.get("tab_uid", ""),
                    result.calls,
                    refine=tag.get("refine"),
                    thread_key=tag.get("thread_key"),
                    exchange=tag.get("exchange"),
                    prompt=tag.get("prompt"),
                    scene=tag.get("scene"),
                    reply=result.reply,
                    repairs_used=result.repairs_used,
                )
                return
            if (
                isinstance(result, Answer)
                and result.action is not None
                and result.action.get("kind") == "character_plan"
            ):
                # T7: a character plan waits for a press -- it is never acted
                # out the way a navigate/draft is (see ``Answer.action``'s
                # own docstring), just shown, so the plan card can draw
                # itself from ``ui.plan`` instead of the transcript.
                ui.plan = result.action
                text = "Here's a plan -- create it, open it in Create, or discard."
                citations = ()
            elif isinstance(result, Answer) and result.action is not None:
                # T8: a navigate/create route -- act it out here, on the
                # frame thread (``familiar_doors`` reaches the palette,
                # ``state.set_mode`` and Create's form, none of which a
                # worker thread may touch), then say what happened (or why
                # not) in the thread the same way every other reply does.
                text = _run_door(ctx, result.action)
                citations = ()
            elif isinstance(result, Answer):
                text, citations = result.text, result.citations
            else:
                # Defensive, not exercised by a real ``ask`` call: a bare
                # string result is still treated as an uncited chat reply
                # rather than crashing on ``.text``.
                text, citations = (result if isinstance(result, str) else str(result)), ()
            thread = tag.get("thread_key")
            threads_obj = getattr(ctx, "familiar_threads", None)
            if thread is not None and threads_obj is not None and text is not None:
                from ...familiar import threads

                threads_obj.append(thread, threads.Turn("familiar", text, citations))
        else:
            # A failed CHAT_KEY task is also where a routed build's own
            # refusal lands (``ask`` calls ``clay_build`` directly for a
            # ``clay_build``/``clay_edit`` route, so its ``FamiliarRefusal``
            # fails the whole task rather than coming back as an ``Answer``)
            # -- said here, once, for every CHAT_KEY failure rather than
            # only the build-routed ones, since there is nothing in ``done``
            # that tells the two apart.
            ui.reason, ui.message = _reason_and_message(done)
            _say(ctx, tag.get("thread_key"), ui.message)
        return

    if done.key == BUILD_KEY:
        ui.thinking = ""
        if not done.ok:
            ui.reason, ui.message = _reason_and_message(done)
            _say(ctx, tag.get("thread_key"), ui.message)
            if tag.get("refine") is None:
                _clear_preview(ui)
            return
        from ...service.familiar import ClayBuildResult

        build_result = done.result
        if isinstance(build_result, ClayBuildResult):
            calls, reply, repairs_used, retries = (
                build_result.calls,
                build_result.reply,
                build_result.repairs_used,
                build_result.retries,
            )
        else:
            # Back-compat with a hand-built ``Done(result=<plain list>)`` --
            # every test written before self-repair existed, and any other
            # caller that never runs the real ``clay_build`` door.
            calls = build_result if isinstance(build_result, list) else []
            reply, repairs_used, retries = None, 0, ()
        ui.reason = None
        ui.message = None
        for refusal_sentence in retries:
            _say(ctx, tag.get("thread_key"), f"Retrying after a refusal: {refusal_sentence}")
        _submit_build_preview(
            ctx,
            ui,
            tag.get("tab_uid", ""),
            calls,
            refine=tag.get("refine"),
            thread_key=tag.get("thread_key"),
            exchange=tag.get("exchange"),
            prompt=tag.get("prompt"),
            scene=tag.get("scene"),
            reply=reply,
            repairs_used=repairs_used,
        )
        return

    if done.key == LAND_KEY:
        # Phase two: the worker's ``clay_batch`` run has landed -- see
        # :func:`_land_build_preview`'s own docstring for why every
        # staleness fact is asked again here rather than trusted from the
        # submit that started it.
        _land_build_preview(ctx, ui, done)
        return

    if done.key == CHARACTER_KEY:
        if not done.ok:
            # A plain ``service.errors.Invalid`` (Blender missing, a stale
            # theme) has no ``.reason`` -- ``_reason_and_message`` already
            # answers ``None`` for that, the same shape a non-Familiar
            # refusal is shown in everywhere else.
            #
            # The 2026-09-18 audit (familiar-03): this branch set
            # ``ui.message`` but never called ``_say``, unlike every other
            # CHAT_KEY/BUILD_KEY failure branch -- with the pane collapsed a
            # failed character creation left no transcript turn and no toast
            # at all, the one Familiar exit that said nothing.
            ui.reason, ui.message = _reason_and_message(done)
            _say(ctx, tag.get("thread_key"), ui.message)
            return
        ui.reason = None
        ui.message = None
        _say(ctx, tag.get("thread_key"), "Character queued -- it will appear in the Library.")
        return


def _run_door(ctx: Any, action: dict[str, Any]) -> str:
    """T8: act out a routed ``navigate``/``create`` decision. -> the
    sentence the transcript should show.

    A shape neither of :func:`familiar_doors.navigate`/:func:`draft_in_create`
    itself refuses (an ``action`` this build does not recognise -- there is
    none today, but a future skill's own action kind must not crash the
    frame loop reading a reply that landed) answers plainly rather than
    raising."""
    from . import doors as familiar_doors

    kind = action.get("kind")
    if kind == "navigate":
        return familiar_doors.navigate(ctx, str(action.get("target") or ""))
    if kind == "draft":
        return familiar_doors.draft_in_create(
            ctx, str(action.get("asset_type") or ""), str(action.get("prompt") or "")
        )
    return "I'm not sure what to do with that."


def _refusal_sentence(result: dict) -> str:
    """The door's own sentence for a refused ``clay_batch``. A batch's text is
    its whole structured payload dumped as JSON, so the readable sentence is
    the one on the entry named by ``stopped_at``; a refusal of the batch
    itself (a malformed entry, say) has no such entry and keeps its text."""
    structured = result.get("structuredContent") or {}
    stopped_at = structured.get("stopped_at")
    results = structured.get("results") or []
    if isinstance(stopped_at, int) and 0 <= stopped_at < len(results):
        result = results[stopped_at]
    content = result.get("content") or []
    if content and isinstance(content[0], dict) and content[0].get("text"):
        return str(content[0]["text"])
    return "Familiar's build could not be previewed."


def _clear_preview(ui: FamiliarUIState) -> None:
    ui.preview_calls = None
    ui.preview_scratch = None
    ui.preview_diff = None
    ui.preview_tab_uid = ""


def _preview_sentence(diff: Any) -> str:
    """The "done" sentence a clean preview lands with (2026-09-16 brief: a
    transcript turn naming what the preview does, not just the silent
    ghost). Reads :class:`~.clay.scratch.PreviewDiff` -- ``added``/
    ``removed`` and ``changed`` minus ``added`` (an added object is not also
    counted as "changed") -- singular/plural per clause, and a clause is
    dropped rather than read as "adds 0 objects" when its count is zero.
    Removals are named too, not just additions and changes."""
    if diff.empty:
        return "Done, but the build changed nothing."
    added = len(diff.added)
    removed = len(diff.removed)
    changed = len(diff.changed - diff.added)
    clauses = []
    if added:
        clauses.append(f"adds {added} object{'s' if added != 1 else ''}")
    if removed:
        clauses.append(f"removes {removed} object{'s' if removed != 1 else ''}")
    if changed:
        clauses.append(f"changes {changed}")
    body = " and ".join(clauses) if clauses else "changes the document"
    return f"Done: the preview {body}. Apply to keep it or Discard to drop it."


def _log_preview(exchange_id: Any, *, diff: Any = None, refusal: str | None = None) -> None:
    """Dev-only (REALMSPINNER_FAMILIAR_LOG): one ``preview`` record per landed
    build -- diff counts on a clean preview, the refusal sentence
    otherwise."""
    from ...service import familiar_log

    if not familiar_log.enabled():
        return
    fields: dict[str, Any] = {}
    if refusal is not None:
        fields["refusal"] = refusal
    if diff is not None:
        fields.update(added=len(diff.added), removed=len(diff.removed), changed=len(diff.changed))
    with familiar_log.exchange(exchange_id):
        familiar_log.record("preview", **fields)


def _staleness_refusal(
    state: Any,
    tab: Any,
    tab_uid: str,
    ui: FamiliarUIState,
    refine: Any,
    base_doc_id: int | None = None,
    base_head: int | None = None,
) -> str | None:
    """The facts a build preview's landing depends on, both re-asked by
    :func:`_land_build_preview` after the worker returns as well as checked
    here by :func:`_submit_build_preview` before it ever submits -- the
    2026-09-17 audit (familiar-01) split what used to be one inline check
    into two call sites once the batch itself moved to a worker, since the
    tab can close, the user can switch tabs, or the ghost being refined can
    be applied/discarded during the time the batch spends off the frame
    thread, none of which the pre-submit check alone could still see by the
    time the result lands.

    *base_doc_id*/*base_head* -- ``id(tab.doc)``/``tab.doc.history.head`` as
    :func:`_submit_build_preview` snapshotted them at clone time, passed only
    by :func:`_land_build_preview` (the pre-submit call has no snapshot yet,
    since the clone has not been taken). The 2026-09-20 audit's familiar-01:
    without this, an edit made *while the batch was running on the worker*
    was invisible here -- ``_land_build_preview`` used to stamp
    ``clay_scratch.diff``'s own ``base_head``/``base_doc_id`` from ``tab.doc``
    as it stood *after* the edit, so a concurrently added object landed in
    ``diff.removed`` and ``apply``'s own staleness check (``preview.py``'s
    ``_refusal``) agreed with a baseline that was never the one the clone was
    actually taken from -- Apply silently deleted the user's edit instead of
    refusing. Checking identity and head *here*, against the phase-one
    snapshot, closes that window before the diff is ever computed.
    """
    if tab is None:
        # The tab this build was requested against closed while Familiar
        # was thinking -- the same "preview again" sentence every other
        # familiar_preview refusal uses, for the same reason: there is
        # nothing left to preview against.
        return "the document changed -- preview again"
    if tab_uid != state.active_uid:
        # The 2026-09-15 audit's agents-01: this used to check only that the
        # tab still existed, not that it was still the one on screen. ``set_
        # preview``/``_ghost_draws`` carry no document identity of their own
        # on the shared ``clay_view``, so a build that landed after a tab
        # switch painted the ghost over whichever tab was now in front --
        # Apply already refused a stale base at that point, but the display
        # never did. Landing now refuses the same way Apply always has,
        # rather than showing a ghost for a document nobody is looking at.
        # The 2026-09-23 audit (familiar-01): this branch used to fall
        # through to the generic "the document changed" sentence above, even
        # though nothing about the document changed -- only which tab is on
        # screen. ``preview.py``'s ``apply()`` refuses the identical
        # situation (tab still open, just not the active one) with its own,
        # more specific sentence; this now matches it instead of telling the
        # user to preview again against a document that never moved.
        return "that document is not the one in front -- switch to it, then preview again"
    if refine is not None and (ui.preview_scratch is not refine or ui.preview_tab_uid != tab_uid):
        # Applied or discarded while the model was thinking: these calls were
        # written against a ghost that is gone, and on the real document they
        # would address objects that are not there.
        return "the preview changed -- preview again"
    if base_doc_id is not None and (
        id(tab.doc) != base_doc_id or tab.doc.history.head != base_head
    ):
        # The 2026-09-20 audit's familiar-01 (see the docstring above): the
        # real document moved -- a new object, an undo, a redo, anything
        # that bumps ``history.head`` or swaps the document outright -- in
        # the window between the clone and this landing, so the batch that
        # is about to be diffed was run against a base that no longer
        # describes what is on screen.
        return "the document changed -- preview again"
    return None


def _submit_build_preview(
    ctx: Any,
    ui: FamiliarUIState,
    tab_uid: str,
    calls: list[dict],
    *,
    refine: Any = None,
    thread_key: Any = None,
    exchange: Any = None,
    prompt: str | None = None,
    scene: dict[str, Any] | None = None,
    reply: str | None = None,
    repairs_used: int = 0,
) -> None:
    """Phase one of landing a build: the staleness checks and the scratch
    clone -- both cheap, both fine on the frame thread -- then hand the
    batch itself to a worker under :data:`LAND_KEY`.

    The 2026-09-17 audit (familiar-01): this function used to also run the
    batch (``familiar_preview.run_scratch(..., "clay_batch", ...)``) right
    here, inline on the frame thread -- up to ``agent_clay.BATCH_MAX`` calls,
    booleans included, 624-705 ms wall for 16 uv-sphere adds and 15 unions
    with no GPU or weights involved, freezing ``App.frame`` for ~40 frames on
    an ordinary Build. Now this only clones the base document (a numpy copy,
    not a batch of ops -- see ``clay.scratch.clone``) and submits the batch;
    :func:`_land_build_preview` is the second half, called back from
    :func:`on_task_done` once the worker is done, where the diff is taken and
    the ghost is shown.

    *thread_key* -- the same ``(mode, tab_uid)`` the request was submitted
    under -- is where the "done" sentence (:func:`_preview_sentence`, or a
    refusal) lands as a Familiar turn, via :func:`_say`; *exchange* is the
    dev-log id the same round trip's ``submit``/``request`` records used.
    Both are carried into :data:`LAND_KEY`'s own tag so the second phase can
    use them too.

    **Self-repair, past the card/parse/vocabulary gates.** *prompt*/*scene*/
    *reply*/*repairs_used* -- all optional, all defaulted to "nothing to
    repair with" -- are what :func:`clay_build`/:func:`~.service.familiar.ask`
    already gated *calls* with. If the scratch run refuses (a gate neither of
    those can see: it only runs once *calls* is tried against a real document
    clone), the worker closure itself resends one follow-up turn via
    ``service.familiar.clay_repair`` and re-runs the scratch gate on the
    corrected reply -- up to ``service.familiar.MAX_REPAIRS`` total across
    *both* refusal sources, since *repairs_used* already carries whatever the
    card/parse/vocabulary gate spent. All of this runs on the :data:`LAND_KEY`
    worker, never the frame thread -- the same reason the scratch run itself
    already moved there (the 2026-09-17 audit, familiar-01). With *prompt*
    ``None`` (every pre-self-repair caller, and every test that hand-builds a
    ``BUILD_KEY``/``CHAT_KEY`` result), there is nothing to build a follow-up
    turn from, so a scratch-run refusal is shown exactly as it always was.
    """
    from ..modes.clay import mode as clay_mode
    from . import preview as familiar_preview

    state = clay_mode.ensure(ctx)
    tab = state.get(tab_uid) if tab_uid else None
    refusal = _staleness_refusal(state, tab, tab_uid, ui, refine)
    if refusal is not None:
        ui.message = refusal
        ui.reason = None
        _log_preview(exchange, refusal=refusal)
        _say(ctx, thread_key, refusal)
        return

    # Snapshotted *now*, before the clone -- not read back from ``tab.doc``
    # once the worker returns. The 2026-09-20 audit's familiar-01 (see
    # ``_staleness_refusal``'s own docstring): the batch runs off the frame
    # thread between here and ``_land_build_preview``, and the real document
    # is free to move in that window -- these two facts are what "moved"
    # means, carried in :data:`LAND_KEY`'s own tag so the second phase can
    # refuse against the state that was actually cloned, not whatever
    # ``tab.doc`` happens to be by the time it looks.
    base_doc_id = id(tab.doc)
    base_head = tab.doc.history.head
    base_for_clone = refine if refine is not None else tab.doc
    scratch_ctx = familiar_preview.build(base_for_clone)
    # A second, independent clone -- read off the live document here, on the
    # frame thread, for the exact same reason ``scratch_ctx`` itself is: a
    # self-repair retry needs a *fresh* scratch clone (the first one may have
    # been partway mutated by a batch that failed midway through), and taking
    # that clone from ``tab.doc`` inside the worker closure below would race
    # the frame thread's own edits to it. Cloning ``pristine_doc`` itself
    # (never touched again after this line, and never shared with anything
    # else) inside the worker is safe, because nothing outside this closure
    # can still be mutating it. ``None`` when there is no *prompt* to retry
    # with (see this function's own docstring) -- no repair can ever run, so
    # there is nothing worth the extra clone.
    from ...kernels.mesh import scratch as clay_scratch

    pristine_doc = clay_scratch.clone(base_for_clone) if prompt is not None else None

    # One clay_batch, never call by call: the model names objects made earlier
    # in the same reply as {"$ref": "<name>"}, which only a batch resolves --
    # it is the shape the training data, the eval and the door all share. Run
    # one at a time, the first $ref was refused ("Build the Eiffel Tower" came
    # back as clay_boolean's "uids must be a list of integers.", 2026-09-16).
    # This closure touches only ``scratch_ctx``/``pristine_doc`` -- private
    # clones, never ``ctx.state``, GL or imgui -- and ``ctx.svc`` (for a
    # repair's own model round trip, exactly like every other Familiar
    # closure already does), which is what makes running it off the frame
    # thread safe.
    def run() -> dict:
        from ...service import familiar as svc_familiar

        calls_now, reply_now, repairs_now = calls, reply, repairs_used
        working_ctx = scratch_ctx
        retries: list[str] = []
        result = familiar_preview.run_scratch(working_ctx, "clay_batch", {"calls": calls_now})
        while (
            isinstance(result, dict)
            and result.get("isError")
            and pristine_doc is not None
            and repairs_now < svc_familiar.MAX_REPAIRS
        ):
            refusal_sentence = _refusal_sentence(result)
            retries.append(refusal_sentence)
            build_result = svc_familiar.clay_repair(
                ctx.svc, prompt, scene, reply_now, refusal_sentence, repairs_now
            )
            calls_now, reply_now, repairs_now = (
                build_result.calls,
                build_result.reply,
                build_result.repairs_used,
            )
            working_ctx = familiar_preview.build(pristine_doc)
            result = familiar_preview.run_scratch(
                working_ctx, "clay_batch", {"calls": calls_now}
            )
        return {
            "result": result,
            "calls": calls_now,
            "scratch_ctx": working_ctx,
            "retries": retries,
        }

    tag = {
        "thread_key": thread_key,
        "exchange": exchange,
        "tab_uid": tab_uid,
        "refine": refine,
        "calls": calls,
        "scratch_ctx": scratch_ctx,
        "base_doc_id": base_doc_id,
        "base_head": base_head,
    }
    if not ctx.submit(LAND_KEY, run, tag=tag):
        # A second build already landing on this same key -- CHAT_KEY and
        # BUILD_KEY are independent submits, so (rarely) both can land in the
        # same frame. Dropped rather than queued: the disabled Send/Build
        # button is the real guard against this in the ordinary case, and
        # ``ui.thinking`` is left as this branch found it (``on_task_done``
        # already cleared it to "" before calling here), so nothing is stuck
        # "thinking" over a build that was simply never submitted.
        #
        # familiar-03 (2026-09-18 audit, second run): this used to be a bare
        # `return` -- with the pane collapsed there was no transcript turn
        # and no toast, unlike every other way a Familiar build ends (see
        # `_say`'s own docstring: "Every way a Familiar build ends says so
        # in the transcript"). One line via `_say`, same as the staleness
        # refusal just above in this same function.
        _say(ctx, thread_key, "Another build is still landing -- try again in a moment.")
        return
    ui.thinking = "build"


def _land_build_preview(ctx: Any, ui: FamiliarUIState, done: Any) -> None:
    """Phase two of landing a build (see :func:`_submit_build_preview`),
    called from :func:`on_task_done` for :data:`LAND_KEY`: the worker's
    ``clay_batch`` run has landed, so this takes the diff and shows the
    ghost -- or refuses, re-asking :func:`_staleness_refusal` the same
    question the submit side already asked, since the tab or the ghost being
    refined can have moved again while the batch ran."""
    from ...kernels.mesh import scratch as clay_scratch
    from ..modes.clay import mode as clay_mode

    ui.thinking = ""
    tag = done.tag if isinstance(done.tag, dict) else {}
    thread_key, exchange = tag.get("thread_key"), tag.get("exchange")
    tab_uid, refine = tag.get("tab_uid", ""), tag.get("refine")
    scratch_ctx, calls = tag.get("scratch_ctx"), tag.get("calls") or []
    base_doc_id, base_head = tag.get("base_doc_id"), tag.get("base_head")

    state = clay_mode.ensure(ctx)
    tab = state.get(tab_uid) if tab_uid else None
    # The 2026-09-20 audit's familiar-01: this must run, with the phase-one
    # snapshot, *before* the diff below is computed -- see
    # ``_staleness_refusal``'s own docstring for why a check made only after
    # the diff already exists is too late (the diff's own ``base_head`` would
    # already have been stamped from the moved document).
    refusal = _staleness_refusal(state, tab, tab_uid, ui, refine, base_doc_id, base_head)
    if refusal is not None:
        ui.message = refusal
        ui.reason = None
        _log_preview(exchange, refusal=refusal)
        _say(ctx, thread_key, refusal)
        return

    if not done.ok:
        # ``run_scratch``/``agent_clay.call`` do not normally raise -- a
        # refusal comes back as ``isError`` in the result, handled below --
        # but a defensive path all the same, the same shape a raised
        # BUILD_KEY/CHAT_KEY task already lands with.
        ui.message = done.message or "Something went wrong."
        ui.reason = None
        _log_preview(exchange, refusal=ui.message)
        _say(ctx, thread_key, ui.message)
        return

    payload = done.result if isinstance(done.result, dict) else {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    # ``calls``/``scratch_ctx`` may have been replaced by self-repair (a
    # corrected reply's own calls, run against a fresh clone) -- the
    # worker's own payload names the ones the *landed* result actually
    # describes, falling back to what phase one submitted with for a
    # defensive ``done.result`` that is not this shape at all.
    calls = payload.get("calls") if isinstance(payload.get("calls"), list) else calls
    scratch_ctx = payload.get("scratch_ctx", scratch_ctx)
    for refusal_sentence in payload.get("retries") or ():
        _say(ctx, thread_key, f"Retrying after a refusal: {refusal_sentence}")
    if result.get("isError"):
        ui.message = _refusal_sentence(result)
        ui.reason = "parse"
        _log_preview(exchange, refusal=ui.message)
        _say(ctx, thread_key, ui.message)
        return

    base_calls = list(ui.preview_calls or []) if refine is not None else []
    scratch_doc = scratch_ctx.state.clay.get(scratch_ctx.tab_uid).doc
    diff = clay_scratch.diff(tab.doc, scratch_doc)
    ui.preview_calls = base_calls + list(calls)
    ui.preview_scratch = scratch_doc
    ui.preview_diff = diff
    ui.preview_tab_uid = tab_uid
    view = getattr(ctx, "clay_view", None)
    if view is not None:
        view.set_preview(diff, scratch_doc)
    _log_preview(exchange, diff=diff)
    _say(ctx, thread_key, _preview_sentence(diff))


def apply_preview(ctx: Any) -> None:
    """Apply the pending preview, if any -- see ``familiar_preview.apply``'s
    own docstring for the refusals it can still return (the document moved
    since the preview was computed)."""
    ui = ensure(ctx)
    if ui.preview_calls is None:
        return
    from ...service import familiar_log
    from . import preview as familiar_preview

    result = familiar_preview.apply(ctx, ui.preview_tab_uid, ui.preview_diff, ui.preview_scratch)
    key = thread_key(ctx)
    if familiar_log.enabled():
        familiar_log.record("apply", ok=bool(result.get("ok")), message=result.get("message"))
    if not result.get("ok"):
        ui.message = result.get("message")
        ui.reason = None
        _say(ctx, key, ui.message)
        return
    ui.message = None
    ui.reason = None
    _clear_preview(ui)
    _say(ctx, key, "Applied to the scene.")


def discard_preview(ctx: Any) -> None:
    """Drop the pending preview, if any, leaving the document untouched."""
    ui = ensure(ctx)
    if ui.preview_calls is None:
        return
    from ...service import familiar_log
    from . import preview as familiar_preview

    familiar_preview.discard(ctx, ui.preview_tab_uid)
    if familiar_log.enabled():
        familiar_log.record("discard")
    key = thread_key(ctx)
    _clear_preview(ui)
    # No toast (unlike Apply): discarding is a quiet "never mind", not a
    # result to be told about across the room.
    _say(ctx, key, "Preview discarded.", toast=False)


# --- T7: the character plan card ------------------------------------------


def _character_fields(plan: dict[str, Any]) -> dict[str, Any]:
    """*plan* (:func:`~.familiar.character_plan.parse_plan`'s own shape) ->
    the Create form's own ``character_*`` field names -- the same subset
    ``poser_mode.vary_in_create`` writes for a recipe it is varying, built
    only from whichever of *plan*'s fields are actually present (never
    invents a theme, a camera or a name the plan itself does not carry).

    ``movements`` is filtered against ``character_engine.MOVEMENTS`` (the
    closed default trio Create's own action checkboxes offer), the same
    filter ``vary_in_create`` applies to a recipe's ``animations`` keys --
    Create's form has no control for a movement outside that ladder, so one
    from the plan's own wider vocabulary (any shipped clip, not just the
    default three) is silently absent from the brief rather than fed to a
    checkbox that cannot show it; the plan itself still built the full list
    into *its own* ``overrides``, for the Create button's own path.
    """
    from ..modes.create.engine import character as character_engine

    fields: dict[str, Any] = {}
    if "family" in plan:
        fields["character_family"] = plan["family"]
    if "theme" in plan:
        fields["character_theme"] = plan["theme"]
    if "movements" in plan:
        wanted = set(plan["movements"])
        fields["character_actions"] = ",".join(
            name for name, _frames in character_engine.MOVEMENTS if name in wanted
        )
    if "size" in plan:
        fields["character_pixel"] = str(int(plan["size"]))
    if "name" in plan:
        fields["character_name"] = plan["name"]
    return fields


def submit_character(ctx: Any) -> bool:
    """Mint the pending plan's character. -> whether the press was taken.

    Busy-guarded on :data:`CHARACTER_KEY` -- its own key, not :data:`CHAT_KEY`
    -- and refused outright with no plan waiting. The plan is cleared the
    moment the submit is *accepted*, not once it lands: the card should not
    keep showing Create/Open/Discard against a request already in flight,
    the same "clear on accept" rule :func:`draw_expanded` already applies to
    the input line.
    """
    ui = ensure(ctx)
    if ui.plan is None or ctx.busy(CHARACTER_KEY):
        return False
    action = ui.plan
    plan = action["plan"]

    from ...service import familiar as svc_familiar
    from ...service import familiar_log

    exchange_id = familiar_log.new_exchange_id()

    def run() -> Any:
        with familiar_log.exchange(exchange_id):
            return svc_familiar.create_planned_character(
                ctx.svc, action["prompt"], action["overrides"], plan.get("name")
            )

    tag = {"thread_key": thread_key(ctx), "exchange": exchange_id}
    if not ctx.submit(CHARACTER_KEY, run, tag=tag):
        return False
    if familiar_log.enabled():
        with familiar_log.exchange(exchange_id):
            familiar_log.record("submit", submit_kind="character", prompt=action["prompt"])
    ui.plan = None
    ui.reason = None
    ui.message = None
    return True


def open_character_in_create(ctx: Any) -> None:
    """Draft the pending plan's character into Create instead of minting it
    -- :func:`~.familiar_doors.draft_in_create`'s own ``character_fields``
    door, the one T8 built exactly for this caller. Clears the plan only
    once the draft actually lands and appends the door's own sentence to the
    transcript, the same landing every other routed action already gets.

    The 2026-09-23 audit (familiar-01): this used to clear ``ui.plan``
    unconditionally, so a gated Create (``model_gate.mode_gate`` refusing)
    dropped the proposed plan along with the refusal sentence -- the user
    was left with neither a drafted brief nor the card to retry from.
    :func:`submit_character` already only clears on acceptance; this now
    matches it. The gate is read here rather than through ``draft_in_create``'s
    own return value -- a side-effect-free read, so checking it twice per
    press is free -- to leave that function's return shape (the sentence
    alone) untouched for its other caller and for the tests that monkeypatch
    it as a plain string-returning function.
    """
    ui = ensure(ctx)
    if ui.plan is None:
        return
    action = ui.plan
    from ..panes import model_gate
    from . import doors as familiar_doors

    where, _blocked = model_gate.mode_gate(ctx, "create")
    text = familiar_doors.draft_in_create(
        ctx, "character", action["prompt"], character_fields=_character_fields(action["plan"])
    )
    if not where:
        ui.plan = None
    threads_obj = getattr(ctx, "familiar_threads", None)
    if threads_obj is not None:
        from ...familiar import threads

        threads_obj.append(thread_key(ctx), threads.Turn("familiar", text))


def install_missing_familiar(ctx: Any) -> None:
    """The Install... action behind a "missing" refusal's expanded card.

    familiar-05 (the 2026-09-23 audit): ``ui.reason`` was set on every
    refusal but never read anywhere, contrary to this module's own docstring
    and ``service.familiar``'s ("the pane can choose an icon or an action --
    an Install... button"). A missing engine or weights refusal is the one
    reason Settings can actually fix, so this ticks Familiar's three rows
    (``MISSING_FAMILIAR_ROWS``) and takes the user to Settings -> Models --
    ``model_gate.request_install``, the same door a missing-weights Create
    gate already uses, rather than a copy of its two-line body.

    A plain function, not inlined in :func:`draw_expanded`'s ``if`` --
    headless-testable the way :func:`submit_chat`/:func:`follow_citation`
    already are, since ``draw_expanded`` itself needs a live imgui frame.
    """
    from ..panes import model_gate

    model_gate.request_install(ctx, MISSING_FAMILIAR_ROWS)


def discard_character_plan(ctx: Any) -> None:
    """Drop the pending plan, if any -- nothing was ever queued, so there is
    nothing to undo, only the card to stop showing."""
    ui = ensure(ctx)
    ui.plan = None


# --- drawing -----------------------------------------------------------


def follow_citation(ctx: Any, citation: Any) -> None:
    """Open the Manual at *citation*'s own chapter/section.

    The click action behind a transcript's ``[n]`` link -- factored out of
    :func:`draw_expanded` so a headless test can drive it directly, the same
    reason :func:`submit_chat`/:func:`submit_build` are functions a button's
    ``if`` just calls rather than inline imgui-handler bodies.
    """
    from ..manual import render as manual_render

    manual_render.open_at(ctx, (citation.chapter, citation.anchor))


def _draw_plan_card(ctx: Any, ui: FamiliarUIState) -> None:
    """The pending plan: species, theme, movements, directions, an estimate,
    and whatever the prompt named that the plan could not use -- then
    Create/Open in Create/Discard. ``ui.message`` (a refused Create press)
    is drawn by the caller, above the transcript, the same way every other
    refusal already is -- this function only draws the plan's own summary
    and its three buttons.
    """
    from imgui_bundle import imgui

    from .. import controls, widgets

    action = ui.plan or {}
    summary = action.get("summary") or {}

    imgui.text_wrapped(str(summary.get("species") or ""))
    if summary.get("theme"):
        widgets.secondary(f"Theme: {summary['theme']}")
    movements = summary.get("movements") or []
    if movements:
        widgets.secondary("Movements: " + ", ".join(movements))
    if summary.get("directions"):
        widgets.secondary(f"Directions: {summary['directions']}")
    widgets.secondary(
        f"~{summary.get('estimate_minutes', 0):.1f} min, {summary.get('cells', 0)} cells"
    )
    for item in summary.get("ignored") or ():
        widgets.secondary(f"(not used: {item.get('text', '')})")

    busy = ctx.busy(CHARACTER_KEY)
    if controls.small_button("Create##familiar/character-create", enabled=not busy):
        submit_character(ctx)
    imgui.same_line()
    if controls.small_button("Open in Create##familiar/character-open", enabled=not busy):
        open_character_in_create(ctx)
    imgui.same_line()
    if controls.small_button("Discard##familiar/character-discard", enabled=not busy):
        discard_character_plan(ctx)


def draw_expanded(ctx: Any) -> None:
    """The dock's body once expanded: the thread transcript, the input line,
    Send, and -- in Clay, with a tab open -- Build, plus Apply/Discard once a
    preview is ready. Drawn by ``familiar_dock.draw`` inside its own child
    region; this function assumes it is already inside one.
    """
    from imgui_bundle import imgui

    from .. import controls, theme, tokens

    ui = ensure(ctx)
    key = thread_key(ctx)
    turns = ctx.familiar_threads.get(key) if getattr(ctx, "familiar_threads", None) else ()

    # The transcript takes exactly what the dock's height leaves after the
    # footer below it -- the message line, the preview buttons or plan card,
    # and the input with its Send row -- so the input stays pinned to the
    # dock's bottom edge and the transcript is the only thing that scrolls.
    # The footer's height is *measured* (last frame's, in ``_FOOTER_H``)
    # rather than estimated: the estimate this replaced counted the input and
    # Send as one row when they are two, so the footer overflowed the dock and
    # the whole dock scrolled, input and all (2026-09-24). A first frame with
    # nothing measured falls back to that estimate, corrected.
    footer = _FOOTER_H[0]
    if footer <= 0.0:
        footer = imgui.get_frame_height_with_spacing() * 2.0
        if ui.message:
            footer += imgui.get_text_line_height_with_spacing()
        if ui.plan is not None:
            footer += imgui.get_frame_height_with_spacing() * 4.0
        if ui.preview_calls is not None:
            footer += imgui.get_frame_height_with_spacing()
    transcript_h = max(tokens.sp(40.0), imgui.get_content_region_avail().y - footer)

    # A turn already at the bottom stays pinned to it as new ones arrive; one
    # scrolled up to reread history is left alone. Read *before* this frame's
    # content is drawn, against last frame's scroll range, which is the
    # standard chat-log idiom -- there is no other point at which "was the
    # user already at the bottom" can be asked.
    was_at_bottom = imgui.get_scroll_y() >= imgui.get_scroll_max_y() - 1.0

    pad = tokens.sp(tokens.SP_2)
    imgui.push_style_var(imgui.StyleVar_.window_padding.value, (pad, pad))
    imgui.push_style_var(imgui.StyleVar_.item_spacing.value, (pad, pad * 0.5))
    imgui.begin_child("##familiar-transcript", (0, transcript_h), imgui.ChildFlags_.borders.value)
    wrap_width = imgui.get_content_region_avail().x
    for turn_idx, turn in enumerate(turns[-20:]):
        prefix = "You: " if turn.role == "user" else "Familiar: "
        text = prefix + turn.text
        bubble = theme.BUBBLE_USER if turn.role == "user" else theme.BUBBLE_ASSISTANT
        size = imgui.calc_text_size(text, None, False, wrap_width)
        origin = imgui.get_cursor_screen_pos()
        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect_filled(
            (origin.x - pad * 0.5, origin.y - pad * 0.25),
            (origin.x + size.x + pad * 0.5, origin.y + size.y + pad * 0.25),
            imgui.get_color_u32(imgui.ImVec4(*theme.rgba(bubble))),
            rounding=pad * 0.5,
        )
        imgui.text_wrapped(text)
        # A Manual answer's own [n] markers, each a small link back to the
        # section it came from -- ``cited`` already guarantees every one of
        # these actually appeared in the reply, so there is no dead link to
        # guard against here, only ids: turn_idx keeps two different turns'
        # citation buttons from colliding once imgui hashes the label.
        for cite_idx, citation in enumerate(turn.citations):
            if cite_idx:
                imgui.same_line()
            label = f"[{citation.n}] {citation.title_path}##familiar-cite-{turn_idx}-{cite_idx}"
            if controls.small_button(label):
                follow_citation(ctx, citation)
    if was_at_bottom:
        imgui.set_scroll_here_y(1.0)
    imgui.end_child()
    imgui.pop_style_var(2)

    footer_top = imgui.get_cursor_pos_y()
    try:
        _draw_footer(ctx, ui)
    finally:
        # The cursor sits one item spacing below the last footer item, which
        # is exactly the spacing the transcript's own end left above it.
        _FOOTER_H[0] = imgui.get_cursor_pos_y() - footer_top


#: The footer's measured height last frame, physical px -- see
#: :func:`draw_expanded`. Module state, not ``FamiliarUIState``: it is a fact
#: about the last frame's layout, not about the conversation.
_FOOTER_H = [0.0]


def _draw_footer(ctx: Any, ui: FamiliarUIState) -> None:
    """Everything under the transcript: the message line, Apply/Discard or
    the plan card, and the input with Send. Split out of
    :func:`draw_expanded` so its drawn height can be measured."""
    from imgui_bundle import imgui

    from .. import controls, theme, widgets

    if ui.message:
        imgui.text_colored(imgui.ImVec4(*theme.rgba(theme.WARN)), ui.message)
        # familiar-05 (2026-09-23 audit): the one reason an Install door
        # actually exists for -- the other reasons (lease, vram, backoff,
        # unhealthy, ...) describe a state Settings cannot fix.
        if ui.reason == "missing" and controls.small_button(
            "Install...##familiar/install-missing"
        ):
            install_missing_familiar(ctx)

    pending = ui.preview_calls is not None
    if pending:
        # Disabled while Familiar is thinking: the reply is a refinement of
        # this ghost, and applying or discarding it underneath would only make
        # that reply land as "preview again".
        if controls.small_button("Apply##familiar/apply", enabled=not ui.thinking):
            apply_preview(ctx)
        imgui.same_line()
        if controls.small_button("Discard##familiar/discard", enabled=not ui.thinking):
            discard_preview(ctx)

    if ui.plan is not None:
        if pending:
            # familiar-01 (2026-09-24 audit): a Clay preview and a landed
            # character plan used to both draw a Discard button in this same
            # row -- two identically labelled controls with different
            # effects. The preview (drawn above) keeps the pane's one row of
            # action buttons; the plan is kept, not dropped, and its card
            # draws on its own turn once the preview is applied or discarded.
            widgets.secondary(
                "A character plan is waiting -- it will show once this preview "
                "is applied or discarded."
            )
            return
        _draw_plan_card(ctx, ui)
        return

    imgui.set_next_item_width(-1.0)
    hint = "Refine the preview..." if pending else "Ask Familiar..."
    _changed, ui.input_text = controls.input_text_with_hint(
        "##familiar-input", hint, ui.input_text
    )
    # Not ``changed and Enter``: pressing Enter does not change the text, so
    # that pairing never fired. Enter deactivates a single-line field on the
    # frame it is pressed, which is the frame to read the key on.
    enter_pressed = imgui.is_item_deactivated() and imgui.is_key_pressed(imgui.Key.enter)

    is_clay = str(getattr(ctx.state, "mode", "")) == "clay" and bool(_active_tab_uid(ctx))
    busy = bool(ui.thinking)
    if is_clay:
        # On the testing pin (empty card_shas) a Build press still submits --
        # ``service.familiar.clay_build`` is what refuses with reason "card",
        # naming familiar_v1.0, and that refusal is what the pane then shows.
        # Refusing here too, before the submit, would just move the same
        # sentence one frame earlier for no gain and a second place to keep
        # it in sync with ``FAMILIAR_V1_NAME``.
        # Clear the line only once a submit is accepted: a refused one (blank,
        # busy, no tab) must leave what the user typed where it was.
        if controls.small_button("Build##familiar/build", enabled=not busy) and submit_build(
            ctx, ui.input_text
        ):
            ui.input_text = ""
        imgui.same_line()
    send = controls.small_button("Send##familiar/send", enabled=not busy)
    if (send or (enter_pressed and not busy)) and submit_chat(ctx, ui.input_text):
        ui.input_text = ""
    if busy:
        from .. import widgets

        imgui.same_line()
        # A sentence the user has to read, so widgets.secondary rather than
        # imgui's disabled role, which fails contrast in both themes (UX-18).
        widgets.secondary("Thinking...")
