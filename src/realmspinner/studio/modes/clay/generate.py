"""Generate into the current Clay tab: a mesh that lands as new objects in the
document already open, rather than as a new tab.

Modelled on Inker's "Regenerate selection" bridge (``studio/modes/inker/ui/
panes/bridge.py``'s ``submit_inpaint``/``poll_inpaint``/``land_inpaint``) and
its Flourish sibling (``studio/modes/inker/flourish.py``'s
``submit_texture``/``poll_texture``): record which tab asked, poll the store
cheaply on the frame thread, do every file read and decode on a task thread,
land by uid rather than by "whatever is active now".

**The job is an ordinary Library row throughout.** Text goes through the same
two-stage pipeline Create's Reference stage does -- a reference the user
approves, then a promoted mesh -- because a blind two-minute trellis run on a
prompt nobody has looked at is exactly the mistake that stage exists to
prevent; the difference from Create is only where the *result* lands. An
image goes straight to a mesh, the same shape Create's own upload path
already takes.

**This is not an agent door.** ``studio/modes/clay/agent/`` derives its tool
list from ``ops.OPS``/``primitives.GENERATORS``/``presets.ASSEMBLIES`` and
never reaches this module -- see ``tests/modes/clay/test_agent_clay_
generate.py``, which walks that surface by AST and fails if it ever does. A
door onto text2image and trellis from inside an agent's own tab is a
capability the brief for this feature explicitly withholds.

**One request for the whole app.** ``ClayState.generate_pending`` holds it
(``None`` when nothing is under way); the popup that starts one is modal in
the sense that there is nowhere else to start a second from, so there is only
ever one to track, the same shape Inker's ``inpaint_pending`` and
``flourish_texture_pending`` already take.

Every task key here carries the ``clay-gen`` prefix -- :data:`TASK_KEYS` is
the exact set, and ``clay_mode.on_task_done``/``on_task_failed`` both check
membership in it *before* their own generic ``clay-*:<uid>`` handling, for
the ``clay-bg`` reason already written there: a generate in flight (or one
that just failed) must never flip ``tab.saving``.
"""

from __future__ import annotations

import time
from typing import Any

from ... import dialogs
from . import mode as clay_mode

#: The three task keys this module ever submits under, each suffixed
#: ``:<tab uid>``. Checked by name, as a set, from both of ``clay_mode``'s
#: task-landing functions -- see this module's own docstring.
GEN_REF_KEY = "clay-gen-ref"
GEN_KEY = "clay-gen"
GEN_LAND_KEY = "clay-gen-land"
TASK_KEYS = frozenset({GEN_REF_KEY, GEN_KEY, GEN_LAND_KEY})

#: How often :func:`poll` asks the store about a pending job -- Inker's own
#: ``INPAINT_POLL_S``/``TEXTURE_POLL_S``, cheap enough to run every frame's
#: worth of nothing happening.
GEN_POLL_S = 0.5

#: The gltfpack tiers offered in the popup, in the order they are shown --
#: :mod:`~.pipelines.optimize`'s own ``PROFILES``, read rather than restated
#: (that module's own docstring: "raw" is the untouched reconstruction, and
#: every other key is a triangle ceiling gltfpack simplifies down to).
#: **Not** "raw" by default: a bare TRELLIS response is routinely ~290k
#: triangles, comfortably past ``clay_mode.SLOW_TRIANGLES`` (200k) -- a user
#: who presses Generate over and over would hit that confirm on every single
#: press for no reason they chose, so the default here is "standard".
BUDGET_ORDER = ("draft", "standard", "detailed", "raw")
DEFAULT_BUDGET = "standard"


def budget_choices() -> tuple[tuple[str, str], ...]:
    """The popup's tier combo, as ``(key, label)`` pairs. Pure, so a test can
    check the labels without imgui."""
    from ....pipelines import optimize

    out = []
    for key in BUDGET_ORDER:
        cap = optimize.PROFILES.get(key)
        label = key.capitalize() if cap is None else f"{key.capitalize()} ({cap:,} tris)"
        out.append((key, label))
    return tuple(out)


def status_line(pending: dict[str, Any] | None) -> str:
    """What the hint line says while a generate is under way, or ``""``.

    Written onto ``ClayTab.bg_busy`` at every transition below -- that field
    is already read by ``hud.hint_line`` through ``viewport_hints.
    resolve_hint`` (see its own docstring, clay-41), so nothing in the pane
    or the viewport needed a new wire for this.
    """
    if pending is None:
        return ""
    stage = pending.get("stage")
    if stage == "reference":
        return "Generating a reference..."
    if stage == "mesh":
        return "Building the mesh..."
    if stage == "landing":
        return "Adding it to the document..."
    return ""  # "preview": the popup itself is the status


def settings_note(ctx: Any) -> str:
    """The popup's own line about what a text Generate press will use.

    Reads ``ctx.state.form_2d`` fresh rather than a value taken at the popup's
    own open -- the same reason :func:`submit_text` reads it fresh too -- so
    the line can never claim a base model or style the actual submit will not
    use. No imgui, so a test can check the sentence without a window.
    """
    form = ctx.state.form_2d
    base = str(form.get("base_model") or "").strip() or "the default model"
    style = str(form.get("style_lora") or "").strip()
    if style:
        return f"Uses Create's own settings: {base}, styled with {style}."
    return f"Uses Create's own settings: {base}, no style LoRA."


# --- bookkeeping shared by every stage ---------------------------------------


def _set_busy(state: Any, pending: dict[str, Any] | None) -> None:
    tab = state.get(pending["tab_uid"]) if pending is not None else None
    if tab is not None:
        tab.bg_busy = status_line(pending)


def _clear(state: Any, tab_uid: str) -> None:
    """Drop the pending request for *tab_uid*, and its hint line with it.

    Identity is checked against ``tab_uid`` rather than against a held
    ``pending`` reference: several call sites read ``state.generate_pending``
    fresh right before this, and a stale reference clearing a *newer* request
    for the same tab would be a bug this guards against for free.
    """
    tab = state.get(tab_uid)
    if tab is not None:
        tab.bg_busy = ""
    pending = state.generate_pending
    if pending is not None and pending.get("tab_uid") == tab_uid:
        state.generate_pending = None


def _tab_busy(ctx: Any, tab: Any) -> bool:
    """Whether *tab* is mid-save or mid-drag -- both mutate the document in
    ways a landing merge must not race."""
    if tab.saving:
        return True
    view = getattr(ctx, "clay_view", None)
    return bool(view is not None and getattr(view, "dragging", False))


# --- text: reference, then approval, then mesh --------------------------------


def submit_text(ctx: Any, tab: Any, prompt: str, *, budget: str = DEFAULT_BUDGET) -> bool:
    """Queue a *reference*, never a mesh directly -- the two-step Create's
    own Reference stage takes, so the two minutes of trellis are spent only
    once the picture has been looked at.

    The form is a copy of the app's own 2D form (``ctx.state.form_2d``) --
    whatever base model, style LoRA and conditioning the user already has set
    in Create, so this door adds no second place to choose them -- with the
    prompt, the asset kind and the seed overridden: the kind is always a mesh
    reference regardless of what Create's form last asked for (a tileset, a
    character), and the seed is always fresh, the same "give me another"
    default every ordinary Generate press gets.
    """
    state = clay_mode.ensure(ctx)
    if state.generate_pending is not None:
        ctx.toast("A generation is already under way.", "warn")
        return False
    prompt = prompt.strip()
    if not prompt:
        return False

    from ....service.validation import random_seed

    form = dict(ctx.state.form_2d)
    form["prompt"] = prompt
    form["asset_type"] = "3d_model"
    form["generation_type"] = "3d_model"
    form["output"] = "reference"
    form["count"] = 1
    form["seed"] = random_seed()

    from ..create.engine import recipe as create_recipe

    kwargs = create_recipe.submit_kwargs(form)

    pending: dict[str, Any] = {
        "tab_uid": tab.uid,
        "kind": "text",
        "stage": "reference",
        "budget": budget,
        "prompt": prompt,
        "reference_job_id": "",
        "mesh_job_id": "",
        "next_poll": 0.0,
        "force_offer": False,
    }
    key = f"{GEN_REF_KEY}:{tab.uid}"

    def run() -> Any:
        from ....service import jobs as svc_jobs

        return svc_jobs.create_job(ctx.svc, **kwargs)

    if not ctx.submit(key, run):
        return False
    state.generate_pending = pending
    _set_busy(state, pending)
    return True


def reroll_reference(ctx: Any, tab: Any) -> bool:
    """"Reroll": the same prompt and guidance, a fresh picture.

    ``service.jobs.rerun_job(mode="reroll")`` is the door every reroll button
    in the app already uses; it mints a fresh seed on its own when none is
    given, so this asks for nothing more than "again".
    """
    state = clay_mode.ensure(ctx)
    pending = state.generate_pending
    if (
        pending is None
        or pending.get("tab_uid") != tab.uid
        or pending.get("stage") != "preview"
    ):
        return False
    source_id = pending["reference_job_id"]
    key = f"{GEN_REF_KEY}:{tab.uid}"

    def run() -> Any:
        from ....service import jobs as svc_jobs

        return svc_jobs.rerun_job(ctx.svc, source_id, mode="reroll")

    if not ctx.submit(key, run):
        return False
    pending["stage"] = "reference"
    pending["reference_job_id"] = ""
    pending["force_offer"] = False
    pending["next_poll"] = 0.0
    _set_busy(state, pending)
    return True


def accept_reference(ctx: Any, tab: Any, *, force: bool = False) -> bool:
    """Promote the approved reference to a mesh.

    Mesh kwargs come from the app's own 3D form (``ctx.state.form_3d``,
    ``create.engine.mesh.promote_kwargs``) with the chosen budget written
    into ``profile`` -- overriding whatever profile the form otherwise holds,
    which is the whole point of the tier picker in the popup -- and ``rig``
    forced off: rigging is Poser's door, not this one's.

    ``force`` retries past the soft composition refusal
    (``service._jobs_resubmit.promote_to_model``'s own ``force=`` bypass);
    the popup's "Build anyway" sets it after a first refusal.
    """
    state = clay_mode.ensure(ctx)
    pending = state.generate_pending
    if (
        pending is None
        or pending.get("tab_uid") != tab.uid
        or pending.get("stage") != "preview"
    ):
        return False
    reference_id = pending["reference_job_id"]
    budget = pending.get("budget", DEFAULT_BUDGET)

    form = dict(ctx.state.form_3d)
    form["profile"] = budget
    form["rig"] = False

    def run() -> Any:
        from ....service import jobs as svc_jobs
        from ..create.engine import mesh as create_mesh

        kwargs = create_mesh.promote_kwargs(form)
        kwargs["rig"] = False
        kwargs["force"] = force
        return svc_jobs.promote_to_model(ctx.svc, reference_id, **kwargs)

    key = f"{GEN_KEY}:{tab.uid}"
    if not ctx.submit(key, run):
        return False
    pending["stage"] = "mesh"
    pending["mesh_job_id"] = ""
    pending["next_poll"] = 0.0
    _set_busy(state, pending)
    return True


# --- image: straight to a mesh ------------------------------------------------


def submit_image(ctx: Any, tab: Any, *, budget: str = DEFAULT_BUDGET) -> None:
    """"From an image...": pick a file and queue a mesh, no reference stage.

    Picker, read and submit all happen inside one task, ``ask_import_mesh``'s
    own shape: a native file dialog is modal to the OS and a large file is a
    real read, so neither may run on the frame thread. Cancelling the picker
    (``dialogs.open_file`` returning ``None``) lands as an ordinary "cancelled
    dialog" result, exactly like every other picker task in this package.
    """
    state = clay_mode.ensure(ctx)
    if state.generate_pending is not None:
        ctx.toast("A generation is already under way.", "warn")
        return
    form = dict(ctx.state.form_3d)
    form["profile"] = budget
    form["rig"] = False

    pending: dict[str, Any] = {
        "tab_uid": tab.uid,
        "kind": "image",
        "stage": "mesh",
        "budget": budget,
        "prompt": "",
        "reference_job_id": "",
        "mesh_job_id": "",
        "next_poll": 0.0,
        "force_offer": False,
    }
    key = f"{GEN_KEY}:{tab.uid}"

    def run() -> Any:
        from ....service import jobs as svc_jobs
        from ....service.validation import MAX_UPLOAD_BYTES
        from ..create.engine import mesh as create_mesh

        path = dialogs.open_file("Generate from an image", dialogs.IMAGE_FILTER)
        if path is None:
            return None
        with open(path, "rb") as fh:  # noqa: PTH123 - a plain read, capped below
            data = fh.read(MAX_UPLOAD_BYTES + 1)
        kwargs = create_mesh.upload_kwargs(form)
        kwargs["rig"] = False
        return svc_jobs.create_job(ctx.svc, image=data, **kwargs)

    if not ctx.submit(key, run):
        return
    state.generate_pending = pending
    _set_busy(state, pending)


# --- cancel --------------------------------------------------------------------


def cancel(ctx: Any, tab: Any) -> None:
    """Drop the pending request without landing it.

    The underlying job (a reference already queued, say) is not cancelled on
    the server -- it finishes as an ordinary Library row nobody asked to see
    again, the same as closing Create's own promote-preview modal leaves its
    reference sitting in the Library rather than deleting it.
    """
    state = ctx.state.clay
    if state is None:
        return
    pending = state.generate_pending
    if pending is None or pending.get("tab_uid") != tab.uid:
        return
    _clear(state, tab.uid)


# --- polling ---------------------------------------------------------------


def poll(ctx: Any) -> None:
    """Once a frame: is whatever is pending ready to move to its next stage."""
    state = ctx.state.clay
    if state is None:
        return
    pending = state.generate_pending
    if pending is None:
        return
    now = time.monotonic()
    stage = pending.get("stage")
    if stage == "reference":
        _poll_job(ctx, state, pending, now, job_key="reference_job_id", next_stage="preview")
    elif stage == "mesh":
        _poll_mesh(ctx, state, pending, now)
    elif stage == "landing":
        _retry_deferred(ctx, state, pending, now)


def _poll_job(
    ctx: Any, state: Any, pending: dict[str, Any], now: float, *, job_key: str, next_stage: str
) -> None:
    job_id = pending.get(job_key)
    if not job_id:
        return
    if now < float(pending.get("next_poll") or 0.0):
        return
    pending["next_poll"] = now + GEN_POLL_S
    try:
        job = ctx.svc.store.get(job_id)
    except Exception:  # noqa: BLE001 - the store answers next tick
        return
    if job is None:
        _clear(state, pending["tab_uid"])
        return
    status = job.get("status")
    if status in ("queued", "running"):
        return
    if status != "done":
        ctx.toast(f"The reference {status}: {job.get('error') or 'no result'}.", "warn")
        _clear(state, pending["tab_uid"])
        return
    pending["stage"] = next_stage
    _set_busy(state, pending)


def _poll_mesh(ctx: Any, state: Any, pending: dict[str, Any], now: float) -> None:
    job_id = pending.get("mesh_job_id")
    if not job_id:
        return
    if now < float(pending.get("next_poll") or 0.0):
        return
    pending["next_poll"] = now + GEN_POLL_S
    try:
        job = ctx.svc.store.get(job_id)
    except Exception:  # noqa: BLE001 - the store answers next tick
        return
    if job is None:
        _clear(state, pending["tab_uid"])
        return
    status = job.get("status")
    if status in ("queued", "running"):
        return
    if status != "done":
        ctx.toast(f"The mesh {status}: {job.get('error') or 'no result'}.", "warn")
        _clear(state, pending["tab_uid"])
        return
    pending["stage"] = "landing"
    _set_busy(state, pending)
    tab_uid = pending["tab_uid"]
    group_name = (pending.get("prompt") or "").strip() or "Generated"
    key = f"{GEN_LAND_KEY}:{tab_uid}"

    def run() -> Any:
        return _decode_landing(ctx.svc, job_id, tab_uid, group_name)

    ctx.submit(key, run)


def _retry_deferred(ctx: Any, state: Any, pending: dict[str, Any], now: float) -> None:
    """The landing found the tab mid-save (or mid-drag) last time it tried --
    try again, throttled the same as every other poll here."""
    deferred = pending.get("deferred")
    if deferred is None:
        return
    if now < float(pending.get("next_poll") or 0.0):
        return
    pending["next_poll"] = now + GEN_POLL_S
    land(
        ctx,
        pending["tab_uid"],
        deferred["doc"],
        deferred["triangles"],
        deferred["incoming_bytes"],
        deferred["group_name"],
        confirmed=deferred.get("confirmed", False),
    )


def _decode_landing(svc: Any, job_id: str, tab_uid: str, group_name: str) -> dict[str, Any]:
    """Blocking; task thread only. ``model.glb`` -> a scratch document, plus
    the two numbers :func:`land`'s ceiling check needs.

    ``incoming_bytes`` is *this* document's own encoded size (materials,
    meshes, no camera) -- computed here, off the frame thread, precisely so
    the landing never has to run ``serialize.snapshot_bytes`` on the frame
    thread to answer "would this make the document too big to reopen".
    """
    from ....kernels.mesh import glbimport, serialize

    mesh_path = svc.job_dir(job_id) / "model.glb"
    data = clay_mode._within_mesh_ceiling(mesh_path)
    doc = glbimport.glb_to_claydoc(data, name=group_name)
    triangles = sum(max(len(obj.mesh.starts) - 1, 0) for obj in doc.objects)
    incoming_bytes = len(serialize.snapshot_bytes(serialize.snapshot(doc)))
    return {
        "tab_uid": tab_uid,
        "doc": doc,
        "triangles": triangles,
        "incoming_bytes": incoming_bytes,
        "group_name": group_name,
    }


# --- landing -----------------------------------------------------------------


def _ceiling_refusal(tab: Any, incoming: Any, triangles: int, incoming_bytes: int) -> str | None:
    """Why landing *incoming* in *tab* would leave the document unopenable,
    or ``None``. The three ceilings this door has to answer for, in the order
    a user is most likely to hit them."""
    from ....kernels.mesh import glbimport
    from ....service.files import MAX_CLAY_SOURCE_BYTES

    doc = tab.doc
    existing_triangles = sum(max(len(o.mesh.starts) - 1, 0) for o in doc.objects)
    if existing_triangles + triangles > glbimport.MAX_TRIANGLES:
        return (
            f"Adding this would put the document past the "
            f"{glbimport.MAX_TRIANGLES:,} triangles Clay can edit."
        )
    # +1: a multi-object import gains a group empty on the way in
    # (``merge.merge_into``), so a document sitting exactly at the ceiling
    # must not be pushed one past it by an object nobody modelled.
    if len(doc.objects) + len(incoming.objects) + 1 > glbimport.MAX_OBJECTS:
        return (
            f"Adding this would put the document past the "
            f"{glbimport.MAX_OBJECTS:,} objects Clay holds."
        )
    # ``tab.rblk_bytes`` is 0 for a document this session has never measured
    # (a brand-new or imported tab with no save yet) -- read as "unknown"
    # rather than as zero bytes, the same sentinel ``ClayTab.rblk_bytes``'s
    # own docstring states.
    if tab.rblk_bytes and tab.rblk_bytes + incoming_bytes > MAX_CLAY_SOURCE_BYTES:
        return "Adding this would make the document too large to reopen."
    return None


def land(
    ctx: Any,
    tab_uid: str,
    incoming: Any,
    triangles: int,
    incoming_bytes: int,
    group_name: str,
    *,
    confirmed: bool = False,
) -> None:
    """Merge *incoming* into the tab named by *tab_uid*, on the frame thread.

    Refuses by name, and clears the pending request, when the tab is gone --
    closing a tab does not itself reach into this module (there was nothing
    useful to do at close time that this refusal does not already cover once
    the landing actually arrives) -- or when a ceiling would be crossed.
    Defers (leaving the pending request in place, retried by :func:`poll`)
    while the tab is mid-save or mid-drag, so a landing never mutates a
    document another write already has open. Past ``clay_mode.SLOW_TRIANGLES``
    it asks first, the same confirm an oversized import already gets, and the
    confirm's own callback re-enters this function with ``confirmed=True`` --
    re-resolving the tab by uid rather than trusting one captured before the
    dialog was shown, so a tab closed while the question was on screen is
    still refused by name rather than silently mutated.

    Never changes ``ClayState.active_uid``: this can land on a tab the user
    is not even looking at, and moving it out from under them is the
    INVARIANTS rule ``settle_drag`` exists to enforce for a live transform --
    a merge earns no exception to it.
    """
    state = ctx.state.clay
    if state is None:
        return
    tab = state.get(tab_uid)
    if tab is None:
        ctx.toast("The document the generation was for is closed.", "warn")
        _clear(state, tab_uid)
        return
    if _tab_busy(ctx, tab):
        pending = state.generate_pending
        if pending is not None and pending.get("tab_uid") == tab_uid:
            pending["stage"] = "landing"
            pending["deferred"] = {
                "doc": incoming,
                "triangles": triangles,
                "incoming_bytes": incoming_bytes,
                "group_name": group_name,
                "confirmed": confirmed,
            }
            pending["next_poll"] = 0.0
            _set_busy(state, pending)
        return

    refusal = _ceiling_refusal(tab, incoming, triangles, incoming_bytes)
    if refusal:
        ctx.toast(refusal, "warn")
        _clear(state, tab_uid)
        return

    if triangles > clay_mode.SLOW_TRIANGLES and not confirmed:
        # The job is done and there is nothing left to poll -- only the
        # question is left, so the pending request is cleared here and the
        # confirm's own callback carries what it needs to finish the job.
        _clear(state, tab_uid)
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Add this to the document?",
                message=(
                    f"{triangles:,} faces. Adding it will be slow -- every "
                    "edit afterwards rebuilds the whole mesh, and the undo "
                    "history holds two copies per step."
                ),
                confirm_label="Add anyway",
                cancel_label="Cancel",
                on_confirm=lambda: land(
                    ctx, tab_uid, incoming, triangles, incoming_bytes, group_name,
                    confirmed=True,
                ),
            )
        )
        return

    from ....kernels.mesh import merge

    offset = merge.placement_offset(tab.doc, incoming, list(tab.doc.selection))
    merge.merge_into(tab.doc, incoming, offset=offset, label="Generate", group_name=group_name)
    # clay-03 (the 2026-09-23 audit): ``_ceiling_refusal`` above trusts
    # ``tab.rblk_bytes`` to bound the *next* landing, but nothing here ever
    # grew it after a landing actually added bytes -- a second Generate press
    # read the same number this one started with (0 for a tab never saved
    # this session) and could never refuse. ``incoming_bytes`` is exactly
    # what this landing just added.
    tab.rblk_bytes += incoming_bytes
    _clear(state, tab_uid)
    ctx.toast("Added to the document.")


# --- task landing --------------------------------------------------------------


def on_task_done(ctx: Any, done: Any) -> None:
    """``clay_mode.on_task_done``'s branch for every key in :data:`TASK_KEYS`."""
    state = ctx.state.clay
    if state is None:
        return
    name = done.key.split(":", 1)[0]
    if name == GEN_REF_KEY:
        _queued(ctx, state, done, job_key="reference_job_id", noun="reference")
    elif name == GEN_KEY:
        _queued(ctx, state, done, job_key="mesh_job_id", noun="mesh")
    elif name == GEN_LAND_KEY:
        _landed(ctx, state, done)


def _queued(ctx: Any, state: Any, done: Any, *, job_key: str, noun: str) -> None:
    pending = state.generate_pending
    if pending is None:
        return
    result = done.result
    if result is None:
        # A cancelled picker (the image flow's own dialog) or an empty
        # "not queued" answer read the same way: nothing to poll, no reason
        # to say more than that nothing happened.
        _clear(state, pending["tab_uid"])
        return
    job_id = str(result.get("id") or "") if isinstance(result, dict) else ""
    if not job_id:
        _clear(state, pending["tab_uid"])
        ctx.toast(f"The {noun} was not queued.", "warn")
        return
    pending[job_key] = job_id
    pending["force_offer"] = False
    _set_busy(state, pending)


def _landed(ctx: Any, state: Any, done: Any) -> None:
    result = done.result
    if not isinstance(result, dict):
        pending = state.generate_pending
        if pending is not None:
            _clear(state, pending["tab_uid"])
        return
    pending = state.generate_pending
    if pending is None or pending.get("tab_uid") != result.get("tab_uid"):
        # clay-04 (the 2026-09-23 audit): the decode task this landing comes
        # from was already submitted by the time Cancel was pressed --
        # ``cancel()`` only clears ``generate_pending``, it does not (and
        # cannot) stop the task already in flight. This used to call
        # ``land()`` unconditionally regardless, so a cancel during the
        # "landing" stage cleared the pending request but the mesh still
        # merged in once the decode came back -- exactly what ``cancel()``'s
        # own docstring promises does not happen. No pending request naming
        # this tab means there is nothing left to land for.
        return
    land(
        ctx,
        result["tab_uid"],
        result["doc"],
        int(result.get("triangles", 0)),
        int(result.get("incoming_bytes", 0)),
        result.get("group_name") or "Generated",
    )


def _composition_refused(ctx: Any, pending: dict[str, Any]) -> bool:
    """Whether the promote failed on the one refusal ``force`` bypasses.

    ``promote_to_model`` raises the same ``Invalid`` for its soft composition
    check as for a reference that is not done or has no image, and the VRAM
    and weights doors behind it refuse through the same task failure -- so
    the exception alone cannot say whether "Build anyway" would get past it.
    The reference row's own ``reference_report`` can: it is exactly what the
    door refuses on, and a retry with ``force`` past any *other* refusal
    would only fail again.
    """
    row = ctx.svc.store.get(pending.get("reference_job_id", "")) or {}
    report = (row.get("params") or {}).get("reference_report") or {}
    return report.get("ok") is False


def _composition_reasons(ctx: Any, pending: dict[str, Any]) -> tuple[str, ...]:
    """The sentences behind a composition refusal -- the same
    ``reference_report["reasons"]`` Create's own matte popup already shows
    (``create/ui/panes/settings_3d.py``'s ``_matte_body``).

    docs-02 (the 2026-09-23 audit): the manual promises this beside "Build
    anyway", and the popup drew nothing there -- only the one-shot toast
    every failed ``clay-`` task already gets from ``shell/tasks.py``, gone
    the instant it faded or if the popup was reopened later. Read the same
    way :func:`_composition_refused` does, so the two can never disagree
    about which job's report they mean.
    """
    row = ctx.svc.store.get(pending.get("reference_job_id", "")) or {}
    report = (row.get("params") or {}).get("reference_report") or {}
    return tuple(report.get("reasons") or ())


def on_task_failed(ctx: Any, done: Any) -> None:
    """``clay_mode.on_task_failed``'s branch for every key in :data:`TASK_KEYS`.

    A failed *mesh* task whose reference failed the composition check
    (:func:`_composition_refused`) drops back to the preview stage with
    ``force_offer`` set, so the
    popup can show "Build anyway" beside Accept/Reroll/Cancel rather than
    only the toast the shell already showed (every ``clay-``-keyed task
    failure is toasted there, before this runs -- see ``shell/tasks.py``'s
    ``_collect_tasks``). A second failure with ``force`` already true gives
    up outright: retrying an already-forced promote cannot ask anything new.
    Any other failure (the reference queue door, the landing decode) simply
    drops the pending request -- the toast already said why.
    """
    state = ctx.state.clay
    if state is None:
        return
    name = done.key.split(":", 1)[0]
    pending = state.generate_pending
    if pending is None:
        return
    tab_uid = pending.get("tab_uid", "")
    if (
        name == GEN_KEY
        and pending.get("stage") == "mesh"
        and not pending.get("force_offer")
        and _composition_refused(ctx, pending)
    ):
        pending["stage"] = "preview"
        pending["mesh_job_id"] = ""
        pending["force_offer"] = True
        pending["doubt_reasons"] = _composition_reasons(ctx, pending)
        _set_busy(state, pending)
        return
    _clear(state, tab_uid)
