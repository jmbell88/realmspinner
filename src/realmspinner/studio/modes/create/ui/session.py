"""Frame-thread adoption and draft persistence for asset-first Create."""

from __future__ import annotations

import copy
import uuid
from typing import Any

from ..engine import workspace as families

SETTINGS_KEY = "create_drafts"
MAX_DRAFTS = 64


def index(ctx: Any) -> families.Index:
    state = ctx.state.create
    generation = getattr(ctx.cache, "_generation", None)
    key = (ctx.cache, generation, len(ctx.cache.jobs))
    if generation is not None and state.index_memo is not None and state.index_memo[0] == key:
        return state.index_memo[1]
    result = families.build_index(ctx.cache.jobs)
    if generation is not None:
        state.index_memo = (key, result)
    return result


def save_draft(ctx: Any, outgoing: tuple[Any, Any] | None = None) -> None:
    """File the form under the current creation.

    *outgoing* is the ``(source, selected)`` the creation had before a new
    selection overwrote them (create-01); ``None`` files the live values.
    """
    state = ctx.state.create
    if state.workspace is None:
        return
    from ....settings import sanitise_form

    draft = {
        "form_2d": copy.deepcopy(ctx.state.form_2d),
        "form_3d": copy.deepcopy(ctx.state.form_3d),
        "source": ctx.state.source_job if outgoing is None else outgoing[0],
        "selected": ctx.state.selected if outgoing is None else outgoing[1],
    }
    state.drafts[state.workspace] = draft
    if len(state.drafts) > MAX_DRAFTS:
        state.drafts.pop(next(iter(state.drafts)))
    settings = getattr(ctx, "settings", None)
    if settings is not None:
        stored = settings.get(SETTINGS_KEY)
        stored = dict(stored) if isinstance(stored, dict) else {}
        durable = {
            **draft,
            "form_2d": sanitise_form(draft["form_2d"]),
            "form_3d": sanitise_form(draft["form_3d"]),
        }
        if stored.get(state.workspace) != durable:
            stored.pop(state.workspace, None)
            stored[state.workspace] = durable
            settings.set(SETTINGS_KEY, dict(list(stored.items())[-MAX_DRAFTS:]))


def resume(ctx: Any, job: dict[str, Any]) -> None:
    state = ctx.state.create
    key = index(ctx).by_job.get(str(job["id"]))
    if key is None or key == state.workspace:
        return
    # From the snapshot ``sync`` took last frame, when there is one: by now
    # ``library.select`` has already put the incoming card's id in
    # ``source_job``, and filing that under the OLD creation made going back
    # restore another creation's reference as its source (create-01).
    save_draft(ctx, state.synced)
    state.workspace = key
    state.comparison_pin = state.image_comparing = None
    state.preview_mode = "result"
    draft = state.drafts.get(key)
    if draft is None:
        settings = getattr(ctx, "settings", None)
        stored = settings.get(SETTINGS_KEY) if settings is not None else None
        draft = stored.get(key) if isinstance(stored, dict) else None
        if isinstance(draft, dict):
            from ....settings import restore_form
            from ....state import DEFAULT_FORM_3D, default_form_2d

            draft = {
                **draft,
                "form_2d": restore_form(default_form_2d(), draft.get("form_2d")),
                "form_3d": restore_form(DEFAULT_FORM_3D, draft.get("form_3d")),
            }
    if isinstance(draft, dict):
        ctx.state.form_2d = copy.deepcopy(draft["form_2d"])
        ctx.state.form_3d = copy.deepcopy(draft["form_3d"])
        ctx.state.source_job = draft.get("source")
    else:
        from ....state import DEFAULT_FORM_3D, form_from_params

        family = families.results(index(ctx), key)
        reference = next((j for j in family if j.get("stage") != "model"), job)
        ctx.state.form_2d = form_from_params(
            reference.get("params") or {}, stage=str(reference.get("stage") or "")
        )
        ctx.state.form_2d["prompt"] = str(reference.get("prompt") or "")
        source = job if job.get("stage") == "reference" else ctx.cache.get(job.get("parent_id"))
        ctx.state.source_job = str(source["id"]) if source is not None else None
        ctx.state.form_3d = dict(DEFAULT_FORM_3D)
        mesh = next((j for j in family if j.get("stage") == "model"), None)
        if mesh is not None:
            for name in ctx.state.form_3d:
                if name in (mesh.get("params") or {}):
                    ctx.state.form_3d[name] = mesh["params"][name]
    state.workspace_selection = str(job["id"])
    ctx.state.clear_field_errors()
    state.reference_path_checked = False


def sync(ctx: Any) -> None:
    selected = getattr(ctx.state, "selected", None)
    state = ctx.state.create
    if selected and selected != state.workspace_selection:
        job = ctx.cache.get(selected)
        if job is not None:
            resume(ctx, job)
        state.workspace_selection = selected
    save_draft(ctx)
    state.synced = (ctx.state.source_job, ctx.state.selected)


def metadata(ctx: Any) -> dict[str, str]:
    state = ctx.state.create
    if state.workspace is None:
        state.workspace = "creation:" + uuid.uuid4().hex[:12]
    elif not state.workspace.startswith(("creation:", *families.LEGACY_ROOTS)):
        old_key = state.workspace
        state.workspace = "creation:" + old_key
        if old_key in state.drafts:
            state.drafts[state.workspace] = state.drafts.pop(old_key)
    # Legacy graph roots are valid workspace identities too; prefix is stripped
    # only for new metadata so resuming a new creation reconstructs the same key.
    key = state.workspace.removeprefix("creation:")
    return {families.WORKSPACE_PARAM: key}


def new(ctx: Any, *, reuse: bool = False, then: Any = None) -> None:
    from ....state import DEFAULT_FORM_3D, default_form_2d
    from . import stages

    def reset() -> None:
        save_draft(ctx)
        if stages.at(ctx.state, "pose"):
            from ....panes import pose_panel

            pose_panel.leave(ctx)
        ctx.state.create.workspace = None
        ctx.state.create.workspace_selection = None
        ctx.state.create.comparison_pin = ctx.state.create.image_comparing = None
        ctx.state.create.preview_mode = "result"
        ctx.state.select(None)
        viewer = getattr(ctx, "viewer", None)
        if viewer is not None:
            viewer.clear()
            viewer.clear_reference()
        ctx.state.source_job = None
        ctx.state.create.submit_refusal = ""
        ctx.state.clear_field_errors()
        if not reuse:
            ctx.state.form_2d = default_form_2d()
            ctx.state.form_3d = dict(DEFAULT_FORM_3D)
        stages.go(ctx, "reference", follow=False)
        if then is not None:
            # After the reset, so a pose-guard that defers it still lands the
            # caller's own form and stage last (create-02).
            then()

    if stages.at(ctx.state, "pose"):
        from ....panes import pose_panel

        pose_panel.guard(ctx, "start a new creation", reset)
    else:
        reset()
