"""Headless replay and verification for ``drafts/*.jsonl`` records.

Mirrors ``tests/test_agent_transcripts.py``'s own replay rules -- a fresh ctx
and ``agent_clay.Session()`` per record, ``agent_clay.call`` as the only
door -- but drives whole ``clay_batch`` calls rather than a line-by-line
transcript, because a single ``clay_batch`` call is the shape the trained
model will actually emit at inference (see the plan's "Verifier design"
section). ``clay_render`` is ``agent_clay.BATCH_EXCLUDED`` and needs
``ctx.viewer``, which :class:`headless.HeadlessCtx` deliberately does not
provide -- so no training row can contain it, and the gallery
(``render.py``) is drawn separately, outside this replay.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from warlock.studio import agent_clay, clay_mode  # noqa: E402
from warlock.studio.clay import diagnose as clay_diagnose  # noqa: E402
from warlock.studio.clay import serialize  # noqa: E402

from . import headless  # noqa: E402

MIN_DOC_SIZE = 0.005
MAX_DOC_SIZE = 20.0
"""Rule 4's document-wide bounds ceiling/floor -- from ``clay_scene``'s own
``bounds.size``, over every visible object at once."""

MIN_COMPONENT = 1e-3
"""Rule 4's per-object floor. Compared with ``<=`` rather than the plan's
bare ``<``: ``_scene_row``'s own ``size`` is already rounded to 4 decimal
places (``agent_clay._round``), so a generator asked for exactly this
boundary produces a size that reads back as precisely ``0.001`` -- and a
deliberately-degenerate object at exactly the floor is exactly the case this
check exists to catch, not a value to let through on a coin-flip of float
representation."""

GROUND_EPS = -0.02
"""Rule 5's floor for a visible object's own bbox min Y. Generators are
centred, so a box of height h sits at y=h/2 once placed correctly -- see
``instructions()``'s own opening paragraph -- and this is loose enough to
forgive rounding noise while still catching an object placed at the origin
with no ground offset at all."""


def _refusal_text(result: dict[str, Any]) -> str:
    content = result.get("content") or []
    if content and isinstance(content[0], dict):
        return str(content[0].get("text", ""))
    return "refused (no message on the result)"


def _run_batch(ctx: Any, session: Any, calls: list[dict]) -> dict:
    return agent_clay.call(ctx, session, "clay_batch", {"calls": calls})


def _read_scene(ctx: Any, session: Any) -> dict | None:
    result = agent_clay.call(ctx, session, "clay_scene", {})
    if result.get("isError"):
        return None
    return result.get("structuredContent")


def _tab_doc(ctx: Any, session: Any) -> Any | None:
    if not session.tab_uid:
        return None
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    return tab.doc if tab is not None else None


@dataclass
class Replay:
    """One record's whole replay: a fresh door, run once (or twice, for an
    ``edit``/``query`` record's own ``prior``), captured so :func:`check` can
    grade it without re-running anything."""

    ctx: Any
    session: Any
    doc: Any | None
    prior_result: dict | None
    main_result: dict
    head_before_main: int | None
    head_after_main: int | None
    scene: dict | None


def replay(calls: list[dict], *, prior: list[dict] | None = None) -> Replay:
    """Run *prior* (if given, as one folded ``clay_batch``) and then *calls*
    (also one ``clay_batch``) through a fresh :class:`headless.HeadlessCtx` +
    ``agent_clay.Session`` -- the real door, driven the way the model will
    drive it, so ``{"$ref": "<name>"}`` is resolved by the real
    ``agent_clay._resolve_batch_ref`` rather than assumed to work.

    A ``build`` record calls this with no *prior*. An ``edit`` record passes
    its own ``prior`` (the calls that build the starting scene) here so
    ``check`` can also assert the edit actually moved
    ``doc.history.head`` -- a no-op edit is not a training example of an
    edit. A ``query`` record calls this with its ``prior`` as *calls* and no
    separate prior at all: a query never mutates, so there is nothing for it
    to move.
    """
    ctx = headless.HeadlessCtx()
    session = agent_clay.Session()

    prior_result: dict | None = None
    head_before_main: int | None = None
    if prior:
        prior_result = _run_batch(ctx, session, prior)
        doc = _tab_doc(ctx, session)
        if doc is not None:
            head_before_main = doc.history.head

    main_result = _run_batch(ctx, session, calls)

    doc = _tab_doc(ctx, session)
    head_after_main = doc.history.head if doc is not None else None
    scene = _read_scene(ctx, session) if doc is not None else None

    return Replay(
        ctx=ctx,
        session=session,
        doc=doc,
        prior_result=prior_result,
        main_result=main_result,
        head_before_main=head_before_main,
        head_after_main=head_after_main,
        scene=scene,
    )


def scene(replay_result: Replay) -> dict:
    """The ``clay_scene`` ``structuredContent`` captured at the end of
    *replay_result* -- ``{}`` if the session never minted a document (every
    call refused before the first primitive landed)."""
    return replay_result.scene or {}


def check(replay: Replay, *, allow_below_ground: bool = False) -> list[str]:
    """Every reason *replay* should be rejected from the dataset -- see the
    plan's numbered "verify.check rejects when" list. Empty means clean.

    Every reason is collected, not just the first: a draft a subagent is
    fixing benefits from seeing every problem ``rejects.jsonl`` can name in
    one pass rather than one refusal at a time across repeated
    ``build.py`` runs.
    """
    reasons: list[str] = []

    # Rule 1a: the scene-setup batch (edit/query's own ``prior``) refused.
    if replay.prior_result is not None and replay.prior_result.get("isError"):
        reasons.append(f"prior batch refused: {_refusal_text(replay.prior_result)}")

    # Rule 1b: the record's own batch refused or stopped partway.
    if replay.main_result.get("isError"):
        structured = replay.main_result.get("structuredContent") or {}
        stopped_at = structured.get("stopped_at")
        results = structured.get("results") or []
        detail = (
            _refusal_text(results[stopped_at])
            if isinstance(stopped_at, int) and 0 <= stopped_at < len(results)
            else _refusal_text(replay.main_result)
        )
        reasons.append(f"batch stopped at entry {stopped_at}: {detail}")

    # Rule 2: nothing was built at all.
    doc = replay.doc
    if doc is None or len(doc.objects) == 0:
        reasons.append("no objects were created")
        return reasons  # nothing below has anything to look at

    # Rule 3a: clay/diagnose.findings, direct -- the same function
    # tests/test_agent_transcripts.py's own replay grades against, in
    # creation order.
    for obj in doc.objects:
        rows = clay_diagnose.findings(obj.mesh)
        if rows:
            kinds = sorted(row.kind for row in rows)
            reasons.append(f"{obj.name!r} has diagnose findings: {kinds}")

    # Rule 3b: the same claim, through the tool a real agent would call --
    # a divergence between the two doors is a defect worth seeing, not
    # something to trust blind.
    diag_result = agent_clay.call(replay.ctx, replay.session, "clay_diagnose", {})
    if diag_result.get("isError"):
        reasons.append(f"clay_diagnose refused: {_refusal_text(diag_result)}")
    else:
        for row in (diag_result.get("structuredContent") or {}).get("objects", []):
            if not row.get("clean", False):
                reasons.append(f"{row.get('name')!r} is not clean per clay_diagnose")

    scene_payload = replay.scene or {}

    # Rule 4a: the document-wide bounds, from clay_scene's own payload.
    bounds = scene_payload.get("bounds")
    if bounds is None:
        reasons.append("clay_scene reports no bounds (nothing visible)")
    else:
        for i, s in enumerate(bounds.get("size") or []):
            if s < MIN_DOC_SIZE or s > MAX_DOC_SIZE:
                reasons.append(
                    f"document bounds size[{i}]={s} outside [{MIN_DOC_SIZE}, {MAX_DOC_SIZE}]"
                )

    # Rule 4b/5: per-object degenerate size, and the named ground check.
    for row in scene_payload.get("objects", []):
        name = row.get("name")
        size = row.get("size")
        if size is not None:
            for i, s in enumerate(size):
                if s <= MIN_COMPONENT:
                    reasons.append(
                        f"{name!r} has a near-zero size component (size[{i}]={s})"
                    )
        if not allow_below_ground:
            reasons.extend(_ground_reasons_for(row))

    # Rule 6: the document survives a real serialization round trip.
    try:
        restored = serialize.read_wblk(serialize.wblk_bytes(doc))
    except Exception as exc:  # pragma: no cover - defensive; see docstring
        reasons.append(f"serialize round-trip raised: {exc!r}")
    else:
        if len(restored.objects) != len(doc.objects):
            reasons.append(
                "serialize round-trip object count mismatch: "
                f"{len(doc.objects)} -> {len(restored.objects)}"
            )

    # Rule 7: an edit that did not move the history head is not an edit.
    if (
        replay.prior_result is not None
        and not replay.prior_result.get("isError")
        and replay.head_before_main is not None
        and replay.head_after_main == replay.head_before_main
    ):
        reasons.append("edit made no change: doc.history.head did not move")

    return reasons


def _ground_reasons_for(row: dict[str, Any]) -> list[str]:
    """Rule 5, factored out to a name of its own -- so a family that
    legitimately builds below the ground plane (an in-progress excavation, a
    submerged prop) can opt out of exactly this check with
    ``record["allow_below_ground"] = true``, per the plan, without also
    losing every other check."""
    bbox = row.get("bbox")
    if bbox is None:
        return []
    min_y = bbox["min"][1]
    if min_y < GROUND_EPS:
        return [f"{row.get('name')!r} sits below the ground plane (bbox min y={min_y})"]
    return []
