"""Clay's agent tool surface, the on-demand catalogue lookup: ``clay_catalog``
alone -- the catalogue diet's own tool (``dev/CLAY-PLAN.md`` tranches 6/7's
integration brief).

Before this, five registries' worth of generated prose (every ``clay_op``
row's params, every generator's defaults, every figure's part names, every
``clay_select_by`` query's arguments, every modifier kind's params) sat
inline in a tool's own *description*, which every agent session pays for at
``tools/list`` whether or not it ever calls that tool. ``clay_catalog`` moves
that prose behind a call: the same nine registries (those five, plus
tranche 6/7's own collider kinds, validate profiles, export engines and uv
actions) are still the single source of truth -- :data:`~.schema.CATALOG_TOPICS`
maps a topic straight onto the same builder function a tool description used
to call directly -- only *when* the prose is generated moves, from every
``tools/list`` to the one call that actually wants it.

**No document, no tab, no undo step.** Every registry this reaches
(``clay_ops.OPS``, ``primitives.GENERATORS``, ``presets.ASSEMBLIES``,
``select.QUERIES``, ``modifiers.MODIFIERS``, ``colliders.COLLIDER_KINDS``,
``readiness.PROFILES``, ``engines.ENGINES``, :data:`~.schema.UV_ACTIONS`) is
process-wide, not document state, so this is the one tool in the whole fold
that never calls :func:`~.validate._tab` -- it works with no document open at
all, the same way a reference book does not need the thing it describes open
in front of you first.
"""

from __future__ import annotations

from typing import Any

from .schema import CATALOG_TOPICS
from .validate import Session, _json, fail


def _h_catalog(ctx: Any, session: Session, args: dict) -> dict:
    """The full generated prose for one :data:`~.schema.CATALOG_TOPICS`
    topic. Needs no document -- see this module's own docstring."""
    del ctx, session
    topic = args.get("topic")
    builder = CATALOG_TOPICS.get(topic)
    if builder is None:
        return fail(
            f"topic must be one of {', '.join(sorted(CATALOG_TOPICS))}.", field="topic"
        )
    return _json({"topic": topic, "catalogue": builder()})
