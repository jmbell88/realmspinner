"""``create_job``'s asset-type door agrees with Create's own registry.

The 2026-09-18 audit, finding service-05: ``create_job``'s ``asset_type`` and
``asset_intent`` whitelists (``service/_jobs_create.py``) are a hand copy of
``studio/modes/create/engine/assets.py``'s ``ASSET_TYPES`` registry -- the
single source Create's UI, the library and Review all read from -- with
nothing tying the two together. Inspected at the time of the finding, they
agreed; nothing stopped a future entry added to one from silently going
unrecognised (or over-permissive) at the other.

``create_job``'s two sets are literals inside the function body, not module
constants, so this reads them the way ``inspect``/``ast`` would for any
other undocumented literal: pull the source, find the ``not in {...}``
checks by name, and evaluate the set each one actually guards. That is the
whitelist this door *enforces*, not a second hand-copy of it.
"""

from __future__ import annotations

import ast
import inspect

from warlock.service import _jobs_create
from warlock.studio.modes.create.engine import assets


def _door_whitelist(name: str) -> set[str]:
    """The literal set ``create_job`` refuses ``name`` to fall outside of."""
    tree = ast.parse(inspect.getsource(_jobs_create.create_job))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Name) and left.id == name):
            continue
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            if isinstance(op, ast.NotIn) and isinstance(comparator, ast.Set):
                return set(ast.literal_eval(comparator))
    raise AssertionError(f"create_job no longer guards {name!r} with a `not in {{...}}` literal")


# "character" is the one registry entry deliberately excluded from create_job's
# door -- see assets.py's own comment on the "character" AssetType: its
# ``output`` is not one of create_job's three arms, precisely so create_job
# can never reach it.
_NOT_A_CREATE_JOB_TYPE = {"character"}


def test_create_job_asset_type_whitelist_matches_the_create_registry():
    door = _door_whitelist("asset_type")
    registry = set(assets.ASSET_TYPES) - _NOT_A_CREATE_JOB_TYPE
    assert door == registry, (
        f"create_job's asset_type whitelist has drifted from "
        f"create/engine/assets.ASSET_TYPES: only in create_job {door - registry}, "
        f"only in the registry {registry - door}"
    )


def test_create_job_asset_intent_whitelist_matches_the_create_registry():
    door = _door_whitelist("asset_intent")
    registry = {spec.intent for spec in assets.ASSET_TYPES.values()} - _NOT_A_CREATE_JOB_TYPE
    assert door == registry, (
        f"create_job's asset_intent whitelist has drifted from "
        f"create/engine/assets.ASSET_TYPES' intents: only in create_job "
        f"{door - registry}, only in the registry {registry - door}"
    )
