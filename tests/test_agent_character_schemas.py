"""What ``agent_character``'s tool schemas declare, and whether the handler
behind each one actually enforces it -- the same walk
``tests/test_agent_schemas.py`` runs over Clay's own 26 tools, aimed at
these ten instead.

**Simpler than Clay's own walk, and honestly so.** Clay's schemas use
``anyOf``, ``exclusiveMinimum`` and a handful of per-property baseline
overrides (a query-conditional argument, a compare-only argument) that this
surface's own schemas never do -- every property here is read
unconditionally by its handler, so **one baseline per tool** (this file's
own simplification, stated once rather than per test) is enough. The
keywords walked are exactly Clay's own tracked set, minus the two this
surface never declares (``anyOf``, ``exclusiveMinimum``): ``type``,
``additionalProperties``, ``required``, ``minItems``, ``enum``,
``maxItems``, ``minimum``, ``maximum``. ``pattern``/``minLength``/
``maxLength`` are declared on a few properties (``palette``, ``name``,
``cursor``, ``prompt``) but -- like Clay's own schemas, whose walk does not
track them either -- are not violated here: nothing on this surface (or
Clay's) enforces a pattern or a length at the door, so there is nothing
this walk could prove by sending one.

**Every case asserts a specific ``field`` and a non-generic message**, not
merely ``isError`` -- mirroring ``tests/test_agent_schemas.py::_run_case``'s
own rule for Clay. :func:`agent_character.call`'s blanket ``except
Exception`` would guarantee *some* refusal even for a case nothing was
written to expect, so ``isError`` alone cannot tell "the handler (or fix 4's
own pre-check in :func:`agent_character._structural_refusal`) refused this
on purpose" from "this fell through to the generic 'failed unexpectedly'
catch-all" -- which is exactly the gap fix 4 exists to close (a missing
required argument or a wrong-typed nested item used to fall through to that
catch-all; now :func:`agent_character._structural_refusal` refuses it
before any handler runs). Every violation here names its top-level
property as ``field`` -- this surface's own handlers, like Clay's, validate
a whole top-level argument at once and never a deeper per-index sub-field
(a movement's own bad name inside ``movements[]`` is still reported as
``field="movements"``), so the expected field is always the *first* segment
of the case's own path; the one exception is a root ``additionalProperties``
violation, which names the unexpected key itself
(``agent_character._unknown_argument_refusal``'s own rule), so that case's
recorded "path" is the injected key's name rather than empty.

Because fix 4 now validates a declared ``type`` generically for *every*
property (not only ones that also carry an ``enum``/``minimum``/
``maximum``, the previous restriction here), the type-violation cases below
cover every leaf that declares a ``type`` -- including an array item
declared as an object (``movements[0]``) replaced by a plain scalar, which
is fix 4's own "wrong-typed nested movement item" case.

S1's own doors are real on this branch now (see
``tests/test_agent_character.py``'s own module docstring for the same
finding); this file still stubs every door a baseline needs to succeed
rather than reach for the real ones, since nothing here needs a door to
actually validate a value -- that is this file's own job, and letting the
real doors run (spawning Blender, needing a real rendered sheet on disk)
would make this walk slow for no benefit to what it is checking.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from warlock.studio import agent_character as ac

Args = dict[str, Any]
Mutate = Callable[[Args], Args]


# --- S1 stand-ins and door stubs, args-independent -------------------------


@pytest.fixture(autouse=True)
def _stub_doors(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from warlock import rigging
    from warlock.service import _jobs_lifecycle
    from warlock.service import characters as svc_characters
    from warlock.service import export as svc_export
    from warlock.service import rig as svc_rig
    from warlock.service import troupe as svc_troupe

    # rigging.shipped_clip_templates/shipped_clip_library/shipped_clip_names
    # and clips.shipped_clip_timing are real on this branch (see
    # tests/test_agent_character.py's own module docstring for the same
    # finding) and none of this file's own tests call
    # rigging.set_user_clip_dir, so there is no user library for a
    # user-first door to disagree with the shipped one about -- a swap onto
    # the user-first rigging.clip_library/clips.clip_timing here would be a
    # pure no-op. Nothing left in this fixture needs one.
    monkeypatch.setattr(
        svc_characters,
        "ASSET_FILTERS",
        ("all", "riggable", "rigged", "has_sheets"),
        raising=False,
    )

    monkeypatch.setattr(
        svc_characters,
        "list_character_assets",
        lambda svc, **kw: {"assets": [], "next_cursor": None},
        raising=False,
    )
    monkeypatch.setattr(
        svc_characters,
        "recipe_from_prompt",
        lambda svc, prompt, *, overrides=None: {
            "recipe": {"family": "stub"},
            "resolution": {},
            "ignored": [],
            "cells": 1,
            "estimate_minutes": 0.1,
        },
        raising=False,
    )
    monkeypatch.setattr(
        svc_characters,
        "create_character",
        lambda svc, recipe, *, name=None, prompt="", resolution=None: {
            "id": "aaaaaaaaaaaa",
            "rig": "bbbbbbbbbbbb",
            "kind": "character",
        },
    )
    monkeypatch.setattr(rigging, "read_rig", lambda job_dir: None)
    monkeypatch.setattr(svc_rig, "rig_in_flight", lambda svc, jid: None, raising=False)
    monkeypatch.setattr(
        svc_rig,
        "create_rig",
        lambda svc, jid, *, template=None: {
            "id": "cccccccccccc",
            "source_job": jid,
            "template": template or "humanoid",
        },
    )
    monkeypatch.setattr(
        svc_troupe,
        "send_to_troupe",
        lambda svc, jid, **kw: {
            "id": "dddddddddddd",
            "source_job": jid,
            "sheet_id": "eeeeeeeeeeee",
        },
    )
    monkeypatch.setattr(
        svc_characters,
        "character_job",
        lambda svc, jid: {"job_id": jid, "sheets": []},
        raising=False,
    )
    monkeypatch.setattr(
        svc_characters,
        "sheet_preview_png",
        lambda svc, jid, sid, **kw: (b"\x89PNG\r\n", {"width": 1, "height": 1}),
        raising=False,
    )
    monkeypatch.setattr(
        svc_export,
        "run_character_export",
        lambda svc, key, jid, sid=None, *, stem=None: {
            "copied": 1,
            "dir": str(tmp_path),
            "degraded": [],
        },
    )
    monkeypatch.setattr(_jobs_lifecycle, "cancel_job", lambda svc, jid: {"ok": True})


def _mint_model_job(svc: Any) -> str:
    import uuid

    job_id = uuid.uuid4().hex[:12]
    svc.store.create("image", "a knight", {}, job_id, stage="model", status="done")
    return job_id


# --- one baseline per tool --------------------------------------------------


def _baseline(tool_name: str, svc: Any, e: Any) -> tuple[Any, Args]:
    """``(session, args)`` -- a call this tool's handler accepts as-is.
    ``svc.config.export_dir`` is set here (once, on the shared fixture's own
    ``svc``) rather than per-baseline, since only ``character_export``
    reads it."""
    if tool_name == "character_options":
        return ac.Session(), {}
    if tool_name == "character_assets":
        return ac.Session(), {}
    if tool_name == "character_clips":
        return ac.Session(), {"template": e.sheet_templates[0]}
    if tool_name == "character_create":
        return ac.Session(), {
            "family": e.families[0],
            "movements": [{"name": e.movements[0]}],
        }
    if tool_name == "character_rig":
        return ac.Session(), {"job_id": _mint_model_job(svc)}
    if tool_name == "character_sheet_create":
        return ac.Session(), {
            "job_id": _mint_model_job(svc),
            "movements": [{"name": e.movements[0]}],
        }
    if tool_name == "character_job":
        return ac.Session(), {"job_id": "a" * 12}
    if tool_name == "character_sheet_preview":
        return ac.Session(), {"job_id": "a" * 12, "sheet_id": "b" * 12}
    if tool_name == "character_export":
        # agent_export_stem (real on this branch) calls svc.require_job
        # before run_character_export is ever reached, even though that
        # door is stubbed above -- a bare "a"*12 with no row behind it
        # refuses on "no such job" before the schema walk gets to exercise
        # anything.
        return ac.Session(), {"job_id": _mint_model_job(svc), "format": e.formats[0]}
    if tool_name == "character_cancel":
        session = ac.Session()
        session.minted["a" * 12] = "model"
        return session, {"job_id": "a" * 12}
    raise AssertionError(f"no baseline for {tool_name!r}")  # pragma: no cover


def _tool_names() -> list[str]:
    return sorted(ac.HANDLERS)


@pytest.mark.parametrize("tool_name", _tool_names())
def test_every_baseline_is_itself_accepted(tool_name: str, svc: Any) -> None:
    svc.config.export_dir = svc.config.data_dir / "export"
    e = ac._enums()
    session, args = _baseline(tool_name, svc, e)
    result = ac.call(svc, session, tool_name, args)
    assert result["isError"] is False, result


# --- the discovery-and-exercise walk ----------------------------------------


def _nav(root: Any, path: tuple) -> Any:
    node = root
    for p in path:
        node = node[p]
    return node


def _path_str(path: tuple) -> str:
    return "/".join(str(p) for p in path)


_TYPE_VIOLATIONS = {
    "string": 12345,
    "integer": "not-an-int",
    "number": "not-a-number",
    "boolean": "not-a-bool",
    "object": "not-an-object",
    "array": "not-an-array",
}


def _cases_for_tool(
    tool_name: str, schema: dict, movements_vocab: tuple[str, ...]
) -> list[tuple[str, Mutate, str]]:
    """Every case, as ``(case_id, mutate, expected_field)``. ``expected_field``
    is always the violated argument's *top-level* property name -- see the
    module docstring's own paragraph on why that holds even for a violation
    reached through a nested path (``movements[0].name``'s own enum still
    refuses on ``field="movements"``) -- except the one root
    ``additionalProperties`` case, whose expected field is the unknown key
    this walk injects rather than any property the schema declares at all.
    """
    cases: list[tuple[str, Mutate, str]] = []

    def add(case_id: str, mutate: Mutate, expected_field: str) -> None:
        cases.append((f"{tool_name}:{case_id}", mutate, expected_field))

    def walk(node: dict, path: tuple) -> None:
        top = path[0] if path else None
        for key in node.get("required", []):

            def mutate_req(args: Args, path=path, key=key) -> Args:
                root = copy.deepcopy(args)
                container = _nav(root, path)
                if isinstance(container, dict):
                    container.pop(key, None)
                return root

            add(f"{_path_str(path + (key,))}:required", mutate_req, top if top else key)

        if not path and node.get("additionalProperties") is False:
            # Root only: call()'s own unknown-argument gate (mirrored from
            # agent_clay's) checks a call's *top-level* argument names --
            # nothing on this surface (or Clay's own) checks a nested
            # object's extra keys, so a movement item's own
            # "additionalProperties": false is declared but structurally
            # unenforced, the same shape of gap
            # tests/test_agent_schemas.py documents for its own three
            # schema-valued additionalProperties nodes.

            def mutate_extra(args: Args, path=path) -> Args:
                root = copy.deepcopy(args)
                container = _nav(root, path)
                if isinstance(container, dict):
                    container["__unknown_extra__"] = True
                return root

            add(f"{_path_str(path)}:additionalProperties", mutate_extra, "__unknown_extra__")

        if path and "enum" in node:

            def mutate_enum(args: Args, path=path) -> Args:
                root = copy.deepcopy(args)
                parent = _nav(root, path[:-1])
                parent[path[-1]] = "__not_a_real_enum_value__"
                return root

            add(f"{_path_str(path)}:enum", mutate_enum, top)

        if path and "minimum" in node:

            def mutate_min(args: Args, path=path, node=node) -> Args:
                root = copy.deepcopy(args)
                parent = _nav(root, path[:-1])
                parent[path[-1]] = node["minimum"] - 1
                return root

            add(f"{_path_str(path)}:minimum", mutate_min, top)

        if path and "maximum" in node:

            def mutate_max(args: Args, path=path, node=node) -> Args:
                root = copy.deepcopy(args)
                parent = _nav(root, path[:-1])
                parent[path[-1]] = node["maximum"] + 1
                return root

            add(f"{_path_str(path)}:maximum", mutate_max, top)

        # Fix 4 (agent_character._structural_refusal) type-checks every
        # declared property generically now, not only ones that also carry
        # an enum/minimum/maximum -- so every leaf (and every array item
        # declared as an object, e.g. movements[0] itself) that names a
        # "type" gets a violation case, including job_id, cursor, prompt,
        # appearance, dither, pixel_art, palette, name and a movement item
        # replaced by a plain scalar.
        if path and node.get("type") in _TYPE_VIOLATIONS:

            def mutate_type(args: Args, path=path, node=node) -> Args:
                root = copy.deepcopy(args)
                parent = _nav(root, path[:-1])
                parent[path[-1]] = _TYPE_VIOLATIONS[node["type"]]
                return root

            add(f"{_path_str(path)}:type", mutate_type, top)

        if node.get("type") == "array":
            if "minItems" in node:

                def mutate_min_items(args: Args, path=path) -> Args:
                    root = copy.deepcopy(args)
                    parent = _nav(root, path[:-1])
                    parent[path[-1]] = []
                    return root

                add(f"{_path_str(path)}:minItems", mutate_min_items, top)

            if "maxItems" in node:
                cap = node["maxItems"]
                vocab = movements_vocab or ("x",)

                def mutate_max_items(args: Args, path=path, cap=cap, vocab=vocab) -> Args:
                    root = copy.deepcopy(args)
                    parent = _nav(root, path[:-1])
                    parent[path[-1]] = [
                        {"name": vocab[i % len(vocab)]} for i in range(cap + 1)
                    ]
                    return root

                add(f"{_path_str(path)}:maxItems", mutate_max_items, top)

            items_schema = node.get("items")
            if isinstance(items_schema, dict):
                walk(items_schema, path + (0,))

        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            for key, sub in node["properties"].items():
                walk(sub, path + (key,))

    walk(schema, ())
    return cases


def _all_cases() -> list[tuple[str, Mutate, str]]:
    # No live service needed to build the schemas themselves -- _enums()
    # reads only module-level registries. Movements is read straight off
    # the shipped clip vocabulary so the fixture above need not be active
    # at collection time.
    from warlock import rigging

    templates = tuple(
        row["key"] for row in rigging.catalog() if rigging.clip_library(row["key"]).get("clips")
    )
    movements_vocab = tuple(
        sorted(
            {
                n
                for t in templates
                for n in (c["name"] for c in rigging.clip_library(t)["clips"])
            }
        )
    )
    cases: list[tuple[str, Mutate, str]] = []
    for tool in ac.tools():
        cases.extend(_cases_for_tool(tool.name, tool.schema, movements_vocab))
    return cases


_ALL_CASES = _all_cases()

#: Cases whose refusal cannot honestly name a ``field`` -- see each entry's
#: own reason. Kept as a named, deliberate list (mirroring
#: ``tests/test_agent_schemas.py``'s own ``_FIELD_NOT_ASSERTED``) rather than
#: a blanket exception, so a case added here later still has to justify
#: itself.
_FIELD_NOT_ASSERTED: frozenset[str] = frozenset()


def test_the_exercise_walk_attempts_a_pinned_number_of_cases() -> None:
    """Pinned so a schema edit that silently drops a case from the walk is
    caught here rather than only by a shrinking count nobody notices --
    see ``tests/test_agent_schemas.py``'s identical rule for Clay. Bigger
    than before this file's own fix-4 pass: the type-violation case now
    fires for every property that declares a ``type`` (previously only
    ones that also carried an ``enum``/``minimum``/``maximum``), since fix
    4's structural pre-check enforces all of them now, not just those.

    116 -> 118 (2026-09-13): master's 8b091e98 Send to Troupe widened
    ``size`` to any whole number in ``service.troupe.TROUPE_CUSTOM_SIZE_
    RANGE``, so ``character_create``'s and ``character_sheet_create``'s own
    ``size`` property dropped its ``enum`` case and gained ``minimum`` and
    ``maximum`` ones instead -- net +1 case on each of the two tools that
    declare it."""
    assert len(_ALL_CASES) == 118, (
        f"the walk now attempts {len(_ALL_CASES)} cases, not 118. If this "
        f"growth is deliberate (a tool or a constraint was added), update "
        f"the pinned number in this test and say why in the same commit."
    )


def test_every_documented_field_exception_corresponds_to_a_real_case() -> None:
    """:data:`_FIELD_NOT_ASSERTED` names real cases, not stale ones --
    mirrors ``tests/test_agent_schemas.py``'s identical check."""
    ids = {c[0] for c in _ALL_CASES}
    for case_id in _FIELD_NOT_ASSERTED:
        assert case_id in ids, case_id


@pytest.mark.parametrize(
    "case_id,mutate,expected_field", _ALL_CASES, ids=[c[0] for c in _ALL_CASES]
)
def test_every_declared_constraint_is_enforced(
    case_id: str, mutate: Mutate, expected_field: str, svc: Any
) -> None:
    """Every violation refuses -- with a field pointing at the argument it
    actually broke, and a message that is not the generic "failed
    unexpectedly" catch-all, proving a handler or fix 4's own pre-check
    refused this *on purpose* rather than the call merely raising something
    :func:`agent_character.call`'s blanket ``except Exception`` happened to
    catch. See the module docstring for why ``expected_field`` is always
    the violated argument's top-level name."""
    svc.config.export_dir = svc.config.data_dir / "export"
    tool_name = case_id.split(":", 1)[0]
    e = ac._enums()
    session, baseline_args = _baseline(tool_name, svc, e)
    mutated = mutate(baseline_args)
    result = ac.call(svc, session, tool_name, mutated)
    assert result["isError"] is True, (case_id, mutated, result)
    text = result["content"][0]["text"]
    assert "failed unexpectedly" not in text, (case_id, text)
    if case_id not in _FIELD_NOT_ASSERTED:
        structured = result.get("structuredContent") or {}
        assert structured.get("field") == expected_field, (case_id, result)
