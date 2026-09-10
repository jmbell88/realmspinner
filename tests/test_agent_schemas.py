"""What Clay's agent tool schemas declare, and whether the handler behind
each one actually enforces it.

Every one of the 25 tools :func:`agent_clay.tools` publishes carries a real
JSON Schema -- ``type``, ``enum``, ``minimum``/``maximum``, ``minItems``/
``maxItems``, ``required``, ``additionalProperties``, ``anyOf``,
``exclusiveMinimum``. **Nothing validates any of it at the door.**
``mcp/protocol.py``'s own module docstring is explicit that it knows nothing
about Clay, and ``tools/call`` there checks only that ``arguments`` is a
JSON object before handing it to :func:`agent_clay.call` -- a NaN, a string
where a number was declared, a fourth element in a three-element vector, all
arrive exactly as an agent typed them. That is a deliberate decision, not an
oversight, and it is not being revisited here: a validator bolted onto that
leaf would have had to learn the ``anyOf``/``minimum``/``maximum``/
``exclusiveMinimum`` shapes these schemas actually use, which means teaching
a module whose own rule is "knows nothing about Clay" what Clay's own
argument shapes are. The one exception is an argument's *own name* --
``agent_clay.call`` now refuses a key a tool's schema does not declare in
``properties`` before any handler runs (see that module's docstring and
``tests/test_agent_clay.py``'s unknown-argument section) -- but everything
about a *value* is still the handler's own job, exactly as it always was.

**This file is what stands in for the validator that was never built.** It
does not hand-list which constraints exist -- that would be the exact drift
class ``agent_clay.py``'s own module docstring warns about for a hand-kept
second copy of something a real structure already owns. Instead it walks
:func:`agent_clay.tools`'s own schemas, discovers every declared constraint
from them, and for each one synthesises a call that satisfies everything
else about a tool's arguments and violates exactly that one thing, then
asserts the tool refuses. A tool that gains a constraint tomorrow is covered
here with nobody having to remember to extend a list for it.

**What "declared constraint" means, precisely.** Walking every tool's
``.schema`` recursively and counting occurrences of ``type``,
``additionalProperties``, ``properties``, ``items``, ``required``,
``minItems``, ``enum``, ``maxItems``, ``minimum``, ``maximum``, ``anyOf`` and
``exclusiveMinimum`` reproduces an independently measured count exactly:
143/31/28/29/20/18/16/14/13/10/3/1 respectively (326 total) -- see
``test_the_discovery_walk_finds_every_measured_constraint_marker`` below,
which pins that reproduction so this file's own claim about how much ground
it covers is checked rather than asserted. Two of those twelve keywords,
``properties`` and ``items``, are never violated directly -- they exist only
to route recursion into a nested object's fields or an array's element
shape, so this file's *discovery-and-exercise* walk (:func:`_walk_tool_schema`)
counts them as structural rather than as constraints with a violation of
their own -- and a third marker joins them for the identical reason: three
of the 31 ``additionalProperties`` occurrences are not ``false`` but a
*schema* (``clay_add_primitive``/``clay_set_params``/``clay_op``'s own
``params``, an open-ended object whose keys are never named in
``properties``), so those three route recursion into that open-ended shape
rather than being violated themselves either -- see
:data:`_OPEN_ENDED_PARAMS_TOOLS`. A fourth kind of marker is excluded for a
different reason: each of the 25 tools' own root ``"type": "object"`` is
never a case, because ``agent_clay.call`` only ever reaches a handler with
``arguments`` already a dict -- there is nothing there for a schema's own
root type to promise that is not already true by construction. What
survives after subtracting those (28 ``properties`` + 29 ``items`` + 3
schema-valued ``additionalProperties`` + 25 root ``type``) is 241 violable
markers; ``required``'s remaining 20 occurrences are *lists*, each naming
one or more keys -- 27 individual keys between them, one violation apiece
rather than one per list -- which nets the walk's own exercise total to
**248** concrete violation attempts (241 - 20 + 27), pinned by
``test_the_exercise_walk_attempts_exactly_the_documented_number_of_cases``
so a schema edit that silently drops a case from the walk is caught here
rather than only by a shrinking "exercised" count nobody happens to notice.

**Coverage, honestly.** Every one of the 242 is attempted. What cannot
reliably be asserted the same way for every one is which *field* a refusal
names -- several constraints are enforced by a handler that refuses for a
different, still-correct, reason before it would ever reach the check this
file is nominally exercising (an out-of-range ``uid`` refuses at
``_resolve_uid`` before a sibling argument's own shape is even looked at,
say). Per this file's own rule (below), those are still counted as
*exercised*, because the schema's promise -- a caller sending this will be
refused -- is still kept; only the stronger claim, that the refusal names
this exact field, is dropped for those specific cases, and
:data:`_FIELD_NOT_ASSERTED` names every one with its reason rather than
silently downgrading the assertion. Nothing in this file's exercise loop is
weakened to "something refused" across the board -- every case asserts
``isError`` at minimum, and asserts ``field`` too unless the case is listed
in :data:`_FIELD_NOT_ASSERTED`.

**Several tools' properties are only ever read under a condition one plain
baseline cannot meet at the same time as every other property** -- not
"unreachable", just needing a *different* valid baseline for that one
property, routed to by :data:`_PROPERTY_OVERRIDES`:

* ``clay_select_by``'s seven query-argument schemas (``_QUERY_ARG_SCHEMAS``)
  are all declared as top-level properties of one schema, but which ones a
  given call actually *needs* depends on ``query`` -- a ``slot`` given to a
  ``loop`` query is simply never read. :data:`_SELECT_BY_BASELINES` gives
  ``edge``/``face``/``slot``/``direction``/``max_angle`` each the query that
  actually consults it, rather than the ``bounds`` query the main
  :func:`_b_select_by` baseline uses for ``uid``/``query``/``how``/
  ``expect_stamp``/``min``/``max``/``space``.
* ``clay_render``'s ``view`` conflicts with ``views`` (the main baseline's
  choice) if simply added alongside it, so it gets its own baseline
  (:func:`_b_render_view`); ``compare_mode`` and ``alpha`` are only read once
  ``compare`` names a real reference, and ``alpha`` only in
  ``compare_mode="overlay"`` -- :func:`_b_render_compare` gives all three a
  baseline where they are.
* ``clay_reference_add``'s ``file`` is only read on the ``job_id`` branch,
  never alongside ``png_base64`` (the main baseline's choice) --
  :func:`_b_reference_add_job` mirrors ``tests/test_agent_clay.py``'s own
  ``test_reference_add_from_a_library_job_reads_its_input_png`` fixture to
  give ``job_id`` and ``file`` both a baseline that actually reads them.

**What genuinely is not reachable here, and why**, matching the shape of gap
this module's own docstring asks for rather than a silent skip:

* ``clay_render`` needs a real moderngl context to build its private
  viewport. ``tests/test_agent_clay.py``'s ``_install_fake_view`` (imported
  from there rather than copied -- see the note below) swaps in a fake
  ``ClayView`` that returns a real tiny PNG with no GL at all, which is
  enough to reach every argument-validation refusal in ``_h_render`` --
  every one of those checks runs *before* ``_view_for(ctx)`` is ever called.
  Real pixels from a real GPU are still not exercised by this file, the same
  limit ``test_agent_clay.py`` already documents for itself.
* ``clay_export`` declares no properties at all beyond the top-level
  ``additionalProperties: false`` every tool carries -- there is nothing
  else in its schema to violate, so its own real-service dependency
  (``clay_mode.build_asset``) is never reached by this file: refusing an
  unknown top-level key happens in :func:`agent_clay.call` before any
  handler runs, real service or not.
* An op's own declared parameter *bounds* (``Param.low``/``Param.high`` in
  ``clay_ops.py``) are not part of the JSON schema at all -- ``clay_op``'s
  schema only declares ``params`` an object of numbers, with no per-key
  range, so there is no declared range constraint here to check. What the
  schema *does* declare (each value is a number) is exercised through
  :data:`_OPEN_ENDED_PARAMS_TOOLS`, the same way ``clay_add_primitive`` and
  ``clay_set_params``'s own ``params`` are.

**A cross-test-module import, the first one in this suite.** ``tests/`` has
no ``__init__.py`` and nothing else here imports from a sibling test module
-- but pytest's default rootdir-relative import mode puts ``tests/`` on
``sys.path`` for any test file collected from it, which makes
``from test_agent_clay import ...`` work with no package boilerplate. Reused
rather than copied: ``_Ctx`` (a ctx double with no imgui, no GL, no pygame),
``_payload`` (unwraps a successful call's JSON), and ``_install_fake_view``
(the fake ``ClayView`` for ``clay_render``). Everything else this file needs
-- the baseline argument sets themselves -- has no equivalent there to reuse,
because ``test_agent_clay.py``'s own ``_NEEDS_A_TAB``/``_MINTS_A_TAB`` are
deliberately *minimal* (just enough to reach a tab check), never *valid*
arguments a constraint violation could be measured against.

**Real findings this file's first run turned up, and what was done about
each** -- every one a case where a schema declared something the handler did
not actually enforce, fixed rather than the test being weakened to match:

* ``clay_render``'s ``size`` (``minimum: 64, maximum: 2048``) was *clamped*
  into range rather than refused outside it -- a caller asking for 999999
  silently got 2048 back with no way to tell the ceiling it was told about
  was real. Its ``grid`` (declared a boolean) was coerced with a bare
  ``bool(grid)`` -- ``bool("off")`` is ``True``, which drew the grid an
  agent's own value looked like it was asking not to see -- and its
  ``alpha`` (0..1) was only ever ``float()``-cast, with neither bound nor
  finiteness checked, the identical unvalidated-number hole
  ``clay_material``'s colour and metallic/roughness already had before an
  earlier commit closed it with :func:`agent_clay._validate_unit`.
* ``_QUERY_ARG_SCHEMAS["slot"]``'s ``minimum: 0`` and ``["max_angle"]``'s
  ``minimum: 0.0``/``maximum: 180.0`` were not checked at all -- a negative
  palette slot or an angle past 180 degrees passed straight through to the
  query function and simply matched nothing, a silent no-op standing in for
  the refusal the schema promised.
* ``clay_diagnose``'s ``select`` sub-object and ``clay_batch``'s own
  ``calls[]`` entries both declare ``additionalProperties: false``; neither
  was checked, so an extra key on either rode along unnoticed.
* ``clay_op``'s ``params`` (declared an object) reached ``**params``
  unpacking with no type check of its own -- a non-dict value did get
  refused, but only via ``call()``'s generic backstop and a logged
  exception, not a clean field-named refusal the way every other bad-shaped
  argument in this file gets. ``clay_add_primitive``'s own ``params`` went
  further: unlike ``clay_set_params``, it never validated a value's shape at
  all before handing it to the generator function, so a non-numeric string
  reached ``bp.box(**merged)`` directly and raised a bare ``TypeError`` two
  frames deep.
* ``clay_add_primitive``'s ``name``, ``clay_add_figure``'s ``name_prefix``
  and ``clay_material``'s ``name`` (all three declared strings) were coerced
  with a bare ``str()`` rather than checked -- ``clay_rename``'s identical
  field already used ``isinstance(name, str)``, so all three were brought in
  line with it.
* ``clay_select``'s ``uids`` is declared required, but ``_resolve_uids``
  alone reads a missing value the same as an explicit empty list (its own
  "clear the selection" meaning) -- so *omitting* the argument entirely was
  never refused, only checked for ahead of that shared helper now.
* ``clay_reference_add``'s ``job_id``, ``file`` and ``png_base64`` (each
  declared a string) all reached real code with no type check first: a
  non-string ``job_id`` raised inside ``service.validation.check_job_id``'s
  regex match, a non-string ``file`` raised inside ``pathlib``'s ``/``
  operator via ``svc_files.ready``, and a non-string ``png_base64`` raised
  inside ``base64.b64decode`` -- three bare exceptions, all caught only by
  ``call()``'s generic backstop.

See ``src/warlock/studio/agent_clay.py``'s own module docstring for the
paragraph this file is cited from, and the same commit's diff for each fix.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from test_agent_clay import _Ctx, _install_fake_view, _payload  # see module docstring

from warlock.studio import agent_clay, clay_mode
from warlock.studio.clay import presets

Args = dict[str, Any]
BaselineFactory = Callable[..., tuple[Any, agent_clay.Session, Args]]
"""A tool's baseline builder: takes an optional ``monkeypatch`` (only
``clay_render``'s uses it) and an optional ``svc`` (only ``clay_export``'s
uses it, since ``clay_mode.build_asset`` needs a real service), and returns a
fresh ``(ctx, session, args)`` every time it is called -- fresh, because a
constraint violation must be measured against a document nothing else has
mutated yet, and several of this file's own baselines mutate the very
document they build (``clay_transform``, ``clay_op`` and the rest all change
something)."""


# --- generic constraint discovery and violation synthesis ---------------------


@dataclass(frozen=True)
class _Case:
    """One declared constraint, discovered from a schema, with a function
    that turns *some* valid baseline arguments into arguments that satisfy
    everything else and violate exactly this one thing."""

    tool: str
    path: tuple[Any, ...]
    keyword: str
    mutate: Callable[[Args], Args]

    @property
    def id(self) -> str:
        path = "/".join(str(p) for p in self.path) or "(root)"
        return f"{self.tool}:{path}:{self.keyword}"


def _navigate(obj: Any, path: tuple[Any, ...]) -> Any:
    for seg in path:
        obj = obj[seg]
    return obj


def _set_at(baseline: Args, path: tuple[Any, ...], value: Any) -> Args:
    """A deep copy of *baseline* with the value at *path* replaced -- the
    generic mutation every scalar/array constraint (``type``, ``enum``,
    ``minimum``, ``maximum``, ``exclusiveMinimum``, ``minItems``,
    ``maxItems``, ``anyOf``) uses. The value already there is never read --
    the replacement is built fresh from the target leaf's own schema, so
    what the sibling branches of an ``anyOf`` happened to hold before does
    not matter."""
    mutated = copy.deepcopy(baseline)
    parent = _navigate(mutated, path[:-1])
    parent[path[-1]] = value
    return mutated


def _delete_at(baseline: Args, path_to_parent: tuple[Any, ...], key: str) -> Args:
    mutated = copy.deepcopy(baseline)
    parent = _navigate(mutated, path_to_parent)
    if key in parent:
        del parent[key]
    return mutated


_PROBE_KEY = "___unexpected_probe_key___"


def _add_extra_at(baseline: Args, path_to_parent: tuple[Any, ...]) -> Args:
    mutated = copy.deepcopy(baseline)
    parent = _navigate(mutated, path_to_parent)
    parent[_PROBE_KEY] = "probe"
    return mutated


def _sample(schema: dict) -> Any:
    """*Some* value satisfying *schema*, with no idea what it will be used
    for -- used only to fill in a leaf this file is not trying to violate
    (an item inside an array whose length is being violated, say), never to
    stand in for a real uid or a real generator name. Every baseline's own
    identity-bearing values come from the per-tool baseline factories below,
    not from here."""
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return _sample(schema["anyOf"][0])
    t = schema.get("type")
    if t == "integer" or t == "number":
        lo = schema.get("minimum")
        if lo is None and "exclusiveMinimum" in schema:
            lo = schema["exclusiveMinimum"] + (1 if t == "integer" else 1.0)
        if lo is None:
            lo = 0
        hi = schema.get("maximum", lo + 10)
        val = lo if hi < lo else (lo + hi) / 2
        return int(val) if t == "integer" else float(val)
    if t == "string":
        return "probe"
    if t == "boolean":
        return True
    if t == "array":
        item_schema = schema.get("items", {"type": "number"})
        n = max(schema.get("minItems", 1), 1)
        return [_sample(item_schema) for _ in range(n)]
    if t == "object":
        props = schema.get("properties", {})
        required = schema.get("required", list(props))
        return {k: _sample(props[k]) for k in required if k in props}
    return None


def _violate_type(schema: dict) -> Any:
    wrong = {
        "integer": "not-a-number",
        "number": "not-a-number",
        "string": 12345,
        "array": "not-an-array",
        "object": "not-an-object",
        "boolean": "not-a-boolean",
    }
    return wrong.get(schema.get("type"), "not-a-number")


def _violate_enum(schema: dict) -> Any:
    enum = schema["enum"]
    if all(isinstance(v, str) for v in enum):
        candidate = "___not_a_real_enum_value___"
        while candidate in enum:
            candidate += "_"
        return candidate
    return "___not_a_real_enum_value___"


def _violate_minimum(schema: dict) -> Any:
    m = schema["minimum"]
    return m - 1 if isinstance(m, int) else m - 1.0


def _violate_maximum(schema: dict) -> Any:
    m = schema["maximum"]
    return m + 1 if isinstance(m, int) else m + 1.0


def _violate_exclusive_minimum(schema: dict) -> Any:
    # Exactly the boundary: exclusiveMinimum means "strictly greater than",
    # so the boundary value itself is the smallest value that violates it.
    return schema["exclusiveMinimum"]


def _violate_min_items(schema: dict) -> list:
    n = schema["minItems"]
    item_schema = schema.get("items", {"type": "number"})
    return [_sample(item_schema) for _ in range(max(n - 1, 0))]


def _violate_max_items(schema: dict) -> list:
    n = schema["maxItems"]
    item_schema = schema.get("items", {"type": "number"})
    return [_sample(item_schema) for _ in range(n + 1)]


def _violate_anyof(schema: dict) -> Any:
    """A value shaped like none of *schema*'s alternatives -- picked by JSON
    type, since every ``anyOf`` in this file's schemas distinguishes its
    branches by ``type`` alone (see ``agent_clay.py``'s own three ``anyOf``
    sites: a param value is a number or an array of numbers, a view is a
    string or an object)."""
    alt_types = {alt.get("type") for alt in schema["anyOf"]}
    for candidate_type, candidate_value in (
        ("null", None),
        ("boolean", True),
        ("integer", 999_999),
        ("string", "___no_branch_matches___"),
        ("array", ["___no_branch_matches___"]),
        ("object", {"___no_branch_matches___": 1}),
    ):
        if candidate_type not in alt_types:
            return candidate_value
    return None  # pragma: no cover -- every anyOf in this file excludes something


def _walk_schema(schema: Any, path: tuple[Any, ...]) -> list[_Case]:
    """Every constraint declared at or under *schema*, as ``(path, keyword,
    mutate)`` cases -- the generic recursion this file's whole claim rests
    on. Recurses through ``properties`` (one path segment per key), ``items``
    (one array-index segment, shared by every element), and ``anyOf`` (no
    segment at all: every alternative describes the *same* value at *path*,
    so a constraint nested inside one alternative is violated by replacing
    the whole leaf at *path*, exactly as a top-level constraint there would
    be). ``tool`` is filled in by the caller, not here, since this function
    has no idea which tool's schema it was handed.
    """
    cases: list[_Case] = []
    if not isinstance(schema, dict):
        return cases

    def add(keyword: str, violate: Callable[[dict], Any]) -> None:
        cases.append(_Case("", path, keyword, lambda b, v=violate: _set_at(b, path, v(schema))))

    if "enum" in schema:
        add("enum", _violate_enum)
    if "minimum" in schema:
        add("minimum", _violate_minimum)
    if "maximum" in schema:
        add("maximum", _violate_maximum)
    if "exclusiveMinimum" in schema:
        add("exclusiveMinimum", _violate_exclusive_minimum)
    if "minItems" in schema:
        add("minItems", _violate_min_items)
    if "maxItems" in schema:
        add("maxItems", _violate_max_items)
    if "type" in schema:
        add("type", _violate_type)
    if "required" in schema:
        for key in schema["required"]:
            cases.append(
                _Case("", path + (key,), "required", lambda b, p=path, k=key: _delete_at(b, p, k))
            )
    if schema.get("additionalProperties") is False:
        cases.append(_Case("", path, "additionalProperties", lambda b: _add_extra_at(b, path)))
    if "anyOf" in schema:
        add("anyOf", _violate_anyof)
        for alt in schema["anyOf"]:
            cases.extend(_walk_schema(alt, path))
    if "properties" in schema:
        for key, sub in schema["properties"].items():
            cases.extend(_walk_schema(sub, path + (key,)))
    if "items" in schema:
        cases.extend(_walk_schema(schema["items"], path + (0,)))
    return cases


def _walk_tool_schema(tool_name: str, schema: dict) -> list[_Case]:
    """:func:`_walk_schema`, started at one tool's root -- the root's own
    ``type: object`` is not a case (every ``arguments`` is already a dict by
    the time :func:`agent_clay.call` runs; there is nothing to violate), but
    its ``required`` and top-level ``additionalProperties: false`` are.
    """
    cases: list[_Case] = []
    for key in schema.get("required", []):
        cases.append(_Case(tool_name, (key,), "required", lambda b, k=key: _delete_at(b, (), k)))
    if schema.get("additionalProperties") is False:
        cases.append(_Case(tool_name, (), "additionalProperties", lambda b: _add_extra_at(b, ())))
    for key, sub in schema.get("properties", {}).items():
        cases.extend(
            _Case(tool_name, c.path, c.keyword, c.mutate) for c in _walk_schema(sub, (key,))
        )
    return cases


# The three tools whose ``params`` is declared ``additionalProperties`` as a
# *schema* (``anyOf`` of number/array-of-number/array-of-arrays, or a bare
# number), not ``false`` -- an open-ended object whose keys are never named
# in ``properties``, so :func:`_walk_schema`'s ``properties``-based recursion
# cannot reach into it on its own. Recursed into here instead, using
# whichever key the tool's own baseline already carries there -- the one key
# already known to be valid, so mutating its value in isolation is a real
# "satisfies everything else" violation rather than inventing a key nothing
# else in the call would recognise. See :func:`_open_ended_params_cases`'s
# own comment for why a *nested* case inside one ``anyOf`` branch reshapes
# that one key's value to a fresh sample of that branch first, rather than
# reusing the baseline's own value for every branch.
_OPEN_ENDED_PARAMS_TOOLS = ("clay_add_primitive", "clay_set_params", "clay_op")


def _open_ended_params_cases(tool_name: str, schema: dict, baseline_args: Args) -> list[_Case]:
    node = schema.get("properties", {}).get("params")
    if not isinstance(node, dict) or not isinstance(node.get("additionalProperties"), dict):
        return []
    params_value = baseline_args.get("params") or {}
    if not params_value:
        return []
    key = sorted(params_value)[0]
    path = ("params", key)
    value_schema = node["additionalProperties"]

    # ``clay_add_primitive``/``clay_set_params``'s own value schema
    # (:func:`agent_clay._params_value_schema`) is an ``anyOf`` of three
    # *honestly different* shapes -- a number, a flat array, an array of
    # arrays -- and the one real key this walk can reuse (the baseline's
    # own, ``size``, a flat array of numbers) only ever satisfies one of
    # them structurally. Reusing that literal value for every branch is
    # exactly what worked while there were two branches of increasing depth
    # (a number has no substructure to walk into, and a flat array's single
    # level of ``items`` always had an index 0 to navigate to) -- it stops
    # working the moment a branch nests two levels deep and the baseline's
    # actual value is only one, because there is nothing at ``size[0][0]``
    # to replace. So each branch's own internal structure is walked against
    # a value freshly built to satisfy *that* branch (:func:`_sample`), not
    # against whichever branch the baseline happened to pick for this key --
    # the branch's own top-level ``type``/``anyOf`` cases still replace the
    # whole value outright and need no pre-existing shape there at all.
    if "anyOf" in value_schema:
        cases = [
            _Case(
                tool_name,
                path,
                "anyOf",
                lambda b: _set_at(b, path, _violate_anyof(value_schema)),
            )
        ]
        for alt in value_schema["anyOf"]:
            for c in _walk_schema(alt, path):
                cases.append(
                    _Case(
                        tool_name,
                        c.path,
                        c.keyword,
                        lambda b, alt=alt, m=c.mutate: m(_set_at(b, path, _sample(alt))),
                    )
                )
        return cases
    return [
        _Case(tool_name, c.path, c.keyword, c.mutate)
        for c in _walk_schema(value_schema, path)
    ]


# --- per-tool baselines --------------------------------------------------------
#
# One *valid* argument set per tool -- not the minimal, tab-reaching shape
# ``test_agent_clay.py``'s own ``_NEEDS_A_TAB``/``_MINTS_A_TAB`` keep, a call
# this file can perturb one field at a time and expect every *other* field to
# still be accepted. A fresh ``(ctx, session, args)`` every call, never a
# shared fixture mutated across cases -- several of these baselines change
# the very document they build (``clay_transform``, ``clay_op``, ``clay_delete``
# and the rest), so reusing one across many perturbation calls would make
# "everything else about this call is valid" stop being true partway through
# the walk.


def _new_world(svc: Any = None) -> tuple[Any, agent_clay.Session, int, int]:
    """A fresh session with two boxes already placed, apart in space so a
    boolean has two closed solids to work with. -> ``(ctx, session, uid1,
    uid2)``."""
    ctx = _Ctx(svc=svc)
    session = agent_clay.Session()
    r1 = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    assert r1["isError"] is False, r1
    uid1 = _payload(r1)["uid"]
    r2 = agent_clay.call(
        ctx, session, "clay_add_primitive", {"generator": "box", "translation": [3.0, 0.0, 0.0]}
    )
    assert r2["isError"] is False, r2
    uid2 = _payload(r2)["uid"]
    return ctx, session, uid1, uid2


def _doc(ctx: Any, session: agent_clay.Session) -> Any:
    return clay_mode.ensure(ctx).get(session.tab_uid).doc


def _stamp(ctx: Any, session: agent_clay.Session, uid: int) -> int:
    return _doc(ctx, session).mesh_stamp(uid)


def _b_scene(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, _uid1, _uid2 = _new_world()
    return ctx, session, {}


def _b_add_primitive(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx = _Ctx()
    session = agent_clay.Session()
    args = {
        "generator": "box",
        "params": {"size": [1.0, 1.0, 1.0]},
        "translation": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "name": "probe_prim",
        "material": 0,
    }
    return ctx, session, args


def _b_add_figure(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx = _Ctx()
    session = agent_clay.Session()
    key = sorted(presets.ASSEMBLIES)[0]
    args = {
        "key": key,
        "translation": [0.0, 0.0, 0.0],
        "yaw": 15.0,
        "scale": 1.0,
        "name_prefix": "figpfx_",
    }
    return ctx, session, args


def _b_transform(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, uid1, _uid2 = _new_world()
    args = {
        "uid": uid1,
        "translation": [1.0, 2.0, 3.0],
        "rotation": [10.0, 20.0, 30.0],
        "scale": [1.0, 1.0, 1.0],
    }
    return ctx, session, args


def _b_set_params(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, uid1, _uid2 = _new_world()
    args = {"uid": uid1, "params": {"size": [2.0, 1.0, 1.0]}}
    return ctx, session, args


def _b_material(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, uid1, _uid2 = _new_world()
    args = {
        "uids": [uid1],
        "name": "probe_material",
        "color": [0.5, 0.4, 0.3, 1.0],
        "metallic": 0.2,
        "roughness": 0.5,
    }
    return ctx, session, args


def _b_boolean(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, uid1, uid2 = _new_world()
    args = {"kind": "union", "uids": [uid1, uid2]}
    return ctx, session, args


def _b_select(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, uid1, _uid2 = _new_world()
    return ctx, session, {"uids": [uid1]}


def _b_element_mode(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, _uid1, _uid2 = _new_world()
    return ctx, session, {"mode": "vertex"}


def _b_select_elements(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, uid1, _uid2 = _new_world()
    stamp = _stamp(ctx, session, uid1)
    # verts/edges/faces are independently validated against this object's
    # own mesh regardless of 'mode' (which only decides what gets combined
    # into the selection at the end) -- all three given at once so each of
    # them, and each's own nested item shape, is reachable by this file's
    # per-property violation walk. (0, 1) is a real edge and 0 a real face
    # of a fresh box -- verified once, directly against the generator,
    # rather than assumed.
    args = {
        "uid": uid1,
        "mode": "vertex",
        "verts": [0],
        "edges": [[0, 1]],
        "faces": [0],
        "how": "replace",
        "expect_stamp": stamp,
    }
    return ctx, session, args


def _select_by_call(
    query: str, extra: Args, monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    from warlock.studio.clay import select as clay_select_mod

    ctx, session, uid1, _uid2 = _new_world()
    modes = clay_select_mod.QUERIES[query].modes
    mode_result = agent_clay.call(ctx, session, "clay_element_mode", {"mode": modes[0]})
    assert mode_result["isError"] is False, mode_result
    stamp = _stamp(ctx, session, uid1)
    args = {"uid": uid1, "query": query, "how": "replace", "expect_stamp": stamp, **extra}
    return ctx, session, args


def _b_select_by_bounds(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    extra = {"min": [-10.0, -10.0, -10.0], "max": [10.0, 10.0, 10.0], "space": "world"}
    return _select_by_call("bounds", extra, monkeypatch)


def _b_select_by_loop(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    return _select_by_call("loop", {"edge": [0, 1]}, monkeypatch)


def _b_select_by_face_loop(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    return _select_by_call("face_loop", {"face": 0}, monkeypatch)


def _b_select_by_material(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    return _select_by_call("material", {"slot": 0}, monkeypatch)


def _b_select_by_normal(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    extra = {"direction": [0.0, 1.0, 0.0], "max_angle": 45.0}
    return _select_by_call("normal", extra, monkeypatch)


# ``clay_select_by``'s seven query-argument schemas are declared as top-level
# properties of *one* schema, but a given call only ever consults the subset
# named by ``query`` (see the module docstring's own gap paragraph). The
# default baseline is the ``bounds`` one; a property named here gets a
# different, query-appropriate baseline instead, so ``slot``'s own
# ``minimum`` is actually reachable rather than silently ignored by a
# ``bounds`` call that never reads it.
_SELECT_BY_BASELINES: dict[str, BaselineFactory] = {
    "edge": _b_select_by_loop,
    "face": _b_select_by_face_loop,
    "slot": _b_select_by_material,
    "direction": _b_select_by_normal,
    "max_angle": _b_select_by_normal,
}


def _b_select_by(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    return _b_select_by_bounds(monkeypatch)


def _b_elements(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, uid1, _uid2 = _new_world()
    return ctx, session, {"uid": uid1, "kind": "vertex", "offset": 0, "limit": 10}


def _b_op(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, _uid1, _uid2 = _new_world()
    return ctx, session, {"name": "shade-auto", "params": {"angle": 30.0}}


def _b_render(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del svc
    ctx, session, uid1, _uid2 = _new_world()
    if monkeypatch is not None:
        _install_fake_view(monkeypatch)
    args = {
        "size": 256,
        "views": [{"yaw": 10.0, "pitch": 5.0}],
        "grid": True,
        "focus": [uid1],
    }
    return ctx, session, args


def _b_render_view(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    """``view`` and ``views`` are mutually exclusive -- the main
    :func:`_b_render` baseline carries ``views``, which would make adding a
    ``view`` key of its own a *second*, unintended violation (of the
    give-either-not-both rule) rather than an isolated one. This is
    ``view``'s own baseline instead, routed to by :data:`_RENDER_BASELINES`.
    """
    del svc
    ctx, session, _uid1, _uid2 = _new_world()
    if monkeypatch is not None:
        _install_fake_view(monkeypatch)
    return ctx, session, {"view": "front"}


def _b_render_compare(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    """``compare_mode`` and ``alpha`` are only ever read once ``compare``
    names a real reference (and ``alpha`` only in ``compare_mode="overlay"``)
    -- see ``_h_render``'s own ``if compare is not None:`` guard. Routed to
    by :data:`_RENDER_BASELINES` for ``compare``/``compare_mode``/``alpha``,
    none of which the plain :func:`_b_render` baseline (no ``compare`` at
    all) ever gets far enough to check.
    """
    ctx, session, _uid1, _uid2 = _new_world()
    if monkeypatch is not None:
        _install_fake_view(monkeypatch)
    added = agent_clay.call(
        ctx, session, "clay_reference_add", {"name": "cmp_ref", "png_base64": _tiny_ref_png_b64()}
    )
    assert added["isError"] is False, added
    args = {"compare": "cmp_ref", "compare_mode": "overlay", "alpha": 0.5}
    return ctx, session, args


# ``view``/``compare``/``compare_mode``/``alpha`` all need a baseline other
# than the plain one :func:`_b_render` builds -- see each override's own
# docstring for why. The rest of clay_render's properties (``size``,
# ``views``, ``grid``, ``focus``) are all reachable from :func:`_b_render`
# itself.
_RENDER_BASELINES: dict[str, BaselineFactory] = {
    "view": _b_render_view,
    "compare": _b_render_compare,
    "compare_mode": _b_render_compare,
    "alpha": _b_render_compare,
}


def _b_diagnose(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    """A document with one real ``"hole"`` finding, so ``select`` names a
    real ``(uid, kind)`` pair rather than one this handler would refuse for
    an unrelated reason before ever looking at ``select``'s own shape."""
    del monkeypatch, svc
    ctx, session, uid1, _uid2 = _new_world()
    mode = agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    assert mode["isError"] is False, mode
    sel = agent_clay.call(
        ctx, session, "clay_select_elements", {"uid": uid1, "faces": [0], "how": "replace"}
    )
    assert sel["isError"] is False, sel
    deleted = agent_clay.call(ctx, session, "clay_op", {"name": "delete"})
    assert deleted["isError"] is False, deleted
    back_to_object = agent_clay.call(ctx, session, "clay_element_mode", {"mode": "object"})
    assert back_to_object["isError"] is False, back_to_object
    args = {"uid": uid1, "select": {"uid": uid1, "kind": "hole"}}
    return ctx, session, args


def _b_export(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch
    ctx, session, _uid1, _uid2 = _new_world(svc=svc)
    return ctx, session, {}


def _b_undo(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, _uid1, _uid2 = _new_world()
    return ctx, session, {"steps": 1}


def _b_redo(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, _uid1, _uid2 = _new_world()
    undo = agent_clay.call(ctx, session, "clay_undo", {"steps": 1})
    assert undo["isError"] is False, undo
    return ctx, session, {"steps": 1}


def _b_delete(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, uid1, _uid2 = _new_world()
    return ctx, session, {"uids": [uid1]}


def _b_rename(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, uid1, _uid2 = _new_world()
    return ctx, session, {"uid": uid1, "name": "renamed_probe"}


def _b_batch(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx = _Ctx()
    session = agent_clay.Session()
    args = {"calls": [{"name": "clay_add_primitive", "arguments": {"generator": "box"}}]}
    return ctx, session, args


def _tiny_ref_png_b64() -> str:
    import base64
    import io

    from PIL import Image

    im = Image.new("RGB", (4, 4), "white")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b_reference_add(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, _uid1, _uid2 = _new_world()
    args = {"name": "probe_ref", "png_base64": _tiny_ref_png_b64(), "view": "front"}
    return ctx, session, args


def _b_reference_add_job(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    """``file`` is only ever read on the ``job_id`` branch of
    ``_h_reference_add`` -- given alongside ``png_base64`` instead (the
    plain :func:`_b_reference_add` baseline), it is never looked at, so a
    violation of its own shape would not be refused for what this file is
    nominally testing. Routed to by ``_REFERENCE_ADD_BASELINES``, mirroring
    ``tests/test_agent_clay.py``'s own
    ``test_reference_add_from_a_library_job_reads_its_input_png`` fixture."""
    del monkeypatch
    ctx, session, _uid1, _uid2 = _new_world(svc=svc)
    job_id = svc.store.create("image", None, {}, stage="reference", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(base64_decode_tiny_png())
    args = {"name": "probe_job_ref", "job_id": job_id, "file": "input.png"}
    return ctx, session, args


def base64_decode_tiny_png() -> bytes:
    import base64

    return base64.b64decode(_tiny_ref_png_b64())


_REFERENCE_ADD_BASELINES: dict[str, BaselineFactory] = {
    "job_id": _b_reference_add_job,
    "file": _b_reference_add_job,
}


def _b_reference_list(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, _uid1, _uid2 = _new_world()
    return ctx, session, {}


def _b_reference_get(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    del monkeypatch, svc
    ctx, session, _uid1, _uid2 = _new_world()
    added = agent_clay.call(
        ctx, session, "clay_reference_add", {"name": "probe_ref", "png_base64": _tiny_ref_png_b64()}
    )
    assert added["isError"] is False, added
    return ctx, session, {"name": "probe_ref"}


def _b_reference_remove(
    monkeypatch: Any = None, svc: Any = None
) -> tuple[Any, agent_clay.Session, Args]:
    return _b_reference_get(monkeypatch)


_BASELINES: dict[str, BaselineFactory] = {
    "clay_scene": _b_scene,
    "clay_add_primitive": _b_add_primitive,
    "clay_add_figure": _b_add_figure,
    "clay_transform": _b_transform,
    "clay_set_params": _b_set_params,
    "clay_material": _b_material,
    "clay_boolean": _b_boolean,
    "clay_select": _b_select,
    "clay_element_mode": _b_element_mode,
    "clay_select_elements": _b_select_elements,
    "clay_select_by": _b_select_by,
    "clay_elements": _b_elements,
    "clay_op": _b_op,
    "clay_render": _b_render,
    "clay_diagnose": _b_diagnose,
    "clay_export": _b_export,
    "clay_undo": _b_undo,
    "clay_redo": _b_redo,
    "clay_delete": _b_delete,
    "clay_rename": _b_rename,
    "clay_batch": _b_batch,
    "clay_reference_add": _b_reference_add,
    "clay_reference_list": _b_reference_list,
    "clay_reference_get": _b_reference_get,
    "clay_reference_remove": _b_reference_remove,
}

# Some tools' properties are only ever read by the handler under a
# condition the *plain* baseline in _BASELINES does not meet (clay_render's
# view/compare/compare_mode/alpha; clay_select_by's seven query-conditional
# arguments; clay_reference_add's job_id-only file) -- see each override
# factory's own docstring for why. Keyed by tool name, then by the specific
# top-level property that needs the different baseline.
_PROPERTY_OVERRIDES: dict[str, dict[str, BaselineFactory]] = {
    "clay_select_by": _SELECT_BY_BASELINES,
    "clay_render": _RENDER_BASELINES,
    "clay_reference_add": _REFERENCE_ADD_BASELINES,
}


def test_every_handler_has_a_baseline() -> None:
    """The gate the module docstring promises: a hand-kept table is only
    acceptable here if a new tool cannot be added without someone giving it
    one -- the same exhaustiveness ``test_agent_clay.py`` already holds
    ``_NEEDS_A_TAB``/``_MINTS_A_TAB``/``_SESSION_ONLY`` to."""
    assert set(_BASELINES) == set(agent_clay._HANDLERS)


@pytest.mark.parametrize("tool_name", sorted(_BASELINES))
def test_every_baseline_is_itself_accepted(
    tool_name: str, monkeypatch: pytest.MonkeyPatch, svc
) -> None:
    """Sanity: before perturbing anything, prove the baseline this file is
    about to violate one field of is actually valid on its own. ``svc`` (the
    shared fixture ``clay_export`` needs for ``clay_mode.build_asset``) is
    passed to every factory; every one but ``_b_export`` ignores it."""
    ctx, session, args = _BASELINES[tool_name](monkeypatch, svc)
    result = agent_clay.call(ctx, session, tool_name, args)
    assert result["isError"] is False, result


def _all_override_baselines() -> dict[str, BaselineFactory]:
    """Every override factory named in :data:`_PROPERTY_OVERRIDES`, by a
    unique label -- the same sanity :func:`test_every_baseline_is_itself_
    accepted` gives the primary baselines, extended to the per-property ones
    a handful of tools need instead (see each override's own docstring)."""
    named: dict[str, BaselineFactory] = {}
    for tool_name, overrides in _PROPERTY_OVERRIDES.items():
        for prop, factory in overrides.items():
            named[f"{tool_name}:{prop}"] = factory
    return named


_OVERRIDE_BASELINES = _all_override_baselines()


@pytest.mark.parametrize("label", sorted(_OVERRIDE_BASELINES))
def test_every_override_baseline_is_itself_accepted(
    label: str, monkeypatch: pytest.MonkeyPatch, svc
) -> None:
    tool_name = label.split(":", 1)[0]
    ctx, session, args = _OVERRIDE_BASELINES[label](monkeypatch, svc)
    result = agent_clay.call(ctx, session, tool_name, args)
    assert result["isError"] is False, result


# --- the discovery walk itself, pinned against the measured ground truth ------


def test_the_discovery_walk_finds_every_measured_constraint_marker() -> None:
    """The ground-truth counts this file's own docstring cites, reproduced
    by an independent recursive walk (not :func:`_walk_tool_schema`, which
    only emits *violable* cases) -- so the coverage claim in this file's
    docstring is checked against the schemas themselves rather than trusted.
    """
    from collections import Counter

    tracked = (
        "type",
        "additionalProperties",
        "properties",
        "items",
        "required",
        "minItems",
        "enum",
        "maxItems",
        "minimum",
        "maximum",
        "anyOf",
        "exclusiveMinimum",
    )

    def walk(node: Any, counts: Counter) -> None:
        if isinstance(node, dict):
            for key in node:
                if key in tracked:
                    counts[key] += 1
            for value in node.values():
                walk(value, counts)
        elif isinstance(node, list):
            for value in node:
                walk(value, counts)

    counts: Counter = Counter()
    for tool in agent_clay.tools():
        walk(tool.schema, counts)

    assert dict(counts) == {
        "type": 143,
        "additionalProperties": 31,
        "properties": 28,
        "items": 29,
        "required": 20,
        "minItems": 18,
        "enum": 16,
        "maxItems": 14,
        "minimum": 13,
        "maximum": 10,
        "anyOf": 3,
        "exclusiveMinimum": 1,
    }


def _all_cases() -> list[_Case]:
    tools = {t.name: t for t in agent_clay.tools()}
    cases: list[_Case] = []
    for name, tool in tools.items():
        found = _walk_tool_schema(name, tool.schema)
        cases.extend(found)
        if name in _OPEN_ENDED_PARAMS_TOOLS:
            # The baseline's own args are needed to know which real key to
            # recurse into -- see _open_ended_params_cases's own docstring.
            _ctx, _session, baseline_args = _BASELINES[name]()
            cases.extend(_open_ended_params_cases(name, tool.schema, baseline_args))
    return cases


_ALL_CASES = _all_cases()


def test_the_exercise_walk_attempts_exactly_the_documented_number_of_cases() -> None:
    """248 -- see the module docstring's own derivation: 326 measured markers,
    minus 60 structural ones that only route recursion (28 ``properties`` +
    29 ``items`` + 3 schema-valued ``additionalProperties``), minus 25 root
    ``type: object`` markers that are true by construction, minus 20
    ``required`` *lists* replaced by the 27 individual keys they actually
    name. Pinned so a schema edit that silently drops a case from the walk
    is caught here rather than only by a shrinking "exercised" count nobody
    happens to notice.
    """
    assert len(_ALL_CASES) == 248


# --- the exercise itself: for each declared constraint, prove a refusal -------

# A handful of cases are refused for a real, correct, but *different* stated
# reason -- an out-of-range uid refuses at ``_resolve_uid`` before a sibling
# argument's own shape is checked, say -- so this file does not assert which
# ``field`` the refusal names for them. Every entry names the reason; nothing
# is silently downgraded without one. Keyed by ``_Case.id``.
_FIELD_NOT_ASSERTED: dict[str, str] = {
    "clay_diagnose:select/uid:required": (
        "_resolve_uid names the field it was actually given ('uid'), not "
        "the outer 'select' object it lives in -- more specific than "
        "this file's own top-level-property default, not less correct."
    ),
    "clay_diagnose:select/uid:type": (
        "same as clay_diagnose:select/uid:required above: _resolve_uid "
        "names 'uid', the real field, not 'select'."
    ),
    "clay_batch:calls/0/name:required": (
        "a call entry missing 'name' is refused as 'None is not a "
        "batchable tool' (field='calls'), not a required-key message."
    ),
    "clay_op:params/angle:type": (
        "a non-numeric op param value is refused by call()'s own generic "
        "ValueError backstop (float() raising inside clay_ops.run), with "
        "no field named at all."
    ),
}


def _run_case(case: _Case, monkeypatch: pytest.MonkeyPatch, svc: Any) -> None:
    # clay_export's baseline needs no real svc: its one case (the top-level
    # additionalProperties every tool carries) is refused in agent_clay.call
    # before _h_export -- the only handler that reaches for ctx.svc -- ever
    # runs, so no service fixture is threaded through this parametrized walk.
    overrides = _PROPERTY_OVERRIDES.get(case.tool, {})
    override = overrides.get(case.path[0]) if case.path else None
    factory = override if override is not None else _BASELINES[case.tool]
    ctx, session, baseline_args = factory(monkeypatch, svc)
    mutated = case.mutate(baseline_args)
    result = agent_clay.call(ctx, session, case.tool, mutated)
    assert result["isError"] is True, (case.id, result)
    if case.id not in _FIELD_NOT_ASSERTED:
        structured = result.get("structuredContent") or {}
        # Every handler in this file validates a whole top-level argument's
        # shape at once and reports the field it was given -- never a
        # deeper per-index sub-field -- which is why a nested constraint
        # (pitch's own range inside one ``views[]`` entry, a vec3's own
        # item type inside ``min``) is still named by its *top-level*
        # property, not the path this case's own violation reaches into.
        # A root-level (path == ()) additionalProperties violation is the
        # one case with no top-level property to name at all -- it names
        # the unexpected key itself, via agent_clay._unknown_argument_refusal.
        expected_field = case.path[0] if case.path else _PROBE_KEY
        assert structured.get("field") == expected_field, (case.id, result)


@pytest.mark.parametrize("case", _ALL_CASES, ids=[c.id for c in _ALL_CASES])
def test_a_declared_constraint_is_enforced(
    case: _Case, monkeypatch: pytest.MonkeyPatch, svc
) -> None:
    _run_case(case, monkeypatch, svc)


def test_every_documented_field_exception_corresponds_to_a_real_case() -> None:
    """:data:`_FIELD_NOT_ASSERTED` names real cases, not stale ones -- a
    keyword that stops matching (the schema changed, or a handler got
    fixed to name the field after all) is caught here rather than left
    describing a case that no longer exists."""
    ids = {c.id for c in _ALL_CASES}
    for case_id in _FIELD_NOT_ASSERTED:
        assert case_id in ids, case_id
