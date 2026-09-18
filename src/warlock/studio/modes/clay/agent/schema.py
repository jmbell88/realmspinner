"""Clay's agent tool surface, the schema half: every JSON-schema fragment and
every prose catalogue :func:`agent_clay.tools`/:func:`agent_clay.instructions`
compose into the published tool list, plus the numeric ceilings that shape
both a tool's declared schema and the handler that enforces it.

Split out of ``studio/modes/clay/agent/dispatch.py`` in the P4 restructure (``dev/RESTRUCTURE.md``)
-- that module's own "# --- the tools" and "# --- output schemas" sections,
minus the pieces that turned out on a close read to belong elsewhere (see
below). **This module holds no dispatch state and calls no handler**: every
function here is a pure builder, taking at most a registry entry and
returning a ``dict``/``str``, which is what makes it safe to import from
both ``studio/modes/clay/agent/dispatch.py`` (to build the published tool list) and the handler
modules (``agent_clay_tools*.py``, to enforce the same ceiling a tool's own
schema advertises) with no import cycle between any of the three -- this
module reaches back into neither.

**The numeric constants below are schema vocabulary, not dispatch state,
which is why they live here and not in ``studio/modes/clay/agent/dispatch.py``.** Each one is a
number a tool's own published schema or description names (``BATCH_MAX`` in
``clay_batch``'s ``maxItems`` and its own sentence, ``MAX_MESH_VERTICES`` in
``clay_add_mesh``'s description, and so on) and that the matching handler
enforces against the identical value -- two readers of one number, never a
schema-side copy that could drift from the handler's own ceiling. Contrast
``agent_clay.PROGRAM_DEADLINE_S`` and ``agent_clay._view_for``, which stay in
``studio/modes/clay/agent/dispatch.py`` itself even though a handler here reads them too: both are
monkeypatched directly on ``agent_clay`` by name in ``tests/modes/clay/test_agent_clay.py``
and ``tests/mcp/test_rpc_studio.py`` (``PROGRAM_DEADLINE_S`` to shrink a
program's deadline to zero for a timeout test, ``_view_for`` to fake a
viewport with no GL context), which only works if the patched name is read
back through ``agent_clay`` itself at call time -- a handler that imported
either as a plain name from wherever it "really" lives would keep its own,
unpatched copy. Every constant below carries no such test, so a plain
top-of-file import is enough and the handler always sees the live value
because there is only ever one copy of it, here.

Every catalogue function (``_op_catalog``, ``_generator_catalog``,
``_figure_part_catalog``, ``_query_catalog``) and every one of the five
declared ``outputSchema`` builders reads a live registry
(``primitives.GENERATORS``, ``presets.ASSEMBLIES``, ``clay_ops.OPS``,
``select.QUERIES``) rather than a hand-kept copy of it -- the derivation
rule ``studio/modes/clay/agent/dispatch.py``'s own module docstring states for the tool list as a
whole applies just as much to the sentence describing one tool's argument as
to the enum naming it, and moving those functions to a second file must not
turn either into a written-out copy. Nothing here does: a thirteenth
generator or a thirteenth op still reaches an agent through the exact same
two functions it always did, now living one file over.
"""

from __future__ import annotations

import functools

from .....kernels.mesh import presets
from .....kernels.mesh import primitives as bp
from .....kernels.mesh import select as bsel
from .. import ops as clay_ops

# --- module constants, the schema-vocabulary half ----------------------------
#
# See the module docstring for why these live here rather than in
# ``studio/modes/clay/agent/dispatch.py``: each is a ceiling or a count a published tool schema or
# description names, read back by the matching handler in ``agent_clay_tools*.py``
# to enforce the identical number -- never a second, hand-copied ceiling.

BATCH_MAX = 32
"""The most tool calls one ``clay_batch`` request may fold into a single undo
step. The byte budget (``UndoStack``) already bounds what a single step can
*cost*, but nothing bounded how many calls a batch could ask for before this
-- an unbounded batch would let one MCP round trip queue an arbitrarily long
run with no natural place to hit a ceiling first."""

MINTS_A_DOCUMENT: tuple[str, ...] = (
    "clay_add_primitive",
    "clay_add_figure",
    "clay_add_mesh",
)
"""The three tools that may start a document from nothing -- and so the three
a ``clay_batch`` may open with when the session owns no tab yet.

A constant rather than a tuple literal inside ``_h_batch`` because the list
was already written out in two places that then disagreed: the handler's own
membership check grew ``clay_add_mesh`` when that tool landed, and
``clay_batch``'s published description -- the sentence an agent reads before
it ever calls anything -- did not, so the catalogue told a model that a batch
starting with ``clay_add_mesh`` would be refused while the code was happily
running it. That is the hand-kept-copy-of-another-table drift this file
already refuses to write for its generator, op and query enums; the
description now interpolates this tuple instead of restating it, and
``tests/modes/clay/test_agent_clay.py`` pins it against the same ``_MINTS_A_TAB`` list
that keeps every tool classified.

The three prose refusals that also name these three (``agent_clay._tab``'s two
and ``agent_clay_tools_batch._h_batch``'s own) are left as English rather than
interpolated: "call clay_add_primitive, clay_add_figure or clay_add_mesh" is a
sentence, not a list, and the test below is what catches one of them going
stale.
"""

REFERENCE_TOOLS = frozenset(
    {
        "clay_reference_add",
        "clay_reference_get",
        "clay_reference_list",
        "clay_reference_remove",
    }
)
"""The four reference tools, named once, so nothing else in this fold has to
relist them by hand -- ``studio/assistant/preview.py``'s own
``PREVIEW_EXCLUDED`` reads this set (through ``dispatch.py``'s re-export)
rather than writing the four names out a second time.

**Deliberately not folded into ``BATCH_EXCLUDED`` below.** The 2026-09-18
audit's familiar-04 first tried exactly that -- ``BATCH_EXCLUDED`` had
always named only ``clay_reference_get``, so ``clay_reference_add``/
``_list``/``_remove`` were batchable, and folding this whole set in closed
that hole -- but ``BATCH_EXCLUDED`` is *published*: ``clay_batch``'s own
description sentence interpolates ``set(_HANDLERS) - BATCH_EXCLUDED``
(``studio/modes/clay/agent/dispatch.py``'s ``batch_names``), so widening it
changed the live tool catalogue's text and, with it, the catalogue hash a
training dataset had pinned (``dev/tests/familiar/test_contract.py``'s
``test_derive_clay_card_reproduces_the_dataset_manifest_tools_sha``) --
and silently made three tools unbatchable for every external MCP agent, a
public-surface change familiar-04 never asked for. The fix moved to
``studio/assistant/preview.py``'s ``run_scratch`` instead: it now walks a
``clay_batch`` call's own ``calls`` before running it and refuses any nested
name in ``PREVIEW_EXCLUDED``, the same set a *direct* call already checks --
closing the preview-only hole without touching what an ordinary MCP client
may batch."""

BATCH_EXCLUDED = frozenset(
    {
        "clay_batch",  # nesting buys nothing and bounds nothing
        # A program already folds its own run into one step the identical
        # way; nesting one inside a batch entry buys nothing either, and
        # agent_program.compile_program never emits a name in this set (see
        # test_the_compiler_never_emits_a_batch_excluded_tool), so the two
        # surfaces cannot disagree about what is nestable.
        "clay_program",
        # An image block is a result a client must see as one; a batch can
        # only hand back JSON text, so neither of these has a shape a batch
        # result could carry.
        "clay_render",
        "clay_reference_get",
        "clay_export",  # a deliberate one-shot that mints a Library row
        # A batch is itself one undo step; moving the history head from
        # inside the very run that is about to fold into one is incoherent.
        "clay_undo",
        "clay_redo",
    }
)
"""Tools ``clay_batch`` refuses to run -- see ``agent_clay_tools_batch._h_batch``'s
docstring for the derivation, and the comments above for why each one is
excluded. See ``REFERENCE_TOOLS``'s own docstring for why the other three
reference tools stay off this particular list."""

PROGRAM_DEADLINE_S = 4.0
"""The wall-clock budget one ``clay_program`` call gets, measured from the
moment its compiled calls start running and checked between them (never
mid-call, so one already-running call is never cut off) -- past it, the run
rolls back and refuses rather than keep going into a second, third frame.
``clay_program`` is deliberately not chunked across frames the way ``pump``'s
own queued-job budget chunks ordinary calls (see ``dev/INVARIANTS.md``'s
agent paragraph): a program's whole point is that it is one MCP round trip,
and a caller that needs more than this buys should split the program into
several smaller ``clay_program`` calls rather than have this tool silently
spread one across an unbounded number of frames.

Defined here, in the schema module, but read by
``agent_clay_tools_batch._h_program`` through ``agent_clay.PROGRAM_DEADLINE_S``
-- qualified, at call time -- rather than imported by name: two of this
file's own tests (``tests/modes/clay/test_agent_clay.py``) monkeypatch it on
``agent_clay`` itself to shrink a program's deadline for a timeout test, and
that only works if every reader looks it up through the same name at call
time. See ``studio/modes/clay/agent/dispatch.py``'s own module docstring for the constant's actual
value and that reasoning in full; it is a schema-module constant re-exported
there, not the other way around, only because ``studio/modes/clay/agent/dispatch.py`` is what the
test patches by name."""

MAX_REFERENCES = 8
"""How many pictures one session may hold at once. A session's references
live in memory for the session's whole life (``agent_clay_validate.Session.
references``) and nothing ever evicts one on its own, so a ceiling is what
keeps a client that forgets ``clay_reference_remove`` from growing an
unbounded set of decoded PNGs behind a session nobody is watching."""

RENDER_PIXEL_BUDGET = 6 * 1024 * 1024
"""The most total pixels one ``clay_render`` call may ask for, summed across
every view it requests. A caller asking for the maximum -- a dozen views at
2048x2048 -- would otherwise spend real GPU minutes producing a reply that
was refused for size the moment it tried to leave; this stops that before
the first frame is drawn rather than after."""

RENDER_FRAME_RESERVE = 64 * 1024
"""Headroom subtracted from ``protocol.MAX_FRAME`` when checking whether a
render's base64-encoded payload will fit in one reply frame. The JSON
envelope around the image blocks costs bytes of its own; without this an
encode that is over by a few hundred bytes would reach ``send_bytes`` and
fail there, past the point a refusal could explain itself."""

RENDER_SHADINGS = (
    "unlit", "lit", "wireframe", "wire_overlay", "xray", "object_id",
)
"""``clay_render``'s ``shading`` enum, one place rather than two: this tuple
is what the schema's own ``enum`` is built from below, so a seventh value
added here reaches the wire with no second edit. ``unlit`` is first because
it is the default -- ``args.get("shading", RENDER_SHADINGS[0])`` in
``agent_clay_tools_ops._h_render`` reads that position rather than a
duplicated literal, so the two cannot name a different default by accident.
The first five map onto ``ClayView.render_png``'s own ``shading`` (see
``clay_view._SHADING_DRAW_KWARGS`` for that table); ``object_id`` is answered
by ``ClayView.render_ids`` instead, a different draw with a different return
shape."""

ELEMENT_PAGE_MAX = 4096
"""The most element indices one ``clay_elements`` call may hand back for one
kind, per object. This is **not** a wire limit the way ``RENDER_PIXEL_BUDGET``
and ``protocol.MAX_FRAME`` are -- a mesh's entire face list is real image-sized
data with nowhere smaller to fit, and a whole one would clear the frame budget
with room to spare. The bound here is the *reader*: four thousand bare
integers is on the order of twenty thousand tokens dropped into the context of
whatever is about to act on them, and an agent that genuinely needs that many
at once is reasoning about the document a different way than one call's reply
can serve -- ``offset``/``limit`` paging is the answer, not a bigger ceiling."""

ELEMENT_PAGE_DEFAULT = 256
"""``clay_elements``'s own default ``limit`` when none is given -- small
enough that the common case ("what did that extrude just make") comes back as
a paragraph rather than a printout, and still a small fraction of
``ELEMENT_PAGE_MAX`` for the rarer caller that has to page through more."""

MAX_MESH_VERTICES = 50_000
MAX_MESH_FACES = 50_000
"""The most vertices and faces one ``clay_add_mesh`` call may hand over,
checked -- and refused, naming the field -- before either array in ``mesh.
from_faces`` is built. Both stay well under ``glbimport.MAX_TRIANGLES``
(2,000,000 triangles), the document-wide ceiling ``serialize.py`` enforces on
import, because one call is one object among however many a document already
holds and this tool has no business spending most of that budget in one
shot. In practice this is the ceiling that actually fires: even a maximally
compact encoding of 50,000 vertices or faces (bare digits, no whitespace)
fits inside ``protocol.MAX_FRAME``'s 8 MiB request budget with room to
spare, so the wire frame limit is a backstop for a verbose encoding, not the
check doing the real work here."""


# --- shape helpers, shared by several tool schemas ---------------------------


def _vec3_schema(unit: str) -> dict:
    """The one shape every optional TRS vector in this module shares."""
    return {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
        "description": unit,
    }


def _params_value_schema() -> dict:
    """The one shape every ``params`` value in this module shares: a number, a
    flat array of numbers, or an array of arrays of numbers.

    Written out twice before this -- at ``clay_add_primitive`` and
    ``clay_set_params`` -- as an ``anyOf`` of only the first two branches,
    which is exactly the drift the module docstring's derivation paragraphs
    exist to rule out for everything else in this file. The third branch is
    for ``lathe``'s ``profile``: a cylinder's ``radius`` is one number, a
    box's ``size`` is three, and a lathe's ``profile`` is an array of
    ``[radius, y]`` pairs -- the first generator parameter this registry has
    whose own elements are arrays rather than numbers. A schema declaring
    this shape is not enforcement of it; ``agent_clay_validate._validate_number_or_vec``
    is the handler-side half both tools already call.
    """
    return {
        "anyOf": [
            {"type": "number"},
            {"type": "array", "items": {"type": "number"}},
            {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        ]
    }


# The one mapping from a query argument's *name* to the JSON-schema fragment
# that describes it -- ``select.py`` holds no JSON-schema knowledge at all
# (its own module docstring's rule), so this is where that vocabulary is
# spelled out, once, for every query that shares an argument name rather than
# once per query. Keys are the union of every ``Query.args`` tuple in
# :data:`select.QUERIES`; ``test_every_query_argument_name_has_a_schema_
# fragment_and_vice_versa`` gates both directions, so a query that grows an
# argument nobody here can express -- or an entry here nothing asks for any
# more -- fails the suite instead of drifting quietly.
_QUERY_ARG_SCHEMAS: dict[str, dict] = {
    "edge": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 2,
        "description": "A [vertex, vertex] pair naming one edge -- the seed "
        "to walk the loop or ring from.",
    },
    "face": {
        "type": "integer",
        "description": "A face index -- the seed to walk the face loop from.",
    },
    "slot": {
        "type": "integer",
        "minimum": 0,
        "description": "A palette material index -- see clay_scene's 'materials'.",
    },
    "direction": _vec3_schema(
        "world-space; need not be unit length. Converted into the object's "
        "own local frame before the query runs -- see clay_scene's "
        "'rotation'/'scale' for the transform an agent read this out of."
    ),
    "max_angle": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 180.0,
        "description": "Degrees off 'direction' still counted as facing it. Default 45.",
    },
    "min": _vec3_schema("metres, the box's lower corner"),
    "max": _vec3_schema("metres, the box's upper corner"),
    "space": {
        "type": "string",
        "enum": ["world", "local"],
        "description": "Which frame 'min'/'max' are given in. Default 'world'.",
    },
}

# The two query arguments that are optional -- ``max_angle`` (``_q_normal``'s
# own default) and ``space`` (``agent_clay_tools_ops``'s own default of
# "world", see ``_h_select_by``). Every other name in ``_QUERY_ARG_SCHEMAS`` is
# required whenever a query declares it.
_QUERY_OPTIONAL_ARGS = frozenset({"max_angle", "space"})


# --- output schemas -----------------------------------------------------------
#
# Five tools below declare an ``outputSchema`` at all -- ``clay_scene``,
# ``clay_add_primitive``, ``clay_add_mesh``, ``clay_diagnose`` and
# ``clay_analyze``. Every other tool's result is small and self-explanatory
# (a uid, a count, a list of names); writing a schema for each would be
# schema authoring with no reader, so this file deliberately does not.
# These five are the ones whose shape is worth writing down once rather
# than making a client work it back out of a sample reply.
#
# None of the five declares ``required``, and none sets
# ``additionalProperties: false``. That is not an oversight -- a refusal
# from any of these tools answers through the *same* result envelope
# (``protocol.fail``), and a refusal's own ``structuredContent`` is whatever
# ``fail``'s ``**extra`` was given -- ``{"field": "uids"}`` and nothing else.
# A ``required`` list on the success shape would make every refusal of
# these tools non-conforming for a client that validates strictly against
# ``outputSchema``, and MCP's own wording on whether an ``isError`` result
# must still conform to it is not explicit enough to bet a client's error
# handling on that reading. Declaring ``properties`` still documents the
# shape for a reader -- it just never claims a key the envelope cannot
# promise to keep filled.


def _sel_counts_schema() -> dict:
    """The shape ``agent_clay_validate._sel_counts`` returns -- three counts,
    shared by the object row schema below and ``clay_diagnose``'s own
    ``selected`` field, exactly as the one ``_sel_counts`` function is shared
    by both callers."""
    return {
        "type": "object",
        "properties": {
            "verts": {"type": "integer"},
            "edges": {"type": "integer"},
            "faces": {"type": "integer"},
        },
    }


def _object_row_output_schema() -> dict:
    """The JSON Schema for one row of ``agent_clay_validate._scene_row`` --
    read that function, not this one, when deciding what belongs here: every
    key it returns must appear below with the right type, or this schema has
    drifted from the function that actually builds the row. That is the
    drift class this module's own docstring warns about for a hand-written
    second copy of a shape a real function already owns, so this helper is
    the one place it is written, used by both :func:`_clay_scene_output_schema`
    (inside ``objects``) and ``clay_add_primitive``'s own declared schema,
    which *is* this schema -- its result is one row, unwrapped.

    ``bbox``, ``size`` and ``center`` admit ``null``: ``_scene_row`` reports
    all three as ``None`` for an object with no box. ``params`` is an open
    object -- what is in it depends on ``generator``, which this schema has
    no way to branch on. See the module comment above for why nothing here
    is ``required``.
    """
    vec3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    nullable_vec3 = {"anyOf": [{"type": "null"}, vec3]}
    return {
        "type": "object",
        "properties": {
            "uid": {"type": "integer"},
            "name": {"type": "string"},
            "visible": {"type": "boolean"},
            "generator": {"type": "string"},
            "params": {"type": "object"},
            "faces": {"type": "integer"},
            "material": {"type": "integer"},
            "bbox": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {
                            "min": {"type": "array", "items": {"type": "number"}},
                            "max": {"type": "array", "items": {"type": "number"}},
                        },
                    },
                ]
            },
            "translation": vec3,
            "rotation": vec3,
            "scale": vec3,
            "size": nullable_vec3,
            "center": nullable_vec3,
            "verts": {"type": "integer"},
            "stamp": {"type": "integer"},
            "selected": _sel_counts_schema(),
        },
    }


def _clay_scene_output_schema() -> dict:
    """``clay_scene``'s declared ``outputSchema`` -- built from what
    ``agent_clay_tools._h_scene`` actually returns (``objects``, ``selection``,
    ``element_mode``, ``dirty``, ``object_count``, ``bounds``, ``materials``),
    with ``objects`` built from the one shared :func:`_object_row_output_schema`
    rather than a second, hand-written copy of the row shape."""
    vec3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    return {
        "type": "object",
        "properties": {
            "objects": {"type": "array", "items": _object_row_output_schema()},
            "selection": {"type": "array", "items": {"type": "integer"}},
            "element_mode": {"type": "string"},
            "dirty": {"type": "boolean"},
            "object_count": {"type": "integer"},
            "bounds": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {
                            "min": vec3,
                            "max": vec3,
                            "size": vec3,
                            "center": vec3,
                        },
                    },
                ]
            },
            "materials": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "name": {"type": "string"},
                        "color": {"type": "array", "items": {"type": "number"}},
                        "metallic": {"type": "number"},
                        "roughness": {"type": "number"},
                    },
                },
            },
        },
    }


def _finding_row_output_schema() -> dict:
    """The shape of one ``clay_diagnose.Finding`` row, as JSON -- shared by
    ``clay_diagnose``'s own output schema and ``clay_add_mesh``'s (see
    :func:`_mesh_row_output_schema`), so a finding's shape is written once
    rather than copied the second time a tool needed to describe it."""
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "label": {"type": "string"},
            "count": {"type": "integer"},
            "mode": {"type": "string"},
        },
    }


def _mesh_row_output_schema() -> dict:
    """``clay_add_mesh``'s declared ``outputSchema`` -- the same object row
    :func:`_object_row_output_schema` already describes, composed rather than
    copied, plus the two keys only this tool answers with: ``closed`` (no
    hole, no non-manifold edge -- see ``agent_clay_tools._h_add_mesh``'s own
    docstring for why those two findings are what "closed" means here) and
    ``findings`` (the same rows ``clay_diagnose`` reports, via the shared
    :func:`_finding_row_output_schema`), so an agent that just handed over
    geometry learns what -- if anything -- is wrong with it in the same call
    that placed it."""
    row = _object_row_output_schema()
    return {
        **row,
        "properties": {
            **row["properties"],
            "closed": {"type": "boolean"},
            "findings": {"type": "array", "items": _finding_row_output_schema()},
        },
    }


def _clay_diagnose_output_schema() -> dict:
    """``clay_diagnose``'s declared ``outputSchema`` -- built from what
    ``agent_clay_tools_ops._h_diagnose`` actually returns: ``objects`` always,
    ``selected`` only when ``select`` was given and matched a finding."""
    return {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "uid": {"type": "integer"},
                        "name": {"type": "string"},
                        "clean": {"type": "boolean"},
                        "findings": {
                            "type": "array",
                            "items": _finding_row_output_schema(),
                        },
                    },
                },
            },
            # Document-level rows, present only on a whole-document call and
            # only when there is something to say. Not ``findings``' shape:
            # they point at objects (``uids``) rather than at elements, so
            # they carry no ``mode`` and are not reachable through ``select``.
            "scene": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string"},
                        "label": {"type": "string"},
                        "uids": {"type": "array", "items": {"type": "integer"}},
                    },
                },
            },
            "selected": {
                "type": "object",
                "properties": {
                    "uid": {"type": "integer"},
                    "kind": {"type": "string"},
                    "mode": {"type": "string"},
                    "stamp": {"type": "integer"},
                    "selected": _sel_counts_schema(),
                },
            },
        },
    }


def _clay_analyze_output_schema() -> dict:
    """``clay_analyze``'s declared ``outputSchema`` -- built from what
    ``agent_clay_tools_ops._h_analyze`` actually returns. ``bounds`` admits
    ``null`` for an object with no vertices, exactly as ``clay_scene``'s own
    ``bbox`` does, and for the same reason: ``analyze.analyze`` cannot
    measure a box around nothing."""
    vec3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    ground_schema = {
        "type": "object",
        "properties": {
            "min_y": {"type": "number"},
            "contact": {"type": "boolean"},
            "penetration": {"type": "number"},
        },
    }
    overlap_schema = {
        "type": "object",
        "properties": {
            "volume": {"type": "number"},
            "depth": {"type": "number"},
        },
    }
    return {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "uid": {"type": "integer"},
                        "name": {"type": "string"},
                        "bounds": {
                            "anyOf": [
                                {"type": "null"},
                                {"type": "object", "properties": {"min": vec3, "max": vec3}},
                            ]
                        },
                        "area": {"type": "number"},
                        "volume": {"anyOf": [{"type": "null"}, {"type": "number"}]},
                        "closed": {"type": "boolean"},
                        "components": {"type": "integer"},
                        "ground": {"anyOf": [{"type": "null"}, ground_schema]},
                        "symmetry": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                    },
                },
            },
            "pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "uids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "distance": {"anyOf": [{"type": "null"}, {"type": "number"}]},
                        # null -- unknown, not "no" -- once a pair clears
                        # MAX_TRIANGLE_PAIRS and the full narrow phase never
                        # runs (analyze.py's own PairAnalysis.intersects
                        # docstring); widened alongside distance's own
                        # anyOf when _h_analyze started emitting it
                        # (2026-09-14 audit, clay-01 follow-up: this was
                        # still declared a bare boolean after analyze.py's
                        # clay-01 fix started returning None here).
                        "intersects": {"anyOf": [{"type": "null"}, {"type": "boolean"}]},
                        "contact": {"type": "boolean"},
                        "overlap": {"anyOf": [{"type": "null"}, overlap_schema]},
                        "exact": {"type": "boolean"},
                    },
                },
            },
            "floating": {"type": "array", "items": {"type": "integer"}},
            "truncated": {"type": "boolean"},
            "tolerances": {
                "type": "object",
                "properties": {
                    "contact_tol": {"type": "number"},
                    "near": {"type": "number"},
                    "symmetry_tol": {"type": "number"},
                },
            },
        },
    }


# --- prose catalogues, built from the live registries ------------------------


def _param_prose(p: clay_ops.Param) -> str:
    """One param, worded for what it actually *is*.

    A bare "name (low-high, default x)" describes a number field, and that is
    all a model is ever told about an op's argument -- so it was a poor
    description of the boolean and three-way choice ``clay_ops.Param`` grew
    on 2026-09-10 (``place-between``'s ``fit``, ``array-radial`` and
    ``mirror-copy``'s ``axis``): "fit (0.0-1.0, default 1.0)" does not say
    "this is on unless you turn it off", and "axis (0.0-2.0, default 1.0)"
    does not say which number is which axis. The call itself is unchanged --
    ``params`` in the schema below is still ``{"type": "number"}`` for every
    field, a checkbox is still 0/1 and a choice is still its index -- only
    the sentence describing it gets to say what kind it is.
    """
    if p.boolean:
        return f"{p.name} (boolean 0/1, default {int(p.default)})"
    if p.choices:
        named = ", ".join(f"{i}={choice}" for i, choice in enumerate(p.choices))
        return f"{p.name} (choice: {named}, default {int(p.default)})"
    return f"{p.name} ({p.low}-{p.high}, default {p.default})"


def _op_catalog(names: list[str]) -> str:
    """Every op's params and bounds, folded into one sentence.

    Built from :data:`clay_ops.OPS` rather than written out, so a thirteenth
    op -- with or without parameters -- appears here the next time ``tools()``
    is called and nowhere needs editing for it to.
    """
    parts = []
    for op in clay_ops.OPS:
        modes = "/".join(op.modes)
        if op.params:
            fields = ", ".join(_param_prose(p) for p in op.params)
            parts.append(f"{op.name} [{modes}: {fields}]")
        else:
            parts.append(f"{op.name} [{modes}]")
    del names
    return "; ".join(parts)


def _generator_catalog() -> str:
    """Every generator's own keys and their defaults, folded into one sentence.

    Built from :data:`primitives.GENERATORS` rather than written out, so a
    thirteenth primitive's parameters appear here the next time ``tools()`` is
    called and nowhere needs editing for it to -- the same rule
    :func:`_op_catalog` already follows for ``clay_op``.
    """
    parts = []
    for name in sorted(bp.GENERATORS):
        defaults = bp.GENERATORS[name][0]
        fields = ", ".join(f"{key}={value!r}" for key, value in defaults.items())
        parts.append(f"{name} [{fields}]")
    return "; ".join(parts)


@functools.lru_cache(maxsize=4)
def _figure_part_catalog(keys: tuple[str, ...]) -> str:
    """Every figure key's own part names, folded into one sentence.

    Built from :data:`presets.ASSEMBLIES` rather than written out, so a
    ninth figure's part names appear here the next time ``tools()`` is
    called and nowhere needs editing for it to -- the same rule
    :func:`_generator_catalog` and :func:`_op_catalog` already follow for
    their own registries. Run A of the Clay-assistant fine-tune
    (2026-09-12, ``dev/measurements/data/clay-assistant/run-A/``) showed 15
    refusals of the shape ``no object named '...'`` because nothing told the
    model a figure preset's part names -- nine of those, in the
    ``creatures`` family, were guesses at a generated figure's own parts
    (``hound_Beak``, ``t_Shank.R``, ``s_Tail 01``) that would have been
    right had the model been told what a figure preset actually names them.

    Reads each key's raw builder (``ASSEMBLIES[key][1]()``) rather than
    calling :func:`presets.build`, which also grounds the assembly (shifts
    it so its lowest vertex sits on Y=0, ``presets._grounded``) -- work this
    catalogue has no use for, and which raises ``ValueError: min() iterable
    argument is empty`` on ``test_a_ninth_figure_reaches_the_agent_surface_
    with_no_edit_here``'s ``monkeypatch.setitem(presets.ASSEMBLIES,
    "ninth_figure", ("Ninth", lambda: ()))`` -- an assembly with no parts at
    all has no lowest vertex to measure. Grounding only ever changes a
    part's translation, never its name, so the raw builder answers this
    catalogue's own question just as well and without that crash.

    Memoised on *keys* (``tuple(sorted(presets.ASSEMBLIES))``), not bare --
    building every part of every figure is real work worth skipping across
    the many ``tools/list`` requests one session makes, but a bare
    no-argument cache would survive that same test's
    ``monkeypatch.setitem`` and keep answering with the pre-patch table for
    the rest of the test session once ``monkeypatch`` undoes the patch.
    Keying on the sorted key tuple makes that edit a cache miss instead of a
    stale hit, while the normal case -- the same eight keys, every call, for
    a whole process's life -- still hits every time after the first.
    """
    parts = []
    for key in keys:
        _label, builder = presets.ASSEMBLIES[key]
        names = ", ".join(part.name for part in builder())
        parts.append(f"{key}: {names}")
    return "; ".join(parts)


def _query_catalog() -> str:
    """Every ``clay_select_by`` query's own argument names and its hint,
    folded into one sentence.

    Built from :data:`select.QUERIES` rather than written out, so a seventh
    query appears here the next time ``tools()`` is called and nowhere needs
    editing for it to -- the same rule :func:`_op_catalog` and
    :func:`_generator_catalog` already follow for their own registries.
    """
    parts = []
    for name in sorted(bsel.QUERIES):
        query = bsel.QUERIES[name]
        args = ", ".join(query.args)
        parts.append(f"{name} [{args}]: {query.hint}")
    return " ".join(parts)
