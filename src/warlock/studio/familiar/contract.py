"""The Clay-assistant contract: frozen prompt cards, the sampler settings run
A's own measurement pinned, and the wire-format helpers ``training/clay-
assistant`` and Familiar's own spawn path both need to agree on byte for
byte.

This is a T3 extraction, not a rewrite. Every function here used to live in
``training/clay-assistant/gen/convert.py`` (built against the *live*
``agent_clay`` registry, so a new generator or op changed the dataset on the
next ``build.py`` run) and ``training/clay-assistant/eval/run_val.py`` (the
reply grammar a trained model is scored against). The user chose to freeze
run A's own card (``cards/clay-1.txt``, sha256 ``70697ece…8aab``, recorded as
``settings.card_sha256`` in run A's own eval JSONs under
``docs/measurements/data/clay-assistant/run-A/``) as what actually ships, so
this module is where "the card a running Familiar loads" and "the card
training built its dataset's system prompt from" meet: :func:`load_card`
reads the frozen text; :func:`derive_clay_card` still rebuilds the live
equivalent, kept only so a training run (or a test) can prove the two have
not silently drifted apart (:func:`derived_card_sha` versus
``training/clay-assistant/dataset/manifest.json``'s own ``tools_sha``).

**No imgui, moderngl, pygame, httpx, service or queue import, ever, even
transitively at module scope** -- this module has to be importable by
``pipelines/llama.py`` (which runs on the asyncio loop thread, nowhere near a
GL context) and by a training script that must never touch the GPU or the
app's own config while a training run owns the machine. The one exception is
``warlock.studio.agent_clay`` itself, imported lazily inside
:func:`derive_clay_card`'s body -- that registry is what the live half of
this contract is *of*, and nothing else here needs it.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only for the type hints on build_manual_messages/cited below -- a
    # runtime import here would be harmless (retrieval.py is as pure as
    # this module) but pointless, since neither function does anything
    # with a Citation beyond duck-typed attribute access.
    from collections.abc import Sequence

    from . import retrieval

CARDS: dict[str, str] = {"clay": "clay-1.txt", "router": "router-1.txt"}
"""Skill name -> the frozen card file beside this module, under ``cards/``.
Two entries: Clay's own trained card, and T6's router card. The router card
is frozen for a different reason than Clay's -- not because a fine-tune was
trained against its exact bytes, but because the few-shot examples it gives
each skill are exactly what a routing decision is supposed to be stable
against; a card that drifted between runs would make "why did this route
differently today" impossible to answer. See :data:`SAMPLING`'s ``"router"``
row and :func:`build_router_messages` for how it is actually used -- unlike
Clay, the router never gates on a trained weights pin (:func:`~.service.
familiar.ask` passes ``expected_card_sha=None`` for it), because
``card_shas`` records what a *fine-tune* was validated against, and the
router card is prompt-engineered against whatever instruct model is running,
never trained on."""

_CARDS_DIR = Path(__file__).resolve().parent / "cards"


@functools.cache
def load_card(skill: str) -> str:
    """*skill*'s frozen prompt card, read verbatim (UTF-8, no normalisation)
    from ``cards/<file>``. Cached: the file never changes under a running
    process, and this is called once per ``ensure_started``/chat turn."""
    path = _CARDS_DIR / CARDS[skill]
    return path.read_bytes().decode("utf-8")


@functools.cache
def card_sha(skill: str) -> str:
    """``sha256(<card file's own bytes>)``, hex -- compared against
    ``models.FamiliarModel.card_shas`` by ``pipelines/llama.py``'s
    ``ensure_started`` before a skill's llama-server child is ever spawned.
    Hashes the file's raw bytes rather than ``load_card(skill).encode()`` so
    this can never disagree with a plain ``sha256sum`` run against the file
    on disk (a BOM or a decode/round-trip artefact would otherwise be
    invisible here and visible there)."""
    path = _CARDS_DIR / CARDS[skill]
    return hashlib.sha256(path.read_bytes()).hexdigest()


SAMPLING: dict[str, dict[str, float | int]] = {
    "clay": {"temperature": 0.2, "top_k": 64, "top_p": 0.95, "max_tokens": 4096},
    # T5's plain-chat sampling -- there is no measurement behind this row the
    # way there is for "clay" (that one is
    # docs/measurements/2026-09-13-clay-assistant-sampling.md's own pick out
    # of a 232-row corpus): base Gemma 4 E2B has no fine-tune or eval corpus
    # of its own here yet, so this is Google's own stock instruct-model
    # recommendation (t0.7/top-k 64/top-p 0.95), carried over unmeasured. A
    # dated measurement document should replace this comment before the
    # number is trusted for anything beyond "a reasonable default".
    "chat": {"temperature": 0.7, "top_k": 64, "top_p": 0.95, "max_tokens": 1024},
    # Greedy and starved on purpose: the router answers one enum value
    # (:data:`~.router.SKILLS`) through ``response_format``'s constrained
    # decoding, so there is nothing for temperature/top-k/top-p to usefully
    # vary and no reason to let the model ramble past the JSON object.
    # 32, not the 16 first chosen: on the real card (2026-09-14) the model
    # spelled `{"skill": "clay_build"}` with spacing in exactly 16 tokens and
    # finish_reason 'length', so 16 left no margin for the longest names.
    "router": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "max_tokens": 32},
    # Unmeasured, same caveat as "chat" above: base Gemma has no eval corpus
    # for citation-style answers yet. 768 tokens is well past what a few
    # sentences with inline [n] markers needs -- see
    # ``tests/familiar/test_contract.py::
    # test_the_manual_prompt_fits_one_slot_at_the_retrieval_budget`` for the
    # arithmetic that keeps this, plus the retrieval budget, under one
    # llama-server slot.
    "manual": {"temperature": 0.3, "top_k": 64, "top_p": 0.95, "max_tokens": 768},
    # T8: greedy and starved for the same reason "router" is -- constrained
    # decoding to one of a short enum plus "none" (``doors.navigate_schema``),
    # so there is nothing for sampling to vary and no reason to let the model
    # ramble past the JSON object. 24 rather than "router"'s 16: a destination
    # key can be longer than a skill name (``settings:appearance``, a
    # ``tour:<key>``), so the ceiling is generous rather than measured.
    "navigate": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "max_tokens": 24},
    # Unmeasured, same caveat as "manual" above: no eval corpus for a
    # create-draft turn exists yet. 400 tokens is well past what one JSON
    # object with a <=1000-char prompt field needs -- generous rather than
    # tight, because a caller (``familiar_doors.draft_in_create``) never
    # submits this, so a slightly long reply costs latency, not a bad
    # generation.
    "create": {"temperature": 0.3, "top_k": 64, "top_p": 0.95, "max_tokens": 400},
    # T7: greedy and starved, the same "nothing for sampling to vary" reason
    # "router"/"navigate" already give -- constrained decoding to
    # ``character_plan.character_schema``'s own enums/ranges (a species key,
    # a handful of movement names, an int direction count, an int size, a
    # short name). Unmeasured, same caveat as "create": no eval corpus for a
    # character-plan turn exists yet. 200 tokens is generous for one JSON
    # object shaped like that.
    "character": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "max_tokens": 200},
}
"""Per-skill sampling defaults for a real chat turn. Clay's own settings are
``docs/measurements/2026-09-13-clay-assistant-sampling.md``'s own measured
pick, Q8_0 door acceptance out of 232: greedy (t0, 173) and t1.0 n1 (178,
Google's stock Gemma recommendation, what run A was first scored at) were
both beaten by t0.2/top-k 64/top-p 0.95 sampled three times a row (185.0,
versus t1.0's own 177.7 over the same three samples)."""

#: Skills whose reply must be *sized*, via ``/tokenize`` plus
#: :func:`output_budget`, against a trained context window --
#: :data:`CARDS` membership is not the right key for this: the router card
#: is in ``CARDS`` too (it is frozen), but the router's reply is a fixed
#: twelve-token JSON object with no trained window to overrun, and paying
#: for a ``/tokenize`` round trip before every routing decision would slow
#: down the one request Familiar's design wants to feel instant. Only a
#: fine-tune has a trained window that a flat ``max_tokens`` can overrun --
#: today, only Clay.
SIZED_SKILLS: frozenset[str] = frozenset({"clay"})

CHAT_SYSTEM = (
    "You are Familiar, Warlock Studio's offline assistant. You run entirely "
    "on this machine and never reach the network. Right now you can only "
    "talk -- you cannot see or change anything in the app from this "
    "conversation yet. Answer briefly, and say plainly when you are not "
    "sure of something rather than guessing."
)
"""The short system prompt for plain chat, on the base testing pin, in every
mode -- unlike Clay's frozen card (:data:`CARDS`), this is not trained
against and carries no hash pin: it is prose for whatever instruct model
happens to be behind the base weights row, not a contract a fine-tune was
built to match."""

#: How many of a thread's most recent turns :func:`build_chat_messages` folds
#: into a plain-chat prompt. One llama-server slot is ``CTX_SIZE //
#: PARALLEL_SLOTS == 8192`` tokens (``pipelines/llama.py``); at a generous
#: few hundred tokens per turn, eight turns (four back-and-forths) leaves
#: comfortable room for the system prompt and the reply budget without
#: measuring real conversations first. Unmeasured, same caveat as
#: ``SAMPLING["chat"]`` above -- the point is a chat thread degrades by
#: forgetting its oldest turns, never by having the whole request refused
#: the way an oversized Clay prompt is.
HISTORY_TURNS = 8


def build_chat_messages(prompt: str, history: tuple[Any, ...] = ()) -> list[dict[str, str]]:
    """One chat-completions message list for plain conversation: the short
    system prompt (:data:`CHAT_SYSTEM`), up to the last :data:`HISTORY_TURNS`
    turns of *history* (``threads.Turn``, ``role`` mapped ``"familiar"`` ->
    ``"assistant"``), then *prompt* as the final user turn.

    Unlike :func:`build_messages` (Clay's frozen card plus one merged
    "here is the scene" user turn), plain chat has no scene to compact and no
    card frozen against a training run -- it sends real chat-completions
    history because there is nothing here it could disagree with, the way a
    scene-shaped extra turn would disagree with what Clay's fine-tune was
    trained to see (``threads.py``'s own docstring: a thread is display and
    refinement context, never fed back as model context, *for a skill with a
    frozen card* -- plain chat has no such card to protect).
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": CHAT_SYSTEM}]
    role_map = {"user": "user", "familiar": "assistant"}
    for turn in history[-HISTORY_TURNS:]:
        messages.append({"role": role_map.get(turn.role, turn.role), "content": turn.text})
    messages.append({"role": "user", "content": prompt})
    return messages


def build_router_messages(prompt: str, mode: str) -> list[dict[str, str]]:
    """One ``[system, user]`` turn for the router: :data:`CARDS`'s
    ``"router"`` card as the system message, *mode* folded into the user
    turn alongside *prompt* -- "make a chair" means something different sent
    from Clay than sent from Home, and the card's own few-shot lines have
    nothing else to disambiguate that with."""
    return [
        {"role": "system", "content": load_card("router")},
        {"role": "user", "content": f"[mode: {mode}] {prompt}"},
    ]


MANUAL_SYSTEM = (
    "Answer the user's question using only the numbered excerpts below, "
    "each drawn from Warlock Studio's own Manual. Cite every claim you make "
    "with the excerpt's own [n] marker. If the excerpts do not answer the "
    "question, say plainly that the Manual does not cover it rather than "
    "guessing or answering from anything else. Be brief."
)
"""The system prompt for a retrieval-grounded Manual answer -- unlike
:data:`CHAT_SYSTEM`, this is not "answer from what you know", it is "answer
only from what you were just handed", because :func:`~.service.familiar.
cited` can only ever point a reader at a section :func:`build_manual_messages`
actually gave the model; anything else in the reply would be an uncited
claim with no [n] a citation link could ever attach to."""


def build_manual_messages(
    prompt: str, citations: Sequence[retrieval.Citation]
) -> list[dict[str, str]]:
    """One ``[system, user]`` turn for a Manual answer: :data:`MANUAL_SYSTEM`
    as the system message, every one of *citations* (``retrieval.Citation``)
    rendered ``"[n] <title path>\\n<text>"`` and joined, then *prompt*, as
    the user message -- so the model sees each excerpt numbered exactly the
    way :func:`~.service.familiar.cited` (and the pane's own citation links)
    expect a ``[n]`` marker in the reply to mean."""
    excerpts = "\n\n".join(f"[{c.n}] {c.title_path}\n{c.text}" for c in citations)
    return [
        {"role": "system", "content": MANUAL_SYSTEM},
        {"role": "user", "content": excerpts + "\n\n" + prompt},
    ]


#: How many real (BPE) tokens a whitespace-counted word can hide -- both
#: :data:`SAMPLING`'s manual arithmetic and ``retrieval.Index.search``'s own
#: ``budget_tokens`` count *whitespace* tokens (``retrieval._whitespace_
#: token_count``), which undercounts a real tokenizer's output (a word often
#: splits into more than one subword piece). Undercounting here is the same
#: unsafe direction ``llama_client.TEMPLATE_MARGIN_TOKENS``'s own docstring
#: warns about, so a slot-fit check must inflate the word count rather than
#: trust it directly. 1.5x is a generous ceiling for English prose, not a
#: measurement -- there is no fine-tune or eval corpus for Manual answers
#: yet to measure a real ratio from.
WORD_TOKEN_SAFETY = 1.5


_CITATION_MARKER = re.compile(r"\[(\d+)\]")


def cited(
    reply: str, citations: Sequence[retrieval.Citation]
) -> tuple[retrieval.Citation, ...]:
    """The subsequence of *citations* whose ``[n]`` marker actually appears
    in *reply*, in first-appearance order, each number kept only once.

    A marker naming a number outside ``1..len(citations)`` -- the model
    inventing a source it was never handed -- is silently skipped rather
    than raised: a citation link must never point at a section
    :func:`build_manual_messages` did not actually retrieve, and a dead
    link is worse than a missing one.
    """
    by_n = {c.n: c for c in citations}
    seen: set[int] = set()
    result = []
    for match in _CITATION_MARKER.finditer(reply):
        n = int(match.group(1))
        if n in seen or n not in by_n:
            continue
        seen.add(n)
        result.append(by_n[n])
    return tuple(result)


# ---------------------------------------------------------------------------
# The tool card itself -- KEEP_TOOLS, the behaviour paragraph, the sentence-
# and summary-trimming helpers, and derive_clay_card/derived_card_sha. Moved
# verbatim from training/clay-assistant/gen/convert.py's compact_tools()/
# tools_sha(); convert.py re-exports the old names so build.py, run_val.py,
# drafts/_gen_queries.py and the training tests keep working unchanged.
# ---------------------------------------------------------------------------

KEEP_TOOLS: tuple[str, ...] = (
    "clay_batch",
    "clay_scene",
    "clay_add_primitive",
    "clay_add_figure",
    "clay_transform",
    "clay_set_params",
    "clay_material",
    "clay_boolean",
    "clay_select",
    "clay_op",
    "clay_delete",
    "clay_rename",
    "clay_diagnose",
)
"""The compact tool card's membership, in the order it is printed. Thirteen,
not the plan's original twelve: ``clay_add_figure`` was omitted from the
first count and folded in once figures were counted, per the plan's own
parenthetical."""

BEHAVIOUR_PARAGRAPH = (
    "You are Warlock's Clay assistant. Answer a build request with exactly "
    "one clay_batch tool call whose calls list builds the object; name "
    "every object; put generator parameters under params; place objects so "
    "they rest on the ground; reuse a material index from clay_scene when "
    "one fits, otherwise add one clay_material call at the end. Answer a "
    "question about the scene in one or two sentences with no tool call."
)
"""Verbatim from the plan's "convert.compact_tools()" paragraph -- the one
piece of this card that is prose about training behaviour rather than
derived from a live registry, because nothing in ``agent_clay`` already says
"answer with exactly one clay_batch call"; that is this dataset's own
convention, not the app's."""

_SENTENCE_RE = re.compile(r".+?\.(?=\s|$)", re.S)

_ENUMERATION_PREFIXES = ("Known generators:", "Known ops:", "Parts,")
"""The three catalogue sentences run A's own refusals show a fine-tune is
scored on (measured on 232 val+corpus rows, Q8_0): ``clay_add_primitive``/
``clay_set_params``'s "Known generators: cylinder [radius, height,
segments] ..." (7 refusals for an unknown generator param, e.g. ``depth`` on
a cylinder), ``clay_op``'s "Known ops: array-radial [count, angle, axis]
..." (``clay_op`` given ``axis`` outside ``params`` among 6 other refusals),
and ``clay_add_figure``'s "Parts, each prefixed by name_prefix: humanoid:
Hips, Spine, ..." (15 ``no object named '...'`` refusals, nine of them
creatures-family guesses at a generated figure's own part names --
``hound_Beak``, ``t_Shank.R``, ``s_Tail 01``). ``_first_sentence`` kept only
each description's opening line and trusted the schema's own enums to carry
the rest, which is true for a plain string enum (a generator's *name*, an
op's *name*) but not for what a generator's own params are called, what an
op's own params are called or bounded to, or what a figure preset's own
part names are -- none of that is expressible as a JSON Schema enum here,
because a generator's ``params``/an op's ``params`` is one open
``{string: number}`` object (the value shape varies per key) and a figure's
parts are never an argument at all. These sentences are the only place any
of that is written down."""


def _sentences(text: str) -> list[str]:
    """*text*, split into whole sentences -- the same one-period-plus-
    whitespace-or-end rule ``_first_sentence`` used to apply to the first
    sentence only, walked to the end of the string instead of stopping
    there. A catalogue sentence's own periods (a generator's ``segments=32``
    default, a part name like ``Shoulder.L``) are never followed by
    whitespace, so they never end a sentence early here -- only the period
    that actually closes the sentence, followed by a space or the string's
    end, does."""
    stripped = text.strip()
    sentences: list[str] = []
    pos = 0
    while pos < len(stripped):
        match = _SENTENCE_RE.match(stripped[pos:])
        if not match:
            sentences.append(stripped[pos:])
            break
        sentences.append(match.group(0))
        pos += match.end()
        while pos < len(stripped) and stripped[pos].isspace():
            pos += 1
    return sentences


def _summary(text: str, *, seen: set[str]) -> str:
    """*text*'s first sentence, plus every later sentence beginning
    ``Known generators:``, ``Known ops:`` or ``Parts,`` -- the enumerations
    the model is scored on (see ``_ENUMERATION_PREFIXES``). *seen* is the
    card's own running set of catalogue sentences already printed: dropped
    (a bare first sentence keeps the shorter summary) rather than fenced
    against here, so ``clay_set_params``'s "Known generators: ..." -- word
    for word ``clay_add_primitive``'s own, both built from the same
    :func:`agent_clay._generator_catalog` -- is not printed twice. Measured:
    959 of those chars, once. Anything else in ``_ENUMERATION_PREFIXES`` is
    unique per tool in ``KEEP_TOOLS`` today (``clay_op``'s ops, ``clay_add_
    figure``'s parts), so this dedupe currently only ever fires once, but it
    is a running set rather than a hand-picked "skip clay_set_params" rule
    because the next generator or op added to a second tool's description
    should not have to earn its own special case here."""
    sentences = _sentences(text)
    if not sentences:
        return text.strip()
    keep = [sentences[0]]
    for sentence in sentences[1:]:
        if not sentence.startswith(_ENUMERATION_PREFIXES):
            continue
        if sentence in seen:
            continue
        seen.add(sentence)
        keep.append(sentence)
    return " ".join(keep)


def derive_clay_card() -> str:
    """The system-prompt tool card every training row shares, rebuilt fresh
    from the *live* ``agent_clay`` registries -- never cached, so it can
    never drift from what the running app's own tool surface actually
    publishes. This is ``compact_tools()``, moved out of ``training/clay-
    assistant/gen/convert.py`` unchanged (byte-identical output): what
    training built its dataset's system prompt from, and what
    :func:`derived_card_sha` hashes to compare against ``dataset/
    manifest.json``'s own ``tools_sha``. The card Familiar actually loads at
    runtime is the *frozen* one, :func:`load_card` -- this function exists
    so that comparison can still be made, not to serve a live chat turn.

    Each tool's own summary keeps its first sentence plus its catalogue
    sentences (:func:`_summary`) rather than the first sentence alone: run A
    (2026-09-13, Q8_0, 232 val+corpus rows) showed exactly the refusals that
    dropping them causes -- 15 ``no object named '...'`` (a figure preset's
    part names, nine of them in ``creatures``), 7 unknown params for a
    generator (e.g. ``depth`` on a cylinder), and ``clay_op`` given ``axis``
    outside ``params`` among 6 other refusals naming an op's own arguments.
    See ``training/clay-assistant/README.md``'s "The compact tool card"
    section for the fix's own accounting.

    Imports ``agent_clay`` lazily, inside this function, rather than at
    module scope: this module must stay importable with no imgui/moderngl/
    pygame/httpx/service/queue in the process (``pipelines/llama.py`` on the
    asyncio loop thread, a training script that must never touch the GPU),
    and ``agent_clay`` is the one thing here that is not true of.
    """
    from warlock.studio import agent_clay

    tool_map = {t.name: t for t in agent_clay.tools()}
    instructions = agent_clay.instructions()
    paragraphs = instructions.split("\n\n")

    lines = [paragraphs[0], "", paragraphs[1], "", BEHAVIOUR_PARAGRAPH, "", "Tools:"]
    seen: set[str] = set()
    for name in KEEP_TOOLS:
        tool = tool_map[name]
        summary = _summary(tool.description, seen=seen)
        schema_json = json.dumps(tool.schema, sort_keys=True, separators=(",", ":"))
        lines.append(f"- {name}: {summary}")
        lines.append(f"  schema: {schema_json}")
    return "\n".join(lines)


def derived_card_sha() -> str:
    """``sha256(derive_clay_card())``, hex -- what ``build.py`` pins in
    ``dataset/manifest.json`` and every accepted record's own ``verified``
    block, so a registry change (a new generator, say) that regenerates a
    different tool card is caught rather than silently appended past. This
    is ``tools_sha()``, moved here unchanged."""
    return hashlib.sha256(derive_clay_card().encode("utf-8")).hexdigest()


@functools.cache
def allowed_calls(skill: str) -> frozenset[str]:
    """The tool names *skill*'s frozen card actually lets a ``clay_batch``
    entry name -- parsed from the card's own ``clay_batch`` schema line
    (the ``"name":{"enum":[...]}`` JSON Schema enum), not from whatever
    ``agent_clay.tools()`` publishes live today. The frozen card is what a
    running model was actually trained to see; the live registry can grow a
    fourteenth tool tomorrow and this must not silently start accepting a
    call name the model has never read a schema for.

    Cached per skill: the card file does not change under a running
    process."""
    card = load_card(skill)
    lines = card.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("- clay_batch:"):
            schema_line = lines[i + 1].strip()
            break
    else:
        raise ValueError(f"{skill!r} card has no clay_batch tool line")
    prefix = "schema:"
    if not schema_line.startswith(prefix):
        raise ValueError(f"{skill!r} card's clay_batch line has no schema: {schema_line!r}")
    schema = json.loads(schema_line[len(prefix) :].strip())
    enum = schema["properties"]["calls"]["items"]["properties"]["name"]["enum"]
    return frozenset(enum)


_SCENE_ROW_KEYS: tuple[str, ...] = (
    "uid",
    "name",
    "generator",
    "params",
    "translation",
    "rotation",
    "scale",
    "size",
    "material",
)
"""The Verifier design section's "Compact scene for edit/query rows"
paragraph, verbatim: ``_scene_row`` carries fifteen fields; the training
state turn keeps these nine plus the document's own ``materials``/``bounds``,
dropping ``faces``/``verts``/``stamp``/``selected``/``visible``/``bbox``/
``center`` -- the ones a token budget cannot afford and an edit prompt does
not need to answer."""


def compact_scene(structured: dict[str, Any]) -> dict[str, Any]:
    """*structured* (a ``clay_scene`` ``structuredContent`` payload),
    projected to the fields an edit/query training row's "here is the scene"
    turn actually needs. Moved from ``training/clay-assistant/gen/
    convert.py`` unchanged."""
    objects = [
        {key: row.get(key) for key in _SCENE_ROW_KEYS} for row in structured.get("objects", [])
    ]
    return {
        "objects": objects,
        "materials": structured.get("materials", []),
        "bounds": structured.get("bounds"),
    }


def user_turn(prompt: str, scene: dict[str, Any] | None = None) -> str:
    """The user turn exactly as *training* built it for an edit/query row:
    ``"Here is the scene:\\n" + <compact json, sorted keys, no whitespace>
    + "\\n\\n" + prompt`` -- the same same-role merge ``train/train_a.py``
    (:117-136) folds into one user turn, with ``json.dumps(scene,
    sort_keys=True, separators=(",", ":"))`` matching that file's own compact
    encoding. *scene* is ``None`` for a plain build row (no prior state to
    describe), in which case this is just *prompt*.

    ``training/clay-assistant/eval/run_val.py``'s own ``_user_turn`` builds
    the same shape but with ``json.dumps(scene, sort_keys=True)`` -- the
    *default* separators (``", "``/``": "``), not this compact form. That is
    a real, deliberate difference, not a bug: run A's whole eval corpus was
    scored against that slightly longer encoding, and switching ``run_val.py``
    over to this compact one now would change every prompt's token count and
    make a new eval no longer comparable with ``docs/measurements/data/
    clay-assistant/run-A/``'s own recorded numbers. ``run_val.py`` keeps its
    own ``_user_turn`` for that reason; this function is what a *new* caller
    (Familiar's own runtime, a future eval line) should build on.
    """
    if scene is None:
        return prompt
    scene_text = json.dumps(scene, sort_keys=True, separators=(",", ":"))
    return "Here is the scene:\n" + scene_text + "\n\n" + prompt


def build_messages(
    skill: str, prompt: str, scene: dict[str, Any] | None = None
) -> list[dict[str, str]]:
    """One ``[system, user]`` chat turn for *skill*: the frozen card as the
    system message, :func:`user_turn`'s shape as the user message."""
    return [
        {"role": "system", "content": load_card(skill)},
        {"role": "user", "content": user_turn(prompt, scene)},
    ]


# ---------------------------------------------------------------------------
# Reply grammar -- moved from training/clay-assistant/eval/run_val.py, same
# fence and the same three failure details ("no fenced json", "json: ...",
# "no calls list"). run_val.py and eval/tier_two.py both import parse_calls
# from here now.
# ---------------------------------------------------------------------------

FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.S)


# ---------------------------------------------------------------------------
# Context window -- run A's trained max sequence length is one llama-server
# slot, not the server's whole context. See output_budget's own docstring.
# ---------------------------------------------------------------------------

TRAINED_WINDOW = 8192
"""Run A's own trained max sequence length (``train/train_a.py``'s
``max_seq_length``). ``pipelines/llama.py`` runs the server at
``CTX_SIZE = 16384`` with ``PARALLEL_SLOTS = 2``, so each slot gets exactly
this many tokens -- the server's own total context is not the number a single
chat turn has to fit inside."""

MIN_REPLY_TOKENS = 1893
"""A floor below which :func:`output_budget` refuses rather than hand back a
reserve too small for any real Clay reply. Measured 2026-09-14: the longest
assistant reply text (a build/edit row's fenced ``{"calls": [...]}}`` JSON, or
a query row's ``answer``) across ``training/clay-assistant/dataset/{train,
val}.jsonl`` is 5,167 chars (``vehicles-0029``); at the same 2.73 chars/token
ratio :func:`output_budget`'s own caller measures prompts with, that is
``ceil(5167 / 2.73) == 1893`` tokens."""


def output_budget(skill: str, prompt_tokens: int) -> int:
    """How many tokens *skill*'s reply may use, given a prompt of
    *prompt_tokens* -- the smaller of the skill's own configured
    ``max_tokens`` (:data:`SAMPLING`) and whatever is left of
    :data:`TRAINED_WINDOW` after the prompt. Refuses (``ValueError``) rather
    than hand back a reserve under :data:`MIN_REPLY_TOKENS`: a real Clay reply
    cannot fit in less, so a caller that got there sized its prompt wrong
    rather than found a valid budget."""
    remaining = TRAINED_WINDOW - prompt_tokens
    budget = min(SAMPLING[skill]["max_tokens"], remaining)
    if budget < MIN_REPLY_TOKENS:
        raise ValueError(
            f"{skill!r}: only {budget} reply tokens left of the {TRAINED_WINDOW}-token "
            f"window after a {prompt_tokens}-token prompt -- under the "
            f"{MIN_REPLY_TOKENS}-token floor a real reply needs"
        )
    return budget


def parse_calls(reply: str) -> tuple[list[dict] | None, str | None]:
    """Parse *reply* the way the training convention expects: a ```json```
    fence whose body is ``{"calls": [...]}}``. Returns ``(calls, None)`` on
    success or ``(None, detail)`` naming which of the three failure modes
    hit -- no fence, bad JSON, or no ``calls`` list."""
    m = FENCE.search(reply)
    if m is None:
        return None, "no fenced json"
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        return None, f"json: {exc}"[:200]
    calls = obj.get("calls") if isinstance(obj, dict) else None
    if not isinstance(calls, list) or not calls:
        return None, "no calls list"
    return calls, None
