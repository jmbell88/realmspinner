"""``clay_program``'s compiler: a small declarative program -> an expanded,
validated list of tool calls. Pure -- stdlib and numpy-free registry reads
only (``primitives.GENERATORS``, ``presets.ASSEMBLIES``, ``clay_ops.OPS``),
no imgui/moderngl/pygame/service -- because it runs before there is a
document, or even a session, to run against.

**This module does not execute anything.** :func:`compile_program` turns a
program into a :class:`Compiled` -- a flat list of ``(tool_name, arguments,
path)`` entries plus ``("live", kind, arguments, path)`` placeholders for
the four kinds in :data:`LIVE_KINDS` -- and stops there. ``agent_clay._h_program``
is what runs the result, folding ``calls`` into one atomic run through the
same ``_fold_run`` ``clay_batch`` uses (reusing that tool's own ``$ref``
convention, which is why a compiled reference is already shaped
``{"$ref": "<name>"}`` rather than something this module invents). A
``("live", kind, ...)`` placeholder's own turn in that run does not call
``call()`` the way a real tool entry does: ``move``/``turn``/``scale_by``
read the target's current transform straight off the live document, compose
*this step's own* delta onto it, and issue their own ``clay_transform``
call per resolved target (one entry per group member, so a group of N still
folds into the program's single undo step); ``assert`` evaluates its
condition -- compiled here to a validated AST plus the ``$var`` scope it
closed over, never a raw string re-parsed later -- against :data:`FACTS`,
and a false result (or a :class:`ConditionError`) refuses the whole run
exactly like a failed real tool call would. None of that needs a *name*
this module could compile ahead of time to, which is why these four stay
placeholders here rather than becoming a fifth shape of ``calls`` entry.
Because there is no document here, an id resolves to a *name* -- ``{"$ref":
name}`` for a real tool call's argument, or a bare ``("id", name)`` AST leaf
inside an ``assert`` condition -- never to a uid; a literal integer or
``{"uid": n}``/``{"uids": [...]}`` passes straight through instead,
addressing an object that already exists live, outside this program's own
namespace.

**Expressions** are a tiny numeric mini-language, never Python: a tokenizer,
a recursive-descent parser (precedence climbing over binary operators) into
tuple nodes, and an evaluator over those nodes -- no ``eval``, ``exec``,
``compile`` or ``ast`` of caller-supplied text anywhere, so a string like
``"__import__('os')"`` is refused as a bad token (there is no string-literal
grammar to hold the quoted part, and ``__import__`` is not a known
function) rather than executed. Power is spelled ``^`` -- ``**`` is not
special-cased and would parse as two ``*`` tokens, refused as a bad token
sequence by the parser rather than silently meaning something else.

**Two different placeholder syntaxes, on purpose.** A numeric grammar field
(``translation``, ``params`` values, a range bound, ...) takes a plain
number or a ``$name``-bearing expression string, because it is a number
either way once evaluated -- ``"5"`` and ``5`` mean the same thing, and a
string with no ``$`` or operator in it still parses (trivially) rather than
needing a special case. An ``id`` field is never a number, so it takes a
different grammar: a plain string, or one templated with ``{name}``
placeholders (``"leg_{i}"``), substituted from the current scope. Mixing the
two -- ``$`` in an id, or ``{}`` in a numeric field -- is refused rather than
guessed at.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .....kernels.mesh import presets
from .....kernels.mesh import primitives as bp
from .. import ops as clay_ops

__all__ = [
    "EXPR_MAX_CHARS",
    "EXPR_MAX_DEPTH",
    "FACTS",
    "PROGRAM_MAX_BOOLEANS",
    "PROGRAM_MAX_CALLS",
    "PROGRAM_MAX_NESTING",
    "PROGRAM_MAX_REPEAT",
    "PROGRAM_MAX_STEPS",
    "PROGRAM_MAX_VARIABLES",
    "STEP_KINDS",
    "UID_BEARING_KEYS",
    "LIVE_KINDS",
    "Compiled",
    "ConditionError",
    "ProgramError",
    "compile_program",
    "evaluate_condition",
]

# --- limits ------------------------------------------------------------

EXPR_MAX_CHARS = 256
"""An expression string longer than this is refused before it is even
tokenized -- the same reasoning ``protocol.MAX_FRAME`` uses at the wire, one
level down: a program's whole point is compactness, and an expression this
long is already better written as several ``let`` steps."""

EXPR_MAX_DEPTH = 32
"""Bounds parser recursion -- nested parentheses, function-call arguments,
and a chain of unary ``-``/``+`` -- not the flat width of a same-precedence
chain (``$a+$b+$c+...``), which ``EXPR_MAX_CHARS`` already bounds far
earlier in practice. The two limits are independent on purpose: one guards
the parser's own call stack, the other guards how much an agent can ask for
in one field."""

PROGRAM_MAX_STEPS = 64
"""The most entries any one ``steps`` list may hold -- the top-level list and
every ``repeat``/``if`` branch's own, independently."""

PROGRAM_MAX_CALLS = 256
"""The most *expanded* tool-call units one program may compile to. A
``clay_add_figure`` counts as its preset's own part count (``presets.build``)
rather than 1, because it is one wire call standing in for that many objects;
every other step counts each wire call it emits (``clay_op``'s two-call
``op`` step included) as 1. Checked as the running total grows, so the
refusal names the step that pushed it over rather than a final tally with no
address."""

PROGRAM_MAX_REPEAT = 64
"""The most iterations one ``repeat`` (or the ``array`` sugar built on it)
may produce -- the product of every named range's own length."""

PROGRAM_MAX_NESTING = 4
"""How many ``steps`` lists may sit inside one another -- the top-level list
is depth 1, a ``repeat`` or taken ``if`` branch's own list is one deeper."""

PROGRAM_MAX_BOOLEANS = 4
"""The most ``boolean`` steps one compiled program may contain, counting
every iteration a ``repeat`` expands one into. Each is `MAX_BOOLEAN_TRIANGLES`
work on the frame thread (see ``dev/INVARIANTS.md``'s agent paragraph); a
program is one MCP round trip and should not be able to queue an unbounded
amount of that behind it."""

PROGRAM_MAX_VARIABLES = 64
"""The most variable names simultaneously in scope -- the top-level
``variables`` block plus every ``let``/``repeat`` binding visible at a given
point in the program, checked as each is added."""


# --- errors --------------------------------------------------------------


class ProgramError(Exception):
    """One refusal shape for the whole compiler.

    ``field`` is one of ``"steps"``, ``"variables"`` or ``"dry_run"`` --
    ``clay_program``'s own three top-level arguments, the ones a client
    would actually branch on or highlight -- and ``path`` is the exact
    location within whichever of those the refusal came from, e.g.
    ``"steps[3].repeat.steps[1].add.translation[0]"``. Both are also folded
    into the exception's own message, because an agent reading this refusal
    has no second channel to read ``path`` from unless a caller goes out of
    its way to thread it through -- ``str(exc)`` alone is always enough to
    act on.
    """

    def __init__(self, message: str, *, field: str, path: str) -> None:
        located = f"{path}: {message}" if path else message
        super().__init__(located)
        self.field = field
        self.path = path
        self.reason = message


def _err(message: str, *, field: str, path: str) -> ProgramError:
    return ProgramError(message, field=field, path=path)


class _ExprError(Exception):
    """Internal only -- caught at the field boundary and re-raised as a
    :class:`ProgramError` carrying that field's own path."""


# --- the grammar table -----------------------------------------------------

STEP_KINDS: dict[str, frozenset[str]] = {
    "add": frozenset(
        {"generator", "params", "translation", "rotation", "scale", "id", "material"}
    ),
    "figure": frozenset({"key", "translation", "yaw", "scale", "id"}),
    "mesh": frozenset(
        {"positions", "faces", "uv", "translation", "rotation", "scale", "id", "material"}
    ),
    "transform": frozenset({"uid", "translation", "rotation", "scale"}),
    "params": frozenset({"uid", "uids", "params"}),
    "material": frozenset({"uids", "name", "color", "metallic", "roughness"}),
    "delete": frozenset({"uids"}),
    "op": frozenset({"name", "params", "uids"}),
    "boolean": frozenset({"kind", "uids"}),
    "select": frozenset({"uids"}),
    "repeat": frozenset({"ranges", "steps"}),
    "array": frozenset({"id", "count", "var", "add"}),
    "mirror": frozenset({"axis", "add"}),
    "group": frozenset({"id", "members"}),
    "let": frozenset({"vars"}),
    "if": frozenset({"cond", "then", "else"}),
    "move": frozenset({"uid", "by"}),
    "turn": frozenset({"uid", "by"}),
    "scale_by": frozenset({"uid", "factor"}),
    "assert": frozenset({"uid", "condition"}),
}
"""Every step kind's own top-level keys, exhaustive -- a key not in this set
for its kind is refused before anything about its *value* is looked at.
``tests/modes/clay/test_agent_program.py`` (and a later transcript test) walk this
rather than a hand-kept prose list, the same rule ``agent_clay.tools()``
follows for its own registries."""

UID_BEARING_KEYS: dict[str, frozenset[str]] = {
    "transform": frozenset({"uid"}),
    "params": frozenset({"uid", "uids"}),
    "material": frozenset({"uids"}),
    "delete": frozenset({"uids"}),
    "op": frozenset({"uids"}),
    "boolean": frozenset({"uids"}),
    "select": frozenset({"uids"}),
    "move": frozenset({"uid"}),
    "turn": frozenset({"uid"}),
    "scale_by": frozenset({"uid"}),
    "assert": frozenset({"uid"}),
}
"""Which of each kind's own keys carry a reference (a uid, a program id, a
group, or ``$ref``) rather than a plain value -- a subset of that kind's
entry in :data:`STEP_KINDS`, and absent entirely for a kind that carries
none (``add``, ``figure``, ``mesh``, which only ever *create*; ``repeat``,
``array``, ``mirror``, ``group``, ``let``, ``if``, which are compile-time
sugar or control flow)."""

LIVE_KINDS = frozenset({"move", "turn", "scale_by", "assert"})
"""Step kinds this compiler shape-checks and validates fully but never
compiles to a fixed tool call, because each needs the live document a batch
alone cannot answer for: ``move``/``turn``/``scale_by`` read the target's
*current* transform before they know what absolute values to send, and
``assert`` evaluates its condition against the document's live geometry
(:data:`FACTS`). Each compiles to a ``("live", kind, arguments, path)``
placeholder instead -- ``agent_clay._h_program`` is what actually runs one,
per its own turn in the same fold a real tool call's entry runs in, so a
program with a live step is still one atomic undo step start to finish."""

_CREATOR_KINDS = frozenset({"add", "figure", "mesh"})
_WRAPPER_KINDS = frozenset({"repeat", "array", "mirror", "group", "let", "if"})
_TOP_LEVEL_KEYS = frozenset({"variables", "steps", "dry_run"})


# --- expressions: tokenizer -------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
      (?P<WS>\s+)
    | (?P<NUM>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
    | (?P<VAR>\$[A-Za-z_][A-Za-z0-9_]*)
    | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<LE><=) | (?P<GE>>=) | (?P<EQ>==) | (?P<NE>!=)
    | (?P<LT><) | (?P<GT>>)
    | (?P<PLUS>\+) | (?P<MINUS>-) | (?P<STAR>\*) | (?P<SLASH>/)
    | (?P<PERCENT>%) | (?P<CARET>\^)
    | (?P<LPAREN>\() | (?P<RPAREN>\)) | (?P<COMMA>,)
    """,
    re.VERBOSE,
)

_KEYWORDS = frozenset({"and", "or", "not"})

_COMPARISON_OPS = frozenset({"<", "<=", ">", ">=", "==", "!="})
_ADD_OPS = frozenset({"+", "-"})
_MUL_OPS = frozenset({"*", "/", "%"})


@dataclass(frozen=True)
class _Tok:
    kind: str
    text: str
    pos: int


def _tokenize(text: str) -> list[_Tok]:
    tokens: list[_Tok] = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise _ExprError(f"bad token at position {pos}: {text[pos:pos + 1]!r}.")
        kind = m.lastgroup
        assert kind is not None
        if kind != "WS":
            tokens.append(_Tok(kind, m.group(), pos))
        pos = m.end()
    tokens.append(_Tok("EOF", "", n))
    return tokens


# --- expressions: parser (tuple nodes) --------------------------------------
#
# Node shapes: ("num", float) | ("var", name) | ("const", name)
#            | ("call", name, [nodes]) | ("neg"|"pos"|"not", node)
#            | ("binop", op, left, right)


class _Parser:
    def __init__(self, tokens: list[_Tok], *, allow_ids: bool = False) -> None:
        self.toks = tokens
        self.i = 0
        self.depth = 0
        # False for every numeric-grammar field (translation, a range bound,
        # ...), where a bare identifier has always been a mistake -- $name is
        # how those fields spell a variable. True only for an ``assert``
        # condition (see ``_parse_condition``), the one grammar that needs a
        # bare identifier to name an object rather than a number.
        self.allow_ids = allow_ids

    def _peek(self) -> _Tok:
        return self.toks[self.i]

    def _advance(self) -> _Tok:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > EXPR_MAX_DEPTH:
            raise _ExprError(f"expression nesting exceeds EXPR_MAX_DEPTH ({EXPR_MAX_DEPTH}).")

    def _exit(self) -> None:
        self.depth -= 1

    def parse(self) -> Any:
        node = self._or()
        if self._peek().kind != "EOF":
            tok = self._peek()
            raise _ExprError(f"unexpected token {tok.text!r} at position {tok.pos}.")
        return node

    # or -> and (OR and)*
    def _or(self) -> Any:
        self._enter()
        try:
            left = self._and()
            while self._peek().kind == "IDENT" and self._peek().text == "or":
                self._advance()
                right = self._and()
                left = ("binop", "or", left, right)
            return left
        finally:
            self._exit()

    # and -> not_expr (AND not_expr)*
    def _and(self) -> Any:
        left = self._not_expr()
        while self._peek().kind == "IDENT" and self._peek().text == "and":
            self._advance()
            right = self._not_expr()
            left = ("binop", "and", left, right)
        return left

    # not_expr -> NOT not_expr | comparison
    def _not_expr(self) -> Any:
        if self._peek().kind == "IDENT" and self._peek().text == "not":
            self._advance()
            self._enter()
            try:
                operand = self._not_expr()
            finally:
                self._exit()
            return ("not", operand)
        return self._comparison()

    # comparison -> additive (CMP additive)?
    def _comparison(self) -> Any:
        left = self._additive()
        tok = self._peek()
        op = {"LT": "<", "LE": "<=", "GT": ">", "GE": ">=", "EQ": "==", "NE": "!="}.get(tok.kind)
        if op is not None:
            self._advance()
            right = self._additive()
            return ("binop", op, left, right)
        return left

    # additive -> multiplicative ((+|-) multiplicative)*
    def _additive(self) -> Any:
        left = self._multiplicative()
        while self._peek().kind in ("PLUS", "MINUS"):
            op = self._advance().text
            right = self._multiplicative()
            left = ("binop", op, left, right)
        return left

    # multiplicative -> unary ((*|/|%) unary)*
    def _multiplicative(self) -> Any:
        left = self._unary()
        while self._peek().kind in ("STAR", "SLASH", "PERCENT"):
            op = self._advance().text
            right = self._unary()
            left = ("binop", op, left, right)
        return left

    # unary -> (+|-) unary | power
    def _unary(self) -> Any:
        if self._peek().kind in ("PLUS", "MINUS"):
            op = self._advance().kind
            self._enter()
            try:
                operand = self._unary()
            finally:
                self._exit()
            return ("neg", operand) if op == "MINUS" else ("pos", operand)
        return self._power()

    # power -> atom (^ unary)?   -- right-associative, binds tighter than unary
    def _power(self) -> Any:
        left = self._atom()
        if self._peek().kind == "CARET":
            self._advance()
            right = self._unary()
            return ("binop", "^", left, right)
        return left

    def _atom(self) -> Any:
        tok = self._peek()
        if tok.kind == "NUM":
            self._advance()
            return ("num", float(tok.text))
        if tok.kind == "VAR":
            self._advance()
            return ("var", tok.text[1:])
        if tok.kind == "LPAREN":
            self._advance()
            self._enter()
            try:
                node = self._or()
            finally:
                self._exit()
            self._expect("RPAREN", "expected ')'")
            return node
        if tok.kind == "IDENT":
            name = tok.text
            if name in _KEYWORDS:
                raise _ExprError(f"unexpected keyword {name!r} at position {tok.pos}.")
            self._advance()
            if self._peek().kind == "LPAREN":
                self._advance()
                args = []
                if self._peek().kind != "RPAREN":
                    self._enter()
                    try:
                        args.append(self._or())
                        while self._peek().kind == "COMMA":
                            self._advance()
                            args.append(self._or())
                    finally:
                        self._exit()
                self._expect("RPAREN", "expected ')'")
                return ("call", name, args)
            if name == "pi":
                return ("const", "pi")
            if self.allow_ids:
                return ("id", name)
            raise _ExprError(
                f"unknown name {name!r} at position {tok.pos} (expected a function call)."
            )
        raise _ExprError(f"unexpected token {tok.text!r} at position {tok.pos}.")

    def _expect(self, kind: str, message: str) -> _Tok:
        tok = self._peek()
        if tok.kind != kind:
            raise _ExprError(f"{message}; got {tok.text!r} at position {tok.pos}.")
        return self._advance()


# --- expressions: functions and evaluation ----------------------------------


def _arity_ok(name: str, lo: int, hi: int | None, n: int) -> bool:
    return n >= lo and (hi is None or n <= hi)


_FUNCTIONS: dict[str, tuple[int, int | None, Any]] = {
    "sin": (1, 1, lambda x: math.sin(math.radians(x))),
    "cos": (1, 1, lambda x: math.cos(math.radians(x))),
    "tan": (1, 1, lambda x: math.tan(math.radians(x))),
    "asin": (1, 1, lambda x: math.degrees(math.asin(x))),
    "acos": (1, 1, lambda x: math.degrees(math.acos(x))),
    "atan2": (2, 2, lambda y, x: math.degrees(math.atan2(y, x))),
    "sqrt": (1, 1, math.sqrt),
    "abs": (1, 1, abs),
    "min": (2, None, lambda *a: min(a)),
    "max": (2, None, lambda *a: max(a)),
    "floor": (1, 1, lambda x: float(math.floor(x))),
    "ceil": (1, 1, lambda x: float(math.ceil(x))),
    "clamp": (3, 3, lambda x, lo, hi: min(max(x, lo), hi)),
    "lerp": (3, 3, lambda a, b, t: a + (b - a) * t),
    "round": (1, 2, lambda x, n=0.0: float(round(x, int(n)))),
}
"""``sin``/``cos``/``tan`` take degrees; ``asin``/``acos``/``atan2`` return
degrees, matching every rotation field in this program grammar -- an
expression that computes an angle never needs a caller-side ``radians()``
conversion this language does not have."""


def _apply_binop(op: str, left: float, right: float) -> float:
    """The arithmetic/comparison half of a ``binop`` node -- shared by
    :func:`_evaluate` (compile-time, a numeric field) and
    :func:`evaluate_condition` (run-time, an ``assert``), so the two stay one
    language rather than drifting into two dialects of ``+``. ``and``/``or``
    are not here -- both evaluators short-circuit them themselves, before
    this is ever reached, so the right operand is only evaluated when it
    matters.

    Raises plain ``ValueError`` on a divide-by-zero or non-finite power;
    each caller wraps that in its own exception type (``_ExprError`` at
    compile time, :class:`ConditionError` at run time) rather than this
    shared helper picking one on their behalf.
    """
    if op in _COMPARISON_OPS:
        result = {
            "<": left < right, "<=": left <= right, ">": left > right,
            ">=": left >= right, "==": left == right, "!=": left != right,
        }[op]
        return 1.0 if result else 0.0
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        if right == 0.0:
            raise ValueError("division by zero.")
        return left / right
    if op == "%":
        if right == 0.0:
            raise ValueError("division by zero.")
        return math.fmod(left, right)
    if op == "^":
        try:
            return float(left**right)
        except ZeroDivisionError:
            raise ValueError("division by zero.") from None
        except (ValueError, OverflowError):
            raise ValueError("expression evaluates to a non-finite number.") from None
        except TypeError:
            # 2026-09-16 audit: a negative `left` with a non-integer `right`
            # (e.g. `(0-4)^0.5`) makes `float.__pow__` return a `complex`,
            # and `float(complex)` raises `TypeError`, not `ValueError` --
            # neither caller's `except ValueError` catches that, so it used
            # to escape uncaught instead of becoming this refusal.
            raise ValueError("expression evaluates to a non-finite number.") from None
    # pragma: no cover - the parser emits no other op
    raise AssertionError(f"unknown operator {op!r}.")


def _evaluate(node: Any, scope: dict[str, float]) -> float:
    tag = node[0]
    if tag == "num":
        return node[1]
    if tag == "var":
        name = node[1]
        if name not in scope:
            raise _ExprError(f"unknown variable ${name}.")
        return scope[name]
    if tag == "const":
        if node[1] == "pi":
            return math.pi
        raise _ExprError(f"unknown constant {node[1]!r}.")  # pragma: no cover - unreachable
    if tag == "neg":
        return -_evaluate(node[1], scope)
    if tag == "pos":
        return +_evaluate(node[1], scope)
    if tag == "not":
        return 0.0 if _evaluate(node[1], scope) != 0.0 else 1.0
    if tag == "call":
        name, arg_nodes = node[1], node[2]
        spec = _FUNCTIONS.get(name)
        if spec is None:
            raise _ExprError(f"unknown function {name!r}.")
        lo, hi, fn = spec
        if not _arity_ok(name, lo, hi, len(arg_nodes)):
            expected = f"{lo}" if lo == hi else f"{lo}-{hi}" if hi is not None else f"at least {lo}"
            raise _ExprError(f"{name!r} takes {expected} argument(s), got {len(arg_nodes)}.")
        args = [_evaluate(a, scope) for a in arg_nodes]
        try:
            return float(fn(*args))
        except ZeroDivisionError:
            raise _ExprError(f"{name}(...) divided by zero.") from None
        except (ValueError, OverflowError):
            raise _ExprError(f"{name}(...) is not finite for these arguments.") from None
    if tag == "binop":
        op = node[1]
        left = _evaluate(node[2], scope)
        # `and`/`or` short-circuit -- the right operand of a false `and` or a
        # true `or` is never evaluated, so `$defined and $maybe_undefined`
        # (guarding one variable's existence with another) does not refuse.
        if op == "and":
            return 0.0 if left == 0.0 else (0.0 if _evaluate(node[3], scope) == 0.0 else 1.0)
        if op == "or":
            return 1.0 if left != 0.0 else (0.0 if _evaluate(node[3], scope) == 0.0 else 1.0)
        right = _evaluate(node[3], scope)
        try:
            return _apply_binop(op, left, right)
        except ValueError as exc:
            raise _ExprError(str(exc)) from None
    raise _ExprError(f"unknown node {tag!r}.")  # pragma: no cover - unreachable


def _eval_expr(text: str, scope: dict[str, float]) -> float:
    if len(text) > EXPR_MAX_CHARS:
        raise _ExprError(
            f"expression is {len(text)} characters, over EXPR_MAX_CHARS ({EXPR_MAX_CHARS})."
        )
    tokens = _tokenize(text)
    node = _Parser(tokens).parse()
    try:
        value = _evaluate(node, scope)
    except OverflowError:
        raise _ExprError("expression evaluates to a non-finite number.") from None
    if not math.isfinite(value):
        raise _ExprError("expression evaluates to a non-finite number.")
    return value


# --- assert conditions: an id-bearing dialect of the same expression language ---
#
# An ``assert`` step's condition is the one place this grammar names an
# object rather than a number, because it is the one place that needs to:
# every other reference in this program (a ``uid`` field, a ``uids`` field)
# is a plain string or a small ``{"id"|"name"|"$ref"|"uid": ...}`` wrapper
# *outside* the expression language, resolved by :func:`_resolve_singular`/
# :func:`_resolve_plural` long before an expression is ever involved. A
# condition has no such second channel -- ``"touches(a, b) and size(a, 1) >
# 1"`` names two objects and a number in one string -- so the expression
# grammar itself grows exactly one construct, a bare identifier that is not
# a function call, parsed by :func:`_parse_condition` (``_Parser(...,
# allow_ids=True)``) to an ``("id", name)`` leaf. :func:`_validate_condition`
# then refuses, by name and at compile time, any use of that leaf a fact
# does not accept in that exact argument position -- so by the time
# :func:`evaluate_condition` runs, at least one of every id/group it touches
# is known to have named something this program actually built.


class ConditionError(Exception):
    """Raised by :func:`evaluate_condition` (or a :data:`FACTS`
    implementation it calls) for anything that can only go wrong once the
    live document is in hand -- an id :func:`_validate_condition` approved at
    compile time but that no longer resolves, a division by zero or a
    non-finite result inside a numeric sub-expression, an axis argument that
    is not 0, 1 or 2. Distinct from :class:`ProgramError` and ``_ExprError``,
    both of which are compile-time-only: this module never raises this one
    itself before a document exists to evaluate against."""


def _axis_index(axis: float) -> int:
    """0, 1 or 2 -- never ``"x"``/``"y"``/``"z"``, because a string literal
    would collide with the same bare-identifier grammar an id argument uses
    (``lo(box, x)`` could not be told from ``lo(box, <the object named x>)``
    without a second, incompatible reading of the same token), and the
    numeric spelling is what every other rotation/vector field in this
    program already uses for an axis (``mirror``'s own ``axis`` is the one
    exception, spelled out because it is a fixed enum of exactly three
    values chosen up front, not a computed expression)."""
    i = int(axis)
    if float(i) != axis or i not in (0, 1, 2):
        raise ConditionError(f"axis must be 0, 1 or 2; got {axis!r}.")
    return i


def _fact_lo(access: Any, uid: Any, axis: float) -> float:
    lo, _hi = access.bounds(uid)
    return float(lo[_axis_index(axis)])


def _fact_hi(access: Any, uid: Any, axis: float) -> float:
    _lo, hi = access.bounds(uid)
    return float(hi[_axis_index(axis)])


def _fact_size(access: Any, uid: Any, axis: float) -> float:
    lo, hi = access.bounds(uid)
    i = _axis_index(axis)
    return float(hi[i] - lo[i])


def _fact_center(access: Any, uid: Any, axis: float) -> float:
    lo, hi = access.bounds(uid)
    i = _axis_index(axis)
    return float((lo[i] + hi[i]) * 0.5)


def _fact_count(access: Any, uids: Any) -> float:
    del access
    return float(len(uids))


def _fact_exists(access: Any, name: Any) -> float:
    return 1.0 if access.exists(name) else 0.0


def _fact_touches(access: Any, uid_a: Any, uid_b: Any) -> float:
    return 1.0 if access.touches(uid_a, uid_b) else 0.0


def _fact_grounded(access: Any, uid: Any) -> float:
    return 1.0 if access.grounded(uid) else 0.0


def _fact_floating(access: Any, uid: Any) -> float:
    return 1.0 if access.floating(uid) else 0.0


def _fact_volume(access: Any, uid: Any) -> float:
    return float(access.volume(uid))


FACTS: dict[str, tuple[tuple[str, ...], Any]] = {
    "lo": (("id", "num"), _fact_lo),
    "hi": (("id", "num"), _fact_hi),
    "size": (("id", "num"), _fact_size),
    "center": (("id", "num"), _fact_center),
    "count": (("group",), _fact_count),
    "exists": (("any",), _fact_exists),
    "touches": (("id", "id"), _fact_touches),
    "grounded": (("id",), _fact_grounded),
    "floating": (("id",), _fact_floating),
    "volume": (("id",), _fact_volume),
}
"""Every function name an ``assert`` condition may call beyond
:data:`_FUNCTIONS`'s plain math, name -> (argument kinds, implementation).
An argument kind is ``"id"`` (a bare identifier naming one object this
program placed -- resolved through *access*, never a group), ``"group"`` (a
bare identifier naming a ``group`` step's own id), ``"any"`` (a bare
identifier taken as a plain string and never checked against anything --
``exists`` is the one fact whose entire point is that its argument might
name nothing) or ``"num"`` (an ordinary numeric sub-expression, evaluated
exactly like any other -- ``lo``/``hi``/``size``/``center``'s own axis
argument is the only user of this kind today).

Each implementation is ``(access, *values) -> float`` -- 0.0/1.0 for the
boolean-shaped facts, matching every comparison and ``and``/``or``/``not``
this language already returns as a float rather than a real bool. *access*
is whatever :func:`evaluate_condition`'s own caller passed it -- this module
never constructs one and never imports a document type to describe its
shape; see that function's own docstring for the small surface it must
provide. ``lo``/``hi``/``size``/``center`` read ``access.bounds(uid)`` --
world-space, and conservative under rotation exactly like ``clay_scene``'s
own bounds block, not the exact-vertex box :mod:`.clay.analyze` computes for
its own object rows -- one object's box is cheap enough to recompute per
fact call, and matching the number an agent already read off ``clay_scene``
matters more here than shaving a rotated box down to its true extent.
``touches``/``grounded``/``floating``/``volume`` are thin wrappers over
:mod:`.clay.analyze` instead, because those facts -- contact, ground,
closed-mesh volume -- are exactly what that module already measures and
this registry has no reason to recompute."""


def _parse_condition(text: str) -> Any:
    """Parse one ``assert`` condition to the same tuple-node shape
    :func:`_eval_expr` builds for a numeric field, except a bare identifier
    that is not a function call parses to an ``("id", name)`` leaf instead of
    being refused -- see this section's own header comment. Grammar only:
    :func:`_validate_condition` is what refuses an id used somewhere a fact
    does not accept one, or naming nothing this program knows."""
    if len(text) > EXPR_MAX_CHARS:
        raise _ExprError(
            f"expression is {len(text)} characters, over EXPR_MAX_CHARS ({EXPR_MAX_CHARS})."
        )
    tokens = _tokenize(text)
    return _Parser(tokens, allow_ids=True).parse()


def _validate_condition(
    node: Any,
    objects: dict[str, _Obj],
    groups: dict[str, tuple[str, ...]],
    scope: dict[str, float],
    path: str,
) -> None:
    """Walk one parsed ``assert`` condition and refuse, by name and with
    *path*, anything :func:`evaluate_condition` could not possibly answer at
    run time: an unknown ``$var``, an unknown fact or math function, a wrong
    argument count, a bare id used somewhere other than a fact's own
    id/group argument slot, or an id/group naming nothing this program has
    created (or already consumed) -- reusing :func:`_resolve_singular` for
    that last check, the identical function (and identical wording) every
    other reference field in this program already refuses through. Never
    evaluates anything; :func:`evaluate_condition` is the run-time twin that
    shares this exact tree once this has approved it.
    """
    tag = node[0]
    if tag in ("num", "const"):
        return
    if tag == "var":
        name = node[1]
        if name not in scope:
            raise _err(f"unknown variable ${name}.", field="steps", path=path)
        return
    if tag == "id":
        name = node[1]
        raise _err(
            f"{name!r} is a bare id -- it may only appear as a fact's own "
            f"argument, e.g. touches({name}, other) or lo({name}, 0), not as "
            "a value on its own.",
            field="steps", path=path,
        )
    if tag in ("neg", "pos", "not"):
        _validate_condition(node[1], objects, groups, scope, path)
        return
    if tag == "binop":
        _validate_condition(node[2], objects, groups, scope, path)
        _validate_condition(node[3], objects, groups, scope, path)
        return
    if tag == "call":
        name, arg_nodes = node[1], node[2]
        if name in FACTS:
            arg_kinds, _impl = FACTS[name]
            if len(arg_nodes) != len(arg_kinds):
                raise _err(
                    f"{name!r} takes {len(arg_kinds)} argument(s), got {len(arg_nodes)}.",
                    field="steps", path=path,
                )
            for arg_node, arg_kind in zip(arg_nodes, arg_kinds, strict=True):
                if arg_kind == "num":
                    _validate_condition(arg_node, objects, groups, scope, path)
                    continue
                if arg_node[0] != "id":
                    raise _err(
                        f"{name!r} takes a bare id here, not an expression.",
                        field="steps", path=path,
                    )
                if arg_kind == "id":
                    _resolve_singular(arg_node[1], objects, groups, path, "steps")
                elif arg_kind == "group" and arg_node[1] not in groups:
                    raise _err(f"unknown group {arg_node[1]!r}.", field="steps", path=path)
                # "any" (exists's own argument): deliberately unchecked --
                # the fact this program is asking exists() is precisely
                # whether that name resolves to anything at all.
            return
        if name in _FUNCTIONS:
            lo, hi, _fn = _FUNCTIONS[name]
            if not _arity_ok(name, lo, hi, len(arg_nodes)):
                expected = (
                    f"{lo}" if lo == hi else f"{lo}-{hi}" if hi is not None else f"at least {lo}"
                )
                raise _err(
                    f"{name!r} takes {expected} argument(s), got {len(arg_nodes)}.",
                    field="steps", path=path,
                )
            for arg_node in arg_nodes:
                _validate_condition(arg_node, objects, groups, scope, path)
            return
        raise _err(f"unknown fact or function {name!r}.", field="steps", path=path)
    raise AssertionError(tag)  # pragma: no cover - the parser emits no other node tag


def evaluate_condition(ast: Any, scope: dict[str, float], access: Any) -> float:
    """Evaluate one compiled ``assert`` condition's AST -- the exact tuple
    tree :func:`_parse_condition` built and :func:`_validate_condition` has
    already approved -- against *access*, a small duck-typed adapter the
    caller (``agent_clay``) builds around its own live document. This module
    imports no document type and never will (see the module docstring), so
    *access* is the one seam through which a live fact reaches an actual
    object: it must provide

    - ``resolve(name) -> uid``: the live uid of the program-placed object
      called *name*, or raise :class:`ConditionError`;
    - ``resolve_group(name) -> Sequence[uid]``: the live uids of a
      ``group`` step's own members;
    - ``exists(name) -> bool``;
    - ``bounds(uid) -> (lo, hi)``, two length-3 sequences;
    - ``touches(uid, uid) -> bool``, ``grounded(uid) -> bool``,
      ``floating(uid) -> bool``, ``volume(uid) -> float``.

    *scope* is the ``{name: float}`` snapshot the program's own ``$var``
    bindings held when this ``assert`` step compiled -- a condition never
    sees a variable the program itself did not already resolve, so this is
    a plain lookup, never a fresh bind the way ``repeat``/``let`` build one
    at compile time.

    Raises :class:`ConditionError` for anything that can only go wrong once
    the live document is in hand; never raises anything else itself, and
    trusts *access*'s own geometry calls (``analyze.analyze`` included) to
    have already converted their own exceptions the same way before this
    function ever sees them.
    """
    tag = ast[0]
    if tag == "num":
        return ast[1]
    if tag == "var":
        return scope[ast[1]]
    if tag == "const":
        return math.pi
    if tag == "neg":
        return -evaluate_condition(ast[1], scope, access)
    if tag == "pos":
        return +evaluate_condition(ast[1], scope, access)
    if tag == "not":
        return 0.0 if evaluate_condition(ast[1], scope, access) != 0.0 else 1.0
    if tag == "binop":
        op = ast[1]
        left = evaluate_condition(ast[2], scope, access)
        if op == "and":
            return 0.0 if left == 0.0 else (
                0.0 if evaluate_condition(ast[3], scope, access) == 0.0 else 1.0
            )
        if op == "or":
            return 1.0 if left != 0.0 else (
                0.0 if evaluate_condition(ast[3], scope, access) == 0.0 else 1.0
            )
        right = evaluate_condition(ast[3], scope, access)
        try:
            return _apply_binop(op, left, right)
        except ValueError as exc:
            raise ConditionError(str(exc)) from None
    if tag == "call":
        name, arg_nodes = ast[1], ast[2]
        if name in _FUNCTIONS:
            _lo, _hi, fn = _FUNCTIONS[name]
            args = [evaluate_condition(a, scope, access) for a in arg_nodes]
            try:
                return float(fn(*args))
            except ZeroDivisionError:
                raise ConditionError(f"{name}(...) divided by zero.") from None
            except (ValueError, OverflowError):
                raise ConditionError(f"{name}(...) is not finite for these arguments.") from None
        arg_kinds, impl = FACTS[name]
        values: list[Any] = []
        for arg_node, arg_kind in zip(arg_nodes, arg_kinds, strict=True):
            if arg_kind == "id":
                values.append(access.resolve(arg_node[1]))
            elif arg_kind == "group":
                values.append(access.resolve_group(arg_node[1]))
            elif arg_kind == "any":
                values.append(arg_node[1])
            else:  # "num"
                values.append(evaluate_condition(arg_node, scope, access))
        return float(impl(access, *values))
    # pragma: no cover - _validate_condition already refused anything else
    raise AssertionError(tag)


# --- numeric-grammar and id-template fields ---------------------------------


def _num(value: Any, scope: dict[str, float], path: str, field_name: str) -> float:
    """One numeric grammar field: a plain number, or an expression string.
    Booleans are refused explicitly -- JSON's ``true``/``false`` silently
    becoming 1.0/0.0 here would be a caller's typo wearing a valid shape."""
    if isinstance(value, bool):
        raise _err(
            "expected a number or expression string, got a boolean.", field=field_name, path=path
        )
    if isinstance(value, (int, float)):
        v = float(value)
        if not math.isfinite(v):
            raise _err("must be finite.", field=field_name, path=path)
        return v
    if isinstance(value, str):
        try:
            return _eval_expr(value, scope)
        except _ExprError as exc:
            raise _err(str(exc), field=field_name, path=path) from None
    raise _err("expected a number or expression string.", field=field_name, path=path)


def _vec3(
    value: Any, scope: dict[str, float], path: str, field_name: str, key: str
) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise _err(
            f"{key} must be an array of exactly 3 numbers or expressions.",
            field=field_name, path=path,
        )
    return [_num(v, scope, f"{path}.{key}[{i}]", field_name) for i, v in enumerate(value)]


_ID_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _fmt_id_num(v: float) -> str:
    return str(int(v)) if v == int(v) else format(v, "g")


def _render_id(value: Any, scope: dict[str, float], path: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _err("id must be a non-empty string.", field=field_name, path=path)
    if "$" in value:
        raise _err("id templates use {name}, not $name -- a numeric grammar field takes $name.",
                    field=field_name, path=path)

    def _sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in scope:
            raise _err(f"unknown variable {{{name}}} in id template.", field=field_name, path=path)
        return _fmt_id_num(scope[name])

    return _ID_PLACEHOLDER_RE.sub(_sub, value)


def _eval_params_tree(value: Any, scope: dict[str, float], path: str, field_name: str) -> Any:
    """``params``' own value shape (a scalar, a flat vector, or a vector of
    vectors -- ``_validate_number_or_vec``'s own shape) is a generator's own
    business at the real tool door, not this compiler's: every leaf here is
    evaluated as a numeric grammar field and the nesting is passed through
    unexamined."""
    if isinstance(value, list):
        return [
            _eval_params_tree(v, scope, f"{path}[{i}]", field_name) for i, v in enumerate(value)
        ]
    return _num(value, scope, path, field_name)


# --- reference resolution ----------------------------------------------------


@dataclass
class _Obj:
    alive: bool = True
    consumed_by: str | None = None


class _Scope(dict):
    """A plain dict of name -> float; a subclass only so ``repeat``/``let``
    can hand out a child without the caller needing to know it is one --
    ``dict(parent)`` already copies everything a lookup needs."""


def _resolve_singular(
    value: Any,
    objects: dict[str, _Obj],
    groups: dict[str, tuple[str, ...]],
    path: str,
    field_name: str,
) -> Any:
    """One reference field (``uid``, ``transform``'s target, ...) -> an
    integer uid (passed straight through) or ``{"$ref": name}``."""
    if isinstance(value, bool):
        raise _err("expected a uid or an id, not a boolean.", field=field_name, path=path)
    if isinstance(value, int):
        return value
    name: str | None = None
    if isinstance(value, str):
        name = value
    elif isinstance(value, dict):
        keys = set(value)
        if keys == {"uid"}:
            v = value["uid"]
            if isinstance(v, bool) or not isinstance(v, int):
                raise _err("uid must be an integer.", field=field_name, path=path)
            return v
        if keys == {"id"}:
            name = value["id"]
        elif keys == {"name"}:
            name = value["name"]
        elif keys == {"$ref"}:
            name = value["$ref"]
        elif keys == {"group"}:
            raise _err(
                f"{value['group']!r} is a group; this field takes one id, not a group.",
                field=field_name, path=path,
            )
        else:
            raise _err(
                f"unrecognised reference shape: {sorted(keys)}.", field=field_name, path=path
            )
        if not isinstance(name, str) or not name:
            raise _err(
                "a reference name must be a non-empty string.", field=field_name, path=path
            )
    else:
        raise _err(
            "expected a uid, an id, {\"$ref\": name}, or {\"uid\": n}.", field=field_name, path=path
        )
    if name in groups:
        raise _err(
            f"{name!r} is a group; this field takes one id, not a group.",
            field=field_name, path=path,
        )
    obj = objects.get(name)
    if obj is None:
        raise _err(f"unknown id {name!r}.", field=field_name, path=path)
    if not obj.alive:
        raise _err(f"{name!r} was consumed by a {obj.consumed_by} step and no longer exists.",
                    field=field_name, path=path)
    return {"$ref": name}


def _resolve_plural(
    value: Any,
    objects: dict[str, _Obj],
    groups: dict[str, tuple[str, ...]],
    path: str,
    field_name: str,
) -> list[Any]:
    """A ``uids``-shaped field: a list of individual references, or a bare
    string/dict naming a single id or a whole group -- a group expands to
    every one of its (already-validated-alive) members."""
    if isinstance(value, dict) and set(value) == {"uids"}:
        value = value["uids"]
    if isinstance(value, dict) and set(value) == {"group"}:
        value = value["group"]
    if isinstance(value, str) and value in groups:
        return [{"$ref": member} for member in groups[value]]
    if isinstance(value, dict) and set(value) in ({"id"}, {"name"}, {"$ref"}):
        return [_resolve_singular(value, objects, groups, path, field_name)]
    if isinstance(value, (str, int)):
        return [_resolve_singular(value, objects, groups, path, field_name)]
    if not isinstance(value, list) or not value:
        raise _err(
            "expected a non-empty list of ids/uids, or a group.", field=field_name, path=path
        )
    out: list[Any] = []
    for i, item in enumerate(value):
        if isinstance(item, str) and item in groups:
            out.extend({"$ref": member} for member in groups[item])
            continue
        out.append(_resolve_singular(item, objects, groups, f"{path}[{i}]", field_name))
    return out


def _register_id(
    new_id: Any, objects: dict[str, _Obj], groups: dict[str, tuple[str, ...]],
    live_names: frozenset[str], path: str, field_name: str,
) -> str:
    if not isinstance(new_id, str) or not new_id:
        raise _err("id must be a non-empty string.", field=field_name, path=path)
    if new_id in objects:
        raise _err(f"id {new_id!r} is already used in this program.", field=field_name, path=path)
    if new_id in groups:
        raise _err(
            f"id {new_id!r} collides with a group of the same name.", field=field_name, path=path
        )
    if new_id in live_names:
        raise _err(
            f"id {new_id!r} collides with an object already in the live document.",
            field=field_name, path=path,
        )
    objects[new_id] = _Obj()
    return new_id


# --- compiled output ---------------------------------------------------------


@dataclass(frozen=True)
class Compiled:
    calls: tuple[tuple[Any, ...], ...]
    """Each entry is ``(tool_name, arguments, path)`` for a real tool call,
    or ``(\"live\", kind, arguments, path)`` for one of :data:`LIVE_KINDS`."""
    objects: dict[str, str]
    """Program id -> the wire ``name`` it was placed under (always the id
    itself -- there is no second name to keep of its own)."""
    groups: dict[str, tuple[str, ...]]
    """Group name -> the flattened, ordered ids it names."""
    expanded: int
    """The call-budget total :data:`PROGRAM_MAX_CALLS` was checked against."""


@dataclass
class _Compiler:
    live_names: frozenset[str]
    objects: dict[str, _Obj] = field(default_factory=dict)
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    calls: list[tuple[Any, ...]] = field(default_factory=list)
    expanded: int = 0
    boolean_count: int = 0

    def _budget(self, units: int, path: str) -> None:
        self.expanded += units
        if self.expanded > PROGRAM_MAX_CALLS:
            raise _err(
                f"program expands to over PROGRAM_MAX_CALLS ({PROGRAM_MAX_CALLS}) tool calls.",
                field="steps", path=path,
            )

    def _emit(self, tool: str, args: dict[str, Any], path: str) -> None:
        self.calls.append((tool, args, path))
        self._budget(1, path)

    def _emit_live(self, kind: str, args: dict[str, Any], path: str) -> None:
        self.calls.append(("live", kind, args, path))
        self._budget(1, path)

    # -- scope bookkeeping --

    @staticmethod
    def _bind(
        scope: dict[str, float], name: str, value: float, path: str, field_name: str
    ) -> dict[str, float]:
        if name in scope and scope[name] == value:
            return scope
        new_scope = dict(scope)
        new_scope[name] = value
        if len(new_scope) > PROGRAM_MAX_VARIABLES:
            raise _err(
                f"program has over PROGRAM_MAX_VARIABLES "
                f"({PROGRAM_MAX_VARIABLES}) variables in scope.",
                field=field_name, path=path,
            )
        return new_scope

    # -- one step dispatch --
    #
    # `let` and `if` are deliberately not cases here -- they change what the
    # *rest of the enclosing steps list* sees (a new binding, or nesting one
    # level deeper for a taken branch), which a per-step dispatch that
    # returns nothing has no way to hand back to its caller. `_compile_top`
    # intercepts both before reaching this method, for every steps list in
    # the program (top-level and every `repeat`/`array`/`if` body alike --
    # `_expand` calls `_compile_top`, never this method's own dispatch, for
    # exactly that reason).

    def compile_step(self, step: Any, path: str, scope: dict[str, float], nesting: int) -> None:
        if not isinstance(step, dict) or len(step) != 1:
            raise _err(
                "each step must be an object with exactly one kind key.", field="steps", path=path
            )
        (kind, body), = step.items()
        if kind not in STEP_KINDS:
            raise _err(f"unknown step kind {kind!r}.", field="steps", path=path)
        if not isinstance(body, dict):
            raise _err(f"{kind} must be an object.", field="steps", path=path)
        extra = set(body) - STEP_KINDS[kind]
        if extra:
            raise _err(f"unknown keys for {kind}: {sorted(extra)}.", field="steps", path=path)
        kind_path = f"{path}.{kind}"

        if kind in _CREATOR_KINDS:
            self._compile_creator(kind, body, kind_path, scope)
        elif kind == "transform":
            self._compile_transform(body, kind_path, scope)
        elif kind == "params":
            self._compile_params(body, kind_path, scope)
        elif kind == "material":
            self._compile_material(body, kind_path, scope)
        elif kind == "delete":
            self._compile_delete(body, kind_path, scope)
        elif kind == "op":
            self._compile_op(body, kind_path, scope)
        elif kind == "select":
            self._compile_select(body, kind_path, scope)
        elif kind == "boolean":
            self._compile_boolean(body, kind_path, scope)
        elif kind == "repeat":
            self._compile_repeat(body, kind_path, scope, nesting)
        elif kind == "array":
            self._compile_array(body, kind_path, scope, nesting)
        elif kind == "mirror":
            self._compile_mirror(body, kind_path, scope)
        elif kind == "group":
            self._compile_group(body, kind_path)
        elif kind == "let":  # pragma: no cover - intercepted by _compile_top
            raise AssertionError("let is handled by the caller, which owns the scope")
        elif kind == "if":  # pragma: no cover - intercepted by _compile_top
            raise AssertionError("if is handled by the caller, which owns nesting")
        elif kind in LIVE_KINDS:
            self._compile_live(kind, body, kind_path, scope)
        else:  # pragma: no cover - STEP_KINDS and this dispatch are kept in sync by hand
            raise _err(f"step kind {kind!r} has no compiler.", field="steps", path=path)

    # -- creators: add / figure / mesh --

    def _compile_creator(self, kind: str, body: dict, path: str, scope: dict[str, float]) -> None:
        raw_id = body.get("id")
        new_id = _render_id(raw_id, scope, path, "steps") if raw_id is not None else None

        if kind == "add":
            if "generator" not in body:
                raise _err("add requires generator.", field="steps", path=path)
            generator = body["generator"]
            if generator not in bp.GENERATORS:
                raise _err(f"unknown generator {generator!r}.", field="steps", path=path)
            args: dict[str, Any] = {"generator": generator}
            if "params" in body:
                args["params"] = self._compile_params_value(body["params"], scope, f"{path}.params")
            self._compile_place(body, args, scope, path)
            if "material" in body:
                args["material"] = self._compile_material_index(body["material"], path)
            if new_id is not None:
                args["name"] = _register_id(
                    new_id, self.objects, self.groups, self.live_names, path, "steps"
                )
            self._emit("clay_add_primitive", args, path)

        elif kind == "figure":
            if "key" not in body:
                raise _err("figure requires key.", field="steps", path=path)
            key = body["key"]
            if key not in presets.ASSEMBLIES:
                raise _err(f"unknown figure key {key!r}.", field="steps", path=path)
            args = {"key": key}
            if "translation" in body:
                args["translation"] = _vec3(
                    body["translation"], scope, path, "steps", "translation"
                )
            if "yaw" in body:
                args["yaw"] = _num(body["yaw"], scope, f"{path}.yaw", "steps")
            if "scale" in body:
                args["scale"] = _num(body["scale"], scope, f"{path}.scale", "steps")
            if new_id is not None:
                args["name_prefix"] = _register_id(
                    new_id, self.objects, self.groups, self.live_names, path, "steps"
                )
            self.calls.append(("clay_add_figure", args, path))
            self._budget(max(1, len(presets.build(key))), path)

        else:  # mesh
            for required in ("positions", "faces"):
                if required not in body:
                    raise _err(f"mesh requires {required}.", field="steps", path=path)
            args = {"positions": body["positions"], "faces": body["faces"]}
            if "uv" in body:
                args["uv"] = body["uv"]
            self._compile_place(body, args, scope, path)
            if "material" in body:
                args["material"] = self._compile_material_index(body["material"], path)
            if new_id is not None:
                args["name"] = _register_id(
                    new_id, self.objects, self.groups, self.live_names, path, "steps"
                )
            self._emit("clay_add_mesh", args, path)

    def _compile_place(self, body: dict, args: dict, scope: dict[str, float], path: str) -> None:
        for key in ("translation", "rotation", "scale"):
            if key in body:
                args[key] = _vec3(body[key], scope, path, "steps", key)

    @staticmethod
    def _compile_material_index(value: Any, path: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _err(
                "material must be a non-negative integer palette index.", field="steps", path=path
            )
        return value

    def _compile_params_value(
        self, params: Any, scope: dict[str, float], path: str
    ) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise _err("params must be an object.", field="steps", path=path)
        return {k: _eval_params_tree(v, scope, f"{path}.{k}", "steps") for k, v in params.items()}

    # -- transform / params / material / delete --

    def _compile_transform(self, body: dict, path: str, scope: dict[str, float]) -> None:
        if "uid" not in body:
            raise _err("transform requires uid.", field="steps", path=path)
        args: dict[str, Any] = {"uid": self._ref1(body["uid"], path)}
        for key in ("translation", "rotation", "scale"):
            if key in body:
                args[key] = _vec3(body[key], scope, path, "steps", key)
        self._emit("clay_transform", args, path)

    def _compile_params(self, body: dict, path: str, scope: dict[str, float]) -> None:
        has_uid, has_uids = "uid" in body, "uids" in body
        if has_uid == has_uids:
            raise _err("params requires exactly one of uid or uids.", field="steps", path=path)
        if "params" not in body:
            raise _err("params requires params.", field="steps", path=path)
        args: dict[str, Any] = {
            "params": self._compile_params_value(body["params"], scope, f"{path}.params")
        }
        if has_uid:
            args["uid"] = self._ref1(body["uid"], path)
        else:
            args["uids"] = self._refN(body["uids"], path)
        self._emit("clay_set_params", args, path)

    def _compile_material(self, body: dict, path: str, scope: dict[str, float]) -> None:
        for required in ("uids", "color"):
            if required not in body:
                raise _err(f"material requires {required}.", field="steps", path=path)
        color = body["color"]
        if not isinstance(color, list) or len(color) not in (3, 4):
            raise _err(
                "color must be an array of 3 or 4 numbers or expressions.", field="steps", path=path
            )
        args: dict[str, Any] = {
            "uids": self._refN(body["uids"], path),
            "color": [_num(v, scope, f"{path}.color[{i}]", "steps") for i, v in enumerate(color)],
        }
        if "name" in body:
            args["name"] = body["name"]
        for key in ("metallic", "roughness"):
            if key in body:
                args[key] = _num(body[key], scope, f"{path}.{key}", "steps")
        self._emit("clay_material", args, path)

    def _compile_delete(self, body: dict, path: str, scope: dict[str, float]) -> None:
        del scope
        if "uids" not in body:
            raise _err("delete requires uids.", field="steps", path=path)
        refs, names = self._refN_with_names(body["uids"], path)
        for name in names:
            self.objects[name].alive = False
            self.objects[name].consumed_by = "delete"
        self._emit("clay_delete", {"uids": refs}, path)

    # -- op / select / boolean --

    def _compile_op(self, body: dict, path: str, scope: dict[str, float]) -> None:
        for required in ("name", "uids"):
            if required not in body:
                raise _err(f"op requires {required}.", field="steps", path=path)
        op_name = body["name"]
        try:
            op = clay_ops.get(op_name)
        except KeyError:
            raise _err(f"unknown op {op_name!r}.", field="steps", path=path) from None
        if "object" not in op.modes:
            raise _err(
                f"{op_name!r} is an element-mode op; clay_program only compiles object-mode ops.",
                field="steps", path=path,
            )
        refs = self._refN(body["uids"], path)
        self._emit("clay_select", {"uids": refs}, path)
        args: dict[str, Any] = {"name": op_name}
        if "params" in body:
            raw = body["params"]
            if not isinstance(raw, dict):
                raise _err("op params must be an object.", field="steps", path=path)
            args["params"] = {
                k: _num(v, scope, f"{path}.params.{k}", "steps") for k, v in raw.items()
            }
        self._emit("clay_op", args, path)

    def _compile_select(self, body: dict, path: str, scope: dict[str, float]) -> None:
        del scope
        if "uids" not in body:
            raise _err("select requires uids.", field="steps", path=path)
        self._emit("clay_select", {"uids": self._refN(body["uids"], path)}, path)

    def _compile_boolean(self, body: dict, path: str, scope: dict[str, float]) -> None:
        del scope
        for required in ("kind", "uids"):
            if required not in body:
                raise _err(f"boolean requires {required}.", field="steps", path=path)
        if body["kind"] not in ("union", "difference", "intersection"):
            raise _err("kind must be union, difference or intersection.", field="steps", path=path)
        refs, names = self._refN_with_names(body["uids"], path)
        if len(refs) < 2:
            raise _err("boolean requires at least 2 uids.", field="steps", path=path)
        self.boolean_count += 1
        if self.boolean_count > PROGRAM_MAX_BOOLEANS:
            raise _err(
                f"program has over PROGRAM_MAX_BOOLEANS ({PROGRAM_MAX_BOOLEANS}) boolean steps.",
                field="steps", path=path,
            )
        # clay_boolean keeps the input that comes first in *document* order.
        # When every input was made by this program, creation order is
        # document order, so that survivor is known here and stays
        # addressable -- otherwise a program could never move its own
        # boolean result. A live uid among the inputs could sort anywhere,
        # so then every input is dead.
        survivor = None
        if len(names) == len(refs):
            survivor = next(n for n in self.objects if n in names)
        for name in names:
            if name == survivor:
                continue
            self.objects[name].alive = False
            self.objects[name].consumed_by = "boolean"
        self._emit("clay_boolean", {"kind": body["kind"], "uids": refs}, path)

    # -- reference helpers --

    def _ref1(self, value: Any, path: str) -> Any:
        return _resolve_singular(value, self.objects, self.groups, path, "steps")

    def _refN(self, value: Any, path: str) -> list[Any]:
        return _resolve_plural(value, self.objects, self.groups, path, "steps")

    def _refN_with_names(self, value: Any, path: str) -> tuple[list[Any], list[str]]:
        """Like :meth:`_refN`, plus the plain program-id names among the
        result (never a live uid or an already-dead id, both filtered by
        the resolver already) -- for a caller that needs to mark them dead
        afterward."""
        refs = self._refN(value, path)
        names = [r["$ref"] for r in refs if isinstance(r, dict) and "$ref" in r]
        return refs, names

    # -- repeat / array / mirror / group --

    def _expand(
        self, ranges: Any, body_steps: Any, path: str, scope: dict[str, float], nesting: int,
    ) -> None:
        if not isinstance(ranges, dict) or not ranges:
            raise _err(
                "ranges must be a non-empty object of name -> range/list.", field="steps", path=path
            )
        names = list(ranges)
        value_lists: list[list[float]] = []
        for name in names:
            spec = ranges[name]
            if isinstance(spec, list):
                spec = {"list": spec}
            if not isinstance(spec, dict):
                raise _err(f"range {name!r} must be a list or a from/to/step object.",
                            field="steps", path=path)
            if set(spec) == {"list"}:
                raw_values = spec["list"]
                if not isinstance(raw_values, list) or not raw_values:
                    raise _err(
                        f"range {name!r}'s list must be non-empty.", field="steps", path=path
                    )
                values = [_num(v, scope, f"{path}.ranges.{name}", "steps") for v in raw_values]
            elif {"from", "to"} <= set(spec) <= {"from", "to", "step"}:
                lo = _num(spec["from"], scope, f"{path}.ranges.{name}.from", "steps")
                hi = _num(spec["to"], scope, f"{path}.ranges.{name}.to", "steps")
                step = (
                    _num(spec["step"], scope, f"{path}.ranges.{name}.step", "steps")
                    if "step" in spec
                    else 1.0
                )
                if step == 0.0:
                    raise _err(f"range {name!r}'s step must not be 0.", field="steps", path=path)
                values = []
                v = lo
                guard = 0
                while (step > 0 and v < hi) or (step < 0 and v > hi):
                    values.append(v)
                    v += step
                    guard += 1
                    if guard > PROGRAM_MAX_REPEAT:
                        break
                if not values:
                    raise _err(f"range {name!r} produces no values.", field="steps", path=path)
            else:
                raise _err(
                    f"range {name!r} must be "
                    '{"list": [...]} or {"from", "to", "step"}.',
                    field="steps", path=path,
                )
            value_lists.append(values)

        total = 1
        for values in value_lists:
            total *= len(values)
        if total > PROGRAM_MAX_REPEAT:
            raise _err(
                f"repeat expands to {total} iterations, "
                f"over PROGRAM_MAX_REPEAT ({PROGRAM_MAX_REPEAT}).",
                field="steps", path=path,
            )
        if nesting + 1 > PROGRAM_MAX_NESTING:
            raise _err(f"nesting exceeds PROGRAM_MAX_NESTING ({PROGRAM_MAX_NESTING}).",
                        field="steps", path=path)

        combos = _cartesian(value_lists)
        for combo in combos:
            child_scope = dict(scope)
            for name, value in zip(names, combo, strict=True):
                child_scope = self._bind(child_scope, name, value, path, "steps")
            _compile_top(self, body_steps, f"{path}.steps", child_scope, nesting + 1)

    def _compile_repeat(self, body: dict, path: str, scope: dict[str, float], nesting: int) -> None:
        for required in ("ranges", "steps"):
            if required not in body:
                raise _err(f"repeat requires {required}.", field="steps", path=path)
        self._expand(body["ranges"], body["steps"], path, scope, nesting)

    def _compile_array(self, body: dict, path: str, scope: dict[str, float], nesting: int) -> None:
        for required in ("count", "add"):
            if required not in body:
                raise _err(f"array requires {required}.", field="steps", path=path)
        count_f = _num(body["count"], scope, f"{path}.count", "steps")
        if count_f != int(count_f) or count_f < 1:
            raise _err("count must be a positive whole number.", field="steps", path=path)
        count = int(count_f)
        var = body.get("var", "i")
        if not isinstance(var, str) or not var:
            raise _err("var must be a non-empty string.", field="steps", path=path)
        add_body = body.get("add")
        if not isinstance(add_body, dict):
            raise _err("array's add must be an object.", field="steps", path=path)
        if "id" not in add_body and "id" in body:
            add_body = {**add_body, "id": body["id"]}
        synthetic = [{"add": add_body}]
        self._expand({var: {"from": 0, "to": count, "step": 1}}, synthetic, path, scope, nesting)

    def _compile_mirror(self, body: dict, path: str, scope: dict[str, float]) -> None:
        for required in ("axis", "add"):
            if required not in body:
                raise _err(f"mirror requires {required}.", field="steps", path=path)
        axis = body["axis"]
        if axis not in ("x", "y", "z"):
            raise _err("axis must be x, y or z.", field="steps", path=path)
        add_body = body.get("add")
        if not isinstance(add_body, dict) or "id" not in add_body:
            raise _err("mirror's add must be an object with its own id.", field="steps", path=path)

        # Placement only -- the copy shares the original's geometry and is
        # reflected by negating one translation axis and the *other* two
        # rotation axes (the standard rule for reflecting an Euler XYZ
        # triple across a plane normal to `axis`). It does not mirror the
        # mesh itself: a symmetric primitive (a box, a cylinder) looks
        # identical either way, but an asymmetric one (hand-built geometry,
        # a lettered plaque) comes out wearing the original's own shape at
        # the mirrored placement, not a true reflection of it -- use the
        # clay_op `mirror-copy` row (an `op` step) for that instead.
        self.compile_step({"add": add_body}, f"{path}.original", scope, 1)
        axis_i = {"x": 0, "y": 1, "z": 2}[axis]
        mirrored = dict(add_body)
        mirrored["id"] = f"{add_body['id']}_mirror"
        if "translation" in mirrored:
            t = list(mirrored["translation"])
            t[axis_i] = _negate_grammar(t[axis_i])
            mirrored["translation"] = t
        if "rotation" in mirrored:
            r = list(mirrored["rotation"])
            for i in range(3):
                if i != axis_i:
                    r[i] = _negate_grammar(r[i])
            mirrored["rotation"] = r
        self.compile_step({"add": mirrored}, f"{path}.mirrored", scope, 1)

    def _compile_group(self, body: dict, path: str) -> None:
        for required in ("id", "members"):
            if required not in body:
                raise _err(f"group requires {required}.", field="steps", path=path)
        gid = body["id"]
        if not isinstance(gid, str) or not gid:
            raise _err("group id must be a non-empty string.", field="steps", path=path)
        if gid in self.objects or gid in self.groups:
            raise _err(f"id {gid!r} is already used in this program.", field="steps", path=path)
        members = body["members"]
        if not isinstance(members, list) or not members:
            raise _err("members must be a non-empty list.", field="steps", path=path)
        flat: list[str] = []
        for m in members:
            if not isinstance(m, str):
                raise _err("each member must be an id or group name.", field="steps", path=path)
            if m in self.groups:
                flat.extend(self.groups[m])
                continue
            obj = self.objects.get(m)
            if obj is None:
                raise _err(f"unknown id {m!r}.", field="steps", path=path)
            if not obj.alive:
                raise _err(f"{m!r} was consumed by a {obj.consumed_by} step and no longer exists.",
                            field="steps", path=path)
            flat.append(m)
        seen: list[str] = []
        for name in flat:
            if name not in seen:
                seen.append(name)
        self.groups[gid] = tuple(seen)

    # -- live kinds --

    def _compile_live(self, kind: str, body: dict, path: str, scope: dict[str, float]) -> None:
        if kind == "assert":
            self._compile_assert(body, path, scope)
            return

        if "uid" not in body:
            raise _err(f"{kind} requires uid.", field="steps", path=path)
        # A group target expands to one live entry per member here, at
        # compile time -- clay_transform (what each entry eventually calls)
        # takes exactly one uid, so a group move genuinely is N calls, not
        # one call with a list. All N still land inside the same
        # ``_fold_run`` the whole program folds into, so the program itself
        # still spends one undo step, whatever `len(targets)` turns out to
        # be. ``_refN`` (plural resolution: a group name, a plain id, or an
        # explicit list of either) is what gives this its width; a single
        # id or uid comes back as a one-element list and costs nothing extra
        # below -- ``entry_path`` stays exactly *path*, unindexed, so an
        # existing single-target program's own ``stopped_at``/``calls``
        # shape does not change.
        targets = self._refN(body["uid"], path)

        if kind in ("move", "turn"):
            if "by" not in body:
                raise _err(f"{kind} requires by.", field="steps", path=path)
            by = _vec3(body["by"], scope, path, "steps", "by")
            for i, target in enumerate(targets):
                entry_path = path if len(targets) == 1 else f"{path}[{i}]"
                self._emit_live(kind, {"uid": target, "by": list(by)}, entry_path)
            return

        # scale_by
        if "factor" not in body:
            raise _err("scale_by requires factor.", field="steps", path=path)
        factor_raw = body["factor"]
        factor = (
            _vec3(factor_raw, scope, path, "steps", "factor")
            if isinstance(factor_raw, list)
            else _num(factor_raw, scope, f"{path}.factor", "steps")
        )
        for i, target in enumerate(targets):
            entry_path = path if len(targets) == 1 else f"{path}[{i}]"
            args: dict[str, Any] = {
                "uid": target,
                "factor": list(factor) if isinstance(factor, list) else factor,
            }
            self._emit_live(kind, args, entry_path)

    def _compile_assert(self, body: dict, path: str, scope: dict[str, float]) -> None:
        if "uid" in body:
            # Optional, and not read by :func:`evaluate_condition` at all --
            # a human-readable hint about which object this assertion is
            # chiefly about. Checked here anyway, the same as every other
            # reference in this program, so a stale hint refuses at compile
            # time rather than being silently ignored.
            self._ref1(body["uid"], path)
        if "condition" not in body:
            raise _err("assert requires condition.", field="steps", path=path)
        condition = body["condition"]
        if not isinstance(condition, str) or not condition:
            raise _err("condition must be a non-empty string.", field="steps", path=path)
        try:
            ast = _parse_condition(condition)
        except _ExprError as exc:
            raise _err(str(exc), field="steps", path=path) from None
        _validate_condition(ast, self.objects, self.groups, scope, path)
        # ``scope`` is copied, not aliased, for the identical reason
        # ``clay_material``'s color and every other numeric field already
        # copies its own evaluated result: a *later* step's ``let`` rebinding
        # the same name must not reach back and change what this already-
        # compiled condition would see when it finally runs.
        args = {"condition": condition, "ast": ast, "scope": dict(scope)}
        self._emit_live("assert", args, path)


def _negate_grammar(value: Any) -> Any:
    """Negate one numeric-grammar leaf without evaluating it early -- a
    plain number negates directly; an expression string is wrapped in a
    unary minus rather than evaluated now (evaluation happens once, later,
    against the step's own scope)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return -value
    if isinstance(value, str):
        return f"-({value})"
    return value


def _cartesian(lists: list[list[float]]) -> list[tuple[float, ...]]:
    combos: list[tuple[float, ...]] = [()]
    for values in lists:
        combos = [combo + (v,) for combo in combos for v in values]
    return combos


# --- top-level compile -------------------------------------------------------


def compile_program(program: dict, *, live_names: frozenset[str] = frozenset()) -> Compiled:
    """Compile *program* (``{"variables": {...}, "steps": [...], "dry_run":
    bool}``) into a :class:`Compiled` list of tool calls. Raises
    :class:`ProgramError` naming the field and the exact path of the first
    problem found; a refused program compiles nothing (every step is
    validated as it is reached, and reaching a later step is itself proof
    the ones before it were fine, so there is no separate up-front pass to
    keep in sync with this one).

    *live_names* -- object names already in the live document this program
    would run against, if any -- guards a program id from colliding with
    one; omitted, no collision is possible to check and none is assumed.
    """
    if not isinstance(program, dict):
        raise _err("program must be an object.", field="steps", path="")
    extra = set(program) - _TOP_LEVEL_KEYS
    if extra:
        raise _err(f"unknown top-level keys: {sorted(extra)}.", field="steps", path="")

    if "dry_run" in program and not isinstance(program["dry_run"], bool):
        raise _err("dry_run must be a boolean.", field="dry_run", path="dry_run")

    scope: dict[str, float] = {}
    variables = program.get("variables", {})
    if not isinstance(variables, dict):
        raise _err("variables must be an object.", field="variables", path="variables")
    for name, value in variables.items():
        v = _num(value, scope, f"variables[{name}]", "variables")
        scope = _Compiler._bind(scope, name, v, f"variables[{name}]", "variables")

    if "steps" not in program:
        raise _err("steps is required.", field="steps", path="steps")
    # The wire schema declares the top-level list ``minItems: 1`` -- checked
    # here, once, rather than inside `_compile_top`, because that function is
    # also how a `repeat`/`if` branch's own nested list compiles, and an
    # empty `if` branch (`body.get(branch_key, [])` when the taken side was
    # never given) is a legitimate empty list this same walk must accept.
    if isinstance(program["steps"], list) and not program["steps"]:
        raise _err("steps must not be empty.", field="steps", path="steps")

    compiler = _Compiler(live_names=frozenset(live_names))
    _compile_top(compiler, program["steps"], "steps", scope, nesting=1)

    return Compiled(
        calls=tuple(compiler.calls),
        objects={name: name for name in compiler.objects},
        groups=dict(compiler.groups),
        expanded=compiler.expanded,
    )


def _compile_top(
    compiler: _Compiler, steps: Any, path: str, scope: dict[str, float], nesting: int
) -> None:
    """The same walk :meth:`_Compiler.compile_step` does for one step, except ``let``
    and ``if`` are intercepted here rather than in ``compile_step``: both
    need to change what the *rest of this same list* sees (a new binding,
    or nesting one level deeper for a taken branch), which a per-step
    dispatch that returns nothing has no way to hand back to its caller.
    """
    if not isinstance(steps, list):
        raise _err("steps must be a list.", field="steps", path=path)
    if len(steps) > PROGRAM_MAX_STEPS:
        raise _err(
            f"a steps list may hold at most PROGRAM_MAX_STEPS ({PROGRAM_MAX_STEPS}) entries.",
            field="steps", path=path,
        )
    for i, step in enumerate(steps):
        step_path = f"{path}[{i}]"
        if not isinstance(step, dict) or len(step) != 1:
            raise _err(
                "each step must be an object with exactly one kind key.",
                field="steps", path=step_path,
            )
        (kind, body), = step.items()
        if kind == "let":
            scope = _compile_let(compiler, body, f"{step_path}.let", scope)
        elif kind == "if":
            scope = _compile_if(compiler, body, f"{step_path}.if", scope, nesting)
        else:
            compiler.compile_step(step, step_path, scope, nesting)


def _compile_let(
    compiler: _Compiler, body: Any, path: str, scope: dict[str, float]
) -> dict[str, float]:
    if not isinstance(body, dict) or set(body) != {"vars"}:
        raise _err("let takes exactly one key, vars.", field="steps", path=path)
    assignments = body["vars"]
    if not isinstance(assignments, dict) or not assignments:
        raise _err("let.vars must be a non-empty object.", field="steps", path=path)
    for name, value in assignments.items():
        v = _num(value, scope, f"{path}.vars.{name}", "steps")
        scope = _Compiler._bind(scope, name, v, f"{path}.vars.{name}", "steps")
    return scope


def _compile_if(
    compiler: _Compiler, body: Any, path: str, scope: dict[str, float], nesting: int,
) -> dict[str, float]:
    if not isinstance(body, dict) or "cond" not in body:
        raise _err("if requires cond.", field="steps", path=path)
    extra = set(body) - STEP_KINDS["if"]
    if extra:
        raise _err(f"unknown keys for if: {sorted(extra)}.", field="steps", path=path)
    cond = body["cond"]
    if not isinstance(cond, str) or not cond:
        raise _err("cond must be a non-empty expression string.", field="steps", path=path)
    try:
        truth = _eval_expr(cond, scope) != 0.0
    except _ExprError as exc:
        raise _err(str(exc), field="steps", path=f"{path}.cond") from None
    branch_key = "then" if truth else "else"
    branch = body.get(branch_key, [])
    if nesting + 1 > PROGRAM_MAX_NESTING:
        raise _err(
            f"nesting exceeds PROGRAM_MAX_NESTING ({PROGRAM_MAX_NESTING}).",
            field="steps", path=path,
        )
    _compile_top(compiler, branch, f"{path}.{branch_key}", scope, nesting + 1)
    return scope
