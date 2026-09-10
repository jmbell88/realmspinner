"""The gate that did not exist: ``cli.main``'s subcommand list vs. its dispatch.

``cli.main`` builds one flat ``choices=[...]`` list for the ``command``
argument and then dispatches each with its own ``if args.command == "...":``
branch calling a same-named ``_run_<name>`` function -- and nothing anywhere
asserted the three lists (the choices, the dispatch branches, the handler
functions) actually agree with each other. Three subcommands (``doctor``,
``sweep``, ``mcp``) have been added to this file over its life with nobody
checking that a new choice landed with a branch *and* a handler *and* a
mention in the help text; a fourth added with only two of the three would
``argparse`` its way to a silent no-op rather than an error, because a
``command`` that matches no ``if`` branch in ``main`` simply falls through to
opening the app.

Parsed with ``ast`` rather than imported and executed, the same precedent
``tests/test_studio_logging.py`` already sets with
``ast.parse(inspect.getsource(cli.main))``: running ``cli.main`` for real
opens a window or spawns the app's own subcommands, neither of which belongs
in a unit test, and a hand-maintained list of "the subcommands" in this file
would be exactly the kind of second copy that could drift the same way the
real one did.
"""

from __future__ import annotations

import ast
import inspect

from warlock import cli


def _main_tree() -> ast.FunctionDef:
    return ast.parse(inspect.getsource(cli.main)).body[0]


def _command_add_argument_call() -> ast.Call:
    """The one ``parser.add_argument(...)`` call that carries ``choices=``.

    There are several ``add_argument`` calls in ``main`` (``--image``,
    ``--bands``, ``--seed``...); the one this file cares about is the single
    positional ``command`` argument, identified by being the only one with a
    ``choices`` keyword at all.
    """
    tree = _main_tree()
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and any(kw.arg == "choices" for kw in node.keywords)
    ]
    assert len(calls) == 1, "expected exactly one add_argument call with choices="
    return calls[0]


def _choices() -> list[str]:
    call = _command_add_argument_call()
    for kw in call.keywords:
        if kw.arg == "choices":
            return [elt.value for elt in kw.value.elts if isinstance(elt, ast.Constant)]
    raise AssertionError("unreachable: _command_add_argument_call already found 'choices'")


def _help_text() -> str:
    call = _command_add_argument_call()
    for kw in call.keywords:
        if kw.arg == "help" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    raise AssertionError("the command argument's add_argument call has no plain-string help=")


def _dispatch_targets() -> set[str]:
    """Every string literal ``args.command`` is compared against with ``==``."""
    tree = _main_tree()
    targets: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Attribute) and left.attr == "command"):
            continue
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            if isinstance(op, ast.Eq) and isinstance(comparator, ast.Constant):
                targets.add(comparator.value)
    return targets


def _module_level_function_names() -> set[str]:
    tree = ast.parse(inspect.getsource(cli))
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


# --- the gate itself -----------------------------------------------------


def test_the_choices_list_is_not_empty_so_this_gate_actually_checks_something() -> None:
    """A regression that emptied ``choices=[]`` would make every test below
    vacuously pass; this is what would catch that instead."""
    assert len(_choices()) >= 1


def test_every_subcommand_choice_has_its_own_dispatch_branch_in_main() -> None:
    choices = _choices()
    dispatched = _dispatch_targets()
    missing = [name for name in choices if name not in dispatched]
    assert not missing, (
        f"{missing} appear in choices=[...] but main() has no "
        f"'if args.command == \"...\"' branch for them"
    )


def test_every_subcommand_choice_has_a_matching_run_function() -> None:
    choices = _choices()
    functions = _module_level_function_names()
    missing = [name for name in choices if f"_run_{name}" not in functions]
    assert not missing, (
        f"{missing} appear in choices=[...] but cli.py defines no _run_<name> for them"
    )


def test_the_help_text_names_every_subcommand_choice() -> None:
    help_text = _help_text()
    choices = _choices()
    missing = [name for name in choices if f"'{name}'" not in help_text]
    assert not missing, f"{missing} are not mentioned (quoted) in the command argument's help"


def test_today_s_three_known_subcommands_are_exactly_what_this_gate_sees() -> None:
    """Pins today's list too, so a silent *removal* -- as well as a half-added
    subcommand -- shows up as a failure naming exactly what changed."""
    assert _choices() == ["doctor", "sweep", "mcp"]
    assert _dispatch_targets() >= {"doctor", "sweep", "mcp"}
    assert {"_run_doctor", "_run_sweep", "_run_mcp"} <= _module_level_function_names()
