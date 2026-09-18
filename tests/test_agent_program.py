"""``studio/modes/clay/agent/program.py``: the ``clay_program`` compiler.

No document, no session, no MCP host anywhere in this file -- the module
under test never touches one, and neither does this suite. Every check here
is a call to :func:`agent_program.compile_program` and an assertion about
the :class:`~agent_program.Compiled` it returns or the
:class:`~agent_program.ProgramError` it raises.

Four things this file exists to prove, beyond "the obvious case works":

* **The expression mini-language never runs Python.** A string that looks
  like an attack (``__import__('os')``) is refused as a bad token, not
  executed and not specially detected -- there is no string-literal grammar
  for the quoted part to land in, so nothing here has to recognise the
  attack by name.
* **Every limit actually refuses**, naming the field and a path that leads
  straight to the offending step.
* **A compiled call's argument keys are real** -- checked against
  ``agent_clay.tools()``'s own schemas, so a typo in this compiler (writing
  ``uids`` where the real tool wants ``uid``, say) fails here rather than
  being discovered by a model getting a mystery refusal from the next
  change's executor.
* **The module stays pure** -- an AST pin, the same claim
  ``tests/clay/test_clay_imports.py`` makes for ``studio/clay/``.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from warlock.studio.modes.clay.agent import dispatch as agent_clay
from warlock.studio.modes.clay.agent import program as ap


def compile_ok(program: dict, **kwargs) -> ap.Compiled:
    return ap.compile_program(program, **kwargs)


def compile_err(program: dict, **kwargs) -> ap.ProgramError:
    with pytest.raises(ap.ProgramError) as exc_info:
        ap.compile_program(program, **kwargs)
    return exc_info.value


def one_add(**overrides) -> dict:
    body = {"generator": "box"}
    body.update(overrides)
    return {"steps": [{"add": body}]}


# --- expressions: precedence, functions, degrees ---------------------------


class TestExpressionArithmetic:
    def test_precedence_multiplication_before_addition(self):
        c = compile_ok(one_add(translation=["2+3*4", 0, 0]))
        assert c.calls[0][1]["translation"][0] == 14.0

    def test_precedence_parentheses_override(self):
        c = compile_ok(one_add(translation=["(2+3)*4", 0, 0]))
        assert c.calls[0][1]["translation"][0] == 20.0

    def test_power_operator(self):
        c = compile_ok(one_add(translation=["2^3", 0, 0]))
        assert c.calls[0][1]["translation"][0] == 8.0

    def test_power_binds_tighter_than_unary_minus(self):
        # -2^2 == -(2^2) == -4, not (-2)^2 == 4 -- the usual math convention.
        c = compile_ok(one_add(translation=["-2^2", 0, 0]))
        assert c.calls[0][1]["translation"][0] == -4.0

    def test_power_is_right_associative(self):
        # 2^(3^2) == 2^9 == 512, not (2^3)^2 == 64.
        c = compile_ok(one_add(translation=["2^3^2", 0, 0]))
        assert c.calls[0][1]["translation"][0] == 512.0

    def test_power_operator_refuses_a_negative_base_with_a_fractional_exponent_instead_of_crashing(self):  # noqa: E501
        # 2026-09-16 audit: `float(left**right)` for a negative `left` and a
        # non-integer `right` returns a Python `complex` (Python's own
        # `float.__pow__` behaviour), and `float(complex)` raises `TypeError`
        # -- a type the `^` branch's `except (ValueError, OverflowError)`
        # did not catch, so it escaped uncaught instead of becoming the same
        # field-named refusal `sqrt(-1)` already gets.
        err = compile_err(one_add(translation=["(0-4)^0.5", 0, 0]))
        assert "not finite" in err.reason or "non-finite" in err.reason

    def test_double_star_is_not_power(self):
        # Spelled `^` only, by this module's own documented choice; `**`
        # tokenizes as two `*` and is refused as a malformed multiplication.
        err = compile_err(one_add(translation=["2**3", 0, 0]))
        assert err.field == "steps"

    def test_unary_minus_and_plus(self):
        c = compile_ok(one_add(translation=["-5", "+5", 0]))
        assert c.calls[0][1]["translation"][:2] == [-5.0, 5.0]

    def test_modulo(self):
        c = compile_ok(one_add(translation=["7%3", 0, 0]))
        assert c.calls[0][1]["translation"][0] == 1.0

    def test_variable_substitution(self):
        prog = {
            "variables": {"h": 4},
            "steps": [{"add": {"generator": "box", "translation": ["$h/2", 0, 0]}}],
        }
        c = compile_ok(prog)
        assert c.calls[0][1]["translation"][0] == 2.0

    def test_bare_number_string_and_literal_number_agree(self):
        c = compile_ok(one_add(translation=["5", 5, 0]))
        t = c.calls[0][1]["translation"]
        assert t[0] == t[1] == 5.0


class TestExpressionDegrees:
    def test_sin_cos_tan_take_degrees(self):
        c = compile_ok(one_add(translation=["sin(90)", "cos(180)", "tan(45)"]))
        t = c.calls[0][1]["translation"]
        assert t[0] == pytest.approx(1.0)
        assert t[1] == pytest.approx(-1.0)
        assert t[2] == pytest.approx(1.0)

    def test_asin_acos_return_degrees(self):
        c = compile_ok(one_add(translation=["asin(1)", "acos(-1)", 0]))
        t = c.calls[0][1]["translation"]
        assert t[0] == pytest.approx(90.0)
        assert t[1] == pytest.approx(180.0)

    def test_atan2_returns_degrees(self):
        c = compile_ok(one_add(translation=["atan2(1,1)", 0, 0]))
        assert c.calls[0][1]["translation"][0] == pytest.approx(45.0)

    def test_pi_constant(self):
        c = compile_ok(one_add(translation=["pi", 0, 0]))
        assert c.calls[0][1]["translation"][0] == pytest.approx(math.pi)


class TestExpressionFunctions:
    def test_sqrt_abs_floor_ceil_round(self):
        c = compile_ok(one_add(translation=["sqrt(9)", "abs(-4)", "floor(1.9)"]))
        t = c.calls[0][1]["translation"]
        assert t == [3.0, 4.0, 1.0]

    def test_min_max_variadic(self):
        c = compile_ok(one_add(translation=["min(3,1,2)", "max(3,1,2)", 0]))
        t = c.calls[0][1]["translation"]
        assert t[:2] == [1.0, 3.0]

    def test_clamp(self):
        c = compile_ok(one_add(translation=["clamp(5,0,1)", "clamp(-5,0,1)", "clamp(0.5,0,1)"]))
        assert c.calls[0][1]["translation"] == [1.0, 0.0, 0.5]

    def test_lerp(self):
        c = compile_ok(one_add(translation=["lerp(0,10,0.5)", 0, 0]))
        assert c.calls[0][1]["translation"][0] == 5.0

    def test_round_one_and_two_arg(self):
        c = compile_ok(one_add(translation=["round(1.6)", "round(1.234,1)", 0]))
        assert c.calls[0][1]["translation"][:2] == [2.0, 1.2]

    def test_comparisons_and_booleans_are_one_or_zero(self):
        c = compile_ok(one_add(translation=["1<2", "1>2", "1==1"]))
        assert c.calls[0][1]["translation"] == [1.0, 0.0, 1.0]

    def test_and_or_not(self):
        c = compile_ok(one_add(translation=["1 and 0", "0 or 1", "not 0"]))
        assert c.calls[0][1]["translation"] == [0.0, 1.0, 1.0]

    def test_and_short_circuits(self):
        # $missing is never evaluated because the left side of `and` is 0.
        c = compile_ok(one_add(translation=["0 and $missing", 0, 0]))
        assert c.calls[0][1]["translation"][0] == 0.0

    def test_or_short_circuits(self):
        c = compile_ok(one_add(translation=["1 or $missing", 0, 0]))
        assert c.calls[0][1]["translation"][0] == 1.0


class TestExpressionRefusals:
    def test_import_is_refused_not_executed(self):
        err = compile_err(one_add(translation=["__import__('os')", 0, 0]))
        assert err.field == "steps"
        assert "bad token" in err.reason

    def test_unknown_variable(self):
        err = compile_err(one_add(translation=["$ghost", 0, 0]))
        assert "unknown variable $ghost" in err.reason

    def test_unknown_function(self):
        err = compile_err(one_add(translation=["frobnicate(1)", 0, 0]))
        assert "unknown function" in err.reason.lower() or "unknown" in err.reason.lower()

    def test_division_by_zero(self):
        err = compile_err(one_add(translation=["1/0", 0, 0]))
        assert "division by zero" in err.reason

    def test_modulo_by_zero(self):
        err = compile_err(one_add(translation=["1%0", 0, 0]))
        assert "division by zero" in err.reason

    def test_non_finite_result(self):
        err = compile_err(one_add(translation=["sqrt(-1)", 0, 0]))
        assert "not finite" in err.reason or "non-finite" in err.reason

    def test_wrong_arity(self):
        err = compile_err(one_add(translation=["sqrt(1,2)", 0, 0]))
        assert "argument" in err.reason

    def test_bad_token(self):
        err = compile_err(one_add(translation=["1 @ 2", 0, 0]))
        assert "bad token" in err.reason

    def test_expr_max_chars_exceeded(self):
        long_expr = "1" + "+1" * 200
        assert len(long_expr) > ap.EXPR_MAX_CHARS
        err = compile_err(one_add(translation=[long_expr, 0, 0]))
        assert "EXPR_MAX_CHARS" in err.reason

    def test_expr_max_depth_exceeded_by_nested_parens(self):
        deep = "(" * (ap.EXPR_MAX_DEPTH + 8) + "1" + ")" * (ap.EXPR_MAX_DEPTH + 8)
        err = compile_err(one_add(translation=[deep, 0, 0]))
        assert "EXPR_MAX_DEPTH" in err.reason

    def test_expr_max_depth_exceeded_by_unary_chain(self):
        deep = "-" * (ap.EXPR_MAX_DEPTH + 8) + "1"
        err = compile_err(one_add(translation=[deep, 0, 0]))
        assert "EXPR_MAX_DEPTH" in err.reason

    def test_boolean_value_refused_in_numeric_field(self):
        err = compile_err(one_add(translation=[True, 0, 0]))
        assert "boolean" in err.reason

    def test_wrong_type_refused_in_numeric_field(self):
        err = compile_err(one_add(translation=[[1, 2], 0, 0]))
        assert err.field == "steps"


# --- program-level limits ----------------------------------------------------


class TestLimits:
    def test_program_max_steps(self):
        steps = [{"add": {"generator": "box"}} for _ in range(ap.PROGRAM_MAX_STEPS + 1)]
        err = compile_err({"steps": steps})
        assert "PROGRAM_MAX_STEPS" in err.reason
        assert err.field == "steps"

    def test_program_max_steps_boundary_is_accepted(self):
        steps = [{"add": {"generator": "box"}} for _ in range(ap.PROGRAM_MAX_STEPS)]
        c = compile_ok({"steps": steps})
        assert c.expanded == ap.PROGRAM_MAX_STEPS

    def test_program_max_calls(self):
        steps = [{"figure": {"key": "humanoid", "id": f"h{i}"}} for i in range(20)]
        err = compile_err({"steps": steps})
        assert "PROGRAM_MAX_CALLS" in err.reason

    def test_program_max_repeat(self):
        prog = {
            "steps": [
                {
                    "repeat": {
                        "ranges": {"i": {"from": 0, "to": ap.PROGRAM_MAX_REPEAT + 1}},
                        "steps": [{"add": {"generator": "box"}}],
                    }
                }
            ]
        }
        err = compile_err(prog)
        assert "PROGRAM_MAX_REPEAT" in err.reason

    def test_program_max_nesting(self):
        deep = [{"add": {"generator": "box"}}]
        for _ in range(ap.PROGRAM_MAX_NESTING + 2):
            deep = [{"repeat": {"ranges": {"i": {"from": 0, "to": 1}}, "steps": deep}}]
        err = compile_err({"steps": deep})
        assert "PROGRAM_MAX_NESTING" in err.reason

    def test_program_max_booleans(self):
        steps = []
        for i in range(ap.PROGRAM_MAX_BOOLEANS + 1):
            steps.append({"add": {"generator": "box", "id": f"a{i}"}})
            steps.append({"add": {"generator": "box", "id": f"b{i}"}})
            steps.append({"boolean": {"kind": "union", "uids": [f"a{i}", f"b{i}"]}})
        err = compile_err({"steps": steps})
        assert "PROGRAM_MAX_BOOLEANS" in err.reason

    def test_program_max_variables_top_level(self):
        variables = {f"v{i}": i for i in range(ap.PROGRAM_MAX_VARIABLES + 1)}
        err = compile_err({"variables": variables, "steps": [{"add": {"generator": "box"}}]})
        assert "PROGRAM_MAX_VARIABLES" in err.reason
        assert err.field == "variables"

    def test_program_max_variables_via_let(self):
        # One `let` step binding PROGRAM_MAX_VARIABLES + 1 names at once --
        # spread across that many steps would hit PROGRAM_MAX_STEPS first.
        variables = {f"v{i}": i for i in range(ap.PROGRAM_MAX_VARIABLES + 1)}
        steps = [{"let": {"vars": variables}}, {"add": {"generator": "box"}}]
        err = compile_err({"steps": steps})
        assert "PROGRAM_MAX_VARIABLES" in err.reason

    def test_boolean_requires_at_least_two_uids(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"boolean": {"kind": "union", "uids": ["a"]}},
            ]
        }
        err = compile_err(prog)
        assert err.field == "steps"


# --- repeat / array / mirror expansion --------------------------------------


class TestRepeat:
    def test_cartesian_product_over_two_ranges(self):
        prog = {
            "steps": [
                {
                    "repeat": {
                        "ranges": {"i": {"from": 0, "to": 2}, "j": {"list": [10, 20]}},
                        "steps": [{"add": {"generator": "box", "id": "b_{i}_{j}"}}],
                    }
                }
            ]
        }
        c = compile_ok(prog)
        names = sorted(c.objects)
        assert names == ["b_0_10", "b_0_20", "b_1_10", "b_1_20"]

    def test_range_from_to_step(self):
        prog = {
            "steps": [
                {
                    "repeat": {
                        "ranges": {"i": {"from": 0, "to": 6, "step": 2}},
                        "steps": [{"add": {"generator": "box", "id": "b_{i}"}}],
                    }
                }
            ]
        }
        c = compile_ok(prog)
        assert sorted(c.objects) == ["b_0", "b_2", "b_4"]

    def test_range_list_form(self):
        prog = {
            "steps": [
                {
                    "repeat": {
                        "ranges": {"i": [5, 7, 9]},
                        "steps": [{"add": {"generator": "box", "id": "b_{i}"}}],
                    }
                }
            ]
        }
        c = compile_ok(prog)
        assert sorted(c.objects) == ["b_5", "b_7", "b_9"]

    def test_repeat_var_visible_in_translation(self):
        prog = {
            "steps": [
                {
                    "repeat": {
                        "ranges": {"i": {"from": 0, "to": 3}},
                        "steps": [
                            {
                                "add": {
                                    "generator": "box", "id": "b_{i}",
                                    "translation": ["$i*2", 0, 0],
                                }
                            }
                        ],
                    }
                }
            ]
        }
        c = compile_ok(prog)
        xs = sorted(call[1]["translation"][0] for call in c.calls)
        assert xs == [0.0, 2.0, 4.0]


class TestArray:
    def test_array_expands_count_adds_with_var(self):
        prog = {
            "steps": [
                {
                    "array": {
                        "count": 4,
                        "add": {
                            "generator": "cylinder", "id": "post_{i}",
                            "translation": ["$i*2", 0, 0],
                        },
                    }
                }
            ]
        }
        c = compile_ok(prog)
        assert len(c.calls) == 4
        assert sorted(c.objects) == ["post_0", "post_1", "post_2", "post_3"]
        xs = sorted(call[1]["translation"][0] for call in c.calls)
        assert xs == [0.0, 2.0, 4.0, 6.0]

    def test_array_custom_var_name(self):
        prog = {
            "steps": [
                {
                    "array": {
                        "count": 2, "var": "n",
                        "add": {"generator": "box", "id": "b_{n}", "translation": ["$n", 0, 0]},
                    }
                }
            ]
        }
        c = compile_ok(prog)
        assert sorted(c.objects) == ["b_0", "b_1"]

    def test_array_count_must_be_whole_number(self):
        prog = {"steps": [{"array": {"count": 2.5, "add": {"generator": "box", "id": "b_{i}"}}}]}
        err = compile_err(prog)
        assert err.field == "steps"

    def test_array_respects_program_max_repeat(self):
        prog = {
            "steps": [
                {
                    "array": {
                        "count": ap.PROGRAM_MAX_REPEAT + 1,
                        "add": {"generator": "box", "id": "b_{i}"},
                    }
                }
            ]
        }
        err = compile_err(prog)
        assert "PROGRAM_MAX_REPEAT" in err.reason


class TestMirror:
    def test_mirror_produces_two_objects(self):
        prog = {
            "steps": [
                {
                    "mirror": {
                        "axis": "x",
                        "add": {
                            "generator": "box", "id": "wheel",
                            "translation": [1, 0, 0], "rotation": [0, 10, 20],
                        },
                    }
                }
            ]
        }
        c = compile_ok(prog)
        assert sorted(c.objects) == ["wheel", "wheel_mirror"]
        original = next(call for call in c.calls if call[1]["name"] == "wheel")
        mirrored = next(call for call in c.calls if call[1]["name"] == "wheel_mirror")
        assert mirrored[1]["translation"] == [-1.0, 0.0, 0.0]
        assert mirrored[1]["rotation"] == [0.0, -10.0, -20.0]
        assert original[1]["translation"] == [1.0, 0.0, 0.0]

    def test_mirror_y_axis_negates_x_and_z_rotation(self):
        prog = {
            "steps": [
                {
                    "mirror": {
                        "axis": "y",
                        "add": {
                            "generator": "box", "id": "w",
                            "translation": [1, 2, 3], "rotation": [5, 6, 7],
                        },
                    }
                }
            ]
        }
        c = compile_ok(prog)
        mirrored = next(call for call in c.calls if call[1]["name"] == "w_mirror")
        assert mirrored[1]["translation"] == [1.0, -2.0, 3.0]
        assert mirrored[1]["rotation"] == [-5.0, 6.0, -7.0]

    def test_mirror_requires_id_on_add(self):
        prog = {"steps": [{"mirror": {"axis": "x", "add": {"generator": "box"}}}]}
        err = compile_err(prog)
        assert err.field == "steps"

    def test_mirror_bad_axis(self):
        prog = {"steps": [{"mirror": {"axis": "w", "add": {"generator": "box", "id": "a"}}}]}
        err = compile_err(prog)
        assert err.field == "steps"


# --- id templating -----------------------------------------------------------


class TestIdTemplating:
    def test_template_substitutes_integer_loop_var(self):
        prog = {
            "steps": [
                {
                    "repeat": {
                        "ranges": {"i": [3]},
                        "steps": [{"add": {"generator": "box", "id": "leg_{i}"}}],
                    }
                }
            ]
        }
        c = compile_ok(prog)
        assert list(c.objects) == ["leg_3"]

    def test_template_formats_non_integral_compactly(self):
        prog = {
            "steps": [
                {
                    "repeat": {
                        "ranges": {"i": [2.5]},
                        "steps": [{"add": {"generator": "box", "id": "leg_{i}"}}],
                    }
                }
            ]
        }
        c = compile_ok(prog)
        assert list(c.objects) == ["leg_2.5"]

    def test_id_template_unknown_variable(self):
        err = compile_err(one_add(id="leg_{missing}"))
        assert "unknown variable" in err.reason

    def test_id_cannot_use_dollar_syntax(self):
        err = compile_err(one_add(id="leg_$i"))
        assert "{name}" in err.reason or "\\$name" in err.reason

    def test_literal_id_with_no_placeholders(self):
        c = compile_ok(one_add(id="just_a_box"))
        assert list(c.objects) == ["just_a_box"]


# --- reference resolution ----------------------------------------------------


class TestReferences:
    def test_ref_by_bare_string(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"transform": {"uid": "a", "translation": [1, 0, 0]}},
            ]
        }
        c = compile_ok(prog)
        assert c.calls[1][1]["uid"] == {"$ref": "a"}

    def test_ref_by_name_dict(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"transform": {"uid": {"name": "a"}, "translation": [1, 0, 0]}},
            ]
        }
        c = compile_ok(prog)
        assert c.calls[1][1]["uid"] == {"$ref": "a"}

    def test_ref_by_dollar_ref_dict(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"transform": {"uid": {"$ref": "a"}, "translation": [1, 0, 0]}},
            ]
        }
        c = compile_ok(prog)
        assert c.calls[1][1]["uid"] == {"$ref": "a"}

    def test_literal_uid_passes_through(self):
        prog = {"steps": [{"transform": {"uid": 42, "translation": [1, 0, 0]}}]}
        c = compile_ok(prog)
        assert c.calls[0][1]["uid"] == 42

    def test_uid_dict_form_passes_through(self):
        prog = {"steps": [{"transform": {"uid": {"uid": 42}, "translation": [1, 0, 0]}}]}
        c = compile_ok(prog)
        assert c.calls[0][1]["uid"] == 42

    def test_uids_dict_form_passes_through(self):
        prog = {"steps": [{"delete": {"uids": {"uids": [1, 2, 3]}}}]}
        c = compile_ok(prog)
        assert c.calls[0][1]["uids"] == [1, 2, 3]

    def test_undefined_id_is_refused(self):
        err = compile_err({"steps": [{"transform": {"uid": "ghost", "translation": [0, 0, 0]}}]})
        assert "unknown id" in err.reason

    def test_id_consumed_by_delete_is_refused(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"delete": {"uids": ["a"]}},
                {"transform": {"uid": "a", "translation": [1, 0, 0]}},
            ]
        }
        err = compile_err(prog)
        assert "consumed by a delete" in err.reason

    def test_id_consumed_by_boolean_is_refused(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"add": {"generator": "box", "id": "b"}},
                {"boolean": {"kind": "union", "uids": ["a", "b"]}},
                {"transform": {"uid": "b", "translation": [1, 0, 0]}},
            ]
        }
        err = compile_err(prog)
        assert "consumed by a boolean" in err.reason

    def test_the_boolean_survivor_stays_addressable_when_every_input_is_the_programs(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "b"}},
                {"add": {"generator": "box", "id": "a"}},
                {"boolean": {"kind": "union", "uids": ["a", "b"]}},
                {"transform": {"uid": "b", "translation": [1, 0, 0]}},
            ]
        }
        compiled = compile_ok(prog)
        assert compiled.calls[-1][0] == "clay_transform"

    def test_id_collision_within_program(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"add": {"generator": "box", "id": "a"}},
            ]
        }
        err = compile_err(prog)
        assert "already used" in err.reason

    def test_id_colliding_with_live_document_name(self):
        err = compile_err(one_add(id="Box"), live_names=frozenset({"Box"}))
        assert "live document" in err.reason

    def test_group_reference_expands_to_members(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"add": {"generator": "box", "id": "b"}},
                {"group": {"id": "pair", "members": ["a", "b"]}},
                {"material": {"uids": "pair", "color": [1, 0, 0]}},
            ]
        }
        c = compile_ok(prog)
        assert c.calls[-1][1]["uids"] == [{"$ref": "a"}, {"$ref": "b"}]
        assert c.groups["pair"] == ("a", "b")

    def test_group_cannot_be_used_as_a_single_ref(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"group": {"id": "g", "members": ["a"]}},
                {"transform": {"uid": "g", "translation": [1, 0, 0]}},
            ]
        }
        err = compile_err(prog)
        assert "group" in err.reason

    def test_group_of_dead_id_is_refused(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"delete": {"uids": ["a"]}},
                {"group": {"id": "g", "members": ["a"]}},
            ]
        }
        err = compile_err(prog)
        assert "consumed by a delete" in err.reason

    def test_nested_groups_flatten(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"add": {"generator": "box", "id": "b"}},
                {"add": {"generator": "box", "id": "c"}},
                {"group": {"id": "ab", "members": ["a", "b"]}},
                {"group": {"id": "all", "members": ["ab", "c"]}},
            ]
        }
        c = compile_ok(prog)
        assert c.groups["all"] == ("a", "b", "c")


# --- op: object-mode only ------------------------------------------------


class TestOp:
    def test_op_compiles_to_select_then_op(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"op": {"name": "drop-to-ground", "uids": ["a"]}},
            ]
        }
        c = compile_ok(prog)
        assert c.calls[1][0] == "clay_select"
        assert c.calls[1][1]["uids"] == [{"$ref": "a"}]
        assert c.calls[2][0] == "clay_op"
        assert c.calls[2][1]["name"] == "drop-to-ground"

    def test_op_refuses_element_mode_rows(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"op": {"name": "extrude", "uids": ["a"]}},
            ]
        }
        err = compile_err(prog)
        assert "element-mode" in err.reason

    def test_op_refuses_unknown_name(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"op": {"name": "not-a-real-op", "uids": ["a"]}},
            ]
        }
        err = compile_err(prog)
        assert "unknown op" in err.reason

    def test_op_params_are_numeric_grammar(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"op": {"name": "snap-to-grid", "uids": ["a"], "params": {"step": "1+1"}}},
            ]
        }
        c = compile_ok(prog)
        assert c.calls[2][1]["params"]["step"] == 2.0

    def test_every_object_mode_op_is_accepted(self):
        object_mode_ops = [op.name for op in ap.clay_ops.OPS if "object" in op.modes]
        assert object_mode_ops  # sanity: the registry is not empty
        for name in object_mode_ops:
            prog = {
                "steps": [
                    {"add": {"generator": "box", "id": "a"}},
                    {"op": {"name": name, "uids": ["a"]}},
                ]
            }
            compile_ok(prog)  # must not raise


# --- if / let / group --------------------------------------------------------


class TestControlFlow:
    def test_if_then_branch(self):
        prog = {
            "variables": {"n": 3},
            "steps": [
                {"if": {"cond": "$n > 2", "then": [{"add": {"generator": "box", "id": "yes"}}],
                        "else": [{"add": {"generator": "box", "id": "no"}}]}},
            ],
        }
        c = compile_ok(prog)
        assert list(c.objects) == ["yes"]

    def test_if_else_branch(self):
        prog = {
            "variables": {"n": 1},
            "steps": [
                {"if": {"cond": "$n > 2", "then": [{"add": {"generator": "box", "id": "yes"}}],
                        "else": [{"add": {"generator": "box", "id": "no"}}]}},
            ],
        }
        c = compile_ok(prog)
        assert list(c.objects) == ["no"]

    def test_if_with_no_else_and_false_condition_is_a_no_op(self):
        prog = {"steps": [{"if": {"cond": "0", "then": [{"add": {"generator": "box"}}]}}]}
        c = compile_ok(prog)
        assert c.calls == ()

    def test_if_condition_is_compile_time_only_over_variables(self):
        # A `$var` inside `cond` must already be bound -- there is no live
        # document for `if` to consult.
        err = compile_err({"steps": [{"if": {"cond": "$undefined", "then": []}}]})
        assert "unknown variable" in err.reason

    def test_let_binds_a_variable_for_later_steps(self):
        prog = {
            "steps": [
                {"let": {"vars": {"h": 5}}},
                {"add": {"generator": "box", "translation": ["$h", 0, 0]}},
            ]
        }
        c = compile_ok(prog)
        assert c.calls[0][1]["translation"][0] == 5.0

    def test_let_can_reference_earlier_let_in_same_block(self):
        prog = {
            "steps": [
                {"let": {"vars": {"a": 2, "b": "$a*3"}}},
                {"add": {"generator": "box", "translation": ["$b", 0, 0]}},
            ]
        }
        c = compile_ok(prog)
        assert c.calls[0][1]["translation"][0] == 6.0

    def test_let_and_if_work_inside_repeat(self):
        prog = {
            "steps": [
                {
                    "repeat": {
                        "ranges": {"i": {"from": 0, "to": 2}},
                        "steps": [
                            {"let": {"vars": {"double": "$i*2"}}},
                            {
                                "add": {
                                    "generator": "box", "id": "b_{i}",
                                    "translation": ["$double", 0, 0],
                                }
                            },
                        ],
                    }
                }
            ]
        }
        c = compile_ok(prog)
        xs = sorted(call[1]["translation"][0] for call in c.calls)
        assert xs == [0.0, 2.0]


# --- live kinds ---------------------------------------------------------


class TestLiveKinds:
    def test_move_compiles_to_live_entry(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"move": {"uid": "a", "by": [1, 0, 0]}},
            ]
        }
        c = compile_ok(prog)
        assert c.calls[1] == (
            "live", "move", {"uid": {"$ref": "a"}, "by": [1.0, 0.0, 0.0]}, "steps[1].move"
        )

    def test_turn_compiles_to_live_entry(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"turn": {"uid": "a", "by": [0, 90, 0]}},
            ]
        }
        c = compile_ok(prog)
        assert c.calls[1][0] == "live"
        assert c.calls[1][1] == "turn"

    def test_scale_by_accepts_scalar_or_vec3(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"scale_by": {"uid": "a", "factor": 2}},
            ]
        }
        c = compile_ok(prog)
        assert c.calls[1][2]["factor"] == 2.0

        prog2 = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"scale_by": {"uid": "a", "factor": [1, 2, 3]}},
            ]
        }
        c2 = compile_ok(prog2)
        assert c2.calls[1][2]["factor"] == [1.0, 2.0, 3.0]

    def test_assert_requires_condition(self):
        prog = {"steps": [{"add": {"generator": "box", "id": "a"}}, {"assert": {"uid": "a"}}]}
        err = compile_err(prog)
        assert "condition" in err.reason

    def test_assert_compiles_to_live_entry(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"assert": {"uid": "a", "condition": "grounded(a)"}},
            ]
        }
        c = compile_ok(prog)
        assert c.calls[1][:2] == ("live", "assert")
        args = c.calls[1][2]
        assert args["condition"] == "grounded(a)"
        assert args["scope"] == {}
        assert c.calls[1][3] == "steps[1].assert"

    def test_assert_uid_is_optional_and_condition_alone_is_enough(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"assert": {"condition": "exists(a)"}},
            ]
        }
        c = compile_ok(prog)
        assert "uid" not in c.calls[1][2]

    def test_a_group_move_expands_to_one_live_entry_per_member(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"add": {"generator": "box", "id": "b"}},
                {"group": {"id": "pair", "members": ["a", "b"]}},
                {"move": {"uid": "pair", "by": [1, 0, 0]}},
            ]
        }
        c = compile_ok(prog)
        moves = [call for call in c.calls if call[0] == "live" and call[1] == "move"]
        assert len(moves) == 2
        assert {m[2]["uid"]["$ref"] for m in moves} == {"a", "b"}
        assert {m[3] for m in moves} == {"steps[3].move[0]", "steps[3].move[1]"}
        # Every expanded member entry counts against the call budget too.
        assert c.expanded == 4

    def test_a_single_target_move_keeps_the_unindexed_path(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"move": {"uid": "a", "by": [1, 0, 0]}},
            ]
        }
        c = compile_ok(prog)
        assert c.calls[1][3] == "steps[1].move"

    def test_all_live_kinds_recognised_not_unknown(self):
        for kind in ap.LIVE_KINDS:
            assert kind in ap.STEP_KINDS


class TestAssertConditions:
    """Static validation of an ``assert`` condition -- what
    :func:`agent_program._validate_condition` refuses at compile time,
    before there is ever a document to evaluate against."""

    def test_unknown_fact_is_refused_with_a_path(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"assert": {"condition": "frobnicate(a)"}},
            ]
        }
        err = compile_err(prog)
        assert err.field == "steps"
        assert err.path == "steps[1].assert"
        assert "frobnicate" in err.reason

    def test_unknown_id_is_refused_with_a_path(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"assert": {"condition": "touches(a, ghost)"}},
            ]
        }
        err = compile_err(prog)
        assert err.path == "steps[1].assert"
        assert "unknown id 'ghost'" in err.reason

    def test_a_bare_id_used_outside_a_fact_argument_is_refused(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"assert": {"condition": "a"}},
            ]
        }
        err = compile_err(prog)
        assert "bare id" in err.reason

    def test_a_group_used_where_a_fact_wants_one_id_is_refused(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"group": {"id": "g", "members": ["a"]}},
                {"assert": {"condition": "grounded(g)"}},
            ]
        }
        err = compile_err(prog)
        assert "group" in err.reason

    def test_count_wants_a_group_not_a_plain_id(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"assert": {"condition": "count(a) == 1"}},
            ]
        }
        err = compile_err(prog)
        assert "unknown group" in err.reason

    def test_exists_argument_needs_no_prior_registration(self):
        c = compile_ok({"steps": [{"assert": {"condition": "not exists(nothing_here)"}}]})
        assert c.calls[0][0] == "live"

    def test_wrong_fact_arity_is_refused(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"assert": {"condition": "touches(a)"}},
            ]
        }
        err = compile_err(prog)
        assert "takes 2 argument" in err.reason

    def test_unknown_variable_inside_a_condition_is_refused(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"assert": {"condition": "lo(a, $missing) > 0"}},
            ]
        }
        err = compile_err(prog)
        assert "unknown variable $missing" in err.reason

    def test_a_math_function_still_works_mixed_with_a_fact(self):
        c = compile_ok(
            {
                "steps": [
                    {"add": {"generator": "box", "id": "a"}},
                    {"assert": {"condition": "size(a, 1) > sqrt(1) - 0.5"}},
                ]
            }
        )
        assert c.calls[-1][0] == "live"


# --- figure part-count weighting ------------------------------------------


class TestFigure:
    def test_figure_is_one_call_weighted_by_part_count(self):
        c = compile_ok({"steps": [{"figure": {"key": "humanoid", "id": "hero"}}]})
        assert len(c.calls) == 1
        assert c.calls[0][0] == "clay_add_figure"
        assert c.expanded == len(ap.presets.build("humanoid"))

    def test_unknown_figure_key(self):
        err = compile_err({"steps": [{"figure": {"key": "not-a-key"}}]})
        assert "unknown figure key" in err.reason

    def test_figure_id_becomes_name_prefix(self):
        c = compile_ok({"steps": [{"figure": {"key": "humanoid", "id": "hero"}}]})
        assert c.calls[0][1]["name_prefix"] == "hero"


# --- mesh -----------------------------------------------------------------


class TestMesh:
    def test_mesh_step_compiles(self):
        prog = {
            "steps": [
                {
                    "mesh": {
                        "positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "faces": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
                        "id": "tet",
                    }
                }
            ]
        }
        c = compile_ok(prog)
        assert c.calls[0][0] == "clay_add_mesh"
        assert c.calls[0][1]["name"] == "tet"

    def test_mesh_requires_positions_and_faces(self):
        err = compile_err({"steps": [{"mesh": {"positions": [[0, 0, 0]]}}]})
        assert err.field == "steps"


# --- exact message paths ------------------------------------------------


class TestMessagePaths:
    def test_path_through_repeat_and_add(self):
        prog = {
            "steps": [
                {"add": {"generator": "box"}},
                {"add": {"generator": "box"}},
                {"add": {"generator": "box"}},
                {
                    "repeat": {
                        "ranges": {"i": {"from": 0, "to": 2}},
                        "steps": [
                            {"add": {"generator": "box"}},
                            {"add": {"generator": "box", "translation": ["$w", 0, 0]}},
                        ],
                    }
                },
            ]
        }
        err = compile_err(prog)
        assert err.path == "steps[3].repeat.steps[1].add.translation[0]"
        assert err.field == "steps"
        assert str(err) == "steps[3].repeat.steps[1].add.translation[0]: unknown variable $w."

    def test_path_for_a_top_level_step(self):
        err = compile_err({"steps": [{"transform": {"uid": "ghost"}}]})
        assert err.path == "steps[0].transform"

    def test_path_for_a_variables_error(self):
        prog = {"variables": {"a": "$missing"}, "steps": [{"add": {"generator": "box"}}]}
        err = compile_err(prog)
        assert err.path == "variables[a]"
        assert err.field == "variables"

    def test_path_for_dry_run_type_error(self):
        err = compile_err({"steps": [{"add": {"generator": "box"}}], "dry_run": "yes"})
        assert err.field == "dry_run"
        assert err.path == "dry_run"


# --- structural refusals -----------------------------------------------


class TestStructuralRefusals:
    def test_unknown_step_kind(self):
        err = compile_err({"steps": [{"not_a_kind": {}}]})
        assert "unknown step kind" in err.reason

    def test_step_with_two_kind_keys(self):
        err = compile_err({"steps": [{"add": {"generator": "box"}, "delete": {"uids": [1]}}]})
        assert "exactly one kind key" in err.reason

    def test_unknown_key_within_a_kind(self):
        err = compile_err({"steps": [{"add": {"generator": "box", "bogus": 1}}]})
        assert "unknown keys for add" in err.reason

    def test_unknown_top_level_key(self):
        err = compile_err({"steps": [], "bogus": 1})
        assert "unknown top-level keys" in err.reason

    def test_empty_top_level_steps_is_refused(self):
        """``clay_program``'s wire schema declares ``steps`` ``minItems: 1``;
        an empty top-level list used to compile to a program with zero calls
        rather than being refused, so the declared constraint was not
        actually enforced -- see ``tests/test_agent_schemas.py``'s own
        exercise walk, which is what a schema/enforcement mismatch like this
        is built to catch."""
        err = compile_err({"steps": []})
        assert err.field == "steps"
        assert "must not be empty" in err.reason

    def test_an_if_branch_with_no_matching_side_is_an_empty_list_not_a_refusal(self):
        """The check above is deliberately top-level only: an ``if`` whose
        taken branch was never given (``body.get(branch_key, [])``) compiles
        an empty nested steps list, and that must stay legal."""
        compiled = compile_ok(
            {"steps": [{"if": {"cond": "0", "then": [{"add": {"generator": "box"}}]}}]}
        )
        assert compiled.calls == ()

    def test_unknown_generator(self):
        err = compile_err({"steps": [{"add": {"generator": "not-a-generator"}}]})
        assert "unknown generator" in err.reason

    def test_params_exactly_one_of_uid_or_uids(self):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"params": {"uid": "a", "uids": ["a"], "params": {"radius": 1}}},
            ]
        }
        err = compile_err(prog)
        assert "exactly one of uid or uids" in err.reason

        prog2 = {
            "steps": [
                {"add": {"generator": "box", "id": "a"}},
                {"params": {"params": {"radius": 1}}},
            ]
        }
        err2 = compile_err(prog2)
        assert "exactly one of uid or uids" in err2.reason

    def test_material_requires_uids_and_color(self):
        err = compile_err({"steps": [{"material": {"uids": [1]}}]})
        assert err.field == "steps"


# --- grammar table -------------------------------------------------------


class TestGrammarTable:
    def test_step_kinds_covers_live_kinds(self):
        assert set(ap.STEP_KINDS) >= ap.LIVE_KINDS

    def test_uid_bearing_keys_are_subsets_of_step_kinds(self):
        for kind, keys in ap.UID_BEARING_KEYS.items():
            assert kind in ap.STEP_KINDS
            assert keys <= ap.STEP_KINDS[kind]

    def test_creator_kinds_have_no_uid_bearing_keys(self):
        for kind in ("add", "figure", "mesh"):
            assert kind not in ap.UID_BEARING_KEYS

    def test_every_step_kind_is_reachable_or_documented_live(self):
        # A cheap tripwire against a kind being added to STEP_KINDS with no
        # compiler branch and no LIVE_KINDS membership -- compile_step's own
        # final ``else`` would refuse it at run time, but this catches the
        # mismatch without needing a program that exercises every kind.
        wrapper_or_control = {"repeat", "array", "mirror", "group", "let", "if"}
        for kind in ap.STEP_KINDS:
            assert kind in ap.LIVE_KINDS or kind in wrapper_or_control or kind in (
                "add", "figure", "mesh", "transform", "params", "material",
                "delete", "op", "boolean", "select",
            )


# --- cross-check against the real tool schemas --------------------------


class TestAgainstRealToolSchemas:
    """Every compiled call's argument keys must be accepted by the real
    tool's own ``inputSchema`` -- the drift this whole cross-check exists to
    catch is a compiler writing ``uids`` where the wire tool wants ``uid``,
    or vice versa, which nothing about this module's own tests would
    otherwise notice."""

    @pytest.fixture(scope="class")
    @classmethod
    def schemas(cls):
        return {tool.name: tool for tool in agent_clay.tools()}

    def _assert_call_matches_schema(self, call, schemas):
        tool_name, args, path = call
        tool = schemas.get(tool_name)
        assert tool is not None, f"{path}: compiled call names unknown tool {tool_name!r}"
        allowed = set(tool.schema.get("properties", {}))
        extra = set(args) - allowed
        assert not extra, f"{path}: {tool_name} does not declare {sorted(extra)}"

    def test_every_kind_of_call_matches_its_tool_schema(self, schemas):
        prog = {
            "steps": [
                {"add": {"generator": "box", "id": "a", "translation": [0, 0, 0], "material": 0}},
                {"add": {"generator": "box", "id": "b"}},
                {"figure": {"key": "humanoid", "id": "hero"}},
                {
                    "mesh": {
                        "positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "faces": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
                        "id": "tet",
                    }
                },
                {"transform": {"uid": "a", "translation": [1, 0, 0]}},
                {"params": {"uid": "a", "params": {"size": [1, 1, 1]}}},
                {"material": {"uids": ["a", "b"], "color": [1, 0, 0], "metallic": 0.0}},
                {"select": {"uids": ["a"]}},
                {"op": {"name": "drop-to-ground", "uids": ["a"]}},
                {"boolean": {"kind": "union", "uids": ["a", "b"]}},
            ]
        }
        c = compile_ok(prog)
        for call in c.calls:
            self._assert_call_matches_schema(call, schemas)

    def test_delete_call_matches_schema(self, schemas):
        prog = {"steps": [{"add": {"generator": "box", "id": "a"}}, {"delete": {"uids": ["a"]}}]}
        c = compile_ok(prog)
        for call in c.calls:
            self._assert_call_matches_schema(call, schemas)


# --- import pin: this module stays pure -----------------------------------


class TestImportsStayPure:
    """The same claim ``tests/clay/test_clay_imports.py`` makes for
    ``studio/clay/``: this compiler runs before there is a document, so it
    must be importable with no GL context, no window and no service layer
    behind it."""

    BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}

    @staticmethod
    def _outward_module_names() -> set[str]:
        import warlock.studio.modes.clay.agent.program as mod

        path = Path(mod.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # ``mod.__package__`` is "warlock.studio" -- this module is a leaf,
        # not a package, so its own package is the one holding it. Resolved
        # the way ``importlib._bootstrap._resolve_name`` resolves any level
        # of relative import (``package.rsplit(".", level - 1)[0]``) rather
        # than the level-1-only special case this used to hand-roll: P3 of
        # the restructure (dev/RESTRUCTURE.md) moved ``clay/presets.py`` and
        # ``clay/primitives.py`` to ``warlock/kernels/mesh/``, reached from
        # here as ``from ..kernels.mesh import presets`` (level 2, climbing
        # past ``warlock.studio`` to ``warlock``), which the old level-1
        # assumption resolved to the wrong, never-real name
        # ``warlock.studio.kernels.mesh.presets``.
        package = mod.__package__ or ""
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    found.add(node.module or "")
                else:
                    base = package.rsplit(".", node.level - 1)[0]
                    prefix = f"{base}.{node.module}" if node.module else base
                    found.update(f"{prefix}.{alias.name}" for alias in node.names)
        return found

    def test_no_banned_window_or_gl_imports(self):
        names = self._outward_module_names()
        roots = {name.split(".")[0] for name in names}
        assert not (roots & self.BANNED_ROOTS)

    def test_no_service_layer_import(self):
        names = self._outward_module_names()
        assert not any("warlock.service" in name for name in names)

    def test_only_the_documented_registries_are_reached_for(self):
        names = self._outward_module_names()
        internal = {name for name in names if name.startswith("warlock")}
        assert internal == {
            "warlock.studio.modes.clay.ops",
            "warlock.kernels.mesh.presets",
            "warlock.kernels.mesh.primitives",
        }

    def test_module_imports_with_no_optional_dependency_present(self):
        import importlib

        importlib.reload(importlib.import_module("warlock.studio.modes.clay.agent.program"))
