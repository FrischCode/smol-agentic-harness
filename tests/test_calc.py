"""The calculator's two promises: an exact number out, and nothing else in.

Exactness is worth a battery because the obvious implementation — evaluate, coerce to a double, print —
is wrong in a way that never raises. `2**100` comes back as `1.2676506002282294e+30`, which reads like
an answer, and the run has no way to notice it was handed a rounding of its own integer.

The closed set is worth a battery because the tool is handed a string and told to compute with it. The
walk over a whitelisted tree is what makes "this cannot run code" a property of the code rather than a
hope, and a whitelist rots in both directions: a name that reaches the model's schema but not the
function table is a tool that refuses its own advertised calls, and one in the table but not the schema
is a capability nobody agreed to advertise.

The size limits are worth a battery because the same walk would otherwise be asked to *compute* ten
million digits before declining to print them. A ceiling applied after the arithmetic is not a ceiling,
so the tests below pin where the boundary sits — and what the refusal claims about it.
"""
import math

import pytest

from sah.tools import calc
from sah.agent import Agent
from sah.confine import Confine
from sah.model import Caps, Fake, turn_text, turn_tool
from sah.record import Log, Refused
from sah.contract import arguments_for, declaration, spec_for

# The boundary of the size guard, worked out rather than observed. `_bound_pow` refuses when
# `exponent * log10(base) + 2 > MAX_DIGITS` (the two digits are the slack that keeps the estimate an
# upper bound), so for base 2 the last accepted exponent is the largest e with e*0.30103 <= 4298:
# 14277, whose answer is 4298 digits and still printable. 14278 is declined although its answer would
# have fitted — that is the accepted cost of deciding before computing, and it is pinned here so the
# day the arithmetic changes, this goes red rather than the cap quietly moving.
LAST_EXPONENT = 14277
PAST_EXPONENT = 14278


def test_an_integer_power_comes_back_digit_by_digit():
    """No exponent form, no float: every digit of the integer the model asked for.

    This is the test that reddens if any intermediate is coerced to a double — the mutation that makes
    the tool simpler, faster and quietly wrong."""
    answer = calc.calc(expr="2**100")
    assert answer == "1267650600228229401496703205376", answer
    assert "e" not in answer and "." not in answer


def test_a_division_reports_the_type_it_went_through():
    """`10/2` answers `5.0`, not `5`: the `.0` is the calculation saying it passed through a double.

    Collapsing a whole float to an integer reads prettier and reports a fact about types that never
    held — the same class of lie as the exponent form above, one decimal point smaller."""
    assert calc.calc(expr="7/2") == "3.5"
    assert calc.calc(expr="10/2") == "5.0"
    assert calc.calc(expr="7//2") == "3"
    assert calc.calc(expr="2**-1") == "0.5"


def test_a_float_keeps_every_round_trip_digit():
    """`0.1+0.2` is not `0.3`, and a tool that printed `0.3` would be editing the answer.

    Rounding to something agreeable is what makes a float impossible to audit: the run cannot see that
    it accumulated, and neither can the next reader of the log."""
    assert calc.calc(expr="0.1+0.2") == repr(0.1 + 0.2) == "0.30000000000000004"


def test_round_follows_python_and_rounds_half_to_even():
    """Pinned because nothing in the name says which way a half goes, and a tool that rounded half up
    would be a different tool with the same description."""
    assert calc.calc(expr="round(2.5)") == "2"
    assert calc.calc(expr="round(3.5)") == "4"
    assert calc.calc(expr="round(3.14159, 2)") == "3.14"


@pytest.mark.parametrize("expr, why", [
    ("__import__('os')", "no function"),
    ("open('log.jsonl')", "no function"),
    ("abs.__class__", "not arithmetic"),
    ("(1).__class__", "not arithmetic"),
    ("'a' * 3", "not a number"),
    ("x + 1", "no constant"),
    ("True + 1", "not a number"),
    ("None", "not a number"),
    ("1j", "not a number"),
    ("1 < 2", "not arithmetic"),
    ("1 if 2 else 3", "not arithmetic"),
    ("(a := 1)", "not arithmetic"),
    ("(lambda: 1)()", "bare function name"),
    ("[1, 2]", "not arithmetic"),
    ("(i for i in (1,))", "not arithmetic"),
    ("min(*[1, 2])", "not arithmetic"),
    ("eval('1')", "no function"),
    ("exec('x=1')", "no function"),
    ("1;2", "not an expression I can parse"),
    ("import os", "not an expression I can parse"),
    ("(1", "not an expression I can parse"),
])
def test_anything_outside_the_whitelist_is_refused_with_a_repair(expr, why):
    """Every refusal is text that names what was wrong, because a model shown `invalid` repeats the
    call. The list is the constructs a model actually reaches for: a call, an attribute, a string, a
    comparison, a comprehension, an assignment, a statement.

    None of these may evaluate. A tool that answers with prose about `os` is fine; one that answers
    with what `os` said is not a calculator."""
    answer = calc.calc(expr=expr)
    assert answer.startswith("calc:"), answer
    assert why in answer, answer
    assert answer.count("\n") == 0            # one refusal, not a traceback


def test_every_advertised_name_exists_and_every_existing_name_is_advertised():
    """The closed set in both directions.

    A name in the schema but not in the table is a tool that refuses its own advertised calls, and the
    model learns that by wasting a turn on each one. A name in the table but not the schema is a
    capability nobody agreed to put in front of the model — invisible, so no test of the advertised
    surface would ever miss it."""
    described = spec_for(calc.calc).parameters["properties"]["expr"]["description"]
    for name in list(calc.FUNCTIONS) + list(calc.CONSTANTS):
        assert name in described, f"{name} exists but the model is never told it does"
    for word in described.replace(";", " ").replace(",", " ").split():
        bare = word.strip("().")
        if bare in math.__dict__ or bare in ("abs", "min", "max", "round"):
            assert bare in calc.FUNCTIONS or bare in calc.CONSTANTS, \
                f"{bare} is advertised but is not in the tables"


@pytest.mark.parametrize("expr, answer", [
    ("abs(-3)", "3"), ("min(3, 1, 2)", "1"), ("max(3, 1, 2)", "3"),
    ("floor(-1.5)", "-2"), ("ceil(-1.5)", "-1"),
    ("sqrt(2)", repr(math.sqrt(2))), ("hypot(3, 4)", "5.0"), ("exp(1)", repr(math.exp(1))),
    ("log(8, 2)", "3.0"), ("log2(1024)", "10.0"), ("log10(1000)", "3.0"),
    ("sin(0)", "0.0"), ("cos(0)", "1.0"), ("tan(0)", "0.0"),
    ("asin(1)", repr(math.asin(1))), ("acos(1)", "0.0"), ("atan(1)", repr(math.atan(1))),
    ("atan2(1, 1)", repr(math.atan2(1, 1))),
    ("pi", repr(math.pi)), ("tau", repr(math.tau)), ("e", repr(math.e)),
])
def test_each_advertised_name_computes_what_its_name_promises(expr, answer):
    """A typo in the table — `"log2": math.log10` — answers, is bounded, and is wrong. Only the numbers
    catch that, and the model has no way to tell."""
    assert calc.calc(expr=expr) == answer


def test_a_power_too_large_to_write_down_is_refused_before_it_is_computed():
    """The guard's whole purpose is that the number is never made. The boundary is the arithmetic in
    `POW_LAST`/`POW_PAST`, so this pins the size and the wording at once.

    With the pre-check deleted, the refused side computes in milliseconds and answers with digits — so
    this reddens cheaply rather than by hanging, which is the property a mutation test needs."""
    assert len(calc.calc(expr=f"2**{LAST_EXPONENT}")) == 4298
    refused = calc.calc(expr=f"2**{PAST_EXPONENT}")
    assert refused.startswith("calc:") and "4300" in refused and "refused" in refused, refused
    assert not refused[4:].strip().isdigit(), "the refused power was computed and printed anyway"


def test_an_exponent_too_large_to_hold_as_a_float_is_still_refused():
    """`2 ** 10**400` has no floating point answer: the exponent is an integer the estimate cannot hold.

    The guard tests the exponent in whole numbers *before* taking a log, because `10**400 * 0.30103`
    raises `OverflowError` — a tool that crashed on the size check would have crashed while refusing to
    do the arithmetic it was refusing for."""
    for expr in ("2 ** 10**9", "2 ** 10**400", "10 ** 10**10"):
        refused = calc.calc(expr=expr)
        assert refused.startswith("calc:") and "digits" in refused, expr


def test_a_multiplication_chain_that_outgrows_the_cap_is_refused_afterwards():
    """The second check, for the case no exponent check sees: two legal numbers whose product is not.

    Each `10**3000` is under the cap, so nothing predicts the 6001-digit answer. Asking the interpreter
    to write it is the exact test of "can this be an answer" — and the refusal says *more than the
    limit*, which is the only thing the tool knows at that moment."""
    assert len(calc.calc(expr="10**2000 * 10**2000")) == 4001         # fits: 4001 digits
    refused = calc.calc(expr="10**3000 * 10**3000")
    assert refused.startswith("calc:") and "4300" in refused, refused


def test_a_result_that_left_the_double_range_says_so():
    """`1e308*10` is `inf` and Python does not raise. Bare `inf` in a result reads to the next reader as
    a quantity, so it is labelled as the non-answer it is."""
    for expr in ("1e308*10", "1e308*1e308", "1e308*1e308-1e308*1e308", "1e308+1e308"):
        answer = calc.calc(expr=expr)
        assert "finite" in answer and answer.lstrip("-").startswith(("inf", "nan")), (expr, answer)


def test_a_power_of_zero_or_one_answers_instead_of_crashing():
    """`0 ** 5` is an ordinary thing to type, and the size estimate cannot be asked about it: the log of
    zero raises, and a tool that crashed while checking whether a sum was allowed would be a worse tool
    than one that refuses.

    The guard declines to size a base that cannot grow, which is what makes a zero safe — and the same
    early return is why `1 ** 10**9` answers `1` without anyone trying to hold a billion digits.
    """
    assert calc.calc(expr="0**5") == "0"
    assert calc.calc(expr="5**0") == "1"
    assert calc.calc(expr="0**0") == "1"
    assert calc.calc(expr="1**10**9") == "1"
    assert calc.calc(expr="(-1)**10**9") == "1"
    assert calc.calc(expr="2**-3") == "0.125"          # a shrinking power needs no size check
    assert calc.calc(expr="0**-1").startswith("calc: division by zero")


def test_a_functions_own_complaint_names_the_call():
    """`sqrt(-1)` and `round(1.5, 2.0)` fail for different reasons than the whitelist, and the repair is
    different too: one needs a different expression, the other a different argument type.

    The call is quoted back because `round(1.5, 2.0)` failing with `'float' object cannot be
    interpreted as an integer` names a type but not which of three arguments held it."""
    # The interpreter's complaint changes between CPython releases; the tool's framing -- and that it
    # names the call -- is what is under test.
    assert calc.calc(expr="sqrt(-1)").startswith("calc: sqrt(-1) has no value:")
    bad = calc.calc(expr="round(1.5, 2.0)")
    assert bad.startswith("calc: round(1.5, 2.0) has no value:"), bad


def test_a_refusal_is_returned_rather_than_raised():
    """A raised error reaches the model as `calc raised ValueError: ...`, which is a fact about the
    harness and no help at all. The reader who has to repair the call is the model, so the answer is
    prose it can act on and the run records a call that worked."""
    assert calc.calc(expr="1/0").startswith("calc: division by zero")
    assert calc.calc(expr="").startswith("calc: no expression given")
    assert calc.calc(expr="   ").startswith("calc: no expression given")


def test_the_expression_is_typed_at_the_boundary():
    """The `read(path=0)` family of bug: a value the schema never offered must never reach the tool, let
    alone the policy. A number in place of an expression is refused with the argument named.

    This is checked here rather than only once in the tools battery because `calc` is the one tool whose
    argument is *code-shaped text*: it is the call a server is most likely to send unquoted."""
    with pytest.raises(Refused, match="must be a string"):
        arguments_for(calc.calc, {"expr": 5})
    with pytest.raises(Refused, match="must be a string"):
        arguments_for(calc.calc, {"expr": ["1+1"]})


def test_an_overlong_expression_is_declined_before_it_is_parsed():
    """The limit is on the parser and the recursive walk, not on arithmetic: a runaway argument of
    nested parentheses costs the harness, and no calculation anyone wants needs 2000 characters."""
    answer = calc.calc(expr="1+" * calc.MAX_EXPR_CHARS + "1")
    assert answer.startswith("calc:") and "characters" in answer, answer
    # 801 characters: the parser's limit, not this one
    nested = calc.calc(expr="(" * 400 + "1" + ")" * 400)
    assert nested.startswith("calc:"), nested


def test_the_answer_reaches_the_log_whole(tmp_path):
    """The exact digits survive the trip through the result bound into the record.

    Driven through an `Agent` because the size a tool returns and the size a run stores are decided in
    two modules, and a 4298-digit single-line answer is exactly the shape a clip would eat."""
    agent = Agent(backend=Fake([turn_tool("c1", "calc", {"expr": f"2**{LAST_EXPONENT}"}), turn_text("done")],
                               caps=Caps(context_tokens=4000, model="t")),
                  tools={"calc": calc.calc}, log=Log(tmp_path / "log.jsonl"), max_turns=4,
                  run_id="r1", root=tmp_path)
    agent.run("go")
    result = next(s for s in agent.log.steps() if s["kind"] == "result")
    assert result["ok"] and not result.get("truncated"), result
    assert result["content"] == str(2 ** LAST_EXPONENT)


def test_the_tool_touches_no_place_and_is_offered_to_a_run_granted_nothing():
    """`none` is the declaration that makes an always-offered tool safe: there is no address, so the
    veto point has nothing to weigh and a run with no grants can still add two numbers.

    Asserted against a `Confine` with no roots at all, because "always offered" is only harmless while
    the policy genuinely has nothing to withhold."""
    assert declaration(calc.calc) == ("none", None)
    assert Confine()("calc", {"expr": "1+1"}, access="none", address=None).allowed
    assert spec_for(calc.calc).parameters["required"] == ["expr"]
