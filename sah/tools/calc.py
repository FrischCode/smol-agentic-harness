"""sah.tools.calc — arithmetic for a run that has no shell.

A default run executes nothing: `--exec` is a widening most runs are not given, so there is no
interpreter one call away. A model asked to scale a batch, convert a unit or compound a rate then has
two options and both are bad — say it cannot do arithmetic, or do the sum in prose, which is where a
small local model gets `0.15 * 240` wrong with total confidence. This is the third option: an
expression goes in, a number comes out, nothing else happens.

**The whitelist is the evaluator.** The expression is parsed with `ast.parse` in `eval` mode — one
expression, never a statement — and the tree is walked by hand below. A node type that is not listed
is refused, so there is no `eval`, no `exec`, no import, no attribute access, no subscript, no name
that is not one of the constants and no text. The evaluation *is* the walk: nothing is compiled, and
the only things called are the entries of `FUNCTIONS`.

**Integers stay exact.** A literal keeps the type it parses as, so `2**100` returns all 31 digits and
`7/2` returns `3.5`. Coercing every intermediate to a double — the obvious shortcut, and the one that
makes the size problem below disappear — turns `2**100` into `1.2676506002282294e+30`, which looks
like an answer and is not one.

**A number too large to be an answer is refused.** Two checks, because they defend two different costs:

- Before a power is taken, its size is *estimated* from the operands (`_bound_pow`) and the call
  refused if the estimate exceeds `MAX_DIGITS`. This is the check that matters: computing the number is
  the expensive thing, so a ceiling applied afterwards is not a ceiling. The estimate is a bound
  carrying slack, and the refusal says "about", because the tool has not counted digits it has not
  computed.
- After every step, the value is tested by the only authority on whether it can be written down: the
  interpreter's own conversion (`_printable`). This one is exact and costs nothing, because the number
  already exists by then — it catches the multiplication chain that no exponent check saw coming.

**A float that leaves the range of a double is named.** `1e308 * 10` is `inf`, and Python reports that
without raising. Bare `inf` in a result reads to the next reader as a quantity, so it comes back
labelled.
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Annotated

from sah.contract import Param

MAX_DIGITS = 4300
"""The largest integer this tool returns, in digits.

Chosen, not invented: 4300 is the ceiling the interpreter puts on writing an integer as text, so an
answer under this cap is always printable and an answer over it could not be delivered anyway. It also
bounds chained exponentiation, where every step is legal on its own and the fifth is terminal. The cap
stays ours rather than tracking `sys.get_int_max_str_digits()`: an operator who lifted the interpreter's
limit would still not want a hundred-thousand-digit number in the run's context, where the result bound
would clip it away anyway.
"""

MAX_EXPR_CHARS = 2000
"""How long an expression may be. Nothing real needs more, and the limit is what keeps a runaway
argument from becoming a parser's problem: a deeply nested expression costs the parser and the
recursive walk below, not the arithmetic."""


class _Reject(Exception):
    """Why this expression will not be evaluated, in the words the model is sent.

    Deliberately not a `ValueError`: the whitelisted functions raise that for their own reasons
    (`sqrt(-1)` is a domain error), and the two answers differ — one is "this is not arithmetic", the
    other is "this arithmetic has no value". Merging them costs the repair, which is the point of a
    refusal.
    """


FUNCTIONS = {
    "abs": abs, "min": min, "max": max, "round": round,
    "floor": math.floor, "ceil": math.ceil,
    "sqrt": math.sqrt, "hypot": math.hypot, "exp": math.exp,
    "log": math.log, "log2": math.log2, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
}
"""The calls that exist here, bound once to their real values.

No `sum`, no `range`, no `pow`: the arguments are numbers rather than a sequence to build, and `**` is
already the power. `round(x)` follows Python and rounds half to even; that behaviour is a stated
contract, not an accident to be discovered — a tool that rounded half up would be a different tool and
nothing in its name would say so.
"""

CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
"""The only names an expression may use. A name that is not one of these is a refusal, not a zero."""

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
           ast.Pow: operator.pow}

_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _whole(value) -> bool:
    """A true integer, with the bool that Python counts as one sent away.

    `True + 1` answering `2` would be a tool agreeing that text-ish values are numbers, and the same
    confusion in `_bound_pow` would size a power by a value the expression never meant."""
    return isinstance(value, int) and not isinstance(value, bool)


def _printable(value):
    """Hand back an integer only if the interpreter will write it down; refuse the one it will not.

    Asking `str()` is the exact question, and it is free at this point because the number is already
    computed. The estimate in `_bound_pow` cannot replace it: an estimate is why a power is not
    computed, and this is why a chain of multiplications never becomes a value nobody can print."""
    if not _whole(value):
        return value
    try:
        str(value)
    except ValueError:
        raise _Reject(f"the answer has more than {MAX_DIGITS} digits, which is more than can be "
                      "written down; scale the calculation down or take it in pieces") from None
    return value


def _brief(value) -> str:
    """A number short enough to quote in a refusal.

    A refusal that prints its own 400-digit exponent spends the message on the wrong number: the model
    needs to see how big the exponent was, not have it read back. `calc` answers a real number in full
    (`_say`); this is only for prose *about* a number.
    """
    text = _say(value)
    return text if len(text) <= 40 else f"a {len(text)}-digit number"


def _bound_pow(base, exponent) -> None:
    """Refuse a power too large to print *before* computing it.

    digits(base ** exponent) = floor(exponent * log10 base) + 1, and `log10` of an integer the
    interpreter already holds is exact enough to bound it with two digits of slack. Those two digits
    mean a handful of answers just under the cap are declined — the alternative is computing the number
    to learn whether computing the number was allowed, which is the cost this function exists to avoid.

    The exponent is tested in whole numbers first, because a float cannot hold `10**400` and the
    estimate would raise where it means to refuse. A power past ten times the cap cannot come back
    under it from a base of 2 or more, so that test needs no floating point at all.
    """
    if not (_whole(base) and _whole(exponent)) or exponent < 1 or abs(base) < 2:
        return                      # 0**e is 0, 1**e is 1, and a negative power shrinks into 0..1
    if exponent > MAX_DIGITS * 10:
        raise _Reject(f"{_brief(base)} raised to {_brief(exponent)} cannot come back under "
                      f"{MAX_DIGITS} digits; refused before being computed")
    estimate = exponent * math.log10(abs(base)) + 2
    if estimate > MAX_DIGITS:
        raise _Reject(f"{_brief(base)} ** {_brief(exponent)} needs about {math.ceil(estimate)} digits, "
                      f"over the {MAX_DIGITS} this tool can write down; refused before being computed")


def _value(node: ast.AST):
    """Evaluate one node. The recursion is the evaluation, and the isinstance tests are the policy.

    An int stays an int and a float stays a float — see the module docstring on why nothing is coerced.
    """
    if isinstance(node, ast.Expression):
        return _value(node.body)
    if isinstance(node, ast.Constant):
        if _whole(node.value) or isinstance(node.value, float):
            return node.value
        named = "text" if isinstance(node.value, str) else repr(node.value)
        raise _Reject(f"{named} is not a number; this evaluates arithmetic over numbers only")
    if isinstance(node, ast.BinOp):
        func = _BINOPS.get(type(node.op))
        if func is None:
            raise _Reject(f"no operator {type(node.op).__name__}; the operators are + - * / // % **")
        left, right = _value(node.left), _value(node.right)
        if func is operator.pow:
            _bound_pow(left, right)
        return _printable(func(left, right))
    if isinstance(node, ast.UnaryOp):
        func = _UNARY.get(type(node.op))
        if func is None:
            raise _Reject(f"no unary operator {type(node.op).__name__}; only + and -")
        return _printable(func(_value(node.operand)))
    if isinstance(node, ast.Call):
        return _printable(_call(node))
    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise _Reject(f"no constant {node.id!r}; the names are {', '.join(sorted(CONSTANTS))} "
                      f"(this tool has no variables)")
    raise _Reject(f"{type(node).__name__} is not arithmetic; allowed are numbers, + - * / // % **, "
                  f"parentheses, the functions {', '.join(sorted(FUNCTIONS))} and the constants "
                  f"{', '.join(sorted(CONSTANTS))}")


def _call(node: ast.Call):
    """Apply a whitelisted function to already-evaluated numeric arguments."""
    if not isinstance(node.func, ast.Name):
        raise _Reject("only a bare function name may be called — no attribute access, nothing chained")
    name = node.func.id
    func = FUNCTIONS.get(name)
    if func is None:
        raise _Reject(f"no function {name!r}; the functions are {', '.join(sorted(FUNCTIONS))}")
    if node.keywords:
        raise _Reject(f"{name} takes positional numbers only, no keyword arguments")
    if not node.args:
        raise _Reject(f"{name} needs at least one number")
    args = [_value(a) for a in node.args]
    try:
        return func(*args)
    except (ValueError, TypeError, OverflowError) as e:
        # The function's own complaint, with the call shown: `round(1.5, 2.0)` fails on a float where
        # an integer belongs, and naming the call is what lets the model fix it in one turn rather than
        # guess which of its arguments Python disliked.
        shown = ", ".join(_say(a) for a in args)
        raise _Reject(f"{name}({shown}) has no value: {e}") from None


def _say(value) -> str:
    """How a value is written back — exact for an integer, round-trip for a float.

    An integer is printed digit by digit with no exponent, because `1e+30` is a lossy answer to
    `2**100`. A float keeps the interpreter's own `repr`, so `10/2` reads `5.0`: that `.0` is the
    calculation saying it went through a double, and hiding it would report a fact about types that
    never held.
    """
    if _whole(value):
        return str(value)
    if not math.isfinite(value):
        return (f"{value!r} — not a finite number: the calculation left the range of a double, so this "
                "is not a quantity to use")
    return repr(value)


def calc(expr: Annotated[str, Param(
    "one arithmetic expression over numbers: + - * / // % ** and parentheses; the functions abs, min, "
    "max, round, floor, ceil, sqrt, hypot, exp, log, log2, log10, sin, cos, tan, asin, acos, atan, "
    "atan2; the constants pi, e, tau. No variables, no text, no statements.")]) -> str:
    """The walk over the parsed tree against a fixed list of node types is the whole evaluation (see the
    module docstring), so there is nothing here that can run code, open a place or reach the network. A
    refusal comes back as text rather than as an exception, because the reader who has to repair the
    call is the model.
    """
    text = expr.strip()
    if not text:
        return "calc: no expression given — send something like 0.15 * 240"
    if len(text) > MAX_EXPR_CHARS:
        return (f"calc: that expression is {len(text)} characters and this tool takes {MAX_EXPR_CHARS}; "
                "split the calculation across two calls")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        where = f" at character {e.offset}" if e.offset else ""
        return (f"calc: not an expression I can parse ({e.msg}{where}); send one arithmetic expression "
                "with no assignment and no semicolon")
    except (RecursionError, MemoryError):
        return "calc: too deeply nested to parse; split the expression"
    try:
        value = _value(tree)
    except _Reject as why:
        return f"calc: {why}"
    except ZeroDivisionError:
        return "calc: division by zero — the expression has no value"
    except OverflowError:
        return "calc: the result left the range of a double-precision number"
    except RecursionError:
        return "calc: too deeply nested to evaluate"
    return _say(value)


calc.description = ("Evaluate one arithmetic expression and return its number — no shell, no code "
                    "execution, no variables. Integers stay exact at any size; anything outside the "
                    "whitelist is refused with the reason and the allowed set, not with a bare "
                    "\"invalid\".")


# No place is touched: the expression is the whole input and the number is the whole output. The policy
# therefore has nothing to judge (`sah.confine` allows `none` outright), which is why this tool is
# offered without a flag — there is no reach to widen and nothing a grant could mean.
calc.access, calc.address = "none", None


class Calc:
    """Arithmetic, always offered.

    `--exec` would give a run a far better calculator, but exec is a widening most runs are not given,
    and a run without it still has to add up. The tool touches no place, so there is nothing to gate:
    the veto point has no address to look at, and the schema costs the run one tool entry.
    """

    tools = (calc,)
