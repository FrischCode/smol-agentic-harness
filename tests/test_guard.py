"""The repetition guard: pure detection tested on its own, and the loop's reaction to it tested
end to end against a scripted backend. The guard decides; the loop writes — these assert both halves."""
from sah import model
from sah.agent import Agent
from sah.guard import RepetitionGuard, act_key
from sah.record import Log


def echo(text: str) -> str:
    """Upper-case the text."""
    return text.upper()


echo.access, echo.address = "none", None        # a double that touches no place says so
echo.description = "Upper-case the text."


def one(agent, kind):
    return next(s for s in agent.log.steps() if s["kind"] == kind)


def many(agent, kind):
    return [s for s in agent.log.steps() if s["kind"] == kind]


# --- the guard in isolation: it only counts and decides ------------------------------------------------

def test_only_consecutive_identical_acts_streak():
    g = RepetitionGuard(threshold=3, steer_limit=2)
    g.observe("echo", {"text": "x"}, True)
    g.observe("echo", {"text": "x"}, True)
    assert g.escalation() is None, "two is not yet a problem"
    g.observe("echo", {"text": "x"}, True)
    assert g.escalation() == "nudge"


def test_a_different_act_resets_the_streak():
    g = RepetitionGuard(threshold=3)
    g.observe("echo", {"text": "a"}, True)
    g.observe("echo", {"text": "a"}, True)
    g.observe("echo", {"text": "b"}, True)          # changed one argument — adapting, not stuck
    assert g.escalation() is None


def test_argument_order_is_the_same_act_but_a_different_result_is_not():
    assert act_key("f", {"a": 1, "b": 2}, True) == act_key("f", {"b": 2, "a": 1}, True)
    assert act_key("f", {"a": 1}, True) != act_key("f", {"a": 1}, False), \
        "same call that now fails is different news"


def test_two_nudges_then_stop():
    g = RepetitionGuard(threshold=3, steer_limit=2)
    for _ in range(3):
        g.observe("echo", {"text": "x"}, True)
    assert g.escalation() == "nudge"
    g.spend_nudge()
    for _ in range(3):
        g.observe("echo", {"text": "x"}, True)
    assert g.escalation() == "nudge"
    g.spend_nudge()
    for _ in range(3):
        g.observe("echo", {"text": "x"}, True)
    assert g.escalation() == "stop", "a third crossing with nudges spent is a stop"


def test_escalation_does_not_consume_state():
    g = RepetitionGuard(threshold=2)
    g.observe("echo", {"text": "x"}, True)
    g.observe("echo", {"text": "x"}, True)
    assert g.escalation() == "nudge" and g.escalation() == "nudge", "calling twice is safe"


def test_threshold_below_two_is_refused():
    try:
        RepetitionGuard(threshold=1)
        assert False, "threshold 1 would nudge a first-ever call"
    except ValueError:
        pass


# --- the loop's reaction: nudge is a core say + notice; a third crossing stalls --------------------------

def build(turns, tmp_path, tools, *, max_turns=20, **kw):
    # A scripted session an operator is reading, so prose completes; pass `attended=False` to model
    # the unattended contract instead.
    kw.setdefault("attended", True)
    agent = Agent(backend=model.Fake(turns, caps=model.Caps(context_tokens=1000, model="t")),
                  tools=tools, log=Log(tmp_path / "log.jsonl"), max_turns=max_turns,
                  run_id="r1", root=tmp_path, **kw)
    return agent


def test_a_run_that_repeats_identically_is_nudged_then_stalls(tmp_path):
    # The same tool call, same arguments, forever — the exact non-convergence the guard is for.
    turn = [model.ToolCall("c", "echo", {"text": "x"}), model.Stopped("tool_calls")]
    agent = build([turn] * 30, tmp_path, {"echo": echo})
    agent.run("go")
    nudges = [s for s in many(agent, "notice") if s["subject"] == "nudge"]
    changes = [s for s in nudges if "nudged to change" in s["text"]]
    stops = [s for s in nudges if "stopped" in s["text"]]
    assert len(changes) == 2, "steer_limit nudges before it stops"
    assert len(stops) == 1, "and the crossing after nudges are spent is a stop"
    assert one(agent, "ended")["reason"] == "stalled"
    assert one(agent, "ended")["exit_code"] == 1
    # the nudge reached the model as a message, not just the log
    steers = [m for m in agent.messages if m.role == "user" and "same call" in m.content]
    assert steers, "a nudge the model never sees is not a nudge"


def test_a_run_that_varies_its_calls_is_never_nudged(tmp_path):
    turns = [[model.ToolCall(f"c{i}", "echo", {"text": str(i)}), model.Stopped("tool_calls")]
             for i in range(5)] + [[model.TextDelta("done"), model.Stopped("stop")]]
    agent = build(turns, tmp_path, {"echo": echo})
    agent.run("go")
    assert not [s for s in agent.log.steps() if s["kind"] == "notice" and s.get("subject") == "nudge"]
    assert one(agent, "ended")["reason"] == "complete"


def test_a_disabled_guard_lets_a_loop_run_to_its_turn_limit(tmp_path):
    turn = [model.ToolCall("c", "echo", {"text": "x"}), model.Stopped("tool_calls")]
    agent = build([turn] * 30, tmp_path, {"echo": echo}, guard=None, max_turns=3)
    agent.run("go")
    assert one(agent, "ended")["reason"] == "max_turns", "with the guard off, max_turns still bounds it"


def test_a_refused_call_repeated_also_trips_the_guard(tmp_path):
    # Hammering a denial is the same non-convergence; the guard observes refused acts too.
    from sah.agent import Verdict
    deny = lambda name, args, **_: Verdict(False, "not allowed here")
    turn = [model.ToolCall("c", "echo", {"text": "x"}), model.Stopped("tool_calls")]
    agent = build([turn] * 30, tmp_path, {"echo": echo}, policy=deny)
    agent.run("go")
    assert one(agent, "ended")["reason"] == "stalled"


def test_a_reset_forgets_the_previous_objective():
    """The guard counts acts, and a new objective is a new run of acts. Nudges spent while one
    objective was stuck are not evidence that the next one is stuck too, so the core resets it
    between objectives — the same rule that keeps an `interrupt` from outliving its objective."""
    g = RepetitionGuard(threshold=2, steer_limit=1)
    g.observe("read", {"path": "a"}, True)
    g.observe("read", {"path": "a"}, True)
    assert g.escalation() == "nudge"
    g.spend_nudge()
    g.observe("read", {"path": "a"}, True)
    g.observe("read", {"path": "a"}, True)
    assert g.escalation() == "stop"
    g.reset()
    assert (g.escalation(), g.nudges, g.repeating) == (None, 0, "")
    g.observe("read", {"path": "a"}, True)
    assert g.escalation() is None, "one act into a new objective is not yet a streak"
