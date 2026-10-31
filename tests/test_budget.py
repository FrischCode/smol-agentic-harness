"""The budgets: the turn ceiling and the time wall, and the wrap-up that goes with both.

A budget that runs out is told to the model, with grace turns to deliver what it has; a run
that answers in them ends complete, and one that spends the grace ends with the budget's
reason. Every test drives a scripted backend and, where the wall is the subject, a scripted
clock — the stand-in seam the loop reads instead of sleeping — so these assert the harness's
own behaviour and never a wall's."""
import pytest

from sah import model, prompts, record
from sah.agent import Agent, Verdict
from sah.record import Log


class Clock:
    """A clock a test moves forward instead of sleeping."""

    def __init__(self, start=0.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


class TickingBackend:
    """A scripted backend that advances a clock by a fixed amount each turn it serves, so a
    turn costs wall time the way a real one does. The advance happens when the turn starts,
    so the boundary before turn N+1 sees the time turn N spent."""

    def __init__(self, turns, caps, clock, tick):
        self.fake = model.Fake(turns, caps=caps)
        self.clock = clock
        self.tick = tick

    def capabilities(self):
        return self.fake.capabilities()

    def complete(self, **request):
        self.clock.advance(self.tick)
        return iter(self.fake.complete(**request))


def echo(text: str) -> str:
    """Upper-case the text."""
    return text.upper()


echo.access, echo.address = "none", None
echo.description = "A test double: echo."
TOOLS = {"echo": echo}


def ALLOW(name, args, *, access=None, address=None):
    return Verdict(True)


def build(turns, tmp_path, *, backend=None, max_turns=6, max_runtime=0, grace_turns=3,
          clock=None):
    """A run against a scripted backend with a scripted clock, wired the way `cli` wires it.

    `clock` is taken from the caller when one is passed, because a test that ticks the clock
    (a `TickingBackend`) must tick the same clock the loop reads."""
    if clock is None:
        clock = Clock()
    log = Log(tmp_path / "log.jsonl")
    if backend is None:
        backend = model.Fake(turns, caps=model.Caps(context_tokens=1000, model="test-1"))
    agent = Agent(backend=backend, tools=TOOLS, log=log, policy=ALLOW,
                  max_turns=max_turns, max_runtime=max_runtime, grace_turns=grace_turns,
                  clock=clock, run_id="r1", root=tmp_path, attended=True)
    return agent, clock


def steps(agent, kind, **fields):
    return [s for s in agent.log.steps() if s["kind"] == kind
            and all(s.get(k) == v for k, v in fields.items())]


def ended(agent):
    return steps(agent, "ended")[0]


def assistant_turns(agent):
    return steps(agent, "message", role="assistant")


def test_a_run_inside_its_budgets_is_untouched(tmp_path):
    """The budgets are a bound, not a presence: a run that finishes inside both never hears
    about them, and the record carries them on the fixed head for the reader who will ask."""
    agent, _ = build([model.turn_text("done")], tmp_path, max_turns=12, max_runtime=100)
    assert agent.run("go") == "done"
    assert ended(agent)["reason"] == "complete" and ended(agent)["exit_code"] == 0
    assert not steps(agent, "notice"), "no budget trip, no notice"
    head = steps(agent, "started")[0]
    assert head["max_turns"] == 12 and head["grace_turns"] == 3 and head["max_runtime"] == 100


def test_an_unset_time_budget_leaves_no_field_on_the_head(tmp_path):
    agent, _ = build([model.turn_text("done")], tmp_path, max_turns=12)
    agent.run("go")
    head = steps(agent, "started")[0]
    assert "max_runtime" not in head, "a bound the operator set none must not be stamped"
    assert head["max_turns"] == 12 and head["grace_turns"] == 3


def test_a_turn_budget_trip_is_told_to_wrap_up_and_an_answer_ends_complete(tmp_path):
    """The subagent property: two work turns, the trip at the boundary before the third, and
    one grace turn that delivers. The run ends complete — a bounded run ends with an answer."""
    turns = [model.turn_tool("c1", "echo", {"text": "a"}),
             model.turn_tool("c2", "echo", {"text": "b"}),
             model.turn_text("here is the answer")]
    agent, _ = build(turns, tmp_path, max_turns=2, grace_turns=3)
    assert agent.run("work") == "here is the answer"
    end = ended(agent)
    assert end["reason"] == "complete" and end["exit_code"] == 0
    assert len(assistant_turns(agent)) == 3, "two work turns plus the grace turn that delivered"
    notices = steps(agent, "notice", subject="nudge")
    assert len(notices) == 1 and "2 of 2 turns" in notices[0]["text"]
    wrap = steps(agent, "message", role="user", text=prompts.load("wrap_up"))
    assert wrap and wrap[0]["issuer"] == "core", "the wrap-up is a core word in the conversation"


def test_grace_spent_without_an_answer_ends_with_the_budget_reason(tmp_path):
    """Leeway spent is not leeway skipped: the budget's two turns run, the trip is told, the
    two grace turns run, and only then does the run end — with the budget's word, exit 1."""
    turns = [model.turn_tool(f"c{i}", "echo", {"text": str(i)}) for i in range(4)]
    agent, _ = build(turns, tmp_path, max_turns=2, grace_turns=2)
    agent.run("work")
    end = ended(agent)
    assert end["reason"] == "max_turns" and end["exit_code"] == 1
    assert len(assistant_turns(agent)) == 4, "two work turns plus two grace turns, no more"
    assert steps(agent, "message", role="user", text=prompts.load("wrap_up")), \
        "the model was still told to wrap up, and spent the telling on more work"


def test_a_time_budget_trip_names_max_runtime(tmp_path):
    """The wall trips alone: the turn count is far inside its bound, the clock is not. The
    run is named for the one that ran out — a reader grouping runs by reason is not told a
    turns story for a time stop."""
    turns = [model.turn_tool(f"c{i}", "echo", {"text": str(i)}) for i in range(4)]
    clock = Clock()
    agent, _ = build(turns, tmp_path, max_turns=50, max_runtime=100, grace_turns=2,
                     clock=clock,
                     backend=TickingBackend(turns, model.Caps(context_tokens=1000,
                                                              model="test-1"), clock,
                                            tick=50))
    agent.run("work")
    end = ended(agent)
    assert end["reason"] == "max_runtime" and end["exit_code"] == 1
    assert len(assistant_turns(agent)) == 4, "two work turns plus two grace turns"
    notices = steps(agent, "notice", subject="nudge")
    assert len(notices) == 1
    assert notices[0]["text"].startswith("budget ran out (100s of 100s)"), notices[0]["text"]


def test_a_time_budget_trip_that_delivers_ends_complete(tmp_path):
    turns = [model.turn_tool("c1", "echo", {"text": "a"}),
             model.turn_tool("c2", "echo", {"text": "b"}),
             model.turn_text("done under the wall")]
    clock = Clock()
    agent, _ = build(turns, tmp_path, max_turns=50, max_runtime=100, grace_turns=2,
                     clock=clock,
                     backend=TickingBackend(turns, model.Caps(context_tokens=1000,
                                                              model="test-1"), clock,
                                            tick=50))
    assert agent.run("work") == "done under the wall"
    end = ended(agent)
    assert end["reason"] == "complete" and end["exit_code"] == 0


def test_when_both_budgets_run_out_the_turns_budget_names_the_run(tmp_path):
    """One trip, named for the unit the loop counts: both run out on the same boundary, the
    run is named for the turn bound, and the notice says both, because the record does not
    smooth."""
    turns = [model.turn_tool("c1", "echo", {"text": "a"}),
             model.turn_tool("c2", "echo", {"text": "b"}),
             model.turn_text("late but delivered")]
    clock = Clock()
    agent, _ = build(turns, tmp_path, max_turns=2, max_runtime=100, grace_turns=3,
                     clock=clock,
                     backend=TickingBackend(turns, model.Caps(context_tokens=1000,
                                                              model="test-1"), clock,
                                            tick=50))
    assert agent.run("work") == "late but delivered"
    end = ended(agent)
    assert end["reason"] == "complete" and end["exit_code"] == 0
    notices = steps(agent, "notice", subject="nudge")
    assert notices[0]["text"].startswith("budget ran out (2 of 2 turns; 100s of 100s)"), \
        notices[0]["text"]


def test_zero_grace_is_the_hard_stop_with_a_notice_but_no_word_to_the_model(tmp_path):
    """Grace 0 is the old wall, made explicit: the trip is recorded, the run ends at the
    boundary, and no wrap-up message spends a turn that does not exist."""
    turns = [model.turn_tool("c1", "echo", {"text": "a"}), model.turn_text("never reached")]
    agent, _ = build(turns, tmp_path, max_turns=1, grace_turns=0)
    agent.run("work")
    end = ended(agent)
    assert end["reason"] == "max_turns" and end["exit_code"] == 1
    assert len(assistant_turns(agent)) == 1, "the budget's one turn, no grace"
    assert steps(agent, "notice", subject="nudge"), "the trip is still recorded"
    assert not steps(agent, "message", role="user", text=prompts.load("wrap_up")), \
        "a word the model would never get to answer is not said"


def test_each_objective_starts_its_own_budget(tmp_path):
    """The budgets judge the objective, not the session: the second objective is not judged
    by the turns or the clock the first spent, the way the recovery counters are not."""
    agent, _ = build([model.turn_text("one"), model.turn_text("two")], tmp_path,
                     max_turns=1, grace_turns=0)
    assert agent.run("first") == "one"
    assert agent.run("second") == "two"
    assert [s["reason"] for s in steps(agent, "ended")] == ["complete", "complete"]


def test_the_record_round_trips_the_new_fields_and_reason(tmp_path):
    """The taxonomy is closed and the reader is strict: a log carrying the new head fields and
    the new reason word is read back strict, and a reason the vocabulary does not carry is
    still refused on read."""
    agent, _ = build([model.turn_text("done")], tmp_path, max_turns=5, max_runtime=75)
    agent.run("go")
    path = tmp_path / "log.jsonl"
    steps_read, damage = record.read(path)
    assert damage == "" and steps_read
    head = next(s for s in steps_read if s["kind"] == "started")
    assert head["max_turns"] == 5 and head["max_runtime"] == 75

    log = Log(tmp_path / "foreign.jsonl")
    log.append("started", "core", run_id="x", root=str(tmp_path),
               max_turns=5, grace_turns=2, max_runtime=75)
    log.append("ended", "core", reason="max_runtime", exit_code=1)
    steps_read, damage = record.read(tmp_path / "foreign.jsonl")
    assert damage == "" and steps_read[-1]["reason"] == "max_runtime"


def test_a_reason_the_vocabulary_does_not_carry_is_refused(tmp_path):
    log = Log(tmp_path / "log.jsonl")
    with pytest.raises(record.Refused):
        log.append("ended", "core", reason="max_wall", exit_code=1)
