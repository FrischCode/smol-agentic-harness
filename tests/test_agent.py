"""The loop: what gets appended, when a command takes effect, and what a refusal looks like
from both sides. Every test drives a scripted backend, so these assert the harness's own
behaviour and never a server's."""
import os
import pathlib
import re
import time

import pytest

from sah import compact, ingest, model, prompts
from sah.tools import fs
from sah.agent import Agent, NUDGE_LIMIT, Verdict
from sah.record import Log
from sah.contract import Media


def echo(text: str) -> str:
    """Upper-case the text."""
    return text.upper()


def save(path: str, text: str) -> str:
    """Write text to a path."""
    return f"saved {path}"


def crash(text: str) -> str:
    """Always fails."""
    raise RuntimeError("disk on fire")


TOOLS = {"echo": echo, "save": save, "crash": crash}
# A toy tool declares its access like any other, because the loop reads a declaration for every tool it
# is given: `echo` and `crash` take text and touch no place, while `save` names the place the model typed
# so the veto point has something to judge -- dispatch follows the declaration now, not the tool's name.
echo.access, echo.address = "none", None
crash.access, crash.address = "none", None
save.access, save.address = "write", "path"


def ALLOW(name, args, *, access=None, address=None):
    return Verdict(True)


def DENY(name, args, *, access=None, address=None):
    return Verdict(False, "path is outside the run root; widen the grant to write it")


def _toy(func, access="none", address=None):
    """A test double declares how it touches the world, like any other tool. Most doubles only take text
    and touch no place; a test that wants the veto point to judge something passes access/address.
    The declared description gets a default the same way: the loop reads one for every tool it is
    given, and a double that never says what it does is a hole the production path now closes.
    This file tests the loop -- the declaration contract itself lives in test_ext.py (boot refusal) and
    test_cli.py (every built-in tool's declaration is truthful).
    """
    if getattr(func, "access", None) is None:
        func.access, func.address = access, address
    if getattr(func, "description", None) is None:
        func.description = f"A test double: {func.__name__}."
    return func


def build(turns, tmp_path, *, policy=ALLOW, tools=None, max_turns=6, vision=False, context=1000,
          system="", grace_turns=3, attended=True, cwd=None):
    table = {name: _toy(func) for name, func in (tools or TOOLS).items()}
    log = Log(tmp_path / "log.jsonl")
    backend = model.Fake(turns, caps=model.Caps(context_tokens=context, model="test-1",
                                                supports_vision=vision))
    agent = Agent(backend=backend, tools=table, log=log, policy=policy,
                  max_turns=max_turns, grace_turns=grace_turns, run_id="r1", root=tmp_path,
                  system=system, attended=attended, cwd=cwd)
    return agent, backend


def kinds(agent):
    return [s["kind"] for s in agent.log.steps()]


def one(agent, kind):
    return next(s for s in agent.log.steps() if s["kind"] == kind)


def many(agent, kind):
    return [s for s in agent.log.steps() if s["kind"] == kind]


def last(agent, kind):
    return [s for s in agent.log.steps() if s["kind"] == kind][-1]


def test_a_plain_turn_runs_and_reports_why_it_stopped(tmp_path):
    agent, _ = build([model.turn_text("hello", " world")], tmp_path)
    assert agent.run("hi") == "hello world"
    assert kinds(agent) == ["started", "message", "message", "ended"]
    assert one(agent, "started")["model"] == "test-1"
    ended = one(agent, "ended")
    assert ended["reason"] == "complete" and ended["exit_code"] == 0


def test_a_turn_records_its_wall_time_and_the_window_it_was_written_against(tmp_path):
    """`message.ms` and `message.context_max` are what the console footer prints, and the loop wrote
    neither: a declared field with a reader and no producer is worse than an absent one, because the
    taxonomy reads as a promise and the reader's branch reads as working.

    The ceiling is taken from the live backend's declaration and not from a run-wide remembered value,
    because a resume may name a different `--context`, and occupancy without its ceiling is not yet a
    statement about pressure — 4,000 tokens is nothing at 131,072 and nearly everything at 4,096."""
    agent, _ = build([[model.TextDelta("hi"), model.Usage(50, 2), model.Stopped("stop")]], tmp_path)
    agent.run("go")
    msg = last(agent, "message")
    assert isinstance(msg["ms"], int) and msg["ms"] >= 0
    assert msg["context_max"] == 1000                     # the Caps this run was built with


def test_the_wall_time_covers_the_stream_and_not_the_call_that_asked_for_it(tmp_path):
    """`complete()` returns a generator, so nothing is sent until the loop asks for the first event: the
    time a turn took is the time its stream took, not the time it took to build the request. A clock
    stopped when `complete()` returned reports ~0 ms for the backend below, which spends 60 ms before its
    first token, and 150 ms of margin because a sleep is a promise about a quantum and not about a
    duration: a threshold at the sleep's own length goes red on a loaded box. The clock is the real one,
    because a frozen one makes the two waits identical."""
    class Slow:
        """A backend that thinks for 250 ms before its first token, and reports no usage at all."""

        def capabilities(self):
            return model.Caps(context_tokens=100, model="slow-1")

        def complete(self, **_request):
            def stream():
                time.sleep(0.25)
                yield model.TextDelta("late")
                yield model.Stopped("stop")
            return stream()

    agent = Agent(backend=Slow(), tools={}, log=Log(tmp_path / "log.jsonl"), policy=ALLOW,
                  max_turns=2, run_id="r1", root=tmp_path)
    agent.run("go")
    msg = last(agent, "message")
    assert msg["ms"] >= 150, f"the turn's own time was not measured: {msg}"
    assert "tokens_in" not in msg                         # a server that says nothing still took time


def test_what_the_server_said_of_cache_and_speed_reaches_the_step_only_when_said(tmp_path):
    """Each figure is recorded when the server volunteered it and omitted when it did not. Writing zero
    for silence would make every cold run look like a warm one that missed, and every slow server look
    like a fast one — and the taxonomy refuses an optional field carrying None anyway."""
    agent, _ = build([[*model.turn_tool("c1", "echo", {"text": "hi"}),
                       model.Usage(50, 2, cached=40, tps=45.0, prefill_tps=800.0)],
                      [*model.turn_text("done"), model.Usage(60, 3)]], tmp_path)
    agent.run("go")
    turns = [s for s in many(agent, "message") if s.get("role") == "assistant"]
    assert len(turns) == 2
    assert turns[0]["tokens_cached"] == 40 and turns[0]["tps"] == 45.0
    assert turns[0]["prefill_tps"] == 800.0
    for field in ("tokens_cached", "tps", "prefill_tps"):
        assert field not in turns[1], f"{field} was invented for a turn whose server said nothing"


def test_a_tool_round_trip_and_the_model_sees_the_result(tmp_path):
    agent, backend = build([model.turn_tool("c1", "echo", {"text": "hi"}),
                            model.turn_text("done")], tmp_path)
    agent.run("go")
    assert kinds(agent) == ["started", "message", "message", "call", "operation", "result",
                            "message", "ended"]
    sent = backend.calls[1].messages
    assert [m.role for m in sent] == ["user", "assistant", "tool"]
    assert sent[-1].content == "HI" and sent[-1].tool_call_id == "c1"


def test_the_address_a_call_touches_is_recorded(tmp_path):
    agent, _ = build([model.turn_tool("c1", "save", {"path": "notes/x.md", "text": "t"}),
                      model.turn_text("ok")], tmp_path)
    agent.run("go")
    assert one(agent, "operation")["address"] == "notes/x.md"


def test_a_refusal_is_recorded_and_is_also_what_the_model_is_told(tmp_path):
    agent, backend = build([model.turn_tool("c1", "save", {"path": "/etc/passwd", "text": "t"}),
                            model.turn_text("understood")], tmp_path, policy=DENY)
    agent.run("go")
    operation, result = one(agent, "operation"), one(agent, "result")
    assert operation["outcome"] == "refused" and "widen the grant" in operation["reason"]
    assert result["ok"] is False and result["content"] == operation["reason"]
    assert backend.calls[1].messages[-1].content == operation["reason"]


def test_an_unknown_tool_is_refused_naming_the_declared_tools(tmp_path):
    agent, _ = build([model.turn_tool("c1", "delete", {"path": "x"}),
                      model.turn_text("sorry")], tmp_path)
    agent.run("go")
    assert one(agent, "operation")["outcome"] == "refused"
    assert "declared tools are" in one(agent, "operation")["reason"]


def test_a_raising_tool_becomes_a_failed_result_not_a_crash(tmp_path):
    agent, _ = build([model.turn_tool("c1", "crash", {"text": "x"}),
                      model.turn_text("recovered")], tmp_path)
    agent.run("go")
    result = one(agent, "result")
    assert result["ok"] is False and "RuntimeError: disk on fire" in result["content"]


def test_a_steer_lands_at_the_boundary_and_is_attributed(tmp_path):
    agent, backend = build([model.turn_tool("c1", "echo", {"text": "a"}),
                            model.turn_text("done")], tmp_path)
    agent.submit("steer", args={"text": "use the other file"}, issuer="operator")
    agent.run("go")
    assert one(agent, "command")["issuer"] == "operator"
    assert one(agent, "decision")["outcome"] == "accepted"
    steer = [s for s in many(agent, "message") if s["role"] == "user" and s["issuer"] == "operator"]
    assert steer[0]["text"] == "use the other file"
    assert any(m.content == "use the other file" for m in backend.calls[1].messages)


def test_an_empty_steer_is_refused(tmp_path):
    agent, _ = build([model.turn_text("x")], tmp_path)
    agent.submit("steer", args={}, issuer="operator")
    agent.run("go")
    assert one(agent, "decision")["outcome"] == "refused"
    assert "steer needs args.text" in one(agent, "decision")["reason"]


def test_an_undeclared_action_is_refused_by_name_and_changes_nothing(tmp_path):
    agent, _ = build([model.turn_text("fine")], tmp_path)
    agent.submit("pause", issuer="operator")
    agent.run("go")
    assert one(agent, "decision")["outcome"] == "refused"
    assert "undeclared action 'pause'" in one(agent, "decision")["reason"]
    assert one(agent, "ended")["reason"] == "complete"


def test_an_interrupt_finishes_the_in_flight_turn_then_stops(tmp_path):
    holder = {}

    def halt(text: str) -> str:
        """Stop the run."""
        holder["agent"].submit("interrupt", issuer="operator")
        return "halting"

    script = [model.turn_tool("c1", "halt", {"text": "x"}), model.turn_text("unreached")]
    agent, backend = build(script, tmp_path, tools={"halt": halt}, max_turns=5)
    holder["agent"] = agent
    agent.run("go")
    assert one(agent, "ended")["reason"] == "interrupted"
    assert one(agent, "ended")["exit_code"] == 0
    assert backend.calls[0].messages[0].content == "go", "the turn in flight still completed"


def test_a_failed_backend_is_a_failure_not_an_empty_turn(tmp_path):
    agent, _ = build([[model.Failed("connection refused")]], tmp_path)
    agent.run("go")
    assert last(agent, "message")["outcome"] == "failed"
    assert last(agent, "message")["error"] == "connection refused"
    assert one(agent, "ended")["reason"] == "failed" and one(agent, "ended")["exit_code"] == 1


def test_a_call_from_a_turn_that_died_mid_stream_is_recorded_but_not_run(tmp_path):
    """A stream can hand over a complete tool call and then die (the socket drops, a later chunk is
    malformed). Running it means the harness performs an act on an instruction the model never finished
    giving -- and the act may be a write or a command. The run ends `failed` either way, so nothing is
    gained by acting first.

    The record stays complete, though: the call happened, so it belongs in the log, refused, with its
    result. What is not done is asking the policy or the operator about a call that will never run.
    """
    asked: list = []

    def spy(name, args, **_):
        asked.append(name)
        return Verdict(True)

    agent, _ = build([[model.ToolCall("c1", "save", {"path": "x", "text": "t"}),
                       model.Failed("connection reset by peer")]], tmp_path, policy=spy)
    agent.run("go")

    assert "call" in kinds(agent), "the call the model made has to be in the log even though it was not run"
    assert asked == [], "the policy was consulted for a call that was never going to run"
    res = last(agent, "result")
    assert res["ok"] is False and "not run" in res["content"]
    assert "saved" not in res["content"], "the tool ran on a turn the backend never finished"
    assert last(agent, "operation")["outcome"] == "refused"
    assert last(agent, "ended")["reason"] == "failed"


def test_a_call_refused_by_a_failed_turn_keeps_the_pairing_replay_needs(tmp_path):
    """The pairing invariant applies here: replay rebuilds the conversation `step()` had. A `call` logged
    with no `result`, or an assistant message carrying a tool call nobody answered, resumes as a
    conversation with a dangling call -- which a chat template rejects, so the run cannot be resumed at
    all. That is the reason to record the refusal rather than drop the call silently.
    """
    agent, _ = build([[model.ToolCall("c1", "save", {"path": "x", "text": "t"}),
                       model.Failed("connection reset by peer")]], tmp_path)
    agent.run("go")
    assert len(agent.messages) == len(agent._trace), "the message list and its trace came apart"
    assistant = [m for m in agent.messages if m.role == "assistant" and m.tool_calls]
    answered = [m.tool_call_id for m in agent.messages if m.role == "tool"]
    # A message carries its calls as (id, name, arguments) tuples, which is the shape the adapter sends.
    assert [c[0] for a in assistant for c in a.tool_calls] == answered, \
        "a call was left unanswered in the conversation a resume would send"


def test_a_turn_cut_off_by_the_limit_records_no_call_at_all(tmp_path):
    """The asymmetry between this and a failed turn, written down because it looks like an inconsistency:
    a turn the output limit ended is *retried*, so its calls leave no trace (a turn that acted while the
    model was being asked to try again would leave the log describing acts the turn never committed to);
    a turn that failed is *terminal*, so its calls are recorded as refused. Each records what the next
    reader will find true.
    """
    agent, _ = build([[model.ToolCall("c1", "save", {"path": "x", "text": "t"}),
                       model.Failed("malformed arguments", truncated=True),
                       ], [model.TextDelta("saved it"), model.Stopped("stop")]], tmp_path)
    agent.run("go")
    assert "call" not in kinds(agent), \
        "a retried turn's calls were recorded; a retry must not re-log what the log already holds"
    assert last(agent, "ended")["reason"] == "complete"


# --- a turn cut off by the output limit ---------------------------------------------------------
#
# `--budget` (max_tokens) can end a turn mid-sentence. The server says `finish_reason: "length"`, and
# treating that as a finished answer is the quiet failure: the operator reads a truncated reply and
# nothing in the log or the exit code says it was cut. A cut-off turn can carry prose, a half-built
# tool call, or nothing at all (reasoning ate the budget), so recovery covers all three.

def test_an_answer_cut_off_by_the_limit_is_continued_and_reads_as_one_reply(tmp_path):
    """Recover rather than end: the pieces concatenate into the answer the run reports, because a
    caller reading `run()`'s return value -- or a parent reading a child's last said -- wants the
    answer, not its last fragment.
    """
    agent, _ = build([[model.TextDelta("half"), model.Stopped("length")],
                      [model.TextDelta(" rest"), model.Stopped("stop")]], tmp_path)
    text = agent.run("go")
    assert one(agent, "ended")["reason"] == "complete"
    assert text == "half rest", f"the answer is the pieces, not the last one: {text!r}"
    assistant = [s for s in many(agent, "message") if s["role"] == "assistant"]
    assert assistant[0]["outcome"] == "truncated"
    nudges = [s for s in many(agent, "notice") if s["subject"] == "nudge"]
    assert len(nudges) == 1, [s["text"] for s in nudges]
    assert "output limit" in nudges[0]["text"]


def test_a_cut_off_reply_says_continue_without_repeating(tmp_path):
    """The nudge is model-facing instruction, so it lives in `sah/prompts/` like the guard's steer, and
    it has to say what to do (continue where it stopped) and what not to do (start over -- a model that
    restates the answer burns the budget it was just given).
    """
    agent, _ = build([[model.TextDelta("half"), model.Stopped("length")],
                      [model.TextDelta(" rest"), model.Stopped("stop")]], tmp_path)
    agent.run("go")
    asked = [s["text"] for s in many(agent, "message")
             if s["role"] == "user" and s["issuer"] == "core"]
    assert len(asked) == 1, asked
    assert "Continue exactly where it stopped" in asked[0] and "do not repeat" in asked[0].lower()


def test_a_whole_call_from_a_turn_the_limit_ended_is_still_not_run(tmp_path):
    """The obvious exception to make and refuse. A turn can hold a well-formed call and still be cut off
    after it, and running that call while the model is being told "it was not run, send a smaller one"
    makes the log describe an act the model believes never happened -- and the write it performed is one
    the next turn may repeat. A truncated turn dispatches nothing; the whole turn is retried.
    """
    agent, _ = build([[model.ToolCall("c1", "echo", {"text": "x"}), model.Stopped("length")],
                      [model.TextDelta("redone"), model.Stopped("stop")]], tmp_path)
    agent.run("go")
    assert many(agent, "call") == [], "a truncated turn runs no calls, whole ones included"
    assert one(agent, "ended")["reason"] == "complete"


def test_still_cut_off_after_two_continuations_the_run_ends_truncated(tmp_path):
    """Recovery is bounded, and the ending names the cause and the remedy. The reason is its own, not
    `failed`: "the model could not answer" and "the output limit is too small" are different things to
    read at 3am, and only one of them is fixed by raising --budget.
    """
    agent, _ = build([[model.TextDelta("a"), model.Stopped("length")],
                      [model.TextDelta("b"), model.Stopped("length")],
                      [model.TextDelta("c"), model.Stopped("length")]], tmp_path)
    agent.run("go")
    ended = one(agent, "ended")
    assert ended["reason"] == "truncated" and ended["exit_code"] == 1
    last = [s["text"] for s in many(agent, "notice") if s["subject"] == "nudge"][-1]
    assert "--budget" in last, f"the ending has to name the remedy: {last!r}"


def test_a_cut_off_tool_call_is_never_run(tmp_path):
    """The adapter reports a half-built call as a failure to parse; with a `length` finish reason the
    loop must know the *limit* broke it, not the model. Running it is not an option -- the arguments
    are cut off mid-JSON -- so the call is not dispatched, and the model is told to send a smaller one.
    """
    agent, _ = build([[model.Failed("malformed tool call arguments for 'write': ...", truncated=True),
                       model.Stopped("length")],
                      [model.TextDelta("done smaller"), model.Stopped("stop")]], tmp_path)
    agent.run("go")
    assert many(agent, "call") == [], "a call whose arguments were cut off must not be dispatched"
    assert one(agent, "ended")["reason"] == "complete"
    asked = [s["text"] for s in many(agent, "message")
             if s["role"] == "user" and s["issuer"] == "core"]
    assert asked and "smaller" in asked[0]


def test_a_turn_cut_off_while_reasoning_says_answer_more_briefly(tmp_path):
    """Measured on this server with a small budget: all 40 tokens went to reasoning and the content
    came back empty with `finish_reason: length`. Telling that model to "continue where you stopped"
    is nonsense -- it wrote nothing to continue from.
    """
    agent, _ = build([[model.ReasoningDelta("thinking hard"), model.Stopped("length")],
                      [model.TextDelta("brief answer"), model.Stopped("stop")]], tmp_path)
    agent.run("go")
    asked = [s["text"] for s in many(agent, "message")
             if s["role"] == "user" and s["issuer"] == "core"]
    assert asked and "more briefly" in asked[0]
    assert one(agent, "ended")["reason"] == "complete"


def test_the_reasoning_signature_reaches_the_message_and_the_log(tmp_path):
    """The field that carried a turn's reasoning rides with the turn: into the stored message the
    next request re-sends, and into the log step a resume re-reads. A turn that arrived in
    `reasoning` must not be recorded as if it had arrived in `reasoning_content`."""
    agent, _ = build([[model.ReasoningDelta("th", "reasoning"),
                      model.ReasoningDelta("ink", "reasoning"),
                      model.TextDelta("done"),
                      model.Stopped("stop")],
                     [model.TextDelta("second"), model.Stopped("stop")]], tmp_path)
    agent.run("go")
    reasoned = [m for m in agent.messages if m.reasoning]
    assert len(reasoned) == 1 and reasoned[0].reasoning == "think" \
        and reasoned[0].reasoning_field == "reasoning", reasoned
    steps = [s for s in many(agent, "message") if s.get("reasoning")]
    assert steps and all(s["reasoning_field"] == "reasoning" for s in steps), steps


def test_an_empty_turn_is_nudged_three_times_and_then_fails(tmp_path):
    agent, _ = build([[], [], [], []], tmp_path, max_turns=9)
    agent.run("go")
    assert len(many(agent, "notice")) == 3
    assert many(agent, "notice")[0]["subject"] == "nudge"
    assert one(agent, "ended")["reason"] == "failed"


def test_an_empty_turn_nudge_reaches_the_model_with_a_remedial_word(tmp_path):
    """The nudge used to be a notice only: the operator saw it, the model never did, and a silent
    re-ask gives a model no reason to answer an empty turn differently. The model is told what came
    back and the two exits that leave it — re-emit the call fresh, or say done."""
    agent, _ = build([[], model.turn_text("done")], tmp_path, max_turns=6)
    agent.run("go")
    core_msgs = [s for s in agent.log.steps()
                 if s["kind"] == "message" and s.get("issuer") == "core"]
    assert len(core_msgs) == 1
    # Compared against the file, the way the disclosure test compares against the policy constant:
    # the word the model receives is the prompt file's word, byte for byte.
    assert core_msgs[0]["text"] == prompts.load("empty_turn")
    assert "already complete" in core_msgs[0]["text"], "the completion exit is missing from the nudge"
    assert one(agent, "ended")["reason"] == "complete"


def test_an_unattended_prose_turn_is_nudged_not_completed(tmp_path):
    """The T3-v1 shape: planning prose in a run no operator is watching. Reading a prose-only turn
    as completion ended that run complete with nothing delivered; the structural nudge asks for
    the call or the declaration instead, and exhaustion ends blocked — an unwatched run that
    cannot prove done must not end complete."""
    agent, _ = build([model.turn_text("first I will survey the tree"),
                      model.turn_text("then I will write the file"),
                      model.turn_text("then I will run the checks"),
                      model.turn_text("finally I will report")],
                     tmp_path, max_turns=9, attended=False)
    agent.run("do the task")
    ended = one(agent, "ended")
    assert ended["reason"] == "blocked" and ended["exit_code"] == 1
    # three tellings, each the harness's word in the user's seat; the fourth no-call turn is the
    # exhaustion, named in the record. The fixture's turns report a finish word but no usage, so
    # the clause stands at its floor (the turn ended) — the numbers are a projection of a
    # measurement, never a guess — and the pin renders the file the way the call site does,
    # with the floor value.
    nudges = [s for s in many(agent, "message") if s.get("issuer") == "core"]
    assert len(nudges) == NUDGE_LIMIT
    assert all(s["text"] == prompts.load("no_call_nudge", evidence="— it ended") for s in nudges)
    notices = many(agent, "notice")
    assert notices[0]["text"] == "the turn produced prose but no tool call"
    assert "cannot prove done and ends blocked" in notices[-1]["text"]


def test_the_no_call_nudge_quotes_the_turns_wire_facts_when_reported(tmp_path):
    """The corrective clause is a measurement, not an assertion: when the server reports the
    turn's completion count and finish word, the nudge quotes both, so the model is told what
    actually happened instead of inferring a cause for the missing call. The later turns report
    no usage: their nudges carry the sentence without the numbers, and no number is invented."""
    agent, _ = build([[model.TextDelta("first I will survey the tree"),
                       model.Usage(prompt_tokens=900, completion_tokens=1142),
                       model.Stopped("stop")],
                      model.turn_text("then I will write the file"),
                      model.turn_text("then I will run the checks"),
                      model.turn_text("finally I will report")],
                     tmp_path, max_turns=9, attended=False, context=4096)
    agent.run("do the task")
    nudges = [s["text"] for s in many(agent, "message") if s.get("issuer") == "core"]
    assert len(nudges) == NUDGE_LIMIT
    assert "it ended at 1142 tokens with finish=stop" in nudges[0]
    assert nudges[1] == prompts.load("no_call_nudge", evidence="— it ended")
    assert one(agent, "ended")["reason"] == "blocked"


def test_an_unattended_empty_turn_gets_the_no_call_nudge(tmp_path):
    """The empty-turn path, which the prose-case tests do not reach: a turn with no text and no
    call is nudged with the no-call nudge (the wire facts carry the correction — was not cut off,
    the finish word), gets the neither-prose notice, and exhaustion ends blocked."""
    agent, _ = build([[model.Stopped("stop")],
                      [model.Stopped("stop")],
                      [model.Stopped("stop")],
                      [model.Stopped("stop")]],
                     tmp_path, max_turns=9, attended=False)
    agent.run("do the task")
    nudges = [s for s in many(agent, "message") if s.get("issuer") == "core"]
    assert len(nudges) == NUDGE_LIMIT
    assert all(s["text"] == prompts.load("no_call_nudge", evidence="— it ended") for s in nudges)
    assert many(agent, "notice")[0]["text"] == "the turn produced neither prose nor a tool call"
    assert one(agent, "ended")["reason"] == "blocked"


class CrashingBackend:
    """A backend that raises where one should stream: the crash class the `ended` writer is for.
    A scripted error *event* ends the turn the normal way (a failed turn, recorded and terminal);
    this is the exception that would otherwise take the process with it, and the run still has to
    say it ended."""

    def __init__(self):
        self.caps = model.Caps(context_tokens=100, model="t")

    def capabilities(self):
        return self.caps

    def complete(self, *, system, messages, tools, thinking="off", max_tokens=None):
        raise RuntimeError("the wire died")


def test_a_crash_ends_the_run_and_the_ending_says_crashed(tmp_path):
    """A run that died of any other exception used to leave no `ended` line, which `recall` and
    `child_status` read as still running. The crash is recorded beside its ending -- the traceback
    on the notice, the word on the ending -- and the exception goes on: the operator still gets
    the traceback, the log gets the fact."""
    log = Log(tmp_path / "log.jsonl")
    agent = Agent(backend=CrashingBackend(), tools={}, log=log, policy=ALLOW, max_turns=3,
                  run_id="r1", root=tmp_path, attended=True)
    with pytest.raises(RuntimeError, match="the wire died"):
        agent.run("go")
    steps = agent.log.steps()
    ended = [s for s in steps if s["kind"] == "ended"]
    assert len(ended) == 1, "the crash writes exactly one ending, and no other path may have"
    assert ended[0]["reason"] == "crashed" and ended[0]["exit_code"] == 1
    assert steps[-1]["kind"] == "ended", "the ending is the last word in the record"
    crash = [s for s in steps if s["kind"] == "notice" and s.get("subject") == "crash"]
    assert len(crash) == 1
    assert "the wire died" in crash[0]["text"] and "Traceback" in crash[0]["text"]


def test_an_unattended_run_ends_complete_when_the_declaration_lands(tmp_path):
    """The declaration is the exit an unattended run is offered: a prose turn would be nudged, but
    the verify_objective call ends the run complete, recorded beside the ending, with no nudge
    spent. The harness records the act; whether the task is served is the caller's check."""
    def declare() -> str:
        """A test double for the declaration: no world touch, no parameters."""
        return "declared done. the run ends now"

    agent, _ = build([model.turn_tool("c1", "echo", {"text": "x"}),
                      model.turn_tool("c2", "verify_objective", {})],
                     tmp_path, max_turns=6, attended=False,
                     tools={"echo": echo, "verify_objective": declare})
    agent.run("do the task")
    ended = one(agent, "ended")
    assert ended["reason"] == "complete" and ended["exit_code"] == 0
    assert last(agent, "call")["name"] == "verify_objective"
    assert last(agent, "notice")["subject"] == "verify"
    assert [s for s in many(agent, "message") if s.get("issuer") == "core"] == []


def test_a_run_that_never_stops_ends_when_its_turns_run_out(tmp_path):
    """The ceiling still holds: a run that never stops spends its work turns, is told to wrap
    up, spends the grace, and only then ends with the budget's word. The wrap-up is part of
    the ending, not an interruption of it."""
    forever = [[model.ToolCall("c", "echo", {"text": "x"}), model.Stopped("tool_calls")]]
    agent, _ = build(forever * 6, tmp_path, max_turns=3, grace_turns=3)
    agent.run("go")
    assert one(agent, "ended")["reason"] == "max_turns"
    assert one(agent, "ended")["exit_code"] == 1
    assert len([s for s in agent.log.steps() if s["kind"] == "message"
                and s["role"] == "assistant"]) == 6, "three work turns plus three grace turns"
    assert any(s["text"] == prompts.load("wrap_up") for s in agent.log.steps()
               if s["kind"] == "message" and s["role"] == "user"), "the model was told"


def test_no_step_is_stamped_with_an_issuer_the_loop_did_not_derive(tmp_path):
    agent, _ = build([model.turn_tool("c1", "echo", {"text": "hi"}),
                      model.turn_text("done")], tmp_path)
    agent.submit("steer", args={"text": "hey"}, issuer="operator")
    agent.run("go", issuer="user")
    assert {s["issuer"] for s in agent.log.steps()} <= {"core", "model", "user", "operator"}


def test_every_verdict_in_the_run_is_readable_from_the_log(tmp_path):
    agent, _ = build([model.turn_tool("c1", "save", {"path": "x", "text": "t"}),
                      model.turn_text("ok")], tmp_path, policy=DENY)
    agent.run("go")
    verdicts = agent.log.verdicts()
    assert ("operation.outcome", "refused") in verdicts
    assert ("result.ok", False) in verdicts
    assert [v for k, v in verdicts if k == "message.outcome"] == ["ok", "ok"]


def test_the_log_is_one_object_per_line_with_a_sequence_the_log_owns(tmp_path):
    agent, _ = build([model.turn_text("a"), model.turn_text("b")], tmp_path)
    agent.run("one")
    agent.run("two")
    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(agent.log.steps()) == len({s["id"] for s in agent.log.steps()})
    assert [s["id"] for s in agent.log.steps()] == sorted(s["id"] for s in agent.log.steps())


def test_streamed_partials_are_live_only_never_stored(tmp_path):
    """The streamed-partials decision, made falsifiable: deltas reach a live client, but only the
    assembled final text becomes a step. A log of half-messages would make 'read the log' mean
    something different from 'watch the run', and replay could not tell a finished sentence from one
    cut mid-word. What streams and what is stored are the same facts, but only one is kept."""
    from sah.model import ReasoningDelta, TextDelta, Stopped
    agent, _ = build([[ReasoningDelta("th"), ReasoningDelta("ink"),
                       TextDelta("he"), TextDelta("llo"), Stopped("stop")]], tmp_path)
    seen = []
    agent.subscribe_event(seen.append)   # a live subscriber, e.g. the terminal
    agent.run("go")
    msgs = [s for s in agent.log.steps() if s["kind"] == "message" and s["role"] == "assistant"]
    assert len(msgs) == 1                                    # one assembled turn, not four partials
    assert msgs[0]["text"] == "hello" and msgs[0]["reasoning"] == "think"
    # the partials really did stream live, even though none of them was stored
    assert sum(1 for e in seen if isinstance(e, TextDelta)) == 2
    assert sum(1 for e in seen if isinstance(e, ReasoningDelta)) == 2


def spew(path: str = "") -> str:
    """Return a very long result."""
    return "\n".join(f"line {i} " + "z" * 80 for i in range(5000))


# Declared `none`: the result is generated, not read, whatever the unused `path` parameter hints.
spew.access, spew.address = "none", None
spew.description = "Return a very long result."


def test_an_over_long_result_is_bounded_and_the_whole_thing_spilled(tmp_path):
    """A tool that returns more than the cap cannot fill the context: the model gets a bounded
    preview + the path, the FULL output is saved under the run's own directory, and the log records
    the preview, not the blob."""
    from sah import ingest
    turns = [[model.ToolCall("c", "spew", {"path": "x"}), model.Stopped("tool_calls")],
             [model.TextDelta("done"), model.Stopped("stop")]]
    agent = Agent(backend=model.Fake(turns, caps=model.Caps(context_tokens=100, model="t")),
                  tools={"spew": spew}, log=Log(tmp_path / "log.jsonl"), policy=ALLOW,
                  max_turns=6, run_id="r1", root=tmp_path)
    agent.run("go")
    res = one(agent, "result")
    spilled = tmp_path / "out" / f"{one(agent, 'call')['id']}.txt"   # named by the log, not the backend
    assert res.get("truncated") is True and res["bytes"] > ingest.MAX_BYTES
    assert spilled.exists() and spilled.read_text(encoding="utf-8") == spew(), \
        "the whole output lives in the run's owned dir"
    assert "full output saved to" in res["content"] and spilled.name in res["content"]
    assert len(res["content"]) < res["bytes"], "the log stores the preview, not the blob"
    tool_msg = next(m for m in agent.messages if m.role == "tool")
    assert len(tool_msg.content) < res["bytes"], "the context the model reads next is the preview"


def test_a_small_result_is_neither_truncated_nor_spilled(tmp_path):
    turns = [[model.ToolCall("c", "echo", {"text": "hi"}), model.Stopped("tool_calls")],
             [model.TextDelta("done"), model.Stopped("stop")]]
    agent = Agent(backend=model.Fake(turns, caps=model.Caps(context_tokens=100, model="t")),
                  tools=TOOLS, log=Log(tmp_path / "log.jsonl"), policy=ALLOW, max_turns=6,
                  run_id="r1", root=tmp_path)
    agent.run("go")
    res = one(agent, "result")
    assert res["content"] == "HI" and not res.get("truncated") and "spilled" not in res
    assert not (tmp_path / "out").exists()


# --- the event bus & generator tools ----------------------------------------

def _turn_tool(name, call_id="c1"):
    return [[model.ToolCall(call_id, name, {}), model.Stopped("tool_calls")],
            [model.TextDelta("done"), model.Stopped("stop")]]


def _run_tool(tmp_path, tool, name):
    """Run one agent turn whose only act is `tool`, returning (agent, live events seen)."""
    log = Log(tmp_path / "log.jsonl")
    seen: list = []
    agent = Agent(backend=model.Fake(_turn_tool(name)), tools={name: _toy(tool)}, log=log,
                  policy=ALLOW, run_id="r1", root=tmp_path)
    agent.subscribe_event(seen.append)
    agent.run("go")
    return agent, seen


def test_a_streaming_tool_reports_chunks_live_stamped_with_its_call_id(tmp_path):
    def slow(n: int = 1) -> str:
        """A tool that yields progress."""
        yield "one"
        yield model.TextDelta(" two")          # a raw string and a TextDelta both work
        yield " three"
        return "the assembled result"

    agent, seen = _run_tool(tmp_path, slow, "slow")
    chunks = [e for e in seen if isinstance(e, model.TextDelta) and e.call_id == "c1"]
    assert "".join(c.text for c in chunks) == "one two three"
    assert all(c.call_id == "c1" for c in chunks)        # the loop stamps it; the tool never set it
    # one chunk type, routed by call_id: the turn's own prose is the same type but carries no id
    prose = [e for e in seen if isinstance(e, model.TextDelta) and e.text == "done"]
    assert prose and prose[0].call_id is None
    assert one(agent, "result")["content"] == "the assembled result"   # return wins, not the chunks


def test_a_plain_returning_tool_records_its_value_and_emits_no_stream_chunk(tmp_path):
    def ping(n: int = 1) -> str:
        """Returns a value; it does not stream."""
        return "pong"

    agent, seen = _run_tool(tmp_path, ping, "ping")
    assert one(agent, "result")["content"] == "pong"              # the value, unchanged
    # a plain return is NOT a stream: not one chunk reaches the bus (if it were mis-pumped as a
    # generator, iter("pong") would drip p/o/n/g out as live chunks — this pins that it does not)
    assert not any(isinstance(e, model.TextDelta) and e.call_id == "c1" for e in seen)


def test_a_plain_returning_stream_tool_records_the_concat_of_its_chunks(tmp_path):
    def lines(n: int = 1):
        """No return statement: the record is the concatenation of what it yielded."""
        yield "a\n"
        yield "b\n"

    agent, _ = _run_tool(tmp_path, lines, "lines")
    assert one(agent, "result")["content"] == "a\nb\n"


def test_a_stream_that_raises_midway_is_a_verdict_not_a_crash(tmp_path):
    def half_broken(n: int = 1):
        """Yields once, then explodes."""
        yield "start "
        raise RuntimeError("pipe burst")

    agent, seen = _run_tool(tmp_path, half_broken, "half_broken")   # must not raise out of run()
    result = one(agent, "result")
    assert result["ok"] is False and "pipe burst" in result["content"]
    # streaming is best-effort: the live chunk shown before the burst is display, but the *record*
    # is the failure, not the partial output — display and truth are allowed to differ here
    assert any(isinstance(e, model.TextDelta) and e.text == "start " for e in seen)
    assert "start" not in result["content"]


def test_a_broken_subscriber_cannot_stop_the_run(tmp_path):
    log = Log(tmp_path / "log.jsonl")
    agent = Agent(backend=model.Fake([[model.TextDelta("hi"), model.Stopped("stop")]]),
                  tools={}, log=log, policy=ALLOW, run_id="r1", root=tmp_path)

    def explode(event):
        raise ZeroDivisionError

    agent.subscribe_event(explode)
    agent.run("go")                                     # a renderer bug must not abort the turn
    assistant = next(s for s in agent.log.steps()
                     if s["kind"] == "message" and s.get("role") == "assistant")
    assert assistant["text"] == "hi"                    # the turn completed and recorded despite it


def test_an_empty_assistant_turn_never_enters_the_conversation(tmp_path):
    """A model turn that produced no prose and no tool call is recorded but NOT appended as an
    empty assistant message. The real bug: qwen3_coder renders an empty assistant turn as a blank
    'conversation start marker', and the next request reads that as 'the user message is empty'."""
    empty = [model.ReasoningDelta("thinking"), model.Stopped("stop")]  # reasoning, no prose, no call
    agent, _ = build([empty, model.turn_text("recovered")], tmp_path)
    agent.say("user", "go", issuer="user")
    agent.step()                             # the empty turn
    msgs = [m for m in agent.messages if m.role == "assistant"]
    assert not any(not m.content and not m.tool_calls for m in msgs), \
        "an empty assistant message with no tool_calls was put into the context"
    # it is still in the LOG: the turn happened, we just refuse to make it context
    assert any(s["kind"] == "message" and s.get("role") == "assistant" and not s["text"]
               for s in agent.log.steps())


def test_a_reasoning_only_turn_then_a_tool_call_still_works(tmp_path):
    """The empty turn is skipped, but the NEXT turn's tool call round-trips normally -- the guard
    is not allowed to break a real turn that follows the blank one."""
    empty = [model.ReasoningDelta("hmm"), model.Stopped("stop")]
    agent, _ = build([empty, model.turn_tool("c1", "echo", {"text": "hi"}, "did it"),
                      model.turn_text("done")], tmp_path)
    agent.say("user", "go", issuer="user")
    agent.run()
    assert [s["kind"] for s in agent.log.steps()].count("result") == 1
    assert agent.final_text == "done"


def test_a_tool_call_with_no_prose_still_enters_the_conversation(tmp_path):
    """The guard must NOT skip a turn that has no prose but DOES call a tool -- that is a normal
    tool turn and its tool_calls are required for the round-trip."""
    agent, _ = build([model.turn_tool("c1", "echo", {"text": "hi"}), model.turn_text("done")], tmp_path)
    agent.say("user", "go", issuer="user")
    agent.step()
    asst = [m for m in agent.messages if m.role == "assistant"]
    assert asst and asst[-1].tool_calls, "a tool turn with empty prose must still carry its calls"


# --- what is per objective, not per process ---------------------------------------------------

def test_an_interrupt_does_not_outlive_its_objective(tmp_path):
    """The REPL calls `run` once per objective, so the control state is per objective: one
    `interrupt` ends the objective it was typed into and the session goes on. Left set, every
    later objective drains and stops before the backend is ever asked, which is a session that
    looks alive and answers nothing."""
    agent, backend = build([model.turn_text("one"), model.turn_text("two"),
                            model.turn_text("three")], tmp_path)
    agent.run("first")
    agent.submit("interrupt", issuer="operator")
    agent.run("second")
    agent.run("third")
    assert [s["reason"] for s in agent.log.of_kind("ended")] == \
        ["complete", "interrupted", "complete"]
    assert len(backend.calls) == 2, \
        "two: the interrupted objective stops at the turn boundary without asking, the third asks"


def test_a_later_objective_does_not_return_an_earlier_answer(tmp_path):
    """`run` returns this objective's answer. An objective that produced no prose has no answer,
    and handing back the previous objective's tells the caller — or a parent reading a child —
    something that never happened in this run."""
    agent, _ = build([model.turn_text("answer one")], tmp_path)
    assert agent.run("first") == "answer one"
    agent.submit("interrupt", issuer="operator")
    assert agent.run("second") == ""


def test_a_spent_nudge_does_not_count_against_the_next_objective(tmp_path):
    """A nudge is spent on a streak, and a streak belongs to one objective. Two spent while the
    first objective was hammering a tool must not make the second objective's first streak a
    stop: it has not been given the chance to be nudged yet."""
    turn = [model.ToolCall("c", "echo", {"text": "x"}), model.Stopped("tool_calls")]
    agent, _ = build([turn] * 60, tmp_path, max_turns=30)
    agent.run("first")
    assert one(agent, "ended")["reason"] == "stalled"
    spent = len(agent.log.steps())
    agent.run("second")
    later = [s for s in agent.log.steps()[spent:]
             if s["kind"] == "notice" and s["subject"] == "nudge"]
    assert any("nudged to change approach" in s["text"] for s in later), \
        "a fresh streak in a fresh objective earns a nudge before it earns a stop"


# --- the spill name is the log's to mint, not the backend's -----------------------------------

def _spewing_agent(root, ids):
    """One agent that spills once per id in `ids`, then finishes. Those ids are what the *server*
    claims, which is exactly the input a spill path must not put into a filename."""
    turns = [[model.ToolCall(i, "spew", {"path": "x"}), model.Stopped("tool_calls")] for i in ids]
    turns.append([model.TextDelta("done"), model.Stopped("stop")])
    return Agent(backend=model.Fake(turns, caps=model.Caps(context_tokens=100, model="t")),
                 tools={"spew": spew}, log=Log(root / "log.jsonl"), policy=ALLOW,
                 max_turns=6, run_id="r1", root=root, attended=True)


def test_a_hostile_call_id_cannot_move_a_spill_outside_the_run(tmp_path):
    r"""The spill name used to be built out of whatever id the server streamed, so the server chose
    the path. Measured against the pre-fix code on Windows: `..\..\..\..\evil` wrote one level above
    the run root (Windows resolves `result-\..` lexically, so four levels of `..` land one above),
    and `..\..\pwned` chose a name inside `out/`. On POSIX those same ids make `open` raise, and the
    raise happened inside the bound, which sits outside the tool's `try`, so the run died mid-flight
    with no `ended` step. Naming the spill after the step id the log minted takes the backend out of
    the path on either platform, and the bound no longer raises at all -- which is what the `ended`
    assertion below is really guarding."""
    for index, hostile in enumerate((r"..\..\..\..\evil", "../../x", r"..\..\pwned")):
        root = tmp_path / f"case{index}"
        root.mkdir()
        agent = _spewing_agent(root, [hostile])
        agent.run("go")
        assert one(agent, "ended")["reason"] == "complete", f"{hostile!r} ended the run"
        call_id = [s for s in agent.log.steps() if s["kind"] == "call"][0]["id"]
        written = sorted(p.name for p in (root / "out").iterdir())
        assert written == [f"{call_id}.txt"], f"{hostile!r} chose the file name: {written}"
    assert not [p for p in tmp_path.iterdir() if p.is_file()], "a payload wrote beside the run dirs"


def test_two_calls_that_report_the_same_id_spill_to_two_files(tmp_path):
    """Two spills under one name was the quiet failure: the later write replaced the earlier file,
    and the older result's `spilled` pointer went on naming content that no longer existed. The
    backend need not be hostile for this -- the Fake omits ids, and real servers reuse `call_0` --
    which is why the name comes from the log's sequence rather than from the protocol."""
    root = tmp_path / "two"
    root.mkdir()
    agent = _spewing_agent(root, ["", ""])
    agent.run("go")
    written = sorted(p.name for p in (root / "out").iterdir())
    assert len(written) == 2 and len(set(written)) == 2, written
    pointers = [s["spilled"] for s in agent.log.steps()
                if s["kind"] == "result" and "spilled" in s]
    assert len(set(pointers)) == 2, f"two results, one file: {pointers}"
    assert all(os.path.isfile(p) for p in pointers), pointers


def test_a_spill_that_cannot_be_written_costs_the_run_nothing(tmp_path):
    """Losing the whole output is a real loss; losing the run is not allowed, and one OSError from
    the bound used to do exactly that. The model has to be told the preview is all there is, rather
    than being sent to read a file that was never written."""
    (tmp_path / "out").write_text("not a directory", encoding="utf-8")   # makedirs must fail
    agent = _spewing_agent(tmp_path, ["c"])
    agent.run("go")
    assert one(agent, "ended")["reason"] == "complete"
    res = one(agent, "result")
    assert res.get("truncated") is True, "the result was over the cap either way"
    assert "could not be saved" in res["content"], res["content"]
    assert "full output saved to" not in res["content"], "the model was sent to a missing file"
    assert "spilled" not in res, "the log must not point at a file that does not exist"


def test_a_call_with_the_wrong_argument_type_is_a_verdict_and_the_tool_never_runs(tmp_path):
    """`fs.read(path=0)` is the reason this exists: `open(0)` is stdin, so the model would have read
    the operator's terminal input with no grant held and no path involved — and the veto point would
    have been asked to judge a location the tool never touched. So the type check runs before the
    policy, the tool is never entered, and the refusal is what the model reads back."""
    touched = []

    def peek(path: str) -> str:
        """Read a path."""
        touched.append(path)
        return "contents"

    peek.access, peek.address = "read", "path"
    peek.description = "Read a path."

    turns = [[model.ToolCall("c", "peek", {"path": 0}), model.Stopped("tool_calls")],
             [model.TextDelta("done"), model.Stopped("stop")]]
    agent = Agent(backend=model.Fake(turns, caps=model.Caps(context_tokens=100, model="t")),
                  tools={"peek": peek}, log=Log(tmp_path / "log.jsonl"), policy=ALLOW,
                  max_turns=6, run_id="r1", root=tmp_path, attended=True)
    agent.run("go")
    assert touched == [], "the tool ran with an argument it was never told it could receive"
    operation = one(agent, "operation")
    assert operation["outcome"] == "refused" and "must be a string" in operation["reason"]
    assert one(agent, "ended")["reason"] == "complete", "a bad argument is not a fatal one"
    tool_msg = next(m for m in agent.messages if m.role == "tool")
    assert "must be a string" in tool_msg.content, "the model was not told what to send instead"


def test_a_signal_while_a_tool_streams_still_leaves_an_ended_step(tmp_path):
    """The log has to say the run stopped, whatever stopped it.

    `recall` and `child_status` ask the record whether a run is over. Without an `ended` step a run cut
    short by a signal is reported as still running, forever, and the exit code a caller sees has no
    source. The graceful stop an operator queues already writes this; a signal must not be the one path
    that leaves the log mid-sentence."""
    def boom(text: str) -> str:
        """The signal arrives while the tool is running."""
        raise KeyboardInterrupt

    agent, _ = build([model.turn_tool("c1", "boom", {"text": "x"})], tmp_path, tools={"boom": boom})
    with pytest.raises(KeyboardInterrupt):
        agent.run("go")
    ended = one(agent, "ended")
    assert ended["reason"] == "interrupted"
    assert ended["exit_code"] == 128 + 2, "a signal is 128+SIGINT, not the clean stop an operator queued"


def test_a_consumer_that_raises_while_pumping_closes_the_tool_stream(tmp_path):
    """A stream the pump stopped pumping is closed, not left suspended.

    The display bus deliberately isolates its subscribers, so a renderer cannot abort a run; the
    interruption that matters is one raised in the pump itself -- a signal landing between chunks, or a
    defect in the code that wraps each chunk. Either way the generator is left parked at its `yield`,
    which for the one streaming tool means a command running with nobody watching it. `bash` does its
    cleanup at close, and a cleanup that waits for the collector to notice is not one an operator can
    rely on."""
    closed = {"ran": False}

    class Fatal:
        """A chunk the pump cannot process: stands in for the signal arriving mid-pump."""

        def __str__(self):
            raise KeyboardInterrupt

    def flood(text: str):
        """One ordinary chunk, then one the pump cannot wrap, and a `finally` that says whether anyone
        closed me."""
        try:
            yield "one\n"
            yield Fatal()
        finally:
            closed["ran"] = True

    agent, _ = build([model.turn_tool("c1", "flood", {"text": "x"}), model.turn_text("unreached")],
                     tmp_path, tools={"flood": flood})
    with pytest.raises(KeyboardInterrupt) as caught:
        agent.run("go")
    # Asserted while the traceback is still held, on purpose. A traceback keeps the pump's frame alive,
    # and that frame references the stream, so nothing can free the generator while it is read: the
    # fallback of "the collector closes it" is exactly what cannot happen here, and a caller that stores
    # the exception (a log line, an error report, `sys.last_traceback`) hits the same thing. A first
    # version of this test dropped the traceback and passed with the close removed, which is how it was
    # learned that the assertion had to pin the reference.
    assert caught.value.__traceback__ is not None, "the traceback, and so the pump's frame, is alive"
    assert closed["ran"], "the abandoned stream was closed, so the tool's own cleanup ran"


def test_a_run_with_no_opinion_records_the_word_not_the_absence(tmp_path):
    """The loop's "no preference" is `None`, but the log field is an ident: a reader enumerating old runs
    asks which posture a run had, and a missing field cannot be told apart from a run that recorded no
    posture at all. The record therefore names the harness's own word, which is also what the operator
    would have typed.

    This path existed before any test drove it: every battery build used the `"off"` default, so a
    `None` level reaching `started` was reached only by a real `--think none` run -- and it raised
    `Refused: started.thinking must be ident, got NoneType` in the middle of a live run.
    """
    log = Log(tmp_path / "log.jsonl")
    backend = model.Fake([model.turn_text("done")], caps=model.Caps(context_tokens=100, model="t"))
    agent = Agent(backend=backend, tools={}, log=log, policy=ALLOW,
                  thinking=None, run_id="r1", root=tmp_path)
    agent.run("go")
    started = [s for s in log.steps() if s["kind"] == "started"][-1]
    assert started["thinking"] == "none", started


def test_the_transport_knobs_are_stamped_only_when_the_operator_turned_them_off(tmp_path):
    """`started.streaming` and `started.usage` are stamped the way `max_runtime` is: absent at their
    defaults (streaming, usage requested), present only on the deviation. A grep for
    `\"streaming\": false` then finds exactly the runs that talked to the server in the other shape,
    without a field every ordinary run carries."""
    log = Log(tmp_path / "log.jsonl")
    agent = Agent(backend=model.Fake([model.turn_text("done")],
                                     caps=model.Caps(context_tokens=100, model="t")),
                  tools={}, log=log, policy=ALLOW, run_id="r1", root=tmp_path)
    agent.run("go")
    started = [s for s in log.steps() if s["kind"] == "started"][-1]
    assert "streaming" not in started and "usage" not in started, started

    log2 = Log(tmp_path / "log2.jsonl")
    agent2 = Agent(backend=model.Fake([model.turn_text("done")],
                                      caps=model.Caps(context_tokens=100, model="t")),
                   tools={}, log=log2, policy=ALLOW, run_id="r2", root=tmp_path,
                   streaming=False, usage=False)
    agent2.run("go")
    started2 = [s for s in log2.steps() if s["kind"] == "started"][-1]
    assert started2["streaming"] is False and started2["usage"] is False, started2


def test_the_reasoning_wire_knobs_are_stamped_only_when_the_operator_named_them(tmp_path):
    """`started.reasoning_out` and `started.require_reasoning_content` follow the `streaming`
    idiom: absent at the defaults (the mirror, no pin), present only on the deviation, so a reader
    of the wire trail can tell why a request spoke the server's other shape."""
    log = Log(tmp_path / "log.jsonl")
    agent = Agent(backend=model.Fake([model.turn_text("done")],
                                     caps=model.Caps(context_tokens=100, model="t")),
                  tools={}, log=log, policy=ALLOW, run_id="r1", root=tmp_path)
    agent.run("go")
    started = [s for s in log.steps() if s["kind"] == "started"][-1]
    assert "reasoning_out" not in started and "require_reasoning_content" not in started, started

    log2 = Log(tmp_path / "log2.jsonl")
    agent2 = Agent(backend=model.Fake([model.turn_text("done")],
                                      caps=model.Caps(context_tokens=100, model="t")),
                   tools={}, log=log2, policy=ALLOW, run_id="r2", root=tmp_path,
                   require_reasoning_content=True, reasoning_out="reasoning")
    agent2.run("go")
    started2 = [s for s in log2.steps() if s["kind"] == "started"][-1]
    assert started2["require_reasoning_content"] is True, started2
    assert started2["reasoning_out"] == "reasoning", started2


# --- CMP-1: the one situation compaction exists for used to end the run ---

# Measured against a live llama.cpp server (127.0.0.1:8081), not invented: this is the body a real
# server returns for a request over its window, wrapped in the shape `openai.OpenAI.complete` puts
# around an HTTPError. The OpenAI dialect's wording ("maximum context length", "too long") appears
# nowhere in this text — measured against a live llama.cpp server — which is why the detector's markers
# are split into measured and inferred in `sah/compact.py` rather than guessed at.
OVERFLOW = ('transport: HTTPError (HTTP 400): Bad Request [server said: {"error":{"code":400,'
            '"message":"request (200013 tokens) exceeds the available context size (16128 tokens), '
            'try increasing it","type":"exceed_context_size_error","n_ctx":16128}}]')


def fill(agent, n: int = 8) -> None:
    """Give the conversation enough turns that compaction has something it is allowed to drop.

    Deliberately short. `KEEP_RECENT` is six, so a kept tail of six small notes sits far below the
    trigger beside the resident parts: a test that means to be about something else must not have
    the kept-tail guard fire on it."""
    for i in range(n):
        agent.say("user", f"note {i}", issuer="core")


def pad(agent, n: int = 240) -> None:
    """Fill the dropped zone so the canonical room arithmetic has something to summarise.

    `fill` is too small for the emergency path these tests drive: the room is the limit minus the
    resident load (the tool surface), the head and the backfill, and a handful of notes leaves no
    positive room — the compaction would take its failed note for want of space, not of the server.
    The notes are short on purpose: the backfill that always survives stays small, and it is the
    dropped zone that must be long enough for the summary's room to exist."""
    for i in range(n):
        agent.say("user", f"note {i}", issuer="core")


def test_a_request_refused_as_too_large_makes_room_and_retries_the_turn_once(tmp_path):
    """A server refusing a request for its size is the single event compaction was built for, and until
    now it ended the run: `step()` recorded the failure, `run()` broke on it, and the check that would
    have summarised the conversation sat after the break. The one emergency the mechanism exists to meet
    was the one it never handled."""
    agent, backend = build([[model.Failed(OVERFLOW)],
                            [model.TextDelta("we read the config and fixed the port"),
                             model.Stopped("stop")],
                            [model.TextDelta("recovered"), model.Stopped("stop")]], tmp_path)
    pad(agent)
    agent.run("carry on")
    assert one(agent, "ended")["reason"] == "complete"

    summaries = many(agent, "summary")
    assert len(summaries) == 1, "the run made room exactly once"
    assert summaries[0]["outcome"] == "ok"
    assert last(agent, "message")["text"] == "recovered"


def test_the_retry_asks_with_the_conversation_that_was_made_smaller(tmp_path):
    """The retry has to be a *different request*. Retrying the identical request after a refusal for its
    size would be refused identically, and a test that only asserted the run completed would pass while
    the harness burned a second refusal — so this reads the requests the backend actually received."""
    agent, backend = build([[model.Failed(OVERFLOW)],
                            [model.TextDelta("notes"), model.Stopped("stop")],
                            [model.TextDelta("recovered"), model.Stopped("stop")]], tmp_path)
    pad(agent)
    agent.run("carry on")

    refused, summary, retry = backend.calls
    assert len(retry.messages) < len(refused.messages), "the retry carried the same conversation"
    assert any(m.role == "user" and "<summary>" in m.content for m in retry.messages), \
        "no bounded summary record"
    assert len(summary.messages) < len(refused.messages), "the summary request was not bounded either"


def test_a_context_refusal_is_retried_once_and_never_twice(tmp_path):
    """The single-retry rule, stated as a bound rather than as intent. Without it a server whose window
    is simply too small for the task makes the run compact, retry, get refused, compact, retry, until
    the turn budget runs out — each compaction throwing away conversation, on a run that was never
    going to finish. Ending `failed` with one summary is the honest outcome."""
    agent, backend = build([[model.Failed(OVERFLOW)],
                            [model.TextDelta("notes"), model.Stopped("stop")],
                            [model.Failed(OVERFLOW)]], tmp_path, max_turns=8)
    pad(agent)
    agent.run("carry on")
    assert one(agent, "ended")["reason"] == "failed"
    assert len(many(agent, "summary")) == 1
    assert len(backend.calls) == 3, "the run kept asking after the second refusal"


def test_a_refusal_that_is_not_about_size_compacts_nothing(tmp_path):
    """The other half of the rule, and the more dangerous half. A network blip is not a full context; if
    the detector matched loosely, every dropped connection would quietly summarise the run's history to
    nothing, and the log would read as though the agent chose to forget. Compaction loses information,
    so it is only ever reached for a reason that names the loss it can fix."""
    agent, backend = build([[model.Failed("transport: ConnectionResetError: connection reset by peer")]],
                           tmp_path)
    fill(agent)
    agent.run("carry on")
    assert one(agent, "ended")["reason"] == "failed"
    assert many(agent, "summary") == []
    assert len(backend.calls) == 1, "the run retried a request that never reached a server"


def test_the_notice_carries_the_servers_own_words(tmp_path):
    """Why the run spent a request writing notes instead of answering: a compaction on a run that ends
    in one turn looks like a stall unless the record says what the server said. The server's text is the
    diagnosis — window size and all — so it goes in the notice rather than a fixed phrase."""
    agent, _ = build([[model.Failed(OVERFLOW)],
                      [model.TextDelta("notes"), model.Stopped("stop")],
                      [model.TextDelta("recovered"), model.Stopped("stop")]], tmp_path)
    pad(agent)
    agent.run("carry on")

    notices = [s for s in agent.log.steps() if s["kind"] == "notice" and s["subject"] == "compact"]
    assert "exceeds the available context size" in notices[0]["text"]
    assert len(notices) == 2, \
        "the server-words notice and the act's notice, in that order: " \
        f"{[s['text'] for s in notices]}"


def test_the_fit_check_refuses_before_spending_a_summary_call(tmp_path):
    """The cut rung reaches the tail, but it cannot cut the resident load: the system prompt and
    the tool surface travel on every request, so a window smaller than the resident load and the
    slot’s smallest text holds no compaction at all, whatever the tail is cut to.

    So this is still refused before anything is asked: the fit check runs before any summary
    call, the cut rung empties the backfill first — its work is in the record, so a reader sees
    the rung ran and still could not fit — and only then does the run end `blocked`: the
    declared ending for a window that cannot hold what every request carries, with the numbers
    an operator can act on (a bigger window, a smaller resident load) rather than a doom loop.
    """
    agent, backend = build([[model.TextDelta("working"), model.Usage(10_000, 5),
                             model.Stopped("stop")],
                            [model.TextDelta("working again"), model.Usage(10_000, 5),
                             model.Stopped("stop")]], tmp_path, max_turns=2, context=400)
    for i in range(8):
        agent.say("user", f"small note {i}", issuer="core")
    agent.messages.append(model.Message("tool", "RESULT " + "x" * 60_000))
    agent._trace.append("")
    agent.run("go on")

    assert many(agent, "summary") == [], "the run paid for notes that could not have made room"
    assert len(backend.calls) == 1, "a summarisation request was sent anyway"
    ended = one(agent, "ended")
    assert ended["reason"] == "blocked" and ended["exit_code"] == 1,\
        "a window that cannot hold the resident load is a declared ending, not a failure"
    notices = [s for s in agent.log.steps() if s["kind"] == "notice" and s["subject"] == "compact"]
    assert notices, "the refusal to compact was not recorded"
    text = notices[0]["text"]
    assert "window 400" in text, text
    assert "resident" in text and "tool schemas" in text,\
        "the resident load and its tool component are not in the record"
    assert "cut to 0 messages" in text, "the cut rung’s work is not in the record"
    assert len(agent.messages) == 11,\
        "the conversation was rewritten anyway: 8 notes + result + objective + answer"


def test_a_backfill_too_big_for_the_limit_is_cut_and_the_run_continues(tmp_path):
    """The other side of the fit check: when the backfill itself is the window — a 60,000-character
    result inside the last six messages — the last rung reaches the tail. The oldest backfill
    messages, in whole units, are dropped until the tail fits the limit; the record names the cut
    (backfill_cut on the summary step, the budget line beside it); the frame’s note tells the
    model the gap is a gap; and the run continues, which is the whole job of the rung. The blocked
    ending is left for the window that cannot hold the resident load (the test above), not for a
    tail a cut can make fit — measured on a run that died `blocked` over exactly this shape on a
    window it had not been told the size of.
    """
    agent, backend = build([[model.TextDelta("working"), model.Usage(10_000, 5),
                             model.Stopped("stop")],
                            [model.TextDelta("the summary"), model.Usage(200, 5),
                             model.Stopped("stop")],
                            [model.TextDelta("done"), model.Usage(900, 5),
                             model.Stopped("stop")]], tmp_path, max_turns=2)
    for i in range(8):
        agent.say("user", f"small note {i}", issuer="core")
    agent.messages.append(model.Message("tool", "RESULT " + "x" * 60_000))
    agent._trace.append("")
    agent.run("go on")

    ended = one(agent, "ended")
    assert ended["reason"] == "complete" and ended["exit_code"] == 0,\
        "the cut rung exists so this run does not end blocked"
    step = one(agent, "summary")
    assert step["backfill_cut"] == 4, step["budget"]
    assert step["kept"] == 2, "the surviving tail is the objective and the turn in progress"
    assert step["stage"] == 1 and step["slot"] == "summary", step
    assert "backfill cut by 4 messages" in step["budget"], step["budget"]
    # The model is told the gap is a gap: the frame’s cut note beside the backfill’s note.
    frame = agent.messages[0].content
    assert "working on potentially missing information" in frame, frame
    assert "The next 2 messages are kept verbatim" in frame, frame
    # The list is the slot and the surviving tail, and the next turn ran against it.
    assert [m.content for m in agent.messages[1:3]] == ["go on", "working"]
    assert len(agent.messages) == 4, "slot + 2 backfill + the turn that ran against them"
    # A resume rebuilds the same list: the replay’s arithmetic skips the cut tail.
    fresh = Agent(backend=model.Fake([], caps=model.Caps(context_tokens=1000, model="test-1")),
                  tools={}, log=agent.log, root=tmp_path, policy=ALLOW)
    fresh.replay()
    assert [m.content for m in fresh.messages] == [m.content for m in agent.messages],\
        "the cut tail came back on resume, or the kept tail did not"


def test_an_ordinary_run_still_compacts_when_the_tail_is_small(tmp_path):
    """The control, and the reason the guard above is a guard rather than a switch: the ordinary case is
    a kept tail of a few kilobytes and older turns worth dropping. A guard that always refused would
    pass the test above and leave every long run to overflow."""
    agent, backend = build([[model.TextDelta("answer"), model.Usage(900, 5), model.Stopped("stop")],
                            [model.TextDelta("next"), model.Usage(900, 5), model.Stopped("stop")]],
                           tmp_path, max_turns=2)
    for i in range(8):
        agent.say("user", f"small note {i}", issuer="core")
    agent.run("go on")

    assert len(many(agent, "summary")) == 1
    assert many(agent, "summary")[0]["outcome"] == "ok"


def test_a_compaction_that_runs_tells_the_model_and_the_record(tmp_path):
    """The act is told the way a nudge is told: a core message the model sees in its own
    conversation, because the conversation is the only place it can be told its context changed
    under it — and the same text in the record, so every rendered surface (the log, the report,
    the live stream) shows it. The wording carries no token numbers: a number in model-facing
    text is a budget the model can act on — measured in a pi run, a model told its context sat
    at 80% ended the run itself to "take a break" — so the act names the event, and the
    operator's numbers stay where the arithmetic is: the step's budget field and the usage the
    turns around the act report."""
    agent, _ = build([[model.TextDelta("answer"), model.Usage(900, 5), model.Stopped("stop")],
                     [model.TextDelta("next"), model.Usage(900, 5), model.Stopped("stop")]],
                    tmp_path, max_turns=2)
    for i in range(8):
        agent.say("user", f"small note {i}", issuer="core")
    agent.run("go on")

    notices = [s for s in agent.log.steps()
               if s["kind"] == "notice" and s.get("subject") == "compact"
               and "compaction occurred" in s.get("text", "")]
    assert len(notices) == 1, \
        f"the act notice is missing or doubled: {[s['text'] for s in notices]}"
    act = notices[0]["text"]
    assert not any(c.isdigit() for c in act), \
        f"a number reached the model-facing act text: {act!r}"
    assert agent.messages[-1].content == act, \
        "the model's conversation does not end on the act's word"
    steps = agent.log.steps()
    summary_index = next(i for i, s in enumerate(steps) if s["kind"] == "summary")
    act_index = next(i for i, s in enumerate(steps)
                     if s["kind"] == "message" and s.get("issuer") == "core"
                     and s.get("text") == act)
    assert act_index > summary_index, "the act's word must come after the act's record"
    assert "pre ~" in one(agent, "summary")["budget"], \
        "the operator's numbers left the record: the budget field is where the arithmetic is"


def test_a_refusal_is_retried_with_the_failed_note_when_the_summary_failed(tmp_path):
    """The slot is always filled, so a compaction whose attempts all failed still rewrites the
    conversation — head, the failed note, the backfill — and the retry is a different request: the
    drops are out of it and the note says what is missing. The old rule (no retry, "the request
    would be identical") no longer holds, because a compaction that ran always shrinks the
    conversation; what still binds is the single-compaction-per-refusal rule, and what the record
    must carry is the verdict with the attempts' words, beside a conversation the model can still
    answer from."""
    agent, backend = build([[model.Failed(OVERFLOW)],
                            [model.Failed("the summary engine died")],
                            [model.Failed("the summary engine died")],
                            [model.TextDelta("gone"), model.Stopped("stop")]], tmp_path, max_turns=8)
    pad(agent)
    agent.run("carry on")

    assert one(agent, "ended")["reason"] == "complete", \
        "the retry carried a smaller conversation with the failed note; a server that takes it completes"
    step = one(agent, "summary")
    assert step["outcome"] == "failed" and step["slot"] == "failed", step
    assert step["reason"].count("the summary engine died") == 2, \
        f"the record does not carry the attempts' words: {step['reason']!r}"
    assert len(backend.calls) == 4, \
        f"one turn, two failed attempts and the retry — {len(backend.calls)} calls"
    retry = backend.calls[-1]
    assert any(m.role == "user" and "earlier turns are not in this context" in str(m.content)
               for m in retry.messages), "the failed note is not where the model would see it"
    assert len(retry.messages) < len(backend.calls[0].messages), \
        "the retry carried the same conversation the server refused"


# --- WEB-3: a fetched page the model cannot reach is a page it does not have -------------------

def web_agent(tmp_path, page: bytes, monkeypatch, *, ctype="text/html; charset=utf-8"):
    """An agent whose real `web_fetch` is wired to a canned page, and whose one turn calls it.

    The composition is the point: `web` no longer clips and `agent` spills, so neither module's own test
    can see the property that matters — that the end of the page still exists somewhere the run can go."""
    from sah.tools import web
    net = web.Net()
    monkeypatch.setattr(net, "_get", lambda url, **k: (page, ctype, url))
    tools = {t.__name__: t for t in web.extension(net).tools}
    agent, _ = build([[model.ToolCall("c1", "web_fetch", {"url": "http://8.8.8.8/big"}),
                       model.TextDelta("read it"), model.Stopped("stop")]], tmp_path, tools=tools)
    return agent


def test_the_end_of_a_large_page_survives_in_a_file_the_run_can_page(tmp_path, monkeypatch):
    """Pre-clipped, the tail of a big page existed nowhere: not in the context, not on disk, and no tool
    could get it back, because re-fetching is not the same bytes. Spilling the whole page instead costs a
    file inside the run's own directory, which the run already owns and can read with `offset`.

    The assertion walks the whole spill rather than trusting the preview, because the failure this
    guards is a preview that looks fine."""
    page = ("<html><body>" + "".join(f"<p>paragraph {i} fills the page out</p>" for i in range(6000))
            + "<p>THE LAST PARAGRAPH</p></body></html>").encode("utf-8")
    agent = web_agent(tmp_path, page, monkeypatch)
    agent.run("read the page")

    res = last(agent, "result")
    assert res.get("spilled"), "the result was truncated with nowhere left to go"
    spilled = pathlib.Path(res["spilled"])
    assert spilled.is_relative_to(tmp_path.resolve()), "the spill left the run's own directory"
    assert "THE LAST PARAGRAPH" in spilled.read_text(encoding="utf-8"), "the tail was not in the spill"
    assert "THE LAST PARAGRAPH" not in res["content"], "the whole page went into the context"
    assert "read it with offset" in res["content"], "the preview did not name its continuation"


def test_a_page_that_fits_is_neither_spilled_nor_declared_lossy(tmp_path, monkeypatch):
    """The control. A spill for every fetch would bury the run's directory in strangers' pages and make
    the pointer noise; the guard is only correct if the ordinary small page still comes back whole."""
    agent = web_agent(tmp_path, b"<html><body><p>short</p></body></html>", monkeypatch)
    agent.run("read the page")
    res = last(agent, "result")
    assert "spilled" not in res and not res.get("truncated")
    assert "short" in res["content"]


def test_the_end_of_a_one_line_page_is_reached_using_only_the_pointers_given(tmp_path, monkeypatch):
    """WEB-3 settled, in the only form that means anything: start at the top of a spilling fetch and walk
    to its end using nothing but the hints the results hand back, with no shell and no extra permission.

    The fixture is a page with no newlines at all (a minified script, a one-line JSON blob), which is the
    worst case: the byte cap stops inside a single line, and paging there needs `start_char`, not
    `offset`. A run told to read a page it cannot finish is the failure this finding was about, and a
    preview that says "read it with offset" would have been a pointer to an argument that keeps refusing.
    """
    page = b"x" * (ingest.MAX_BYTES * 3) + b"THE END OF THE PAGE"
    agent = web_agent(tmp_path, page, monkeypatch, ctype="text/plain; charset=utf-8")
    agent.run("read the page")

    res = last(agent, "result")
    assert res.get("spilled"), "a page this large must spill"
    assert "THE END OF THE PAGE" not in res["content"], "the whole page went into the context"

    seen, offset, start = "", 1, 0
    for _ in range(12):
        got = fs.read(path=res["spilled"], offset=offset, start_char=start)
        if "THE END OF THE PAGE" in got:
            break
        nxt_off, nxt_char = re.search(r"offset=(\d+)", got), re.search(r"start_char=(\d+)", got)
        if not (nxt_off or nxt_char):
            break                                   # no pointer left: the walk has dead-ended
        if nxt_char:
            start = int(nxt_char.group(1))
        if nxt_off:
            offset = int(nxt_off.group(1))
        assert (offset, start) != (int(nxt_off.group(1)) if nxt_off else offset, start) \
            or nxt_char, "a pointer that names where it already is"
        seen += got
    else:
        raise AssertionError("the walk ran out of steps before the end of the page")

    assert "THE END OF THE PAGE" in got, "the tail was unreachable: the pointers dead-ended"


def test_an_assistant_turn_carries_its_reasoning_into_the_next_request(tmp_path):
    """The reasoning of a turn is recorded in the log and streamed to the terminal, and none of that puts
    it back in front of the model. An interleaved-thinking server answers a tool call expecting its own
    reasoning to still be in the conversation; without it the next turn re-derives from nothing and the
    model contradicts itself across a tool boundary.

    Carrying it here is a fact about the conversation, not about transport: whether it is *sent* is the
    adapter's decision, taken per server dialect, and asserted in the adapter's own tests."""
    from sah.model import ReasoningDelta, Stopped, TextDelta, ToolCall
    agent, _ = build([[ReasoningDelta("the path is relative, so resolve it first"),
                       TextDelta("working"), ToolCall("c1", "read", {"path": "x"}),
                       Stopped("tool_calls")],
                      [ReasoningDelta("the file said what I expected"), TextDelta("done"),
                       Stopped("stop")]], tmp_path)
    agent.run("read x")
    turns = [m for m in agent.messages if m.role == "assistant"]
    assert len(turns) == 2, turns
    # Both branches of the loop's assistant append: the turn that called a tool, and the turn that only
    # spoke. Covering one and not the other is how half an invariant stays true.
    assert turns[0].reasoning == "the path is relative, so resolve it first", turns[0]
    assert turns[1].reasoning == "the file said what I expected", turns[1]


def test_a_failed_compaction_records_the_verdict_and_the_reason_apart(tmp_path):
    """The loop is the link between the stages and the record.

    The stages return a verdict and a sentence, and only the loop files the sentence in the step's `reason`
    and the slot's kind in `slot`. A test on `summarise` cannot see that (it stops at the tuple) and a
    test on the projections cannot either (they are handed a step), so a mutant that dropped the filing
    survived both, which is what this test is for. Every stage is scripted to fail here: a run's own
    refusal is one event and each stage consumes its own response, so one failure only ever takes out
    the first stage.

    The record then has to carry both facts as fields: that the slot holds the failed note, and what
    the attempts cost, in their words."""
    agent, _ = build([[model.Failed(OVERFLOW)],
                      [model.Failed("the summary engine died")],
                      [model.Failed("the summary engine died")],
                      [model.TextDelta("carried on"), model.Stopped("stop")]], tmp_path)
    pad(agent)
    agent.run("carry on")

    step = one(agent, "summary")
    assert step["outcome"] == "failed" and step["slot"] == "failed" and step["stage"] == 4, step
    assert step["reason"].count("the summary engine died") == 2, \
        f"the verdict was recorded and the attempts' words were not: {step}"
    assert "carried on" in agent.final_text, \
        "the retry carried the failed note and the backfill; the run still reaches its answer"


def test_the_summary_arrives_as_a_bounded_record_not_an_instruction(tmp_path):
    """A summary is a model's paraphrase of the operator's own earlier requests, so its frame has to say what
    it is and where it ends.

    Two failures are guarded here. Unbounded quoted prose reads as the last thing the operator said, and a
    run will then obey a paraphrase it should only remember; and if the frame ever moved into the system
    prompt, that paraphrase would gain instruction authority, which is why the role is asserted alongside the
    shape."""
    agent, backend = build([[model.Failed(OVERFLOW)],
                            [model.TextDelta("the older turns, in order"), model.Stopped("stop")],
                            [model.TextDelta("recovered"), model.Stopped("stop")]], tmp_path)
    pad(agent)
    agent.run("carry on")

    sent = [m for m in backend.calls[-1].messages if "the older turns" in str(m.content)]
    assert len(sent) == 1, f"the summary was not the single framed message I expected: {sent}"
    summary = sent[0]
    assert summary.role == "user", "a summary is a record; the system block is where authority lives"
    assert "compact" in summary.content.lower(), summary.content
    assert summary.content.count("<summary>") == 1 and summary.content.count("</summary>") == 1, \
        summary.content
    assert (summary.content.index("<summary>") < summary.content.index("the older turns")
            < summary.content.index("</summary>")), summary.content


def test_the_summary_frame_is_an_editable_file_not_a_python_constant():
    """Model-facing prose lives in `sah/prompts`, so the check is that the frame *comes* from there."""
    assert compact.frame("BODY") == prompts.load("summary_frame").replace("{{summary}}", "BODY")
    assert "BODY" in compact.frame("BODY") and "{{summary}}" not in compact.frame("BODY")


def test_a_summary_that_quotes_an_insert_is_not_resolved_as_one():
    """A summary is text a model wrote, and whatever it quotes must reach the server as itself.

    This pins the outcome, not the mechanism, and the difference earned its keep: the mutant that passed the
    body through the template as an override *survived* this test, because the renderer substitutes in a
    single pass and would not have resolved the braces either way — measured directly. Two implementations
    that both satisfy a guarantee is still the right test, since the guarantee is what a caller cares about;
    what is not allowed is the comment beside the code claiming a mechanism it never demonstrated."""
    framed = compact.frame("the operator asked about {{datetime:%Y}} and {{sys.os}}")
    assert "{{datetime:%Y}}" in framed and "{{sys.os}}" in framed, framed


def test_a_picture_is_attached_to_the_answer_and_only_its_path_is_recorded(tmp_path, picture):
    """A picture the model asked for has to arrive as a picture. The record is a different object: a
    base64 blob in the log is a megabyte every later reader pays for, and the path is the fact the log
    can hold — it is also what a resume re-reads, which is exactly what a resumed text read already
    does."""
    import base64
    agent, _ = build([model.turn_tool("c1", "read", {"path": str(picture)}), model.turn_text("seen")],
                     tmp_path, tools={"read": fs.read}, vision=True)
    agent.run("look at the picture")
    assert one(agent, "result")["images"] == [str(picture)], one(agent, "result")
    assert "base64" not in (tmp_path / "log.jsonl").read_text(encoding="utf-8"), "the blob reached the log"
    answer = [m for m in agent.messages if m.role == "tool"][-1]
    urls = [p["image_url"]["url"] for p in (answer.content if isinstance(answer.content, list) else [])
            if "image_url" in p]
    assert len(urls) == 1 and urls[0].startswith("data:image/png;base64,"), answer.content
    assert base64.b64decode(urls[0].split(",", 1)[1]) == picture.read_bytes()


def test_a_picture_is_not_offered_without_the_vision_declaration(tmp_path, picture):
    """The authority on whether a picture may travel is the flag the operator typed, not something the
    server reports about itself — and the answer the model reads must name the act that would have
    permitted the call, like every other refusal does.

    Measured on a box where a run reached the raw CLI past its front door, so the door's `--vision` was
    never passed: the model was told "this endpoint declares no vision" about a server that answers a
    picture in a tool result with the right colour. The sentence blamed a server for a flag nobody typed,
    and the model has no way to type one."""
    agent, _ = build([model.turn_tool("c1", "read", {"path": str(picture)}), model.turn_text("seen")],
                     tmp_path, tools={"read": fs.read})
    agent.run("look at the picture")
    step = one(agent, "result")
    assert not step.get("images"), step
    assert "--vision" in step["content"], step["content"]
    assert "endpoint" not in step["content"], f"a flag was refused, a server was blamed: {step['content']}"
    answer = [m for m in agent.messages if m.role == "tool"][-1]
    assert isinstance(answer.content, str), answer.content


def _replayed(tmp_path, vision):
    """A fresh Agent over the log the test run just wrote, which is what a resume is."""
    return Agent(backend=model.Fake([], caps=model.Caps(context_tokens=100, model="test-1",
                                                        supports_vision=vision)),
                 tools={"read": fs.read}, log=Log(tmp_path / "log.jsonl"), root=tmp_path,
                 policy=ALLOW)


def test_a_resume_rebuilds_the_picture_from_the_path_the_record_named(tmp_path, picture):
    """The bytes were never in the log, so a resume re-opens the file — exactly what a resumed text read
    does with a file that changed underneath it. What must not change is the *shape*: a conversation that
    lost its picture on the trip through a resume would show the model a sentence where it once saw an
    image, and it would answer a question nobody asked."""
    agent, _ = build([model.turn_tool("c1", "read", {"path": str(picture)}), model.turn_text("seen")],
                     tmp_path, tools={"read": fs.read}, vision=True)
    agent.run("look at the picture")
    fresh = _replayed(tmp_path, vision=True)
    fresh.replay()
    answer = [m for m in fresh.messages if m.role == "tool"][-1]
    urls = [p["image_url"]["url"] for p in (answer.content if isinstance(answer.content, list) else [])
            if "image_url" in p]
    assert len(urls) == 1 and urls[0].startswith("data:image/png;base64,"), answer.content


def test_a_resume_carries_the_reasoning_signature_of_the_turns_it_rebuilt(tmp_path):
    """The signature is a fact of the log, not of memory: a fresh Agent over the log re-reads which
    field a turn's reasoning came in on, so the resumed conversation re-sends in the same dialect it
    arrived in — a turn that came in on `reasoning` does not go out on `reasoning_content` after a
    restart."""
    agent, _ = build([[model.ReasoningDelta("th", "reasoning"),
                      model.ReasoningDelta("ink", "reasoning"),
                      model.TextDelta("done"),
                      model.Stopped("stop")],
                     [model.TextDelta("second"), model.Stopped("stop")]], tmp_path)
    agent.run("go")
    fresh = _replayed(tmp_path, vision=False)
    fresh.replay()
    reasoned = [m for m in fresh.messages if m.reasoning]
    assert len(reasoned) == 1 and reasoned[0].reasoning == "think" \
        and reasoned[0].reasoning_field == "reasoning", reasoned


def test_a_resume_that_declares_no_vision_sends_no_picture(tmp_path, picture):
    """The declaration is re-typed every run, like the grants are, so a log full of pictures does not
    oblige a blind endpoint to receive one. Without this the resumed conversation would reach the adapter
    with an image part and be refused there — a true refusal, but raised at the first request of a run
    that could have carried on happily with the text."""
    agent, _ = build([model.turn_tool("c1", "read", {"path": str(picture)}), model.turn_text("seen")],
                     tmp_path, tools={"read": fs.read}, vision=True)
    agent.run("look at the picture")
    fresh = _replayed(tmp_path, vision=False)
    fresh.replay()
    answer = [m for m in fresh.messages if m.role == "tool"][-1]
    assert isinstance(answer.content, str), answer.content


# --- the local window model: the fallback behind the server's number ------------------------

def test_a_backend_that_reports_no_usage_still_compacts_from_the_runs_own_estimate(tmp_path):
    """The graceful path: a backend that never reports usage leaves the server-number trigger
    blind, and the run's own estimate of what the next request will carry — the resident parts
    plus the conversation — is what fires the compaction instead. The notice says which source
    decided, because the operator reading the log has to be able to tell."""
    agent, _ = build([model.turn_text("ok"), model.turn_text("summary")], tmp_path)
    # Thirty old turns, each 99 characters: the dropped zone outmasses the backfill by enough that
    # the head check sees a compaction that fits — with the kept zone holding half the history no
    # compaction could ever fit, and the notice says which source the decision ran on.
    for i in range(30):
        agent.messages.append(model.Message("user", "old turn " + "x" * 91))
        agent._trace.append(f"old-{i}")
    agent.run("go")
    assert one(agent, "summary")["outcome"] == "ok"
    notice = next(s for s in agent.log.steps()
                  if s["kind"] == "notice" and s.get("subject") == "compact")
    assert "own estimate" in notice["text"], notice
    assert "system prompt" in notice["text"] and "conversation" in notice["text"], \
        "the notice names the split the decision was made on"


def test_a_server_count_corrects_the_runs_own_estimate(tmp_path):
    """The backend reports what it counted for a request the run measured; the ratio between the
    two becomes the rate the estimate runs at from here on (see compact.Context.calibrate). The
    request was "go" — two characters, zero estimated tokens — so the whole request as the run
    measured it is the resident parts, and the ratio is the server's number over exactly that."""
    turns = [[model.TextDelta("ok"), model.Usage(400, 10), model.Stopped("stop")]]
    agent, _ = build(turns, tmp_path, context=10000)
    agent.run("go")
    assert agent.context.ratio == pytest.approx(400 / agent.context.resident())


def test_a_request_that_carries_an_image_does_not_teach_the_rate(tmp_path, picture):
    """An image part is counted by its wire bytes while the server counts it as image tokens: the
    ratio between the two is not a rate for text, so the usage an image-carrying request reports
    is not learned from, and the estimate keeps the standing rate until a text request reports."""
    def show_picture(path: str) -> Media:
        return Media(text="the picture", path=path, mime="image/png")

    show_picture.access, show_picture.address = "read", "path"
    agent, _ = build([model.turn_tool("c1", "show_picture", {"path": str(picture)}),
                     [model.TextDelta("done"), model.Usage(900, 5), model.Stopped("stop")]],
                     tmp_path, tools={"show_picture": show_picture}, vision=True)
    agent.run("go")
    assert agent.context.ratio is None, "a ratio learned from an image request would be wrong " \
        "for every text request after it"


# --- the head and the pin of the render input -------------------------------------

def test_the_run_stamps_the_fixed_head_and_the_wire_pin(tmp_path):
    """The ground truth records what the model was told and what was sent: `started` holds the
    exact system prose, the tool contract, the head's identity and size, and the budget; every
    `message` holds the raw finish word and the fingerprint of the logical request. And the record
    alone re-derives the fingerprint — that is the pin's whole job."""
    system = "you are a careful test agent"
    agent, _ = build([model.turn_text("done")], tmp_path, system=system)
    agent.run("go")
    started = one(agent, "started")
    assert started["system"] == system, "the head is the exact prose, not a pointer at a file"
    assert set(t["name"] for t in started["tools"]) == set(TOOLS)
    assert len(started["prefix_digest"]) == 16
    assert isinstance(started["resident_tokens"], int) and started["resident_tokens"] > 0
    msg = next(s for s in agent.log.steps()
               if s["kind"] == "message" and s["role"] == "assistant")
    assert msg["finish"] == "stop"
    assert len(msg["req_sha"]) == 16
    assert isinstance(msg["req_chars"], int) and msg["req_chars"] > 0
    # Re-derive the fingerprint from the record alone — the head from `started`, the stream from
    # the steps — and it must match what was stamped. A record that cannot do this has pinned
    # nothing: the pin would be a label, not an address.
    from sah.agent import _request_digest
    specs = [model.ToolSpec(t["name"], t["description"], t["parameters"])
             for t in started["tools"]]
    user = next(s for s in agent.log.steps()
                if s["kind"] == "message" and s["role"] == "user")
    derived = _request_digest(started["system"], [model.Message("user", user["text"])],
                              specs, started["thinking"], started.get("max_tokens"))
    assert derived[0] == msg["req_sha"], "the record cannot re-derive the pin it stamped"


def test_started_stamps_the_working_directory(tmp_path):
    """`started` records where the run stood: the operator's working directory, the one fact
    about the workspace a reader of the log can use. Inherited from the process by default;
    a caller may name it instead. The process is stamped too: when a run dies silent, the PID is
    the witness a reader on the same machine can check against the process table."""
    agent, _ = build([model.turn_text("done")], tmp_path)
    agent.run("go")
    started = one(agent, "started")
    assert started["cwd"] == os.getcwd()
    assert started["pid"] == os.getpid()
    second = tmp_path / "second"     # its own log file: the first run's `started` is already in one
    second.mkdir()
    agent2, _ = build([model.turn_text("done")], second, cwd="/workplace/of/choice")
    agent2.run("go")
    assert one(agent2, "started")["cwd"] == "/workplace/of/choice"


def test_the_head_is_stamped_once_and_not_reserialized_into_every_turn(tmp_path):
    """The conversation half of the request is the stream, recorded step by step. The record must
    not copy it into itself: the fixed head appears once, at run start, and the per-turn pin is a
    digest and a count, never a second copy of the head or the stream."""
    agent, _ = build([model.turn_tool("c1", "echo", {"text": "a"}),
                      model.turn_text("done")], tmp_path)
    agent.run("go")
    steps = agent.log.steps()
    assert sum(1 for s in steps if "system" in s) == 1, "the head is stamped once, at run start"
    assistant = [s for s in steps if s["kind"] == "message" and s["role"] == "assistant"]
    assert len(assistant) == 2, "two turns, so the per-turn pin must appear on each"
    assert all(s["req_sha"] for s in assistant), "every turn carries its own pin"
    assert all("system" not in s and "tools" not in s for s in assistant), \
        "a turn that carries the head has copied the render input into itself"


def test_the_fingerprint_is_a_function_of_the_five_render_inputs():
    """The pin moves when any of the five render inputs moves, and stays put when none do. That is
    the property that makes it a pin rather than a label: it addresses the request, so it must
    change with the request and with nothing else."""
    from sah.agent import _request_digest, _prefix_digest
    system, thinking = "sys", "off"
    specs = [model.ToolSpec("echo", "d", {"type": "object"})]
    msgs = [model.Message("user", "hi")]
    sha, chars = _request_digest(system, msgs, specs, thinking, 512)
    assert sha == _request_digest(system, msgs, specs, thinking, 512)[0], "same inputs, same pin"
    assert sha != _request_digest("sys2", msgs, specs, thinking, 512)[0], "the system moved"
    assert sha != _request_digest(system, [model.Message("user", "bye")], specs, thinking, 512)[0], \
        "the stream moved"
    assert sha != _request_digest(system, msgs,
                                  [model.ToolSpec("echo", "d2", {"type": "object"})],
                                  thinking, 512)[0], "the schema moved"
    assert sha != _request_digest(system, msgs, specs, "high", 512)[0], "the thinking moved"
    assert sha != _request_digest(system, msgs, specs, thinking, 1024)[0], "the budget moved"
    assert len(sha) == 16 and chars > 0
    # The head identity is the head alone: it ignores the conversation and the budget, so it is the
    # address for "did this run ask the same standing question?" — a question the per-turn pin
    # cannot answer because it folds the conversation in.
    prefix = _prefix_digest(system, specs)
    assert prefix == _prefix_digest(system, specs)
    assert prefix != _prefix_digest("sys2", specs), "a reworded prompt is a different head"
    assert prefix != _prefix_digest(system, [model.ToolSpec("echo", "d2", {"type": "object"})]), \
        "an edited schema is a different head"
    assert prefix != sha, "the per-turn pin is not the head identity"


# --- a resume of a compacted run: what stood before the slot ----------------

def test_a_resume_keeps_exactly_what_stood_before_the_slot(tmp_path):
    """What stood in front of a summary's slot is the record's own arithmetic, not the position
    the log's first user message holds: the walk accumulates everything the live list held
    before the rewrite, and the step says what it dropped and what it kept — the remainder is
    the entry the live loop stood in front of the slot. A layout that kept a head (a log
    written before the goal became the summariser's to keep) replays with it; a layout that
    summarised the opening turn replays without it; and a second compaction drops the first
    slot like any message. A resumed run must send the same list the live run sent, whichever
    generation wrote the log — resurrecting a head the live list did not stand is a
    conversation that changed shape across a restart."""
    for with_head in (True, False):
        folder = tmp_path / ("head" if with_head else "no_head")
        log = Log(folder / "log.jsonl")
        log.append("started", "core", run_id="r1", root=str(folder))
        ids = []
        if with_head:
            ids.append(log.append("message", "user", role="user", text="the objective")["id"])
        ids += [log.append("message", "user", role="user", text=f"turn {i}")["id"]
                for i in range(12)]
        n_head = 1 if with_head else 0
        # S1 keeps the six most recent and drops the six before them. In the layout that kept a
        # head the head is not among the dropped — it stood in front of the slot, and that is
        # exactly what the arithmetic (accumulated − dropped − kept) must recover.
        first = log.append("summary", "core", refs=ids[n_head:n_head + 6],
                           kept=6, dropped=6,
                           text="the older turns, in order", outcome="ok", stage=1,
                           slot="summary")
        # S2: the first slot is now an ordinary message the second compaction may drop — with
        # the backfill's dropped half, the way the live loop drops it.
        log.append("summary", "core",
                   refs=[first["id"]] + ids[n_head + 6:n_head + 10],
                   kept=2, dropped=5,
                   text="the newer notes", outcome="ok", stage=1, slot="summary")
        fresh = Agent(backend=model.Fake([], caps=model.Caps(context_tokens=100, model="test-1")),
                      tools={}, log=log, root=folder, policy=ALLOW)
        fresh.replay()
        assert len(fresh.messages) == (4 if with_head else 3), fresh.messages
        frame_at = 1 if with_head else 0
        if with_head:
            assert fresh.messages[0].content == "the objective", \
                "the head the record's arithmetic says stood is not the first message"
        frame = fresh.messages[frame_at]
        assert "<summary>" in frame.content and "the newer notes" in frame.content
        assert [m.content for m in fresh.messages[frame_at + 1:]] == ["turn 10", "turn 11"], \
            "the backfill is not the kept tail verbatim"
        assert len(fresh._trace) == len(fresh.messages), \
            "the trace and the list disagree on how many entries there are"
