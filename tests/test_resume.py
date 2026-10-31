"""Compaction and resume driven through the loop, against a scripted backend.

These are the integration points the pure-module tests cannot reach: a summary landing mid-run as
its own step, the message list and its trace shrinking together, and a fresh Agent rebuilding a
conversation from a log so a run can continue. The backend is Fake throughout; nothing here needs a
server, and that is the point -- the whole compaction/resume contract must be checkable in-process.
"""

from sah import model
from sah.agent import Agent
from sah.model import Caps, TextDelta, ToolCall, Usage
from sah.record import Log


def _tools():
    def read(path: str) -> str:
        """Read a file."""
        return "contents of " + path

    # A double that stands in for a file tool declares what the real one declares, so the loop has a
    # place to judge and the resume tests exercise the same shape production runs.
    read.access, read.address = "read", "path"
    read.description = "Read a file."
    return {"read": read}


def build(tmp_path, scripted, *, context=1000):
    log = Log(tmp_path / "log.jsonl")
    backend = model.Fake(scripted, caps=Caps(context_tokens=context, model="t"))
    agent = Agent(backend=backend, tools=_tools(), log=log, max_turns=30, run_id="r",
                  root=tmp_path, queue=tmp_path / "control.md")
    return agent, backend, log


def kinds(log):
    return [s["kind"] for s in log.steps()]


# --- compaction through the loop ---------------------------------------------

def grow(agent, n):
    """Push n user/assistant turns into the conversation and its trace the way the loop would, so
    compact() has real messages and real ids to work with. Driving compact() directly, rather than
    scripting a run that happens to fill up, keeps the summary-call and turn-call streams from
    fighting over one scripted backend."""
    for i in range(n):
        agent.say("user", f"question {i}", issuer="user")
        agent.say("assistant", f"answer {i}", issuer="model")
    return agent


def test_a_summary_step_names_the_messages_it_replaced(tmp_path):
    """refs must be real log-step ids that existed before the summary. Without them the step is
    decoration -- you cannot see what was dropped. And the count must agree with the refs. No
    message is carved out: the same 90 messages lose 84, the opening one included."""
    agent, backend, log = build(tmp_path, [[TextDelta("notes"), Usage(60, 20), model.Stopped("stop")]])
    grow(agent, 45)
    before = {s["id"] for s in log.steps()}
    agent.compact()
    summary = log.of_kind("summary")[0]
    assert summary["outcome"] == "ok"
    assert summary["refs"] and all(r in before for r in summary["refs"])
    assert summary["dropped"] == len(summary["refs"]) == 84     # 90 msgs - 6 kept
    assert summary["kept"] == 6
    # the opening message is among the replaced -- no message is privileged -- and the record's
    # budget line shows the tool component
    assert log.of_kind("message")[0]["id"] in summary["refs"], \
        "the opening message is summarisable content, not a carved-out head"
    assert "tool schemas" in summary["budget"], \
        f"the budget line does not show the tool surface: {summary['budget']!r}"
    # the summary step reports what compaction cost, so "how much context did we burn summarising"
    # is answerable from the log; a zero always recorded is a field that stopped meaning anything
    assert summary["tokens_cost"] > 0


def test_the_level_the_operator_named_for_summaries_reaches_the_summary_request(tmp_path):
    """`--summary-think` exists because a thinking server can spend the summary's room on thinking the
    record will not keep, and the escape is the knob: the level the operator names for summaries,
    applied to every summary attempt. The default keeps the run's own level untouched (A8): left
    alone, not overridden. So this names a level and reads it off the summary request the backend
    actually received -- and reads the run's own level off the request of a run that named none.
    Asserting `agent.summary_thinking == "low"` would read an attribute the Agent stores for itself,
    and a build that stored it and never sent it would keep that test green."""
    scripted = [[TextDelta("notes"), Usage(60, 20), model.Stopped("stop")]]
    log = Log(tmp_path / "log.jsonl")
    backend = model.Fake([list(t) for t in scripted], caps=Caps(context_tokens=1000, model="t"))
    agent = Agent(backend=backend, tools=_tools(), log=log, max_turns=30, run_id="r",
                  root=tmp_path, queue=tmp_path / "control.md",
                  thinking="high", summary_thinking="low")
    grow(agent, 45)
    agent.compact()
    assert backend.calls[-1].thinking == "low", \
        f"the summary asked for {backend.calls[-1].thinking!r}: the level never reached the request"

    log2 = Log(tmp_path / "log2.jsonl")
    backend2 = model.Fake([list(scripted[0])], caps=Caps(context_tokens=1000, model="t"))
    agent2 = Agent(backend=backend2, tools=_tools(), log=log2, max_turns=30, run_id="r",
                   root=tmp_path, queue=tmp_path / "control.md", thinking="high")
    grow(agent2, 45)
    agent2.compact()
    assert backend2.calls[-1].thinking == "high", \
        "the run's own level was not left alone: the default overrode it"


def test_compaction_shrinks_the_message_list_and_trace_together(tmp_path):
    """The two must never disagree on length -- a message with no trace id, or a trace id with no
    message, breaks the next compaction's refs. The slot takes the conversation's head:
    90 messages become slot + 6 kept + the act's word."""
    agent, backend, log = build(tmp_path, [[TextDelta("notes"), Usage(60, 20), model.Stopped("stop")]])
    grow(agent, 45)
    assert len(agent.messages) == len(agent._trace) == 90
    agent.compact()
    assert len(agent.messages) == len(agent._trace) == 8      # 1 slot + 6 kept + the act's word


def test_a_second_compaction_remembers_the_first_summary(tmp_path):
    """The trace slot the summary occupies has to hold the summary's own step id, not an empty string.
    Two things ride on it. A second compaction's `refs` is built by filtering out falsy trace entries,
    so the first summary silently vanished from the record of what was dropped -- the log then described
    a compaction that forgot something it never named. And the second compaction's source must carry the
    first summary's text (each summary re-summarises the last slot), or the first compaction's detail is
    lost the moment the second drops the slot. The live path and the replay must agree, and a resumed run
    and a live run of the same log must carry the same trace.
    """
    agent, backend, log = build(tmp_path, [[TextDelta("notes"), Usage(60, 20), model.Stopped("stop")],
                                           [TextDelta("more"), Usage(60, 20), model.Stopped("stop")]])
    grow(agent, 45)
    agent.compact()
    first = log.of_kind("summary")[0]
    assert agent._trace[0] == first["id"], \
        f"the summary's trace slot holds {agent._trace[0]!r}; a later compaction cannot name what it drops"

    grow(agent, 45)
    agent.compact()
    second = log.of_kind("summary")[1]
    assert first["id"] in second["refs"], \
        "the second compaction dropped the first summary and did not say so"
    second_source = backend.calls[-1].messages[0].content
    assert "notes" in second_source, \
        "the second compaction's source does not carry the first summary: the merge is missing"


def test_the_summary_is_first_and_the_opening_turn_is_summarised(tmp_path):
    """No message is privileged: whatever compaction did, the slot is first in the context -- the
    run's opening turn was summarised with the rest, and keeping the goal standing is the
    summariser's job over the whole history. A resumed run (whose replay stands the slot first,
    with no head beside it) sees the same order a live run does."""
    agent, backend, log = build(tmp_path, [[TextDelta("the notes"), Usage(60, 20),
                                             model.Stopped("stop")]])
    grow(agent, 45)
    agent.compact()
    assert agent.messages[0].role == "user"
    assert "the notes" in agent.messages[0].content, \
        f"the slot is not first: {agent.messages[0].content!r}"
    assert "question 0" not in "".join(m.content for m in agent.messages
                                        if isinstance(m.content, str)), \
        "the opening turn survived the rewrite: no message is privileged"


def test_the_act_notice_replays_beside_the_conversation_it_told(tmp_path):
    """The act's word is a core message in the live conversation and a core message step in the log
    after the summary step, so replay must rebuild it in the same place: a resumed run that drops
    it never gets told its context was compacted, and the live and resumed runs diverge from the
    moment after the compaction, in the one message whose whole job is to say they should not."""
    agent, backend, log = build(tmp_path, [[TextDelta("notes"), Usage(60, 20), model.Stopped("stop")]])
    grow(agent, 45)
    agent.compact()
    act = [s for s in log.steps() if s["kind"] == "notice"
           and s.get("subject") == "compact" and "compaction occurred" in s.get("text", "")]
    assert len(act) == 1, f"the act notice is missing or doubled: {[s['text'] for s in act]}"
    fresh = _reload(log, tmp_path)
    fresh.replay()
    assert fresh.messages[-1].content == act[0]["text"], \
        "replay dropped the act's word: a resumed run never gets told its context was compacted"
    assert _shape(fresh) == _shape(agent), f"\n live:   {_shape(agent)}\n replay: {_shape(fresh)}"
    assert fresh._trace == agent._trace


def test_a_compaction_whose_attempts_all_fail_still_fills_the_slot(tmp_path):
    """No path leaves the conversation untouched and doomed: the slot is always filled, so a
    compaction whose attempts all failed (the backend cannot summarise) still rewrites the
    conversation -- the failed note, the backfill -- and the record says what the slot
    holds (`slot: failed`) and what the attempts cost, in their words. The backfill note picks its
    uncovered words: a tail beside a failed note is not covered by anything above it."""
    empty = [Usage(50, 0), model.Stopped("stop")]
    agent, backend, log = build(tmp_path, [list(empty) for _ in range(2)])
    grow(agent, 45)
    n_before = len(agent.messages)
    agent.compact()
    failed = [s for s in log.of_kind("summary") if s["outcome"] != "ok"]
    assert failed and failed[0]["outcome"] == "failed" and failed[0]["slot"] == "failed"
    assert "no summary text" in failed[0]["reason"], \
        f"the record does not say what the attempts cost: {failed[0]['reason']!r}"
    assert len(agent.messages) == 8 == len(agent._trace) < n_before, \
        "the conversation was left untouched: a failed slot beside an un-shrunk context is the doom"
    frame = agent.messages[0]
    assert "84 earlier turns" in frame.content
    assert "covered above" not in frame.content, \
        "the backfill note claims coverage the failed note does not give"
    assert "compaction occurred" in agent.messages[-1].content, \
        "the act's word is owed even when the slot holds the failed note"


# --- resume -------------------------------------------------------------------

def test_replay_rebuilds_the_conversation(tmp_path):
    """A fresh Agent over the same log reconstructs the turns -- resume means continue, not restart."""
    agent, backend, log = build(tmp_path, [[TextDelta("hello"), Usage(10, 3), model.Stopped("stop")]])
    agent.run("hi there")
    fresh = Agent(backend=model.Fake([]), tools=_tools(), log=log, root=tmp_path)
    n = fresh.replay()
    assert n >= 2                                    # the user turn and the assistant reply
    assert any(m.role == "assistant" for m in fresh.messages)
    assert any(m.role == "user" and m.content == "hi there" for m in fresh.messages)


def test_replay_links_a_tool_result_to_its_call(tmp_path):
    """The bug resume would introduce: a tool result whose id matches no assistant call, which a
    real server rejects. After replay the tool_call_id equals the call's recorded backend id."""
    scripted = [[TextDelta("let me look"), ToolCall("call-abc", "read", {"path": "x"}),
                 Usage(10, 3), model.Stopped("tool_calls")],
                [TextDelta("done"), Usage(10, 3), model.Stopped("stop")]]
    agent, backend, log = build(tmp_path, scripted)
    agent.run("read x")
    fresh = Agent(backend=model.Fake([]), tools=_tools(), log=log, root=tmp_path)
    fresh.replay()
    call_msg = next(m for m in fresh.messages if m.tool_calls)
    result_msg = next(m for m in fresh.messages if m.role == "tool")
    # the backend's own id ("call-abc"), not a fabricated log-step id: a server matches on this and
    # rejects an orphan, so resume must carry the real one across the reload
    assert call_msg.tool_calls[0][0] == "call-abc"
    assert result_msg.tool_call_id == "call-abc"


def test_replay_skips_audit_that_the_model_never_saw(tmp_path):
    """A command/operation/decision is not conversation. Replaying it would hand the model authority
    it never held -- resume rebuilds turns, never grants or verdicts."""
    agent, backend, log = build(tmp_path, [[TextDelta("hi"), Usage(5, 2), model.Stopped("stop")]])
    agent.run("go")
    log.append("decision", "core", refs="x", outcome="refused", reason="not granted")
    fresh = Agent(backend=model.Fake([]), tools=_tools(), log=log, root=tmp_path)
    fresh.replay()
    assert all(m.role in ("user", "assistant", "tool", "system") for m in fresh.messages)


def test_resume_after_a_summary_starts_from_the_notes(tmp_path):
    """The point of a summary is that the next run does not re-read the turns it replaced: replay
    puts the notes where the old context was."""
    log = Log(tmp_path / "log.jsonl")
    log.append("message", "user", role="user", text="old task")
    log.append("message", "model", role="assistant", text="old answer")
    log.append("summary", "core", refs=["000000-message", "000001-message"], kept=0, dropped=2,
               text="the notes", outcome="ok", tokens_cost=5)
    fresh = Agent(backend=model.Fake([]), tools=_tools(), log=log, root=tmp_path)
    fresh.replay()
    assert len(fresh.messages) == 1 and "the notes" in fresh.messages[0].content
    assert "old answer" not in "".join(getattr(m, "content", "") if isinstance(m.content, str)
                                       else "" for m in fresh.messages)


def test_resume_rebuilds_conversation_but_never_permission(tmp_path):
    """The security-relevant half of resume: a run that was granted a folder, logged a write there,
    and is resumed WITHOUT that grant must not be able to write it again. Replaying the turns says
    nothing about permissions -- the log holds no authority, exactly as the notes require.

    (The old tree sniffed a substring in the transcript head to re-grant on resume; this test is the
    wall against ever doing that again.)"""
    from sah.confine import Confine

    log = Log(tmp_path / "log.jsonl")
    # a past run that wrote a file, recorded in its log
    log.append("message", "user", role="user", text="write it")
    log.append("message", "model", role="assistant", text="done")
    log.append("operation", "core", op="write", address="secret.txt", outcome="allowed")

    # resumed with NO grants at all
    fresh = Agent(backend=model.Fake([]), tools=_tools(), log=log, root=tmp_path,
                  policy=Confine(read_roots=[], write_roots=[]))
    fresh.replay()
    assert any(m.role == "assistant" for m in fresh.messages)     # conversation came back
    # but the past write grants nothing now: the same write is refused
    assert not fresh.policy("write", {"path": "secret.txt", "content": "x"},
                            access="write", address="path").allowed


def test_replay_records_the_line_it_could_not_read(tmp_path):
    """A run killed mid-append leaves a torn last line, and a resume continues over that hole. The
    turns before the damage still come back — but the log gains a notice naming the lost line,
    because a later reader deserves to know the history they are reading has a gap, and stderr is
    not something a later reader is handed."""
    agent, backend, log = build(tmp_path, [[TextDelta("hello"), Usage(10, 3), model.Stopped("stop")]])
    agent.run("hi there")
    with (tmp_path / "log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"message","id":"000099-mes')       # a later session died mid-write
    fresh = Agent(backend=model.Fake([]), tools=_tools(), log=Log(tmp_path / "log.jsonl"),
                  root=tmp_path)
    n = fresh.replay()
    assert n >= 2, "the turns before the damage still come back"
    notices = [s for s in fresh.log.read()[0]
               if s["kind"] == "notice" and s["subject"] == "replay"]
    assert notices and "unreadable" in notices[0]["text"]


# --- replay fidelity: the resumed list must BE the live list -------------------

def _shape(agent):
    """The conversation as one comparable value: what a server would actually be sent. Compared as a
    whole, because the contract is that resume hands back the same conversation — not a near-miss
    that a lenient server tolerates and a prefix cache silently punishes."""
    return [(m.role, m.content, m.tool_calls, m.tool_call_id) for m in agent.messages]


def _reload(log, tmp_path):
    return Agent(backend=model.Fake([]), tools=_tools(), log=log, root=tmp_path)


def test_replay_rebuilds_the_conversation_the_live_loop_had(tmp_path):
    """Two parallel calls in a turn with no prose is the shape replay gets wrong: the log holds one
    `message` step (empty) plus two `call` steps, while the conversation holds ONE assistant message
    carrying both calls. Replaying each step to one message splits the turn in three and separates the
    prose from its calls, so the resumed request is not the request the live loop would have sent."""
    scripted = [[ToolCall("c1", "read", {"path": "a"}), ToolCall("c2", "read", {"path": "b"}),
                 Usage(10, 4), model.Stopped("tool_calls")],
                [TextDelta("done"), Usage(10, 2), model.Stopped("stop")]]
    agent, backend, log = build(tmp_path, scripted)
    agent.run("go")
    back = _reload(log, tmp_path)
    back.replay()
    assert _shape(back) == _shape(agent), f"\n live:   {_shape(agent)}\n replay: {_shape(back)}"
    assert back._trace == agent._trace


def test_a_turn_that_produced_nothing_enters_neither_list(tmp_path):
    """An assistant turn with no text and no call is recorded in the log and deliberately left out of
    the conversation: to a chat template a blank assistant message is a turn the next request reads as
    an empty prompt, and a model told to always call a tool then invents one. So replay must drop it as
    well — and the trace must not gain an id for a message that was never added, because compaction
    pairs the two lists by index and a shift sends a summary after the wrong turns."""
    scripted = [[Usage(9, 0), model.Stopped("stop")],
                [TextDelta("now then"), Usage(9, 2), model.Stopped("stop")]]
    agent, backend, log = build(tmp_path, scripted)
    agent.run("go")
    assert len(agent.messages) == len(agent._trace)
    back = _reload(log, tmp_path)
    back.replay()
    assert not [m for m in back.messages if m.role == "assistant" and not m.content
                and not m.tool_calls]
    assert _shape(back) == _shape(agent), f"\n live:   {_shape(agent)}\n replay: {_shape(back)}"
    assert back._trace == agent._trace


def test_a_failed_turn_replays_as_the_live_loop_sent_it(tmp_path):
    """A failed turn that wrote prose is in the conversation (the loop sent it, and a server accepted
    it); a failed turn that wrote nothing is not. Replay follows whatever the loop did rather than
    running a rule of its own, so the two are compared directly across a failure."""
    scripted = [[TextDelta("half a sentence"), model.Failed("boom"), Usage(8, 1)],
                [TextDelta("recovered"), Usage(8, 2), model.Stopped("stop")]]
    agent, backend, log = build(tmp_path, scripted)
    agent.run("go")                                   # the failure ends the run
    assert agent.messages[-1].content == "half a sentence"
    agent.run("again")
    back = _reload(log, tmp_path)
    back.replay()
    assert _shape(back) == _shape(agent), f"\n live:   {_shape(agent)}\n replay: {_shape(back)}"
    assert back._trace == agent._trace


def test_a_call_whose_header_was_lost_still_has_something_to_answer(tmp_path):
    """A run killed mid-append can lose the `message` line and keep the `call` line that referred to it.
    Dropping such a call would leave its result answering nothing, which a server rejects outright; so
    the turn is rebuilt from its calls, with no text, because that is all the log can attest."""
    scripted = [[ToolCall("c1", "read", {"path": "a"}), Usage(10, 3), model.Stopped("tool_calls")],
                [TextDelta("done"), Usage(10, 2), model.Stopped("stop")]]
    agent, backend, log = build(tmp_path, scripted)
    agent.run("go")
    import json
    header = [s["id"] for s in log.steps() if s["kind"] == "message" and s["role"] == "assistant"][0]
    # the log as bytes, minus the one line that died: only the core appends authoritative steps, so a
    # test that needs a damaged history writes the damaged history
    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if json.loads(line)["id"] != header]
    damaged = tmp_path / "damaged"
    damaged.mkdir()
    with (damaged / "log.jsonl").open("w", encoding="utf-8", newline="") as handle:
        for line in kept:
            handle.write(line)
            handle.write(chr(10))          # one JSON object per line, LF as the log always writes
    back = Agent(backend=model.Fake([]), tools=_tools(), log=Log(damaged / "log.jsonl"),
                 root=damaged)
    back.replay()
    roles = [m.role for m in back.messages]
    turn = next(i for i, m in enumerate(back.messages) if m.tool_calls)
    tool = next(i for i, m in enumerate(back.messages) if m.role == "tool")
    assert back.messages[turn].tool_calls[0][0] == "c1", "the call survives its lost header"
    assert back.messages[tool].tool_call_id == "c1", "and its result still answers it"
    # order is the whole point, not just presence: a tool result that arrives before the assistant turn
    # calling it is the orphan shape a server rejects, so the rebuilt turn goes first
    assert turn < tool, roles


def test_a_summary_drops_the_tool_turns_it_replaced(tmp_path):
    """Compaction replaces the turns before it, so a resume after a summary must not resurrect the
    calls of those turns: the notes stand in for the whole turn, calls and results included."""
    scripted = [[ToolCall("c1", "read", {"path": "a"}), Usage(10, 3), model.Stopped("tool_calls")],
                [TextDelta("done"), Usage(10, 2), model.Stopped("stop")]]
    agent, backend, log = build(tmp_path, scripted)
    agent.run("go")
    refs = [s["id"] for s in log.steps() if s["kind"] in ("message", "call", "result")]
    log.append("summary", "core", refs=refs, kept=0, dropped=len(refs), text="the notes",
               outcome="ok", tokens_cost=4)
    back = _reload(log, tmp_path)
    back.replay()
    assert [m.role for m in back.messages] == ["user"]
    assert "the notes" in back.messages[0].content
    assert back._trace == [log.of_kind("summary")[0]["id"]]


def test_a_summary_also_drops_a_turn_whose_header_was_lost(tmp_path):
    """Two damage paths meeting: a `call` whose `message` line the crash took, and a later session that
    summarised the run. The notes stand in for everything before them, so an un-headered call must not be
    rebuilt after the summary, answering nothing, in a conversation that no longer contains it."""
    log = Log(tmp_path / "log.jsonl")
    log.append("message", "user", role="user", text="go")
    lost = "000002-message"                       # the header line the crash took
    log.append("call", "model", refs=lost, name="read", call_id="c1", args={"path": "a"})
    log.append("summary", "core", refs=[lost], kept=0, dropped=1, text="the notes",
               outcome="ok", tokens_cost=3)
    back = _reload(log, tmp_path)
    back.replay()
    assert [m.role for m in back.messages] == ["user"]
    assert not [m for m in back.messages if m.tool_calls]


def test_replay_carries_reasoning_back_into_the_conversation(tmp_path):
    """Replay mirrors `step()`, and that rule is what makes a resumed run behave like a continued one. A
    resumed conversation that dropped each turn's reasoning would send the model a different history than
    the one it answered, and the hazard reasoning exists to avoid returns exactly when a run is long
    enough to have been resumed."""
    from sah.model import ReasoningDelta, Stopped
    scripted = [[ReasoningDelta("check the grant first"), TextDelta("looking"), Stopped("stop")]]
    agent, backend, log = build(tmp_path, scripted)
    agent.run("go")
    fresh = Agent(backend=model.Fake([]), tools=_tools(), log=log, root=tmp_path)
    fresh.replay()
    turns = [m for m in fresh.messages if m.role == "assistant"]
    assert turns and turns[0].reasoning == "check the grant first", turns


def test_the_count_replay_returns_is_the_number_of_messages_it_rebuilt(tmp_path):
    """`replay`'s docstring said "how many turns came back" and the resume line printed the same words,
    while the value it returns is `len(self.messages)`. Measured on a log whose rebuilt conversation held
    one message and no assistant turn at all, the operator was told `1 turns back` — a count of one thing
    named as another, in the one sentence whose entire job is to say how much history came back. The
    wrongness even broke its own grammar, which is how it was first noticed.

    The fixture is required to distinguish the two counts, or the test could pass while the name stayed
    wrong: an assertion about naming is worthless when every input makes both numbers equal."""
    log = Log(tmp_path / "log.jsonl")
    log.append("message", "user", role="user", text="do the thing")
    log.append("message", "model", role="assistant", text="working on it")
    fresh = Agent(backend=model.Fake([]), tools=_tools(), log=log, root=tmp_path)
    n = fresh.replay()
    assert n == len(fresh.messages), (n, len(fresh.messages))
    assistant = sum(1 for m in fresh.messages if m.role == "assistant")
    assert assistant != n, "this fixture must make messages and turns differ, or it cannot test the name"


def test_a_resumed_run_carries_the_same_summary_frame_as_the_live_one(tmp_path):
    """Two builders, one message: the live loop and replay both stand a summary in for the turns it replaced.

    A divergence here is invisible until a template or a cache key notices it — the resumed run would read its
    own history under different framing than the run that wrote it, and the prompt file exists precisely so
    that framing can change. So the assertion is equality of the bytes, not of a substring."""
    agent, backend, log = build(tmp_path, [[TextDelta("the notes"), Usage(60, 20),
                                            model.Stopped("stop")]])
    grow(agent, 45)
    agent.compact()
    live = agent.messages[0].content       # the slot takes the conversation's head

    fresh = Agent(backend=model.Fake([]), tools=_tools(), log=log, root=tmp_path)
    fresh.replay()
    resumed = [m for m in fresh.messages if "the notes" in m.content]
    assert len(resumed) == 1, f"replay did not stand one summary in: {resumed}"
    assert resumed[0].content == live, \
        f"replay framed the summary differently:\n live: {live!r}\n resume: {resumed[0].content!r}"


# --- a cut resume: the tail stays in the file, out of the conversation ---------------------------


def test_a_cut_resume_continues_from_the_prefix_and_keeps_the_tail_in_the_file(tmp_path):
    """A cut is recorded, not performed: the file keeps the detached tail, the rebuilt
    conversation does not, and the model is never told either thing. A resumed run steers again
    from the step it was cut to, and the old exchange stands in the record beside the new line.
    """
    agent, backend, log = build(tmp_path,
                                [[TextDelta("steered"), Usage(10, 5), model.Stopped("stop")]])
    agent.say("user", "first question", issuer="user")
    agent.say("assistant", "first answer", issuer="model")
    agent.say("user", "old steer", issuer="user")
    agent.say("assistant", "answer to old steer", issuer="model")
    target = log.of_kind("message")[1]["id"]          # after the first answer
    log.append("cut", "core", to=target)
    restored = agent.replay()
    assert restored == 2
    assert [m.content for m in agent.messages] == ["first question", "first answer"], \
        "the tail the cut detached came back into the conversation"
    steps = log.steps()
    assert steps[-1]["kind"] == "cut" and steps[-1]["to"] == target, \
        "the cut stands in the file and names the step the conversation continues from"
    assert len(log.of_kind("message")) == 4, "the tail still stands in the record"
